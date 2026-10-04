//! 原搜索版的策略组件：信念统计（`Know`）+ 快速启发式策略。
//!
//! `Know` 是"某个座位在当前局面下所知道的一切"的紧凑快照，只读取
//! `Session` 中该座位有权知道的信息（`CardState::is_known_to`），因此
//! 模拟里的每个座位都会得到自己的、不完整的信息。
//!
//! 同一套 `Know` 既能从 `Session` 直接构建（rollout 内使用，避免投影开销），
//! 也能从 `PlayerView` 构建（根节点候选生成用）。

use rand::RngCore;

use crate::game::view::{Panel, PlayerView};
use crate::game::{
    CardId, Command, Pending, Phase, Pile, PlayerId, PowerUse, Session, SlotId, RANK_COPIES,
};

/// 一副牌的总张数与点数池。
const _TOTAL: usize = 52;

// ---------------------------------------------------------------- 信念统计

/// 位置化信念模型的温度参数（离线标定工具见历史备份分支的 `examples/fit.rs`）。
///
/// 看不见的牌并不是同一个分布：实测（对照真值）四类位置的平均点数差别很大——
/// 对手暗牌平均 4.3、我的暗牌 6.5、摸牌堆 6.5、弃牌堆底 8.4。
/// 直觉：玩家会主动把手里的大牌换出去（所以对手暗牌偏低），
/// 而换出去的牌都躺在弃牌堆里（所以弃牌堆偏高）。
/// 用竞争性权重 w(r)=exp(θ·r) 拟合各类位置。
///
/// **对手暗牌的 θ 必须随回合推进而变化**：分阶段标定显示，
/// 每人平均行动过的回合数 t 越多，对手暗牌被"换成低牌"得越彻底：
///   t=0.6 → θ=-0.07，t=6 → θ=-0.24，t=12 → θ=-0.48，t=19 → θ=-0.66
/// （2/3/4 人局的标定点都落在同一条曲线上）。固定 θ 会高估后期对手的手牌
/// 约 2-3 点，而那正是决定要不要宣告 Cabo 的时刻，实测代价很大。
pub const THETA_DISCARD: f64 = 0.08;

/// 按"每人平均已行动回合数"推算对手暗牌的倾斜温度。
///
/// `known_frac` = 弃牌堆里已被公共历史恢复的比例（0~1）。
/// 倾斜的本质是"大牌会迁移到弃牌堆"；这部分已经被"记牌"直接从池子里扣掉了，
/// 所以只剩 `1 - known_frac` 还没被解释，倾斜要按这个比例缩放，
/// 否则会把同一个效应扣两次，导致对手手牌被系统性低估（实测会多宣告、多挨罚）。
///
/// 环境变量 `CABO_THETA_OPP` 可以把它钉成固定值、`CABO_THETA_SCALE` 可以整体缩放
/// （回归实验 / 调参用）。
pub fn theta_opp_for_scaled(n: usize, deck_count: usize, known_frac: f64) -> f64 {
    use std::sync::OnceLock;
    static FIXED: OnceLock<Option<f64>> = OnceLock::new();
    static SCALE: OnceLock<f64> = OnceLock::new();
    let fixed = *FIXED.get_or_init(|| {
        std::env::var("CABO_THETA_OPP").ok().and_then(|v| v.trim().parse::<f64>().ok())
    });
    if let Some(v) = fixed {
        return v;
    }
    let scale = *SCALE.get_or_init(|| {
        std::env::var("CABO_THETA_SCALE").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(1.0)
    });
    let initial_deck = 51.0 - 4.0 * n as f64;
    let draws = (initial_deck - deck_count as f64).max(0.0);
    let turns = draws / n.max(1) as f64;
    let residual = (1.0 - known_frac.clamp(0.0, 1.0)).max(0.0);
    (-(0.04 + 0.033 * turns) * scale * residual).clamp(-0.70, 0.0)
}

/// 不启用"记牌"时的倾斜温度。
pub fn theta_opp_for(n: usize, deck_count: usize) -> f64 {
    theta_opp_for_scaled(n, deck_count, 0.0)
}

/// 某座位视角下的已知信息快照。
#[derive(Clone, Debug)]
pub struct Know {
    pub n: usize,
    pub me: PlayerId,
    /// 每个座位的手牌（下标 = 座位；`None` = 该座位不知道点数）。
    pub hands: Vec<Vec<Option<u8>>>,
    /// 我抽到、尚未处理的那张牌（只有我知道）。
    pub limbo: Option<u8>,
    pub discard_top: Option<u8>,
    pub discard_count: usize,
    pub deck_count: usize,
    pub totals: Vec<u32>,
    pub penalty: u32,
    pub target: u32,
    /// 我看不到的牌的点数分布（0~13 各剩几张）。
    pub pool: [u16; 14],
    /// 从公共历史（日志+堆顶观测）里恢复出来的弃牌堆构成（**不含堆顶**）。
    /// 这部分牌本来躺在 `pool` 里当"未知"，实际是公开信息，扣掉它信念会锐利很多。
    pub known_discard: [u16; 14],
    /// 本轮是否已经有人宣告 Cabo。
    pub cabo_called: bool,
    /// 位置化信念：我自己暗牌的期望点数 / 方差。
    pub mean_me: f64,
    pub var_me: f64,
    /// 对手暗牌的期望点数 / 方差。
    pub mean_opp: f64,
    pub var_opp: f64,
    /// 摸牌堆的期望点数（我下一张会摸到什么）。
    pub mean_deck: f64,
    /// 我暗牌的张数（用于无放回方差修正）。
    pub my_unknown_slots: usize,
    pub opp_unknown_slots: usize,
}

impl Know {
    fn blank(n: usize, me: PlayerId) -> Know {
        Know {
            n,
            me,
            hands: vec![Vec::new(); n],
            limbo: None,
            discard_top: None,
            discard_count: 0,
            deck_count: 0,
            totals: vec![0; n],
            penalty: 10,
            target: 100,
            pool: RANK_COPIES.map(|c| c as u16),
            known_discard: [0; 14],
            cabo_called: false,
            mean_me: 6.5,
            var_me: 12.0,
            mean_opp: 6.5,
            var_opp: 12.0,
            mean_deck: 6.5,
            my_unknown_slots: 0,
            opp_unknown_slots: 0,
        }
    }

    /// 从完整局面里构建座位 `me` 的视角。
    pub fn from_session(s: &Session, me: PlayerId) -> Know {
        let n = s.players.len();
        let mut k = Know::blank(n, me);
        for i in 0..n {
            let mut hand = Vec::with_capacity(s.players[i].slots.len());
            for &cid in &s.players[i].slots {
                let cs = &s.cards[cid as usize];
                if cs.is_known_to(me) {
                    hand.push(Some(cs.card.rank));
                } else {
                    hand.push(None);
                }
            }
            k.hands[i] = hand;
            k.totals[i] = s.players[i].total_score;
        }
        match &s.phase {
            Phase::Turn { pending: Some(Pending::Drew { card }), .. }
            | Phase::Turn { pending: Some(Pending::PowerAiming { card, .. }), .. } => {
                let cs = &s.cards[*card as usize];
                if cs.is_known_to(me) {
                    k.limbo = Some(cs.card.rank);
                }
            }
            Phase::Turn {
                pending: Some(Pending::SwapSelecting { card, source: Pile::Draw, .. }),
                ..
            } => {
                let cs = &s.cards[*card as usize];
                if cs.is_known_to(me) {
                    k.limbo = Some(cs.card.rank);
                }
            }
            _ => {}
        }
        if let Some(&top) = s.discard.last() {
            let cs = &s.cards[top as usize];
            if cs.is_known_to(me) {
                k.discard_top = Some(cs.card.rank);
            }
        }
        k.discard_count = s.discard.len();
        k.deck_count = s.deck.len();
        k.penalty = s.settings.cabo_penalty;
        k.target = s.settings.target_score;
        k.cabo_called = s.cabo_caller.is_some();
        k.recount_pool();
        k
    }

    /// 从玩家视图构建（根节点用，不依赖重建出的具体点数）。
    /// `known_discard` 是 [`crate::ai::history`] 恢复出来的弃牌堆构成。
    pub fn from_view(v: &PlayerView) -> Know {
        Self::from_view_with(v, &[0; 14])
    }

    pub fn from_view_with(v: &PlayerView, known_discard: &[u16; 14]) -> Know {
        let n = v.all_seats.len();
        let me = v.me.unwrap_or(0);
        let mut k = Know::blank(n, me);
        for (i, seat) in v.all_seats.iter().enumerate() {
            k.hands[i] = seat.slots.iter().map(|c| c.value).collect();
            k.totals[i] = seat.total_score;
        }
        k.limbo = v.drawn;
        k.discard_top = v.discard_top;
        k.discard_count = v.discard_count;
        k.deck_count = v.deck_count;
        k.penalty = v.cabo_penalty;
        k.target = v.target_score;
        k.cabo_called = v.cabo_caller.is_some();
        k.known_discard = *known_discard;
        k.recount_pool();
        k
    }

    /// 统计"我看不到的牌"还剩哪些点数，再按位置类型算出各自的期望点数。
    fn recount_pool(&mut self) {
        let mut pool = RANK_COPIES.map(|c| c as i32);
        let dec = |pool: &mut [i32; 14], r: u8| {
            if (r as usize) < pool.len() && pool[r as usize] > 0 {
                pool[r as usize] -= 1;
            }
        };
        for hand in &self.hands {
            for c in hand.iter().flatten() {
                dec(&mut pool, *c);
            }
        }
        if let Some(r) = self.limbo {
            dec(&mut pool, r);
        }
        if let Some(r) = self.discard_top {
            dec(&mut pool, r);
        }
        // 已知躺在弃牌堆里的牌：从"未知池"里划掉（它们本就不在任何手牌/摸牌堆里）
        for r in 0..14 {
            pool[r] = (pool[r] - self.known_discard[r] as i32).max(0);
        }
        for (i, &c) in pool.iter().enumerate() {
            self.pool[i] = c.max(0) as u16;
        }
        self.recount_positions();
    }

    /// 位置化信念：把看不见的牌按"竞争性权重"分配到四类位置
    /// （我暗牌 / 对手暗牌 / 摸牌堆 / 弃牌堆底），各类型有自己的 θ。
    /// 位置数（各类张数）是已知的，池构成也是已知的，因此各类均值可以算出来。
    fn recount_positions(&mut self) {
        let c_me = self.hands[self.me].iter().filter(|c| c.is_none()).count() as f64;
        let c_opp = self
            .hands
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.me)
            .map(|(_, h)| h.iter().filter(|c| c.is_none()).count())
            .sum::<usize>() as f64;
        let c_deck = self.deck_count as f64;
        // 弃牌堆里"我还不知道点数"的张数：已知的那些已经被划出池子了
        let c_disc = (self.discard_count.saturating_sub(1) as f64
            - self.known_discard.iter().map(|&c| c as f64).sum::<f64>())
        .max(0.0);
        self.my_unknown_slots = c_me as usize;
        self.opp_unknown_slots = c_opp as usize;
        let counts = [c_me, c_opp, c_deck, c_disc];
        let known_total: f64 = self.known_discard.iter().map(|&c| c as f64).sum();
        let disc_below = self.discard_count.saturating_sub(1) as f64;
        let known_frac = if disc_below > 0.0 { known_total / disc_below } else { 0.0 };
        let thetas = [
            0.0,
            theta_opp_for_scaled(self.n, self.deck_count, known_frac),
            0.0,
            THETA_DISCARD,
        ];
        let mut num = [0f64; 4];
        let mut sq = [0f64; 4];
        let mut den = [0f64; 4];
        for r in 0..14 {
            let n_r = self.pool[r] as f64;
            if n_r <= 0.0 {
                continue;
            }
            let mut w = [0f64; 4];
            let mut total = 0f64;
            for t in 0..4 {
                w[t] = counts[t] * (thetas[t] * r as f64).exp();
                total += w[t];
            }
            if total <= 0.0 {
                continue;
            }
            for t in 0..4 {
                let share = n_r * w[t] / total;
                num[t] += share * r as f64;
                sq[t] += share * (r * r) as f64;
                den[t] += share;
            }
        }
        let stat = |t: usize| -> (f64, f64) {
            if den[t] <= 0.0 {
                return (6.5, 12.0);
            }
            let mean = num[t] / den[t];
            (mean, (sq[t] / den[t] - mean * mean).max(0.0))
        };
        let (m_me, v_me) = stat(0);
        let (m_opp, v_opp) = stat(1);
        let (m_deck, _) = stat(2);
        self.mean_me = m_me;
        self.var_me = v_me;
        self.mean_opp = m_opp;
        self.var_opp = v_opp;
        self.mean_deck = m_deck;
    }

    fn raw_pool_mean(&self) -> f64 {
        let mut total = 0f64;
        let mut cnt = 0f64;
        for (rank, &c) in self.pool.iter().enumerate() {
            total += rank as f64 * c as f64;
            cnt += c as f64;
        }
        if cnt <= 0.0 {
            6.5
        } else {
            total / cnt
        }
    }

    fn raw_pool_var(&self, mean: f64) -> f64 {
        let mut sq = 0f64;
        let mut cnt = 0f64;
        for (rank, &c) in self.pool.iter().enumerate() {
            sq += (rank * rank) as f64 * c as f64;
            cnt += c as f64;
        }
        if cnt <= 0.0 {
            12.0
        } else {
            (sq / cnt - mean * mean).max(0.0)
        }
    }

    /// 我完全看不到点数的牌位总数（未知手牌 + 摸牌堆 + 弃牌堆里我没记住的部分）。
    pub fn unknown_positions(&self) -> usize {
        self.pool.iter().map(|&c| c as usize).sum()
    }

    /// 未知牌池的均值 / 方差（未做倾斜修正）。
    pub fn pool_mean_var(&self) -> (f64, f64) {
        let mean = self.raw_pool_mean();
        (mean, self.raw_pool_var(mean))
    }

    /// 座位 `seat` 的最终总点数估计（已知牌 + 未知牌的期望）。
    /// 未知牌按位置区分：自己暗牌用 `mean_me`，对手暗牌用 `mean_opp`。
    pub fn est(&self, seat: PlayerId) -> (f64, f64) {
        let hand = &self.hands[seat];
        let mut known = 0f64;
        let mut unknown = 0usize;
        for c in hand {
            match c {
                Some(r) => known += *r as f64,
                None => unknown += 1,
            }
        }
        let (mean, var, positions) = if seat == self.me {
            (self.mean_me, self.var_me, self.my_unknown_slots)
        } else {
            (self.mean_opp, self.var_opp, self.opp_unknown_slots)
        };
        let k = unknown as f64;
        let n_pos = positions.max(1) as f64;
        // 无放回抽样的方差（有限总体修正）。
        let fpc = if n_pos > 1.0 { ((n_pos - k) / (n_pos - 1.0)).clamp(0.0, 1.0) } else { 1.0 };
        (known + k * mean, k * var * fpc)
    }

    /// 我"严格最低"的概率估计（正态近似，各对手相互独立）。
    pub fn p_strict_lowest(&self) -> f64 {
        let (em, vm) = self.est(self.me);
        let mut p = 1.0;
        for j in 0..self.n {
            if j == self.me {
                continue;
            }
            let (ej, vj) = self.est(j);
            let sd = (vm + vj).sqrt().max(1e-6);
            p *= norm_cdf((ej - em) / sd);
        }
        p
    }

    pub fn hand_size(&self, seat: PlayerId) -> usize {
        self.hands[seat].len()
    }

    /// 我已知点数的槽位。
    pub fn known_slots(&self, seat: PlayerId) -> Vec<(SlotId, u8)> {
        self.hands[seat]
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|r| (i as SlotId, r)))
            .collect()
    }

    /// 我未知点数的槽位。
    pub fn unknown_slots(&self, seat: PlayerId) -> Vec<SlotId> {
        self.hands[seat]
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_none())
            .map(|(i, _)| i as SlotId)
            .collect()
    }

    /// 我已知的同点数组（按点数分组，只返回 ≥2 张的组）。
    pub fn groups(&self, seat: PlayerId) -> Vec<(u8, Vec<SlotId>)> {
        let mut out: Vec<(u8, Vec<SlotId>)> = Vec::new();
        for (slot, rank) in self.known_slots(seat) {
            match out.iter_mut().find(|(r, _)| *r == rank) {
                Some((_, v)) => v.push(slot),
                None => out.push((rank, vec![slot])),
            }
        }
        out.retain(|(_, v)| v.len() >= 2);
        out
    }

    /// 已知点数最高的槽位。
    pub fn highest_known(&self, seat: PlayerId) -> Option<(SlotId, u8)> {
        self.known_slots(seat).into_iter().max_by_key(|(_, r)| *r)
    }

    /// 我的一张暗牌的期望点数（"该不该拿它换牌"用它）。
    pub fn mean(&self) -> f64 {
        self.mean_me
    }
}

/// 标准正态 CDF 的 logistic 近似（足够精确且便宜）。
pub fn norm_cdf(x: f64) -> f64 {
    1.0 / (1.0 + (-1.702 * x).exp())
}

// ---------------------------------------------------------------- 策略参数

#[derive(Clone, Copy, Debug)]
pub struct PolicyCfg {
    /// 宣告 Cabo 的"严格最低"概率阈值。
    pub cabo_p: f64,
    /// 宣告 Cabo 时，自身期望点数至少要比最强劲对手低多少。
    pub cabo_margin: f64,
    /// 摸到未知牌时，用未知手牌换入的期望收益阈值。
    pub swap_unknown_gain: f64,
    /// 用弃牌堆顶换掉已知牌时，允许的最小收益（负数 = 允许轻微亏损换手牌变少）。
    pub group_min_gain: f64,
    /// 能力牌发动的最低点差（摸到的牌比手上最高的牌低多少才直接换入）。
    pub power_swap_gain: f64,
    /// 是否使用 11/12 的交换能力。
    pub use_swap_power: bool,
    /// 是否使用 9/10 的间谍能力。
    pub use_spy_power: bool,
    /// 是否使用 7/8 的偷看能力。
    pub use_peek_power: bool,
}

impl Default for PolicyCfg {
    fn default() -> Self {
        PolicyCfg {
            // 实测：宣告 Cabo 的 +10 惩罚风险远大于"锁住自己手牌"的收益，
            // 阈值越高越强，`2.0`（等价于从不宣告）在自对弈里显著最优。
            cabo_p: 2.0,
            cabo_margin: 1.0,
            // 未知牌换弃牌堆顶：阈值取负数最好（实测 -4 比 +1 强约 9 分/局）。
            // 除了期望点数，换取"已知牌 + 少让对手拿到低牌"本身也有价值。
            swap_unknown_gain: -4.0,
            // 同点数组换牌：手牌变少很赚，但不该为此接受明显的点数亏损。
            group_min_gain: 0.0,
            power_swap_gain: 3.0,
            use_swap_power: true,
            use_spy_power: true,
            use_peek_power: true,
        }
    }
}

impl PolicyCfg {
    /// 调参用：`key=value` 覆盖（未知键返回 false）。
    pub fn set(&mut self, key: &str, val: &str) -> bool {
        let f = |v: &str| v.parse::<f64>().ok();
        let b = |v: &str| match v {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        };
        match key {
            "cabo_p" => f(val).map(|v| self.cabo_p = v),
            "cabo_margin" => f(val).map(|v| self.cabo_margin = v),
            "swap_unknown_gain" => f(val).map(|v| self.swap_unknown_gain = v),
            "group_min_gain" => f(val).map(|v| self.group_min_gain = v),
            "power_swap_gain" => f(val).map(|v| self.power_swap_gain = v),
            "use_swap_power" => b(val).map(|v| self.use_swap_power = v),
            "use_spy_power" => b(val).map(|v| self.use_spy_power = v),
            "use_peek_power" => b(val).map(|v| self.use_peek_power = v),
            _ => None,
        }
        .is_some()
    }
}

// ---------------------------------------------------------------- 启发式策略

/// 快速策略：给任何座位在任意待决状态下生成一条合法命令。
/// 既作为启发式 Bot 的主体，也作为 PIMC rollout 的默认策略。
pub fn policy(s: &Session, me: PlayerId, cfg: &PolicyCfg, _rng: &mut dyn RngCore) -> Command {
    match &s.phase {
        Phase::Peeking { done } => {
            if done.contains_key(&me) {
                return raw_fallback(s, me);
            }
            let cnt = s.players[me].slots.len();
            if cnt >= 2 {
                // 开局四张牌对称，看哪两张都一样。
                Command::PeekInitial { slots: [0, 1] }
            } else {
                raw_fallback(s, me)
            }
        }
        Phase::Turn { current, pending } if *current == me => match pending {
            None => idle_decision(s, me, cfg),
            Some(Pending::Drew { card }) => drawn_decision(s, me, *card, cfg),
            Some(Pending::SwapSelecting { selected, .. }) => {
                if selected.is_empty() {
                    let slot = s.players[me].slots.first().map(|_| 0u8).unwrap_or(0);
                    Command::SwapToggle { slot }
                } else {
                    Command::SwapCommit
                }
            }
            Some(Pending::PowerAiming { .. }) => Command::Cancel,
            Some(Pending::ConfirmCabo) => Command::CallCabo,
        },
        _ => raw_fallback(s, me),
    }
}

/// 不依赖视图的保底合法命令（rollout 出错时的兜底）。
pub fn raw_fallback(s: &Session, _me: PlayerId) -> Command {
    match &s.phase {
        Phase::Peeking { .. } => Command::PeekInitial { slots: [0, 1] },
        Phase::Turn { pending, .. } => match pending {
            None => {
                if !s.deck.is_empty() {
                    Command::BeginDraw
                } else if !s.discard.is_empty() {
                    Command::SwapOnce { slots: vec![0] }
                } else {
                    Command::CallCabo
                }
            }
            Some(Pending::Drew { .. }) => Command::DiscardDrawn { power: None },
            Some(Pending::PowerAiming { .. }) => Command::Cancel,
            Some(Pending::SwapSelecting { selected, .. }) => {
                if selected.is_empty() {
                    Command::SwapToggle { slot: 0 }
                } else {
                    Command::SwapCommit
                }
            }
            Some(Pending::ConfirmCabo) => Command::CallCabo,
        },
        _ => Command::Cancel,
    }
}

/// 是否值得宣告 Cabo。
pub fn should_call_cabo(k: &Know, cfg: &PolicyCfg) -> bool {
    if k.cabo_called {
        return false;
    }
    let (em, _) = k.est(k.me);
    let mut min_opp = f64::MAX;
    for j in 0..k.n {
        if j != k.me {
            min_opp = min_opp.min(k.est(j).0);
        }
    }
    if em > min_opp - cfg.cabo_margin {
        return false;
    }
    k.p_strict_lowest() >= cfg.cabo_p
}

fn idle_decision(s: &Session, me: PlayerId, cfg: &PolicyCfg) -> Command {
    let k = Know::from_session(s, me);
    if should_call_cabo(&k, cfg) {
        return Command::CallCabo;
    }
    if let Some(top) = k.discard_top {
        let topf = top as f64;
        // 1) 已知的同点数组：一次换掉整组，手牌变少。
        let mut best: Option<(f64, Vec<SlotId>)> = None;
        for (rank, slots) in k.groups(me) {
            let gain = slots.len() as f64 * rank as f64 - topf;
            if gain >= cfg.group_min_gain && best.as_ref().is_none_or(|(g, _)| gain > *g) {
                best = Some((gain, slots));
            }
        }
        if let Some((_, slots)) = best {
            return Command::SwapOnce { slots };
        }
        // 2) 单张：换掉我已知点数里最大、且比堆顶差的那张。
        if let Some((slot, val)) = k.highest_known(me) {
            if (val as f64) > topf {
                return Command::SwapOnce { slots: vec![slot] };
            }
        }
        // 3) 未知牌：堆顶明显低于未知池均值时才值得换取情报。
        if topf + cfg.swap_unknown_gain < k.mean() {
            if let Some(slot) = k.unknown_slots(me).first().copied() {
                return Command::SwapOnce { slots: vec![slot] };
            }
        }
    }
    if !s.deck.is_empty() {
        return Command::BeginDraw;
    }
    Command::CallCabo
}

fn drawn_decision(s: &Session, me: PlayerId, card: CardId, cfg: &PolicyCfg) -> Command {
    let k = Know::from_session(s, me);
    let rank = s.cards[card as usize].card.rank;
    let rankf = rank as f64;
    let highest = k.highest_known(me);
    let direct_gain = highest.map(|(_, v)| v as f64 - rankf).unwrap_or(f64::NEG_INFINITY);

    // 1) 摸到的牌与手里已知牌同点：整组换掉，手牌变少。
    let same: Vec<SlotId> = k
        .known_slots(me)
        .into_iter()
        .filter(|(_, r)| *r == rank)
        .map(|(s, _)| s)
        .collect();
    if same.len() >= 2 {
        return Command::DrawSwap { slots: same };
    }

    // 2) 能力牌：先判断"直接换入"是否已经更划算。
    let direct_ok = direct_gain >= cfg.power_swap_gain;
    if !direct_ok {
        match rank {
            11..=12 if cfg.use_swap_power => {
                // 把自己已知的高牌盲换给对手，换回一张未知牌。
                if let Some((slot, val)) = highest {
                    if val >= 8 {
                        if let Some((p, os)) = spy_target(&k) {
                            return Command::DiscardDrawn {
                                power: Some(PowerUse::Swap { my_slot: slot, player: p, slot: os }),
                            };
                        }
                    }
                }
            }
            9..=10 if cfg.use_spy_power => {
                if let Some((p, os)) = spy_target(&k) {
                    return Command::DiscardDrawn { power: Some(PowerUse::Spy { player: p, slot: os }) };
                }
            }
            7..=8 if cfg.use_peek_power => {
                if let Some(slot) = k.unknown_slots(me).first().copied() {
                    return Command::DiscardDrawn { power: Some(PowerUse::PeekOwn { slot }) };
                }
            }
            _ => {}
        }
    }

    // 3) 直接换入：比手上最高的已知牌小。
    if let Some((slot, val)) = highest {
        if rankf < val as f64 {
            return Command::DrawSwap { slots: vec![slot] };
        }
    }
    // 4) 未知牌：明显优于未知池均值时换入（顺便获得确定信息）。
    if k.mean() > rankf + cfg.swap_unknown_gain {
        if let Some(slot) = k.unknown_slots(me).first().copied() {
            return Command::DrawSwap { slots: vec![slot] };
        }
    }
    Command::DiscardDrawn { power: None }
}

/// 间谍 / 交换能力的目标：手牌最多、未知牌最多的对手身上的一张未知牌。
pub fn spy_target(k: &Know) -> Option<(PlayerId, SlotId)> {
    let mut best: Option<(PlayerId, SlotId, usize)> = None;
    for j in 0..k.n {
        if j == k.me {
            continue;
        }
        let unknown = k.unknown_slots(j);
        if unknown.is_empty() {
            continue;
        }
        let score = unknown.len();
        if best.as_ref().is_none_or(|(_, _, bs)| score > *bs) {
            best = Some((j, unknown[0], score));
        }
    }
    best.map(|(p, s, _)| (p, s))
}

// 以下策略函数是原搜索版的先验和 rollout 组件。

/// 只依赖视图的启发式决策。
pub fn view_policy(view: &PlayerView, cfg: &PolicyCfg, rng: &mut dyn RngCore) -> Command {
    view_policy_with(view, cfg, rng, &[0; 14])
}

/// 只依赖视图 + 公共历史（弃牌堆构成）的启发式决策。
pub fn view_policy_with(
    view: &PlayerView,
    cfg: &PolicyCfg,
    rng: &mut dyn RngCore,
    known_discard: &[u16; 14],
) -> Command {
    let k = Know::from_view_with(view, known_discard);
    let me = k.me;
    match &view.panel {
        Panel::Idle { can_draw, can_swap_discard, can_cabo } => {
            if *can_cabo && should_call_cabo(&k, cfg) {
                return Command::CallCabo;
            }
            if *can_swap_discard {
                if let Some(cmd) = view_swap_from_discard(&k, cfg) {
                    return cmd;
                }
            }
            if *can_draw {
                return Command::BeginDraw;
            }
            Command::CallCabo
        }
        Panel::Drew { rank, .. } => view_drawn(&k, *rank, cfg, me, view),
        Panel::AimPeek => {
            let slot = k.unknown_slots(me).first().copied().unwrap_or(0);
            Command::PowerPickOwn { slot }
        }
        Panel::AimSpy | Panel::AimSwap { .. } => match spy_target(&k) {
            Some((p, s)) => Command::PowerPickOther { player: p, slot: s },
            None => Command::Cancel,
        },
        Panel::SwapSelecting { count, source, .. } => {
            if *count >= 1 {
                Command::SwapCommit
            } else if matches!(source, Pile::Draw) {
                let slot = k
                    .unknown_slots(me)
                    .first()
                    .copied()
                    .or_else(|| k.highest_known(me).map(|(s, _)| s))
                    .unwrap_or(0);
                Command::SwapToggle { slot }
            } else {
                Command::Cancel
            }
        }
        Panel::ConfirmCabo => Command::CallCabo,
        _ => {
            let _ = rng;
            crate::ai::fallback_command(view)
        }
    }
}

fn view_swap_from_discard(k: &Know, cfg: &PolicyCfg) -> Option<Command> {
    let me = k.me;
    let top = k.discard_top?;
    let topf = top as f64;
    let mut best: Option<(f64, Vec<SlotId>)> = None;
    for (rank, slots) in k.groups(me) {
        let gain = slots.len() as f64 * rank as f64 - topf;
        if gain >= cfg.group_min_gain && best.as_ref().is_none_or(|(g, _)| gain > *g) {
            best = Some((gain, slots));
        }
    }
    if let Some((_, slots)) = best {
        return Some(Command::SwapOnce { slots });
    }
    if let Some((slot, val)) = k.highest_known(me) {
        if (val as f64) > topf {
            return Some(Command::SwapOnce { slots: vec![slot] });
        }
    }
    if topf + cfg.swap_unknown_gain < k.mean() {
        if let Some(slot) = k.unknown_slots(me).first().copied() {
            return Some(Command::SwapOnce { slots: vec![slot] });
        }
    }
    None
}

fn view_drawn(
    k: &Know,
    rank: u8,
    cfg: &PolicyCfg,
    me: PlayerId,
    view: &PlayerView,
) -> Command {
    let rankf = rank as f64;
    let highest = k.highest_known(me);
    let direct_gain = highest.map(|(_, v)| v as f64 - rankf).unwrap_or(f64::NEG_INFINITY);
    let same: Vec<SlotId> = k
        .known_slots(me)
        .into_iter()
        .filter(|(_, r)| *r == rank)
        .map(|(s, _)| s)
        .collect();
    if same.len() >= 2 {
        return Command::DrawSwap { slots: same };
    }
    let direct_ok = direct_gain >= cfg.power_swap_gain;
    if !direct_ok {
        match rank {
            11..=12 if cfg.use_swap_power => {
                if let Some((slot, val)) = highest {
                    if val >= 8 {
                        if let Some((p, os)) = spy_target(k) {
                            return Command::DiscardDrawn {
                                power: Some(PowerUse::Swap { my_slot: slot, player: p, slot: os }),
                            };
                        }
                    }
                }
            }
            9..=10 if cfg.use_spy_power => {
                if let Some((p, os)) = spy_target(k) {
                    return Command::DiscardDrawn { power: Some(PowerUse::Spy { player: p, slot: os }) };
                }
            }
            7..=8 if cfg.use_peek_power => {
                if let Some(slot) = k.unknown_slots(me).first().copied() {
                    return Command::DiscardDrawn { power: Some(PowerUse::PeekOwn { slot }) };
                }
            }
            _ => {}
        }
    }
    if let Some((slot, val)) = highest {
        if rankf < val as f64 {
            return Command::DrawSwap { slots: vec![slot] };
        }
    }
    if k.mean() > rankf + cfg.swap_unknown_gain {
        if let Some(slot) = k.unknown_slots(me).first().copied() {
            return Command::DrawSwap { slots: vec![slot] };
        }
    }
    let _ = view;
    Command::DiscardDrawn { power: None }
}
