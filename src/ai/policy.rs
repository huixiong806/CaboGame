//! 策略网络（候选打分）：**阶段 1** 的学习组件。
//!
//! 设计要点（见历史备份分支的 `DL_ROUTE.md` §4/§5）：
//! - 输入 = **视图**（不完美信息）——因为标签"教师选了哪个候选"是视图的确定函数；
//! - 输出 = 对**候选集合**打分 → 候选间 softmax（列表排序，天然支持变长动作集）；
//! - 关系信息**显式塞进候选特征**（谁换谁、会不会公开某张牌），不依赖注意力去发现；
//! - 卡片集合一律用**重数向量**（副本对称性由编码保证，计数查询是线性的）；
//! - 手牌张数用**聚合特征**而非定长槽位向量，因此对 >4 张天然可用（配合硬开关）。

use crate::ai::tactics::Know;
use crate::game::view::{Panel, PlayerView};
use crate::game::{power_kind_of, Command, Pile, PlayerId, PowerUse, RANK_COPIES};

/// 状态特征维度。
pub const N_STATE: usize = 168;
/// 单候选特征维度。
pub const N_CAND: usize = 30;
/// 打分网络输入维度。
pub const N_IN: usize = N_STATE + N_CAND;

fn put(f: &mut [f32], k: &mut usize, v: f32) {
    if *k < f.len() {
        f[*k] = v;
    }
    *k += 1;
}

/// 点数重数向量（14 维，已按该点数总张数归一化 → "这个点数还剩几成"）。
fn push_multiplicity(f: &mut [f32], k: &mut usize, counts: &[u16; 14]) {
    for r in 0..14 {
        put(f, k, counts[r] as f32 / RANK_COPIES[r] as f32);
    }
}

/// 由"已知点数的槽位"统计重数。
fn known_counts(hand: &[Option<u8>]) -> [u16; 14] {
    let mut c = [0u16; 14];
    for v in hand.iter().flatten() {
        c[*v as usize] += 1;
    }
    c
}

/// 状态特征的**来源上下文**：既能从 [`PlayerView`]（搜索根节点）构造，
/// 也能从 [`crate::game::Session`]（rollout 内部）构造——后者的每个座位也在决策，
/// 但拿不到投影视图。
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    pub round_no: u32,
    pub extra_left: usize,
    pub my_slot_count: usize,
    pub panel: PanelKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelKind {
    Idle,
    Drew,
    Selecting,
    Power,
    Other,
}

impl Ctx {
    pub fn from_view(view: &PlayerView) -> Ctx {
        Ctx {
            round_no: view.round_no,
            extra_left: view.extra_left,
            my_slot_count: view.me_seat.as_ref().map(|s| s.slots.len()).unwrap_or(0),
            panel: match view.panel {
                Panel::Idle { .. } => PanelKind::Idle,
                Panel::Drew { .. } => PanelKind::Drew,
                Panel::SwapSelecting { .. } => PanelKind::Selecting,
                Panel::PeekPick { .. } => PanelKind::Other,
                _ => PanelKind::Power,
            },
        }
    }

    /// 从完整局面构造（rollout 内部用）：面板类别由 `pending` 推出。
    pub fn from_session(s: &crate::game::Session, pid: PlayerId) -> Ctx {
        use crate::game::{Pending, Phase};
        let (panel, extra_left) = match &s.phase {
            Phase::Turn { pending, current } if *current == pid => (
                match pending {
                    None => PanelKind::Idle,
                    Some(Pending::Drew { .. }) => PanelKind::Drew,
                    Some(Pending::SwapSelecting { .. }) => PanelKind::Selecting,
                    Some(Pending::PowerAiming { .. }) => PanelKind::Power,
                    Some(Pending::ConfirmCabo) => PanelKind::Other,
                },
                s.extra_turns.len(),
            ),
            _ => (PanelKind::Other, s.extra_turns.len()),
        };
        Ctx {
            round_no: s.round_no,
            extra_left,
            my_slot_count: s.players.get(pid).map(|p| p.slots.len()).unwrap_or(0),
            panel,
        }
    }
}

/// 状态特征（视图视角）。
pub fn state_features(view: &PlayerView, k: &Know) -> Vec<f32> {
    state_features_ctx(&Ctx::from_view(view), k)
}

/// 状态特征（上下文入口）：rollout 内部只有 `Session`，用 `Ctx::from_session` 构造。
pub fn state_features_ctx(ctx: &Ctx, k: &Know) -> Vec<f32> {
    let mut f = vec![0f32; N_STATE];
    let mut i = 0usize;
    let me = k.me;
    let n = k.n;

    // ---- 我的手牌：重数 + 聚合（不用定长槽位 → 张数无关）----
    let my = known_counts(&k.hands[me]);
    push_multiplicity(&mut f, &mut i, &my);
    let my_cnt = k.hands[me].len() as f32;
    let my_known: f32 = k.hands[me].iter().flatten().map(|&r| r as f32).sum();
    let my_unknown = k.hands[me].iter().filter(|c| c.is_none()).count() as f32;
    put(&mut f, &mut i, my_cnt / 8.0);
    put(&mut f, &mut i, my_known / 40.0);
    put(&mut f, &mut i, my_unknown / 8.0);
    put(&mut f, &mut i, (my_known + my_unknown * k.mean_me as f32) / 40.0); // 期望总点数
    put(&mut f, &mut i, k.est(me).0 as f32 / 40.0);
    put(&mut f, &mut i, k.hands[me].iter().flatten().copied().max().unwrap_or(0) as f32 / 13.0);

    // ---- 对手（按累计分从低到高排序，最多 3 个）----
    let mut opp: Vec<(PlayerId, f32)> =
        (0..n).filter(|&j| j != me).map(|j| (j, k.est(j).0 as f32)).collect();
    opp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for idx in 0..3 {
        match opp.get(idx) {
            Some(&(j, _)) => {
                let cnt = known_counts(&k.hands[j]);
                push_multiplicity(&mut f, &mut i, &cnt);
                let total = k.totals.get(j).copied().unwrap_or(0) as f32;
                let unknown = k.hands[j].iter().filter(|c| c.is_none()).count() as f32;
                put(&mut f, &mut i, k.hands[j].len() as f32 / 8.0);
                put(&mut f, &mut i, unknown / 8.0);
                put(&mut f, &mut i, k.est(j).0 as f32 / 40.0);
                put(&mut f, &mut i, total / 100.0);
                put(&mut f, &mut i, (k.target as f32 - total) / 100.0); // 分数余量
                put(&mut f, &mut i, (k.est(j).0 - k.est(me).0) as f32); // 相对优势
            }
            None => {
                for _ in 0..(14 + 5) {
                    put(&mut f, &mut i, 0.0);
                }
            }
        }
    }

    // ---- 未见牌池（已扣除公共历史恢复的弃牌堆）与弃牌堆构成 ----
    push_multiplicity(&mut f, &mut i, &k.pool);
    let mut disc = [0u16; 14];
    // 弃牌堆里已知的部分 = 池子被扣掉的那部分，这里用"整副牌 − 池 − 已知"近似不了，
    // 直接用 history 恢复的信息（存在 Know.known_discard 里）
    for r in 0..14 {
        disc[r] = k.known_discard[r];
    }
    push_multiplicity(&mut f, &mut i, &disc);
    // 能力类别剩余
    let peek: u16 = k.pool[7] + k.pool[8];
    let spy: u16 = k.pool[9] + k.pool[10];
    let swp: u16 = k.pool[11] + k.pool[12];
    put(&mut f, &mut i, peek as f32 / 8.0);
    put(&mut f, &mut i, spy as f32 / 8.0);
    put(&mut f, &mut i, swp as f32 / 8.0);

    // ---- 桌面与上下文 ----
    put(&mut f, &mut i, k.deck_count as f32 / 52.0);
    put(&mut f, &mut i, k.discard_count as f32 / 52.0);
    for r in 0..14 {
        put(&mut f, &mut i, if k.discard_top == Some(r as u8) { 1.0 } else { 0.0 });
    }
    put(&mut f, &mut i, k.discard_top.unwrap_or(0) as f32 / 13.0);
    put(&mut f, &mut i, ctx.round_no as f32 / 20.0);
    put(&mut f, &mut i, k.target as f32 / 100.0);
    put(&mut f, &mut i, k.penalty as f32 / 10.0);
    put(&mut f, &mut i, n as f32 / 4.0);
    put(&mut f, &mut i, if k.cabo_called { 1.0 } else { 0.0 });
    put(&mut f, &mut i, ctx.extra_left as f32 / 4.0);
    put(&mut f, &mut i, ctx.my_slot_count as f32 / 8.0);
    // 待决状态（面板类别）
    let (p_idle, p_drew, p_sel, p_power) = match ctx.panel {
        PanelKind::Idle => (1.0, 0.0, 0.0, 0.0),
        PanelKind::Drew => (0.0, 1.0, 0.0, 0.0),
        PanelKind::Selecting => (0.0, 0.0, 1.0, 0.0),
        PanelKind::Power => (0.0, 0.0, 0.0, 1.0),
        PanelKind::Other => (0.0, 0.0, 0.0, 0.0),
    };
    put(&mut f, &mut i, p_idle);
    put(&mut f, &mut i, p_drew);
    put(&mut f, &mut i, p_sel);
    put(&mut f, &mut i, p_power);
    // 专家中间量（少量）
    put(&mut f, &mut i, k.p_strict_lowest() as f32);
    put(&mut f, &mut i, 1.0); // 偏置
    debug_assert!(i <= N_STATE, "状态特征超维：{i} > {N_STATE}");
    f
}

/// 候选特征（把关系显式化）。
///
/// `prior` = 启发式先验选中的候选下标（None 表示不适用），`static_score` = 候选生成器
/// 给出的静态评分。**这两个"基线"信号必须喂进去**：教师有 ~89% 的时候与先验一致，
/// 让网络从零学整个映射会把绝大多数容量浪费在"复述先验"上（实测 top-1 反而低于先验）。
/// 给了基线，网络只需要学"什么时候该偏离先验、往哪偏"——和价值的残差是同一个道理。
pub fn candidate_features(
    view: &PlayerView,
    k: &Know,
    cmd: &Command,
    cand_idx: usize,
    prior: Option<usize>,
    static_score: f32,
) -> Vec<f32> {
    let mut f = vec![0f32; N_CAND];
    let mut i = 0usize;
    let me = k.me;
    let known_slots = k.known_slots(me);
    let sum_known = |slots: &[u8]| -> f32 {
        slots
            .iter()
            .filter_map(|s| known_slots.iter().find(|(x, _)| x == s).map(|(_, r)| *r as f32))
            .sum()
    };
    let max_known = |slots: &[u8]| -> f32 {
        slots
            .iter()
            .filter_map(|s| known_slots.iter().find(|(x, _)| x == s).map(|(_, r)| *r as f32))
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let unknown_in = |slots: &[u8]| -> f32 {
        slots
            .iter()
            .filter(|s| !known_slots.iter().any(|(x, _)| x == *s))
            .count() as f32
    };
    // 动作类型 one-hot
    let (t_draw, t_swap1, t_swapn, t_ds1, t_dsn, t_disc, t_pk, t_sp, t_sw) = match cmd {
        Command::BeginDraw => (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        Command::SwapOnce { slots } => {
            if slots.len() > 1 {
                (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            } else {
                (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        }
        Command::DrawSwap { slots } => {
            if slots.len() > 1 {
                (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        }
        Command::DiscardDrawn { power: None } => (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        Command::DiscardDrawn { power: Some(p) } => match p {
            PowerUse::PeekOwn { .. } => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            PowerUse::Spy { .. } => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            PowerUse::Swap { .. } => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
        Command::CallCabo | Command::CallCaboArm => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        _ => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    };
    for v in [t_draw, t_swap1, t_swapn, t_ds1, t_dsn, t_disc, t_pk, t_sp, t_sw] {
        put(&mut f, &mut i, v);
    }
    put(&mut f, &mut i, if matches!(cmd, Command::CallCabo) { 1.0 } else { 0.0 });

    // 涉及槽位的聚合
    let slots: Vec<u8> = match cmd {
        Command::SwapOnce { slots } | Command::DrawSwap { slots } => slots.clone(),
        Command::DiscardDrawn { power: Some(PowerUse::Swap { my_slot, .. }) } => vec![*my_slot],
        _ => Vec::new(),
    };
    let n_slots = slots.len() as f32;
    let known_part = sum_known(&slots);
    let unk = unknown_in(&slots);
    put(&mut f, &mut i, n_slots / 8.0);
    put(&mut f, &mut i, known_part / 40.0);
    put(&mut f, &mut i, unk / 8.0);
    let mx = max_known(&slots);
    put(&mut f, &mut i, if mx.is_finite() { mx / 13.0 } else { 0.0 });
    if !slots.is_empty() {
        let ranks: Vec<u8> = slots
            .iter()
            .filter_map(|s| known_slots.iter().find(|(x, _)| x == s).map(|(_, r)| *r))
            .collect();
        let same = !ranks.is_empty() && ranks.iter().all(|r| *r == ranks[0]);
        put(&mut f, &mut i, if same { 1.0 } else { 0.0 });
        put(&mut f, &mut i, ranks.len() as f32 / 8.0);
    } else {
        put(&mut f, &mut i, 0.0);
        put(&mut f, &mut i, 0.0);
    }
    // 弃牌堆顶对比：换掉已知牌是否划算
    let gain = match (
        k.discard_top.map(|t| t as f32),
        slots.iter().filter_map(|s| known_slots.iter().find(|(x, _)| x == s).map(|(_, r)| *r as f32)).fold(f32::NEG_INFINITY, f32::max),
    ) {
        (Some(t), g) if g.is_finite() => (g - t) / 13.0,
        _ => 0.0,
    };
    put(&mut f, &mut i, gain);
    // 信息暴露：会公开的点数（未知牌被公开的期望代价 ≈ mean_me）
    let exposes = matches!(cmd, Command::SwapOnce { .. } | Command::DrawSwap { .. });
    put(&mut f, &mut i, if exposes { 1.0 } else { 0.0 });
    put(
        &mut f,
        &mut i,
        if exposes {
            (k.mean_me - k.discard_top.unwrap_or(0) as f64) as f32 / 13.0
        } else {
            0.0
        },
    );
    // 能力目标对手信息
    let target_opp: Option<(PlayerId, u8)> = match cmd {
        Command::DiscardDrawn { power: Some(PowerUse::Spy { player, slot }) }
        | Command::DiscardDrawn { power: Some(PowerUse::Swap { player, slot, .. }) } => {
            Some((*player, *slot))
        }
        _ => None,
    };
    match target_opp {
        Some((p, s)) => {
            put(&mut f, &mut i, 1.0);
            put(&mut f, &mut i, k.totals.get(p).copied().unwrap_or(0) as f32 / 100.0);
            put(&mut f, &mut i, k.hands[p].len() as f32 / 8.0);
            put(&mut f, &mut i, k.est(p).0 as f32 / 40.0);
            put(
                &mut f,
                &mut i,
                k.hands[p].get(s as usize).and_then(|c| *c).map(|r| r as f32 / 13.0).unwrap_or(0.5),
            );
        }
        None => {
            for _ in 0..5 {
                put(&mut f, &mut i, 0.0);
            }
        }
    }
    // 来源（摸牌堆 / 弃牌堆）
    let from_draw = matches!(cmd, Command::DrawSwap { .. } | Command::BeginDraw | Command::DiscardDrawn { .. });
    put(&mut f, &mut i, if from_draw { 1.0 } else { 0.0 });
    let _ = Pile::Draw;
    // ---- 基线信号：静态评分 + 是否先验之选 ----
    put(&mut f, &mut i, static_score / 10.0);
    put(&mut f, &mut i, if prior == Some(cand_idx) { 1.0 } else { 0.0 });
    put(&mut f, &mut i, 1.0); // 偏置
    debug_assert!(i <= N_CAND, "候选特征超维：{i} > {N_CAND}");
    f
}

/// 候选是否"意味着发动能力牌"（供上层做日志/统计）。
pub fn is_power(cmd: &Command) -> bool {
    matches!(cmd, Command::DiscardDrawn { power: Some(_) })
        || matches!(cmd, Command::BeginDraw if power_kind_of(7).is_some())
}

// ---------------------------------------------------------------- 网络推理

use std::sync::OnceLock;

use super::nn::Mlp;

static NET: OnceLock<Option<Mlp>> = OnceLock::new();

/// 载入嵌入的策略网络（未训练时返回 None）。
pub fn net() -> Option<&'static Mlp> {
    NET.get_or_init(|| {
        let data = super::weights::POLICY_WEIGHTS;
        if data.is_empty() || super::weights::POLICY_SIZES.len() < 2 {
            return None;
        }
        Mlp::import(data)
    })
    .as_ref()
}

pub fn available() -> bool {
    net().is_some()
}

/// 对一个候选打分（logit）。
pub fn score(net: &Mlp, state: &[f32], cand: &[f32]) -> f32 {
    let mut x = Vec::with_capacity(N_IN);
    x.extend_from_slice(state);
    x.extend_from_slice(cand);
    net.predict(&x)
}

/// 用策略网络做决策的 Bot（"学生"）。用于度量"学生 vs 教师"的配对悔值，
/// 以及作为搜索 rollout 里的行动策略。
pub struct PolicyBot {
    pub history: super::history::History,
}

impl PolicyBot {
    pub fn new() -> Self {
        PolicyBot { history: super::history::History::new() }
    }
}

impl Default for PolicyBot {
    fn default() -> Self {
        Self::new()
    }
}

impl super::Bot for PolicyBot {
    fn id(&self) -> &'static str {
        "policy"
    }
    fn name(&self) -> &'static str {
        "策略 AI"
    }
    fn decide(&self, view: &PlayerView, rng: &mut dyn rand::RngCore) -> crate::game::Command {
        if let Panel::PeekPick { .. } = view.panel {
            return crate::game::Command::PeekInitial { slots: [0, 1] };
        }
        let info = self.history.observe(view);
        let k = Know::from_view_with(view, &info.known);
        let cfg = crate::ai::tactics::PolicyCfg::default();
        choose(view, &k, &info.known, rng, &cfg)
            .unwrap_or_else(|| crate::ai::tactics::view_policy_with(view, &cfg, rng, &info.known))
    }
}

/// 从 **Session** 出发用策略网络挑动作（rollout 内部用）。
///
/// 成本分解（release，实测）：project 1.9µs + Know 0.4µs + 候选 0.74µs +
/// 特征 0.33µs + 前向（64 宽 5µs / 256 宽 59µs）。
pub fn choose_from_session(
    s: &crate::game::Session,
    pid: PlayerId,
    rng: &mut dyn rand::RngCore,
    policy_cfg: &crate::ai::tactics::PolicyCfg,
) -> Option<crate::game::Command> {
    let net = net()?;
    let view = crate::game::view::project(s, Some(pid), 0);
    // 弃牌堆构成：真实 bot 靠 history.rs 恢复（实测召回 68.6%），
    // rollout 里直接用该确定性世界的弃牌堆（完美记忆），保证与训练时的特征分布一致。
    let mut known_discard = [0u16; 14];
    for cid in &s.discard {
        known_discard[s.cards[*cid as usize].card.rank as usize] += 1;
    }
    let k = Know::from_view_with(&view, &known_discard);
    let cfg = policy_cfg;
    let scored = crate::ai::search::candidates_scored(&view, &k, 10);
    if scored.len() < 2 {
        return scored.first().map(|(_, c)| c.clone());
    }
    let prior_cmd = crate::ai::tactics::view_policy_with(&view, cfg, rng, &known_discard);
    let prior = scored.iter().position(|(_, c)| *c == prior_cmd);
    let state = state_features(&view, &k);
    let mut best: Option<(usize, f32)> = None;
    for (i, (sc, c)) in scored.iter().enumerate() {
        let cf = candidate_features(&view, &k, c, i, prior, *sc as f32);
        let z = score(net, &state, &cf);
        if best.is_none_or(|(_, bz)| z > bz) {
            best = Some((i, z));
        }
    }
    best.map(|(i, _)| scored[i].1.clone())
}

/// 用策略网络在候选集合上挑一个动作（"学生"的决策）。
/// 网络缺失或候选不足时返回 None，由调用方回退。
pub fn choose(
    view: &PlayerView,
    k: &Know,
    known_discard: &[u16; 14],
    rng: &mut dyn rand::RngCore,
    policy_cfg: &crate::ai::tactics::PolicyCfg,
) -> Option<crate::game::Command> {
    let net = net()?;
    let scored = crate::ai::search::candidates_scored(view, k, 10);
    if scored.len() < 2 {
        return scored.first().map(|(_, c)| c.clone());
    }
    // 候选特征里要带"启发式先验之选"这个基线信号（训练时就是这么喂的）
    let prior_cmd = crate::ai::tactics::view_policy_with(view, policy_cfg, rng, known_discard);
    let prior = scored.iter().position(|(_, c)| *c == prior_cmd);
    let state = state_features(view, k);
    let mut best: Option<(usize, f32)> = None;
    for (i, (s, c)) in scored.iter().enumerate() {
        let cf = candidate_features(view, k, c, i, prior, *s as f32);
        let z = score(net, &state, &cf);
        if best.is_none_or(|(_, bz)| z > bz) {
            best = Some((i, z));
        }
    }
    best.map(|(i, _)| scored[i].1.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tactics::Know;
    use crate::game::sim::make_ai_session;
    use crate::game::view::project;
    use crate::game::Settings;

    #[test]
    fn feature_dims_are_exact() {
        let mut s = make_ai_session(5, Settings::default(), &["challenger", "challenger", "challenger"]);
        s.start_game().unwrap();
        for _ in 0..40 {
            let pid = match &s.phase {
                crate::game::Phase::Peeking { done } => (0..3).find(|p| !done.contains_key(p)),
                crate::game::Phase::Turn { current, .. } => Some(*current),
                _ => None,
            };
            let Some(pid) = pid else { break };
            let view = project(&s, Some(pid), 0);
            let k = Know::from_view(&view);
            let f = state_features(&view, &k);
            assert_eq!(f.len(), N_STATE);
            assert!(f.iter().all(|v| v.is_finite()));
            let cmd = crate::ai::fallback_command(&view);
            let cf = candidate_features(&view, &k, &cmd, 0, None, 0.0);
            assert_eq!(cf.len(), N_CAND);
            assert!(cf.iter().all(|v| v.is_finite()));
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &crate::ai::fallback_command(&view));
            }
            if matches!(s.phase, crate::game::Phase::RoundEnd) {
                s.next_round().unwrap();
            }
        }
    }
}
