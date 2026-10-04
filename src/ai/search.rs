//! 搜索型 Bot：确定性化 + 前瞻模拟（PIMC）。
//!
//! 思路：
//! 1. 用 [`crate::ai::belief::reconstruct`] 把"我看到的视图"随机补全成一个完整局面
//!    （未知牌按**位置**分别抽样：对手暗牌偏低、弃牌堆底偏高、摸牌堆居中）；
//! 2. 枚举当前状态下所有值得考虑的命令（候选动作，按类别保底 + 评分补足裁剪）；
//! 3. 每个候选 × 多个确定性世界，用真实规则引擎把本轮打到结算
//!    （所有座位都用启发式策略行动，只看得见自己该看的牌）；
//! 4. 回报取**胜率口径**：终局 1/0，非终局用"与最强劲对手的分差"过 logistic；
//! 5. 两道保险：宣告 Cabo 与偏离启发式策略都需要额外"加价"（见下文字段说明）。
//!
//! 同一批确定性世界对所有候选共用（公共随机数），全部计算受单步时间预算约束。

use std::time::Instant;

use rand::RngCore;

use crate::game::view::{Panel, PlayerView};
use crate::game::{Command, Phase, PlayerId, PowerUse, Session, SlotId};

use super::belief::reconstruct;
use super::stats;
use super::tactics::{policy, raw_fallback, view_policy_with, Know, PolicyCfg};

/// 单次 rollout 的最大动作数（防止病态循环）。
const MAX_ROLLOUT_ACTIONS: usize = 4000;

/// 回报口径。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    /// **学习到的价值网络**（自我对弈训练，见 [`crate::ai::value`]）：直接估 P(赢下整局)。
    /// 权重未训练时自动退回 `WinProb`。配合 `value_rollout_steps=0` 使用即"1 层搜索"。
    Learned,
    /// **分数上下文胜率模型**：`P(赢 | 分数, 轮次)`，逻辑回归（低维、可解释、校准好）。
    /// 现行解析式 `sigmoid(领先/8)` 完全不看分数位置 —— 领先 20 分在总分 90 和总分 10
    /// 时被估成同一个概率，这正是"领先了也不宣告/错失机会"的建模根源之一。
    ScoreCtx,
    /// **学习到的终局分差修正**（点数口径）：`lead_now + 网络修正量`。
    /// 稠密 target（终局分差回归）+ 残差结构 → 局部差分是点数，
    /// `improve_margin` / `cabo_margin` 的点数语义重新成立。
    LearnedMargin,
    /// **胜率估计**（启发式）：终局直接 1/0，非终局用"与最强劲对手的分差"过一个
    /// logistic 得到胜率。
    WinProb,
    /// 单纯的分差（会与终局奖励量纲不一致，仅用于对照实验）。
    Lead,
    /// 只关心本轮得分。
    Round,
}

/// 搜索参数。
#[derive(Clone, Copy, Debug)]
pub struct SearchCfg {
    /// 单步决策时间上限（微秒）。
    pub budget_us: u64,
    /// 最多采样多少个确定性世界。
    pub max_worlds: u32,
    /// 至少采样多少个世界（即使超时）。
    pub min_worlds: u32,
    /// 候选动作上限（按静态评分裁剪）。
    pub max_cands: usize,
    /// rollout 使用的策略参数。
    pub policy: PolicyCfg,
    /// 回报口径。
    pub value: ValueKind,
    /// 关闭搜索（退化为纯启发式策略，用于对照实验）。
    pub plain_policy: bool,
    /// 是否允许考虑"宣告 Cabo"这一候选动作（实验开关）。
    pub allow_cabo: bool,
    /// 宣告 Cabo 的"加价"，单位是**点数**：只有搜索估值比其它候选高出"这么多点"
    /// 所对应的回报差，才会真的宣告。
    ///
    /// 理由：Cabo 的结果是双峰的（0 分 或 手牌+10），估值方差远大于其他候选；
    /// 在噪声里取 argmax 会系统性高估它（optimizer's curse），而失败的代价是 +10。
    pub cabo_margin: f64,
    /// 宣告 Cabo 的**绝对**门槛：模拟出来的"严格最低"成功率必须达到这个值。
    ///
    /// 为什么需要它：`cabo_margin` 是**相对**比较（比其它候选好多少）。
    /// 当局面整体不利时（继续打下去的估值也很低），相对比较会让机器人去做
    /// 胜率只有五成的赌博式宣告——实测成功率只有 33~50%，而同期启发式策略
    /// 用"p(严格最低) ≥ 0.7"这条**绝对**规则能到 56~58%。
    /// 这里直接统计 CallCabo 分支在世界里"本轮得 0 分"的比例，作为绝对门槛。
    pub cabo_min_success: f64,
    /// 胜率口径的温度：领先 `value_k` 分时胜率约 73%。
    pub value_k: f64,
    /// 宣告 Cabo 是否**只认启发式规则的判断**（搜索不得自行发起宣告）。
    ///
    /// 实测：打开（宣告权交给启发式规则）在对"会宣告的对手"时反而略差
    /// （30 组配对：+2.98 vs -2.98，即关掉更好约 6 分），因此**默认关闭**。
    /// 保留这个开关是因为它揭示了一个真问题：PIMC 自己发起的宣告成功率
    /// 只有 33~50%，而启发式的 p 阈值规则能到 56~58%——
    /// 说明 CallCabo 的回报估计仍偏乐观（结果双峰、样本有限）。
    pub cabo_from_prior: bool,
    /// 是否启用"记牌"（用公共历史重建弃牌堆构成）。做 A/B 对比用。
    pub use_memory: bool,
    /// 策略先验是否改用**学习到的策略网络**（学生），而不是手写启发式。
    pub prior_from_policy: bool,
    /// rollout 内部是否用**策略网络**替代手写启发式：
    /// 0=不用，1=只用在我方座位，2=所有座位，3=只用在本轮 rollout 的第一步（对手的即时回应）。
    pub rollout_net: u8,
    /// rollout 打多少步之后再用价值口径折算（0 = 我动作生效后立刻估值）。
    /// 0 配合 `ValueKind::Learned` 就是"学习价值 + 单层搜索"；
    /// 数值越大越接近"打到本轮结算"，代价是每个候选贵得多、世界数少得多。
    pub value_rollout_steps: u32,
    /// "策略先验"加价：搜索必须**比启发式策略的选择好出这么多**才会偏离它。
    ///
    /// 必要性：估值是有噪声的（世界数有限）。当候选之间的真实差距小于噪声时，
    /// 纯 argmax 等于随机选，会把启发式策略本来连贯的计划打散
    /// （例如"先吃下弃牌堆顶凑成对子、下一手整组换掉"这种两步计划，
    /// 第一手故意吃亏，若第二手随机不做就纯亏）。实测在 2 人局里，
    /// 没有这条规则时搜索会明显弱于它自己用的策略。
    pub improve_margin: f64,
    /// 尺度无关的偏离门槛：最优候选要比在位者好出多少个**候选值标准差**（0 = 关闭）。
    pub improve_sigma: f64,
    /// 信息暴露惩罚：动作把自己**已知的低牌**公开（翻进弃牌堆）时，从该候选的
    /// 平均回报里扣掉的量（与估值同量纲；0 = 关闭）。
    pub expose_penalty: f64,
    /// 方差感知收缩（**注意：CRN 下这个是错的**，保留仅用于对照）。
    pub shrink_noise: bool,
    /// 配对显著性门槛：偏离在位者要求 `mean(d) > paired_t · se(d)`（逐世界配对差值）。
    /// 尺度无关、随世界数自适应，替代单位耦合的 `improve_margin`。
    pub paired_gate: bool,
    pub paired_t: f64,
    /// 定向延长：当 CallCabo 候选挤进前二名时，把预算放宽到 `extend_budget_us`。
    /// 依据：世界数扫描显示普通决策在 K≈256 就稳定，翻的都是宣告类决策；
    /// 而 `cabo_min_success` 这个门槛正是最吃精度的地方。0 = 关闭。
    pub extend_budget_us: u64,
}

impl Default for SearchCfg {
    fn default() -> Self {
        SearchCfg {
            // 时间预算的实测取舍（见 AI_NOTES.md）：
            // 对弱对手：2ms→10ms 提升很大，10ms 之后基本持平；
            // 对自己这种强对手：8ms→50ms 仍有约 7 分/局的提升（配对自对弈）。
            // 默认 300ms —— 实测这是**唯一有显著棋力收益**的旋钮（DL_ROUTE.md D5/D6）：
            // 瓶颈是候选估值的蒙特卡洛精度，误差 ∝ 1/√世界数，可以直接用时间买。
            // 10 倍预算（15ms→150ms）实测 ★ −11.1 / −4.4（两组），
            // 平均累计分差按逆方差合并 = −7.15 ± 1.86（3.8σ）。
            // 规则上限是**单步 5 秒**，300ms 留了 16 倍余量。
            // 想更快可以调小（CABO_AI_BUDGET_US=8000 时约损失 7 分/局）。
            budget_us: std::env::var("CABO_AI_BUDGET_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300_000),
            // 原上限 256 会在 50ms 预算下就先卡住（实测平均只用 55.77ms/步，时间没用满）。
            // 放开到 2048，让**时间**成为唯一约束。
            max_worlds: 2048,
            min_worlds: 3,
            max_cands: 10,
            policy: PolicyCfg::default(),
            value: ValueKind::WinProb,
            plain_policy: false,
            allow_cabo: true,
            cabo_margin: 4.0,
            cabo_min_success: 0.75,
            cabo_from_prior: false,
            use_memory: true,
            prior_from_policy: false,
            // rollout 里**我方座位**改用学习到的策略网络（DL_ROUTE.md E34）：
            // 让我方未来走法与被评估的策略一致。此前在 15~60ms 操作点上因算力惩罚失败，
            // 300ms 操作点上实测 ★ −11.56 ± 6.94、平均累计分差 −10.06 ± 1.37（7.3σ）。
            // 全部座位都用（rollout_net=2）效果更弱：★ −4.06 ± 5.03。
            rollout_net: 1,
            value_rollout_steps: 1000,
            value_k: 8.0,
            improve_margin: 0.04,
            improve_sigma: 0.0,
            expose_penalty: 0.0,
            shrink_noise: false,
            paired_gate: false,
            paired_t: 1.0,
            extend_budget_us: 0,
        }
    }
}

/// 搜索型 Bot。
pub struct SearchBot {
    pub cfg: SearchCfg,
    /// 公共历史记忆（像人一样记住弃牌堆里有什么）。
    pub history: super::history::History,
}

impl SearchBot {
    pub fn new(cfg: SearchCfg) -> Self {
        SearchBot { cfg, history: super::history::History::new() }
    }
}

impl SearchCfg {
    /// 把"点数加价"折算成当前回报口径下的加价。
    ///
    /// 胜率口径的回报落在 [0,1]，直接拿点数当加价会让阈值变成"永不宣告"；
    /// 这里统一按"当前分差处多领先 `cabo_margin` 点值多少胜率"来折算。
    fn cabo_margin_value(
        &self,
        my: f64,
        min_other: f64,
        second_other: f64,
        round_no: f64,
        target: f64,
        lead: f64,
    ) -> f64 {
        match self.value {
            // 分数上下文模型本身就是概率：直接用它在 (lead) 与 (lead + cabo_margin)
            // 两点上的概率差作为门槛，量纲自然一致、且随分数位置自动变化。
            ValueKind::ScoreCtx => {
                let p0 = crate::ai::value::score_ctx_prob_at(my, min_other, second_other, round_no, target);
                let p1 = crate::ai::value::score_ctx_prob_at(
                    my,
                    min_other + self.cabo_margin,
                    second_other,
                    round_no,
                    target,
                );
                (p1 - p0).max(0.0)
            }
            // `Learned` 输出的也是**概率**，所以门槛必须与 WinProb 同样做单位换算；
            // 否则 4.0 会被当成"概率差 4.0"，实际等价于永不宣告（历史混杂）。
            ValueKind::WinProb | ValueKind::Learned => {
                let k = self.value_k.max(1.0);
                let s = |x: f64| 1.0 / (1.0 + (-x / k).exp());
                (s(lead + self.cabo_margin) - s(lead)).max(0.0)
            }
            _ => self.cabo_margin,
        }
    }

    /// 调参用：`key=value` 覆盖；`policy.*` 前缀转给 rollout 策略。
    pub fn set(&mut self, key: &str, val: &str) -> bool {
        if let Some(rest) = key.strip_prefix("policy.") {
            return self.policy.set(rest, val);
        }
        let u = |v: &str| v.parse::<u64>().ok();
        match key {
            "budget_us" => u(val).map(|v| self.budget_us = v),
            "max_worlds" => u(val).map(|v| self.max_worlds = v as u32),
            "min_worlds" => u(val).map(|v| self.min_worlds = v as u32),
            "max_cands" => u(val).map(|v| self.max_cands = v as usize),
            "plain_policy" => match val {
                "1" | "true" => Some(self.plain_policy = true),
                "0" | "false" => Some(self.plain_policy = false),
                _ => None,
            },
            "value" => match val {
                "learned" => Some(self.value = ValueKind::Learned),
                "learned_margin" | "margin" => Some(self.value = ValueKind::LearnedMargin),
                "score_ctx" | "scorectx" => Some(self.value = ValueKind::ScoreCtx),
                "winprob" => Some(self.value = ValueKind::WinProb),
                "lead" => Some(self.value = ValueKind::Lead),
                "round" => Some(self.value = ValueKind::Round),
                _ => None,
            },
            "allow_cabo" => match val {
                "1" | "true" => Some(self.allow_cabo = true),
                "0" | "false" => Some(self.allow_cabo = false),
                _ => None,
            },
            "cabo_margin" => val.parse::<f64>().ok().map(|v| self.cabo_margin = v),
            "cabo_min_success" => val.parse::<f64>().ok().map(|v| self.cabo_min_success = v),
            "rollout_net" => val.parse::<u8>().ok().map(|v| self.rollout_net = v),
            "prior_from_policy" => match val {
                "1" | "true" => Some(self.prior_from_policy = true),
                "0" | "false" => Some(self.prior_from_policy = false),
                _ => None,
            },
            "use_memory" => match val {
                "1" | "true" => Some(self.use_memory = true),
                "0" | "false" => Some(self.use_memory = false),
                _ => None,
            },
            "cabo_from_prior" => match val {
                "1" | "true" => Some(self.cabo_from_prior = true),
                "0" | "false" => Some(self.cabo_from_prior = false),
                _ => None,
            },
            "value_k" => val.parse::<f64>().ok().map(|v| self.value_k = v),
            "value_rollout_steps" => {
                val.parse::<u64>().ok().map(|v| self.value_rollout_steps = v as u32)
            }
            "improve_margin" => val.parse::<f64>().ok().map(|v| self.improve_margin = v),
            "improve_sigma" => val.parse::<f64>().ok().map(|v| self.improve_sigma = v),
            "expose_penalty" => val.parse::<f64>().ok().map(|v| self.expose_penalty = v),
            "paired_gate" => match val {
                "1" | "true" => Some(self.paired_gate = true),
                "0" | "false" => Some(self.paired_gate = false),
                _ => None,
            },
            "paired_t" => val.parse::<f64>().ok().map(|v| self.paired_t = v),
            "extend_budget_us" => {
                val.parse::<u64>().ok().map(|v| self.extend_budget_us = v)
            }
            "shrink_noise" => match val {
                "1" | "true" => Some(self.shrink_noise = true),
                "0" | "false" => Some(self.shrink_noise = false),
                _ => None,
            },
            _ => None,
        }
        .is_some()
    }
}

impl super::Bot for SearchBot {
    fn id(&self) -> &'static str {
        "search"
    }
    fn name(&self) -> &'static str {
        "搜索 AI"
    }

    fn decide(&self, view: &PlayerView, rng: &mut dyn RngCore) -> Command {
        let t0 = Instant::now();
        let cmd = self.decide_inner(view, rng);
        stats::note_decide(t0.elapsed().as_nanos() as u64);
        cmd
    }
}

impl SearchBot {
    fn decide_inner(&self, view: &PlayerView, rng: &mut dyn RngCore) -> Command {
        // 开局查看：四张牌对称，看哪两张都等价。
        if let Panel::PeekPick { .. } = view.panel {
            return Command::PeekInitial { slots: [0, 1] };
        }
        // 先从公共历史里恢复弃牌堆构成（人类看得见的公开信息）。
        let info = self.history.observe(view);
        let discard_known = if self.cfg.use_memory { info.known } else { [0u16; 14] };
        let me = match view.me {
            Some(m) => m,
            None => return super::fallback_command(view),
        };
        // 重建不可用（非搜索面板 / 信息不自洽）时退回启发式。
        if self.cfg.plain_policy
            || !matches!(
                view.panel,
                Panel::Idle { .. } | Panel::Drew { .. } | Panel::SwapSelecting { .. } | Panel::ConfirmCabo
            )
        {
            return view_policy_with(view, &self.cfg.policy, rng, &discard_known);
        }

        let k = Know::from_view_with(view, &discard_known);
        self.decide_with_know(view, &k, &discard_known, rng)
    }

    /// 用外部给定的信念快照跑搜索（预言机诊断用；`k` 可以换成"全知"版本）。
    pub fn decide_with_know(
        &self,
        view: &PlayerView,
        k: &Know,
        discard_known: &[u16; 14],
        rng: &mut dyn RngCore,
    ) -> Command {
        // 启发式策略的选择：作为"先验候选"参与比较（见 `improve_margin`）。
        let me = match view.me {
            Some(m) => m,
            None => return super::fallback_command(view),
        };

        let prior = if self.cfg.prior_from_policy {
            crate::ai::policy::choose(view, k, discard_known, rng, &self.cfg.policy)
                .unwrap_or_else(|| view_policy_with(view, &self.cfg.policy, rng, discard_known))
        } else {
            view_policy_with(view, &self.cfg.policy, rng, discard_known)
        };
        let cands = if let Panel::ConfirmCabo = view.panel {
            vec![Command::CallCabo]
        } else {
            let mut c = candidates(view, k, self.cfg.max_cands);
            if !c.contains(&prior) {
                c.push(prior.clone());
            }
            if !self.cfg.allow_cabo {
                c.retain(|cmd| !matches!(cmd, Command::CallCabo));
            }
            // 宣告权交给标定过的启发式规则：搜索不得自行发起宣告。
            if self.cfg.cabo_from_prior && !matches!(prior, Command::CallCabo) {
                c.retain(|cmd| !matches!(cmd, Command::CallCabo));
            }
            c
        };
        if cands.is_empty() {
            return view_policy_with(view, &self.cfg.policy, rng, discard_known);
        }
        if cands.len() == 1 {
            return cands[0].clone();
        }

        let deadline = Instant::now() + std::time::Duration::from_micros(self.cfg.budget_us);
        let mut sums = vec![0f64; cands.len()];
        let mut sumsq = vec![0f64; cands.len()];
        let mut counts = vec![0u32; cands.len()];
        let mut valid = vec![true; cands.len()];
        // CallCabo 分支的"本轮得 0 分"次数（= 严格最低、宣告成功）。
        let mut cabo_ok = 0u32;
        // 与在位者的**配对差值**（同一世界内相减 → 公共随机数带来的共模噪声抵消）
        let prior_ci_for_diff = cands.iter().position(|c| *c == prior);
        let mut dsum = vec![0f64; cands.len()];
        let mut dsumsq = vec![0f64; cands.len()];
        let mut dcount = vec![0u32; cands.len()];
        let cabo_ci = cands.iter().position(|c| matches!(c, Command::CallCabo));
        let mut worlds = 0u32;

        // 定向延长：一旦发现"宣告"挤进前二名，就把截止时间放宽（其它决策不受影响）。
        let hard_deadline = deadline
            + std::time::Duration::from_micros(
                self.cfg.extend_budget_us.saturating_sub(self.cfg.budget_us),
            );
        let mut extended = false;
        while worlds < self.cfg.max_worlds
            && (worlds < self.cfg.min_worlds
                || Instant::now() < if extended { hard_deadline } else { deadline })
        {
            let Some(mut world) = reconstruct(view, rng, &discard_known) else { break };
            worlds += 1;
            let mut vals: Vec<Option<f64>> = vec![None; cands.len()];
            for (ci, cand) in cands.iter().enumerate() {
                if !valid[ci] {
                    continue;
                }
                let mut s = world.clone();
                stats::note_clone();
                if s.apply(me, cand).is_err() {
                    valid[ci] = false;
                    continue;
                }
                let v = evaluate_after(&mut s, me, &self.cfg, rng);
                sums[ci] += v;
                sumsq[ci] += v * v;
                counts[ci] += 1;
                vals[ci] = Some(v);
                if Some(ci) == cabo_ci && s.players[me].round_score == Some(0) {
                    cabo_ok += 1;
                }
            }
            // 配对差：同一世界内 (候选 − 在位者)
            if self.cfg.paired_gate {
                if let Some(pci) = prior_ci_for_diff {
                    if let Some(pv) = vals[pci] {
                        for ci in 0..cands.len() {
                            if let Some(v) = vals[ci] {
                                let d = v - pv;
                                dsum[ci] += d;
                                dsumsq[ci] += d * d;
                                dcount[ci] += 1;
                            }
                        }
                    }
                }
            }
            // 定向延长判定：CallCabo 是否在前二名
            if self.cfg.extend_budget_us > 0 && !extended {
                if let Some(cci) = cabo_ci {
                    if counts[cci] > 0 {
                        let cm = sums[cci] / counts[cci] as f64;
                        let mut better = 0usize;
                        for ci in 0..cands.len() {
                            if ci == cci || counts[ci] == 0 {
                                continue;
                            }
                            let m = sums[ci] / counts[ci] as f64;
                            if m > cm {
                                better += 1;
                            }
                        }
                        if better < 2 {
                            extended = true;
                            stats::note_extend();
                        }
                    }
                }
            }
            if worlds >= self.cfg.min_worlds && Instant::now() >= deadline && !extended {
                break;
            }
            world.log.clear();
        }
        let cabo_success =
            if cabo_ci.is_some() { cabo_ok as f64 / worlds.max(1) as f64 } else { 1.0 };

        // 选出平均回报最高的候选；样本为 0（重建失败）则退回启发式。
        let mut best: Option<(f64, usize)> = None;
        let mut best_other: Option<(f64, usize)> = None;
        let mut prior_val: Option<f64> = None;
        for ci in 0..cands.len() {
            if counts[ci] == 0 {
                continue;
            }
            let mut mean = sums[ci] / counts[ci] as f64;
            if self.cfg.shrink_noise {
                mean = shrink_prob_value(self.cfg.value, mean, sumsq[ci], counts[ci]);
            }
            if self.cfg.expose_penalty > 0.0 {
                let (tot, low) = exposed_cards(&cands[ci], &k);
                mean -= self.cfg.expose_penalty * (low as f64 + 0.25 * (tot - low) as f64);
            }
            if cands[ci] == prior {
                prior_val = Some(mean);
            }
            if best.is_none_or(|(bv, _)| mean > bv) {
                best = Some((mean, ci));
            }
            if !matches!(cands[ci], Command::CallCabo) && best_other.is_none_or(|(bv, _)| mean > bv) {
                best_other = Some((mean, ci));
            }
        }
        // Cabo 需要"加价"才值得冒 +10 惩罚的风险（见 `cabo_margin` 说明）。
        let lead_now = {
            let my = view.all_seats[me].total_score as f64;
            let min_other = view
                .all_seats
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != me)
                .map(|(_, s)| s.total_score as f64)
                .fold(f64::MAX, f64::min);
            min_other - my
        };
        let (my_t, min_other_t, second_other_t) = {
            let my = view.all_seats[me].total_score as f64;
            let mut others: Vec<f64> = view
                .all_seats
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != me)
                .map(|(_, s)| s.total_score as f64)
                .collect();
            others.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (my, others.first().copied().unwrap_or(0.0), others.get(1).copied().unwrap_or(0.0))
        };
        let margin = self.cfg.cabo_margin_value(
            my_t,
            min_other_t,
            second_other_t,
            view.round_no as f64,
            view.target_score as f64,
            lead_now,
        );
        if let (Some((bv, bi)), Some((ov, oi))) = (best, best_other) {
            let is_cabo = matches!(cands[bi], Command::CallCabo);
            if is_cabo && (bv < ov + margin || cabo_success < self.cfg.cabo_min_success) {
                best = Some((ov, oi));
            }
        }
        // 策略先验：搜索必须明显更好才偏离启发式策略，否则噪声会把连贯计划打散。
        //
        // **尺度无关的偏离规则**：`improve_margin` 是绝对阈值，只在"概率口径的启发式估值"下标定过。
        // 换成学习估值后其局部梯度更平，同一个 0.04 会让搜索几乎永不偏离先验、退化成启发式本身
        // （这正好解释了"换学习估值就掉 22~24 分"）。所以再加一条按候选值标准差归一化的规则。
        if self.cfg.paired_gate {
            // 配对显著性：只在"比在位者显著更好"的候选里挑最好的；都不显著就听在位者。
            if let (Some(pci), Some(_)) = (prior_ci_for_diff, prior_val) {
                let mut pick: Option<(f64, usize)> = None;
                for ci in 0..cands.len() {
                    if counts[ci] == 0 || dcount[ci] < 2 || ci == pci {
                        continue;
                    }
                    let n = dcount[ci] as f64;
                    let md = dsum[ci] / n;
                    let var = (dsumsq[ci] / n - md * md).max(0.0);
                    let se = (var / n).sqrt();
                    if md > self.cfg.paired_t * se {
                        let mean = sums[ci] / counts[ci] as f64;
                        if pick.is_none_or(|(bv, _)| mean > bv) {
                            pick = Some((mean, ci));
                        }
                    }
                }
                best = match pick {
                    Some((v, ci)) => Some((v, ci)),
                    None => Some((prior_val.unwrap_or(0.0), pci)),
                };
            }
        } else if let (Some((bv, _)), Some(pv)) = (best, prior_val) {
            let margin = if self.cfg.improve_sigma > 0.0 {
                let (mut sum, mut sq, mut n) = (0.0f64, 0.0f64, 0.0f64);
                for ci in 0..cands.len() {
                    if counts[ci] == 0 {
                        continue;
                    }
                    let m = sums[ci] / counts[ci] as f64;
                    sum += m;
                    sq += m * m;
                    n += 1.0;
                }
                let sd = if n > 1.0 {
                    ((sq / n) - (sum / n) * (sum / n)).max(0.0).sqrt()
                } else {
                    0.0
                };
                self.cfg.improve_sigma * sd
            } else {
                0.0
            };
            if bv < pv + margin + self.cfg.improve_margin {
                best = Some((pv, cands.iter().position(|c| *c == prior).unwrap_or(0)));
            }
        }
        if std::env::var_os("CABO_AI_DEBUG").is_some() {
            let mut report: Vec<String> = Vec::new();
            for ci in 0..cands.len() {
                let mean = if counts[ci] > 0 { sums[ci] / counts[ci] as f64 } else { f64::NAN };
                let mark = if Some(ci) == best.map(|(_, i)| i) { "*" } else { " " };
                report.push(format!("{mark}{:?}=>{mean:.2}(n={})", cands[ci], counts[ci]));
            }
            eprintln!(
                "[search P{me} r{} 世界{} Cabo成功率{:.2} 总分{:?}] 手牌{:?} est={:.1} 对手est={:?} 弃牌顶{:?} 牌堆{} | {}",
                view.round_no,
                worlds,
                cabo_success,
                view.all_seats.iter().map(|s| s.total_score).collect::<Vec<_>>(),
                k.hands[me],
                k.est(me).0,
                (0..k.n).map(|j| (k.est(j).0 * 10.0).round() / 10.0).collect::<Vec<_>>(),
                k.discard_top,
                view.deck_count,
                report.join("  ")
            );
        }
        match best {
            Some((_, ci)) => cands[ci].clone(),
            None => view_policy_with(view, &self.cfg.policy, rng, &discard_known),
        }
    }
}


/// `E[sigma(x)]` 的 probit 近似：`x ~ N(mean, se^2)` -> `sigma( mean / sqrt(1 + pi*se^2/8) )`。
///
/// 把蒙特卡洛噪声从候选价值里挤出去：噪声越大，估值越向 0.5 收缩。
fn shrink_prob_value(value_kind: ValueKind, mean: f64, sumsq: f64, count: u32) -> f64 {
    if count < 2 {
        return mean;
    }
    let p = mean.clamp(1e-6, 1.0 - 1e-6);
    let logit_mean = match value_kind {
        ValueKind::WinProb | ValueKind::Learned | ValueKind::ScoreCtx => (p / (1.0 - p)).ln(),
        _ => return mean,
    };
    let n = count as f64;
    let var = (sumsq / n - mean * mean).max(0.0);
    let se = (var / n).sqrt();
    // 概率的标准误 -> logit 尺度上的标准误：dp/dlogit = p(1-p)
    let se_logit = se / (p * (1.0 - p)).max(1e-6);
    let shrunk = logit_mean / (1.0 + std::f64::consts::PI * se_logit * se_logit / 8.0).sqrt();
    1.0 / (1.0 + (-shrunk).exp())
}

/// rollout 内部的行棋策略：可选改用学习到的策略网络（见 `SearchCfg::rollout_net`）。
///
/// 网络调用成本实测 ≈ 8µs/次（project 1.9 + Know 0.4 + 候选 0.74 + 特征 0.33 + 前向 5），
/// 手写启发式约 2~5µs/次 —— 所以"全部座位都用网络"会让一次 rollout 慢 3~4 倍，
/// 换来的是更准的价值估计。哪种更值，用擂台配对实验回答。
fn rollout_policy(
    s: &Session,
    pid: PlayerId,
    cfg: &SearchCfg,
    rng: &mut dyn RngCore,
    me: PlayerId,
    step: usize,
) -> Command {
    let use_net = match cfg.rollout_net {
        0 => false,
        1 => pid == me,
        2 => true,
        3 => step == 0,
        _ => false,
    };
    if use_net {
        if let Some(c) = crate::ai::policy::choose_from_session(s, pid, rng, &cfg.policy) {
            return c;
        }
    }
    policy(s, pid, &cfg.policy, rng)
}


/// 该候选动作会**公开**几张自己已知的牌，以及其中"低牌"（≤5）的张数。
///
/// 换入弃牌堆顶 = 把自己原来那张翻到弃牌堆，对手全看得见；低牌被看见最亏
/// （等于告诉所有人"我手里还有更好的"）。弃置摸到的牌只暴露"我不要它"，
/// 能力牌（偷看/窥探/交换）不暴露自己的手牌。
fn exposed_cards(cand: &Command, k: &Know) -> (usize, usize) {
    let me = k.me;
    let known = |slot: &u8| -> Option<u8> { k.hands[me].get(*slot as usize).and_then(|c| *c) };
    let slots: Vec<u8> = match cand {
        Command::SwapOnce { slots } => slots.clone(),
        _ => Vec::new(),
    };
    let mut total = 0usize;
    let mut low = 0usize;
    for s in &slots {
        total += 1;
        if matches!(known(s), Some(r) if r <= 5) {
            low += 1;
        }
    }
    (total, low)
}

/// 从完整局面出发，把本轮打完，返回回报值。
fn rollout_value(s: &mut Session, me: PlayerId, cfg: &SearchCfg, rng: &mut dyn RngCore) -> f64 {
    let mut guard = 0usize;
    loop {
        let pid = match &s.phase {
            Phase::Peeking { done } => (0..s.players.len()).find(|p| !done.contains_key(p)),
            Phase::Turn { current, .. } => Some(*current),
            _ => None,
        };
        let Some(pid) = pid else { break };
        let cmd = rollout_policy(s, pid, cfg, rng, me, guard);
        if s.apply(pid, &cmd).is_err() {
            let fb = raw_fallback(s, pid);
            if s.apply(pid, &fb).is_err() {
                break;
            }
        }
        guard += 1;
        if guard > MAX_ROLLOUT_ACTIONS {
            break;
        }
    }
    stats::note_rollout(guard as u64);
    value_of(s, me, cfg)
}

/// 候选动作生效后：先按 `value_rollout_steps` 走几步（0 步 = 立刻估值），再折算回报。
fn evaluate_after(s: &mut Session, me: PlayerId, cfg: &SearchCfg, rng: &mut dyn RngCore) -> f64 {
    // `value_rollout_steps >= 1000` 视为"一直打到本轮结算"（默认行为）
    let limit = if cfg.value_rollout_steps >= 1000 {
        MAX_ROLLOUT_ACTIONS
    } else {
        cfg.value_rollout_steps as usize
    };
    let mut steps = 0usize;
    while steps < limit {
        let pid = match &s.phase {
            Phase::Peeking { done } => (0..s.players.len()).find(|p| !done.contains_key(p)),
            Phase::Turn { current, .. } => Some(*current),
            _ => None,
        };
        let Some(pid) = pid else { break };
        let cmd = rollout_policy(s, pid, cfg, rng, me, steps);
        if s.apply(pid, &cmd).is_err() {
            let fb = raw_fallback(s, pid);
            if s.apply(pid, &fb).is_err() {
                break;
            }
        }
        steps += 1;
    }
    stats::note_rollout(steps as u64);
    value_of(s, me, cfg)
}

/// 回报：把"打到本轮结束后的局面"折算成一个统一的胜率估计。
///
/// 量纲自洽非常重要：早先的实现把"终局胜负"记成 ±1000、把"分差"记成 ±20，
/// 于是搜索会为了避免"游戏在这一轮结束"而做出荒谬的选择（例如提早宣告 Cabo
/// 把回合掐断），实测让对战成绩大幅退化。这里统一成胜率（0~1）。
fn value_of(s: &Session, me: PlayerId, cfg: &SearchCfg) -> f64 {
    let my = s.players[me].total_score as f64;
    let min_other = s
        .players
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != me)
        .map(|(_, p)| p.total_score as f64)
        .fold(f64::MAX, f64::min);
    let lead = min_other - my;
    if let Phase::GameOver { winners } = &s.phase {
        return if winners.contains(&me) { 1.0 } else { 0.0 };
    }
    match cfg.value {
        ValueKind::Learned => {
            // 学习价值：直接估 P(赢下整局)；权重缺失时退回启发式胜率口径。
            crate::ai::value::evaluate(s, me).unwrap_or_else(|| win_prob_from_lead(lead, cfg))
        }
        ValueKind::LearnedMargin => {
            // 点数口径：解析基线（当前领先量）+ 网络对"终局分差"的修正。
            lead + crate::ai::value::correction(s, me).unwrap_or(0.0)
        }
        ValueKind::WinProb => win_prob_from_lead(lead, cfg),
        ValueKind::ScoreCtx => crate::ai::value::score_ctx_prob(s, me),
        ValueKind::Lead => lead,
        ValueKind::Round => -my,
    }
}

fn win_prob_from_lead(lead: f64, cfg: &SearchCfg) -> f64 {
    let k = cfg.value_k.max(1.0);
    1.0 / (1.0 + (-lead / k).exp())
}

// ---------------------------------------------------------------- 候选动作

/// 枚举当前状态下值得搜索的命令。`max_cands` 为上限（按静态评分裁剪）。
pub fn candidates(view: &PlayerView, k: &Know, max_cands: usize) -> Vec<Command> {
    candidates_scored(view, k, max_cands).into_iter().map(|(_, c)| c).collect()
}

/// 候选生成的实际实现（返回 (静态评分, 命令)，已按类别保底裁剪）。
fn candidates_impl(view: &PlayerView, k: &Know, max_cands: usize) -> Vec<(f64, Command)> {
    let me = k.me;
    let mut out: Vec<(f64, Command)> = Vec::new();
    match &view.panel {
        Panel::Idle { can_draw, can_swap_discard, can_cabo } => {
            if *can_cabo {                // 宣告 Cabo：用"严格最低"的概率粗略打分（细节交给搜索评估）。
                let p = k.p_strict_lowest();
                let (em, _) = k.est(me);
                out.push((p * 6.0 - (1.0 - p) * (em * 0.3 + k.penalty as f64), Command::CallCabo));
            }
            if *can_draw {
                out.push((0.0, Command::BeginDraw));
            }
            if *can_swap_discard {
                if let Some(top) = k.discard_top {
                    let topf = top as f64;
                    // 单张换。
                    for (slot, val) in k.known_slots(me) {
                        let gain = val as f64 - topf;
                        out.push((gain, Command::SwapOnce { slots: vec![slot] }));
                    }
                    // 多张同点：手牌变少，额外奖励。
                    for (rank, slots) in k.groups(me) {
                        let gain = slots.len() as f64 * rank as f64 - topf
                            + 2.0 * (slots.len() as f64 - 1.0);
                        out.push((gain, Command::SwapOnce { slots }));
                    }
                    // 未知牌换顶：期望收益 + 情报价值。
                    for slot in k.unknown_slots(me) {
                        out.push((k.mean_me - topf + 0.5, Command::SwapOnce { slots: vec![slot] }));
                    }
                }
            }
        }
        Panel::Drew { rank, .. } => {
            let drawn = *rank;
            let drawnf = drawn as f64;
            out.push((-2.0, Command::DiscardDrawn { power: None }));
            // 单张换入。
            for (slot, val) in k.known_slots(me) {
                out.push((val as f64 - drawnf, Command::DrawSwap { slots: vec![slot] }));
            }
            // 未知牌换入。
            for slot in k.unknown_slots(me) {
                out.push((k.mean_me - drawnf + 0.5, Command::DrawSwap { slots: vec![slot] }));
            }
            // 同点整组换入：手牌变少。
            let same: Vec<SlotId> = k
                .known_slots(me)
                .into_iter()
                .filter(|(_, r)| *r == drawn)
                .map(|(s, _)| s)
                .collect();
            if same.len() >= 2 {
                let gain = same.len() as f64 * drawnf - drawnf + 2.0 * (same.len() as f64 - 1.0);
                out.push((gain, Command::DrawSwap { slots: same }));
            }
            // 能力牌。
            if crate::game::power_kind_of(drawn).is_some() {
                match drawn {
                    7..=8 => {
                        for slot in k.unknown_slots(me) {
                            out.push((
                                1.5,
                                Command::DiscardDrawn { power: Some(PowerUse::PeekOwn { slot }) },
                            ));
                        }
                    }
                    9..=10 => {
                        for j in 0..k.n {
                            if j == me {
                                continue;
                            }
                            for slot in k.unknown_slots(j) {
                                out.push((
                                    1.2,
                                    Command::DiscardDrawn {
                                        power: Some(PowerUse::Spy { player: j, slot }),
                                    },
                                ));
                            }
                        }
                    }
                    11..=12 => {
                        for (my_slot, val) in k.known_slots(me) {
                            for j in 0..k.n {
                                if j == me {
                                    continue;
                                }
                                for slot in k.unknown_slots(j) {
                                    // 把高点数的牌盲换出去，换回一张未知牌。
                                    let gain = val as f64 - k.mean_opp;
                                    out.push((
                                        gain + 1.0,
                                        Command::DiscardDrawn {
                                            power: Some(PowerUse::Swap { my_slot, player: j, slot }),
                                        },
                                    ));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Panel::ConfirmCabo => out.push((0.0, Command::CallCabo)),
        Panel::SwapSelecting { count, source, .. } => {
            if *count >= 1 {
                out.push((0.0, Command::SwapCommit));
            } else {
                for slot in 0..k.hand_size(me) as SlotId {
                    out.push((0.0, Command::SwapToggle { slot }));
                }
                let _ = source;
            }
        }
        _ => {}
    }

    // 裁剪：**按类别保底**后再用剩余评分补足。
    //
    // 否则"11/12 交换能力"这种组合爆炸的候选（我每张已知牌 × 每个对手的每张未知牌，
    // 动辄上百个）会靠静态评分把 max_cands 全部占满，连"直接弃置""单张换入"
    // 这类基本动作都进不了搜索——实测会造成严重退化。
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut picked: Vec<(f64, Command)> = Vec::with_capacity(max_cands);
    let push = |score: f64, cmd: Command, picked: &mut Vec<(f64, Command)>| {
        if picked.len() < max_cands && !picked.iter().any(|(_, c)| *c == cmd) {
            picked.push((score, cmd));
        }
    };
    // 第 1 轮：每个"类别"最多收 1 个（类别 = 命令种类 + 能力种类）。
    let mut seen: Vec<u8> = Vec::new();
    for (s, c) in &out {
        let key = category_key(c);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        push(*s, c.clone(), &mut picked);
    }
    // 第 2 轮：按评分补足，但能力类候选最多再补 3 个。
    let mut power_added = 0usize;
    for (s, c) in &out {
        let is_power = matches!(c, Command::DiscardDrawn { power: Some(_) });
        if is_power && power_added >= 3 {
            continue;
        }
        if is_power {
            power_added += 1;
        }
        push(*s, c.clone(), &mut picked);
    }
    picked
}

/// 与 [`candidates`] 相同，但保留每个候选的**静态评分**（启发式自己的排序信号）。
/// 训练策略网络时把它作为候选特征喂进去，网络就只需要学"相对这个信号的修正量"。
pub fn candidates_scored(
    view: &PlayerView,
    k: &Know,
    max_cands: usize,
) -> Vec<(f64, Command)> {
    let mut v = candidates_impl(view, k, max_cands);
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// 候选类别：用于"每类至少一个"的裁剪策略。
///
/// 返回 `u8` 而不是 `String`：候选生成在 **rollout 里每步都要跑**，
/// 每个候选一次字符串分配会直接吃掉时间预算。
fn category_key(c: &Command) -> u8 {
    match c {
        Command::CallCabo => 0,
        Command::BeginDraw => 1,
        Command::DiscardDrawn { power: None } => 2,
        Command::DiscardDrawn { power: Some(u) } => match power_tag(u) {
            "peek" => 3,
            "spy" => 4,
            _ => 5,
        },
        Command::DrawSwap { slots } => {
            if slots.len() > 1 {
                6
            } else {
                7
            }
        }
        Command::SwapOnce { slots } => {
            if slots.len() > 1 {
                8
            } else {
                9
            }
        }
        Command::SwapToggle { .. } => 10,
        Command::SwapCommit => 11,
        Command::Cancel => 12,
        _ => 13,
    }
}

fn power_tag(u: &PowerUse) -> &'static str {
    match u {
        PowerUse::PeekOwn { .. } => "peek",
        PowerUse::Spy { .. } => "spy",
        PowerUse::Swap { .. } => "swap",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::BotRegistry;
    use crate::game::sim::{make_ai_session, run_game};
    use crate::game::Settings;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// 搜索 Bot 必须能完整打完对局，且每一步都合法。
    #[test]
    fn search_bot_plays_legal_games() {
        let mut reg = BotRegistry::new();
        reg.register(std::sync::Arc::new(SearchBot::new(SearchCfg { budget_us: 2_000, max_worlds: 6, min_worlds: 2, ..SearchCfg::default() })));
        reg.register(std::sync::Arc::new(crate::ai::simple::SimpleBot));
        for seed in [1u64, 7, 99] {
            let mut s = make_ai_session(seed, Settings::default(), &["search", "simple", "search"]);
            let mut rng = StdRng::seed_from_u64(seed);
            let o = run_game(&mut s, &reg, &mut rng, 200_000).expect("对局应当正常结束");
            assert_eq!(o.totals.len(), 3);
        }
    }
}
