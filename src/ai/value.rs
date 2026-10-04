//! 学习到的价值函数：`v(局面, 我) ≈ P(我最终赢下这一整局)`。
//!
//! 这是"深度强化学习 + 搜索"里的**学习部分**：用自我对弈产出的
//! `(局面特征, 最终胜负)` 样本训练一个小型 MLP（见 [`crate::ai::nn`]），
//! 再把搜索的叶子估值换成它。权重由历史备份分支中的 `cabo-train` 训练后写成
//! `src/ai/weights.rs` 嵌进二进制（保持"单文件部署、零运行时依赖"）。
//!
//! 特征是从**完美信息局面**上取的（`Session` 是确定性化的世界）：
//! 这在确定性化搜索里是自洽的——每个世界被当作真实世界估值，再对世界求期望。

use std::sync::OnceLock;

use crate::game::{Phase, PlayerId, Session};

use super::nn::Mlp;

/// 特征维度（改特征必须同步改历史训练工具 `cabo-train`）。
pub const N_FEATURES: usize = 64;

/// 学习到的价值网络。
pub struct ValueNet {
    pub net: Mlp,
}

static NET: OnceLock<Option<ValueNet>> = OnceLock::new();

/// 特征里"领先量"（已除以 50）所在的下标；残差价值要用它作为解析基线。
pub const FEAT_LEAD_IDX: usize = 33;

/// 解析基线：把"与最强劲对手的分差"折算成 logit（和启发式口径一致，斜率 8 分）。
pub fn baseline_logit(x: &[f32]) -> f32 {
    x[FEAT_LEAD_IDX] * 50.0 / 8.0
}

/// **残差价值**：`p = sigmoid(解析基线 + 网络输出)`。
///
/// 为什么不让网络直接输出概率：搜索比较的候选之间只差几个点数，
/// 局面几乎一样；纯回归的网络在这类"邻域"上输出很平，
/// argmax 就变成噪声（实测：直接学 P(赢) 的网络 Brier 0.083、
/// 比启发式好 3 倍，但用来做搜索估值反而输 30 分）。
/// 让网络只学"相对解析基线的修正量"，主信号（分差）不会被抹平，
/// 网络只贡献它真正多出来的信息（手牌构成、剩余牌、轮次等）。
pub fn compose(net: &Mlp, x: &[f32]) -> f32 {
    let z = net.predict(x) + baseline_logit(x);
    1.0 / (1.0 + (-z).exp())
}

/// 载入嵌入的权重（训练产物 `weights.rs`；为空表示还没训练，退回启发式）。
pub fn net() -> Option<&'static ValueNet> {
    NET.get_or_init(|| {
        let sizes = super::weights::VALUE_SIZES;
        let data = super::weights::VALUE_WEIGHTS;
        if data.is_empty() || sizes.len() < 2 {
            return None;
        }
        Mlp::import(data).map(|net| ValueNet { net })
    })
    .as_ref()
}

/// 取特征：从 `me` 的视角（完美信息）描述局面。
pub fn features(s: &Session, me: PlayerId) -> Vec<f32> {
    let mut f = vec![0f32; N_FEATURES];
    let n = s.players.len();
    let mut k = 0usize;
    let mut push = |f: &mut Vec<f32>, k: &mut usize, v: f32| {
        if *k < f.len() {
            f[*k] = v;
        }
        *k += 1;
    };

    // 我的手牌：张数、点数和、点数直方图
    let my_hand: Vec<u8> = s.players[me].slots.iter().map(|c| s.cards[*c as usize].card.rank).collect();
    push(&mut f, &mut k, my_hand.len() as f32 / 8.0);
    push(&mut f, &mut k, my_hand.iter().map(|&r| r as f32).sum::<f32>() / 40.0);
    let mut hist = [0f32; 14];
    for &r in &my_hand {
        hist[r as usize] += 1.0;
    }
    for h in hist.iter() {
        push(&mut f, &mut k, h / 4.0);
    }

    // 分数：我的、以及"由强到弱"排序后的对手（顺序无关，避免网络学座位号）
    let my_total = s.players[me].total_score as f32;
    push(&mut f, &mut k, my_total / 100.0);
    let mut opp: Vec<(u32, f32, f32)> = (0..n)
        .filter(|&j| j != me)
        .map(|j| {
            let hand: Vec<u8> =
                s.players[j].slots.iter().map(|c| s.cards[*c as usize].card.rank).collect();
            (
                s.players[j].total_score,
                hand.len() as f32 / 8.0,
                hand.iter().map(|&r| r as f32).sum::<f32>() / 40.0,
            )
        })
        .collect();
    opp.sort_by_key(|(t, _, _)| *t);
    for i in 0..3 {
        let (t, cnt, sum) = opp.get(i).copied().unwrap_or((0, 0.0, 0.0));
        push(&mut f, &mut k, t as f32 / 100.0);
        push(&mut f, &mut k, cnt);
        push(&mut f, &mut k, sum);
    }
    // 领先量（对手最强者的累计分 − 我）
    let min_other = opp.first().map(|(t, _, _)| *t).unwrap_or(0) as f32;
    push(&mut f, &mut k, (min_other - my_total) / 50.0);

    // 桌面
    push(&mut f, &mut k, s.deck.len() as f32 / 52.0);
    push(&mut f, &mut k, s.discard.len() as f32 / 52.0);
    let top = s.discard.last().map(|c| s.cards[*c as usize].card.rank).unwrap_or(0);
    push(&mut f, &mut k, top as f32 / 13.0);
    push(&mut f, &mut k, s.round_no as f32 / 20.0);
    push(&mut f, &mut k, s.settings.target_score as f32 / 100.0);
    push(&mut f, &mut k, s.settings.cabo_penalty as f32 / 10.0);
    push(&mut f, &mut k, if s.cabo_caller.is_some() { 1.0 } else { 0.0 });
    push(&mut f, &mut k, s.extra_turns.len() as f32 / 4.0);
    push(&mut f, &mut k, if s.cabo_caller == Some(me) { 1.0 } else { 0.0 });
    push(&mut f, &mut k, if matches!(s.phase, Phase::GameOver { .. }) { 1.0 } else { 0.0 });
    // 我的手牌里最大的那张（便于网络判断"我还有没有大牌"）
    let worst = my_hand.iter().copied().max().unwrap_or(0) as f32;
    push(&mut f, &mut k, worst / 13.0);
    push(&mut f, &mut k, 1.0); // 偏置位（方便网络）
    debug_assert!(k <= N_FEATURES, "特征维度超出 {N_FEATURES}：{k}");
    f.truncate(N_FEATURES);
    f
}

/// 估值成功时返回 P(赢)，未训练时返回 `None`（调用方退回启发式）。
pub fn evaluate(s: &Session, me: PlayerId) -> Option<f64> {
    let net = net()?;
    let x = features(s, me);
    Some(compose(&net.net, &x) as f64)
}

/// 是否已加载训练权重。
pub fn available() -> bool {
    net().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::sim::make_ai_session;
    use crate::game::Settings;

    #[test]
    fn features_are_bounded_and_stable() {
        let mut s = make_ai_session(3, Settings::default(), &["challenger", "challenger", "challenger"]);
        s.start_game().unwrap();
        let f = features(&s, 1);
        assert_eq!(f.len(), N_FEATURES);
        assert!(f.iter().all(|v| v.is_finite()));
        assert!(f.iter().all(|v| v.abs() <= 20.0), "特征量纲异常");
    }
}

// ---------------------------------------------------------------- 完整局面特征（v2）

/// 完整局面特征维度（价值网络用；含所有座位的真实手牌构成）。
pub const N_FULL: usize = 150;

fn pf(f: &mut [f32], k: &mut usize, v: f32) {
    if *k < f.len() {
        f[*k] = v;
    }
    *k += 1;
}

/// 某座位手牌的点数重数（归一化到该点数总张数）。
fn mult(f: &mut [f32], k: &mut usize, s: &Session, pid: PlayerId) {
    let mut c = [0u16; 14];
    for cid in &s.players[pid].slots {
        c[s.cards[*cid as usize].card.rank as usize] += 1;
    }
    for r in 0..14 {
        pf(f, k, c[r] as f32 / crate::game::RANK_COPIES[r] as f32);
    }
}

/// **完整局面**特征：所有座位的真实手牌 + 未见池 + 弃牌堆 + 分数与轮次上下文 + 暴露度。
///
/// `lead_now` 由调用方另行给出（作为解析基线），这里不再重复。
pub fn features_full(s: &Session, me: PlayerId) -> Vec<f32> {
    let n = s.players.len();
    let mut f = vec![0f32; N_FULL];
    let mut i = 0usize;

    // 我：重数 + 聚合
    mult(&mut f, &mut i, s, me);
    let mine: Vec<u8> = s.players[me]
        .slots
        .iter()
        .map(|c| s.cards[*c as usize].card.rank)
        .collect();
    pf(&mut f, &mut i, mine.len() as f32 / 8.0);
    pf(&mut f, &mut i, mine.iter().map(|&r| r as f32).sum::<f32>() / 40.0);
    pf(&mut f, &mut i, s.players[me].total_score as f32 / 100.0);
    pf(&mut f, &mut i, (s.settings.target_score as i32 - s.players[me].total_score as i32) as f32 / 100.0);

    // 对手（按当前位置从好到差排序，最多 3 个）：重数 + 聚合
    let mut opp: Vec<PlayerId> = (0..n).filter(|&j| j != me).collect();
    opp.sort_by_key(|&j| s.players[j].total_score);
    for idx in 0..3 {
        match opp.get(idx) {
            Some(&j) => {
                mult(&mut f, &mut i, s, j);
                let cards: Vec<u8> = s.players[j]
                    .slots
                    .iter()
                    .map(|c| s.cards[*c as usize].card.rank)
                    .collect();
                pf(&mut f, &mut i, cards.len() as f32 / 8.0);
                pf(&mut f, &mut i, cards.iter().map(|&r| r as f32).sum::<f32>() / 40.0);
                pf(&mut f, &mut i, s.players[j].total_score as f32 / 100.0);
                pf(&mut f, &mut i, (s.players[j].total_score as i32 - s.players[me].total_score as i32) as f32 / 100.0);
            }
            None => {
                for _ in 0..18 {
                    pf(&mut f, &mut i, 0.0);
                }
            }
        }
    }

    // 未见池（整副牌 − 弃牌堆 − 所有已知手牌）
    let mut pool = [0u16; 14];
    for r in 0..14 {
        pool[r] = crate::game::RANK_COPIES[r] as u16;
    }
    for cid in &s.discard {
        pool[s.cards[*cid as usize].card.rank as usize] -= 1;
    }
    for p in 0..n {
        for cid in &s.players[p].slots {
            pool[s.cards[*cid as usize].card.rank as usize] -= 1;
        }
    }
    for r in 0..14 {
        pf(&mut f, &mut i, pool[r] as f32 / crate::game::RANK_COPIES[r] as f32);
    }
    // 弃牌堆构成
    let mut disc = [0u16; 14];
    for cid in &s.discard {
        disc[s.cards[*cid as usize].card.rank as usize] += 1;
    }
    for r in 0..14 {
        pf(&mut f, &mut i, disc[r] as f32 / crate::game::RANK_COPIES[r] as f32);
    }

    // 桌面与上下文
    pf(&mut f, &mut i, s.deck.len() as f32 / 52.0);
    pf(&mut f, &mut i, s.discard.len() as f32 / 52.0);
    let top = s.discard.last().map(|c| s.cards[*c as usize].card.rank);
    for r in 0..14 {
        pf(&mut f, &mut i, if top == Some(r as u8) { 1.0 } else { 0.0 });
    }
    pf(&mut f, &mut i, top.unwrap_or(0) as f32 / 13.0);
    pf(&mut f, &mut i, s.round_no as f32 / 20.0);
    pf(&mut f, &mut i, s.settings.target_score as f32 / 100.0);
    pf(&mut f, &mut i, s.settings.cabo_penalty as f32 / 10.0);
    pf(&mut f, &mut i, if s.cabo_caller.is_some() { 1.0 } else { 0.0 });
    pf(&mut f, &mut i, if s.cabo_caller == Some(me) { 1.0 } else { 0.0 });
    pf(&mut f, &mut i, s.extra_turns.len() as f32 / 4.0);
    // 暴露度：我有几张牌已被别人看过、平均被几个人看过
    let mut exposed = 0usize;
    let mut knowers = 0usize;
    for cid in &s.players[me].slots {
        let cs = &s.cards[*cid as usize];
        let others = cs.known_by.iter().filter(|&&p| p != me).count();
        if others > 0 {
            exposed += 1;
        }
        knowers += others;
    }
    pf(&mut f, &mut i, exposed as f32 / 8.0);
    pf(&mut f, &mut i, knowers as f32 / 12.0);
    pf(&mut f, &mut i, 1.0); // 偏置
    f
}

// ---------------------------------------------------------------- 价值 v3 推理

static NET2: std::sync::OnceLock<Option<Mlp>> = std::sync::OnceLock::new();

fn net2() -> Option<&'static Mlp> {
    NET2.get_or_init(|| {
        let data = super::weights::VALUE2_WEIGHTS;
        if data.is_empty() || super::weights::VALUE2_SIZES.len() < 2 {
            return None;
        }
        Mlp::import(data)
    })
    .as_ref()
}

/// 价值 v3 是否可用。
pub fn margin_available() -> bool {
    net2().is_some()
}

/// 终局分差的**修正量**（点数）：预测 = `lead_now` + 本函数。
pub fn correction(s: &Session, me: PlayerId) -> Option<f64> {
    let net = net2()?;
    let x = features_full(s, me);
    Some(net.predict(&x) as f64)
}

// ---------------------------------------------------------------- 分数上下文胜率模型

/// 逻辑回归系数（顺序与 `score_ctx_features` 一致）。
/// 历史备份分支的训练方式：`cargo run --release --bin cabo-train -- --mode scorectx` +
/// `python tools/fit_scorectx.py`（43.9 万样本，验证 logloss 0.4090）。
pub const SCORECTX_COEF: [f64; 9] = [
    -0.5585, 1.4896, 0.5647, -0.3873, 0.1069, 2.4141, -1.9079, 0.8216, -0.6680,
];

/// 分数上下文特征（与 `tools/fit_scorectx.py::feats` 严格一致）。
fn score_ctx_features(
    my: f64,
    min_other: f64,
    second_other: f64,
    round_no: f64,
    target: f64,
) -> [f64; 9] {
    let lead = min_other - my;
    let lead2 = second_other - my;
    let head = (target - my).max(1.0);
    [
        1.0,
        lead / 10.0,
        lead2 / 10.0,
        head / 100.0,
        round_no / 10.0,
        (lead / head).clamp(-3.0, 3.0),
        (lead / 10.0) * (head / 100.0),
        lead.max(0.0) / 10.0,
        (-lead).max(0.0) / 10.0,
    ]
}

pub fn score_ctx_prob_at(my: f64, min_other: f64, second_other: f64, round_no: f64, target: f64) -> f64 {
    let f = score_ctx_features(my, min_other, second_other, round_no, target);
    let z: f64 = f.iter().zip(SCORECTX_COEF.iter()).map(|(a, b)| a * b).sum();
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

/// P(最终获胜 | 分数上下文)：只看公开的分数与轮次，不看手牌。
pub fn score_ctx_prob(s: &Session, me: PlayerId) -> f64 {
    let target = s.settings.target_score as f64;
    let my = s.players[me].total_score as f64;
    let mut others: Vec<f64> =
        s.players.iter().enumerate().filter(|(i, _)| *i != me).map(|(_, p)| p.total_score as f64).collect();
    others.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_other = others.first().copied().unwrap_or(0.0);
    let second_other = others.get(1).copied().unwrap_or(0.0);
    score_ctx_prob_at(my, min_other, second_other, s.round_no as f64, target)
}

/// 领先 `lead` 分时的胜率（给 Cabo 门槛做单位换算用；其它座位的分数近似不变）。
pub fn score_ctx_prob_with_lead(s: &Session, me: PlayerId, lead: f64) -> f64 {
    let target = s.settings.target_score as f64;
    let my = s.players[me].total_score as f64;
    let mut others: Vec<f64> =
        s.players.iter().enumerate().filter(|(i, _)| *i != me).map(|(_, p)| p.total_score as f64).collect();
    others.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_other = others.first().copied().unwrap_or(0.0) + lead;
    let second_other = others.get(1).copied().unwrap_or(0.0);
    score_ctx_prob_at(my, min_other, second_other, s.round_no as f64, target)
}
