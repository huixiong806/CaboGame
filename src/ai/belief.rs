//! 信念状态（belief state）：从「玩家视图」重建一份自洽的完整局面。
//!
//! PIMC（Perfect Information Monte Carlo）的核心步骤是"确定性化"：
//! 把机器人看不到的牌，按剩余牌池均匀随机地指派到各个未知位置，
//! 得到一份完整 `Session`，就可以用真实规则引擎做前瞻模拟。
//!
//! 关键约束：这里**只使用 `PlayerView` 里已有的信息**（该座位有权知道的公共知识
//! + 自己的私有知识），未知牌一律从剩余牌池随机抽样，绝不作弊。

use std::collections::BTreeSet;

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

use crate::game::view::{Panel, PlayerView};
use crate::game::{
    Card, CardId, CardState, Pending, Phase, Pile, PlayerId, PlayerState, Session, Settings, SlotId,
    RANK_COPIES,
};

/// 一副牌的物理张数。
const TOTAL_CARDS: usize = 52;

/// 由 `view` 随机重建一份完整局面；信息不自洽（牌数/点数对不上）时返回 `None`。
///
/// `known_discard` 是公共历史恢复出来的弃牌堆构成（不含堆顶）：这些牌是**公开信息**，
/// 直接按已知点数摆进弃牌堆，不再参与抽样 —— 机器人因此不再"忘记"弃牌堆。
pub fn reconstruct(
    view: &PlayerView,
    rng: &mut dyn RngCore,
    known_discard: &[u16; 14],
) -> Option<Session> {
    let me = view.me?;
    let n = view.all_seats.len();
    if n < 2 || me >= n {
        return None;
    }

    // ---------------------------------------------------------------- 待决状态
    // 行动 A 抽到的牌处于"悬空"状态（不在任一牌堆中）。
    let mut limbo_rank: Option<u8> = None;
    let mut confirm_cabo = false;
    let mut swap_source: Option<Pile> = None;
    match &view.panel {
        Panel::Idle { .. } => {}
        Panel::Drew { rank, .. } => limbo_rank = Some(*rank),
        Panel::SwapSelecting { source, drawn, .. } => {
            swap_source = Some(*source);
            if matches!(source, Pile::Draw) {
                limbo_rank = drawn.or(view.drawn);
            }
        }
        Panel::ConfirmCabo => confirm_cabo = true,
        _ => return None,
    }
    let limbo = usize::from(limbo_rank.is_some());

    // ---------------------------------------------------------------- 牌数守恒
    let hand_sizes: Vec<usize> = view.all_seats.iter().map(|s| s.slots.len()).collect();
    let total = hand_sizes.iter().sum::<usize>() + view.deck_count + view.discard_count + limbo;
    if total != TOTAL_CARDS {
        return None;
    }

    // ---------------------------------------------------------------- 点数守恒
    let mut pool = RANK_COPIES;
    let take = |pool: &mut [usize; 14], rank: u8| -> bool {
        let r = rank as usize;
        if r >= pool.len() || pool[r] == 0 {
            return false;
        }
        pool[r] -= 1;
        true
    };
    for seat in &view.all_seats {
        for cell in &seat.slots {
            if let Some(v) = cell.value {
                if !take(&mut pool, v) {
                    return None;
                }
            }
        }
    }
    if let Some(v) = limbo_rank {
        if !take(&mut pool, v) {
            return None;
        }
    }
    if let Some(v) = view.discard_top {
        if !take(&mut pool, v) {
            return None;
        }
    }
    // 已知的弃牌堆底牌：同样是"已知点数"，直接扣掉
    for (r, &cnt) in known_discard.iter().enumerate() {
        for _ in 0..cnt {
            if !take(&mut pool, r as u8) {
                return None;
            }
        }
    }

    // ---------------------------------------------------------------- 组牌
    // 看不见的牌按"位置类型"分别抽样：对手暗牌偏低、弃牌堆底偏高、摸牌堆/我的暗牌居中。
    // 这四类的张数都是公开信息，池构成也是已知的，所以可以逐类抽取（见 tactics::THETA_*）。
    let mut cards: Vec<CardState> = Vec::with_capacity(TOTAL_CARDS);
    let mut me_unknown_ids: Vec<CardId> = Vec::new();
    let mut opp_unknown_ids: Vec<CardId> = Vec::new();
    let mut deck_ids: Vec<CardId> = Vec::new();
    let mut disc_unknown_ids: Vec<CardId> = Vec::new();
    #[allow(clippy::too_many_arguments)]
    fn push_card(
        cards: &mut Vec<CardState>,
        bucket: &mut Vec<CardId>,
        rank: Option<u8>,
        revealed: bool,
        knowers: BTreeSet<PlayerId>,
    ) -> CardId {
        let id = cards.len() as CardId;
        cards.push(CardState {
            card: Card { id, rank: rank.unwrap_or(0) },
            revealed,
            known_by: knowers,
        });
        if rank.is_none() {
            bucket.push(id);
        }
        id
    }

    let all_players: BTreeSet<PlayerId> = (0..n).collect();
    let mut slot_ids: Vec<Vec<CardId>> = Vec::with_capacity(n);
    for (p, seat) in view.all_seats.iter().enumerate() {
        let mut ids = Vec::with_capacity(seat.slots.len());
        for cell in &seat.slots {
            // 已知者集合重建：明置牌人人皆知；我知道的牌至少我在集合里。
            //
            // 关键细节：手牌中最常见的"知道者"就是**牌主本人**（开局查看自己的两张牌）。
            // 若把这点弄错（把知识随机分给别人），模拟中的对手就"不认识自己的牌"，
            // 于是它们看起来几乎无法改良手牌 —— 会导致搜索严重高估"宣告 Cabo"的价值。
            let mut knowers: BTreeSet<PlayerId> = BTreeSet::new();
            if cell.revealed {
                knowers = all_players.clone();
            } else {
                if cell.i_know {
                    knowers.insert(me);
                }
                let target = cell.knower_count.min(n);
                // 本人优先（我自己的牌除外：我不知道就说明是别人间谍看到的）。
                if p != me && knowers.len() < target {
                    knowers.insert(p);
                }
                while knowers.len() < target {
                    let cand = rng.random_range(0..n);
                    knowers.insert(cand);
                }
            }
            let bucket = if p == me { &mut me_unknown_ids } else { &mut opp_unknown_ids };
            ids.push(push_card(&mut cards, bucket, cell.value, cell.revealed, knowers));
        }
        slot_ids.push(ids);
    }

    // 悬空牌（只有我知道）。
    let mut limbo_bucket: Vec<CardId> = Vec::new();
    let limbo_id = limbo_rank.map(|r| {
        let mut k = BTreeSet::new();
        k.insert(me);
        push_card(&mut cards, &mut limbo_bucket, Some(r), false, k)
    });

    // 摸牌堆：全员未知。
    let mut deck: Vec<CardId> = Vec::with_capacity(view.deck_count);
    for _ in 0..view.deck_count {
        deck.push(push_card(&mut cards, &mut deck_ids, None, false, BTreeSet::new()));
    }

    // 弃牌堆：全部明置（堆底的点数来自公共历史，未知的部分才参与抽样）。
    let mut discard: Vec<CardId> = Vec::with_capacity(view.discard_count);
    let mut known_pool = *known_discard;
    for i in 0..view.discard_count {
        let top = i + 1 == view.discard_count;
        let rank = if top {
            view.discard_top
        } else {
            // 均匀地把"已知的堆底牌"铺进堆底位置（顺序不重要）
            let total: u16 = known_pool.iter().sum();
            if total == 0 {
                None
            } else {
                let mut pick = rng.random_range(0..total);
                let mut chosen = None;
                for (r, c) in known_pool.iter_mut().enumerate() {
                    if *c == 0 {
                        continue;
                    }
                    if pick < *c {
                        *c -= 1;
                        chosen = Some(r as u8);
                        break;
                    }
                    pick -= *c;
                }
                chosen
            }
        };
        discard.push(push_card(
            &mut cards,
            &mut disc_unknown_ids,
            rank,
            true,
            all_players.clone(),
        ));
    }

    // ---------------------------------------------------------------- 填入未知点数
    let mut leftover: Vec<u8> = Vec::new();
    for (rank, &cnt) in pool.iter().enumerate() {
        for _ in 0..cnt {
            leftover.push(rank as u8);
        }
    }
    let want = me_unknown_ids.len()
        + opp_unknown_ids.len()
        + deck_ids.len()
        + disc_unknown_ids.len();
    if leftover.len() != want {
        return None;
    }
    // 逐类抽取：先弃牌堆（高牌偏好），再对手暗牌（低牌偏好），最后我的暗牌与摸牌堆。
    let mut take_tilted = |count: usize, theta: f64, pool: &mut Vec<u8>| -> Vec<u8> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count.min(pool.len()) {
            let weights: Vec<f64> = pool.iter().map(|&r| (theta * r as f64).exp()).collect();
            let total: f64 = weights.iter().sum();
            let mut pick = pool.len() - 1;
            if total > 0.0 {
                let mut x = rng.random_range(0.0..total);
                for (i, w) in weights.iter().enumerate() {
                    x -= *w;
                    if x <= 0.0 {
                        pick = i;
                        break;
                    }
                }
            }
            out.push(pool.swap_remove(pick));
        }
        out
    };
    let disc_ranks =
        take_tilted(disc_unknown_ids.len(), super::tactics::THETA_DISCARD, &mut leftover);
    // 记牌已经扣掉了大部分"大牌迁移到弃牌堆"的效应，对手暗牌的倾斜按剩余比例缩放。
    let known_total: f64 = known_discard.iter().map(|&c| c as f64).sum();
    let disc_below = view.discard_count.saturating_sub(1) as f64;
    let known_frac = if disc_below > 0.0 { (known_total / disc_below).clamp(0.0, 1.0) } else { 0.0 };
    let theta_opp = super::tactics::theta_opp_for_scaled(n, view.deck_count, known_frac);
    let opp_ranks = take_tilted(opp_unknown_ids.len(), theta_opp, &mut leftover);
    let rest_ranks = take_tilted(me_unknown_ids.len() + deck_ids.len(), 0.0, &mut leftover);
    for (id, rank) in disc_unknown_ids.iter().zip(disc_ranks.into_iter()) {
        cards[*id as usize].card.rank = rank;
    }
    for (id, rank) in opp_unknown_ids.iter().zip(opp_ranks.into_iter()) {
        cards[*id as usize].card.rank = rank;
    }
    for (id, rank) in me_unknown_ids.iter().chain(deck_ids.iter()).zip(rest_ranks.into_iter()) {
        cards[*id as usize].card.rank = rank;
    }
    if !leftover.is_empty() {
        return None;
    }

    // ---------------------------------------------------------------- 组装局面
    let mut players: Vec<PlayerState> = Vec::with_capacity(n);
    for (p, seat) in view.all_seats.iter().enumerate() {
        players.push(PlayerState {
            name: seat.name.clone(),
            is_ai: true,
            bot_id: String::new(),
            token: None,
            slots: slot_ids[p].clone(),
            peeked_slots: Vec::new(),
            total_score: seat.total_score,
            round_score: None,
        });
    }

    let cabo_caller = view.all_seats.iter().find(|s| s.is_caller).map(|s| s.id);

    let pending = if let Some(source) = swap_source {
        let card = match source {
            Pile::Draw => limbo_id?,
            Pile::Discard => *discard.last()?,
        };
        let selected: Vec<SlotId> = view
            .me_seat
            .as_ref()
            .map(|s| s.slots.iter().filter(|c| c.selected).map(|c| c.slot).collect())
            .unwrap_or_default();
        Some(Pending::SwapSelecting { card, source, selected })
    } else if let Some(card) = limbo_id {
        Some(Pending::Drew { card })
    } else if confirm_cabo {
        Some(Pending::ConfirmCabo)
    } else {
        None
    };

    // 加时队列：当前行动者之后还剩 `extra_left` 个座位。
    let mut extra_turns = std::collections::VecDeque::new();
    if cabo_caller.is_some() {
        for step in 1..=view.extra_left {
            extra_turns.push_back((me + step) % n);
        }
    }

    let settings = Settings {
        cabo_penalty: view.cabo_penalty,
        target_score: view.target_score,
        memory_mode: false,
    };

    Some(Session {
        seed: rng.next_u64(),
        rng: StdRng::seed_from_u64(rng.next_u64()),
        settings,
        round_no: view.round_no,
        players,
        cards,
        deck,
        discard,
        phase: Phase::Turn { current: me, pending },
        cabo_caller,
        extra_turns,
        peek_flash: Default::default(),
        log: Vec::new(),
        public_events: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::planner::ChallengerBot;
    use crate::ai::Bot;
    use crate::game::view::project;
    use crate::game::sim::make_ai_session;
    use rand::rngs::StdRng;

    /// 重建局面必须与视图一致：我已知的点数、牌数、弃牌堆顶都要对得上。
    #[test]
    fn reconstruction_matches_view() {
        let bot = ChallengerBot;
        let mut rng = StdRng::seed_from_u64(7);
        let mut seen_turn = 0usize;
        // 主动陪练会更早结束一轮，跨多个发牌保持重建检查覆盖率。
        for seed in [42, 7, 99, 101] {
            let mut s = make_ai_session(seed, Settings::default(), &["challenger", "challenger", "challenger"]);
            s.start_game().unwrap();
            // 走若干步，覆盖各种待决状态。
            for _step in 0..600 {
                let pid = match &s.phase {
                    Phase::Peeking { done } => (0..s.players.len()).find(|p| !done.contains_key(p)),
                    Phase::Turn { current, .. } => Some(*current),
                    _ => None,
                };
                let Some(pid) = pid else { break };
                let view = project(&s, Some(pid), 0);
                if matches!(s.phase, Phase::Turn { .. }) {
                    seen_turn += 1;
                    let rebuilt = reconstruct(&view, &mut rng, &[0; 14]).expect("应当可以重建");
                    assert_eq!(rebuilt.players.len(), s.players.len());
                    for (i, p) in rebuilt.players.iter().enumerate() {
                        assert_eq!(p.slots.len(), s.players[i].slots.len(), "手牌数不一致");
                        assert_eq!(p.total_score, s.players[i].total_score);
                        for (j, cid) in p.slots.iter().enumerate() {
                            let mine = view.all_seats[i].slots[j].value;
                            let got = rebuilt.cards[*cid as usize].card.rank;
                            if let Some(v) = mine {
                                assert_eq!(got, v, "我已知的牌重建后必须相同");
                            }
                        }
                    }
                    assert_eq!(rebuilt.deck.len(), s.deck.len());
                    assert_eq!(rebuilt.discard.len(), s.discard.len());
                    // 点数守恒：重建局面必须是合法的一副牌。
                    let mut count = [0usize; 14];
                    for c in &rebuilt.cards {
                        count[c.card.rank as usize] += 1;
                    }
                    assert_eq!(count, RANK_COPIES, "重建后点数分布必须是合法牌组");
                    // 弃牌堆顶必须与视图一致。
                    if let (Some(top), Some(exp)) = (rebuilt.discard.last(), view.discard_top) {
                        assert_eq!(rebuilt.cards[*top as usize].card.rank, exp);
                    }
                }
                let cmd = bot.decide(&view, &mut rng);
                if s.apply(pid, &cmd).is_err() {
                    let _ = s.apply(pid, &crate::ai::fallback_command(&view));
                }
            }
        }
        assert!(seen_turn > 50, "应当覆盖足够多的回合状态，实际 {seen_turn}");
    }
}
