//! 简单启发式 Bot：用于测试与陪玩。
//!
//! 决策只依赖 [`PlayerView`](crate::game::view::PlayerView)（该座位有权知道的信息）
//! + 全局牌组构成的公开常识，通过统计推断未知牌的期望点数，不作弊、不读随机数种子。

use rand::seq::SliceRandom;

use crate::game::view::{Panel, PlayerView};
use crate::game::{Command, Pile, PowerUse, RANK_COPIES};

pub struct SimpleBot;

impl super::Bot for SimpleBot {
    fn id(&self) -> &'static str {
        "simple"
    }

    fn name(&self) -> &'static str {
        "简单 AI"
    }

    fn decide(&self, view: &PlayerView, rng: &mut dyn rand::RngCore) -> Command {
        match &view.panel {
            Panel::PeekPick { .. } => peek_initial(view, rng),
            Panel::Idle { can_draw, can_swap_discard, can_cabo } => {
                decide_idle(view, *can_draw, *can_swap_discard, *can_cabo)
            }
            Panel::Drew { rank, .. } => decide_drawn(view, *rank),
            // Bot 正常不会进入逐步瞄准状态（它用一次性命令），这里仅作兜底。
            Panel::AimPeek => {
                Command::PowerPickOwn { slot: unknown_own_slots(view).first().copied().unwrap_or(0) }
            }
            Panel::AimSpy => {
                let (p, s) = best_spy_target(view).unwrap_or((0, 0));
                Command::PowerPickOther { player: p, slot: s }
            }
            Panel::AimSwap { .. } => {
                let (p, s) = best_spy_target(view).unwrap_or((0, 0));
                Command::PowerPickOther { player: p, slot: s }
            }
            // 摸牌后进入的逐步选择状态（DrawSwap 空槽位形式）：
            // 重新计算该换哪张：已知牌里比手中新牌差的优先，否则选一张未知牌。
            Panel::SwapSelecting { count, source, .. } => {
                if *count >= 1 {
                    Command::SwapCommit
                } else {
                    assert!(*source == Pile::Draw, "Bot 不会主动进入弃牌堆选择模式");
                    Command::SwapToggle { slot: pick_slot_for_drawn(view) }
                }
            }
            Panel::ConfirmCabo => Command::CallCabo,
            _ => super::fallback_command(view),
        }
    }
}

// ---------------------------------------------------------------- 视图统计

/// 我已知点数的槽位。
fn known_own(view: &PlayerView) -> Vec<(u8, u8)> {
    view.me_seat
        .as_ref()
        .map(|s| {
            s.slots
                .iter()
                .filter(|c| c.i_know)
                .map(|c| (c.slot, c.value.unwrap()))
                .collect()
        })
        .unwrap_or_default()
}

/// 我不知道点数的槽位。
fn unknown_own_slots(view: &PlayerView) -> Vec<u8> {
    view.me_seat
        .as_ref()
        .map(|s| s.slots.iter().filter(|c| !c.i_know).map(|c| c.slot).collect())
        .unwrap_or_default()
}

/// 从我的视角看，一张未知牌的期望点数。
///
/// 可见信息：所有已亮出的牌、我知道的点数、弃牌堆顶、自己刚抽的牌。
/// 剩余牌池 = 摸牌堆 + 所有仍保密的槽位牌。
fn expected_unknown(view: &PlayerView) -> f64 {
    let mut remaining: Vec<i32> = RANK_COPIES.iter().map(|&c| c as i32).collect();
    let mut dec = |rank: u8| {
        if (rank as usize) < remaining.len() && remaining[rank as usize] > 0 {
            remaining[rank as usize] -= 1;
        }
    };
    if let Some(seat) = &view.me_seat {
        for c in &seat.slots {
            if let Some(v) = c.value {
                dec(v);
            }
        }
    }
    for opp in &view.opponents {
        for c in &opp.slots {
            if let Some(v) = c.value {
                dec(v);
            }
        }
    }
    if let Some(t) = view.discard_top {
        dec(t);
    }
    if let Some(d) = view.drawn {
        dec(d);
    }
    let hidden_slots: usize = view
        .me_seat
        .iter()
        .flat_map(|s| s.slots.iter())
        .chain(view.opponents.iter().flat_map(|s| s.slots.iter()))
        .filter(|c| c.value.is_none())
        .count();
    let pool = view.deck_count + hidden_slots;
    if pool == 0 {
        return 6.5;
    }
    let total: i64 =
        remaining.iter().enumerate().map(|(rank, &cnt)| rank as i64 * cnt as i64).sum();
    total as f64 / pool as f64
}

/// 间谍/交换的最好目标：未知牌最多的对手及其一个未知槽位。
fn best_spy_target(view: &PlayerView) -> Option<(usize, u8)> {
    let mut best: Option<(usize, u8, usize)> = None;
    for opp in &view.opponents {
        let unknown: Vec<u8> =
            opp.slots.iter().filter(|c| c.value.is_none()).map(|c| c.slot).collect();
        if unknown.is_empty() {
            continue;
        }
        let cnt = unknown.len();
        if best.is_none_or(|(_, _, bc)| cnt > bc) {
            best = Some((opp.id, unknown[0], cnt));
        }
    }
    best.map(|(p, s, _)| (p, s))
}

/// 我已知牌按点数分组。
fn pair_groups(view: &PlayerView) -> Vec<(u8, Vec<u8>)> {
    let mut groups: Vec<(u8, Vec<u8>)> = Vec::new();
    for (slot, rank) in known_own(view) {
        match groups.iter_mut().find(|(r, _)| *r == rank) {
            Some((_, slots)) => slots.push(slot),
            None => groups.push((rank, vec![slot])),
        }
    }
    groups
}

/// 摸到一张点数为 `view.drawn` 的牌后，最值得用它换掉的槽位：
/// 已知牌里点数最大且比它大的；否则一张未知牌；否则任意槽位。
fn pick_slot_for_drawn(view: &PlayerView) -> u8 {
    let drawn = view.drawn.unwrap_or(0);
    if let Some(&(slot, _val)) =
        known_own(view).iter().max_by_key(|(_, v)| *v).filter(|(_, v)| *v > drawn)
    {
        return slot;
    }
    unknown_own_slots(view)
        .first()
        .copied()
        .unwrap_or_else(|| (view.my_slot_count().max(1) - 1) as u8)
}

// ---------------------------------------------------------------- 各状态决策

fn peek_initial(view: &PlayerView, rng: &mut dyn rand::RngCore) -> Command {
    let mut slots: Vec<u8> =
        view.me_seat.as_ref().map(|s| s.slots.iter().map(|c| c.slot).collect()).unwrap_or_default();
    slots.shuffle(rng);
    if slots.len() >= 2 {
        Command::PeekInitial { slots: [slots[0], slots[1]] }
    } else {
        Command::PeekInitial { slots: [0, 1] }
    }
}

fn decide_idle(
    view: &PlayerView,
    can_draw: bool,
    can_swap_discard: bool,
    can_cabo: bool,
) -> Command {
    let known = known_own(view);
    let unknown_cnt = unknown_own_slots(view).len();
    let e = expected_unknown(view);
    let est = known.iter().map(|(_, v)| *v as f64).sum::<f64>() + unknown_cnt as f64 * e;

    // 1) 值得宣告 Cabo 吗？全知且足够低，或期望极低时才冒 +惩罚 的风险。
    if can_cabo && ((unknown_cnt == 0 && est <= 5.0) || (unknown_cnt > 0 && est <= 2.5)) {
        return Command::CallCabo;
    }
    // 2) 有值得换出的同点数组吗？（多张换 1 张，缩小手牌）
    if let Some(cmd) = swap_group_command(view, can_swap_discard) {
        return cmd;
    }
    // 3) 弃牌堆顶比我最差的已知牌好：单张换入。
    if can_swap_discard {
        if let Some(top) = view.discard_top {
            if let Some(&(slot, val)) = known.iter().max_by_key(|(_, v)| *v) {
                if top < val {
                    return Command::SwapOnce { slots: vec![slot] };
                }
            }
        }
    }
    // 4) 摸牌（摸到后看牌再决定弃置 / 发动能力 / 换入）。
    if can_draw {
        return Command::BeginDraw;
    }
    Command::CallCabo
}

fn swap_group_command(view: &PlayerView, can_swap_discard: bool) -> Option<Command> {
    let t = if can_swap_discard { view.discard_top? as f64 } else { return None };
    let mut best: Option<(f64, u8, Vec<u8>)> = None;
    for (rank, slots) in pair_groups(view) {
        if slots.len() < 2 {
            continue;
        }
        let gain = slots.len() as f64 * rank as f64 - t;
        if gain >= 2.0 && best.as_ref().is_none_or(|(bg, _, _)| gain > *bg) {
            best = Some((gain, rank, slots));
        }
    }
    best.map(|(_, _, slots)| Command::SwapOnce { slots })
}

fn decide_drawn(view: &PlayerView, v: u8) -> Command {
    let unknown = unknown_own_slots(view);
    // a) 7/8 偷看：弃掉换情报。
    if (7..=8).contains(&v) && !unknown.is_empty() {
        return Command::DiscardDrawn { power: Some(PowerUse::PeekOwn { slot: unknown[0] }) };
    }
    // b) 9/10 间谍：侦查未知牌最多的对手。
    if (9..=10).contains(&v) {
        if let Some((p, s)) = best_spy_target(view) {
            return Command::DiscardDrawn { power: Some(PowerUse::Spy { player: p, slot: s }) };
        }
    }
    // c) 11/12 交换：把手里的高牌盲换成对手的未知牌。
    if (11..=12).contains(&v) {
        let known = known_own(view);
        let has_high = known.iter().any(|(_, val)| *val >= 10);
        if has_high {
            if let Some(&(slot, _)) =
                known.iter().filter(|(_, val)| *val >= 10).max_by_key(|(_, val)| *val)
            {
                if let Some((p, s)) = best_spy_target(view) {
                    return Command::DiscardDrawn {
                        power: Some(PowerUse::Swap { my_slot: slot, player: p, slot: s }),
                    };
                }
            }
        }
    }
    // d) 牌不错（比某张已知牌小）：直接与手牌交换（单张，必成功）。
    if let Some(&(slot, val)) = known_own(view).iter().max_by_key(|(_, val)| *val) {
        if v < val {
            return Command::DrawSwap { slots: vec![slot] };
        }
    }
    // e) 否则弃置。
    Command::DiscardDrawn { power: None }
}
