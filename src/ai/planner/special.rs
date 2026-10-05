//! Observable special-rule opportunities. No hidden ranks enter these priors or policies.
use super::policy::Info;
use super::state::{Action, Stage, State};
use crate::game::{scoring, PowerUse};

fn known_hand(hand: &[Option<u8>]) -> Option<Vec<u8>> {
    hand.iter().copied().collect()
}

/// A deliberately failed trade can append the exact points needed for a reset.
/// Force it only on the last Cabo response, when every hand is legally known
/// and real settlement proves the actor will uniquely win the whole match.
pub(super) fn proven_reset_trade(s: &State) -> Option<Action> {
    let me = s.actor;
    if !s.proven_reset_trade
        || me != s.policy_player
        || s.caller.is_none()
        || s.caller == Some(me)
        || !s.extra.is_empty()
        || s.score_reset_used[me]
    {
        return None;
    }
    let incoming = match s.stage {
        Stage::Idle => s.visible(*s.discard.last()?, me)?,
        Stage::Drew(id) => s.visible(id, me)?,
        Stage::End => return None,
    };
    let mut hands: Vec<Vec<u8>> = s
        .hands
        .iter()
        .map(|hand| hand.iter().map(|&id| s.visible(id, me)).collect())
        .collect::<Option<_>>()?;
    if s.totals[me] + hands[me].iter().map(|&r| r as u32).sum::<u32>() + incoming as u32
        != scoring::RESET_AT
    {
        return None;
    }
    let pair = (0..hands[me].len()).find_map(|i| {
        (i + 1..hands[me].len())
            .find(|&j| hands[me][i] != hands[me][j])
            .map(|j| vec![i as u8, j as u8])
    })?;
    let unique_win = |totals: &[u32]| {
        totals.iter().any(|&t| t >= s.target)
            && totals
                .iter()
                .enumerate()
                .all(|(p, &t)| p == me || totals[me] < t)
    };
    let before = scoring::settle(&hands, &s.totals, &s.score_reset_used, s.caller, s.penalty);
    if unique_win(&before.totals) {
        return None;
    }
    hands[me].push(incoming);
    let after = scoring::settle(&hands, &s.totals, &s.score_reset_used, s.caller, s.penalty);
    if !after.reset_triggered[me] || !unique_win(&after.totals) {
        return None;
    }
    Some(match s.stage {
        Stage::Idle => Action::Exchange(pair),
        Stage::Drew(_) => Action::Replace(pair),
        Stage::End => unreachable!(),
    })
}

fn complete(hand: &[Option<u8>]) -> bool {
    hand.len() == 4
        && hand.iter().filter(|&&r| r == Some(12)).count() == 2
        && hand.iter().filter(|&&r| r == Some(13)).count() == 2
}

pub(super) fn call_relevant(s: &State) -> bool {
    if !s.special_tactics || s.caller.is_some() || s.stage != Stage::Idle {
        return false;
    }
    let hand: Vec<_> = s.hands[s.actor]
        .iter()
        .map(|&id| s.visible(id, s.actor))
        .collect();
    if s.reset_policy_player == Some(s.actor)
        && !s.score_reset_used[s.actor]
        && s.totals[s.actor] < scoring::RESET_AT
        && s.totals[s.actor] + 13 * hand.len() as u32 + s.penalty >= scoring::RESET_AT
    {
        return true;
    }
    let Some(hand) = known_hand(&hand) else {
        return false;
    };
    scoring::is_high_pairs(&hand)
        || (!s.score_reset_used[s.actor]
            && s.totals[s.actor] + hand.iter().map(|&r| r as u32).sum::<u32>() + s.penalty
                == scoring::RESET_AT)
}

// Distribution convolution, not rounding a hand's mean to an integer. This is
// an action-ranking approximation; actual search settlement uses joint finite-deck worlds.
pub(super) fn uncertain_reset_bonus(s: &State, info: &Info, action: &Action) -> f64 {
    let me = s.actor;
    if !s.special_tactics
        || s.reset_policy_player != Some(me)
        || s.score_reset_used[me]
        || s.totals[me] >= scoring::RESET_AT
        || s.totals[me] + 13 * s.hands[me].len() as u32 + s.penalty < scoring::RESET_AT
    {
        return 0.0;
    }
    let mut after = info.probs[me].clone();
    let mut values = info.values[me].clone();
    match action {
        Action::Exchange(slots) | Action::Replace(slots) => {
            if slots.len() > 1
                && !slots.iter().all(|&i| {
                    values[i as usize].is_some() && values[i as usize] == values[slots[0] as usize]
                })
            {
                return 0.0;
            }
            let id = match action {
                Action::Exchange(_) => *s.discard.last().unwrap(),
                _ => {
                    let Stage::Drew(id) = s.stage else { return 0.0 };
                    id
                }
            };
            let Some(rank) = s.visible(id, me) else {
                return 0.0;
            };
            let mut p = [0.0; 14];
            p[rank as usize] = 1.0;
            for i in (0..after.len()).rev() {
                if slots.contains(&(i as u8)) {
                    after.remove(i);
                    values.remove(i);
                }
            }
            let first = *slots.iter().min().unwrap() as usize;
            after.insert(first, p);
            values.insert(first, Some(rank));
        }
        Action::Discard(Some(PowerUse::Swap {
            my_slot,
            player,
            slot,
        })) => {
            after[*my_slot as usize] = info.probs[*player][*slot as usize];
            values[*my_slot as usize] = info.values[*player][*slot as usize];
        }
        Action::Cabo => {}
        _ => return 0.0,
    }
    let potential = |probs: &[[f64; 14]], values: &[Option<u8>], call: bool| {
        // Existing deterministic special tactics already account for known hands.
        if values.iter().all(Option::is_some) {
            return 0.0;
        }
        let mut d = vec![1.0];
        for p in probs {
            let mut next = vec![0.0; d.len() + 13];
            for (i, &v) in d.iter().enumerate() {
                for (r, &w) in p.iter().enumerate() {
                    next[i + r] += v * w;
                }
            }
            d = next;
        }
        let target = scoring::RESET_AT - s.totals[me];
        let direct = d.get(target as usize).copied().unwrap_or(0.0);
        let via_call = if s.caller.is_none() && target >= s.penalty {
            let sum = target - s.penalty;
            d.get(sum as usize).copied().unwrap_or(0.0) * (1.0 - info.p_lowest_at(me, sum))
        } else {
            0.0
        };
        50.0 * if call {
            via_call
        } else {
            direct.max(0.4 * via_call)
        }
    };
    if matches!(action, Action::Cabo) {
        potential(&after, &values, true)
    } else {
        potential(&after, &values, false) - potential(&info.probs[me], &info.values[me], false)
    }
}

fn known_call_value(s: &State, info: &Info) -> Option<f64> {
    let hands: Option<Vec<Vec<u8>>> = info.values.iter().map(|h| known_hand(h)).collect();
    let result = scoring::settle(
        &hands?,
        &s.totals,
        &s.score_reset_used,
        Some(s.actor),
        s.penalty,
    );
    Some(s.utility_with_resets(&result.totals, &result.reset_used, s.actor))
}

pub(super) fn preferred_call(s: &State, info: &Info) -> bool {
    if !call_relevant(s) {
        return false;
    }
    if let Some(value) = known_call_value(s, info) {
        // A combo/reset is not itself a win. Compare match totals after everyone's resets.
        return value > s.utility(s.actor) + 0.08;
    }
    complete(&info.values[s.actor])
}

fn reset_value(s: &State, info: &Info, hand: &[Option<u8>]) -> f64 {
    let me = s.actor;
    if s.score_reset_used[me] || complete(hand) {
        return 0.0;
    }
    let Some(hand) = known_hand(hand) else {
        return 0.0;
    };
    let sum: u32 = hand.iter().map(|&r| r as u32).sum();
    if s.totals[me] + sum == scoring::RESET_AT {
        return 50.0;
    }
    // A future call can earn a penalty only if it actually fails. Don't treat a
    // distribution's mean as an exact integer hand, or a successful call as +penalty.
    if s.caller.is_none() && s.totals[me] + sum + s.penalty == scoring::RESET_AT {
        return 50.0 * (1.0 - info.p_lowest_at(me, sum));
    }
    0.0
}

pub(super) fn bonus(s: &State, info: &Info, action: &Action) -> f64 {
    if !s.special_tactics {
        return 0.0;
    }
    let me = s.actor;
    if matches!(action, Action::Cabo) {
        if !call_relevant(s) {
            return 0.0;
        }
        if let Some(value) = known_call_value(s, info) {
            return 60.0 * (value - s.utility(me));
        }
        return if complete(&info.values[me]) {
            80.0
        } else {
            reset_value(s, info, &info.values[me])
        };
    }
    // Most early-round hands cannot complete a combo or reach 100 in one action.
    // Avoid allocating projected hands in those ordinary decisions.
    let own = &info.values[me];
    let can_complete = own.iter().filter(|&&r| matches!(r, Some(12 | 13))).count() >= 3;
    let can_reset = !s.score_reset_used[me]
        && own.iter().filter(|r| r.is_none()).count() <= 1
        && s.totals[me] + 13 * own.len() as u32 + s.penalty >= scoring::RESET_AT;
    let can_break = info
        .values
        .iter()
        .enumerate()
        .any(|(p, h)| p != me && complete(h));
    if !can_complete && !can_reset && !can_break {
        return 0.0;
    }
    let mut after = info.values.clone();
    match action {
        Action::Exchange(slots) | Action::Replace(slots) => {
            // Uncertain multi-card success is left to posterior-world simulation.
            if slots.len() > 1
                && !slots.iter().all(|&i| {
                    after[me][i as usize].is_some()
                        && after[me][i as usize] == after[me][slots[0] as usize]
                })
            {
                return 0.0;
            }
            let incoming = match action {
                Action::Exchange(_) => s.visible(*s.discard.last().unwrap(), me),
                _ => {
                    let Stage::Drew(id) = s.stage else { return 0.0 };
                    s.visible(id, me)
                }
            };
            let first = *slots.iter().min().unwrap() as usize;
            for i in (0..after[me].len()).rev() {
                if slots.contains(&(i as u8)) {
                    after[me].remove(i);
                }
            }
            after[me].insert(first, incoming);
        }
        Action::Discard(Some(PowerUse::Swap {
            my_slot,
            player,
            slot,
        })) => {
            let own = after[me][*my_slot as usize];
            after[me][*my_slot as usize] = after[*player][*slot as usize];
            after[*player][*slot as usize] = own;
        }
        _ => return 0.0,
    }
    let mut gain = 80.0
        * (complete(&after[me]) as u8 as f64 - complete(&info.values[me]) as u8 as f64)
        + reset_value(s, info, &after[me])
        - reset_value(s, info, &info.values[me]);
    for p in 0..s.n() {
        if p != me {
            gain +=
                60.0 * (complete(&info.values[p]) as u8 as f64 - complete(&after[p]) as u8 as f64);
        }
    }
    gain
}
