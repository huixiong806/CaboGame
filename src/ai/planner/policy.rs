//! Candidate generation and inexpensive, knowledge-respecting opponent policies.
use super::state::{Action, Stage, State};
use crate::game::PowerUse;

pub(super) struct Info {
    pub values: Vec<Vec<Option<u8>>>,
    pub means: Vec<Vec<f64>>,
    pub(super) probs: Vec<Vec<[f64; 14]>>,
    pub sums: Vec<f64>,
    pub lowest_prob: f64,
}

impl Info {
    pub fn new(s: &State, me: usize, need_call: bool) -> Self {
        let pool = s.pool(me);
        let mut values = Vec::new();
        let mut means = Vec::new();
        let mut probs = Vec::new();
        for hand in &s.hands {
            let mut vs = Vec::new();
            let mut ms = Vec::new();
            let mut ps = Vec::new();
            for &id in hand {
                let v = s.visible(id, me);
                let mut p = [0.0; 14];
                if let Some(r) = v {
                    p[r as usize] = 1.0;
                } else {
                    let c = s.cards[id as usize];
                    let mut total = 0.0;
                    for r in 0..14 {
                        p[r] = pool[r] as f64 * (c.log_weights[r] as f64).exp();
                        total += p[r];
                    }
                    if total > 0.0 {
                        for v in &mut p {
                            *v /= total;
                        }
                    } else {
                        p.fill(1.0 / 14.0);
                    }
                }
                ms.push(p.iter().enumerate().map(|(r, &p)| r as f64 * p).sum());
                ps.push(p);
                vs.push(v);
            }
            values.push(vs);
            means.push(ms);
            probs.push(ps);
        }
        let sums = means.iter().map(|v| v.iter().sum()).collect();
        let mut info = Self {
            values,
            means,
            probs,
            sums,
            lowest_prob: 0.0,
        };
        if need_call {
            info.lowest_prob = info.p_lowest(me);
        }
        info
    }

    fn sum_distribution(&self, player: usize) -> Vec<f64> {
        let mut dist = vec![1.0];
        for p in &self.probs[player] {
            let mut next = vec![0.0; dist.len() + 13];
            for (i, &v) in dist.iter().enumerate() {
                if v < 1e-12 {
                    continue;
                }
                for (r, &w) in p.iter().enumerate() {
                    next[i + r] += v * w;
                }
            }
            dist = next;
        }
        dist
    }

    fn p_lowest(&self, me: usize) -> f64 {
        let my = self.sum_distribution(me);
        let mut tails = Vec::new();
        for j in 0..self.values.len() {
            if j == me {
                continue;
            }
            let d = self.sum_distribution(j);
            let mut tail = vec![0.0; d.len()];
            let mut sum = 0.0;
            for k in (0..d.len()).rev() {
                tail[k] = sum;
                sum += d[k];
            }
            tails.push(tail);
        }
        my.iter()
            .enumerate()
            .map(|(sum, &p)| {
                p * tails
                    .iter()
                    .map(|t| t.get(sum).copied().unwrap_or(0.0))
                    .product::<f64>()
            })
            .sum()
    }

    pub(super) fn p_lowest_at(&self, me: usize, sum: u32) -> f64 {
        (0..self.values.len())
            .filter(|&p| p != me)
            .map(|p| {
                self.sum_distribution(p)
                    .iter()
                    .skip(sum as usize + 1)
                    .sum::<f64>()
            })
            .product()
    }

    fn match_probability(&self, me: usize, slots: &[u8]) -> f64 {
        (0..14)
            .map(|r| {
                slots
                    .iter()
                    .map(|&slot| self.probs[me][slot as usize][r])
                    .product::<f64>()
            })
            .sum()
    }
}

fn groups(info: &Info, me: usize) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let size = info.values[me].len();
    for r in 0..14 {
        let g: Vec<u8> = (0..size)
            .filter(|&i| info.values[me][i] == Some(r))
            .map(|i| i as u8)
            .collect();
        if g.len() > 1 {
            result.push(g);
        }
    }
    // Small hands: every subset that is not already known to fail is considered.
    if size <= 8 {
        for mask in 1usize..(1 << size) {
            if mask.count_ones() < 2 {
                continue;
            }
            let g: Vec<u8> = (0..size)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| i as u8)
                .collect();
            let known: Vec<u8> = g
                .iter()
                .filter_map(|&i| info.values[me][i as usize])
                .collect();
            if known.windows(2).all(|w| w[0] == w[1]) && !result.contains(&g) {
                result.push(g);
            }
        }
    } else {
        // Polynomial fallback for pathological hands, with no four-card assumption.
        for a in 0..size {
            for b in a + 1..size {
                if info.values[me][a].is_none() || info.values[me][b].is_none() {
                    result.push(vec![a as u8, b as u8]);
                }
            }
        }
    }
    result
}

pub(super) fn candidates(s: &State, info: &Info) -> Vec<Action> {
    let me = s.actor;
    let mut out = Vec::new();
    let mut selections: Vec<Vec<u8>> = (0..s.hands[me].len()).map(|i| vec![i as u8]).collect();
    selections.extend(groups(info, me));
    match s.stage {
        Stage::Idle => {
            if !s.deck.is_empty() {
                out.push(Action::Draw);
            }
            if s.caller.is_none() {
                out.push(Action::Cabo);
            }
            if !s.discard.is_empty() {
                for slots in selections {
                    out.push(Action::Exchange(slots));
                }
            }
        }
        Stage::Drew(id) => {
            for slots in selections {
                out.push(Action::Replace(slots));
            }
            out.push(Action::Discard(None));
            match s.cards[id as usize].rank {
                7..=8 => {
                    for slot in 0..s.hands[me].len() {
                        if info.values[me][slot].is_none() {
                            out.push(Action::Discard(Some(PowerUse::PeekOwn {
                                slot: slot as u8,
                            })));
                        }
                    }
                }
                9..=10 => {
                    for p in 0..s.n() {
                        if p != me {
                            for slot in 0..s.hands[p].len() {
                                if info.values[p][slot].is_none() {
                                    out.push(Action::Discard(Some(PowerUse::Spy {
                                        player: p,
                                        slot: slot as u8,
                                    })));
                                }
                            }
                        }
                    }
                }
                11..=12 => {
                    for my_slot in 0..s.hands[me].len() {
                        for player in 0..s.n() {
                            if player != me {
                                for slot in 0..s.hands[player].len() {
                                    out.push(Action::Discard(Some(PowerUse::Swap {
                                        my_slot: my_slot as u8,
                                        player,
                                        slot: slot as u8,
                                    })));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Stage::End => {}
    }
    out
}

/// Score uses means of distributions, never the hidden card rank stored in the simulation.
pub(super) fn score(s: &State, info: &Info, action: &Action) -> f64 {
    ordinary_score(s, info, action)
        + super::special::bonus(s, info, action)
        + super::special::uncertain_reset_bonus(s, info, action)
}

fn ordinary_score(s: &State, info: &Info, action: &Action) -> f64 {
    let me = s.actor;
    let future = s.caller.is_none() && s.deck.len() > 1;
    let threat = |p: usize| {
        let min = *s.totals.iter().min().unwrap() as f64;
        (0.65 * (-(s.totals[p] as f64 - min) / 25.0).exp())
            + if s.caller == Some(p) { 0.35 } else { 0.0 }
    };
    match action {
        Action::Draw => 1.6,
        Action::Cabo => {
            let own = info.sums[me];
            info.lowest_prob * own - (1.0 - info.lowest_prob) * s.penalty as f64 - 2.0
        }
        Action::Exchange(slots) | Action::Replace(slots) => {
            let incoming = match action {
                Action::Exchange(_) => s.cards[*s.discard.last().unwrap() as usize].rank,
                _ => {
                    let Stage::Drew(id) = s.stage else {
                        unreachable!()
                    };
                    s.cards[id as usize].rank
                }
            };
            let psuccess = if slots.len() == 1 {
                1.0
            } else {
                info.match_probability(me, slots)
            };
            let removed: f64 = if slots.len() == 1 {
                info.means[me][slots[0] as usize]
            } else {
                // Conditional on matching, all removed cards have the same rank.
                (0..14)
                    .map(|r| {
                        slots.len() as f64
                            * r as f64
                            * slots
                                .iter()
                                .map(|&slot| info.probs[me][slot as usize][r])
                                .product::<f64>()
                    })
                    .sum()
            };
            let gain = removed - incoming as f64;
            let unknown = slots
                .iter()
                .filter(|&&i| info.values[me][i as usize].is_none())
                .count();
            let duplicate = info.values[me]
                .iter()
                .enumerate()
                .any(|(i, &v)| !slots.contains(&(i as u8)) && v == Some(incoming));
            let bonus = if future {
                0.8 * psuccess * (slots.len() - 1) as f64
                    + 0.9 * unknown as f64 * psuccess
                    + if duplicate {
                        incoming.min(5) as f64 * 0.55
                    } else {
                        0.0
                    }
            } else {
                0.0
            };
            gain + bonus
        }
        Action::Discard(None) => 0.0,
        Action::Discard(Some(PowerUse::PeekOwn { .. })) => {
            if future {
                2.0
            } else {
                0.0
            }
        }
        Action::Discard(Some(PowerUse::Spy { player, slot })) => {
            if future {
                0.5 + threat(*player) + 0.06 * info.means[*player][*slot as usize]
            } else {
                0.0
            }
        }
        Action::Discard(Some(PowerUse::Swap {
            my_slot,
            player,
            slot,
        })) => {
            let gain = info.means[me][*my_slot as usize] - info.means[*player][*slot as usize];
            let knowledge_bonus = if future
                && info.values[me][*my_slot as usize].is_none()
                && info.values[*player][*slot as usize].is_some()
            {
                0.8
            } else {
                0.0
            };
            let own_known = info.values[me][*my_slot as usize];
            let rival = info.values[*player][*slot as usize];
            let breaks_pair = own_known.is_some()
                && info.values[me].iter().filter(|&&v| v == own_known).count() >= 2;
            let makes_pair = rival.is_some() && info.values[me].contains(&rival);
            gain * (1.0 + threat(*player))
                + knowledge_bonus
                + if future {
                    1.2 * (makes_pair as u8 as f64 - breaks_pair as u8 as f64)
                } else {
                    0.0
                }
        }
    }
}

pub(super) fn ranked(s: &State, max_actions: usize) -> Vec<(Action, f64)> {
    let info = Info::new(s, s.actor, s.caller.is_none() && s.stage == Stage::Idle);
    let mut actions: Vec<_> = candidates(s, &info)
        .into_iter()
        .map(|a| {
            let value = score(s, &info, &a);
            (a, value)
        })
        .collect();
    actions.sort_by(|a, b| b.1.total_cmp(&a.1));
    // If large hands force pruning, preserve draw, discard and call and each power family.
    if actions.len() > max_actions {
        let mut keep: Vec<_> = actions
            .iter()
            .filter(|(a, _)| matches!(a, Action::Draw | Action::Cabo | Action::Discard(None)))
            .cloned()
            .collect();
        for (a, v) in actions {
            if keep.len() >= max_actions {
                break;
            }
            if !keep.iter().any(|(b, _)| *b == a) {
                keep.push((a, v));
            }
        }
        keep.sort_by(|a, b| b.1.total_cmp(&a.1));
        keep
    } else {
        actions
    }
}

pub(super) fn search_ranked(s: &State, max_actions: usize, learned: bool) -> Vec<(Action, f64)> {
    if learned {
        if let Some(model) = &s.policy_model {
            return model.ranked(s, max_actions);
        }
    }
    ranked(s, max_actions)
}

/// Four distinct styles: proactive, balanced, cautious, and a passive control.
/// Each simulation samples a style per seat, held fixed throughout the round.
pub(super) fn choose(s: &State, style: u8) -> Action {
    let info = Info::new(s, s.actor, s.caller.is_none() && s.stage == Stage::Idle);
    if style != 3 && super::special::preferred_call(s, &info) {
        return Action::Cabo;
    }
    let threshold = match style {
        0 => 0.70,
        1 => 0.82,
        2 => 0.94,
        _ => 2.0,
    };
    if s.stage == Stage::Idle && s.caller.is_none() && info.lowest_prob >= threshold {
        let me = s.actor;
        let min_other = (0..s.n())
            .filter(|&p| p != me)
            .map(|p| info.sums[p])
            .fold(f64::INFINITY, f64::min);
        // Budget for opponents' final improvements; the tree itself simulates actual responses.
        if info.sums[me] + 2.0 < min_other {
            return Action::Cabo;
        }
    }
    candidates(s, &info)
        .into_iter()
        .filter(|a| !matches!(a, Action::Cabo))
        .max_by(|a, b| score(s, &info, a).total_cmp(&score(s, &info, b)))
        .unwrap_or(Action::Cabo)
}

/// Learned continuations are separate from the frozen incumbent and root ranking.
pub(super) fn rollout(s: &State, style: u8) -> Action {
    let active = match s.rollout_policy {
        1 => true,
        2 => s.actor != s.policy_player,
        3 => s.actor == s.policy_player,
        _ => false,
    };
    if active && s.target == 100 && s.penalty == 10 {
        if let Some(model) = &s.policy_model {
            return model.choose(s, s.policy_actions);
        }
    }
    choose(s, style)
}
