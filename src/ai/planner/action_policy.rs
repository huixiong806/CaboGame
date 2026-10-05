//! Portable search-policy distillation. Inputs are actor-visible distributions only.
use super::{
    policy::{self, Info},
    state::{Action, Stage, State},
};
use crate::game::PowerUse;
use std::{path::Path, sync::Arc};

pub(super) const INPUTS: usize = 64;
const WEIGHTS: usize = 1873;

#[derive(Debug)]
pub(super) struct ActionPolicy {
    weights: Vec<f32>,
    wide: bool,
}

impl ActionPolicy {
    pub fn ranked(&self, s: &State, max_actions: usize) -> Vec<(Action, f64)> {
        if s.target != 100 || s.penalty != 10 {
            return policy::ranked(s, max_actions);
        }
        let info = Info::new(s, s.actor, s.stage == Stage::Idle && s.caller.is_none());
        let ctx = context(s, &info);
        let mut actions: Vec<_> = policy::candidates(s, &info)
            .into_iter()
            .map(|a| {
                let value = 4. * self.logit(&action_features(s, &info, &a, &ctx)) as f64;
                (a, value)
            })
            .collect();
        actions.sort_by(|a, b| b.1.total_cmp(&a.1));
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
    pub fn load(path: &Path) -> Result<Arc<Self>, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::decode(&bytes).map(Arc::new)
    }
    pub(super) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let wide = match bytes.get(..8) {
            Some(b"CABOPL01") => false,
            Some(b"CABOPL02") => true,
            _ => return Err("invalid action-policy header".into()),
        };
        if bytes.len() != 8 + (if wide { 6273 } else { WEIGHTS }) * 4 {
            return Err("invalid action-policy format/size".into());
        }
        let weights: Vec<_> = bytes[8..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if weights.iter().any(|w| !w.is_finite() || w.abs() > 100.) {
            return Err("invalid action-policy weights".into());
        }
        Ok(Self { weights, wide })
    }
    pub fn logit(&self, x: &[f32; INPUTS]) -> f32 {
        if self.wide {
            self.infer::<64, 32>(x)
        } else {
            self.infer::<24, 12>(x)
        }
    }
    fn infer<const A: usize, const B: usize>(&self, x: &[f32; INPUTS]) -> f32 {
        let mut h = [0.; A];
        for o in 0..A {
            h[o] = (self.weights[64 * A + o]
                + (0..64)
                    .map(|i| self.weights[o * 64 + i] * x[i])
                    .sum::<f32>())
            .max(0.);
        }
        let second = 65 * A;
        let bias2 = second + A * B;
        let output = bias2 + B;
        let mut h2 = [0.; B];
        for o in 0..B {
            h2[o] = (self.weights[bias2 + o]
                + (0..A)
                    .map(|i| self.weights[second + o * A + i] * h[i])
                    .sum::<f32>())
            .max(0.);
        }
        x[55] * 5.
            + self.weights[output + B]
            + (0..B)
                .map(|i| self.weights[output + i] * h2[i])
                .sum::<f32>()
    }
    pub fn choose(&self, s: &State, max_actions: usize) -> Action {
        if s.target != 100 || s.penalty != 10 {
            return policy::choose(s, 1);
        }
        let info = Info::new(s, s.actor, s.stage == Stage::Idle && s.caller.is_none());
        if super::special::preferred_call(s, &info) {
            return Action::Cabo;
        }
        let mut ranked: Vec<_> = policy::candidates(s, &info)
            .into_iter()
            .map(|a| {
                let score = policy::score(s, &info, &a);
                (a, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut actions: Vec<_> = ranked
            .iter()
            .take(max_actions)
            .map(|(a, _)| a.clone())
            .collect();
        for (a, _) in &ranked {
            if matches!(a, Action::Draw | Action::Cabo | Action::Discard(None))
                && !actions.contains(a)
            {
                actions.push(a.clone());
            }
        }
        let ctx = context(s, &info);
        actions
            .into_iter()
            .map(|a| {
                let logit = self.logit(&action_features(s, &info, &a, &ctx));
                (a, logit)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(a, _)| a)
            .unwrap_or(Action::Cabo)
    }
}

pub(super) fn features(s: &State, info: &Info, a: &Action) -> [f32; INPUTS] {
    action_features(s, info, a, &context(s, info))
}

fn context(s: &State, info: &Info) -> [f32; 38] {
    let me = s.actor;
    let hand = &info.means[me];
    let size = hand.len();
    let rivals: Vec<_> = info
        .sums
        .iter()
        .enumerate()
        .filter(|(p, _)| *p != me)
        .map(|(_, v)| *v)
        .collect();
    let unknown = info.values[me].iter().filter(|v| v.is_none()).count();
    let variance: f64 = info.probs[me]
        .iter()
        .zip(hand)
        .map(|(p, m)| {
            p.iter()
                .enumerate()
                .map(|(r, w)| w * (r as f64 - m).powi(2))
                .sum::<f64>()
        })
        .sum();
    // Incoming rank is always public discard or the actor's private draw; never deck top.
    let incoming = match s.stage {
        Stage::Drew(id) => s.visible(id, me),
        _ => s.discard.last().and_then(|&id| s.visible(id, me)),
    };
    let mut x = Vec::with_capacity(INPUTS);
    x.extend([
        matches!(s.stage, Stage::Drew(_)) as u8 as f64,
        s.n() as f64 / 4.,
        s.deck.len() as f64 / 52.,
        size as f64 / 8.,
        unknown as f64 / 8.,
        info.sums[me] / 52.,
        variance.sqrt() / 26.,
        hand.iter().copied().fold(13., f64::min) / 13.,
        hand.iter().copied().fold(0., f64::max) / 13.,
        rivals.iter().copied().fold(f64::INFINITY, f64::min) / 52.,
        rivals.iter().sum::<f64>() / rivals.len() as f64 / 52.,
        rivals.iter().copied().fold(0., f64::max) / 52.,
        info.lowest_prob,
        s.totals[me] as f64 / 100.,
        s.score_reset_used[me] as u8 as f64,
        (s.totals[me] - s.totals.iter().min().unwrap()) as f64 / 100.,
        (s.target as f64 - *s.totals.iter().max().unwrap() as f64) / 100.,
        s.caller.is_some() as u8 as f64,
        (s.caller == Some(me)) as u8 as f64,
        s.extra.len() as f64 / s.n() as f64,
        incoming.unwrap_or(0) as f64 / 13.,
        incoming.is_some() as u8 as f64,
        s.penalty as f64 / 10.,
        s.target as f64 / 100.,
    ]);
    for r in 0..14 {
        x.push(info.probs[me].iter().map(|p| p[r]).sum::<f64>() / size.max(1) as f64);
    }
    std::array::from_fn(|i| x[i] as f32)
}

fn action_features(s: &State, info: &Info, a: &Action, ctx: &[f32; 38]) -> [f32; INPUTS] {
    let me = s.actor;
    let hand = &info.means[me];
    let size = hand.len();
    let incoming = match s.stage {
        Stage::Drew(id) => s.visible(id, me),
        _ => s.discard.last().and_then(|&id| s.visible(id, me)),
    };
    let mut x: Vec<f64> = ctx.iter().map(|&v| v as f64).collect();
    let kind = match a {
        Action::Draw => 0,
        Action::Cabo => 1,
        Action::Exchange(_) => 2,
        Action::Replace(_) => 3,
        Action::Discard(None) => 4,
        Action::Discard(Some(PowerUse::PeekOwn { .. })) => 5,
        Action::Discard(Some(PowerUse::Spy { .. })) => 6,
        Action::Discard(Some(PowerUse::Swap { .. })) => 7,
    };
    for k in 0..8 {
        x.push((kind == k) as u8 as f64);
    }
    let slots: Vec<u8> = match a {
        Action::Exchange(g) | Action::Replace(g) => g.clone(),
        Action::Discard(Some(PowerUse::PeekOwn { slot })) => vec![*slot],
        Action::Discard(Some(PowerUse::Swap { my_slot, .. })) => vec![*my_slot],
        _ => Vec::new(),
    };
    let matching = if slots.len() < 2 {
        1.
    } else {
        (0..14)
            .map(|r| {
                slots
                    .iter()
                    .map(|&i| info.probs[me][i as usize][r])
                    .product::<f64>()
            })
            .sum()
    };
    let duplicate = incoming.map_or(0., |r| {
        1. - (0..size)
            .filter(|i| !slots.contains(&(*i as u8)))
            .map(|i| 1. - info.probs[me][i][r as usize])
            .product::<f64>()
    });
    let removed: f64 = if slots.len() < 2 {
        slots.iter().map(|&i| hand[i as usize]).sum()
    } else {
        (0..14)
            .map(|r| {
                slots.len() as f64
                    * r as f64
                    * slots
                        .iter()
                        .map(|&i| info.probs[me][i as usize][r])
                        .product::<f64>()
            })
            .sum()
    };
    let entropy: f64 = slots
        .iter()
        .map(|&i| {
            info.probs[me][i as usize]
                .iter()
                .filter(|&&p| p > 0.)
                .map(|&p| -p * p.ln())
                .sum::<f64>()
        })
        .sum();
    x.extend([
        slots.len() as f64 / 8.,
        slots.iter().map(|&i| hand[i as usize]).sum::<f64>() / 52.,
        slots
            .iter()
            .filter(|&&i| info.values[me][i as usize].is_none())
            .count() as f64
            / 8.,
        slots
            .iter()
            .map(|&i| hand[i as usize])
            .reduce(f64::min)
            .unwrap_or(0.)
            / 13.,
        slots
            .iter()
            .map(|&i| hand[i as usize])
            .reduce(f64::max)
            .unwrap_or(0.)
            / 13.,
        matching,
        duplicate,
        entropy / 8.,
        removed / 52.,
        policy::score(s, info, a).clamp(-80., 80.) / 20.,
    ]);
    let target = match a {
        Action::Discard(Some(PowerUse::Spy { player, slot }))
        | Action::Discard(Some(PowerUse::Swap { player, slot, .. })) => {
            Some((*player, *slot as usize))
        }
        _ => None,
    };
    if let Some((p, i)) = target {
        x.extend([
            info.means[p][i] / 13.,
            info.values[p][i].is_some() as u8 as f64,
            s.totals[p] as f64 / 100.,
            s.score_reset_used[p] as u8 as f64,
            info.sums[p] / 52.,
            (s.caller == Some(p)) as u8 as f64,
        ]);
    } else {
        x.extend([0.; 6]);
    }
    let pair = slots.first().map_or(0., |&i| {
        (0..size)
            .filter(|j| !slots.contains(&(*j as u8)))
            .map(|j| {
                (0..14)
                    .map(|r| info.probs[me][i as usize][r] * info.probs[me][j][r])
                    .sum::<f64>()
            })
            .sum::<f64>()
    });
    let high_pair = if size == 4 {
        (0usize..16)
            .filter(|m| m.count_ones() == 2)
            .map(|m| {
                (0..4)
                    .map(|i| info.probs[me][i][if m & (1 << i) != 0 { 12 } else { 13 }])
                    .product::<f64>()
            })
            .sum()
    } else {
        0.
    };
    x.extend([pair / 8., high_pair]);
    assert_eq!(x.len(), INPUTS);
    std::array::from_fn(|i| x[i] as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_corrupt_policy() {
        assert!(ActionPolicy::decode(b"CABOPL01").is_err());
        let mut b = b"CABOPL01".to_vec();
        b.resize(8 + WEIGHTS * 4, 0);
        b[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(ActionPolicy::decode(&b).is_err());
    }
    #[test]
    #[ignore = "requires local action-policy model and Python fixtures"]
    fn local_action_policy_parity() {
        let path = std::env::var("CABO_ACTION_POLICY_TEST")
            .unwrap_or("data/action_policy/model-v1.bin".into());
        let model = ActionPolicy::load(Path::new(&path)).unwrap();
        for line in std::fs::read_to_string(Path::new(&path).with_extension("predictions.tsv"))
            .unwrap()
            .lines()
        {
            let v: Vec<f32> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let x: [f32; 64] = v[..64].try_into().unwrap();
            assert!((model.logit(&x) - v[64]).abs() < 2e-5);
        }
    }
}
