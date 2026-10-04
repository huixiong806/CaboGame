//! Compact round simulator. Planning uses only actions expressible by the real engine.
use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

use crate::game::{Command, Pile, PowerUse, PublicEvent, RANK_COPIES};

#[derive(Clone, Copy, Debug)]
pub(super) struct Card {
    pub rank: u8,
    pub known: u8,
    pub revealed: bool,
    /// Public-action soft evidence; never fitted using a hidden rank.
    pub log_weights: [f32; 14],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Stage {
    Idle,
    Drew(u8),
    End,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Action {
    Draw,
    Exchange(Vec<u8>),
    Replace(Vec<u8>),
    Discard(Option<PowerUse>),
    Cabo,
}

impl Action {
    pub fn command(&self) -> Command {
        match self {
            Self::Draw => Command::BeginDraw,
            Self::Exchange(slots) => Command::SwapOnce {
                slots: slots.clone(),
            },
            Self::Replace(slots) => Command::DrawSwap {
                slots: slots.clone(),
            },
            Self::Discard(power) => Command::DiscardDrawn { power: *power },
            Self::Cabo => Command::CallCabo,
        }
    }
}

#[derive(Clone)]
pub(super) struct State {
    pub cards: Vec<Card>,
    pub hands: Vec<Vec<u8>>,
    pub deck: Vec<u8>,
    pub discard: Vec<u8>,
    pub totals: Vec<u32>,
    pub round_scores: Vec<u32>,
    pub score_reset_used: Vec<bool>,
    /// Strategy ablation only; settlement rules are always enabled.
    pub special_tactics: bool,
    pub actor: usize,
    pub stage: Stage,
    pub caller: Option<usize>,
    pub extra: VecDeque<usize>,
    pub penalty: u32,
    pub target: u32,
    pub public_hash: u64,
}

/// A broad likelihood, with a lapse component to avoid asserting human rationality.
pub(super) fn keep_evidence(weights: &mut [f32; 14], removed: f32) {
    for (r, w) in weights.iter_mut().enumerate() {
        *w += (0.12 + 0.88 / (1.0 + ((r as f32 - removed) / 1.8).exp())).ln();
    }
    normalize(weights);
}

pub(super) fn survival_evidence(weights: &mut [f32; 14], rejected: u8) {
    for (r, w) in weights.iter_mut().enumerate() {
        *w += (0.3 + 0.7 / (1.0 + ((r as f32 - rejected as f32) / 2.0).exp())).ln();
    }
    normalize(weights);
}

fn normalize(weights: &mut [f32; 14]) {
    let top = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for w in weights {
        *w = (*w - top).max(-12.0);
    }
}

impl State {
    pub fn n(&self) -> usize {
        self.hands.len()
    }
    pub fn all_known(&self) -> u8 {
        (1 << self.n()) - 1
    }
    pub fn visible(&self, cid: u8, viewer: usize) -> Option<u8> {
        let c = self.cards[cid as usize];
        if c.revealed || c.known & (1 << viewer) != 0 {
            Some(c.rank)
        } else {
            None
        }
    }

    fn event(&mut self, event: PublicEvent) {
        let mut h = DefaultHasher::new();
        self.public_hash.hash(&mut h);
        event.hash(&mut h);
        self.public_hash = h.finish();
    }

    pub fn apply(&mut self, action: &Action) {
        let p = self.actor;
        match action {
            Action::Draw => {
                assert_eq!(self.stage, Stage::Idle);
                let cid = self.deck.pop().expect("nonempty deck");
                self.cards[cid as usize].known |= 1 << p;
                self.stage = Stage::Drew(cid);
                // Drawing is public; its rank is not.
                let mut h = DefaultHasher::new();
                (self.public_hash, p, "draw").hash(&mut h);
                self.public_hash = h.finish();
            }
            Action::Exchange(slots) | Action::Replace(slots) => {
                let source = if matches!(action, Action::Exchange(_)) {
                    Pile::Discard
                } else {
                    Pile::Draw
                };
                let taken = match (source, self.stage) {
                    (Pile::Discard, Stage::Idle) => self.discard.pop().expect("nonempty discard"),
                    (Pile::Draw, Stage::Drew(cid)) => cid,
                    _ => panic!("wrong exchange stage"),
                };
                let mut slots = slots.clone();
                slots.sort_unstable();
                slots.dedup();
                assert!(!slots.is_empty());
                let ranks: Vec<u8> = slots
                    .iter()
                    .map(|&s| self.cards[self.hands[p][s as usize] as usize].rank)
                    .collect();
                let success = ranks.iter().all(|&r| r == ranks[0]);
                let incoming = if source == Pile::Discard {
                    Some(self.cards[taken as usize].rank)
                } else {
                    None
                };
                let previously_known = slots
                    .iter()
                    .all(|&s| self.cards[self.hands[p][s as usize] as usize].known & (1 << p) != 0);
                if success {
                    // This order matters: the smallest selected slot becomes discard top.
                    for &s in slots.iter().rev() {
                        let old = self.hands[p].remove(s as usize);
                        self.cards[old as usize].known = self.all_known();
                        self.cards[old as usize].revealed = true;
                        self.discard.push(old);
                    }
                    if slots.len() == 1 && previously_known {
                        for &id in &self.hands[p] {
                            if self.cards[id as usize].known & (1 << p) != 0 {
                                survival_evidence(
                                    &mut self.cards[id as usize].log_weights,
                                    ranks[0],
                                );
                            }
                        }
                    }
                } else {
                    for &s in &slots {
                        let cid = self.hands[p][s as usize];
                        self.cards[cid as usize].known = self.all_known();
                        self.cards[cid as usize].revealed = true;
                    }
                }
                self.cards[taken as usize].revealed = false;
                if source == Pile::Draw && success && previously_known {
                    keep_evidence(
                        &mut self.cards[taken as usize].log_weights,
                        ranks.iter().map(|&r| r as f32).sum(),
                    );
                }
                if success {
                    self.hands[p].insert(slots[0] as usize, taken);
                } else {
                    self.hands[p].push(taken);
                }
                self.event(PublicEvent::Exchange {
                    player: p,
                    source,
                    slots,
                    exposed: ranks,
                    incoming,
                    success,
                });
                self.finish_turn();
            }
            Action::Discard(power) => {
                let Stage::Drew(cid) = self.stage else {
                    panic!("must draw first")
                };
                if let Some(power) = power {
                    match *power {
                        PowerUse::PeekOwn { slot } => {
                            let id = self.hands[p][slot as usize];
                            self.cards[id as usize].known |= 1 << p;
                        }
                        PowerUse::Spy { player, slot } => {
                            let id = self.hands[player][slot as usize];
                            self.cards[id as usize].known |= 1 << p;
                        }
                        PowerUse::Swap {
                            my_slot,
                            player,
                            slot,
                        } => {
                            let a = self.hands[p][my_slot as usize];
                            let b = self.hands[player][slot as usize];
                            self.hands[p][my_slot as usize] = b;
                            self.hands[player][slot as usize] = a;
                        }
                    }
                    self.event(PublicEvent::Power {
                        player: p,
                        power: *power,
                    });
                } else {
                    let r = self.cards[cid as usize].rank;
                    for &id in &self.hands[p] {
                        if self.cards[id as usize].known & (1 << p) != 0 {
                            survival_evidence(&mut self.cards[id as usize].log_weights, r);
                        }
                    }
                }
                self.cards[cid as usize].known = self.all_known();
                self.cards[cid as usize].revealed = true;
                self.discard.push(cid);
                self.event(PublicEvent::Discard {
                    player: p,
                    rank: self.cards[cid as usize].rank,
                    powered: power.is_some(),
                });
                self.finish_turn();
            }
            Action::Cabo => {
                assert_eq!(self.stage, Stage::Idle);
                assert!(self.caller.is_none());
                self.caller = Some(p);
                self.extra = (1..self.n()).map(|i| (p + i) % self.n()).collect();
                self.event(PublicEvent::Cabo { player: p });
                self.finish_turn();
            }
        }
    }

    fn finish_turn(&mut self) {
        if self.deck.is_empty() {
            self.end_round();
            return;
        }
        if self.caller.is_some() {
            if let Some(next) = self.extra.pop_front() {
                self.actor = next;
            } else {
                self.end_round();
                return;
            }
        } else {
            self.actor = (self.actor + 1) % self.n();
        }
        self.stage = Stage::Idle;
    }

    pub fn end_round(&mut self) {
        if self.stage == Stage::End {
            return;
        }
        let hands: Vec<Vec<u8>> = self
            .hands
            .iter()
            .map(|h| h.iter().map(|&id| self.cards[id as usize].rank).collect())
            .collect();
        let result = crate::game::scoring::settle(
            &hands,
            &self.totals,
            &self.score_reset_used,
            self.caller,
            self.penalty,
        );
        self.round_scores = result.scores;
        self.totals = result.totals;
        self.score_reset_used = result.reset_used;
        for p in 0..self.n() {
            for &id in &self.hands[p] {
                self.cards[id as usize].known = (1 << self.hands.len()) - 1;
                self.cards[id as usize].revealed = true;
            }
        }
        self.stage = Stage::End;
    }

    pub fn utility(&self, me: usize) -> f64 {
        self.utility_for(&self.totals, me)
    }

    pub fn utility_for(&self, totals: &[u32], me: usize) -> f64 {
        let max = *totals.iter().max().unwrap();
        if max >= self.target {
            let min = *totals.iter().min().unwrap();
            return if totals[me] == min {
                1.0 / totals.iter().filter(|&&t| t == min).count() as f64
            } else {
                0.0
            };
        }
        // Approximate future-round uncertainty. Multiseat normalization: equal totals => 1/n.
        let temp = (8.0 * ((self.target - max) as f64 / 8.0).sqrt()).max(6.0);
        let min = *totals.iter().min().unwrap() as f64;
        let weights: Vec<f64> = totals
            .iter()
            .map(|&s| (-(s as f64 - min) / temp).exp())
            .collect();
        weights[me] / weights.iter().sum::<f64>()
    }

    /// Only observable state and observable history enter the information-set key.
    pub fn observation_key(&self, me: usize) -> u64 {
        let mut h = DefaultHasher::new();
        (
            self.public_hash,
            self.actor,
            self.caller,
            self.deck.len(),
            &self.extra,
            &self.totals,
            &self.score_reset_used,
        )
            .hash(&mut h);
        for hand in &self.hands {
            hand.len().hash(&mut h);
            for &id in hand {
                let c = self.cards[id as usize];
                (self.visible(id, me), c.known, c.revealed).hash(&mut h);
            }
        }
        self.discard
            .iter()
            .map(|&id| self.cards[id as usize].rank)
            .collect::<Vec<_>>()
            .hash(&mut h);
        match self.stage {
            Stage::Idle => 0u8.hash(&mut h),
            Stage::Drew(id) => (1u8, self.visible(id, me)).hash(&mut h),
            Stage::End => 2u8.hash(&mut h),
        }
        h.finish()
    }

    pub fn pool(&self, viewer: usize) -> [u8; 14] {
        let mut pool = RANK_COPIES.map(|n| n as u8);
        for hand in &self.hands {
            for &id in hand {
                if let Some(r) = self.visible(id, viewer) {
                    pool[r as usize] -= 1;
                }
            }
        }
        for &id in &self.discard {
            pool[self.cards[id as usize].rank as usize] -= 1;
        }
        if let Stage::Drew(id) = self.stage {
            if let Some(r) = self.visible(id, viewer) {
                pool[r as usize] -= 1;
            }
        }
        pool
    }
}
