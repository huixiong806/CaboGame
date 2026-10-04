//! Finite-deck posterior from public observations. Rank evidence is deliberately soft.
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};

use crate::game::view::{Panel, PlayerView};
use crate::game::{Pile, PowerUse, PublicEvent, RANK_COPIES};

use super::state::{keep_evidence, survival_evidence, Card, Stage, State};

#[derive(Clone, Copy, Default)]
struct Trace {
    weights: [f32; 14],
    known: u8,
}

fn evidence(view: &PlayerView) -> Vec<Vec<[f32; 14]>> {
    let n = view.all_seats.len();
    let blank = || {
        view.all_seats
            .iter()
            .map(|s| vec![[0.0; 14]; s.slots.len()])
            .collect()
    };
    let mut hands = vec![vec![Trace::default(); 4]; n];
    for event in &view.public_events {
        // Events come from the trusted engine, but synthetic fixtures may omit a prefix.
        match event {
            PublicEvent::InitialPeek { player, slots } => {
                for &s in slots {
                    let Some(t) = hands.get_mut(*player).and_then(|h| h.get_mut(s as usize)) else {
                        return blank();
                    };
                    t.known |= 1 << player;
                }
            }
            PublicEvent::Power {
                player: actor,
                power,
            } => match *power {
                PowerUse::PeekOwn { slot } => {
                    let Some(t) = hands.get_mut(*actor).and_then(|h| h.get_mut(slot as usize))
                    else {
                        return blank();
                    };
                    t.known |= 1 << actor;
                }
                PowerUse::Spy { player, slot } => {
                    let Some(t) = hands.get_mut(player).and_then(|h| h.get_mut(slot as usize))
                    else {
                        return blank();
                    };
                    t.known |= 1 << actor;
                }
                PowerUse::Swap {
                    my_slot,
                    player,
                    slot,
                } => {
                    if *actor >= n
                        || player >= n
                        || my_slot as usize >= hands[*actor].len()
                        || slot as usize >= hands[player].len()
                    {
                        return blank();
                    }
                    let a = hands[*actor][my_slot as usize];
                    hands[*actor][my_slot as usize] = hands[player][slot as usize];
                    hands[player][slot as usize] = a;
                }
            },
            PublicEvent::Discard {
                player,
                rank,
                powered: false,
            } => {
                let Some(hand) = hands.get_mut(*player) else {
                    return blank();
                };
                for t in hand {
                    if t.known & (1 << player) != 0 {
                        survival_evidence(&mut t.weights, *rank);
                    }
                }
            }
            PublicEvent::Exchange {
                player,
                source,
                slots,
                exposed,
                success,
                ..
            } => {
                let Some(hand) = hands.get_mut(*player) else {
                    return blank();
                };
                if slots.iter().any(|&s| s as usize >= hand.len()) {
                    return blank();
                }
                let all_known = slots
                    .iter()
                    .all(|&s| hand[s as usize].known & (1 << player) != 0);
                let mut new = Trace {
                    known: if *source == Pile::Discard {
                        (1 << n) - 1
                    } else {
                        1 << player
                    },
                    ..Trace::default()
                };
                if *success {
                    for &s in slots.iter().rev() {
                        hand.remove(s as usize);
                    }
                    // Selecting a known single card also says something about the known cards
                    // left behind. Ignore this when the selection was blind or a group plan.
                    if slots.len() == 1 && all_known {
                        for t in hand.iter_mut() {
                            if t.known & (1 << player) != 0 {
                                survival_evidence(&mut t.weights, exposed[0]);
                            }
                        }
                    }
                    if *source == Pile::Draw && all_known {
                        keep_evidence(&mut new.weights, exposed.iter().map(|&r| r as f32).sum());
                    }
                } else {
                    for &s in slots {
                        hand[s as usize].known = (1 << n) - 1;
                    }
                }
                hand.push(new);
            }
            _ => {}
        }
    }
    if hands
        .iter()
        .zip(&view.all_seats)
        .any(|(h, s)| h.len() != s.slots.len())
    {
        return blank();
    }
    hands
        .into_iter()
        .map(|h| h.into_iter().map(|t| t.weights).collect())
        .collect()
}

pub(super) struct Sampler {
    pub template: State,
    unknown: Vec<usize>,
    use_evidence: bool,
}

impl Sampler {
    pub fn new(view: &PlayerView, rng: &mut dyn RngCore, use_evidence: bool) -> Option<Self> {
        let me = view.me?;
        let n = view.all_seats.len();
        if !(2..=4).contains(&n) || me >= n || view.discard_ranks.len() != view.discard_count {
            return None;
        }
        let stage_draw = match view.panel {
            Panel::Idle { .. } | Panel::ConfirmCabo => None,
            Panel::Drew { rank, .. } => Some(rank),
            _ => return None,
        };
        if view.all_seats.iter().map(|s| s.slots.len()).sum::<usize>()
            + view.deck_count
            + view.discard_count
            + usize::from(stage_draw.is_some())
            != 52
        {
            return None;
        }
        let weights = evidence(view);
        let mut pool = RANK_COPIES;
        let mut cards = Vec::with_capacity(52);
        let mut unknown = Vec::new();
        let mut push =
            |rank: Option<u8>, known: u8, revealed: bool, log_weights: [f32; 14]| -> Option<u8> {
                let id = cards.len() as u8;
                if let Some(r) = rank {
                    if r > 13 || pool[r as usize] == 0 {
                        return None;
                    }
                    pool[r as usize] -= 1;
                } else {
                    unknown.push(id as usize);
                }
                cards.push(Card {
                    rank: rank.unwrap_or(0),
                    known,
                    revealed,
                    log_weights,
                });
                Some(id)
            };
        let mut hands = Vec::new();
        for (p, seat) in view.all_seats.iter().enumerate() {
            let mut hand = Vec::new();
            for (s, c) in seat.slots.iter().enumerate() {
                let mut mask = 0;
                for &who in &c.known_by {
                    if who >= n {
                        return None;
                    }
                    mask |= 1 << who;
                }
                if c.revealed {
                    mask = (1 << n) - 1;
                }
                if c.value.is_some() != (mask & (1 << me) != 0) {
                    return None;
                }
                hand.push(push(c.value, mask, c.revealed, weights[p][s])?);
            }
            hands.push(hand);
        }
        let mut discard = Vec::new();
        for &r in &view.discard_ranks {
            discard.push(push(Some(r), (1 << n) - 1, true, [0.0; 14])?);
        }
        let mut deck = Vec::new();
        for _ in 0..view.deck_count {
            deck.push(push(None, 0, false, [0.0; 14])?);
        }
        let stage = if let Some(r) = stage_draw {
            Stage::Drew(push(Some(r), 1 << me, false, [0.0; 14])?)
        } else {
            Stage::Idle
        };
        let mut leftover: Vec<u8> = pool
            .iter()
            .enumerate()
            .flat_map(|(r, &c)| std::iter::repeat_n(r as u8, c))
            .collect();
        if unknown.len() != leftover.len() {
            return None;
        }
        leftover.shuffle(rng);
        for (&id, r) in unknown.iter().zip(leftover) {
            cards[id].rank = r;
        }
        let caller = view.all_seats.iter().find(|s| s.is_caller).map(|s| s.id);
        let extra = if caller.is_some() {
            (1..=view.extra_left).map(|i| (me + i) % n).collect()
        } else {
            Default::default()
        };
        let template = State {
            cards,
            hands,
            deck,
            discard,
            actor: me,
            stage,
            caller,
            extra,
            totals: view.all_seats.iter().map(|s| s.total_score).collect(),
            round_scores: vec![0; n],
            penalty: view.cabo_penalty,
            target: view.target_score,
            public_hash: 0,
        };
        let mut sampler = Self {
            template,
            unknown,
            use_evidence,
        };
        sampler.mix(rng, sampler.unknown.len() * 40);
        Some(sampler)
    }

    fn mix(&mut self, rng: &mut dyn RngCore, steps: usize) {
        if !self.use_evidence || self.unknown.len() < 2 {
            return;
        }
        for _ in 0..steps {
            let a = self.unknown[rng.random_range(0..self.unknown.len())];
            let b = self.unknown[rng.random_range(0..self.unknown.len())];
            let ca = self.template.cards[a];
            let cb = self.template.cards[b];
            let delta = ca.log_weights[cb.rank as usize] + cb.log_weights[ca.rank as usize]
                - ca.log_weights[ca.rank as usize]
                - cb.log_weights[cb.rank as usize];
            if delta >= 0.0 || rng.random::<f32>() < delta.exp() {
                self.template.cards[a].rank = cb.rank;
                self.template.cards[b].rank = ca.rank;
            }
        }
    }

    pub fn sample(&mut self, rng: &mut dyn RngCore) -> State {
        if self.use_evidence {
            self.mix(rng, self.unknown.len() * 4);
        } else {
            let mut ranks: Vec<u8> = self
                .unknown
                .iter()
                .map(|&id| self.template.cards[id].rank)
                .collect();
            ranks.shuffle(rng);
            for (&id, r) in self.unknown.iter().zip(ranks) {
                self.template.cards[id].rank = r;
            }
        }
        self.template.clone()
    }
}
