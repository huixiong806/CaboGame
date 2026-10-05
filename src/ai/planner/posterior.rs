//! Finite-deck posterior from public observations. Rank evidence is deliberately soft.
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};

use crate::game::view::{Panel, PlayerView};
use crate::game::{Pile, PowerUse, PublicEvent, RANK_COPIES};

use super::state::call_threshold;
use super::state::{
    keep_evidence, prior_mean, survival_evidence, swap_evidence, Card, Stage, State,
};

#[derive(Clone, Copy, Default)]
struct Trace {
    weights: [f32; 14],
    known: u8,
}

fn evidence(
    view: &PlayerView,
    blind_keep: bool,
    behavioral: bool,
    call: bool,
    rule_evidence: bool,
) -> Vec<Vec<[f32; 14]>> {
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
                    let mut a = hands[*actor][my_slot as usize];
                    let mut b = hands[player][slot as usize];
                    if behavioral && !(rule_evidence && non_greedy(view, *actor, &hands[*actor])) {
                        let am = prior_mean(&a.weights);
                        let bm = prior_mean(&b.weights);
                        if a.known & (1 << actor) != 0 {
                            swap_evidence(&mut a.weights, bm, true);
                        }
                        if b.known & (1 << actor) != 0 {
                            swap_evidence(&mut b.weights, am, false);
                        }
                    }
                    hands[*actor][my_slot as usize] = b;
                    hands[player][slot as usize] = a;
                }
            },
            PublicEvent::Discard {
                player,
                rank,
                powered,
            } => {
                let Some(hand) = hands.get_mut(*player) else {
                    return blank();
                };
                if rule_evidence && non_greedy(view, *player, hand) {
                    continue;
                }
                if *powered && (!behavioral || *rank < 9) {
                    continue;
                }
                let rejected = if *powered { (rank + 2).min(13) } else { *rank };
                for t in hand {
                    if t.known & (1 << player) != 0 {
                        survival_evidence(&mut t.weights, rejected);
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
                let non_greedy = rule_evidence && non_greedy(view, *player, hand);
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
                    if slots.len() == 1 && all_known && !non_greedy {
                        for t in hand.iter_mut() {
                            if t.known & (1 << player) != 0 {
                                survival_evidence(&mut t.weights, exposed[0]);
                            }
                        }
                    }
                    if *source == Pile::Draw && all_known && !non_greedy {
                        keep_evidence(&mut new.weights, exposed.iter().map(|&r| r as f32).sum());
                    } else if blind_keep && *source == Pile::Draw && slots.len() == 1 && !non_greedy
                    {
                        // The old rank was unknown to the actor before choosing. Its newly
                        // exposed value is not evidence about why they chose the incoming card.
                        // Use a broad prior threshold, with the existing 12% lapse component.
                        keep_evidence(&mut new.weights, 7.0);
                    }
                } else {
                    for &s in slots {
                        hand[s as usize].known = (1 << n) - 1;
                    }
                }
                if *success {
                    hand.insert(slots[0] as usize, new);
                } else {
                    hand.push(new);
                }
            }
            PublicEvent::Cabo { player } if call => {
                if rule_evidence && non_greedy(view, *player, &hands[*player]) {
                    continue;
                }
                let known = hands[*player]
                    .iter()
                    .filter(|t| t.known & (1 << player) != 0)
                    .count();
                let rival_size = (0..n)
                    .filter(|&p| p != *player)
                    .map(|p| hands[p].len())
                    .min()
                    .unwrap();
                let seat = &view.all_seats[*player];
                if let Some(threshold) = call_threshold(
                    known,
                    hands[*player].len() - known,
                    rival_size,
                    seat.total_score,
                    seat.score_reset_used,
                ) {
                    for t in &mut hands[*player] {
                        if t.known & (1 << player) != 0 {
                            swap_evidence(&mut t.weights, threshold, false);
                        }
                    }
                }
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

fn non_greedy(view: &PlayerView, player: usize, hand: &[Trace]) -> bool {
    let seat = &view.all_seats[player];
    super::state::non_greedy_possible(
        seat.total_score,
        seat.score_reset_used,
        hand.len(),
        hand.iter().filter(|t| t.known & (1 << player) != 0).count(),
        view.cabo_penalty,
    )
}

pub(super) struct Sampler {
    pub template: State,
    unknown: Vec<usize>,
    use_evidence: bool,
}

impl Sampler {
    pub fn new(view: &PlayerView, rng: &mut dyn RngCore, use_evidence: bool) -> Option<Self> {
        Self::with_blind_keep(view, rng, use_evidence, false)
    }

    pub fn with_blind_keep(
        view: &PlayerView,
        rng: &mut dyn RngCore,
        use_evidence: bool,
        blind_keep: bool,
    ) -> Option<Self> {
        Self::configured(view, rng, use_evidence, blind_keep, false, false)
    }

    pub fn configured(
        view: &PlayerView,
        rng: &mut dyn RngCore,
        use_evidence: bool,
        blind_keep: bool,
        behavioral: bool,
        call: bool,
    ) -> Option<Self> {
        Self::configured_with_rules(view, rng, use_evidence, blind_keep, behavioral, call, false)
    }

    pub fn configured_with_rules(
        view: &PlayerView,
        rng: &mut dyn RngCore,
        use_evidence: bool,
        blind_keep: bool,
        behavioral: bool,
        call: bool,
        rule_evidence: bool,
    ) -> Option<Self> {
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
        let weights = evidence(view, blind_keep, behavioral, call, rule_evidence);
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
            proven_reset_trade: false,
            rollout_temperature: 0.,
            policy_random: std::array::from_fn(|_| std::cell::Cell::new(0)),
            rule_evidence,
            exchange_cycle: None,
            speculative_loss: 0.,
            rollout_call: false,
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
            score_reset_used: view.all_seats.iter().map(|s| s.score_reset_used).collect(),
            special_tactics: true,
            blind_keep_evidence: blind_keep,
            reset_policy_player: None,
            behavioral_evidence: behavioral,
            call_evidence: call,
            value_model: None,
            policy_model: None,
            rollout_policy: 0,
            policy_player: me,
            policy_actions: 12,
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
        let world = self.template.clone();
        if world.rollout_temperature > 0.
            && world.rollout_policy > 0
            && world.policy_model.is_some()
            && world.target == 100
            && world.penalty == 10
        {
            for cell in world.policy_random.iter().take(world.n()) {
                cell.set(rng.next_u64());
            }
        }
        world
    }
}
