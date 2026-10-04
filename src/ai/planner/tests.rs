use super::state::Card;
use super::*;
use crate::game::sim::make_ai_session;
use crate::game::view::project;
use crate::game::{Pending, Phase, PowerUse, Session, Settings, RANK_COPIES};
use rand::{rngs::StdRng, SeedableRng};

fn dealt(seed: u64, n: usize) -> Session {
    let mut s = make_ai_session(seed, Settings::default(), &vec!["challenger"; n]);
    s.start_game().unwrap();
    for p in 0..n {
        s.apply(p, &Command::PeekInitial { slots: [0, 1] }).unwrap();
    }
    s
}

fn exact_state(s: &Session) -> State {
    let (actor, stage) = match s.phase {
        Phase::Turn {
            current,
            pending: None,
        } => (current, Stage::Idle),
        Phase::Turn {
            current,
            pending: Some(Pending::Drew { card }),
        } => (current, Stage::Drew(card as u8)),
        _ => (0, Stage::End),
    };
    State {
        cards: s
            .cards
            .iter()
            .map(|c| Card {
                rank: c.card.rank,
                known: c.known_by.iter().fold(0, |m, &p| m | (1 << p)),
                revealed: c.revealed,
                log_weights: [0.0; 14],
            })
            .collect(),
        hands: s
            .players
            .iter()
            .map(|p| p.slots.iter().map(|&c| c as u8).collect())
            .collect(),
        deck: s.deck.iter().map(|&c| c as u8).collect(),
        discard: s.discard.iter().map(|&c| c as u8).collect(),
        totals: s.players.iter().map(|p| p.total_score).collect(),
        score_reset_used: s.players.iter().map(|p| p.score_reset_used).collect(),
        round_scores: s
            .players
            .iter()
            .map(|p| p.round_score.unwrap_or(0))
            .collect(),
        actor,
        stage,
        caller: s.cabo_caller,
        extra: s.extra_turns.clone(),
        penalty: s.settings.cabo_penalty,
        target: s.settings.target_score,
        public_hash: 0,
    }
}

fn assert_same(a: &State, b: &Session) {
    let exp = exact_state(b);
    assert_eq!(a.stage, exp.stage);
    if a.stage != Stage::End {
        assert_eq!(a.actor, exp.actor);
    }
    assert_eq!(a.hands, exp.hands);
    assert_eq!(a.deck, exp.deck);
    assert_eq!(a.discard, exp.discard);
    assert_eq!(a.totals, exp.totals);
    assert_eq!(a.score_reset_used, exp.score_reset_used);
    assert_eq!(a.round_scores, exp.round_scores);
    assert_eq!(a.caller, exp.caller);
    assert_eq!(a.extra, exp.extra);
    for (c, e) in a.cards.iter().zip(&exp.cards) {
        assert_eq!((c.rank, c.known, c.revealed), (e.rank, e.known, e.revealed));
    }
}

#[test]
fn simulator_matches_engine_step_by_step() {
    let mut rng = StdRng::seed_from_u64(701);
    let mut checked = 0;
    let mut failures = 0;
    let mut powers = [0; 3];
    for n in 2..=4 {
        for seed in 0..80 {
            let mut s = dealt(seed, n);
            let mut fast = exact_state(&s);
            for step in 0..400 {
                if fast.stage == Stage::End {
                    break;
                }
                let info = policy::Info::new(&fast, fast.actor, false);
                let mut actions = policy::candidates(&fast, &info);
                // Random play plus some delayed calls exercises empty decks and multi-card failures.
                if step < 30 {
                    actions.retain(|a| !matches!(a, Action::Cabo));
                }
                let action = &actions[rng.random_range(0..actions.len())];
                if let Action::Discard(Some(power)) = action {
                    powers[match power {
                        PowerUse::PeekOwn { .. } => 0,
                        PowerUse::Spy { .. } => 1,
                        _ => 2,
                    }] += 1;
                }
                let before_size = fast.hands[fast.actor].len();
                let p = fast.actor;
                s.apply(p, &action.command()).unwrap();
                fast.apply(action);
                if fast.hands[p].len() > before_size {
                    failures += 1;
                }
                assert_same(&fast, &s);
                checked += 1;
            }
        }
    }
    assert!(checked > 8000, "checked {checked}");
    assert!(failures > 100);
    assert!(powers.iter().all(|&n| n > 10), "powers={powers:?}");
}

#[test]
fn simulator_resolves_last_draw_before_scoring() {
    for seed in 0..24 {
        let mut s = dealt(seed, 3);
        let mut fast = exact_state(&s);
        while !s.deck.is_empty() {
            let me = s.actor().unwrap();
            s.apply(me, &Command::BeginDraw).unwrap();
            fast.apply(&Action::Draw);
            assert_same(&fast, &s);
            let Stage::Drew(id) = fast.stage else {
                unreachable!()
            };
            let power = if s.deck.is_empty() {
                match fast.cards[id as usize].rank {
                    7..=8 => Some(PowerUse::PeekOwn { slot: 2 }),
                    9..=10 => Some(PowerUse::Spy {
                        player: (me + 1) % 3,
                        slot: 2,
                    }),
                    11..=12 => Some(PowerUse::Swap {
                        my_slot: 1,
                        player: (me + 1) % 3,
                        slot: 3,
                    }),
                    _ => None,
                }
            } else {
                None
            };
            let action = Action::Discard(power);
            s.apply(me, &action.command()).unwrap();
            fast.apply(&action);
            assert_same(&fast, &s);
        }
        assert_eq!(fast.stage, Stage::End);
    }
}

#[test]
fn posterior_preserves_all_observed_facts_and_deck_counts() {
    let mut rng = StdRng::seed_from_u64(733);
    let bot = ChallengerBot;
    let mut checks = 0;
    for seed in 0..6 {
        let mut s = dealt(seed, 4);
        for _ in 0..150 {
            let Some(actor) = s.actor() else { break };
            let v = project(&s, Some(actor), 0);
            let mut sampler = Sampler::new(&v, &mut rng, true).unwrap();
            for _ in 0..8 {
                let world = sampler.sample(&mut rng);
                let mut counts = [0usize; 14];
                for c in &world.cards {
                    counts[c.rank as usize] += 1;
                }
                assert_eq!(counts, RANK_COPIES);
                assert_eq!(world.extra, s.extra_turns);
                assert_eq!(
                    world
                        .discard
                        .iter()
                        .map(|&id| world.cards[id as usize].rank)
                        .collect::<Vec<_>>(),
                    v.discard_ranks
                );
                for (p, seat) in v.all_seats.iter().enumerate() {
                    for (i, slot) in seat.slots.iter().enumerate() {
                        let card = world.cards[world.hands[p][i] as usize];
                        if let Some(r) = slot.value {
                            assert_eq!(card.rank, r);
                        }
                        assert_eq!(
                            card.known,
                            slot.known_by.iter().fold(0, |m, &p| m | (1 << p))
                        );
                    }
                }
                checks += 1;
            }
            s.apply(actor, &bot.decide(&v, &mut rng)).unwrap();
        }
    }
    assert!(checks > 200);
}

#[test]
fn hidden_values_do_not_change_observation_policy_or_decision() {
    let s = dealt(16, 4);
    let p = s.actor().unwrap();
    let mut a = exact_state(&s);
    let mut b = a.clone();
    let unseen: Vec<usize> = a
        .cards
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.revealed && c.known & (1 << p) == 0)
        .map(|(id, _)| id)
        .collect();
    let ranks: Vec<u8> = unseen.iter().map(|&id| a.cards[id].rank).collect();
    for (&id, &rank) in unseen.iter().zip(ranks.iter().rev()) {
        b.cards[id].rank = rank;
    }
    assert_eq!(a.observation_key(p), b.observation_key(p));
    assert_eq!(policy::ranked(&a, 128), policy::ranked(&b, 128));
    for style in 0..4 {
        assert_eq!(policy::choose(&a, style), policy::choose(&b, style));
    }
    // Identical views and RNG seeds yield identical complete search reports, even with different ground truth.
    let mut other = s.clone();
    for &id in &unseen {
        other.cards[id].card.rank = b.cards[id].rank;
    }
    let bot = PlannerBot::new(PlannerCfg {
        budget_us: 0,
        simulations: 64,
        ..PlannerCfg::default()
    });
    let r1 = bot.analyze(&project(&s, Some(p), 0), &mut StdRng::seed_from_u64(7));
    let r2 = bot.analyze(&project(&other, Some(p), 0), &mut StdRng::seed_from_u64(7));
    assert_eq!(r1.command, r2.command);
    assert_eq!(r1.simulations, r2.simulations);
    assert_eq!(
        r1.candidates
            .iter()
            .map(|c| (c.visits, c.mean))
            .collect::<Vec<_>>(),
        r2.candidates
            .iter()
            .map(|c| (c.visits, c.mean))
            .collect::<Vec<_>>()
    );
    // Secret draw outcomes must split the next information node.
    a.apply(&Action::Draw);
    b.apply(&Action::Draw);
    if let (Stage::Drew(x), Stage::Drew(y)) = (a.stage, b.stage) {
        if a.cards[x as usize].rank != b.cards[y as usize].rank {
            assert_ne!(a.observation_key(p), b.observation_key(p));
        }
    }
}

/// Build legal card containers with controlled hands and match totals.
fn scoring_fixture(hands: &[Vec<u8>], totals: &[u32], used: &[bool]) -> Session {
    let mut s = dealt(901, hands.len());
    let mut pool: Vec<u16> = (0..52).collect();
    for (p, ranks) in hands.iter().enumerate() {
        s.players[p].slots = ranks
            .iter()
            .map(|&rank| {
                let i = pool
                    .iter()
                    .position(|&id| s.cards[id as usize].card.rank == rank)
                    .unwrap();
                pool.remove(i)
            })
            .collect();
        s.players[p].total_score = totals[p];
        s.players[p].score_reset_used = used[p];
    }
    s.discard = vec![pool.pop().unwrap()];
    s.deck = pool;
    // These fixtures represent a fully observed state, not a secret oracle for decisions.
    for card in &mut s.cards {
        card.revealed = true;
        card.known_by = (0..hands.len()).collect();
    }
    s.phase = Phase::Turn {
        current: 0,
        pending: None,
    };
    s
}

#[test]
fn simulator_matches_special_settlements_and_terminal_utility() {
    for n in 2..=4 {
        for caller in 0..n {
            for exhausted in [false, true] {
                for used in [false, true] {
                    let mut hands = vec![vec![1], vec![2], vec![3], vec![4]];
                    hands.truncate(n);
                    hands[0] = vec![13, 12, 13, 12];
                    let mut s = scoring_fixture(&hands, &vec![50; n], &vec![used; n]);
                    s.phase = Phase::Turn {
                        current: caller,
                        pending: None,
                    };
                    let mut fast = exact_state(&s);
                    if !exhausted {
                        s.apply(caller, &Command::CallCabo).unwrap();
                        fast.apply(&Action::Cabo);
                        assert_same(&fast, &s);
                    }
                    while let Some(actor) = s.actor() {
                        s.apply(actor, &Command::BeginDraw).unwrap();
                        fast.apply(&Action::Draw);
                        assert_same(&fast, &s);
                        s.apply(actor, &Command::DiscardDrawn { power: None })
                            .unwrap();
                        fast.apply(&Action::Discard(None));
                        assert_same(&fast, &s);
                    }
                    assert_eq!(fast.round_scores[0], 0);
                    assert!(fast.round_scores[1..].iter().all(|&v| v == 50));
                    assert!(fast.totals[1..]
                        .iter()
                        .all(|&v| v == if used { 100 } else { 50 }));
                    assert_eq!(matches!(s.phase, Phase::GameOver { .. }), used);
                    let expected_utility = if used { 1.0 } else { 1.0 / n as f64 };
                    assert!((fast.utility(0) - expected_utility).abs() < 1e-12);
                }
            }
        }
    }
    for used in [false, true] {
        let hands = vec![vec![5, 5, 0, 0], vec![2, 2, 3, 4], vec![1, 1, 1, 1]];
        let mut s = scoring_fixture(&hands, &[80, 90, 95], &[used, false, false]);
        let mut fast = exact_state(&s);
        s.apply(0, &Command::CallCabo).unwrap();
        fast.apply(&Action::Cabo);
        while let Some(actor) = s.actor() {
            s.apply(actor, &Command::BeginDraw).unwrap();
            fast.apply(&Action::Draw);
            s.apply(actor, &Command::DiscardDrawn { power: None })
                .unwrap();
            fast.apply(&Action::Discard(None));
            assert_same(&fast, &s);
        }
        assert_eq!(fast.utility(0), if used { 0.0 } else { 1.0 });
    }
}

#[test]
fn reset_entitlement_survives_both_beliefs_and_changes_information_key() {
    let mut s = dealt(903, 3);
    let me = s.actor().unwrap();
    s.players[1].score_reset_used = true;
    let view = project(&s, Some(me), 0);
    let mut rng = StdRng::seed_from_u64(77);
    let sampler = Sampler::new(&view, &mut rng, false).unwrap();
    assert_eq!(sampler.template.score_reset_used, [false, true, false]);
    let old = crate::ai::belief::reconstruct(&view, &mut rng, &[0; 14]).unwrap();
    assert_eq!(
        old.players
            .iter()
            .map(|p| p.score_reset_used)
            .collect::<Vec<_>>(),
        [false, true, false]
    );
    let mut other = sampler.template.clone();
    other.score_reset_used[1] = false;
    assert_ne!(
        other.observation_key(me),
        sampler.template.observation_key(me)
    );
}

#[test]
fn replacement_evidence_follows_the_replaced_slot() {
    let mut s = dealt(27, 3);
    let p = s.actor().unwrap();
    s.apply(p, &Command::BeginDraw).unwrap();
    s.apply(p, &Command::DrawSwap { slots: vec![1] }).unwrap();
    let view = project(&s, Some(s.actor().unwrap()), 0);
    let sampler = Sampler::new(&view, &mut StdRng::seed_from_u64(91), true).unwrap();
    let world = sampler.template;
    let weights = |slot: usize| world.cards[world.hands[p][slot] as usize].log_weights;
    assert!(weights(1).iter().any(|&w| w != 0.0), "保留新牌的行动证据跟随新牌");
    assert_eq!(weights(2), [0.0; 14], "未选的未知牌不继承新牌证据");
    assert_eq!(weights(3), [0.0; 14]);
}

#[test]
fn candidate_set_includes_group_exchange_and_known_low_power_target() {
    let mut s = dealt(2, 3);
    let me = s.actor().unwrap();
    s.apply(me, &Command::BeginDraw).unwrap();
    let mut fast = exact_state(&s);
    let Stage::Drew(d) = fast.stage else {
        unreachable!()
    };
    fast.cards[d as usize].rank = 1;
    for i in 0..2 {
        let id = fast.hands[me][i] as usize;
        fast.cards[id].rank = 8;
        fast.cards[id].known |= 1 << me;
    }
    // Use unpruned candidate generation for this deliberately nonconserving tactical fixture.
    let info = policy::Info {
        values: vec![vec![Some(8), Some(8), None, None]; 3],
        means: vec![vec![8.0, 8.0, 6.0, 6.0]; 3],
        probs: vec![vec![[1.0 / 14.0; 14]; 4]; 3],
        sums: vec![28.0; 3],
        lowest_prob: 0.0,
    };
    assert!(policy::candidates(&fast, &info).contains(&Action::Replace(vec![0, 1])));
    fast.cards[d as usize].rank = 11;
    let target = (me + 1) % 3;
    assert!(
        policy::candidates(&fast, &info).contains(&Action::Discard(Some(PowerUse::Swap {
            my_slot: 2,
            player: target,
            slot: 0
        })))
    );
}

#[test]
fn new_bots_play_strictly_legal_games_without_fallbacks() {
    let bot = PlannerBot::new(PlannerCfg {
        budget_us: 0,
        simulations: 24,
        ..PlannerCfg::default()
    });
    for n in 2..=4 {
        for seed in 0..3 {
            let mut s = dealt(seed, n);
            let mut rng = StdRng::seed_from_u64(seed + 91);
            for _ in 0..2000 {
                match s.phase {
                    Phase::Turn { current, .. } => {
                        let v = project(&s, Some(current), 0);
                        let c = if current == 0 {
                            bot.decide(&v, &mut rng)
                        } else {
                            ChallengerBot.decide(&v, &mut rng)
                        };
                        s.apply(current, &c)
                            .unwrap_or_else(|e| panic!("{e}: {c:?} panel {:?}", v.panel));
                    }
                    Phase::RoundEnd => {
                        s.next_round().unwrap();
                        for p in 0..n {
                            s.apply(p, &Command::PeekInitial { slots: [0, 1] }).unwrap();
                        }
                    }
                    Phase::GameOver { .. } => break,
                    _ => unreachable!(),
                }
            }
            assert!(matches!(s.phase, Phase::GameOver { .. }));
        }
    }
}

#[test]
fn planner_steals_a_known_zero_in_final_response() {
    let mut s = dealt(922, 2);
    let me = s.actor().unwrap();
    let other = 1 - me;
    s.apply(me, &Command::BeginDraw).unwrap();
    let Phase::Turn {
        pending: Some(Pending::Drew { card: drawn }),
        ..
    } = s.phase
    else {
        unreachable!()
    };
    // Build a conserving tactical fixture: all hand ranks public, only the remaining deck hidden.
    let mut remaining = RANK_COPIES;
    let mut assigned = vec![false; 52];
    for (p, ranks) in [(me, [13, 8, 8, 0]), (other, [0, 1, 2, 3])] {
        for (&id, r) in s.players[p].slots.iter().zip(ranks) {
            s.cards[id as usize].card.rank = r;
            s.cards[id as usize].known_by = (0..2).collect();
            assigned[id as usize] = true;
            remaining[r as usize] -= 1;
        }
        s.players[p].total_score = 86;
    }
    s.cards[drawn as usize].card.rank = 11;
    assigned[drawn as usize] = true;
    remaining[11] -= 1;
    let mut rest = remaining
        .iter()
        .enumerate()
        .flat_map(|(r, &n)| std::iter::repeat_n(r as u8, n));
    for (id, c) in s.cards.iter_mut().enumerate() {
        if !assigned[id] {
            c.card.rank = rest.next().unwrap();
        }
    }
    s.cabo_caller = Some(other);
    s.extra_turns.clear();
    let bot = PlannerBot::new(PlannerCfg {
        budget_us: 0,
        simulations: 512,
        ..PlannerCfg::default()
    });
    let report = bot.analyze(&project(&s, Some(me), 0), &mut StdRng::seed_from_u64(21));
    assert_eq!(
        report.command,
        Command::DiscardDrawn {
            power: Some(PowerUse::Swap {
                my_slot: 0,
                player: other,
                slot: 0
            })
        }
    );
    s.apply(me, &report.command).unwrap();
    assert!(matches!(s.phase,Phase::GameOver { winners } if winners == vec![me]));
}

#[test]
fn planner_budget_and_large_hands_are_bounded() {
    let mut s = dealt(42, 4);
    let me = s.actor().unwrap();
    // Deliberately fail a legal exchange to create a fifth card and public exposures.
    s.apply(
        me,
        &Command::SwapOnce {
            slots: vec![0, 1, 2, 3],
        },
    )
    .unwrap();
    assert_eq!(s.players[me].slots.len(), 5);
    // Advance the other seats with complete draws, then exercise the large-hand root.
    while s.actor() != Some(me) {
        let p = s.actor().unwrap();
        s.apply(p, &Command::BeginDraw).unwrap();
        s.apply(p, &Command::DiscardDrawn { power: None }).unwrap();
    }
    let view = project(&s, Some(me), 0);
    let bot = PlannerBot::new(PlannerCfg {
        budget_us: 10_000,
        ..PlannerCfg::default()
    });
    let report = bot.analyze(&view, &mut StdRng::seed_from_u64(23));
    assert!(report.belief_ok);
    assert!(report.elapsed_us < 200_000, "{}us", report.elapsed_us);
    s.apply(me, &report.command).unwrap();
}

#[test]
fn terminal_utility_respects_ties_and_configured_target() {
    let s = dealt(32, 3);
    let mut fast = exact_state(&s);
    fast.totals = vec![50, 50, 61];
    fast.target = 60;
    assert_eq!(fast.utility(0), 0.5);
    assert_eq!(fast.utility(1), 0.5);
    assert_eq!(fast.utility(2), 0.0);
    fast.target = 100;
    let sum: f64 = (0..3).map(|p| fast.utility(p)).sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn posterior_preserves_uncertainty_in_own_two_unseen_cards() {
    let s = dealt(7017, 2);
    let me = s.actor().unwrap();
    let view = project(&s, Some(me), 0);
    assert_eq!(
        view.me_seat
            .as_ref()
            .unwrap()
            .slots
            .iter()
            .filter(|c| c.value.is_none())
            .count(),
        2
    );
    let mut rng = StdRng::seed_from_u64(17);
    let mut sampler = Sampler::new(&view, &mut rng, true).unwrap();
    let (mut sum, mut squares) = (0.0, 0.0);
    for _ in 0..256 {
        let world = sampler.sample(&mut rng);
        let total = world.hands[me]
            .iter()
            .map(|&id| world.cards[id as usize].rank as f64)
            .sum::<f64>();
        sum += total;
        squares += total * total;
    }
    assert!(squares / 256.0 - (sum / 256.0).powi(2) > 1.0);
}

#[test]
#[ignore = "diagnostic: posterior calibration against independently observed hidden state"]
fn posterior_calibration() {
    let mut rng = StdRng::seed_from_u64(811);
    let mut errors = [0.0; 2];
    let mut checks = 0;
    let mut bins = [[0.0; 3]; 5];
    for seed in 0..40 {
        let mut s = dealt(8100 + seed, 4);
        for step in 0..160 {
            let Some(me) = s.actor() else { break };
            let view = project(&s, Some(me), 0);
            if step % 4 == 0 && matches!(view.panel, Panel::Idle { can_cabo: true, .. }) {
                let mut sampler = Sampler::new(&view, &mut rng, true).unwrap();
                let true_state = exact_state(&s);
                let actual_sum: Vec<u32> = true_state
                    .hands
                    .iter()
                    .map(|h| {
                        h.iter()
                            .map(|&id| true_state.cards[id as usize].rank as u32)
                            .sum()
                    })
                    .collect();
                let truth = actual_sum[me]
                    < (0..4)
                        .filter(|&p| p != me)
                        .map(|p| actual_sum[p])
                        .min()
                        .unwrap();
                let mut estimates = vec![0.0; 4];
                let mut wins = 0;
                for _ in 0..64 {
                    let world = sampler.sample(&mut rng);
                    let sums: Vec<u32> = world
                        .hands
                        .iter()
                        .map(|h| {
                            h.iter()
                                .map(|&id| world.cards[id as usize].rank as u32)
                                .sum()
                        })
                        .collect();
                    for p in 0..4 {
                        estimates[p] += sums[p] as f64 / 64.0;
                    }
                    wins += usize::from(
                        sums[me] < (0..4).filter(|&p| p != me).map(|p| sums[p]).min().unwrap(),
                    );
                }
                for p in 0..4 {
                    if p != me {
                        let d = estimates[p] - actual_sum[p] as f64;
                        errors[0] += d;
                        errors[1] += d.abs();
                        checks += 1;
                    }
                }
                let prob = wins as f64 / 64.0;
                let b = ((prob * 5.0) as usize).min(4);
                bins[b][0] += 1.0;
                bins[b][1] += prob;
                bins[b][2] += truth as u8 as f64;
            }
            s.apply(me, &ChallengerBot.decide(&view, &mut rng)).unwrap();
        }
    }
    println!(
        "posterior checks={checks} mean_bias={:.3} MAE={:.3}",
        errors[0] / checks as f64,
        errors[1] / checks as f64
    );
    for b in bins {
        if b[0] > 0.0 {
            println!(
                "n={} predicted={:.3} actual={:.3}",
                b[0],
                b[1] / b[0],
                b[2] / b[0]
            );
        }
    }
}
