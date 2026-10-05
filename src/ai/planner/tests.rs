use super::state::Card;
use super::*;
use crate::game::sim::make_ai_session;
use crate::game::view::project;
use crate::game::{Pending, Phase, PowerUse, PublicEvent, Session, Settings, RANK_COPIES};
use rand::{rngs::StdRng, SeedableRng};

fn dealt(seed: u64, n: usize) -> Session {
    let mut s = make_ai_session(seed, Settings::default(), &vec!["challenger"; n]);
    s.start_game().unwrap();
    for p in 0..n {
        s.apply(p, &Command::PeekInitial { slots: [0, 1] }).unwrap();
    }
    s
}

#[test]
#[ignore = "offline audit of speculative exchanges on complete real-engine rounds"]
fn audit_speculative_exchanges() {
    let mut cfg = PlannerCfg {
        proven_reset_trade: false,
        budget_us: 0,
        simulations: 256,
        confirmation_samples: 64,
        ..PlannerCfg::hard_with_model(None)
    };
    for pair in std::env::var("CABO_AUDIT_CFG")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
    {
        let (k, v) = pair.split_once('=').unwrap();
        assert!(cfg.set(k, v));
    }
    let bot = PlannerBot::new(cfg);
    let mut decisions = 0;
    let mut groups = 0;
    let mut failed = 0;
    for seed in 820000..820100 {
        let mut s = dealt(seed, 4);
        for (p, score) in s.players.iter_mut().zip([69, 57, 62, 26]) {
            p.total_score = score;
        }
        let mut rng = StdRng::seed_from_u64(seed ^ 18371);
        for _ in 0..2000 {
            let Some(p) = s.actor() else { break };
            let view = project(&s, Some(p), 0);
            let report = bot.analyze(&view, &mut rng);
            decisions += 1;
            if let Command::SwapOnce { slots } | Command::DrawSwap { slots } = &report.command {
                if slots.len() > 1 {
                    groups += 1;
                    let known: Vec<_> = slots
                        .iter()
                        .filter_map(|&i| view.all_seats[p].slots[i as usize].value)
                        .collect();
                    assert!(
                        known.windows(2).all(|r| r[0] == r[1]),
                        "AI selected a known mismatch"
                    );
                    let actual: Vec<_> = slots
                        .iter()
                        .map(|&i| s.cards[s.players[p].slots[i as usize] as usize].card.rank)
                        .collect();
                    if !actual.windows(2).all(|r| r[0] == r[1]) {
                        failed += 1;
                        let incoming = match view.panel {
                            Panel::Drew { rank, .. } => Some(rank),
                            _ => view.discard_ranks.last().copied(),
                        };
                        eprintln!("AUDIT seed={seed} p={p} visible={:?} command={:?} incoming={incoming:?} sims={} confirm={} gain={:.4} se={:.4} truth_after_choice={actual:?}", view.all_seats[p].slots.iter().map(|c| c.value).collect::<Vec<_>>(), report.command, report.simulations, report.confirmations, report.confirmed_gain, report.confirmed_se);
                    }
                }
            }
            s.apply(p, &report.command).unwrap();
        }
        assert!(matches!(s.phase, Phase::RoundEnd | Phase::GameOver { .. }));
    }
    eprintln!("AUDIT decisions={decisions} groups={groups} failures={failed}");
}

#[test]
fn speculative_group_filter_is_observable_and_preserves_known_matches() {
    let s = dealt(830001, 4);
    let me = s.actor().unwrap();
    let view = project(&s, Some(me), 0);
    let mut sampler = Sampler::new(&view, &mut StdRng::seed_from_u64(19), true).unwrap();
    let mut state = sampler.sample(&mut StdRng::seed_from_u64(23));
    state.speculative_loss = 2.;
    // Nonconserving, candidate-only fixture: no unseen rank is ever read.
    let incoming = *state.discard.last().unwrap() as usize;
    state.cards[incoming].rank = 9;
    let mut info = policy::Info::new(&sampler.template, me, false);
    info.values[me] = vec![Some(12), Some(12), None, None];
    let actions = policy::candidates(&state, &info);
    assert!(actions.contains(&Action::Exchange(vec![0, 1])));
    assert!(!actions.contains(&Action::Exchange(vec![2, 3])));
    for id in 0..state.cards.len() {
        if state.visible(id as u8, me).is_none() {
            state.cards[id].rank = ((id * 3 + 7) % 14) as u8;
        }
    }
    assert_eq!(actions, policy::candidates(&state, &info));
    state.cards[incoming].rank = 0;
    assert!(policy::candidates(&state, &info).contains(&Action::Exchange(vec![2, 3])));
    state.speculative_loss = 0.;
    state.cards[incoming].rank = 9;
    assert!(policy::candidates(&state, &info).contains(&Action::Exchange(vec![2, 3])));
}

#[test]
fn risk_and_response_search_use_only_legal_information() {
    for root_racing in [false, true] {
        let bot = PlannerBot::new(PlannerCfg {
            budget_us: 0,
            simulations: 128,
            confirmation_samples: 32,
            speculative_loss: 2.,
            rollout_call: true,
            rollout_cycles: true,
            rule_evidence: true,
            root_racing,
            ..PlannerCfg::hard_with_model(None)
        });
        for n in 2..=4 {
            for drew in [false, true] {
                let mut s = dealt(830900 + n as u64, n);
                let me = s.actor().unwrap();
                if drew {
                    s.apply(me, &Command::BeginDraw).unwrap();
                }
                let mut changed = s.clone();
                let unknown: Vec<_> = (0..s.cards.len())
                    .filter(|&i| !s.cards[i].revealed && !s.cards[i].known_by.contains(&me))
                    .collect();
                let ranks: Vec<_> = unknown.iter().map(|&i| s.cards[i].card.rank).collect();
                for (&id, rank) in unknown.iter().zip(ranks.iter().rev()) {
                    changed.cards[id].card.rank = *rank;
                }
                let a = bot.analyze(&project(&s, Some(me), 0), &mut StdRng::seed_from_u64(19));
                let b = bot.analyze(
                    &project(&changed, Some(me), 0),
                    &mut StdRng::seed_from_u64(19),
                );
                assert!(a.simulations > 0);
                assert_eq!(
                    (a.command.clone(), a.simulations, a.confirmations),
                    (b.command, b.simulations, b.confirmations)
                );
                assert_eq!(
                    a.candidates
                        .iter()
                        .map(|c| (c.visits, c.mean))
                        .collect::<Vec<_>>(),
                    b.candidates
                        .iter()
                        .map(|c| (c.visits, c.mean))
                        .collect::<Vec<_>>()
                );
                s.apply(me, &a.command).unwrap();
            }
        }
    }
}

#[test]
fn rollout_cycle_guard_requires_two_complete_public_cycles() {
    let mut state = exact_state(&dealt(830911, 2));
    let cycle = [(0, 1, 9, 8), (1, 2, 8, 9)];
    state.exchange_cycle = Some(cycle.into_iter().collect());
    assert!(!state.repeated_exchange_cycle());
    state.exchange_cycle.as_mut().unwrap().extend(cycle);
    assert!(state.repeated_exchange_cycle());
    assert_eq!(policy::rollout(&state, 1), Action::Draw);
    state.apply(&Action::Draw);
    assert!(state.exchange_cycle.as_ref().unwrap().is_empty());
    assert!(!state.repeated_exchange_cycle());
    // Even the chosen guard action uses exactly the ordinary engine semantics.
    let mut session = dealt(830911, 2);
    let me = session.actor().unwrap();
    session.apply(me, &Command::BeginDraw).unwrap();
    assert_same(&state, &session);
}

#[test]
fn failed_exchange_exposure_does_not_imply_prior_knowledge() {
    let mut s = scoring_fixture(
        &[
            vec![5, 5, 12, 10],
            vec![1, 2, 3, 4],
            vec![6, 7, 8, 9],
            vec![0, 1, 2, 3],
        ],
        &[62, 57, 26, 69],
        &[false; 4],
    );
    for i in [2, 3] {
        let id = s.players[0].slots[i] as usize;
        s.cards[id].revealed = false;
        s.cards[id].known_by.clear();
    }
    let before = project(&s, Some(0), 0);
    assert_eq!(before.all_seats[0].slots[2].value, None);
    assert_eq!(before.all_seats[0].slots[3].value, None);
    let sampler = Sampler::new(&before, &mut StdRng::seed_from_u64(2), true).unwrap();
    assert!(policy::ranked(&sampler.template, 64)
        .iter()
        .any(|(a, _)| *a == Action::Exchange(vec![2, 3])));
    s.apply(0, &Command::SwapOnce { slots: vec![2, 3] })
        .unwrap();
    let after = project(&s, Some(0), 0);
    assert_eq!(after.all_seats[0].slots[2].value, Some(12));
    assert_eq!(after.all_seats[0].slots[3].value, Some(10));
    assert_eq!(s.players[0].slots.len(), 5);
    // Put the actor back on an ordinary legal turn: known mismatch is excluded.
    s.discard.push(s.deck.pop().unwrap());
    s.phase = Phase::Turn {
        current: 0,
        pending: None,
    };
    let sampler = Sampler::new(
        &project(&s, Some(0), 0),
        &mut StdRng::seed_from_u64(2),
        true,
    )
    .unwrap();
    assert!(policy::ranked(&sampler.template, 64)
        .iter()
        .any(|(a, _)| matches!(a, Action::Exchange(_))));
    assert!(!policy::ranked(&sampler.template, 64)
        .iter()
        .any(|(a, _)| *a == Action::Exchange(vec![2, 3])));
}

#[test]
#[ignore = "offline diagnosis of full-trajectory completion under identical observations"]
fn audit_rollout_completion() {
    let mut baseline = PlannerCfg::hard_with_model(None);
    baseline.budget_us = 0;
    baseline.simulations = 128;
    baseline.confirmation_samples = 32;
    let old = PlannerBot::new(baseline.clone());
    baseline.rollout_cycles = true;
    let new = PlannerBot::new(baseline);
    let mut checks = 0;
    let mut truncated = [0, 0];
    let mut samples = [0u64, 0];
    let mut us = [0u64, 0];
    for seed in 834000..834030 {
        let mut s = dealt(seed, 4);
        let mut rng = StdRng::seed_from_u64(seed);
        for step in 0..200 {
            let Some(me) = s.actor() else { break };
            let view = project(&s, Some(me), 0);
            if step % 3 == 0 {
                for (i, bot) in [&old, &new].into_iter().enumerate() {
                    let r = bot.analyze(&view, &mut StdRng::seed_from_u64(seed ^ step));
                    samples[i] += r.simulations as u64;
                    truncated[i] += u64::from(r.simulations < 128);
                    us[i] += r.elapsed_us;
                }
                checks += 1;
            }
            s.apply(me, &ChallengerBot.decide(&view, &mut rng)).unwrap();
        }
        assert!(matches!(s.phase, Phase::RoundEnd | Phase::GameOver { .. }));
    }
    eprintln!("COMPLETION checks={checks} early_stop={truncated:?} completed_samples={samples:?} total_us={us:?}");
}

#[test]
fn special_rule_evidence_replay_matches_simulation_and_retains_ordinary_inference() {
    for (total, used, relaxed) in [(80, false, true), (0, false, false), (80, true, false)] {
        let mut session = dealt(839001, 4);
        let actor = session.actor().unwrap();
        session.players[actor].total_score = total;
        session.players[actor].score_reset_used = used;
        let mut sim = exact_state(&session);
        sim.rule_evidence = true;
        sim.apply(&Action::Draw);
        session.apply(actor, &Command::BeginDraw).unwrap();
        sim.apply(&Action::Discard(None));
        session
            .apply(actor, &Command::DiscardDrawn { power: None })
            .unwrap();
        assert_same(&sim, &session);
        let me = session.actor().unwrap();
        let view = project(&session, Some(me), 0);
        let new = Sampler::configured_with_rules(
            &view,
            &mut StdRng::seed_from_u64(7),
            true,
            false,
            false,
            false,
            true,
        )
        .unwrap();
        let old = Sampler::new(&view, &mut StdRng::seed_from_u64(7), true).unwrap();
        for slot in [0, 1] {
            let before = old.template.cards[old.template.hands[actor][slot] as usize].log_weights;
            let after = new.template.cards[new.template.hands[actor][slot] as usize].log_weights;
            assert!(before.iter().any(|&w| w != 0.));
            assert_eq!(after == [0.; 14], relaxed);
            assert_eq!(
                after,
                sim.cards[sim.hands[actor][slot] as usize].log_weights
            );
            assert_eq!(
                old.template.visible(old.template.hands[actor][slot], me),
                new.template.visible(new.template.hands[actor][slot], me)
            );
        }
    }
}

#[test]
fn exported_action_features_keep_their_baseline_with_a_live_match_value_model() {
    let session = scoring_fixture(
        &[vec![5, 5, 0, 0], vec![1], vec![2]],
        &[80, 70, 60],
        &[false, true, false],
    );
    let mut state = exact_state(&session);
    let info = policy::Info::new(&state, 0, true);
    let before = action_policy::features(&state, &info, &Action::Cabo);
    let score_before = policy::score(&state, &info, &Action::Cabo);
    let mut weights = vec![0f32; 3425];
    for i in [1, 256, 1312] {
        weights[i] = 1.;
    }
    weights[3392] = 10.;
    let mut bytes = b"CABOMV01".to_vec();
    for w in weights {
        bytes.extend(w.to_le_bytes());
    }
    state.value_model = Some(std::sync::Arc::new(
        match_value::MatchValue::decode(&bytes).unwrap(),
    ));
    assert!((score_before - policy::score(&state, &info, &Action::Cabo)).abs() > 1.);
    assert_eq!(
        before,
        action_policy::features(&state, &info, &Action::Cabo)
    );
    let bot = PlannerBot {
        cfg: PlannerCfg {
            learned_value: true,
            ..PlannerCfg::default()
        },
        value_model: state.value_model.clone(),
        policy_model: None,
    };
    let exported = bot
        .policy_features(
            &project(&session, Some(0), 0),
            &mut StdRng::seed_from_u64(29),
        )
        .unwrap();
    let row = exported
        .iter()
        .find(|(a, _)| *a == Command::CallCabo)
        .unwrap();
    assert_eq!(row.1, before.to_vec());
}

#[test]
fn stochastic_policy_matches_its_probabilities_and_clones_paired_random_streams() {
    let mut session = scoring_fixture(
        &[vec![8, 9, 10, 11], vec![0, 1, 2, 3]],
        &[0, 0],
        &[false; 2],
    );
    session.apply(0, &Command::BeginDraw).unwrap();
    let state = exact_state(&session);
    let mut bytes = b"CABOPL01".to_vec();
    bytes.resize(7500, 0);
    let model = action_policy::ActionPolicy::decode(&bytes).unwrap();
    let paired = state.clone();
    for _ in 0..64 {
        assert_eq!(model.sample(&state, 64, 1.), model.sample(&paired, 64, 1.));
    }
    let info = policy::Info::new(&state, 0, false);
    let actions = policy::candidates(&state, &info);
    let logits: Vec<_> = actions
        .iter()
        .map(|a| model.logit(&action_policy::features(&state, &info, a)) as f64)
        .collect();
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logits.iter().map(|v| (*v - top).exp()).collect();
    let sum = weights.iter().sum::<f64>();
    let mut counts = vec![0; actions.len()];
    for _ in 0..4096 {
        let a = model.sample(&state, 64, 1.);
        counts[actions.iter().position(|b| *b == a).unwrap()] += 1;
    }
    for (count, w) in counts.into_iter().zip(weights) {
        assert!((count as f64 / 4096. - w / sum).abs() < 0.05);
    }
}

#[test]
#[ignore = "local opponent policy/value models; actual Hard latency and engine legality"]
fn local_opponent_policy_production_budget() {
    let path = std::env::var("CABO_ACTION_POLICY_TEST")
        .unwrap_or("data/action_policy/model-v1.bin".into());
    let temperature = std::env::var("CABO_POLICY_TEMPERATURE_TEST")
        .unwrap_or("0".into())
        .parse()
        .unwrap();
    let mut cfg = PlannerCfg::hard_with_model(Some("data/match_value/model-v1.bin".into()));
    cfg.rollout_policy = 2;
    cfg.policy_model_path = path;
    cfg.rollout_temperature = temperature;
    cfg.budget_us = 600000;
    let bot = PlannerBot::new(cfg);
    for n in 2..=4 {
        for drew in [false, true] {
            let mut s = dealt(845000 + n as u64, n);
            for (p, score) in s.players.iter_mut().zip([69, 57, 62, 26]) {
                p.total_score = score;
            }
            let me = s.actor().unwrap();
            if drew {
                s.apply(me, &Command::BeginDraw).unwrap();
            }
            let report = bot.analyze(&project(&s, Some(me), 0), &mut StdRng::seed_from_u64(19));
            assert!(report.elapsed_us < 1600000);
            assert!(report.simulations <= 8192);
            s.apply(me, &report.command).unwrap();
            println!(
                "OPPONENT_BUDGET n={n} drew={drew} us={} samples={} confirmations={}",
                report.elapsed_us, report.simulations, report.confirmations
            );
        }
    }
}

#[test]
fn normal_frozen_decisions() {
    // Frozen on 3d6f668: exercise complete rounds, private draws, powers and late totals.
    let bot = PlannerBot::new(PlannerCfg {
        budget_us: 0,
        simulations: 96,
        confirmation_samples: 24,
        ..PlannerCfg::default()
    });
    let mut hash = 0xcbf29ce484222325u64;
    let mut decisions = 0;
    for n in 2..=4 {
        for seed in 0..6 {
            let mut s = dealt(61000 + seed, n);
            for p in 0..n {
                s.players[p].total_score = [80, 90, 95, 70][p];
            }
            let mut rng = StdRng::seed_from_u64(seed + n as u64 * 101);
            for _ in 0..120 {
                let Some(p) = s.actor() else { break };
                let v = project(&s, Some(p), 0);
                let c = bot.decide(&v, &mut rng);
                for b in format!("{c:?}").bytes() {
                    hash = (hash ^ b as u64).wrapping_mul(0x100000001b3);
                }
                decisions += 1;
                s.apply(p, &c).unwrap();
            }
        }
    }
    assert_eq!(
        (decisions, hash),
        (458, 16368717308336445282),
        "Normal must keep its frozen decisions"
    );
}

#[test]
fn action_features_do_not_read_sampled_hidden_ranks_or_card_ids() {
    let mut rng = StdRng::seed_from_u64(81000);
    for n in 2..=4 {
        for drew in [false, true] {
            let mut session = dealt(81000 + n as u64, n);
            let me = session.actor().unwrap();
            if drew {
                session.apply(me, &Command::BeginDraw).unwrap();
            }
            let view = project(&session, Some(me), 0);
            let mut sampler = Sampler::new(&view, &mut rng, true).unwrap();
            let world = sampler.sample(&mut rng);
            let observation = policy::Info::new(
                &world,
                me,
                world.stage == Stage::Idle && world.caller.is_none(),
            );
            let actions = policy::candidates(&world, &observation);
            let original: Vec<_> = actions
                .iter()
                .map(|a| action_policy::features(&world, &observation, a))
                .collect();
            let mut changed = world.clone();
            for id in 0..changed.cards.len() {
                if changed.visible(id as u8, me).is_none() {
                    changed.cards[id].rank = ((id * 7 + 3) % 14) as u8;
                }
            }
            let info = policy::Info::new(
                &changed,
                me,
                world.stage == Stage::Idle && world.caller.is_none(),
            );
            for (a, before) in actions.iter().zip(original) {
                let after = action_policy::features(&changed, &info, a);
                assert_eq!(before, after);
                assert!(after.iter().all(|x| x.is_finite()));
            }
            let sample2 = sampler.sample(&mut rng);
            let info2 = policy::Info::new(
                &sample2,
                me,
                world.stage == Stage::Idle && world.caller.is_none(),
            );
            for a in &actions {
                assert_eq!(
                    action_policy::features(&world, &observation, a),
                    action_policy::features(&sample2, &info2, a)
                );
            }
        }
    }
}

#[test]
fn action_policy_opt_in_requires_a_valid_model_and_leaves_normal_off() {
    assert_eq!(PlannerCfg::default().rollout_policy, 0);
    assert!(!PlannerCfg::default().policy_only);
    assert!(PlannerBot::try_new(PlannerCfg {
        rollout_policy: 1,
        policy_model_path: "data/missing-policy.bin".into(),
        ..PlannerCfg::default()
    })
    .is_err());
    assert!(PlannerBot::try_new(PlannerCfg {
        policy_only: true,
        policy_model_path: "data/missing-policy.bin".into(),
        ..PlannerCfg::default()
    })
    .is_err());
}

#[test]
fn action_learning_preserves_frozen_search_for_untrained_scoring_settings() {
    let mut bytes = b"CABOPL01".to_vec();
    bytes.resize(7500, 0);
    let model = std::sync::Arc::new(action_policy::ActionPolicy::decode(&bytes).unwrap());
    for racing in [false, true] {
        let cfg = PlannerCfg {
            budget_us: 0,
            simulations: 48,
            confirmation_samples: 16,
            root_racing: racing,
            racing_actions: 4,
            ..PlannerCfg::default()
        };
        let baseline = PlannerBot::new(cfg.clone());
        let learned = PlannerBot {
            cfg: PlannerCfg {
                rollout_policy: 1,
                policy_prior: true,
                policy_puct: 1.,
                rollout_temperature: 1.,
                ..cfg
            },
            value_model: None,
            policy_model: Some(model.clone()),
        };
        for (target, penalty) in [(80, 10), (100, 15)] {
            let mut s = dealt(81300, 3);
            s.settings.target_score = target;
            s.settings.cabo_penalty = penalty;
            let view = project(&s, Some(s.actor().unwrap()), 0);
            let old = baseline.analyze(&view, &mut StdRng::seed_from_u64(47));
            let new = learned.analyze(&view, &mut StdRng::seed_from_u64(47));
            assert_eq!(old.command, new.command);
            assert_eq!(old.simulations, new.simulations);
            for (a, b) in old.candidates.iter().zip(&new.candidates) {
                assert_eq!((a.visits, a.mean), (b.visits, b.mean));
            }
        }
    }
}

#[test]
#[ignore = "requires local action-policy model; verifies complete learned-rollout searches"]
fn learned_action_search_uses_only_legal_information() {
    let path = std::env::var("CABO_ACTION_POLICY_TEST")
        .unwrap_or("data/action_policy/model-v1.bin".into());
    for racing in [false, true] {
        for (mode, prior, puct, temperature) in [
            (1, false, 0., 0.),
            (2, false, 0., 0.),
            (3, false, 0., 0.),
            (0, true, 0., 0.),
            (0, false, 1., 0.),
            (2, false, 0., 1.),
        ] {
            let bot = PlannerBot::new(PlannerCfg {
                policy_model_path: path.clone(),
                rollout_policy: mode,
                policy_prior: prior,
                policy_puct: puct,
                rollout_temperature: temperature,
                budget_us: 0,
                simulations: 48,
                confirmation_samples: 16,
                root_racing: racing,
                racing_actions: 4,
                ..PlannerCfg::default()
            });
            let mut session = dealt(81100, 3);
            let me = session.actor().unwrap();
            session.apply(me, &Command::BeginDraw).unwrap();
            let first = bot.analyze(
                &project(&session, Some(me), 0),
                &mut StdRng::seed_from_u64(41),
            );
            assert!(
                first.simulations > 0,
                "learned rollout must reach actual round settlement"
            );
            for card in &mut session.cards {
                if !card.revealed && !card.known_by.contains(&me) {
                    card.card.rank = 13;
                }
            }
            let second = bot.analyze(
                &project(&session, Some(me), 0),
                &mut StdRng::seed_from_u64(41),
            );
            assert_eq!(first.command, second.command);
            assert_eq!(first.simulations, second.simulations);
            assert_eq!(first.candidates.len(), second.candidates.len());
            for (a, b) in first.candidates.iter().zip(&second.candidates) {
                assert_eq!((a.visits, a.mean), (b.visits, b.mean));
            }
        }
    }
}

#[test]
#[ignore = "requires local action model; checks time bounds and real-engine legality"]
fn local_action_policy_budget_and_games() {
    let path = std::env::var("CABO_ACTION_POLICY_TEST")
        .unwrap_or("data/action_policy/model-v1.bin".into());
    for n in 2..=4 {
        let direct = PlannerBot::new(PlannerCfg {
            policy_only: true,
            policy_actions: 64,
            policy_model_path: path.clone(),
            avoid_cycles: true,
            ..PlannerCfg::default()
        });
        let mut session = dealt(81200 + n as u64, n);
        let mut rng = StdRng::seed_from_u64(43);
        for _ in 0..10000 {
            match session.phase {
                Phase::Turn { current, .. } => {
                    let c = direct.decide(&project(&session, Some(current), 0), &mut rng);
                    session.apply(current, &c).unwrap();
                }
                Phase::RoundEnd => {
                    session.next_round().unwrap();
                    for p in 0..n {
                        session
                            .apply(p, &Command::PeekInitial { slots: [0, 1] })
                            .unwrap();
                    }
                }
                Phase::GameOver { .. } => break,
                _ => unreachable!(),
            }
        }
        assert!(matches!(session.phase, Phase::GameOver { .. }));
        for (mode, prior, puct) in [
            (1, false, 0.),
            (3, false, 0.),
            (0, true, 0.),
            (0, false, 1.),
        ] {
            let bot = PlannerBot::new(PlannerCfg {
                policy_model_path: path.clone(),
                rollout_policy: mode,
                policy_prior: prior,
                policy_puct: puct,
                budget_us: 600000,
                ..PlannerCfg::hard_with_model(None)
            });
            let mut s = dealt(81230 + n as u64, n);
            let actor = s.actor().unwrap();
            s.apply(actor, &Command::BeginDraw).unwrap();
            let start = Instant::now();
            let report = bot.analyze(&project(&s, Some(actor), 0), &mut rng);
            assert!(start.elapsed() < std::time::Duration::from_millis(1600));
            assert!(report.simulations > 0);
            s.apply(actor, &report.command).unwrap();
            println!(
                "n={n} scope={mode} prior={prior} puct={puct} ms={:.1} sims={}",
                report.elapsed_us as f64 / 1000.,
                report.simulations
            );
        }
    }
}

#[test]
fn hard_breaks_a_repeated_public_rotation_but_preserves_one_cycle() {
    let mut s = dealt(530200, 2);
    let me = s.actor().unwrap();
    let cycle: Vec<_> = (0..6)
        .map(|i| PublicEvent::Exchange {
            player: (me + i) % 2,
            source: crate::game::Pile::Discard,
            slots: vec![0],
            exposed: vec![[5, 7, 12][i % 3]],
            incoming: Some([12, 5, 7][i % 3]),
            success: true,
        })
        .collect();
    s.public_events = cycle.clone();
    assert!(!repeated_public_cycle(&project(&s, Some(me), 0)));
    s.public_events.extend(cycle);
    assert!(repeated_public_cycle(&project(&s, Some(me), 0)));
    for root_racing in [false, true] {
        let bot = PlannerBot::new(PlannerCfg {
            root_racing,
            ..PlannerCfg::hard()
        });
        assert_eq!(
            bot.decide(&project(&s, Some(me), 0), &mut StdRng::seed_from_u64(7)),
            Command::BeginDraw
        );
        // The cycle decision uses only public events, even if hidden ground truth differs.
        for card in &mut s.cards {
            if !card.revealed && !card.known_by.contains(&me) {
                card.card.rank = 13;
            }
        }
        assert_eq!(
            bot.decide(&project(&s, Some(me), 0), &mut StdRng::seed_from_u64(7)),
            Command::BeginDraw
        );
    }
    s.public_events.push(PublicEvent::Discard {
        player: me,
        rank: 7,
        powered: false,
    });
    assert!(!repeated_public_cycle(&project(&s, Some(me), 0)));
    assert!(!PlannerCfg::default().avoid_cycles);
}

#[test]
fn model_configuration_fails_explicitly_and_default_normal_never_loads_it() {
    let missing = "data/nonexistent-match-value-model.bin".to_string();
    assert!(
        crate::ai::build_bot("normal", &[("value_model_path".into(), missing.clone())]).is_some()
    );
    assert!(crate::ai::build_bot(
        "hard",
        &[
            ("value_model_path".into(), missing),
            ("learned_value".into(), "true".into())
        ]
    )
    .is_none());
}

#[test]
fn local_hard_model_opt_in_enables_the_validated_bundle_only_for_hard() {
    let ordinary = PlannerCfg::hard_with_model(None);
    assert!(!ordinary.learned_value && !ordinary.validate_calls);
    assert_eq!(ordinary.min_cabo_success, 0.75);
    let learned = PlannerCfg::hard_with_model(Some("data/missing-model.bin".into()));
    assert!(learned.learned_value && learned.validate_calls && learned.avoid_cycles);
    assert_eq!(learned.min_cabo_success, 0.0);
    assert_eq!(learned.simulations, ordinary.simulations);
    assert!(PlannerBot::try_new(learned).is_err());
    assert!(!PlannerCfg::default().learned_value && !PlannerCfg::default().validate_calls);
}

#[test]
#[ignore = "requires local model; checks production Hard time limit and legal commands"]
fn local_learned_hard_default_budget_is_bounded() {
    let path =
        std::env::var("CABO_MATCH_VALUE_TEST").unwrap_or("data/match_value/model-v1.bin".into());
    let mut cfg = PlannerCfg::hard_with_model(Some(path));
    cfg.budget_us = 600000;
    let bot = PlannerBot::new(cfg);
    for n in 2..=4 {
        for draw in [false, true] {
            let mut s = dealt(41 + n as u64, n);
            let me = s.actor().unwrap();
            if draw {
                s.apply(me, &Command::BeginDraw).unwrap();
            }
            let start = Instant::now();
            let report = bot.analyze(&project(&s, Some(me), 0), &mut StdRng::seed_from_u64(42));
            assert!(start.elapsed() < std::time::Duration::from_millis(1600));
            assert!(report.simulations <= 8192);
            s.apply(me, &report.command).unwrap();
            println!(
                "n={n} drew={draw} elapsed_ms={:.1} simulations={}",
                report.elapsed_us as f64 / 1000.0,
                report.simulations
            );
        }
    }
}

#[test]
#[ignore = "requires local trained model; full search must remain isolated from hidden ground truth"]
fn learned_search_uses_only_legal_information() {
    let path =
        std::env::var("CABO_MATCH_VALUE_TEST").unwrap_or("data/match_value/model-v2.bin".into());
    let s = dealt(16, 4);
    let me = s.actor().unwrap();
    let mut other = s.clone();
    let ids: Vec<_> = s
        .cards
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.revealed && !c.known_by.contains(&me))
        .map(|(i, _)| i)
        .collect();
    let ranks: Vec<_> = ids.iter().map(|&i| s.cards[i].card.rank).collect();
    for (&i, &r) in ids.iter().zip(ranks.iter().rev()) {
        other.cards[i].card.rank = r;
    }
    for root_racing in [false, true] {
        let bot = PlannerBot::new(PlannerCfg {
            budget_us: 0,
            simulations: 96,
            confirmation_samples: 24,
            root_racing,
            learned_value: true,
            value_model_path: path.clone(),
            blind_keep_evidence: true,
            behavioral_evidence: true,
            call_evidence: true,
            validate_calls: true,
            ..PlannerCfg::default()
        });
        let a = bot.analyze(&project(&s, Some(me), 0), &mut StdRng::seed_from_u64(7));
        let b = bot.analyze(&project(&other, Some(me), 0), &mut StdRng::seed_from_u64(7));
        assert_eq!(a.command, b.command);
        assert_eq!(a.simulations, b.simulations);
        assert_eq!(
            a.candidates
                .iter()
                .map(|c| (c.visits, c.mean))
                .collect::<Vec<_>>(),
            b.candidates
                .iter()
                .map(|c| (c.visits, c.mean))
                .collect::<Vec<_>>()
        );
    }
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
        proven_reset_trade: false,
        rollout_temperature: 0.,
        policy_random: std::array::from_fn(|_| std::cell::Cell::new(0)),
        rule_evidence: false,
        exchange_cycle: None,
        speculative_loss: 0.,
        rollout_call: false,
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
        special_tactics: true,
        blind_keep_evidence: false,
        reset_policy_player: None,
        behavioral_evidence: false,
        call_evidence: false,
        value_model: None,
        policy_model: None,
        rollout_policy: 0,
        policy_player: actor,
        policy_actions: 12,
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
    for root_racing in [false, true] {
        let bot = PlannerBot::new(PlannerCfg {
            budget_us: 0,
            simulations: 64,
            root_racing,
            blind_keep_evidence: true,
            behavioral_evidence: true,
            call_evidence: true,
            validate_calls: true,
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
    }
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

fn final_reset_trade_fixture(drew: bool, n: usize) -> Session {
    let mut hands = vec![vec![4, 5, 6, 7], vec![0, 1], vec![3], vec![2]];
    hands.truncate(n);
    let mut s = scoring_fixture(&hands, &[69, 65, 98, 80][..n], &vec![false; n]);
    s.phase = Phase::Turn {
        current: 1,
        pending: None,
    };
    s.apply(1, &Command::CallCabo).unwrap();
    for actor in 2..n {
        let i = s
            .deck
            .iter()
            .position(|&id| s.cards[id as usize].card.rank == 9)
            .unwrap();
        let last = s.deck.len() - 1;
        s.deck.swap(i, last);
        s.apply(actor, &Command::BeginDraw).unwrap();
        s.apply(actor, &Command::DiscardDrawn { power: None })
            .unwrap();
    }
    if drew {
        let i = s
            .deck
            .iter()
            .position(|&id| s.cards[id as usize].card.rank == 9)
            .unwrap();
        let last = s.deck.len() - 1;
        s.deck.swap(i, last);
        s.apply(0, &Command::BeginDraw).unwrap();
    }
    assert_eq!(s.actor(), Some(0));
    assert!(s.extra_turns.is_empty());
    s
}

#[test]
fn hard_proves_intentional_failure_wins_using_real_final_settlement() {
    for n in 3..=4 {
        for drew in [false, true] {
            for racing in [false, true] {
                let mut s = final_reset_trade_fixture(drew, n);
                let view = project(&s, Some(0), 0);
                let mut rng = StdRng::seed_from_u64(94823);
                let cfg = PlannerCfg {
                    root_racing: racing,
                    budget_us: 0,
                    simulations: 32,
                    confirmation_samples: 16,
                    ..PlannerCfg::hard_with_model(None)
                };
                let bot = PlannerBot::new(cfg);
                let features = bot.policy_features(&view, &mut rng).unwrap();
                let report = bot.analyze(&view, &mut rng);
                let expected = if drew {
                    Command::DrawSwap { slots: vec![0, 1] }
                } else {
                    Command::SwapOnce { slots: vec![0, 1] }
                };
                assert_eq!(report.command, expected);
                assert_eq!(features.len(), 1);
                assert_eq!(features[0].0, expected);
                assert!(report.belief_ok);
                assert_eq!(report.simulations, 0);
                s.apply(0, &report.command).unwrap();
                assert!(matches!(s.phase, Phase::GameOver { ref winners } if winners == &[0]));
                assert_eq!(
                    s.players.iter().map(|p| p.total_score).collect::<Vec<_>>(),
                    &[50, 65, 101, 82][..n]
                );
                assert!(s.players[0].score_reset_used);
                assert_eq!(s.players[0].slots.len(), 5);
            }
        }
    }
}

#[test]
fn proven_reset_trade_rejects_unknown_nonterminal_tied_or_used_opportunities() {
    let base = final_reset_trade_fixture(false, 3);
    let mut state = exact_state(&base);
    state.proven_reset_trade = true;
    assert_eq!(
        special::proven_reset_trade(&state),
        Some(Action::Exchange(vec![0, 1]))
    );
    let mut cases = Vec::new();
    let mut s = state.clone();
    s.proven_reset_trade = false;
    cases.push(s);
    let mut s = state.clone();
    s.policy_player = 1;
    cases.push(s);
    let mut s = state.clone();
    s.score_reset_used[0] = true;
    cases.push(s);
    let mut s = state.clone();
    s.extra.push_back(2);
    cases.push(s);
    let mut s = state.clone();
    s.totals[0] = 68;
    cases.push(s);
    let mut s = state.clone();
    s.totals[1] = 50;
    cases.push(s);
    let mut s = state.clone();
    s.totals[1] = 49;
    cases.push(s);
    let mut s = state.clone();
    s.totals[1] = 99;
    cases.push(s); // Preserving the current hand already uniquely wins.
    let mut s = state.clone();
    s.target = 200;
    cases.push(s);
    let mut s = state.clone();
    s.totals[2] = 90;
    cases.push(s);
    let mut s = state.clone();
    s.caller = None;
    cases.push(s);
    let mut s = state.clone();
    s.caller = Some(0);
    cases.push(s);
    for p in 0..3 {
        let mut s = state.clone();
        let id = s.hands[p][0] as usize;
        s.cards[id].revealed = false;
        s.cards[id].known &= !1;
        for rank in 0..14 {
            s.cards[id].rank = rank;
            assert_eq!(special::proven_reset_trade(&s), None, "hidden rank {rank}");
        }
        cases.push(s);
    }
    for s in cases {
        assert_eq!(special::proven_reset_trade(&s), None);
    }
    let mut normal = state.clone();
    normal.proven_reset_trade = PlannerCfg::default().proven_reset_trade;
    let info = policy::Info::new(&normal, 0, false);
    assert!(!policy::candidates(&normal, &info).contains(&Action::Exchange(vec![0, 1])));
}

#[test]
fn hard_keeps_frozen_search_when_final_reset_cannot_be_proven() {
    for n in 3..=4 {
        for drew in [false, true] {
            for racing in [false, true] {
                let mut s = final_reset_trade_fixture(drew, n);
                let id = s.players[1].slots[0] as usize;
                s.cards[id].revealed = false;
                s.cards[id].known_by = [1].into_iter().collect();
                let view = project(&s, Some(0), 0);
                let cfg = PlannerCfg {
                    budget_us: 0,
                    simulations: 32,
                    confirmation_samples: 16,
                    root_racing: racing,
                    ..PlannerCfg::hard_with_model(None)
                };
                let mut old = cfg.clone();
                old.proven_reset_trade = false;
                let before = PlannerBot::new(old).analyze(&view, &mut StdRng::seed_from_u64(90812));
                let after = PlannerBot::new(cfg).analyze(&view, &mut StdRng::seed_from_u64(90812));
                assert_eq!(before.command, after.command);
                assert_eq!(before.simulations, after.simulations);
                assert_eq!(before.nodes, after.nodes);
                assert_eq!(before.candidates.len(), after.candidates.len());
                assert_eq!(before.confirmations, after.confirmations);
                assert_eq!(before.confirmed_gain, after.confirmed_gain);
                assert_eq!(before.confirmed_se, after.confirmed_se);
                for (a, b) in before.candidates.iter().zip(&after.candidates) {
                    assert_eq!(format!("{a:?}"), format!("{b:?}"));
                }
                s.apply(0, &after.command).unwrap();
            }
        }
    }
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

fn draw_rank_next(s: &mut Session, rank: u8) {
    let i = s
        .deck
        .iter()
        .position(|&id| s.cards[id as usize].card.rank == rank)
        .unwrap();
    let id = s.deck.remove(i);
    s.deck.push(id);
}

fn force_discard_rank(s: &mut Session, rank: u8) {
    draw_rank_next(s, rank);
    let id = s.deck.pop().unwrap();
    s.deck.extend(s.discard.drain(..));
    s.discard.push(id);
}

fn special_test_bot(enabled: bool) -> PlannerBot {
    PlannerBot::new(PlannerCfg {
        budget_us: 0,
        simulations: 256,
        tree_depth: 1,
        special_tactics: enabled,
        confirmation_samples: 48,
        ..PlannerCfg::default()
    })
}

#[test]
fn planner_can_deliberately_fail_cabo_to_reset_and_win() {
    let mut s = scoring_fixture(
        &[vec![5, 5, 0, 0], vec![2, 2, 3, 4], vec![1, 1, 1, 1]],
        &[80, 90, 95],
        &[false; 3],
    );
    force_discard_rank(&mut s, 11);
    draw_rank_next(&mut s, 13);
    // Put both 13s next: neither opponent can invoke an ability or improve via the discard.
    let last = s.deck.pop().unwrap();
    draw_rank_next(&mut s, 13);
    s.deck.push(last);
    let v = project(&s, Some(0), 0);
    let improved = special_test_bot(true).analyze(&v, &mut StdRng::seed_from_u64(815));
    let old = special_test_bot(false).analyze(&v, &mut StdRng::seed_from_u64(815));
    assert_eq!(improved.command, Command::CallCabo);
    assert_ne!(old.command, Command::CallCabo, "旧普通 Cabo 门槛应拒绝本例");
    assert!(
        improved
            .candidates
            .iter()
            .find(|c| c.command == Command::CallCabo)
            .unwrap()
            .reset_probability
            > 0.0
    );
    assert_eq!(
        improved
            .candidates
            .iter()
            .find(|c| c.command == Command::CallCabo)
            .unwrap()
            .cabo_success,
        Some(0.0)
    );
    s.apply(0, &improved.command).unwrap();
    for p in [1, 2] {
        let action = ChallengerBot.decide(
            &project(&s, Some(p), 0),
            &mut StdRng::seed_from_u64(p as u64),
        );
        s.apply(p, &action).unwrap();
        let action = ChallengerBot.decide(
            &project(&s, Some(p), 0),
            &mut StdRng::seed_from_u64(p as u64),
        );
        s.apply(p, &action).unwrap();
    }
    assert_eq!(s.phase, Phase::GameOver { winners: vec![0] });
    assert_eq!(s.players[0].total_score, 50);
    assert!(s.players[0].score_reset_this_round);
}

#[test]
fn ordinary_incumbent_calls_are_checked_after_final_responses() {
    let mut s = scoring_fixture(&[vec![0, 0, 4, 4], vec![2, 2, 4, 4]], &[0, 0], &[false; 2]);
    force_discard_rank(&mut s, 13);
    let state = exact_state(&s);
    assert_eq!(policy::choose(&state, 1), Action::Cabo);
    let view = project(&s, Some(0), 0);
    let cfg = PlannerCfg {
        budget_us: 0,
        simulations: 512,
        confirmation_samples: 48,
        ..PlannerCfg::default()
    };
    let old = PlannerBot::new(cfg.clone()).analyze(&view, &mut StdRng::seed_from_u64(33));
    let new = PlannerBot::new(PlannerCfg {
        validate_calls: true,
        ..cfg
    })
    .analyze(&view, &mut StdRng::seed_from_u64(33));
    assert_eq!(old.command, Command::CallCabo);
    assert_eq!(new.command, Command::BeginDraw);
    let call = new
        .candidates
        .iter()
        .find(|c| c.command == Command::CallCabo)
        .unwrap();
    assert!(call.cabo_success.unwrap() < 0.75);
}

#[test]
fn higher_hand_call_can_have_positive_match_value_without_forcing_a_call() {
    let mut s = scoring_fixture(
        &[vec![7, 7, 8, 8], vec![3, 4, 5, 6]],
        &[50, 85],
        &[true, false],
    );
    force_discard_rank(&mut s, 13);
    let view = project(&s, Some(0), 0);
    let cfg = PlannerCfg {
        budget_us: 0,
        simulations: 1024,
        confirmation_samples: 128,
        opponent_style: 1,
        validate_calls: true,
        min_cabo_success: 0.,
        ..PlannerCfg::default()
    };
    let report = PlannerBot::new(cfg).analyze(&view, &mut StdRng::seed_from_u64(29));
    let call = report
        .candidates
        .iter()
        .find(|c| c.command == Command::CallCabo)
        .unwrap();
    assert_eq!(call.cabo_success, Some(0.));
    assert!(
        call.mean > 0.6,
        "a failed Cabo can still win the whole match"
    );
    assert_ne!(
        report.command,
        Command::CallCabo,
        "removing a strict-lowest gate must not force a call when another action is better"
    );
}

#[test]
fn final_response_preserves_exact_reset_instead_of_lowering_hand_sum() {
    let mut s = scoring_fixture(&[vec![1, 1], vec![5, 5, 5, 5]], &[60, 80], &[false; 2]);
    s.cabo_caller = Some(0);
    s.extra_turns.clear(); // 当前已经是最后一位加时玩家，队列只包含尚未行动者。
    s.phase = Phase::Turn {
        current: 1,
        pending: None,
    };
    draw_rank_next(&mut s, 0);
    s.apply(1, &Command::BeginDraw).unwrap();
    let v = project(&s, Some(1), 0);
    let cmd = special_test_bot(true).decide(&v, &mut StdRng::seed_from_u64(21));
    assert_eq!(cmd, Command::DiscardDrawn { power: None });
    s.apply(1, &cmd).unwrap();
    assert_eq!(s.players[1].total_score, 50);
    assert!(s.players[1].score_reset_used);
}

#[test]
fn planner_completes_high_pairs_by_taking_a_higher_card() {
    let mut s = scoring_fixture(&[vec![13, 13, 12, 5], vec![1, 1]], &[0, 0], &[false; 2]);
    s.settings.target_score = 50;
    s.cabo_caller = Some(1);
    s.extra_turns.clear();
    draw_rank_next(&mut s, 12);
    s.apply(0, &Command::BeginDraw).unwrap();
    let v = project(&s, Some(0), 0);
    let report = special_test_bot(true).analyze(&v, &mut StdRng::seed_from_u64(51));
    let cmd = report.command;
    assert_eq!(cmd, Command::DrawSwap { slots: vec![3] });
    assert_eq!(
        report
            .candidates
            .iter()
            .find(|c| c.command == cmd)
            .unwrap()
            .high_pairs_probability,
        1.0
    );
    s.apply(0, &cmd).unwrap();
    assert_eq!(s.phase, Phase::GameOver { winners: vec![0] });
    assert_eq!(s.players[0].round_score, Some(0));
    assert_eq!(s.players[1].round_score, Some(50));
}

#[test]
fn planner_compresses_extra_high_cards_into_exactly_four_and_preserves_a_complete_combo() {
    for ranks in [vec![13, 13, 12, 12, 12], vec![13, 13, 12, 12]] {
        let mut s = scoring_fixture(&[ranks.clone(), vec![1, 1]], &[0, 0], &[false; 2]);
        s.settings.target_score = 50;
        s.cabo_caller = Some(1);
        draw_rank_next(&mut s, if ranks.len() == 5 { 12 } else { 0 });
        s.apply(0, &Command::BeginDraw).unwrap();
        let cmd = special_test_bot(true)
            .decide(&project(&s, Some(0), 0), &mut StdRng::seed_from_u64(119));
        if ranks.len() == 5 {
            assert!(
                matches!(&cmd, Command::DrawSwap { slots } if slots.len() == 2 && slots.iter().all(|&i| i >= 2))
            );
        } else {
            assert_eq!(cmd, Command::DiscardDrawn { power: None });
        }
        s.apply(0, &cmd).unwrap();
        assert_eq!(s.phase, Phase::GameOver { winners: vec![0] });
        assert_eq!(s.players[0].slots.len(), 4);
        assert_eq!(s.players[0].round_score, Some(0));
    }
}

#[test]
fn planner_breaks_an_opponents_high_pairs_with_a_power() {
    let mut s = scoring_fixture(
        &[vec![5, 5, 5, 5], vec![12, 13, 12, 13]],
        &[0, 0],
        &[false; 2],
    );
    s.settings.target_score = 50;
    s.cabo_caller = Some(1);
    s.extra_turns.clear();
    draw_rank_next(&mut s, 12);
    s.apply(0, &Command::BeginDraw).unwrap();
    let v = project(&s, Some(0), 0);
    let cmd = special_test_bot(true).decide(&v, &mut StdRng::seed_from_u64(62));
    assert!(matches!(
        cmd,
        Command::DiscardDrawn {
            power: Some(PowerUse::Swap { player: 1, .. })
        }
    ));
    s.apply(0, &cmd).unwrap();
    assert_eq!(s.phase, Phase::GameOver { winners: vec![0] });
    assert!(s.players[1].round_score.unwrap() >= 50);
}

#[test]
fn special_calls_require_observable_opportunity_and_favorable_match_totals() {
    let hands = [vec![12, 12, 13, 13], vec![1, 1], vec![2, 2]];
    let s = scoring_fixture(&hands, &[80, 80, 0], &[false; 3]);
    let mut state = exact_state(&s);
    let info = policy::Info::new(&state, 0, true);
    assert!(special::call_relevant(&state));
    assert!(
        !special::preferred_call(&state, &info),
        "完整组合也不能覆盖大局已经输掉的事实"
    );
    state.cards[state.hands[0][0] as usize].revealed = false;
    state.cards[state.hands[0][0] as usize].known = 0;
    assert!(
        !special::call_relevant(&state),
        "不能读取自己的未知槽位确认组合"
    );

    let s = scoring_fixture(
        &[vec![5, 5, 0, 0], vec![1, 1, 1, 1]],
        &[80, 90],
        &[true, false],
    );
    let state = exact_state(&s);
    assert!(
        !special::call_relevant(&state),
        "已用资格不能再利用惩罚凑满分"
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
    assert!(
        weights(1).iter().any(|&w| w != 0.0),
        "保留新牌的行动证据跟随新牌"
    );
    assert_eq!(weights(2), [0.0; 14], "未选的未知牌不继承新牌证据");
    assert_eq!(weights(3), [0.0; 14]);
}

#[test]
fn blind_replacement_evidence_never_uses_the_exposed_old_rank() {
    let mut s = dealt(28, 3);
    let p = s.actor().unwrap();
    s.apply(p, &Command::BeginDraw).unwrap();
    s.apply(p, &Command::DrawSwap { slots: vec![2] }).unwrap();
    let view = project(&s, Some(s.actor().unwrap()), 0);
    let mut altered = view.clone();
    for e in &mut altered.public_events {
        if let PublicEvent::Exchange { exposed, .. } = e {
            exposed[0] = (exposed[0] + 5) % 14;
        }
    }
    let a =
        Sampler::configured(&view, &mut StdRng::seed_from_u64(1), true, true, true, true).unwrap();
    let b = Sampler::configured(
        &altered,
        &mut StdRng::seed_from_u64(1),
        true,
        true,
        true,
        true,
    )
    .unwrap();
    let weights = |s: &Sampler| s.template.cards[s.template.hands[p][2] as usize].log_weights;
    assert_eq!(weights(&a), weights(&b));
    assert!(weights(&a).iter().any(|&w| w != 0.0));
    for slot in [0, 1, 3] {
        let id = a.template.hands[p][slot] as usize;
        assert_eq!(a.template.cards[id].log_weights, [0.0; 14]);
    }
}

#[test]
fn enhanced_evidence_replays_like_the_simulator() {
    for seed in 0..12 {
        let mut s = dealt(110000 + seed, 4);
        let mut fast = exact_state(&s);
        fast.blind_keep_evidence = true;
        fast.behavioral_evidence = true;
        fast.call_evidence = true;
        let mut rng = StdRng::seed_from_u64(seed);
        for step in 0..120 {
            let Some(actor) = s.actor() else { break };
            let c = ChallengerBot.decide(&project(&s, Some(actor), 0), &mut rng);
            let a = match &c {
                Command::BeginDraw => Action::Draw,
                Command::DrawSwap { slots } => Action::Replace(slots.clone()),
                Command::SwapOnce { slots } => Action::Exchange(slots.clone()),
                Command::DiscardDrawn { power } => Action::Discard(*power),
                Command::CallCabo => Action::Cabo,
                _ => panic!("unexpected {c:?}"),
            };
            s.apply(actor, &c).unwrap();
            fast.apply(&a);
            let Some(next) = s.actor() else { break };
            let replay = Sampler::configured(
                &project(&s, Some(next), 0),
                &mut rng,
                true,
                true,
                true,
                true,
            )
            .unwrap();
            for p in 0..4 {
                for slot in 0..s.players[p].slots.len() {
                    assert_eq!(
                        fast.cards[fast.hands[p][slot] as usize].log_weights,
                        replay.template.cards[replay.template.hands[p][slot] as usize].log_weights,
                        "seed {seed} player {p} slot {slot} step {step} events {:?}",
                        s.public_events
                    );
                }
            }
        }
    }
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
    for root_racing in [false, true] {
        let bot = PlannerBot::new(PlannerCfg {
            budget_us: 0,
            simulations: 24,
            root_racing,
            blind_keep_evidence: true,
            behavioral_evidence: true,
            call_evidence: true,
            validate_calls: true,
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

#[test]
#[ignore = "offline comparison of legal posterior predictions with withheld ground truth"]
fn blind_keep_calibration() {
    let mut error = [[0.0; 4]; 4];
    let mut checks = 0;
    let base = std::env::var("CABO_DIAG_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(106000);
    let mut blocks: Vec<Vec<[f64; 2]>> = vec![Vec::new(); 4];
    for seed in 0..100 {
        let mut s = dealt(base + seed, 4);
        let mut play_rng = StdRng::seed_from_u64(seed);
        let mut seed_error = [[0.0; 4]; 4];
        let mut seed_checks = 0;
        for step in 0..120 {
            let Some(me) = s.actor() else { break };
            let v = project(&s, Some(me), 0);
            if step % 5 == 0 && matches!(v.panel, Panel::Idle { .. }) {
                for (mode, row) in seed_error.iter_mut().enumerate() {
                    let mut rng = StdRng::seed_from_u64(seed * 131 + step);
                    let mut sampler =
                        Sampler::configured(&v, &mut rng, true, mode >= 1, mode >= 2, mode >= 3)
                            .unwrap();
                    let mut means = [0.0; 4];
                    let mut lowest = 0;
                    for _ in 0..128 {
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
                            means[p] += sums[p] as f64 / 128.0;
                        }
                        lowest +=
                            u32::from((0..4).filter(|&p| p != me).all(|p| sums[me] < sums[p]));
                    }
                    let sums: Vec<u32> = s
                        .players
                        .iter()
                        .map(|p| {
                            p.slots
                                .iter()
                                .map(|&id| s.cards[id as usize].card.rank as u32)
                                .sum()
                        })
                        .collect();
                    for p in 0..4 {
                        if p != me {
                            row[0] += means[p] - sums[p] as f64;
                            row[1] += (means[p] - sums[p] as f64).abs();
                        }
                    }
                    let pred = lowest as f64 / 128.0;
                    let truth =
                        u8::from((0..4).filter(|&p| p != me).all(|p| sums[me] < sums[p])) as f64;
                    row[2] += (pred - truth).powi(2);
                    row[3] += pred - truth;
                }
                seed_checks += 1;
            }
            s.apply(me, &ChallengerBot.decide(&v, &mut play_rng))
                .unwrap();
        }
        checks += seed_checks;
        for mode in 0..4 {
            for i in 0..4 {
                error[mode][i] += seed_error[mode][i];
            }
            blocks[mode].push([
                seed_error[mode][1] / (3 * seed_checks) as f64,
                seed_error[mode][2] / seed_checks as f64,
            ]);
        }
    }
    for (i, e) in error.iter().enumerate() {
        println!("blind_keep={i} observations={checks} opponent_bias={:.3} MAE={:.3} lowest_Brier={:.4} lowest_bias={:.4}",e[0]/(checks*3) as f64,e[1]/(checks*3) as f64,e[2]/checks as f64,e[3]/checks as f64);
    }
    for mode in 1..4 {
        for (j, name) in ["MAE", "Brier"].iter().enumerate() {
            let d: Vec<f64> = blocks[mode]
                .iter()
                .zip(&blocks[0])
                .map(|(a, b)| a[j] - b[j])
                .collect();
            let n = d.len() as f64;
            let mean = d.iter().sum::<f64>() / n;
            let ci =
                1.96 * (d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();
            println!("base={base} mode={mode} paired_delta_{name}={mean:+.5} +/- {ci:.5} (100 seed-block 95% CI)");
        }
    }
}
