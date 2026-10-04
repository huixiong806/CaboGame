//! Paired root policy improvement. All contenders see the same posterior world,
//! deck order and per-seat opponent style. Continuations still act on their own knowledge.
use super::*;

#[derive(Default)]
struct Estimate {
    n: u32,
    sum: f64,
    cabo: u32,
    resets: u32,
    pairs: u32,
    match_wins: u32,
}
impl Estimate {
    fn mean(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }
}

fn styles(s: &State, cfg: &PlannerCfg, rng: &mut dyn RngCore) -> Vec<u8> {
    (0..s.n())
        .map(|_| {
            if cfg.opponent_style < 4 {
                cfg.opponent_style
            } else {
                rng.random_range(0..3)
            }
        })
        .collect()
}

fn rollout(
    world: &State,
    action: &Action,
    me: usize,
    styles: &[u8],
    start: Instant,
    limit: u64,
) -> Option<State> {
    let mut s = world.clone();
    s.apply(action);
    for step in 0..512 {
        if s.stage == Stage::End {
            return Some(s);
        }
        if step % 16 == 0 && limit > 0 && start.elapsed().as_micros() >= limit as u128 {
            return None;
        }
        let a = policy::choose(&s, if s.actor == me { 1 } else { styles[s.actor] });
        s.apply(&a);
    }
    None
}

pub(super) fn analyze(
    view: &PlayerView,
    rng: &mut dyn RngCore,
    cfg: &PlannerCfg,
    value_model: Option<std::sync::Arc<match_value::MatchValue>>,
    start: Instant,
    mut report: DecisionReport,
) -> DecisionReport {
    let me = view.me.unwrap();
    let Some(mut sampler) = Sampler::configured(
        view,
        rng,
        cfg.use_evidence,
        cfg.blind_keep_evidence,
        cfg.behavioral_evidence,
        cfg.call_evidence,
    ) else {
        return report;
    };
    sampler.template.special_tactics = cfg.special_tactics;
    sampler.template.value_model = value_model.clone();
    sampler.template.reset_policy_player = cfg.probabilistic_reset.then_some(me);
    report.belief_ok = true;
    let special_call = special::call_relevant(&sampler.template);
    let fast = policy::choose(&sampler.template, 1);
    let incumbent = if cfg.validate_calls
        && matches!(fast, Action::Cabo)
        && !special_call
        && !sampler.template.deck.is_empty()
    {
        Action::Draw
    } else {
        fast
    };
    // Full legal candidate generation happens before observation-only heuristic screening.
    // Retain the call, draw and incumbent even when they fall below the initial shortlist.
    let ranked = policy::ranked(&sampler.template, cfg.max_actions);
    let mut actions: Vec<Action> = ranked
        .iter()
        .take(cfg.racing_actions)
        .map(|(a, _)| a.clone())
        .collect();
    for a in ranked
        .iter()
        .map(|(a, _)| a)
        .filter(|a| matches!(a, Action::Draw | Action::Cabo) || **a == incumbent)
    {
        if !actions.contains(a) {
            actions.push(a.clone());
        }
    }
    if !actions.contains(&incumbent) {
        actions.push(incumbent.clone());
    }
    let mut estimates: Vec<Estimate> = actions.iter().map(|_| Estimate::default()).collect();
    let mut active: Vec<usize> = (0..actions.len()).collect();
    let limit = if cfg.confirmation_samples > 0 {
        cfg.budget_us * 2 / 3
    } else {
        cfg.budget_us
    };
    // Count individual trajectories, like the existing planner. Never use partial batches:
    // even a timeout must not privilege cheap-to-simulate actions.
    let mut batch_size = 8;
    let mut batch = 0;
    while report.simulations + active.len() as u32 <= cfg.simulations {
        if limit > 0 && start.elapsed().as_micros() >= limit as u128 {
            break;
        }
        let world = sampler.sample(rng);
        let style = styles(&world, cfg, rng);
        let states: Option<Vec<State>> = active
            .iter()
            .map(|&i| rollout(&world, &actions[i], me, &style, start, limit))
            .collect();
        let Some(states) = states else { break };
        for (&i, s) in active.iter().zip(states) {
            let e = &mut estimates[i];
            e.n += 1;
            e.sum += s.utility(me);
            e.match_wins +=
                u32::from(s.totals.iter().any(|&t| t >= s.target) && s.utility(me) > 0.0);
            e.resets += u32::from(s.score_reset_used[me] && !world.score_reset_used[me]);
            let hand = |p: usize| {
                s.hands[p]
                    .iter()
                    .map(|&id| s.cards[id as usize].rank)
                    .collect::<Vec<_>>()
            };
            let own = hand(me);
            let sum = own.iter().map(|&r| r as u32).sum::<u32>();
            e.cabo += u32::from(
                (0..s.n())
                    .filter(|&p| p != me)
                    .all(|p| sum < hand(p).iter().map(|&r| r as u32).sum()),
            );
            e.pairs += u32::from(crate::game::scoring::is_high_pairs(&own));
            report.simulations += 1;
        }
        batch += 1;
        if batch == batch_size && active.len() > 2 {
            active.sort_by(|&a, &b| estimates[b].mean().total_cmp(&estimates[a].mean()));
            active.truncate(active.len().div_ceil(2));
            batch = 0;
            batch_size *= 2;
        }
    }
    let allowed = |i: usize| {
        !matches!(actions[i], Action::Cabo)
            || (estimates[i].n >= 16
                && (special_call
                    || (cfg.match_cabo
                        && estimates[i].match_wins as f64 / estimates[i].n as f64 >= 0.65)
                    || estimates[i].cabo as f64 / estimates[i].n as f64 >= cfg.min_cabo_success))
    };
    let best = active
        .iter()
        .copied()
        .filter(|&i| estimates[i].n > 0 && allowed(i))
        .max_by(|&a, &b| estimates[a].mean().total_cmp(&estimates[b].mean()));
    let proposal = best
        .map(|i| actions[i].clone())
        .unwrap_or_else(|| incumbent.clone());
    let mut chosen = proposal.clone();
    if cfg.confirmation_samples > 0 && proposal != incumbent {
        chosen = incumbent.clone();
        if let Some(mut fresh) = Sampler::configured(
            view,
            rng,
            cfg.use_evidence,
            cfg.blind_keep_evidence,
            cfg.behavioral_evidence,
            cfg.call_evidence,
        ) {
            fresh.template.special_tactics = cfg.special_tactics;
            fresh.template.value_model = value_model.clone();
            fresh.template.reset_policy_player = cfg.probabilistic_reset.then_some(me);
            let (mut sum, mut sq) = (0.0, 0.0);
            for _ in 0..cfg.confirmation_samples {
                let world = fresh.sample(rng);
                let style = styles(&world, cfg, rng);
                let (Some(a), Some(b)) = (
                    rollout(&world, &proposal, me, &style, start, cfg.budget_us),
                    rollout(&world, &incumbent, me, &style, start, cfg.budget_us),
                ) else {
                    break;
                };
                let d = a.utility(me) - b.utility(me);
                sum += d;
                sq += d * d;
                report.confirmations += 1;
            }
            let n = report.confirmations as f64;
            if n >= 8.0 {
                report.confirmed_gain = sum / n;
                report.confirmed_se = ((sq - sum * sum / n).max(0.0) / (n * (n - 1.0))).sqrt();
                if report.confirmed_gain > cfg.confirmation_t * report.confirmed_se {
                    chosen = proposal;
                }
            }
        }
    }
    report.command = chosen.command();
    report.candidates = actions
        .iter()
        .zip(estimates)
        .map(|(a, e)| CandidateReport {
            command: a.command(),
            visits: e.n,
            mean: e.mean(),
            cabo_success: if matches!(a, Action::Cabo) && e.n > 0 {
                Some(e.cabo as f64 / e.n as f64)
            } else {
                None
            },
            reset_probability: e.resets as f64 / e.n.max(1) as f64,
            high_pairs_probability: e.pairs as f64 / e.n.max(1) as f64,
        })
        .collect();
    report.nodes = 1;
    report.elapsed_us = start.elapsed().as_micros() as u64;
    report
}
