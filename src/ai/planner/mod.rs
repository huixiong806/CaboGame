//! Single-observer information-set planning against a mixture of opponent models.
//! The bot accepts only PlayerView; tree keys contain no unobserved ranks.
mod match_value;
mod policy;
mod posterior;
mod racing;
mod special;
mod state;
#[cfg(test)]
mod tests;

use super::{fallback_command, stats, Bot};
use crate::game::view::{Panel, PlayerView};
use crate::game::Command;
use posterior::Sampler;
use rand::{Rng, RngCore};
use state::{Action, Stage, State};
use std::collections::HashMap;
use std::time::Instant;

fn repeated_public_cycle(view: &PlayerView) -> bool {
    if !matches!(view.panel, Panel::Idle { can_draw: true, .. }) {
        return false;
    }
    let n = view.all_seats.len();
    let events = &view.public_events;
    // At most n selected cards plus the discard can rotate. Require two identical
    // full cycles, including public ranks and selected slots, after information settled.
    (1..=n + 1).any(|turns| {
        let period = n * turns;
        if period == 0 || events.len() < 2 * period {
            return false;
        }
        let tail = &events[events.len() - 2 * period..];
        tail[..period] == tail[period..]
            && tail.iter().all(|e| matches!(e, crate::game::PublicEvent::Exchange {
                source: crate::game::Pile::Discard, slots, exposed, incoming: Some(_), success: true, ..
            } if slots.len() == 1 && exposed.len() == 1))
    })
}

#[derive(Clone, Debug)]
pub struct PlannerCfg {
    /// Break an observed, repeated public discard-exchange cycle by drawing.
    pub avoid_cycles: bool,
    /// Offline research only. Defaults never read a local model or change Normal's utility.
    pub learned_value: bool,
    /// CABOMV01 model; required when learned_value is explicitly enabled.
    pub value_model_path: String,
    /// Hard-only paired root racing; false preserves the frozen Normal planner.
    pub root_racing: bool,
    /// Evidence for voluntarily keeping a draw after replacing an unseen single card.
    pub blind_keep_evidence: bool,
    pub racing_actions: usize,
    /// Rejected development ablation, retained for reproducible comparisons.
    pub probabilistic_reset: bool,
    /// Soft likelihoods for voluntary power trades and discards.
    pub behavioral_evidence: bool,
    /// Permit a call that reliably wins the whole match even when the hand is not lowest.
    pub match_cabo: bool,
    /// Soft likelihood for an opponent's public Cabo declaration.
    pub call_evidence: bool,
    /// Ordinary incumbent calls must pass the same post-response check as search calls.
    pub validate_calls: bool,
    /// 0 => deterministic simulation count, otherwise a wall-clock limit.
    pub budget_us: u64,
    pub simulations: u32,
    /// Own decisions to optimize beyond the root; zero is the flat-rollout ablation.
    pub tree_depth: usize,
    pub max_actions: usize,
    pub exploration: f64,
    pub use_evidence: bool,
    /// Disable for comparison against the pre-special-tactics strategy. Rules remain active.
    pub special_tactics: bool,
    /// Search-proposed calls need this estimated strict-lowest rate after final responses.
    pub min_cabo_success: f64,
    /// Independent paired confirmation against the fast incumbent avoids selection noise.
    pub confirmation_samples: u32,
    pub confirmation_t: f64,
    /// 4 = per-seat style mixture, 0..3 = one fixed style for diagnosis.
    pub opponent_style: u8,
}

impl Default for PlannerCfg {
    fn default() -> Self {
        Self {
            avoid_cycles: false,
            learned_value: false,
            value_model_path: "data/match_value/model-v1.bin".into(),
            root_racing: false,
            blind_keep_evidence: false,
            racing_actions: 12,
            probabilistic_reset: false,
            behavioral_evidence: false,
            match_cabo: false,
            call_evidence: false,
            validate_calls: false,
            budget_us: std::env::var("CABO_V4_BUDGET_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300_000),
            simulations: 4096,
            tree_depth: 6,
            max_actions: 64,
            exploration: 0.10,
            use_evidence: true,
            special_tactics: true,
            min_cabo_success: 0.75,
            confirmation_samples: 48,
            confirmation_t: 1.64,
            opponent_style: 4,
        }
    }
}

impl PlannerCfg {
    /// Normal remains frozen; only independently validated changes are promoted here.
    pub fn hard() -> Self {
        Self::hard_with_model(std::env::var("CABO_HARD_VALUE_MODEL").ok())
    }

    fn hard_with_model(model: Option<String>) -> Self {
        let mut cfg = Self {
            avoid_cycles: true,
            root_racing: false,
            budget_us: std::env::var("CABO_HARD_BUDGET_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600_000),
            simulations: 8192,
            confirmation_samples: 96,
            ..Self::default()
        };
        // Explicit local opt-in: training artifacts are never bundled in Git.
        if let Some(path) = model.filter(|p| !p.trim().is_empty()) {
            cfg.learned_value = true;
            cfg.value_model_path = path;
            cfg.validate_calls = true;
            cfg.min_cabo_success = 0.0;
        }
        cfg
    }
    pub fn set(&mut self, key: &str, value: &str) -> bool {
        match key {
            "avoid_cycles" => match value {
                "true" | "1" => Some(self.avoid_cycles = true),
                "false" | "0" => Some(self.avoid_cycles = false),
                _ => None,
            },
            "value_model_path" => {
                self.value_model_path = value.to_string();
                Some(())
            }
            "learned_value" => match value {
                "true" | "1" => Some(self.learned_value = true),
                "false" | "0" => Some(self.learned_value = false),
                _ => None,
            },
            "validate_calls" => match value {
                "true" | "1" => Some(self.validate_calls = true),
                "false" | "0" => Some(self.validate_calls = false),
                _ => None,
            },
            "call_evidence" => match value {
                "true" | "1" => Some(self.call_evidence = true),
                "false" | "0" => Some(self.call_evidence = false),
                _ => None,
            },
            "match_cabo" => match value {
                "true" | "1" => Some(self.match_cabo = true),
                "false" | "0" => Some(self.match_cabo = false),
                _ => None,
            },
            "behavioral_evidence" => match value {
                "true" | "1" => Some(self.behavioral_evidence = true),
                "false" | "0" => Some(self.behavioral_evidence = false),
                _ => None,
            },
            "probabilistic_reset" => match value {
                "true" | "1" => Some(self.probabilistic_reset = true),
                "false" | "0" => Some(self.probabilistic_reset = false),
                _ => None,
            },
            "racing_actions" => value
                .parse::<usize>()
                .ok()
                .filter(|&v| (2..=64).contains(&v))
                .map(|v| self.racing_actions = v),
            "blind_keep_evidence" => match value {
                "true" | "1" => Some(self.blind_keep_evidence = true),
                "false" | "0" => Some(self.blind_keep_evidence = false),
                _ => None,
            },
            "root_racing" => match value {
                "true" | "1" => Some(self.root_racing = true),
                "false" | "0" => Some(self.root_racing = false),
                _ => None,
            },
            "budget_us" => value.parse().ok().map(|v| self.budget_us = v),
            "simulations" | "worlds" => value
                .parse::<u32>()
                .ok()
                .filter(|&v| v > 0)
                .map(|v| self.simulations = v),
            "tree_depth" => value
                .parse::<usize>()
                .ok()
                .filter(|&v| v <= 32)
                .map(|v| self.tree_depth = v),
            "max_actions" => value
                .parse::<usize>()
                .ok()
                .filter(|&v| (4..=512).contains(&v))
                .map(|v| self.max_actions = v),
            "exploration" => value
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .map(|v| self.exploration = v),
            "opponent_style" => value
                .parse::<u8>()
                .ok()
                .filter(|&v| v <= 4)
                .map(|v| self.opponent_style = v),
            "use_evidence" => match value {
                "true" | "1" => Some(self.use_evidence = true),
                "false" | "0" => Some(self.use_evidence = false),
                _ => None,
            },
            "special_tactics" => match value {
                "true" | "1" => Some(self.special_tactics = true),
                "false" | "0" => Some(self.special_tactics = false),
                _ => None,
            },
            "min_cabo_success" => value
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                .map(|v| self.min_cabo_success = v),
            "confirmation_samples" => value
                .parse::<u32>()
                .ok()
                .map(|v| self.confirmation_samples = v),
            "confirmation_t" => value
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .map(|v| self.confirmation_t = v),
            _ => None,
        }
        .is_some()
    }
}

struct Edge {
    action: Action,
    visits: u32,
    sum: f64,
    prior: f64,
    cabo_ok: u32,
    resets: u32,
    high_pairs: u32,
    match_wins: u32,
}
struct Node {
    visits: u32,
    edges: Vec<Edge>,
}

impl Node {
    fn new(s: &State, me: usize, cfg: &PlannerCfg) -> Self {
        let prior_base = s.utility(me);
        Self {
            visits: 0,
            edges: policy::ranked(s, cfg.max_actions)
                .into_iter()
                .map(|(action, score)| Edge {
                    action,
                    visits: 0,
                    sum: 0.0,
                    cabo_ok: 0,
                    resets: 0,
                    high_pairs: 0,
                    match_wins: 0,
                    prior: (prior_base + score.clamp(-30.0, 30.0) / 100.0).clamp(0.0, 1.0),
                })
                .collect(),
        }
    }

    fn select(&self, exploration: f64) -> usize {
        // Try each action; deterministic ties follow the observation-based tactical ranking.
        if let Some(i) = self.edges.iter().position(|e| e.visits == 0) {
            return i;
        }
        self.edges
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let value = |e: &Edge| {
                    (e.sum + 3.0 * e.prior) / (e.visits as f64 + 3.0)
                        + exploration * ((self.visits as f64 + 1.0).ln() / e.visits as f64).sqrt()
                };
                value(a).total_cmp(&value(b))
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

pub struct PlannerBot {
    pub cfg: PlannerCfg,
    value_model: Option<std::sync::Arc<match_value::MatchValue>>,
}

#[derive(Clone, Debug)]
pub struct CandidateReport {
    pub command: Command,
    pub visits: u32,
    pub mean: f64,
    pub cabo_success: Option<f64>,
    pub reset_probability: f64,
    pub high_pairs_probability: f64,
}

#[derive(Clone, Debug)]
pub struct DecisionReport {
    pub command: Command,
    pub simulations: u32,
    pub nodes: usize,
    pub elapsed_us: u64,
    pub belief_ok: bool,
    pub candidates: Vec<CandidateReport>,
    pub confirmations: u32,
    pub confirmed_gain: f64,
    pub confirmed_se: f64,
}

impl PlannerBot {
    pub fn new(cfg: PlannerCfg) -> Self {
        Self::try_new(cfg).expect("invalid planner model configuration")
    }

    pub fn try_new(cfg: PlannerCfg) -> Result<Self, String> {
        let value_model = if cfg.learned_value {
            Some(match_value::MatchValue::load(std::path::Path::new(
                &cfg.value_model_path,
            ))?)
        } else {
            None
        };
        Ok(Self { cfg, value_model })
    }

    pub fn analyze(&self, view: &PlayerView, rng: &mut dyn RngCore) -> DecisionReport {
        let start = Instant::now();
        let mut report = DecisionReport {
            command: fallback_command(view),
            simulations: 0,
            nodes: 0,
            elapsed_us: 0,
            belief_ok: false,
            candidates: Vec::new(),
            confirmations: 0,
            confirmed_gain: 0.0,
            confirmed_se: 0.0,
        };
        if matches!(view.panel, Panel::PeekPick { .. }) {
            report.command = Command::PeekInitial { slots: [0, 1] };
            return report;
        }
        if matches!(view.panel, Panel::ConfirmCabo) {
            report.command = Command::CallCabo;
            return report;
        }
        let Some(me) = view.me else { return report };
        if self.cfg.avoid_cycles && repeated_public_cycle(view) {
            report.command = Command::BeginDraw;
            report.elapsed_us = start.elapsed().as_micros() as u64;
            return report;
        }
        if self.cfg.root_racing {
            return racing::analyze(
                view,
                rng,
                &self.cfg,
                self.value_model.clone(),
                start,
                report,
            );
        }
        let Some(mut sampler) = Sampler::configured(
            view,
            rng,
            self.cfg.use_evidence,
            self.cfg.blind_keep_evidence,
            self.cfg.behavioral_evidence,
            self.cfg.call_evidence,
        ) else {
            return report;
        };
        report.belief_ok = true;
        sampler.template.special_tactics = self.cfg.special_tactics;
        sampler.template.value_model = self.value_model.clone();
        sampler.template.reset_policy_player = self.cfg.probabilistic_reset.then_some(me);
        let special_call = special::call_relevant(&sampler.template);
        let root_key = sampler.template.observation_key(me);
        let fast = policy::choose(&sampler.template, 1);
        let incumbent = if self.cfg.validate_calls
            && matches!(fast, Action::Cabo)
            && !special_call
            && !sampler.template.deck.is_empty()
        {
            Action::Draw
        } else {
            fast
        };
        let tree_budget = if self.cfg.confirmation_samples > 0 {
            self.cfg.budget_us * 2 / 3
        } else {
            self.cfg.budget_us
        };
        let mut tree = vec![Node::new(&sampler.template, me, &self.cfg)];
        let mut keys = HashMap::new();
        keys.insert(root_key, 0usize);
        let mut path = Vec::with_capacity(self.cfg.tree_depth + 1);
        let mut rollout_actions = 0;
        for _ in 0..self.cfg.simulations {
            if report.simulations > 0
                && self.cfg.budget_us > 0
                && start.elapsed().as_micros() >= tree_budget as u128
            {
                break;
            }
            let mut s = sampler.sample(rng);
            let styles: Vec<u8> = (0..s.n())
                .map(|_| {
                    if self.cfg.opponent_style < 4 {
                        self.cfg.opponent_style
                    } else {
                        rng.random_range(0..3)
                    }
                })
                .collect();
            path.clear();
            let mut own_depth = 0;
            let mut expanded = false;
            let mut steps = 0;
            while s.stage != Stage::End && steps < 512 {
                let action = if s.actor == me && !expanded && own_depth <= self.cfg.tree_depth {
                    let key = s.observation_key(me);
                    let ni = if let Some(&i) = keys.get(&key) {
                        i
                    } else {
                        let i = tree.len();
                        tree.push(Node::new(&s, me, &self.cfg));
                        keys.insert(key, i);
                        i
                    };
                    let ei = tree[ni].select(self.cfg.exploration);
                    expanded = tree[ni].edges[ei].visits == 0;
                    path.push((ni, ei));
                    own_depth += 1;
                    tree[ni].edges[ei].action.clone()
                } else {
                    policy::choose(&s, if s.actor == me { 1 } else { styles[s.actor] })
                };
                s.apply(&action);
                steps += 1;
                // Keep even a pathological simulation bounded by the same per-decision budget.
                if steps % 32 == 0
                    && self.cfg.budget_us > 0
                    && start.elapsed().as_micros() >= tree_budget as u128
                {
                    break;
                }
            }
            // Cycles/timeouts cannot silently reveal hidden hands for scoring a live state.
            if s.stage != Stage::End {
                break;
            }
            let reward = s.utility(me);
            let match_win = *s.totals.iter().max().unwrap() >= s.target && reward > 0.0;
            let own_sum: u32 = s.hands[me]
                .iter()
                .map(|&id| s.cards[id as usize].rank as u32)
                .sum();
            let other_min = (0..s.n())
                .filter(|&p| p != me)
                .map(|p| {
                    s.hands[p]
                        .iter()
                        .map(|&id| s.cards[id as usize].rank as u32)
                        .sum::<u32>()
                })
                .min()
                .unwrap();
            let call_ok = own_sum < other_min;
            let reset = s.score_reset_used[me] && !sampler.template.score_reset_used[me];
            let high_pairs = crate::game::scoring::is_high_pairs(
                &s.hands[me]
                    .iter()
                    .map(|&id| s.cards[id as usize].rank)
                    .collect::<Vec<_>>(),
            );
            for &(ni, ei) in &path {
                let node = &mut tree[ni];
                node.visits += 1;
                let edge = &mut node.edges[ei];
                edge.visits += 1;
                edge.sum += reward;
                edge.resets += u32::from(reset);
                edge.high_pairs += u32::from(high_pairs);
                edge.match_wins += u32::from(match_win);
                if matches!(edge.action, Action::Cabo) && call_ok {
                    edge.cabo_ok += 1;
                }
            }
            report.simulations += 1;
            rollout_actions += steps;
        }
        let root = &tree[0];
        // Most visited is less vulnerable to a lucky, almost unexplored action than raw argmax.
        let best_action = root
            .edges
            .iter()
            .filter(|e| {
                !matches!(e.action, Action::Cabo)
                    || (e.visits >= 16
                        && (special_call
                            || (self.cfg.match_cabo
                                && e.match_wins as f64 / e.visits as f64 >= 0.65)
                            || e.cabo_ok as f64 / e.visits as f64 >= self.cfg.min_cabo_success))
            })
            .max_by(|a, b| {
                a.visits.cmp(&b.visits).then_with(|| {
                    let mean = |e: &Edge| (e.sum + 3.0 * e.prior) / (e.visits as f64 + 3.0);
                    mean(a).total_cmp(&mean(b))
                })
            })
            .map(|edge| edge.action.clone())
            .unwrap_or_else(|| incumbent.clone());
        let mut chosen = best_action.clone();
        if self.cfg.confirmation_samples > 0 && best_action != incumbent {
            // A fresh chain and fresh worlds: evaluation data are not the samples used to pick
            // the challenger. Both branches share the exact world and the same opponent styles.
            chosen = incumbent.clone();
            if let Some(mut confirm) = Sampler::configured(
                view,
                rng,
                self.cfg.use_evidence,
                self.cfg.blind_keep_evidence,
                self.cfg.behavioral_evidence,
                self.cfg.call_evidence,
            ) {
                confirm.template.special_tactics = self.cfg.special_tactics;
                confirm.template.value_model = self.value_model.clone();
                confirm.template.reset_policy_player = self.cfg.probabilistic_reset.then_some(me);
                let mut sum = 0.0;
                let mut sq = 0.0;
                for _ in 0..self.cfg.confirmation_samples {
                    if self.cfg.budget_us > 0
                        && start.elapsed().as_micros() >= self.cfg.budget_us as u128
                    {
                        break;
                    }
                    let world = confirm.sample(rng);
                    let styles: Vec<u8> = (0..world.n())
                        .map(|_| {
                            if self.cfg.opponent_style < 4 {
                                self.cfg.opponent_style
                            } else {
                                rng.random_range(0..3)
                            }
                        })
                        .collect();
                    let mut values = [0.0; 2];
                    let mut complete = true;
                    for (branch, action) in [&best_action, &incumbent].into_iter().enumerate() {
                        let mut state = world.clone();
                        state.apply(action);
                        let mut steps = 0;
                        while state.stage != Stage::End && steps < 512 {
                            let next = policy::choose(
                                &state,
                                if state.actor == me {
                                    1
                                } else {
                                    styles[state.actor]
                                },
                            );
                            state.apply(&next);
                            steps += 1;
                            if steps % 16 == 0
                                && self.cfg.budget_us > 0
                                && start.elapsed().as_micros() >= self.cfg.budget_us as u128
                            {
                                break;
                            }
                        }
                        if state.stage != Stage::End {
                            complete = false;
                            break;
                        }
                        values[branch] = state.utility(me);
                        stats::note_rollout(steps);
                    }
                    if !complete {
                        break;
                    }
                    let d = values[0] - values[1];
                    sum += d;
                    sq += d * d;
                    report.confirmations += 1;
                }
                let n = report.confirmations as f64;
                if n >= 8.0 {
                    report.confirmed_gain = sum / n;
                    report.confirmed_se = ((sq - sum * sum / n).max(0.0) / (n * (n - 1.0))).sqrt();
                    if report.confirmed_gain > self.cfg.confirmation_t * report.confirmed_se {
                        chosen = best_action;
                    }
                }
            }
        }
        report.command = chosen.command();
        report.candidates = root
            .edges
            .iter()
            .map(|e| CandidateReport {
                command: e.action.command(),
                visits: e.visits,
                mean: if e.visits == 0 {
                    e.prior
                } else {
                    e.sum / e.visits as f64
                },
                cabo_success: if matches!(e.action, Action::Cabo) && e.visits > 0 {
                    Some(e.cabo_ok as f64 / e.visits as f64)
                } else {
                    None
                },
                reset_probability: e.resets as f64 / e.visits.max(1) as f64,
                high_pairs_probability: e.high_pairs as f64 / e.visits.max(1) as f64,
            })
            .collect();
        report.nodes = tree.len();
        report.elapsed_us = start.elapsed().as_micros() as u64;
        stats::note_rollouts(report.simulations as u64, rollout_actions);
        report
    }
}

impl Bot for PlannerBot {
    fn id(&self) -> &'static str {
        "planner"
    }
    fn name(&self) -> &'static str {
        "信念规划 AI（v4）"
    }
    fn decide(&self, view: &PlayerView, rng: &mut dyn RngCore) -> Command {
        let start = Instant::now();
        let report = self.analyze(view, rng);
        if std::env::var_os("CABO_V4_DEBUG").is_some() {
            eprintln!(
                "[v4] P{:?} sims={} nodes={} {}us {:?} {:?}",
                view.me,
                report.simulations,
                report.nodes,
                report.elapsed_us,
                report.command,
                report.candidates
            );
        }
        stats::note_decide(start.elapsed().as_nanos() as u64);
        report.command
    }
}

/// Independent fast sparring bot, with accurate public memory and proactive calls.
pub struct ChallengerBot;
/// Offline training roster. The registered Challenger continues to use style 1.
pub struct StyledChallengerBot(pub u8);
impl Bot for StyledChallengerBot {
    fn id(&self) -> &'static str {
        "styled-challenger"
    }
    fn name(&self) -> &'static str {
        "训练陪练"
    }
    fn decide(&self, view: &PlayerView, rng: &mut dyn RngCore) -> Command {
        if matches!(view.panel, Panel::PeekPick { .. }) {
            return Command::PeekInitial { slots: [0, 1] };
        }
        if matches!(view.panel, Panel::ConfirmCabo) {
            return Command::CallCabo;
        }
        Sampler::new(view, rng, false)
            .map(|s| policy::choose(&s.template, self.0.min(2)).command())
            .unwrap_or_else(|| fallback_command(view))
    }
}
impl Bot for ChallengerBot {
    fn id(&self) -> &'static str {
        "challenger"
    }
    fn name(&self) -> &'static str {
        "主动战术 AI（陪练）"
    }
    fn decide(&self, view: &PlayerView, rng: &mut dyn RngCore) -> Command {
        if matches!(view.panel, Panel::PeekPick { .. }) {
            return Command::PeekInitial { slots: [0, 1] };
        }
        if matches!(view.panel, Panel::ConfirmCabo) {
            return Command::CallCabo;
        }
        Sampler::new(view, rng, false)
            .map(|sampler| policy::choose(&sampler.template, 1).command())
            .unwrap_or_else(|| fallback_command(view))
    }
}
