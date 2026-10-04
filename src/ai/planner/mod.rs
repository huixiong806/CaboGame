//! Single-observer information-set planning against a mixture of opponent models.
//! The bot accepts only PlayerView; tree keys contain no unobserved ranks.
mod policy;
mod posterior;
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

#[derive(Clone, Debug)]
pub struct PlannerCfg {
    /// 0 => deterministic simulation count, otherwise a wall-clock limit.
    pub budget_us: u64,
    pub simulations: u32,
    /// Own decisions to optimize beyond the root; zero is the flat-rollout ablation.
    pub tree_depth: usize,
    pub max_actions: usize,
    pub exploration: f64,
    pub use_evidence: bool,
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
            budget_us: std::env::var("CABO_V4_BUDGET_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300_000),
            simulations: 4096,
            tree_depth: 6,
            max_actions: 64,
            exploration: 0.10,
            use_evidence: true,
            min_cabo_success: 0.75,
            confirmation_samples: 48,
            confirmation_t: 1.64,
            opponent_style: 4,
        }
    }
}

impl PlannerCfg {
    pub fn set(&mut self, key: &str, value: &str) -> bool {
        match key {
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
}

#[derive(Clone, Debug)]
pub struct CandidateReport {
    pub command: Command,
    pub visits: u32,
    pub mean: f64,
    pub cabo_success: Option<f64>,
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
        Self { cfg }
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
        let Some(mut sampler) = Sampler::new(view, rng, self.cfg.use_evidence) else {
            return report;
        };
        report.belief_ok = true;
        let root_key = sampler.template.observation_key(me);
        let incumbent = policy::choose(&sampler.template, 1);
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
            for &(ni, ei) in &path {
                let node = &mut tree[ni];
                node.visits += 1;
                let edge = &mut node.edges[ei];
                edge.visits += 1;
                edge.sum += reward;
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
                        && e.cabo_ok as f64 / e.visits as f64 >= self.cfg.min_cabo_success)
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
            if let Some(mut confirm) = Sampler::new(view, rng, self.cfg.use_evidence) {
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
