//! `cabo-decide`：阶段 0 的**成对决策检验**。
//!
//! 回答一个必须先回答的问题：**搜索选的动作 a\*，是否真的比启发式策略选的动作 a_p 更好？**
//! 如果连这都不成立，那么"用搜索当教师产标签"就是把噪声固化。
//!
//! 做法（同一确定性世界配对，运气项抵消）：
//! 1. 让当前最强搜索自我对弈，遇到它和启发式**意见不同**的决策点时取样；
//! 2. 对 K 个**共享的确定性世界**分别把 a\* 与 a_p 走一遍，各自打到本轮结算；
//! 3. 比较"本轮结束时与最强劲对手的分差"，做配对统计。
//!
//! 用法：
//! ```text
//! cargo run --release --bin cabo-decide -- --games 20 --worlds 24 --budget-us 4000
//! ```

use std::sync::Arc;
use std::time::Instant;

use cabo::ai::belief::reconstruct;
use cabo::ai::history::History;
use cabo::ai::search::candidates;
use cabo::ai::policy::PolicyBot;
use cabo::ai::Bot;
use cabo::ai::tactics::{policy, raw_fallback, view_policy_with, Know, PolicyCfg};
use cabo::ai::{build_bot, BotRegistry};
use cabo::game::sim::make_ai_session;
use cabo::game::view::{project, PlayerView};
use cabo::game::{Command, Phase, PlayerId, Session, Settings};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

/// 把局面打到本轮结算，返回"与最强劲对手的分差"（越大越好）。
fn rollout_lead(s: &mut Session, me: PlayerId, cfg: &PolicyCfg, rng: &mut dyn RngCore) -> f64 {
    let n = s.players.len();
    let mut guard = 0usize;
    loop {
        let pid = match &s.phase {
            Phase::Peeking { done } => (0..n).find(|p| !done.contains_key(p)),
            Phase::Turn { current, .. } => Some(*current),
            _ => None,
        };
        let Some(pid) = pid else { break };
        let cmd = policy(s, pid, cfg, rng);
        if s.apply(pid, &cmd).is_err() {
            let fb = raw_fallback(s, pid);
            if s.apply(pid, &fb).is_err() {
                break;
            }
        }
        guard += 1;
        if guard > 4000 {
            break;
        }
    }
    let my = s.players[me].total_score as f64;
    let min_other = s
        .players
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != me)
        .map(|(_, p)| p.total_score as f64)
        .fold(f64::MAX, f64::min);
    min_other - my
}

struct Args {
    games: usize,
    players: usize,
    worlds: u32,
    budget_us: u64,
    seed: u64,
    max_cands: usize,
    target: u32,
    verbose: bool,
    /// 与教师对比的"先验"策略：heuristic（默认）或 policy（学习到的策略网络）
    prior: String,
    /// 预言机模式：对比"视图决策"与"全知决策"，量化信息的机会成本。
    oracle: bool,
    /// 预言机档位：`full`（全知）或 `discard`（只给真实弃牌堆，公开信息）。
    oracle_mode: String,
}

fn parse() -> Args {
    let mut a = Args {
        games: 20,
        players: 4,
        worlds: 24,
        budget_us: 4_000,
        seed: 1,
        max_cands: 10,
        target: 100,
        verbose: false,
        prior: String::from("heuristic"),
        oracle: false,
        oracle_mode: String::from("full"),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_default();
        match k.as_str() {
            "--games" => a.games = v().parse().unwrap_or(a.games),
            "--players" => a.players = v().parse().unwrap_or(a.players),
            "--worlds" => a.worlds = v().parse().unwrap_or(a.worlds),
            "--budget-us" => a.budget_us = v().parse().unwrap_or(a.budget_us),
            "--seed" => a.seed = v().parse().unwrap_or(a.seed),
            "--max-cands" => a.max_cands = v().parse().unwrap_or(a.max_cands),
            "--target" => a.target = v().parse().unwrap_or(a.target),
            "--verbose" => a.verbose = true,
            "--prior" => a.prior = v(),
            "--oracle" => a.oracle = true,
            "--oracle-mode" => a.oracle_mode = v(),
            other => {
                eprintln!("未知参数 {other}");
                std::process::exit(2);
            }
        }
    }
    a
}


/// 构造"全知"信念快照：所有手牌、牌堆、弃牌堆都可见（预言机用）。
///
/// 只用于**离线诊断**（`examples/` 与 `bin/decide` 这类工具），绝不参与线上决策。
fn oracle_know(s: &cabo::game::Session, me: PlayerId, mode: &str) -> Know {
    if mode == "discard" {
        // 只补上"真实弃牌堆构成"（公开信息），其余仍按视图。
        let mut td = [0u16; 14];
        for cid in &s.discard {
            td[s.cards[*cid as usize].card.rank as usize] += 1;
        }
        return Know::from_view_with(&cabo::game::view::project(s, Some(me), 0), &td);
    }
    use cabo::game::RANK_COPIES;
    let n = s.players.len();
    let mut hands: Vec<Vec<Option<u8>>> = Vec::with_capacity(n);
    for p in 0..n {
        hands.push(
            s.players[p]
                .slots
                .iter()
                .map(|cid| Some(s.cards[*cid as usize].card.rank))
                .collect(),
        );
    }
    let mut pool = [0u16; 14];
    for r in 0..14 {
        pool[r] = RANK_COPIES[r] as u16;
    }
    for cid in &s.discard {
        pool[s.cards[*cid as usize].card.rank as usize] -= 1;
    }
    for p in 0..n {
        for cid in &s.players[p].slots {
            pool[s.cards[*cid as usize].card.rank as usize] -= 1;
        }
    }
    let mut k = Know::from_view_with(
        &cabo::game::view::project(s, Some(me), 0),
        &[0u16; 14],
    );
    k.hands = hands;
    k.pool = pool;
    k.deck_count = s.deck.len();
    k.discard_count = s.discard.len();
    k.discard_top = s
        .discard
        .last()
        .map(|c| s.cards[*c as usize].card.rank);
    k
}

fn main() {
    let args = parse();
    let settings = Settings {
        cabo_penalty: 10,
        target_score: args.target,
        memory_mode: false,
    };
    let teacher_id: &'static str = "teacher";
    let mut registry = BotRegistry::new();
    let teacher = build_bot(
        "search",
        &[("budget_us".to_string(), args.budget_us.to_string())],
    )
    .expect("搜索 Bot");
    registry.register(Arc::new(cabo::ai::AliasBot {
        inner: teacher,
        id: teacher_id,
        name: "teacher",
    }));
    let policy_cfg = PolicyCfg::default();
    let policy_bot = PolicyBot::new();
    // 预言机用的同一套搜索（只把信念快照换成全知）
    let oracle_bot = cabo::ai::search::SearchBot::new(cabo::ai::search::SearchCfg {
        budget_us: args.budget_us,
        ..cabo::ai::search::SearchCfg::default()
    });
    let use_policy_prior = args.prior == "policy";
    if use_policy_prior {
        assert!(cabo::ai::policy::available(), "策略网络权重缺失，先用 cabo-train 训练");
    }

    let mut compared = 0usize; // 有分歧的决策数
    let mut agreed = 0usize; // 意见一致的决策数
    let mut better = 0usize; // a* 更好
    let mut worse = 0usize; // a_p 更好
    let mut tie = 0usize;
    let mut sum_d = 0f64;
    let mut sum_d2 = 0f64;
    let mut n_pairs = 0f64;
    let t0 = Instant::now();

    for g in 0..args.games {
        let game_seed = args.seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let bots: Vec<&str> = (0..args.players).map(|_| teacher_id).collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &bots);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
        let history = History::new();
        for _ in 0..200_000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..args.players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                Phase::RoundEnd => {
                    s.next_round().unwrap();
                    continue;
                }
                Phase::GameOver { .. } | Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            let view = project(&s, Some(pid), 0);

            // ---- 只在"我的回合、空闲面板"上做检验（其它面板留给下一步）----
            if matches!(view.panel, cabo::game::view::Panel::Idle { .. }) {
                let info = history.observe(&view);
                let k = Know::from_view_with(&view, &info.known);
                let cands = candidates(&view, &k, args.max_cands);
                let a_teacher = registry.get(teacher_id).unwrap().decide(&view, &mut rng);
                let a_prior = if args.oracle {
                    // 全知弃牌堆（预言机连弃牌堆构成也知道）
                    let mut td = [0u16; 14];
                    for cid in &s.discard {
                        td[s.cards[*cid as usize].card.rank as usize] += 1;
                    }
                    oracle_bot.decide_with_know(&view, &oracle_know(&s, pid, &args.oracle_mode), &td, &mut rng)
                } else if use_policy_prior {
                    policy_bot.decide(&view, &mut rng)
                } else {
                    view_policy_with(&view, &policy_cfg, &mut rng, &info.known)
                };
                if cands.len() >= 2 {
                    if a_teacher == a_prior {
                        agreed += 1;
                    } else {
                        compared += 1;
                        let mut d_sum = 0f64;
                        let mut d_n = 0f64;
                        for w in 0..args.worlds {
                            let mut wrng = StdRng::seed_from_u64(
                                game_seed ^ (w as u64).wrapping_mul(0x9E3779B97F4A7C15),
                            );
                            let Some(world) = reconstruct(&view, &mut wrng, &info.known) else {
                                continue;
                            };
                            // 两侧共用同一随机流：轨迹相同处随机性完全一致
                            let world_seed = 0x51ED2701u64 ^ (w as u64) << 32 ^ pid as u64;
                            let mut sa = world.clone();
                            let mut sb = world.clone();
                            if sa.apply(pid, &a_teacher).is_err() {
                                continue;
                            }
                            if sb.apply(pid, &a_prior).is_err() {
                                continue;
                            }
                            let mut ra = StdRng::seed_from_u64(world_seed);
                            let mut rb = StdRng::seed_from_u64(world_seed);
                            let va = rollout_lead(&mut sa, pid, &policy_cfg, &mut ra);
                            let vb = rollout_lead(&mut sb, pid, &policy_cfg, &mut rb);
                            d_sum += va - vb;
                            d_n += 1.0;
                        }
                        if d_n > 0.0 {
                            let d = d_sum / d_n;
                            sum_d += d;
                            sum_d2 += d * d;
                            n_pairs += 1.0;
                            if d > 1e-9 {
                                better += 1;
                            } else if d < -1e-9 {
                                worse += 1;
                            } else {
                                tie += 1;
                            }
                            if args.verbose && compared % 25 == 0 {
                                eprintln!(
                                    "[决策 #{compared}] P{pid} 教师 {:?} vs 先验 {:?} → Δ={d:+.2}（{d_n} 世界）",
                                    a_teacher, a_prior
                                );
                            }
                        }
                    }
                }
            }

            let cmd = registry.get(teacher_id).unwrap().decide(&view, &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
    }

    println!(
        "=== cabo-decide：成对决策检验（教师 vs {}）===",
        if args.oracle {
            if args.oracle_mode == "discard" {
                "预言机（只给真实弃牌堆）"
            } else {
                "完美信息预言机（全知）"
            }
        } else if args.prior == "policy" {
            "学习策略"
        } else {
            "启发式先验"
        }
    );
    println!(
        "自我对弈 {} 局（{} 人，预算 {}us），耗时 {:.0}s",
        args.games,
        args.players,
        args.budget_us,
        t0.elapsed().as_secs_f64()
    );
    let total = compared + agreed;
    println!(
        "空闲面板决策 {} 个：教师与先验**意见一致** {} 个（{:.1}%），有分歧 {} 个",
        total,
        agreed,
        100.0 * agreed as f64 / total.max(1) as f64,
        compared
    );
    if n_pairs > 0.0 {
        let mean = sum_d / n_pairs;
        let var = (sum_d2 / n_pairs - mean * mean).max(0.0);
        let se = (var / n_pairs).sqrt();
        println!(
            "配对比较 {} 个决策 × {} 世界：平均 Δ(教师−先验) = {:+.3} ± {:.3}（1se）",
            n_pairs as usize, args.worlds, mean, se
        );
        println!(
            "逐决策胜负：教师更好 {} / 先验更好 {} / 打平 {}（胜率 {:.1}%）",
            better,
            worse,
            tie,
            100.0 * better as f64 / (better + worse).max(1) as f64
        );
        if mean.abs() > 2.0 * se.max(1e-9) {
            println!(
                "→ 结论：搜索的选择{}显著更好（t≈{:.1}）",
                if mean > 0.0 { "" } else { "**不**" },
                mean / se.max(1e-9)
            );
        } else {
            println!("→ 结论：**没有显著差异**——此时用搜索当教师产标签只是在固化噪声");
        }
    } else {
        println!("没有采集到有分歧的决策，样本不足");
    }
}
