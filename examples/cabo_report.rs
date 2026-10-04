//! 宣告（Cabo）决策诊断：搜索 AI 实际宣告的**成功率**与**频率**是多少？
//!
//! 动机（DL_ROUTE.md E29/E31）：决策稳定点扫描显示搜索的选择在 K≈256 就不再变化，
//! 唯一会翻的正是 `BeginDraw` vs `CallCabo` 这类高影响决策 —— 一次翻转值 10~20 分。
//! 但宣告判定里有两个**人工常数**：`cabo_min_success=0.75` 与 `cabo_margin`（点数）。
//! 本工具量出它们在真实对局里的表现，看门槛是否标定正确、有没有 headroom。
//!
//! `cargo run --release --example cabo_report -- --games 6 --players 4`

use std::sync::Arc;

use cabo::ai::search::{SearchBot, SearchCfg};
use cabo::ai::{Bot, BotRegistry};
use cabo::game::sim::make_ai_session;
use cabo::game::view::project;
use cabo::game::{Command, Phase, PlayerId, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Default)]
struct SeatStat {
    calls: usize,
    success: usize,
    /// 宣告当轮该座位自己的点数
    round_score_sum: i64,
    /// 同轮其他座位的最低点数之和（用来对比"宣告划算吗"）
    best_other_sum: i64,
    /// 不宣告时（继续打）该座位的本轮点数
    cont_round_sum: i64,
    cont_n: usize,
}

fn main() {
    let mut games = 6usize;
    let mut players = 4usize;
    let mut seed = 1u64;
    let mut budget_us = 300_000u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_default();
        match a.as_str() {
            "--games" => games = v().parse().unwrap_or(games),
            "--players" => players = v().parse().unwrap_or(players),
            "--seed" => seed = v().parse().unwrap_or(seed),
            "--budget-us" => budget_us = v().parse().unwrap_or(budget_us),
            other => {
                eprintln!("未知参数 {other}");
                std::process::exit(2);
            }
        }
    }

    let settings = Settings { cabo_penalty: 10, target_score: 100, memory_mode: false };
    let mut registry = BotRegistry::new();
    registry.register(Arc::new(SearchBot::new(SearchCfg {
        budget_us,
        ..SearchCfg::default()
    })));

    let mut stats: Vec<SeatStat> = (0..players).map(|_| SeatStat::default()).collect();
    let mut rounds = 0usize;
    let t0 = std::time::Instant::now();

    for g in 0..games {
        let game_seed = seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let ids: Vec<&str> = (0..players).map(|_| "search").collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &ids);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
        // 本轮里谁宣告了（用于结算时记账）
        let mut caller_this_round: Option<PlayerId> = None;
        for _ in 0..200_000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                Phase::RoundEnd => {
                    rounds += 1;
                    if let Some(caller) = caller_this_round {
                        stats[caller].calls += 1;
                        let mine = s.players[caller].round_score.unwrap_or(0) as i64;
                        stats[caller].round_score_sum += mine;
                        let best_other = s
                            .players
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != caller)
                            .map(|(_, p)| p.round_score.unwrap_or(0) as i64)
                            .min()
                            .unwrap_or(0);
                        stats[caller].best_other_sum += best_other;
                        if mine == 0 {
                            stats[caller].success += 1;
                        }
                    }
                    caller_this_round = None;
                    s.next_round().unwrap();
                    continue;
                }
                Phase::GameOver { .. } | Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            let view = project(&s, Some(pid), 0);
            let cmd = registry.get("search").unwrap().decide(&view, &mut rng);
            if matches!(cmd, Command::CallCabo) {
                caller_this_round = Some(pid);
            }
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
    }

    let total_calls: usize = stats.iter().map(|x| x.calls).sum();
    let total_success: usize = stats.iter().map(|x| x.success).sum();
    println!("=== 宣告决策诊断（{} 局 / {} 轮，预算 {}µs）===", games, rounds, budget_us);
    println!(
        "宣告次数 {}（每局 {:.2} 次），其中**严格最低**（宣告成功）{} → 成功率 **{:.1}%**",
        total_calls,
        total_calls as f64 / games.max(1) as f64,
        total_success,
        100.0 * total_success as f64 / total_calls.max(1) as f64
    );
    if total_calls > 0 {
        let mine: i64 = stats.iter().map(|x| x.round_score_sum).sum();
        let other: i64 = stats.iter().map(|x| x.best_other_sum).sum();
        println!(
            "宣告当轮：自己的点数均值 {:.1}，对手最低点数均值 {:.1} → 每轮净收益 {:.1} 分",
            mine as f64 / total_calls as f64,
            other as f64 / total_calls as f64,
            (other - mine) as f64 / total_calls as f64
        );
    }
    println!("（门槛 cabo_min_success=0.75 的语义：模拟成功率低于 75% 就不宣告）");
    println!("耗时 {:.0}s", t0.elapsed().as_secs_f64());
}
