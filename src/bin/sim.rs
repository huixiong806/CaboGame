//! `cabo-sim`：无头模拟器。让若干 Bot 完整对局，输出统计，用于验证引擎与调参。
//!
//! 用法：`cargo run --release --bin cabo-sim -- --games 500 --players 4 --seed 1`

use cabo::ai::BotRegistry;
use cabo::game::sim::{aggregate, make_ai_session, run_game};
use cabo::game::Settings;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut games = 200usize;
    let mut players = 4usize;
    let mut seed: u64 = 1;
    let mut target: u32 = 100;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--games" => games = args.next().and_then(|v| v.parse().ok()).unwrap_or(games),
            "--players" => players = args.next().and_then(|v| v.parse().ok()).unwrap_or(players),
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            "--target" => target = args.next().and_then(|v| v.parse().ok()).unwrap_or(target),
            other => {
                eprintln!("未知参数 {other}。用法: cabo-sim [--games N] [--players 2-4] [--seed S] [--target T]");
                std::process::exit(2);
            }
        }
    }
    if !(2..=4).contains(&players) {
        eprintln!("players 必须在 2~4");
        std::process::exit(2);
    }

    let registry = BotRegistry::with_builtins();
    let bot_ids: Vec<&str> = (0..players).map(|_| "challenger").collect();
    let names: Vec<String> = (0..players).map(|i| format!("P{}", i + 1)).collect();

    let t0 = std::time::Instant::now();
    let mut outcomes = Vec::with_capacity(games);
    let mut failures = 0usize;
    for i in 0..games {
        let game_seed = seed.wrapping_add(i as u64 * 6364136223846793005);
        let mut s = make_ai_session(game_seed, Settings { cabo_penalty: 10, target_score: target, memory_mode: false }, &bot_ids);
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x9E3779B97F4A7C15);
        match run_game(&mut s, &registry, &mut rng, 50_000) {
            Ok(outcome) => outcomes.push(outcome),
            Err(e) => {
                failures += 1;
                eprintln!("[失败] seed {game_seed}: {e}");
                if failures >= 5 {
                    eprintln!("失败过多，中止");
                    std::process::exit(1);
                }
            }
        }
    }
    let agg = aggregate(&outcomes, &names);
    println!("=== cabo-sim ===");
    println!("对局数 {}（失败 {failures}），耗时 {:.1}s", agg.games, t0.elapsed().as_secs_f64());
    println!("平均轮数 {:.1}，平均动作数 {:.0}，单座位平均累计分 {:.1}", agg.avg_rounds, agg.avg_actions, agg.avg_total);
    for (name, wins) in &agg.wins {
        let pct = if agg.games > 0 { 100.0 * *wins as f64 / agg.games as f64 } else { 0.0 };
        println!("{name}: 获胜 {wins} 次（{pct:.1}%）");
    }
    if failures > 0 {
        std::process::exit(1);
    }
}
