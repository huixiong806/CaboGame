//! `cabo-arena`：两套 Bot 的配对自对弈擂台，用于版本迭代与调参。
//!
//! 用法：
//! ```text
//! cargo run --release --bin cabo-arena -- --a planner --b challenger --games 20
//! ```
//!
//! 配对设计：同一个种子跑两局，把 A/B 组的座位对调（发牌完全相同），
//! 这样座位先手与牌运的偏差会被抵消，统计量的方差大幅下降。

use cabo::ai::BotRegistry;
use cabo::game::sim::{make_ai_session, run_game, GameOutcome};
use cabo::game::Settings;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Args {
    a: String,
    b: String,
    a_cfg: Vec<(String, String)>,
    b_cfg: Vec<(String, String)>,
    players: usize,
    a_seats: usize,
    games: usize,
    seed: u64,
    target: u32,
    penalty: u32,
}

/// 解析 `k=v,k=v` 形式的参数覆盖。
fn parse_cfg(s: &str) -> Vec<(String, String)> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .filter_map(|p| {
            let (k, v) = p.split_once('=')?;
            Some((k.trim().to_string(), v.trim().to_string()))
        })
        .collect()
}

fn parse_args() -> Args {
    let mut a = "planner".to_string();
    let mut b = "challenger".to_string();
    let mut a_cfg = Vec::new();
    let mut b_cfg = Vec::new();
    let mut players = 4usize;
    let mut a_seats = 0usize;
    let mut games = 200usize;
    let mut seed = 1u64;
    let mut target = 100u32;
    let mut penalty = 10u32;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_default();
        match arg.as_str() {
            "--a" => a = val(),
            "--b" => b = val(),
            "--a-cfg" => a_cfg = parse_cfg(&val()),
            "--b-cfg" => b_cfg = parse_cfg(&val()),
            "--players" => players = val().parse().unwrap_or(players),
            "--a-seats" => a_seats = val().parse().unwrap_or(a_seats),
            "--games" => games = val().parse().unwrap_or(games),
            "--seed" => seed = val().parse().unwrap_or(seed),
            "--target" => target = val().parse().unwrap_or(target),
            "--penalty" => penalty = val().parse().unwrap_or(penalty),
            other => {
                eprintln!("未知参数 {other}");
                std::process::exit(2);
            }
        }
    }
    if !(2..=4).contains(&players) {
        eprintln!("players 必须在 2~4");
        std::process::exit(2);
    }
    if a_seats == 0 {
        a_seats = players / 2;
    }
    if a_seats >= players {
        a_seats = players - 1;
    }
    Args { a, b, a_cfg, b_cfg, players, a_seats, games, seed, target, penalty }
}

#[derive(Default, Clone, Copy)]
struct Side {
    games: usize,
    wins: f64,
    total_sum: f64,
    total_n: f64,
}

impl Side {
    fn add(&mut self, o: &GameOutcome, seats: &[usize]) {
        self.games += 1;
        let mut won = false;
        for &s in seats {
            self.total_sum += o.totals[s] as f64;
            self.total_n += 1.0;
            if o.winners.contains(&s) {
                won = true;
            }
        }
        self.wins += if won { 1.0 } else { 0.0 };
    }
    fn avg_total(&self) -> f64 {
        if self.total_n > 0.0 {
            self.total_sum / self.total_n
        } else {
            0.0
        }
    }
}

fn main() {
    let args = parse_args();
    let mut registry = BotRegistry::new();
    let bot_a = match cabo::ai::build_bot(&args.a, &args.a_cfg) {
        Some(b) => b,
        None => {
            eprintln!("未注册的 bot: {}", args.a);
            std::process::exit(2);
        }
    };
    let bot_b = match cabo::ai::build_bot(&args.b, &args.b_cfg) {
        Some(b) => b,
        None => {
            eprintln!("未注册的 bot: {}", args.b);
            std::process::exit(2);
        }
    };
    // 注册 id 一律用"命令行里的名字"：版本别名底层可能是同一个实现，
    // 直接注册会撞 id；同 id 不同参数也要能区分开。
    let a_key: &'static str = Box::leak(args.a.clone().into_boxed_str());
    let b_key: &'static str = Box::leak(
        if args.a == args.b { format!("{}#b", args.b) } else { args.b.clone() }.into_boxed_str(),
    );
    let a_name = bot_a.name();
    let b_name = bot_b.name();
    registry.register(std::sync::Arc::new(cabo::ai::AliasBot {
        inner: bot_a,
        id: a_key,
        name: a_name,
    }));
    registry.register(std::sync::Arc::new(cabo::ai::AliasBot {
        inner: bot_b,
        id: b_key,
        name: b_name,
    }));
    let (a_id, b_id) = (a_key, b_key);
    let settings =
        Settings { cabo_penalty: args.penalty, target_score: args.target, memory_mode: false };

    let mut side_a = Side::default();
    let mut side_b = Side::default();
    let mut pairs = 0usize;
    let mut rounds = 0u64;
    let mut diff_sum = 0f64;
    let mut diff_sq = 0f64;
    // "两边最强座位"的比较：胜负由**最低**累计分决定，平均分会被"某个座位爆掉"带偏，
    // 因此这个指标（每局 min(A 座位) − min(B 座位)）比平均分更贴近真实胜负。
    let mut best_sum = 0f64;
    let mut best_sq = 0f64;
    let mut best_n = 0usize;
    let mut win_a = 0f64;
    let mut win_b = 0f64;
    let mut failures = 0usize;
    let t0 = std::time::Instant::now();

    for i in 0..args.games {
        let game_seed = args.seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        // 座位排布：正向 / 镜像（同一种子 ⇒ 同一副牌）。
        let bots_fwd: Vec<&str> = (0..args.players)
            .map(|s| if s < args.a_seats { a_id } else { b_id })
            .collect();
        let bots_rev: Vec<&str> = (0..args.players)
            .map(|s| if s < args.players - args.a_seats { b_id } else { a_id })
            .collect();

        let mut pair_diff: Vec<f64> = Vec::new();
        for bots in [&bots_fwd, &bots_rev] {
            let mut session = make_ai_session(game_seed, settings.clone(), bots);
            let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
            match run_game(&mut session, &registry, &mut rng, 200_000) {
                Ok(o) => {
                    rounds += o.rounds as u64;
                    let a_seats: Vec<usize> =
                        (0..args.players).filter(|&s| bots[s] == a_id).collect();
                    let b_seats: Vec<usize> =
                        (0..args.players).filter(|&s| bots[s] == b_id).collect();
                    side_a.add(&o, &a_seats);
                    side_b.add(&o, &b_seats);
                    let sa: f64 = a_seats.iter().map(|&s| o.totals[s] as f64).sum::<f64>()
                        / a_seats.len() as f64;
                    let sb: f64 = b_seats.iter().map(|&s| o.totals[s] as f64).sum::<f64>()
                        / b_seats.len() as f64;
                    pair_diff.push(sa - sb);
                    // 每局"两边各自最低累计分"的差（真正决定胜负的量）。
                    let ma = a_seats.iter().map(|&s| o.totals[s]).min().unwrap_or(0) as f64;
                    let mb = b_seats.iter().map(|&s| o.totals[s]).min().unwrap_or(0) as f64;
                    best_sum += ma - mb;
                    best_sq += (ma - mb) * (ma - mb);
                    best_n += 1;
                    let a_won = a_seats.iter().any(|&s| o.winners.contains(&s));
                    let b_won = b_seats.iter().any(|&s| o.winners.contains(&s));
                    if a_won && !b_won {
                        win_a += 1.0;
                    } else if b_won && !a_won {
                        win_b += 1.0;
                    } else if a_won && b_won {
                        win_a += 0.5;
                        win_b += 0.5;
                    }
                }
                Err(e) => {
                    failures += 1;
                    eprintln!("[失败] seed {game_seed}: {e}");
                    if failures > 5 {
                        std::process::exit(1);
                    }
                }
            }
        }
        if pair_diff.len() == 2 {
            pairs += 1;
            let d = (pair_diff[0] + pair_diff[1]) / 2.0;
            diff_sum += d;
            diff_sq += d * d;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    // Both Side accumulators count the same games, not two disjoint sets of games.
    let games_total = side_a.games;
    println!("=== cabo-arena ===");
    println!(
        "A = {}（{} 座）  B = {}（{} 座）  玩家 {}  配对 {} 组（共 {} 局）  阈值 {}",
        args.a, args.a_seats, args.b, args.players - args.a_seats, args.players, pairs, games_total, args.target
    );
    println!(
        "耗时 {:.1}s（{:.1} 局/秒），平均每局 {:.1} 轮，失败 {}",
        elapsed,
        games_total as f64 / elapsed.max(1e-9),
        rounds as f64 / games_total.max(1) as f64,
        failures
    );
    let wins_total = side_a.wins + side_b.wins;
    println!(
        "A 获胜 {:.0} 局（{:.1}%），平均累计分 {:.2}",
        side_a.wins,
        100.0 * side_a.wins / wins_total.max(1.0),
        side_a.avg_total()
    );
    println!(
        "B 获胜 {:.0} 局（{:.1}%），平均累计分 {:.2}",
        side_b.wins,
        100.0 * side_b.wins / wins_total.max(1.0),
        side_b.avg_total()
    );
    if pairs > 0 {
        let mean = diff_sum / pairs as f64;
        let var = (diff_sq / pairs as f64 - mean * mean).max(0.0);
        let se = (var / pairs as f64).sqrt();
        let decisive = win_a + win_b;
        let share = 100.0 * win_a / decisive.max(1.0);
        println!(
            "平均累计分差（A − B）: {:+.2} ± {:.2}（1se）  胜率份额 A {:.1}% / B {:.1}%",
            mean,
            se,
            share,
            100.0 - share
        );
        if best_n > 0 {
            let bmean = best_sum / best_n as f64;
            let bvar = (best_sq / best_n as f64 - bmean * bmean).max(0.0);
            let bse = (bvar / best_n as f64).sqrt();
            println!(
                "★ 最强座位分差（每局 minA − minB 的均值）: {:+.2} ± {:.2}（1se，n={}）",
                bmean, bse, best_n
            );
            if bmean.abs() > 2.0 * bse.max(1e-9) {
                let better = if bmean < 0.0 { &args.a } else { &args.b };
                println!("→ 显著更强的是: {better}");
            } else {
                println!("→ 尚无显著差异");
            }
        }
    }
    let stats = cabo::ai::stats_string();
    if !stats.is_empty() {
        println!("{stats}");
    }
    if failures > 0 {
        std::process::exit(1);
    }
}
