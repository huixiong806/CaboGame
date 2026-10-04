//! 决策稳定点扫描：搜索的**选择**随世界数怎么收敛？
//!
//! 上一轮确认"瓶颈是候选估值的蒙特卡洛精度、可以用时间买"（DL_ROUTE.md D6），
//! 那自然要问：买到哪里就够了？本工具在真实局面的决策点上，用固定世界数
//! K ∈ {…} 各跑一次搜索，以最大 K 的选择为参照，量"选择一致率"随 K 的变化。
//!
//! 一致率早就到 100% ⇒ 再多的世界也改不了决策，预算该停在那里。
//!
//! `cargo run --release --example world_sweep -- --games 3 --players 4`

use std::sync::Arc;

use cabo::ai::history::History;
use cabo::ai::search::{SearchBot, SearchCfg};
use cabo::ai::tactics::Know;
use cabo::ai::{Bot, BotRegistry};
use cabo::game::sim::make_ai_session;
use cabo::game::view::{project, Panel};
use cabo::game::{Phase, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut games = 3usize;
    let mut players = 4usize;
    let mut ks: Vec<u32> = vec![64, 256, 1024, 4096];
    let mut seed = 1u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_default();
        match a.as_str() {
            "--games" => games = v().parse().unwrap_or(games),
            "--players" => players = v().parse().unwrap_or(players),
            "--ks" => {
                ks = v()
                    .split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect()
            }
            "--seed" => seed = v().parse().unwrap_or(seed),
            other => {
                eprintln!("未知参数 {other}");
                std::process::exit(2);
            }
        }
    }
    ks.sort_unstable();
    let kmax = *ks.last().unwrap();

    // 每个 K 一个搜索实例：固定世界数（min=max=K，时间不设限）
    let bots: Vec<(u32, SearchBot)> = ks
        .iter()
        .map(|&k| {
            (
                k,
                SearchBot::new(SearchCfg {
                    budget_us: u64::MAX / 2,
                    max_worlds: k,
                    min_worlds: k,
                    ..SearchCfg::default()
                }),
            )
        })
        .collect();

    // 用一个便宜的对局驱动产生真实局面
    let mut driver = BotRegistry::new();
    driver.register(Arc::new(SearchBot::new(SearchCfg {
        budget_us: 3_000,
        ..SearchCfg::default()
    })));

    let settings = Settings { cabo_penalty: 10, target_score: 100, memory_mode: false };
    let mut agree = vec![0usize; ks.len()];
    let mut total = 0usize;
    let mut changed_examples = 0usize;
    let t0 = std::time::Instant::now();

    for g in 0..games {
        let game_seed = seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let ids: Vec<&str> = (0..players).map(|_| "search").collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &ids);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
        let history = History::new();
        for _ in 0..200_000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                Phase::RoundEnd => {
                    s.next_round().unwrap();
                    continue;
                }
                Phase::GameOver { .. } | Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            let view = project(&s, Some(pid), 0);
            if matches!(view.panel, Panel::Idle { .. }) {
                let info = history.observe(&view);
                let k = Know::from_view_with(&view, &info.known);
                let mut choices: Vec<(u32, String)> = Vec::new();
                for (kk, bot) in &bots {
                    let mut r = StdRng::seed_from_u64(0xC0FFEE ^ *kk as u64);
                    let cmd = bot.decide_with_know(&view, &k, &info.known, &mut r);
                    choices.push((*kk, format!("{cmd:?}")));
                }
                let reference = choices.last().unwrap().1.clone();
                for (i, (kk, c)) in choices.iter().enumerate() {
                    if *c == reference {
                        agree[i] += 1;
                    } else if *kk == kmax / 4 && changed_examples < 3 {
                        changed_examples += 1;
                        eprintln!(
                            "[分歧示例] P{pid} r{} K={kk} 选 {c}，而 K={kmax} 选 {reference}",
                            view.round_no
                        );
                    }
                }
                total += 1;
            }
            let cmd = driver.get("search").unwrap().decide(&view, &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
        eprintln!("[world_sweep] 第 {} 局完成，已采样 {} 个决策点", g + 1, total);
    }

    println!("=== 决策稳定点扫描（{} 个真实决策点，参照 K={kmax}）===", total);
    println!("{:>8}  {:>10}  {:>12}", "世界数 K", "选择一致率", "不一致数");
    for (i, &kk) in ks.iter().enumerate() {
        let a = agree[i] as f64 / total.max(1) as f64;
        println!("{kk:>8}  {:>9.2}%  {:>12}", 100.0 * a, total - agree[i]);
    }
    let knee = ks
        .iter()
        .enumerate()
        .find(|(i, _)| agree[*i] as f64 / total.max(1) as f64 >= 0.99)
        .map(|(_, k)| *k);
    match knee {
        Some(k) => println!(
            "\n→ 决策在 K≈{k} 就已稳定（与参照一致率 ≥99%）：预算买到这个量级即可，再多改不了选择。\n\
             耗时 {:.0}s",
            t0.elapsed().as_secs_f64()
        ),
        None => println!(
            "\n→ 到 K={kmax} 仍未稳定（一致率 <99%）：说明还可能从更大预算里获益。耗时 {:.0}s",
            t0.elapsed().as_secs_f64()
        ),
    }
}
