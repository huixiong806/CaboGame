//! 对局统计：跑若干局并汇总每轮结算，用于诊断策略行为。
//! `cargo run --release --example analyze -- search simple 2 60`

use std::sync::Arc;

use cabo::ai::BotRegistry;
use cabo::game::sim::make_ai_session;
use cabo::game::Settings;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut args = std::env::args().skip(1);
    let a = args.next().unwrap_or_else(|| "simple".into());
    let b = args.next().unwrap_or_else(|| "simple".into());
    let players: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    let games: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(200);
    let a_cfg = args.next().unwrap_or_default();
    let b_cfg = args.next().unwrap_or_default();
    let parse = |s: &str| -> Vec<(String, String)> {
        s.split(',')
            .filter(|p| !p.trim().is_empty())
            .filter_map(|p| {
                let (k, v) = p.split_once('=')?;
                Some((k.trim().to_string(), v.trim().to_string()))
            })
            .collect()
    };
    let mut registry = BotRegistry::new();
    let bot_a = cabo::ai::build_bot(&a, &parse(&a_cfg)).expect("bot A");
    let bot_b = cabo::ai::build_bot(&b, &parse(&b_cfg)).expect("bot B");
    let (ka, kb): (&'static str, &'static str) = (
        Box::leak(format!("A:{a}").into_boxed_str()),
        Box::leak(format!("B:{b}").into_boxed_str()),
    );
    registry.register(Arc::new(cabo::ai::AliasBot { inner: bot_a, id: ka, name: ka }));
    registry.register(Arc::new(cabo::ai::AliasBot { inner: bot_b, id: kb, name: kb }));
    let bots: Vec<&str> =
        (0..players).map(|i| if i % 2 == 0 { ka } else { kb }).collect();
    let names: Vec<String> = (1..=players).map(|i| format!("P{i}")).collect();
    let mut per_bot_score = vec![0f64; players];
    let mut per_bot_rounds = vec![0f64; players];
    let mut cabo_calls = vec![0usize; players];
    let mut cabo_success = vec![0usize; players];
    let mut rounds = 0usize;
    let mut hand_sizes = vec![0f64; players];
    let mut game_rounds = 0f64;
    for i in 0..games {
        let seed = 1 + i as u64 * 7919;
        let mut s = make_ai_session(seed, Settings::default(), &bots);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEECE66D);
        let o = cabo::game::sim::run_game(&mut s, &registry, &mut rng, 200_000).unwrap();
        game_rounds += o.rounds as f64;
        let mut cabo_by: Option<usize> = None;
        for e in &s.log {
            let t = &e.text;
            if t.contains("轮开始") {
                cabo_by = None;
            } else if t.contains("宣告 Cabo") {
                rounds += 1;
                for (idx, name) in names.iter().enumerate() {
                    if t.contains(name.as_str()) {
                        cabo_by = Some(idx);
                        cabo_calls[idx] += 1;
                        break;
                    }
                }
            } else if t.contains("本轮 +") {
                let idx = names.iter().position(|n| t.starts_with(n.as_str()));
                let Some(idx) = idx else { continue };
                let score: f64 = t
                    .rsplit("本轮 +")
                    .next()
                    .map(|v| {
                        let d: String =
                            v.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
                        d.parse().unwrap_or(0.0)
                    })
                    .unwrap_or(0.0);
                per_bot_score[idx] += score;
                per_bot_rounds[idx] += 1.0;
                if let Some(cnt) = t.split('（').nth(1).and_then(|x| x.split(' ').next()) {
                    if let Ok(c) = cnt.parse::<f64>() {
                        hand_sizes[idx] += c;
                    }
                }
                if cabo_by == Some(idx) && t.contains("Cabo 成功") {
                    cabo_success[idx] += 1;
                }
            }
        }
    }
    println!("=== analyze: {bots:?}  共 {games} 局, 平均 {:.2} 轮/局 ===", game_rounds / games as f64);
    for i in 0..players {
        let n = per_bot_rounds[i].max(1.0);
        println!(
            "P{} {:<10} 轮均分 {:5.2}  平均手牌数 {:.2}  Cabo 宣告 {} 次，成功 {} 次（{:.0}%）",
            i + 1,
            bots[i],
            per_bot_score[i] / n,
            hand_sizes[i] / n,
            cabo_calls[i],
            cabo_success[i],
            100.0 * cabo_success[i] as f64 / cabo_calls[i].max(1) as f64
        );
    }
    println!("Cabo 宣告总数 {rounds}");
}
