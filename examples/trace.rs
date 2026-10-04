//! 单局追踪：打印一局的公开日志（可选实时），用于分析策略行为。
//! `cargo run --release --example trace -- search simple 42 99 4`
//! 参数：A B seed 显示轮数 [人数]
//! 环境变量 `TRACE_LIVE=1` 时实时打印（与 Bot 的调试输出交错）。

use cabo::ai::BotRegistry;
use cabo::game::sim::make_ai_session;
use cabo::game::{LogKind, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut args = std::env::args().skip(1);
    let a = args.next().unwrap_or_else(|| "simple".into());
    let b = args.next().unwrap_or_else(|| "simple".into());
    let seed: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(42);
    let quiet_rounds: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(99);
    let players: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    let live = std::env::var_os("TRACE_LIVE").is_some();
    let registry = BotRegistry::with_builtins();
    let bots: Vec<&str> =
        (0..players).map(|i| if i % 2 == 0 { a.as_str() } else { b.as_str() }).collect();
    if live {
        let mut s = make_ai_session(seed, Settings::default(), &bots);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEECE66D);
        s.start_game().unwrap();
        let mut cursor = 0usize;
        loop {
            let pid = match &s.phase {
                cabo::game::Phase::Peeking { done } => {
                    (0..s.players.len()).find(|p| !done.contains_key(p))
                }
                cabo::game::Phase::Turn { current, .. } => Some(*current),
                cabo::game::Phase::RoundEnd => {
                    s.next_round().unwrap();
                    continue;
                }
                cabo::game::Phase::GameOver { .. } => break,
                cabo::game::Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            let view = cabo::game::view::project(&s, Some(pid), 0);
            let cmd = registry.get(bots[pid]).unwrap().decide(&view, &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
            for e in &s.log[cursor..] {
                println!("[{:?}] {}", e.audience, e.text);
            }
            cursor = s.log.len();
            if s.log.len() > 4000 {
                s.log.drain(..2000);
                cursor -= 2000;
            }
        }
        println!("最终累计分: {:?}", s.players.iter().map(|p| p.total_score).collect::<Vec<_>>());
        return;
    }
    let mut s = make_ai_session(seed, Settings::default(), &bots);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEECE66D);
    let outcome = cabo::game::sim::run_game(&mut s, &registry, &mut rng, 200_000).unwrap();
    println!("座位: {bots:?}  结果: {outcome:?}");
    let mut round = 0usize;
    for e in &s.log {
        if let LogKind::Info = e.kind {
            if e.text.contains("轮开始") {
                round += 1;
            }
        }
        if round <= quiet_rounds {
            println!("[{:?}] {}", e.audience, e.text);
        }
    }
    println!("最终累计分: {:?}", s.players.iter().map(|p| p.total_score).collect::<Vec<_>>());
    println!("手牌数: {:?}", s.players.iter().map(|p| p.slots.len()).collect::<Vec<_>>());
}
