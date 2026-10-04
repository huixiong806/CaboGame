//! 校验"记牌"重建：账目是否准确、能恢复多少张弃牌堆底牌。
//! `cargo run --release --example memory_check -- 60 4`

use cabo::ai::history::History;
use cabo::ai::BotRegistry;
use cabo::game::sim::make_ai_session;
use cabo::game::view::project;
use cabo::game::{Phase, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(60);
    let players: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    let registry = BotRegistry::with_builtins();
    let bots: Vec<&str> = (0..players).map(|_| "simple").collect();
    let hist = History::new();
    let mut recalled = 0usize;
    let mut examples: Vec<(Vec<u8>, Vec<String>)> = Vec::new();
    let mut hidden = 0usize;
    let mut bad_account = 0usize;
    let mut false_positive = 0usize;
    let mut checks = 0usize;
    for i in 0..games {
        let seed = 1 + i as u64 * 7919;
        let mut s = make_ai_session(seed, Settings::default(), &bots);
        let _ = s.start_game();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEECE66D);
        for _ in 0..20000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                _ => None,
            };
            let Some(pid) = pid else { break };
            // 现实约束：Bot 只在自己行动时被调用（所以只在 0 号座位行动时观察）
            let mine = pid == 0;
            if mine {
                let view0 = project(&s, Some(0), 0);
                let info = hist.observe(&view0);
                if matches!(s.phase, Phase::Turn { .. }) && s.discard.len() > 1 {
                    checks += 1;
                    let below = &s.discard[..s.discard.len() - 1];
                    let claimed: usize = info.known.iter().map(|&c| c as usize).sum();
                    if claimed + info.unknown != below.len() {
                        bad_account += 1;
                    }
                    let mut truth = [0usize; 14];
                    for &c in below {
                        truth[s.cards[c as usize].card.rank as usize] += 1;
                    }
                    for r in 0..14 {
                        if info.known[r] as usize > truth[r] {
                            false_positive += 1;
                        }
                    }
                    recalled += claimed;
                    // ---- 漏牌取证：把"真实存在但没恢复出来"的牌连同近期日志打出来 ----
                    if examples.len() < 6 {
                        let mut missing: Vec<u8> = Vec::new();
                        for r in 0..14 {
                            let want = truth[r];
                            let got = info.known[r] as usize;
                            for _ in got..want {
                                missing.push(r as u8);
                            }
                        }
                        if !missing.is_empty() {
                            let log = &view0.log;
                            let tail: Vec<String> = log
                                .iter()
                                .rev()
                                .take(8)
                                .map(|e| e.text.clone())
                                .collect();
                            examples.push((missing.clone(), tail));
                        }
                    }
                    hidden += below.len();
                }
            }
            let view = project(&s, Some(pid), 0);
            let cmd = {
                let bot = registry.get(bots[pid]).unwrap();
                bot.decide(&view, &mut rng)
            };
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
    }
    println!("检查 {checks} 次（弃牌堆底牌 {hidden} 张）");
    println!("账目不符 {bad_account} 次；凭空捏造 {false_positive} 次");
    println!("\n=== 漏牌取证（最多 6 例，附近期日志）===");
    for (miss, tail) in &examples {
        println!("漏掉点数 {miss:?}；此前日志（新→旧）：");
        for t in tail {
            println!("    {t}");
        }
    }
    println!(
        "召回率：{recalled}/{hidden} = {:.1}% 的堆底牌点数被恢复",
        100.0 * recalled as f64 / hidden.max(1) as f64
    );
}
