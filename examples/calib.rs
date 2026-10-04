//! 标定诊断：机器人对各家手牌总点数的估计是否准确？"严格最低"概率是否可信？
//!
//! `cargo run --release --example calib -- 300`

use cabo::ai::tactics::Know;
use cabo::ai::BotRegistry;
use cabo::game::sim::make_ai_session;
use cabo::game::view::project;
use cabo::game::{Phase, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(200);
    let players: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    let registry = BotRegistry::with_builtins();
    let bots: Vec<&str> = (0..players).map(|_| "simple").collect();
    let mut err_me = 0f64;
    let mut err_opp = 0f64;
    let mut n_me = 0f64;
    let mut n_opp = 0f64;
    let mut opp_slots = 0f64;
    // p_strict_lowest 的分箱标定
    let mut bins = [(0usize, 0usize); 10];
    for i in 0..games {
        let seed = 1 + i as u64 * 7919;
        let mut s = make_ai_session(seed, Settings::default(), &bots);
        let _ = s.start_game();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEECE66D);
        for _ in 0..4000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                _ => None,
            };
            let Some(pid) = pid else { break };
            if matches!(s.phase, Phase::Turn { .. }) {
                let view = project(&s, Some(pid), 0);
                let k = Know::from_view(&view);
                // 真实手牌点数（离线工具可以用真值）
                let truth: Vec<f64> = s
                    .players
                    .iter()
                    .map(|p| {
                        p.slots
                            .iter()
                            .map(|c| s.cards[*c as usize].card.rank as f64)
                            .sum::<f64>()
                    })
                    .collect();
                err_me += k.est(pid).0 - truth[pid];
                n_me += 1.0;
                for j in 0..players {
                    if j == pid {
                        continue;
                    }
                    err_opp += k.est(j).0 - truth[j];
                    opp_slots += k.hands[j].iter().filter(|c| c.is_none()).count() as f64;
                    n_opp += 1.0;
                }
                // "严格最低"标定：以当前真实手牌判断
                let p = k.p_strict_lowest();
                let bin = ((p * 10.0).floor() as usize).min(9);
                let actually =
                    (0..players).filter(|&j| j != pid).all(|j| truth[pid] < truth[j]);
                bins[bin].0 += 1;
                if actually {
                    bins[bin].1 += 1;
                }
            }
            let view = project(&s, Some(pid), 0);
            let cmd = {
                let bot = registry.get(bots[pid]).unwrap();
                bot.decide(&view, &mut rng)
            };
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&project(&s, Some(pid), 0)));
            }
        }
    }
    println!("[{} 人局]", players);
    println!("自己的手牌估计误差 : {:+.3} 点（n={n_me:.0}）", err_me / n_me);
    println!(
        "对手手牌估计误差   : {:+.3} 点（n={n_opp:.0}，平均 {:.2} 张未知牌）",
        err_opp / n_opp,
        opp_slots / n_opp
    );
    println!("--- p(严格最低) 标定（用当前真实手牌检验） ---");
    for (i, (cnt, hit)) in bins.iter().enumerate() {
        if *cnt == 0 {
            continue;
        }
        println!(
            "p∈[{:.1},{:.1})  样本 {:6}  实际命中率 {:.3}",
            i as f64 / 10.0,
            (i + 1) as f64 / 10.0,
            cnt,
            *hit as f64 / *cnt as f64
        );
    }
}
