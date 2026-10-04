//! 分阶段标定：把"本轮进度"分箱，分别拟合对手暗牌的 θ。
//! 关键结论：对手暗牌的均值随回合推进急剧下降（2 人局后期只有 1.3 点），
//! 固定的 θ 会系统性高估后期对手手牌 —— 而那正是宣告 Cabo 的时刻。
//!
//! `cargo run --release --example fit -- 200 4`

use cabo::ai::tactics::Know;
use cabo::ai::BotRegistry;
use cabo::game::sim::make_ai_session;
use cabo::game::view::project;
use cabo::game::{Phase, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Default, Clone)]
struct Stage {
    pool: [f64; 14],
    counts: [f64; 4],
    opp_sum: f64,
    opp_cnt: f64,
    me_sum: f64,
    me_cnt: f64,
    deck_sum: f64,
    deck_cnt: f64,
    disc_sum: f64,
    disc_cnt: f64,
    samples: f64,
}

fn predict(pool: &[f64; 14], counts: &[f64; 4], theta_opp: f64) -> [f64; 4] {
    let thetas = [0.0, theta_opp, 0.0, 0.08];
    let mut num = [0f64; 4];
    let mut den = [0f64; 4];
    for r in 0..14 {
        let mut w = [0f64; 4];
        let mut total = 0f64;
        for t in 0..4 {
            w[t] = counts[t] * (thetas[t] * r as f64).exp();
            total += w[t];
        }
        if total <= 0.0 {
            continue;
        }
        for t in 0..4 {
            let share = pool[r] * w[t] / total;
            num[t] += share * r as f64;
            den[t] += share;
        }
    }
    let mut out = [0f64; 4];
    for t in 0..4 {
        out[t] = if den[t] > 0.0 { num[t] / den[t] } else { 0.0 };
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(200);
    let players: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    let registry = BotRegistry::with_builtins();
    let bots: Vec<&str> = (0..players).map(|_| "simple").collect();
    let initial_deck = (52 - 4 * players - 1) as f64;
    const NSTAGE: usize = 6;
    let mut stages = vec![Stage::default(); NSTAGE];
    for i in 0..games {
        let seed = 1 + i as u64 * 7919;
        let mut s = make_ai_session(seed, Settings::default(), &bots);
        let _ = s.start_game();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEECE66D);
        for _ in 0..8000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                _ => None,
            };
            let Some(pid) = pid else { break };
            if matches!(s.phase, Phase::Turn { .. }) {
                let view = project(&s, Some(pid), 0);
                let k = Know::from_view(&view);
                let progress = (1.0 - k.deck_count as f64 / initial_deck).clamp(0.0, 0.9999);
                let si = ((progress * NSTAGE as f64) as usize).min(NSTAGE - 1);
                let st = &mut stages[si];
                for (r, c) in k.pool.iter().enumerate() {
                    st.pool[r] += *c as f64;
                }
                let mut c_me = 0f64;
                let mut c_opp = 0f64;
                for (j, p) in s.players.iter().enumerate() {
                    for &c in &p.slots {
                        let cs = &s.cards[c as usize];
                        if cs.is_known_to(pid) {
                            continue;
                        }
                        let rank = cs.card.rank as f64;
                        if j == pid {
                            c_me += 1.0;
                            st.me_sum += rank;
                            st.me_cnt += 1.0;
                        } else {
                            c_opp += 1.0;
                            st.opp_sum += rank;
                            st.opp_cnt += 1.0;
                        }
                    }
                }
                let mut c_deck = 0f64;
                for &c in &s.deck {
                    st.deck_sum += s.cards[c as usize].card.rank as f64;
                    st.deck_cnt += 1.0;
                    c_deck += 1.0;
                }
                let dn = s.discard.len().saturating_sub(1);
                let mut c_disc = 0f64;
                for &c in s.discard[..dn].iter() {
                    st.disc_sum += s.cards[c as usize].card.rank as f64;
                    st.disc_cnt += 1.0;
                    c_disc += 1.0;
                }
                st.counts[0] += c_me;
                st.counts[1] += c_opp;
                st.counts[2] += c_deck;
                st.counts[3] += c_disc;
                st.samples += 1.0;
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
    println!("[{} 人局] 初始摸牌堆 {:.0} 张", players, initial_deck);
    println!("阶段  进度      平均已抽    每人回合数  对手暗牌真值  最优θ   模型预测");
    for (si, st) in stages.iter().enumerate() {
        if st.samples <= 0.0 {
            continue;
        }
        let pool: [f64; 14] = std::array::from_fn(|i| st.pool[i] / st.samples);
        let counts: [f64; 4] = std::array::from_fn(|i| st.counts[i] / st.samples);
        let opp_true = st.opp_sum / st.opp_cnt.max(1.0);
        let draws = initial_deck - counts[2];
        let turns = draws / players as f64;
        let mut best = (f64::MAX, 0.0f64);
        let mut th = -0.8f64;
        while th <= 0.001 {
            let p = predict(&pool, &counts, th);
            let err = (p[1] - opp_true).powi(2);
            if err < best.0 {
                best = (err, th);
            }
            th += 0.005;
        }
        let p = predict(&pool, &counts, best.1);
        println!(
            "  {}   {:.2}-{:.2}   {:5.1}      {:5.1}      {:6.3}     {:+.3}   {:.3}",
            si + 1,
            si as f64 / NSTAGE as f64,
            (si + 1) as f64 / NSTAGE as f64,
            draws,
            turns,
            opp_true,
            best.1,
            p[1]
        );
    }
}
