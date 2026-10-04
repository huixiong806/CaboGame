//! rollout 策略的**估值准确度**评估（快速迭代指标）。
//!
//! 擂台跑一局要几分钟，而 ExIt 每轮都要判断"新的 rollout 策略是不是更好"。
//! 本工具直接从根上量：对同一个动作、同一批确定性世界，用两种 rollout 策略
//! 各估一次价值，再与**真实发生的结果**（本轮结束时与最强劲对手的分差）对比。
//!
//! 指标：估计值与真实值的
//! - 相关（越高越好，说明排序能力）
//! - MAE（越低越好）
//! - 符号命中率（估的"好/坏"是否与真实一致）
//!
//! `cargo run --release --example rollout_eval -- --games 20 --worlds 24`

use std::sync::Arc;

use cabo::ai::belief::reconstruct;
use cabo::ai::history::History;
use cabo::ai::policy::choose_from_session;
use cabo::ai::search::{SearchBot, SearchCfg};
use cabo::ai::tactics::{policy, raw_fallback, Know, PolicyCfg};
use cabo::ai::{Bot, BotRegistry};
use cabo::game::sim::make_ai_session;
use cabo::game::view::{project, Panel};
use cabo::game::{Command, Phase, Session, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 打到本轮结束，返回"与最强劲对手的分差"。
fn rollout_lead(
    s: &mut Session,
    me: usize,
    cfg: &PolicyCfg,
    use_net_for_me: bool,
    rng: &mut dyn rand::RngCore,
) -> f64 {
    let n = s.players.len();
    for _ in 0..4000 {
        let pid = match &s.phase {
            Phase::Peeking { done } => (0..n).find(|p| !done.contains_key(p)),
            Phase::Turn { current, .. } => Some(*current),
            _ => break,
        };
        let Some(pid) = pid else { break };
        let cmd = if use_net_for_me && pid == me {
            choose_from_session(s, pid, rng, cfg).unwrap_or_else(|| policy(s, pid, cfg, rng))
        } else {
            policy(s, pid, cfg, rng)
        };
        if s.apply(pid, &cmd).is_err() {
            let fb = raw_fallback(s, pid);
            if s.apply(pid, &fb).is_err() {
                break;
            }
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

fn main() {
    let mut games = 20usize;
    let mut players = 4usize;
    let mut worlds = 24u32;
    let mut seed = 1u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_default();
        match a.as_str() {
            "--games" => games = v().parse().unwrap_or(games),
            "--players" => players = v().parse().unwrap_or(players),
            "--worlds" => worlds = v().parse().unwrap_or(worlds),
            "--seed" => seed = v().parse().unwrap_or(seed),
            other => {
                eprintln!("未知参数 {other}");
                std::process::exit(2);
            }
        }
    }
    assert!(cabo::ai::policy::available(), "策略网络权重缺失");

    let cfg = PolicyCfg::default();
    let settings = Settings { cabo_penalty: 10, target_score: 100, memory_mode: false };
    let mut registry = BotRegistry::new();
    // 驱动用**固定世界数**而不是时间预算：时间预算会让结果随机器负载变化，跨次不可比。
    registry.register(Arc::new(SearchBot::new(SearchCfg {
        budget_us: u64::MAX / 2,
        max_worlds: 6,
        min_worlds: 6,
        ..SearchCfg::default()
    })));

    // (估(启发式rollout), 估(学习rollout), 真实结果) 三元组
    let mut pairs: Vec<(f64, f64, f64)> = Vec::new();
    let t0 = std::time::Instant::now();
    for g in 0..games {
        let game_seed = seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let ids: Vec<&str> = (0..players).map(|_| "search").collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &ids);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
        let history = History::new();
        // 待核对的动作：本轮里记录"动作生效后的估值 vs 真实结算"
        let mut pending: Vec<(usize, Command, f64, f64)> = Vec::new();
        for _ in 0..200_000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                Phase::RoundEnd => {
                    // 结算：把每个待核对的估值与真实结果配对
                    for (seat, _cmd, vh, vn) in pending.drain(..) {
                        let my = s.players[seat].total_score as f64;
                        let min_other = s
                            .players
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != seat)
                            .map(|(_, p)| p.total_score as f64)
                            .fold(f64::MAX, f64::min);
                        pairs.push((vh, vn, min_other - my));
                    }
                    s.next_round().unwrap();
                    continue;
                }
                Phase::GameOver { .. } | Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            let view = project(&s, Some(pid), 0);
            let info = history.observe(&view);
            // 与策略网络在 rollout 里的实际使用分布一致：采所有面板
            if matches!(
                view.panel,
                Panel::Idle { .. }
                    | Panel::Drew { .. }
                    | Panel::SwapSelecting { .. }
                    | Panel::ConfirmCabo
            ) {
                let k = Know::from_view_with(&view, &info.known);
                let cmd = registry.get("search").unwrap().decide(&view, &mut rng);
                // 同一批世界、两种 rollout 策略各估一次
                let (mut sh, mut sn, mut cnt) = (0f64, 0f64, 0f64);
                for w in 0..worlds {
                    let mut wr = StdRng::seed_from_u64(game_seed ^ (w as u64).wrapping_mul(0x9E3779B97F4A7C15));
                    let Some(world) = reconstruct(&view, &mut wr, &info.known) else { continue };
                    let ws = 0x1234_5678u64 ^ ((w as u64) << 32) ^ pid as u64;
                    let mut a = world.clone();
                    let mut b = world.clone();
                    if a.apply(pid, &cmd).is_err() || b.apply(pid, &cmd).is_err() {
                        continue;
                    }
                    let mut r1 = StdRng::seed_from_u64(ws);
                    let mut r2 = StdRng::seed_from_u64(ws);
                    sh += rollout_lead(&mut a, pid, &cfg, false, &mut r1);
                    sn += rollout_lead(&mut b, pid, &cfg, true, &mut r2);
                    cnt += 1.0;
                }
                if cnt > 0.0 {
                    pending.push((pid, cmd.clone(), sh / cnt, sn / cnt));
                }
                if s.apply(pid, &cmd).is_err() {
                    let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
                }
            } else {
                let cmd = registry.get("search").unwrap().decide(&view, &mut rng);
                if s.apply(pid, &cmd).is_err() {
                    let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
                }
            }
        }
    }

    let n = pairs.len();
    let corr = |idx: usize| -> f64 {
        let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = pairs.iter().map(|p| p.2).collect();
        let _ = idx;
        let mx = xs.iter().sum::<f64>() / n as f64;
        let my = ys.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut dx = 0.0;
        let mut dy = 0.0;
        for i in 0..n {
            num += (xs[i] - mx) * (ys[i] - my);
            dx += (xs[i] - mx).powi(2);
            dy += (ys[i] - my).powi(2);
        }
        num / (dx.sqrt() * dy.sqrt()).max(1e-9)
    };
    let mae = |idx: usize| -> f64 {
        pairs
            .iter()
            .map(|p| ((if idx == 0 { p.0 } else { p.1 }) - p.2).abs())
            .sum::<f64>()
            / n.max(1) as f64
    };
    let sign_hit = |idx: usize| -> f64 {
        let ok = pairs
            .iter()
            .filter(|p| {
                let v = if idx == 0 { p.0 } else { p.1 };
                (v > 0.0) == (p.2 > 0.0)
            })
            .count();
        ok as f64 / n.max(1) as f64
    };

    println!("=== rollout 策略估值准确度（{} 个样本，{} 世界/样本）===", n, worlds);
    println!("{:<26} {:>10} {:>10} {:>12}", "rollout 策略", "相关", "MAE", "符号命中");
    println!(
        "{:<26} {:>10.4} {:>10.3} {:>11.1}%",
        "启发式（所有座位）",
        corr(0),
        mae(0),
        100.0 * sign_hit(0)
    );
    println!(
        "{:<26} {:>10.4} {:>10.3} {:>11.1}%",
        "我方座位用学习策略",
        corr(1),
        mae(1),
        100.0 * sign_hit(1)
    );
    println!("耗时 {:.0}s", t0.elapsed().as_secs_f64());
}
