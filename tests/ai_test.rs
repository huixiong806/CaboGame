//! AI 层的集成测试：所有内置 Bot 在所有人数下都必须能打完合法对局，
//! 且搜索型 Bot 的单步决策必须守住时间预算。

use std::sync::Arc;

use cabo::ai::search::{SearchBot, SearchCfg};
use cabo::ai::BotRegistry;
use cabo::game::sim::{make_ai_session, run_game};
use cabo::game::view::project;
use cabo::game::Settings;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 每个注册的 Bot 都能在 2/3/4 人局里打完一整局，不出非法命令、不卡死。
///
/// 为了让 `cargo test` 保持快速，这里把搜索型 Bot 的预算压到 1ms
/// （只验证合法性/健壮性，不验证棋力；棋力由 cabo-arena 度量）。
#[test]
fn all_builtin_bots_finish_games() {
    let mut registry = BotRegistry::new();
    registry.register(Arc::new(cabo::ai::planner::ChallengerBot));
    registry.register(Arc::new(cabo::ai::planner::PlannerBot::new(cabo::ai::planner::PlannerCfg {
        budget_us: 1_000,
        simulations: 4,
        ..cabo::ai::planner::PlannerCfg::default()
    })));
    registry.register(Arc::new(SearchBot::new(SearchCfg { budget_us: 1_000, max_worlds: 4, min_worlds: 2, ..SearchCfg::default() })));
    for id in ["planner", "challenger", "search"] {
        for players in [2usize, 3, 4] {
            let bots: Vec<&str> = (0..players).map(|_| id).collect();
            for seed in [1u64, 7] {
                let mut s = make_ai_session(seed, Settings::default(), &bots);
                let mut rng = StdRng::seed_from_u64(seed);
                let o = run_game(&mut s, &registry, &mut rng, 300_000)
                    .unwrap_or_else(|e| panic!("{id} 在 {players} 人局里失败: {e}"));
                assert_eq!(o.totals.len(), players);
                assert!(!o.winners.is_empty());
            }
        }
    }
}

/// 搜索型 Bot 的单步耗时必须守住预算（留足余量，避免 CI 抖动误报）。
#[test]
fn search_bot_respects_time_budget() {
    let budget_us = 20_000u64;
    let mut reg = BotRegistry::new();
    reg.register(Arc::new(SearchBot::new(SearchCfg { budget_us, ..SearchCfg::default() })));
    reg.register(Arc::new(cabo::ai::planner::ChallengerBot));
    let mut worst = 0u128;
    let mut s = make_ai_session(99, Settings::default(), &["search", "challenger", "search"]);
    let mut rng = StdRng::seed_from_u64(99);
    s.start_game().unwrap();
    for _ in 0..600 {
        let pid = match &s.phase {
            cabo::game::Phase::Peeking { done } => (0..3).find(|p| !done.contains_key(p)),
            cabo::game::Phase::Turn { current, .. } => Some(*current),
            _ => None,
        };
        let Some(pid) = pid else { break };
        let view = project(&s, Some(pid), 0);
        let bot = reg.get(bots_id(&view, pid)).unwrap();
        let t0 = std::time::Instant::now();
        let cmd = bot.decide(&view, &mut rng);
        let dt = t0.elapsed().as_micros();
        if s.players[pid].bot_id == "search" {
            worst = worst.max(dt);
        }
        if s.apply(pid, &cmd).is_err() {
            let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
        }
        if matches!(s.phase, cabo::game::Phase::RoundEnd) {
            s.next_round().unwrap();
        }
    }
    // 允许显著的调度抖动余量：预算是 20ms，断言 10 倍以内。
    assert!(worst < budget_us as u128 * 10, "最慢单步 {worst}us 超出预算 10 倍");
}

fn bots_id(_view: &cabo::game::view::PlayerView, pid: usize) -> &'static str {
    if pid % 2 == 0 {
        "search"
    } else {
        "challenger"
    }
}
