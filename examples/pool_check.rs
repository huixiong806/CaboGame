//! 信念账目核对：`Know.pool` 的总量是否等于四类"看不见的位置"张数之和？
//!
//! 为什么查这个：E18 实测"给搜索真实的弃牌堆构成"值 **+1.54/决策**，
//! 但 E20 实测"记牌开/关"在棋力上**中性**。如果池子总量与位置张数对不上，
//! 位置化信念（`recount_positions`）的权重就会被错误缩放 ——
//! 信息确实进了 `known_discard`，却在分配环节被摊薄掉。
//!
//! `cargo run --release --example pool_check -- --games 20`

use cabo::ai::history::History;
use cabo::ai::tactics::Know;
use cabo::game::sim::make_ai_session;
use cabo::game::view::{project, Panel};
use cabo::game::{Phase, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut games = 20usize;
    let mut players = 4usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_default();
        match a.as_str() {
            "--games" => games = v().parse().unwrap_or(games),
            "--players" => players = v().parse().unwrap_or(players),
            other => {
                eprintln!("未知参数 {other}");
                std::process::exit(2);
            }
        }
    }
    let settings = Settings { cabo_penalty: 10, target_score: 100, memory_mode: false };
    let mut checked = 0usize;
    let mut mismatch = 0usize;
    let mut worst = 0i64;
    let mut sum_gap = 0i64;
    let mut with_mem = 0usize;
    for g in 0..games {
        let seed = 1 + g as u64 * 7919;
        let ids: Vec<&str> = (0..players).map(|_| "tactician").collect();
        let mut s = make_ai_session(seed, settings.clone(), &ids);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEECE66D);
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
                // 池子总量
                let pool_total: i64 = k.pool.iter().map(|&c| c as i64).sum();
                // 四类"看不见的位置"张数
                let me_unknown = k.hands[k.me].iter().filter(|c| c.is_none()).count() as i64;
                let opp_unknown: i64 = k
                    .hands
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != k.me)
                    .map(|(_, h)| h.iter().filter(|c| c.is_none()).count() as i64)
                    .sum();
                let deck = k.deck_count as i64;
                let known_disc: i64 = k.known_discard.iter().map(|&c| c as i64).sum();
                let disc_unknown = (k.discard_count as i64 - 1 - known_disc).max(0);
                let rhs = me_unknown + opp_unknown + deck + disc_unknown;
                let gap = pool_total - rhs;
                checked += 1;
                if gap != 0 {
                    mismatch += 1;
                    sum_gap += gap;
                    worst = worst.max(gap.abs());
                }
                if known_disc > 0 {
                    with_mem += 1;
                }
            }
            let cmd = cabo::ai::tactics::policy(&s, pid, &cabo::ai::tactics::PolicyCfg::default(), &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
    }
    println!("=== 信念账目核对（{checked} 个决策点）===");
    println!(
        "池子总量 ≠ 四类位置张数之和 的情况：**{mismatch}** 次（{:.2}%），平均偏差 {:+.2} 张，最大绝对偏差 {} 张",
        100.0 * mismatch as f64 / checked.max(1) as f64,
        sum_gap as f64 / mismatch.max(1) as f64,
        worst
    );
    println!(
        "其中 {} 个决策点带非零记牌信息（{:.1}%）",
        with_mem,
        100.0 * with_mem as f64 / checked.max(1) as f64
    );
    if mismatch == 0 {
        println!("→ 账目闭合：池子总量与位置张数一致，问题不在这里。");
    } else {
        println!("→ **账目不闭合**：位置化信念的权重被错误缩放，信息会在分配环节被摊薄。");
    }
}
