//! 策略网络进 rollout 的**成本分解**基准：决定"用在哪里、用几张牌"。
//!
//! `cargo run --release --example policy_bench`

use cabo::ai::history::History;
use cabo::ai::policy::{candidate_features, state_features, Ctx, N_CAND, N_STATE};
use cabo::ai::search::candidates_scored;
use cabo::ai::tactics::{policy, Know, PolicyCfg};
use cabo::game::sim::make_ai_session;
use cabo::game::view::project;
use cabo::game::{Phase, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let settings = Settings { cabo_penalty: 10, target_score: 100, memory_mode: false };
    let mut s = make_ai_session(11, settings, &["tactician", "tactician", "tactician", "tactician"]);
    s.start_game().unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    // 走到中局，拿一个有代表性的局面
    for _ in 0..120 {
        let pid = match &s.phase {
            Phase::Peeking { done } => (0..4).find(|p| !done.contains_key(p)),
            Phase::Turn { current, .. } => Some(*current),
            Phase::RoundEnd => {
                s.next_round().unwrap();
                continue;
            }
            _ => None,
        };
        let Some(pid) = pid else { break };
        let view = project(&s, Some(pid), 0);
        let cmd = policy(&s, pid, &PolicyCfg::default(), &mut rng);
        if s.apply(pid, &cmd).is_err() {
            let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
        }
    }

    let pid = match &s.phase {
        Phase::Turn { current, .. } => *current,
        _ => 0,
    };
    println!("=== 成本分解：一次策略网决策的各环节 ===");
    let reps = 20_000usize;

    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..reps {
        let v = project(&s, Some(pid), 0);
        acc += v.opponents.len();
    }
    let t_proj = t0.elapsed().as_secs_f64() / reps as f64;
    println!("project()               {:>8.2} µs", t_proj * 1e6);

    let view = project(&s, Some(pid), 0);
    let t0 = Instant::now();
    for _ in 0..reps {
        let k = Know::from_session(&s, pid);
        acc += k.hands[0].len();
    }
    let t_know = t0.elapsed().as_secs_f64() / reps as f64;
    println!("Know::from_session()    {:>8.2} µs", t_know * 1e6);

    let info = History::new().observe(&view);
    let k = Know::from_view_with(&view, &info.known);
    let t0 = Instant::now();
    for _ in 0..reps {
        let c = candidates_scored(&view, &k, 10);
        acc += c.len();
    }
    let t_cand = t0.elapsed().as_secs_f64() / reps as f64;
    println!("candidates_scored()     {:>8.2} µs", t_cand * 1e6);

    let scored = candidates_scored(&view, &k, 10);
    let t0 = Instant::now();
    for _ in 0..reps {
        let f = state_features(&view, &k);
        let cf = candidate_features(&view, &k, &scored[0].1, 0, Some(0), scored[0].0 as f32);
        acc += f.len() + cf.len();
    }
    let t_feat = t0.elapsed().as_secs_f64() / reps as f64;
    println!("state+candidate 特征     {:>8.2} µs", t_feat * 1e6);

    let t0 = Instant::now();
    for _ in 0..reps {
        let ctx = Ctx::from_session(&s, pid);
        let k = Know::from_session(&s, pid);
        acc += ctx.round_no as usize + k.hands[0].len();
    }
    let t_sess = t0.elapsed().as_secs_f64() / reps as f64;
    println!("(Session 入口 ctx+Know)  {:>8.2} µs", t_sess * 1e6);

    std::hint::black_box(acc);
    println!();
    println!("状态特征 {} 维，候选特征 {} 维，平均候选数 {}", N_STATE, N_CAND, scored.len());
    println!("关键结论：project() 是最大单项开销 → rollout 内部应避免投影，改用 Session 入口。");
    println!(
        "每步成本（Session 入口 + 候选 + 特征 + 小网前向）≈ {:.1} µs + 网络",
        (t_sess + t_cand + t_feat) * 1e6
    );
}
