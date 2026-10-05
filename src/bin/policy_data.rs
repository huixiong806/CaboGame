//! Search distillation on independent real-engine rounds, all seats use a legal teacher.
//! Every decision is one variable-size action group; splits must be by round seed.
use cabo::{
    ai::{
        planner::{PlannerBot, PlannerCfg},
        Bot,
    },
    game::{sim::make_ai_session, view::project, Phase, Settings},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

fn main() {
    let mut games = 600usize;
    let mut seed = 700000u64;
    let mut jobs = 4usize;
    let mut out = "data/action_policy/train-v1.jsonl".to_string();
    let mut max_actions = 4000usize;
    let mut config = Vec::new();
    let mut cfg = PlannerCfg {
        budget_us: 0,
        simulations: 128,
        confirmation_samples: 32,
        avoid_cycles: true,
        validate_calls: true,
        min_cabo_success: 0.,
        ..PlannerCfg::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        let v = args.next().expect("missing argument value");
        match k.as_str() {
            "--games" => games = v.parse().unwrap(),
            "--seed" => seed = v.parse().unwrap(),
            "--jobs" => jobs = v.parse().unwrap(),
            "--out" => out = v,
            "--max-actions" => max_actions = v.parse().unwrap(),
            "--cfg" => {
                for x in v.split(',') {
                    let (k, v) = x.split_once('=').unwrap();
                    assert!(cfg.set(k, v), "invalid config {x}");
                    config.push((k.to_string(), v.to_string()));
                }
            }
            _ => panic!("unknown argument {k}"),
        }
    }
    assert!(games > 0 && (1..=64).contains(&jobs) && max_actions > 0);
    assert_eq!(cfg.budget_us, 0, "teacher must use deterministic count");
    let bot = PlannerBot::try_new(cfg.clone()).expect("invalid teacher model");
    let path = std::path::Path::new(&out);
    assert!(!path.exists(), "completed split exists");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let pending = path.with_extension("jsonl.incomplete");
    let writer = Mutex::new(BufWriter::new(std::fs::File::create(&pending).unwrap()));
    let next = AtomicUsize::new(0);
    let groups = AtomicUsize::new(0);
    let rows = AtomicUsize::new(0);
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= games { break; }
                let game_seed = seed + index as u64;
                let mut setup = StdRng::seed_from_u64(game_seed ^ 0xF3B172A903E569CD);
                let n = setup.random_range(2..=4);
                let mut session = make_ai_session(game_seed, Settings::default(), &vec!["teacher"; n]);
                session.start_game().unwrap();
                if setup.random_bool(0.75) {
                    for p in &mut session.players {
                        p.total_score = setup.random_range(0..100);
                        p.score_reset_used = p.total_score >= 50 && setup.random_bool(0.3);
                    }
                }
                let mut rngs: Vec<_> = (0..n).map(|p| {
                    StdRng::seed_from_u64(game_seed ^ (p as u64 + 1).wrapping_mul(0xD1B54A32D192ED03))
                }).collect();
                let mut feature_rng = StdRng::seed_from_u64(game_seed ^ 0xAC7100FEE12345);
                let mut examples = Vec::new();
                for step in 0..max_actions {
                    let actor = match &session.phase {
                        Phase::Peeking { done } => (0..n).find(|p| !done.contains_key(p)).unwrap(),
                        Phase::Turn { current, .. } => *current,
                        Phase::RoundEnd | Phase::GameOver { .. } => break,
                        phase => panic!("unexpected phase {phase:?}"),
                    };
                    let view = project(&session, Some(actor), 0);
                    let command = bot.decide(&view, &mut rngs[actor]);
                    if let Some(candidates) = bot.policy_features(&view, &mut feature_rng) {
                        let label = candidates.iter().position(|(c, _)| *c == command).unwrap_or_else(|| {
                            panic!("teacher action absent seed={game_seed} step={step} {command:?}")
                        });
                        let x: Vec<_> = candidates.into_iter().map(|(_, x)| x).collect();
                        examples.push(serde_json::json!({"seed":game_seed,"step":step,"n":n,"x":x,"label":label}));
                    }
                    session.apply(actor, &command).unwrap_or_else(|e| {
                        panic!("illegal teacher seed={game_seed} step={step} {command:?}: {e}")
                    });
                }
                assert!(matches!(session.phase, Phase::RoundEnd | Phase::GameOver { .. }),
                    "round action limit seed={game_seed}; split remains incomplete");
                let mut w = writer.lock().unwrap();
                for e in &examples {
                    serde_json::to_writer(&mut *w, e).unwrap();
                    writeln!(w).unwrap();
                }
                groups.fetch_add(examples.len(), Ordering::Relaxed);
                rows.fetch_add(examples.iter().map(|e| e["x"].as_array().unwrap().len()).sum::<usize>(), Ordering::Relaxed);
                if (index + 1) % 100 == 0 {
                    eprintln!("round_index={}/{games} groups={} seconds={:.1}",
                        index + 1, groups.load(Ordering::Relaxed), start.elapsed().as_secs_f64());
                }
            });
        }
    });
    writer.lock().unwrap().flush().unwrap();
    drop(writer);
    let manifest = serde_json::json!({"rounds":games,"seed_start":seed,"seed_end":seed+games as u64-1,"groups":groups.load(Ordering::Relaxed),"action_rows":rows.load(Ordering::Relaxed),"features":64,"teacher":format!("{cfg:?}"),"overrides":config,"illegal_commands":0,"seconds":start.elapsed().as_secs_f64()});
    std::fs::write(
        path.with_extension("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    std::fs::rename(pending, path).unwrap();
    println!(
        "rounds={games} groups={} actions={} seconds={:.1} illegal_commands=0 out={out}",
        groups.load(Ordering::Relaxed),
        rows.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
}
