//! Monte Carlo match-value targets from the real engine. No hidden cards are exported.
//! Every boundary in a match shares its seed, so splits must be made by seed, not row.
use cabo::{
    ai::{build_bot, planner::StyledChallengerBot, Bot},
    game::{sim::make_ai_session, view::project, Phase, Settings},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

fn main() {
    let mut games = 10000usize;
    let mut seed = 220000u64;
    let mut jobs = 4usize;
    let mut players = 0usize;
    let mut out = "research/artifacts/data/match_value/train.tsv".to_string();
    let mut bot = "fast".to_string();
    let mut roster = "mixed".to_string();
    let mut config = Vec::new();
    let mut max_actions = 100000usize;
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        let v = args.next().expect("missing argument value");
        match k.as_str() {
            "--games" => games = v.parse().unwrap(),
            "--seed" => seed = v.parse().unwrap(),
            "--jobs" => jobs = v.parse().unwrap(),
            "--players" => players = v.parse().unwrap(),
            "--out" => out = v,
            "--bot" => bot = v,
            "--roster" => roster = v,
            "--max-actions" => max_actions = v.parse().unwrap(),
            "--cfg" => {
                config = v
                    .split(',')
                    .map(|x| {
                        let (k, v) = x.split_once('=').unwrap();
                        (k.to_string(), v.to_string())
                    })
                    .collect()
            }
            _ => panic!("unknown argument {k}"),
        }
    }
    assert!(games > 0 && (1..=64).contains(&jobs) && (players == 0 || (2..=4).contains(&players)));
    assert!(matches!(roster.as_str(), "mixed" | "all"));
    let path = std::path::Path::new(&out);
    assert!(
        !path.exists(),
        "refusing to overwrite a completed split: {out}"
    );
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    // A failed or interrupted batch must never look like a complete training split.
    let pending = path.with_extension("tsv.incomplete");
    let writer = Mutex::new(BufWriter::new(std::fs::File::create(&pending).unwrap()));
    writeln!(
        writer.lock().unwrap(),
        "seed\tn\ttarget\tpenalty\ts0\ts1\ts2\ts3\tu0\tu1\tu2\tu3\tw0\tw1\tw2\tw3"
    )
    .unwrap();
    let next = AtomicUsize::new(0);
    let rows = AtomicUsize::new(0);
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= games {
                    break;
                }
                let game_seed = seed + i as u64;
                let mut setup = StdRng::seed_from_u64(game_seed ^ 0xF3B172A903E569CD);
                let n = if players == 0 {
                    setup.random_range(2..=4)
                } else {
                    players
                };
                let settings = Settings::default();
                let mut session = make_ai_session(game_seed, settings, &vec!["training"; n]);
                session.start_game().unwrap();
                // Mix normal openings and reachable public score/reset states for late-game coverage.
                if setup.random_bool(0.75) {
                    for p in &mut session.players {
                        p.total_score = setup.random_range(0..100);
                        p.score_reset_used = p.total_score >= 50 && setup.random_bool(0.3);
                    }
                }
                let bots: Vec<Arc<dyn Bot>> = (0..n)
                    .map(|p| {
                        if bot != "fast" && (roster == "all" || p % 2 == i % 2) {
                            build_bot(&bot, &config).expect("invalid bot/config")
                        } else {
                            Arc::new(StyledChallengerBot(setup.random_range(0..3))) as Arc<dyn Bot>
                        }
                    })
                    .collect();
                let mut rngs: Vec<_> = (0..n)
                    .map(|p| {
                        StdRng::seed_from_u64(
                            game_seed ^ (p as u64 + 1).wrapping_mul(0xD1B54A32D192ED03),
                        )
                    })
                    .collect();
                let mut boundaries = Vec::new();
                let mut record = |s: &cabo::game::Session| {
                    let mut line = format!("{game_seed}\t{n}\t100\t{}", s.settings.cabo_penalty);
                    for p in 0..4 {
                        line.push_str(&format!(
                            "\t{}",
                            s.players.get(p).map_or(0, |x| x.total_score)
                        ));
                    }
                    for p in 0..4 {
                        line.push_str(&format!(
                            "\t{}",
                            s.players.get(p).is_some_and(|x| x.score_reset_used) as u8
                        ));
                    }
                    boundaries.push(line);
                };
                record(&session);
                let mut finished = false;
                for _ in 0..max_actions {
                    let actor = match &session.phase {
                        Phase::Peeking { done } => (0..n).find(|p| !done.contains_key(p)).unwrap(),
                        Phase::Turn { current, .. } => *current,
                        Phase::RoundEnd => {
                            record(&session);
                            session.next_round().unwrap();
                            continue;
                        }
                        Phase::GameOver { winners } => {
                            let mut labels = String::new();
                            for p in 0..4 {
                                labels.push_str(&format!(
                                    "\t{}",
                                    if winners.contains(&p) {
                                        1.0 / winners.len() as f64
                                    } else {
                                        0.0
                                    }
                                ));
                            }
                            let mut w = writer.lock().unwrap();
                            for line in &boundaries {
                                writeln!(w, "{line}{labels}").unwrap();
                            }
                            rows.fetch_add(boundaries.len(), Ordering::Relaxed);
                            finished = true;
                            break;
                        }
                        Phase::Lobby => panic!("unexpected lobby"),
                    };
                    let cmd =
                        bots[actor].decide(&project(&session, Some(actor), 0), &mut rngs[actor]);
                    session
                        .apply(actor, &cmd)
                        .unwrap_or_else(|e| panic!("seed={game_seed} actor={actor} {cmd:?}: {e}"));
                }
                if !finished {
                    let history = &session.public_events;
                    let snapshot = format!("seed={game_seed} round={} deck={} phase={:?}\nview={:#?}\nlast_events={:#?}\n", session.round_no, session.deck.len(), session.phase, project(&session, Some(0), 0), &history[history.len().saturating_sub(20)..]);
                    std::fs::write(path.with_extension(format!("{game_seed}.failure.txt")), snapshot).unwrap();
                    panic!("match exceeded action limit: {game_seed}");
                }
                if (i + 1) % 500 == 0 {
                    eprintln!(
                        "completed index {} / {games}, rows={}, {:.1}s",
                        i + 1,
                        rows.load(Ordering::Relaxed),
                        start.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    writer.lock().unwrap().flush().unwrap();
    drop(writer);
    let manifest = serde_json::json!({
        "games": games, "seed_start": seed, "seed_end": seed + games as u64 - 1,
        "rows": rows.load(Ordering::Relaxed), "bot": bot, "roster": roster,
        "config": config, "players": players, "max_actions": max_actions,
        "illegal_commands": 0, "seconds": start.elapsed().as_secs_f64()
    });
    std::fs::write(
        path.with_extension("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    std::fs::rename(&pending, path).unwrap();
    println!(
        "games={games} rows={} seconds={:.1} illegal_commands=0 out={out}",
        rows.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
}
