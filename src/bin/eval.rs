//! One candidate against independent opponents, every seat, paired by seed block.
//! Fail on an illegal command instead of concealing it with a fallback.
use cabo::ai::{build_bot, Bot};
use cabo::game::{sim::make_ai_session, view::project, Phase, Settings};
use rand::{rngs::StdRng, SeedableRng};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Args {
    candidate: String,
    compare: Option<String>,
    opponents: Vec<String>,
    cfg: Vec<(String, String)>,
    compare_cfg: Vec<(String, String)>,
    opponent_cfg: Vec<(String, String)>,
    players: usize,
    blocks: usize,
    seed: u64,
    jobs: usize,
    target: u32,
    out: Option<String>,
}

fn cfg(s: &str) -> Vec<(String, String)> {
    s.split(',')
        .filter(|s| !s.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').expect("configuration must be key=value");
            (k.to_string(), v.to_string())
        })
        .collect()
}

fn args() -> Args {
    let mut a = Args {
        candidate: "v4".into(),
        compare: None,
        opponents: vec!["challenger".into()],
        cfg: vec![],
        compare_cfg: vec![],
        opponent_cfg: vec![],
        players: 4,
        blocks: 20,
        seed: 20261004,
        jobs: 4,
        target: 100,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(key) = it.next() {
        if key == "--help" {
            println!("cabo-eval --candidate v4 [--compare search] --opponents challenger,simple --blocks 20 --players 4 --jobs 4 --seed 20261004 --cfg budget_us=0,simulations=256 [--compare-cfg ...] [--opponent-cfg ...]");
            std::process::exit(0);
        }
        let v = it
            .next()
            .unwrap_or_else(|| panic!("missing value for {key}"));
        match key.as_str() {
            "--candidate" => a.candidate = v,
            "--compare" => a.compare = Some(v),
            "--opponents" => a.opponents = v.split(',').map(String::from).collect(),
            "--cfg" => a.cfg = cfg(&v),
            "--compare-cfg" => a.compare_cfg = cfg(&v),
            "--opponent-cfg" => a.opponent_cfg = cfg(&v),
            "--players" => a.players = v.parse().unwrap(),
            "--blocks" => a.blocks = v.parse().unwrap(),
            "--seed" => a.seed = v.parse().unwrap(),
            "--jobs" => a.jobs = v.parse().unwrap(),
            "--target" => a.target = v.parse().unwrap(),
            "--out" => a.out = Some(v),
            _ => panic!("unknown argument: {key}"),
        }
    }
    assert!(
        (2..=4).contains(&a.players)
            && a.blocks >= 2
            && (1..=64).contains(&a.jobs)
            && a.target > 0
            && !a.opponents.is_empty()
            && a.opponents.iter().all(|s| !s.is_empty())
    );
    a
}

#[derive(Clone, Copy, Debug, Default)]
struct ResultRow {
    gap: f64,
    win: f64,
    actions: u64,
    decisions: u64,
    decision_us: u64,
    max_us: u64,
    calls: u64,
    successes: u64,
}
impl ResultRow {
    fn add(&mut self, other: Self) {
        self.gap += other.gap;
        self.win += other.win;
        self.actions += other.actions;
        self.decisions += other.decisions;
        self.decision_us += other.decision_us;
        self.max_us = self.max_us.max(other.max_us);
        self.calls += other.calls;
        self.successes += other.successes;
    }
}

fn play(
    a: &Args,
    id: &str,
    config: &[(String, String)],
    seed: u64,
    seat: usize,
) -> Result<ResultRow, String> {
    let player =
        build_bot(id, config).ok_or_else(|| format!("unknown bot or config: {id} {config:?}"))?;
    let mut bots: Vec<Arc<dyn Bot>> = Vec::new();
    let mut opponent_index = 0usize;
    for p in 0..a.players {
        bots.push(if p == seat {
            player.clone()
        } else {
            // Fill every noncandidate seat from the requested roster. Rotating the candidate
            // must not accidentally remove the strongest opponent from a mixed roster.
            let opponent = &a.opponents[opponent_index % a.opponents.len()];
            opponent_index += 1;
            build_bot(opponent, &a.opponent_cfg)
                .ok_or_else(|| format!("unknown opponent or config {opponent}"))?
        });
    }
    let mut s = make_ai_session(
        seed,
        Settings {
            target_score: a.target,
            ..Settings::default()
        },
        &vec!["eval"; a.players],
    );
    s.start_game().map_err(|e| e.to_string())?;
    // Separate decision random streams avoid coupling opponents to candidate RNG consumption.
    let mut rngs: Vec<StdRng> = (0..a.players)
        .map(|p| StdRng::seed_from_u64(seed ^ (p as u64 + 1).wrapping_mul(0xD1B54A32D192ED03)))
        .collect();
    let mut row = ResultRow::default();
    for _ in 0..100_000 {
        let actor = match &s.phase {
            Phase::Peeking { done } => (0..a.players).find(|p| !done.contains_key(p)),
            Phase::Turn { current, .. } => Some(*current),
            Phase::RoundEnd => {
                if s.cabo_caller == Some(seat) && s.players[seat].round_score == Some(0) {
                    row.successes += 1;
                }
                s.next_round().map_err(|e| e.to_string())?;
                continue;
            }
            Phase::GameOver { winners } => {
                if s.cabo_caller == Some(seat) && s.players[seat].round_score == Some(0) {
                    row.successes += 1;
                }
                row.gap = s.players[seat].total_score as f64
                    - (0..a.players)
                        .filter(|&p| p != seat)
                        .map(|p| s.players[p].total_score)
                        .min()
                        .unwrap() as f64;
                row.win = if winners.contains(&seat) {
                    1.0 / winners.len() as f64
                } else {
                    0.0
                };
                return Ok(row);
            }
            Phase::Lobby => return Err("unexpected lobby".into()),
        }
        .ok_or("no actor")?;
        let view = project(&s, Some(actor), 0);
        let start = Instant::now();
        let cmd = bots[actor].decide(&view, &mut rngs[actor]);
        let us = start.elapsed().as_micros() as u64;
        if actor == seat {
            row.decisions += 1;
            row.decision_us += us;
            row.max_us = row.max_us.max(us);
            if matches!(cmd, cabo::game::Command::CallCabo) {
                row.calls += 1;
            }
        }
        s.apply(actor, &cmd)
            .map_err(|e| format!("seed={seed} seat={seat} actor={actor} {cmd:?}: {e}"))?;
        row.actions += 1;
    }
    Err(format!("seed={seed} exceeded action limit"))
}

fn summarize(label: &str, rows: &[ResultRow], seats: usize) {
    let n = rows.len() as f64;
    let mean = rows.iter().map(|r| r.gap / seats as f64).sum::<f64>() / n;
    let se = (rows
        .iter()
        .map(|r| (r.gap / seats as f64 - mean).powi(2))
        .sum::<f64>()
        / (n * (n - 1.0)))
        .sqrt();
    let win = rows.iter().map(|r| r.win / seats as f64).sum::<f64>() / n;
    let win_se = (rows
        .iter()
        .map(|r| (r.win / seats as f64 - win).powi(2))
        .sum::<f64>()
        / (n * (n - 1.0)))
        .sqrt();
    let t = critical_t(rows.len());
    // A conservative block-size Wilson envelope prevents reporting certainty after zero wins.
    // These are clustered, fractional observations, so this interval is descriptive, not exact.
    let z = 1.96;
    let denom = 1.0 + z * z / n;
    let center = (win + z * z / (2.0 * n)) / denom;
    let half = z * (win * (1.0 - win) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    let lo = (win - t * win_se).min(center - half).max(0.0);
    let hi = (win + t * win_se).max(center + half).min(1.0);
    let mut totals = ResultRow::default();
    for &r in rows {
        totals.add(r);
    }
    println!("{label}: games={} win_share={:.1}% (descriptive 95% block interval {:.1}..{:.1}%); score_gap={:+.2} +/- {:.2} (95% block t CI); decision_mean={:.2}ms max={:.2}ms Cabo={}/{}",
        rows.len()*seats,100.0*win,100.0*lo,100.0*hi,mean,t*se,totals.decision_us as f64/totals.decisions.max(1) as f64/1000.0,totals.max_us as f64/1000.0,totals.successes,totals.calls);
}

fn critical_t(blocks: usize) -> f64 {
    const T: [f64; 30] = [
        0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060,
        2.056, 2.052, 2.048, 2.045,
    ];
    T.get(blocks - 1)
        .copied()
        .unwrap_or(if blocks < 61 { 2.0 } else { 1.96 })
}

fn summarize_difference(rows: &[ResultRow], seats: usize) {
    let n = rows.len() as f64;
    let mean = rows.iter().map(|r| r.gap / seats as f64).sum::<f64>() / n;
    let se = (rows
        .iter()
        .map(|r| (r.gap / seats as f64 - mean).powi(2))
        .sum::<f64>()
        / (n * (n - 1.0)))
        .sqrt();
    let win = rows.iter().map(|r| r.win / seats as f64).sum::<f64>() / n;
    let win_se = (rows
        .iter()
        .map(|r| (r.win / seats as f64 - win).powi(2))
        .sum::<f64>()
        / (n * (n - 1.0)))
        .sqrt();
    let t = critical_t(rows.len());
    println!("paired candidate-minus-compare: score_gap_delta={:+.2} +/- {:.2} (95% block t CI); win_share_delta={:+.1} +/- {:.1}pp; independent_blocks={}",mean,t*se,100.0*win,100.0*t*win_se,rows.len());
}

fn main() {
    let a = Arc::new(args());
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    let failures = Mutex::new(Vec::new());
    let start = Instant::now();
    println!("cabo-eval: {:?}", a);
    std::thread::scope(|scope| {
        for _ in 0..a.jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= a.blocks {
                    break;
                }
                let seed = a
                    .seed
                    .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let mut row = ResultRow::default();
                let mut other = ResultRow::default();
                let mut error = None;
                for seat in 0..a.players {
                    match play(&a, &a.candidate, &a.cfg, seed, seat) {
                        Ok(r) => row.add(r),
                        Err(e) => {
                            error = Some(e);
                            break;
                        }
                    }
                    if let Some(id) = &a.compare {
                        match play(&a, id, &a.compare_cfg, seed, seat) {
                            Ok(r) => other.add(r),
                            Err(e) => {
                                error = Some(e);
                                break;
                            }
                        }
                    }
                }
                if let Some(e) = error {
                    failures.lock().unwrap().push(e);
                } else {
                    results.lock().unwrap().push((i, row, other));
                }
                eprintln!(
                    "completed block {}/{} in {:.1}s",
                    i + 1,
                    a.blocks,
                    start.elapsed().as_secs_f64()
                );
            });
        }
    });
    let failed = failures.into_inner().unwrap();
    if !failed.is_empty() {
        for e in failed {
            eprintln!("FAILED: {e}");
        }
        std::process::exit(1);
    }
    let mut result = results.into_inner().unwrap();
    result.sort_by_key(|r| r.0);
    if let Some(out) = &a.out {
        let mut text=format!("# {:?}\nblock\tseed\tbot\tgames\tmean_gap\twin_share\tdecisions\tdecision_us\tmax_us\tcalls\tsuccesses\n",a);
        for (i, row, other) in &result {
            let seed = a
                .seed
                .wrapping_add((*i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            for (name, r) in
                std::iter::once((&a.candidate, row)).chain(a.compare.as_ref().map(|id| (id, other)))
            {
                text.push_str(&format!(
                    "{i}\t{seed}\t{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    a.players,
                    r.gap / a.players as f64,
                    r.win / a.players as f64,
                    r.decisions,
                    r.decision_us,
                    r.max_us,
                    r.calls,
                    r.successes
                ));
            }
        }
        std::fs::write(out, text).expect("could not write per-block results");
    }
    let rows: Vec<_> = result.iter().map(|r| r.1).collect();
    summarize(&a.candidate, &rows, a.players);
    if let Some(id) = &a.compare {
        let others: Vec<_> = result.iter().map(|r| r.2).collect();
        summarize(id, &others, a.players);
        let diffs: Vec<_> = result
            .iter()
            .map(|r| ResultRow {
                gap: r.1.gap - r.2.gap,
                win: r.1.win - r.2.win,
                ..ResultRow::default()
            })
            .collect();
        summarize_difference(&diffs, a.players);
    }
    println!(
        "elapsed={:.1}s; illegal_commands=0; blocks={} seats_per_block={}",
        start.elapsed().as_secs_f64(),
        a.blocks,
        a.players
    );
}
