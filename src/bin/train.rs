//! `cabo-train`：自我对弈 + 学习价值函数（Expert Iteration 的"学习"那一半）。
//!
//! 流程：
//! 1. **自我对弈产数据**：让搜索型 Bot 互相对打若干局；在每个决策点（以及
//!    "我的动作生效后"的节点）记录特征向量与**最终整局胜负**作为标签；
//! 2. **训练**：小 MLP + Adam + BCE，留出验证集，报告与启发式基线的对比；
//! 3. **导出**：把权重写成 `src/ai/weights.rs`（编译期嵌入，保持单文件部署）。
//!
//! 用法：
//! ```text
//! cargo run --release --bin cabo-train -- --games 120 --players 4 --epochs 60
//! cargo run --release --bin cabo-train -- --games 120 --out src/ai/weights.rs
//! ```

use std::sync::Arc;
use std::time::Instant;

use cabo::ai::nn::{Adam, Mlp};
use cabo::ai::search::{SearchBot, SearchCfg};
use cabo::ai::value::{features, N_FEATURES};
use cabo::ai::BotRegistry;
use cabo::game::sim::make_ai_session;
use cabo::game::view::project;
use cabo::game::{Phase, Settings};
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Sample {
    x: Vec<f32>,
    y: f32,
    /// 样本来自哪一局（划分验证集必须按局切，否则同局样本泄漏会让验证分数虚高）
    game: u32,
}

#[allow(clippy::too_many_arguments)]
fn train_policy(
        games: usize,
    players: usize,
    epochs: usize,
    hidden: usize,
    seed: u64,
    target: u32,
    budget_us: u64,
    out: &str,
    write: bool,
    dump: Option<&str>,
    fixed_worlds: u32,
) {
    use cabo::ai::history::History;
    use cabo::ai::policy::{candidate_features, state_features, N_CAND, N_IN, N_STATE};
    use cabo::ai::search::candidates_scored;
    use cabo::ai::tactics::{view_policy_with, Know, PolicyCfg};
    use cabo::game::view::Panel;
    use cabo::game::Command;

    let settings = Settings { cabo_penalty: 10, target_score: target, memory_mode: false };
    let mut registry = BotRegistry::new();
    let mut scfg = SearchCfg { budget_us, ..SearchCfg::default() };
    if fixed_worlds > 0 {
        // 固定世界数 = **确定性教师**（时间预算会让决策随机器负载变化，标签会带噪声）
        scfg.min_worlds = fixed_worlds;
        scfg.max_worlds = fixed_worlds;
        scfg.budget_us = u64::MAX / 2;
    }
    registry.register(Arc::new(SearchBot::new(scfg)));
    let policy_cfg = PolicyCfg::default();

    // 一条样本 = 一个决策点：状态特征 + 各候选特征 + 教师选择的下标
    struct Decision {
        state: Vec<f32>,
        cands: Vec<Vec<f32>>,
        label: usize,
        prior: Option<usize>,
        game: u32,
    }
    let t0 = Instant::now();
    let mut data: Vec<Decision> = Vec::new();
    let mut n_agree_prior = 0usize;
    for g in 0..games {
        let game_seed = seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let bots: Vec<&str> = (0..players).map(|_| "search").collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &bots);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
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
            // 采**所有会被 rollout 用到的面板**：以前只采 Idle，导致"摸牌后"这类
            // 在 rollout 里最常出现的决策网络从没学过（分布不匹配）。
            if matches!(
                view.panel,
                Panel::Idle { .. }
                    | Panel::Drew { .. }
                    | Panel::SwapSelecting { .. }
                    | Panel::ConfirmCabo
            ) {
                let info = history.observe(&view);
                let k = Know::from_view_with(&view, &info.known);
                let scored = candidates_scored(&view, &k, 10);
                if scored.len() >= 2 {
                    let cmds: Vec<Command> = scored.iter().map(|(_, c)| c.clone()).collect();
                    let chosen = registry.get("search").unwrap().decide(&view, &mut rng);
                    if let Some(label) = cmds.iter().position(|c| *c == chosen) {
                        let prior_cmd = view_policy_with(&view, &policy_cfg, &mut rng, &info.known);
                        let prior = cmds.iter().position(|c| *c == prior_cmd);
                        if prior == Some(label) {
                            n_agree_prior += 1;
                        }
                        let state = state_features(&view, &k);
                        let cands: Vec<Vec<f32>> = scored
                            .iter()
                            .enumerate()
                            .map(|(i, (s, c))| {
                                candidate_features(&view, &k, c, i, prior, *s as f32)
                            })
                            .collect();
                        data.push(Decision { state, cands, label, prior, game: g as u32 });
                    }
                }
            }
            let cmd = registry.get("search").unwrap().decide(&view, &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
    }
    println!(
        "=== cabo-train（策略头）===\n对局 {} 局，决策样本 {}，产数据耗时 {:.0}s",
        games,
        data.len(),
        t0.elapsed().as_secs_f64()
    );
    let n_cand_avg: f64 =
        data.iter().map(|d| d.cands.len() as f64).sum::<f64>() / data.len().max(1) as f64;
    println!("平均候选数 {n_cand_avg:.2}");

    // ---- 导出数据集给 PyTorch（GPU）训练用 ----
    if let Some(path) = dump {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(N_STATE as u32).to_le_bytes());
        buf.extend_from_slice(&(N_CAND as u32).to_le_bytes());
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        for d in &data {
            buf.extend_from_slice(&(d.cands.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(d.label as u32).to_le_bytes());
            buf.extend_from_slice(&(d.prior.unwrap_or(u32::MAX as usize) as u32).to_le_bytes());
            buf.extend_from_slice(&d.game.to_le_bytes());
            for v in &d.state {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            for c in &d.cands {
                for v in c {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        std::fs::write(path, &buf).expect("写数据集失败");
        println!(
            "数据集已写入 {path}（{} 决策，{:.1} MB）",
            data.len(),
            buf.len() as f64 / 1e6
        );
        return;
    }
    println!(
        "启发式先验的 top-1 命中率 = {:.1}%（这就是策略头要超越的基线）",
        100.0 * n_agree_prior as f64 / data.len().max(1) as f64
    );

    // 按局切分
    let mut rng = StdRng::seed_from_u64(seed ^ 0x99);
    let mut gids: Vec<u32> = (0..games as u32).collect();
    rand::seq::SliceRandom::shuffle(&mut gids[..], &mut rng);
    let n_val = (games / 5).max(1);
    let val_games: std::collections::HashSet<u32> = gids[..n_val].iter().copied().collect();
    let val: Vec<usize> =
        (0..data.len()).filter(|&i| val_games.contains(&data[i].game)).collect();
    let train: Vec<usize> =
        (0..data.len()).filter(|&i| !val_games.contains(&data[i].game)).collect();
    println!("按局划分：验证 {} 局 / {} 决策，训练 {} 决策", n_val, val.len(), train.len());

    let mut net = Mlp::new(&[N_IN, hidden, hidden, 1], &mut StdRng::seed_from_u64(seed ^ 0x77));
    let last = net.acts.len() - 1;
    net.acts[last] = cabo::ai::nn::Act::Linear; // 打分是 logit
    let mut opt = Adam::new(net.params(), 0.02);
    let score = |net: &Mlp, d: &Decision, ci: usize| -> f32 {
        let mut x = Vec::with_capacity(N_IN);
        x.extend_from_slice(&d.state);
        x.extend_from_slice(&d.cands[ci]);
        net.predict(&x)
    };
    // 列表 softmax 交叉熵（数值稳定）
    let softmax = |z: &[f32]| -> Vec<f32> {
        let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|v| v / s.max(1e-9)).collect()
    };
    let top1 = |net: &Mlp, idx: &[usize]| -> f64 {
        let mut hit = 0usize;
        for &i in idx {
            let d = &data[i];
            let z: Vec<f32> = (0..d.cands.len()).map(|c| score(net, d, c)).collect();
            let best = z
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0);
            if best == d.label {
                hit += 1;
            }
        }
        hit as f64 / idx.len().max(1) as f64
    };
    // 先验在验证集上的命中率（同口径）
    let prior_hit = {
        let mut hit = 0usize;
        let mut n = 0usize;
        for &i in &val {
            if let Some(p) = data[i].prior {
                n += 1;
                if p == data[i].label {
                    hit += 1;
                }
            }
        }
        hit as f64 / n.max(1) as f64
    };
    println!("验证集上启发式先验命中率 = {prior_hit:.4}");

    let batch = 32usize;
    let mut best_acc = 0f64;
    let mut best = net.clone();
    for ep in 0..epochs {
        let mut order = train.clone();
        rand::seq::SliceRandom::shuffle(&mut order[..], &mut rng);
        let mut loss_sum = 0f64;
        let mut nb = 0usize;
        for chunk in order.chunks(batch) {
            let mut grads = net.zero_grad();
            for &i in chunk {
                let d = &data[i];
                let z: Vec<f32> = (0..d.cands.len()).map(|c| score(&net, d, c)).collect();
                let p = softmax(&z);
                loss_sum += -(p[d.label].max(1e-9) as f64).ln();
                for (c, pc) in p.iter().enumerate() {
                    let delta = if c == d.label { *pc - 1.0 } else { *pc };
                    let mut x = Vec::with_capacity(N_IN);
                    x.extend_from_slice(&d.state);
                    x.extend_from_slice(&d.cands[c]);
                    net.backward_delta(&x, delta, &mut grads, 1e-5);
                }
            }
            opt.step(&mut net, &mut grads, chunk.len() as f32);
            nb += 1;
        }
        let acc = top1(&net, &val);
        if acc > best_acc {
            best_acc = acc;
            best = net.clone();
        }
        if ep % 5 == 0 || ep + 1 == epochs {
            println!(
                "epoch {:3}  训练 loss {:.4}  验证 top-1 {:.4}（先验 {:.4}）",
                ep + 1,
                loss_sum / (nb * batch) as f64,
                acc,
                prior_hit
            );
        }
    }
    let train_acc = top1(&best, &train);
    println!(
        "最佳验证 top-1 = {best_acc:.4}（先验 {prior_hit:.4}，提升 {:+.4}）；训练集 top-1 = {train_acc:.4}",
        best_acc - prior_hit
    );
    if !write {
        println!("（--dry：不写文件）");
        return;
    }
    // 追加到权重文件（保留 value 段）
    let data_flat = best.export();
    let existing = std::fs::read_to_string(out).unwrap_or_default();
    let head = match existing.find("/// 策略网络（候选打分）各层宽度") {
        Some(p) => existing[..p].to_string(),
        None => existing,
    };
    let mut text = head;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&format!(
        "/// 策略网络（候选打分）各层宽度：输入 = 状态特征 ⊕ 候选特征。\n/// 训练：{} 局自我对弈 / {} 决策；验证 top-1 {:.4}（启发式先验 {:.4}）。\npub const POLICY_SIZES: &[usize] = &[{}, {}];\n\n/// 策略网络参数。\npub const POLICY_WEIGHTS: &[f32] = &[\n",
        games,
        data.len(),
        best_acc,
        prior_hit,
        N_STATE + N_CAND,
        best.layers.iter().map(|l| l.n_out.to_string()).collect::<Vec<_>>().join(", ")
    ));
    for chunk in data_flat.chunks(8) {
        text.push_str("    ");
        for v in chunk {
            text.push_str(&format!("{v:?}, "));
        }
        text.push('\n');
    }
    text.push_str("];\n");
    std::fs::write(out, text).expect("写入权重失败");
    println!("策略网络权重已写入 {out}（{} 个参数，输入 {} 维）", data_flat.len(), N_IN);
}


/// `--mode value3`：导出完整局面 + 稠密 target（终局分差）的价值数据集。
///
/// 采样点 = 本轮结算后（搜索里价值被调用的位置）；解析基线 `lead_now` 一并导出，
/// 于是网络只负责学"从现在到终局的修正量"。
#[allow(clippy::too_many_arguments)]
fn dump_value3(
    games: usize,
    players: usize,
    seed: u64,
    target: u32,
    budget_us: u64,
    path: &str,
) {
    use cabo::ai::value::{features_full, N_FULL};

    let settings = Settings { cabo_penalty: 10, target_score: target, memory_mode: false };
    let mut registry = BotRegistry::new();
    registry.register(Arc::new(SearchBot::new(SearchCfg {
        budget_us,
        ..SearchCfg::default()
    })));

    struct Row {
        x: Vec<f32>,
        lead: f32,
        game: u32,
        seat: u32,
    }
    let t0 = Instant::now();
    let mut rows: Vec<Row> = Vec::new();
    let mut finals: Vec<(u32, u32, f32)> = Vec::new();
    for g in 0..games {
        let game_seed = seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let bots: Vec<&str> = (0..players).map(|_| "search").collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &bots);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
        for _ in 0..200_000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                Phase::RoundEnd => {
                    for seat in 0..players {
                        let mine = s.players[seat].total_score as f32;
                        let min_other = s
                            .players
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != seat)
                            .map(|(_, p)| p.total_score as f32)
                            .fold(f32::MAX, f32::min);
                        rows.push(Row {
                            x: features_full(&s, seat),
                            lead: min_other - mine,
                            game: g as u32,
                            seat: seat as u32,
                        });
                    }
                    s.next_round().unwrap();
                    continue;
                }
                Phase::GameOver { .. } | Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            let view = project(&s, Some(pid), 0);
            let cmd = registry.get("search").unwrap().decide(&view, &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
        // 本局终局分差（每个座位一行）
        for seat in 0..players {
            let mine = s.players[seat].total_score as f32;
            let min_other = s
                .players
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != seat)
                .map(|(_, p)| p.total_score as f32)
                .fold(f32::MAX, f32::min);
            finals.push((g as u32, seat as u32, min_other - mine));
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(N_FULL as u32).to_le_bytes());
    buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for r in &rows {
        buf.extend_from_slice(&r.lead.to_le_bytes());
        buf.extend_from_slice(&r.game.to_le_bytes());
        buf.extend_from_slice(&r.seat.to_le_bytes());
        for v in &r.x {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(finals.len() as u32).to_le_bytes());
    for (g, seat, margin) in &finals {
        buf.extend_from_slice(&g.to_le_bytes());
        buf.extend_from_slice(&seat.to_le_bytes());
        buf.extend_from_slice(&margin.to_le_bytes());
    }
    std::fs::write(path, &buf).expect("写数据集失败");
    println!(
        "=== cabo-train（价值 v3：完整局面 + 终局分差）===\n对局 {} 局，样本 {}，终局记录 {}，耗时 {:.0}s\n数据集 {path}（{:.1} MB）",
        games,
        rows.len(),
        finals.len(),
        t0.elapsed().as_secs_f64(),
        buf.len() as f64 / 1e6
    );
}


/// `--mode scorectx`：导出"分数上下文 → 最终胜负"的数据集（低维、可查校准）。
fn dump_scorectx(games: usize, players: usize, seed: u64, target: u32, path: &str) {
    let settings = Settings { cabo_penalty: 10, target_score: target, memory_mode: false };
    let mut registry = BotRegistry::new();
    registry.register(Arc::new(cabo::ai::tactics::TacticianBot::new(
        cabo::ai::tactics::PolicyCfg::default(),
    )));

    // 一行 = 某个座位在某轮结算后的分数上下文
    let mut rows: Vec<[f32; 5]> = Vec::new(); // my_total, min_other, second_other, round_no, target
    let mut labels: Vec<f32> = Vec::new();
    let mut games_done = 0usize;
    for g in 0..games {
        let game_seed = seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let bots: Vec<&str> = (0..players).map(|_| "tactician").collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &bots);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
        let mut pending: Vec<[f32; 5]> = Vec::new();
        for _ in 0..200_000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                Phase::RoundEnd => {
                    for seat in 0..players {
                        let mut totals: Vec<f32> = (0..players)
                            .map(|i| s.players[i].total_score as f32)
                            .collect();
                        let mine = totals[seat];
                        totals.remove(seat);
                        totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        pending.push([
                            mine,
                            totals.first().copied().unwrap_or(0.0),
                            totals.get(1).copied().unwrap_or(0.0),
                            s.round_no as f32,
                            target as f32,
                        ]);
                    }
                    s.next_round().unwrap();
                    continue;
                }
                Phase::GameOver { .. } | Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            let view = project(&s, Some(pid), 0);
            let cmd = registry.get("tactician").unwrap().decide(&view, &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
        }
        let best = s.players.iter().map(|p| p.total_score).min().unwrap_or(0);
        // 标签：是否并列最低（与其它工具的"胜者"口径一致）
        for seat in 0..players {
            let _ = seat;
        }
        for row in &pending {
            // 用行里的 my_total 找座位不可靠 → 重新按座位顺序推
            let _ = row;
        }
        // 重新按座位标注（pending 是按 seat 顺序 push 的，每轮 players 行）
        for (i, row) in pending.iter().enumerate() {
            let seat = i % players;
            let _ = row;
            labels.push(if s.players[seat].total_score == best { 1.0 } else { 0.0 });
        }
        rows.extend(pending);
        games_done += 1;
        if games_done % 1000 == 0 {
            eprintln!("[scorectx] {games_done}/{games} 局，样本 {}", rows.len());
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for (row, y) in rows.iter().zip(labels.iter()) {
        for v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&y.to_le_bytes());
    }
    std::fs::write(path, &buf).expect("写数据集失败");
    println!(
        "=== cabo-train（scorectx）===\n对局 {} 局，样本 {}（正例 {:.1}%）\n数据集 {path}（{:.2} MB）",
        games,
        rows.len(),
        100.0 * labels.iter().sum::<f32>() / labels.len().max(1) as f32,
        buf.len() as f64 / 1e6
    );
}

fn main() {
    let mut games = 120usize;
    let mut players = 4usize;
    let mut epochs = 60usize;
    let mut hidden = 64usize;
    let mut seed = 1u64;
    let mut out = String::from("src/ai/weights.rs");
    let mut target = 100u32;
    let mut budget_us = 4000u64;
    let mut fixed_worlds = 0u32;
    let mut write = true;
    let mut step_mode = true;  // 只记录"决策点 + 本轮结算后"（终端估值器用法）
    let mut mode = String::from("value");
    let mut dump: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_default();
        match a.as_str() {
            "--mode" => mode = v(),
            "--dump" => dump = Some(v()),
            "--games" => games = v().parse().unwrap_or(games),
            "--players" => players = v().parse().unwrap_or(players),
            "--epochs" => epochs = v().parse().unwrap_or(epochs),
            "--hidden" => hidden = v().parse().unwrap_or(hidden),
            "--seed" => seed = v().parse().unwrap_or(seed),
            "--out" => out = v(),
            "--target" => target = v().parse().unwrap_or(target),
            "--budget-us" => budget_us = v().parse().unwrap_or(budget_us),
            "--worlds" => fixed_worlds = v().parse().unwrap_or(fixed_worlds),
            "--dry" => write = false,
            "--all-states" => step_mode = false,
            other => {
                eprintln!("未知参数 {other}");
                std::process::exit(2);
            }
        }
    }

    if mode == "scorectx" {
        let path = dump.clone().unwrap_or_else(|| "data/scorectx.bin".to_string());
        dump_scorectx(games, players, seed, target, &path);
        return;
    }
    if mode == "value3" {
        let path = dump.clone().unwrap_or_else(|| "data/value3.bin".to_string());
        dump_value3(games, players, seed, target, budget_us, &path);
        return;
    }
    if mode == "policy" {
        train_policy(
            games,
            players,
            epochs,
            hidden,
            seed,
            target,
            budget_us,
            &out,
            write,
            dump.as_deref(),
            fixed_worlds,
        );
        return;
    }

    let settings = Settings { cabo_penalty: 10, target_score: target, memory_mode: false };
    let mut registry = BotRegistry::new();
    registry.register(Arc::new(SearchBot::new(SearchCfg {
        budget_us,
        ..SearchCfg::default()
    })));

    // ---------------------------------------------------------------- 产数据
    let t0 = Instant::now();
    let mut samples: Vec<Sample> = Vec::new();
    let mut wins = 0usize;
    for g in 0..games {
        let game_seed = seed.wrapping_add(g as u64 * 0x9E3779B97F4A7C15);
        let bots: Vec<&str> = (0..players).map(|_| "search").collect();
        let mut s = make_ai_session(game_seed, settings.clone(), &bots);
        s.start_game().unwrap();
        let mut rng = StdRng::seed_from_u64(game_seed ^ 0x5DEECE66D);
        // 记录本局所有座位的 (特征, 座位)
        let mut pending: Vec<(usize, Vec<f32>)> = Vec::new();
        for _ in 0..200_000 {
            let pid = match &s.phase {
                Phase::Peeking { done } => (0..players).find(|p| !done.contains_key(p)),
                Phase::Turn { current, .. } => Some(*current),
                Phase::RoundEnd => {
                    // 记录"本轮结算后"的局面：这正是搜索里 rollout 打到的地方，
                    // 用它训练 = 学习一个**终局估值器**（分布一致、标签有效）。
                    if step_mode {
                        for pid in 0..players {
                            pending.push((pid, features(&s, pid)));
                        }
                    }
                    s.next_round().unwrap();
                    continue;
                }
                Phase::GameOver { .. } => break,
                Phase::Lobby => break,
            };
            let Some(pid) = pid else { break };
            // 决策点特征（只在自己回合记录，避免海量重复）
            if step_mode && matches!(s.phase, Phase::Turn { .. }) {
                pending.push((pid, features(&s, pid)));
            }
            let view = project(&s, Some(pid), 0);
            let cmd = registry.get(&s.players[pid].bot_id).unwrap().decide(&view, &mut rng);
            if s.apply(pid, &cmd).is_err() {
                let _ = s.apply(pid, &cabo::ai::fallback_command(&view));
            }
            // "我的动作生效后"的节点（搜索真正估值的位置）
            if !step_mode
                && matches!(s.phase, Phase::Turn { .. })
                && matches!(s.phase, Phase::Turn { current, .. } if current != pid)
            {
                pending.push((pid, features(&s, pid)));
            }
        }
        let winners: Vec<usize> =
            (0..players).filter(|&i| s.players[i].total_score == s.players.iter().map(|p| p.total_score).min().unwrap()).collect();
        if !winners.is_empty() {
            wins += 1;
        }
        for (pid, x) in pending {
            let y = if winners.contains(&pid) { 1.0 } else { 0.0 };
            samples.push(Sample { x, y, game: g as u32 });
        }
        if (g + 1) % 10 == 0 {
            eprintln!(
                "[数据] {}/{} 局，样本 {}，累计 {:.0}s",
                g + 1,
                games,
                samples.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let pos = samples.iter().filter(|s| s.y > 0.5).count();
    println!(
        "=== cabo-train ===\n对局 {} 局（{} 人，阈值 {}），样本 {}（正例 {:.1}%），产数据耗时 {:.0}s",
        games,
        players,
        target,
        samples.len(),
        100.0 * pos as f64 / samples.len().max(1) as f64,
        t0.elapsed().as_secs_f64()
    );
    let _ = wins;

    // 按**局**划分训练/验证（同局样本高度相关，按样本随机切会严重高估）
    let mut rng = StdRng::seed_from_u64(seed ^ 0xABCDEF);
    let n_val_games = (games / 10).max(1);
    let mut game_ids: Vec<u32> = (0..games as u32).collect();
    rand::seq::SliceRandom::shuffle(&mut game_ids[..], &mut rng);
    let val_games: std::collections::HashSet<u32> =
        game_ids[..n_val_games].iter().copied().collect();
    let val_idx: Vec<usize> =
        (0..samples.len()).filter(|&i| val_games.contains(&samples[i].game)).collect();
    let train_idx: Vec<usize> =
        (0..samples.len()).filter(|&i| !val_games.contains(&samples[i].game)).collect();
    println!("按局划分：验证 {} 局 / {} 条样本，训练 {} 条", n_val_games, val_idx.len(), train_idx.len());

    // 启发式基线：sigmoid(领先/8) 的 Brier 分数，用来判断"学出来的到底有没有用"
    let heuristic = |s: &Sample| -> f64 {
        // 特征里"领先量"在第 3*3+2+14+1 = 33 位（见 value::features）
        let lead = s.x[33] as f64 * 50.0;
        1.0 / (1.0 + (-lead / 8.0).exp())
    };
    let brier_h: f64 = val_idx.iter().map(|&i| (heuristic(&samples[i]) - samples[i].y as f64).powi(2)).sum::<f64>()
        / val_idx.len() as f64;
    println!("启发式(领先/8 logistic) 验证集 Brier = {brier_h:.4}（越低越好，0.25 = 瞎猜）");

    // ---------------------------------------------------------------- 训练
    let base = samples.iter().map(|s| s.y as f64).sum::<f64>() / samples.len() as f64;
    let brier_base: f64 =
        val_idx.iter().map(|&i| (base - samples[i].y as f64).powi(2)).sum::<f64>() / val_idx.len() as f64;
    println!("常数基线(全预测正例率)  验证集 Brier = {brier_base:.4}");

    let net_rng = StdRng::seed_from_u64(seed ^ 0x1234);
    let mut net = Mlp::new(&[N_FEATURES, hidden, hidden, 1], &mut { net_rng });
    // 残差价值：最后一层线性输出 logit，外部与解析基线相加后再 sigmoid
    let last = net.acts.len() - 1;
    net.acts[last] = cabo::ai::nn::Act::Linear;
    let predict = |net: &Mlp, x: &[f32]| -> f64 {
        let z = net.predict(x) + cabo::ai::value::baseline_logit(x);
        (1.0 / (1.0 + (-z).exp())) as f64
    };
    let mut opt = Adam::new(net.params(), 0.01);
    let mut best_brier = f64::MAX;
    let mut best = net.clone();
    let batch = 64usize;
    for ep in 0..epochs {
        let mut order = train_idx.to_vec();
        rand::seq::SliceRandom::shuffle(&mut order[..], &mut rng);
        let mut loss_sum = 0f64;
        let mut nb = 0usize;
        for chunk in order.chunks(batch) {
            let mut grads = net.zero_grad();
            for &i in chunk {
                let x = &samples[i].x;
                let y = samples[i].y;
                // 组合模型的损失与梯度：p = sigmoid(基线 + z)，dL/dz = p - y
                let p = predict(&net, x).clamp(1e-6, 1.0 - 1e-6);
                loss_sum += -(y as f64 * p.ln() + (1.0 - y as f64) * (1.0 - p).ln());
                net.backward_delta(x, (p - y as f64) as f32, &mut grads, 1e-5);
            }
            opt.step(&mut net, &mut grads, chunk.len() as f32);
            nb += 1;
        }
        let brier: f64 = val_idx
            .iter()
            .map(|&i| (predict(&net, &samples[i].x) - samples[i].y as f64).powi(2))
            .sum::<f64>()
            / val_idx.len() as f64;
        if brier < best_brier {
            best_brier = brier;
            best = net.clone();
        }
        if ep % 10 == 0 || ep + 1 == epochs {
            println!(
                "epoch {:3}  训练 loss {:.4}  验证 Brier {:.4}{}",
                ep + 1,
                loss_sum / (nb * batch) as f64,
                brier,
                if brier <= best_brier { "  *" } else { "" }
            );
        }
        if ep > 20 && brier > best_brier * 1.02 {
            println!("验证集不再改善，提前停止于 epoch {}", ep + 1);
            break;
        }
    }
    let net = best;
    println!(
        "最佳验证 Brier = {best_brier:.4}  相对启发式 {:+.4}  相对常数基线 {:+.4}",
        best_brier - brier_h,
        best_brier - brier_base
    );
    // 校准观察：分箱平均预测 vs 实际频率
    println!("校准检查（验证集）:");
    for b in 0..5 {
        let lo = b as f64 / 5.0;
        let hi = (b + 1) as f64 / 5.0;
        let mut cnt = 0usize;
        let mut psum = 0f64;
        let mut ysum = 0f64;
        for &i in val_idx.iter() {
            let p = predict(&net, &samples[i].x);
            if p >= lo && p < hi {
                cnt += 1;
                psum += p;
                ysum += samples[i].y as f64;
            }
        }
        if cnt > 0 {
            println!(
                "  p∈[{lo:.1},{hi:.1})  n={cnt:5}  预测均值 {:.3}  实际胜率 {:.3}",
                psum / cnt as f64,
                ysum / cnt as f64
            );
        }
    }

    // ---------------------------------------------------------------- 导出
    if !write {
        println!("（--dry：不写文件）");
        return;
    }
    let data = net.export();
    let mut text = String::new();
    text.push_str("//! 训练产物：由 `cargo run --release --bin cabo-train` 自动生成，请勿手改。\n");
    text.push_str(&format!(
        "//!\n//! 样本 {} 条（{} 局自我对弈），验证集 Brier {:.4}（启发式基线 {:.4}，常数基线 {:.4}）。\n",
        samples.len(),
        games,
        best_brier,
        brier_h,
        brier_base
    ));
    text.push_str("\n/// 网络各层宽度。\npub const VALUE_SIZES: &[usize] = &[");
    let sizes: Vec<String> = net.layers.iter().map(|l| l.n_out.to_string()).collect();
    text.push_str(&format!("{}, {}", N_FEATURES, sizes.join(", ")));
    text.push_str("];\n\n/// `Mlp::export()` 的产物（扁平参数）。\npub const VALUE_WEIGHTS: &[f32] = &[\n");
    for chunk in data.chunks(8) {
        text.push_str("    ");
        for v in chunk {
            text.push_str(&format!("{v:?}, "));
        }
        text.push('\n');
    }
    text.push_str("];\n");
    std::fs::write(&out, text).expect("写入权重失败");
    println!("权重已写入 {out}（{} 个参数）", data.len());
}
