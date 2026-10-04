//! 无头对局驱动：让若干 Bot 完整打完一局/一整个游戏。
//! 供 `cabo-sim` 模拟器与集成测试复用。

use std::collections::BTreeMap;

use rand::rngs::StdRng;

use crate::ai::{fallback_command, BotRegistry};
use crate::game::view::{project, Panel, PlayerView};
use crate::game::{Phase, PlayerId, PlayerState, Session, Settings};

/// 建一个全员 AI 的测试局。
pub fn make_ai_session(seed: u64, settings: Settings, bot_ids: &[&str]) -> Session {
    let players = bot_ids
        .iter()
        .enumerate()
        .map(|(i, &bot)| PlayerState {
            name: format!("P{}", i + 1),
            is_ai: true,
            bot_id: bot.to_string(),
            token: None,
            slots: Vec::new(),
            peeked_slots: Vec::new(),
            total_score: 0,
            round_score: None,
            score_reset_used: false,
            score_reset_this_round: false,
        })
        .collect();
    Session::new_lobby(seed, settings, players)
}

/// 单局游戏统计。
#[derive(Debug, Default, Clone)]
pub struct GameOutcome {
    pub rounds: u32,
    pub actions: u64,
    /// 各座位最终累计分（座位序）。
    pub totals: Vec<u32>,
    /// 获胜座位序。
    pub winners: Vec<PlayerId>,
}

/// 驱动一整局（多轮）直到 GameOver。`max_actions` 防死循环。
pub fn run_game(
    session: &mut Session,
    registry: &BotRegistry,
    rng: &mut StdRng,
    max_actions: u64,
) -> Result<GameOutcome, String> {
    session.start_game().map_err(|e| format!("开局失败: {e}"))?;
    let mut actions = 0u64;
    loop {
        if actions > max_actions {
            return Err("超出动作上限，可能死循环".into());
        }
        match &session.phase.clone() {
            Phase::Peeking { done } => {
                let mut acted = false;
                for pid in 0..session.players.len() {
                    if !done.contains_key(&pid) {
                        step(session, registry, rng, pid, &mut actions)?;
                        acted = true;
                        break;
                    }
                }
                if !acted && done.len() == session.players.len() {
                    return Err("查看阶段无人可行动".into());
                }
            }
            Phase::Turn { current, .. } => {
                let pid = *current;
                if !session.players[pid].is_ai {
                    return Err(format!("模拟器只支持全 AI 局，座位 {pid} 是人类"));
                }
                step(session, registry, rng, pid, &mut actions)?;
            }
            Phase::RoundEnd => {
                actions += 1;
                session.next_round().map_err(|e| format!("下一轮失败: {e}"))?;
            }
            Phase::GameOver { winners } => {
                return Ok(GameOutcome {
                    rounds: session.round_no,
                    actions,
                    totals: session.players.iter().map(|p| p.total_score).collect(),
                    winners: winners.clone(),
                });
            }
            Phase::Lobby => return Err("对局尚未开始".into()),
        }
    }
}

/// 让座位 `pid` 的 Bot 行动一步（含非法决策时的兜底重试）。
fn step(
    session: &mut Session,
    registry: &BotRegistry,
    rng: &mut StdRng,
    pid: PlayerId,
    actions: &mut u64,
) -> Result<(), String> {
    let bot = registry
        .get(&session.players[pid].bot_id)
        .ok_or_else(|| format!("未注册的 bot: {}", session.players[pid].bot_id))?;
    for attempt in 0..3 {
        let view: PlayerView = project(session, Some(pid), 0);
        if matches!(view.panel, Panel::Waiting { .. } | Panel::Spectating) {
            return Err("座位未被要求行动".into());
        }
        let cmd = if attempt == 0 { bot.decide(&view, rng) } else { fallback_command(&view) };
        if std::env::var("SIM_DEBUG").is_ok() {
            eprintln!("[step] P{pid} attempt{attempt} panel={:?} cmd={cmd:?}", view.panel);
        }
        *actions += 1;
        match session.apply(pid, &cmd) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if attempt == 2 {
                    return Err(format!(
                        "P{pid} 连续决策非法: {e}（命令 {cmd:?}，面板 {:?}，阶段 {:?}）",
                        view.panel, session.phase
                    ));
                }
            }
        }
    }
    Ok(())
}

/// 跑多局并汇总。
pub struct Aggregate {
    pub games: usize,
    pub wins: BTreeMap<String, usize>,
    pub avg_rounds: f64,
    pub avg_actions: f64,
    pub avg_total: f64,
}

pub fn aggregate(outcomes: &[GameOutcome], names: &[String]) -> Aggregate {
    let mut wins: BTreeMap<String, usize> = BTreeMap::new();
    for n in names {
        wins.insert(n.clone(), 0);
    }
    let mut rounds = 0u64;
    let mut actions = 0u64;
    let mut total = 0u64;
    let mut cells = 0u64;
    for o in outcomes {
        rounds += o.rounds as u64;
        actions += o.actions;
        for (i, &t) in o.totals.iter().enumerate() {
            total += t as u64;
            cells += 1;
            let _ = i;
        }
        for &w in &o.winners {
            if let Some(name) = names.get(w) {
                *wins.entry(name.clone()).or_insert(0) += 1;
            }
        }
    }
    let n = outcomes.len().max(1) as f64;
    Aggregate {
        games: outcomes.len(),
        wins,
        avg_rounds: rounds as f64 / n,
        avg_actions: actions as f64 / n,
        avg_total: if cells == 0 { 0.0 } else { total as f64 / cells as f64 },
    }
}
