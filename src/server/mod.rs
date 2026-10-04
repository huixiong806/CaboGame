//! Web 服务端：房间生命周期、会话令牌、AI 驱动循环、SSE 推送。
//!
//! 并发模型：
//! - 每个房间一个 `Arc<Room>`；房间内可变状态放在 `Mutex<RoomInner>`，
//!   临界区内不做任何 `.await`。
//! - 每次状态变更把 `watch::Sender<u64>` 的版本号 +1；每条 SSE 连接
//!   `changed().await` 等待版本变化后，各自按"该玩家视角"渲染一次画面。
//!   服务端只保存牌局状态本身，不为连接保存 UI 状态（客户端无状态）。
//! - AI 座位由每房间唯一的 `ai_loop` 任务驱动：发现轮到 AI 就稍作停顿、
//!   重新取视图、调 Bot 决策并应用。

pub mod render;
pub mod routes;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::Rng;
use tokio::sync::watch;

use crate::ai::{fallback_command, Bot, BotRegistry};
use crate::game::view::{project, PlayerView};
use crate::game::{Command, Phase, PlayerId, PlayerState, Session, Settings};

/// AI 出牌思考时间（毫秒），让真人能看清节奏。
pub const AI_THINK_MS: u64 = 900;
/// 房间无活动且无连接时的回收时间。
const ROOM_IDLE: Duration = Duration::from_secs(60 * 60 * 2);
const ROOM_CODE_ALPHABET: &[u8] = b"23456789ABCDEFGHJKMNPQRSTUVWXYZ";

pub struct Shared {
    pub rooms: Mutex<HashMap<String, Arc<Room>>>,
    pub bots: BotRegistry,
}

pub struct Room {
    pub code: String,
    pub inner: Mutex<RoomInner>,
    pub version: watch::Sender<u64>,
    /// 存活的 SSE 连接数（用于回收判定）。
    pub conns: AtomicUsize,
}

pub struct RoomInner {
    pub session: Session,
    pub host: PlayerId,
    pub spectators: Vec<Spectator>,
    /// 最近一次操作错误（仅展示给操作者）。
    pub last_error: Option<(PlayerId, String)>,
    pub last_active: Instant,
}

pub struct Spectator {
    pub token: String,
    pub name: String,
}

/// 令牌对应的身份：座位玩家或观战者。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Viewer {
    Player(PlayerId),
    Spectator(usize),
}

impl Room {
    pub fn bump(&self) {
        self.version.send_modify(|v| *v += 1);
    }

    /// 加锁；若某次渲染 panic 毒化了锁则恢复而非连锁崩溃。
    pub fn inner_lock(&self) -> std::sync::MutexGuard<'_, RoomInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub fn resolve_viewer(inner: &RoomInner, token: &str) -> Option<Viewer> {
    if let Some(i) =
        inner.session.players.iter().position(|p| p.token.as_deref() == Some(token))
    {
        return Some(Viewer::Player(i));
    }
    inner.spectators.iter().position(|s| s.token == token).map(Viewer::Spectator)
}

fn new_token() -> String {
    let b: [u8; 16] = rand::rng().random();
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn new_code() -> String {
    (0..4).map(|_| ROOM_CODE_ALPHABET[rand::rng().random_range(0..ROOM_CODE_ALPHABET.len())] as char)
        .collect()
}

/// 建立新房间并返回 (房间, 房主令牌)。
pub fn create_room(shared: &Shared, host_name: String, settings: Settings) -> (Arc<Room>, String) {
    let mut rooms = shared.rooms.lock().unwrap();
    let code = loop {
        let c = new_code();
        if !rooms.contains_key(&c) {
            break c;
        }
    };
    let token = new_token();
    let host = PlayerState {
        name: host_name,
        is_ai: false,
        bot_id: String::new(),
        token: Some(token.clone()),
        slots: Vec::new(),
        peeked_slots: Vec::new(),
        total_score: 0,
        round_score: None,
    };
    let session = Session::new_lobby(rand::rng().random(), settings, vec![host]);
    let room = Arc::new(Room {
        code: code.clone(),
        inner: Mutex::new(RoomInner {
            session,
            host: 0,
            spectators: Vec::new(),
            last_error: None,
            last_active: Instant::now(),
        }),
        version: watch::Sender::new(0),
        conns: AtomicUsize::new(0),
    });
    rooms.insert(code, room.clone());
    drop(rooms);
    tokio::spawn(ai_loop(room.clone(), shared.bots.clone()));
    (room, token)
}

/// 加入房间：大厅且有空位时落座，否则观战。
pub fn join_room(shared: &Shared, code: &str, name: String) -> Result<(Arc<Room>, String), String> {
    let room = shared
        .rooms
        .lock()
        .unwrap()
        .get(code.to_uppercase().as_str())
        .cloned()
        .ok_or_else(|| "房间不存在".to_string())?;
    let mut inner = room.inner_lock();
    let token = new_token();
    let can_seat = inner.session.phase == Phase::Lobby
        && inner.session.players.len() < 4;
    if can_seat {
        inner.session.players.push(PlayerState {
            name,
            is_ai: false,
            bot_id: String::new(),
            token: Some(token.clone()),
            slots: Vec::new(),
            peeked_slots: Vec::new(),
            total_score: 0,
            round_score: None,
        });
    } else {
        if inner.spectators.len() >= 8 {
            return Err("观战席已满".into());
        }
        inner.spectators.push(Spectator { token: token.clone(), name });
    }
    inner.last_active = Instant::now();
    drop(inner);
    room.bump();
    Ok((room, token))
}

/// 离开房间：大厅移除座位；对局中把座位交给 AI 打完。
pub fn leave_room(room: &Arc<Room>, viewer: Viewer) {
    let mut inner = room.inner_lock();
    match viewer {
        Viewer::Player(pid) => {
            if inner.session.phase == Phase::Lobby {
                if inner.session.players.len() > 1 {
                    inner.session.players.remove(pid);
                    if inner.host > pid {
                        inner.host -= 1;
                    } else if inner.host == pid {
                        inner.host = 0;
                    }
                }
            } else if !inner.session.players[pid].is_ai {
                let name = inner.session.players[pid].name.clone();
                let p = &mut inner.session.players[pid];
                p.is_ai = true;
                p.bot_id = "challenger".into();
                p.token = None;
                inner.session.log_public(
                    crate::game::LogKind::Info,
                    format!("{name} 离开，座位由 AI 接管"),
                );
            }
        }
        Viewer::Spectator(i) => {
            inner.spectators.remove(i);
        }
    }
    inner.last_active = Instant::now();
    drop(inner);
    room.bump();
}

/// 当前是否轮到某个 AI 座位行动；是则返回 (座位, Bot, 该座位视角)。
fn ai_job(inner: &RoomInner, bots: &BotRegistry) -> Option<(PlayerId, Arc<dyn Bot>, PlayerView)> {
    match &inner.session.phase {
        Phase::Peeking { done } => {
            let pid = (0..inner.session.players.len())
                .find(|&i| inner.session.players[i].is_ai && !done.contains_key(&i))?;
            let bot = bots.get(&inner.session.players[pid].bot_id)?;
            let view = project(&inner.session, Some(pid), inner.host);
            Some((pid, bot, view))
        }
        Phase::Turn { current, .. } => {
            let pid = *current;
            if !inner.session.players[pid].is_ai {
                return None;
            }
            let bot = bots.get(&inner.session.players[pid].bot_id)?;
            let view = project(&inner.session, Some(pid), inner.host);
            Some((pid, bot, view))
        }
        _ => None,
    }
}

/// 每个房间的 AI 驱动循环。
async fn ai_loop(room: Arc<Room>, bots: BotRegistry) {
    let mut rx = room.version.subscribe();
    loop {
        // 只要还有 AI 座位需要行动就持续驱动（每次行动之间留出思考时间）。
        loop {
            let pending = {
                let inner = room.inner_lock();
                ai_job(&inner, &bots).map(|(pid, _, _)| pid)
            };
            if pending.is_none() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(AI_THINK_MS)).await;
            // 睡眠期间状态可能变化（真人行动 / 房主调整），重新取视图与命令。
            let job = {
                let inner = room.inner_lock();
                ai_job(&inner, &bots)
            };
            let Some((pid, bot, view)) = job else { break };
            let mut err_count = 0;
            let result = loop {
                let cmd = if err_count == 0 {
                    bot.decide(&view, &mut rand::rng())
                } else {
                    fallback_command(&view)
                };
                let mut inner = room.inner_lock();
                break match inner.session.apply(pid, &cmd) {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        err_count += 1;
                        if err_count >= 3 {
                            tracing::error!("房间 {} AI 座位 {pid} 连续非法决策：{e}（{cmd:?}）", room.code);
                            Err(e)
                        } else {
                            continue;
                        }
                    }
                };
            };
            if result.is_ok() {
                room.bump();
            } else {
                // AI 彻底无法行动：跳过它（换下家），避免卡死牌局。
                let mut inner = room.inner_lock();
                let ai_name = inner.session.players[pid].name.clone();
                inner.session.log_public(
                    crate::game::LogKind::Info,
                    format!("AI 座位 {ai_name} 无法行动，跳过"),
                );
                // 引擎没有"跳过"原语：用兜底命令再试一次（BeginDraw/BeginTake 必然合法）。
                let cmd = fallback_command(&project(&inner.session, Some(pid), inner.host));
                match inner.session.apply(pid, &cmd) {
                    Ok(()) => {
                        drop(inner);
                        room.bump();
                    }
                    Err(_) => {
                        tracing::error!("房间 {} AI 座位 {pid} 兜底也失败，牌局可能停滞", room.code);
                    }
                }
            }
        }
        if rx.changed().await.is_err() {
            return; // 房间已回收
        }
    }
}

/// 应用一条玩家命令；无论成败都更新版本号（失败也推送，让操作者看到错误提示）。
pub fn apply_command(room: &Arc<Room>, viewer: Viewer, cmd: Command) -> Result<(), String> {
    let pid = match viewer {
        Viewer::Player(i) => i,
        Viewer::Spectator(_) => return Err("观战者不能操作".into()),
    };
    let mut inner = room.inner_lock();
    if matches!(inner.last_error, Some((p, _)) if p == pid) {
        inner.last_error = None;
    }
    inner.last_active = Instant::now();
    let result = inner.session.apply(pid, &cmd);
    if let Err(e) = &result {
        inner.last_error = Some((pid, e.to_string()));
    }
    drop(inner);
    room.bump();
    result.map_err(|e| e.to_string())
}

/// 房主操作（不涉及引擎命令）。
pub fn apply_host_op(
    room: &Arc<Room>,
    bots: &BotRegistry,
    viewer: Viewer,
    op: HostOp,
) -> Result<(), String> {
    let pid = match viewer {
        Viewer::Player(i) => i,
        Viewer::Spectator(_) => return Err("观战者不能操作".into()),
    };
    let mut inner = room.inner_lock();
    if matches!(inner.last_error, Some((p, _)) if p == pid) {
        inner.last_error = None;
    }
    inner.last_active = Instant::now();
    let result = (|| -> Result<(), String> {
        if pid != inner.host {
            return Err("只有房主可以执行该操作".into());
        }
        match op {
            HostOp::AddAI { bot_id } => {
                if inner.session.phase != Phase::Lobby {
                    return Err("游戏进行中不能添加 AI".into());
                }
                if inner.session.players.len() >= 4 {
                    return Err("座位已满".into());
                }
                let bot_id = if bots.get(&bot_id).is_some() {
                    bot_id
                } else {
                    bots.default_id().to_string()
                };
                let ai_name = next_ai_name(&inner.session);
                inner.session.players.push(PlayerState {
                    name: ai_name,
                    is_ai: true,
                    bot_id,
                    token: None,
                    slots: Vec::new(),
                    peeked_slots: Vec::new(),
                    total_score: 0,
                    round_score: None,
                });
            }
            HostOp::SeatToAI { seat } => {
                let taken = {
                    let p = inner.session.players.get_mut(seat).ok_or("座位不存在")?;
                    if p.is_ai {
                        return Err("该座位已经是 AI".into());
                    }
                    let tok = p.token.take();
                    p.is_ai = true;
                    p.bot_id = bots.default_id().into();
                    let name = p.name.clone();
                    (tok, name)
                };
                if let Some(tok) = taken.0 {
                    if inner.spectators.len() < 8 {
                        inner.spectators.push(Spectator { token: tok, name: taken.1.clone() });
                    }
                }
                inner.session.log_public(
                    crate::game::LogKind::Info,
                    format!("{} 的座位切换为 AI 控制", taken.1),
                );
            }
            HostOp::RemoveSeat { seat } => {
                if inner.session.phase != Phase::Lobby {
                    return Err("游戏进行中不能移除座位".into());
                }
                if inner.session.players.len() <= 2 {
                    return Err("至少保留 2 名玩家".into());
                }
                if seat >= inner.session.players.len() {
                    return Err("座位不存在".into());
                }
                let removed = inner.session.players.remove(seat);
                if inner.host >= seat && inner.host > 0 {
                    inner.host -= 1;
                }
                let name = removed.name.clone();
                let _ = name;
                // 该玩家令牌失效：下次请求会被视为游客。
            }
            HostOp::Start => inner.session.start_game().map_err(|e| e.to_string())?,
            HostOp::NextRound => inner.session.next_round().map_err(|e| e.to_string())?,
            HostOp::Rematch => inner.session.rematch().map_err(|e| e.to_string())?,
            HostOp::Settings { penalty, target, memory } => {
                if inner.session.phase != Phase::Lobby {
                    return Err("游戏进行中不能修改设置".into());
                }
                if !(0..=50).contains(&penalty) || !(10..=500).contains(&target) {
                    return Err("参数超出范围（惩罚 0~50，阈值 10~500）".into());
                }
                inner.session.settings.cabo_penalty = penalty;
                inner.session.settings.target_score = target;
                inner.session.settings.memory_mode = memory;
            }
        }
        Ok(())
    })();
    if let Err(e) = &result {
        if let Viewer::Player(p) = viewer {
            inner.last_error = Some((p, e.clone()));
        }
    }
    drop(inner);
    room.bump();
    result
}

pub enum HostOp {
    AddAI { bot_id: String },
    SeatToAI { seat: usize },
    RemoveSeat { seat: usize },
    Start,
    NextRound,
    Rematch,
    Settings { penalty: u32, target: u32, memory: bool },
}

fn next_ai_name(session: &Session) -> String {
    let n = session.players.iter().filter(|p| p.is_ai).count() + 1;
    format!("AI-{n}")
}

/// 后台回收闲置房间。
pub async fn reaper(shared: Arc<Shared>) {
    let mut itv = tokio::time::interval(Duration::from_secs(120));
    loop {
        itv.tick().await;
        let mut rooms = shared.rooms.lock().unwrap();
        rooms.retain(|code, room| {
            let keep = room.conns.load(Ordering::Relaxed) > 0
                || room.inner_lock().last_active.elapsed() < ROOM_IDLE;
            if !keep {
                tracing::info!("回收闲置房间 {code}");
            }
            keep
        });
    }
}
