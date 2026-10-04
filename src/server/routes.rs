//! HTTP 路由与处理器。

use std::sync::atomic::Ordering;
use std::sync::Arc;

use askama::Template;
use axum::extract::{Form, Path, Query, State};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use serde::Deserialize;

use super::render::{board_event, render_board, IndexPage, RoomPage};
use super::{
    apply_command, apply_host_op, create_room, join_room, leave_room, resolve_viewer, HostOp,
    Room, Shared, Viewer,
};
use crate::ai::BotRegistry;
use crate::game::{Command, PowerKind, Settings, DEFAULT_PENALTY, DEFAULT_TARGET};

pub type AppState = Arc<Shared>;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/htmx.min.js", get(serve_htmx))
        .route("/static/ext-sse.js", get(serve_sse_ext))
        .route("/static/style.css", get(serve_css))
        .route("/static/room.js", get(serve_room_js))
        .route("/create", post(create))
        .route("/join", post(join))
        .route("/room/{code}", get(room_page))
        .route("/room/{code}/events", get(sse))
        .route("/room/{code}/cmd", post(cmd))
        .route("/room/{code}/host", post(host))
        .route("/room/{code}/leave", post(leave))
        .with_state(state)
}

// ---------------------------------------------------------------- 静态资源

async fn serve_htmx() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "application/javascript")],
        include_str!("../../static/htmx.min.js"),
    )
}

async fn serve_sse_ext() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "application/javascript")],
        include_str!("../../static/ext-sse.js"),
    )
}

async fn serve_css() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../../static/style.css"),
    )
}

async fn serve_room_js() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "application/javascript; charset=utf-8")],
        include_str!("../../static/room.js"),
    )
}

// ---------------------------------------------------------------- 首页 / 创建 / 加入

#[derive(Deserialize)]
struct IndexQuery {
    #[serde(default)]
    code: String,
}

async fn index(Query(q): Query<IndexQuery>) -> Html<String> {
    Html(IndexPage {
        join_code: q.code.to_uppercase(),
        penalty: DEFAULT_PENALTY,
        target: DEFAULT_TARGET,
    }
    .render()
    .unwrap())
}

#[derive(Deserialize)]
struct CreateForm {
    #[serde(default)]
    name: String,
    #[serde(default)]
    penalty: Option<u32>,
    #[serde(default)]
    target: Option<u32>,
}

async fn create(State(state): State<AppState>, Form(f): Form<CreateForm>) -> Response {
    let Some(name) = sanitize_name(&f.name) else {
        return Redirect::to("/?error=name").into_response();
    };
    let settings = Settings {
        cabo_penalty: f.penalty.unwrap_or(DEFAULT_PENALTY),
        target_score: f.target.unwrap_or(DEFAULT_TARGET),
        memory_mode: false,
    };
    let (room, token) = create_room(&state, name, settings);
    Redirect::to(&format!("/room/{}?t={token}", room.code)).into_response()
}

#[derive(Deserialize)]
struct JoinForm {
    #[serde(default)]
    code: String,
    #[serde(default)]
    name: String,
}

async fn join(State(state): State<AppState>, Form(f): Form<JoinForm>) -> Response {
    let code = f.code.trim().to_uppercase();
    let Some(name) = sanitize_name(&f.name) else {
        return Redirect::to(&format!("/?code={code}&error=name")).into_response();
    };
    match join_room(&state, &code, name) {
        Ok((room, token)) => Redirect::to(&format!("/room/{}?t={token}", room.code)).into_response(),
        Err(_) => Redirect::to(&format!("/?code={code}&error=room")).into_response(),
    }
}

fn sanitize_name(raw: &str) -> Option<String> {
    let name = raw.trim();
    let n = name.chars().count();
    if n == 0 || n > 12 || name.chars().any(|c| c.is_control()) {
        return None;
    }
    Some(name.to_string())
}

// ---------------------------------------------------------------- 房间页 / SSE

#[derive(Deserialize)]
struct TokenQuery {
    #[serde(default)]
    t: String,
}

async fn room_page(
    State(state): State<AppState>,
    Path(code): Path<String>,
    Query(q): Query<TokenQuery>,
) -> Response {
    let Some(room) = state.rooms.lock().unwrap().get(&code).cloned() else {
        return Redirect::to("/?error=room").into_response();
    };
    let (board, settings) = {
        let inner = room.inner_lock();
        let Some(viewer) = resolve_viewer(&inner, &q.t) else {
            drop(inner);
            return Redirect::to(&format!("/?code={code}&error=token")).into_response();
        };
        let board = render_board(&code, &q.t, &inner, &viewer, Some(&state.bots));
        (board, inner.session.settings.clone())
    };
    Html(RoomPage {
        code,
        token: q.t,
        board,
        penalty: settings.cabo_penalty,
        target: settings.target_score,
    }
    .render()
    .unwrap())
    .into_response()
}

/// SSE 流：连接即推一次当前画面，之后每次版本变化各推一次。
/// 客户端断开时流被丢弃，`SseCtx::drop` 负责递减房间连接计数。
async fn sse(
    State(state): State<AppState>,
    Path(code): Path<String>,
    Query(q): Query<TokenQuery>,
) -> Response {
    let Some(room) = state.rooms.lock().unwrap().get(&code).cloned() else {
        return (axum::http::StatusCode::NOT_FOUND, "房间不存在").into_response();
    };
    let viewer = {
        let inner = room.inner_lock();
        resolve_viewer(&inner, &q.t)
    };
    let Some(viewer) = viewer else {
        return (axum::http::StatusCode::FORBIDDEN, "无效的令牌").into_response();
    };
    room.conns.fetch_add(1, Ordering::Relaxed);

    let ctx = SseCtx {
        room,
        viewer,
        code,
        token: q.t,
        bots: state.bots.clone(),
    };
    let rx = ctx.room.version.subscribe();
    let stream = tokio_stream::wrappers::WatchStream::new(rx).map(
        move |_version| -> Result<axum::response::sse::Event, std::convert::Infallible> {
            let inner = ctx.room.inner_lock();
            let html = render_board(&ctx.code, &ctx.token, &inner, &ctx.viewer, Some(&ctx.bots));
            Ok(board_event(html))
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

/// SSE 连接上下文：随流一起存活，丢弃时递减连接计数。
struct SseCtx {
    room: Arc<Room>,
    viewer: Viewer,
    code: String,
    token: String,
    bots: BotRegistry,
}

impl Drop for SseCtx {
    fn drop(&mut self) {
        self.room.conns.fetch_sub(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------- 游戏命令

#[derive(Deserialize)]
struct CmdForm {
    cmd: String,
    #[serde(default)]
    slot: Option<u8>,
    #[serde(default)]
    player: Option<usize>,
}

fn parse_command(f: &CmdForm) -> Result<Command, &'static str> {
    Ok(match f.cmd.as_str() {
        "peek_toggle" => Command::PeekToggle { slot: f.slot.ok_or("缺少 slot")? },
        "cancel" => Command::Cancel,
        "draw" => Command::BeginDraw,
        "discard" => Command::DiscardDrawn { power: None },
        "arm_peek" => Command::ArmPower { kind: PowerKind::Peek },
        "arm_spy" => Command::ArmPower { kind: PowerKind::Spy },
        "arm_swap" => Command::ArmPower { kind: PowerKind::Swap },
        "pick_own" => Command::PowerPickOwn { slot: f.slot.ok_or("缺少 slot")? },
        "pick_other" => Command::PowerPickOther {
            player: f.player.ok_or("缺少 player")?,
            slot: f.slot.ok_or("缺少 slot")?,
        },
        "begin_swap" => Command::BeginSwap,
        "swap_toggle" => Command::SwapToggle { slot: f.slot.ok_or("缺少 slot")? },
        "swap_commit" => Command::SwapCommit,
        "draw_swap" => Command::DrawSwap {
            slots: f.slot.map(|s| vec![s]).unwrap_or_default(),
        },
        "cabo_arm" => Command::CallCaboArm,
        "cabo" => Command::CallCabo,
        _other => return Err("未知命令"),
    })
}

async fn cmd(
    State(state): State<AppState>,
    Path(code): Path<String>,
    Query(q): Query<TokenQuery>,
    Form(f): Form<CmdForm>,
) -> Response {
    let Some(room) = state.rooms.lock().unwrap().get(&code).cloned() else {
        return (axum::http::StatusCode::NOT_FOUND, "房间不存在").into_response();
    };
    let viewer = {
        let inner = room.inner_lock();
        resolve_viewer(&inner, &q.t)
    };
    let Some(viewer) = viewer else {
        return (axum::http::StatusCode::FORBIDDEN, "无效的令牌").into_response();
    };
    let Ok(command) = parse_command(&f) else {
        return (axum::http::StatusCode::BAD_REQUEST, "无效的命令").into_response();
    };
    let _ = apply_command(&room, viewer, command);
    // 无论成败都返回 204：画面由 SSE 统一推送（失败时推送错误横幅）。
    axum::http::StatusCode::NO_CONTENT.into_response()
}

// ---------------------------------------------------------------- 房主操作 / 离开

#[derive(Deserialize)]
struct HostForm {
    action: String,
    #[serde(default)]
    seat: Option<usize>,
    #[serde(default)]
    bot: Option<String>,
    #[serde(default)]
    penalty: Option<u32>,
    #[serde(default)]
    target: Option<u32>,
    #[serde(default)]
    memory: Option<bool>,
}

async fn host(
    State(state): State<AppState>,
    Path(code): Path<String>,
    Query(q): Query<TokenQuery>,
    Form(f): Form<HostForm>,
) -> Response {
    let Some(room) = state.rooms.lock().unwrap().get(&code).cloned() else {
        return (axum::http::StatusCode::NOT_FOUND, "房间不存在").into_response();
    };
    let viewer = {
        let inner = room.inner_lock();
        resolve_viewer(&inner, &q.t)
    };
    let Some(viewer) = viewer else {
        return (axum::http::StatusCode::FORBIDDEN, "无效的令牌").into_response();
    };
    let op = match f.action.as_str() {
        "add_ai" => HostOp::AddAI { bot_id: f.bot.unwrap_or_default() },
        "to_ai" => HostOp::SeatToAI { seat: f.seat.unwrap_or(usize::MAX) },
        "remove" => HostOp::RemoveSeat { seat: f.seat.unwrap_or(usize::MAX) },
        "start" => HostOp::Start,
        "next_round" => HostOp::NextRound,
        "rematch" => HostOp::Rematch,
        "settings" => HostOp::Settings {
            penalty: f.penalty.unwrap_or(DEFAULT_PENALTY),
            target: f.target.unwrap_or(DEFAULT_TARGET),
            memory: f.memory.unwrap_or(false),
        },
        _ => return (axum::http::StatusCode::BAD_REQUEST, "无效操作").into_response(),
    };
    let _ = apply_host_op(&room, &state.bots, viewer, op);
    axum::http::StatusCode::NO_CONTENT.into_response()
}

async fn leave(
    State(state): State<AppState>,
    Path(code): Path<String>,
    Query(q): Query<TokenQuery>,
    Form(_f): Form<LeaveForm>,
) -> Response {
    let Some(room) = state.rooms.lock().unwrap().get(&code).cloned() else {
        return Redirect::to("/").into_response();
    };
    let viewer = {
        let inner = room.inner_lock();
        resolve_viewer(&inner, &q.t)
    };
    if let Some(viewer) = viewer {
        leave_room(&room, viewer);
    }
    Redirect::to("/").into_response()
}

#[derive(Deserialize)]
struct LeaveForm {}
