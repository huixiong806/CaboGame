//! 视图 → HTML：askama 模板结构与 fragment 渲染。
//!
//! 页面骨架只渲染一次；之后所有变化都通过 SSE 推送"board fragment"
//! （每个连接按自己的视角独立渲染，互不可见他人的暗牌信息）。

use askama::Template;

use super::{RoomInner, Viewer};
use crate::ai::BotRegistry;
use crate::game::view::{project, PlayerView};
use crate::game::Phase;

#[derive(Template)]
#[template(path = "index.html")]
pub struct IndexPage {
    pub join_code: String,
    pub penalty: u32,
    pub target: u32,
}

#[derive(Template)]
#[template(path = "room.html")]
pub struct RoomPage {
    pub code: String,
    pub token: String,
    pub board: String,
    pub penalty: u32,
    pub target: u32,
}

#[derive(Template)]
#[template(path = "frag_lobby.html")]
pub struct LobbyFrag {
    pub code: String,
    pub token: String,
    pub is_host: bool,
    pub can_start: bool,
    pub me_name: String,
    pub players: Vec<LobbySeat>,
    /// 可选的 bot 列表 (id, 名称)，供房主下拉选择。
    pub bots: Vec<(String, String)>,
    pub penalty: u32,
    pub target: u32,
    pub memory_mode: bool,
    pub err: String,
}

pub struct LobbySeat {
    pub id: usize,
    pub name: String,
    pub is_ai: bool,
    pub is_host: bool,
    pub vacant: bool,
    pub is_me: bool,
}

#[derive(Template)]
#[template(path = "frag_game.html")]
pub struct GameFrag {
    pub code: String,
    pub token: String,
    pub v: PlayerView,
    pub err: String,
}

/// 渲染某个视角的 board fragment（大厅或牌局）。`bots` 仅大厅需要（添加 AI 下拉框）。
pub fn render_board(
    code: &str,
    token: &str,
    inner: &RoomInner,
    viewer: &Viewer,
    bots: Option<&BotRegistry>,
) -> String {
    let pid = match viewer {
        Viewer::Player(i) => Some(*i),
        Viewer::Spectator(_) => None,
    };
    let err = inner
        .last_error
        .as_ref()
        .filter(|(p, _)| Some(*p) == pid)
        .map(|(_, m)| m.clone())
        .unwrap_or_default();

    if inner.session.phase == Phase::Lobby {
        let view = project(&inner.session, pid, inner.host);
        let players = inner
            .session
            .players
            .iter()
            .enumerate()
            .map(|(i, p)| LobbySeat {
                id: i,
                name: p.name.clone(),
                is_ai: p.is_ai,
                is_host: i == inner.host,
                vacant: !p.is_ai && p.token.is_none(),
                is_me: pid == Some(i),
            })
            .collect();
        LobbyFrag {
            code: code.to_string(),
            token: token.to_string(),
            is_host: viewer == &Viewer::Player(inner.host),
            can_start: inner.session.seats_ready(),
            me_name: view.me_name.clone(),
            players,
            bots: bots
                .map(|r| r.list().iter().map(|(a, b)| (a.to_string(), b.to_string())).collect())
                .unwrap_or_default(),
            penalty: inner.session.settings.cabo_penalty,
            target: inner.session.settings.target_score,
            memory_mode: inner.session.settings.memory_mode,
            err,
        }
        .render()
        .unwrap_or_else(|e| format!("渲染失败：{e}"))
    } else {
        let view = project(&inner.session, pid, inner.host);
        GameFrag { code: code.to_string(), token: token.to_string(), v: view, err }
            .render()
            .unwrap_or_else(|e| format!("渲染失败：{e}"))
    }
}

/// 构造 SSE board 事件。axum 0.8 的 `Event::data` 只能调用一次（重复调用会 panic），
/// 但字符串中的换行会被自动拆成多条 `data:` 行，因此整段 HTML 一次性传入。
pub fn board_event(html: String) -> axum::response::sse::Event {
    use axum::response::sse::Event;
    let data = if html.is_empty() { " ".to_string() } else { html };
    Event::default().event("board").data(data)
}
