//! 按玩家视角投影牌局：只泄露该玩家有权知道的信息。
//!
//! 同一份 `Session`，不同玩家看到不同的 `PlayerView`——这是"已知者集合"
//! 公共知识模型的直接体现：点数只在 `revealed` 或自己在 `known_by` 中时可见，
//! 但"谁知道这张牌"（knower_count）对所有人可见。Bot 也只读取这个视图决策，
//! 从机制上杜绝作弊。
//!
//! 记忆模式（`settings.memory_mode`）只影响"显示"不影响知识本身：
//! 自己秘密查看过的点数不再常显，仅保留"已看"角标；公开信息（明置牌、
//! 全场皆知的牌）始终显示。Bot 始终读取完整的 `value`（它就是该座位的记忆）。

use super::{power_kind_of, Audience, Pending, Phase, PlayerId, Session, SlotId};

/// 点数分档配色（0~3 优、4~6 中、7~12 能力、13 高危）。
pub fn rank_class(rank: u8) -> &'static str {
    match rank {
        0..=3 => "card r-low",
        4..=6 => "card r-mid",
        7..=12 => "card r-power",
        _ => "card r-high",
    }
}

pub fn power_label(rank: u8) -> Option<&'static str> {
    Some(match power_kind_of(rank)? {
        super::PowerKind::Peek => "偷看",
        super::PowerKind::Spy => "间谍",
        super::PowerKind::Swap => "交换",
    })
}

#[derive(Clone, Debug)]
pub struct SlotView {
    pub slot: SlotId,
    /// 该玩家知道的点数（None = 对他保密）。Bot 决策使用，不受记忆模式影响。
    pub value: Option<u8>,
    /// UI 是否显示点数（公开信息常显；私有知识在记忆模式下隐藏）。
    pub shown: bool,
    pub revealed: bool,
    /// 我是否知道这张牌（用于交互可用性）。
    pub i_know: bool,
    /// 知道这张牌的人数（公共知识，人人都可见）。
    pub knower_count: usize,
    /// Exact public knowledge membership. This exposes no additional rank information.
    pub known_by: Vec<PlayerId>,
    /// 是否被当前玩家点选（初始查看 / 交换选择暂存）。
    pub selected: bool,
}

impl SlotView {
    pub fn css_class(&self) -> String {
        match (self.shown, self.value) {
            (true, Some(r)) => format!("{} {}", rank_class(r), rank_class_extra(r)),
            _ => "card back".to_string(),
        }
    }
    pub fn rank_text(&self) -> String {
        match (self.shown, self.value) {
            (true, Some(r)) => r.to_string(),
            _ => "?".to_string(),
        }
    }
    pub fn is_face(&self) -> bool {
        self.shown && self.value.is_some()
    }
    pub fn power_text(&self) -> String {
        match (self.shown, self.value) {
            (true, Some(r)) => power_label(r).unwrap_or("").to_string(),
            _ => String::new(),
        }
    }
    /// 我看不到、但别人可能知道的牌 → 显示"👁n"角标；
    /// 记忆模式下自己已知但被隐藏的牌也用它充当"已看"标记。
    pub fn knows_badge(&self) -> bool {
        !self.shown && self.knower_count > 0
    }
}

fn rank_class_extra(rank: u8) -> &'static str {
    match power_kind_of(rank) {
        Some(_) => "has-power",
        None => "",
    }
}

#[derive(Clone, Debug)]
pub struct SeatView {
    pub id: PlayerId,
    pub name: String,
    pub is_ai: bool,
    pub is_host: bool,
    pub vacant: bool,
    pub slots: Vec<SlotView>,
    pub total_score: u32,
    pub round_score: Option<u32>,
    pub is_current: bool,
    pub is_me: bool,
    pub is_caller: bool,
}

impl SeatView {
    pub fn round_score_text(&self) -> String {
        self.round_score.map(|s| s.to_string()).unwrap_or_else(|| "—".into())
    }
}

/// 行动面板状态（强类型，供 Bot 决策使用）。
#[derive(Clone, Debug)]
pub enum Panel {
    /// 初始查看：自己还没选（staged 为已点选的槽位）。
    PeekPick { staged: Vec<SlotId> },
    /// 初始查看：已提交，等待其他人。
    PeekWait,
    /// 我的回合，空闲。
    Idle { can_draw: bool, can_swap_discard: bool, can_cabo: bool },
    /// 已摸牌待弃置（rank 仅自己可见，能力可选）。
    Drew { rank: u8, power: Option<&'static str> },
    /// 能力瞄准中。
    AimPeek,
    AimSpy,
    AimSwap { my_slot: Option<SlotId> },
    /// 交换：已获得牌，选择 1~N 张手牌。
    SwapSelecting { source: super::Pile, drawn: Option<u8>, count: usize },
    /// 确认宣告 Cabo。
    ConfirmCabo,
    /// 其他人的回合 / 等待中。
    Waiting { text: String },
    /// 本轮结算。
    RoundEnd { is_host: bool },
    /// 游戏结束。
    GameOver { winners: Vec<String>, is_host: bool },
    /// 观战 / 大厅。
    Spectating,
}

/// 模板友好的扁平面板信息（askama 模板里只做 if/for，不处理枚举与 Option）。
#[derive(Clone, Debug)]
pub struct PanelInfo {
    pub is_peek_pick: bool,
    pub staged: Vec<SlotId>,
    pub is_peek_wait: bool,
    pub is_idle: bool,
    pub can_draw: bool,
    pub can_swap_discard: bool,
    pub can_cabo: bool,
    pub is_drew: bool,
    pub drawn_rank: String,
    pub drawn_power: String,
    pub arm_cmd: String,
    pub is_aim_peek: bool,
    pub is_aim_spy: bool,
    pub is_aim_swap: bool,
    pub aim_has_my_slot: bool,
    pub aim_my_slot: SlotId,
    pub is_swap_sel: bool,
    pub swap_source_draw: bool,
    pub swap_drawn_rank: String,
    pub swap_count: usize,
    pub swap_can_cancel: bool,
    pub is_confirm: bool,
    pub is_waiting: bool,
    pub waiting_text: String,
    pub is_round_end: bool,
    pub is_game_over: bool,
    pub winners: Vec<String>,
    pub is_spectating: bool,
    /// 我的牌当前点击提交的命令（"" = 不可点）。
    pub own_click_cmd: String,
    /// 对手的牌当前点击提交的命令（"" = 不可点）。
    pub other_click_cmd: String,
}

impl From<&Panel> for PanelInfo {
    fn from(p: &Panel) -> Self {
        let mut info = PanelInfo {
            is_peek_pick: false,
            staged: Vec::new(),
            is_peek_wait: false,
            is_idle: false,
            can_draw: false,
            can_swap_discard: false,
            can_cabo: false,
            is_drew: false,
            drawn_rank: String::new(),
            drawn_power: String::new(),
            arm_cmd: String::new(),
            is_aim_peek: false,
            is_aim_spy: false,
            is_aim_swap: false,
            aim_has_my_slot: false,
            aim_my_slot: 0,
            is_swap_sel: false,
            swap_source_draw: false,
            swap_drawn_rank: String::new(),
            swap_count: 0,
            swap_can_cancel: false,
            is_confirm: false,
            is_waiting: false,
            waiting_text: String::new(),
            is_round_end: false,
            is_game_over: false,
            winners: Vec::new(),
            is_spectating: false,
            own_click_cmd: String::new(),
            other_click_cmd: String::new(),
        };
        match p {
            Panel::PeekPick { staged } => {
                info.is_peek_pick = true;
                info.staged = staged.clone();
                info.own_click_cmd = "peek_toggle".into();
            }
            Panel::PeekWait => info.is_peek_wait = true,
            Panel::Idle { can_draw, can_swap_discard, can_cabo } => {
                info.is_idle = true;
                info.can_draw = *can_draw;
                info.can_swap_discard = *can_swap_discard;
                info.can_cabo = *can_cabo;
            }
            Panel::Drew { rank, power } => {
                info.is_drew = true;
                info.drawn_rank = rank.to_string();
                info.drawn_power = power.unwrap_or("").to_string();
                info.arm_cmd = match power {
                    Some("偷看") => "arm_peek".into(),
                    Some("间谍") => "arm_spy".into(),
                    Some("交换") => "arm_swap".into(),
                    _ => String::new(),
                };
                // 桌面直点：点一张手牌 = 用刚摸的牌与它交换（单张，必成功）。
                info.own_click_cmd = "draw_swap".into();
            }
            Panel::AimPeek => {
                info.is_aim_peek = true;
                info.own_click_cmd = "pick_own".into();
            }
            Panel::AimSpy => {
                info.is_aim_spy = true;
                info.other_click_cmd = "pick_other".into();
            }
            Panel::AimSwap { my_slot } => {
                info.is_aim_swap = true;
                match my_slot {
                    Some(s) => {
                        info.aim_has_my_slot = true;
                        info.aim_my_slot = *s;
                        info.other_click_cmd = "pick_other".into();
                    }
                    None => info.own_click_cmd = "pick_own".into(),
                }
            }
            Panel::SwapSelecting { source, drawn, count } => {
                info.is_swap_sel = true;
                info.swap_source_draw = matches!(source, super::Pile::Draw);
                info.swap_drawn_rank = drawn.map(|r| r.to_string()).unwrap_or_default();
                info.swap_count = *count;
                info.swap_can_cancel = !info.swap_source_draw;
                info.own_click_cmd = "swap_toggle".into();
            }
            Panel::ConfirmCabo => info.is_confirm = true,
            Panel::Waiting { text } => {
                info.is_waiting = true;
                info.waiting_text = text.clone();
            }
            Panel::RoundEnd { is_host: _ } => info.is_round_end = true,
            Panel::GameOver { winners, is_host: _ } => {
                info.is_game_over = true;
                info.winners = winners.clone();
            }
            Panel::Spectating => info.is_spectating = true,
        }
        info
    }
}

#[derive(Clone, Debug)]
pub struct LogLine {
    pub kind: super::LogKind,
    pub text: String,
}

impl LogLine {
    pub fn kind_str(&self) -> &'static str {
        match self.kind {
            super::LogKind::Info => "info",
            super::LogKind::Peek => "peek",
            super::LogKind::Swap => "swap",
            super::LogKind::Power => "power",
            super::LogKind::Cabo => "cabo",
            super::LogKind::Score => "score",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlayerView {
    pub round_no: u32,
    pub me: Option<PlayerId>,
    pub me_name: String,
    pub is_host: bool,
    /// 其他座位（不含自己），按座位顺序。
    pub opponents: Vec<SeatView>,
    /// 包含自己的全部座位（结算表格用）。
    pub all_seats: Vec<SeatView>,
    /// 自己的座位（观战为 None）。
    pub me_seat: Option<SeatView>,
    pub deck_count: usize,
    pub discard_top: Option<u8>,
    pub discard_count: usize,
    /// All cards in this public pile, bottom to top, as a perfect public memory.
    pub discard_ranks: Vec<u8>,
    pub public_events: Vec<super::PublicEvent>,
    /// 我抽到尚未弃置的牌（行动 A，仅自己可见其点数）。
    pub drawn: Option<u8>,
    pub panel: Panel,
    pub panel_info: PanelInfo,
    pub cabo_caller: Option<String>,
    pub extra_left: usize,
    /// 记忆模式下的"最近查看"闪显（仅自己可见，下一次行动前有效；空串 = 无）。
    pub peek_flash: String,
    pub memory_mode: bool,
    /// 日志（按受众过滤，最新在前）。
    pub log: Vec<LogLine>,
    pub cabo_penalty: u32,
    pub target_score: u32,
}

impl PlayerView {
    pub fn is_spectator(&self) -> bool {
        self.me.is_none()
    }
    pub fn discard_top_text(&self) -> String {
        self.discard_top.map(|r| r.to_string()).unwrap_or_default()
    }
    pub fn discard_css(&self) -> String {
        match self.discard_top {
            Some(r) => format!("{} {}", rank_class(r), rank_class_extra(r)),
            None => "card empty".into(),
        }
    }
    pub fn discard_power_text(&self) -> String {
        self.discard_top.and_then(power_label).unwrap_or("").to_string()
    }
    pub fn drawn_text(&self) -> String {
        self.drawn.map(|r| r.to_string()).unwrap_or_default()
    }
    pub fn drawn_css(&self) -> String {
        match self.drawn {
            Some(r) => format!("{} {}", rank_class(r), rank_class_extra(r)),
            None => "card empty".into(),
        }
    }
    pub fn drawn_power_text(&self) -> String {
        self.drawn.and_then(power_label).unwrap_or("").to_string()
    }
    pub fn cabo_text(&self) -> String {
        self.cabo_caller.clone().unwrap_or_default()
    }
    /// 我当前拥有的牌数。
    pub fn my_slot_count(&self) -> usize {
        self.me_seat.as_ref().map(|s| s.slots.len()).unwrap_or(0)
    }
}

/// 构建某视角的画面数据。`viewer = None` 表示观战。
pub fn project(session: &Session, viewer: Option<PlayerId>, host: PlayerId) -> PlayerView {
    let n = session.players.len();
    let names: Vec<String> = session.players.iter().map(|p| p.name.clone()).collect();

    // 我的槽位选择暂存（初始查看 / 交换选择）。
    let my_selected: Vec<SlotId> = match (&session.phase, viewer) {
        (Phase::Peeking { .. }, Some(v)) => session.players[v].peeked_slots.clone(),
        (
            Phase::Turn {
                current,
                pending: Some(Pending::SwapSelecting { selected, .. }),
            },
            Some(v),
        ) if *current == v => selected.clone(),
        _ => Vec::new(),
    };

    let seat_view = |pid: PlayerId, is_me: bool| SeatView {
        id: pid,
        name: names[pid].clone(),
        is_ai: session.players[pid].is_ai,
        is_host: pid == host,
        vacant: !session.players[pid].is_ai && session.players[pid].token.is_none(),
        slots: (0..session.players[pid].slots.len() as SlotId)
            .map(|s| {
                let cid = session.players[pid].slots[s as usize];
                let cs = &session.cards[cid as usize];
                let i_know = cs.is_known_to(viewer.unwrap_or(PlayerId::MAX));
                let value = if i_know { Some(cs.card.rank) } else { None };
                let revealed = cs.revealed;
                let knower_count = cs.known_by.len();
                // 显示规则：公开信息（明置 / 全场皆知）常显；
                // 私有知识仅在显示模式常显，记忆模式下隐藏（保留角标）。
                let shown = match value {
                    None => false,
                    Some(_) => {
                        revealed
                            || knower_count >= n
                            || !session.settings.memory_mode
                    }
                };
                SlotView {
                    slot: s,
                    value,
                    shown,
                    revealed,
                    i_know,
                    knower_count,
                    known_by: cs.known_by.iter().copied().collect(),
                    selected: is_me && my_selected.contains(&s),
                }
            })
            .collect(),
        total_score: session.players[pid].total_score,
        round_score: session.players[pid].round_score,
        is_current: session.actor() == Some(pid),
        is_me,
        is_caller: session.cabo_caller == Some(pid),
    };

    let me_seat = viewer.map(|v| seat_view(v, true));
    let all_seats: Vec<SeatView> = (0..n).map(|i| seat_view(i, viewer == Some(i))).collect();
    let opponents: Vec<SeatView> =
        all_seats.iter().filter(|s| Some(s.id) != viewer).cloned().collect();

    // 面板状态。
    let panel = match (&session.phase, viewer) {
        (Phase::Lobby, _) => Panel::Spectating,
        (Phase::RoundEnd, v) => Panel::RoundEnd { is_host: v == Some(host) },
        (Phase::GameOver { winners }, v) => Panel::GameOver {
            winners: winners.iter().map(|&i| names[i].clone()).collect(),
            is_host: v == Some(host),
        },
        (Phase::Peeking { done }, Some(v)) => {
            if done.contains_key(&v) {
                Panel::PeekWait
            } else {
                Panel::PeekPick { staged: session.players[v].peeked_slots.clone() }
            }
        }
        (Phase::Peeking { .. }, None) => Panel::Spectating,
        (Phase::Turn { current, pending }, Some(v)) if *current == v => {
            panel_for_pending(session, pending.as_ref())
        }
        (Phase::Turn { current, .. }, viewer_opt) => {
            let whose = names[*current].clone();
            let text = match viewer_opt {
                Some(_) => format!("等待 {whose} 行动…"),
                None => format!("观战中：{whose} 正在行动"),
            };
            Panel::Waiting { text }
        }
    };

    // 日志：按受众过滤，最新在前，只保留最近 30 条。
    let log: Vec<LogLine> = session
        .log
        .iter()
        .filter(|e| match e.audience {
            Audience::Public => true,
            Audience::Player(p) => viewer == Some(p),
        })
        .rev()
        .take(30)
        .map(|e| LogLine { kind: e.kind, text: e.text.clone() })
        .collect();

    // 行动 A 摸到的牌（仅自己可见）与交换暂存中从摸牌堆获得的牌。
    let drawn = match (&session.phase, viewer) {
        (
            Phase::Turn {
                current,
                pending:
                    Some(Pending::Drew { card })
                    | Some(Pending::PowerAiming { card, .. })
                    | Some(Pending::SwapSelecting { card, source: super::Pile::Draw, .. }),
            },
            Some(v),
        ) if *current == v => Some(session.cards[*card as usize].card.rank),
        _ => None,
    };

    let peek_flash = match viewer {
        Some(v) => session.peek_flash.get(&v).cloned().unwrap_or_default(),
        None => String::new(),
    };

    let panel_info = PanelInfo::from(&panel);
    PlayerView {
        round_no: session.round_no,
        me: viewer,
        me_name: viewer.map(|v| names[v].clone()).unwrap_or_else(|| "观战".into()),
        is_host: viewer == Some(host),
        opponents,
        all_seats,
        me_seat,
        deck_count: session.deck.len(),
        discard_top: session.discard_top().map(|c| session.cards[c as usize].card.rank),
        discard_count: session.discard.len(),
        discard_ranks: session.discard.iter().map(|&c| session.cards[c as usize].card.rank).collect(),
        public_events: session.public_events.clone(),
        drawn,
        panel,
        panel_info,
        cabo_caller: session.cabo_caller.map(|c| names[c].clone()),
        extra_left: session.extra_turns.len(),
        peek_flash,
        memory_mode: session.settings.memory_mode,
        log,
        cabo_penalty: session.settings.cabo_penalty,
        target_score: session.settings.target_score,
    }
}

/// 我是自己回合时的行动面板。
fn panel_for_pending(session: &Session, pending: Option<&Pending>) -> Panel {
    match pending {
        None => {
            Panel::Idle {
                can_draw: !session.deck.is_empty(),
                can_swap_discard: session.discard_top().is_some(),
                can_cabo: session.cabo_caller.is_none(),
            }
        }
        Some(Pending::Drew { card }) => {
            let rank = session.cards[*card as usize].card.rank;
            Panel::Drew { rank, power: power_label(rank) }
        }
        Some(Pending::PowerAiming { kind, my_slot, .. }) => match kind {
            super::PowerKind::Peek => Panel::AimPeek,
            super::PowerKind::Spy => Panel::AimSpy,
            super::PowerKind::Swap => Panel::AimSwap { my_slot: *my_slot },
        },
        Some(Pending::SwapSelecting { card, source, selected }) => {
            let drawn = match source {
                super::Pile::Draw => Some(session.cards[*card as usize].card.rank),
                super::Pile::Discard => None,
            };
            Panel::SwapSelecting {
                source: *source,
                drawn,
                count: selected.len(),
            }
        }
        Some(Pending::ConfirmCabo) => Panel::ConfirmCabo,
    }
}
