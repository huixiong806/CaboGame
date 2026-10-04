//! Cabo 规则引擎。
//!
//! 规则依据 `cabo rule.md`（软件需求规格说明，2026-10 修订版：统一交换行动）。
//! 核心建模：
//! - 每张牌维护：物理位置（由容器归属隐含）、可见性 `revealed`、已知者集合 `known_by`。
//! - `known_by` 及其变化过程是公共知识：所有玩家都知道"谁知道这张牌"，但只有集合内的玩家知道点数。
//! - 回合内行动 A（摸牌）/ B（交换）/ C（宣告 Cabo）互斥，每回合恰好执行一项。

pub mod engine;
pub mod sim;
pub mod view;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rand::rngs::StdRng;
use rand::SeedableRng;

pub type CardId = u16;
/// 玩家 = 座位下标（0..人数）。
pub type PlayerId = usize;
/// 手牌槽位下标。开局 4 个槽位；交换失败会追加槽位。
pub type SlotId = u8;

pub const INITIAL_SLOTS: usize = 4;
/// 每个点数的张数：0 有 2 张，1~12 各 4 张，13 有 2 张，共 52 张。
pub const RANK_COPIES: [usize; 14] = [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2];

/// 一张牌（点数即分值；花色无关规则，仅用于展示配色，见 view 层）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Card {
    pub id: CardId,
    pub rank: u8,
}

/// 牌的运行时状态：位置由容器归属隐含，这里维护可见性与已知者集合。
#[derive(Clone, Debug)]
pub struct CardState {
    pub card: Card,
    pub revealed: bool,
    pub known_by: BTreeSet<PlayerId>,
}

impl CardState {
    /// 玩家 `p` 是否知道这张牌的点数（公开牌人人皆知）。
    pub fn is_known_to(&self, p: PlayerId) -> bool {
        self.revealed || self.known_by.contains(&p)
    }
}

/// 点数 7~12 的能力种类。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerKind {
    /// 7/8：偷看自己的一张牌。
    Peek,
    /// 9/10：间谍，看别人的一张牌。
    Spy,
    /// 11/12：交换，自己一张牌与别人一张牌盲换。
    Swap,
}

pub fn power_kind_of(rank: u8) -> Option<PowerKind> {
    match rank {
        7..=8 => Some(PowerKind::Peek),
        9..=10 => Some(PowerKind::Spy),
        11..=12 => Some(PowerKind::Swap),
        _ => None,
    }
}

/// 能力的具体目标（在行动 A 弃置时随命令一次性提交）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PowerUse {
    PeekOwn { slot: SlotId },
    Spy { player: PlayerId, slot: SlotId },
    Swap { my_slot: SlotId, player: PlayerId, slot: SlotId },
}

/// 换入来源：弃牌堆顶 / 摸牌堆顶（交换行动用）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pile {
    Draw,
    Discard,
}

/// 玩家可执行的一条命令。人类 UI 逐步操作，AI 可用组合命令一步到位。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// 初始阶段：秘密查看自己的两张牌（两个不同且已有牌的槽位）。
    PeekInitial { slots: [SlotId; 2] },
    /// 初始阶段（人类 UI）：点选槽位；凑满 2 张自动提交。
    PeekToggle { slot: SlotId },
    /// 行动 A 第一步：从摸牌堆抽一张牌并秘密查看（此步之后不可取消）。
    BeginDraw,
    /// 行动 A：弃置刚抽的牌；能力牌可选发动能力。
    DiscardDrawn { power: Option<PowerUse> },
    /// 行动 A（摸牌后）：把刚抽的牌与 1~N 张手牌交换。
    /// `slots` 为空 = 进入逐步选择模式（人类多张交换）；非空 = 选定这些槽位并立即结算。
    DrawSwap { slots: Vec<SlotId> },
    /// 行动 A（人类 UI 逐步）：宣布要发动能力，进入选目标状态。
    ArmPower { kind: PowerKind },
    /// 行动 A（人类 UI 逐步）：选择自己槽位（偷看的目标 / 交换的己方一端）。
    PowerPickOwn { slot: SlotId },
    /// 行动 A（人类 UI 逐步）：选择别人槽位（间谍的目标 / 交换的对方一端）。
    PowerPickOther { player: PlayerId, slot: SlotId },
    /// 行动 B：拿弃牌堆顶的牌（公开信息，确认前可取消），与 1~N 张手牌交换。
    BeginSwap,
    /// 行动 B（人类 UI 逐步）：把一张手牌加入/移出交换选择。
    SwapToggle { slot: SlotId },
    /// 行动 B：确认交换。单张必成功；多张同点数成功；否则失败（明置 + 追加新牌）。
    /// 手牌始终保持紧凑：换走的牌移除、获得的牌追加到末尾。
    SwapCommit,
    /// 组合命令（AI/测试用）：拿弃牌堆顶并与 `slots` 交换，一步到位。
    SwapOnce { slots: Vec<SlotId> },
    /// 行动 C 第一步：确认宣告 Cabo（两步确认防误触）。
    CallCaboArm,
    /// 行动 C：宣告 Cabo，进入终局加时阶段。
    CallCabo,
    /// 取消当前暂存（仅限不涉密信息的暂存：交换-弃牌堆来源、配对选择、能力瞄准、Cabo 确认）。
    Cancel,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum GameError {
    #[error("当前阶段不能进行该操作")]
    WrongPhase,
    #[error("还没有轮到你")]
    NotYourTurn,
    #[error("{0}")]
    Invalid(&'static str),
}

/// 可配置参数。
pub const DEFAULT_PENALTY: u32 = 10;
pub const DEFAULT_TARGET: u32 = 100;

#[derive(Clone, Debug)]
pub struct Settings {
    /// Cabo 喊错惩罚分。
    pub cabo_penalty: u32,
    /// 游戏结束阈值：累计总分达到该值的玩家触发生局结算。
    pub target_score: u32,
    /// 记忆模式：true 时 UI 不常显自己秘密查看过的点数（仅影响显示，不影响规则与 K）。
    pub memory_mode: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { cabo_penalty: DEFAULT_PENALTY, target_score: DEFAULT_TARGET, memory_mode: false }
    }
}

/// 座位上的一名玩家（AI 或真人）。
#[derive(Clone, Debug)]
pub struct PlayerState {
    pub name: String,
    pub is_ai: bool,
    /// 该座位使用的 Bot id（AI 座位有效）。
    pub bot_id: String,
    /// 真人玩家的会话令牌；AI / 空座位为 None。
    pub token: Option<String>,
    /// 手牌（开局 4 张；交换后保持紧凑无空位，新牌追加到末尾；失败追加新牌可超过 4 张）。
    pub slots: Vec<CardId>,
    /// 初始查看阶段的人类 UI 暂存（点选中的槽位，凑满 2 个自动提交）。
    pub peeked_slots: Vec<SlotId>,
    /// 累计总分。
    pub total_score: u32,
    /// 本轮得分（回合结束后填写，用于结算展示）。
    pub round_score: Option<u32>,
}

impl PlayerState {
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

/// 回合内的暂存状态（等待当前玩家继续选择）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pending {
    /// 已从摸牌堆抽牌（仅本人可见其点数），必须弃置（可发动能力）。
    Drew { card: CardId },
    /// 宣布发动能力，正在选目标。
    PowerAiming { card: CardId, kind: PowerKind, my_slot: Option<SlotId> },
    /// 交换：已获得一张牌，正在选择要交换的手牌。
    /// 摸牌堆来源的牌已抽出（不可取消）；弃牌堆来源仅暂存堆顶 id（可取消）。
    SwapSelecting { card: CardId, source: Pile, selected: Vec<SlotId> },
    /// 二次确认宣告 Cabo。
    ConfirmCabo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    /// 大厅：等待玩家、配置、开始。
    Lobby,
    /// 开局：每人秘密查看自己 2 张牌；全部提交后统一公开并随机确定起始玩家。
    /// `done` 记录已提交的玩家及其选择（对其他人保密，直到全员提交）。
    Peeking { done: BTreeMap<PlayerId, [SlotId; 2]> },
    /// 轮到某玩家行动。
    Turn { current: PlayerId, pending: Option<Pending> },
    /// 本轮结束，展示结算，等待房主开下一轮。
    RoundEnd,
    /// 有人累计达到阈值，游戏结束。
    GameOver { winners: Vec<PlayerId> },
}

/// 日志受众：公共事件 / 仅某玩家可见（如偷看结果）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Audience {
    Public,
    Player(PlayerId),
}

/// 日志种类（用于前端配色）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogKind {
    Info,
    Peek,
    Swap,
    Power,
    Cabo,
    Score,
}

#[derive(Clone, Debug)]
pub struct LogEntry {
    pub audience: Audience,
    pub kind: LogKind,
    pub text: String,
}

/// Typed public history for decision programs. No physical card IDs or secret ranks.
/// Reset each round; initial choices are published only after everyone has submitted.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PublicEvent {
    InitialPeek { player: PlayerId, slots: [SlotId; 2] },
    Power { player: PlayerId, power: PowerUse },
    Discard { player: PlayerId, rank: u8, powered: bool },
    Exchange {
        player: PlayerId,
        source: Pile,
        slots: Vec<SlotId>,
        exposed: Vec<u8>,
        /// Known only for discard-pile acquisitions, never for secret draws.
        incoming: Option<u8>,
        success: bool,
    },
    Cabo { player: PlayerId },
}

/// 一整局游戏（可能包含多轮）。
#[derive(Clone, Debug)]
pub struct Session {
    pub seed: u64,
    pub rng: StdRng,
    pub settings: Settings,
    pub round_no: u32,
    pub players: Vec<PlayerState>,
    /// 所有牌的运行时状态，下标即 CardId。
    pub cards: Vec<CardState>,
    /// 摸牌堆（末尾为堆顶）。
    pub deck: Vec<CardId>,
    /// 弃牌堆（末尾为堆顶）。
    pub discard: Vec<CardId>,
    pub phase: Phase,
    /// 本轮宣告 Cabo 的玩家（None = 尚未宣告）。
    pub cabo_caller: Option<PlayerId>,
    /// 宣告后其余玩家的加时回合队列。
    pub extra_turns: VecDeque<PlayerId>,
    /// 最近一次秘密查看的闪显（记忆模式下展示一次）；随该玩家下一次行动清除。
    pub peek_flash: BTreeMap<PlayerId, String>,
    pub log: Vec<LogEntry>,
    pub public_events: Vec<PublicEvent>,
}

impl Session {
    /// 建立大厅（未发牌）。座位即 `players` 的顺序。
    pub fn new_lobby(seed: u64, settings: Settings, players: Vec<PlayerState>) -> Session {
        let rng = StdRng::seed_from_u64(seed);
        Session {
            seed,
            rng,
            settings,
            round_no: 0,
            players,
            cards: Vec::new(),
            deck: Vec::new(),
            discard: Vec::new(),
            phase: Phase::Lobby,
            cabo_caller: None,
            extra_turns: VecDeque::new(),
            peek_flash: BTreeMap::new(),
            log: Vec::new(),
            public_events: Vec::new(),
        }
    }

    pub fn log_public(&mut self, kind: LogKind, text: impl Into<String>) {
        self.log.push(LogEntry { audience: Audience::Public, kind, text: text.into() });
    }

    pub fn log_to(&mut self, who: PlayerId, kind: LogKind, text: impl Into<String>) {
        self.log.push(LogEntry { audience: Audience::Player(who), kind, text: text.into() });
    }

    /// 弃牌堆顶的牌。
    pub fn discard_top(&self) -> Option<CardId> {
        self.discard.last().copied()
    }

    pub fn card(&self, id: CardId) -> &CardState {
        &self.cards[id as usize]
    }

    pub fn rank_str(&self, id: CardId) -> String {
        self.cards[id as usize].card.rank.to_string()
    }

    /// 自己秘密查看的点数是否需要在日志中隐去（记忆模式）。
    fn hide_private_value(&self) -> bool {
        self.settings.memory_mode
    }
}
