//! AI 玩家：Bot trait 与注册表。
//!
//! 接入新 Bot 只需两步：
//! 1. 实现 [`Bot`] trait（`decide` 只读取 [`PlayerView`]，即"该座位有权知道的信息"，
//!    框架保证 Bot 无法作弊）；
//! 2. 在 [`BotRegistry::with_builtins`]（或你自己的组装代码）里调用 `register`。
//!
//! 注册后，房主在房间里就能把座位切换成对应的 AI。

use std::sync::Arc;

use crate::game::view::{Panel, PlayerView};
use crate::game::Command;

pub mod belief;
pub mod history;
pub mod nn;
pub mod policy;
pub mod planner;
pub mod value;
pub mod weights;
pub mod search;
pub mod simple;
pub mod tactics;
pub mod versions;

/// 一个可接入游戏的 AI 策略。
pub trait Bot: Send + Sync {
    /// 注册用的稳定 id（如 `"simple"`）。
    fn id(&self) -> &'static str;
    /// 展示名（如 `"简单 AI"`）。
    fn name(&self) -> &'static str;
    /// 根据当前玩家视图给出下一条命令。`rng` 供策略做随机选择。
    fn decide(&self, view: &PlayerView, rng: &mut dyn rand::RngCore) -> Command;
}

/// 给 Bot 换个 id / 展示名（同一实现用不同参数对打时用）。
pub struct AliasBot {
    pub inner: Arc<dyn Bot>,
    pub id: &'static str,
    pub name: &'static str,
}

impl Bot for AliasBot {
    fn id(&self) -> &'static str {
        self.id
    }
    fn name(&self) -> &'static str {
        self.name
    }
    fn decide(&self, view: &PlayerView, rng: &mut dyn rand::RngCore) -> Command {
        self.inner.decide(view, rng)
    }
}

/// 已注册的 Bot 集合。
#[derive(Clone)]
pub struct BotRegistry {
    bots: Vec<Arc<dyn Bot>>,
}

impl Default for BotRegistry {
    fn default() -> Self {
        BotRegistry::new()
    }
}

/// 按 id 构造 Bot，`cfg` 是 `key=value` 形式的参数覆盖（调参 / 擂台用）。
///
/// 可用 id：`simple`/`v0`（基准）、`tactician`/`v1`（启发式）、
/// `search`/`v3`（当前最强）、`v2`（第一版搜索，用于新旧对打）。
pub fn build_bot(id: &str, cfg: &[(String, String)]) -> Option<Arc<dyn Bot>> {
    if matches!(id, "planner" | "v4" | "challenger") {
        let mut p = planner::PlannerCfg::default();
        for (key, value) in cfg {
            if !p.set(key, value) { return None; }
        }
        return Some(if id == "challenger" { Arc::new(planner::ChallengerBot) }
            else { Arc::new(planner::PlannerBot::new(p)) });
    }
    let mut policy_cfg = match id {
        "simple" | "v0" | "tactician" | "v1" => tactics::PolicyCfg::default(),
        "v2" => versions::v2_cfg().policy,
        "search" | "v3" => versions::v3_cfg().policy,
        _ => return None,
    };
    let mut search_cfg = match id {
        "v2" => versions::v2_cfg(),
        _ => search::SearchCfg::default(),
    };
    for (k, v) in cfg {
        if let Some(rest) = k.strip_prefix("policy.") {
            if !policy_cfg.set(rest, v) { return None; }
        } else if !search_cfg.set(k, v) && !policy_cfg.set(k, v) {
            return None;
        }
    }
    search_cfg.policy = policy_cfg;
    Some(match id {
        "simple" | "v0" => Arc::new(simple::SimpleBot),
        "tactician" | "v1" => Arc::new(tactics::TacticianBot::new(policy_cfg)),
        _ => Arc::new(search::SearchBot::new(search_cfg)),
    })
}

impl BotRegistry {
    /// 空注册表（调参 / 擂台用，自行注册 Bot）。
    pub fn new() -> Self {
        BotRegistry { bots: Vec::new() }
    }

    /// 默认注册表：内置所有自带 Bot。
    pub fn with_builtins() -> Self {
        let mut reg = BotRegistry { bots: Vec::new() };
        reg.register(Arc::new(planner::PlannerBot::new(planner::PlannerCfg::default())));
        reg.register(Arc::new(planner::ChallengerBot));
        reg.register(Arc::new(simple::SimpleBot));
        reg.register(Arc::new(tactics::TacticianBot::new(tactics::PolicyCfg::default())));
        reg.register(Arc::new(search::SearchBot::new(search::SearchCfg::default())));
        // 对照组：把学习组件关掉的搜索 AI（即"深度学习之前"的版本）。
        // 同一局里同时加"搜索 AI"和它，可以直接感受学习组件带来的差别。
        reg.register(Arc::new(AliasBot {
            inner: Arc::new(search::SearchBot::new(search::SearchCfg {
                rollout_net: 0,
                ..search::SearchCfg::default()
            })),
            id: "search-classic",
            name: "搜索 AI（对照 · 无学习组件）",
        }));
        reg
    }

    pub fn register(&mut self, bot: Arc<dyn Bot>) {
        assert!(!self.bots.iter().any(|b| b.id() == bot.id()), "重复的 bot id: {}", bot.id());
        self.bots.push(bot);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Bot>> {
        self.bots.iter().find(|b| b.id() == id).cloned()
    }

    /// 默认 Bot id（新 AI 座位使用）。
    pub fn default_id(&self) -> &'static str {
        self.bots.first().map(|b| b.id()).unwrap_or("simple")
    }

    /// (id, 展示名) 列表，供房间前端下拉选择。
    pub fn list(&self) -> Vec<(&'static str, &'static str)> {
        self.bots.iter().map(|b| (b.id(), b.name())).collect()
    }
}

/// 全局统计计数器：供擂台 / 模拟器观察搜索型 Bot 的开销。
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static DECISIONS: AtomicU64 = AtomicU64::new(0);
    /// 定向延长（宣告进前二名）触发次数。
    pub static EXTEND: AtomicU64 = AtomicU64::new(0);
    pub static DECIDE_NS: AtomicU64 = AtomicU64::new(0);
    pub static DECIDE_NS_MAX: AtomicU64 = AtomicU64::new(0);
    pub static ROLLOUTS: AtomicU64 = AtomicU64::new(0);
    pub static ROLLOUT_ACTIONS: AtomicU64 = AtomicU64::new(0);
    pub static WORLD_CLONES: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        DECISIONS.store(0, Ordering::Relaxed);
        DECIDE_NS.store(0, Ordering::Relaxed);
        DECIDE_NS_MAX.store(0, Ordering::Relaxed);
        ROLLOUTS.store(0, Ordering::Relaxed);
        ROLLOUT_ACTIONS.store(0, Ordering::Relaxed);
        WORLD_CLONES.store(0, Ordering::Relaxed);
    }

    pub fn note_decide(ns: u64) {
        DECISIONS.fetch_add(1, Ordering::Relaxed);
        DECIDE_NS.fetch_add(ns, Ordering::Relaxed);
        DECIDE_NS_MAX.fetch_max(ns, Ordering::Relaxed);
    }

    pub fn note_rollout(actions: u64) {
        ROLLOUTS.fetch_add(1, Ordering::Relaxed);
        ROLLOUT_ACTIONS.fetch_add(actions, Ordering::Relaxed);
    }

    pub fn note_rollouts(count: u64, actions: u64) {
        ROLLOUTS.fetch_add(count, Ordering::Relaxed);
        ROLLOUT_ACTIONS.fetch_add(actions, Ordering::Relaxed);
    }

    pub fn note_clone() {
        WORLD_CLONES.fetch_add(1, Ordering::Relaxed);
    }

    /// 定向延长触发次数（宣告进前二名）。
    pub fn note_extend() {
        EXTEND.fetch_add(1, Ordering::Relaxed);
    }
}

/// 统计报告（无数据时返回空串）。
pub fn stats_string() -> String {
    let d = stats::DECISIONS.load(std::sync::atomic::Ordering::Relaxed);
    if d == 0 {
        return String::new();
    }
    let ns = stats::DECIDE_NS.load(std::sync::atomic::Ordering::Relaxed);
    let max = stats::DECIDE_NS_MAX.load(std::sync::atomic::Ordering::Relaxed);
    let r = stats::ROLLOUTS.load(std::sync::atomic::Ordering::Relaxed);
    let ra = stats::ROLLOUT_ACTIONS.load(std::sync::atomic::Ordering::Relaxed);
    format!(
        "搜索统计: 决策 {d} 次，平均 {:.2}ms，最慢 {:.1}ms；rollout {r} 次，平均 {:.0} 步/次",
        ns as f64 / d as f64 / 1e6,
        max as f64 / 1e6,
        if r > 0 { ra as f64 / r as f64 } else { 0.0 }
    )
}

/// 兜底命令：Bot 决策非法时按面板状态给一个"合法保底"动作，保证牌局永不卡死。
pub fn fallback_command(view: &PlayerView) -> Command {
    let my_slots = || -> Vec<u8> {
        view.me_seat.as_ref().map(|s| s.slots.iter().map(|c| c.slot).collect()).unwrap_or_default()
    };
    match &view.panel {
        Panel::PeekPick { .. } => {
            let mut s = my_slots();
            s.truncate(2);
            if s.len() == 2 {
                Command::PeekInitial { slots: [s[0], s[1]] }
            } else {
                Command::PeekInitial { slots: [0, 1] }
            }
        }
        Panel::Idle { can_draw, can_swap_discard, .. } => {
            if *can_draw {
                Command::BeginDraw
            } else if *can_swap_discard {
                Command::BeginSwap
            } else {
                Command::CallCabo
            }
        }
        Panel::Drew { .. } => Command::DiscardDrawn { power: None },
        Panel::AimPeek | Panel::AimSpy | Panel::AimSwap { .. } => Command::Cancel,
        Panel::SwapSelecting { count, .. } => {
            if *count >= 1 {
                Command::SwapCommit
            } else {
                let s = my_slots().into_iter().next().unwrap_or(0);
                Command::SwapToggle { slot: s }
            }
        }
        Panel::ConfirmCabo => Command::CallCabo,
        _ => Command::Cancel,
    }
}

/// 一行人类可读的"学习组件状态"，供服务端启动时打印。
///
/// 目的：让"深度学习到底有没有生效"变成**看得见**的事实，而不是一句口头承诺。
pub fn learned_status() -> String {
    let policy = if policy::available() {
        "策略网络 ✓ 已加载（搜索 AI 的 rollout：我方座位）"
    } else {
        "策略网络 ✗ 未加载（搜索 AI 退化为启发式 rollout）"
    };
    let value = if value::margin_available() {
        "价值网络 ✓ 已加载（默认未启用，可对照）"
    } else {
        "价值网络 — 未加载"
    };
    format!("{policy}；{value}")
}
