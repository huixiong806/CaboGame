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

pub mod simple;

/// 一个可接入游戏的 AI 策略。
pub trait Bot: Send + Sync {
    /// 注册用的稳定 id（如 `"simple"`）。
    fn id(&self) -> &'static str;
    /// 展示名（如 `"简单 AI"`）。
    fn name(&self) -> &'static str;
    /// 根据当前玩家视图给出下一条命令。`rng` 供策略做随机选择。
    fn decide(&self, view: &PlayerView, rng: &mut dyn rand::RngCore) -> Command;
}

/// 已注册的 Bot 集合。
#[derive(Clone)]
pub struct BotRegistry {
    bots: Vec<Arc<dyn Bot>>,
}

impl BotRegistry {
    /// 默认注册表：内置所有自带 Bot。
    pub fn with_builtins() -> Self {
        let mut reg = BotRegistry { bots: Vec::new() };
        reg.register(Arc::new(simple::SimpleBot));
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
