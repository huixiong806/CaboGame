//! 版本快照：把"迭代过程中出现过的几代 AI"固化成可直接对打的预设。
//!
//! 迭代链条（详见仓库根目录 `AI_NOTES.md`）：
//! - `v1` 战术 AI：纯启发式（信念统计 + 固定规则），微秒级决策。
//! - `v2` 搜索 AI v2：第一版 PIMC，回报用"与最强劲对手的分差"，无 Cabo 加价。
//! - `v3` 搜索 AI v3：PIMC + **胜率口径回报** + Cabo 加价 + 位置化信念 + 调过的策略参数。
//!
//! `build_bot` 认识 `simple` / `tactician` / `search` 以及 `v1` / `v2` / `v3`，
//! 因此擂台可以直接 `--a v3 --b v2` 让新旧版本互殴。

use std::sync::Arc;

use super::search::{SearchBot, SearchCfg, ValueKind};
use super::tactics::{PolicyCfg, TacticianBot};
use super::{simple::SimpleBot, Bot};

/// v1（战术 AI）的策略参数。
pub fn v1_policy() -> PolicyCfg {
    PolicyCfg::default()
}

/// v2（第一版搜索）的配置：分差口径、无 Cabo 加价、早期策略参数。
pub fn v2_cfg() -> SearchCfg {
    SearchCfg {
        value: ValueKind::Lead,
        cabo_margin: 0.0,
        value_k: 8.0,
        policy: PolicyCfg {
            swap_unknown_gain: 1.0,
            group_min_gain: -1.0,
            ..PolicyCfg::default()
        },
        ..SearchCfg::default()
    }
}

/// v3（当前最强）的配置。
pub fn v3_cfg() -> SearchCfg {
    SearchCfg {
        value: ValueKind::WinProb,
        cabo_margin: 16.0,
        value_k: 8.0,
        improve_margin: 0.04,
        policy: PolicyCfg::default(),
        ..SearchCfg::default()
    }
}

/// 按版本 id 构造 Bot。
pub fn build(id: &str) -> Option<Arc<dyn Bot>> {
    match id {
        "simple" | "v0" => Some(Arc::new(SimpleBot)),
        "tactician" | "v1" => Some(Arc::new(TacticianBot::new(v1_policy()))),
        "v2" => Some(Arc::new(SearchBot::new(v2_cfg()))),
        "search" | "v3" => Some(Arc::new(SearchBot::new(v3_cfg()))),
        _ => None,
    }
}

/// 版本展示名。
pub fn label(id: &str) -> &'static str {
    match id {
        "v0" | "simple" => "简单 AI（基准）",
        "v1" | "tactician" => "战术 AI（v1 启发式）",
        "v2" => "搜索 AI v2（分差口径）",
        "v3" | "search" => "搜索 AI v3（胜率口径）",
        "v4" | "planner" => "信念规划 AI（v4）",
        "challenger" => "主动战术 AI（陪练）",
        _ => "未知版本",
    }
}
