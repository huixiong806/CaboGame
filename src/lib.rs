//! Cabo 卡牌游戏 —— 规则引擎、AI、Web 服务端。
//!
//! 模块划分：
//! - [`game`]：纯规则的回合制状态机，不依赖 tokio / axum，可独立测试。
//! - [`ai`]：Bot trait 与内置机器人（只根据"该玩家已知的信息"决策，不作弊）。
//! - [`server`]：axum HTTP 服务（房间、SSE 推送、模板渲染）。

pub mod ai;
pub mod game;
pub mod server;
