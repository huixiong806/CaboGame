//! `cabo-server`：启动 Web 服务。
//!
//! 环境变量：
//! - `PORT`：监听端口（默认 8080）
//! - `RUST_LOG`：日志级别（默认 info）

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cabo::ai::BotRegistry;
use cabo::server::{reaper, routes, Shared};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let state = Arc::new(Shared {
        rooms: Mutex::new(HashMap::new()),
        bots: BotRegistry::with_builtins(),
    });

    tokio::spawn(reaper(state.clone()));

    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .unwrap_or_else(|e| panic!("无法绑定端口 {port}: {e}"));
    tracing::info!("Cabo 服务已启动: http://localhost:{port}");
    tracing::info!("AI 组件: {}", cabo::ai::learned_status());
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("服务退出: {e}");
    }
}
