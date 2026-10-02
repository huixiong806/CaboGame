# CaboGame
一个使用 Rust 和 HTMX 构建的多人在线 Cabo 卡牌游戏，可人机对战

玩家访问网页即可创建/加入房间游玩，房主可以把任意座位切换成 AI（内置一个简单 Bot，架构上可方便接入更多难度）。

## 运行

```bash
cargo run --release --bin cabo-server
# 浏览器访问 http://localhost:8080
```

- `PORT`：监听端口，默认 `8080`
- `RUST_LOG`：日志级别，默认 `info`

## 玩法速览

1. 首页输入昵称**创建房间**，把 4 位房间号告诉朋友（或**添加 AI** 凑人）。
2. 开局每人 4 张暗牌，先秘密查看其中 2 张（查看哪两张是公开的，点数只有自己知道）。
3. 每回合三选一：
   - **摸牌**：从摸牌堆抽一张（只有你看得见），看到牌面后决定：
     **弃置**（7~12 可先发动能力）、或**点一张手牌直接换入**（必成功）、或进入**多张交换模式**
     （选 1~N 张手牌，点数全部相同才成功——赌对了手牌变少，赌失败则这些牌明置、新牌追加入手）。
   - **拿弃牌堆顶交换**：拿弃牌堆顶的牌（公开），同样与 1~N 张手牌交换。
   - **宣告 Cabo**：直接结束回合进入终局。
   手牌始终保持紧凑（无空位），换来的牌加入手牌末尾；交换不发动能力。
4. Cabo 后其余玩家各再行动一次，然后亮牌计分：宣告者严格最低得 0 分，否则 +10 惩罚；
   其余玩家各得自己的点数（含交换失败多出来的牌）。有人累计到 100 分时游戏结束，总分最低者获胜。
5. 房主在大厅可配置：Cabo 惩罚（默认 10）、结束阈值（默认 100）、**记忆模式**（默认关闭）。

## 部署

单二进制，模板/脚本/CSS 全部编译期打包，无运行时外部依赖：

```bash
cargo build --release
# 目标产物：target/release/cabo-server.exe（Linux 下无 .exe 后缀）
PORT=8080 ./target/release/cabo-server
```

或使用 Docker：

```bash
docker build -t cabo .
docker run --rm -p 8080:8080 cabo
```

## 架构

```
src/
  game/            纯规则引擎（不依赖 tokio/axum，可独立测试）
    mod.rs         牌、槽位、已知者集合 K、命令与日志类型
    engine.rs      状态机：A/B/C/D 行动、加时回合、计分、游戏结束
    view.rs        按玩家视角投影（信息隔离；Bot 与前端共用）
    sim.rs         无头对局驱动（模拟器与测试复用）
  ai/
    mod.rs         Bot trait + 注册表（新 Bot 实现 trait 后 register 即可）
    simple.rs      内置简单启发式 Bot（只按视图决策，不作弊）
  server/
    mod.rs         房间生命周期、会话令牌、AI 驱动循环
    render.rs      askama 模板与 SSE fragment 组装
    routes.rs      HTTP 路由（创建/加入/命令/SSE/静态资源）
templates/         全部界面为服务端渲染的 HTML 模板
static/            手写 CSS + vendored htmx（唯一第三方脚本）
```

### 记忆模式

房主可在大厅开启"记忆模式"：自己秘密查看过的点数（初始查看 / 偷看 / 间谍）
只在查看瞬间显示一次，之后界面上只保留"已看"角标、不再显示点数，需要玩家自己记忆；
日志也不再写出点数。**公开信息不受影响**（明置牌、全场皆知的牌始终显示）。
该开关只改变 UI 显示，不改变规则与已知者集合的判定。

### 前端为何没有一行手写 JS

所有游戏状态只存在于服务端；玩家界面是**按各自视角渲染的 HTML 片段**。
浏览器端用 htmx 完成两件事：表单提交命令（POST 后返回 204，不刷新页面）、
通过 SSE 接收 `board` 事件并原样替换页面主区域。状态变化由服务端的
`watch` 版本号广播，只在有人行动时渲染推送，服务端不为连接保存任何 UI 状态。
信息隔离天然成立：浏览器拿到的 HTML 里本来就没有它不该看到的牌面。

### 接入新 Bot

```rust
// src/ai/ 下新建文件，实现 trait：
struct AggressiveBot;
impl Bot for AggressiveBot {
    fn id(&self) -> &'static str { "aggressive" }
    fn name(&self) -> &'static str { "激进 AI" }
    fn decide(&self, view: &PlayerView, rng: &mut dyn RngCore) -> Command { /* ... */ }
}

// 并在 BotRegistry::with_builtins 中注册：
reg.register(Arc::new(AggressiveBot));
```

注册后大厅"添加 AI"的下拉框会自动出现新选项。`decide` 只能读取
`PlayerView`（该座位有权知道的信息），框架层面杜绝 Bot 作弊。

## 测试

```bash
cargo test                          # 15 个规则集成测试
cargo run --release --bin cabo-sim -- --games 1000 --players 4 --seed 1
# 无头模拟 1000 局 Bot 对战，输出胜率/轮数统计，用于回归与调参
```
