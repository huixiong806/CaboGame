# CaboGame

CaboGame 是一个支持 2–4 人的在线 Cabo 卡牌游戏。你可以与朋友一起游玩，也可以添加 AI 进行单人练习。浏览器访问即可加入房间，无需安装客户端。

游戏使用 Rust、Axum 和 HTMX 构建，支持实时同步、独立玩家视角、三种 AI 难度，以及高牌双对和凑满分规则。

## 开始游玩

下载本仓库，或克隆到本地：

```bash
git clone https://github.com/huixiong806/CaboGame.git
cd CaboGame
```

从源码启动需要安装 [Rust](https://www.rust-lang.org/tools/install)。预训练的 Hard 模型已随仓库提供，运行不需要 Python、PyTorch、GPU 或模型训练。

Windows 双击 `start.cmd`，或在项目目录运行：

```powershell
.\start.ps1
```

Linux / macOS：

```bash
./start.sh
```

启动脚本会构建游戏、打开浏览器并显示本机及局域网地址。默认地址为 **http://localhost:8080**。在首页创建房间，把房间号和服务地址告诉同一网络的朋友，或由房主添加 AI。

Windows 可用 `start.ps1 -Port 9000` 更换端口，`-NoBrowser` 仅启动服务，`-Stop` 停止服务。更新源码前若已有服务运行，先停止再重新启动。其他玩家连接时，主机需要放行对应端口的防火墙访问。

## 玩法

目标是让自己的累计分尽可能低。每轮开始时，每人获得四张暗牌，并秘密查看其中两张。

轮到你时，选择摸牌、拿弃牌堆顶交换，或宣告 Cabo。摸牌后可以弃置、发动能力，或与手牌交换。单张交换必定成功；多张交换只有点数全部相同才成功，成功时减少手牌数量，失败时参与交换的手牌公开，新牌加入手牌末尾。

7–8 可以偷看自己的牌，9–10 可以查看对手的牌，11–12 可以交换双方各一张手牌。能力只在摸牌后弃置该牌时发动。

宣告 Cabo 后，其他玩家各再行动一次，然后亮牌结算。宣告者的点数严格最低时本轮得 0 分，否则获得自己的点数加 Cabo 惩罚；其他人获得各自手牌点数。摸牌堆耗尽也会结束本轮。

游戏还包含两项特殊计分：

- **高牌双对**：整手恰好两张 12、两张 13，自己本轮得 0 分，其他人各得 50 分，覆盖普通计分。
- **凑满分**：累计恰好达到 100 分时自动降为 50 分，每人每大局一次。

先应用凑满分，再判断结束阈值；整局结束时累计分最低者获胜。房主可以设置结束阈值、Cabo 惩罚和记忆模式。完整规则见 [rule.md](rule.md)。

记忆模式下，秘密查看的点数只在查看时显示，之后需要玩家自行记忆；公开牌仍然可见。

## AI 难度

| 难度 | 特点 | 默认决策预算 |
|---|---|---:|
| Easy | 使用原搜索策略，适合熟悉玩法 | 原搜索配置 |
| Normal | 根据已知牌与公开行动规划，能利用特殊计分 | 约 300ms |
| Hard | 更充分的搜索、整局价值模型、Cabo 独立复核和确定终局策略 | 约 600ms |

所有 AI 都只读取对应玩家有权知道的信息。Hard 默认使用随仓库分发的 [整局价值模型](models/README.md)，由 Rust 原生推理；模型仅约 13.4 KiB。Normal 保持独立的稳定策略。

Hard 会结合累计分和重置资格评估整局收益，并能在最后一次 Cabo 响应中识别确定必胜的凑满分交换。它也会从公开历史识别重复弃牌交换循环，主动摸牌推进对局。自定义结束阈值或惩罚时，自动采用适用这些规则的搜索策略。

可用 `CABO_V4_BUDGET_US`、`CABO_HARD_BUDGET_US` 调整 Normal、Hard 的预算，单位为微秒。`CABO_HARD_VALUE_MODEL=off` 可关闭 Hard 的价值模型；通常无需更改这些设置。

## 部署

网页模板、CSS 和脚本均嵌入服务程序。运行时保留 `models/` 目录，并从项目或发布包目录启动：

```bash
cargo build --release --bin cabo-server
./target/release/cabo-server
```

Windows 对应程序为 `target/release/cabo-server.exe`。服务默认监听 `0.0.0.0:8080`；`PORT` 可更换端口，`RUST_LOG` 可调整日志级别。

也可以使用 Docker：

```bash
docker build -t cabogame .
docker run --rm -p 8080:8080 cabogame
```

## 开发

```bash
cargo test --release --lib --tests
```

项目结构：

```text
src/game/       规则、计分与玩家视图
src/ai/         AI 策略与信息集搜索
src/server/     房间管理、HTTP 路由和实时推送
src/bin/        游戏服务入口
templates/      页面模板
static/         样式、交互脚本与 HTMX
models/         随游戏分发的 Hard 模型
scripts/        发布包构建工具
tests/          规则与 AI 集成测试
research/       离线研究工具、评估数据与历史文档
```

规则引擎与界面共用玩家视图投影，服务端只发送该玩家可见的信息。浏览器通过 HTMX 提交行动，并通过 SSE 接收更新。AI 实现统一的 `Bot` 接口，方便维护和扩展。

Windows 免安装包的构建方法见 [scripts/README.md](scripts/README.md)。研究工具与归档入口见 [research/README.md](research/README.md)。

## 许可

源码与随附模型采用 [MIT 许可](LICENSE)。第三方脚本的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，Rust 依赖版本记录于 `Cargo.lock`。
