# CaboGame
一个使用 Rust 和 HTMX 构建的多人在线 Cabo 卡牌游戏，可人机对战

玩家访问网页即可创建/加入房间游玩，房主可以把任意座位切换成 AI。大厅提供 Easy、Normal、Hard 三种难度。

## 新版决策程序（v4）

简短路线审计见 [AI_REBUILD.md](AI_REBUILD.md)，旧规则下的实现说明与实测见 [v4 验收记录](reports/V4_RESULTS.md)，新增策略见 [特殊规则 AI 验收记录](reports/SPECIAL_RULES_RESULTS.md)。当前难度分工见 [AI_DIFFICULTIES.md](AI_DIFFICULTIES.md)：

- **Easy / 简单 AI**：原搜索实现及默认策略网络，保留为简单难度和对照。
- **Normal / 普通 AI**：当前信念规划加特殊规则策略，默认新增 AI；能利用凑满分、完成/保护高牌双对、破坏对手组合。默认每步最多约 300ms，无需 Python/GPU/模型文件。固定为稳定基准，新的研究不会改变默认策略。
- **Hard / 困难 AI**：持续迭代的最强版本入口。目前保留已验收规划策略，将搜索预算提高到约 600ms、模拟上限与独立复核样本加倍，并能从公开历史识别重复弃牌交换循环后摸牌打破停滞；更高预算本身不能证明棋力更强，学习候选通过独立对照验收晋升。

`challenger` 陪练保留为离线评估对手，隐藏于大厅难度选单。

Hard 的学习路线、真实引擎自我对弈和本地价值网络实验见 [HARD_LEARNING.md](HARD_LEARNING.md)。训练样本和模型仅保存在本机 `data/`；没有显式环境变量时不读取研究模型。已验证的 V1 加宣告目标组合可通过 `CABO_HARD_VALUE_MODEL` 本地试用，完整结果与适用条件见报告。

本轮配对对局、学习指标、未晋升候选与防循环修复的结果见 [HARD_ITERATION_RESULTS.md](reports/HARD_ITERATION_RESULTS.md)。

旧简单 AI、旧战术 AI、v2 和无学习组件的独立入口，以及旧训练/实验工具，已归档到 [历史备份分支](https://github.com/huixiong806/CaboGame/tree/codex/archive-legacy-ai-20261004)。该分支保存清理前的源码、文本权重与报告；二进制训练数据只保留在本机。

当前网页对局尚未持久化录制，也不会在线训练；离线评估结果不能替代真人对局验证。

已有服务在运行时，先执行 `.\start.ps1 -Stop`，再执行 `.\start.ps1` 构建新版，以免 Windows 锁定旧二进制。随后在房间里选择上述 AI 即可。调整 Normal 速度用 `CABO_V4_BUDGET_US`，Hard 用 `CABO_HARD_BUDGET_US`（均为微秒）。

新的评估器严格检查命令合法性，让**一个候选对其余独立对手**，轮换所有座位。结果按种子组计算区间；`--blocks 20 --players 4` 是每个候选 80 局，不是 20 局。每局使用独立 Bot 实例，每个座位使用独立随机流；原始种子组数据可写到 TSV。

```powershell
# 相同时间预算，在会主动宣告的对手上比较新旧版本
cargo run --release --bin cabo-eval -- --candidate v4 --compare search --opponents challenger --blocks 20 --players 4 --jobs 4 --seed 71004 --cfg budget_us=50000 --compare-cfg budget_us=50000 --out reports/eval.tsv

# 固定模拟次数，可复现且不受机器负载影响
cargo run --release --bin cabo-eval -- --candidate v4 --compare challenger --opponents challenger --blocks 20 --cfg budget_us=0,simulations=512 --seed 81004

# 特殊策略消融：两边仍使用完整新规则，仅对照是否主动利用特殊规则
cargo run --release --bin cabo-eval -- --candidate normal --compare planner --opponents challenger --blocks 20 --players 4 --cfg budget_us=0,simulations=256 --compare-cfg budget_us=0,simulations=256,special_tactics=false --start-scores 80,90,95,70 --seed 91004

# 消融：关闭后续树规划、公开行动推断或独立复核（分别测试，不混在一起）
# --cfg budget_us=0,simulations=512,tree_depth=0
# --cfg budget_us=0,simulations=512,use_evidence=false
# --cfg budget_us=0,simulations=512,confirmation_samples=0

# 规则差分、信息隔离、完整对局、时间预算检查
cargo test --release --lib --tests
```

固定模拟模式的 `simulations` 是整条树轨迹数；旧版 `max_worlds` 是每个候选的世界数，两者不能按相同数字当作相同计算量。棋力比较同时看实际耗时。`use_evidence=false` 关闭采样中的行动似然，保留策略所用的公共行动信息；`tree_depth=0` 只优化根决策。`confirmation_samples=0` 可还原未加复核的搜索实验版。

## 运行

**一键开服（推荐）**：双击 `start.cmd`，或

```powershell
.\start.ps1                 # 自动构建 → 清理旧实例 → 启动 → 打开浏览器
.\start.ps1 -Port 9000      # 换端口
.\start.ps1 -NoBrowser      # 不自动开浏览器
.\start.ps1 -SkipBuild      # 跳过构建（改过源码就别加这个）
.\start.ps1 -Stop           # 停止正在运行的服务
```

Linux / macOS：`./start.sh`（参数同上，用 `--no-build` / `--no-browser` / `--stop`）。

脚本会打印**本机地址与局域网地址**（把房间号告诉同一网络的朋友即可加入）；
若朋友连不上，检查防火墙是否放行 `cabo-server`。

**手动运行**：

```bash
cargo run --release --bin cabo-server
# 浏览器访问 http://localhost:8080
```

- `PORT`：监听端口，默认 `8080`（绑定 `0.0.0.0`，局域网可直连）
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
   手牌始终保持紧凑（无空位），单张交换原位替换，多张成功合并将新牌放入首个被选位置，仅压缩多出的空位；失败时新牌追加到末尾。交换不发动能力。
4. Cabo 后其余玩家各再行动一次，然后亮牌计分：宣告者严格最低得 0 分，否则 +10 惩罚；
   其余玩家各得自己的点数（含交换失败多出来的牌）。**高牌双对**：整手恰好两张 12、两张 13，则本人本轮 0 分、其他人各 50 分，覆盖普通计分。**凑满分**：累计恰好 100 自动重置为 50，每人每大局一次。先重置，再判断是否有人累计达到结束阈值，总分最低者获胜。
5. 房主在大厅可配置：Cabo 惩罚（默认 10）、结束阈值（默认 100）、**记忆模式**（默认关闭）。

完整规则见 [rule.md](rule.md)。特殊规则已接入真实引擎与 AI 模拟结算；后续如何让 AI 主动利用它们，见 [特殊规则 AI 计划](AI_SPECIAL_RULES_PLAN.md)。

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
    planner/       v4 信念规划与主动战术陪练（独立实现）
    tactics.rs     原搜索版的策略组件（先验/rollout 共用）
    belief.rs      确定性化（按已知信息重建一个自洽世界）
    history.rs     公共历史记牌（从日志恢复弃牌堆构成）
    search.rs      原版 PIMC 搜索 AI（保留对照）
    nn.rs          手写 MLP + Adam（梯度检查有测试）
    policy.rs      策略网络（候选打分，模仿搜索的决策）
    value.rs       价值网络（终局分差修正 / 分数上下文胜率模型）
    weights.rs     训练产物（编译期嵌入，保持单文件部署）
  bin/
    arena.rs       配对自对弈擂台（配对种子、配对显著性）
    decide.rs      成对决策检验（同世界配对；含完美信息预言机）
    eval.rs        单候选对多个独立对手的评估（轮换座位、原始 TSV）
  server/
    mod.rs         房间生命周期、会话令牌、AI 驱动循环
    render.rs      askama 模板与 SSE fragment 组装
    routes.rs      HTTP 路由（创建/加入/命令/SSE/静态资源）
templates/         全部界面为服务端渲染的 HTML 模板
static/            手写 CSS + vendored htmx 1.9.12 与官方 SSE 扩展（第三方脚本）
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
cargo test                          # 规则集成测试 + AI 测试
cargo run --release --bin cabo-sim -- --games 1000 --players 4 --seed 1
# 无头模拟 1000 局 Bot 对战，输出胜率/轮数统计，用于回归与调参
```

## AI 选择与调参

| 大厅选项 | 命令行 ID | 用途 |
|---|---|---|
| 简单 AI（Easy） | `easy`（兼容 `search`、`v3`） | 原 PIMC 搜索及默认学习组件 |
| 普通 AI（Normal） | `normal`（兼容 `planner`、`v4`） | 默认选择，信念规划与特殊规则策略，默认每步上限约 300ms |
| 困难 AI（Hard） | `hard` | 当前规划策略采用更高预算，后续作为项目内最强版本的晋升入口，默认每步上限约 600ms |

陪练 `challenger` 仍可在 CLI 和评估器中使用。旧规则下新版优于原搜索版的结果见 [旧验收记录](reports/V4_RESULTS.md)；新规则下的策略与测试需单独评估，不能沿用旧胜率或声称已稳定强于真人。

原搜索版依赖的 `tactics`、`nn`、`policy`、`value` 和 `weights` 是内部组件，不是额外的 AI 选项。文本权重随源码编译，无需上传模型二进制。

```powershell
# 原版搜索每步预算（微秒）；新版使用 CABO_V4_BUDGET_US
$env:CABO_AI_BUDGET_US = "50000"

# 两组策略对打；一个 AI 对其余独立对手的比较优先使用上面的 cabo-eval
cargo run --release --bin cabo-arena -- --a v4 --b challenger --games 20 --seed 5

# 原版搜索的成对决策诊断
cargo run --release --bin cabo-decide -- --games 20 --worlds 24 --budget-us 4000
```

历史版本与深度学习路线报告见 [备份分支](https://github.com/huixiong806/CaboGame/tree/codex/archive-legacy-ai-20261004)：[AI_NOTES.md](https://github.com/huixiong806/CaboGame/blob/codex/archive-legacy-ai-20261004/AI_NOTES.md)、[DL_ROUTE.md](https://github.com/huixiong806/CaboGame/blob/codex/archive-legacy-ai-20261004/DL_ROUTE.md)。历史混合对手实验需要在该分支复现；主分支不再接受 `simple`、`tactician`、`v0`、`v1`、`v2` 或 `search-classic`。

所有分支均忽略 `target/`、`data/`、可执行文件、模型二进制和 Python 缓存；本地训练数据不会上传 GitHub。

## 版权声明与第三方依赖

本项目以 [MIT 许可](LICENSE) 发布。

内嵌的 `static/htmx.min.js` 与 `static/ext-sse.js`
来自 htmx 1.9.12 及其官方 SSE 扩展，以 0BSD 许可分发，见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

其余 Rust 依赖由 Cargo 从 crates.io
获取，版本锁定在 `Cargo.lock`。
