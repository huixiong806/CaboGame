# CaboGame
一个使用 Rust 和 HTMX 构建的多人在线 Cabo 卡牌游戏，可人机对战

玩家访问网页即可创建/加入房间游玩，房主可以把任意座位切换成 AI（内置一个简单 Bot，架构上可方便接入更多难度）。

## 新版决策程序（v4）

简短路线审计见 [AI_REBUILD.md](AI_REBUILD.md)，实现说明与实测见 [v4 验收记录](reports/V4_RESULTS.md)。新版已注册到房间下拉框：

- **信念规划 AI（v4）**：默认新增 AI。准确读取公共知识和弃牌记录；从公开行动推断暗牌；按玩家可观察信息共享搜索树；用新的配对样本复核搜索提案。默认每步最多约 300ms，不需要 Python/GPU/模型文件。
- **主动战术 AI（陪练）**：独立的快速策略，会主动宣告、使用已知低牌和能力交换，也适合直接试玩。
- 原有搜索 AI、无学习组件对照和简单 AI 继续保留。

重新运行 `start.ps1` 后，在房间里选择上述 AI 即可。只想调整新版速度，可设置 `CABO_V4_BUDGET_US`（微秒）。

新的评估器严格检查命令合法性，让**一个候选对其余独立对手**，轮换所有座位。结果按种子组计算区间；`--blocks 20 --players 4` 是每个候选 80 局，不是 20 局。每局使用独立 Bot 实例，每个座位使用独立随机流；原始种子组数据可写到 TSV。

```powershell
# 相同时间预算，在会主动宣告的对手上比较新旧版本
cargo run --release --bin cabo-eval -- --candidate v4 --compare search --opponents challenger --blocks 20 --players 4 --jobs 4 --seed 71004 --cfg budget_us=50000 --compare-cfg budget_us=50000 --out reports/eval.tsv

# 固定模拟次数，可复现且不受机器负载影响
cargo run --release --bin cabo-eval -- --candidate v4 --compare challenger --opponents simple,tactician,challenger --blocks 20 --cfg budget_us=0,simulations=512 --seed 81004

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
    tactics.rs     手写战术策略（先验/rollout 共用）
    belief.rs      确定性化（按已知信息重建一个自洽世界）
    history.rs     公共历史记牌（从日志恢复弃牌堆构成）
    search.rs      PIMC 搜索 AI（当前最强）
    nn.rs          手写 MLP + Adam（梯度检查有测试）
    policy.rs      策略网络（候选打分，模仿搜索的决策）
    value.rs       价值网络（终局分差修正 / 分数上下文胜率模型）
    weights.rs     训练产物（编译期嵌入，保持单文件部署）
  bin/
    arena.rs       配对自对弈擂台（配对种子、配对显著性）
    decide.rs      成对决策检验（同世界配对；含完美信息预言机）
    train.rs       数据生成（策略/价值/分数上下文）
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

## AI 强度与调参

内置三个 Bot（大厅"添加 AI"可选）：`简单 AI`、`战术 AI`、`搜索 AI`（最强）。

> **学习组件在哪里？** 没有单独的"深度学习 AI"——训练出来的策略网络是**搜索 AI 内部的一个组件**，
> 默认开启（搜索 AI 在 rollout 里、我方座位改用策略网络，实测约 +6~8 分）。
> 大厅里还提供一个对照项 **`搜索 AI（对照 · 无学习组件）`**：把它和"搜索 AI"加进同一局，
> 就能直接对比学习组件带来的差别。
> 服务端启动时会打印一行组件状态，例如
> `AI 组件: 策略网络 ✓ 已加载（搜索 AI 的 rollout：我方座位）；价值网络 ✓ 已加载（默认未启用，可对照）`。

```bash
# 配对自对弈擂台（同一批种子、逐座位配对，输出最强座位分差与 1se）
cargo run --release --bin cabo-arena -- --a v3 --b simple --games 25 --seed 5
cargo run --release --bin cabo-arena -- --a v3 --b tactician --games 16 --seed 5

# 成对决策检验：回答"搜索的选择是否真的比启发式更好"
cargo run --release --bin cabo-decide -- --games 20 --worlds 24 --budget-us 4000

# 单步时间预算（默认 300ms；规则上限 5s）
CABO_AI_BUDGET_US=8000 cargo run --release --bin cabo-arena -- --a v3 --b v1 --games 20
```

**搜索 AI 的时间预算默认 300ms/步**：实测瓶颈是候选估值的蒙特卡洛精度
（误差 ∝ 1/√世界数），10 倍预算 ≈ 7 分/局；300ms 之后再加大**已饱和**（1s 无增益）。
另外 rollout 里**我方座位**会使用训练出来的策略网络（`rollout_net=1`），
让我方未来走法与被评估的策略一致，实测再带来约 10 分（平均累计分差口径，7.3σ）。
设计细节与全部实验记录见 [DL_ROUTE.md](DL_ROUTE.md) 与 [AI_NOTES.md](AI_NOTES.md)。

### 学习组件（可开关，均有配对显著性记录）

| 组件 | 开关 | 状态 |
|---|---|---|
| 策略网络（候选打分，模仿搜索决策）| `rollout_net` / `prior_from_policy` | **已启用**（rollout 我方座位）|
| 分数上下文胜率模型（逻辑回归，校准好）| `value=score_ctx` | 对照用（中性）|
| 价值网络（终局分差修正）| `value=learned_margin` | 对照用（更差）|
| 配对显著性门槛 / 方差收缩 / 定向延长 | `paired_gate` / `shrink_noise` / `extend_budget_us` | 对照用（均更差）|

训练数据与权重：`cabo-train --mode policy|value3|scorectx` 产出数据集，
`tools/train_*.py`（PyTorch，GPU）训练并把权重写回 `src/ai/weights.rs`（编译期嵌入）。

## 版权声明与第三方依赖

本项目以 [MIT 许可](LICENSE) 发布。

内嵌的 `static/htmx.min.js` 与 `static/ext-sse.js`
来自 htmx 1.9.12 及其官方 SSE 扩展，以 0BSD 许可分发，见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

其余 Rust 依赖由 Cargo 从 crates.io
获取，版本锁定在 `Cargo.lock`。
