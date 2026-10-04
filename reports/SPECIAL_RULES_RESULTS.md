# 特殊规则 AI 第一版验收（2026-10-05）

本版保留信念采样与信息集搜索路线，修复普通 Cabo 成功率门槛阻止有利特殊策略的问题。Normal 现在能主动利用精确凑满分，完成和保护已知高牌双对，以及用交换能力破坏对手已知组合。特殊机会仍按大局收益比较，并经过独立样本复核。

## 方法

每项实验使用 20 个独立种子组，每组轮换四个座位，一个候选对三个独立 `challenger` 陪练，每个版本 80 个完整大局。每座位有独立随机流；配对版本使用相同起始种子。四项配对实验共 640 局，没有非法命令。结束阈值 100，Cabo 惩罚 10。

Normal 消融对照使用相同规划器，设置 `special_tactics=false`；两边均保留正确的新计分规则。陪练和模拟后续策略也感知特殊规则。临近终局实验在发牌后设置累计分 `[80,90,95,70]`，每人重置资格未用。这是有意覆盖特殊规则的压力场景，不能视为自然对局分布。

胜率计入并列获胜份额。“分差”指候选最终累计分减其他人中的最低累计分，越小越好。下表的差值区间按 20 个配对种子组的差计算，使用 Student t 95% 区间；不是把 80 局当作互相独立。时间预算模式受机器负载影响；固定轨迹数模式有更好复现性，但不保证相同计算时间。

## 结果

| 场景与预算 | 候选 / 对照胜率 | 胜率差及 95% 区间 | 候选 / 对照分差 | 分差变化及 95% 区间 | 重置次数 |
|---|---|---|---|---|---|
| 普通开局；各 256 条轨迹 | 25.0% / 25.0% | 0.0 pp ± 14.7 pp | 14.68 / 19.24 | −4.56 ± 7.77 | 49 / 22 |
| 临近终局；各 256 条轨迹 | 30.6% / 17.5% | +13.1 pp ± 13.2 pp | 12.97 / 24.59 | −11.61 ± 8.35 | 46 / 18 |
| 独立种子临近终局；各 50ms | 35.6% / 19.4% | +16.2 pp ± 10.6 pp | 14.86 / 24.82 | −9.96 ± 7.05 | 35 / 15 |
| 普通开局；Hard 60ms / Normal 30ms | 38.1% / 35.0% | +3.1 pp ± 14.1 pp | 12.26 / 14.05 | −1.79 ± 8.29 | 39 / 45 |

重置次数只统计候选自身在实际结算时触发的次数，不是模拟计数。50ms 复核使用独立于固定轨迹实验的种子，平均决策耗时 38.10 / 37.85ms，最长 51.12 / 50.48ms。这组结果支持本版在该临近终局基准上改进了胜率与最终分差；不支持据此宣布普遍超越人类。普通开局未看出明确胜率差异，样本不足以证明没有退化。

固定轨迹实验中新策略平均耗时分别为 52.34 / 72.83ms，对照为 40.19 / 47.61ms，因此单看同轨迹数的胜率不足以说明效率改善。Hard 对照平均耗时 44.30 / 22.32ms；本次只是缩短预算的二倍预算实验，并未验证默认 600 / 300ms 下显著更强。当前 Hard 仍是同一已验证策略的更多计算入口。

640 局中没有自然触发高牌双对；其行为由确定场景测试验证，整体胜率贡献尚无证据。

原始种子组数据：[普通开局消融](SPECIAL_NORMAL_START.tsv)、[临近终局消融](SPECIAL_NORMAL_LATE.tsv)、[等时间独立复核](SPECIAL_NORMAL_LATE_TIME.tsv)、[Hard 预算对照](SPECIAL_HARD_START.tsv)。TSV 首行保存实际参数。

## 行为与工程检查

源码测试共 58 项通过、1 项忽略，覆盖共享计分差分、玩家信息隔离、规则回归、完整合法对局，以及新增策略场景：

- 三人局主动接受 Cabo 惩罚，累计 `80 + 10 + 10 = 100 → 50`，经真实引擎处理其他人的最后行动后获胜；普通 Cabo 严格最低概率为零也可选出该策略。
- 最后一回合保留较高点数，避免换低牌破坏自己的精确重置。
- 换入 12 完成 `[12,12,13,13]`，及将多余高牌合并成恰好四张。
- 已成型组合遇到 0 仍保护组合；交换能力破坏对手已知完整组合。
- 未知牌与已用重置资格不能冒充确定机会；高牌双对在当前累计分下仍会输时，不因完整组合而自动 Cabo。

候选报告增加重置和高牌双对的模拟触发比例；这些是当前采样与后续策略下的估计，不等于真实玩家胜率。根策略只使用该玩家合法可见的牌和公开信息，未知牌的期望不被当作恰好整数点数。

Windows 上使用 `--target-dir target/ai-validation` 隔离构建与验证，避免覆盖正在运行的玩家服务。独立端口服务的 HTTP 检查确认大厅选单只有三种难度，大厅与牌局座位均显示各自难度，局内规则说明包含两个特殊规则。二进制、日志和临时预览均不纳入 Git。

## 复现

```powershell
cargo build --release --target-dir target/ai-validation --bin cabo-eval

.\target\ai-validation\release\cabo-eval.exe --candidate normal --compare planner --opponents challenger --blocks 20 --players 4 --jobs 2 --seed 91004 --cfg budget_us=0,simulations=256 --compare-cfg budget_us=0,simulations=256,special_tactics=false --out reports/SPECIAL_NORMAL_START.tsv

.\target\ai-validation\release\cabo-eval.exe --candidate normal --compare planner --opponents challenger --blocks 20 --players 4 --jobs 2 --seed 92004 --cfg budget_us=0,simulations=256 --compare-cfg budget_us=0,simulations=256,special_tactics=false --start-scores 80,90,95,70 --out reports/SPECIAL_NORMAL_LATE.tsv

.\target\ai-validation\release\cabo-eval.exe --candidate normal --compare planner --opponents challenger --blocks 20 --players 4 --jobs 2 --seed 94004 --cfg budget_us=50000 --compare-cfg budget_us=50000,special_tactics=false --start-scores 80,90,95,70 --out reports/SPECIAL_NORMAL_LATE_TIME.tsv

.\target\ai-validation\release\cabo-eval.exe --candidate hard --compare normal --opponents challenger --blocks 20 --players 4 --jobs 2 --seed 93004 --cfg budget_us=60000 --compare-cfg budget_us=30000 --out reports/SPECIAL_HARD_START.tsv

cargo test --release --target-dir target/ai-validation --lib --tests --quiet
```

## 剩余工作

当前优先识别完整、可观察的特殊机会。部分高牌双对的长期收集概率、对手为特殊规则保留高牌时的信念校准、重置资格的跨轮价值仍未实现。Normal 等真人验收后固定版本；后续策略先进入 Hard 候选，在独立测试中超过现有冠军后晋升，见 [难度版本约定](../AI_DIFFICULTIES.md)。
