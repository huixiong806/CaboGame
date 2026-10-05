> 归档文档：保留当时的方案、默认配置与结论。当前版本见 [项目 README](../../README.md)，路径和复现说明见 [研究索引](../README.md)。

# Hard 的学习与搜索实验

Normal 保持 `3d6f668` 的默认决策，Hard 的候选通过显式配置进入评估，也可通过 `CABO_HARD_VALUE_MODEL` 在本机网页中试用。模型、训练样本、可执行文件全部保存在忽略的 `data/`、`target/`，不能加入 Git。是否晋升以独立整场对局结果为准，预测误差下降不能单独作为棋力证明。

## 路线选择

启发式适合修复规则理解、信息边界和动作覆盖，但继续加局部奖励容易互相冲突。长期更值得投入的是自我对弈学习策略/价值，加合法信息集中的搜索。学习负责长期取舍与搜索先验，搜索负责当前具体局面的反事实比较。Cabo 是多人、部分可观察博弈，不能把两人零和算法的收敛保证直接当作本项目的保证。

[ReBeL](https://arxiv.org/abs/2007.13544) 展示了隐藏信息下的学习与搜索组合；[DeepNash](https://arxiv.org/abs/2206.15378) 展示了另一条不依赖搜索的深度强化学习路线。它们支持尝试学习路线，但不能证明某个 Cabo 实现会进步。本项目仍需要多种对手、座位轮换、独立种子和完整比赛验收。

本次首先实现 Monte Carlo 整局价值学习：从真实引擎的完整比赛取得最终胜负，训练回合边界的胜率网络，再用网络评价搜索中的回合结算。这不是已经完成的端到端策略强化学习。下一轮用当前搜索策略产生新比赛、重新学习价值，是一次近似策略迭代；如果价值学习的实战收益很小，应转向学习当前手牌信息与候选动作的策略/价值，而不是无限扩大同类分数数据。

## 信息与模型

输入仅包含公开累计分、每人的重置是否已用、人数、目标分和惩罚分。未来回合重新洗牌、随机确定先手，因此这些特征适合预测回合结束后的继续比赛价值。它们不会表达当前未结束回合的私有手牌质量。

网络为 Deep Sets：每人 7 个输入，经 `7→32→32` 编码，汇总所有玩家的平均编码，再经 `64→32→1` 输出相对原公式的 logit 修正。Softmax 给出和为 1 的胜率；结构不依赖座位编号。共 3,425 个参数，13,708 字节模型。支持 NumPy/Adam 和 PyTorch/CUDA 训练，Rust 原生推理，运行游戏不依赖 Python、GPU 或外部模型运行时。

当前训练覆盖 2/3/4 人、目标 100、惩罚 10。其他计分设置保留原价值公式；整局已经结束时始终使用真实赢家和并列分摊，网络不覆盖结算。网络文件有版本头、固定长度和有限数校验；显式请求不存在或损坏模型时构造失败，不能把缺模型的对照误记成神经网络结果。

## 数据与复现

`cabo-match-data` 使用真实 `Session`，每个 AI 只接收 `PlayerView`。非法命令立即报错，无纠错回退。数据只导出公开的回合边界与最终胜负，不导出手牌或牌堆。同一比赛所有回合共享 seed，训练/选模型/测试必须按完整比赛分开。

采集先写 `.tsv.incomplete`，整批比赛全部成功后才改为正式 TSV，并写比赛范围与配置 manifest。`--max-actions` 超限时报错，保留失败 seed 和合法视角快照；训练器拒绝明确标记为 incomplete/invalid 的文件，并在有 manifest 时检查完整 seed 覆盖。旧数据没有 manifest，其完成数量由当时的采集结果核对。

默认 25% 普通开局，75% 从随机公开累计分开始，覆盖残局。已用重置的玩家只从 50～99 开始，避免“重置已用而累计分小于 50”的不可能状态。这些随机起始状态用于覆盖策略状态空间，不应把其样本数量当作自然开局的实战场数。

默认训练陪练从积极、均衡、谨慎三种风格中为每个座位固定抽样。`--bot normal` 配合 `--cfg` 可混入搜索型对手；`--roster all` 则所有座位使用指定搜索 AI。设置 `budget_us=0`，真实引擎与各座位随机流按 seed 独立，便于重复对局。

```powershell
cargo build --release --bin cabo-match-data --bin cabo-eval

# 第一版：18,000 场训练、4,000 场选模型、4,000 场独立测试
target/release/cabo-match-data.exe --games 18000 --seed 220000 --jobs 4 --out data/match_value/train.tsv
target/release/cabo-match-data.exe --games 4000 --seed 240000 --jobs 4 --out data/match_value/validation.tsv
target/release/cabo-match-data.exe --games 4000 --seed 260000 --jobs 4 --out data/match_value/test.tsv
python research/tools/train_match_value.py --train data/match_value/train.tsv --validation data/match_value/validation.tsv --test data/match_value/test.tsv --epochs 45 --out data/match_value/model-v1.bin --report research/reports/MATCH_VALUE_V1_LEARNING.json

# 显式开启学习价值；Normal 对照保持冻结策略
target/release/cabo-eval.exe --candidate hard --compare normal --opponents challenger --players 4 --blocks 40 --jobs 4 --seed 280000 --cfg budget_us=0,simulations=128,confirmation_samples=48,learned_value=true,value_model_path=data/match_value/model-v1.bin --compare-cfg budget_us=0,simulations=128,confirmation_samples=48 --out research/reports/value-eval.tsv

# 独立语言实现的预测一致性，以及完整搜索的信息隔离检查
$env:CABO_MATCH_VALUE_TEST='data/match_value/model-v2.bin'
cargo test --release --lib local_model_inference_parity_and_seat_symmetry -- --ignored
cargo test --release --lib learned_search_uses_only_legal_information -- --ignored
```

训练仅需 NumPy。训练器自检有限差分梯度，按整场比赛均衡损失，使用独立验证集挑 checkpoint，最终测试集不参与梯度或 checkpoint 选择。`--sort-seeds` 可将并行采集的文件按比赛 seed 排序；否则给定保存的 TSV 顺序可以重复训练，但重新并行采集时文件中的比赛顺序可能变化。

第二版对应训练 `320000..419999` 共 100,000 场、验证 `440000..451999` 共 12,000 场、测试 `460000..471999` 共 12,000 场；架构不变，训练 25 个 epoch。`480000..480799` 的 800 场混合 Normal 迁移测试独立于训练。

本机已有 `D:\miniconda3\envs\py311\python.exe`，PyTorch 2.9.1+cu128，可使用 RTX 3060 Laptop GPU（6 GB）。直接复用，不另建环境：

```powershell
D:/miniconda3/envs/py311/python.exe research/tools/train_match_value_gpu.py --train data/match_value/train-v2.tsv --validation data/match_value/validation-v2.tsv --test data/match_value/test-v2.tsv --epochs 25 --batch-size 512 --out data/match_value/model-v2-gpu.bin --report research/reports/MATCH_VALUE_V2_GPU_LEARNING.json
```

此次 GPU 训练耗时 137.8 秒；CUDA 与 NumPy 概率最大误差约 `2.16e-7`，导出模型也通过 Rust 推理检查。没有记录等条件 CPU 耗时，因此不声称实测速比。模拟比赛采集仍主要消耗 CPU。GPU 训练器支持 `--init-model` 与 `--extra-train/validation/test`，按整场比赛均衡采样两个策略域；验证集也按指定域权重选模型，初始模型作为 epoch 0 参与选择，独立测试不参与选择。

V3 是一次自我对弈价值迭代：全部座位使用 V2 搜索，`budget_us=0,simulations=64,confirmation_samples=16,validate_calls=true,min_cabo_success=0,avoid_cycles=true`。训练 `530000..531199` 共 1,200 场/7,989 行，验证 `540000..540299` 共 300 场/1,961 行，独立测试 `550000..550599` 共 600 场/3,975 行。三批全部成功结束，再检查完整 seed 范围、同场人数/胜负标签一致性与 SHA256，才用于训练。旧失败批次不参与。

V3 从 V2 开始，等权混合原快策略与自我对弈域，25 epoch、每 epoch 65,536 样本、batch 1,024、学习率 0.0003，按 seed 排序；GPU 训练耗时 12.8 秒。该采样量与 V2 重训不同，不能据此比较训练速度。

```powershell
D:/miniconda3/envs/py311/python.exe research/tools/train_match_value_gpu.py --train data/match_value/train-v2.tsv --validation data/match_value/validation-v2.tsv --test data/match_value/test-v2.tsv --extra-train data/match_value/search-train-v3.tsv --extra-validation data/match_value/search-validation-v3.tsv --extra-test data/match_value/search-test-v3.tsv --extra-weight 0.5 --init-model data/match_value/model-v2.bin --samples-per-epoch 65536 --batch-size 1024 --epochs 25 --lr 0.0003 --sort-seeds --out data/match_value/model-v3-gpu.bin --report research/reports/MATCH_VALUE_V3_GPU_LEARNING.json
```

研究开关均默认关闭：`root_racing` 用共同世界比较根动作，`racing_actions` 控制观察信息筛选的候选数；`blind_keep_evidence`、`behavioral_evidence`、`call_evidence` 使用公开行为的软似然；`validate_calls` 复核原快策略的普通宣告；`match_cabo` 允许可靠的整局赢家宣告；`probabilistic_reset` 是已出现回归的概率重置奖励消融。`min_cabo_success=0` 可以研究用整局价值选择宣告的路线，仍需独立确认其相对摸牌的收益。

在预先固定的最终四人独立对照中，V1 加 `validate_calls=true,min_cabo_success=0` 组合取得胜率 +4.5±3.9 个百分点、分差 -2.44±2.05 分（2,048 场，双方 128 轨迹/48 复核）。另有 2,048 场两人独立对照，胜率 +8.6±3.1 个百分点、分差 -4.27±2.65 分。这是整个组合的收益；两人纯网络消融 +0.4±3.1 个百分点、四人纯网络消融 -1.3±6.2，均未确认网络单独贡献。面对搜索型 Normal 尚无可靠优势，也不能直接当作 600ms 默认预算下的同幅收益。

本地网页试用：设置下列环境变量，再按现有启动/重启方式启动服务，房间选 Hard 即可。它保留 Hard 的 600ms/8,192 轨迹/96 复核和防循环，开启学习价值、普通宣告复核、门槛 0；Normal 不读取此变量。

```powershell
$env:CABO_HARD_VALUE_MODEL=(Resolve-Path 'data/match_value/model-v1.bin').Path
```

要恢复没有学习模型的 Hard，清除变量后重启服务：

```powershell
Remove-Item Env:CABO_HARD_VALUE_MODEL -ErrorAction SilentlyContinue
```

没有变量时不自动扫描本地模型，也不依赖任何训练产物。显式配置不存在/损坏模型会构造失败，不能悄悄回退为没有网络的 Hard。复现 TSV 对照时清除此变量，或明确覆盖完整策略配置，避免环境改变未写在参数覆盖中的默认值。

`root_racing` 是根动作搜索，不使用 `tree_depth` 优化后续自己的树节点；其他开关在两种搜索中均生效。保留这些研究参数是为了复现对照，不代表默认 Hard 启用它们。原始对局 TSV 和汇总报告位于 `reports/`；训练样本与权重只在本机 `data/match_value/`。

Hard 默认新增 `avoid_cycles=true`：只在可摸牌时，最近公开历史连续两次重复完整的成功单张弃牌交换循环，才摸牌打破循环。检查完整的玩家、槽位、公开旧点数与新点数；允许一次循环，也不干预摸牌、能力、失败交换或宣告之后的历史。该检查同时覆盖两种根搜索，完全不读取隐藏牌。Normal 默认关闭该开关。它解决真实自我对弈停滞，不应把结束能力改善说成普通对局胜率显著提高。
