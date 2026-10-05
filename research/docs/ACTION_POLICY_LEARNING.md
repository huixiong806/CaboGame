> 归档文档：保留当时的方案、默认配置与结论。当前版本见 [项目 README](../../README.md)，路径和复现说明见 [研究索引](../README.md)。

# 当前回合动作学习

这是上一轮累计分价值学习之外的新路线：用搜索在真实引擎中作决策，学习当前合法观察下的动作分布，再检验动作网络能否帮助搜索。当前实现是搜索策略蒸馏，不是已经收敛的端到端强化学习。Normal 默认不启用任何新开关。

## 信息与输入

每个候选动作有 64 个特征。前 38 个为当前行动者的知识摘要：阶段、人数、牌堆长度、自己手牌的期望点数/不确定性/14 档点数分布、对手点数摘要、累计分与重置资格、公开终局状态，以及弃牌顶或自己刚摸到的牌。后 26 个描述动作种类、选牌数量与匹配概率、所选牌的可知质量、重复牌概率、原动作评分、交换/间谍目标的可知质量和公开计分状态。

输入不含真实未知手牌、牌堆顺序、seed、座位编号或教师搜索采样的隐藏结果。特征只从 `Info` 的观察分布和公开字段取得。同一玩家观察不变时，重新采样未知世界不改变特征；模拟对手时只能使用那个对手的知识。已通过隐藏点数扰动、独立后验世界、完整搜索决策一致性检查。

小网络 `64→24→12→1` 有 1,873 个参数，`CABOPL01` 文件 7,500 字节；容量对照 `64→64→32→1` 有 6,273 个参数，`CABOPL02` 文件 25,100 字节。输出是原评分 logit 的学习修正。文件有固定头、长度及有限权重校验，Rust 原生推理；模型不随 Git 发布。

## 训练与分割

`cabo-policy-data` 采集独立完整回合，所有座位使用合法搜索教师。每个决策导出一组候选及教师最终采用的动作；初始偷看不作为网络样本。25% 零分开局、75% 随机公开计分状态，覆盖残局与重置资格；不将这种训练回合当作自然开局的完整比赛。

默认教师使用固定 128 条轨迹、32 次确认、普通宣告复核、门槛 0、防循环。本次加入冻结的整局价值 V1。教师特征抽取使用独立 RNG，不改变教师游戏随机流。动作缺失或非法、回合超限立即失败，整批只保留 `.incomplete`，成功完成后才生成正式 JSONL 和 manifest。

训练/验证/测试分别为 `700000..701199` 的 1,200 个回合、`710000..710299` 的 300 个回合、`720000..720299` 的 300 个回合；39,627 / 10,482 / 10,892 个决策组，427,978 / 110,361 / 111,266 个动作。三批全部完成，无非法命令。训练按回合均衡采样，以防长回合占据过大权重。

只有验证集交叉熵决定 checkpoint，测试不参与梯度或 checkpoint 选择。研究中多个模型复用第一批预测测试集，不能将它当作完全封存的最终验收；晋升依据另行固定新 seed 的独立完整实战。可知特征完全相同的动作无法学习任意槽位编号，因此训练标签在这些等价行中均分，诊断一致率也接受等价行。分数评分最大值的一致率基准不是完整 Challenger 策略，更不是棋力基准。

```powershell
cargo build --release --bin cabo-policy-data --bin cabo-eval
target/release/cabo-policy-data.exe --games 1200 --seed 700000 --jobs 4 --out data/action_policy/train-v1.jsonl --cfg learned_value=true,value_model_path=data/match_value/model-v1.bin
target/release/cabo-policy-data.exe --games 300 --seed 710000 --jobs 2 --out data/action_policy/validation-v1.jsonl --cfg learned_value=true,value_model_path=data/match_value/model-v1.bin
target/release/cabo-policy-data.exe --games 300 --seed 720000 --jobs 2 --out data/action_policy/test-v1.jsonl --cfg learned_value=true,value_model_path=data/match_value/model-v1.bin
D:/miniconda3/envs/py311/python.exe research/tools/train_action_policy.py --train data/action_policy/train-v1.jsonl --validation data/action_policy/validation-v1.jsonl --test data/action_policy/test-v1.jsonl --epochs 40 --out data/action_policy/model-v1.bin --report research/reports/ACTION_POLICY_V1_LEARNING.json
```

复用既有 conda/PyTorch/CUDA 环境与 RTX 3060，不需为游戏安装模型运行时。并行采集顺序不固定，训练器按 seed、step 排序。正式文件拒绝覆盖；重新采集须另选文件名。

## 搜索接入与消融

- `policy_model_path` 明确指定动作模型；开关启用而模型缺失/损坏时构造失败。
- `policy_only=true` 直接用网络行动，用于检验速度与误差积累。`policy_actions` 默认筛选原评分前 12 个，始终保留摸牌、宣告、弃置；设 64 比较完整候选。
- `rollout_policy=1/2/3` 分别替换所有玩家/对手/自身后续模拟行动。模拟玩家的动作使用该玩家的知识；网络策略没有原来的风格差异。
- `rollout_temperature=0` 保留旧 argmax；正值以该温度从 softmax 采样续招。每座位有独立随机流，独立复核的两个分支共享克隆后的流；这是默认关闭的研究项。
- `policy_prior=true` 在搜索节点用学习 logit 排序动作和初始化边价值，保留原 UCT、快策略 incumbent 和独立复核；不替换 rollout。
- `policy_puct=1` 改用动作概率引导树搜索分配，logit 温度 2、10% 均匀探索成分。它是独立实验；根动作 racing 只使用网络排序，PUCT 系数不改变 racing 分配。

默认所有新开关关闭。模型准确率、预测损失和搜索轨迹数量均不能代替完整比赛验收，训练教师的能力也不是学习策略能超过的上界证明。实战结果见 [本轮报告](../reports/HARD_POLICY_ITERATION_RESULTS.md)。

后续 Normal 专用对手教师为 32/16、无整局价值、无普通宣告复核，另采集 1,800/400/400 个训练/验证/测试回合。V5 及随机续招评估见 [后续报告](../reports/HARD_BLIND_AND_OPPONENT_RESULTS.md)。动作特征 55 与候选短名单固定使用训练时的评分基准；运行时整局价值和概率凑满分开关不会改变其含义，真实搜索效用仍照常使用这些配置。

训练器支持 `--extra-train/validation/test` 和 `--extra-weight`。新增域必须同样按完整回合分割，六个文件的 seed 都互不重叠；按指定权重均衡两个域的完整回合，混合验证损失选择 checkpoint，同时单独报告各域指标。正式模型也拒绝覆盖，以免修改正在评估的冻结权重。
