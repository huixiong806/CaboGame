# v1.0 发布版默认 AI 对战

本次测试衡量玩家实际使用的默认难度，不调整搜索预算、不启用实验开关、不筛选对局。三组各 64 个种子、双座位轮换，共 384 个完整大局；全部通过真实引擎的合法指令验证。

| 对战（前者 / 后者） | 整局数 | 胜率（前者 / 后者） | 平均分差（95% 区间） |
|---|---:|---:|---:|
| Hard / Normal | 128 | 57.8% / 42.2% | -2.41 [-8.46, +3.65] |
| Hard / Easy | 128 | 85.9% / 14.1% | -41.30 [-47.15, -35.44] |
| Normal / Easy | 128 | 82.8% / 17.2% | -42.48 [-49.02, -35.94] |

胜率为整局胜率份额，并列获胜各计半胜。分差为前者最终累计分减后者最终累计分。每个种子的两局先取平均，再以 64 个种子块作为独立单位；分差区间使用 Student-t（df=63，临界值 1.9983405425），不把同种子的两局视作独立样本。CLI 输出的分差区间使用渐近临界值 1.96，本表根据原始 TSV 重新计算。

| 前者 / 后者 | 前者胜率描述性 95% 区间 | 前者平均决策耗时 | 整组耗时 |
|---|---:|---:|---:|
| Hard / Normal | 45.6%–69.1% | 201.29ms | 626.1s |
| Hard / Easy | 75.4%–92.4% | 184.99ms | 878.1s |
| Normal / Easy | 71.8%–90.1% | 83.24ms | 655.0s |

胜率区间是按种子块计算的 t 区间与块数量 Wilson 包络的并集，截断到 [0,1]；因包含平局的分数份额，这属于描述性区间。Hard 对 Normal 的区间包含持平，当前样本不足以证明其显著领先。不同人数、规则、硬件和后台负载会改变结果；时间预算决定搜索量，因此同一种子也不保证逐次重跑完全一致。

## 配置与来源

- 平台：Windows x64，Intel Core i7-12700H，4 个并行比赛工作线程。
- 阈值 100、Cabo 惩罚 10；高牌双对与每人一次的 100→50 重置正常启用；从零累计分开始，完整大局结束。
- Easy：默认 300,000µs、最多 2,048 个世界，使用内嵌策略权重。
- Normal：默认 300,000µs、4,096 次模拟、48 次复核。
- Hard：默认 600,000µs、8,192 次模拟、96 次复核，自动加载随库冠军模型并使用默认 Cabo 复核与确定终局策略。
- 启动前清除子进程的 CABO_* 环境覆盖，命令行不传任何 cfg 覆盖。
- AI 只获得对应座位的 PlayerView；非法指令或大局超限会使测试失败，不使用保底动作掩盖问题。
- 源码基准：`0e0c5460c67ae6e6e488c99a6ce2c9e303d59378`，本地仅版本、界面与发布说明待提交；AI 和游戏源码校验和记录于协议。
- 模型 SHA256：`f41dedda1ab25df3b0f7692807724831a89dffdd9118fa05d55f8fb6ecd96fc8`。
- 评估程序 SHA256：`6ce9a3756bf00349d73ab766966745c75aed3dd4a4c1b2813a26c86441f1750e`；release 优化，静态 CRT。

## 复现与原始记录

```powershell
$env:CARGO_TARGET_DIR = 'research/artifacts/builds/portable'
$env:RUSTFLAGS = '-C target-feature=+crt-static'
cargo build --release --bin cabo-eval
Get-ChildItem Env:CABO_* | Remove-Item
./research/artifacts/builds/portable/release/cabo-eval.exe --candidate hard --opponents normal --players 2 --blocks 64 --jobs 4 --seed 2026100501 --out research/artifacts/hard-normal.tsv
./research/artifacts/builds/portable/release/cabo-eval.exe --candidate hard --opponents easy --players 2 --blocks 64 --jobs 4 --seed 2026100501 --out research/artifacts/hard-easy.tsv
./research/artifacts/builds/portable/release/cabo-eval.exe --candidate normal --opponents easy --players 2 --blocks 64 --jobs 4 --seed 2026100501 --out research/artifacts/normal-easy.tsv
```

[协议与源码校验和](V1_RELEASE_PROTOCOL.json) · [汇总 JSON](V1_RELEASE_SUMMARY.json)

- Hard / Normal：[种子块 TSV](V1_RELEASE_HARD_NORMAL.tsv)、[原始输出](V1_RELEASE_HARD_NORMAL.txt)。
- Hard / Easy：[种子块 TSV](V1_RELEASE_HARD_EASY.tsv)、[原始输出](V1_RELEASE_HARD_EASY.txt)。
- Normal / Easy：[种子块 TSV](V1_RELEASE_NORMAL_EASY.tsv)、[原始输出](V1_RELEASE_NORMAL_EASY.txt)。
