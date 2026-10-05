# 离线研究与归档

本目录集中保存 AI 研究、全部实验记录、未晋升方案和旧路线，供复现与后续研究使用。游戏介绍和当前默认配置见根目录 README 与 `models/README.md`。

- `reports/`：完整配对比赛、开发/留出结果、学习指标与审计，包括失败实验；原始 TSV/JSON 保留。
- `docs/`：各阶段路线、训练方法及策略设计文档。
- `bin/`：评估、对局模拟和训练数据采集的 Rust 入口。
- `tools/`：Python 训练、预测评估和配对汇总工具。
- `artifacts/`：仅本地保存的数据、其他模型、日志、预览构建与可玩发布包；Git 忽略整个目录。

最新实验审计见 [Hard 盲配与对手学习报告](reports/HARD_BLIND_AND_OPPONENT_RESULTS.md)。所有未通过独立验收的候选继续关闭；默认 Hard 使用仓库内已验收的价值模型与宣告组合。

早期原始记录中的 `data/`、`tools/`、`reports/` 分别对应现在的 `research/artifacts/data/`、`research/tools/`、`research/reports/`。原始记录包含当时路径及默认配置，不为迁移改写数据字段或校验值。复现旧无模型 Hard 时设置 `CABO_HARD_VALUE_MODEL=off`；R43–R56 还需关闭 `proven_reset_trade`。R57 的研究构建差异见相应报告。

从项目根目录运行研究工具，例如：

```powershell
cargo run --release --bin cabo-eval -- --candidate hard --compare normal --opponents challenger --blocks 20 --players 4 --cfg budget_us=0,simulations=256 --compare-cfg budget_us=0,simulations=256 --out research/artifacts/eval.tsv
python research/tools/summarize_eval.py research/artifacts/eval.tsv
```

研究模型和训练数据没有自动部署流程。改动只有在规则、信息隔离、合法完整对局及独立验收通过后才进入游戏默认配置。文档中各阶段实验结论属于相应版本的记录。
