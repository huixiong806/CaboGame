"""离线分析：策略网络的"置信度"能否预测它会不会和教师分歧？

动机：搜索的瓶颈是世界数（估值精度）。如果学生能在"简单决策"上以极低成本给出与教师
一致的答案，就可以把这些决策交给它（~8µs），把省下的时间预算全部投给"难决策"，
让搜索在关键处多跑 5~10 倍的世界。这条路成立的前提是：
**学生的 top-1 分差（margin）必须能预测"它会不会选错"。**
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from train_policy import Scorer, load_dump  # noqa: E402

import re
from pathlib import Path


def load_weights_from_rust(path: str):
    s = Path(path).read_text(encoding="utf-8")
    m = re.search(r"POLICY_SIZES: &\[usize\] = &\[([0-9,\s]+)\]", s)
    sizes = [int(x) for x in m.group(1).split(",") if x.strip()]
    m2 = re.search(r"POLICY_WEIGHTS: &\[f32\] = &\[(.*?)\];", s, re.S)
    flat = [float(x) for x in m2.group(1).replace("\n", " ").split(",") if x.strip()]
    return sizes, flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/policy_400.bin", "data/policy_900.bin"])
    ap.add_argument("--weights", default="src/ai/weights.rs")
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()

    sizes, flat = load_weights_from_rust(a.weights)
    print("网络层宽:", sizes)
    layers = []
    off = 1
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        off += 2
        w = np.array(flat[off : off + n_in * n_out], dtype=np.float32).reshape(n_out, n_in)
        off += n_in * n_out
        b = np.array(flat[off : off + n_out], dtype=np.float32)
        off += n_out
        layers.append((w, b))

    def forward(x: np.ndarray) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0.0)
        return h.reshape(-1)

    # 汇总所有数据集，按局划分
    all_states, all_cvs, all_labels, all_priors, all_games = [], [], [], [], []
    for path in a.data:
        n_state, n_cand, st, cv, lab, pri, gm = load_dump(path)
        all_states += st
        all_cvs += cv
        all_labels += lab
        all_priors += pri
        all_games += [g + 10_000_000 * len(all_games) if False else g for g in gm]
    print(f"决策样本 {len(all_states)}（{len(a.data)} 个数据集）")

    gids = np.array(sorted(set(all_games)))
    rng = np.random.default_rng(0)
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * a.val_frac))
    val_games = set(gids[:n_val].tolist())

    margins, correct, prior_ok = [], [], []
    for i in range(len(all_states)):
        if all_games[i] not in val_games:
            continue
        cvs = all_cvs[i]
        z = np.array([forward(np.concatenate([all_states[i], c]))[0] for c in cvs])
        order = np.argsort(-z)
        top1 = int(order[0])
        margin = float(z[order[0]] - z[order[1]]) if len(z) > 1 else 0.0
        margins.append(margin)
        correct.append(1 if top1 == all_labels[i] else 0)
        prior_ok.append(1 if all_priors[i] == all_labels[i] else 0)
    margins = np.array(margins)
    correct = np.array(correct)
    print(f"验证集决策 {len(margins)}，学生 top-1 命中 {correct.mean():.4f}，先验命中 {np.mean(prior_ok):.4f}")

    print("\n按置信度分箱（margin = top1 − top2 的 logit 差）：")
    print(f"{'分位':>12}  {'n':>6}  {'学生命中':>8}  {'该档占全部错误':>14}")
    qs = np.quantile(margins, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    total_err = (1 - correct).sum()
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (margins >= lo) & (margins <= hi if hi == qs[-1] else margins < hi)
        if sel.sum() == 0:
            continue
        err = (1 - correct[sel]).sum()
        print(
            f"[{lo:6.2f},{hi:6.2f})  {sel.sum():>6}  {correct[sel].mean():>8.4f}  "
            f"{100.0*err/max(1,total_err):>13.1f}%"
        )

    # 关键问题：如果只把"高置信"的那部分交给学生，覆盖多少决策、承担多少错误？
    print("\n门控模拟（阈值 = margin 分位）：")
    for frac in (0.5, 0.6, 0.7, 0.8, 0.9):
        thr = np.quantile(margins, 1 - frac)
        sel = margins >= thr
        cover = sel.mean()
        # 学生在这部分决策上的错误率 vs 教师（教师 100% 正确）
        err_rate = 1 - correct[sel].mean()
        print(
            f"  交给学生 {cover*100:5.1f}% 的决策（margin≥{thr:5.2f}）：学生错误率 {err_rate*100:5.2f}%"
            f"（随机抽同样比例的先验错误率 {1-np.mean(np.array(prior_ok)[sel])*1:.4f}）"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
