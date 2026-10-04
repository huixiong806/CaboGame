"""拟合"分数上下文 → 胜率"，替代（或修正）搜索里那个不看分数上下文的 `sigmoid(领先/8)`。

输入（5 维，全部来自公开信息）：
  my_total, min_other_total, second_other_total, round_no, target
派生特征（可解释、低维，避免黑箱）：
  lead = min_other − my
  headroom = target − my
  lead_ratio = lead / max(headroom, 1)      # 领先相对于"离终点还有多远"
以及若干交互项。

输出：P(我最终获胜)，用逻辑回归（可解释、可查校准、可直接硬编码进 Rust）。
同时与现行解析估值 `sigmoid(lead/8)` 做校准对比。
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def load(path: str):
    raw = Path(path).read_bytes()
    (n,) = struct.unpack_from("<I", raw, 0)
    off = 4
    rows = np.zeros((n, 5), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        rows[i] = struct.unpack_from("<5f", raw, off)
        off += 20
        (y[i],) = struct.unpack_from("<f", raw, off)
        off += 4
    return rows, y


def feats(rows: np.ndarray) -> np.ndarray:
    my, mo, so, rnd, tgt = (rows[:, i] for i in range(5))
    lead = mo - my
    head = np.maximum(tgt - my, 1.0)
    lead2 = so - my
    return np.stack(
        [
            np.ones_like(my),
            lead / 10.0,
            lead2 / 10.0,
            head / 100.0,
            rnd / 10.0,
            (lead / head).clip(-3, 3),          # 领先相对于剩余路程
            (lead / 10.0) * (head / 100.0),     # 交互：领先 × 剩余路程
            np.maximum(lead, 0) / 10.0,         # 非对称：领先与落后不对称
            np.maximum(-lead, 0) / 10.0,
        ],
        axis=1,
    ).astype(np.float64)


NAMES = ["bias", "lead/10", "lead2/10", "headroom/100", "round/10",
         "lead/headroom", "lead×headroom", "lead+", "lead-"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/scorectx.bin")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.5)
    a = ap.parse_args()

    rows, y = load(a.data)
    print(f"样本 {len(y)}  正例 {y.mean():.3f}")

    # 按"局"划分做不到（没存局号）→ 用哈希分桶做近似的时间/随机切分
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_val = len(y) // 5
    va, tr = idx[:n_val], idx[n_val:]

    X = feats(rows)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    w = np.zeros(X.shape[1])
    for ep in range(a.epochs):
        z = Xtr @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xtr.T @ (p - ytr) / len(ytr)
        w -= a.lr * g
        if ep % 1000 == 0:
            pv = 1.0 / (1.0 + np.exp(-np.clip(Xva @ w, -30, 30)))
            print(f"  epoch {ep:5d}  验证 logloss {-np.mean(yva*np.log(pv+1e-9)+(1-yva)*np.log(1-pv+1e-9)):.4f}")

    pv = 1.0 / (1.0 + np.exp(-np.clip(Xva @ w, -30, 30)))
    ll = -np.mean(yva * np.log(pv + 1e-9) + (1 - yva) * np.log(1 - pv + 1e-9))
    base = -np.mean(yva * np.log(ytr.mean()) + (1 - yva) * np.log(1 - ytr.mean()))
    print(f"\n学习模型  验证 logloss {ll:.4f}  （常数基线 {base:.4f}）")
    print("系数：")
    for n, c in zip(NAMES, w):
        print(f"  {n:<16} {c:+.4f}")

    # 与现行解析估值对比：sigmoid(lead/8)
    lead_va = (rows[va, 1] - rows[va, 0])
    p_an = 1.0 / (1.0 + np.exp(-lead_va / 8.0))
    ll_an = -np.mean(yva * np.log(p_an + 1e-9) + (1 - yva) * np.log(1 - p_an + 1e-9))
    brier_an = np.mean((p_an - yva) ** 2)
    brier_ml = np.mean((pv - yva) ** 2)
    print(f"\n现行解析 sigmoid(lead/8)：logloss {ll_an:.4f}  Brier {brier_an:.4f}")
    print(f"学习模型                ：logloss {ll:.4f}  Brier {brier_ml:.4f}")

    # 校准对照（分箱）
    print("\n校准对照（验证集，按学习模型预测值分箱）：")
    print(f"{'p 区间':>14} {'n':>7} {'学习预测':>9} {'解析预测':>9} {'实际':>8}")
    for b in range(5):
        lo, hi = b / 5, (b + 1) / 5
        sel = (pv >= lo) & (pv < hi if b < 4 else pv <= hi)
        if sel.sum() == 0:
            continue
        print(
            f"[{lo:.1f},{hi:.1f})     {sel.sum():>7} {pv[sel].mean():>9.3f} "
            f"{p_an[sel].mean():>9.3f} {yva[sel].mean():>8.3f}"
        )

    # 关键情形对照：同样是领先 20 分，但分数位置不同
    print("\n关键情形（同样领先 20 分，分数位置不同）：")
    print(f"{'我的总分':>8} {'对手最低':>8} {'轮次':>5} {'学习 P(赢)':>11} {'解析 P(赢)':>11}")
    for my, mo, rnd in [(10, 30, 2), (50, 70, 3), (80, 100, 5), (95, 115, 6)]:
        r = np.array([[my, mo, mo + 5, rnd, 100.0]])
        pf = feats(r)
        pl = 1.0 / (1.0 + np.exp(-np.clip(pf @ w, -30, 30)))
        pa = 1.0 / (1.0 + np.exp(-(mo - my) / 8.0))
        print(f"{my:>8} {mo:>8} {rnd:>5} {pl[0]:>11.3f} {pa[0]:>11.3f}")

    # 导出成 Rust 可直接用的系数
    print("\n=== Rust 常量 ===")
    print("pub const SCORECTX_COEF: [f64; %d] = [%s];" % (len(w), ", ".join(f"{c:.6}" for c in w)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
