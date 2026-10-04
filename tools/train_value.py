"""价值网络 v3（残差回归）GPU 训练器。

数据：`cabo-train --mode value3 --dump data/value3_600.bin`
- 输入：**完整局面**特征（150 维，含所有座位真实手牌构成、未见池、弃牌堆、暴露度）
- 目标：**终局分差** `min_other_final - my_final`（连续、低方差）
- 结构：`预测分差 = lead_now + 网络(x)` —— 网络只学"从现在到终局的修正量"（解析基线免费）

评估口径（不是 Brier，而是决策需要的口径）：
- MAE / 相关系数，对比基线 `lead_now`（解析基线）与常数基线；
- **排序质量**：同一局面内不同候选的排序（这里用"同局内不同座位"的横向比较近似）。
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---- 段落拼接：每个导出器只替换自己那一段，避免互相截断 ----
MARKERS = ["/// 网络各层宽度", "/// 策略网络（候选打分）", "/// 价值网络 v3"]


def splice_section(path: Path, marker: str, new_text: str) -> str:
    s = path.read_text(encoding="utf-8") if path.exists() else ""
    i = s.find(marker)
    if i < 0:
        return (s.rstrip() + "\n\n" + new_text) if s.strip() else new_text
    nxt = len(s)
    for m in MARKERS:
        j = s.find(m, i + len(marker))
        if j >= 0:
            nxt = min(nxt, j)
    return s[:i] + new_text + s[nxt:]


def load_value3(path: str):
    raw = Path(path).read_bytes()
    off = 0
    (n_full,) = struct.unpack_from("<I", raw, off)
    off += 4
    (n_rows,) = struct.unpack_from("<I", raw, off)
    off += 4
    x = np.zeros((n_rows, n_full), dtype=np.float32)
    lead = np.zeros(n_rows, dtype=np.float32)
    game = np.zeros(n_rows, dtype=np.int64)
    seat = np.zeros(n_rows, dtype=np.int64)
    for i in range(n_rows):
        (lead[i],) = struct.unpack_from("<f", raw, off)
        off += 4
        (game[i],) = struct.unpack_from("<I", raw, off)
        off += 4
        (seat[i],) = struct.unpack_from("<I", raw, off)
        off += 4
        x[i] = np.frombuffer(raw, dtype=np.float32, count=n_full, offset=off)
        off += 4 * n_full
    (n_fin,) = struct.unpack_from("<I", raw, off)
    off += 4
    fin = {}
    for _ in range(n_fin):
        (g,) = struct.unpack_from("<I", raw, off)
        off += 4
        (s_,) = struct.unpack_from("<I", raw, off)
        off += 4
        (m,) = struct.unpack_from("<f", raw, off)
        off += 4
        fin[(g, s_)] = m
    y = np.array([fin[(int(game[i]), int(seat[i]))] for i in range(n_rows)], dtype=np.float32)
    return n_full, x, lead, y, game, seat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/value3_600.bin")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="src/ai/weights.rs")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_full, x, lead, y, game, seat = load_value3(a.data)
    print(f"device={dev}  样本 {len(y)}  特征 {n_full} 维  终局分差 mean={y.mean():.2f} sd={y.std():.2f}")

    gids = np.unique(game)
    rng = np.random.default_rng(0)
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * a.val_frac))
    val_g = set(gids[:n_val].tolist())
    vm = np.array([g in val_g for g in game])
    tr, va = np.where(~vm)[0], np.where(vm)[0]

    X = torch.tensor(x, device=dev)
    L = torch.tensor(lead, device=dev)
    Y = torch.tensor(y, device=dev)

    # 基线：直接用 lead_now 预测终局分差
    def report(name, pred):
        p = pred.detach().cpu().numpy()
        mae = np.abs(p[va] - y[va]).mean()
        base = np.abs(lead[va] - y[va]).mean()
        const = np.abs(y[tr].mean() - y[va]).mean()
        corr = np.corrcoef(p[va], y[va])[0, 1]
        print(f"{name:<28} 验证 MAE {mae:6.2f}  (lead 基线 {base:6.2f}，常数 {const:6.2f})  相关 {corr:+.3f}")

    report("解析基线 lead_now", L)

    class Net(nn.Module):
        def __init__(self, d, h, depth):
            super().__init__()
            layers, cur = [], d
            for _ in range(depth):
                layers += [nn.Linear(cur, h), nn.ReLU()]
                cur = h
            layers += [nn.Linear(cur, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = Net(n_full, a.hidden, a.depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.SmoothL1Loss()

    with torch.no_grad():
        report("网络(未训练)", L + model(X))

    best_mae, best_state = 1e18, None
    for ep in range(a.epochs):
        idx = rng.permutation(tr)
        tot, nb = 0.0, 0
        for s0 in range(0, len(idx), a.batch):
            b = idx[s0 : s0 + a.batch]
            pred = L[b] + model(X[b])
            loss = lossf(pred, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        with torch.no_grad():
            pv = (L[va] + model(X[va])).cpu().numpy()
            mae = np.abs(pv - y[va]).mean()
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep+1:4d}  train loss {tot/max(1,nb):7.3f}  验证 MAE {mae:6.2f}")
    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        report("学习修正（最佳）", L + model(X))

    if a.dry:
        return 0

    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    flat: list[float] = [float(len(layers))]
    sizes: list[int] = [n_full]
    for lin in layers:
        w = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()
        sizes.append(w.shape[0])
        flat += [float(w.shape[1]), float(w.shape[0])]
        flat += [float(v) for v in w.reshape(-1)]
        flat += [float(v) for v in b]
    out = Path(a.out)
    body = [
        f"/// 价值网络 v3：完整局面特征 → 终局分差的修正量（残差接在 lead_now 上）。\n"
        f"/// 训练：{len(y)} 样本（本轮结算后采样），验证 MAE {best_mae:.2f}。\n"
        f"pub const VALUE2_SIZES: &[usize] = &[{', '.join(str(s) for s in sizes)}];\n\n"
        f"pub const VALUE2_WEIGHTS: &[f32] = &["
    ]
    body.append(", ".join(repr(v) for v in flat))
    body.append("];\n")
    out.write_text(splice_section(out, "/// 价值网络 v3", "\n".join(body)), encoding="utf-8")
    print(f"权重已写入 {a.out}（{len(flat)} 个数，层宽 {sizes}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
