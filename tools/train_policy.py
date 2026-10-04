"""策略头（候选打分）GPU 训练器。

数据由 Rust 侧产出：
    cargo run --release --bin cabo-train -- --mode policy --games 400 --dump data/policy_400.bin

设计要点（见 DL_ROUTE.md）：
- 标签 = 教师（搜索）选中的候选下标 → **零发牌噪声**的监督信号；
- 输入 = 状态特征 ⊕ 候选特征（候选特征里已含"静态评分""是不是先验之选"两个**基线信号**）；
- 输出 = 每个候选一个 logit → 候选间 softmax → 列表交叉熵；
- 评估 = 与教师 top-1 一致率，**必须超过"启发式先验命中率"**才算有用。

训练完把权重写成 Rust 能读的 `src/ai/weights.rs`（`Mlp::import` 的格式）。
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


def load_dump(path: str):
    raw = Path(path).read_bytes()
    off = 0
    n_state, n_cand, n_dec = struct.unpack_from("<III", raw, off)
    off += 12
    states, cvs, labels, priors, games, counts = [], [], [], [], [], []
    for _ in range(n_dec):
        (nc,) = struct.unpack_from("<I", raw, off)
        off += 4
        (lab,) = struct.unpack_from("<I", raw, off)
        off += 4
        (pri,) = struct.unpack_from("<I", raw, off)
        off += 4
        (gm,) = struct.unpack_from("<I", raw, off)
        off += 4
        st = np.frombuffer(raw, dtype=np.float32, count=n_state, offset=off).copy()
        off += 4 * n_state
        cd = np.frombuffer(raw, dtype=np.float32, count=n_cand * nc, offset=off).copy()
        off += 4 * n_cand * nc
        states.append(st)
        cvs.append(cd.reshape(nc, n_cand))
        labels.append(lab)
        priors.append(-1 if pri == 0xFFFFFFFF else pri)
        games.append(gm)
        counts.append(nc)
    return n_state, n_cand, states, cvs, labels, priors, games


class Scorer(nn.Module):
    """把 (状态, 候选) 映射成打分；同一套权重对所有候选共享。"""

    def __init__(self, n_in: int, hidden: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, state, cands, mask=None):
        # state: [B, S]  cands: [B, C, K]  ->  [B, C]
        b, c, k = cands.shape
        x = torch.cat([state.unsqueeze(1).expand(b, c, state.shape[-1]), cands], dim=-1)
        z = self.net(x).squeeze(-1)
        if mask is not None:
            z = z.masked_fill(~mask, float("-inf"))
        return z


def make_batch(idx, states, cvs, labels, max_c, k):
    b = len(idx)
    cand = np.zeros((b, max_c, k), dtype=np.float32)
    mask = np.zeros((b, max_c), dtype=bool)
    lab = np.zeros(b, dtype=np.int64)
    for r, i in enumerate(idx):
        nc = cvs[i].shape[0]
        cand[r, :nc] = cvs[i]
        mask[r, :nc] = True
        lab[r] = labels[i]
        if nc > max_c:  # 不应发生
            raise RuntimeError("候选数超过 max_c")
    return cand, mask, lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/policy_400.bin"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2, help="按局划分的验证集比例")
    ap.add_argument("--out", default="src/ai/weights.rs")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_state, n_cand, states_l, cvs_l, labels, priors, games = 0, 0, [], [], [], [], []
    for gi, path in enumerate(a.data):
        ns, nc, st, cv, lb, pr, gm = load_dump(path)
        n_state, n_cand = ns, nc
        states_l += st
        cvs_l += cv
        labels += lb
        priors += pr
        games += [g + 10_000_000 * gi for g in gm]
    n_in = n_state + n_cand
    n_dec = len(states_l)
    states = torch.tensor(np.stack(states_l), device=dev)
    max_c = max(c.shape[0] for c in cvs_l)
    print(
        f"device={dev}  决策 {n_dec}  状态 {n_state} 维 + 候选 {n_cand} 维 = 输入 {n_in} 维，"
        f"平均候选 {np.mean([c.shape[0] for c in cvs_l]):.2f}，最多 {max_c}"
    )

    # ---- 按局划分（同局样本高度相关，必须整局切）----
    gids = np.array(sorted(set(games)))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * a.val_frac))
    val_games = set(gids[:n_val].tolist())
    tr = [i for i in range(n_dec) if games[i] not in val_games]
    va = [i for i in range(n_dec) if games[i] in val_games]
    print(f"按局划分：验证 {n_val}/{len(gids)} 局（{len(va)} 决策），训练 {len(tr)} 决策")

    def prior_acc(idx):
        n = sum(1 for i in idx if priors[i] >= 0)
        hit = sum(1 for i in idx if priors[i] == labels[i])
        return hit / max(1, n)

    print(f"启发式先验 top-1 命中率：验证 {prior_acc(va):.4f}  训练 {prior_acc(tr):.4f}")

    model = Scorer(n_in, a.hidden, a.depth).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数 {n_par}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss()

    def evaluate(idx, bs=512, mask_prior=False):
        """mask_prior=True 时把候选特征里最后两维（静态评分 / 是否先验之选）置零，
        用来量化"网络到底有多依赖这个拐杖"。"""
        model.eval()
        hit = 0
        with torch.no_grad():
            for s in range(0, len(idx), bs):
                chunk = idx[s : s + bs]
                cand, mask, lab = make_batch(chunk, states, cvs_l, labels, max_c, n_cand)
                if mask_prior:
                    cand[:, :, -2:] = 0.0
                cand = torch.tensor(cand, device=dev)
                mask = torch.tensor(mask, device=dev)
                lab = torch.tensor(lab, device=dev)
                z = model(states[chunk], cand, mask)
                hit += (z.argmax(dim=1) == lab).sum().item()
        model.train()
        return hit / max(1, len(idx))

    best_acc, best_state = -1.0, None
    for ep in range(a.epochs):
        order = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for s in range(0, len(order), a.batch):
            chunk = [tr[j] for j in order[s : s + a.batch]]
            cand, mask, lab = make_batch(chunk, states, cvs_l, labels, max_c, n_cand)
            cand_t = torch.tensor(cand, device=dev)
            mask_t = torch.tensor(mask, device=dev)
            lab_t = torch.tensor(lab, device=dev)
            z = model(states[chunk], cand_t, mask_t)
            loss = lossf(z, lab_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            acc = evaluate(va)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"epoch {ep+1:3d}  loss {tot/max(1,nb):.4f}  验证 top-1 {acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    tr_acc, va_acc = evaluate(tr), evaluate(va)
    print(
        f"最佳验证 top-1 = {va_acc:.4f}（先验 {prior_acc(va):.4f}，提升 {va_acc - prior_acc(va):+.4f}）；"
        f"训练集 top-1 = {tr_acc:.4f}（先验 {prior_acc(tr):.4f}）"
    )

    if not a.dry:
        pass
    ablation = evaluate(va, mask_prior=True)
    print(
        f"消融：把先验特征置零后，验证 top-1 = {ablation:.4f}"
        f"（完整 {va_acc:.4f}，先验自身 {prior_acc(va):.4f}）→ 净损失 {va_acc - ablation:+.4f}"
    )
    if a.dry:
        return 0

    # ---- 导出为 Rust `Mlp::import` 能读的扁平格式 ----
    # 格式：层数, 然后每层 [n_in, n_out, W..., b...]
    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    flat: list[float] = [float(len(layers))]
    sizes: list[int] = [n_in]
    for lin in layers:
        w = lin.weight.detach().cpu().numpy()  # [out, in]
        b = lin.bias.detach().cpu().numpy()
        sizes.append(w.shape[0])
        flat += [float(w.shape[1]), float(w.shape[0])]
        flat += [float(v) for v in w.reshape(-1)]
        flat += [float(v) for v in b]
    out = Path(a.out)
    body = [
        f"/// 策略网络（候选打分）：输入 = 状态特征 ⊕ 候选特征。\n"
        f"/// 训练：{n_dec} 决策（GPU/PyTorch），验证 top-1 {va_acc:.4f}（启发式先验 {prior_acc(va):.4f}）。\n"
        f"pub const POLICY_SIZES: &[usize] = &[{', '.join(str(s) for s in sizes)}];\n\n"
        f"/// 策略网络参数。\npub const POLICY_WEIGHTS: &[f32] = &["
    ]
    vals = ", ".join(repr(v) for v in flat)
    body.append(vals)
    body.append("];\n")
    out.write_text(splice_section(out, "/// 策略网络（候选打分）", "\n".join(body)), encoding="utf-8")
    print(f"权重已写入 {a.out}（{len(flat)} 个数，层宽 {sizes}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
