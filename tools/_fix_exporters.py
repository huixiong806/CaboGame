"""让两个导出器**只替换自己那一段**，不再截断文件（避免互相删掉常量）。"""
from pathlib import Path

HELPER = '''

# ---- 段落拼接：每个导出器只替换自己那一段，避免互相截断 ----
MARKERS = ["/// 网络各层宽度", "/// 策略网络（候选打分）", "/// 价值网络 v3"]


def splice_section(path: Path, marker: str, new_text: str) -> str:
    s = path.read_text(encoding="utf-8") if path.exists() else ""
    i = s.find(marker)
    if i < 0:
        return (s.rstrip() + "\\n\\n" + new_text) if s.strip() else new_text
    nxt = len(s)
    for m in MARKERS:
        j = s.find(m, i + len(marker))
        if j >= 0:
            nxt = min(nxt, j)
    return s[:i] + new_text + s[nxt:]
'''

# ---- train_policy.py ----
p = Path("tools/train_policy.py")
s = p.read_text(encoding="utf-8")
if "splice_section" not in s:
    s = s.replace("def load_dump(path: str):", HELPER.strip() + "\n\n\ndef load_dump(path: str):", 1)
    old = '''    out = Path(a.out)
    head = out.read_text(encoding="utf-8") if out.exists() else ""
    cut = head.find("/// 策略网络")
    head = head[:cut] if cut >= 0 else head
    if head and not head.endswith("\\n"):
        head += "\\n"
    body = [head.rstrip("\\n")] if head.strip() else []
    body.append(
        f"/// 策略网络（候选打分）：输入 = 状态特征 ⊕ 候选特征。\\n"'''
    new = '''    out = Path(a.out)
    body = [
        f"/// 策略网络（候选打分）：输入 = 状态特征 ⊕ 候选特征。\\n"'''
    assert old in s, "train_policy 截断段没找到"
    s = s.replace(old, new, 1)
    s = s.replace(
        '    out.write_text("\\n".join(body), encoding="utf-8")',
        '    out.write_text(splice_section(out, "/// 策略网络（候选打分）", "\\n".join(body)), encoding="utf-8")',
        1,
    )
p.write_text(s, encoding="utf-8")

# ---- train_value.py ----
v = Path("tools/train_value.py")
vs = v.read_text(encoding="utf-8")
if "splice_section" not in vs:
    vs = vs.replace("def load_value3(path: str):", HELPER.strip() + "\n\n\ndef load_value3(path: str):", 1)
    old = '''    out = Path(a.out)
    head = out.read_text(encoding="utf-8") if out.exists() else ""
    for marker in ("/// 价值网络 v3", "/// 策略网络（候选打分）", "/// 网络各层宽度"):
        cut = head.find(marker)
        if cut >= 0:
            head = head[:cut]
            break
    body = [head.rstrip("\\n")] if head.strip() else []
    body.append(
        f"/// 价值网络 v3'''
    new = '''    out = Path(a.out)
    body = [
        f"/// 价值网络 v3'''
    assert old in vs, "train_value 截断段没找到"
    vs = vs.replace(old, new, 1)
    vs = vs.replace(
        '    out.write_text("\\n".join(body), encoding="utf-8")',
        '    out.write_text(splice_section(out, "/// 价值网络 v3", "\\n".join(body)), encoding="utf-8")',
        1,
    )
v.write_text(vs, encoding="utf-8")
print("policy splice:", "splice_section" in s, "| value splice:", "splice_section" in vs)
