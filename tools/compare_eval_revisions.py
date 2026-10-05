"""Compare candidate revisions with matching paired seeds and unchanged controls.

Each input contains candidate then control per block. Time measurements may differ;
game outcomes and decision counts of the control must agree exactly.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from summarize_eval import interval


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader((r for r in f if not r.startswith("#")), delimiter="\t"))
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block"], []).append(r)
    assert len(blocks) >= 2 and all(len(p) == 2 for p in blocks.values())
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out")
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    assert old.keys() == new.keys(), "different seed blocks"
    same = ["seed", "bot", "games", "mean_gap", "win_share", "decisions",
            "calls", "successes", "resets", "high_pairs"]
    for k in old:
        assert old[k][0]["seed"] == new[k][0]["seed"]
        assert all(old[k][1][c] == new[k][1][c] for c in same), "control changed"
    win, win_ci = interval([100 * (float(new[k][0]["win_share"]) - float(old[k][0]["win_share"])) for k in old])
    gap, gap_ci = interval([float(new[k][0]["mean_gap"]) - float(old[k][0]["mean_gap"]) for k in old])
    result = {"old": a.old, "new": a.new, "blocks": len(old),
              "control_outcomes_and_decision_counts_identical": True,
              "win_delta_pp": win, "win_delta_95_ci_half": win_ci,
              "gap_delta": gap, "gap_delta_95_ci_half": gap_ci,
              "input_sha256": [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.old, a.new]]}
    if a.out:
        assert not Path(a.out).exists(), "refusing to overwrite revision comparison"
        Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
