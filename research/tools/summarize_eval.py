"""Summarize paired cabo-eval TSVs, treating seed blocks as independent units.

Two rows per block are candidate then compare, including when bot IDs are equal.
Usage: python research/tools/summarize_eval.py research/reports/HARD_*.tsv
"""
import csv
import glob
import math
import statistics
import sys
from pathlib import Path


def interval(values):
    n = len(values)
    if n < 2:
        raise ValueError("need at least two independent blocks")
    # Same Student-t critical values as cabo-eval, conservative above 30 blocks.
    t = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
         2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
         2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
         2.060, 2.056, 2.052, 2.048, 2.045]
    critical = t[n-1] if n <= 30 else (2.0 if n < 61 else 1.96)
    return statistics.mean(values), critical * statistics.stdev(values) / math.sqrt(n)


def summarize(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
    blocks = {}
    for row in rows:
        blocks.setdefault(row["block"], []).append(row)
    if not blocks or any(len(pair) != 2 or pair[0]["seed"] != pair[1]["seed"] for pair in blocks.values()):
        raise ValueError(f"{path}: expected two paired rows per block")
    pairs = list(blocks.values())
    win, win_ci = interval([100*(float(a["win_share"])-float(b["win_share"])) for a,b in pairs])
    gap, gap_ci = interval([float(a["mean_gap"])-float(b["mean_gap"]) for a,b in pairs])
    means = [statistics.mean(float(p[i]["win_share"]) for p in pairs)*100 for i in [0,1]]
    us = [sum(int(p[i]["decision_us"]) for p in pairs)/sum(int(p[i]["decisions"]) for p in pairs)/1000 for i in [0,1]]
    games = sum(int(p[0]["games"])+int(p[1]["games"]) for p in pairs)
    return f"| {Path(path).stem} | {games} | {means[0]:.1f} / {means[1]:.1f} | {win:+.1f} ± {win_ci:.1f} | {gap:+.2f} ± {gap_ci:.2f} | {us[0]:.2f} / {us[1]:.2f} |"


if __name__ == "__main__":
    print("| Experiment | Total games | Win % | Δ win pp, 95% CI | Δ gap, 95% CI | Mean decision ms |")
    print("|---|---:|---:|---:|---:|---:|")
    paths = sorted({p for arg in sys.argv[1:] for p in glob.glob(arg)})
    if not paths:
        raise SystemExit("no input files")
    for path in paths:
        print(summarize(path))
