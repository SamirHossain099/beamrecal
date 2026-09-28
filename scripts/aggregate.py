"""Aggregate stream result files into mean +- std tables over seeds.

    python scripts/aggregate.py --tag _E1_anchored_K20 [--tag _E1_plain_K20] [--per_domain] [--md out.md]

Groups files by tag (substring of the file name), then by method; seeds are the `seed` field.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(tag):
    runs = defaultdict(dict)  # method -> seed -> result
    meta = None
    for f in sorted((ROOT / "results").glob(f"stream_*{tag}_*.json")):
        r = json.loads(f.read_text())
        meta = meta or {"stream": r["stream"], "bs": r["bs"], "sweep_every": r["sweep_every"], "source": r["source"]}
        for m, res in r["methods"].items():
            runs[m][r["seed"]] = res  # latest file wins per (method, seed)
    return meta, runs


def fmt(vals, digits=3):
    vals = np.asarray(vals, float)
    return f"{vals.mean():.{digits}f}" + (f" ± {vals.std(ddof=0):.{digits}f}" if len(vals) > 1 else "")


def table(meta, runs, per_domain=False):
    lines = [f"source {meta['source']} · stream {' → '.join(meta['stream'])} · batch {meta['bs']} · sweep every "
             f"{meta['sweep_every']} frames", ""]
    cols = ["method", "seeds", "top-1", "top-3", "DBA", "power loss (dB)", "ms/frame"]
    if per_domain:
        cols += [d.replace("scenario", "s") + " top-1" for d in meta["stream"]]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for m, by_seed in runs.items():
        rs = list(by_seed.values())
        row = [m, str(len(rs)), fmt([r["mean"]["top1"] for r in rs]), fmt([r["mean"]["top3"] for r in rs]),
               fmt([r["mean"]["dba"] for r in rs]), fmt([r["mean"]["ploss_db"] for r in rs], 2),
               fmt([r["latency_ms_per_frame"] for r in rs], 2)]
        if per_domain:
            row += [fmt([r[d]["top1"] for r in rs]) for d in meta["stream"]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--per_domain", action="store_true")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    out = []
    for tag in a.tag:
        meta, runs = load(tag)
        if meta is None:
            out.append(f"no results for tag {tag}")
            continue
        out.append(f"## {tag}\n\n" + table(meta, runs, a.per_domain) + "\n")
    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
