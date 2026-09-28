"""Beam-measurement overhead vs received-power loss (the system-level figure).

Overhead per frame (in beam measurements) = 64 / K for a full P1 sweep every K frames (K = 0: none)
                                          + k for serving the best of the top-k predicted beams (k = 1: none).
Loss = mean received-power loss (dB) of the served beam vs the best beam, averaged over the stream's sites.

Reads stream result files (per-seed means) for the given tags; writes results/overhead_<name>.json and
results/overhead_<name>.png.

    python scripts/overhead_tradeoff.py --name trackB --tags _OH_trackB_geo _OH_trackB_nn
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    pts = defaultdict(list)  # (method, K, k) -> [loss per seed]
    for tag in a.tags:
        for f in (ROOT / "results").glob(f"stream_*{tag}_*.json"):
            r = json.loads(f.read_text())
            K = r["sweep_every"]
            for m, res in r["methods"].items():
                for k, key in ((1, "ploss_db"), (3, "ploss_k3_db"), (5, "ploss_k5_db")):
                    if key in res["mean"]:
                        pts[(m, K, k)].append(res["mean"][key])
    rows = []
    for (m, K, k), v in sorted(pts.items()):
        oh = (64.0 / K if K else 0.0) + (k if k > 1 else 0)
        rows.append({"method": m, "K": K, "k": k, "overhead_beams_per_frame": oh,
                     "loss_db": float(np.mean(v)), "loss_db_std": float(np.std(v)), "n": len(v)})
    out = ROOT / "results" / f"overhead_{a.name}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"{'method':22s} {'K':>4s} {'k':>2s} {'beams/frame':>11s} {'loss dB':>9s}")
    for r in sorted(rows, key=lambda r: (r["method"], r["overhead_beams_per_frame"])):
        print(f"{r['method']:22s} {r['K']:4d} {r['k']:2d} {r['overhead_beams_per_frame']:11.2f} {r['loss_db']:9.2f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for m in sorted({r["method"] for r in rows}):
            rr = sorted([r for r in rows if r["method"] == m], key=lambda r: r["overhead_beams_per_frame"])
            # Pareto front of this method's operating points
            front, best = [], np.inf
            for r in rr:
                if r["loss_db"] < best:
                    front.append(r)
                    best = r["loss_db"]
            ax.plot([r["overhead_beams_per_frame"] for r in front], [r["loss_db"] for r in front], "o-", ms=3, label=m)
        ax.set_xscale("symlog", linthresh=0.5, linscale=0.5)
        ax.set_xlim(-0.05, 80)
        ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20, 64])
        ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20", "64\n(exhaustive)"])
        ax.axvline(64, color="0.6", ls=":", lw=1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("beam measurements per frame (sweeps + top-k refinement)")
        ax.set_ylabel("received-power loss (dB)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(ROOT / "results" / f"overhead_{a.name}.png", dpi=150)
        print("wrote", f"results/overhead_{a.name}.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
