"""Model-free look at the data before any learning.

For each cached scenario: sample count, sequences, beam histogram entropy, GPS extent. Then a
concept-shift test that needs no neural network: a k-NN map from GPS position (relative to the BS) to
beam, fit on one scenario and applied to every other. If position -> beam is the same function at two
sites, cross-scenario k-NN accuracy stays high; if the arrays face differently, it collapses even though
positions overlap. Writes results/sanity_knn.json and results/sanity_gps_beam.png.

    python scripts/sanity_check.py --scenarios scenario1 scenario2 scenario31 scenario32 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import CACHE, load_scenario  # noqa: E402


def knn_predict(train_x, train_y, test_x, k=5):
    d = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    votes = train_y[nn]
    return np.array([np.bincount(v, minlength=64).argmax() for v in votes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--out", default="sanity_knn.json", help="file name under results/")
    a = ap.parse_args()
    names = a.scenarios or sorted(p.stem for p in CACHE.glob("scenario*.npz"))
    data = {n: load_scenario(n) for n in names}

    print(f"{'scenario':11s} {'N':>6s} {'seqs':>5s} {'beam-entropy':>12s}  gps extent (m, E x N)")
    for n, d in data.items():
        h = np.bincount(d["y"], minlength=64) / len(d["y"])
        ent = -(h[h > 0] * np.log2(h[h > 0])).sum()
        g = d["gps"] * 30
        print(f"{n:11s} {len(d['y']):6d} {len(np.unique(d['seq'])):5d} {ent:12.2f}  "
              f"[{g[:, 0].min():.0f},{g[:, 0].max():.0f}] x [{g[:, 1].min():.0f},{g[:, 1].max():.0f}]")

    # k-NN transfer matrix on GPS -> beam (within-scenario uses a sequence-wise 80/20 split)
    from tta_beam.data import split_by_sequence
    mat = {}
    print("\nGPS k-NN top-1 (rows: fit on, cols: applied to). Diagonal = held-out sequences of the same scenario.")
    print(f"{'':11s}" + "".join(f"{c[8:]:>8s}" for c in names))
    for src in names:
        tr, va = split_by_sequence(data[src], 0.8, 0)
        row = {}
        for tgt in names:
            if tgt == src:
                pred = knn_predict(data[src]["gps"][tr], data[src]["y"][tr], data[src]["gps"][va])
                y = data[src]["y"][va]
            else:
                pred = knn_predict(data[src]["gps"], data[src]["y"], data[tgt]["gps"])
                y = data[tgt]["y"]
            d = pred.astype(int) - y.astype(int)
            row[tgt] = {"top1": float((pred == y).mean()), "mean_abs_dbeam": float(np.abs(d).mean()),
                        "median_dbeam": float(np.median(d)), "within3": float((np.abs(d) <= 3).mean()),
                        "within3_after_median_shift": float((np.abs(d - np.median(d)) <= 3).mean())}
        mat[src] = row
        print(f"{src:11s}" + "".join(f"{row[c]['top1']:8.3f}" for c in names))
    print("\nmedian signed beam offset (pred - true) / frac within +-3 / frac within +-3 after removing the offset")
    for src in names:
        print(f"{src:11s}" + "".join(f"  {mat[src][c]['median_dbeam']:+4.0f}/{mat[src][c]['within3']:.2f}/"
                                     f"{mat[src][c]['within3_after_median_shift']:.2f}" for c in names))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / a.out).write_text(json.dumps(mat, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = min(len(names), 5)
        rows = int(np.ceil(len(names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
        for ax, n in zip(axes.flat, names):
            g = data[n]["gps"] * 30
            sc = ax.scatter(g[:, 0], g[:, 1], c=data[n]["y"], s=3, cmap="twilight", vmin=0, vmax=63)
            ax.scatter([0], [0], marker="^", c="k", s=60)
            ax.set_title(n)
            ax.set_aspect("equal")
        for ax in axes.flat[len(names):]:
            ax.axis("off")
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="best beam")
        fig.savefig(ROOT / "results" / "sanity_gps_beam.png", dpi=120)
        print("\nwrote results/sanity_gps_beam.png")
    except ImportError:
        print("matplotlib not installed; skipped figure")


if __name__ == "__main__":
    main()
