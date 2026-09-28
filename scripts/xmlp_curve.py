"""Pinhole camera map against an MLP on the same camera coordinate, over sweep intervals (results/xmlp_curve.json).

Reads the `_X1_*_xmlp` runs (camera map with and without sweep refits, the MLP with and without sweep
fine-tuning; fine-tuning rate tuned at a sweep every 20 frames) and the `_X2_*_xmlp_lowlr` runs (the MLP
fine-tuned at a lower rate, which suits sparse sweeps better). For each interval the MLP row takes the
better of the two settings by mean power loss, so every choice made on the test stream favors the control,
not the camera map. Three seeds per point.

    python scripts/xmlp_curve.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
STREAMS = {"track_b": "trackB", "cross_unit": "crossunit"}
KS = (10, 20, 50, 100, 200)
METRICS = ("top1", "dba", "ploss_db")


def runs(tag, K, method):
    out = []
    for f in sorted(R.glob(f"stream_*_K{K}_s*{tag}_*.json")):
        m = json.loads(f.read_text())["methods"]
        if method in m:
            out.append(m[method]["mean"])
    return out


def stats(rs):
    return {**{x: float(np.mean([r[x] for r in rs])) for x in METRICS},
            **{x + "_std": float(np.std([r[x] for r in rs])) for x in METRICS}, "seeds": len(rs)}


def main():
    out = []
    for stream, short in STREAMS.items():
        for K in KS:
            for method in ("sense-geo", "xmlp", "sense-geo+calib"):
                rs = runs(f"_X1_{short}_xmlp", K, method)
                assert len(rs) == 3, (stream, K, method, len(rs))
                out.append({"stream": stream, "K": K, "method": method, **stats(rs)})
            cands = {"ft_lr from K=20 tuning": runs(f"_X1_{short}_xmlp", K, "xmlp+ft"),
                     "ft_lr=1e-4": runs(f"_X2_{short}_xmlp_lowlr", K, "xmlp+ft")}
            cands = {k: v for k, v in cands.items() if len(v) == 3}
            best = min(cands, key=lambda k: np.mean([r["ploss_db"] for r in cands[k]]))
            out.append({"stream": stream, "K": K, "method": "xmlp+ft", "setting": best, **stats(cands[best])})
    (R / "xmlp_curve.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"{r['stream']:10s} K={r['K']:3d} {r['method']:16s} loss {r['ploss_db']:.2f} ± {r['ploss_db_std']:.2f}  "
              f"top1 {r['top1']:.3f}  DBA {r['dba']:.3f}  {r.get('setting', '')}")
    print("wrote results/xmlp_curve.json")


if __name__ == "__main__":
    main()
