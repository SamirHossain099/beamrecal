"""Timing offset between logged vehicle positions and the beam labels, per scenario (results/position_lag.json).

For each lag L the position prior of Section 4.2 is fitted to pairs (position L frames earlier in the same
pass, beam label now); the fraction of frames within three beams of the fitted curve peaks at the lag that
best aligns the two logs. A peak at L > 0 means the logged positions lead the beam measurements.

    python scripts/position_lag.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import load_scenario  # noqa: E402
from tta_beam.models import PhysicsPrior  # noqa: E402

SCEN = ["scenario1", "scenario2", "scenario5", "scenario6", "scenario7", "scenario8", "scenario9",
        "scenario31", "scenario32", "scenario33", "scenario34", "scenario35"]
LAGS = range(-4, 7)


def lagged(d, L):
    """Pairs (gps[t-L], y[t]) within each pass, in time order."""
    g, y, s = d["gps"], d["y"], d["seq"]
    order = np.lexsort((np.arange(len(y)), s))
    g, y, s = g[order], y[order], s[order]
    if L == 0:
        return g, y
    if L > 0:
        ok = s[L:] == s[:-L]
        return g[:-L][ok], y[L:][ok]
    ok = s[:L] == s[-L:]
    return g[-L:][ok], y[:L][ok]


def main():
    out = {}
    for sc in SCEN:
        d = load_scenario(sc)
        w3 = {}
        for L in LAGS:
            g, y = lagged(d, L)
            g, y = torch.as_tensor(g), torch.as_tensor(y)
            p = PhysicsPrior().fit(g, y, fit_origin=True)
            w3[L] = float(((p.mean_beam(g).round().clamp(0, 63) - y).abs() <= 3).float().mean())
        best = max(w3, key=w3.get)
        out[sc] = {"within3_by_lag": w3, "best_lag": best}
        print(f"{sc:11s} best lag {best:+d}  " + " ".join(f"{L:+d}:{v:.3f}" for L, v in w3.items()))
    (ROOT / "results" / "position_lag.json").write_text(json.dumps(out, indent=1))
    print("wrote results/position_lag.json")


if __name__ == "__main__":
    main()
