"""Zero-label transfer of a learned map on the camera coordinate (control for Figure 3).

For every pair of Testbed 1 scenarios, an MLP on the horizontal image coordinate of the served vehicle
(the `xmlp` method of geo_methods.py, same input as the pinhole camera map) is trained on the row scenario
and applied to the column scenario with no target labels. Three seeds. Writes results/xmlp_transfer.json
with the same layout as results/fig3_camera_transfer.json, which holds the pinhole map's matrix.

    python scripts/xmlp_transfer.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import load_scenario  # noqa: E402
from tta_beam.geo import load_cam_x  # noqa: E402
from tta_beam.geo_methods import XMLP  # noqa: E402

SCEN = ["scenario1", "scenario2", "scenario5", "scenario7", "scenario6", "scenario8", "scenario9"]
BS1 = ["scenario1", "scenario2", "scenario5", "scenario7"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    host = torch.nn.Linear(1, 1).to(dev)            # XMLP only reads the device from the model it is given
    D = {}
    for s in SCEN:
        x, y = load_cam_x(s), load_scenario(s)["y"]
        ok = np.isfinite(x)
        D[s] = (x[ok].astype(np.float32), y[ok])
    W3 = np.zeros((3, len(SCEN), len(SCEN)))
    T1 = np.zeros_like(W3)
    for seed in range(3):
        for i, a in enumerate(SCEN):
            torch.manual_seed(seed)
            m = XMLP(host, src={"cam_x": D[a][0], "y": D[a][1]})
            for j, b in enumerate(SCEN):
                pr = m.step({"cam_x": torch.as_tensor(D[b][0])}).argmax(1).cpu().numpy()
                W3[seed, i, j] = (np.abs(pr - D[b][1]) <= 3).mean()
                T1[seed, i, j] = (pr == D[b][1]).mean()
    off = [(i, j) for i, a in enumerate(SCEN) for j, b in enumerate(SCEN) if a in BS1 and b in BS1 and a != b]
    out = {"scenarios": SCEN, "within3": W3.mean(0).tolist(), "within3_std": W3.std(0).tolist(),
           "top1": T1.mean(0).tolist(), "seeds": 3,
           "bs1_offdiag_within3": [float(W3.mean(0)[i, j]) for i, j in off],
           "note": "MLP on the camera x-coordinate, trained on row, applied to column, no target labels; "
                   "annotated UE boxes as in fig3_camera_transfer.json"}
    (ROOT / "results" / "xmlp_transfer.json").write_text(json.dumps(out, indent=1))
    pin = json.loads((ROOT / "results" / "fig3_camera_transfer.json").read_text())
    P = np.array(pin["within3"])
    print("within 3 beams, BS1 off-diagonal:  MLP %.2f-%.2f   pinhole %.2f-%.2f" % (
        min(W3.mean(0)[i, j] for i, j in off), max(W3.mean(0)[i, j] for i, j in off),
        min(P[i, j] for i, j in off), max(P[i, j] for i, j in off)))
    print("cross-unit (row 1 -> 8, 6):  MLP %.2f, %.2f   pinhole %.2f, %.2f" % (
        W3.mean(0)[0, 5], W3.mean(0)[0, 4], P[0, 5], P[0, 4]))
    print("wrote results/xmlp_transfer.json")


if __name__ == "__main__":
    main()
