"""Upper bound for boresight calibration: pick theta per target scenario using ALL its labels.

    python scripts/oracle_boresight.py --ckpt checkpoints/src_scenario32_gps-cam_s0.pt --targets scenario33 ...

Prints, per target: source top-1/DBA (theta = 0), oracle theta (deg), and top-1 / within-3 / DBA at it.
Also the same with the theta chosen from only the first `--n_labels` labels (a cheap few-label estimate).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import BeamSet, load_scenario  # noqa: E402
from tta_beam.metrics import Meter  # noqa: E402
from tta_beam.models import build_model  # noqa: E402


@torch.no_grad()
def evaluate(model, d, theta, dev, idx=None):
    model.enc["gps"].theta.fill_(float(theta))
    ds = BeamSet(d, idx)
    m = Meter()
    for b in torch.utils.data.DataLoader(ds, batch_size=1024):
        b = {k: v.to(dev) for k, v in b.items()}
        m.update(model(b), b["y"], b["pwr"])
    r = m.result()
    return r


def search(model, d, dev, idx=None):
    coarse = np.deg2rad(np.arange(-180, 180, 4.0))
    best = max(coarse, key=lambda t: evaluate(model, d, t, dev, idx)["top1"])
    fine = best + np.deg2rad(np.arange(-4, 4.01, 0.5))
    return max(fine, key=lambda t: evaluate(model, d, t, dev, idx)["top1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--n_labels", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(ck.get("arch", "plain"), ck["modalities"], cam_ch=ck["cam_ch"]).to(dev)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    assert not unexpected, unexpected
    if "prior.K_ref" in missing:          # checkpoints from before the MAP re-fit: K_ref := training K
        model.prior.K_ref.copy_(model.prior.K)
        missing.remove("prior.K_ref")
    assert not missing, missing
    model.eval()
    print(f"source {ck['source']} {ck['modalities']}  within-site val top1 {ck['val']['top1']:.3f} dba {ck['val']['dba']:.3f}")
    print(f"{'target':10s} | {'src top1':>8s} {'dba':>5s} | {'oracle deg':>10s} {'top1':>6s} {'w3':>5s} {'dba':>5s} | "
          f"{a.n_labels:d}-label deg {'top1':>6s} {'dba':>5s}")
    for t in a.targets:
        d = load_scenario(t)
        r0 = evaluate(model, d, 0.0, dev)
        th = search(model, d, dev)
        r = evaluate(model, d, th, dev)
        # few-label: first n sweep-like labels spread over the first pass(es)
        idx = np.arange(0, min(len(d["y"]), 20 * a.n_labels), 20)[: a.n_labels]
        th_few = search(model, d, dev, idx)
        rf = evaluate(model, d, th_few, dev)
        w3 = None
        with torch.no_grad():
            model.enc["gps"].theta.fill_(float(th))
            ds = BeamSet(d)
            pred = torch.cat([model({k: v.to(dev) for k, v in b.items()}).argmax(1).cpu()
                              for b in torch.utils.data.DataLoader(ds, batch_size=1024)])
            w3 = ((pred - torch.as_tensor(d["y"])).abs() <= 3).float().mean().item()
        print(f"{t:10s} | {r0['top1']:8.3f} {r0['dba']:5.2f} | {np.degrees(th):+10.1f} {r['top1']:6.3f} {w3:5.2f} "
              f"{r['dba']:5.2f} | {np.degrees(th_few):+12.1f} {rf['top1']:6.3f} {rf['dba']:5.2f}")
    model.enc["gps"].theta.fill_(0.0)


if __name__ == "__main__":
    main()
