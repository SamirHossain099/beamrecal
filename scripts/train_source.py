"""Train a source beam predictor on one scenario (or several) and save a checkpoint.

    python scripts/train_source.py --source scenario32 --modalities gps cam radar
    python scripts/train_source.py --synthetic --source syn32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import BeamSet, load_scenario, split_by_sequence  # noqa: E402
from tta_beam.metrics import Meter  # noqa: E402
from tta_beam.models import build_model  # noqa: E402


def concat(ds):
    return {k: np.concatenate([x[k] + (100000 * i if k == "seq" else 0) for i, x in enumerate(ds)]) for k in ds[0]}


def soft_targets(y, dev, sigma=1.0):
    """Gaussian over neighbouring beams, as the 2022 challenge winners did to match the DBA score."""
    grid = torch.arange(64, device=dev).float()
    t = torch.exp(-(grid[None] - y[:, None].float()) ** 2 / (2 * sigma**2))
    return t / t.sum(1, keepdim=True)


def fit(d, modalities, epochs=30, bs=64, lr=1e-3, seed=0, dev="cuda", verbose=True, aug=True, arch="plain"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = split_by_sequence(d, 0.8, seed)
    tr_dl = torch.utils.data.DataLoader(BeamSet(d, tr), batch_size=bs, shuffle=True, drop_last=True)
    va_dl = torch.utils.data.DataLoader(BeamSet(d, va), batch_size=256)
    model = build_model(arch, modalities, cam_ch=d["cam"].shape[1]).to(dev)
    if arch == "anchored":  # physics prior fitted once on the source labels, before the residual is trained
        model.prior.fit(torch.as_tensor(d["gps"][tr]), torch.as_tensor(d["y"][tr]))
        if verbose:
            print(f"prior: c0 {model.prior.c0.item():.1f} K {model.prior.K.item():.1f} "
                  f"theta {np.degrees(model.prior.theta.item()):.1f} deg", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    res = None
    for ep in range(epochs):
        model.train()
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            if aug:
                b = augment(b)
            loss = -(soft_targets(b["y"], dev) * F.log_softmax(model(b), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (ep % 10 == 9 or ep == epochs - 1):
            res = evaluate(model, va_dl, dev)
            print(f"ep {ep + 1:3d}  loss {loss.item():.3f}  val top1 {res['top1']:.3f} top3 {res['top3']:.3f} "
                  f"dba {res['dba']:.3f}", flush=True)
    res = res or evaluate(model, va_dl, dev)
    model.eval()
    return model, res


def augment(b):
    """Light train-time augmentation: brightness/contrast jitter and pixel noise on images, GPS jitter."""
    out = dict(b)
    if "cam" in b:
        n = b["cam"].shape[0]
        gain = torch.empty(n, 1, 1, 1, device=b["cam"].device).uniform_(0.7, 1.3)
        bias = torch.empty(n, 1, 1, 1, device=b["cam"].device).uniform_(-0.1, 0.1)
        out["cam"] = (b["cam"] * gain + bias + 0.02 * torch.randn_like(b["cam"])).clamp(0, 1)
    if "gps" in b:
        out["gps"] = b["gps"] + 0.01 * torch.randn_like(b["gps"])  # 30 cm
    return out


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    m = Meter()
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        m.update(model(b), b["y"], b["pwr"])
    return m.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", default=["scenario32"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--modalities", nargs="+", default=["gps", "cam", "radar"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arch", default="plain", choices=["plain", "anchored"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = concat([load_scenario(s, seed=a.seed) for s in a.source])
    model, res = fit(d, a.modalities, a.epochs, a.bs, a.lr, a.seed, dev, arch=a.arch)

    tag = "" if a.arch == "plain" else f"_{a.arch}"
    out = Path(a.out or ROOT / "checkpoints" / f"src_{'+'.join(a.source)}_{'-'.join(a.modalities)}{tag}_s{a.seed}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "modalities": a.modalities, "cam_ch": d["cam"].shape[1],
                "source": a.source, "val": res, "arch": a.arch}, out)
    print("saved", out, json.dumps(res))


if __name__ == "__main__":
    main()
