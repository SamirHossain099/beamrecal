"""Run TTA methods over a continual stream of target scenarios and save one json per run.

    python scripts/run_stream.py --ckpt checkpoints/src_scenario32_gps-cam-radar_s0.pt \
        --stream scenario33 scenario34 scenario31 --methods source norm tent eata sar cotta t3a beamtta
    python scripts/run_stream.py --synthetic --ckpt checkpoints/src_syn32_gps-cam-radar_s0.pt \
        --stream syn33 syn34 syn31
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import load_scenario  # noqa: E402
from tta_beam.models import build_model  # noqa: E402
from tta_beam.stream import make_stream, run  # noqa: E402
from tta_beam.tta import METHODS as _TTA  # noqa: E402
from tta_beam.geo import attach_geo  # noqa: E402
from tta_beam.geo_methods import GEO_METHODS  # noqa: E402

METHODS = {**_TTA, **GEO_METHODS}


def _parse(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stream", nargs="+", default=["scenario33", "scenario34", "scenario31"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--sweep_every", type=int, default=0, help="P1 sweep period in frames (labels for calib/beamtta); 0 = none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--kw", nargs="*", default=[], help="method constructor overrides, e.g. jump_beams=10 jump_n=6")
    ap.add_argument("--device", default=None, help="cuda (default if available) or cpu")
    ap.add_argument("--max_frames", type=int, default=0, help="truncate each domain to this many frames (0 = all)")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(ck.get("arch", "plain"), ck["modalities"], cam_ch=ck["cam_ch"]).to(dev)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    assert not unexpected, unexpected
    if "prior.K_ref" in missing:          # checkpoints from before the MAP re-fit: K_ref := training K
        model.prior.K_ref.copy_(model.prior.K)
        missing.remove("prior.K_ref")
    assert not missing, missing
    model.eval()
    parts = make_stream([attach_geo(load_scenario(s, seed=a.seed + 1), s) for s in a.stream], a.stream)
    src = None
    if any(getattr(METHODS[m], "needs_source", False) for m in a.methods):
        ds = [attach_geo(load_scenario(s), s) for s in ck["source"]]
        src = {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}
    if a.max_frames:
        for _, ds in parts:
            ds.t = {k: v[: a.max_frames] for k, v in ds.t.items()}

    results = {"ckpt": a.ckpt, "source": ck["source"], "stream": [d for d, _ in parts], "bs": a.bs,
               "sweep_every": a.sweep_every, "seed": a.seed, "kw": a.kw, "device": dev, "max_frames": a.max_frames,
               "methods": {}}
    for name in a.methods:
        torch.manual_seed(a.seed)
        kw = {k: _parse(v) for k, v in (x.split("=", 1) for x in a.kw)}
        if getattr(METHODS[name], "needs_source", False):
            kw = {**kw, "src": src}
        method = METHODS[name](model, **kw)
        r = run(method, parts, a.bs, dev, sweep_every=a.sweep_every)
        results["methods"][name] = r
        per = "  ".join(f"{d}:{r[d]['top1']:.3f}" for d, _ in parts)
        print(f"{name:8s} top1 {r['mean']['top1']:.3f} top3 {r['mean']['top3']:.3f} dba {r['mean']['dba']:.3f} "
              f"ploss {r['mean']['ploss_db']:.2f}dB  {r['latency_ms_per_frame']:.3f}ms/frame  | {per}"
              + f"  | w3 {r['mean']['within3']:.2f} k3 {r['mean']['ploss_k3_db']:.2f}dB"
              + (f"  | resets {r['calibrator_resets']}" if "calibrator_resets" in r else ""))

    out = ROOT / "results" / f"stream_{'-'.join(a.stream)}_bs{a.bs}_K{a.sweep_every}_s{a.seed}{a.tag}_{int(time.time())}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
