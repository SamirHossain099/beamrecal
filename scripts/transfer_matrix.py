"""E0: source-only transfer matrix. Train on each scenario, test on every scenario, per modality set.

    python scripts/transfer_matrix.py --scenarios scenario32 scenario33 scenario34 scenario31 scenario35 \
        --modality_sets gps cam radar gps+cam+radar --seeds 0 1 2

Writes results/transfer_<tag>.json and prints one top-1 matrix per modality set (mean over seeds).
Diagonal = held-out sequences of the training scenario.
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
sys.path.insert(0, str(ROOT / "scripts"))
from tta_beam.data import BeamSet, load_scenario, split_by_sequence  # noqa: E402
from train_source import evaluate, fit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--modality_sets", nargs="+", default=["gps", "cam", "gps+cam"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = {n: load_scenario(n) for n in a.scenarios}
    loaders = {n: torch.utils.data.DataLoader(BeamSet(d), batch_size=256) for n, d in data.items()}

    out = {"scenarios": a.scenarios, "seeds": a.seeds, "epochs": a.epochs, "results": {}}
    for mset in a.modality_sets:
        mods = mset.split("+")
        res = {}  # src -> tgt -> metric -> list over seeds
        for src in a.scenarios:
            for seed in a.seeds:
                t = time.time()
                model, _ = fit(data[src], mods, a.epochs, seed=seed, dev=dev, verbose=False)
                for tgt in a.scenarios:
                    if tgt == src:
                        _, va = split_by_sequence(data[src], 0.8, seed)
                        r = evaluate(model, torch.utils.data.DataLoader(BeamSet(data[src], va), batch_size=256), dev)
                    else:
                        r = evaluate(model, loaders[tgt], dev)
                    for k, v in r.items():
                        res.setdefault(src, {}).setdefault(tgt, {}).setdefault(k, []).append(v)
                print(f"[{mset}] {src} seed {seed} trained+evaluated in {time.time() - t:.0f}s", flush=True)
        out["results"][mset] = res
        print(f"\n== {mset}: top-1, mean over {len(a.seeds)} seed(s); rows = trained on, cols = tested on ==")
        print(f"{'':11s}" + "".join(f"{c.replace('scenario', 's'):>8s}" for c in a.scenarios))
        for src in a.scenarios:
            print(f"{src:11s}" + "".join(f"{np.mean(res[src][c]['top1']):8.3f}" for c in a.scenarios))
        print(f"-- {mset}: DBA score --")
        for src in a.scenarios:
            print(f"{src:11s}" + "".join(f"{np.mean(res[src][c]['dba']):8.3f}" for c in a.scenarios))
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / f"transfer{a.tag}_{int(time.time())}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
