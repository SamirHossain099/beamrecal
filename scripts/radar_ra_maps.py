"""Cache per-frame range-angle power maps (float16, angle 64 x all positive range bins) for scenarios 31-35.

Static clutter removed (mean over chirps), chirps subsampled 5x, non-coherent sum. Used to locate the UE in
the radar by its GPS range instead of taking the strongest reflector.

    python scripts/radar_ra_maps.py --scenarios scenario31 ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import CACHE, _col, find_csv  # noqa: E402


def ra_map(cube, n_angle=64, chirp_step=5):
    c = np.asarray(cube, np.complex64)
    c = c - c.mean(axis=2, keepdims=True)
    c = c[:, :, ::chirp_step]
    rng = np.fft.fft(c, axis=1)[:, : c.shape[1] // 2, :]
    ang = np.fft.fftshift(np.fft.fft(rng, n=n_angle, axis=0), axes=0)
    return (np.abs(ang) ** 2).sum(axis=2).astype(np.float16)   # angle x range


def main():
    import pandas as pd
    from tqdm import tqdm
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["scenario32", "scenario31", "scenario33", "scenario34", "scenario35"])
    a = ap.parse_args()
    for name in a.scenarios:
        out = CACHE / f"{name}_radar_ra.npy"
        if out.exists():
            print("cached", out.name, flush=True)
            continue
        csv = find_csv(name)
        df = pd.read_csv(csv)
        col = _col(df, "radar")
        maps = [ra_map(np.load(csv.parent / str(p).lstrip("./"))) for p in tqdm(df[col], desc=name, mininterval=30)]
        np.save(out, np.stack(maps))
        print("wrote", out.name, np.stack(maps).shape, flush=True)


if __name__ == "__main__":
    main()
