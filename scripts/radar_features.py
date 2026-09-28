"""Physics radar front-end: per frame, the moving target's (range bin, sin-angle, peak power).

Static clutter is removed by subtracting the mean over chirps (zero-Doppler removal), then a
range-angle map is formed (range FFT over samples, zero-padded angle FFT over the 4 rx antennas), and
the strongest cell within the configured range gate is taken as the target. The angle is returned as
u = sin(phi) in [-1, 1] with parabolic interpolation around the peak bin. No learning, no labels.

    python scripts/radar_features.py --scenarios scenario31 scenario32 scenario33 scenario34 scenario35

Writes data/cache/<scenario>_radar_target.npz with arrays r_bin, u, p_db (aligned with the scenario CSV
rows, i.e. with load_scenario() order).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import CACHE, _col, find_csv  # noqa: E402


def target(cube, n_angle=128, r_min=4, chirp_step=5):
    c = np.asarray(cube, np.complex64)
    c = c - c.mean(axis=2, keepdims=True)                     # zero-Doppler (static clutter) removal
    c = c[:, :, ::chirp_step]                                 # non-coherent sum below: subsampling is enough
    rng = np.fft.fft(c, axis=1)[:, : c.shape[1] // 2, :]      # positive ranges
    ang = np.fft.fftshift(np.fft.fft(rng, n=n_angle, axis=0), axes=0)   # angle x range x chirp
    ra = (np.abs(ang) ** 2).sum(axis=2)                       # angle x range, non-coherent over chirps
    ra[:, :r_min] = 0                                         # drop the leakage bins next to the radar
    a, r = np.unravel_index(np.argmax(ra), ra.shape)
    # parabolic interpolation on the angle axis
    if 0 < a < n_angle - 1:
        y0, y1, y2 = np.log(ra[a - 1, r] + 1e-12), np.log(ra[a, r] + 1e-12), np.log(ra[a + 1, r] + 1e-12)
        da = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    else:
        da = 0.0
    k = (a + da) - n_angle / 2                                # fftshifted bin index
    u = np.clip(2 * k / n_angle, -1, 1)                       # half-wavelength spacing: bin -> sin(phi)
    return r, u, 10 * np.log10(ra[a, r] + 1e-12)


def main():
    import pandas as pd
    from tqdm import tqdm
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["scenario31", "scenario32", "scenario33", "scenario34", "scenario35"])
    a = ap.parse_args()
    for name in a.scenarios:
        out = CACHE / f"{name}_radar_target.npz"
        if out.exists():
            print("cached", out.name, flush=True)
            continue
        csv = find_csv(name)
        df = pd.read_csv(csv)
        base = csv.parent
        col = _col(df, "radar")
        R, U, P = [], [], []
        for p in tqdm(df[col], desc=name, mininterval=20):
            r, u, pdb = target(np.load(base / str(p).lstrip("./")))
            R.append(r), U.append(u), P.append(pdb)
        np.savez_compressed(out, r_bin=np.array(R), u=np.array(U, np.float32), p_db=np.array(P, np.float32))
        print("wrote", out.name, len(R), flush=True)


if __name__ == "__main__":
    main()
