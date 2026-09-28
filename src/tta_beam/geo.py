"""Geometric (non-learned) sensing features and beam maps.

Per-frame features, aligned with `load_scenario()` rows:
  cam_x  horizontal centre of the UE's bounding box, normalised to [0, 1] (NaN when not annotated).
         From DeepSense's own `resources/annotations/bbox` (class 0 = the UE vehicle); scenarios 1-9.
  rad_u  sin(angle) of the strongest moving radar target (scripts/radar_features.py); scenarios 31-35.
  rad_r  its range bin.

Beam maps (all: mean beam index as a function of one angle-like feature; parameters fitted, not learned):
  camera:  beam = c0 + K * sin(atan(a * (x - x0)))        pinhole camera, array on the same unit
  radar :  beam = c0 + K * u                              uniform-in-sine codebook, radar on the same unit
  GPS   :  PhysicsPrior in models.py (bearing, boresight, origin)
"""
import glob
import os

import numpy as np
import torch
from scipy.optimize import least_squares

from .data import CACHE, RAW, find_csv


# ---------------------------------------------------------------- features
def load_cam_x(name: str) -> np.ndarray:
    f = CACHE / f"{name}_cam_x.npz"
    if f.exists():
        return np.load(f)["cam_x"]
    import pandas as pd
    csv = find_csv(name)
    df = pd.read_csv(csv)
    dirs = glob.glob(str(RAW / name / "**" / "annotations" / "bbox"), recursive=True)
    out = np.full(len(df), np.nan, np.float32)
    if dirs:
        bdir = dirs[0]
        for i, img in enumerate(df["unit1_rgb"] if "unit1_rgb" in df else df[[c for c in df if "rgb" in c][0]]):
            p = os.path.join(bdir, os.path.basename(str(img)).replace(".jpg", ".txt"))
            if not os.path.exists(p):
                continue
            ue = [l.split() for l in open(p) if l.strip() and l.split()[0] == "0"]
            if len(ue) == 1:
                out[i] = float(ue[0][1])
    np.savez_compressed(f, cam_x=out)
    return out


def load_cam_boxes(name: str, max_boxes: int = 8):
    """All annotated boxes per frame (any class), as (N, max_boxes) arrays of x-centre (NaN padded) and
    an index of which column is the UE (class 0), for the no-oracle-association experiment."""
    f = CACHE / f"{name}_cam_boxes.npz"
    if f.exists():
        z = np.load(f)
        return z["xs"], z["ue_col"]
    import pandas as pd
    csv = find_csv(name)
    df = pd.read_csv(csv)
    dirs = glob.glob(str(RAW / name / "**" / "annotations" / "bbox"), recursive=True)
    xs = np.full((len(df), max_boxes), np.nan, np.float32)
    ue = np.full(len(df), -1, np.int64)
    if dirs:
        col = "unit1_rgb" if "unit1_rgb" in df else [c for c in df if "rgb" in c][0]
        for i, img in enumerate(df[col]):
            p = os.path.join(dirs[0], os.path.basename(str(img)).replace(".jpg", ".txt"))
            if not os.path.exists(p):
                continue
            rows = [l.split() for l in open(p) if l.strip()][:max_boxes]
            for j, r in enumerate(rows):
                xs[i, j] = float(r[1])
                if r[0] == "0" and ue[i] < 0:
                    ue[i] = j
    np.savez_compressed(f, xs=xs, ue_col=ue)
    return xs, ue


def load_det_boxes(name: str):
    """Detector boxes (scripts/detect_vehicles.py): x-centres (N, M) sorted by score, NaN padded; None if absent."""
    f = CACHE / f"{name}_det_boxes.npz"
    return np.load(f)["xs"] if f.exists() else None


def load_radar(name: str):
    f = CACHE / f"{name}_radar_target.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    return z["u"].astype(np.float32), z["r_bin"].astype(np.float32)


def attach_geo(d: dict, name: str) -> dict:
    """Add cam_x / rad_u / rad_r (NaN where unavailable) to a scenario dict in place."""
    n = len(d["y"])
    cx = load_cam_x(name) if not name.startswith("syn") else np.full(n, np.nan, np.float32)
    ru, rr = (load_radar(name) if not name.startswith("syn") else (None, None))
    d["cam_x"] = cx if cx is not None and len(cx) == n else np.full(n, np.nan, np.float32)
    d["rad_u"] = ru if ru is not None and len(ru) == n else np.full(n, np.nan, np.float32)
    d["rad_r"] = rr if rr is not None and len(rr) == n else np.full(n, np.nan, np.float32)
    if not name.startswith("syn") and np.isfinite(d["cam_x"]).any():
        xs, _ = load_cam_boxes(name)
        if len(xs) == n:
            d["cam_boxes"] = xs
    det = None if name.startswith("syn") else load_det_boxes(name)
    if det is not None and len(det) == n:
        d["det_boxes"] = det
    return d


# ---------------------------------------------------------------- beam maps
class GeoMap:
    """A parametric map feature -> mean beam, with robust fitting and optional MAP refits."""

    kind = "base"

    def __init__(self, params, prior_sigma=None):
        self.p = np.asarray(params, float)
        self.p_ref = self.p.copy()
        self.prior_sigma = None if prior_sigma is None else np.asarray(prior_sigma, float)

    def mean(self, f, p=None):
        raise NotImplementedError

    def fit(self, f, y, free=None, map_weight=0.0, starts=None):
        f, y = np.asarray(f, float), np.asarray(y, float)
        ok = np.isfinite(f)
        f, y = f[ok], y[ok]
        if len(y) < 3:
            return self
        free = np.ones(len(self.p), bool) if free is None else np.asarray(free, bool)
        base = self.p.copy()

        def resid(q):
            p = base.copy()
            p[free] = q
            r = self.mean(f, p) - y
            if map_weight > 0 and self.prior_sigma is not None:
                pen = (p[free] - self.p_ref[free]) / self.prior_sigma[free] * map_weight * np.sqrt(len(y)) / 4
                r = np.concatenate([r, pen])
            return r

        best = None
        for s in (starts or [base[free]]):
            r = least_squares(resid, np.asarray(s, float), loss="soft_l1", max_nfev=200)
            if best is None or r.cost < best.cost:
                best = r
        self.p[free] = best.x
        return self

    def logits(self, f, sigma=2.0, fallback=None):
        """Gaussian log-scores over 64 beams; rows with NaN features get `fallback` logits (or flat)."""
        f = f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f)
        mu = self.mean(np.nan_to_num(f, nan=0.0))
        grid = np.arange(64)[None, :]
        lg = -((grid - mu[:, None]) ** 2) / (2 * sigma**2)
        bad = ~np.isfinite(f)
        if bad.any():
            lg[bad] = 0.0 if fallback is None else fallback[bad]
        return torch.as_tensor(lg, dtype=torch.float32)


def fit_cam_map_em(boxes, y, iters=6):
    """Fit a CamMap from multi-box frames with labels only (no box identity): alternate between assigning each
    frame the box whose predicted beam is closest to its label and re-fitting the map. Returns (map, chosen x)."""
    boxes, y = np.asarray(boxes, float), np.asarray(y, float)
    ok = np.isfinite(boxes).any(1)
    b, yy = boxes[ok], y[ok]
    m = CamMap()
    # init: the highest-scoring box in each frame (column 0), robust fit
    x = b[:, 0]
    m.fit_source(x, yy)
    for _ in range(iters):
        mu = m.mean(np.nan_to_num(b, nan=-9.0))
        mu[~np.isfinite(b)] = np.inf
        j = np.argmin(np.abs(mu - yy[:, None]), axis=1)
        x = b[np.arange(len(j)), j]
        m.fit_source(x, yy)
    return m, x


class CamMap(GeoMap):
    kind = "cam"

    def __init__(self, params=(31.5, 50.0, 1.2, 0.55)):
        super().__init__(params, prior_sigma=(8.0, 15.0, 0.4, 0.08))

    def mean(self, x, p=None):
        c0, K, a, x0 = self.p if p is None else p
        return c0 + K * np.sin(np.arctan(a * (np.asarray(x) - x0)))

    def fit_source(self, x, y):
        starts = [[31.5, 50.0, a0, 0.5] for a0 in (0.8, 1.2, 2.0, 3.0)]
        # constrain the aperture: K*sin(atan(a*0.5)) should stay within the codebook, avoids degenerate fits
        self.fit(x, y, starts=starts)
        c0, K, a, x0 = self.p
        if not (0 < a < 5 and 20 < K < 120):
            self.p = np.array([31.5, 50.0, 1.2, 0.55])
            self.fit(x, y, free=[True, True, False, True])
        self.p_ref = self.p.copy()
        return self


class RadarMap(GeoMap):
    kind = "radar"

    def __init__(self, params=(31.5, 40.0)):
        super().__init__(params, prior_sigma=(8.0, 12.0))

    def mean(self, u, p=None):
        c0, K = self.p if p is None else p
        return c0 + K * np.asarray(u)

    def fit_source(self, u, y):
        self.fit(u, y)
        self.p_ref = self.p.copy()
        return self
