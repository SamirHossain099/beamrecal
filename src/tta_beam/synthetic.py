"""Synthetic stand-in for DeepSense 6G scenarios 31-34 so the pipeline runs before real data arrives.

A roadside BS with a 16-element ULA and a 64-beam codebook watches vehicles drive past. Each site has its
own BS heading, road geometry and static radar clutter; night lowers camera SNR. This reproduces the
*kinds* of shift in the real split (day/night at one site, cross-site), not their magnitude.
Results on it are sanity checks only.
"""
from dataclasses import dataclass
import numpy as np

N_ANT, N_BEAMS = 16, 64
BEAM_ANGLES = np.linspace(-np.pi / 3, np.pi / 3, N_BEAMS)
CAM_H, CAM_W, RAD_R, RAD_A = 24, 64, 32, 32


def steer(theta):
    n = np.arange(N_ANT)
    return np.exp(1j * np.pi * n[None, :] * np.sin(np.atleast_1d(theta))[:, None]) / np.sqrt(N_ANT)


CODEBOOK = steer(BEAM_ANGLES)  # N_BEAMS x N_ANT


@dataclass
class Site:
    name: str
    heading: float        # BS boresight azimuth in the global frame (rad)
    road_dist: float      # perpendicular distance BS -> road (m)
    road_angle: float     # road direction relative to boresight normal (rad)
    night: bool
    clutter_seed: int
    speed: tuple = (8.0, 15.0)
    dt: float = 0.1


SITES = {
    # analogues of DeepSense: 32/33 same location day/night, 34 other location night, 31 third location day
    "syn32": Site("syn32", heading=0.00, road_dist=12.0, road_angle=0.00, night=False, clutter_seed=2),
    "syn33": Site("syn33", heading=0.00, road_dist=12.0, road_angle=0.00, night=True, clutter_seed=2),
    "syn34": Site("syn34", heading=0.35, road_dist=18.0, road_angle=0.15, night=True, clutter_seed=4),
    "syn31": Site("syn31", heading=-0.30, road_dist=9.0, road_angle=-0.10, night=False, clutter_seed=1, dt=0.05),
}


def _blob(h, w, cy, cx, sy, sx):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((yy - cy) ** 2) / (2 * sy**2) - ((xx - cx) ** 2) / (2 * sx**2))


def generate(site_name: str, n_seq: int = 60, seed: int = 0):
    s = SITES[site_name]
    rng = np.random.default_rng(seed)
    crng = np.random.default_rng(s.clutter_seed)
    clutter = sum(_blob(RAD_R, RAD_A, crng.uniform(3, 28), crng.uniform(2, 30), 1.0, 1.5) * crng.uniform(0.3, 0.8)
                  for _ in range(5))
    cam_bg = crng.uniform(0.2, 0.6, (CAM_H, CAM_W)) * (0.15 if s.night else 1.0)

    out = {k: [] for k in ("gps", "cam", "radar", "y", "pwr", "seq")}
    for q in range(n_seq):
        v = rng.uniform(*s.speed) * rng.choice([-1, 1])
        along = np.arange(-60, 60, abs(v) * s.dt)[:: int(np.sign(v))]
        for a in along:
            # vehicle position in BS frame (x along boresight, y lateral)
            xb = s.road_dist + a * np.sin(s.road_angle) + rng.normal(0, 0.3)
            yb = a * np.cos(s.road_angle) + rng.normal(0, 0.3)
            theta = np.arctan2(yb, xb)
            if abs(theta) > np.pi / 3 - 0.02:
                continue
            r = np.hypot(xb, yb)
            # 64-beam receive power: LoS + one weak reflection + noise, with random blockage dips
            h = steer(theta)[0] * (1.0 if rng.random() > 0.05 else 0.2)
            h = h + 0.25 * steer(theta + rng.normal(0.4, 0.1))[0] * np.exp(1j * rng.uniform(0, 2 * np.pi))
            pwr = np.abs(CODEBOOK.conj() @ h) ** 2 / r**2 + rng.exponential(2e-5, N_BEAMS)
            # GPS in the global frame (east, north) relative to BS, 10 cm RTK-ish noise
            ang_g = theta + s.heading
            gps = np.array([r * np.cos(ang_g), r * np.sin(ang_g)]) + rng.normal(0, 0.1, 2)
            # camera: vehicle blob at column set by angle, brightness/SNR drop at night
            cx = (np.tan(theta) / np.tan(np.pi / 3) + 1) / 2 * (CAM_W - 1)
            snr = 0.25 if s.night else 1.0
            cam = cam_bg + snr * _blob(CAM_H, CAM_W, CAM_H * 0.6, cx, 3, 60 / r + 1) \
                + rng.normal(0, 0.08 if s.night else 0.03, (CAM_H, CAM_W))
            # radar range-angle map in BS frame + site clutter
            ri = r / 60 * (RAD_R - 1)
            ai = (theta / (np.pi / 3) + 1) / 2 * (RAD_A - 1)
            radar = clutter + _blob(RAD_R, RAD_A, ri, ai, 0.8, 1.0) + rng.rayleigh(0.08, (RAD_R, RAD_A))

            out["gps"].append(gps / 30.0)
            out["cam"].append(cam[None])
            out["radar"].append(radar[None])
            out["pwr"].append(pwr)
            out["y"].append(int(np.argmax(pwr)))
            out["seq"].append(q)
    return {k: np.asarray(v, dtype=np.int64 if k in ("y", "seq") else np.float32) for k, v in out.items()}
