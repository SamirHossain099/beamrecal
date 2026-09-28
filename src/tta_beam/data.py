"""Data loading. Every scenario becomes one dict of arrays:

    gps   (N, 2)        UE position relative to BS, east/north, scaled (m / 30)
    cam   (N, 3, H, W)  RGB uint8, downsampled (BeamSet converts to float in [0, 1])
    radar (N, 1, R, A)  range-angle magnitude map (zeros when the scenario has no radar)
    pwr   (N, 64)       measured receive power per beam (linear, NaN -> 0)
    y     (N,)          best beam 0..63 (nanargmax of pwr, as the DeepSense baseline does)
    seq   (N,)          sequence (vehicle pass) id, samples in time order within a sequence

Real DeepSense scenarios live extracted under DEEPSENSE_ROOT/scenarioNN/ (any nesting; the first CSV found
outside `resources/` is the index). Each is parsed once and cached to data/cache/scenarioNN.npz.

Two on-disk layouts exist and both are handled:
  Testbed 1 (scenarios 1-9):  Scenario1/scenario1.csv, cols unit1_rgb, unit1_pwr_60ghz, unit1_loc,
                              unit2_loc (8, 9 add unit2_loc_cal, preferred), unit1_beam_index, seq_index
  Testbed 5 (scenarios 31-34): scenario32_dev.csv at the root, adds unit1_radar (.npy cube), unit1_lidar
  Scenario 35:                 cols unit1_gps, unit2_gps, unit1_pwr, unit1_rgb, unit1_radar, best_beam;
                               radar cube has 512 samples/chirp (31-34: 256), pooled to the same 32 range bins
"""
import os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("DEEPSENSE_ROOT", r"N:\Datasets\DeepSense 6G\extracted"))
CACHE = ROOT / "data" / "cache"
KEYS = ("gps", "cam", "radar", "pwr", "y", "seq")
GEO_KEYS = ("cam_x", "rad_u", "rad_r", "cam_boxes", "det_boxes")  # optional per-frame geometric features (tta_beam.geo)
CAM_HW, RAD_RA = (72, 128), (32, 32)


def load_scenario(name: str, seed: int = 0) -> dict:
    if name.startswith("syn"):
        from .synthetic import generate
        return generate(name, seed=seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if not f.exists():
        np.savez_compressed(f, **parse_deepsense(name))
    z = np.load(f)
    return {k: z[k] for k in KEYS}


def _col(df, *alts, required=True):
    """First column whose name contains every substring of any alternative in `alts`.

    An alternative is a string (one substring) or a tuple of substrings that must all appear.
    """
    for alt in alts:
        subs = (alt,) if isinstance(alt, str) else alt
        for c in df.columns:
            if all(s in c.lower() for s in subs):
                return c
    if required:
        raise KeyError(f"no column matching {alts} in {list(df.columns)}")
    return None


def find_csv(name: str) -> Path:
    folder = RAW / name
    csvs = [p for p in folder.rglob("*.csv") if "resources" not in p.parts]
    if not csvs:
        raise FileNotFoundError(f"no index CSV under {folder}")
    return sorted(csvs, key=lambda p: (len(p.parts), p.name))[0]


def parse_deepsense(name: str, verbose=True) -> dict:
    import pandas as pd
    import utm
    from PIL import Image
    from tqdm import tqdm

    csv = find_csv(name)
    df = pd.read_csv(csv)
    base = csv.parent
    path = lambda p: base / str(p).lstrip("./")  # noqa: E731

    pwr = np.stack([np.loadtxt(path(p)) for p in tqdm(df[_col(df, "pwr")], desc=f"{name} pwr", disable=not verbose, mininterval=10)])
    pwr = pwr.astype(np.float32)
    y = np.nanargmax(np.where(np.isnan(pwr), -np.inf, pwr), axis=1)
    n_nan = int(np.isnan(pwr).any(1).sum())
    pwr = np.nan_to_num(pwr, nan=0.0)
    bcol = _col(df, "unit1_beam", "best_beam", required=False)
    if bcol is not None and verbose:
        agree = float((df[bcol].values - 1 == y).mean())
        print(f"{name}: {len(df)} rows, {n_nan} rows with NaN power, argmax agrees with {bcol} on {agree:.3%}")

    # UE position: the calibrated column is preferred when a scenario provides one (8, 9)
    bs = np.loadtxt(path(df[_col(df, "unit1_loc", ("unit1", "gps"))].iloc[0]))[:2]
    bs_e, bs_n, *_ = utm.from_latlon(bs[0], bs[1])
    gps = []
    for p in df[_col(df, "unit2_loc_cal", "unit2_loc", ("unit2", "gps"))]:
        lat, lon = np.loadtxt(path(p))[:2]
        e, n, *_ = utm.from_latlon(lat, lon)
        gps.append([(e - bs_e) / 30.0, (n - bs_n) / 30.0])

    cam = np.stack([np.asarray(Image.open(path(p)).convert("RGB").resize(CAM_HW[::-1]), np.uint8)
                    for p in tqdm(df[_col(df, "rgb")], desc=f"{name} cam", disable=not verbose, mininterval=10)])
    cam = np.ascontiguousarray(cam.transpose(0, 3, 1, 2))  # uint8, N x 3 x H x W

    rcol = _col(df, "radar", required=False)
    if rcol is not None and not str(df[rcol].iloc[0]).endswith(".npy"):
        if verbose:
            print(f"{name}: radar files are not .npy ({df[rcol].iloc[0]}); skipping radar")
        rcol = None
    if rcol is not None:
        radar = np.stack([radar_range_angle(np.load(path(p)), RAD_RA)
                          for p in tqdm(df[rcol], desc=f"{name} radar", disable=not verbose, mininterval=10)])
    else:
        radar = np.zeros((len(df), 1, *RAD_RA), np.float32)

    seq_col = _col(df, "seq", required=False)
    seq = df[seq_col].values if seq_col else np.zeros(len(df), int)
    return {"gps": np.asarray(gps, np.float32), "cam": cam, "radar": radar.astype(np.float32), "pwr": pwr,
            "y": y.astype(np.int64), "seq": np.asarray(seq, np.int64)}


def radar_range_angle(cube: np.ndarray, out=(32, 32), n_angle=32, chirp_step=5) -> np.ndarray:
    """Raw FMCW cube (rx, samples_per_chirp, chirps) -> normalised log range-angle map of shape (1, R, A).

    Range FFT over samples, angle FFT (zero-padded to `n_angle` bins) over the rx antennas, magnitude
    summed over every `chirp_step`-th chirp (a static map: no Doppler), positive ranges only, then pooled
    to `out`. Sub-sampling chirps and computing at `n_angle` directly is 12x faster than the full version and
    correlates with it at 0.99.
    """
    c = np.asarray(cube, np.complex64)
    if c.ndim == 2:
        c = c[:, :, None]
    c = c[:, :, ::chirp_step]
    rng_fft = np.fft.fft(c - c.mean(axis=1, keepdims=True), axis=1).astype(np.complex64)
    ang_fft = np.fft.fftshift(np.fft.fft(rng_fft, n=n_angle, axis=0), axes=0)
    ra = np.abs(ang_fft).sum(axis=2).T[: rng_fft.shape[1] // 2]   # range x angle, positive ranges
    r, a = out
    ra = ra[: (ra.shape[0] // r) * r].reshape(r, -1, n_angle).mean(1)
    if n_angle != a:
        ra = ra.reshape(r, a, n_angle // a).mean(2)
    ra = np.log1p(ra)
    return ((ra - ra.mean()) / (ra.std() + 1e-6))[None].astype(np.float32)


class BeamSet(torch.utils.data.Dataset):
    def __init__(self, d: dict, idx=None):
        idx = np.arange(len(d["y"])) if idx is None else idx
        self.t = {k: torch.as_tensor(d[k][idx]) for k in KEYS + GEO_KEYS if k in d}
        if self.t["cam"].dtype == torch.uint8:
            self.t["cam"] = self.t["cam"].float() / 255.0

    def __len__(self):
        return len(self.t["y"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.t.items()}


def split_by_sequence(d: dict, frac: float = 0.8, seed: int = 0):
    seqs = np.unique(d["seq"])
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    tr = np.isin(d["seq"], seqs[: int(len(seqs) * frac)])
    return np.where(tr)[0], np.where(~tr)[0]
