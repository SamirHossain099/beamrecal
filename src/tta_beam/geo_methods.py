"""Online methods built on geometric maps (no neural network in the prediction path).

All follow the Method protocol of tta.py: step(b) -> logits, then feedback(b, fb). Methods that need the
source scenario to fit their maps declare `needs_source = True`; run_stream passes `src` (a scenario dict
with geometric features attached).

  sweep-hold          no ML: serve the beam of the last full sweep (same pass if any, else last overall)
  cam-geo             camera map fitted on the source; ZERO target labels
  cam-geo+calib       + MAP re-fit of the camera map on the sweep window (with change detection)
  offset-ma           cam-geo + moving-average beam-index offset from sweeps (ViBe-MA-style, arXiv 2605.05071)
  radar-geo           radar map fitted on the source; ZERO target labels
  radar-geo+calib     + MAP re-fit on the sweep window
  gps-selfcal         physics GPS prior (anchored checkpoint) calibrated from camera/radar pseudo-labels on
                      every frame; ZERO sweeps
  geo-fuse            product of the calibrated GPS prior and the (re-fitted) camera/radar map; uses sweeps
  geo-fuse-selfcal    same fusion with ZERO sweeps (GPS calibrated from the sensing map)
"""
import copy

import numpy as np
import torch

from .calib import PhysicsCalibrator
from .geo import CamMap, RadarMap, fit_cam_map_em
from .tta import Method


def _sensor(src):
    """Pick the angle sensor available in the source: camera boxes (Testbed 1) or radar (Testbed 5)."""
    if np.isfinite(src.get("cam_x", np.array([np.nan]))).mean() > 0.5:
        return "cam_x", CamMap
    if np.isfinite(src.get("rad_u", np.array([np.nan]))).mean() > 0.5:
        return "rad_u", RadarMap
    raise ValueError("source has neither camera boxes nor radar targets")


class _Window:
    """Label window with the same change detector as PhysicsCalibrator (median |mean - y| over the last
    `jump_n` new labels > `jump_beams` -> flush)."""

    def __init__(self, size=48, jump_n=4, jump_beams=8.0, min_n=6):
        self.size, self.jump_n, self.jump_beams, self.min_n = size, jump_n, jump_beams, min_n
        self.f, self.y, self.err, self.resets = [], [], [], 0

    def add(self, f, y, err):
        self.err = (self.err + list(err))[-self.jump_n:]
        if len(self.y) >= self.min_n and len(self.err) >= self.jump_n and np.median(self.err) > self.jump_beams:
            self.f, self.y, self.err, self.resets = [], [], [], self.resets + 1
        self.f = (self.f + list(f))[-self.size:]
        self.y = (self.y + list(y))[-self.size:]


class GeoBase(Method):
    needs_source = True
    name = "geo-base"

    def __init__(self, model, src=None, sigma=2.0, **kw):
        # deliberately no network: prediction comes from the maps alone
        self.model = model
        self.sigma = sigma
        self.key, Map = _sensor(src)
        self.map = Map().fit_source(src[self.key], src["y"])
        self.dev = next(model.parameters()).device

    def _sense_logits(self, b):
        return self.map.logits(b[self.key], self.sigma).to(self.dev)

    @torch.no_grad()
    def step(self, b):
        return self._sense_logits(b)


class SweepHold(Method):
    name = "sweep-hold"

    def __init__(self, model, **kw):
        self.dev = next(model.parameters()).device
        self.last_pass, self.last_any = {}, 31

    @torch.no_grad()
    def step(self, b):
        held = [self.last_pass.get(s, self.last_any) for s in b["seq"].tolist()]
        lg = torch.full((len(held), 64), -10.0, device=self.dev)
        lg[torch.arange(len(held)), torch.as_tensor(held, device=self.dev)] = 0.0
        return lg

    def feedback(self, b, fb):
        for s, y in zip(b["seq"][fb["sweep"]].tolist(), fb["y_sweep"].tolist()):
            self.last_pass[s] = y
            self.last_any = y


class CamGeo(GeoBase):
    name = "sense-geo"


class SenseGeoCal(GeoBase):
    """Sensing map re-fitted online (MAP around the source parameters) on the sweep window."""
    name = "sense-geo+calib"

    def __init__(self, model, src=None, **kw):
        super().__init__(model, src, **kw)
        self.win = _Window()

    def feedback(self, b, fb):
        if not fb["sweep"].any():
            return
        f = b[self.key][fb["sweep"]].cpu().numpy()
        y = fb["y_sweep"].cpu().numpy()
        err = np.abs(self.map.mean(np.nan_to_num(f)) - y)
        err[~np.isfinite(f)] = 0
        before = self.win.resets
        self.win.add(f, y, err)
        if self.win.resets != before:           # detected a new site: restart from the source map
            self.map.p = self.map.p_ref.copy()
        if len(self.win.y) >= 4:
            self.map.fit(self.win.f, self.win.y, map_weight=3.0)

    @property
    def calibrator_resets(self):
        return self.win.resets


class OffsetMA(GeoBase):
    """ViBe-MA-style: sensing map + moving average of (true - predicted) beam index over recent sweeps."""
    name = "offset-ma"

    def __init__(self, model, src=None, window=10, **kw):
        super().__init__(model, src, **kw)
        self.hist, self.window = [], window

    @torch.no_grad()
    def step(self, b):
        off = int(round(np.mean(self.hist))) if self.hist else 0
        return torch.roll(self._sense_logits(b), shifts=off, dims=1)

    def feedback(self, b, fb):
        if not fb["sweep"].any():
            return
        f = b[self.key][fb["sweep"]].cpu().numpy()
        ok = np.isfinite(f)
        pred = np.round(self.map.mean(np.nan_to_num(f)))
        self.hist = (self.hist + list((fb["y_sweep"].cpu().numpy() - pred)[ok]))[-self.window:]


class GpsSelfCal(Method):
    """Physics GPS prior (from an anchored checkpoint) calibrated with the sensing map's beams as
    pseudo-labels on every frame. No sweeps are read. Predicts from the GPS prior alone."""
    needs_source = True
    name = "gps-selfcal"

    def __init__(self, model, src=None, sigma=2.0, conf_beams=6.0, **kw):
        assert hasattr(model, "prior"), "gps-selfcal needs an anchored checkpoint (for the GPS prior)"
        self.model = copy.deepcopy(model).eval()
        self.model.gate_logit.fill_(-30.0)            # prior only
        self.key, Map = _sensor(src)
        self.map = Map().fit_source(src[self.key], src["y"])
        self.cal = PhysicsCalibrator(self.model)
        self.sigma, self.conf = sigma, conf_beams

    @torch.no_grad()
    def step(self, b):
        return self.model(b)

    def feedback(self, b, fb):
        f = b[self.key].cpu().numpy()
        ok = np.isfinite(f)
        if not ok.any():
            return
        pseudo = np.clip(np.round(self.map.mean(f[ok])), 0, 63).astype(np.int64)
        idx = torch.as_tensor(np.flatnonzero(ok)[::4])  # thin to ~4 pseudo-labels per batch (labels are correlated)
        self.cal.observe(b["gps"][idx], torch.as_tensor(pseudo[::4], device=b["gps"].device))

    @property
    def phys(self):
        return self.cal


class GeoFuse(Method):
    """Product of experts: calibrated GPS prior x (re-fitted) sensing map. `selfcal=True` reads no sweeps."""
    needs_source = True
    name = "geo-fuse"

    def __init__(self, model, src=None, selfcal=False, sigma=2.0, **kw):
        assert hasattr(model, "prior"), "geo-fuse needs an anchored checkpoint (for the GPS prior)"
        self.model = copy.deepcopy(model).eval()
        self.model.gate_logit.fill_(-30.0)
        self.key, Map = _sensor(src)
        self.map = Map().fit_source(src[self.key], src["y"])
        self.cal = PhysicsCalibrator(self.model)
        self.win = _Window()
        self.selfcal, self.sigma = selfcal, sigma
        self.dev = next(model.parameters()).device

    @torch.no_grad()
    def step(self, b):
        gps_lg = self.model.prior(b["gps"])                     # log-probs, sigma 2
        sense = self.map.logits(b[self.key], self.sigma, fallback=np.zeros((len(b["y"]), 64))).to(self.dev)
        return gps_lg + sense

    def feedback(self, b, fb):
        f = b[self.key].cpu().numpy()
        ok = np.isfinite(f)
        if self.selfcal:
            if ok.any():
                pseudo = np.clip(np.round(self.map.mean(f[ok])), 0, 63).astype(np.int64)
                idx = torch.as_tensor(np.flatnonzero(ok)[::4])
                self.cal.observe(b["gps"][idx], torch.as_tensor(pseudo[::4], device=b["gps"].device))
            return
        if not fb["sweep"].any():
            return
        self.cal.observe(b["gps"][fb["sweep"]], fb["y_sweep"])
        fs, ys = f[fb["sweep"].cpu().numpy()], fb["y_sweep"].cpu().numpy()
        err = np.abs(self.map.mean(np.nan_to_num(fs)) - ys)
        err[~np.isfinite(fs)] = 0
        before = self.win.resets
        self.win.add(fs, ys, err)
        if self.win.resets != before:
            self.map.p = self.map.p_ref.copy()
        if len(self.win.y) >= 4:
            self.map.fit(self.win.f, self.win.y, map_weight=3.0)

    @property
    def phys(self):
        return self.cal


class GeoFuseSelfCal(GeoFuse):
    name = "geo-fuse-selfcal"

    def __init__(self, model, src=None, **kw):
        super().__init__(model, src, selfcal=True, **kw)


class CamAssoc(Method):
    """No oracle UE identity: among ALL annotated boxes in the frame, serve the camera-map beam of the box
    whose beam is closest to the GPS prior's beam. The GPS prior is calibrated from sweeps (PhysicsCalibrator);
    with `selfcal=True` it is calibrated from the chosen boxes themselves (bootstrapped from the source
    prior), so no sweeps are read."""
    needs_source = True
    name = "cam-assoc"

    def __init__(self, model, src=None, selfcal=False, sigma=2.0, **kw):
        assert hasattr(model, "prior"), "cam-assoc needs an anchored checkpoint (for the GPS prior)"
        self.model = copy.deepcopy(model).eval()
        self.model.gate_logit.fill_(-30.0)
        self.map = CamMap().fit_source(src["cam_x"], src["y"])
        self.cal = PhysicsCalibrator(self.model)
        self.selfcal, self.sigma = selfcal, sigma
        self.dev = next(model.parameters()).device
        self.chosen = None

    @torch.no_grad()
    def step(self, b):
        gps_mu = self.model.prior.mean_beam(b["gps"]).cpu().numpy()
        xs = b["cam_boxes"].cpu().numpy()
        mus = self.map.mean(np.nan_to_num(xs, nan=-9.0))
        mus[~np.isfinite(xs)] = np.inf
        j = np.argmin(np.abs(mus - gps_mu[:, None]), axis=1)
        x = xs[np.arange(len(j)), j]
        self.chosen = x
        return self.map.logits(x, self.sigma, fallback=self.model.prior(b["gps"]).cpu().numpy()).to(self.dev)

    def feedback(self, b, fb):
        if self.selfcal:
            ok = np.isfinite(self.chosen)
            if ok.any():
                pseudo = np.clip(np.round(self.map.mean(self.chosen[ok])), 0, 63).astype(np.int64)
                idx = torch.as_tensor(np.flatnonzero(ok)[::4])
                self.cal.observe(b["gps"][idx], torch.as_tensor(pseudo[::4], device=b["gps"].device))
        elif fb["sweep"].any():
            self.cal.observe(b["gps"][fb["sweep"]], fb["y_sweep"])

    @property
    def phys(self):
        return self.cal


class CamAssocSelfCal(CamAssoc):
    name = "cam-assoc-selfcal"

    def __init__(self, model, src=None, **kw):
        super().__init__(model, src, selfcal=True, **kw)


class DetAssoc(Method):
    """Deployable camera pipeline with a real detector (no annotation boxes, no identity oracle).

    Source: camera map fitted by EM over detected boxes using source labels only (fit_cam_map_em).
    Target, per frame: the GPS prior's mean beam selects one detected box; its camera-map beam is served
    (GPS prior logits when nothing is detected). Feedback:
      sweeps  (default)   GPS prior calibrated from sweep labels (PhysicsCalibrator), and the camera map
                          re-fitted (MAP) on (box nearest the sweep label, label) pairs from the sweep window
      selfcal=True        no sweeps read: GPS prior calibrated from the camera beams of the chosen boxes
    """
    needs_source = True
    name = "det-assoc"

    def __init__(self, model, src=None, selfcal=False, refit_map=True, sigma=2.0, **kw):
        assert hasattr(model, "prior"), "det-assoc needs an anchored checkpoint (for the GPS prior)"
        assert "det_boxes" in src, "run scripts/detect_vehicles.py on the source scenario first"
        self.model = copy.deepcopy(model).eval()
        self.model.gate_logit.fill_(-30.0)
        self.map, _ = fit_cam_map_em(src["det_boxes"], src["y"])
        self.cal = PhysicsCalibrator(self.model)
        self.win = _Window()
        self.selfcal, self.refit_map, self.sigma = selfcal, refit_map, sigma
        self.dev = next(model.parameters()).device
        self.chosen = None

    @torch.no_grad()
    def step(self, b):
        gps_mu = self.model.prior.mean_beam(b["gps"]).cpu().numpy()
        xs = b["det_boxes"].cpu().numpy()
        mus = self.map.mean(np.nan_to_num(xs, nan=-9.0))
        mus[~np.isfinite(xs)] = np.inf
        j = np.argmin(np.abs(mus - gps_mu[:, None]), axis=1)
        x = xs[np.arange(len(j)), j]
        self.chosen = x
        return self.map.logits(x, self.sigma, fallback=self.model.prior(b["gps"]).cpu().numpy()).to(self.dev)

    def feedback(self, b, fb):
        if self.selfcal:
            ok = np.isfinite(self.chosen)
            if ok.any():
                pseudo = np.clip(np.round(self.map.mean(self.chosen[ok])), 0, 63).astype(np.int64)
                idx = torch.as_tensor(np.flatnonzero(ok)[::4])
                self.cal.observe(b["gps"][idx], torch.as_tensor(pseudo[::4], device=b["gps"].device))
            return
        if not fb["sweep"].any():
            return
        self.cal.observe(b["gps"][fb["sweep"]], fb["y_sweep"])
        if not self.refit_map:
            return
        xs = b["det_boxes"][fb["sweep"]].cpu().numpy()
        ys = fb["y_sweep"].cpu().numpy()
        mus = self.map.mean(np.nan_to_num(xs, nan=-9.0))
        mus[~np.isfinite(xs)] = np.inf
        j = np.argmin(np.abs(mus - ys[:, None]), axis=1)
        x = xs[np.arange(len(j)), j]
        err = np.abs(self.map.mean(np.nan_to_num(x)) - ys)
        err[~np.isfinite(x)] = 0
        before = self.win.resets
        self.win.add(x, ys, err)
        if self.win.resets != before:
            self.map.p = self.map.p_ref.copy()
        if len(self.win.y) >= 4:
            self.map.fit(self.win.f, self.win.y, map_weight=3.0)

    @property
    def phys(self):
        return self.cal


class DetAssocSelfCal(DetAssoc):
    name = "det-assoc-selfcal"

    def __init__(self, model, src=None, **kw):
        super().__init__(model, src, selfcal=True, **kw)


class DetAssocFixed(DetAssoc):
    """Sweeps calibrate the GPS prior (used only to pick the box); the camera map stays as fitted on the source."""
    name = "det-assoc-fixedmap"

    def __init__(self, model, src=None, **kw):
        super().__init__(model, src, refit_map=False, **kw)


GEO_METHODS = {c.name: c for c in (SweepHold, CamGeo, SenseGeoCal, OffsetMA, GpsSelfCal, GeoFuse, GeoFuseSelfCal,
                                   CamAssoc, CamAssocSelfCal, DetAssoc, DetAssocSelfCal, DetAssocFixed)}
