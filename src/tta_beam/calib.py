"""Label-space calibration for beam prediction under deployment shift.

The sanity check on DeepSense showed that re-mounting the array turns the source model's predictions
into (mirror?) then (shift by delta beams) of the truth, with delta up to +-40. A source model cannot
undo that from features. A base station can, cheaply, from what it observes anyway:

  * P1 sweeps: every K frames the BS measures all 64 beams -> a true best-beam label.
  * served-beam power: every frame the BS knows the power of the beam it served.

`OffsetCalibrator` keeps a robust running estimate of (mirror, delta) from sweep labels and applies it
to the logits by a circular shift of the beam axis. It is gradient-free, costs a 64-vector roll per frame,
and is the baseline every feature-space TTA method should be combined with.

`PowerBandit` estimates delta without any sweep, from served-beam power alone: it keeps a score per
candidate offset and occasionally serves a neighbouring offset (epsilon-greedy), comparing the power it
gets with what the incumbent got on the adjacent frame of the same pass. It is slower to converge and
mirror-blind, but it needs no labels at all.
"""
import numpy as np
import torch

N_BEAMS = 64


def apply_transform(logits: torch.Tensor, delta: int, mirror: bool) -> torch.Tensor:
    """Return logits re-indexed so that beam b of the output = beam (mirror? 63-b : b) - delta of the input.

    If the true beam is  y = T(pred) with T(b) = (63-b if mirror else b) + delta, then
    argmax(apply_transform(logits)) == T(argmax(logits)).
    """
    if mirror:
        logits = logits.flip(1)
    return torch.roll(logits, shifts=int(delta), dims=1)


class OffsetCalibrator:
    """Running (mirror, delta) estimate from sweep labels.

    delta = median over the last `window` sweeps of (y_true - pred_uncalibrated), computed separately
    for the un-mirrored and mirrored hypotheses; the hypothesis with the smaller median absolute
    residual wins. Needs `min_sweeps` labels before it changes anything.
    """

    def __init__(self, window: int = 25, min_sweeps: int = 3):
        self.window, self.min_sweeps = window, min_sweeps
        self.pred, self.true = [], []
        self.delta, self.mirror = 0, False

    def observe(self, pred_uncal: torch.Tensor, y_true: torch.Tensor):
        self.pred += pred_uncal.tolist()
        self.true += y_true.tolist()
        self.pred, self.true = self.pred[-self.window:], self.true[-self.window:]
        if len(self.pred) < self.min_sweeps:
            return
        p, t = np.array(self.pred), np.array(self.true)
        best = None
        for mirror in (False, True):
            q = (N_BEAMS - 1 - p) if mirror else p
            d = t - q
            delta = int(np.round(np.median(d)))
            resid = np.median(np.abs(d - delta))
            if best is None or resid < best[0]:
                best = (resid, delta, mirror)
        _, self.delta, self.mirror = best

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return apply_transform(logits, self.delta, self.mirror)


class PowerBandit:
    """Offset estimate from served-beam power only (no labels).

    Scores candidate offsets in [-max_delta, max_delta]. Serves the incumbent argmax offset except with
    prob. `eps`, when it explores incumbent +-1..+-step. Each served frame's power, normalised by the
    running power of the pass, updates that candidate's score with an EMA. Mirror is not estimated.
    """

    def __init__(self, max_delta: int = 40, eps: float = 0.2, step: int = 3, ema: float = 0.9, seed: int = 0):
        self.max_delta, self.eps, self.step, self.ema = max_delta, eps, step, ema
        self.score = np.zeros(2 * max_delta + 1)
        self.count = np.zeros(2 * max_delta + 1)
        self.rng = np.random.default_rng(seed)
        self.delta = 0
        self.pass_ref = {}  # pass id -> EMA of served power (normaliser)

    def choose(self, n: int):
        """Offsets to serve on the next n frames."""
        cands = np.full(n, self.delta)
        explore = self.rng.random(n) < self.eps
        cands[explore] = np.clip(self.delta + self.rng.integers(-self.step, self.step + 1, explore.sum()),
                                 -self.max_delta, self.max_delta)
        return cands

    def update(self, served_delta, power, pass_id):
        for d, p, s in zip(np.asarray(served_delta), np.asarray(power, float), np.asarray(pass_id)):
            ref = self.pass_ref.get(s, p)
            self.pass_ref[s] = 0.9 * ref + 0.1 * p
            rel = 10 * np.log10(max(p, 1e-12) / max(ref, 1e-12))     # dB relative to the pass's running level
            i = int(d) + self.max_delta
            self.score[i] = self.ema * self.score[i] + (1 - self.ema) * rel if self.count[i] else rel
            self.count[i] += 1
        seen = self.count > 0
        if seen.any():
            self.delta = int(np.flatnonzero(seen)[np.argmax(self.score[seen])]) - self.max_delta


class BoresightCalibrator:
    """Per-deployment boresight angle for a model with a GPSEnc branch, from sweep labels.

    Keeps the last `window` (input, true beam) pairs from sweep frames and picks the theta on a grid that
    maximises the model's agreement on them (top-1 hits, with a small tie-breaker on |beam error| and on
    staying close to the current theta). Coarse-to-fine: 4 deg over [-180, 180), then 0.5 deg.

    Cost: the non-position branches are run once per observe and cached; each grid point re-runs only the
    position MLP, the fusion layer and the head on <= `window` frames. Test-time normalisation layers are
    frozen (momentum 0) during the search so repeated evaluations do not drift their statistics.
    """

    def __init__(self, model, window: int = 24, min_sweeps: int = 2):
        self.model, self.window, self.min_sweeps = model, window, min_sweeps
        self.enc = model.enc["gps"]
        self.xs, self.ys = [], []

    def _norms(self):
        return [m for m in self.model.modules() if hasattr(m, "m") and hasattr(m, "mean")]

    @torch.no_grad()
    def _logits(self, theta, b, cached):
        old = self.enc.theta.clone()
        self.enc.theta.fill_(float(theta))
        feats = [self.enc(b["gps"]) if m == "gps" else cached[m] for m in self.model.modalities]
        out = self.model.head(self.model.fuse(torch.cat(feats, 1)))
        self.enc.theta.copy_(old)
        return out

    def _score(self, theta, b, y, cached):
        pred = self._logits(theta, b, cached).argmax(1)
        err = (pred - y).abs().float().clamp(max=5)
        return (pred == y).float().sum().item() - 0.1 * (err.sum().item() / 5)

    @torch.no_grad()
    def observe(self, b_sweep: dict, y_sweep: torch.Tensor):
        for i in range(len(y_sweep)):
            self.xs.append({k: v[i:i + 1] for k, v in b_sweep.items()})
            self.ys.append(y_sweep[i:i + 1])
        self.xs, self.ys = self.xs[-self.window:], self.ys[-self.window:]
        if len(self.ys) < self.min_sweeps:
            return
        b = {k: torch.cat([x[k] for x in self.xs]) for k in self.xs[0]}
        y = torch.cat(self.ys)
        was_training = self.model.training
        self.model.eval()
        norms = self._norms()
        moms = [n.m for n in norms]
        for n in norms:
            n.m = 0.0
        cached = {m: self.model.enc[m](b[m]) for m in self.model.modalities if m != "gps"}
        cur = self.enc.theta.item()
        coarse = np.deg2rad(np.arange(-180, 180, 4.0))
        best = max(coarse, key=lambda t: (self._score(t, b, y, cached), -abs(t - cur)))
        fine = best + np.deg2rad(np.arange(-4, 4.01, 0.5))
        best = max(fine, key=lambda t: (self._score(t, b, y, cached), -abs(t - cur)))
        self.enc.theta.fill_(float(best))
        for n, mm in zip(norms, moms):
            n.m = mm
        self.model.train(was_training)


class PhysicsCalibrator:
    """Re-fit the anchored model's prior (theta, and K once enough labels exist) from sweep labels.

    Least squares on the parametric prior alone: no network evaluation, microseconds per update.
    """

    def __init__(self, model, window: int = 48, min_sweeps: int = 4, min_for_K: int = 12, jump_beams: float = 8.0,
                 jump_n: int = 4, min_for_origin: int = 16, jump_beams_young: float = 20.0,
                 min_bearing_span_deg: float = 30.0, origin_window: int = 200):
        self.model, self.window, self.min_sweeps, self.min_for_K = model, window, min_sweeps, min_for_K
        self.min_for_origin, self.jump_beams_young = min_for_origin, jump_beams_young
        self.min_bearing_span_deg, self.origin_window = min_bearing_span_deg, origin_window
        self.jump_beams, self.jump_n = jump_beams, jump_n
        self.g, self.y = [], []
        self.recent_err = []   # |prior mean beam - y| on the newest sweeps, for change detection
        self.theta_known = None  # boresight of the current site once fitted (enables the local search)
        self.n_resets = 0
        self.on_reset = None   # callback (used by gate selection to flush its own window)

    def observe(self, gps_sweep: torch.Tensor, y_sweep: torch.Tensor):
        g_new, y_new = gps_sweep.detach().cpu(), y_sweep.detach().cpu()
        with torch.no_grad():  # change detection: does the *current* prior still explain fresh labels?
            err = (self.model.prior.mean_beam(g_new.to(self.model.prior.c0.device)).cpu() - y_new.float()).abs()
        self.recent_err += err.tolist()
        self.recent_err = self.recent_err[-self.jump_n:]
        have = len(torch.cat(self.y)) if self.y else 0
        # refractory rule: a young window (fewer labels than the origin fit needs) is still converging, so
        # only a very large jump may reset it again; otherwise multi-lane sites reset in a loop
        threshold = self.jump_beams if have >= self.min_for_origin else self.jump_beams_young
        if have >= self.min_sweeps and len(self.recent_err) >= self.jump_n and np.median(self.recent_err) > threshold:
            self.g, self.y, self.recent_err, self.n_resets = [], [], [], self.n_resets + 1
            self.model.prior.origin.zero_()  # a new site: no reason to keep the old origin correction
            self.theta_known = None
            if self.on_reset is not None:
                self.on_reset()
        self.g.append(g_new)
        self.y.append(y_new)
        g_all, y_all = torch.cat(self.g)[-self.origin_window:], torch.cat(self.y)[-self.origin_window:]
        self.g, self.y = [g_all], [y_all]
        g, y = g_all[-self.window:], y_all[-self.window:]   # short window for theta/K (tracks drift)
        if len(y) < self.min_sweeps:
            return
        # theta/K every sweep (cheap); the origin only when 8 new labels have arrived since its last fit
        # (bounded 5-parameter solve); theta is searched locally once it is known for this site.
        n = len(y_all)
        # identifiability guard: the origin is only fitted when the labels span enough bearing (several
        # passes / lanes); from a single short pass it is unidentifiable and would drift
        bearing = torch.atan2(g_all[:, 0] - self.model.prior.origin[0].cpu(), g_all[:, 1] - self.model.prior.origin[1].cpu())
        span_ok = float(bearing.max() - bearing.min()) > np.deg2rad(self.min_bearing_span_deg)
        refit_origin = span_ok and n >= self.min_for_origin and (n - self.min_for_origin) % 8 == 0
        if refit_origin:   # long buffer: the origin needs the spread of many passes
            self.model.prior.fit(g_all, y_all, fit_c0=False, fit_K=True, fit_origin=True, local_theta=self.theta_known)
        else:              # short window: cheap theta/K tracking
            self.model.prior.fit(g, y, fit_c0=False, fit_K=len(y) >= self.min_for_K, fit_origin=False,
                                 local_theta=self.theta_known)
        self.theta_known = float(self.model.prior.theta.item())
        # recent_err is kept across re-fits: within a site a re-fit moves the prior little, so stale
        # errors stay small; at a site change the newest errors are all large and the detector fires.
