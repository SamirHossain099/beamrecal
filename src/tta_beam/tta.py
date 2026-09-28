"""Online test-time adaptation methods for beam prediction.

Every method implements `step(batch, feedback) -> logits`. Logits are the predictions used for scoring and
are produced *before* the method adapts on that batch (online protocol). `feedback` is the free signal a
BS gets after serving the predicted beams (see `stream.py`); only BeamTTA uses it, and only if its rate > 0.

Norm layers are swapped for `EMANorm`, which updates running statistics from the test stream with a
momentum and normalises with them. That works at batch size 1, where plain batch-stat BN breaks.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .calib import BoresightCalibrator, OffsetCalibrator, PhysicsCalibrator, PowerBandit
from .data import KEYS


# ---------------------------------------------------------------- normalisation
class EMANorm(nn.Module):
    """Normalisation with test-time EMA statistics. `momentum` is defined per batch of `ref_batch` frames
    and rescaled to the actual batch size, so the adaptation rate per frame is the same at batch 1 and 16."""
    ref_batch = 16

    def __init__(self, bn: nn.modules.batchnorm._BatchNorm, momentum: float):
        super().__init__()
        self.weight, self.bias, self.eps, self.m = bn.weight, bn.bias, bn.eps, momentum
        self.register_buffer("mean", bn.running_mean.clone())
        self.register_buffer("var", bn.running_var.clone())

    def forward(self, x):
        dims = [0] + list(range(2, x.dim()))
        shape = [1, -1] + [1] * (x.dim() - 2)
        if self.m > 0:
            m = 1 - (1 - self.m) ** (x.shape[0] / self.ref_batch)
            with torch.no_grad():
                self.mean.mul_(1 - m).add_(m * x.mean(dims))
                self.var.mul_(1 - m).add_(m * x.var(dims, unbiased=False))
        return (x - self.mean.view(shape)) / torch.sqrt(self.var.view(shape) + self.eps) * self.weight.view(shape) \
            + self.bias.view(shape)


def swap_norms(model: nn.Module, momentum: float, only: tuple = None, _prefix: str = "") -> nn.Module:
    """Replace BatchNorm by EMANorm. With `only`, layers whose qualified name starts with one of the given
    prefixes get `momentum`; all others get 0 (frozen source statistics). E.g. only=("enc.cam",) adapts
    the camera branch and leaves the position branch and the fusion layer untouched."""
    for name, mod in model.named_children():
        q = f"{_prefix}{name}"
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            m = momentum if (only is None or any(q.startswith(o) for o in only)) else 0.0
            setattr(model, name, EMANorm(mod, m))
        else:
            swap_norms(mod, momentum, only, q + ".")
    return model


def norm_params(model):
    return [p for m in model.modules() if isinstance(m, EMANorm) for p in (m.weight, m.bias)]


def select_params(model, which: str):
    if which == "norm":
        ps = norm_params(model)
    elif which == "norm+head":
        ps = norm_params(model) + list(model.head.parameters())
    elif which == "all":
        ps = list(model.parameters())
    else:
        raise ValueError(which)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    return ps


def entropy(logits):
    p = logits.softmax(1)
    return -(p * logits.log_softmax(1)).sum(1)


def beam_smooth(p, sigma=1.5):
    """Convolve a beam distribution with a Gaussian over neighbouring beam indices."""
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=p.device, dtype=p.dtype) ** 2 / (2 * sigma**2))
    k = (k / k.sum()).view(1, 1, -1)
    return F.conv1d(F.pad(p[:, None], (r, r), mode="replicate"), k)[:, 0]


def augment(b, scale=0.05):
    out = dict(b)
    for m in ("gps", "cam", "radar"):
        if m in b and b[m].is_floating_point():
            out[m] = b[m] + scale * torch.randn_like(b[m]) * (b[m].std() + 1e-6)
    return out


# ---------------------------------------------------------------- methods
class Method:
    """step(b) -> logits to serve/score; feedback(b, fb) is called afterwards with what the BS observed."""
    name = "base"

    def __init__(self, model, momentum=0.0, norm_only=None, **kw):
        self.model = swap_norms(copy.deepcopy(model), momentum, norm_only).eval()

    @torch.no_grad()
    def step(self, b):
        return self.model(b)

    def feedback(self, b, fb):
        pass


class Source(Method):
    name = "source"


class NormAdapt(Method):
    """Test-time normalisation only (BN-adapt / AdaBN with EMA), no gradient."""
    name = "norm"

    def __init__(self, model, momentum=0.05, **kw):
        super().__init__(model, momentum)


class Tent(Method):
    name = "tent"

    def __init__(self, model, momentum=0.05, lr=1e-3, params="norm", **kw):
        super().__init__(model, momentum)
        self.opt = torch.optim.Adam(select_params(self.model, params), lr=lr)

    def step(self, b):
        logits = self.model(b)
        loss = entropy(logits).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return logits.detach()


class EATA(Method):
    """Entropy-filtered, re-weighted Tent with an anchor to source weights (Fisher-weighted if given)."""
    name = "eata"

    def __init__(self, model, momentum=0.05, lr=1e-3, e_margin=0.4, d_margin=0.05, fisher=None, lam=1.0,
                 params="norm", **kw):
        super().__init__(model, momentum)
        self.ps = select_params(self.model, params)
        self.anchor = [p.detach().clone() for p in self.ps]
        self.fisher = fisher or [torch.ones_like(p) for p in self.ps]
        self.opt = torch.optim.Adam(self.ps, lr=lr)
        self.e0, self.d_margin, self.lam, self.avg_p = e_margin * math.log(64), d_margin, lam, None

    def step(self, b):
        logits = self.model(b)
        ent = entropy(logits)
        keep = ent < self.e0
        p = logits.softmax(1).detach()
        if self.avg_p is not None:
            keep &= F.cosine_similarity(p, self.avg_p[None], dim=1).abs() < 1 - self.d_margin
        if keep.any():
            self.avg_p = p[keep].mean(0) if self.avg_p is None else 0.9 * self.avg_p + 0.1 * p[keep].mean(0)
            w = torch.exp(self.e0 - ent[keep].detach())
            loss = (ent[keep] * w).mean()
            loss = loss + self.lam * sum((f * (q - a) ** 2).sum() for f, q, a in zip(self.fisher, self.ps, self.anchor))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return logits.detach()


class SAR(Method):
    """Sharpness-aware reliable entropy minimisation with model reset on collapse."""
    name = "sar"

    def __init__(self, model, momentum=0.05, lr=1e-3, rho=0.05, e_margin=0.4, reset_th=0.2, params="norm", **kw):
        super().__init__(model, momentum)
        self.ps = select_params(self.model, params)
        self.init = copy.deepcopy(self.model.state_dict())
        self.opt = torch.optim.SGD(self.ps, lr=lr * 10, momentum=0.9)
        self.rho, self.e0, self.reset_th, self.ema = rho, e_margin * math.log(64), reset_th, None

    def step(self, b):
        logits = self.model(b)
        ent = entropy(logits)
        keep = ent < self.e0
        if keep.any():
            ent[keep].mean().backward()
            grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.ps]
            norm = torch.norm(torch.stack([g.norm() for g in grads])) + 1e-12
            eps = [self.rho * g / norm for g in grads]
            with torch.no_grad():
                for p, e in zip(self.ps, eps):
                    p.add_(e)
            self.opt.zero_grad()
            ent2 = entropy(self.model(b))
            keep2 = ent2 < self.e0
            if keep2.any():
                loss2 = ent2[keep2].mean()
                loss2.backward()
                self.ema = loss2.item() if self.ema is None else 0.9 * self.ema + 0.1 * loss2.item()
            with torch.no_grad():
                for p, e in zip(self.ps, eps):
                    p.sub_(e)
            self.opt.step()
            self.opt.zero_grad()
            if self.ema is not None and self.ema < self.reset_th:
                self.model.load_state_dict(self.init)
                self.ema = None
        return logits.detach()


class CoTTA(Method):
    """Mean-teacher with augmentation-averaged pseudo-labels and stochastic restore."""
    name = "cotta"

    def __init__(self, model, momentum=0.05, lr=1e-4, alpha=0.999, restore_p=0.01, n_aug=4, ap=0.9, **kw):
        super().__init__(model, momentum)
        self.teacher = copy.deepcopy(self.model)
        self.source = copy.deepcopy(self.model.state_dict())
        self.ps = select_params(self.model, "all")
        self.opt = torch.optim.Adam(self.ps, lr=lr)
        self.alpha, self.restore_p, self.n_aug, self.ap = alpha, restore_p, n_aug, ap

    def step(self, b):
        with torch.no_grad():
            t = self.teacher(b)
            if t.softmax(1).max(1).values.mean() < self.ap:
                t = torch.stack([self.teacher(augment(b)) for _ in range(self.n_aug)]).mean(0)
        s = self.model(b)
        loss = -(t.softmax(1) * s.log_softmax(1)).sum(1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.model.parameters()):
                pt.mul_(self.alpha).add_((1 - self.alpha) * ps)
            for n, p in self.model.named_parameters():
                mask = (torch.rand_like(p) < self.restore_p).float()
                p.copy_(mask * self.source[n] + (1 - mask) * p)
        return t


class T3A(Method):
    """Forward-only: replace the linear head with prototypes built from confident test features."""
    name = "t3a"

    def __init__(self, model, momentum=0.0, filter_k=20, **kw):
        super().__init__(model, momentum)
        W = self.model.head.weight.detach()
        self.support = F.normalize(W, dim=1)
        self.labels = torch.eye(64, device=W.device)
        self.ent = entropy(self.model.head(W))  # warm-start with head weights, as in the paper
        self.k = filter_k

    @torch.no_grad()
    def step(self, b):
        z = self.model.features(b)
        logits = self.model.head(z)
        yhat = F.one_hot(logits.argmax(1), 64).float()
        self.support = torch.cat([self.support, F.normalize(z, dim=1)])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, entropy(logits)])
        # keep the k lowest-entropy supports per class
        keep = torch.zeros(len(self.ent), dtype=torch.bool, device=z.device)
        cls = self.labels.argmax(1)
        for c in cls.unique():
            idx = (cls == c).nonzero().squeeze(1)
            keep[idx[self.ent[idx].argsort()[: self.k]]] = True
        self.support, self.labels, self.ent = self.support[keep], self.labels[keep], self.ent[keep]
        protos = F.normalize(self.labels.T @ self.support, dim=1)
        return F.normalize(z, dim=1) @ protos.T * 10.0  # scored logits (temperature only affects entropy)


class CalibIdx(Method):
    """Source model + beam-index calibration (mirror, offset) from P1-sweep labels. Gradient-free."""
    name = "calib-idx"

    def __init__(self, model, momentum=0.0, window=25, min_sweeps=3, **kw):
        super().__init__(model, momentum)
        self.cal = OffsetCalibrator(window, min_sweeps)
        self.raw = None

    @torch.no_grad()
    def step(self, b):
        self.raw = self.model(b)
        return self.cal(self.raw)

    def feedback(self, b, fb):
        if fb["sweep"].any():
            self.cal.observe(self.raw.argmax(1)[fb["sweep"]], fb["y_sweep"])


class Calib(Method):
    """Source model + boresight calibration of the position branch from P1-sweep labels. Gradient-free.

    Falls back to beam-index calibration when the model has no gps branch.
    """
    name = "calib"

    def __init__(self, model, momentum=0.0, window=24, min_sweeps=2, jump_beams=8.0, jump_n=4, phys_window=48, **kw):
        super().__init__(model, momentum, **kw)
        self.phys = (PhysicsCalibrator(self.model, window=phys_window, jump_beams=jump_beams, jump_n=jump_n)
                     if hasattr(self.model, "prior") else None)
        self.bore = (BoresightCalibrator(self.model, window, min_sweeps)
                     if self.phys is None and "gps" in self.model.enc else None)
        self.idx = OffsetCalibrator(25, 3) if (self.phys is None and self.bore is None) else None
        self.raw = None

    @torch.no_grad()
    def step(self, b):
        self.raw = self.model(b)
        return self.raw if self.idx is None else self.idx(self.raw)

    def feedback(self, b, fb):
        if not fb["sweep"].any():
            return
        if self.phys is not None:
            self.phys.observe(b["gps"][fb["sweep"]], fb["y_sweep"])
        elif self.bore is not None:
            self.bore.observe({k: v[fb["sweep"]] for k, v in b.items()}, fb["y_sweep"])
        else:
            self.idx.observe(self.raw.argmax(1)[fb["sweep"]], fb["y_sweep"])


class CalibNorm(Calib):
    """Calibration + test-time normalisation statistics (appearance shift)."""
    name = "calib+norm"

    def __init__(self, model, momentum=0.05, **kw):
        super().__init__(model, momentum, **kw)


CAM_NORMS = ("enc.cam", "enc.radar")


class Prior(Method):
    """Anchored model with the residual switched off (gate 0) and the source prior: physics only."""
    name = "prior"

    def __init__(self, model, **kw):
        super().__init__(model, 0.0)
        self.model.gate_logit.fill_(-30.0)


class CalibGate(Calib):
    """Physics calibration + online gate on the residual, chosen on the sweep window.

    After each calibration the gate g in `grid` that maximises the log-likelihood of the window's sweep
    labels under logits = log prior + g * residual is kept. Gradient-free; the residual features of the
    window are cached, so the search costs |grid| head evaluations on <= window frames.
    """
    name = "calib+gate"
    grid = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

    def __init__(self, model, momentum=0.0, norm_only=None, window=48, **kw):
        super().__init__(model, momentum, norm_only=norm_only, **kw)
        assert self.phys is not None, "calib+gate needs an anchored model"
        self.win_b, self.win_y, self.window = [], [], window
        self.phys.on_reset = self._flush  # a detected site change also flushes the gate window

    def _flush(self):
        self.win_b, self.win_y = [], []
        self.model.gate_logit.fill_(-30.0)  # trust the physics alone until labels accumulate again

    def feedback(self, b, fb):
        if not fb["sweep"].any():
            return
        super().feedback(b, fb)
        self.win_b.append({k: v[fb["sweep"]] for k, v in b.items() if k in KEYS})
        self.win_y.append(fb["y_sweep"])
        bw = {k: torch.cat([x[k] for x in self.win_b])[-self.window:] for k in self.win_b[0]}
        yw = torch.cat(self.win_y)[-self.window:]
        self.win_b, self.win_y = [bw], [yw]
        if len(yw) < 8:
            return
        m = self.model
        with torch.no_grad():
            norms = [n for n in m.modules() if hasattr(n, "m") and hasattr(n, "mean")]
            moms = [n.m for n in norms]
            for n in norms:
                n.m = 0.0
            prior = m.prior(bw["gps"])
            resid = m.residual(bw)
            for n, mm in zip(norms, moms):
                n.m = mm
            best = max(self.grid, key=lambda g: F.log_softmax(prior + g * resid, 1).gather(1, yw[:, None]).mean().item())
            m.gate_logit.fill_(float(torch.logit(torch.tensor(best).clamp(1e-6, 1 - 1e-6))))


class CalibCamNorm(Calib):
    """Physics calibration + normalisation statistics adapted in the camera/radar branches only."""
    name = "calib+camnorm"

    def __init__(self, model, momentum=0.05, **kw):
        super().__init__(model, momentum, norm_only=CAM_NORMS, **kw)


class CalibGateCamNorm(CalibGate):
    name = "calib+gate+camnorm"

    def __init__(self, model, momentum=0.05, **kw):
        super().__init__(model, momentum, norm_only=CAM_NORMS, **kw)


class CalibGateNorm(CalibGate):
    """Calibration + gate + normalisation statistics adapted in *all* branches (position, fusion, camera).

    Without the gate this collapses (the position branch's statistics encode the source bearing range);
    with the gate the prior carries the position information, and adapting every norm layer helps the
    residual more than camera-only adaptation does.
    """
    name = "calib+gate+norm"

    def __init__(self, model, momentum=0.05, **kw):
        super().__init__(model, momentum, norm_only=None, **kw)


class SupFT(Method):
    """Supervised online fine-tuning on the sweep labels only (the natural competitor to calibration).

    Adam on `params` ("head" or "all"), one CE step per batch that contains sweep frames, replaying the
    last `replay` labelled frames to reduce variance. No physics calibration, no unlabeled-data terms.
    """
    name = "supft"

    def __init__(self, model, momentum=0.0, lr=1e-4, params="all", replay=48, calibrate=False, **kw):
        super().__init__(model, momentum)
        self.ps = [p for p in self.model.parameters()] if params == "all" else list(self.model.head.parameters())
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.ps:
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.ps, lr=lr)
        self.replay, self.buf_b, self.buf_y = replay, [], []
        self.phys = PhysicsCalibrator(self.model) if (calibrate and hasattr(self.model, "prior")) else None

    def feedback(self, b, fb):
        if not fb["sweep"].any():
            return
        if self.phys is not None:
            self.phys.observe(b["gps"][fb["sweep"]], fb["y_sweep"])
        self.buf_b.append({k: v[fb["sweep"]] for k, v in b.items() if k in KEYS})  # model inputs only
        self.buf_y.append(fb["y_sweep"])
        bb = {k: torch.cat([x[k] for x in self.buf_b])[-self.replay:] for k in self.buf_b[0]}
        yy = torch.cat(self.buf_y)[-self.replay:]
        self.buf_b, self.buf_y = [bb], [yy]
        with torch.enable_grad():
            loss = F.cross_entropy(self.model(bb), yy)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


class CalibSupFT(SupFT):
    """Physics calibration + supervised fine-tuning of the residual on the sweep labels."""
    name = "calib+supft"

    def __init__(self, model, **kw):
        super().__init__(model, calibrate=True, **kw)


class Full(CalibGate):
    """Calibration + residual fine-tuning on sweep labels + gate + camera-branch normalisation (ours)."""
    name = "full"

    def __init__(self, model, momentum=0.05, lr=1e-4, replay=48, norm_only=None, **kw):
        super().__init__(model, momentum, norm_only=norm_only, **kw)  # all norm layers adapt (better with the gate)
        self.ps = [p for n, p in self.model.named_parameters() if not n.startswith("prior")]
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.ps:
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.ps, lr=lr)
        self.replay, self.buf_b, self.buf_y = replay, [], []
        self._init_state = copy.deepcopy(self.model.state_dict())

    def _flush(self):
        super()._flush()
        self.buf_b, self.buf_y = [], []
        prior = {k: v.clone() for k, v in self.model.state_dict().items() if k.startswith("prior")}
        self.model.load_state_dict(self._init_state)      # residual back to source on a site change
        self.model.load_state_dict({**self.model.state_dict(), **prior})
        self.opt = torch.optim.Adam(self.ps, lr=self.opt.param_groups[0]["lr"])

    def feedback(self, b, fb):
        if not fb["sweep"].any():
            return
        self.buf_b.append({k: v[fb["sweep"]] for k, v in b.items() if k in KEYS})  # model inputs only
        self.buf_y.append(fb["y_sweep"])
        bb = {k: torch.cat([x[k] for x in self.buf_b])[-self.replay:] for k in self.buf_b[0]}
        yy = torch.cat(self.buf_y)[-self.replay:]
        self.buf_b, self.buf_y = [bb], [yy]
        gate = self.model.gate_logit.clone()
        self.model.gate_logit.fill_(6.0)                   # fine-tune the residual at full strength
        with torch.enable_grad():
            loss = F.cross_entropy(self.model(bb), yy)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.model.gate_logit.copy_(gate)
        super().feedback(b, fb)                            # then calibrate and re-select the gate


class Bandit(Method):
    """Offset from served-beam power only: no labels, epsilon-greedy exploration of neighbouring offsets."""
    name = "bandit"

    def __init__(self, model, momentum=0.0, max_delta=40, eps=0.2, step=3, ema=0.9, **kw):
        super().__init__(model, momentum)
        self.bandit = PowerBandit(max_delta, eps, step, ema)
        self.deltas = None

    @torch.no_grad()
    def step(self, b):
        logits = self.model(b)
        self.deltas = self.bandit.choose(len(logits))
        idx = (torch.arange(64, device=logits.device)[None, :]
               - torch.as_tensor(self.deltas, device=logits.device)[:, None]) % 64
        return logits.gather(1, idx)  # per-frame roll: out[b] = in[b - delta]

    def feedback(self, b, fb):
        self.bandit.update(self.deltas, fb["p_served"].tolist(), fb["seq"].tolist())


class BeamTTA(Method):
    """Beam-aware continual TTA (ours) = label-space calibration + feature adaptation with beam structure.

    Calibration: OffsetCalibrator on sweep labels (mirror, offset), applied to the logits before scoring.
    Feature adaptation (norm affine or norm+head), loss on the calibrated logits of the batch just scored:
      lam_e * H(smooth(p))                         beam-smoothed entropy on confident frames
      lam_t * KL(smooth(p_t) || smooth(p_{t-1}))     temporal consistency within a pass
      lam_s * CE(p, y_sweep)                       true labels on sweep frames
      lam_c * -log(1 - p[served])                  complementary label when served power dropped
      lam_a * ||theta - theta_src||^2              anchor to source
    Reset to source on entropy collapse.
    """
    name = "beamtta"

    def __init__(self, model, momentum=0.05, lr=1e-3, params="norm", sigma=1.5, e_margin=0.4, lam_e=1.0,
                 lam_t=1.0, lam_s=1.0, lam_c=0.5, lam_a=1.0, reset_th=0.2, window=25, min_sweeps=3,
                 norm_only=CAM_NORMS, **kw):
        super().__init__(model, momentum, norm_only=norm_only)
        self.ps = select_params(self.model, params)
        if norm_only is not None:  # adapt only the affine parameters of the adapted (camera/radar) norms
            named = dict(self.model.named_modules())
            keep = {id(p) for n, m in named.items() if isinstance(m, EMANorm) and any(n.startswith(o) for o in norm_only)
                    for p in (m.weight, m.bias)}
            for p in self.ps:
                if id(p) not in keep and params == "norm":
                    p.requires_grad_(False)
            self.ps = [p for p in self.ps if p.requires_grad]
        self.anchor = [p.detach().clone() for p in self.ps]
        self.init = copy.deepcopy(self.model.state_dict())
        self.opt = torch.optim.Adam(self.ps, lr=lr)
        self.phys = PhysicsCalibrator(self.model) if hasattr(self.model, "prior") else None
        self.bore = (BoresightCalibrator(self.model, window, min_sweeps)
                     if self.phys is None and "gps" in self.model.enc else None)
        self.cal = OffsetCalibrator(window, min_sweeps) if (self.phys is None and self.bore is None) else (lambda z: z)
        self.sigma, self.e0 = sigma, e_margin * math.log(64)
        self.lam = dict(e=lam_e, t=lam_t, s=lam_s, c=lam_c, a=lam_a)
        self.reset_th, self.ema = reset_th, None
        self.raw = self.cal_logits = None
        self.prev = None  # (pass id, smoothed prob) of the last frame seen

    def step(self, b):
        with torch.enable_grad():
            self.raw = self.model(b)
        self.cal_logits = self.cal(self.raw)
        return self.cal_logits.detach()

    def feedback(self, b, fb):
        logits = self.cal_logits
        ps = beam_smooth(logits.softmax(1), self.sigma).clamp_min(1e-8)
        ent = -(ps * ps.log()).sum(1)
        keep = ent < self.e0
        loss = self.lam["e"] * (ent[keep].mean() if keep.any() else ent.sum() * 0)

        seq = b["seq"]
        prev_p = torch.cat([self.prev[1] if self.prev is not None else ps[:1].detach(), ps[:-1].detach()])
        prev_s = torch.cat([self.prev[0] if self.prev is not None else seq[:1] - 1, seq[:-1]])
        same = seq == prev_s
        if same.any():
            kl = (ps * (ps.log() - prev_p.clamp_min(1e-8).log())).sum(1)
            loss = loss + self.lam["t"] * kl[same].mean()

        if fb["sweep"].any():
            loss = loss + self.lam["s"] * F.cross_entropy(logits[fb["sweep"]], fb["y_sweep"])
        bad = fb["served_low"] & ~fb["sweep"]
        if bad.any():
            p_served = logits.softmax(1)[bad].gather(1, fb["served"][bad, None]).squeeze(1)
            loss = loss + self.lam["c"] * -(1 - p_served).clamp_min(1e-6).log().mean()

        loss = loss + self.lam["a"] * sum(((q - a) ** 2).sum() for q, a in zip(self.ps, self.anchor))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if fb["sweep"].any():  # calibrate after the gradient step
            if self.phys is not None:
                self.phys.observe(b["gps"][fb["sweep"]], fb["y_sweep"])
            elif self.bore is not None:
                self.bore.observe({k: v[fb["sweep"]] for k, v in b.items()}, fb["y_sweep"])
            else:
                self.cal.observe(self.raw.detach().argmax(1)[fb["sweep"]], fb["y_sweep"])

        self.ema = ent.mean().item() if self.ema is None else 0.95 * self.ema + 0.05 * ent.mean().item()
        if self.ema < self.reset_th:
            self.model.load_state_dict(self.init)
            self.ema = None
        with torch.no_grad():
            self.prev = (seq[-1:], ps[-1:].detach())


METHODS = {c.name: c for c in (Source, NormAdapt, Tent, EATA, SAR, CoTTA, T3A, CalibIdx, Calib, CalibNorm,
                               Prior, CalibGate, CalibCamNorm, CalibGateCamNorm, CalibGateNorm, SupFT, CalibSupFT, Full, Bandit, BeamTTA)}
