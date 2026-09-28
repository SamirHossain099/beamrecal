"""Per-modality encoders + late fusion. BatchNorm throughout so norm-based TTA methods apply."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BEAMS = 64


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class GPSEnc(nn.Module):
    """Position encoder in the array's own frame.

    Input is the UE position relative to the BS (east, north) / 30 m. It is turned into the bearing
    BS->UE minus a boresight angle `theta` (a buffer, radians, 0 at training) and the range; the network
    sees [sin, cos of the relative bearing, range]. A re-mounted or relocated array changes only `theta`,
    which a calibrator can set at deployment; the learned map from relative bearing to beam is shared.
    """

    def __init__(self, d=128):
        super().__init__()
        self.register_buffer("theta", torch.zeros(()))
        self.net = nn.Sequential(nn.Linear(3, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d), nn.BatchNorm1d(d),
                                 nn.ReLU())

    def forward(self, x):
        bearing = torch.atan2(x[:, 0], x[:, 1]) - self.theta          # clockwise from north, as in the plots
        rng = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        return self.net(torch.stack([torch.sin(bearing), torch.cos(bearing), rng], 1))


class ConvEnc(nn.Module):
    def __init__(self, cin=1, d=128):
        super().__init__()
        self.net = nn.Sequential(conv_block(cin, 32), conv_block(32, 64), conv_block(64, d),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class BeamNet(nn.Module):
    def __init__(self, modalities=("gps", "cam", "radar"), cam_ch=1, d=128, mod_drop=0.3):
        super().__init__()
        self.modalities = tuple(modalities)
        self.mod_drop = mod_drop  # train-time prob. of zeroing a whole modality's features (never all of them)
        enc = {"gps": lambda: GPSEnc(d), "cam": lambda: ConvEnc(cam_ch, d), "radar": lambda: ConvEnc(1, d)}
        self.enc = nn.ModuleDict({m: enc[m]() for m in self.modalities})
        self.fuse = nn.Sequential(nn.Linear(d * len(self.modalities), 256), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Linear(256, N_BEAMS)

    def features(self, b):
        feats = [self.enc[m](b[m]) for m in self.modalities]
        if self.training and self.mod_drop > 0 and len(feats) > 1:
            n = feats[0].shape[0]
            keep = torch.rand(n, len(feats), device=feats[0].device) >= self.mod_drop
            keep[torch.arange(n), torch.randint(len(feats), (n,), device=keep.device)] = True  # keep >= 1
            feats = [f * keep[:, i:i + 1].float() for i, f in enumerate(feats)]
        return self.fuse(torch.cat(feats, 1))

    def forward(self, b):
        return self.head(self.features(b))


class PhysicsPrior(nn.Module):
    """Parametric prior over beams from the BS->UE bearing: mean beam = c0 + K * sin(bearing - theta).

    Returns log-probabilities of a discretised Gaussian of width `sigma` (beams) around the mean. c0, K,
    theta are buffers: fitted on the source scenario at training time and re-fitted (theta, K) at
    deployment by `PhysicsCalibrator` from sweep labels. No gradients flow into them.
    """

    def __init__(self, sigma=2.0, max_origin_m=25.0, K_prior_sigma=12.0, origin_prior_sigma_m=15.0, prior_weight=3.0):
        super().__init__()
        # soft priors for the deployment re-fit (MAP): K stays near the array's training value and the
        # origin correction stays within ~10 m; each 1-sigma deviation costs `prior_weight` beams of residual
        self.K_prior_sigma, self.origin_prior_sigma, self.prior_weight = K_prior_sigma, origin_prior_sigma_m / 30.0, prior_weight
        self.register_buffer("K_ref", torch.tensor(38.0))
        self.register_buffer("c0", torch.tensor(31.5))
        self.register_buffer("K", torch.tensor(38.0))
        self.register_buffer("theta", torch.tensor(0.0))
        # array phase-centre offset from the logged BS position (east, north), in the /30 m input units;
        # the logged position can be >10 m off, which bends the bearing->beam curve into parallel lines
        self.register_buffer("origin", torch.zeros(2))
        self.sigma, self.max_origin = sigma, max_origin_m / 30.0
        self.register_buffer("grid", torch.arange(N_BEAMS).float())

    def mean_beam(self, gps):
        rel = gps - self.origin
        bearing = torch.atan2(rel[:, 0], rel[:, 1])
        return self.c0 + self.K * torch.sin(bearing - self.theta)

    def forward(self, gps):
        mu = self.mean_beam(gps)[:, None]
        return F.log_softmax(-((self.grid[None] - mu) ** 2) / (2 * self.sigma**2), dim=1)

    @torch.no_grad()
    def fit(self, gps, y, fit_c0=True, fit_K=True, fit_origin=True, n_theta=360, local_theta=None, regularise=None):
        """Robust grid + least-squares fit of (c0, K, theta[, origin]) to labelled positions (numpy, CPU).

        theta is first found on a grid with the current (c0, K, origin); then the requested parameters
        are refined jointly with a soft-L1 loss. The origin is bounded to +-max_origin.
        """
        import numpy as np
        from scipy.optimize import least_squares
        g = gps.detach().cpu().numpy()
        yy = y.detach().cpu().numpy().astype(float)
        c0, K = self.c0.item(), self.K.item()
        if regularise is None:
            regularise = not fit_c0   # training fit (fit_c0=True) is unregularised; deployment re-fits are MAP
        ox, oy = self.origin.tolist()
        m = self.max_origin

        def mean(params):
            cc = params[0] if fit_c0 else c0
            kk = params[1] if fit_K else K
            dx, dy = (params[3], params[4]) if fit_origin else (ox, oy)
            b = np.arctan2(g[:, 0] - dx, g[:, 1] - dy)
            return cc + kk * np.sin(b - params[2])

        K_ref, w = self.K_ref.item(), self.prior_weight
        n_lab = max(len(yy), 1)

        def resid(params):
            r = mean(params) - yy
            if regularise:  # MAP: prior terms scaled so their pull does not vanish with many labels
                pen = np.array([(params[1] - K_ref) / self.K_prior_sigma,
                                params[3] / self.origin_prior_sigma, params[4] / self.origin_prior_sigma]) * w * np.sqrt(n_lab) / 4
                r = np.concatenate([r, pen])
            return r

        bear = np.arctan2(g[:, 0] - ox, g[:, 1] - oy)
        if local_theta is None:   # unknown boresight: global grid
            thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        else:                     # known to within a few degrees: local grid around it
            thetas = local_theta + np.deg2rad(np.arange(-10, 10.01, 0.5))
        sinb = np.sin(bear[None, :] - thetas[:, None])
        err = np.median(np.abs(np.clip(np.round(c0 + K * sinb), 0, 63) - yy[None, :]), axis=1)
        th = float(thetas[int(np.argmin(err))])
        lo = [0.0, 10.0, th - np.pi, -m, -m]
        hi = [63.0, 80.0, th + np.pi, m, m]
        if fit_origin:
            # multi-start: the origin optimum can be > 10 m from the current estimate and the bounded
            # solver creeps there from a single warm start; 9 starts on a +-8 m grid cure that
            step = 8.0 / 30.0
            starts = [(ox + i * step, oy + j * step) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            best = None
            for sx, sy in starts:
                x0 = [c0, K, th, float(np.clip(sx, -m, m)), float(np.clip(sy, -m, m))]
                x0 = [float(np.clip(v, l + 1e-6, h - 1e-6)) for v, l, h in zip(x0, lo, hi)]
                cand = least_squares(resid, x0, loss="soft_l1", bounds=(lo, hi), max_nfev=200)
                if best is None or cand.cost < best.cost:
                    best = cand
            sol = best
        else:
            x0 = [c0, K, th, ox, oy]
            x0 = [float(np.clip(v, l + 1e-6, h - 1e-6)) for v, l, h in zip(x0, lo, hi)]
            sol = least_squares(resid, x0, loss="soft_l1", bounds=(lo, hi), max_nfev=40)
        if fit_c0:
            self.c0.fill_(float(sol.x[0]))
        if fit_K:
            self.K.fill_(float(sol.x[1]))
        if fit_c0:                    # training fit: remember the array's K as the deployment prior
            self.K_ref.fill_(float(sol.x[1]))
        self.theta.fill_(float(sol.x[2]))
        if fit_origin:
            self.origin.copy_(torch.tensor([float(sol.x[3]), float(sol.x[4])], device=self.origin.device))
        return self


class AnchoredBeamNet(nn.Module):
    """Physics-anchored beam predictor: logits = log prior(bearing) + gate * residual(features).

    The residual network sees the camera (and, if present, radar) plus the position expressed in the
    array frame [sin, cos of bearing - theta, range] so it can learn lane/range corrections. `gate` is a
    scalar in [0, 1] (sigmoid of a buffer, 1 at training) that test-time methods may lower when the
    residual's statistics have shifted. Modality dropout as in BeamNet.
    """

    def __init__(self, modalities=("gps", "cam"), cam_ch=3, d=128, sigma=2.0, mod_drop=0.3):
        super().__init__()
        self.modalities = tuple(modalities)
        assert "gps" in self.modalities, "the anchored model needs position"
        self.prior = PhysicsPrior(sigma)
        self.mod_drop = mod_drop
        enc = {"gps": lambda: GPSEnc(d), "cam": lambda: ConvEnc(cam_ch, d), "radar": lambda: ConvEnc(1, d)}
        self.enc = nn.ModuleDict({m: enc[m]() for m in self.modalities})
        self.fuse = nn.Sequential(nn.Linear(d * len(self.modalities), 256), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Linear(256, N_BEAMS)
        self.register_buffer("gate_logit", torch.tensor(6.0))  # sigmoid(6) = 0.9975

    @property
    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def features(self, b):
        self.enc["gps"].theta = self.prior.theta  # position branch shares the calibrated boresight
        feats = [self.enc[m](b[m]) for m in self.modalities]
        if self.training and self.mod_drop > 0 and len(feats) > 1:
            n = feats[0].shape[0]
            keep = torch.rand(n, len(feats), device=feats[0].device) >= self.mod_drop
            keep[torch.arange(n), torch.randint(len(feats), (n,), device=keep.device)] = True
            feats = [f * keep[:, i:i + 1].float() for i, f in enumerate(feats)]
        return self.fuse(torch.cat(feats, 1))

    def residual(self, b):
        return self.head(self.features(b))

    def forward(self, b):
        return self.prior(b["gps"]) + self.gate * self.residual(b)


def build_model(arch, modalities, cam_ch=3, **kw):
    if arch == "anchored":
        return AnchoredBeamNet(modalities, cam_ch=cam_ch, **kw)
    return BeamNet(modalities, cam_ch=cam_ch, **kw)
