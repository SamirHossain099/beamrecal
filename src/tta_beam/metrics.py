"""Beam-prediction metrics: top-k accuracy, DeepSense DBA score, received-power loss."""
import numpy as np
import torch


def topk_correct(logits: torch.Tensor, y: torch.Tensor, ks=(1, 3)) -> dict:
    top = logits.topk(max(ks), dim=1).indices
    hit = top.eq(y[:, None])
    return {f"top{k}": hit[:, :k].any(1).float().sum().item() for k in ks}


def dba_terms(logits: torch.Tensor, y: torch.Tensor, delta: float = 5.0) -> float:
    """Sum over samples of the DeepSense 2022 DBA score (top-3, Δ=5); divide by N later.

    Y_k = 1 - mean_n min_{j<=k} min(|ŷ_nj - y_n| / Δ, 1);  DBA = mean(Y_1, Y_2, Y_3).
    """
    top3 = logits.topk(3, dim=1).indices.float()
    d = ((top3 - y[:, None].float()).abs() / delta).clamp(max=1.0)
    per_k = torch.stack([d[:, :k].min(1).values for k in (1, 2, 3)], 1)  # N x 3 distances
    return (1.0 - per_k).mean(1).sum().item()


def power_loss_db(logits: torch.Tensor, pwr: torch.Tensor) -> float:
    """Sum over samples of 10log10(P_opt / P_pred) using the measured 64-beam power vector."""
    pred = logits.argmax(1)
    p_pred = pwr.gather(1, pred[:, None]).squeeze(1).clamp_min(1e-12)
    p_opt = pwr.max(1).values.clamp_min(1e-12)
    return (10 * torch.log10(p_opt / p_pred)).sum().item()


def refine_loss_db(logits: torch.Tensor, pwr: torch.Tensor, k: int) -> float:
    """Sum of power loss (dB) when the BS measures the top-k predicted beams and serves the best of them."""
    top = logits.topk(k, dim=1).indices
    p_best_k = pwr.gather(1, top).max(1).values.clamp_min(1e-12)
    p_opt = pwr.max(1).values.clamp_min(1e-12)
    return (10 * torch.log10(p_opt / p_best_k)).sum().item()


class Meter:
    def __init__(self):
        self.n = 0
        self.s = {"top1": 0.0, "top3": 0.0, "dba": 0.0, "ploss_db": 0.0, "within3": 0.0,
                  "ploss_k3_db": 0.0, "ploss_k5_db": 0.0}

    def update(self, logits, y, pwr):
        logits, y, pwr = logits.detach().float().cpu(), y.cpu(), pwr.cpu()
        for k, v in topk_correct(logits, y).items():
            self.s[k] += v
        self.s["dba"] += dba_terms(logits, y)
        self.s["ploss_db"] += power_loss_db(logits, pwr)
        self.s["within3"] += ((logits.argmax(1) - y).abs() <= 3).float().sum().item()
        self.s["ploss_k3_db"] += refine_loss_db(logits, pwr, 3)
        self.s["ploss_k5_db"] += refine_loss_db(logits, pwr, 5)
        self.n += len(y)

    def result(self):
        return {k: v / max(self.n, 1) for k, v in self.s.items()} | {"n": self.n}
