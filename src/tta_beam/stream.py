"""Online streaming evaluator.

Frames arrive in time order, vehicle pass by vehicle pass, domain by domain. For each batch of consecutive
frames the method predicts (`step`), the prediction is scored, and only then does the method receive the
feedback a base station would actually get about those frames (`feedback`):

  served      the beam it served = argmax of the returned logits (exploration counts against the score)
  p_served    measured power on that beam
  served_low  served power fell > `low_db` below a decaying per-pass reference
  sweep       frames on which a full P1 sweep happened (every `sweep_every` frames; 0 = never)
  y_sweep     true best beam on those frames (only there)
  seq         pass id per frame
"""
import time
import numpy as np
import torch

from .data import BeamSet
from .metrics import Meter


def make_stream(domains: list, names: list):
    """Parts are keyed by visit: a scenario visited twice gets 'scenario33' and 'scenario33#2' so that
    return visits are scored separately (recovery after a site change is a question of its own)."""
    parts, seen = [], {}
    for name, d in zip(names, domains):
        order = np.lexsort((np.arange(len(d["y"])), d["seq"]))  # by pass, then time
        s = BeamSet(d, order)
        s.t["seq"] = s.t["seq"] + 100000 * len(parts)            # unique pass ids across domains
        seen[name] = seen.get(name, 0) + 1
        parts.append((name if seen[name] == 1 else f"{name}#{seen[name]}", s))
    return parts


def run(method, parts, batch_size=16, device="cuda", sweep_every=0, low_db=3.0, decay_db=0.3):
    out, frame, lat = {}, 0, []
    ref = {}  # pass id -> reference power (dB-decaying max of served power, reset by sweeps)
    for name, ds in parts:
        meter = Meter()
        for i in range(0, len(ds), batch_size):
            b = {k: v[i:i + batch_size].to(device) for k, v in ds.t.items()}
            n = len(b["y"])
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = method.step(b)
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000 / n)
            meter.update(logits, b["y"], b["pwr"])

            sweep = torch.zeros(n, dtype=torch.bool, device=device)
            if sweep_every:
                sweep = (torch.arange(frame, frame + n, device=device) % sweep_every) == 0
            frame += n
            served = logits.argmax(1)
            p_served = b["pwr"].gather(1, served[:, None]).squeeze(1)
            low = torch.zeros(n, dtype=torch.bool, device=device)
            ps, pb = p_served.tolist(), b["pwr"].max(1).values.tolist()
            for j, s in enumerate(b["seq"].tolist()):
                r = ref.get(s, 0.0) * 10 ** (-decay_db / 10)
                if sweep[j]:
                    r = pb[j]
                low[j] = bool(r > 0 and 10 * np.log10(r / max(ps[j], 1e-12)) > low_db)
                ref[s] = max(r, ps[j])
            fb = {"served": served, "p_served": p_served, "served_low": low, "sweep": sweep,
                  "y_sweep": b["y"][sweep], "seq": b["seq"]}
            t0 = time.perf_counter()
            method.feedback(b, fb)
            if device == "cuda":
                torch.cuda.synchronize()
            lat[-1] += (time.perf_counter() - t0) * 1000 / n
        out[name] = meter.result()
    out["mean"] = {k: float(np.mean([out[d][k] for d, _ in parts]))
                   for k in ("top1", "top3", "dba", "ploss_db", "within3", "ploss_k3_db", "ploss_k5_db")}
    out["latency_ms_per_frame"] = float(np.median(lat))
    if hasattr(type(method), "calibrator_resets"):
        out["calibrator_resets"] = int(method.calibrator_resets)
    phys = getattr(method, "phys", None)
    if phys is not None:
        out["calibrator_resets"] = int(phys.n_resets)
        out["prior_final"] = {k: float(getattr(method.model.prior, k)) for k in ("c0", "K", "theta")}
        out["prior_final"]["origin_m"] = [float(v) * 30 for v in method.model.prior.origin.tolist()]
    return out
