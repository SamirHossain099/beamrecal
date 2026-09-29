"""The deployable pipeline with degraded position reports (results/position.json, Figure 6 of the paper).

The pipeline reads the reported vehicle position to pick the served vehicle's box and to calibrate the
position prior at sweeps. The `_P1_*` runs replace it everywhere by the position `pos_delay` frames earlier
in the same pass, or by the position plus Gaussian noise of `pos_noise_m` metres per axis, a sweep every
20 frames, three seeds. The undegraded point is the pipeline's own run in Table 2 (`_D*_det_K20`).
Frame intervals are 92 to 200 ms in these scenarios (about 10 Hz), so d frames is about d/10 s.

    python scripts/position_degradation.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
STREAMS = {"track_a": ("A", "_D1_trackA_det_K20"), "track_b": ("B", "_D2_trackB_det_K20"),
           "cross_unit": ("X", "_D3_crossunit_det_K20")}
DELAYS = (0, 2, 5, 10, 20)
NOISES = (0, 2, 5, 10)
METRICS = ("top1", "dba", "ploss_db")


def stats(tag):
    rs = [json.loads(f.read_text())["methods"]["det-assoc"]["mean"] for f in sorted(R.glob(f"stream_*{tag}_*.json"))]
    assert len(rs) == 3, (tag, len(rs))
    return {**{x: float(np.mean([r[x] for r in rs])) for x in METRICS},
            **{x + "_std": float(np.std([r[x] for r in rs])) for x in METRICS}, "seeds": len(rs)}


def main():
    head = {(r["stream"], r["label"].split(":")[0]): r for r in json.loads((R / "headline.json").read_text())}
    out = []
    for stream, (short, base) in STREAMS.items():
        ft = next(r for (s, lab), r in head.items() if s == stream and lab.startswith("supervised online FT"))
        for kind, levels in (("delay", DELAYS), ("noise", NOISES)):
            for v in levels:
                st = stats(base if v == 0 else f"_P1_{short}_{kind}{v}")
                out.append({"stream": stream, "kind": kind, "level": v, **st, "supft_ploss_db": ft["ploss_db"]})
    (R / "position.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"{r['stream']:10s} {r['kind']:5s} {r['level']:3d}  loss {r['ploss_db']:.2f} ± {r['ploss_db_std']:.2f}  "
              f"top1 {r['top1']:.3f}  DBA {r['dba']:.3f}   (tuned FT {r['supft_ploss_db']:.2f})")
    print("wrote results/position.json")


if __name__ == "__main__":
    main()
