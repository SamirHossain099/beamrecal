"""Component ablation of the camera pipeline (Table 3 of the paper), written to results/ablation.json.

Same aggregation as headline.py: mean and standard deviation over the source-model seeds of one tag.

    python scripts/ablation.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# (stream, manuscript label, file tag, method key). Annotated-box rows exist only where DeepSense
# ships boxes (Testbed 1); Track A has detections only.
ROWS = [
    ("track_a", "Detections, map refit on sweeps (pipeline)", "_D1_trackA_det_K20", "det-assoc"),
    ("track_a", "Detections, map fixed at source fit", "_D1_trackA_det_K20", "det-assoc-fixedmap"),
    ("track_a", "Detections, no sweeps", "_D1_trackA_det_K20", "det-assoc-selfcal"),
    ("track_b", "Last sweep held", "_G2_trackB_geo_K20", "sweep-hold"),
    ("track_b", "Annotated box, map refit on sweeps", "_G2_trackB_geo_K20", "sense-geo+calib"),
    ("track_b", "Annotated box, offset tracking", "_G2_trackB_geo_K20", "offset-ma"),
    ("track_b", "Detections, map refit on sweeps (pipeline)", "_D2_trackB_det_K20", "det-assoc"),
    ("track_b", "Detections, map fixed at source fit", "_D2_trackB_det_K20", "det-assoc-fixedmap"),
    ("track_b", "Detections, no sweeps", "_D2_trackB_det_K20", "det-assoc-selfcal"),
    ("cross_unit", "Last sweep held", "_G3_crossunit_geo_K20", "sweep-hold"),
    ("cross_unit", "Annotated box, map refit on sweeps", "_G3_crossunit_geo_K20", "sense-geo+calib"),
    ("cross_unit", "Annotated box, offset tracking", "_G3_crossunit_geo_K20", "offset-ma"),
    ("cross_unit", "Detections, map refit on sweeps (pipeline)", "_D3_crossunit_det_K20", "det-assoc"),
    ("cross_unit", "Detections, map fixed at source fit", "_D3_crossunit_det_K20", "det-assoc-fixedmap"),
    ("cross_unit", "Detections, no sweeps", "_D3_crossunit_det_K20", "det-assoc-selfcal"),
]
METRICS = ("top1", "dba", "ploss_db", "ploss_k3_db")


def collect(tag, key):
    runs = {}
    for f in sorted(R.glob(f"stream_*{tag}_*.json")):
        r = json.loads(f.read_text())
        if key in r["methods"]:
            runs[r["seed"]] = r["methods"][key]["mean"]
    if not runs:
        raise FileNotFoundError(f"no results for {tag} / {key}")
    v = list(runs.values())
    return {m: (float(np.mean([x[m] for x in v])), float(np.std([x[m] for x in v]))) for m in METRICS}, len(v)


def main():
    out = []
    for stream, label, tag, key in ROWS:
        stats, n = collect(tag, key)
        out.append({"stream": stream, "label": label, "method": key, "seeds": n,
                    **{m: round(stats[m][0], 3) for m in METRICS},
                    **{m + "_std": round(stats[m][1], 3) for m in METRICS}})
        print(f"{stream:10s} {label:45s} n={n} top1 {stats['top1'][0]:.3f} ± {stats['top1'][1]:.3f} "
              f"DBA {stats['dba'][0]:.3f} loss {stats['ploss_db'][0]:.2f} top3 {stats['ploss_k3_db'][0]:.2f}")
    (R / "ablation.json").write_text(json.dumps(out, indent=1))
    print("wrote results/ablation.json")


if __name__ == "__main__":
    main()
