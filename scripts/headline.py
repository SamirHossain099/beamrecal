"""Compute the paper's headline numbers from results/ and write results/headline.json.

Table 2 of the paper is generated from this file; tests/test_results.py checks it against the
per-run stream files.

    python scripts/headline.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# (stream label, method label, file tag, method key)
ROWS = [
    ("track_a", "source (plain, the model TTA adapts)", "_T1_trackA_tta6", "source"),
    ("track_a", "TTA, mean of six methods", "_T1_trackA_tta6", ("mean", ["norm", "tent", "eata", "sar", "cotta", "t3a"])),
    ("track_a", "supervised online FT (tuned, plain)", "_E1_plain_supft_lr1e-3_K20", "supft"),
    ("track_a", "physics-anchored net: calib+gate+norm", "_E1v24_anchored_K20", "calib+gate+norm"),
    ("track_a", "camera + GPS association (detector): det-assoc", "_D1_trackA_det_K20", "det-assoc"),
    ("track_b", "source (plain, the model TTA adapts)", "_T1_trackB_tta6", "source"),
    ("track_b", "TTA, mean of six methods", "_T1_trackB_tta6", ("mean", ["norm", "tent", "eata", "sar", "cotta", "t3a"])),
    ("track_b", "supervised online FT (tuned, plain)", "_E2_plain_supft_lr1e-3_K20", "supft"),
    ("track_b", "physics-anchored net: calib+supft", "_E2v24_trackB_anchored_K20", "calib+supft"),
    ("track_b", "camera map, zero target labels: sense-geo", "_G2_trackB_geo_K20", "sense-geo"),
    ("track_b", "camera + GPS association (detector): det-assoc", "_D2_trackB_det_K20", "det-assoc"),
    ("cross_unit", "source (plain, the model TTA adapts)", "_T1_crossunit_tta6", "source"),
    ("cross_unit", "TTA, mean of six methods", "_T1_crossunit_tta6", ("mean", ["norm", "tent", "eata", "sar", "cotta", "t3a"])),
    ("cross_unit", "supervised online FT (tuned, plain)", "_G3_crossunit_plain_K20", "supft"),
    ("cross_unit", "camera map, zero target labels: sense-geo", "_G3_crossunit_geo_K20", "sense-geo"),
    ("cross_unit", "camera + GPS association (detector): det-assoc", "_D3_crossunit_det_K20", "det-assoc"),
]
METRICS = ("top1", "top3", "dba", "ploss_db")


def collect(tag, key):
    """Mean and standard deviation over seeds. `key` is one method, a list (the best of them by top-1, kept
    for older tables), or ("mean", [methods]): per seed the average over the methods, which is how Table 2
    reports TTA, so no method is selected on the test stream."""
    files = sorted(R.glob(f"stream_*{tag}_*.json"))
    if isinstance(key, tuple) and key[0] == "mean":
        runs = []
        for f in files:
            m = json.loads(f.read_text())["methods"]
            assert all(k in m for k in key[1]), (f.name, key[1])
            runs.append({x: float(np.mean([m[k]["mean"][x] for k in key[1]])) for x in METRICS})
        if not runs:
            raise FileNotFoundError(f"no results for {tag} / {key}")
        return "mean-of-" + str(len(key[1])), {x: (float(np.mean([r[x] for r in runs])),
                                                  float(np.std([r[x] for r in runs]))) for x in METRICS}, len(runs)
    by = {}
    for f in files:
        r = json.loads(f.read_text())
        keys = key if isinstance(key, list) else [key]
        for k in keys:
            if k in r["methods"]:
                by.setdefault(k, {})[r["seed"]] = r["methods"][k]["mean"]
    if not by:
        raise FileNotFoundError(f"no results for {tag} / {key}")
    # "best of" rows: the method with the highest mean top-1
    k = max(by, key=lambda m: np.mean([v["top1"] for v in by[m].values()]))
    runs = list(by[k].values())
    return k, {m: (float(np.mean([x[m] for x in runs])), float(np.std([x[m] for x in runs]))) for m in METRICS}, len(runs)


def main():
    out = []
    for stream, label, tag, key in ROWS:
        k, stats, n = collect(tag, key)
        out.append({"stream": stream, "label": label, "method": k, "seeds": n,
                    **{m: round(stats[m][0], 3) for m in METRICS}, **{m + "_std": round(stats[m][1], 3) for m in METRICS}})
        print(f"{stream:10s} {label:50s} [{k:16s}] n={n}  top1 {stats['top1'][0]:.3f} ± {stats['top1'][1]:.3f}  "
              f"top3 {stats['top3'][0]:.3f}  DBA {stats['dba'][0]:.3f}  loss {stats['ploss_db'][0]:.2f} dB")
    (R / "headline.json").write_text(json.dumps(out, indent=1))
    print("wrote results/headline.json")


if __name__ == "__main__":
    main()
