"""Every TTA method on every stream (Table 4 of the paper), written to results/tta6.json.

Reads the `_T1_*_tta6` runs: the plain source model and the six TTA methods, a sweep every 20 frames, three
seeds. Reporting every method, rather than the best one chosen on the test stream, is the point of this table.

    python scripts/tta_table.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
STREAMS = {"track_a": "_T1_trackA_tta6", "track_b": "_T1_trackB_tta6", "cross_unit": "_T1_crossunit_tta6"}
METHODS = {"source": "Source model", "norm": "Normalization statistics", "tent": "Tent", "eata": "EATA",
           "sar": "SAR", "cotta": "CoTTA", "t3a": "T3A"}
METRICS = ("top1", "dba", "ploss_db")


def main():
    out = []
    for stream, tag in STREAMS.items():
        runs = [json.loads(f.read_text())["methods"] for f in sorted(R.glob(f"stream_*{tag}_*.json"))]
        assert len(runs) == 3, (stream, len(runs))
        for m, label in METHODS.items():
            v = {x: [r[m]["mean"][x] for r in runs] for x in METRICS}
            out.append({"stream": stream, "method": m, "label": label, "seeds": len(runs),
                        **{x: round(float(np.mean(v[x])), 3) for x in METRICS},
                        **{x + "_std": round(float(np.std(v[x])), 3) for x in METRICS}})
    (R / "tta6.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"{r['stream']:10s} {r['label']:26s} top1 {r['top1']:.3f} ± {r['top1_std']:.3f}  "
              f"DBA {r['dba']:.3f}  loss {r['ploss_db']:.2f}")
    print("wrote results/tta6.json")


if __name__ == "__main__":
    main()
