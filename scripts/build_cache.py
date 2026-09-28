"""Parse every wanted DeepSense scenario into data/cache/scenarioNN.npz.

Waits for each scenario's extraction to finish (reads the extractor's log) so it can run alongside it.

    python scripts/build_cache.py --scenarios 1 2 3 4 5 6 7 8 9 31 32 33 34 35
"""
import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tta_beam.data import CACHE, RAW, load_scenario  # noqa: E402

LOG = RAW / "extract_log.txt"


def extracted(n: int) -> bool:
    if not LOG.exists():
        return (RAW / f"scenario{n}").exists()
    txt = LOG.read_text()
    return "ALL DONE" in txt or re.search(rf"done [Ss]cenario{n}[_.]", txt) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32, 33, 34, 35])
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for n in a.scenarios:
        name = f"scenario{n}"
        if (CACHE / f"{name}.npz").exists():
            print("cached", name, flush=True)
            continue
        while not extracted(n):
            print("waiting for extraction of", name, flush=True)
            time.sleep(30)
        t = time.time()
        d = load_scenario(name)
        print(f"{name}: {len(d['y'])} samples, {len(set(d['seq'].tolist()))} sequences, "
              f"radar={'yes' if d['radar'].any() else 'no'}, {time.time() - t:.0f}s", flush=True)
    print("CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
