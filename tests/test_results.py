"""The aggregated tables in results/ are what the per-run stream files give (public-release test).

Run: python -m pytest tests/ -q
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_headline_json_matches_the_stream_files():
    h = _load("headline")
    saved = json.loads((ROOT / "results" / "headline.json").read_text())
    for (stream, label, tag, key), row in zip(h.ROWS, saved):
        k, stats, n = h.collect(tag, key)
        assert (row["stream"], row["label"], row["method"], row["seeds"]) == (stream, label, k, n)
        for m in h.METRICS:
            assert abs(row[m] - stats[m][0]) < 6e-4, (label, m)


def test_ablation_json_matches_the_stream_files():
    a = _load("ablation")
    saved = json.loads((ROOT / "results" / "ablation.json").read_text())
    for (stream, label, tag, key), row in zip(a.ROWS, saved):
        stats, n = a.collect(tag, key)
        assert (row["stream"], row["method"], row["seeds"]) == (stream, key, n)
        for m in a.METRICS:
            assert abs(row[m] - stats[m][0]) < 6e-4, (label, m)


def test_every_headline_row_has_three_seeds():
    assert all(r["seeds"] == 3 for r in json.loads((ROOT / "results" / "headline.json").read_text()))
    assert all(r["seeds"] == 3 for r in json.loads((ROOT / "results" / "ablation.json").read_text()))
