# beamrecal: geometric recalibration for sensing-aided mmWave beam prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23021570.svg)](https://doi.org/10.5281/zenodo.23021570)

Code and results for *Geometric Recalibration for Sensing-Aided Millimeter-Wave Beam Prediction Under
Deployment Shift* (Samir Hossain, Texas Tech University).

A beam predictor trained on sensing data (position, camera, radar) at one DeepSense 6G site fails at
another: cross-site top-1 accuracy is at most 0.064, and six test-time adaptation methods (norm
statistics, Tent, EATA, SAR, CoTTA, T3A) keep it at or below 0.079. The shift is geometric. The map from
sensed position to beam index is rotated by the orientation of the array and shifted by an error of the
logged base-station position. What transfers is an input tied to the unit's geometry, the vehicle's
position in the camera image: a pinhole map or a small network on it carries over between sites of one
unit. The physical form is the better one to recalibrate: refitted from the periodic beam sweeps a base
station already performs, it loses less received power than the network fine-tuned on the same sweeps.

Every number in the paper is computed from `results/` by a script; `tests/test_results.py` recomputes the
aggregated tables (`headline.json`, `ablation.json`) from the per-run files, and
`python scripts/manuscript_numbers.py` reproduces every number quoted in the paper.

## Layout

| Path | Contents |
|---|---|
| `src/tta_beam/` | data loading for both DeepSense testbeds, learned models, the physics prior, calibrators, TTA methods, geometric methods, streaming evaluator, metrics |
| `scripts/` | training, streaming runs, transfer matrices, detection, figures, and the scripts that compute every reported number |
| `results/` | one JSON per streaming run, plus the aggregated `headline.json`, `ablation.json`, `manuscript_numbers.json` |
| `figures/` | the paper's figures (PNG and PDF) |
| `tests/` | checks that the paper's numbers and tables match `results/` |

## Data

DeepSense 6G (https://www.deepsense6g.net) is distributed by its maintainers under CC BY-NC-ND 4.0
and is not redistributed here, nor is anything derived from its images. Download scenarios 1 to 9 and
31 to 35 and extract them to `<root>/scenarioNN/`, then set `DEEPSENSE_ROOT=<root>`. The first run of
any script parses each scenario and caches it to `data/cache/`.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/train_source.py --source scenario1 --modalities gps cam --arch anchored --seed 0
python scripts/detect_vehicles.py --scenarios scenario1 scenario2 scenario5 scenario6 scenario7   # cached per scenario
python scripts/run_stream.py --ckpt checkpoints/src_scenario1_gps-cam_anchored_s0.pt \
    --stream scenario2 scenario5 scenario6 scenario7 --sweep_every 20 --methods det-assoc sense-geo
python scripts/headline.py && python scripts/ablation.py && python scripts/manuscript_numbers.py
python scripts/write_manuscript_table.py && python -m pytest tests/ -q
```

The streams are Track A (source 32; stream 33, 34, 31, 35), Track B (source 1; stream 2, 5, 6, 7) and
cross-unit (source 1; stream 8, 9, 6). `python scripts/run_stream.py --help` lists the methods. A
synthetic smoke test needs no data: `python scripts/train_source.py --synthetic --source syn32`.

## Citing

Cite the concept DOI, which always resolves to the latest release: https://doi.org/10.5281/zenodo.23021570. `CITATION.cff` has the full entry.

## License

MIT for the code. The DeepSense 6G data keep their own license.
