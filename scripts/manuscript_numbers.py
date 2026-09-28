"""Every number quoted in the paper's prose, computed from results/.

Writes results/manuscript_numbers.json: {name: {"value": float, "text": formatted string, "source": ...}}.
The paper's tests assert that each "text" appears verbatim in the manuscript where the prose uses it.

    python scripts/manuscript_numbers.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

OUT = {}


def put(name, value, fmt, source):
    OUT[name] = {"value": float(value), "text": fmt.format(value), "source": source}


def transfer_matrices():
    """Latest 3-seed transfer files: Track A (31-35) and Track B core (1, 2, 5, 8, 9)."""
    mats = {}
    for tag in ("_trackA_3seeds", "_trackB_core_3seeds", "_trackB_3seeds"):
        fs = sorted(R.glob(f"transfer{tag}_*.json"))
        if fs:
            mats[tag] = json.loads(fs[-1].read_text())
    return mats


# unit/site identity: scenarios recorded at the same spot (same logged BS position)
SAME_SPOT = [{"scenario1", "scenario2"}, {"scenario3", "scenario4"}, {"scenario8", "scenario9"}, {"scenario32", "scenario33"}]


def same_spot(a, b):
    return any({a, b} <= s for s in SAME_SPOT)


def main():
    mats = transfer_matrices()
    # --- E0: within-site ranges and cross-scenario maxima -----------------------------------------------
    cross_site, cross_same, within = [], [], {}
    for tag, m in mats.items():
        if tag == "_trackB_3seeds":   # superseded for gps rows by the calibrated-GPS core run; keep cam only
            keep = {"cam"}
        else:
            keep = set(m["results"])
        for mod, res in m["results"].items():
            if mod not in keep:
                continue
            for s, row in res.items():
                for t, v in row.items():
                    top1 = float(np.mean(v["top1"]))
                    if s == t:
                        within.setdefault(mod, {})[s] = top1
                    elif same_spot(s, t):
                        cross_same.append((top1, mod, s, t))
                    else:
                        cross_site.append((top1, mod, s, t))
    mx = max(cross_site)
    put("cross_site_max_top1", mx[0], "{:.3f}", f"E0 max over different-site pairs: {mx[1]} {mx[2]}->{mx[3]}")
    mxs = max(cross_same)
    put("cross_samespot_max_top1", mxs[0], "{:.3f}", f"E0 max over same-spot pairs: {mxs[1]} {mxs[2]}->{mxs[3]}")
    ta = mats["_trackA_3seeds"]["results"]
    ta_cross = [float(np.mean(v["top1"])) for mod in ta for s, row in ta[mod].items() for t, v in row.items()
                if s != t and not same_spot(s, t)]
    ta_dba = [float(np.mean(v["dba"])) for mod in ta for s, row in ta[mod].items() for t, v in row.items()
              if s != t and not same_spot(s, t)]
    put("trackA_cross_site_max_dba", max(ta_dba), "{:.2f}", "E0 Track A, different-site pairs, all modalities")
    g = [within["gps"][s] for s in within["gps"]]
    put("within_gps_min", min(g), "{:.2f}", "E0 within-site gps top-1, all scenarios in the matrices")
    put("within_gps_max", max(g), "{:.2f}", "E0 within-site gps top-1")

    # --- source models of the three streams (within-site val top-1) ------------------------------------
    import torch
    vals = []
    for ck in list((ROOT / "checkpoints").glob("src_scenario32_gps-cam_anchored_s*.pt")) + \
            list((ROOT / "checkpoints").glob("src_scenario1_gps-cam_anchored_s*.pt")):
        vals.append(torch.load(ck, map_location="cpu", weights_only=False)["val"]["top1"])
    put("source_within_min", min(vals), "{:.2f}", "anchored source checkpoints (32, 1), 3 seeds each")
    put("source_within_max", max(vals), "{:.2f}", "anchored source checkpoints (32, 1)")

    # --- headline table (already pinned) -----------------------------------------------------------------
    hl = {(r["stream"], r["label"].split(":")[0]): r for r in json.loads((R / "headline.json").read_text())}

    def h(stream, prefix):
        return next(r for (s, lab), r in hl.items() if s == stream and lab.startswith(prefix))
    det = [h(s, "camera + GPS association")["ploss_db"] for s in ("track_a", "track_b", "cross_unit")]
    ft = [h(s, "supervised online FT")["ploss_db"] for s in ("track_a", "track_b", "cross_unit")]
    put("det_loss_min", min(det), "{:.2f}", "headline det-assoc power loss")
    put("det_loss_max", max(det), "{:.2f}", "headline det-assoc power loss")
    put("ft_loss_min", min(ft), "{:.2f}", "headline tuned supft power loss")
    put("ft_loss_max", max(ft), "{:.2f}", "headline tuned supft power loss")
    for s, n in (("track_a", "A"), ("track_b", "B"), ("cross_unit", "X")):
        put(f"det_minus_ft_loss_{n}", h(s, "supervised online FT")["ploss_db"] - h(s, "camera + GPS association")["ploss_db"],
            "{:.2f}", "headline difference, dB")
        put(f"det_minus_ft_dba_{n}", h(s, "camera + GPS association")["dba"] - h(s, "supervised online FT")["dba"],
            "{:.2f}", "headline difference, DBA")
    for s, n in (("track_a", "A"), ("track_b", "B")):
        put(f"tta_best_loss_{n}", h(s, "best generic TTA")["ploss_db"], "{:.2f}", "headline best TTA loss")
        put(f"source_loss_{n}", h(s, "source")["ploss_db"], "{:.2f}", "headline source loss")
        put(f"ft_top1_{n}", h(s, "supervised online FT")["top1"], "{:.2f}", "headline supft top-1")

    # --- overhead (seed 0) --------------------------------------------------------------------------------
    oh = json.loads((R / "overhead_trackB.json").read_text())
    sg0 = next(r for r in oh if r["method"] == "sense-geo" and r["K"] == 0 and r["k"] == 1)
    sg5 = next(r for r in oh if r["method"] == "sense-geo" and r["K"] == 0 and r["k"] == 5)
    put("oh_camera_0meas", sg0["loss_db"], "{:.2f}", "overhead_trackB sense-geo K=0 k=1")
    put("oh_camera_5meas", sg5["loss_db"], "{:.2f}", "overhead_trackB sense-geo K=0 k=5")
    learned = [r for r in oh if r["method"] in ("calib+gate+norm", "calib+supft", "supft", "tent", "source")]
    best_learned = min(learned, key=lambda r: r["loss_db"])
    put("oh_best_learned_loss", best_learned["loss_db"], "{:.2f}", f"overhead best learned: {best_learned['method']}")
    put("oh_best_learned_meas", best_learned["overhead_beams_per_frame"], "{:.1f}", "its measurements per frame")
    below6 = [r["loss_db"] for r in learned if r["overhead_beams_per_frame"] < 6]
    put("oh_learned_min_below6", min(below6), "{:.1f}", "min learned loss below 6 measurements/frame")
    hold = [r["loss_db"] for r in oh if r["method"] == "sweep-hold"]
    put("oh_hold_min", min(hold), "{:.1f}", "sweep-hold loss range")
    put("oh_hold_max", max(hold), "{:.1f}", "sweep-hold loss range")

    # --- figure 2 data: geometry ---------------------------------------------------------------------------
    geo = json.loads((R / "fig2_geometry.json").read_text())
    for s_ in ("scenario2", "scenario5"):
        put(f"geo_w3_no_{s_}", geo[s_]["within3_no_origin"], "{:.2f}", "fig2_geometry.json")
        put(f"geo_w3_yes_{s_}", geo[s_]["within3_with_origin"], "{:.2f}", "fig2_geometry.json")
        put(f"geo_origin_m_{s_}", float(np.hypot(*geo[s_]["origin_m"])), "{:.1f}", "fig2_geometry.json |o|")
    w3 = [v["within3_with_origin"] for v in geo.values()]
    put("geo_w3_min", min(w3), "{:.2f}", "fig2_geometry.json, all 11 scenarios, with origin")
    put("geo_w3_max", max(w3), "{:.2f}", "fig2_geometry.json")
    ks = [v["K"] for v in geo.values()]
    put("geo_K_min", min(ks), "{:.0f}", "fitted codebook scale K")
    put("geo_K_max", max(ks), "{:.0f}", "fitted codebook scale K")

    # --- figure 3 data: camera map transfer -----------------------------------------------------------
    cam = json.loads((R / "fig3_camera_transfer.json").read_text())
    sc = cam["scenarios"]
    W3, T1 = np.array(cam["within3"]), np.array(cam["top1"])
    bs1 = [sc.index(x) for x in ("scenario1", "scenario2", "scenario5", "scenario7")]
    off = [(i, j) for i in bs1 for j in bs1 if i != j]
    put("cam_bs1_w3_min", min(W3[i, j] for i, j in off), "{:.2f}", "fig3 off-diagonal within unit BS1")
    put("cam_bs1_w3_max", max(W3[i, j] for i, j in off), "{:.2f}", "fig3")
    put("cam_bs1_top1_min", min(T1[i, j] for i, j in off), "{:.2f}", "fig3")
    put("cam_bs1_top1_max", max(T1[i, j] for i, j in off), "{:.2f}", "fig3")
    diag = [T1[i, i] for i in bs1]
    put("cam_bs1_self_top1_min", min(diag), "{:.2f}", "fig3 diagonal (in-sample)")
    put("cam_bs1_self_top1_max", max(diag), "{:.2f}", "fig3 diagonal")
    i1 = sc.index("scenario1")
    put("cam_1_to_8_w3", W3[i1, sc.index("scenario8")], "{:.2f}", "fig3")
    put("cam_1_to_6_w3", W3[i1, sc.index("scenario6")], "{:.2f}", "fig3")

    # --- detector vs annotation -----------------------------------------------------------------------
    dm = json.loads((R / "detector_match.json").read_text())
    m02 = [v["within_0.02"] for v in dm.values() if "within_0.02" in v]
    put("det_match_min_pct", 100 * min(m02), "{:.1f}", "detector_match.json")
    put("det_match_max_pct", 100 * max(m02), "{:.0f}", "detector_match.json")
    dpf = [v["detections_per_frame"] for v in dm.values()]
    put("det_per_frame_min", min(dpf), "{:.1f}", "detector_match.json")
    put("det_per_frame_max", max(dpf), "{:.0f}", "detector_match.json")

    # --- cross-unit stream: refit vs offset tracking vs fixed map (3 seeds, deterministic) --------------
    def stream_mean(tag, method, key):
        v = [json.loads(f.read_text())["methods"][method]["mean"][key] for f in R.glob(f"stream_*{tag}_*.json")
             if method in json.loads(f.read_text())["methods"]]
        return float(np.mean(v))
    put("xu_refit_top1", stream_mean("_G3_crossunit_geo_K20", "sense-geo+calib", "top1"), "{:.3f}", "G3")
    put("xu_refit_dba", stream_mean("_G3_crossunit_geo_K20", "sense-geo+calib", "dba"), "{:.3f}", "G3")
    put("xu_fixed_top1", stream_mean("_G3_crossunit_geo_K20", "sense-geo", "top1"), "{:.3f}", "G3")
    put("xu_fixed_dba", stream_mean("_G3_crossunit_geo_K20", "sense-geo", "dba"), "{:.3f}", "G3")
    put("xu_refit_loss", stream_mean("_G3_crossunit_geo_K20", "sense-geo+calib", "ploss_db"), "{:.2f}", "G3")
    put("xu_offset_loss", stream_mean("_G3_crossunit_geo_K20", "offset-ma", "ploss_db"), "{:.2f}", "G3")

    # --- label efficiency (E4, Track A, 3 seeds) ------------------------------------------------------
    def e4(tag, method, K):
        v = [json.loads(f.read_text())["methods"][method]["mean"]["top1"] for f in R.glob(f"stream_*_K{K}_s*{tag}_*.json")
             if method in json.loads(f.read_text())["methods"]]
        return float(np.mean(v)), len(v)
    for K in (10, 100):
        ours, n1 = e4("_E4v22_anchored", "calib+gate+norm", K)
        ft, n2 = e4("_E4v22_plain_supft", "supft", K)
        assert n1 == n2 == 3, (K, n1, n2)
        put(f"e4_gap_K{K}", ours - ft, "{:.2f}", f"E4v22 K={K}, 3 seeds")
        if K == 100:
            put(f"e4_ft_K{K}", ft, "{:.2f}", f"E4v22 supft K={K}")

    # --- component ablation (Table 3, results/ablation.json from scripts/ablation.py) ------------------
    abl = {(r["stream"], r["method"]): r for r in json.loads((R / "ablation.json").read_text())}
    put("abl_fixed_loss_B", abl[("track_b", "det-assoc-fixedmap")]["ploss_db"], "{:.2f}", "ablation Track B fixed map")
    put("abl_fixed_loss_X", abl[("cross_unit", "det-assoc-fixedmap")]["ploss_db"], "{:.2f}", "ablation cross-unit fixed map")
    put("abl_annot_loss_B", abl[("track_b", "sense-geo+calib")]["ploss_db"], "{:.2f}", "ablation Track B annotated box, refit")
    put("abl_pipe_loss_B", abl[("track_b", "det-assoc")]["ploss_db"], "{:.2f}", "ablation Track B pipeline")
    nosw = [abl[(s, "det-assoc-selfcal")]["ploss_db"] for s in ("track_a", "track_b", "cross_unit")]
    put("abl_nosweep_loss_min", min(nosw), "{:.2f}", "ablation no-sweep variant")
    put("abl_nosweep_loss_max", max(nosw), "{:.2f}", "ablation no-sweep variant")
    hold = [abl[(s, "sweep-hold")]["ploss_db"] for s in ("track_b", "cross_unit")]
    put("abl_hold_loss_min", min(hold), "{:.2f}", "ablation last sweep held")
    put("abl_hold_loss_max", max(hold), "{:.2f}", "ablation last sweep held")

    # --- latency ----------------------------------------------------------------------------------------
    lat = [json.loads(f.read_text())["methods"]["det-assoc"]["latency_ms_per_frame"] for tag in ("_D1_trackA_det_K20", "_D2_trackB_det_K20", "_D3_crossunit_det_K20")
           for f in R.glob(f"stream_*{tag}_*.json")]
    put("lat_det_min", min(lat), "{:.1f}", "D1-D3 det-assoc ms/frame (GPU, after detection)")
    put("lat_det_max", max(lat), "{:.1f}", "D1-D3")
    f = sorted(R.glob("stream_*_E5_cpu_bs1_anchored_v2_*.json"))[-1]
    put("lat_cpu_bs1_calib", json.loads(f.read_text())["methods"]["calib"]["latency_ms_per_frame"], "{:.1f}", "E5 CPU bs1 v2")

    (R / "manuscript_numbers.json").write_text(json.dumps(OUT, indent=1))
    for k, v in OUT.items():
        print(f"{k:32s} {v['text']:>8s}   {v['source']}")


if __name__ == "__main__":
    main()
