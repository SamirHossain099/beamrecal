"""Paper figures (600 dpi PNG + vector PDF, one figure per file) with their data cached to JSON.

  fig1_pipeline      schematic of the deployable pipeline (no data)
  fig2_geometry      (a) best beam vs bearing, 11 scenarios; (b) the same against the fitted relative angle
                     -> results/fig2_geometry.json: per-scenario within-3 of the bearing prior without / with the
                        base-station origin correction, fitted parameters
  fig3_camera        zero-label transfer matrix of the camera map (annotated boxes), within-3 and top-1
                     -> results/fig3_camera_transfer.json
  fig4_overhead      Track B overhead vs received-power loss (from results/overhead_trackB.json)
  detector_match     not a figure: detections vs annotated UE box -> results/detector_match.json

    python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "figures"
R = ROOT / "results"
FIG.mkdir(exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     # STIX math is Times-like, so axis labels with math match the Times text around them
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "savefig.bbox": "tight"})
GEO_SCEN = ["scenario32", "scenario33", "scenario34", "scenario31", "scenario35", "scenario1", "scenario2", "scenario5",
            "scenario6", "scenario8", "scenario9"]
CAM_SCEN = ["scenario1", "scenario2", "scenario5", "scenario7", "scenario6", "scenario8", "scenario9"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600)
    plt.close(fig)
    print("wrote", f"figures/{name}.png/.pdf")


def short(s):
    return s.replace("scenario", "")


# ------------------------------------------------------------------ figure 1: pipeline schematic
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.axis("off")
    W, H = 0.19, 0.30
    nl = chr(10)
    X = [0.0, 0.225, 0.45, 0.675, 0.90]         # box columns; arrows are derived from these
    boxes = {"cam": (X[0], 0.58, "Camera frame"), "gps": (X[0], 0.12, "Reported vehicle" + nl + "position (GPS)"),
             "det": (X[1], 0.58, "Vehicle detector"), "pri": (X[1], 0.12, "Position prior" + nl + "(bearing, boresight)"),
             "sel": (X[2], 0.35, "Select the served" + nl + "vehicle's box"),
             "map": (X[3], 0.35, "Camera map" + nl + "(pinhole model)"),
             "out": (X[4], 0.35, "Served beam" + nl + "(top-$k$ measured)")}
    for x, y, t in boxes.values():
        ax.add_patch(plt.Rectangle((x, y), W, H, fill=False, lw=0.7))
        ax.text(x + W / 2, y + H / 2, t, ha="center", va="center", fontsize=7.5)

    def arrow(a, b, **kw):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=0.7, **kw))
    arrow((X[0] + W, 0.73), (X[1], 0.73))
    arrow((X[0] + W, 0.27), (X[1], 0.27))
    arrow((X[1] + W, 0.73), (X[2], 0.60))
    arrow((X[1] + W, 0.27), (X[2], 0.40))
    arrow((X[2] + W, 0.50), (X[3], 0.50))
    arrow((X[3] + W, 0.50), (X[4], 0.50))
    # feedback: sweeps refit the prior and the map
    ax.plot([X[4] + W / 2, X[4] + W / 2, X[1] + W / 2], [0.35, 0.03, 0.03], color="k", lw=0.6, ls="--")
    arrow((X[1] + W / 2, 0.03), (X[1] + W / 2, 0.12), ls="--")
    arrow((X[3] + W / 2, 0.03), (X[3] + W / 2, 0.35), ls="--")
    ax.text((X[1] + X[4] + W) / 2, -0.075, "periodic full sweep: refit prior and map", ha="center", fontsize=7.5)
    ax.set_xlim(-0.01, X[4] + W + 0.01)
    ax.set_ylim(-0.12, 0.92)
    save(fig, "fig1_pipeline")


# ------------------------------------------------------------------ figure 2: geometry of the shift
def fig2():
    import torch
    from tta_beam.data import load_scenario
    from tta_beam.models import PhysicsPrior
    data, colors = {}, plt.cm.tab20(np.linspace(0, 1, len(GEO_SCEN)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"wspace": 0.38})
    for i, s in enumerate(GEO_SCEN):
        d = load_scenario(s)
        g, y = torch.as_tensor(d["gps"]), torch.as_tensor(d["y"])
        bear = np.degrees(np.arctan2(d["gps"][:, 0], d["gps"][:, 1]))
        if s == "scenario34":
            bear = np.where(bear < 0, bear + 360, bear)
        p0 = PhysicsPrior().fit(g, y, fit_origin=False)
        w3_no = float(((p0.mean_beam(g).round().clamp(0, 63) - y).abs() <= 3).float().mean())
        p1 = PhysicsPrior().fit(g, y, fit_origin=True)
        w3_yes = float(((p1.mean_beam(g).round().clamp(0, 63) - y).abs() <= 3).float().mean())
        rel = g - p1.origin
        u = torch.sin(torch.atan2(rel[:, 0], rel[:, 1]) - p1.theta).numpy()
        data[s] = {"n": int(len(y)), "within3_no_origin": w3_no, "within3_with_origin": w3_yes,
                   "K": float(p1.K), "theta_deg": float(np.degrees(p1.theta)), "origin_m": [float(v) * 30 for v in p1.origin]}
        idx = np.random.default_rng(0).choice(len(y), min(800, len(y)), replace=False)
        a1.scatter(bear[idx], d["y"][idx], s=0.6, color=colors[i], label=short(s), rasterized=True)
        a2.scatter(u[idx], (d["y"][idx] - float(p1.c0)) / float(p1.K), s=0.6, color=colors[i], rasterized=True)
    a1.set_xlabel("bearing from base station to vehicle (deg)")
    a1.set_ylabel("best beam index")
    a2.set_xlabel(r"$\sin(\beta(p-o)-\theta)$ after per-scenario fit")
    a2.set_ylabel(r"$(\mathrm{beam}-c_0)/K$")
    a2.plot([-1, 1], [-1, 1], color="0.3", lw=0.6)
    for ax, lab in ((a1, "(a)"), (a2, "(b)")):
        ax.text(0.02, 0.95, lab, transform=ax.transAxes, fontsize=9, va="top")
    # one row under both panels, clear of the data
    fig.legend(*a1.get_legend_handles_labels(), title="scenario", title_fontsize=7.5, fontsize=7.5, markerscale=6,
               ncol=len(GEO_SCEN), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.1, columnspacing=1.0)
    (R / "fig2_geometry.json").write_text(json.dumps(data, indent=1))
    save(fig, "fig2_geometry")


# ------------------------------------------------------------------ figure 3: camera map transfer
def fig3():
    from tta_beam.data import load_scenario
    from tta_beam.geo import CamMap, load_cam_x
    D = {}
    for s in CAM_SCEN:
        x, y = load_cam_x(s), load_scenario(s)["y"]
        ok = np.isfinite(x)
        D[s] = (x[ok], y[ok])
    maps = {s: CamMap().fit_source(*D[s]) for s in CAM_SCEN}
    W3 = np.zeros((len(CAM_SCEN), len(CAM_SCEN)))
    T1 = np.zeros_like(W3)
    for i, a in enumerate(CAM_SCEN):
        for j, b in enumerate(CAM_SCEN):
            pr = np.clip(np.round(maps[a].mean(D[b][0])), 0, 63)
            W3[i, j] = (np.abs(pr - D[b][1]) <= 3).mean()
            T1[i, j] = (pr == D[b][1]).mean()
    (R / "fig3_camera_transfer.json").write_text(json.dumps(
        {"scenarios": CAM_SCEN, "within3": W3.tolist(), "top1": T1.tolist(),
         "note": "fit on row, applied to column; annotated UE boxes (class 0); diagonal = in-sample fit"}, indent=1))
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(W3, vmin=0, vmax=1, cmap="Greys")
    for i in range(len(CAM_SCEN)):
        for j in range(len(CAM_SCEN)):
            ax.text(j, i, f"{W3[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if W3[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(CAM_SCEN)), [short(s) for s in CAM_SCEN])
    ax.set_yticks(range(len(CAM_SCEN)), [short(s) for s in CAM_SCEN])
    ax.set_xlabel("applied to scenario")
    ax.set_ylabel("fitted on scenario")
    ax.axhline(3.5, color="0.5", lw=0.6)
    ax.axvline(3.5, color="0.5", lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction within 3 beams")
    save(fig, "fig3_camera")


# ------------------------------------------------------------------ figure 4: overhead
NAMES = {"sense-geo": "camera map, zero target labels", "sense-geo+calib": "camera map refitted on sweeps",
         "offset-ma": "camera map + offset tracking (ViBe-style)", "cam-assoc": "camera map, GPS association",
         "calib+gate+norm": "physics-anchored network", "calib+supft": "physics-anchored network + fine-tuning",
         "supft": "supervised online fine-tuning", "tent": "Tent", "source": "source model", "sweep-hold": "sweep and hold"}


def fig4():
    rows = json.loads((R / "overhead_trackB.json").read_text())
    # full width with the legend at the side: a one-column version needs a legend taller than the plot
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    styles = {"sense-geo": "k-", "sense-geo+calib": "k--", "offset-ma": "k:", "cam-assoc": "k-.",
              "calib+gate+norm": "C0-", "calib+supft": "C0--", "supft": "C1-", "tent": "C2-", "source": "C3-", "sweep-hold": "C4-"}
    for m, st in styles.items():
        rr = sorted([r for r in rows if r["method"] == m], key=lambda r: r["overhead_beams_per_frame"])
        front, best = [], np.inf
        for r in rr:
            if r["loss_db"] < best:
                front.append(r)
                best = r["loss_db"]
        if front:
            ax.plot([r["overhead_beams_per_frame"] for r in front], [r["loss_db"] for r in front], st, lw=0.9,
                    marker="o", ms=2, label=NAMES[m])
    ax.set_xscale("symlog", linthresh=0.5, linscale=0.5)
    ax.set_xlim(-0.05, 80)
    ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20, 64], ["0", "0.5", "1", "2", "5", "10", "20", "64"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("beam measurements per frame")
    ax.set_ylabel("received-power loss (dB)")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    save(fig, "fig4_overhead")


# ------------------------------------------------------------------ figure 5: pinhole map vs MLP on x
def fig5():
    rows = json.loads((R / "xmlp_curve.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"wspace": 0.28})
    styles = {"sense-geo": ("k", "--", "pinhole map, no target labels"),
              "sense-geo+calib": ("k", "-", "pinhole map refitted on sweeps"),
              "xmlp": ("C0", "--", "MLP on the same input, no target labels"),
              "xmlp+ft": ("C0", "-", "MLP fine-tuned on the same sweeps")}
    for ax, (stream, title) in zip(axes, (("track_b", "(a) Track B, same unit"),
                                          ("cross_unit", "(b) cross-unit"))):
        for m, (c, ls, lab) in styles.items():
            rr = sorted([r for r in rows if r["stream"] == stream and r["method"] == m], key=lambda r: r["K"])
            ax.errorbar([r["K"] for r in rr], [r["ploss_db"] for r in rr], yerr=[r["ploss_db_std"] for r in rr],
                        color=c, ls=ls, lw=0.9, marker="o", ms=2.5, capsize=1.5, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
        ax.minorticks_off()
        ax.set_xlabel("frames between full sweeps")
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_ylabel("received-power loss (dB)")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig5_mlp_control")


# ------------------------------------------------------------------ detector vs annotation (numbers only)
def detector_match():
    from tta_beam.geo import load_cam_x, load_det_boxes
    out = {}
    for s in ("scenario1", "scenario2", "scenario5", "scenario6", "scenario7", "scenario8"):
        cx, det = load_cam_x(s), load_det_boxes(s)
        ok = np.isfinite(cx)
        err = np.nanmin(np.abs(det[ok] - cx[ok, None]), axis=1)
        out[s] = {"frames": int(ok.sum()), "median_dx": float(np.nanmedian(err)), "within_0.02": float(np.nanmean(err < 0.02)),
                  "detections_per_frame": float(np.isfinite(det).sum(1).mean())}
    for s in ("scenario31", "scenario32", "scenario33", "scenario34", "scenario35", "scenario9"):
        det = load_det_boxes(s)
        out[s] = {"detections_per_frame": float(np.isfinite(det).sum(1).mean())}
    (R / "detector_match.json").write_text(json.dumps(out, indent=1))
    print("wrote results/detector_match.json")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "detector_match"]
    for w in which:
        globals()[w]()
