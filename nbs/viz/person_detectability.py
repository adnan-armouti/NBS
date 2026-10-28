"""Figure for T9: measured vs predicted residual energy maps (one recording per band), z-score distributions over
recordings and the repetition-rate check.  cd nbs && python -m viz.person_detectability"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from preprocessing import octonet_config as C

DATA_OUT = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
IN = os.path.join(DATA_OUT, "detectability_S1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "viz", "render_vs_gt")
AXIS = {"sfcw": ("range (m)", 0.0375), "uwb": ("range (m)", 5 * 9.9 / 1535), "wifi": ("receiver x CIR tap (7.5 m)", 1.0)}


def main():
    S = json.load(open(os.path.join(IN, "summary.json"))); recs = S["recordings"]; bands = ("sfcw", "uwb", "wifi")
    fig, axs = plt.subplots(3, 4, figsize=(24, 15), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1, 1]))
    for i, band in enumerate(bands):
        # best-z recording for the maps
        cand = [(r, e["bands"][band]["scores"]["both"]["z"]) for r, e in recs.items() if band in e["bands"] and "scores" in e["bands"][band]]
        if not cand: continue
        rec = max(cand, key=lambda x: x[1])[0]; z = np.load(os.path.join(IN, f"{rec}_{band}.npz"))
        Em, Ep = z["E_meas"], z["E_pred_both"]; act = z["active"]
        lab, step = AXIS[band]
        for j, (E, title) in enumerate(((Em, "measured residual |.| per bin"), (Ep, "predicted residual (direct + occlusion)"))):
            ax = axs[i, j]; E = E / (E.max() + 1e-30)
            ext = [0, E.shape[1] * step, 0, len(E)]
            im = ax.imshow(E, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
            if band != "wifi": ax.set_xlim(0, min(E.shape[1] * step, 9.9 if band == "uwb" else 3.75))
            ax.set_xlabel(lab); ax.set_ylabel("selected frame"); ax.set_title(f"{band.upper()} {rec} {recs[rec]['activity']}: {title}", fontsize=10)
            for b in act[:40]: ax.axvline(b * step, color="c", alpha=0.15, lw=0.8)
        # z distribution
        ax = axs[i, 2]
        for k, v in enumerate(("direct", "occlusion", "both")):
            zs = [e["bands"][band]["scores"][v]["z"] for e in recs.values() if band in e["bands"] and "scores" in e["bands"][band]]
            names = [e["activity"] for e in recs.values() if band in e["bands"] and "scores" in e["bands"][band]]
            ax.bar(np.arange(len(zs)) + 0.27 * (k - 1), zs, width=0.27, label=v)
        ax.axhline(3, color="r", ls="--", lw=1, label="z = 3"); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, fontsize=8)
        tb = S["decision_table"][f"{band}/both"]; ax.set_title(f"{band.upper()}: z-score per recording (null = frame permutation)\nboth: median z {tb['median_z']:.1f}, frac z>3 {tb['frac_z_gt_3']:.2f} -> {S['verdict'][band].split(' ')[0]}", fontsize=10)
        ax.set_ylabel("z"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        # repetition check
        ax = axs[i, 3]
        pts = [(e["activity"], e["f0_hz"], e["f0_prominence"], e["bands"][band]["repetition"]["z"]) for e in recs.values()
               if band in e["bands"] and e["bands"][band].get("repetition")]
        if pts:
            per = [p for p in pts if p[2] > 3]; non = [p for p in pts if p[2] <= 3]
            if per: ax.scatter([p[1] for p in per], [p[3] for p in per], c="tab:blue", label="periodic activity (MoCap prominence > 3)")
            if non: ax.scatter([p[1] for p in non], [p[3] for p in non], c="0.6", label="aperiodic")
            for p in pts: ax.annotate(p[0], (p[1], p[3]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(3, color="r", ls="--", lw=1); ax.set_xlabel("MoCap repetition rate f0 (Hz)"); ax.set_ylabel("z of residual power at f0")
        tr = S["decision_table"][f"{band}/repetition"]
        ax.set_title(f"{band.upper()}: model-free check (residual energy spectrum at f0)\nperiodic n={tr['n_periodic']}, median z {tr['median_z'] if tr['median_z'] is None else round(tr['median_z'],1)}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("T9 person detectability: physics-predicted residual (SMPL from MoCap, direct return + occlusion of the calibrated scaffold) vs measured residual, "
                 f"{len(recs)} Scene-1 recordings, {S['config']['frames']} frames each, {S['config']['n_perm']} permutations", fontsize=12)
    fig.tight_layout(); os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "person_detectability.png"), dpi=140); plt.close(fig)
    cap = ("person_detectability.png: per band (rows) the measured vs predicted residual energy maps of the best-z recording, the per-recording z-scores "
           "(direct / occlusion / both; null = permuting the frame order of the prediction, 1000 draws) and the model-free repetition-rate check. Verdicts: "
           + "; ".join(f"{b}: {S['verdict'][b]}" for b in bands) + ". Caveats: UWB/radar pose = node_1 marker + camera yaw (assumption); Wi-Fi Tx/Rx schematic (+-0.5 m); "
           "uniform concrete material; the person mesh is SMPL fitted to the OptiTrack skeleton (2-3 cm joint RMSE) at the Vayyar frame times, UWB/Wi-Fi frames nearest in time.\n")
    with open(os.path.join(OUT, "captions.txt"), "a") as f: f.write(cap)
    print("wrote", os.path.join(OUT, "person_detectability.png"))


if __name__ == "__main__":
    sys.exit(main())
