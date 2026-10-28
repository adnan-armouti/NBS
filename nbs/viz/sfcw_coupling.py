"""Figure for tracker T4: viz/render_vs_gt/sfcw_coupling.png from method.sfcw_coupling's study outputs.
Run after `python -m method.sfcw_coupling`:  cd nbs && python -m viz.sfcw_coupling"""
from __future__ import annotations
import json, os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import octonet_config as C, vayyar_radar_config as VR
from mm3DPS.data import ra_utils as RA
from method import sfcw_coupling as SC

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ = os.path.join(HERE, "..", "viz", "render_vs_gt")
DATA = os.path.join(C.NBS_DATA_OUT, "sfcw_target_S1")


def db(x, ref=None, floor=-70.0):
    x = np.asarray(x, float); ref = np.nanmax(x) if ref is None else ref
    return np.maximum(20 * np.log10(np.maximum(x, 1e-300) / ref), floor)


def main():
    st = torch.load(os.path.join(DATA, "coupling_study.pt"), weights_only=False)
    rep = json.load(open(os.path.join(DATA, "coupling_study.json")))
    r_axis = np.arange(VR.N_FREQ) * VR.RANGE_RES_M
    fig, axs = plt.subplots(2, 3, figsize=(21, 11))
    # (a) one pair, one frame: raw vs tones vs cleaned, Hann profile dB, and rectangular-window profile
    ax = axs[0, 0]
    Hc, Hraw, tone = [t.to(torch.complex128) for t in st["fit_sit_tone2"]["pair_example"]]
    for H, lab, ls in ((Hraw, "raw sweep", "-"), (tone, "fitted coupling (2 tones)", "--"), (Hc, "raw - coupling", "-")):
        ax.plot(r_axis, db(RA.sfcw_range_compress(H[None])[0].abs().numpy(), ref=RA.sfcw_range_compress(Hraw[None])[0].abs().max().item()), ls, label=lab + " (Hann)")
    ax.plot(r_axis, db(torch.fft.ifft(Hraw).abs().numpy(), ref=RA.sfcw_range_compress(Hraw[None])[0].abs().max().item()), ":", c="0.5", label="raw, rectangular window")
    ax.axvspan(0, 6 * VR.RANGE_RES_M, color="r", alpha=0.08, label="guard bins 0-5"); ax.axvspan(6 * VR.RANGE_RES_M, 26 * VR.RANGE_RES_M, color="orange", alpha=0.08, label="bins 6-25 (leakage zone)")
    ax.set_xlim(0, 1.6); ax.set_ylim(-70, 2); ax.set_xlabel("range (m)"); ax.set_ylabel("dB rel. raw peak"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    s1 = rep["sit_tone2"]
    ax.set_title(f"pair (tx0,rx0), frame 0: per-frame coupling fit\nfitted bin {s1["bin_median"]:.2f} = {s1["bin_median"]*3.75:.1f} cm (on-board path); energy left in bins 0-5 {s1['residual_db']:+.1f} dB", fontsize=10)
    # (b) pooled pair-mean profile for the four target variants
    ax = axs[0, 1]
    for name, c in (("baseline_hann", "k"), ("decoupled_hann", "tab:red"), ("baseline_bh4", "0.6"), ("decoupled_bh4", "tab:blue")):
        rp = st["targets"][name]["rp_mean"].abs().mean(0).numpy(); ax.plot(r_axis, db(rp, ref=np.max(rp[6:94])), c=c, label=name)
    ax.axvspan(6 * VR.RANGE_RES_M, 26 * VR.RANGE_RES_M, color="orange", alpha=0.08); ax.set_xlim(0, 3.7); ax.set_ylim(-30, 25)
    ax.set_xlabel("range (m)"); ax.set_ylabel("pair-mean |rp|, dB rel. guarded max"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    lk = {n: rep[f"target_{n}"]["leakage"]["bins6_25_mean_rel_peak_db"] for n in ("baseline_hann", "decoupled_hann", "baseline_bh4", "decoupled_bh4")}
    ax.set_title("pooled 5-recording target, pair-mean profile\nbins 6-25 mean rel. guarded peak: " + ", ".join(f"{k.replace('_',' ')} {v:+.1f}" for k, v in lk.items()) + " dB", fontsize=9)
    # (c) reliability: cross-recording corr per bin, before/after
    ax = axs[0, 2]
    for name, c in (("baseline_hann", "k"), ("decoupled_hann", "tab:red"), ("decoupled_bh4", "tab:blue")):
        ax.plot(r_axis, st["targets"][name]["xcorr"].numpy(), c=c, label=f"xcorr {name}")
        ax.plot(r_axis, st["targets"][name]["rel"].numpy(), c=c, ls=":", alpha=.6, label=f"rel (fast) {name}")
    ax.axvspan(6 * VR.RANGE_RES_M, 26 * VR.RANGE_RES_M, color="orange", alpha=0.08); ax.set_xlim(0, 3.7); ax.set_ylim(0, 1.02)
    ax.set_xlabel("range (m)"); ax.set_ylabel("per-bin reliability"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    xc = {n: rep[f"target_{n}"]["xcorr_bins6_25"] for n in ("baseline_hann", "decoupled_hann", "decoupled_bh4")}
    ax.set_title("target reliability per bin (cross-recording corr = the loss weight)\nbins 6-25 mean xcorr: " + ", ".join(f"{k} {v:.3f}" for k, v in xc.items()), fontsize=9)
    # (d) fitted rho vs range for the fits
    ax = axs[1, 0]
    edges = np.arange(0, 7.5, 0.25)
    for name, c in (("baseline_hann", "k"), ("decoupled_hann", "tab:red"), ("decoupled_bh4", "tab:blue"), ("raw+coupling_in_render_hann_guard3", "tab:green"), ("decoupled_hann_guard24", "tab:purple")):
        f = st["fits"][name]; r, rho = f["r_pt"], f["rho"]
        med = [np.median(rho[(r >= a) & (r < b)]) if np.any((r >= a) & (r < b)) else np.nan for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(0.5 * (edges[:-1] + edges[1:]), np.log10(med), c=c, marker="o", ms=3, label=name)
    ax.axvline(1.0, color="orange", ls="--"); ax.axvspan(3.71, 7.5, color="0.9"); ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("point range from the radar (m)"); ax.set_ylabel("median log10 rho"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("fitted reflectivity vs range (guard24 = bins < 0.9 m masked: points < 1 m then sit at the prior)", fontsize=10)
    # (e) per-bin residual after fit
    ax = axs[1, 1]
    for name, c in (("baseline_hann", "k"), ("decoupled_hann", "tab:red"), ("decoupled_bh4", "tab:blue"), ("raw+coupling_in_render_hann_guard3", "tab:green")):
        ax.plot(r_axis, st["fits"][name]["per_bin_after"], c=c, label=name)
    ax.set_xlim(0, 3.7); ax.set_xlabel("range (m)"); ax.set_ylabel("per-bin |log|rp| residual| after fit"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("per-bin residual after the fit (each on its own target)", fontsize=10)
    # (f) numbers
    ax = axs[1, 2]; ax.axis("off")
    lines = ["fit                                   pool loss        held (sit) loss   rho<1m  rho 1-3.5m  held resid bins 24-93"]
    for name in ("baseline_hann", "decoupled_hann", "decoupled_bh4", "raw+coupling_in_render_hann_guard3", "decoupled_hann_guard24"):
        f = rep[f"fit_{name}"]
        lines.append(f"{name:36s} {f['loss_before']['pool']:.3f}->{f['loss_after']['pool']:.3f}   {f['loss_before']['single']:.3f}->{f['loss_after']['single']:.3f}   {f['rho_median_lt1m']:.3f}   {f['rho_median_1to3p5m']:.3f}       {f['held_resid_bins24_93']:.3f}")
    lk = rep["sit_tone2_leakage_only"]
    lines += ["", f"coupling alone after Hann (fitted tones re-synthesised): bins 6-25 {lk['coupling_only_bins6_25_rel_peak_db']:+.1f} dB, bins 6-12 {lk['coupling_only_bins6_12_rel_peak_db']:+.1f} dB rel. scene peak;",
              f"measured bins 6-25 {lk['measured_bins6_25_rel_peak_db']:+.1f} dB -> the near bins are static NEAR-FIELD returns (cover/housing/table), not window leakage"]
    lines += ["", "windows (PSF): " + "; ".join(f"{w}: -3dB {v['width_3db_cm']:.1f} cm, null {v['first_null_bins']:.1f} bins, sidelobe {v['peak_sidelobe_db']:.0f} dB" for w, v in rep["windows"].items()),
              "", f"coupling tone (sit, 2 tones): bin {rep['sit_tone2']['bin_median']:.2f}, bin std over frames {rep['sit_tone2']['b_std_over_frames']:.3f}, |A| rel. std {rep['sit_tone2']['A_rel_std_over_frames']:.3f}",
              f"bins 6-25 rel. peak: raw {rep['sit_tone1']['before']['bins6_25_mean_rel_peak_db']:+.1f} dB -> 1 tone {rep['sit_tone1']['after']['bins6_25_mean_rel_peak_db']:+.1f} -> 2 tones {rep['sit_tone2']['after']['bins6_25_mean_rel_peak_db']:+.1f} dB"]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    fig.suptitle("SFCW direct-coupling removal (T4): per-frame tone fit in the sweep domain, window choice, coupling term in the render", fontsize=13)
    fig.tight_layout(); os.makedirs(VIZ, exist_ok=True); fig.savefig(os.path.join(VIZ, "sfcw_coupling.png"), dpi=150); plt.close(fig)
    cap = (f"sfcw_coupling.png: direct coupling fitted per frame & pair as 1-2 tones in the un-windowed sweep (bin {rep["sit_tone2"]["bin_median"]:.2f} = {rep["sit_tone2"]["bin_median"]*3.75:.1f} cm, the on-board path) and subtracted; "
           f"bins 6-25 level rel. guarded peak {rep['target_baseline_hann']['leakage']['bins6_25_mean_rel_peak_db']:+.1f} -> {rep['target_decoupled_hann']['leakage']['bins6_25_mean_rel_peak_db']:+.1f} dB (Hann), "
           f"{rep['target_decoupled_bh4']['leakage']['bins6_25_mean_rel_peak_db']:+.1f} dB (Blackman-Harris); cross-rec corr bins 6-25 {rep['target_baseline_hann']['xcorr_bins6_25']:.3f} -> {rep['target_decoupled_hann']['xcorr_bins6_25']:.3f}; "
           f"refit rho median <1 m: baseline {rep['fit_baseline_hann']['rho_median_lt1m']:.3f}, decoupled {rep['fit_decoupled_hann']['rho_median_lt1m']:.3f}, decoupled+BH4 {rep['fit_decoupled_bh4']['rho_median_lt1m']:.3f}, "
           f"coupling-in-render {rep['fit_raw+coupling_in_render_hann_guard3']['rho_median_lt1m']:.3f}; held (sit) loss after: {rep['fit_baseline_hann']['loss_after']['single']:.3f} / {rep['fit_decoupled_hann']['loss_after']['single']:.3f} / "
           f"{rep['fit_decoupled_bh4']['loss_after']['single']:.3f} / {rep['fit_raw+coupling_in_render_hann_guard3']['loss_after']['single']:.3f}.")
    with open(os.path.join(VIZ, "captions.txt"), "a") as f: f.write(cap + "\n")
    print("wrote", os.path.join(VIZ, "sfcw_coupling.png")); print(cap)


if __name__ == "__main__":
    main()
