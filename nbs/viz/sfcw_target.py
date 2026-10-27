"""viz/render_vs_gt/sfcw_static_target.png: the SFCW static target and its reliability.
Run from nbs/:  python -m viz.sfcw_target
"""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import octonet_scene as S
from method import sfcw_target as ST

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "viz", "render_vs_gt")
os.makedirs(OUT, exist_ok=True)

def db(x, m=None):
    x = np.abs(np.asarray(x)); ref = x[m].max() if m is not None else x.max(); return 20 * np.log10(x / (ref + 1e-30) + 1e-30)

def main():
    V, _, p = ST.load_recording(S.REC_SIT); tg = ST.sfcw_static_target(V)
    recs = ST.pick_scene1_recordings(5); pool = ST.pool_recordings(recs)
    r = tg["range_m"].numpy(); m = tg["mask"].numpy(); g = np.where(m, 1.0, np.nan)
    rp_a, rp_b = tg["rp_half"]; rc1, rc2 = tg["rp_contig"]
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    # (a) halves + difference
    ax = axs[0, 0]
    ref = (tg["rp_mean"].abs().mean(0).numpy() * g)[m].max()   # dB rel. the guarded-band max
    for lab, x, st in (("mean channel (all frames)", tg["rp_mean"], "k-"), ("interleaved half A", rp_a, "C0-"), ("interleaved half B", rp_b, "C1--"),
                       ("|A - B| (fast noise)", rp_a - rp_b, "C3-"), ("|first half - second half| (drift)", rc1 - rc2, "C2:")):
        y = 20 * np.log10(x.abs().mean(0).numpy() / ref + 1e-30); ax.plot(r, y * g, st, label=lab, lw=1.5)
    ax.axvline(float(np.linalg.norm(S.NODE_POS_S1[1][:2])), color="r", ls="--", lw=1, label="subject spot 2.92 m")
    ax.set_ylim(-60, 2); ax.set_xlabel("range (m)"); ax.set_ylabel("|range profile| mean over 400 pairs (dB rel. guarded-band max)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"(a) {os.path.basename(p).split('_node')[0]} sit: T={tg['T']} frames; guard bins 0-5 and last 6 (coupling / IFFT wrap)", fontsize=10)
    # (b) reliability + snr
    ax = axs[0, 1]
    ax.plot(r, tg["rel"].numpy() * g, "C0-", label="rel: split-half corr, interleaved (fast noise)")
    ax.plot(r, tg["rel_contig"].numpy() * g, "C2-", label="rel_contig: first vs second half (drift-sensitive)")
    ax.plot(r, tg["coh"].numpy() * g, "C1:", label="complex coherence, interleaved")
    ax.set_ylim(0, 1.05); ax.set_ylabel("reliability (corr across 400 pairs)"); ax.set_xlabel("range (m)"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left")
    ax2 = ax.twinx(); ax2.plot(r, tg["snr_db"].numpy() * g, "k--", label="SNR of the mean channel (dB)"); ax2.set_ylabel("SNR (dB)"); ax2.legend(fontsize=8, loc="lower right")
    s = tg["summary"]; ax.set_title(f"(b) per-bin reliability: median rel {s['rel_median']:.3f}, rel_contig {s['rel_contig_median']:.3f}, SNR {s['snr_db_median']:.1f} dB [{s['snr_db_min']:.1f}, {s['snr_db_max']:.1f}]", fontsize=10)
    # (c) pooled vs single + xcorr
    ax = axs[1, 0]
    ax.plot(r, db(tg["rp_mean"].abs().mean(0).numpy(), m) * g, "k-", label="single recording (sit)")
    ax.plot(r, db(pool["rp_mean"].abs().mean(0).numpy(), m) * g, "C3-", label=f"pooled {pool['summary']['n_rec']} recordings (T={pool['T']})")
    for tg_i, nm in zip(pool["per_rec"], pool["names"]):
        ax.plot(r, db(tg_i["rp_mean"].abs().mean(0).numpy(), m) * g, lw=0.6, alpha=0.5, label=nm.split("_activity_")[1].split("_trial")[0])
    ax.set_ylim(-20, 2); ax.set_xlabel("range (m)"); ax.set_ylabel("dB rel. guarded-band max"); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    ax2 = ax.twinx(); ax2.plot(r, pool["xcorr"].numpy() * g, "C0--", lw=2, label="cross-recording corr of |rp_mean| per bin"); ax2.set_ylim(0, 1.05); ax2.set_ylabel("cross-recording corr"); ax2.legend(fontsize=8, loc="lower right")
    ax.set_title(f"(c) pooled target: cross-recording corr median {pool['summary']['xcorr_median']:.3f} (min {pool['summary']['xcorr_min']:.3f}) in the guarded band", fontsize=10)
    # (d) residual autocorrelation
    ax = axs[1, 1]; rho = tg["rho"]; lag_s = np.arange(len(rho)) / 3.1
    ax.plot(lag_s, rho, "k-"); ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("lag (s) at 3.1 Hz"); ax.set_ylabel("autocorr of the residual V - H_mean (guarded band)"); ax.grid(alpha=.3)
    ax.set_title(f"(d) residual is drift-dominated: n_eff = {tg['n_eff']:.1f} of T = {tg['T']} frames (Bartlett)", fontsize=10)
    fig.suptitle("SFCW static target = clutter-guarded temporal-mean channel, with per-bin reliability (Vayyar node_1, Scene 1)", fontsize=13)
    fig.tight_layout(); out = os.path.join(OUT, "sfcw_static_target.png"); fig.savefig(out, dpi=150); plt.close(fig)
    with open(os.path.join(OUT, "captions.txt"), "a") as f:
        f.write(f"\nsfcw_static_target.png: the SFCW objective is the temporal-mean channel with bins 0-5 and the last 6 guarded. (a) interleaved split halves agree to {s['rel_median']:.3f} median corr across the 400 pairs (fast noise); first-vs-second half {s['rel_contig_median']:.3f} (drift); the mean-channel SNR is {s['snr_db_median']:.1f} dB median in the guarded band. (c) pooling {pool['summary']['n_rec']} Scene-1 recordings of different activities: per-bin cross-recording corr of |rp_mean| median {pool['summary']['xcorr_median']:.3f} (min {pool['summary']['xcorr_min']:.3f}) -> the static target is the same across recordings and can be pooled (T={pool['T']}). (d) the residual V-H_mean is drift-dominated: n_eff={tg['n_eff']:.1f} independent frames of T={tg['T']}. Loss: method.sfcw_target.sfcw_static_loss = reliability-weighted masked L1 on median-normalised log|rp| (or log|RA|).\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
