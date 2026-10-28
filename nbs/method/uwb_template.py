"""UWB (Novelda X4M200, node_1) pulse template from the data: the direct Tx->Rx coupling at the start of every
frame is an isolated copy of the transmitted pulse as the receiver sees it, so its temporal mean is the range kernel
the UWB head should use instead of a datasheet Gaussian (tracker T16, plan §4c).

Facts (2026-10-28): frames are RF-sampled at fs = 23.328 GS/s (bin = c / 2 fs = 6.43 mm; 1535 bins = 9.87 m),
spectral peak ~7.2 GHz, 17.06 Hz frame rate, ~1576 frames per recording, static-dominated (residual -31 dB).

Outputs ($NBS_DATA_OUT/uwb_template_S1/): template.npz (t_rf (K,) RF template centred on the coupling peak,
t_env (K,) its analytic envelope, fc_hz, bw_hz (-10 dB), fs_hz, bin_m, peak_bin, per-recording variants) and
viz/render_vs_gt/uwb_template.png.   Run on the node:  cd nbs && python -m method.uwb_template
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S

FS_HZ = 23.328e9
C_LIGHT = 299792458.0
BIN_M = C_LIGHT / (2 * FS_HZ)
HALF = 40          # template half-width in bins (+-26 cm) around the coupling peak


def analytic(x):
    X = np.fft.fft(x); n = len(x); h = np.zeros(n); h[0] = 1; h[1:(n + 1) // 2] = 2
    if n % 2 == 0: h[n // 2] = 1
    return np.fft.ifft(X * h)


def spectrum_stats(x, fs=FS_HZ):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))**2; f = np.fft.rfftfreq(len(x), 1 / fs)
    fc = float(f[np.argmax(X)]); above = f[X > X.max() * 0.1]
    return fc, float(above.max() - above.min()), f, X


def uwb_path_for(rec: str) -> str:
    """UWB file of a recording given ANY of its node_1 stamps (the vayyar file's stamp can differ by ~2 s)."""
    df = IO.load_index()
    m = df["node_1_uwb_data_path"].astype(str).str.contains(rec) | df["node_1_vayyar_data_path"].astype(str).str.contains(rec) \
        | df["node_1_depthCamera_data_path"].astype(str).str.contains(rec)
    if not m.any():
        raise KeyError(rec)
    return IO.resolve(df[m].iloc[0]["node_1_uwb_data_path"])


def template_from_recording(rec: str):
    frames, t = IO.load_uwb(uwb_path_for(rec))
    mean = frames.mean(0)
    env = np.abs(analytic(mean))
    pk = int(np.argmax(env[:120]))                                   # the coupling: first strong peak
    lo, hi = pk - HALF, pk + HALF                                     # may start before bin 0: zero-pad on the left
    pad = max(-lo, 0); lo = max(lo, 0)
    seg = np.concatenate([np.zeros(pad), mean[lo:hi]]); seg_env = np.concatenate([np.zeros(pad), env[lo:hi]])
    fc, bw, f, X = spectrum_stats(seg)
    # pulse width: FWHM of the envelope, and the -10 dB width
    e = seg_env / seg_env.max(); i = int(np.argmax(e))
    def width(th):
        l = i; r = i
        while l > 0 and e[l] > th: l -= 1
        while r < len(e) - 1 and e[r] > th: r += 1
        return (r - l) * BIN_M
    # how stable is the coupling across frames? (template SNR)
    seg_frames = frames[:, lo:hi]; resid = seg_frames - seg[pad:]
    snr_db = float(10 * np.log10((seg**2).mean() / (resid**2).mean()))
    return dict(rec=rec, n_frames=int(len(frames)), rate_hz=float(1 / np.median(np.diff(t))), peak_bin=pk, lo=lo, hi=hi,
                t_rf=seg, t_env=seg_env, fc_hz=fc, bw_10db_hz=bw, fwhm_m=width(0.5), w10db_m=width(10 ** (-0.5)),
                template_snr_db=snr_db, mean_frame=mean, env_frame=env, spec_f=f, spec=X)


def main():
    out = os.path.join(C.NBS_DATA_OUT, "uwb_template_S1"); os.makedirs(out, exist_ok=True)
    recs = [S.REC_SIT, S.REC_WALK]
    try:
        from method.sfcw_target import pick_scene1_recordings
        recs = [os.path.basename(str(r))[:14] for r in pick_scene1_recordings(5)]   # it returns vayyar paths; keep the stamp
    except Exception:
        pass
    res = [template_from_recording(r) for r in recs]
    # consensus template: align on the peak bin, average the RF (they are phase-locked copies if the sampler is)
    K = 2 * HALF; T = np.zeros(K); n = 0
    for r in res:
        seg = r["t_rf"]
        if len(seg) == K: T += seg; n += 1
    T /= max(n, 1); E = np.abs(analytic(T)); fc, bw, f, X = spectrum_stats(T)
    cross = np.corrcoef(np.stack([r["t_rf"] for r in res if len(r["t_rf"]) == K]))
    summary = dict(fs_hz=FS_HZ, bin_m=BIN_M, n_bins=1535, max_range_m=1535 * BIN_M, half_width_bins=HALF,
                   fc_hz=fc, bw_10db_hz=bw, fwhm_m=float(res[0]["fwhm_m"]), w10db_m=float(res[0]["w10db_m"]),
                   template_cross_recording_corr_min=float(cross[np.triu_indices(len(cross), 1)].min()),
                   per_recording={r["rec"]: {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()
                                             if k in ("n_frames", "rate_hz", "peak_bin", "fc_hz", "bw_10db_hz", "fwhm_m", "w10db_m", "template_snr_db")} for r in res},
                   note="RF template (carrier included) centred on the coupling peak; the head's range kernel = t_rf shifted to each path delay; "
                        "t_env for an incoherent comparison. The coupling peak sits at ~bin %d = %.2f m: the on-board delay to subtract from ranges." % (res[0]["peak_bin"], res[0]["peak_bin"] * BIN_M))
    np.savez(os.path.join(out, "template.npz"), t_rf=T, t_env=E, per_recording_peak_bin=float(np.median([r["peak_bin"] for r in res])),
             **{k: v for k, v in summary.items() if not isinstance(v, (dict, str))})
    json.dump(summary, open(os.path.join(out, "template.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_recording"}, indent=1))
    for r in res: print(r["rec"], {k: round(float(r[k]), 4) if isinstance(r[k], (float, np.floating)) else r[k] for k in ("n_frames", "peak_bin", "fc_hz", "bw_10db_hz", "fwhm_m", "template_snr_db")})
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    viz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", "render_vs_gt"); os.makedirs(viz, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    r0 = res[0]; x = np.arange(1535) * BIN_M
    ax[0, 0].plot(x, r0["mean_frame"], lw=0.6, label="temporal-mean frame (RF)"); ax[0, 0].plot(x, r0["env_frame"], lw=0.8, label="envelope")
    ax[0, 0].set_xlim(0, 2.0); ax[0, 0].set_xlabel("range [m] (bin 6.43 mm)"); ax[0, 0].set_title(f"{r0['rec']}: start of the frame; coupling peak at bin {r0['peak_bin']} = {r0['peak_bin']*BIN_M:.2f} m"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    xx = (np.arange(K) - HALF) * BIN_M * 100
    for r in res: ax[0, 1].plot(xx, r["t_rf"] / np.abs(r["t_rf"]).max(), lw=0.8, alpha=0.6, label=r["rec"][:8] + "…")
    ax[0, 1].plot(xx, T / np.abs(T).max(), "k", lw=1.5, label="consensus"); ax[0, 1].plot(xx, E / np.abs(T).max(), "r--", lw=1, label="envelope")
    ax[0, 1].set_xlabel("offset from coupling peak [cm]"); ax[0, 1].set_title(f"pulse template (RF): FWHM {summary['fwhm_m']*100:.1f} cm, -10 dB width {summary['w10db_m']*100:.1f} cm"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)
    ax[1, 0].semilogy(f / 1e9, X / X.max()); ax[1, 0].axvline(fc / 1e9, color="r", ls="--", label=f"fc {fc/1e9:.2f} GHz"); ax[1, 0].set_xlim(0, 11.6)
    ax[1, 0].set_xlabel("frequency [GHz] (fs = 23.328 GS/s)"); ax[1, 0].set_title(f"template spectrum: -10 dB bandwidth {bw/1e9:.2f} GHz"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    ax[1, 1].axis("off")
    ax[1, 1].text(0, 1, "per recording\n" + "\n".join(f"{r['rec']}: frames {r['n_frames']}, peak bin {r['peak_bin']}, fc {r['fc_hz']/1e9:.2f} GHz, BW {r['bw_10db_hz']/1e9:.2f} GHz, FWHM {r['fwhm_m']*100:.1f} cm, template SNR {r['template_snr_db']:.1f} dB" for r in res)
                  + f"\n\ncross-recording corr of the RF template: min {summary['template_cross_recording_corr_min']:.3f}\n"
                  "Use: UWB head range kernel = this RF template at each path delay (replaces the datasheet Gaussian);\n"
                  "ranges are offset by the coupling peak position (on-board delay).", va="top", family="monospace", fontsize=9)
    fig.suptitle("UWB X4M200 (node_1): data-derived pulse template from the direct-coupling peak", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(viz, "uwb_template.png"), dpi=150)
    with open(os.path.join(viz, "captions.txt"), "a") as fh:
        fh.write(f"uwb_template.png: UWB pulse template from the coupling peak of the temporal-mean frame (RF, fs 23.328 GS/s): fc {fc/1e9:.2f} GHz, -10 dB BW {bw/1e9:.2f} GHz, FWHM {summary['fwhm_m']*100:.1f} cm, peak at bin {r0['peak_bin']} ({r0['peak_bin']*BIN_M:.2f} m on-board delay); cross-recording corr min {summary['template_cross_recording_corr_min']:.3f}. Replaces the datasheet Gaussian in the UWB head (T16).\n")


if __name__ == "__main__":
    main()
