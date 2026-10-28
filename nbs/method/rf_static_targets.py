"""Static targets for UWB and Wi-Fi (tracker T17), built like the SFCW one (method/sfcw_target.py): the per-recording
temporal mean, pooled over Scene-1 recordings, with a per-bin / per-subcarrier reliability = cross-recording
correlation, and the data's own answers to two questions the heads depend on:
  * UWB: is the RF frame phase-stable frame to frame (coherent mean usable) and what is the coupling guard?
  * WiFi: does the complex CSI survive temporal averaging (per-packet CFO/SFO phase would kill it), and which
    subcarriers are fixed HT40 nulls / pilots?  Also the antenna-relative phase conj(a0)*a1, which cancels a common
    per-packet phase.
Outputs $NBS_DATA_OUT/uwb_target_S1/target_pool5.npz, wifi_target_S1/target_pool5.npz (+ .json summaries) and
viz/render_vs_gt/rf_static_targets.png.    Run on the node: cd nbs && python -m method.rf_static_targets
"""
from __future__ import annotations
import json, os, pickle
import numpy as np
from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S
from method.uwb_template import uwb_path_for, analytic, FS_HZ, BIN_M

UWB_GUARD = 45            # bins (0.29 m): the direct pulse (peak bin 7, FWHM 20 bins) and its tail
WIFI_RX = (1, 2, 3, 4)


def recs5():
    from method.sfcw_target import pick_scene1_recordings
    return [os.path.basename(str(r))[:14] for r in pick_scene1_recordings(5)]


def row_for(rec):
    df = IO.load_index()
    m = df["node_1_uwb_data_path"].astype(str).str.contains(rec) | df["node_1_vayyar_data_path"].astype(str).str.contains(rec) \
        | df["node_1_depthCamera_data_path"].astype(str).str.contains(rec)
    return df[m].iloc[0]


def corr_rows(A):
    c = np.corrcoef(A); return c[np.triu_indices(len(A), 1)]


# ----------------------------------------------------------------------------- UWB
def uwb_static(rec):
    fr, t = IO.load_uwb(uwb_path_for(rec))
    mean = fr.mean(0); env = np.abs(analytic(mean))
    # frame-to-frame RF coherence per bin: |mean frame| / mean |analytic frame|  (1 = phase-stable)
    an = np.stack([analytic(f) for f in fr[::10]]); coh = np.abs(an.mean(0)) / np.abs(an).mean(0).clip(1e-12)
    resid = fr - mean
    return dict(rec=rec, n=len(fr), mean_rf=mean, env=env, coh=coh, resid_db=float(10*np.log10((resid**2).mean()/(mean**2).mean())))


def uwb_pool(recs):
    per = [uwb_static(r) for r in recs]
    E = np.stack([p["env"] for p in per]); R = np.stack([p["mean_rf"] for p in per])
    w = np.array([p["n"] for p in per], float); w /= w.sum()
    pooled_rf = (R * w[:, None]).sum(0); pooled_env = np.abs(analytic(pooled_rf))
    # reliability per bin: cross-recording correlation of the envelope in a sliding 9-bin window (a per-bin corr needs several samples)
    rel = np.zeros(E.shape[1]); half = 4
    for b in range(E.shape[1]):
        lo, hi = max(0, b-half), min(E.shape[1], b+half+1)
        rel[b] = np.mean(corr_rows(E[:, lo:hi])) if hi - lo > 2 else 0.0
    mask = np.ones(E.shape[1], bool); mask[:UWB_GUARD] = False
    coh = np.mean([p["coh"] for p in per], 0)
    return dict(recs=recs, mean_rf=pooled_rf, env=pooled_env, rel=np.nan_to_num(rel), mask=mask, coh=coh, range_m=np.arange(E.shape[1]) * BIN_M,
                per_rec_env=E, resid_db=[p["resid_db"] for p in per],
                summary=dict(n_frames=[p["n"] for p in per], guard_bins=UWB_GUARD, median_rel_guarded=float(np.median(rel[mask])),
                             frac_rel_gt_0p8=float((rel[mask] > 0.8).mean()), rf_coherence_median=float(np.median(coh[mask])),
                             env_cross_rec_corr_guarded=float(np.mean(corr_rows(E[:, mask]))), resid_db=[p["resid_db"] for p in per]))


# ----------------------------------------------------------------------------- WiFi
def wifi_static(rec, k):
    row = row_for(rec); x = pickle.load(open(IO.resolve(row[f"node_{k}_wifi_data_path"]), "rb"))
    csi = np.asarray(x["data"])                                   # (F,2,114) complex
    mag = np.abs(csi); mean_mag = mag.mean(0)                      # (2,114)
    cmean = csi.mean(0); coh = np.abs(cmean) / mag.mean(0).clip(1e-12)        # complex-mean survival per antenna/subcarrier
    relph = (np.conj(csi[:, 0]) * csi[:, 1]); relph_coh = np.abs(relph.mean(0)) / np.abs(relph).mean(0).clip(1e-12)   # antenna-relative phase stability
    # fixed notches: subcarriers whose magnitude is < 10 % of the median on > 90 % of packets
    notch = ((mag < 0.1 * np.median(mag, axis=2, keepdims=True)).mean(0) > 0.9).any(0)
    resid_db = float(10*np.log10(((mag - mean_mag)**2).mean() / (mean_mag**2).mean()))
    return dict(rec=rec, k=k, n=len(csi), mean_mag=mean_mag, cmean=cmean, coh=coh, relph_mean=relph.mean(0), relph_coh=relph_coh, notch=notch, resid_db=resid_db)


def wifi_pool(recs):
    out = {}
    for k in WIFI_RX:
        per = []
        for r in recs:
            try: per.append(wifi_static(r, k))
            except Exception as e: print(f"  wifi rx{k} {r}: missing ({e.__class__.__name__})")
        if not per: continue
        M = np.stack([p["mean_mag"] for p in per])                # (R,2,114)
        w = np.array([p["n"] for p in per], float); w /= w.sum()
        notch = np.any([p["notch"] for p in per], 0)
        rel = np.zeros(114)
        for sc in range(114):
            lo, hi = max(0, sc-2), min(114, sc+3)
            rel[sc] = np.mean(corr_rows(M[:, :, lo:hi].reshape(len(per), -1)))
        mask = ~notch
        out[k] = dict(recs=[p["rec"] for p in per], mean_mag=(M * w[:, None, None]).sum(0), rel=np.nan_to_num(rel), mask=mask, notch=notch,
                      coh=np.mean([p["coh"] for p in per], 0), relph_coh=np.mean([p["relph_coh"] for p in per], 0),
                      relph_mean=np.mean([p["relph_mean"] for p in per], 0), per_rec_mag=M,
                      summary=dict(n_packets=[p["n"] for p in per], n_notch=int(notch.sum()), notch_idx=np.nonzero(notch)[0].tolist(),
                                   median_rel=float(np.median(rel[mask])), complex_mean_coherence_median=float(np.median(np.mean([p["coh"] for p in per], 0)[:, mask])),
                                   antenna_relative_coherence_median=float(np.median(np.mean([p["relph_coh"] for p in per], 0)[mask])),
                                   mag_cross_rec_corr=float(np.mean(corr_rows(M[:, :, mask].reshape(len(per), -1)))), resid_db=[p["resid_db"] for p in per]))
    return out


def main():
    recs = recs5(); print("recordings:", recs)
    du = os.path.join(C.NBS_DATA_OUT, "uwb_target_S1"); dw = os.path.join(C.NBS_DATA_OUT, "wifi_target_S1"); os.makedirs(du, exist_ok=True); os.makedirs(dw, exist_ok=True)
    U = uwb_pool(recs); print("UWB:", json.dumps(U["summary"]))
    np.savez(os.path.join(du, "target_pool5.npz"), **{k: v for k, v in U.items() if isinstance(v, np.ndarray)}, recs=np.array(recs))
    json.dump(U["summary"], open(os.path.join(du, "target_pool5.json"), "w"), indent=1)
    W = wifi_pool(recs)
    for k, v in W.items():
        print(f"WiFi rx{k}:", json.dumps(v["summary"]))
        np.savez(os.path.join(dw, f"target_pool5_rx{k}.npz"), **{kk: vv for kk, vv in v.items() if isinstance(vv, np.ndarray)}, recs=np.array(v["recs"]))
        json.dump(v["summary"], open(os.path.join(dw, f"target_pool5_rx{k}.json"), "w"), indent=1)
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    viz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", "render_vs_gt")
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    r = U["range_m"]; a = ax[0, 0]
    for e, rc in zip(U["per_rec_env"], recs): a.plot(r, 20*np.log10(e/U["env"].max()+1e-9), lw=0.6, alpha=0.6, label=rc[:14])
    a.plot(r, 20*np.log10(U["env"]/U["env"].max()+1e-9), "k", lw=1.2, label="pooled"); a.axvspan(0, UWB_GUARD*BIN_M, color="0.85", label="guard (direct pulse)")
    a.set_xlim(0, 9.9); a.set_ylim(-70, 2); a.set_xlabel("range [m]"); a.set_ylabel("|envelope| dB rel. max"); a.set_title("UWB static target: temporal-mean frame envelope, 5 recordings"); a.legend(fontsize=7); a.grid(alpha=.3)
    a = ax[0, 1]; a.plot(r, U["rel"], label="cross-recording reliability (9-bin window)"); a.plot(r, U["coh"], label="frame-to-frame RF coherence"); a.axvspan(0, UWB_GUARD*BIN_M, color="0.85")
    a.set_xlim(0, 9.9); a.set_ylim(0, 1.05); a.set_xlabel("range [m]"); a.set_title(f"UWB: median reliability {U['summary']['median_rel_guarded']:.2f}, RF coherence {U['summary']['rf_coherence_median']:.3f}; residual {np.mean(U['summary']['resid_db']):.1f} dB"); a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[0, 2]; a.axis("off")
    txt = ["UWB", f"  frames per recording {U['summary']['n_frames']}", f"  guard {UWB_GUARD} bins = {UWB_GUARD*BIN_M:.2f} m", f"  envelope cross-rec corr (guarded) {U['summary']['env_cross_rec_corr_guarded']:.3f}", "", "WiFi (per receiver)"]
    for k, v in W.items():
        s = v["summary"]; txt.append(f"  rx{k}: packets {s['n_packets']}, notches {s['n_notch']} {s['notch_idx'][:12]}{'…' if s['n_notch']>12 else ''}")
        txt.append(f"        |CSI| cross-rec corr {s['mag_cross_rec_corr']:.3f}, rel {s['median_rel']:.2f}; complex-mean coherence {s['complex_mean_coherence_median']:.3f}; antenna-relative coherence {s['antenna_relative_coherence_median']:.3f}; residual {np.mean(s['resid_db']):.1f} dB")
    a.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8.5)
    for i, k in enumerate(list(W)[:3]):
        v = W[k]; a = ax[1, i]; sc = np.arange(114)
        for ant in (0, 1):
            for m in v["per_rec_mag"]: a.plot(sc, 20*np.log10(m[ant]/v["mean_mag"].max()+1e-9), lw=0.5, alpha=0.4, color=["tab:blue", "tab:orange"][ant])
            a.plot(sc, 20*np.log10(v["mean_mag"][ant]/v["mean_mag"].max()+1e-9), lw=1.5, color=["tab:blue", "tab:orange"][ant], label=f"ant {ant} pooled")
        a.plot(sc[v["notch"]], np.full(v["notch"].sum(), -40), "kx", label="fixed notch (masked)")
        a2 = a.twinx(); a2.plot(sc, v["coh"].mean(0), "g--", lw=1, label="complex-mean coherence"); a2.plot(sc, v["relph_coh"], "m:", lw=1.2, label="antenna-relative coherence"); a2.set_ylim(0, 1.05); a2.legend(loc="lower right", fontsize=7)
        a.set_ylim(-45, 2); a.set_xlabel("subcarrier"); a.set_ylabel("|CSI| dB"); a.set_title(f"WiFi Rx-{k} static target: |CSI| per antenna, 5 recordings"); a.legend(loc="lower left", fontsize=7); a.grid(alpha=.3)
    fig.suptitle("UWB / WiFi static targets (T17): temporal mean pooled over 5 Scene-1 recordings, with reliability and coherence", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(viz, "rf_static_targets.png"), dpi=150)
    with open(os.path.join(viz, "captions.txt"), "a") as fh:
        fh.write(f"rf_static_targets.png: UWB and WiFi static targets pooled over 5 recordings. UWB: guard {UWB_GUARD} bins, median cross-rec reliability {U['summary']['median_rel_guarded']:.2f}, RF frame coherence {U['summary']['rf_coherence_median']:.3f}. WiFi: " +
                 "; ".join(f"rx{k} notches {v['summary']['n_notch']}, |CSI| cross-rec corr {v['summary']['mag_cross_rec_corr']:.2f}, complex-mean coherence {v['summary']['complex_mean_coherence_median']:.2f}, antenna-relative coherence {v['summary']['antenna_relative_coherence_median']:.2f}" for k, v in W.items()) + ".\n")


if __name__ == "__main__":
    main()
