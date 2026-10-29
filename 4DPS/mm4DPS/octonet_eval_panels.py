"""Single-modality fits with GT / iter-0 / trained panels and metric tables (viz/render_vs_gt_<tag>).

Each band (SFCW, UWB, WiFi) is fitted ALONE with octonet_fit_multiband's model: per-point 3DPS six-parameter material
(default; ``--material grid`` = the T11 voxel field), the ported 4DPS noise floor and coupling block (``--noise-model``,
``--coupling``), under TIGHT ITU bounds (a in [1,15], b in +-0.1, c <= 0.1 S/m, d <= 1.0) for 3 seeds, plus one
LOOSE-bound run (b +-0.2, c <= 2, d <= 1.5). "iter 0" = the material's initialisation = uniform ITU concrete with the
iteration-0 gain / floor.

    cd 4DPS && CUDA_VISIBLE_DEVICES=3 python -m mm4DPS.octonet_eval_panels fit --band sfcw [--seeds 0,1,2] [--steps 400]
    python -m mm4DPS.octonet_eval_panels figures            # all bands -> viz/render_vs_gt_v2/ + README.md + captions.txt
Outputs under $NBS_DATA_OUT/eval_panels_S1/<band>_<bounds>_s<seed>.{json,npz}.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch

from mm4DPS import octonet_fit_multiband as MB
from method.freq_material import ITU_GT

# OCTONET_PANELS_TAG selects the output set: "v2" (default, historical) or e.g. "v3" (clean rerun with the 3DPS-routed heads)
_TAG = os.environ.get("OCTONET_PANELS_TAG", "v2")
OUT = os.path.join(MB.DATA_OUT, "eval_panels_S1" if _TAG == "v2" else f"eval_panels_S1_{_TAG}")
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", f"render_vs_gt_{_TAG}")
BANDS = ("sfcw", "uwb", "wifi")
TIGHT = dict(a=(1.0, 15.0), b=0.1, c=(1e-3, 0.1), d=1.0)
LOOSE = dict(a=(1.0, 15.0), b=0.2, c=(1e-3, 2.0), d=1.5)
EPS = 1e-12


# ----------------------------------------------------------------------------- bounds switch (monkey-patches the fit module)
def set_bounds(kind: str):
    B = TIGHT if kind == "tight" else LOOSE
    a0, a1 = B["a"]; c0, c1 = B["c"]

    def abcd_from_raw(raw):
        a = a0 + (a1 - a0) * torch.sigmoid(raw[..., 0]); b = B["b"] * torch.tanh(raw[..., 1])
        c = torch.exp(math.log(c0) + torch.sigmoid(raw[..., 2]) * (math.log(c1) - math.log(c0))); d = B["d"] * torch.sigmoid(raw[..., 3])
        return torch.stack([a, b, c, d], -1)

    def raw_from_abcd(abcd):
        a, b, c, d = [float(v) for v in abcd]
        def logit(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
        return torch.tensor([logit((a - a0) / (a1 - a0)), math.atanh(max(min(b / B["b"], 0.999), -0.999)),
                             logit((math.log(max(c, c0 * 1.001)) - math.log(c0)) / (math.log(c1) - math.log(c0))), logit(min(d / B["d"], 0.999))], dtype=torch.float64)
    MB.abcd_from_raw = abcd_from_raw; MB.raw_from_abcd = raw_from_abcd
    return B


# ----------------------------------------------------------------------------- metrics
def db(x): return 20.0 * np.log10(np.asarray(x, float) + EPS)


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum()) + EPS
    return float((a * b).sum() / d)


def _smooth_rows(A, w):
    k = np.ones(w) / w
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, A) if w > 1 else A


def metrics_sfcw(rp, tg):
    """rp (400,100) complex render; tg: dict with rp_mean (400,100), mask (100,). ``*_far`` = bins >= 24 (0.9 m), the pre-coupling-model
    supervised region (comparable with v2/v3 and with the data ceiling); ``corr_image_far_5bin`` at 19 cm smoothing."""
    def core(m, w=1):
        A = _smooth_rows(db(np.abs(rp)), w)[:, m]; G = _smooth_rows(db(np.abs(tg["rp_mean"])), w)[:, m]
        A = A - np.median(A); G = G - np.median(G)
        return A, G
    m = tg["mask"]; A, G = core(m)
    per_pair = np.array([_pearson(A[i], G[i]) for i in range(A.shape[0])])
    pa, pg = A.mean(0), G.mean(0)
    far = m.copy(); far[:24] = False
    Af, Gf = core(far); Af5, Gf5 = core(far, 5)
    return dict(rmse_db=float(np.sqrt(((A - G) ** 2).mean())), corr_image=_pearson(A.ravel(), G.ravel()), corr_pair_median=float(np.median(per_pair)),
                corr_pair_mean_profile=_pearson(pa, pg), rmse_db_pair_mean_profile=float(np.sqrt(((pa - pg) ** 2).mean())),
                rmse_db_far=float(np.sqrt(((Af - Gf) ** 2).mean())), corr_image_far=_pearson(Af.ravel(), Gf.ravel()), corr_image_far_5bin=_pearson(Af5.ravel(), Gf5.ravel()),
                corr_pair_mean_profile_far=_pearson(Af.mean(0), Gf.mean(0)), first_bin=int(np.argmax(m)))


def metrics_uwb(env, tg):
    """``corr_ge1m`` / ``corr_1_9m``: bins >= 1.0 m (the pre-coupling-model region) and 1-9 m (without the unexplained 9-10 m return);
    ``corr_1_9m_61bin`` at 39 cm smoothing. All median-normalised on their own region."""
    m = tg["mask"]; r = tg["range_m"]; A = db(env); G = db(tg["env"])
    def c(mm, w=1):
        a = _smooth_rows(A, w)[mm]; g = _smooth_rows(G, w)[mm]; a = a - np.median(a); g = g - np.median(g)
        return _pearson(a, g), float(np.sqrt(((a - g) ** 2).mean()))
    An = A - np.median(A[m]); Gn = G - np.median(G[m])
    near = m & (r >= 0.3) & (r < 4.0); far = m & (r >= 4.0); ge1 = m & (r >= 1.0); mid = m & (r >= 1.0) & (r < 9.0)
    return dict(rmse_db=float(np.sqrt(((An - Gn)[m] ** 2).mean())), corr_all=_pearson(An[m], Gn[m]), corr_0p3_4m=_pearson(An[near], Gn[near]), corr_4_9p9m=_pearson(An[far], Gn[far]),
                rmse_db_0p3_4m=float(np.sqrt(((An - Gn)[near] ** 2).mean())), rmse_db_4_9p9m=float(np.sqrt(((An - Gn)[far] ** 2).mean())),
                corr_ge1m=c(ge1)[0], rmse_db_ge1m=c(ge1)[1], corr_1_9m=c(mid)[0], corr_1_9m_61bin=c(mid, 61)[0], corr_ge1m_61bin=c(ge1, 61)[0], first_bin=int(np.argmax(m)))


def metrics_wifi(csi, tg):
    out = {}; alls = []; allg = []
    for k in (1, 2, 3, 4):
        m = tg[k]["mask"]; A = db(csi[k]); G = db(tg[k]["mean_mag"])
        A = A - np.median(A[:, m], axis=1, keepdims=True); G = G - np.median(G[:, m], axis=1, keepdims=True)
        out[f"rx{k}"] = dict(rmse_db=float(np.sqrt(((A - G)[:, m] ** 2).mean())), corr=_pearson(A[:, m].ravel(), G[:, m].ravel()))
        alls.append(A[:, m].ravel()); allg.append(G[:, m].ravel())
    a = np.concatenate(alls); g = np.concatenate(allg)
    out["all"] = dict(rmse_db=float(np.sqrt(((a - g) ** 2).mean())), corr=_pearson(a, g))
    return out


def targets_np(targets):
    t = {"sfcw": {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in targets["sfcw"].items() if k in ("rp_mean", "mask", "range_m", "xcorr")},
         "uwb": {k: v.cpu().numpy() for k, v in targets["uwb"].items() if torch.is_tensor(v)},
         "wifi": {k: {kk: vv.cpu().numpy() for kk, vv in d.items()} for k, d in targets["wifi"].items()}}
    return t


def render_all(band, cloud, mat, sensors, targets, nuisance=None):
    with torch.no_grad():
        _, r = MB.band_loss(band, cloud, mat, sensors, targets, nuisance=nuisance)
    if band == "sfcw": return r["rp"].cpu().numpy()
    if band == "uwb": return r["env"].cpu().numpy()
    return {k: v.cpu().numpy() for k, v in r["csi"].items()}


def metrics(band, rend, tnp):
    return {"sfcw": metrics_sfcw, "uwb": metrics_uwb, "wifi": metrics_wifi}[band](rend, tnp[band])


DB_PER_LOG = 20.0 / math.log(10.0)          # log-amplitude -> dB


def gt_channel_gains(tg):
    """The GT's per-pair median level (dB, tx-major 20x20) and its additive per-Tx / per-Rx decomposition (D13: 92 % of the
    4 dB spread is tx + rx). Returns dict(level (20,20), tx (20,), rx (20,)) in dB about the global mean."""
    m = tg["mask"]; G = db(np.abs(tg["rp_mean"]))[:, m]
    lvl = np.median(G, 1).reshape(20, 20); lvl = lvl - lvl.mean()
    return dict(level=lvl, tx=lvl.mean(1), rx=lvl.mean(0))


def gain_metrics(gains: dict, tg) -> dict:
    """Fitted gains vs the GT-derived ones: stds in dB and Pearson correlations (per-Tx, per-Rx, per-pair pattern)."""
    if not gains or not gains.get("enabled"):
        return dict(enabled=False)
    gtx = DB_PER_LOG * np.asarray(gains["tx_log"], float); grx = DB_PER_LOG * np.asarray(gains["rx_log"], float)
    gtx = gtx - gtx.mean(); grx = grx - grx.mean()
    ref = gt_channel_gains(tg)
    fitted_pairs = gtx[:, None] + grx[None, :]
    return dict(enabled=True, tx_std_db=float(gtx.std()), rx_std_db=float(grx.std()), gt_tx_std_db=float(ref["tx"].std()), gt_rx_std_db=float(ref["rx"].std()),
                corr_tx=_pearson(gtx, ref["tx"]), corr_rx=_pearson(grx, ref["rx"]), corr_pairs=_pearson(fitted_pairs.ravel(), ref["level"].ravel()),
                tx_db=gtx.tolist(), rx_db=grx.tolist(), gt_tx_db=ref["tx"].tolist(), gt_rx_db=ref["rx"].tolist())



NOTES = {
 "sfcw": "SFCW alone (T23 head = the 3DPS rasterizer; D13 per-channel gains {gains_state}): the trained pair-mean profile follows the GT bin for bin over "
         "0.9-3.5 m including the floor-ring dips and the rise beyond 2.6 m (the folded -X wall), profile corr {profile_corr:.3f}, RMSE {profile_rmse:.2f} dB. "
         "The per-pair image correlation is {corr_image:.3f} (per-pair median {corr_pair:.3f}); without gains a pair-blind render is capped at ~0.45 by the GT's "
         "fixed per-Tx/per-Rx gain banding (4 dB spread, 92 % additive). The fitted gains have std {gain_tx:.2f}/{gain_rx:.2f} dB (GT-derived {gt_tx:.2f}/{gt_rx:.2f}) "
         "and correlate {corr_tx:.2f}/{corr_rx:.2f} with the GT's per-Tx/per-Rx pattern ({corr_pairs:.2f} on the 400-pair pattern). What remains is the per-pair "
         "ripple within each profile, which is below the scaffold's fidelity (8-bit depth, 3.9 cm quantisation at lambda 4.6 mm) and the uncalibrated per-element "
         "phases, so it is not a target. Loose bounds: see the table.",
 "uwb": "UWB alone: the trained envelope tracks the GT mean level over the whole 0.3-9.9 m span (iter 0 was +15 dB too bright below 1.5 m and -20 dB too dark at "
        "6-7 m), and the 6.3-6.8 m far-wall cluster appears at the right range. The GT's fine structure (20-30 dB interference fades every few cm) is not "
        "reproduced, so the correlation saturates near 0.5 and the RMSE at 6.7 dB: that structure needs the scaffold geometry to sub-cm accuracy and the UWB "
        "mount pose, both of which are assumed (camera yaw, node marker). The near range (0.3-1.5 m) keeps a +5-10 dB excess: node body returns absent from the scaffold.",
 "wifi": "WiFi alone: the trained |CSI| reproduces the broad spectral shape of every receiver and antenna (corr 0.87, RMSE 1.3 dB), including Rx-2's deep fade at "
         "subcarrier ~35 on antenna 0 and the NLOS Rx-4 slope; residuals are within +-3 dB except Rx-1 antenna 1 above subcarrier 90. This is achieved with "
         "SCHEMATIC Tx/Rx positions (+-0.5 m), so the material field is absorbing geometry error: the fitted eps' (median 5.2) differs from the SFCW-alone "
         "(3.2) and UWB-alone (1.6) fields of the same scene, which is the cross-band caveat.",
}

# ----------------------------------------------------------------------------- fit command
BAND_DEFAULTS = dict(sfcw=(0.03, 400), uwb=(0.03, 400), wifi=(0.01, 600))      # (lr, steps)


def cmd_fit(a):
    os.makedirs(OUT, exist_ok=True)
    if a.lr is None: a.lr = BAND_DEFAULTS[a.band][0]
    if a.steps is None: a.steps = BAND_DEFAULTS[a.band][1]
    dev = torch.device(a.device); band = a.band
    seeds = [int(s) for s in a.seeds.split(",")]
    plan = [("tight", s) for s in seeds] + ([("loose", seeds[0])] if a.loose else [])
    for kind, seed in plan:
        B = set_bounds(kind); tag = f"{band}_{kind}_s{seed}"
        print(f"=== {tag}: V={a.voxels} steps={a.steps} bounds={B}", flush=True)
        res, grid, abcd, cloud, sensors, targets = MB.fit((band,), dev, a.points, a.steps, a.lr, a.voxels, a.tv, seed, tag=tag, gains=not a.no_gains, gain_l2=a.gain_l2, patience=a.patience,
                                                          **MB.fit_kwargs(a))
        nuis = res.pop("nuis")
        tnp = targets_np(targets)
        mat0 = dict(abcd=torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=dev).expand(cloud["xyz"].shape[0], 4).contiguous(), rho=None)
        mat1 = grid.material_at(cloud["xyz"])
        nuis0 = None
        if band in nuis:                                   # iteration-0 nuisance state (gain / floor at their init), not the trained one
            nuis0 = MB.BandNuisance(band, dev, a.noise_model == "on", a.coupling == "on"); nuis0.init(cloud, mat0, MB.sensors_without_gains(sensors), targets)
        r0 = render_all(band, cloud, mat0, MB.sensors_without_gains(sensors), targets, nuis0); r1 = render_all(band, cloud, mat1, sensors, targets, nuis.get(band))
        m0 = metrics(band, r0, tnp); m1 = metrics(band, r1, tnp)
        gm = gain_metrics(res.get("gains"), tnp["sfcw"]) if band == "sfcw" else dict(enabled=False)
        res = dict(res); res.update(bounds=kind, bounds_values={k: (list(v) if isinstance(v, tuple) else v) for k, v in B.items()}, metrics_iter0=m0, metrics_trained=m1,
                   dispersion=MB.dispersion_summary(abcd), loss_iter0=res["start"][band], loss_trained=res["end"][band], gain_metrics=gm)
        res.pop("log", None)
        json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
        save = dict(xyz=cloud["xyz"].cpu().numpy(), abcd=abcd, mask=tnp[band]["mask"] if band != "wifi" else np.zeros(0))
        save["physics"] = MB.physics_columns(mat1, MB.BAND_FC_GHZ[band])         # six columns at the band centre (default roughness for the grid)
        if gm.get("enabled"):
            save.update(gain_tx_db=np.asarray(gm["tx_db"]), gain_rx_db=np.asarray(gm["rx_db"]), gt_gain_tx_db=np.asarray(gm["gt_tx_db"]), gt_gain_rx_db=np.asarray(gm["gt_rx_db"]))
        if band == "wifi":
            for k in (1, 2, 3, 4): save[f"r0_{k}"] = r0[k]; save[f"r1_{k}"] = r1[k]
        else:
            save["r0"] = r0; save["r1"] = r1
        np.savez(os.path.join(OUT, tag + ".npz"), **save)
        print(f"  {tag}: loss {res['loss_iter0']:.4f} -> {res['loss_trained']:.4f} | {json.dumps(m1)[:300]}", flush=True)
        if gm.get("enabled"):
            print(f"  {tag}: gains tx/rx std {gm['tx_std_db']:.2f}/{gm['rx_std_db']:.2f} dB (GT {gm['gt_tx_std_db']:.2f}/{gm['gt_rx_std_db']:.2f}); corr tx {gm['corr_tx']:.3f} rx {gm['corr_rx']:.3f} pairs {gm['corr_pairs']:.3f}", flush=True)


# ----------------------------------------------------------------------------- figures + tables
def load_runs(band):
    runs = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith(band + "_") and f.endswith(".json"):
            runs[f[:-5]] = json.load(open(os.path.join(OUT, f)))
    return runs


def best_tight(band, runs):
    cands = {k: v for k, v in runs.items() if v["bounds"] == "tight"}
    return min(cands, key=lambda k: cands[k]["loss_trained"]) if cands else None


def ms(vals, fmt="{:.3f}"):
    v = np.array(vals, float)
    return (fmt.format(v.mean()) + " ± " + fmt.format(v.std())) if len(v) > 1 else fmt.format(v[0])


def fig_sfcw(tnp, z, tag, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tg = tnp["sfcw"]; m = tg["mask"]; r = tg["range_m"][m]
    G = db(np.abs(tg["rp_mean"]))[:, m]; A0 = db(np.abs(z["r0"]))[:, m]; A1 = db(np.abs(z["r1"]))[:, m]
    G -= np.median(G); A0 -= np.median(A0); A1 -= np.median(A1)
    vmin, vmax = np.percentile(G, 2), np.percentile(G, 98)
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    for j, (X, t) in enumerate(((G, "GT (pooled 5 recordings)"), (A0, "rendered, iter 0 (uniform concrete)"), (A1, "rendered, trained (SFCW alone)"))):
        im = ax[0, j].imshow(X, aspect="auto", origin="lower", extent=[r[0], r[-1], 0, 400], vmin=vmin, vmax=vmax, cmap="viridis")
        ax[0, j].set_title(f"|rp| per pair, dB rel. median\n{t}", fontsize=10); ax[0, j].set_xlabel("range [m]"); ax[0, j].set_ylabel("Tx-Rx pair (tx-major)")
        plt.colorbar(im, ax=ax[0, j], label="dB")
        ax[1, j].plot(r, G.mean(0), "k", lw=1.5, label="GT"); ax[1, j].plot(r, X.mean(0), "r" if j else "k", lw=1.2, label=t if j else None)
        ax[1, j].set_xlabel("range [m]"); ax[1, j].set_ylabel("pair-mean dB rel. median"); ax[1, j].set_title(f"pair-mean range profile (supervised bins {int(np.argmax(m))}-93 = {r.min():.2f}-{r.max():.2f} m)"); ax[1, j].grid(alpha=.3); ax[1, j].legend(fontsize=8)
    im = ax[0, 3].imshow(A1 - G, aspect="auto", origin="lower", extent=[r[0], r[-1], 0, 400], vmin=-10, vmax=10, cmap="coolwarm"); plt.colorbar(im, ax=ax[0, 3], label="dB")
    ax[0, 3].set_title("residual: trained - GT [dB]"); ax[0, 3].set_xlabel("range [m]"); ax[0, 3].set_ylabel("pair")
    ax[1, 3].plot(r, A0.mean(0) - G.mean(0), "C0", label="iter 0 - GT"); ax[1, 3].plot(r, A1.mean(0) - G.mean(0), "C3", label="trained - GT"); ax[1, 3].axhline(0, color="k", lw=.5)
    ax[1, 3].set_xlabel("range [m]"); ax[1, 3].set_ylabel("dB"); ax[1, 3].set_title("pair-mean residual"); ax[1, 3].grid(alpha=.3); ax[1, 3].legend(fontsize=8)
    fig.suptitle(f"SFCW (Vayyar, node_1, Scene 1): GT vs rendered at iter 0 vs trained — {tag}: per-point 3DPS material, learned noise floor + coupling block (4DPS models), tight ITU bounds; supervised from {r.min():.2f} m", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_uwb(tnp, z, tag, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tg = tnp["uwb"]; m = tg["mask"]; r = tg["range_m"]
    G = db(tg["env"]); A0 = db(z["r0"]); A1 = db(z["r1"]); G -= np.median(G[m]); A0 -= np.median(A0[m]); A1 -= np.median(A1[m])
    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    for j, (X, t, c) in enumerate(((G, "GT (pooled envelope)", "k"), (A0, "rendered, iter 0 (uniform concrete)", "C0"), (A1, "rendered, trained (UWB alone)", "C3"))):
        for i, (lo, hi) in enumerate(((0.3, 9.9), (0.3, 4.0))):
            a = ax[i, j]
            if j: a.plot(r, G, "k", lw=0.8, alpha=0.5, label="GT")
            a.plot(r, X, c, lw=0.9, label=t); a.set_xlim(lo, hi); a.set_ylim(-45, 25); a.set_xlabel("range [m] (bin 6.43 mm; 7-bin on-board delay removed)"); a.set_ylabel("log envelope, dB rel. masked median")
            a.set_title(f"UWB envelope {lo}-{hi} m: {t}"); a.grid(alpha=.3); a.legend(fontsize=8); a.axvspan(0, r[np.argmax(m)], color="0.85")
    for i, (lo, hi) in enumerate(((0.3, 9.9), (0.3, 4.0))):
        a = ax[i, 3]; a.plot(r, A0 - G, "C0", lw=0.7, label="iter 0 - GT"); a.plot(r, A1 - G, "C3", lw=0.7, label="trained - GT"); a.axhline(0, color="k", lw=.5)
        a.set_xlim(lo, hi); a.set_ylim(-30, 30); a.set_xlabel("range [m]"); a.set_ylabel("dB"); a.set_title(f"residual {lo}-{hi} m"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"UWB (X4M200, node_1): GT vs rendered at iter 0 vs trained — {tag}; measured pulse template; per-point material, learned noise floor + coupling block; supervised from {r[np.argmax(m)]:.2f} m (grey)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_wifi(tnp, z, tag, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 4, figsize=(22, 16)); sc = np.arange(114)
    for i, k in enumerate((1, 2, 3, 4)):
        tg = tnp["wifi"][k]; m = tg["mask"]; G = db(tg["mean_mag"]); A0 = db(z[f"r0_{k}"]); A1 = db(z[f"r1_{k}"])
        G -= np.median(G[:, m], 1, keepdims=True); A0 -= np.median(A0[:, m], 1, keepdims=True); A1 -= np.median(A1[:, m], 1, keepdims=True)
        for j, (X, t) in enumerate(((G, "GT (pooled |CSI|)"), (A0, "rendered, iter 0"), (A1, "rendered, trained (WiFi alone)"))):
            a = ax[i, j]
            for ant, c in ((0, "C0"), (1, "C1")):
                if j: a.plot(sc[m], G[ant][m], color="k", lw=0.8, alpha=0.4)
                a.plot(sc[m], X[ant][m], color=c, lw=1.3, label=f"antenna {ant}")
            a.plot(sc[~m], np.full((~m).sum(), -14), "kx", ms=5, label="notch (masked)" if (i == 0 and j == 0) else None)
            a.set_ylim(-16, 10); a.set_xlabel("subcarrier (HT40, 5.18 GHz)"); a.set_ylabel("|CSI| dB rel. median"); a.set_title(f"Rx-{k}{' (NLOS)' if k == 4 else ''}: {t}"); a.grid(alpha=.3); a.legend(fontsize=7, loc="lower left")
        a = ax[i, 3]
        for ant, c in ((0, "C0"), (1, "C1")):
            a.plot(sc[m], (A0 - G)[ant][m], color=c, lw=0.8, ls=":", label=f"iter0-GT ant {ant}"); a.plot(sc[m], (A1 - G)[ant][m], color=c, lw=1.3, label=f"trained-GT ant {ant}")
        a.axhline(0, color="k", lw=.5); a.set_ylim(-12, 12); a.set_xlabel("subcarrier"); a.set_ylabel("dB"); a.set_title(f"Rx-{k}: residual"); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle(f"WiFi CSI (4 receivers x 2 antennas, magnitude only): GT vs rendered at iter 0 vs trained — {tag}; schematic Tx/Rx positions", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_gains(tnp, z, run, tag, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = run["gain_metrics"]; ref = gt_channel_gains(tnp["sfcw"])
    ftx, frx = np.asarray(g["tx_db"]), np.asarray(g["rx_db"])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for a, f, r, nm in ((ax[0], ftx, ref["tx"], "Tx"), (ax[1], frx, ref["rx"], "Rx")):
        a.plot(r, "k.-", label=f"GT-derived (per-pair median level, {nm} mean)"); a.plot(f, "C3.-", label="fitted (log-amplitude gain, dB)")
        a.set_xlabel(f"{nm} index (tx-major pair order)"); a.set_ylabel("gain [dB] about the mean"); a.grid(alpha=.3); a.legend(fontsize=8)
        a.set_title(f"per-{nm} channel gain: fitted std {f.std():.2f} dB vs GT {r.std():.2f} dB, corr {_pearson(f, r):.2f}")
    fp = (ftx[:, None] + frx[None, :]).ravel(); gp = ref["level"].ravel()
    ax[2].scatter(gp, fp, s=8, alpha=.6); lim = [min(gp.min(), fp.min()) - .5, max(gp.max(), fp.max()) + .5]; ax[2].plot(lim, lim, "k--", lw=.8)
    ax[2].set_xlabel("GT per-pair median level [dB]"); ax[2].set_ylabel("fitted tx + rx gain [dB]"); ax[2].grid(alpha=.3)
    ax[2].set_title(f"400-pair gain pattern: corr {_pearson(fp, gp):.2f} (GT pattern is 92 % additive tx+rx)")
    fig.suptitle(f"SFCW per-channel gains (D13), run {tag}: L2 {run['gains'].get('l2', 0):g} on the log-amplitudes", fontsize=12); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_material(band, z, run, tag, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = MB.BAND_FC_GHZ[band]; abcd = z["abcd"]; xyz = z["xyz"]
    er = abcd[:, 0] * f ** abcd[:, 1]; sg = abcd[:, 2] * f ** abcd[:, 3]
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    for a, v, t, cmap in ((ax[0], er, f"eps' at {f} GHz", "viridis"), (ax[1], np.log10(sg), f"log10 sigma [S/m] at {f} GHz", "magma")):
        o = np.argsort(xyz[:, 2]); im = a.scatter(xyz[o, 0], xyz[o, 1], c=v[o], s=2, cmap=cmap, vmin=np.percentile(v, 2), vmax=np.percentile(v, 98))
        a.plot(0, 0, "k*", ms=10); a.plot(2.921, 0.011, "r^", ms=9); a.set_aspect("equal"); a.set_xlabel("X [m]"); a.set_ylabel("Y [m]"); a.set_title(f"fitted {t} (top-down; star = subject spot, triangle = node_1)"); plt.colorbar(im, ax=a)
    names = list(run["dispersion"]["nearest_itu_class"]); cnt = [run["dispersion"]["nearest_itu_class"][n] for n in names]
    ax[2].bar(names, cnt); ax[2].set_ylabel("points"); ax[2].set_title("nearest ITU class per point (log eps', log sigma at 5.2/7/65 GHz)")
    d = run["dispersion"][f"{f}GHz"]
    ax[2].text(0.02, 0.98, f"eps' median {d['eps_median']:.2f} [p5 {d['eps_p05']:.2f}, p95 {d['eps_p95']:.2f}]\nsigma median {d['sigma_median']:.3f} [p5 {d['sigma_p05']:.3f}, p95 {d['sigma_p95']:.3f}] S/m\nbounds: {run['bounds']} {run['bounds_values']}",
               transform=ax[2].transAxes, va="top", fontsize=8.5, family="monospace")
    fig.suptitle(f"{band.upper()}: fitted material ({tag}; {run.get('material', 'grid')})", fontsize=12); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def cmd_figures(a):
    os.makedirs(VIZ, exist_ok=True)
    dev = torch.device("cpu")
    # the figures' masks follow the runs: first supervised bin of the best tight run per band (the coupling model opens the near bins)
    g = {}
    for band in ("sfcw", "uwb"):
        runs_ = load_runs(band); bt_ = best_tight(band, runs_) if runs_ else None
        g[band] = (runs_[bt_].get("guards", {}) or {}).get(band) if bt_ else None
    targets = MB.load_targets(dev, sfcw_guard=g["sfcw"] if g["sfcw"] is not None else MB.GUARD_LO_SFCW, uwb_guard=g["uwb"]); tnp = targets_np(targets)
    README = [f"# Single-modality training check (render_vs_gt_{_TAG})", "",
              "Each band is fitted ALONE (its own target only) with `4DPS/mm4DPS/octonet_fit_multiband.py`: the 3DPS per-point six-parameter material "
              "(eps from a per-point ITU (a, c) at the band centre with the exponents frozen, sigma_h / l_c / tau / thickness per point in the 3DPS raw space; "
              "20k scaffold points, Adam lr 0.03, L2 0.05 to the init, 400 steps), the ported 4DPS receiver noise floor (measured held-out envelope prior x "
              "learned scale x 8-coefficient DCT range correction, added in power) and the ported 4DPS coupling block (coherent near-bin mean of held-out "
              "recordings, added after a learned absolute gain where it reproduces the target; the guard stays elsewhere), under TIGHT ITU bounds "
              "(a ∈ [1,15], b ∈ ±0.1, c ≤ 0.1 S/m, d ≤ 1.0) for 3 seeds, plus one LOOSE run (b ±0.2, c ≤ 2, d ≤ 1.5). "
              "'iter 0' = uniform ITU concrete with the iteration-0 gain and floor. SFCW/UWB losses are log-magnitude L1 in absolute units "
              "(SFCW: pair-wise range profiles, xcorr-weighted, 20+20 channel gains; UWB: log-envelope, reliability-weighted); WiFi keeps the median-normalised "
              "magnitude-only loss (6 HT40 notches masked). Metrics are computed on the masked, median-normalised dB quantities (so they are comparable with v2/v3). "
              "Figures: `<band>_gt_iter0_trained.png`, `<band>_material.png`. Produced by `mm4DPS/octonet_eval_panels.py`.", ""]
    summary = ["| band | loss iter 0 | loss trained (tight, 3 seeds) | loss trained (loose, 1 seed) | dB RMSE iter 0 → trained (SFCW ≥ 0.9 m, UWB ≥ 1 m) | corr iter 0 → trained (SFCW pair image ≥ 0.9 m, ceiling 0.89; UWB envelope ≥ 1 m, ceiling 0.95; WiFi all) | steps / runtime |", "|---|---|---|---|---|---|---|"]
    captions = []
    for band in BANDS:
        runs = load_runs(band)
        if not runs: README.append(f"## {band.upper()}: no runs found"); continue
        tight = [v for v in runs.values() if v["bounds"] == "tight"]; loose = [v for v in runs.values() if v["bounds"] == "loose"]
        bt = best_tight(band, runs); z = np.load(os.path.join(OUT, bt + ".npz")); run = runs[bt]
        mk = {"sfcw": ("rmse_db_far", "corr_image_far"), "uwb": ("rmse_db_ge1m", "corr_ge1m"), "wifi": None}[band]
        def M(r, which, key):
            m = r[which]; return m["all"][key] if band == "wifi" else m[key]
        k_r, k_c = ("rmse_db", "corr") if band == "wifi" else mk
        summary.append(f"| {band.upper()} | {ms([t['loss_iter0'] for t in tight])} | **{ms([t['loss_trained'] for t in tight])}** | {ms([l['loss_trained'] for l in loose]) if loose else '—'} | "
                       f"{ms([M(t,'metrics_iter0',k_r) for t in tight], '{:.2f}')} → **{ms([M(t,'metrics_trained',k_r) for t in tight], '{:.2f}')}** | "
                       f"{ms([M(t,'metrics_iter0',k_c) for t in tight])} → **{ms([M(t,'metrics_trained',k_c) for t in tight])}** | {ms([t['steps'] for t in tight], '{:.0f}')} / {ms([t['runtime_s'] for t in tight], '{:.0f}')} s |")
        README += [f"## {band.upper()}", "", f"Best tight seed: `{bt}` (loss {run['loss_trained']:.4f}). Figures: `{band}_gt_iter0_trained.png`, `{band}_material.png`.", ""]
        # per-band table
        if band == "sfcw":
            README += ["| run | gains | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |", "|---|---|---|---|---|---|---|---|---|---|"]
            for k, v in runs.items():
                a, b = v["metrics_iter0"], v["metrics_trained"]; g = v.get("gain_metrics", {}) or {}
                gcell = (f"{g['tx_std_db']:.2f}/{g['rx_std_db']:.2f} ({g['gt_tx_std_db']:.2f}/{g['gt_rx_std_db']:.2f}) | {g['corr_tx']:.2f} / {g['corr_rx']:.2f} / {g['corr_pairs']:.2f}" if g.get("enabled") else "— | —")
                README.append(f"| {k} | {'yes' if g.get('enabled') else 'no'} | {v['loss_iter0']:.4f} → {v['loss_trained']:.4f} | {a['rmse_db']:.2f} → {b['rmse_db']:.2f} | {a['corr_image']:.3f} → {b['corr_image']:.3f} | {a.get('corr_image_far', float('nan')):.3f} → **{b.get('corr_image_far', float('nan')):.3f}** / {b.get('corr_image_far_5bin', float('nan')):.3f} | {a['corr_pair_median']:.3f} → {b['corr_pair_median']:.3f} | {a['corr_pair_mean_profile']:.3f} → {b['corr_pair_mean_profile']:.3f} / {a['rmse_db_pair_mean_profile']:.2f} → {b['rmse_db_pair_mean_profile']:.2f} | {gcell} |")
        elif band == "uwb":
            README += ["| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m | RMSE dB 0.3–4 / 4–9.9 m |", "|---|---|---|---|---|---|---|---|---|"]
            for k, v in runs.items():
                a, b = v["metrics_iter0"], v["metrics_trained"]
                README.append(f"| {k} | {v['loss_iter0']:.4f} → {v['loss_trained']:.4f} | {a['rmse_db']:.2f} → {b['rmse_db']:.2f} | {a['corr_all']:.3f} → {b['corr_all']:.3f} | {a.get('corr_ge1m', float('nan')):.3f} → **{b.get('corr_ge1m', float('nan')):.3f}** / {b.get('corr_ge1m_61bin', float('nan')):.3f} | {b.get('corr_1_9m', float('nan')):.3f} / {b.get('corr_1_9m_61bin', float('nan')):.3f} | {a['corr_0p3_4m']:.3f} → {b['corr_0p3_4m']:.3f} | {a['corr_4_9p9m']:.3f} → {b['corr_4_9p9m']:.3f} | {a['rmse_db_0p3_4m']:.2f} → {b['rmse_db_0p3_4m']:.2f} / {a['rmse_db_4_9p9m']:.2f} → {b['rmse_db_4_9p9m']:.2f} |")
        else:
            README += ["| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |", "|---|---|---|---|---|---|---|---|"]
            for k, v in runs.items():
                a, b = v["metrics_iter0"], v["metrics_trained"]
                cells = " | ".join(f"{a[f'rx{i}']['rmse_db']:.2f}→{b[f'rx{i}']['rmse_db']:.2f} / {a[f'rx{i}']['corr']:.2f}→{b[f'rx{i}']['corr']:.2f}" for i in (1, 2, 3, 4))
                README.append(f"| {k} | {v['loss_iter0']:.4f} → {v['loss_trained']:.4f} | {a['all']['rmse_db']:.2f} → {b['all']['rmse_db']:.2f} | {a['all']['corr']:.3f} → {b['all']['corr']:.3f} | {cells} |")
        d = run["dispersion"]; f = MB.BAND_FC_GHZ[band]; dd = d[f"{f}GHz"]
        note = NOTES[band]
        if band == "sfcw":
            mt = run["metrics_trained"]; g = run.get("gain_metrics", {}) or {}
            fill = dict(gains_state="fitted" if g.get("enabled") else "OFF", profile_corr=mt["corr_pair_mean_profile"], profile_rmse=mt["rmse_db_pair_mean_profile"],
                        corr_image=mt["corr_image"], corr_pair=mt["corr_pair_median"],
                        **{k: g.get(k2, float("nan")) for k, k2 in (("gain_tx", "tx_std_db"), ("gain_rx", "rx_std_db"), ("gt_tx", "gt_tx_std_db"), ("gt_rx", "gt_rx_std_db"), ("corr_tx", "corr_tx"), ("corr_rx", "corr_rx"), ("corr_pairs", "corr_pairs"))})
            note = note.format(**fill)
        README += ["", note, ""]
        README += ["", f"Fitted field (best tight seed) at {f} GHz: ε′ median {dd['eps_median']:.2f} [p5 {dd['eps_p05']:.2f}, p95 {dd['eps_p95']:.2f}], σ median {dd['sigma_median']:.3f} [p5 {dd['sigma_p05']:.3f}, p95 {dd['sigma_p95']:.3f}] S/m; "
                   f"nearest ITU class: {d['nearest_itu_class']}.", ""]
        if band == "sfcw":
            fig_sfcw(tnp, z, bt, os.path.join(VIZ, "sfcw_gt_iter0_trained.png"))
            captions.append("sfcw_gt_iter0_trained.png: SFCW pair-wise |range profile| (dB, median-normalised, all supervised bins) and the pair-mean profile: GT pooled target vs rendered at iter 0 (uniform concrete, iteration-0 gain/floor) vs trained (SFCW alone, per-point 3DPS material + channel gains + learned noise floor + coupling block, tight ITU bounds); last column = residual.")
            if run.get("gain_metrics", {}).get("enabled"):
                fig_gains(tnp, z, run, bt, os.path.join(VIZ, "sfcw_gains.png"))
                captions.append("sfcw_gains.png: the fitted 20 per-Tx and 20 per-Rx log-amplitude gains (dB) of the SFCW-alone fit vs the gains implied by the GT's per-pair median levels (additive tx + rx decomposition), and the 400-pair gain pattern fitted vs GT.")
        elif band == "uwb":
            fig_uwb(tnp, z, bt, os.path.join(VIZ, "uwb_gt_iter0_trained.png"))
            captions.append("uwb_gt_iter0_trained.png: UWB log-envelope (0.3-9.9 m and 0.3-4 m zoom): GT pooled vs rendered iter 0 vs trained (UWB alone, per-point material + learned noise floor + coupling block); measured pulse template, on-board delay removed; last column = residual.")
        else:
            fig_wifi(tnp, z, bt, os.path.join(VIZ, "wifi_gt_iter0_trained.png"))
            captions.append("wifi_gt_iter0_trained.png: WiFi |CSI| per receiver and antenna (dB, median-normalised, notches masked): GT pooled vs rendered iter 0 vs trained (WiFi alone); last column = residual.")
        fig_material(band, z, run, bt, os.path.join(VIZ, f"{band}_material.png"))
        captions.append(f"{band}_material.png: fitted ITU field of the {band.upper()}-alone fit at {f} GHz (eps', sigma top-down) and the nearest-ITU-class histogram.")
    README[4:4] = ["## Summary (tight bounds, mean ± std over 3 seeds)", ""] + summary + [""]
    guards = {}
    for band in ("sfcw", "uwb"):
        runs = load_runs(band); bt = best_tight(band, runs) if runs else None
        if bt: guards[band] = runs[bt].get("guards", {}).get(band); 
    README += ["## Model state of these runs", "", f"Material / nuisances of the best tight runs: " + "; ".join(
        f"{b}: material {runs_[bt_]['material']}, noise model {'on' if runs_[bt_].get('noise_model') else 'off'}, coupling {'on' if runs_[bt_].get('coupling') else 'off'}, first supervised bin {runs_[bt_].get('guards', {}).get(b)}, nuisance {json.dumps(runs_[bt_].get('nuisance', {}).get(b))}"
        for b in ("sfcw", "uwb") for runs_ in [load_runs(b)] for bt_ in [best_tight(b, runs_)] if bt_), ""]
    README += ["## Reproduce", "", "```", "source tools/env_node.sh && cd 4DPS", "python -m mm4DPS.octonet_noise measure        # held-out envelope + coupling (once)",
               f"OCTONET_PANELS_TAG={_TAG} CUDA_VISIBLE_DEVICES=3 python -m mm4DPS.octonet_eval_panels fit --band sfcw --seeds 0,1,2 --loose     # lr 0.03, 400 steps",
               f"OCTONET_PANELS_TAG={_TAG} CUDA_VISIBLE_DEVICES=4 python -m mm4DPS.octonet_eval_panels fit --band uwb  --seeds 0,1,2 --loose",
               f"OCTONET_PANELS_TAG={_TAG} CUDA_VISIBLE_DEVICES=5 python -m mm4DPS.octonet_eval_panels fit --band wifi --seeds 0,1,2 --loose     # lr 0.01, 600 steps",
               f"OCTONET_PANELS_TAG={_TAG} python -m mm4DPS.octonet_eval_panels figures", "```", ""]
    open(os.path.join(VIZ, "README.md"), "w").write("\n".join(README) + "\n")
    open(os.path.join(VIZ, "captions.txt"), "w").write("\n".join(captions) + "\n")
    print("\n".join(summary))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--band", required=True, choices=BANDS); f.add_argument("--seeds", default="0,1,2"); f.add_argument("--loose", action="store_true")
    f.add_argument("--points", type=int, default=20000); f.add_argument("--steps", type=int, default=None, help="default 400 (WiFi 600)"); f.add_argument("--lr", type=float, default=None, help="default 0.03 (WiFi 0.01: at 0.03 one seed in three bounced out of its minimum, 2026-10-29)")
    f.add_argument("--voxels", type=int, default=64); f.add_argument("--tv", type=float, default=1e-3); f.add_argument("--device", default="cuda")
    f.add_argument("--no-gains", action="store_true", help="SFCW: do not fit the 20+20 per-channel gains (pre-D13 behaviour)")
    f.add_argument("--gain-l2", type=float, default=0.01, help="L2 weight on the SFCW log-amplitude channel gains")
    f.add_argument("--patience", type=int, default=0, help="stop when the best loss has not improved for N steps (0 = run all --steps; the panels use all steps)")
    MB.add_model_args(f)
    sub.add_parser("figures")
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    main()
