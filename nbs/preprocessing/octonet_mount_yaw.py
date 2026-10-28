"""octonet_mount_yaw: mount orientation of the Vayyar SFCW radar and the UWB radar on node_1.

The cameras are calibrated (octonet_calibrate); the radars on node_1 were given the camera yaw BY
ASSUMPTION. Here the boresight is swept around the camera yaw (position fixed at the node_1 rigid
body) and, for each hypothesis, the calibrated static scaffold is rendered and compared with the
measured static return:
  SFCW : (i) sfcw_target.sfcw_static_loss (log|range profile| per pair, pooled 5-recording target,
         xcorr weights), (ii) masked correlation of the pair-mean log|range profile|, (iii) masked 2-D
         correlation of log|RA| (only meaningful if the pair-order check passes).
  UWB  : prototype incoherent Gaussian-pulse sum (2.5 GHz, 1/R^4) over the scaffold inside a beam
         cone, vs the measured CIR beyond the coupling (masked correlation of dB magnitudes).
Preceded by a PAIR-ORDER SANITY CHECK: the direct Tx->Rx coupling (bins 0-2) must fall off smoothly
with the Tx-Rx element distance under the assumed tx-major ordering of the 400 pairs.

ASSUMPTIONS stated in the outputs: the SFCW forward model has no antenna pattern, so an element
pattern w(theta) = cos^p(theta) (theta = angle off boresight, zero beyond 60 deg; p = 2, and p = 4 as
a sensitivity check) is applied to the scaffold points - without it the pair-mean range profile is
yaw-invariant. Uniform ITU concrete, normal incidence, single bounce. UWB = prototype, not a head.

Writes $NBS_DATA_OUT/calib_S1/mount_yaw.json and viz/render_vs_gt/mount_yaw_sweep.png.
Run on the node: cd nbs && CUDA_VISIBLE_DEVICES=5 python -m preprocessing.octonet_mount_yaw
"""
from __future__ import annotations
import os, json, pickle, math, time
import numpy as np
import torch

from preprocessing import octonet_scene as S, octonet_io as IO, octonet_config as C, vayyar_radar_config as VR
from method import sfcw_forward as SF, sfcw_target as ST
from method.freq_material import ITU_GT, C_LIGHT
from mm3DPS.data import ra_utils as RA

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = np.arange(-60, 61, 2)                 # yaw offsets (deg) around the camera yaw
PITCHES = np.array([-15, -10, -5, 0, 5, 10, 15])
GRID_FULL = np.arange(-180, 181, 4)          # coarse full circle: is the +-60 edge trend a global optimum?
BEAM_MAX_DEG = 60.0
BEAM_P = (2, 4)
UWB_CONES = (32.0, 45.0)
N_MAX = 20000
EPS = 1e-12


def calib_dir(): return os.path.join(C.NBS_DATA_OUT, "calib_S1")


# --------------------------------------------------------------------------- inputs
def load_target() -> dict:
    z = np.load(os.path.join(C.NBS_DATA_OUT, "sfcw_target_S1", "target_pool5.npz"))
    return dict(rp_mean=torch.from_numpy(z["rp_mean"]).to(torch.complex64), mask=torch.from_numpy(z["mask"]).bool(),
                xcorr=torch.from_numpy(z["xcorr"]).float(), rel=torch.from_numpy(z["rel"]).float(), range_m=z["range_m"])


def load_scaffold(n_max=N_MAX):
    p = os.path.join(C.NBS_DATA_OUT, "scaffold_S1_calibrated", "pcl.npy")
    a = np.load(p); pcl, nrm = a[:, :3], a[:, 3:6]
    r_xy = np.linalg.norm(pcl[:, :2], axis=1); keep = (r_xy > 0.7) & (pcl[:, 2] > -0.3) & (pcl[:, 2] < 3.0)
    pcl, nrm = pcl[keep], nrm[keep]
    if len(pcl) > n_max:
        idx = np.random.default_rng(0).choice(len(pcl), n_max, replace=False); pcl, nrm = pcl[idx], nrm[idx]
    return pcl, nrm, p


def camera_yaw_deg() -> float:
    R1, _, src = S.camera_pose(1); f = R1[:, 2]
    return float(np.degrees(np.arctan2(f[1], f[0]))), src


def boresight(psi_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    p, q = np.radians(psi_deg), np.radians(pitch_deg)
    return np.array([np.cos(p) * np.cos(q), np.sin(p) * np.cos(q), np.sin(q)])


def pearson(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + EPS))


# --------------------------------------------------------------------------- 1a pair-order check
def pair_order_check(target: dict, radar_pos: np.ndarray) -> dict:
    rp = target["rp_mean"].numpy()                                  # (400,100)
    coup = np.abs(rp[:, :3]).sum(1)                                 # direct Tx->Rx coupling energy per pair
    M = IO.pairs_to_tx_rx(coup[:, None])[:, :, 0]                   # (20 tx, 20 rx) under tx-major
    cfg = VR.build_config(radar_pos=radar_pos, boresight=(1.0, 0.0, 0.0))
    tx = np.array([a["pos_mm"] for a in cfg["tx_array"]]) / 1000; rx = np.array([a["pos_mm"] for a in cfg["rx_array"]]) / 1000
    d = np.linalg.norm(tx[:, None, :] - rx[None, :, :], axis=-1)    # (20,20) element distance (uniform-pitch L-array)
    logM = np.log(M + EPS)
    c_tx = pearson(logM, -np.log(d)); c_rx = pearson(logM.T, -np.log(d))
    # smoothness: correlation with the 4-neighbour mean (an interleaved/scrambled order would be rough)
    pad = np.pad(logM, 1, mode="edge"); nb = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
    smooth = pearson(logM, nb)
    # also: does the coupling depend mostly on one index (row) or both? row/col variance share
    rv = logM.mean(1).var(); cv = logM.mean(0).var(); tv = logM.var()
    ok = (max(c_tx, c_rx) > 0.5) and (smooth > 0.8)
    verdict = ("PASS: coupling is smooth over the 20x20 grid (smoothness %.2f) and correlates with -log(element distance): tx-major %.2f, rx-major %.2f -> the 400-pair axis is a %s 20x20 grid" % (smooth, c_tx, c_rx, "tx-major" if c_tx >= c_rx else "rx-major (TRANSPOSED vs the assumption)")) if ok else \
              ("FAIL: coupling smoothness %.2f, distance corr tx-major %.2f / rx-major %.2f -> the pair order / element layout is unverified; azimuth evidence is NOT used" % (smooth, c_tx, c_rx))
    return dict(coupling_20x20=M.tolist(), corr_neglogdist_txmajor=c_tx, corr_neglogdist_rxmajor=c_rx, smoothness=smooth,
                row_var_share=float(rv / tv), col_var_share=float(cv / tv), passed=bool(ok), verdict=verdict, element_dist=d.tolist())


# --------------------------------------------------------------------------- SFCW render with an element pattern
def beam_weights(pcl: np.ndarray, radar_pos: np.ndarray, bore: np.ndarray, p: int) -> np.ndarray:
    v = pcl - radar_pos; v /= np.linalg.norm(v, axis=1, keepdims=True) + EPS
    cos_t = np.clip(v @ bore, -1, 1)
    w = np.where(np.degrees(np.arccos(cos_t)) < BEAM_MAX_DEG, np.maximum(cos_t, 0) ** p, 0.0)
    return w


class SfcwRenderer:
    def __init__(self, pcl, radar_pos):
        self.pcl = pcl; self.radar_pos = radar_pos
        self.pos = torch.tensor(pcl, dtype=torch.float64, device=DEV)
        self.abcd = torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=DEV).expand(len(pcl), 4)
        self.freqs = SF.sfcw_freqs_hz(VR.F_START_HZ, VR.F_STOP_HZ, VR.N_FREQ)

    def render(self, bore: np.ndarray, beam_p: int, n_levels: int = 8) -> torch.Tensor:
        """(400,100) complex range profiles, tx-major pair order, element pattern applied by weight groups."""
        cfg = VR.build_config(radar_pos=self.radar_pos, boresight=bore)
        tx = torch.tensor([a["pos_mm"] for a in cfg["tx_array"]], dtype=torch.float64, device=DEV) / 1000
        rx = torch.tensor([a["pos_mm"] for a in cfg["rx_array"]], dtype=torch.float64, device=DEV) / 1000
        w = beam_weights(self.pcl, self.radar_pos, bore, beam_p)
        edges = np.linspace(0, 1, n_levels + 1); H = torch.zeros(VR.N_TX, VR.N_RX, VR.N_FREQ, dtype=torch.complex128, device=DEV)
        with torch.no_grad():
            for g in range(n_levels):
                sel = (w > edges[g]) & (w <= edges[g + 1]) if g > 0 else (w > 0) & (w <= edges[1])
                if not sel.any(): continue
                wg = float(w[sel].mean())
                H += wg * SF.render_sfcw_pairs(self.abcd[sel], self.pos[sel], tx, rx, self.freqs, geom_mode="monostatic")
            rp = SF.sfcw_to_range(H)                                        # (20,20,100)
        return rp.reshape(VR.N_TX * VR.N_RX, VR.N_FREQ).to(torch.complex64).cpu()


def sfcw_metrics(rp_r: torch.Tensor, target: dict, ra_ok: bool) -> dict:
    loss, _ = ST.sfcw_static_loss(rp_r, target, kind="logmag", weight="xcorr")
    m = target["mask"].numpy()
    pr = np.log(rp_r.abs().mean(0).numpy() + EPS); pt = np.log(target["rp_mean"].abs().mean(0).numpy() + EPS)
    prof_corr = pearson(pr[m], pt[m])
    out = dict(loss=float(loss), prof_corr=prof_corr, ra_corr=float("nan"))
    if ra_ok:
        ra_r = np.log(RA.vayyar_to_ra_complex(ST._pairs_to_txrx_torch(rp_r), n_az=128).abs().numpy() + EPS)
        ra_t = np.log(RA.vayyar_to_ra_complex(ST._pairs_to_txrx_torch(target["rp_mean"]), n_az=128).abs().numpy() + EPS)
        out["ra_corr"] = pearson(ra_r[:, m], ra_t[:, m])
    return out


# --------------------------------------------------------------------------- UWB prototype
def uwb_gt():
    frames, _t = IO.load_uwb(S.modality_path(1, "uwb", S.REC_SIT))     # 17 Hz stream (the pickle holds one dict per frame)
    mag = np.abs(frames).mean(0); nb = len(mag); r = np.arange(nb) * S.UWB["max_range_m"] / S.UWB["n_bins"]
    return mag, r


def uwb_render(pcl, radar_pos, bore, cone_deg, r):
    v = pcl - radar_pos; R = np.linalg.norm(v, axis=1); cos_t = (v @ bore) / (R + EPS)
    sel = np.degrees(np.arccos(np.clip(cos_t, -1, 1))) < cone_deg
    Rs = R[sel]; amp = 1.0 / Rs ** 4
    sig_r = (C_LIGHT / S.UWB["bandwidth_hz"]) / 2.355 / 2
    cir = np.zeros(len(r))
    for lo in range(0, len(Rs), 4000):
        cir += (amp[lo:lo + 4000, None] * np.exp(-0.5 * ((r[None, :] - Rs[lo:lo + 4000, None]) / sig_r) ** 2)).sum(0)
    return cir, int(sel.sum())


def db(x): x = np.abs(np.asarray(x, float)); return 20 * np.log10(x / (x.max() + EPS) + 1e-30)


# --------------------------------------------------------------------------- peak analysis
def peak_analysis(grid: np.ndarray, curve: np.ndarray, higher_is_better=True, excl_deg=10.0) -> dict:
    c = np.asarray(curve, float); c = c if higher_is_better else -c
    i = int(np.nanargmax(c)); rest = np.abs(grid - grid[i]) > excl_deg
    med, sd = float(np.nanmedian(c[rest])), float(np.nanstd(c[rest]))
    z = (c[i] - med) / (sd + EPS); edge = i in (0, len(grid) - 1)
    real = bool(z > 3 and not edge and np.isfinite(z))
    return dict(best_offset_deg=float(grid[i]), best_value=float(curve[i]), value_at_zero=float(curve[np.argmin(np.abs(grid))]),
                flat_level=float(med if higher_is_better else -med), prominence_z=float(z), at_edge=bool(edge), real_peak=real)


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    target = load_target(); pcl, nrm, scaffold_path = load_scaffold()
    radar_pos = S.NODE_POS_S1[1].copy(); psi_cam, pose_src = camera_yaw_deg()
    print(f"camera yaw {psi_cam:.1f} deg ({pose_src}); scaffold {len(pcl)} pts from {scaffold_path}")
    poc = pair_order_check(target, radar_pos); print("pair-order check:", poc["verdict"])
    ren = SfcwRenderer(pcl, radar_pos)
    res = {}
    for p in BEAM_P:
        rows = []
        for d in GRID:
            rp = ren.render(boresight(psi_cam + d), p); rows.append(sfcw_metrics(rp, target, poc["passed"]))
        res[p] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        print(f"  SFCW beam cos^{p}: {len(GRID)} yaws done ({time.time() - t0:.0f} s)")
    sf = {}
    for p in BEAM_P:
        sf[p] = dict(loss=peak_analysis(GRID, res[p]["loss"], higher_is_better=False), prof_corr=peak_analysis(GRID, res[p]["prof_corr"]))
        if poc["passed"]: sf[p]["ra_corr"] = peak_analysis(GRID, res[p]["ra_corr"])
    # pitch sweep at the best loss yaw (beam p=2)
    best_yaw = sf[2]["loss"]["best_offset_deg"]
    pitch_rows = [sfcw_metrics(ren.render(boresight(psi_cam + best_yaw, q), 2), target, poc["passed"]) for q in PITCHES]
    pitch_curve = {k: np.array([r[k] for r in pitch_rows]) for k in pitch_rows[0]}
    # UWB
    mag, r = uwb_gt(); m_u = (r > 0.3) & (r < 7.0); gt_db = db(mag)
    uw = {}
    for cone in UWB_CONES:
        cc = []; npts = []
        for d in GRID:
            cir, n = uwb_render(pcl, radar_pos, boresight(psi_cam + d), cone, r); npts.append(n)
            cc.append(pearson(db(cir + EPS)[m_u], gt_db[m_u]) if n > 50 else float("nan"))
        uw[cone] = dict(curve=np.array(cc), n_pts=np.array(npts), peak=peak_analysis(GRID, np.array(cc)))
    # full-circle coarse sweep (beam cos^2 / cone 32): tells whether the fine-grid edge trends are global optima
    full_rows = [sfcw_metrics(ren.render(boresight(psi_cam + d), 2), target, False) for d in GRID_FULL]
    full = {k: np.array([r[k] for r in full_rows]) for k in ("loss", "prof_corr")}
    full["uwb"] = np.array([pearson(db(uwb_render(pcl, radar_pos, boresight(psi_cam + d), UWB_CONES[0], r)[0] + EPS)[m_u], gt_db[m_u]) for d in GRID_FULL])
    full_pk = dict(loss=peak_analysis(GRID_FULL, full["loss"], higher_is_better=False, excl_deg=20), prof_corr=peak_analysis(GRID_FULL, full["prof_corr"], excl_deg=20),
                   uwb=peak_analysis(GRID_FULL, full["uwb"], excl_deg=20))
    print(f"  full circle ({time.time() - t0:.0f} s): " + "; ".join(f"{k}: best {v['best_offset_deg']:+.0f} deg z={v['prominence_z']:.1f} real={v['real_peak']}" for k, v in full_pk.items()))
    # verdicts
    def verdict(pk, name):
        return (f"measured: {name} peak at d-yaw {pk['best_offset_deg']:+.0f} deg (value {pk['best_value']:.3f} vs {pk['value_at_zero']:.3f} at the camera yaw; prominence z={pk['prominence_z']:.1f})"
                if pk["real_peak"] else f"unmeasurable with this evidence ({name}: best {pk['best_offset_deg']:+.0f} deg, z={pk['prominence_z']:.1f}{', at grid edge' if pk['at_edge'] else ''}); keeping camera yaw")
    sf_main = sf[2]["loss"]; sf_verdict = verdict(sf_main, "SFCW static loss, beam cos^2")
    agree = all(abs(sf[p][k]["best_offset_deg"] - sf_main["best_offset_deg"]) <= 6 for p in BEAM_P for k in sf[p] if sf[p][k]["real_peak"]) if sf_main["real_peak"] else False
    uw_main = uw[UWB_CONES[0]]["peak"]; uw_verdict = verdict(uw_main, f"UWB prototype, cone +-{UWB_CONES[0]:.0f} deg")
    chosen_sfcw = psi_cam + sf_main["best_offset_deg"] if sf_main["real_peak"] else psi_cam
    chosen_uwb = psi_cam + uw_main["best_offset_deg"] if uw_main["real_peak"] else psi_cam
    out = dict(camera_yaw_deg=psi_cam, pose_source=pose_src, grid_offset_deg=GRID.tolist(), pitch_grid_deg=PITCHES.tolist(),
               assumptions=["element pattern cos^p(theta), zero beyond 60 deg (the SFCW forward model has no antenna pattern)",
                            "radar position = node_1 rigid body", "uniform ITU concrete, normal incidence, single bounce",
                            "UWB render is a prototype incoherent pulse-sum, not a 4DPS head", "RA evidence valid only if the pair-order check passes"],
               pair_order_check={k: v for k, v in poc.items() if k not in ("coupling_20x20", "element_dist")},
               sfcw={f"beam_cos{p}": dict(curves={k: v.tolist() for k, v in res[p].items()}, peaks=sf[p]) for p in BEAM_P},
               sfcw_pitch_at_best_yaw={k: v.tolist() for k, v in pitch_curve.items()},
               uwb={f"cone_{int(c)}": dict(curve=v["curve"].tolist(), n_pts=v["n_pts"].tolist(), peak=v["peak"]) for c, v in uw.items()},
               verdict_sfcw=sf_verdict + ("" if not sf_main["real_peak"] else (" ; other metrics/beams agree within 6 deg" if agree else " ; WARNING other metrics/beams disagree -> treat as weak")),
               verdict_uwb=uw_verdict, chosen_yaw_deg=dict(vayyar=float(chosen_sfcw), uwb=float(chosen_uwb)),
               full_circle=dict(grid_deg=GRID_FULL.tolist(), curves={k: v.tolist() for k, v in full.items()}, peaks=full_pk),
               scaffold=scaffold_path, n_points=len(pcl), seconds=time.time() - t0)
    os.makedirs(calib_dir(), exist_ok=True)
    with open(os.path.join(calib_dir(), "mount_yaw.json"), "w") as f: json.dump(out, f, indent=1)
    print("SFCW:", out["verdict_sfcw"]); print("UWB :", out["verdict_uwb"])
    for p in BEAM_P:
        for k, v in sf[p].items(): print(f"   beam cos^{p} {k:9s}: best {v['best_offset_deg']:+.0f} deg z={v['prominence_z']:.1f} real={v['real_peak']} (at 0: {v['value_at_zero']:.3f}, best {v['best_value']:.3f}, flat {v['flat_level']:.3f})")
    for c, v in uw.items(): print(f"   UWB cone {c:.0f}: best {v['peak']['best_offset_deg']:+.0f} deg z={v['peak']['prominence_z']:.1f} real={v['peak']['real_peak']} (at 0: {v['peak']['value_at_zero']:.3f}, best {v['peak']['best_value']:.3f})")
    figure(out, poc, res, pitch_curve, uw, gt_db, r, m_u, full, full_pk)
    print("wrote", os.path.join(calib_dir(), "mount_yaw.json"), f"({time.time() - t0:.0f} s)")


def figure(out, poc, res, pitch_curve, uw, gt_db, r, m_u, full, full_pk):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    viz = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "viz", "render_vs_gt"); os.makedirs(viz, exist_ok=True)
    fig, ax = plt.subplots(3, 3, figsize=(19, 15))
    M = np.log(np.array(poc["coupling_20x20"]) + EPS); d = np.array(poc["element_dist"])
    a = ax[0, 0]; im = a.imshow(M, cmap="viridis", origin="lower"); a.set_xlabel("rx index"); a.set_ylabel("tx index"); plt.colorbar(im, ax=a, label="log |coupling| (bins 0-2)")
    a.set_title(f"pair-order check: coupling on the assumed tx-major 20x20 grid\nsmoothness {poc['smoothness']:.2f}; corr with -log(dist): tx-major {poc['corr_neglogdist_txmajor']:.2f}, rx-major {poc['corr_neglogdist_rxmajor']:.2f}", fontsize=9)
    a = ax[0, 1]; a.scatter(d.ravel() * 1000, M.ravel(), s=6, label="tx-major"); a.scatter(d.T.ravel() * 1000, M.ravel(), s=6, alpha=.5, label="rx-major (transposed)")
    a.set_xlabel("Tx-Rx element distance (mm, uniform-pitch L-array)"); a.set_ylabel("log |coupling|"); a.legend(); a.grid(alpha=.3)
    a.set_title("PASS" if poc["passed"] else "FAIL: azimuth axis unverified", color="g" if poc["passed"] else "r")
    a = ax[0, 2]
    for p, ls in zip(BEAM_P, ("-", "--")):
        a.plot(GRID, res[p]["loss"], "C0", ls=ls, label=f"static loss (logmag, xcorr w), beam cos^{p}")
    a.set_xlabel("yaw offset from camera yaw (deg)"); a.set_ylabel("loss (lower = better)", color="C0"); a.axvline(0, color="k", lw=.8); a.grid(alpha=.3)
    b = a.twinx()
    for p, ls in zip(BEAM_P, ("-", "--")):
        b.plot(GRID, res[p]["prof_corr"], "C1", ls=ls, label=f"pair-mean log|profile| corr, cos^{p}")
    b.set_ylabel("correlation (higher = better)", color="C1"); a.legend(loc="upper left", fontsize=8); b.legend(loc="upper right", fontsize=8)
    a.set_title("SFCW yaw sweep vs pooled static target\n" + out["verdict_sfcw"][:110], fontsize=9)
    a = ax[1, 0]
    if poc["passed"]:
        for p, ls in zip(BEAM_P, ("-", "--")): a.plot(GRID, res[p]["ra_corr"], "C2", ls=ls, label=f"log|RA| 2-D corr, cos^{p}")
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "RA correlation not used:\npair order unverified", ha="center", va="center", transform=a.transAxes)
    a.axvline(0, color="k", lw=.8); a.set_xlabel("yaw offset (deg)"); a.set_ylabel("correlation"); a.grid(alpha=.3); a.set_title("SFCW RA (azimuth) evidence", fontsize=9)
    a = ax[1, 1]; a.plot(PITCHES, pitch_curve["loss"], "C0o-", label="static loss"); a.set_xlabel("pitch (deg) at the best yaw, beam cos^2"); a.set_ylabel("loss", color="C0"); a.grid(alpha=.3)
    b = a.twinx(); b.plot(PITCHES, pitch_curve["prof_corr"], "C1s--", label="profile corr"); b.set_ylabel("corr", color="C1"); a.set_title("SFCW pitch sweep (position fixed)", fontsize=9)
    a = ax[1, 2]
    for c, v in uw.items(): a.plot(GRID, v["curve"], label=f"UWB prototype: dB|CIR| corr, cone +-{c:.0f} deg")
    a.axvline(0, color="k", lw=.8); a.set_xlabel("yaw offset (deg)"); a.set_ylabel("correlation (0.3-7 m)"); a.grid(alpha=.3); a.legend(fontsize=8)
    a.set_title("UWB yaw sweep (PROTOTYPE pulse-sum, not a head)\n" + out["verdict_uwb"][:110], fontsize=9)
    a = ax[2, 0]; a.plot(GRID_FULL, full["loss"], "C0"); a.set_ylabel("static loss", color="C0"); b = a.twinx(); b.plot(GRID_FULL, full["prof_corr"], "C1"); b.set_ylabel("profile corr", color="C1")
    a.axvline(0, color="k", lw=.8); a.axvspan(-60, 60, color="k", alpha=.05); a.set_xlabel("yaw offset (deg), full circle, 4 deg steps"); a.grid(alpha=.3)
    a.set_title(f"SFCW full circle (beam cos^2): loss best {full_pk['loss']['best_offset_deg']:+.0f} deg z={full_pk['loss']['prominence_z']:.1f}; profile corr best {full_pk['prof_corr']['best_offset_deg']:+.0f} deg z={full_pk['prof_corr']['prominence_z']:.1f}", fontsize=9)
    a = ax[2, 1]; a.plot(GRID_FULL, full["uwb"], "C0"); a.axvline(0, color="k", lw=.8); a.axvspan(-60, 60, color="k", alpha=.05); a.set_xlabel("yaw offset (deg), full circle"); a.set_ylabel("dB|CIR| corr"); a.grid(alpha=.3)
    a.set_title(f"UWB full circle (cone +-32): best {full_pk['uwb']['best_offset_deg']:+.0f} deg z={full_pk['uwb']['prominence_z']:.1f} real={full_pk['uwb']['real_peak']}", fontsize=9)
    a = ax[2, 2]; a.axis("off")
    a.text(0, 1, "Reading the sweeps\n\n* The SFCW forward model has no antenna pattern; the cos^p weighting is an assumption\n  and the only thing that makes the pair-mean range profile yaw-dependent.\n* A real mount yaw should give a peak that (a) stands > 3 sigma above the rest of the\n  curve, (b) is not at the grid edge, (c) agrees across metrics and beam widths.\n* The pair-order check: the 400 pairs DO form a smooth 20x20 grid (smoothness %.2f),\n  but the coupling follows the Tx index in two blocks of 10, not the uniform-pitch\n  L-array distance (corr %.2f): the element geometry is wrong, so azimuth/RA is unverified.\n* UWB: prototype pulse-sum, no antenna pattern beyond a hard cone." % (poc["smoothness"], poc["corr_neglogdist_txmajor"]), va="top", family="monospace", fontsize=8.5)
    fig.suptitle(f"node_1 radar mount yaw sweep around the calibrated camera yaw ({out['camera_yaw_deg']:.1f} deg); radar position = node_1 rigid body; element pattern cos^p (assumed)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(viz, "mount_yaw_sweep.png"), dpi=150); plt.close(fig)
    with open(os.path.join(viz, "captions.txt"), "a") as f:
        f.write(f"\nmount_yaw_sweep.png: node_1 Vayyar/UWB mount yaw swept +-60 deg around the calibrated camera yaw ({out['camera_yaw_deg']:.1f} deg), radar at the node_1 rigid body, "
                f"static scaffold rendered with an ASSUMED element pattern cos^p (p=2,4; the forward model has none). Pair-order check: {poc['verdict']}. "
                f"SFCW verdict: {out['verdict_sfcw']}. UWB verdict (prototype pulse-sum, not a head): {out['verdict_uwb']}.\n")


if __name__ == "__main__":
    main()
