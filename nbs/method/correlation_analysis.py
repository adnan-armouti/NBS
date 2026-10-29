"""Why are the SFCW / UWB correlations lower than 3DPS's FMCW results, and what would raise them?  (2026-10-29)
Measured, not guessed:
  A. ceilings: (A1) GT reproducibility across recordings (what any render can reach against the pooled target);
     (A2) geometric-uncertainty decay: render-vs-render correlation when the scaffold is perturbed by 0.5-4 cm / 10 deg;
  B. scale curve: correlation vs range smoothing (fine-scale geometry deficit vs coarse-scale physics deficit);
  C. per-pair range delays: the raw Vayyar channels' coupling peaks sit at different fractional bins per pair (electrical
     delays, not calibrated); estimate them, decompose into Tx + Rx, and apply them to the GT -> correlation change;
  D. the static person/chair at the origin (cut from the scaffold) added back as a reflector;
  E. per-point six-parameter fit (octonet_fit) vs the 24 cm voxel grid (eval panels);
  U. the UWB equivalents of A/B.
Run on the node: cd nbs && CUDA_VISIBLE_DEVICES=3 python -m method.correlation_analysis
"""
import json, os, sys, math
import numpy as np, torch
from preprocessing import octonet_config as C, octonet_scene as S, octonet_io as IO, vayyar_radar_config as VR
from method import sfcw_target as ST
from mm3DPS.data.ra_utils import sfcw_range_compress
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs
DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT); dev = "cuda"; out = {}
def db(x): return 20 * np.log10(np.abs(np.asarray(x)) + 1e-300)
def norm_img(X, mask):          # masked log-magnitude image, median-normalised (same quantity as the fit metrics)
    Y = db(X)[:, mask]; return Y - np.median(Y)
def corr(a, b): a = np.asarray(a).ravel(); b = np.asarray(b).ravel(); return float(np.corrcoef(a, b)[0, 1])
def smooth(X, w):               # boxcar along the last axis
    if w <= 1: return X
    k = np.ones(w) / w; return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), -1, X)
K = VR.N_FREQ
# ------------------------------------------------------------------ SFCW inputs
z = np.load(os.path.join(DATA, "sfcw_target_S1", "target_pool5.npz")); mask = z["mask"].astype(bool); mask[:24] = False
rp_pool = z["rp_mean"]; H_pool = np.asarray(z["H_mean"]) if "H_mean" in z.files else None
G = norm_img(rp_pool, mask); prof_gt = db(rp_pool).mean(0)
ev = np.load(os.path.join(DATA, "eval_panels_S1_v3", "sfcw_tight_s0.npz")); r1_grid = ev["r1"]
fit = np.load(os.path.join(DATA, "fit_S1_sfcw", "material.npz"), allow_pickle=True)
rec_paths = [str(r) for r in ST.pick_scene1_recordings(5)]; recs = [os.path.basename(r)[:14] for r in rec_paths]
def load_rec(rec): return IO.load_vayyar(rec_paths[recs.index(rec)])
# A1. GT reproducibility: per-recording targets vs the pool
singles = []
for rec in recs:
    V, _ = load_rec(rec); tg = ST.sfcw_static_target(V, coupling="tone2")
    singles.append((rec, tg["rp_mean"]))
c_single_pool = [corr(G, norm_img(rp, mask)) for _, rp in singles]
c_single_single = [corr(norm_img(a, mask), norm_img(b, mask)) for i, (_, a) in enumerate(singles) for (_, b) in singles[i + 1:]]
prof_corr_single = [corr(prof_gt[mask] - np.median(prof_gt[mask]), db(rp).mean(0)[mask] - np.median(db(rp).mean(0)[mask])) for _, rp in singles]
out["sfcw_A1_gt_reproducibility"] = dict(pair_image_corr_single_vs_pool=c_single_pool, pair_image_corr_single_vs_single=c_single_single,
                                         profile_corr_single_vs_pool=prof_corr_single)
print("SFCW A1: pair-image corr single-vs-pool %.3f (min %.3f), single-vs-single median %.3f; profile corr single-vs-pool %.3f" % (np.mean(c_single_pool), np.min(c_single_pool), np.median(c_single_single), np.mean(prof_corr_single)))
# E + B: trained renders vs GT at smoothing scales
cloud, material0, sensor = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=30000)
xyz_fit = torch.as_tensor(fit["xyz"], dtype=torch.float64, device=dev); nrm_fit = torch.as_tensor(fit["normals"], dtype=torch.float64, device=dev)
phys = torch.as_tensor(fit["physics"], dtype=torch.float32, device=dev)
sens_fit = dict(sensor, gain_tx=np.exp(fit["gain_tx"]), gain_rx=np.exp(fit["gain_rx"]))
with torch.no_grad():
    r1_pp = OH.sfcw_head({"xyz": xyz_fit, "normals": nrm_fit}, {"physics": phys}, sens_fit).cpu().numpy()
scales = [1, 3, 5, 9, 15]
out["sfcw_B_scale_curve"] = {}
for name, R in (("grid V=64 (eval panels)", r1_grid), ("per-point six-param (fit-static)", r1_pp)):
    Rn = norm_img(R, mask); row = {}
    for w in scales: row[str(w)] = corr(smooth(G, w), smooth(Rn, w))
    row["profile_corr"] = corr(prof_gt[mask] - np.median(prof_gt[mask]), db(R).mean(0)[mask] - np.median(db(R).mean(0)[mask]))
    out["sfcw_B_scale_curve"][name] = row; print(f"SFCW B/E {name}: image corr vs smoothing {[(w, round(row[str(w)],3)) for w in scales]}, profile corr {row['profile_corr']:.3f}")
gt_scale = {str(w): float(np.mean([corr(smooth(G, w), smooth(norm_img(rp, mask), w)) for _, rp in singles])) for w in scales}
out["sfcw_B_scale_curve"]["GT single-vs-pool (ceiling)"] = gt_scale; print("SFCW B ceiling GT single-vs-pool:", {w: round(v, 3) for w, v in gt_scale.items()})
# A2. geometric-uncertainty decay (render-vs-render, per-point material)
out["sfcw_A2_geometry_decay"] = {}
with torch.no_grad():
    base = norm_img(r1_pp, mask)
    for sig in (0.005, 0.01, 0.02, 0.04):
        g = torch.Generator(device=dev).manual_seed(0)
        xyz_p = xyz_fit + torch.randn(xyz_fit.shape, generator=g, device=dev, dtype=torch.float64) * sig
        rp = OH.sfcw_head({"xyz": xyz_p, "normals": nrm_fit}, {"physics": phys}, sens_fit).cpu().numpy()
        out["sfcw_A2_geometry_decay"][f"{sig*100:.1f} cm"] = dict(image=corr(base, norm_img(rp, mask)), image_smooth5=corr(smooth(base, 5), smooth(norm_img(rp, mask), 5)),
                                                                 profile=corr(db(r1_pp).mean(0)[mask], db(rp).mean(0)[mask]))
        print(f"SFCW A2 position noise {sig*100:.1f} cm: render-vs-render image corr {out['sfcw_A2_geometry_decay'][f'{sig*100:.1f} cm']['image']:.3f}, smooth5 {out['sfcw_A2_geometry_decay'][f'{sig*100:.1f} cm']['image_smooth5']:.3f}, profile {out['sfcw_A2_geometry_decay'][f'{sig*100:.1f} cm']['profile']:.3f}")
    # normals noise 10 deg
    g = torch.Generator(device=dev).manual_seed(1); n_p = nrm_fit + torch.randn(nrm_fit.shape, generator=g, device=dev, dtype=torch.float64) * math.tan(math.radians(10)); n_p = n_p / n_p.norm(dim=1, keepdim=True)
    rp = OH.sfcw_head({"xyz": xyz_fit, "normals": n_p}, {"physics": phys}, sens_fit).cpu().numpy()
    out["sfcw_A2_geometry_decay"]["normals 10 deg"] = dict(image=corr(base, norm_img(rp, mask)), profile=corr(db(r1_pp).mean(0)[mask], db(rp).mean(0)[mask]))
    print("SFCW A2 normals 10 deg: image corr %.3f, profile %.3f" % (out["sfcw_A2_geometry_decay"]["normals 10 deg"]["image"], out["sfcw_A2_geometry_decay"]["normals 10 deg"]["profile"]))
# C. per-pair range delays from the coupling (raw mean sweep, before coupling removal)
V, _ = IO.load_vayyar(S.modality_path(1, "vayyar", S.REC_SIT)); Hraw = V.mean(0)                     # (400,100) raw
f = np.asarray(VR.F_START_HZ + np.arange(K) * VR.FREQ_STEP_HZ); hann = np.hanning(K)
taus_bins = np.linspace(-1.0, 12.0, 1301)                                                            # candidate delays in range bins (3.75 cm)
E = np.exp(1j * 2 * np.pi * (f - f[0])[None, :] * (taus_bins[:, None] * (2 * VR.RANGE_RES_M) / 3e8))  # phase ramp for delay tau (round-trip 2*R)
score = np.abs((Hraw * hann[None, :]) @ E.T)                                                          # (400, n_tau)
tau_p = taus_bins[np.argmax(score, axis=1)]                                                           # per-pair coupling peak position (bins)
M = tau_p.reshape(20, 20); tx_d = M.mean(1) - M.mean(); rx_d = M.mean(0) - M.mean(); add = M.mean() + tx_d[:, None] + rx_d[None, :]
out["sfcw_C_delays"] = dict(peak_bin_median=float(np.median(tau_p)), p5=float(np.percentile(tau_p, 5)), p95=float(np.percentile(tau_p, 95)), std_bins=float(tau_p.std()),
                            additive_explained=float(1 - ((M - add) ** 2).mean() / M.var()), tx_delay_bins=tx_d.tolist(), rx_delay_bins=rx_d.tolist())
print("SFCW C per-pair coupling peak: median %.2f bins, p5-p95 %.2f-%.2f, std %.2f bins; tx+rx additive explains %.0f%%" % (np.median(tau_p), np.percentile(tau_p, 5), np.percentile(tau_p, 95), tau_p.std(), 100 * out["sfcw_C_delays"]["additive_explained"]))
# apply the per-pair delay to the GT pool (remove the electrical delay: shift each pair's profile by -(tau_p - ref)), ref = the physical on-board path (min)
if H_pool is None:
    H_pool = np.mean([_np(ST.sfcw_static_target(load_rec(r)[0], coupling="tone2")["H_mean"]) for r in recs], axis=0)
def _np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def shift_pairs(Hm, delta_bins):
    Hm = _np(Hm)
    ramp = np.exp(1j * 2 * np.pi * (f - f[0])[None, :] * (-delta_bins[:, None] * (2 * VR.RANGE_RES_M) / 3e8))
    return sfcw_range_compress(torch.from_numpy((Hm * ramp).astype(np.complex128))).numpy()
out["sfcw_C_apply"] = {}
for name, delta in (("per-pair (400)", tau_p - np.median(tau_p)), ("tx+rx additive (39)", add.ravel() - np.median(add))):
    rp_cal = shift_pairs(H_pool, delta); Gc = norm_img(rp_cal, mask)
    out["sfcw_C_apply"][name] = dict(grid=corr(Gc, norm_img(r1_grid, mask)), per_point=corr(Gc, norm_img(r1_pp, mask)),
                                     gt_single_vs_pool=float(np.mean([corr(Gc, norm_img(shift_pairs(ST.sfcw_static_target(load_rec(r)[0], coupling="tone2")["H_mean"], delta), mask)) for r in recs[:2]])))
    print(f"SFCW C apply {name} delays to GT: image corr vs grid render {out['sfcw_C_apply'][name]['grid']:.3f} (was {corr(G, norm_img(r1_grid, mask)):.3f}), vs per-point {out['sfcw_C_apply'][name]['per_point']:.3f} (was {corr(G, norm_img(r1_pp, mask)):.3f})")
# D. the static person at the origin
try:
    ps = np.load(os.path.join(DATA, "4dps_person_S1", S.REC_SIT, "person_sequence.npz"))
    vtx = ps["vertices_world"][ps["valid"].astype(bool)].mean(0); nv = ps["normals_world"][ps["valid"].astype(bool)].mean(0); nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-9
    idx = np.random.default_rng(0).choice(len(vtx), 3000, replace=False)
    with torch.no_grad():
        xyz_d = torch.cat([xyz_fit, torch.as_tensor(vtx[idx], dtype=torch.float64, device=dev)]); nrm_d = torch.cat([nrm_fit, torch.as_tensor(nv[idx], dtype=torch.float64, device=dev)])
        phys_d = torch.cat([phys, phys.mean(0, keepdim=True).expand(len(idx), 6)])
        rp_d = OH.sfcw_head({"xyz": xyz_d, "normals": nrm_d}, {"physics": phys_d}, sens_fit).cpu().numpy()
    pm = prof_gt[mask] - np.median(prof_gt[mask]); band = (np.arange(K)[mask] * 0.0375 > 2.6)
    out["sfcw_D_person"] = dict(profile_corr_without=corr(pm, db(r1_pp).mean(0)[mask] - np.median(db(r1_pp).mean(0)[mask])), profile_corr_with=corr(pm, db(rp_d).mean(0)[mask] - np.median(db(rp_d).mean(0)[mask])),
                                image_corr_without=corr(G, norm_img(r1_pp, mask)), image_corr_with=corr(G, norm_img(rp_d, mask)),
                                far_band_rmse_without=float(np.sqrt(np.mean((pm[band] - (db(r1_pp).mean(0)[mask] - np.median(db(r1_pp).mean(0)[mask]))[band]) ** 2))),
                                far_band_rmse_with=float(np.sqrt(np.mean((pm[band] - (db(rp_d).mean(0)[mask] - np.median(db(rp_d).mean(0)[mask]))[band]) ** 2))))
    print("SFCW D person added (mean SMPL of the sit recording, 3k pts, mean material):", {k: round(v, 3) for k, v in out["sfcw_D_person"].items()})
except Exception as e:
    print("SFCW D skipped:", e)
# ------------------------------------------------------------------ UWB
from scipy.signal import hilbert
uz = np.load(os.path.join(DATA, "uwb_target_S1", "target_pool5.npz")); umask = uz["mask"].astype(bool); env_pool = uz["env"]; prec = uz["per_rec_env"]
ue = np.load(os.path.join(DATA, "eval_panels_S1_v3", "uwb_tight_s2.npz")); r1u = ue["r1"]; env_r = np.abs(hilbert(r1u)) if np.isrealobj(r1u) else np.abs(r1u)
uscales = [1, 5, 15, 31, 61]
def ucorr(a, b, w): A = smooth(db(a)[None, :], w)[0][umask]; B = smooth(db(b)[None, :], w)[0][umask]; return corr(A - np.median(A), B - np.median(B))
out["uwb_scale_curve"] = {"GT single-vs-pool (ceiling)": {str(w): float(np.mean([ucorr(e, env_pool, w) for e in prec])) for w in uscales},
                          "trained render vs pool": {str(w): ucorr(env_r, env_pool, w) for w in uscales}}
for k, v in out["uwb_scale_curve"].items(): print(f"UWB {k}: corr vs smoothing (bins of 6.4 mm) {[(w, round(v[str(w)],3)) for w in uscales]}")
# UWB geometry decay with the grid material at the points
try:
    cl_u = {"xyz": torch.as_tensor(ue["xyz"], dtype=torch.float64, device=dev), "normals": None}
    mat_u = {"abcd": torch.as_tensor(ue["abcd"], dtype=torch.float64, device=dev)}
    sens_u = OH.sensor_from_extrinsics(json.load(open(os.path.join(DATA, "calib_S1", "extrinsics.json"))), "uwb")
    with torch.no_grad():
        rf0, env0 = OH.uwb_head(cl_u, mat_u, sens_u); env0 = env0.cpu().numpy()
        out["uwb_geometry_decay"] = {}
        for sig in (0.005, 0.01, 0.02, 0.04):
            g = torch.Generator(device=dev).manual_seed(0); xp = cl_u["xyz"] + torch.randn(cl_u["xyz"].shape, generator=g, device=dev, dtype=torch.float64) * sig
            _, envp = OH.uwb_head({"xyz": xp, "normals": None}, mat_u, sens_u); envp = envp.cpu().numpy()
            out["uwb_geometry_decay"][f"{sig*100:.1f} cm"] = {str(w): ucorr(env0, envp, w) for w in (1, 15, 61)}
            print(f"UWB geometry noise {sig*100:.1f} cm: render-vs-render env corr at 1/15/61 bins", {w: round(out['uwb_geometry_decay'][f'{sig*100:.1f} cm'][str(w)], 3) for w in (1, 15, 61)})
except Exception as e:
    print("UWB geometry decay skipped:", e)
json.dump(out, open(os.path.join(DATA, "correlation_analysis.json"), "w"), indent=1, default=float); print("saved")
