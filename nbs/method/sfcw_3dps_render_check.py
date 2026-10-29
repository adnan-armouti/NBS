"""Render the Vayyar SFCW with the REAL 3DPS rasterizer (mm3DPS.rasterizer_factorized.render_factorized: Jones BSDF
with the six material parameters, antenna patterns, MC hemisphere weight, Hann PSF, detached phase) on the calibrated
Scene-1 scaffold, and compare with the GT pooled target at iteration 0. This is what the SFCW head should have been.
Run on the node: cd nbs && CUDA_VISIBLE_DEVICES=3 python -m method.sfcw_3dps_render_check
"""
import json, os, sys, math, tempfile
import numpy as np, torch
from preprocessing import octonet_config as C, vayyar_radar_config as VR
from mm3DPS import rasterizer as RZ
from mm3DPS.rasterizer_factorized import render_factorized
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS.octonet import render_inputs
DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT); dev = "cuda:0"; out = {}
def db(x): return 20 * np.log10(np.abs(x) + 1e-300)
z = np.load(os.path.join(DATA, "sfcw_target_S1", "target_pool5.npz")); mask = z["mask"].astype(bool); mask[:24] = False
G = db(z["rp_mean"]); gt = G.mean(0); g = gt[mask]
# --- inputs: scaffold + calibrated Vayyar pose (same as the head)
cloud, material, sensor = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=40000)
pos = cloud["xyz"].float(); nrm = cloud["normals"].float(); M = len(pos)
# --- Vayyar SFCW config at node_1 (radar frame from the calibrated boresight), written to a temp file
cfg = VR.build_config(radar_pos=np.asarray(sensor["pos"], float), boresight=np.asarray(sensor["boresight"], float))
tmp = tempfile.mkdtemp(); cfg_path = os.path.join(tmp, "vayyar_s1.json"); json.dump(cfg, open(cfg_path, "w"))
# --- element pattern from the datasheet beamwidths: E-plane 180 deg, H-plane 82 deg (3 dB); gain(theta) = cos^p, p from the 3 dB point
def pattern_npy(path, e_bw=180.0, h_bw=82.0, gmax_db=5.8):
    ang = np.deg2rad(np.arange(361) - 180.0)                                   # angle from boresight, -180..180
    def plane(bw):
        p = math.log(0.5) / math.log(math.cos(math.radians(bw / 2))) if bw < 179 else 0.05
        return gmax_db + 10 * np.log10(np.clip(np.cos(ang), 1e-3, 1) ** p)      # dB, back hemisphere floor -30 dB
    arr = np.stack([plane(e_bw), plane(h_bw)], 1); np.save(path, arr); return arr
pat = os.path.join(tmp, "vayyar_pattern.npy"); pattern_npy(pat)
rast = RZ.Rasterizer(cfg_path, "/none.ply", pat, pat, device=dev)
print(f"rasterizer: {rast.modulation_scheme}, K={rast.K}, n_tx={rast.n_tx}, n_rx={rast.n_rx}, lambda {rast.wavelength*1e3:.2f} mm, range_res {rast.range_res*100:.2f} cm")
# --- materials: ITU concrete at 65 GHz (eps' 5.24, eps'' 0.46), sigma_h 0.1 mm, l_c 1 cm, tau 0.5, thickness 0.1 m
def raw_for(eps_r, eps_i, sigma_h, l_c, tau, thick):
    phys = np.tile(np.array([eps_r, eps_i, sigma_h, l_c, tau, thick], float), (M, 1))
    return torch.as_tensor(RZ.inverse_reparameterize_torch(phys), dtype=torch.float32, device=dev)
areas = torch.ones(M, device=dev)                                            # 3DPS release: MC hemisphere weight = 1 per active point
def render(raw, detach=True):
    rr, ri = render_factorized(pos, nrm, areas, raw, rast, RZ.reparameterize_torch, detach_phase=detach, use_cuda_kernels=False)
    return torch.complex(rr, ri)                                             # (n_tx, n_rx, K) range profiles
def corr(a, b): return float(np.corrcoef(a - np.median(a), b - np.median(b))[0, 1])
rows = []
with torch.no_grad():
    for sh in (3e-5, 1e-4, 3e-4, 1e-3):
        for lc in (3e-3, 1e-2, 3e-2):
            rp = render(raw_for(5.24, 0.46, sh, lc, 0.5, 0.1)); prof = db(rp.abs().mean((0, 1)).cpu().numpy())
            rows.append(dict(sigma_h_mm=sh * 1e3, l_c_cm=lc * 1e2, corr=corr(g, prof[mask]), rmse_db=float(np.sqrt(np.mean(((g - np.median(g)) - (prof[mask] - np.median(prof[mask]))) ** 2)))))
            print(f"3DPS render, concrete, sigma_h {sh*1e3:.2f} mm, l_c {lc*1e2:.0f} cm: iter-0 pair-mean profile corr {rows[-1]['corr']:+.3f}, RMSE {rows[-1]['rmse_db']:.2f} dB")
best = max(rows, key=lambda r: r["corr"]); out["grid"] = rows; out["best"] = best; print("best:", best)
# region shares at the best setting
with torch.no_grad():
    raw = raw_for(5.24, 0.46, best["sigma_h_mm"] * 1e-3, best["l_c_cm"] * 1e-2, 0.5, 0.1)
    full = render(raw).abs().mean((0, 1)).cpu().numpy(); p = pos.cpu().numpy(); R = np.linalg.norm(p - np.asarray(sensor["pos"]), axis=1)
    reg = {"near < 1.5 m": R < 1.5, "floor z<0.15, R>1.5": (p[:, 2] < 0.15) & (R >= 1.5), "-X wall": p[:, 0] < -3.3, "beyond 3.71 m": R > VR.R_MAX_M}
    for nm, m in reg.items():
        mt = torch.as_tensor(m, device=dev); pr = torch.complex(*render_factorized(pos[mt], nrm[mt], areas[mt], raw[mt], rast, RZ.reparameterize_torch, detach_phase=True, use_cuda_kernels=False)).abs().mean((0, 1)).cpu().numpy()
        share = float((pr[mask] ** 2).sum() / (full[mask] ** 2).sum()); out.setdefault("region_share", {})[nm] = share; print(f"  region {nm:22s}: energy share {share:.3f}")
    out["profile_best"] = full.tolist(); out["gt_profile"] = gt.tolist()
    # per-pair image correlation (pair-blind ceiling 0.45)
    img = db(render(raw).abs().reshape(400, -1).cpu().numpy())[:, mask]; Gm = G[:, mask]
    out["image_corr_iter0"] = float(np.corrcoef((Gm - np.median(Gm)).ravel(), (img - np.median(img)).ravel())[0, 1]); print("pair-image corr at iter 0:", round(out["image_corr_iter0"], 3))
# gradient norms: materials (6 raw) and normals, detached vs attached phase
tgt = torch.as_tensor(gt, device=dev, dtype=torch.float64); mt = torch.as_tensor(mask, device=dev)
def l1log(rp):
    a = torch.log(rp.abs().mean((0, 1))[mt].double() + 1e-30); b = tgt[mt] / 20 * math.log(10); return (a - a.median() - (b - b.median())).abs().mean()
for detach in (True, False):
    raw = raw_for(5.24, 0.46, best["sigma_h_mm"] * 1e-3, best["l_c_cm"] * 1e-2, 0.5, 0.1).requires_grad_(True)
    n2 = nrm.clone().requires_grad_(True); p2 = pos.clone().requires_grad_(True)
    rr, ri = render_factorized(p2, n2, areas, raw, rast, RZ.reparameterize_torch, detach_phase=detach, use_cuda_kernels=False)
    L = l1log(torch.complex(rr, ri)); L.backward()
    gm = raw.grad.abs().median(0).values.cpu().numpy(); gn = float(n2.grad.norm(dim=1).median()); gx = float(p2.grad.norm(dim=1).median())
    d = dict(loss=float(L), raw_material_median=dict(zip(["eps_r", "eps_i", "sigma_h", "l_c", "tau", "thickness"], map(float, gm))), normals_median=gn, xyz_median=gx, xyz_over_eps_r=gx / max(float(gm[0]), 1e-30))
    out.setdefault("grads", {})[f"detach_phase={detach}"] = d
    print(f"detach_phase={detach}: loss {float(L):.3f}; |grad| medians raw materials {dict((k, f'{v:.2e}') for k, v in d['raw_material_median'].items())}, normals {gn:.2e}, xyz {gx:.2e}, xyz/eps_r {d['xyz_over_eps_r']:.1f}")
json.dump(out, open(os.path.join(DATA, "sfcw_3dps_render_check.json"), "w"), indent=1); print("saved")
