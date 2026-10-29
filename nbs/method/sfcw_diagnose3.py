"""SFCW diagnosis part 3: does the 3DPS Jones BSDF (specular/rough lobes, cos projection) fix the iter-0 shape?
Per point: P_i = f_jones(wi=wo=w_radar) * cos_i * A / R^4  (monostatic radar equation with the BSDF as the scattering
coefficient), amplitude sqrt(P_i); optional element pattern cos^p about the boresight (H-plane 82 deg); incoherent
expected-power profile (sum P_i |PSF|^2) and coherent sum; pair image with the 39 per-Tx/per-Rx gains fitted from GT.
Run: cd nbs && CUDA_VISIBLE_DEVICES=3 python -m method.sfcw_diagnose3
"""
import json, os, sys, math, itertools
import numpy as np, torch
from preprocessing import octonet_config as C, octonet_scene as S, vayyar_radar_config as VR
from method import sfcw_forward as SF
from method.freq_material import eps_from_dispersion
from mm3DPS import bsdf_torch as B
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs
DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT); dev = "cuda"; out = {}
def db(x): return 20 * np.log10(np.abs(x) + 1e-300)
z = np.load(os.path.join(DATA, "sfcw_target_S1", "target_pool5.npz")); rp = z["rp_mean"]; mask = z["mask"].astype(bool); mask[:24] = False
G = db(rp); gt_prof = G.mean(0); lvl = np.median(G[:, mask], 1).reshape(20, 20)
gtx = lvl.mean(1) - lvl.mean(); grx = lvl.mean(0) - lvl.mean()                       # per-Tx / per-Rx gains (dB) from GT
cloud, material, sensor = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=40000)
xyz = cloud["xyz"]; nrm = cloud["normals"]; pos = torch.as_tensor(sensor["pos"], device=dev, dtype=torch.float64)
bore = torch.as_tensor(sensor["boresight"], device=dev, dtype=torch.float64); bore = bore / bore.norm()
rel = xyz - pos; R = rel.norm(dim=1); w = -rel / R[:, None]                              # surface -> radar
cos_i = (w * nrm).sum(1).abs().clamp(1e-3, 1.0); cosb = (-w @ bore).clamp(min=1e-6)
K = VR.N_FREQ; freqs = SF.sfcw_freqs_hz(VR.F_START_HZ, VR.F_STOP_HZ, K)
er, ei = eps_from_dispersion(material["abcd"], 65.0)
def jones_power(sigma_h_m, l_c_m, tau=0.5, thick=0.1):
    B.set_frequency(65.0e9)
    M = len(R); f = B.evaluate_bsdf_jones(cos_i.float(), w.float(), w.float(), nrm.float(), er.float(), ei.float(),
                                          torch.full((M,), sigma_h_m, device=dev), torch.full((M,), l_c_m, device=dev),
                                          torch.full((M,), tau, device=dev), torch.full((M,), thick, device=dev))
    return (f.double().clamp(min=0) * cos_i / R ** 4)                                    # power per point (area uniform)
def fresnel_power():
    refl = OH.point_reflectivity(cloud, material, 65.0, w)[:, 0].abs() ** 2
    return refl / R ** 4
k = torch.arange(K, device=dev, dtype=torch.float64); hann = torch.hann_window(K, periodic=False, device=dev, dtype=torch.float64)
def psf_pow(d):
    ph = torch.exp(-1j * 2 * math.pi * k[None, :] * d[:, None] / K) * hann[None, :]
    return (ph.sum(1).abs() / hann.sum()) ** 2
def incoherent(P, chunk=4096):
    grid = torch.arange(K, device=dev, dtype=torch.float64); prof = torch.zeros(K, device=dev, dtype=torch.float64)
    bins = (2 * R / (3e8 / VR.BANDWIDTH_HZ)) % K
    for lo in range(0, len(R), chunk):
        d = grid[None, :] - bins[lo:lo + chunk, None]; d = (d + K / 2) % K - K / 2
        prof = prof + (P[lo:lo + chunk, None] * psf_pow(d.reshape(-1)).reshape(d.shape)).sum(0)
    return prof.sqrt()
def corr(a, b): return float(np.corrcoef(a - np.median(a), b - np.median(b))[0, 1])
g = gt_prof[mask]
rows = []
with torch.no_grad():
    for name, P in [("fresnel-isotropic (current head)", fresnel_power())] + \
                   [(f"jones sigma_h={sh*1e3:.1f}mm l_c={lc*1e2:.0f}cm", jones_power(sh, lc)) for sh, lc in itertools.product((0.3e-3, 1e-3, 3e-3), (1e-2, 3e-2, 10e-2))]:
        for pexp in (0.0, 2.0):
            Pw = P * cosb ** pexp
            pr = db(incoherent(Pw).cpu().numpy())[mask]
            cc = corr(g, pr); rmse = float(np.sqrt(np.mean(((g - np.median(g)) - (pr - np.median(pr))) ** 2)))
            # region shares
            near = (R < 1.5).double(); far = (R > VR.R_MAX_M).double()
            e_near = float((incoherent(Pw * near)[torch.as_tensor(mask, device=dev)] ** 2).sum() / (incoherent(Pw)[torch.as_tensor(mask, device=dev)] ** 2).sum())
            e_far = float((incoherent(Pw * far)[torch.as_tensor(mask, device=dev)] ** 2).sum() / (incoherent(Pw)[torch.as_tensor(mask, device=dev)] ** 2).sum())
            rows.append(dict(model=name, pattern=f"cos^{pexp:.0f}", corr=cc, rmse_db=rmse, near_share=e_near, aliased_far_share=e_far))
            print(f"{name:40s} pattern cos^{pexp:.0f}: iter-0 profile corr {cc:+.3f}, RMSE {rmse:5.2f} dB, near(<1.5m) share {e_near:.2f}, aliased-far share {e_far:.2f}")
    # pair image with per-channel gains: coherent Jones head render at the best setting, gains applied
    best = max(rows, key=lambda r: r["corr"]); print("best:", best)
    out["rows"] = rows; out["best"] = best
    # coherent pair image under the current head + GT channel gains -> image corr (upper bound of the gain fix alone)
    img = db(OH.sfcw_head(cloud, material, sensor).cpu().numpy())[:, mask]
    gains = (gtx[:, None] + grx[None, :]).reshape(400, 1)
    def imgcorr(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    Gm = G[:, mask]
    out["image_corr"] = dict(no_gains=imgcorr(Gm - np.median(Gm), img - np.median(img)), with_gt_channel_gains=imgcorr(Gm - np.median(Gm), img + gains - np.median(img + gains)),
                             gt_vs_gains_only=imgcorr(Gm - np.median(Gm), np.repeat(gains, Gm.shape[1], 1) + np.median(Gm - np.median(Gm, 1, keepdims=True), 0, keepdims=True)))
    print("pair-image corr (current head, iter 0): no gains %.3f, + GT's 39 channel gains %.3f; GT vs (gains + GT pair-mean) %.3f" % tuple(out["image_corr"].values()))
json.dump(out, open(os.path.join(DATA, "sfcw_diagnose3.json"), "w"), indent=1); print("saved")
