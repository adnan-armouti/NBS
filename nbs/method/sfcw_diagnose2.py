"""SFCW diagnosis part 2: (A) per-Tx/per-Rx additive gain decomposition of the GT pair levels; (B) is the GT rise beyond
2.6 m the aliased far wall? (fold the scaffold's far points into the 3.71 m window); (C) an INCOHERENT (speckle-free)
objective: expected power per bin = sum_i |a_i|^2 |PSF(r-R_i)|^2 -- its iter-0 correlation with the (smoothed) GT and the
material-vs-position gradient ratio under it, versus the coherent head. (D) specular-plane check: rendered return of the
-X wall vs range scaling.  Run: cd nbs && CUDA_VISIBLE_DEVICES=3 python -m method.sfcw_diagnose2
"""
import json, os, sys, math
import numpy as np, torch
from preprocessing import octonet_config as C, octonet_scene as S, vayyar_radar_config as VR
from method import sfcw_target as ST, sfcw_forward as SF
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs
DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT); dev = "cuda"; out = {}
def db(x): return 20 * np.log10(np.abs(x) + 1e-300)
z = np.load(os.path.join(DATA, "sfcw_target_S1", "target_pool5.npz")); rp = z["rp_mean"]; mask = z["mask"].astype(bool); mask[:24] = False
G = db(rp)[:, mask]; lvl = np.median(G, 1).reshape(20, 20)
# A. additive tx+rx decomposition
tx = lvl.mean(1, keepdims=True) - lvl.mean(); rx = lvl.mean(0, keepdims=True) - lvl.mean()
resid = lvl - lvl.mean() - tx - rx
out["gain_decomp"] = dict(total_std_db=float(lvl.std()), tx_std=float(tx.std()), rx_std=float(rx.std()), residual_std_db=float(resid.std()),
                          explained=float(1 - resid.var() / lvl.var()))
print("GT pair level: total std %.2f dB; tx+rx additive explains %.0f%% (residual %.2f dB)" % (lvl.std(), 100 * out["gain_decomp"]["explained"], resid.std()))
# B. folded far-wall ranges
cloud, material, sensor = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=40000)
xyz = cloud["xyz"]; nrm = cloud["normals"]; pos = torch.as_tensor(sensor["pos"], device=dev, dtype=torch.float64)
R = (xyz - pos).norm(dim=1).cpu().numpy(); p = xyz.cpu().numpy()
Rmax = VR.R_MAX_M; folded = np.mod(R, Rmax)
far = R > Rmax
h, e = np.histogram(folded[far], bins=np.arange(0, Rmax + 0.0375, 0.0375 * 4)); ctr = 0.5 * (e[1:] + e[:-1])
gt_prof = db(rp).mean(0); rng = np.arange(100) * 0.0375
out["folded_far_hist"] = dict(range_m=ctr.tolist(), count=h.tolist())
# correlate the folded-far-point density with the GT profile in 2.5-3.5 m
sel = (ctr > 2.5) & (ctr < 3.5); gsel = np.interp(ctr[sel], rng, gt_prof)
out["folded_vs_gt_corr_2p5_3p5"] = float(np.corrcoef(h[sel], gsel)[0, 1]) if sel.sum() > 3 else None
print("far points (>3.71 m): %d of %d; folded-density vs GT profile corr in 2.5-3.5 m: %s" % (far.sum(), len(R), out["folded_vs_gt_corr_2p5_3p5"]))
wall_far = far & (p[:, 0] < -3.3); print("  -X wall points fold to %.2f-%.2f m (median %.2f)" % (folded[wall_far].min(), folded[wall_far].max(), np.median(folded[wall_far])))
# C. incoherent objective
freqs = SF.sfcw_freqs_hz(VR.F_START_HZ, VR.F_STOP_HZ, VR.N_FREQ); K = len(freqs)
def psf_power(r_bins):                      # |Hann-windowed IFFT of a unit tone| at fractional bin offset (K-periodic)
    k = torch.arange(K, device=dev, dtype=torch.float64); w = torch.hann_window(K, periodic=False, device=dev, dtype=torch.float64)
    # PSF(d) = |sum_k w_k exp(-j 2 pi k d / K)| / sum w
    d = r_bins[:, None]; ph = torch.exp(-1j * 2 * math.pi * k[None, :] * d / K) * w[None, :]
    return (ph.sum(1).abs() / w.sum()) ** 2
def incoherent_profile(cl, mat, chunk=4096):
    rel = cl["xyz"] - pos; Rr = rel.norm(dim=1); w_in = -rel / Rr[:, None]
    refl = OH.point_reflectivity(cl, mat, 65.0, w_in)[:, 0].abs() ** 2 / Rr ** 4          # power per point (monostatic centre)
    grid = torch.arange(K, device=dev, dtype=torch.float64); prof = torch.zeros(K, device=dev, dtype=torch.float64)
    bins = (2 * Rr / (3e8 / VR.BANDWIDTH_HZ)) % K                                         # fractional bin of each point (round-trip / c/B)
    for lo in range(0, len(Rr), chunk):
        d = grid[None, :] - bins[lo:lo + chunk, None]; d = (d + K / 2) % K - K / 2
        prof = prof + (refl[lo:lo + chunk, None] * psf_power(d.reshape(-1)).reshape(d.shape)).sum(0)
    return prof.sqrt()                                                                     # amplitude-equivalent
with torch.no_grad():
    inc = incoherent_profile(cloud, material); coh = OH.sfcw_head(cloud, material, sensor).abs().mean(0)
    g = gt_prof[mask]; gs = np.convolve(gt_prof, np.ones(3) / 3, mode="same")[mask]
    for nm, pr in (("coherent head (pair-mean)", coh), ("incoherent expected-power", inc)):
        r_ = db(pr.cpu().numpy())[mask]
        out.setdefault("iter0_profile_corr", {})[nm] = dict(vs_gt=float(np.corrcoef(g - np.median(g), r_ - np.median(r_))[0, 1]), vs_gt_smoothed3=float(np.corrcoef(gs - np.median(gs), r_ - np.median(r_))[0, 1]))
        print(f"iter-0 profile corr {nm}: vs GT {out['iter0_profile_corr'][nm]['vs_gt']:.3f}, vs 3-bin-smoothed GT {out['iter0_profile_corr'][nm]['vs_gt_smoothed3']:.3f}")
# gradient ratio material vs position under both objectives (L1 on log profile vs GT pair-mean)
tgt = torch.as_tensor(gt_prof, device=dev, dtype=torch.float64); m_t = torch.as_tensor(mask, device=dev)
def l1log(pr): a = torch.log(pr[m_t] + 1e-30); b = tgt[m_t] / 20 * math.log(10); return (a - a.median() - (b - b.median())).abs().mean()
for nm, fn in (("coherent", lambda cl, mt: OH.sfcw_head(cl, mt, sensor).abs().mean(0)), ("incoherent", incoherent_profile)):
    cl = {"xyz": xyz.clone().requires_grad_(True), "normals": nrm.clone().requires_grad_(True)}
    mt = {"abcd": material["abcd"].clone().requires_grad_(True), "rho": material["rho"].clone().requires_grad_(True)}
    L = l1log(fn(cl, mt)); L.backward()
    gr = mt["rho"].grad.abs().median().item(); gx = cl["xyz"].grad.norm(dim=1).median().item(); gn = cl["normals"].grad.norm(dim=1).median().item()
    out.setdefault("grad_ratio", {})[nm] = dict(loss=float(L), rho_median=gr, xyz_median=gx, normals_median=gn, xyz_over_rho=gx / gr)
    print(f"{nm}: loss {float(L):.3f}; median |grad| rho {gr:.2e}, xyz {gx:.2e}, normals {gn:.2e}; xyz/rho = {gx/gr:.0f}")
# D. specular plane check: the -X wall's rendered return vs an ideal mirror (image-antenna) scaling
with torch.no_grad():
    wall = torch.as_tensor((p[:, 0] < -3.3) & (np.abs(p[:, 1]) < 2.0) & (p[:, 2] > 0.3) & (p[:, 2] < 2.3), device=dev)
    Rw = (xyz[wall] - pos).norm(dim=1); n_w = int(wall.sum())
    coh_w = OH.sfcw_head({"xyz": xyz[wall], "normals": nrm[wall]}, {"abcd": material["abcd"][wall], "rho": material["rho"][wall]}, sensor).abs().mean(0)
    peak_coh = float(coh_w.max()); gamma = OH.point_reflectivity({"xyz": xyz[wall], "normals": nrm[wall]}, {"abcd": material["abcd"][wall], "rho": material["rho"][wall]}, 65.0, None)[:, 0].abs().mean().item()
    # ideal specular plane at distance D: amplitude ~ |Gamma| / (2D) (image antenna at 2D, one-way field), versus the point sum ~ N |Gamma| / D^2 (incoherent)
    D = float(Rw.median()); spec = gamma / (2 * D); inco = math.sqrt(n_w) * gamma / D ** 2
    # compare to the floor patch at 1.2 m as a reference scale in the same render
    out["wall_specular_check"] = dict(wall_points=n_w, D_m=D, rendered_peak=peak_coh, ideal_specular_amp=spec, incoherent_amp=inco, note="units are the head's; compare the RATIOS")
    print(f"-X wall: {n_w} pts at D={D:.2f} m: rendered peak {peak_coh:.3e}; ideal mirror-plane {spec:.3e}; incoherent sqrt(N)/D^2 {inco:.3e}")
json.dump(out, open(os.path.join(DATA, "sfcw_diagnose2.json"), "w"), indent=1); print("saved")
