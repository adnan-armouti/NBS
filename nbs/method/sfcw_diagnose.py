"""SFCW diagnosis (2026-10-28): why is the pair-image correlation only 0.27?
Probes: (1) GT per-pair gain banding and what per-pair normalisation does to the correlation; (2) normal quality of the
scaffold; (3) gradient norms of the SFCW loss w.r.t. rho, abcd, normals, xyz; (4) coherent-vs-incoherent scaling of the
head's point sum with point density (the specular-plane test); (5) which scaffold regions make the rendered profile;
(6) element antenna pattern sensitivity. Run on the node: cd nbs && CUDA_VISIBLE_DEVICES=3 python -m method.sfcw_diagnose
"""
import json, os, sys, math
import numpy as np, torch
from preprocessing import octonet_config as C, octonet_scene as S
from method import sfcw_target as ST
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs

DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
dev = "cuda"
out = {}

def db(x): return 20 * np.log10(np.abs(x) + 1e-300)

# ---------------------------------------------------------------- GT side
z = np.load(os.path.join(DATA, "sfcw_target_S1", "target_pool5.npz"))
rp = z["rp_mean"]; mask = z["mask"].astype(bool); mask[:24] = False
G = db(rp)[:, mask]                                   # (400, 70) dB
pair_med = np.median(G, axis=1)                       # per-pair level
out["gt_pair_gain_spread_db"] = dict(std=float(pair_med.std()), p5_p95=[float(np.percentile(pair_med, 5)), float(np.percentile(pair_med, 95))])
M = pair_med.reshape(20, 20)
out["gt_pair_gain_by_tx_std_db"] = float(M.mean(1).std()); out["gt_pair_gain_by_rx_std_db"] = float(M.mean(0).std())
print("GT per-pair level: std %.2f dB (by-tx %.2f, by-rx %.2f)" % (pair_med.std(), M.mean(1).std(), M.mean(0).std()))
# frame-to-frame per-pair level stability (is the banding a fixed channel gain?)
from preprocessing import octonet_io as IO
V, t = IO.load_vayyar(S.modality_path(1, "vayyar", S.REC_SIT))
from mm3DPS.data.ra_utils import sfcw_range_compress
rpf = sfcw_range_compress(torch.from_numpy(V[::40]).to(torch.complex64)).numpy()   # (F,400,100)
lvl = np.median(db(rpf)[:, :, mask], axis=2)          # (F,400)
out["pair_level_frame_corr"] = float(np.corrcoef(lvl)[np.triu_indices(len(lvl), 1)].mean())
print("per-pair level pattern correlation across frames: %.3f (1 = fixed channel gain)" % out["pair_level_frame_corr"])

# ---------------------------------------------------------------- render side
cloud, material, sensor = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=30000)
xyz = cloud["xyz"]; nrm = cloud["normals"]
# 2. normals quality
n = nrm.cpu().numpy(); p = xyz.cpu().numpy()
floor = p[:, 2] < 0.12; wallX = p[:, 0] < -3.3
ang_floor = np.degrees(np.arccos(np.clip(np.abs(n[floor, 2]), 0, 1))); ang_wall = np.degrees(np.arccos(np.clip(np.abs(n[wallX, 0]), 0, 1)))
out["normals"] = dict(floor_points=int(floor.sum()), floor_angle_to_up_deg_median=float(np.median(ang_floor)), floor_within_20deg=float((ang_floor < 20).mean()),
                      wallX_points=int(wallX.sum()), wallX_angle_to_x_deg_median=float(np.median(ang_wall)), wallX_within_20deg=float((ang_wall < 20).mean()),
                      unit_norm_median=float(np.median(np.linalg.norm(n, axis=1))))
print("normals:", out["normals"])
# incidence stats seen by the radar
pos = torch.as_tensor(sensor["pos"], device=dev, dtype=torch.float64); rel = xyz - pos; R = rel.norm(dim=1); w_in = -rel / R[:, None]
cos_i = (w_in * nrm).sum(1).abs().cpu().numpy()
out["cos_incidence"] = dict(median=float(np.median(cos_i)), frac_below_0p3=float((cos_i < 0.3).mean()))
print("cos(incidence) median %.2f, frac < 0.3: %.2f" % (np.median(cos_i), (cos_i < 0.3).mean()))

# 3. gradient norms
tgt = {k: torch.as_tensor(z[k]) for k in z.files}; tgt["rp_mean"] = tgt["rp_mean"].to(torch.complex128).to(dev); tgt["mask"] = torch.as_tensor(mask).to(dev)
for k in ("rel", "xcorr", "snr_db"):
    if k in tgt: tgt[k] = tgt[k].double().to(dev)
cl = {"xyz": xyz.clone().requires_grad_(True), "normals": nrm.clone().requires_grad_(True)}
mat = {"abcd": material["abcd"].clone().requires_grad_(True), "rho": material["rho"].clone().requires_grad_(True)}
rp_r = OH.sfcw_head(cl, mat, sensor)
loss = ST.sfcw_static_loss(rp_r, tgt, kind="logmag", weight="xcorr")
loss = loss[0] if isinstance(loss, (tuple, list)) else loss
loss.backward()
def gstats(g, name):
    gn = g.norm(dim=-1) if g.dim() > 1 else g.abs()
    gn = gn.detach().cpu().numpy()
    d = dict(norm_total=float(np.linalg.norm(gn)), per_point_median=float(np.median(gn)), per_point_p95=float(np.percentile(gn, 95)), frac_zero=float((gn == 0).mean()))
    out.setdefault("grad", {})[name] = d; print(f"grad {name}: {d}")
    return gn
g_rho = gstats(mat["rho"].grad, "rho"); g_abcd = mat["abcd"].grad.detach().cpu().numpy()
for i, nm in enumerate("abcd"): gstats(mat["abcd"].grad[:, i], f"abcd[{nm}]")
g_n = gstats(cl["normals"].grad, "normals"); g_x = gstats(cl["xyz"].grad, "xyz")
# where do rho gradients live? by range and region
Rn = R.detach().cpu().numpy()
for lo, hi in ((0.9, 2.0), (2.0, 3.0), (3.0, 3.71), (3.71, 6.0), (6.0, 9.0)):
    m = (Rn >= lo) & (Rn < hi); print(f"  rho grad by range {lo}-{hi} m: n={m.sum()}, |g| median {np.median(g_rho[m]) if m.any() else float('nan'):.2e}")
out["loss_iter0"] = float(loss)

# 4. coherent vs incoherent: the -X wall alone, at two point densities
with torch.no_grad():
    wall = torch.as_tensor(wallX, device=dev)
    def prof(cl_, mat_): return OH.sfcw_head(cl_, mat_, sensor).abs().mean(0).cpu().numpy()
    sub = {"xyz": xyz[wall], "normals": nrm[wall]}; msub = {"abcd": material["abcd"][wall], "rho": material["rho"][wall]}
    p1 = prof(sub, msub)
    # double the density by jittering copies within the voxel (5 cm)
    jit = torch.randn_like(sub["xyz"]) * 0.015; sub2 = {"xyz": torch.cat([sub["xyz"], sub["xyz"] + jit]), "normals": torch.cat([sub["normals"]] * 2)}
    msub2 = {"abcd": torch.cat([msub["abcd"]] * 2), "rho": torch.cat([msub["rho"]] * 2)}
    p2 = prof(sub2, msub2)
    ratio = (p2[mask].sum() / p1[mask].sum())
    out["wall_density_test"] = dict(points=int(wall.sum().item()), amplitude_ratio_2x_points=float(ratio), note="2.0 = coherent (specular-like), 1.41 = incoherent (rough) sum")
    print("-X wall alone: doubling point density scales the summed |rp| by %.2f (2.0 coherent, 1.41 incoherent)" % ratio)
    # 5. region contributions to the rendered profile (uniform concrete)
    regions = {"floor z<0.15": p[:, 2] < 0.15, "ceiling z>2.3": p[:, 2] > 2.3, "wall -X (x<-3.3)": wallX, "near node (<1.5 m)": Rn < 1.5,
               "mid 1.5-3.7 m": (Rn >= 1.5) & (Rn < 3.71), "beyond 3.71 m (aliased)": Rn >= 3.71}
    full = prof(cloud, material); out["region_profiles_db"] = {"range_m": (np.arange(100) * 0.0375).tolist(), "full": db(full).tolist()}
    for nm, m in regions.items():
        mt = torch.as_tensor(m, device=dev)
        pr = prof({"xyz": xyz[mt], "normals": nrm[mt]}, {"abcd": material["abcd"][mt], "rho": material["rho"][mt]})
        out["region_profiles_db"][nm] = db(pr).tolist()
        print(f"  region {nm:26s}: n={int(m.sum()):6d}, share of |rp| energy in masked band {float((pr[mask]**2).sum() / (full[mask]**2).sum()):.3f}")
    # 6. element pattern sensitivity: cos^p weighting about the boresight in azimuth (H-plane 82 deg) and elevation (E-plane 180 deg)
    bore = torch.as_tensor(sensor["boresight"], device=dev, dtype=torch.float64); bore = bore / bore.norm()
    cosb = ((rel / R[:, None]) @ bore).clamp(min=1e-6)
    for pexp in (0.0, 1.0, 2.0, 4.0):
        w = cosb ** pexp
        pr = OH.sfcw_head(cloud, {"abcd": material["abcd"], "rho": material["rho"] * w}, sensor).abs().mean(0).cpu().numpy()
        g = db(rp).mean(0)[mask]; r_ = db(pr)[mask]; cc = np.corrcoef(g - np.median(g), r_ - np.median(r_))[0, 1]
        print(f"  element pattern cos^{pexp}: pair-mean profile corr vs GT = {cc:.3f}"); out.setdefault("pattern_corr", {})[str(pexp)] = float(cc)
    # per-pair normalisation effect on the image correlation (uniform concrete render)
    Rimg = db(OH.sfcw_head(cloud, material, sensor).cpu().numpy())[:, mask]
    def imgcorr(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    out["image_corr"] = dict(global_norm=imgcorr(G - np.median(G), Rimg - np.median(Rimg)),
                             per_pair_norm=imgcorr(G - np.median(G, 1, keepdims=True), Rimg - np.median(Rimg, 1, keepdims=True)),
                             per_bin_norm=imgcorr(G - np.median(G, 0, keepdims=True), Rimg - np.median(Rimg, 0, keepdims=True)),
                             gt_vs_its_pair_mean=imgcorr(G - np.median(G, 1, keepdims=True), np.repeat(np.median(G - np.median(G, 1, keepdims=True), 0, keepdims=True), 400, 0)))
    print("image corr (uniform concrete): global-norm %.3f, per-pair-norm %.3f, per-bin-norm %.3f; GT image vs its own pair-mean (ceiling for a pair-blind render) %.3f" % tuple(out["image_corr"].values()))
json.dump(out, open(os.path.join(DATA, "sfcw_diagnose.json"), "w"), indent=1)
print("saved", os.path.join(DATA, "sfcw_diagnose.json"))
