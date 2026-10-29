"""SFCW diagnosis part 4: (A) physical-optics coherent sum over DENSE planar patches (lambda/4 sampling) for the -X wall
at normal incidence (D~6.6 m) vs a floor patch at grazing incidence (R~1.2 m): what a correct specular model predicts
for the far/near amplitude ratio, versus the 5 cm point sum; (B) figure: GT pair-mean profile (pooled + the 5 single
recordings), the current head and the Jones render at iter 0, per-region curves.  Run on the node (GPU 3)."""
import json, os, sys, math
import numpy as np, torch
from preprocessing import octonet_config as C, vayyar_radar_config as VR
from method import sfcw_forward as SF
from method.freq_material import reflectivity
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "4DPS"))
from mm4DPS import octonet_heads as OH
from mm4DPS.octonet import render_inputs
DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT); dev = "cuda"
def db(x): return 20 * np.log10(np.abs(x) + 1e-300)
lam = 3e8 / 65e9; K = VR.N_FREQ; freqs = torch.as_tensor(SF.sfcw_freqs_hz(VR.F_START_HZ, VR.F_STOP_HZ, K), device=dev, dtype=torch.float64)
abcd = torch.tensor([[5.24, 0.0, 0.0462, 0.7822]], device=dev, dtype=torch.float64)      # ITU concrete
def po_patch(D, theta_deg, size_m, dx, label):
    """Coherent PO sum of a planar patch: centre at distance D, plane tilted so incidence = theta; samples every dx.
    Amplitude per element = Gamma(theta_local) * cos(theta_local) * dA / (lambda R^2) * exp(-j 2 k R)   (PO, monostatic).
    Returns the Hann range profile peak amplitude (units consistent across calls)."""
    n = int(size_m / dx); u = (torch.arange(n, device=dev, dtype=torch.float64) - n / 2) * dx
    U, Vv = torch.meshgrid(u, u, indexing="ij"); U = U.reshape(-1); Vv = Vv.reshape(-1)
    th = math.radians(theta_deg)
    # radar at origin looking along +x; patch centre at (D,0,0); patch normal rotated by theta about z (in the x-y plane)
    nrm = torch.tensor([-math.cos(th), math.sin(th), 0.0], device=dev, dtype=torch.float64)
    e1 = torch.tensor([math.sin(th), math.cos(th), 0.0], device=dev, dtype=torch.float64); e2 = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=torch.float64)
    P = torch.tensor([D, 0.0, 0.0], device=dev, dtype=torch.float64)[None, :] + U[:, None] * e1[None, :] + Vv[:, None] * e2[None, :]
    R = P.norm(dim=1); w = -P / R[:, None]; cos_i = (w @ nrm).abs().clamp(1e-3, 1)
    G = reflectivity(abcd.expand(len(R), 4), 65.0, cos_i).to(torch.complex128)
    amp = G * (cos_i * dx * dx / (lam * R ** 2)).to(torch.complex128)
    H = torch.stack([(amp * torch.exp(-1j * 4 * math.pi * f * R / 3e8)).sum() for f in freqs])
    rp = torch.fft.ifft(H * torch.hann_window(K, periodic=False, device=dev, dtype=torch.float64))
    pk = float(rp.abs().max()); print(f"PO {label:34s}: {n*n:8d} elems (dx {dx*1e3:.2f} mm): |rp| peak {pk:.3e}  ({db(pk):+.1f} dB)")
    return pk
with torch.no_grad():
    wall_dense = po_patch(6.6, 0.0, 1.0, lam / 4, "-X wall 1m^2, normal incidence")
    wall_coarse = po_patch(6.6, 0.0, 1.0, 0.05, "-X wall 1m^2, 5 cm samples (ours)")
    wall_dense_5deg = po_patch(6.6, 5.0, 1.0, lam / 4, "-X wall, 5 deg off-normal")
    floor_dense = po_patch(1.2, 65.0, 1.0, lam / 4, "floor 1m^2 at 1.2 m, 65 deg grazing")
    floor_coarse = po_patch(1.2, 65.0, 1.0, 0.05, "floor 1m^2, 5 cm samples (ours)")
    floor_dense_80 = po_patch(2.5, 80.0, 1.0, lam / 4, "floor 1m^2 at 2.5 m, 80 deg")
    res = dict(wall_dense=wall_dense, wall_coarse=wall_coarse, wall_5deg=wall_dense_5deg, floor_dense=floor_dense, floor_coarse=floor_coarse, floor_2p5_dense=floor_dense_80,
               ratio_wall_over_floor_db_dense=db(wall_dense) - db(floor_dense), ratio_wall_over_floor_db_coarse=db(wall_coarse) - db(floor_coarse))
    print("PO: wall/floor ratio dense %.1f dB vs 5 cm-sampled %.1f dB" % (res["ratio_wall_over_floor_db_dense"], res["ratio_wall_over_floor_db_coarse"]))
# B. figure
z = np.load(os.path.join(DATA, "sfcw_target_S1", "target_pool5.npz")); mask = z["mask"].astype(bool); mask[:24] = False
rng = np.arange(K) * 0.0375; gt = db(z["rp_mean"]).mean(0)
singles = []
for f in sorted(os.listdir(os.path.join(DATA, "sfcw_target_S1"))):
    if f.startswith("target_rec") and f.endswith(".npz") and "decoupled" not in f and "bh4" not in f:
        zz = np.load(os.path.join(DATA, "sfcw_target_S1", f)); singles.append((f, db(zz["rp_mean"]).mean(0)))
cloud, material, sensor = render_inputs(os.path.join(DATA, "4dps_scaffold_S1"), os.path.join(DATA, "calib_S1", "extrinsics.json"), device=dev, n_points=40000)
xyz = cloud["xyz"]; nrm = cloud["normals"]; pos = torch.as_tensor(sensor["pos"], device=dev, dtype=torch.float64); R = (xyz - pos).norm(dim=1); p = xyz.cpu().numpy(); Rn = R.cpu().numpy()
with torch.no_grad():
    head = db(OH.sfcw_head(cloud, material, sensor).abs().mean(0).cpu().numpy())
    regions = {"near < 1.5 m": Rn < 1.5, "floor z<0.15, R>1.5": (p[:, 2] < 0.15) & (Rn >= 1.5), "-X wall": p[:, 0] < -3.3, "beyond 3.71 m (all)": Rn > VR.R_MAX_M, "1.5-3.71 m, not floor": (Rn >= 1.5) & (Rn <= VR.R_MAX_M) & (p[:, 2] >= 0.15)}
    reg = {}
    for nm, m in regions.items():
        mt = torch.as_tensor(m, device=dev); reg[nm] = db(OH.sfcw_head({"xyz": xyz[mt], "normals": nrm[mt]}, {"abcd": material["abcd"][mt], "rho": material["rho"][mt]}, sensor).abs().mean(0).cpu().numpy())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(21, 6))
a = ax[0]; a.plot(rng[mask], gt[mask] - np.median(gt[mask]), "k", lw=2, label="GT pooled (5 rec.)")
for f, s in singles: a.plot(rng[mask], s[mask] - np.median(s[mask]), lw=0.8, alpha=0.6, label=f.replace("target_rec_", "").replace(".npz", ""))
a.axvline(6.6 - VR.R_MAX_M, color="r", ls="--", lw=1, label="-X wall (6.6 m) folded"); a.axvline(2.92, color="m", ls=":", lw=1, label="subject spot 2.92 m")
a.set_xlabel("range [m]"); a.set_ylabel("pair-mean |rp| dB rel. median"); a.set_title("GT: pooled vs single recordings (the far rise is in every recording)"); a.legend(fontsize=7); a.grid(alpha=.3)
a = ax[1]; a.plot(rng[mask], gt[mask] - np.median(gt[mask]), "k", lw=2, label="GT pooled"); a.plot(rng[mask], head[mask] - np.median(head[mask]), "C3", lw=1.5, label="current head, iter 0 (Fresnel isotropic, 1/R^4 point sum)")
for nm, r_ in reg.items(): a.plot(rng[mask], r_[mask] - np.median(head[mask]), lw=0.9, alpha=0.8, label=f"head: {nm} only")
a.set_xlabel("range [m]"); a.set_title("rendered profile and where it comes from (uniform concrete)"); a.legend(fontsize=7); a.grid(alpha=.3)
a = ax[2]; a.axis("off")
a.text(0, 1, "Physical-optics dense-patch test (1 m^2 patches, concrete, 65 GHz)\n\n" + "\n".join(f"{k:34s} {v:+8.1f} dB" if "ratio" in k else f"{k:34s} {db(v):+8.1f} dB" for k, v in res.items()) +
       "\n\nReading: a correctly sampled specular wall at 6.6 m (normal incidence) is STRONGER than a\n1 m^2 grazing floor patch at 1.2 m; the 5 cm-sampled point sum inverts this. The Fresnel-isotropic\nhead has no specular (Fresnel-zone) behaviour and no grazing fall-off, so it is bright near and dark far.",
       va="top", family="monospace", fontsize=9)
fig.suptitle("SFCW diagnosis: GT far rise = folded specular far wall; the point-sum head cannot produce it", fontsize=12); fig.tight_layout()
viz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", "render_vs_gt_v2"); fig.savefig(os.path.join(viz, "sfcw_diagnosis.png"), dpi=140)
json.dump(res, open(os.path.join(DATA, "sfcw_diagnose4.json"), "w"), indent=1); print("saved figure + json")
