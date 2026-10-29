"""Scaffold geometry for the OctoNet SFCW head (findings D16 normals, D7 occlusion).

* ``planar_normals_depth``: per-pixel normals from robust local plane fits on the FULL-RESOLUTION median depth image
  (windowed weighted PCA in camera coordinates; window pixels whose depth differs from the centre by more than the
  quantisation-aware tolerance are excluded), oriented toward the camera. Replaces PCA on the 5 cm voxel cloud, whose
  normals on the 8-bit-depth staircase of the far wall were 33 deg off.
* ``backproject_with_normals`` / ``voxel_average``: carry the per-pixel normal through back-projection and voxelise by
  averaging positions and normals per voxel.
* ``visibility_mask``: depth-buffer visibility from the radar on a spherical (azimuth, elevation) grid: per cell only
  the nearest point (within a tolerance) is visible; points behind the board (back hemisphere) are occluded. This is the
  point-scaffold stand-in for 3DPS's mesh ray-cast ``shadow_mask``.
* ``python -m mm4DPS.octonet_geometry check --v1 <dir> --v2 <dir>``: iteration-0 comparison of the uniform-concrete
  render through ``octonet_heads.sfcw_head`` for v1 / v2 normals / v2 normals + shadow mask, with the figure
  ``viz/render_vs_gt_v2/scaffold_v2_normals_occlusion.png``.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

from preprocessing import octonet_config as C, octonet_scene as S

DEPTH_STEP_M = 10.0 / 255.0        # 8-bit depth quantisation before the per-camera scale


def planar_normals_depth(depth_m: np.ndarray, K: S.Intrinsics, r: int = 7, step_m: float = DEPTH_STEP_M,
                         rel_tol: float = 0.05, min_pts: int = 12, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel plane normals (H,W,3) in the CAMERA frame (x right, y down, z forward), oriented toward the camera,
    and a validity mask (H,W). Windowed weighted PCA with window (2r+1)^2; a window pixel is used only if it is valid
    and |z - z_centre| < max(3*step_m, rel_tol*z_centre)."""
    import torch
    H, W = depth_m.shape
    z = torch.as_tensor(depth_m, dtype=torch.float32, device=device)
    valid = z > 0.3
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    x = (u - K.cx) / K.fx * z; y = (v - K.cy) / K.fy * z
    P = torch.stack([x, y, z], 0)[None]                                         # (1,3,H,W)
    k = 2 * r + 1
    Pu = torch.nn.functional.unfold(P, k, padding=r).reshape(3, k * k, H, W)     # (3,k^2,H,W)
    Vu = torch.nn.functional.unfold(valid[None, None].float(), k, padding=r).reshape(k * k, H, W)
    tol = torch.maximum(torch.full_like(z, 3 * step_m), rel_tol * z)
    w = Vu * ((Pu[2] - z[None]).abs() < tol[None]).float() * valid[None].float()   # (k^2,H,W)
    n_pts = w.sum(0)
    ok = (n_pts >= min_pts) & valid
    wsum = n_pts.clamp(min=1e-6)
    mean = (Pu * w[None]).sum(1) / wsum[None]                                    # (3,H,W)
    D = Pu - mean[:, None]                                                       # (3,k^2,H,W)
    cov = torch.einsum("ikhw,jkhw->hwij", D * w[None], D) / wsum[:, :, None, None]   # (H,W,3,3)
    cov = torch.nan_to_num(cov.reshape(-1, 3, 3).double())
    cov = cov + 1e-12 * torch.eye(3, device=device, dtype=cov.dtype)[None]
    evals, evecs = torch.linalg.eigh(cov.cpu())                                 # CPU: the cuSOLVER batched syev rejects these batches
    n = evecs[:, :, 0].to(device).reshape(H, W, 3).float()                       # ascending -> smallest = plane normal
    to_cam = -torch.stack([x, y, z], -1); flip = (n * to_cam).sum(-1, keepdim=True) < 0
    n = torch.where(flip, -n, n)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return n.cpu().numpy(), ok.cpu().numpy()


def backproject_with_normals(depth_m: np.ndarray, normals: np.ndarray, ok: np.ndarray, K: S.Intrinsics, stride: int = 2,
                             z_min: float = 0.3, z_max: float = 9.5) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    z = depth_m[::stride, ::stride]; nm = normals[::stride, ::stride]; okk = ok[::stride, ::stride]
    m = (z > z_min) & (z < z_max) & okk
    x = (u - K.cx) / K.fx * z; y = (v - K.cy) / K.fy * z
    return np.stack([x[m], y[m], z[m]], 1), nm[m]


def voxel_average(pts: np.ndarray, normals: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Centroid position and mean (renormalised) normal per occupied voxel."""
    key = np.floor((pts - pts.min(0)) / voxel_m).astype(np.int64)
    kid = key[:, 0] + (key[:, 1] << 20) + (key[:, 2] << 40)
    uniq, inv = np.unique(kid, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    p = np.stack([np.bincount(inv, pts[:, i]) for i in range(3)], 1) / cnt[:, None]
    n = np.stack([np.bincount(inv, normals[:, i]) for i in range(3)], 1)
    n = n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    return p, n


def visibility_mask(xyz: np.ndarray, sensor_pos: np.ndarray, boresight: np.ndarray, cell_deg: float = 0.5,
                    tol_m: float = 0.10, back_hemisphere: bool = True) -> np.ndarray:
    """True = visible from the radar: nearest point per (azimuth, elevation) cell within ``tol_m``; points behind the
    board (direction . boresight <= 0) are occluded when ``back_hemisphere``."""
    rel = xyz - sensor_pos[None, :]; R = np.linalg.norm(rel, axis=1); d = rel / R[:, None].clip(1e-9)
    b = boresight / np.linalg.norm(boresight); up = np.array([0.0, 0.0, 1.0])
    right = np.cross(b, up); right /= np.linalg.norm(right); upv = np.cross(right, b)
    az = np.degrees(np.arctan2(d @ right, d @ b)); el = np.degrees(np.arcsin(np.clip(d @ upv, -1, 1)))
    ia = np.floor((az + 180.0) / cell_deg).astype(np.int64); ie = np.floor((el + 90.0) / cell_deg).astype(np.int64)
    cell = ia * 100000 + ie
    order = np.lexsort((R, cell)); cs = cell[order]
    first = np.r_[True, cs[1:] != cs[:-1]]
    rmin = np.empty(len(xyz)); rmin[order] = np.repeat(R[order][first], np.diff(np.r_[np.nonzero(first)[0], len(cs)]))
    vis = R <= rmin + tol_m
    if back_hemisphere:
        vis &= (d @ b) > 0
    return vis


# ----------------------------------------------------------------------------- check
def _profile(cloud, material, sensor):
    import torch
    from mm4DPS import octonet_heads as OH
    with torch.no_grad():
        return OH.sfcw_head(cloud, material, sensor).abs().mean(0).cpu().numpy()


def check(args) -> dict:
    import torch
    from mm4DPS.octonet import render_inputs
    data = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT); dev = "cuda"
    z = np.load(os.path.join(data, "sfcw_target_S1", "target_pool5.npz")); mask = z["mask"].astype(bool); mask[:24] = False
    gt = 20 * np.log10(np.abs(z["rp_mean"]) + 1e-300).mean(0); rng_m = np.arange(len(gt)) * 0.0375
    ext = os.path.join(data, "calib_S1", "extrinsics.json")
    def corr(a, b): return float(np.corrcoef(a - np.median(a), b - np.median(b))[0, 1])
    def rmse(a, b): return float(np.sqrt(np.mean(((a - np.median(a)) - (b - np.median(b))) ** 2)))
    res = {}; profs = {}
    for name, d, use_mask in (("v1 (PCA normals)", args.v1, False), ("v2 planar normals", args.v2, False), ("v2 planar + shadow mask", args.v2, True)):
        cloud, material, sensor = render_inputs(d, ext, device=dev, n_points=args.points, seed=0, use_shadow_mask=use_mask)
        pr = 20 * np.log10(_profile(cloud, material, sensor) + 1e-300)
        res[name] = dict(corr=corr(gt[mask], pr[mask]), rmse_db=rmse(gt[mask], pr[mask]), points=int(len(cloud["xyz"])),
                         visible=(float(cloud["shadow_mask"].float().mean()) if "shadow_mask" in cloud else 1.0)); profs[name] = pr
        print(f"{name:28s}: iter-0 pair-mean profile corr {res[name]['corr']:+.3f}, RMSE {res[name]['rmse_db']:.2f} dB (visible {res[name]['visible']:.2f})")
    # normal-quality statistics v1 vs v2
    def nstats(d):
        pcl = np.load(os.path.join(d, "pcl.npy")); p, n = pcl[:, :3], pcl[:, 3:6]
        wall = p[:, 0] < -3.3; floor = p[:, 2] < 0.12
        aw = np.degrees(np.arccos(np.clip(np.abs(n[wall, 0]), 0, 1))); af = np.degrees(np.arccos(np.clip(np.abs(n[floor, 2]), 0, 1)))
        return dict(wall_n=int(wall.sum()), wall_med=float(np.median(aw)), wall_10=float((aw < 10).mean()), wall_20=float((aw < 20).mean()),
                    floor_n=int(floor.sum()), floor_med=float(np.median(af)), floor_10=float((af < 10).mean()), floor_20=float((af < 20).mean()), aw=aw, af=af)
    s1, s2 = nstats(args.v1), nstats(args.v2)
    for nm, st in (("v1", s1), ("v2", s2)):
        print(f"{nm} normals: -X wall median {st['wall_med']:.1f} deg (within 10/20 deg: {st['wall_10']:.2f}/{st['wall_20']:.2f}, n={st['wall_n']}); floor median {st['floor_med']:.1f} deg ({st['floor_10']:.2f}/{st['floor_20']:.2f})")
    sm = np.load(os.path.join(args.v2, "shadow_mask.npy")); pcl2 = np.load(os.path.join(args.v2, "pcl.npy"))
    meta = json.load(open(os.path.join(args.v2, "scaffold_metadata.json")))
    out = dict(iter0=res, normals={"v1": {k: v for k, v in s1.items() if k not in ("aw", "af")}, "v2": {k: v for k, v in s2.items() if k not in ("aw", "af")}},
               shadow=meta.get("shadow_mask", {}))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(26, 6))
    a = ax[0]; a.hist(s1["aw"], bins=45, range=(0, 90), alpha=0.6, label=f"v1 PCA: median {s1['wall_med']:.1f} deg"); a.hist(s2["aw"], bins=45, range=(0, 90), alpha=0.6, label=f"v2 planar: median {s2['wall_med']:.1f} deg")
    a.set_xlabel("-X wall normal angle to +-X [deg]"); a.set_title("wall normals (x < -3.3 m)"); a.legend(); a.grid(alpha=.3)
    a = ax[1]; a.hist(s1["af"], bins=45, range=(0, 90), alpha=0.6, label=f"v1: median {s1['floor_med']:.1f} deg"); a.hist(s2["af"], bins=45, range=(0, 90), alpha=0.6, label=f"v2: median {s2['floor_med']:.1f} deg")
    a.set_xlabel("floor normal angle to +-Z [deg]"); a.set_title("floor normals (z < 0.12 m)"); a.legend(); a.grid(alpha=.3)
    a = ax[2]; p = pcl2[:, :3]; a.scatter(p[~sm, 0], p[~sm, 1], s=0.3, c="0.75", label=f"occluded ({(~sm).mean():.0%})"); a.scatter(p[sm, 0], p[sm, 1], s=0.3, c="tab:red", label=f"visible from the Vayyar ({sm.mean():.0%})")
    vp = np.asarray(meta["vayyar_pose"]["t_w"]); a.plot(vp[0], vp[1], "k^", ms=10); a.set_aspect("equal"); a.set_xlabel("X [m]"); a.set_ylabel("Y [m]"); a.set_title(f"shadow mask (depth-buffer visibility, {meta.get('shadow_mask', {}).get('cell_deg', '?')} deg cells, tol {meta.get('shadow_mask', {}).get('tol_m', '?')} m, back hemisphere off)"); a.legend(markerscale=10, fontsize=8); a.grid(alpha=.3)
    a = ax[3]; a.plot(rng_m[mask], gt[mask] - np.median(gt[mask]), "k", lw=2, label="GT pooled")
    for nm, pr in profs.items(): a.plot(rng_m[mask], pr[mask] - np.median(pr[mask]), lw=1.2, label=f"{nm}: corr {res[nm]['corr']:+.3f}, RMSE {res[nm]['rmse_db']:.2f} dB")
    a.set_xlabel("range [m]"); a.set_ylabel("pair-mean |rp| dB rel. median"); a.set_title("iteration 0 (uniform concrete) through the 3DPS head"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("Scaffold v2: planar-fit normals on the full-resolution depth (D16) and radar visibility mask (D7)", fontsize=12); fig.tight_layout()
    viz = Path(__file__).resolve().parent.parent.parent / "viz" / "render_vs_gt_v2"; viz.mkdir(parents=True, exist_ok=True)
    fig.savefig(viz / "scaffold_v2_normals_occlusion.png", dpi=140)
    with open(viz / "captions.txt", "a") as fh:
        fh.write("scaffold_v2_normals_occlusion.png: scaffold v2 = planar-fit normals on the full-resolution median depth (wall median %.1f -> %.1f deg) + depth-buffer visibility mask from the Vayyar (%.0f%% visible); iteration-0 uniform-concrete profile corr v1 %+.3f -> v2 %+.3f -> v2+mask %+.3f (RMSE %.2f / %.2f / %.2f dB).\n"
                 % (s1["wall_med"], s2["wall_med"], 100 * sm.mean(), res["v1 (PCA normals)"]["corr"], res["v2 planar normals"]["corr"], res["v2 planar + shadow mask"]["corr"],
                    res["v1 (PCA normals)"]["rmse_db"], res["v2 planar normals"]["rmse_db"], res["v2 planar + shadow mask"]["rmse_db"]))
    json.dump(out, open(os.path.join(data, "scaffold_v2_check.json"), "w"), indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--v1", required=True); c.add_argument("--v2", required=True); c.add_argument("--points", type=int, default=40000)
    pr = sub.add_parser("probe"); pr.add_argument("--v2", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        check(a)
    elif a.cmd == "probe":
        probe(a)



def probe(args):
    """Why do planar normals score worse than PCA on the wall? Per-camera normal errors vs window radius, evaluated on
    the back-projected points of each camera's median depth; and what the shadow mask removes."""
    import torch
    from mm4DPS.octonet import median_depth, OctoNetSequence
    K = S.d455_nominal_intrinsics(); rec = S.REC_WALK
    for n in (1, 3, 5):
        med, _ = median_depth(n, rec, 30); R, t, _ = S.camera_pose(n)
        for r in (3, 7, 12):
            nmap, ok = planar_normals_depth(med, K, r=r, step_m=10.0 / 255.0 * S.depth_scale(n), device="cuda")
            pts, nm = backproject_with_normals(med, nmap, ok, K, stride=2)
            w = pts @ R.T + t; nw = nm @ R.T
            wall = (w[:, 0] < -3.3) & (np.abs(w[:, 1]) < 1.5) & (w[:, 2] > 0.3) & (w[:, 2] < 2.3); floor = (w[:, 2] < 0.12) & (w[:, 2] > -0.1)
            aw = np.degrees(np.arccos(np.clip(np.abs(nw[wall, 0]), 0, 1))) if wall.any() else np.array([np.nan])
            af = np.degrees(np.arccos(np.clip(np.abs(nw[floor, 2]), 0, 1)))
            print(f"node_{n} r={r:2d}: wall(open area) n={wall.sum():6d} median {np.nanmedian(aw):5.1f} deg (<20: {np.nanmean(aw < 20):.2f}); floor n={floor.sum():6d} median {np.median(af):4.1f} deg (<10: {np.mean(af < 10):.2f})")
        # PCA on the voxel cloud of this camera (v1 route) for the same metric
        from mm4DPS.reconstruction import voxel_downsample_torch, pca_normals_torch
        pts_cam = S.backproject(med, stride=2); w1 = pts_cam @ R.T + t
        cl = voxel_downsample_torch(torch.as_tensor(w1, dtype=torch.float64, device="cuda"), 0.05)
        nn = pca_normals_torch(cl, 0.2, 40, torch.as_tensor(t, dtype=torch.float64, device="cuda")).cpu().numpy(); cl = cl.cpu().numpy()
        wall = (cl[:, 0] < -3.3) & (np.abs(cl[:, 1]) < 1.5) & (cl[:, 2] > 0.3) & (cl[:, 2] < 2.3); floor = (cl[:, 2] < 0.12) & (cl[:, 2] > -0.1)
        aw = np.degrees(np.arccos(np.clip(np.abs(nn[wall, 0]), 0, 1))) if wall.any() else np.array([np.nan]); af = np.degrees(np.arccos(np.clip(np.abs(nn[floor, 2]), 0, 1)))
        print(f"node_{n} PCA (v1): wall n={wall.sum():6d} median {np.nanmedian(aw):5.1f} deg (<20: {np.nanmean(aw < 20):.2f}); floor n={floor.sum():6d} median {np.median(af):4.1f} deg (<10: {np.mean(af < 10):.2f})")
    # shadow mask: what is removed, and sensitivity
    seq = OctoNetSequence(rec); R_v, t_v = seq.sensor_pose("vayyar"); pcl = np.load(os.path.join(args.v2, "pcl.npy")); p = pcl[:, :3]
    Rr = np.linalg.norm(p - t_v, axis=1)
    for cell, tol in ((0.5, 0.10), (1.0, 0.10), (0.5, 0.25), (0.25, 0.10)):
        vis = visibility_mask(p, np.asarray(t_v), np.asarray(R_v)[:, 2], cell_deg=cell, tol_m=tol)
        regs = {"near<1.5m": Rr < 1.5, "floor z<0.15": p[:, 2] < 0.15, "-X wall": p[:, 0] < -3.3, "beyond 3.71": Rr > 3.71, "behind board": ((p - t_v) @ np.asarray(R_v)[:, 2]) <= 0}
        print(f"shadow cell {cell} deg tol {tol} m: visible {vis.mean():.2f}; visible fraction per region: " + ", ".join(f"{k} {vis[m].mean():.2f}" for k, m in regs.items()))

if __name__ == "__main__":
    main()
