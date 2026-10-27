"""Stage 1 driver: XRF55 clip -> metric init point cloud for 3DPS.

Pipeline (one clip):
  1. extract a representative RGB frame from the .mkv (ffmpeg)
  2. MiDaS monocular relative depth
  3. unproject to a camera-frame point cloud
  4. measured radar RA = mean over frames of mmWave[:, :, 64:128]
  5. align: solve metric depth-scale + camera->radar pose by RA correlation
  6. build the metric radar-frame point cloud, estimate normals
  7. save: scene/pcl.npy (N,6 xyz+normals), alignment.json, debug overlay PNG

Run in an env with torch+timm+imageio (e.g. ~/.conda/envs/mmir):
  python -m preprocessing.run_preprocess --scene Scene1 --index 0
Use --dry-run to skip MiDaS and substitute a synthetic depth (geometry smoke test).
"""
import os, sys, json, argparse
import numpy as np

# allow running as a script (python preprocessing/run_preprocess.py) or as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO          # noqa: E402
from preprocessing import xrf55_config as C        # noqa: E402
from preprocessing import ra_geometry as G         # noqa: E402
from preprocessing import align as A               # noqa: E402


def _save_overlay(out_png, radar_ra, pseudo_ra, corr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    axs[0].imshow(radar_ra, aspect="auto", origin="lower"); axs[0].set_title("measured radar RA")
    axs[1].imshow(pseudo_ra, aspect="auto", origin="lower"); axs[1].set_title("pseudo RA (depth)")
    ov = np.zeros((*radar_ra.shape, 3)); ov[..., 0] = radar_ra; ov[..., 1] = pseudo_ra
    axs[2].imshow(ov, aspect="auto", origin="lower"); axs[2].set_title(f"overlay (R=radar,G=pseudo) corr={corr:.3f}")
    for ax in axs: ax.set_xlabel("azimuth bin"); ax.set_ylabel("range bin")
    fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)


def _synthetic_depth(h=C.IMG_H, w=C.IMG_W):
    """A simple wall+floor relative-depth for the dry-run geometry smoke test."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = 2.0 + 3.0 * (yy / h)                 # nearer at top, farther at bottom
    d += 0.5 * np.sin(xx / w * 6.28)         # some lateral structure
    return d.astype(np.float32)


def process(scene, index, dry_run=False, stride=4, backend="dav2_metric",
            reduce=C.RADAR_REDUCE, az_sin=C.AZ_SIN_LIMIT, blur=C.BLUR_RANGE_AZ):
    mmw_paths = IO.list_mmwave(scene)
    if not mmw_paths:
        raise SystemExit(f"no mmWave files for {scene}")
    mmw = mmw_paths[index]
    key = IO.parse_key(mmw)
    clip = f"{key['subject']}_{key['action']}_{key['trial']}"
    out_dir = os.path.join(C.NBS_DATA_OUT, clip)
    os.makedirs(os.path.join(out_dir, "scene"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "views"), exist_ok=True)
    print(f"[clip] {clip}  scene={scene}")

    # 4. measured radar RA (range x azimuth)
    radar_ra = IO.radar_ra(mmw, reduce=reduce)
    print(f"[radar] RA {radar_ra.shape} reduce={reduce}")

    # 1-2. RGB frame + depth (cached per backend as views/depth_<backend>.npy)
    K = C.camera_K()
    depth_cache = os.path.join(out_dir, "views", f"depth_{backend}.npy")
    if dry_run:
        depth = _synthetic_depth()
        print("[depth] DRY RUN synthetic depth")
    elif os.path.exists(depth_cache):
        depth = np.load(depth_cache)
        print(f"[depth] cached {backend} {depth.shape}")
    else:
        from preprocessing import mono_depth as MD
        mkv = IO.video_path_for(mmw, scene)
        if not os.path.exists(mkv):
            raise SystemExit(f"video missing: {mkv}")
        pngs = IO.extract_frames(mkv, os.path.join(out_dir, "views"), n_frames=1)
        rgb = MD.read_rgb(pngs[0])
        depth, is_metric = MD.get_depth(rgb, backend=backend)
        np.save(depth_cache, depth)
        print(f"[depth] {backend} {depth.shape} metric={is_metric} on {os.path.basename(pngs[0])} (cached)")

    # 3. unproject -> affine bases
    pts0, rays, intens = G.unproject_bases(depth, K, stride=stride)
    print(f"[unproject] {pts0.shape[0]} points (stride {stride})")

    # 5. align (affine metric depth; pose fixed to the scene rig if available)
    base_ext = C.Extrinsics()
    rig_path = os.path.join(C.NBS_DATA_OUT, f"{scene}_rig.json")
    fit_pose = True
    if os.path.exists(rig_path):
        rig = json.load(open(rig_path))
        base_ext.yaw_deg, base_ext.cam_x, base_ext.cam_y, base_ext.cam_z, base_ext.pitch_deg = (
            rig["yaw_deg"], rig["cam_x"], rig["cam_y"], rig["cam_z"], rig["pitch_deg"])
        if "level_R" in rig:
            base_ext.level_R = np.array(rig["level_R"])
        fit_pose = False
        print(f"[rig] fixed pose from {os.path.basename(rig_path)} "
              f"(yaw={rig['yaw_deg']:.1f} x={rig['cam_x']:.2f} y={rig['cam_y']:.2f} z={rig['cam_z']:.2f})")
    sb = C.DEPTH_SCALE_BOUNDS_METRIC if backend == "dav2_metric" else C.DEPTH_SCALE_BOUNDS
    res = A.align(pts0, rays, intens, radar_ra, base_ext, scale_bounds=sb,
                  az_sin_limit=az_sin, blur=blur, fit_pose=fit_pose)

    # 6. build metric radar-frame cloud at solved params
    p = res["params"]
    ext = A._ext_from_vec(base_ext, res["vec"])
    pts_radar = G.transform_pose(G.affine_points(pts0, rays, p["depth_scale"], p["depth_shift"]), ext)
    # near-field cull (same as RA binning)
    r = np.linalg.norm(pts_radar, axis=1)
    pts_radar = pts_radar[(r >= C.NEAR_FIELD_M) & (r < C.MMW_N_RANGE * C.RANGE_RES_M)]
    print(f"[cloud] {pts_radar.shape[0]} metric points")

    # normals via open3d
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts_radar)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
        pc.orient_normals_towards_camera_location(np.zeros(3))  # toward radar origin
        normals = np.asarray(pc.normals)
    except Exception as e:
        print(f"[normals] open3d failed ({e}); using radial normals")
        normals = -pts_radar / (np.linalg.norm(pts_radar, axis=1, keepdims=True) + 1e-9)

    pcl = np.concatenate([pts_radar, normals], axis=1).astype(np.float32)
    np.save(os.path.join(out_dir, "scene", "pcl.npy"), pcl)

    # 7. save alignment + overlay
    meta = {"clip": clip, "scene": scene, "mmwave_file": mmw,
            "n_points": int(pcl.shape[0]), "range_res_m": C.RANGE_RES_M,
            "dry_run": dry_run, "reduce": reduce, "az_sin_limit": az_sin, "blur": list(blur),
            **res["params"],
            "correlation": res["correlation"], "coarse_correlation": res["coarse_correlation"],
            "ASSUMPTIONS": ["camera_K HFOV90", "range_res 0.05 m",
                            "extrinsics init from rig photo"]}
    json.dump(meta, open(os.path.join(out_dir, "alignment.json"), "w"), indent=2)
    _save_overlay(os.path.join(out_dir, "ra_overlay.png"), res["radar_ra"], res["pseudo_ra"], res["correlation"])
    print(f"[done] pcl={pcl.shape} corr={res['correlation']:.3f} -> {out_dir}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--backend", default="dav2_metric", choices=["dav2_metric", "midas"])
    ap.add_argument("--reduce", default=C.RADAR_REDUCE, choices=["mean", "median", "max", "min"])
    ap.add_argument("--az-sin", type=float, default=C.AZ_SIN_LIMIT)
    ap.add_argument("--blur-range", type=float, default=C.BLUR_RANGE_AZ[0])
    ap.add_argument("--blur-az", type=float, default=C.BLUR_RANGE_AZ[1])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    process(a.scene, a.index, dry_run=a.dry_run, stride=a.stride, backend=a.backend,
            reduce=a.reduce, az_sin=a.az_sin, blur=(a.blur_range, a.blur_az))
