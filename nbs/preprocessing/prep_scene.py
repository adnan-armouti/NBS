"""End-to-end per-scene Stage-1 prep (run in the mmir env):
   rig calibration -> static human-masked scaffold -> manifest.

Each scene is one room with its own fixed rig, so this is run once per scene.
Uses the pose-matched MIDDLE frame for everything (consistent depth cache).

  python -m preprocessing.prep_scene --scene Scene2 --n 12
"""
import os, sys, json, argparse, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C, ra_geometry as G, align as A


def clips_with_media(scene):
    out = []
    for p in IO.list_mmwave(scene):
        k = IO.parse_key(p)
        vid = IO.video_path_for(p, scene)
        pose = os.path.join(C.XRF55_ROOT, "pose", "XRF55Videos_Res", "open", scene.lower(),
                            k["subject"], f"{k['subject']}_{k['action']}.json")
        if os.path.exists(vid) and os.path.exists(pose):
            out.append(p)
    return out


def clip_depth_mid(mmw, scene):
    """Extract the pose-matched middle frame, return (metric depth, keypoints)."""
    from preprocessing import mono_depth as MD
    clip = "_".join(IO.parse_key(mmw).values())
    vdir = os.path.join(C.NBS_DATA_OUT, clip, "views")
    pose = IO.load_pose(mmw, scene); idx = len(pose) // 2
    png = IO.extract_frame_at(IO.video_path_for(mmw, scene), vdir, idx)
    dcache = os.path.join(vdir, f"depth_dav2_metric_{idx:05d}.npy")
    if os.path.exists(dcache):
        depth = np.load(dcache)
    else:
        depth = MD.get_depth(MD.read_rgb(png), backend="dav2_metric")[0]
        np.save(dcache, depth)
    return depth, np.asarray(pose[idx]["pose_keypoints"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=12, help="clips to use for rig+scaffold")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--voxel", type=float, default=0.05)
    a = ap.parse_args()
    K = C.camera_K()
    clips = clips_with_media(a.scene)
    if not clips:
        raise SystemExit(f"no clips with video+pose in {a.scene}")
    pick = [clips[i] for i in np.linspace(0, len(clips) - 1, min(a.n, len(clips))).astype(int)]
    print(f"[{a.scene}] {len(clips)} clips with media; using {len(pick)}")

    # cache depth for all picks once
    depths = {}
    for mmw in pick:
        try:
            depths[mmw] = clip_depth_mid(mmw, a.scene)
        except Exception as e:
            print(f"  depth failed {os.path.basename(mmw)}: {type(e).__name__} {e}")

    # ---- Phase 1: rig (joint pose solve per clip, human INCLUDED) -> median ----
    base = C.Extrinsics(); poses, corrs = [], []
    for mmw, (depth, kp) in depths.items():
        pts0, rays, intens = G.unproject_bases(depth, K, stride=a.stride)
        radar = IO.radar_ra(mmw, C.RADAR_REDUCE)
        res = A.align(pts0, rays, intens, radar, base,
                      scale_bounds=C.DEPTH_SCALE_BOUNDS_METRIC, fit_pose=True, verbose=False)
        poses.append([res["params"][k] for k in ("yaw_deg", "cam_x", "cam_y", "cam_z", "pitch_deg")])
        corrs.append(res["correlation"])
    med = np.median(np.array(poses), axis=0)
    rig = {"scene": a.scene, "yaw_deg": float(med[0]), "cam_x": float(med[1]),
           "cam_y": float(med[2]), "cam_z": float(med[3]), "pitch_deg": float(med[4]),
           "n_clips": len(poses), "joint_corr_mean": float(np.mean(corrs)),
           "pose_std": [float(s) for s in np.array(poses).std(0)]}
    json.dump(rig, open(os.path.join(C.NBS_DATA_OUT, f"{a.scene}_rig.json"), "w"), indent=2)
    print(f"[rig] yaw={med[0]:.1f} x={med[1]:.2f} y={med[2]:.2f} z={med[3]:.2f} "
          f"corr_mean={np.mean(corrs):.3f} std={np.array(poses).std(0)}")

    ext = C.Extrinsics()
    ext.yaw_deg, ext.cam_x, ext.cam_y, ext.cam_z, ext.pitch_deg = med

    # ---- Phase 1b: LEVELING — estimate the camera's unobserved pitch/roll from the floor
    # (the RA objective is horizontal-plane, so it does not constrain vertical tilt). ----
    d0, _ = next(iter(depths.values()))
    pts0, _, _ = G.unproject_bases(d0, K, stride=a.stride)
    cloud0 = G.transform_pose(pts0, ext)                  # level_R still identity here
    ext.level_R = G.estimate_level(cloud0)
    rig["level_R"] = ext.level_R.tolist()
    json.dump(rig, open(os.path.join(C.NBS_DATA_OUT, f"{a.scene}_rig.json"), "w"), indent=2)
    print(f"[level] camera tilt corrected via floor plane (level_R stored in rig)")

    # ---- Phase 2: scaffold (human MASKED, a=1, fixed rig + leveling) ----
    clouds = []
    for mmw, (depth, kp) in depths.items():
        d = depth.copy()
        m = G.human_mask(d.shape, kp)
        if m is not None:
            d[m] = -1.0
        pts0, rays, _ = G.unproject_bases(d, K, stride=a.stride)
        pts = G.transform_pose(pts0, ext)
        r = np.linalg.norm(pts, axis=1)
        c = pts[(r >= C.NEAR_FIELD_M) & (r < C.MMW_N_RANGE * C.RANGE_RES_M)]
        if len(c):
            clouds.append(c)
    pts = np.concatenate(clouds, axis=0)

    import open3d as o3d
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(a.voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    pc.orient_normals_towards_camera_location(np.zeros(3))
    P, Nrm = np.asarray(pc.points), np.asarray(pc.normals)
    pcl = np.concatenate([P, Nrm], axis=1).astype(np.float32)
    sdir = os.path.join(C.NBS_DATA_OUT, a.scene, "scene"); os.makedirs(sdir, exist_ok=True)
    np.save(os.path.join(sdir, "pcl.npy"), pcl)
    json.dump({"scene": a.scene, "n_clips": len(clouds), "n_points": int(len(pcl)),
               "voxel_m": a.voxel}, open(os.path.join(sdir, "scaffold.json"), "w"), indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(P[:, 0], P[:, 1], s=0.5, c=P[:, 2], cmap="viridis")
    ax.scatter([0], [0], c="red", marker="^", s=80, label="radar")
    ax.set_xlabel("x right (m)"); ax.set_ylabel("y range (m)"); ax.axis("equal"); ax.legend()
    ax.set_title(f"{a.scene} scaffold {len(pcl)} pts")
    fig.tight_layout(); fig.savefig(os.path.join(sdir, "scaffold_topdown.png"), dpi=90); plt.close(fig)
    print(f"[scaffold] {pcl.shape} -> {sdir}/pcl.npy")

    # ---- Phase 3: manifest ----
    subprocess.run([sys.executable, "-m", "preprocessing.build_manifest", "--scene", a.scene],
                   cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    main()
