"""Build the per-SCENE static geometry scaffold (handles the moving human).

A scene (e.g. Scene1) is one room with a fixed camera/radar rig; only the human
moves. So the static geometry is shared across all the scene's clips. We:
  1. take several diverse clips (human in different places),
  2. extract the pose-matched RGB frame + metric depth,
  3. MASK OUT the human (2D-pose bbox) so the cloud is static room only,
  4. unproject at true metric scale (a=1) and transform to the radar frame with
     the fixed scene rig,
  5. aggregate across clips (human-occluded holes in one clip are filled by
     others), voxel-downsample, estimate normals,
  6. save data/XRF55/<scene>/scene/pcl.npy  (the shared scaffold).

Run in the mmir env (metric depth):
  python -m preprocessing.build_scaffold --scene Scene1 --n 16
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C, ra_geometry as G


def _load_rig(scene):
    rp = os.path.join(C.NBS_DATA_OUT, f"{scene}_rig.json")
    if not os.path.exists(rp):
        raise SystemExit(f"missing {rp}; run calibrate_rig first")
    rig = json.load(open(rp))
    e = C.Extrinsics()
    e.yaw_deg, e.cam_x, e.cam_y, e.cam_z, e.pitch_deg = (
        rig["yaw_deg"], rig["cam_x"], rig["cam_y"], rig["cam_z"], rig["pitch_deg"])
    if "level_R" in rig:
        e.level_R = np.array(rig["level_R"])
    return e


def static_cloud_for_clip(mmw, scene, ext, K, stride=4, masked=True):
    """Human-masked metric static cloud (radar frame) for one clip, a=1 (true metric)."""
    from preprocessing import mono_depth as MD
    clip = "_".join(IO.parse_key(mmw).values())
    out_dir = os.path.join(C.NBS_DATA_OUT, clip)
    pose = IO.load_pose(mmw, scene)
    if pose is None:
        return None
    idx = len(pose) // 2                              # representative middle frame
    mkv = IO.video_path_for(mmw, scene)
    if not os.path.exists(mkv):
        return None
    png = IO.extract_frame_at(mkv, os.path.join(out_dir, "views"), idx)
    dcache = os.path.join(out_dir, "views", f"depth_dav2_metric_{idx:05d}.npy")  # frame-indexed
    if os.path.exists(dcache):
        depth = np.load(dcache)
    else:
        depth = MD.get_depth(MD.read_rgb(png), backend="dav2_metric")[0]
        np.save(dcache, depth)
    if masked:
        m = G.human_mask(depth.shape, pose[idx]["pose_keypoints"])
        if m is not None:
            depth = depth.copy(); depth[m] = -1.0     # invalidate human pixels
    pts0, rays, intens = G.unproject_bases(depth, K, stride=stride)
    pts = G.transform_pose(pts0, ext)                 # a=1, b=0 (trust metric depth)
    r = np.linalg.norm(pts, axis=1)
    return pts[(r >= C.NEAR_FIELD_M) & (r < C.MMW_N_RANGE * C.RANGE_RES_M)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1")
    ap.add_argument("--n", type=int, default=16, help="num clips to aggregate")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--voxel", type=float, default=0.05, help="downsample voxel (m)")
    a = ap.parse_args()
    ext, K = _load_rig(a.scene), C.camera_K()
    mmw_all = IO.list_mmwave(a.scene)
    pick = np.linspace(0, len(mmw_all) - 1, a.n).astype(int)
    clouds = []
    for i in pick:
        try:
            c = static_cloud_for_clip(mmw_all[i], a.scene, ext, K, stride=a.stride)
        except Exception as e:
            print(f"  clip {i} failed: {type(e).__name__} {e}"); continue
        if c is not None and len(c):
            clouds.append(c); print(f"  clip {i}: +{len(c)} static pts")
    pts = np.concatenate(clouds, axis=0)
    print(f"[aggregate] {len(pts)} pts from {len(clouds)} clips")

    import open3d as o3d
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(a.voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    pc.orient_normals_towards_camera_location(np.zeros(3))
    P, Nrm = np.asarray(pc.points), np.asarray(pc.normals)
    pcl = np.concatenate([P, Nrm], axis=1).astype(np.float32)

    out_dir = os.path.join(C.NBS_DATA_OUT, a.scene, "scene")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pcl.npy"), pcl)
    json.dump({"scene": a.scene, "n_clips": len(clouds), "n_points": int(len(pcl)),
               "voxel_m": a.voxel, "scale": "a=1 metric (dav2)", "human": "masked via 2D pose"},
              open(os.path.join(out_dir, "scaffold.json"), "w"), indent=2)
    print(f"[done] scaffold {pcl.shape} -> {out_dir}/pcl.npy")

    # top-down viz
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(P[:, 0], P[:, 1], s=0.5, c=P[:, 2], cmap="viridis")
    ax.scatter([0], [0], c="red", marker="^", s=80, label="radar")
    ax.set_xlabel("x (right, m)"); ax.set_ylabel("y (boresight/range, m)")
    ax.set_title(f"{a.scene} static scaffold (top-down), {len(pcl)} pts"); ax.legend(); ax.axis("equal")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "scaffold_topdown.png"), dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
