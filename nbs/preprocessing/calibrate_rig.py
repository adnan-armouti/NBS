"""Estimate the per-scene camera->radar RIG extrinsics once, by jointly solving
pose on several clips (with cached depth) and taking the robust median. The rig
is a scene constant; per-clip preprocessing then fixes it and solves only the
depth scale (see run_preprocess --use-rig).

  python -m preprocessing.calibrate_rig --scene Scene1 --indices 0 300 600 900 1200
Writes data/XRF55/<scene>_rig.json.
"""
import os, sys, json, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C, ra_geometry as G, align as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 300, 600, 900, 1200])
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--backend", default="dav2_metric")
    a = ap.parse_args()
    mmw_all = IO.list_mmwave(a.scene)
    base = C.Extrinsics()
    poses, corrs = [], []
    for idx in a.indices:
        mmw = mmw_all[idx]; clip = "_".join(IO.parse_key(mmw).values())
        dpath = os.path.join(C.NBS_DATA_OUT, clip, "views", f"depth_{a.backend}.npy")
        if not os.path.exists(dpath):
            print(f"skip {clip} (no cached {a.backend} depth; run run_preprocess first)"); continue
        depth = np.load(dpath)
        pts0, rays, intens = G.unproject_bases(depth, C.camera_K(), stride=a.stride)
        radar = IO.radar_ra(mmw, C.RADAR_REDUCE)
        sb = C.DEPTH_SCALE_BOUNDS_METRIC if a.backend == "dav2_metric" else C.DEPTH_SCALE_BOUNDS
        res = A.align(pts0, rays, intens, radar, base, scale_bounds=sb, fit_pose=True, verbose=False)
        p = res["params"]
        poses.append([p["yaw_deg"], p["cam_x"], p["cam_y"], p["cam_z"], p["pitch_deg"]])
        corrs.append(res["correlation"])
        print(f"{clip}: corr={res['correlation']:.3f} yaw={p['yaw_deg']:.1f} "
              f"x={p['cam_x']:.2f} y={p['cam_y']:.2f} z={p['cam_z']:.2f}")
    poses = np.array(poses)
    med = np.median(poses, axis=0)
    rig = {"scene": a.scene, "yaw_deg": float(med[0]), "cam_x": float(med[1]),
           "cam_y": float(med[2]), "cam_z": float(med[3]), "pitch_deg": float(med[4]),
           "n_clips": len(poses), "joint_corr_mean": float(np.mean(corrs)),
           "pose_std": [float(s) for s in poses.std(0)]}
    out = os.path.join(C.NBS_DATA_OUT, f"{a.scene}_rig.json")
    json.dump(rig, open(out, "w"), indent=2)
    print(f"\nRIG (median over {len(poses)} clips): yaw={med[0]:.1f} x={med[1]:.2f} "
          f"y={med[2]:.2f} z={med[3]:.2f} pitch={med[4]:.2f}")
    print(f"pose std: {poses.std(0)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
