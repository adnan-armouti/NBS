"""Determine radar range resolution dR (m/bin) directly from data, pose FIXED to the calibrated
rig so dR is the ONLY range-scale knob (the earlier free-pose sweep let the boresight offset
re-absorb the scale -> monotonic, no peak).

Metric depth (a=1) gives an ABSOLUTE range profile in meters. The radar RA gives a range profile
in bins. dR that best aligns the two (in dB, structure over the bright few) is the data-driven
value. Physical floor dR>=0.0375 m (IWR6843 max 4 GHz).

  python -m preprocessing.determine_range_res_direct --scene Scene1
"""
import os, sys, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C, ra_geometry as G

C_LIGHT = 299792458.0


def db_profile(p):
    p = np.nan_to_num(p.astype(np.float64)); p = p / (p.max() + 1e-12)
    return np.clip(20 * np.log10(p + 1e-12), -40, 0)


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    rig = json.load(open(os.path.join(C.NBS_DATA_OUT, f"{a.scene}_rig.json")))
    ext = C.Extrinsics()
    ext.yaw_deg, ext.cam_x, ext.cam_y = rig["yaw_deg"], rig["cam_x"], rig["cam_y"]
    ext.cam_z, ext.pitch_deg = rig["cam_z"], rig["pitch_deg"]
    if "level_R" in rig: ext.level_R = np.array(rig["level_R"])

    mmws = IO.list_mmwave(a.scene)
    grid = np.round(np.arange(0.0375, 0.1001, 0.0025), 4)
    N = C.MMW_N_RANGE
    rng_bins = np.arange(N)
    per_clip = []
    for mmw in mmws:
        clip = "_".join(IO.parse_key(mmw).values())
        dpath = os.path.join(C.NBS_DATA_OUT, clip, "views", "depth_dav2_metric.npy")
        if not os.path.exists(dpath):
            continue
        depth = np.load(dpath)
        pts0, rays, intens = G.unproject_bases(depth, C.camera_K(), stride=a.stride)
        pts = G.affine_points(pts0, rays, 1.0, 0.0)          # a=1: metric, absolute meters
        pr = G.transform_pose(pts, ext)                      # radar frame, rig pose fixed
        r = np.linalg.norm(pr, axis=1)
        keep = (r >= C.NEAR_FIELD_M) & np.isfinite(r)
        # depth range profile sampled at candidate radar ranges r=bin*dR
        radar_ra = IO.radar_ra(mmw, "max")                   # (256, az)
        p_radar = db_profile(radar_ra.sum(1))                # bin-indexed
        cs = []
        for dr in grid:
            edges = np.arange(N + 1) * dr
            p_depth, _ = np.histogram(r[keep], bins=edges, weights=intens[keep])
            cs.append(corr(p_radar, db_profile(p_depth)))
        per_clip.append(cs)
        print(f"  clip {clip}: peak dR={grid[int(np.argmax(cs))]:.4f} (corr {max(cs):.3f})")

    M = np.array(per_clip); mean_c = M.mean(0)
    best = grid[int(np.argmax(mean_c))]
    print(f"\n=== {len(per_clip)} clips, pose-fixed depth-vs-radar range-profile match ===")
    for dr, c in zip(grid, mean_c):
        print(f"  dR={dr:.4f} m  mean corr={c:+.3f}  {'#'*max(0,int(c*40))}{'  <== peak' if dr==best else ''}")
    print(f"\nDATA-DRIVEN range resolution = {best:.4f} m/bin")
    print(f"  => B ~= {C_LIGHT/(2*best)/1e9:.2f} GHz, max range {best*N:.1f} m, margin +/-{best*100:.1f} cm/bin")


if __name__ == "__main__":
    main()
