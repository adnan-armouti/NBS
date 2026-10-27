"""Determine the radar range resolution dR (m/bin) DIRECTLY FROM DATA.

dR = c/(2B) is set by the chirp bandwidth B (unpublished for XRF55; band is 60-64 GHz so B<=4 GHz
=> dR >= 0.0375 m). We read dR off the data using the METRIC monocular depth as the absolute
scale reference: fix the depth scale a=1.0 (trust the metric depth in meters) and find the dR
that maximizes the dB-RA correlation between the metric-depth pseudo-RA and the measured radar RA.
With a fixed, the a<->dR degeneracy is broken, so the correlation peaks at the true dR.

  python -m preprocessing.determine_range_res --scene Scene1 --nclips 8
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C, ra_geometry as G, align as A

C_LIGHT = 299792458.0
DR_MIN = C_LIGHT / (2 * 4e9)            # 0.0375 m: physical floor (IWR6843 max 4 GHz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--nclips", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6); ap.add_argument("--backend", default="dav2_metric")
    a = ap.parse_args()
    mmws = IO.list_mmwave(a.scene)
    idx = np.linspace(0, len(mmws) - 1, a.nclips).astype(int)

    grid = np.round(np.arange(0.0375, 0.1001, 0.0025), 4)   # candidate dR, from the physical floor up
    per_clip = []
    used = 0
    for i in idx:
        mmw = mmws[i]; clip = "_".join(IO.parse_key(mmw).values())
        dpath = os.path.join(C.NBS_DATA_OUT, clip, "views", f"depth_{a.backend}.npy")
        if not os.path.exists(dpath):
            continue
        depth = np.load(dpath)
        pts0, rays, intens = G.unproject_bases(depth, C.camera_K(), stride=a.stride)
        radar = IO.radar_ra(mmw, "max")
        base = C.Extrinsics()
        cs = []
        for dr in grid:
            # a fixed at 1.0 (metric truth); pose+shift free so they can't be blamed for the scale
            res = A.align(pts0, rays, intens, radar, base, scale_bounds=(1.0, 1.0),
                          range_res=float(dr), az_sin_limit=C.AZ_SIN_LIMIT, blur=C.BLUR_RANGE_AZ[::-1],
                          coarse_scale=1, coarse_yaw=4, fit_pose=True, verbose=False)
            cs.append(res["correlation"])
        per_clip.append(cs); used += 1
        print(f"  clip {clip}: peak dR={grid[int(np.argmax(cs))]:.4f} m  (corr {max(cs):.3f})")

    M = np.array(per_clip)                      # (clips, dR)
    mean_c = M.mean(0)
    best = grid[int(np.argmax(mean_c))]
    B_implied = C_LIGHT / (2 * best) / 1e9
    print(f"\n=== {used} clips, depth-anchored dR sweep (a=1.0 fixed) ===")
    for dr, c in zip(grid, mean_c):
        mark = "  <== peak" if dr == best else ""
        print(f"  dR={dr:.4f} m  mean corr={c:.3f}  {'#'*int(c*40)}{mark}")
    print(f"\nDATA-DRIVEN range resolution = {best:.4f} m/bin  (=> B ~= {B_implied:.2f} GHz, max range {best*256:.1f} m)")
    print(f"physical floor dR>=0.0375 (B<=4 GHz). margin of error per wall = +/-{best*100:.1f} cm.")


if __name__ == "__main__":
    main()
