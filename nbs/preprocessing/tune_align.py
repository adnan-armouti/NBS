"""Empirical alignment sweep (fast, base env): uses cached depth.npy.

Sweeps radar frame-reduction x azimuth FOV x anisotropic blur and reports the
refined dB-RA correlation for each, so we tune the calibration by measurement.
  python -m preprocessing.tune_align --scene Scene1 --index 0
"""
import os, sys, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_io as IO, xrf55_config as C, ra_geometry as G, align as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1"); ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--backend", default="dav2_metric")
    a = ap.parse_args()
    mmw = IO.list_mmwave(a.scene)[a.index]
    clip = "_".join(IO.parse_key(mmw).values())
    depth = np.load(os.path.join(C.NBS_DATA_OUT, clip, "views", f"depth_{a.backend}.npy"))
    pts0, rays, intens = G.unproject_bases(depth, C.camera_K(), stride=a.stride)
    base = C.Extrinsics()

    reduces = ["max", "median"]
    az_sins = [0.3, 0.4, 0.5, 0.6]
    blur_azs = [4.0, 6.0, 8.0, 10.0]
    blur_rs = [1.0, 2.0]

    results = []
    radar_cache = {r: IO.radar_ra(mmw, r) for r in reduces}
    for red, azs, baz, br in itertools.product(reduces, az_sins, blur_azs, blur_rs):
        res = A.align(pts0, rays, intens, radar_cache[red], base,
                      az_sin_limit=azs, blur=(br, baz),
                      coarse_scale=6, coarse_yaw=4, verbose=False)
        results.append((res["correlation"], red, azs, baz, res["params"], res["vec"]))
        print(f"reduce={red:6s} az_sin={azs:.1f} blur_az={baz:.0f} blur_r={br:.0f} -> corr={res['correlation']:.4f}")

    results.sort(key=lambda x: -x[0])
    print("\n=== TOP 8 ===")
    for c, red, azs, baz, p, v in results[:8]:
        print(f"corr={c:.4f}  reduce={red:6s} az_sin={azs:.1f} blur_az={baz:.0f}  "
              f"a={p['depth_scale']:.2f} b={p['depth_shift']:.2f} yaw={p['yaw_deg']:.0f} "
              f"x={p['cam_x']:.2f} y={p['cam_y']:.2f} z={p['cam_z']:.2f}")

    # save best overlay
    bc, bred, bazs, bbaz, bp, bv = results[0]
    res = A.align(pts0, rays, intens, radar_cache[bred], base,
                  az_sin_limit=bazs, blur=(blur_r, bbaz), verbose=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    rr, pr = res["radar_ra"], res["pseudo_ra"]
    axs[0].imshow(rr, aspect="auto", origin="lower"); axs[0].set_title(f"radar RA ({bred}, dB)")
    axs[1].imshow(pr, aspect="auto", origin="lower"); axs[1].set_title(f"pseudo RA (az_sin={bazs},blur_az={bbaz:.0f})")
    ov = np.zeros((*rr.shape, 3)); ov[..., 0] = rr; ov[..., 1] = pr
    axs[2].imshow(ov, aspect="auto", origin="lower"); axs[2].set_title(f"overlay corr={bc:.3f}")
    for ax in axs: ax.set_xlabel("azimuth bin"); ax.set_ylabel("range bin")
    out = os.path.join(C.NBS_DATA_OUT, clip, "ra_overlay_best.png")
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)
    print(f"\nbest corr={bc:.4f}  saved {out}")


if __name__ == "__main__":
    main()
