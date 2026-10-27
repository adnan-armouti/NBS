"""Calibrate relative monocular depth to metric, and refine camera->radar pose,
by maximizing dB-RA correlation between the depth-derived pseudo-RA and the
measured radar RA. XRF55 analogue of 3DGPS optimize_alignment, with an added
AFFINE metric-depth model (MiDaS depth is relative; LiDAR was metric).

Param vector v = [a, b, yaw_deg, cam_x, cam_y, cam_z, pitch_deg]
  metric depth d' = a*d_rel + b  ->  points = a*pts0 + b*rays
Objective: -corr_dB(pseudo_RA(v), radar_RA) + soft-bound penalty.
"""
import copy
import numpy as np
from scipy.optimize import minimize
from . import ra_geometry as G
from . import xrf55_config as C


def _ext_from_vec(base, v):
    e = copy.copy(base)
    e.yaw_deg, e.cam_x, e.cam_y, e.cam_z, e.pitch_deg = v[2], v[3], v[4], v[5], v[6]
    return e


def align(pts0, rays, intens, radar_ra, base_ext,
          scale_bounds=C.DEPTH_SCALE_BOUNDS, shift_bounds=C.DEPTH_SHIFT_BOUNDS,
          range_res=C.RANGE_RES_M, az_sin_limit=C.AZ_SIN_LIMIT,
          blur=(1.0, 3.0), coarse_scale=8, coarse_yaw=6, fit_pose=True, verbose=True):
    """Solve depth affine [a,b] (and, if fit_pose, the rig pose [yaw,x,y,z,pitch])
    maximizing dB-RA correlation.

    fit_pose=True  : joint solve (use for per-scene rig calibration).
    fit_pose=False : pose FIXED to base_ext (a scene constant); solve only [a,b]
                     per clip. Cleaner and more robust once the rig is known.
    """
    b_ext = base_ext

    def build(v):
        ext = _ext_from_vec(b_ext, v)
        pts = G.affine_points(pts0, rays, v[0], v[1])
        pr = G.transform_pose(pts, ext)
        return G.points_to_ra(pr, intens, range_res=range_res,
                              az_sin_limit=az_sin_limit, smooth=blur)

    def corr(v):
        return G.ra_correlation(build(v), radar_ra, log=True)

    pose0 = [b_ext.yaw_deg, b_ext.cam_x, b_ext.cam_y, b_ext.cam_z, b_ext.pitch_deg]

    # ---- Phase 1: coarse grid over scale a (and yaw if fitting pose) ----
    scales = np.linspace(scale_bounds[0], scale_bounds[1], coarse_scale)
    yaws = np.linspace(b_ext.yaw_bounds[0], b_ext.yaw_bounds[1], coarse_yaw) if fit_pose else [b_ext.yaw_deg]
    best_v, best_c = None, -np.inf
    for a in scales:
        for y in yaws:
            v = [a, 0.0, y, b_ext.cam_x, b_ext.cam_y, b_ext.cam_z, b_ext.pitch_deg]
            c = corr(v)
            if c > best_c:
                best_c, best_v = c, list(v)
    if verbose:
        print(f"[align] coarse best corr={best_c:.4f} a={best_v[0]:.2f} yaw={best_v[2]:.1f}")

    # ---- Phase 2: Nelder-Mead (all 7 if fit_pose, else only [a,b]) ----
    bounds = [scale_bounds, shift_bounds, b_ext.yaw_bounds, b_ext.cam_x_bounds,
              b_ext.cam_y_bounds, b_ext.cam_z_bounds, b_ext.pitch_bounds]

    def penalty(v):
        p = 0.0
        for val, (lo, hi) in zip(v, bounds):
            if val < lo: p += (lo - val) ** 2
            if val > hi: p += (val - hi) ** 2
        return p

    if fit_pose:
        x0 = best_v
        obj = lambda v: -corr(v) + 10.0 * penalty(v)
    else:                       # solve only [a, b]; pose fixed
        x0 = best_v[:2]
        obj = lambda ab: -corr([ab[0], ab[1], *pose0]) + 10.0 * penalty([ab[0], ab[1], *pose0])

    res = minimize(obj, x0=x0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
    v = list(res.x) + pose0 if not fit_pose else res.x
    final_c = corr(v)
    if verbose:
        print(f"[align] refined corr={final_c:.4f} a={v[0]:.3f} b={v[1]:.3f} yaw={v[2]:.2f} "
              f"x={v[3]:.2f} y={v[4]:.2f} z={v[5]:.2f} pitch={v[6]:.2f}")

    return {
        "params": {"depth_scale": float(v[0]), "depth_shift": float(v[1]),
                   "yaw_deg": float(v[2]), "cam_x": float(v[3]), "cam_y": float(v[4]),
                   "cam_z": float(v[5]), "pitch_deg": float(v[6])},
        "correlation": float(final_c), "coarse_correlation": float(best_c),
        "radar_ra": G.to_db(radar_ra), "pseudo_ra": G.to_db(build(v)),
        "build": build, "vec": [float(x) for x in v],
    }
