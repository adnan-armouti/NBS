"""Geometry for XRF55 init: depth -> camera points -> radar frame -> pseudo-RA.

Conventions (match 3DGPS cascaded_lidar.py):
  radar frame:  +Y = boresight (range/forward), +X = right (positive azimuth),
                +Z = up (elevation).
  azimuth:      az = arctan2(-x, y)   ; elevation el = arcsin(z / r)
  az grid:      uniform in sin(az) over [-AZ_SIN_LIMIT, +AZ_SIN_LIMIT], 64 bins
                (the angle-FFT is uniform in sin(theta)).
  camera optical frame: +X right, +Y down, +Z forward (standard pinhole).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from . import xrf55_config as C


def _Rz(deg):
    """Rotation about radar +Z (up). Positive turns camera forward (+Y) toward +X (right)."""
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)


def _Rx(deg):
    """Rotation about radar +X (lateral) for pitch/tilt."""
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def unproject_bases(depth, K, stride=4):
    """Back-project a relative-depth map to affine bases for fast scale+shift tuning.

    Returns (pts0, rays, intens) where, for relative depth d at each kept pixel,
      pts0 = ray * d   (camera-optical, X right/Y down/Z fwd),
      rays = ray * 1   (unit-depth direction).
    An AFFINE metric depth d' = a*d + b then gives metric points = a*pts0 + b*rays
    (linear, so no re-unprojection per optimization step).
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vs, us].ravel()
    us = us.ravel().astype(np.float64); vs = vs.ravel().astype(np.float64)
    good = np.isfinite(d) & (d > 0)
    d, us, vs = d[good], us[good], vs[good]
    rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(d)], axis=1)
    pts0 = rays * d[:, None]
    return pts0, rays, np.ones_like(d, dtype=np.float32)


def affine_points(pts0, rays, a, b):
    """Metric camera-optical points for affine depth d' = a*d + b."""
    return a * pts0 + b * rays


def human_mask(shape, keypoints, conf_thr=0.3, dilate_frac=0.35, pad_px=25):
    """Boolean mask (True = human) from 2D pose keypoints (N,3 [x,y,conf]).

    Builds a bbox over confident joints, dilates it (humans are wider than the
    skeleton), and clips to the image. Returns None if too few confident joints.
    shape = (H, W) of the depth/image.
    """
    H, W = shape
    kp = np.asarray(keypoints, dtype=np.float64)
    good = kp[:, 2] >= conf_thr
    if good.sum() < 3:
        return None
    x, y = kp[good, 0], kp[good, 1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    dw, dh = (x1 - x0) * dilate_frac + pad_px, (y1 - y0) * dilate_frac + pad_px
    x0, x1 = max(0, int(x0 - dw)), min(W, int(x1 + dw))
    y0, y1 = max(0, int(y0 - dh)), min(H, int(y1 + dh))
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def transform_pose(pts_cam, ext):
    """Camera-optical -> radar frame (pose only; depth already metric).

    camera-optical axes (X right, Y down, Z fwd) -> radar-like local
    (x'=X right, y'=Z fwd, z'=-Y up), then yaw about +Z, pitch about +X, then a leveling
    rotation (corrects unestimated camera pitch/roll so the floor is horizontal), then translate.
    """
    p = pts_cam.astype(np.float64)
    p_rl = np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)
    level = getattr(ext, "level_R", np.eye(3))
    R = level @ _Rz(ext.yaw_deg) @ _Rx(ext.pitch_deg)
    t = np.array([ext.cam_x, ext.cam_y, ext.cam_z], dtype=np.float64)
    return (R @ p_rl.T).T + t


def estimate_level(pts):
    """Leveling rotation Rm that sends the dominant FLOOR-plane normal to +z, so the
    reconstructed room is upright in the radar frame. Estimates the camera's unobserved
    pitch/roll from the scene structure (gravity = floor normal)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts[:, :3])
    best = None
    rest = pc
    for _ in range(6):
        if len(rest.points) < 500:
            break
        model, inl = rest.segment_plane(0.08, 3, 400)
        n = np.array(model[:3]); n /= np.linalg.norm(n)
        cz = np.asarray(rest.points)[inl][:, 2].mean()
        if abs(n[2]) > 0.6:                      # ~horizontal plane
            cand = (len(inl), cz, n)
            if best is None or cz < best[1]:     # lowest horizontal plane = floor
                best = cand
        rest = rest.select_by_index(inl, invert=True)
    if best is None:
        return np.eye(3)
    up = best[2] * np.sign(best[2][2])           # floor normal, pointing up
    z = np.array([0., 0., 1.]); v = np.cross(up, z); s = np.linalg.norm(v); c = up @ z
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def camera_to_radar(pts_cam, ext, depth_scale):
    """Backward-compatible: scale relative-depth camera points then pose-transform."""
    return transform_pose(pts_cam.astype(np.float64) * float(depth_scale), ext)


def points_to_ra(pts_radar, intens, range_res=C.RANGE_RES_M,
                 n_range=C.MMW_N_RANGE, n_az=C.MMW_N_AZIMUTH,
                 az_sin_limit=C.AZ_SIN_LIMIT, near_field_m=C.NEAR_FIELD_M,
                 smooth=(1.0, 3.0)):
    """Bin radar-frame points into a pseudo (range x azimuth) magnitude map.

    Returns (n_range, n_az) float32, collapsed over elevation. smooth=(sigma_range,
    sigma_az): an anisotropic, azimuth-heavy blur matches the radar's coarse angular
    resolution (only 12 virtual antennas -> a reflector smears across azimuth).
    """
    x, y, z = pts_radar[:, 0], pts_radar[:, 1], pts_radar[:, 2]
    r = np.linalg.norm(pts_radar, axis=1)
    keep = (r >= near_field_m) & np.isfinite(r)
    x, y, z, r, w = x[keep], y[keep], z[keep], r[keep], intens[keep]
    if x.size == 0:
        return np.zeros((n_range, n_az), np.float32)
    az = np.arctan2(-x, y)                       # radar azimuth convention
    s = np.sin(az)
    # range bins
    r_idx = np.floor(r / range_res).astype(int)
    # azimuth bins uniform in sin(az) over [-lim, lim]
    a_idx = np.floor((s + az_sin_limit) / (2 * az_sin_limit) * n_az).astype(int)
    ok = (r_idx >= 0) & (r_idx < n_range) & (a_idx >= 0) & (a_idx < n_az)
    r_idx, a_idx, w = r_idx[ok], a_idx[ok], w[ok]
    ra = np.zeros((n_range, n_az), np.float32)
    np.add.at(ra, (r_idx, a_idx), w)
    sig = smooth if np.ndim(smooth) else (smooth, smooth)
    if sig[0] > 0 or sig[1] > 0:
        ra = gaussian_filter(ra, sigma=sig)
    return ra


def minmax(a):
    a = np.nan_to_num(a.astype(np.float64))
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def to_db(a, floor_db=-40.0):
    """Log-compress a magnitude map to dB and min-max to [0,1] (radar RA has huge
    dynamic range; dB correlation emphasizes structure over the few bright returns)."""
    a = np.nan_to_num(a.astype(np.float64))
    a = a / (a.max() + 1e-12)
    db = 20.0 * np.log10(a + 1e-12)
    db = np.clip(db, floor_db, 0.0)
    return (db - floor_db) / (-floor_db)


def ra_correlation(ra_a, ra_b, log=True):
    """Pearson correlation between two RA maps (dB-compressed by default)."""
    fa = to_db if log else minmax
    a, b = fa(ra_a).ravel(), fa(ra_b).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
