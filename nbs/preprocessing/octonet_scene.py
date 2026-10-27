"""OctoNet scene layout: node positions, sensor placement, RGB-D decoding, camera back-projection.

Everything here is in the **MoCap (OptiTrack) world frame**: metres, Z up, floor at Z=0, origin at the
subject's standing spot (the skeleton hip/head hover around X,Y ~ 0 in every take).

Sources (verified 2026-10-27 on mmsd-h100):
- NODE positions: the OptiTrack CSV exports (``mocap_csv_final/mocap_S1/**/*.csv``) carry rigid bodies
  ``Node1``..``Node5`` (3 reflective markers per node, paper Sec. 3.2). They are constant across every
  Scene-1 take (std 0 mm). Scenes 2 and 3 have NO node rigid bodies in their CSVs.
- WiFi Tx/Rx and the obstacle are NOT tracked; they are read off the schematic in paper Fig. 2(a)
  (4.1 m x 4.1 m sensing area) and are APPROXIMATE (+-0.5 m). Figure axes -> MoCap axes: figure "up"
  = +X (Node-1, which "faces participants directly", sits at X=+2.9), figure "right" = -Y.
- Depth PNGs are 8-bit, linear 0..10 m (OctoNet ``dataset_loader.png_to_depth(min_dist=0, max_dist=10)``),
  i.e. 3.9 cm per grey level. The RGB .mp4 is NOT shipped (only cut-window .json/.txt).
- Camera intrinsics are NOT shipped: nominal RealSense D455 depth FoV 87 x 58 deg at 640x480.
- Camera pose = node position, looking horizontally at the origin (nodes face the subject). This is an
  ASSUMPTION until ``octonet_calibrate`` exists; label figures accordingly.
"""
from __future__ import annotations
import csv, glob, os
from dataclasses import dataclass
import numpy as np

from preprocessing import octonet_config as C

# ---------------------------------------------------------------------------
# Scene-1 node rigid bodies (metres, MoCap frame), mean over 'Take 2024-05-18 01.43.58 PM_subject_2.csv'
# and identical (<3 mm) in every other S1 take checked.
# ---------------------------------------------------------------------------
NODE_POS_S1 = {
    1: np.array([2.921, 0.011, 0.901]),
    2: np.array([-2.257, -0.887, 0.792]),
    3: np.array([2.431, 1.714, 0.925]),
    4: np.array([1.416, -1.511, 0.865]),
    5: np.array([-2.100, 1.198, 0.891]),
}

# Per-node sensors, from the node_*/ directories on disk + cut_manual.csv (not the paper table).
NODE_SENSORS = {
    1: ["vayyar(SFCW)", "uwb", "wifi", "depthCamera", "mmWave(TI)", "IRA", "acoustic", "seekThermal", "polar(HR)"],
    2: ["wifi", "mmWave(TI)", "IRA"],
    3: ["wifi", "depthCamera", "mmWave(TI)", "IRA", "seekThermal"],
    4: ["wifi", "mmWave(TI)", "IRA", "ToF", "acoustic"],
    5: ["depthCamera", "mmWave(TI)", "IRA"],
}
SENSOR_COLUMNS = ["vayyar(SFCW)", "uwb", "wifi", "depthCamera", "mmWave(TI)", "IRA", "ToF", "acoustic",
                  "seekThermal", "polar(HR)"]

SENSING_AREA_M = 4.1                      # paper Fig. 2(a); square, taken as centred on the MoCap origin

def _fig(fx: float, fy: float) -> np.ndarray:
    """Paper Fig. 2(a) schematic coords (fx right, fy up, both in [0,1] over the 4.1 m square) -> MoCap XY."""
    h = SENSING_AREA_M / 2
    return np.array([(fy - 0.5) * 2 * h, -(fx - 0.5) * 2 * h])

# APPROXIMATE (schematic) positions, metres. Heights are guesses (table/tripod level).
WIFI_TX_APPROX = np.array([*_fig(1.00, 0.78), 0.9])            # red triangle, right edge just below Node-1
WIFI_RX_APPROX = {
    1: np.array([*_fig(0.00, 0.80), 0.9]),                     # left edge, upper
    2: np.array([*_fig(1.00, 0.20), 0.9]),                     # right edge, lower
    3: np.array([*_fig(0.15, 0.00), 0.9]),                     # bottom edge, left
    4: np.array([*_fig(0.62, 1.00), 0.9]),                     # top edge, behind the obstacle (NLOS)
}
OBSTACLE_APPROX = {"center": np.array([*_fig(0.72, 0.95), 0.9]), "length_m": 0.8, "thickness_m": 0.03,
                   "note": "3 cm wooden board blocking Tx->Rx-4 (paper Sec. 3.2); position schematic"}
WIFI_RX_HOST_NODE = {1: 1, 2: 2, 3: 3, 4: 4}                   # node_k/wifi/ holds receiver k's CSI

# Representative Scene-1 recordings with depth(1,3,5)+uwb+wifi+vayyar+mocap all present.
REC_SIT = "20240518134358"    # subject 2, 'sit', trial 1  (CSV 'Take 2024-05-18 01.43.58 PM_subject_2.csv')
REC_WALK = "20240518135416"   # subject 2, 'walk', trial 1

# ---------------------------------------------------------------------------
# Rigid bodies from an OptiTrack CSV
# ---------------------------------------------------------------------------
def load_rigid_bodies(csv_path: str) -> dict[str, np.ndarray]:
    """Mean Position (m) of every non-skeleton rigid body in an OptiTrack CSV export."""
    rows = list(csv.reader(open(csv_path)))
    names, kinds = rows[3], rows[5]
    data = np.array([[float(v) if v not in ("", "nan") else np.nan for v in r] for r in rows[7:]], dtype=float)
    out = {}
    for nm in sorted(set(n for n in names if n and ":" not in n and not n.startswith("Skeleton") and n != "Name")):
        cols = [i for i, (n, k) in enumerate(zip(names, kinds)) if n == nm and k == "Position"]
        if len(cols) == 3:
            out[nm] = np.nanmean(data[:, cols], axis=0) / 1000.0
    return out

def mocap_csv_for_recording(rec: str) -> str | None:
    """OptiTrack CSV whose take name matches a cut_manual recording_time (YYYYmmddHHMMSS)."""
    import datetime as dt
    t = dt.datetime.strptime(rec[:14], "%Y%m%d%H%M%S")
    pat = t.strftime("Take %Y-%m-%d %I.%M.%S %p")
    hits = glob.glob(os.path.join(C.OCTONET_ROOT, "mocap_csv_final", "*", "*", pat + "*.csv"))
    return hits[0] if hits else None

# ---------------------------------------------------------------------------
# RGB-D
# ---------------------------------------------------------------------------
DEPTH_MAX_M = 10.0
D455_HFOV_DEG, D455_VFOV_DEG = 87.0, 58.0   # nominal depth FoV (datasheet); intrinsics are not shipped
W, H = 640, 480

@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float; w: int = W; h: int = H

def d455_nominal_intrinsics() -> Intrinsics:
    fx = (W / 2) / np.tan(np.deg2rad(D455_HFOV_DEG / 2))
    fy = (H / 2) / np.tan(np.deg2rad(D455_VFOV_DEG / 2))
    return Intrinsics(fx, fy, W / 2, H / 2)

def depth_clip_dir(node: int, rec: str) -> str:
    hits = glob.glob(os.path.join(C.OCTONET_ROOT, f"node_{node}", "depthCamera", "data", f"{rec}*"))
    if not hits:
        raise FileNotFoundError(f"no depth clip for node_{node} rec {rec}")
    return hits[0]

def depth_frames(clip_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(clip_dir, "depth_*.png")))

def load_depth_m(png_path: str) -> np.ndarray:
    """8-bit PNG -> metres (0 = invalid/no return), per OctoNet's loader (linear 0..10 m)."""
    from PIL import Image
    img = np.asarray(Image.open(png_path)).astype(np.float32)
    return img / 255.0 * DEPTH_MAX_M

def backproject(depth_m: np.ndarray, K: Intrinsics | None = None, stride: int = 2,
                z_min: float = 0.3, z_max: float = 9.5) -> np.ndarray:
    """Depth image -> (N,3) points in the CAMERA frame (x right, y down, z forward), metres."""
    K = K or d455_nominal_intrinsics()
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    z = depth_m[::stride, ::stride]
    ok = (z > z_min) & (z < z_max)
    x = (u - K.cx) / K.fx * z
    y = (v - K.cy) / K.fy * z
    return np.stack([x[ok], y[ok], z[ok]], axis=1)

def camera_pose_assumed(node: int, target_xy=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """ASSUMED camera extrinsics: at the node rigid-body position, optical axis horizontal, pointing
    at the subject spot. Returns (R_wc (3,3), t_w (3,)) with p_world = R_wc @ p_cam + t_w.
    Camera frame: x right, y down, z forward. World: Z up."""
    t = NODE_POS_S1[node].copy()
    fwd = np.array([target_xy[0] - t[0], target_xy[1] - t[1], 0.0]); fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = -up
    R_wc = np.stack([right, down, fwd], axis=1)   # columns = camera axes expressed in world
    return R_wc, t

def cam_to_world(p_cam: np.ndarray, node: int) -> np.ndarray:
    R, t = camera_pose_assumed(node)
    return p_cam @ R.T + t

# ---------------------------------------------------------------------------
# RF constants used by the viz (SFCW ones live in octonet_config / vayyar_radar_config)
# ---------------------------------------------------------------------------
UWB = dict(device="Novelda XeThru X4M200", bandwidth_hz=2.5e9, n_bins=1535, max_range_m=9.9,
           bin_m=9.9 / 1535, node=1, note="per-frame CIR magnitude, 1535 range bins (paper Tab. 2)")
WIFI = dict(tx="Xiaomi AX6000 (1 antenna)", rx="4x RPi CM4 + Intel AX200 (2 antennas each)",
            channel=36, fc_hz=5.18e9, bw_hz=40e6, n_sub=114, rx_antennas=2, nodes=(1, 2, 3, 4),
            note="CSI (F, 2, 114) complex per receiver; receiver k stored under node_k/wifi")
