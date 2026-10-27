"""XRF55 preprocessing configuration and assumptions.

Every value here that is an ASSUMPTION (not read from the dataset) is marked
ASSUMPTION and should be revisited when the XRF55 authors confirm calibration.
See data/XRF55/AUDIT.md and EXECUTION_PLAN.md (Stage 1, G0/G-XAXIS).
"""
from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
XRF55_ROOT = "/home/adnan/Documents/Data/XRF55"
NBS_DATA_OUT = "/home/adnan/Desktop/NBS/data/XRF55"

# ---------------------------------------------------------------------------
# mmWave tensor layout  (CONFIRMED: see memory nbs-xrf55-mmwave-layout)
#   raw file: (1, 17, 256, 128) -> drop leading 1 -> (frames=17, range=256, 128)
#   128-axis: [ Doppler 0:64 | azimuth 64:128 ]
#   RA map  = A[frame, :, 64:128]  -> (range=256, azimuth=64)
#   RD map  = A[frame, :, 0:64]    -> (range=256, doppler=64)
# ---------------------------------------------------------------------------
MMW_N_FRAMES = 17
MMW_N_RANGE = 256
MMW_DOPPLER_SLICE = (0, 64)
MMW_AZIMUTH_SLICE = (64, 128)
MMW_N_AZIMUTH = 64

# Radar range resolution (m/bin). dR = c/(2B), set by the (unpublished) chirp bandwidth B;
# the band is 60-64 GHz (B<=4 GHz => physical floor dR>=0.0375 m). DATA-DRIVEN determination
# (preprocessing/determine_range_res_direct.py): with pose fixed to the calibrated rig and the
# METRIC depth as absolute scale (a=1), the depth-vs-radar range-profile match peaks at
# dR=0.0675 m across 4 clips (per-clip peaks 0.065-0.080; corr 0.73). => B~=2.2 GHz, max range
# 17.3 m, +/-6.8 cm/bin. The physical floor 0.0375 (B=4 GHz) is firmly excluded (corr 0.25).
# Replaces the old fit 0.076 (was ~11% high). Scaffolds built before 2026-06-09 used 0.076 and
# are ~11% too deep -> rebuild when convenient. Pin B exactly from the authors' chirp cfg.
RANGE_RES_M = 0.0675  # DATA-DRIVEN (metric-depth-anchored, pose-fixed); was 0.076

# Azimuth bin centers. The angle-FFT is uniform in sin(theta); 64 bins.
# TUNED (sweep on Scene1): a narrow fan (~±24 deg, sin=0.4) best matches the
# measured RA, consistent with a 12-virtual-antenna array's usable FOV.
AZ_SIN_LIMIT = 0.4  # TUNED (was assumed 1.0)

# ---------------------------------------------------------------------------
# Camera intrinsics (RGB-only videos, 1280x720).  ASSUMPTION: Azure-Kinect-like
# color camera, horizontal FOV ~90 deg -> fx = (W/2)/tan(HFOV/2).
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 1280, 720
CAM_HFOV_DEG = 90.0  # ASSUMPTION
def camera_K():
    fx = (IMG_W / 2.0) / np.tan(np.radians(CAM_HFOV_DEG) / 2.0)
    fy = fx  # ASSUMPTION square pixels
    return np.array([[fx, 0, IMG_W / 2.0],
                     [0, fy, IMG_H / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)

# ---------------------------------------------------------------------------
# Camera -> radar extrinsics, INITIAL GUESS from the XRF55 website rig photo
# (https://aiotgroup.github.io/XRF55/). Radar frame: +Y = boresight (range),
# +X = right (positive azimuth), +Z = up (elevation). Radar is 1.42 m above
# ground. Values are midpoints of the ranges the PI gave; the aligner refines
# them within [lo, hi].  ASSUMPTION (refined by RA correlation).
# ---------------------------------------------------------------------------
@dataclass
class Extrinsics:
    # camera origin in radar frame (meters)
    cam_x: float = -1.65   # radar is 1.55-1.75 m to the RIGHT of camera -> camera left
    cam_y: float = 0.80    # camera is 0.75-0.85 m AHEAD of radar along boresight
    cam_z: float = -0.35   # radar 0.35 m HIGHER than camera
    # camera yaw about radar +Z (deg): camera rotated 60-75 deg in +azimuth vs boresight
    yaw_deg: float = 67.0
    pitch_deg: float = 0.0  # tilt about lateral axis (refined)
    # leveling rotation (3x3): corrects the camera's unestimated pitch/roll so the
    # reconstructed floor is horizontal in the radar frame. Identity until estimated
    # from the floor plane (estimate_level). Left-multiplied onto the pose rotation.
    level_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    # search bounds (lo, hi) used by the aligner
    cam_x_bounds: tuple = (-1.75, -1.55)
    cam_y_bounds: tuple = (0.75, 0.85)
    cam_z_bounds: tuple = (-0.45, -0.25)
    yaw_bounds: tuple = (60.0, 75.0)
    pitch_bounds: tuple = (-10.0, 10.0)

# Affine depth model d' = a*d_rel + b.
# MiDaS (relative): a is a wide per-clip unknown.
DEPTH_SCALE_BOUNDS = (0.05, 8.0)
# Metric backend (dav2): depth already in meters, so a is just a small range-res
# correction near ~0.65 (=> effective range_res ~7.7 cm). Tight bounds avoid a
# degenerate large-a optimum that pushes points past max range.
DEPTH_SCALE_BOUNDS_METRIC = (0.3, 1.6)
DEPTH_SHIFT_BOUNDS = (-2.0, 2.0)   # b: depth offset (meters)

# Pseudo-RA blur (sigma_range, sigma_az); azimuth-heavy to match radar angular res.
BLUR_RANGE_AZ = (1.0, 8.0)         # TUNED (sweep on Scene1)
RADAR_REDUCE = "max"               # TUNED: peak-hold over 17 frames beats mean

NEAR_FIELD_M = 0.5   # drop returns closer than this (m)
RADAR_HEIGHT_M = 1.42  # given

# ---------------------------------------------------------------------------
# WiFi geometry (Intel 5300, ch128 = 5.64 GHz). From the XRF55 paper (3643543) +
# the rig description: 1 Tx (1 antenna) at the corner closest to the camera, 3 Rx
# (3 antennas each, 9 links total) at the other three corners of a 3.1x3.1 m square.
# All WiFi sensors 1.2 m above ground (radar 1.42 m -> WiFi at radar-z = -0.22 m).
# The radar sits 0.75 m behind the near edge, centered on it. Radar frame: +Y boresight
# (range), +X right, +Z up, radar at origin.
# ---------------------------------------------------------------------------
WIFI_FREQ_GHZ = 5.64
WIFI_SQUARE_M = 3.1
WIFI_Z = 1.20 - RADAR_HEIGHT_M     # = -0.22 m
WIFI_NEAR_Y = 0.75                  # radar is 0.75 m behind the near edge
WIFI_RX_ANTENNAS = 3               # Intel 5300 has 3 Rx antennas per receiver
def wifi_layout():
    """Return (tx_xyz, [rx0_xyz, rx1_xyz, rx2_xyz]) corner centers in the radar frame.
    Tx = near-left corner (by the camera); Rx = near-right, far-left, far-right."""
    s = WIFI_SQUARE_M / 2.0
    near, far = WIFI_NEAR_Y, WIFI_NEAR_Y + WIFI_SQUARE_M
    tx = (-s, near, WIFI_Z)
    rx = [(s, near, WIFI_Z), (-s, far, WIFI_Z), (s, far, WIFI_Z)]
    return tx, rx
