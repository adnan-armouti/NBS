"""OctoNet (NeurIPS 2025, hku-aiot/OctoNet) preprocessing configuration.

Mirror of ``preprocessing.xrf55_config``, retargeted from XRF55 (TI FMCW mmWave, RGB-only, no
MoCap) to OctoNet (Vayyar IMAGEVK-74 **SFCW** radar + real RGB-D + 120 Hz OptiTrack MoCap).

Values that are CONFIRMED from the Vayyar datasheet / OctoNet paper / repo dataloader are marked
CONFIRMED. Values that must be read back from the actual data once the 768 GB download completes are
marked ``FLAG-DATA`` and default to the best current estimate (so Phase-1 code is runnable now).

World frame = the MoCap (OptiTrack) world; every sensor is registered into it (see
``octonet_calibrate``). This replaces XRF55's radar-origin frame + mono-depth/RA-correlation
bootstrap, which is unnecessary now that metric RGB-D and MoCap ground truth exist.
"""
from dataclasses import dataclass, field
import numpy as np

C_LIGHT = 299792458.0

# ---------------------------------------------------------------------------
# Dataset paths.  OctoNet lives on the dedicated datasets SSD (/data, device
# nvme0n1p1), symlinked at ~/Documents/Data/OctoNet for tooling convention.
# ---------------------------------------------------------------------------
import os as _os
# Overridable per machine (tools/env_node.sh sets them on mmsd-h100, where the data lives at
# /data/mmsd/datasets/OctoNet/OctoNet-upload). The defaults are the original workstation paths.
OCTONET_ROOT = _os.environ.get("OCTONET_DATASET_ROOT", "/data/datasets/OctoNet/OctoNet-upload")
OCTONET_STORE = _os.path.dirname(OCTONET_ROOT)                 # download/extract location
NBS_DATA_OUT = _os.environ.get("NBS_DATA_OUT", "/home/adnan/Desktop/NBS/data/OctoNet")
# Master index joining every recording's per-node/per-modality paths + timing (CONFIRMED).
# Columns: user_id, activity, recording_time, node_<n>_<modality>_data_path ..., vayyar via
# node_1_vayyar_data_path, mocap_data_path, imu_data_path, start_time, cut_points, cut_ratio,
# cut_timestamps.  Vayyar lives ONLY on node_1 (flat files under vayyar_pickle/).
CUT_MANUAL_CSV = OCTONET_ROOT + "/cut_manual.csv"

# numpy 2.x pickled the mocap_pose/*.npy files; loading them under numpy 1.x needs a module
# alias (numpy._core -> numpy.core). octonet_io.numpy2_compat() installs it.  (env mmir = 1.26)

# ---------------------------------------------------------------------------
# Vayyar SFCW tensor layout  (CONFIRMED against the real vayyar_pickle files)
#   Each .pickle is a DICT with keys ['timestamps','data'] (NOT a list of dicts):
#     data       : (400, 100, T) complex128  -> transpose to (T, 400, 100) for per-frame use
#     timestamps : (T, 1) of datetime.datetime, one per frame
#   400 = 20 Tx x 20 Rx pair index; 100 = swept CW frequency points (63-67 GHz); T = frames.
#   Files are flat under vayyar_pickle/ and all belong to node_1 (the Vayyar sits on node 1):
#     <YYYYmmddHHMMSS>_node_1_modality_vayyarmmwave_subject_<n>_activity_<a>_trial_<k>.pickle
#   Range compression = IFFT over the 100-freq axis (see mm3DPS sfcw_to_range) -- there is NO
#   range-FFT-over-fast-time and NO Doppler-half packing (contrast XRF55's (17,256,128)).
# ---------------------------------------------------------------------------
VAY_N_TX = 20
VAY_N_RX = 20
VAY_N_PAIRS = 400                                  # 20 Tx x 20 Rx
VAY_N_FREQ = 100
VAY_F_START_HZ = 63.0e9
VAY_F_STOP_HZ = 67.0e9
VAY_F_CENTER_HZ = 65.0e9
VAY_BW_HZ = 4.0e9
VAY_DF_HZ = VAY_BW_HZ / (VAY_N_FREQ - 1)           # ~40.4 MHz per step
VAY_LAMBDA_M = C_LIGHT / VAY_F_CENTER_HZ           # ~4.61 mm @ 65 GHz
# FLAG-DATA: the 400-pair index order (Tx-major pair = tx*20+rx vs Rx-major) is an ASSUMPTION until
# confirmed against the real vayyar_pickle / Vayyar vSDK antenna map.
VAY_PAIR_ORDER = "tx_major"                        # ASSUMPTION

def vayyar_freqs_hz():
    """The 100 swept CW tone frequencies (Hz), f_start + n*df."""
    return VAY_F_START_HZ + np.arange(VAY_N_FREQ) * VAY_DF_HZ

# Radar range resolution (m/bin) = c/(2*BW). Unlike XRF55 this is the exact SFCW physical value;
# no data-fit is needed because the sweep bandwidth is known (4 GHz).
RANGE_RES_M = C_LIGHT / (2.0 * VAY_BW_HZ)           # 0.0375 m
R_MAX_M = C_LIGHT / (2.0 * VAY_DF_HZ)               # ~3.71 m max unambiguous range

NEAR_FIELD_M = 0.3   # drop returns closer than this (m); Vayyar is a near-field imager

# ---------------------------------------------------------------------------
# MoCap (OptiTrack) = world reference.  ALL CONFIRMED against real mocap_pose/*_skeleton.npy
# (dict with keys ['timestamps','positions']; timestamps are ISO-8601 strings).
# ---------------------------------------------------------------------------
MOCAP_N_JOINTS = 20                                 # positions (F, 20, 3)   CONFIRMED
MOCAP_HZ = 125.0                                    # CONFIRMED: median dt = 0.00800 s (NOT 120)
MOCAP_UNITS = "mm"                                  # CONFIRMED: |max| ~ 1.6e3 -> millimetres
MOCAP_UP_AXIS = "z"                                 # CONFIRMED: Z is up (per-frame joint span ~1.4 m)
MOCAP_FLOOR_REFERENCED = True                       # CONFIRMED: min(Z) ~ 3-5 mm -> Z=0 is the floor
MOCAP_TO_M = 1e-3                                   # multiply raw positions by this to get metres

# !! IMPORTANT (CONFIRMED, changes the calibration strategy) !!
# MoCap XY is RECENTERED on the subject -- it is NOT absolute room position. Across 18 clips
# spanning 18 different subjects the body centroid sits at (-0.22, -0.21) m with std (0.14, 0.10);
# even a "walk" clip translates only 0.11 m over 100 s (subjects perform activities IN PLACE).
# Consequences:
#   * MoCap supplies POSE + absolute HEIGHT (Z), not room location.
#   * The mocap origin is effectively the subject's fixed standing spot, so it IS a consistent
#     per-scene room datum -- sensor extrinsics are expressed relative to that spot.
#   * Sensor poses CANNOT be triangulated from subject translation; use the RGB-D cameras (which
#     see both the room and the subject) as the bridge, anchored on the standing spot, plus the
#     radar's measured range/azimuth to the subject as extra constraints.
MOCAP_XY_RECENTERED = True
MOCAP_SUBJECT_SPOT_XY = (-0.22, -0.21)              # mean centroid XY (m) = the standing spot

# ---------------------------------------------------------------------------
# Per-modality sample rates for temporal sync.  Common time base = MoCap (125 Hz).
# OctoNet stamps every frame (vayyar 'timestamps' = datetime objects; mocap = ISO strings), so
# real timestamp-based resampling is possible -- unlike XRF55, which had no usable timestamps.
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = {
    "vayyar": 3.1,         # CONFIRMED: median dt 0.325 s over 3 clips -> 3.08-3.12 Hz
    "wifi": 75.0,          # FLAG-DATA (not yet measured)
    "depthCamera": 30.0,   # FLAG-DATA (Intel RealSense D455C; not yet measured)
    "mocap": 125.0,        # CONFIRMED: median dt 0.00800 s
}

# ---------------------------------------------------------------------------
# WiFi CSI (Intel AX200, 5 GHz): tensor (4 nodes, F, 2, 114 subcarriers).
# Bistatic / LOS handling is DEFERRED this pass (see plan); node poses are recorded but the
# renderer's bistatic mode + occlusion are future work.
# ---------------------------------------------------------------------------
WIFI_N_NODES = 4
WIFI_N_SUBCARRIERS = 114
WIFI_FREQ_GHZ = 5.0                                 # FLAG-DATA (AX200 5 GHz channel)

# ---------------------------------------------------------------------------
# RGB-D (3x Intel RealSense D455C, 640x480).  Intrinsics ship with the depth stream;
# FLAG-DATA: read the real K per camera from the dataset rather than assuming.
# ---------------------------------------------------------------------------
RGBD_N_CAMERAS = 3
RGBD_W, RGBD_H = 640, 480                            # CONFIRMED (paper)

# ---------------------------------------------------------------------------
# Sensor extrinsics: DERIVED in Phase 2 (octonet_calibrate) by registering each sensor to the
# MoCap world frame using real RGB-D depth + skeleton, seeded by priors from the OctoNet
# paper/supplement.  Identity placeholders until calibrated; per-scene results -> octonet_rig.json.
# ---------------------------------------------------------------------------
@dataclass
class SensorPose:
    name: str
    R: np.ndarray = field(default_factory=lambda: np.eye(3))    # sensor -> world rotation
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))  # sensor origin in world frame (m)
    calibrated: bool = False                                    # False until octonet_calibrate runs
