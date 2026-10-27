"""OctoNet temporal synchronization.

XRF55 had NO usable per-frame timestamps, so the old pipeline collapsed radar time (peak-hold over
17 frames) and picked a single "representative" video frame. OctoNet stamps every frame, so we can
do real timestamp-based alignment across modalities running at very different rates:

    Vayyar SFCW  ~3.1 Hz   (datetime objects)      <- the SLOW modality; drives the frame grid
    MoCap        125.0 Hz  (ISO-8601 strings)      <- interpolated onto the Vayyar frame times
    RGB-D        ~30 Hz    (FLAG-DATA)
    WiFi CSI     ~75 Hz    (FLAG-DATA)

Design: the radar is the coarsest stream and cannot be meaningfully interpolated (0.32 s between
frames), so it defines the reference grid; every finer modality is resampled ONTO the Vayyar frame
timestamps -- poses by linear interpolation, discrete/array streams by nearest neighbour.

``cut_manual.csv`` additionally carries ``cut_timestamps`` marking each repetition of the activity
within a recording, which :func:`segment_indices` turns into per-repetition index ranges.
"""
import os, sys, ast
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import octonet_config as C
from preprocessing import octonet_io as IO


# ---------------------------------------------------------------------------
# cut_manual.csv timing fields
# ---------------------------------------------------------------------------
def parse_cut_timestamps(s):
    """Parse the ``cut_timestamps`` cell into epoch seconds.

    Stored as a stringified list with DOT-separated time fields, e.g.
    ``"['2024-05-18 13.44.02.848', ...]"`` -> ``[epoch, ...]``."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.zeros(0)
    if isinstance(s, str):
        try:
            s = ast.literal_eval(s)
        except Exception:
            return np.zeros(0)
    out = []
    for x in s:
        x = str(x).strip()
        d, _, t = x.partition(" ")
        parts = t.split(".")
        if len(parts) >= 3:                       # HH.MM.SS[.mmm]
            hh, mm, ss = parts[0], parts[1], parts[2]
            frac = parts[3] if len(parts) > 3 else "0"
            iso = f"{d}T{hh}:{mm}:{ss}.{frac:0<6.6}"
        else:
            iso = x.replace(" ", "T")
        try:
            out.append(datetime.fromisoformat(iso).timestamp())
        except Exception:
            pass
    return np.asarray(out, dtype=float)


def parse_start_time(s):
    """``start_time`` cell (e.g. '2024-05-18 13:43:59.273') -> epoch seconds."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return float("nan")
    try:
        return datetime.fromisoformat(str(s).strip().replace(" ", "T")).timestamp()
    except Exception:
        return float("nan")


def segment_indices(t, cut_epochs):
    """Per-repetition index ranges for a stream with timestamps ``t``.

    Returns a list of ``(lo, hi)`` half-open index ranges delimited by ``cut_epochs``."""
    t = np.asarray(t, dtype=float)
    cuts = np.asarray(cut_epochs, dtype=float)
    if t.size == 0 or cuts.size == 0:
        return [(0, int(t.size))]
    edges = np.concatenate(([t[0]], cuts[(cuts > t[0]) & (cuts < t[-1])], [t[-1] + 1e-6]))
    idx = np.searchsorted(t, edges)
    return [(int(a), int(b)) for a, b in zip(idx[:-1], idx[1:]) if b > a]


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------
def interp_poses(t_src, P, t_dst):
    """Linearly interpolate skeleton poses ``P`` (F,J,3) from times ``t_src`` onto ``t_dst``.

    Returns ``(P_dst (D,J,3), valid (D,) bool)``; ``valid`` is False where ``t_dst`` falls outside
    the source coverage (those rows are clamped to the nearest endpoint)."""
    t_src = np.asarray(t_src, float); t_dst = np.asarray(t_dst, float)
    P = np.asarray(P, float)
    valid = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    flat = P.reshape(len(t_src), -1)
    out = np.empty((len(t_dst), flat.shape[1]), dtype=float)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_dst, t_src, flat[:, c])
    return out.reshape((len(t_dst),) + P.shape[1:]), valid


def nearest_index(t_src, t_dst, max_dt=None):
    """Nearest-neighbour index mapping ``t_dst -> t_src`` (for streams that must not be blended).

    Returns ``(idx (D,), ok (D,) bool)``; ``ok`` is False where the nearest match is further than
    ``max_dt`` seconds away."""
    t_src = np.asarray(t_src, float); t_dst = np.asarray(t_dst, float)
    j = np.searchsorted(t_src, t_dst).clip(1, len(t_src) - 1)
    left, right = t_src[j - 1], t_src[j]
    idx = np.where(np.abs(t_dst - left) <= np.abs(right - t_dst), j - 1, j)
    ok = np.ones(len(t_dst), bool)
    if max_dt is not None:
        ok = np.abs(t_src[idx] - t_dst) <= max_dt
    return idx, ok


# ---------------------------------------------------------------------------
# clip-level sync
# ---------------------------------------------------------------------------
def sync_clip(row, want_mocap=True):
    """Load one recording (a ``cut_manual.csv`` row) with every stream aligned to the Vayyar grid.

    Returns a dict:
      ``V``     (T,400,100) complex Vayyar frames   -- the reference grid
      ``t``     (T,) epoch seconds of those frames
      ``pose``  (T,20,3) MoCap poses in METRES interpolated at ``t`` (None if unavailable)
      ``pose_valid`` (T,) bool -- False where ``t`` lies outside MoCap coverage
      ``segments``   list of (lo,hi) index ranges into the Vayyar grid, one per activity repetition
      ``meta``  dict(subject, activity, start_time, overlap_s, offset_s)
    """
    vp = IO.resolve(row["node_1_vayyar_data_path"])
    V, t = IO.load_vayyar(vp)

    pose = pose_valid = None
    offset = overlap = float("nan")
    if want_mocap and isinstance(row.get("mocap_data_path"), str):
        stem = os.path.splitext(os.path.basename(row["mocap_data_path"]))[0]
        npy = os.path.join(C.OCTONET_ROOT, "mocap_pose", stem + "_skeleton.npy")
        if os.path.exists(npy):
            P, tm = IO.load_mocap(npy)                       # metres, Z-up, floor-referenced
            pose, pose_valid = interp_poses(tm, P, t)
            offset = float(t[0] - tm[0])
            overlap = float(min(t[-1], tm[-1]) - max(t[0], tm[0]))

    cuts = parse_cut_timestamps(row.get("cut_timestamps"))
    return {
        "V": V, "t": t, "pose": pose, "pose_valid": pose_valid,
        "segments": segment_indices(t, cuts),
        "meta": {
            "subject": row.get("user_id"), "activity": row.get("activity"),
            "start_time": parse_start_time(row.get("start_time")),
            "vayyar_hz": IO.rate_hz(t), "offset_s": offset, "overlap_s": overlap,
            "path": vp,
        },
    }
