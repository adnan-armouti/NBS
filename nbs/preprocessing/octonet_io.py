"""OctoNet I/O — loaders for the modalities the NBS SFCW pipeline consumes.

Analogue of ``preprocessing.xrf55_io`` for the OctoNet dataset.  Every format detail below was
CONFIRMED by inspecting the real extracted files (see octonet_config for the constants).

  * Vayyar SFCW radar : ``vayyar_pickle/*.pickle`` -> dict{'timestamps','data'};
                        data (400,100,T) complex128, timestamps (T,1) of datetime.
  * MoCap skeleton    : ``mocap_pose/*_skeleton.npy`` -> dict{'timestamps','positions'};
                        positions (F,20,3) float64 in MILLIMETRES, Z-up, floor-referenced;
                        timestamps ISO-8601 strings; 125 Hz.
  * Master index      : ``cut_manual.csv`` -- joins every recording's per-node/per-modality paths
                        with start_time / cut_points / cut_timestamps (activity segmentation).

NOTE: the mocap .npy files were pickled by numpy 2.x. Under numpy 1.x (env ``mmir`` = 1.26) they
raise ``ModuleNotFoundError: numpy._core``; :func:`numpy2_compat` installs the module aliases that
make them loadable. It is called automatically by :func:`load_mocap`.
"""
import os, sys, glob, pickle
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import octonet_config as C


# ---------------------------------------------------------------------------
# numpy 2.x -> 1.x pickle compatibility
# ---------------------------------------------------------------------------
_COMPAT_DONE = False

def numpy2_compat():
    """Alias ``numpy._core*`` -> ``numpy.core*`` so numpy-2-pickled .npy files load under numpy 1.x.

    Note: numpy 1.26 exposes a PARTIAL ``numpy._core`` (no ``.multiarray``), so we must register
    each submodule explicitly rather than short-circuiting on ``hasattr(np, "_core")``."""
    global _COMPAT_DONE
    if _COMPAT_DONE:
        return
    for sub in ("", ".multiarray", ".umath", "._multiarray_umath", ".numeric"):
        name = "numpy._core" + sub
        if name in sys.modules:
            continue
        try:
            sys.modules[name] = __import__("numpy.core" + sub, fromlist=["*"])
        except Exception:
            pass
    _COMPAT_DONE = True


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------
def to_epoch(ts):
    """Normalize a timestamp array to float epoch seconds.

    Handles the two flavours OctoNet ships: ``datetime.datetime`` objects (Vayyar) and ISO-8601
    strings (MoCap).  Returns a 1-D float64 array."""
    a = np.asarray(ts).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=np.float64)
    x = a[0]
    if isinstance(x, str):
        return np.array([datetime.fromisoformat(s).timestamp() for s in a], dtype=np.float64)
    if hasattr(x, "timestamp"):
        return np.array([s.timestamp() for s in a], dtype=np.float64)
    return a.astype(np.float64)


def rate_hz(t):
    """Robust sample rate from epoch timestamps (median inter-sample dt)."""
    d = np.diff(np.asarray(t, dtype=float))
    d = d[d > 0]
    return float(1.0 / np.median(d)) if d.size else float("nan")


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def resolve(rel):
    """Resolve a cut_manual.csv relative path against the dataset root."""
    if not rel or (isinstance(rel, float) and np.isnan(rel)):
        return None
    rel = str(rel)
    return rel if os.path.isabs(rel) else os.path.join(C.OCTONET_ROOT, rel)


def vayyar_files():
    """All Vayyar SFCW clips (sorted)."""
    return sorted(glob.glob(os.path.join(C.OCTONET_ROOT, "vayyar_pickle", "*.pickle")))


def mocap_files():
    """All MoCap skeleton .npy files (sorted)."""
    return sorted(glob.glob(os.path.join(C.OCTONET_ROOT, "mocap_pose", "*_skeleton.npy")))


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_vayyar(path, frames_first=True, dtype=np.complex64):
    """Load one Vayyar SFCW clip.

    Returns ``(data, t)`` where ``data`` is (T, 400, 100) complex (frames_first=True, the natural
    per-frame layout) or the raw (400, 100, T), and ``t`` is (T,) float epoch seconds.

    The 100 axis is the swept CW tone index (63-67 GHz, see ``octonet_config.vayyar_freqs_hz``);
    feed it to ``mm3DPS.data.ra_utils.sfcw_range_compress`` (IFFT over frequency) for range."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    if not isinstance(d, dict) or "data" not in d:
        raise ValueError(f"unexpected vayyar pickle structure in {path}: {type(d)}")
    a = np.asarray(d["data"])                                   # (400, 100, T) complex128
    if a.ndim != 3 or a.shape[0] != C.VAY_N_PAIRS or a.shape[1] != C.VAY_N_FREQ:
        raise ValueError(f"unexpected vayyar shape {a.shape}, expected "
                         f"({C.VAY_N_PAIRS},{C.VAY_N_FREQ},T) in {path}")
    t = to_epoch(d.get("timestamps", []))
    if frames_first:
        a = np.transpose(a, (2, 0, 1))                          # -> (T, 400, 100)
    return a.astype(dtype, copy=False), t


def load_mocap(path, to_meters=True):
    """Load one MoCap skeleton clip.

    Returns ``(pos, t)``: ``pos`` (F, 20, 3) float64 — converted to METRES by default (the raw
    file is millimetres) — and ``t`` (F,) float epoch seconds.  World frame is Z-up and
    floor-referenced (Z=0 is the floor); 125 Hz."""
    numpy2_compat()
    md = np.load(path, allow_pickle=True)
    if isinstance(md, np.ndarray) and md.dtype == object and md.ndim == 0:
        md = md.item()
    if not isinstance(md, dict) or "positions" not in md:
        raise ValueError(f"unexpected mocap structure in {path}")
    p = np.asarray(md["positions"], dtype=np.float64)           # (F, 20, 3) millimetres
    if p.ndim != 3 or p.shape[1] != C.MOCAP_N_JOINTS or p.shape[2] != 3:
        raise ValueError(f"unexpected mocap shape {p.shape}, expected (F,{C.MOCAP_N_JOINTS},3)")
    if to_meters:
        p = p * C.MOCAP_TO_M
    return p, to_epoch(md.get("timestamps", []))


def load_index(csv_path=None):
    """Load ``cut_manual.csv`` (the master join table) as a pandas DataFrame.

    One row per recording; carries every modality's relative path plus ``start_time``,
    ``cut_points`` / ``cut_ratio`` / ``cut_timestamps`` (per-repetition activity segmentation).
    Use :func:`resolve` on the ``*_data_path`` columns to get absolute paths."""
    import pandas as pd
    return pd.read_csv(csv_path or C.CUT_MANUAL_CSV)


def pairs_to_tx_rx(a, order=None):
    """Reshape the flat 400-pair axis into (20 Tx, 20 Rx).

    ``a``: (..., 400, ...) with the pair axis at position -2 for the natural (T,400,100) layout.
    Applied to the LAST-BUT-ONE axis. ``order`` defaults to ``octonet_config.VAY_PAIR_ORDER``.

    FLAG: the pair index order (tx-major ``p = tx*20 + rx`` vs rx-major) is still an ASSUMPTION --
    it affects only the azimuth/elevation assignment in RA formation, not range. Verify against the
    Vayyar vSDK antenna map (or empirically, by which order yields a coherent array manifold)."""
    order = order or C.VAY_PAIR_ORDER
    a = np.asarray(a)
    ax = a.ndim - 2
    if a.shape[ax] != C.VAY_N_PAIRS:
        raise ValueError(f"axis {ax} is {a.shape[ax]}, expected {C.VAY_N_PAIRS}")
    new = a.shape[:ax] + (C.VAY_N_TX, C.VAY_N_RX) + a.shape[ax + 1:]
    out = a.reshape(new)
    if order == "rx_major":                                     # stored rx-fastest -> swap
        out = np.swapaxes(out, ax, ax + 1)
    elif order != "tx_major":
        raise ValueError(f"unknown pair order {order!r}")
    return out


def clip_key(path):
    """``<timestamp>_node_1_..._subject_N_activity_A_trial_K`` -> that stem (join key across
    modalities).  Works for any OctoNet per-clip filename."""
    return os.path.splitext(os.path.basename(path))[0]


def load_pickle_stream(path):
    """Every object in a pickle file that was written by repeated ``pickle.dump`` calls (OctoNet's UWB files:
    one dict per frame appended; ``pickle.load`` alone returns only the first frame). Mirrors OctoNet's
    ``load_data_from_pickle``."""
    import pickle
    out = []
    with open(path, "rb") as fh:
        while True:
            try:
                out.append(pickle.load(fh))
            except EOFError:
                break
    return out


def load_uwb(path):
    """One UWB recording: ``(frames (F,1535) float64, t (F,) epoch s)``. F ~ 1500-1700 at 17.07 Hz.
    The frame is the RF-sampled X4 waveform (23.328 GS/s, carrier ~7.2 GHz), bin = c / (2 fs) = 6.43 mm."""
    entries = load_pickle_stream(path)
    if len(entries) == 1 and isinstance(entries[0], list):
        entries = entries[0]
    frames = np.stack([np.asarray(e["frame"], dtype=np.float64) for e in entries])
    return frames, to_epoch([e["timestamp"] for e in entries])
