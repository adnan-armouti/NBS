"""XRF55 I/O: clip discovery, RGB-frame extraction (ffmpeg), mmWave RA/RD loading.

XRF55 layout on disk:
  videos/open/sceneN/<subj>/<subj>_<action>.mkv        (RGB-only, 1280x720)
  RF/SceneN/{mmWave,WiFi,RFID}/<subj>_<action>_<trial>.npy
  pose/XRF55Videos_Res/open/sceneN/<subj>/<subj>_<action>.json
Note: video files are per (subject, action); RF files are per (subject, action, trial).
"""
import os, glob, subprocess, json
import numpy as np
from . import xrf55_config as C


def list_mmwave(scene="Scene1"):
    """Return sorted list of mmWave .npy paths for a scene."""
    return sorted(glob.glob(os.path.join(C.XRF55_ROOT, "RF", scene, "mmWave", "*.npy")))


def parse_key(path):
    """'01_02_03.npy' -> dict(subject='01', action='02', trial='03')."""
    s, a, t = os.path.splitext(os.path.basename(path))[0].split("_")
    return {"subject": s, "action": a, "trial": t}


def load_mmwave(path):
    """Load an XRF55 mmWave file -> dict with RA and RD stacks.

    Returns:
      ra: (frames=17, range=256, azimuth=64)   magnitude
      rd: (frames=17, range=256, doppler=64)    magnitude
    """
    a = np.load(path)
    if a.ndim == 4:
        a = a[0]                       # (17, 256, 128)
    assert a.shape == (C.MMW_N_FRAMES, C.MMW_N_RANGE, 128), f"unexpected mmWave shape {a.shape}"
    rd = a[:, :, C.MMW_DOPPLER_SLICE[0]:C.MMW_DOPPLER_SLICE[1]]
    ra = a[:, :, C.MMW_AZIMUTH_SLICE[0]:C.MMW_AZIMUTH_SLICE[1]]
    return {"ra": ra.astype(np.float32), "rd": rd.astype(np.float32)}


def radar_ra(path, reduce="mean"):
    """Reduce the 17-frame RA stack -> (range=256, azimuth=64).

    reduce: 'mean' (suppress motion, emphasize static scene), 'median' (robust),
    'max' (peak hold), 'min' (most static / clutter only).
    """
    ra = load_mmwave(path)["ra"]
    return {"mean": ra.mean(0), "median": np.median(ra, 0),
            "max": ra.max(0), "min": ra.min(0)}[reduce]


def radar_ra_mean(path):
    return radar_ra(path, "mean")


def video_path_for(mmwave_path, scene="Scene1"):
    """Map an RF mmWave path to its RGB .mkv (per subject/action; trial dropped)."""
    k = parse_key(mmwave_path)
    scl = scene.lower()            # 'Scene1' -> 'scene1'
    return os.path.join(C.XRF55_ROOT, "videos", "open", scl, k["subject"],
                        f"{k['subject']}_{k['action']}.mkv")


def extract_frame_at(mkv_path, out_dir, index):
    """Extract the exact video frame at 0-based `index` (so it matches pose[index]).
    Returns the PNG path."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"frame_{index:05d}.png")
    if not os.path.exists(out):
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", mkv_path,
               "-vf", f"select=eq(n\\,{index})", "-vframes", "1", out]
        subprocess.run(cmd, check=True)
    return out


def extract_frames(mkv_path, out_dir, n_frames=1, every=None):
    """Extract RGB frames from an .mkv using ffmpeg. Returns sorted PNG paths.

    n_frames: if every is None, grab a single middle frame (n_frames=1) or the
    first n_frames evenly spaced frames. Static-scene init only needs ~1.
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if existing:
        return existing
    if n_frames == 1:
        # middle frame: use the thumbnail filter to pick a representative frame
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", mkv_path,
               "-vf", "thumbnail", "-frames:v", "1",
               os.path.join(out_dir, "frame_000.png")]
    else:
        fps_expr = every if every else f"1/{max(1, 999)}"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", mkv_path,
               "-vf", f"select='not(mod(n\\,{every or 30}))'", "-vsync", "vfr",
               os.path.join(out_dir, "frame_%03d.png")]
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(out_dir, "*.png")))


def load_pose(mmwave_path, scene="Scene1"):
    """Load the 2D pose json for a clip (list of frames, 25 keypoints each)."""
    k = parse_key(mmwave_path); scl = scene.lower()
    p = os.path.join(C.XRF55_ROOT, "pose", "XRF55Videos_Res", "open", scl,
                     k["subject"], f"{k['subject']}_{k['action']}.json")
    return json.load(open(p)) if os.path.exists(p) else None
