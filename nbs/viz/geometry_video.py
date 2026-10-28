"""Geometry videos v2 (OctoNet Scene 1): the static scaffold and the dynamic SMPL person, in the 4DPS style
(dark background, grey scene points, warm person points, skeleton on top, metre grid, radar marker + range ring).

    cd nbs && python -m viz.geometry_video --rec 20240518135416 [--seconds 25] [--fps 10] [--workers 16]
    cd nbs && python -m viz.geometry_video --overview

Left panel: top-down X-Y (MoCap world, metres); right: 3-D perspective, slowly orbiting. Person = SMPL fitted to the
OptiTrack skeleton (mm4DPS.octonet_person), preferably the 10 Hz refit under $NBS_DATA_OUT/person10hz_S1/<rec>/,
else the Vayyar-rate (3.1 Hz) sequence under 4dps_person_S1/<rec>/ (nearest frame). Frames are rendered to
/ephemeral/scratch/nbs/frames_<rec>/ with multiprocessing and encoded with ffmpeg (libx264, yuv420p, crf 20).
Output: ../viz/geometry_v2/<rec>_<activity>.mp4 and scene_overview.mp4.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from multiprocessing import Pool
import numpy as np

from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S

DATA = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", "geometry_v2")
SCRATCH = "/ephemeral/scratch/nbs" if os.path.isdir("/ephemeral/scratch") else os.path.join(DATA, "scratch")
BG, GRID, SCENE_C, PERSON_C, BONE_C = "#12121a", "#2a2a36", "#aaaaaa", "#ffc83c", "#ff5a2d"
CAM_C = {1: "#e8544f", 3: "#4fc46a", 5: "#4f8de8"}
NODE_C = {1: "#e8544f", 2: "#f0a040", 3: "#4fc46a", 4: "#b06ee8", 5: "#4f8de8"}
# SMPL kinematic tree (24 joints), parents
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SFCW_RMAX = 3.71
W, H, DPI = 1920, 1080, 100


# ----------------------------------------------------------------------------- data
def load_scene(max_points=40000, seed=0):
    d = os.path.join(DATA, "4dps_scaffold_S1")
    pcl = np.load(os.path.join(d, "pcl.npy")); src = np.load(os.path.join(d, "source_node.npy"))
    if len(pcl) > max_points:
        k = np.random.default_rng(seed).choice(len(pcl), max_points, replace=False); pcl, src = pcl[k], src[k]
    return pcl[:, :3].astype(np.float64), src


def load_person(rec):
    for d in (os.path.join(DATA, "person10hz_S1", rec), os.path.join(DATA, "4dps_person_S1", rec), os.path.join(DATA, "person", rec)):
        p = os.path.join(d, "person_sequence.npz")
        if os.path.exists(p):
            z = np.load(p); return dict(v=z["vertices_world"], j=z["joints_world"], t=z["t"], valid=z["valid"], src=d)
    raise FileNotFoundError(rec)


def activity_of(rec):
    df = IO.load_index()
    m = df["node_1_depthCamera_data_path"].astype(str).str.contains(rec) | df["node_1_uwb_data_path"].astype(str).str.contains(rec) | df["node_1_vayyar_data_path"].astype(str).str.contains(rec)
    return str(df[m].iloc[0]["activity"]) if m.any() else "unknown"


def sensor_marks():
    e = json.load(open(os.path.join(DATA, "calib_S1", "extrinsics.json")))
    R = np.array(e["vayyar"]["R_wc"]); t = np.array(e["vayyar"]["t_w"]); bore = R[:, 2].copy(); bore[2] = 0; bore /= np.linalg.norm(bore)
    return dict(vayyar_pos=t, vayyar_bore=bore, wifi_tx=S.WIFI_TX_APPROX, wifi_rx=S.WIFI_RX_APPROX, nodes=S.NODE_POS_S1)


# ----------------------------------------------------------------------------- drawing
def _style(ax, three_d=False):
    ax.set_facecolor(BG)
    if three_d:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.07, 0.07, 0.10, 1.0)); axis._axinfo["grid"]["color"] = GRID
        ax.tick_params(colors="#9a9ab0", labelsize=8)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.label.set_color("#c8c8d8")
    else:
        ax.grid(color=GRID, lw=0.6); ax.tick_params(colors="#9a9ab0", labelsize=9)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.xaxis.label.set_color("#c8c8d8"); ax.yaxis.label.set_color("#c8c8d8")


def draw_marks_2d(ax, mk):
    sq = S.SENSING_AREA_M / 2
    ax.plot([-sq, sq, sq, -sq, -sq], [-sq, -sq, sq, sq, -sq], color="#55aa88", lw=0.8, ls="--", alpha=0.8)
    th = np.linspace(0, 2 * np.pi, 200); p = mk["vayyar_pos"]
    ax.plot(p[0] + SFCW_RMAX * np.cos(th), p[1] + SFCW_RMAX * np.sin(th), color="#ff2d55", lw=0.7, ls=":", alpha=0.9)
    ax.annotate("", xy=(p[0] + 1.2 * mk["vayyar_bore"][0], p[1] + 1.2 * mk["vayyar_bore"][1]), xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="-|>", color="#ff2d55", lw=1.4))
    for n, q in mk["nodes"].items():
        ax.plot(q[0], q[1], "s", ms=9, color=NODE_C[n], mec="white", mew=0.6)
        ax.text(q[0] + 0.12, q[1] + 0.12, f"node {n}", color="white", fontsize=9)
    ax.plot(mk["wifi_tx"][0], mk["wifi_tx"][1], "^", ms=9, color="#ff6b6b", mec="white", mew=0.5)
    ax.text(mk["wifi_tx"][0] + 0.1, mk["wifi_tx"][1] - 0.3, "WiFi Tx~", color="#ff9b9b", fontsize=8)
    for k, q in mk["wifi_rx"].items():
        ax.plot(q[0], q[1], "v", ms=8, color="#7fb2ff", mec="white", mew=0.5); ax.text(q[0] + 0.1, q[1] - 0.3, f"Rx{k}~", color="#a8c8ff", fontsize=8)
    ax.plot(0, 0, "*", ms=11, color="white", mec="black", mew=0.4)


def draw_marks_3d(ax, mk):
    th = np.linspace(0, 2 * np.pi, 200); p = mk["vayyar_pos"]
    ax.plot(p[0] + SFCW_RMAX * np.cos(th), p[1] + SFCW_RMAX * np.sin(th), np.zeros_like(th), color="#ff2d55", lw=0.6, ls=":", alpha=0.8)
    for n, q in mk["nodes"].items():
        ax.scatter([q[0]], [q[1]], [q[2]], marker="s", s=40, color=NODE_C[n], edgecolor="white", linewidth=0.5, depthshade=False)
        ax.plot([q[0], q[0]], [q[1], q[1]], [0, q[2]], color=NODE_C[n], lw=0.6, alpha=0.6)
        ax.text(q[0], q[1], q[2] + 0.15, f"node {n}", color="white", fontsize=8)
    ax.quiver(p[0], p[1], p[2], mk["vayyar_bore"][0], mk["vayyar_bore"][1], 0, length=1.2, color="#ff2d55", lw=1.2, arrow_length_ratio=0.25)
    sq = S.SENSING_AREA_M / 2
    ax.plot([-sq, sq, sq, -sq, -sq], [-sq, -sq, sq, sq, -sq], [0, 0, 0, 0, 0], color="#55aa88", lw=0.7, ls="--", alpha=0.8)


def render_frame(args):
    (i, out_png, scene, src, colour_by_cam, person_v, person_j, title, subtitle, azim, mk, show_person) = args
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax2 = fig.add_axes([0.04, 0.07, 0.42, 0.84]); ax3 = fig.add_axes([0.48, 0.02, 0.52, 0.94], projection="3d")
    _style(ax2); _style(ax3, True)
    sc_col = np.array([CAM_C[int(s)] for s in src]) if colour_by_cam else SCENE_C
    ax2.scatter(scene[:, 0], scene[:, 1], s=0.8, c=sc_col, alpha=0.6, linewidths=0, rasterized=True)
    if show_person and person_v is not None:
        ax2.scatter(person_v[:, 0], person_v[:, 1], s=2.0, c=PERSON_C, alpha=0.9, linewidths=0)
        for j, pj in enumerate(SMPL_PARENTS):
            if pj >= 0: ax2.plot([person_j[j, 0], person_j[pj, 0]], [person_j[j, 1], person_j[pj, 1]], color=BONE_C, lw=1.6)
        ax2.plot(person_j[:, 0], person_j[:, 1], "o", ms=2.5, color="white")
    draw_marks_2d(ax2, mk)
    ax2.set_xlim(-4.3, 4.6); ax2.set_ylim(-4.6, 4.3); ax2.set_aspect("equal"); ax2.set_xlabel("X [m]  (MoCap world)"); ax2.set_ylabel("Y [m]")
    ax2.set_title("top-down", color="white", fontsize=11, loc="left")
    ax3.scatter(scene[:, 0], scene[:, 1], scene[:, 2], s=1.1, c=sc_col, alpha=0.7, linewidths=0, depthshade=False, rasterized=True)
    if show_person and person_v is not None:
        ax3.scatter(person_v[:, 0], person_v[:, 1], person_v[:, 2], s=2.0, c=PERSON_C, alpha=0.95, linewidths=0, depthshade=False)
        for j, pj in enumerate(SMPL_PARENTS):
            if pj >= 0: ax3.plot([person_j[j, 0], person_j[pj, 0]], [person_j[j, 1], person_j[pj, 1]], [person_j[j, 2], person_j[pj, 2]], color=BONE_C, lw=1.4)
    draw_marks_3d(ax3, mk)
    xx, yy = np.meshgrid([-4.3, 4.6], [-4.6, 4.3]); ax3.plot_surface(xx, yy, np.zeros_like(xx) - 0.02, color="#20202e", alpha=0.18, linewidth=0)
    ax3.set_xlim(-4.3, 4.6); ax3.set_ylim(-4.6, 4.3); ax3.set_zlim(0, 3.0); ax3.set_box_aspect((8.9, 8.9, 3.0))
    ax3.view_init(elev=28, azim=azim); ax3.set_xlabel("X [m]"); ax3.set_ylabel("Y [m]"); ax3.set_zlabel("Z [m]")
    ax3.set_title("3-D view", color="white", fontsize=11, loc="left")
    fig.text(0.02, 0.965, title, color="white", fontsize=14, weight="bold"); fig.text(0.02, 0.935, subtitle, color="#c8c8d8", fontsize=10)
    fig.text(0.02, 0.012, "grey = static scaffold (3 depth cameras, calibrated poses + depth scales)    gold = SMPL person fitted to the OptiTrack skeleton    "
                         "red = Vayyar/UWB at node 1: boresight + 3.71 m SFCW ring    squares = node rigid bodies    ~ = schematic WiFi Tx/Rx (±0.5 m)    star = subject spot",
             color="#9a9ab0", fontsize=9)
    fig.savefig(out_png, dpi=DPI, facecolor=BG); plt.close(fig)
    return i


def encode(frames_dir, output, fps):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames_dir, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", output], check=True)


# ----------------------------------------------------------------------------- videos
def recording_video(rec, seconds, fps, workers, n_person=2000, keep_frames=False):
    scene, src = load_scene(); mk = sensor_marks(); act = activity_of(rec); per = load_person(rec)
    first = int(np.argmax(per["valid"])) if per["valid"].any() else 0          # start where the MoCap covers the frames
    t0 = per["t"][first]; times = np.arange(0, min(seconds, per["t"][-1] - t0), 1.0 / fps)
    rng = np.random.default_rng(0); vsel = rng.choice(per["v"].shape[1], n_person, replace=False)
    frames_dir = os.path.join(SCRATCH, f"frames_{rec}"); shutil.rmtree(frames_dir, ignore_errors=True); os.makedirs(frames_dir)
    src_rate = 1.0 / np.median(np.diff(per["t"])); jobs = []
    for i, tt in enumerate(times):
        k = int(np.argmin(np.abs(per["t"] - (t0 + tt)))); ok = bool(per["valid"][k])
        pv = per["v"][k][vsel] if ok else None; pj = per["j"][k] if ok else None
        azim = -60 + 25 * np.sin(2 * np.pi * tt / 40.0)
        title = f"OctoNet Scene 1  |  recording {rec}  |  activity: {act}"
        sub = f"t = {tt:6.2f} s   frame {i:4d}/{len(times)}   person source {os.path.basename(os.path.dirname(per['src']))} @ {src_rate:.1f} Hz" + ("" if ok else "   (no valid MoCap frame)")
        jobs.append((i, os.path.join(frames_dir, f"{i:05d}.png"), scene, src, False, pv, pj, title, sub, azim, mk, True))
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(render_frame, jobs, chunksize=4): pass
    os.makedirs(OUT_DIR, exist_ok=True); out = os.path.join(OUT_DIR, f"{rec}_{act}.mp4"); encode(frames_dir, out, fps)
    if not keep_frames: shutil.rmtree(frames_dir, ignore_errors=True)
    print("wrote", out, len(times), "frames"); return out


def overview_video(fps, workers, rec="20240518135416"):
    scene, src = load_scene(); mk = sensor_marks(); per = load_person(rec); act = activity_of(rec)
    rng = np.random.default_rng(0); vsel = rng.choice(per["v"].shape[1], 2000, replace=False)
    k = len(per["t"]) // 2; pv, pj = per["v"][k][vsel], per["j"][k]
    n = 15 * fps; frames_dir = os.path.join(SCRATCH, "frames_overview"); shutil.rmtree(frames_dir, ignore_errors=True); os.makedirs(frames_dir)
    jobs = []
    for i in range(n):
        phase = i / n; azim = -60 + 360 * phase
        if phase < 1 / 3: cb, sp, sub = False, False, "1/3: the static scaffold alone (temporal-median depth of 3 cameras, person column cut, 5 cm voxels)"
        elif phase < 2 / 3: cb, sp, sub = True, False, "2/3: the same points coloured by source camera: red node 1, green node 3, blue node 5 (calibrated poses + depth scales)"
        else: cb, sp, sub = False, True, f"3/3: the SMPL person added (recording {rec}, {act}, one frame)"
        jobs.append((i, os.path.join(frames_dir, f"{i:05d}.png"), scene, src, cb, pv, pj, "OctoNet Scene 1 geometry: scaffold, cameras, person", sub, azim, mk, sp))
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(render_frame, jobs, chunksize=4): pass
    os.makedirs(OUT_DIR, exist_ok=True); out = os.path.join(OUT_DIR, "scene_overview.mp4"); encode(frames_dir, out, fps)
    shutil.rmtree(frames_dir, ignore_errors=True); print("wrote", out, n, "frames"); return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rec", action="append", default=[]); p.add_argument("--overview", action="store_true")
    p.add_argument("--seconds", type=float, default=25.0); p.add_argument("--fps", type=int, default=10); p.add_argument("--workers", type=int, default=16)
    p.add_argument("--keep-frames", action="store_true")
    a = p.parse_args()
    for rec in a.rec: recording_video(rec, a.seconds, a.fps, a.workers, keep_frames=a.keep_frames)
    if a.overview: overview_video(a.fps, a.workers)


if __name__ == "__main__":
    main()
