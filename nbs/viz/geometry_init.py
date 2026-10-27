"""Viz group 2 -- initialized scene + person geometry on OctoNet (Scene 1) and how it is built.

Run from nbs/:  python -m viz.geometry_init      (outputs -> <repo>/viz/geometry_init/)

Scene scaffold: RGB-D of nodes 1/3/5 (8-bit depth, 0..10 m), per-pixel temporal MEDIAN over ~30
frames of the 'walk' clip (the subject walks in place at the origin, so the median suppresses the
moving person), back-projected with nominal D455 intrinsics, placed with the ASSUMED node poses
(MoCap rigid-body position, horizontal optical axis aimed at the origin), person column cut,
voxel-downsampled, PCA normals.  Person: the 20-joint MoCap skeleton (npy == OptiTrack CSV bones).
"""
from __future__ import annotations
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch

from preprocessing import octonet_scene as S, octonet_io as IO, octonet_config as C

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "viz", "geometry_init")
os.makedirs(OUT, exist_ok=True)
POSE = S.pose_source()
POSE_LABEL = {"assumed": "assumed extrinsics (node rigid-body position + look-at origin; no calibration)",
              "calibrated": "calibrated extrinsics (octonet_calibrate: floor pitch/roll, MoCap-person yaw + per-camera depth scale; position = node rigid body + floor height)"}[POSE]
DATA_OUT = os.path.join(os.environ.get("NBS_DATA_OUT", os.path.join(REPO, "data", "OctoNet")), f"scaffold_S1_{POSE}")
os.makedirs(DATA_OUT, exist_ok=True)

CAM_NODES = (1, 3, 5)
N_FRAMES = 30
STRIDE = 3
VOXEL = 0.05
PERSON_R, PERSON_ZMAX = 0.7, 2.2
CAM_COLORS = {1: "tab:red", 3: "tab:green", 5: "tab:blue"}
NOTE = POSE_LABEL

# npy joint order == OptiTrack CSV bone order (verified: 0.0 mm difference on frame 0 of the 'sit' take)
JOINTS = ["Hip", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
          "RFArm", "RHand", "LThigh", "LShin", "LFoot", "LToe", "RThigh", "RShin", "RFoot", "RToe"]
BONES = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11),
         (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]

captions = []


# --------------------------------------------------------------------------- scaffold
def median_depth(node: int, rec: str):
    clip = S.depth_clip_dir(node, rec)
    frames = S.depth_frames(clip)
    sel = [frames[i] for i in np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)]
    stack = np.stack([S.load_depth_m_calibrated(p, node) for p in sel])   # (F,H,W), per-camera depth scale divided out
    stack[stack <= 0] = np.nan
    med = np.nanmedian(stack, axis=0)
    valid = np.isfinite(stack).mean(0) >= 0.5
    med[~valid] = 0.0
    med = np.nan_to_num(med)
    return med, os.path.basename(clip), [os.path.basename(p) for p in sel]


def person_column_mask_world(w: np.ndarray) -> np.ndarray:
    r = np.hypot(w[:, 0], w[:, 1])
    return (r < PERSON_R) & (w[:, 2] > 0.0) & (w[:, 2] < PERSON_ZMAX)


def pixel_world(depth: np.ndarray, node: int, stride: int):
    """World coords for every (strided) pixel; returns (pts_world (N,3), index image (h,w) of row ids or -1)."""
    K = S.d455_nominal_intrinsics()
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    ok = (z > 0.3) & (z < 9.5)
    x = (u - K.cx) / K.fx * z
    y = (v - K.cy) / K.fy * z
    cam = np.stack([x[ok], y[ok], z[ok]], 1)
    w = S.cam_to_world(cam, node)
    idx = -np.ones(z.shape, dtype=int)
    idx[ok] = np.arange(len(w))
    return cam, w, idx


def voxel_down(pts: np.ndarray, voxel: float) -> np.ndarray:
    key = np.floor(pts / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = inv.max() + 1
    acc = np.zeros((n, 3)); cnt = np.zeros(n)
    np.add.at(acc, inv, pts); np.add.at(cnt, inv, 1)
    return acc / cnt[:, None]


def pca_normals(pts: np.ndarray, k: int = 20) -> np.ndarray:
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(k))
        n = np.asarray(pc.normals)
    except Exception:
        from scipy.spatial import cKDTree
        t = cKDTree(pts); _, nb = t.query(pts, k)
        n = np.zeros_like(pts)
        for i in range(len(pts)):
            q = pts[nb[i]] - pts[nb[i]].mean(0)
            n[i] = np.linalg.svd(q, full_matrices=False)[2][-1]
    # orient toward the origin (sensors sit around the subject spot)
    flip = np.einsum("ij,ij->i", n, -pts) < 0
    n[flip] *= -1
    return n


def build_scaffold():
    per_cam = {}
    meta = {"recording": S.REC_WALK, "frames_per_camera": N_FRAMES, "stride": STRIDE, "voxel_m": VOXEL,
            "person_cut": {"radius_m": PERSON_R, "z_max_m": PERSON_ZMAX},
            "camera_pose": NOTE, "intrinsics": "nominal D455 87x58 deg FoV, 640x480",
            "depth_decode": "8-bit PNG, linear 0..10 m (OctoNet loader png_to_depth)", "cameras": {}}
    fused, src = [], []
    for n in CAM_NODES:
        med, clip, sel = median_depth(n, S.REC_WALK)
        cam, w, idx = pixel_world(med, n, STRIDE)
        pm = person_column_mask_world(w)
        keep = (~pm) & (w[:, 2] > -0.1) & (w[:, 2] < 3.0)
        per_cam[n] = dict(med=med, cam=cam, w=w, idx=idx, person=pm, keep=keep)
        fused.append(w[keep]); src.append(np.full(keep.sum(), n))
        R, t, _ = S.camera_pose(n)
        meta["cameras"][f"node_{n}"] = {"clip": clip, "frames": sel, "R_wc": R.tolist(), "t_w": t.tolist(),
                                        "points_kept": int(keep.sum())}
    fused = np.concatenate(fused); src = np.concatenate(src)
    # voxel-down per camera so the colour survives
    down, dsrc = [], []
    for n in CAM_NODES:
        d = voxel_down(fused[src == n], VOXEL); down.append(d); dsrc.append(np.full(len(d), n))
    down = np.concatenate(down); dsrc = np.concatenate(dsrc)
    normals = pca_normals(down)
    pcl = np.concatenate([down, normals], 1).astype(np.float32)
    np.save(os.path.join(DATA_OUT, "pcl.npy"), pcl)
    np.save(os.path.join(DATA_OUT, "source_node.npy"), dsrc.astype(np.int8))
    meta["n_points"] = int(len(pcl)); meta["pcl"] = "pcl.npy (N,6): xyz [m, MoCap frame, Z up] + unit normal"
    json.dump(meta, open(os.path.join(DATA_OUT, "scaffold_metadata.json"), "w"), indent=1)
    return per_cam, down, dsrc, normals, meta


def draw_layout(ax):
    h = S.SENSING_AREA_M / 2
    ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fill=False, ls="--", lw=1.2, ec="k", label="4.1 m sensing area (paper)"))
    ax.add_patch(Circle((0, 0), PERSON_R, fill=False, ls=":", lw=1.2, ec="m", label=f"person column cut (r<{PERSON_R} m)"))
    for n, p in S.NODE_POS_S1.items():
        c = CAM_COLORS.get(n, "0.4")
        ax.plot(p[0], p[1], "s", ms=9, mec="k", mfc=c)
        ax.annotate(f"Node-{n}" + (" (RGB-D)" if n in CAM_NODES else ""), (p[0], p[1]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        if n in CAM_NODES:
            d = -p[:2] / np.linalg.norm(p[:2]); ax.annotate("", xy=p[:2] + 0.6 * d, xytext=p[:2], arrowprops=dict(arrowstyle="->", color=c))


def fig_scaffold(down, dsrc):
    fig, ax = plt.subplots(figsize=(9, 8))
    for n in CAM_NODES:
        m = dsrc == n
        ax.scatter(down[m, 0], down[m, 1], s=1.5, c=CAM_COLORS[n], alpha=0.5, label=f"from node_{n} RGB-D ({m.sum()} pts)")
    draw_layout(ax)
    ax.set_aspect("equal"); ax.set_xlabel("X [m]  (MoCap frame)"); ax.set_ylabel("Y [m]")
    ax.set_title(f"Initialized static scene scaffold, top-down (Scene 1, rec {S.REC_WALK}, voxel {VOXEL*100:.0f} cm)\n{NOTE}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, markerscale=4); ax.grid(alpha=0.3)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "scene_scaffold_topdown.png"), dpi=170); plt.close(fig)
    captions.append("scene_scaffold_topdown.png: fused static scaffold from the 3 RGB-D nodes (temporal median over "
                    f"{N_FRAMES} frames of the walk clip, person column cut), coloured by source camera; node rigid bodies "
                    "from the OptiTrack CSV; camera poses: " + POSE_LABEL + ".")

    fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection="3d")
    sub = np.random.default_rng(0).choice(len(down), min(len(down), 40000), replace=False)
    for n in CAM_NODES:
        m = sub[dsrc[sub] == n]
        ax.scatter(down[m, 0], down[m, 1], down[m, 2], s=1, c=CAM_COLORS[n], alpha=0.4, label=f"node_{n}")
    for n, p in S.NODE_POS_S1.items():
        ax.scatter([p[0]], [p[1]], [p[2]], s=60, c=CAM_COLORS.get(n, "0.3"), marker="s", edgecolors="k")
        ax.text(p[0], p[1], p[2] + 0.15, f"N{n}", fontsize=9)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m] (floor = 0)")
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-0.2, 3.2)
    ax.set_title(f"Fused scaffold, 3-D ({len(down)} pts)\n{NOTE}", fontsize=10); ax.legend(fontsize=8, markerscale=6)
    ax.view_init(elev=28, azim=-50)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "scene_scaffold_3d.png"), dpi=170); plt.close(fig)
    captions.append("scene_scaffold_3d.png: the same fused scaffold in 3-D with the five node positions (squares, Z = mounting height).")


def fig_per_camera(per_cam):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for r, n in enumerate(CAM_NODES):
        d = per_cam[n]; med = d["med"].copy()
        # human-column mask in image space from the strided world points
        mask_img = np.zeros(med[::STRIDE, ::STRIDE].shape, bool)
        idx = d["idx"]; ok = idx >= 0
        mask_img[ok] = d["person"][idx[ok]]
        mask_full = np.kron(mask_img, np.ones((STRIDE, STRIDE), bool))[:med.shape[0], :med.shape[1]]
        shown = np.ma.masked_where(med <= 0, med)
        ax = axes[r, 0]; im = ax.imshow(shown, cmap="viridis", vmin=0, vmax=8)
        ax.contour(mask_full, levels=[0.5], colors="m", linewidths=1.2)
        ax.set_title(f"node_{n}: temporal-median depth ({N_FRAMES} frames, walk clip), person column outlined", fontsize=9)
        ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]"); plt.colorbar(im, ax=ax, fraction=0.035, label="depth [m]")
        ax = axes[r, 1]; cam = d["cam"]; keep_c = ~d["person"]
        ax.scatter(cam[keep_c, 0], cam[keep_c, 2], s=0.8, c=CAM_COLORS[n], alpha=0.4, label="scene")
        ax.scatter(cam[~keep_c, 0], cam[~keep_c, 2], s=0.8, c="m", alpha=0.6, label="cut (person column)")
        ax.set_xlabel("x_cam [m] (right)"); ax.set_ylabel("z_cam [m] (forward)"); ax.set_aspect("equal")
        ax.set_title(f"node_{n}: single-camera cloud, camera frame (needs NO extrinsics)", fontsize=9)
        ax.set_xlim(-5, 5); ax.set_ylim(0, 9); ax.grid(alpha=0.3); ax.legend(fontsize=8, markerscale=6, loc="upper right")
    fig.suptitle("Per-camera scene initialisation (nominal D455 intrinsics; depth = 8-bit PNG x 10 m/255)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "scene_scaffold_per_camera.png"), dpi=150); plt.close(fig)
    captions.append("scene_scaffold_per_camera.png: per RGB-D node, the median depth image with the person column outlined "
                    "(magenta) and its back-projected cloud in the camera frame (top-down x vs z); extrinsics-free.")


# --------------------------------------------------------------------------- person
def npy_for_row(row):
    take = os.path.basename(str(row["mocap_data_path"])).replace(".csv", "")
    p = os.path.join(C.OCTONET_ROOT, "mocap_pose", take + "_skeleton.npy")
    return p if os.path.exists(p) else None


def pick_frames():
    df = IO.load_index()
    out = {}
    pos, _ = IO.load_mocap(npy_for_row(S.index_row(S.REC_WALK))); out["standing (walk)"] = (pos[len(pos) // 2], S.REC_WALK)
    pos, _ = IO.load_mocap(npy_for_row(S.index_row(S.REC_SIT))); out["sitting (sit)"] = (pos[len(pos) // 2], S.REC_SIT)
    sub = df[(df.activity == "sleep") & df.mocap_data_path.astype(str).str.contains("mocap_S1")]
    lying = None
    for _, r in sub.iterrows():
        p = npy_for_row(r)
        if not p: continue
        q, _ = IO.load_mocap(p); hz = q[:, 3, 2]
        low = np.where(hz < 0.45)[0]
        if len(low) > 50:
            f = low[len(low) // 2]; lying = (q[f], os.path.basename(p), int(f), len(low), len(q)); break
    out["lying (sleep)"] = (lying[0], lying[1]) if lying else None
    return out, lying


def fig_skeleton(frames, lying):
    fig = plt.figure(figsize=(15, 5.5))
    for i, (name, item) in enumerate(frames.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        if item is None:
            ax.set_title(name + ": not found"); continue
        p, src = item
        for a, b in BONES:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]], "-", c="tab:blue", lw=2)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="k", s=12)
        ax.text(p[3, 0], p[3, 1], p[3, 2] + 0.08, f"head z = {p[3, 2]:.2f} m", fontsize=9, color="r")
        ax.text(p[0, 0], p[0, 1], p[0, 2], " hip", fontsize=8)
        xs, ys = p[:, 0], p[:, 1]; cx, cy = xs.mean(), ys.mean()
        ax.set_xlim(cx - 1, cx + 1); ax.set_ylim(cy - 1, cy + 1); ax.set_zlim(0, 2)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.set_title(f"{name}\n{src[:40]}", fontsize=9)
    fig.suptitle("Person geometry init: 20-joint OptiTrack skeleton (125 Hz, metres, Z up, floor = 0). "
                 "Subject stands at the MoCap origin: the skeleton gives pose + height; room placement comes from the fixed spot.", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "person_skeleton.png"), dpi=170); plt.close(fig)
    ly = f"lying frame {lying[2]} of {lying[1]} ({lying[3]} of {lying[4]} frames have head z < 0.45 m)" if lying else "no lying frame found"
    captions.append("person_skeleton.png: MoCap skeleton (joint order = OptiTrack CSV bone order, verified to 0.0 mm) for a standing "
                    f"walk frame, a sitting frame and a lying 'sleep' frame; {ly}. Head height annotated.")


# --------------------------------------------------------------------------- pipeline diagram
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 7)); ax.set_xlim(0, 15); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f0fe", ls="-", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="k", ls=ls, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=14, lw=1.2, ls=ls))

    # scene row
    box(0.2, 4.6, 2.4, 1.6, "RGB-D, nodes 1/3/5\nRealSense D455, 30 Hz\n8-bit depth PNG\n= 0..10 m / 255 (3.9 cm)")
    arrow(2.6, 5.4, 3.1, 5.4)
    box(3.1, 4.6, 2.1, 1.6, f"temporal median\n{N_FRAMES} frames, walk clip\n(suppresses the\nmoving person)")
    arrow(5.2, 5.4, 5.7, 5.4)
    box(5.7, 4.6, 2.2, 1.6, "back-project\nnominal D455 intrinsics\n87 x 58 deg FoV\n(not shipped)")
    arrow(7.9, 5.4, 8.4, 5.4)
    box(8.4, 4.6, 2.4, 1.6, ("camera -> MoCap frame\nCALIBRATED pose:\nnode rigid body (CSV)\n+ floor pitch/roll\n+ person yaw" if POSE == "calibrated" else "camera -> MoCap frame\nASSUMED pose:\nnode rigid body (CSV)\n+ look-at origin"), fc=("#e8f5e9" if POSE == "calibrated" else "#fff4e5"))
    arrow(10.8, 5.4, 11.3, 5.4)
    box(11.3, 4.6, 1.6, 1.6, "fuse\n3 clouds")
    arrow(12.9, 5.4, 13.4, 5.4)
    box(13.4, 4.6, 1.5, 1.6, "cut person\ncolumn\nr<0.7 m")
    arrow(14.15, 4.6, 14.15, 3.9)
    box(10.9, 2.3, 4.0, 1.6, f"static scene scaffold\npcl.npy (N,6) = xyz + PCA normal, voxel {VOXEL*100:.0f} cm\n"
        "+ scaffold_metadata.json  (same contract as 4DPS build_scaffold)", fc="#e6f4ea")
    # calibration pending
    box(8.4, 2.3, 1.9, 1.6, "octonet_calibrate\nRGB-D <-> room\n(PENDING)", fc="white", ls="--")
    arrow(9.35, 3.9, 9.5, 4.6, ls="--")
    # person row
    box(0.2, 0.4, 2.4, 1.6, "OptiTrack MoCap\n12 cameras, 125 Hz\n50 markers -> 20 joints\nmm, Z up, floor = 0")
    arrow(2.6, 1.2, 3.1, 1.2)
    box(3.1, 0.4, 2.1, 1.6, "npy == CSV bones\n(verified 0.0 mm)\nmm -> m")
    arrow(5.2, 1.2, 5.7, 1.2)
    box(5.7, 0.4, 2.2, 1.6, "person geometry\n20-joint skeleton\n(pose + height;\nsubject at origin)", fc="#e6f4ea")
    arrow(7.9, 1.2, 8.4, 1.2)
    box(8.4, 0.4, 1.9, 1.6, "SMPL fit /\n4DPS person path\n(next)", fc="white", ls="--")
    # node positions
    box(10.6, 0.4, 4.3, 1.6, "Node1..5 rigid bodies in the OptiTrack CSV (Scene 1 only)\n-> sensor positions (std 0 mm across takes);\nWiFi Tx/Rx + obstacle: paper Fig. 2(a), approximate", fc="#fff4e5")
    arrow(10.75, 2.0, 10.75, 4.55, ls="-")
    ax.text(7.5, 6.75, "OctoNet scene + person initialisation (Scene 1)", ha="center", fontsize=12, weight="bold")
    ax.text(0.2, 6.45, "solid = implemented in viz/geometry_init.py & preprocessing/octonet_scene.py; dashed = pending; orange = assumption", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "init_pipeline.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    captions.append("init_pipeline.png: block diagram of the initialisation pipeline (scene: RGB-D -> median -> back-project -> assumed "
                    "pose -> fuse -> person cut -> scaffold; person: MoCap -> skeleton); dashed boxes are pending (octonet_calibrate, SMPL fit).")


def main():
    per_cam, down, dsrc, normals, meta = build_scaffold()
    print(f"scaffold: {len(down)} pts -> {DATA_OUT}/pcl.npy")
    fig_scaffold(down, dsrc)
    fig_per_camera(per_cam)
    frames, lying = pick_frames()
    print("lying:", lying[1:] if lying else None)
    fig_skeleton(frames, lying)
    fig_pipeline()
    with open(os.path.join(OUT, "captions.txt"), "w") as f:
        f.write("\n\n".join(captions) + "\n\nscaffold saved: " + DATA_OUT + f"  ({meta['n_points']} points)\n")
    print("\n".join(captions))


if __name__ == "__main__":
    main()
