"""octonet_calibrate: RGB-D camera extrinsics in the MoCap world frame (Scene 1).

Per camera node (1, 3, 5), with the camera placed at the node rigid-body position:
  1. PITCH + ROLL from the floor: RANSAC plane on the lower half of the temporal-median depth of the
     'walk' clip; the plane normal must map to world +Z.  The plane distance is a camera-height check
     against the rigid-body Z.
  2. YAW from the person: in K single frames of the 'walk' clip the person is the foreground
     (median_depth - frame_depth > 0.35 m); its centroid ray + median depth gives a camera-frame point,
     and the MoCap hip (joint 0) at the frame timestamp gives the same point in the world, so the yaw
     follows from the two XY directions seen from the node.  Also returns a range check
     (camera distance vs world distance) that validates the 8-bit 0-10 m decode and nominal intrinsics.
  3. Cross-camera check: median nearest-neighbour distance between the fused clouds, assumed vs
     calibrated poses; optional open3d point-to-plane ICP of nodes 3/5 onto node 1 (reported, applied
     only if it is a small correction that improves the overlap).

Writes ``$NBS_DATA_OUT/calib_S1/extrinsics.json`` (R_wc, t_w per node; the Vayyar on node_1 is given
node_1's camera rotation as an explicit ASSUMPTION) and ``viz/geometry_init/calibration.png``.
``octonet_scene.camera_pose`` picks the calibrated pose up automatically once the file exists.

Run on the node:  cd nbs && python -m preprocessing.octonet_calibrate
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime
import numpy as np

from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S

NODES = (1, 3, 5)
N_BG = 30           # frames for the median background
K_PERSON = 16       # frames for the yaw solve
FLOOR_TOL = 0.06    # m, inlier tolerance (depth quantisation 3.9 cm)
FG_THRESH = 0.35    # m closer than background = foreground


def _epoch_of_depth_png(p: str) -> float:
    s = os.path.basename(p)[len("depth_"):-len(".png")]
    return datetime.fromisoformat(s).timestamp()


def median_background(node: int, rec: str):
    clip = S.depth_clip_dir(node, rec)
    frames = S.depth_frames(clip)
    sel = [frames[i] for i in np.linspace(0, len(frames) - 1, N_BG).astype(int)]
    st = np.stack([S.load_depth_m(p) for p in sel]); st[st <= 0] = np.nan
    bg = np.nanmedian(st, 0); bg[np.isfinite(st).mean(0) < 0.5] = 0; bg = np.nan_to_num(bg)
    return bg, frames


def ransac_floor(bg: np.ndarray, K: S.Intrinsics, iters=600, seed=0):
    """Floor plane in the camera frame from the lower part of the median depth image."""
    H, W = bg.shape
    v, u = np.mgrid[0:H:2, 0:W:2]; z = bg[::2, ::2]
    ok = (z > 0.8) & (z < 7.0) & (v > 0.45 * H)
    P = np.stack([(u[ok] - K.cx) / K.fx * z[ok], (v[ok] - K.cy) / K.fy * z[ok], z[ok]], 1)
    rng = np.random.default_rng(seed); best = (0, None, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn
        if n[1] > 0: n = -n                       # camera 'up' is -y: normal must point to -y
        if n[1] > -np.cos(np.deg2rad(35)): continue  # floor normal within 35 deg of camera up
        d = -n @ a
        if not (0.4 < abs(d) < 1.6): continue      # camera height 0.4..1.6 m above the plane
        inl = np.abs(P @ n + d) < FLOOR_TOL
        if inl.sum() > best[0]: best = (int(inl.sum()), n, d)
    cnt, n, d = best
    if n is None: raise RuntimeError("no floor plane found")
    inl = np.abs(P @ n + d) < FLOOR_TOL
    Q = P[inl]; c0 = Q.mean(0); _, _, vt = np.linalg.svd(Q - c0); n = vt[2]
    if n[1] > 0: n = -n
    d = -n @ c0
    return n, float(d), Q, float(inl.mean()), len(P)


def tilt_rotation(n_c: np.ndarray) -> np.ndarray:
    """Rotation (camera frame) taking the floor normal to the ideal camera-up (0,-1,0)."""
    e = np.array([0.0, -1.0, 0.0]); n = n_c / np.linalg.norm(n_c)
    ax = np.cross(n, e); s = np.linalg.norm(ax); c = float(n @ e)
    if s < 1e-9: return np.eye(3)
    ax /= s; th = np.arctan2(s, c)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def level_rotation(psi: float) -> np.ndarray:
    """World-from-levelled-camera for a horizontal camera looking along azimuth psi (cols = right, down, fwd)."""
    fwd = np.array([np.cos(psi), np.sin(psi), 0.0]); right = np.array([np.sin(psi), -np.cos(psi), 0.0])
    return np.stack([right, np.array([0, 0, -1.0]), fwd], 1)


def person_observations(node: int, rec: str, bg: np.ndarray, frames: list[str], K: S.Intrinsics):
    from scipy import ndimage
    pos, tm = IO.load_mocap(os.path.join(C.OCTONET_ROOT, "mocap_pose",
                            os.path.splitext(os.path.basename(S.index_row(rec)["mocap_data_path"]))[0] + "_skeleton.npy"))
    hip = pos[:, 0, :]
    obs = []
    for p in [frames[i] for i in np.linspace(len(frames) * 0.1, len(frames) * 0.9, K_PERSON).astype(int)]:
        te = _epoch_of_depth_png(p); j = int(np.argmin(np.abs(tm - te)))
        if abs(tm[j] - te) > 0.1: continue
        D = S.load_depth_m(p)
        fg = (bg > 0) & (D > 1.0) & (D < 6.0) & (bg - D > FG_THRESH)
        fg = ndimage.binary_opening(fg, iterations=2)
        lab, n = ndimage.label(fg)
        if n == 0: continue
        sizes = ndimage.sum(fg, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] < 1500: continue
        m = lab == k; vv, uu = np.nonzero(m); zc = float(np.median(D[m]))
        u0, v0 = float(uu.mean()), float(vv.mean())
        pc = np.array([(u0 - K.cx) / K.fx * zc, (v0 - K.cy) / K.fy * zc, zc])
        obs.append(dict(png=os.path.basename(p), t=te, hip=hip[j].tolist(), p_cam=pc.tolist(), uv=[u0, v0], area=int(sizes[k - 1])))
    return obs


def solve_node(node: int, rec: str):
    K = S.d455_nominal_intrinsics(); t_w = S.NODE_POS_S1[node].copy()
    bg, frames = median_background(node, rec)
    n_c, d, Q, inl_frac, n_cand = ransac_floor(bg, K)
    R_tilt = tilt_rotation(n_c); h_floor = abs(d)
    obs = person_observations(node, rec, bg, frames, K)
    if len(obs) < 3: raise RuntimeError(f"node_{node}: only {len(obs)} person observations")
    psis, rng_cam, rng_w, elev = [], [], [], []
    for o in obs:
        q = R_tilt @ np.array(o["p_cam"])                 # levelled camera frame
        a_cam = np.arctan2(-q[0], q[2])                    # azimuth of the ray in the psi=0 world
        tgt = np.array(o["hip"]) - t_w
        psis.append(np.arctan2(tgt[1], tgt[0]) - a_cam)
        rng_cam.append(float(np.hypot(q[0], q[2]))); rng_w.append(float(np.hypot(tgt[0], tgt[1])))
        elev.append(float(np.degrees(np.arctan2(-q[1], np.hypot(q[0], q[2])))))
    psis = np.array(psis); psi = float(np.arctan2(np.sin(psis).mean(), np.cos(psis).mean()))
    psi_std = float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.exp(1j * psis).mean()) + 1e-12))))
    R_wc = level_rotation(psi) @ R_tilt
    # pitch/roll of the raw camera relative to level, for the report
    fwd_c = R_wc @ np.array([0, 0, 1.0]); pitch = float(np.degrees(np.arcsin(fwd_c[2])))
    right_c = R_wc @ np.array([1.0, 0, 0]); roll = float(np.degrees(np.arcsin(right_c[2])))
    # assumed pose for comparison
    R_as, _ = S.camera_pose_assumed(node); psi_as = float(np.arctan2((R_as @ [0, 0, 1])[1], (R_as @ [0, 0, 1])[0]))
    return dict(node=node, rec=rec, R_wc=R_wc.tolist(), t_w=t_w.tolist(), yaw_deg=float(np.degrees(psi)),
                yaw_assumed_deg=float(np.degrees(psi_as)), yaw_std_deg=psi_std, pitch_deg=pitch, roll_deg=roll,
                floor_height_m=h_floor, node_marker_z_m=float(t_w[2]), floor_inlier_frac=inl_frac, floor_candidates=n_cand,
                person_obs=len(obs), range_cam_over_world=float(np.mean(rng_cam) / np.mean(rng_w)),
                range_cam_m=float(np.mean(rng_cam)), range_world_m=float(np.mean(rng_w)),
                person_elev_deg=float(np.mean(elev)), hip_z_m=float(np.mean([o["hip"][2] for o in obs])),
                _bg=bg, _floor=Q, _obs=obs)


# --------------------------------------------------------------------------- cross-camera check
def cloud_world(bg, R, t, K, stride=3):
    v, u = np.mgrid[0:bg.shape[0]:stride, 0:bg.shape[1]:stride]; z = bg[::stride, ::stride]
    ok = (z > 0.5) & (z < 8.0)
    P = np.stack([(u[ok] - K.cx) / K.fx * z[ok], (v[ok] - K.cy) / K.fy * z[ok], z[ok]], 1) @ R.T + t
    r = np.hypot(P[:, 0], P[:, 1]); keep = ~((r < 0.8) & (P[:, 2] < 2.2)) & (P[:, 2] > -0.2) & (P[:, 2] < 3.0)
    return P[keep]


def overlap_metric(clouds: dict) -> dict:
    from scipy.spatial import cKDTree
    out = {}
    for a in clouds:
        for b in clouds:
            if a >= b: continue
            d1, _ = cKDTree(clouds[b]).query(clouds[a], distance_upper_bound=0.5)
            d1 = d1[np.isfinite(d1)]
            out[f"{a}-{b}"] = dict(median_nn_m=float(np.median(d1)) if len(d1) else None, frac_within_0p5=float(len(d1) / len(clouds[a])))
    return out


def icp_refine(src: np.ndarray, dst: np.ndarray, thresh=0.3):
    import open3d as o3d
    def pc(P):
        q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); q.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20)); return q
    res = o3d.pipelines.registration.registration_icp(pc(src), pc(dst), thresh, np.eye(4),
          o3d.pipelines.registration.TransformationEstimationPointToPlane(),
          o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    T = np.asarray(res.transformation); ang = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    return T, ang, float(np.linalg.norm(T[:3, 3])), float(res.fitness), float(res.inlier_rmse)


# --------------------------------------------------------------------------- figure
def figure(sol: dict, clouds_as: dict, clouds_cal: dict, ov_as, ov_cal, out_png: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    K = S.d455_nominal_intrinsics()
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    for i, n in enumerate(NODES):
        s = sol[n]; ax = axs[i, 0]
        im = ax.imshow(np.where(s["_bg"] > 0, s["_bg"], np.nan), cmap="viridis", vmin=0, vmax=8)
        Q = s["_floor"]; ax.scatter(Q[:, 0] / Q[:, 2] * K.fx + K.cx, Q[:, 1] / Q[:, 2] * K.fy + K.cy, s=1, c="lime", alpha=0.3)
        for o in s["_obs"]: ax.plot(*o["uv"], "r+", ms=10, mew=2)
        ax.set_title(f"node_{n}: median depth; floor inliers (green), person centroids (red +)\n"
                     f"floor height {s['floor_height_m']:.2f} m vs marker z {s['node_marker_z_m']:.2f} m; pitch {s['pitch_deg']:+.1f} deg, roll {s['roll_deg']:+.1f} deg", fontsize=9)
        ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]"); plt.colorbar(im, ax=ax, label="depth [m]")
        ax = axs[i, 1]
        hips = np.array([o["hip"] for o in s["_obs"]]); t = np.array(s["t_w"])
        ax.plot([t[0]], [t[1]], "ks", ms=10, label=f"node_{n} (rigid body)")
        ax.plot(hips[:, 0], hips[:, 1], "r.", label="MoCap hip at the frame times")
        for psi, lab, col in ((np.radians(s["yaw_assumed_deg"]), "assumed yaw (look-at origin)", "0.6"), (np.radians(s["yaw_deg"]), f"calibrated yaw {s['yaw_deg']:.1f} deg (+-{s['yaw_std_deg']:.1f})", "b")):
            ax.plot([t[0], t[0] + 3.5 * np.cos(psi)], [t[1], t[1] + 3.5 * np.sin(psi)], color=col, lw=2, label=lab)
        ax.plot(0, 0, "k*", ms=10); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim(-3.5, 3.8); ax.set_ylim(-2.6, 2.6); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_title(f"node_{n}: yaw from {s['person_obs']} person observations\n"
                     f"range check cam/world = {s['range_cam_over_world']:.3f} ({s['range_cam_m']:.2f} / {s['range_world_m']:.2f} m)", fontsize=9)
    for j, (cl, ov, lab) in enumerate(((clouds_as, ov_as, "ASSUMED poses (look-at origin)"), (clouds_cal, ov_cal, "CALIBRATED poses (floor + person)"))):
        ax = axs[j, 2]
        for n, col in zip(NODES, ("tab:red", "tab:green", "tab:blue")):
            P = cl[n]; ax.scatter(P[:, 0], P[:, 1], s=0.3, c=col, alpha=0.35, label=f"node_{n}")
            ax.plot(*S.NODE_POS_S1[n][:2], "o", color=col, mec="k", ms=9)
        ax.plot(0, 0, "k*", ms=10); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(markerscale=12, fontsize=8)
        ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_title(f"fused top-down, {lab}\nmedian NN dist: " + ", ".join(f"{k} {v['median_nn_m']:.2f} m" for k, v in ov.items()), fontsize=9)
    ax = axs[2, 2]; ax.axis("off")
    txt = "\n".join(f"node_{n}: yaw {sol[n]['yaw_deg']:7.1f} (assumed {sol[n]['yaw_assumed_deg']:7.1f}) pitch {sol[n]['pitch_deg']:+5.1f} roll {sol[n]['roll_deg']:+5.1f} | floor h {sol[n]['floor_height_m']:.2f} vs z {sol[n]['node_marker_z_m']:.2f} | range ratio {sol[n]['range_cam_over_world']:.3f}" for n in NODES)
    ax.text(0, 0.95, "octonet_calibrate summary (deg, m)\n" + txt + "\n\nPosition = node rigid body (camera offset from the marker centroid ignored).\nIntrinsics nominal D455; depth 8-bit 0-10 m.", va="top", family="monospace", fontsize=9)
    fig.suptitle("OctoNet Scene 1 RGB-D extrinsics: floor plane -> pitch/roll, MoCap-synchronised person -> yaw", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    out_dir = os.path.join(C.NBS_DATA_OUT, "calib_S1"); os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", "geometry_init"); os.makedirs(viz_dir, exist_ok=True)
    K = S.d455_nominal_intrinsics()
    sol = {n: solve_node(n, S.REC_WALK) for n in NODES}
    for n in NODES:
        s = sol[n]; print(f"node_{n}: yaw {s['yaw_deg']:.1f} (assumed {s['yaw_assumed_deg']:.1f}, std {s['yaw_std_deg']:.1f}) pitch {s['pitch_deg']:+.1f} roll {s['roll_deg']:+.1f} | floor h {s['floor_height_m']:.2f} vs z {s['node_marker_z_m']:.2f} (inl {s['floor_inlier_frac']:.2f}) | obs {s['person_obs']} range ratio {s['range_cam_over_world']:.3f} elev {s['person_elev_deg']:+.1f} hip z {s['hip_z_m']:.2f}")
    clouds_as = {n: cloud_world(sol[n]["_bg"], *S.camera_pose_assumed(n), K) for n in NODES}
    clouds_cal = {n: cloud_world(sol[n]["_bg"], np.array(sol[n]["R_wc"]), np.array(sol[n]["t_w"]), K) for n in NODES}
    ov_as, ov_cal = overlap_metric(clouds_as), overlap_metric(clouds_cal)
    print("overlap assumed   :", ov_as); print("overlap calibrated:", ov_cal)
    icp = {}
    for n in (3, 5):
        try:
            T, ang, tr, fit, rmse = icp_refine(clouds_cal[n], clouds_cal[1])
            icp[n] = dict(rot_deg=ang, trans_m=tr, fitness=fit, inlier_rmse=rmse, T=T.tolist()); print(f"ICP node_{n}->node_1: rot {ang:.2f} deg, trans {tr:.3f} m, fitness {fit:.2f}, rmse {rmse:.3f}")
        except Exception as e:
            print("ICP skipped:", e)
    figure(sol, clouds_as, clouds_cal, ov_as, ov_cal, os.path.join(viz_dir, "calibration.png"))
    ext = {"frame": "MoCap world, metres, Z up, origin = subject spot", "method": __doc__.split("\n")[0],
           "recording": S.REC_WALK, "intrinsics": "nominal D455 87x58 deg, 640x480", "depth_decode": "8-bit * 10 m / 255",
           "cameras": {f"node_{n}": {k: v for k, v in sol[n].items() if not k.startswith("_")} for n in NODES},
           "vayyar": {"node": 1, "R_wc": sol[1]["R_wc"], "t_w": sol[1]["t_w"], "ASSUMPTION": "radar shares node_1's camera rotation and marker position"},
           "overlap_assumed": ov_as, "overlap_calibrated": ov_cal, "icp_to_node1_not_applied": icp}
    with open(os.path.join(out_dir, "extrinsics.json"), "w") as f: json.dump(ext, f, indent=1)
    print("wrote", os.path.join(out_dir, "extrinsics.json"))


if __name__ == "__main__":
    main()
