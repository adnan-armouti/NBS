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


def person_observations(node: int, rec: str, bg: np.ndarray, frames: list[str], K: S.Intrinsics,
                        R_tilt: np.ndarray, h_floor: float, t_w: np.ndarray):
    """Person = the largest blob of pixels at the MoCap-predicted camera range (+-0.6 m), standing
    0.2..2.0 m above the fitted floor. (The subject walks IN PLACE, so they are part of the temporal
    median and a motion-based foreground is empty.) Returns the blob's mean 3-D point per frame."""
    from scipy import ndimage
    pos, tm = IO.load_mocap(os.path.join(C.OCTONET_ROOT, "mocap_pose",
                            os.path.splitext(os.path.basename(S.index_row(rec)["mocap_data_path"]))[0] + "_skeleton.npy"))
    hip = pos[:, 0, :]
    obs, dts = [], []
    H, W = bg.shape; v, u = np.mgrid[0:H:2, 0:W:2]
    for p in [frames[i] for i in np.linspace(len(frames) * 0.1, len(frames) * 0.9, K_PERSON).astype(int)]:
        te = _epoch_of_depth_png(p); j = int(np.argmin(np.abs(tm - te))); dts.append(abs(tm[j] - te))
        if abs(tm[j] - te) > 0.25: continue
        D = S.load_depth_m(p)[::2, ::2]
        r_exp = float(np.linalg.norm(hip[j] - t_w))
        P = np.stack([(u - K.cx) / K.fx * D, (v - K.cy) / K.fy * D, D], -1)        # (h,w,3) camera frame
        q = P @ R_tilt.T                                                           # levelled
        height = h_floor - q[..., 1]
        rng = np.linalg.norm(q, axis=-1)
        m = (D > 0.5) & (np.abs(rng - r_exp) < 0.6) & (height > 0.2) & (height < 2.0)
        m = ndimage.binary_opening(m, iterations=1)
        lab, n = ndimage.label(m)
        if n == 0: continue
        sizes = ndimage.sum(m, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] < 400: continue
        blob = lab == k; pc = P[blob].mean(0); vv, uu = np.nonzero(blob)
        obs.append(dict(png=os.path.basename(p), t=te, dt=float(abs(tm[j] - te)), hip=hip[j].tolist(), p_cam=pc.tolist(),
                        uv=[float(uu.mean() * 2), float(vv.mean() * 2)], area=int(sizes[k - 1]),
                        blob_height_m=float(height[blob].max() - height[blob].min())))
    print(f"  node_{node}: depth<->mocap |dt| min {min(dts):.3f} s, median {np.median(dts):.3f} s; {len(obs)} person blobs, "
          f"blob heights {[round(o['blob_height_m'],2) for o in obs][:6]}")
    return obs


def solve_node(node: int, rec: str):
    K = S.d455_nominal_intrinsics(); t_w = S.NODE_POS_S1[node].copy()
    bg, frames = median_background(node, rec)
    n_c, d, Q, inl_frac, n_cand = ransac_floor(bg, K)
    R_tilt = tilt_rotation(n_c); h_floor = abs(d)
    obs = person_observations(node, rec, bg, frames, K, R_tilt, h_floor, t_w)
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
                _bg=bg, _floor=Q, _obs=obs, _R_tilt=R_tilt)


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


# --------------------------------------------------------------------------- v2: landmarks + offset/scale
LM_HEIGHT = (0.3, 1.4)      # m above floor: a node box + sensors on a tripod
LM_WIN = (60, 45)           # half-window (px) around the predicted pixel: ~5 deg of yaw/offset slack at 5 m


def landmark_observations(node: int, bg: np.ndarray, K: S.Intrinsics, R_wc: np.ndarray, t_w: np.ndarray,
                          R_tilt: np.ndarray, h_floor: float):
    """The OTHER nodes as landmarks: project each known rigid-body position into this camera under the
    current pose, then look for an object 0.3-1.4 m above the floor near the predicted pixel and range.
    Returns one observation per detected landmark: camera-frame blob mean, measured vs predicted range."""
    from scipy import ndimage
    H, W = bg.shape; R_cw = R_wc.T; out = []
    for m, pos in S.NODE_POS_S1.items():
        if m == node: continue
        pc = R_cw @ (pos - t_w)
        if pc[2] < 0.5: continue
        u0, v0 = K.fx * pc[0] / pc[2] + K.cx, K.fy * pc[1] / pc[2] + K.cy
        r_pred = float(np.linalg.norm(pc))
        if not (LM_WIN[0] < u0 < W - LM_WIN[0] and LM_WIN[1] < v0 < H - LM_WIN[1]):
            out.append(dict(landmark=m, visible=False, uv_pred=[u0, v0], r_pred=r_pred)); continue
        us, vs = slice(int(u0 - LM_WIN[0]), int(u0 + LM_WIN[0])), slice(int(v0 - LM_WIN[1]), int(v0 + LM_WIN[1]))
        v, u = np.mgrid[vs, us]; D = bg[vs, us]
        P = np.stack([(u - K.cx) / K.fx * D, (v - K.cy) / K.fy * D, D], -1); q = P @ R_tilt.T
        height = h_floor - q[..., 1]; rng = np.linalg.norm(q, axis=-1)
        msk = (D > 0.5) & (height > LM_HEIGHT[0]) & (height < LM_HEIGHT[1]) & (rng > 0.6 * r_pred) & (rng < 1.4 * r_pred)
        msk = ndimage.binary_opening(msk, iterations=1); lab, n = ndimage.label(msk)
        best = None
        for k in range(1, n + 1):
            b = lab == k
            if b.sum() < 60: continue
            rm = float(np.median(rng[b]))
            score = b.sum() / (1 + 4 * abs(rm - r_pred) / r_pred)       # big and near the predicted range
            if best is None or score > best[0]: best = (score, b, rm)
        if best is None:
            out.append(dict(landmark=m, visible=True, detected=False, uv_pred=[u0, v0], r_pred=r_pred)); continue
        _, b, rm = best; vv, uu = np.nonzero(b)
        out.append(dict(landmark=m, visible=True, detected=True, uv_pred=[u0, v0], r_pred=r_pred, r_meas=rm,
                        uv_meas=[float(uu.mean() + us.start), float(vv.mean() + vs.start)], p_cam=P[b].mean(0).tolist(),
                        area=int(b.sum()), target=pos.tolist()))
    return out


def fit_pose_v2(obs_person, obs_lm, R_tilt, t_marker, h_floor, mode: str):
    """Horizontal least squares over (yaw, dx, dy[, scale]) with the person + landmark observations.
    world = R_level(yaw) R_tilt (s * p_cam) + t_marker + (dx, dy, dz);  dz = s*h_floor - z_marker.
    mode: 'offset' (s=1), 'scale' (dx=dy=0), 'both'."""
    from scipy.optimize import least_squares
    P = [np.array(o["p_cam"]) for o in obs_person] + [np.array(o["p_cam"]) for o in obs_lm]
    T = [np.array(o["hip"]) for o in obs_person] + [np.array(o["target"]) for o in obs_lm]
    wts = np.array([1.0 / max(len(obs_person), 1)] * len(obs_person) + [1.0] * len(obs_lm))  # person counts as one target
    def unpack(x):
        psi = x[0]; dx = dy = 0.0; s = 1.0
        if mode in ("offset", "both"): dx, dy = x[1], x[2]
        if mode == "scale": s = x[1]
        if mode == "both": s = x[3]
        return psi, dx, dy, s
    def resid(x):
        psi, dx, dy, s = unpack(x); R = level_rotation(psi) @ R_tilt
        r = []
        for p, t, w in zip(P, T, wts):
            w_pred = R @ (s * p) + t_marker + np.array([dx, dy, 0.0]); r.append(np.sqrt(w) * (w_pred[:2] - t[:2]))
        return np.concatenate(r)
    x0 = {"offset": [0.0, 0.0, 0.0], "scale": [0.0, 1.0], "both": [0.0, 0.0, 0.0, 1.0]}[mode]
    # initial yaw from the person alone
    o = obs_person[0]; q = R_tilt @ np.array(o["p_cam"]); tgt = np.array(o["hip"]) - t_marker
    x0[0] = float(np.arctan2(tgt[1], tgt[0]) - np.arctan2(-q[0], q[2]))
    sol = least_squares(resid, x0)
    psi, dx, dy, s = unpack(sol.x); R = level_rotation(psi) @ R_tilt
    res = resid(sol.x).reshape(-1, 2); rms = float(np.sqrt((res ** 2).sum(1).mean()))
    per = {}
    for i, o in enumerate(obs_lm): per[f"node_{o['landmark']}"] = float(np.linalg.norm(res[len(obs_person) + i]) / np.sqrt(wts[len(obs_person) + i]))
    per["person"] = float(np.mean([np.linalg.norm(res[i]) / np.sqrt(wts[i]) for i in range(len(obs_person))]))
    dz = s * h_floor - t_marker[2]
    return dict(mode=mode, yaw_deg=float(np.degrees(psi)), dx=float(dx), dy=float(dy), dz=float(dz), scale=float(s),
                rms_m=rms, resid_m=per, n_landmarks=len(obs_lm), R_wc=R.tolist(), t_w=(t_marker + np.array([dx, dy, dz])).tolist())


def figure_v2(sol, lm, fits, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
    for ax, n in zip(axs[:3], NODES):
        s = sol[n]; ax.imshow(np.where(s["_bg"] > 0, s["_bg"], np.nan), cmap="viridis", vmin=0, vmax=8)
        for o in lm[n]:
            u, v = o["uv_pred"]
            if not o["visible"]:
                continue
            ax.add_patch(plt.Rectangle((u - LM_WIN[0], v - LM_WIN[1]), 2 * LM_WIN[0], 2 * LM_WIN[1], fill=False, ec="w", lw=1, ls="--"))
            ax.plot(u, v, "ws", mfc="none", ms=10, mew=1.5)
            if o.get("detected"):
                ax.plot(*o["uv_meas"], "o", color="orange", ms=7); ax.text(u + 5, v - 50, f"node_{o['landmark']}\npred {o['r_pred']:.2f} m\nmeas {o['r_meas']:.2f} m", color="w", fontsize=8)
            else:
                ax.text(u + 5, v - 50, f"node_{o['landmark']}\npred {o['r_pred']:.2f} m\nNOT found", color="r", fontsize=8)
        ax.set_title(f"node_{n}: other nodes as landmarks (white = predicted under v1 pose, orange = detected)", fontsize=9)
        ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]")
    ax = axs[3]; ax.axis("off"); lines = ["per-camera fits (horizontal residual rms, m)", ""]
    for n in NODES:
        lines.append(f"node_{n}: landmarks detected {sum(1 for o in lm[n] if o.get('detected'))}")
        for mode in ("offset", "scale", "both"):
            f = fits[n].get(mode)
            if f: lines.append(f"  {mode:6s} yaw {f['yaw_deg']:7.1f} dx {f['dx']:+.2f} dy {f['dy']:+.2f} dz {f['dz']:+.2f} s {f['scale']:.3f} rms {f['rms_m']:.3f}")
        lines.append(f"  chosen: {fits[n]['chosen']}  ({fits[n]['reason']})"); lines.append("")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    fig.suptitle("octonet_calibrate v2: camera offset from the marker vs depth scale, resolved with the other nodes as range landmarks", fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


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
    # ---- v2: landmarks -> offset vs scale
    lm, fits = {}, {}
    for n in NODES:
        s = sol[n]; R1, t1 = np.array(s["R_wc"]), np.array(s["t_w"])
        lm[n] = landmark_observations(n, s["_bg"], K, R1, t1, s["_R_tilt"], s["floor_height_m"])
        det = [o for o in lm[n] if o.get("detected")]
        for o in lm[n]:
            print(f"  node_{n} landmark node_{o['landmark']}: " + ("not in view" if not o["visible"] else (f"pred {o['r_pred']:.2f} m -> meas {o['r_meas']:.2f} m (ratio {o['r_meas']/o['r_pred']:.3f}), {o['area']} px" if o.get("detected") else f"pred {o['r_pred']:.2f} m, NOT detected")))
        fits[n] = {}
        modes = ("offset", "scale", "both") if len(det) >= 1 else ("offset",)
        for mode in modes:
            try: fits[n][mode] = fit_pose_v2(s["_obs"], det, s["_R_tilt"], S.NODE_POS_S1[n], s["floor_height_m"], mode)
            except Exception as e: print("   fit", mode, "failed:", e)
        # choose: with no landmark only the along-boresight offset is identifiable -> 'offset' (it reduces to that); with
        # landmarks prefer 'offset' unless 'scale' beats it clearly (> 25 % lower rms), 'both' only if it beats both by 25 %.
        fo, fs, fb = fits[n].get("offset"), fits[n].get("scale"), fits[n].get("both")
        chosen, reason = "offset", "no landmark: only the along-boresight offset is identifiable" if not det else "offset explains the landmarks"
        if det and fs and fs["rms_m"] < 0.75 * fo["rms_m"]: chosen, reason = "scale", "depth scale fits the landmarks clearly better than an offset"
        if det and fb and fb["rms_m"] < 0.75 * min(fo["rms_m"], fs["rms_m"] if fs else 9): chosen, reason = "both", "offset+scale clearly better than either alone"
        fits[n]["chosen"], fits[n]["reason"] = chosen, reason
        f = fits[n][chosen]; print(f"node_{n} v2 [{chosen}]: yaw {f['yaw_deg']:.1f} (v1 {s['yaw_deg']:.1f}) offset ({f['dx']:+.2f},{f['dy']:+.2f},{f['dz']:+.2f}) m scale {f['scale']:.3f} rms {f['rms_m']:.3f} m; {reason}")
    figure_v2(sol, lm, fits, os.path.join(viz_dir, "calibration_v2.png"))
    clouds_v2 = {n: cloud_world(sol[n]["_bg"] * fits[n][fits[n]["chosen"]]["scale"], np.array(fits[n][fits[n]["chosen"]]["R_wc"]), np.array(fits[n][fits[n]["chosen"]]["t_w"]), K) for n in NODES}
    ov_v2 = overlap_metric(clouds_v2); print("overlap v2:", {k: round(v["median_nn_m"], 3) for k, v in ov_v2.items()})
    ext = {"frame": "MoCap world, metres, Z up, origin = subject spot", "method": __doc__.split("\n")[0],
           "recording": S.REC_WALK, "intrinsics": "nominal D455 87x58 deg, 640x480", "depth_decode": "8-bit * 10 m / 255",
           "cameras": {f"node_{n}": dict(fits[n][fits[n]["chosen"]], depth_scale=fits[n][fits[n]["chosen"]]["scale"], v2_mode=fits[n]["chosen"], v2_reason=fits[n]["reason"],
                                        landmarks=[{k: v for k, v in o.items()} for o in lm[n]], pitch_deg=sol[n]["pitch_deg"], roll_deg=sol[n]["roll_deg"],
                                        floor_height_m=sol[n]["floor_height_m"], node_marker_pos=S.NODE_POS_S1[n].tolist(), all_fits={m: {k: v for k, v in f.items() if k not in ("R_wc", "t_w")} for m, f in fits[n].items() if isinstance(f, dict)}) for n in NODES},
           "cameras_v1_look_at_person": {f"node_{n}": {k: v for k, v in sol[n].items() if not k.startswith("_")} for n in NODES},
           "overlap_v2": ov_v2,
           "vayyar": {"node": 1, "R_wc": fits[1][fits[1]["chosen"]]["R_wc"], "t_w": S.NODE_POS_S1[1].tolist(), "ASSUMPTION": "radar shares node_1's camera rotation; placed at the marker position"},
           "overlap_assumed": ov_as, "overlap_calibrated": ov_cal, "icp_to_node1_not_applied": icp}
    with open(os.path.join(out_dir, "extrinsics.json"), "w") as f: json.dump(ext, f, indent=1)
    print("wrote", os.path.join(out_dir, "extrinsics.json"))


if __name__ == "__main__":
    main()
