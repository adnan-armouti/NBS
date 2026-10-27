"""OctoNet person stage: SMPL fitted to the 20-joint OptiTrack skeleton (no RGB video in OctoNet, so no GVHMR).

One body shape (betas) per recording, per-frame pose + translation, fitted in batch on the GPU to the
MoCap joints subsampled to ~31 Hz, then interpolated (quaternion slerp / linear) to the requested
timestamps (default: the recording's Vayyar frame times). Output = ``person_sequence.npz`` in the MoCap
world frame (metres, Z up, floor at 0, subject at the origin), the person contract the OctoNet branch of
``reconstruction.run_person_sequence`` hands to the heads.

Joint map (OptiTrack bone origin -> SMPL joint), checked against the data (bone lengths and rest layout):
  Hip->midpoint of L_hip/R_hip (OptiTrack's root sits at hip-joint height, SMPL's pelvis joint ~9 cm above it), Chest->spine3, Neck->neck, Head->head (OptiTrack 'Head' sits ~15 cm above the neck, SMPL's
  head joint ~8 cm: weight 0.5), LShoulder/RShoulder (clavicle origin)->L/R_collar, LUArm/RUArm (shoulder
  joint)->L/R_shoulder, LFArm/RFArm (elbow)->L/R_elbow, LHand/RHand (wrist)->L/R_wrist,
  LThigh/RThigh (hip joint)->L/R_hip, LShin/RShin (knee)->L/R_knee, LFoot/RFoot (ankle)->L/R_ankle,
  LToe/RToe (toe base)->L/R_foot.
SMPL template: x = subject's left, y = up, z = front. MoCap: Z up. The root orientation is initialised from
the hips (left = LThigh - RThigh, up = +Z, front = left x up) so the optimiser starts within a few degrees.

CLI (node):  cd 4DPS && CUDA_VISIBLE_DEVICES=4 python -m mm4DPS.octonet_person fit --rec 20240518135416 \\
                 --out /data/mmsd/NBS/data/OctoNet/person/20240518135416
             python -m mm4DPS.octonet_person check --npz <out>/person_sequence.npz
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch

OPTITRACK_NAMES = ["Hip", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
                   "RFArm", "RHand", "LThigh", "LShin", "LFoot", "LToe", "RThigh", "RShin", "RFoot", "RToe"]
SMPL_NAMES = ["pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2", "L_ankle", "R_ankle", "spine3",
              "L_foot", "R_foot", "neck", "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder", "L_elbow",
              "R_elbow", "L_wrist", "R_wrist", "L_hand", "R_hand"]
#: OptiTrack joint index -> SMPL joint index
OPTITRACK_TO_SMPL = {0: 0, 1: 9, 2: 12, 3: 15, 4: 13, 5: 16, 6: 18, 7: 20, 8: 14, 9: 17, 10: 19, 11: 21,
                     12: 1, 13: 4, 14: 7, 15: 10, 16: 2, 17: 5, 18: 8, 19: 11}
#: per-OptiTrack-joint weight in the data term (joints whose SMPL counterpart is not the same anatomical point get 0.5)
JOINT_WEIGHTS = {3: 0.5, 1: 0.5, 4: 0.5, 8: 0.5, 15: 0.5, 19: 0.5}
HIP_MIDPOINT = True                               # OptiTrack Hip <-> mean(SMPL L_hip, R_hip) instead of the pelvis joint
#: per-OptiTrack-joint constant offset in the mapped SMPL joint's LOCAL frame (one vector per recording): the OptiTrack
#: marker-set joint centres are not SMPL's (hip joints wider/higher, knee lower, head centre vs skull base). L2 prior.
JOINT_OFFSETS = True
W_OFFSET = 0.02
#: SMPL joints the OptiTrack skeleton does not constrain -> stronger pose prior
UNCONSTRAINED_SMPL = (3, 6, 22, 23, 10, 11)      # spine1, spine2, hands, feet (toes only weakly)
SUBSAMPLE = 4                                    # 125 Hz / 4 = 31.25 Hz fit rate


# ----------------------------------------------------------------------------- helpers
def axis_angle_to_quat(aa: torch.Tensor) -> torch.Tensor:
    angle = aa.norm(dim=-1, keepdim=True)
    axis = aa / angle.clamp(min=1e-9)
    half = angle / 2
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)   # (…,4) w,x,y,z


def quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    w = q[..., :1].clamp(-1, 1); v = q[..., 1:]
    angle = 2 * torch.acos(w)
    s = torch.sqrt((1 - w * w).clamp(min=1e-12))
    axis = torch.where(s > 1e-6, v / s, torch.zeros_like(v))
    return axis * angle


def slerp_quat(q0, q1, u):
    """q0,q1 (…,4), u (…,1) in [0,1]."""
    d = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1); d = d.abs().clamp(max=1.0)
    theta = torch.acos(d)
    small = theta < 1e-4
    s = torch.sin(theta).clamp(min=1e-12)
    w0 = torch.where(small, 1 - u, torch.sin((1 - u) * theta) / s)
    w1 = torch.where(small, u, torch.sin(u * theta) / s)
    return w0 * q0 + w1 * q1


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def root_orientation_from_hips(pos: np.ndarray) -> np.ndarray:
    """Rotation matrix (SMPL template axes -> world) from one MoCap frame: left = LThigh - RThigh (horizontal),
    up = +Z, front = left x up."""
    left = pos[12] - pos[16]; left[2] = 0.0; left /= np.linalg.norm(left) + 1e-9
    up = np.array([0.0, 0.0, 1.0]); front = np.cross(left, up)
    return np.stack([left, up, front], axis=1)


def mocap_npy_for(rec: str) -> str:
    from preprocessing import octonet_config as C, octonet_scene as S
    stem = os.path.splitext(os.path.basename(S.index_row(rec)["mocap_data_path"]))[0]
    return os.path.join(C.OCTONET_ROOT, "mocap_pose", stem + "_skeleton.npy")


def vayyar_times(rec: str) -> np.ndarray:
    from preprocessing import octonet_scene as S, octonet_io as IO
    _, t = IO.load_vayyar(S.modality_path(1, "vayyar", rec))
    return np.asarray(t, dtype=np.float64)


# ----------------------------------------------------------------------------- fitting
def fit_smpl_sequence(pos: np.ndarray, *, frames: np.ndarray | None = None, device: str = "cuda",
                      iters_shape: int = 250, iters_pose: int = 400, lr: float = 0.02,
                      w_prior: float = 1e-3, w_prior_unconstrained: float = 3e-2, w_smooth: float = 5e-2,
                      w_betas: float = 1e-5, model_path: Path | None = None, verbose: bool = True,
                      joint_map: dict | None = None) -> dict:
    """Fit SMPL to OptiTrack joints ``pos`` (F,20,3) m on the frames ``frames`` (default every SUBSAMPLE-th).

    Returns betas (10,), pose (F',24,3) axis-angle (root first, SMPL template axes -> world absorbed by the root),
    trans (F',3), joints_rmse_m (F',) over the 20 mapped joints, per_joint_rmse_m (20,), frames (F',).
    """
    from mm4DPS.data import SmplBody
    from mm4DPS.data import DEFAULT_MODEL
    joint_map = joint_map or OPTITRACK_TO_SMPL
    dev = torch.device(device)
    body = SmplBody(model_path or DEFAULT_MODEL).to(dev)
    F = len(pos)
    frames = np.arange(0, F, SUBSAMPLE) if frames is None else np.asarray(frames)
    B = len(frames)
    ot = torch.as_tensor(pos[frames], dtype=torch.float32, device=dev)                 # (B,20,3)
    ot_idx = torch.as_tensor(sorted(joint_map), device=dev)
    smpl_idx = torch.as_tensor([joint_map[k] for k in sorted(joint_map)], device=dev)
    wj = torch.as_tensor([JOINT_WEIGHTS.get(int(k), 1.0) for k in sorted(joint_map)], dtype=torch.float32, device=dev)
    # parameters
    betas = torch.zeros(10, device=dev, requires_grad=True)
    root0 = np.stack([matrix_to_axis_angle(root_orientation_from_hips(pos[f])) for f in frames])
    root = torch.as_tensor(root0, dtype=torch.float32, device=dev).requires_grad_(True)
    body_pose = torch.zeros(B, 23, 3, device=dev, requires_grad=True)
    with torch.no_grad():
        J0 = (body.joint_regressor @ body.v_template)[0]                                 # rest pelvis
    trans = (ot[:, 0] - J0[None]).clone().requires_grad_(True)
    offsets = torch.zeros(len(ot_idx), 3, device=dev, requires_grad=True)
    prior_w = torch.full((23,), w_prior, device=dev)
    for j in UNCONSTRAINED_SMPL:
        prior_w[j - 1] = w_prior_unconstrained

    def mapped(joints, rots, use_offsets=True):
        pred = joints[:, smpl_idx]
        if HIP_MIDPOINT and 0 in joint_map:
            col = sorted(joint_map).index(0)
            mid = 0.5 * (joints[:, 1] + joints[:, 2])
            pred = torch.cat([pred[:, :col], mid[:, None], pred[:, col + 1:]], dim=1)
        if JOINT_OFFSETS and use_offsets:
            pred = pred + torch.einsum("bjmn,jn->bjm", rots[:, smpl_idx], offsets)    # offset follows the limb
        return pred

    def forward():
        pose = torch.cat([root[:, None], body_pose], dim=1)
        _, joints, rots = body(betas[None].expand(B, -1), pose, trans, return_global_rotations=True)
        return pose, joints, rots

    def losses(stage):
        pose, joints, rots = forward()
        d = mapped(joints, rots) - ot[:, ot_idx]
        data = (wj[None, :, None] * d * d).sum(-1).mean()
        prior = (prior_w[None, :, None] * body_pose * body_pose).sum() / B
        smooth = ((body_pose[1:] - body_pose[:-1]) ** 2).sum() / B + ((trans[1:] - trans[:-1]) ** 2).sum() / B \
            + ((root[1:] - root[:-1]) ** 2).sum() / B
        beta_l = (betas * betas).sum()
        total = data + (3 if stage == 1 else 1) * prior + w_smooth * smooth + w_betas * beta_l + W_OFFSET * (offsets * offsets).sum()
        return total, data

    t0 = time.time()
    # stage 1: shape + root + trans (+ pose under a strong prior)
    opt = torch.optim.Adam([betas, root, trans, body_pose, offsets], lr=lr)
    for it in range(iters_shape):
        opt.zero_grad(); total, data = losses(1); total.backward(); opt.step()
        if verbose and it % 50 == 0: print(f"  stage1 it {it:4d} data rmse {data.sqrt().item()*100:.2f} cm")
    # stage 2: pose with the weak prior, shape frozen
    opt = torch.optim.Adam([root, trans, body_pose, offsets], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters_pose, eta_min=lr / 20)
    for it in range(iters_pose):
        opt.zero_grad(); total, data = losses(2); total.backward(); opt.step(); sched.step()
        if verbose and it % 100 == 0: print(f"  stage2 it {it:4d} data rmse {data.sqrt().item()*100:.2f} cm")
    with torch.no_grad():
        pose, joints, rots = forward()
        d = mapped(joints, rots) - ot[:, ot_idx]
        per_frame = d.norm(dim=-1).pow(2).mean(1).sqrt()                                  # (B,) with the joint offsets
        per_joint = d.norm(dim=-1).pow(2).mean(0).sqrt()                                  # (20,)
        d0 = mapped(joints, rots, use_offsets=False) - ot[:, ot_idx]
        per_frame_raw = d0.norm(dim=-1).pow(2).mean(1).sqrt(); per_joint_raw = d0.norm(dim=-1).pow(2).mean(0).sqrt()
    if verbose: print(f"  fit: {B} frames, {time.time()-t0:.1f} s, mean joint RMSE {per_frame.mean().item()*100:.2f} cm with offsets "
                      f"({per_frame_raw.mean().item()*100:.2f} cm raw joint-to-joint); |offsets| max {offsets.norm(dim=-1).max().item()*100:.1f} cm")
    return dict(betas=betas.detach().cpu().numpy(), pose=pose.detach().cpu().numpy(), trans=trans.detach().cpu().numpy(),
                joints_rmse_m=per_frame.cpu().numpy(), per_joint_rmse_m=per_joint.cpu().numpy(), frames=frames,
                joints_rmse_raw_m=per_frame_raw.cpu().numpy(), per_joint_rmse_raw_m=per_joint_raw.cpu().numpy(),
                offsets_local_m=offsets.detach().cpu().numpy(),
                fit_seconds=time.time() - t0, ot_joint_order=[int(k) for k in sorted(joint_map)])


def interpolate_fit(fit: dict, t_fit: np.ndarray, t_target: np.ndarray):
    """Pose (slerp per joint) and translation (linear) at ``t_target`` from the fitted frames at ``t_fit``."""
    pose = torch.as_tensor(fit["pose"], dtype=torch.float64); trans = torch.as_tensor(fit["trans"], dtype=torch.float64)
    idx = np.clip(np.searchsorted(t_fit, t_target, side="right") - 1, 0, len(t_fit) - 2)
    u = (t_target - t_fit[idx]) / np.maximum(t_fit[idx + 1] - t_fit[idx], 1e-9)
    u = torch.as_tensor(np.clip(u, 0, 1), dtype=torch.float64)[:, None, None]
    q0, q1 = axis_angle_to_quat(pose[idx]), axis_angle_to_quat(pose[idx + 1])
    pose_t = quat_to_axis_angle(slerp_quat(q0, q1, u))
    trans_t = trans[idx] * (1 - u[:, :, 0]) + trans[idx + 1] * u[:, :, 0]
    valid = (t_target >= t_fit[0]) & (t_target <= t_fit[-1])
    return pose_t.float().numpy(), trans_t.float().numpy(), valid


def run_octonet_person(rec: str, out_dir: Path, *, target_times: np.ndarray | None = None, device: str = "cuda",
                       model_path: Path | None = None, verbose: bool = True, **fit_kw) -> dict:
    """Fit the recording's MoCap and write ``person_sequence.npz`` + ``person_metadata.json`` at the target times
    (default: the Vayyar frame times)."""
    from preprocessing import octonet_io as IO
    from mm4DPS.data import SmplBody, DEFAULT_MODEL
    from mm4DPS.scene import vertex_normals
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    npy = mocap_npy_for(rec)
    pos, t = IO.load_mocap(npy)
    if target_times is None:
        target_times = vayyar_times(rec)
    target_times = np.asarray(target_times, dtype=np.float64)
    fit = fit_smpl_sequence(pos, device=device, model_path=model_path, verbose=verbose, **fit_kw)
    t_fit = t[fit["frames"]]
    pose_t, trans_t, valid = interpolate_fit(fit, t_fit, target_times)
    dev = torch.device(device)
    body = SmplBody(model_path or DEFAULT_MODEL).to(dev)
    verts, joints, normals, rots = [], [], [], []
    with torch.no_grad():
        betas = torch.as_tensor(fit["betas"], dtype=torch.float32, device=dev)
        for i in range(0, len(target_times), 256):
            p = torch.as_tensor(pose_t[i:i + 256], device=dev); tr = torch.as_tensor(trans_t[i:i + 256], device=dev)
            v, j, rr = body(betas[None].expand(len(p), -1), p, tr, return_global_rotations=True)
            verts.append(v.cpu()); joints.append(j.cpu()); normals.append(vertex_normals(v, body.faces).cpu()); rots.append(rr.cpu())
    verts = torch.cat(verts).numpy().astype(np.float32); joints = torch.cat(joints).numpy().astype(np.float32)
    normals = torch.cat(normals).numpy().astype(np.float32); rots = torch.cat(rots).numpy()
    # joint RMSE at the target times against the MoCap interpolated there (mapped joints)
    ot_idx = fit["ot_joint_order"]; smpl_idx = [OPTITRACK_TO_SMPL[k] for k in ot_idx]
    ot_t = np.stack([np.interp(target_times, t, pos[:, k, c]) for k in ot_idx for c in range(3)], 1).reshape(len(target_times), len(ot_idx), 3)
    pred_t = joints[:, smpl_idx].copy()
    if HIP_MIDPOINT: pred_t[:, ot_idx.index(0)] = 0.5 * (joints[:, 1] + joints[:, 2])
    rmse_raw_t = np.sqrt(((pred_t - ot_t) ** 2).sum(-1).mean(1)); rmse_raw_t[~valid] = np.nan
    if JOINT_OFFSETS: pred_t = pred_t + np.einsum("bjmn,jn->bjm", rots[:, smpl_idx], fit["offsets_local_m"])
    rmse_t = np.sqrt(((pred_t - ot_t) ** 2).sum(-1).mean(1)); rmse_t[~valid] = np.nan
    np.savez_compressed(out_dir / "person_sequence.npz", vertices_world=verts, normals_world=normals, joints_world=joints,
                        t=target_times, valid=valid, betas=fit["betas"], pose=pose_t, trans=trans_t,
                        joints_rmse_m=rmse_t.astype(np.float32), joints_rmse_raw_m=rmse_raw_t.astype(np.float32), faces=body.faces.cpu().numpy(),
                        offsets_local_m=fit["offsets_local_m"], per_joint_rmse_raw_m=fit["per_joint_rmse_raw_m"],
                        fit_frames=fit["frames"], fit_t=t_fit, fit_pose=fit["pose"], fit_trans=fit["trans"],
                        fit_joints_rmse_m=fit["joints_rmse_m"], per_joint_rmse_m=fit["per_joint_rmse_m"])
    meta = dict(recording=rec, mocap_npy=npy, mocap_frames=int(len(pos)), mocap_rate_hz=float(1 / np.median(np.diff(t))),
                fit_frames=int(len(fit["frames"])), fit_rate_hz=float(1 / np.median(np.diff(t_fit))), fit_seconds=fit["fit_seconds"],
                target_frames=int(len(target_times)), target_rate_hz=float(1 / np.median(np.diff(target_times))) if len(target_times) > 1 else None,
                target_valid=int(valid.sum()), frame="MoCap world, metres, Z up, floor 0, subject at origin",
                joint_rmse_m=dict(fit_mean=float(fit["joints_rmse_m"].mean()), fit_max=float(fit["joints_rmse_m"].max()),
                                  target_mean=float(np.nanmean(rmse_t)), raw_joint_to_joint_mean=float(fit["joints_rmse_raw_m"].mean()),
                                  per_joint={OPTITRACK_NAMES[k]: float(v) for k, v in zip(ot_idx, fit["per_joint_rmse_m"])},
                                  per_joint_raw={OPTITRACK_NAMES[k]: float(v) for k, v in zip(ot_idx, fit["per_joint_rmse_raw_m"])}),
                joint_offsets_local_cm={OPTITRACK_NAMES[k]: [round(float(x) * 100, 1) for x in o] for k, o in zip(ot_idx, fit["offsets_local_m"])},
                joint_map={OPTITRACK_NAMES[k]: SMPL_NAMES[v] for k, v in OPTITRACK_TO_SMPL.items()},
                joint_weights={OPTITRACK_NAMES[k]: v for k, v in JOINT_WEIGHTS.items()},
                betas=fit["betas"].tolist(), hip_midpoint=HIP_MIDPOINT,
                assumptions=["one SMPL_NEUTRAL shape per recording", "pose prior = L2 on axis-angle (stronger on spine1/2, hands, feet)",
                             "temporal smoothness on pose and translation at the fit rate", "slerp/linear interpolation to the target times",
                             "OptiTrack Head/Chest/clavicle/toe joints are not the same anatomical points as SMPL's (weight 0.5)",
                             "one constant offset per OptiTrack joint in the mapped SMPL joint's local frame (marker-set joint centres vs SMPL's), L2 prior W_OFFSET"],
                columns=dict(vertices_world="(T,6890,3) float32 m", normals_world="(T,6890,3) area-weighted vertex normals",
                             joints_world="(T,24,3) SMPL joints", t="(T,) epoch s", valid="(T,) inside MoCap coverage",
                             pose="(T,24,3) axis-angle, root first", trans="(T,3)", faces="(13776,3)"))
    (out_dir / "person_metadata.json").write_text(json.dumps(meta, indent=1))
    if verbose: print(f"wrote {out_dir/'person_sequence.npz'}: T={len(target_times)} valid={valid.sum()} rmse target {np.nanmean(rmse_t)*100:.2f} cm")
    return meta


# ----------------------------------------------------------------------------- figure
def figure(npz_paths: list[Path], out_png: Path, labels: list[str] | None = None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from preprocessing import octonet_io as IO
    labels = labels or [p.parent.name for p in npz_paths]
    n = len(npz_paths)
    fig = plt.figure(figsize=(20, 5.5 * n))
    for r, (p, lab) in enumerate(zip(npz_paths, labels)):
        d = np.load(p); meta = json.loads((p.parent / "person_metadata.json").read_text())
        pos, t = IO.load_mocap(meta["mocap_npy"])
        ax = fig.add_subplot(n, 5, 5 * r + 1)
        ax.plot(d["fit_t"] - d["fit_t"][0], d["fit_joints_rmse_m"] * 100, lw=0.8, label="fit frames (31 Hz), with joint offsets")
        if "joints_rmse_raw_m" in d: ax.plot(d["t"] - d["fit_t"][0], d["joints_rmse_raw_m"] * 100, "x", ms=3, c="0.5", label="target frames, raw joint-to-joint")
        ax.plot(d["t"] - d["fit_t"][0], d["joints_rmse_m"] * 100, ".", ms=3, label="target frames (Vayyar 3.1 Hz)")
        ax.set_xlabel("time [s]"); ax.set_ylabel("joint RMSE [cm]"); ax.set_title(f"{lab}: joint RMSE over time (20 mapped joints)\nmean {meta['joint_rmse_m']['fit_mean']*100:.2f} cm", fontsize=9)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        ot_idx = [int(k) for k in sorted(OPTITRACK_TO_SMPL)]; smpl_idx = [OPTITRACK_TO_SMPL[k] for k in ot_idx]
        for c, fi in enumerate(np.linspace(0, len(d["t"]) - 1, 3).astype(int)):
            ax = fig.add_subplot(n, 5, 5 * r + 2 + c, projection="3d")
            v = d["vertices_world"][fi]; j = d["joints_world"][fi]; tt = d["t"][fi]
            k = int(np.argmin(np.abs(t - tt))); o = pos[k]
            sub = v[::12]; ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.5, c="0.7", alpha=0.4)
            ax.scatter(o[:, 0], o[:, 1], o[:, 2], s=25, c="r", label="OptiTrack")
            ax.scatter(j[smpl_idx, 0], j[smpl_idx, 1], j[smpl_idx, 2], s=18, c="b", marker="^", label="SMPL (mapped)")
            for a, b in zip(o[ot_idx], j[smpl_idx]): ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "k-", lw=0.5)
            ax.set_title(f"{lab} frame {fi} (t+{tt-d['t'][0]:.1f} s): rmse {d['joints_rmse_m'][fi]*100:.1f} cm", fontsize=8)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            if c == 0: ax.legend(fontsize=7)
        ax = fig.add_subplot(n, 5, 5 * r + 5)
        pj = d["per_joint_rmse_m"] * 100; names = [OPTITRACK_NAMES[k] for k in ot_idx]
        if "per_joint_rmse_raw_m" in d: ax.bar(np.arange(len(pj)) - 0.2, d["per_joint_rmse_raw_m"] * 100, width=0.4, color="0.6", label="raw joint-to-joint")
        ax.bar(np.arange(len(pj)) + 0.2, pj, width=0.4, color=["tab:orange" if JOINT_WEIGHTS.get(k, 1.0) < 1 else "tab:blue" for k in ot_idx], label="with per-joint offsets")
        ax.legend(fontsize=7)
        ax.set_xticks(range(len(pj))); ax.set_xticklabels(names, rotation=70, fontsize=7); ax.set_ylabel("RMSE [cm]")
        ax.set_title("per-joint RMSE (orange = weight 0.5, different anatomical point)", fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("OctoNet person stage: SMPL fitted to the OptiTrack skeleton (one shape per recording, per-frame pose, slerp to the Vayyar frame times)", fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--rec", required=True); f.add_argument("--out", type=Path, required=True)
    f.add_argument("--device", default="cuda"); f.add_argument("--iters-shape", type=int, default=250); f.add_argument("--iters-pose", type=int, default=400)
    f.add_argument("--times", type=Path, default=None, help="npy of target epoch seconds (default: the Vayyar frame times)")
    c = sub.add_parser("check"); c.add_argument("--npz", type=Path, required=True, nargs="+")
    g = sub.add_parser("figure"); g.add_argument("--npz", type=Path, required=True, nargs="+"); g.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        times = np.load(a.times) if a.times else None
        meta = run_octonet_person(a.rec, a.out, target_times=times, device=a.device, iters_shape=a.iters_shape, iters_pose=a.iters_pose)
        print(json.dumps(meta["joint_rmse_m"], indent=1))
    elif a.cmd == "check":
        for p in a.npz:
            d = np.load(p); r = d["joints_rmse_m"]
            print(f"{p}: T={len(d['t'])} valid={int(d['valid'].sum())} rmse mean {np.nanmean(r)*100:.2f} max {np.nanmax(r)*100:.2f} cm (raw joint-to-joint {np.nanmean(d['joints_rmse_raw_m'])*100:.2f} cm) | fit frames {len(d['fit_t'])} rmse {d['fit_joints_rmse_m'].mean()*100:.2f} cm")
            print("   per joint (cm):", {OPTITRACK_NAMES[k]: round(float(v) * 100, 1) for k, v in zip(sorted(OPTITRACK_TO_SMPL), d["per_joint_rmse_m"])})
    elif a.cmd == "figure":
        figure(a.npz, a.out)


if __name__ == "__main__":
    main()
