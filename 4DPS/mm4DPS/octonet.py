"""OctoNet in 4DPS: the dataset's sequence access, sensor descriptors and the RGB-D + MoCap scaffold stage.

OctoNet (Scene 1) replaces 4DPS's two inputs:
  * scene:  three RealSense D455 depth streams (nodes 1/3/5; 8-bit 0-10 m PNGs, per-camera depth scale) with
            calibrated extrinsics (nbs/preprocessing/octonet_calibrate.py) instead of Depth-Anything on one RGB frame;
  * person: the 125 Hz 20-joint OptiTrack skeleton with SMPL fitted to it (mm4DPS/octonet_person.py) instead of GVHMR.
The measured side is the NBS loaders (nbs/preprocessing/octonet_io.py); the SFCW target is the clutter-guarded
temporal-mean channel of nbs/method/sfcw_target.py. The rendering heads live in mm4DPS/octonet_heads.py.

The scaffold stage writes the 4DPS contract so later stages stay dataset-agnostic:
    pcl.npy                 (N,6) float32: x y z (MoCap world, metres, Z up) + unit normal
    vayyar_static_rp.npy    (400,100) complex64: the pooled static range profiles (the measured target's rp_mean)
    vayyar_static_target.npz the full target (mask, reliability weights, ...), nbs/method/sfcw_target.save_target
    scaffold_metadata.json

    python -m mm4DPS.octonet scaffold --rec 20240518135416 --output-dir <dir>
    python -m mm4DPS.octonet sequence-info --rec 20240518134358
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# nbs/ is on PYTHONPATH (tools/env_node.sh): the OctoNet code is single-sourced there
from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S

C_LIGHT = 299792458.0


@dataclass(frozen=True)
class Sensor:
    """A non-FMCW OctoNet sensor: what its head renders and where it is."""
    name: str
    kind: str                 # "sfcw" | "uwb" | "wifi"
    node: int
    fc_hz: float
    bandwidth_hz: float
    bins: int                 # SFCW: tones; UWB: CIR bins; WiFi: subcarriers
    rate_hz: float
    position_source: str      # where the pose comes from
    note: str = ""


SENSORS = {
    "vayyar": Sensor("vayyar", "sfcw", 1, 65.0e9, 4.0e9, 100, 3.1, "calib_S1/extrinsics.json: vayyar (node_1 marker + camera yaw)",
                     "20x20 L-array, pair order tx-major, Tx layout UNVERIFIED (two sub-groups); pair-wise range profiles only"),
    "uwb": Sensor("uwb", "uwb", 1, 7.2e9, 1.5e9, 1535, 17.06, "calib_S1/extrinsics.json: uwb (= node_1 marker + camera yaw)",
                  "Novelda X4M200, single Tx/Rx, RF-sampled frame (23.328 GS/s, carrier ~7.2 GHz), 9.9 m / 1535 bins, 17.06 Hz stream (pickle = one dict per frame)"),
    "wifi": Sensor("wifi", "wifi", 0, 5.18e9, 40.0e6, 114, 75.6, "octonet_scene.WIFI_TX_APPROX / WIFI_RX_APPROX (schematic, +-0.5 m)",
                   "Xiaomi AX6000 Tx (1 ant) -> 4x RPi CM4 + AX200 Rx (2 ant), ch 36 HT40; receiver k under node_k/wifi; Rx-4 NLOS"),
}


# --------------------------------------------------------------------------- sequence access
class OctoNetSequence:
    """One OctoNet recording as 4DPS sees it (the analogue of data.RTPoseSequence).

    ``rec`` is the 14-digit stamp of node_1's files (cut_manual.csv row); other nodes' files carry their own
    stamp, resolved through the index (octonet_scene.modality_path).
    """

    def __init__(self, rec: str, data_out: Path | None = None):
        self.rec = str(rec)
        self.row = S.index_row(self.rec)
        self.data_out = Path(data_out or C.NBS_DATA_OUT)

    # -- paths
    def path(self, node: int, modality: str) -> str:
        return S.modality_path(node, modality, self.rec)

    def mocap_npy(self) -> str:
        stem = os.path.splitext(os.path.basename(str(self.row["mocap_data_path"])))[0]
        return os.path.join(C.OCTONET_ROOT, "mocap_pose", stem + "_skeleton.npy")

    def calib_json(self) -> Path:
        return self.data_out / "calib_S1" / "extrinsics.json"

    def scaffold_dir(self) -> Path:
        return self.data_out / "scaffold_S1_calibrated"

    # -- measured streams
    def vayyar(self):
        """(T,400,100) complex64 sweeps, (T,) epoch s."""
        return IO.load_vayyar(self.path(1, "vayyar"))

    def uwb(self):
        """(F,1535) float64 RF-sampled CIR frames at 17.07 Hz (the pickle is a stream of per-frame dicts), (F,) epoch s."""
        return IO.load_uwb(self.path(1, "uwb"))

    def wifi(self, receiver: int):
        """(F,2,114) complex CSI of receiver k (stored under node_k/wifi), (F,) epoch s."""
        import pickle
        x = pickle.load(open(self.path(receiver, "wifi"), "rb"))
        data = np.asarray(x["data"]); t = np.array([v.timestamp() if hasattr(v, "timestamp") else float(v) for v in x["timestamp"]])
        return data, t

    def mocap(self):
        """(F,20,3) metres (Z up, subject at the origin), (F,) epoch s."""
        return IO.load_mocap(self.mocap_npy())

    # -- geometry
    def extrinsics(self) -> dict:
        return json.loads(self.calib_json().read_text())

    def sensor_pose(self, sensor: str) -> tuple[np.ndarray, np.ndarray]:
        """(R_wc, t_w) of 'vayyar' | 'uwb' | 'node_<n>' (cameras), MoCap world frame."""
        e = self.extrinsics()
        entry = e["cameras"][sensor] if sensor.startswith("node_") else e[sensor]
        return np.array(entry["R_wc"]), np.array(entry["t_w"])

    def scaffold(self) -> np.ndarray:
        return np.load(self.scaffold_dir() / "pcl.npy")

    def static_target(self, pooled: bool = True) -> dict:
        from method import sfcw_target as ST
        d = self.data_out / "sfcw_target_S1"
        p = d / ("target_pool5.npz" if pooled else f"target_rec_{self.rec}.npz")
        if not p.exists():
            p = d / "target_rec_sit.npz" if (d / "target_rec_sit.npz").exists() and self.rec == S.REC_SIT else p
        if p.exists():
            z = np.load(p, allow_pickle=True)
            return {k: z[k] for k in z.files}
        V, _ = self.vayyar()
        return ST.sfcw_static_target(V)

    def summary(self) -> dict:
        V, tv = self.vayyar(); pos, tm = self.mocap(); u, tu = self.uwb()
        out = {"rec": self.rec, "subject": int(self.row["user_id"]), "activity": str(self.row["activity"]),
               "vayyar": {"frames": int(V.shape[0]), "shape": list(V.shape), "rate_hz": float(1 / np.median(np.diff(tv)))},
               "mocap": {"frames": int(pos.shape[0]), "rate_hz": float(1 / np.median(np.diff(tm))), "head_z_m": float(pos[:, 3, 2].mean())},
               "uwb": {"frames": int(u.shape[0]), "bins": int(u.shape[1])},
               "wifi": {}, "depth_nodes": [], "calibration": self.calib_json().exists(), "scaffold": (self.scaffold_dir() / "pcl.npy").exists()}
        for k in (1, 2, 3, 4):
            try:
                d, tw = self.wifi(k); out["wifi"][f"rx{k}"] = {"frames": int(d.shape[0]), "shape": list(d.shape)}
            except Exception as e:
                out["wifi"][f"rx{k}"] = f"missing ({e.__class__.__name__})"
        for n in (1, 3, 5):
            try:
                out["depth_nodes"].append({"node": n, "frames": len(S.depth_frames(S.depth_clip_dir(n, self.rec))), "depth_scale": S.depth_scale(n)})
            except Exception as e:
                out["depth_nodes"].append({"node": n, "missing": e.__class__.__name__})
        return out


# --------------------------------------------------------------------------- render inputs (the heads' convention)
def render_inputs(scaffold_dir: Path, extrinsics: dict | Path, device="cuda", n_points: int | None = None,
                  seed: int = 0, material_rho: float = 1.0, use_shadow_mask: bool | None = None) -> tuple[dict, dict, dict]:
    """``(cloud, material, sensor)`` for mm4DPS.octonet_heads from a scaffold-stage directory.

    cloud    = {xyz (N,3) float64, normals (N,3) float64}            MoCap world, metres
    material = {abcd (N,4) float64 ITU P.2040 (concrete), rho (N,) float64 reflectivity scale (1 = the material's own)}
    sensor   = {pos (3,), boresight (3,)} for the Vayyar from calib_S1/extrinsics.json (camera yaw by assumption)
    """
    import torch
    from mm4DPS.octonet_heads import sensor_from_extrinsics
    from method import freq_material as FM
    pcl = np.load(Path(scaffold_dir) / "pcl.npy").astype(np.float64)
    sm_path = Path(scaffold_dir) / "shadow_mask.npy"
    # D7: the mask is written by the stage but consumed only on request -- at iteration 0 it did not help the SFCW
    # profile (0.636 -> 0.614 with 0.25 deg cells / 0.15 m; the bin < 0.9 m guard already removes the node's own body)
    if use_shadow_mask is None:
        use_shadow_mask = os.environ.get("OCTONET_SHADOW_MASK", "0") == "1"
    shadow = np.load(sm_path) if (use_shadow_mask and sm_path.exists()) else None
    if n_points and len(pcl) > n_points:
        idx = np.random.default_rng(seed).choice(len(pcl), n_points, replace=False)
        pcl = pcl[idx]; shadow = None if shadow is None else shadow[idx]
    cloud = {"xyz": torch.as_tensor(pcl[:, :3], dtype=torch.float64, device=device),
             "normals": torch.as_tensor(pcl[:, 3:6], dtype=torch.float64, device=device)}
    if shadow is not None:
        cloud["shadow_mask"] = torch.as_tensor(shadow, dtype=torch.bool, device=device)      # D7: True = visible from the radar
    abcd = FM.ITU_MATERIALS["concrete"] if hasattr(FM, "ITU_MATERIALS") else (5.24, 0.0, 0.0462, 0.7822)
    material = {"abcd": torch.as_tensor(np.tile(np.asarray(abcd, float), (len(pcl), 1)), dtype=torch.float64, device=device),
                "rho": torch.full((len(pcl),), float(material_rho), dtype=torch.float64, device=device)}
    ext = extrinsics if isinstance(extrinsics, dict) else json.loads(Path(extrinsics).read_text())
    return cloud, material, sensor_from_extrinsics(ext, "vayyar")


# --------------------------------------------------------------------------- scaffold stage
CAM_NODES = (1, 3, 5)


def median_depth(node: int, rec: str, n_frames: int) -> tuple[np.ndarray, list[str]]:
    clip = S.depth_clip_dir(node, rec); frames = S.depth_frames(clip)
    sel = [frames[i] for i in np.linspace(0, len(frames) - 1, n_frames).astype(int)]
    st = np.stack([S.load_depth_m_calibrated(p, node) for p in sel]); st[st <= 0] = np.nan
    med = np.nanmedian(st, 0); med[np.isfinite(st).mean(0) < 0.5] = 0.0
    return np.nan_to_num(med), [os.path.basename(p) for p in sel]


def run_octonet_scaffold(args: argparse.Namespace) -> dict:
    """RGB-D (3 nodes, calibrated poses + depth scales) -> fused static scaffold in the 4DPS contract."""
    import torch
    from mm4DPS.reconstruction import voxel_downsample_torch, pca_normals_torch
    from method import sfcw_target as ST
    seq = OctoNetSequence(args.rec, args.data_out)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device
    clouds, sources, cams = [], [], {}
    normals_mode = getattr(args, "normals", "planar")
    for n in CAM_NODES:
        med, sel = median_depth(n, args.rec, args.frames)
        R, t, src = S.camera_pose(n)
        if normals_mode == "planar":
            # D16: robust local plane fits on the full-resolution median depth, normals carried per point, voxel-averaged
            from mm4DPS.octonet_geometry import planar_normals_depth, backproject_with_normals, voxel_average
            K = S.d455_nominal_intrinsics()
            nmap, okmap = planar_normals_depth(med, K, r=args.plane_radius, step_m=10.0 / 255.0 * S.depth_scale(n), device=device)
            pts_cam, n_cam = backproject_with_normals(med, nmap, okmap, K, stride=args.stride)
            w = pts_cam @ R.T + t; n_w = n_cam @ R.T
        else:
            pts_cam = S.backproject(med, stride=args.stride)
            w = pts_cam @ R.T + t; n_w = None
        r = np.hypot(w[:, 0], w[:, 1])
        person = (r < args.person_radius) & (w[:, 2] > 0.0) & (w[:, 2] < args.person_zmax)
        keep = (~person) & (w[:, 2] > -0.1) & (w[:, 2] < args.z_max)
        w = w[keep]
        if normals_mode == "planar":
            cloud_np, normals_np = voxel_average(w, n_w[keep], args.voxel_m)
            clouds.append(np.concatenate([cloud_np, normals_np], 1).astype(np.float32)); n_kept = len(cloud_np)
        else:
            # per-camera voxel + PCA normals on the device (4DPS's own routines, oriented toward that camera)
            cloud = voxel_downsample_torch(torch.as_tensor(w, dtype=torch.float64, device=device), args.voxel_m)
            normals = pca_normals_torch(cloud, max(0.15, args.voxel_m * 4), 40, torch.as_tensor(t, dtype=torch.float64, device=device))
            clouds.append(torch.cat([cloud.float(), normals.float()], 1).cpu().numpy()); n_kept = len(cloud)
        sources.append(np.full(n_kept, n, dtype=np.int8))
        cams[f"node_{n}"] = {"frames": sel, "R_wc": R.tolist(), "t_w": t.tolist(), "pose_source": src,
                             "depth_scale": S.depth_scale(n), "points_kept": int(n_kept), "normals": normals_mode}
    pcl = np.concatenate(clouds); src = np.concatenate(sources)
    np.save(out / "pcl.npy", pcl); np.save(out / "source_node.npy", src)
    shadow_meta = None
    if getattr(args, "shadow_mask", True):
        # D7: depth-buffer visibility from the Vayyar (the point-scaffold stand-in for 3DPS's mesh ray cast)
        from mm4DPS.octonet_geometry import visibility_mask
        R_v0, t_v0 = seq.sensor_pose("vayyar")
        vis = visibility_mask(pcl[:, :3], np.asarray(t_v0, float), np.asarray(R_v0)[:, 2], cell_deg=args.shadow_cell_deg, tol_m=args.shadow_tol_m)
        np.save(out / "shadow_mask.npy", vis)
        shadow_meta = {"visible_fraction": float(vis.mean()), "cell_deg": args.shadow_cell_deg, "tol_m": args.shadow_tol_m,
                       "method": "nearest point per (az, el) cell from the Vayyar within tol; back hemisphere occluded", "file": "shadow_mask.npy (True = visible)"}
    # the measured static target (pooled over Scene-1 recordings when available)
    target = seq.static_target(pooled=True)
    rp = np.asarray(target["rp_mean"]).astype(np.complex64)
    np.save(out / "vayyar_static_rp.npy", rp)
    np.savez(out / "vayyar_static_target.npz", **{k: np.asarray(v) for k, v in target.items() if not isinstance(v, (dict, str))})
    R_v, t_v = seq.sensor_pose("vayyar")
    metadata = {
        "dataset": "octonet", "recording": args.rec, "scene": "S1",
        "scene_source": "RGB-D: 3 RealSense D455 depth streams (nodes 1/3/5), 8-bit 0-10 m PNG x per-camera depth scale, temporal "
                        f"median over {args.frames} frames (the subject walks in place; the person column r<{args.person_radius} m is cut)",
        "camera_poses": "octonet_calibrate: node rigid body + floor height; floor-plane pitch/roll; MoCap-synchronised person yaw",
        "intrinsics": "nominal D455 87x58 deg, 640x480 (none shipped)",
        "cameras": cams, "point_count": int(len(pcl)), "merged_voxel_m": args.voxel_m,
        "pcl_columns": ["x_m", "y_m", "z_m", "normal_x", "normal_y", "normal_z"], "frame": "MoCap world, metres, Z up, origin = subject spot",
        "radar_target": "vayyar_static_rp.npy = pooled clutter-guarded temporal-mean range profiles (400 pairs x 100 bins), "
                        "nbs/method/sfcw_target.py; weights in vayyar_static_target.npz",
        "vayyar_pose": {"R_wc": R_v.tolist(), "t_w": t_v.tolist(), "assumption": "camera yaw, level, at the node_1 marker"},
        "sensors": {k: asdict(v) for k, v in SENSORS.items()},
        "normals": normals_mode, "shadow_mask": shadow_meta,
    }
    (out / "scaffold_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main_scaffold(argv=None) -> int:
    p = argparse.ArgumentParser(description="OctoNet scaffold stage (RGB-D + calibration -> 4DPS contract)")
    p.add_argument("--dataset", default="octonet"); p.add_argument("--rec", default=S.REC_WALK)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--data-out", type=Path, default=None)
    p.add_argument("--frames", type=int, default=30); p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel-m", type=float, default=0.05); p.add_argument("--person-radius", type=float, default=0.7)
    p.add_argument("--person-zmax", type=float, default=2.2); p.add_argument("--z-max", type=float, default=3.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--normals", choices=("planar", "pca"), default="planar", help="planar = robust plane fits on the full-res depth (D16); pca = the v1 voxel-cloud PCA")
    p.add_argument("--plane-radius", type=int, default=12, help="half window (px) of the plane fit (12 = 25 px; r=7 scored worse at iteration 0)")
    p.add_argument("--shadow-mask", dest="shadow_mask", action="store_true", default=True); p.add_argument("--no-shadow-mask", dest="shadow_mask", action="store_false")
    p.add_argument("--shadow-cell-deg", type=float, default=0.25); p.add_argument("--shadow-tol-m", type=float, default=0.15)
    args = p.parse_args(argv)
    meta = run_octonet_scaffold(args)
    print(json.dumps({k: meta[k] for k in ("recording", "point_count", "cameras")}, indent=1))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "scaffold":
        return main_scaffold(rest)
    if cmd == "sequence-info":
        p = argparse.ArgumentParser(); p.add_argument("--rec", default=S.REC_SIT); a = p.parse_args(rest)
        print(json.dumps(OctoNetSequence(a.rec).summary(), indent=1)); return 0
    print(f"unknown command {cmd!r}; commands: scaffold, sequence-info"); return 2


if __name__ == "__main__":
    sys.exit(main())
