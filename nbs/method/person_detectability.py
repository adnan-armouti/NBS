"""T9: is the person observable in SFCW / UWB / Wi-Fi? Physics-predicted residual vs measured residual.

Per recording and band:
  measured residual  = stream - its temporal mean (the static scene + coupling), energy per range bin at K frames
  predicted residual = render(static scaffold + person as DIRECT return + person as OCCLUDER of the scaffold)
                       - its mean over the same K frames, energy per range bin
  score = mean over the 'active' bins (top 20 % of predicted residual energy) of the Pearson correlation over the K
          frames between measured and predicted energy;  null = frame-order permutations of the prediction (1000)
  variants: direct only, occlusion only, both.
Model-free check: the MoCap repetition rate f0 (hip/hand speed spectrum) must show in the measured residual energy's
temporal spectrum above a permutation null.
The person mesh comes from the 4DPS person stage (SMPL fitted to the OptiTrack skeleton) at the Vayyar frame times;
UWB / Wi-Fi frames are the nearest in time (<= 0.1 s).

    cd nbs && CUDA_VISIBLE_DEVICES=4 python -m method.person_detectability [--recs ...] [--frames 30]
Outputs: $NBS_DATA_OUT/detectability_S1/{summary.json, <rec>_<band>.npz}
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S
from mm4DPS.octonet import OctoNetSequence, render_inputs
from mm4DPS import octonet_heads as OH
from method import wifi_forward as WF

DATA_OUT = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
OUT_DIR = os.path.join(DATA_OUT, "detectability_S1")
PERSON_DIR = os.path.join(DATA_OUT, "4dps_person_S1")
RECS = [("20240518134358", "sit", 2), ("20240518135416", "walk", 2), ("20240518140012", "sleep", 2),
        ("20240519164052", "bow", 1), ("20240519165055", "dance", 1), ("20240519165404", "jog", 1),
        ("20240521213123", "falldown", 3), ("20240521213436", "jump", 3), ("20240521213724", "jumpingjack", 3),
        ("20240523205938", "squat", 4), ("20240523210234", "lunge", 4), ("20240523210515", "turn", 4),
        ("20240525164329", "pushup", 5), ("20240525164620", "legraise", 5)]
BANDS = ("sfcw", "uwb", "wifi")
N_PERSON = 3000
N_SCAFFOLD = 10000
OCC_DEG = 1.0            # angular radius around a mesh vertex that shadows a scaffold point behind it
LOS_BLOCK_M = 0.25       # Tx-Rx segment closer than this to the mesh = LOS blocked
N_PERM = 1000
HANN = np.hanning(100)


# ----------------------------------------------------------------------------- geometry helpers
def occluded_mask(xyz_s: torch.Tensor, xyz_m: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """Scaffold points whose direction from ``origin`` is within OCC_DEG of a mesh vertex and farther than it."""
    ds = xyz_s - origin; Rs = ds.norm(dim=1); us = ds / Rs[:, None]
    dm = xyz_m - origin; Rm = dm.norm(dim=1); um = dm / Rm[:, None]
    cos = us @ um.T                                                           # (Ns, Nm)
    best, idx = cos.max(dim=1)
    return (best > np.cos(np.radians(OCC_DEG))) & (Rs > Rm[idx])


def segment_blocked(a: torch.Tensor, b: torch.Tensor, xyz_m: torch.Tensor) -> bool:
    """Does the segment a->b pass within LOS_BLOCK_M of any mesh vertex?"""
    ab = b - a; L2 = (ab @ ab).clamp(min=1e-9)
    t = ((xyz_m - a) @ ab / L2).clamp(0, 1)
    d = (xyz_m - (a + t[:, None] * ab)).norm(dim=1)
    return bool((d < LOS_BLOCK_M).any())


def person_cloud(npz, k: int, pick: np.ndarray, device) -> dict:
    v = torch.as_tensor(npz["vertices_world"][k][pick], dtype=torch.float64, device=device)
    n = torch.as_tensor(npz["normals_world"][k][pick], dtype=torch.float64, device=device)
    return {"xyz": v, "normals": n}


def uniform_material(n: int, device, like: dict) -> dict:
    return {"abcd": like["abcd"][:1].expand(n, -1).clone(), "rho": torch.ones(n, dtype=torch.float64, device=device)}


# ----------------------------------------------------------------------------- band renderers (complex / real "signal" per frame)
def render_sfcw(cloud, mat, sensor):
    return OH.sfcw_channel(cloud, mat, sensor)                                # (20,20,100) complex

def render_uwb(cloud, mat, sensor):
    return OH.uwb_head(cloud, mat, sensor)[0]                                  # rf (1535,) real

def render_wifi(cloud, mat, sensor):
    return OH.wifi_head(cloud, mat, sensor)                                    # (4,2,114) complex


def energy_sfcw(H):
    """H (...,20,20,100) complex -> |rp| mean over pairs per bin (..., 100) with the guard applied."""
    rp = np.fft.ifft(H * HANN, axis=-1)
    e = np.abs(rp).reshape(H.shape[:-3] + (400, 100)).mean(-2)
    e[..., :6] = 0; e[..., -6:] = 0
    return e

def energy_uwb(x):
    """x (..., 1535) real RF -> envelope via the analytic signal, summed in groups of 5 bins (3.2 cm) -> (..., 307)."""
    from scipy.signal import hilbert
    env = np.abs(hilbert(x, axis=-1))
    env[..., :12] = 0                                                         # direct coupling
    return env[..., :1535].reshape(x.shape[:-1] + (307, 5)).sum(-1)

def energy_wifi(H):
    """H (...,4,2,114) complex -> |CIR| per tap, antennas averaged, receivers concatenated (..., 4*114)."""
    cir = np.abs(np.fft.ifft(H, axis=-1)).mean(-2)                           # (...,4,114)
    return cir.reshape(H.shape[:-3] + (4 * 114,))


# ----------------------------------------------------------------------------- measured side
def measured(seq: OctoNetSequence, band: str, t_sel: np.ndarray):
    """Residual energy at the selected times (K, bins) and the full-rate residual energy time series for the
    repetition check (t_full, s_full)."""
    if band == "sfcw":
        V, t = seq.vayyar(); Hs = V.mean(0)
        k = np.array([int(np.argmin(np.abs(t - x))) for x in t_sel])
        E = energy_sfcw((V[k] - Hs).reshape(len(k), 20, 20, 100))
        full = energy_sfcw((V - Hs).reshape(len(V), 20, 20, 100))
        return E, t, full, float(np.max(np.abs(t[k] - t_sel)))
    if band == "uwb":
        U, t = seq.uwb(); Us = U.mean(0)
        k = np.array([int(np.argmin(np.abs(t - x))) for x in t_sel])
        E = energy_uwb(U[k] - Us); full = energy_uwb(U - Us)
        return E, t, full, float(np.max(np.abs(t[k] - t_sel)))
    if band == "wifi":
        Hs, ts, ks = [], None, []
        per_rx = []
        for r in (1, 2, 3, 4):
            D, t = seq.wifi(r); per_rx.append((D, t, D.mean(0)))
        # common clock: receiver 1's frames; others nearest in time
        t1 = per_rx[0][1]
        k1 = np.array([int(np.argmin(np.abs(t1 - x))) for x in t_sel])
        sel = []
        for (D, t, Dm) in per_rx:
            kk = np.array([int(np.argmin(np.abs(t - x))) for x in t_sel]); sel.append(D[kk] - Dm)
        E = energy_wifi(np.stack(sel, 1))                                      # (K,4,2,114) -> (K,456)
        # full-rate series on receiver 1 (75 Hz), the other receivers resampled to it
        step = max(1, len(t1) // 4000); tf = t1[::step]
        fulls = []
        for (D, t, Dm) in per_rx:
            kk = np.array([int(np.argmin(np.abs(t - x))) for x in tf]); fulls.append(D[kk] - Dm)
        full = energy_wifi(np.stack(fulls, 1))
        return E, tf, full, float(np.max(np.abs(t1[k1] - t_sel)))
    raise ValueError(band)


# ----------------------------------------------------------------------------- predicted side
def predicted(band: str, frames: list[int], npz, pick, scaffold: dict, mat_s: dict, sensors: dict, device):
    """Per variant ('direct','occlusion','both'): the energy of (signal(t) - mean_t signal) at the frames, (K, bins)."""
    render = {"sfcw": render_sfcw, "uwb": render_uwb, "wifi": render_wifi}[band]
    energy = {"sfcw": energy_sfcw, "uwb": energy_uwb, "wifi": energy_wifi}[band]
    sensor = sensors[band]
    with torch.no_grad():
        static = render(scaffold, mat_s, sensor)
        xyz_s = scaffold["xyz"]
        out = {"direct": [], "occlusion": [], "both": []}
        los_log = []
        for k in frames:
            pc = person_cloud(npz, k, pick, device); mat_p = uniform_material(len(pick), device, mat_s)
            direct = render(pc, mat_p, sensor)
            if band == "wifi":
                tx = torch.as_tensor(sensor["tx"], dtype=torch.float64, device=device)
                occ = occluded_mask(xyz_s, pc["xyz"], tx)
                los = {r: (sensor.get("los", {}).get(r, r != 4) and not segment_blocked(tx, torch.as_tensor(sensor["rx"][r], dtype=torch.float64, device=device), pc["xyz"])) for r in (1, 2, 3, 4)}
                los_log.append([int(los[r]) for r in (1, 2, 3, 4)])
                sensor_k = dict(sensor, los=los)
                static_k = render(scaffold, mat_s, sensor_k)                    # LOS toggle counts as occlusion
            else:
                origin = torch.as_tensor(sensor["pos"], dtype=torch.float64, device=device)
                occ = occluded_mask(xyz_s, pc["xyz"], origin); static_k = static
            keep = ~occ
            shadowed = render({"xyz": xyz_s[keep], "normals": scaffold["normals"][keep]},
                              {"abcd": mat_s["abcd"][keep], "rho": mat_s["rho"][keep]}, sensor_k if band == "wifi" else sensor)
            occl_effect = shadowed - static                                      # what the person removes
            out["direct"].append((static + direct).cpu().numpy())
            out["occlusion"].append((static + occl_effect).cpu().numpy())
            out["both"].append((static + direct + occl_effect).cpu().numpy())
    res = {}
    for v, sig in out.items():
        sig = np.stack(sig); sig = sig - sig.mean(0, keepdims=True)
        res[v] = energy(sig)
    return res, los_log


# ----------------------------------------------------------------------------- scoring
def score(E_meas: np.ndarray, E_pred: np.ndarray, rng, n_perm=N_PERM, top=0.2):
    act = np.argsort(E_pred.mean(0))[::-1][:max(3, int(top * E_pred.shape[1]))]
    def s(P):
        a = E_meas[:, act] - E_meas[:, act].mean(0); b = P[:, act] - P[:, act].mean(0)
        c = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-30)
        return float(np.nanmean(c))
    obs = s(E_pred)
    null = np.array([s(E_pred[rng.permutation(len(E_pred))]) for _ in range(n_perm)])
    return obs, float((obs - null.mean()) / (null.std() + 1e-12)), float((null >= obs).mean()), act


def repetition_rate(pos: np.ndarray, t: np.ndarray):
    """Dominant frequency (0.2-2.5 Hz) of the hip+hands speed, and its prominence (peak / median)."""
    dt = np.median(np.diff(t)); sp = np.linalg.norm(np.diff(pos[:, [0, 7, 11], :], axis=0), axis=-1).mean(1) / dt
    sp = sp - sp.mean(); n = len(sp); f = np.fft.rfftfreq(n, dt); P = np.abs(np.fft.rfft(sp * np.hanning(n))) ** 2
    band = (f > 0.2) & (f < 2.5); i = np.argmax(P * band); f0 = float(f[i]); prom = float(P[i] / (np.median(P[band]) + 1e-30))
    return f0, prom


def spectral_check(t: np.ndarray, s: np.ndarray, f0: float, rng, n_perm=N_PERM):
    """Power of the residual-energy series at f0 (+-0.06 Hz) vs the permutation null; needs f0 < Nyquist."""
    dt = np.median(np.diff(t)); n = len(s)
    if f0 >= 0.5 / dt or n < 32: return None
    f = np.fft.rfftfreq(n, dt); band = np.abs(f - f0) < 0.06
    def pw(x):
        x = x - x.mean(); P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2; return float(P[band].sum() / (P[1:].sum() + 1e-30))
    obs = pw(s); null = np.array([pw(s[rng.permutation(n)]) for _ in range(n_perm)])
    return {"obs": obs, "z": float((obs - null.mean()) / (null.std() + 1e-12)), "p": float((null >= obs).mean())}


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--recs", nargs="*", default=None); ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--bands", nargs="*", default=list(BANDS))
    a = ap.parse_args(argv); dev = a.device; os.makedirs(OUT_DIR, exist_ok=True); rng = np.random.default_rng(0)
    calib = os.path.join(DATA_OUT, "calib_S1", "extrinsics.json"); ext = json.load(open(calib))
    scaffold, mat_s, sensor_v = render_inputs(os.path.join(DATA_OUT, "scaffold_S1_calibrated"), ext, device=dev, n_points=N_SCAFFOLD)
    sensors = {"sfcw": sensor_v, "uwb": OH.sensor_from_extrinsics(ext, "uwb"),
               "wifi": {"tx": S.WIFI_TX_APPROX, "rx": {k: S.WIFI_RX_APPROX[k] for k in (1, 2, 3, 4)}, "los": {1: True, 2: True, 3: True, 4: False}}}
    recs = [r for r in RECS if (a.recs is None or r[0] in a.recs)]
    pick = np.random.default_rng(1).choice(6890, N_PERSON, replace=False)
    summary = {"recordings": {}, "config": {"frames": a.frames, "n_person": N_PERSON, "n_scaffold": N_SCAFFOLD, "occ_deg": OCC_DEG, "los_block_m": LOS_BLOCK_M, "n_perm": N_PERM}}
    for rec, act, subj in recs:
        pz = os.path.join(PERSON_DIR, rec, "person_sequence.npz")
        if not os.path.exists(pz):
            alt = os.path.join(DATA_OUT, "person", rec, "person_sequence.npz")
            if os.path.exists(alt): pz = alt
            else: print(f"{rec} {act}: no person_sequence.npz, skipped"); continue
        npz = np.load(pz); valid = np.nonzero(npz["valid"])[0]; tP = npz["t"]
        frames = [int(valid[i]) for i in np.linspace(int(0.05 * len(valid)), int(0.95 * len(valid)) - 1, a.frames).astype(int)]
        t_sel = tP[frames]
        seq = OctoNetSequence(rec); pos, tm = seq.mocap(); f0, prom = repetition_rate(pos, tm)
        entry = {"activity": act, "subject": subj, "f0_hz": f0, "f0_prominence": prom, "bands": {}}
        t0 = time.time()
        for band in a.bands:
            try:
                E_meas, t_full, s_full, dt_max = measured(seq, band, t_sel)
                preds, los_log = predicted(band, frames, npz, pick, scaffold, mat_s, sensors, dev)
                b = {"max_dt_s": dt_max, "scores": {}}
                for v in ("direct", "occlusion", "both"):
                    obs, z, p, actb = score(E_meas, preds[v], rng); b["scores"][v] = {"corr": obs, "z": z, "p": p, "n_active_bins": int(len(actb))}
                # repetition check on the full-rate measured residual energy summed over the 'both' active bins
                actb = np.argsort(preds["both"].mean(0))[::-1][:max(3, int(0.2 * preds["both"].shape[1]))]
                b["repetition"] = spectral_check(t_full, s_full[:, actb].sum(1), f0, rng)
                if band == "wifi": b["los_blocked_frames"] = (1 - np.array(los_log)).sum(0).tolist()
                entry["bands"][band] = b
                np.savez(os.path.join(OUT_DIR, f"{rec}_{band}.npz"), E_meas=E_meas, E_pred_direct=preds["direct"], E_pred_occlusion=preds["occlusion"],
                         E_pred_both=preds["both"], t_sel=t_sel, t_full=t_full, s_full=s_full[:, actb].sum(1), active=actb)
                print(f"{rec} {act:12s} {band:5s} z: direct {b['scores']['direct']['z']:+5.1f} occl {b['scores']['occlusion']['z']:+5.1f} both {b['scores']['both']['z']:+5.1f}"
                      f" | corr both {b['scores']['both']['corr']:+.3f} | rep f0 {f0:.2f} Hz (prom {prom:.0f}) z {b['repetition']['z'] if b['repetition'] else float('nan'):+.1f} | {time.time()-t0:.0f} s", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc(); entry["bands"][band] = {"error": repr(e)}
        summary["recordings"][rec] = entry
        json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=1)
    # decision table
    table = {}
    for band in a.bands:
        for v in ("direct", "occlusion", "both"):
            zs = [e["bands"][band]["scores"][v]["z"] for e in summary["recordings"].values() if band in e["bands"] and "scores" in e["bands"][band]]
            subj = {}
            for e in summary["recordings"].values():
                if band in e["bands"] and "scores" in e["bands"][band]: subj.setdefault(e["subject"], []).append(e["bands"][band]["scores"][v]["z"])
            table[f"{band}/{v}"] = {"n": len(zs), "median_z": float(np.median(zs)) if zs else None, "frac_z_gt_3": float(np.mean(np.array(zs) > 3)) if zs else None,
                                    "subjects_median_z": {str(k): float(np.median(x)) for k, x in subj.items()}}
        reps = [e["bands"][band]["repetition"]["z"] for e in summary["recordings"].values() if band in e["bands"] and e["bands"][band].get("repetition") and e["f0_prominence"] > 3]
        table[f"{band}/repetition"] = {"n_periodic": len(reps), "median_z": float(np.median(reps)) if reps else None, "frac_z_gt_3": float(np.mean(np.array(reps) > 3)) if reps else None}
    summary["decision_table"] = table
    for band in a.bands:
        tb = table[f"{band}/both"]; ok = tb["median_z"] is not None and tb["median_z"] > 3 and all(v > 3 for v in tb["subjects_median_z"].values())
        summary.setdefault("verdict", {})[band] = ("PASS" if ok else "FAIL") + f" (median z both {tb['median_z']}, per-subject {tb['subjects_median_z']})"
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=1)
    print(json.dumps(table, indent=1)); print(summary["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
