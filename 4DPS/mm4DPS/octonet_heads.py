"""OctoNet rendering heads for the 4DPS copy: SFCW (Vayyar), Wi-Fi CSI, UWB CIR.

Each head is a free function with ONE shared input convention and returns the measured-domain
quantity the matching target is stored in. No Doppler anywhere (none of the three modalities
records slow-time Doppler); the range kernel is what differs per head:

    SFCW  : stepped-frequency channel -> Hann IFFT over the 100 tones (3.75 cm bins)
    Wi-Fi : 114 subcarriers over 40 MHz, bistatic Tx -> scatterer -> Rx (+ explicit LOS)
    UWB   : the 3DPS rasterizer in UWB mode -- the measured X4 pulse template as the range kernel, on-board delay as a bin offset (the point sum is deprecated)

Shared input convention (what ``mm4DPS.octonet`` produces from the scaffold + calibration):
    cloud    : dict(xyz (N,3) float64 tensor [MoCap world, m], normals (N,3) float64 tensor or None)
    material : dict(abcd (N,4) float64 tensor = ITU-R P.2040 dispersion coefficients (a,b,c,d), the
               NBS parametrisation; rho (N,) float64 tensor = per-point real reflectivity scale (1 =
               the material's own Fresnel coefficient). Either may require grad.)
    sensor   : SFCW / UWB: dict(pos (3,), boresight (3,)) [world]; both at node_1 (pos = rigid body,
               boresight = calibrated camera yaw, horizontal, an assumption -- see extrinsics.json)
               Wi-Fi: dict(tx (3,), rx {k: (3,)}, los {k: bool}) [world, schematic +-0.5 m]

The SFCW math is single-sourced in nbs/method/sfcw_forward.py (reflectivity, geometry, phase);
``sfcw_head`` is the 3DPS rasterizer (render_factorized) on the Vayyar SFCW config (T23); the old point sum is deprecated.
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

# NBS code (nbs/ is on PYTHONPATH via tools/env_node.sh)
from method import sfcw_forward as SF
from method import wifi_forward as WF
from method.freq_material import C_LIGHT, ITU_GT, reflectivity as itu_reflectivity
from preprocessing import vayyar_radar_config as VR
from preprocessing import octonet_config as C
from preprocessing import octonet_scene as S
from mm3DPS.data import ra_utils as RA

SFCW_FREQS_HZ = SF.sfcw_freqs_hz(VR.F_START_HZ, VR.F_STOP_HZ, VR.N_FREQ)      # (100,)
SFCW_BIN_M = VR.RANGE_RES_M                                                   # 0.0375
WIFI_SC_GHZ = WF.subcarrier_ghz(fc_ghz=S.WIFI["fc_hz"] / 1e9, bw_mhz=S.WIFI["bw_hz"] / 1e6, n_sc=S.WIFI["n_sub"])
# UWB constants from the DATA (nbs/method/uwb_template.py, T16): RF-sampled frames at 23.328 GS/s, carrier 7.0 GHz,
# -10 dB bandwidth 1.46 GHz (the paper's 2.5 GHz is not what the frames show), on-board delay = coupling peak at bin 7.
UWB_FS_HZ = 23.328e9
UWB_BIN_M = C_LIGHT / (2.0 * UWB_FS_HZ)              # 6.43 mm one-way per bin (9.9 m / 1535 agrees)
UWB_N_BINS = S.UWB["n_bins"]
UWB_FC_HZ, UWB_BW_HZ = 7.0e9, 1.46e9                 # fallback Gaussian only; the measured template is used when present
UWB_TEMPLATE_PATH = os.path.join(os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT), "uwb_template_S1", "template.npz")
_UWB_TEMPLATE = {}


def uwb_template(device):
    """(t_rf (K,), t_env (K,), centre index, on-board delay in bins) from the measured coupling peak, or None."""
    key = str(device)
    if key not in _UWB_TEMPLATE:
        if os.path.exists(UWB_TEMPLATE_PATH):
            z = np.load(UWB_TEMPLATE_PATH)
            t_rf = torch.as_tensor(z["t_rf"], dtype=torch.float64, device=device); t_env = torch.as_tensor(z["t_env"], dtype=torch.float64, device=device)
            half = int(z["half_width_bins"]) if "half_width_bins" in z.files else len(t_rf) // 2
            peak = int(round(float(z["per_recording_peak_bin"]))) if "per_recording_peak_bin" in z.files else 7
            _UWB_TEMPLATE[key] = (t_rf / t_rf.abs().max(), t_env / t_env.max(), half, peak)
        else:
            _UWB_TEMPLATE[key] = None
    return _UWB_TEMPLATE[key]


def _interp1(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ``table`` (K,) at fractional indices ``x`` (differentiable in x), 0 outside."""
    K = table.shape[0]
    x0 = torch.floor(x); f = x - x0; i0 = x0.long()
    inside = (i0 >= 0) & (i0 < K - 1)
    i0c = i0.clamp(0, K - 2)
    y = table[i0c] * (1 - f) + table[i0c + 1] * f
    return torch.where(inside, y, torch.zeros_like(y))


# ----------------------------------------------------------------------------- helpers
def _t(x, device, dtype=torch.float64):
    return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, float), dtype=dtype, device=device)


def point_reflectivity(cloud: dict, material: dict, f_ghz, w_in: torch.Tensor | None) -> torch.Tensor:
    """(N,Nf) complex128 Fresnel reflectivity of each point at the given frequencies (GHz, tensor
    (Nf,) or scalar), at the incidence the hit->sensor direction ``w_in`` (N,3) makes with the
    normal (normal incidence when normals/w_in are None), times the per-point scale ``rho``."""
    abcd = material["abcd"]
    cos_i = SF._cos_incidence(w_in, cloud.get("normals")) if w_in is not None else 1.0
    fg = f_ghz if torch.is_tensor(f_ghz) else torch.as_tensor(np.atleast_1d(np.asarray(f_ghz, float)), dtype=abcd.dtype, device=abcd.device)
    refl = SF._reflectivity_per_freq(abcd, fg, cos_i).to(torch.complex128)                 # (N,Nf)
    rho = material.get("rho")
    if rho is not None:
        refl = refl * rho.to(torch.complex128)[:, None]
    return refl


def vayyar_arrays(sensor: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tx (20,3) and Rx (20,3) world positions (m) from the NBS config builder (uniform-pitch L-array
    assumption: azimuth/RA unverified, range unaffected)."""
    cfg = VR.build_config(radar_pos=np.asarray(sensor["pos"], float), boresight=np.asarray(sensor["boresight"], float))
    tx = torch.tensor([a["pos_mm"] for a in cfg["tx_array"]], dtype=torch.float64, device=device) / 1000.0
    rx = torch.tensor([a["pos_mm"] for a in cfg["rx_array"]], dtype=torch.float64, device=device) / 1000.0
    return tx, rx


def sensor_from_extrinsics(ext: dict, key: str = "vayyar") -> dict:
    R = np.asarray(ext[key]["R_wc"], float); bore = R[:, 2].copy(); bore[2] = 0.0; bore /= np.linalg.norm(bore)
    return dict(pos=np.asarray(ext[key]["t_w"], float), boresight=bore)


# ----------------------------------------------------------------------------- SFCW head
def sfcw_channel_pointsum_deprecated(cloud: dict, material: dict, sensor: dict, chunk: int = 4096) -> torch.Tensor:
    _deprecated_guard("sfcw_channel_pointsum_deprecated")
    """Monostatic SFCW channel H (20 Tx, 20 Rx, 100) complex128 -- the vectorised twin of
    ``sfcw_forward.render_sfcw_pairs(geom_mode='monostatic')``: reflectivity once per point at the
    backscatter direction toward the array centre (with the point's normal), per-pair path phase,
    1/(Rt*Rr) spreading. Chunked over points; differentiable."""
    xyz = cloud["xyz"]; dev = xyz.device
    tx, rx = vayyar_arrays(sensor, dev)
    f = torch.as_tensor(SFCW_FREQS_HZ, dtype=torch.float64, device=dev)                   # (Nf,)
    c0 = 0.5 * (tx.mean(0) + rx.mean(0))
    d0 = (xyz - c0).norm(dim=1).clamp(min=1e-9)
    w_in = (c0 - xyz) / d0[:, None]
    refl = point_reflectivity(cloud, material, f / 1e9, w_in)                              # (N,Nf)
    H = torch.zeros(tx.shape[0], rx.shape[0], f.shape[0], dtype=torch.complex128, device=dev)
    k2pi = -2j * math.pi
    for lo in range(0, xyz.shape[0], chunk):
        p = xyz[lo:lo + chunk]; r = refl[lo:lo + chunk]                                    # (M,3), (M,Nf)
        Rt = torch.cdist(tx, p)                                                            # (Ntx,M)
        Rr = torch.cdist(rx, p)                                                            # (Nrx,M)
        for it in range(tx.shape[0]):
            tau = (Rt[it][None, :] + Rr) / C_LIGHT                                         # (Nrx,M)
            geom = (1.0 / (Rt[it][None, :] * Rr + 1e-9)).to(torch.complex128)             # (Nrx,M)
            ph = torch.exp(k2pi * tau[..., None] * f[None, None, :])                       # (Nrx,M,Nf)
            H[it] = H[it] + torch.einsum("rm,mf,rmf->rf", geom, r, ph)
    return H


# ============================================================================ SFCW head = the 3DPS rasterizer (T23)
# The SFCW head IS mm3DPS.rasterizer_factorized.render_factorized (Jones BSDF with the six material parameters, antenna
# element patterns, the MC hemisphere weight, Hann range PSF, detached phase) on the Vayyar SFCW config built at the
# calibrated pose. The only OctoNet-specific additions: epsilon from the ITU dispersion law at the band centre, a
# datasheet stand-in antenna pattern, and the 20 + 20 per-channel gains of the raw Vayyar phasors. The earlier point-sum
# head is kept only as `sfcw_head_pointsum_deprecated` and raises unless OCTONET_ALLOW_DEPRECATED_SFCW=1
# (docs/findings/sfcw_head_vs_3dps.md).
SFCW_FC_GHZ = 65.0
SFCW_DEFAULT_ROUGHNESS = dict(sigma_h=1.0e-4, l_c=1.0e-2, tau=0.5, thickness=0.1)      # 3DPS physics units (m, m, -, m)
VAYYAR_PATTERN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "nbs", "assets", "antenna_pattern", "VAYYAR")
_SFCW_RAST = {}


def _deprecated_guard(name: str):
    if os.environ.get("OCTONET_ALLOW_DEPRECATED_SFCW") != "1":
        raise RuntimeError(f"{name} is the pre-T23 point-sum SFCW model (no Jones BSDF, 1/R^4, no antenna gain, attached phase); "
                           "it is wrong for surfaces at 65 GHz. Use octonet_heads.sfcw_head (3DPS render_factorized). "
                           "Set OCTONET_ALLOW_DEPRECATED_SFCW=1 only for the historical comparison.")


def write_vayyar_pattern(path: str, e_bw_deg: float = 180.0, h_bw_deg: float = 82.0, gmax_db: float = 5.8) -> np.ndarray:
    """Datasheet stand-in element pattern in the 3DPS (361,2) [E_dB, H_dB] format: cos^p planes whose 3 dB beamwidths
    are the IMAGEVK-74 values (E 180 deg, H 82 deg, 5.8 dBi). Replace with a measured pattern when one exists."""
    ang = np.deg2rad(np.arange(361) - 180.0)
    def plane(bw):
        p = math.log(0.5) / math.log(math.cos(math.radians(bw / 2))) if bw < 179 else 0.05
        return gmax_db + 10 * np.log10(np.clip(np.cos(ang), 1e-3, 1) ** p)
    arr = np.stack([plane(e_bw_deg), plane(h_bw_deg)], 1).astype(np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, arr); return arr


def vayyar_rasterizer(sensor: dict, device):
    """The 3DPS Rasterizer for the Vayyar SFCW config at ``sensor['pos']`` / ``sensor['boresight']`` (cached per pose)."""
    from mm3DPS import rasterizer as RZ
    pos = np.round(np.asarray(sensor["pos"], float), 6); bore = np.asarray(sensor["boresight"], float); bore = bore / np.linalg.norm(bore)
    key = (tuple(pos), tuple(np.round(bore, 6)), str(device))
    if key not in _SFCW_RAST:
        import json, hashlib
        cfg = VR.build_config(radar_pos=pos, boresight=bore)
        h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        cfg_dir = os.path.join(os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT), "sfcw_configs"); os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, f"vayyar_sfcw_{h}.json")
        if not os.path.exists(cfg_path):
            json.dump(cfg, open(cfg_path, "w"), indent=1)
        tx_pat = os.path.join(VAYYAR_PATTERN_DIR, "tx_65.npy"); rx_pat = os.path.join(VAYYAR_PATTERN_DIR, "rx_65.npy")
        for pth in (tx_pat, rx_pat):
            if not os.path.exists(pth):
                write_vayyar_pattern(pth)
        _SFCW_RAST[key] = RZ.Rasterizer(cfg_path, "/none.ply", tx_pat, rx_pat, device=str(device))
    return _SFCW_RAST[key]


def sfcw_physics(cloud: dict, material: dict, f_ghz: float = SFCW_FC_GHZ) -> torch.Tensor:
    """The 3DPS six-parameter physics (M,6) = [eps_r, eps_i, sigma_h, l_c, tau, thickness] from the OctoNet material dict:
    either ``raw6`` (3DPS raw space, through reparameterize_torch) or ``abcd`` (ITU dispersion -> eps at the band centre)
    plus optional per-point ``sigma_h``/``l_c``/``tau``/``thickness`` (defaults SFCW_DEFAULT_ROUGHNESS)."""
    from mm3DPS.rasterizer import reparameterize_torch
    from method.freq_material import eps_from_dispersion
    xyz = cloud["xyz"]; M = xyz.shape[0]; dev = xyz.device
    if "physics" in material:                                   # (M,6) physics directly (e.g. a saved fit)
        return material["physics"].float()
    if "raw6" in material:
        return reparameterize_torch(material["raw6"].float())
    if "rho" in material:
        r = material["rho"]
        if torch.is_tensor(r) and (r.detach() != 1).any():
            raise ValueError("per-point 'rho' is not a 3DPS material parameter; optimise raw6 (eps, sigma_h, l_c, tau, thickness) or abcd")
    er, ei = eps_from_dispersion(material["abcd"], f_ghz)
    cols = [er.float(), ei.float()]
    for k in ("sigma_h", "l_c", "tau", "thickness"):
        v = material.get(k, None)
        cols.append(v.float().expand(M) if torch.is_tensor(v) else torch.full((M,), float(SFCW_DEFAULT_ROUGHNESS[k] if v is None else v), device=dev))
    return torch.stack(cols, 1)


def sfcw_head(cloud: dict, material: dict, sensor: dict, chunk: int | None = None, detach_phase: bool = True,
              use_cuda_kernels: bool = False, shadow_mask=None) -> torch.Tensor:
    """SFCW pair-wise range profiles (400,100) complex128, tx-major, through the 3DPS rasterizer.

    cloud    {xyz (N,3), normals (N,3)}  MoCap world, metres (normals are flipped toward the radar: two-sided points)
    material {abcd (N,4) [+ sigma_h, l_c, tau, thickness]} or {raw6 (N,6)}; see sfcw_physics
    sensor   {pos (3,), boresight (3,) [, gain_tx (20,), gain_rx (20,)]}  linear amplitude channel gains (default 1)
    """
    from mm3DPS.rasterizer_factorized import render_factorized
    xyz = cloud["xyz"]; dev = xyz.device
    rast = vayyar_rasterizer(sensor, dev)
    pos = torch.as_tensor(sensor["pos"], dtype=torch.float64, device=dev)
    w_in = pos[None, :] - xyz; nrm = cloud["normals"]
    if nrm is None:
        nrm = w_in / w_in.norm(dim=1, keepdim=True)
    else:
        flip = (nrm * w_in).sum(1, keepdim=True) < 0
        nrm = torch.where(flip, -nrm, nrm)
    physics = sfcw_physics(cloud, material)
    areas = torch.ones(xyz.shape[0], device=dev, dtype=torch.float32)
    if shadow_mask is None and cloud.get("shadow_mask", None) is not None:
        shadow_mask = cloud["shadow_mask"]                       # (N,) or (N, n_tx) bool: False = occluded from the radar
    if shadow_mask is not None and shadow_mask.dim() == 1:
        shadow_mask = shadow_mask[:, None].expand(-1, rast.n_tx)
    rr, ri = render_factorized(xyz.float(), nrm.float(), areas, physics, rast, lambda phys: phys,
                               detach_phase=detach_phase, shadow_mask=shadow_mask, use_cuda_kernels=use_cuda_kernels)
    rp = torch.complex(rr.double(), ri.double()).reshape(rast.n_tx * rast.n_rx, rast.K)      # tx-major pairs
    g_tx = sensor.get("gain_tx", None); g_rx = sensor.get("gain_rx", None)
    if g_tx is not None or g_rx is not None:
        g_tx = torch.ones(rast.n_tx, dtype=torch.float64, device=dev) if g_tx is None else _t(g_tx, dev)
        g_rx = torch.ones(rast.n_rx, dtype=torch.float64, device=dev) if g_rx is None else _t(g_rx, dev)
        rp = rp * (g_tx[:, None] * g_rx[None, :]).reshape(-1, 1).to(rp.dtype)
    return rp


def sfcw_head_pointsum_deprecated(cloud: dict, material: dict, sensor: dict, chunk: int = 4096) -> torch.Tensor:
    _deprecated_guard("sfcw_head_pointsum_deprecated")
    """Pair-wise range profiles rp (400, 100) complex128 in the target's layout: Hann-windowed IFFT
    over the tones (``ra_utils.sfcw_range_compress``), pairs flattened tx-major (p = tx*20 + rx,
    ``octonet_config.VAY_PAIR_ORDER``)."""
    H = sfcw_channel_pointsum_deprecated(cloud, material, sensor, chunk)
    rp = RA.sfcw_range_compress(H)                                                        # (20,20,100)
    return rp.reshape(-1, rp.shape[-1])


# ----------------------------------------------------------------------------- Wi-Fi head
def wifi_rx_antennas(center, device) -> torch.Tensor:
    lam = C_LIGHT / S.WIFI["fc_hz"]
    c = np.asarray(center, float)
    return torch.tensor([c + np.array([-lam / 4, 0, 0]), c + np.array([lam / 4, 0, 0])], dtype=torch.float64, device=device)


def wifi_head(cloud: dict, material: dict, sensor: dict) -> torch.Tensor:
    """CSI (4 receivers, 2 antennas, 114 subcarriers) complex128 via ``wifi_forward.render_csi``
    (bistatic single bounce, reflectivity at the 5.18 GHz centre, incidence at the half-vector to
    Tx and the receiver centre), plus an explicit LOS term 1/d exp(-j2pi f d/c) for receivers with
    ``los[k]`` True (Rx-4 sits behind the obstacle). Positions are schematic (+-0.5 m)."""
    xyz = cloud["xyz"]; dev = xyz.device
    tx = _t(sensor["tx"], dev).reshape(1, 3)
    f = torch.as_tensor(WIFI_SC_GHZ, dtype=torch.float64, device=dev) * 1e9
    out = []
    for k in (1, 2, 3, 4):
        rxc = _t(sensor["rx"][k], dev)
        rx = wifi_rx_antennas(sensor["rx"][k], dev)
        w_t = (tx[0] - xyz); w_t = w_t / w_t.norm(dim=1, keepdim=True).clamp(min=1e-9)
        w_r = (rxc - xyz); w_r = w_r / w_r.norm(dim=1, keepdim=True).clamp(min=1e-9)
        amp = point_reflectivity(cloud, material, S.WIFI["fc_hz"] / 1e9, 0.5 * (w_t + w_r))[:, 0]   # (N,)
        H = WF.render_csi(amp, xyz, tx, rx, WIFI_SC_GHZ)                                   # (2,114)
        if sensor.get("los", {}).get(k, k != 4):
            d = (rx - tx[0]).norm(dim=1)                                                   # (2,)
            H = H + (1.0 / d)[:, None].to(torch.complex128) * torch.exp(-2j * math.pi * f[None, :] * (d / C_LIGHT)[:, None])
        out.append(H)
    return torch.stack(out)                                                                # (4,2,114)


# ----------------------------------------------------------------------------- UWB head
def uwb_pulse(r_rel: torch.Tensor, rf: bool) -> torch.Tensor:
    """Range-domain pulse at offset ``r_rel`` (m, one-way) from a path's delay: the MEASURED X4 template (RF with
    carrier, or its envelope) from the direct-coupling peak when available, else a Gaussian of the measured
    bandwidth with the carrier cos(4 pi fc r / c). The measured frame is RF-sampled, so ``rf=True`` is the
    comparable output; the envelope is the incoherent fallback."""
    tpl = uwb_template(r_rel.device)
    if tpl is not None:
        t_rf, t_env, half, _ = tpl
        return _interp1(t_rf if rf else t_env, r_rel / UWB_BIN_M + half)
    sig = (C_LIGHT / UWB_BW_HZ) / 2.0 / 2.355
    env = torch.exp(-0.5 * (r_rel / sig) ** 2)
    if rf:
        env = env * torch.cos(4.0 * math.pi * UWB_FC_HZ * r_rel / C_LIGHT)
    return env


def uwb_range_offset_m(device="cpu") -> float:
    """Measured ranges = true one-way range + the on-board delay (the coupling peak's bin)."""
    tpl = uwb_template(device)
    return (tpl[3] * UWB_BIN_M) if tpl is not None else 0.0


def uwb_head_pointsum_deprecated(cloud: dict, material: dict, sensor: dict, chunk: int = 8192, beam_deg: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    _deprecated_guard("uwb_head_pointsum_deprecated")
    """Monostatic UWB CIR on the X4's 1535-bin one-way range grid: sum over points of
    |refl(fc)| / (Rt*Rr) * pulse(r - (Rt+Rr)/2). Returns (cir_rf, cir_env): the coherent real RF
    waveform (carrier at fc, phases from the path) and the incoherent envelope sum (what the
    measured frame's magnitude is compared against; the measured frame is real baseband amplitude,
    so neither is exactly it -- state in captions). ``beam_deg``: optional hard cone half-angle
    about the boresight (the X4 beam is ~65 deg wide)."""
    xyz = cloud["xyz"]; dev = xyz.device
    pos = _t(sensor["pos"], dev); bore = _t(sensor["boresight"], dev); bore = bore / bore.norm()
    rel = xyz - pos; R = rel.norm(dim=1).clamp(min=1e-9)
    w_in = -rel / R[:, None]
    amp = point_reflectivity(cloud, material, UWB_FC_HZ / 1e9, w_in)[:, 0].abs() / (R * R)   # (N,)
    if beam_deg is not None:
        cosang = (rel @ bore) / R
        amp = amp * (cosang > math.cos(math.radians(beam_deg))).to(amp.dtype)
    grid = torch.arange(UWB_N_BINS, dtype=torch.float64, device=dev) * UWB_BIN_M - uwb_range_offset_m(dev)   # (B,) measured bin -> true range
    rf = torch.zeros(UWB_N_BINS, dtype=torch.float64, device=dev); env = torch.zeros_like(rf)
    for lo in range(0, xyz.shape[0], chunk):
        rr = grid[None, :] - R[lo:lo + chunk, None]                                           # (M,B)
        a = amp[lo:lo + chunk, None]
        env = env + (a * uwb_pulse(rr, rf=False)).sum(0)
        rf = rf + (a * uwb_pulse(rr, rf=True)).sum(0)
    return rf, env


# ============================================================================ UWB head = the 3DPS rasterizer (UWB mode)
# Same route as the SFCW head (T23): mm3DPS.rasterizer_factorized.render_factorized with the X4 config
# (nbs/preprocessing/x4_radar_config.py): Jones BSDF with the six material parameters (epsilon from the ITU law at 7 GHz),
# a datasheet stand-in element pattern (65 deg), the MC hemisphere weight, detached phase, and -- the UWB-specific
# minimal changes -- the MEASURED pulse template as the range kernel (carrier included, per-path carrier phase zero:
# a path delay is a pure time shift of the pulse) and the on-board delay as a bin offset. Output is the real RF frame
# on the 1535-bin grid plus its Hilbert envelope. The old point sum is `uwb_head_pointsum_deprecated` (guarded).
X4_PATTERN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "nbs", "assets", "antenna_pattern", "X4")
_UWB_RAST = {}


def x4_rasterizer(sensor: dict, device):
    """The 3DPS Rasterizer in UWB mode for the X4 at ``sensor['pos']`` / ``sensor['boresight']`` (cached per pose)."""
    from mm3DPS import rasterizer as RZ
    from preprocessing import x4_radar_config as X4
    pos = np.round(np.asarray(sensor["pos"], float), 6); bore = np.asarray(sensor["boresight"], float); bore = bore / np.linalg.norm(bore)
    key = (tuple(pos), tuple(np.round(bore, 6)), str(device))
    if key not in _UWB_RAST:
        import json, hashlib
        cfg = X4.build_config(radar_pos=pos, boresight=bore)
        h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        cfg_dir = os.path.join(os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT), "uwb_configs"); os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, f"x4_uwb_{h}.json")
        if not os.path.exists(cfg_path):
            json.dump(cfg, open(cfg_path, "w"), indent=1)
        tx_pat = os.path.join(X4_PATTERN_DIR, "tx_7.npy"); rx_pat = os.path.join(X4_PATTERN_DIR, "rx_7.npy")
        for pth in (tx_pat, rx_pat):
            if not os.path.exists(pth):
                write_vayyar_pattern(pth, e_bw_deg=X4.BEAMWIDTH_DEG, h_bw_deg=X4.BEAMWIDTH_DEG, gmax_db=5.0)
        _UWB_RAST[key] = RZ.Rasterizer(cfg_path, "/none.ply", tx_pat, rx_pat, device=str(device))
    return _UWB_RAST[key]


def hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    """|analytic signal| of a real 1-D tensor (differentiable)."""
    n = x.shape[-1]; X = torch.fft.fft(x.to(torch.float64))
    h = torch.zeros(n, dtype=torch.float64, device=x.device); h[0] = 1; h[1:(n + 1) // 2] = 2
    if n % 2 == 0: h[n // 2] = 1
    return torch.fft.ifft(X * h).abs()


def uwb_head(cloud: dict, material: dict, sensor: dict, chunk: int | None = None, detach_phase: bool = True,
             shadow_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
    """UWB frame through the 3DPS rasterizer: returns (cir_rf (1535,) real, cir_env (1535,)) on the X4's measured
    bin grid (on-board delay included). cloud/material/sensor as for sfcw_head (material at 7.0 GHz)."""
    from mm3DPS.rasterizer_factorized import render_factorized
    from preprocessing import x4_radar_config as X4
    xyz = cloud["xyz"]; dev = xyz.device
    rast = x4_rasterizer(sensor, dev)
    pos = torch.as_tensor(sensor["pos"], dtype=torch.float64, device=dev)
    w_in = pos[None, :] - xyz; nrm = cloud["normals"]
    if nrm is None:
        nrm = w_in / w_in.norm(dim=1, keepdim=True)
    else:
        flip = (nrm * w_in).sum(1, keepdim=True) < 0
        nrm = torch.where(flip, -nrm, nrm)
    physics = sfcw_physics(cloud, material, f_ghz=X4.FC_HZ / 1e9)
    areas = torch.ones(xyz.shape[0], device=dev, dtype=torch.float32)
    if shadow_mask is None and cloud.get("shadow_mask", None) is not None:
        shadow_mask = cloud["shadow_mask"]
    if shadow_mask is not None and shadow_mask.dim() == 1:
        shadow_mask = shadow_mask[:, None].expand(-1, rast.n_tx)
    rr, ri = render_factorized(xyz.float(), nrm.float(), areas, physics, rast, lambda phys: phys,
                               detach_phase=detach_phase, shadow_mask=shadow_mask, use_cuda_kernels=False)
    rf = rr.double().reshape(-1)                                       # (1535,) real RF frame (imag is zero by construction)
    return rf, hilbert_envelope(rf)


# ----------------------------------------------------------------------------- self-test
def _cloud(points, normals=None, device="cpu"):
    xyz = torch.tensor(np.asarray(points, float), dtype=torch.float64, device=device)
    return dict(xyz=xyz, normals=None if normals is None else torch.tensor(np.asarray(normals, float), dtype=torch.float64, device=device))


def _material(n, device, grad=False):
    abcd = torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=device).expand(n, 4).clone()
    rho = torch.ones(n, dtype=torch.float64, device=device)
    if grad:
        abcd.requires_grad_(True); rho.requires_grad_(True)
    return dict(abcd=abcd, rho=rho)


def selftest(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> bool:
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    pos0 = np.array([2.921, 0.011, 0.901]); bore = np.array([-1.0, 0.0, 0.0])
    sensor = dict(pos=pos0, boresight=bore)
    # --- SFCW (T23): the head IS render_factorized; the deprecated point sum must raise
    try:
        sfcw_head_pointsum_deprecated(_cloud([[0, 0, 0]], None, device), _material(1, device), sensor); raised = False
    except RuntimeError:
        raised = True
    check("SFCW deprecated point-sum head raises", raised)
    from mm3DPS.rasterizer_factorized import render_factorized
    rng = np.random.default_rng(0)
    cl = _cloud(rng.normal(size=(50, 3)) * 0.5 + pos0 + bore * 2.5, np.tile(-bore, (50, 1)), device); mat = _material(50, device, grad=True)
    rast = vayyar_rasterizer(sensor, device)
    rr, ri = render_factorized(cl["xyz"].float(), cl["normals"].float(), torch.ones(50, device=device), sfcw_physics(cl, mat), rast, lambda p: p, detach_phase=True, use_cuda_kernels=False)
    H_ref = torch.complex(rr.double(), ri.double()).reshape(400, -1); H_new = sfcw_head(cl, mat, sensor)
    rel = float((H_new - H_ref).abs().max() / H_ref.abs().max().clamp_min(1e-30))
    check("SFCW head == direct render_factorized (Vayyar SFCW config, 3DPS physics)", rel < 1e-6, f"(max rel diff {rel:.2e})")
    for R0 in (2.0,):
        cl1 = _cloud([pos0 + bore * R0], [-bore], device); m1 = _material(1, device, grad=True)
        rp = sfcw_head(cl1, m1, sensor); prof = rp.abs().mean(0); k = int(prof[3:-3].argmax()) + 3
        check(f"SFCW point target {R0} m -> bin {k} (expect {round(R0 / SFCW_BIN_M)})", abs(k - round(R0 / SFCW_BIN_M)) <= 1)
        (prof.sum()).backward(); check("SFCW gradient flows to abcd", m1["abcd"].grad is not None and torch.isfinite(m1["abcd"].grad).all())
    raw6 = torch.zeros(1, 6, dtype=torch.float32, device=device, requires_grad=True)
    prof6 = sfcw_head(cl1, {"raw6": raw6}, sensor).abs().mean(0); prof6.sum().backward()
    check("SFCW gradient flows to the six 3DPS raw parameters", raw6.grad is not None and torch.isfinite(raw6.grad).all() and float(raw6.grad.abs().sum()) > 0)
    cl2 = _cloud([pos0 + bore * 1.5, pos0 + bore * 2.5], [-bore, -bore], device); m2 = _material(2, device)
    prof = sfcw_head(cl2, m2, sensor).abs().mean(0); pk = [int(prof[3:-3].argmax()) + 3]
    prof2 = prof.clone(); prof2[max(pk[0] - 6, 0):pk[0] + 7] = 0; pk.append(int(prof2[3:-3].argmax()) + 3)
    check("SFCW two targets 1.5/2.5 m resolved", all(min(abs(p - round(r / SFCW_BIN_M)) for r in (1.5, 2.5)) <= 1 for p in pk), f"(bins {sorted(pk)}, expect {round(1.5/SFCW_BIN_M)},{round(2.5/SFCW_BIN_M)})")
    gsens = dict(sensor, gain_tx=np.full(20, 2.0), gain_rx=np.ones(20))
    check("SFCW channel gains scale the pair image", abs(float(sfcw_head(cl2, m2, gsens).abs().sum() / sfcw_head(cl2, m2, sensor).abs().sum()) - 2.0) < 1e-6)
    # --- Wi-Fi: scatterer with Tx->p->Rx path 15 m, no LOS -> CIR tap 2 (7.5 m taps); gradient
    txp = np.array([2.0, -2.0, 0.9]); rxp = np.array([-2.0, 2.0, 0.9]); d = np.linalg.norm(rxp - txp)
    # point on the perpendicular bisector such that Rt + Rr = 15 m
    mid = 0.5 * (txp + rxp); perp = np.array([1.0, 1.0, 0.0]) / math.sqrt(2); h = math.sqrt((7.5) ** 2 - (d / 2) ** 2)
    p = mid + perp * h
    clw = _cloud([p], None, device); mw = _material(1, device, grad=True)
    sens_w = dict(tx=txp, rx={k: rxp for k in (1, 2, 3, 4)}, los={k: False for k in (1, 2, 3, 4)})
    csi = wifi_head(clw, mw, sens_w); cir = WF.csi_to_cir(csi[0]).abs()
    tap = int(cir[0].argmax()); check(f"WiFi scatterer path 15 m -> CIR tap {tap} (expect 2)", tap == 2)
    (csi.abs().sum()).backward(); check("WiFi gradient flows", mw["rho"].grad is not None and torch.isfinite(mw["rho"].grad).all())
    sens_l = dict(tx=txp, rx={k: rxp for k in (1, 2, 3, 4)}, los={k: True for k in (1, 2, 3, 4)})
    cl0 = _cloud([p + 100.0], None, device)      # scatterer far away: LOS dominates
    cir = WF.csi_to_cir(wifi_head(cl0, _material(1, device), sens_l)[0]).abs(); tap = int(cir[0].argmax())
    check(f"WiFi LOS term -> first tap (d={d:.2f} m, tap {tap}, expect {round(d / 7.5)})", tap == round(d / 7.5))
    # --- UWB (3DPS rasterizer in UWB mode, measured pulse template): deprecated point sum raises; point target bin;
    #     pulse shape == measured template; two targets; gradients
    try:
        uwb_head_pointsum_deprecated(_cloud([[0, 0, 0]], None, device), _material(1, device), sensor); raised = False
    except RuntimeError:
        raised = True
    check("UWB deprecated point-sum head raises", raised)
    from preprocessing import x4_radar_config as X4
    off = X4.delay_bins()
    clu = _cloud([pos0 + bore * 3.0], [-bore], device); mu = _material(1, device, grad=True)
    rf, env = uwb_head(clu, mu, sensor); k = int(env.argmax())
    check(f"UWB point target 3 m -> bin {k} (expect {round(3.0 / X4.BIN_M + off)}, incl. {off:.0f}-bin on-board delay)", abs(k - round(3.0 / X4.BIN_M + off)) <= 2)
    tpl = uwb_template(device)
    if tpl is not None:
        t_rf, _, half, _ = tpl; seg = rf[k - half:k - half + t_rf.shape[0]].detach().cpu().numpy(); tr = t_rf.cpu().numpy()
        cc = float(np.corrcoef(seg, tr)[0, 1]) if len(seg) == len(tr) else float("nan")
        check(f"UWB rendered point-target pulse == measured template (corr {cc:.3f})", cc > 0.95)
    env.sum().backward(); check("UWB gradient flows to abcd", mu["abcd"].grad is not None and torch.isfinite(mu["abcd"].grad).all())
    raw6 = torch.zeros(1, 6, dtype=torch.float32, device=device, requires_grad=True)
    _, e6 = uwb_head(clu, {"raw6": raw6}, sensor); e6.sum().backward()
    check("UWB gradient flows to the six 3DPS raw parameters", raw6.grad is not None and torch.isfinite(raw6.grad).all() and float(raw6.grad.abs().sum()) > 0)
    clu2 = _cloud([pos0 + bore * 1.0, pos0 + bore * 2.0], [-bore, -bore], device); _, env2 = uwb_head(clu2, _material(2, device), sensor)
    k1 = int(env2[:233].argmax()); k2 = int(env2[233:].argmax()) + 233
    check("UWB two targets 1/2 m resolved", abs(k1 - round(1.0 / X4.BIN_M + off)) <= 2 and abs(k2 - round(2.0 / X4.BIN_M + off)) <= 2, f"(bins {k1},{k2})")
    print("octonet_heads selftest:", "ALL PASS" if ok else "FAILURES")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest"); s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return 0 if selftest(a.device) else 1


if __name__ == "__main__":
    sys.exit(main())
