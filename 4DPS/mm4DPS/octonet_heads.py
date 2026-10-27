"""OctoNet rendering heads for the 4DPS copy: SFCW (Vayyar), Wi-Fi CSI, UWB CIR.

Each head is a free function with ONE shared input convention and returns the measured-domain
quantity the matching target is stored in. No Doppler anywhere (none of the three modalities
records slow-time Doppler); the range kernel is what differs per head:

    SFCW  : stepped-frequency channel -> Hann IFFT over the 100 tones (3.75 cm bins)
    Wi-Fi : 114 subcarriers over 40 MHz, bistatic Tx -> scatterer -> Rx (+ explicit LOS)
    UWB   : Gaussian-modulated 2.5 GHz pulse at delay (Rt+Rr)/c on the X4's 1535-bin grid

Shared input convention (what ``mm4DPS.octonet`` produces from the scaffold + calibration):
    cloud    : dict(xyz (N,3) float64 tensor [MoCap world, m], normals (N,3) float64 tensor or None)
    material : dict(abcd (N,4) float64 tensor = ITU-R P.2040 dispersion coefficients (a,b,c,d), the
               NBS parametrisation; rho (N,) float64 tensor = per-point real reflectivity scale (1 =
               the material's own Fresnel coefficient). Either may require grad.)
    sensor   : SFCW / UWB: dict(pos (3,), boresight (3,)) [world]; both at node_1 (pos = rigid body,
               boresight = calibrated camera yaw, horizontal, an assumption -- see extrinsics.json)
               Wi-Fi: dict(tx (3,), rx {k: (3,)}, los {k: bool}) [world, schematic +-0.5 m]

The SFCW math is single-sourced in nbs/method/sfcw_forward.py (reflectivity, geometry, phase);
``sfcw_head`` only vectorises its per-pair loop and is asserted equal to it in ``selftest``.
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np
import torch

# NBS code (nbs/ is on PYTHONPATH via tools/env_node.sh)
from method import sfcw_forward as SF
from method import wifi_forward as WF
from method.freq_material import C_LIGHT, ITU_GT, reflectivity as itu_reflectivity
from preprocessing import vayyar_radar_config as VR
from preprocessing import octonet_scene as S
from mm3DPS.data import ra_utils as RA

SFCW_FREQS_HZ = SF.sfcw_freqs_hz(VR.F_START_HZ, VR.F_STOP_HZ, VR.N_FREQ)      # (100,)
SFCW_BIN_M = VR.RANGE_RES_M                                                   # 0.0375
WIFI_SC_GHZ = WF.subcarrier_ghz(fc_ghz=S.WIFI["fc_hz"] / 1e9, bw_mhz=S.WIFI["bw_hz"] / 1e6, n_sc=S.WIFI["n_sub"])
UWB_F_LO_HZ, UWB_F_HI_HZ = 7.25e9, 10.2e9            # Novelda X4 band; centre 8.725 GHz, BW ~2.5-2.95 GHz
UWB_FC_HZ = 0.5 * (UWB_F_LO_HZ + UWB_F_HI_HZ)
UWB_BW_HZ = S.UWB["bandwidth_hz"]                     # 2.5e9 (paper)
UWB_BIN_M = S.UWB["bin_m"]                            # 9.9/1535 one-way range per bin
UWB_N_BINS = S.UWB["n_bins"]


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
def sfcw_channel(cloud: dict, material: dict, sensor: dict, chunk: int = 4096) -> torch.Tensor:
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


def sfcw_head(cloud: dict, material: dict, sensor: dict, chunk: int = 4096) -> torch.Tensor:
    """Pair-wise range profiles rp (400, 100) complex128 in the target's layout: Hann-windowed IFFT
    over the tones (``ra_utils.sfcw_range_compress``), pairs flattened tx-major (p = tx*20 + rx,
    ``octonet_config.VAY_PAIR_ORDER``)."""
    H = sfcw_channel(cloud, material, sensor, chunk)
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
    """Range-domain pulse: Gaussian envelope whose round-trip FWHM is c/B (one-way sigma
    c/(2B)/2.355), optionally carrying the RF carrier cos(4 pi fc r / c) (the X4 frame is a real
    sampled RF/baseband waveform; the incoherent envelope is the robust comparison)."""
    sig = (C_LIGHT / UWB_BW_HZ) / 2.0 / 2.355
    env = torch.exp(-0.5 * (r_rel / sig) ** 2)
    if rf:
        env = env * torch.cos(4.0 * math.pi * UWB_FC_HZ * r_rel / C_LIGHT)
    return env


def uwb_head(cloud: dict, material: dict, sensor: dict, chunk: int = 8192, beam_deg: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
    grid = torch.arange(UWB_N_BINS, dtype=torch.float64, device=dev) * UWB_BIN_M             # (B,)
    rf = torch.zeros(UWB_N_BINS, dtype=torch.float64, device=dev); env = torch.zeros_like(rf)
    for lo in range(0, xyz.shape[0], chunk):
        rr = grid[None, :] - R[lo:lo + chunk, None]                                           # (M,B)
        a = amp[lo:lo + chunk, None]
        env = env + (a * uwb_pulse(rr, rf=False)).sum(0)
        rf = rf + (a * uwb_pulse(rr, rf=True)).sum(0)
    return rf, env


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
    # --- SFCW: equivalence with the NBS per-pair loop (50 random points with normals)
    rng = np.random.default_rng(0)
    pts = pos0 + bore * rng.uniform(1.0, 3.5, (50, 1)) + rng.normal(0, 0.5, (50, 3)); nrm = rng.normal(size=(50, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    cl = _cloud(pts, nrm, device); mat = _material(50, device)
    tx, rx = vayyar_arrays(sensor, device)
    H_ref = SF.render_sfcw_pairs(mat["abcd"], cl["xyz"], tx, rx, SFCW_FREQS_HZ, geom_mode="monostatic", normals=cl["normals"])
    H_new = sfcw_channel(cl, mat, sensor, chunk=16)
    rel = float((H_new - H_ref).abs().max() / H_ref.abs().max())
    check("SFCW vectorised == NBS render_sfcw_pairs", rel < 1e-8, f"(max rel diff {rel:.2e})")
    # --- SFCW: single target on boresight at 2.0 m -> bin 53; two targets resolved; gradient
    for R0 in (2.0,):
        cl1 = _cloud([pos0 + bore * R0], None, device); m1 = _material(1, device, grad=True)
        rp = sfcw_head(cl1, m1, sensor); prof = rp.abs().mean(0); k = int(prof[3:-3].argmax()) + 3
        check(f"SFCW point target {R0} m -> bin {k} (expect {round(R0 / SFCW_BIN_M)})", abs(k - round(R0 / SFCW_BIN_M)) <= 1)
        (prof.sum()).backward(); check("SFCW gradient flows to rho and abcd", m1["rho"].grad is not None and torch.isfinite(m1["rho"].grad).all() and m1["abcd"].grad is not None)
    cl2 = _cloud([pos0 + bore * 1.5, pos0 + bore * 2.5], None, device); m2 = _material(2, device)
    prof = sfcw_head(cl2, m2, sensor).abs().mean(0); pk = [int(prof[3:-3].argmax()) + 3]
    prof2 = prof.clone(); prof2[max(pk[0] - 3, 0):pk[0] + 4] = 0; pk.append(int(prof2[3:-3].argmax()) + 3)
    check("SFCW two targets 1.5/2.5 m resolved", sorted(pk) == sorted([round(1.5 / SFCW_BIN_M), round(2.5 / SFCW_BIN_M)]) or all(min(abs(p - round(r / SFCW_BIN_M)) for r in (1.5, 2.5)) <= 1 for p in pk), f"(bins {sorted(pk)}, expect {round(1.5/SFCW_BIN_M)},{round(2.5/SFCW_BIN_M)})")
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
    # --- UWB: target at 3 m -> bin 465 (+-2); two targets; gradient
    clu = _cloud([pos0 + bore * 3.0], None, device); mu = _material(1, device, grad=True)
    rf, env = uwb_head(clu, mu, sensor); k = int(env.argmax()); check(f"UWB point target 3 m -> bin {k} (expect {round(3.0 / UWB_BIN_M)})", abs(k - round(3.0 / UWB_BIN_M)) <= 2)
    env.sum().backward(); check("UWB gradient flows", mu["rho"].grad is not None and torch.isfinite(mu["rho"].grad).all())
    clu2 = _cloud([pos0 + bore * 1.0, pos0 + bore * 2.0], None, device); _, env2 = uwb_head(clu2, _material(2, device), sensor)
    k1 = int(env2[:233].argmax()); k2 = int(env2[233:].argmax()) + 233
    check("UWB two targets 1/2 m resolved", abs(k1 - round(1.0 / UWB_BIN_M)) <= 2 and abs(k2 - round(2.0 / UWB_BIN_M)) <= 2, f"(bins {k1},{k2})")
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
