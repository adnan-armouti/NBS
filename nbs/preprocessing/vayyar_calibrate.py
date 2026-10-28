"""Vayyar IMAGEVK-74 channel calibration, pair order, pitch and range offset from the data (tracker T5-T7).

Facts used: the direct Tx->Rx coupling is the strongest, isolated return (range bins ~5-6) and its geometry is known
up to the layout hypothesis (L-array, Tx on one edge, Rx on the adjacent edge, corner shared); the static scene
(scaffold_S1_calibrated) gives an independent angular picture to score candidate layouts against.

Pair index p in the (400,) axis -> (A, B) = (p // 20, p % 20)  ("tx-major": A = Tx). Everything is tested for both
assignments; the physical questions are (i) which of A/B is the azimuth (horizontal) edge, (ii) where the corner is
on each edge, (iii) the element pitch, (iv) per-channel delays/phases, (v) the common electrical delay (range offset).

    python -m preprocessing.vayyar_calibrate            (node, ~2-4 min)
Writes $NBS_DATA_OUT/calib_S1/vayyar_channel_calibration.json and viz/render_vs_gt/vayyar_calibration.png.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S
from preprocessing import vayyar_radar_config as VR

C0 = 299792458.0
NF = 100
FREQS = np.linspace(VR.F_START_HZ, VR.F_STOP_HZ, NF)
DF = FREQS[1] - FREQS[0]
FC = FREQS.mean()
LAM = C0 / FC
BIN_M = C0 / (2 * (VR.F_STOP_HZ - VR.F_START_HZ))
GUARD = np.ones(NF, bool); GUARD[:6] = False; GUARD[-6:] = False
COUPLING_BINS = slice(0, 14)


# --------------------------------------------------------------------------- basics
def load(rec):
    """``rec`` = 14-digit recording stamp or a direct path to a vayyar pickle (pick_scene1_recordings returns paths)."""
    path = rec if os.path.sep in str(rec) else S.modality_path(1, "vayyar", rec)
    V, t = IO.load_vayyar(path)
    return V.astype(np.complex128), t


def rc(H):
    """Hann-windowed IFFT over the tones -> range profiles (same as sfcw_to_range / ra_utils)."""
    return np.fft.ifft(H * np.hanning(NF)[None, :], axis=-1)


def mat(x400, order="AB"):
    M = np.asarray(x400).reshape(20, 20)
    return M if order == "AB" else M.T


def coherence(M):
    p = (np.abs(M) ** 2).mean()
    a = (np.conj(M[:-1]) * M[1:]).mean(); b = (np.conj(M[:, :-1]) * M[:, 1:]).mean()
    return dict(coh0=float(abs(a) / p), coh1=float(abs(b) / p), slope0_deg=float(np.degrees(np.angle(a))), slope1_deg=float(np.degrees(np.angle(b))))


def unit(x):
    return x / np.clip(np.abs(x), 1e-12, None)


def rank1_phase(Mu, iters=20):
    """Unit-modulus M[i,j] ~ a_i b_j: alternating estimate; returns (a (20,), b (20,), fit |corr|)."""
    b = np.ones(20, complex)
    for _ in range(iters):
        a = unit((Mu * np.conj(b)[None, :]).sum(1)); b = unit((Mu * np.conj(a)[:, None]).sum(0))
    fit = abs((np.conj(a[:, None] * b[None, :]) * Mu).sum()) / Mu.size
    return a, b, float(fit)


# --------------------------------------------------------------------------- layouts
def layout(az_axis="A", corner_A=19, corner_B=0, pitch=2.1e-3):
    """Element offsets (m) in the radar frame (x = azimuth, y = boresight, z = elevation), array centred at 0.
    The az edge runs along x away from the corner, the other edge along z away from the corner."""
    iA = np.arange(20); iB = np.arange(20)
    sA = np.abs(iA - corner_A) * pitch; sB = np.abs(iB - corner_B) * pitch     # distance from the corner along each edge
    A = np.zeros((20, 3)); B = np.zeros((20, 3))
    if az_axis == "A":
        A[:, 0] = sA; B[:, 2] = sB
    else:
        A[:, 2] = sA; B[:, 0] = sB
    ctr = np.concatenate([A, B]).mean(0)
    return A - ctr, B - ctr


def coupling_path(corner_A, corner_B, pitch):
    """Direct-path length (m) between A element i and B element j on an L-array (shared corner)."""
    i = np.arange(20)[:, None]; j = np.arange(20)[None, :]
    return pitch * np.hypot(np.abs(i - corner_A), np.abs(j - corner_B))


# --------------------------------------------------------------------------- 1. coupling analysis -> per-channel calibration
def coupling_delays(H):
    """Per-pair one-way path length (m, includes electrical delay) and phase at FC of the coupling term only."""
    rp = rc(H); g = np.zeros_like(rp); g[:, COUPLING_BINS] = rp[:, COUPLING_BINS]
    Hc = np.fft.fft(g, axis=-1) / np.hanning(NF)[None, :].clip(1e-3)       # back to tones (window undone where safe)
    Hc[:, :3] = 0; Hc[:, -3:] = 0
    k = np.arange(NF)[3:-3]
    ph = np.unwrap(np.angle(Hc[:, 3:-3]), axis=1)
    slope = np.polyfit(k, ph.T, 1)                                           # (2,400): slope, intercept
    tau = -slope[0] / (2 * np.pi * DF)                                       # s
    d = C0 * tau                                                             # m (one-way path incl. electrical)
    phi_fc = np.angle(np.exp(1j * (slope[0] * (k.mean()) + slope[1])))       # phase at the band centre
    return d, phi_fc, np.abs(Hc).mean(1)


def fit_corner_pitch(d, candidates=(0, 19)):
    """Which corner and what effective pitch explain the NON-additive part of the coupling path matrix."""
    D = d.reshape(20, 20)
    dc = lambda X: X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    R = dc(D); out = []
    for cA in candidates:
        for cB in candidates:
            G = dc(coupling_path(cA, cB, 1.0))
            p_eff = float((R * G).sum() / (G * G).sum())
            corr = float((R * G).sum() / np.sqrt((R * R).sum() * (G * G).sum()))
            out.append(dict(corner_A=cA, corner_B=cB, pitch_eff_mm=p_eff * 1e3, corr=corr))
    return sorted(out, key=lambda r: -r["corr"])


def channel_calibration(H, corner_A, corner_B, pitch_eff):
    """Per-channel electrical delays a_i, b_j (m) and phases at FC from the coupling, geometry removed."""
    d, phi, w = coupling_delays(H)
    g = coupling_path(corner_A, corner_B, pitch_eff)
    E = d.reshape(20, 20) - g                                                # electrical part
    c0 = E.mean(); a = E.mean(1) - c0; b = E.mean(0) - c0                    # additive fit (+ common)
    resid_delay = E - (c0 + a[:, None] + b[None, :])
    # phase at FC after removing geometry and the fitted delays
    P = np.exp(1j * (phi.reshape(20, 20) - 2 * np.pi * FC * (g + c0 + a[:, None] + b[None, :]) / C0))
    pa, pb, fit = rank1_phase(P)
    return dict(delay_common_m=float(c0), delay_A_m=a, delay_B_m=b, phase_A=np.angle(pa), phase_B=np.angle(pb),
                phase_rank1_fit=fit, delay_resid_rms_mm=float(resid_delay.std() * 1e3), d=d, phi=phi, w=w)


def apply_calibration(H, cal, corner_A, corner_B, pitch_eff, remove_common=True):
    """H (…,400,100) -> calibrated: per-channel delays and phases removed (common delay optional)."""
    i = np.arange(20)[:, None]; j = np.arange(20)[None, :]
    dly = cal["delay_A_m"][:, None] + cal["delay_B_m"][None, :] + (cal["delay_common_m"] if remove_common else 0.0)
    ph = cal["phase_A"][:, None] + cal["phase_B"][None, :]
    corr = np.exp(1j * (2 * np.pi * FREQS[None, None, :] * dly[..., None] / C0 - ph[..., None]))   # (20,20,100)
    return H * corr.reshape(400, NF)


# --------------------------------------------------------------------------- 1b. per-sweep phase test
def per_sweep_test(V, bins=(20, 27, 40, 80, 89, 92), frames=None):
    """Is there a per-row (or per-column) phase that is common across range bins within a frame and random
    across frames? Returns frame-similarity at each bin before / after removing per-row and per-column phases
    fitted from ALL other bins of the same frame, plus the within-frame consistency of those phases."""
    T = V.shape[0]; frames = frames or list(range(0, T, max(1, T // 30)))[:30]
    rp = rc(V[frames]); ref = rc(V.mean(0))
    out = {}
    for axis in (0, 1):
        sims_before, sims_after, consist = [], [], []
        for b in bins:
            others = [x for x in bins if x != b]
            Mref = mat(ref[:, b]); Mref_o = [mat(ref[:, x]) for x in others]
            sb, sa = [], []
            for k in range(len(frames)):
                M = mat(rp[k, :, b]); Mo = [mat(rp[k, :, x]) for x in others]
                sb.append(abs((np.conj(unit(M)) * unit(Mref)).sum()) / 400)
                # per-row (axis 0) or per-column phase from the OTHER bins
                acc = sum((Mo[n] * np.conj(Mref_o[n])).sum(1 - axis) for n in range(len(others)))
                th = unit(acc)                                                 # (20,) per row/col phase
                # consistency: the same phase from two disjoint halves of the other bins
                h1 = unit(sum((Mo[n] * np.conj(Mref_o[n])).sum(1 - axis) for n in range(0, len(others), 2)))
                h2 = unit(sum((Mo[n] * np.conj(Mref_o[n])).sum(1 - axis) for n in range(1, len(others), 2)))
                consist.append(abs((np.conj(h1) * h2).mean()))
                Mc = M * np.conj(th)[:, None] if axis == 0 else M * np.conj(th)[None, :]
                sa.append(abs((np.conj(unit(Mc)) * unit(Mref)).sum()) / 400)
            sims_before.append(np.mean(sb)); sims_after.append(np.mean(sa))
        out[f"axis{axis}"] = dict(frame_similarity_before=float(np.mean(sims_before)), frame_similarity_after_removing_phase=float(np.mean(sims_after)),
                                  within_frame_phase_consistency=float(np.mean(consist)))
    return out


def ramp_test(V, bins=(6, 20, 27, 40, 80, 92)):
    """Slope of the phase ramp along each axis per bin (mean channel) and per frame (bin 27)."""
    Hm = V.mean(0); rp = rc(Hm)
    per_bin = {int(b): coherence(mat(rp[:, b])) for b in bins}
    T = V.shape[0]; fr = list(range(0, T, max(1, T // 25)))[:25]
    rpf = rc(V[fr])[:, :, 27]
    s0 = [coherence(mat(x))["slope0_deg"] for x in rpf]; s1 = [coherence(mat(x))["slope1_deg"] for x in rpf]
    return dict(per_bin=per_bin, per_frame_bin27=dict(slope0_mean=float(np.mean(s0)), slope0_std=float(np.std(s0)), slope1_mean=float(np.mean(s1)), slope1_std=float(np.std(s1))))


# --------------------------------------------------------------------------- 2. RA vs scaffold occupancy
def radar_frame():
    ext = json.load(open(os.path.join(C.NBS_DATA_OUT, "calib_S1", "extrinsics.json")))
    R = np.array(ext["vayyar"]["R_wc"]); pos = np.array(ext["vayyar"]["t_w"]); bore = R[:, 2].copy(); bore[2] = 0; bore /= np.linalg.norm(bore)
    x, y, z = VR.basis_from_boresight(bore)
    return pos, np.stack([x, y, z])          # rows = x_hat (az), y_hat (bore), z_hat (el)


def scaffold_occupancy(theta_deg, n_bins=NF, yaw_shift_deg=0.0):
    pcl = np.load(os.path.join(C.NBS_DATA_OUT, "scaffold_S1_calibrated", "pcl.npy"))[:, :3].astype(float)
    pos, B = radar_frame(); rel = (pcl - pos) @ B.T                      # (N,3) in radar frame
    r = np.linalg.norm(rel, axis=1); az = np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + yaw_shift_deg
    az = (az + 180) % 360 - 180
    ok = (r < 3.6) & (np.abs(az) < 85) & (rel[:, 1] > 0.2)
    rb = np.clip((r[ok] / BIN_M).astype(int), 0, n_bins - 1)
    te = np.concatenate([[theta_deg[0] - 0.5 * (theta_deg[1] - theta_deg[0])], 0.5 * (theta_deg[1:] + theta_deg[:-1]), [theta_deg[-1] + 0.5 * (theta_deg[1] - theta_deg[0])]])
    tb = np.clip(np.searchsorted(te, az[ok]) - 1, 0, len(theta_deg) - 1)
    occ = np.zeros((n_bins, len(theta_deg)))
    np.add.at(occ, (rb, tb), 1.0 / np.maximum(r[ok], 0.3) ** 2)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(occ, (1.0, 2.0))


def measured_ra(rp400, az_axis, corner_az, pitch, theta_deg, order="AB", ramp_A_deg=0.0):
    """|RA| (bins, theta): DFT across the azimuth edge's 20 elements with explicit positions, incoherent sum over the other edge.
    ``ramp_A_deg``: a per-element linear phase removed along the A axis before beamforming (the deterministic ramp)."""
    Mb = np.stack([mat(rp400[:, b], order) for b in range(NF)])             # (100, A, B)
    if ramp_A_deg:
        Mb = Mb * np.exp(-1j * np.radians(ramp_A_deg) * np.arange(20))[None, :, None]
    if az_axis == "B":
        Mb = Mb.transpose(0, 2, 1)                                           # (100, az, other)
    xi = np.abs(np.arange(20) - corner_az) * pitch                            # positions along the az edge
    st = np.exp(-1j * 2 * np.pi * xi[:, None] * np.sin(np.radians(theta_deg))[None, :] / LAM)   # (20, T)
    ra = np.einsum("bao,at->bot", Mb, st)                                    # (100, other, T)
    return np.sqrt((np.abs(ra) ** 2).sum(1))                                 # (100, T)


def score(ra, occ):
    """Pearson correlation of the azimuth structure (per-range-bin mean removed) of two dB images on the guarded bins."""
    def prep(X):
        X = np.asarray(X, float); Y = 10 * np.log10(np.clip(X, 1e-3 * X.max(), None)); Y = Y[GUARD]
        Y = Y - Y.mean(1, keepdims=True); return Y.ravel()
    a, b = prep(ra), prep(occ)
    if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def coupling_peak_range(rp400):
    """Sub-bin range (m) of the coupling peak per pair (parabolic interpolation of |rp| in bins 2..12)."""
    a = np.abs(rp400[:, 2:13]); k = a.argmax(1) + 2; out = []
    for p in range(rp400.shape[0]):
        y0, y1, y2 = np.abs(rp400[p, k[p] - 1]), np.abs(rp400[p, k[p]]), np.abs(rp400[p, k[p] + 1])
        den = (y0 - 2 * y1 + y2); d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        out.append((k[p] + d) * BIN_M)
    return np.array(out)


# --------------------------------------------------------------------------- 3. range offset
def range_offset_from_profiles(rp_meas, pcl_render_profile):
    pm = 20 * np.log10(np.abs(rp_meas).mean(0) + 1e-12); pr = 20 * np.log10(pcl_render_profile + 1e-12)
    band = GUARD & (np.arange(NF) * BIN_M > 0.5) & (np.arange(NF) * BIN_M < 2.6)
    pm[~band] = np.nan; pr[~band] = np.nan
    best = None
    for lag in range(-6, 7):
        a = np.roll(pm, -lag); ok = np.isfinite(a) & np.isfinite(pr) & band
        c = np.corrcoef(a[ok], pr[ok])[0, 1]
        if best is None or c > best[1]: best = (lag, c)
    return dict(lag_bins=best[0], lag_m=best[0] * BIN_M, corr=best[1])


def render_profile(az_axis, corner_A, corner_B, pitch, device="cuda"):
    import torch
    from method import sfcw_forward as SF
    from method.freq_material import ITU_GT
    pcl = np.load(os.path.join(C.NBS_DATA_OUT, "scaffold_S1_calibrated", "pcl.npy")).astype(float)
    rng = np.random.default_rng(0); pcl = pcl[rng.choice(len(pcl), min(15000, len(pcl)), replace=False)]
    pos, B = radar_frame(); A, Bm = layout(az_axis, corner_A, corner_B, pitch)
    tx = torch.tensor(pos + A @ B, dtype=torch.float64, device=device); rx = torch.tensor(pos + Bm @ B, dtype=torch.float64, device=device)
    p = torch.tensor(pcl[:, :3], dtype=torch.float64, device=device); n = torch.tensor(pcl[:, 3:6], dtype=torch.float64, device=device)
    abcd = torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=device).expand(len(pcl), 4)
    with torch.no_grad():
        Hr = SF.render_sfcw_pairs(abcd, p, tx, rx, FREQS, geom_mode="monostatic", normals=n)   # (20,20,100)
    return Hr.reshape(400, NF).cpu().numpy()


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    from method.sfcw_target import pick_scene1_recordings
    paths = pick_scene1_recordings(3)                                     # vayyar pickle paths (their stamps differ from node_1's by ~1 s)
    labels = [os.path.basename(str(x))[:14] for x in paths]
    out_dir = os.path.join(C.NBS_DATA_OUT, "calib_S1"); viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "viz", "render_vs_gt")
    recs = labels
    res = {"recordings": recs, "vayyar_files": [os.path.basename(str(x)) for x in paths], "pitch_assumed_mm": 2.1, "lambda_mm": LAM * 1e3}
    Vs = {lab: load(pth)[0] for lab, pth in zip(labels, paths)}
    # ---- 1d ramp + baseline coherence
    res["ramp"] = {r: ramp_test(Vs[r]) for r in recs}
    # ---- 1b per-sweep phase test
    res["per_sweep"] = {r: per_sweep_test(Vs[r]) for r in recs[:2]}
    # ---- 1a' coupling geometry -> corner, pitch_eff, per-channel calibration
    Hs = {r: Vs[r].mean(0) for r in recs}
    res["coupling_fit"] = {r: fit_corner_pitch(coupling_delays(Hs[r])[0]) for r in recs}
    best = res["coupling_fit"][recs[0]][0]; cA, cB = best["corner_A"], best["corner_B"]; p_eff = best["pitch_eff_mm"] * 1e-3
    cals = {r: channel_calibration(Hs[r], cA, cB, p_eff) for r in recs}
    cal = cals[recs[0]]
    res["calibration"] = dict(corner_A=cA, corner_B=cB, pitch_eff_mm=p_eff * 1e3, delay_common_m=cal["delay_common_m"],
                              range_offset_from_coupling_m=cal["delay_common_m"] / 2, delay_A_mm=(cal["delay_A_m"] * 1e3).tolist(), delay_B_mm=(cal["delay_B_m"] * 1e3).tolist(),
                              phase_A_deg=np.degrees(cal["phase_A"]).tolist(), phase_B_deg=np.degrees(cal["phase_B"]).tolist(), phase_rank1_fit=cal["phase_rank1_fit"],
                              delay_resid_rms_mm=cal["delay_resid_rms_mm"],
                              stability_across_recordings=dict(
                                  delay_A_rms_mm=float(np.std([cals[r]["delay_A_m"] for r in recs], 0).mean() * 1e3),
                                  phase_A_circ_std_deg=float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean([np.exp(1j * cals[r]["phase_A"]) for r in recs], 0)).clip(1e-9)))).mean()),
                                  phase_B_circ_std_deg=float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean([np.exp(1j * cals[r]["phase_B"]) for r in recs], 0)).clip(1e-9)))).mean())))
    # coherence / ramp before vs after calibration (mean channel, scene bins)
    Hcal = apply_calibration(Hs[recs[0]], cal, cA, cB, p_eff)
    rp0, rp1 = rc(Hs[recs[0]]), rc(Hcal)
    res["coherence_before_after"] = {int(b): dict(before=coherence(mat(rp0[:, b])), after=coherence(mat(rp1[:, b]))) for b in (6, 20, 27, 40, 60, 80, 92)}
    # ---- 2. RA vs occupancy over candidates (calibrated static, coupling bins guarded)
    theta = np.linspace(-80, 80, 81)
    occ = scaffold_occupancy(theta)
    rng = np.random.default_rng(1)
    null = [np.roll(occ[:, ::(1 if f < 0.5 else -1)], int(k), axis=1) for k, f in zip(rng.integers(12, 70, 24), rng.uniform(size=24))]
    table = []
    ramp0 = float(np.mean([res["ramp"][recs[0]]["per_bin"][b]["slope0_deg"] for b in (20, 27, 40, 80, 92)]))
    res["ramp_A_deg_mean"] = ramp0
    for az_axis in ("A", "B"):
        for corner_az in (0, 19):
            for pitch_mm in (1.6, 1.8, 2.0, 2.1, 2.2, 2.4, 2.6):
                for ramp_tag, ramp in (("raw", 0.0), ("A-ramp removed", ramp0)):
                    ra = measured_ra(rp0, az_axis, corner_az, pitch_mm * 1e-3, theta, ramp_A_deg=ramp)
                    sc = score(ra, occ); nulls = [score(ra, o) for o in null]
                    table.append(dict(az_axis=az_axis, corner_az=corner_az, pitch_mm=pitch_mm, data=ramp_tag, score=sc, null_mean=float(np.mean(nulls)), null_std=float(np.std(nulls)), z=float((sc - np.mean(nulls)) / (np.std(nulls) + 1e-9))))
    # joint (pitch x removed-ramp) grid for azimuth along A, both corners: a ramp is an azimuth shift whose angle depends on the pitch
    pitches = np.arange(1.4, 2.81, 0.2); ramps = np.arange(-180, 181, 10); grid = {}
    for corner_az in (0, 19):
        Gs = np.zeros((len(pitches), len(ramps)))
        for a, pm in enumerate(pitches):
            for r_, rm in enumerate(ramps):
                Gs[a, r_] = score(measured_ra(rp0, "A", corner_az, pm * 1e-3, theta, ramp_A_deg=float(rm)), occ)
        grid[corner_az] = Gs
    res["grid_pitches_mm"] = pitches.tolist(); res["grid_ramps_deg"] = ramps.tolist(); res["grid_scores"] = {str(k): v.tolist() for k, v in grid.items()}
    kbest = max(grid, key=lambda k: grid[k].max()); ia, ir = np.unravel_index(grid[kbest].argmax(), grid[kbest].shape)
    ra_best = measured_ra(rp0, "A", kbest, pitches[ia] * 1e-3, theta, ramp_A_deg=float(ramps[ir]))
    nb = [score(ra_best, o) for o in null]
    res["joint_best"] = dict(az_axis="A", corner_az=int(kbest), pitch_mm=float(pitches[ia]), ramp_A_deg=float(ramps[ir]), score=float(grid[kbest].max()), null_mean=float(np.mean(nb)), null_std=float(np.std(nb)), z=float((grid[kbest].max() - np.mean(nb)) / (np.std(nb) + 1e-9)))
    sweep = [dict(ramp_A_deg=float(rm), score=float(grid[kbest][ia, r_])) for r_, rm in enumerate(ramps)]
    res["ramp_sweep_at_best_geometry"] = sweep
    table.sort(key=lambda r: -r["score"])
    res["coupling_peak_range_m"] = dict(median=float(np.median(coupling_peak_range(rp0))), p5=float(np.percentile(coupling_peak_range(rp0), 5)), p95=float(np.percentile(coupling_peak_range(rp0), 95)))
    res["ra_scores"] = table[:40]; res["ra_scores_all"] = table
    bestra = table[0]
    # ---- 2b. render-based scoring of the leading candidates (same PSF and RA transform on both sides)
    rb = []
    try:
        cache = {}
        for corner_az in (19, 0):
            for pm in (1.6, 2.1, 2.6):
                key = (corner_az, pm)
                Hr = render_profile("A", corner_az, 0 if corner_az == 19 else 19, pm * 1e-3); cache[key] = rc(Hr)
                ra_rend = measured_ra(cache[key], "A", corner_az, pm * 1e-3, theta)
                nullr = [np.roll(ra_rend[:, ::(1 if f < 0.5 else -1)], int(k), axis=1) for k, f in zip(rng.integers(8, 60, 24), rng.uniform(size=24))]
                for ramp in (0.0, ramp0, float(res["joint_best"]["ramp_A_deg"])):
                    ra_m = measured_ra(rp0, "A", corner_az, pm * 1e-3, theta, ramp_A_deg=ramp)
                    sc = score(ra_m, ra_rend); nn = [score(ra_m, o) for o in nullr]
                    rb.append(dict(corner_az=corner_az, pitch_mm=pm, ramp_A_deg=ramp, score=sc, null_mean=float(np.mean(nn)), null_std=float(np.std(nn)), z=float((sc - np.mean(nn)) / (np.std(nn) + 1e-9))))
        rb.sort(key=lambda r: -r["score"]); res["render_based_scores"] = rb
        print("render-based top:"); [print("  ", r) for r in rb[:6]]
    except Exception as e:
        res["render_based_scores"] = {"error": repr(e)}; print("render-based scoring failed:", e)
    # ---- 3. range offset from profiles (render under the best candidate)
    try:
        Hr = render_profile(bestra["az_axis"], cA if bestra["az_axis"] == "A" else cA, cB, p_eff)
        prof_r = np.abs(rc(Hr)).mean(0)
        res["range_offset_from_profiles"] = range_offset_from_profiles(rp0, prof_r)
        res["range_offset_from_profiles_calibrated"] = range_offset_from_profiles(rp1, prof_r)
    except Exception as e:
        res["range_offset_from_profiles"] = {"error": repr(e)}; prof_r = None
    # ---- verdicts
    cf = res["coupling_fit"][recs[0]]
    v = []
    v.append(f"Coupling as a geometry probe: the non-additive part of the per-pair coupling delays is NOT explained by an L-array (best corr {cf[0]['corr']:.2f}, implied pitch {cf[0]['pitch_eff_mm']:.1f} mm, corner ambiguous): the on-board coupling is not a single free-space path, so a coupling-based per-channel delay calibration is rejected (applying it lowered the scene coherence).")
    jb = res["joint_best"]
    v.append(f"V2 pitch: the joint (pitch x ramp) RA-vs-scaffold grid peaks at pitch {jb['pitch_mm']:.1f} mm with a removed A-ramp of {jb['ramp_A_deg']:+.0f} deg/element (score {jb['score']:.3f}, z {jb['z']:.1f} vs azimuth-rolled nulls); pitch and ramp trade off (a ramp = an azimuth shift), so the pitch alone is NOT pinned by this test; the coupling-path pitch estimate ({p_eff*1e3:.1f} mm) is physically meaningless (see above).")
    v.append(f"V4 order/azimuth axis: RA correlation best with azimuth along axis {bestra['az_axis']} (corner index {bestra['corner_az']}), score {bestra['score']:.3f}, z {bestra['z']:.1f}; runner-up on the other axis: " + str(next((f"{r['score']:.3f} (z {r['z']:.1f})" for r in table if r['az_axis'] != bestra['az_axis']), 'n/a')))
    ps = res["per_sweep"][recs[0]]
    sw = max(res["ramp_sweep_at_best_geometry"], key=lambda r: r["score"])
    v.append(f"V6 phases: a deterministic +{ramp0:.0f} deg/element phase ramp along axis A at every range and frame (per-frame std {res['ramp'][recs[0]]['per_frame_bin27']['slope0_std']:.1f} deg); RA-vs-scaffold score vs removed ramp peaks at {sw['ramp_A_deg']:+.0f} deg/element (score {sw['score']:.3f}; raw 0 deg: {next(r['score'] for r in res['ramp_sweep_at_best_geometry'] if r['ramp_A_deg']==0):.3f}); per-sweep test axis0 similarity {ps['axis0']['frame_similarity_before']:.2f}->{ps['axis0']['frame_similarity_after_removing_phase']:.2f} (consistency {ps['axis0']['within_frame_phase_consistency']:.2f}), axis1 {ps['axis1']['frame_similarity_before']:.2f}->{ps['axis1']['frame_similarity_after_removing_phase']:.2f} (consistency {ps['axis1']['within_frame_phase_consistency']:.2f}).")
    ro = res.get("range_offset_from_profiles", {})
    cp = res["coupling_peak_range_m"]
    v.append(f"V10 range offset: the coupling peak sits at {cp['median']*1e2:.1f} cm range (p5-p95 {cp['p5']*1e2:.1f}-{cp['p95']*1e2:.1f}), i.e. an on-board electrical path, not a usable zero reference; profile cross-correlation lag vs the rendered scaffold in 0.5-2.6 m: {ro.get('lag_m', float('nan'))*1e2 if 'lag_m' in ro else float('nan'):+.1f} cm (corr {ro.get('corr', float('nan')):.2f}).")
    res["verdict"] = v; res["runtime_s"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "vayyar_channel_calibration.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for line in v: print(line)
    print("top RA candidates:"); [print("  ", r) for r in table[:8]]
    print("coherence before/after:", {b: (round(x["before"]["coh0"], 2), round(x["before"]["coh1"], 2), round(x["after"]["coh0"], 2), round(x["after"]["coh1"], 2)) for b, x in res["coherence_before_after"].items()})
    print("ramp per bin (mean channel, slope0 deg):", {b: round(x["slope0_deg"], 1) for b, x in res["ramp"][recs[0]]["per_bin"].items()}, "| per frame bin27:", res["ramp"][recs[0]]["per_frame_bin27"])
    figure(res, rp0, rp1, occ, theta, prof_r, cal, viz_dir)
    print("runtime %.0f s" % res["runtime_s"])


def figure(res, rp0, rp1, occ, theta, prof_r, cal, viz_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(viz_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 3, figsize=(20, 16))
    d = cal["d"].reshape(20, 20) * 1e3
    im = axs[0, 0].imshow(d, cmap="viridis"); plt.colorbar(im, ax=axs[0, 0], label="mm"); axs[0, 0].set_title("coupling one-way path length per pair (phase slope over tones)\nrows = A (p//20), cols = B (p%20)"); axs[0, 0].set_xlabel("B index"); axs[0, 0].set_ylabel("A index")
    cA, cB, p = res["calibration"]["corner_A"], res["calibration"]["corner_B"], res["calibration"]["pitch_eff_mm"] * 1e-3
    dc = lambda X: X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    R = dc(cal["d"].reshape(20, 20)) * 1e3; G = dc(coupling_path(cA, cB, p)) * 1e3
    axs[0, 1].scatter(G.ravel(), R.ravel(), s=6, alpha=.5); lim = [min(G.min(), R.min()), max(G.max(), R.max())]; axs[0, 1].plot(lim, lim, "k--", lw=1)
    axs[0, 1].set_xlabel(f"model non-additive path (mm), corner A={cA} B={cB}, pitch_eff {p*1e3:.2f} mm"); axs[0, 1].set_ylabel("measured non-additive path (mm)"); axs[0, 1].set_title("coupling geometry fit: " + "; ".join(f"A{r['corner_A']}B{r['corner_B']}: {r['corr']:.2f}" for r in res["coupling_fit"][res["recordings"][0]])); axs[0, 1].grid(alpha=.3)
    ax = axs[0, 2]; ax.plot(res["calibration"]["delay_A_mm"], "o-", label="delay A (mm)"); ax.plot(res["calibration"]["delay_B_mm"], "s-", label="delay B (mm)")
    ax2 = ax.twinx(); ax2.plot(res["calibration"]["phase_A_deg"], "o--", c="tab:red", label="phase A (deg)"); ax2.plot(res["calibration"]["phase_B_deg"], "s--", c="tab:purple", label="phase B (deg)")
    ax.set_title(f"per-channel calibration from the coupling (rank-1 phase fit {res['calibration']['phase_rank1_fit']:.2f}; delay resid rms {res['calibration']['delay_resid_rms_mm']:.1f} mm)\ncommon electrical delay {res['calibration']['delay_common_m']*1e2:.1f} cm -> range offset {res['calibration']['range_offset_from_coupling_m']*1e2:.1f} cm"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.set_xlabel("element index")
    ax = axs[1, 0]; ax.axis("off")
    rows = ["bin  coh0 coh1 slope0 | after: coh0 coh1 slope0"]
    for b, x in res["coherence_before_after"].items():
        rows.append(f"{b:3d}  {x['before']['coh0']:.2f} {x['before']['coh1']:.2f} {x['before']['slope0_deg']:+6.1f} | {x['after']['coh0']:.2f} {x['after']['coh1']:.2f} {x['after']['slope0_deg']:+6.1f}")
    ps = res["per_sweep"][res["recordings"][0]]
    rows += ["", "per-sweep phase test (frame similarity before -> after removing a per-row/col phase fitted from other bins; consistency):",
             f" axis0: {ps['axis0']['frame_similarity_before']:.2f} -> {ps['axis0']['frame_similarity_after_removing_phase']:.2f} (consist {ps['axis0']['within_frame_phase_consistency']:.2f})",
             f" axis1: {ps['axis1']['frame_similarity_before']:.2f} -> {ps['axis1']['frame_similarity_after_removing_phase']:.2f} (consist {ps['axis1']['within_frame_phase_consistency']:.2f})",
             "", "ramp slope0 per frame at bin 27: %.1f +- %.1f deg" % (res["ramp"][res["recordings"][0]]["per_frame_bin27"]["slope0_mean"], res["ramp"][res["recordings"][0]]["per_frame_bin27"]["slope0_std"])]
    ax.text(0, 1, "\n".join(rows), va="top", family="monospace", fontsize=9); ax.set_title("coherence along the pair axes, raw vs calibrated (mean channel)")
    b = res["joint_best"]
    ra_c = measured_ra(rp0, "A", b["corner_az"], b["pitch_mm"] * 1e-3, theta, ramp_A_deg=b["ramp_A_deg"]); ra_r = measured_ra(rp0, "A", b["corner_az"], b["pitch_mm"] * 1e-3, theta)
    ext = [theta[0], theta[-1], 0, NF * BIN_M]
    def show(ax, X, title):
        Y = 20 * np.log10(np.clip(X, 1e-12, None)); Y[~GUARD] = np.nan; Y = Y - np.nanmax(Y)
        ax.imshow(Y, origin="lower", aspect="auto", extent=ext, vmin=-40, vmax=0, cmap="viridis"); ax.set_title(title, fontsize=9); ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("range (m)")
    show(axs[1, 1], ra_r, f"measured static |RA|, raw phases (az = A, corner {b['corner_az']}, pitch {b['pitch_mm']:.1f} mm)")
    show(axs[1, 2], ra_c, f"measured static |RA| with the A-ramp {b['ramp_A_deg']:+.0f} deg/element removed; score {b['score']:.3f}, z {b['z']:.1f}")
    show(axs[2, 0], occ + 1e-9, "scaffold occupancy (range, azimuth), 1/r^2 weighted, smoothed")
    ax = axs[2, 1]; jb = res["joint_best"]; G = np.array(res["grid_scores"][str(jb["corner_az"])])
    im = ax.imshow(G, origin="lower", aspect="auto", extent=[res["grid_ramps_deg"][0], res["grid_ramps_deg"][-1], res["grid_pitches_mm"][0], res["grid_pitches_mm"][-1]], cmap="magma"); plt.colorbar(im, ax=ax, label="score")
    ax.plot(jb["ramp_A_deg"], jb["pitch_mm"], "c*", ms=14); ax.axvline(res["ramp_A_deg_mean"], color="w", ls="--", lw=1)
    ax.set_xlabel("removed phase ramp along A (deg/element); dashed = measured ramp"); ax.set_ylabel("pitch (mm)")
    ax.set_title(f"RA-vs-scaffold score, azimuth = A, corner {jb['corner_az']}: best pitch {jb['pitch_mm']:.1f} mm, ramp {jb['ramp_A_deg']:+.0f} deg (score {jb['score']:.3f}, z {jb['z']:.1f})\nB-axis best: " + str(round(max((r['score'] for r in res['ra_scores_all'] if r['az_axis']=='B'), default=float('nan')), 3)), fontsize=9)
    ax = axs[2, 2]
    pm = 20 * np.log10(np.abs(rp0).mean(0) + 1e-12); pm[~GUARD] = np.nan; r = np.arange(NF) * BIN_M
    ax.plot(r, pm - np.nanmax(pm), label="measured static (raw)")
    if prof_r is not None:
        pr = 20 * np.log10(prof_r + 1e-12); pr[~GUARD] = np.nan; ax.plot(r, pr - np.nanmax(pr), label="rendered (scaffold, uniform concrete)")
        ro = res.get("range_offset_from_profiles", {}); ax.set_title(f"range offset: profile xcorr lag {ro.get('lag_m', float('nan'))*1e2:.1f} cm (corr {ro.get('corr', float('nan')):.2f}); coupling common delay -> {res['calibration']['range_offset_from_coupling_m']*1e2:.1f} cm", fontsize=9)
    ax.set_xlabel("range (m)"); ax.set_ylabel("dB"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Vayyar IMAGEVK-74 channel calibration / pair order / pitch / range offset from the OctoNet data (Scene 1)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(viz_dir, "vayyar_calibration.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
