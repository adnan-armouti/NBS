"""GT (measured) vs rendered for the three OctoNet RF modalities: SFCW (Vayyar), UWB, Wi-Fi CSI.

Run from nbs/:  python -m viz.render_vs_gt   ->  ../viz/render_vs_gt/*.png + captions.txt

Caveats (also written to captions.txt):
* SFCW: the moving person is NOT observable in the Vayyar data (99.8 % static, dynamic -24 dB). The
  "GT static" panels are the range profile / RA of the TEMPORAL-MEAN channel (bins 0-2 = Tx->Rx
  coupling, guarded); the "residual" panel is what is left after per-pair temporal-mean subtraction
  (drift + noise, not a human return).
* Rendered panels use ASSUMED extrinsics (sensor at the Scene-1 node_1 rigid body, boresight toward the
  MoCap origin), a UNIFORM material (ITU concrete), normal incidence, and a scaffold back-projected
  from node_1's 8-bit depth (nominal D455 intrinsics) unless the geometry group's scaffold exists.
* UWB rendered = PROTOTYPE pulse-sum, not a 4DPS head. Wi-Fi Tx/Rx positions are schematic (+-0.5 m).
"""
import os, sys, pickle, math, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import octonet_scene as S, octonet_io as IO, octonet_config as C
from preprocessing import vayyar_radar_config as VR
POSE = S.pose_source()
POSE_LABEL = {"assumed": "assumed extrinsics", "calibrated": "calibrated extrinsics (octonet_calibrate)"}[POSE]
from method import sfcw_forward as SF, wifi_forward as WF
from method.freq_material import ITU_GT, C_LIGHT
from mm3DPS.data import ra_utils as RA

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "viz", "render_vs_gt")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REC = S.REC_SIT
SUBJECT_R = float(np.linalg.norm(S.NODE_POS_S1[1][:2]))          # 2.92 m node_1 -> subject spot
DB_FLOOR = -50.0
CAP = []

def db(x, floor=DB_FLOOR):
    x = np.abs(np.asarray(x)); x = 20 * np.log10(x / (x.max() + 1e-30) + 1e-30)
    return np.maximum(x, floor)

# ----------------------------------------------------------------------------- scaffold
def load_or_build_scaffold(n_max=20000):
    p = os.path.join(os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT), f"scaffold_S1_{POSE}", "pcl.npy")
    if os.path.exists(p):
        pcl = np.load(p)[:, :3]; src = f"geometry group scaffold {p}"
    else:
        d = S.depth_clip_dir(1, S.REC_WALK); fr = S.depth_frames(d)
        sel = fr[:: max(1, len(fr) // 20)][:20]
        z = np.median(np.stack([S.load_depth_m(f) for f in sel]), axis=0)
        pcl = S.cam_to_world(S.backproject(z, stride=3), 1)
        src = f"node_1 depth temporal median of {len(sel)} frames of REC_WALK ({os.path.basename(d)}), nominal D455 intrinsics, " + POSE_LABEL
    r_xy = np.linalg.norm(pcl[:, :2], axis=1)
    pcl = pcl[(r_xy > 0.7) & (pcl[:, 2] > -0.3) & (pcl[:, 2] < 3.0)]          # drop the subject column + outliers
    rng = np.random.default_rng(0)
    if len(pcl) > n_max:
        pcl = pcl[rng.choice(len(pcl), n_max, replace=False)]
    CAP.append(f"Scaffold: {src}; {len(pcl)} points after dropping r_xy<0.7 m (subject column) and z outside [-0.3,3] m.")
    return pcl

# ----------------------------------------------------------------------------- SFCW
def sfcw():
    d, t = IO.load_vayyar(S.modality_path(1, "vayyar", REC))                  # (T,400,100) complex64
    T = d.shape[0]
    H = torch.from_numpy(d).to(torch.complex64)
    freqs = SF.sfcw_freqs_hz(VR.F_START_HZ, VR.F_STOP_HZ, VR.N_FREQ)
    r_axis = SF.range_axis_m(VR.N_FREQ, VR.BANDWIDTH_HZ)
    guard = np.ones(VR.N_FREQ, bool); guard[:6] = False; guard[-6:] = False   # coupling (+Hann sidelobes) + IFFT wrap
    # GT static = temporal mean channel
    Hs = H.mean(0)                                                             # (400,100)
    rp_s = RA.sfcw_range_compress(Hs)                                          # (400,100)
    prof_s = rp_s.abs().mean(0).numpy(); prof_s[~guard] = np.nan
    # GT residual after per-pair temporal-mean subtraction
    rp_res = RA.sfcw_range_compress(H - Hs[None])                              # (T,400,100)
    res_map = rp_res.abs().mean(1).numpy(); res_map[:, ~guard] = np.nan       # (T,100)
    g = torch.from_numpy(guard)
    res_level = float(10 * np.log10((rp_res[..., g].abs() ** 2).mean().item() / (rp_s[..., g].abs() ** 2).mean().item()))   # dynamic/static energy, guarded band
    # GT RA (static)
    rp_s_txrx = torch.from_numpy(IO.pairs_to_tx_rx(rp_s.numpy()))             # (20,20,100)
    ra_gt = RA.vayyar_to_ra_complex(rp_s_txrx, n_az=128).abs().numpy(); ra_gt[:, ~guard] = np.nan
    # rendered
    pcl = load_or_build_scaffold()
    R1, radar, _ = S.camera_pose(1); bore = R1[:, 2].copy(); bore[2] = 0.0; bore /= np.linalg.norm(bore)   # radar shares node_1 camera yaw (assumption), horizontal
    cfg = VR.build_config(radar_pos=radar, boresight=bore)
    tx = torch.tensor([a["pos_mm"] for a in cfg["tx_array"]], dtype=torch.float64, device=DEV) / 1000
    rx = torch.tensor([a["pos_mm"] for a in cfg["rx_array"]], dtype=torch.float64, device=DEV) / 1000
    pos = torch.tensor(pcl, dtype=torch.float64, device=DEV)
    abcd = torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=DEV).expand(len(pcl), 4)
    with torch.no_grad():
        Hr = SF.render_sfcw_pairs(abcd, pos, tx, rx, freqs, geom_mode="monostatic")   # (20,20,100)
        rp_r = SF.sfcw_to_range(Hr).cpu()
    prof_r = rp_r.abs().reshape(-1, VR.N_FREQ).mean(0).numpy(); prof_r[~guard] = np.nan
    ra_r = RA.vayyar_to_ra_complex(rp_r, n_az=128).abs().numpy(); ra_r[:, ~guard] = np.nan
    # azimuth axis of the Tx ULA
    u = np.fft.fftshift(np.fft.fftfreq(128)); sin_t = np.clip(u * VR.LAMBDA_M / VR.TX_PITCH_M, -1, 1)
    az = np.rad2deg(np.arcsin(sin_t))
    # scaffold in radar frame for the inset
    x_hat, y_hat, _ = VR.basis_from_boresight(bore); rel = pcl - radar
    sx, sy = rel @ x_hat, rel @ y_hat

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    a = ax[0, 0]; a.plot(r_axis, db(np.nan_to_num(prof_s, nan=0)), lw=1.4)
    a.axvline(SUBJECT_R, color="r", ls="--", lw=1, label=f"subject spot {SUBJECT_R:.2f} m"); a.set_ylim(DB_FLOOR, 1)
    a.set_xlabel("range (m), bin 3.75 cm"); a.set_ylabel("|range profile| (dB rel. max)")
    a.set_title("GT static: temporal-mean channel, mean over 400 pairs\n(bins 0-5 & last 6 guarded = Tx->Rx coupling / IFFT wrap)"); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]; im = a.imshow(db(np.nan_to_num(res_map, nan=0)), aspect="auto", origin="lower", cmap="magma",
                              extent=[r_axis[0], r_axis[-1], 0, T], vmin=DB_FLOOR, vmax=0)
    a.axvline(SUBJECT_R, color="c", ls="--", lw=1); a.set_xlabel("range (m)"); a.set_ylabel("frame (3.1 Hz)")
    a.set_title(f"GT residual after per-pair temporal-mean subtraction\n(drift/noise: energy {res_level:.1f} dB vs static; NOT a human return)"); plt.colorbar(im, ax=a, label="dB rel. max")
    a = ax[0, 2]; im = a.imshow(db(np.nan_to_num(ra_gt, nan=0)).T, aspect="auto", origin="lower", cmap="viridis",
                              extent=[az[0], az[-1], r_axis[0], r_axis[-1]], vmin=DB_FLOOR, vmax=0)
    a.axhline(SUBJECT_R, color="r", ls="--", lw=1); a.set_xlabel("azimuth (deg, Tx ULA, uniform-pitch assumption)"); a.set_ylabel("range (m)")
    a.set_title("GT static RA |.| (elevation marginalized)"); plt.colorbar(im, ax=a, label="dB")
    a = ax[1, 0]; a.plot(r_axis, db(np.nan_to_num(prof_r, nan=0)), lw=1.4, color="C1")
    a.axvline(SUBJECT_R, color="r", ls="--", lw=1); a.set_ylim(DB_FLOOR, 1); a.set_xlabel("range (m)"); a.set_ylabel("|range profile| (dB rel. max)")
    a.set_title(f"rendered ({POSE_LABEL}, uniform material)\nsfcw_forward.render_sfcw_pairs on the static scaffold, mean over 400 pairs"); a.grid(alpha=.3)
    a = ax[1, 1]; a.scatter(sx, sy, s=1, c=pcl[:, 2], cmap="coolwarm"); a.scatter([0], [0], marker="^", c="k", s=80, label="Vayyar (node_1)")
    a.scatter([0], [SUBJECT_R], marker="x", c="r", s=60, label="subject spot"); a.set_aspect("equal"); a.set_xlabel("azimuth x (m, radar frame)"); a.set_ylabel("boresight y (m)")
    a.set_title(f"scaffold rendered ({len(pcl)} pts, colour = height)"); a.legend(loc="lower right"); a.grid(alpha=.3)
    a = ax[1, 2]; im = a.imshow(db(np.nan_to_num(ra_r, nan=0)).T, aspect="auto", origin="lower", cmap="viridis",
                              extent=[az[0], az[-1], r_axis[0], r_axis[-1]], vmin=DB_FLOOR, vmax=0)
    a.axhline(SUBJECT_R, color="r", ls="--", lw=1); a.set_xlabel("azimuth (deg)"); a.set_ylabel("range (m)")
    a.set_title(f"rendered RA |.| ({POSE_LABEL}, uniform material)"); plt.colorbar(im, ax=a, label="dB")
    fig.suptitle(f"SFCW (Vayyar IMAGEVK-74, 63-67 GHz, 20x20) — recording {REC} (subject 2, sit) — static scene only", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sfcw_gt_vs_rendered.png"), dpi=160); plt.close(fig)
    pk_gt = r_axis[np.nanargmax(prof_s)]; pk_r = r_axis[np.nanargmax(prof_r)]
    CAP.append(f"sfcw_gt_vs_rendered.png: {T} frames. GT static profile peak at {pk_gt:.2f} m; rendered peak at {pk_r:.2f} m; clutter-removed residual energy {res_level:.1f} dB relative to the temporal-mean return in the guarded band (i.e. beyond the coupling the static return is near the per-frame noise floor; the -24 dB of WRITE_UP #19 is relative to the coupling-dominated total). SFCW shows the STATIC scene only — the person is not recoverable (WRITE_UP #19). Pair order tx-major and uniform 2.1 mm Tx pitch are ASSUMPTIONS (affect azimuth only).")
    return dict(pk_gt=pk_gt, pk_r=pk_r, res_level=res_level)

# ----------------------------------------------------------------------------- UWB
def uwb(pcl):
    x = pickle.load(open(S.modality_path(1, "uwb", REC), "rb"))
    frames = np.stack([np.asarray(e["frame"], float) for e in x]) if isinstance(x, list) else np.asarray(x["frame"], float)[None]
    nb = frames.shape[1]; r = np.arange(nb) * S.UWB["max_range_m"] / S.UWB["n_bins"]
    mag = np.abs(frames)
    # prototype pulse-sum render (monostatic at node_1), incoherent envelope
    R = np.linalg.norm(pcl - S.NODE_POS_S1[1], axis=1)
    sig_r = (C_LIGHT / S.UWB["bandwidth_hz"]) / 2.355 / 2          # FWHM = c/B round trip -> one-way range FWHM c/(2B)=6 cm
    amp = 1.0 / R ** 4
    cir = np.zeros(nb)
    for lo in range(0, len(R), 4000):
        cir += (amp[lo:lo + 4000, None] * np.exp(-0.5 * ((r[None, :] - R[lo:lo + 4000, None]) / sig_r) ** 2)).sum(0)
    fig, ax = plt.subplots(1, 3 if len(frames) > 1 else 2, figsize=(16 if len(frames) > 1 else 12, 4.5))
    a = ax[0]; a.plot(r, db(mag[len(mag) // 2]), lw=.8, label="one frame"); 
    if len(frames) > 1: a.plot(r, db(mag.mean(0)), lw=1.2, label="temporal mean")
    a.axvline(SUBJECT_R, color="r", ls="--", lw=1, label=f"subject spot {SUBJECT_R:.2f} m"); a.set_ylim(DB_FLOOR, 1)
    a.set_xlabel(f"range (m), bin {S.UWB['bin_m']*100:.2f} cm, {nb} bins"); a.set_ylabel("|CIR| (dB rel. max)")
    a.set_title(f"GT UWB CIR (X4M200, node_1) — {len(frames)} frame(s) in this pickle"); a.legend(); a.grid(alpha=.3)
    k = 1
    if len(frames) > 1:
        a = ax[1]; im = a.imshow(db(mag), aspect="auto", origin="lower", extent=[r[0], r[-1], 0, len(mag)], cmap="magma", vmin=DB_FLOOR, vmax=0)
        a.set_xlabel("range (m)"); a.set_ylabel("frame"); a.set_title("GT range-time"); plt.colorbar(im, ax=a); k = 2
    a = ax[k]; a.plot(r, db(cir), lw=1, color="C1"); a.axvline(SUBJECT_R, color="r", ls="--", lw=1); a.set_ylim(DB_FLOOR, 1)
    a.set_xlabel("range (m)"); a.set_ylabel("|CIR| (dB rel. max)")
    a.set_title("rendered: PROTOTYPE incoherent pulse-sum on the static scaffold\n(Gaussian 2.5 GHz pulse, 1/R^4) — NOT a 4DPS head (pending)", fontsize=10); a.grid(alpha=.3)
    fig.suptitle(f"UWB — recording {REC} — radar pose = node_1 rigid body + node_1 camera yaw ({POSE})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "uwb_gt_vs_rendered.png"), dpi=160); plt.close(fig)
    pk = r[np.argmax(mag.mean(0)[20:]) + 20]
    CAP.append(f"uwb_gt_vs_rendered.png: pickle holds {len(frames)} frame(s) x {nb} bins (paper Tab. 2: 1x1535, 17 Hz — the per-recording pickle apparently stores one frame). GT peak (beyond bin 20) at {pk:.2f} m. Rendered is a PROTOTYPE incoherent pulse-sum, not a 4DPS head; UWB radar position/orientation assumed = node_1.")

# ----------------------------------------------------------------------------- WiFi
def wifi(pcl):
    sc = WF.subcarrier_ghz(fc_ghz=S.WIFI["fc_hz"] / 1e9, bw_mhz=S.WIFI["bw_hz"] / 1e6, n_sc=S.WIFI["n_sub"])
    lam = C_LIGHT / S.WIFI["fc_hz"]
    tx = torch.tensor(np.asarray([S.WIFI_TX_APPROX]), dtype=torch.float64, device=DEV)
    pos = torch.tensor(pcl, dtype=torch.float64, device=DEV)
    amp = torch.full((len(pcl),), 0.3, dtype=torch.complex128, device=DEV)      # uniform reflectivity
    path_m = np.arange(S.WIFI["n_sub"]) * C_LIGHT / S.WIFI["bw_hz"]              # CIR tap -> path length (7.5 m)
    fig, ax = plt.subplots(4, 4, figsize=(19, 14))
    for k in (1, 2, 3, 4):
        y = pickle.load(open(S.modality_path(k, "wifi", REC), "rb")); csi = np.asarray(y["data"])   # (F,2,114)
        F = csi.shape[0]; one = csi[F // 2]; mean_mag = np.abs(csi).mean(0)
        cir_gt = WF.csi_to_cir(torch.from_numpy(one)).abs().numpy()                                  # (2,114)
        # rendered: scatter + explicit LOS (omitted for Rx-4, NLOS behind the obstacle)
        rxc = S.WIFI_RX_APPROX[k]; rx = torch.tensor([rxc + np.array([-lam / 4, 0, 0]), rxc + np.array([lam / 4, 0, 0])], dtype=torch.float64, device=DEV)
        with torch.no_grad():
            Hr = WF.render_csi(amp, pos, tx, rx, sc)                                                   # (2,114)
            if k != 4:
                f = torch.tensor(sc, dtype=torch.float64, device=DEV) * 1e9
                for a_i in range(2):
                    Rd = (rx[a_i] - tx[0]).norm(); Hr[a_i] += (1.0 / Rd) * torch.exp(-1j * 2 * math.pi * f * Rd / C_LIGHT)
        Hr = Hr.cpu(); cir_r = WF.csi_to_cir(Hr).abs().numpy(); Hr = Hr.abs().numpy()
        los = "LOS" if k != 4 else "NLOS (obstacle)"
        a = ax[0, k - 1]
        for a_i in range(2): a.plot(db(np.abs(one[a_i]), -40), lw=.8, label=f"ant {a_i} (frame {F//2})"); a.plot(db(mean_mag[a_i], -40), lw=1.2, ls="--", label=f"ant {a_i} (mean of {F})")
        a.set_ylim(-40, 1); a.set_xlabel("subcarrier (114 over 40 MHz, ch 36; notches = null/pilot subcarriers)"); a.set_ylabel("|CSI| dB rel. max"); a.set_title(f"GT Rx-{k} ({los}) |CSI|"); a.grid(alpha=.3)
        if k == 1: a.legend(fontsize=7)
        a = ax[1, k - 1]
        for a_i in range(2): a.plot(path_m[:24], db(cir_gt[a_i][:24], -40), marker=".", lw=.8)
        a.set_ylim(-40, 1); a.set_xlabel("path length (m), tap = 7.5 m"); a.set_ylabel("|CIR| dB"); a.set_title(f"GT Rx-{k} CIR (csi_to_cir, one frame)"); a.grid(alpha=.3)
        a = ax[2, k - 1]
        for a_i in range(2): a.plot(db(Hr[a_i], -40), lw=1)
        a.set_ylim(-40, 1); a.set_xlabel("subcarrier"); a.set_ylabel("|CSI| dB"); a.set_title(f"rendered Rx-{k} |CSI| (scaffold{' + LOS' if k != 4 else ', no LOS'})"); a.grid(alpha=.3)
        a = ax[3, k - 1]
        for a_i in range(2): a.plot(path_m[:24], db(cir_r[a_i][:24], -40), marker=".", lw=.8)
        a.set_ylim(-40, 1); a.set_xlabel("path length (m)"); a.set_ylabel("|CIR| dB"); a.set_title(f"rendered Rx-{k} CIR"); a.grid(alpha=.3)
        CAP.append(f"wifi Rx-{k}: {F} CSI frames (2 ant x 114 sc), GT CIR first-tap/second-tap ratio {db(cir_gt[0][:3],-60)[1]:.1f} dB; Tx-Rx distance (schematic) {np.linalg.norm(rxc - S.WIFI_TX_APPROX):.2f} m, {los}.")
    fig.suptitle(f"Wi-Fi CSI — recording {REC} — rendered with wifi_forward.render_csi, schematic Tx/Rx positions (+-0.5 m), uniform reflectivity, Rx-4 NLOS", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wifi_gt_vs_rendered.png"), dpi=150); plt.close(fig)

if __name__ == "__main__":
    info = sfcw()
    pcl = load_or_build_scaffold()
    uwb(pcl); wifi(pcl)
    CAP.insert(0, f"Recording {REC} (Scene 1, subject 2, 'sit'); frame = MoCap world (Z up, origin = subject spot); node_1 rigid body at {np.round(S.NODE_POS_S1[1],3).tolist()} m, {SUBJECT_R:.2f} m from the subject spot.")
    CAP.append("Wi-Fi GT notches at fixed subcarriers are the HT40 null/pilot subcarriers (not propagation); the rendered CSI has no such notches. The GT CIR tap spacing is 7.5 m (40 MHz), far coarser than the room, so only the first 1-2 taps are informative.")
    CAP.append(f"Global caveats: sensor extrinsics = {POSE_LABEL} (position = node rigid body; the radar/UWB share node_1 camera yaw by assumption); material uniform (ITU concrete / const reflectivity), normal incidence, no normals; single bounce; no LOS/occlusion in the splat; Wi-Fi Tx/Rx and the obstacle are schematic from paper Fig. 2(a).")
    open(os.path.join(OUT, "captions.txt"), "w").write("\n".join(CAP) + "\n")
    print("\n".join(CAP))
