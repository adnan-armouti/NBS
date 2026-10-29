"""First end-to-end OctoNet fit: the static Scene-1 scaffold against the Vayyar SFCW static target.

    python -m mm4DPS.octonet_fit fit-static [--points 30000] [--steps 300] [--lr 0.05] [--l2 1e-2]

Parametrisation (the NBS physics, per point): refl_m(f) = Fresnel(ITU concrete, cos_i_m, f) * rho_m,
T23: the head is the 3DPS rasterizer; the state is the six-parameter 3DPS material per point + 39 channel gains
(the former per-point rho model is gone; see docs/findings/sfcw_head_vs_3dps.md). Older text:
points run to ~1000x and explain the profile alone). The ITU
dispersion (a,b,c,d) stays fixed at concrete: with a 4 GHz band the per-point dispersion exponents
are not identifiable from one static profile (the gate experiments in docs/WRITE_UP.md), whereas a
per-point scale is what the range profile actually constrains. No global gain: the loss normalises
log|rp| by its guarded-band median on both sides (``sfcw_target.sfcw_static_loss``).

Loss = sfcw_static_loss(kind='logmag', weight='xcorr') on the pooled 5-recording target (bins 0-5 and
the last 6 masked, per-bin weight = cross-recording correlation) + l2 * mean(theta^2). Evaluation on
the single-recording target (target_rec_sit) says whether the fit explains the scene or the pool.
Outputs under $NBS_DATA_OUT/fit_S1_sfcw/ and viz/render_vs_gt/sfcw_fit_static.png.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch

from mm4DPS import octonet_heads as OH
from method import sfcw_target as ST
from method.freq_material import ITU_GT
from preprocessing import octonet_config as C, octonet_scene as S, vayyar_radar_config as VR

DATA_OUT = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
# the scaffold STAGE output (python -m mm4DPS.reconstruction scaffold --dataset octonet) when present, else the viz one
_SC = [d for d in ("4dps_scaffold_S1", "scaffold_S1_calibrated") if os.path.isdir(os.path.join(DATA_OUT, d))]   # v1 default; v2 (planar normals) via OCTONET_SCAFFOLD_DIR, see docs/findings
SCAFFOLD_DIR = os.environ.get("OCTONET_SCAFFOLD_DIR", os.path.join(DATA_OUT, _SC[0] if _SC else "4dps_scaffold_S1"))
EXTRINSICS = os.environ.get("OCTONET_EXTRINSICS", os.path.join(DATA_OUT, "calib_S1", "extrinsics.json"))
VIZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "viz", "render_vs_gt")


GUARD_LO = int(os.environ.get("OCTONET_SFCW_GUARD_LO", 24))   # bins < 24 (0.9 m) are the node's own near-field return, absent from the scaffold (T4)


def load_target(path: str, device, guard_lo: int | None = None) -> dict:
    z = np.load(path)
    t = {k: torch.as_tensor(z[k]) for k in z.files}
    t["rp_mean"] = t["rp_mean"].to(torch.complex128).to(device); t["mask"] = t["mask"].bool().to(device)
    g = GUARD_LO if guard_lo is None else guard_lo
    if g: t["mask"][:g] = False
    for k in ("rel", "xcorr", "snr_db"):
        if k in t: t[k] = t[k].double().to(device)
    return t


def load_inputs(n_points: int, device, seed: int = 0):
    pcl = np.load(os.path.join(SCAFFOLD_DIR, "pcl.npy")).astype(np.float64)
    if n_points and len(pcl) > n_points:
        pcl = pcl[np.random.default_rng(seed).choice(len(pcl), n_points, replace=False)]
    ext = json.load(open(EXTRINSICS))
    sensor = OH.sensor_from_extrinsics(ext, "vayyar")
    cloud = dict(xyz=torch.tensor(pcl[:, :3], device=device), normals=torch.tensor(pcl[:, 3:6], device=device))
    return pcl, cloud, sensor


def db(x, floor=-60.0):
    x = np.asarray(x, float); m = np.nanmax(x)
    return np.maximum(20 * np.log10(np.maximum(x, 1e-300) / max(m, 1e-300)), floor)


def fit_static(args) -> dict:
    """Static-scene SFCW material fit through the 3DPS rasterizer head (T23): per-point six-parameter 3DPS material
    (eps_r, eps_i, sigma_h, l_c, tau, thickness in the 3DPS raw space, bounded by reparameterize_torch) plus the
    20 + 20 per-channel log-amplitude gains of the raw Vayyar phasors. Loss = sfcw_static_loss(logmag, xcorr)."""
    from mm3DPS.rasterizer import inverse_reparameterize_torch, reparameterize_torch
    from method.freq_material import eps_from_dispersion
    dev = torch.device(args.device)
    torch.manual_seed(0)
    pcl, cloud, sensor = load_inputs(args.points, dev)
    tgt_pool = load_target(os.path.join(DATA_OUT, "sfcw_target_S1", "target_pool5.npz"), dev)
    tgt_sit = load_target(os.path.join(DATA_OUT, "sfcw_target_S1", "target_rec_sit.npz"), dev)
    n = len(pcl)
    abcd = torch.tensor(ITU_GT["concrete"], dtype=torch.float64)
    er, ei = eps_from_dispersion(abcd, OH.SFCW_FC_GHZ)
    phys0 = np.tile([float(er), float(ei), OH.SFCW_DEFAULT_ROUGHNESS["sigma_h"], OH.SFCW_DEFAULT_ROUGHNESS["l_c"],
                     OH.SFCW_DEFAULT_ROUGHNESS["tau"], OH.SFCW_DEFAULT_ROUGHNESS["thickness"]], (n, 1))
    raw0 = torch.as_tensor(inverse_reparameterize_torch(phys0), dtype=torch.float32, device=dev)
    raw6 = raw0.clone().requires_grad_(True)
    g_tx = torch.zeros(20, dtype=torch.float64, device=dev, requires_grad=True)
    g_rx = torch.zeros(20, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [raw6], "lr": args.lr}, {"params": [g_tx, g_rx], "lr": args.lr}])
    log = []; t0 = time.time()

    def render(r6, gt, gr):
        return OH.sfcw_head(cloud, {"raw6": r6}, dict(sensor, gain_tx=torch.exp(gt), gain_rx=torch.exp(gr)))

    with torch.no_grad():
        rp0 = render(raw6, g_tx, g_rx)
        l0_pool, pb0_pool = ST.sfcw_static_loss(rp0, tgt_pool, kind="logmag", weight="xcorr")
        l0_sit, pb0_sit = ST.sfcw_static_loss(rp0, tgt_sit, kind="logmag", weight="xcorr")
    print(f"points {n}; before: loss pool {float(l0_pool):.4f}, sit {float(l0_sit):.4f}; render {time.time() - t0:.1f} s")
    best = (float(l0_pool), raw6.detach().clone(), g_tx.detach().clone(), g_rx.detach().clone(), 0)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        rp = render(raw6, g_tx, g_rx)
        loss_data, _ = ST.sfcw_static_loss(rp, tgt_pool, kind="logmag", weight="xcorr")
        reg = args.l2 * ((raw6 - raw0) ** 2).mean() + args.gain_l2 * ((g_tx ** 2).mean() + (g_rx ** 2).mean())
        loss = loss_data + reg
        loss.backward(); opt.step()
        log.append(dict(step=step, loss=float(loss), data=float(loss_data), reg=float(reg), t=time.time() - t0))
        if float(loss_data) < best[0]:
            best = (float(loss_data), raw6.detach().clone(), g_tx.detach().clone(), g_rx.detach().clone(), step)
        if step % 10 == 0 or step == 1:
            ph = reparameterize_torch(raw6.detach())
            print(f"  step {step:4d} loss {float(loss):.4f} (data {float(loss_data):.4f} reg {float(reg):.4f}) eps' [{float(ph[:,0].min()):.2f},{float(ph[:,0].max()):.2f}] sigma_h med {float(ph[:,2].median())*1e3:.3f} mm gains [{float(g_tx.min()):+.2f},{float(g_tx.max()):+.2f}] {time.time() - t0:.0f} s")
        if step >= 50 and len(log) > 30 and abs(log[-1]["data"] - log[-31]["data"]) < 1e-4:
            print("  converged"); break
    raw_b, gt_b, gr_b = best[1], best[2], best[3]
    with torch.no_grad():
        rp1 = render(raw_b, gt_b, gr_b)
        l1_pool, pb1_pool = ST.sfcw_static_loss(rp1, tgt_pool, kind="logmag", weight="xcorr")
        l1_sit, pb1_sit = ST.sfcw_static_loss(rp1, tgt_sit, kind="logmag", weight="xcorr")
        physics = reparameterize_torch(raw_b).cpu().numpy()
    rho = physics[:, 0]                                   # eps' per point (the figure's colour; name kept for the plotting code)
    print(f"after (best step {best[4]}): loss pool {float(l1_pool):.4f}, sit {float(l1_sit):.4f}; eps' median {np.median(rho):.2f} [{rho.min():.2f},{rho.max():.2f}], sigma_h median {np.median(physics[:,2])*1e3:.3f} mm; gains tx std {float(gt_b.std()):.3f} rx std {float(gr_b.std()):.3f}; {time.time() - t0:.0f} s")
    theta_b = raw_b
    out = os.path.join(DATA_OUT, "fit_S1_sfcw"); os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "material.npz"), xyz=pcl[:, :3], normals=pcl[:, 3:6], physics=physics, raw6=theta_b.cpu().numpy(), gain_tx=gt_b.cpu().numpy(), gain_rx=gr_b.cpu().numpy(), physics_columns=np.array(["eps_r", "eps_i", "sigma_h", "l_c", "tau", "thickness"]))
    mask = tgt_pool["mask"].cpu().numpy(); rng_m = tgt_pool["range_m"].cpu().numpy()
    summary = dict(points=n, steps=len(log), best_step=best[4], head="3DPS render_factorized (T23)", gains=dict(tx=gt_b.cpu().numpy().tolist(), rx=gr_b.cpu().numpy().tolist()), loss_before=dict(pool=float(l0_pool), sit=float(l0_sit)),
                   loss_after=dict(pool=float(l1_pool), sit=float(l1_sit)), eps_r=dict(median=float(np.median(rho)), min=float(rho.min()), max=float(rho.max()), p05=float(np.percentile(rho, 5)), p95=float(np.percentile(rho, 95))), sigma_h_mm_median=float(np.median(physics[:, 2]) * 1e3),
                   per_bin_before_pool=np.nan_to_num(pb0_pool.cpu().numpy(), nan=-1).tolist(), per_bin_after_pool=np.nan_to_num(pb1_pool.cpu().numpy(), nan=-1).tolist(),
                   per_bin_before_sit=np.nan_to_num(pb0_sit.cpu().numpy(), nan=-1).tolist(), per_bin_after_sit=np.nan_to_num(pb1_sit.cpu().numpy(), nan=-1).tolist(),
                   seconds=time.time() - t0, args=vars(args), log=log, parametrisation="3DPS render_factorized: six-parameter Jones material per point (raw space, reparameterize_torch bounds) + 20+20 log-amplitude channel gains; loss logmag xcorr-weighted + l2 on (raw - raw0) + gain_l2")
    json.dump(summary, open(os.path.join(out, "fit_log.json"), "w"), indent=1)
    figure(pcl, rho, rp0.cpu(), rp1.cpu(), tgt_pool, tgt_sit, summary, sensor)
    return summary


def figure(pcl, rho, rp0, rp1, tgt_pool, tgt_sit, summary, sensor):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mask = tgt_pool["mask"].cpu().numpy(); r = tgt_pool["range_m"].cpu().numpy(); w = tgt_pool["xcorr"].cpu().numpy()
    def prof(rp): p = rp.abs().mean(0).numpy(); p[~mask] = np.nan; return p
    gt_pool = prof(tgt_pool["rp_mean"].cpu()); gt_sit = prof(tgt_sit["rp_mean"].cpu()); p0 = prof(rp0); p1 = prof(rp1)
    def norm_db(p): d = 20 * np.log10(np.where(np.isfinite(p), p, np.nan) + 1e-300); return d - np.nanmedian(d[mask])
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    a = ax[0, 0]; a.plot([l["step"] for l in summary["log"]], [l["data"] for l in summary["log"]], label="data loss (pool)"); a.plot([l["step"] for l in summary["log"]], [l["loss"] for l in summary["log"]], "--", label="data + l2")
    a.set_xlabel("Adam step"); a.set_ylabel("xcorr-weighted L1 of log|rp|"); a.set_title(f"loss: pool {summary['loss_before']['pool']:.3f} -> {summary['loss_after']['pool']:.3f}; held single recording {summary['loss_before']['sit']:.3f} -> {summary['loss_after']['sit']:.3f}"); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]
    a.plot(r, norm_db(gt_pool), "k", lw=2, label="GT pooled (5 rec.)"); a.plot(r, norm_db(gt_sit), "k:", lw=1, label="GT single (sit)")
    a.plot(r, norm_db(p0), "C0", lw=1.2, label="rendered before (uniform concrete)"); a.plot(r, norm_db(p1), "C3", lw=1.4, label="rendered after (per-point rho)")
    a.axvline(2.92, color="r", ls="--", lw=.8); a.set_xlabel("range (m)"); a.set_ylabel("log|rp| mean over 400 pairs, dB rel. guarded median"); a.set_title("pair-mean range profile (guarded band)"); a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[0, 2]
    pb0 = np.array(summary["per_bin_before_pool"], float); pb1 = np.array(summary["per_bin_after_pool"], float); pb0[pb0 < 0] = np.nan; pb1[pb1 < 0] = np.nan
    a.plot(r, pb0 * w, "C0", label="before"); a.plot(r, pb1 * w, "C3", label="after"); a.plot(r, w * mask, "k:", lw=.8, label="weight (xcorr)")
    a.set_xlabel("range (m)"); a.set_ylabel("weighted per-bin residual"); a.set_title("per-bin residual x weight (pool target)"); a.legend(); a.grid(alpha=.3)
    a = ax[1, 0]
    for i, pidx in enumerate((0, 105, 210, 399)):
        g = 20 * np.log10(np.abs(tgt_pool["rp_mean"].cpu().numpy()[pidx]) + 1e-300); g[~mask] = np.nan; g -= np.nanmedian(g)
        q = 20 * np.log10(np.abs(rp1.numpy()[pidx]) + 1e-300); q[~mask] = np.nan; q -= np.nanmedian(q)
        a.plot(r, g + 25 * i, "k", lw=1); a.plot(r, q + 25 * i, "C3", lw=1); a.text(r[8], 25 * i + 8, f"pair {pidx} (tx {pidx // 20}, rx {pidx % 20})", fontsize=8)
    a.set_xlabel("range (m)"); a.set_ylabel("log|rp| dB (offset per pair)"); a.set_title("individual pairs: GT (black) vs fitted render (red)"); a.grid(alpha=.3)
    a = ax[1, 1]
    sc = a.scatter(pcl[:, 0], pcl[:, 1], c=rho, s=1.5, cmap="coolwarm", vmin=-1.3, vmax=1.3)
    a.plot(*sensor["pos"][:2], "k^", ms=10); a.plot(0, 0, "k*", ms=9); a.set_aspect("equal"); a.set_xlabel("X [m]"); a.set_ylabel("Y [m]"); a.set_title("fitted per-point eps' (65 GHz) (top-down, MoCap frame)"); plt.colorbar(sc, ax=a, label="eps'")
    a = ax[1, 2]
    rr = np.linalg.norm(pcl[:, :3] - sensor["pos"], axis=1); hb = np.arange(0, 7.5, 0.1); idx = np.digitize(rr, hb)
    med = [np.median(np.log10(rho[idx == k])) if (idx == k).sum() > 5 else np.nan for k in range(1, len(hb))]
    a.plot(hb[:-1] + 0.05, med, "C3"); a.axvspan(0, r[mask][0], color="0.9"); a.axvspan(r[mask][-1], 7.5, color="0.9", label="outside the 3.71 m unambiguous range")
    a.set_xlabel("point range from the radar (m)"); a.set_ylabel("median eps'"); a.set_title("fitted rho vs range (grey: bins the loss cannot see)"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"SFCW static fit, Scene 1, {summary['points']} scaffold points, {summary['steps']} Adam steps ({summary['seconds']:.0f} s) -- 3DPS render_factorized head (Jones BSDF, six parameters per point) + 39 channel gains; loss on pair-wise range profiles only (RA axis unverified)", fontsize=11)
    fig.tight_layout(); os.makedirs(VIZ_DIR, exist_ok=True); fig.savefig(os.path.join(VIZ_DIR, "sfcw_fit_static.png"), dpi=150); plt.close(fig)
    with open(os.path.join(VIZ_DIR, "captions.txt"), "a") as f:
        f.write(f"sfcw_fit_static.png: first end-to-end static SFCW fit (mm4DPS.octonet_fit) on the calibrated scaffold ({summary['points']} pts) with the sfcw_head; per-point reflectivity scale rho on ITU concrete; xcorr-weighted log|rp| loss on the pooled target {summary['loss_before']['pool']:.3f} -> {summary['loss_after']['pool']:.3f}, held single recording {summary['loss_before']['sit']:.3f} -> {summary['loss_after']['sit']:.3f}; rho median {summary['rho']['median']:.2f} [{summary['rho']['p05']:.2f},{summary['rho']['p95']:.2f}] (5-95 %). Radar pose = node_1 rigid body + camera yaw (assumption); uniform-pitch array (range only).\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit-static")
    s.add_argument("--scaffold-dir", default=None, help=f"scaffold stage dir (default {SCAFFOLD_DIR})")
    s.add_argument("--guard-lo", type=int, default=GUARD_LO, help="mask range bins below this (default 24 = 0.9 m: the node's own near-field return)")
    s.add_argument("--points", type=int, default=30000); s.add_argument("--steps", type=int, default=300)
    s.add_argument("--lr", type=float, default=0.05); s.add_argument("--l2", type=float, default=0.1)
    s.add_argument("--chunk", type=int, default=4096); s.add_argument("--device", default="cuda")
    s.add_argument("--gain-l2", type=float, default=0.01, help="L2 on the 20+20 log-amplitude channel gains")
    a = ap.parse_args(argv)
    if a.cmd == "fit-static":
        if getattr(a, 'scaffold_dir', None): globals()['SCAFFOLD_DIR'] = a.scaffold_dir
    globals()['GUARD_LO'] = a.guard_lo
    fit_static(a); return 0


if __name__ == "__main__":
    sys.exit(main())
