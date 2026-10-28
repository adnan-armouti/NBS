"""SFCW direct-coupling removal for the Vayyar static target (tracker T4, plan §2).

The direct Tx->Rx leakage sits at ~bin 5-6 (0.2 m on-board path) in every pair, 15-25 dB above the scene,
and its Hann sidelobes leak into bins 6-25 (to ~1 m). It drifts frame to frame, so it is fitted and removed
PER FRAME AND PAIR in the un-windowed sweep domain:

    H[k] ~ sum_j A_j exp(-j 2 pi k b_j / K),  b_j in [3, 8] (fractional range bin), k = tone index

A single such tone's least-squares amplitude is the zero-padded IFFT of H at fractional bin b, so the LS
fit = the fractional-bin peak search of the rectangular-window range profile in [3, 8]; a second tone
(greedy, alternating LS) absorbs the cover reflection / mainlobe asymmetry. The fitted tones are
subtracted, then the usual window + IFFT applies. `fit_coupling` returns the tones, the cleaned sweeps
and the coupling sweep per frame; `coupling_sweep_mean` is the (400,100) temporal mean to put back into a
RENDER so a fit compares like with like (`add_coupling`).

Windows (`range_compress`): 'hann' (the existing default, identical to ra_utils.sfcw_range_compress),
'bh4' (4-term Blackman-Harris, -92 dB sidelobes, -3 dB width 1.90 bins = 7.1 cm vs Hann 1.44 bins =
5.4 cm), 'kaiser9'. The head's matching PSF is simply the same window applied to the rendered sweep H
before the IFFT (`octonet_heads.sfcw_channel` returns H; apply `range_compress(H, window)`), because the
renderer works in the sweep domain -- there is no separate analytic PSF to change.

Using the coupling term in a fit (octonet_fit would do):

    from method import sfcw_coupling as SC, sfcw_target as ST
    V, _, _ = ST.load_recording(rec)
    fit = SC.fit_coupling(V, n_tones=2)                 # per frame/pair tones
    coup = SC.coupling_sweep_mean(fit)                  # (400,100) complex, temporal mean of the fitted tones
    target = ST.sfcw_static_target(V, guard_lo=4)       # RAW target (not decoupled), smaller guard
    ...
    H = OH.sfcw_channel(cloud, material, sensor)        # (20,20,100) rendered sweep
    rp = SC.range_compress(SC.add_coupling(H, coup))    # render + measured coupling, same window
    loss, _ = ST.sfcw_static_loss(rp, target, kind='logmag', weight='xcorr')

Run:  cd nbs && CUDA_VISIBLE_DEVICES=6 python -m method.sfcw_coupling      (metrics, targets, refits)
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch

from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S
from preprocessing import vayyar_radar_config as VR
from method import sfcw_target as ST
from mm3DPS.data import ra_utils as RA

EPS = 1e-12
K = VR.N_FREQ


# --------------------------------------------------------------------------- windows
def window(kind: str, n: int = K, device=None, dtype=torch.float64) -> torch.Tensor:
    if kind == "hann":
        return torch.hann_window(n, periodic=False, device=device, dtype=dtype)
    if kind == "rect":
        return torch.ones(n, device=device, dtype=dtype)
    if kind == "bh4":
        from scipy.signal.windows import blackmanharris
        return torch.as_tensor(blackmanharris(n, sym=True), device=device, dtype=dtype)
    if kind == "kaiser9":
        return torch.kaiser_window(n, periodic=False, beta=9.0, device=device, dtype=dtype)
    raise ValueError(kind)


def range_compress(H: torch.Tensor, kind: str = "hann") -> torch.Tensor:
    """Windowed IFFT over the last (tone) axis. 'hann' is bitwise the ra_utils/sfcw_forward default."""
    if kind == "hann":
        return RA.sfcw_range_compress(H)
    w = window(kind, H.shape[-1], H.device, H.real.dtype).to(H.dtype)
    return torch.fft.ifft(H * w, dim=-1)


def window_resolution(kind: str) -> dict:
    """-3 dB mainlobe width (bins, cm), first null (bins) and peak sidelobe (dB) of the window's PSF."""
    w = window(kind, K).numpy(); n = 64 * K
    P = np.abs(np.fft.fft(w, n)) ** 2; P /= P.max(); Pdb = 10 * np.log10(P[: n // 2] + 1e-30)
    half_width = 2 * np.argmax(Pdb < -3) / 64
    d = np.diff(Pdb); mins = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    first_null = (mins[0]) / 64 if len(mins) else float("nan")
    sl = float(Pdb[mins[0]:].max()) if len(mins) else float("nan")
    return dict(window=kind, width_3db_bins=float(half_width), width_3db_cm=float(half_width * VR.RANGE_RES_M * 100),
                first_null_bins=float(first_null), peak_sidelobe_db=sl)


# --------------------------------------------------------------------------- coupling fit
def _tones(bins: torch.Tensor, device) -> torch.Tensor:
    """(nB, K) complex tones exp(-j 2 pi k b / K)."""
    k = torch.arange(K, device=device, dtype=torch.float64)
    return torch.exp(-2j * math.pi * k[None, :] * bins[:, None].to(device) / K)


def _fit_one_tone(Hres: torch.Tensor, bins: torch.Tensor, tones: torch.Tensor):
    """LS single-tone fit of every row of Hres (...,K) on the grid: returns (A (...,), b (...,))."""
    proj = Hres @ tones.conj().T / K                               # (..., nB) = zero-padded IFFT at fractional bins
    idx = proj.abs().argmax(-1)                                    # (...)
    A = torch.gather(proj, -1, idx[..., None])[..., 0]
    return A, bins.to(Hres.device)[idx]


def fit_coupling(V, lo: float = 0.0, hi: float = 6.0, step: float = 0.02, n_tones: int = 2, iters: int = 3,
                 second_halfwidth: float = 3.0, device: str | None = None, batch: int = 64) -> dict:
    """Per frame & pair coupling tones in the un-windowed sweep. V (T,400,K) complex.

    Returns dict(V_clean (T,400,K) complex64 on CPU, coupling (T,400,K) complex64 = the fitted tones,
    A (n_tones,T,400) complex, b (n_tones,T,400) fractional bins, residual_db (T,400): energy left in
    bins 0-5 of the Hann profile relative to before)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Vt = ST._to_torch(V); T, P, Kk = Vt.shape
    bins = torch.arange(lo, hi + 1e-9, step, dtype=torch.float64)
    tones = _tones(bins, dev)
    A_all = torch.zeros(n_tones, T, P, dtype=torch.complex128); b_all = torch.zeros(n_tones, T, P, dtype=torch.float64)
    V_clean = torch.empty_like(Vt); coup = torch.empty_like(Vt)
    kidx = torch.arange(Kk, device=dev, dtype=torch.float64)
    for t0 in range(0, T, batch):
        H = Vt[t0:t0 + batch].to(dev).to(torch.complex128)        # (Bt,P,K)
        A = [torch.zeros(H.shape[:2], dtype=torch.complex128, device=dev) for _ in range(n_tones)]
        b = [torch.full(H.shape[:2], float("nan"), dtype=torch.float64, device=dev) for _ in range(n_tones)]
        def tone_of(Aj, bj):
            return Aj[..., None] * torch.exp(-2j * math.pi * kidx[None, None, :] * bj[..., None] / Kk)
        for it in range(iters if n_tones > 1 else 1):
            for j in range(n_tones):
                others = sum(tone_of(A[i], b[i]) for i in range(n_tones) if i != j and not torch.isnan(b[i]).all()) if n_tones > 1 else 0.0
                Hres = H - others
                if j == 0:
                    A[j], b[j] = _fit_one_tone(Hres, bins, tones)
                else:
                    # second tone: within +-second_halfwidth of the first, excluding the first's own bin (+-0.3)
                    proj = Hres @ tones.conj().T / Kk
                    d = (bins.to(dev)[None, None, :] - b[0][..., None]).abs()
                    allowed = (d <= second_halfwidth) & (d >= 0.3)
                    proj = torch.where(allowed, proj, torch.zeros_like(proj))
                    idx = proj.abs().argmax(-1); A[j] = torch.gather(proj, -1, idx[..., None])[..., 0]; b[j] = bins.to(dev)[idx]
        total = sum(tone_of(A[j], b[j]) for j in range(n_tones))
        V_clean[t0:t0 + batch] = (H - total).to(torch.complex64).cpu(); coup[t0:t0 + batch] = total.to(torch.complex64).cpu()
        for j in range(n_tones):
            A_all[j, t0:t0 + batch] = A[j].cpu(); b_all[j, t0:t0 + batch] = b[j].cpu()
    rp0 = RA.sfcw_range_compress(Vt); rp1 = RA.sfcw_range_compress(V_clean)
    e0 = (rp0[..., :6].abs() ** 2).sum(-1); e1 = (rp1[..., :6].abs() ** 2).sum(-1)
    return dict(V_clean=V_clean, coupling=coup, A=A_all, b=b_all, residual_db=(10 * torch.log10((e1 + EPS) / (e0 + EPS))),
                grid=(lo, hi, step), n_tones=n_tones)


def coupling_only_level_db(fit: dict, rp_measured_mean: torch.Tensor, window_kind: str = "hann") -> dict:
    """Range-compress the fitted coupling alone (temporal mean) and report its level in bins 6-25 and 26-93
    relative to the MEASURED guarded peak: the true window-leakage contribution of the coupling."""
    c = range_compress(coupling_sweep_mean(fit)[None], window_kind)[0].abs() ** 2    # (400,K)
    m = rp_measured_mean.abs() ** 2
    g = m.mean(0)[6:94].max()
    return dict(coupling_only_bins6_25_rel_peak_db=float(10 * torch.log10(c.mean(0)[6:26].mean() / (g + EPS))),
                coupling_only_bins6_12_rel_peak_db=float(10 * torch.log10(c.mean(0)[6:13].mean() / (g + EPS))),
                coupling_only_bins26_93_rel_peak_db=float(10 * torch.log10(c.mean(0)[26:94].mean() / (g + EPS))),
                measured_bins6_25_rel_peak_db=float(10 * torch.log10(m.mean(0)[6:26].mean() / (g + EPS))),
                coupling_peak_rel_scene_peak_db=float(10 * torch.log10(c.mean(0).max() / (g + EPS))))


def coupling_sweep_mean(fit: dict) -> torch.Tensor:
    """(400,K) complex128: temporal mean of the fitted coupling tones (what a render should add back)."""
    return fit["coupling"].to(torch.complex128).mean(0)


def add_coupling(H_render: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
    """Rendered sweep H (20,20,K) or (400,K) + measured coupling sweep (400,K) -> same shape as H."""
    c = coupling.to(H_render.device).to(H_render.dtype)
    return H_render + (c.reshape(H_render.shape) if H_render.ndim == 3 else c)


# --------------------------------------------------------------------------- metrics
def leakage_metrics(rp_mean: torch.Tensor) -> dict:
    """On a (400,K) range profile: energy in bins 6-25 and 26-93 relative to the coupling bins 0-5, and the
    bins 6-25 level relative to the guarded-band peak (dB, mean over pairs)."""
    p = rp_mean.abs() ** 2
    e_c = p[:, :6].sum(1).mean(); e_near = p[:, 6:26].sum(1).mean(); e_far = p[:, 26:94].sum(1).mean()
    prof = p.mean(0); g = prof[6:94].max()
    return dict(near_over_coupling_db=float(10 * torch.log10(e_near / (e_c + EPS))),
                far_over_coupling_db=float(10 * torch.log10(e_far / (e_c + EPS))),
                near_over_far_db=float(10 * torch.log10(e_near / (e_far + EPS))),
                bins6_25_mean_rel_peak_db=float(10 * torch.log10(prof[6:26].mean() / (g + EPS))),
                coupling_bins0_5_rel_peak_db=float(10 * torch.log10(prof[:6].mean() / (g + EPS))))


# --------------------------------------------------------------------------- targets with options
def build_targets(recs: list[str], coupling: str | None, win: str, guard_lo: int = 6) -> dict:
    """pool (and the first recording's single) target through sfcw_target with the new options."""
    pool = ST.pool_recordings(recs, guard_lo=guard_lo, coupling=coupling, window=win)
    single = pool["per_rec"][0]
    return dict(pool=pool, single=single)


# --------------------------------------------------------------------------- decisive refit (copy of octonet_fit's loop)
def refit(tgt_pool: dict, tgt_single: dict, win: str = "hann", coupling_add: torch.Tensor | None = None,
          points: int = 30000, steps: int = 200, lr: float = 0.05, l2: float = 0.1, rho_max: float = 20.0,
          device: str = "cuda", chunk: int = 4096, seed: int = 0) -> dict:
    """The static fit of mm4DPS.octonet_fit (per-point rho, concrete dispersion, logmag xcorr loss) with a
    chosen window and optional measured-coupling term added to the render. Reports rho near/far."""
    from mm4DPS import octonet_heads as OH
    from mm4DPS import octonet_fit as OF
    from method.freq_material import ITU_GT
    dev = torch.device(device); torch.manual_seed(seed)
    pcl, cloud, sensor = OF.load_inputs(points, dev, seed)
    def to_dev(t):
        d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items() if k in ("rp_mean", "mask", "rel", "xcorr", "snr_db", "range_m")}
        d["rp_mean"] = d["rp_mean"].to(torch.complex128); d["mask"] = d["mask"].bool()
        for k in ("rel", "xcorr", "snr_db"):
            if k in d: d[k] = d[k].double()
        return d
    tp, ts = to_dev(tgt_pool), to_dev(tgt_single)
    n = len(pcl)
    abcd = torch.tensor(ITU_GT["concrete"], dtype=torch.float64, device=dev).expand(n, 4).contiguous()
    theta = torch.zeros(n, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    rho_of = lambda th: torch.exp(math.log(rho_max) * torch.tanh(th))
    coup = None if coupling_add is None else coupling_add.to(dev)
    def render(th):
        H = OH.sfcw_channel(cloud, dict(abcd=abcd, rho=rho_of(th)), sensor, chunk=chunk)
        if coup is not None: H = add_coupling(H, coup)
        return range_compress(H, win).reshape(-1, K)
    t0 = time.time()
    with torch.no_grad():
        rp0 = render(theta); l0p, _ = ST.sfcw_static_loss(rp0, tp, "logmag", "xcorr"); l0s, _ = ST.sfcw_static_loss(rp0, ts, "logmag", "xcorr")
    best = (float(l0p), theta.detach().clone(), 0); log = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        ld, _ = ST.sfcw_static_loss(render(theta), tp, "logmag", "xcorr"); loss = ld + l2 * (theta ** 2).mean()
        loss.backward(); opt.step(); log.append(float(ld))
        if float(ld) < best[0]: best = (float(ld), theta.detach().clone(), step)
        if step >= 50 and abs(log[-1] - log[-31]) < 1e-4: break
    with torch.no_grad():
        rp1 = render(best[1]); l1p, pb1 = ST.sfcw_static_loss(rp1, tp, "logmag", "xcorr"); l1s, _ = ST.sfcw_static_loss(rp1, ts, "logmag", "xcorr")
    rho = rho_of(best[1]).cpu().numpy()
    r_pt = np.linalg.norm(pcl[:, :3] - np.asarray(sensor["pos"]), axis=1)
    near, far = rho[r_pt < 1.0], rho[(r_pt >= 1.0) & (r_pt < 3.5)]
    return dict(window=win, coupling_added=coup is not None, steps=len(log), best_step=best[2], seconds=time.time() - t0,
                loss_before=dict(pool=float(l0p), single=float(l0s)), loss_after=dict(pool=float(l1p), single=float(l1s)),
                rho_median_lt1m=float(np.median(near)), rho_median_1to3p5m=float(np.median(far)), rho_median_all=float(np.median(rho)),
                rho_p05=float(np.percentile(rho, 5)), rho_p95=float(np.percentile(rho, 95)), n_near=int(len(near)),
                rho=rho, r_pt=r_pt, rp_before=rp0.cpu(), rp_after=rp1.cpu(), per_bin_after=pb1.cpu().numpy())


# --------------------------------------------------------------------------- main
def main():
    out_dir = os.path.join(C.NBS_DATA_OUT, "sfcw_target_S1"); os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"windows": {w: window_resolution(w) for w in ("hann", "bh4", "kaiser9")}}
    print("window PSFs:", json.dumps(report["windows"], indent=None))
    # ---- 1. per-frame coupling fit on REC_SIT
    V, t, p = ST.load_recording(S.REC_SIT)
    for nt in (1, 2):
        fit = fit_coupling(V, n_tones=nt, device=dev)
        rp0 = RA.sfcw_range_compress(V.mean(0)); rp1 = RA.sfcw_range_compress(fit["V_clean"].mean(0))
        m0, m1 = leakage_metrics(rp0), leakage_metrics(rp1)
        res = float(fit["residual_db"].mean()); b0 = fit["b"][0]
        lk = coupling_only_level_db(fit, rp0); report[f"sit_tone{nt}_leakage_only"] = lk
        print(f"[{nt} tone] coupling alone (Hann) in bins 6-25: {lk['coupling_only_bins6_25_rel_peak_db']:+.1f} dB rel. scene peak (bins 6-12: {lk['coupling_only_bins6_12_rel_peak_db']:+.1f}); measured bins 6-25: {lk['measured_bins6_25_rel_peak_db']:+.1f} dB; coupling peak {lk['coupling_peak_rel_scene_peak_db']:+.1f} dB above the scene peak")
        print(f"[{nt} tone] fitted bin median {float(b0.median()):.2f} (p05 {float(b0.quantile(0.05)):.2f}, p95 {float(b0.quantile(0.95)):.2f}); "
              f"coupling energy left in bins 0-5: {res:+.1f} dB; bins 6-25 rel peak: {m0['bins6_25_mean_rel_peak_db']:+.1f} -> {m1['bins6_25_mean_rel_peak_db']:+.1f} dB; "
              f"near/far energy: {m0['near_over_far_db']:+.1f} -> {m1['near_over_far_db']:+.1f} dB")
        report[f"sit_tone{nt}"] = dict(bin_median=float(b0.median()), residual_db=res, before=m0, after=m1,
                                       b_std_over_frames=float(b0.std(0).mean()), A_rel_std_over_frames=float((fit["A"][0].abs().std(0) / (fit["A"][0].abs().mean(0) + EPS)).mean()))
    # ---- 2+5. targets: baseline / decoupled (2 tones) / decoupled + bh4, pool of 5 + single
    recs = ST.pick_scene1_recordings(5)
    variants = {"baseline_hann": (None, "hann"), "decoupled_hann": ("tone2", "hann"), "decoupled_bh4": ("tone2", "bh4"), "baseline_bh4": (None, "bh4")}
    targets = {}
    for name, (cp, win) in variants.items():
        tg = build_targets(recs, cp, win); targets[name] = tg
        pool = tg["pool"]; g = pool["mask"]
        near = slice(6, 26)
        info = dict(xcorr_median=float(pool["xcorr"][g].median()), xcorr_min=float(pool["xcorr"][g].min()),
                    xcorr_bins6_25=float(pool["xcorr"][near].mean()), xcorr_bins26_93=float(pool["xcorr"][26:94].mean()),
                    rel_bins6_25=float(pool["rel"][near].mean()), snr_bins6_25_db=float(pool["snr_db"][near].median()),
                    leakage=leakage_metrics(pool["rp_mean"]))
        report[f"target_{name}"] = info
        print(f"target {name:15s}: xcorr median {info['xcorr_median']:.3f} min {info['xcorr_min']:.3f} | bins 6-25 xcorr {info['xcorr_bins6_25']:.3f} rel {info['rel_bins6_25']:.3f} "
              f"| bins 6-25 rel peak {info['leakage']['bins6_25_mean_rel_peak_db']:+.1f} dB, near/far {info['leakage']['near_over_far_db']:+.1f} dB")
        suffix = {"baseline_hann": None, "decoupled_hann": "target_pool5_decoupled.npz", "decoupled_bh4": "target_pool5_decoupled_bh4.npz", "baseline_bh4": "target_pool5_bh4.npz"}[name]
        if suffix:
            ST.save_target(pool, os.path.join(out_dir, suffix), dict(recordings=pool["names"], coupling=cp, window=win))
            ST.save_target(tg["single"], os.path.join(out_dir, suffix.replace("pool5", "rec_sit")), dict(recording=pool["names"][0], coupling=cp, window=win))
    # ---- 3. coupling term for the render (from the sit recording, 2 tones) saved
    fit2 = fit_coupling(V, n_tones=2, device=dev); coup = coupling_sweep_mean(fit2)
    np.save(os.path.join(out_dir, "coupling_sweep_rec_sit.npy"), coup.numpy())
    # ---- 4. decisive refits
    fits = {}
    cfgs = [("baseline_hann", "hann", None, "baseline_hann"), ("decoupled_hann", "hann", None, "decoupled_hann"),
            ("decoupled_bh4", "bh4", None, "decoupled_bh4"), ("raw+coupling_in_render_hann_guard3", "hann", coup, "guard3"),
            ("decoupled_hann_guard24", "hann", None, "guard24")]
    for name, win, cadd, tname in cfgs:
        if tname == "guard3":
            tgp = ST.pool_recordings(recs, guard_lo=3); tgs = tgp["per_rec"][0]
        elif tname == "guard24":
            tgp = ST.pool_recordings(recs, guard_lo=24, coupling="tone2"); tgs = tgp["per_rec"][0]
        else:
            tgp, tgs = targets[tname]["pool"], targets[tname]["single"]
        r = refit(tgp, tgs, win=win, coupling_add=cadd, device=dev)
        # comparable held metric: xcorr-weighted residual on bins 24-93 only (the scaffold-covered range), single recording
        pb = torch.as_tensor(r["per_bin_after"]); w = tgp["xcorr"].clone().cpu(); w[:24] = 0; w[94:] = 0; w = torch.nan_to_num(w)
        r["held_resid_bins24_93"] = float((torch.nan_to_num(pb) * w).sum() / (w.sum() + EPS))
        fits[name] = r
        print(f"fit {name:36s}: loss pool {r['loss_before']['pool']:.3f}->{r['loss_after']['pool']:.3f}, single {r['loss_before']['single']:.3f}->{r['loss_after']['single']:.3f} | "
              f"rho median <1 m {r['rho_median_lt1m']:.3f}, 1-3.5 m {r['rho_median_1to3p5m']:.3f}, all {r['rho_median_all']:.3f} [{r['rho_p05']:.2f},{r['rho_p95']:.2f}] | held resid bins 24-93 {r['held_resid_bins24_93']:.3f} | {r['steps']} steps {r['seconds']:.0f} s")
        report[f"fit_{name}"] = {k: v for k, v in r.items() if k not in ("rho", "r_pt", "rp_before", "rp_after", "per_bin_after")}
    torch.save(dict(fits={k: dict(rho=v["rho"], r_pt=v["r_pt"], rp_before=v["rp_before"], rp_after=v["rp_after"], per_bin_after=v["per_bin_after"]) for k, v in fits.items()},
                    targets={k: dict(rp_mean=v["pool"]["rp_mean"], xcorr=v["pool"]["xcorr"], rel=v["pool"]["rel"], mask=v["pool"]["mask"]) for k, v in targets.items()},
                    fit_sit_tone2=dict(A=fit2["A"], b=fit2["b"], residual_db=fit2["residual_db"], V_clean_mean=fit2["V_clean"].mean(0), V_mean=ST._to_torch(V).mean(0),
                                       pair_example=(fit2["V_clean"][0, 0], ST._to_torch(V)[0, 0], fit2["coupling"][0, 0]))),
               os.path.join(out_dir, "coupling_study.pt"))
    json.dump(report, open(os.path.join(out_dir, "coupling_study.json"), "w"), indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
