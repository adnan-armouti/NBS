"""SFCW static target: the clutter-guarded temporal-mean Vayyar channel, with per-bin reliability.

Why: the Vayyar capture is 99.8 % a fixed instrument/room response; the moving person is unobservable
(WRITE_UP #19). The only fittable signal is the STATIC scene, i.e. the temporal-mean channel H_mean, and
beyond the Tx->Rx coupling (bins 0..5) it sits close to the per-frame noise floor. So the objective is
the mean channel, with the coupling and IFFT-wrap bins guarded, and every range bin weighted by how
repeatable it is.

Reliability estimators (all per range bin k, statistics across the 400 Tx/Rx pairs):
  rel[k]        split-half agreement with INTERLEAVED halves (frames 0,2,4.. vs 1,3,5..): Pearson
                corr of |rp_a[:,k]| vs |rp_b[:,k]|, clipped to [0,1]. Interleaving shares slow drift
                between the halves, so this measures fast (frame-to-frame) noise only.
  rel_contig[k] same with CONTIGUOUS halves (first T/2 vs last T/2): sensitive to drift.
  coh[k]        complex coherence |sum_p rp_a conj(rp_b)| / sqrt(sum|rp_a|^2 sum|rp_b|^2) (interleaved).
  snr_db[k]     10 log10( mean_p |rp_mean[p,k]|^2 / noise_k ) with noise_k = mean_p |rp_a - rp_b|^2 / 4:
                each half-mean carries sigma^2/(T/2) of noise, their difference 4 sigma^2/T, and the
                full mean sigma^2/T = var(diff)/4.
  n_eff         T / (1 + 2 sum_{l>=1} rho_l) with rho_l the lag autocorrelation of the range-compressed
                residual (V - H_mean) in the guarded band, summed up to the first non-positive lag
                (Bartlett). The residual is drift-dominated, so n_eff << T.
Cross-recording reliability (pool_recordings): per-bin corr across pairs of |rp_mean| between every two
recordings of the same scene, averaged -> xcorr[k].

Loss (sfcw_static_loss, torch, differentiable in rp_render):
  logmag: a = log|rp_render|, b = log|rp_target|, each minus its own median over (pairs x guarded bins)
          so the uniform-material scale drops out; L = sum_{p,k in mask} w_k |a-b| / (P * sum_k w_k),
          w_k = rel[k] (or rel_contig / coh / xcorr via `weight=`).
  ra:     the same on log|RA| from mm3DPS.data.ra_utils.vayyar_to_ra_complex (azimuth FFT over the Tx
          axis), pairs reshaped (20 Tx, 20 Rx) in octonet_config.VAY_PAIR_ORDER.

Run on the node:  cd nbs && python -m method.sfcw_target     (prints the summary, saves the targets)
"""
from __future__ import annotations
import os, json
import numpy as np
import torch

from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S
from preprocessing import vayyar_radar_config as VR
from method import sfcw_forward as SF
from mm3DPS.data import ra_utils as RA

EPS = 1e-9


def _to_torch(V) -> torch.Tensor:
    return V if torch.is_tensor(V) else torch.from_numpy(np.ascontiguousarray(V)).to(torch.complex64)


def _corr_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along axis 0 (pairs) for every column (bin): (P,K),(P,K) -> (K,)."""
    a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
    return (a * b).sum(0) / torch.sqrt((a * a).sum(0) * (b * b).sum(0) + EPS)


def guard_mask(n_freq: int = VR.N_FREQ, guard_lo: int = 6, guard_hi: int = 6) -> torch.Tensor:
    m = torch.ones(n_freq, dtype=torch.bool); m[:guard_lo] = False
    if guard_hi > 0: m[-guard_hi:] = False
    return m


def residual_autocorr(V: torch.Tensor, H_mean: torch.Tensor, mask: torch.Tensor, max_lag: int | None = None):
    """Lag autocorrelation rho_l of the range-compressed residual, averaged over pairs and guarded bins."""
    T = V.shape[0]; max_lag = max_lag or T // 3
    r = RA.sfcw_range_compress(V - H_mean[None])[..., mask]            # (T,P,Kg)
    den = (r.abs() ** 2).sum().real
    rho = [1.0]
    for l in range(1, max_lag + 1):
        rho.append(float(((r[:-l] * r[l:].conj()).sum().real / den) * T / (T - l)))
    rho = np.array(rho)
    s = 0.0
    for l in range(1, len(rho)):
        if rho[l] <= 0: break
        s += rho[l]
    return rho, float(T / (1 + 2 * s))


def sfcw_static_target(V, guard_lo: int = 6, guard_hi: int = 6, split: bool = True,
                       coupling: str | None = None, window: str = "hann") -> dict:
    """``coupling``: None (as recorded) | 'tone1' | 'tone2' = remove the per-frame direct-coupling tones first
    (method.sfcw_coupling.fit_coupling). ``window``: 'hann' (default, unchanged) | 'bh4' | 'kaiser9' for the
    range compression (the renderer must use the same: sfcw_coupling.range_compress)."""
    V = _to_torch(V)
    if coupling:
        from method import sfcw_coupling as SC
        V = SC.fit_coupling(V, n_tones=int(coupling[-1]))["V_clean"]
    rc = RA.sfcw_range_compress if window == "hann" else (lambda H: __import__("method.sfcw_coupling", fromlist=["range_compress"]).range_compress(H, window))
    T, P, K = V.shape
    mask = guard_mask(K, guard_lo, guard_hi)
    H_mean = V.mean(0); rp_mean = rc(H_mean)
    out = dict(H_mean=H_mean, rp_mean=rp_mean, mask=mask, T=T, guard=(guard_lo, guard_hi), coupling=coupling, window=window,
               range_m=torch.from_numpy(np.asarray(SF.range_axis_m(K, VR.BANDWIDTH_HZ), dtype=np.float32)))
    if split:
        rp_a, rp_b = (rc(V[0::2].mean(0)), rc(V[1::2].mean(0)))
        rp_c1, rp_c2 = (rc(V[:T // 2].mean(0)), rc(V[T // 2:].mean(0)))
        out["rel"] = _corr_pairs(rp_a.abs(), rp_b.abs()).clamp(0, 1)
        out["rel_contig"] = _corr_pairs(rp_c1.abs(), rp_c2.abs()).clamp(0, 1)
        out["coh"] = ((rp_a * rp_b.conj()).sum(0).abs() / torch.sqrt((rp_a.abs() ** 2).sum(0) * (rp_b.abs() ** 2).sum(0) + EPS)).clamp(0, 1)
        noise = ((rp_a - rp_b).abs() ** 2).mean(0) / 4
        out["snr_db"] = 10 * torch.log10((rp_mean.abs() ** 2).mean(0) / (noise + EPS))
        out["rp_half"] = (rp_a, rp_b); out["rp_contig"] = (rp_c1, rp_c2)
        rho, n_eff = residual_autocorr(V, H_mean, mask)
        out["rho"] = rho; out["n_eff"] = n_eff
        g = mask
        out["summary"] = dict(T=T, n_eff=n_eff, rel_median=float(out["rel"][g].median()),
                              rel_contig_median=float(out["rel_contig"][g].median()), coh_median=float(out["coh"][g].median()),
                              snr_db_median=float(out["snr_db"][g].median()), snr_db_min=float(out["snr_db"][g].min()),
                              snr_db_max=float(out["snr_db"][g].max()))
    return out


def load_recording(rec_or_path: str):
    p = rec_or_path if os.path.isabs(rec_or_path) else S.modality_path(1, "vayyar", rec_or_path)
    V, t = IO.load_vayyar(p)
    return _to_torch(V), t, p


def pool_recordings(recs: list[str], guard_lo: int = 6, guard_hi: int = 6,
                    coupling: str | None = None, window: str = "hann") -> dict:
    """Pooled static target over recordings of the same scene, each weighted by its frame count."""
    targets, Hs, Ts, names = [], [], [], []
    for r in recs:
        V, _, p = load_recording(r); tg = sfcw_static_target(V, guard_lo, guard_hi, coupling=coupling, window=window)
        targets.append(tg); Hs.append(tg["H_mean"] * tg["T"]); Ts.append(tg["T"]); names.append(os.path.basename(p))
    H_pool = torch.stack(Hs).sum(0) / sum(Ts)
    rc = RA.sfcw_range_compress if window == "hann" else (lambda H: __import__("method.sfcw_coupling", fromlist=["range_compress"]).range_compress(H, window))
    rp_pool = rc(H_pool)
    mask = targets[0]["mask"]
    mags = [tg["rp_mean"].abs() for tg in targets]
    xc = []
    for i in range(len(mags)):
        for j in range(i + 1, len(mags)):
            xc.append(_corr_pairs(mags[i], mags[j]))
    xcorr = torch.stack(xc).mean(0).clamp(0, 1) if xc else torch.ones(mask.numel())
    # pooled fast-noise reliability from the individual targets (frame-weighted mean)
    rel = torch.stack([tg["rel"] * tg["T"] for tg in targets]).sum(0) / sum(Ts)
    snr = torch.stack([tg["snr_db"] for tg in targets])
    return dict(H_mean=H_pool, rp_mean=rp_pool, mask=mask, T=sum(Ts), rel=rel, xcorr=xcorr, per_rec=targets, coupling=coupling, window=window,
                names=names, range_m=targets[0]["range_m"], snr_db=10 * torch.log10((10 ** (snr / 10)).sum(0)),
                summary=dict(n_rec=len(recs), T_total=sum(Ts), xcorr_median=float(xcorr[mask].median()),
                             xcorr_min=float(xcorr[mask].min()), rel_median=float(rel[mask].median())))


def _pairs_to_txrx_torch(rp: torch.Tensor) -> torch.Tensor:
    r = rp.reshape(C.VAY_N_TX, C.VAY_N_RX, rp.shape[-1])
    return r if C.VAY_PAIR_ORDER == "tx_major" else r.transpose(0, 1)


def sfcw_static_loss(rp_render: torch.Tensor, target: dict, kind: str = "logmag", weight: str = "rel", n_az: int = 128):
    """Masked, reliability-weighted L1 on normalised log-magnitude. Returns (scalar, per-bin residual (K,)).
    ``rp_render``: (400,100) complex range profiles from the renderer (already range-compressed)."""
    mask = target["mask"].to(rp_render.device)
    w = target.get(weight, None)
    w = torch.ones(mask.numel(), device=rp_render.device) if w is None else w.to(rp_render.device)
    w = w * mask
    tgt = target["rp_mean"].to(rp_render.device)
    if kind == "logmag":
        a, b = rp_render, tgt
    elif kind == "ra":
        a = RA.vayyar_to_ra_complex(_pairs_to_txrx_torch(rp_render), n_az=n_az)
        b = RA.vayyar_to_ra_complex(_pairs_to_txrx_torch(tgt), n_az=n_az)
    else:
        raise ValueError(kind)
    la = torch.log(a.abs() + EPS); lb = torch.log(b.abs() + EPS)
    la = la - la[:, mask].median(); lb = lb - lb[:, mask].median()
    res = (la - lb).abs()                                          # (P or n_az, K)
    per_bin = res.mean(0)
    loss = (per_bin * w).sum() / (w.sum() + EPS)
    per_bin = torch.where(mask, per_bin, torch.full_like(per_bin, float("nan")))
    return loss, per_bin


def save_target(target: dict, path: str, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in target.items()
           if k in ("H_mean", "rp_mean", "mask", "rel", "rel_contig", "coh", "snr_db", "xcorr", "range_m", "rho")}
    np.savez(path, **arr)
    meta = dict(target.get("summary", {})); meta.update(extra or {}); meta["T"] = int(target["T"])
    with open(path.replace(".npz", ".json"), "w") as f: json.dump(meta, f, indent=1)


def pick_scene1_recordings(n: int = 5, first: str = S.REC_SIT) -> list[str]:
    df = IO.load_index()
    df = df[df.mocap_data_path.astype(str).str.contains("mocap_S1") & df.node_1_vayyar_data_path.notna()]
    first_row = df[df.node_1_vayyar_data_path.astype(str).str.contains(first)].iloc[0]
    out, acts = [IO.resolve(first_row.node_1_vayyar_data_path)], {first_row.activity}
    for _, r in df.iterrows():
        if r.activity in acts: continue
        out.append(IO.resolve(r.node_1_vayyar_data_path)); acts.add(r.activity)
        if len(out) >= n: break
    return out


def main():
    out_dir = os.path.join(C.NBS_DATA_OUT, "sfcw_target_S1")
    V, t, p = load_recording(S.REC_SIT)
    tg = sfcw_static_target(V)
    print(f"single {os.path.basename(p)}: {tg['summary']}")
    save_target(tg, os.path.join(out_dir, "target_rec_sit.npz"), dict(recording=os.path.basename(p)))
    recs = pick_scene1_recordings(5)
    pool = pool_recordings(recs, coupling="tone2")          # default pool: two-tone on-board coupling removed (T4, 2026-10-28)
    print("pool recordings:", [os.path.basename(r).split("_activity_")[1].split("_trial")[0] for r in recs])
    print(f"pool: {pool['summary']}")
    for tg_i, nm in zip(pool["per_rec"], pool["names"]):
        print(f"   {nm.split('_activity_')[1]:30s} T={tg_i['T']:4d} n_eff={tg_i['n_eff']:6.1f} rel={tg_i['summary']['rel_median']:.3f} "
              f"rel_contig={tg_i['summary']['rel_contig_median']:.3f} snr={tg_i['summary']['snr_db_median']:5.1f} dB")
    save_target(pool, os.path.join(out_dir, "target_pool5.npz"), dict(recordings=pool["names"]))
    # loss sanity: target vs itself, scaled target, random, gradient
    rp = tg["rp_mean"].clone().requires_grad_(True)
    for name, cand in (("target itself", tg["rp_mean"]), ("target x3", tg["rp_mean"] * 3), ("random", torch.randn_like(tg["rp_mean"]))):
        for kind in ("logmag", "ra"):
            l, _ = sfcw_static_loss(cand.clone(), tg, kind=kind); print(f"   loss[{kind:6s}] {name:14s} = {float(l):.4f}")
    l, _ = sfcw_static_loss(rp * 1.3 + 0.01 * torch.randn_like(rp), tg); l.backward()
    print(f"   grad ok: |grad| mean {rp.grad.abs().mean():.3e}")
    print("saved", out_dir)


if __name__ == "__main__":
    main()
