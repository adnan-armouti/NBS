"""Scoring: the measures, the paper's tables, and the per-workspace evaluation.

Four modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C9 --
`evaluation/measures.py`, `evaluation/tables.py`, `evaluation/scoring.py` and
`evaluate.py` -- in dependency order.

    measures   every metric the paper quotes, on linear magnitude
    tables     tab:fidelity and the three supplement tables, .md and .tex
    scoring    the micro-Doppler score, the permutation null, the sweeps
    evaluate   one workspace (or a directory of them) end to end

`RADARS` was declared twice, identically, and the two `main()`s became one
dispatcher: `python -m mm4DPS.evaluation {evaluate,score-microdoppler,make-tables}`.
"""

from __future__ import annotations

import argparse
import json
import mm4DPS
import numpy as np
import subprocess

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from skimage.measure import euler_number, label
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize

from mm4DPS import Dataset, current_dataset, dispatch


# --------------------------------------------------------------------
# measures: every metric the paper quotes
# --------------------------------------------------------------------

__all___metrics = [
    "amplitude", "pearson", "cfar_mask", "hysteresis_mask",
    "range_azimuth_metrics", "range_doppler_metrics", "microdoppler_metrics",
    "cl_dice", "betti_numbers", "clutter_removed",
    "range_azimuth_suppressed_metrics",
]



def amplitude(power: np.ndarray) -> np.ndarray:
    """The house domain: linear amplitude, `sqrt(max(power, 0))`."""
    return np.sqrt(np.maximum(np.asarray(power, dtype=np.float64), 0.0))



def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, exact at 1.0 for identical input. See module docstring."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc @ xc) * (yc @ yc))
    if den == 0.0:
        return float("nan")
    return float((xc @ yc) / den)



# --------------------------------------------------------------------- masks

def _noise_scale(a: np.ndarray) -> float:
    """Robust noise amplitude: the median, which a sparse target barely moves."""
    return float(np.median(a))



def cfar_mask(real: np.ndarray, p_fa: float = 1e-3) -> np.ndarray:
    """**Mask A** -- cell-averaging CFAR on the REAL product, applied to both.

    Rayleigh amplitude under noise: P(a > T) = exp(-T^2 / (2 sigma^2)), so the
    threshold for a false-alarm rate is sigma * sqrt(-2 ln p_fa). sigma is
    estimated from the median (median = sigma * sqrt(2 ln 2) for Rayleigh),
    which a sparse target barely moves.
    """
    a = np.asarray(real, dtype=np.float64)
    sigma = _noise_scale(a) / np.sqrt(2.0 * np.log(2.0))
    if sigma <= 0:
        return np.zeros(a.shape, dtype=bool)
    return a > sigma * np.sqrt(-2.0 * np.log(p_fa))



def hysteresis_mask(a: np.ndarray, p_fa: float = 1e-3,
                    loose: float = 1e-1) -> np.ndarray:
    """**Mask B** -- seed at a strict `p_fa`, grow into 4-connected neighbours.

    Computed INDEPENDENTLY on each signature: the question is whether the
    rendered structure has the same connectivity as the real one, and a shared
    mask would assume that rather than measure it. Hysteresis because plain
    thresholding fragments the weak limb arcs and destroys exactly the
    connectivity the topology metrics exist to score.
    """
    seed = cfar_mask(a, p_fa)
    grow = cfar_mask(a, loose)
    if not seed.any():
        return seed
    lab = label(grow, connectivity=1)
    keep = np.unique(lab[seed & (lab > 0)])
    return np.isin(lab, keep[keep > 0])



# ------------------------------------------------------------- range-azimuth

def range_azimuth_metrics(real: np.ndarray, rendered: np.ndarray) -> dict:
    """Per frame, then mean and std across frames. Inputs are (frames, ...) amplitude."""
    real, rendered = np.asarray(real, np.float64), np.asarray(rendered, np.float64)
    if real.shape != rendered.shape:
        raise ValueError(f"shape mismatch: {real.shape} vs {rendered.shape}")
    r, s, n, p = [], [], [], []
    for f in range(real.shape[0]):
        for acc, v in zip((r, s, n, p), _image_frame_metrics(real[f], rendered[f])):
            acc.append(v)
    return _image_summary(r, s, n, p, real.shape[0])


def _image_frame_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """One frame's Pearson, SSIM, NRMSE and PSNR on linear amplitude (``a`` measured, ``b`` rendered), any rank."""
    r = pearson(a, b)
    rng = float(a.max())
    s = _ssim(a, b, rng if rng > 0 else 1.0)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    denom = float(np.sqrt(np.mean(a ** 2)))
    n = rmse / denom if denom > 0 else float("nan")
    # PSNR: the one logarithm, on a scalar ratio of linear quantities.
    p = float(20.0 * np.log10(rng / rmse)) if rmse > 0 else float("inf")
    return r, s, n, p


def _image_summary(r, s, n, p, frames: int) -> dict:
    return {
        "pearson_mean": float(np.mean(r)), "pearson_std": float(np.std(r)),
        "ssim_mean": float(np.mean(s)), "ssim_std": float(np.std(s)),
        "nrmse_mean": float(np.mean(n)), "nrmse_std": float(np.std(n)),
        "psnr_mean": float(np.mean(p)), "psnr_std": float(np.std(p)),
        "frames": int(frames),
    }



# ------------------------------------------- range-azimuth-elevation, complex range profile, ADC

def range_azimuth_elevation_metrics(real_rp: np.ndarray, rendered_rp: np.ndarray, radar) -> dict:
    """The range-azimuth figures on the full array's ``|RAE|`` volume: per frame, then mean and std across frames.

    Inputs are the first loop's per-channel complex range profiles, ``(frames, tx, rx, range)``. Both sides take
    the trainer's own RAE transform (`forward_render.profiles_to_rae_spectrum_reference`: every TX-RX channel on
    the 2D azimuth x elevation grid, both windows, both FFTs, fftshifted), so the view is the one the elevation
    terms of the objective fit, and the per-frame Pearson / SSIM / NRMSE / PSNR are range_azimuth_metrics' own.
    """
    import torch
    from mm4DPS.forward_render import profiles_to_rae_spectrum_reference

    def rae(x):
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.complex64))
        return profiles_to_rae_spectrum_reference(t.real, t.imag, radar=radar).abs().double().numpy()

    r, s, n, p = [], [], [], []
    for f in range(real_rp.shape[0]):
        a, b = rae(real_rp[f]), rae(rendered_rp[f])
        values = list(_image_frame_metrics(a, b))
        # SSIM per elevation slice (a range x azimuth image, the RA SSIM's own geometry), then the mean over the
        # slices: HuPR's array has two elevation rows, too few for any SSIM window along that axis
        rng = float(a.max())
        values[1] = float(np.nanmean([_ssim(a[..., e], b[..., e], rng if rng > 0 else 1.0) for e in range(a.shape[-1])]))
        for acc, v in zip((r, s, n, p), values):
            acc.append(v)
    out = _image_summary(r, s, n, p, real_rp.shape[0])
    out["shape"] = list(rae(real_rp[0]).shape) if real_rp.shape[0] else None      # (range, azimuth, elevation)
    return out


def _minmax(a: np.ndarray) -> np.ndarray:
    mn, mx = float(a.min()), float(a.max())
    return np.zeros_like(a, dtype=np.float64) if mx - mn < 1e-30 else (a.astype(np.float64) - mn) / (mx - mn)


def _magnitude_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """The 3DPS magnitude columns: |.| of each side min-max normalised on its own, then Pearson, PSNR, SSIM, RMSE."""
    p, g = _minmax(np.abs(pred)), _minmax(np.abs(gt))
    mse = float(np.mean((p - g) ** 2))
    return {"corr": pearson(g, p), "psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
            "ssim": _ssim(g, p, 1.0), "rmse": float(np.sqrt(mse))}


def _differential_phase(pred: np.ndarray, gt: np.ndarray, axis: int) -> tuple[float, float]:
    """Relative-phase fidelity along one axis: the phase step between neighbours, rendered against measured.

    dphi[i] = arg(x[i+1] conj(x[i])) along ``axis``; the score is the |gt[i]| |gt[i+1]|-weighted mean of
    cos(dphi_pred - dphi_gt) (1 = the same relative phase, 0 = random) and the weighted RMSE of the wrapped
    difference in degrees. Invariant to any per-bin phase rotation along the other axis by construction, so it
    does not penalise the absolute phase a magnitude-only objective leaves free.
    """
    p, g = np.moveaxis(pred, axis, -1), np.moveaxis(gt, axis, -1)
    if p.shape[-1] < 2:
        return float("nan"), float("nan")
    diff = np.angle((p[..., 1:] * np.conj(p[..., :-1])) * np.conj(g[..., 1:] * np.conj(g[..., :-1])))
    w = np.abs(g[..., 1:]) * np.abs(g[..., :-1])
    total = float(w.sum())
    if total <= 1e-30:
        return float("nan"), float("nan")
    return float((w * np.cos(diff)).sum() / total), float(np.degrees(np.sqrt((w * diff ** 2).sum() / total)))


def _phase_correct_channel_range(pred: np.ndarray, gt: np.ndarray, start_bin: int, iterations: int = 3) -> np.ndarray:
    """The rank-1 calibration nuisance removed: alternate the per-range and per-channel LS phase rotations of
    ``pred`` onto ``gt`` (the per-channel one over the retained range bins only). Only the ADC envelope uses it:
    |ADC| mixes the CRP's phase through the inverse range FFT; every |CRP| figure is invariant to it."""
    pred = pred.copy()
    for _ in range(iterations):
        pred = pred * np.exp(1j * np.angle(np.sum(gt * np.conj(pred), axis=0)))[None, :]
        pred = pred * np.exp(1j * np.angle(np.sum(gt[:, start_bin:] * np.conj(pred[:, start_bin:]), axis=1)))[:, None]
    return pred


def complex_profile_metrics(real_rp: np.ndarray, rendered_rp: np.ndarray, radar, start_bin: int) -> dict:
    """Complex range profile (CRP) and ADC fidelity, the protocol of 3DPS (its supplement, "CRP and ADC evaluation";
    mmir/evaluation/eval_crp_adc.py), per frame, then mean and std across frames.

    Per frame, both sides: the first loop's azimuth virtual array on its grid with the azimuth window
    (`forward_render._virtual_channels`, the channels the RA image is formed from), the range bins before
    ``start_bin`` (the render's first retained bin) zeroed. CRP: the magnitude columns on bins >= start_bin, and the
    relative phase across range (dphi_r) and across virtual channels (dphi_v) on the raw signals. ADC = the inverse
    range FFT of that CRP (the fast-time samples, range window included on both sides): the envelope correlation
    after the per-channel + per-range phase calibration, the magnitude columns, and the relative phase across
    fast time (dphi_t) on the raw signals.
    """
    import torch
    from mm4DPS.forward_render import _virtual_channels

    def grid(x):
        g = _virtual_channels(torch.from_numpy(np.ascontiguousarray(x, dtype=np.complex64)), radar).numpy()
        return g.astype(np.complex128)                                            # (G, range)

    rows: dict[str, list] = {}
    def put(prefix, d):
        for k, v in d.items():
            rows.setdefault(f"{prefix}{k}", []).append(v)

    for f in range(real_rp.shape[0]):
        gt, pr = grid(real_rp[f]), grid(rendered_rp[f])
        corrected = _phase_correct_channel_range(pr, gt, start_bin)
        gt[:, :start_bin] = 0; pr[:, :start_bin] = 0; corrected[:, :start_bin] = 0
        put("crp.", _magnitude_scores(pr[:, start_bin:], gt[:, start_bin:]))
        for name, axis in (("dphi_r", -1), ("dphi_v", 0)):
            coh, rmse = _differential_phase(pr[:, start_bin:], gt[:, start_bin:], axis)
            put("crp.", {f"{name}_coherence": coh, f"{name}_rmse_deg": rmse})
        adc_gt, adc_raw, adc_cal = (np.fft.ifft(x, axis=-1) for x in (gt, pr, corrected))
        mag = _magnitude_scores(adc_cal, adc_gt)
        put("adc.", {"envelope_corr": mag.pop("corr"), **mag})
        coh, rmse = _differential_phase(adc_raw, adc_gt, -1)
        put("adc.", {"dphi_t_coherence": coh, "dphi_t_rmse_deg": rmse})
    out = {"crp": {}, "adc": {}, "frames": int(real_rp.shape[0]), "first_retained_range_bin": int(start_bin),
           "protocol": "3DPS CRP/ADC: azimuth virtual array, azimuth window, near-field bins zeroed; magnitude on "
                       "independently min-max-normalised |.|; relative phase = |gt|-weighted <cos(dphi)> on raw signals; "
                       "ADC envelope after per-channel + per-range phase calibration"}
    for key, values in rows.items():
        family, name = key.split(".", 1)
        v = np.asarray(values, dtype=np.float64)
        out[family][f"{name}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        out[family][f"{name}_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
    return out


def clutter_removed(x: np.ndarray) -> np.ndarray:
    """Slow-time mean removal, each side using its OWN mean.

    The same clutter removal the official preprocessing performs. Each side
    removes its own temporal mean, so no measured data enters the rendered
    path -- a shared mean would be exactly the replay this project forbids.
    """
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)



def range_azimuth_suppressed_metrics(real: np.ndarray, rendered: np.ndarray,
                                     p_fa: float = 1e-3) -> dict:
    """Range-azimuth agreement with the STATIC SCENE SUPPRESSED.

    The plain range-azimuth Pearson does not discriminate: scored against a
    *different* sequence's real product it reaches +0.7735 against a weakest
    matched pair of +0.7315, and SSIM is worse still (permuted +0.7974 above a
    matched mean of +0.7409). Both are dominated by the room, the range gate
    and the antenna pattern, which every HuPR recording shares, so neither can
    separate two sequences and neither would separate two methods.

    Removing each side's slow-time mean fixes it. Measured over nine ordered
    sequence pairs per radar:

        variant                    matched min  permuted max  overlap
        raw                             0.7315        0.7735    2/12
        occupied-raw                    0.7407        0.7936    2/12
        clutter-removed                 0.0398        0.0183    0/12
        occupied-clutter-removed        0.0469        0.0217    0/12

    Occupancy alone does not help, because the static scene is bright and sits
    *inside* the occupied cells. Clutter removal is what does the work; Mask A
    on top of it adds a little margin.

    The absolute value falls from ~0.80 to ~0.30, and that is the point: about
    0.67 of the old number was scene the model never had to earn. What remains
    is attributable, because the permuted control sits at 0.0013.
    """
    real, rendered = np.asarray(real, np.float64), np.asarray(rendered, np.float64)
    if real.shape != rendered.shape:
        raise ValueError(f"shape mismatch: {real.shape} vs {rendered.shape}")
    rc, sc = clutter_removed(real), clutter_removed(rendered)
    plain, occupied, frac = [], [], []
    for f in range(real.shape[0]):
        plain.append(pearson(rc[f].ravel(), sc[f].ravel()))
        m = cfar_mask(np.abs(rc[f]), p_fa)          # Mask A, from the REAL side
        frac.append(float(m.mean()))
        occupied.append(pearson(rc[f][m].ravel(), sc[f][m].ravel())
                        if m.sum() >= 8 else np.nan)
    plain, occupied = np.asarray(plain), np.asarray(occupied)
    if real.shape[0] < 3:
        # The clutter removal subtracts the mean over the scored frames, so
        # on one or two frames (a still person's motion subset: RT-Pose 157
        # has one) it leaves nothing, and an empty Mask A is not a broken
        # product but an undefined figure. Recorded as such.
        nan = float("nan")
        return {"clutter_removed_pearson_mean": nan, "clutter_removed_pearson_std": nan,
                "occupied_clutter_removed_pearson_mean": nan, "occupied_clutter_removed_pearson_std": nan,
                "occupancy_fraction": float(np.mean(frac)), "frames": int(real.shape[0]),
                "note": "fewer than three motion frames: the clutter-removed metrics are undefined"}
    if not np.isfinite(occupied).any():
        # Never return NaN here. An empty Mask A on every frame means the CFAR
        # threshold found no target in the clutter-removed product at all, which
        # is a broken input or a wrong p_fa, not a score of "no agreement".
        raise ValueError(
            f"Mask A is empty on all {real.shape[0]} frames at p_fa={p_fa}: the "
            f"clutter-removed real product has no cell above the CFAR "
            f"threshold. That is a broken product or a wrong p_fa, not a "
            f"result. Mean occupancy was {np.mean(frac):.6f}.")
    return {
        "clutter_removed_pearson_mean": float(np.nanmean(plain)),
        "clutter_removed_pearson_std": float(np.nanstd(plain)),
        "occupied_clutter_removed_pearson_mean": float(np.nanmean(occupied)),
        "occupied_clutter_removed_pearson_std": float(np.nanstd(occupied)),
        "occupancy_fraction": float(np.mean(frac)),
        "frames": int(real.shape[0]),
    }



# ------------------------------------------------------------- range-Doppler

def range_doppler_metrics(real: np.ndarray, rendered: np.ndarray,
                          p_fa: float = 1e-3) -> dict:
    """Split by Mask A. Ratios are rendered over real, so 1.00 is exact."""
    real, rendered = np.asarray(real, np.float64), np.asarray(rendered, np.float64)
    if real.shape != rendered.shape:
        raise ValueError(f"shape mismatch: {real.shape} vs {rendered.shape}")
    occ_p, lvl, sig = [], [], []
    sig_num = sig_den = 0.0
    scnr_r, scnr_s = [], []
    thin = 0
    for f in range(real.shape[0]):
        a, b = real[f], rendered[f]
        M = cfar_mask(a, p_fa)
        U = ~M
        if M.any():
            # A Pearson needs cells to correlate over: a frame whose Mask A
            # holds fewer than 8 (a still person: one or two cells) has no
            # defined correlation and is left out of the mean rather than
            # contributing a NaN that empties it (seq 38, 140 scored frames,
            # most of them with a single occupied cell). The power ratio
            # still counts every occupied cell.
            if int(M.sum()) >= 8:
                occ_p.append(pearson(a[M], b[M]))
            else:
                thin += 1
            sig_num += float(np.sum(b[M] ** 2))
            sig_den += float(np.sum(a[M] ** 2))
        if U.any():
            ma, mb = float(np.mean(a[U])), float(np.mean(b[U]))
            lvl.append(mb / ma if ma > 0 else float("nan"))
            sa, sb = float(np.std(a[U])), float(np.std(b[U]))
            sig.append(sb / sa if sa > 0 else float("nan"))
        if M.any() and U.any():
            scnr_r.append(float(np.mean(a[M] ** 2) / np.mean(a[U] ** 2)))
            scnr_s.append(float(np.mean(b[M] ** 2) / np.mean(b[U] ** 2)))
    # floor diagnostics, pooled over every unoccupied bin and every frame
    Ms = np.stack([cfar_mask(real[f], p_fa) for f in range(real.shape[0])])
    ua, ub = real[~Ms], rendered[~Ms]
    w1 = float(wasserstein_distance(ub, ua)) if ua.size else float("nan")
    occ_p = [v for v in occ_p if np.isfinite(v)]
    return {
        # over the frames with a defined correlation; 0 with
        # `occupied_pearson_frames` 0 when no scored frame has 8 occupied
        # cells (nothing moved), never NaN
        "occupied_pearson_mean": float(np.mean(occ_p)) if occ_p else 0.0,
        "occupied_pearson_std": float(np.std(occ_p)) if occ_p else 0.0,
        "occupied_pearson_frames": len(occ_p),
        "occupied_thin_frames": thin,
        "signal_power_ratio": sig_num / sig_den if sig_den > 0 else float("nan"),
        "floor_level_ratio": float(np.mean(lvl)) if lvl else float("nan"),
        "floor_sigma_ratio": float(np.mean(sig)) if sig else float("nan"),
        "scnr_ratio": (float(np.mean(scnr_s)) / float(np.mean(scnr_r))
                       if scnr_r and np.mean(scnr_r) > 0 else float("nan")),
        "floor_wasserstein1": w1,
        "floor_lag1_real": _lag1(real, ~Ms),
        "floor_lag1_rendered": _lag1(rendered, ~Ms),
        "p_fa": p_fa, "frames": int(real.shape[0]),
    }



def _lag1(x: np.ndarray, unocc: np.ndarray) -> float:
    """Mean per-bin lag-1 autocorrelation across frames, over always-unoccupied bins."""
    always = unocc.all(axis=0)
    if x.shape[0] < 3 or not always.any():
        return float("nan")
    series = x[:, always]
    vals = [pearson(series[:-1, i], series[1:, i]) for i in range(series.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")



# -------------------------------------------------------------- micro-Doppler

def _moments(S: np.ndarray, bins: np.ndarray):
    w = S ** 2
    tot = w.sum(axis=0)
    good = tot > 0
    cen = np.full(S.shape[1], np.nan)
    spr = np.full(S.shape[1], np.nan)
    cen[good] = (bins[:, None] * w[:, good]).sum(axis=0) / tot[good]
    spr[good] = np.sqrt((((bins[:, None] - cen[good]) ** 2) * w[:, good]).sum(axis=0) / tot[good])
    return cen, spr



def cl_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Centreline Dice between two Mask B segmentations.

    Tprec is the fraction of **b's skeleton that lies inside a**, Tsens the
    fraction of **a's skeleton that lies inside b**, and clDice their harmonic
    mean. The indexing direction matters and is easy to invert: indexing the
    skeleton by the mask (`sb[a]`) asks what fraction of the mask is skeleton,
    which is a thickness measure and is not 1.0 for a mask against itself. The
    identity check caught exactly that.
    """
    sa, sb = skeletonize(a), skeletonize(b)
    t_prec = float(a[sb].mean()) if sb.any() else 0.0
    t_sens = float(b[sa].mean()) if sa.any() else 0.0
    if (t_prec + t_sens) == 0:
        return 0.0
    return float(2.0 * t_prec * t_sens / (t_prec + t_sens))



def betti_numbers(mask: np.ndarray) -> tuple[int, int]:
    """(components, holes) for a 2-D binary mask, 4-connected foreground."""
    comps = int(label(mask, connectivity=1).max())
    holes = int(comps - euler_number(mask, connectivity=1))
    return comps, holes



def _ssim(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """SSIM with skimage's 7x7 window, shrunk (odd) to fit a small image; NaN below 3 pixels a side.

    The micro-Doppler spectrogram of a still sequence has only its motion
    frames as columns (HuPR 76: 5 frames of 600 above the pose-speed
    threshold), fewer than the window, and skimage refuses.
    """
    side = int(min(a.shape))
    window = min(7, side if side % 2 else side - 1)
    if window < 3:
        return float("nan")
    return float(structural_similarity(a, b, data_range=data_range, win_size=window))


def microdoppler_metrics(real: np.ndarray, rendered: np.ndarray,
                         p_fa: float = 1e-3, frame_rate_hz: float = 10.0) -> dict:
    """`S[d, t]`: amplitude spectrogram, Doppler bin by frame."""
    real, rendered = np.asarray(real, np.float64), np.asarray(rendered, np.float64)
    if real.shape != rendered.shape:
        raise ValueError(f"shape mismatch: {real.shape} vs {rendered.shape}")
    if real.shape[1] < 3:
        # a still person: fewer than three motion frames, nothing to correlate
        # over (HuPR 76: five; RT-Pose 157: one). Undefined, recorded as such.
        nan = float("nan")
        return {key: nan for key in ("centroid_r", "spread_r", "spectrogram_r", "energy_r", "env_plus_r",
                                     "env_minus_r", "bandwidth_ratio", "cl_dice", "cadence_error_hz",
                                     "d_betti_0", "d_betti_1", "spectrogram_ssim",
                                     "median_spread_bins_real", "median_spread_bins_rendered")} | {
            "p_fa": p_fa, "frames": int(real.shape[1]),
            "note": "fewer than three motion frames: the motion-subset metrics are undefined"}
    n_d = real.shape[0]
    bins = np.arange(n_d, dtype=np.float64) - n_d // 2      # signed from DC
    cr, sr = _moments(real, bins)
    cs, ss = _moments(rendered, bins)
    ok = np.isfinite(cr) & np.isfinite(cs)
    mr, ms = hysteresis_mask(real, p_fa), hysteresis_mask(rendered, p_fa)

    def envelope(S, mask, positive):
        out = np.full(S.shape[1], np.nan)
        for t in range(S.shape[1]):
            idx = np.flatnonzero(mask[:, t])
            if idx.size == 0:
                continue
            d = bins[idx]
            d = d[d > 0] if positive else d[d < 0]
            if d.size:
                out[t] = d.max() if positive else d.min()
        return out

    ep_r, ep_s = envelope(real, mr, True), envelope(rendered, ms, True)
    en_r, en_s = envelope(real, mr, False), envelope(rendered, ms, False)
    bw_r = np.nanmean(ep_r - en_r)
    bw_s = np.nanmean(ep_s - en_s)
    br0, br1 = betti_numbers(mr)
    bs0, bs1 = betti_numbers(rendered_mask := ms)
    rng = float(real.max())
    return {
        "centroid_r": pearson(cr[ok], cs[ok]),
        "spread_r": pearson(sr[ok], ss[ok]),
        "spectrogram_r": pearson(real, rendered),
        "energy_r": pearson((real ** 2).sum(axis=0), (rendered ** 2).sum(axis=0)),
        "env_plus_r": _pearson_finite(ep_r, ep_s),
        "env_minus_r": _pearson_finite(en_r, en_s),
        "bandwidth_ratio": float(bw_s / bw_r) if bw_r else float("nan"),
        "cl_dice": cl_dice(mr, ms),
        "cadence_error_hz": _cadence_error(real, rendered, frame_rate_hz),
        "d_betti_0": abs(bs0 - br0),
        "d_betti_1": abs(bs1 - br1),
        "spectrogram_ssim": _ssim(real, rendered, rng if rng > 0 else 1.0),
        # Reported as a PAIR, not only as a correlation: a high correlation
        # hides a systematically narrower render (the known under-dispersion).
        "median_spread_bins_real": float(np.nanmedian(sr)),
        "median_spread_bins_rendered": float(np.nanmedian(ss)),
        "p_fa": p_fa, "frames": int(real.shape[1]),
    }



def _pearson_finite(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return pearson(a[m], b[m]) if m.sum() > 2 else float("nan")



def _cadence_error(real, rendered, frame_rate_hz: float) -> float:
    """Peak of the mean cadence spectrum: FFT along time, averaged over bins."""
    def peak(S):
        spec = np.abs(np.fft.rfft(S - S.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
        if spec.size < 2:
            return float("nan")
        k = int(np.argmax(spec[1:]) + 1)
        return k * frame_rate_hz / S.shape[1]
    return float(abs(peak(rendered) - peak(real)))

# ----------------------------------------------------------------------
# persistence: Threshold-free topology: persistence diagrams and their Wasserstein distance.
# ----------------------------------------------------------------------

__all___persistence = ["persistence_diagram", "diagram_wasserstein_distance", "persistence_metrics"]


# Points closer to the diagonal than this are numerical noise, not features.
# It is not a detection threshold: it does not select features, it discards
# pairs whose birth and death are equal to floating-point precision.
_NOISE = 1e-12

# --------------------------------------------------------------------
# tables: tab:fidelity and the supplements
# --------------------------------------------------------------------

def _radars() -> tuple[str, ...]:
    """The current dataset's radar names, resolved when a command runs."""
    return current_dataset().names

#: (paper symbol, family, key, direction). `ratio` means 1.00 is exact and
#: either side is wrong; it is NOT a "higher is better" column.
MAIN_COLUMNS = (
    (r"$\rho$", "range_azimuth", "pearson_mean", "up"),
    ("SSIM", "range_azimuth", "ssim_mean", "up"),
    ("NRMSE", "range_azimuth", "nrmse_mean", "down"),
    ("PSNR", "range_azimuth", "psnr_mean", "up"),
    (r"$\rho_M$", "range_doppler", "occupied_pearson_mean", "up"),
    (r"$P_M$", "range_doppler", "signal_power_ratio", "ratio"),
    (r"$L_U$", "range_doppler", "floor_level_ratio", "ratio"),
    (r"$\sigma_U$", "range_doppler", "floor_sigma_ratio", "ratio"),
    (r"$\rho_c$", "microdoppler", "centroid_r", "up"),
    (r"$\rho_\sigma$", "microdoppler", "spread_r", "up"),
    (r"$\rho_S$", "microdoppler", "spectrogram_r", "up"),
    (r"$\rho_{\mathrm{env}}$", "microdoppler", "env_plus_r", "up"),
    ("BW", "microdoppler", "bandwidth_ratio", "ratio"),
    ("clDice", "microdoppler", "cl_dice", "up"),
)

#: The supplement tables, in the order they appear.
SUPPLEMENT = (
    ("Noise floor beyond level and texture",
     ((r"$L_U$", "range_doppler", "floor_level_ratio"),
      (r"$W_1$", "range_doppler", "floor_wasserstein1"),
      (r"$r_1$ real", "range_doppler", "floor_lag1_real"),
      (r"$r_1$ rend.", "range_doppler", "floor_lag1_rendered"))),
    ("Micro-Doppler structure and periodicity",
     ((r"$\Delta f_c$ (Hz)", "microdoppler", "cadence_error_hz"),
      (r"$\Delta\beta_0$", "microdoppler", "d_betti_0"),
      (r"$\Delta\beta_1$", "microdoppler", "d_betti_1"),
      (r"SSIM$_S$", "microdoppler", "spectrogram_ssim"))),
    ("Micro-Doppler metrics demoted from the main table",
     ((r"$\rho_E$", "microdoppler", "energy_r"),
      (r"$\rho_{\mathrm{env}}^{+}$", "microdoppler", "env_plus_r"),
      (r"$\rho_{\mathrm{env}}^{-}$", "microdoppler", "env_minus_r"))),
)


def _value(record: dict, radar: str, family: str, key: str) -> str:
    try:
        value = record["radars"][radar][family][key]
    except (KeyError, TypeError):
        return "--"
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def markdown(records: list[dict], label: str = "4DPS (Ours)") -> str:
    """The fidelity table plus the supplement tables, as Markdown."""
    lines: list[str] = ["# Rendering fidelity", "",
                        "Linear magnitude throughout. No dB anywhere except PSNR, "
                        "which is $20\\log_{10}$ of a scalar ratio.",
                        "",
                        "| seq | radar | " + " | ".join(c[0] for c in MAIN_COLUMNS) + " |",
                        "|---|---|" + "---:|" * len(MAIN_COLUMNS)]
    for record in records:
        for radar in _radars():
            if radar not in record.get("radars", {}):
                continue
            cells = [_value(record, radar, family, key)
                     for _, family, key, _ in MAIN_COLUMNS]
            lines.append(f"| {record.get('sequence', '?')} | {radar[0].upper()} | "
                         + " | ".join(cells) + " |")
    for title, columns in SUPPLEMENT:
        lines += ["", f"## {title}", "",
                  "| seq | radar | " + " | ".join(c[0] for c in columns) + " |",
                  "|---|---|" + "---:|" * len(columns)]
        for record in records:
            for radar in _radars():
                if radar not in record.get("radars", {}):
                    continue
                cells = [_value(record, radar, family, key)
                         for _, family, key in columns]
                lines.append(f"| {record.get('sequence', '?')} | {radar[0].upper()} | "
                             + " | ".join(cells) + " |")
    lines += ["", f"Method row: {label}. Frames scored per sequence: "
              + ", ".join(f"seq {r.get('sequence','?')}={r.get('frames_scored','?')}"
                          for r in records) + "."]
    return "\n".join(lines) + "\n"


def latex(records: list[dict], label: str = r"\textbf{4DPS (Ours)}") -> str:
    """The main table's BODY ROWS only, as an \\input{}-able fragment.

    Deliberately not a whole table environment: the paper owns the caption, the
    column spec and the rules, and this replaces only the row that was typed by
    hand. Nothing here can silently change the table's shape.
    """
    lines = [r"% generated by mm4DPS.evaluation.tables -- do not edit",
             r"% one row per sequence and radar; columns follow tab:fidelity"]
    for record in records:
        for radar in _radars():
            if radar not in record.get("radars", {}):
                continue
            cells = [_value(record, radar, family, key)
                     for _, family, key, _ in MAIN_COLUMNS]
            name = f"{label} seq {record.get('sequence','?')} ({radar[0].upper()})"
            lines.append(name + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def load(directory: Path, dataset: str) -> list[dict]:
    """The per-sequence JSONs the suite wrote, in sequence order."""
    records = []
    for path in sorted(directory.glob(f"{dataset}_seq*.json")):
        record = json.loads(path.read_text())
        record.setdefault("sequence",
                          "".join(c for c in path.stem.split("seq")[-1] if c.isdigit()))
        records.append(record)
    return records


def write(directory: Path, dataset: str, emit: str = "both") -> list[Path]:
    """Render whichever formats were asked for, beside the JSON."""
    records = load(directory, dataset)
    if not records:
        return []
    written = []
    if emit in ("md", "both"):
        target = directory / f"{dataset}_fidelity.md"
        target.write_text(markdown(records))
        written.append(target)
    if emit in ("tex", "both"):
        target = directory / f"{dataset}_fidelity.tex"
        target.write_text(latex(records))
        written.append(target)
    return written

# --------------------------------------------------------------------
# evaluate: one workspace, end to end
# --------------------------------------------------------------------

SPEC_VERSION = "1.0.0"
DEFAULT_P_FA = 1e-3


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(mm4DPS.REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:                       # noqa: BLE001
        return f"unavailable: {exc}"



def _frame_count(product_dir: Path, spectrogram_dir: Path | None) -> int:
    """Frames available, from whichever source the spectrogram will come from."""
    if spectrogram_dir is not None:
        p = spectrogram_dir / f"{_radars()[0]}_measured_microdoppler_power.npy"
        if p.exists():
            return int(np.load(p, mmap_mode="r").shape[1])
    p = product_dir / f"{_radars()[0]}_measured_first_loop_ra_complex.npy"
    return int(np.load(p, mmap_mode="r").shape[0])


def _load(product_dir: Path, radar: str, side: str, kind: str) -> np.ndarray:
    path = product_dir / f"{radar}_{side}_{kind}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing product {path}")
    a = np.load(path)
    if np.iscomplexobj(a):
        a = np.abs(a)                              # magnitude, then amplitude below
        return a.astype(np.float64)
    return amplitude(a)


def _spectrogram(product_dir: Path, radar: str, side: str,
                 spectrogram_dir: Path | None = None,
                 allow_ungated: bool = False) -> tuple[np.ndarray, str]:
    """(doppler, frames) amplitude, and the name of the reduction used.

    **The two available reductions are not interchangeable and one of them is
    wrong.** The pipeline's spectrogram stage keeps a 21-bin Hann-weighted
    range window centred on the person's own p25 range, so it measures the
    person. Summing `motion_dra` over range and angle keeps everything --
    the clutter, the room, any other scatterer -- and calls it micro-Doppler.

    Measured 2026-09-03 on the shipped products: the two disagree by up to
    0.2567 in centroid correlation and 0.1669 in spread, against a measured
    reproducibility floor of 0.009-0.073 (`docs/DETERMINISM_FLOOR.md`). Ten of
    sixteen metrics moved by more than the floor. A silent fallback between
    them means two invocations of the "single entry point" report different
    numbers for the same product, which is what this suite exists to prevent.

    So the ungated path is opt-in, never the silent default, and the reduction
    actually used is returned so it can be stamped into the JSON.

    The suite never re-renders to recover a deleted product: that would hide
    the cost and let the scored products drift from the shipped ones.
    """
    if spectrogram_dir is not None:
        path = spectrogram_dir / f"{radar}_{side}_microdoppler_power.npy"
        if path.exists():
            return amplitude(np.load(path)), "person-range-gated"
    cube = product_dir / f"{radar}_{side}_motion_dra.npy"
    if not cube.exists():
        raise FileNotFoundError(
            f"neither {spectrogram_dir}/{radar}_{side}_microdoppler_power.npy "
            f"nor {cube} exists. The chain was run with KEEP_EVAL_PRODUCTS=0, "
            f"which deletes motion_dra. Re-run that render with "
            f"KEEP_EVAL_PRODUCTS=1; the suite does not re-render for you.")
    if not allow_ungated:
        raise SystemExit(
            f"refusing to sum {cube.name} over range and angle to stand in for "
            f"a micro-Doppler spectrogram. That reduction is ungated -- it "
            f"keeps the clutter and the room, where the pipeline's spectrogram "
            f"keeps a 21-bin Hann window at the person's range -- and the two "
            f"disagree by up to 0.26 in centroid correlation, far above the "
            f"reproducibility floor. Pass --spectrogram-dir-template to score "
            f"the reduction the pipeline and the paper use, building it first "
            f"with `python -m mm4DPS.evaluation.media` if "
            f"the product has none. --allow-ungated-spectrogram overrides this "
            f"deliberately and stamps the JSON so the number is identifiable.")
    power = np.asarray(np.load(cube), dtype=np.float64) ** 2
    return amplitude(power.sum(axis=(2, 3))).T, "ungated-motion-dra-sum"


def evaluate_sequence(product_dir: Path, sequence: str, p_fa: float,
                      frame_limit: int | None,
                      spectrogram_dir: Path | None = None,
                      sequence_dir: Path | None = None,
                      frame_selection: str = "pose-speed",
                      allow_ungated: bool = False) -> dict:
    # The frame subset, decided ONCE here and applied to every metric, so the
    # rule reported and the frames scored cannot disagree. An entry point that
    # says 600 while the scorer used 86 is the exact class of bug the
    # specification requires to fail loudly.
    if frame_selection == "pose-speed":
        if sequence_dir is None:
            raise ValueError(
                "--frame-selection pose-speed needs --sequence-dir-template; "
                "the motion subset is derived from the person cloud. Use "
                "--frame-selection all to score every frame deliberately.")
        n_total = _frame_count(product_dir, spectrogram_dir)
        selected = pose_speed_frames(sequence_dir, n_total)
        rule = (f"pose speed >= {POSE_SPEED_THRESHOLD_M_S} m/s, "
                f"{selected.size}/{n_total} frames "
                f"(pre-registered; docs/PREREGISTERED_CRITERIA.md)")
    else:
        selected = None
        rule = "every frame, requested explicitly with --frame-selection all"
    out: dict = {
        "spec_version": SPEC_VERSION,
        "git_sha": git_sha(),
        "sequence": sequence,
        "product_dir": str(product_dir),
        "cfar_p_fa": p_fa,
        "frame_selection": frame_selection,
        "frame_selection_rule": rule,
        "radars": {},
    }
    reductions: set[str] = set()
    for radar in _radars():
        ra_real = _load(product_dir, radar, "measured", "first_loop_ra_complex")
        ra_rend = _load(product_dir, radar, "rendered", "first_loop_ra_complex")
        rd_real = _load(product_dir, radar, "measured", "rd")
        rd_rend = _load(product_dir, radar, "rendered", "rd")
        md_real, red_a = _spectrogram(product_dir, radar, "measured",
                                      spectrogram_dir, allow_ungated)
        md_rend, red_b = _spectrogram(product_dir, radar, "rendered",
                                      spectrogram_dir, allow_ungated)
        if red_a != red_b:
            raise ValueError(f"measured and rendered spectrograms used "
                             f"different reductions: {red_a} vs {red_b}")
        reductions.add(red_a)
        if frame_limit is not None:
            ra_real, ra_rend = ra_real[:frame_limit], ra_rend[:frame_limit]
            rd_real, rd_rend = rd_real[:frame_limit], rd_rend[:frame_limit]
            md_real, md_rend = md_real[:, :frame_limit], md_rend[:, :frame_limit]
        if selected is not None:
            ra_idx = selected[selected < ra_real.shape[0]]
            md_idx = selected[selected < md_real.shape[1]]
            ra_real, ra_rend = ra_real[ra_idx], ra_rend[ra_idx]
            rd_real, rd_rend = rd_real[ra_idx], rd_rend[ra_idx]
            md_real, md_rend = md_real[:, md_idx], md_rend[:, md_idx]
        out["radars"][radar] = {
            "range_azimuth": range_azimuth_metrics(ra_real, ra_rend),
            # The discriminative range-azimuth figure. The plain one above
            # fails the permutation check and is retained as a sanity number.
            "range_azimuth_suppressed":
                range_azimuth_suppressed_metrics(ra_real, ra_rend, p_fa),
            "range_doppler": range_doppler_metrics(rd_real, rd_rend, p_fa),
            "microdoppler": microdoppler_metrics(md_real, md_rend, p_fa),
            "frames_scored": int(ra_real.shape[0]),
        }
        # The same loop in the downstream consumers' view -- no azimuth taper, as HuPR's released preprocessing and
        # the HRRadarPose adapter beamform (docs/STATIC_SEQS_NEXT.md 1d: a rendered wall reads 1x tapered and 10x
        # untapered). Scored beside the pipeline's view when the renderer wrote the products (video.py).
        if (product_dir / f"{radar}_measured_first_loop_ra_untapered_complex.npy").exists():
            rau_real = _load(product_dir, radar, "measured", "first_loop_ra_untapered_complex")
            rau_rend = _load(product_dir, radar, "rendered", "first_loop_ra_untapered_complex")
            if frame_limit is not None:
                rau_real, rau_rend = rau_real[:frame_limit], rau_rend[:frame_limit]
            if selected is not None:
                rau_real, rau_rend = rau_real[ra_idx], rau_rend[ra_idx]
            out["radars"][radar]["range_azimuth_untapered"] = range_azimuth_metrics(rau_real, rau_rend)
            out["radars"][radar]["range_azimuth_suppressed_untapered"] = range_azimuth_suppressed_metrics(rau_real, rau_rend, p_fa)
        # The complex views (the first loop's per-channel range profiles, both sides): range-azimuth-elevation, and
        # the 3DPS complex-range-profile / ADC protocol. Same frames as every figure above.
        if (product_dir / f"{radar}_measured_first_loop_rp_complex.npy").exists():
            rp_real = np.load(product_dir / f"{radar}_measured_first_loop_rp_complex.npy", mmap_mode="r")
            rp_rend = np.load(product_dir / f"{radar}_rendered_first_loop_rp_complex.npy", mmap_mode="r")
            if frame_limit is not None:
                rp_real, rp_rend = rp_real[:frame_limit], rp_rend[:frame_limit]
            if selected is not None:
                rp_real, rp_rend = rp_real[ra_idx], rp_rend[ra_idx]
            rp_real, rp_rend = np.asarray(rp_real), np.asarray(rp_rend)
            radar_layout = current_dataset().radar(radar)
            try:
                start_bin = int(json.loads((product_dir / "render_result.json").read_text())["first_retained_range_bin"])
            except Exception:
                start_bin = 0
            out["radars"][radar]["range_azimuth_elevation"] = range_azimuth_elevation_metrics(rp_real, rp_rend, radar_layout)
            out["radars"][radar]["complex_range_profile_adc"] = complex_profile_metrics(rp_real, rp_rend, radar_layout, start_bin)
    scored = {r: out["radars"][r]["frames_scored"] for r in _radars()}
    if len(set(scored.values())) != 1:
        raise ValueError(f"radars scored different frame counts: {scored}")
    out["frames_scored"] = scored[_radars()[0]]
    # Stamped: the reduction changes the numbers by more than the
    # reproducibility floor, so a reader must be able to tell which they hold.
    out["spectrogram_reduction"] = (sorted(reductions)[0]
                                    if len(reductions) == 1 else sorted(reductions))
    return out


def aggregate(per_sequence: list[dict]) -> dict:
    """Per dataset. Still never averages the two radars."""
    agg: dict = {"spec_version": SPEC_VERSION, "git_sha": git_sha(),
                 "sequences": [d["sequence"] for d in per_sequence],
                 "cfar_p_fa": per_sequence[0]["cfar_p_fa"] if per_sequence else None,
                 "radars": {}}
    for radar in _radars():
        block: dict = {}
        for family in ("range_azimuth", "range_doppler", "microdoppler"):
            keys = per_sequence[0]["radars"][radar][family].keys()
            block[family] = {}
            for k in keys:
                vals = [d["radars"][radar][family][k] for d in per_sequence]
                if all(isinstance(v, (int, float)) and np.isfinite(v) for v in vals):
                    block[family][k] = {"mean": float(np.mean(vals)),
                                        "std": float(np.std(vals)),
                                        "per_sequence": [float(v) for v in vals]}
                else:
                    block[family][k] = {"per_sequence": vals}
        # the scatter the spec asks for: MD centroid r against motion energy
        block["centroid_r_vs_motion_energy"] = [
            {"sequence": d["sequence"],
             "centroid_r": d["radars"][radar]["microdoppler"]["centroid_r"],
             "motion_energy": d["radars"][radar]["range_doppler"]["signal_power_ratio"]}
            for d in per_sequence]
        agg["radars"][radar] = block
    return agg


def _entry_evaluate() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, required=True,
                    help="a fitted 4DPS workspace, or a directory of them")
    ap.add_argument("--dataset", choices=("hupr", "rtpose"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--product-glob", default="rendered/seq*_final_600",
                    help="product directories to score, relative to the workspace")
    ap.add_argument("--p-fa", type=float, default=DEFAULT_P_FA,
                    help="CFAR false-alarm rate. Chosen once from the noise "
                         "statistics and swept for sensitivity, never selected "
                         "to improve a result.")
    ap.add_argument("--frame-limit", type=int, default=None)
    ap.add_argument("--allow-ungated-spectrogram", action="store_true",
                    help="sum motion_dra over range and angle when no built "
                         "spectrogram exists. Off by default: that reduction "
                         "is ungated and disagrees with the pipeline's by "
                         "more than the reproducibility floor. The choice is "
                         "stamped into every JSON as spectrogram_reduction.")
    ap.add_argument("--frame-selection", choices=("pose-speed", "all"),
                    default="pose-speed")
    ap.add_argument("--sequence-dir-template", default=None,
                    help="e.g. outputs/vf_scratch_{seq}/person_sequence_{seq}_v2clean")
    ap.add_argument("--spectrogram-dir-template", default=None,
                    help="e.g. outputs/vf_scratch_{seq}/deliverables/"
                         "microdoppler_seq{seq}/hw10_v2 -- reuses the "
                         "pipeline's own spectrogram instead of motion_dra")
    ap.add_argument("--emit", choices=("none", "md", "tex", "both"),
                    default="both",
                    help="also render the paper's fidelity table beside the "
                         "JSON. The numbers used to be transcribed by hand, "
                         "which was the one step in this pipeline with no gate "
                         "on it.")
    args = ap.parse_args()
    Dataset.of(args)

    dirs = sorted(args.workspace.glob(args.product_glob))
    if not dirs:
        raise SystemExit(f"no product directories matched "
                         f"{args.workspace / args.product_glob}")
    args.out.mkdir(parents=True, exist_ok=True)
    per_sequence = []
    for d in dirs:
        seq = "".join(c for c in d.name.split("_")[0] if c.isdigit()) or d.name
        spec = (args.spectrogram_dir_template.format(seq=seq)
                if args.spectrogram_dir_template else None)
        sd = (args.sequence_dir_template.format(seq=seq)
              if args.sequence_dir_template else None)
        result = evaluate_sequence(d, seq, args.p_fa, args.frame_limit,
                                   Path(spec) if spec else None,
                                   Path(sd) if sd else None,
                                   args.frame_selection,
                                   args.allow_ungated_spectrogram)
        result["dataset"] = args.dataset
        (args.out / f"{args.dataset}_seq{seq}.json").write_text(
            json.dumps(result, indent=2))
        per_sequence.append(result)
        print(f"  seq {seq}: frames_scored={result['frames_scored']}  "
              f"({result['frame_selection_rule']})")
    agg = aggregate(per_sequence)
    agg["dataset"] = args.dataset
    (args.out / f"{args.dataset}_aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"wrote {len(per_sequence)} sequence JSONs and one aggregate to {args.out}")
    if args.emit != "none":
        for path in write(args.out, args.dataset, args.emit):
            print(f"wrote {path}")
    return 0

# --------------------------------------------------------------------
# scoring: the micro-Doppler score and the null
# --------------------------------------------------------------------

def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or a[good].std() < 1e-12 or b[good].std() < 1e-12:
        return float("nan")
    # `metrics.pearson`, not `np.corrcoef`. The specification requires the
    # identity check to return EXACTLY 1.0, and corrcoef misses it by one ULP:
    # scoring 200 random products against themselves left 199 of 800 fields at
    # 1 - 2.22e-16 instead of 1.0. Nothing downstream moves by that much, but a
    # check written as "== 1.0" either passes or it does not.
    return pearson(a[good], b[good])



def centroid_and_spread(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power-weighted mean and standard deviation over the Doppler axis.

    Bins are measured from the DC bin, so the centroid is signed and its sign is
    the direction of travel. ``power`` is (doppler_bins, frames).
    """
    bins, _ = power.shape
    velocity_bin = np.arange(bins, dtype=np.float64) - bins // 2
    weight = np.maximum(power, 0.0)
    total = weight.sum(axis=0)
    safe = np.where(total > 0, total, np.nan)
    centroid = (weight * velocity_bin[:, None]).sum(axis=0) / safe
    variance = (weight * (velocity_bin[:, None] - centroid[None, :]) ** 2).sum(axis=0) / safe
    return centroid, np.sqrt(variance)



def score(
    measured: np.ndarray, rendered: np.ndarray, frames: np.ndarray | None = None,
) -> dict:
    if measured.shape != rendered.shape:
        raise ValueError(f"shape mismatch: {measured.shape} vs {rendered.shape}")
    if frames is None:
        frames = np.arange(measured.shape[1])
    frames = np.asarray(frames, dtype=int)
    if len(frames) < 3:
        # A still person (RT-Pose 157: one frame of 299 above the pose-speed
        # threshold) has no motion subset to correlate over. The product is
        # still delivered; the motion-subset figures are undefined, not
        # wrong, and say so.
        nan = float("nan")
        return {"frames_scored": int(len(frames)),
                "centroid_correlation": nan, "spread_correlation": nan,
                "spectrogram_correlation": nan, "energy_correlation": nan,
                "measured_spread_median_bins": nan, "rendered_spread_median_bins": nan,
                "note": "fewer than three motion frames: the motion-subset metrics are undefined"}

    measured_centroid, measured_spread = centroid_and_spread(measured)
    rendered_centroid, rendered_spread = centroid_and_spread(rendered)

    return {
        "frames_scored": int(len(frames)),
        "centroid_correlation": _correlate(measured_centroid[frames], rendered_centroid[frames]),
        "spread_correlation": _correlate(measured_spread[frames], rendered_spread[frames]),
        # Amplitude, sqrt(max(power, 0)), for this one field. Every metric in
        # this project is on linear amplitude and a power-domain correlation is
        # a bug: on seq 38 horizontal the whole-array Pearson reads +0.8149 in
        # power and +0.9078 in amplitude.
        #
        # It is ONLY this field. The centroid, spread and energy figures below
        # are power-WEIGHTED quantities, where the weighting is part of the
        # definition rather than the domain of a comparison -- `mm4DPS.evaluate`
        # holds an amplitude array S and forms exactly these with `w = S**2`.
        # Taking sqrt before them instead of inside them changes the weights
        # from power to amplitude and moves the median spread on seq 38
        # horizontal from 7.05/5.10 bins to 13.92/12.35.
        "spectrogram_correlation": _correlate(
            np.sqrt(np.maximum(measured[:, frames], 0.0)),
            np.sqrt(np.maximum(rendered[:, frames], 0.0)),
        ),
        "energy_correlation": _correlate(
            measured[:, frames].sum(axis=0), rendered[:, frames].sum(axis=0),
        ),
        "measured_spread_median_bins": float(np.nanmedian(measured_spread[frames])),
        "rendered_spread_median_bins": float(np.nanmedian(rendered_spread[frames])),
    }



#: Pre-registered in docs/PREREGISTERED_CRITERIA.md, chosen once from the pooled
#: speed distribution of sequences 38/102/256 before any score was recomputed.
#: NOT to be tuned to improve a result.
POSE_SPEED_THRESHOLD_M_S = 0.15



def pose_speed(sequence_dir: Path, frame_count: int,
               frame_rate_hz: float = 10.0) -> np.ndarray:
    """Per-frame speed of the person-cloud centroid, m/s. Camera-derived only."""
    frames_dir = Path(sequence_dir) / "person_frames"
    centroids = []
    for frame in range(frame_count):
        path = frames_dir / f"frame_{frame:03d}.npz"
        if not path.exists():
            break
        centroids.append(np.median(
            np.load(path)["pcl"][:, :3].astype(np.float64), axis=0))
    if len(centroids) < 2:
        raise FileNotFoundError(
            f"need at least two person frames under {frames_dir} to measure "
            f"pose speed; found {len(centroids)}")
    c = np.asarray(centroids)
    speed = np.linalg.norm(np.diff(c, axis=0), axis=1) * frame_rate_hz
    return np.concatenate([speed[:1], speed])          # frame 0 takes frame 1's



def pose_speed_frames(sequence_dir: Path, frame_count: int,
                      threshold_m_s: float = POSE_SPEED_THRESHOLD_M_S) -> np.ndarray:
    """The motion subset: frames whose pose speed is at or above the threshold.

    This replaces the detector-derived subset. `detect_person_motion` rejects
    on a detection confidence and a range residual, neither of which is a
    motion test, and it accepts 0 of 600 frames on all three sequences -- its
    observed range sits ~1.06 m beyond the person cloud on every frame and its
    two co-located radars disagree by up to 0.84 m. See
    docs/FROM_SCRATCH_ATTEMPT.md for the diagnosis.

    Camera-derived, deterministic, no learned segmenter, and the threshold is
    pre-registered rather than fitted.
    """
    speed = pose_speed(sequence_dir, frame_count)
    frames = np.flatnonzero(speed >= threshold_m_s).astype(int)
    if frames.size == 0:
        raise ValueError(
            f"no frame in {sequence_dir} reaches {threshold_m_s} m/s "
            f"(max {speed.max():.4f}); this sequence has no motion subset")
    if frames.size == frame_count:
        # A person who walks for the whole recording qualifies every frame
        # (RT-Pose seq 153: 98-100% of frames above the threshold); the
        # selection is then the whole sequence, and the rule recorded with
        # the score says so. It was an error before, which stopped a chain
        # on a sequence with no still frame.
        print(f"pose-speed selection: every frame in {sequence_dir} reaches {threshold_m_s} m/s; "
              f"the motion subset is the whole sequence", flush=True)
    return frames



def motion_frames(pose_audit: Path | None, frame_count: int) -> np.ndarray:
    """Frames the pose audit marked as motion.

    **Fails loudly when a motion subset was requested and cannot be built.**
    This used to return every frame in that case, twice over: once when the
    audit file was missing and once when it carried no runs. Both look like a
    successful score with `frames_scored == frame_count`, which is why every
    score in the tree said 600 -- the number reported was the whole sequence,
    not the moving part of it, and nothing said so.

    Passing `pose_audit=None` is the honest way to score every frame.
    """
    if pose_audit is None:
        return np.arange(frame_count)          # no subset requested
    if not pose_audit.is_file():
        raise FileNotFoundError(
            f"pose audit {pose_audit} does not exist. A motion subset was "
            f"requested; scoring every frame instead would report "
            f"frames_scored={frame_count} for a number that is not the motion "
            f"subset. Pass no --pose-audit to score every frame deliberately.")
    audit = json.loads(pose_audit.read_text())
    runs = audit.get("motion_runs") or audit.get("runs") or []
    keep: list[int] = []
    for run in runs:
        start, end = (run["start"], run["end"]) if isinstance(run, dict) else run[:2]
        keep.extend(range(max(0, int(start)), min(frame_count, int(end) + 1)))
    if not keep:
        raise ValueError(
            f"pose audit {pose_audit} has no motion runs (looked for "
            f"'motion_runs' and 'runs'), so no motion subset can be built. "
            f"Scoring every frame here is what made every score in the tree "
            f"report frames_scored={frame_count}. Pass no --pose-audit to "
            f"score every frame deliberately.")
    frames = np.asarray(sorted(set(keep)), dtype=int)
    if frames.size == frame_count:
        raise ValueError(
            f"pose audit {pose_audit} marks all {frame_count} frames as "
            f"motion. That is indistinguishable from the silent fallback this "
            f"check replaces; if it is genuinely correct, score without "
            f"--pose-audit and say so.")
    return frames



def select_frames(args: argparse.Namespace, frame_count: int) -> tuple[np.ndarray, str]:
    """The frame subset and the rule that produced it, recorded together."""
    if args.frame_selection == "all":
        return np.arange(frame_count), "every frame, requested explicitly"
    if args.frame_selection == "audit":
        return (motion_frames(args.pose_audit, frame_count),
                f"pose-audit motion runs ({args.pose_audit})")
    if args.sequence_dir is None:
        # Without this the None reaches pathlib and raises a TypeError forty
        # lines away, which is what happened: the chain calls this with
        # --pose-audit and no --sequence-dir, the default selection is
        # pose-speed, and every cold repeat this session wrote no score.json
        # at all while its driver reported success.
        raise SystemExit(
            "--frame-selection pose-speed needs --sequence-dir: the motion "
            "subset is derived from the person cloud, not from the pose audit. "
            "Pass --sequence-dir <...>_v2clean, or choose --frame-selection "
            "audit or all deliberately.")
    frames = pose_speed_frames(args.sequence_dir, frame_count,
                               args.pose_speed_threshold)
    return frames, (f"pose speed >= {args.pose_speed_threshold} m/s "
                    f"(pre-registered; docs/PREREGISTERED_CRITERIA.md)")



def run(args: argparse.Namespace) -> dict:
    result: dict = {"spectrogram_dir": str(args.spectrogram_dir), "radars": {}}
    for radar in _radars():
        measured = np.load(args.spectrogram_dir / f"{radar}_measured_microdoppler_power.npy")
        rendered = np.load(args.spectrogram_dir / f"{radar}_rendered_microdoppler_power.npy")
        frames, rule = select_frames(args, measured.shape[1])
        result["frame_selection_rule"] = rule
        result["radars"][radar] = score(measured, rendered, frames)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    return result



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrogram-dir", type=Path, required=True)
    parser.add_argument("--pose-audit", type=Path,
                        help="restrict the correlations to the motion runs")
    parser.add_argument(
        "--frame-selection", choices=("pose-speed", "audit", "all"),
        default="pose-speed",
        help="pose-speed (default): frames whose person-cloud centroid moves at "
             "or above --pose-speed-threshold. audit: the pose audit's motion "
             "runs, which the detector does not currently produce (it accepts "
             "0/600; see docs/FROM_SCRATCH_ATTEMPT.md). all: every frame, said "
             "out loud rather than reached by a silent fallback.")
    parser.add_argument("--sequence-dir", type=Path,
                        help="person sequence directory, for pose-speed selection")
    parser.add_argument("--pose-speed-threshold", type=float,
                        default=POSE_SPEED_THRESHOLD_M_S,
                        help="pre-registered in docs/PREREGISTERED_CRITERIA.md. "
                             "Not to be tuned to improve a result.")
    parser.add_argument("--output", type=Path)
    return parser


def _entry_score_microdoppler():
        parsed = make_parser().parse_args()
        outcome = run(parsed)
        for name, values in outcome["radars"].items():
            print(f"{name:11s} centroid {values['centroid_correlation']:+.3f}  "
                  f"spread {values['spread_correlation']:+.3f}  "
                  f"spectrogram {values['spectrogram_correlation']:+.3f}  "
                  f"energy {values['energy_correlation']:+.3f}  "
                  f"({values['frames_scored']} frames)")

# ----------------------------------------------------------------------
# make_tables: Generate the paper's LaTeX tables from the evaluation JSONs.
# ----------------------------------------------------------------------

# (row label, JSON path under radars.<radar>, format)
RANGE_AZIMUTH = [
    # The discriminative rows first: these are the ones that pass the
    # permutation check. The plain rows below are sanity figures and the
    # caption says so.
    ("Clutter-removed Pearson",
     "range_azimuth_suppressed.clutter_removed_pearson_mean", "{:.3f}"),
    ("Occupied clutter-removed Pearson",
     "range_azimuth_suppressed.occupied_clutter_removed_pearson_mean", "{:.3f}"),
    ("Occupancy fraction",
     "range_azimuth_suppressed.occupancy_fraction", "{:.3f}"),
    ("Pearson (sanity)",       "range_azimuth.pearson_mean",  "{:.3f}"),
    ("Pearson std (sanity)",   "range_azimuth.pearson_std",   "{:.3f}"),
    ("SSIM (sanity)",          "range_azimuth.ssim_mean",     "{:.3f}"),
    ("NRMSE (sanity)",         "range_azimuth.nrmse_mean",    "{:.3f}"),
    ("PSNR dB (sanity)",       "range_azimuth.psnr_mean",     "{:.1f}"),
]

RANGE_DOPPLER = [
    ("Occupied Pearson",   "range_doppler.occupied_pearson_mean", "{:.3f}"),
    ("Signal power ratio", "range_doppler.signal_power_ratio",    "{:.3f}"),
    ("Floor level ratio",  "range_doppler.floor_level_ratio",     "{:.3f}"),
    ("Floor sigma ratio",  "range_doppler.floor_sigma_ratio",     "{:.3f}"),
    ("SCNR ratio",         "range_doppler.scnr_ratio",            "{:.3f}"),
]

MICRODOPPLER = [
    ("Centroid $r$",       "microdoppler.centroid_r",     "{:.3f}"),
    ("Spread $r$",         "microdoppler.spread_r",       "{:.3f}"),
    ("Spectrogram $r$",    "microdoppler.spectrogram_r",  "{:.3f}"),
    ("Energy $r$",         "microdoppler.energy_r",       "{:.3f}"),
    ("Env$+$ $r$",         "microdoppler.env_plus_r",     "{:.3f}"),
    ("Env$-$ $r$",         "microdoppler.env_minus_r",    "{:.3f}"),
    ("BW ratio",           "microdoppler.bandwidth_ratio","{:.3f}"),
    ("clDice",             "microdoppler.cl_dice",        "{:.3f}"),
    # dBetti-0 and dBetti-1 are NOT here. They do not separate matched from
    # permuted pairs at all -- a permuted pair scores a perfect 0 where the
    # matched one scores 8 -- and the persistence diagram distance the spec
    # prescribed as their replacement does not separate either. See README,
    # "Removed 2026-09-03". clDice passes both the sweep and the permutation
    # and is the topology metric this project reports.
]

# The weak numbers. Printed, not footnoted.
WEAK = [
    ("Median spread, real (bins)",     "microdoppler.median_spread_bins_real",     "{:.2f}"),
    ("Median spread, rendered (bins)", "microdoppler.median_spread_bins_rendered", "{:.2f}"),
    ("Floor level ratio",              "range_doppler.floor_level_ratio",          "{:.3f}"),
    ("Floor sigma ratio",              "range_doppler.floor_sigma_ratio",          "{:.3f}"),
]



def dig(block: dict, path: str):
    for part in path.split("."):
        block = block[part]
    return block



def table(rows, per_seq: list[dict], caption: str, label: str) -> str:
    seqs = [d["sequence"] for d in per_seq]
    cols = "l" + "rr" * len(seqs)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{seq {s}}}" for s in seqs)
    sub = " & ".join(["H & V"] * len(seqs))
    out = [f"% generated by mm4DPS.evaluation.make_tables -- do not edit",
           f"\\begin{{table}}[t]\\centering",
           f"\\begin{{tabular}}{{{cols}}}\\toprule",
           f"Metric (JSON field) & {head} \\\\",
           f" & {sub} \\\\\\midrule"]
    for lab, path, fmt in rows:
        cells = []
        for d in per_seq:
            for radar in _radars():
                v = dig(d["radars"][radar], path)
                cells.append(fmt.format(v) if isinstance(v, (int, float)) else "--")
        # Underscores are subscript operators in LaTeX text mode, so a raw
        # field name inside \texttt{} does not compile. Every JSON field here
        # contains at least one.
        shown = path.replace("_", r"\_")
        out.append(f"{lab} {{\\scriptsize\\texttt{{{shown}}}}} & "
                   + " & ".join(cells) + " \\\\")
    out += [f"\\bottomrule\\end{{tabular}}",
            f"\\caption{{{caption}}}\\label{{{label}}}",
            f"\\end{{table}}"]
    return "\n".join(out)



def main_make_tables() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--dataset", default="hupr")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    Dataset.of(args)

    files = sorted(p for p in args.eval_dir.glob(f"{args.dataset}_seq*.json"))
    if not files:
        raise SystemExit(f"no {args.dataset}_seq*.json in {args.eval_dir}")
    per_seq = [json.loads(p.read_text()) for p in files]
    shas = {d["git_sha"] for d in per_seq}
    pfa = {d["cfar_p_fa"] for d in per_seq}
    if len(shas) != 1 or len(pfa) != 1:
        raise SystemExit(f"mixed provenance: git_sha={shas} p_fa={pfa}")
    args.out.mkdir(parents=True, exist_ok=True)

    specs = [
        ("range_azimuth", RANGE_AZIMUTH, "Range-azimuth agreement, per radar. "
         f"Linear amplitude; PSNR is the sole logarithm. CFAR $P_{{fa}}$="
         f"{list(pfa)[0]:g}. \\textbf{{These are sanity figures, not evidence "
         f"of a correct render.}} Range-azimuth Pearson and SSIM both fail the "
         f"permutation check. Scored against a \\emph{{different}} sequence's "
         f"real product, Pearson reaches $+0.774$ against a weakest matched "
         f"pair of $+0.732$, and SSIM reaches $+0.797$ against a matched "
         f"\\emph{{mean}} of $+0.741$ -- 11 of 12 permuted pairs outscore the "
         f"weakest matched pair, and four renders score higher against another "
         f"sequence than against their own. "
         f"The metric is dominated by the static scene every recording shares. "
         f"See the validation report.", "tab:ra"),
        ("range_doppler", RANGE_DOPPLER, "Range-Doppler agreement, per radar. "
         "Ratios are rendered over real, so 1.00 is exact.", "tab:rd"),
        ("microdoppler", MICRODOPPLER, "Micro-Doppler agreement, per radar. "
         "These metrics pass the permutation check with a wide margin: matched "
         "centroid $r\\ge+0.974$ against a permuted range of $[-0.200,+0.044]$, "
         "with nothing between. clDice is "
         "the topology metric reported here, having passed both the threshold "
         "sweep and the permutation check (matched $0.8434$ against permuted "
         "$0.5406$). The Betti errors are deliberately absent: they do not "
         "separate matched from permuted pairs at all.",
         "tab:md"),
        ("weak_numbers", WEAK, "Known limitations, reported rather than "
         "footnoted. The range-Doppler floor is not matched in level or in "
         "spread. The median-spread pair is printed for both, but note that "
         "the under-dispersion reported previously was largely an artefact of "
         "scoring still frames: on the motion subset the rendered spread is "
         "$0.93$--$1.18\\times$ the real, against $0.72$--$0.98\\times$ over "
         "all 600 frames, so the render is not systematically narrower. "
         "Whichever frame set produced these cells is named in "
         "\\texttt{frame\\_selection\\_rule}.", "tab:weak"),
    ]
    for name, rows, caption, label in specs:
        text = table(rows, per_seq, caption, label)
        (args.out / f"{name}.tex").write_text(text + "\n")
        print(f"  wrote {args.out / f'{name}.tex'}  ({len(rows)} rows x "
              f"{len(per_seq)} sequences x 2 radars)")
    (args.out / "PROVENANCE.txt").write_text(
        f"generated from {args.eval_dir}\n"
        f"git_sha {list(shas)[0]}\nspec_version {per_seq[0]['spec_version']}\n"
        f"cfar_p_fa {list(pfa)[0]}\n"
        f"sequences {[d['sequence'] for d in per_seq]}\n"
        f"frames_scored {[d['frames_scored'] for d in per_seq]}\n")
    print(f"  wrote {args.out / 'PROVENANCE.txt'}")
    return 0


def _entry_make_tables():
        return (main_make_tables())



COMMANDS = {
    "score-microdoppler": _entry_score_microdoppler,
    "make-tables": _entry_make_tables,
    # evaluate.py's main(), which collided with this module's. cli.dispatch pops
    # the subcommand and leaves the rest of argv to its own parser, so every
    # flag `python -m mm4DPS.evaluate` took is unchanged.
    "evaluate": _entry_evaluate,
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
