"""Receiver noise floor and near-field coupling for the OctoNet bands: the 4DPS models, ported (2026-10-29).

Two nuisances the 4DPS forward model carries that the OctoNet fits did not, each measured from HELD-OUT recordings of
the same scene (never from the pooled target they are then used against):

  1. **Learned noise floor** (`mm4DPS/signal_model.py::LearnedNoiseFloor`, `calibration.py::run_noise_envelope`):
     the receiver's expected noise POWER per channel and range bin = a measured envelope prior (the power of the
     frame-to-frame residual about each recording's coherent mean, the process that does not repeat from frame to
     frame; the person's range band dropped and interpolated) times a learned scale and a learned low-order (8 DCT
     coefficients, +-3 in the log) range correction. Power adds incoherently, so the floor enters UNDER the square
     root of the rendered power: |model| = sqrt(|render|^2 + P). Nothing is drawn: this is the expectation the
     measured temporal mean is compared against.
  2. **Coupling block** (`calibration.py::run_sequence_coupling`, `forward_render.py::coupling_block`): the board's
     static near-field return (direct Tx->Rx path, housing, table) is the coherent mean over frames and recordings
     of the complex range profiles at the near bins, per channel, in product units; the head adds it to the
     rendered profiles AFTER the absolute gain, and the loss then sees the near bins instead of gating them.
     Adoption is decided per bin region by a rule fixed before the measurement: the held-out block must predict
     the pooled target's near bins with a per-channel log-magnitude correlation >= 0.9 (the GT's own far-bin
     reproducibility); where it does not, the guard stays.

Both need the render in absolute units: a learned per-band log gain (init from the iteration-0 median ratio) replaces
the median normalisation of the loss (`sfcw_static_loss(..., normalize=False)`), as 4DPS's per-radar absolute gain.

    cd 4DPS && python -m mm4DPS.octonet_noise measure            # held-out recordings 6-10 of Scene 1 -> $NBS_DATA_OUT/noise_S1/
    python -m mm4DPS.octonet_noise report                        # the saved decisions
Consumers: `octonet_fit_multiband.fit(noise_model=, coupling=)`, `octonet_fit fit-static --noise-model --coupling`.
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np
import torch

from mm4DPS import octonet_heads as OH
from method import sfcw_target as ST
from preprocessing import octonet_config as C, octonet_io as IO, octonet_scene as S

DATA_OUT = os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT)
NOISE_DIR = os.path.join(DATA_OUT, "noise_S1")
# near region per band: SFCW bins < 24 (0.9 m) are the node's own return (T4); UWB: 1.0 m covers the direct pulse
# (peak bin 7, tail to ~45) and the housing/table. Neither is an antenna far-field bound (2D^2/lambda: Vayyar 4 cm
# aperture -> 0.7 m; X4 ~2 cm -> 2 cm): the region is the extent of the structure the scaffold does not contain.
NEAR_BINS = dict(sfcw=24, uwb=int(round(1.0 / OH.UWB_BIN_M)))
ADOPT_CORR = 0.9                      # the adoption rule (see module docstring)
END_BINS = 80                         # UWB frame-end artefact region (last 0.51 m of the 9.87 m frame)
PERSON_MARGIN_M = 0.75                # 4DPS noise-envelope default
N_COEFF = 8
EPS = 1e-30


def heldout_recordings(n: int = 5) -> list[str]:
    """Scene-1 recordings 6..5+n of the activity-distinct list (1..5 are the pooled target)."""
    return ST.pick_scene1_recordings(5 + n)[5:]


# ----------------------------------------------------------------------------- the learned floor (port)
def range_dct_basis(n_range: int, n_coeff: int) -> torch.Tensor:
    """``(n_coeff, n_range)`` DCT-II basis without the constant term (the scale carries the level)."""
    index = np.arange(n_range, dtype=np.float64)
    rows = [np.cos(np.pi * (order + 1) * (index + 0.5) / n_range) for order in range(n_coeff)]
    return torch.from_numpy(np.stack(rows))


class LearnedNoiseFloor(torch.nn.Module):
    """Expected noise power over (channel, range), differentiably: 4DPS `signal_model.LearnedNoiseFloor` with the
    (radar, Doppler, range) prior replaced by the band's (channel, range) prior POWER (no Doppler axis here)."""

    def __init__(self, prior_power: np.ndarray, n_coeff: int = N_COEFF, device="cpu"):
        super().__init__()
        p = np.asarray(prior_power, dtype=np.float64)
        if p.ndim == 1: p = p[None, :]
        if not np.all(np.isfinite(p)) or np.any(p < 0): raise ValueError("prior power must be finite and nonnegative")
        self.n_channels, self.n_range = p.shape
        self.register_buffer("prior_power", torch.from_numpy(p).to(device))
        self.register_buffer("basis", range_dct_basis(self.n_range, n_coeff).to(device))
        self.log_scale = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device=device))
        self.range_coeff = torch.nn.Parameter(torch.zeros(n_coeff, dtype=torch.float64, device=device))
        self.register_buffer("extra_power", torch.zeros_like(self.prior_power))

    def set_extra_power(self, power: torch.Tensor) -> None:
        """A frozen contribution of another process (4DPS: the diffuse floor). Not learned."""
        if power.shape != self.extra_power.shape: raise ValueError("extra power shape")
        self.extra_power = power.to(self.extra_power)

    def learned_power(self) -> torch.Tensor:
        correction = torch.exp(torch.clamp(self.range_coeff @ self.basis, -3.0, 3.0))           # (K,)
        scale = torch.exp(torch.clamp(2.0 * self.log_scale, -60.0, 60.0))
        return scale * self.prior_power * correction[None, :]

    def power(self) -> torch.Tensor:
        return self.learned_power() + self.extra_power

    @torch.no_grad()
    def init_scale(self, target_power_level: float) -> None:
        """Start the floor at ``target_power_level`` (median over the supervised cells): the trained scale is learned."""
        cur = float(self.prior_power.median())
        self.log_scale.fill_(0.5 * math.log(max(target_power_level, 1e-300) / max(cur, 1e-300)))

    def summary(self) -> dict:
        with torch.no_grad():
            corr = torch.exp(torch.clamp(self.range_coeff @ self.basis, -3.0, 3.0))
            return dict(log_scale=float(self.log_scale), range_coeff=self.range_coeff.cpu().tolist(),
                        range_correction_min_max=[float(corr.min()), float(corr.max())], median_power=float(self.learned_power().median()))

    @staticmethod
    def load(band: str, device="cpu") -> "LearnedNoiseFloor":
        z = np.load(os.path.join(NOISE_DIR, f"{band}_envelope.npz"))
        return LearnedNoiseFloor(z["prior_power"], device=device)


def floor_magnitude(x: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """sqrt(|x|^2 + P) with 4DPS's zero-safe guard (a cell no path lands in is exactly zero; sqrt'(0) is NaN)."""
    p = (x.real ** 2 + x.imag ** 2) if torch.is_complex(x) else x ** 2
    return torch.sqrt(torch.clamp(p + power, min=EPS))


def with_floored_magnitude(rp: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """Complex profiles whose magnitude is replaced by the floored one (phase kept), so every downstream consumer of
    complex ``rp`` (metrics, figures) sees the model's magnitude."""
    mag = floor_magnitude(rp, power); a = rp.abs()
    return torch.where(a > 0, rp * (mag / a.clamp(min=1e-300)), mag.to(rp.dtype))


# ----------------------------------------------------------------------------- the coupling block (port)
class CouplingBlock:
    """Per-channel coherent near-bin block in product units (complex (C, near) for SFCW, real RF (near,) for UWB), plus an
    optional END block on the last bins (UWB: the X4 frame-end artefact, a static return at the last bin in every recording
    and every scene; same measurement and adoption rule as the near block)."""

    def __init__(self, block: np.ndarray, adopt_from_bin: int, device="cpu", end_block: np.ndarray | None = None, end_adopted: bool = False):
        self.block = torch.as_tensor(block, device=device)
        self.near = int(self.block.shape[-1]); self.adopt_from_bin = int(adopt_from_bin)
        self.end_block = torch.as_tensor(end_block, device=device) if (end_block is not None and end_adopted) else None
        self.n_end = int(self.end_block.shape[-1]) if self.end_block is not None else 0

    def add(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., K) + the block on bins [0, near) (+ the end block on bins [K - n_end, K))."""
        K = x.shape[-1]
        mid = torch.zeros(K - self.near - self.n_end, dtype=self.block.dtype, device=x.device)
        parts = [self.block.to(x.device), mid.expand(*self.block.shape[:-1], -1)]
        if self.n_end: parts.append(self.end_block.to(x.device).expand(*self.block.shape[:-1], -1) if self.block.dim() > 1 else self.end_block.to(x.device))
        return x + torch.cat(parts, dim=-1).to(x.dtype)

    @staticmethod
    def load(band: str, device="cpu") -> "CouplingBlock":
        z = np.load(os.path.join(NOISE_DIR, f"{band}_coupling.npz"))
        return CouplingBlock(z["block"], int(z["adopt_from_bin"]), device=device, end_block=(z["end_block"] if "end_block" in z.files else None),
                             end_adopted=bool(z["end_adopted"]) if "end_adopted" in z.files else False)


def coupling_available(band: str) -> bool:
    return os.path.exists(os.path.join(NOISE_DIR, f"{band}_coupling.npz"))


def guard_bins(band: str, coupling: bool) -> int:
    """First supervised bin: the block's adoption bin when the coupling model is on and adopted, else the near region."""
    if coupling and coupling_available(band):
        return CouplingBlock.load(band).adopt_from_bin
    return NEAR_BINS[band]


# ----------------------------------------------------------------------------- applying both to a head output
def apply_sfcw(rp: torch.Tensor, log_gain: torch.Tensor | None, block: CouplingBlock | None, floor: LearnedNoiseFloor | None) -> torch.Tensor:
    """rendered (400,K) complex -> absolute gain, + coupling block, floored magnitude."""
    if log_gain is not None: rp = rp * torch.exp(log_gain).to(rp.dtype)
    if block is not None: rp = block.add(rp)
    if floor is not None: rp = with_floored_magnitude(rp, floor.power().to(rp.real.dtype))
    return rp


def apply_uwb(rf: torch.Tensor, log_gain: torch.Tensor | None, block: CouplingBlock | None, floor: LearnedNoiseFloor | None):
    """rendered RF frame (K,) -> (rf, env) with the absolute gain, the coupling block on the RF, the floor on the envelope power."""
    if log_gain is not None: rf = rf * torch.exp(log_gain)
    if block is not None: rf = block.add(rf)
    env = OH.hilbert_envelope(rf)
    if floor is not None: env = floor_magnitude(env, floor.power()[0].to(env.dtype))
    return rf, env


def init_log_gain(model_mag: torch.Tensor, target_mag: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Iteration-0 absolute gain: the median log ratio over the supervised cells (then learned)."""
    m = mask
    a = torch.log(model_mag[..., m].abs() + 1e-300); b = torch.log(target_mag[..., m].abs() + 1e-300)
    return torch.tensor([float(b.median() - a.median())], dtype=torch.float64, device=model_mag.device)


# ----------------------------------------------------------------------------- measurement from held-out recordings
def _interp_band(power: np.ndarray, drop: np.ndarray) -> np.ndarray:
    """Replace the dropped bins of a (..., K) log-power array by linear interpolation along range."""
    out = power.copy(); K = power.shape[-1]; idx = np.arange(K); keep = ~drop
    flat = out.reshape(-1, K)
    for row in flat:
        row[drop] = np.interp(idx[drop], idx[keep], row[keep])
    return flat.reshape(power.shape)


def _logmag_corr(a: np.ndarray, b: np.ndarray) -> float:
    la = np.log(np.abs(a).ravel() + 1e-300); lb = np.log(np.abs(b).ravel() + 1e-300)
    la = la - la.mean(); lb = lb - lb.mean()
    return float((la * lb).sum() / (np.sqrt((la * la).sum() * (lb * lb).sum()) + 1e-300))


def measure_sfcw(recs: list[str], device) -> dict:
    """Envelope prior (400,100) power and coupling block (400,24) from the held-out recordings, both on the two-tone-cleaned
    profiles the target is built from. Person band NOT dropped: the person is not observable in the SFCW (measured)."""
    from method import sfcw_coupling as SC
    from mm3DPS.data import ra_utils as RA
    pw, blocks, Ts, names = [], [], [], []
    near = NEAR_BINS["sfcw"]
    for r in recs:
        V, _, p = ST.load_recording(r); names.append(os.path.basename(p))
        V = SC.fit_coupling(V.to(device), n_tones=2)["V_clean"]
        rp = RA.sfcw_range_compress(V.to(torch.complex64))                       # (T,400,100)
        mean = rp.mean(0)
        pw.append(((rp - mean[None]).abs() ** 2).mean(0).cpu().numpy())          # (400,100) process power
        blocks.append(mean[:, :near].cpu().numpy()); Ts.append(int(rp.shape[0]))
        del V, rp
    power = np.median(np.stack(pw), 0)                                          # robust over recordings
    level = float(np.median(power[:, near:]))
    T = np.asarray(Ts, float)
    block = (np.stack(blocks) * T[:, None, None]).sum(0) / T.sum()
    per_rec_near_power = float(np.mean([np.abs(b) ** 2 for b in blocks]))
    coherence = float((np.abs(block) ** 2).mean() / max(per_rec_near_power, 1e-300))
    # reproducibility against the POOLED target's near bins (bins 0-5 direct coupling residue, 6-23 housing/table)
    z = np.load(os.path.join(DATA_OUT, "sfcw_target_S1", "target_pool5.npz")); pool = z["rp_mean"]
    regions = {"0-5": (0, 6), "6-23": (6, near)}
    corr = {k: _logmag_corr(block[:, a:b], pool[:, a:b]) for k, (a, b) in regions.items()}
    pool_vs_pool_far = float(np.mean(z["xcorr"][near:]))
    adopt = near
    if corr["6-23"] >= ADOPT_CORR: adopt = 6
    if corr["6-23"] >= ADOPT_CORR and corr["0-5"] >= ADOPT_CORR: adopt = 0
    os.makedirs(NOISE_DIR, exist_ok=True)
    np.savez(os.path.join(NOISE_DIR, "sfcw_envelope.npz"), prior_power=power / level, level=level, frames=T, recordings=np.array(names),
             model="power of the frame-to-frame residual about each recording's coherent mean (two-tone-cleaned, Hann range-compressed), median over 5 held-out recordings, normalised to its median over bins >= 24")
    np.savez(os.path.join(NOISE_DIR, "sfcw_coupling.npz"), block=block.astype(np.complex128), adopt_from_bin=adopt, near_bins=near, coherence=coherence,
             corr_vs_pool=json.dumps(corr), frames=T, recordings=np.array(names),
             model="frame-weighted coherent mean over frames and 5 held-out recordings of the two-tone-cleaned range profiles at bins < 24, per Tx-Rx pair")
    return dict(band="sfcw", recordings=names, frames=Ts, envelope_level=level, envelope_shape_range=[float(power.min() / level), float(power.max() / level)],
                coupling_coherent_fraction=coherence, coupling_corr_vs_pool=corr, pool_far_bin_xcorr=pool_vs_pool_far, adopt_from_bin=adopt)


def measure_uwb(recs: list[str], device=None) -> dict:
    """Envelope prior (1535,) power and the RF coupling block (155,) from the held-out recordings' UWB streams."""
    from method.uwb_template import uwb_path_for, analytic
    near = NEAR_BINS["uwb"]
    pw, blocks, ends, Fs, names = [], [], [], [], []
    for r in recs:
        stamp = os.path.basename(r)[:14] if os.path.isabs(r) else r
        p = uwb_path_for(stamp); names.append(os.path.basename(p))
        fr, _ = IO.load_uwb(p)                                                  # (F,1535) real
        mean = fr.mean(0); an_mean = analytic(mean)
        resid = np.zeros(fr.shape[1])
        for f in fr: resid += np.abs(analytic(f) - an_mean) ** 2
        pw.append(resid / len(fr)); blocks.append(mean[:near]); ends.append(mean[-END_BINS:]); Fs.append(len(fr))
    power = np.median(np.stack(pw), 0)
    # the person's range band (subject at the MoCap origin), dropped and interpolated in the log as in 4DPS
    ext = json.load(open(os.path.join(DATA_OUT, "calib_S1", "extrinsics.json")))
    pos = np.asarray(OH.sensor_from_extrinsics(ext, "uwb")["pos"], float)
    centre = float(np.hypot(pos[0], pos[1])); rng = np.arange(power.shape[0]) * OH.UWB_BIN_M
    drop = (rng > centre - PERSON_MARGIN_M) & (rng < centre + PERSON_MARGIN_M)
    power = np.exp(_interp_band(np.log(power + 1e-300), drop))
    level = float(np.median(power[near:]))
    F = np.asarray(Fs, float); block = (np.stack(blocks) * F[:, None]).sum(0) / F.sum()
    coherence = float((block ** 2).mean() / max(np.mean([b ** 2 for b in blocks]), 1e-300))
    z = np.load(os.path.join(DATA_OUT, "uwb_target_S1", "target_pool5.npz")); pool_rf = z["mean_rf"]
    # the frame-END artefact (last END_BINS): a static return peaking at the last bin in every recording and scene (2026-10-29 diag);
    # measured like the near block from the held-out coherent means, adopted by the same rule against the pooled target
    end_block = (np.stack(ends) * F[:, None]).sum(0) / F.sum()
    env_eb = np.abs(analytic(np.concatenate([np.zeros(pool_rf.shape[0] - END_BINS), end_block])))[-END_BINS:]
    env_pe = np.abs(analytic(pool_rf))[-END_BINS:]
    end_corr = _logmag_corr(env_eb, env_pe); end_adopted = bool(end_corr >= ADOPT_CORR)
    env_b = np.abs(analytic(np.concatenate([block, np.zeros(pool_rf.shape[0] - near)])))
    env_p = np.abs(analytic(pool_rf))
    regions = {"0-44 (direct pulse)": (0, 45), "45-154 (housing/table)": (45, near)}
    corr = {k: _logmag_corr(env_b[a:b], env_p[a:b]) for k, (a, b) in regions.items()}
    rf_corr = {k: float(np.corrcoef(block[a:b], pool_rf[a:b])[0, 1]) for k, (a, b) in regions.items()}
    adopt = near
    if corr["45-154 (housing/table)"] >= ADOPT_CORR: adopt = 45
    if adopt == 45 and corr["0-44 (direct pulse)"] >= ADOPT_CORR: adopt = 0
    os.makedirs(NOISE_DIR, exist_ok=True)
    np.savez(os.path.join(NOISE_DIR, "uwb_envelope.npz"), prior_power=(power / level)[None, :], level=level, frames=F, recordings=np.array(names), person_band_dropped=drop,
             model="power of the frame-to-frame residual of the analytic RF frame about each recording's coherent mean, median over 5 held-out recordings, person band +-0.75 m dropped and interpolated, normalised to its median over bins >= 155")
    np.savez(os.path.join(NOISE_DIR, "uwb_coupling.npz"), block=block.astype(np.float64), adopt_from_bin=adopt, near_bins=near, coherence=coherence,
             corr_vs_pool=json.dumps(corr), rf_corr_vs_pool=json.dumps(rf_corr), frames=F, recordings=np.array(names),
             end_block=end_block.astype(np.float64), end_bins=END_BINS, end_corr_vs_pool=end_corr, end_adopted=end_adopted,
             model="frame-weighted coherent mean over frames and 5 held-out recordings of the RF frame at bins < 155 (1.0 m); end_block = the same at the last END_BINS (frame-end artefact)")
    return dict(band="uwb", recordings=names, frames=Fs, envelope_level=level, person_band_m=[centre - PERSON_MARGIN_M, centre + PERSON_MARGIN_M],
                envelope_shape_range=[float(power.min() / level), float(power.max() / level)], coupling_coherent_fraction=coherence,
                coupling_env_corr_vs_pool=corr, coupling_rf_corr_vs_pool=rf_corr, adopt_from_bin=adopt, end_bins=END_BINS, end_corr_vs_pool=end_corr, end_adopted=end_adopted)


def cmd_measure(a):
    dev = torch.device(a.device)
    recs = heldout_recordings(a.n_heldout)
    print("held-out recordings:", [os.path.basename(r).split("_activity_")[1].split("_trial")[0] for r in recs])
    out = {}
    for band in a.bands.split(","):
        out[band] = measure_sfcw(recs, dev) if band == "sfcw" else measure_uwb(recs)
        print(json.dumps(out[band], indent=1, default=str))
    json.dump(out, open(os.path.join(NOISE_DIR, "measure.json"), "w"), indent=1, default=str)


def cmd_report(a):
    for band in ("sfcw", "uwb"):
        for kind in ("envelope", "coupling"):
            p = os.path.join(NOISE_DIR, f"{band}_{kind}.npz")
            if not os.path.exists(p): print(f"{band} {kind}: missing"); continue
            z = np.load(p)
            print(f"{band} {kind}:", {k: (z[k].tolist() if z[k].ndim == 0 or z[k].size < 12 else f"array{z[k].shape}") for k in z.files})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure"); m.add_argument("--bands", default="sfcw,uwb"); m.add_argument("--n-heldout", type=int, default=5); m.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub.add_parser("report")
    a = ap.parse_args(argv); {"measure": cmd_measure, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
