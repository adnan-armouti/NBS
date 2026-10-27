"""The 3DPS differentiable rasterizer, vendored. See mm4DPS/splat/PROVENANCE.md.

Five modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C7, in
dependency order:

    psf                    the Hann-window range PSF
    bsdf_torch             the full Jones BSDF, all six ITU parameters
    rasterizer             Rasterizer, material reparameterization
    rasterizer_factorized  render_factorized, the range-profile splat path
    primitives             PointPrimitives, render_gaussians, cull_gaussians

They are ONE module because they are mutually dependent: `primitives` imports
`rasterizer`, and `rasterizer_factorized` imports `bsdf_torch`. Those cross in
opposite directions, so splitting the five any two ways puts a cycle between the
halves. `C_LIGHT` was declared three times and `TWO_PI` twice, identically.

`_bsdf_mod` is gone: it existed so `K_WAVE` was read after `set_frequency()`
could rebind it, and a module global read inside a function is that same late
binding now that the BSDF lives here too.

mm4DPS/splat/ remains, holding what is not Python -- the MMWCAS antenna patterns
that `MM3DPS_ROOT` resolves to, the CUDA sources, and the attribution.
"""

import json
import math
import mitsuba as mi
import numpy as np
import os
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple


# --------------------------------------------------------------------
# the Hann-window range PSF
# --------------------------------------------------------------------

PI = math.pi
TWO_PI = 2.0 * PI


def dirichlet_kernel(delta: Tensor, K: int) -> Tensor:
    """Dirichlet kernel (periodic sinc): D(δ) = sin(πδ) / sin(πδ/K).

    Returns complex values including the phase shift exp(-jπδ(K-1)/K).
    """
    pi_delta = PI * delta
    pi_delta_K = PI * delta / K

    sin_num = torch.sin(pi_delta)
    sin_den = torch.sin(pi_delta_K)
    near_integer = sin_den.abs() < 1e-8
    sin_den_safe = torch.where(near_integer, torch.ones_like(sin_den), sin_den)
    magnitude = torch.where(near_integer, torch.full_like(delta, float(K)), sin_num / sin_den_safe)

    phase = -PI * delta * (K - 1) / K
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)

    return torch.complex(real, imag)


def hann_psf(delta: Tensor, K: int) -> Tensor:
    """Range PSF for Hann-windowed FFT (analytic, slow for large tensors).

    PSF_hann(δ) = 0.5 * D(δ) - 0.25 * D(δ-1) - 0.25 * D(δ+1)
    """
    return (0.5 * dirichlet_kernel(delta, K)
            - 0.25 * dirichlet_kernel(delta - 1.0, K)
            - 0.25 * dirichlet_kernel(delta + 1.0, K))


class HannPSFTable:
    """Precomputed Hann PSF lookup table for fast evaluation.

    Precomputes PSF(dn - frac) for all integer offsets dn in [-spread//2, spread//2]
    and fractional offsets frac on a fine grid in [0, 1). Evaluation is a single
    table lookup + linear interpolation — no trig functions at runtime.
    """

    def __init__(self, K: int, spread: int, n_grid: int = 1024, device: str = 'cuda:0'):
        self.K = K
        self.spread = spread
        self.n_grid = n_grid
        self.device = device

        # Fractional offset grid: frac ∈ [0, 1), n_grid points
        frac_grid = torch.linspace(0, 1 - 1/n_grid, n_grid, device=device)  # (n_grid,)

        # Integer offsets
        dn_offsets = torch.arange(-(spread // 2), spread // 2 + 1, device=device)  # (spread,)

        # Precompute: PSF(dn - frac) for all (spread, n_grid) combinations
        delta = dn_offsets[:, None].float() - frac_grid[None, :]   # (spread, n_grid)
        psf_vals = hann_psf(delta, K)                               # (spread, n_grid) complex

        self.psf_real = psf_vals.real.contiguous()                  # (spread, n_grid)
        self.psf_imag = psf_vals.imag.contiguous()                  # (spread, n_grid)

    def evaluate(self, n_frac: Tensor) -> tuple:
        """Look up PSF values for fractional offsets via linear interpolation.

        Args:
            n_frac: (P,) fractional part of range bin, in [0, 1)

        Returns:
            psf_real: (spread, P) real parts of PSF at all offsets
            psf_imag: (spread, P) imag parts of PSF at all offsets
        """
        # Map frac to grid index
        idx_f = n_frac * self.n_grid                                # (P,) in [0, n_grid)
        idx_lo = idx_f.long().clamp(0, self.n_grid - 1)            # (P,)
        idx_hi = (idx_lo + 1).clamp(0, self.n_grid - 1)
        frac = idx_f - idx_lo.float()                               # (P,) interpolation weight

        # Gather: (spread, P) from (spread, n_grid)
        # psf_real[:, idx_lo] gives (spread, P)
        pr_lo = self.psf_real[:, idx_lo]                            # (spread, P)
        pr_hi = self.psf_real[:, idx_hi]
        pi_lo = self.psf_imag[:, idx_lo]
        pi_hi = self.psf_imag[:, idx_hi]

        # Linear interpolation
        f = frac[None, :]                                           # (1, P)
        psf_r = pr_lo * (1 - f) + pr_hi * f                       # (spread, P)
        psf_i = pi_lo * (1 - f) + pi_hi * f                       # (spread, P)

        return psf_r, psf_i

# --------------------------------------------------------------------
# the full Jones BSDF for mmWave radar
# --------------------------------------------------------------------

C_LIGHT = 299_792_458.0
WAVELENGTH = 3.9e-3                   # metres (matches the differentiable renderer's WAVELENGTH_77GHZ exactly)
K_WAVE = 2.0 * math.pi / WAVELENGTH  # ~1611 rad/m


def set_frequency(frequency_hz):
    """Set the carrier used by the RF material model at runtime."""
    global WAVELENGTH, K_WAVE, KAPPA_77GHZ
    WAVELENGTH = 299792458.0 / float(frequency_hz)
    K_WAVE = 2.0 * math.pi / WAVELENGTH
    KAPPA_77GHZ = K_WAVE
INV_PI = 1.0 / math.pi

# Default TX/RX polarization: vertical = (0, 0, 1)
DEFAULT_POL = None  # set lazily on first use


# ===================================================================
#  Complex number helpers (using PyTorch native complex where useful)
# ===================================================================

# Complex working precision. complex64 is the historical default; the
# verification harness switches to complex128 to arbitrate kernel-vs-torch
# discrepancies (the CUDA slab path computes in fp64 — see utils.cuh).
_CDTYPE = torch.complex64
_RDTYPE = torch.float32




def _cpx(r: Tensor, i: Tensor) -> Tensor:
    """Build complex tensor from real and imaginary parts."""
    return torch.complex(r.to(_RDTYPE), i.to(_RDTYPE))


def _cpx_abs_sq(z: Tensor) -> Tensor:
    """Power = |z|² = real² + imag²."""
    return z.real ** 2 + z.imag ** 2


# ===================================================================
#  Permittivity → IOR
# ===================================================================



# ===================================================================
#  Fresnel reflection
# ===================================================================





# ===================================================================
#  ITU Slab Fresnel (P.2040-4 Eqs 43-44)
# ===================================================================

def itu_slab_fresnel(
    eps_real: Tensor, eps_imag: Tensor,
    cos_theta_i: Tensor, thickness: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ITU-R P.2040-4 single-layer slab reflection + transmission.

    Returns (R_TE, R_TM, T_TE, T_TM) as complex tensors.
    """
    cos_i = cos_theta_i.clamp(1e-6, 1.0)
    sin_sq = 1.0 - cos_i ** 2

    # Complex permittivity
    eta = _cpx(eps_real, -eps_imag)

    # √(η - sin²θ)
    a = torch.sqrt(eta - sin_sq.to(_CDTYPE))

    cos_c = cos_i.to(_CDTYPE)

    # Single-interface Fresnel coefficients
    r_te = (cos_c - a) / (cos_c + a + 1e-10)
    r_tm = (eta * cos_c - a) / (eta * cos_c + a + 1e-10)

    # Phase delay through slab
    q = (TWO_PI / WAVELENGTH) * thickness.to(_CDTYPE) * a

    # exp(-jq) and exp(-j2q)
    ejq = torch.exp(-1j * q)
    ej2q = torch.exp(-2j * q)

    # Slab reflection: R = r(1 - exp(-j2q)) / (1 - r²exp(-j2q))
    R_TE = r_te * (1.0 - ej2q) / (1.0 - r_te ** 2 * ej2q + 1e-10)
    R_TM = r_tm * (1.0 - ej2q) / (1.0 - r_tm ** 2 * ej2q + 1e-10)

    # Slab transmission: T = (1 - r²)exp(-jq) / (1 - r²exp(-j2q))
    T_TE = (1.0 - r_te ** 2) * ejq / (1.0 - r_te ** 2 * ej2q + 1e-10)
    T_TM = (1.0 - r_tm ** 2) * ejq / (1.0 - r_tm ** 2 * ej2q + 1e-10)

    return R_TE, R_TM, T_TE, T_TM




# ===================================================================
#  SPM Validity + Parameter Enforcement
# ===================================================================



# ===================================================================
#  Scattering Lobes
# ===================================================================









# ===================================================================
#  Coherent / Incoherent Blend Factors
# ===================================================================

def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x.clamp(-50.0, 50.0))






# ===================================================================
#  Coherent Backscatter Enhancement
# ===================================================================





# ===================================================================
#  Jones Polarization
# ===================================================================

def _get_default_pol(device):
    """Lazy-init default vertical polarization vector."""
    global DEFAULT_POL
    if DEFAULT_POL is None or DEFAULT_POL.device != device:
        DEFAULT_POL = torch.tensor([0.0, 0.0, 1.0], device=device)
    return DEFAULT_POL


X_AXIS = None


def _get_x_axis(device):
    """Lazy-init +X unit vector (grazing-incidence fallback reference).

    Building it with torch.tensor(..., device=cuda) inside the hot path is a
    blocking host-to-device copy; caching it makes the call free.
    """
    global X_AXIS
    if X_AXIS is None or X_AXIS.device != device:
        X_AXIS = torch.tensor([1.0, 0.0, 0.0], device=device)
    return X_AXIS


def compute_sp_basis(wi: Tensor, n: Tensor) -> Tuple[Tensor, Tensor]:
    """s/p polarization basis for plane of incidence.

    s = (wi × n) / |wi × n|  (perpendicular to incidence plane)
    p = (s × wi) / |s × wi|  (in incidence plane)
    """
    s = torch.cross(wi, n, dim=-1)
    s_norm = s.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    s = s / s_norm

    # Fallback for grazing incidence.
    #
    # Runtime: `if grazing.any()` read a GPU boolean back to the host on every
    # call, and the constant inside was rebuilt with torch.tensor(...,
    # device=cuda) — two CPU<->GPU round trips per render. Evaluating the
    # fallback unconditionally and selecting with torch.where gives bitwise
    # the same `s` (where() returns the primary branch wherever grazing is
    # False) and the same gradient (the unselected branch receives exactly
    # zero). The fallback is well-conditioned here: s_fb = wi x (1,0,0) =
    # (0, wi_z, -wi_y) can only vanish for wi exactly +-X, and wi always
    # points from a body point to a boresight-forward (+Y) antenna.
    grazing = s_norm.squeeze(-1) < 1e-4
    fallback_ref = _get_x_axis(wi.device).expand_as(wi)
    s_fb = torch.cross(wi, fallback_ref, dim=-1)
    s_fb = s_fb / s_fb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    s = torch.where(grazing.unsqueeze(-1), s_fb, s)

    p = torch.cross(s, wi, dim=-1)
    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return s, p




# ===================================================================
#  Main BSDF Entry Points
# ===================================================================





# ===================================================================
#  Backward-compatible aliases (used by adc_synthesis.py)
# ===================================================================

# Keep Tier 1 for testing/fallback


KAPPA_77GHZ = K_WAVE  # alias for adc_synthesis.py

# --------------------------------------------------------------------
# Rasterizer: material reparameterization and the slimmed c6 path
# --------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Material reparameterization (matches DrJit bounds exactly)
# ---------------------------------------------------------------------------

def reparameterize_torch(raw: Tensor) -> Tensor:
    """Map raw params -> physics params.

    Original DrJit bounds (mm3DPS/preprocessing/reparameterization.py L109-114):
      eps_real:  1.5 + 8.5 * sigmoid(x)        -> [1.5, 10]
      eps_imag:  exp(clamp(x, -7.0, 16.0))     -> [~1e-3, ~9e6]
      sigma_h:   exp(clamp(x, -16.0, -7.0))    -> [~1e-7, ~1e-3]
      l_c:       exp(clamp(x, -7.6, -2.3))     -> [~5e-4, 0.1]
      tau:       0.05 + 0.9 * sigmoid(x)       -> [0.05, 0.95]
      thickness: exp(clamp(x, -7.0, -1.2))     -> [~1e-3, 0.3]

    Post-2026-04-13 Fix 1: thickness upper bound widened from -1.2 to 2.0
    (exp(-1.2)=0.3 m → exp(2.0)=7.39 m). Phase P5 audit showed 78% of
    points under random init were saturating at the 300 mm upper bound,
    so the optimizer was pushing thickness past the clamp. The new bound
    gives effectively unbounded upward movement (7 m covers "infinite-
    thickness" interferometric averaging without losing physics sense).

    Post-2026-04-13 Fix 3: l_c reparam range widened from [-7.6, -2.3]
    (0.5 mm – 100 mm) to [-10, 2] (45 μm – 7.39 m). P5 showed 64% of
    points hit the old clamp under random init. Physical correlation
    lengths at mmWave-scale roughness span anywhere from tens of μm
    (fine textures) to meters (large-scale structures), well beyond the
    original bound.

    Post-2026-04-13 Fix 4: eps_real sigmoid → softplus. Old reparam
    `1.5 + 8.5*sigmoid(x)` mapped to [1.5, 10] with hard saturation at
    ±5 in raw space (45% of points saturated under random init). New
    reparam `1.0 + softplus(x)` is monotonic, min=1.0 (vacuum), with
    no upper saturation. softplus(0)≈0.69 so default eps_real≈1.69 at
    raw=0 (reasonable low-dielectric starting point). Gradient never
    dies.
    """
    out = torch.empty_like(raw)
    out[..., 0] = 1.0 + torch.nn.functional.softplus(raw[..., 0])
    out[..., 1] = torch.exp(torch.clamp(raw[..., 1], -7.0, 16.0))
    out[..., 2] = torch.exp(torch.clamp(raw[..., 2], -16.0, -7.0))
    out[..., 3] = torch.exp(torch.clamp(raw[..., 3], -10.0, 2.0))
    out[..., 4] = 0.05 + 0.9 * torch.sigmoid(raw[..., 4])
    out[..., 5] = torch.exp(torch.clamp(raw[..., 5], -7.0, 2.0))
    return out


def inverse_reparameterize_torch(physics: np.ndarray) -> np.ndarray:
    """Physics params -> raw params (numpy, used at init)."""
    raw = np.empty_like(physics)

    def _logit(x):
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(x / (1.0 - x))

    def _softplus_inv(y):
        # softplus^-1(y) = log(exp(y) - 1). Clip y > 0.
        return np.log(np.expm1(np.clip(y, 1e-6, None)))

    raw[..., 0] = _softplus_inv(physics[..., 0] - 1.0)
    raw[..., 1] = np.log(np.clip(physics[..., 1], np.exp(-7.0), np.exp(16.0)))
    raw[..., 2] = np.log(np.clip(physics[..., 2], np.exp(-16.0), np.exp(-7.0)))
    raw[..., 3] = np.log(np.clip(physics[..., 3], np.exp(-10.0), np.exp(2.0)))
    raw[..., 4] = _logit((physics[..., 4] - 0.05) / 0.9)
    raw[..., 5] = np.log(np.clip(physics[..., 5], np.exp(-7.0), np.exp(2.0)))
    return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# Antenna pattern (PyTorch evaluation, matches the differentiable renderer element_patterns.py)
# ---------------------------------------------------------------------------

class AntennaPatternTorch:
    """Antenna pattern for PyTorch evaluation, with injectable learned patterns."""

    def __init__(self, pattern_path: str, device: str = "cuda:0"):
        data = np.load(pattern_path)  # (361, 2) [E_dB, H_dB]
        E_lin = np.power(10.0, data[:, 0] / 10.0)
        H_lin = np.power(10.0, data[:, 1] / 10.0)

        G_max_dB = max(data[:, 0].max(), data[:, 1].max())
        G_max_lin = 10.0 ** (G_max_dB / 10.0)
        P_max = (E_lin * H_lin).max()
        self.C_scale = float(G_max_lin / P_max) if P_max > 1e-12 else 1.0

        self.E = torch.from_numpy(E_lin.astype(np.float32)).to(device)
        self.H = torch.from_numpy(H_lin.astype(np.float32)).to(device)
        self.device = device
        # Cached axis constants for evaluate(); see the note there.
        self._aux_x = torch.tensor([1.0, 0.0, 0.0], device=device)
        self._aux_z = torch.tensor([0.0, 0.0, 1.0], device=device)

    def evaluate(self, directions: Tensor, boresights: Tensor) -> Tensor:
        """Evaluate antenna gain. directions, boresights: (N, 3)."""
        y_local = boresights

        # Gram-Schmidt local frame (matches the differentiable renderer element_patterns.py L638-651)
        #
        # Runtime: the two axis constants used to be built by
        # torch.tensor(..., device='cuda') on every call — a blocking H2D copy
        # each — and the degenerate case sat behind `if parallel.any()`, a
        # device-to-host read. Four CPU<->GPU round trips per render call, all
        # of them for two compile-time constants. The constants are cached and
        # the branch becomes an unconditional torch.where, which returns `aux`
        # untouched wherever nothing is parallel — bitwise what the branch did.
        aux = self._aux_x.expand_as(y_local)
        parallel = torch.abs((y_local * aux).sum(-1)) > 0.99
        alt = self._aux_z.expand_as(y_local)
        aux = torch.where(parallel.unsqueeze(-1), alt, aux)

        z_local = torch.cross(y_local, aux, dim=-1)
        z_local = z_local / z_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_local = torch.cross(z_local, y_local, dim=-1)

        d_x = (directions * x_local).sum(-1)
        d_y = (directions * y_local).sum(-1)
        d_z = (directions * z_local).sum(-1)

        angle_E_deg = torch.rad2deg(torch.atan2(d_z, d_y)) + 180.0
        angle_H_deg = torch.rad2deg(torch.atan2(d_x, d_y)) + 180.0

        gain_E = self._interp(angle_E_deg, self.E)
        gain_H = self._interp(angle_H_deg, self.H)
        return (self.C_scale * gain_E * gain_H).clamp(min=0.0)

    def _interp(self, angle_deg: Tensor, pattern: Tensor) -> Tensor:
        """Interpolate 361-sample pattern matching the differentiable renderer's indexing exactly."""
        idx_f = angle_deg.clamp(0.0, 360.0)
        idx_lo = idx_f.floor().long().clamp(0, 360)
        idx_hi = (idx_lo + 1)
        idx_hi = torch.where(idx_hi >= 361, torch.zeros_like(idx_hi), idx_hi)
        frac = idx_f - idx_f.floor()
        return pattern[idx_lo] * (1.0 - frac) + pattern[idx_hi] * frac



# ---------------------------------------------------------------------------
# Rasterizer: FMCW config + array geometry + antenna patterns + Mitsuba scene
# ---------------------------------------------------------------------------

class Rasterizer:
    """Holds the static radar configuration that the renderer reads at every
    iteration: FMCW parameters, TX/RX positions and boresights, antenna
    patterns, and the Mitsuba scene used for one-shot init-time ray tracing.
    """

    def __init__(self, config_file, mesh_file, tx_pattern_file, rx_pattern_file,
                 device="cuda:0"):
        with open(config_file) as f:
            radar_cfg = json.load(f)

        # FMCW parameters
        self.center_freq = float(radar_cfg['carrierFrequency'])
        self.slope = float(radar_cfg['freqSlope'])
        self.sample_rate = float(radar_cfg['sampleRate'])
        self.num_samples = int(radar_cfg['numAdcSamples'])
        self.min_freq = self.center_freq
        self.wavelength = C_LIGHT / self.center_freq
        self.lambda_squared = self.wavelength ** 2
        self.K = self.num_samples

        # TX/RX positions (mm -> m) and boresights
        tx_elements = radar_cfg['tx_array']
        rx_elements = radar_cfg['rx_array']
        self.n_tx = len(tx_elements)
        self.n_rx = len(rx_elements)

        self.tx_positions = torch.tensor(
            [e['pos_mm'] for e in tx_elements], dtype=torch.float32,
            device=device) / 1000.0
        self.rx_positions = torch.tensor(
            [e['pos_mm'] for e in rx_elements], dtype=torch.float32,
            device=device) / 1000.0
        self.tx_boresights = torch.tensor(
            [e['boresight'] for e in tx_elements], dtype=torch.float32,
            device=device)
        self.rx_boresights = torch.tensor(
            [e['boresight'] for e in rx_elements], dtype=torch.float32,
            device=device)

        # Radar constants (must match the differentiable renderer)
        Pt_watts = 10 ** (13.0 / 10) / 1000.0
        antenna_loss_linear = 10 ** (-4.0 / 10)
        four_pi_squared = (4 * np.pi) ** 2
        self.rx_dBFS_scale = 10 ** ((30.0 - 13.0) / 20)
        self.adc_scale = np.sqrt(50.0) * 32768.0
        # Empirical GT-match factor. The physical constants above yield
        # C_radar ≈ 45.33, but at factory antenna patterns + ITU concrete init,
        # rendered |RA| is ~100× dimmer than GT across all 7 scenes (0.22 decades
        # residual spread). Multiplying radar_constant by 100 brings rendered
        # within factor ~1.5 of GT at init, which is the regime where raw MSE
        # loss can work. See md/v4_unnormalized_mse_loss_plan.md, Phase α.
        self.C_radar_gt_match = 100.0
        self.radar_constant = float(np.sqrt(
            Pt_watts * antenna_loss_linear * self.lambda_squared
            / four_pi_squared)) * self.C_radar_gt_match

        # Antenna patterns (PyTorch)
        self.tx_antenna = AntennaPatternTorch(tx_pattern_file, device)
        self.rx_antenna = AntennaPatternTorch(rx_pattern_file, device)

        # Store paths so the Mitsuba scene can be lazy-loaded for ray tracing
        self._config_file = config_file
        self._mesh_file = mesh_file
        self._tx_pattern_file = tx_pattern_file
        self._rx_pattern_file = rx_pattern_file
        self._mi_scene = None

        self.device = device

    def load_mi_scene(self):
        """Lazy-load a minimal Mitsuba scene for visibility ray tracing.

        Called once at init by the visibility ray-test code. The mesh is the
        only remaining mesh dependency — used solely for occlusion tests via
        ``scene.ray_test``.
        """
        if self._mi_scene is not None:
            return self._mi_scene
        import mitsuba as mi
        if mi.variant() is None:
            mi.set_variant("cuda_ad_rgb")
        self._mi_scene = mi.load_dict({
            "type": "scene",
            "mesh": {"type": "ply", "filename": str(self._mesh_file)},
        })
        return self._mi_scene

# --------------------------------------------------------------------
# render_factorized: the factorized renderer with range-profile splatting
# --------------------------------------------------------------------


# HuPR runtime: the splat's step-5 tail — the PSF table lerp and the
# carrier x PSF contribution products — is ~15 separate elementwise passes
# over (SPREAD, paths) tensors (~1.2 GB per training step at frame-batch
# 24). torch.compile fuses each group into one kernel. Two functions, not
# one, deliberately: the geometry-cached path recomputes contributions from
# the *stored* cos/sin/PSF of its fill pass, and routing both paths through
# the same `_splat_contribs` graph keeps cached == uncached bitwise (same
# kernel, same inputs) — the property the trainer's background re-render
# gate reads as 0.000e+00. HUPR_COMPILED_STEP5=0 restores the eager chain.
_COMPILED_STEP5 = os.environ.get("HUPR_COMPILED_STEP5", "1") != "0"


def _psf_lerp_eager(table_real, table_imag, n_frac, n_grid: int):
    """HannPSFTable.evaluate, expression for expression (see psf.py)."""
    idx_f = n_frac * n_grid
    idx_lo = idx_f.long().clamp(0, n_grid - 1)
    idx_hi = (idx_lo + 1).clamp(0, n_grid - 1)
    frac = (idx_f - idx_lo.float())[None, :]
    psf_r = table_real[:, idx_lo] * (1 - frac) + table_real[:, idx_hi] * frac
    psf_i = table_imag[:, idx_lo] * (1 - frac) + table_imag[:, idx_hi] * frac
    return psf_r, psf_i


def _carrier_eager(w_flat, cos_phi, sin_phi):
    """Active-path masking and the carrier phasor alone, ``(paths, 2)``.

    The fused scene path (`forward_render.assemble_static_profile_frames_cuda`)
    takes the carrier and synthesises carrier x PSF inside its kernels, so the
    (spread, paths) contribution pair is never built. The same two statements
    `_splat_contribs_eager` opens with.
    """
    w_act = w_flat * (w_flat > 1e-20).to(w_flat.dtype)
    return torch.stack([w_act * cos_phi, w_act * sin_phi], dim=-1)


def _splat_contribs_eager(w_flat, cos_phi, sin_phi, psf_r, psf_i):
    """Active-path masking, carrier phasor, and carrier x PSF products."""
    w_act = w_flat * (w_flat > 1e-20).to(w_flat.dtype)
    carrier_real = w_act * cos_phi
    carrier_imag = w_act * sin_phi
    contrib_real = carrier_real[None, :] * psf_r - carrier_imag[None, :] * psf_i
    contrib_imag = carrier_real[None, :] * psf_i + carrier_imag[None, :] * psf_r
    return contrib_real, contrib_imag


# dynamic=True, deliberately: one shape-generic kernel serves every path
# count, so the same inputs give bitwise the same outputs whatever the batch
# — the property test_two_radars_fold_into_one_render_call pins at
# rtol=atol=0 (shape-specialized kernels broke it at ~1e-05: different
# codegen, different contraction).
_psf_lerp = (torch.compile(_psf_lerp_eager, dynamic=True)
             if _COMPILED_STEP5 else _psf_lerp_eager)
_splat_contribs = (torch.compile(_splat_contribs_eager, dynamic=True)
                   if _COMPILED_STEP5 else _splat_contribs_eager)
_carrier = (torch.compile(_carrier_eager, dynamic=True)
            if _COMPILED_STEP5 else _carrier_eager)


def _prelude_eager(raw_materials, normals, areas, wi, wo, G_rx, reparameterize_fn,
                   slab_disabled: bool):
    """Steps 1-3 of `render_factorized` on a filled geometry cache, as one function.

    Statement for statement the inline code (the per-point material prep,
    the per-TX incidence terms and Fresnel, the per-RX observation terms),
    for the case the trainer's cached renders take: `bsdf_mode="full"`, no
    disabled components but possibly the slab, `wi`/`wo`/`G_rx` from the
    cache. Compiled (`_prelude`), the ~150 elementwise launches become a
    handful of fused kernels; the values move at the fp32 floor (fusion
    changes the rounding), which is why the inline path stays for everything
    that gates bitwise. The complex Fresnel is carried as real pairs so the
    whole function fuses.
    """
    physics = reparameterize_fn(raw_materials)
    eps_real = physics[:, 0]
    eps_imag = physics[:, 1]
    sigma_h = physics[:, 2]
    l_c = physics[:, 3]
    tau_base = physics[:, 4]
    thickness = physics[:, 5]
    if slab_disabled:
        thickness = torch.full_like(thickness, 1e-9)
    sh = sigma_h
    lc = l_c
    alpha_raw = 4.0 * math.pi * sh / WAVELENGTH
    alpha_ggx = torch.sqrt(alpha_raw.clamp(min=0.0025)).clamp(0.05, 0.95)
    alpha_sq = alpha_ggx ** 2
    l_c_lam = lc / WAVELENGTH
    roughness_slope = sh / lc.clamp(min=1e-8)
    kappa_SPM = (torch.sqrt(l_c_lam.clamp(min=0.0)) / (1.0 + 3.0 * roughness_slope)).clamp(0.5, 10.0)
    kappa_SPM_c = kappa_SPM.clamp(max=50.0)
    sinh_SPM = (torch.exp(kappa_SPM_c) - torch.exp(-kappa_SPM_c)) / 2.0
    norm_SPM = kappa_SPM / (4.0 * math.pi * sinh_SPM.clamp(min=1e-10))
    eps_contrast = torch.abs(eps_real - 1.0) + eps_imag
    eps_factor = (eps_contrast / 5.0).clamp(0.2, 1.0)

    M, n_tx = wi.shape[0], wi.shape[1]
    cos_i = (wi * normals[:, None, :]).sum(-1)
    normal_sign = torch.where(cos_i < 0, -torch.ones_like(cos_i), torch.ones_like(cos_i))
    cos_i = cos_i.abs().clamp(min=1e-6)
    n_eff = normals[:, None, :] * normal_sign.unsqueeze(-1)
    wi_dot_n = (wi * n_eff).sum(-1, keepdim=True)
    wi_r = 2.0 * wi_dot_n * n_eff - wi
    tan_sq_i = (1.0 / cos_i.clamp(min=1e-6) ** 2) - 1.0
    lambda_i = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_i).clamp(min=0.0))) / 2.0
    s_in_flat, p_in_flat = compute_sp_basis(wi.reshape(-1, 3), n_eff.reshape(-1, 3))
    s_in = s_in_flat.reshape(M, n_tx, 3)
    p_in = p_in_flat.reshape(M, n_tx, 3)
    tx_pol = _get_default_pol(wi.device)
    tx_s = (tx_pol * s_in).sum(-1)
    tx_p = (tx_pol * p_in).sum(-1)
    cos_i_flat = cos_i.reshape(-1)
    eps_r_flat = eps_real[:, None].expand(-1, n_tx).reshape(-1)
    eps_i_flat = eps_imag[:, None].expand(-1, n_tx).reshape(-1)
    thick_flat = thickness[:, None].expand(-1, n_tx).reshape(-1)
    R_TE_re, R_TE_im, R_TM_re, R_TM_im = _slab_fresnel_real(eps_r_flat, eps_i_flat, cos_i_flat, thick_flat)
    r_s_re, r_s_im = R_TE_re.reshape(M, n_tx), R_TE_im.reshape(M, n_tx)
    r_p_re, r_p_im = R_TM_re.reshape(M, n_tx), R_TM_im.reshape(M, n_tx)
    E_s_re, E_s_im = r_s_re * tx_s, r_s_im * tx_s
    E_p_re, E_p_im = r_p_re * tx_p, r_p_im * tx_p
    tau_angle = _sigmoid((cos_i - 0.94) * 20.0)
    kl = K_WAVE * lc
    v_KA = _sigmoid((kl - 6.0) * 2.0)
    kh = K_WAVE * sh
    v_SPM = _sigmoid((0.3 - kh) * 10.0)
    w_KA = tau_angle * tau_base[:, None] * v_KA[:, None]
    w_SPM = (1.0 - tau_angle) * (1.0 - tau_base[:, None]) * v_SPM[:, None]
    tau_eff = (w_KA / (w_KA + w_SPM).clamp(min=1e-6)).clamp(0.01, 0.99)

    cos_o_raw = (wo * normals[:, None, :]).sum(-1)
    cos_o = cos_o_raw.abs().clamp(min=1e-6)
    dOmega = areas[:, None]
    tan_sq_o = (1.0 / cos_o.clamp(min=1e-6) ** 2) - 1.0
    lambda_o = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_o).clamp(min=0.0))) / 2.0
    alpha_rx = torch.sqrt((G_rx * dOmega).clamp(min=0.0))
    return (eps_real, eps_imag, thickness, alpha_sq, kappa_SPM, norm_SPM, eps_factor,
            cos_i, n_eff, wi_r, lambda_i, s_in, E_s_re, E_s_im, E_p_re, E_p_im, tau_eff,
            cos_o, lambda_o, alpha_rx)


def _slab_fresnel_real(eps_real, eps_imag, cos_theta_i, thickness):
    """`itu_slab_fresnel`'s R_TE and R_TM in real arithmetic (re, im pairs).

    The complex expressions written out so the compiler fuses them: eta =
    eps_real - j eps_imag; a = sqrt(eta - sin^2); r = (c - a) / (c + a),
    r_tm = (eta c - a) / (eta c + a); q = k d a; R = r (1 - e^{-2jq}) /
    (1 - r^2 e^{-2jq}), with the same 1e-10 regularisers.
    """
    cos_i = cos_theta_i.clamp(1e-6, 1.0)
    sin_sq = 1.0 - cos_i ** 2
    eta_re, eta_im = eps_real, -eps_imag
    # principal complex sqrt of (eta_re - sin_sq) + j eta_im, in the
    # cancellation-free form: t = sqrt((|z| + |re|) / 2), the other part
    # im / (2 t) (the naive sqrt((|z| - re) / 2) loses the imaginary part
    # at small eps_imag: 4% errors in float32)
    zr, zi = eta_re - sin_sq, eta_im
    mod = torch.sqrt(zr * zr + zi * zi)
    t = torch.sqrt(((mod + zr.abs()) / 2.0).clamp(min=1e-30))
    sign_zi = torch.where(zi < 0, -torch.ones_like(zi), torch.ones_like(zi))
    a_re = torch.where(zr >= 0, t, zi.abs() / (2.0 * t))
    a_im = torch.where(zr >= 0, zi / (2.0 * t), sign_zi * t)
    c = cos_i
    def cdiv(nr, ni, dr, di):
        # Smith's scaled division, as torch's complex division does it. The
        # branch not taken is evaluated too (torch.where), so its divisor is
        # replaced by one where it would be zero: an inf or nan there would
        # poison the gradient (nan * 0) even though the value is discarded.
        big = dr.abs() >= di.abs()
        one = torch.ones_like(dr)
        dr_s = torch.where(big, dr, one)
        di_s = torch.where(big, one, di)
        r1 = di / dr_s
        den1 = dr_s + di * r1
        re1, im1 = (nr + ni * r1) / den1, (ni - nr * r1) / den1
        r2 = dr / di_s
        den2 = di_s + dr * r2
        re2, im2 = (nr * r2 + ni) / den2, (ni * r2 - nr) / den2
        return torch.where(big, re1, re2), torch.where(big, im1, im2)
    r_te_re, r_te_im = cdiv(c - a_re, -a_im, c + a_re + 1e-10, a_im)
    ec_re, ec_im = eta_re * c, eta_im * c
    r_tm_re, r_tm_im = cdiv(ec_re - a_re, ec_im - a_im, ec_re + a_re + 1e-10, ec_im + a_im)
    k = TWO_PI / WAVELENGTH
    q_re, q_im = k * thickness * a_re, k * thickness * a_im
    # exp(-2j q) = exp(2 q_im) (cos(2 q_re) - j sin(2 q_re))
    mag = torch.exp(2.0 * q_im)
    e2_re, e2_im = mag * torch.cos(2.0 * q_re), -mag * torch.sin(2.0 * q_re)
    def slab(r_re, r_im):
        r2_re, r2_im = r_re * r_re - r_im * r_im, 2.0 * r_re * r_im
        num_re, num_im = r_re * (1.0 - e2_re) + r_im * e2_im, r_im * (1.0 - e2_re) - r_re * e2_im
        den_re = 1.0 - (r2_re * e2_re - r2_im * e2_im) + 1e-10
        den_im = -(r2_re * e2_im + r2_im * e2_re)
        return cdiv(num_re, num_im, den_re, den_im)
    R_TE_re, R_TE_im = slab(r_te_re, r_te_im)
    R_TM_re, R_TM_im = slab(r_tm_re, r_tm_im)
    return R_TE_re, R_TE_im, R_TM_re, R_TM_im


def _amplitude_eager(f_cos, alpha_tx, alpha_rx, cos_i, cos_o, shadow_mask, C_radar: float):
    """Step 5's amplitude: the BSDF's cosine-weighted value into the path weight, masked."""
    sqrt_f_cos = torch.sqrt(f_cos.clamp(min=1e-20))
    w_full = C_radar * sqrt_f_cos * alpha_tx.unsqueeze(-1) * alpha_rx.unsqueeze(-2)
    active_i = cos_i > 1e-6
    if shadow_mask is not None:
        active_i = active_i & shadow_mask
    active_o = cos_o > 1e-6
    active = active_i.unsqueeze(-1) & active_o.unsqueeze(-2)
    return w_full * active.float()


# The cached-path prelude and amplitude, compiled: HUPR_COMPILED_BSDF=0
# restores the inline eager chain (bitwise the reference; the compiled one
# is at the fp32 floor of it).
_COMPILED_BSDF = os.environ.get("HUPR_COMPILED_BSDF", "1") != "0"
_prelude = (torch.compile(_prelude_eager, dynamic=True) if _COMPILED_BSDF else _prelude_eager)
_amplitude = (torch.compile(_amplitude_eager, dynamic=True) if _COMPILED_BSDF else _amplitude_eager)


def _cached_components(w_full, geometry_cache: dict, carrier_only: bool) -> dict:
    """The cached path's components from the step's amplitudes.

    The BSDF amplitude is the only thing that moved; every other factor of
    a contribution -- the carrier phasor, the PSF weights, the destination
    indices -- is the cached constant. `_splat_contribs` is the same
    function graph the uncached path (and so the fill pass) runs, on the
    stored values, so at equal parameters the components are bitwise
    identical. ``carrier_only`` returns the carrier, the fractional bins and
    the PSF table instead (the fused scene and sorted Doppler routes).
    """
    if carrier_only:
        return {
            'carrier': _carrier(w_full.reshape(-1), geometry_cache['cos_phi'],
                                geometry_cache['sin_phi']),
            'sorted': geometry_cache.get('sorted'),
            'n_frac': geometry_cache['n_frac'],
            'psf_table_real': geometry_cache['psf_table_real'],
            'psf_table_imag': geometry_cache['psf_table_imag'],
            'flat_indices': geometry_cache['flat_indices'],
            'point_indices': geometry_cache['point_indices'],
            'tx_indices': geometry_cache['tx_indices'],
            'rx_indices': geometry_cache['rx_indices'],
            'output_shape': geometry_cache['output_shape'],
            # the trainer's row geometry enumerates every (point, tx, rx) in
            # row-major order: `forward_render._path_rates_of` reads the rates
            # as a view then
            'paths_are_rows': bool(geometry_cache.get('paths_are_rows', False)),
        }
    if 'psf_real' in geometry_cache:
        psf_r = geometry_cache['psf_real']
        psf_i = geometry_cache['psf_imag']
    else:
        # Slim cache: re-lerp the PSF from the cached fractional bins -- the
        # same `_psf_lerp` graph the fill pass ran on the same inputs, so
        # the values are bitwise the fill's.
        psf_r, psf_i = _psf_lerp(
            geometry_cache['psf_table_real'],
            geometry_cache['psf_table_imag'],
            geometry_cache['n_frac'], geometry_cache['n_grid'])
    contrib_real_all, contrib_imag_all = _splat_contribs(
        w_full.reshape(-1),
        geometry_cache['cos_phi'], geometry_cache['sin_phi'],
        psf_r, psf_i)
    return {
        'contrib_real': contrib_real_all,
        'contrib_imag': contrib_imag_all,
        'flat_indices': geometry_cache['flat_indices'],
        'point_indices': geometry_cache['point_indices'],
        'tx_indices': geometry_cache['tx_indices'],
        'rx_indices': geometry_cache['rx_indices'],
        'output_shape': geometry_cache['output_shape'],
        'paths_are_rows': bool(geometry_cache.get('paths_are_rows', False)),
    }


def render_factorized(
    positions,          # (M, 3) Gaussian centers
    normals,            # (M, 3) surface normals
    areas,              # (M,) area weights (vertex_area × opacity)
    raw_materials,      # (M, 6) raw material params
    rast,               # RasterizerTorch instance (for constants + antenna)
    reparameterize_fn,  # reparameterize_torch function
    detach_phase=True,
    shadow_mask=None,   # (M, n_tx) bool — False = occluded, zero weight
    bsdf_mode='full',   # 'full' (default), or 'scalar' (B1 baseline)
    disabled_components=None,  # Phase 1.5: set of {'cbs','directive','broad','spm','ka','blend','jones','slab'}
    use_cuda_kernels=True,  # Phase B: route Step 4 through the fused CUDA kernel
    psf_spread=None,    # Hann PSF kernel half-width (None -> default 15; ablation: {5,9,15,21,25})
    return_splat_components=False,  # expose per-point complex splats before coherent summation
    tx_positions=None,  # (M, n_tx, 3) per-point TX array override; default rast.tx_positions
    rx_positions=None,  # (M, n_rx, 3) per-point RX array override; default rast.rx_positions
    geometry_cache=None,  # dict: reuse (or fill) the position-only splat geometry
    cache_psf=True,     # False: cache n_frac (~3 MB) and re-lerp the PSF per call
                        # instead of storing the (SPREAD, P) PSF pair (~86 MB) —
                        # for callers holding many caches at once. Bitwise: the
                        # recompute is the same _psf_lerp graph the fill ran.
    carrier_only=False, # a filled geometry_cache only: return the (paths, 2)
                        # carrier, n_frac and the PSF table in place of the
                        # (spread, paths) contributions, for the fused scene
                        # path that synthesises them in its kernels
):
    """Range-profile splatting renderer. Returns (rp_real, rp_imag).

    Output shape: (n_tx, n_rx, K) complex range profiles.
    These must be passed through azimuth FFT (not range FFT) to get RA image.

    Cost: M BSDF evals + M×28 antenna + M×192×5 scatter ops.

    ``geometry_cache``: with ``detach_phase=True`` and frozen ``positions``,
    ``areas`` and array geometry, everything that does not read ``normals`` or
    ``raw_materials`` is a constant — distances, antenna gains, carrier
    phases, PSF weights and every splat index. Pass an empty dict to have one
    call fill it; pass it back (non-empty) to skip recomputing those
    constants and re-evaluate only the BSDF amplitude. The cached path runs
    the *same multiplies in the same order* on the stored values, so at equal
    parameters the output is bitwise identical to the uncached call (pinned
    in the suite and by the trainer's background re-render gate). The caller
    owns the invariants: a cache must never be reused across different
    positions, areas, arrays, ``psf_spread`` or rasterizer constants.
    Requires ``return_splat_components=True`` (the trainer's cached-splat
    consumers) and refuses trainable positions, whose gradient the cached
    geometry would silently drop.
    """
    if geometry_cache is not None:
        if not detach_phase or not return_splat_components:
            raise ValueError(
                "geometry_cache requires detach_phase=True and "
                "return_splat_components=True")
        if positions.requires_grad:
            raise ValueError("geometry_cache requires frozen positions")
    device = positions.device
    M = positions.shape[0]
    n_tx = rast.n_tx
    n_rx = rast.n_rx
    K = rast.K

    if M == 0 and not return_splat_components:
        return (torch.zeros(n_tx, n_rx, K, device=device),
                torch.zeros(n_tx, n_rx, K, device=device))
    if M == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return {
            'contrib_real': torch.empty(0, 0, device=device),
            'contrib_imag': torch.empty(0, 0, device=device),
            'flat_indices': torch.empty(0, 0, dtype=torch.long, device=device),
            'point_indices': empty,
            'tx_indices': empty,
            'rx_indices': empty,
            'output_shape': (n_tx, n_rx, K),
        }

    disabled = set(disabled_components) if disabled_components else set()

    # The compiled cached path: steps 1-3 as one fused function, the step-4
    # kernel, the amplitude as another, then the cached components. Only on
    # a filled geometry cache with the full BSDF (the slab may be disabled);
    # fp32 floor against the inline chain below (HUPR_COMPILED_BSDF=0).
    if (bool(geometry_cache) and _COMPILED_BSDF and use_cuda_kernels
            and bsdf_mode == 'full' and not (disabled - {'slab'})):
        try:
            from mm4DPS import cuda_ext as _v5cuda
            _fast = _v5cuda.is_available()
        except Exception:
            _fast = False
        if _fast:
            from mm4DPS.cuda_ext import bsdf_step4_forward as _cuda_bsdf_step4
            wi_c, wo_c = geometry_cache['wi'], geometry_cache['wo']
            (eps_real, eps_imag, thickness, alpha_sq, kappa_SPM, norm_SPM, eps_factor,
             cos_i, n_eff, wi_r, lambda_i, s_in, E_s_re, E_s_im, E_p_re, E_p_im, tau_eff,
             cos_o, lambda_o, alpha_rx) = _prelude(
                raw_materials, normals, areas, wi_c, wo_c, geometry_cache['G_rx'],
                reparameterize_fn, 'slab' in disabled)
            f_cos = _cuda_bsdf_step4(
                wi_c.contiguous(), wi_r.contiguous(), wo_c.contiguous(),
                n_eff.contiguous(), s_in.contiguous(),
                cos_i.contiguous(), cos_o.contiguous(),
                lambda_i.contiguous(), lambda_o.contiguous(),
                alpha_sq.contiguous(),
                kappa_SPM.contiguous(), norm_SPM.contiguous(),
                eps_factor.contiguous(),
                eps_real.contiguous(), eps_imag.contiguous(),
                thickness.contiguous(),
                E_s_re.contiguous(), E_s_im.contiguous(),
                E_p_re.contiguous(), E_p_im.contiguous(),
                tau_eff.contiguous(),
                float(K_WAVE),
            )
            w_full = _amplitude(
                f_cos, geometry_cache['alpha_tx'], alpha_rx, cos_i, cos_o, shadow_mask,
                float(rast.radar_constant * rast.rx_dBFS_scale * rast.adc_scale))
            return _cached_components(w_full, geometry_cache, carrier_only)

    # ================================================================
    # Step 1: Per-Gaussian material prep (M evaluations)
    # ================================================================
    physics = reparameterize_fn(raw_materials)
    eps_real = physics[:, 0]
    eps_imag = physics[:, 1]
    sigma_h = physics[:, 2]
    l_c = physics[:, 3]
    tau_base = physics[:, 4]
    thickness = physics[:, 5]

    # C8: slab disable → collapse multi-layer Fresnel to single-layer by
    # forcing thickness to a tiny value. The slab phase q ≈ (2π/λ)·d·... → 0
    # so the multi-layer reflection coefficient reduces to first-surface only.
    if 'slab' in disabled:
        thickness = torch.full_like(thickness, 1e-9)

    # Fix 2 (2026-04-13): removed enforce_spm_validity. The SPM validity
    # clamps from CSVBSDF paper Eq. 19 (kh << 1, k³h²l << 1, √2 h/l < 0.3)
    # constrain sigma_h ≤ 62 μm at 77 GHz, which is "smoother than glass" —
    # too restrictive for any real automotive scene material (asphalt,
    # concrete, brick, foliage all have σ_h ≫ 62 μm). P5 audit showed 74%
    # of sigma_h points were dead-clamped under random init. Letting the
    # optimizer use sigma_h and l_c directly, even outside theoretical SPM
    # validity — the loss will tell us if the resulting scattering
    # coefficient is useful regardless of its physical interpretation.
    sh = sigma_h
    lc = l_c

    # GGX alpha for KA lobe
    alpha_raw = 4.0 * math.pi * sh / WAVELENGTH
    alpha_ggx = torch.sqrt(alpha_raw.clamp(min=0.0025)).clamp(0.05, 0.95)
    alpha_sq = alpha_ggx ** 2

    # vMF parameters for SPM lobe
    l_c_lam = lc / WAVELENGTH
    roughness_slope = sh / lc.clamp(min=1e-8)
    kappa_SPM = (torch.sqrt(l_c_lam.clamp(min=0.0)) / (1.0 + 3.0 * roughness_slope)).clamp(0.5, 10.0)
    kappa_SPM_c = kappa_SPM.clamp(max=50.0)
    sinh_SPM = (torch.exp(kappa_SPM_c) - torch.exp(-kappa_SPM_c)) / 2.0
    norm_SPM = kappa_SPM / (4.0 * math.pi * sinh_SPM.clamp(min=1e-10))

    # SPM material modulation
    eps_contrast = torch.abs(eps_real - 1.0) + eps_imag
    eps_factor = (eps_contrast / 5.0).clamp(0.2, 1.0)

    # Radar constant
    C_radar = rast.radar_constant * rast.rx_dBFS_scale * rast.adc_scale

    # ================================================================
    # Step 2: Per-TX computation (M × n_tx)
    # ================================================================
    # Distances and directions: (M, n_tx)
    #
    # `tx_positions`/`rx_positions` let one call cover several radars at once
    # by giving each point its own array geometry (HuPR's two radars differ in
    # nothing else — same patterns, boresights, chirp and constants), so the
    # radar axis folds into the point axis instead of costing a second full
    # render. Omitted, this is the rasterizer's own (1, n_tx, 3) broadcast and
    # every downstream value is bitwise what it was.
    _geometry_ready = bool(geometry_cache)
    if _geometry_ready:
        wi = geometry_cache['wi']                                     # (M, n_tx, 3) hit->TX
    else:
        tx_array = rast.tx_positions[None, :, :] if tx_positions is None else tx_positions
        diff_tx = tx_array - positions[:, None, :]                   # (M, n_tx, 3)
        d_tx = diff_tx.norm(dim=-1).clamp(min=1e-6)                  # (M, n_tx)
        wi = diff_tx / d_tx.unsqueeze(-1)                             # (M, n_tx, 3) hit->TX

    # Cosine of incidence angle (M, n_tx)
    cos_i = (wi * normals[:, None, :]).sum(-1)                        # may be negative
    # Double-sided: flip sign where cos_i < 0
    normal_sign = torch.where(cos_i < 0, -torch.ones_like(cos_i), torch.ones_like(cos_i))
    cos_i = cos_i.abs().clamp(min=1e-6)
    # Effective normals per (M, n_tx) for reflection computation
    n_eff = normals[:, None, :] * normal_sign.unsqueeze(-1)           # (M, n_tx, 3)

    # Specular reflection of wi about n_eff: wi_r = 2(wi·n)n - wi
    wi_dot_n = (wi * n_eff).sum(-1, keepdim=True)                     # (M, n_tx, 1)
    wi_r = 2.0 * wi_dot_n * n_eff - wi                               # (M, n_tx, 3)

    # Smith G1 for incidence (KA)
    tan_sq_i = (1.0 / cos_i.clamp(min=1e-6) ** 2) - 1.0
    lambda_i = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_i).clamp(min=0.0))) / 2.0

    # Jones Fresnel: wi-dependent terms
    # s/p basis from incidence direction
    # compute_sp_basis expects (*, 3) — reshape to (M*n_tx, 3)
    wi_flat = wi.reshape(-1, 3)
    n_eff_flat = n_eff.reshape(-1, 3)
    s_in_flat, p_in_flat = compute_sp_basis(wi_flat, n_eff_flat)
    s_in = s_in_flat.reshape(M, n_tx, 3)                              # (M, n_tx, 3)
    p_in = p_in_flat.reshape(M, n_tx, 3)

    # TX polarization projection
    tx_pol = _get_default_pol(device)  # (3,)
    tx_s = (tx_pol * s_in).sum(-1)                                     # (M, n_tx)
    tx_p = (tx_pol * p_in).sum(-1)                                     # (M, n_tx)

    # Fresnel coefficients (depend on cos_theta_i and materials)
    cos_i_flat = cos_i.reshape(-1)
    eps_r_flat = eps_real[:, None].expand(-1, n_tx).reshape(-1)
    eps_i_flat = eps_imag[:, None].expand(-1, n_tx).reshape(-1)
    thick_flat = thickness[:, None].expand(-1, n_tx).reshape(-1)

    R_TE, R_TM, _, _ = itu_slab_fresnel(eps_r_flat, eps_i_flat, cos_i_flat, thick_flat)
    r_s = R_TE.reshape(M, n_tx)
    r_p = R_TM.reshape(M, n_tx)

    # Reflected field (complex): E_s_out = r_s × tx_s, E_p_out = r_p × tx_p
    E_s_out = r_s * tx_s.to(torch.complex64)                          # (M, n_tx) complex
    E_p_out = r_p * tx_p.to(torch.complex64)                          # (M, n_tx) complex

    # Blend factors that depend on cos_theta_i (M, n_tx)
    tau_angle = _sigmoid((cos_i - 0.94) * 20.0)
    kl = K_WAVE * lc
    v_KA = _sigmoid((kl - 6.0) * 2.0)
    kh = K_WAVE * sh
    v_SPM = _sigmoid((0.3 - kh) * 10.0)
    w_KA = tau_angle * tau_base[:, None] * v_KA[:, None]
    w_SPM = (1.0 - tau_angle) * (1.0 - tau_base[:, None]) * v_SPM[:, None]
    tau_eff = (w_KA / (w_KA + w_SPM).clamp(min=1e-6)).clamp(0.01, 0.99)  # (M, n_tx)

    if _geometry_ready:
        alpha_tx = geometry_cache['alpha_tx']                          # (M, n_tx)
    else:
        # TX antenna gain (M, n_tx)
        tx_bore_exp = rast.tx_boresights[None, :, :].expand(M, -1, -1)  # (M, n_tx, 3)
        G_tx = rast.tx_antenna.evaluate(
            wi.reshape(-1, 3), tx_bore_exp.reshape(-1, 3)
        ).reshape(M, n_tx)

        # TX amplitude factor: sqrt(G_tx) / d_tx
        alpha_tx = torch.sqrt(G_tx.clamp(min=0.0)) / d_tx.clamp(min=1e-4)  # (M, n_tx)

        # TX phase factors
        tau_tx = d_tx / C_LIGHT                                        # (M, n_tx)
        phi_tx_const = TWO_PI * rast.center_freq * tau_tx              # (M, n_tx)
        phi_tx_slope = TWO_PI * rast.slope * tau_tx                    # (M, n_tx)

    # ================================================================
    # Step 3: Per-RX computation (M × n_rx)
    # ================================================================
    if _geometry_ready:
        dir_hit_to_rx = geometry_cache['wo']                           # (M, n_rx, 3) hit->RX
    else:
        rx_array = rast.rx_positions[None, :, :] if rx_positions is None else rx_positions
        diff_rx = positions[:, None, :] - rx_array                     # (M, n_rx, 3)
        d_rx = diff_rx.norm(dim=-1).clamp(min=1e-6)                    # (M, n_rx)
        dir_hit_to_rx = -diff_rx / d_rx.unsqueeze(-1)                  # (M, n_rx, 3) hit->RX
        dir_rx_to_hit = diff_rx / d_rx.unsqueeze(-1)                   # (M, n_rx, 3) RX->hit

    # cos(theta_o) for BSDF (Smith masking, Cook-Torrance denominator, back-face culling)
    cos_o_raw = (dir_hit_to_rx * normals[:, None, :]).sum(-1)          # (M, n_rx)
    cos_o = cos_o_raw.abs().clamp(min=1e-6)                            # (M, n_rx) double-sided

    # Hemisphere solid angle: density-ratio MC weight (p/q × opacity)
    # or surface integral weight (A × cos_o / d_rx²) — areas carries either
    dOmega = areas[:, None]                                            # (M, n_rx) broadcast

    # Smith G1 for observation (KA)
    tan_sq_o = (1.0 / cos_o.clamp(min=1e-6) ** 2) - 1.0
    lambda_o = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_o).clamp(min=0.0))) / 2.0

    # RX Jones: p_out depends on wo
    # s_out = s_in (same for all RX, but varies per TX)
    # p_out = cross(s_out, wo) — varies per (TX, RX)
    # We'll handle this in the inner loop (Step 4)

    # RX polarization projections: rx_s = rx_pol · s_out (same as s_in, per TX)
    rx_pol = _get_default_pol(device)
    # rx_s depends on s_in which varies per TX — computed in inner loop
    # But s_in is the same across RX for a given TX, so rx_s is per-TX only

    if _geometry_ready:
        # G_rx is the position-only constant; `areas` may move between
        # calls (the person's gains are trainable), so alpha_rx is
        # recomputed with the same expression the fill pass used — bitwise
        # at equal areas, and carrying areas' gradient when they train.
        G_rx = geometry_cache['G_rx']                                  # (M, n_rx)
        alpha_rx = torch.sqrt((G_rx * dOmega).clamp(min=0.0))          # (M, n_rx)
    else:
        # RX antenna gain (M, n_rx)
        rx_bore_exp = rast.rx_boresights[None, :, :].expand(M, -1, -1)
        G_rx = rast.rx_antenna.evaluate(
            dir_rx_to_hit.reshape(-1, 3), rx_bore_exp.reshape(-1, 3)
        ).reshape(M, n_rx)

        # RX amplitude factor: sqrt(G_rx × dΩ)
        alpha_rx = torch.sqrt((G_rx * dOmega).clamp(min=0.0))         # (M, n_rx)

        # RX phase factors
        tau_rx = d_rx / C_LIGHT                                        # (M, n_rx)
        phi_rx_const = TWO_PI * rast.center_freq * tau_rx              # (M, n_rx)
        phi_rx_slope = TWO_PI * rast.slope * tau_rx                    # (M, n_rx)

    # ================================================================
    # Step 4: Per-(TX, RX) lightweight BSDF inner loop
    # Compute f_total(t,r) for all M × n_tx × n_rx using precomputed terms
    # ================================================================

    # wo directions: (M, n_rx, 3) — dir_hit_to_rx
    wo = dir_hit_to_rx  # (M, n_rx, 3)

    # Phase B: route Step 4 through the fused CUDA kernel when available.
    # Falls back to PyTorch on any of:
    #   - extension not built
    #   - bsdf_mode != 'full' (scalar mode or anything exotic)
    #   - any component in `disabled` (jones/ka/spm ablations bypass this)
    if getattr(render_factorized, '_debug_capture', False):
        render_factorized._debug_tensors = {
            name: value for name, value in locals().items()
            if name in ('wi', 'wo', 'n_eff', 's_in', 'p_in', 'cos_i', 'cos_o',
                        'lambda_i', 'lambda_o', 'alpha_sq', 'kappa_SPM',
                        'norm_SPM', 'eps_factor', 'eps_real', 'eps_imag',
                        'thickness', 'E_s_out', 'E_p_out', 'tau_eff')
        }
    _use_cuda = use_cuda_kernels and bsdf_mode == 'full' and not disabled
    if _use_cuda:
        try:
            from mm4DPS import cuda_ext as _v5cuda
            _use_cuda = _v5cuda.is_available()
        except Exception:
            _use_cuda = False

    if _use_cuda:
        from mm4DPS.cuda_ext import bsdf_step4_forward as _cuda_bsdf_step4
        # Read the wave number from the module, not the frozen from-import:
        # set_frequency() rebinds K_WAVE at runtime with `global` and the slab
        # interference phase must follow the true carrier. Reading the module
        # global here is that same late binding, now that the two are one module.
        # wi_r = 2*(wi·n_eff)*n_eff - wi  (precomputed per TX, matches the
        # PyTorch Step 2 block earlier in this function).
        wi_dot_n = (wi * n_eff).sum(-1, keepdim=True)
        wi_r = 2.0 * wi_dot_n * n_eff - wi
        f_cos = _cuda_bsdf_step4(
            wi.contiguous(), wi_r.contiguous(), wo.contiguous(),
            n_eff.contiguous(), s_in.contiguous(),
            cos_i.contiguous(), cos_o.contiguous(),
            lambda_i.contiguous(), lambda_o.contiguous(),
            alpha_sq.contiguous(),
            kappa_SPM.contiguous(), norm_SPM.contiguous(),
            eps_factor.contiguous(),
            eps_real.contiguous(), eps_imag.contiguous(),
            thickness.contiguous(),
            E_s_out.real.contiguous(), E_s_out.imag.contiguous(),
            E_p_out.real.contiguous(), E_p_out.imag.contiguous(),
            tau_eff.contiguous(),
            float(K_WAVE),
        )
    else:

        # We need f_bsdf(t,r) for each (M, n_tx, n_rx).
        # Precomputed per-TX: wi_r (M, n_tx, 3), E_s_out/E_p_out (M, n_tx) complex,
        #   tau_eff (M, n_tx), lambda_i (M, n_tx), s_in (M, n_tx, 3), cos_i (M, n_tx)
        # Precomputed per-RX: cos_o (M, n_rx), lambda_o (M, n_rx), wo (M, n_rx, 3)
        # Precomputed per-Gaussian: alpha_sq (M,), kappa_SPM (M,), norm_SPM (M,),
        #   eps_factor (M,)

        # --- KA lobe: need half vector h = norm(wo + wi), then h·n ---
        # wi: (M, n_tx, 3), wo: (M, n_rx, 3)
        # h: (M, n_tx, n_rx, 3) — this is the 192× expansion, but only for h·n (scalar)
        # h·n = (wo + wi)·n / |wo + wi|
        # Numerator: wo·n + wi·n = cos_o_signed + cos_i_signed
        # But we need to use the TX-specific flipped normal for wi·n consistency.
        #
        # More carefully: h = normalize(wo + wi). h·n depends on the actual vectors.
        # We can compute h·n without forming the full (M, n_tx, n_rx, 3) tensor:
        #
        # (wo + wi)·n = wo·n + wi·n
        # |wo + wi|² = |wo|² + |wi|² + 2(wo·wi) = 2 + 2(wo·wi)  (unit vectors)
        # |wo + wi| = sqrt(2 + 2 wo·wi)
        #
        # wo·wi: (M, n_tx, n_rx) via einsum
        wo_dot_wi = torch.einsum('mrj,mtj->mtr', wo, wi)                  # (M, n_tx, n_rx)

        # wo·n_eff and wi·n_eff (using TX-flipped normals)
        # wi·n_eff = cos_i (already computed, (M, n_tx))
        # wo·n_eff: need per (M, n_tx, n_rx) since n_eff varies per TX
        wo_dot_n = torch.einsum('mrj,mtj->mtr', wo, n_eff)                # (M, n_tx, n_rx)

        h_dot_n_num = wo_dot_n + cos_i.unsqueeze(-1)                      # (M, n_tx, n_rx)
        h_len = torch.sqrt((2.0 + 2.0 * wo_dot_wi).clamp(min=1e-10))     # (M, n_tx, n_rx)
        h_dot_n = (h_dot_n_num / h_len).clamp(min=0.0)                    # (M, n_tx, n_rx)

        # GGX NDF: D = α² / (π ((h·n)²(α²-1)+1)²)
        denom_ndf = (h_dot_n ** 2 * (alpha_sq[:, None, None] - 1.0) + 1.0) ** 2
        D_KA = alpha_sq[:, None, None] / (math.pi * denom_ndf.clamp(min=1e-20))

        # Smith G (joint form, exact): G = 1 / (1 + Λ_i + Λ_o)
        G_KA = 1.0 / (1.0 + lambda_i.unsqueeze(-1) + lambda_o.unsqueeze(-2)).clamp(min=1e-10)

        # Cook-Torrance: f_KA = D × G / (4 cos_i cos_o)
        f_KA = D_KA * G_KA / (4.0 * cos_i.unsqueeze(-1) * cos_o.unsqueeze(-2)).clamp(min=1e-10)
        if 'ka' in disabled:
            f_KA = torch.zeros_like(f_KA)

        # --- SPM lobe: cos_dev = wo · wi_r ---
        cos_dev = torch.einsum('mrj,mtj->mtr', wo, wi_r)                  # (M, n_tx, n_rx)
        cos_dev = cos_dev.clamp(-1.0, 1.0)
        f_SPM = norm_SPM[:, None, None] * torch.exp(kappa_SPM[:, None, None] * (cos_dev - 1.0))
        f_SPM = f_SPM * eps_factor[:, None, None]
        if 'spm' in disabled:
            f_SPM = torch.zeros_like(f_SPM)

        # --- Jones Fresnel power per (TX, RX) ---
        # Macro-normal RX projection basis (shared by KA and SPM). See note on
        # the half-vector refinement below.
        rx_s = (rx_pol * s_in).sum(-1)                                     # (M, n_tx) -- same for all RX

        # p_out = cross(s_in, wo) -- varies per (TX, RX)
        # Triple product: rx_pol · (s_in × wo) = wo · (rx_pol × s_in)
        rx_pol_cross_s = torch.cross(
            rx_pol.expand_as(s_in), s_in, dim=-1)                          # (M, n_tx, 3)
        rx_p = torch.einsum('mrj,mtj->mtr', wo, rx_pol_cross_s)           # (M, n_tx, n_rx)
        # Normalize by |cross(s_in, wo)| = sin(angle)
        s_cross_wo_sq = 1.0 - torch.einsum('mrj,mtj->mtr', wo, s_in) ** 2
        p_out_norm = torch.sqrt(s_cross_wo_sq.clamp(min=1e-12))
        rx_p = rx_p / p_out_norm                                           # (M, n_tx, n_rx)

        # --- R_jones_macro: macro-normal Fresnel, used for SPM lobe ---
        # E_s_out / E_p_out are from Fresnel at cos_i = wi · n (macro-normal).
        E_rx = E_s_out.unsqueeze(-1) * rx_s.unsqueeze(-1).to(torch.complex64) \
             + E_p_out.unsqueeze(-1) * rx_p.to(torch.complex64)           # (M, n_tx, n_rx) complex
        R_jones_macro = _cpx_abs_sq(E_rx).clamp(0.0, 1.0)                 # (M, n_tx, n_rx)

        # --- R_jones_h: full microfacet-correct Jones for the KA lobe (1A'') ---
        # Walter et al. 2007, "Microfacet Models for Refraction through Rough
        # Surfaces": for a GGX microfacet BSDF, the specular reflection at a
        # path (wi, wo) happens at a microfacet whose normal is the half-vector
        # h = normalize(wi + wo). The local incidence angle, polarization basis,
        # and Fresnel coefficients should all be computed relative to h, not
        # the macro normal n.
        #
        # This block computes a fully microfacet-correct Jones for the KA
        # (coherent specular) lobe: per-path s/p polarization basis (s_h,
        # p_h_in, p_h_out), per-path microfacet Fresnel at cos_h, and per-path
        # recombination through the microfacet basis. The result R_jones_h is
        # exact for any GGX roughness and any TX/RX array geometry (monostatic,
        # compact bistatic, wide bistatic, distributed arrays).
        #
        # At the flat-surface limit (α → 0, h → n), s_h → s_in and the result
        # reduces to the macro Jones R_jones_macro. For moderate roughness
        # (α ~ 0.4) and compact arrays, the correction is small but
        # physically consistent.
        #
        # The SPM lobe continues to use R_jones_macro (incoherent average over
        # microfacets uses the mean-surface / macro-normal Fresnel).
        #
        # Memory: 4 new (M, n_tx, n_rx, 3) float32 tensors ≈ 400 MB at
        # M=45K, 192 channels. Scales linearly with channel count.

        # Per-path microfacet geometry
        wi_exp = wi.unsqueeze(2).expand(-1, -1, n_rx, -1).contiguous()     # (M, n_tx, n_rx, 3)
        wo_exp = wo.unsqueeze(1).expand(-1, n_tx, -1, -1).contiguous()     # (M, n_tx, n_rx, 3)

        h_vec = wi_exp + wo_exp
        h_norm = h_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        h_vec = h_vec / h_norm                                              # (M, n_tx, n_rx, 3)
        cos_h = (wi_exp * h_vec).sum(-1).clamp(min=1e-6)                   # (M, n_tx, n_rx)

        # Microfacet s-basis: perpendicular to the microfacet plane of
        # incidence (containing wi and h). Unit vector.
        # Note: s_h ⊥ wi by construction, so downstream cross products with
        # wi or wo are well-behaved (nonzero magnitude).
        s_h_raw = torch.cross(wi_exp, h_vec, dim=-1)
        s_h_norm = s_h_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        s_h_vec = s_h_raw / s_h_norm                                        # (M, n_tx, n_rx, 3)

        # Microfacet p-basis for the incoming wave: in the plane of incidence,
        # perpendicular to wi. Since s_h ⊥ wi, |s_h × wi| = 1 exactly at the
        # limit where s_h is a unit vector perpendicular to wi, so the norm
        # below is ~1 up to numerical noise. Still normalize for safety.
        p_h_in_raw = torch.cross(s_h_vec, wi_exp, dim=-1)
        p_h_in = p_h_in_raw / p_h_in_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Microfacet p-basis for the outgoing wave: in the plane of incidence,
        # perpendicular to wo. For bistatic arrays, wo is NOT parallel to the
        # reflection of wi about h in general (only at the specific microfacet
        # that satisfies the reflection law, which gives wo = reflect(wi, h)).
        # At that specific microfacet, wo is in the (wi, h) plane, so p_h_out
        # is well-defined and is perpendicular to wo.
        p_h_out_raw = torch.cross(s_h_vec, wo_exp, dim=-1)
        p_h_out = p_h_out_raw / p_h_out_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Project TX and RX polarization vectors onto microfacet basis
        tx_pol_b = tx_pol.view(1, 1, 1, 3)
        rx_pol_b = rx_pol.view(1, 1, 1, 3)

        tx_s_h = (tx_pol_b * s_h_vec).sum(-1)                              # (M, n_tx, n_rx)
        tx_p_h_in = (tx_pol_b * p_h_in).sum(-1)                            # (M, n_tx, n_rx)

        rx_s_h = (rx_pol_b * s_h_vec).sum(-1)                              # (M, n_tx, n_rx)
        rx_p_h_out = (rx_pol_b * p_h_out).sum(-1)                          # (M, n_tx, n_rx)

        # Per-path slab Fresnel at the microfacet incidence angle cos_h
        cos_h_flat = cos_h.reshape(-1)
        eps_r_flat_h = eps_real[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
        eps_i_flat_h = eps_imag[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
        thick_flat_h = thickness[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
        R_TE_h, R_TM_h, _, _ = itu_slab_fresnel(
            eps_r_flat_h, eps_i_flat_h, cos_h_flat, thick_flat_h)
        r_s_h = R_TE_h.reshape(M, n_tx, n_rx)                               # (M, n_tx, n_rx) complex
        r_p_h = R_TM_h.reshape(M, n_tx, n_rx)                               # (M, n_tx, n_rx) complex

        # Reflected field components in microfacet basis
        E_s_out_h = r_s_h * tx_s_h.to(torch.complex64)                     # (M, n_tx, n_rx) complex
        E_p_out_h = r_p_h * tx_p_h_in.to(torch.complex64)                  # (M, n_tx, n_rx) complex

        # Project onto RX polarization in microfacet outgoing basis
        E_rx_h = (E_s_out_h * rx_s_h.to(torch.complex64)
                  + E_p_out_h * rx_p_h_out.to(torch.complex64))             # (M, n_tx, n_rx) complex
        R_jones_h = _cpx_abs_sq(E_rx_h).clamp(0.0, 1.0)                    # (M, n_tx, n_rx)

        # Jones disable path: replace BOTH Fresnel versions with scalar |r|².
        # For the microfacet path, use the per-path r_s_h / r_p_h values.
        if 'jones' in disabled:
            Rs_sq = _cpx_abs_sq(E_s_out).clamp(0.0, 1.0)
            Rp_sq = _cpx_abs_sq(E_p_out).clamp(0.0, 1.0)
            R_scalar = 0.5 * (Rs_sq + Rp_sq)
            R_jones_macro = R_scalar.unsqueeze(-1).expand(-1, -1, n_rx)
            Rs_sq_h = _cpx_abs_sq(r_s_h).clamp(0.0, 1.0)
            Rp_sq_h = _cpx_abs_sq(r_p_h).clamp(0.0, 1.0)
            R_jones_h = 0.5 * (Rs_sq_h + Rp_sq_h)

        # --- Final BSDF: per-lobe Fresnel ---
        # KA uses half-vector Fresnel (microfacet-correct).
        # SPM uses macro-normal Fresnel (incoherent average over microfacets).
        if bsdf_mode == 'scalar':
            # B1 baseline: f_cos = sigmoid(reflectivity) × cos_i
            rho = torch.sigmoid(raw_materials[:, 0])
            f_cos = rho[:, None, None] * cos_i.unsqueeze(-1)
            f_cos = f_cos.expand(-1, -1, n_rx).contiguous()
        else:
            tau_eff_ext = tau_eff.unsqueeze(-1)                            # (M, n_tx, 1)
            f_coh = tau_eff_ext * R_jones_h * f_KA + (1.0 - tau_eff_ext) * R_jones_macro * f_SPM
            f_cos = f_coh * cos_i.unsqueeze(-1)                            # (M, n_tx, n_rx)
            # Verification mode may run the BSDF in fp64; the splat stage
            # stays fp32 either way.
            f_cos = f_cos.float()

    # ================================================================
    # Step 5: Range-profile splatting
    # ================================================================
    # Instead of computing cos/sin for K=256 ADC samples per path,
    # deposit each Gaussian's amplitude at ~5 range bins per channel.

    # Weight per (M, n_tx, n_rx)
    sqrt_f_cos = torch.sqrt(f_cos.clamp(min=1e-20))
    w_full = C_radar * sqrt_f_cos * alpha_tx.unsqueeze(-1) * alpha_rx.unsqueeze(-2)

    # Zero out back-facing and occluded paths
    active_i = cos_i > 1e-6
    if shadow_mask is not None:
        active_i = active_i & shadow_mask
    active_o = cos_o > 1e-6
    active = active_i.unsqueeze(-1) & active_o.unsqueeze(-2)
    w_full = w_full * active.float()

    if _geometry_ready:
        return _cached_components(w_full, geometry_cache, carrier_only)

    # Range bin for each (M, n_tx, n_rx) path:
    # n_peak = (d_tx + d_rx) × S × K / (c × fs)
    #        = phi_slope_total / (2π × fs / K)  [since phi_slope = 2πSτ and τ = d/c]
    # Or equivalently: n_peak = (d_tx + d_rx) / (2 × range_res)
    # where range_res = c / (2 × BW) = c × fs / (2 × S × K)
    d_total = d_tx.unsqueeze(-1) + d_rx.unsqueeze(-2)          # (M, n_tx, n_rx)
    range_res = C_LIGHT * rast.sample_rate / (2.0 * rast.slope * K)
    n_peak = d_total / (2.0 * range_res)                        # (M, n_tx, n_rx) real-valued

    # Carrier phase: 2π × f0 × τ  (encodes azimuth information)
    phi_carrier = phi_tx_const.unsqueeze(-1) + phi_rx_const.unsqueeze(-2)  # (M, n_tx, n_rx)

    if detach_phase:
        n_peak = n_peak.detach()
        phi_carrier = phi_carrier.detach()

    # Splat to range bins with Hann PSF (precomputed lookup table).
    # `psf_spread` (Tier-2 ablation axis) overrides the default kernel
    # half-width L. Even values produce L+1 bins due to the asymmetry of
    # arange(-(L//2), L//2+1) below, so callers should keep L odd.
    SPREAD = 15 if psf_spread is None else int(psf_spread)

    # Lazy-init PSF table (created once, reused across calls)
    if not hasattr(render_factorized, '_psf_table') or \
       render_factorized._psf_table.K != K or \
       render_factorized._psf_table.spread != SPREAD or \
       render_factorized._psf_table.device != str(device):
        render_factorized._psf_table = HannPSFTable(K, SPREAD, n_grid=1024, device=str(device))
    psf_table = render_factorized._psf_table

    # Phase D: fused CUDA Step-5 kernel — avoids materializing the
    # (SPREAD, M·n_tx·n_rx) contrib tensors (~540 MB at target_n=90K).
    # Falls back to PyTorch scatter_add when the extension isn't loaded
    # or use_cuda_kernels is False.
    _use_cuda_step5 = (
        use_cuda_kernels and detach_phase and not return_splat_components
    )  # fused bwd assumes detached phase and performs the final coherent sum
    if _use_cuda_step5:
        try:
            from mm4DPS import cuda_ext as _v5cuda
            _use_cuda_step5 = _v5cuda.is_available()
        except Exception:
            _use_cuda_step5 = False

    if _use_cuda_step5:
        from mm4DPS.cuda_ext import step5_fused as _cuda_step5
        rp_real, rp_imag = _cuda_step5(
            w_full.contiguous(),
            phi_carrier.contiguous(),
            n_peak.contiguous(),
            psf_table.psf_real,
            psf_table.psf_imag,
            K,
            1e-20,
        )
        return rp_real, rp_imag

    n_floor = n_peak.floor().long()
    n_frac = n_peak - n_floor.float()

    # Flatten all (M, n_tx, n_rx) paths
    w_flat = w_full.reshape(-1)
    phi_flat = phi_carrier.reshape(-1)
    n_floor_flat = n_floor.reshape(-1)
    n_frac_flat = n_frac.reshape(-1)

    # E3 (runtime): no path compaction. Zero the inactive paths' weight in
    # place and keep the regular (M, n_tx, n_rx) layout.
    #
    # Compaction was `w_flat > 1e-20` followed by a `nonzero` and eight
    # `index_select` gathers. Measured over 253 slow-time calls, the active
    # fraction is 0.999993–1.000000 — it discarded at most 7 paths per
    # million — and it is vacuous *by construction*, not by luck of this
    # data: the renderer is double-sided, so `cos_i` and `cos_o` have already
    # been `.abs().clamp(min=1e-6)`-ed by line 511 and the "active" test can
    # only fail on an exactly grazing path. `shadow_mask` is None at every
    # call site in the HuPR trainer.
    #
    # Exactly value-preserving. For an inactive path `w` becomes exactly 0.0,
    # so its carrier is 0.0 and every one of its contributions is 0.0 — it
    # scatters nothing, which is precisely what compaction achieved. For an
    # active path `w * 1.0` is `w` bitwise. The backward matches too: the
    # mask is a constant, so gradients are zeroed on exactly the paths
    # `index_select`'s backward left at zero.
    #
    # What it buys: the `nonzero` was the last data-dependent shape and the
    # last CPU<->GPU round trip in the training step, and removing it makes
    # every tensor shape in the step static — the precondition for CUDA-graph
    # capture (E4a).
    # The masking itself lives in `_splat_contribs` (it is idempotent: a
    # masked weight re-masked is bitwise itself), so both the cached and
    # the uncached path apply it through the same function graph.
    w_act = w_flat
    phi_act = phi_flat
    n_floor_act = n_floor_flat
    n_frac_act = n_frac_flat

    # Channel and point indices for active paths.
    # These depend ONLY on (M, n_tx, n_rx) — all static
    # for the lifetime of the training loop. Cache on the function.
    cache_key = (M, n_tx, n_rx, str(device))
    if not hasattr(render_factorized, '_idx_cache') or \
       render_factorized._idx_cache.get('key') != cache_key:
        t_idx_full = torch.arange(n_tx, device=device).unsqueeze(0).unsqueeze(-1).expand(M, -1, n_rx).reshape(-1)
        r_idx_full = torch.arange(n_rx, device=device).unsqueeze(0).unsqueeze(0).expand(M, n_tx, -1).reshape(-1)
        m_idx_full = torch.arange(M, device=device).unsqueeze(-1).unsqueeze(-1).expand(-1, n_tx, n_rx).reshape(-1)
        render_factorized._idx_cache = {
            'key': cache_key,
            't_idx_full': t_idx_full,
            'r_idx_full': r_idx_full,
            'm_idx_full': m_idx_full,
        }
    t_idx_full = render_factorized._idx_cache['t_idx_full']
    r_idx_full = render_factorized._idx_cache['r_idx_full']
    m_idx_full = render_factorized._idx_cache['m_idx_full']
    t_idx_act = t_idx_full
    r_idx_act = r_idx_full
    base_idx = t_idx_act * (n_rx * K) + r_idx_act * K

    # Carrier phase trig, materialized: the fill pass stores exactly these
    # tensors, so the cached path's `_splat_contribs` sees the same inputs
    # the uncached path's does.
    cos_phi = torch.cos(phi_act)
    sin_phi = torch.sin(phi_act)

    # PSF lookup (table interpolation — no trig at runtime)
    if _COMPILED_STEP5:
        psf_r, psf_i = _psf_lerp(psf_table.psf_real, psf_table.psf_imag,
                                 n_frac_act, psf_table.n_grid)
    else:
        psf_r, psf_i = psf_table.evaluate(n_frac_act)             # (SPREAD, P_active)

    # Bin indices: (SPREAD, P_active)
    # dn_offsets depends only on SPREAD; cache on the function.
    if not hasattr(render_factorized, '_dn_offsets_cache') or \
       render_factorized._dn_offsets_cache.get('key') != (SPREAD, str(device)):
        render_factorized._dn_offsets_cache = {
            'key': (SPREAD, str(device)),
            'dn_offsets': torch.arange(-(SPREAD // 2), SPREAD // 2 + 1, device=device),
        }
    dn_offsets = render_factorized._dn_offsets_cache['dn_offsets']
    bin_all = (n_floor_act[None, :] + dn_offsets[:, None]) % K
    flat_idx_all = base_idx[None, :] + bin_all

    # Contributions: masking, carrier and carrier × PSF in one fused graph
    # (the same graph the cached path replays on its stored constants).
    contrib_real_all, contrib_imag_all = _splat_contribs(
        w_act, cos_phi, sin_phi, psf_r, psf_i)

    if return_splat_components:
        if geometry_cache is not None and not geometry_cache:
            # Fill pass: store the position-only constants for the next call.
            # `phi_act` is detached (geometry_cache enforces detach_phase) and
            # the PSF weights come from a table lookup on the detached
            # fractional bin, so nothing here carries gradient.
            geometry_cache.update({
                'wi': wi.detach(),
                'wo': dir_hit_to_rx.detach(),
                'alpha_tx': alpha_tx.detach(),
                'G_rx': G_rx.detach(),
                'cos_phi': cos_phi.detach(),
                'sin_phi': sin_phi.detach(),
                'flat_indices': flat_idx_all,
                'point_indices': m_idx_full,
                'tx_indices': t_idx_act,
                'rx_indices': r_idx_act,
                'output_shape': (n_tx, n_rx, K),
            })
            # The fractional bins and the table always (the fused scene
            # path lerps from them in its kernels); the (SPREAD, P) PSF
            # pair only when asked.
            geometry_cache.update({
                'n_frac': n_frac_act.detach(),
                'psf_table_real': psf_table.psf_real,
                'psf_table_imag': psf_table.psf_imag,
                'n_grid': psf_table.n_grid,
            })
            if cache_psf:
                geometry_cache.update({
                    'psf_real': psf_r.detach(),
                    'psf_imag': psf_i.detach(),
                })
        # The ordinary renderer coherently sums these values immediately.  HuPR's
        # slow-time model needs the same values before summation so each surface
        # point can receive its own motion phase.  This opt-in return keeps every
        # existing caller and the fast fused path unchanged.
        return {
            'contrib_real': contrib_real_all,
            'contrib_imag': contrib_imag_all,
            'flat_indices': flat_idx_all,
            'point_indices': m_idx_full,
            'tx_indices': t_idx_act,
            'rx_indices': r_idx_act,
            'output_shape': (n_tx, n_rx, K),
        }

    # Phase D: fused CUDA scatter splat (autograd-wrapped). Falls back
    # to the PyTorch scatter_add_ path when the extension isn't loaded
    # or use_cuda_kernels is False.
    _use_cuda_scatter = use_cuda_kernels
    if _use_cuda_scatter:
        try:
            from mm4DPS import cuda_ext as _v5cuda
            _use_cuda_scatter = _v5cuda.is_available()
        except Exception:
            _use_cuda_scatter = False

    if _use_cuda_scatter:
        from mm4DPS.cuda_ext import splat_scatter as _cuda_splat_scatter
        rp_real, rp_imag = _cuda_splat_scatter(
            contrib_real_all.reshape(-1),
            contrib_imag_all.reshape(-1),
            flat_idx_all.reshape(-1),
            (n_tx, n_rx, K),
        )
    else:
        rp_real = torch.zeros(n_tx, n_rx, K, device=device)
        rp_imag = torch.zeros(n_tx, n_rx, K, device=device)
        rp_real.view(-1).scatter_add_(0, flat_idx_all.reshape(-1), contrib_real_all.reshape(-1))
        rp_imag.view(-1).scatter_add_(0, flat_idx_all.reshape(-1), contrib_imag_all.reshape(-1))

    return rp_real, rp_imag

# --------------------------------------------------------------------
# PointPrimitives and the Gaussian render
# --------------------------------------------------------------------

mi.set_variant('cuda_ad_rgb')




DEVICE = "cuda:0"

# =========================================================================
# Learnable parameter toggles (matches the differentiable renderer's LEARN_* pattern in train.py)
# =========================================================================
# Each flag controls (a) whether the underlying parameter has requires_grad
# and (b) whether it appears in the optimizer's param_groups. Flip a flag,
# and the parameter is frozen / unfrozen uniformly.
LEARN_POSITIONS = False    # frozen — pcl positions are treated as fixed reference
LEARN_NORMALS   = True     # quaternion → surface normal; refines noisy pcl normals
LEARN_MATERIALS = True     # 6 ITU/Cook-Torrance params per point

# When True (default after the raw-MSE plan), skip the differentiable renderer-trained antenna
# pattern injection at init and use factory patterns from the MMWCAS .npy
# files instead. the differentiable renderer-trained patterns absorb per-scene amplitude bias and
# break the cross-scene scale uniformity required for raw MSE. See
# md/v4_unnormalized_mse_loss_plan.md Phase α findings.

# Default starting per-point material (ITU concrete).
# sigma_h lowered from 1e-4 (100 μm) to 5e-5 (50 μm) to stay inside the SPM
# validity regime (h_max_1 = 0.1/k ≈ 62 μm at 77 GHz). At 100 μm the
# enforce_spm_validity clamp routes grad through a constant, making sigma_h
# dead in backward — confirmed via gradient/drift/Fisher diagnostics.
ITU_CONCRETE = np.array([5.31, 0.0326, 5e-5, 5e-3, 0.5, 0.15], dtype=np.float32)


# =========================================================================
# Range profile -> RA image (azimuth FFT only, no range FFT)
# =========================================================================

# Module-level cache for the azimuth Hann window. Keyed by (length, device,
# complex dtype). The window depends only on the static array geometry, so
# computing it once per process is plenty.








# =========================================================================
# GPU eval helpers (replace scipy griddata + numpy corrcoef)
# =========================================================================











# =========================================================================
# Point primitive model
# =========================================================================

class PointPrimitives(torch.nn.Module):
    """Point-based primitives for radar rendering.

    Per-point state: position(3) + quaternion(4) + raw_materials(6) = 13 floats.
    Each parameter's `requires_grad` is set from the LEARN_* flags above so that
    flipping a flag uniformly freezes/unfreezes the parameter for both autograd
    and the optimizer (matches the differentiable renderer's LEARN_* convention).

    Plus 1444 global antenna pattern values (TX/RX × E/H × 361 samples), gated
    by LEARN_PATTERNS in the training loop.
    """

    def __init__(self, N, device=DEVICE):
        super().__init__()
        self.device = device
        self.positions = torch.nn.Parameter(
            torch.zeros(N, 3, device=device), requires_grad=LEARN_POSITIONS)
        self.rotations = torch.nn.Parameter(
            torch.zeros(N, 4, device=device), requires_grad=LEARN_NORMALS)
        self.raw_materials = torch.nn.Parameter(
            torch.zeros(N, 6, device=device), requires_grad=LEARN_MATERIALS)

        with torch.no_grad():
            self.rotations[:, 0] = 1.0  # identity quaternion [w,x,y,z]

    @property
    def N(self):
        return self.positions.shape[0]

    def get_normals(self):
        """(N, 3) surface normals = third column of rotation matrix."""
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        nx = 2 * (x * z + w * y)
        ny = 2 * (y * z - w * x)
        nz = 1 - 2 * (x * x + y * y)
        return torch.stack([nx, ny, nz], dim=-1)


# =========================================================================
# Geometry helpers (no mesh required — pcl carries normals directly)
# =========================================================================

def _normals_to_quaternions(normals):
    """Convert (N, 3) unit normals to (N, 4) quaternions [w,x,y,z] that
    rotate the local +z axis onto the surface normal."""
    N = len(normals)
    nrm = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    n = normals / nrm

    # axis = cross(up=(0,0,1), n) = (-n_y, n_x, 0)
    axis = np.zeros((N, 3), dtype=np.float32)
    axis[:, 0] = -n[:, 1]
    axis[:, 1] = n[:, 0]

    axis_len = np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    axis = axis / axis_len

    dot = n[:, 2].copy()
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    half_angle = angle / 2.0

    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = np.cos(half_angle)
    quats[:, 1] = axis[:, 0] * np.sin(half_angle)
    quats[:, 2] = axis[:, 1] * np.sin(half_angle)
    quats[:, 3] = axis[:, 2] * np.sin(half_angle)

    parallel = dot > 0.9999
    quats[parallel] = [1, 0, 0, 0]
    anti = dot < -0.9999
    quats[anti] = [0, 1, 0, 0]
    return quats


def _farthest_point_sampling(pts, n_samples):
    """GPU farthest-point sampling.

    Sequentially picks n_samples points such that each new pick is the
    one farthest (Euclidean) from all already-picked points. Inherently
    sequential — no parallelism across iterations possible.

    Three optimizations vs the textbook implementation:
      1. Use squared distances throughout (avoids per-iter sqrt). Argmax of
         d² is the same as argmax of d.
      2. `torch.index_select` instead of fancy indexing `pts[idx]` for the
         single-row anchor lookup. Significantly faster on CUDA — fancy
         indexing dispatches a slower kernel for variable-length gathers.
      3. Skip the explicit `min_dist[idx] = 0` after each pick. The next
         (pts - pts[idx])² computation already produces a 0 at the picked
         row, and the torch.minimum() carries it through.

    Empirical (130K → 50K, RTX 4090): textbook 2.94 s → this version 1.77 s
    (~40% faster). Sample diversity unchanged (same average pairwise
    distance, same bounding-box coverage as the textbook variant).
    """
    N = pts.shape[0]
    if n_samples >= N:
        return torch.arange(N, device=pts.device)

    selected = torch.empty(n_samples, dtype=torch.long, device=pts.device)
    selected[0] = 0
    diff = pts - pts[0:1]
    min_dist_sq = (diff * diff).sum(-1)

    for j in range(1, n_samples):
        idx = torch.argmax(min_dist_sq)
        selected[j] = idx
        anchor = torch.index_select(pts, 0, idx.unsqueeze(0))
        diff = pts - anchor
        d_sq = (diff * diff).sum(-1)
        min_dist_sq = torch.minimum(min_dist_sq, d_sq)
    return selected


# =========================================================================
# Init-time ray tracing (the only ray tracing in v4)
# =========================================================================

def _ray_test_visibility_batched(points_np, rx_center_np, mi_scene, chunk=200000):
    """Batched RX-side visibility ray test for the full point cloud.

    For each point, casts a ray from the point toward the RX array center.
    If the ray hits any mesh surface before reaching the RX, the point is
    occluded. This is the ONE ray-tracing call we keep in v4 — it filters
    the raw point cloud down to only RX-visible points before FPS.
    """
    import drjit as dr  # noqa: F401
    N = len(points_np)
    visible = np.zeros(N, dtype=bool)
    eps = 1e-3
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        pts = points_np[i:j]
        delta = rx_center_np - pts
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
        direction = delta / dist[:, None]
        origins = pts + eps * direction
        o = mi.Point3f(
            mi.Float(origins[:, 0]), mi.Float(origins[:, 1]), mi.Float(origins[:, 2]))
        d = mi.Vector3f(
            mi.Float(direction[:, 0]), mi.Float(direction[:, 1]), mi.Float(direction[:, 2]))
        rays = mi.Ray3f(o, d)
        rays.maxt = mi.Float(dist - 2 * eps)
        occluded = mi_scene.ray_test(rays)
        visible[i:j] = ~np.array(occluded)
    return visible


# =========================================================================
# FOV culling (no mesh, pure radar geometry)
# =========================================================================

def cull_gaussians(model, rast, cos_threshold=0.05, near_field_m=1.5):
    """Pre-filter Gaussians to only those visible to the radar (FOV only)."""
    if near_field_m < 0:
        raise ValueError("near_field_m must be non-negative")
    with torch.no_grad():
        positions = model.positions
        normals = model.get_normals()

        radar_center = (rast.tx_positions.mean(0) + rast.rx_positions.mean(0)) / 2
        boresight = rast.tx_boresights.mean(0)
        boresight = boresight / boresight.norm().clamp(min=1e-6)

        to_radar = radar_center - positions
        dist = to_radar.norm(dim=-1)
        to_radar_dir = to_radar / dist.clamp(min=1e-6).unsqueeze(-1)

        cos_normal = torch.abs((normals * to_radar_dir).sum(-1))

        from_radar = -to_radar_dir
        cos_bore = (from_radar * boresight).sum(-1)

        max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

        active = (
            (dist > near_field_m) &
            (dist < max_range) &
            (cos_bore > 0.0) &
            (cos_normal > cos_threshold)
        )
        return active


# =========================================================================
# Init: visible-weighted FPS from raw point cloud
# =========================================================================

def init_visible_weighted(scene, rast, target_n=90000,
                          cos_bore_min=0.1761,
                          n_intermediate=None,
                          device=DEVICE,
                          return_pool=False,
                          enable_cull=True,
                          enable_occlusion=True,
                          enable_cosine_resample=True,
                          enable_fps=True,
                          near_field_m=1.5):
    # Default: make the intermediate pool large enough that the cosine
    # importance resample step doesn't bottleneck FPS selection. The
    # n_resample = min(n_intermediate, 3*target_n) line then picks 3*target_n.
    if n_intermediate is None:
        n_intermediate = max(200000, 3 * target_n)
    """Initialize Gaussians via visible-weighted FPS over the raw point cloud.

    Pipeline:
      1. Load pcl.npy. Columns 0–2 are xyz, 3–5 are unit normals.
      2. FOV restrict by cos_bore_min and radar range.
      3. RX-side visibility ray test (the ONLY mesh use in v4 init).
      4. Cosine-hemisphere importance resample (prob ∝ cos_bore).
      5. FPS to target_n.
      6. Build the PointPrimitives model with normals from the pcl.

    Returns the PointPrimitives model. The caller is responsible for
    config/pattern_data loading.
    """
    pcl_path = rast._mesh_file.replace('scene/mesh.ply', 'scene/pcl.npy')
    if not os.path.exists(pcl_path):
        raise FileNotFoundError(
            f"pcl.npy not found at {pcl_path}. v4 requires the point cloud "
            f"(with normals in columns 3–5).")

    pcl = np.load(pcl_path)
    if pcl.shape[1] < 6:
        raise ValueError(
            f"{pcl_path} has shape {pcl.shape}; expected at least 6 columns "
            f"(xyz + unit normals).")

    xyz_full = pcl[:, :3].astype(np.float32)
    nrm_full = pcl[:, 3:6].astype(np.float32)
    nrm_full = nrm_full / np.maximum(
        np.linalg.norm(nrm_full, axis=1, keepdims=True), 1e-12)

    print(f"  [v4 init] Full pcl: {len(xyz_full)} points")

    # --- Step 1: FOV restrict ---
    # Ablation hook (Tier-1 axis 4a): `enable_cull=False` drops the azimuth
    # cone test and the range gate, keeping every point of the raw pcl that
    # passes the configurable near-field guard (kept unconditionally to
    # avoid d->0 singularities downstream).
    rx_center = rast.rx_positions.mean(dim=0).cpu().numpy()
    boresight = rast.tx_boresights.mean(dim=0).cpu().numpy()
    boresight = boresight / max(np.linalg.norm(boresight), 1e-8)

    delta = xyz_full - rx_center
    dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
    dir_to_point = delta / dist[:, None]
    cos_bore_full = (dir_to_point * boresight).sum(axis=-1)

    max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

    if near_field_m < 0:
        raise ValueError("near_field_m must be non-negative")
    if enable_cull:
        fov_mask = ((cos_bore_full > cos_bore_min) &
                    (dist > near_field_m) & (dist < max_range))
    else:
        fov_mask = dist > near_field_m
    xyz_fov = xyz_full[fov_mask]
    nrm_fov = nrm_full[fov_mask]
    cos_bore_fov = cos_bore_full[fov_mask]
    print(f"  [v4 init] After FOV: {len(xyz_fov)} points (enable_cull={enable_cull})")

    # --- Step 2: RX visibility ray tracing (the only mesh use) ---
    # Ablation hook (Tier-1 axis 4b): skip the Mitsuba ray cast, keep all
    # FOV-survivors. Saves ~2 s of init but lets back-of-wall / occluded
    # points pollute the optimisation pool.
    if enable_occlusion:
        rast.load_mi_scene()
        visible = _ray_test_visibility_batched(xyz_fov, rx_center, rast._mi_scene)
    else:
        visible = np.ones(len(xyz_fov), dtype=bool)
    xyz_vis = xyz_fov[visible]
    nrm_vis = nrm_fov[visible]
    cos_bore_vis = cos_bore_fov[visible]
    print(f"  [v4 init] After RX visibility: {len(xyz_vis)} points (enable_occlusion={enable_occlusion})")

    if len(xyz_vis) < 100:
        raise RuntimeError(
            f"v4 init: only {len(xyz_vis)} points survived visibility — "
            f"check the Mitsuba scene and FOV settings.")

    # --- Step 3: Cosine-hemisphere importance resample ---
    # Ablation hook (Tier-1 axis 4c): swap the cos_bore-proportional
    # importance weights for uniform weights (every visible point equally
    # likely to be drawn into the FPS pool).
    if enable_cosine_resample:
        weights = np.maximum(cos_bore_vis, 0.01)
    else:
        weights = np.ones_like(cos_bore_vis)
    probs = weights / weights.sum()
    n_resample = min(n_intermediate, 3 * target_n)
    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(len(xyz_vis), size=n_resample, replace=True, p=probs)
    unique_idx = np.unique(sampled_idx)
    xyz_weighted = xyz_vis[unique_idx]
    nrm_weighted = nrm_vis[unique_idx]
    print(f"  [v4 init] After cosine importance resample: {len(xyz_weighted)} points "
          f"(enable_cosine_resample={enable_cosine_resample})")

    # --- Step 4: FPS to target_n ---
    # Ablation hook (Tier-1 axis 4d): replace farthest-point sampling with
    # uniform random subsampling of the same `target_n` points (seed 42 for
    # reproducibility). Removes the spatial-coverage prior FPS provides.
    if len(xyz_weighted) > target_n:
        if enable_fps:
            pts_t = torch.from_numpy(xyz_weighted).to(device)
            sel = _farthest_point_sampling(pts_t, target_n).cpu().numpy()
        else:
            sel = rng.choice(len(xyz_weighted), size=target_n, replace=False)
        xyz = xyz_weighted[sel]
        normals = nrm_weighted[sel]
    else:
        xyz = xyz_weighted
        normals = nrm_weighted
        sel = None
    print(f"  [v4 init] After FPS: {len(xyz)} points (enable_fps={enable_fps})")

    N = len(xyz)

    # --- Step 5: Build the model ---
    model = PointPrimitives(N, device=device)
    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz).to(device))

        raw_default = inverse_reparameterize_torch(ITU_CONCRETE)
        model.raw_materials.copy_(
            torch.from_numpy(np.tile(raw_default, (N, 1))).to(device))

        quats = _normals_to_quaternions(normals)
        model.rotations.copy_(torch.from_numpy(quats).to(device))

    if return_pool:
        # Post-visibility pool (post-FOV + post-visibility ray test, NO
        # cosine resample, NO FPS). Size is typically several hundred
        # thousand points — the largest honest "radar-visible LiDAR"
        # candidate set. Used by S4 pool-kNN child spawning so children
        # are drawn from un-FPS-biased LiDAR neighbours of high-Fisher
        # parents (post-resample would bias toward cos_bore-aligned
        # points already close to the FPS-chosen training set).
        #
        # `pool_fps_sel` gives the indices into (xyz_vis, nrm_vis) that
        # map each of the target_n training slots to a real pool point.
        # Derivation:
        #   training pt i = xyz[i]
        #                 = xyz_weighted[sel[i]]   (sel: fps index → weighted)
        #                 = xyz_vis[unique_idx[sel[i]]]
        # so pool index for slot i = unique_idx[sel[i]].
        if len(xyz_weighted) > target_n:
            pool_fps_sel = unique_idx[sel].astype(np.int64)
        else:
            pool_fps_sel = unique_idx.astype(np.int64)
        return model, xyz_vis, nrm_vis, pool_fps_sel
    return model


# =========================================================================
# Renderer wrapper (uniform hemisphere weights)
# =========================================================================

def render_gaussians(model, rast, vertex_areas, active_mask=None,
                     shadow_mask=None, detach_phase=True, bsdf_mode='full',
                     disabled_components=None, positions_override=None,
                     use_cuda_kernels=True, psf_spread=None):
    """Range-profile splatting renderer wrapper.

    `vertex_areas` carries the precomputed per-point hemisphere weight
    (uniform = 1.0 for active points, 0.0 for inactive).

    `positions_override`: if provided, an (N, 3) tensor used as positions
    instead of `model.positions`. Used by MLP-A to inject pose-conditioned
    position deformations without mutating model state. Must have the
    same N as the model, before any active_mask is applied.

    Renders the full active set in one shot — no chunking. The dominant
    intermediate is the (M, n_tx, n_rx) BSDF tensor which at our scales
    (M ≈ 45K active, n_tx=12, n_rx=16, fp32) is ~35 MB. Total renderer
    working set is well under 1 GB on a 4090.
    """
    positions = positions_override if positions_override is not None else model.positions
    normals = model.get_normals()
    raw_materials = model.raw_materials

    # Point-based primitive: no opacity multiply. vertex_areas is already
    # the binary 1.0/0.0 active mask for the hemisphere weight.
    areas = vertex_areas

    sm = None
    if shadow_mask is not None and active_mask is not None:
        sm = shadow_mask[active_mask]
    elif shadow_mask is not None:
        sm = shadow_mask

    if active_mask is not None:
        positions = positions[active_mask]
        normals = normals[active_mask]
        raw_materials = raw_materials[active_mask]
        areas = areas[active_mask]

    return render_factorized(
        positions, normals, areas, raw_materials, rast,
        reparameterize_torch, detach_phase=detach_phase,
        shadow_mask=sm, bsdf_mode=bsdf_mode,
        disabled_components=disabled_components,
        use_cuda_kernels=use_cuda_kernels,
        psf_spread=psf_spread)


# =========================================================================
# Training loop (LR schedule + grad clip helpers inlined)
# =========================================================================

def get_lr_scale(iteration, warmup_iters=5, warmup_factor=0.3,
                 total_iters=None, decay_start=100, min_lr_factor=0.01):
    """Linear warmup -> cosine decay to min_lr_factor."""
    if iteration < warmup_iters:
        return warmup_factor + (1.0 - warmup_factor) * (iteration / max(warmup_iters, 1))
    if total_iters is None or iteration < decay_start:
        return 1.0
    progress = (iteration - decay_start) / max(total_iters - decay_start, 1)
    progress = min(progress, 1.0)
    return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1 + math.cos(math.pi * progress))


def rms_clip_grad(param, max_rms):
    """Per-parameter RMS gradient clipping.

    Data-dependent Python branch on `rms > max_rms`. Not CUDA-graph
    compatible — use `rms_clip_grad_graph_safe` for graph-captured
    training loops.
    """
    if param.grad is None:
        return
    g = param.grad.data
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    rms = torch.sqrt(torch.mean(g ** 2))
    if rms > max_rms:
        g.mul_(max_rms / rms)
    param.grad.data = g
