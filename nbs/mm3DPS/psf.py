"""Range PSF for a Hann-windowed length-K transform.

Applies to BOTH radar modulations, because the range point-spread is the DFT of the Hann window
either way and the pipeline takes |RA| downstream (so the FFT-vs-IFFT conjugation is immaterial):
  * FMCW  -- K-point range-FFT over fast-time ADC samples (adc_to_ra_image), K = numAdcSamples.
  * SFCW  -- K-point range-IFFT over the swept CW tones (sfcw_forward.sfcw_to_range), K = n_freq.
The splat in rasterizer_factorized Step 5 deposits into K range bins using this same kernel; only
the range-bin mapping (n_peak = d_total/(2*range_res)) differs between modulations.

The range FFT in adc_to_ra_image applies:
    1. Hann window: W[k] = 0.5 - 0.5*cos(2πk/K) for k=0..K-1
    2. FFT: F[n] = Σ_k (x[k] * W[k]) * exp(-j2πnk/K)

For a single sinusoid x[k] = exp(j*ω*k), the FFT output is a shifted
Dirichlet kernel (DFT of the Hann window). The Hann window's DFT is
the sum of 3 shifted Dirichlet kernels:
    PSF_hann(δ) = 0.5*D(δ) - 0.25*D(δ-1) - 0.25*D(δ+1)
where D(δ) = sin(πδ)/sin(πδ/K) * exp(-jπδ(K-1)/K) is the Dirichlet kernel.

For fast evaluation, we precompute the PSF on a fine grid and interpolate.
"""

import math
import torch
from torch import Tensor

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


class TemplatePSFTable:
    """PSF lookup table built from a MEASURED real range kernel (UWB: the X4 pulse as received, carrier included).

    Same interface as HannPSFTable (``evaluate(n_frac) -> (psf_real, psf_imag)`` over ``spread`` integer offsets),
    so rasterizer_factorized Step 5 can splat with it unchanged. The kernel value at bin offset d from the path's
    fractional bin position is ``template[centre + d]`` (linear interpolation); ``psf_imag`` is zero because the
    template is the real RF waveform itself -- the carrier phase of a path is then the pulse's own carrier shifted
    by the path delay, which is exactly what a time shift of the received pulse is. (The caller sets the per-path
    carrier phase to zero for this kernel: ``Rasterizer.psf_carrier_hz = 0``.)
    """

    def __init__(self, template, centre: int, n_grid: int = 1024, device: str = 'cuda:0'):
        t = torch.as_tensor(template, dtype=torch.float32, device=device).reshape(-1)
        t = t / t.abs().max().clamp(min=1e-30)
        L = int(t.shape[0]); half = min(centre, L - 1 - centre)
        self.K = None                                              # not tied to the profile length
        self.spread = 2 * half + 1
        self.n_grid = n_grid
        self.device = device
        self.template = t; self.centre = int(centre)
        frac_grid = torch.linspace(0, 1 - 1 / n_grid, n_grid, device=device)      # (n_grid,)
        dn = torch.arange(-half, half + 1, device=device).float()                  # (spread,)
        x = self.centre + dn[:, None] - frac_grid[None, :]                        # fractional template index (spread, n_grid)
        x0 = x.floor(); f = x - x0; i0 = x0.long()
        inside = (i0 >= 0) & (i0 < L - 1); i0c = i0.clamp(0, L - 2)
        vals = torch.where(inside, t[i0c] * (1 - f) + t[i0c + 1] * f, torch.zeros_like(f))
        self.psf_real = vals.contiguous()
        self.psf_imag = torch.zeros_like(vals).contiguous()

    def evaluate(self, n_frac):
        idx_f = n_frac * self.n_grid
        idx_lo = idx_f.long().clamp(0, self.n_grid - 1)
        idx_hi = (idx_lo + 1).clamp(0, self.n_grid - 1)
        frac = idx_f - idx_lo.float()
        f = frac[None, :]
        psf_r = self.psf_real[:, idx_lo] * (1 - f) + self.psf_real[:, idx_hi] * f
        psf_i = self.psf_imag[:, idx_lo] * (1 - f) + self.psf_imag[:, idx_hi] * f
        return psf_r, psf_i
