"""Frequency-differentiable ITU material model + Fresnel + FMCW point-scatter forward
model, all in PyTorch with gradients flowing to the DISPERSION COEFFICIENTS (a,b,c,d).

This is the differentiable mechanism that the in-repo path lacked: mmIR's
`itu_materials.get_itu_properties` is a non-differentiable float lookup. Here

    eps_real(f) = a * f_GHz**b
    sigma(f)    = c * f_GHz**d
    eps_imag(f) = 17.98 * sigma(f) / f_GHz     (ITU-R P.2040 conductivity->eps'')

and (a,b,c,d) are nn.Parameters, so they can be RECOVERED by back-propagating a
complex-signal loss and then analytically CONTINUED to an unseen band.

Fresnel / permittivity ported from mm3DPS/bsdf_torch.py.
"""
import math
import torch

C_LIGHT = 2.99792458e8


def eps_from_dispersion(abcd, f_ghz):
    """abcd: (...,4) [a,b,c,d]; returns complex relative permittivity eps_r - j eps_i."""
    a, b, c, d = abcd[..., 0], abcd[..., 1], abcd[..., 2], abcd[..., 3]
    eps_real = a * f_ghz ** b
    sigma = c * f_ghz ** d
    eps_imag = 17.98 * sigma / f_ghz
    return eps_real, eps_imag


def permittivity_to_ior(eps_real, eps_imag):
    """eps* = eps_r - j eps_i -> (n, kappa). (mm3DPS/bsdf_torch.py:55)"""
    eps_mag = torch.sqrt((eps_real ** 2 + eps_imag ** 2).clamp(min=1e-20))
    n = torch.sqrt(((eps_mag + eps_real) / 2.0).clamp(min=1e-20))
    kappa = torch.sqrt(((eps_mag - eps_real) / 2.0).clamp(min=0.0))
    return n, kappa


def fresnel_rs(n, kappa, cos_i):
    """Complex Fresnel s-pol reflection coefficient. (mm3DPS/bsdf_torch.py:70)"""
    cos_i = torch.as_tensor(cos_i, dtype=n.dtype).clamp(1e-6, 1.0)
    sin_sq = 1.0 - cos_i ** 2
    n_sq = torch.complex(n ** 2 - kappa ** 2, -2.0 * n * kappa)
    xi = n_sq - sin_sq.to(torch.complex64)
    sq = torch.sqrt(xi)
    cos_c = cos_i.to(torch.complex64)
    return (cos_c - sq) / (cos_c + sq + 1e-10)


def reflectivity(abcd, f_ghz, cos_i=1.0):
    """Complex material reflectivity at frequency f (per scatterer)."""
    er, ei = eps_from_dispersion(abcd, f_ghz)
    n, kap = permittivity_to_ior(er, ei)
    return fresnel_rs(n, kap, cos_i)


def render_adc(abcd, ranges, f0_ghz, S_hz_per_s, K, chirp_T, geom=None):
    """Differentiable FMCW complex ADC for a set of point scatterers.

    SFCW/FSCW (Vayyar) analogue: ``method.sfcw_forward.render_sfcw_mono`` (discrete stepped tones,
    no chirp-slope ``S*tau*t`` term; range compression via IFFT-over-freq).  Both share the
    differentiable G-ITU dispersion reflectivity below.

    abcd:    (M,4) dispersion coeffs per scatterer (grad-enabled)
    ranges:  (M,) scatterer ranges (m), KNOWN geometry (the scaffold)
    f0_ghz:  carrier (GHz);  S_hz_per_s: chirp slope;  K: ADC samples; chirp_T: s
    Returns: complex ADC (K,)   s_k = sum_m refl_m * exp(j[2pi f0 tau_m + 2pi S tau_m t_k])
    """
    f0 = f0_ghz * 1e9
    t = torch.arange(K, dtype=torch.float64, device=abcd.device) * (chirp_T / K)
    tau = (2.0 * ranges.double() / C_LIGHT)                      # (M,)
    refl = reflectivity(abcd, f0_ghz)                            # (M,) complex
    if geom is None:
        geom = 1.0 / (ranges.double() ** 2 + 1e-6)
    amp = refl * geom.to(refl.dtype)                             # (M,)
    phase = 2 * math.pi * (f0 * tau)[:, None] + 2 * math.pi * (S_hz_per_s * tau)[:, None] * t[None, :]
    s = (amp[:, None] * torch.exp(1j * phase.to(torch.complex128))).sum(0)  # (K,)
    return s.to(torch.complex64)


def range_profile(adc):
    """Range FFT magnitude (windowed)."""
    K = adc.shape[-1]
    win = torch.hann_window(K, device=adc.device, dtype=torch.float32)
    return torch.fft.rfft(adc * win.to(adc.dtype)).abs()


# --- a few ITU-R P.2040 materials (a, b, c, d) for synthetic ground truth ---
ITU_GT = {
    "concrete":     (5.24, 0.0, 0.0462, 0.7822),
    "brick":        (3.91, 0.0, 0.0238, 0.16),
    "glass":        (6.31, 0.0, 0.0036, 1.3394),
    "wood":         (1.99, 0.0, 0.0047, 1.0718),
    "plasterboard": (2.73, 0.0, 0.0085, 0.9395),
    "marble":       (7.074, 0.0, 0.0055, 0.9262),
}
