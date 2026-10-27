"""SFCW / FSCW forward model (Vayyar IMAGEVK-74) for the NBS material-recovery gates.

Stepped-frequency analogue of the FMCW ADC synthesis in ``method/freq_material.render_adc`` /
``method/ra_forward``, and the frequency-band generalization of ``method/wifi_forward.render_csi``.

Physics -- Hofmann et al., *IEEE J. Microwaves* 2025, Eq. 10, evaluated at DISCRETE carriers::

    H(f_k, Tx_i, Rx_j) = sum_m  refl_m(f_k) * geom_m * exp(-j 2*pi f_k * tau_m)

with ``f_k = f_start + k*df`` the swept CW tones (Vayyar: 63-67 GHz, 100 pts) and ``tau_m`` the ray
path delay.  Only ``f_c`` differs from FMCW (discrete steps vs a continuous chirp ramp), so there is
NO ``exp(j 2*pi S tau t)`` slope term -- see ``freq_material.render_adc`` for the FMCW form.

Two geometry modes (orthogonal to the modulation; see plan):

* ``monostatic`` (Vayyar, co-located 80 mm board ~ 17 lambda): the reflectivity/BSDF is evaluated
  ONCE per scatterer at the backscatter direction (fast -- this is what makes 3DPS >> mmIR), while
  the per-pair PHASE still uses the true path so range/angle stay accurate.  ``geom = 1/R^2``.
* ``bistatic`` (WiFi, genuinely separated Tx/Rx -- DEFERRED): reflectivity is evaluated PER Tx-Rx
  pair with distinct incidence/reflection directions (correct but O(M*Ntx*Nrx)); ``geom=1/(Rt*Rr)``.
  This is the regime ``method/wifi_forward.render_csi`` already handles.

Range compression: ``sfcw_to_range`` = IFFT over the frequency-step axis (replaces the FMCW
range-FFT-over-fast-time).  Reflectivity uses the differentiable G-ITU dispersion model in
``method/freq_material`` evaluated PER f_k, because the 4 GHz sweep spans non-negligible dispersion
(unlike WiFi's 40 MHz sub-band, where a single-frequency reflectivity suffices).
"""
import math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method.freq_material import C_LIGHT, permittivity_to_ior, fresnel_rs


def sfcw_freqs_hz(f_start=63.0e9, f_stop=67.0e9, n_freq=100):
    """The N swept CW tone frequencies (Hz): ``f_start + k*df``, ``df=(f_stop-f_start)/(n-1)``."""
    return f_start + np.arange(n_freq) * ((f_stop - f_start) / (n_freq - 1))


def _reflectivity_per_freq(abcd, f_ghz, cos_i=1.0):
    """Complex Fresnel reflectivity per scatterer PER frequency: (M,4),(Nf,) -> (M,Nf) complex.

    Broadcasts the G-ITU dispersion model (``freq_material.eps_from_dispersion``) across all swept
    tones at once.  ``cos_i`` may be a scalar, (M,), or (M,Nf)."""
    a, b, c, d = abcd[:, 0:1], abcd[:, 1:2], abcd[:, 2:3], abcd[:, 3:4]      # (M,1)
    fg = (f_ghz if torch.is_tensor(f_ghz)
          else torch.as_tensor(np.asarray(f_ghz, float))).to(
              dtype=abcd.dtype, device=abcd.device).reshape(1, -1)           # (1,Nf)
    eps_real = a * fg ** b                                                   # (M,Nf)
    sigma = c * fg ** d
    eps_imag = 17.98 * sigma / fg
    n, kap = permittivity_to_ior(eps_real, eps_imag)                        # (M,Nf)
    if not torch.is_tensor(cos_i):
        cos_i = torch.as_tensor(float(cos_i), dtype=n.dtype, device=n.device)
    if cos_i.ndim == 0:
        ci = cos_i.expand_as(n)
    elif cos_i.ndim == 1:
        ci = cos_i[:, None].expand_as(n)                                    # (M,)->(M,Nf)
    else:
        ci = cos_i
    return fresnel_rs(n, kap, ci)                                           # (M,Nf) complex


def _cos_incidence(w_in, normals):
    """cos of incidence angle between hit->source direction ``w_in`` (M,3) and ``normals`` (M,3).
    Returns 1.0 (normal incidence) when normals is None."""
    if normals is None:
        return 1.0
    c = (w_in * normals).sum(-1).abs().clamp(1e-6, 1.0)
    return c


def render_sfcw_mono(abcd, ranges, freqs_hz, cos_i=1.0, geom=None):
    """Monostatic point-scatter SFCW response ``H (Nfreq,)`` -- the direct SFCW analogue of
    ``freq_material.render_adc`` (which returns FMCW ADC (K,)).

    abcd:   (M,4) dispersion coeffs (grad-enabled).
    ranges: (M,) ONE-WAY scatterer ranges from the radar phase center (m); round-trip = 2R.
    Returns complex64 ``H`` over the swept tones (feed to ``sfcw_to_range`` for a range profile).
    """
    dev = abcd.device
    f = torch.as_tensor(np.asarray(freqs_hz, float), dtype=torch.float64, device=dev)   # (Nf,)
    R = ranges.double()
    refl = _reflectivity_per_freq(abcd, f / 1e9, cos_i).to(torch.complex128)            # (M,Nf)
    if geom is None:
        geom = 1.0 / (R ** 2 + 1e-9)                                                    # (M,)
    amp = refl * geom.to(refl.dtype)[:, None]                                           # (M,Nf)
    tau = (2.0 * R / C_LIGHT)                                                           # (M,) round-trip
    ph = torch.exp(-1j * 2 * math.pi * f[None, :] * tau[:, None])                       # (M,Nf)
    return (amp * ph).sum(0).to(torch.complex64)                                        # (Nf,)


def render_sfcw_amp(amp, ranges, freqs_hz, geom=None):
    """Monostatic SFCW response ``H (Nfreq,)`` from a PRECOMPUTED complex reflectivity ``amp``
    (skips the dispersion model).  ``amp`` may be (M,) constant-across-band or (M,Nfreq) per-tone.
    Used by the per-band-eps interpolation baseline (which recovers eps directly, not via a,b,c,d)."""
    dev = amp.device
    f = torch.as_tensor(np.asarray(freqs_hz, float), dtype=torch.float64, device=dev)   # (Nf,)
    R = ranges.double()
    amp = amp.to(torch.complex128)
    if amp.ndim == 1:
        amp = amp[:, None]                                                              # (M,1) broadcast
    if geom is None:
        geom = 1.0 / (R ** 2 + 1e-9)
    tau = (2.0 * R / C_LIGHT)                                                           # round-trip
    ph = torch.exp(-1j * 2 * math.pi * f[None, :] * tau[:, None])                       # (M,Nf)
    return (amp * geom.to(torch.complex128)[:, None] * ph).sum(0).to(torch.complex64)   # (Nf,)


def render_sfcw_pairs(abcd, pos, tx_pos, rx_pos, freqs_hz, geom_mode="monostatic",
                      normals=None, array_center=None):
    """Complex SFCW channel response ``H (Ntx, Nrx, Nfreq)`` for point scatterers, per Tx-Rx pair.

    Generalizes ``wifi_forward.render_csi`` to (a) per-frequency dispersion and (b) an explicit
    ``geom_mode`` switch:
      * ``monostatic`` -> reflectivity evaluated once per scatterer (backscatter dir), geom=1/R^2,
        true per-pair path phase exp(-j 2pi f (Rt+Rr)/c).  Fast; correct for the compact Vayyar array.
      * ``bistatic``   -> reflectivity per (M,Ntx,Nrx) with distinct incidence/reflection dirs,
        geom=1/(Rt*Rr).  Correct for separated Tx/Rx (WiFi); O(M*Ntx*Nrx).
    """
    dev = pos.device
    f = torch.as_tensor(np.asarray(freqs_hz, float), dtype=torch.float64, device=dev)   # (Nf,)
    f_ghz = f / 1e9                                                                      # (Nf,) tensor
    Ntx, Nrx, Nf = tx_pos.shape[0], rx_pos.shape[0], f.shape[0]
    H = torch.zeros(Ntx, Nrx, Nf, dtype=torch.complex128, device=dev)

    if geom_mode == "monostatic":
        c0 = (0.5 * (tx_pos.mean(0) + rx_pos.mean(0))) if array_center is None \
            else torch.as_tensor(array_center, dtype=pos.dtype, device=dev)
        d0 = (pos - c0).norm(dim=1).clamp(min=1e-9)                                      # (M,)
        w_in = (c0 - pos) / d0[:, None]                                                  # hit->center
        refl = _reflectivity_per_freq(abcd, f_ghz, _cos_incidence(w_in, normals)) \
            .to(torch.complex128)                                                        # (M,Nf)
        for it in range(Ntx):
            Rt = (pos - tx_pos[it]).norm(dim=1).double()
            for ir in range(Nrx):
                Rr = (pos - rx_pos[ir]).norm(dim=1).double()
                geom = (1.0 / (Rt * Rr + 1e-9)).to(refl.dtype)[:, None]                 # (M,1)
                tau = ((Rt + Rr) / C_LIGHT)                                              # (M,)
                ph = torch.exp(-1j * 2 * math.pi * f[None, :] * tau[:, None])            # (M,Nf)
                H[it, ir] = (refl * geom * ph).sum(0)
        return H

    if geom_mode == "bistatic":
        for it in range(Ntx):
            Rt = (pos - tx_pos[it]).norm(dim=1).double()
            w_t = (tx_pos[it] - pos) / Rt[:, None].clamp(min=1e-9)                       # hit->tx
            for ir in range(Nrx):
                Rr = (pos - rx_pos[ir]).norm(dim=1).double()
                w_r = (rx_pos[ir] - pos) / Rr[:, None].clamp(min=1e-9)                   # hit->rx
                cos_i = _cos_incidence(0.5 * (w_t + w_r), normals)                       # half-vec
                refl = _reflectivity_per_freq(abcd, f_ghz, cos_i).to(torch.complex128)   # (M,Nf)
                geom = (1.0 / (Rt * Rr + 1e-9)).to(refl.dtype)[:, None]
                tau = ((Rt + Rr) / C_LIGHT)
                ph = torch.exp(-1j * 2 * math.pi * f[None, :] * tau[:, None])
                H[it, ir] = (refl * geom * ph).sum(0)
        return H

    raise ValueError(f"unknown geom_mode {geom_mode!r} (expected 'monostatic' or 'bistatic')")


def sfcw_to_range(H, window=True):
    """Range profile = IFFT over the frequency-step axis (last dim).  Replaces the FMCW
    range-FFT-over-fast-time.  ``H`` complex (..., Nfreq) -> complex (..., Nfreq) range bins;
    bin size = c/(2*BW).  A Hann window over the sweep suppresses range sidelobes."""
    Nf = H.shape[-1]
    if window:
        w = torch.hann_window(Nf, periodic=False, device=H.device, dtype=torch.float64)
        H = H * w.to(H.dtype)
    return torch.fft.ifft(H, dim=-1)


def range_axis_m(n_freq, bw_hz):
    """The physical range (m) of each of the ``n_freq`` IFFT range bins for an SFCW sweep of
    bandwidth ``bw_hz``.  bin k -> k * c/(2*BW) (round-trip range)."""
    dr = C_LIGHT / (2.0 * bw_hz)
    return np.arange(n_freq) * dr
