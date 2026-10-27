"""Single-chip (IWR6843 / XRF55) multi-antenna FMCW RA forward model using the REAL
3DPS BSDF and the repo's single-chip RA pipeline.

Hardens the G-ITU gate beyond the single-antenna toy:
  - reflectivity = real ITU-R P.2040 slab Fresnel (`bsdf_torch.itu_slab_fresnel`),
    frequency-dependent via eps(f) (dispersion) AND the slab phase
    (`bsdf_torch.set_frequency`), plus the real Rayleigh roughness term;
  - the actual single-chip array geometry (mmIR `antenna_layout_single_chip`: 3 Tx, 4 Rx;
    azimuth ULA = Tx{0,2} x 4 Rx = 8 antennas at 0.5*lambda) -> the inter-antenna phase
    encodes azimuth, so |RA| carries the RELATIVE inter-antenna phase (the NBS claim);
  - RA formed by the repo's `adc_to_ra_complex_single_chip` (range FFT -> azimuth FFT);
  - optional complex measurement noise.

Geometry (scatterer positions/normals, array) is KNOWN (the scaffold). Only the
per-scatterer MATERIAL (dispersion a,b,c,d; thickness; roughness) is recovered.
"""
import math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm3DPS import bsdf_torch as B
from mm3DPS.data.ra_utils import adc_to_ra_complex_single_chip
from method.freq_material import eps_from_dispersion, C_LIGHT

LAMBDA_REF_M = C_LIGHT / 77e9        # design wavelength (physical array spacing is fixed)


def single_chip_layout_m(device="cpu"):
    """Physical Rx/Tx positions (meters, 3D x=azimuth,y=boresight=0,z=elevation),
    from mmIR antenna_layout_single_chip at the 77 GHz design wavelength."""
    A1, C1 = 0.5, 1.0
    lam_mm = LAMBDA_REF_M * 1000.0
    B1 = 3.724 / lam_mm
    rx = np.array([(i * A1, 0.0) for i in range(4)], float)
    sx = 4 * A1 + B1
    tx = np.array([(sx + i * C1, 0.0) for i in range(3)], float)
    tx[1, 1] += A1                                   # Tx1 elevated
    allp = np.vstack([rx, tx])
    ctr = (allp.min(0) + allp.max(0)) / 2.0
    rel = (allp - ctr) * LAMBDA_REF_M                # (x_azimuth, z_elevation), meters
    rx_m, tx_m = rel[:4], rel[4:]
    def to3d(p):
        return torch.tensor(np.stack([p[:, 0], np.zeros(len(p)), p[:, 1]], 1),
                            dtype=torch.float64, device=device)
    return to3d(rx_m), to3d(tx_m)


def _amp_from_eps(er, ei, thickness, sigma_h, pos, normals, f0_ghz):
    """Per-scatterer complex reflectivity from the real ITU slab Fresnel + roughness."""
    B.set_frequency(f0_ghz * 1e9)
    lam = C_LIGHT / (f0_ghz * 1e9)
    inc = -pos / pos.norm(dim=1, keepdim=True)        # scatterer -> radar origin
    cos_t = (inc * normals).sum(1).clamp(1e-3, 1.0)
    R_TE, _, _, _ = B.itu_slab_fresnel(er.float(), ei.float(), cos_t.float(), thickness.float())
    rough = torch.exp(-0.5 * (4 * math.pi * sigma_h * cos_t / lam) ** 2)
    return (R_TE * rough.to(R_TE.dtype)).to(torch.complex128)   # (M,)


def _render_ra(amp, pos, f0_ghz, S, K, chirp_T, n_az, noise_db, gen, device):
    rx3, tx3 = single_chip_layout_m(device)
    f0 = f0_ghz * 1e9
    t = torch.arange(K, dtype=torch.float64, device=device) * (chirp_T / K)
    adc = torch.zeros(4, 3, K, dtype=torch.complex128, device=device)
    for j in range(3):
        Rtx = (pos - tx3[j]).norm(dim=1)              # (M,)
        for i in range(4):
            Rrx = (pos - rx3[i]).norm(dim=1)
            tau = (Rtx + Rrx) / C_LIGHT               # (M,)
            geom = 1.0 / (Rtx * Rrx + 1e-9)
            phase = 2 * math.pi * (f0 * tau)[:, None] + 2 * math.pi * (S * tau)[:, None] * t[None, :]
            adc[i, j] = ((amp * geom.to(amp.dtype))[:, None] * torch.exp(1j * phase)).sum(0)
    if noise_db is not None:
        p = (adc.abs() ** 2).mean()
        sig = torch.sqrt(p * 10 ** (noise_db / 10.0) / 2.0)
        nz = (torch.randn(adc.shape, generator=gen, dtype=torch.float64, device=device)
              + 1j * torch.randn(adc.shape, generator=gen, dtype=torch.float64, device=device))
        adc = adc + sig * nz
    return adc_to_ra_complex_single_chip(adc, n_az=n_az)         # complex (n_az, K)


def render_ra_disp(abcd, thickness, sigma_h, pos, normals, f0_ghz,
                   S=60e12, K=128, chirp_T=60e-6, n_az=64, noise_db=None, gen=None):
    """RA from the dispersion law eps_r=a*f^b, sigma=c*f^d (grad to a,b,c,d)."""
    er, ei = eps_from_dispersion(abcd, f0_ghz)
    amp = _amp_from_eps(er, ei, thickness, sigma_h, pos, normals, f0_ghz)
    return _render_ra(amp, pos, f0_ghz, S, K, chirp_T, n_az, noise_db, gen, abcd.device)


def render_ra_eps(er, ei, thickness, sigma_h, pos, normals, f0_ghz,
                  S=60e12, K=128, chirp_T=60e-6, n_az=64, noise_db=None, gen=None):
    """RA from per-band complex permittivity directly (for the interpolation baseline)."""
    amp = _amp_from_eps(er, ei, thickness, sigma_h, pos, normals, f0_ghz)
    return _render_ra(amp, pos, f0_ghz, S, K, chirp_T, n_az, noise_db, gen, er.device)


# ===================================================================
#  SFCW Vayyar (IMAGEVK-74: 20 Tx x 20 Rx) RA forward — the OctoNet analogue of the
#  single-chip FMCW path above. Same rich slab-Fresnel + roughness material model
#  (_amp_from_eps), re-evaluated PER swept tone (4 GHz dispersion), but: (a) stepped
#  discrete tones with NO chirp slope (Hofmann Eq.10, discrete f_c); (b) range via
#  IFFT-over-frequency; (c) the 20 Tx (azimuth) x 20 Rx (elevation) L-array, elevation
#  marginalized for the 2-D RA. geom_mode="monostatic" (co-located board): reflectivity
#  is the backscatter value (radar at origin), per-pair phase uses the true path.
# ===================================================================

def vayyar_layout_m(device="cpu"):
    """Vayyar 20 Tx (azimuth/+X) + 20 Rx (elevation/+Z) L-array positions (meters, 3-D),
    centered on the array phase center. Returns (rx (20,3), tx (20,3))."""
    from preprocessing.vayyar_radar_config import antenna_layout_vayyar
    rx, tx = antenna_layout_vayyar()
    return (torch.tensor(rx, dtype=torch.float64, device=device),
            torch.tensor(tx, dtype=torch.float64, device=device))


def render_vayyar_ra_disp(abcd, thickness, sigma_h, pos, normals, freqs_hz,
                          n_az=128, noise_db=None, gen=None):
    """SFCW Vayyar RA (complex, n_az x n_freq) from the dispersion law eps_r=a*f^b, sigma=c*f^d.

    Range = IFFT over the swept tones (Hann-windowed); azimuth = FFT over the 20 Tx after
    marginalizing elevation. Material (slab Fresnel + roughness) re-evaluated at every tone.
    """
    device = abcd.device
    rx3, tx3 = vayyar_layout_m(device)
    f = torch.as_tensor(np.asarray(freqs_hz, float), dtype=torch.float64, device=device)  # (Nf,)
    Nf, Ntx, Nrx = f.shape[0], tx3.shape[0], rx3.shape[0]
    # Rich material reflectivity per (M, Nf): re-evaluate the slab+roughness model at each tone.
    amp = torch.stack([
        _amp_from_eps(*eps_from_dispersion(abcd, float(f[k]) / 1e9),
                      thickness, sigma_h, pos, normals, float(f[k]) / 1e9)
        for k in range(Nf)], dim=1).to(torch.complex128)                                   # (M, Nf)
    H = torch.zeros(Ntx, Nrx, Nf, dtype=torch.complex128, device=device)
    for it in range(Ntx):
        Rt = (pos - tx3[it]).norm(dim=1).double()
        for ir in range(Nrx):
            Rr = (pos - rx3[ir]).norm(dim=1).double()
            geom = (1.0 / (Rt * Rr + 1e-9)).to(torch.complex128)[:, None]
            tau = ((Rt + Rr) / C_LIGHT)
            ph = torch.exp(-1j * 2 * math.pi * f[None, :] * tau[:, None])                   # (M,Nf)
            H[it, ir] = (amp * geom * ph).sum(0)
    if noise_db is not None:
        p = (H.abs() ** 2).mean()
        sig = torch.sqrt(p * 10 ** (noise_db / 10.0) / 2.0)
        nz = (torch.randn(H.shape, generator=gen, dtype=torch.float64, device=device)
              + 1j * torch.randn(H.shape, generator=gen, dtype=torch.float64, device=device))
        H = H + sig * nz
    from mm3DPS.data.ra_utils import sfcw_range_compress, vayyar_to_ra_complex
    rp = sfcw_range_compress(H)                                                             # (Ntx,Nrx,Nf)
    return vayyar_to_ra_complex(rp, n_az=n_az)                                              # (n_az, Nf)
