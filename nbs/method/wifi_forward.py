"""WiFi CSI/CIR forward model (bistatic), analogous to the mmWave RA model.

We render the COMPLEX channel frequency response H(subcarrier) per Tx-Rx antenna pair (and
its CIR = IFFT over subcarriers), and supervise the MAGNITUDE |CSI| only — exactly as the
mmWave path renders complex ADC and supervises |RA|. XRF55 WiFi being amplitude-only is
therefore not a blocker.

H(f, tx, rx) = sum_m amp_m(f) * geom * exp(-j 2*pi*f*(R_tx_m + R_rx_m)/c)   (single bounce)
The 30 subcarriers span ~40 MHz, so the material reflectivity amp(f) is ~constant across
them (negligible dispersion); the subcarrier PHASE carries the delay/CIR structure.
"""
import math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method.freq_material import C_LIGHT


def subcarrier_ghz(fc_ghz=5.64, bw_mhz=40.0, n_sc=30):
    df = bw_mhz * 1e6 / n_sc
    k = np.arange(n_sc) - (n_sc - 1) / 2.0
    return fc_ghz + k * df / 1e9


def real_wifi_antennas(device="cpu"):
    """XRF55 real WiFi geometry: 1 Tx (1 antenna) + 3 Rx (3 antennas each, 9 links), 5.64 GHz.
    Returns (tx_pos (1,3), rx_pos (9,3)) in the radar frame. Each Rx corner is a 3-antenna ULA
    at half-wavelength spacing (~2.66 cm at 5.64 GHz); orientation assumed along +x."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing import xrf55_config as C
    tx_c, rx_c = C.wifi_layout()
    lam = C_LIGHT / (C.WIFI_FREQ_GHZ * 1e9); d = lam / 2.0
    tx = torch.tensor([tx_c], dtype=torch.float64, device=device)
    rows = []
    for cx, cy, cz in rx_c:
        for a in range(C.WIFI_RX_ANTENNAS):
            rows.append([cx + (a - 1) * d, cy, cz])      # 3-antenna ULA along x, centered
    return tx, torch.tensor(rows, dtype=torch.float64, device=device)


def render_csi(amp, pos, tx_pos, rx_pos, sc_ghz, noise_db=None, gen=None):
    """Complex CSI (Ntx*Nrx, Nsc) from per-scatterer reflectivity amp (M,) at the WiFi band.

    amp: complex (M,) reflectivity at the WiFi centre frequency (negligible sub-band dispersion).
    pos (M,3); tx_pos (Ntx,3); rx_pos (Nrx,3); sc_ghz (Nsc,) subcarrier freqs.
    """
    dev = pos.device
    f = torch.tensor(sc_ghz, dtype=torch.float64, device=dev) * 1e9          # (Nsc,)
    Ntx, Nrx, Nsc = tx_pos.shape[0], rx_pos.shape[0], len(sc_ghz)
    H = torch.zeros(Ntx, Nrx, Nsc, dtype=torch.complex128, device=dev)
    amp = amp.to(torch.complex128)
    for it in range(Ntx):
        Rt = (pos - tx_pos[it]).norm(dim=1)
        for ir in range(Nrx):
            Rr = (pos - rx_pos[ir]).norm(dim=1)
            tau = ((Rt + Rr) / C_LIGHT)                                       # (M,)
            geom = (1.0 / (Rt * Rr + 1e-9)).to(torch.complex128)
            ph = torch.exp(-1j * 2 * math.pi * f[None, :] * tau[:, None])     # (M,Nsc)
            H[it, ir] = ((amp * geom)[:, None] * ph).sum(0)                   # (Nsc,)
    H = H.reshape(Ntx * Nrx, Nsc)
    if noise_db is not None:
        p = (H.abs() ** 2).mean()
        sig = torch.sqrt(p * 10 ** (noise_db / 10.0) / 2.0)
        nz = (torch.randn(H.shape, generator=gen, dtype=torch.float64, device=dev)
              + 1j * torch.randn(H.shape, generator=gen, dtype=torch.float64, device=dev))
        H = H + sig * nz
    return H                                                                 # complex CSI


def csi_to_cir(H):
    """CIR = IFFT over subcarriers (per antenna pair). Complex (pairs, Nsc)."""
    return torch.fft.ifft(H, dim=-1)
