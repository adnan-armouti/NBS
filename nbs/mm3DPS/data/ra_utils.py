"""Range-Azimuth (RA) map utilities for training."""

import os
import torch
import numpy as np
from scipy import interpolate
from typing import Union


def txrx_to_vx_chirps_torch(adc_data: torch.Tensor) -> torch.Tensor:
    """Torch port of txrx_to_vx_chirps.
    adc_data: complex tensor with shape (NUM_CHIRPS=1, NUM_RX=16, NUM_TX=12, NUM_ADC_SAMPLES)
    returns: complex tensor of shape (1, 7, 86, NUM_ADC_SAMPLES)
    """
    assert adc_data.ndim == 4 and adc_data.size(0) == 1, "expected (1, RX, TX, ADC)"
    NUM_RX, NUM_TX = adc_data.size(1), adc_data.size(2)
    rx_locations = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0), (14, 0), (46, 0), (47, 0), (48, 0), (49, 0), (50, 0), (51, 0), (52, 0), (53, 0)]
    tx_locations = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6), (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]
    device = adc_data.device
    dtype = adc_data.dtype
    out = torch.zeros((1, 7, 86, adc_data.size(-1)), dtype=dtype, device=device)
    filled = torch.zeros((1, 7, 86), dtype=torch.bool, device=device)
    for rx_id in range(NUM_RX):
        rx_loc = rx_locations[rx_id]
        for tx_id in range(NUM_TX):
            tx_loc = tx_locations[tx_id]
            vx_x = rx_loc[0] + tx_loc[0]
            vx_y = rx_loc[1] + tx_loc[1]
            if not filled[0, vx_y, vx_x]:
                out[:, vx_y, vx_x, :] = adc_data[:, rx_id, tx_id, :]
                filled[0, vx_y, vx_x] = True
            else:
                out[:, vx_y, vx_x, :] = (out[:, vx_y, vx_x, :] + adc_data[:, rx_id, tx_id, :]) / 2
    return out


def adc_to_ra_image(adc_ri: torch.Tensor) -> torch.Tensor:
    """Build RA magnitude image from ADC real-imag tensor using the notebook pipeline.
    adc_ri shape: 4D real-imag with last dim == 2; contains TX(12), RX(16), ADC(256) in some order.
    returns: 2D magnitude tensor [A=127, R=256] matching the notebook (after azimuth FFT and trimming first bin).
    """
    assert adc_ri.ndim == 4 and adc_ri.size(-1) == 2, "expected (..., 2) real-imag"
    dims = list(adc_ri.shape[:-1])
    idx_adc = int(dims.index(256))
    idx_tx  = int(dims.index(12))
    idx_rx  = int(dims.index(16))
    order = [idx_rx, idx_tx, idx_adc, len(dims)]
    x = adc_ri.permute(order)  # (RX, TX, ADC, 2)
    x_c = torch.complex(x[..., 0].contiguous(), x[..., 1].contiguous())  # (RX, TX, ADC)
    x_c = x_c.unsqueeze(0)  # (1, RX, TX, ADC)
    vx = txrx_to_vx_chirps_torch(x_c)  # (1, 7, 86, ADC)
    ra = vx[0, 0, :, :]  # pick elevation row 0 -> (86, ADC)
    # range window and FFT
    num_adc = ra.size(-1)
    ra = ra * torch.hann_window(num_adc, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[None, :]
    ra = torch.fft.fft(ra, n=num_adc, dim=-1)
    # azimuth window and FFT to 128 bins
    num_vx = ra.size(0)
    ra = ra * torch.hann_window(num_vx, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[:, None]
    ra = torch.fft.ifftshift(ra, dim=0)
    ra = torch.fft.fft(ra, n=128, dim=0)
    ra = ra[1:, :]  # drop first azimuth bin
    ra = torch.fft.fftshift(ra, dim=0)
    return ra.abs().float()


def adc_to_ra_complex(adc_ri: torch.Tensor) -> torch.Tensor:
    """Build complex RA image from ADC real-imag tensor (for gradient-based training).

    Same as adc_to_ra_image but returns COMPLEX values (not magnitude) to enable
    loss computation on both real and imaginary components.

    adc_ri shape: 4D real-imag with last dim == 2; contains TX(12), RX(16), ADC(256) in some order.
    returns: 2D complex tensor [A=127, R=256] (complex64)
    """
    assert adc_ri.ndim == 4 and adc_ri.size(-1) == 2, "expected (..., 2) real-imag"
    dims = list(adc_ri.shape[:-1])
    idx_adc = int(dims.index(256))
    idx_tx  = int(dims.index(12))
    idx_rx  = int(dims.index(16))
    order = [idx_rx, idx_tx, idx_adc, len(dims)]
    x = adc_ri.permute(order)  # (RX, TX, ADC, 2)
    x_c = torch.complex(x[..., 0].contiguous(), x[..., 1].contiguous())  # (RX, TX, ADC)
    x_c = x_c.unsqueeze(0)  # (1, RX, TX, ADC)
    vx = txrx_to_vx_chirps_torch(x_c)  # (1, 7, 86, ADC)
    ra = vx[0, 0, :, :]  # pick elevation row 0 -> (86, ADC)
    # range window and FFT
    num_adc = ra.size(-1)
    ra = ra * torch.hann_window(num_adc, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[None, :]
    ra = torch.fft.fft(ra, n=num_adc, dim=-1)
    # azimuth window and FFT to 128 bins
    num_vx = ra.size(0)
    ra = ra * torch.hann_window(num_vx, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[:, None]
    ra = torch.fft.ifftshift(ra, dim=0)
    ra = torch.fft.fft(ra, n=128, dim=0)
    ra = ra[1:, :]  # drop first azimuth bin
    ra = torch.fft.fftshift(ra, dim=0)
    # Return COMPLEX values (not magnitude) for loss on real + imag
    return ra  # Complex tensor (127, 256)
# ===================================================================
#  Single-chip radar (IWR6843: 3 Tx, 4 Rx) — XRF55 geometry
#  Azimuth RA uses the two NON-elevated Tx (0 and 2) x 4 Rx = 8 virtual antennas,
#  a uniform 0.5*lambda line (Tx1 is elevated -> excluded). Matches mmIR
#  antenna_layout_single_chip and the XRF55 single-chip radar (64 azimuth bins).
# ===================================================================
SINGLE_CHIP_AZ_TX = (0, 2)


def single_chip_to_vx(adc_c: torch.Tensor) -> torch.Tensor:
    """adc_c complex (RX=4, TX=3, ADC) or (CHIRP, RX, TX, ADC) -> ordered 8-vx ULA (8, ADC).

    Virtual antenna x-position = tx_x + rx_x; with Tx{0,2} (spaced 2*lambda) x 4 Rx
    (spaced 0.5*lambda) this yields 8 antennas at uniform 0.5*lambda spacing, ordered
    Tx0.Rx0..3, Tx2.Rx0..3.
    """
    if adc_c.ndim == 4:
        adc_c = adc_c[0]
    nrx = adc_c.size(0)
    rows = [adc_c[r, t, :] for t in SINGLE_CHIP_AZ_TX for r in range(nrx)]
    return torch.stack(rows, dim=0)            # (8, ADC)


def adc_to_ra_complex_single_chip(adc_c: torch.Tensor, n_az: int = 64) -> torch.Tensor:
    """Complex RA (n_az, ADC) for the single-chip radar (XRF55). Mirrors
    adc_to_ra_complex (range FFT -> azimuth FFT) on the 8-element azimuth ULA.

    adc_c: complex (RX=4, TX=3, ADC) or (CHIRP, RX, TX, ADC).
    """
    vx = single_chip_to_vx(adc_c)              # (8, ADC) complex
    K = vx.size(-1)
    vx = vx * torch.hann_window(K, device=vx.device, dtype=vx.real.dtype).to(vx.dtype)[None, :]
    rng = torch.fft.fft(vx, n=K, dim=-1)       # range FFT
    nvx = vx.size(0)
    rng = rng * torch.hann_window(nvx, device=rng.device, dtype=rng.real.dtype).to(rng.dtype)[:, None]
    rng = torch.fft.ifftshift(rng, dim=0)
    ra = torch.fft.fft(rng, n=n_az, dim=0)     # azimuth FFT
    ra = torch.fft.fftshift(ra, dim=0)
    return ra                                  # complex (n_az, K)


# ===================================================================
#  Vayyar SFCW radar (IMAGEVK-74: 20 Tx, 20 Rx) — OctoNet geometry
#  L-shaped 2-D array: Tx along azimuth (+X), Rx along elevation (+Z); the co-array
#  vx[i,j] = tx_i + rx_j fills a 2-D azimuth x elevation aperture. For the 2-D
#  range-azimuth target we marginalize elevation (coherent sum over Rx = the
#  elevation-broadside cut), then take an azimuth FFT over the 20 Tx. Range comes
#  from the SFCW IFFT-over-frequency (already applied by the renderer's Step-5 splat,
#  or via sfcw_range_compress for raw vayyar_pickle data). No range-FFT-over-ADC.
# ===================================================================

def sfcw_range_compress(H, window: bool = True):
    """Raw Vayyar per-pair channel response H (..., n_freq) -> complex range profile (..., n_freq)
    via a Hann-windowed IFFT over the frequency-step axis. Mirrors
    method.sfcw_forward.sfcw_to_range; use on real vayyar_pickle data before RA formation
    (the renderer already outputs range-compressed profiles)."""
    Nf = H.shape[-1]
    if window:
        w = torch.hann_window(Nf, periodic=False, device=H.device, dtype=H.real.dtype).to(H.dtype)
        H = H * w[..., :]
    return torch.fft.ifft(H, dim=-1)


def vayyar_pairs_to_az_ula(rp: torch.Tensor) -> torch.Tensor:
    """rp complex (n_tx, n_rx, K) range profiles -> azimuth ULA (n_tx, K) by coherently summing
    over the Rx (elevation) axis — i.e. the elevation-broadside (el=0) cut of the co-array."""
    if rp.ndim != 3:
        raise ValueError("expected (n_tx, n_rx, K)")
    return rp.sum(dim=1)                        # (n_tx, K) complex


def vayyar_to_ra_complex(rp: torch.Tensor, n_az: int = 128) -> torch.Tensor:
    """Complex RA (n_az, K) from Vayyar range profiles rp (n_tx, n_rx, K). Azimuth FFT over the
    Tx (X) axis after marginalizing elevation. Mirrors adc_to_ra_complex_single_chip but with
    NO range FFT (rp is already range-compressed by the SFCW renderer / sfcw_range_compress).
    The full 3-D az x el x range volume is available via a 2-D FFT over the co-array if needed."""
    vx = vayyar_pairs_to_az_ula(rp)            # (n_tx, K) complex
    nvx = vx.size(0)
    vx = vx * torch.hann_window(nvx, device=vx.device, dtype=vx.real.dtype).to(vx.dtype)[:, None]
    vx = torch.fft.ifftshift(vx, dim=0)
    ra = torch.fft.fft(vx, n=n_az, dim=0)      # azimuth FFT
    ra = torch.fft.fftshift(ra, dim=0)
    return ra                                  # complex (n_az, K)


# TX and RX locations for virtual array mapping (matches inspect_ra_coir.py)
RX_LOCATIONS = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0), (14, 0),
                (46, 0), (47, 0), (48, 0), (49, 0), (50, 0), (51, 0), (52, 0), (53, 0)]
TX_LOCATIONS = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6),
                (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]


def txrx_to_virtual_array_numpy(adc_txrx: np.ndarray) -> np.ndarray:
    """Convert TX/RX format ADC data to virtual array format (NumPy version).

    Matches the implementation in inspect_ra_coir.py exactly.

    Args:
        adc_txrx: Complex array of shape (TX=12, RX=16, ADC=256)

    Returns:
        np.ndarray: Virtual array of shape (86, 256) - elevation 0 only
    """
    NUM_TX, NUM_RX = 12, 16
    NUM_ADC = adc_txrx.shape[-1]

    # Create output arrays for full virtual array (7, 86, ADC)
    vx_full = np.zeros((7, 86, NUM_ADC), dtype=np.complex128)
    filled = np.zeros((7, 86), dtype=bool)

    for rx_id in range(NUM_RX):
        rx_loc = RX_LOCATIONS[rx_id]
        for tx_id in range(NUM_TX):
            tx_loc = TX_LOCATIONS[tx_id]
            vx_x = rx_loc[0] + tx_loc[0]  # azimuth (0-85)
            vx_y = rx_loc[1] + tx_loc[1]  # elevation (0-6)

            if not filled[vx_y, vx_x]:
                vx_full[vx_y, vx_x, :] = adc_txrx[tx_id, rx_id, :]
                filled[vx_y, vx_x] = True
            else:
                # Average with existing value (same as ra_utils.py)
                vx_full[vx_y, vx_x, :] = (vx_full[vx_y, vx_x, :] + adc_txrx[tx_id, rx_id, :]) / 2

    # Return elevation 0 only: (86, 256)
    return vx_full[0, :, :]


def virtual_array_to_ra_polar_numpy(vx_data: np.ndarray) -> np.ndarray:
    """Convert virtual array data to RA polar magnitude image (NumPy version).

    Matches the implementation in inspect_ra_coir.py exactly.

    Args:
        vx_data: Complex array of shape (86, 256) - azimuth x ADC samples

    Returns:
        np.ndarray: RA polar magnitude of shape (127, 256)
    """
    ra = vx_data.copy()
    num_adc = ra.shape[1]
    num_vx = ra.shape[0]

    # Range window and FFT
    hann_range = np.hanning(num_adc)
    ra = ra * hann_range[np.newaxis, :]
    ra = np.fft.fft(ra, n=num_adc, axis=1)

    # Azimuth window and FFT to 128 bins
    hann_az = np.hanning(num_vx)
    ra = ra * hann_az[:, np.newaxis]
    ra = np.fft.ifftshift(ra, axes=0)
    ra = np.fft.fft(ra, n=128, axis=0)
    ra = ra[1:, :]  # drop first azimuth bin -> (127, 256)
    ra = np.fft.fftshift(ra, axes=0)

    return np.abs(ra).astype(np.float32)


def adc_to_ra_image_numpy(adc_ri: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Build RA magnitude image from ADC real-imag tensor using NumPy FFT.

    This function matches the inspect_ra_coir.py pipeline exactly for consistent
    visualization between training and reference images.

    Args:
        adc_ri: ADC data of shape (TX=12, RX=16, ADC=256, 2) real-imag format
                Can be torch.Tensor or np.ndarray

    Returns:
        np.ndarray: RA polar magnitude of shape (127, 256)
    """
    # Convert to numpy if torch tensor
    if isinstance(adc_ri, torch.Tensor):
        adc_ri = adc_ri.detach().cpu().numpy()

    # Convert real-imag to complex
    adc_complex = adc_ri[..., 0] + 1j * adc_ri[..., 1]  # (TX, RX, ADC) complex

    # Convert to virtual array
    vx_data = txrx_to_virtual_array_numpy(adc_complex)  # (86, 256) complex

    # Convert to RA polar magnitude
    ra_polar = virtual_array_to_ra_polar_numpy(vx_data)  # (127, 256) float

    return ra_polar


def compute_cartesian_ra_metrics(ra_rendered_cart: np.ndarray, ra_gt_cart: np.ndarray) -> dict:
    """Compute all RA Cartesian image metrics with independent min-max normalization.

    This is the canonical metric computation used by both training and evaluation.
    Both inputs are 2D float arrays (Cartesian RA magnitude images).

    Returns dict with keys: cart_corr, cart_psnr, cart_ssim, cart_mse, cart_rmse
    """
    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-30:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    gt_norm = _minmax(ra_gt_cart)
    rend_norm = _minmax(ra_rendered_cart)

    gt_flat = gt_norm.ravel()
    rend_flat = rend_norm.ravel()

    cart_corr = float(np.corrcoef(gt_flat, rend_flat)[0, 1])
    if np.isnan(cart_corr):
        cart_corr = 0.0

    cart_mse = float(np.mean((rend_norm - gt_norm) ** 2))
    cart_rmse = float(np.sqrt(cart_mse))
    cart_psnr = float(10.0 * np.log10(1.0 / cart_mse)) if cart_mse > 0 else float("inf")

    try:
        from skimage.metrics import structural_similarity as ssim_fn
        cart_ssim = float(ssim_fn(gt_norm, rend_norm, data_range=1.0))
    except ImportError:
        cart_ssim = None

    return {
        "cart_corr": cart_corr,
        "cart_psnr": cart_psnr,
        "cart_ssim": cart_ssim,
        "cart_mse": cart_mse,
        "cart_rmse": cart_rmse,
    }


def save_ra_cartesian_png(
    ra_cart: np.ndarray,
    output_path: str,
    range_res: float = 0.0557,
    scale: str = "dB",
    title: str = "",
    cmap: str = "hot",
    db_floor: float = -40.0,
    dpi: int = 100,
):
    """Save a single RA Cartesian image as .png (linear or dB scale).

    Args:
        ra_cart: 2D Cartesian RA magnitude array (rows=range, cols=azimuth).
        output_path: Path to save PNG.
        range_res: Range resolution in meters (for axis extents).
        scale: 'dB' or 'linear'.
        title: Figure title.
        cmap: Colormap name.
        db_floor: dB floor for dB scale.
        dpi: Output resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ra = np.asarray(ra_cart, dtype=np.float64)
    n_range, n_az = ra.shape
    range_depth = n_range * range_res
    range_width = range_depth / 2.0
    extent = [-range_width, range_width, 0.0, range_depth]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

    if scale == "dB":
        ra_max = np.max(ra) if np.max(ra) > 0 else 1.0
        ra_ratio = np.clip(ra / ra_max, 1e-30, None)
        ra_disp = np.clip(20.0 * np.log10(ra_ratio), db_floor, 0.0)
        im = ax.imshow(ra_disp, cmap=cmap, aspect="auto", origin="lower",
                        vmin=db_floor, vmax=0.0, extent=extent)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("dB")
    else:
        mn, mx = ra.min(), ra.max()
        ra_disp = (ra - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(ra)
        im = ax.imshow(ra_disp, cmap=cmap, aspect="auto", origin="lower",
                        vmin=0, vmax=1, extent=extent)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xlabel("Azimuth (m)")
    ax.set_ylabel("Range (m)")
    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def clip_and_normalize_ra(ra_gt: torch.Tensor, ra_pr: torch.Tensor, percentile: float | None) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Clip both RA tensors to shared percentile and normalize to [0,1].
    Returns (ra_gt_c, ra_pr_c, vmax) where vmax is the clipping value, or None.
    """
    if not percentile or percentile <= 0:
        return ra_gt, ra_pr, None
    with torch.no_grad():
        vals = torch.cat([ra_gt.reshape(-1), ra_pr.reshape(-1)])
        v = vals.detach().float().cpu().numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(v, float(percentile)))
            if vmax <= 0:
                vmax = float(np.max(v)) if np.max(v) > 0 else 1.0
    ra_gt_c = torch.clamp(ra_gt, 0, vmax) / vmax
    ra_pr_c = torch.clamp(ra_pr, 0, vmax) / vmax
    return ra_gt_c, ra_pr_c, vmax



def ra_polar_to_cartesian(ra_az_by_range: np.ndarray, range_res: float,
                          grid_res: int = 400, range_bias: float = 0.0) -> np.ndarray:
    """Convert RA map from polar (Az x Range) to cartesian grid (Y,X) like plot_range_azimuth_heatmap_img."""
    # Expect shape (azimuth_bins, range_bins)
    data = np.asarray(ra_az_by_range, dtype=np.float32)
    num_angle_bins = int(data.shape[0]) + 1
    num_adc        = int(data.shape[1])
    data = data.T  # now (range, azimuth)

    # angle/range axes
    t = np.arange(-num_angle_bins//2 + 1, num_angle_bins//2) * (2.0 / float(num_angle_bins))
    t = np.arcsin(t)
    r = np.arange(num_adc, dtype=np.float32) * float(range_res)

    range_depth = float(num_adc) * float(range_res)
    range_width = range_depth / 2.0
    xi = np.linspace(-range_width,  range_width, grid_res)
    yi = np.linspace(0.0,          range_depth, grid_res)
    xi, yi = np.meshgrid(xi, yi)

    x = r[:, None] * np.sin(t)
    y = r[:, None] * np.cos(t) - float(range_bias)

    zi = interpolate.griddata((x.ravel(), y.ravel()), data.ravel(), (xi, yi), method='linear')
    # Drop last row/col to mirror reference and remove potential extrapolated boundary
    zi = zi[:-1, :-1]
    # Replace NaNs with zeros to avoid blank renders
    zi = np.nan_to_num(zi, nan=0.0, posinf=0.0, neginf=0.0)
    # Flip to match plotting in reference (top becomes bottom, bottom becomes top)
    return zi[:, ::-1]


def save_ra_image(ra_2d: Union[torch.Tensor, np.ndarray], out_path: str, cmap: str = "plasma") -> None:
    """Save a 2D RA map (torch.Tensor or np.ndarray) to image using a non-interactive backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if isinstance(ra_2d, torch.Tensor):
            arr = ra_2d.detach().cpu().numpy()
        else:
            arr = np.asarray(ra_2d)
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(1, 1, 1)
        ax.imshow(arr, cmap=cmap, origin="lower", aspect="auto")
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        # Fallback to numpy save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path + ".npy", ra_2d.detach().cpu().numpy() if isinstance(ra_2d, torch.Tensor) else ra_2d)










