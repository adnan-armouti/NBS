"""Generate the OctoNet Vayyar IMAGEVK-74 SFCW radar config json that 3DPS needs.

Mirror of ``preprocessing.xrf55_radar_config`` (which emits the TI single-chip **FMCW** config) but
for the Vayyar IMAGEVK-74 **stepped-frequency-continuous-wave (SFCW / FSCW)** imaging radar used by
OctoNet.  What changes vs FMCW:

  * No chirp.  The radar sweeps N discrete CW tones f_start..f_stop; the raw capture is already the
    per-tone complex Tx/Rx-pair channel response H(f) (OctoNet vayyar_pickle: (T,400,100) complex).
    Range compression is an **IFFT over the frequency-step axis** (see mm3DPS ``sfcw_to_range``), NOT
    a range-FFT over fast-time ADC.  So the config carries ``f_start/f_stop/n_freq`` instead of
    ``freqSlope/sampleRate/numAdcSamples/rampEndTime``.  This is exactly Hofmann et al. Eq. 10 with
    the carrier evaluated at discrete steps f_c(n)=f_start+n*df (vs FMCW's continuous ramp).

  * Array: 20 Tx + 20 Rx (400 pairs) in an L-shaped 2-D layout on an 80x80 mm PCB (aperture ~40 mm),
    vs the TI 3Tx x 4Rx.  Tx along azimuth (+X), Rx along elevation (+Z); the co-array
    ``vx = tx + rx`` fills a 2-D azimuth x elevation aperture (enables 3-D imaging; the 2-D-RA path
    marginalizes elevation).

  * ``geometry_mode = "monostatic"``: the 80 mm board is effectively co-located (~17 lambda) vs
    metre-scale scenes, so the renderer uses the fast per-scatterer backscatter approximation
    (array-manifold phase per element for angle).  Genuinely separated Tx/Rx (WiFi) would use
    ``geometry_mode="bistatic"`` (deferred).

Hardware constants: Mini-Circuits/Vayyar IMAGEVK-74 datasheet (VYYR7201-A0 SoC, 62-69 GHz band,
2-151 freq points, 5.8 dBi, EIRP -5 dBm, -10 dBm on-chip Tx, linear pol).  OctoNet uses a
63-67 GHz / 100-point / 20-Tx (LOW_RATE) profile.

FLAG: the exact per-element Tx/Rx coordinates and the 400-pair index order are approximated here as
a uniform L-array; refine from the Vayyar vSDK antenna map / the real vayyar_pickle once available.

  python -m preprocessing.vayyar_radar_config --scene Scene1
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import octonet_config as C

C_LIGHT = 299792458.0

# --- Vayyar IMAGEVK-74 SFCW hardware (OctoNet 63-67 GHz, 100-point, 20-Tx LOW_RATE profile) ---
F_START_HZ = C.VAY_F_START_HZ                 # 63 GHz
F_STOP_HZ = C.VAY_F_STOP_HZ                   # 67 GHz
N_FREQ = C.VAY_N_FREQ                         # 100 swept CW tones
F_CENTER_HZ = C.VAY_F_CENTER_HZ              # 65 GHz
BANDWIDTH_HZ = C.VAY_BW_HZ                    # 4 GHz
FREQ_STEP_HZ = C.VAY_DF_HZ                    # ~40.4 MHz
LAMBDA_M = C.VAY_LAMBDA_M                     # ~4.61 mm @ 65 GHz
RANGE_RES_M = C.RANGE_RES_M                   # c/(2B) = 0.0375 m
R_MAX_M = C.R_MAX_M                           # c/(2*df) ~ 3.71 m unambiguous

N_TX = C.VAY_N_TX                             # 20
N_RX = C.VAY_N_RX                             # 20
ARRAY_APERTURE_M = 0.04                       # ~40 mm array length (datasheet D used for ang. res.)
TX_PITCH_M = ARRAY_APERTURE_M / (N_TX - 1)    # ~2.1 mm (~0.46 lambda)  <-- FLAG: approx uniform pitch
RX_PITCH_M = ARRAY_APERTURE_M / (N_RX - 1)


def antenna_layout_vayyar():
    """L-shaped Vayyar array in the radar frame, centered on the array phase center (meters).

    Tx along azimuth (+X), Rx along elevation (+Z); the board lies in the X-Z plane with boresight
    +Y.  Returns ``(rx_rel (N_RX,3), tx_rel (N_TX,3))`` as 3-D offsets (m) from the phase center,
    matching the (rx, tx) return order of ``xrf55_radar_config.antenna_layout_single_chip``.

    FLAG: uniform-pitch approximation of the true IMAGEVK-74 L-array.  The co-array vx = tx + rx
    spans a 2-D azimuth x elevation aperture; replace with the exact vSDK per-element coordinates
    when known (only the physical positions change; the forward model is unaffected).
    """
    itx = np.arange(N_TX) - (N_TX - 1) / 2.0
    irx = np.arange(N_RX) - (N_RX - 1) / 2.0
    tx = np.zeros((N_TX, 3), float); tx[:, 0] = itx * TX_PITCH_M      # Tx along +X (azimuth)
    rx = np.zeros((N_RX, 3), float); rx[:, 2] = irx * RX_PITCH_M      # Rx along +Z (elevation)
    return rx, tx                                                     # (rx_rel, tx_rel), meters


def basis_from_boresight(b):
    """Right-handed basis: y_hat = boresight, x_hat = azimuth, z_hat = elevation (up).
    Identical convention to xrf55_radar_config.basis_from_boresight."""
    y = np.asarray(b, float); y = y / np.linalg.norm(y)
    zr = np.array([0., 0., 1.]) if abs(float(y @ [0, 0, 1])) <= 0.99 else np.array([0., 1., 0.])
    x = np.cross(y, zr); x /= np.linalg.norm(x)
    z = np.cross(x, y);  z /= np.linalg.norm(z)
    return x, y, z


def _to_world(rel, radar_pos, x, y, z):
    """Map local element offsets (cols = az->x_hat, ~0->y_hat, el->z_hat) into the world frame."""
    return np.array([radar_pos + p[0] * x + p[1] * y + p[2] * z for p in rel])


def sfcw_params():
    """SFCW sweep parameters (analogue of xrf55_radar_config.chirp_params for FMCW)."""
    return dict(f_start=F_START_HZ, f_stop=F_STOP_HZ, n_freq=N_FREQ, freq_step=FREQ_STEP_HZ,
                bandwidth=BANDWIDTH_HZ, range_res_m=RANGE_RES_M, r_max_m=R_MAX_M)


def build_config(radar_pos=(0., 0., 0.), boresight=(0., 1., 0.)):
    """Vayyar SFCW config dict (schema parallels xrf55 single_chip.json, minus chirp params).

    ``carrierFrequency`` is kept (= the sweep CENTER freq) so wavelength/BSDF code that reads it
    keeps working; the SFCW-specific keys (f_start/f_stop/n_freq/...) drive range compression.
    """
    rx_rel, tx_rel = antenna_layout_vayyar()
    x, y, z = basis_from_boresight(boresight)
    rp = np.asarray(radar_pos, float)
    tx_w, rx_w = _to_world(tx_rel, rp, x, y, z), _to_world(rx_rel, rp, x, y, z)
    sp = sfcw_params()
    b = list(np.asarray(boresight, float).round(8))
    return {
        "modulation_scheme": "SFCW",
        "geometry_mode": "monostatic",
        "carrierFrequency": F_CENTER_HZ,                       # sweep center (bsdf/wavelength read this)
        "f_start": sp["f_start"], "f_stop": sp["f_stop"], "n_freq": sp["n_freq"],
        "freq_step": sp["freq_step"], "bandwidth": sp["bandwidth"],
        "range_res_m": sp["range_res_m"], "r_max_m": sp["r_max_m"],
        "n_tx": N_TX, "n_rx": N_RX,
        "tx_array": [{"name": f"TX{i+1}", "pos_mm": list((tx_w[i] * 1000.0).round(6)),
                      "boresight": b, "radius": 1.0, "polarization": "H"} for i in range(N_TX)],
        "rx_array": [{"name": f"RX{i+1}", "pos_mm": list((rx_w[i] * 1000.0).round(6)),
                      "boresight": b, "power_dBm": -10.0, "polarization": "H"} for i in range(N_RX)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1")
    a = ap.parse_args()
    cfg = build_config()                                        # radar at origin, +Y boresight
    out_dir = os.path.join(C.NBS_DATA_OUT, a.scene, "configs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "vayyar.json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{a.scene}] wrote {out}  (SFCW {F_START_HZ/1e9:.0f}-{F_STOP_HZ/1e9:.0f} GHz, "
          f"{N_FREQ} pts, dR={RANGE_RES_M*100:.2f} cm, R_max={R_MAX_M:.2f} m, {N_TX}Tx x {N_RX}Rx)")


if __name__ == "__main__":
    main()
