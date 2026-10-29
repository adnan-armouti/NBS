"""Novelda XeThru X4M200 (OctoNet node_1 UWB) config for the 3DPS rasterizer's ``UWB`` mode.

Facts from the data (nbs/method/uwb_template.py): frames are the real RF waveform sampled at 23.328 GS/s, 1535 bins
(bin = c / (2 fs) = 6.43 mm one-way, 9.87 m), spectral peak ~7.0 GHz, -10 dB bandwidth 1.46 GHz, the direct coupling
peaks at bin 7 (= the on-board delay, added to every path), pulse template = the coupling peak's temporal mean
($NBS_DATA_OUT/uwb_template_S1/template.npz: t_rf (80,), centre index 40). Datasheet: ~65 deg beamwidth; one Tx and
one Rx ~2 cm apart on the board (positions below are that guess, +-1 cm along the azimuth axis).
Schema parallels vayyar_radar_config.build_config (same tx_array/rx_array/boresight keys).
"""
from __future__ import annotations
import os
import numpy as np
from preprocessing import octonet_config as C
from preprocessing.vayyar_radar_config import basis_from_boresight, _to_world

C_LIGHT = 299792458.0
FS_HZ = 23.328e9
FC_HZ = 7.0e9
N_BINS = 1535
BIN_M = C_LIGHT / (2.0 * FS_HZ)            # 6.43 mm one-way
BW_10DB_HZ = 1.46e9
BEAMWIDTH_DEG = 65.0
TX_RX_SEPARATION_M = 0.02                  # guess (board layout not published)
TEMPLATE_PATH = os.path.join(os.environ.get("NBS_DATA_OUT", C.NBS_DATA_OUT), "uwb_template_S1", "template.npz")


def delay_bins(template_path: str = TEMPLATE_PATH) -> float:
    if os.path.exists(template_path):
        z = np.load(template_path)
        if "per_recording_peak_bin" in z.files:
            return float(z["per_recording_peak_bin"])
    return 7.0


def antenna_layout_x4():
    """(rx_rel (1,3), tx_rel (1,3)) offsets from the board centre, metres: Tx and Rx on the azimuth axis."""
    tx = np.array([[-TX_RX_SEPARATION_M / 2, 0.0, 0.0]])
    rx = np.array([[+TX_RX_SEPARATION_M / 2, 0.0, 0.0]])
    return rx, tx


def build_config(radar_pos=(0., 0., 0.), boresight=(0., 1., 0.), template_path: str = TEMPLATE_PATH) -> dict:
    rx_rel, tx_rel = antenna_layout_x4()
    x, y, z = basis_from_boresight(boresight)
    rp = np.asarray(radar_pos, float)
    tx_w, rx_w = _to_world(tx_rel, rp, x, y, z), _to_world(rx_rel, rp, x, y, z)
    b = list(np.asarray(boresight, float).round(8))
    return {
        "modulation_scheme": "UWB",
        "geometry_mode": "monostatic",
        "carrierFrequency": FC_HZ,
        "sample_rate_hz": FS_HZ, "n_bins": N_BINS, "range_res_m": BIN_M, "bandwidth_10db_hz": BW_10DB_HZ,
        "delay_bins": delay_bins(template_path), "pulse_template": template_path if os.path.exists(template_path) else None,
        "beamwidth_deg": BEAMWIDTH_DEG,
        "n_tx": 1, "n_rx": 1,
        "tx_array": [{"name": "TX1", "pos_mm": list((tx_w[0] * 1000.0).round(6)), "boresight": b, "radius": 1.0, "polarization": "H"}],
        "rx_array": [{"name": "RX1", "pos_mm": list((rx_w[0] * 1000.0).round(6)), "boresight": b, "power_dBm": -10.0, "polarization": "H"}],
    }
