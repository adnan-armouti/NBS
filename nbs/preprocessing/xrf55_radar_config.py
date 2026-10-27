"""Generate the XRF55 single-chip (TI IWR6843ISK) radar config json that 3DPS needs.

3DPS reads a per-scene config (antenna layout + pose + chirp params; schema = mmIR
serialize_single_chip_config / mm3DPS _pose_utils.load_pose_from_config). XRF55's mmWave is a
single-chip IWR6843ISK (3 Tx, 4 Rx). We reuse mmIR's antenna_layout_single_chip geometry:
  - 4 Rx at 0.5*lambda azimuth spacing,
  - 3 Tx at 1.0*lambda azimuth spacing, the MIDDLE Tx ELEVATED by 0.5*lambda,
mapped into the radar frame (origin = radar, +Y boresight) via basis_from_boresight, so
azimuth -> +X and elevation -> +Z (up, toward the ceiling). The elevated middle Tx therefore sits
HIGHER (closer to the ceiling) than the two outer Tx — matching the physical IWR6843ISK mount.

The azimuth RA virtual array (single_chip_to_vx) uses the two NON-elevated Tx{0,2} x 4 Rx = 8
phase centres at uniform 0.5*lambda -> the 64-az ULA.

Chirp params are set so mm3DPS compute_range_res_from_cfg == RANGE_RES_M (data-driven 0.0675 m).
The true XRF55 chirp is unpublished, so we use a self-consistent set (numAdcSamples = 256, carrier
= 62 GHz band centre); only range_res and the carrier are load-bearing for the RA forward model.

  python -m preprocessing.xrf55_radar_config --scene Scene1
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import xrf55_config as C

C_LIGHT = 299792458.0
CARRIER_HZ = 62.0e9            # IWR6843ISK band 60-64 GHz -> use the centre (adjustable)
LAMBDA_M = C_LIGHT / CARRIER_HZ
RX_TX_GAP_MM = 3.724          # physical Rx-block -> Tx-block gap (from mmIR layout)


def antenna_layout_single_chip():
    """Local IWR6843 single-chip layout in lambda units (col0 = azimuth, col1 = elevation).
    Ported verbatim from mmIR preprocessing/config_utils.antenna_layout_single_chip."""
    A1, C1 = 0.5, 1.0                                  # Rx 0.5 lambda, Tx 1.0 lambda spacing
    B1 = RX_TX_GAP_MM / (LAMBDA_M * 1000.0)            # gap in lambda units
    rx = np.array([(i * A1, 0.0) for i in range(4)], float)
    start_x = 4 * A1 + B1
    tx = np.array([(start_x + i * C1, 0.0) for i in range(3)], float)
    tx[1, 1] += A1                                     # MIDDLE Tx elevated by 0.5 lambda
    allp = np.vstack([rx, tx])
    ctr = (allp.min(0) + allp.max(0)) / 2.0
    rel = allp - ctr
    return rel[:4], rel[4:]                            # rx_rel, tx_rel (lambda units)


def basis_from_boresight(b):
    """Right-handed basis: y_hat = boresight, x_hat = azimuth, z_hat = elevation (up)."""
    y = np.asarray(b, float); y = y / np.linalg.norm(y)
    zr = np.array([0., 0., 1.]) if abs(float(y @ [0, 0, 1])) <= 0.99 else np.array([0., 1., 0.])
    x = np.cross(y, zr); x /= np.linalg.norm(x)
    z = np.cross(x, y);  z /= np.linalg.norm(z)
    return x, y, z


def _to_world(rel, radar_pos, x, y, z):
    return np.array([radar_pos + p[0] * LAMBDA_M * x + p[1] * LAMBDA_M * z for p in rel])


def chirp_params(range_res, n_adc=256, sample_rate=5.0e6):
    """FMCW params s.t. c/(2*freqSlope*(rampEndTime-adcStartTime)) == range_res."""
    B = C_LIGHT / (2.0 * range_res)                   # required bandwidth (Hz)
    t_adc = n_adc / sample_rate                        # ADC window (s)
    return dict(sampleRate=sample_rate, numAdcSamples=n_adc, freqSlope=B / t_adc,
                adcStartTime=0.0, rampEndTime=t_adc)


def build_config(radar_pos=(0., 0., 0.), boresight=(0., 1., 0.), range_res=None):
    """Single-chip config dict. Radar at the scaffold-frame origin (=radar), +Y boresight."""
    range_res = C.RANGE_RES_M if range_res is None else range_res
    rx_rel, tx_rel = antenna_layout_single_chip()
    x, y, z = basis_from_boresight(boresight)
    rp = np.asarray(radar_pos, float)
    tx_w, rx_w = _to_world(tx_rel, rp, x, y, z), _to_world(rx_rel, rp, x, y, z)
    cp = chirp_params(range_res)
    b = list(np.asarray(boresight, float).round(8))
    return {
        "modulation_scheme": "FMCW", "carrierFrequency": CARRIER_HZ,
        "sampleRate": cp["sampleRate"], "freqSlope": cp["freqSlope"],
        "numAdcSamples": cp["numAdcSamples"], "adcStartTime": cp["adcStartTime"],
        "rampEndTime": cp["rampEndTime"],
        "tx_array": [{"name": f"TX{i+1}", "pos_mm": list((tx_w[i] * 1000.0).round(6)),
                      "boresight": b, "radius": 1.0, "polarization": "V"} for i in range(3)],
        "rx_array": [{"name": f"RX{i+1}", "pos_mm": list((rx_w[i] * 1000.0).round(6)),
                      "boresight": b, "power_dBm": 10.0, "polarization": "V"} for i in range(4)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Scene1")
    a = ap.parse_args()
    cfg = build_config()                               # radar at origin, +Y boresight, dR=RANGE_RES_M
    out_dir = os.path.join(C.NBS_DATA_OUT, a.scene, "configs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "single_chip.json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{a.scene}] wrote {out}  (carrier {CARRIER_HZ/1e9:.0f} GHz, numAdc {cfg['numAdcSamples']})")


if __name__ == "__main__":
    main()
