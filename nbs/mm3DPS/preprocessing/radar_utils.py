import os
import numpy as np

from mm3DPS.preprocessing.ColoRadar_tools.dataset_loaders import (
    get_cascade_params,
    get_adc_frame,
    get_timestamps,
)
from mm3DPS.preprocessing.ColoRadar_tools.calibration import (
    apply_phase_calibration,
    apply_frequency_calibration,
)

from mm3DPS.preprocessing.io_paths import get_base_dir, get_subdir


def _save_adc(base_dir: str, kind: str, frame_idx: int, adc: np.ndarray) -> str:
    radar_dir = get_subdir(base_dir, "radar")
    fname = f"{kind}_frame_{frame_idx}.npy"
    path = os.path.join(radar_dir, fname)
    np.save(path, adc)
    return path


def generate_adc_window(*,
                        seq_idx: int,
                        center_frame_idx: int,
                        num_radar_frames: int,
                        dataset_dir: str,
                        calib_path: str,
                        out_root: str,
                        run_cascade: bool,
                        verbose: bool = False) -> dict:
    if num_radar_frames < 1 or (num_radar_frames % 2) != 1:
        raise ValueError("num_radar_frames must be odd and >= 1")

    seq_path = dataset_dir + str(seq_idx)
    base_dir = get_base_dir(seq_idx, center_frame_idx, out_root)

    half = num_radar_frames // 2
    cascade_indices = list(range(center_frame_idx - half, center_frame_idx + half + 1))

    results = {"cascade_adc": []}

    if run_cascade:
        all_params = get_cascade_params(calib_path)
        wf = all_params["waveform"]
        for c_idx in cascade_indices:
            adc = get_adc_frame(c_idx, seq_path, wf)
            # adc = apply_phase_calibration(adc, all_params["phase"])  # phase first - DISABLED to match CoIR reference
            adc = apply_frequency_calibration(adc, all_params["frequency"], wf)  # freq only (matches CoIR)
            adc = adc.transpose(2, 1, 0, 3)  # (N_C, N_Rx, N_Tx, N_ADC)
            if verbose:
                print(f"[adc] cascade idx={c_idx} shape={adc.shape}")
            results["cascade_adc"].append(_save_adc(base_dir, "cascaded", c_idx, adc))

    return results


