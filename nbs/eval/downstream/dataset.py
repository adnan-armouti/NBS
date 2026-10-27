"""XRF55 per-modality HAR dataset, driven by the Stage-1 manifest.

55-class action recognition. Label = action id - 1 (0..54). Subject-disjoint split
comes from the manifest's `split` column. Supports an `input_dir` override so the SAME
loader can serve REAL data (from the dataset) or SYNTHESIZED band data (Stage 6 swap):
if input_dir is given, sample arrays are read from input_dir/<clip>.npy instead of disk.
"""
import os, csv, zlib
import numpy as np
import torch
from torch.utils.data import Dataset

_MOD_DIR = {"mmwave": "mmWave", "wifi": "WiFi", "rfid": "RFID"}


def _modality_path(mmwave_path, modality):
    """Map the manifest mmWave path to the requested modality's npy path."""
    return mmwave_path.replace("/mmWave/", f"/{_MOD_DIR[modality]}/")


def _norm(x, modality):
    x = x.astype(np.float32)
    if modality == "mmwave":              # huge dynamic range -> log then standardize
        x = np.log1p(np.maximum(x, 0))
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-6)


class XRFHar(Dataset):
    def __init__(self, manifest_csv, modality, split, scenes=None,
                 trials=None, input_dir=None, split_mode="subject"):
        assert modality in _MOD_DIR
        assert split_mode in ("subject", "random")
        self.modality, self.input_dir = modality, input_dir
        rows = list(csv.DictReader(open(manifest_csv)))
        self.items = []
        for r in rows:
            if split_mode == "subject":           # manifest subject-disjoint split
                r_split = r["split"]
            else:                                  # within-distribution random 80/20 by clip
                r_split = "test" if (zlib.crc32(r["clip"].encode()) % 5 == 0) else "train"
            if split and r_split != split:
                continue
            if scenes and r["mmwave"].split("/RF/")[1].split("/")[0] not in scenes:
                continue
            if trials and r["trial"] not in trials:
                continue
            label = int(r["action"]) - 1
            self.items.append((r["clip"], _modality_path(r["mmwave"], modality), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip, path, label = self.items[i]
        if self.input_dir:                 # synthesized-input swap
            arr = np.load(os.path.join(self.input_dir, f"{clip}.npy"))
        else:
            arr = np.load(path)
        if arr.ndim == 4:                  # mmWave (1,17,256,128) -> (17,256,128)
            arr = arr[0]
        x = _norm(arr, self.modality)
        return torch.from_numpy(x).float(), label


def in_channels(modality):
    return {"mmwave": 17, "wifi": 270, "rfid": 23}[modality]
