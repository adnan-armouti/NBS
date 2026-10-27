"""Measured data: the ADC archives, the dataset registry, and the model inputs.

Five modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C3 --
`adc.py`, `datasets.py` and the three under `data/`. Zero name collisions
between them and no `__file__`-derived values, so every body below is verbatim.

    ADC             DCA1000 archive decode, range/angle/slow-time cubes
    datasets        which datasets exist, their flags and scene alignment
    SmplBody        the SMPL model and its HuPR joint mapping
    RadProposer     the published RadProposer point proposals

`radproposer` used to import from `adc`; that is now
references inside one module and their import lines are gone.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import re
import torch
import zipfile

from numpy.lib.format import open_memmap
from pathlib import Path

from mm4DPS import DATASETS, HUPR, HUPR_NEAR_FIELD_M, SMPL_MODEL_DIR, Dataset, Radar, current_dataset
from mm4DPS.signal_model import apply_fft_window, fft_window



# --------------------------------------------------------------------
# ADC: the DCA1000 archives and the cubes decoded from them
# --------------------------------------------------------------------

# HuPR's IWR1843 numbers, kept under their old names for the HuPR-only paths
# (RadProposer, the seeds). Everything shared between datasets takes a
# :class:`Radar` and reads the same numbers from it.
_IWR1843 = HUPR.radars[0]
NUM_RX = _IWR1843.n_rx
NUM_TX = _IWR1843.n_tx
ADC_SAMPLES = _IWR1843.samples
TDM_LOOPS = _IWR1843.loops
PHYSICAL_CHIRPS = _IWR1843.chirps
INT16_PER_COMPLEX = 2
BYTES_PER_FRAME = _IWR1843.bytes_per_frame
AZIMUTH_TX = _IWR1843.azimuth_tx


def _radar(radar: Radar | None) -> Radar:
    """The radar a shape-bearing helper was handed, else the current dataset's.

    Every dataset's radars share one chirp, so the first one answers any
    chirp question; a caller that cares WHICH radar passes it.
    """
    return radar if radar is not None else current_dataset().radars[0]


def tdm_split(adc, radar: Radar | None = None):
    """``(rx, chirp, sample)`` -> ``(rx, tx, loop, sample)``, indexed by physical TX.

    Slot ``k`` of every loop carries physical TX ``radar.tx_order[k]``; HuPR's
    order is the identity, the cascade's runs 12 down to 1. numpy and torch
    inputs both work (stack is the same name in both).
    """
    radar = _radar(radar)
    stack = np.stack if isinstance(adc, np.ndarray) else __import__("torch").stack
    slots = [adc[:, k::radar.n_tx, :] for k in range(radar.n_tx)]
    by_tx = [None] * radar.n_tx
    for slot, tx in enumerate(radar.tx_order):
        by_tx[tx] = slots[slot]
    return stack(by_tx, **({"axis": 1} if isinstance(adc, np.ndarray) else {"dim": 1}))


def find_archive(dataset_root: Path, sequence: int) -> Path:
    for path in sorted((dataset_root / "radar").glob("radar_seq_*.zip")):
        match = re.search(r"seq_(\d+)-(\d+)", path.name)
        if match and int(match.group(1)) <= sequence <= int(match.group(2)):
            return path
    raise FileNotFoundError(f"no HuPR radar archive contains sequence {sequence}")


def decode_dca1000_frame(payload: bytes, radar: Radar | None = None) -> np.ndarray:
    """Decode one DCA1000 frame to complex64 ``(rx, chirp, sample)``."""
    radar = _radar(radar)
    if len(payload) != radar.bytes_per_frame:
        raise ValueError(f"expected {radar.bytes_per_frame} frame bytes, got {len(payload)}")
    raw = np.frombuffer(payload, dtype=np.int16).reshape(-1, 4)
    lvds_i = raw[:, :2].reshape(-1)
    lvds_q = raw[:, 2:].reshape(-1)
    stream = lvds_i.astype(np.float32) + 1j * lvds_q.astype(np.float32)
    blocks = stream.reshape(radar.chirps, radar.n_rx, radar.samples)
    return blocks.transpose(1, 0, 2).astype(np.complex64, copy=False)


def decode_dca1000_frame_torch(raw_int16, radar: Radar | None = None):
    """Decode one frame from its raw int16 words, on whatever device they are.

    The exact operations of :func:`decode_dca1000_frame` in the same order:
    int16 -> float32 is exact, and the interleave/reshape/permute are pure
    data movement, so this is **bitwise identical** to the numpy decode (the
    suite pins it). Taking the raw words rather than a decoded frame is what
    makes it useful: 0.79 MB of int16 crosses to the device instead of the
    1.57 MB complex64 frame, and the decode itself stops being CPU work.
    """
    import torch

    radar = _radar(radar)
    expected = radar.raw_words_per_frame
    if raw_int16.numel() != expected:
        raise ValueError(
            f"expected {expected} int16 words, got {raw_int16.numel()}"
        )
    grouped = raw_int16.reshape(-1, 4)
    lvds_i = grouped[:, :2].reshape(-1).to(torch.float32)
    lvds_q = grouped[:, 2:].reshape(-1).to(torch.float32)
    stream = torch.complex(lvds_i, lvds_q)
    blocks = stream.reshape(radar.chirps, radar.n_rx, radar.samples)
    return blocks.permute(1, 0, 2).contiguous()


def _member(sensor) -> str:
    """The archive tag: a :class:`Radar`, its name, or the tag itself."""
    if isinstance(sensor, Radar):
        return sensor.member
    for radar in current_dataset().radars:
        if sensor == radar.name:
            return radar.member
    if sensor not in {radar.member for radar in current_dataset().radars}:
        raise ValueError(f"unknown radar or member tag {sensor!r}")
    return sensor


def load_adc_sequence_bytes(dataset_root: Path, sequence: int, sensor) -> bytes:
    """Read one sensor's whole ADC stream in a single sequential pass.

    The archive member is DEFLATE-compressed, so a per-frame ``seek`` must
    decompress from the member's start each call — caching N frames that way
    costs O(N^2). One sequential read costs O(N) once (~0.88 GB raw) and
    frames are then sliced from memory; see ``adc_frame_from_bytes``.
    """
    archive_path = find_archive(dataset_root, sequence)
    member = f"single_{sequence}/{_member(sensor)}/adc_data.bin"
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source:
        return source.read()


def adc_frame_from_bytes(payload: bytes, frame: int, radar: Radar | None = None) -> np.ndarray:
    """Decode one frame from a full-sequence ADC byte buffer."""
    radar = _radar(radar)
    if frame < 0:
        raise ValueError("frame index must be non-negative")
    start = frame * radar.bytes_per_frame
    chunk = payload[start:start + radar.bytes_per_frame]
    if len(chunk) != radar.bytes_per_frame:
        raise ValueError(f"frame {frame} extends past the ADC stream")
    return decode_dca1000_frame(chunk, radar)


def load_adc_frame(dataset_root: Path, sequence: int, sensor, frame: int,
                   radar: Radar | None = None) -> np.ndarray:
    radar = _radar(radar)
    if frame < 0:
        raise ValueError("frame index must be non-negative")
    archive_path = find_archive(dataset_root, sequence)
    member = f"single_{sequence}/{_member(sensor)}/adc_data.bin"
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source:
        source.seek(frame * radar.bytes_per_frame)
        payload = source.read(radar.bytes_per_frame)
    return decode_dca1000_frame(payload, radar)


# --------------------------------------------------------------------
# The cascade: TI TDA2 capture, four device files, and its calibration
# --------------------------------------------------------------------

CASCADE_DEVICES = ("master", "slave1", "slave2", "slave3")
RX_PER_DEVICE = 4


def tda2_frame_count(idx_path: Path) -> int:
    """Frames a TDA2 ``*_idx.bin`` declares: uint32 word 4 of its header."""
    header = np.fromfile(idx_path, dtype=np.uint32, count=6)
    return int(header[3])


class CascadeRecording:
    """One sequence's cascade radar: ``radar/bin/{master,slave1-3}_0000_data.bin``.

    Each device file holds its four RX channels for every chirp of every
    frame, int16 I/Q interleaved; the sixteen channels of a frame are the
    four devices' concatenated in device order. Read sequentially, a frame
    at a time, the way the renderer consumes HuPR's archives.
    """

    def __init__(self, sequence_dir: Path, radar: Radar):
        self.radar = radar
        self.paths = [sequence_dir / "radar" / "bin" / f"{d}_0000_data.bin" for d in CASCADE_DEVICES]
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.frames = min(tda2_frame_count(p.with_name(p.name.replace("_data", "_idx")))
                          for p in self.paths)
        self.words_per_device = RX_PER_DEVICE * radar.chirps * radar.samples * 2
        self.position = 0

    @property
    def bytes_per_device_frame(self) -> int:
        return self.words_per_device * 2

    def frame_words(self, frame: int | None = None) -> np.ndarray:
        """``(4 devices, words)`` int16 of one frame; the next one by default."""
        frame = self.position if frame is None else frame
        if not 0 <= frame < self.frames:
            raise ValueError(f"frame {frame} outside the recording's {self.frames}")
        words = np.stack([
            np.fromfile(path, dtype=np.int16, count=self.words_per_device,
                        offset=frame * self.bytes_per_device_frame)
            for path in self.paths])
        self.position = frame + 1
        return words


def decode_tda2_frame_torch(words, radar: Radar | None = None):
    """``(4 devices, words)`` int16 -> ``(16 rx, chirp, sample)`` complex64.

    TI's ``readBinFile``: I/Q interleaved, then (column-major) reshaped to
    ``(rx, sample, slot, loop)``; row-major that is ``(loop, slot, sample,
    rx)``. Chirp ``loop * n_tx + slot`` matches the DCA1000 layout, so every
    downstream helper (tdm_split, the beamformers) applies unchanged.
    """
    import torch

    radar = _radar(radar)
    expected = (len(CASCADE_DEVICES), RX_PER_DEVICE * radar.chirps * radar.samples * 2)
    if tuple(words.shape) != expected:
        raise ValueError(f"expected {expected} words, got {tuple(words.shape)}")
    real = words[:, 0::2].to(torch.float32)
    imag = words[:, 1::2].to(torch.float32)
    stream = torch.complex(real, imag)                         # (4, loop*slot*sample*rx)
    blocks = stream.reshape(len(CASCADE_DEVICES), radar.loops, radar.n_tx, radar.samples, RX_PER_DEVICE)
    # (device, loop, slot, sample, rx) -> (device, rx, loop, slot, sample) -> (16, chirp, sample)
    frame = blocks.permute(0, 4, 1, 2, 3).reshape(radar.n_rx, radar.chirps, radar.samples)
    return frame.contiguous()


def cascade_calibration(radar: Radar, path: Path, interp: int = 5, phase_only: bool = True):
    """TI's ``calibrationCascade`` as one ``(rx, slot, sample)`` complex multiplier.

    From ``configs/mmwcas_calibration.npz`` (RangeMat, PeakValMat of TI's
    ``calibrateResults_high.mat``, both (TX, RX)
    in the recorded RX order): per (TX, RX) a range-offset term, a linear
    phase across samples, and a phase correction to the reference (the first
    enabled TX, RX 0). Diagonal, so the calibrated ADC inverts exactly.
    """
    import torch

    with np.load(path) as result:                                   # RangeMat, PeakValMat of TI's .mat
        range_mat = result["RangeMat"].astype(np.float64)           # (12, 16)
        peak_mat = result["PeakValMat"].astype(np.complex128)       # (12, 16)
    slope_calib, fs_calib = 78.986e12, 8.0e6                      # hardware_param.m
    tx_ref = radar.tx_order[0]
    samples = np.arange(radar.samples, dtype=np.float64)
    cal = np.empty((radar.n_rx, radar.n_tx, radar.samples), dtype=np.complex128)
    for slot, tx in enumerate(radar.tx_order):
        freq = ((range_mat[tx] - range_mat[tx_ref, 0]) * fs_calib / radar.sample_rate_hz
                * radar.slope_hz_s / slope_calib)
        freq = 2.0 * np.pi * freq / (radar.samples * interp)             # (16,)
        phase = peak_mat[tx_ref, 0] / peak_mat[tx]                        # (16,)
        if phase_only:
            phase = phase / np.abs(phase)
        cal[:, slot, :] = np.exp(1j * samples[None, :] * freq[:, None]) * phase[:, None]
    return torch.from_numpy(cal.astype(np.complex64))


def apply_cascade_calibration(frame, calibration, radar: Radar | None = None):
    """Multiply a decoded ``(rx, chirp, sample)`` frame by the per-slot calibration."""
    radar = _radar(radar)
    shaped = frame.reshape(radar.n_rx, radar.loops, radar.n_tx, radar.samples)
    return (shaped * calibration.to(frame.device)[:, None, :, :]).reshape(frame.shape)


class MeasuredReader:
    """A sequence's recording of one radar, read as decoded frame batches on a device.

    HuPR: the archive member, decoded with the DCA1000 decode. The cascade:
    the four TDA2 device files, decoded and calibrated. Either way a batch
    is ``(frames, rx, chirp, sample)`` complex64, the layout every batched
    helper takes.

    A HuPR member is deflated, and inflating it is what a pass over the
    sequence costs (1.4 s per 450 frames). The chain's preprocessing makes
    three such passes in one process (the coupling, the person sequence,
    the motion detection), so the raw int16 frames of a member are kept in
    memory once read (`_RAW_FRAMES`, 0.8 MB a frame, 470 MB a board for 600
    frames) and released by `MeasuredReader.release()` when the stage is
    done. In memory for this process only: nothing is written, a later run
    reads the archive again.
    """

    _RAW_FRAMES: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    _STREAMS: dict[tuple[str, str], tuple[object, int]] = {}     # one inflating stream per member, and its frame

    @classmethod
    def release(cls) -> None:
        """Drop every member's frames kept in memory and close its stream."""
        cls._RAW_FRAMES.clear()
        for stream, _ in cls._STREAMS.values():
            stream.close()
        cls._STREAMS.clear()

    def __init__(self, dataset: Dataset, sequence: int, radar: Radar, dataset_root: Path | None = None,
                 config_dir: Path | None = None):
        self.dataset, self.sequence, self.radar = dataset, int(sequence), radar
        if radar.decoder == "dca1000":
            root = dataset_root if dataset_root is not None else dataset.root()
            self.archive = zipfile.ZipFile(find_archive(root, sequence))
            self.member = f"single_{sequence}/{radar.member}/adc_data.bin"
            self.frames = self.archive.getinfo(self.member).file_size // radar.bytes_per_frame
            self.calibration = None
        elif radar.decoder == "tda2":
            self.recording = CascadeRecording(dataset.sequence_dir(sequence), radar)
            self.frames = self.recording.frames
            from mm4DPS import Install
            self.calibration = cascade_calibration(
                radar, (config_dir or Install.from_env().config_dir) / radar.calibration)
        else:
            raise NotImplementedError(radar.decoder)

    def batch(self, start: int, count: int, device):
        """Frames ``start .. start + count`` decoded on ``device``."""
        import torch

        radar = self.radar
        if start < 0 or start + count > self.frames:
            raise ValueError(f"frames {start}..{start + count - 1} requested of a {self.frames}-frame recording")
        if radar.decoder == "dca1000":
            # The member is deflated: a `seek` on it rewinds and decompresses
            # forward from the member's start, so reopening per batch made a
            # sequential pass over 600 frames quadratic (motion detection:
            # 140 s, 117 s of them in zlib). One stream per member for the
            # whole process, only ever read forward, every frame it passes
            # kept; a frame behind the stream is served from what was kept.
            key = (self.archive.filename, self.member)
            kept = self._RAW_FRAMES.setdefault(key, {})
            wanted = [frame for frame in range(start, start + count) if frame not in kept]
            if wanted:
                stream, position = self._STREAMS.get(key, (None, 0))
                if stream is None:
                    stream = self.archive.open(self.member)
                    position = 0
                first, last = min(wanted), max(wanted)
                if first < position:
                    raise RuntimeError(f"frame {first} of {self.member} was passed without being kept")
                raw = np.frombuffer(stream.read(radar.bytes_per_frame * (last + 1 - position)),
                                    dtype=np.int16).reshape(last + 1 - position, -1).copy()
                for offset_in_block in range(len(raw)):
                    kept[position + offset_in_block] = raw[offset_in_block]
                self._STREAMS[key] = (stream, last + 1)
            block = np.stack([kept[frame] for frame in range(start, start + count)])
            return decode_dca1000_frames_torch(torch.from_numpy(block).to(device), radar)
        frames = []
        calibration = self.calibration.to(device)
        for frame in range(start, start + count):
            words = torch.from_numpy(self.recording.frame_words(frame)).to(device)
            frames.append(apply_cascade_calibration(decode_tda2_frame_torch(words, radar), calibration, radar))
        return torch.stack(frames)

    def close(self) -> None:
        # the member's stream and kept frames outlive the reader (`release`)
        pass


class RTPoseSequence:
    """One RT-Pose sequence on disk: its radar, LiDAR, cameras and annotation.

    The LiDAR file names step by 2 or 3 per radar frame depending on the
    sequence; the annotation (Train.json) carries the LiDAR file <-> radar
    frame pairing, so that is the map used. Annotations are 15 Human3.6M
    joints in a frame with the LiDAR's axes (x forward, y left, z up) whose
    origin sits ABOUT 0.2 M LEFT OF THE LIDAR: on seq 153 the LiDAR's own
    person points lie 0.20 +/- 0.02 m left of the annotated centroid (n = 34
    frames), and the radar's person track puts the radar 0.22 m right of the
    annotation origin (n = 101, rms sin 0.017) -- so the radar is within
    0.02 m of the LiDAR laterally and the annotation frame is a third origin,
    presumably the radar frame as RT-POSE's own (unreleased) calibration had
    it. The y-mirror is excluded either way (0 LiDAR points on it).
    """
    #: annotation frame -> LiDAR frame, metres (x forward, y left, z up)
    ANNOTATION_TO_LIDAR = np.array([0.0, 0.2, 0.0])

    JOINTS = ("pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
              "left_ankle", "thorax", "head", "right_shoulder", "right_elbow", "right_wrist",
              "left_shoulder", "left_elbow", "left_wrist")
    #: LiDAR frame (x forward, y left, z up) -> radar world (+X right, +Y forward, +Z up)
    LIDAR_TO_WORLD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def __init__(self, dataset: Dataset, sequence: int):
        self.dataset, self.sequence = dataset, int(sequence)
        self.root = dataset.root()
        self.dir = dataset.sequence_dir(sequence)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"RT-Pose sequence {sequence}: {self.dir}")
        self._annotation = None

    @property
    def radar(self) -> Radar:
        return self.dataset.radars[0]

    def recording(self) -> "CascadeRecording":
        return CascadeRecording(self.dir, self.radar)

    def annotation(self) -> dict[int, dict]:
        """``{radar frame: {"lidar": file stem, "joints": (15, 3), "lidar_valid": bool}}``."""
        if self._annotation is None:
            entries = json.loads((self.root / "Train.json").read_text())[str(self.sequence)]
            out = {}
            for stem, rows in entries.items():
                row = rows[0]
                out[int(row["Radar_frameID"])] = {
                    "lidar": stem, "lidar_valid": str(row["LiDAR val"]) in ("1", "True", "true"),
                    "joints": np.asarray(row["pose"], dtype=np.float64).reshape(-1, 3)}
            self._annotation = dict(sorted(out.items()))
        return self._annotation

    def lidar_stems(self) -> list[str]:
        return sorted(p.stem for p in (self.dir / "lidar").glob("*.pcd"))

    def lidar(self, stem: str) -> np.ndarray:
        """``(N, 3)`` float64 points of one LiDAR file, in the LiDAR frame."""
        return read_pcd(self.dir / "lidar" / f"{stem}.pcd")

    def image_path(self, stem: str, camera: str = "left") -> Path:
        return self.dir / "camera" / camera / f"{stem}.png"

    def camera(self, camera: str = "left") -> dict:
        """The stereo calibration as shipped: ``extrinsic`` (LiDAR -> camera, 4x4) and ``intrinsic`` (3x4)."""
        raw = json.loads((self.root / "calib" / "camera" / f"{camera}.json").read_text())
        return {"extrinsic": np.asarray(raw["extrinsic"], dtype=np.float64).reshape(4, 4),
                "intrinsic": np.asarray(raw["intrinsic"], dtype=np.float64).reshape(3, 4)}

    def clock(self) -> tuple[int, int]:
        """The LiDAR/camera file numbering as a function of the radar frame: ``stem = a * frame + b``.

        The file stems are ticks of the rig's master clock; the radar frame
        is a slower count of it. The annotation pairs a radar frame with a
        stem on every annotated frame and the pairing is exactly linear on
        every sequence (85 at 3 f - 4, 17 at 2 f - 2 -- the camera and LiDAR
        ran at three or two ticks a radar frame), so it is fitted here on
        the annotated pairs and required to reproduce them exactly.
        """
        annotation = self.annotation()
        frames = np.array(sorted(annotation), dtype=np.float64)
        stems = np.array([int(annotation[f]["lidar"]) for f in sorted(annotation)], dtype=np.float64)
        if len(frames) < 2:
            raise ValueError(f"RT-Pose sequence {self.sequence}: too few annotated frames to pair the streams")
        (a, b), *_ = np.linalg.lstsq(np.stack([frames, np.ones_like(frames)], 1), stems, rcond=None)
        a, b = int(round(a)), int(round(b))
        if np.any(a * frames + b != stems):
            raise ValueError(f"RT-Pose sequence {self.sequence}: the annotation's radar-to-file pairing is "
                             f"not linear (fit {a} f + {b})")
        return a, b

    def camera_stems(self, camera: str = "left") -> list[str]:
        return sorted(p.stem for p in (self.dir / "camera" / camera).glob("*.png"))

    def stem_for_frame(self, frame: int, stems: list[str] | None = None) -> str:
        """The LiDAR (default) or camera file paired with a radar frame.

        The annotation's own pairing where it exists; elsewhere the file whose
        stem is nearest the clock's tick for that frame (`clock`), which is
        the annotation's choice on every annotated frame of every sequence.
        """
        stems = self.lidar_stems() if stems is None else stems
        annotation = self.annotation()
        if frame in annotation and annotation[frame]["lidar"] in stems:
            return annotation[frame]["lidar"]
        a, b = self.clock()
        numbers = np.array([int(stem) for stem in stems])
        return stems[int(np.abs(numbers - (a * frame + b)).argmin())]

    def camera_index(self, frame: int, camera: str = "left") -> int:
        """The row of a per-camera-frame array (GVHMR's, in file order) paired with a radar frame."""
        stems = self.camera_stems(camera)
        return stems.index(self.stem_for_frame(frame, stems))

    def to_lidar(self, points_world: np.ndarray, extrinsic: dict | None = None) -> np.ndarray:
        """The inverse of :meth:`to_world`."""
        rotation = self.LIDAR_TO_WORLD
        translation = np.zeros(3)
        if extrinsic is not None:
            rotation = np.asarray(extrinsic["rotation"], dtype=np.float64) @ rotation
            translation = np.asarray(extrinsic["translation_m"], dtype=np.float64)
        return (points_world - translation) @ rotation

    def camera_matrix(self, camera: str = "left", image_size=None) -> np.ndarray:
        """The 3x3 intrinsics, re-centred for the images on disk.

        The calibration is for the 1468 x 1128 sensor; the PNGs are its
        centred 1440 x 1080 crop, so the principal point moves by half the
        difference and the focal length is unchanged.
        """
        K = self.camera(camera)["intrinsic"][:, :3].copy()
        if image_size is not None:
            K[0, 2] -= (2 * K[0, 2] - image_size[0]) / 2.0
            K[1, 2] -= (2 * K[1, 2] - image_size[1]) / 2.0
        return K

    def project(self, points_lidar: np.ndarray, camera: str = "left", image_size=None) -> np.ndarray:
        """LiDAR-frame points -> pixels of the (centre-cropped) camera image, with depth as column 2."""
        T = self.camera(camera)["extrinsic"]
        K = self.camera_matrix(camera, image_size)
        cam = points_lidar @ T[:3, :3].T + T[:3, 3]
        depth = np.clip(cam[:, 2], 1e-6, None)
        return np.stack([K[0, 0] * cam[:, 0] / depth + K[0, 2], K[1, 1] * cam[:, 1] / depth + K[1, 2], cam[:, 2]], axis=1)

    def to_world(self, points_lidar: np.ndarray, extrinsic: dict | None = None) -> np.ndarray:
        """LiDAR-frame points -> radar world, through the radar extrinsic (rotation + translation)."""
        rotation = self.LIDAR_TO_WORLD
        translation = np.zeros(3)
        if extrinsic is not None:
            rotation = np.asarray(extrinsic["rotation"], dtype=np.float64) @ rotation
            translation = np.asarray(extrinsic["translation_m"], dtype=np.float64)
        return points_lidar @ rotation.T + translation


def read_pcd(path: Path) -> np.ndarray:
    """PCD v0.7, ascii or binary, any field list starting with x y z -> ``(N, 3)`` float64."""
    with open(path, "rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: no DATA line")
            header.append(line)
            if line.startswith(b"DATA"):
                break
        get = lambda key: [l for l in header if l.startswith(key)][0].split()[1:]
        points = int(get(b"POINTS")[0])
        if header[-1].split()[1] == b"ascii":
            return np.loadtxt(handle, dtype=np.float64, max_rows=points)[:, :3]
        fields = [f.decode() for f in get(b"FIELDS")]
        dtype = np.dtype([(field, {"F": "<f", "U": "<u", "I": "<i"}[t.decode()] + s.decode())
                          for field, t, s in zip(fields, get(b"TYPE"), get(b"SIZE"))])
        table = np.frombuffer(handle.read(points * dtype.itemsize), dtype=dtype)
        return np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float64)


def virtual_array_slow_time(frame_adc: np.ndarray, radar: Radar | None = None) -> np.ndarray:
    """Return the HuPR azimuth/elevation ULA as (loop=64, virtual=8, ADC=256).

    A HuPR "chirp index" after TDM demultiplexing is a slow-time loop. Each
    loop contains three sequential physical TX chirps. TX0 and TX2, each with
    four receivers, form the released code's eight-element virtual ULA. A
    single-loop RA image must therefore retain both physical TX chirps rather
    than selecting only one physical chirp and losing half the aperture.
    """
    radar = _radar(radar)
    if frame_adc.shape != radar.adc_frame_shape:
        raise ValueError(
            f"expected {radar.adc_frame_shape}, got {frame_adc.shape}"
        )
    tdm = tdm_split(frame_adc, radar)
    # (RX, TX, loop, ADC) -> TX0.RX0..3, TX2.RX0..3 -> (loop, virtual, ADC)
    virtual = np.concatenate([tdm[:, tx] for tx in radar.azimuth_tx], axis=0).transpose(1, 0, 2)
    return virtual_to_grid(virtual, radar, axis=1)


def virtual_to_grid(virtual, radar: Radar, axis: int):
    """Place virtual channels on the radar's half-wavelength azimuth grid, windowed for the angle FFT.

    HuPR's eight channels are eight slots in order; the cascade's 144 land
    on 86 slots, channels sharing a slot averaged. The azimuth window
    (`signal_model.fft_window("azimuth", G)`) is then applied per slot --
    the same window `forward_render._to_grid` applies to the rendered
    channels, so both sides' angle FFTs see it. numpy and torch inputs both
    work.
    """
    slots, n = radar.azimuth_grid()
    identity = n == virtual.shape[axis] and bool((slots == np.arange(n)).all())
    if isinstance(virtual, np.ndarray):
        if identity:
            return apply_fft_window(virtual, "azimuth", axis)
        shape = list(virtual.shape); shape[axis] = n
        grid = np.zeros(shape, dtype=virtual.dtype)
        counts = np.zeros(n)
        moved = np.moveaxis(grid, axis, 0)
        for v, slot in enumerate(slots):
            moved[slot] += np.take(virtual, v, axis=axis)
            counts[slot] += 1
        moved /= counts.reshape((n,) + (1,) * (virtual.ndim - 1))
        return apply_fft_window(grid, "azimuth", axis)
    import torch
    if identity:
        return apply_fft_window(virtual, "azimuth", axis)
    index = torch.as_tensor(slots, device=virtual.device)
    counts = torch.zeros(n, device=virtual.device, dtype=virtual.real.dtype if virtual.is_complex() else virtual.dtype)
    counts.index_add_(0, index, torch.ones(len(slots), device=virtual.device, dtype=counts.dtype))
    shape = list(virtual.shape); shape[axis] = n
    grid = virtual.new_zeros(shape)
    grid.index_add_(axis, index, virtual)
    weights = (1.0 / counts).reshape([n if d == axis % virtual.ndim else 1 for d in range(virtual.ndim)])
    return apply_fft_window(grid * weights, "azimuth", axis)


def elevation_grid_plan(radar: Radar) -> dict:
    """Where every (TX, RX) channel of the FULL array sits on the 2D (azimuth, elevation) half-wavelength grid.

    The azimuth slots coincide with :meth:`Radar.azimuth_grid` (the same
    origin: the azimuth array's smallest phase centre), so the (range,
    azimuth) marginal of this view is the beamformer's own axis; the
    elevation slots start at the smallest elevation phase centre. ``n_el`` is
    the smallest power of two that holds the elevation span (HuPR's two rows
    -> 2, the cascade's seven -> 8). Channels sharing a slot are averaged, as
    in :func:`virtual_to_grid`. Both sides of the elevation term
    (`forward_render.profiles_to_rae_spectrum_fused`, the trainer's targets)
    read this one plan.
    """
    rx, tx = radar.rotated_positions()
    az = np.array([tx[t, 0] + rx[r, 0] for t in range(radar.n_tx) for r in range(radar.n_rx)])
    el = np.array([tx[t, 1] + rx[r, 1] for t in range(radar.n_tx) for r in range(radar.n_rx)])
    az_origin = min(tx[t, 0] + rx[r, 0] for t in radar.azimuth_tx for r in range(radar.n_rx))
    az_slots = np.round(az - az_origin).astype(int)
    el_slots = np.round(el - el.min()).astype(int)
    n_az = int(az_slots.max()) + 1
    n_el_slots = int(el_slots.max()) + 1
    n_el = 2
    while n_el < n_el_slots:
        n_el *= 2
    return {"az_slots": az_slots, "el_slots": el_slots, "n_az": n_az, "n_el_slots": n_el_slots,
            "n_el": n_el, "n_angle": radar.n_angle}


_GRID2D_CACHE: dict = {}


def channels_to_grid2d(channels, radar: Radar, axis: int, window: bool = True):
    """``(..., n_tx * n_rx, K)`` complex channels (TX-major, physical TX order) -> ``(..., G_az, G_el, K)``, windowed on both axes.

    ``window=False``: no azimuth taper (a downstream tensor's convention; the level calibration compares both sides so).

    torch only. The azimuth window is `fft_window("azimuth", G_az)` per
    azimuth slot (as the 1D beamformer applies it); the elevation axis is
    untapered; channels sharing a slot are averaged. The
    slot index and the combined weights are built once per (radar, device,
    dtype, window) and reused -- no host copies inside a captured step.
    """
    import torch
    from mm4DPS.signal_model import window_cache_key
    channels = torch.movedim(channels, axis, -2)                           # (..., V, K)
    real_dtype = channels.real.dtype if channels.is_complex() else channels.dtype
    key = (radar.name, radar.n_tx, str(channels.device), real_dtype, window_cache_key() if window else "none")
    plan = _GRID2D_CACHE.get(key)
    if plan is None:
        p = elevation_grid_plan(radar)
        n_az, n_el = p["n_az"], p["n_el_slots"]
        flat = torch.as_tensor(p["az_slots"] * n_el + p["el_slots"], device=channels.device)
        counts = torch.zeros(n_az * n_el, device=channels.device, dtype=real_dtype)
        counts.index_add_(0, flat, torch.ones(len(flat), device=channels.device, dtype=real_dtype))
        # the azimuth window as the 1D beamformer's; NO taper over elevation: the aperture is
        # 2 (HuPR) or 4 occupied rows of 7 (cascade), and a Hann over the slot span would weight
        # the cascade's end rows -- the 9 azimuth TX and TX0, the rows that resolve elevation --
        # at 0.146 (docs/FIDELITY_FIX_RESULTS.md 3.8). Both sides take this one transform.
        w_az = fft_window("azimuth", n_az) if window else None
        w_az = np.ones(n_az) if w_az is None else w_az
        w_el = np.ones(n_el)
        weights = torch.as_tensor(np.outer(w_az, w_el).reshape(-1), device=channels.device, dtype=real_dtype)
        weights = weights / counts.clamp_min(1.0)
        plan = _GRID2D_CACHE[key] = (flat, weights[:, None], n_az, n_el, len(p["az_slots"]))
    flat, weights, n_az, n_el, n_channels = plan
    if channels.shape[-2] != n_channels:
        raise ValueError(f"expected {n_channels} channels (every TX x RX), got {channels.shape[-2]}")
    grid = channels.new_zeros((*channels.shape[:-2], n_az * n_el, channels.shape[-1]))
    grid.index_add_(-2, flat, channels)
    grid = grid * weights
    return grid.reshape(*channels.shape[:-2], n_az, n_el, channels.shape[-1])


def range_angle_elevation_slow_time_cube_frames_torch(frames_adc, radar: Radar | None = None, window: bool = True):
    """``(frames, rx, chirps, samples)`` ADC -> ``(frames, loops, range, n_angle, n_el)`` complex64: every TX on the 2D grid.

    The range FFT of every channel exactly as :func:`range_profiles_all_channels_torch`
    (Hann range window, the recording's calibration applied by the reader),
    then the 2D grid of :func:`elevation_grid_plan`, the azimuth FFT (fftshifted,
    ``radar.n_angle``) and the elevation FFT (fftshifted, ``n_el``).
    """
    import torch
    radar = _radar(radar)
    profiles = torch.stack([range_profiles_all_channels_torch(frames_adc[i], radar=radar)
                            for i in range(frames_adc.shape[0])])              # (F, loops, TX, RX, K)
    F, L, T, R, K = profiles.shape
    grid = channels_to_grid2d(profiles.reshape(F, L, T * R, K), radar, axis=-2, window=window)   # (F, L, G_az, G_el, K)
    plan = elevation_grid_plan(radar)
    moved = grid.permute(0, 1, 4, 2, 3).contiguous()                            # (F, L, K, G_az, G_el)
    out = torch.fft.fftshift(torch.fft.fft(moved, n=plan["n_angle"], dim=-2), dim=-2)
    out = torch.fft.fftshift(torch.fft.fft(out, n=plan["n_el"], dim=-1), dim=-1)
    return out.to(torch.complex64).contiguous()


def range_angle_slow_time_cube(
    frame_adc: np.ndarray,
    n_angle: int | None = None,
    remove_static_clutter: bool = False,
    radar: Radar | None = None,
) -> np.ndarray:
    """Return complex per-loop RA data with shape (64, range=256, angle=64).

    No Doppler FFT or slow-time reduction is performed. This is the
    authoritative representation for the single-loop and future all-loop
    motion-rendering experiments. The range FFT takes the range window
    (`signal_model.fft_window`) over the ADC samples; the angle FFT takes
    the azimuth window, applied on the grid by `virtual_to_grid`.
    """
    radar = _radar(radar)
    n_angle = radar.n_angle if n_angle is None else n_angle
    selected = virtual_array_slow_time(frame_adc, radar)
    if remove_static_clutter:
        selected = selected - selected.mean(axis=0, keepdims=True)
    selected = apply_fft_window(selected, "range", -1)
    range_fft = np.fft.fft(np.ascontiguousarray(selected), axis=-1)
    # The angle transform runs over the 8-element virtual array zero-padded
    # to n_angle. Taken over axis 1 of (loop, virtual, range) it is the
    # renderer's single most expensive numpy call -- not for its arithmetic
    # (16k transforms of length 64) but because that axis is strided and the
    # zero-padded working buffer is 8x the data. Moving the transform axis
    # last first, and transposing once instead of at the end, is the same
    # sequence of per-signal transforms on the same values: pocketfft plans
    # from the length alone, so the result is bitwise identical (measured on
    # a real sequence-102 frame) at 16.2 ms -> 12.3 ms.
    moved = np.ascontiguousarray(range_fft.transpose(0, 2, 1))
    angle_fft = np.fft.fftshift(np.fft.fft(moved, n=n_angle, axis=-1), axes=-1)
    return angle_fft.astype(np.complex64)


def range_profiles_all_channels_torch(frame_adc, device=None, radar: Radar | None = None) -> "torch.Tensor":
    """(64 loops, 3 TX, 4 RX, 256 range) complex64 range profiles of one frame.

    Every channel the radar recorded, range-windowed and range-FFT'd: the
    measured half of the product the renderer ships, the exact counterpart
    of the renderer's per-loop profiles (whose range PSF is the same window's
    response). `range_angle_slow_time_cube_torch` below is the beamformed
    view the metrics use, and is derivable from this.
    """
    import torch

    adc = frame_adc if isinstance(frame_adc, torch.Tensor) else torch.from_numpy(
        np.ascontiguousarray(frame_adc))
    if device is not None:
        adc = adc.to(device)
    radar = _radar(radar)
    if tuple(adc.shape) != radar.adc_frame_shape:
        raise ValueError(f"expected {radar.adc_frame_shape}, got {tuple(adc.shape)}")
    # (RX, TX, loop, range) -> (loop, TX, RX, range), the renderer's layout
    tdm = tdm_split(adc, radar)
    profiles = tdm.permute(2, 1, 0, 3).to(torch.complex64)
    return torch.fft.fft(apply_fft_window(profiles, "range", -1), dim=-1)


class ProductCube:
    """A shipped slow-time cube, whichever way it was stored, read as complex64 per frame.

    ``<name>.npy`` is float16 ``(frames, loop, tx, rx, range, 2)`` (real,
    imag) with ``<name>_scale.npy``, one float32 per frame, the factor each
    frame was divided by before rounding -- the GT ADC's four bytes per
    complex sample (a complex64 file from before this form is still read).
    ``cube[frame]`` returns the frame as complex64; ``cube.shape`` is the
    complex shape.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.array = np.load(self.path, mmap_mode="r")
        scale_path = self.path.with_name(self.path.stem + "_scale.npy")
        self.half = self.array.dtype == np.float16
        if self.half:
            if not scale_path.is_file():
                raise FileNotFoundError(f"{self.path} is half precision but {scale_path} is missing")
            self.scale = np.load(scale_path).astype(np.float32)
            self.shape = tuple(self.array.shape[:-1])
        else:
            self.scale = None
            self.shape = tuple(self.array.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, frame):
        if self.half:
            pair = np.asarray(self.array[frame], dtype=np.float32)
            frame_scale = self.scale[frame]
            if np.ndim(frame) == 0 and not isinstance(frame, slice):
                return (pair[..., 0] + 1j * pair[..., 1]).astype(np.complex64) * np.float32(frame_scale)
            return ((pair[..., 0] + 1j * pair[..., 1]).astype(np.complex64)
                    * np.asarray(frame_scale, dtype=np.float32)[(slice(None),) + (None,) * (pair.ndim - 2)])
        return np.asarray(self.array[frame], dtype=np.complex64)


def slow_time_product(range_profiles: "torch.Tensor") -> "torch.Tensor":
    """The shipped product: the per-loop range profiles themselves, complex64.

    Shipped in slow time, not Doppler: the Doppler transform is mean removal,
    the Doppler window and the FFT (`forward_render.slow_time_to_doppler`),
    and a windowed transform cannot be undone by a consumer that wants to
    remove the mean first -- so every consumer takes the transform itself
    from the one definition. Left on the device: the product writer stages
    rows there and copies them into its memmap itself.
    """
    import torch

    return range_profiles.to(torch.complex64)


def product_to_doppler(cube, remove_static_clutter: bool = True):
    """A shipped slow-time cube ``(loop, ...)`` to fftshifted Doppler, numpy or torch."""
    import torch
    from mm4DPS.forward_render import slow_time_to_doppler
    values = cube if isinstance(cube, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(cube))
    out = slow_time_to_doppler(values, dim=0, remove_static_clutter=remove_static_clutter)
    return out if isinstance(cube, torch.Tensor) else out.numpy()


def range_angle_slow_time_cube_torch(
    frame_adc,
    n_angle: int | None = None,
    remove_static_clutter: bool = False,
    device=None,
    dtype=None,
    radar: Radar | None = None,
):
    """The same beamforming in torch, for callers that do not need bitwise.

    Measured against the numpy form above on sequence-102 frame 0, in the
    renderer's configuration (``remove_static_clutter=False``):

    * on the **CPU** in complex128 it reproduces the numpy cube **bitwise**;
    * on the **GPU** it differs on 0.0001% of elements, by **5.441e-16**
      relative to the cube's peak (2.3e-10 absolute against a peak of
      4.3e+05) -- cuFFT's reduction order, surfacing through the final cast
      to complex64;
    * in complex64 throughout, 1.83e-07.

    Speed: **0.69 ms against 16.2 ms** with host transfers, 0.27 ms if the
    result stays on the GPU.

    With ``remove_static_clutter=True`` the two also differ by ~1e-07,
    because numpy and torch reduce the complex64 slow-time mean in different
    orders. The renderer does not use that option.

    **The renderer calls this.** The consequence is stated plainly because
    it is a real change to what the measured products mean: they are no
    longer bitwise identical to the historical numpy values, but differ from
    them by up to 5.441e-16 relative to the cube's peak. They remain
    **bitwise reproducible run to run**, because cuFFT is deterministic for
    a fixed plan, shape and device -- so the property the gate actually
    relies on (a measured product that moves between two runs indicates a
    bug) survives; the reference point moved once, here.

    The numpy form above stays as the arbiter and is what the suite compares
    against.
    """
    import torch

    adc = frame_adc if isinstance(frame_adc, torch.Tensor) else torch.from_numpy(
        np.ascontiguousarray(frame_adc)
    )
    if device is not None:
        adc = adc.to(device)
    radar = _radar(radar)
    if tuple(adc.shape) != radar.adc_frame_shape:
        raise ValueError(
            f"expected {radar.adc_frame_shape}, "
            f"got {tuple(adc.shape)}"
        )
    n_angle = radar.n_angle if n_angle is None else n_angle
    tdm = tdm_split(adc, radar)
    selected = torch.cat([tdm[:, tx] for tx in radar.azimuth_tx], dim=0).permute(1, 0, 2)
    selected = virtual_to_grid(selected, radar, axis=1)
    if remove_static_clutter:
        selected = selected - selected.mean(dim=0, keepdim=True)
    selected = apply_fft_window(selected.to(dtype or torch.complex128), "range", -1)
    range_fft = torch.fft.fft(selected, dim=-1)
    # Mirror the numpy form: move the virtual-array axis last *before* the
    # angle transform and return a contiguous cube. Returning the permuted
    # view instead is not merely untidy — the next FFT in the renderer runs
    # over a strided tensor and rounds differently, which showed up as
    # 5e-08 on exactly the clutter-removed measured products while the
    # others stayed bitwise.
    moved = range_fft.permute(0, 2, 1).contiguous()
    angle_fft = torch.fft.fftshift(
        torch.fft.fft(moved, n=n_angle, dim=-1), dim=-1,
    )
    return angle_fft.to(torch.complex64).contiguous()


def range_angle_map(
    frame_adc: np.ndarray,
    n_angle: int | None = None,
    remove_static_clutter: bool = False,
    radar: Radar | None = None,
) -> np.ndarray:
    """Return mean-magnitude RA with shape (range=256, local-angle=64).

    TX0 and TX2 form the eight-element lambda/2 virtual ULA used by HuPR.
    The horizontal board's local angle is world azimuth; after the physical
    90-degree board roll, the vertical board's local angle is world elevation.

    Static clutter removal is optional and disabled for scene reconstruction
    because walls/furniture are the signal.
    """
    cube = range_angle_slow_time_cube(
        frame_adc, n_angle=n_angle,
        remove_static_clutter=remove_static_clutter, radar=radar,
    )
    return np.abs(cube).mean(axis=0).astype(np.float32)


def single_loop_range_angle_map(
    frame_adc: np.ndarray,
    loop_index: int = 0,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> np.ndarray:
    """Return one TDM slow-time loop as a (range=256, angle=64) RA map.

    Static clutter removal is intentionally unavailable here: subtracting a
    mean across a single selected loop would erase the sample. Select first,
    compare against the renderer's single static range-profile prediction,
    and reserve slow-time filtering/Doppler processing for the all-loop stage.
    """
    if not 0 <= loop_index < TDM_LOOPS:
        raise ValueError(f"loop_index must be in [0, {TDM_LOOPS - 1}]")
    cube = range_angle_slow_time_cube(
        frame_adc, n_angle=n_angle, remove_static_clutter=False, radar=radar,
    )
    return np.abs(cube[loop_index]).astype(np.float32)


def range_bin_spacing_m(radar: Radar | None = None) -> float:
    """c * fs / (2 * slope * N): the current dataset's chirp unless told otherwise."""
    return _radar(radar).range_bin_m


def range_keep_mask(
    minimum_range_m: float = HUPR_NEAR_FIELD_M,
    n_range: int | None = None,
    radar: Radar | None = None,
) -> np.ndarray:
    """Return bins retained after the single-chip near-field guard."""
    radar = _radar(radar)
    n_range = radar.samples if n_range is None else n_range
    if minimum_range_m < 0:
        raise ValueError("minimum_range_m must be non-negative")
    if n_range <= 0:
        raise ValueError("n_range must be positive")
    ranges = np.arange(n_range, dtype=np.float64) * range_bin_spacing_m(radar)
    return ranges >= minimum_range_m


# ----------------------------------------------------------------------
# Batched GPU forms, for callers that process a whole sequence at once.
#
# The per-frame `*_torch` functions above still take one frame, which leaves
# the caller with a Python loop over frames and launches too small to fill the
# card. `detect-person-motion` ran at 0.0% GPU for exactly that reason. These
# take a leading frame axis and keep everything on the device.
# ----------------------------------------------------------------------


def decode_dca1000_frames_torch(raw_int16, radar: Radar | None = None):
    """Decode a batch of frames from raw int16 words.

    ``raw_int16`` is ``(frames, NUM_RX*PHYSICAL_CHIRPS*ADC_SAMPLES*2)``; the
    result is ``(frames, NUM_RX, PHYSICAL_CHIRPS, ADC_SAMPLES)`` complex64.
    The per-frame form's operations with a batch axis in front: int16 ->
    float32 is exact and the rest is data movement, so this is bitwise what
    :func:`decode_dca1000_frame` produces, frame for frame.
    """
    import torch

    radar = _radar(radar)
    frames = raw_int16.shape[0]
    expected = radar.raw_words_per_frame
    if raw_int16.shape[-1] != expected:
        raise ValueError(
            f"expected {expected} int16 words per frame, got {raw_int16.shape[-1]}")
    grouped = raw_int16.reshape(frames, -1, 4)
    lvds_i = grouped[:, :, :2].reshape(frames, -1).to(torch.float32)
    lvds_q = grouped[:, :, 2:].reshape(frames, -1).to(torch.float32)
    stream = torch.complex(lvds_i, lvds_q)
    blocks = stream.reshape(frames, radar.chirps, radar.n_rx, radar.samples)
    return blocks.permute(0, 2, 1, 3).contiguous()


def range_angle_slow_time_cube_frames_torch(
    frames_adc, n_angle: int | None = None, dtype=None, radar: Radar | None = None,
    remove_static_clutter: bool = False,
):
    """``(frames, NUM_RX, PHYSICAL_CHIRPS, ADC)`` -> ``(frames, 64, 256, 64)``.

    The batched form of :func:`range_angle_slow_time_cube_torch`, same
    operations in the same order with a leading frame axis, and the same
    accuracy statement: complex128 through the transforms, cast to complex64
    on the way out, differing from the numpy arbiter by ~5e-16 relative to
    the cube's peak through cuFFT's reduction order.
    """
    import torch

    radar = _radar(radar)
    if tuple(frames_adc.shape[1:]) != radar.adc_frame_shape:
        raise ValueError(
            f"expected (frames, {', '.join(map(str, radar.adc_frame_shape))}), "
            f"got {tuple(frames_adc.shape)}")
    slots = [frames_adc[:, :, k::radar.n_tx, :] for k in range(radar.n_tx)]
    by_tx = [None] * radar.n_tx
    for slot, tx in enumerate(radar.tx_order):
        by_tx[tx] = slots[slot]
    n_angle = radar.n_angle if n_angle is None else n_angle
    tdm = torch.stack(by_tx, dim=2)
    selected = torch.cat(
        [tdm[:, :, tx] for tx in radar.azimuth_tx], dim=1).permute(0, 2, 1, 3)
    selected = virtual_to_grid(selected, radar, axis=2)
    if remove_static_clutter:
        selected = selected - selected.mean(dim=1, keepdim=True)      # the slow-time mean, per frame
    selected = apply_fft_window(selected.to(dtype or torch.complex128), "range", -1)
    range_fft = torch.fft.fft(selected, dim=-1)
    moved = range_fft.permute(0, 1, 3, 2).contiguous()
    angle_fft = torch.fft.fftshift(
        torch.fft.fft(moved, n=n_angle, dim=-1), dim=-1)
    return angle_fft.to(torch.complex64).contiguous()


DATASETS = ("hupr", "rtpose")

# The two-way element pattern's 0 dB crossings either side of boresight,
# measured from assets/antenna_pattern/MMWCAS/{tx1_76,rx1_76}.npy: E-plane is
# ELEVATION, H-plane is AZIMUTH, index 180 is boresight.
#
# Elevation is NARROW -- about as narrow as the camera's azimuth reach. Both
# restrictions land on the same window, bins 21-43 of 64, so measured energy
# outside it is outside the camera's reach AND outside the radar's beam. The
# strong measured elevation tail at bin ~19 (-24 deg) is in exactly that
# region: a physically correct render will not put energy there, and matching
# it by moving the scene would be fitting an artefact of the array response.
RADAR_FOV_ZERO_DB_DEG = {radar.name: radar.fov_zero_db_deg for radar in HUPR.radars}


def fov_zero_db_deg(dataset: Dataset | None = None) -> dict[str, float]:
    return {radar.name: radar.fov_zero_db_deg
            for radar in (dataset or current_dataset()).radars}

# The alignment is expressed against this base pose: the midpoint of the two
# board centres, looking down +Y. `radar_centre_and_boresight` reproduces it
# from the configs; it is repeated here so this module has no import cycle.
HUPR_BASE_ORIGIN = np.array([0.0, 0.0, 0.0])


def dataset_flag(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """``--dataset``: required, and with no default, on purpose."""
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    return parser


ADC_FRAME_SHAPE = _IWR1843.adc_frame_shape
RAW_WORDS_PER_FRAME = _IWR1843.raw_words_per_frame
# 2: the payload changed from (decoded frame, RA cube) to raw int16 words.
MANIFEST_VERSION = 2






# --------------------------------------------------------------------
# SMPL: the body model and its HuPR joint mapping
# --------------------------------------------------------------------

DEFAULT_MODEL = SMPL_MODEL_DIR / "SMPL_NEUTRAL.npz"

# SMPL's 24 joints, by name, in its own order.
SMPL_JOINT_INDEX = {
    "pelvis": 0, "L_Hip": 1, "R_Hip": 2, "spine1": 3, "L_Knee": 4, "R_Knee": 5,
    "spine2": 6, "L_Ankle": 7, "R_Ankle": 8, "spine3": 9, "L_Foot": 10,
    "R_Foot": 11, "Neck": 12, "L_Collar": 13, "R_Collar": 14, "Head": 15,
    "L_Shoulder": 16, "R_Shoulder": 17, "L_Elbow": 18, "R_Elbow": 19,
    "L_Wrist": 20, "R_Wrist": 21, "L_Hand": 22, "R_Hand": 23,
}

# HuPR's 14 keypoints, verbatim from the official loader
# (hupr_official/datasets/base.py). Note that the legs and the arms run in
# opposite order: indices 0-2 are the *right* leg while 8-10 are the *left* arm.
# Pairing them as two same-side chains, which the shape of the list invites,
# mirrors the upper body against the lower one, and neither the joint error nor
# a one-sided surface fit can see it - the SMPL template is near enough
# symmetric that both readings score the same to within half a millimetre. The
# names settle it, so there is nothing here to infer.
HUPR_KEYPOINT_NAMES = (
    "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "Neck", "Head",
    "L_Shoulder", "L_Elbow", "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist",
)


def smpl_to_hupr_indices() -> list[int]:
    """SMPL joint index for each of HuPR's 14, in HuPR's order."""
    return [SMPL_JOINT_INDEX[name] for name in HUPR_KEYPOINT_NAMES]


def axis_angle_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Rodrigues, batched over any leading dimensions. (...,3) -> (...,3,3)."""
    if rotation.shape[-1] != 3:
        raise ValueError(f"expected axis-angle (...,3), got {tuple(rotation.shape)}")
    x, y, z = rotation.unbind(dim=-1)
    zero = torch.zeros_like(x)
    # Skew of the *unnormalised* vector, paired below with sinc-style
    # coefficients. Normalising first and multiplying by sin(angle) would make
    # the linear term quadratic in the rotation near zero, which sends the
    # gradient to zero exactly where the fit starts from - the pose optimiser
    # would never leave its initialisation.
    skew = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    angle = rotation.norm(dim=-1, keepdim=True)
    small = angle < 1e-4
    safe = torch.where(small, torch.ones_like(angle), angle)
    # sin(t)/t and (1-cos t)/t^2, both taken to their Taylor limits near zero so
    # the expression stays finite and differentiable at the identity.
    sinc = torch.where(small, 1.0 - angle ** 2 / 6.0, torch.sin(safe) / safe)
    versine = torch.where(
        small, 0.5 - angle ** 2 / 24.0, (1.0 - torch.cos(safe)) / safe ** 2,
    )
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    eye = eye.expand(*rotation.shape[:-1], 3, 3)
    return eye + sinc[..., None] * skew + versine[..., None] * (skew @ skew)


class SmplBody(torch.nn.Module):
    """SMPL, loaded from the converted npz, differentiable in shape and pose."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, dtype=torch.float32):
        super().__init__()
        data = np.load(model_path)
        self.register_buffer("v_template", torch.as_tensor(data["v_template"], dtype=dtype))
        self.register_buffer("shapedirs", torch.as_tensor(data["shapedirs"], dtype=dtype))
        self.register_buffer("posedirs", torch.as_tensor(data["posedirs"], dtype=dtype))
        self.register_buffer("joint_regressor", torch.as_tensor(data["joint_regressor"], dtype=dtype))
        self.register_buffer("skinning_weights", torch.as_tensor(data["skinning_weights"], dtype=dtype))
        self.register_buffer("faces", torch.as_tensor(data["faces"], dtype=torch.long))
        self.register_buffer("parents", torch.as_tensor(data["parents"], dtype=torch.long))
        self.joint_count = int(self.joint_regressor.shape[0])
        self.vertex_count = int(self.v_template.shape[0])

    def forward(
        self,
        betas: torch.Tensor,
        pose: torch.Tensor,
        translation: torch.Tensor | None = None,
        return_global_rotations: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Vertices and joints for a batch of bodies.

        ``betas`` is (batch,10), ``pose`` is (batch,24,3) axis-angle with the
        root rotation first, ``translation`` is (batch,3) in metres.
        Returns vertices (batch,6890,3) and joints (batch,24,3), and with
        ``return_global_rotations`` also each joint's global rotation
        (batch,24,3,3), which a keypoint offset needs so it follows the limb.
        """
        batch = betas.shape[0]
        if pose.shape != (batch, self.joint_count, 3):
            raise ValueError(f"pose must be (batch,{self.joint_count},3), got {tuple(pose.shape)}")

        shaped = self.v_template + torch.einsum("bl,vcl->bvc", betas, self.shapedirs)
        rest_joints = torch.einsum("jv,bvc->bjc", self.joint_regressor, shaped)

        rotations = axis_angle_to_matrix(pose)
        # Pose blend shapes act on the deviation of each non-root joint's
        # rotation from the identity, flattened row-major, which is how the
        # released posedirs are indexed.
        eye = torch.eye(3, dtype=pose.dtype, device=pose.device)
        residual = (rotations[:, 1:] - eye).reshape(batch, -1)
        posed = shaped + torch.einsum("bl,vcl->bvc", residual, self.posedirs)

        # Walk the kinematic tree: each joint's global transform is its parent's
        # composed with its own rotation about its rest offset from the parent.
        relative = rest_joints.clone()
        relative[:, 1:] = rest_joints[:, 1:] - rest_joints[:, self.parents[1:]]
        globals_: list[torch.Tensor] = []
        for joint in range(self.joint_count):
            local = torch.cat([
                torch.cat([rotations[:, joint], relative[:, joint, :, None]], dim=-1),
                torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pose.dtype, device=pose.device)
                .expand(batch, 1, 4),
            ], dim=-2)
            parent = int(self.parents[joint])
            globals_.append(local if parent < 0 else globals_[parent] @ local)
        transforms = torch.stack(globals_, dim=1)

        # Remove the rest pose, so a zero pose leaves the template untouched.
        rest_homogeneous = torch.cat(
            [rest_joints, torch.zeros(batch, self.joint_count, 1, dtype=pose.dtype, device=pose.device)],
            dim=-1,
        )
        offset = torch.zeros_like(transforms)
        offset[:, :, :, 3] = torch.einsum("bjmn,bjn->bjm", transforms, rest_homogeneous)
        relative_transforms = transforms - offset

        blended = torch.einsum("vj,bjmn->bvmn", self.skinning_weights, relative_transforms)
        homogeneous = torch.cat([posed, torch.ones_like(posed[..., :1])], dim=-1)
        vertices = torch.einsum("bvmn,bvn->bvm", blended, homogeneous)[..., :3]
        joints = torch.einsum("jv,bvc->bjc", self.joint_regressor, vertices)

        if translation is not None:
            vertices = vertices + translation[:, None, :]
            joints = joints + translation[:, None, :]
        if return_global_rotations:
            return vertices, joints, transforms[:, :, :3, :3]
        return vertices, joints


# --------------------------------------------------------------------
# RadProposer: the published point proposals
# --------------------------------------------------------------------

RADPROPOSER_PADDED_TX = 4
RADPROPOSER_SINGLE_RADAR_SHAPE = (
    TDM_LOOPS,
    RADPROPOSER_PADDED_TX,
    NUM_RX,
    ADC_SAMPLES,
)


def _as_complex_tensor(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.from_numpy(values)
    if not torch.is_complex(tensor):
        raise ValueError("RadProPoser input must be complex")
    return tensor


def _radproposer_window(values: torch.Tensor, dimension: int) -> torch.Tensor:
    """The pipeline's window for one RadProposer axis: range, RX (azimuth), padded TX (elevation), Doppler.

    The padded-TX axis is windowed over the ``NUM_TX`` real entries only; the
    zero padding stays zero.
    """
    axis = {-1: "range", -2: "azimuth", -3: "elevation", -4: "doppler"}[dimension]
    if axis == "elevation":
        window = fft_window("elevation", NUM_TX)
        if window is None:
            return values
        taper = torch.zeros(values.shape[-3], dtype=values.real.dtype, device=values.device)
        taper[:NUM_TX] = torch.from_numpy(window).to(taper)
        return values * taper[:, None, None]
    return apply_fft_window(values, axis, dimension)


def _official_fft_4d(values: torch.Tensor, skip_range: bool = False) -> torch.Tensor:
    """The official forward-normalized, shifted 4-D FFT, each axis windowed first.

    HuPR's released tensor is unwindowed; this is that tensor with the
    pipeline's window on every axis (``MM4DPS_FFT_WINDOW=none`` restores the
    released form). ``skip_range`` is for input that is already range-FFT'd.
    """
    result = values
    for dimension in (-1, -2, -3, -4):
        if dimension == -1 and skip_range:
            continue
        result = torch.fft.fftshift(
            torch.fft.fft(_radproposer_window(result, dimension), dim=dimension, norm="forward"),
            dim=dimension,
        )
    return result


def adc_frame_to_radproposer_tensor(
    frame_adc: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Convert one decoded HuPR radar frame to the official complex tensor.

    Args:
        frame_adc: Complex data with axes ``(RX=4, physical chirp=192, ADC=256)``.

    Returns:
        Complex64 tensor with axes ``(Doppler=64, padded-TX=4, RX=4, range=256)``.
    """
    adc = _as_complex_tensor(frame_adc)
    expected = (NUM_RX, PHYSICAL_CHIRPS, ADC_SAMPLES)
    if tuple(adc.shape) != expected:
        raise ValueError(f"expected ADC shape {expected}, got {tuple(adc.shape)}")

    padded = torch.zeros(
        RADPROPOSER_SINGLE_RADAR_SHAPE,
        dtype=adc.dtype,
        device=adc.device,
    )
    for transmitter in range(NUM_TX):
        padded[:, transmitter] = adc[:, transmitter::NUM_TX].permute(1, 0, 2)

    padded = padded - padded.mean(dim=0, keepdim=True)
    return _official_fft_4d(padded).to(torch.complex64)


def range_profiles_to_radproposer_tensor(
    range_profiles: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Convert renderer range-FFT profiles to the official complex tensor.

    Renderer profiles have already undergone the unnormalized ADC/range FFT.
    Static removal commutes with this linear FFT, so the slow-time mean is
    removed in range-profile space.  The official range ``fftshift`` and
    ``norm='forward'`` scaling are then restored before the RX, padded-TX, and
    Doppler FFTs.

    Args:
        range_profiles: Complex profiles with axes
            ``(slow-time=64, TX=3, RX=4, unshifted-range=256)``.

    Returns:
        Complex64 tensor with the same contract as
        :func:`adc_frame_to_radproposer_tensor`.
    """
    profiles = _as_complex_tensor(range_profiles)
    expected = (TDM_LOOPS, NUM_TX, NUM_RX, ADC_SAMPLES)
    if tuple(profiles.shape) != expected:
        raise ValueError(
            f"expected range-profile shape {expected}, got {tuple(profiles.shape)}"
        )

    padded = torch.zeros(
        RADPROPOSER_SINGLE_RADAR_SHAPE,
        dtype=profiles.dtype,
        device=profiles.device,
    )
    padded[:, :NUM_TX] = profiles
    padded = padded - padded.mean(dim=0, keepdim=True)

    # The renderer already performed an unnormalized range FFT (with the
    # range window's PSF). Reproduce the official forward-normalized range
    # result without applying a second FFT, then the other three axes.
    padded = torch.fft.fftshift(padded, dim=-1) / ADC_SAMPLES
    return _official_fft_4d(padded, skip_range=True).to(torch.complex64)
