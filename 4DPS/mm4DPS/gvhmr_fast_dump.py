"""Headless GVHMR for one radar sequence, from the frames in memory: -> vertices_camera npz.

Runs in the GVHMR environment with the GVHMR checkout as the working
directory (`mm4DPS.gvhmr` launches it so); it imports `hmr4d` from there.

The demo pipeline (`hupr_dump.py`) encodes a video, then YOLO, ViTPose and
the HMR2 feature extractor each decode it again and crop frame by frame.
This tool takes the frames as an array (a directory of images, or every
frame of a HuPR sequence out of `frames.zip`), never touches a video, runs
the same three models with the same weights -- YOLO in half precision,
ViTPose and the ViT feature extractor under autocast, larger batches --
then GVHMR in static-camera mode and the SMPL vertices in the camera frame,
exactly as `hupr_dump.py` does after its preprocess.

    python mm4DPS/gvhmr_fast_dump.py --hupr-zip /path/frames.zip --sequence 256 --out seq256.npz
    python mm4DPS/gvhmr_fast_dump.py --frames /path/camera/left --f_mm 25.6545 --out seq183.npz
"""
import argparse
import io
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import initialize_config_module, compose
import hydra

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor
from hmr4d.utils.preproc.vitfeat_extractor import get_batch
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.seq_utils import (frame_id_to_mask, rearrange_by_mask, get_frame_id_list_from_mask,
                                   linear_interpolate_frame_ids)
from hmr4d.utils.net_utils import moving_average_smooth


def load_frames(args) -> np.ndarray:
    """(F, H, W, 3) RGB uint8."""
    if args.hupr_zip:
        archive = zipfile.ZipFile(args.hupr_zip)
        prefix = f"frames/single_{args.sequence}/processed/images/"
        names = sorted(n for n in archive.namelist() if n.startswith(prefix) and n.endswith(".jpg"))
        payloads = [archive.read(n) for n in names]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as pool:
            frames = list(pool.map(lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1], payloads))
    else:
        paths = sorted(Path(args.frames).glob("*.png")) or sorted(Path(args.frames).glob("*.jpg"))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as pool:
            frames = list(pool.map(lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR)[..., ::-1], paths))
    if not frames:
        raise FileNotFoundError("no frames")
    return np.ascontiguousarray(np.stack(frames))


def track_one_person(tracker: Tracker, frames: np.ndarray) -> torch.Tensor:
    """`Tracker.get_one_track` on frames in memory: the longest / largest track, gaps interpolated, smoothed."""
    history = []
    for frame in frames:
        result = tracker.yolo.track(frame[..., ::-1], persist=True, device="cuda", conf=0.5, classes=0,
                                    verbose=False, half=not tracker.fp32)[0]
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            history.append([{"id": ids[i], "bbx_xyxy": boxes[i]} for i in range(len(ids))])
        else:
            history.append([])
    length, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
    # Tracker.sort_track_length, with the frame geometry given instead of read from a video
    from collections import defaultdict
    id_to_frame_ids, id_to_bbx = defaultdict(list), defaultdict(list)
    for frame_id, dets in enumerate(history):
        for det in dets:
            id_to_frame_ids[det["id"]].append(frame_id)
            id_to_bbx[det["id"]].append(det["bbx_xyxy"])
    if not id_to_frame_ids:
        raise RuntimeError("YOLO found no person in the sequence")
    id_to_bbx = {k: np.array(v) for k, v in id_to_bbx.items()}
    id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
    id_area = {k: float(((v[:, 2:] - v[:, :2]).prod(1)).sum()) for k, v in id_to_bbx.items()}
    # the demo ranks by length first, then by the summed area within the longest ones
    best_len = max(id_length.values())
    candidates = [k for k, n in id_length.items() if n == best_len]
    track_id = max(candidates, key=lambda k: id_area[k])
    frame_ids = torch.tensor(id_to_frame_ids[track_id])
    boxes = torch.tensor(id_to_bbx[track_id])
    mask = frame_id_to_mask(frame_ids, length)
    one = rearrange_by_mask(boxes, mask)
    one = linear_interpolate_frame_ids(one, get_frame_id_list_from_mask(~mask))
    assert (one.sum(1) != 0).all()
    one = moving_average_smooth(one, window_size=5, dim=0)
    one = moving_average_smooth(one, window_size=5, dim=0)
    return one.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hupr-zip", type=str, default=None, help="HuPR frames.zip; with --sequence")
    parser.add_argument("--sequence", type=int, default=None)
    parser.add_argument("--frames", type=str, default=None, help="a directory of png/jpg frames (RT-Pose camera/left)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--f_mm", type=float, default=None, help="full-frame-equivalent focal length (RT-Pose 25.6545)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--fp32", action="store_true", help="the ViTs in float32 (the demo's precision)")
    parser.add_argument("--full-res", action="store_true",
                        help="crop from the frames at full resolution (the demo crops at half: img_ds 0.5, "
                             "and a full-resolution crop moves the mesh ~4 cm median)")
    parser.add_argument("--yolo-fp32", action="store_true")
    parser.add_argument("--resize", default="area", choices=("area", "cubic", "linear"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    times = {}
    t0 = time.time()

    # every model first, so the frame load overlaps nothing it need not
    def onto_the_gpu(build, tries: int = 24, wait_s: float = 5.0):
        """A model load that waits out the chain's preprocessing peak on the shared GPU.

        This tool starts together with the reference stage and shares its
        card; a transient peak there (the LiDAR scaffold's neighbour search
        once took the whole card) made the ~26 MiB weight copies fail. The
        load is retried for up to two minutes rather than failing the run.
        """
        for attempt in range(tries):
            try:
                return build()
            except torch.cuda.OutOfMemoryError:
                if attempt == tries - 1:
                    raise
                torch.cuda.empty_cache()
                time.sleep(wait_s)

    tracker = onto_the_gpu(Tracker)
    tracker.fp32 = args.yolo_fp32
    vitpose = onto_the_gpu(VitPoseExtractor)
    extractor = onto_the_gpu(Extractor)
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=["video_name=fast", "static_cam=True", "verbose=False",
                                                     "use_dpvo=False", f"f_mm={args.f_mm}"])
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = onto_the_gpu(lambda: model.eval().cuda())
    smplx = onto_the_gpu(lambda: make_smplx("supermotion").cuda())
    smplx2smpl = onto_the_gpu(lambda: torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda())
    times["models"] = time.time() - t0

    t = time.time()
    frames = load_frames(args)
    length, height, width = frames.shape[:3]
    times["frames"] = time.time() - t

    t = time.time()
    bbx_xyxy = track_one_person(tracker, frames)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    times["track"] = time.time() - t

    t = time.time()
    # the demo's crops, from the frames as they are (img_ds=1: no half-scale
    # decode first); `get_batch` blurs and crops to 256 on the CPU
    if not args.full_res:
        # the demo decodes the video at half scale (ffmpeg's scale filter, bicubic) before cropping
        interp = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR}[args.resize]
        small = np.stack([cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=interp) for f in frames])
        imgs, bbx_xys_ds = get_batch(small, bbx_xys * 0.5, img_ds=1.0, path_type="np")
        bbx_xys_ds = bbx_xys_ds * 2.0
    else:
        imgs, bbx_xys_ds = get_batch(frames, bbx_xys, img_ds=1.0, path_type="np")
    times["crops"] = time.time() - t

    # the two ViTs in half precision: their forward under autocast, the
    # outputs back in float32 for the post-processing that follows
    class Half(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *a, **k):
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.inner(*a, **k)
            return out.float() if torch.is_tensor(out) else out
    if not args.fp32:
        vitpose.pose = Half(vitpose.pose)
        extractor.extractor = Half(extractor.extractor)
    t = time.time()
    with torch.no_grad():
        kp2d = vitpose.extract(imgs, bbx_xys_ds)
    times["vitpose"] = time.time() - t
    t = time.time()
    with torch.no_grad():
        f_imgseq = extractor.extract_video_features(imgs, bbx_xys_ds).float()
    times["vitfeat"] = time.time() - t

    t = time.time()
    if args.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, args.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    data = {
        "length": torch.tensor(length), "bbx_xys": bbx_xys, "kp2d": kp2d, "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(length, 1, 1)), "f_imgseq": f_imgseq,
    }
    with torch.no_grad():
        pred = detach_to_cpu(model.predict(data, static_cam=True))
        out = smplx(**to_cuda(pred["smpl_params_incam"]))
        verts = torch.stack([torch.matmul(smplx2smpl, v) for v in out.vertices]).cpu().numpy().astype(np.float32)
    times["gvhmr"] = time.time() - t
    np.savez_compressed(args.out, vertices_camera=verts)
    if args.debug:
        torch.save({"bbx_xys": bbx_xys, "kp2d": kp2d, "f_imgseq": f_imgseq}, str(args.out) + ".debug.pt")
    times["total"] = time.time() - t0
    Log.info(f"[fast_dump] vertices_camera {verts.shape} -> {args.out}  "
             + " ".join(f"{k} {v:.1f}s" for k, v in times.items()))


if __name__ == "__main__":
    main()
