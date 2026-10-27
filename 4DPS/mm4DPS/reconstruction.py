"""Monocular depth -> geometry: the scene cloud, the multi-view scaffold, and
the person sequence lifted from it.

Grouped by the dependency graph, not by theme: person_sequence and scaffold
both import scene_pointcloud, so keeping the three together leaves no
cross-module import to break.
"""

from __future__ import annotations

import ast
import io
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import time
import zipfile


from mm4DPS.data import (
    BYTES_PER_FRAME,
    decode_dca1000_frame,
    find_archive,
    load_adc_frame,
    range_angle_map,
    single_loop_range_angle_map)

from PIL import Image, ImageDraw


from mm4DPS.data import dataset_flag
from mm4DPS import (
    DATASET_ROOT,
    DEPTH_ANYTHING_PATH,
    HUPR_OFFICIAL_ROOT,
    WORKSPACE)


from numpy.lib.format import open_memmap

import argparse, sys

import matplotlib
from pathlib import Path
from mm4DPS import Dataset, dispatch


# ----------------------------------------------------------------------
# scene_pointcloud: Create a HuPR metric monocular-depth scene cloud with oriented normals.
# ----------------------------------------------------------------------
matplotlib.use("Agg")



# The Large metric-depth checkpoint, resolved through hupr_paths so it follows
# HF_HOME. This default used to point at the *Small* model inside a different
# project's cache: every driver passes --model-path explicitly, so it was
# masked, but omitting the flag silently downgraded the depth model, and Small
# against Large is the difference between 6 and 3 bad joints per 420.
DEFAULT_MODEL = DEPTH_ANYTHING_PATH

DEFAULT_OFFICIAL_CODE = Path(
    HUPR_OFFICIAL_ROOT
)

_DEPTH_RUNTIME: dict[str, object] = {}

_ANNOTATION_CACHE: dict[tuple[str, str], tuple[list[int], list]] = {}



_FRAMES_ZIP: dict[str, zipfile.ZipFile] = {}


def read_rgb_from_zip(dataset_root: Path, sequence: int, frame: int) -> np.ndarray:
    member = f"frames/single_{sequence}/processed/images/{frame:09d}.jpg"
    # opened once per process: the central directory of 166k members is
    # 0.5 s to parse, and the scaffold reads four frames
    key = str(dataset_root / "frames.zip")
    if key not in _FRAMES_ZIP:
        _FRAMES_ZIP[key] = zipfile.ZipFile(key)
    try:
        payload = _FRAMES_ZIP[key].read(member)
    except KeyError as exc:
        raise FileNotFoundError(f"{member} not found in frames.zip") from exc
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))



def infer_metric_depths(rgbs: list[np.ndarray], model_path: Path) -> list[np.ndarray]:
    """`infer_metric_depth` over several images, one forward each.

    One image per forward on purpose: a batched forward moves the depth by
    up to 2.8 mm (the network's kernels pick other algorithms for another
    batch shape), which moves the voxelised cloud's membership, and saves
    nothing -- the 0.2 s per image is the model, the 2 s is its loading.
    """
    return [infer_metric_depth(rgb, model_path) for rgb in rgbs]


def infer_metric_depth(rgb: np.ndarray, model_path: Path) -> np.ndarray:
    """Run a local Depth Anything V2 metric checkpoint and return meters."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError(
            "Depth inference needs transformers. Install the HuPR environment "
            "dependencies or pass --depth-npy with a cached metric depth map."
        ) from exc
    if not model_path.is_dir():
        raise FileNotFoundError(f"local depth model snapshot not found: {model_path}")
    cache_key = str(model_path.resolve())
    if _DEPTH_RUNTIME.get("key") != cache_key:
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        _DEPTH_RUNTIME.update(key=cache_key, processor=processor, model=model, device=device)
    processor = _DEPTH_RUNTIME["processor"]
    model = _DEPTH_RUNTIME["model"]
    device = _DEPTH_RUNTIME["device"]
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        post = processor.post_process_depth_estimation(
            outputs, target_sizes=[rgb.shape[:2]]
        )[0]["predicted_depth"]
    return post.detach().cpu().numpy().astype(np.float32)



def parse_split_ids(config_path: Path, split: str) -> list[int]:
    match = re.search(rf"{split}Name:\s*(\[.*?\])", config_path.read_text(), re.S)
    if not match:
        raise ValueError(f"cannot find {split}Name in {config_path}")
    return list(ast.literal_eval(match.group(1)))



def frame_annotation(
    dataset_root: Path,
    official_code: Path,
    sequence: int,
    frame: int,
) -> dict | None:
    config = official_code / "config/mscsa_prgcn.yaml"
    for split in ("train", "val", "test"):
        cache_key = (str(dataset_root.resolve()), split)
        if cache_key not in _ANNOTATION_CACHE:
            ids = parse_split_ids(config, split)
            if sequence not in ids:
                continue                      # not this split: its annotations stay unread
            annotations = json.loads(
                (dataset_root / f"annotations/hrnet_annot_{split}.json").read_text()
            )
            _ANNOTATION_CACHE[cache_key] = (ids, annotations)
        ids, annotations = _ANNOTATION_CACHE[cache_key]
        if sequence not in ids:
            continue
        return annotations[ids.index(sequence)][frame]
    return None



def camera_intrinsics(height: int, width: int, hfov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0],
         [0.0, focal, (height - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )



def mask_bbox(depth: np.ndarray, bbox: list[float], padding_frac: float = 0.12) -> np.ndarray:
    """Remove the annotated person from a depth map (bbox is x0,y0,x1,y1)."""
    masked = depth.copy()
    x0, y0, x1, y1 = map(float, bbox)
    dx, dy = (x1 - x0) * padding_frac, (y1 - y0) * padding_frac
    h, w = depth.shape
    xa, xb = max(0, int(x0 - dx)), min(w, int(math.ceil(x1 + dx)))
    ya, yb = max(0, int(y0 - dy)), min(h, int(math.ceil(y1 + dy)))
    masked[ya:yb, xa:xb] = np.nan
    return masked



def person_depth_mask(
    depth: np.ndarray,
    bbox: list[float],
    joints: list[list[float]],
    annotation_to_native_scale_xy: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """Return an interpretable bbox+depth person proxy mask.

    HuPR provides pose keypoints and a bounding box but no segmentation. The
    keypoint depths robustly estimate the person's depth; pixels are retained
    only inside the annotated box and within an adaptive metric band. This
    removes the distant wall/floor visible through the rectangular box without
    introducing an opaque learned segmentation model.
    """
    h, w = depth.shape
    x0, y0, x1, y1 = map(float, bbox)
    xa, xb = max(0, int(x0)), min(w, int(math.ceil(x1)))
    ya, yb = max(0, int(y0)), min(h, int(math.ceil(y1)))
    sx, sy = annotation_to_native_scale_xy
    native_joints = [
        (int(round(float(joint_x) * sx)), int(round(float(joint_y) * sy)))
        for joint_x, joint_y in joints
    ]
    samples = []
    for u, v in native_joints:
        if not (0 <= u < w and 0 <= v < h):
            continue
        patch = depth[max(0, v - 2):min(h, v + 3), max(0, u - 2):min(w, u + 3)]
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
        if valid.size:
            samples.append(float(np.median(valid)))
    if len(samples) < 3:
        raise ValueError("at least three valid joint-depth samples are required")
    centre = float(np.median(samples))
    mad = float(np.median(np.abs(np.asarray(samples) - centre)))
    tolerance = float(np.clip(max(0.55, 4.0 * mad), 0.55, 1.10))
    depth_band_mask = np.zeros(depth.shape, dtype=bool)
    region = depth[ya:yb, xa:xb]
    depth_band_mask[ya:yb, xa:xb] = (
        np.isfinite(region) & (region > 0.05) & (np.abs(region - centre) <= tolerance)
    )
    # HuPR's 14-keypoint order is legs (0..5), neck/head (6..7), right arm
    # (8..10), left arm (11..13). Build an interpretable 2-D capsule support
    # rather than accepting every same-depth pixel in the rectangular bbox.
    skeleton = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(skeleton)
    bbox_width = max(1.0, x1 - x0)
    limb_width = max(5, int(round(0.15 * bbox_width)))
    edges = [
        (0, 1), (1, 2), (3, 4), (4, 5), (0, 3),
        (0, 6), (3, 6), (6, 7),
        (6, 8), (8, 9), (9, 10),
        (6, 11), (11, 12), (12, 13),
    ]
    if len(native_joints) >= 14:
        for start, end in edges:
            draw.line((native_joints[start], native_joints[end]), fill=255, width=limb_width)
        draw.polygon(
            [native_joints[index] for index in (11, 8, 3, 0)], fill=255
        )
        joint_radius = max(3, limb_width // 2)
        for u, v in native_joints:
            draw.ellipse((u - joint_radius, v - joint_radius,
                          u + joint_radius, v + joint_radius), fill=255)
        head_u, head_v = native_joints[7]
        head_radius = max(joint_radius, int(round(0.16 * bbox_width)))
        draw.ellipse((head_u - head_radius, head_v - head_radius,
                      head_u + head_radius, head_v + head_radius), fill=255)
    else:
        draw.rectangle((xa, ya, xb, yb), fill=255)
    skeleton_mask = np.asarray(skeleton, dtype=np.uint8) > 0
    mask = depth_band_mask & skeleton_mask
    return mask, {
        "method": "14-keypoint skeleton capsules/torso intersected with adaptive joint-depth band",
        "joint_depth_median_m": centre,
        "joint_depth_mad_m": mad,
        "depth_tolerance_m": tolerance,
        "skeleton_support_pixels": int(skeleton_mask.sum()),
        "native_mask_pixels": int(mask.sum()),
    }



def unproject(depth: np.ndarray, K: np.ndarray, stride: int) -> np.ndarray:
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[vv, uu]
    good = np.isfinite(d) & (d > 0.05)
    u, v, z = uu[good].astype(float), vv[good].astype(float), d[good].astype(float)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    # Camera optical (+X right, +Y down, +Z forward) -> rig (+X right, +Y forward, +Z up).
    return np.column_stack((x, z, -y))



def voxel_downsample_torch(points: "torch.Tensor", voxel_m: float) -> "torch.Tensor":
    """Open3D's `voxel_down_sample` on the device: the centroid of each occupied voxel.

    Voxels are indexed from the cloud's minimum corner less half a voxel,
    as Open3D does, so the same voxels are occupied and the centroids agree
    to float64 rounding; the order of the output is by voxel key.
    """
    import torch
    origin = points.amin(0) - voxel_m / 2.0
    index = torch.floor((points - origin) / voxel_m).long()
    extent = index.amax(0) + 1
    key = (index[:, 0] * extent[1] + index[:, 1]) * extent[2] + index[:, 2]
    unique, inverse = torch.unique(key, return_inverse=True)
    total = torch.zeros((len(unique), 3), dtype=points.dtype, device=points.device).index_add_(0, inverse, points)
    count = torch.zeros(len(unique), dtype=points.dtype, device=points.device).index_add_(
        0, inverse, torch.ones(len(points), dtype=points.dtype, device=points.device))
    return total / count[:, None]


def pca_normals_torch(points: "torch.Tensor", radius_m: float, max_nn: int | None,
                      camera: "torch.Tensor", chunk: int = 1024) -> "torch.Tensor":
    """Open3D's `estimate_normals` on the device, oriented toward `camera`.

    Per point, the neighbours within `radius_m` (the `max_nn` nearest of
    them when given -- Open3D's hybrid search; all of them otherwise -- its
    radius search), the eigenvector of their covariance's smallest
    eigenvalue, flipped to face the camera. The all-neighbours form takes
    the covariance by masked matmul over the chunk's distance matrix, so a
    dense cloud's thousands of neighbours cost one product; the k-nearest
    form gathers them. Float32 arithmetic on centred neighbourhoods.
    """
    import torch
    n = len(points)
    # sorted along x, so a chunk of consecutive rows spans a thin slab and
    # only the columns within `radius_m` of it in x can be neighbours: a
    # contiguous run found by two searches, a tenth of the cloud instead of
    # all of it in every distance matrix
    order = torch.argsort(points[:, 0])
    p32 = points[order].float()
    xs = p32[:, 0].contiguous()
    normals = torch.zeros((n, 3), dtype=torch.float32, device=points.device)
    # The distance matrix and its two or three same-sized temporaries are the
    # peak: bounded at `cells` entries by taking fewer rows when the slab is
    # wide (a 485k-point LiDAR scaffold whose slab spans most of the cloud
    # took the whole card, RT-Pose 123/169). Each row's sums run over the
    # same columns whatever the split, so the normals do not depend on it.
    cells = 32 * 1024 * 1024
    start = 0
    while start < n:
        first = int(torch.searchsorted(xs, p32[start, 0] - radius_m))
        end = min(start + chunk, n)
        last = int(torch.searchsorted(xs, p32[end - 1, 0] + radius_m, right=True))
        if (end - start) * (last - first) > cells:
            end = start + max(1, cells // (last - first))
            last = int(torch.searchsorted(xs, p32[end - 1, 0] + radius_m, right=True))
        rows = p32[start:end]
        columns = p32[first:last]
        d2 = torch.cdist(rows, columns).square()                               # (C, M)
        within = d2 <= radius_m * radius_m
        if max_nn is not None:
            k = min(max_nn, len(columns))
            d2 = torch.where(within, d2, torch.full_like(d2, float("inf")))
            nearest = d2.topk(k, dim=1, largest=False)
            neighbours = columns[nearest.indices]                              # (C, k, 3)
            valid = torch.isfinite(nearest.values)[:, :, None]
            count = valid.sum(1).clamp_min(1)
            mean = (neighbours * valid).sum(1) / count
            centred = (neighbours - mean[:, None, :]) * valid
            cov = centred.transpose(1, 2) @ centred / count[:, :, None]
        else:
            weight = within.float()
            count = weight.sum(1, keepdim=True).clamp_min(1.0)
            mean = (weight @ columns) / count
            # E[(p - mu)(p - mu)^T] = E[p p^T] - mu mu^T, both by one matmul
            outer = (columns[:, :, None] * columns[:, None, :]).reshape(len(columns), 9)
            second = (weight @ outer).reshape(-1, 3, 3) / count[:, :, None]
            cov = second - mean[:, :, None] * mean[:, None, :]
        values, vectors = torch.linalg.eigh(cov)                              # ascending eigenvalues
        normal = vectors[:, :, 0]
        toward = camera.float()[None, :] - rows
        flip = (normal * toward).sum(1, keepdim=True) < 0
        normals[start:end] = torch.where(flip, -normal, normal)
        start = end
    unsorted = torch.empty_like(normals)
    unsorted[order] = normals
    normals = unsorted
    return torch.nn.functional.normalize(normals, dim=1)


def estimate_normals(points: np.ndarray, voxel_m: float, camera_m: np.ndarray,
                     device: str = "cuda") -> np.ndarray:
    """Voxelise and give every point a normal facing the camera (the GPU form of Open3D's pair)."""
    import torch
    cloud = voxel_downsample_torch(torch.as_tensor(points, dtype=torch.float64, device=device), voxel_m)
    normals = pca_normals_torch(cloud, max(0.15, voxel_m * 4), 40,
                                torch.as_tensor(camera_m, dtype=torch.float64, device=device))
    return torch.cat([cloud.float(), normals], dim=1).cpu().numpy()



def save_preview_scene_pointcloud(rgb: np.ndarray, depth: np.ndarray, pcl: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("HuPR RGB")
    image = axes[1].imshow(depth, cmap="turbo")
    axes[1].set_title("DA-V2 metric depth [m]")
    fig.colorbar(image, ax=axes[1], fraction=0.046)
    axes[2].scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 2], s=1, cmap="viridis", rasterized=True)
    axes[2].scatter([0], [0], c="red", marker="^", s=60, label="initial radar/camera")
    axes[2].set_title("Scene cloud, top down")
    axes[2].set_xlabel("+X right [m]")
    axes[2].set_ylabel("+Y forward [m]")
    axes[2].axis("equal")
    axes[2].legend()
    for axis in axes[:2]:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)



def build_scene(
    dataset_root: Path,
    official_code: Path,
    model_path: Path,
    output_dir: Path,
    sequence: int,
    frame: int,
    hfov_deg: float,
    stride: int,
    voxel_m: float,
    mask_person: bool,
    depth_npy: Path | None,
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    preview: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = read_rgb_from_zip(dataset_root, sequence, frame)
    if preview:
        Image.fromarray(rgb).save(output_dir / "rgb.png")
    depth_path = output_dir / "depth_metric_m.npy"
    if depth is not None:
        depth_source = str(model_path.resolve())
    elif depth_npy is not None:
        depth = np.load(depth_npy).astype(np.float32)
        depth_source = str(depth_npy.resolve())
    elif depth_path.exists():
        depth = np.load(depth_path).astype(np.float32)
        depth_source = "output cache"
    else:
        depth = infer_metric_depth(rgb, model_path)
        depth_source = str(model_path.resolve())
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth {depth.shape} does not match RGB {rgb.shape[:2]}")

    annotation = frame_annotation(dataset_root, official_code, sequence, frame)
    work_depth = depth
    annotation_bbox = None
    scaled_bbox = None
    scaled_joints = None
    annotation_resolution = [256, 256]
    if annotation is not None:
        annotation_bbox = list(map(float, annotation["bbox"]))
        sx = rgb.shape[1] / annotation_resolution[0]
        sy = rgb.shape[0] / annotation_resolution[1]
        scaled_bbox = [
            annotation_bbox[0] * sx,
            annotation_bbox[1] * sy,
            annotation_bbox[2] * sx,
            annotation_bbox[3] * sy,
        ]
        scaled_joints = [[float(x) * sx, float(y) * sy] for x, y in annotation["joints"]]
    if mask_person:
        if annotation is None:
            raise ValueError(f"no annotation found for sequence {sequence}")
        work_depth = mask_bbox(depth, scaled_bbox)
    K = camera_intrinsics(*rgb.shape[:2], hfov_deg)
    points = unproject(work_depth, K, stride)
    pcl = estimate_normals(points, voxel_m, np.zeros(3))
    pcl_path = output_dir / "pcl.npy"
    np.save(pcl_path, pcl)
    person_proxy = None
    if not mask_person and annotation is not None:
        person_mask, person_proxy = person_depth_mask(
            depth, scaled_bbox, annotation["joints"], (sx, sy)
        )
        person_depth = np.where(person_mask, depth, np.nan).astype(np.float32)
        person_points = unproject(person_depth, K, stride)
        person_pcl = estimate_normals(person_points, min(voxel_m, 0.025), np.zeros(3))
        np.save(output_dir / "person_pcl.npy", person_pcl)
        Image.fromarray((person_mask * 255).astype(np.uint8)).save(output_dir / "person_mask.png")
        person_proxy["point_count"] = int(len(person_pcl))
    if preview:
        save_preview_scene_pointcloud(rgb, work_depth, pcl, output_dir / "scene_preview.png")

    metadata = {
        "dataset": "HuPR",
        "sequence": sequence,
        "frame": frame,
        "rgb_zip_member": f"frames/single_{sequence}/processed/images/{frame:09d}.jpg",
        "depth_source": depth_source,
        "depth_is_metric_meters": True,
        "person_masked": mask_person,
        "annotation_resolution_wh": annotation_resolution,
        "person_bbox_annotation_xyxy": annotation_bbox,
        "person_bbox_native_rgb_xyxy": scaled_bbox,
        "person_joints_native_rgb_xy": scaled_joints,
        "person_proxy": person_proxy,
        "camera_intrinsics": K.tolist(),
        "assumed_horizontal_fov_deg": hfov_deg,
        "camera_to_initial_rig": "co-located and co-boresighted; optical XYZ -> rig X,Z,-Y",
        "stride_px": stride,
        "voxel_m": voxel_m,
        "point_count": int(len(pcl)),
        "pcl_columns": ["x_m", "y_m", "z_m", "normal_x", "normal_y", "normal_z"],
        "calibration_status": "initialization only; HuPR camera intrinsics/extrinsics are unpublished",
    }
    (output_dir / "scene_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata






# ----------------------------------------------------------------------
# scaffold: Build a multi-view HuPR static scaffold and static dual-radar targets.
# ----------------------------------------------------------------------
matplotlib.use("Agg")



DEFAULT_SAMPLES = "265:0,83:0,186:0,10:0,166:0"



def clean_scene_maps(dataset, sequence: int, device: str, stride: int = 4, quantile: float = 0.25,
                     batch: int = 16) -> dict[str, "torch.Tensor"]:
    """Each radar's person-free |RA| map: a low quantile per cell over the sequence's frames.

    The person is the only mover, and occupies any one cell in a minority
    of the frames; the per-cell 25th percentile of the loop-mean |RA| over
    every `stride`-th frame is the static scene with the person gone (the
    wall behind them included, since they stand in front of it only some
    of the time). On the GPU, one pass over the archive (frames the later
    stages reuse from memory).
    """
    import torch
    from mm4DPS.data import MeasuredReader, range_angle_slow_time_cube_frames_torch
    maps = {}
    for radar in dataset.radars:
        reader = MeasuredReader(dataset, sequence, radar)
        count = reader.frames
        stack = []
        for start in range(0, count, batch):
            block = reader.batch(start, min(batch, count - start), device)
            picked = block[(stride - start % stride) % stride::stride]
            if len(picked):
                stack.append(range_angle_slow_time_cube_frames_torch(picked, radar=radar).abs().mean(1))
        maps[radar.name] = torch.quantile(torch.cat(stack).float(), quantile, dim=0)
    return maps


def fit_range_alignment(points: np.ndarray, dataset, sequence: int, device: str,
                        search_m: float = 2.4, coarse_m: float = 0.02, near_m: float = 0.5) -> dict:
    """The monocular scene's range translation into the radar frame, from this recording.

    The camera is placed at the rig, co-boresighted, and Depth-Anything's
    metric depth is off by a per-reconstruction amount; the fit is of that
    one number, y along the boresight. Objective (3DPS's): min-max
    normalised Pearson correlation between the shifted cloud's (range,
    angle) occupancy and the person-free measured map (`clean_scene_maps`),
    mean over both boards, inside the scene's own angular reach (the camera
    sees +/-20 degrees of the radar's +/-75; the cells beyond have no model
    behind them) and beyond the near field. Lateral and height are NOT
    fitted: the map barely constrains them -- competing optima 0.6-1.4 m
    apart within 0.03 of the best score on every sequence -- while the
    range optimum is unique (256: -0.34 m, 38: -0.82, 102: -0.74; r 0.25 ->
    0.47, 0.16 -> 0.48, 0.16 -> 0.50). A 1-D grid at `coarse_m`, then a
    refinement at a tenth of it; every candidate scored at once on the GPU.
    """
    import torch
    from mm4DPS import Install, array_axis, config_centre_m
    configs = {r.name: json.loads((Install.from_env().config_dir / r.config).read_text()) for r in dataset.radars}
    clean = clean_scene_maps(dataset, sequence, device)
    cloud = torch.as_tensor(points, device=device, dtype=torch.float32)
    spacing, n_range, n_angle = dataset.radars[0].range_bin_m, dataset.radars[0].samples, dataset.radars[0].n_angle
    half = n_angle / 2.0

    def cells(shifted, radar):                      # (B, N, 3) -> range bin, angle bin, distance
        centre = torch.as_tensor(config_centre_m(configs[radar.name]), device=device, dtype=torch.float32)
        axis = torch.as_tensor(array_axis(configs[radar.name]), device=device, dtype=torch.float32)
        delta = shifted - centre
        distance = delta.norm(dim=-1).clamp_min(1e-9)
        sine = (delta @ axis) / distance
        angle = (half - half * sine) if radar.name == "horizontal" else (half + half * sine)
        return (distance / spacing).round().long(), angle.round().long(), distance

    masks = {}
    for radar in dataset.radars:
        _, angle, _ = cells(cloud[None], radar)
        support = torch.zeros(n_angle, dtype=torch.bool, device=device)
        support[int(angle.min().clamp(0)):int(angle.max().clamp(max=n_angle - 1)) + 1] = True
        rows = torch.arange(n_range, device=device) >= int(np.ceil(near_m / spacing))
        masks[radar.name] = rows[:, None] & support[None, :]

    def score(translations):                        # (B, 3) -> (B,)
        total = torch.zeros(len(translations), device=device)
        for radar in dataset.radars:
            range_bin, angle_bin, distance = cells(cloud[None] + translations[:, None, :], radar)
            keep = ((distance >= near_m) & (angle_bin >= 0) & (angle_bin < n_angle)
                    & (range_bin >= 0) & (range_bin < n_range))
            flat = (torch.arange(len(translations), device=device)[:, None] * n_range * n_angle
                    + range_bin * n_angle + angle_bin)[keep]
            grid = torch.zeros(len(translations) * n_range * n_angle, device=device)
            grid.index_add_(0, flat, torch.ones(len(flat), device=device))
            x = grid.view(-1, n_range, n_angle)[:, masks[radar.name]]
            y = clean[radar.name][masks[radar.name]][None].expand(len(x), -1)
            x = (x - x.amin(1, keepdim=True)) / (x.amax(1, keepdim=True) - x.amin(1, keepdim=True)).clamp_min(1e-30)
            y = (y - y.amin(1, keepdim=True)) / (y.amax(1, keepdim=True) - y.amin(1, keepdim=True)).clamp_min(1e-30)
            x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
            total += (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-30)
        return total / len(dataset.radars)

    def along_y(values):
        return torch.stack([torch.zeros_like(values), values, torch.zeros_like(values)], 1)

    coarse = torch.arange(-search_m, search_m + 1e-6, coarse_m, device=device)
    scores = score(along_y(coarse))
    best = coarse[int(scores.argmax())]
    fine = best + torch.arange(-coarse_m, coarse_m + 1e-9, coarse_m / 10.0, device=device)
    fine_scores = score(along_y(fine))
    y = float(fine[int(fine_scores.argmax())])
    return {"translation_m": [0.0, round(y, 4), 0.0], "rotation_elev_deg": 0.0, "rotation_azim_deg": 0.0,
            "correlation": round(float(fine_scores.max()), 4),
            "correlation_at_identity": round(float(score(torch.zeros(1, 3, device=device))[0]), 4),
            "estimator": "range translation only (lateral and height 0: the co-located rig); min-max normalised "
                         "Pearson of the cloud's (range, angle) occupancy against the per-cell 25th-percentile "
                         "|RA| over every 4th frame (person-free), mean over both boards, inside the cloud's "
                         "own angle bins and beyond 0.5 m; grid 2 cm over +/-2.4 m, then 2 mm",
            "support_bins": {r.name: [int(masks[r.name].any(0).nonzero().min()), int(masks[r.name].any(0).nonzero().max())]
                             for r in dataset.radars}}


def sampled_range_angle_maps(dataset, samples: list[tuple[int, int]], device: str,
                             batch: int = 16) -> dict[str, list[np.ndarray]]:
    """Each radar's loop-mean |RA| map at the sampled frames, one archive pass per (sequence, radar).

    The GPU form of `range_angle_map` (`range_angle_slow_time_cube_torch`:
    5e-16 of the numpy arbiter relative to the peak).
    """
    import torch
    from mm4DPS.data import MeasuredReader, range_angle_slow_time_cube_torch
    maps = {radar.name: [None] * len(samples) for radar in dataset.radars}
    for sequence in sorted({s for s, _ in samples}):
        wanted = {frame: i for i, (s, frame) in enumerate(samples) if s == sequence}
        last = max(wanted) + 1
        for radar in dataset.radars:
            reader = MeasuredReader(dataset, sequence, radar)
            try:
                for start in range(0, last, batch):
                    block = reader.batch(start, min(batch, last - start), device)
                    for offset in range(len(block)):
                        if start + offset in wanted:
                            cube = range_angle_slow_time_cube_torch(block[offset], radar=radar)
                            maps[radar.name][wanted[start + offset]] = cube.abs().mean(0).float().cpu().numpy()
            finally:
                reader.close()
    return maps


def parse_samples(text: str) -> list[tuple[int, int]]:
    samples = []
    for token in text.split(","):
        sequence, frame = token.strip().split(":", 1)
        samples.append((int(sequence), int(frame)))
    if len(samples) < 2:
        raise ValueError("multi-view scaffold requires at least two sequence:frame samples")
    return samples



def merge_clouds(clouds: list[np.ndarray], voxel_m: float, normal_radius_m: float | None = None,
                 device: str = "cuda") -> np.ndarray:
    """Merge, voxelise, and give every point a normal facing the radar.

    ``normal_radius_m`` switches the PCA neighbourhood from HuPR's hybrid
    search (radius 0.18 m, 50 nearest) to a pure radius search. The Livox
    LiDAR needs it: its scan lines run radially from the sensor about 0.4 m
    apart on the far wall, so any neighbourhood that fits within one line
    gives a normal perpendicular to the line of sight (median |cos| 0.007
    with the hybrid search; 0.84 on the wall at 0.5 m radius). Points with
    such a normal are what cull_gaussians drops, so this decides whether the
    scene renders at all.
    """
    import torch

    points = np.concatenate([cloud[:, :3] for cloud in clouds], axis=0)
    points = points[
        np.isfinite(points).all(axis=1)
        & (np.linalg.norm(points, axis=1) >= 0.5)
        & (np.linalg.norm(points, axis=1) <= 10.5)
    ]
    cloud = voxel_downsample_torch(torch.as_tensor(points, dtype=torch.float64, device=device), voxel_m)
    normals = pca_normals_torch(
        cloud, normal_radius_m if normal_radius_m else max(0.18, voxel_m * 4),
        None if normal_radius_m else 50, torch.zeros(3, dtype=torch.float64, device=device))
    return torch.cat([cloud.float(), normals], dim=1).cpu().numpy()



def save_preview_scaffold(pcl: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 2], s=0.5, cmap="viridis", rasterized=True)
    axes[0].set_xlabel("+X right [m]")
    axes[0].set_ylabel("+Y forward [m]")
    axes[0].set_title("Multi-view HuPR scaffold: top down")
    axes[0].axis("equal")
    axes[1].scatter(pcl[:, 0], pcl[:, 2], c=pcl[:, 1], s=0.5, cmap="plasma", rasterized=True)
    axes[1].set_xlabel("+X right [m]")
    axes[1].set_ylabel("+Z up [m]")
    axes[1].set_title("Front elevation")
    axes[1].axis("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)



def bbox_masked_fraction(bbox_256: list[float], width: int = 512, height: int = 512,
                         padding_frac: float = 0.12) -> float:
    """The fraction of the image `mask_bbox` removes for an annotation bbox (256-space)."""
    x0, y0, x1, y1 = (float(v) * (width / 256) if i % 2 == 0 else float(v) * (height / 256)
                      for i, v in enumerate(bbox_256))
    dx, dy = (x1 - x0) * padding_frac, (y1 - y0) * padding_frac
    xa, xb = max(0, int(x0 - dx)), min(width, int(math.ceil(x1 + dx)))
    ya, yb = max(0, int(y0 - dy)), min(height, int(math.ceil(y1 + dy)))
    return max(0, xb - xa) * max(0, yb - ya) / float(width * height)


def usable_samples(dataset_root: Path, official_code: Path, samples: list[tuple[int, int]],
                   empty_at: float = 0.98, prefer_below: float = 0.9, step: int = 5,
                   ) -> tuple[list[tuple[int, int]], dict[int, int]]:
    """The sampled frames, a frame whose person mask covers the image replaced by its nearest usable one.

    When the person stands close enough to fill the frame (HuPR 52 at frame
    150, 154 at 300: the padded bbox masks 100% of the pixels) the frame has
    no scene to contribute. Such a frame (>= `empty_at` masked) is swapped
    for the nearest annotated frame, searched outward in `step`s, that masks
    less than `prefer_below`. Every other sequence's frames are untouched
    (the most masked sampled frame among the 233 others is 0.953).
    """
    chosen, substituted = [], {}
    for sequence, frame in samples:
        annotation = frame_annotation(dataset_root, official_code, sequence, frame)
        if annotation is None or bbox_masked_fraction(annotation["bbox"]) < empty_at:
            chosen.append((sequence, frame))
            continue
        for offset in range(step, 600, step):
            candidates = [f for f in (frame - offset, frame + offset) if 0 <= f < 600]
            found = next((f for f in candidates
                          if (a := frame_annotation(dataset_root, official_code, sequence, f)) is not None
                          and bbox_masked_fraction(a["bbox"]) < prefer_below), None)
            if found is not None:
                break
        else:
            raise ValueError(f"sequence {sequence}: no frame near {frame} shows the scene around the person")
        substituted[frame] = found
        chosen.append((sequence, found))
    return chosen, substituted


def build_scaffold(
    dataset_root: Path,
    official_code: Path,
    model_path: Path,
    output_dir: Path,
    samples: list[tuple[int, int]],
    hfov_deg: float,
    stride: int,
    source_voxel_m: float,
    merged_voxel_m: float,
    dataset: str,
    scene_alignment: bool = True,
    preview: bool = False,
    device: str = "cuda:0",
) -> dict:
    if dataset != "hupr":
        raise NotImplementedError(f"the monocular scaffold is HuPR's; {dataset!r} has its own")
    source_root = output_dir / "sources"
    clouds, horizontal_maps, vertical_maps, sources = [], [], [], []
    samples, substituted = usable_samples(dataset_root, official_code, samples)
    # the sampled frames' images through Depth-Anything in one forward pass
    rgbs = [read_rgb_from_zip(dataset_root, sequence, frame) for sequence, frame in samples]
    depths = infer_metric_depths(rgbs, model_path)
    # and their ADC frames in one sequential pass per board: the archive
    # member is deflated, so a seek to frame 450 inflates 450 frames (1.4 s
    # a board); one pass to the last wanted frame reads each frame once
    static_maps = sampled_range_angle_maps(Dataset.named(dataset), samples, device)
    for index, (sequence, frame) in enumerate(samples):
        source_dir = source_root / f"single_{sequence}_frame_{frame}"
        metadata = build_scene(
            dataset_root, official_code, model_path, source_dir,
            sequence, frame, hfov_deg, stride, source_voxel_m,
            True, None, rgb=rgbs[index], depth=depths[index], preview=preview,
        )
        cloud = np.load(source_dir / "pcl.npy")
        horizontal, vertical = static_maps["horizontal"][index], static_maps["vertical"][index]
        clouds.append(cloud)
        horizontal_maps.append(horizontal)
        vertical_maps.append(vertical)
        sources.append({
            "sequence": sequence,
            "frame": frame,
            "person_bbox_native_rgb_xyxy": metadata["person_bbox_native_rgb_xyxy"],
            "point_count": int(len(cloud)),
        })

    if not any(len(cloud) for cloud in clouds):
        raise ValueError(f"no scene points in any sampled frame of sequence {samples[0][0]}")
    scaffold = merge_clouds(clouds, merged_voxel_m)
    # The scene alignment is part of building the scaffold, not a later
    # correction. Everything downstream -- the background material fit, the
    # point pool, the renders, the shadow masks -- reads pcl.npy and nothing
    # else, so applying it here is what makes it the default and keeps a
    # half-aligned stack impossible. Fitted here, from this recording
    # (`fit_range_alignment`); `--no-scene-alignment` leaves the scene at the
    # camera's assumed pose.
    alignment = None
    if scene_alignment:
        sequences = {s for s, _ in samples}
        if len(sequences) != 1:
            raise ValueError(
                "the scene alignment is fitted PER SEQUENCE, so a scaffold "
                f"merging {sorted(sequences)} has no single transform. Build "
                "one scaffold per sequence, or pass --no-scene-alignment.")
        sequence = sequences.pop()
        alignment = fit_range_alignment(scaffold[:, :3], Dataset.named(dataset), sequence, device)
        scaffold[:, :3] += np.asarray(alignment["translation_m"], dtype=scaffold.dtype)[None, :]
    horizontal_static = np.median(np.stack(horizontal_maps), axis=0).astype(np.float32)
    vertical_static = np.median(np.stack(vertical_maps), axis=0).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pcl.npy", scaffold)
    np.save(output_dir / "horizontal_static_ra.npy", horizontal_static)
    np.save(output_dir / "vertical_static_ra.npy", vertical_static)
    if preview:
        save_preview_scaffold(scaffold, output_dir / "scaffold_preview.png")
    metadata = {
        "dataset": "HuPR",
        "samples": sources,
        "substituted_frames": {str(k): v for k, v in substituted.items()},   # a frame the person filled -> the one used
        "point_count": int(len(scaffold)),
        "merged_voxel_m": merged_voxel_m,
        "radar_target_reduce": "pixelwise median across synchronized raw-ADC range-angle maps",
        "camera_assumptions": {
            "horizontal_fov_deg": hfov_deg,
            # "co-located, co-boresighted" is where the camera STARTS, a
            # placeholder rather than a measurement. scene_alignment below is
            # the correction to it, and is null only if it was switched off.
            "camera_to_rig_initial": "co-located, co-boresighted",
        },
        "scene_alignment": alignment,
        "scene_alignment_source": (
            None if alignment is None else
            "fit_range_alignment: the range translation of the camera-posed scene that best "
            "correlates its occupancy with this recording's person-free |RA| maps, both boards "
            "jointly, inside the scene's own angular reach; lateral and height 0 (the co-located "
            "rig), rotations 0"),
        "pcl_columns": ["x_m", "y_m", "z_m", "normal_x", "normal_y", "normal_z"],
    }
    (output_dir / "scaffold_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata



def main_scaffold() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--official-code", type=Path, default=DEFAULT_OFFICIAL_CODE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES,
                        help="Comma-separated sequence:frame pairs selected for different person positions.")
    parser.add_argument("--output-dir", type=Path, default=Path(WORKSPACE / "multiview_scaffold"))
    parser.add_argument("--hfov-deg", type=float, default=70.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--source-voxel-m", type=float, default=0.04)
    parser.add_argument("--merged-voxel-m", type=float, default=0.05)
    dataset_flag(parser)
    parser.add_argument("--no-scene-alignment", dest="scene_alignment",
                        action="store_false",
                        help="leave the scene at the camera's assumed pose (no range fit)")
    parser.add_argument("--preview", action="store_true",
                        help="also write the per-frame rgb/scene previews and scaffold_preview.png")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    Dataset.select(args.dataset)
    metadata = build_scaffold(
        args.dataset_root, args.official_code, args.model_path, args.output_dir,
        parse_samples(args.samples), args.hfov_deg, args.stride,
        args.source_voxel_m, args.merged_voxel_m, args.dataset,
        args.scene_alignment, preview=args.preview, device=args.device,
    )
    print(json.dumps(metadata, indent=2))


def _entry_scaffold():
        from mm4DPS import dataset_from_argv
        if dataset_from_argv().name == "octonet":
            from mm4DPS import octonet
            return octonet.main_scaffold()     # RGB-D (3 nodes) + MoCap, no Depth-Anything: mm4DPS/octonet.py
        main_scaffold()

# ----------------------------------------------------------------------
# person_sequence: Precompute a person-inclusive HuPR sequence and first-loop radar targets.
# ----------------------------------------------------------------------



ANNOTATION_SIZE = (256, 256)















def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--official-code", type=Path, default=DEFAULT_OFFICIAL_CODE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mask-erosion-px", type=int, default=1,
                        help="erode the SAM2 mask before unprojecting, to drop "
                             "the silhouette pixels that carry background depth")
    parser.add_argument(
        "--sam2-masks", type=Path,
        help="directory from mm4DPS.preprocessing.sam2_person_masks; when given it "
             "replaces the depth-band mask and the joints are lifted in-mask",
    )
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--loop-index", type=int, default=0)
    parser.add_argument("--hfov-deg", type=float, default=70.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--person-voxel-m", type=float, default=0.025)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path,
        default=WORKSPACE / "person_sequence_1_first_loop",
    )
    return parser



# ----------------------------------------------------------------------
# masked_joint_depth: Lift joints using only pixels the segmentation says belong to the person.
# ----------------------------------------------------------------------


DEFAULT_MINIMUM_SAMPLES = 5

DEFAULT_MAXIMUM_RADIUS = 12









# ----------------------------------------------------------------------
# RT-Pose: the scene from the LiDAR, aligned to the radar; the person from GVHMR
#
# HuPR had no depth sensor, so its scene came from monocular depth on four
# frames and the alignment was a hard-won per-sequence table. RT-Pose ships a
# LiDAR and no radar extrinsic, so the scene is metric from the start and the
# alignment IS the missing extrinsic: a translation search that maximises the
# correlation between the LiDAR occupancy (in range-azimuth) and the measured
# static range-azimuth map, the same objective HuPR's step 1 used.
# ----------------------------------------------------------------------

# LiDAR points this close to the ANNOTATED person's column were cut as the person. The annotation is 0.45-0.6 m off the
# registered mesh on 108 / 168 / 164 (docs/STATIC_SEQS_NEXT.md 1c), so the column cut scene beside the person and left the
# person; since c0f9241 the registered mesh cuts the person out in person_stack (scaffold-cut), and the column is off by
# default (0). MM4DPS_PERSON_CLEAR_RADIUS_M=0.6 restores it for an A/B.
PERSON_CLEAR_RADIUS_M = float(os.environ.get("MM4DPS_PERSON_CLEAR_RADIUS_M", "0"))


def rtpose_extrinsic_seed(repo: Path) -> dict:
    """seeds/rtpose/radar_extrinsic.json, or the identity if it is absent."""
    path = repo / "seeds" / "rtpose" / "radar_extrinsic.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"rotation": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0], "status": "identity"}


def lidar_occupancy_ra(points_world: np.ndarray, radar, translation: np.ndarray) -> np.ndarray:
    """LiDAR points -> (range, azimuth) occupancy counts under a candidate translation.

    Azimuth bin ``N/2 - N/2 sin(theta_X)``, the convention the measured maps use
    (:func:`mm4DPS.qa.person_angle_bins`); range bins of the radar's spacing;
    the near field and the elevation outside +/-30 degrees dropped.
    """
    delta = points_world + translation[None, :]
    distance = np.linalg.norm(delta, axis=1)
    keep = distance > 0.5
    delta, distance = delta[keep], distance[keep]
    sine = delta[:, 0] / distance
    elevation = delta[:, 2] / distance
    n = radar.n_angle
    angle_bin = np.round(n / 2 - n / 2 * sine).astype(int)
    range_bin = np.round(distance / radar.range_bin_m).astype(int)
    ok = ((angle_bin >= 0) & (angle_bin < n) & (range_bin >= 0) & (range_bin < radar.samples)
          & (np.abs(elevation) < 0.5))
    grid = np.zeros((radar.samples, n), dtype=np.float64)
    np.add.at(grid, (range_bin[ok], angle_bin[ok]), 1.0)
    return grid


def normalised_correlation(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> float:
    """3DPS's objective: min-max normalise both maps, then Pearson over ``keep``."""
    def unit(v):
        low, high = v.min(), v.max()
        return (v - low) / max(high - low, 1e-30)
    x, y = unit(a), unit(b)
    if keep is not None:
        x, y = x[keep], y[keep]
    x, y = x.reshape(-1) - x.mean(), y.reshape(-1) - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-30 else 0.0


def fit_lidar_alignment(points_world: np.ndarray, measured_ra: np.ndarray, radar,
                        lateral_m: float = 0.5, forward_m: float = 0.3, height_m: float = 0.2,
                        step_m: float = 0.05) -> dict:
    """Grid search of the LiDAR->radar translation against the measured static map.

    Objective: normalised correlation between the LiDAR occupancy map (range x
    azimuth, log1p of the counts so a dense wall does not swamp the rest) and
    the measured static range-azimuth magnitude (linear, never dB) over range
    bins 12..200. Coarse grid first, then a 1 cm refinement around the optimum.
    Estimator named in the result.
    """
    rows = slice(12, 200)
    target = measured_ra[rows]
    def score(t):
        return normalised_correlation(np.log1p(lidar_occupancy_ra(points_world, radar, t))[rows], target)
    best, best_t = -2.0, np.zeros(3)
    grid = [np.arange(-lateral_m, lateral_m + 1e-9, step_m), np.arange(-forward_m, forward_m + 1e-9, step_m),
            np.arange(-height_m, height_m + 1e-9, step_m)]
    for x in grid[0]:
        for y in grid[1]:
            for z in grid[2]:
                value = score(np.array([x, y, z]))
                if value > best:
                    best, best_t = value, np.array([x, y, z])
    coarse = best_t.copy()
    fine = np.arange(-0.04, 0.041, 0.01)
    for dx in fine:
        for dy in fine:
            for dz in fine:
                t = coarse + np.array([dx, dy, dz])
                value = score(t)
                if value > best:
                    best, best_t = value, t
    return {"translation_m": [round(float(v), 3) for v in best_t], "correlation": best,
            "correlation_at_zero": score(np.zeros(3)),
            "estimator": "normalised correlation (min-max, Pearson) of log1p LiDAR occupancy vs measured "
                         "static |RA|, range bins 12-200, all azimuth bins; grid 5 cm then 1 cm",
            "search_m": {"lateral": lateral_m, "forward": forward_m, "height": height_m}}


def rgb_depth_cloud(seq, entry: dict, lidar_points: np.ndarray, model_path: Path, stride: int = 3,
                    max_depth_m: float = 11.0, depth: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """A dense scene cloud from the left image's metric depth, scaled to the LiDAR.

    Depth-Anything's metric depth is dense but its scale is not to be
    trusted; the LiDAR is sparse but metric. So: the LiDAR points of the
    same frame are projected into the image, the ratio LiDAR depth /
    predicted depth is taken at their pixels, and its median rescales the
    whole depth map (the IQR is recorded -- a wide one means the two
    disagree in shape, not just scale). The result is in the LiDAR frame,
    with the person's column still in it; the caller cuts that out.
    """
    from PIL import Image
    image = np.asarray(Image.open(seq.image_path(entry["lidar"])).convert("RGB"))
    height, width = image.shape[:2]
    if depth is None:
        depth = infer_metric_depth(image, model_path)                      # (H, W) metres
    K = seq.camera_matrix(image_size=(width, height))
    pixels = seq.project(lidar_points, image_size=(width, height))
    u, v, z = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    inside = (z > 0.3) & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    predicted = depth[np.round(v[inside]).astype(int), np.round(u[inside]).astype(int)]
    ratio = z[inside] / np.clip(predicted, 1e-3, None)
    good = np.isfinite(ratio) & (ratio > 0.2) & (ratio < 5.0)
    ratio, uu, vv = ratio[good], u[inside][good], v[inside][good]
    # The correction is a FIELD over the image, not a scalar: on this room
    # Depth-Anything's back wall runs 6.2 -> 7.2 m across the columns where
    # the LiDAR reads a flat 6.1 m, and its near floor is 34% too far. The
    # LiDAR points sample the ratio LiDAR / predicted at ~6.8k pixels; they
    # are binned on a coarse grid (cell medians), empty cells take the
    # nearest filled cell's value, and the grid is interpolated bilinearly
    # to every pixel. Where the LiDAR sees, the depth is pinned to it; where
    # it does not (the top and bottom rows beyond its +/-12.5 degree band),
    # the nearest column's correction carries over.
    cells_u, cells_v = 12, 9
    cu = np.clip((uu / width * cells_u).astype(int), 0, cells_u - 1)
    cv = np.clip((vv / height * cells_v).astype(int), 0, cells_v - 1)
    grid = np.full((cells_v, cells_u), np.nan)
    counts = np.zeros((cells_v, cells_u), dtype=int)
    for j in range(cells_v):
        for i in range(cells_u):
            sel = (cu == i) & (cv == j)
            counts[j, i] = int(sel.sum())
            if counts[j, i] >= 15:
                grid[j, i] = np.median(ratio[sel])
    filled = np.argwhere(np.isfinite(grid))
    if len(filled) == 0:
        raise ValueError("no LiDAR points fall in the image; cannot scale the depth")
    for j, i in np.argwhere(~np.isfinite(grid)):
        nearest = filled[np.argmin(((filled - [j, i]) ** 2).sum(axis=1))]
        grid[j, i] = grid[nearest[0], nearest[1]]
    from scipy.ndimage import map_coordinates
    vv_all, uu_all = np.mgrid[0:height, 0:width]
    scale_map = map_coordinates(
        grid, [(vv_all + 0.5) / height * cells_v - 0.5, (uu_all + 0.5) / width * cells_u - 0.5],
        order=1, mode="nearest")
    sampled = scale_map[np.round(vv).astype(int), np.round(uu).astype(int)]
    residual = ratio / sampled - 1.0
    keep = np.ones(len(ratio), dtype=bool)
    coefficients = np.array([float(np.nanmedian(grid)), float(np.ptp(grid)), float(np.isfinite(grid).mean())])
    scaled = depth * scale_map
    scale = float(np.nanmedian(grid))
    scaled[~np.isfinite(scaled) | (scaled > max_depth_m)] = np.nan
    cloud_camera = unproject(scaled, K, stride)                                # rig axes: x right, y fwd, z up
    cloud_camera = np.column_stack((cloud_camera[:, 0], -cloud_camera[:, 2], cloud_camera[:, 1]))  # back to camera optical
    T = seq.camera("left")["extrinsic"]
    cloud_lidar = camera_to_lidar(cloud_camera, T)
    return cloud_lidar, {"depth_scale_median": scale, "scale_field_range": float(coefficients[1]),
                         "lidar_points_used": int(keep.sum()),
                         "residual_iqr_relative": [float(np.percentile(residual, 25)), float(np.percentile(residual, 75))],
                         "estimator": "LiDAR depth / predicted depth, medians on a 12 x 9 image grid (>= 15 points a "
                                      "cell, empty cells from the nearest filled one), bilinear to every pixel"}


def run_lidar_scaffold(args: argparse.Namespace) -> dict:
    """The RT-Pose scene scaffold: LiDAR frames merged, the person cut out, aligned to the radar.

    Writes what `scaffold` writes for HuPR -- pcl.npy (x, y, z, normals),
    <radar>_static_ra.npy and scaffold_metadata.json -- so every stage after
    it reads the same files.
    """
    import torch
    from mm4DPS import RTPOSE, Install
    from mm4DPS.data import (RTPoseSequence, apply_cascade_calibration, cascade_calibration,
                             decode_tda2_frame_torch, range_angle_slow_time_cube_torch)
    dataset = Dataset.select("rtpose")
    seq = RTPoseSequence(dataset, args.sequence)
    radar = seq.radar
    annotation = seq.annotation()
    annotated = [f for f in sorted(annotation) if annotation[f]["lidar_valid"]] or sorted(annotation)
    if not annotated:
        raise ValueError(f"sequence {args.sequence} has no annotated frames to sample")
    wanted = [int(f) for f in args.frames.split(",")] if args.frames else None
    if wanted is None:
        picks = [annotated[int(round(i))] for i in np.linspace(0, len(annotated) - 1, args.sample_count)]
    else:
        picks = [min(annotated, key=lambda f: abs(f - w)) for w in wanted]
    seed = rtpose_extrinsic_seed(args.repo)
    recording = seq.recording()
    calibration = cascade_calibration(radar, Install.from_env().config_dir / radar.calibration)

    clouds, maps, sources = [], [], []
    from mm4DPS.data import RTPoseSequence as _RTPose
    depths = [None] * len(picks)
    if args.rgb_fusion:
        # the sampled frames' images through Depth-Anything in one forward pass
        from PIL import Image
        images = [np.asarray(Image.open(seq.image_path(annotation[f]["lidar"])).convert("RGB")) for f in picks]
        depths = infer_metric_depths(images, args.model_path)
    # The person cut out of each sampled frame BY THE REGISTERED MESH OF THAT FRAME (--person-sequence-dir): the frame's
    # LiDAR and depth points within `person_cut_m` of the person's surface at that radar frame go, everything else stays --
    # the chair the person sits on in frame 47 is scene in the six frames they are elsewhere, so it survives, where a sweep
    # of the whole sequence's poses over the merged cloud (scaffold-cut) removed it (RT-Pose 164, docs/STATIC_SEQS_NEXT.md
    # 1e). The annotation column (PERSON_CLEAR_RADIUS_M) is the fallback when no registered person exists yet.
    person_frames_dir = args.person_sequence_dir / "person_frames" if args.person_sequence_dir else None
    per_frame_cut = 0
    def cut_person(points_world: np.ndarray, frame: int) -> np.ndarray:
        nonlocal per_frame_cut
        if person_frames_dir is None:
            return points_world
        path = person_frames_dir / f"frame_{min(frame, 10 ** 6):03d}.npz"
        if not path.is_file():
            return points_world
        from scipy.spatial import cKDTree
        mesh = np.load(path)["pcl"][:, :3].astype(np.float64); floor = float(np.percentile(mesh[:, 2], 2))
        distance, _ = cKDTree(mesh).query(points_world[:, :3].astype(np.float64), distance_upper_bound=args.person_cut_m)
        hit = (distance < args.person_cut_m) & (points_world[:, 2] > floor + 0.08)
        per_frame_cut += int(hit.sum())
        return points_world[~hit]
    for index, frame in enumerate(picks):
        entry = annotation[frame]
        lidar = seq.lidar(entry["lidar"])
        joints = entry["joints"] + _RTPose.ANNOTATION_TO_LIDAR
        column = np.hypot(lidar[:, 0] - joints[:, 0].mean(), lidar[:, 1] - joints[:, 1].mean())
        scene = lidar[column > PERSON_CLEAR_RADIUS_M]
        clouds.append(cut_person(seq.to_world(scene, seed), frame))
        fusion = None
        if args.rgb_fusion:
            # RGB + LiDAR: the LiDAR's scan lines leave the floor, walls and
            # ceiling sparse (15k points at 2 cm from eight frames); the left
            # image's metric depth, rescaled to this frame's LiDAR, fills them.
            dense, fusion = rgb_depth_cloud(seq, entry, lidar, args.model_path, stride=args.rgb_stride,
                                            depth=depths[index])
            column = np.hypot(dense[:, 0] - joints[:, 0].mean(), dense[:, 1] - joints[:, 1].mean())
            clouds.append(cut_person(seq.to_world(dense[column > PERSON_CLEAR_RADIUS_M], seed), frame))
        # The static map's frame on the GPU: the torch beamformer is 1.8e-7
        # of the numpy one in complex64 (its docstring) and 20x faster; the
        # numpy form cost 2 s of the scaffold's 11 in `take`.
        words = torch.from_numpy(recording.frame_words(frame)).to(args.device)
        adc = apply_cascade_calibration(decode_tda2_frame_torch(words, radar), calibration.to(args.device), radar)
        maps.append(range_angle_slow_time_cube_torch(adc, radar=radar).abs().mean(0).float().cpu().numpy())
        sources.append({"radar_frame": int(frame), "lidar": entry["lidar"], "lidar_points": int(len(lidar)),
                        "scene_points": int(len(scene)), "rgb_depth": fusion})
    scaffold = merge_clouds([np.concatenate([c, np.zeros_like(c)], axis=1) for c in clouds], args.merged_voxel_m,
                            normal_radius_m=args.normal_radius_m)
    static = np.median(np.stack(maps), axis=0).astype(np.float32)

    # The translation search is OFF by default. On seq 183 it moves the
    # correlation from 0.189 to 0.235 while walking the height to the edge of
    # its range: the measured map's azimuth structure is the sidelobe smear of
    # a few specular walls, which is not enough to identify a lateral shift
    # (HuPR found the same). The extrinsic seed is what the person tracks and
    # the LiDAR person support (seeds/rtpose/radar_extrinsic.json), and it is
    # applied as is; the correlation at the seed is recorded for the record.
    if args.fit_alignment:
        fit = fit_lidar_alignment(scaffold[:, :3].astype(np.float64), static.astype(np.float64), radar,
                                  lateral_m=args.lateral_search_m)
        scaffold[:, :3] += np.asarray(fit["translation_m"], dtype=np.float32)[None, :]
    else:
        rows = slice(12, 200)
        fit = {"translation_m": [0.0, 0.0, 0.0], "fitted": False,
               "correlation_at_seed": normalised_correlation(
                   np.log1p(lidar_occupancy_ra(scaffold[:, :3].astype(np.float64), radar, np.zeros(3)))[rows],
                   static.astype(np.float64)[rows]),
               "estimator": "normalised correlation (min-max, Pearson) of log1p LiDAR occupancy vs "
                            "measured static |RA|, range bins 12-200"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "pcl.npy", scaffold)
    # HuPR's global_pointset step wrote the scene the visibility reads under
    # this name; the LiDAR scaffold is that scene, so it is written here too.
    np.save(args.output_dir / "global_scene_points.npy", scaffold)
    np.save(args.output_dir / f"{radar.name}_static_ra.npy", static)
    if args.preview:
        save_preview_scaffold(scaffold, args.output_dir / "scaffold_preview.png")   # 2 s of matplotlib
    metadata = {
        "dataset": "rtpose", "sequence": int(args.sequence), "samples": sources,
        "person_cut": ({"rule": "the registered mesh of each sampled frame", "radius_m": args.person_cut_m, "points_removed": per_frame_cut,
                        "person_sequence_dir": str(args.person_sequence_dir)} if person_frames_dir is not None else None),
        "point_count": int(len(scaffold)), "merged_voxel_m": args.merged_voxel_m,
        "radar_target_reduce": "pixelwise median across the sampled frames' calibrated range-angle maps",
        "scene_source": ("Livox LiDAR" + (" + the left image's Depth-Anything metric depth rescaled to the "
                                          "LiDAR frame by frame" if args.rgb_fusion else "")
                         + f", the annotated person's column cut out (radius {PERSON_CLEAR_RADIUS_M} m), "
                         f"normals by PCA over a {args.normal_radius_m} m radius oriented toward the radar"),
        "extrinsic_seed": seed,
        "scene_alignment": fit,
        "scene_alignment_source": "LiDAR occupancy vs measured static |RA| translation search, PER SEQUENCE; "
                                  "rotations exactly 0; applied on top of the seed",
        "pcl_columns": ["x_m", "y_m", "z_m", "normal_x", "normal_y", "normal_z"],
    }
    (args.output_dir / "scaffold_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def make_parser_lidar_scaffold() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_lidar_scaffold.__doc__)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--frames", default=None, help="radar frames to sample (nearest annotated); default: "
                                                       "--sample-count spread over the annotated frames")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-voxel-m", type=float, default=0.03,
                        help="2 cm: eight Livox frames merge to ~25k points, the density HuPR's "
                             "depth scaffold gave the background model (its target_n is 20k)")
    parser.add_argument("--normal-radius-m", type=float, default=0.3,
                        help="PCA neighbourhood for the normals; the Livox alone needs 0.5 (merge_clouds), "
                             "the RGB-fused cloud is dense enough for 0.3")
    parser.add_argument("--person-sequence-dir", type=Path, default=None,
                        help="the registered person sequence (person_frames/frame_%03d.npz): each sampled frame's points "
                             "within --person-cut-m of that frame's mesh are the person and are cut; the chain's scaffold-cut "
                             "sweep is then skipped (scaffold_metadata.person_cut)")
    parser.add_argument("--person-cut-m", type=float, default=0.25)
    parser.add_argument("--no-rgb-fusion", dest="rgb_fusion", action="store_false",
                        help="LiDAR only: no Depth-Anything cloud from the left image")
    parser.add_argument("--rgb-stride", type=int, default=3, help="pixel stride of the depth unprojection")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="the Depth-Anything snapshot")
    parser.add_argument("--lateral-search-m", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
                        help="where the static map's frames are beamformed")
    parser.add_argument("--fit-alignment", action="store_true",
                        help="search the LiDAR->radar translation against the measured static map "
                             "(off: the seed extrinsic is applied as is; see run_lidar_scaffold)")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--dataset", default="rtpose", choices=("rtpose",))
    parser.add_argument("--preview", action="store_true", help="also write scaffold_preview.png")
    return parser


def _entry_lidar_scaffold():
    print(json.dumps(run_lidar_scaffold(make_parser_lidar_scaffold().parse_args()), indent=2))


def camera_to_lidar(points_camera: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Left-camera-frame points -> LiDAR frame, through the shipped LiDAR->camera 4x4."""
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    return (points_camera - translation) @ rotation


def radar_person_ranges(frames_adc, radar, min_bin: int = 10, max_bin: int = 200,
                        window: int = 5, sector=None) -> tuple["torch.Tensor", "torch.Tensor"]:
    """The radar's range to the moving person in each frame, and how sure it is.

    The clutter-removed range-azimuth cube (the slow-time mean removed, so
    the static room is gone) averaged over loops, summed over angle: a range
    profile of moving energy. Its peak bin, refined by a parabola over its
    two neighbours, is the person's range (`range_bin_m` per bin); the
    confidence is the fraction of the profile's energy within `window` bins
    of the peak -- near one when one mover dominates, low when the person
    is still (noise) or the multipath trail behind them spreads the energy.
    Bins below `min_bin` (the coupling column) and above `max_bin` are out.
    A block of frames at once, on the device: ``(frames,)`` ranges in
    metres and confidences.
    """
    import torch
    from mm4DPS.data import range_angle_slow_time_cube_frames_torch
    ra = range_angle_slow_time_cube_frames_torch(frames_adc, radar=radar, remove_static_clutter=True).abs().mean(1)
    power = ra.square()                                                 # (frames, range, angle)
    if sector is not None:
        # docs/FIDELITY_FIX_PLAN.md (F6, 2026-09-16): the search confined to the camera's angular sector
        # (the person's direction from GVHMR, which pixels give reliably) and to the scale-bound
        # range window about GVHMR's range; the radar still decides the range inside it. Without
        # this, another mover or rain at a different range wins the argmax (RT-Pose 108: the
        # person registered at 4.1 m, the labels at 5.6 m).
        angle_mask, lo, hi = sector                                     # (frames, angle) bool, (frames,) bins
        bins_r = torch.arange(power.shape[1], device=power.device)
        inside = (bins_r[None, :] >= lo[:, None]) & (bins_r[None, :] <= hi[:, None])
        power = power * inside.to(power.dtype)[:, :, None]
        # the azimuth is measured over every angle in the range window: the sector is
        # the camera's own azimuth, the very thing the azimuth registration checks
        power_all_angles = power
        power = power * angle_mask.to(power.dtype)[:, None, :]
    else:
        power_all_angles = power
    profile = power.sum(dim=2)                                          # (frames, range) moving power
    profile[:, :min_bin] = 0.0
    profile[:, max_bin:] = 0.0
    peak = profile.argmax(dim=1)
    bins = torch.arange(profile.shape[1], device=profile.device)
    near = (bins[None, :] - peak[:, None]).abs() <= window
    confidence = (profile * near).sum(1) / profile.sum(1).clamp_min(1e-30)
    rows = torch.arange(len(profile), device=profile.device)
    y0 = profile[rows, (peak - 1).clamp_min(0)]
    y1 = profile[rows, peak]
    y2 = profile[rows, (peak + 1).clamp_max(profile.shape[1] - 1)]
    denominator = y0 - 2.0 * y1 + y2
    refined = peak.double() + torch.where(denominator < 0, 0.5 * (y0 - y2).double() / denominator.double(),
                                          torch.zeros_like(denominator, dtype=torch.float64))
    # The azimuth of the same return: the angle profile of the moving power over the
    # `window` range rows about the peak, its peak bin refined the same way, as the
    # beamformer's spatial sine (bin = n/2 + n/2 sin; person_range_angle_mask_torch).
    # The azimuth of the same return: the angle profile of the moving power over the
    # `window` range rows about the peak, the floor (its median over angles) removed, and
    # the energy centroid over +-0.19 in sine about its peak -- not the peak itself: the
    # recording's return carries a lateral pedestal (multipath, the array's sidelobes) that
    # the peak ignores and the downstream model reads (3.11: the peak over-registers by
    # 1-1.5 degrees). The sine is the beamformer's (bin = n/2 + n/2 sin).
    angle_profile = (power_all_angles * near.to(power.dtype)[:, :, None]).sum(dim=1)     # (frames, angle)
    n_angle = angle_profile.shape[1]
    angle_profile = (angle_profile - angle_profile.median(dim=1, keepdim=True).values).clamp_min(0.0)
    apeak = angle_profile.argmax(dim=1)
    half = max(1, int(round(0.19 * n_angle / 2.0)))
    bins_a = torch.arange(n_angle, device=profile.device)
    in_window = ((bins_a[None, :] - apeak[:, None]).abs() <= half).to(angle_profile.dtype)
    weighted = angle_profile * in_window
    centroid = (weighted * bins_a[None, :].to(weighted.dtype)).sum(1) / weighted.sum(1).clamp_min(1e-30)
    sine = ((centroid.double() - n_angle / 2.0) / (n_angle / 2.0)).clamp(-1.0, 1.0)
    return refined * radar.range_bin_m, confidence.double(), sine


def radar_depth_scales(radar_range: np.ndarray, confidence: np.ndarray, gvhmr_range: np.ndarray,
                       frames: range, min_confidence: float = 0.35, smooth_frames: int = 7,
                       polish_frames: int = 51, scale_bounds: tuple[float, float] = (0.5, 2.0)) -> dict:
    """GVHMR's depth against the radar's, frame by frame.

    GVHMR places the body from its size prior and the focal length it
    assumes -- ~14% short on RT-Pose, ~30% long on HuPR -- and the radar
    measures the person's range directly. One scale for the sequence (the
    median ratio over annotated frames, the first form of this) left seq
    153's person 9-16 range bins (30-55 cm) from the radar's for 41% of its
    frames and 40 bins off where GVHMR lost the track, and the frame-0 range
    sweep then calibrated on one of the wrong frames and moved the whole
    sequence by -0.55 m: the measured micro-Doppler was read at the wrong
    range (docs/FIDELITY_PLAN.md 1.1). So every frame the person moves in
    gets its own ratio r_f = radar range / GVHMR centroid range:

    * frames whose detection confidence (`radar_person_range`) is below
      `min_confidence` -- the person still, or no person -- are interpolated
      from their neighbours (held at the ends);
    * the series is median-filtered over +/-`smooth_frames` frames and
      clamped to `scale_bounds`;
    * a frame whose median-filtered ratio sits outside the confident frames'
      IQR widened three times is flagged (`flagged_frames`): GVHMR's track
      is wrong there, and the radar's range is trusted over it as it is;
    * every unflagged run of frames is then smoothed once more with a
      Savitzky-Golay filter (`polish_frames` wide, quadratic): the true ratio
      drifts slowly with the person's distance, and what a median filter
      leaves is steps of the detection's jitter, which the person cloud
      would inherit as spurious radial motion (0.2-0.4 m/s at the 90th
      percentile against 0.1 with the polish).

    Takes the three per-frame series (`person_range_series` measures the
    first two) and returns the per-frame scales, the flags, and the median
    as the summary.
    """
    count = len(frames)
    raw = np.where(np.isfinite(radar_range) & np.isfinite(gvhmr_range),
                   radar_range / np.maximum(gvhmr_range, 1e-3), np.nan)
    confident = np.isfinite(raw) & (confidence >= min_confidence)
    if confident.sum() < 3:
        raise RuntimeError(f"radar_depth_scales: only {int(confident.sum())} frames with a confident "
                           f"detection (threshold {min_confidence}); the person is not moving")
    index = np.arange(count)
    series = np.interp(index, index[confident], raw[confident])           # held at the ends
    q25, q75 = np.percentile(raw[confident], [25, 75])
    smoothed = np.array([np.median(series[max(i - smooth_frames, 0):i + smooth_frames + 1]) for i in index])
    smoothed = np.clip(smoothed, *scale_bounds)
    band = (q25 - 3.0 * (q75 - q25), q75 + 3.0 * (q75 - q25))
    is_flagged = ~((smoothed >= band[0]) & (smoothed <= band[1]))
    flagged = [int(frames[i]) for i in index if is_flagged[i]]
    polished = smoothed.copy()
    if polish_frames >= 5:
        from scipy.signal import savgol_filter
        start = 0
        for stop in list(np.flatnonzero(np.diff(is_flagged.astype(int)) != 0) + 1) + [count]:
            if not is_flagged[start] and stop - start >= 5:
                window = min(polish_frames | 1, stop - start)
                window -= 1 - window % 2                       # odd, and no longer than the run
                polished[start:stop] = savgol_filter(smoothed[start:stop], window, 2)
            start = stop
        polished = np.clip(polished, *scale_bounds)
    return {
        "depth_scale": float(np.median(raw[confident])),
        "depth_scale_per_frame": [float(v) for v in polished],
        "depth_scale_per_frame_median_filtered": [float(v) for v in smoothed],
        "radar_range_m_per_frame": [None if not np.isfinite(v) else float(v) for v in radar_range],
        "gvhmr_range_m_per_frame": [None if not np.isfinite(v) else float(v) for v in gvhmr_range],
        "detection_confidence_per_frame": [float(v) for v in confidence],
        "confident_frames": int(confident.sum()), "frames": int(count),
        "ratio_iqr": [float(q25), float(q75)],
        "flagged_frames": flagged,
        "min_confidence": min_confidence, "smooth_frames": smooth_frames, "polish_frames": polish_frames,
        "scale_bounds": list(scale_bounds),
        "estimator": "per frame: radar moving-power range-profile peak (parabolic) / GVHMR centroid range; "
                     "frames below the detection confidence interpolated; median over +/-smooth_frames; "
                     "unflagged runs Savitzky-Golay polished (polish_frames, quadratic); "
                     "the summary depth_scale is the median over confident frames",
    }


def radar_azimuth_yaw(radar_sine: np.ndarray, confidence: np.ndarray, mesh_sine: np.ndarray, frames: range,
                      min_confidence: float = 0.35, smooth_frames: int = 7, polish_frames: int = 51,
                      yaw_bound_rad: float = 0.35) -> dict:
    """The camera's azimuth against the radar's, frame by frame: the same registration as the
    depth's (`radar_depth_scales`), for the other coordinate the radar measures directly.

    The depth registration puts the person at the radar's range; its azimuth stayed the
    camera's -- the camera -> radar extrinsic's -- and nothing checked it, because the mesh
    and the dataset's labels share that extrinsic. Measured (docs/FIDELITY_FIX_RESULTS.md
    3.11), the moving return the radar sees sits 3-5 degrees from the mesh on RT-Pose 183 /
    186 (0.2-0.3 m at the person's range), a near-constant yaw per sequence. The downstream
    model trained on the recording learns the label's offset from the return it sees; a
    render that puts the return at the label's azimuth is then misread by that offset.

    Per frame: yaw_f = asin(radar sine) - asin(mesh sine), the radar's moving-power peak
    azimuth against the scaled mesh's centroid azimuth, in the beamformer's own convention;
    frames without a confident detection interpolated, median over +/-smooth_frames, the
    IQR flag, the Savitzky-Golay polish of the unflagged runs -- `radar_depth_scales`'
    schedule exactly. Applied as a rotation about the radar's vertical axis through its
    centre (the pixels are the camera's; the extrinsic's yaw is what moves).
    """
    count = len(frames)
    raw = np.where(np.isfinite(radar_sine) & np.isfinite(mesh_sine),
                   np.arcsin(np.clip(radar_sine, -1, 1)) - np.arcsin(np.clip(mesh_sine, -1, 1)), np.nan)
    confident = np.isfinite(raw) & (confidence >= min_confidence)
    if confident.sum() < 3:
        raise RuntimeError(f"radar_azimuth_yaw: only {int(confident.sum())} frames with a confident detection")
    index = np.arange(count)
    series = np.interp(index, index[confident], raw[confident])
    q25, q75 = np.percentile(raw[confident], [25, 75])
    smoothed = np.array([np.median(series[max(i - smooth_frames, 0):i + smooth_frames + 1]) for i in index])
    smoothed = np.clip(smoothed, -yaw_bound_rad, yaw_bound_rad)
    band = (q25 - 3.0 * (q75 - q25), q75 + 3.0 * (q75 - q25))
    is_flagged = ~((smoothed >= band[0]) & (smoothed <= band[1]))
    polished = smoothed.copy()
    if polish_frames >= 5:
        from scipy.signal import savgol_filter
        start = 0
        for stop in list(np.flatnonzero(np.diff(is_flagged.astype(int)) != 0) + 1) + [count]:
            if not is_flagged[start] and stop - start >= 5:
                window = min(polish_frames | 1, stop - start)
                window -= 1 - window % 2
                polished[start:stop] = savgol_filter(smoothed[start:stop], window, 2)
            start = stop
        polished = np.clip(polished, -yaw_bound_rad, yaw_bound_rad)
    # The affine map of the camera's azimuth onto the radar's, sin_radar = a + b sin_mesh, on the
    # confident frames: the annotation frame the mesh inherits is scaled laterally per recording
    # session (RT-Pose Indoor-Clean 0.92-0.99, Indoor-Cluttered 1.02-1.19, Outdoor 1.01-1.14 against
    # the labels, docs/FIDELITY_FIX_RESULTS.md 3.14), and a yaw alone leaves an azimuth-dependent
    # residual. Applied per frame as the yaw asin(a + b s_f) - asin(s_f) of the mesh's own sine s_f.
    # The scale is identifiable only where the person spans azimuth (RT-Pose walking: a sine
    # std of 0.2-0.3; HuPR's person 0.01-0.08, a standing person 0.002) and the two sines
    # track (r >= 0.9); elsewhere the map is the yaw alone (b = 1, a = the median offset).
    spread = float(np.std(mesh_sine[confident])); tracking = float(np.corrcoef(mesh_sine[confident], radar_sine[confident])[0, 1])
    if spread >= 0.10 and tracking >= 0.90:
        design = np.stack([np.ones(int(confident.sum())), mesh_sine[confident]], 1)
        a_fit, b_fit = np.linalg.lstsq(design, radar_sine[confident], rcond=None)[0]
        b_fit = float(np.clip(b_fit, 0.7, 1.3)); a_fit = float(np.clip(a_fit, -0.3, 0.3))
    else:
        b_fit = 1.0; a_fit = float(np.clip(np.median(radar_sine[confident] - mesh_sine[confident]), -0.3, 0.3))
    mapped = np.clip(a_fit + b_fit * np.clip(mesh_sine, -1, 1), -1, 1)
    affine_yaw = np.arcsin(mapped) - np.arcsin(np.clip(mesh_sine, -1, 1))
    affine_yaw = np.where(np.isfinite(affine_yaw), affine_yaw, 0.0)
    return {
        "yaw_rad": float(np.median(raw[confident])),
        "affine": {"a_sine": a_fit, "b_scale": b_fit, "mesh_sine_std": spread, "sine_correlation": tracking,
                   "residual_deg_rms": float(np.degrees(np.sqrt(np.mean(
                       (np.arcsin(np.clip(radar_sine[confident], -1, 1)) - np.arcsin(mapped[confident])) ** 2))))},
        "affine_yaw_rad_per_frame": [float(v) for v in affine_yaw],
        "yaw_rad_per_frame": [float(v) for v in polished],
        "yaw_rad_per_frame_median_filtered": [float(v) for v in smoothed],
        "radar_sine_per_frame": [None if not np.isfinite(v) else float(v) for v in radar_sine],
        "mesh_sine_per_frame": [None if not np.isfinite(v) else float(v) for v in mesh_sine],
        "confident_frames": int(confident.sum()), "frames": int(count),
        "yaw_iqr_rad": [float(q25), float(q75)],
        "flagged_frames": [int(frames[i]) for i in index if is_flagged[i]],
        "min_confidence": min_confidence, "smooth_frames": smooth_frames, "polish_frames": polish_frames,
        "yaw_bound_rad": yaw_bound_rad,
        "estimator": "per frame: asin(radar moving-power azimuth centroid over +-0.19 in sine about its peak, "
                     "floor removed, at the registered range) - asin(scaled mesh centroid azimuth), the "
                     "beamformer's spatial sine; frames below the "
                     "detection confidence interpolated; median over +/-smooth_frames; unflagged runs "
                     "Savitzky-Golay polished; applied as a rotation about the radar's vertical axis",
    }


def person_range_series(dataset, sequence: int, frames: range, device: str, batch: int = 16,
                        sectors: dict | None = None,
                        ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """One pass over the recording: the person's radar range per frame, and the single-loop targets.

    Every radar of the rig is decoded once per frame (`MeasuredReader`), in
    blocks on the device; from each block come `radar_person_ranges` (the
    moving-power peak and its confidence) and the loop-0 |RA| maps the
    person material fit trains on (`{radar}_first_loop_ra.npy`, the GPU
    form of `single_loop_range_angle_map`: ~5e-16 relative to the numpy
    arbiter), one transfer per block. A rig of two boards contributes the
    confidence-weighted mean of their ranges and the mean confidence.
    """
    import torch
    from mm4DPS.data import MeasuredReader, range_angle_slow_time_cube_frames_torch
    count = len(frames)
    ranges = np.zeros((len(dataset.radars), count))
    confidences = np.zeros((len(dataset.radars), count))
    sines = np.zeros((len(dataset.radars), count))
    targets = {radar.name: np.zeros((count, radar.samples, radar.n_angle), dtype=np.float32)
               for radar in dataset.radars}
    for r, radar in enumerate(dataset.radars):
        reader = MeasuredReader(dataset, sequence, radar)
        try:
            for start in range(0, count, batch):
                stop = min(start + batch, count)
                block = reader.batch(frames[start], stop - start, device)
                sector = None
                if sectors is not None:
                    angle_mask, lo, hi = sectors[radar.name]
                    sector = (angle_mask[start:stop].to(device), lo[start:stop].to(device), hi[start:stop].to(device))
                measured, confidence, sine = radar_person_ranges(block, radar, sector=sector)
                ranges[r, start:stop] = measured.cpu().numpy()
                confidences[r, start:stop] = confidence.cpu().numpy()
                sines[r, start:stop] = sine.cpu().numpy()
                targets[radar.name][start:stop] = range_angle_slow_time_cube_frames_torch(
                    block, radar=radar)[:, 0].abs().cpu().numpy()
        finally:
            reader.close()
    weight = confidences / np.maximum(confidences.sum(0, keepdims=True), 1e-12)
    person_range_series.last_sines = sines          # per radar, read by the azimuth registration
    return (ranges * weight).sum(0), confidences.mean(0), targets


def run_person_sequence(args: argparse.Namespace) -> dict:
    """The person sequence the chain consumes, from a GVHMR dump of the RGB camera.

    Per frame the SMPL vertices with normals in the radar world, GVHMR's
    depth rescaled by the radar frame by frame (`radar_depth_scales`, the
    scale applied ABOUT THE CAMERA ORIGIN so every vertex keeps its pixel:
    GVHMR's error is its size prior and its assumed focal length, a body k
    times larger and farther, not a translation), the 14 HuPR joints from
    the SMPL regressor, the joints projected into the image; the loop-0
    measured range-azimuth targets; the scene points; then the per-frame
    visibility (compute-frame-visibility) on top.

    The camera's place in the radar world is the dataset's:

    * RT-Pose: camera -> LiDAR through the shipped calibration, LiDAR ->
      radar through `seeds/rtpose/radar_extrinsic.json` (the identity: the
      radar sits at the LiDAR); GVHMR row i is camera image i in file order,
      and radar frame f pairs with the image the rig's clock puts at f
      (`RTPoseSequence.camera_index`: the annotation's pairing, exactly
      linear -- three camera frames a radar frame on 85 sequences, two on 17).
    * HuPR: the camera at the rig, co-boresighted -- the camera frame's
      (right, down, forward) is the radar world's (x, -z, y); no extrinsic
      is published, and the person track fit of an earlier pass identified
      only the range (its similarity fit failed its own gate), which the
      per-frame registration now supplies. Camera image i is radar frame i.
    """
    import torch
    from mm4DPS.data import SmplBody, smpl_to_hupr_indices
    from mm4DPS.scene import vertex_normals
    from mm4DPS import Install
    dataset = Dataset.select(args.dataset)
    if dataset.name == "octonet":
        # no RGB camera and no GVHMR: the person is SMPL fitted to the 20-joint OptiTrack skeleton, already in
        # the MoCap world frame (mm4DPS/octonet_person.py writes person_sequence.npz + person_metadata.json)
        from mm4DPS.octonet_person import run_octonet_person
        return run_octonet_person(str(args.sequence), Path(args.output_dir), device=args.device)
    vertices_camera = np.load(args.vertices)["vertices_camera"].astype(np.float64)
    if dataset.name == "rtpose":
        from mm4DPS.data import RTPoseSequence
        seq = RTPoseSequence(dataset, args.sequence)
        recorded = seq.recording().frames
        clock = seq.clock()
        camera_rows = [seq.camera_index(frame) for frame in range(recorded)]
        if len(camera_rows) and max(camera_rows) >= len(vertices_camera):
            raise ValueError(f"GVHMR wrote {len(vertices_camera)} rows, the camera has {len(seq.camera_stems())} frames")
        seed = rtpose_extrinsic_seed(args.repo)
        camera = seq.camera("left")
        extrinsic, K = camera["extrinsic"], camera["intrinsic"][:, :3].copy()
        # The images on disk are a centred 1440x1080 crop of the calibrated 1468x1128 sensor.
        from PIL import Image
        width, height = Image.open(seq.image_path(seq.annotation()[min(seq.annotation())]["lidar"])).size
        K[0, 2] -= (2 * K[0, 2] - width) / 2.0
        K[1, 2] -= (2 * K[1, 2] - height) / 2.0
        no_shift = {"rotation": seed["rotation"], "translation_m": [0.0, 0.0, 0.0]}

        def to_world(points_camera: np.ndarray, vectors: bool = False) -> np.ndarray:
            moved = camera_to_lidar(points_camera, extrinsic)
            if vectors:
                moved = moved - camera_to_lidar(np.zeros((1, 3)), extrinsic)
            return seq.to_world(moved, no_shift if vectors else seed)
        placement = {"camera": {"intrinsics_cropped": K.tolist(), "image_size": [width, height],
                                "lidar_to_camera": extrinsic.tolist()},
                     "extrinsic_seed": seed,
                     "camera_pairing": {"clock": f"file stem = {clock[0]} * radar frame + {clock[1]}",
                                        "camera_row_per_frame": camera_rows,
                                        "camera_frames": len(seq.camera_stems())},
                     "geometry_method": "GVHMR SMPL mesh from the left camera; camera -> LiDAR by the shipped "
                                        "calibration, LiDAR -> radar by seeds/rtpose/radar_extrinsic.json; "
                                        "depth rescaled by the radar per frame"}
    elif dataset.name == "hupr":
        recorded = dataset.frame_count
        camera_rows = list(range(recorded))            # one camera frame per radar frame
        width = height = 512
        K = camera_intrinsics(height, width, dataset.camera_hfov_deg)        # the radar-estimated camera (mm4DPS.HUPR_CAMERA_HFOV_DEG)
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # (x, y, z)_cam -> (x, z, -y)

        def to_world(points_camera: np.ndarray, vectors: bool = False) -> np.ndarray:
            return points_camera @ axes.T
        placement = {"camera": {"intrinsics_assumed": K.tolist(), "image_size": [width, height],
                                "assumption": f"co-located with the rig, co-boresighted; {dataset.camera_hfov_deg:g} degree "
                                              "horizontal field of view (radar-estimated: GVHMR's default focal x the radar depth scale)"},
                     "camera_pairing": {"clock": "camera frame = radar frame", "camera_frames": recorded},
                     "geometry_method": "GVHMR SMPL mesh from the RGB camera; camera (right, down, forward) "
                                        "-> radar world (x, -z, y) at the rig; depth rescaled by the radar per frame"}
    else:
        raise NotImplementedError(dataset.name)
    frames = range(min(recorded, args.max_frames or 10 ** 9))

    def camera_index(frame: int) -> int:
        return min(camera_rows[frame], len(vertices_camera) - 1)

    gvhmr_range = np.array([
        float(np.hypot(*to_world(vertices_camera[camera_index(frame)].mean(0, keepdims=True))[0, :2]))
        for frame in frames])
    sectors = None
    if getattr(args, "registration_sector", "on") == "on":
        # F6: per radar, the (range, angle) mask of GVHMR's UNSCALED points (the renderer's own
        # `person_range_angle_masks_torch`, so the angle convention is the beamformer's) gives the
        # angular columns; the range window is the scale bounds about GVHMR's range.
        import torch
        from mm4DPS import array_axis, config_centre_m
        from mm4DPS.signal_model import person_range_angle_masks_torch
        sectors = {}
        pts = torch.stack([torch.as_tensor(to_world(vertices_camera[camera_index(frame)]), dtype=torch.float32)
                           for frame in frames]).to(args.device)                              # (F, N, 3)
        lo_m, hi_m = 0.5 * gvhmr_range, 2.0 * gvhmr_range
        for radar in dataset.radars:
            config = json.loads(dataset.config_path(radar).read_text())
            centre = torch.as_tensor(config_centre_m(config), dtype=torch.float32, device=args.device)
            axis_v = torch.as_tensor(np.asarray(array_axis(config)), dtype=torch.float32, device=args.device)
            # both mirrors: the measured cube's angle axis runs opposite to the mask's
            # n/2 + n/2 sin convention (docs/FIDELITY_FIX_RESULTS.md 3.11: the mesh's
            # column is the mirror of the column the return is in), and the flag exists
            # for exactly that; one-sided, the sector held the person's sidelobes only
            mask = person_range_angle_masks_torch(pts, centre, axis_v, radar.range_bin_m,
                                                  n_range=radar.samples, n_angle=radar.n_angle,
                                                  range_padding_bins=4, angle_padding_bins=3,
                                                  include_mirrored_angle=True)                    # (F, range, angle)
            angle_mask = mask.any(dim=1)
            lo = torch.as_tensor(np.floor(lo_m / radar.range_bin_m), dtype=torch.long)
            hi = torch.as_tensor(np.ceil(hi_m / radar.range_bin_m), dtype=torch.long)
            sectors[radar.name] = (angle_mask.cpu(), lo, hi)
        print(f"registration sector: the camera's angular columns ({int(sectors[dataset.radars[0].name][0].sum(1).float().mean())} "
              f"of {dataset.radars[0].n_angle} bins a frame) and the range window 0.5-2x GVHMR's", flush=True)
    radar_range, confidence, targets = person_range_series(dataset, args.sequence, frames, args.device, sectors=sectors)
    if args.depth_scale == "auto":
        scale = radar_depth_scales(radar_range, confidence, gvhmr_range, frames)
        scale["mode"] = dataset.depth_scale_mode
        if dataset.depth_scale_mode == "sequence":
            # one number for the sequence (Dataset.depth_scale_mode): GVHMR's depth error is its assumed focal, a constant
            # of the camera, so the median ratio over confident frames is the estimate; the per-frame series stays in the
            # metadata as a diagnostic and is not applied (HuPR: the board's per-frame range of a slow person is unreliable)
            per_frame = [scale["depth_scale"]] * len(frames)
        else:
            per_frame = scale["depth_scale_per_frame"]
        print(f"depth scale ({dataset.depth_scale_mode}): median {scale['depth_scale']:.3f}, IQR {scale['ratio_iqr'][0]:.3f}-"
              f"{scale['ratio_iqr'][1]:.3f}, {scale['confident_frames']} of {scale['frames']} frames detected, "
              f"{len(scale['flagged_frames'])} flagged"
              + (" (the per-frame series recorded, not applied)" if dataset.depth_scale_mode == "sequence" else ""), flush=True)
    else:
        # one given number for every frame: a diagnostic, never the chain's path
        scale = {"depth_scale": float(args.depth_scale), "estimator": "given"}
        per_frame = [scale["depth_scale"]] * len(frames)

    body = SmplBody(args.model_path)
    faces = body.faces
    regressor = body.joint_regressor.numpy()
    joint_map = smpl_to_hupr_indices()
    # every frame's vertices at once: scaled about the camera origin, then
    # the normals in the camera frame (float64, as the arbiter computed them)
    if args.depth_registration == "translate":
        # The body keeps GVHMR's metric size and moves along the camera ray of its centroid
        # to the radar's range. GVHMR's error is its depth (the focal length it assumes),
        # not the body's size: scaled about the camera origin, the HuPR person came out
        # 1.07 m tall (GVHMR's 1.8 m x the 0.71 range ratio) and the RT-Pose limbs 1.0-1.3x
        # the labels' (docs/FIDELITY_FIX_RESULTS.md 3.12). The pixels are no longer the
        # camera's; only the diagnostic video's overlay reads them.
        scaled = np.stack([
            vertices_camera[camera_index(frame)]
            + (per_frame[i] - 1.0) * vertices_camera[camera_index(frame)].mean(0, keepdims=True)
            for i, frame in enumerate(frames)])
    else:
        scaled = np.stack([per_frame[i] * vertices_camera[camera_index(frame)] for i, frame in enumerate(frames)])
    image_registration = None
    if getattr(args, "image_registration", "gt") == "gt" and dataset.image_keypoints:
        # The ray from the dataset's own 2D keypoints, the range from the radar: the body (GVHMR's shape and pose) is
        # shifted in the camera frame, at its depth, so that its projected joints centre on the released keypoints
        # (the median offset over the 14 joints, a running median over +-5 frames against the annotation's jitter);
        # GVHMR's translation -- the part of its output that carries its assumed focal -- no longer places the person.
        # The scale between the projected skeleton and the keypoints, per frame, is the focal the image implies at
        # the radar's range: reported as the camera estimate (mm4DPS.HUPR_CAMERA_HFOV_DEG is set from it).
        from downstream.common.split import Split
        from downstream.hupr_mscsa.dataset import load_annotations
        blocks = load_annotations(Split.load(dataset.name))[int(args.sequence)]
        label_w = 256.0                                                            # the annotations' frame
        gt = np.stack([np.array(blocks[camera_index(frame)]["joints"], float) for frame in frames]) * (width / label_w)   # (F, 14, 2)
        c = np.array([K[0, 2], K[1, 2]])
        du, dv, f_est = np.zeros(len(frames)), np.zeros(len(frames)), np.zeros(len(frames))
        for i in range(len(frames)):
            joints_cam = (regressor @ scaled[i])[joint_map]; z = np.clip(joints_cam[:, 2], 1e-6, None)
            proj = np.stack([K[0, 0] * joints_cam[:, 0] / z + K[0, 2], K[1, 1] * joints_cam[:, 1] / z + K[1, 2]], -1)
            off = np.median(gt[i] - proj, axis=0); du[i], dv[i] = off
            a = (proj - c).ravel(); b = (gt[i] - c - off).ravel()
            f_est[i] = K[0, 0] * float(a @ b / max(a @ a, 1e-9))
        from scipy.ndimage import median_filter
        du_s, dv_s = median_filter(du, 11, mode="nearest"), median_filter(dv, 11, mode="nearest")
        for i in range(len(frames)):
            zbar = float(scaled[i][:, 2].mean())
            scaled[i] = scaled[i] + np.array([du_s[i] * zbar / K[0, 0], dv_s[i] * zbar / K[1, 1], 0.0])
        image_registration = {"source": "the dataset's 2D keypoints", "offset_px_median": [float(np.median(du)), float(np.median(dv))],
                              "offset_px_iqr": [float(np.percentile(du, 75) - np.percentile(du, 25)), float(np.percentile(dv, 75) - np.percentile(dv, 25))],
                              "focal_px_estimate_median": float(np.median(f_est)), "focal_px_estimate_iqr": [float(np.percentile(f_est, 25)), float(np.percentile(f_est, 75))],
                              "focal_px_assumed": float(K[0, 0]), "image_width": int(width)}
        print(f"image registration on the dataset's 2D keypoints: median offset ({np.median(du):+.1f}, {np.median(dv):+.1f}) px of {width}, "
              f"focal the image implies at the radar's range: median {np.median(f_est):.0f} px (IQR {np.percentile(f_est, 25):.0f}-{np.percentile(f_est, 75):.0f}; "
              f"assumed {K[0, 0]:.0f}; HFOV {np.degrees(2 * np.arctan(width / 2 / np.median(f_est))):.1f} deg)", flush=True)
    normals_camera = vertex_normals(torch.as_tensor(scaled, device=args.device), faces.to(args.device)).cpu().numpy()

    # the per-frame visibility of the person's vertices and of the scene
    # points behind them, on the device, before anything is written
    from mm4DPS.pointset import radar_centre_and_boresight, sequence_visibility
    scene_np = np.load(args.scaffold_dir / "pcl.npy")
    world = np.stack([to_world(scaled[i]) for i in range(len(frames))])
    world_normals = np.stack([to_world(normals_camera[i], vectors=True) for i in range(len(frames))])
    yaw = None
    azimuth_skipped = None
    if getattr(args, "registration_azimuth", "off") == "on":
        # the azimuth registration (radar_azimuth_yaw), on the dataset's azimuth array
        from mm4DPS import array_axis, config_centre_m
        first = dataset.radars[0]
        config = json.loads(dataset.config_path(first).read_text())
        centre_np = np.asarray(config_centre_m(config), dtype=np.float64)
        axis_np = np.asarray(array_axis(config), dtype=np.float64); axis_np /= np.linalg.norm(axis_np)
        delta = world.mean(1) - centre_np[None]                                              # (F, 3)
        # the beamformer's spatial sine runs opposite to the mask convention's (3.11); the
        # sign is checked on the confident frames rather than trusted, and the check is loud
        mesh_sine = -(delta @ axis_np) / np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
        radar_sine = person_range_series.last_sines[0]
        sure = confidence >= 0.35
        fits = [float(np.median(np.abs(radar_sine[sure] - sign * mesh_sine[sure]))) for sign in (1.0, -1.0)]
        print(f"azimuth registration: median |radar sine - mesh sine| {fits[0]:.3f} with the beamformer's sign, "
              f"{fits[1]:.3f} mirrored", flush=True)
        if fits[0] > fits[1]:
            # The registration corrects the extrinsic's yaw, which needs a return attributable to
            # this mesh's person. When the mirrored sign fits better the radar's moving return is
            # not that person's (a second person on the other side: RT-Pose 100-105, 117-140,
            # 170-175 record two people and GVHMR tracks one); the convention itself holds on
            # every one-person sequence of both datasets. The mesh keeps the camera's azimuth.
            azimuth_skipped = (f"off: the radar's return does not fit this mesh's person (median |sine| "
                               f"residual {fits[0]:.3f} with the array's sign, {fits[1]:.3f} mirrored)")
            print(f"azimuth registration {azimuth_skipped}", flush=True)
    applied_rotation = None
    if getattr(args, "registration_azimuth", "off") == "on" and dataset.azimuth_registration == "constant":
        # one rig yaw for the dataset (Dataset.camera_radar_yaw_deg), the same rotation on every frame; the per-sequence
        # fit is kept as a diagnostic in the metadata (it is the board's noise around the constant on HuPR)
        fit = radar_azimuth_yaw(radar_sine, confidence, mesh_sine, frames) if azimuth_skipped is None else None
        boresight = np.zeros(3); boresight[1] = 1.0
        up = np.cross(axis_np, boresight); up /= np.linalg.norm(up)

        def rotation(angle: float) -> np.ndarray:
            c, s_, u = np.cos(angle), np.sin(angle), up
            k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            return np.eye(3) + s_ * k + (1 - c) * (k @ k)
        probe = -(rotation(0.01) @ delta[0]) @ axis_np / np.linalg.norm(delta[0]) - mesh_sine[0]
        sign = 1.0 if probe > 0 else -1.0
        angle = sign * np.radians(dataset.camera_radar_yaw_deg)
        R = rotation(angle)
        for i in range(len(frames)):
            world[i] = (world[i] - centre_np) @ R.T + centre_np
            world_normals[i] = world_normals[i] @ R.T
        applied_rotation = {"mode": "constant", "yaw_deg": float(dataset.camera_radar_yaw_deg), "rotation_rad_about_up": float(angle),
                            "up": [float(v) for v in up], "centre_m": [float(v) for v in centre_np],
                            "sequence_fit": (fit if fit is not None else azimuth_skipped)}
        yaw = applied_rotation; azimuth_skipped = None
        print(f"azimuth registration: the rig constant {dataset.camera_radar_yaw_deg:+.2f} deg applied (rotation {np.degrees(angle):+.2f} deg about the board's up); "
              f"this sequence's own fit: {('yaw median %+.2f deg (IQR %+.2f..%+.2f)' % (np.degrees(fit['yaw_rad']), np.degrees(fit['yaw_iqr_rad'][0]), np.degrees(fit['yaw_iqr_rad'][1]))) if fit is not None else 'skipped (the mirror fit better)'}", flush=True)
    if yaw is None and azimuth_skipped is None and getattr(args, "registration_azimuth", "off") == "on":
        yaw = radar_azimuth_yaw(radar_sine, confidence, mesh_sine, frames)
        # the rotation axis: the board's vertical, boresight x array axis, signed so that a
        # positive yaw increases the spatial sine (checked numerically on the first frame)
        boresight = np.zeros(3); boresight[1] = 1.0
        up = np.cross(axis_np, boresight); up /= np.linalg.norm(up)

        def rotation(angle: float) -> np.ndarray:
            c, s_, u = np.cos(angle), np.sin(angle), up
            k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            return np.eye(3) + s_ * k + (1 - c) * (k @ k)
        probe = -(rotation(0.01) @ delta[0]) @ axis_np / np.linalg.norm(delta[0]) - mesh_sine[0]
        sign = 1.0 if probe > 0 else -1.0
        # one yaw for the sequence: the offset is the extrinsic's (a fixed rotation between the
        # camera's frame and the radar's), and the per-frame estimate wanders with the limb
        # phase (+-1.3 deg after smoothing = 8 cm at 3.5 m); the per-frame series stays in the
        # record as the diagnostic
        yaw["applied"] = "affine: asin(a + b sin_mesh) - asin(sin_mesh) per frame"
        for i in range(len(frames)):
            R = rotation(sign * yaw["affine_yaw_rad_per_frame"][i])
            world[i] = (world[i] - centre_np) @ R.T + centre_np
            world_normals[i] = world_normals[i] @ R.T
        after = -(world.mean(1) - centre_np) @ axis_np / np.maximum(np.linalg.norm(world.mean(1) - centre_np, axis=1), 1e-8)
        residual = np.degrees(np.arcsin(np.clip(np.asarray(yaw["radar_sine_per_frame"], dtype=float), -1, 1)) - np.arcsin(np.clip(after, -1, 1)))
        confident = np.asarray(confidence) >= yaw["min_confidence"]
        print(f"azimuth registration: affine sin_radar = {yaw['affine']['a_sine']:+.4f} + {yaw['affine']['b_scale']:.4f} sin_mesh "
              f"(residual {yaw['affine']['residual_deg_rms']:.2f} deg rms); yaw median {np.degrees(yaw['yaw_rad']):+.2f} deg (IQR "
              f"{np.degrees(yaw['yaw_iqr_rad'][0]):+.2f}..{np.degrees(yaw['yaw_iqr_rad'][1]):+.2f}), "
              f"{yaw['confident_frames']} of {yaw['frames']} frames detected, {len(yaw['flagged_frames'])} flagged; "
              f"residual radar - mesh azimuth over confident frames {np.nanmean(residual[confident]):+.2f} +- "
              f"{np.nanstd(residual[confident]):.2f} deg (before: {np.degrees(yaw['yaw_rad']):+.2f})", flush=True)
    centre, boresight = radar_centre_and_boresight(Install.from_env().config_dir, args.device)
    person_visible, scene_visible = sequence_visibility(
        torch.as_tensor(world, device=args.device, dtype=torch.float32),
        torch.as_tensor(scene_np[:, :3], device=args.device), torch.zeros((0, 3, 3), device=args.device),
        faces.to(device=args.device, dtype=torch.long), centre, boresight)
    person_visible, scene_visible = person_visible.cpu().numpy(), scene_visible.cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "person_frames").mkdir(exist_ok=True)
    records = []
    for i, frame in enumerate(frames):
        verts_cam = scaled[i]
        verts = world[i]
        normals = world_normals[i]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)
        joints_cam = (regressor @ verts_cam)[joint_map]
        # the joints from the (registered) world vertices: the regressor is linear
        joints = (regressor @ verts)[joint_map]
        depth = np.clip(joints_cam[:, 2], 1e-6, None)
        joints_xy = np.stack([K[0, 0] * joints_cam[:, 0] / depth + K[0, 2],
                              K[1, 1] * joints_cam[:, 1] / depth + K[1, 2]], axis=-1)
        bbox = np.array([joints_xy[:, 0].min(), joints_xy[:, 1].min(), joints_xy[:, 0].max(), joints_xy[:, 1].max()])
        np.savez(
            args.output_dir / "person_frames" / f"frame_{frame:03d}.npz",
            pcl=np.concatenate([verts, normals], axis=-1).astype(np.float32),
            joints_xyz=joints.astype(np.float32), joints_xy=joints_xy.astype(np.float32),
            bbox_xyxy=bbox.astype(np.float32), intrinsics=K.astype(np.float32),
            mask_pixels=np.int64(0), visible=person_visible[i])
        records.append({"frame": int(frame), "person_points": int(len(verts)), "valid_3d_joints": 14,
                        "mask_pixels": 0})
    for radar in dataset.radars:
        np.save(args.output_dir / f"{radar.name}_first_loop_ra.npy", targets[radar.name])
    np.save(args.output_dir / "applied_range_shift_m.npy", np.zeros(len(frames)))
    np.save(args.output_dir / "global_scene_points.npy", scene_np)
    np.save(args.output_dir / "scene_visibility.npy", scene_visible)
    visibility = {
        "global_scene_points": int(len(scene_np)), "person_points": int(body.vertex_count),
        "global_total": int(len(scene_np) + body.vertex_count),
        "person_visible": {"median": int(np.median(person_visible.sum(1))),
                           "min": int(person_visible.sum(1).min()), "max": int(person_visible.sum(1).max())},
        "scene_shadowed_by_person": {"median": int(np.median((~scene_visible).sum(1))),
                                     "min": int((~scene_visible).sum(1).min()), "max": int((~scene_visible).sum(1).max())},
        "steps_per_frame": ["fov", "ray_cast_occlusion"],
        "note": "no cosine, no FPS per frame: visibility is deterministic and identity-preserving; "
                "beam gain lives in the renderer",
    }
    first = dataset.radars[0]
    metadata = {
        "dataset": dataset.name, "sequence": int(args.sequence), "start_frame": 0, "frames": len(frames),
        "slow_time_loop_index": 0,
        "single_loop_definition": ("one TDM loop, the nine azimuth TX with all sixteen RX on the 86-slot grid"
                                   if dataset.name == "rtpose" else "one TDM loop of the azimuth TX"),
        "target_shape": list(targets[first.name].shape),
        "person_point_count": {"min": int(body.vertex_count), "median": float(body.vertex_count),
                               "max": int(body.vertex_count)},
        "all_frames_complete": True,
        "material_consistency_contract": "geometry varies per frame; downstream training must broadcast one "
                                         "shared human material parameter set to every frame",
        **placement,
        "gvhmr_conversion": {"vertices_dump": str(args.vertices.resolve()), "vertex_stride": 1,
                             "vertices_written": int(body.vertex_count), "depth_scale": scale,
                             "camera_pairing": placement["camera_pairing"],
                             "depth_scaling": ("a translation along the centroid's camera ray to k times its "
                                               "range, the body's size GVHMR's"
                                               if args.depth_registration == "translate" else
                                               "about the camera origin: pixels preserved, body k times larger "
                                               "and farther")
                                              + (" -- k per frame" if dataset.depth_scale_mode == "frame" else
                                                 " -- one k for the sequence, the median over confident frames"),
                             "depth_scale_mode": dataset.depth_scale_mode,
                             "depth_scale_applied_per_frame": [float(v) for v in per_frame],
                             "depth_registration": args.depth_registration,
                             "image_registration": image_registration,
                             "azimuth_registration": yaw if yaw is not None else (azimuth_skipped or "off")},
        "records": records,
        "frame_visibility": visibility,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"status": "completed", "frames": len(frames), "depth_scale": scale, "output_dir": str(args.output_dir),
            "visibility": visibility}


def make_parser_person_sequence() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_person_sequence.__doc__)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--vertices", type=Path, required=True, help="GVHMR dump: npz with vertices_camera")
    parser.add_argument("--scaffold-dir", type=Path, required=True, help="the scene scaffold (pcl.npy)")
    parser.add_argument("--output-dir", type=Path, required=True)
    from mm4DPS.data import DEFAULT_MODEL as SMPL_MODEL
    parser.add_argument("--model-path", type=Path, default=SMPL_MODEL, help="the SMPL npz")
    parser.add_argument("--image-registration", choices=("gt", "off"), default="gt",
                        help="gt (default): on a dataset with 2D keypoints in the RGB frame (Dataset.image_keypoints: HuPR), the body's "
                             "ray comes from them and its range from the radar (GVHMR's translation no longer places it); off / a "
                             "dataset without them: GVHMR's ray")
    parser.add_argument("--depth-registration", choices=("translate", "scale"), default="translate",
                        help="how the radar's range ratio moves the body: `translate` along the centroid's "
                             "camera ray, the body's size GVHMR's (the production setting, 3.12); `scale` "
                             "about the camera origin, pixels preserved but the body resized -- the "
                             "pre-2026-09-17 form, an ablation")
    parser.add_argument("--registration-azimuth", choices=("off", "on"), default="on",
                        help="register the person's azimuth to the radar's moving return (radar_azimuth_yaw: "
                             "one yaw per sequence about the radar's vertical), as its range is registered "
                             "per frame (docs/FIDELITY_FIX_RESULTS.md 3.11, the production setting); off = "
                             "the camera's azimuth, an ablation")
    parser.add_argument("--registration-sector", choices=("off", "on"), default="on",
                        help="the radar range search confined to the camera's angular sector and the "
                             "scale-bound range window about GVHMR's range (docs/FIDELITY_FIX_RESULTS.md 3.5, "
                             "the production setting); off = the pre-2026-09-16 search over every angle")
    parser.add_argument("--depth-scale", default="auto",
                        help="'auto' (the only production path): from the radar (radar_depth_scales), per frame or "
                             "one per sequence as Dataset.depth_scale_mode says; or one number for every frame, a diagnostic")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--dataset", required=True, choices=("hupr", "rtpose"))
    return parser


def _entry_person_sequence():
    from mm4DPS import dataset_from_argv
    if dataset_from_argv().name == "octonet":
        # the HuPR/RT-Pose parser demands a GVHMR dump; OctoNet has none
        p = argparse.ArgumentParser(description="OctoNet person stage: SMPL fitted to the OptiTrack skeleton")
        p.add_argument("--dataset"); p.add_argument("--sequence", required=True, help="14-digit recording stamp")
        p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--device", default="cuda:0")
        args = p.parse_args([a for a in sys.argv[1:] if a != "person-sequence"])   # dispatch() strips the command
        Dataset.select("octonet")
        result = run_person_sequence(args)
        print(json.dumps({k: v for k, v in (result or {}).items() if not hasattr(v, "shape")}, indent=1, default=str)[:3000])
        return
    print(json.dumps(run_person_sequence(make_parser_person_sequence().parse_args()), indent=2))


# ----------------------------------------------------------------------
# scaffold-cut: the person's swept volume out of the scaffold (docs/STATIC_SEQS_NEXT.md 2b)
# ----------------------------------------------------------------------
SCAFFOLD_CUT_RADIUS_M = 0.25     # a scaffold point this close to the registered person's surface in any frame is the person
SCAFFOLD_CUT_FLOOR_M = 0.08      # ... unless it lies within this of the floor under the feet (the floor stays whole)


def run_scaffold_cut(args) -> dict:
    """Remove the person's remnants from the scaffold with the registered mesh, frame by frame.

    The scaffold cut the person out where the dataset's annotation put them
    (RT-Pose: a 0.6 m column about the annotated joints; HuPR: the bbox in
    the depth map). Where the annotation is off the person -- RT-Pose 108 /
    168 / 164: 0.45-0.6 m laterally, the same offset the mesh-derived labels
    correct -- the person's own LiDAR and depth points stay in the scene and
    render as a static scatterer at the person's place that the recording
    does not have (the ghost that beats the person in the downstream's RA on
    164 / 108). The person sequence registered to the radar is where the
    person is; every scaffold point within `radius_m` of the person's points
    in any sampled frame is removed, except those at floor level under the
    feet, so the floor stays whole. The background is fitted again on the
    cut scaffold by the chain (person_stack).
    """
    from scipy.spatial import cKDTree
    scaffold_path = args.scaffold_dir / "pcl.npy"
    scaffold = np.load(scaffold_path)
    frames = sorted((args.sequence_dir / "person_frames").glob("frame_*.npz"))
    if not frames:
        raise FileNotFoundError(f"no person frames under {args.sequence_dir}")
    keep = np.ones(len(scaffold), dtype=bool)
    xyz = scaffold[:, :3].astype(np.float64)
    floors = []
    for path in frames[::args.stride]:
        person = np.load(path)["pcl"][:, :3].astype(np.float64)
        floor = float(np.percentile(person[:, 2], 2))            # the feet
        floors.append(floor)
        distance, _ = cKDTree(person).query(xyz, distance_upper_bound=args.radius_m)
        hit = (distance < args.radius_m) & (xyz[:, 2] > floor + args.floor_m)
        keep &= ~hit
    removed = int((~keep).sum())
    if not args.scaffold_dir.joinpath("pcl_before_cut.npy").is_file():
        np.save(args.scaffold_dir / "pcl_before_cut.npy", scaffold)  # the uncut scaffold, for the record and the viewers
    cut = scaffold[keep]
    np.save(scaffold_path, cut)
    if (args.scaffold_dir / "global_scene_points.npy").is_file():
        np.save(args.scaffold_dir / "global_scene_points.npy", cut)
    result = {"scaffold_points": int(len(scaffold)), "removed": removed, "kept": int(len(cut)),
              "radius_m": args.radius_m, "floor_m": args.floor_m, "frames_sampled": len(frames[::args.stride]),
              "floor_z_m": {"median": float(np.median(floors)), "min": float(min(floors)), "max": float(max(floors))},
              "rule": "scaffold points within radius_m of the registered person's points in any sampled frame, above "
                      "the feet by floor_m, are the person's remnants and are removed; the floor under the feet stays"}
    (args.scaffold_dir / "scaffold_cut.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_parser_scaffold_cut() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_scaffold_cut.__doc__)
    parser.add_argument("--scaffold-dir", type=Path, required=True, help="the workspace's own scaffold (pcl.npy is rewritten)")
    parser.add_argument("--sequence-dir", type=Path, required=True, help="the registered person sequence (person_frames/*.npz)")
    parser.add_argument("--radius-m", type=float, default=SCAFFOLD_CUT_RADIUS_M)
    parser.add_argument("--floor-m", type=float, default=SCAFFOLD_CUT_FLOOR_M)
    parser.add_argument("--stride", type=int, default=2, help="every stride-th frame of the person sequence")
    return parser


def _entry_scaffold_cut():
    print(json.dumps(run_scaffold_cut(make_parser_scaffold_cut().parse_args()), indent=2))


COMMANDS = {
    "scaffold": _entry_scaffold,
    "lidar-scaffold": _entry_lidar_scaffold,
    "person-sequence": _entry_person_sequence,
    "scaffold-cut": _entry_scaffold_cut,
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
