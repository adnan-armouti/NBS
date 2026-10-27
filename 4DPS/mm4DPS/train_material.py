"""The human material trainer, and the SMPL fit it can start from.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time

from mm4DPS.data import range_bin_spacing_m, range_keep_mask

from mm4DPS import Dataset, current_dataset, dispatch, radar_configs, sequence_dir

from mm4DPS.data import SmplBody, smpl_to_hupr_indices

from mm4DPS.scene import (
    person_solid_angle_weights, scene_solid_angle_weights, smpl_faces,
    assert_persistent_person_geometry,
    transform_person_pcl,
    vertex_normals)

import argparse

import numpy as np

import torch
from mm4DPS.signal_model import person_range_angle_mask_torch, person_range_angle_masks_torch
from mm4DPS.forward_render import (log_pearson_loss,
                                              range_profiles_to_ra_magnitude,
                                              range_profiles_to_ra_magnitude_frames)
from mm4DPS.cuda_ext import splat_scatter
from mm4DPS import MM3DPS_ROOT, WORKSPACE
from mm4DPS import HUPR_NEAR_FIELD_M, array_axis, config_centre_m
from pathlib import Path


# ----------------------------------------------------------------------
# person_sequence: Train one interpretable human material across every HuPR sequence frame.
# ----------------------------------------------------------------------



DEFAULT_MM3DPS_ROOT = Path(MM3DPS_ROOT)



class SharedHumanMaterial(torch.nn.Module):
    """Shared material with a constrained, reciprocal H/V amplitude balance."""

    def __init__(self, initial_raw: torch.Tensor, initial_area_scale: float = 0.02) -> None:
        super().__init__()
        if initial_raw.shape != (6,):
            raise ValueError(f"expected six raw material parameters, got {initial_raw.shape}")
        if initial_area_scale <= 0:
            raise ValueError("initial area scale must be positive")
        self.raw = torch.nn.Parameter(initial_raw.clone())
        self.log_area_scale = torch.nn.Parameter(
            torch.tensor(float(np.log(initial_area_scale)), dtype=initial_raw.dtype,
                         device=initial_raw.device)
        )
        # A zero-mean log gain prevents two unconstrained radar-specific
        # materials: the dataset's first radar uses exp(+delta), its second
        # exp(-delta); a single radar has no balance to learn and gets 1.
        self.radar_log_gain_delta = torch.nn.Parameter(
            torch.zeros((), dtype=initial_raw.dtype, device=initial_raw.device)
        )

    def for_points(self, count: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("person cloud must contain at least one point")
        return self.raw.unsqueeze(0).expand(count, -1)

    def areas_for_points(self, count: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("person cloud must contain at least one point")
        return self.log_area_scale.exp().expand(count)

    def radar_gain(self, radar_name: str) -> torch.Tensor:
        names = current_dataset().names
        if radar_name not in names:
            raise ValueError(f"unknown radar name: {radar_name}")
        if len(names) == 1:
            return torch.ones_like(self.radar_log_gain_delta)
        if len(names) != 2:
            raise NotImplementedError("the radar balance is a two-radar constraint")
        sign = 1.0 if names.index(radar_name) == 0 else -1.0
        return (sign * self.radar_log_gain_delta).exp()






def prepare_scene_layout(static_pcl: Path, output_dir: Path) -> Path:
    scene_dir = output_dir / "input_scene" / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    destination = scene_dir / "pcl.npy"
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(static_pcl.resolve())
    except OSError:
        shutil.copyfile(static_pcl, destination)
    return scene_dir / "mesh.ply"






def train(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("sequence training requires an available CUDA device")
    if not 0 < args.min_area_scale <= args.initial_area_scale <= 1.0:
        raise ValueError("require 0 < min_area_scale <= initial_area_scale <= 1")
    if not 0.0 <= args.amplitude_weight <= 1.0:
        raise ValueError("amplitude_weight must lie in [0,1]")
    if not 0.0 < args.amplitude_reference_quantile < 1.0:
        raise ValueError("amplitude_reference_quantile must lie in (0,1)")
    if args.radar_balance_lr < 0 or args.max_radar_balance_db < 0:
        raise ValueError("radar balance learning rate and limit must be nonnegative")
    if args.frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from mm4DPS import rasterizer as bsdf_torch
    from mm4DPS.rasterizer import (
        Rasterizer,
        inverse_reparameterize_torch,
        reparameterize_torch,
    )
    from mm4DPS.rasterizer import render_factorized
    from mm4DPS.rasterizer import (
        ITU_CONCRETE,
        PointPrimitives,
        cull_gaussians,
        render_gaussians,
        rms_clip_grad,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    # GVHMR only: the material this fits is per-point, so a cloud that is
    # rebuilt every frame with a different length cannot carry it. Checked,
    # not documented, because training on one does not fail -- it quietly
    # collapses to a single shared vector.
    assert_persistent_person_geometry(args.sequence_dir)
    sequence_metadata = json.loads((args.sequence_dir / "metadata.json").read_text())
    if not sequence_metadata["all_frames_complete"]:
        raise ValueError("sequence preprocessing is incomplete")
    available_frames = int(sequence_metadata["frames"])
    frames = list(range(0, available_frames, args.frame_stride))
    excluded_frames = {
        int(value) for value in args.exclude_frames.split(",") if value.strip()
    }
    if any(frame < 0 or frame >= available_frames for frame in excluded_frames):
        raise ValueError("excluded frame is outside the sequence")
    frames = [frame for frame in frames if frame not in excluded_frames]
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError("no training frames selected")
    dataset = Dataset.of(args)
    names = dataset.names
    targets = {name: np.load(args.sequence_dir / f"{name}_first_loop_ra.npy", mmap_mode="r")
               for name in names}
    if len({t.shape for t in targets.values()}) != 1 or min(
            t.shape[0] for t in targets.values()) < len(frames):
        raise ValueError("sequence radar targets are incompatible")
    # the per-cell median over the training frames, on the device (numpy's
    # partition of 600 x 256 x 64 was 0.8 s of this stage; the same value
    # to an ULP: the mean of the two middle values)
    medians = {
        name: torch.quantile(torch.from_numpy(np.asarray(targets[name])[frames]).to(device), 0.5, dim=0)
        for name in names}

    people: dict[int, np.ndarray] = {}
    for frame in frames:
        pcl = np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")["pcl"]
        if pcl.ndim != 2 or pcl.shape[1] != 6 or not np.isfinite(pcl).all():
            raise ValueError(f"invalid person cloud for frame {frame}: {pcl.shape}")
        people[frame] = transform_person_pcl(
            pcl, args.geometry_depth_scale, args.geometry_translation_y_m,
        )

    config_values = dataset.configs(args)
    # The masks, on the device. The numpy form runs a Python loop of
    # 2 x 9 x 7 = 126 boolean scatters over the whole cloud per (frame,
    # radar) -- profiled at 12 ms a call, which over 600 frames and two
    # radars was **14.2 s of this stage's 82**, the largest single item in
    # it. `person_range_angle_mask_torch` was written for exactly this and
    # was being called only by the renderer. The arithmetic is integer after
    # `round`, so the two agree exactly unless a point's spatial sine lands
    # within an ULP of a half-bin boundary; test_hupr_chirp_motion.py pins
    # them against each other.
    person_ra_masks: dict[int, dict[str, "torch.Tensor"]] = {frame: {} for frame in people}
    spacing = range_bin_spacing_m()
    order = list(people)
    if len({people[frame].shape for frame in order}) == 1:
        # a persistent cloud: every frame's mask in one scatter per radar
        stacked = torch.from_numpy(np.stack([people[frame][:, :3] for frame in order])).to(device)
        for name, config in config_values.items():
            masks = person_range_angle_masks_torch(
                stacked, torch.as_tensor(config_centre_m(config), dtype=stacked.dtype, device=device),
                torch.as_tensor(np.asarray(array_axis(config)), dtype=stacked.dtype, device=device),
                spacing, include_mirrored_angle=True)
            for index, frame in enumerate(order):
                person_ra_masks[frame][name] = masks[index]
    else:
        for frame, pcl in people.items():
            points = torch.from_numpy(pcl[:, :3]).to(device)
            for name, config in config_values.items():
                person_ra_masks[frame][name] = person_range_angle_mask_torch(
                    points,
                    torch.as_tensor(config_centre_m(config),
                                    dtype=points.dtype, device=device),
                    torch.as_tensor(np.asarray(array_axis(config)),
                                    dtype=points.dtype, device=device),
                    spacing, include_mirrored_angle=True,
                )

    pattern_root = args.mm3dps_root / "assets/antenna_pattern/MMWCAS"
    mesh_file = prepare_scene_layout(args.static_pcl, args.output_dir)
    common = dict(
        mesh_file=str(mesh_file),
        tx_pattern_file=str(pattern_root / "tx1_76.npy"),
        rx_pattern_file=str(pattern_root / "rx1_76.npy"),
        device=str(device),
    )
    rasterizers = {name: Rasterizer(config_file=str(path), **common)
                   for name, path in dataset.config_flags(args).items()}
    bsdf_torch.set_frequency(rasterizers[names[0]].center_freq)

    background_state = torch.load(args.background_model, map_location=device, weights_only=False)
    background = PointPrimitives(len(background_state["positions"]), device=str(device))
    background.load_state_dict(background_state)
    background.requires_grad_(False).eval()
    background.raw_materials.requires_grad_(args.train_background_materials)
    background_masks: dict[str, torch.Tensor] = {}
    background_areas: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, rasterizer in rasterizers.items():
            mask = cull_gaussians(
                background, rasterizer, near_field_m=HUPR_NEAR_FIELD_M,
            )
            background_masks[name] = mask
            background_areas[name] = mask.to(dtype=torch.float32)
    # Option B: the scene's physical slot and unit, the person's per-frame physical slot in that unit
    solid_angle_unit: dict[str, float] = {}
    person_slot: dict[str, torch.Tensor] = {}
    if args.scene_solid_angle == "physical" or args.person_solid_angle_weight == "physical":
        faces = smpl_faces() if args.person_solid_angle_weight == "physical" else None
        with torch.no_grad():
            for name, config in config_values.items():
                centre = np.asarray(config_centre_m(config), dtype=np.float64)
                weights, unit = scene_solid_angle_weights(background.positions.detach(), background.get_normals().detach(), centre)
                solid_angle_unit[name] = unit
                if args.scene_solid_angle == "physical":
                    background_areas[name] = background_areas[name] * weights.float()
                if faces is not None:
                    person_slot[name] = torch.stack([
                        person_solid_angle_weights(people[frame], faces, np.arange(len(people[frame])), centre, unit, device).float()
                        for frame in frames])                                              # (frames, V)
                    print(f"solid angle ({name}): scene unit {unit:.3e} sr; person slot p10/p50/p90 "
                          f"{float(person_slot[name].quantile(0.1)):.4f}/{float(person_slot[name].median()):.4f}/{float(person_slot[name].quantile(0.9)):.4f}", flush=True)

    def render_background(
        name: str, rasterizer: Rasterizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if name in background_profile_cache:
            return background_profile_cache[name]
        return render_gaussians(
            background, rasterizer, vertex_areas=background_areas[name],
            active_mask=background_masks[name], shadow_mask=None,
            detach_phase=True, bsdf_mode="full", disabled_components=None,
            use_cuda_kernels=True,
        )

    background_profile_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if not args.train_background_materials:
        with torch.no_grad():
            for name, rasterizer in rasterizers.items():
                background_profile_cache[name] = tuple(
                    value.detach() for value in render_gaussians(
                        background, rasterizer, vertex_areas=background_areas[name],
                        active_mask=background_masks[name], shadow_mask=None,
                        detach_phase=True, bsdf_mode="full", disabled_components=None,
                        use_cuda_kernels=True,
                    )
                )

    initial_raw = torch.from_numpy(inverse_reparameterize_torch(ITU_CONCRETE)).to(device)
    physical_person = args.person_solid_angle_weight == "physical"
    material = SharedHumanMaterial(initial_raw, 1.0 if physical_person else args.initial_area_scale).to(device)
    if physical_person:
        material.log_area_scale.requires_grad_(False)          # the slot is the geometry; no learned scale on top
    optimizer_groups = [
        {"params": [material.raw], "lr": args.material_lr},
        *([] if physical_person else [{"params": [material.log_area_scale], "lr": args.area_scale_lr}]),
        {"params": [material.radar_log_gain_delta], "lr": args.radar_balance_lr},
    ]
    if args.train_background_materials:
        optimizer_groups.append({
            "params": [background.raw_materials], "lr": args.background_material_lr,
        })
    optimizer = torch.optim.Adam(optimizer_groups)
    keep_range = torch.from_numpy(range_keep_mask()).to(device)
    measured_references = {
        name: torch.quantile(
            medians[name][keep_range], args.amplitude_reference_quantile,
        ).detach() for name in names}

    people_tensors = {
        frame: torch.from_numpy(pcl).to(device) for frame, pcl in people.items()
    }
    # One (frames, points, 6) tensor rather than a dict of per-frame clouds.
    # `assert_persistent_person_geometry` above already refused a cloud whose
    # length changes, so the stack always exists on the GVHMR geometry this
    # trainer requires; a batch is then one `index_select` instead of a
    # Python list comprehension and a `torch.cat` of eight tensors per
    # (step, radar).
    person_point_counts = {int(tensor.shape[0]) for tensor in people_tensors.values()}
    if len(person_point_counts) != 1:
        raise RuntimeError(
            "person clouds have inconsistent lengths after the persistence "
            f"check: {sorted(person_point_counts)}")
    person_stack = torch.stack([people_tensors[frame] for frame in frames])
    frame_row = {frame: row for row, frame in enumerate(frames)}

    # ------------------------------------------------------------------
    # The constant target side of the three per-frame losses.
    #
    # Everything the losses take from the measured side is an input, not a
    # parameter: the target map, its positive deviation from the temporal
    # median, the person ROI, the amplitude quantile threshold and the
    # measured reference amplitude are all fixed for the whole run. The inner
    # loop recomputed them once per (frame, radar) -- 3,600 times in a
    # 600-frame, 3-epoch pass -- and, worse, *selected* them with boolean
    # masks. Every boolean-mask index is a `nonzero`, and `nonzero` has a
    # data-dependent output shape, so each one is a host-device
    # synchronisation. Measured on pass B before this change: 192 `nonzero`
    # calls and 218 `cudaStreamSynchronize` per optimizer step, with the GPU
    # 15% busy over the pass.
    #
    # Hoisting them here and resolving each boolean mask to the integer
    # indices it selects leaves the arithmetic identical -- a bool-mask
    # select returns the same elements in the same row-major order as a
    # gather on its sorted flat indices -- and leaves the step with no
    # data-dependent shape at all.
    #
    # The ROI sizes are ragged (316-588 cells of 16,384 on sequence 38), so
    # the per-frame index lists are padded to the widest and carried with a
    # validity mask. Padding rows gather cell 0 and are multiplied by zero
    # before every reduction, which also zeroes their gradient, so a padded
    # row contributes neither value nor gradient.
    # ------------------------------------------------------------------
    keep_indices = torch.from_numpy(
        np.flatnonzero(range_keep_mask()).astype(np.int64)).to(device)
    minimum_amplitude_ratio = 10.0 ** (args.amplitude_floor_db / 20.0)

    def _to_db_rows(values: torch.Tensor) -> torch.Tensor:
        """`log_pearson_loss.to_db`, per row of a (rows, cells) batch."""
        peak = values.detach().amax(dim=1, keepdim=True).clamp_min(1e-20)
        floor = peak * (10.0 ** (-42.0 / 20.0))
        return 20.0 * torch.log10(values.clamp_min(floor) / peak)

    def _centre_db_rows(
        values: torch.Tensor, valid: torch.Tensor | None,
        count: torch.Tensor | None,
    ) -> torch.Tensor:
        """dB, mean-removed per row, with padded columns held at exactly 0."""
        if valid is None:
            centred = _to_db_rows(values)
            return centred - centred.mean(dim=1, keepdim=True)
        centred = _to_db_rows(values * valid) * valid
        return (centred
                - centred.sum(dim=1, keepdim=True) / count[:, None]) * valid

    def _pearson_rows(
        prediction: torch.Tensor, truth_centred: torch.Tensor,
        truth_square_sum: torch.Tensor, valid: torch.Tensor | None,
        count: torch.Tensor | None,
    ) -> torch.Tensor:
        """`log_pearson_loss` for a whole batch, one row per frame."""
        centred = _centre_db_rows(prediction, valid, count)
        denom = torch.sqrt(
            centred.square().sum(dim=1) * truth_square_sum).clamp_min(1e-20)
        return 1.0 - (centred * truth_centred).sum(dim=1) / denom

    def _pad_rows(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor,
                                                       torch.Tensor]:
        counts = torch.tensor([int(row.numel()) for row in values], device=device)
        width = max(1, int(counts.max()))
        occupied = (torch.arange(width, device=device)[None, :]
                    < counts[:, None])
        index = torch.zeros((len(values), width), dtype=torch.long, device=device)
        index[occupied] = torch.cat(values) if values else index.new_empty((0,))
        return index, occupied.to(dtype=torch.float32), counts.to(torch.float32)

    loss_targets: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for radar in names:
            median_map, raw_targets = medians[radar], targets[radar]
            stacked = torch.from_numpy(
                np.asarray(raw_targets)[frames].astype(np.float32)).to(device)
            full = stacked.index_select(1, keep_indices).reshape(len(frames), -1)
            full_centred = _centre_db_rows(full, None, None)
            dynamic = (stacked - median_map[None]).clamp_min(0.0).reshape(
                len(frames), -1)

            roi_rows = [
                person_ra_masks[frame][radar].reshape(-1).nonzero().reshape(-1)
                for frame in frames
            ]
            if any(int(row.numel()) == 0 for row in roi_rows):
                raise ValueError(
                    f"a {radar} person ROI is empty; the loss it supervises "
                    "would be undefined")
            roi_index, roi_valid, roi_count = _pad_rows(roi_rows)
            roi_target = dynamic.gather(1, roi_index)
            roi_centred = _centre_db_rows(roi_target, roi_valid, roi_count)

            # The amplitude ROI: the upper quantile of each frame's dynamic
            # residual. `torch.quantile` is taken per frame on exactly the
            # vector the inner loop handed it, so the threshold is the one
            # the eager path computed.
            reference = measured_references[radar]
            amplitude_rows: list[torch.Tensor] = []
            target_ratio_rows: list[torch.Tensor] = []
            for row, roi_cells in enumerate(roi_rows):
                values = dynamic[row].index_select(0, roi_cells)
                threshold = torch.quantile(values, args.amplitude_target_quantile)
                active = (values >= threshold) & (values > 0)
                selected = active.nonzero().reshape(-1)
                amplitude_rows.append(roi_cells.index_select(0, selected))
                target_ratio_rows.append(
                    (values.index_select(0, selected)
                     / reference.clamp_min(1e-20)).clamp_min(
                         minimum_amplitude_ratio))
            amplitude_index, amplitude_valid, amplitude_count = _pad_rows(
                amplitude_rows)
            target_ratio = torch.zeros_like(amplitude_valid)
            target_ratio[amplitude_valid > 0] = torch.cat(target_ratio_rows)
            # `fixed_reference_amplitude_loss` returns a hard zero when no cell
            # clears the threshold. Carrying that as a 0/1 row scale keeps the
            # value and the (zero) gradient without a host-side branch.
            amplitude_present = (amplitude_count > 0).to(torch.float32)
            loss_targets[radar] = {
                "full_centred": full_centred,
                "full_square_sum": full_centred.square().sum(dim=1),
                "roi_index": roi_index,
                "roi_valid": roi_valid,
                "roi_count": roi_count,
                "roi_centred": roi_centred,
                "roi_square_sum": roi_centred.square().sum(dim=1),
                "amplitude_index": amplitude_index,
                "amplitude_valid": amplitude_valid,
                "amplitude_count": amplitude_count.clamp_min(1.0),
                "amplitude_present": amplitude_present,
                "target_ratio": target_ratio,
                "target_ratio_square_mean": (
                    (target_ratio.square() * amplitude_valid).sum(dim=1)
                    / amplitude_count.clamp_min(1.0)).clamp_min(1e-30),
                "target_ratio_db": torch.where(
                    amplitude_valid > 0, 20.0 * torch.log10(
                        target_ratio.clamp_min(minimum_amplitude_ratio)),
                    torch.zeros_like(target_ratio)),
            }
            del stacked, dynamic, full
    # The masks and the per-frame targets are now folded into `loss_targets`;
    # nothing reads them again and together they are ~120 MB of device memory.
    person_ra_masks.clear()

    def _amplitude_rows(
        prediction: torch.Tensor, plan: dict[str, torch.Tensor],
        rows: torch.Tensor, prediction_reference: torch.Tensor,
    ) -> torch.Tensor:
        """`fixed_reference_amplitude_loss` for a whole batch, one row a frame."""
        valid = plan["amplitude_valid"].index_select(0, rows)
        count = plan["amplitude_count"].index_select(0, rows)
        ratio = (prediction / prediction_reference.detach().clamp_min(1e-20)
                 ).clamp_min(minimum_amplitude_ratio)
        if args.amplitude_loss_domain == "linear":
            residual = (ratio - plan["target_ratio"].index_select(0, rows)
                        ).square() * valid
            value = ((residual.sum(dim=1) / count)
                     / plan["target_ratio_square_mean"].index_select(0, rows))
        else:
            residual = (20.0 * torch.log10(ratio)
                        - plan["target_ratio_db"].index_select(0, rows)
                        ).abs() * valid
            value = (residual.sum(dim=1) / count) / abs(args.amplitude_floor_db)
        return value * plan["amplitude_present"].index_select(0, rows)

    def render_person(
        geometry: torch.Tensor, radar_name: str, rasterizer: Rasterizer, slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(geometry)
        return render_factorized(
            geometry[:, :3], geometry[:, 3:6],
            material.areas_for_points(count) * material.radar_gain(radar_name) * (1.0 if slot is None else slot),
            material.for_points(count), rasterizer, reparameterize_torch,
            detach_phase=True, shadow_mask=None, bsdf_mode="full",
            disabled_components=None, use_cuda_kernels=True,
        )

    frame_of_point_cache: dict[tuple[int, int], torch.Tensor] = {}

    def render_person_frames(
        clouds: "list[torch.Tensor] | torch.Tensor", radar_name: str,
        rasterizer: Rasterizer, slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every frame of a batch in ONE render call.

        The person material is a shared six-vector broadcast to every point
        (`for_points` is an `expand`), so a frame's points do not care which
        frame they came from: the clouds concatenate, and the only thing that
        has to stay separate is where each path's splat lands.
        `return_splat_components=True` hands back the per-path contributions
        and their flat destination indices before the coherent sum, so
        offsetting each path by its frame's slice of the output buffer routes
        them without touching the arithmetic that produced them.

        Ragged clouds are fine -- the frame of a path is looked up from its
        point index rather than divided out -- so a list of per-frame clouds
        of differing length still works. A ``(frames, points, 6)`` tensor is
        the persistent-geometry case: the flat cloud is then a reshape rather
        than a concatenation and the frame-of-point map is a constant, cached
        on its shape, instead of a fresh `torch.tensor(counts)` host-to-device
        copy on every call.

        Not bitwise against the per-frame calls: one scatter over F frames'
        paths orders its atomicAdds differently from F scatters. Per frame it
        is the same set of contributions into the same bins.
        """
        if isinstance(clouds, torch.Tensor):
            batch, points = int(clouds.shape[0]), int(clouds.shape[1])
            flat_cloud = clouds.reshape(batch * points, clouds.shape[2])
            total = batch * points
            key = (batch, points)
            frame_of_point = frame_of_point_cache.get(key)
            if frame_of_point is None:
                frame_of_point = torch.arange(
                    batch, device=flat_cloud.device
                ).repeat_interleave(points)
                frame_of_point_cache[key] = frame_of_point
        else:
            batch = len(clouds)
            counts = [len(cloud) for cloud in clouds]
            total = sum(counts)
            flat_cloud = torch.cat(clouds, dim=0)
            frame_of_point = torch.repeat_interleave(
                torch.arange(batch, device=flat_cloud.device),
                torch.tensor(counts, device=flat_cloud.device),
            )
        splats = render_factorized(
            flat_cloud[:, :3], flat_cloud[:, 3:6],
            material.areas_for_points(total) * material.radar_gain(radar_name) * (1.0 if slot is None else slot.reshape(-1)),
            material.for_points(total), rasterizer, reparameterize_torch,
            detach_phase=True, shadow_mask=None, bsdf_mode="full",
            disabled_components=None, use_cuda_kernels=True,
            return_splat_components=True,
        )
        n_tx, n_rx, n_range = splats["output_shape"]
        stride = n_tx * n_rx * n_range
        unit = frame_of_point[splats["point_indices"]]
        destinations = splats["flat_indices"] + unit[None, :] * stride
        real, imag = splat_scatter(
            splats["contrib_real"].reshape(-1),
            splats["contrib_imag"].reshape(-1),
            destinations.reshape(-1),
            (batch * n_tx, n_rx, n_range),
        )
        shape = (batch, n_tx, n_rx, n_range)
        return real.reshape(shape), imag.reshape(shape)

    history: list[dict] = []
    best_epoch_loss = float("inf")
    best_raw = material.raw.detach().clone()
    best_log_area_scale = material.log_area_scale.detach().clone()
    best_radar_log_gain_delta = material.radar_log_gain_delta.detach().clone()
    best_background_materials = background.raw_materials.detach().clone()
    optimizer_steps = 0

    def clip_material_gradients() -> None:
        """`rms_clip_grad` for the three material parameters, in place.

        3DPS's `rms_clip_grad` (train_gaussian.py) finishes with
        ``param.grad.data = g``, where ``g`` is the tensor ``nan_to_num``
        allocated -- so it silently rebinds ``.grad`` to a NEW buffer on
        every call. A captured step accumulates into the address it was
        captured with, so after one rebinding the graph writes a buffer
        nothing reads and the optimizer steps on a buffer nothing writes:
        training stops after the first step and coasts on Adam's momentum.
        That passes a first-step gate and shows up three epochs later as a
        visibly different checkpoint, which is how it was found -- the
        learned radar balance stopped at -2.4 dB instead of reaching the
        -6 dB limit, 130-220x the reference's own five-run spread.

        Same three expressions, same order, writing through the same
        storage: ``nan_to_num_`` is the in-place form of the same
        elementwise map, and the clip branch and its scale are unchanged.
        """
        for parameter in (material.raw, material.log_area_scale,
                          material.radar_log_gain_delta):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            gradient.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            rms = torch.sqrt(torch.mean(gradient ** 2))
            if rms > args.grad_clip:
                gradient.mul_(args.grad_clip / rms)

    def material_gradient_vector() -> torch.Tensor:
        values = []
        for parameter in (
            material.raw, material.log_area_scale, material.radar_log_gain_delta,
        ):
            if parameter.grad is None:
                values.append(torch.zeros_like(parameter).flatten())
            else:
                values.append(parameter.grad.detach().flatten().clone())
        return torch.cat(values)

    # Loop invariants. When the background is frozen -- which is every run
    # that does not pass --train-background-materials, i.e. the shipped one --
    # `render_background` returns the pair cached above, so the background RA
    # map and the amplitude reference quantile taken from it are constant for
    # the whole run. The inner loop recomputed both once per (frame, radar):
    # 3,600 times each in a 600-frame, 3-epoch pass.
    frozen_background = not args.train_background_materials
    constant_background_map: dict[str, torch.Tensor] = {}
    constant_amplitude_reference: dict[str, torch.Tensor] = {}
    if frozen_background:
        with torch.no_grad():
            for name in names:
                profiles = background_profile_cache[name]
                constant_background_map[name] = range_profiles_to_ra_magnitude(
                    *profiles)
                constant_amplitude_reference[name] = torch.quantile(
                    constant_background_map[name][keep_range].detach(),
                    args.amplitude_reference_quantile,
                )

    # ------------------------------------------------------------------
    # The training step, at a shape that never changes.
    #
    # `batch_rows` is the step's only per-step input and it is a *device*
    # buffer: everything the step needs -- the person clouds, the ROI
    # indices, the constant target side -- is a fixed table gathered by those
    # rows inside the step, so the host does nothing per step but write eight
    # integers into `batch_rows`. That is what makes the step capturable.
    #
    # `batch_size` is a Python int rather than a shape read off `batch_rows`
    # so a captured step keeps the shape it was captured at; the epoch's
    # short last batch (there is none at 600 frames and `--frame-batch-size
    # 8`, but `--max-frames` can make one) calls this with its own size and
    # runs eagerly.
    # ------------------------------------------------------------------
    batch_rows = torch.zeros(args.frame_batch_size, dtype=torch.long,
                             device=device)

    def training_step(rows: torch.Tensor, batch_size: int,
                      batch_scale: float) -> torch.Tensor:
        """Forward, backward and the step's recorded scalars: a (frames, 6 x radars).

        One render and one backward per (batch, radar), not per (frame,
        radar). The batch's clouds go through `render_person_frames` in a
        single `render_factorized` call and the per-frame losses are summed
        into one objective per radar, so a 600-frame epoch issues 150 renders
        and 150 backwards instead of 1,200 of each.

        The two radars stay separate backwards deliberately: the gradient
        guard downstream measures each one's contribution, and a single fused
        backward would lose the attribution entirely. What it does lose is
        the PER-FRAME attribution -- `grad_h` and `grad_v` are a batch's
        norms, recorded against each of its frames. Nothing outside this
        module reads those fields.
        """
        grad_marks = [material_gradient_vector()]
        radar_terms: dict[str, tuple[torch.Tensor, ...]] = {}
        for radar in names:
            rasterizer, reference = rasterizers[radar], measured_references[radar]
            person_real, person_imag = render_person_frames(
                person_stack.index_select(0, rows), radar, rasterizer,
                None if radar not in person_slot else person_slot[radar].index_select(0, rows),
            )
            # NOT `background`: that name is the scene's PointPrimitives
            # in the enclosing scope, and shadowing it here made
            # `best_background_materials` read a tuple.
            background_profiles = render_background(radar, rasterizer)
            person_maps = range_profiles_to_ra_magnitude_frames(
                person_real, person_imag)
            pred_maps = range_profiles_to_ra_magnitude_frames(
                background_profiles[0][None] + person_real,
                background_profiles[1][None] + person_imag,
            )
            amplitude_reference = (
                constant_amplitude_reference[radar] if frozen_background
                else torch.quantile(
                    range_profiles_to_ra_magnitude(*background_profiles)[
                        keep_range].detach(),
                    args.amplitude_reference_quantile))
            # The whole batch's three losses in one set of reductions,
            # against the precomputed constant target side. Same
            # quantities as the per-frame calls, one row a frame.
            plan = loss_targets[radar]
            loss_full = _pearson_rows(
                pred_maps.index_select(1, keep_indices).reshape(
                    batch_size, -1),
                plan["full_centred"].index_select(0, rows),
                plan["full_square_sum"].index_select(0, rows), None, None)
            person_cells = person_maps.reshape(batch_size, -1)
            loss_correlation = _pearson_rows(
                person_cells.gather(
                    1, plan["roi_index"].index_select(0, rows)),
                plan["roi_centred"].index_select(0, rows),
                plan["roi_square_sum"].index_select(0, rows),
                plan["roi_valid"].index_select(0, rows),
                plan["roi_count"].index_select(0, rows))
            loss_amplitude = _amplitude_rows(
                person_cells.gather(
                    1, plan["amplitude_index"].index_select(0, rows)),
                plan, rows, amplitude_reference)
            loss_dynamic = (
                (1.0 - args.amplitude_weight) * loss_correlation
                + args.amplitude_weight * loss_amplitude
            )
            loss = ((1.0 - args.dynamic_weight) * loss_full
                    + args.dynamic_weight * loss_dynamic)
            radar_terms[radar] = (loss.detach(), loss_full.detach(),
                                  loss_dynamic.detach(),
                                  loss_correlation.detach(),
                                  loss_amplitude.detach())
            (batch_scale * loss.sum()).backward()
            grad_marks.append(material_gradient_vector())

        # Columns: one gradient norm per radar, then the five loss terms
        # term-major, radars interleaved -- (frames, 6 x radars).
        grad_norms = [torch.linalg.vector_norm(grad_marks[i + 1] - grad_marks[i])
                      for i in range(len(names))]
        return torch.stack(
            [norm.expand(batch_size) for norm in grad_norms]
            + [term for terms in zip(*(radar_terms[radar] for radar in names))
               for term in terms],
            dim=1)

    # ------------------------------------------------------------------
    # Capture the step's forward and backward in one CUDA graph.
    #
    # The step is host-bound, and by a lot: measured on pass B of sequence 38
    # after the loss-side work above, ~1,900 device launches issued over
    # ~35 ms of wall clock against ~8 ms of GPU. Neither syncs nor batching
    # are left to remove -- the step's synchronisation count is 4, all of
    # them after the backward -- so what remains is launch issue cost, and a
    # graph replaces it with one graph launch.
    #
    # **Forward and backward are captured; the optimizer is not**, and that
    # is a correctness decision rather than an oversight. `torch.optim.Adam`
    # has two arithmetic paths, and `capturable=True` -- which capture would
    # force, because the bias corrections can no longer be Python floats --
    # folds the step size into the denominator with an extra division. The
    # two agree in exact arithmetic and disagree in the last bit, on every
    # parameter, on every step. Three scalars' worth of optimizer is a few
    # dozen launches; it stays outside, eager and exact. This is the same
    # split `velocity_field.py` made for the same reason.
    #
    # What the capture requires, and what provides it:
    #   * static shapes            -- the GVHMR person geometry is 6,890
    #     points in every frame (`assert_persistent_person_geometry` refuses
    #     anything else), and the padded ROI tables above are rectangular, so
    #     nothing in the step is ragged;
    #   * static input buffers     -- `batch_rows`; every other per-step
    #     quantity is gathered from a fixed table inside the step;
    #   * no CPU<->GPU sync inside -- measured 0 in the captured region
    #     (`verify.py verify-person-material --count-syncs`);
    #   * stable gradient buffers  -- `zero_grad(set_to_none=False)`, so the
    #     graph accumulates into the same addresses on every replay. Starting
    #     a step from a zeroed buffer instead of from `None` is exact: the
    #     accumulation adds the step's gradient to 0.0, and every parameter
    #     here receives a gradient on every step anyway.
    # ------------------------------------------------------------------
    # Capture is for the full-size batch on CUDA only. `--cuda-graph off`
    # runs the identical step eagerly and is the arbiter the gate compares
    # against.
    # `--train-background-materials` is excluded deliberately rather than
    # left to chance: it puts a fourth parameter in the optimizer whose
    # gradient buffer the capture would have to hold stable too, and it
    # un-freezes the background render inside the step. It is not the
    # shipped configuration and it has not been measured under capture.
    capture_step = (args.cuda_graph == "on" and device.type == "cuda"
                    and not args.train_background_materials)
    graph_state: dict = {"graph": None, "outputs": None, "index_cache": None,
                         "capture_s": 0.0}

    def zero_graph_gradients() -> None:
        for parameter in (material.raw, material.log_area_scale,
                          material.radar_log_gain_delta):
            if parameter.grad is not None:
                parameter.grad.zero_()

    def capture_training_step(batch_scale: float) -> None:
        """Warm up on a side stream, then capture one step."""
        capture_started = time.perf_counter()
        torch.cuda.synchronize()
        # Two eager iterations on the default stream first: they create the
        # gradient buffers at stable addresses, build the compiled step-5
        # kernels and the renderer's channel-index cache, and leave the
        # autograd graph the capture must find ready. Three more on a side
        # stream, as capture requires.
        for _ in range(2):
            training_step(batch_rows, args.frame_batch_size, batch_scale)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                zero_graph_gradients()
                training_step(batch_rows, args.frame_batch_size, batch_scale)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        zero_graph_gradients()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = training_step(batch_rows, args.frame_batch_size,
                                    batch_scale)
        torch.cuda.synchronize()
        zero_graph_gradients()
        graph_state.update({
            "graph": graph, "outputs": outputs,
            # `render_factorized` caches its three `(M x n_tx x n_rx,)`
            # channel index tensors on the function, keyed by
            # `(M, n_tx, n_rx, device)`, and *replaces the whole dict* on a
            # miss. The capture bakes in pointers to the training-M entry;
            # the representative render at the end of this function uses a
            # different M, misses, and would drop the last reference to it.
            # Holding the dict keeps those tensors alive at the addresses the
            # graph recorded, for as long as the graph.
            "index_cache": getattr(render_factorized, "_idx_cache", None),
            "capture_s": time.perf_counter() - capture_started,
        })
        print(f"captured person training step: {args.frame_batch_size} frames "
              f"x {person_stack.shape[1]} points per radar, "
              f"{graph_state['capture_s']:.2f} s", flush=True)

    started = time.time()
    for epoch in range(args.epochs):
        epoch_frames = frames.copy()
        random.Random(args.seed + epoch).shuffle(epoch_frames)
        # The shuffle is host-side and stays there; what the step needs is the
        # row each frame occupies in the precomputed tables, uploaded once for
        # the epoch rather than once per batch.
        epoch_rows = torch.tensor(
            [frame_row[frame] for frame in epoch_frames], device=device)
        epoch_loss = 0.0
        for batch_start in range(0, len(epoch_frames), args.frame_batch_size):
            batch_frames = epoch_frames[batch_start:batch_start + args.frame_batch_size]
            rows = epoch_rows[batch_start:batch_start + args.frame_batch_size]
            batch_scale = 0.5 / len(batch_frames)
            # `set_to_none=False`: the captured step accumulates into fixed
            # gradient addresses, so the buffers must survive the step. Every
            # parameter here receives a gradient on every step, so starting
            # from a zeroed buffer rather than from `None` reaches Adam with
            # exactly the same numbers.
            optimizer.zero_grad(set_to_none=False)
            if (graph_state["graph"] is None and capture_step
                    and len(batch_frames) == args.frame_batch_size):
                batch_rows.copy_(rows)
                capture_training_step(batch_scale)
            if (graph_state["graph"] is not None
                    and len(batch_frames) == args.frame_batch_size):
                # Forward and backward are one graph launch. The outputs are
                # the captured tensors, refreshed in place by the replay, and
                # are read out below by a `.cpu()` that copies out of them
                # before the next replay overwrites them.
                batch_rows.copy_(rows)
                graph_state["graph"].replay()
                block_terms = graph_state["outputs"]
            else:
                block_terms = training_step(rows, len(batch_frames), batch_scale)

            batch_records = [
                {"epoch": epoch, "step": batch_start + batch_offset,
                 "batch_step": optimizer_steps, "frame": frame}
                for batch_offset, frame in enumerate(batch_frames)
            ]
            # One transfer for the batch: (frames, 6 x radars) instead of
            # that many syncs a frame. Still ahead of the gradient guard and
            # `optimizer.step`, so a batch with a missing radar gradient
            # raises before it is applied, exactly as it did.
            block = block_terms.cpu().numpy()
            n = len(names)
            for offset, record in enumerate(batch_records):
                values = block[offset]
                grads = {name: float(values[i]) for i, name in enumerate(names)}
                if not all(g > 0.0 for g in grads.values()):
                    raise RuntimeError(
                        f"missing radar gradient at epoch={record['epoch']} "
                        f"frame={record['frame']}"
                    )
                term = lambda t, i: float(values[n + t * n + i])
                total_loss = sum(term(0, i) for i in range(n)) / n
                epoch_loss += total_loss
                for i, name in enumerate(names):
                    record.update({
                        f"loss_{name}": term(0, i),
                        f"loss_{name}_full": term(1, i),
                        f"loss_{name}_dynamic": term(2, i),
                        f"loss_{name}_dynamic_correlation": term(3, i),
                        f"loss_{name}_dynamic_amplitude": term(4, i),
                        f"grad_{name}": grads[name],
                    })
                record["loss"] = total_loss
                history.append(record)
            clip_material_gradients()
            if args.train_background_materials:
                rms_clip_grad(background.raw_materials, args.grad_clip)
            optimizer.step()
            optimizer_steps += 1
            with torch.no_grad():
                material.log_area_scale.clamp_(np.log(args.min_area_scale), np.log(1.0))
                max_delta = args.max_radar_balance_db * np.log(10.0) / 40.0
                material.radar_log_gain_delta.clamp_(-max_delta, max_delta)
            step = batch_start + len(batch_frames) - 1
            frame = batch_frames[-1]
            if step % args.progress_every == 0 or step == len(epoch_frames) - 1:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(epoch_frames)} "
                    f"frame={frame} loss={total_loss:.5f} "
                    + " ".join(f"grad_{name}={grads[name]:.2e}" for name in names), flush=True,
                )
        mean_epoch_loss = epoch_loss / len(epoch_frames)
        if mean_epoch_loss < best_epoch_loss:
            best_epoch_loss = mean_epoch_loss
            best_raw = material.raw.detach().clone()
            best_log_area_scale = material.log_area_scale.detach().clone()
            best_radar_log_gain_delta = material.radar_log_gain_delta.detach().clone()
            best_background_materials = background.raw_materials.detach().clone()

    with torch.no_grad():
        material.raw.copy_(best_raw)
        material.log_area_scale.copy_(best_log_area_scale)
        material.radar_log_gain_delta.copy_(best_radar_log_gain_delta)
        background.raw_materials.copy_(best_background_materials)
    physical = reparameterize_torch(material.raw).detach().cpu().numpy()
    # ------------------------------------------------------------------
    # The level calibration (docs/STATIC_SEQS_NEXT.md step 1): the person's rendered level is what the downstream model
    # reads, and the fit above does not anchor it (its amplitude term is a dB ratio to a reference quantile with a floor;
    # measured, the fitted person peaks at 0.3-0.6x the recording's and the velocity stage lowers it further). Here the
    # peak |RA| over the person's mask, render against measurement, is matched exactly: per radar the amplitude factor k
    # with median_frames[max_mask |B + k P| / max_mask |T|] = 1 (B the fixed scene, P the person, both complex range
    # profiles at the fitted material; T the measured first-loop map), by bisection on log k -- the ratio is monotone in
    # k once the person dominates its own mask. Label-free: the mask is the mesh's.
    # ------------------------------------------------------------------
    level_calibration: dict[str, dict] = {}
    if args.level_calibration == "peak":
        with torch.no_grad():
            ks, gains = {}, {}
            for name, rasterizer in rasterizers.items():
                bg = render_background(name, rasterizer)
                B_re, B_im = bg[0][None], bg[1][None]
                parts_re, parts_im = [], []
                for start in range(0, len(frames), args.frame_batch_size):
                    rows = torch.arange(start, min(start + args.frame_batch_size, len(frames)), device=device)
                    pr, pi = render_person_frames(person_stack.index_select(0, rows), name, rasterizer,
                                                  None if name not in person_slot else person_slot[name].index_select(0, rows))
                    parts_re.append(pr.reshape(len(rows), *bg[0].shape)); parts_im.append(pi.reshape(len(rows), *bg[1].shape))
                P_re, P_im = torch.cat(parts_re), torch.cat(parts_im)
                # the person mask's cells per frame, as the loss kept them (flat indices, padded, a validity mask)
                roi_index, roi_valid = loss_targets[name]["roi_index"], loss_targets[name]["roi_valid"] > 0
                T = torch.from_numpy(np.asarray(targets[name])[frames].astype(np.float32)).to(device)        # measured first-loop |RA|
                masked_peak = lambda maps: torch.where(roi_valid, maps.flatten(1).gather(1, roi_index), torch.zeros_like(roi_index, dtype=maps.dtype)).max(dim=1).values
                t_peak = masked_peak(T).clamp_min(1e-30)
                valid = roi_valid.any(dim=1)
                # the receiver gain, from the scene: the velocity stage's own init recipe (measured / rendered level of
                # the whole map at gain 1), the fixed scene alone, median over frames -- one constant per radar
                B_map = range_profiles_to_ra_magnitude(*bg)[keep_range]
                g = float(torch.sqrt(T[:, keep_range].square().flatten(1).sum(dim=1) / B_map.square().sum().clamp_min(1e-30)).median())
                gains[name] = g
                def peak_ratio(log_k: float) -> float:
                    k = float(np.exp(log_k))
                    pred = g * range_profiles_to_ra_magnitude_frames(B_re + k * P_re, B_im + k * P_im)
                    return float((masked_peak(pred) / t_peak)[valid].median())
                lo, hi = float(np.log(1 / 64)), float(np.log(64)); r0 = peak_ratio(0.0)
                floor_ratio = peak_ratio(lo)
                if floor_ratio >= 1.0:
                    # the scene alone exceeds the measured peak inside the person's mask: a ghost of the person in the
                    # scaffold (its own LiDAR / depth points at the start position) or a wall the background fit made too
                    # bright there; no person amplitude can match, so the fit's level is kept and the case is recorded
                    ks[name] = 1.0
                    print(f"level calibration ({name}): the scene alone is {floor_ratio:.2f}x the measured peak in the person mask "
                          f"(receiver gain {g:.3f}); the person's level cannot be set here -- kept at the fit (scaffold ghost?)", flush=True)
                    level_calibration[name] = {"absolute_gain": g, "k_amplitude": 1.0, "peak_ratio_at_fit": r0, "peak_ratio_after": r0,
                                               "scene_alone_ratio": floor_ratio, "frames": int(valid.sum()), "status": "scene-dominated mask"}
                    continue
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if peak_ratio(mid) < 1.0: lo = mid
                    else: hi = mid
                ks[name] = float(np.exp(0.5 * (lo + hi)))
                level_calibration[name] = {"absolute_gain": g, "k_amplitude": ks[name], "peak_ratio_at_fit": r0, "peak_ratio_after": peak_ratio(float(np.log(ks[name]))),
                                           "frames": int(valid.sum())}
                print(f"level calibration ({name}): receiver gain from the scene {g:.3f}; person mask peak render/measured {r0:.3f} at the fit "
                      f"under it -> x{ks[name]:.3f} in amplitude -> {level_calibration[name]['peak_ratio_after']:.3f} (median over {int(valid.sum())} frames)", flush=True)
                del P_re, P_im
            k_shared = float(np.exp(np.mean([np.log(k) for k in ks.values()])))
            material.log_area_scale.add_(2.0 * float(np.log(k_shared)))                          # amplitude ~ sqrt(area)
            if len(names) == 2:
                material.radar_log_gain_delta.add_(0.5 * float(np.log(ks[names[0]] / ks[names[1]])))
            level_calibration["shared_k_amplitude"] = k_shared
            level_calibration["area_scale"] = float(material.log_area_scale.exp())
            print(f"level calibration: area scale -> {level_calibration['area_scale']:.4g}" + (f", radar delta -> {float(material.radar_log_gain_delta):+.4f}" if len(names) == 2 else ""), flush=True)
            background_profile_cache.clear()

    checkpoint = {
        "shared_human_raw_material": material.raw.detach().cpu(),
        "shared_human_physical_material": torch.from_numpy(physical),
        "shared_human_area_scale": material.log_area_scale.exp().detach().cpu(),
        "person_radar_log_gain_delta": material.radar_log_gain_delta.detach().cpu(),
        "person_radar_amplitude_gains": {
            name: material.radar_gain(name).detach().cpu()
            for name in names
        },
        "level_calibration": level_calibration,
        "absolute_gain": {name: level_calibration[name]["absolute_gain"] for name in names} if level_calibration else None,
        "geometry_translation_y_m": args.geometry_translation_y_m,
        "geometry_depth_scale": args.geometry_depth_scale,
        "near_field_guard_m": HUPR_NEAR_FIELD_M,
        "background_model_initialization": str(args.background_model.resolve()),
        "shared_background_state": {
            key: value.detach().cpu() for key, value in background.state_dict().items()
        },
    }
    torch.save(checkpoint, args.output_dir / "best_shared_human_material.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    # Render representative frames for figures/video validation.
    representative = sorted(set([frames[0], frames[len(frames) // 4],
                                 frames[len(frames) // 2], frames[3 * len(frames) // 4],
                                 frames[-1]]))
    evaluation: dict[str, dict] = {}
    with torch.no_grad():
        for frame in representative:
            evaluation[str(frame)] = {}
            for name, rasterizer in rasterizers.items():
                radar_targets = targets[name]
                person_profile = render_person(people_tensors[frame], name, rasterizer,
                                               None if name not in person_slot else person_slot[name][frame_row[frame]])
                background_profile = render_background(name, rasterizer)
                background_map = range_profiles_to_ra_magnitude(*background_profile)
                prediction = range_profiles_to_ra_magnitude(
                    background_profile[0] + person_profile[0],
                    background_profile[1] + person_profile[1],
                )
                target = torch.from_numpy(np.array(radar_targets[frame])).to(device)
                correlation = 1.0 - float(log_pearson_loss(prediction, target))
                background_correlation = 1.0 - float(
                    log_pearson_loss(background_map, target)
                )
                np.save(
                    args.output_dir / f"frame_{frame:03d}_{name}_prediction.npy",
                    prediction.cpu().numpy(),
                )
                evaluation[str(frame)][name] = {
                    "correlation": correlation,
                    "background_only_correlation": background_correlation,
                }

    result = {
        "status": "completed",
        "frames_seen_each_epoch": len(frames),
        "epochs": args.epochs,
        "optimizer_steps": optimizer_steps,
        "frame_batch_size": args.frame_batch_size,
        "static_background_cached": not args.train_background_materials,
        "training_data_preloaded_to_gpu": True,
        "cuda_graph": args.cuda_graph,
        # Whether the step actually ran as a replayed graph, not just whether
        # it was asked to: capture is skipped on CPU, with trainable
        # background materials, and when no batch reaches --frame-batch-size.
        "cuda_graph_captured": graph_state["graph"] is not None,
        "cuda_graph_capture_seconds": graph_state["capture_s"],
        "best_mean_epoch_loss": best_epoch_loss,
        "elapsed_seconds": time.time() - started,
        "shared_material_vectors": 1,
        "shared_material_raw": material.raw.detach().cpu().tolist(),
        "shared_material_physical_columns": [
            "eps_real", "eps_imag", "sigma_h_m", "correlation_length_m",
            "coherent_fraction", "thickness_m",
        ],
        "shared_material_physical": physical.tolist(),
        "shared_human_area_scale": float(material.log_area_scale.exp().detach()),
        "person_radar_amplitude_gains": {
            name: float(material.radar_gain(name).detach())
            for name in names
        },
        "person_radar_gain_ratio_db": float(
            20.0 * torch.log10(
                material.radar_gain(names[0]) / material.radar_gain(names[1])
            ).detach()
        ) if len(names) == 2 else None,
        "radar_gain_constraint": (
            f"{names[0]}=exp(delta), {names[1]}=exp(-delta); geometric mean is exactly 1"
            if len(names) == 2 else "single radar: gain 1"
        ),
        "minimum_human_area_scale": args.min_area_scale,
        "shared_background_materials": bool(args.train_background_materials),
        "geometry_translation_y_m": args.geometry_translation_y_m,
        "geometry_depth_scale": args.geometry_depth_scale,
        "excluded_frames": sorted(excluded_frames),
        "near_field_guard_m": HUPR_NEAR_FIELD_M,
        "near_field_applied_symmetrically_to_measured_and_rendered_loss": True,
        "dynamic_residual_weight": args.dynamic_weight,
        "dynamic_amplitude_weight": args.amplitude_weight,
        "dynamic_amplitude_floor_db": args.amplitude_floor_db,
        "dynamic_amplitude_target_quantile": args.amplitude_target_quantile,
        "amplitude_reference_quantile": args.amplitude_reference_quantile,
        "dynamic_supervision": (
            "positive deviation from the 600-frame temporal median inside the metric "
            "person range-angle ROI; correlation supervises spatial shape and fixed "
            "scene-relative log-amplitude supervises person brightness"
        ),
        "every_step_has_both_radar_gradients": all(
            all(row[f"grad_{name}"] > 0 for name in names) for row in history
        ),
        "material_consistency_proof": (
            "one human Parameter of shape (6,) was expanded to every person point; "
            "the background model is also shared across all frames; no frame-specific "
            "material parameters exist"
        ),
        "representative_evaluation": evaluation,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sequence_dir(parser)
    parser.add_argument(
        "--background-model", type=Path,
        default=WORKSPACE / "training/dual_radar_gpu_full/best_model.pt",
    )
    parser.add_argument(
        "--static-pcl", type=Path,
        default=WORKSPACE / "multiview_scaffold/pcl.npy",
    )
    radar_configs(parser)
    parser.add_argument("--mm3dps-root", type=Path, default=DEFAULT_MM3DPS_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=WORKSPACE / "training/person_sequence_1_first_loop_shared",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--frame-batch-size", type=int, default=8,
                        help="frames accumulated before each optimizer update")
    parser.add_argument(
        "--cuda-graph", choices=("on", "off"), default="on",
        help="capture the step's forward and backward in one CUDA graph and "
             "replay it. The step is host-bound -- ~1,900 device launches "
             "over ~35 ms of wall clock against ~8 ms of GPU -- and the "
             "graph replaces the issue cost with one launch. `off` runs the "
             "identical step eagerly and is the arbiter the equivalence gate "
             "compares against (`verify.py verify-person-material "
             "--candidate-trainer-arg`).")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--exclude-frames", default="",
                        help="comma-separated zero-based frames excluded from training")
    parser.add_argument("--geometry-translation-y-m", type=float, default=-0.9)
    parser.add_argument("--geometry-depth-scale", type=float, default=1.0)
    parser.add_argument("--material-lr", type=float, default=2e-3)
    parser.add_argument("--background-material-lr", type=float, default=5e-4)
    background_group = parser.add_mutually_exclusive_group()
    background_group.add_argument(
        "--train-background-materials", dest="train_background_materials",
        action="store_true",
        help="Jointly fine-tune background materials (default keeps them fixed).",
    )
    background_group.add_argument(
        "--freeze-background-materials", dest="train_background_materials",
        action="store_false",
        help="Explicitly keep the initialization background fixed.",
    )
    parser.set_defaults(train_background_materials=False)
    parser.add_argument("--person-solid-angle-weight", choices=("off", "physical"), default="off",
                        help="physical (Option B): the person's areas slot is A_i |cos|/d^2 from the posed mesh's vertex "
                             "areas and normals in the scene's unit, computed per frame, not learned; the area scale is then 1 and untrained")
    parser.add_argument("--scene-solid-angle", choices=("off", "physical"), default="off",
                        help="physical: the fixed scene's areas slot is A_i |cos|/d^2 in its median unit instead of 1.0")
    parser.add_argument("--level-calibration", choices=("off", "peak"), default="peak",
                        help="peak (default since 2026-09-20, docs/STATIC_SEQS_NEXT.md step 1): after the fit, the person's "
                             "amplitude factor k that makes the person mask's PEAK |RA| of the render (person + fixed scene) "
                             "match the measured first-loop |RA|'s, median over the frames, solved by bisection; the area "
                             "scale carries k^2 (two radars: the geometric mean, the reciprocal H/V delta the ratio). The "
                             "velocity stage then keeps it frozen. off: the fit's own level")
    parser.add_argument("--initial-area-scale", type=float, default=0.02)
    parser.add_argument("--min-area-scale", type=float, default=0.002)
    parser.add_argument("--area-scale-lr", type=float, default=2e-2)
    parser.add_argument("--radar-balance-lr", type=float, default=1e-2)
    parser.add_argument("--max-radar-balance-db", type=float, default=6.0,
                        help="maximum learned horizontal/vertical person gain ratio")
    parser.add_argument("--dynamic-weight", type=float, default=0.35)
    parser.add_argument("--amplitude-weight", type=float, default=0.5)
    parser.add_argument(
        "--amplitude-loss-domain", choices=("linear", "db"), default="db",
        help="what the person amplitude ROI loss compares in. `linear` is a "
             "normalized L2 on linear magnitude ratios, the form every other "
             "loss here uses; `db` is an L1 on 20*log10 of the same ratios and "
             "is what every checkpoint before 2026-09-03 was fitted with. See "
             "measured in the person-loss-domain A/B.")
    parser.add_argument("--amplitude-floor-db", type=float, default=-60.0,
                        help="only used by --amplitude-loss-domain db")
    parser.add_argument("--amplitude-target-quantile", type=float, default=0.75)
    parser.add_argument("--amplitude-reference-quantile", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _entry_person_sequence():
        print(json.dumps(train(make_parser().parse_args()), indent=2))

# ----------------------------------------------------------------------
# smpl_fit: Fit SMPL to a HuPR sequence: 14 joints, the person cloud, and time.
# ----------------------------------------------------------------------


DEFAULT_JOINT_WEIGHT = 1.0

DEFAULT_SURFACE_WEIGHT = 0.35

DEFAULT_SHAPE_WEIGHT = 0.001

DEFAULT_POSE_WEIGHT = 0.0015

DEFAULT_SMOOTH_POSE_WEIGHT = 0.6

DEFAULT_SMOOTH_TRANSLATION_WEIGHT = 2.0

DEFAULT_HUBER_DELTA_M = 0.08

SURFACE_CLOUD_SAMPLES = 700

SURFACE_MESH_SAMPLES = 1400









HUPR_LEFT_SHOULDER, HUPR_RIGHT_SHOULDER = 8, 11

HUPR_LEFT_HIP, HUPR_RIGHT_HIP = 3, 0












COMMANDS = {
    "person-sequence": _entry_person_sequence,
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
