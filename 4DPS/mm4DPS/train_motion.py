"""The scene material fit (dual-radar) and the absolute-gain diagnostic.

    python -m mm4DPS.train_motion {dual-radar,absolute-gain}

The joint-gain inheritance that lived here (a seed checkpoint's 14 joint
scattering gains passed through to the velocity trainer) is gone: the
trainer learns the joint gains (docs/SEEDS.md).

Four ROOT constants went with the merge -- every use read
`ROOT / WORKSPACE / "..."`, and WORKSPACE is absolute, so pathlib discarded the
left operand and the prefix was already a no-op.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import shutil
import time
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from mm4DPS import HUPR_NEAR_FIELD_M, MM3DPS_ROOT, WORKSPACE, Dataset, array_axis, validate_configs
from mm4DPS import dispatch, radar_configs
from mm4DPS.forward_render import log_pearson_loss, range_profiles_to_ra_magnitude












def choose_motion_frames(
    velocities: np.ndarray, count: int, edge_guard: int = 5,
) -> list[int]:
    """Choose motion-rich frames by the 90th-percentile joint speed.

    ``edge_guard`` drops that many frames from each end. It was hard-coded at
    5, which is where "590 training frames" came from on a 600-frame
    sequence -- not, as is sometimes assumed, a held-out calibration split.
    Nothing is calibrated on those ten frames; they were simply never offered
    to the trainer.

    The guard is a convention rather than a requirement. The shipped
    differentiator is ``savgol_filter(..., mode="interp")``, which fits its
    polynomial to the edge window and therefore *does* return a velocity at
    every frame; ``central``/``fourth-order`` use ``np.gradient``, which is
    one-sided at the ends. Edge estimates are higher-variance, not invalid.
    ``edge_guard=0`` trains on all 600.
    """
    if edge_guard < 0:
        raise ValueError("edge_guard must be non-negative")
    score = np.percentile(np.linalg.norm(velocities, axis=2), 90, axis=1)
    stop = len(score) - edge_guard if edge_guard else len(score)
    valid = np.arange(edge_guard, stop)
    if len(valid) == 0:
        raise ValueError("edge_guard removed every frame")
    chosen = valid[np.argsort(score[valid])[-count:]]
    return sorted(map(int, chosen))









# ----------------------------------------------------------------------
# surface_motion: Train local-surface HuPR chirp motion with gradients from both radars.
# ----------------------------------------------------------------------









def microdoppler_marginal(
    magnitude: torch.Tensor,
    ra_mask: torch.Tensor,
    range_rows: torch.Tensor,
    motion_doppler: torch.Tensor,
) -> torch.Tensor:
    """Doppler marginal of the person's power inside its own support.

    One column of a micro-Doppler spectrogram: the velocity distribution at this
    instant, with range and angle structure integrated away, so a velocity error
    cannot be hidden by a range match.

    Power, not magnitude.  The micro-Doppler diagnostic established that
    combining range coherently cancels a deep rendered person far harder than a
    shallow measured one, and summing magnitudes has the same defect in milder
    form.  Power is the only combination that does not depend on how the return
    happens to be spread across the window.

    ``magnitude`` is Doppler x range x angle.  The result keeps only the motion
    Doppler bins, so the static return, including DC, never enters the loss.
    """
    power = magnitude.square() * ra_mask[None]
    return power.sum(dim=2)[:, range_rows].sum(dim=1)[motion_doppler]



def microdoppler_marginal_batched(
    magnitude: torch.Tensor,
    ra_mask: torch.Tensor,
    range_rows: torch.Tensor,
    motion_index: torch.Tensor | None,
) -> torch.Tensor:
    """``microdoppler_marginal`` over a leading batch of units.

    The single reason the loss chain could not be batched: ``range_rows`` is a
    variable-length index list per unit, so ``[:, range_rows]`` has a
    unit-dependent shape. Here it stays a fixed-shape boolean mask and the
    index-then-sum becomes multiply-then-sum over the same elements — the rows
    it drops are already exactly zero, because ``ra_mask`` zeroed them and
    ``range_rows`` is precisely ``ra_mask.any(dim=1)``.

    ``motion_index`` is the *shared* motion-Doppler bin index (the same mask
    for every unit), passed pre-resolved so the selection is an
    ``index_select`` rather than a boolean index that would sync. Selecting
    Doppler bins commutes with the range/angle sums, so it is applied first
    and the sums only run on the ~20 bins the marginal keeps.

    ``motion_index=None`` means the Doppler axis is *already* restricted to
    the motion bins — which is what the Doppler-first ordering in
    ``motion_doppler_magnitude_batched`` produces.

    ``magnitude`` is (units, Doppler, range, angle); the result is
    (units, motion bins).
    """
    if magnitude.ndim != 4:
        raise ValueError("expected batched Doppler magnitude (units,D,range,angle)")
    if ra_mask.shape != magnitude.shape[-2:] and ra_mask.shape != magnitude.shape[:1] + magnitude.shape[-2:]:
        raise ValueError("one range-angle mask per unit is required")
    kept = (magnitude if motion_index is None
            else magnitude.index_select(-3, motion_index))
    power = kept.square() * ra_mask[:, None]
    return (power.sum(dim=-1) * range_rows[:, None, :]).sum(dim=-1)


















# ----------------------------------------------------------------------
# joint_scattering: Fit per-radar joint scattering gains with fixed v22 motion and geometry.
# ----------------------------------------------------------------------




















# --------------------------------------------------------------------
# dual-radar: the two-radar fit
# --------------------------------------------------------------------


DEFAULT_MM3DPS_ROOT = Path(MM3DPS_ROOT)



def load_json(path: Path) -> dict:
    return json.loads(path.read_text())



def validate_inputs(
    dataset: Dataset,
    configs: dict[str, Path],
    pcl_path: Path,
    targets: dict[str, Path],
) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    config_data = {name: load_json(path) for name, path in configs.items()}
    config_report = validate_configs(dataset, config_data)
    if not config_report["valid"]:
        raise ValueError("invalid radar configs: " + "; ".join(config_report["errors"]))
    if dataset.name == "hupr":
        vertical_roll = float(config_data["vertical"]["hupr_provenance"]["board_roll_about_boresight_deg"])
        if not np.isclose(vertical_roll, 90.0):
            raise ValueError(f"expected evidence-backed +90 degree vertical roll, got {vertical_roll}")
        np.testing.assert_allclose(array_axis(config_data["horizontal"]), [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(array_axis(config_data["vertical"]), [0.0, 0.0, -1.0], atol=1e-6)

    pcl = np.load(pcl_path)
    target_data = {name: np.load(path) for name, path in targets.items()}
    if pcl.ndim != 2 or pcl.shape[1] < 6 or len(pcl) < 100:
        raise ValueError(f"expected nontrivial XYZ+normal cloud, got {pcl.shape}")
    normal_norm = np.linalg.norm(pcl[:, 3:6], axis=1)
    if not np.isfinite(pcl[:, :6]).all() or not np.allclose(normal_norm, 1.0, atol=2e-3):
        raise ValueError("point cloud contains invalid positions or non-unit normals")
    for name, target in target_data.items():
        expected = (dataset.radars[0].samples, dataset.radars[0].n_angle)
        if target.shape != expected or not np.isfinite(target).all() or target.max() <= 0:
            raise ValueError(f"invalid {name} target: shape={target.shape}")
    return config_report, pcl, {name: t.astype(np.float32) for name, t in target_data.items()}



def prepare_scene_layout(pcl_path: Path, output_dir: Path) -> Path:
    scene_dir = output_dir / "input_scene" / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    destination = scene_dir / "pcl.npy"
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(pcl_path.resolve())
    except OSError:
        shutil.copyfile(pcl_path, destination)
    # init_visible_weighted replaces this suffix with scene/pcl.npy. No mesh is
    # loaded because HuPR initialization disables occlusion ray tracing.
    return scene_dir / "mesh.ply"



def build_manifest(args: argparse.Namespace, config_report: dict, pcl: np.ndarray,
                   targets: dict[str, np.ndarray]) -> dict:
    dataset = Dataset.of(args)
    names = dataset.names
    first = targets[names[0]]
    return {
        "dataset": dataset.name,
        "shared_model": True,
        "radar_order_per_iteration": [
            *[f"{name}_forward_backward" for name in names], "single_optimizer_step"
        ],
        "view_loss_weights": {name: 1.0 / len(names) for name in names},
        "gradient_audit_required_per_view": True,
        **{f"{name}_config": str(path.resolve())
           for name, path in dataset.config_flags(args).items()},
        "pcl": str(args.pcl.resolve()),
        "pcl_points": int(len(pcl)),
        **{f"{r.name}_target": str(getattr(args, r.attr("", "target")).resolve())
           for r in dataset.radars},
        "target_shape": list(first.shape),
        "target_max": {name: float(target.max()) for name, target in targets.items()},
        "config_validation": config_report,
        "iterations": args.iters,
        "target_n": args.target_n,
        "material_lr": args.material_lr,
        "rotation_lr": args.rotation_lr,
        "seed": args.seed,
        "device": args.device,
        "status": "validated_inputs" if args.validate_only else "training_requested",
    }



def train(args: argparse.Namespace, manifest: dict, pcl: np.ndarray,
          targets_np: dict[str, np.ndarray]) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("full HuPR mm3DPS training requires an available CUDA device")
    mm3dps_root = args.mm3dps_root.resolve()
    # What the render needs from the 3DPS tree is its antenna patterns; the
    # renderer itself is mm4DPS.rasterizer. (A probe for train_gaussian.py
    # stood here from before the vendoring and never ran, because every HuPR
    # workspace had a seeded background model; the first from-scratch fit --
    # RT-Pose's -- hit it.)
    probe = mm3dps_root / "assets/antenna_pattern/MMWCAS/tx1_76.npy"
    if not probe.is_file():
        raise FileNotFoundError(
            f"antenna patterns not found: expected {probe}; --mm3dps-root "
            f"points at the vendored mm4DPS/splat tree")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Delay CUDA/Mitsuba imports until after --validate-only can exit on CPU.
    from mm4DPS import rasterizer as bsdf_torch
    from mm4DPS.rasterizer import Rasterizer
    from mm4DPS.rasterizer import (
        cull_gaussians,
        get_lr_scale,
        init_visible_weighted,
        render_gaussians,
        rms_clip_grad,
    )

    device = args.device
    pattern_root = mm3dps_root / "assets/antenna_pattern/MMWCAS"
    tx_pattern = pattern_root / "tx1_76.npy"
    rx_pattern = pattern_root / "rx1_76.npy"
    mesh_file = prepare_scene_layout(args.pcl, args.output_dir)
    common = dict(
        mesh_file=str(mesh_file), tx_pattern_file=str(tx_pattern),
        rx_pattern_file=str(rx_pattern), device=device,
    )
    dataset = Dataset.of(args)
    names = dataset.names
    rasterizers = {name: Rasterizer(config_file=str(path), **common)
                   for name, path in dataset.config_flags(args).items()}
    bsdf_torch.set_frequency(rasterizers[names[0]].center_freq)

    model = init_visible_weighted(
        "hupr_multiview", rasterizers[names[0]], target_n=args.target_n,
        device=device, enable_occlusion=False, near_field_m=HUPR_NEAR_FIELD_M,
    )
    masks, areas = {}, {}
    for name in names:
        masks[name] = cull_gaussians(model, rasterizers[name], near_field_m=HUPR_NEAR_FIELD_M)
        np.save(args.output_dir / f"{name}_training_mask.npy", masks[name].cpu().numpy())
        areas[name] = torch.zeros(model.N, device=device)
        areas[name][masks[name]] = 1.0
    if any(int(mask.sum()) == 0 for mask in masks.values()):
        raise RuntimeError("one radar culled every point; check config boresight/origin")

    targets = {name: torch.from_numpy(targets_np[name]).to(device) for name in names}
    optimizer = torch.optim.Adam([
        {"params": [model.raw_materials], "lr": args.material_lr, "name": "materials"},
        {"params": [model.rotations], "lr": args.rotation_lr, "name": "rotations"},
    ], betas=(0.9, 0.999), eps=1e-8)
    base_lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}

    per_view_kwargs = {
        name: {"vertex_areas": areas[name], "active_mask": masks[name]} for name in names
    }
    shared_render_kwargs = {
        "shadow_mask": None, "detach_phase": True, "bsdf_mode": "full",
        "disabled_components": None, "use_cuda_kernels": True,
    }

    def clip_gradients(shared_model: torch.nn.Module, _: torch.optim.Optimizer) -> None:
        rms_clip_grad(shared_model.raw_materials, 1.0)
        rms_clip_grad(shared_model.rotations, 0.5)

    history = []
    best_loss = float("inf")
    best_state = None
    started = time.time()
    for iteration in range(args.iters):
        lr_scale = get_lr_scale(
            iteration, total_iters=args.iters, warmup_iters=0,
            warmup_factor=1.0, decay_start=max(1, args.iters // 2),
        )
        for group in optimizer.param_groups:
            group["lr"] = base_lrs[group["name"]] * lr_scale
        report = train_hupr_iteration(
            model, optimizer, rasterizers, targets, render_gaussians,
            render_kwargs=shared_render_kwargs,
            per_view_render_kwargs=per_view_kwargs,
            before_step=clip_gradients,
        )
        report["iteration"] = iteration
        report["lr_scale"] = lr_scale
        history.append(report)
        if report["total_weighted_loss"] < best_loss:
            best_loss = report["total_weighted_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if iteration % 10 == 0 or iteration == args.iters - 1:
            print(
                f"iter={iteration} loss={report['total_weighted_loss']:.6f} "
                + " ".join(f"grad_{name}={report['gradient_norms'][name]:.3e}"
                           for name in names),
                flush=True,
            )

    if best_state is not None:
        torch.save(best_state, args.output_dir / "best_model.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = {
        "status": "completed",
        "iterations": args.iters,
        "best_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        **{f"{name}_active_points": int(masks[name].sum()) for name in names},
        "seed": args.seed,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "every_iteration_has_both_gradients": all(
            all(row["gradient_norms"][name] > 0 for name in names)
            for row in history
        ),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest.update({"status": "completed", "result": result})
    (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result



def make_parser() -> argparse.ArgumentParser:
    from mm4DPS import dataset_from_argv
    dataset = dataset_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    radar_configs(parser, dataset)
    parser.add_argument("--pcl", type=Path,
                        default=WORKSPACE / "multiview_scaffold/pcl.npy")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("", "target"), type=Path,
                            default=WORKSPACE / f"multiview_scaffold/{radar.name}_static_ra.npy")
    parser.add_argument("--mm3dps-root", type=Path, default=DEFAULT_MM3DPS_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=WORKSPACE / "training/dual_radar")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--target-n", type=int, default=20_000)
    parser.add_argument("--material-lr", type=float, default=1e-2)
    parser.add_argument("--rotation-lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    return parser



def main_dual_radar() -> None:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.of(args)
    config_report, pcl, targets = validate_inputs(
        dataset, dataset.config_flags(args), args.pcl,
        {r.name: getattr(args, r.attr("", "target")) for r in dataset.radars},
    )
    manifest = build_manifest(args, config_report, pcl, targets)
    (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.validate_only:
        print(json.dumps(manifest, indent=2))
        return
    result = train(args, manifest, pcl, targets)
    print(json.dumps(result, indent=2))


def _entry_dual_radar():
        main_dual_radar()


# ----------------------------------------------------------------------
# absolute-gain: the receiver gain of a dataset that ships none
# ----------------------------------------------------------------------

def run_absolute_gain(args: argparse.Namespace) -> dict:
    """Fit the frozen receiver gain from the background render against the measured static map.

    HuPR's gain was fitted on held-out sequences and shipped as a seed. For a
    dataset without one, the trained background model is rendered through
    the same rasterizer the renderer uses, beamformed to |RA|, and the gain
    is the median of measured / rendered over the cells where the measured
    static map is above its 90th percentile -- the scene's own bright
    returns, on linear magnitude. Written in the render_result.json form
    `_absolute_gain` reads. Estimator named in the output.
    """
    from mm4DPS import Dataset, rasterizer as bsdf_torch
    from mm4DPS.rasterizer import PointPrimitives, Rasterizer, cull_gaussians, render_gaussians
    from mm4DPS.forward_render import range_profiles_to_ra_magnitude
    dataset = Dataset.of(args)
    device = args.device
    mm3dps_root = args.mm3dps_root.resolve()
    pattern_root = mm3dps_root / "assets/antenna_pattern/MMWCAS"
    mesh_file = prepare_scene_layout(args.pcl, args.output_dir)
    common = dict(mesh_file=str(mesh_file), tx_pattern_file=str(pattern_root / "tx1_76.npy"),
                  rx_pattern_file=str(pattern_root / "rx1_76.npy"), device=device)
    state = torch.load(args.background_model, map_location=device, weights_only=False)
    model = PointPrimitives(len(state["positions"]), device=str(device))
    model.load_state_dict(state)
    model.requires_grad_(False).eval()
    gains, detail = {}, {}
    for radar in dataset.radars:
        rasterizer = Rasterizer(config_file=str(getattr(args, radar.attr("", "config"))), **common)
        bsdf_torch.set_frequency(rasterizer.center_freq)
        mask = cull_gaussians(model, rasterizer, near_field_m=HUPR_NEAR_FIELD_M)
        areas = torch.zeros(model.N, device=device)
        areas[mask] = 1.0
        with torch.no_grad():
            real, imag = render_gaussians(model, rasterizer, vertex_areas=areas, active_mask=mask,
                                          shadow_mask=None, detach_phase=True, bsdf_mode="full",
                                          disabled_components=None, use_cuda_kernels=True)
            rendered = range_profiles_to_ra_magnitude(real, imag, radar=radar).cpu().numpy()
        measured = np.load(args.scaffold_dir / f"{radar.name}_static_ra.npy").astype(np.float64)
        if measured.shape != rendered.shape:
            raise ValueError(f"{radar.name}: measured {measured.shape} vs rendered {rendered.shape}")
        # The near-field coupling column (the first `exclude_near_bins` range
        # bins) is in the measured map and not in the render: on the cascade
        # it is 49% of the static map's energy and a quarter of the brightest
        # decile, and with the render at ~0 there the median ratio was pulled
        # to 4.8 (IQR 0.57-167). Those bins are the coupling calibration's
        # business (`calibration sequence-coupling`), not the gain's.
        scored = measured.copy()
        scored[:args.exclude_near_bins] = 0.0
        keep = (scored > np.percentile(scored, 90)) & (rendered > 0)
        m, r = measured[keep], rendered[keep]
        ratio = m / r
        # The median of the per-cell ratios. Known to be a poor estimator
        # of a rig constant: the ratios are bimodal (~0.6 on the specular
        # wall the scaffold renders as a bright point, > 100 on the diffuse
        # room it renders as nothing), so the median follows whichever mode
        # holds more cells and with it the background fit's convergence
        # (seq 183: 8.4 after 500 iterations, 36.9 after 150) and the
        # scene (183: 7.3, 153: 23). The energy ratio sqrt(sum m^2 / sum
        # r^2), tried in its place, is stable to the fit but puts seq 183
        # at 1.2 -- the wall's level -- which the trainer's materials could
        # not raise back (400 epochs: spectrogram +0.935 -> +0.877). The
        # gain should be one number per rig, fitted across recordings:
        # docs/FIDELITY_PLAN.md 2.2(a). Both estimators are recorded.
        gains[radar.name] = float(np.median(ratio))
        detail[radar.name] = {"cells": int(keep.sum()),
                              "energy_ratio": float(np.sqrt((m * m).sum() / (r * r).sum())),
                              "ratio_iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
                              "excluded_near_bins": int(args.exclude_near_bins)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "completed", "dataset": dataset.name, "dataset_sequence": int(args.sequence),
              "model_label": f"seq {args.sequence}: absolute gain from the background render",
              "absolute_receiver_gain": gains,
              "absolute_gain_fit": {**detail, "estimator": "median of measured / rendered static |RA| over cells "
                                    "where the measured map, its near-field coupling bins zeroed, exceeds its "
                                    "90th percentile; linear magnitude (the energy ratio is recorded beside it)"},
              "geometry_translation_y_m": 0.0}
    (args.output_dir / "render_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_parser_absolute_gain() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_absolute_gain.__doc__)
    radar_configs(parser)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--background-model", type=Path, required=True)
    parser.add_argument("--pcl", type=Path, required=True)
    parser.add_argument("--scaffold-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-near-bins", type=int, default=15,
                        help="range bins left out of the fit: the antenna coupling column (HuPR's seed and "
                             "`calibration sequence-coupling` cover 15)")
    parser.add_argument("--mm3dps-root", type=Path, default=DEFAULT_MM3DPS_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _entry_absolute_gain():
    print(json.dumps(run_absolute_gain(make_parser_absolute_gain().parse_args()), indent=2))

# ----------------------------------------------------------------------
# dual_radar_training: Shared-parameter training step in which both HuPR radars contribute gradients.
# ----------------------------------------------------------------------


@dataclass
class RadarView:
    name: str
    rasterizer: Any
    target: Any
    pose: Any = None
    weight: float = 1.0



def _trainable(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]



def _snapshot_grad(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]



def dual_radar_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    views: list[RadarView],
    render_fn: Callable[[torch.nn.Module, RadarView], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    *,
    normalize_weights: bool = True,
    require_each_gradient: bool = True,
    before_step: Callable[[torch.nn.Module, torch.optim.Optimizer], None] | None = None,
) -> dict:
    """Run one forward/backward per view and one shared optimizer step.

    Gradients are zeroed once. Each radar is rendered with its own rasterizer,
    config, pose, and target, and its weighted loss immediately calls backward.
    PyTorch therefore accumulates both contributions in the same shared model
    parameters before the single optimizer step. Per-view incremental gradient
    norms make this contract observable even when the two gradients cancel.
    """
    if not views or len({view.name for view in views}) != len(views):
        raise ValueError("views must name each radar once")
    total_weight = sum(float(view.weight) for view in views)
    if total_weight <= 0:
        raise ValueError("radar weights must sum to a positive value")
    parameters = _trainable(model.parameters())
    if not parameters:
        raise ValueError("model has no trainable parameters")

    optimizer.zero_grad(set_to_none=True)
    report = {"losses": {}, "weighted_losses": {}, "gradient_norms": {}}
    for view in views:
        scale = float(view.weight) / total_weight if normalize_weights else float(view.weight)
        before = _snapshot_grad(parameters)
        prediction = render_fn(model, view)
        loss = loss_fn(prediction, view.target)
        if loss.ndim != 0:
            raise ValueError(f"{view.name} loss must be scalar, got {tuple(loss.shape)}")
        weighted_loss = loss * scale
        weighted_loss.backward()
        contribution_sq = torch.zeros((), device=loss.device)
        for parameter, previous in zip(parameters, before):
            current = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            contribution_sq = contribution_sq + (current - previous).pow(2).sum()
        contribution_norm = float(torch.sqrt(contribution_sq).detach().cpu())
        if require_each_gradient and not contribution_norm > 0.0:
            raise RuntimeError(f"{view.name} radar produced no gradient contribution")
        report["losses"][view.name] = float(loss.detach().cpu())
        report["weighted_losses"][view.name] = float(weighted_loss.detach().cpu())
        report["gradient_norms"][view.name] = contribution_norm

    total_grad_sq = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            total_grad_sq = total_grad_sq + parameter.grad.pow(2).sum()
    report["combined_gradient_norm"] = float(torch.sqrt(total_grad_sq).detach().cpu())
    report["total_weighted_loss"] = sum(report["weighted_losses"].values())
    if before_step is not None:
        before_step(model, optimizer)
    optimizer.step()
    return report



def train_hupr_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rasterizers: dict[str, Any],
    targets: dict[str, torch.Tensor],
    render_gaussians_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    *,
    render_kwargs: dict[str, Any] | None = None,
    per_view_render_kwargs: dict[str, dict[str, Any]] | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = log_pearson_loss,
    before_step: Callable[[torch.nn.Module, torch.optim.Optimizer], None] | None = None,
) -> dict:
    """Run the required H forward/backward, V forward/backward, shared step.

    Construct each rasterizer from its own generated JSON config. Both calls
    operate on ``model``—the same Gaussian/material parameters—so their
    gradients accumulate. Returned per-view norms prove that neither view was
    accidentally detached, skipped, or overwritten.
    """
    kwargs = {} if render_kwargs is None else dict(render_kwargs)
    views = [RadarView(name, rasterizers[name], targets[name]) for name in rasterizers]

    def render(shared_model: torch.nn.Module, view: RadarView) -> torch.Tensor:
        view_kwargs = dict(kwargs)
        if per_view_render_kwargs is not None:
            view_kwargs.update(per_view_render_kwargs.get(view.name, {}))
        real, imag = render_gaussians_fn(shared_model, view.rasterizer, **view_kwargs)
        return range_profiles_to_ra_magnitude(real, imag)

    return dual_radar_step(
        model, optimizer, views, render, loss_fn,
        normalize_weights=True, require_each_gradient=True,
        before_step=before_step,
    )



COMMANDS = {
    "dual-radar": _entry_dual_radar,
    "absolute-gain": _entry_absolute_gain,
    # was mm4DPS.training.motion's own COMMANDS and main(); dispatch() pops the
    # subcommand and leaves the rest of argv to its parser, so the flags are
    # exactly the ones `python -m mm4DPS.training.motion` took.
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
