"""Per-scene training configuration dataclasses for mm3DPS.

This module owns the canonical config schema used by the training entry points.
JSON configs (``configs/<scene>/config.json``) are loaded into a
:class:`TrainingConfigSionna` instance, which holds the per-scene file paths
plus all renderer / optimizer / loss hyperparameters used in the paper.

Path resolution
---------------
Path-valued fields (``config_file``, ``scene_file``, ``gt_adc_file``,
``tx_pattern_file``, ``rx_pattern_file``) may be either absolute or relative.
Relative paths are resolved against ``$MM3DPS_ROOT`` if set, otherwise against
the parent of the ``mm3DPS`` package directory (which is the submission root).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List


def submission_root() -> str:
    """Return the submission root directory.

    Resolves to ``$MM3DPS_ROOT`` if set, otherwise the parent of the
    ``mm3DPS`` package directory (i.e. the repo root containing
    ``mm3DPS/``, ``configs/``, ``data/``, ``assets/``, ``checkpoints/``).
    """
    env = os.environ.get("MM3DPS_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(submission_root(), path))


@dataclass
class LossConfigSionna:
    """Unified configuration for all loss functions."""
    adc_mag_weight: float = 0.0
    adc_phase_weight: float = 0.0
    ra_mag_weight: float = 1.0
    ra_mag_ssim_weight: float = 0.0
    ra_mag_perceptual_weight: float = 0.0
    mat_regularization_weight: float = 0.0

    adc_mag_use_log: bool = False
    adc_mag_loss_type: str = "l2"

    adc_phase_loss_type: str = "none"

    ra_mag_use_log: bool = True
    ra_mag_loss_type: str = "l2"

    log_epsilon: float = 1e-6
    log_scale: float = 1.0

    phase_magnitude_weighting: bool = True
    phase_magnitude_power: float = 0.5
    phase_reference_channel: int = 0
    phase_use_gt_magnitude: bool = True
    phase_magnitude_clip_max: float = 0.0

    huber_delta: float = 1.0

    use_joint_normalization: bool = False

    use_parameter_anchoring: bool = False
    parameter_anchor_checkpoint: str = ""
    parameter_anchor_weight: float = 0.01

    use_multi_chirp_loss: bool = False
    multi_chirp_phase_weight: float = 0.0
    multi_chirp_top_percentile: float = 0.9

    phase_metrics_type: str = "unit_phasor"


@dataclass
class TrainingConfigSionna:
    """Per-scene training configuration."""

    # ---- Files & Paths ----
    config_file: str = ""
    scene_file: str = ""
    gt_adc_file: str = ""
    output_dir: str = "output/train_sionna"
    tx_pattern_file: str = ""
    rx_pattern_file: str = ""

    # ---- Differentiable renderer config ----
    n_hits_per_rx: int = 500
    n_rays_per_res: int = 16
    n_avg_samples: int = 1
    bsdf_model: str = "mmwave_scalar"
    mmwave_polarization: str = "vertical"
    hemisphere_sampling: str = "cosine"
    double_sided: bool = True
    use_pattern_importance_sampling: bool = False

    # ---- Specular path finding ----
    enable_image_method: bool = False
    enable_sms: bool = True
    sms_max_iterations: int = 20
    sms_solver_threshold: float = 1e-5
    sms_use_smooth_normals: bool = False
    sms_cache_interval: int = 0
    enable_diffraction: bool = False
    use_patch_clustering: bool = True
    patch_angle_threshold_deg: float = 10.0
    patch_distance_threshold: float = 0.05
    patch_max_tris: int = 2000
    use_spatial_adjacency: bool = False
    spatial_radius: float = 0.05

    # ---- End-to-end differentiable & multibounce ----
    use_end_to_end_ad: bool = True
    max_bounces: int = 3
    nee_every_bounce: bool = True
    rr_start_bounce: int = 3
    rr_prob: float = 0.5
    e2e_ray_seed_rotation_interval: int = 10
    e2e_phase_chunk_size: int = 64
    enable_grad_phase: bool = False
    enable_boundary_gradients: bool = True

    # ---- Physics material model ----
    physics_mode: bool = True
    param_mode: str = "per_vertex"
    initial_material: str = ""
    enable_error_map_init: bool = False
    material_columns: int = 6

    # ---- Material bounds (raw-space clamps) ----
    eps_imag_raw_upper: float = 4.6
    sigma_h_raw_upper: float = -2.3
    thickness_raw_upper: float = -0.7

    # ---- Optimization ----
    num_iterations: int = 200
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    lr_materials: float = 0.5
    lr_pose_rotation: float = 5e-3
    lr_pose_translation: float = 5e-4
    lr_normals: float = 0.01
    lr_vertex_positions: float = 5e-4
    lr_patterns: float = 0.05

    clip_materials: float = 1.0
    clip_pose_rotation: float = 0.5
    clip_pose_translation: float = 0.5
    clip_normals: float = 0.5
    clip_vertex_positions: float = 0.1
    clip_patterns: float = 1.0

    LEARN_MAT: bool = True
    LEARN_POSE: bool = False
    LEARN_NRM: bool = False
    LEARN_VTX: bool = False
    LEARN_PAT: bool = False

    enable_staged_training: bool = False
    stage1_iterations: int = 50
    material_warmup_iters: int = 5

    enable_lr_schedule: bool = True
    lr_warmup_iters: int = 5
    lr_warmup_factor: float = 0.3

    pose_rotation_max_rad: float = 0.0349
    pose_translation_max_m: float = 1.95e-3
    geom_laplacian_weight: float = 0.3

    init_pitch_deg: float = 0.0
    init_roll_deg: float = 0.0
    init_yaw_deg: float = 0.0
    init_translation_x: float = 0.0
    init_translation_y: float = 0.0
    init_translation_z: float = 0.0

    loss_config: LossConfigSionna = field(default_factory=LossConfigSionna)

    visualization_interval: int = 10
    checkpoint_interval: int = 50
    log_interval: int = 1
    viz_scale: str = "dB"
    viz_db_floor: float = -40.0

    target_correlation: float = 0.90
    target_mse: float = 1e-4
    enable_early_stopping: bool = False

    pattern_mode: str = "shared"

    init_checkpoint: str = ""

    frozen_params: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tx_pattern_file and self.pattern_mode in ("shared", "mmwcas"):
            assets = os.path.join(submission_root(), "assets/antenna_pattern/MMWCAS")
            self.tx_pattern_file = os.path.join(assets, "tx1_76.npy")
            self.rx_pattern_file = os.path.join(assets, "rx1_76.npy")

        if self.enable_sms:
            self.enable_image_method = False

        if self.physics_mode:
            self.material_columns = 6

        if not self.frozen_params:
            self.frozen_params = []
            if not self.LEARN_MAT:
                self.frozen_params.append("materials")
            if not self.LEARN_POSE:
                self.frozen_params.extend(["pose_rotation", "pose_translation"])
            if not self.LEARN_NRM:
                self.frozen_params.append("normals")
            if not self.LEARN_VTX:
                self.frozen_params.append("vertex_positions")
            if not self.LEARN_PAT:
                self.frozen_params.append("patterns")


def load_config_from_json(json_path: str) -> TrainingConfigSionna:
    """Load a per-scene config from JSON.

    Relative path-valued fields are resolved against the submission root
    (the parent of the ``mm3DPS`` package directory) or ``$MM3DPS_ROOT``.
    """
    with open(json_path, "r") as f:
        config_dict = json.load(f)

    config = TrainingConfigSionna()

    for key, value in config_dict.items():
        if hasattr(config, key):
            if key == "loss_config" and isinstance(value, dict):
                loss_cfg = LossConfigSionna()
                for lk, lv in value.items():
                    if hasattr(loss_cfg, lk):
                        setattr(loss_cfg, lk, lv)
                config.loss_config = loss_cfg
            else:
                setattr(config, key, value)

    config.frozen_params = []
    config.__post_init__()

    for path_attr in ("config_file", "scene_file", "gt_adc_file",
                      "tx_pattern_file", "rx_pattern_file"):
        v = getattr(config, path_attr, "")
        if v:
            setattr(config, path_attr, _resolve(v))

    return config
