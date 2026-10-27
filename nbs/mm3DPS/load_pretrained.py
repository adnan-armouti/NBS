"""Per-scene config + (optional) trained-pattern loaders.

``load_trained_config(scene)`` reads the scene's JSON config from
``configs/<scene>/config.json`` (relative to the submission root) and
returns a :class:`mm3DPS.config.TrainingConfigSionna` with all paths
resolved. The default behaviour assumes the layout shipped with the
release:

::

    <root>/
        mm3DPS/
        configs/<scene>/config.json
        data/<scene>/...
        assets/antenna_pattern/MMWCAS/{tx1_76,rx1_76}.npy
        checkpoints/<scene>/best_model.pt

Set ``MM3DPS_ROOT`` to override the auto-detected root (useful for
running from a custom layout).

``load_pattern_data(scene)`` is provided for compatibility with the
``inject_trained_params`` API. The release ships factory antenna patterns
only, so it always returns ``None`` and the rasterizer uses the factory
patterns from ``assets/antenna_pattern/MMWCAS/``.
"""

from __future__ import annotations

import os

from mm3DPS.config import (
    TrainingConfigSionna,
    load_config_from_json,
    submission_root,
)


SCENES = [
    "seq_0_frame_135",
    "seq_1_frame_185",
    "seq_1_frame_438",
    "seq_2_frame_105",
    "seq_2_frame_160",
    "seq_2_frame_300",
]


def configs_dir() -> str:
    return os.path.join(submission_root(), "configs")


# Kept as a module-level constant for backwards compatibility with helper
# scripts that read it. Resolved lazily via ``submission_root()``.
TRAIN_OUTPUT_DIR = configs_dir()


def load_trained_config(scene: str) -> TrainingConfigSionna:
    """Load the per-scene training config from ``configs/<scene>/config.json``."""
    config_path = os.path.join(configs_dir(), scene, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found for scene {scene!r}: {config_path}. "
            f"Set MM3DPS_ROOT or run from the submission root."
        )
    return load_config_from_json(config_path)


def load_pattern_data(scene: str):
    """Return None; the release ships factory antenna patterns only.

    The training scripts call this and pass the result to
    ``Rasterizer.inject_trained_params``. Returning ``None`` keeps the
    factory ``.npy`` patterns loaded at construction time.
    """
    return None
