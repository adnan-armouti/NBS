# `mm4DPS/splat/` — provenance and attribution

This directory is **3DPS code, adapted**. It was vendored at
`mm4DPS/thirdparty_3dps/` and is folded in here because 4DPS runs it: keeping a
whole third-party library to use a fraction of it made the dependency look
larger and more separable than it is.

## Upstream

**3DPS — differentiable 3D point-scatterer rendering for mmWave radar.** See the
citation in the repository README; the vendored subset came from the authors'
submission, which also lives at `third_party/3DPS_submission/` untouched. The
copyright and licence of the original code remain with its authors. Nothing here
is claimed as 4DPS's own work except the deletions and the file layout.

## What was kept, and why

4DPS imports **13 symbols**. Everything here is their transitive closure:

| now at | was | holds |
|---|---|---|
| `mm4DPS/rasterizer.py` | `train_gaussian.py` | `PointPrimitives`, `render_gaussians`, `cull_gaussians`, `init_visible_weighted`, `rms_clip_grad`, `get_lr_scale`, `ITU_CONCRETE`, and their private helpers |
| `mm4DPS/rasterizer.py` | `rasterizer.py` | `Rasterizer`, `reparameterize_torch`, `inverse_reparameterize_torch` |
| `mm4DPS/rasterizer.py` | `rasterizer_factorized.py` | `render_factorized` |
| `mm4DPS/rasterizer.py` | `bsdf_torch.py` | the BSDF |
| `mm4DPS/rasterizer.py` | `psf.py` | the Hann-window range PSF |
| `mm4DPS/cuda_ext.py` | `cuda/__init__.py` | the extension loader and its autograd wrappers |
| `cuda/` | same | the `mm3DPS_cuda` sources and the built extension — `splat_scatter` |
| `assets/antenna_pattern/` | same | the MMWCAS patterns `MM3DPS_ROOT` resolves to |

C7 of `docs/CONSOLIDATION_AND_OO_PLAN.md` moved the Python up one level. The five
modules became one because they are mutually dependent -- `primitives` imports
`rasterizer` and `rasterizer_factorized` imports `bsdf_torch`, in opposite
directions -- so no two-way split of them is acyclic. This directory keeps what
is not Python: the antenna patterns, the CUDA sources, and this attribution. The
licence applies to the code wherever it sits; the table above is how to find it.

`config.py` was listed here as "needed by the above" and it was not. It held a
docstring describing `TrainingConfigSionna` dataclasses that no longer existed,
plus four unused imports, and nothing imported it. Deleted in C7.

`train_gaussian.py` was renamed to `primitives.py` because it no longer trains
anything: it holds the splat primitives and the renderer, and the name would have
been actively misleading.

## What was removed, and why that is not misrepresentation

**Upstream's Gaussian training program**: `train_gaussians` (555 lines),
`run_all_scenes`, the RA-loss and polar/cartesian helpers — 15 definitions, 792
lines. 4DPS does not run it; `mm4DPS/training/material.py` and
`mm4DPS/training/velocity_field.py` are its trainers, and upstream's read 263
never-executed statements, the largest single dead block in the tree.

With it went the four files only it reached: `data/ra_utils.py` (228 statements,
21 ever executed), `material_diagnostics.py` (85/12), `data/io_utils.py` (40/10),
`load_pretrained.py` (14/9).

Removing code we do not run does not misrepresent authorship: the attribution
above is unchanged, the licence still applies to what remains, and the complete
original is both in `third_party/3DPS_submission/` and recoverable at
`git show pre-deploy-strip:mm4DPS/thirdparty_3dps/`.

## The extension's name is upstream's, deliberately

The built module is still `mm3DPS_cuda`, declared in `cuda/bindings.cpp`.
Renaming it would touch upstream's C++ for no functional gain, so the directory
moved and the name did not.
