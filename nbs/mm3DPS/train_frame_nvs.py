"""Frame-level NVS trainer (v5 CUDA, pass-2 alignment).

Trains on all 16 chirp loops of N training frames (total 16·N RA maps),
holds out one test frame entirely, and evaluates on a single loop of the
test frame (default loop 0 — "first chirp"). The remaining 15 loops of
the test frame are discarded so the NVS test is a single held-out RA.

Pose synthesis:
  For each (frame, loop) pair we synthesise a per-loop pose by linearly
  interpolating between two aligned cascade anchors (``F-1`` and ``F+1``)
  at alpha = 0.5 + k·loop_dt/(2·frame_period). Edge train frames missing a
  neighbour fall back to the frame's own aligned pose for all 16 loops.
  The test frame's loop-0 pose is interpolated between its own neighbours
  (test_frame-1 and test_frame+1), so the test pose is NVS-inferred —
  never taken from test_frame's own aligned config.

Usage (example):
    python -m mm3DPS.train_frame_nvs \
        --scene seq_1_frame_438 \
        --test_frame 438 \
        --train_frames 434,435,436,437,439,440,441,442 \
        --iters 500 \
        --use_pass2_alignment
"""

# ============================================================================
# XRF55 ADAPTATION (copy in misc/ — all edits marked with `# XRF55:`)
# ----------------------------------------------------------------------------
# Set DATASET = 'xrf55' to train the mmWave RA renderer on XRF55 instead of the
# stock ColoRadar cascaded ADC. Two things change vs stock (rest is untouched):
#   1. GT LOADING — XRF55 ships processed magnitude heatmaps (1,17,256,128) with
#      [doppler 0:64 | azimuth 64:128]. We use the AZIMUTH HALF ONLY ([...,64:128]);
#      the Doppler half is NOT used by the static-RA renderer. There is no ADC and
#      no complex range profile, so the GT is the RA magnitude image directly.
#   2. LOSS — compute_ra_loss_rp is already a *magnitude*-RA loss (it does
#      range_profile_to_ra(rp).abs() vs a magnitude GT). XRF55 just supplies that
#      magnitude GT directly. BUT the stock RA assembly is the 127-az CASCADED
#      array; XRF55 is the 64-az SINGLE-CHIP ULA, so we swap in a single-chip
#      range-profile -> RA (mirrors data/ra_utils.adc_to_ra_complex_single_chip).
#
# STILL TO WIRE for an end-to-end XRF55 run (out of scope of this GT+loss change,
# flagged honestly):
#   * POSE: XRF55 radar is FIXED (only the human moves) -> a SINGLE rig pose, not
#     the per-frame/-loop alignment-config interpolation in build_frame_level_dataset/
#     _build_frame_poses. Replace with the per-scene rig (data/XRF55/<scene>_rig.json).
#   * DATA DISCOVERY: iterate XRF55 clip .npy files instead of cascaded_frame_<F>.npy.
#   * RASTERIZER ANTENNA GEOMETRY: configure the 3Tx x 4Rx single-chip array so
#     render_gaussians returns rp shaped (TX=3, RX=4, K); see ra_forward.single_chip_layout_m.
#   * CART metric grid + range_profile_to_ra_mag call-sites use the 64-az single-chip
#     forms below (already branched in the loss/eval path).
# ============================================================================
DATASET = 'coloradar'          # XRF55: set to 'xrf55'
XRF55_N_AZ = 64                # single-chip azimuth bins ([...,64:128])

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm3DPS.data.ra_utils import adc_to_ra_complex
from mm3DPS.data.io_utils import compute_range_res_from_cfg

from mm3DPS.rasterizer import Rasterizer, reparameterize_torch  # noqa: F401
from mm3DPS import cuda as v5cuda
from mm3DPS.train_gaussian import (
    DEVICE,
    init_visible_weighted,
    cull_gaussians,
    render_gaussians,
    range_profile_to_ra_mag,
    build_polar_to_cart_grid,
    polar_to_cart_torch,
    cart_corr_torch,
    precompute_gt_loss_norm,
    compute_ra_loss_rp,
    get_lr_scale,
    rms_clip_grad,
    USE_FACTORY_PATTERNS,
)
from mm3DPS.load_pretrained import load_trained_config, load_pattern_data

from mm3DPS._pose_utils import (
    build_per_loop_poses, apply_pose,
    load_pose_from_config as _load_pose_from_config,
)

# XRF55: single-chip vx mapping (8-element azimuth ULA) — the renderer-side mod.
from mm3DPS.data.ra_utils import single_chip_to_vx


# ---------------------------------------------------------------------------
# XRF55: RA-half GT loading + single-chip magnitude-RA loss
# ---------------------------------------------------------------------------
def load_xrf55_ra(clip_npy, reduce='mean'):
    """XRF55 mmWave heatmap (1,17,256,128) -> RA magnitude (256 range, 64 az).

    Uses ONLY the azimuth half [..., 64:128]; the Doppler half [..., 0:64] is
    NOT used by the static-RA renderer (mirrors preprocessing/xrf55_io.radar_ra).
    """
    a = np.load(clip_npy)
    if a.ndim == 4:
        a = a[0]                                   # (17, 256, 128)
    ra = a[:, :, 64:128]                           # azimuth half -> (17, 256, 64)
    red = {'mean': ra.mean(0), 'max': ra.max(0),
           'median': np.median(ra, 0)}[reduce]
    return red.astype(np.float32)                  # (256 range, 64 az)


def range_profile_to_ra_single_chip(rp_real, rp_imag, n_az=XRF55_N_AZ):
    """Rendered complex range profiles -> single-chip RA magnitude (n_az, K).

    Mirrors data/ra_utils.adc_to_ra_complex_single_chip's AZIMUTH FFT on the
    8-element ULA, but the input is already range-profiled (no range FFT).
    render_gaussians returns rp as (TX=3, RX=4, K); single_chip_to_vx wants (RX=4, TX=3, K),
    so we permute (matches the stock cascaded range_profile_to_ra, which also permutes TX<->RX).
    (The stock cascaded assembly uses a 12x16 array -> 127 az; this is the 8-vx ULA -> 64 az.)
    """
    rp_c = torch.complex(rp_real, rp_imag)         # (TX=3, RX=4, K)
    rp_c = rp_c.permute(1, 0, 2).contiguous()      # -> (RX=4, TX=3, K)
    vx = single_chip_to_vx(rp_c)                   # (8, K) complex — azimuth ULA
    nvx = vx.size(0)
    vx = vx * torch.hann_window(nvx, device=vx.device,
                                dtype=vx.real.dtype).to(vx.dtype)[:, None]
    vx = torch.fft.ifftshift(vx, dim=0)
    ra = torch.fft.fft(vx, n=n_az, dim=0)          # azimuth FFT -> n_az bins
    ra = torch.fft.fftshift(ra, dim=0)
    return ra.abs().float()                        # (n_az, K) magnitude


def _mag_ra_loss(ra_rend_mag, gt_mag, loss_type='pearson'):
    """Magnitude-RA loss (same forms as compute_ra_loss_rp, but the input is an
    already-assembled single-chip |RA| and the GT is the XRF55 |RA| half).
    'pearson' is the recommended default (scale-invariant; XRF55 |RA| is in
    arbitrary processed units, so absolute-scale MSE is not meaningful)."""
    if loss_type == 'pearson':
        r = ra_rend_mag.reshape(-1); g = gt_mag.reshape(-1)
        rc = r - r.mean(); gc = g - g.mean()
        corr = (rc * gc).sum() / torch.sqrt((rc * rc).sum() * (gc * gc).sum()).clamp(min=1e-30)
        loss = 1.0 - corr
        return loss, {'pearson_loss': loss.item()}
    elif loss_type == 'mse':
        mn, mx = ra_rend_mag.min(), ra_rend_mag.max()
        rend_norm = (ra_rend_mag - mn) / (mx - mn).clamp(min=1e-30)
        loss = torch.mean((rend_norm - gt_mag) ** 2)
        return loss, {'ra_mse': loss.item()}
    else:  # 'mse_raw' — requires the rendered scale to match XRF55's processed scale
        gt_scale = gt_mag.mean().detach().clamp(min=1e-30) ** 2
        loss = ((ra_rend_mag - gt_mag) ** 2).mean() / gt_scale
        return loss, {'ra_mse_raw': loss.item()}


def _build_xrf55_gt(clip_npy, loss_type, device, reduce='mean'):
    """XRF55 GT for the magnitude-RA loss: the RA-half magnitude only (no ADC,
    no complex range profiles). Drop-in replacement for _build_per_loop_gt.

    Returns the GT in the renderer's RA layout (az, range): the XRF55 heatmap is
    (range, az), so we transpose. NOTE: verify the azimuth sign/fftshift matches
    range_profile_to_ra_single_chip on a known clip before trusting numbers.
    """
    ra = load_xrf55_ra(clip_npy, reduce)                       # (256 range, 64 az)
    gt = torch.from_numpy(ra).to(device).T.contiguous()        # -> (64 az, 256 range)
    if loss_type != 'mse_raw':
        mn, mx = gt.min(), gt.max()
        gt = (gt - mn) / (mx - mn).clamp(min=1e-30)
    gt = gt.detach()
    return {'gt_loss': gt, 'gt_ra_polar': gt}


def _ra_mag(rp_real, rp_imag):
    """RA magnitude from rendered range profiles — single-chip (64 az) for XRF55,
    stock cascaded (127 az) otherwise. Used for both the loss GT-compare and the
    cart_corr metric so the rendered RA always matches the GT azimuth grid."""
    if DATASET == 'xrf55':
        return range_profile_to_ra_single_chip(rp_real, rp_imag)
    return range_profile_to_ra_mag(rp_real, rp_imag)


def _compute_ra_loss(rp_real, rp_imag, gt_cached, loss_type):
    """Dataset-aware RA loss. XRF55: single-chip |RA| magnitude loss vs the
    RA-half GT. Stock: the cascaded range-profile magnitude loss (compute_ra_loss_rp)."""
    if DATASET == 'xrf55':
        ra_rend = range_profile_to_ra_single_chip(rp_real, rp_imag)
        return _mag_ra_loss(ra_rend, gt_cached, loss_type=loss_type)
    return compute_ra_loss_rp(rp_real, rp_imag, gt_cached, loss_type=loss_type)


# ---------------------------------------------------------------------------
# Per-scene data loading (train and test frames)
# ---------------------------------------------------------------------------

def _aligned_config_path(scene, frame, use_pass2, data_root):
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    suffix = '_aligned_pass2' if use_pass2 else '_aligned'
    p = os.path.join(align_dir, f'cascaded_frame_{frame}{suffix}.json')
    return p if os.path.isfile(p) else None


def _pass3_config_path(scene, frame, chirp, data_root):
    """Per-chirp Stage 3 config path (produced by
    ``per_chirp_alignment.run_stage3_for_scene``)."""
    return os.path.join(
        data_root, 'alignment_data', scene, 'cascade', 'per_chirp',
        f'cascaded_frame_{frame}_chirp{chirp:02d}_aligned_pass3.json')


def _build_frame_poses(scene, frame, use_pass2, data_root,
                       loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.1,
                       anchor_source='pass2_lerp',
                       device=DEVICE):
    """Return a list of 16 per-loop pose dicts for ``frame``.

    ``anchor_source``:
      * ``'pass2_lerp'`` (default) — LERP between pass-2 F-1 and F+1
        aligned configs at per-chirp α. This is the historical
        behaviour of the trainer.
      * ``'pass3_per_chirp'`` — load each (frame, chirp)'s own
        Stage 3 refined config directly from
        ``data/alignment_data/<scene>/cascade/per_chirp/
          cascaded_frame_<F>_chirp<CC>_aligned_pass3.json``.
        Falls back to ``pass2_lerp`` for any chirp whose Stage 3 file
        is missing.

    For edge train frames (no F-1 or F+1 under pass2_lerp), falls back
    to ``frame``'s own aligned pose for all 16 loops (sub-mm
    chirp-loop motion is negligible vs the frame-level motion we're
    training on).
    """
    # Pass-3 path: load per-chirp configs directly, one per (F, c).
    if anchor_source == 'pass3_per_chirp':
        poses = []
        fallback_used = False
        for c in range(16):
            cfg_p = _pass3_config_path(scene, frame, c, data_root)
            if not os.path.isfile(cfg_p):
                fallback_used = True
                break
            tx, rx, tb, rb = _load_pose_from_config(cfg_p, device=device)
            poses.append({
                'alpha': 0.5 + c * loop_dt_s / (2 * frame_period_s),
                'tx_positions':  tx.contiguous(),
                'rx_positions':  rx.contiguous(),
                'tx_boresights': tb.contiguous(),
                'rx_boresights': rb.contiguous(),
            })
        if not fallback_used:
            return poses, 'pass3_per_chirp'
        # else fall through to pass2_lerp

    # Pass-2 LERP path (default / fallback).
    cfg_A = _aligned_config_path(scene, frame - 1, use_pass2, data_root)
    cfg_B = _aligned_config_path(scene, frame + 1, use_pass2, data_root)
    if cfg_A is not None and cfg_B is not None:
        poses, _ = build_per_loop_poses(
            cfg_A, cfg_B, n_loops=16,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            device=device)
        return poses, 'interp_neighbours'

    cfg_F = _aligned_config_path(scene, frame, use_pass2, data_root)
    assert cfg_F is not None, (
        f'{scene} f={frame}: no aligned config (neither neighbour nor '
        f'self); cannot build poses')
    tx, rx, tb, rb = _load_pose_from_config(cfg_F, device=device)
    pose = {
        'alpha': 0.5,
        'tx_positions':  tx.contiguous(),
        'rx_positions':  rx.contiguous(),
        'tx_boresights': tb.contiguous(),
        'rx_boresights': rb.contiguous(),
    }
    return [pose] * 16, 'frame_own_pose'


def _build_per_loop_gt(adc_npy_path, loop_idx, loss_type, device):
    """Load a single (frame, loop) GT as a dict with 'gt_loss' +
    'gt_ra_polar' (magnitude). Caller converts polar→cart later once the
    sample_grid is built.
    """
    # XRF55: no ADC/loops — the GT is the RA-half magnitude of the clip heatmap.
    # (loop_idx is ignored; the static-RA GT is one image per clip.)
    if DATASET == 'xrf55':
        return _build_xrf55_gt(adc_npy_path, loss_type, device)

    arr = np.load(adc_npy_path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'expected cascaded ADC shape (16,RX,TX,K); got {arr.shape}')
    ri = np.stack([arr[loop_idx].real, arr[loop_idx].imag],
                  axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)  # (TX, RX, K, 2)
    gt_adc_ri = torch.from_numpy(ri).to(device)
    gt_loss = precompute_gt_loss_norm(gt_adc_ri, loss_type=loss_type)
    with torch.no_grad():
        ra_c = adc_to_ra_complex(gt_adc_ri)
        ra_mag = torch.abs(ra_c).float()
    return {'gt_loss': gt_loss, 'gt_ra_polar': ra_mag}


def build_frame_level_dataset(scene, train_frames, test_frame,
                               held_out_loop, use_pass2,
                               train_loops=None,
                               loss_type='mse_raw',
                               loop_dt_s=7.87e-3 / 16.0,
                               frame_period_s=0.1,
                               data_root='/path/to/mm3DPS/data',
                               device=DEVICE,
                               anchor_source='pass2_lerp',
                               verbose=True):
    """Return (train_samples, test_sample, diagnostics).

    train_samples is a list of |train_loops|·|train_frames| dicts with keys:
        'frame_idx', 'loop_idx', 'pose', 'gt_loss', 'gt_ra_polar'
    test_sample is a single dict with the same keys (+ nothing under
    'gt_loss'; test uses cart_corr only).

    ``train_loops`` defaults to all 16 chirp loops. Pass e.g. ``[0]`` to
    train only on the first chirp loop of each train frame (the
    "first-chirp" NVS variant). Training data comes from the TI MMWCAS
    12x16 cascaded radar.
    """
    if train_loops is None:
        train_loops = list(range(16))
    train_loops = list(train_loops)

    scene_dir = os.path.join(data_root, scene)
    radar_dir = os.path.join(scene_dir, 'radar')

    train_samples = []
    edges = []  # frames that fell back to frame-own pose
    for f in train_frames:
        poses, mode = _build_frame_poses(
            scene, f, use_pass2=use_pass2, data_root=data_root,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            anchor_source=anchor_source, device=device)
        if mode != 'interp_neighbours':
            edges.append((f, mode))
        adc_npy = os.path.join(radar_dir, f'cascaded_frame_{f}.npy')
        for k in train_loops:
            gt = _build_per_loop_gt(adc_npy, k, loss_type, device)
            train_samples.append({
                'frame_idx': int(f), 'loop_idx': int(k),
                'pose': poses[k],
                **gt,
            })

    # Test: one (frame, loop); pose interpolated from test_frame's neighbours
    # (or loaded directly from per-chirp Stage 3 config under anchor_source='pass3_per_chirp').
    test_poses, test_mode = _build_frame_poses(
        scene, test_frame, use_pass2=use_pass2, data_root=data_root,
        loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
        anchor_source=anchor_source, device=device)
    test_adc = os.path.join(radar_dir, f'cascaded_frame_{test_frame}.npy')
    test_gt = _build_per_loop_gt(test_adc, held_out_loop, loss_type, device)
    test_sample = {
        'frame_idx': int(test_frame), 'loop_idx': int(held_out_loop),
        'pose': test_poses[held_out_loop],
        **test_gt,
    }

    diag = {
        'n_train_samples': len(train_samples),
        'edge_frames': edges,
        'test_pose_mode': test_mode,
    }
    if verbose:
        print(f'  [data] train_samples: {len(train_samples)} '
              f'({len(train_frames)} frames × 16 loops)')
        if edges:
            print(f'  [data] edge frames (fell back to own pose): {edges}')
        print(f'  [data] test (frame={test_frame} loop={held_out_loop})  '
              f'pose_mode={test_mode}')
    return train_samples, test_sample, diag


# ---------------------------------------------------------------------------
# XRF55: fixed-pose dataset (the radar is STATIC; only the human moves)
# ---------------------------------------------------------------------------
def build_xrf55_dataset(scene, n_clips=40, loss_type='pearson', reduce='mean',
                        device=DEVICE, verbose=True):
    """XRF55 static-RA dataset: a SINGLE fixed-pose sample (no frame-NVS).

    Pose comes from the scene's single_chip.json (preprocessing.xrf55_radar_config) —
    the radar sits at the scaffold-frame origin with +Y boresight and never moves. The
    GT is the STATIC RA: the median ACROSS clips of each clip's frame-reduced RA-half
    [...,64:128], which suppresses the moving human and keeps the static room. Returns
    (train_samples, test_sample, diag) matching build_frame_level_dataset's contract so
    the rest of the trainer is unchanged.
    """
    import glob
    from preprocessing import xrf55_config as XC

    cfg_path = os.path.join(XC.NBS_DATA_OUT, scene, 'configs', 'single_chip.json')
    assert os.path.isfile(cfg_path), f'missing {cfg_path} (run preprocessing.xrf55_radar_config)'
    tx, rx, tb, rb = _load_pose_from_config(cfg_path, device=device)
    pose = {'alpha': 0.5, 'tx_positions': tx.contiguous(), 'rx_positions': rx.contiguous(),
            'tx_boresights': tb.contiguous(), 'rx_boresights': rb.contiguous()}

    files = sorted(glob.glob(os.path.join(XC.XRF55_ROOT, 'RF', scene, 'mmWave', '*.npy')))
    assert files, f'no XRF55 mmWave clips for {scene}'
    idx = np.linspace(0, len(files) - 1, min(n_clips, len(files))).astype(int)
    stack = []
    for i in idx:
        a = np.load(files[i])
        if a.ndim == 4:
            a = a[0]
        ra = a[:, :, 64:128]                                  # RA-half (17,256,64)
        stack.append({'mean': ra.mean(0), 'max': ra.max(0),
                      'median': np.median(ra, 0)}[reduce])
    static = np.median(np.stack(stack), 0).astype(np.float32)  # (256 range, 64 az) static RA
    gt = torch.from_numpy(static).to(device).T.contiguous()    # -> (64 az, 256 range)
    if loss_type != 'mse_raw':
        mn, mx = gt.min(), gt.max()
        gt = (gt - mn) / (mx - mn).clamp(min=1e-30)
    gt = gt.detach()

    sample = {'frame_idx': 0, 'loop_idx': 0, 'pose': pose,
              'gt_loss': gt, 'gt_ra_polar': gt}
    diag = {'n_train_samples': 1, 'static_clips': int(len(idx)), 'pose_mode': 'xrf55_fixed_rig'}
    if verbose:
        print(f'  [data:xrf55] fixed-pose static-RA from {len(idx)} clips; GT {tuple(gt.shape)} (az,range)')
    # train == test (single static scene); the held-out NVS notion doesn't apply.
    return [sample], sample, diag


def init_xrf55_model(scene, target_n=90000, enable_cull=True,
                     enable_cosine_resample=True, enable_fps=True, device=DEVICE):
    """Build the Rasterizer + Gaussian model for an XRF55 scene (no ColoRadar config).

    Rasterizer from the scene's single_chip.json (antennas + chirp) with FACTORY antenna
    patterns; mesh_file points at <scene>/scene/mesh.ply purely so init_visible_weighted
    resolves the scaffold at <scene>/scene/pcl.npy. Occlusion is OFF (no scene mesh), so
    no .ply is required and the Mitsuba scene is never loaded.
    """
    from preprocessing import xrf55_config as XC
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # NBS repo root
    cfg = os.path.join(XC.NBS_DATA_OUT, scene, 'configs', 'single_chip.json')
    mesh_file = os.path.join(XC.NBS_DATA_OUT, scene, 'scene', 'mesh.ply')   # not loaded; resolves pcl path
    tx_pat = os.path.join(root, 'assets', 'antenna_pattern', 'MMWCAS', 'tx1_76.npy')
    rx_pat = os.path.join(root, 'assets', 'antenna_pattern', 'MMWCAS', 'rx1_76.npy')
    rast = Rasterizer(config_file=cfg, mesh_file=mesh_file,
                      tx_pattern_file=tx_pat, rx_pattern_file=rx_pat, device=device)
    # BSDF carrier must match the config (stock global is 77 GHz); set to XRF55's ~62 GHz so
    # the slab phase / roughness (4*pi*sigma_h/lambda) / SPM physics use the right wavelength.
    from mm3DPS import bsdf_torch as _B
    _B.set_frequency(rast.center_freq)
    model = init_visible_weighted(
        scene, rast, target_n=target_n, enable_cull=enable_cull,
        enable_occlusion=False, enable_cosine_resample=enable_cosine_resample,
        enable_fps=enable_fps, device=device)
    return model, rast


# ---------------------------------------------------------------------------
# S4 — adaptive density: signal-weighted split + prune, fixed N budget.
# ---------------------------------------------------------------------------

def _densify_step(model, optimizer, fisher_per_pt, split_n, prune_n,
                  pos_jitter_m, mat_jitter, rot_jitter=0.05,
                  init_raw_ref=None, init_normals_ref=None,
                  verbose=False):
    """In-place split + prune to reallocate the fixed N-point budget.

    The bottom ``prune_n`` points (lowest signal) are *replaced* in-place
    by children of the top ``split_n`` points (highest signal). Child =
    parent params + Gaussian noise (position σ = ``pos_jitter_m``,
    rotation σ = ``rot_jitter``, material σ = ``mat_jitter``); Adam
    can repair this quickly.

    Tensor indices of non-modified points are preserved (prune slots are
    overwritten, not deleted). Adam state at replaced slots is zeroed so
    children start with no momentum. ``init_raw_ref`` / ``init_normals_ref``
    are patched so drift penalties see zero drift at birth.

    Requires ``split_n == prune_n`` and both ≤ 0.45·N.
    """
    assert split_n == prune_n, 'split_n and prune_n must match to preserve N'
    N = model.N
    assert split_n <= 0.45 * N, 'split/prune frac too large; may overlap'

    _, order = fisher_per_pt.sort(descending=True)
    split_idx = order[:split_n].clone()
    prune_idx = order[-prune_n:].clone()

    overlap = set(split_idx.tolist()) & set(prune_idx.tolist())
    assert not overlap, f'split/prune index overlap: {overlap}'

    with torch.no_grad():
        # Gather parent params
        pos_parent = model.positions[split_idx].clone()
        rot_parent = model.rotations[split_idx].clone()
        raw_parent = model.raw_materials[split_idx].clone()

        # Generate child position + rotation via Gaussian jitter.
        pos_new = pos_parent + torch.randn_like(pos_parent) * pos_jitter_m
        rot_new = torch.nn.functional.normalize(
            rot_parent + torch.randn_like(rot_parent) * rot_jitter, dim=-1)

        # Material: parent + Gaussian noise
        raw_new = raw_parent + torch.randn_like(raw_parent) * mat_jitter

        # In-place overwrite of prune slots with children
        model.positions[prune_idx]     = pos_new
        model.rotations[prune_idx]     = rot_new
        model.raw_materials[prune_idx] = raw_new

        # Adam state surgery: zero momentum + velocity at replaced slots.
        for p in [model.positions, model.rotations, model.raw_materials]:
            st = optimizer.state.get(p)
            if st is None:
                continue
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st and st[k].shape == p.shape:
                    st[k][prune_idx] = 0.0

        # Patch init_* so drift penalties see zero drift at birth
        if init_raw_ref is not None:
            init_raw_ref[prune_idx] = raw_new
        if init_normals_ref is not None:
            q = torch.nn.functional.normalize(rot_new, dim=-1)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            nx = 2 * (x * z + w * y)
            ny = 2 * (y * z - w * x)
            nz = 1 - 2 * (x * x + y * y)
            init_normals_ref[prune_idx] = torch.stack([nx, ny, nz], dim=-1)

    if verbose:
        print(f'  [S4] densify: split {split_n} top-signal → replaced '
              f'{prune_n} bottom-signal  (pos_jitter={pos_jitter_m:.3f} m)')


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_frame_nvs(scene,
                    train_frames,
                    test_frame,
                    held_out_loop=0,
                    train_loops=None,
                    num_iters=500,
                    mat_lr=0.01,
                    rot_lr=5e-3,
                    loss_type='mse_raw',
                    target_n=90000,
                    frame_period_s=0.1,
                    loop_dt_s=7.87e-3 / 16.0,
                    use_pass2_alignment=True,
                    anchor_source='pass2_lerp',
                    data_root='/path/to/mm3DPS/data',
                    verbose=True,
                    output_dir=None,
                    reg_warm_iters=50,
                    reg_densify_interval=0,
                    reg_densify_split_frac=0.02,
                    reg_densify_prune_frac=0.02,
                    reg_densify_until=300,
                    reg_densify_pos_jitter_m=0.02,
                    reg_densify_mat_jitter=0.1,
                    seed_frame=None,
                    learn_positions_lr=1e-5,  # v5_v2 Phase 1 carryover: > 0 → enable
                                              # learnable positions, AMPLITUDE GRADIENT
                                              # ONLY (phase remains detached). Sub-mm
                                              # bounded; L2 anchor to LiDAR init.
                    learn_positions_l2=100.0,  # L2 anchor coefficient on (pos − init)
                    # Ablation knobs (md/ablations_plan.md, Tier 1+2). Defaults
                    # match the canonical 3DPS recipe used in the main paper.
                    detach_phase=True,             # Tier-2 axis 7
                    use_mimo_factorization=True,   # Tier-1 axis 5 (False → exercise
                                                    # PyTorch fallback in render_factorized)
                    psf_spread=None,               # Tier-2 axis 6 (None → 15)
                    enable_cull=True,              # Tier-1 axis 4a
                    enable_occlusion=True,         # Tier-1 axis 4b
                    enable_cosine_resample=True,   # Tier-1 axis 4c
                    enable_fps=True):              # Tier-1 axis 4d
    assert v5cuda.is_available(), (
        'v5 CUDA extension not built. '
        'cd mm3DPS/cuda && python setup.py build_ext --inplace')

    if train_loops is None:
        train_loops = list(range(16))
    train_loops = list(train_loops)
    test_in_train = int(test_frame) in [int(f) for f in train_frames]

    if verbose:
        print('=' * 72)
        print(f'frame-level NVS  scene={scene}')
        print(f'  train frames ({len(train_frames)}): {train_frames}')
        print(f'  train loops  ({len(train_loops)}): {train_loops}')
        print(f'  test  frame  = {test_frame}  (test loop={held_out_loop})'
              + ('   [test frame ALSO in train_frames → upper-bound run]'
                 if test_in_train else ''))
        print(f'  alignment    = {"pass-2" if use_pass2_alignment else "pass-1"}')
        print(f'  iters        = {num_iters}   loss_type={loss_type}')
        print('=' * 72)

    # ── paths ──
    config = None if DATASET == 'xrf55' else load_trained_config(scene)

    # The seed frame's pose drives FOV culling + RX visibility before FPS
    # subsampling, so it determines *which* 90 000-point subset of pcl.npy
    # this run trains on. Default = `test_frame` so HO and UB variants share
    # an identical position grid (UB's middle-of-train-frames index already
    # lands on test_frame; HO's previously landed on `test_frame+1`,
    # producing two different ~90k subsets — see md/frame_nvs_analysis/
    # findings.md §0). Pass `seed_frame=<other>` to override (e.g. to
    # reproduce the legacy HO grid use `seed_frame=test_frame+1`).
    if DATASET == 'xrf55':                       # XRF55: single fixed-pose single-chip config
        from preprocessing import xrf55_config as _XC
        seed_cfg = os.path.join(_XC.NBS_DATA_OUT, scene, 'configs', 'single_chip.json')
    else:
        if seed_frame is None:
            seed_frame = int(test_frame)
        suffix = '_aligned_pass2' if use_pass2_alignment else '_aligned'
        align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
        seed_cfg = os.path.join(align_dir,
                                f'cascaded_frame_{seed_frame}{suffix}.json')
        if verbose:
            print(f'  seed frame   = {seed_frame}  '
                  f'(drives FOV+visibility before FPS → position grid)')
    assert os.path.exists(seed_cfg), f'missing: {seed_cfg}'

    # ── Rasterizer + model ──
    if DATASET == 'xrf55':                       # XRF55: single-chip config + scaffold pcl, no mesh/occlusion
        model, rast = init_xrf55_model(
            scene, target_n=target_n, enable_cull=enable_cull,
            enable_cosine_resample=enable_cosine_resample, enable_fps=enable_fps,
            device=DEVICE)
    else:
        rast = Rasterizer(
            config_file=seed_cfg,
            mesh_file=config.scene_file,
            tx_pattern_file=config.tx_pattern_file,
            rx_pattern_file=config.rx_pattern_file,
            device=DEVICE)
        if not USE_FACTORY_PATTERNS:
            rast.inject_trained_params(pattern_data=load_pattern_data(scene))

        # Init model at seed pose — FOV will be recomputed per-sample via
        # cull_gaussians during render_gaussians. We need a Mitsuba scene for
        # the init-time visibility test (skipped when enable_occlusion=False).
        model = init_visible_weighted(
            scene, rast, target_n=target_n,
            enable_cull=enable_cull,
            enable_occlusion=enable_occlusion,
            enable_cosine_resample=enable_cosine_resample,
            enable_fps=enable_fps)
        rast.free_mi_scene()

    # Bundle the per-call render kwargs once. All four render_gaussians
    # sites in this function (initial CC, training step, final test export,
    # per-train-frame export) consume the same set; pinning them here
    # guarantees consistency across train + eval + figure render.
    _render_kw = dict(
        bsdf_mode='full',
        disabled_components=None,
        detach_phase=detach_phase,
        use_cuda_kernels=use_mimo_factorization,
        psf_spread=psf_spread,
    )
    import gc; gc.collect(); torch.cuda.empty_cache()

    # FOV mask for the seed pose — reused as a conservative initial mask.
    # Per-sample rendering re-masks via cull_gaussians if needed.
    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    # ── polar → cart grid ──
    range_res = compute_range_res_from_cfg(seed_cfg)
    _n_az_grid = XRF55_N_AZ if DATASET == 'xrf55' else 127   # XRF55: 64-az single-chip RA
    sample_grid = build_polar_to_cart_grid(_n_az_grid, 256, range_res, 400, DEVICE)

    # ── Load train + test data ──
    if DATASET == 'xrf55':                       # XRF55: single fixed-pose static-RA sample
        train_samples, test_sample, diag = build_xrf55_dataset(
            scene=scene, loss_type=loss_type, device=DEVICE, verbose=verbose)
    else:
        train_samples, test_sample, diag = build_frame_level_dataset(
            scene=scene, train_frames=train_frames, test_frame=test_frame,
            held_out_loop=held_out_loop, use_pass2=use_pass2_alignment,
            train_loops=train_loops,
            loss_type=loss_type, loop_dt_s=loop_dt_s,
            frame_period_s=frame_period_s, data_root=data_root,
            anchor_source=anchor_source,
            device=DEVICE, verbose=verbose,
        )

    # Finalize GT cart norms for cart_corr evaluation
    def _finalize_cart(sample):
        with torch.no_grad():
            ra_cart = polar_to_cart_torch(sample['gt_ra_polar'], sample_grid)
            mn, mx = ra_cart.min(), ra_cart.max()
            sample['gt_cart_norm'] = (
                (ra_cart - mn) / (mx - mn).clamp(min=1e-30)).detach()
    for s in train_samples:
        _finalize_cart(s)
    _finalize_cart(test_sample)

    # ── Initial (pre-training) test cart_corr ──
    def _render_and_cart_corr(sample):
        apply_pose(rast, sample['pose'])
        with torch.no_grad():
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                **_render_kw)
            ra_polar = _ra_mag(rp_real, rp_imag)   # XRF55: 64-az single-chip RA
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
            cc = cart_corr_torch(ra_cart, sample['gt_cart_norm']).item()
        return float(cc)

    init_test_cc = _render_and_cart_corr(test_sample)
    init_train_ccs = [_render_and_cart_corr(s) for s in train_samples]
    if verbose:
        print(f'\n  [init] test  cc = {init_test_cc:.4f}')
        print(f'  [init] train mean cc = {np.mean(init_train_ccs):.4f} '
              f'(std {np.std(init_train_ccs):.4f}) over {len(init_train_ccs)} samples')

    # ── Optimizer ──
    param_groups = [
        {'params': [model.raw_materials], 'lr': mat_lr, 'name': 'materials'},
        {'params': [model.rotations],     'lr': rot_lr, 'name': 'rotations'},
    ]
    clip_vals = {'materials': 1.0, 'rotations': 0.5}
    # v5_v2 Phase 1 carryover — learnable positions, AMPLITUDE GRADIENT ONLY.
    # detach_phase=True (v5 default) ensures position gradient flows ONLY
    # through alpha_tx, BSDF cos terms, antenna gain — not through
    # phi_carrier or n_peak. NO PHASE GRADIENTS, by construction.
    if learn_positions_lr > 0.0:
        model.positions.requires_grad_(True)
        param_groups.append({'params': [model.positions], 'lr': learn_positions_lr,
                              'name': 'positions'})
        # Tight clip — bounded sub-mm motion. Phase wraps every 0.97 mm at
        # 77 GHz so we want movement << 1 mm per step.
        clip_vals['positions'] = 0.001  # 1 mm/step grad-norm cap
        if verbose:
            print(f'  [v5_v2 P1] learn_positions_lr={learn_positions_lr:g} '
                  f'(amplitude-path only, L2 anchor λ={learn_positions_l2:g})')
    base_lrs = {g['name']: g['lr'] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    loss_scale = 1.0 / len(train_samples)

    # ── Regularisation setup ──
    # Snapshot init raw_materials (uniform ITU_CONCRETE across N at init time).
    init_raw = model.raw_materials.detach().clone()
    # Snapshot init surface normals (from the quaternion init computed from
    # pcl.npy LiDAR normals). Used by densify to patch init refs at birth.
    init_normals = model.get_normals().detach().clone()  # (N, 3)
    # v5_v2 Phase 1 — snapshot init positions for L2 anchor against drift.
    init_positions = model.positions.detach().clone()  # (N, 3)
    # S4: adaptive density (split/prune) bookkeeping
    n_densify_rounds = 0
    # v5_v4 Phase 3 — position-gradient-magnitude accumulator for densify
    # selection. positions get .requires_grad_(True) so backward() populates
    # .grad through the AMPLITUDE PATH ONLY (phase path is detached as in v5).
    # We accumulate ‖∇p L‖ across iters between densify steps and use it as
    # the densify selection signal.
    pos_grad_accum = None
    # Diagnostics: a never-reset accumulator capturing ‖∇p L‖ across the
    # entire training run, plus per-densify concentration snapshots
    # (top-k cumulative fraction). Saved at end-of-training for offline
    # Fisher-imbalance analysis. Cheap (N×4 bytes ≈ 80 KB at N=20k).
    career_pos_grad = None
    grad_concentration_log = []
    if reg_densify_interval > 0:
        model.positions.requires_grad_(True)
        pos_grad_accum = torch.zeros(model.N, device=DEVICE)
        career_pos_grad = torch.zeros(model.N, device=DEVICE)
        if verbose:
            print(f'  [v5_v4 Phase 3] densify_signal=pos_grad_amp; positions '
                  f'requires_grad=True (NOT in optimizer; selection only)')
    if verbose and reg_densify_interval > 0:
        print(f'  [reg] densify_interval={reg_densify_interval}  '
              f'warm_iters={reg_warm_iters}  until={reg_densify_until}')

    # ── Training loop (full-batch: every sample every iter) ──
    best_mean_train_cc = -1.0
    best_iter = 0
    best_state = None
    history = []
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        per_sample_cc = []
        loss_sum = 0.0
        for s in train_samples:
            apply_pose(rast, s['pose'])
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                **_render_kw)
            loss_k, _ = _compute_ra_loss(            # XRF55: single-chip |RA| loss vs RA-half GT
                rp_real, rp_imag, s['gt_loss'], loss_type)
            (loss_k * loss_scale).backward()
            with torch.no_grad():
                ra_polar = _ra_mag(rp_real.detach(), rp_imag.detach())  # XRF55: 64-az
                ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
                cc = cart_corr_torch(ra_cart, s['gt_cart_norm']).item()
                per_sample_cc.append(cc)
            loss_sum += loss_k.item() * loss_scale
            del rp_real, rp_imag, ra_polar, ra_cart, loss_k

        # v5_v4 Phase 3 — capture full-batch position-gradient magnitude
        # AFTER all train_samples backward(). positions.grad has been
        # accumulated across all samples (PyTorch default). We only use
        # the magnitude as a densify-selection signal; positions are NOT
        # in the optimizer (they never move).
        if pos_grad_accum is not None and model.positions.grad is not None:
            with torch.no_grad():
                step_grad = model.positions.grad.detach().norm(dim=-1)
                pos_grad_accum += step_grad
                if career_pos_grad is not None:
                    career_pos_grad += step_grad

        # v5_v2 Phase 1 — L2 anchor on positions when learnable.
        # Phase wraps every 0.97 mm at 77 GHz; we want positions to drift
        # only sub-mm at most. λ ~ 1e3 keeps ‖Δp‖ in O(0.1 mm).
        if learn_positions_lr > 0.0 and learn_positions_l2 > 0.0:
            pos_drift = model.positions - init_positions
            reg_pos = learn_positions_l2 * pos_drift.pow(2).mean()
            reg_pos.backward()
            loss_sum += float(reg_pos.item())

        # LR schedule (same as chirp-loop trainer)
        lr_scale = get_lr_scale(
            it, total_iters=num_iters,
            warmup_iters=0, warmup_factor=1.0, decay_start=200)
        for g in optimizer.param_groups:
            g['lr'] = base_lrs[g['name']] * lr_scale

        for g in optimizer.param_groups:
            clip = clip_vals.get(g['name'], 1.0)
            for p in g['params']:
                rms_clip_grad(p, clip)

        optimizer.step()

        # S4: adaptive density — split top-signal + prune bottom-signal.
        # Triggered at iter = reg_warm_iters and every reg_densify_interval
        # iters thereafter, up to reg_densify_until. Selection signal is the
        # accumulated amplitude-path position-gradient magnitude (Phase 3
        # `pos_grad_amp`).
        if (reg_densify_interval > 0
                and it >= reg_warm_iters
                and it <= reg_densify_until
                and (it - reg_warm_iters) % reg_densify_interval == 0):
            f_comb = None
            if pos_grad_accum is not None and pos_grad_accum.max() > 0:
                f_comb = pos_grad_accum.clone()  # (N,)
                # Diagnostic: cumulative top-k fraction of total ‖∇p L‖
                # in this densify window. Tells us how concentrated the
                # signal is among a few points (Fisher-imbalance probe).
                with torch.no_grad():
                    sg, _ = f_comb.sort(descending=True)
                    total = sg.sum().clamp(min=1e-30)
                    cumsum = (sg.cumsum(0) / total)
                    Nm = int(model.N)
                    snap = {'iter': int(it)}
                    parts = []
                    for frac in (0.005, 0.01, 0.05, 0.10, 0.25, 0.50):
                        k = max(1, int(frac * Nm))
                        v = float(cumsum[k - 1].item()) * 100.0
                        snap[f'top_{frac:.4f}'] = v
                        parts.append(f'top-{frac*100:.1f}%={v:.1f}%')
                    grad_concentration_log.append(snap)
                    if verbose:
                        print(f'  [grad-conc iter={it}] N={Nm}  '
                              + '  '.join(parts))
                # Reset for next interval.
                pos_grad_accum.zero_()
            if f_comb is not None:
                split_n = int(reg_densify_split_frac * model.N)
                prune_n = int(reg_densify_prune_frac * model.N)
                if split_n > 0 and prune_n > 0 and split_n == prune_n:
                    _densify_step(
                        model, optimizer, f_comb,
                        split_n=split_n, prune_n=prune_n,
                        pos_jitter_m=reg_densify_pos_jitter_m,
                        mat_jitter=reg_densify_mat_jitter,
                        rot_jitter=0.05,
                        init_raw_ref=init_raw,
                        init_normals_ref=init_normals,
                        verbose=verbose,
                    )
                    n_densify_rounds += 1
                    # Positions moved — recompute FOV mask at the *first*
                    # train sample's pose (any train pose is fine for the
                    # "is this point in FOV anywhere" estimate). Apply it
                    # as the new conservative mask.
                    apply_pose(rast, train_samples[0]['pose'])
                    new_mask = cull_gaussians(model, rast)
                    with torch.no_grad():
                        active_mask = new_mask
                        vertex_areas = torch.zeros(model.N, device=DEVICE)
                        vertex_areas[active_mask] = 1.0

        mean_tc = float(np.mean(per_sample_cc))
        history.append({
            'iter': it, 'loss': loss_sum,
            'mean_train_cc': mean_tc,
        })

        if mean_tc > best_mean_train_cc:
            best_mean_train_cc = mean_tc
            best_iter = it
            best_state = {k: v.data.clone() for k, v in model.state_dict().items()}

        if verbose and (it % 10 == 0 or it == num_iters - 1):
            elapsed = time.time() - t0
            print(f'  iter {it:4d}: loss={loss_sum:.5e}  '
                  f'mean_train_cc={mean_tc:.4f} '
                  f'(best={best_mean_train_cc:.4f}@{best_iter})  '
                  f'[{elapsed:.0f}s, n={len(train_samples)}]')

    train_elapsed = time.time() - t0

    # ── Restore best state + final test eval ──
    if best_state is not None:
        with torch.no_grad():
            for k, v in best_state.items():
                model.state_dict()[k].copy_(v)

    final_test_cc = _render_and_cart_corr(test_sample)
    final_train_ccs = [_render_and_cart_corr(s) for s in train_samples]
    final_train_mean = float(np.mean(final_train_ccs))

    # Export the final test-pose render (polar |RA| + cart |RA|) and the GT
    # for the figure pipeline. Saved as `rendered_test_ra_polar.npy` and
    # `rendered_test_ra_cart.npy`. Matches the baseline shape conventions
    # so the comparison figure can pick it up.
    apply_pose(rast, test_sample['pose'])
    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            **_render_kw)
        test_ra_polar = _ra_mag(rp_real, rp_imag)   # XRF55: 64-az single-chip RA
        test_ra_cart = polar_to_cart_torch(test_ra_polar, sample_grid)
        gt_ra_cart = polar_to_cart_torch(test_sample['gt_ra_polar'], sample_grid)

    # Compute the full RA metric set on the test frame (matches baselines).
    from mm3DPS.evaluation.utils.metrics import compute_cart_ra_metrics as _cm
    test_metrics = _cm(
        gt_ra_cart.detach().cpu().numpy().astype(np.float32),
        test_ra_cart.detach().cpu().numpy().astype(np.float32),
    )
    final_test_cart_corr = float(test_metrics['cart_corr'])
    final_test_cart_mse = float(test_metrics['mse'])
    final_test_cart_rmse = float(test_metrics['rmse'])
    final_test_cart_psnr = float(test_metrics['psnr'])
    final_test_cart_ssim = (float(test_metrics['ssim'])
                             if test_metrics['ssim'] is not None else None)

    if verbose:
        print(f'\n  [final] test  cc = {final_test_cc:.4f} '
              f'(init was {init_test_cc:.4f}, Δ {final_test_cc-init_test_cc:+.4f})')
        print(f'  [final] train mean cc = {final_train_mean:.4f}  '
              f'(best iter {best_iter})')
        print(f'  [final] elapsed {train_elapsed:.0f}s '
              f'({train_elapsed*1000/num_iters:.1f} ms/iter)')

    # ── Save ──
    if output_dir is None:
        tag = (f'train{len(train_frames)}frames_{len(train_loops)}loops'
               f'_test{test_frame}_loop{held_out_loop}')
        if test_in_train:
            tag = f'{tag}_ub'
        if use_pass2_alignment:
            tag = f'{tag}_pass2'
        if anchor_source == 'pass3_per_chirp':
            tag = f'{tag}_pass3anchor'
        if seed_frame != int(test_frame):
            tag = f'{tag}_seed{seed_frame}'
        if int(target_n) != 90000:
            tag = f'{tag}_N{int(target_n)}'
        output_dir = os.path.join(
            PROJECT_ROOT, 'mm3DPS', 'output_frame_nvs', f'{scene}_{tag}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'scene': scene,
        'train_frames': list(map(int, train_frames)),
        'train_loops': list(map(int, train_loops)),
        'test_frame': int(test_frame),
        'held_out_loop': int(held_out_loop),
        'test_in_train': bool(test_in_train),
        'num_iters': int(num_iters),
        'best_iter': int(best_iter),
        'best_mean_train_cc': float(best_mean_train_cc),
        'init_test_cc': float(init_test_cc),
        'final_test_cc': float(final_test_cc),
        # Full RA metric set on the test frame (matches baselines' metrics.json).
        'final_test_cart_corr': final_test_cart_corr,
        'final_test_cart_mse':  final_test_cart_mse,
        'final_test_cart_rmse': final_test_cart_rmse,
        'final_test_cart_psnr': final_test_cart_psnr,
        'final_test_cart_ssim': final_test_cart_ssim,
        'init_train_mean_cc': float(np.mean(init_train_ccs)),
        'final_train_mean_cc': float(final_train_mean),
        'init_train_std_cc': float(np.std(init_train_ccs)),
        'final_train_std_cc': float(np.std(final_train_ccs)),
        'n_train_samples': int(len(train_samples)),
        'edge_frames': diag['edge_frames'],
        'test_pose_mode': diag['test_pose_mode'],
        'alignment_source': 'pass-2' if use_pass2_alignment else 'pass-1',
        'anchor_source':    anchor_source,
        'seed_frame':       int(seed_frame),
        'mat_lr': float(mat_lr), 'rot_lr': float(rot_lr),
        'loss_type': loss_type,
        'target_n': int(target_n),
        'elapsed_s': float(train_elapsed),
        'reg_warm_iters': int(reg_warm_iters),
        'reg_densify_interval': int(reg_densify_interval),
        'reg_densify_split_frac': float(reg_densify_split_frac),
        'reg_densify_prune_frac': float(reg_densify_prune_frac),
        'reg_densify_until': int(reg_densify_until),
        'reg_densify_pos_jitter_m': float(reg_densify_pos_jitter_m),
        'reg_densify_mat_jitter': float(reg_densify_mat_jitter),
        'reg_densify_rounds_completed': int(n_densify_rounds),
        'learn_positions_lr': float(learn_positions_lr),
        'learn_positions_l2': float(learn_positions_l2),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(output_dir, 'history.npz'),
             iters=np.array([h['iter'] for h in history]),
             loss=np.array([h['loss'] for h in history]),
             mean_train_cc=np.array([h['mean_train_cc'] for h in history]))
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))

    # Export rendered test |RA| (polar + cart) + GT cart for figures pipeline.
    test_ra_polar_np = test_ra_polar.detach().cpu().numpy().astype(np.float32)
    test_ra_cart_np = test_ra_cart.detach().cpu().numpy().astype(np.float32)
    test_gt_cart_np = gt_ra_cart.detach().cpu().numpy().astype(np.float32)
    # Complex per-(TX,RX) range profile -- the renderer's native output.
    # Range FFT magnitude gives |RA|; inverse range FFT recovers raw ADC.
    # Shape: (n_tx, n_rx, K_range_bins), dtype complex64.
    test_rp_complex_np = (rp_real + 1j * rp_imag).detach().cpu().numpy().astype(np.complex64)
    np.save(os.path.join(output_dir, 'rendered_test_ra_polar.npy'), test_ra_polar_np)
    np.save(os.path.join(output_dir, 'rendered_test_ra_cart.npy'), test_ra_cart_np)
    np.save(os.path.join(output_dir, 'gt_test_ra_cart.npy'), test_gt_cart_np)
    np.save(os.path.join(output_dir, 'rendered_test_rp_complex.npy'), test_rp_complex_np)
    # dB / linear PNGs for the test frame (matches baseline finalize PNGs).
    _save_ra_pngs_torch(test_ra_cart_np, output_dir, 'rendered_ra', range_res,
                         f'mm3DGS test frame {test_frame} loop {held_out_loop}')
    _save_ra_pngs_torch(test_gt_cart_np, output_dir, 'gt_ra', range_res,
                         f'GT test frame {test_frame} loop {held_out_loop}')

    # ----------------------------------------------------------------
    # Per-train-frame export (supplement figure pipeline).
    # For each train_sample, render at its pose, save polar |RA| + cart |RA|
    # + GT cart + dB/linear PNGs under train_frames/frame_<F_train>/, and
    # write metrics_train.json with per-frame ra_corr (cart_corr_torch).
    # ----------------------------------------------------------------
    _save_train_frames_export(
        output_dir=output_dir,
        scene=scene,
        train_samples=train_samples,
        rast=rast,
        model=model,
        vertex_areas=vertex_areas,
        active_mask=active_mask,
        sample_grid=sample_grid,
        range_res=range_res,
        render_kwargs=_render_kw,
    )

    # Fisher-imbalance diagnostic dump (Phase A of the v5_v4 follow-on
    # plan). Captures the never-reset career ‖∇p L‖ accumulator, the
    # per-densify concentration log, and final + init positions so we
    # can offline-analyse where the live points are in 3D space.
    if career_pos_grad is not None:
        diag_path = os.path.join(output_dir, 'fisher_diagnostic.pt')
        torch.save({
            'career_pos_grad': career_pos_grad.detach().cpu(),
            'last_window_pos_grad': (pos_grad_accum.detach().cpu()
                                      if pos_grad_accum is not None else None),
            'final_positions': model.positions.detach().cpu(),
            'init_positions': init_positions.detach().cpu(),
            'concentration_log': grad_concentration_log,
            'final_test_cc': float(final_test_cc),
            'final_train_mean_cc': float(final_train_mean),
            'N': int(model.N),
        }, diag_path)
        if verbose:
            print(f'  fisher diagnostic saved to: {diag_path}')

    if verbose:
        print(f'\n  results saved to: {output_dir}')
    return results


# ---------------------------------------------------------------------------
# Per-train-frame export (supplement figure pipeline)
# ---------------------------------------------------------------------------

def _save_ra_pngs_torch(ra_cart_np, out_dir, prefix, range_res, title_prefix):
    """Save dB + linear PNGs of a Cartesian RA magnitude image.

    Mirrors the pattern used by ``mm3DPS.evaluation.utils.visualization.
    save_ra_cartesian_png`` (matplotlib, plasma cmap by default — kept as
    'hot' for visual consistency with the baseline finalize PNGs).
    """
    from mm3DPS.evaluation.utils.visualization import save_ra_cartesian_png
    for scale in ('linear', 'dB'):
        save_ra_cartesian_png(
            ra_cart_np,
            os.path.join(out_dir, f'{prefix}_{scale}.png'),
            range_res=range_res, scale=scale,
            title=f'{title_prefix} ({scale})',
        )


def _save_train_frames_export(*, output_dir, scene, train_samples, rast, model,
                              render_kwargs=None,
                              vertex_areas, active_mask, sample_grid,
                              range_res):
    """For each train_sample, render at its pose and save the supplement-figure
    artefacts (rendered/GT cart + polar + dB/linear PNGs + per-frame
    ra_corr metrics). Aggregates across train frames into ``metrics_train.json``.
    """
    train_dir_root = os.path.join(output_dir, 'train_frames')
    os.makedirs(train_dir_root, exist_ok=True)

    per_frame_records = []
    ra_corr_per_frame = []

    for s in train_samples:
        f = int(s['frame_idx'])
        loop_idx = int(s['loop_idx'])
        # Folder: when multiple loops per frame are saved, the first loop wins
        # (later loops overwrite). For the standard --train_loops 0 default
        # this is exactly one entry per train frame.
        frame_dir = os.path.join(train_dir_root, f'frame_{f}')
        os.makedirs(frame_dir, exist_ok=True)

        apply_pose(rast, s['pose'])
        with torch.no_grad():
            _rk = render_kwargs if render_kwargs is not None else dict(
                bsdf_mode='full', disabled_components=None)
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                **_rk)
            ra_polar = _ra_mag(rp_real, rp_imag)   # XRF55: 64-az single-chip RA
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
            gt_polar = s['gt_ra_polar']
            gt_cart_full = polar_to_cart_torch(gt_polar, sample_grid)

        rendered_polar_np = ra_polar.detach().cpu().numpy().astype(np.float32)
        rendered_cart_np = ra_cart.detach().cpu().numpy().astype(np.float32)
        gt_polar_np = gt_polar.detach().cpu().numpy().astype(np.float32)
        gt_cart_np = gt_cart_full.detach().cpu().numpy().astype(np.float32)

        np.save(os.path.join(frame_dir, 'rendered_ra_polar.npy'),
                rendered_polar_np)
        np.save(os.path.join(frame_dir, 'rendered_ra_cart.npy'),
                rendered_cart_np)
        np.save(os.path.join(frame_dir, 'gt_ra_polar_full.npy'), gt_polar_np)
        np.save(os.path.join(frame_dir, 'gt_ra_cart.npy'), gt_cart_np)
        # Per-train-frame complex range profile (renderer's native output).
        rp_complex_np = (rp_real + 1j * rp_imag).detach().cpu().numpy().astype(np.complex64)
        np.save(os.path.join(frame_dir, 'rendered_rp_complex.npy'), rp_complex_np)

        _save_ra_pngs_torch(rendered_cart_np, frame_dir, 'rendered_ra',
                            range_res, f'mm3DGS train frame {f} loop {loop_idx}')
        _save_ra_pngs_torch(gt_cart_np, frame_dir, 'gt_ra',
                            range_res, f'GT train frame {f} loop {loop_idx}')

        # Compute the full RA metric set via the SAME harness the baselines
        # use (mm3DPS.evaluation.utils.metrics.compute_cart_ra_metrics) so
        # numbers are bit-identical to baseline metrics.json.
        from mm3DPS.evaluation.utils.metrics import compute_cart_ra_metrics
        m = compute_cart_ra_metrics(gt_cart_np, rendered_cart_np)
        per_frame_metric = {
            'baseline': 'mm3dgs',
            'scene': scene,
            'frame': f,
            'loop_idx': loop_idx,
            'ra_corr':         m['cart_corr'],
            'cart_mse':        m['mse'],
            'cart_rmse':       m['rmse'],
            'cart_psnr':       m['psnr'],
            'cart_ssim':       m['ssim'],
            'range_profile_corr': None,
        }
        with open(os.path.join(frame_dir, 'metrics.json'), 'w') as f_:
            json.dump(per_frame_metric, f_, indent=2)
        per_frame_records.append(per_frame_metric)
        ra_corr_per_frame.append(per_frame_metric['ra_corr'])

    # Aggregate every per-frame metric across the train set.
    def _agg(key):
        vals = [r.get(key) for r in per_frame_records
                if r.get(key) is not None]
        return ({'mean': float(np.mean(vals)),
                  'std':  float(np.std(vals)),
                  'per_frame': vals}
                 if vals else None)
    agg = {
        'per_frame': per_frame_records,
        'n_train_frames': len(per_frame_records),
        'ra_corr':   _agg('ra_corr'),
        'cart_mse':  _agg('cart_mse'),
        'cart_rmse': _agg('cart_rmse'),
        'cart_psnr': _agg('cart_psnr'),
        'cart_ssim': _agg('cart_ssim'),
    }
    with open(os.path.join(output_dir, 'metrics_train.json'), 'w') as f_:
        json.dump(agg, f_, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Frame-level NVS trainer')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--test_frame', type=int, required=True)
    ap.add_argument('--train_frames', required=True,
                    help='comma-separated list, e.g. 434,435,436,437,439,440,441,442')
    ap.add_argument('--held_out_loop', type=int, default=0)
    ap.add_argument('--train_loops', default='0',
                    help='Comma-separated chirp-loop indices to train on; '
                         'default = "0" (first chirp loop only — the canonical '
                         'paper recipe). Pass "" or omit to fall back to None '
                         'which expands to all 16 loops of the cascaded radar.')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--mat_lr', type=float, default=0.01)
    ap.add_argument('--rot_lr', type=float, default=5e-3)
    ap.add_argument('--loss_type', default='mse_raw',
                    choices=['mse', 'pearson', 'mse_raw'])
    ap.add_argument('--target_n', type=int, default=20000,
                    help='Target Gaussian count after init. Default 20000 '
                         '(v5_v4 combo_jitter recipe).')
    ap.add_argument('--frame_period_s', type=float, default=0.1)
    ap.add_argument('--loop_dt_s', type=float, default=7.87e-3 / 16.0)
    ap.add_argument('--anchor_source', default='pass2_lerp',
                    choices=['pass2_lerp', 'pass3_per_chirp'],
                    help='Per-chirp pose source. pass2_lerp (default): '
                         'LERP between pass-2 F-1 and F+1 (historical). '
                         'pass3_per_chirp: load each (frame, chirp)\'s Stage 3 '
                         'refined config directly; falls back to pass2_lerp '
                         'if any per-chirp file is missing.')
    ap.add_argument('--no_pass2', action='store_true',
                    help='Opt out of pass-2 trajectory-aware alignment. '
                         'Default is pass-2 (cascaded_frame_<F>_aligned_pass2.json); '
                         'this flag falls back to pass-1 (cascaded_frame_<F>_aligned.json).')
    ap.add_argument('--reg_warm_iters', type=int, default=100,
                    help='Warmup iters before the first densify round. '
                         'Default 100 (v5_v4 combo_jitter recipe).')
    ap.add_argument('--reg_densify_interval', type=int, default=100,
                    help='S4: densify/prune every N iters after --reg_warm_iters. '
                         'Default 100 (v5_v4 combo_jitter); 0 = disabled.')
    ap.add_argument('--reg_densify_split_frac', type=float, default=0.05,
                    help='S4: fraction of top-signal points to split per round. '
                         'Default 0.05 (v5_v4 combo_jitter).')
    ap.add_argument('--reg_densify_prune_frac', type=float, default=0.05,
                    help='S4: fraction of bottom-signal points to prune per round '
                         '(must equal --reg_densify_split_frac to keep N fixed). '
                         'Default 0.05 (v5_v4 combo_jitter).')
    ap.add_argument('--reg_densify_until', type=int, default=400,
                    help='S4: stop densifying past this iter (lets the final '
                         'iters converge on a stable point set). Default 400 '
                         '(v5_v4 combo_jitter).')
    ap.add_argument('--reg_densify_pos_jitter_m', type=float, default=0.02,
                    help='S4: std (m) of isotropic Gaussian position noise for '
                         'new child points. 0.02 = 2 cm ≈ 5λ at 77 GHz.')
    ap.add_argument('--reg_densify_mat_jitter', type=float, default=0.1,
                    help='S4: std of Gaussian noise added to raw_materials for '
                         'child points (raw units).')
    ap.add_argument('--seed_frame', type=int, default=None,
                    help='Frame whose pose seeds the rasterizer (drives FOV + '
                         'RX visibility before FPS, so it determines the 90k-'
                         'point subset of pcl.npy used for training). Default = '
                         '`test_frame` so HO and UB variants share an identical '
                         'position grid (see md/frame_nvs_analysis/findings.md '
                         '§0). Pass an explicit value to reproduce the legacy '
                         'HO grid (`--seed_frame {test_frame+1}`) or to probe '
                         'sensitivity to the seed.')
    ap.add_argument('--learn_positions_lr', type=float, default=1e-5,
                    help='v5_v2 Phase 1 carryover. > 0 enables learnable '
                         'positions (added to Adam at this LR). Default '
                         '1e-5 (v5_v4 combo_jitter recipe). Gradient flows '
                         'AMPLITUDE-PATH ONLY (detach_phase=True is enforced '
                         'by the renderer; phi_carrier and n_peak are '
                         'detached). NO PHASE GRADIENTS by construction. '
                         'Set 0.0 to freeze positions.')
    ap.add_argument('--learn_positions_l2', type=float, default=100.0,
                    help='L2 anchor coefficient on (positions − init). '
                         'Default 100 (v5_v4 combo_jitter). Keeps positions '
                         'sub-mm from LiDAR seed.')
    # ── Ablation knobs (md/ablations_plan.md, Tier 1+2) ──
    # Defaults match the canonical 3DPS recipe; flipping any of these
    # exercises a single ablation axis. The runner script encodes the
    # chosen axis into --output_dir so each run lands in its own
    # mm3DPS/output_ablations/<tier>/<axis>/<config>/<scene>...
    ap.add_argument('--output_dir', default=None,
                    help='Override the auto-constructed run directory. The '
                         'ablation runner sets this to '
                         'mm3DPS/output_ablations/<tier>/<axis>/<config>/<scene_run_tag>/.')
    ap.add_argument('--no_phase_detach', action='store_true',
                    help='Tier-2 axis 7: do NOT detach n_peak / phi_carrier '
                         'before scatter — gradients flow through the '
                         'carrier phase. Forces the PyTorch Step-5 fallback '
                         'since the fused CUDA kernel assumes detached phase.')
    ap.add_argument('--no_mimo_factorization', action='store_true',
                    help='Tier-1 axis 5: disable the fused CUDA kernels for '
                         'Step-4 (BSDF) and Step-5 (range splat). Falls '
                         'through to the PyTorch path that materialises full '
                         '(M, n_tx, n_rx) BSDF and (spread, M·n_tx·n_rx) '
                         'splat tensors. ~3-10x slower per Adam step; '
                         'memory may OOM at N=20k → may need to reduce N.')
    ap.add_argument('--psf_spread', type=int, default=None,
                    help='Tier-2 axis 6: Hann PSF kernel half-width L '
                         '(default 15; ablation sweeps {5, 9, 21, 25}). Even '
                         'values are blocked downstream since they produce '
                         'L+1 bins via arange(-(L//2), L//2+1).')
    ap.add_argument('--no_cull', action='store_true',
                    help='Tier-1 axis 4a: disable the FOV / azimuth-cone cull '
                         'in init_visible_weighted (keeps every pcl point '
                         'past the 1.5 m near-field guard).')
    ap.add_argument('--no_occlusion', action='store_true',
                    help='Tier-1 axis 4b: skip the Mitsuba RX-side ray test in '
                         'init_visible_weighted (no occlusion filtering).')
    ap.add_argument('--no_cosine_resample', action='store_true',
                    help='Tier-1 axis 4c: replace cos_bore importance weights '
                         'with uniform weights at the resample stage.')
    ap.add_argument('--no_fps', action='store_true',
                    help='Tier-1 axis 4d: replace farthest-point sampling '
                         'with uniform random subsampling (seed 42).')
    args = ap.parse_args()

    train_frames = [int(x) for x in args.train_frames.split(',') if x.strip()]
    train_loops = (None if args.train_loops is None else
                   [int(x) for x in args.train_loops.split(',') if x.strip()])
    if args.psf_spread is not None and args.psf_spread % 2 == 0:
        raise SystemExit(
            f'--psf_spread must be odd (got {args.psf_spread}); '
            'arange(-(L//2), L//2+1) yields L+1 bins for even L.')
    train_frame_nvs(
        scene=args.scene,
        train_frames=train_frames,
        test_frame=args.test_frame,
        held_out_loop=args.held_out_loop,
        train_loops=train_loops,
        num_iters=args.iters,
        mat_lr=args.mat_lr, rot_lr=args.rot_lr,
        loss_type=args.loss_type, target_n=args.target_n,
        frame_period_s=args.frame_period_s, loop_dt_s=args.loop_dt_s,
        use_pass2_alignment=not args.no_pass2,
        anchor_source=args.anchor_source,
        reg_warm_iters=args.reg_warm_iters,
        reg_densify_interval=args.reg_densify_interval,
        reg_densify_split_frac=args.reg_densify_split_frac,
        reg_densify_prune_frac=args.reg_densify_prune_frac,
        reg_densify_until=args.reg_densify_until,
        reg_densify_pos_jitter_m=args.reg_densify_pos_jitter_m,
        reg_densify_mat_jitter=args.reg_densify_mat_jitter,
        seed_frame=args.seed_frame,
        learn_positions_lr=args.learn_positions_lr,
        learn_positions_l2=args.learn_positions_l2,
        output_dir=args.output_dir,
        # Ablation knobs (defaults preserve canonical recipe).
        detach_phase=not args.no_phase_detach,
        use_mimo_factorization=not args.no_mimo_factorization,
        psf_spread=args.psf_spread,
        enable_cull=not args.no_cull,
        enable_occlusion=not args.no_occlusion,
        enable_cosine_resample=not args.no_cosine_resample,
        enable_fps=not args.no_fps)
