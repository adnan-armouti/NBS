# Does the differentiable renderer execute the same path on both datasets?

Audited at `6486a02` (`runtime/2x2`), on `mmsd-h100`, RT-Pose 176 against HuPR 256.

## Why this audit exists, and the claim it is checking

In a runtime summary I wrote that the renderer "is two different kernels depending on
the dataset", citing `doppler_plane_sorted` on RT-Pose and `static_scene_frames` on
HuPR. **That sentence was wrong and I am retracting it.** What I had measured was a
different *distribution of time* over the same kernels; I described it as different
kernels. Both kernels run on both datasets — `static_scene_frames_forward` is 0.967
ms/step on RT-Pose and `doppler_plane_sorted_forward` is 2.094 ms/step on HuPR, in the
same two profiles I was quoting from.

The design requirement is that it is the same renderer, the same kernels and the same
functions for both datasets. This audit tests that requirement directly rather than
taking my word for it, with three independent instruments, and reports everything that
differs regardless of which layer it sits in.

## Method

Three instruments, all on the same commit, same node, same flags as the production fit
(each sequence's own `trainer_argv.json`, replayed verbatim).

1. **CUDA kernel set and per-step launch counts** — `torch.profiler` over 20 captured
   steps, `MM4DPS_PROFILE_STACK=1 --profile-steps 20`. Because the training step is
   captured into a CUDA graph and replayed, the kernel list is the *complete* device-side
   path, not a sample.
2. **Python function set and call counts** — `sys.settrace` with a global hook that
   records `call` events for code objects under `mm4DPS/` and declines line tracing
   (returning `None`), so the cost is one callback per function call. One run per dataset
   at `--epochs 1`. The renderer's Python path executes exactly once, during graph
   capture — replay calls no Python at all — so a function-level trace of one run is the
   whole Python path rather than a sample of it.
3. **Static branch audit** — every dataset-, radar- or array-conditional predicate in
   `forward_render.py`, `velocity_field.py`, `rasterizer.py`, `data.py`, `signal_model.py`.

Reproduce:

```
# instrument 1
/tmp/node_prof.sh on_rt176 rtpose 176 0 ; /tmp/node_prof.sh on_hu256 hupr 256 1
python /tmp/attrib2.py /tmp/prof2x2/on_rt176/prof.log      # bucket by kernel signature
python /tmp/top2.py   /tmp/prof2x2/on_rt176/prof.log render: unattributed
# instrument 2
/tmp/node_trace.sh tr_rt176 rtpose 176 2 ; /tmp/node_trace.sh tr_hu256 hupr 256 3
cut -f2,3 trace.tsv | sort   # then comm -23 / -13 the two
```

Scripts are archived beside this document in
`mmSD/artifacts/renderer_parity_audit/`.

## Result 1: the renderer's CUDA kernels are identical

Twelve project kernels (`mm25v5::`) execute. **All twelve execute on both datasets.
None is unique to either.** `comm -23` and `comm -13` of the two sorted kernel lists are
both empty.

| kernel | RT-Pose 176 launches/step | HuPR 256 launches/step |
|---|---|---|
| `doppler_plane_sorted_forward_kernel` | 1 | 1 |
| `doppler_plane_sorted_backward_kernel` | 1 | 1 |
| `doppler_plane_amplitudes_sorted_kernel` | 1 | 1 |
| `doppler_plane_grad_transpose_kernel` | 1 | 1 |
| `static_scene_frames_forward_kernel` | 1 | 6 |
| `static_scene_frames_backward_kernel` | 1 | 6 |
| `bsdf_step4_forward_kernel` | 2 | 3 |
| `bsdf_step4_backward_warp_kernel` | 2 | 3 |
| `rae_spectrum_input_kernel` | 2 | 2 |
| `rae_spectrum_input_backward_kernel` | 2 | 2 |
| `rae_plane_term_forward_kernel` | 1 | 1 |
| `rae_plane_term_backward_kernel` | 1 | 1 |

Ten of the twelve have identical launch counts. The two that differ are explained by
configuration, not by code:

**`static_scene_frames`, 1 against 6.** `forward_render.py` caps a launch at
`_SCENE_MAX_FRAMES = 8` frames, because the kernel holds that many frame accumulators in
registers (`float2 accum[SCENE_MAX_FRAMES]`), and the wrapper slices the batch:

```python
outs = [_StaticSceneFrames.apply(..., frame_gains[start:start + _SCENE_MAX_FRAMES].t()...)
        for start in range(0, frame_gains.shape[0], _SCENE_MAX_FRAMES)]
```

RT-Pose's `trainer_argv.json` passes `--frame-batch 4`, so `ceil(4/8) = 1` launch, times
one board. HuPR passes no `--frame-batch` and takes the default 24, so `ceil(24/8) = 3`
launches, times two boards (horizontal and vertical) = 6. The comment at that line
already says "a larger batch (HuPR's 24) goes through in slices". Same loop, same
kernel, different slice count.

Per frame rendered, RT-Pose is the *more* expensive of the two — 0.967 ms for 4 frames
on one board is 0.242 ms a frame, against 4.597 ms for 24 frames on two boards, 0.096 ms
a frame — which is what a 192-channel cascade against an 8-element ULA should look like.

**`bsdf_step4`, 2 against 3.** Both call sites are inside the single function
`rasterizer.render_factorized` (its `_fast` geometry-cache branch at `rasterizer.py:956`
and its general branch at `:1209`). The Python-side count is 1:1 with
`render_factorized` itself — 913 calls on RT-Pose, 1802 on HuPR — so the ratio tracks the
number of boards rendered. The captured step's 2 against 3 rather than 2 against 4 I did
not fully account for; it is bounded at 0.95 ms against 1.01 ms, so nothing in this
audit's conclusions turns on it. **This is the one number here I am not able to close.**

## Result 2: kernel-*variant* differences are not code-path differences

Across all kernels of any origin, 45 distinct kernels appear on RT-Pose and 41 on HuPR.
Every entry unique to one side is a PyTorch or Inductor *variant* of an op that runs on
both, selected by tensor shape or dtype:

| RT-Pose side | HuPR side | what selects it |
|---|---|---|
| `indexFuncLargeIndex<complex<float>,…>` | `indexFuncSmallIndex<complex<float>,…>` | PyTorch's index kernels switch on index size |
| `reduce_kernel<256, 2, …sum_functor…>` | `reduce_kernel<512, 1, …>` | the reduction's block config, chosen by shape |
| `triton_red_fused__unsafe_view_abs_add_clamp_clone_cos…` | `triton_poi_fused__unsafe_view_abs_add_clamp_clone_cos…` | Inductor schedules the *same* graph as a reduction or a pointwise kernel by shape |
| `_cuda_scatter_gather_internal_kernel<false, OpaqueType<1>,…>` | `…OpaqueType<2>,…` | element size in bytes |

This distinction matters and is the one I collapsed in the sentence being retracted:
a different kernel *variant* chosen by the dispatcher from the same Python op is not a
different code path. The one genuine exception is treated in Result 4.

## Result 3: the Python function sets, and every difference in them

RT-Pose executes **330** distinct functions under `mm4DPS/`, HuPR **321**, with **316 in
common**. All nineteen differences are listed below — none of them is in the renderer.

### RT-Pose only (14)

| function | layer | why |
|---|---|---|
| `data.py:182 tda2_frame_count` | data loader | RT-Pose's cascade records TDA2 |
| `data.py:197 CascadeRecording.__init__` | data loader | " |
| `data.py:203 <genexpr>` | data loader | " |
| `data.py:208 bytes_per_device_frame` | data loader | " |
| `data.py:212 CascadeRecording.frame_words` | data loader | " |
| `data.py:225 decode_tda2_frame_torch` | data loader | " |
| `data.py:248 cascade_calibration` | data loader | the cascade ships a per-channel calibration; HuPR does not |
| `data.py:277 apply_cascade_calibration` | data loader | " |
| `__init__.py:616 Dataset.root` | paths | reached only from the cascade reader |
| `__init__.py:622 Dataset.sequence_dir` | paths | " |
| `signal_model.py:577 doppler_bin_mps` | measured-side convention | see below |
| `velocity_field.py:1513 downstream_view.view` | gain calibration | see below |
| `velocity_field.py:1645 measured_cube_channels_of` | gain calibration | reached only from that branch |
| `velocity_field.py:4535 calibrate_person_level.ratio` | gain calibration | see below |

### HuPR only (5)

| function | layer | why |
|---|---|---|
| `data.py:76 find_archive` | data loader | HuPR ships sequences as archives |
| `data.py:970 decode_dca1000_frames_torch` | data loader | HuPR's boards record DCA1000 |
| `forward_render.py:1472 profiles_to_rae_magnitude` | gain calibration | 10,200 calls, see below |
| `forward_render.py:1492 profiles_to_rae_spectrum_reference` | gain calibration | called 1:1 by the above |
| `velocity_field.py:4632 calibrate_person_level.ratio` | gain calibration | see below |

Ten of the nineteen are the **ADC decoder**: the two datasets store raw captures in
different file formats (TDA2 against DCA1000) and the cascade ships a calibration table.
A reader cannot be format-agnostic, and this is upstream of the renderer — it produces
the measured target, then stops.

### Positive control: the renderer's own functions all run on both

| function | RT-Pose calls | HuPR calls |
|---|---|---|
| `unit_profiles` | 2 | 2 |
| `frame_components` | 75 | 300 |
| `assemble_static_profile_frames_cuda` | 6 | 12 |
| `assemble_static_profile_cuda` | 601 | 2400 |
| `sorted_doppler_order` | 299 | 1200 |
| `render_factorized` | 913 | 1802 |
| `profiles_to_rae_spectrum_fused` | 464 | 252 |
| `rae_terms_of` | 232 | 126 |
| `static_losses` | 6 | 6 |

Every count difference is a multiple of frames times boards times frame-batch. Nothing
in the renderer is present on one side and absent on the other.

## Result 4: the three real asymmetries, in the layers they are in

None is in the renderer. Two are deliberate and documented in the code; one is mine.

### 4a. The person-level gain calibration takes a different branch per dataset

**This is the substantive finding, and it is not what I originally claimed.**

`calibrate_person_level` (`velocity_field.py`, once per run, under `torch.no_grad()`,
before training) has three modes. RT-Pose takes the `peak-doppler` + `downstream`-view
branch at `:4506`; HuPR falls through to the plain `peak-doppler` branch at `:4553`,
which is why `profiles_to_rae_magnitude` runs 10,200 times on HuPR and never on
RT-Pose. Each branch defines its own nested `ratio`, which is why two `ratio` closures
appear in the diff.

The gate is `downstream_view()`, and it states its own reason:

```python
def downstream_view():
    """... or None when the consumer has no absolute level (HuPR: its input is
    min-max normalised per map). RT-Pose: the HRRadarPose adapter's untapered polar
    transform on the TI virtual-array table ..."""
    if current_dataset().name != "rtpose":
        return None
```

So: RT-Pose's downstream consumer has an absolute level and the person's gain can be
calibrated in the consumer's own view; HuPR's consumer min-max normalises every map, so
there is no absolute level to calibrate against, and the calibration uses the trainer's
own RAE peak-Doppler view instead. `args.person_level_calibration_view` is
`downstream` in *both* sequences' argv — the asymmetry comes entirely from this return.

What this does and does not mean:

- It is **not** the renderer, and not a gradient path. It runs once, under `no_grad`,
  and is frozen afterwards.
- It **does** set the initial `log_area_scale`, so it changes what the fit starts from,
  and therefore the result.
- It is a deliberate consequence of a property of HuPR's downstream consumer, not an
  accident of implementation. Whether "the same functions for both datasets" should
  extend to the pre-training calibration is a call for you, not me. If it should, the
  options are to use the trainer's own view on both (drop the `downstream` branch) or to
  give HuPR an absolute-level view — the second reopens a question the docstring says
  was settled against.

### 4b. `doppler_bin_mps`, the Doppler band's definition

In `signal_model.py:614`:

```python
if radar.name in HUPR.names:
    mask[OFFICIAL_CENTRAL_DOPPLER_SLICE] = True
else:
    per_bin = doppler_bin_mps(radar)
    lo = max(0, center + int(np.floor(MOTION_BAND_MPS[0] / per_bin)))
    ...
```

HuPR uses the released preprocessing's fixed central Doppler slice; RT-Pose derives the
band from `MOTION_BAND_MPS` in physical units. This is the measured-side convention —
which bins the target is formed over — and matching HuPR's released pipeline is the
point of it. Not the renderer.

### 4c. The `compact` branch in the RAE transform — this one is mine

`forward_render.py:1927`:

```python
compact = _RAE_C2 and len(occupied_el) < n_el and real.is_cuda and _RAE_INPUT == "direct"
```

Taken on the cascade, whose elevation grid is sparse (per-row occupancy
16/0/16/0/0/16/86, so fewer occupied rows than `n_el = 8`); not taken on HuPR, whose
rows are all occupied. When taken, the azimuth FFT runs over the occupied columns only
and one `index_copy_` un-compacts into a zero-padded buffer before the elevation FFT:

```python
angle = torch.fft.fft(grid, dim=-2)
full = torch.zeros((*angle.shape[:-1], n_el), ...)
full.index_copy_(-1, occ_index, angle)
spectrum = torch.fft.fft(full, dim=-1)
```

The same function is called, the same `rae_spectrum_input_kernel` runs, and the values
are identical either way — the skipped columns are exactly zero. But it **is** a
conditional whose taken branch depends on the array, and I introduced it (item C2, worth
−0.85 ms on a 4090). Its `index_copy_` now costs **0.63 ms/step** on the H100, and the
saving it buys there has not been measured since. It should be re-measured; if it is no
longer positive, removing it both recovers time and removes the asymmetry. This is the
one item in this audit that I would change on runtime grounds alone.

**Resolved 2026-09-23: keep it.** Measured on both machines (RT-Pose 176, two
replicates per arm): C2 saves **0.58 ms/step on a 4090** and costs 0.22 ms/step on the
H100 at the production frame batch. Removing it would trade the machine that is short of
headroom for the one that has plenty, so the branch stays and the asymmetry is now
justified by a current measurement on both machines. See RUNTIME_NEXT_STEPS.md, Step 3.

It sits in the loss-side RAE transform, not the renderer.

## Summary

| claim | verdict |
|---|---|
| the renderer uses different kernels per dataset | **false** — 12 project kernels, all 12 on both, none unique |
| the renderer uses different functions per dataset | **false** — 316 of 330/321 functions shared; no renderer function differs |
| the renderer's kernel launch counts differ | **true, and config-explained** — `--frame-batch` 4 against the default 24, `_SCENE_MAX_FRAMES = 8`, one board against two |
| some *kernel variants* differ | **true, and not a code path** — dispatcher and Inductor picking by shape and dtype |
| the ADC decoder differs | **true and necessary** — TDA2 against DCA1000 |
| the person-level gain calibration differs | **true, deliberate, documented** — HuPR's consumer has no absolute level (4a) |
| the Doppler band's definition differs | **true, deliberate** — HuPR's released slice (4b) |
| one conditional in the RAE transform differs | **true, mine, and worth removing** (4c) |

## Limits of this audit

State them so they can be checked:

- Both traced runs used `--epochs 1`. Anything reachable only from a longer run, a
  different `--eval-every`, or a flag these sequences do not pass is out of scope.
  `gradient_probe` did not execute in either run, so its two `profiles_to_rae_magnitude`
  call sites are not covered.
- The tracer records Python `call` events only. The C++ and CUDA layers are covered by
  the kernel diff instead, not by the trace.
- HuPR's ordered call log hit its 400,000-entry cap. The **counts** are complete; only
  the ordered prefix is truncated, and no conclusion here rests on ordering.
- Both datasets were audited on one sequence each (176 and 256). A sequence whose
  geometry changes `len(occupied_el)`, the frame batch or the board count would shift the
  launch counts, not the kernel or function sets.
- `bsdf_step4`'s 2-against-3 per-step launch count is accounted for only to the board
  count, not exactly.
