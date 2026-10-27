# Paper figure & table scripts

Each script regenerates a figure or table that appears in the paper.
Scripts assume they are invoked from the repo root with the mm3DPS
package importable, the six paper checkpoints under `checkpoints/`, and
the per-baseline `results/<scene>/metrics.json` files produced by
`baselines/<name>/runner/run_phase4.py`.

## Figures

| Output PDF                                              | Script                                                                |
|---------------------------------------------------------|-----------------------------------------------------------------------|
| `figs/teaser_3dps_v9b.pdf`                              | `generate_fig_teaser_v9.py` (uses helpers below)                      |
| `figs/pipeline_3dps.pdf`                                | `generate_fig_pipeline_3dps_v2.py`                                    |
| `figs/training_ra_comparison.pdf`                       | `generate_fig_training_ra.py`                                         |
| `figs/supplement/supplement_seq_<scene>.pdf` (×6)       | `generate_fig_training_ra_supplement.py`                              |
| `figs/crp_adc/fig_crp_adc_heatmaps_linear.pdf`          | `generate_fig_crp_adc.py --plot heatmaps_linear`                      |
| `figs/crp_adc/fig_crp_adc_iq_traces.pdf`                | `generate_fig_crp_adc.py --plot iq_traces`                            |

Teaser helpers (called from `generate_fig_teaser_v9.py` or used to pre-
render panels written to disk):

- `prep_teaser_panels_v4.py`
- `render_pipeline_panel_a.py`
- `render_colored_bg_panels.py`
- `render_nvs_grid_panels.py`

`fig_common.py` is a shared style/layout helper used by every figure.

## Tables (.tex)

| Output .tex                                             | Script                                |
|---------------------------------------------------------|---------------------------------------|
| `tables/test_ra_results.tex`                            | `generate_tables.py`                  |
| `tables/train_ra_results.tex`                           | `generate_tables.py`                  |
| `tables/crp_adc_results.tex`                            | `generate_crp_adc_paper_table.py`     |
| `tables/crp_adc_supplement/*.tex`                       | `generate_crp_adc_table.py`           |
| `tables/ablations.tex`                                  | `generate_ablation_table.py`          |

## Inputs each script expects

- `checkpoints/<scene>/best_model.pt` — the six paper checkpoints (shipped).
- `checkpoints/<scene>/metadata.json` — per-scene metadata + reproduction command.
- `baselines/<name>/results/<scene>/metrics.json` — produced by each baseline's
  `runner/run_phase4.py`. Generate these before running the comparison
  tables/figures.
- `output/crp_adc_eval/<scene>/{crp,adc}_{pred_corr,gt}.npy` — produced by
  `mm3DPS.evaluation.eval_crp_adc` for the CRP/ADC product-agnostic
  evaluation. Generate this with the canonical CRP/ADC eval before
  running `generate_fig_crp_adc.py`.

## Notes

- Only paper-figure scripts are shipped. High-resolution and ablation-
  development variants (`*_highres.py`, `prep_teaser_panels_v3.py`, the
  Blender video pipeline, `diag_phase_substitution_experiment.py`) are
  not part of the release.
- The materials-visualisation figures (`figs/materials/*.pdf`) are
  produced by an offline Open3D / Mitsuba viewing pipeline that is not
  bundled in this release.
- All scripts read paths via CLI flags; defaults assume the layout
  described above. Pass `--help` for the full set of overrides.
