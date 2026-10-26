# tests/: unit tests and guards

Tests of the library and of the validation code, almost all on tiny
synthetic data written by the tests themselves: no network, no data cube, no
credentials. Run them from the repository root, one module at a time on a
small machine:

```bash
python -X utf8 -m pytest -q -p no:cacheprovider tests/test_r1_guard.py
python -X utf8 -m pytest -q -p no:cacheprovider tests/test_quality_rule.py tests/test_parabolic_argmax.py
```

`tests/test_r1_guard.py` also runs as a script
(`python -X utf8 -m tests.test_r1_guard`).

## What is tested

| test modules | what they pin |
|---|---|
| `test_r1_guard.py` | rule R1: no file of `pyintertidal/` or `experiments/` except `experiments/v3_open_reserved.py` asks for the reserved RTK points; `pyintertidal.rtk.load_rtk` rejects an unauthorised caller and returns development points only |
| `test_quality_rule.py`, `test_parabolic_argmax.py` | MAREA's per-pixel quality rule (at least three looks below and three above the elevation) and the sub-grid optimum of the delay scores |
| `test_waterline_contour.py` | the ITEM/NIDEM re-implementation (`pyintertidal.waterline_contour`) on a synthetic flat, before any real site |
| `test_granadeiro2021.py` | the re-implementation of Granadeiro et al. (2021) and its R side (needs `Rscript`; skipped without it) |
| `test_benchmark_*.py` | the validation v2 harness (`experiments/benchmark/`): configuration, observation, inputs, packages, domain, water levels, truths, judges, settings, designs, every lag adapter, read-outs, scoring, inference, gates, positive control, lock, run queue, timing, the command line |
| `test_download_cube_zarr.py`, `test_zarr_angles.py`, `test_download_light.py` | the Zarr downloader and reader, with fake backend responses (no network), including band subsets |
| `test_light_to_extraction.py` | the extraction of pre-2023 scenes from the Zarr stores for v19 (`experiments/light_to_extraction.py`), on a tiny synthetic site |
| `test_scene_count_curves.py`, `test_nidem_scene_curves.py`, `test_window_curves.py`, `test_truth_diagnostics.py`, `test_compute_cost.py`, `test_paired_bootstrap.py` | the method-comparison scripts v14-v18 and the paired bootstrap |
| `test_fixedcfg_methods.py`, `test_fixedcfg_runner.py`, `test_fixedcfg_dea.py`, `test_fixedcfg_stac.py` | the competitors of the fixed-configuration tables, v19's runner, DEA Intertidal (the side-interpreter part is skipped without it) and the pre-2023 scene download |
| `test_validation_v2_truth_years.py` | `truth` on a site whose stores cover fewer years than the default |
| `test_gate_m0.py` to `test_gate_m4.py`, `test_gates_b.py` | the recorded verdicts of the development gates (they read `results/<experiment>/result.json`) |

Several of these modules (`test_window_curves.py`, `test_fixedcfg_stac.py`,
`test_light_to_extraction.py`, `test_download_light.py`) were still being
written on 2026-10-26, alongside the scripts they test, and are not in the
repository until those scripts are committed.

Helpers, not test modules: `benchmark_site.py` (a small synthetic site
written in the production layout) and `benchmark_zarr_site.py` (a synthetic
site in yearly Zarr stores). `data/stac_villaviciosa_2025-06.json` is a
recorded STAC response used without network.

## Optional dependencies

Some tests skip themselves when an optional tool or file is absent: `Rscript`
(Granadeiro), the DEA side interpreter (`venv-dea` or `MAREA_DEA_PYTHON`),
eo-tides with the EOT20 files, a local data cube, `fcntl` (Linux only), or a
laptop copy of server outputs. `V2_CONTROL_FULL=1` enables the full positive
control. A few read local data that a fresh clone does not have and fail
without it: in `test_gate_m0.py`, `test_simulator_null_marginals` needs the
stores under `data_v4/store/` named in `configs/m0.yaml`, which were never
versioned.

A bare `pytest` from the repository root would also collect the `test_*.py`
scratch files under `experiments/prototypes/`; name the folder or the files
instead.
