# experiments/: the scripts of the MAREA project

Every script answers one question, stated at the top of its docstring, and
says there what it reads, what it writes and how to run it. Run them from the
repository root as modules:

```bash
python -X utf8 -m experiments.<name> [arguments]
```

They import the library (`pyintertidal/`); the library never imports them.
What each output file contains is described in
[`../docs/products.md`](../docs/products.md).

To map a site you need no script here: `pyintertidal.marea.reconstruct`
makes the MAREA product from a downloaded cube (see the
[repository README](../README.md#minimal-examples), example 3). The scripts
below validate the method and compare it with others.

The folder holds three groups: validation v2 (in progress), the method
comparison on the existing 10 m inputs, and the development record that led
to the method.

## 1. Validation v2 (in progress)

The factorial validation of MAREA against state-of-the-art lag and elevation
methods, under identical inputs: [`../docs/validation_v2_spec.md`](../docs/validation_v2_spec.md).

| file | role |
|---|---|
| [`validation_v2.py`](validation_v2.py) | the only entry point (command line): `fetch`, `inputs`, `check-env`, `prepare`, `truth`, `pilot`, `plan`, `status`, `gate`, `control`, `lock`, `run`, `score`, `timing`, `diag time`, `extract` (and `report`, a placeholder until milestone V6); its docstring lists every command and option |
| [`benchmark/`](benchmark/README.md) | the harness behind those commands: the observation, water levels, input packages, truths, settings, designs, lag methods, run queue, scoring, inference, lock |
| `zarr_cubes.py` | the format of the validation cubes (one Zarr store per site and year), their reader and checks |
| `download_cube_zarr.py` | downloads those stores from the Copernicus Data Space Ecosystem with openEO (nine bands, plus the sun and view angles) |
| `download_cube.py` | downloads the netCDF cubes of the Westerschelde, Wadden Sea and Ems-Dollard sites on the 10 m validation grid |
| `validation_grid.py` | the grid switch of the validation runs (10 m or 20 m) and the spatial thinning used by the scoring |
| `v13_benchmark.py` | the command line of the earlier benchmark design (`docs/v13_benchmark_spec.md`), kept because validation v2 reuses its commands |

A typical start, on any machine:

```bash
python -X utf8 -m experiments.validation_v2 check-env      # lists every input and how to obtain it
python -X utf8 -m experiments.validation_v2 status
```

The spec orders the stages (the lag benchmark is scored before any
elevation score); `lock` refuses a working tree with uncommitted code; the
authors run the server stages (spec §13).

## 2. Method comparison on the existing 10 m inputs

Questions put by the authors on 2026-10-20 and 2026-10-26, to be answered
from the 10 m inputs and products of MAREA and its competitors already made
for each study site (not in the repository; see
[`../docs/products.md`](../docs/products.md), sections 3, 4 and 8). Outputs
go to `results/<folder>/` (not versioned while the work is in progress).

| script | question | output folder |
|---|---|---|
| `v14_scene_count_curves.py` | How does MAREA's elevation error change with the number of scenes, and is it their number or their screening that matters? | `results/scene_count_curves/` |
| `v15_nidem_scene_curves.py` | The same curve for NIDEM, on exactly MAREA's scene subsets | `results/scene_count_curves/` |
| `v16_window_curves.py` | MAREA's error against the length of the time window | `results/window_curves/` |
| `v17_truth_diagnostics.py` | Truth-side diagnostics of the 10 m production MAREA and NIDEM products: bias and error by 3-km section and by elevation third, and whether each method's own uncertainty is calibrated | `results/truth_diagnostics/` |
| `v18_compute_cost.py` | What MAREA and NIDEM cost to run on equal inputs (wall time, CPU time, memory), and what each needs as input | `results/compute_cost/` |
| `v19_fixed_config_tables.py` + [`fixedcfg/`](fixedcfg/README.md) | Every elevation method under one fixed external configuration, per site and number of screened images | `results/fixedcfg/` |

Modules these scripts reuse:

| file | role |
|---|---|
| `v7_waterline_contour.py` | runs NIDEM (`pyintertidal.waterline_contour`) at every site, as published and on MAREA's record; writes `products_*/nidem*/` |
| `v8_granadeiro2021.py` | runs the re-implementation of Granadeiro et al. (2021) (`pyintertidal.granadeiro2021`, R side in `pyintertidal/r/`) |
| `v9_method_comparison.py` | the elevation comparison of every method on the same pixels, against the same truth, with the same metrics |
| `v10_validation_report.py` | the tables of the earlier validation report, from one run's results |
| `v0_vaklodingen_fetch.py`, `v1_vaklodingen_validate.py` | fetch the Rijkswaterstaat Vaklodingen sheets and judge elevation maps against them |
| `paired_bootstrap.py` | paired cluster bootstrap of the elevation metrics |

Not yet committed on 26 October 2026: `v16_window_curves.py`,
`light_to_extraction.py` (the pre-2023 scenes of the Dutch sites from the
light CDSE Zarr stores, as v19's extra extraction) and `download_cube_stac.py`
(the same scenes from free public archives, an alternative route).

Each of these scripts records, next to its outputs, the git commit and the
sha256 of its own source and of the inputs it read; v15 and v19 also check
the sha256 of v14 before reusing it.

## 3. The reserved RTK points

`v3_open_reserved.py` opens the reserved part of the RTK survey. Only the
authors run it, and every opening is logged to `results/v3_aperturas.jsonl`:
one run so far, on 2026-08-21, which scored four development maps made
before the current MAREA products (`docs/PHASE_LOG.md`). The log records
that this run consumed the reserved set, so the reserved points are no longer
an untouched blind test. Validation v2 uses the development points only.
`tests/test_r1_guard.py` fails if any other
file of `pyintertidal/` or `experiments/` asks for those points. Do not move
or rename this file: the guard names it by its path.

## 4. The development record (August to October 2026)

The scripts that built and tested the method before validation v2, kept as
the record of how each choice was made. Each one states its question and its
pre-registered criterion in its docstring and wrote its verdict to
`results/<name>/result.json`; the dated account is
[`../docs/PHASE_LOG.md`](../docs/PHASE_LOG.md) and the map of every line of
inquiry is [`../docs/INVESTIGATION_MAP.md`](../docs/INVESTIGATION_MAP.md).
Where they overlap with validation v2, validation v2 supersedes them.

| series | scripts | what they test |
|---|---|---|
| m (the delay estimator, gate by gate) | `m0_baseline`, `m1_geometry`, `m2_alias`, `m2_gate_sim`, `m2_real`, `m3_gate_sim`, `m4_gate_sim` | baseline structure; estuary geometry from the archive; which constituents a sun-synchronous archive can resolve; recovery of a planted delay and gain in simulation (gain is not identifiable from wet/dry data); signal in the real archive against a matched null; a planted boundary error; the distributed-tide-gauge operator |
| b (product layers) | `b1_bathymetry`, `b2_sigma`, `b5_gate_sim`, `b5_v3_verdict`, `b6_hydraulic_dem`, `b7_incertidumbre` | the elevation product with the operator; how much of the fitted width is topography; separate flood and ebb delays; spill and ponding layers; per-pixel uncertainty from a calibrated twin |
| p (external sites and exploration) | `p0_download_*`, `p1_swot_coverage`, `p4_site`, `p4_scheldt`, `p5_rmse_aoi`, `p6_terneuzen_comparison`, `p7_gauge_table`, `p8_bracket_regions`, `p9_parametric_lag`, `p9b_rank_slope`, `p9c_two_curves`, `p10_linear_vs_free_aoi`, `p11_ndwi_depth_curve`, `p11b_blind_band`, `p12_extended_pixel_model`, `p13_sigma_uncertainty` | downloads; SWOT coverage; delay profiles at external estuaries; elevation against EMODnet with and without the delay; levels against held-out gauges; a linear delay law by three routes; NDWI as a depth gauge; a six-parameter pixel model; the Cramér-Rao uncertainty |
| c (comparison with Granadeiro et al. 2021) | `sota_granadeiro`, `c0_granadeiro_extract`, `c1_compare_elevation`, `c2_boundary_variants`, `c2b_bootstrap`, `c4_granadeiro_gauges`, `c5_validation_matrix` | the method implemented as published, and compared on the development data |
| v (earlier validation) | `v2_granadeiro_map_figures`, `v4_lag_external`, `v5_rtk_onsite`, `v6_lidar_external`, `v11_local_lag`, `v12_scene_rule_choice` | the figures and tables of the first validation report; a local-lag diagnostic; the truth-free choice of the scene record |
| w (waterline methods) | `w0_waterline_stack`, `w0b_waterline_composites`, `w1_waterline_validate` | sub-pixel waterline stacking and composites against the RTK points |
| e, f, g (side lines) | `e0_escalda_prep`, `e1_download_gauge_sites`, `f0_icecreams_train`, `f1_icecreams_scene`, `g0_gan_dataset`, `g1_gan_train`, `g2_gan_inpaint`, `tidegan` | the Scheldt preparation and gauge-site cubes; a reproduction of the ICE CREAMS vegetation model; a generative model for cloud gaps (outputs flagged as synthetic) |
| campaign | `make_cells_v2`, `campaign_north_v1`, `campaign_north_v2` | 130 terrain-following cells of the north coast of Spain, run without inspection |
| figures | `paper_fig_*`, `paper_style`, `fetch_true_colour`, `onepixel_likelihood_figure`, `rebuild_reference_map`, `rebuild_site_masks` | figures of the manuscript in preparation and the rasters they need |

Some of these scripts read inputs that are not in the repository (cached
cubes, sealed stores under `data_v4/`); they are kept to be read, not
necessarily to be re-run.
