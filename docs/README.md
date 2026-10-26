# docs/: method, validation and products

## Documents

| file | what it is | state |
|---|---|---|
| [`METHOD.md`](METHOD.md) | the full technical method: notation, NDWI and the clear-sky mask, the scene screening, the boundary tide, the pixel model, the profiled Bernoulli likelihood of the delay and its constraints, the smoothing to pixels, the per-pixel inversion and its guards, emersion time and hypsometry, why the tidal gain cannot be estimated, the uncertainty `sigma`, sampling limits, the parameter table with file and line, and Mermaid diagrams | rewritten on 2026-10-26, every formula checked against the code |
| [`VALIDATION_DESIGN.md`](VALIDATION_DESIGN.md) | the validation design in short, with diagrams: questions, sites, the independent checks, Stage L (delays against in-estuary gauges), Stage E (elevation against surveys), the fixed-configuration comparison, the methods compared, the metrics with formulas; no results | 2026-10-26; **the validation is in progress** |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | the code: the modules of `pyintertidal` and how they depend on each other, the data flow from download to product files, the experiments layer | 2026-10-26 |
| [`GLOSSARY.md`](GLOSSARY.md) | plain definitions of the terms a non-specialist meets | 2026-10-26 |
| [`validation_v2_spec.md`](validation_v2_spec.md) | the specification of validation v2: questions, sites and inputs, settings, competing methods, experiment plan, lag benchmark, elevation metrics, statistics, gates, report, code layout, server orchestration, budget, milestones, decisions | revision 2 (2026-10-09) with the authors' decisions of 2026-10-13 to 2026-10-15 at the top; **the validation is in progress** |
| [`v13_benchmark_spec.md`](v13_benchmark_spec.md) | the controlled benchmark v13, whose harness validation v2 reuses; v2 supersedes it wherever they differ | 2026-10-08 |
| [`products.md`](products.md) | every generated product: contents, units, vertical datum, the script and command that make it; schema tables (type, shape, units) of the MAREA product | 2026-10-26 |
| [`one_pixel_marea_example.md`](one_pixel_marea_example.md) ([Spanish](one_pixel_marea_example.es.md)) | one real Villaviciosa pixel through MAREA: the clocks, the likelihood step by step, the decision and the inversion, from `notebooks/pixel_walkthrough_villaviciosa.ipynb` | development state of September 2026 (it still shows the 10-minute detection threshold, later removed) |
| [`tide_adjustment_comparison.md`](tide_adjustment_comparison.md) ([Spanish](tide_adjustment_comparison.es.md)) | the tide adjustment judged by tide gauges and bathymetry, MAREA against Granadeiro et al. (2021), at four estuaries | development comparison of 2026-09-17; validation v2 supersedes it for any conclusion |
| [`prior_art_granadeiro.md`](prior_art_granadeiro.md) | Granadeiro et al. (2021), field by field, against MAREA | 2026-08-19 |
| [`INVESTIGATION_MAP.md`](INVESTIGATION_MAP.md) | every line of inquiry of the project, with its outcome and where its record lives | development record |
| [`PHASE_LOG.md`](PHASE_LOG.md) | the dated diary of the development phases (plan v4): what was done, gate verdicts, decisions | internal working notes, kept because the code cites them |

How they connect: `METHOD.md` is the reference for what the code computes;
`products.md` for what it writes; `ARCHITECTURE.md` for where the code
lives; `VALIDATION_DESIGN.md` summarizes, and `validation_v2_spec.md`
specifies in full, how the method is being validated; `GLOSSARY.md` defines
the terms. The worked example and the comparison are development
documents that show the method on real data; the diary and the map record
how the method was reached. The manuscript in preparation is not in this
repository.

## Notebooks (`../notebooks/`)

| notebook | what it is | state |
|---|---|---|
| `validation/00_explore_zarr_cubes.ipynb` | opens one window of the Zarr validation cubes (`experiments.zarr_cubes.open_cube`) and looks at it: dates and bands, an RGB, the water indices of one date, the SCL and CLD cloud masks, the time series of one pixel and a quick-look grid of several dates; it reads one date, one pixel or a decimated view at a time | committed, no outputs stored |
| `01_fixed_config_n_tables` | the fixed-configuration tables of `experiments/v19_fixed_config_tables.py`, by number of images | in preparation |
| `validation/02_lag_benchmark.ipynb` | the lag benchmark of validation v2 (the delay against the in-estuary tide gauges) | in preparation |
| `validation/03_scene_count_curves_marea_nidem.ipynb` | the scene-count curves of MAREA and NIDEM (`experiments/v14_scene_count_curves.py`, `experiments/v15_nidem_scene_curves.py`) | in preparation |
| `pixel_walkthrough_villaviciosa.ipynb` | the one-pixel walkthrough behind [`one_pixel_marea_example.md`](one_pixel_marea_example.md); run it with the repository root as working directory | development state of September 2026 |

The notebooks in preparation are being written (October 2026) and are not in
this repository yet: they will be added with the results of the validation,
once it is complete.

## Figures (`docs/figures/`)

Several figures show the development state of September 2026, before the
3-km sections replaced the quantile bands and before the detection threshold
and the adoption gate were removed; their documents say so where it matters.

| file | alt text | used in |
|---|---|---|
| `figures/doc/sites_map.png` | Map of western Europe with the five study sites, and a table of each site's elevation check, delay check and tide at the mouth. | README, `VALIDATION_DESIGN.md` |
| `figures/doc/tide_delay_concept.png` | A synthetic tide at the mouth and 60 minutes later inside the estuary, with the Sentinel-2 passes; the looks of one flat against both levels. | README, `METHOD.md` |
| `figures/doc/lag_profile_villaviciosa.png` | The fitted delay of each 3-km section of the Villaviciosa product, with its bootstrap fits and interval, and the delay applied to each pixel. | README, `METHOD.md` |
| `figures/doc/flooding_curve.png` | One synthetic pixel's looks as NDWI against the water level, with the fitted flooding curve; the grid search; a pixel left empty by the quality rule. | README, `METHOD.md` |
| `figures/doc/product_maps_villaviciosa.png` | Four maps of the Villaviciosa product: elevation, sigma, distance from the mouth and the delay of each pixel. | README, `products.md` |
| `figures/doc/product_distributions.png` | The Villaviciosa product as distributions: elevations, the hypsometric curve, sigma, and elevation along the channel. | `products.md` |
| `figures/doc/sentinel_sampling.png` | One month of a synthetic tide with every Sentinel-2 acquisition at Villaviciosa, and the tide levels at all times against those of the kept looks. | `METHOD.md` |
| `figures/doc/affine_identifiability.png` | Two worlds, a gain of 1.0 and 1.2 over pixels at 1.0 and 1.2 m, that give the same wet/dry probabilities: the gain cannot be read from the images. | `METHOD.md` |
| `figures/doc/screening_timeline.png` | Scenes per month at Villaviciosa, 2023-2025: every acquisition, and the scenes kept by the cloud screening. | `METHOD.md` |
| `figures/doc/sections_villaviciosa.png` | The intertidal pixels of Villaviciosa colored by 3-km section, with each section's applied delay and interval. | `METHOD.md` |
| `figures/interior_tide_schematic.png` | Inside an estuary the tide arrives late: the single level read at the mouth is wrong by an amount that grows inland; MAREA estimates one delay per band of distance along the channel. | earlier README |
| `figures/study_area.png` | The Ría de Villaviciosa: the intertidal zone, contours of distance along the water from the mouth, the tide point, the RTK survey route, a location map and a histogram of the tide levels the archive sampled. | earlier README |
| `figures/workflow.png` | The workflow on Villaviciosa data: the multi-temporal archive, the wet/dry classification, one pixel's record with its fitted transition, and the resulting elevation map. | earlier README |
| `figures/datacube.svg` | The Villaviciosa data cube drawn by `pyintertidal.explain`: bands against time, with real thumbnails of the scenes, clouds included. | earlier README |
| `figures/bands_map.png` | The estimation unit of the delay: bands of distance along the water over the ría, each sharing one delay. | earlier README |
| `figures/pixel_model_schematic.png` | One pixel in cross-section at three tide stages, and the S-shaped curve of NDWI against water level they trace, with a, b, z and σ marked. | `METHOD.md` |
| `figures/sampling_schematic.png` | Sun-synchronous sampling: the M2 constituent is seen as a 14.8-day wave, while S2 is frozen at one phase. | `METHOD.md` |
| `figures/north_cells.png` | The 130 terrain-following cells of the north-coast campaign, colored by the verdict of each cell. | earlier README |
| `figures/onepixel/cell07_0.png` | Histogram of NDWI over the clear pixels of the clearest scenes, with Otsu's split, the adopted threshold at 0, and a rug of the example pixel's own looks. | one-pixel example, figure 1 |
| `figures/onepixel/cell10_0.png` | The example pixel's NDWI against time, colored by what the cloud screening did with each visit (scene rejected, pixel not clear, kept). | figure 2 |
| `figures/onepixel/cell13_0.png` | Water frequency in a 1.2-km window around the example pixel (red square): white never wet, dark blue always wet, intermediate blues the intertidal flat. | figure 3 |
| `figures/onepixel/cell12_0.png` | Histogram of water frequency over the transition zone, with the two Otsu valleys, the adopted window and the example pixel's value. | figure 4 |
| `figures/onepixel/cell15_0.png` | The example pixel's kept looks, NDWI against date, wet in blue and dry in brown, each carrying the EOT20 level at its acquisition time. | figure 5 |
| `figures/onepixel/cell19_0.png` | The same looks under five candidate delays: each panel shifts the level of every look and refits the S-curve, marking the fitted elevation. | figure 6 |
| `figures/onepixel/cell22_0.png` | Log-likelihood of the example pixel with no delay over a grid of candidate elevations and widths, with its maximum marked. | figure 7 |
| `figures/onepixel/likelihood_explained.png` | The likelihood drawn: the rule P(wet given h) with the looks on it, the cost of one look as a function of (h - z)/σ, and all the costs sorted, with no delay and with the band's delay. | figure 8 |
| `figures/onepixel/cell24_0.png` | Profiled log-likelihood against candidate delay for the pixel alone and for its band, with the band optimum; below, the profiled elevation and width for each delay. | figure 9 |
| `figures/onepixel/cell29_0.png` | The elevation inversion of the same looks with no delay and with the band's delay, side by side, with the fitted S-curve and elevation. | figure 10 |
| `figures/tide_comparison/<site>_boundary_options.png` (ems, escalda, ferrol, wadden) | Over a spring tide, the candidate water levels at the mouth: EOT20, the nearest gauges and their consensus. | comparison |
| `figures/tide_comparison/<site>_marea_clocks.png` (ems, escalda, ferrol, wadden) | MAREA's fitted delay per band against distance from the mouth, and the profile applied at the time (a 10-minute detection threshold, since removed). | comparison |
| `figures/tide_comparison/<site>_judge_sweep.png` (ems, escalda, ferrol, wadden) | At one tide gauge inside the estuary (the judge), the error of the level predicted from the mouth for every delay applied to it, with the delays of each method marked. | comparison |
| `figures/tide_comparison/<site>_judge_window.png` (escalda, ferrol, wadden) | Two days at the judge gauge: the measured level and the level predicted from the mouth with no delay, with MAREA's delay and with Granadeiro's. | comparison |
| `figures/tide_comparison/<site>_dem_without_vs_with.png` (ems, escalda, ferrol, wadden) | Four maps: elevation with no delay, elevation with MAREA's applied delays, their difference, and the difference the fitted delays would make without the threshold. | comparison |
| `figures/tide_comparison/<site>_granadeiro_maps.png` (ems, escalda, ferrol, wadden) | Granadeiro et al. (2021), as re-implemented: their delay map, their elevation map, the map with their delay in MAREA's inversion, and their difference from MAREA. | comparison |
| `figures/tide_comparison/<site>_granadeiro_vs_marea_scatter.png` (ems, escalda, ferrol, wadden) | Density scatter of Granadeiro's elevation, and of the crossed map, against MAREA's elevation, pixel by pixel, with the 1:1 line. | comparison |
| `figures/tide_comparison/vaklodingen_by_band.png` | Error (top) and slope (bottom) against the Vaklodingen surveys by band of distance from the mouth, for each product, at the three Dutch and German sites. | comparison |
