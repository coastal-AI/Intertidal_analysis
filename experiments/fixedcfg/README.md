# experiments/fixedcfg/: competitors for the fixed-configuration tables

The elevation methods that `experiments/v19_fixed_config_tables.py` runs
side by side (the first author's design of 2026-10-20). Every method gets the
same configuration, external to all of them and identical to MAREA's
production one: MAREA's scene screening, the per-pixel clear mask
(SCL 4-7), NDWI with wet = NDWI > 0, the production pixels and the boundary
tide h_B as in production; a window of the n most recent screened scenes per
site; one column per method. MAREA keeps its delay model (refitted on every
window); every other method reads h_B without delay. MAREA's headline
column uses the quality rule of the 10 m production products
(`min_bracket = 0`); the current rule (`marea.QUALITY_MIN_BRACKET = 3`) is a
declared variant. The design, the truths and the window ends are stated in
full in the docstring of `v19_fixed_config_tables.py`, which is the
reference.

This folder is being extended (October 2026); the module docstrings are the
authority on each method.

## Modules

| file | what it does |
|---|---|
| `methods.py` | the registry `METHODS = {id: (label, run, card)}`, in table-column order, and `VARIANTS` (declared sensitivities); every `run(obs, cfg)` returns `dict(z, sigma, meta)` |
| `elev_base.py` | shared plumbing of the per-pixel competitors: pixel blocks, method metadata, the "not computable" result |
| `elev_logitide.py` | E4, LogiTide2DEM (Chen and Wang 2025): a per-pixel binary logistic regression on the tide |
| `elev_freq.py` | E5, frequency to elevation: Flood2Topo (E5a, Fassoni-Andrade et al.), Tseng et al. 2017 (E5b) and the sampled exceedance (E5c) |
| `elev_coastxray.py` | E7, Coast X-Ray (Fitton et al. 2021): stage-interval composites with a minimum reducer |
| `elev_bracket.py` | E12, the in-house wet/dry bracket of the earlier server-side method (`pyintertidal/legacy/udf.py`) |
| `level_series.py` | the window's full water-level series at 1 min and its harmonic analysis, for the methods that need more than the scene instants (E5a, E5b, E7) |
| `dea_intertidal.py` | DEA Intertidal 2.5.3 (Bishop-Taylor et al.; Apache-2.0) as published, with its default parameters; it runs DEA's own code in a separate interpreter |
| `dea_worker.py` | the worker that runs in that separate interpreter; it refuses to start inside the repository and checks that it imports DEA's wheel, not this repository's legacy `intertidal/` package |
| `smoke_competitors.py` | a check that every competitor runs on a site's real inputs and scores sensibly (not the table runner) |

MAREA (E0b), the same inversion with no delay (E0a) and NIDEM (E1) are run
by `v19_fixed_config_tables.py` itself, with the code of v14 and of
`pyintertidal.waterline_contour`.

## How it connects

```
v19_fixed_config_tables.py  plan / run / tables / curves
   |-- the screened scenes, pixels, truth units and lags   <- experiments/v14_scene_count_curves.py
   |-- the window's level series                           <- level_series.py
   |-- every column                                        <- methods.METHODS
          |-- E4, E5a-c, E7, E12 (elev_*.py, on elev_base.py)
          |-- DEA Intertidal (dea_intertidal.py -> dea_worker.py, separate interpreter)
```

## Environment

The DEA column needs a second Python environment with `dea-intertidal==2.5.3`
(the side interpreter); `dea_intertidal.py` explains where it looks for it.
The other competitors need only the main environment.

## Outputs and tests

`results/fixedcfg/` (per site: `<site>.csv`, `<site>_settings.json`,
`<site>_windows.json`, `<site>_units.npz`, resumable jobs in `<site>_jobs/`;
tables in `tables/`; figures `curves_*`); see
[`../../docs/products.md`](../../docs/products.md). Tests:
`tests/test_fixedcfg_methods.py`, `tests/test_fixedcfg_runner.py`,
`tests/test_fixedcfg_dea.py`.
