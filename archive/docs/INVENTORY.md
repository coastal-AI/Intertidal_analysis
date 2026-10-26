# INVENTORY — existing assets against the phases of plan v4

Date: 2026-08-18. Real state verified on disk, not from memory.
Legend: **PKG** = in the `pyintertidal/` package (production) · **SCR** = prototype in
the session scratchpad (temporary directory, **lost if not ported**) ·
**DONE** = result already obtained today, with numbers · **MISSING** = does not exist.

## 0. Data on disk

| asset | path | status |
|---|---|---|
| Villaviciosa 10-year cube (B03/B08/SCL, 10 m, 1379 t, 915×915) | `ndwi_cube_villaviciosa_grande_10y.nc` (3.09 GB) | PKG |
| Villaviciosa v2 cube (B03/**B04**/B08/**B11**/SCL, 10 m, 465 t, **967×915**) | `swir_cube_villaviciosa_2023-2025.nc` (1.85 GB) | PKG — see grid note |
| Scheldt 20 m cube (464 t, 734×1051) | `ndwi_cube_escalda_2023-2025_20m.nc` (1.13 GB) | PKG |
| Saint-Malo / Sheerness 20 m cubes | `ndwi_cube_{stmalo,sheerness}_2023-2025_20m.nc` | PKG |
| Cuxhaven cube | openEO job `j-26081815190240959d…` queued when the session died | MISSING re-attach |
| Tagus / Vadehavet / Santander cubes (old epochs) | `ndwi_cube_{tejo,vadehavet,santander}_*.nc` | PKG (Part IV) |
| Raw Villaviciosa RTK | `data/villaviciosa_rtk_gnss.csv` — 361 FIX, 248 px, 54 px with 2 points, 23 with 3+ | PKG |
| Extracted intertidal series (Y, C, keep, dates) | `scratchpad/ndwi_intermareal.npz` (175 MB), `swir_intermareal.npz` (148 MB), `escalda_intermareal.npz` | **SCR — port** |
| Geodesic distance along the channel + sea/reachable masks | `scratchpad/canal.npz` | **SCR — port** |
| Cached IOC tide gauges (bon2, sev2, fer1/2, vlis, trnz, tdan/tdas, goer/gohi, hono, keehi, barc/2, leha, diep…) | `scratchpad/ioc_*.json` | **SCR — port** |
| Real S2 overpass times (STAC; the cube only stores the DATE) | via `pyintertidal.overpass.get_overpass_times` | PKG |

**Grid note (institutional, cost two incidents):** the v2 cube came down as 967×915; it
contains the EXACT canonical 915×915 grid with a 26-row offset in y (verified);
crop `[26:941, 0:915]`. Every ingest must `assert` against the canonical grid —
a silent mismatch invalidated an entire morning of ML results.

## 1. `pyintertidal/` package → architecture §4

| existing module | §4 destination | notes |
|---|---|---|
| `cube.py` (SentinelCube, `extra_bands`, `last_job_id`) | data/ | openEO download; CDSE limits to **1 connection** |
| `scenes.py` (connect), `overpass.py` (STAC times) | data/ | real overpass = 11:21 UTC, not 11:00 |
| `water.py`, `stability.py`, `frequency.py` | data/, optics/ | detectors and intertidal |
| `elevation.py` — `_fit_block` (closed form), `fit_hsr`, `fit_step`, `extract_intertidal_ndwi` | inference/ | §1 baseline; **amplitude guards** `0.15<b<2.5`, `|a|<2.5` learned today (frozen prediction blows up without them) |
| `tides.py`, `tidemodels.py`, `tidecheck.py` (nearest_ocean_km, phase_lag, range_dependence) | tide/ | |
| **`boundary.py`** — BoundaryProvider, PyTMDBoundary, GaugeBoundary, EnsembleBoundary, `fit_harmonics`, `alias_periods`, **fixed EPOCH** | tide/ | **M3 almost entirely exists already**; missing ClimatologyBoundary and the (1+γ_k, Δφ_k) correction with priors. BUG solved: phases anchored to a fixed epoch (referencing them to the start of the series dephases between windows) |
| **`estuary.py`** — EstuaryTransfer (`from_gauges` VALIDATED, `from_geometry` NOT VALIDATED, `residual_gain` measured, `validate_against_gauge` with 3 terms) | tide/ | the 3rd term (harmonics-only) is mandatory: without it, 96% of an improvement that was filtering got attributed to the operator |
| `validation.py` — `compare_dems`, `transect_profile`, **`point_sampling_error`** | stats/ | point-vs-pixel decomposition (see §3 DONE) |
| `gauges.py`, `hypsometry.py`, `mosaic.py`, `shoreline.py`, `aoi.py` | geometry/, data/ | M1 and Part IV |
| `campaign_north_coast.ipynb` (116 cells; `rows` `NameError` fixed) | Part IV | |

## 2. Scratchpad prototypes → port (lost with the session)

| prototype | v4 destination | what it does |
|---|---|---|
| `sesgo.py` / `modelo.py::simulate_training_set` | simulator/ | **the tribunal**: real tide + (a,b,σ,noise) resampled per WHOLE pixel + scene systematic + planted elevations. Missing a pluggable-mechanisms API |
| `lamina.py::flood_threshold` | optics/ + simulator/ | connectivity censoring (minimax by morphological reconstruction; tested on synthetic) |
| `nulo_bloques.py` | stats/ | **MASTER NULL B1 — ALREADY RUN** (see §3) |
| `atenuacion.py`, `varianzas.py` | stats/ | k=1 vs k=2 disattenuation; σ_e by two independent routes |
| `forma_cx.py` | stats/ | c(x) regression with interactions against a null; **sealed ad-hoc sha256 `adadcb0ed33d3f39`** — migrate to `seal.py` |
| `retardo_imagen.py`, `escalda_valida.py` (lag_for, h(t−τ) sweep, synthetic calibration) | tide/ (basis of M2c/M2d) | lag by flooded-area ranks; ±4 min calibrated precision |
| `metodo_b.py`, `metodo_b2.py` | tide/ (relative of M2a) | per-band warp (A, τ_subida, τ_bajada), selection on train, out-of-sample judge + specular control. Profiles z with `_fit_block` (Gaussian; M2a wants Bernoulli — minor change) |
| `charcos.py` | optics/ | per-pixel ponding index at matched levels, with a limb-permutation null |
| `doble_indice.py` | optics/ B2 | δ̂ = ẑ_MNDWI − ẑ_NDWI with a two-fits-of-the-same-truth null |
| `busca_mareografos.py`, `pares_mundo.py`, `curva_estuarios.py`, `vs_modelos.py`, `contornos.py` | tide/, Part IV | multi-estuary IOC admittance; EOT20/GOT/ensemble/operator comparison against 17 tide gauges |
| `hora_paso.py` | tide/alias_table.py | **alias table already computed** (see §3) |

## 3. Results already obtained that pre-load v4 gates

| v4 gate | status | numbers |
|---|---|---|
| M0 (block structure) | reference numbers measured | dev 0.698 [0.608–0.782]; ~~holdout 0.890~~ ⚠ see blocking question Q1: reproducing 0.890 requires reading the holdout and would violate R1 |
| M2 alias_table | **DONE** | estimable M2 (14.8 d), N2 (9.6), O1 (14.2), Q1, M4, M6; **S2 invisible by construction**; K1/P1 annual alias |
| M2d (Granadeiro) | relative already validated externally | lag from imagery: Villaviciosa profile 0→+74 min (±4 min calibrated); **Scheldt: gradient 0.9 min/km = tide gauges; recovers 75% of the achievable correction against Terneuzen** |
| M2a (partial) | prototype with a passed gate | band 0.98–3.53 km: warp (A=0.85, τ_s=0, τ_b=+20) **PASSES out of sample with specular control** (0.1996→0.1924; control 0.2112). Hysteresis measured, not imposed |
| M3 (boundary) | modules done + measurement | EOT20/GOT4.10/GOT4.8 against 17 tide gauges: at the coast they tie (~0.075 m); inside an estuary EOT20 0.222 beats GOT 0.32 and **the series ensemble (destructive phase interference — average harmonics, not series)** |
| B1 (dilution null) | **DONE — gate closed** | unbiased estimator with planted compression 0: slope 1.019, none of 200 replicates goes down to the observed 0.561; naive I² inflated (the null alone already gives 40%, max 85% vs 86% observed). Classical dilution ruled out (noise in the response, not the predictor) |
| B1 (c(x) regression) | **DONE — sealed** | no available covariate modulates c(x) (all within the null); only the global compression (z=−3.48). sha `adadcb0ed33d3f39` |
| B2 (double index) | **DONE** | real δ (1.44× the null, MNDWI −0.120 m vs NDWI) but **does not predict the field error** (r=+0.010) nor track turbidity (+0.066) |
| — (not in v4, affects B1/V3) | **DONE** | point-vs-pixel: σ_e=0.214 m by two independent agreeing routes; **true c = 1.041 [0.79, 1.58]**; of the reported error, 61–88% was the reference. `validation.point_sampling_error` in the package |

## 4. Net absences (build from scratch)

`seal.py` and `sealed/` · R1 guard with `allow_reserved` + import test · zarr ingest
partitioned by block · `src/` layout + YAML configs + `tests/test_gate_*` ·
full Bernoulli M2a (harmonic in t × spline in s, L-BFGS-B ~20–40 params; today only
the per-band version with a 3-parameter warp exists) · M2b copulas · M2c differential
waterline · Toffolon–Savenije ODE (M4 prior) · joint EM B3 · P1 SWOT ·
ClimatologyBoundary · boundary correction with priors.

## 5. Institutional memory (mistakes that already cost hours — do not repeat)

1. Grids: always `assert`; 3 incidents today (887×673, 967×915, 588×1852).
2. The cube stores the **date, not the time**: join STAC times; the real overpass is 11:21±10 UTC.
3. Harmonic phases: reference to a **fixed EPOCH**, never to the start of the series.
4. `harmonics(constituents, ...)` positional: the parameter order is a contract.
5. Unbounded amplitudes (a, b) blow up in frozen prediction.
6. CDSE: 1 connection; serial downloads; float32 (OOM with 0.47 GB free killed a run).
7. Naive p-values invalid (autocorrelated pixels): two "p=1e-37" results died today against matched nulls.
8. Comparing operators against the raw series gives the filtering away for free: 3 terms always.

## Added in phase M2–M4 (2026-08-19)

New package modules:
- `alias_table.py` — constituent decision with real overpass times.
- `tide_estimators.py` — ShiftBank, quantile bands, M2a (phase-only Rasch,
  profiled σ), M2b (concordance), M2c (waterline by ranks), M2d (Granadeiro),
  `alpha_nll_profile` (the affine-degeneracy meter, with scaling grids).
- `boundary_correction.py` — harmonic decomposition of the boundary, correction
  (1+γ_k, dt_k) on estimables only, `ClimatologyBoundary`, OOS judge of the mouth.
- `operator.py` — `InteriorTide`, the distributed tide gauge with `describe()`.

Experiments (each writes a result.json with input SHAs + figure):
- `m2_alias`, `m2_gate_sim` (planted truth), `m2_real` (real archive vs uniform
  null), `m3_gate_sim` (planted boundary error), `m4_gate_sim` (benefit +
  contraction), `b1_bathymetry` (product with/without operator + DECLARED dev
  diagnostic + GeoTIFFs), `b2_sigma` (σ_topo/σ_nivel decomposition),
  `v3_open_reserved` (human-only, double safeguard).

Gate tests: `test_gate_m2/m3/m4` over the result.json files (plus the earlier M0/M1
and the R1 guard).
