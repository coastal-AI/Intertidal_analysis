# The investigation, line by line

`experiments/` holds the disciplined era — pre-registered criteria, matched
nulls, sealed verdicts. But the project did not start disciplined, and far
more was tried than those ~36 scripts show. This is the curated map of
**every line of inquiry pursued**, with its outcome and where its record
lives. The raw dated diary is `PHASE_LOG.md`; the raw exploratory scripts
(150+, kept verbatim) are published separately on request.

Outcome legend: **ADOPTED** (in the pipeline today) · **DEAD** (tried,
controlled, rejected) · **SUPERSEDED** (worked, replaced by something
better) · **RETRACTED** (published internally, then found flawed and
withdrawn) · **OPEN** (parked, with a registered next step).

## 0 · Origins (the months before the diary existed)

The phase log starts with plan v4; the project does not. The first months
live in the root notebooks (already public since the first pushes) and in
the legacy `intertidal/` package.

| Line | Outcome | Record |
|---|---|---|
| First contact: scene-by-scene SCL exploration of Gijón and the Ría de Foz | SUPERSEDED — grew into the reference-map machinery | `foz_sentinel2_scl.ipynb` (repo root) |
| Water-occurrence mapping of Foz (how often each pixel is wet) | SUPERSEDED → `frequency.water_frequency` | `comparacion_metodos_agua.ipynb` |
| Which water detector, where (SCL vs indices, method comparison) | fed the constants and defaults of `water.py` | `comparacion_metodos_agua.ipynb` |
| Tide-model selection: FES / GOT4.10 / EOT20 tried at Foz and the Ebro delta; CMEMS sea level as the weather-aware alternative | EOT20 **ADOPTED** as default — and the finding that DEA's published model rankings are Australia-only and their "ensemble" a plain mean | `tidemodel_test_foz.ipynb`, `tidemodel_test_ebro.ipynb`, `pyintertidal/tidemodels.py` |
| Tide models vs real gauges (PORTUS/REDMAR) | **ADOPTED** as the standing check | `pyintertidal/gauges.py` |
| The server-side era: reference map and water frequency computed as openEO UDFs, imagery downloaded per date | SUPERSEDED by the cached-cube + streaming design (download once, compute locally) | legacy `intertidal/` package; CATALOG "not ported" table |
| Download optimisation: one cube vs per-date jobs, benchmarked | the cube **ADOPTED**; the benchmark kept | `intertidal/benchmark.py` |
| External truth via IGN LiDAR (5 m, WCS) | **ADOPTED** where surveys are lacking | `validation.download_mdt_ign` |
| The RTK-GNSS field campaign at Villaviciosa (361 FIX points) and the sealed 35 % holdout design | **ADOPTED**; the reserved subset is guarded by a test and was opened exactly once, logged | `pyintertidal/rtk.py`, `tests/test_r1_guard.py`, `experiments/v3_open_reserved.py` |
| First end-to-end pipeline notebook | SUPERSEDED by the study notebooks + the package | `final_notebook.ipynb` |
| Dataset generator for the conditional image model (TideGAN) | **ADOPTED**, now multiband | `modelo_generativo/` |

## 1 · Water detection and mapping

| Line | Outcome | Record |
|---|---|---|
| SCL as wet/dry voter | DEAD — swallows turbid shallow flats (Santander: ~10% of the transition zone, unrecoverable) | `research/08`, PHASE_LOG 2026-08-27 |
| NDWI as voter, SCL as cloud gatekeeper | **ADOPTED** | `pyintertidal/water.py` |
| Multi-index A/B (SCL vs NDWI vs MNDWI vs AWEI, one job) | SUPERSEDED — sound design, ranking scored against reused LiDAR, kept provisional | `pyintertidal/legacy/` |
| Three separate reference-map implementations | SUPERSEDED — consolidated into one streaming pass | CATALOG "not ported" table |
| Transition-zone cloud filter (recover scenes cloudy over land, clear over the estuary) | **ADOPTED** | `stability.date_recovery_gain` |
| Otsu self-calibration of the water threshold | **ADOPTED** (opt-in) | `water.otsu_threshold` |
| NDWI threshold sweep (0 vs +0.1 vs +0.3) | +0.1 dominates; +0.3 is a compression trap | PHASE_LOG (P11 follow-up) |
| Marsh through the canopy (MNDWI/SWIR) | **ADOPTED** as a layer; "no SWIR → declare not-assessed" queued | `marsh.py`, `research/08` |
| Witness scenes + 2-of-5 glint vetting + connectivity readmission | **ADOPTED** as production rule (naive union was 51% glint) | `research/08`, sealed `results/santander_*` |

## 2 · Elevation estimators (the ladder that led to HSR)

| Line | Outcome | Record |
|---|---|---|
| Five server-side bathymetry methods (optimized / pixels / isolines / step / siq) | SUPERSEDED — their quoted RMSEs later found scored on reused LiDAR and retired | `pyintertidal/legacy/bathymetry.py` (verbatim UDFs kept) |
| Per-pixel bracketing "dictionary" (midpoint of max-dry / min-wet) | DEAD — won on Villaviciosa RTK, failed to generalise (compressed slopes at all three external sites) | `experiments/p8_bracket_regions.py` |
| DEA step method (local reimplementation) | kept as the published-literature baseline | `elevation.fit_step` |
| ML regressor with survey labels (Ridge) | works but needs labels — kept as the labels-ceiling reference, not a method | PHASE_LOG |
| HSR: per-pixel sigmoid + sub-pixel super-resolution | **ADOPTED** — the elevation engine | `elevation.fit_hsr`, README method section |
| Six-parameter pixel (drainage + attenuation terms) | DEAD — wins the out-of-sample fit, collapses the elevation slope (gate ii) | `experiments/p12`, `research/06` |

## 3 · The compression saga (months of it)

The validation slope came out well below 1 and we hunted the physics for
weeks. Every hypothesis below was tested with its own control, and every
one failed to explain it — because it was not physics.

| Hypothesis | Outcome | Record |
|---|---|---|
| Tide-model error / spring–neap split | DEAD (matched-noise null: p = 0.31) | PHASE_LOG 2026-08-18 |
| Storm surge (real level = tide + surge) | DEAD — shuffled surge works equally well | PHASE_LOG |
| Censoring at the tidal ceiling | **RETRACTED** — the analysis was circular (regressor identical to the target); retraction kept in writing | PHASE_LOG |
| Morphological drift between survey and archive | DEAD — trend CI crosses zero across epochs | PHASE_LOG |
| Overpass-hour error (nominal vs real acquisition time) | real effect (0.134 m of tide) but averages out; real times **ADOPTED** everywhere | `overpass.py` |
| Optical index bias (MNDWI reads 0.120 m lower than NDWI) | real, does not predict the error (r = +0.01) | PHASE_LOG |
| **Resolution: point-vs-pixel sampling** — a GNSS point is not a 10 m pixel median; regression attenuation from sub-pixel spread explains the slope, quantitatively (σ_e two independent ways: 0.214 m both) | **ADOPTED** as the validation model; the method was essentially unbiased all along | `validation.point_sampling_error`, `research/04` §0 |

## 4 · Estimating the water surface itself (the graveyard that taught the theorem)

| Line | Outcome | Record |
|---|---|---|
| Free water level per scene ("intertidal tide gauge") | DEAD — exactly degenerate with a global elevation shift; the failure that became the affine theorem | `research/01` intro, prototypes |
| Tilted water sheet, free slope γ per scene | DEAD — 23% of scenes pinned at the grid edge, 44% at zero: mis-specified | PHASE_LOG 2026-08-18 |
| γ(tide state), 3 parameters | DEAD against its synthetic nulls | PHASE_LOG |
| Connectivity / bathtub violation ("flooding needs a path") | partially **ADOPTED** — became the spill/ponding layers and the censoring bound | `b6_hydraulic_dem`, `research/05` |
| Time-axis warp (uniform deformation of the tide axis) | SUPERSEDED — subsumed by the spatial interior-tide model | prototypes |

## 5 · The interior tide (from world tour to MAREA)

| Line | Outcome | Record |
|---|---|---|
| Boundary providers (pyTMD / gauge / ensemble, interchangeable) | **ADOPTED** | `boundary.py` |
| Gauge-pair world tour (Guadalquivir 85 km → Tamsui 2 km; 2 controls; 1 broken pair discarded) | estuary *character* decides, not distance; the honest middle term (harmonics-only) exposed 96% over-credit in the first Ferrol validation | `experiments/p7`, `research/03` |
| Alias arithmetic (what a sun-synchronous archive can estimate) | **ADOPTED** as a hard spec: S2 frozen, K1/P1 lost, M2 at 14.8 d | `m2_alias`, `alias_table.py` |
| Image-measured lag v3 (rank-correlation clock sweep) | SUPERSEDED by M2a; its Scheldt number survives as history | PHASE_LOG, `p6` row |
| Per-limb (flood/ebb) clocks | DEAD three times, three distinct failure modes, each caught | `b5_*`, `research/06` |
| Ponding index per pixel (charcos) | evidence of hysteresis upstream; kept as diagnostic | prototypes, PHASE_LOG |
| MAREA M2a (profiled Bernoulli likelihood per geodesic band, matched-null adoption) | **ADOPTED** — the titular method | `m2_*`, `research/01` |
| One-parameter law τ = γ·s (three independent estimators) | DEAD as replacement (inside nulls; damages the convex Tagus), kept as diagnostic; grid-cap lesson recorded | `p9/p9b/p9c/p10`, `research/01` §4 |
| Broken-stick τ(s) | OPEN — registered next rung | PHASE_LOG |

## 6 · Depth beyond the waterline

| Line | Outcome | Record |
|---|---|---|
| Four-regime NDWI master curve (450k px, LiDAR truth) | measured; only the mixing regime inverts per pixel | `p11`, README figure |
| Blind-band depth inversion v1 | DEAD — hallucinated +1.7 m outside the monotone window | PHASE_LOG |
| Blind-band v2 (low-tide scenes, inside window) | DEAD — honest, still no skill vs the trivial baseline (bottom albedo > depth signal) | `p11b`, `research/06` |
| Blue/green-ratio SDB route | OPEN — needs B02, which the multiband generator now downloads | PHASE_LOG |

## 7 · Uncertainty

| Line | Outcome | Record |
|---|---|---|
| Analytic per-pixel bar (Cramér–Rao from each fit) | DEAD as the bar (9% coverage at 68%) — and thereby measured the 5 cm statistical floor: the level, not the fit, is the bottleneck | `p13`, `research/05` |
| σ decomposition (relief vs level error) | **ADOPTED** | `b2`, `research/05` |
| Twin-tabulated σ_z (planted recoveries, calibrated simulator) | **ADOPTED** — conservative coverage | `b7`, `research/05` |

## 8 · Infrastructure lessons that cost real days

| Lesson | Record |
|---|---|
| A duplicated cube left products on two different grids and silently invalidated an afternoon of ML numbers — grid asserts everywhere since | PHASE_LOG |
| openEO `temporal_extent` is right-exclusive: the last requested date silently never downloads | `intertidal/openeo_client.py` comment |
| CDSE cannot filter scattered dates server-side (three formulations tried, all fail); dated note in the code so nobody re-spends that afternoon | same file |
| Campaign v1's rectangular cells cut rias in half (12/38 cells came back empty) → terrain-following cells v2 | `make_cells_v2.py`, README figure |

If a claim anywhere in the repo seems to lack its trial-and-error, it is
almost certainly in this table's pointers — and if it is not, ask: the
prototypes and the full diary go back further than any summary can.
