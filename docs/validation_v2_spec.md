# Validation v2: factorial validation of MAREA against state-of-the-art tidal-lag and intertidal-elevation methods (specification, revision 2)

> **USER DECISIONS (2026-10-14), overriding the 2026-10-13 decisions and everything below where they differ.**
> - **Sensing time: the catalogue time, no G-time gate.** The user judges the solar-angle sensing-time correction to be over-engineering for now and wants no detour from the validation. The sensing instant used everywhere is the catalogue / product-name time (the datatake start), as in every existing production result. `experiments/benchmark/sensing_time.py` stays as an optional diagnostic (`diag time`): it is not a gate, not a setting and not a design row, and it is used only when a sensing-times file is passed explicitly. For the record, at the Ems the tile passes 0-10.6 min after the datatake start (median 6.5 min, 1000 tile-dates, 2023-2025).
> - **If errors turn out large, attribute them before correcting anything:** first decide whether the difference is noise (paired block-bootstrap CIs), then whether it comes from the lag or from the elevation estimation, by comparing on the same pixels, scenes and settings the no-lag level (τ = 0), MAREA's band lags and the measured gauge level (LG, upper bound). The sensing-time diagnostic is reconsidered only if that attribution points to the timing.
> - **Gauge-lag row (approved 2026-10-15).** At every site with in-estuary gauges (Escalda, Wadden, Ems), add one elevation row on the same pixels, scenes and settings as the base pair: MAREA's inversion with the band lags taken from the gauges' measured lags (interpolated along the channel coordinate, τ(0) = 0, monotone) instead of the archive estimate. Read together, the four rows split the error. τ = 0 vs gauge lag is the cost of ignoring the lag. MAREA lag vs gauge lag is the cost of estimating the lag from the archive. Gauge lag vs measured level (LG) is the cost of the one-lag-per-band model. What remains at LG is the elevation-estimation error.
> - **Naming:** the two campaigns are **Campaign 1** (2023-2025, run now; formerly "Tier 1A'") and **Campaign 2** (adds 2017-2022; formerly "Tier 1B"). Both use Sentinel-2 L2A at 10 m (S2A/B/C). "Tier 2/3" elsewhere in this document are method and setting priority tiers, not campaigns.
>
> **USER DECISIONS (2026-10-13), overriding the defaults below where they differ.**
> - **V21 G-time: APPROVED, via the solar angles** (superseded on 2026-10-14, see above). The Zarr downloader fetches sunZenithAngles, sunAzimuthAngles, viewZenithMean and viewAzimuthMean as a separate small float product per site-year (`cubes/<site>/<year>.angles.zarr`, read with `zarr_cubes.open_angles` / `angles_at`); the per-site sensing time is the solar-ephemeris inversion of those angles. A Space-Track TLE cross-check is optional, not required.
> - **V22 water-level sources: ALL FOUR NOW** (LB boundary, LT global model per pixel, LM regional CMEMS AMM15/IBI, LG gauge network as a labelled upper bound), with family P3. The user creates a Copernicus Marine account; credentials and the `copernicusmarine` toolbox live inside the project folder/venv on the server.
> - **V1/V2 cloud grid: OmniCloudMask (C11) INCLUDED** as a setting (Wright et al. 2025, RSE 322:114694, doi:10.1016/j.rse.2025.114694; PyPI `omnicloudmask`), installed with CPU PyTorch inside the project venv; its licence is checked before use; C3 (CLD + buffer) stays as an OFAT cell and the fallback.
> - **V24 scope: Campaign 1 now (2023-2025), Campaign 2 later** as a separate campaign after the 2017-2022 download.
> - Every other decision V1-V30: the recommended default of §17.

- **Date:** 2026-10-09. **Branch:** `validacion-10m`. **Status:** revised after three adversarial reviews: statistics and design, remote sensing, and compute. This is a design for review and is not locked. No code in this document has been run on real 10 m data.
- **Supersedes** `docs/v13_benchmark_spec.md` wherever the two differ. Everything v13 specifies and this document does not change stays in force and is cited as "v13 §x". The v13 harness code (`experiments/benchmark/*.py`, milestone M1) is reused, not rewritten (§12).
- **Inputs:** the per-site, per-year Zarr stores `cubes/<site>/<year>.zarr`, read through `experiments/zarr_cubes.open_cube` (§1.2), with per-scene sensing times re-derived independently of the product names (§1.3).
- **Execution:** the user runs every server command (§13, §15). The assistant writes code and specifications only.
- **Reporting:** only 10 m numbers are reported. Products are 10 m. At the Dutch sites they are scored on the 20 m truth cells (§7.1). 20 m cubes and small Zarr extracts serve only for local tests.
- **What changed after review:** §18 maps every review issue to the change made, or to the reason it was adopted differently.

Conventions:
- **[Vk]** marks a recommended default that still needs the user's confirmation. All of them are listed in §17.
- **"Declared"** means that a deviation from a published method is listed in the deviations table of the supplement (§11.4) and, where it can be, run as a sensitivity row.
- **wh** (worker-hour) is one cheap worker for one hour (§14.1). Costs are estimates extrapolated from v13 §9, the production logs and light probes. Stage 0 recalibrates them before any long run.

---

## 0. Scope

### 0.1 Purpose

MAREA maps intertidal elevation from the Sentinel-2 archive:

- Each pixel's wet/dry record (NDWI) is inverted against the water level by the closed-form sigmoid inversion (`pyintertidal.marea.invert_series`), under a per-pixel quality rule.
- Its distinctive step is a **tidal lag estimated from the archive itself**, one per 3-km band of geodesic distance from the mouth (`tide_estimators.m2a_rasch`: profiled Bernoulli likelihood, monotone, bootstrap CI).
- The local level is therefore h_B(t − τ_band), where h_B is the boundary series (EOT20 plus the outer-gauge consensus). "No delay" is the same inversion with τ = 0.
- The authors fixed the lag-by-bands model. It is not tuned in this study.

Validation v2 asks two questions:
- Is MAREA a good intertidal **bathymetry (elevation) method** compared with the state of the art, and where?
- Is its **lag** a good estimate of the in-estuary tidal delay compared with many other lag methods?

Comparisons are made under identical inputs, across a pre-registered space of processing settings. For elevation, they are also made under the water-level sources the competitors use in practice (§1.6).

### 0.2 Questions

| Id | Question | Answered by |
|---|---|---|
| Q1 | Does MAREA's band lag agree with the in-estuary gauges as well as, or better than, other lag methods (imagery, global and regional models, physics)? | Lag benchmark (§6), run first |
| Q2 | With identical inputs and only a boundary tide, is MAREA's elevation more accurate than each state-of-the-art competitor? | P0 and native cells, family P2 (§5.2, §5.5, §9.2) |
| Q3 | Does the lag correction itself improve elevation (MAREA vs no delay)? This is the non-circular test of the lag. | Family P1 (§9.2) |
| Q4 | Is MAREA, given only a boundary tide, competitive with competitors given a per-pixel tide model or a regional hydrodynamic model? How much headroom remains against gauge-interpolated levels? | Water-level-source rows (§1.6, §5.4), family P3 (§9.2) |
| Q5 | How sensitive is each method, and MAREA's advantage, to cloud handling, spectral index, water detection, band preprocessing and temporal window? | OFAT and combination designs (§5.6-5.7), factor effects (§9.4) |
| Q6 | How many images does each method need? What error floor remains with unlimited images? | Learning curves (§5.10, §7.4) |
| Q7 | Where is MAREA good: along the estuary, by reference lag, elevation zone, slope, substrate, turbidity, amplification and observation density? | Method-independent strata (§7.1, §9.6) |
| Q8 | How much of the error is the mismatch between the image epoch and the truth epoch? | Truth-error model (§1.9), survey-matched views and the epoch analysis (§5.9, §7.5) |
| Q9 | What does each method cost to run, and what accuracy does it buy per unit of compute? | Timing protocol and efficiency report (§8) |

### 0.3 Decisions in force

Carried over from the user's decisions of 2026-10-08 and from v13:

1. **Honest comparison in three layers.**
   - (i) Fidelity of every re-implemented competitor: reproduce a published result where possible, and prove equivalence for every port (§10).
   - (ii) Per competitor, an "as published" row and "common conditions" rows. Every deviation is declared and, where possible, run as a sensitivity (§5.3).
   - (iii) The same rules for every method: a pre-registered protocol, no tuning on truth (except the symmetric, labelled tuning budget of §5.12), and MAREA's home advantage acknowledged in every table note.
2. **One row per method.** Each method keeps its own lag, or none. Methods are never combined into hybrid rows, and no method receives MAREA's lag.
3. **Granadeiro et al. (2021)** fits stay in the R packages the paper names (nplr, lmodel2, mgcv). They run on the server, alone or with at most two cheap workers beside them.
4. **Only 10 m numbers are reported.**
5. **RTK:** read only through `pyintertidal.rtk.load_rtk()` with default arguments (development points). The reserved split stays sealed.
6. **MAREA's production per-pixel cloud mask** (SCL ∈ {4,5,6,7}) is the base cloud level. **floor0** band preprocessing is the base preprocessing (v13 D1, D2).
7. **Lag truths:** (a), τ_ab and (b) are co-primary (v13 D9). **Lag read-out:** one scorer rule for every method's applied lag field, with the symmetric fallback ladder of §6.3 (amends v13 D8).
8. **Failure policy:** NOT COMPUTABLE is reported, never relaxed. It counts as coverage 0 in curves and as a loss in robustness rates (v13 D3, D17; §9.5). MAREA's floor is 40 scenes (v13 D4).

New with the plan of 2026-10-08 (binding):

9. **Settings are factors, defined in Python and applied to every method.** The factors are cloud handling, spectral index, water detection, band preprocessing, temporal window and number of images. Within each combination of settings, only the method changes. The index is a setting applied to every method; native indices remain in the "as published" rows and the native cells (§5.5).
10. **Many state-of-the-art competitors** (§3.2), not only NIDEM and Granadeiro.
11. **Lags are benchmarked first**, against many lag methods (§6).
12. **Inputs:** the Zarr stores, one per site and year: 2023-2025 first, 2017-2022 second.
13. **Report:** tables, figures and conclusions on whether and where MAREA is good (§11). Modules and notebooks live on branch `validacion-10m`.

Proposed in this revision, each pending the user's confirmation (§17):

14. **Scene instants** come from the site sensing-time gate G-time (§1.3), not from the datatake start in the product name [V21].
15. **Water-level source** is a factor for elevation units only: LB (boundary, base), LT (per-pixel global tide model), LM (regional hydrodynamic model), LG (gauge network, labelled upper bound) [V22].
16. **Two campaigns.** Campaign 1 must fit two calendar weeks on the server and needs only the 2023-2025 stores (§5.14, §14). Campaign 2 is a separate later campaign that needs the 2017-2022 batch [V24].

### 0.4 What changes relative to v13

| Topic | v13 | v2 (revision 2) |
|---|---|---|
| Inputs | one netCDF cube per site (B03, B08, SCL), 2023-2025 | Zarr stores per site-year (B02, B03, B04, B08, B8A, B11, B12, SCL, CLD), 2017-2025; the netCDF cubes are kept only for reproduction gates |
| Scene instants | product-name time (datatake start) | site sensing time from sun-angle inversion and orbit propagation (G-time) |
| Water level for elevation | h_B for every method | factor LB/LT/LM/LG; LB in common conditions; native sources in the as-published rows |
| Index | native per method | a setting applied to all methods; indices with a 20 m band computed at 20 m; native index in the as-published rows and the native cells |
| Cloud handling | per-pixel mask, plus scene rules as subset families | one factor (C0-C8, C1d, C11 OmniCloudMask), mixing pixel masks and scene rules |
| Water detection | fixed NDWI > 0 | a factor, D1-D5 (§2.6) |
| Window | 2023-2025 only | nested windows, calendar-year replicate windows, survey-centred windows (§2.7) |
| Elevation competitors | NIDEM, Granadeiro | 11 competitor rows in Tier 1, more in Tier 2 (§3.2) |
| Lag competitors | GM20, HD26, zero lag, EOT20/GOT5.6 | 22 rows in families, regional models in Tier 1 (§3.3) |
| Primary elevation quantities | rmse_meanc and \|1 − b\| | the paired loss differential ΔMSE_c (accuracy) and the attenuation-corrected slope b* (§7.2) |
| Views for inference | common views | pair views per contrast, intersection-union over two views (§7.1, §9.2) |
| Analysis | Holm family of 48 tests | families P1, P2, P3, L-P; Romano-Wolf step-down on a shared-weight bootstrap with B = 20,000; verdicts against margins; headline rule (§9) |
| Timing | wall time per stage | CPU, child-CPU, I/O, peak RSS per stage; dedicated timing runs (§8) |
| Output root | `results/v13_benchmark_10m/` | `results/validation_v2/` |

### 0.5 Non-goals

- **No hybrid rows.** MAREA's lag is never fed to another method. The "modular correction" study is a separate experiment and not part of this specification. In that study, MAREA's band lags are applied as a post-processing transfer function to other methods' maps (decided on 2026-10-07).
- **No tuning of any method on truth**, except two labelled cases:
  - the symmetric internal-parameter budget (§5.12), tuned on Escalda only and printed as "tuned" rows;
  - the leave-one-site-out and oracle rungs of the fairness ladder (§9.7).

  This covers thresholds, radii, p, s_k, t_k, block sizes and margins.
- **No method that needs external elevation data** as input: ICESat-2-calibrated, LiDAR-calibrated, or trained on elevation labels (§3.4).
- **No change to any estimator** in `pyintertidal/` or `pyintertidal/r/` (v13 §0.4). Adapters re-sequence drivers, and gates prove them equivalent.
- **Gauge levels as a method input** appear only in the labelled LG upper-bound rows for elevation (§1.6). They never appear in lag scoring and never in a confirmatory family.

### 0.6 Traceability to the user's plan of 2026-10-08

| Plan item | Where |
|---|---|
| Compare MAREA as a bathymetry method against the state of the art | §3.2, §5, §9.2 (families P2, P3) |
| Settings and combinations in Python: cloud handling, index, water detection, window, number of images | §2, `experiments/benchmark/settings.py` (§12) |
| Cloud: per-pixel mask; transition-zone rule; another filtering method; no filtering | §2.2: C1, C2, C11 OmniCloudMask and C3 (CLD probability + buffer) as "other filtering methods", C0, plus C1d and C4-C8 |
| Index: NIR, NDWI, MNDWI and others | §2.4: I1-I6, NDWI@20, plus Tier 2 |
| Water detection: one, another... | §2.6: D1-D5, plus Tier 2 |
| Window: nested 1/2/3/5/9-year windows ending 2025-12-31, or windows centred on truth surveys | §2.7, §5.8-5.9: both, plus calendar-year replicate windows |
| Number of images: batches and curves | §2.8, §5.10: disjoint batches where they exist, nested subsets labelled |
| Vary only the method within each combination | §5.1: split-plot design with common random numbers |
| Errors, slope, bias, correlation, a new metric | §7.2, §7.3 |
| Error curves vs number of images | §5.10, §7.4 |
| Computation times | §8 |
| Lags first, against many lag methods | §6, §15 (Stage L) |
| Report with tables, figures and conclusions on whether and where MAREA is good | §9.6, §9.7, §11 |
| Everything on the cluster; modules and notebooks on the branch | §12, §13, §15 |

### 0.7 Notation

| Symbol or term | Meaning |
|---|---|
| h_B(t) | Boundary water level: EOT20 plus the outer IOC gauge consensus (EOT20 only at Villaviciosa), as exact levels from the level table (v13 §2.7) |
| LB, LT, LM, LG | Water-level sources for elevation: boundary h_B; per-pixel global tide model; regional hydrodynamic model; gauge-network interpolation along s (§1.6) |
| τ | Lag in minutes; positive means local water is later than h_B |
| instant | The site sensing time of a scene from G-time (§1.3) |
| record | The scene dates of a window that have a G-time instant and a finite h_B, in time order |
| T | Number of scenes in a window's record |
| N | Number of archive scenes given to a method (drawn before any cloud handling) |
| n_clear | Median count of valid observations per domain pixel |
| setting | One level of each factor: (cloud C, preprocessing, index I, detection D), plus the level source for elevation units |
| cell | A setting × window combination |
| instance | (site, cell, subset family, mode, N, replicate): the input every method receives |
| unit | instance × method (× variant): one run, producing one or more products |
| A-mode / F-mode | "The archive has N images" (domain re-derived) / "N images chosen from the full record" (domain from the window's full record) (v13 §0.5) |
| keep, U_rec | The extraction-rule intertidal domain, and its exact superset over subsets (v13 §2.5) |
| S0 | The fixed evaluation support, identical for every cell and method (§2.10) |
| dom | The stored pixel domain of the package, frozen at lock-A (§2.10) |
| D_truth, D_keep | The scoring domain (truth pixels or cells inside the range filter), and keep ∩ truth |
| pair_q:A\|B, pair_raw:A\|B | The pixels (cells) mapped by both A and B, after or before their own quality rules (§7.1) |
| ΔMSE_c(A, B) | Var(e_A) − Var(e_B) on a pair view: the paired loss differential of mean-centred errors (§7.2) |
| σ²_T, λ̂ | Truth-error variance (survey noise plus epoch mismatch) and the reliability ratio 1 − σ²_T/Var(t) (§1.9) |
| b, b* | OLS slope of z on t, and the attenuation-corrected slope b/λ̂ |
| judge | An in-estuary gauge location used as lag truth (§6.1) |
| P0 | The base configuration (§5.2) |
| native cell | Common conditions except a competitor's native cloud rule, index, detector and window (§5.5) |

### 0.8 Scope of the conclusions (stated in the report)

- **Sites.** Every elevation site is one of MAREA's development sites. All are temperate, meso- to macrotidal estuaries, basins or rías in north-western Europe. There is no microtidal site and no competitor's home site unless the away site of §5.14 is approved [V27]. Conclusions are scoped to these conditions.
- **Sensor.** Only Sentinel-2 L2A (Sen2Cor) is used. As published, DEA Intertidal and NIDEM also use Landsat, and Granadeiro used ACOLITE atmospheric correction. These are declared limitations.
- **Water level.** Family P2 answers Q2 "given only a boundary tide". Practical statements against competitors that are normally run with modelled or gauged levels come from family P3 (§9.2).
- **Home advantage.** MAREA's internal parameters were developed on these sites, judges and truths. §9.9 lists the truth-based results the authors had already seen.

---

## 1. Sites, inputs and data prerequisites

### 1.1 Sites

| | villaviciosa | ferrol | escalda (Westerschelde) | wadden (Vlie) | ems (Ems-Dollard) |
|---|---|---|---|---|---|
| Type | ría | ría | alluvial estuary | multi-inlet tidal basin | alluvial estuary |
| Frame H × W (10 m) | 915 × 915 | 905 × 1707 | 1467 × 2101 | 3425 × 2776 | 3206 × 3413 |
| Scenes per year (expected) | ~145-173 | ~145-173 | ~145-173 | ~145-173 | ~145-173 |
| Elevation truth | PNOA LiDAR MDT02 (2 m, ground only, aggregated to 10 m, water-surface masked, §1.8); RTK development points | none (lag only) | Vaklodingen (yearly, 2016-2025) | Vaklodingen (Vlie 2016/2019/2022/2025; inner sheets to 2022) | Vaklodingen (Dutch and German sheets 2020; outer sheets 2019/2022/2025) |
| Scoring support | 10 m pixels | – | 20 m truth cells | 20 m truth cells | 20 m truth cells |
| Judges (headline) | none (lags descriptive) | fer2 | Gat van Borssele; Terneuzen pair (one location) | West-Terschelling; Harlingen | Emshoern, Eemshaven Waddenzee, Dukegat, Knock, Delfzijl, Emden, Pogum, Nieuwe Statenzijl |
| Boundary members | EOT20 only (Gijón as a sensitivity) | fer1 | vlis, brsk | ters | bork |
| Regional model (LM, L16) | IBI 1/36° (expected not to resolve the ría: flagged degenerate) | IBI 1/36° (flagged where the ría is not resolved) | AMM15 1.5 km | AMM15 1.5 km | AMM15 1.5 km (inner Ems flagged where unresolved) |
| Expected confirmatory role | descriptive for P1/P2/P3 unless the truth-free block and band counts of §9.8 allow testing | lag only | confirmatory | confirmatory | confirmatory |
| Mouth side | any | west | west | north + west | north + west |

The site configuration, boundary recipes, judge inclusion rules and neighbourhoods are those of v13 §2.1 and §2.8. Three things change:
- The period becomes the union period 2017-01-01..2025-12-31.
- Paths point to the Zarr stores, plus a `legacy` block (old netCDF cube, production extraction, products) that only the gates use (§12).
- Each site gains a `sensing_times.json` (§1.3) and the level-source configuration (§1.6).

### 1.2 Zarr stores and the reader contract

`experiments/zarr_cubes.py` exists and defines the store format (`FORMAT_ID = "pyintertidal-zarr-cube/1"`):
- one store per site and calendar year, Zarr v2 with consolidated metadata;
- dims `t, y, x` (y descending, x ascending, pixel centres in the site's UTM CRS, 10 m);
- one int16 variable per band (`B02 B03 B04 B08 B8A B11 B12 SCL CLD`), with `_FillValue = −32768` and values exactly as delivered;
- chunks `{t: 16, y: 256, x: 256}`, Blosc/Zstd 5;
- the attribute `complete = 1`, written last;
- the attributes `radiometry` (per date `offset_state`: '+1000 in the values', 'no offset' or 'unclear') and `input_baselines`;
- sidecars `<year>.verify.json`, `<year>.format.txt` and `<year>.job.json`, and `overpass_times.json` per site.

`open_cube(site, start, end, bands, masked=False)` opens the needed yearly stores lazily, checks that they share the grid, concatenates them along time and selects the end-exclusive window.

The harness never reads the stores directly. It reads them through a `SceneSource` protocol (new module `experiments/benchmark/inputs.py`):

| Member | Contract |
|---|---|
| `dates`, `instants` | Window dates and G-time instants (§1.3). A date without a valid instant is not in the record. |
| `grid` | xs, ys, CRS WKT, H, W. Asserted identical across years, equal to the legacy netCDF grid and, at Villaviciosa, to `rtk.DEFAULT_GRID` |
| `lattice20` | The phase (0 or 1 in x and y) of the native 20 m lattice on the 10 m grid, measured in `inputs` from the 2 × 2 replication pattern of B8A, B11, B12 and SCL. `None` if the 20 m bands are not exact 2 × 2 replicas (§2.4). |
| `offset(t)` | The per-scene BOA offset to subtract (0 or 1000 DN), from the store's per-date `offset_state` and the sidecar's processing baseline (§1.10). 'unclear' is an error. |
| `position(t)` | Global record position → (year store, local index) |
| `read_scene(t, bands, rows=None)` | Raw int16 (uint8 for SCL and CLD after a range check), no CF decoding. Backed by a cache of the current 16-date chunk block, so a chunk is decoded once (§13.4). |
| `row_blocks(bands, block_rows=256, dates=...)` | Chunk-aligned iteration in time blocks of 16, in chunk order |
| `fingerprint(year)` | sha256 of the consolidated metadata plus the sorted (chunk key, size) list; optional full content hash |
| `NetCDFSource` | The same protocol over the legacy cubes (2023-2025, B03/B08/SCL), used by the reproduction gates only |

`SceneSource` uses `zarr_cubes.open_cube` (or `open_store`) and adds the record logic, the fingerprints, the block cache and the grid assertions.

Sensitive points that `inputs` asserts (§1.11):
- the encoding of CLD (int16 percentage, 0-100);
- the absence of scale or offset attributes on the spectral bands (`delivered_*` attributes only);
- the BOA offset regime **per scene**, never by date (§1.10, §2.3);
- the nearest-neighbour replication of the 20 m bands (B8A, B11, B12, SCL, CLD) on the 10 m grid, and its lattice phase.

### 1.3 Scene sensing times (gate G-time, Stage 0)

**Problem.** `pyintertidal.overpass` takes the scene time from the product name. That time is the **datatake sensing start** (S2 product naming: https://sentiwiki.copernicus.eu/web/s2-products), not the time the satellite imaged the site.

A light check on 2026-10-09 grouped the 2023 labels in `products_*/overpass_times.json` by 10-day orbit phase. One phase is one relative orbit of one platform, which passes a fixed site at an almost constant UTC.
- Within one phase the labels spread over 7.3-8.5 min (SD 2.7-3.3 min) at Escalda, Ems and Wadden, and in two of the four Villaviciosa phases.
- They spread over 0.8-3.8 min at Ferrol and in the other two Villaviciosa phases.
- The Ems and Wadden labels are identical date by date, which is what datatake-start labels look like.
- The second reviewer's circular-orbit calculation gives possible windows of 10:40.5-10:58.0 UTC at the Ems and 11:17.5-11:31.9 UTC at Villaviciosa. Some phase medians fall before those windows.

Labels are therefore early by roughly 0-10 min, varying by date. The consequences:
- Every imagery lag is biased low by the mean offset relative to the gauge truths. The truths are computed at the same labelled instants, so they are almost unaffected.
- MAREA's pinned band 0 cannot absorb the offset.
- The scatter from date to date adds about 1-2 cm of level error per minute at mid-tide to every elevation method.

**Gate G-time** (new module `experiments/benchmark/sensing_time.py`, command `fetch sensing` plus `gate time`):
1. **Method A, sun-angle inversion.** For each date, read the sun zenith and azimuth at the site centroid and at the four frame corners:
   - either from the CDSE L2A collection's `sunZenithAngles` / `sunAzimuthAngles` bands (5 km; https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html), through one small openEO job per site and year;
   - or from the tile metadata angle grids (MTD_TL.xml).

   Solve for the UTC time t in [label − 5 min, label + 30 min] that matches both angles under the NREL solar position algorithm (Reda & Andreas 2004, doi:10.1016/j.solener.2003.12.003). Record the zenith residual.
2. **Method B, orbit propagation.** Take historical TLEs for Sentinel-2A, 2B and 2C (Space-Track; NORAD 40697, 42063, 60989) and propagate them with SGP4 (skyfield). The instant is the time of closest approach of the sub-satellite track to the site centroid within [label, label + 30 min]. B runs for a sample of at least 10 dates per (site, relative orbit, platform, year). If A fails, B runs for every date.
3. **Criteria:**
   - |A − B| ≤ 0.5 min on the sample;
   - within each (site, relative orbit, platform, year) group, SD(instant) ≤ 0.5 min;
   - every instant falls inside the physically possible window for the site's latitude;
   - instant − label ≥ −0.5 min, because a datatake start cannot be later than the site time.
4. **Fallbacks:**
   - A date that fails takes the median instant of its group, flagged.
   - If method A fails the consistency criterion as a whole (for example because the angle grid is computed at one reference time per tile), method B becomes primary for every date.
   - If neither method passes, G-time fails. Lag scoring and every elevation score at that site wait for the user's decision on whether to proceed with labels plus a per-orbit offset sensitivity [V21].
5. **Outputs:** `cubes/<site>/sensing_times.json`, with per date the label, A, B (where computed), the relative orbit, the platform, the sun zenith and azimuth, and the view zenith and azimuth. The report prints the correction per site and relative orbit.
6. **Consequences:**
   - The record instants, the level tables, h_B at the instants and every lag truth are built from the G-time instants.
   - The production MAREA products and the v8 Granadeiro runs used label instants. They remain only as "production reference (label instants)" rows.
   - A declared sensitivity reruns MAREA and no delay at P0 with label instants, which quantifies the effect. The report re-examines the earlier production lags (the negative Ferrol lag, the low Escalda lag) in its light, descriptively.
   - G-marea (§10.1) still reproduces production on legacy inputs with label instants, because it proves code equivalence, not timing.

### 1.4 Record and windows

- **Record of a window:** the Zarr dates in `[start, end)` that have a valid G-time instant and a finite h_B(t), in time order. Positions are global within the union period, so subsets and level rows are shared across windows.
- **Windows** are defined in §2.7. A window never requires rewriting a store; it is a selection.
- **Same-date double overpasses:** the store holds one mosaic per date. Two tiles of one datatake share an instant to within seconds. Two relative orbits on one date would not; `inputs` counts any such dates per site and declares them.

### 1.5 Boundary h_B over 2017-2025 (built once)

The boundary provider (`levels.make_provider`, v13 §2.7) is built **once per site over the union period**, after every IOC chunk of 2017-2025 has been downloaded, and is then frozen. Building it per window, or before the old chunks are in the cache, would change h_B silently, for two reasons:

1. `GaugeBoundary` returns NaN outside its record, and `EnsembleBoundary` averages with `nanmean`. Instants without gauge data silently become EOT20-only.
2. `make_boundary` demeans each gauge by the median of **all** cached samples, and `load_cached_ioc` concatenates every cached chunk of a code. Adding the 2017-2022 chunks therefore shifts h_B's datum at every instant, including 2023-2025.

Rules:
- `fetch ioc` downloads 2017-2025 for every boundary member (vlis, brsk, ters, bork, fer1) and for Gijón (gij2) before any `prepare`.
- `inputs` reports the gauge coverage per window.
- The IOC chunk hashes enter `package_key` (v13 §2.11).
- Where a member has a gap, the level is EOT20-only there. The share of such instants is reported per window.
- Declared sensitivities:
  - "boundary = EOT20 only", run on the base cell at W3 (Campaign 1) and at W9 (Campaign 2) [V5];
  - at Villaviciosa, "boundary = EOT20 + Gijón residual" and the Cantabrian consensus of `experiments/c2_boundary_variants.py`. This covers the non-tidal residual (surge, inverse barometer, seasonal mean sea level), which clear-sky sampling biases toward anticyclonic weather.
- `levels.boundary_series_1min` drops its assertion on v8's `TIME_EXTENT`. It takes the period as an argument and keeps v8's cache tag format.

### 1.6 Water-level sources for elevation (new factor, elevation units only) [V22]

Under common conditions every method is heighted with h_B. That is exactly the deficiency MAREA's lag corrects, whereas in practice the competitors use local levels:
- Costa et al. (2023) use in-harbour gauges;
- Mason et al. (2010) use a hydrodynamic model;
- Granadeiro et al. (2021) use local tide predictions;
- DEA Intertidal uses per-pixel tides from a locally ranked ensemble of global models (https://knowledge.dea.ga.gov.au/data/product/dea-intertidal).

Family P2 is therefore read as "given only a boundary tide", and a water-level-source factor gives the practical comparison.

| Level | Definition | Sites | Role |
|---|---|---|---|
| **LB** (base) | h_B(t), exactly as in §1.5 | all | Common conditions; P0; families P1, P2 |
| LT | Per-pixel global tide model through `eo_tides` (Bishop-Taylor et al. 2025, doi:10.21105/joss.07786). EOT20, plus FES2022 once installed. Evaluated on a 1 km node grid, with each pixel taking its nearest node (declared). When at least 3 models are installed, the locally ranked ensemble is used: eo-tides' default ranking where it covers the sites, otherwise the truth-free BT26-rank model choice of L12 (declared). | all | DEA, NIDEM and Coast X-Ray practice; family P3 |
| LM | Regional hydrodynamic sea surface height including tides: CMEMS NWS AMM15 (1.5 km) at Escalda, Wadden and Ems, CMEMS IBI (1/36°) at the rías. The 15-min instantaneous 2-D SSH is preferred; hourly SSH is accepted with cubic interpolation (declared). The level at a pixel is the series of its 1-km s bin, built from the model's wet channel cells in the bin. Each series is demeaned against h_B over the window (declared). Bins whose nearest wet cell is more than 1 km away are flagged "unresolved". | all; degenerate at Villaviciosa (flagged) | Mason et al. (2010) practice; family P3 |
| LG | Gauge-network interpolation along s: at each instant, linear in s between the boundary at s = 0 and the judges' series at their s_g, constant beyond the last gauge. Uses the judges, so it is **never** used in lag scoring or in any confirmatory family. | escalda, wadden, ems | Labelled upper bound ("if you had in-estuary gauges"); Costa and classic-waterline as-published practice; headroom of the lag correction |

How a level source enters each method family (declared mappings, identical for all levels):
- **Per-pixel methods** (E0a, E2, E4, E5, E8, E11) take the source's series at the pixel. LB is the same for every pixel.
- **One-tide-per-scene methods** (E1 NIDEM, E7 Coast X-Ray) use ITEM's per-polygon tide-point mechanism. Under LT, LM and LG the method runs once per 3-km band polygon with that band's series, and the polygons are mosaicked. Under LB the single polygon is the whole site, as published.
- **Waterline methods** (E6a, E6b) height each vertex with the source's level at the vertex.
- **MAREA** always uses LB, by design: its lag is the correction. **No delay** runs under every source, so no delay(LG) is the headroom reference.
- **Granadeiro** under LT and LM is Tier 2, because it estimates its own lag map.

Rules:
- Level-source rows run at P0 only (§5.4).
- They are elevation units: their lag products, if any, are not scored.
- Every level source is fetched and frozen before lock-A: `fetch cmems`, then `fetch tides`, then `levels --sources`.
- `fetch cmems --dry-run` checks the dataset ids, the temporal resolution and the archive depth before lock-A. A source that is missing for a window is NOT AVAILABLE there, never substituted.

### 1.7 Gauges: judges, coverage and conventions

- **Judges:** the v13 rules (§2.8): in the frame; at least 50 % of record instants within 6 min; more than 3 km from boundary members; not a dry gauge. The Terneuzen pair counts as one location. Neighbourhood radius R = max(500 m, d_min + 300 m), widened by 250 m until it holds at least 100 keep pixels, capped at 2 km.
  - Headline set: 13 locations (Escalda 2, Wadden 2, Ems 8, Ferrol 1).
  - The coverage rule is evaluated **per window**.
  - Geometry is computed once, on the union-record domain under P0.
- **Time-stamp conventions:** `fetch judges` records each provider's averaging window and stamping convention (start, centre or end of the interval) from the provider metadata. Series are centred before use. A declared sensitivity shifts each series by ± half its averaging window.
- **Development-free subset:** before lock-A, the authors list every judge whose comparison influenced a MAREA parameter. Harlingen is on the list, because TAU_BOUNDS was widened after the Harlingen comparison. The list is frozen in LOCK.json (§9.9), and every lag result is also reported on the judges not on it.
- **Coverage today:** every judge CSV in `data_v4/gauges/estuary_truth/` covers 2023-01-01..2025-12-31 only. Lag truths therefore exist for W1, W2 and W3 and the 2023-2025 calendar years. Other windows wait until the series are extended to 2017-2022 with the existing scripts in `data_v4/gauges/estuary_truth/scripts/` (RWS DD-API, WSV, IOC, PdE) and re-run through their QC [V5]. Until then, the report states this limitation.

### 1.8 Truths and epochs

| Truth | Source | Epoch | Use |
|---|---|---|---|
| Vaklodingen | `data_v4/truth/vaklodingen*.nc`, 20 m RD cells | Annual labels (survey day unknown, except where a source filename embeds it). Escalda every year 2016-2025; Wadden Vlie 2016, 2019, 2022, 2025 (inner sheets to 2022); Ems Dutch and German sheets 2020, outer sheets 2019, 2022, 2025 | Elevation truth at Escalda, Wadden, Ems |
| PNOA LiDAR MDT02 | `data_v4/truth/pnoa/MDT02-WGS84-0015-3-COB2.tif`, ground only, aggregated to 10 m (`v10.lidar_truth`), water-surface masked (below) | Second coverage (programme 2015-2021). The flight date and time come from the CNIG metadata [V18]. | Elevation truth at Villaviciosa |
| RTK | `pyintertidal.rtk.load_rtk()` with default arguments | Campaign of 2026-08-14 (all points; read from the file's time column) | Descriptive point truth at Villaviciosa |
| Judge series | `data_v4/gauges/estuary_truth/` | 2023-2025 (extension pending [V5]) | Lag truths; LG levels (elevation only); HPE reference level (§7.3.1) |

**Vaklodingen (interpolated soundings).** Vaklodingen are 20 m grids interpolated from echosounding and, where it exists, LiDAR, compiled per sheet and year. On the flats much of the data is boat sounding along widely spaced transects, so the intertidal flats are the least constrained part of the grid.
- `fetch vak-meta` collects per sheet and year the source method (sounding or LiDAR), the transect spacing and the data density (RWS/Deltares OpenEarth Vaklodingen documentation) [V25]. These feed a "source class" stratum and a sensitivity restricted to directly measured cells.
- At the Dutch sites the **20 m truth cell is the scoring support**. A product's value in a cell is the mean of its mapped 10 m pixels, and the cell counts if at least 2 of its 4 pixels are mapped. The 10 m pixel view is a sensitivity.
- If RWS intertidal laser altimetry exists for the Westerschelde in 2023-2025, it is added as a second truth at Escalda [V25].

**PNOA (water surface at flight time).** 1064 nm LiDAR does not reach the bed under water; it records the water surface at flight time, or an interpolation. The repository's notes record an earlier IGN LiDAR over the Villaviciosa flat with 83 % of the flat at +2.00 m. A light check of MDT02 COB2 over the ría found constant-value spikes of about 1-2 % each near −0.56/−0.58 m and −0.96/−1.00 m, consistent with flattened water bodies. Therefore:
1. `fetch pnoa-meta` reads the flight date and time per strip from the CNIG metadata.
2. The water level at flight time comes from the Gijón gauge (gij2) plus EOT20's local phase.
3. Truth below that level + 0.15 m is masked before the range filter.
4. Water is detected as connected constant-value patches (at least 50 px of values equal to within 1 mm), replacing the 3 × 3 flatness rule of `v10.lidar_truth`. That rule also discarded genuinely flat dry sand and kept rippled water.
5. The report states PNOA's vertical accuracy (from the CNIG metadata), the epoch gap, and the elevation zones the masked truth still covers. Ground-only returns at flight tide probably truncate the lower intertidal, so the zones Villaviciosa can test are reported.
6. If the flight time cannot be found, only the constant-patch mask is applied and Villaviciosa elevation is descriptive.

Changes to `truth.py`:
- `truth_grid(site, year=Y)` keeps every survey per cell with its year (v1 kept only the latest in 2019-2025).
- **Survey-matched truth is the default for every window:** each truth cell takes the survey year nearest the window centre (ties go to the later survey). "Latest survey per cell 2019-2025" (v13) stays as a continuity view only.
- Every truth cell carries its **epoch gap** |survey epoch − window centre| in years. Epoch convention: mid-year, ±0.5 year.
- **Range filter for D_truth:** [q_0.001(h_B) − 0.25 m, q_0.999(h_B) + 0.25 m], from the frozen 1-min h_B series over 2017-2025. It needs only the IOC fetch, not the Zarr stores, so D_truth is fixed at lock-A and identical for every cell and window.
  - Per band, the counts excluded above and below the range are reported.
  - A sensitivity uses the local range of the LG series (truth-free for elevation) where judges exist, so amplified reaches keep their truth.
- **Stable-cell stratum:** at least 2 surveys within the window ± 1 year, and a truth range over them ≤ 0.20 m [V4]. Otherwise the cell is "not assessable". The report gives the stratum's elevation-zone composition.

### 1.9 Truth-error model (method-free, frozen at lock-A)

Errors in the truth attenuate OLS slopes and inflate every method's RMSE by the same amount. The model below is estimated from the truths alone, before any product is scored, and frozen in FROZEN['truth_error'].

- **Survey noise σ²_s:**
  - At Escalda, from yearly survey pairs: D(Δt) = Var(t_Y1 − t_Y2) = 2σ²_s + g(Δt), with g(0) = 0 and g monotone increasing. The intercept is extrapolated by an isotonic fit plus a power law, and the larger σ²_s of the two is taken (conservative).
  - At Wadden and Ems: from same-year overlapping sheets where they exist, otherwise Escalda's value (declared).
  - At Villaviciosa: PNOA's stated vertical accuracy, reduced for the 2 m → 10 m aggregation of its random part.
- **Epoch mismatch D_W(Y):** the method-free variance between the window-averaged truth surface and survey Y. It is estimated from Escalda's yearly surveys, with every survey's noise counted once (the noise terms are removed analytically). It is transferred to Wadden and Ems through the ratio of their D(Δt) curves (declared).
- **Total:** σ²_T(site, window, view) = σ²_s + D_W(Y), where Y is the truth epoch used by the view. The reliability ratio is λ̂ = 1 − σ²_T / Var(t) on the view.
- **Sensitivity:** σ²_T × {0.5, 1.5} in every quantity that uses it: b*, δ_m (§7.2), s_method (§7.4) and the mismatch model (§7.5).

Light-check orientation values (approximate intertidal band): Escalda RMS dz is 0.29 m at 1 year apart and 0.72 m at 9 years, with repeat-survey noise of about 0.12 m; Wadden 0.23-0.46 m; Ems 0.21-0.30 m.

### 1.10 Scene metadata sidecar

The Zarr stores keep no tile cloud cover, processing baseline, platform or angles. `fetch stac-meta` queries the CDSE STAC once per site and year and writes `cubes/<site>/scene_meta.json`, with per date:
- `eo:cloud_cover`;
- `s2:processing_baseline`;
- platform (S2A, S2B, S2C);
- relative and absolute orbit;
- granule ids;
- mean sun and view zenith and azimuth.

The G-time outputs are merged in. The sidecar serves:
- **The offset regime per scene.** CDSE serves Collection-1 reprocessed L2A: baseline 05.00 for 2015-07-04..2021-12-31 and 05.10 from 2022-01-01 (https://documentation.dataspace.copernicus.eu/Data/Others/Sentinel2_L2A_baseline.html). Which baseline a given date actually carries depends on the reprocessing status at download time. Therefore the BOA offset regime and the Sen2Cor generation are **decided per scene** from `s2:processing_baseline` and the store's per-date `offset_state`, never from the date. The checks "values ≥ 0 before 2022-01-25" and "SCL dilation only from 2022" of revision 1 are withdrawn.
- **The tile cloud-cover rule** (C9, Tier 2).
- **The platform mix** (Sentinel-2C in 2025) and the orbit classes.
- **Geometry diagnostics** (§2.10, §6.4): sun and view angles per scene, relative orbit and season.

The query runs per year, and fails if the last page of results still carries a `next` link (`pyintertidal.overpass` stops silently after 50 pages).

**Amendment of the per-scene offset rule (2026-10-19, before lock-A; declared, nothing tuned on truth data).** The frozen rule of `inputs.OFFSET_RULE` (V1: (1) negative values → 0; (2) baselines < 04.00 → 0; (3) the SCL-water B12; (4) the frame B12; (5) the unanimous regime of the year's strongly decided scenes) is amended as below. The steps keep their V1 labels and are tested in the order (0), (1), (4), (2), (3), then (5) per year. `OFFSET_RULE["revision"]` records the V1 text and parameters, every change, the reason, the evidence and a negative control. Every inputs report carries the rule and the sha256 of the rule and its code (the prep levels of §2.3 hash the same), and `prepare` refuses a report written with another rule (rerun `inputs`).
- **(0), new and strongest:** a scene whose B03 and B08 equal the offset-harmonised legacy netCDF cube on its date → 0. The per-date identity comes from the store-year's `<year>.verify.json` (G-zarr, §1.11 item 4): every common date when the verdict is 'identical' (or 'identical on the common window'), or when the B03 and B08 counts show no difference; with 'differs', a common date absent from the listed differing dates when that list is complete (not truncated at `max_dates_listed`), or listed with differences in SCL only. Nothing is granted when the comparison did not cover the store's whole frame (its per-date counts cover the common window only), or when the file's date counts or per-date B12 counts are not the store's (a verification of another download). It is not applied to a scene with B12 valid on < 1 % of the frame. The legacy cubes come from the same backend, so identity only shows that both cubes carry the same offset: a store-year uses it only when one of its identical dates also shows values impossible with +1000 (rule (1), rule (4) or the dark clear water of (3)). An identical date whose own pixels say '+1000' stays 0 and is listed in the report's warnings; the report also keeps what rules (1)-(5) alone decide on each scene the identity decided.
- **(1)** needs ≥ 100 negative values (V1: any; with +1000 a negative int16 can only be an artefact).
- **(4), dark pixels first:** a frame share of B12 DN in [1, 500) ≥ 0.002 (on ≥ 1 % of the frame valid) → 0, tested before (2) and (3) (V1: > 0.01, after them). With +1000 such a DN would be a reflectance below −0.05; 0.002 is the downloader's own year-level '+1000' bound.
- **(3), clear water under a clear sky (also in (2)):** the water evidence counts SCL 6 pixels with CLD ≤ 10 % and a valid B12 (`n_water_clear`, `n_water_clear_low`, read in the same pass as B12 and SCL; the SCL-water counts `n_water`, `n_water_low` stay in the report). A share of B12 DN < 500 ≥ 0.1 (V1: 0.5) → 0. A share ≤ 0.02 → 1000 only when the frame's SCL 8/9/10 share is ≤ 0.5 (a cloud guard that does not depend on CLD); under more cloud it is no strong evidence. A share between 0.02 and 0.1 is no strong evidence (V1: 'unclear', which failed the check without trying (4) or the regime). Statistics written before the amendment have no clear-water counts: they fall back to the SCL-water counts, and the reason says so.
- **(5), per year** (scenes whose baselines are not all < 04.00): the strong 1000 become weak when the year has ≥ 10 strong 0 and they are ≤ 2 % of its strong decisions (listed in the warnings). When both values remain, the strong decisions of the smaller side are 'unclear' (both sides on a tie, and then the weak scenes too), so a contradiction fails the check even in a year without weak scenes. The other scenes take the regime of the remaining strong decisions, else the store's year `offset_state`, else 'unclear'. Strong 0 are never demoted.
- **Baselines:** the union of `scene_meta.json` and the stores' `input_baselines`. A date on both sides of 04.00 is 'mixed' (no rule-(2) shortcut) and is listed in the warnings; 'N0213' and '0213' read as 02.13.

Reason: the first `inputs` run on the complete Zarr stores (2026-10-18/19) failed at Villaviciosa ({'no offset': 235, 'unclear': 219, '+1000 in the values': 12}, T = 466) and at the Wadden ({'no offset': 319, 'unclear': 122, '+1000 in the values': 21}, T = 462). The rule decided '+1000' on scenes without any offset: bright SCL 'water' that is thick cloud, and clear-sky bright water (glint, haze). Wadden 2025-12-15 has 7,240 SCL-water pixels, none below 500 DN, 84 of 9,507,800 valid frame pixels with B12 in [1, 500) and no negative value. In the 20 m legacy cube that date is 94.6 % SCL 9 and 5.3 % SCL 8, and its SCL-6 pixels have B08 p5/50/95 = 3449/4554/6588 DN: cloud, not water. Their years' decided scenes then disagreed, so every weakly decided scene of those years became 'unclear'. This is a defect that blocked the data entry, not a change of method: the openEO format test of 2026-10-13 found no offset in the delivered values (processing baseline 05.11), and the legacy cubes are offset-harmonised (negative B08 values, impossible with +1000, on 10-46 dates of every year 2016-2025 of the Villaviciosa 10-year cube).

Check, not fit: on a negative control built from the harmonised Villaviciosa SWIR legacy cube (465 scenes; B11 stands in for B12; every SCL-water pixel taken as clear water, the worst case for CLD), the amended rule gives 0 wrong and 0 'unclear' (2 isolated '+1000' demoted). The same pixels with the +1000 put back give 0 wrong and 0 'unclear' (177 strong '+1000'). A first draft of this amendment (identity and clear water only) gave 21 false strong '+1000' and 226 'unclear' on that control. Not verified: whether CLD ≤ 10 % removes the cloud labelled water (no legacy cube has CLD). The rerun of `inputs` reports `n_water`, `n_water_clear` and the SCL cloud counts of every scene.

### 1.11 Data checks (`inputs` command, run by the user before `prepare`)

For each site and year store, `inputs` checks the items below and writes `results/validation_v2/common/inputs_report.json`:

1. **Store:** `complete = 1`; FORMAT_ID; dims; dtype and `_FillValue` per band; no active scale or offset.
2. **Codes and ranges:**
   - SCL codes in 0-11, and the count of SCL 11 (snow/ice) per site and year;
   - CLD in 0-100 or fill;
   - the per-scene offset regime (§1.10). 'unclear' fails the check. The frozen per-scene rule (subtract 1000 DN where the delivered values carry the offset) is written before lock-A.
   - On permanent-water anchors (water frequency ≥ 0.98), the dark-water B08 and NDWI distributions are compared between scenes of the two regimes. A step larger than 0.02 in median NDWI is reported.
3. **Grids:** identical across years; equal to the legacy cube grid and, at Villaviciosa, to `rtk.DEFAULT_GRID`; the 20 m lattice phase (`lattice20`).
4. **G-zarr:** for every 2023-2025 date present in both, Zarr B03/B08/SCL must equal the legacy netCDF bitwise (`zarr_cubes.compare_with_netcdf`; its `<year>.verify.json` is read if present). Any difference is quantified. **Pre-registered response if G-zarr fails for any 2023-2025 year:**
   - production MAREA is rerun on the Zarr inputs, with the production settings and G-time instants, and becomes the as-published reference;
   - G-marea stays on the legacy inputs;
   - the difference is declared.
5. **Sensing times:** the G-time verdict per site (§1.3), and the number of same-date double overpasses.
6. **Coverage:** IOC coverage per boundary member and window; judge coverage per window; the share of EOT20-only instants; the availability of each level source per window (§1.6).
7. **Scene meta:** the sidecar is present; processing baselines, platforms and relative orbits per year.
8. **Disk:** free space and the measured sequential read throughput on the volumes holding the cubes, the package, the results and TMPDIR, against the §14 budget [V9].

`inputs` writes no package and is safe to repeat.

---

## 2. Settings taxonomy

All settings are defined in `experiments/benchmark/settings.py` as registries. Each entry is a frozen dataclass with:
- a short id;
- the bands it needs;
- a reference;
- a deterministic, truth-blind implementation in `obs.py` (pixel rules, indices, detectors) or `package.py` (scene rules, per-scene thresholds, buffers).

Each registry is hashed **per entry** (the source of the level and its helpers) into its own FROZEN section (`cloud`, `prep`, `index`, `detect`, `window`, `level`, `curves`). Adding a level therefore never changes `package_key` or the `code_key` of existing units (§13.2).

### 2.1 Factor overview

| Factor | Base (P0) | Campaign 1 levels | In the combination grid G1' | Campaign 2 | Tier 2 |
|---|---|---|---|---|---|
| Cloud handling C | C1 | C0 (on S0), C1, C1d, C2, C3, C4, C6, C8, C11 | C1, C2, C4, C11 (C3 if C11 is unavailable) | C5, C7; class-11-blocked sensitivity | C1s, C3 with p ∈ {10, 35, 50}, buffers ×0.5 and ×2, C9 tile rule, C10 anchor QA, C12 s2cloudless (only if CDSE exposes it), buffered C11 |
| Band preprocessing | floor0 | floor0, raw | – | – | – |
| Spectral index I | NDWI | NDWI, MNDWI, NIR, AWEI_sh, NDWI@20 | NDWI, MNDWI, NIR | AWEI_nsh, WRI; NDWI@20 slice of G1' | WI2015, MNDWI-mixed (10 m B03 with replicated B11), sharpened MNDWI |
| Water detection D | D1 fixed | D1, D2, D3, D5 | D1, D3 | D4 (naive baseline) | guarded GMM, frame Otsu, binarised input, HSV (Khan), Tide2Topo rule, Dynamic World |
| Water-level source L (elevation only) | LB | LB, LT, LM, LG at P0 (§1.6) | LB | – | Granadeiro under LT/LM; MAREA with the lag estimated relative to LM |
| Window | W3 (2023-2025) | W3; W1 (native cells); calendar years 2023, 2024, 2025 | W3 | W2, W5, W9; calendar years 2017-2022; survey-centred windows; G2 | – |
| N | all | K1' grid (§2.8) | – | K3-K6 | K2 (F-mode), K7-K9 |

### 2.2 Cloud handling

There is one factor because the user's four named options mix pixel masks and scene rules. Pixel masks act per observation. Scene rules act on the record, after the A-mode draw of N archive scenes, and are always combined with the C1 pixel mask.

In every level:
- V (valid) also requires every band the index needs to be present (≠ fill) and the index to be finite.
- No data never counts as cloud.
- Neighbourhood operations (opening, dilation) are computed per scene in the package's settings pass. They are restricted to the bounding box of dom plus 1 km and stored as packed bits at dom.

| Level | Kind | Rule | Reference | Role |
|---|---|---|---|---|
| **C0** none | pixel | V = bands present & SCL ≠ 0 | design control | Lower bound, run as one OFAT cell evaluated on S0 only (§2.10). Never used for conclusions about a method's quality; not in any grid. |
| **C1** SCL allowlist | pixel | V = SCL ∈ {4, 5, 6, 7} | `marea.CLEAR`, `obs.mask_valid('marea_clear')`; Sen2Cor ATBD 2.10 | **Base.** MAREA production; the user's "per-pixel cloud mask" |
| C1d allowlist + dark area | pixel | V = SCL ∈ {2, 4, 5, 6, 7} | this study | MNAR check: freshly exposed wet mud is dark and is often labelled class 2 or 3 |
| **C2** transition-zone rule | scene + C1 | Keep a date iff its SCL{3,8,9,10} share inside the transition zone is ≤ 10 %, or the whole-frame share when the frame is ≤ 5 % bad (`stability.reference_and_clouds`, `filter_dates(·, 0.10)`) | v12; `results/scene_rule_study_2026-09-30/` | MAREA's as-published record; the user's "transition-zone rule" |
| **C3** CLD probability + buffer | pixel | bad = (CLD ≥ p) \| SCL = 10; bad_o = binary opening with a disk of 2 px; masked = EDT(¬bad_o) ≤ 8 px (80 m) \| EDT(¬(SCL = 3)) ≤ 4 px (40 m) \| SCL ∈ {0, 1} \| CLD fill | Sen2Cor ATBD 2.10 §3.8-3.9; sen2chain CM001; DEA `load_ard` mask_filters | A Sen2Cor-based "other filtering method". p frozen by the cross-tab [V1]. |
| **C4** ITEM-PQ blocklist | pixel | V = SCL present & SCL ∉ {0, 1, 3, 8, 9, 10} | Sagar et al. 2017; `waterline_contour.ITEM_PQ_MASKED_SCL` | NIDEM native (competitor-native level in G1'). Keeps SCL 11; a class-11-blocked sensitivity is Campaign 2. |
| C5 blocklist + buffer | pixel | C4, plus SCL{8,9,10} opened with a disk of 2 px and dilated 80 m, and SCL 3 dilated 40 m | DEA `load_ard` mask_filters | DEA-style mask (Campaign 2) |
| **C6** g21 frame rule | scene + C1 | Frame SCL{3,8,9,10} share < 0.10 and no-data share ≤ 0.02 | Granadeiro et al. 2021; `granadeiro2021.scene_rule` | Granadeiro native |
| C7 ard rule | scene + C1 | `load_ard` good fraction (SCL ∉ {1,3,8,9,10}, no data counted good) ≥ 0.5 | DE Africa/DEA `load_ard`; `waterline_contour.s2_scene_rule` | NIDEM-S2 native (Campaign 2; used in the as-published NIDEM-S2 variant) |
| **C8** domain rule | scene + C1 | Share of keep_full(P0) valid under C1 ≥ 0.5 | CoastSat-style rule on the region of interest (Vos et al. 2019) | Comparable across sites, unlike frame rules |
| **C11** OmniCloudMask | pixel | V = bands present & SCL ≠ 0 & OmniCloudMask class = clear. The inputs are the red, green and NIR bands the package documents for Sentinel-2; the band choice (B08 or B8A), the inference resolution and the patch settings are fixed in the card from the package documentation before lock-A. CPU inference with a pinned version and deterministic kernels. | Wright et al. 2025, doi:10.1016/j.rse.2025.114694 | Independent, non-Sen2Cor mask (the reviewers' state-of-the-art "other filtering method"). In G1' [V1]. |

**Why C11 is in Tier 1.** C0-C8 are all functions of the Sen2Cor SCL and CLD layers, so they share Sen2Cor's failure modes over estuaries. Those include commission over bright surfaces and turbid water (Baetens et al. 2019, doi:10.3390/rs11040433; Skakun et al. 2022, doi:10.1016/j.rse.2022.112990). C11 is the only independent mask that the stored bands support.
- It runs in Stage 0 as `prepare --step ocm`, restricted to the bounding box of dom plus 1 km, and is stored as packed bits at dom.
- Estimated cost: 10-20 CPU-hours at W3 over the five sites, scaled from revision 1's estimate of 20-40 CPU-hours for the Ems over 9 years.
- If the package cannot be installed on the server, C3 replaces C11 in G1' and the substitution is declared.

**s2cloudless (CLP/CLM).** On 2026-10-09 the CDSE L2A collection page did not list CLP or CLM. `check-env` queries the collection metadata. If the layers are exposed, they are added as the cheap Tier-2 level C12. Fmask needs L1C and stays excluded.

**CLD cross-tab** (`pilot crosstab`, truth-free; it freezes p and must run before lock-A). Per site and year it reports:
- the SCL × CLD cross-tabulation, giving the effective CLD cut-offs of SCL 7, 8 and 9, per processing baseline;
- the CLD distribution on permanent-water anchors (water frequency ≥ 0.98), on stable land, on **dry flats at low tide** (keep pixels whose full-record water frequency is ≤ 0.3, in scenes with h_B below its 25th percentile), and on **turbid channel water** (channel anchors from EMODnet depth ≥ 2 m below LAT, not from NDWI frequency).

Default p = 20 %. The mechanical rule raises p to 35 % if the median CLD over any of the three anchor classes exceeds 20 % in more than 10 % of scenes at any site. The rule is written into FROZEN before the cross-tab runs.

**State-dependent masking (MNAR diagnostic, truth-free, Stage 0).** On tidal flats, masking depends on the wet/dry state:
- wet mud is dark in NIR and SWIR and attracts SCL classes 2 and 3;
- bright dry sand and turbid water attract the cloud and probability classes.

For every site, window and Tier-1 cloud level, pass 1 therefore stores the masked rate on keep_full(P0) against the h_B decile and the 3-km s band, broken down by SCL class. The slope of a logistic fit of "masked" on h_B is a per-cell covariate in the factor analysis (§9.4). The report shows the curves before any truth is read. C1d (adding class 2) isolates the dark-area effect. C4 differs from C1 in classes 2 and 11 at once, so it cannot.

**Transition zone for C2:** computed once per site and window under the production observation (NDWI > 0, clear classes {4, 5, 6}). The scene rule therefore does not change with the index or detector factor (declared).

**Matched-N controls for scene rules.** C2, C6 and C8 (and C7 in Campaign 2) change both the quality and the number of scenes. Each scene-rule cell is paired with a **C1 cell at the same N**, where N_rule is the rule's kept count:
- the cheap per-pixel methods and the MAREA pair get 3 random draws of N_rule scenes from the C1 record;
- frame methods get 1 draw.

Comparing the two separates selection quality from quantity (v13 §3.9; scene-rule study of 2026-09-30).

**Tier-2 cloud levels:**
- C1s: the strict allowlist {4, 5, 6}.
- C3 with p ∈ {10, 35, 50}.
- C3 and C5 with radii ×0.5 and ×2.
- C9: tile `eo:cloud_cover` ≤ 90 % (DEA Intertidal's rule; needs the sidecar).
- C10: anchor-consistency scene QA, new and declared. A scene is dropped when more than 20 % of its valid water anchors read dry, or more than 20 % of its land anchors read wet, with at least 500 anchors. Anchors come from the full record under P0.
- C11 with a 2 px opening and an 80 m dilation.
- C12: s2cloudless, only if CDSE exposes it.

### 2.3 Band preprocessing

| Level | Rule | Role |
|---|---|---|
| **floor0** | Every band an index uses: ρ = max(DN − o_s, 0)/10000 for non-fill values, where o_s ∈ {0, 1000} is the scene's frozen offset (§1.10) | **Base** (v13 D2) |
| raw | ρ = (DN − o_s)/10000, with no floor | OFAT sensitivity; MAREA's as-published preprocessing |

Notes:
- floor0 applies to every band of every index.
- On the production inputs o_s = 0, so the NDWI path stays byte-identical to production (gate G-marea).
- Offset-sensitive quantities (AWEI's coefficients sum to 0.25; WRI; the Y-layer thresholds) depend on the per-scene rule being right. `inputs` therefore fails on any 'unclear' scene.

### 2.4 Spectral index

Reflectances ρ come from §2.3 with a fixed 1/10000 scaling. The data-dependent AWEI scaling heuristic of `pyintertidal.water.index_block` is never reused. Every index is oriented so that water is high.

**Resolution rule.**
- Every index that uses a 20 m band is computed **at 20 m**: each 10 m band it uses is averaged over the 2 × 2 blocks of the native 20 m lattice (phase `lattice20`, §1.2); the 20 m bands are taken once per block (nearest-neighbour replication is exactly invertible); the index is computed on the blocks and replicated to 10 m.
- If `lattice20` is None (the 20 m bands are not exact 2 × 2 replicas), the 2 × 2 box mean on phase (0, 0) is used and declared.
- This keeps the index effect from mixing with the artificial sub-20 m texture of hybrid 10/20 m ratios.
- **NDWI@20** (2 × 2-mean B03 with B8A) splits any MNDWI − NDWI difference into a spectral part (MNDWI − NDWI@20) and a resolution part (NDWI@20 − NDWI).

| Level | Formula | Bands | Effective resolution | Fixed threshold t_k (D1) | Reference |
|---|---|---|---|---|---|
| **I1 NDWI** (base) | (ρ3 − ρ8)/(ρ3 + ρ8), NaN when the denominator is 0 | B03, B08 | 10 m | 0 | McFeeters 1996, doi:10.1080/01431169608948714 |
| I2 MNDWI | (ρ3 − ρ11)/(ρ3 + ρ11) | B03 (2 × 2 mean), B11 | 20 m | 0 | Xu 2006, doi:10.1080/01431160600589179 |
| I3 NIR | −ρ8 | B08 | 10 m | −t_NIR, where t_NIR = the pooled Otsu of ρ8 over the clear observations of keep_full(P0, W3) per site, from per-year histograms, frozen | Granadeiro et al. 2021; Bué et al. 2020 |
| I4 AWEI_sh | ρ2 + 2.5 ρ3 − 1.5 (ρ8 + ρ11) − 0.25 ρ12 | B02, B03, B08 (2 × 2 means), B11, B12 | 20 m | 0 | Feyisa et al. 2014, doi:10.1016/j.rse.2013.08.029 |
| I5 AWEI_nsh (Campaign 2) | 4 (ρ3 − ρ11) − (0.25 ρ8 + 2.75 ρ12) | B03, B08 (2 × 2 means), B11, B12 | 20 m | 0 | Feyisa et al. 2014 |
| I6 WRI (Campaign 2) | (ρ3 + ρ4)/(ρ8 + ρ11), clipped to [0, 10]; histogram detectors use log(WRI) | B03, B04, B08 (2 × 2 means), B11 | 20 m | 1 | Shen & Li 2010, doi:10.1109/GEOINFORMATICS.2010.5567762 |
| I7 NDWI@20 | (ρ3 − ρ8A)/(ρ3 + ρ8A) | B03 (2 × 2 mean), B8A | 20 m | 0 | resolution-matched control |

Notes:
- **The NDWI path is byte-identical to production** (`obs.ndwi`, `marea.extract`). The gates depend on it.
- **Tier 2:**
  - WI2015 (Fisher et al. 2016, doi:10.1016/j.rse.2015.12.055; threshold 0);
  - MNDWI-mixed, revision 1's hybrid MNDWI (10 m B03 with replicated B11), as a declared sensitivity;
  - MNDWI with B11 sharpened to 10 m (Du et al. 2016, doi:10.3390/rs8040354).
- **Truth-free index diagnostics,** reported per site and window before any truth is read:
  - the median Spearman correlation of the index with h_B on intertidal pixels;
  - the step-fit inconsistency (the minimum share of observations that contradict "wet iff h > z");
  - the share of degenerate pixels (less than 5 % or more than 95 % wet);
  - the index on permanent-water and land anchors per relative orbit and season (geometry and phenology check).

  A revision-1 light check at Villaviciosa gave, for the first two measures, NDWI 0.561 / 0.084, WRI 0.547 / 0.097, MNDWI 0.535 / 0.116 and NIR 0.492 / 0.111. Those MNDWI and WRI values used the hybrid 10/20 m construction.

### 2.5 Index-to-method interface

Each method declares which observation layers it consumes. The harness builds the layers from the setting once per instance, so every method sees identical observations.

| Layer | Definition | Consumers |
|---|---|---|
| X | The raw oriented index (§2.4) | Methods whose published thresholds are in that index's units, used only at the native index |
| Y | Standardised index y = (x − t_k)/s_k. s_k = IQR_k/IQR_NDWI over the pooled clear observations of keep_full(P0, W3) per site, from per-year histograms; truth-blind, frozen, written to `run.json`. For NDWI, y = x exactly. | Continuous estimators: MAREA and the no-delay inversion, Granadeiro (scale-free anyway), Bué, GM20, DEA's rolling medians, Bell, XC-index, HD26-index |
| V | Valid mask under the cloud level | All |
| W | Wet = V & (detector says water) | Binary stages: the harness domain (extraction rule), MAREA's m2a lag, m2b, m2c, XC-wet, LOOP, HD26-wet, PPL, the NIDEM and Coast X-Ray land flags, DEA's candidate filters, LogiTide2DEM, the frequency methods, the waterline engines |
| bands | floor0 reflectances | Granadeiro (NIR and its SD(NDWI) mask), Bué (NIR), Khan (HSV of B02, B04, B08, B11) |
| levels | The level series of the unit's water-level source (§1.6) at the method's support | Every elevation method |

Rules:
1. **Thresholds stated in NDWI units** move with the index through the standardisation. They are applied on Y:
   - MAREA's guards (0.15 < b < 2.5, |a| < 2.5);
   - NIDEM's confidence SD ≤ 0.25;
   - DEA's 0.1 and 0.0;
   - Coast X-Ray's 0.2;
   - SIQ's soft half-width.

   For NDWI this changes nothing. For every other index it is a declared deviation. The as-published rows and the native cells keep the native index and thresholds.
2. NIDEM's filtered product (SD ≤ 0.25) is produced only for normalised-difference indices (NDWI, MNDWI, NDWI@20). For the others, only the unfiltered product is produced (declared).
3. The **binarised-input variant** (Tier 2) feeds continuous estimators Y = 2W − 1, so the detector effect becomes visible for them too [V12].
4. MAREA's inversion uses Y; its m2a lag and its domain use W. Under P0 (NDWI, D1, floor0, LB) the adapter reproduces `marea.reconstruct` (gate G-marea, §10.1).

### 2.6 Water detection

| Level | Rule | Reference | Notes |
|---|---|---|---|
| **D1** fixed (base) | W = V & (x > t_k), with t_k from §2.4 | Sagar et al. 2017; `marea.extract` | Base |
| D2 guarded per-scene Otsu | The domain is fixed for every cell and window: Ω = keep_full(P0, W3) ∪ a 300 m ring of P0 permanent water (full-record water frequency ≥ 0.98) and stable land (≤ 0.02), so both classes are always present (the Bmax idea). For each scene with ≥ 1,000 valid pixels in Ω: t_s = Otsu(values), η = between-class / total variance, and m = the minority share at t_s. Use t_s if η ≥ 0.75 and m ≥ 0.15; otherwise use t_k. | Otsu 1979; Cao et al. 2019 (Bmax Otsu), doi:10.3390/w11040786; hydrafloods `bmax_otsu` | 256 bins over the scene's valid range in Ω. Ω does not depend on N, the cell or the window, so thresholds are computed once per scene. |
| D3 edge-Otsu | b = x > t_k on V; Canny (σ = 1 px, thresholds 0.05) on b, restricted to V eroded by 2 px and to the bounding box of dom plus 1 km; keep 8-connected edges of ≥ 50 px; buffer = the edges dilated by 5 px (50 m), within V. If the buffer holds ≥ 500 pixels, t_s = Otsu(x[buffer]); else t_k. A scene whose valid share is < 20 % is skipped. | Donchyts et al. 2016, doi:10.3390/rs8050386; Markert et al. 2020, doi:10.3390/rs12152469 | Best in the Villaviciosa light check (NDWI step error 0.067 against 0.076 for fixed) |
| D4 SCL water (Campaign 2) | W = V & SCL = 6 | Sen2Cor ATBD 2.10 §3.8; `water.py` 'scl' | **Naive baseline.** Index-free, 20 m, and it reads the same layer as the C factor, so C and D are not independent under D4. The light check found 48 % of flat pixels degenerate. Excluded from every headline robustness set. |
| D5 majority vote | W = V & [(NDWI > 0) + (MNDWI > 0) + (AWEI_nsh > 0) ≥ 2] | in-house (F4); no published rule is claimed | Index-independent for W; effective resolution 20 m |

Implementation rules:
- Per-scene thresholds (D2, D3) are computed in one settings pass after pass 1, from per-scene, per-index histograms on Ω and on the D3 buffers. They are stored as `(T,)` arrays per index, with η, the minority share and the buffer size. New windows only add scenes.
- Truth-free diagnostics per site, reported before scoring: corr(t_s, h_B), which shows whether thresholds drift with the tide stage, and the fallback rate.
- k-means is not run, because in one dimension it equals Otsu.
- **Tier 2:**
  - guarded GMM (Ashman's D ≥ 2, both weights ≥ 0.1);
  - frame-wide Otsu;
  - the binarised-input variant;
  - Khan et al.'s HSV segmentation offered as a detector;
  - the Tide2Topo rule (NDWI > Otsu and NDVI < 0.1 and NDWI > NDVI);
  - the Dynamic World water probability (Brown et al. 2022, doi:10.1038/s41597-022-01307-4) as a learned Sentinel-2 detector, if its public model runs on the server.

### 2.7 Temporal windows

**Nested windows,** all ending 2025-12-31 (end-exclusive 2026-01-01):

| Id | Window | Expected scenes T | Expected n_clear per intertidal pixel (C1) | Campaign |
|---|---|---|---|---|
| W1 | 2025 | ~172 | ~50 | Campaign 1 (native cells), Campaign 2 |
| W2 | 2024-2025 | ~317 | ~95 | Campaign 2 |
| **W3** (base) | 2023-2025 | 461-465 | ~140 | Campaign 1 |
| W5 | 2021-2025 | ~750 | ~230 | Campaign 2 |
| W9 | 2017-2025 | ~1,290-1,390 | ~400 | Campaign 2 |

W1 (2025) holds about 19 % more dates than the other years (Sentinel-2C), so it is not representative of a generic year. The report says so.

**Window length is confounded with epoch.** Nested windows end in 2025, so their centres move from 2025.5 (W1) to 2021.5 (W9). Against the 2025 Escalda truth the epoch gap grows with length; against the 2020 inner-Ems truth it shrinks. Therefore:
- every window is scored primarily on the **survey-matched truth** (§1.8);
- the nested-window effect is **confirmed only at Escalda** (yearly surveys) and on the survey-centred design. At the other sites it is descriptive;
- only W5 and W9 contain scenes on both sides of any processing-baseline change (§1.10), so the per-scene baseline is a covariate in every window comparison.

**Calendar-year replicate windows** `Y2017`..`Y2025`: one per calendar year, scored on the survey-matched truth. They are nearly disjoint archives (they share no scenes, though they share the site and the climate), so they give the between-archive replication that nested subsets cannot (§9.1, §9.7).
- Campaign 1: Y2023, Y2024 and Y2025 (Y2025 = W1).
- Campaign 2: Y2017-Y2022.

**Survey-centred windows** (§5.9) are named `V<epoch>L<length>`, with the epoch at mid-year:

| Site | Windows | Purpose |
|---|---|---|
| escalda | V2019L3 = 2018-2020; V2021L3 = 2020-2022; V2023L3 = 2022-2024; V2021L1 = 2021; V2021L5 = 2019-2023 | Epoch, length and N disentangled at the site with yearly surveys |
| wadden | V2022L3 = 2021-2023; V2022L5 = 2020-2024; V2019L3 = 2018-2020 | Vlie surveys of 2019 and 2022 |
| ems | V2020L3 = 2019-2021; V2020L5 = 2018-2022 | Removes the 3-5-year mismatch of the 2020 inner-Ems truth |
| villaviciosa | trailing windows only, until the PNOA flight date is known [V18] | – |
| ferrol | none (no elevation truth) | – |

**Mismatch curve.** Every product of a window with centre c is scored against **every available survey year** Y, giving RMSE and bias as functions of Δt = Y − c. At Escalda, Δt ranges over about −8..+8 years.

**Long windows: non-stationarity and effective epoch.**
- A truth-free temporal split (the first half of the window's record against the second half, each run through the method unchanged) gives a per-pixel change map. It is distinct from the alternate-scene split-half of §7.3.4. Coverage and error are reported by change class.
- Each product's **effective epoch** is the observation-weighted centre of the valid observations per pixel. It is the reference for the epoch gap in W5/W9 products.

### 2.8 Number of images

N counts **archive scenes inside the window**, drawn **before** any cloud handling, so every cloud level and every method gets the same images. n_clear is reported as the second x-axis.

**Subset families** (stored in `common/<site>/subsets_<window>.npz`):

| Family | Definition |
|---|---|
| `rand` | perm_r = `default_rng([SEEDS['subsets'], SITE_ID, WINDOW_ID, r]).permutation(T_window)`. A curve point with R ≤ floor(T/N) uses **disjoint batches** only: batch r = perm_0[r·N:(r + 1)·N]. A point with R > floor(T/N) uses **nested random subsets** only, sel = sort(perm_r[:N]), and is labelled "nested". The two kinds are never mixed in one point. |
| `win` | Contiguous sub-windows of N scenes (v13 sensitivity; Tier 2) |
| `rule-<C>` | Records passing a scene rule (C2, C6, C8, and C7 in Campaign 2), applied after the draw |
| `matched-<C>` | C1-record subsets of size N_rule: 3 draws for per-pixel methods, 1 for frame methods (§2.2) |
| `summer` (Tier 2) | The May-September scenes of the window, the subset common in published waterline work; compared with a matched-N random draw |

Disjoint batches from one record share its tides, seasons and morphology. They are finite-population partitions of one archive, not independent archives. Between-archive replication comes from the calendar-year windows (§2.7).

**Grids and replicates** (fixed a priori, never adapted to truth):

| Window | N grid | Replicates R (d = disjoint, n = nested) | Campaign |
|---|---|---|---|
| W3 (K1') | {20, 45, 90, 180, 330, all} | 5d, 5d, 3d, 2d, 2n, 1 | Campaign 1 |
| W9 (K4) | {90, 180, 330, 465, 900, all} | 3d, 3d, 3d, 2d, 2n, 1 | Campaign 2 |
| W1 (K5) | {45, 86} | 3d, 2d | Campaign 2 |
| W2 (K5) | {45, 90, 158} | 5d, 3d, 2d | Campaign 2 |
| W5 (K5) | {45, 90, 172, 465} | 5d, 5d, 4d, 2n | Campaign 2 |
| W9 cross-window (K5) | {45, 172} | 5d, 5d | Campaign 2 |
| Granadeiro W3 (K6) | {45, 90, 180} | 3d, 3d, 2d | Campaign 2 |

Shared N values across windows (45, 90, 172 = |W1|, 465 = |W3|) give **span contrasts at equal count**. They are scored only on survey-matched or stable views, with the per-scene processing baseline as a covariate (§2.7).

**Modes:** A-mode is primary. F-mode (K2) is a Tier-2 diagnostic for the methods that use the harness domain (v13 §0.5).

**Aggregation:**
- A curve point is shown only when k/R ≥ 0.5 of its replicates are computable.
- Accuracy uses the jointly computable replicates (v13 D17).
- Subsets are never redrawn because a method fails.
- Curves are scored on an N-invariant pair view (§7.4).

**Expected:** with about 30 % clear observations, MAREA's extraction rule (more than 30 clear observations per pixel) needs N of about 100 or more in A-mode. The MAREA family is therefore expected to be NOT COMPUTABLE at N ≤ 45. That is the honest answer of its own pipeline.

### 2.9 Settings cell, setting id and effective key

- **Setting id:** `<cloud>.<prep>.<index>.<detect>[@<level>]`. Examples: `C1.floor0.ndwi.D1` (P0), `C11.floor0.mndwi.D3`, `C1.floor0.ndwi.D1@LM`. The level suffix is omitted for LB.
- **Window id:** `W1`..`W9`, `Y2017`..`Y2025` or `V<epoch>L<len>`.
- **Cell** = (setting id, window id).

Each method declares a **dependency signature**: the factors it reads, plus whether it consumes the harness domain. The harness domain's extraction rule depends on V and W, and therefore on C, I, D and prep. The planner projects each cell onto the method's signature to get the **effective key**. One unit runs per effective key, and the scorer maps its products to every cell that shares it. Examples:
- Granadeiro does not read W and uses its own SD(NDWI) mask, so it is independent of D.
- Khan is independent of I and D.
- MAREA reads W through its domain and lag, so it depends on D even though its inversion is continuous.
- Lag products do not read the level source.

The effective key is part of `inputs_key` (§13.2).

### 2.10 Supports, dom and pass-1 accounting

Revision 1 defined dom as the union of U_rec over every planned setting. A light check at Escalda (20 m, W3) showed why that fails:
- under C1, U_rec covers 12.9 % of the frame and keep_full 9.6 %;
- under C0, they cover 100 % and 69 %.

Including C0 made dom and the support the whole frame at every site. That inflated the package to about 480 GB at W9 and multiplied the work of every per-pixel method in every cell. Revision 2 fixes two supports, both frozen at lock-A.

**S0, the evaluation support (identical for every cell, window and method):**

S0 = D_truth ∪ the judges' neighbourhoods ∪ dilate(keep_full(P0, W3), 300 m) ∪ Ω (the D2 domain, §2.6).

- D_truth's range filter comes from the 2017-2025 1-min h_B series (§1.8), so S0 needs no Zarr year outside 2023-2025.
- **Per-pixel methods compute on S0 in every cell.** Products are stored on S0. Curve products are stored on D_truth and the neighbourhoods only.

**dom, the stored package domain:**

dom = S0 ∪ dilate(U_rec over the A-mode curve settings of 2023-2025 {P0, C2, C11 or C3}, 3 px).

- dom is frozen at lock-A with a `dom_version`, which enters each year shard's key. New years add shards for the same dom, and no 2023-2025 shard is rewritten.
- The `pilot dom` step prints |S0|, |dom|, |U_rec| and |keep_full| per setting and window. It refuses pass 2 if |dom| > 1.5 × |S0| at any site, or if the package projected at W9 exceeds the disk budget (§14.4).
- In Campaign 2, the share of W5/W9 A-mode keep pixels outside dom is measured and declared. Those pixels are outside the scoring domain anyway, because D_truth ⊂ S0.

**Harness domain per setting.** The extraction rule's counts come from full-frame per-year counters (uint8; no year exceeds 255 dates). Hence keep(setting) is exact on the full frame at N = all. The observations it needs exist only on dom, so the domain-consuming methods run on keep(setting) ∩ dom (declared).
- Under **C0**, keep(C0) ∩ dom is used, every product is evaluated on S0, and C0 never enters dom.

**Frame methods** (NIDEM, waterline engines, Coast X-Ray, the SD masks of Granadeiro, Bué and Costa, Khan, Pool) read a **package-backed frame view**:
- dom values are scattered into NaN frames;
- outside dom, pixels are filled from the P0 full-record classes: water where the frequency is ≥ 0.98, land where it is ≤ 0.02, NaN elsewhere (declared).

Curve units therefore never touch the Zarr stores. Gate G-frameview (§10.1) compares the frame view with direct Zarr reads on the P0 cell at N = all.

**Pass 1** (one read of the 2023-2025 stores per site) accumulates per year shard:
- per planned (cloud pixel rule, prep, index, detector): the clear and wet counters (H × W, uint8) and the U_rec counters;
- the C11 masks (`--step ocm`, bounding box of dom plus 1 km);
- for the A-mode curve settings: packed frame validity bits;
- scene statistics: SCL histograms on the frame, the transition zone and keep_full; CLD statistics; the MNAR table (masked rate × h_B decile × s band × SCL class, §2.2);
- per-scene sun and view angles from the sidecar;
- truth-free Granadeiro SD-mask sizes per site under C1 and the NIR mask at W3, which recalibrate the G-base hours (§14).

Neighbourhood operations and Canny run only on the bounding box of dom plus 1 km. Dask uses the synchronous scheduler, or a fixed pool of 2 threads. A pass-1 process is budgeted at ≤ 8 GB and never runs beside Granadeiro.

**Settings pass** (after pass 1): the D2/D3 per-scene thresholds, t_NIR and s_k, all from per-year histograms (§2.4-2.6).

---

## 3. Method catalogue

### 3.1 Fidelity tiers

Every method row carries a fidelity tier. A row prints only after its validation step and the positive control (§10.2) have passed. Otherwise the row reads "implementation not validated" and carries no numbers (v13 §0.6).

| Tier | Meaning | Validation required before printing |
|---|---|---|
| **F0 in-house** | MAREA, no delay, and the in-house lag estimators | Reproduction gate against production code on production inputs (§10.1) |
| **F1 authors' code** | The authors' implementation is executed, or vendored verbatim with its licence and gated bitwise against the installed package | Identity gate: adapter output = the authors' function output on a synthetic cube and on one real tile |
| **F2 port gated against the authors' code or data** | A port whose output is compared with the authors' code, or a published result is reproduced | Code-level equivalence (declared fixes only), or reproduction of the published product within a pre-registered tolerance |
| **F3 re-implementation from the paper** | No usable authors' code | Unit tests of every published step on synthetic data, plus the positive control. Authors' data requested where it exists. |
| **F4 adaptation** | A published idea transferred to a different sensor or geometry, or new code | Labelled "adapted" in every table; positive control |

Every method has a **method card** in `experiments/benchmark/methods/<id>.py` (module docstring plus a `CARD` dict) and in the supplement. The card holds:
- references with DOI or URL;
- family and kind (elevation, lag, or both) and access pattern (series or frames);
- fidelity tier and the evidence for it;
- the dependency signature (§2.9);
- the as-published settings, including the native water-level source;
- the native cell (§5.5);
- the common-conditions behaviour;
- the declared deviations;
- the internal-parameter tuning grid (§5.12);
- the expected estimand offset in the positive control, derived from theory (§10.2);
- the estimated cost and RAM.

**Author review.** Before lock-A, each competitor's card is sent to its corresponding author with a request to review the configuration [V27]. Replies, and the absence of a reply, are logged in the supplement. No reply blocks a run.

The report lists literature lag methods (F1-F3) separately from the in-house adaptations (F4). Family L-P uses literature rows and the regional model only.

### 3.2 Elevation methods

| Id | Method | Family | Tier | Fidelity | Own lag | Code |
|---|---|---|---|---|---|---|
| E0a | **no delay** | per-pixel sigmoid inversion | ref | F0 | τ = 0 | `marea.invert_series` |
| E0b | **MAREA** | per-pixel sigmoid inversion + band lag | – | F0 | m2a band lags, smoothed per pixel | `marea.py`, `tide_estimators.py` |
| E1 | NIDEM/ITEM (Sagar et al. 2017; Bishop-Taylor et al. 2019) | tide-interval composites → contours → TIN | 1 | F2 | none | `waterline_contour.py`, v7 |
| E2 | DEA Intertidal (Bishop-Taylor et al. 2026; dataset 2024) | per-pixel rolling-median "first-wet tide" | 1 | F1 | none | `dea-intertidal==2.5.3` (Apache-2.0) |
| E3 | Granadeiro et al. 2021 | per-pixel 4PL on NIR + rising/ebbing lag + GAM | 1 | F2 | yes (GAM lag map) | `granadeiro2021.py`, `r/granadeiro_fit.R`, v8 |
| E4 | LogiTide2DEM (Chen & Wang 2025) | per-pixel binary logistic on tide height | 1 | F3 | none | new |
| E5a | Flood2Topo, full series (Fassoni-Andrade et al. 2020, 2021) | frequency → exceedance level of the full level series | 1 | F2 | none | new; Octave reference |
| E5b | Tseng et al. 2017 linear scaling | frequency → linear between the low and high level | 1 (variant) | F3 | none | new |
| E5c | Exceedance at the pixel's own instants ("sampled"; declared sensitivity of E5a) | frequency → quantile of the sampled levels | 1 (variant) | F3 | none | new; also one of the two CAE fills (§7.3.3) |
| E6a | Costa et al. 2023 waterline-SDT | per-scene Otsu waterlines → TIN | 1 | F2 | none | new waterline engine |
| E6b | Optical Sentinel-2 waterline (Salameh et al. 2020); classic fixed-threshold configuration as variant E6b-fixed (Mason et al. 1995, 2010; Ryu et al. 2002; Liu et al. 2013) | per-scene waterlines → TIN | 1 (variant) | F3 | none | waterline engine configuration |
| E7 | Coast X-Ray (Fitton et al. 2021) | stage-interval composites, minimum reducer | 1 | F3 | none | new; reuses the NIDEM composite streamer |
| E8 | Bué et al. 2020 | per-pixel NIR logistic, fixed steepness | 1 | F3 | none | new |
| E9 | Pool et al. 2025 ADP | Otsu waterlines, Gaussian vertex smoothing, kriging | 2 | F3 (clean-room) | none | new (CASSIE is GPL-3.0; no code copied) |
| E10 | Khan et al. 2019 pyIntertidalDEM | HSV segmentation, heighted shoreline pixels | 2 | F1 | none | `pyintdem` (Apache-2.0), side venv |
| E11 | Bell et al. 2016, Sentinel-2 adaptation | template correlation of wet/dry with the tide | 2 | F4 | none | new |
| E12 | In-house wet/dry bracket (legacy UDF) | bracket between the highest dry and the lowest wet level | 2 (optional) | F0 | none | port of `pyintertidal/legacy/udf.py` |

Not run as rows:
- `pyintertidal.elevation.fit_step`: superseded by E2. It differs from the current DEA code in five places.
- The soft inundation quantile (SIQ): the soft version of E5c.
- HSR: MAREA's predecessor.

**Family P2 competitors** (frozen at lock-A): E1, E2, E3, E4, E5a, E6a, E7, E8. E5b, E5c and E6b are variants and are reported descriptively.

#### 3.2.1 Elevation method cards (summary; the full cards live in the code and the supplement)

**E0a no delay / E0b MAREA.**
- Algorithm: one unit, two products, as in v13 §3.2.
  - Domain: the extraction rule (more than 30 clear observations, 0.10 < wet fraction < 0.90, sea at wet fraction ≥ 0.90), mouth seeds, geodesic s, `make_bands_fixed(s, 3 km, 500 px)`, all on the instance's W.
  - m2a band lags on W: up to 1200 px per band, seed 20260819 + 31, monotone, band 0 pinned, τ ∈ [−45, 180] min.
  - `smooth_lag_profile` (3 km).
  - `invert_series` on Y, with exact h_B(t − τ_px) per τ group; MAREA's quality rule (bracket 3/3); floor 40 scenes.
  - No delay = the same inversion at τ = 0. Under LT, LM and LG, no delay runs with that source's level in place of h_B. MAREA runs under LB only.
- **Bootstrap:** n_boot = 30 only in Stage L at P0 (W1, W2, W3), which feeds the interval score. n_boot = 0 in every other unit. n_boot enters `config_key`. The bootstrap was measured at 3569 s at the Ems (results/backup_all_scenes_2026-09-29/tide_boundary_comparison_ems.ipynb).
- **As published:** the production record (C2), raw preprocessing, NDWI, D1, W3, LB, run in the harness with G-time instants. The production products `products_*` (label instants) are kept as the "production reference" row.
- Validation: G-marea (bitwise to `marea.reconstruct` on legacy inputs); positive control.
- Cost: about 1 GB RAM at the Ems at W3; 7-25 min per site at N = all without the bootstrap.

**E1 NIDEM/ITEM.**
- Algorithm:
  - one tide level per scene and polygon;
  - observed tidal range over all record scenes;
  - 10 uniform intervals, with the top two merged into 9;
  - per-interval median composite of the index;
  - land flag where the median ≤ threshold;
  - relative extent R = sum of land flags;
  - valid only where every interval is observed;
  - 2-iteration gap fill;
  - contours keyed by level at 0.5..8.5, tagged with the median tide of each interval;
  - linear TIN;
  - filtered product where the mean interval SD ≤ 0.25.
- An empty interval, or fewer than 1 % of pixels valid, stops the run (NOT COMPUTABLE).
- Adapter: v13 §3.3. The composite reader is fed by `obs` under the cell's setting through `_composite_rows`' read callback, on the package-backed frame view.
- Polygons: under LB, one polygon per site. Under LT, LM and LG, one polygon per 3-km band, mosaicked (§1.6).
- **As published:** C4 (ITEM PQ mask), NDWI, threshold 0, filtered product; tide LT from TPXO9 (OTPS reader in pyTMD) at one ITEM-style offshore tide point if the atlas is installed, else EOT20 (declared). Window W9 in Campaign 2 (the published epoch is 28 years); W3 until then (declared). The S2 variant uses C7 (Campaign 2).
- Native cell: C4.floor0.ndwi.D1 (W3; W9 in Campaign 2).
- Detector: D1 at t_k. D2 and D3 become a per-composite Otsu analog. D4 and D5 become the majority vote of W per interval. All are declared.
- Validation:
  - v7 product reproduction (G-nidem);
  - code-level equivalence with `NIDEM_generation.py` on a synthetic composite stack, identical except for the two declared fixes (contours keyed by level; no +0.5 px shift);
  - positive control, with the 9-level quantisation offset in the card.
- Risks: NOT COMPUTABLE at the Ems at W3 N = all (interval 1 holds one overcast scene).

**E2 DEA Intertidal.** The authors' code is run, not a port.
- Algorithm, from `intertidal/elevation.py` 2.5.3:
  - candidate filters: frequency of NDWI > 0.0 in [0.01, 0.99], and Pearson correlation of binary wetness with tide ≥ 0.15;
  - a per-pixel rolling median over 105 tide windows (radius 0.15 of the pixel's tide range; more than 5 observations per window);
  - interpolation to 200 steps, then a trailing mean (radius 20, min_periods 5);
  - elevation = the lowest smoothed tide at which the smoothed NDWI > 0.1;
  - MAD uncertainty;
  - `clean_edge_pixels` once on the mosaic.
- Environment: a side venv `venv-dea` with `dea-intertidal[datacube]==2.5.3`. The plain package cannot import `elevation()`, because `io.py` imports eodatasets3 at module level [V6]. If datacube cannot be installed, the seven pure functions (`ds_to_flat`, `rolling_tide_window`, `pixel_rolling_median`, `pixel_dem`, `pixel_uncertainty`, `flat_to_ds`, `clean_edge_pixels`) are vendored verbatim with the Apache-2.0 header and NOTICE, and gated bitwise against the package wherever it installs.
- Adapter: tiles of 256 × 256 px (or at most 100k candidate px) on S0. The tide is passed as a 3-D DataArray of the unit's level source and set to NaN where Y is NaN (production semantics). W replaces the binary NDWI > 0 of the candidate filters, and the 0.1 threshold is applied on Y.
- **As published:** NDWI and its thresholds; W3 (3-year epoch); C9 if the sidecar exists, else C1 (declared); tide LT per pixel, with the locally ranked ensemble once at least 3 models are installed, else EOT20 (declared).
- Native cell: as P0, with C9 if the sidecar exists.
- Validation: identity gate against `elevation()` on a synthetic cube and one real tile; positive control, with the 0.1-threshold offset in the card. Optional Tier 2: reproduce one published DEA Intertidal 2024 tile.
- Cost: about 7 µs per pixel-scene (numpy probe); at the Ems W3, about 2-3 CPU-hours. 4 processes per unit.

**E3 Granadeiro et al. 2021.**
- Algorithm (v13 §3.4):
  - SD(NDWI) > 0.2 mask;
  - preliminary 4PL inflection (nplr);
  - a pool of pixels within |prelim − h_c| ≤ 0.25 m, sampled to 50,000;
  - rising and ebbing fits per lag;
  - first argmin of |h_rise − h_ebb|;
  - `mgcv::gam(lag ~ s(lon, lat))`;
  - final 4PL at the GAM lag.
- Variants:
  - `ctl`: exact h_B, lag window −45..180 step 5, calibration off (common conditions);
  - `ctl_unpinned`;
  - `eq1_90`: Eq. 1 tide, ±90 min;
  - `calon`: major-axis calibration on;
  - `paper`: gate only.
- Index setting: the 4PL runs on convertToProp of the chosen index (NIR native; NDWI and MNDWI possible). The SD(NDWI) mask stays as published (declared).
- **As published:** C6, NIR, Eq. 1, ±90 min, calibration on. The existing 10 m v8 runs at Villaviciosa, Escalda and Ems are reused as the as-published row (v13 D19). They used label instants (declared, §1.3).
- Native cell: C1.floor0.nir under LB (`G-nir`).
- Driver: v8 main is re-sequenced with a row-block `MaskedBands` loader over the package-backed frame view. v8's `load_bands` would need about 30 GB per band at the Ems for W9.
- Validation:
  - G-granadeiro at Villaviciosa (v8 `paper` products reproduced);
  - reproduce the Bijagós rasters (Zenodo doi:10.5281/zenodo.4662640) on a pre-registered sub-region (one or two published raster tiles, or a box of about 30 × 30 km) from the paper's scenes; L2A instead of ACOLITE is declared [V14];
  - the lean-vs-nplr proofs (≤ 1e-12) and the major-axis vs lmodel2 proofs (≤ 1e-10);
  - positive control.
- Under LT and LM: Tier 2, because the method estimates its own lag map.

**E4 LogiTide2DEM.**
- Algorithm, per pixel:
  - logistic regression logit P(land) = a + b·h on the binary class (W) by IRLS;
  - Z = −a/b;
  - modified success rate MSR = 0.5 SRW + 0.5 SRL; pixels with MSR < 0.6 or Z outside the observed tide range are dropped;
  - at least 27 images per period.
- **Complete separation** is common on clean pixels. The rule is frozen before any run:
  - primary: Z = the midpoint of the separating gap [max h(land), min h(water)];
  - sensitivities: a Firth-penalised fit, and the unpenalised fit with a 25-iteration cap.
- **As published:** MNDWI (at 20 m, declared), per-image Otsu over the AOI valid pixels (the D2 analog without the guards, declared), scene cloud ≤ 70 % (frame SCL share, declared), W3, LB (the paper's coastal gauge has no closer analog; declared).
- Native cell: C1.floor0.mndwi.D2.
- Validation:
  - synthetic Bernoulli pixels with planted Z;
  - agreement with the unpenalised statsmodels or scikit-learn fit to 1e-6 on non-separable pixels;
  - positive control;
  - the Changhua data requested from the authors [V16].

**E5 frequency-to-elevation** (`freq2elev.py`, three rows). With f = wet/valid per pixel over the instance, and the unit's level series:
- **E5a Flood2Topo:** exactly as `cod_Flood2Topo.m`. Pixels with f = 0 are NaN. The full 1-min level series of the window is sorted in descending order; Freq_k = 100·k/n; z = interp1(Freq, levels, 100·f).
- **E5b Tseng:** z = H_hi − f (H_hi − H_lo), with H_lo and H_hi the minimum and maximum of the level series over the window (declared; the paper uses DTU10 amplitudes).
- **E5c sampled:** z = the (1 − f) quantile of the levels at the pixel's own valid instants. Always-wet pixels get the lowest sampled level; always-dry pixels get the highest.
- As published: E5a and E5b under LB (no native optical index; NDWI, D1; declared). Native cell = P0.
- Validation: run `cod_Flood2Topo.m` in Octave on the server on its Lake Poopó example and on a synthetic map; the Python port must match to 1e-9; positive control.

**E6 waterline engine** (`waterline.py`; shared by E6a, E6b and E9). Per scene:
1. index → water mask by the detector;
2. `skimage.measure.find_contours` on the continuous index at the scene's threshold (sub-pixel);
3. drop vertices within 1 px of masked or invalid pixels, and contours with fewer than 20 vertices;
4. height every vertex with the unit's level source at the vertex and the scene's instant;
5. **aggregate heights per 10 m cell across scenes:** the median, with the count. This replaces stacking every vertex (declared). The stacked TIN is a sensitivity at Villaviciosa and Escalda. Coincident vertices from different scenes made the qhull Delaunay arbitrary and needed 10-20 GB at W9.
6. linear TIN on the 10 m grid, inside the cell hull ∩ the instance's intertidal SD mask.

Configurations:
- **E6a Costa:** SD(index) Otsu mask; per-scene Otsu inside the mask; triangulation (QUICKIN equivalent, declared). The truth-fitted bias formula is never applied.
  - As published: NDWI, per-scene Otsu, the scene rule "< 5 % cloud over the estuary" mapped to the domain rule at ≥ 0.95 (declared), W1, and **in-estuary gauge levels** (LG) at the judged sites, LB elsewhere (declared).
  - Native cell: C1.floor0.ndwi.D2 at W1.
  - Validation: compare waterline vertices with the authors' notebook functions (Zenodo doi:10.5281/zenodo.8357351) on one common scene; positive control. Tauranga reproduction: see the away site (§5.14) [V27].
- **E6b optical Sentinel-2 waterline** after Salameh et al. 2020 (ISPRS J. Photogramm. Remote Sens. 163:98-120, doi:10.1016/j.isprsjprs.2020.03.003). The reviewers describe it as unsupervised (k-means) water classification, active-contour waterline refinement and a summer-only Sentinel-2 subset, with a LiDAR-validated error of about 0.19-0.25 m.
  - The configuration is transcribed from the paper into the card and re-checked before lock-A. If its k-means acts on a single index, the D2 analog is used, because the two are equivalent in one dimension. Active contours use `skimage.segmentation` (declared).
  - As published: the paper's detector and seasonal subset (May-September), W3, LG at the judged sites, LB elsewhere (declared).
  - Variant **E6b-fixed:** fixed threshold, no SD mask, TIN, W1 (the classic optical/SAR waterline papers).

**E7 Coast X-Ray.**
- Algorithm:
  - tidal stage % = (TE + MHWS)/(MHWS − MLWS) × 100, exactly as Eq. 1 is printed, with MHWS and MLWS from a harmonic analysis of the unit's 1-min level series over the window (Z0 ± (M2 + S2) amplitudes, declared);
  - 14 stage bins (< −10, −10..0, ..., 100..110, > 110 %);
  - a median composite per bin, only if the bin holds ≥ 3 images;
  - water where the composite > 0.2 (on Y);
  - elevation = the per-pixel minimum, over the composites where the pixel is water, of the mean tide of that bin's images.
- Sensitivity: stage = (TE − MLWS)/(MHWS − MLWS).
- **As published:** C1 plus a coverage rule (keep images covering ≥ 99 % of the AOI, relaxed to keep ≥ 30 images, never below 90 %), NDWI 0.2, LT (FES2022 if installed, else EOT20; the paper used FES2014; declared), W5 in Campaign 2 (W3 until then, declared). QA60 is mapped to SCL {8, 9, 10} (declared).
- Native cell: P0 at W3 (W5 in Campaign 2).
- Validation: unit tests of Eq. 1, the ≥ 3-image rule and the minimum reducer; positive control, with the expected upward offset of the minimum reducer (up to one bin, about half a bin on average) pre-registered in the card.

**E8 Bué et al. 2020.**
- Algorithm:
  - 3 × 3 median prefilter per band;
  - TV = SD(NDWI) over scenes, with the threshold at the valley of the TV histogram (`granadeiro2021.sd_histogram_valleys`, truth-free);
  - per pixel ρ_i = k / (1 + exp(−a (h_i − h_t))), with k = the pixel's maximum reflectance and a = −8 fixed (declared; the paper allows any value in [−10, −2]);
  - h_t by 1-D search on a 1 cm grid over the pixel's tide range (equal to the clipped closed form);
  - observations with ρ ≥ k or ρ ≤ 0 excluded (declared);
  - a saturation index from the fitted curve at the pixel's minimum and maximum tide, kept if ≥ 0.3 (declared reading);
  - IDW gap fill (quadrant search, 20 m radius) as published; "none" as a sensitivity.
- As published: NIR, C6 as the closest scene rule, W1, LB (declared).
- Native cell: C1.floor0.nir at W1.
- Validation: planted sigmoids; grid search equal to the closed form; the paper's insensitivity of h_t to a ∈ [−10, −2] reproduced; positive control.

**E9 Pool ADP** (Tier 2; a clean-room port from the abstract and the CASSIE description, no code copied):
- intertidal zone = Otsu on a 200-bin histogram of SD(AWEI_sh);
- per-scene NDWI Otsu inside the zone;
- pixel-edge vectorisation (`rasterio.features.shapes`), polygons with ≥ 10 vertices;
- 3-tap Gaussian vertex smoothing;
- heights from the unit's level source;
- local ordinary kriging with a per-site exponential variogram, replacing the proprietary EBK (declared), with the TIN as a sensitivity.

**E10 Khan 2019** (Tier 2):
- a thin adapter calling `pyintdem` on per-scene arrays (HSV of a synthetic RGB from B02, B04, B08 and B11; blob cleaning; 3 × 3 Laplacian shoreline pixels);
- heights from the unit's level source; the union keeps the minimum;
- the published blob sizes, plus a declared sensitivity with blob areas scaled to the AOI;
- identity gate: bitwise-equal BW maps with `workflows.khan2019.extract_shoreline` on exported GeoTIFFs;
- a sparse product, scored with the coverage-aware metrics.

**E11 Bell adapted** (Tier 2):
- R(z) = Pearson(Y_t, 1[h_t > z]) over the pixel's valid scenes, exact over all distinct h_t via cumulative sums;
- z* = argmax R, the lowest z on ties; kept if R_max ≥ 0.3;
- validation: brute force equal to the cumulative version to 1e-12; planted steps; positive control.

### 3.3 Lag methods

The convention is τ > 0 when local water is later than h_B. Every imagery method returns an **applied lag field** on S0 (per pixel, or a band value broadcast to the band's pixels). The scorer reads it out identically for all methods (§6.3).

All scanning methods use one common lag window, **−45..+180 min, step 5**, with the fixed parabolic vertex (`_parabolic_argmax`, commit f9d515b), and flag hits on the window edge. Each method's published window is a variant.

| Id | Method | Family | Tier | Fidelity | Output | Key reference / code |
|---|---|---|---|---|---|---|
| L01 | **MAREA m2a** | likelihood | – | F0 | band → smoothed pixel field | `tide_estimators.m2a_rasch` |
| L02 | m2b concordance | order statistic | 1 | F0 | band | `te.m2b_concordance` |
| L03 | m2c flooded-area rank correlation | order statistic | 1 | F0 | band | `te.m2c_waterline` |
| L04 | XC: lagged cross-correlation of the band aggregate (XC-wet: wet fraction; XC-index: band-median Y) | order statistic | 1 | F4 | band | new, `lags/aggregate.py` |
| L05 | m2d flood/ebb discrepancy | self-consistency | 2 | F0 | band | `te.m2d_flood_ebb` |
| L06 | Granadeiro rising/ebbing 4PL lag + GAM | self-consistency | 1 (with E3) | F2 | pixel map | E3 adapter |
| L07 | LOOP: hysteresis-loop collapse of the band flooded fraction (isotonic limbs) | self-consistency | 1 | F4 | band | new |
| L08 | GM20: per-pixel 12.42-h sinusoid phase (Geyman & Maloof 2020), `rel` primary | harmonic | 1 | F3 | pixel (circular) | new; `truth.fold_phase` |
| L09 | HD26-H: M2 phase of the band wet fraction (Hart-Davis et al. 2026), `rel` | harmonic | 1 | F2 (CoastSat NZ check) | band (circular) | new; `truth.hd26_m2_phase` |
| L10 | HD26-R2: delay-and-smear response with an isotonic link | harmonic/response | 1 | F4 | band | new |
| L11 | PPL: phase-plane logistic (the angle of the separating line in (h, dh/dt/ω)) | harmonic | 1 | F4 | pixel (circular) | new |
| L12 | BT26-shift: per-pixel r(Y, h_B(t − τ)) maximised over τ; BT26-rank once ≥ 3 models are installed | order statistic / model selection | 1 | F4 | pixel / band | new |
| L13 | SWOT KaRIn virtual gauges | independent sensor | 3 | F4 | band | new |
| L14 | Global ocean tide models at the gauge (EOT20, GOT4.10c now; FES2014/2022, TPXO9/10, HAMTIDE11, DTU23, GOT5.5/5.6 once installed): at-gauge vs h_B, and model-internal | models | 1 | – | point | `truth.model_at_gauge` |
| L15 | Equal-weight multi-model ensemble | models | 1 | – | point | new |
| L16 | **Regional hydrodynamic models:** CMEMS NWS AMM15 (Escalda, Wadden, Ems), CMEMS IBI (rías). The same SSH as LM (§1.6). | models | **1** | – | point | new, `lags/regional.py` |
| L16b | MeteoGalicia MOHID (Ferrol) | models | 3 | – | point | new |
| L17 | Zero lag | physics | 1 | – | point | trivial |
| L18 | Shallow-water celerity τ(s) = ∫ ds/√(g h̄) on EMODnet depth | physics | 1 | – | point at s_g | new |
| L19 | Analytic convergent-estuary celerity (Savenije et al. 2008; Cai et al. 2012), K = 45 fixed | physics | 2 | F3 | point at s_g | new |
| L20 | REF: gauge-network leave-one-out interpolation in s | reference | 1 | – | point | new; never ranked |
| L21 | REF: DEM-assisted virtual gauge (uses elevation truth) | oracle | 2 | – | band | new; never ranked |
| L22 | REF: elevation-optimal lag τ* (topographic lag scan) | oracle | 2 | – | band | port of `results/topographic_lag_scan_2026-10-07/` |
| L01v | MAREA diagnostics: m2a free (non-monotone), unanchored band scan, v11 local scan, `hysteresis_oos`, tau120; **per relative orbit** (truth-free consistency) | in-house variants | 2 (per-orbit: 1) | F0 | band | existing |

#### 3.3.1 Lag method notes

- **L01 MAREA:** the estimator is unchanged (configs/m2.yaml). It is wrapped by `lags/marea_family.py` and reuses the marea_pair adapter's lag stage, so the elevation and lag rows share one m2a call per instance.
  - The 30-draw bootstrap CI runs only in Stage L at P0. Its nominal 95 % interval feeds the interval score.
  - Gate: bitwise reproduction of `marea.reconstruct`'s lags (G-marea).
  - Control: a planted 0-to-120-min profile is recovered within 10 min (median over bands).
  - The per-relative-orbit estimate (L01v-orbit, Tier 1): m2a on each relative orbit's scenes separately, a truth-free check for orbit-dependent radiometric bias that would alias into the lag.
- **L02-L05:** thin wrappers over the existing functions with the common window. The production grids are kept as "as built" variants.
  - m2c keeps min_scenes 60, with 20 as a curve variant.
  - XC and LOOP need ≥ 80 % band coverage per scene (sensitivities 0.5 and 0.3), ≥ 30 usable scenes, and ≥ 15 scenes per limb for LOOP.
- **L06:** produced by the E3 runs (Stage G). Until then, the as-published v8 lag maps (Villaviciosa, Escalda, Ems; label instants) give an interim descriptive row.
- **L08 GM20:**
  - x = hours since the latest HW mark of h_B (marks_lp);
  - closed-form OLS on cos and sin of 2πx/P, with P = 12.4206012 h;
  - phase = 60·P·atan2(b, a)/(2π);
  - `rel` subtracts h_B's own fold phase at the pixel's instants;
  - QC: n ≥ 20, SD > 0.2, 0.05 < wet share < 0.95, R² ≥ 0.1;
  - for NIR-type indices, P/2 is added;
  - the uncertainty is the IQR over 20 refits on 70 % subsets, run only in Stage L at P0. For the interval score it is rescaled to a nominal 95 % interval under normality (× 1.96/0.674, declared).

  The method details come from v13's earlier reading and are re-checked against the paper before lock-A. Optional fidelity: an S2 run over NW Andros 2017-2018 against the PANGAEA maps (doi:10.1594/PANGAEA.911255).
- **L09 HD26-H:**
  - proxy = the band wet fraction on scenes with ≥ 80 % valid band;
  - detrend, then one 3σ pass;
  - require ≥ 30 points over ≥ 365 days;
  - `pyTMD.solve.constants` with M2, N2, K2, O1, Q1 and GOT corrections;
  - τ = wrap180(g_ζ − g_h0) × 60/28.9841042.

  A 1-year window is marginal, because under fixed solar-time sampling the M2 and O1 alias frequencies differ by 1 cycle per year. Fidelity check: run the port on New Zealand transects of the public CoastSat Pacific Rim dataset (zenodo.org/records/4760144) against TICON-4 (doi:10.17882/109129). The transect-geometry variant and RTide (HD26-R1) are Tier 2.
- **L10 HD26-R2:** y_k(t) = F_k(Σ_j w_j h_B(t − j·dt)), where w is a Gaussian of centre τ and width s ∈ {0, 15, 30, 60} min and F_k is isotonic. (τ, s) is chosen by minimum SSE, and s is reported; s = 0 reduces the method to an isotonic m2c. Needs ≥ 60 scenes.
- **L11 PPL:**
  - x1 = h_B − mean; x2 = (dh_B/dt)/ω_M2;
  - penalised IRLS (ridge 1e-2·n on b1 and b2) of logit P(wet) = b0 + b1·x1 + b2·x2;
  - τ = atan2(−b2, b1)/ω_M2;
  - QC: n ≥ 20, 0.05 < wet < 0.95, b1 > 0, likelihood-ratio p < 0.01;
  - band value = the circular median.
- **L12 BT26:**
  - Shift: per-pixel Pearson r(Y, h_B(t − τ)) over the window, vertex of the argmax, QC r_max ≥ 0.3, band median.
  - Rank: once at least 3 models are installed, the model with the highest median r is selected per band. Its implied lag at the gauge goes through the truth machinery (Bishop-Taylor et al. 2026, IJRS, doi:10.1080/01431161.2026.2666912). The same truth-free ranking selects the LT ensemble member where eo-tides' default ranking does not cover a site (§1.6).
- **L14/L15/L16:** two read-outs per model.
  - (i) The model at the gauge coordinates (nearest wet cell, distance logged) against h_B: τ_a by E(τ), τ_b by M2 phase, and the HW/LW lags.
  - (ii) Model-internal: the model at the gauge against the same model at the boundary point. It is "degenerate" when both fall in the same or adjacent cells.

  The primary global model is FES2022 if installed by lock-A, else EOT20 [V16]. The 1/8° models are lower bounds in the inner estuaries, where the nearest wet cells are 20-33 km away. For L16, a gauge whose nearest wet model cell is more than 1 km away is flagged "unresolved" (expected at the inner Ems and the rías).
- **L18:** h̄(s) = the mean depth below MSL of the permanently wet channel in 1-km s bins, from the EMODnet DTM 2024 (LAT → MSL with a site offset from h_B; floor 1 m). Variants: the site-mean depth, and HW/LW speeds √(g(h̄ ± a_M2)). In Dutch waters EMODnet includes the Vaklodingen; this is declared for a lag-only row.
- **L20-L22:** references and oracles. They are labelled, never ranked, and never used to tune anything.

### 3.4 Excluded methods (not run)

| Method | Reason |
|---|---|
| Tide2Topo (Chen et al. 2023) | Frequency-to-height relation calibrated on LiDAR/UAV transects; the tide-referenced version of the idea is E5 |
| ICESat-2-calibrated frequency or regression methods (Zheng et al. 2023; Xin et al. 2025; Xu et al. 2024) | Need external elevation, which would compete with the truth |
| Deep-learning DEMs trained on elevation labels (Tang et al. 2025; Li et al. 2026) | Same reason |
| SWOT elevation methods (Sun et al. 2025; Yeasmin et al. 2026); SAR (Catalão & Nico 2017) | Not optical time series. The optical sibling of Catalão & Nico, Bué, is included. SWOT appears only as a lag row (L13). |
| Ratio-log SDB (Stumpf 2003) | Subtidal and depth-calibrated |
| Zhang et al. 2026 (IJDE) | Read from the abstract only; a regional inversion model, presumed calibration-based |
| Tide-table secondary-port time differences | Derived from the judges themselves (circular) |
| EasyGSH-DB | German Bight 1996-2015; epoch mismatch |
| Truth-fitted parts of included papers (Costa's LiDAR bias formula; Bué's chart-chosen TV and saturation thresholds) | Tuning on truth; replaced by truth-free rules |
| In-estuary gauges as a lag source | They are the lag truth. As a water level for elevation they appear only in the labelled LG rows (§1.6), never in lag scoring or in a confirmatory family. |

---

## 4. Applicability matrix

Codes:
- ✓ = the factor applies through the method's own published mechanism.
- B = it applies only through the method's binary stages (W).
- Dm = it applies only through the harness domain.
- ✓d = it applies with a declared mapping (a threshold on Y, a per-composite analog, a majority vote, or a per-band polygon).
- n = native level.
- — = not applicable. The planner writes an explicit `not_applicable` row, never a silent gap.

Every scene rule (C2, C6, C7, C8) applies to every image-based method, because it only changes the record.

### 4.1 Elevation methods

| Method | Pixel cloud (C0, C1, C1d, C3, C4, C5, C11) | Scene rules | Prep | Index | D1 | D2/D3 | D4/D5 | Window | N (smallest computable, expected) | Harness domain |
|---|---|---|---|---|---|---|---|---|---|---|
| E0a no delay | ✓ | ✓ (C2 n) | ✓ | ✓ (Y; guards on Y) | Dm | Dm | Dm | ✓ | ≳ 100 (A-mode) | yes |
| E0b MAREA | ✓ | ✓ (C2 n) | ✓ | ✓ (Y) | B + Dm (lag, domain) | B + Dm | B + Dm | ✓ | ≳ 100 (A-mode) | yes |
| E1 NIDEM | ✓ (C4 n) | ✓ (C7 S2 variant) | ✓ | ✓d (filtered: NDWI, MNDWI, NDWI@20 only) | ✓ | ✓d per-composite Otsu | ✓d majority of W | ✓ | ~90 (every interval observed) | no |
| E2 DEA | ✓ | ✓ (C9 n, Tier 2) | ✓ | ✓d (0.1/0.0 on Y) | B (candidate filters) | B | B | ✓ (W3 n) | ~45-90 (> 5 obs per window) | no |
| E3 Granadeiro | ✓ | ✓ (C6 n) | ✓ | ✓ (4PL on any index; NIR n; SD(NDWI) mask kept) | — | — | — | ✓ | ≥ 8 scenes; both limbs | no |
| E4 LogiTide2DEM | ✓ | ✓ | ✓ | ✓ via W (MNDWI n) | ✓ | ✓ (per-image Otsu n) | ✓ | ✓ | ≥ 27 images | no |
| E5a, E5b, E5c frequency | ✓ | ✓ | ✓ | ✓ via W | ✓ | ✓ | ✓ | ✓ | any (resolution 1/n_valid) | no |
| E6a Costa, E6b | ✓ (vertex rejection) | ✓ | ✓ | ✓ (contours on X at the scene threshold) | ✓ (E6b-fixed n) | ✓ (E6a n: D2 analog) | ✓d (contour of W at 0.5) | ✓ (W1 n) | ~5-10 good scenes | no |
| E7 Coast X-Ray | ✓ | ✓ | ✓ | ✓d (0.2 on Y) | ✓ | ✓d per-composite Otsu | ✓d majority of W | ✓ (W5 n) | any (bins with < 3 images skipped) | no |
| E8 Bué | ✓ | ✓ | ✓ | ✓ (continuous; NIR n) | — | — | — | ✓ (W1 n) | ≥ ~10 clear obs per pixel | no |
| E9 Pool (T2) | ✓ | ✓ | ✓ | ✓ (zone from SD(AWEI_sh) as published) | ✓ | ✓ (n) | ✓d | ✓ | few scenes | no |
| E10 Khan (T2) | ✓ | ✓ | ✓ | — (HSV) | — | — | — | ✓ | tens of scenes | no |
| E11 Bell (T2) | ✓ | ✓ | ✓ | ✓ (Y) | binary variant only | binary variant only | binary variant only | ✓ | ≥ ~10 obs | no |

### 4.2 Lag methods

| Method | Cloud | Prep | Index | Detection | Window | N | Harness domain/bands |
|---|---|---|---|---|---|---|---|
| L01 m2a | ✓ | ✓ | B | ✓ | ✓ | ≳ 100 (A) | yes |
| L02 m2b, L03 m2c | ✓ | ✓ | B | ✓ | ✓ | m2b ≥ 10 clear obs per pixel; m2c ≥ 60 scenes | yes |
| L04 XC-wet / XC-index | ✓ | ✓ | B / ✓ | ✓ / — | ✓ | ≥ 30 usable scenes | yes |
| L05 m2d | ✓ | ✓ | ✓ | — | ✓ | both limbs | yes |
| L06 Granadeiro | ✓ | ✓ | ✓ | — | ✓ | ≥ 8 | no (own mask) |
| L07 LOOP | ✓ | ✓ | B | ✓ | ✓ | ≥ 15 per limb | yes |
| L08 GM20 | ✓ | ✓ | ✓ | — (✓ in the `wet` variant) | ✓ | ≥ 20 valid obs per pixel | yes (keep) |
| L09/L10 HD26 wet / index proxy | ✓ | ✓ | B / ✓ | ✓ / — | ✓ (≥ 365 d for L09) | ≥ 30 / ≥ 60 scenes | yes |
| L11 PPL | ✓ | ✓ | B | ✓ | ✓ | ≥ 20 per pixel | yes |
| L12 BT26 shift / rank | ✓ | ✓ | ✓ | — | ✓ | ≥ 20 per pixel | yes |
| L13 SWOT | — | — | — | — | own (2023-07 onwards) | ~50-150 passes | bands only |
| L14-L16 models | instants only | — | — | — | ✓ | — | no |
| L17 zero | — | — | — | — | — | — | no |
| L18, L19 physics | weak (masks) | — | — | — | — | — | s and the sea mask |
| L20 LOO | — | — | — | — | ✓ (truths) | — | no |
| L21, L22 oracles | ✓ | ✓ | B / ✓ | ✓ / — | ✓ | – | yes |

### 4.3 Water-level sources (elevation units only)

| Method | LB | LT | LM | LG |
|---|---|---|---|---|
| E0a no delay | ✓ (base) | ✓ per pixel | ✓ per s bin | ✓ per s (headroom reference) |
| E0b MAREA | ✓ (by design, the only level) | — | — (Tier 2: lag relative to LM) | — |
| E1 NIDEM, E7 Coast X-Ray | ✓ one polygon | ✓d per band polygon (LT n for the as-published row at one tide point) | ✓d per band polygon | ✓d per band polygon |
| E2 DEA | ✓ | ✓ (n) | ✓ | ✓ |
| E3 Granadeiro | ✓ | Tier 2 | Tier 2 | — |
| E4, E5a-c, E8, E11 | ✓ | ✓ | ✓ | ✓ |
| E6a, E6b | ✓ | ✓ per vertex | ✓ per vertex | ✓ per vertex (LG n as published) |

### 4.4 Rules

1. Methods declare `SUPPORTS` (the factor levels they accept) in their card. Any other level produces a `not_applicable` status row in the plan and in every score table.
2. A ✓d mapping is a declared deviation. The as-published row always shows the native mapping beside it.
3. The base cell P0 is applicable to every Tier-1 method. That is a design requirement, and it is met: every Tier-1 method accepts NDWI, C1, D1, W3 and LB. (Khan, Tier 2, accepts C1 and W3 and ignores I and D.)
4. When a method is not applicable to a factor level, the analysis of that factor excludes the method. It is never imputed.

---

## 5. Experiment plan

### 5.1 Structure: an anchored, cost-tiered split-plot design with common random numbers

**Whole plot = instance.** The harness builds every input instance once: site × cell (setting, window) × subset family × mode × N × replicate. An instance fixes:
- the record and the subset (with G-time instants);
- the observation layers X, Y, V and W;
- the level rows of every level source the cell uses;
- the harness domain (extraction rule, sea mask, s, bands);
- the two CAE fills (§7.3.3).

**Sub-plot = method.** Every method applicable to the cell runs on the same instance. Every method comparison is therefore a **paired contrast inside one instance** (common random numbers). This is the statistical form of "vary only the method".

The scheduling is **instance-major**. One worker claims an instance and runs all of its cheap methods in sequence, fed by one fused block reader (§13.4). Each instance's package columns are therefore read once, not once per method.

**Fixed across the whole design:**
- S0, D_truth and its strata;
- the judges and their neighbourhoods;
- the truths and the truth-error model (per window for lag truth (a));
- the bootstrap blocks and weights (§9.1).

Design families considered, and the verdicts:

| Design | Verdict |
|---|---|
| Full factorial of every factor, N and window | Infeasible: about 70,000 maps per method, and Granadeiro takes hours per map. Not needed. |
| Base + one-factor-at-a-time (OFAT) | Used for every Tier-1 factor at N = all. It gives method × factor effects at the base, but not factor × factor interactions. |
| Full factorial of a few levels | G1' = C × I × D (24 cells) for the cheap methods; in Campaign 2, G2 = C × T (12 cells) and an I × T check (9 cells at one site). They give exact variance shares per site and test interactions. |
| Mixed-level orthogonal array L9(3^4) | For Granadeiro only (Campaign 2). It assumes negligible interactions, which the cheap factorials check. |
| Plackett-Burman, Taguchi S/N ratios | Rejected: the factors are few, multi-level and categorical, and method × setting interactions are the target. |
| Latin hypercube | Only in the optional synthetic arm (continuous planted parameters; sliced LHS, Qian 2012). |
| Sequential / adaptive | Only staged execution with pre-registered or truth-blind rules (cost, computability). Never adaptive on truth metrics. |

### 5.2 Base configuration P0

**P0** = C1 (SCL allowlist), floor0, NDWI, D1 (fixed threshold 0), LB, W3 (2023-01-01..2025-12-31), N = all, A-mode (equal to F-mode at N = all), with G-time instants.

P0 is v13's controlled setting, so the M1 code, the gates and the earlier pre-registration stay valid.

At P0:
- every Tier-1 elevation method (E0-E8) runs at the four elevation sites, and at Ferrol for split-half repeatability only;
- every Tier-1 lag method runs at all five sites;
- Granadeiro runs at all five sites in Stage G (§5.11);
- the cheap stochastic stages (MAREA's m2a column draw, GM20's refits) also run with 4 extra seeds, for the seed-variance floor (§9.1);
- every method also runs on the two alternate-scene halves of the record (split-half repeatability, §7.3.4).

### 5.3 "As published" rows

One row per competitor with its native settings, in our harness, on our inputs. Every deviation from the paper is listed in the deviations table (§11.4). Existing as-published products are reused where v13 D19 says so.

| Method | Cloud | Prep | Index / detection | Window | Water level and other native choices | Reuse |
|---|---|---|---|---|---|---|
| MAREA, no delay | C2 | raw | NDWI, D1 | W3 | LB; production settings, G-time instants | `products_*` kept as the production reference (label instants) |
| NIDEM | C4 (S2 variant: C7, Campaign 2) | floor0 | NDWI, 0 on composites | W9 in Campaign 2; W3 until then (declared) | LT: TPXO9 (OTPS) at one tide point if installed, else EOT20; filtered product | – |
| DEA Intertidal | C9 if the sidecar exists, else C1 (declared) | floor0 | NDWI; 0.0 (filters), 0.1 (elevation) | W3 | LT per pixel; locally ranked ensemble once ≥ 3 models are installed | – |
| Granadeiro | C6 | floor0 | NIR; SD(NDWI) mask | W3 | `paper` variant: Eq. 1, ±90 min, calibration on | v8 10 m runs at Villaviciosa, Escalda, Ems (label instants, declared) |
| LogiTide2DEM | C1 + frame cloud ≤ 70 % (declared) | floor0 | MNDWI, per-image Otsu over the AOI (unguarded) | W3 | LB (declared); ≥ 27 images | – |
| Flood2Topo (E5a) | C1 | floor0 | NDWI, D1 (no native optical index; declared) | W3 | LB, full 1-min series | – |
| Tseng (E5b) | C1 | floor0 | MNDWI, 0 | W3 | LB min/max | – |
| Costa (E6a) | domain rule at ≥ 0.95 (declared for "< 5 % cloud over the estuary") | floor0 | NDWI, SD-mask Otsu + per-scene Otsu | W1 | LG (in-estuary gauges) at the judged sites, LB elsewhere | – |
| Optical S2 waterline (E6b) | per Salameh et al. 2020 | floor0 | per the paper (§3.2.1) | W3, summer subset | LG at the judged sites, LB elsewhere | – |
| Coast X-Ray | C1 + AOI coverage rule (≥ 99 %, relaxed to keep ≥ 30, floor 90 %) | floor0 | NDWI, 0.2 | W5 in Campaign 2; W3 until then | LT (FES2022 if installed, else EOT20); Eq. 1 as printed | – |
| Bué | C6 (closest to hand-picked clear scenes) | floor0 | NIR; TV valley threshold | W1 | LB (declared); a = −8, sat ≥ 0.3, IDW fill | – |
| Pool (T2) | C8 | floor0 | AWEI_sh zone, NDWI per-scene Otsu | W1 | LB; local kriging (EBK substitute) | – |
| Khan (T2) | C6 analog (< 10 % frame cloud) | floor0 | HSV | W3 | LB; published blob sizes | – |

### 5.4 Water-level-source rows (P0, elevation units)

At P0, every Tier-1 elevation method except MAREA (LB by design) and Granadeiro (Tier 2) also runs under:
- LT at the four elevation sites;
- LM at the four elevation sites, flagged "degenerate" at Villaviciosa;
- LG at Escalda, Wadden and Ems.

These rows give:
- **family P3** (§9.2): MAREA(LB) − X(LM) and MAREA(LB) − X(LT) for every P2 competitor X. The question is whether MAREA, given only a boundary tide, matches a competitor given the level it would use in practice;
- **headroom** (descriptive): no delay(LG) − MAREA(LB) and no delay(LG) − no delay(LB). This is how much a perfect gauge-based level would still add, including amplitude changes that a lag-only model cannot correct;
- **the "label instants" sensitivity** for MAREA and no delay (§1.3).

LG rows are labelled "uses in-estuary gauges (upper bound)" in every table. They never enter a confirmatory family or lag scoring.

### 5.5 Native cells (one per P2 competitor)

A native cell keeps common conditions (LB, floor0, A-mode, N = all) but uses the competitor's native cloud rule, index, detector and window where 2023-2025 allows it. MAREA, no delay and every cheap method run in each native cell. Instance-major scheduling makes the additional methods almost free once the instance exists. The headline rule requires a MAREA-vs-X claim to hold at P0 **and** at X's native cell (§9.7).

| Competitor | Native cell (Campaign 1) | Campaign 2 | Already in another design |
|---|---|---|---|
| E1 NIDEM | C4.floor0.ndwi.D1 @W3 | @W9 | OFAT cloud C4 |
| E2 DEA | C9.floor0.ndwi.D1 @W3 if the sidecar exists, else P0 | – | – |
| E3 Granadeiro | C1.floor0.nir @W3 (G-nir) at Villaviciosa and Escalda | G-nir at Wadden and Ems | OFAT index NIR (cheap methods) |
| E4 LogiTide2DEM | C1.floor0.mndwi.D2 @W3 | – | new |
| E5a Flood2Topo | P0 | – | P0 |
| E6a Costa | C1.floor0.ndwi.D2 @W1 | – | new |
| E7 Coast X-Ray | P0 @W3 | @W5 | P0 |
| E8 Bué | C1.floor0.nir @W1 | – | new |

### 5.6 One-factor-at-a-time variations (N = all, LB, all applicable methods)

Each cell differs from P0 in one factor only.

**Campaign 1 (W3):**

| Factor | Cells |
|---|---|
| Cloud (8) | C0 (on S0), C1d, C2, C3, C4, C6, C8, C11. The scene-rule cells (C2, C6, C8) are paired with matched-N C1 controls (§2.2). |
| Preprocessing (1) | raw |
| Index (4) | MNDWI, NIR, AWEI_sh, NDWI@20 |
| Detection (3) | D2, D3, D5 |
| Boundary (1) | "EOT20 only" at W3 (declared) [V5] |
| Villaviciosa boundary (2) | EOT20 + Gijón residual; the Cantabrian consensus (Villaviciosa only) |

**Campaign 2:**
- cloud C5, C7 and the class-11-blocked C4;
- index AWEI_nsh, WRI;
- detection D4 (naive baseline);
- windows W2, W5, W9 (W1 exists through the native cells and Y2025);
- "EOT20 only" at W9.

### 5.7 Combination designs (cheap methods; all Tier-1 methods except Granadeiro)

- **G1', C × I × D at W3, N = all, LB (Campaign 1):**
  - C ∈ {C1, C2, C4, C11}: the per-pixel mask, the transition-zone rule, a competitor-native blocklist, and an independent non-Sen2Cor mask. C3 replaces C11 if OmniCloudMask is unavailable.
  - I ∈ {NDWI, MNDWI, NIR}; D ∈ {D1, D3}.
  - 24 cells; 17 are new beyond P0 and the OFAT cells. Methods that do not read D (Granadeiro, Bué, E11) collapse to 12 effective cells (§2.9).
  - C0 and D4 are never in it. Two of its four cloud levels are MAREA's own (C1, C2); one is a competitor's (C4) and one is independent (C11). The report states this.
  - G1' gives the per-site variance shares and the interaction tests of §9.4. It is also the robustness set for the headline rule (§9.7).
- **Campaign 2:**
  - the NDWI@20 slice of G1' (D1, C ∈ {C2, C4, C11}; 3 new cells);
  - **G2,** C ∈ {C1, C2, C4, C11} × T ∈ {W1, W3, W9} at NDWI and D1: 12 cells, 6 new beyond G1' and the window OFAT. It shows whether cloud handling matters more in short windows;
  - **the I × T check,** I ∈ {NDWI, MNDWI, NIR} × T ∈ {W1, W3, W5} at C1 and D1, at Escalda only: 4 new cells. Until it runs, the Granadeiro L9 main effects may be aliased with I × T, and the report says so.
- **Tier 2: G3 corner curves**, (MNDWI, D1), (NDWI, D3) and (MNDWI, D3) at W3, A-mode, with 5 reps.

### 5.8 Archive replicates: calendar-year windows

The calendar-year windows Y2023, Y2024 and Y2025 (Campaign 1) and Y2017-Y2022 (Campaign 2) run at P0 for every Tier-1 cheap method. They are scored on the survey-matched truth. They provide:
- the between-year SD of every P1/P2 contrast, printed next to the block CIs (§9.1);
- the archive-stability condition of the headline rule. In Campaign 1 it is 3 of 3 years, provisional. In Campaign 2 it is at least 7 of 9 years, final (§9.7).

MAREA is expected to be computable at one year (n_clear ≈ 50 > 30, record ≥ 40 scenes). If it is NOT COMPUTABLE in a year, that year counts as a loss for MAREA.

### 5.9 Temporal windows and the epoch-mismatch analysis (mostly Campaign 2)

- **Nested windows** (P0 setting): W1, W2, W5 and W9 join the base W3. They are scored on:
  - (i) the survey-matched truth (primary);
  - (ii) the stable-cell stratum (§1.8, not assessable where fewer than 2 surveys fall in the window ± 1 year);
  - (iii) "latest survey 2019-2025", for continuity only.

  The nested-window effect is confirmed only at Escalda (§2.7).
- **Survey-centred windows** (P0 setting, all Tier-1 cheap methods): the 10 windows of §2.7 at Escalda, Wadden and Ems. Each window's products are scored against every survey year present, giving RMSE(Δt) and bias(Δt) per method.
- **Mismatch model:** RMSE² = σ²_m(N) + σ²_s + D_W(Y). σ²_s and D_W come from the frozen truth-error model (§1.9), with each survey's noise counted once. The method term σ²_m is reported. A W9 product approximates a time-averaged surface, so D_W, not D(Δt from the centre), is the right epoch term.
- **Covariates:** the per-scene processing baseline (pre/post the Collection-1 baseline change, §1.10) and the per-pixel epoch gap.
- **Long-window diagnostics:** the temporal split and the effective epoch (§2.7).
- **Lag scoring per window** happens only where the judges cover the window (§1.7): today W1, W2, W3 and the 2023-2025 calendar years.

### 5.10 Number-of-images curves (A-mode primary)

| Set | Setting | Window, N grid, reps | Methods | Campaign |
|---|---|---|---|---|
| K1' | P0 | W3: N ∈ {20, 45, 90, 180, 330, all}, R = 5d, 5d, 3d, 2d, 2n, 1 | all Tier-1 elevation and lag methods except Granadeiro | Campaign 1 |
| K3 | C2 and C11 (C3 if C11 is unavailable) | the K1' grid | all Tier-1 cheap methods | Campaign 2 |
| K4 | P0 | W9 grid of §2.8 | all Tier-1 cheap methods | Campaign 2 |
| K5 | P0 | cross-window equal-N (§2.8); scored on survey-matched or stable views only | all Tier-1 cheap methods | Campaign 2 |
| K6 | P0 | Granadeiro W3 {45, 90, 180} at Villaviciosa and Escalda | E3 (+ L06) | Campaign 2 |
| K2 | P0, F-mode | W3 grid, 3 reps | methods on the harness domain (E0a, E0b, L01-L04, L07-L12) | 2 |
| K7 | G3 corners | W3 grid, 5 reps | cheap methods | 2 |
| K8 | P0 | MAREA floors 80 and 0 (v13 D4), A and F | E0a, E0b | 2 |
| K9 | P0 | `win` family (contiguous) at W3, 5 reps | cheap methods | 2 |

Each curve point records:
- the computable fraction k/R and the batch kind (disjoint or nested);
- n_clear (median and p10);
- the tidal-range coverage descriptors (`pyintertidal.coverage`): the coverage, entropy and gaps of the sampled tide, and the low and high offsets against the window's 1-min h_B range;
- the month spread of the draw.

### 5.11 Granadeiro plan (expensive; R on the server)

| Step | Content | Sites | Campaign |
|---|---|---|---|
| G-gate | G-granadeiro (v8 `paper` reproduction) at Villaviciosa; G-bijagos on the pre-registered sub-region; positive control | Villaviciosa; Bijagós sub-region | Campaign 1 (Stage 0) |
| G-base | `ctl` at P0 (NDWI, LB). Runs **right after Stage B**, with at most 2 cheap workers beside it | all 5 | Campaign 1 |
| G-nir | `ctl` at C1 + NIR (Granadeiro's native cell) | Villaviciosa, Escalda (Campaign 1); Wadden, Ems (Campaign 2) | Campaign 1 / Campaign 2 |
| G-L9 | L9(3^4): C ∈ {C1, C2, C6} × I ∈ {NDWI, NIR, MNDWI} × T ∈ {W3, W1, W5}, with the 4th column unassigned (an interaction/error check) | Villaviciosa, Escalda | Campaign 2 |
| G-curves | K6 | Villaviciosa, Escalda | Campaign 2 |
| G-tune | The symmetric tuning grid (§5.12) at Escalda | Escalda | Campaign 2 |
| G-T2 | `ctl_unpinned`, `eq1_90`, `calon`; LT and LM rows; L9 at Wadden and Ems; F-mode curves | – | 2 |

The L9 rows (level 1 = base, so run 1 is P0):

| Run | C | I | T |
|---|---|---|---|
| 1 | C1 | NDWI | W3 |
| 2 | C1 | NIR | W1 |
| 3 | C1 | MNDWI | W5 |
| 4 | C2 | NDWI | W1 |
| 5 | C2 | NIR | W5 |
| 6 | C2 | MNDWI | W3 |
| 7 | C6 | NDWI | W5 |
| 8 | C6 | NIR | W3 |
| 9 | C6 | MNDWI | W1 |

The cheap methods also run all 9 L9 cells in Campaign 2, so MAREA − Granadeiro contrasts exist at every one of them.

**Assumption check:** if, for the cheap methods, the C × I or the I × T interaction share exceeds 10 % of the total variance at a site (§9.4), Granadeiro's array main effects at that site are flagged as possibly aliased.

**Robustness for Granadeiro.** Granadeiro is not in G1'. Its robustness set is the L9 cells, at Villaviciosa and Escalda, in Campaign 2. Until then, and at Wadden and Ems throughout, the robustness of any MAREA-vs-Granadeiro claim is "not assessed", so there is no headline against Granadeiro (§9.7).

### 5.12 Symmetric internal-parameter tuning budget (v13 D21, re-instated)

MAREA's internal parameters were developed on these sites, judges and truths: the 3-km bands, the floor of 40 scenes, the bracket 3/3, TAU_BOUNDS, the guards and the transition-zone rule. Each P2 competitor therefore gets a small, pre-registered grid of internal parameters. It is tuned on **Escalda only** (the development site) at P0, using as the objective the method's CAE under the constant fill (§7.3.3), which penalises abstention. The selected configuration runs as a **"tuned" row** at P0 at the other three elevation sites (fairness ladder rung 2b) and enters the LOSO rung (§9.7).

| Method | Grid (frozen in the card at lock-A) | Configs |
|---|---|---|
| E1 NIDEM | confidence SD threshold {0.15, 0.25, 0.35} (filtered) + the unfiltered product | 4 |
| E2 DEA | correlation filter {0.10, 0.15, 0.20} × elevation threshold {0.0, 0.1} | 6 |
| E4 LogiTide2DEM | MSR {0.5, 0.6, 0.7} × separation rule {midpoint, Firth} | 6 |
| E5a Flood2Topo | no internal parameter | 0 |
| E6a Costa | vertex rejection distance {1, 2} px × minimum contour length {10, 20} vertices | 4 |
| E7 Coast X-Ray | water threshold {0.1, 0.2, 0.3} × minimum images per bin {3, 5} | 6 |
| E8 Bué | a ∈ {−4, −8} × saturation threshold {0.2, 0.3, 0.4} | 6 |
| E3 Granadeiro (Campaign 2) | SD threshold {0.15, 0.20} × lag window {±90 min, −45..180 min} | 4 |

The asymmetry that remains is that MAREA is in-sample at every site, while the tuned competitors are in-sample only at Escalda. It is disclosed in every table note.

### 5.13 Which methods run where

| Design | E0a/E0b | E1 | E2 | E3 | E4 | E5a-c | E6a/b | E7 | E8 | Lag Tier 1 | Tier 2 rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 (+ seeds, split-half) | ✓ | ✓ | ✓ | ✓ (G-base) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | if time |
| As published | ✓ | ✓ | ✓ | reuse v8 | ✓ | ✓ | ✓ | ✓ | ✓ | – | if time |
| Level sources LT/LM/LG | E0a only | ✓ | ✓ | – (T2) | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| Native cells | ✓ | ✓ | ✓ | G-nir | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| OFAT, G1' | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| Calendar years | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (scored where judges cover) | – |
| Tuning (Escalda) and tuned rows | – | ✓ | ✓ | Campaign 2 | ✓ | – | ✓ | ✓ | ✓ | – | – |
| K1' | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| Campaign 2 windows, G2, centred, K3-K5 | ✓ | ✓ | ✓ | L9 rows | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (where judges cover) | – |
| K6 | – | – | – | ✓ | – | – | – | – | – | L06 | – |

Ferrol runs every lag design. Its elevation methods run at P0 only, for truth-free repeatability and coverage.

### 5.14 Campaigns, the two-week plan and the cut list

**Campaign 1** (confirmatory core; must fit two calendar weeks on the server, §14.2) needs only the 2023-2025 stores and the 2023-2025 judge series:
- Stage 0: fetch, inputs, G-time, prepare, truth (with the truth-error model), pilots, gates, positive control, lock-A;
- Stage L: the lag benchmark at P0 (W1-W3);
- Stage T: timing (≤ 10 h);
- Stage B (after lock-B): P0, as published, the level-source rows, the native cells, the calendar years 2023-2025, the tuning budget, seeds and split-half;
- Stage G: G-base at 5 sites and G-nir at 2 sites;
- Stage S': the OFAT cells and G1';
- Stage K: K1';
- score and report.

**Campaign 2** (a separate campaign, gated on the arrival of the 2017-2022 batch and on the gauge extension [V5]): prepare 2017-2022; the nested-window OFAT; G2; the survey-centred windows; the calendar years 2017-2022 (final archive-stability condition); NIDEM at W9 and Coast X-Ray at W5 (as published and native); K3-K6; the I × T check; the NDWI@20 slice; the Campaign-2 OFAT levels; G-nir at Wadden and Ems; G-L9; G-tune; and the away site, if approved.

**Away site** [V27, recommended for Campaign 2]: Tauranga Harbour, New Zealand. It is Costa et al.'s (2023) home site, with public LINZ LiDAR, in-harbour gauges and lagged inner arms. It runs P0, the as-published rows and the native cells for every Tier-1 method, and it doubles as Costa's fidelity check (published RMSE about 0.20 m). It costs one openEO download (W3) and about 20 wh.

**Tier 2:** the Tier-2 methods (E9-E12, L05, L13, L16b, L19, L21, L22, the other L01v variants, the HD26 transect variant and RTide), the Tier-2 levels (§2.1), the summer-only subset family (§2.8), K2, K7-K9 and G-T2.

**Pre-ordered cut list** (mechanical). `plan` prints the projected Campaign 1 total against a 10-day cap at the measured throughput (§14). Cuts are applied in this order until the projection fits:
1. K1' reps to 3d, 3d, 2d, 2d, 1n, 1.
2. Matched-N controls to 1 draw for every method.
3. G1' detection column D3 moved to Campaign 2 (G1' becomes C × I at D1: 12 cells).
4. G-nir at Escalda only in Campaign 1; G-bijagos on one raster tile.
5. OFAT AWEI_sh, NDWI@20 and C1d moved to Campaign 2.
6. LT and LM rows at the three Dutch sites only.
7. Tuned rows at two sites.
8. The Villaviciosa boundary variants moved to Campaign 2.

**Never cut:** G-time; the MAREA, NIDEM, DEA and Granadeiro gates; Stage L at W3; P0 at every site; the as-published rows; the LB-vs-LM/LT practical comparison at three or more sites; the native cells; G-base at three or more sites.

**One-week fallback** (if only one week of server time is available): Stage 0, Stage L, the core of Stage B (P0, as published, the LT and LM rows, the native cells), G-base at Villaviciosa, Escalda and Ems, and the OFAT cloud and index cells. Everything else moves to Campaign 2.

The compute clock starts only when the code of milestones V1-V3 has passed its gates (§15). Otherwise the machine sits idle. Methods that pass their gates later join as new units without touching existing ones.

### 5.15 Configuration count (Campaign 1)

| Item | Count |
|---|---|
| Elevation methods Tier 1 (rows) | 2 ours + 11 competitor rows (E1, E2, E3, E4, E5a, E5b, E5c, E6a, E6b, E7, E8), plus the E6b-fixed variant |
| Lag rows Tier 1 | L01-L04 (XC with two proxies), L01v-orbit, L06 (with E3), L07-L12, L14 (2 models now, more once installed), L15, L16, L17, L18, L20 |
| N = all cells at W3 | 1 (P0) + 16 (OFAT) + 17 (G1' new) + 1 (native cell C1.mndwi.D2) + 1 (EOT20 only) + 2 (Villaviciosa boundary, one site) = 38 |
| N = all cells at other windows | 2 native cells at W1 + 3 calendar years (Y2023, Y2024, and Y2025 = W1 at P0) = 5 |
| Level-source rows at P0 | LT, LM (4 sites) and LG (3 sites) × 10 elevation rows |
| As-published units | ~12 (one per method) |
| Tuning | 32 configurations at Escalda + 7 tuned rows × 3 sites |
| Seeds and split-half at P0 | 4 extra seeds (stochastic stages only) + 2 half-records |
| Matched-N instances | C2, C6, C8 × 3 draws (frame methods 1 draw) |
| Curve instances (K1', per site) | 17 new (18 incl. N = all) |
| Granadeiro units | G-base 5 + G-nir 2 (Campaign 2: G-nir 2 + G-L9 16 + K6 16 + G-tune 4) |
| Products, N = all (approx.) | ~6,800 (about 25 products per cell × 5 sites, plus level-source, tuning, seed and matched rows) |
| Curve products (approx.) | ~2,000 |

The frozen design table `plans/<design>.csv` enumerates every unit with its keys, estimated hours, estimated RAM and read volume (§13.3).

---

## 6. Lag benchmark (runs first)

The lag benchmark is the first scored result (Stage L, §15). Its inputs need no elevation truth, and its methods are cheap. It runs only after G-time has passed, because label instants bias every imagery lag low (§1.3).

Gauge agreement is **weak evidence** about MAREA's lag, for two reasons:
- The lag absorbs wet/dry departures ("bathtub calibration": ebb retention, drainage, high flats lagging low flats).
- Truth (a) is close to MAREA's own estimand.

The decisive, non-circular test of the lag is the elevation contrast MAREA − no delay against external heights (family P1, §9.2). The report says so in the lag chapter.

### 6.1 Judges

The 13 headline locations of §1.7:
- **Escalda:** Gat van Borssele (s 14.6 km); the Terneuzen pair (s ~21.9 km).
- **Wadden:** West-Terschelling (5.9); Harlingen (18.3).
- **Ems:** Emshoern (0.4), Eemshaven Waddenzee (1.9), Dukegat (8.7), Knock (20.0), Delfzijl (24.5), Emden (30.3), Pogum (35.8), Nieuwe Statenzijl (37.0).
- **Ferrol:** fer2 (10.1).

Excluded by the method-independent rules: Borssele and Eemshaven haven (coverage), fer4 (< 3 km from boundary member fer1), Uithuizerwad (dry, outside the frame).

Villaviciosa has no judge, so its lags are descriptive. Two sensitivities run on every lag table:
- "without mouth", which drops judges with s_g < 3 km (v13 D10);
- "development-free judges", which drops the judges listed in LOCK.json as used in MAREA's development (at least Harlingen; §1.7).

### 6.2 Truth definitions (per judge and per window)

Built by `truth lag` from the centred gauge series, the frozen level table and the G-time instants only (`truth.py`, v13 §5.5). They are recomputed per window, and per scene record for the scene-rule cells.

| Truth | Definition | Status |
|---|---|---|
| (a) τ_a | argmin E(τ), where E(τ) = the median-centred RMS of h_B(t_i − τ) − g_i at the window's S2 instants (gauge value nearest each instant, within 6 min). τ ∈ [−60, 240] min at 1 min. Interval I_a = {τ : E ≤ E_min + 1 cm}. 200 calendar-month block bootstraps. | co-primary |
| τ_ab | The same, with a free gain b ∈ [0, 4] | co-primary |
| (b) τ_b | M2 phase lag: centred 10-min gauge vs 10-min h_B on common instants; `pyTMD.solve.constants`, 52 constituents, GOT corrections; τ_b = wrap180(φ_g − φ_B) × 60/28.9841042; SE by quarter-year jackknife (months in 1-year windows); I_b = τ_b ± 1.96 SE | co-primary |
| HW, LW | Gauge high and low waters paired with the nearest h_B mark of the same kind within 3 h, both low-passed with the same 61-min mean; median and IQR; the mean of the two, and LW − HW as a distortion index | reported; co-primary only if the user decides [V19] |
| τ_all | The best constant lag over every 10-min sample (separates phase-locked S2 sampling from physics) | diagnostic |
| Crossing lags | `lag_validation.level_crossing_lags` at the deciles of h_B at the S2 instants, per limb | diagnostic |
| (c) frame truths | The GM20 and HD26 frame truths (v13) | diagnostic |
| Per-year truths | (a), τ_ab and (b) per calendar year | location-year analysis (§6.5) |
| Amplitude ratio | The M2 amplitude ratio gauge/boundary from the (b) fit | strata (§7.1); not a lag truth |

Convention: positive τ means local water later than h_B.

20 m light-check values for orientation, in minutes, (a)/(b), at label instants: Harlingen 82 / 91.3; West-Terschelling 45 / 44.3; Gat van Borssele 10 / 9.8; Terneuzen 18 / 18.7; Delfzijl 61 / 68.4.

### 6.3 Read-outs (one rule for every method, in the scorer; amends v13 D8)

- **Local (primary)**, with a **symmetric fallback ladder** applied identically to every method:
  1. the median of the method's applied lag field over the gauge neighbourhood, if at least 30 pixels are defined;
  2. else the median of the method's own field over the gauge's modal band;
  3. else 0.

  Each step is flagged. Band-broadcast fields always meet step 1, while pixel-QC methods (GM20, PPL, BT26) may not near harbour walls; the ladder keeps such methods from being sent straight to 0.
- **Band:** the median over the gauge's modal band k_g.
- **band_interp:** linear interpolation in s at s_g between band medians.
- **Point:** for model, physics and leave-one-out rows, at the gauge coordinates or at s_g.

Phase-type outputs (GM20, HD26, PPL, model (b)) use circular statistics. They are wrapped to ±P_M2/2 about a **truth-free centre**: 67.5 min, the midpoint of the common lag window. They are never wrapped about the truth.

Also reported:
- the window-edge share;
- the share of each fallback step;
- a sensitivity with step-1 read-outs only.

### 6.4 Lag metrics

Per judge, method, truth and window:

| Metric | Definition |
|---|---|
| e, \|e\| | τ_m − τ*, circular for phase-type outputs |
| err_int | Distance from τ_m to the truth interval (0 inside); descriptive |
| **dE** and **dE_all** (co-headline level-domain metrics) | E_g(τ_m) − E_g(τ_a) ≥ 0, in cm: the extra RMS level misfit at the gauge left by the method's lag, at the S2 instants (dE) and over every 10-min sample (dE_all). dE_all does not favour estimands tied to the S2 sampling. |
| **LLE** | sqrt(mean_i [h_B(t_i − τ_m) − h_B(t_i − τ_a)]²) in cm: the vertical consequence of the lag error at the actual sampling instants, free of gauge noise. About 1 cm per minute of error for a 1.6 m M2 amplitude (synthetic check). |
| explained | 1 − (E(τ_m)/E(0))² |
| AGF | (E(0) − E(τ_m)) / (E(0) − E(τ_a)), reported only where E(0) − E(τ_a) ≥ 2 cm; otherwise "lag not material" |
| Interval score | IS_0.05 and truth coverage, only for nominal 95 % intervals: MAREA's 30-draw bootstrap, and GM20's refit IQR rescaled to 95 % (declared). Both are computed in Stage L at P0 only. |

**Aggregates:**
- per site: MAE, median \|e\|, bias, max \|e\|, the hit rates (\|e\| ≤ 10 min; τ_m ∈ I_truth), and the RMS of dE, dE_all and LLE;
- pooled = the mean of the site values. Each site counts once.

**Profile metrics,** where a site has at least 4 judges (Ems):
- the slope of τ_m on τ* across gauges (apparent celerity, min/km);
- Lin's CCC;
- the shape error after removing the mean offset;
- monotonicity violations.

**Stability:**
- the IQR over N-replicates;
- the drift across windows and calendar years;
- **truth-free reproducibility:** band lags estimated on disjoint calendar years, on disjoint halves of the record, and **per relative orbit** (L01v-orbit). ICC(2,1) across years per method (Shrout & Fleiss 1979, doi:10.1037/0033-2909.86.2.420), and the median \|τ_year1 − τ_year2\| per band.

**Geometry diagnostics:** the lag per relative orbit and per season, and a sensitivity that excludes scenes with solar zenith > 70°.

**Efficiency:** wall time, CPU time and peak RSS per run (§8); the shares of NOT COMPUTABLE runs and fallbacks.

### 6.5 Lag inference and reporting

1. **Primary family L-P1 (Stage L).** Per truth ((a), τ_ab, (b)), MAREA is compared with five methods: zero lag, the primary global model (FES2022 if installed at lock-A, else EOT20), the regional model L16, GM20 and HD26-H. If L16 is not available at lock-A, the family has four comparisons, frozen at lock-A.
   - **Paired quantity** per location: d = \|e_MAREA\| − \|e_X\| in minutes (circular for phase-type outputs). Unlike err_int, it has no structural ties at 0.
   - **Statistic:** the mean over the judged sites of the site-mean d. Each site has weight 1, so the eight Ems locations count once.
   - **Test:** an exact sign-flip permutation over the 13 locations (2^13 = 8192 flips), two-sided. With k exactly-zero differences the attainable minimum p is 2/2^(13−k). It is printed for every comparison next to the Holm threshold (0.01 for the first step with 5 comparisons).
   - **Holm** within each truth over the five comparisons.
   - **Site agreement:** a confirmatory result also requires the direction to agree in at least 3 of the 4 judged sites (Escalda, Wadden, Ems, Ferrol).
   - A **ranking** is claimed only if it holds under all three truths (v13 §5.7).
   - Pooled 13-location sign tests are printed as description only and labelled "pseudo-replicated": 8 of the 13 locations share the Ems boundary and channel.
2. **Family L-P2 (Stage G):** MAREA vs Granadeiro (L06), with the same statistic and test, at α = 0.05 per truth. It is a separate, pre-registered family of one comparison per truth, scored when G-base finishes. Until then the v8 interim row (three sites, label instants) is descriptive.
3. **Location-years:** with the per-year truths, each location-year is scored, with a cluster bootstrap by location within site and site-weighted aggregates.
4. **No per-location TOST.** Revision 1's ±10 min TOST had no stated sampling distribution on the method side, so it is dropped. Statements of closeness use the site-mean \|e\| and LLE with their truth-side month-block bootstrap intervals, labelled as truth-uncertainty only.
5. **Families for description,** never pooled into one ranking:
   - likelihood (L01);
   - elevation-free order statistics (L02, L03, L04, L12-shift);
   - self-consistency (L05, L06, L07);
   - harmonic and phase (L08, L09, L10, L11);
   - model selection (L12-rank);
   - models (L14-L16);
   - physics (L17-L19);
   - references and oracles (L20-L22).

   Literature methods (F1-F3) and in-house adaptations (F4) are shown in separate panels.
6. **Lag learning curves:** MAE, dE, dE_all and LLE against N (A-mode), per window.
7. **Positive control first:** every imagery lag method must recover a planted 0-to-120-min profile (median error ≤ 10 min) in the gating world of §10.2 before its real row prints. The misspecification worlds (hysteresis, connectivity censoring, turbid water, wet mud, MNAR masking) are reported for every method to show its bias mechanism, and they do not gate.

### 6.6 Lag runs

| Run | Content |
|---|---|
| L-base | All Tier-1 lag rows at P0, W3, N = all, at all five sites; MAREA n_boot = 30 and the GM20 refits here only |
| L-windows | P0 at W1, W2 and the calendar years 2023-2025 (judges covered); W5, W9, the centred windows and 2017-2022 after the gauge extension |
| L-curves | K1' for the lag rows (W3) |
| L-settings | Lag products of every OFAT, G1' and native cell (the adapters run in every instance); scored in Stage S' as descriptive robustness |
| L-models | L14, L15 and L16 for every installed model, per window (no imagery) |
| L-physics | L18 (EMODnet tiles fetched once per site), L20 |
| L-orbit | L01v-orbit (m2a per relative orbit) at P0 |
| L-gran | L06 from the G-base runs (Stage G) → family L-P2 |

---

## 7. Elevation scoring and metrics

### 7.1 Supports, views and strata

**Scoring support:**
- At the Dutch sites, the 20 m Vaklodingen cells (§1.8). A product's value in a cell is the mean of its mapped 10 m pixels; the cell counts as mapped if at least 2 of its 4 pixels are mapped.
- At Villaviciosa, 10 m pixels (LiDAR aggregated to 10 m).

**Domains:**
- D_truth: range-filtered truth cells or pixels, fixed at lock-A (§1.8).
- D_keep = keep_full(P0, W3) ∩ D_truth.
- S0: the evaluation support (§2.10). Curve products are stored on D_truth and the neighbourhoods only.

**Views:**

| View | Definition | Role |
|---|---|---|
| **pair_q:A\|B** | Cells mapped by both A and B, each passing its own quality rule | **co-primary for every contrast** (P1, P2, P3, verdicts) |
| **pair_raw:A\|B** | Cells mapped by both A and B before their quality rules (where a method has no separate raw product, its product is used) | **co-primary** |
| own_D | Each method's own mapped cells within D_truth | main table |
| common_all | Cells mapped by every validated method computable in the instance | descriptive only. Its membership changes with every added or failed method, so it is never used for inference. |
| CAE (D_truth) | The whole D_truth, with unmapped cells filled (§7.3.3) | abstention control for every headline |
| survey-matched | Default truth: the survey nearest the window centre per cell (§1.8) | default |
| latest survey 2019-2025 | v13's truth | continuity |
| stable | At least 2 surveys within the window ± 1 year and a truth range ≤ 0.20 m [V4]; otherwise "not assessable"; its zone composition is reported | window and epoch analyses |
| pixels (Dutch sites) | 10 m pixels against the replicated 20 m truth | sensitivity |
| direct-measurement cells | Cells whose Vaklodingen source is LiDAR or dense sounding, if the metadata exist [V25] | sensitivity |
| slope < 2 % | Truth slope below 2 % | headline sensitivity (misregistration) |
| curve view | Fixed and N-invariant (§7.4) | learning curves |

**Strata** (all defined once, before any score; none uses a method's output, and MAREA's lags never define a stratum):
- site;
- 3-km band and s;
- **reference lag class** τ_ref(s): interpolated along s between the window's gauge truths, 0 at the mouth; classes < 10, 10-30, 30-60, 60-120 and > 120 min; only where judges allow (Ems, Escalda, Wadden);
- **amplification class:** the M2 amplitude ratio gauge/boundary (§6.2) interpolated along s between judges; classes < 0.95, 0.95-1.05, 1.05-1.15, > 1.15. A lag-only model cannot correct amplitude, so this stratum separates the two effects;
- **hydroperiod zone of the truth elevation** H(t) under h_ref at 10 min (h_ref as in §7.3.1): subtidal > 90 %, lower 50-90 %, mid 20-50 %, upper 5-20 %, supratidal < 5 % (`hydroperiod.zonation`);
- **position relative to the sampled range:** truth z below the lowest h_ref sampled at the window's S2 instants, inside, or above the highest. Phase-locked sun-synchronous sampling never sees part of the range (Sagar et al. 2018, doi:10.3390/rs10030480);
- **truth slope class** (truth DEM smoothed to 30-50 m): < 0.1 %, 0.1-0.5 %, 0.5-2 %, > 2 %;
- distance to permanent water (full-record water frequency ≥ 0.9);
- **substrate brightness:** per site terciles of the low-tide median ρ11 of dry observations (W = 0, in scenes with h_B in its lowest quartile): dark mud, intermediate, bright sand;
- **vegetation:** the low-tide median NDVI of dry observations: < 0.1, 0.1-0.3, > 0.3 (Zostera, Spartina and pioneer marsh, biofilms);
- **turbidity:** per band, the median ρ4 over channel anchors (EMODnet depth ≥ 2 m below LAT) in clear scenes, as per-site terciles. NDWI frequency is not used, because it is circular in turbid water;
- **epoch gap:** < 1, 1-2, 2-4 and > 4 years;
- n_clear class;
- the Vaklodingen source class and, at Villaviciosa, the LiDAR water-mask flag;
- the settings.

### 7.2 Core metric panel (one definition for every method)

With e = z − t on a view, ē its mean, and ẽ = e − median(e):

**Primary quantities (inference):**

| Quantity | Definition | Why |
|---|---|---|
| **ΔMSE_c(A, B)** | Var(e_A) − Var(e_B) on the pair view: the paired loss differential of mean-centred squared errors (Diebold & Mariano 1995, doi:10.1080/07350015.1995.10524599; Hering & Genton 2011) | Write e = m − ε, with method error m and truth error ε (survey noise plus epoch change). Then Var(e_A) − Var(e_B) = Var(m_A) − Var(m_B) − 2 Cov(m_A − m_B, ε), which is unbiased for the method-error variance difference whenever ε is independent of the method errors. A difference of SDs shrinks with a common truth error, SD_A − SD_B = (Var_A − Var_B)/(SD_A + SD_B), so a fixed 2 cm margin on rmse_meanc is harder to reach at noisy sites. The test of H0: ΔMSE_c = 0 needs no truth-error model. |
| δ_m (verdict scale) | sqrt(max(Var(e_A) − σ²_T, 0)) − sqrt(max(Var(e_B) − σ²_T, 0)): the difference in method-term RMSE, with σ²_T from §1.9 | The 2 cm margin is stated on this scale. If Var(e) < σ²_T for either method, the view is flagged "truth-error model inconsistent", and the margin falls back to the rmse_meanc difference. |
| **b*** | b/λ̂, with b the OLS slope of (z − z̄) on (t − t̄) and λ̂ = 1 − σ²_T/Var(t) on the view (Fuller 1987, doi:10.1002/9780470316665) | OLS of z on t is attenuated by error in t, b_obs = λ b. At W3 and W9 the attenuation is comparable to or larger than the 0.05 margin, and it varies by site and window. Deming regression would need each method's own error variance, which is unknown, so the reliability-ratio correction is used. |
| D_b | \|1 − b*_A\| − \|1 − b*_B\| | Slope contrast. **Non-regular rule:** if either b*'s 95 % CI contains 1, the slope contrast can only be EQUIVALENT or INCONCLUSIVE, never BETTER or WORSE, because the percentile bootstrap of \|1 − b\| is unreliable near 1. |

**Display panel:**

| Metric | Definition | Role |
|---|---|---|
| n, coverage | \|finite z in D\|/\|D\| on D_truth and D_keep, also per elevation decile and band; computable fraction across replicates | main table |
| rmse_meanc | SD(e) | main table (the headline accuracy number shown) |
| bias | median(e); also ME = mean(e) | main |
| RMSE_raw | sqrt(mean e²). It includes the h_B datum, which is common to all methods. | main |
| RMSE_c | sqrt(mean(((z − med z) − (t − med t))²)): the v7/v9/v10 continuity metric | main |
| NMAD | 1.4826 · median\|ẽ\| | main |
| LE95 | 95th percentile of \|ẽ\| | main |
| b, \|1 − b\|, intercept | OLS, for continuity with v7-v10; b < 1 means compression | main (descriptive) |
| b*, (1 − b*) | attenuation-corrected; the signed value is shown | main |
| α, α_IQR | σ_z/σ_t and IQR(z)/IQR(t): the direction of relief distortion | main |
| r, Spearman ρ | Pearson; Spearman as a rank-fidelity check | main (ρ in the appendix) |
| CCC_c | Lin's concordance on bias-centred values | main |
| R²_1:1 | 1 − Σ(e − ē)²/Σ(t − t̄)²; never r² | appendix |
| d_r | Willmott's refined index (2012) | appendix |
| KGE_c | 1 − sqrt((r − 1)² + (α − 1)²); the original β is undefined around a zero datum | appendix |
| Hit10, Hit20 | Share of \|ẽ\| ≤ 0.10 m and ≤ 0.20 m (also on raw e) | main (Hit20) |
| Tail50, Tail100 | Share of \|ẽ\| > 0.5 m and > 1.0 m | main (Tail50) |
| Per-band metrics | Every metric by 3-km band; a band-equal-weighted mean; the along-channel bias gradient (cm/km) | lag signature (§9.6) |
| Error variogram | Range and sill of e per method (spatially coherent error) | appendix |
| MDE | The minimum detectable ΔMSE_c (in δ_m units) at 80 % power from the bootstrap SE | context only; no decision depends on it |

`scoring.stats` keeps v13's tests that pin rmse_c, b and r to `v7.vscore` and `elevation_validation.ols_against_truth` at 1e-12.

**Property tests:**
- identical maps give zero errors and b = r = CCC = 1;
- a scaling z = c·t gives b = α = c;
- adding independent noise to t leaves the expectation of ΔMSE_c unchanged and attenuates b exactly by λ.

### 7.3 New metrics

#### 7.3.1 Hydroperiod error HPE (proposed user-facing new metric [V3])

Each elevation is turned into the fraction of time it is submerged under a reference level, and the difference is scored in hours per day.

- H(z) = (1/K) Σ_k 1[h_ref(t_k) > z], over the scoring window at 10 min.
- **h_ref:** the gauge-interpolated level of the cell's band (the LG series) at Escalda, Wadden and Ems; h_B at Villaviciosa. h_B everywhere is a sensitivity. With h_B alone, levels in amplified reaches saturate (H = 0 for both z and t) exactly where the lag correction acts.
- **HPE_c = 24 · mean_i \|H(z_i − b̂) − H(t_i)\|** in h/day, on the pair views.
- The signed mean gives the emersion bias.
- HPE_D is computed on D_truth with the two CAE fills for unmapped cells.
- Agreement on the five hydroperiod zones is summarised by Cohen's κ.
- **Margin, per cell:** the HPE equivalent of 2 cm at each cell, m_HPE = 24 · mean_i p_h(t_i) · 0.02, from the frozen 10-min h_ref density. Revision 1's fixed 0.1 h/day came from one synthetic case. The local weight p_h(z) is U-shaped for a semidiurnal tide, so a fixed margin was not equivalent to 2 cm across zones or sites.
- Revision 1's HPE_raw is dropped: its "independent datum transfer" was undefined.

Properties:
- bounded in [0, 24] h/day;
- locally, \|dH\| = p_h(z)\|dz\|, so errors are weighted by how long the tide stays at that level; errors near MHW and MLW weigh more than mid-tide errors;
- saturates outside the tidal range, so it complements the accuracy metrics and does not replace them;
- comparable across sites with different ranges;
- a pure time shift leaves it unchanged, so it scores elevation only.

This is the quantity that ecology and habitat users derive from intertidal DEMs. Granadeiro et al. (2021) and DEA Intertidal publish exposure layers.

#### 7.3.2 Hypsometric Fidelity Index HFI_D

How well a map reproduces the estuary's area-elevation curve over the **whole** scoring domain:

- b̂ = median(z − t) over the mapped cells of D.
- z′ = z − b̂ − med(t) over the mapped cells; t′ = t − med(t) over all of D.
- **HFI_D = 1 − W1(F_z′, F_t′) / mean_D\|t′\|**, where W1 = mean over p of \|Q_z′(p) − Q_t′(p)\|, on p = (k − 0.5)/1000.
- HFI_pair uses the pair view for both z and t.

Properties:
- HFI = 1 exactly when the bias-centred hypsometric curves coincide; 0 for a flat map at the median; negative for distortion worse than flat.
- Under pure scaling z′ = c·t′, HFI = 1 − \|1 − c\|.
- It is symmetric in compression and dilation, so it is **not** an anti-shrinkage score. It is always shown with b* and α_IQR.
- It compares marginal distributions only. For a compressed map, added independent noise can raise HFI while every pixelwise error grows; a unit test demonstrates this case. **An HFI verdict is therefore issued only jointly with a ΔMSE_c verdict that is not WORSE.** Otherwise HFI is descriptive.
- It catches elevation-selective gaps, such as truncation at the observed tidal range.

Displays: the hypsometric error curve F_z′(h) − F_t′(h). **Margin 0.02.** Gridded truths only, never RTK points.

References: Vallender 1974, doi:10.1137/1118101; Strahler 1952, doi:10.1130/0016-7606(1952)63[1117:HAAOET]2.0.CO;2; Boon & Byrne 1981, doi:10.1016/0025-3227(81)90041-4.

#### 7.3.3 Coverage-adjusted error CAE

RMSE on a method's own cells rewards abstaining from the hard ones. CAE scores every method on the full D_truth and fills its holes with a pre-registered, truth-free reference predictor computed by the harness per instance. There are **two fills**:
- **F_const:** the record-median level of the unit's level series, a constant. It is not a competitor and is deliberately conservative.
- **F_freq:** the E5c estimate, the (1 − wet fraction) quantile of the levels at the cell's valid instants under the cell's V and W. Always-wet cells get the lowest sampled level, always-dry cells the highest, and cells without clear observations the record median.

Definition:
- r_i = (z_i − t_i) − b̂_m where the method mapped the cell, with b̂_m the method's median error on its mapped cells; r_i = (f_i − t_i) − b̂_f elsewhere, with b̂_f the fill's median error on the unmapped cells. Each part is centred by its own median, so a perfect method with partial coverage does not show a two-cluster residual.
- **CAE = sqrt(mean over D of r_i²).** **Margin 2 cm.**

Rules:
- A CAE verdict needs the same direction under both fills.
- E5a, E5b and E5c are excluded from CAE-based verdicts, because F_freq is their sibling.
- A NOT COMPUTABLE product is all-fill. Such points are marked in curves, because there CAE describes the fill, not the method.

Properties:
- CAE equals the centred RMSE at full coverage.
- Abstaining on a cell lowers CAE only if the method's error there exceeds the fill's (Chow's reject rule with the reject cost set to the reference loss; Chow 1970, doi:10.1109/TIT.1970.1054406).

Companions:
- the risk-coverage curve: cells added in the method's own confidence order (`Product.conf`), rmse_meanc against coverage;
- AURC up to the smallest common coverage;
- RMSE at matched coverage q (q = the smaller coverage of the pair).

#### 7.3.4 Truth-free precision

- **Split-half repeatability:** the record is split into alternate scenes, and each half is run through the method unchanged. For MAREA, the v12 variant that reuses the full record's per-pixel lags is also reported. The metric is RMS(z_a − z_b)/√2 on the cells both halves resolve.
- It is reported for every method at P0 and W3, including **Ferrol**, where it is the only elevation metric.
- It measures precision, not accuracy: a stable method can still be wrong.
- The temporal split of long windows (§2.7) is a separate, non-stationarity diagnostic.

### 7.4 Learning-curve summaries

**Curve view (fixed, N-invariant):**
- For a contrast (MAREA vs X), the cells of pair_raw:MAREA|X at N = all. At each N, cells a method does not map are filled as in CAE (both fills).
- For a single method, the whole D_truth with the CAE fills.
- Coverage is shown beside every curve.
- A method is therefore never scored only on its easiest cells at small N.

For each method × site × window × setting, the curve takes the median over replicates of rmse_meanc on the curve view (and of HPE_c, CAE and coverage) at each N, with its IQR.

Fit **E(n)² = s² + v/n** by weighted NLS, with a bootstrap over replicates, once with n = n_clear and once with n = N scenes:
- s is the systematic floor. **s_method = sqrt(max(s² − σ²_T, 0))** removes the method-free truth-error variance (§1.9);
- v/n is the statistical part;
- n_target = v/(0.21 s_method²) is the number of images needed to come within 10 % of the method's own floor. It is reported together with whether it is reachable within N_max(W9);
- sensitivity fit: E = a + b·n^(−c) (Figueroa et al. 2012, doi:10.1186/1472-6947-12-8).

A point is shown only when k/R ≥ 0.5. Disjoint and nested points are drawn with different markers. Curves are never pooled across sites.

**Replicated-curve model (per site):** D ~ log N × T + (1 | subset), with subset = (N, r) shared across settings (lme4 + emmeans in R [V10]). I and D_det terms are added only if the K7 corner curves run, because no Tier-1 curve set varies them.

**Pre-registered mechanistic prediction** (family E): in bands with τ_ref ≥ 30 min, MAREA lowers s_method, not v, and no delay's floor grows with τ_ref.

### 7.5 Epoch-mismatch analysis

For each window product and survey year Y:
- RMSE(Δt) and bias(Δt), with Δt = Y − the window centre, on all cells and on the stable stratum;
- the mismatch model RMSE² = σ²_m + σ²_s + D_W(Y) (§5.9), with σ²_s and D_W(Y) from the frozen truth-error model and each survey's noise counted once. The method term σ²_m is reported;
- the product's effective epoch, both from the minimum of RMSE(Δt) and from the observation weights (§2.7). The slope of RMSE(Δt) gives the morphological sensitivity.

Survey-matched composite view: each truth cell is scored against the product of its survey-centred window.

The Ems "epoch-mismatched" flag of v13 is replaced by these views once 2017-2022 exists.

### 7.6 Pre-registered sensitivities (descriptive, no tuning)

- **Misregistration:** each method's product shifted by ±1 px in x and in y (4 shifts), reporting the change in rmse_meanc. Methods with more spatial smoothing are less sensitive, and the report says so.
- **Slopes < 2 %:** the headline contrasts recomputed on gentle slopes.
- **Range filter:** the local LG range instead of the h_B range (§1.8).
- **Truth-error model:** σ²_T × {0.5, 1.5}.
- **Slope on stable cells:** OLS b on the stable stratum, where epoch error is smallest, as a model-free check of b*.
- **Support:** the pixel view at the Dutch sites; the direct-measurement cells.
- **Truth epoch:** "latest survey 2019-2025" (continuity).

---

## 8. Timing and efficiency protocol

### 8.1 What is recorded for every unit, stage and product

`runner.stage()` today records only wall time and peak RSS. It is extended (new module `timing.py`) so that each named stage records:

| Field | Source |
|---|---|
| wall_s | `time.perf_counter` |
| cpu_self_s | `resource.getrusage(RUSAGE_SELF)` utime + stime delta (all threads) |
| cpu_children_s | `RUSAGE_CHILDREN` delta after R workers exit, plus `psutil` tree sampling every 1 s for long-lived pools (last `cpu_times` per pid) |
| read_mb | Bytes read through `SceneSource` and the package memmaps (counting wrappers), plus `psutil` `io_counters` deltas |
| peak_rss_mb | Process and tree (existing `_Monitor`) |
| progress | A counter the method advances (pixels, scenes or R jobs done). It is used for hang detection (§13.5). |
| context | Load average at start and end, concurrent active claims, BLAS threads, R workers, CPU model and frequency (`lscpu`, once per run) |

Stage names follow `io.*`, `obs.*`, `domain.*`, `fit.*`, `lag.*` and `write.*`, so I/O separates from estimator time.

**Attribution:**
- Each **harness component** is attributed to the methods whose dependency signature consumes it. Components: the observation layers; the extraction-rule domain with mouth seeds, geodesic s and bands; the level rows of each source; the frame view.
  - The domain, s and bands are consumed only by MAREA, no delay and the band-lag methods, and are charged to them.
  - The CAE fills are a scoring cost and are charged to no method.
- **End-to-end cost per method** = its own stages + its share of the consumed components. Within an instance a component's cost is split equally among its consumers. Every method's **stand-alone** cost is also reported: the full cost of every component it consumes, as if run alone.
- In `marea_pair`, no delay = the τ = 0 inversion. MAREA's marginal cost = m2a + smoothing + the per-τ re-inversion. The runner stores `unit_total` and no longer copies the unit's total into every product (today's `runner.py` L1475 behaviour).
- Granadeiro is attributed by R stage (preliminary fits, lag fits, GAM, final fits). `RRunner` also records jobs and per-job wall and worker seconds, which separates Rscript start-up (one process per job, an implementation overhead) from fitting.

**Derived:** core-seconds per 10⁶ pixel-observations (P × N), CPU- and wall-seconds per km² mapped, GB of peak memory, GB read.

### 8.2 Pilot measurements and dedicated timing runs

**Pilot (Stage 0, contended, used for budgeting only):** for every Tier-1 method, one unit at the Ems and one at Wadden, W3, N = all. It records cpu_self/wall, peak RSS, read_mb and wall time. These set `est_hours`, `est_ram_gb` and `est_read_gb` in the plans, and they decide the worker count (§13.4). The pilot also runs the waterline TIN at Wadden W3 under a memory limit, and times pass 1 per scene.

**Dedicated timing runs (Stage T, a quiet machine, target ≤ 10 h overnight).** Production timings are contended by concurrent workers, I/O and the page cache. They are kept for budgeting, flagged `contended`, and never used in comparisons. Comparisons use dedicated runs; `timing` refuses to start while other claims are active.

**Standard workloads** (fixed before lock-A):
- **Series methods:** a fixed Escalda pixel tile set of 2.5 × 10⁴, 10⁵ and 4 × 10⁵ domain pixels × N ∈ {45, 180, 465} (W3).
- **Frame methods** (NIDEM, waterline, Coast X-Ray, Khan, Pool): an Escalda sub-frame of 1024 × 1024 px × the same N.
- **End-to-end:** the P0 cell at Escalda at N = all, for the cheap methods. The Ems figure comes from its flagged production run (contended).
- **Granadeiro:** no dedicated single-worker run. The per-fit single-thread cost comes from `RRunner`'s worker seconds and job counts in the 16-worker G-base runs. The production wall time comes from the same runs (contended unless G-base ran alone).

**Protocol:**
- one warm-up run discarded, then 2 repeats; the median and the min-max are reported;
- two configurations:
  - single-thread (BLAS 1, 1 process): algorithmic cost, on workloads up to 10⁵ px only;
  - production (the method's declared threads and processes): practical wall time, on every workload;
- the page cache cannot be dropped without root, so the first-touch read of a run is the "cold" proxy and the repeats are "warm".

**Scaling:** fit log cost = a + α·log P + β·log N per method. This gives exponents with CIs and a predicted cost for a standard workload of 10⁶ px × 500 scenes.

### 8.3 Efficiency reporting

1. **Cost table** per site and method at P0:
   - method, shared-split and stand-alone end-to-end core-hours;
   - wall-hours at the declared parallelism;
   - peak GB and GB read;
   - scaling exponents;
   - implementation context (language, parallel scheme, authors' code or not).
2. **Pareto plot** per site: x = method core-hours (log scale), y = rmse_meanc on the curve view (and HPE_c). Each method is drawn as its learning curve over N (cost model × the E(N) fit of §7.4). The non-dominated front is marked.
3. **Time-to-target:** TTT_m = cost_m(N*_m), where N*_m is the smallest N whose fitted method-term error is ≤ E*. Three pre-registered targets: E* = 0.20 m and 0.30 m method-term RMSE (absolute), and 1.1 × the best s_method among the methods at that site (relative). Otherwise "not reachable within N_max".
4. **Skill per decade of compute against no delay:** η_m = (SS_m − SS_nodelay) / log10(C_m/C_nodelay), with SS = 1 − MSE_c/MSE_c,clim. Reported only when C_m/C_nodelay ≥ 2, and as a secondary scalar.
5. Ratios across sites are summarised with geometric means.

All costs are labelled "as implemented here, on one declared server". Implementation differences (numpy vs R subprocesses) are implementation effects, not algorithmic ones (Hoefler & Belli 2015, doi:10.1145/2807591.2807644; Dehghani et al. 2022, arXiv:2110.12894).

---

## 9. Statistical analysis and rules for conclusions

### 9.1 Units of analysis and uncertainty

- **Unit of analysis:** one map, giving one metric vector per view and stratum.
  - Pixels (cells) enter only through the block bootstrap.
  - Archives enter through the calendar-year windows and the replicates.
  - Sites enter through per-site analyses and a descriptive cross-site summary.
  - Metric values are never pooled across sites or truth types.
- **Paired contrasts:** D(A, B) = metric_A − metric_B within one instance.
- **Within-map uncertainty: a spatial block bootstrap with SHARED weights.**
  - **Block size per site** = the largest practical range of the truth-free variograms of the map differences z_A − z_B, over every P1/P2/P3 contrast at P0 on D_truth. It is rounded up to the lattice {250, 500, 1000, 2000, 4000, 8000, 16000} m, so there is no 4 km cap. `score --blocks` computes it from the products alone; the module does not import any truth reader.
    - This targets the correlation length of the differences between methods, which for MAREA − no delay is the along-channel scale of the lag correction.
    - One size per site lets every contrast share weights. Using the largest range is conservative.
  - The effective number of blocks is printed for every contrast. A confirmatory test at a site needs **at least 25 blocks** on the pair view; otherwise that site is descriptive for that contrast.
  - The bootstrap weights are seeded from (site, truth, stratum, block size) only, never from the cell, method, window or N. Any function of the whole factorial (a main effect, an interaction, a contrast of contrasts, a ranking, regret) therefore gets its bootstrap distribution by evaluating it on each draw.
  - **B = 20,000** for every quantity used in a confirmatory family or a headline verdict, and B = 1,000 for descriptive quantities. With B = 20,000 the smallest bootstrap p is 2/20,001 ≈ 1.0 × 10⁻⁴, below Holm's first threshold for the largest frozen family (0.05/128 ≈ 3.9 × 10⁻⁴). A unit test asserts 2/(B + 1) < α/m for every confirmatory family. Revision 1's B = 5,000 gave a minimum of 4.0 × 10⁻⁴, so its 128-test P2 family could never reject.
  - Products are scored into per-block sufficient statistics (§13.6). Additive quantities are therefore bootstrapped exactly as weight-matrix products. These are ΔMSE_c, rmse_meanc, b, b*, r, CCC, HPE_c and CAE. Quantile metrics (median bias, NMAD, LE95), HFI and variograms get CIs only at P0 and in the families.
- **Band-level test (P1 accuracy).** ΔMSE_c is also aggregated per 3-km band (from per-band sufficient statistics) and bootstrapped with a moving-block bootstrap along s, with blocks of 2 bands (B = 20,000). It needs at least 6 bands with truth. A confirmatory P1 accuracy result requires both the 2-D and the band-level test to reject. Both test the same parameter, so the extra condition adds no multiplicity.
- **Between-archive uncertainty:**
  - The **calendar-year windows** give the between-year SD of every P0 contrast, printed next to every block CI (§5.8).
  - Disjoint-batch curve points get a two-level bootstrap (replicates, then blocks), labelled "within one record".
  - Nested points are labelled "conditional on the record".
- **Algorithm-seed variance:** at P0 and W3, 5 seeds for the cheap stochastic steps (MAREA's m2a column draw, the GM20 refits; DEA has none). In Campaign 2, 2 seeds for Granadeiro at Villaviciosa. The seed SD is the floor below which factor effects are reported as nil (§9.4). The production seeds stay primary.
- **Cross-check at P0:** the Hering-Genton test on the spatial loss differentials (doi:10.1198/TECH.2011.10136).

### 9.2 Families and multiplicity (pre-registered)

Each family's statistics are **view-specific**: one statistic per (contrast, quantity, view). A contrast is confirmed only if **both** of its view statistics are rejected (intersection-union; Berger 1982, doi:10.1080/00401706.1982.10487759). FWER is controlled over all the view-specific statistics of the family by a **Romano-Wolf step-down max-T** on studentised statistics from the shared-weight bootstrap (Romano & Wolf 2005, doi:10.1111/j.1468-0262.2005.00615.x). This exploits the strong positive correlation between views, quantities and contrasts. Holm on the marginal bootstrap p-values is printed as a conservative cross-check.

| Family | Content | View-specific statistics m | Control |
|---|---|---|---|
| **P1** (lag effect, confirmatory) | MAREA − no delay at P0 (LB), N = all, at the confirmatory elevation sites; ΔMSE_c and D_b; views pair_q and pair_raw | 4 sites × 2 quantities × 2 views = 16 | Romano-Wolf, FWER 0.05; intersection-union over views; the band-level test for accuracy |
| **P2** (state of the art, common conditions, confirmatory) | MAREA − each P2 competitor (E1, E2, E3, E4, E5a, E6a, E7, E8) at P0 (LB); the same quantities and views | 8 × 4 × 2 × 2 = 128 | the same |
| **P3** (practical, confirmatory, secondary to P2) | MAREA(LB) − X(LM) and MAREA(LB) − X(LT) at P0 for each P2 competitor except Granadeiro; ΔMSE_c only; both views; LM at Villaviciosa excluded (degenerate) | 7 × (3 + 4) × 2 = 98 | the same |
| **L-P1, L-P2** (lags, confirmatory) | §6.5 | 5 (L-P1) and 1 (L-P2) per truth | Holm within each truth; all three truths; site agreement |
| **S** (robustness, secondary) | Sign consistency of every P1/P2 contrast across G1', the native cells, the OFAT cells and the calendar years: win/tie/loss rates, verdicts | – | Descriptive; Benjamini-Yekutieli FDR if p-values are printed |
| **E** (exploratory) | Strata, curves, new metrics, interactions, the mechanistic prediction of §7.4, the LG headroom | – | CIs only, labelled hypothesis-generating. Confirmation on the sealed RTK split or a new site is the authors' decision. |

**Rules:**
- **Family membership** is frozen at lock-A as the nominal sets above. Two steps may then shrink it, both logged and both done before any truth is read:
  - the truth-free `score --blocks` step removes members that cannot be tested (fewer than 25 blocks; fewer than 6 bands for P1 accuracy). Villaviciosa is expected to be removed in this way;
  - a competitor that fails validation is removed before any score.

  The families are then frozen before scoring. With B = 20,000, every family size up to 128 remains attainable (§9.8).
- If the user prefers a fixed sequence (P1 first, then P2 at full α only if P1 rejects at every confirmatory site), that choice is made before lock-A [V19].
- Within a contrast, the test statistic is ΔMSE_c (or D_b). The verdict scale for the margin is δ_m (§7.2).

### 9.3 Verdicts against practical margins

Per cell and contrast, in this order:
1. **EQUIVALENT** if the 90 % CI of the verdict quantity lies inside ± the margin (TOST; Schuirmann 1987, doi:10.1007/BF01068419).
2. Otherwise, **BETTER** or **WORSE** if the 95 % CI of the test quantity excludes 0 and the point estimate on the verdict scale is at least the margin in magnitude. The sub-label **"clear"** is added when the 95 % CI also excludes the margin on that side.
3. Otherwise **INCONCLUSIVE**.

Special rules:
- **Slope:** the non-regular rule of §7.2. If either b*'s CI contains 1, only EQUIVALENT or INCONCLUSIVE are possible.
- **HFI:** issued only jointly with a ΔMSE_c verdict that is not WORSE (§7.3.2).
- **CAE:** the same direction under both fills (§7.3.3).

| Quantity | Margin |
|---|---|
| ΔMSE_c (verdict scale δ_m) | 2 cm of method-term RMSE |
| rmse_meanc, RMSE_c (descriptive verdicts), CAE | 2 cm |
| \|1 − b*\| | 0.05 |
| HFI_D | 0.02 (jointly with ΔMSE_c) |
| HPE_c | per cell, the HPE equivalent of 2 cm (§7.3.1) |
| lag | 10 min; 2 cm in dE, dE_all or LLE |

### 9.4 Factor effects

1. **Variance shares, per site only.**
   - They come from the cheap full factorials: G1' in Campaign 1, and G2 and the I × T check in Campaign 2. These are deterministic at N = all, given the archive and the seeds.
   - A functional-ANOVA decomposition gives the shares of C, I, D and their interactions (exact Sobol main and total indices for a full factorial).
   - Responses per method: CAE under F_const on D_truth (N- and setting-invariant), log rmse_meanc on the cell's own view, b*, HFI_D, HPE_c and coverage. For D(MAREA, X), the cell's pair view.
   - Shares are never pooled across sites, because that would pool LiDAR and Vaklodingen truths.
   - The shares are design-weighted (every level counts equally), and the report says so. They are conditional on one archive and one m2a column draw. Every effect is compared with the P0 seed SD (§9.1), and effects below it are reported as nil.
   - The per-cell MNAR slope (§2.2) is reported beside the cloud effects.
   - Significance comes from the shared-weight bootstrap. Lenth's pseudo-SE (doi:10.1080/00401706.1989.10488595) is the fallback for unreplicated contrasts.
2. **Replicated curves, per site:** the model of §7.4.
3. **Cross-site synthesis:** each Vaklodingen site's D̂ and its bootstrap SE feed a random-effects meta-analysis (REML with Hartung-Knapp-Sidik-Jonkman CIs; metafor as the check), with the 95 % prediction interval for a new estuary. Villaviciosa (LiDAR) is shown beside it and added only in a sensitivity. With k = 3 the meta-analysis is wide and descriptive, which is the honest answer.
4. **Displays:**
   - interaction plots per site;
   - heatmaps of D(MAREA − X) over C × I rows and D (and T, in Campaign 2) columns, coloured by verdict;
   - Taylor diagrams per site (doi:10.1029/2000JD900719).

### 9.5 Robustness summaries

These are given per site and per factor level, never as one global percentage, because the design itself sets the weights. The view is the pair view of each cell.
- **Win/tie/loss/inconclusive rates** of MAREA against each competitor, with cluster-bootstrap CIs over cells. **NOT COMPUTABLE counts as a loss** for the method that is not computable. The computable share is printed beside every rate, and a CAE-filled version is also shown. A method cannot escape the hard cells by failing in them.
- **Rank-flip table:** the levels where the sign of D reverses and both CIs exclude 0.
- **Kendall's W** of method rankings across cells, with ties inside the margin counted as ties.
- **Regret** = metric_MAREA − the minimum over a **fixed comparator set** (the frozen P2 list plus MAREA). In a cell where a comparator is NOT COMPUTABLE, its CAE (F_const) value is used. Regret is summarised by its median and 90th percentile ("MAREA is within 2 cm of the best method in x % of cells"). Fixing the set removes the winner's-curse dependence on how many competitors happen to be computable.
- Friedman/Nemenyi critical-difference diagrams, as description only (the cells are not independent).
- Optional per site: the Bayesian hierarchical comparison of Corani et al. 2017 (doi:10.1007/s10994-017-5641-9) with the same ROPE.

### 9.6 Where MAREA is good

Every stratum is method-independent (§7.1). In particular, MAREA's own lags are never used to define a stratum.

1. **Conditional verdict tables** (§9.3) per stratum: reference-lag class, amplification class, band and s, hydroperiod zone, sampled-range position, slope class, distance to water, substrate, vegetation, turbidity, epoch gap, n_clear class, and the settings.
2. **Decomposition:** MAREA − X = (MAREA − no delay) + (no delay − X), per cell and stratum. The first term is the lag correction. The second is the inversion and algorithm difference. A win against a competitor without lag handling is attributed to the term that carries it, and conclusions are scoped accordingly.
3. **Loss-differential models:** the cell-level d_i = ẽ²_X,i − ẽ²_MAREA,i (mean-centred, matching ΔMSE_c) on the pair view, modelled per method pair with a GAM: s(s_km) + s(H(t)) + s(slope) + s(log n_clear) + s(epoch gap) + amplification class + substrate class + turbidity class + site (mgcv; block-thinned tables; CIs from the spatial block bootstrap). Partial effects are read together with the raw tables, because the strata co-vary along an estuary.
4. **Honest partition:** model-based recursive partitioning on block means. Trees are grown on alternate contiguous segments of s (half of each site's bands) and evaluated on the other segments, and also grown on the other sites and evaluated on the held-out site (Roberts et al. 2017, doi:10.1111/ecog.02881). It yields readable rules such as "τ_ref ≥ 30 min and slope < 0.5 %: MAREA better by y cm". partykit needs approval [V10]; a numpy CART on block means is the fallback.
5. **Advantage maps:** block-smoothed d, stippled where the CI excludes 0.
6. **Lag signature:** per method, the band median bias and RMSE against s, and the along-channel bias gradient in cm/km. No delay should show the gradient in lagged estuaries; MAREA should remove it.
7. **Curves by lag class:** the floors s_method per τ_ref class (the §7.4 prediction).
8. **Headroom map:** no delay(LG) − MAREA(LB) per band, separated by amplification class.

### 9.7 Headline rule, printing rule and fairness ladder

**Printing rule:** a row prints only if its fidelity validation and positive control passed (§10). Otherwise it reads "implementation not validated" and carries no numbers.

**Headline rule.** A claim "MAREA is better (or worse) than X at site S" is a headline only if **all** of these hold. Each condition can only make a claim more conservative, so adding conditions needs no multiplicity correction.
1. **Confirmatory:** rejected in P1 or P2 at S in both pair views, after Romano-Wolf. For P1 accuracy, the band-level test also rejects.
2. **Magnitude:** |δ_m| ≥ 2 cm (accuracy), or the slope rule (§7.2).
3. **Abstention control:** the same direction under CAE with both fills, or in RMSE at matched coverage.
4. **Native cell** (P2 only): the same direction at X's native cell, with the 95 % CI excluding 0.
5. **Archive stability:** the same sign in all three calendar years 2023-2025 (Campaign 1; the headline is then labelled **provisional**), and in at least 7 of the 9 calendar years 2017-2025 (Campaign 2; final). The between-year SD is printed.
6. **Settings robustness:** the same sign in at least 80 % of the G1' cells (for Granadeiro, the L9 cells), with NOT COMPUTABLE counted as a loss. C0 and D4 are never in the set. If the set is empty, the robustness is **"not assessed" and there is no headline.** This applies to Granadeiro in Campaign 1, and at Wadden and Ems throughout.
7. **Scope:** a P2 headline is stated as "given only a boundary tide". A headline "in practice" also needs the P3 verdict BETTER (or EQUIVALENT, for "as good as") against X(LM) and X(LT) where they are available.

A **cross-site** headline needs the same direction confirmed at every confirmatory site where X is validated and computable, no WORSE verdict at any site, and a meta-analytic estimate in the same direction.

Otherwise the claim is labelled "setting-dependent", and the factor levels that flip it are named. A claim that misses only the Campaign-2 part of condition 5 is labelled "provisional".

**Fairness ladder,** one row per rung and competitor, never mixed:

| Rung | Definition |
|---|---|
| 1. As published | §5.3, including native water levels |
| 2. Common conditions | P0 (LB) |
| 2b. Tuned | The symmetric internal-parameter budget: tuned on Escalda, applied at the other sites (§5.12) |
| 3. Common conditions per cell | The OFAT cells, G1' and the native cells |
| 3b. Practical water level | LT and LM rows at P0 (§5.4) |
| 4. LOSO-tuned | Each method's best settings combination **and** internal parameters, chosen by truth on the other sites and scored on the held-out site. MAREA is included, with its lag model fixed. Pure post-processing of the scores. |
| 5. Oracle-best per site | Truth-tuned, labelled as an upper bound |
| Reference | LG rows: "with in-estuary gauges", an upper bound |

Every table note states MAREA's home advantage: P0 is its development setting, and the sites and judges are its development data. §9.9 discloses the results already seen.

### 9.8 Attainability and power check (pre-lock, truth-free)

`lock` refuses to run unless `pilot attain` has written `results/validation_v2/attain.json` with, per family:
- the frozen m (view-specific statistics), B, the smallest attainable bootstrap p 2/(B + 1), and Holm's first threshold α/m;
- for L-P1 and L-P2: the smallest attainable exact p per comparison given the judged locations (2/2^13 with no zero differences), against the Holm thresholds;
- the expected number of blocks per site at block sizes of 1, 4 and 16 km, and the number of 3-km bands with truth. These come from the D_truth geometry alone, because the final block size comes from the products. They also show which sites are expected to be confirmatory;
- the denominators and the rules for untestable members (§9.2).

The minimum detectable effect is reported after scoring, as context only (§7.2).

### 9.9 Pre-registration lock, amendments and forking-paths disclosure

- **lock-A** (before Stage L) freezes:
  - the analysis plan: metrics, views, families, margins, B and the headline rule;
  - the lag methods;
  - **every elevation method's CONFIG**: the cards, the as-published settings, the native cells and the tuning grids;
  - the P2 and P3 lists;
  - the level-source definitions;
  - the truth-error model;
  - D_truth, S0 and dom;
  - the G-time verdicts, the CLD p and the per-scene offset rule;
  - the list of development judges;
  - the forking-paths disclosure (below).
- **lock-B** (before Stage B) adds only implementation-level items that cannot depend on lag scores: the code hashes of the elevation adapters, their gate and positive-control verdicts, and the environment identities. lock-B cannot change any CONFIG, margin, view, family or truth definition.
- `lock` refuses to run while the tree is dirty in `experiments/benchmark`, `pyintertidal` or `tests`.
- It writes `results/validation_v2/LOCK.json` with:
  - git HEAD;
  - every FROZEN section sha, including `design`, `metrics_v2`, `analysis_v2`, `truth_error`, `level`, the method cards, the tuning grids and the P2/P3 lists;
  - the gate, G-time and positive-control verdicts;
  - `attain.json`.
- `score` refuses real-data 10 m products before the relevant lock exists. Exempt: the gates, the controls, the timing runs, the 20 m tests and `score --blocks` (truth-free).
- Every later change to a locked section is logged under `amendments` (date, reason, affected units). It reruns every affected method, never only some (v13 §2.11).

**Forking-paths disclosure.** LOCK.json lists the truth-based results the authors had seen before the lock. For each, it records what was seen and which design choices it could have influenced: P0, the accuracy metric, the views, the margins, TAU_BOUNDS, C2 and the judges. The list starts from the repository:
- `results/v1_vaklodingen_validation.*`, `v4_lag_external.*`, `v5_rtk_onsite.json`, `v6_elevation_external.*`, `v6_lidar_external.*`;
- `c1_dev_comparison.json`, `c2_boundary_variants.json`, `c5_validation_matrix.*`;
- the v7-v10 server runs (`results/server_10m/`);
- `method_comparison_provisional_20m_2026-10-01/`, `scene_rule_study_2026-09-30/`, `lag_estimator_agreement_2026-10-06/`, `topographic_lag_scan_2026-10-07/`, `v13_benchmark_20m/`;
- the gauge table `p7_tabla_mareografos/`;
- the Harlingen comparison that widened TAU_BOUNDS.

The authors complete the list before lock-A.

---

## 10. Validation before scoring

### 10.1 Reproduction and identity gates

| Gate | Checks | Pass criterion |
|---|---|---|
| **G-time** | Site sensing times (§1.3): method A against method B on the sample; within-group SD; physical window; instant ≥ label − 0.5 min | §1.3 criteria |
| G-offset | The per-scene offset rule (§1.10): no 'unclear' scene; dark-water NDWI on anchors continuous across regimes (step ≤ 0.02) | exact rule; step reported |
| G-zarr | Zarr B03/B08/SCL = legacy netCDF, 2023-2025 common dates | bitwise, or the pre-registered response of §1.11 |
| G-record | The v2 record at W3 equals the v13 record (dates, h0 at label instants) | exact |
| G-keep | The extraction-rule domain at P0 equals production keep on legacy inputs | exact |
| G-marea | marea_pair on legacy inputs (NetCDFSource, raw prep, label instants) reproduces `marea.reconstruct` lags and both elevation maps | bitwise (v13 §4.1) |
| G-frameview | Frame methods on the package-backed frame view = the same methods on direct Zarr reads, P0 cell at N = all, on S0 | bitwise on S0 (declared fill outside dom) |
| G-level | LT: eo-tides = direct pyTMD at test points (≤ 1 mm). LM: the subset read = the CMEMS toolbox output (bitwise). LG: the interpolation reproduces each judge exactly at its s_g. | as stated |
| G-ocm | The C11 adapter = OmniCloudMask's own prediction function on one scene | bitwise |
| G-nidem | v7 products reproduced; KeepCheck | v13 §4.1 |
| G-nidem-code | Composite → contour → TIN equals the `NIDEM_generation.py` logic on a synthetic stack, except the two declared fixes | ≤ 1e-9 |
| G-granadeiro | v8 `paper` products reproduced at **Villaviciosa** (discrete outputs exact, continuous ≤ 1e-9, the GAM at v8's points; R identities recorded) | v13 §4.1 |
| G-bijagos | Granadeiro fidelity: the Bijagós DEM and lag rasters reproduced on the pre-registered **sub-region** (one or two published raster tiles, or a box of about 30 × 30 km) from the paper's scenes; the L2A deviation is declared | median \|Δz\| ≤ 0.15 m and median \|Δlag\| ≤ 10 min on the published intertidal mask (tolerances confirmed at lock-A) |
| G-dea | Adapter = `dea-intertidal` 2.5.3 `elevation()` on a synthetic cube and one real tile (or vendored = package) | bitwise |
| G-flood2topo | Python E5a = `cod_Flood2Topo.m` in Octave (Lake Poopó example + synthetic) | ≤ 1e-9 |
| G-costa | Waterline vertices = the authors' notebook functions on one common scene | ≤ 1e-6 px |
| G-khan (T2) | BW maps = `workflows.khan2019.extract_shoreline` | bitwise |
| G-hd26 | HD26-H port on CoastSat NZ transects vs TICON-4 M2 phases | median \|Δg\| ≤ 10° (pre-registered) |
| G-score | The scorer reproduces published v7/v9/v10 numbers from their CSVs | 1e-12 |
| G-lagtruth | A planted 75-min lag is recovered as 75 ± 1 min; the nearest-cell truth lookup equals `v1.sample_truth` | exact |

### 10.2 Positive control (every method, the same scorer)

The synthetic site `pc` is written **as a Zarr store in the same format**, with the same sidecars and planted G-time instants, so the control exercises the same reader.

**Gating world** (deliberately not MAREA's model):
- a planted DEM at 1 m (channels, creeks, flats of several slopes), aggregated to 10 m. The estimand is the 10 m mean elevation;
- a planted lag profile (0 → 120 min at 12 km, plus a zero scenario), with an amplitude ratio of 1;
- reflectance by **linear mixing of library endmembers** per 1 m sub-pixel: clear water with low suspended sediment, wet sand and dry sand. A sub-pixel is wet iff the local level plus a 2 cm wave term exceeds its elevation;
- 10 m bands at 10 m, and 20 m bands generated at 20 m and then replicated;
- an SCL window, haze (2 % of scenes, +400 DN), clouds and cloud shadows with a CLD field.

**Expected estimand offsets.** Each method's card gives its expected offset in this world, derived from theory and written before the run. Examples:
- for a threshold estimand, the level at which the mixed-pixel index crosses the threshold (DEA's NDWI > 0.1; Coast X-Ray's 0.2);
- the minimum reducer's upward offset of up to one stage bin (Coast X-Ray);
- the 9-level quantisation (NIDEM);
- the saturation reading (Bué).

**Tolerances** (v13 D22, amended), fixed in FROZEN before any real score:
- elevation: coverage ≥ 0.5; median \|Δz − offset\| ≤ 0.10 m; \|bias − offset\| ≤ 0.05 m; b ∈ [0.8, 1.2];
- lag: median error over bands ≤ 10 min.

Tolerances are never loosened after real data are seen.

**Failure policy:**
- A failed control blocks a row only when a **documented implementation fault** is found. The fault is fixed, and the method reruns everywhere.
- A failure with no fault found (a model mismatch) leaves the row printable, with the flag "control mismatch", and the failure is reported.
- A competitor is removed from P2 only for a validation failure with a documented fault, before any score.

**Misspecified worlds** (Tier 1, reported for every method, never gating): turbid water (an NIR raise with suspended sediment), wet mud (dark; MNDWI confusion), ebb retention and hysteresis, connectivity censoring, vegetation patches, MNAR masking (dark-area mislabels at low tide), amplitude amplification (ratio 1.2 inland) and sparse tidal sampling. A sliced LHS over planted lag, slope, tidal range, cloudiness and turbidity is optional (Tier 2).

### 10.3 Unit tests (pytest, synthetic, laptop)

Existing: obs, levels, package, runner (88 tests).

New:
- inputs: the Zarr fixture, fingerprints, grid assertions, `lattice20`, the per-scene offset;
- sensing_time: an SPA round trip (time → angles → time within 1 s); SGP4 on a fixed TLE; the G-time criteria on synthetic label offsets;
- settings: the truth table of every cloud rule; every index formula (2 × 2 resolution rule included) against `water.index_block` where one exists; Otsu, the D2 guards, edge-Otsu on synthetic edges; the standardisation; the OCM adapter with a mocked model;
- levels: LT, LM and LG builders, and the per-band polygon mapping;
- domain, judges, truth: a planted lag; multi-epoch Vaklodingen; the truth-error model on synthetic surveys recovers σ²_s and D_W; the stable-stratum rule; the PNOA constant-patch mask;
- product;
- one file per method (the planted-truth tests of §3.2.1);
- readout: the fallback ladder; the truth-free wrapping centre;
- scoring:
  - the property tests of §7.2, including the invariance of ΔMSE_c to independent truth noise and the exact λ attenuation of b;
  - the HFI noise-inflation case;
  - CAE with two fills and per-part centring;
  - sufficient-statistic scores equal to direct scores to 1e-12;
- inference:
  - shared weights;
  - Romano-Wolf step-down against brute force on small cases;
  - the intersection-union over views;
  - the attainability assertion 2/(B + 1) < α/m for every frozen family;
  - exact sign-flip p-values;
  - TOST verdict order;
- timing;
- runner:
  - the plan and effective keys;
  - **adding a dummy registry level leaves every existing `inputs_key` unchanged**;
  - the RAM ledger;
  - the wall-time cap and hang detection;
  - instance-major claims.

The defects found on 2026-10-09 are fixed first (§12.3).

---

## 11. Report structure

`report` builds `results/validation_v2/report/` from the score CSVs only:
- `report.html`, a self-contained page with every table and figure;
- `tables/*.tex` (v10 style) and `figures/*.pdf|png`;
- `numbers.json`, holding every number quoted in the text with its source CSV row.

The report draws conclusions only under the rules of §9.7. Every table carries:
- the fidelity tier of each row;
- its label: as published, common conditions, native cell, tuned, practical water level, or upper bound;
- the home-advantage note;
- "provisional" marks where only Campaign 1 evidence exists;
- NOT COMPUTABLE and not-applicable cells shown as such, never blank.

### 11.1 Chapters

1. **Data and design:** sites, windows, records and **sensing times (G-time)**, gauge coverage, truths, epochs and the **truth-error model**; the settings taxonomy and the MNAR diagnostic; the applicability matrix; the design table summary; the scope of the conclusions (§0.8).
2. **Lags (first):** lag truths per judge and window; method results by family; families L-P1 and L-P2; profiles; per-orbit and per-year reproducibility; the efficiency of the lag methods; the explicit caveat that gauge agreement is weak evidence.
3. **Elevation at P0:** the main table per site; families P1 and P2; as published, tuned and native cells; the new metrics.
4. **Water level in practice:** the LT, LM and LG rows; family P3; headroom.
5. **Robustness to settings:** OFAT, G1', verdict heatmaps, factor shares, regret and rank flips; archive stability across calendar years.
6. **Windows and epochs** (Campaign 2): nested windows (confirmed at Escalda), survey-centred windows, the mismatch curves, the stable stratum, the effective epoch.
7. **How many images:** learning curves, floors, n_target, span vs count, computability.
8. **Where MAREA is good:** conditional verdicts, the decomposition into the lag and algorithm parts, GAM partial effects, partitions, advantage maps, the lag signature.
9. **Efficiency:** the cost table, scaling, Pareto fronts, time-to-target.
10. **Conclusions:** whether and where MAREA is good. Headlines only under §9.7; everything else is labelled setting-dependent, provisional or exploratory.
11. **Supplement:** method cards and the author-review log, deviations, fidelity evidence, gates and control results, FROZEN shas and amendments, the forking-paths disclosure, `attain.json`, the compute log.

### 11.2 Tables

| Id | Content |
|---|---|
| T1 | Sites: frames, records per window, truths and epochs, judges, boundary members, level sources, coverage |
| T2 | Method catalogue: family, tier, fidelity tier, validation status, own lag, native water level, licence |
| T3 | Applicability matrix (§4) |
| T4 | Settings levels with references |
| T5 | Lag truths per judge and window ((a), τ_ab, (b), HW, LW, intervals, amplitude ratio) |
| T6 | Lag results per judge × method at P0: τ, e, err_int, dE, dE_all, LLE, edge and fallback flags |
| T7 | Lag summary per site and family; L-P1 and L-P2 per truth with the attainable p |
| T8 | Elevation main table per site × method at P0 (the §7.2 main columns; views own_D and the pair views against MAREA) |
| T9 | Families P1, P2 and P3: ΔMSE_c, δ_m, b*, CIs, Romano-Wolf and Holm p, verdicts, number of blocks |
| T10 | As published, tuned and native-cell rows per competitor beside the common conditions |
| T11 | Factor variance shares per site and method, and for D(MAREA, X), with the seed floor |
| T12 | Win/tie/loss/inconclusive rates (NOT COMPUTABLE as a loss) and regret per competitor, per factor level and site |
| T13 | Learning-curve summaries: s_method, v, n_target, the smallest computable N, k/R |
| T14 | Windows and epochs: nested and survey-centred windows on the survey-matched and stable views; D_W(Y) |
| T15 | Conditional verdicts per stratum ("where") and the lag/algorithm decomposition |
| T16 | Costs: method, shared-split and stand-alone end-to-end core-hours, peak GB, GB read, scaling exponents, TTT |
| T17 | G-time: label vs instant per site, relative orbit and platform |
| T18 | Water-level rows (LT, LM, LG) and headroom |
| T19 | Archive stability: the sign per calendar year and the between-year SD of each headline contrast |
| T20 | Truth-error model: σ²_s, D_W(Y), λ̂ per site, window and view |
| S1-S8 | Deviations; fidelity evidence; gates; positive control; FROZEN shas and amendments; forking-paths disclosure; author-review log; compute log (contended timings) |

### 11.3 Figures

| Id | Content |
|---|---|
| F1 | Site maps with bands, judges and truth coverage |
| F2 | Lag profiles along s per method family, with the judges' I_a and τ_b |
| F3 | E(τ) sweeps per judge with method lines |
| F4 | Bland-Altman plot of lags (all locations, coloured by site) |
| F5 | Elevation and error maps per method at one site per type (ría, estuary, basin) |
| F6 | Taylor diagrams per site |
| F7 | Hypsometric curves and hypsometric error curves (HFI) |
| F8 | HPE distributions and zone confusion (κ) |
| F9 | Verdict heatmaps of D(MAREA − X): C × I rows, D (and T) columns |
| F10 | Interaction plots per factor |
| F11 | Learning curves (E against N on the curve view, coverage, k/R) per site |
| F12 | Span vs count: equal-N panels across windows (Campaign 2) |
| F13 | RMSE(Δt) epoch-mismatch curves |
| F14 | Advantage maps; GAM partial effects of the loss differential |
| F15 | Lag signature: band bias against s and the along-channel gradient |
| F16 | Accuracy-coverage and risk-coverage curves |
| F17 | Pareto fronts (accuracy against core-hours) |
| F18 | G-time: labels and instants per relative orbit |
| F19 | MNAR: masked rate against h_B decile per SCL class and cloud level |
| F20 | Headroom: no delay(LG) − MAREA(LB) per band, by amplification class |
| F21 | Decomposition of MAREA − X into the lag and algorithm parts, per site |

### 11.4 Deviations table (supplement S1)

One row per method × deviation:
- what the paper does;
- what we do;
- why;
- in which rows;
- the sensitivity row that measures it (or "not measurable").

The table is built from the method cards. The report refuses to build if a card lists a deviation without a row reference.

### 11.5 Notebooks (`notebooks/validation/`)

The notebooks are thin, parameterised viewers. They import harness functions and never re-implement a number the harness computes.
- Each has one papermill-style "Parameters" cell.
- Outputs are stripped in git.
- Heavy cells are guarded by `RUN_HEAVY`.
- On the server, executed copies go to `results/validation_v2/notebooks/`.

| Notebook | Content |
|---|---|
| `00_inputs.ipynb` | `open_cube` per site and year; scene counts per window; RGB and false-colour quicklooks; SCL and CLD statistics; encoding and offset checks; **G-time** labels vs instants; overpass, boundary, level-source and gauge coverage; grid identity and `lattice20` |
| `01_settings.ipynb` | For one site and window, every cloud level (including C11), index (including the 20 m construction) and detector side by side: masks, index maps, Otsu histograms and guards, edge buffers, water frequency, harness domain, transition zone; the MNAR curves; the truth-free diagnostics (§2.4) |
| `02_run_unit.ipynb` | Parameters site, setting, window, method, N, rep. Runs one unit through the runner API on a small window (laptop) or reads a finished unit (server); shows products, statuses and timings |
| `03_inspect_method.ipynb` | Parameter `method`. Shows the method card and internals: MAREA band likelihood and lag profile; NIDEM intervals and contours; DEA windows; LogiTide separation cases; waterline vertices and the cell aggregation; Granadeiro mask, pool, lag sample and GAM; one-pixel views (reusing `marea_one_pixel.ipynb` logic) |
| `04_lags.ipynb` | Lag truths, read-outs and the fallback ladder, profiles, L-P1/L-P2, per-orbit lags, curves |
| `05_elevation.ipynb` | Main tables, families, the truth-error model, verdict heatmaps, factor shares, level sources, windows, curves |
| `06_where.ipynb` | Strata tables, the decomposition, GAM (R through rpy2, or the CSV handed to an R script), partitions, advantage maps |
| `07_efficiency.ipynb` | Timing tables, scaling fits, Pareto, TTT |

---

## 12. Module and notebook layout on the branch

### 12.1 Layout

```
experiments/
  zarr_cubes.py                  EXISTS  store format, open_cube, compare_with_netcdf (reader used by inputs.py)
  download_cube_zarr.py          EXISTS  downloads, verify sidecars, overpass_times.json
  validation_v2.py               NEW     CLI entry point (python -X utf8 -m experiments.validation_v2 <command>)
  v13_benchmark.py               KEEP    continuity alias; shares the command functions
  benchmark/
    config.py                    ADAPT   UNION_PERIOD; SiteConfig.store + legacy block + level sources; FROZEN sections per factor entry; B 20000/1000
    inputs.py                    NEW     SceneSource (Zarr, NetCDF) with the 16-date block cache, lattice20, per-scene offset, fingerprints, inputs report
    sensing_time.py              NEW     G-time: sun-angle inversion (SPA), SGP4 propagation, criteria, sensing_times.json
    reader.py                    NEW     fused block reader over package tiles (instance-major), package-backed frame view
    settings.py                  NEW     registries cloud/prep/index/detect/level/window/curves; Setting; setting ids; per-entry source hashes
    designs.py                   NEW     design tables (base, published, levels, native, ofat, g1, years, tuning, curves-k1, granadeiro-*, and the Campaign-2 designs), plan CSV writer, 10-day cap and cut order
    obs.py                       ADAPT   NDWI core kept bitwise; dict-of-bands signatures; index (20 m rule)/mask/detector registries; observe(setting) -> X, Y, V, W
    levels.py                    ADAPT   provider over UNION_PERIOD; LB/LT/LM/LG level rows per instance; per-band polygon series
    package.py                   ADAPT   year shards keyed by dom_version; per-band pixel-major tiles; uint8 per-setting counters; OCM step; settings pass; MNAR table
    domain.py                    ADAPT   counts from the instance's setting; keep ∩ dom; cache key (setting, window, subset)
    judges.py                    REUSE   coverage rule per window; development-judge list
    truth.py                     ADAPT   multi-epoch Vaklodingen; survey-matched default; cell support; PNOA water mask; truth-error model; range filter from 1-min h_B; strata arrays; per-window lag truths; HW/LW; amplitude ratio
    product.py                   ADAPT   Unit.setting/window/level; S0 support; not_applicable status; per-product timing
    runner.py                    ADAPT   method registry; multi-plan priority queue; instance-major claims; effective keys; RAM ledger; memory limits; wall cap; hang detection
    timing.py                    NEW     stage stopwatch (wall, CPU, child CPU, I/O, progress), pilot and timing commands
    readout.py                   NEW     read-outs with the fallback ladder; truth-free wrapping
    blocks.py                    NEW     truth-free block sizes from map-difference variograms; block and band indices
    scoring.py                   NEW     views, strata, per-block sufficient statistics, metrics (ΔMSE_c, b*, HPE, HFI, CAE), curves
    inference.py                 NEW     shared-weight bootstrap, Romano-Wolf, IUT, Holm/BY, sign-flip tests, TOST verdicts, ANOVA shares, Lenth, meta-analysis, rank stability, regret, attainability
    control.py                   NEW     synthetic Zarr store 'pc' (linear-mixing world), scenarios, tolerances, expected offsets
    report.py                    NEW     tables, figures, numbers.json, report.html
    methods/
      __init__.py                REGISTRY and CARD schema
      elev_marea_pair.py         E0a, E0b (+ L01 lag product)
      elev_nidem.py              E1 (per-band polygons under LT/LM/LG)
      elev_dea.py                E2 (calls venv-dea through a worker subprocess or vendored functions)
      elev_granadeiro.py         E3 (+ L06)
      elev_logitide.py           E4
      elev_freq.py               E5a, E5b, E5c (+ the F_freq fill used by CAE)
      waterline.py               engine (cell aggregation) + E6a, E6b, E6b-fixed (+ E9 Tier 2)
      elev_coastxray.py          E7
      elev_bue.py                E8
      elev_khan.py               E10 (Tier 2)
      elev_bell.py               E11 (Tier 2)
      elev_bracket.py            E12 (Tier 2, in-house)
      published.py               as-published loaders (production products, v8 runs)
    lags/
      marea_family.py            L01, L01v (incl. per orbit), L02, L03, L05
      aggregate.py               L04 XC, L07 LOOP, L10 HD26-R2 (shared band-aggregate builder)
      gm20.py                    L08
      hd26.py                    L09 (+ transect variant and RTide, Tier 2)
      ppl.py                     L11
      bt26.py                    L12 (shift and rank; the rank also serves LT)
      models.py                  L14, L15
      regional.py                L16 (AMM15, IBI), L16b (Tier 3)
      physics.py                 L17, L18, L19
      references.py              L20, L21, L22
      swot.py                    L13 (Tier 3)
    fetch/                       NEW     ioc.py, judges.py (wraps estuary_truth/scripts, records time-stamp conventions), stac_meta.py, sensing.py (angles, TLEs), cmems.py, tides.py, emodnet.py, pnoa_meta.py, vak_meta.py, swot.py
tests/
  test_benchmark_{inputs,sensing_time,settings,obs,levels,package,domain,judges,truth,product,readout,blocks,scoring,inference,control,timing,runner}.py
  test_methods_<id>.py           one per method
notebooks/validation/00-07_*.ipynb
results/validation_v2/           output root (§13.1)
docs/validation_v2_spec.md       this document
```

### 12.2 Reuse map

| Existing | Reused for |
|---|---|
| `experiments/zarr_cubes.py` (`open_cube`, `open_store`, `grid_info`, `read_overpass`, `compare_with_netcdf`, `store_radiometry`, `store_baselines`, `radiometry_check`, `CLEAR_SCL`, `NATIVE_RES_M`) | inputs.py, G-zarr, G-offset, notebooks |
| `docs/v13_benchmark_spec.md` | Unchanged sections cited as "v13 §x": package passes (§2.5), domain (§3.1), marea_pair (§3.2), NIDEM (§3.3), Granadeiro (§3.4), GM20/HD26 (§3.5-3.6), gates (§4.1), truths (§5.5), lag inference (§5.7), keys and lock (§2.11) |
| `experiments/benchmark/{config, obs, levels, package, domain, judges, truth, product, runner}.py` (M1) | Adapted as in §12.1 |
| `pyintertidal/marea.py`, `tide_estimators.py` | E0a, E0b, L01-L03, L05, L01v (unchanged estimators) |
| `pyintertidal/overpass.py` | Labels only; the instants come from sensing_time.py |
| `pyintertidal/waterline_contour.py`, `experiments/v7_waterline_contour.py` | E1 (reader callback), the TIN helper for E6 |
| `pyintertidal/granadeiro2021.py`, `pyintertidal/r/granadeiro_fit.R`, `experiments/v8_granadeiro2021.py` (`RRunner`, `StageCache`, `LazyRows`, proofs) | E3, L06; `sd_histogram_valleys` and `to_reflectance` for E8 |
| `pyintertidal/water.py`, `stability.py` | Index cross-checks, Otsu, the TZ scene rule (C2) |
| `pyintertidal/hydroperiod.py`, `hypsometry.py`, `coverage.py`, `simulator.py` | HPE and zones, HFI, sampling descriptors, control scenarios |
| `pyintertidal/lag_validation.py` | Crossing lags (diagnostic truths) |
| `experiments/c2_boundary_variants.py` | The Villaviciosa boundary sensitivities (Gijón, Cantabrian consensus) |
| `results/lag_estimator_agreement_2026-10-06/lag_estimator_agreement.py` | Template and regression reference for the lag phase (its m2a column reproduced result.json bit for bit) |
| `experiments/v11_local_lag.py`, `v12_scene_rule_choice.py` | L01v local scan; split-half repeatability |
| `results/topographic_lag_scan_2026-10-07/` | L22 |
| `experiments/v1_vaklodingen_validate.py`, `v10_validation_report.py`, `pyintertidal/rtk.py`, `elevation_validation.py`, `validation_grid.py`, `v9_method_comparison.py` | Truths, report formatting, G-score references, as-published loaders |

`v8.wait_for_ram` (free-RAM polling) is **not** reused for claims, because it races between workers. The RAM ledger replaces it (§13.5).

### 12.3 Defects to fix first (milestone V0)

These were found on 2026-10-09 by running the existing tests: 83 passed, 1 skipped and 16 errors on Windows. With a scratch fix for the first defect, 14 of 17 package tests pass.

1. **`package.pass2` keeps a memmap open through the loop variable `mm`** after flushing (L757). On Windows, `os.replace` (L771) then fails with WinError 32, which errors every package test. Fix: `del mm` after the flush loop, also in the except branch.
2. **`scene_stats.csv` is read back without round-trip float precision** (L1095, L1676), and the g21 subset is decided on these floats. Fix: `read_csv(..., float_precision='round_trip')`.
3. **`test_dom_is_keep_prod_urec_and_the_judges_neighbourhoods`:** the fixture's judge neighbourhood lies inside U_rec, so the test cannot pass. Fix the fixture geometry. The test is then rewritten for the §2.10 definition of dom.
4. **`package.load` memoises its verification per manifest mtime** (`_VERIFIED`, L1808), so a file tampered with after a first load in the same process is never re-checked. Fix: re-stat the keyed files (size + mtime_ns) on every call.
5. **`config.py` L452 sets `B_primary = 5000`**, which makes the 128-statistic P2 family unattainable (§9.1). Set 20,000 and add the attainability test.

Then add the missing test files for domain, judges, truth and product, and run the whole suite on the server too.

---

## 13. Orchestration on the cluster

### 13.1 Output tree

```
results/validation_v2/
  common/<site>/            package shards per year and dom_version (pixel-major tiles, uint8 counters, frame bits, OCM bits, scene_stats, MNAR table, thresholds), level rows per source, marks, subsets_<window>.npz, manifest.json
  common/inputs_report.json
  sensing/<site>/           G-time inputs and verdicts (sensing_times.json is also copied to cubes/<site>/)
  truth/<site>/             truth_elev.npz (per epoch, cell support, strata arrays), truth_error.json, lag_truth_<window>.json
  plans/<design>.csv        frozen design tables (sha in FROZEN['design'])
  runs/<site>/<window>/<setting_id>/<method>[__variant]/<family>-<mode>/N<NNN>_r<RR>/   product.npz, run.json, unit.log
  claims/                   claim files, heartbeats, ram_ledger.json, STOP files
  gates/, control/, timing/ results of gates, positive control, dedicated timing runs
  blocks/<site>.json        truth-free block sizes and memberships
  scores/                   suffstats/*.parquet, elev_points.csv, elev_boot.csv, curves_elev.csv, lag_points.csv, lag_summary.csv, verdicts.csv, winrates.csv, regret.csv, shares.csv, timing.csv, status.csv
  report/                   report.html, tables/, figures/, numbers.json
  notebooks/                executed notebook copies
  logs/                     commands.jsonl, <stage>_<worker>.log
  attain.json, LOCK.json
```

The longest path on the laptop stays below Windows' 260-character limit, because setting ids are short codes.

### 13.2 Units, keys and resume

**Unit** = (site, window, setting_id, method, variant, family, mode, N, rep).

Keys (v13 §2.11, extended):
- `package_key` (per year shard): the record, package, obs-core, domain, levels and subsets sections, the store fingerprints, the IOC chunk hashes, the sensing-time file hash and the **dom_version**. No factor level enters it.
- `effective_key`: the projection of the cell onto the method's dependency signature (§2.9).
- `config_key`: the method CONFIG, its FROZEN section, the variant, n_boot and the threads.
- `code_key`: sha256 of:
  - the method module;
  - the shared core (obs core, domain, levels, reader, inputs);
  - the **source of every registry entry the unit's effective key uses** (`inspect.getsource` of the level and its helpers);
  - the imported `pyintertidal` modules;
  - `granadeiro_fit.R` where used.

  Editing or adding one level therefore changes only the units of that level. A unit test asserts that adding a dummy level leaves every existing `inputs_key` unchanged.
- `inputs_key` = sha256(package_keys of the window's years, subset_key, effective_key, method, variant, config_key, code_key).

A unit whose `inputs_key` matches a finished product is skipped (resume). Adding levels, windows, N values or reps creates new units only.

| Change after a unit is done | Invalidates |
|---|---|
| A new year shard (2017-2022) for the same dom_version | Nothing in W1-W3 or Y2023-Y2025 units; W5, W9, Y2017-Y2022 and the centred windows are new units |
| obs core, domain, levels, reader, record, package, subsets or sensing times | Every unit of that site |
| One factor registry entry | The units whose effective key uses that entry |
| A method module, card or FROZEN section | That method's units |
| truth, judges, scoring, inference, curves, metrics_v2, analysis_v2, blocks | `score` and `report` only |

### 13.3 CLI (`python -X utf8 -m experiments.validation_v2 <command>`)

| Command | What it does | Who / where |
|---|---|---|
| `fetch ioc \| judges \| stac-meta \| sensing \| cmems \| tides \| emodnet \| pnoa-meta \| vak-meta \| models` | Network downloads into the gitignored data folders: IOC 2017-2025; judge series (with time-stamp conventions); the STAC sidecar; sun angles and TLEs; CMEMS SSH; eo-tides/pyTMD model checks; EMODnet tiles; PNOA flight metadata; Vaklodingen source metadata; the tide-model availability report | user, server |
| `inputs` | §1.11 checks; writes inputs_report.json | user, server |
| `check-env` | Python/R/pyTMD/eo-tides versions, tide models loadable, R packages, venv-dea, OmniCloudMask, Octave, disk type and throughput per volume, RAM, BLAS; CLP/CLM availability | user, server |
| `gate time` | G-time (§1.3) | user, server |
| `prepare <site> [--years 2023-2025] [--step pass1\|ocm\|settings\|geometry\|levels\|pass2\|tiles\|subsets]` | Builds or extends the package shards (resumable per step and year) | user, server |
| `truth <site>` | Elevation truths (all epochs, cell support, PNOA mask), truth-error model, strata, lag truths per window | user, server |
| `pilot crosstab \| dom \| reps \| units \| attain` | The SCL × CLD cross-tab (freezes p); dom sizes and refusal; computability; per-method cpu/wall, RSS and read_mb at the Ems and Wadden; attainability (§9.8). No truth. | user, server |
| `gate <name> \| all` | §10.1 gates | user, server (some local) |
| `control [--method m] [--world gating\|misspec]` | Positive control on the synthetic store | laptop and server |
| `timing` | Dedicated timing runs (§8.2); refuses unless the machine is quiet | user, server |
| `lock --phase A\|B` | Writes LOCK.json (§9.9) | user, server |
| `plan --design <name> [--sites ...]` | Writes `plans/<design>.csv` with est_hours, est_ram_gb and est_read_gb per unit; prints totals and the Campaign 1 projection against the 10-day cap, with the cut order | user |
| `run --plans <d1,d2,...> --worker-id k [--threads n] [--steal-stale]` | Consumes the plans in priority order through exclusive, instance-major claims under the RAM ledger | user, server (tmux panes) |
| `status [--watch 600]` | status.csv: done, running, stale, hung, failed, not_computable, not_applicable per design; completeness and projected finish | user |
| `score --blocks` | Truth-free block sizes and testable family members (§9.1, §9.2) | user, server |
| `score [--part lag\|elev\|curves\|timing\|all]` | Scores finished products into sufficient statistics and score tables (refuses real 10 m data before the relevant lock) | user, server |
| `report` | §11 | user, server |

### 13.4 Scheduling, workers and I/O

- **Instance-major claims.** A worker claims an instance (site, cell, family, mode, N, rep) and runs all of its cheap methods in sequence. One **fused block reader** streams each package tile once and feeds every method of the instance. Granadeiro and DEA (4 processes) are separate units that read through the same tiles.
- **Package layout.** After pass 2, each year shard is transposed into **pixel-major tiles**: 4,096 dom pixels × all dates of the year × bands, contiguous on disk. A column block then reads in large contiguous chunks instead of T × bands short seeks. If the transposition is not ready, column blocks are at least 512k pixels wide (≥ 1 MB contiguous per row). The package sits on an SSD if the server has one [V9]. The pilot reports read_mb and the throughput per unit.
- **Worker count.**
  - The pilot measures cpu_self/wall per unit. Most per-pixel code is element-wise numpy or single-threaded.
  - If the median ratio is below 1.5, the server runs **8-12 cheap workers at BLAS 1** under the RAM ledger. Otherwise it runs 4 workers at BLAS 4.
  - The budget (§14) is computed at 4 workers and states the gain if more fit.
- **I/O cap:** at most 3 concurrent units that read the Zarr stores (prepare, the OCM step, Campaign-2 passes). A semaphore file in `claims/` enforces it. Frame methods read the package-backed frame view and do not count.
- **Granadeiro** runs in its own pane with at most 2 cheap workers beside it, BLAS 1 and 16 R workers.

### 13.5 RAM, failures and hangs

- **RAM reservation ledger.** `claims/ram_ledger.json` sits under a file lock. A worker reserves `est_ram_gb × 1.25` before claiming a unit, and releases it on finish or failure. A claim proceeds only if the reservations plus a 6 GB OS margin fit in physical RAM. This replaces free-RAM polling, which was a check-then-act race between workers.
- **Memory limits.**
  - Each unit runs under `systemd-run --user --scope -p MemoryMax=<est × 1.5>G`, or `prlimit --as` if user scopes are unavailable. A unit that hits its limit fails and is re-planned with a larger estimate, instead of triggering the OOM killer.
  - Cheap workers set `oom_score_adj = 1000` at start, so Granadeiro's processes are the last to be killed.
- **Granadeiro checkpoints** each collected R job result to its stage-partial file, so a kill loses at most the jobs in flight, not a whole stage. Revision 1's StageCache resumed only per stage: up to about 1.6 h at the Ems.
- **Wall cap:** a unit running longer than 3 × est_hours is marked failed (kept as `.old-<utc>`) and reported.
- **Hang detection:** a unit whose progress counter (§8.1) has not changed for 30 min is marked hung and failed, even if its monitor thread still heart-beats.
- **Claims** (existing runner): exclusive claim files (`O_CREAT|O_EXCL`), 60 s heartbeats, stale steal after 30 min under a steal lock, atomic publish keeping failed attempts as `.old-<utc>`, and `STOP` / `STOP-<worker>` files for graceful stops.

### 13.6 Scoring storage

- `score` writes, per product × view × stratum × block, the **sufficient statistics**: n, Σe, Σe², Σt, Σt², Σz, Σz², Σzt, the HPE sums, and the CAE sums per fill. Pair views get their own records, so ΔMSE_c needs no pixel data. Per-band sums are kept for the band-level test.
- Additive metrics and every confirmatory quantity are bootstrapped from these records as weight-matrix products (B = 20,000 costs seconds per family).
- Exact quantiles (median bias, NMAD, LE95, HFI) and variograms are computed from the pixel data only for P0, the native cells and the family contrasts. Elsewhere they are point values without CIs.
- Scoring is incremental per stage and runs per site in parallel. Its budget is about 10 wh plus one day of the user's time for the report (§14).

### 13.7 tmux layout, queue and logs

- **tmux layout** (session `v2`):

  | Window | Role |
  |---|---|
  | `prep` | prepare, truth, settings pass, scoring |
  | `w1`..`wk` | cheap workers (k from the pilot) |
  | `gran` | Granadeiro, with at most 2 cheap workers beside it |
  | `mon` | `status --watch 600` and `tail -F` of the logs |

- **Multi-plan queue.** Each worker runs `run --plans lag,base,levels,native,years,tuning,g1,ofat,k1 --worker-id k` and always takes the highest-priority claimable unit. The user starts the panes once per working day. STOP files stop them gracefully, and `status --watch` shows the projected finish. The budget assumes 75-80 % utilisation, to allow for nights, weekends and review turnarounds.
- **Logs:** every command is piped through `2>&1 | tee -a results/validation_v2/logs/<stage>_<worker>.log`. Every command line is appended to `logs/commands.jsonl` with git HEAD and a timestamp. Each unit writes `unit.log` and `run.json` (versions, threads, timings, RSS, read volume, keys).

### 13.8 Server command pattern (the user runs every command)

```bash
# once per server session
source env.sh                     # $PY, $RSCRIPT, ZARR_CUBES_ROOT, TMPDIR
tmux new -s v2

# Stage 0 (examples)
$PY -X utf8 -m experiments.validation_v2 fetch ioc        2>&1 | tee -a results/validation_v2/logs/fetch.log
$PY -X utf8 -m experiments.validation_v2 fetch sensing    2>&1 | tee -a results/validation_v2/logs/fetch.log
$PY -X utf8 -m experiments.validation_v2 inputs           2>&1 | tee -a results/validation_v2/logs/inputs.log
$PY -X utf8 -m experiments.validation_v2 gate time        2>&1 | tee -a results/validation_v2/logs/gates.log
$PY -X utf8 -m experiments.validation_v2 prepare escalda --years 2023-2025 2>&1 | tee -a results/validation_v2/logs/prepare_escalda.log
$PY -X utf8 -m experiments.validation_v2 truth escalda
$PY -X utf8 -m experiments.validation_v2 pilot crosstab
$PY -X utf8 -m experiments.validation_v2 pilot dom
$PY -X utf8 -m experiments.validation_v2 pilot units
$PY -X utf8 -m experiments.validation_v2 gate all
$PY -X utf8 -m experiments.validation_v2 control
$PY -X utf8 -m experiments.validation_v2 pilot attain
$PY -X utf8 -m experiments.validation_v2 lock --phase A

# a stage: plan, then one pane per worker
$PY -X utf8 -m experiments.validation_v2 plan --design lag
$PY -X utf8 -m experiments.validation_v2 run --plans lag --worker-id 1 2>&1 | tee -a results/validation_v2/logs/lag_w1.log
$PY -X utf8 -m experiments.validation_v2 score --part lag
```

The assistant hands the user a complete command block per stage, after the code for that stage has been reviewed and pushed.

---

## 14. Compute, memory and disk budget (server: 20 cores, 62 GB RAM)

### 14.1 Assumptions

- **Worker-hour (wh):** one cheap worker with 4 BLAS threads (or 4 processes) for one hour.
  - The base throughput is **4 workers × 24 h × 0.78 utilisation ≈ 75 wh per day**. The utilisation allows for nights, weekends and review turnarounds.
  - If the pilot shows cpu_self/wall < 1.5 per worker, 8-12 workers at BLAS 1 fit, and the cheap stages shrink by roughly 40 % (§13.4).
- **Base cost of one cell** = all five sites, W3, N = all, every Tier-1 cheap elevation and lag method, on S0, without any bootstrap ≈ **7 wh** at P0 and ≈ 6.5 wh in other cells, where Ferrol runs lag methods only. The breakdown comes from v13 §9 and the light probes:

  | Component | wh |
  |---|---|
  | marea_pair + NIDEM + GM20 + HD26 | 2.0 |
  | DEA | 1.5 |
  | LogiTide2DEM | 0.5-1.0 |
  | Bué | 0.5 |
  | waterline E6a/E6b (cell aggregation) | 1.0 |
  | Coast X-Ray | 0.3 |
  | other lag rows | 0.5 |
  | frequency rows | < 0.1 |
  | harness (observations, domain, levels, CAE fills) | 0.3 |

- **The C0 inflation is removed.** S0 is fixed and C0 never enters dom (§2.10), so a C0 cell costs about the same as any other.
- **Window scaling** relative to W3: W1 and calendar years ×0.4, W2 ×0.7, W5 ×1.7, W9 ×3.
- **Effective-key deduplication** removes about 10 % of the G1' cost.
- **Curves:** Σ N·R / T × the cell cost, plus a per-instance constant of about 1 min per site-instance (domain step, level rows, fills, writes), calibrated in the pilot.
- **Bootstrap:** MAREA's 30-draw bootstrap (about 1 h at the Ems) and the GM20 refits run only in Stage L at P0. Revision 1 implicitly ran them in every lag-scored cell, which would have added 150-200 wh.
- **Frame methods** read the package-backed frame view, so a curve unit does not decode whole Zarr chunks. Units that do read the Zarr stores are charged min(1, 16N/T) × the window read.
- Every figure is an extrapolation. The Stage 0 pilot (`pilot units`) and the timing runs recalibrate them, and `plan` prints the projection against the 10-day cap before each stage.

### 14.2 Stage budget

**Campaign 1 (two calendar weeks):**

| Stage | Content | Cheap wh | Granadeiro | Wall (days) |
|---|---|---|---|---|
| 0a | fetch (IOC, judges, STAC, sensing, CMEMS, tides, EMODnet, PNOA and Vaklodingen metadata), inputs, check-env, G-time | – | – | 0.5 |
| 0b | prepare 2023-2025 (pass 1 with uint8 counters, the OCM step of 10-20 CPU-h, the settings pass, pass 2, tiles), truth with the truth-error model, pilots (crosstab, dom, units) | ~25 | – | 1.0 |
| 0c | gates, positive control (gating and misspecified worlds), G-granadeiro at Villaviciosa (~1 h), G-bijagos sub-region (~3-6 h, overnight), attain, lock-A | ~10 | 4-7 h | 1.0 |
| L | Lag benchmark at P0 W3 (n_boot 30 and GM20 refits) + W1, W2, Y2023, Y2024 + models (L14-L16) + physics + per-orbit | ~20 | – | 0.3 |
| T | Dedicated timing runs (quiet machine, ≤ 10 h) | – | – | 0.4 |
| B | lock-B; P0 (7); as published (~8); level sources LT, LM, LG (~15); native cells (~12); calendar years Y2023-Y2025 (~8); tuning at Escalda and tuned rows (~10); seeds (~3); split-half (~7); label instants (~1); EOT20 only at W3 (~7); Villaviciosa boundaries (~0.5) | ~80 | – | 1.1 |
| G | G-base at 5 sites + G-nir at Villaviciosa and Escalda, alone except for 2 cheap workers on S' | (2 workers, ~50 wh of S') | 26-41 h | 1.1-1.7 |
| S' | OFAT (16 cells) + G1' (17 new cells): 33 cells × 6.5 × 0.9 ≈ 195; matched-N controls ≈ 25 | ~220 | – | 2.2 after G |
| K | K1' (17 instances per site) | ~45 | – | 0.6 |
| Final | score --blocks, incremental scoring, report | ~10 | – | 1.0 |
| **Campaign 1 total** | | **~410** | **30-48 h** | **≈ 9.5 days; ≈ 10.5-11 days with a 20 % contingency on the compute stages** |

With 8 or more workers (if the pilot allows), Campaign 1 takes about 8 days including contingency.

**One-week fallback:** Stage 0 (2.5 d); L (0.3 d); the core of B (P0, as published, LT and LM rows, native cells: ~42 wh, 0.6 d); G-base at Villaviciosa, Escalda and Ems (~0.8 d), with 2 workers on the OFAT cloud and index cells (12 cells ≈ 70 wh, finishing about 0.5 d after G); scoring (0.5 d). That makes about 5.5 days, or 6.5 days with contingency.

**Campaign 2 (separate campaign, after the 2017-2022 batch and the gauge extension):**

| Stage | Content | Cheap wh | Granadeiro | Wall (days) |
|---|---|---|---|---|
| 0d | prepare 2017-2022 (pass 1, OCM for the G2 and curve settings, settings pass, pass 2, tiles), truth rebuild, gauge QC | ~35 | – | 1.0 |
| W | Window OFAT W2, W5, W9 (~38); EOT20 only at W9 (~20); G2 new cells (~70); survey-centred windows (~22); Y2017-Y2022 (~17); NIDEM W9 and Coast X-Ray W5 as published and native (~10); Campaign-2 OFAT levels C5, C7, class-11-blocked C4, AWEI_nsh, WRI, D4 (~35); the NDWI@20 slice (~18); the I × T check (~4) | ~235 | – | 3.1 |
| K-1B | K3 (~50), K4 (~60), K5 (~60) | ~170 | – | 2.3 |
| G-1B | G-nir at Wadden and Ems (15-20 h), G-L9 at 2 sites (36-60 h), K6 (15-25 h), G-tune (~8 h) | 2 workers beside | 75-115 h | 3-5 (overlapping) |
| Away | Tauranga: download, prepare, P0, as published, native cells [V27] | ~20 | – | 1 (download) |
| Final | rescoring, report | ~10 | – | 0.7 |
| **Campaign 2 total** | | **~470** | **75-115 h** | **≈ 9-11 days** |

**Tier 2:** about 300-500 wh plus 4-8 days of Granadeiro, i.e. 1-2 weeks.

### 14.3 Memory per unit (worst case, Ems)

| Unit | W3 | W9 | Notes |
|---|---|---|---|
| marea_pair (+ L01) | ~1 GB | ~3 GB | column blocks |
| NIDEM, Coast X-Ray | ~2.5 GB | ~4 GB | per-interval row-block stacks on the frame view |
| DEA | ~6 GB (4 processes) | ~8 GB | tiles ≤ 100k candidate px |
| Waterline (cell aggregation) | ~2 GB | ~3 GB | the pilot at Wadden W3 sets the estimate, under a memory limit |
| LogiTide, Bué, frequency, lag rows | < 2 GB | < 3 GB | pixel chunks |
| Domain step (A-mode) | ~1 GB | ~1.5 GB | |
| prepare (pass 1, per site) | ≤ 8 GB | ≤ 8 GB | never beside Granadeiro |
| OCM step | ~4-6 GB per process | | restricted to the bounding box of dom plus 1 km |
| Granadeiro | 5-6 GB Python + 16 R workers (0.24-1.8 GB each) | ~15-18 GB Python + R | W9 runs are not planned |

The RAM ledger (§13.5) enforces Σ reservations + 6 GB ≤ 62 GB. Granadeiro (≤ 45 GB at W5) runs with at most two cheap workers.

### 14.4 Disk

| Item | Estimate | Basis |
|---|---|---|
| Zarr stores 2023-2025 | ~70-100 GB | 25.9 Mpx of frames × ~465 scenes × 6-7.9 B/px (the downloader's BYTES_PER_PIXEL_DATE = 6.0 is the lower bound) |
| Zarr stores 2017-2025 | ~205-300 GB | the same × ~1,316-1,390 scenes |
| Package shards at dom (6 int16 bands + SCL + CLD = 14 B per observation) | W3 ~45 GB; W9 ~130 GB | dom ≈ 25-30 % of the frames (≈ 7 Mpx), printed by `pilot dom`; refusal above the budget |
| Per-setting counters (uint8 per year) | ~2 GB per year; ~16 GB for 9 years | ~34 settings × 2 counters × 25.9 Mpx |
| Frame validity, OCM and buffer bits | ~15-20 GB at 9 years | curve settings only; OCM and buffer bits at dom |
| Products, N = all (on S0) | ~70 GB (Campaign 1) | ~6,800 products × ~10 MB |
| Curve products (D_truth + neighbourhoods) | ~10 GB (Campaign 1); ~25 GB more in Campaign 2 | ~2,000 products × ~5 MB |
| Sufficient statistics, scores, logs, report | < 10 GB | |
| TMPDIR for R exchange files | ≥ 10 GB | v13 |
| **Total** | **≈ 0.25-0.3 TB for Campaign 1; ≈ 0.5-0.6 TB with Campaign 2** | `inputs` and `check-env` report free space and throughput [V9] |

**Retention policy:**
- Curve product maps of replicates ≥ 1 are deleted once their sufficient statistics are verified; rep 0 is kept.
- Superseded attempts (`.old-*`) are compressed after scoring.
- Contended timing logs are compressed.

**If the volume is still short,** in this order:
1. keep B02, B04 and B12 out of the package, and read them from the Zarr stores in the AWEI cells;
2. move curve products to a second disk;
3. keep N = all products of OFAT and G1' cells as sufficient statistics plus quantile sketches only.

### 14.5 Calendar for Campaign 1

Day 1 is the day the V1-V3 code passes its gates. Before that, the machine is not booked.

| Day | Server | User |
|---|---|---|
| 1 | fetch (all), inputs, check-env, gate time; prepare Villaviciosa, Ferrol and Escalda in parallel | starts the commands, reads the inputs report and the G-time verdicts |
| 2 | prepare Wadden and Ems; OCM step; truth (all sites); pilots | reviews the pilots (dom sizes, units, crosstab → p) |
| 3 | gates, control; G-granadeiro and G-bijagos overnight; attain; lock-A | approves lock-A |
| 4 | Stage L in the morning; score lag; Stage T overnight | reviews the lag chapter |
| 5 | lock-B; Stage B | – |
| 6 | Stage B ends; G-base starts in `gran`, with 2 workers on G1' | – |
| 7 | G-base continues; G-nir at Villaviciosa and Escalda | – |
| 8-9 | S' at 4 workers | – |
| 10 | S' ends; K1' | – |
| 11 | score --blocks, scoring, report draft | reviews the draft |
| 12-13 | contingency (20 %) | – |

Workers keep running through nights and weekends from the priority queue (§13.7). The user intervenes only to start panes, stop them, or approve a lock.

---

## 15. Milestones and order of runs

Each milestone ends with a code review, a commit and push by the authors, and a command block for the user. The assistant never runs server commands.

| Milestone | The assistant delivers (code, tests, laptop checks) | The user runs (server) | Output |
|---|---|---|---|
| **V0** | The five fixes of §12.3; tests for domain, judges, truth and product; the whole suite green on the laptop | `pytest` on the server (Linux) | Green suite on both machines |
| **V1** | `inputs.py` (block cache, lattice20, per-scene offset), `sensing_time.py`, `settings.py` (per-entry hashes), `designs.py`, the obs registries (NDWI path bitwise; 20 m rule), package year shards (dom_version, uint8 counters, OCM step, settings pass, MNAR table, tiles), `levels` (union period; LB/LT/LM/LG), `truth` (survey-matched default, cell support, PNOA mask, truth-error model), `fetch/*`, the CLI commands `fetch/inputs/check-env/gate time/prepare/truth/pilot`; tests; laptop runs on the 20 m legacy cubes and a small Zarr extract | `fetch *`; `inputs`; `check-env`; `gate time`; `prepare <site> --years 2023-2025` (all sites, 2-3 in parallel); `truth <site>`; `pilot crosstab`, `pilot dom` | inputs report, G-time verdicts, crosstab; **the user freezes p [V1] and approves dom** |
| **V2** | Lag adapters (§12.1 `lags/`, including L16 and L01v-orbit), `readout.py`, `blocks.py`, the lag part of `scoring`/`inference`, `control.py` (lag worlds), `timing.py`; gates G-zarr, G-offset, G-record, G-keep, G-marea, G-lagtruth, G-level, G-ocm; **every elevation method card with its CONFIG, native cell and tuning grid** (needed for lock-A); `pilot attain` | `gate ...`; `control --kind lag`; `pilot attain`; `lock --phase A`; `plan --design lag`; `run --plans lag` (2-4 workers); `score --part lag` | **Stage L: the lag chapter (first results), family L-P1** |
| **V3** | Tier-1 cheap elevation adapters (E0-E8, E6b Salameh configuration, `published`), the venv-dea runner, `reader.py` (frame view, fused reader), the elevation part of `scoring`/`inference` (sufficient statistics, Romano-Wolf, IUT), the report skeleton, notebooks 00-05; gates G-nidem, G-nidem-code, G-dea, G-flood2topo, G-costa, G-frameview, G-score; `pilot units` | Install venv-dea [V6], Octave [V17] and OmniCloudMask [V1]; gates; `control`; `timing` (Stage T, overnight); `lock --phase B`; `run --plans base,levels,native,years,tuning,g1,ofat,k1`; `score` | **Stages B, S', K: P0 tables, families P1/P2/P3, native cells, settings robustness, base curves** |
| **V4** | The Granadeiro adapter (re-sequenced v8 on the frame view; job-level checkpoints), the Bijagós sub-region recipe, G-granadeiro, G-bijagos | Bijagós download [V14]; `gate granadeiro`, `gate bijagos` (Stage 0 if ready, otherwise before G); `plan --design granadeiro-base`; `run` in pane `gran` right after Stage B | **Stage G: Granadeiro at P0 and NIR; family L-P2** |
| **V5** | Campaign-2 designs (windows, G2, survey-centred, Y2017-Y2022, K3-K5, the I × T check, Campaign-2 OFAT levels, the NDWI@20 slice), the multi-epoch views and the epoch analysis, Granadeiro L9/curves/tuning, the away site | When 2017-2022 and the gauge extension are complete: `inputs`; `prepare <site> --years 2017-2022`; `truth <site>`; `plan`/`run` for the Campaign-2 designs, then `granadeiro-l9`, `granadeiro-curves`, `granadeiro-tune` | **Campaign 2 campaign** |
| **V6** | The full `inference` (shares, mixed models in R, meta-analysis, verdicts, regret, the where-analysis with the decomposition and honest partitions), `report.py`, notebooks 06-07 | `score --part all`; `report`; papermill runs of the notebooks | **The report** (Campaign 1 version first, marked provisional where needed) |
| **V7** | Tier-2 rows and levels, in the order of §5.14 | as above | Tier-2 supplement |

If V4 is late, Stage G runs after K, and the Campaign 1 calendar grows by Granadeiro's wall time (1.1-1.7 days). It is never cut below three sites.

**Order of runs:**

```
Stage 0 (fetch, inputs, G-time, prepare 2023-2025, truth, pilots, gates, control, attain, lock-A)
  → L (lag benchmark; L-P1)
  → T (timing, overnight)
  → lock-B → B (P0, as published, level sources, native cells, Y2023-Y2025, tuning, seeds, split-half)
  → G (G-base + G-nir; L-P2)  ∥  S' on 2 workers
  → S' (4 workers) → K (K1')
  → score --blocks → score → report (Campaign 1)
  → [2017-2022 batch + gauge extension] Campaign 2 campaign → final report
  → Tier 2
```

The lag benchmark is scored before any elevation score. No lag result can change any elevation setting, because every elevation CONFIG is frozen at lock-A.

---

## 16. Risks

1. **Sensing times.**
   - G-time can fail: the angle grids may be computed at one reference time per tile, or TLE access may be refused.
   - Corrected instants change every earlier lag and level, which must be explained against earlier results.
   - Mitigations: two independent methods, per-group fallbacks, the label-instants sensitivity, and the user's decision if both fail.
2. **Data years before 2023.** Judge series and boundary IOC chunks may be patchy in 2017-2022. Mitigations: the coverage report, the EOT20-only sensitivity, and lag scoring limited to covered windows.
3. **Boundary datum drift.** Adding IOC chunks shifts h_B's demeaning (§1.5). Mitigations: fetch everything before the first `prepare`; IOC hashes in `package_key`.
4. **Radiometric and processing heterogeneity.**
   - Collection-1 reprocessing means baselines are mixed by download date, not only by acquisition date.
   - G-zarr may fail for 2022-2023 if CDSE now delivers baseline 05.10 where the legacy cubes held 05.09.
   - Mitigations: the per-scene offset rule, G-offset, the pre-registered G-zarr response, and the baseline as a covariate.
5. **Truth error and morphological change.**
   - The truth-error model is well identified only at Escalda (yearly surveys). It is transferred to Wadden and Ems.
   - Mitigations: ΔMSE_c tests do not depend on it; σ²_T × {0.5, 1.5} sensitivities; survey-matched truth; the stable stratum.
6. **Weak and pseudo-replicated lag evidence.** There are 13 locations, 8 in the Ems; truth (a) is close to MAREA's estimand; the bathtub calibration moves all imagery estimators together. Mitigations: three co-primary truths, site weighting, site agreement, development-free judges, and the non-circular family P1.
7. **Home advantage.** Mitigations: table notes, the symmetric tuning budget, the forking-paths disclosure, the fairness ladder, the away site (if approved), and the confirmatory split left to the authors.
8. **Competitor fidelity.**
   - Several rows are F3 re-implementations: LogiTide2DEM's separation rule, Coast X-Ray's datum reading, Bué's saturation reading, and the Salameh configuration of E6b.
   - Pool's full text was not accessible.
   - GM20's details were not re-verified.
   - Mitigations: the printing rule, declared deviations, author review of the cards, and authors' data requested.
9. **Level sources.** CMEMS archive depth and temporal resolution may be limited, the inner estuaries may be unresolved, and the eo-tides ranking may not cover the sites. LG uses the judges. Mitigations: availability checks before lock-A, unresolved flags, labelled upper-bound rows, and NOT AVAILABLE rather than substitutes.
10. **Index-scaled constants.** Applying NDWI-unit thresholds to other indices through the standardisation can disadvantage methods tuned on NDWI. Mitigations: native rows and native cells beside them; declared deviations.
11. **Small-N computability.** The MAREA family is NOT COMPUTABLE at N ≲ 100 in A-mode, and NIDEM needs every interval. Mitigations: k/R, the CAE fills, NOT COMPUTABLE counted as a loss and shown.
12. **Granadeiro cost and R drift.** G-base runs right after Stage B, with job-level checkpoints. It is never cut below three sites. The `ctl_unpinned` variant exists, and R identities are recorded.
13. **Dependencies:**
    - DEA's datacube/eodatasets3 stack (side venv);
    - pyIntertidalDEM (conda, git);
    - Octave for the Flood2Topo gate;
    - OmniCloudMask (PyTorch, CPU);
    - skyfield and the TLE archive;
    - eo-tides, pyTMD and the model atlases;
    - the copernicusmarine toolbox;
    - the R packages for the analysis.

    Each is isolated. Its absence turns the dependent row or check into "not validated" or NOT AVAILABLE, never into a silent substitute. The only exception is the declared C3-for-C11 replacement.
14. **Disk and I/O.** The disk type is unknown. Mitigations: pixel-major tiles, instance-major scheduling, the frame view, `pilot dom` and `pilot units`, and the retention policy.
15. **Multiplicity and strictness.** With the headline rule of §9.7, few or no headlines is a likely and legitimate outcome. The report says so in advance.
16. **Power.** Villaviciosa is expected to be descriptive. The confirmatory evidence rests on three Dutch sites, one of them (the Ems) with an epoch-mismatched inner truth. The P1 band-level test with about 7-12 bands per site has modest power.
17. **Zarr format specifics.** The store format is experimental on the openEO side. Mitigations: FORMAT_ID, the G-zarr gate and fingerprints in the keys.
18. **Schedule.** Code readiness (V1-V4) gates the compute clock. The two-week Campaign 1 calendar assumes the code has passed its gates.
19. **Laptop limits** (about 1.5 GB of free RAM). Only 20 m legacy cubes, small Zarr extracts and the synthetic control run locally.

---

## 17. Decisions needing the user

Each item gives the recommended default first.

- **V1. Cloud levels.**
  - C1 (SCL {4,5,6,7}) is the base.
  - G1' cloud levels: C1 (per-pixel mask), C2 (transition-zone rule), C4 (ITEM-PQ, competitor-native) and C11 (OmniCloudMask, an independent "other filtering method").
  - C0 ("no filtering") runs only as an OFAT lower bound on S0, and C3 (CLD + buffer) as an OFAT cell. C3 replaces C11 in G1' if OmniCloudMask cannot be installed.
  - p is set by the mechanical cross-tab rule.
  - Install OmniCloudMask (PyTorch CPU) on the server.
- **V2. Factor levels.**
  - G1' = C{1, 2, 4, 11} × I{NDWI, MNDWI, NIR} × D{fixed, edge-Otsu}.
  - D2 (guarded Otsu) runs as OFAT and in LogiTide's native cell.
  - D4 is a Campaign-2 naive baseline.
  - NDWI@20 joins the index OFAT, and every index with a 20 m band is computed at 20 m.
- **V3. New headline metric.** HPE_c (hydroperiod error, h/day) with a per-cell margin equivalent to 2 cm. HFI_D (joint verdicts only) and CAE (two fills) are companions.
- **V4. Windows and epochs.**
  - Survey-matched truth is the default for every window, and the nested-window effect is confirmed only at Escalda.
  - Stable cells need at least 2 surveys within the window ± 1 year and a range ≤ 0.20 m.
  - Calendar-year windows provide the archive replicates.
- **V5. Extending gauges to 2017-2022.** Extend the judge series (RWS, WSV, IOC, PdE) and the boundary IOC chunks before any prepare. Without the extension, lags are scored only in W1-W3 and the 2023-2025 years. EOT20-only sensitivities run at W3 (Campaign 1) and W9 (Campaign 2).
- **V6. DEA Intertidal environment.** A side venv with `dea-intertidal[datacube]==2.5.3`. Fallback: vendor the seven pure functions (Apache-2.0, NOTICE kept) and gate them bitwise wherever the package installs.
- **V7. Tier-2 scope.** Pool ADP as a clean-room port (CASSIE is GPL-3.0; no code copied); Khan via its Apache-2.0 code; Bell adapted; Dynamic World if its model runs on the server; SWOT, MOHID and C12 (s2cloudless) only if time remains.
- **V8. Granadeiro scope.**
  - Campaign 1: G-base at 5 sites and G-nir at Villaviciosa and Escalda, about 1.1-1.7 days.
  - Campaign 2: G-nir at Wadden and Ems, L9 at Villaviciosa and Escalda, K6 at {45, 90, 180}, and G-tune, about 3-5 days.
  - No headline against Granadeiro is possible in Campaign 1 (robustness not assessed).
- **V9. Server and disk facts.**
  - Report the disk type (HDD or SSD), the free space and the sequential throughput for the volumes holding the cubes, the package, the results and TMPDIR.
  - The need is about 0.3 TB for Campaign 1 and 0.6 TB with Campaign 2.
  - Put the package on an SSD if there is one.
- **V10. New analysis packages on the server:** R lme4, emmeans, metafor and partykit; optionally statsmodels in the Python venv. Without them, the numpy fallbacks are used and labelled.
- **V11. Truth-blind index constants.** The NIR fixed threshold is the pooled Otsu, and the standardisation scale s_k the IQR ratio to NDWI, both on keep_full(P0, W3) per site from per-year histograms. Both are frozen before lock-A.
- **V12. Binarised-input variant** for continuous estimators: Tier 2.
- **V13. Processing heterogeneity.** Decide the offset regime and record the baseline per scene. Do not exclude scenes. The baseline is a covariate in window comparisons.
- **V14. Granadeiro and Bijagós.** Reproduce the Bijagós rasters on a sub-region (one or two published tiles, or about 30 × 30 km), with L2A (declared), at the cost of one extra openEO download. Tolerances: median \|Δz\| ≤ 0.15 m and median \|Δlag\| ≤ 10 min. G-granadeiro runs at Villaviciosa only.
- **V15. Confirmatory evaluation** (the sealed RTK split, or a new site) is not part of this plan. It stays the authors' later decision.
- **V16. External data and registrations:**
  - AVISO (FES2022), TPXO (OTPS), ICDC (HAMTIDE11) and DTU23 for the model rows and LT. FES2022 is the primary global model if installed by lock-A, else EOT20;
  - a Space-Track account for the historical TLEs (G-time, method B);
  - a Copernicus Marine account for AMM15/IBI (LM, L16);
  - ask Chen & Wang for the Changhua data;
  - a GEE account only if the Pool/CASSIE waterline check is wanted.
- **V17. Octave on the server** for the Flood2Topo gate. Without it, E5a is labelled F3, not F2.
- **V18. PNOA flight date and time** per strip, from the CNIG metadata. They are needed for the water-surface mask and the survey-centred windows at Villaviciosa. Without them, Villaviciosa elevation is descriptive.
- **V19. Lag and families.**
  - HW and LW lags are reported, not co-primary.
  - P1, P2 and P3 are separate Romano-Wolf families; a fixed sequence (P1 before P2) applies only if the user prefers it.
  - The P2 list = E1, E2, E3, E4, E5a, E6a, E7, E8.
- **V20. Two-phase lock.** lock-A (before Stage L) freezes the analysis plan, the lag methods **and every elevation-method CONFIG**. lock-B (before Stage B) adds only implementation-level items (code hashes, gate verdicts).
- **V21. Sensing times (G-time).** Approve the gate and its consequences: every instant, level and lag truth is rebuilt; the production lags are re-examined; a label-instants sensitivity runs. Choose the access path: the openEO angle bands (recommended) or the MTD_TL downloads; Space-Track TLEs for the cross-check.
- **V22. Water-level sources.** Approve LB/LT/LM/LG for elevation and family P3. LG uses the judges' series as an elevation-only, labelled upper bound, which amends revision 1's "not planned".
- **V23. Primary statistics.**
  - ΔMSE_c (verdict scale δ_m, margin 2 cm) and b* (margin 0.05) replace rmse_meanc and \|1 − b\| for inference.
  - The pair views, with an intersection-union over the two views.
  - Romano-Wolf with B = 20,000.
  - Block size from the map-difference variograms, without the 4 km cap, plus the band-level test for P1.
  - Villaviciosa is expected to be descriptive.
- **V24. Two campaigns and the cut order.**
  - Campaign 1 takes about 10.5-11 days including contingency, or about 8 with 8 or more workers.
  - Campaign 2 is a later campaign of about 9-11 days.
  - The mechanical cut list of §5.14 and the one-week fallback apply.
- **V25. Truth metadata.**
  - Collect the Vaklodingen source and density per sheet.
  - The 20 m cells are the scoring support at the Dutch sites.
  - RWS intertidal laser altimetry at Escalda becomes a second truth if available.
- **V26. Headline rule.** The seven conditions of §9.7. Campaign 1 headlines are labelled "provisional" until the nine-year archive check.
- **V27. Away site and author review.** Tauranga Harbour in Campaign 2 (Costa's home site; public LINZ LiDAR and gauges). Email each competitor's corresponding author the method card for review before lock-A.
- **V28. Symmetric tuning budget.** The grids of §5.12, tuned on Escalda, with tuned rows elsewhere.
- **V29. Development judges and the forking-paths disclosure.** The authors complete both lists before lock-A (§1.7, §9.9).
- **V30. Workers and the package layout.** The worker count follows the pilot: 8-12 at BLAS 1 if cpu/wall < 1.5, else 4 at BLAS 4. The package is transposed to pixel-major tiles after pass 2.

---

## 18. Changes after review (revision 1 → revision 2, 2026-10-09)

Three adversarial reviews examined revision 1: statistics and design (S), remote sensing (R), and compute (C). Every critical, high and major issue was fixed. The medium, moderate, low and minor issues were also addressed, except where the table says otherwise. The ids below are local to this section.

### 18.1 Statistics and design review (S)

| Id | Severity | Issue | Change | Where |
|---|---|---|---|---|
| S1 | critical | B = 5,000 gave a smallest bootstrap p of 4.0 × 10⁻⁴, above Holm's first threshold for the 128-test P2 family (3.9 × 10⁻⁴), so P2 could never reject | B = 20,000 (smallest p ≈ 1.0 × 10⁻⁴); a unit test asserts 2/(B + 1) < α/m for every frozen family; `pilot attain` before lock-A; Romano-Wolf step-down on the shared-weight bootstrap; intersection-union over the two views | §9.1, §9.2, §9.8, §12.3 (5) |
| S2 | high | Co-primary common views intersected every method, so contrasts depended on unrelated competitors and were restricted to the easy pixels | Pair views pair_q:A\|B and pair_raw:A\|B are co-primary and fixed per contrast; the all-member common view is descriptive; every headline must also hold under CAE (both fills) or at matched coverage | §7.1, §9.7 |
| S3 | high | P2 was tested only at P0 (MAREA's development setting), and the robustness grid was home-tilted, with no competitor-native cloud level | Native cells per P2 competitor, with a native-cell headline condition; G1' cloud levels are C1, C2, C4 (competitor-native) and C11 (independent); an empty robustness set means "not assessed, no headline" | §5.5, §5.7, §9.7 |
| S4 | high | The 4 km block cap understated the along-channel correlation of lag-driven errors; Villaviciosa had too few blocks; lag tests treated the 8 Ems locations as independent | Block size from truth-free variograms of the map differences, lattice up to 16 km, no cap; ≥ 25 blocks for confirmation; a band-level moving-block test for P1; untestable members removed truth-free before scoring; L-P uses a site-weighted statistic, exact sign flips and site agreement; pooled tests labelled pseudo-replicated | §9.1, §9.2, §6.5 |
| S5 | high | Truth and epoch error attenuate b; the SD-difference margin shrinks with common truth noise; \|1 − b\| is non-smooth near 1 | Accuracy is tested on ΔMSE_c (the paired loss differential, invariant in expectation to independent truth error), with verdicts on δ_m (method-term RMSE after σ²_T); slope on b* = b/λ̂, with a non-regular rule; a frozen, method-free truth-error model | §1.9, §7.2, §9.3 |
| S6 | high | Window length was confounded with epoch in opposite directions by site; the default truth was spatially mixed; the subtraction double-counted survey noise; the stable stratum was vacuous | Survey-matched truth is the default; the nested-window effect is confirmed only at Escalda; mismatch model σ²_m + σ²_s + D_W(Y), with noise counted once; baseline covariate; epoch gap as a stratum and GAM term; stable = ≥ 2 surveys within ± 1 year | §1.8, §1.9, §2.7, §5.9, §7.1, §7.5, §9.6 |
| S7 | medium | Disjoint and nested replicates were mixed; the N = 330 replicates overlapped by about 71 % | Disjoint only if R ≤ floor(T/N), nested otherwise, never mixed; calendar-year windows give archive replicates and replace the N = 330 headline condition; between-year SD printed | §2.8, §5.8, §9.1, §9.7 |
| S8 | medium | Lag read-outs penalised pixel methods; phases were wrapped about the truth; dE was tied to S2 instants; ties made Holm unattainable; L06 was missing at Stage L; TOST lacked a variance; TAU_BOUNDS was in-sample at Harlingen | Symmetric fallback ladder; wrapping about the truth-free 67.5 min; dE_all co-headline; \|e\| with exact sign flips and the attainable p printed; L-P2 as a separate Stage-G family; per-location TOST dropped; development-free judges sensitivity | §6.3-6.6 |
| S9 | medium | v13 D21 was dropped in practice; forking paths; lock-B came after the lag results | Symmetric tuning grids (Escalda) with tuned rows and internal parameters in LOSO; forking-paths disclosure; every elevation CONFIG frozen at lock-A; lock-B implementation-only | §5.12, §9.7, §9.9 |
| S10 | medium | The CAE fill was a competitor's sibling (E5c); bias handling was mixed | Two fills (constant and FREQ) with agreement required; per-part median centring; E5a-c excluded from CAE verdicts; fill points marked | §7.3.3 |
| S11 | medium | HPE used h_B, which saturates in amplified reaches; a fixed margin; HPE_raw undefined; HFI could reward noise | Gauge-interpolated h_ref at judged sites; a per-cell margin equivalent to 2 cm; HPE_raw dropped; HFI verdicts only jointly with ΔMSE_c; a noise-inflation unit test | §7.3.1-7.3.2, §10.3 |
| S12 | medium | The curve view was unspecified; the floor included truth error; TTT was unreachable; the curve model could not identify I and D | Fixed N-invariant curve view with fills; s_method after σ²_T; absolute TTT targets (0.20 and 0.30 m) plus the relative one; model with (1 \| subset) and no I or D terms | §7.4, §8.3 |
| S13 | medium | Factor shares pooled across truth types; seed noise was not separated; I × T was never estimated | Shares per site only; the seed-SD floor; an I × T check in Campaign 2, or aliasing stated | §9.4, §5.7, §5.11 |
| S14 | low | Verdict precedence was undefined; NOT COMPUTABLE survivorship; regret against a varying set | An explicit verdict order with the "clear" sub-label; NOT COMPUTABLE counted as a loss with the computable share printed; regret against a fixed comparator set | §9.3, §9.5 |
| S15 | low | The positive control was MAREA's own model; its bias tolerance was inconsistent with the estimands | Linear-mixing world over sub-pixel relief; per-method offsets from theory; removal only for a documented fault; misspecified worlds reported for every method | §10.2 |
| S16 | low | Harness cost was reported as shared; interval scores mixed 95 % and 50 % intervals | Attribution by dependency signature, with shared-split and stand-alone costs; interval scores only for nominal 95 % intervals (GM20 rescaled, declared) | §8.1, §6.4, §3.3.1 |

The "missing" list of review S maps as follows:
- power and attainability: §9.8;
- a genuine between-archive design: §5.8;
- native cells: §5.5;
- the decomposition MAREA − X = (MAREA − E0a) + (E0a − X): §9.6;
- per-pixel tide-model rows for every P2 method: LT, §1.6 and §5.4;
- the frozen truth-error model: §1.9;
- fixed denominators and untestable members: §9.2;
- the forking-paths disclosure: §9.9;
- D21: §5.12;
- the views for curves, shares and robustness: §7.4, §9.4 and §9.5;
- the stable-stratum survey rule and the epoch gap: §1.8 and §9.6;
- spatially honest partitions: §9.6;
- the LiDAR coverage statement: §1.8.

### 18.2 Remote-sensing review (R)

| Id | Severity | Issue | Change | Where |
|---|---|---|---|---|
| R1 | critical | Scene instants were datatake starts (early by about 0-10 min, varying by date), which biases every imagery lag low and adds level noise. A light check confirmed a 7-8 min spread within one relative orbit, and identical Ems and Wadden labels. | Gate G-time: sun-angle inversion (CDSE angle bands or MTD_TL) cross-checked by SGP4 orbit propagation, with consistency criteria, fallbacks and outputs; all instants, levels and lag truths rebuilt; a label-instants sensitivity | §1.3, §10.1, §16 (1) |
| R2 | critical | Every competitor got h_B, the deficiency MAREA's lag corrects, while the competitors use local levels in practice | Water-level factor LB/LT/LM/LG for elevation units; native sources in the as-published rows; family P3 (practical comparison); P2 scoped to "boundary tide only"; LG as a labelled upper bound and headroom | §1.6, §4.3, §5.3, §5.4, §9.2, §9.7 |
| R3 | major | State-dependent (MNAR) cloud masking on flats | Truth-free MNAR table and curves per cloud level; a logistic-slope covariate; C1d (dark area allowed) in OFAT | §2.2, §2.10, §9.4 |
| R4 | major | All cloud levels came from Sen2Cor | OmniCloudMask (C11) in Tier 1 and in G1' at all sites; CLP/CLM checked (not listed by CDSE on 2026-10-09; Tier 2 if exposed); the cross-tab extended to dry flats at low tide and turbid channel water | §2.2 |
| R5 | major | Collection-1 reprocessing made the date-based offset rules wrong; G-zarr might fail | Offset regime and Sen2Cor generation decided per scene; date rules withdrawn; G-offset; a pre-registered G-zarr response | §1.10, §1.11, §2.3, §10.1 |
| R6 | major | Index effects were confounded with 20 m resolution | Every index with a 20 m band computed at 20 m on 2 × 2 means; NDWI@20 in the Tier-1 OFAT and in a Campaign-2 G1' slice | §2.4, §5.6, §5.7 |
| R7 | major | The strata omitted substrate, vegetation, turbidity, sampled-range position and amplification | All added (truth-free or truth-only definitions), and used in conditional verdicts, GAMs and partitions | §7.1, §9.6 |
| R8 | major | PNOA LiDAR records the water surface at flight time | Flight time per strip, a Gijón-based level mask, constant-patch water detection, a coverage statement; descriptive if the flight time is unknown | §1.8, §17 (V18) |
| R9 | major | Vaklodingen were treated as exact 10 m truth | 20 m cells as the scoring support at the Dutch sites; source and density metadata; a direct-measurement sensitivity; RWS LiDAR at Escalda if available | §1.8, §7.1, §17 (V25) |
| R10 | major | Only home sites, and in-house rivals | Proposed away site (Tauranga, Campaign 2); author review of the cards; literature and in-house lag methods shown separately; scope of the conclusions stated | §0.8, §3.1, §5.14, §6.5 |
| R11 | moderate | OLS slope attenuation | b* = b/λ̂ (reliability-ratio correction). Deming regression was not adopted, because it needs each method's own error variance. | §7.2 |
| R12 | moderate | Otsu on near-single-class or multimodal scenes; unclear D2 domain | A fixed domain Ω with a ring of permanent water and land; a minority-share guard ≥ 0.15; corr(t_s, h_B) and the fallback rate reported | §2.6 |
| R13 | moderate | D5 was mis-cited to Murray et al. 2019 | Relabelled as an in-house majority vote (F4); Dynamic World as a Tier-2 learned detector | §2.6 |
| R14 | moderate | E6b had no modern optical Sentinel-2 basis | E6b based on Salameh et al. 2020; the classic fixed-threshold variant kept as E6b-fixed | §3.2, §3.2.1, §5.3 |
| R15 | moderate | The positive control looked tailored | Linear-mixing endmember world over sub-pixel relief, with 20 m bands at native resolution; misspecified worlds in Tier 1, not gating | §10.2 |
| R16 | moderate | Global 1/8° models were strawmen; there was no regional model | L16 (AMM15, IBI) promoted to Tier 1 and put in L-P1; unresolved flags | §3.3, §6.5 |
| R17 | moderate | The range filter hid amplified reaches | Per-band excluded counts; a local LG range sensitivity | §1.8, §7.6 |
| R18 | moderate | Sun/view geometry, orbit and season were ignored | Angles and orbit in the sidecar; lag per relative orbit; anchor index by orbit and season; a low-sun (SZA > 70°) sensitivity; a summer-only subset family in Tier 2 | §1.10, §2.4, §3.3.1, §6.4 |
| R19 | moderate | Non-stationarity in long windows | A temporal-split change map and the effective epoch | §2.7, §7.5 |
| R20 | minor | SCL 11 (snow/ice) kept by C4/C5 | Counted per site-year; a class-11-blocked sensitivity (Campaign 2) | §1.11, §2.2 |
| R21 | minor | EOT20-only boundary at Villaviciosa | Gijón residual and Cantabrian consensus sensitivities | §1.5 |
| R22 | minor | Gauge time-stamp conventions | Recorded, centred, ± half-window sensitivity | §1.7 |
| R23 | minor | Misregistration on channel banks | ±1 px shift sensitivity; a slopes < 2 % view | §7.6 |
| R24 | minor | D4 reads the same layer as C | Labelled a naive baseline, moved to Campaign 2, excluded from the headline robustness sets | §2.6, §9.7 |

The remaining "missing" item of review R, the declared S2-L2A-only limitation, is in §0.8.

### 18.3 Compute review (C)

| Id | Severity | Issue | Change | Where |
|---|---|---|---|---|
| C1 | critical | Revision 1 needed about 3 weeks against a 1-2 week envelope, and its cut list saved only 1.5-2 days | Campaign 1 ≈ 9.5 days (≈ 10.5-11 with contingency; ≈ 8 with 8+ workers); Campaign 2 a separate campaign; a mechanical cut order with a 10-day cap printed by `plan`; a one-week fallback; a calendar | §5.14, §14.2, §14.5 |
| C2 | critical | C0 made dom and S the whole frame (light check: C0 U_rec 100 %, keep_full 69 % at Escalda) | Fixed S0; C0 never in dom or in a grid, run once on S0; `pilot dom` refusal rule | §2.10 |
| C3 | high | dom and the range filter depended on the 2017-2022 data | dom frozen at lock-A on the 2023-2025 envelope; range filter from the 1-min h_B series; dom_version in the shard keys; outside-dom share measured | §1.8, §2.10, §13.2 |
| C4 | high | code_key contradicted the invalidation table | Per-registry-entry source hashes; a test that a dummy level changes no existing key | §2, §13.2, §10.3 |
| C5 | high | Stage T needed 1.2-1.5 quiet days | ≤ 10 h: Granadeiro's per-fit cost from RRunner logs; single-thread only on small workloads; 1 + 2 repeats; end-to-end at Escalda only | §8.2 |
| C6 | high | The 30-draw bootstrap (3569 s at the Ems) would have run in every cell | n_boot = 30 only in Stage L at P0, in config_key; GM20 refits likewise | §3.2.1, §3.3.1, §14.1 |
| C7 | high | Repeated package reads per method; seek-bound I/O | Instance-major scheduling; a fused block reader; pixel-major tiles; SSD if available; read_mb in the pilot | §5.1, §13.4 |
| C8 | high | Low CPU use per worker was unmeasured | cpu/wall in the pilot; 8-12 workers at BLAS 1 if < 1.5 | §8.2, §13.4, §14.1 |
| C9 | medium | Frame methods decoded whole 16-date chunks | A package-backed frame view (G-frameview); a 16-date block cache in SceneSource; Zarr read cost in the model | §1.2, §2.10, §10.1, §14.1 |
| C10 | medium | Waterline TIN memory (10-20 GB at W9) | Per-cell aggregation before triangulation (declared); a pilot under a memory limit | §3.2.1, §8.2, §14.3 |
| C11 | medium | RAM polling race, OOM victims, stage-level resume, hidden hangs | RAM ledger; oom_score_adj; MemoryMax scopes; job-level checkpoints; wall cap; progress-based hang detection | §13.5 |
| C12 | medium | Pass-1 memory and time; D2's per-window dependency; dask using every core | ≤ 8 GB per site and never beside Granadeiro; bounding-box restriction; one settings pass with per-year histograms; a fixed D2 domain; synchronous dask; uint8 counters | §2.6, §2.10 |
| C13 | medium | Scoring cost and storage | Per-block sufficient statistics; quantile CIs only at P0 and in the families; incremental scoring; corrected product counts | §13.6, §5.15 |
| C14 | medium | L06 would arrive days after Stage L; Wadden and Ferrol were missing | G-base right after Stage B; L-P2 a separate Stage-G family | §5.11, §6.5 |
| C15 | medium | The Bijagós gate and G-granadeiro at all sites cost hours | A pre-registered sub-region; G-granadeiro at Villaviciosa only | §10.1 |
| C16 | medium | The BOA offset rule was not frozen | A per-scene rule frozen before lock-A; `inputs` fails on 'unclear'; G-offset | §1.10, §2.3 |
| C17 | medium | About 16 manual launches at 100 % assumed utilisation | A multi-plan priority queue; 75-80 % utilisation; late methods join as new units | §13.7, §14.1 |
| C18 | low | Disk: counters and products not budgeted | uint8 counters; a recomputed budget; a retention policy | §14.4 |
| C19 | low | No per-instance constant; Ferrol ran every design | The constant in the cost model; Ferrol elevation at P0 only | §14.1, §5.13 |

The "missing" list of review C maps as follows:
- server storage facts: V9;
- the dom-size pilot and the pixels each method computes on: §2.10;
- the package contract: §13.4;
- per-unit measurements: §8.2;
- wall caps and hangs: §13.5;
- the scoring design: §13.6;
- Campaign 2 gated on the 2017-2022 arrival: §5.14 and V24;
- Granadeiro SD-mask sizes: §2.10;
- the calendar: §14.5;
- retention: §14.4.

### 18.4 Adopted differently, or not adopted (with reasons)

1. **C0** stays as the user's "no filtering" level, but only as one OFAT lower bound evaluated on S0. Review C proposed dropping it, and review S proposed replacing it in G1. The user's plan names it, and on S0 it no longer inflates any cost.
2. **G1' has four cloud levels** (C1, C2, C4, C11), not three. Review S asked for a competitor-native level and review R for OmniCloudMask, so both were added. The cost is offset by dropping D2 from the G1' detection levels (review C's cut). D2 remains an OFAT cell and LogiTide's native cell.
3. **The intersection-union over views keeps FWER control over the view-specific statistics** (128 in P2), rather than shrinking P2 to 64 tests as review S suggested. The two views measure different parameters (different pixel sets). Controlling each view's family separately would allow up to 2α across contrasts. Romano-Wolf over both views keeps FWER at α, and with B = 20,000 the family stays attainable.
4. **Deming regression is not used** (review R). It needs each method's own error variance. The reliability-ratio correction b/λ̂ needs only the method-free truth-error variance.
5. **MAREA runs under LB only** (review R proposed LB/LM/LG for every method). Its lag is a correction relative to the boundary. MAREA with a lag estimated relative to LM is Tier 2. No delay runs under every source and provides the headroom.
6. **The 20 m cells are the scoring support** of every view at the Dutch sites, not an extra co-primary view (review R). This keeps the intersection-union at two views.
7. **The per-location lag TOST is dropped** rather than given a method-side variance (review S offered both options). Most methods have no method-side interval.
8. **AWEI_nsh, WRI and D4 move to Campaign 2, not Tier 2** (review C), because the user asked for "other indices if useful" and detection alternatives.
9. **G-nir runs at Villaviciosa and Escalda in Campaign 1, and at Wadden and Ems in Campaign 2** (review C proposed two sites only). G-nir is Granadeiro's native cell, which the headline rule needs.
10. **The away site is recommended for Campaign 2, not Campaign 1** (review R). It does not fit the two-week envelope. It remains the user's decision (V27).
11. **The development-free judges are a sensitivity, not the primary set** (review S). Dropping Harlingen from the primary set would remove one of only two Wadden locations.
12. **HPE's margin is computed per cell**, which is the per-pixel form review S offered, not a per-site derived constant.

---

## Appendix A. FROZEN sections (new or changed relative to v13 Appendix A)

| Section | Value |
|---|---|
| record | Zarr dates in the window with a valid G-time instant and a finite h_B; global positions over UNION_PERIOD 2017-01-01..2026-01-01 (end-exclusive) |
| sensing_time | Method A (SPA inversion of the sun angles at the site centroid and frame corners, search [label − 5, label + 30] min); method B (SGP4 with Space-Track TLEs, closest approach); criteria \|A − B\| ≤ 0.5 min on ≥ 10 dates per group, within-group SD ≤ 0.5 min, physical window, instant ≥ label − 0.5 min; fallbacks of §1.3 |
| package | Year shards keyed by dom_version; raw int16 B02, B03, B04, B08, B11, B12 and uint8 SCL, CLD at dom; S0 = D_truth ∪ NB ∪ dilate(keep_full(P0, W3), 300 m) ∪ Ω; dom = S0 ∪ dilate(U_rec over {P0, C2, C11 or C3} 2023-2025, 3 px); refusal if \|dom\| > 1.5 \|S0\|; U_rec = {clr ≥ 31, wet ≥ 4, dry ≥ 4} per setting; uint8 counters; frame bits for curve settings; OCM bits; pixel-major tiles of 4,096 px; settings pass from per-year histograms |
| cloud | C0-C8, C1d and C11 as in §2.2; p from the extended cross-tab rule; opening disk 2 px; radii 8 px (cloud) and 4 px (shadow); TZ 10 %, clean frame 5 %, stability 0.95, buffer 10 px; g21 < 0.10 and ≤ 0.02; ard ≥ 0.5; domain ≥ 0.5; C11 version, bands and resolution from its card; matched-N draws 3 (frame methods 1), seed 20261016 |
| prep | floor0 (base), raw; per-scene offset o_s ∈ {0, 1000} from the frozen rule |
| index | Formulas and orientations of §2.4; the 20 m rule on `lattice20` (box mean at phase (0, 0) if None); t_k table; t_NIR and s_k on keep_full(P0, W3) per site from per-year histograms |
| detect | D1-D5 of §2.6; D2 on Ω, ≥ 1,000 px, η ≥ 0.75, minority ≥ 0.15, 256 bins; D3 Canny σ 1, 0.05, edges ≥ 50 px, buffer 5 px, ≥ 500 px, valid ≥ 0.2, bounding box of dom + 1 km |
| level | LB = h_B; LT = eo-tides (EOT20; FES2022 once installed; locally ranked ensemble with ≥ 3 models; default ranking or BT26-rank), 1 km nodes; LM = AMM15/IBI SSH (15-min preferred, hourly with cubic interpolation), 1-km s bins of wet channel cells, demeaned against h_B, unresolved if the nearest wet cell is > 1 km away; LG = linear in s between the boundary and the judges, constant beyond the last gauge |
| window | W1, W2, W3, W5, W9 (end 2026-01-01); Y2017..Y2025; survey-centred windows of §2.7; epoch at mid-year; temporal split; effective epoch = observation-weighted centre |
| subsets | `default_rng([20261008, SITE_ID, WINDOW_ID, r])`; disjoint iff R ≤ floor(T/N), nested otherwise, never mixed in one point; `win` start `default_rng([20261012, SITE_ID, WINDOW_ID, N, r])` |
| curves | Grids and reps of §2.8; k/R ≥ 0.5; joint-computable accuracy; A-mode primary; the fixed curve view of §7.4 |
| design | Design tables `plans/*.csv` (sha); L9 rows of §5.11; tiers; native cells; the cut order and the 10-day cap |
| levels | Provider over UNION_PERIOD built once after all IOC chunks; integer τ ∈ [−90, max(240, 180·(K − 1))]; 1-min series period ± 2 d; 10-min every 10th; marks_lp 61-min mean |
| methods.* | One section per method (CONFIG of its card), including MAREA (m2.yaml, TAU_BOUNDS −45..180, monotone, seed 20260819 + 31, 3 km, bracket 3, floor 40, n_boot 30 in Stage L at P0 only, else 0), NIDEM (polygons per level source), DEA (2.5.3 parameters), Granadeiro (v13 values), LogiTide (separation rule), frequency, waterline (vertex rules, per-cell aggregation), Coast X-Ray, Bué (a = −8, sat 0.3), Salameh configuration of E6b, lag methods (common window −45..180 step 5, the per-method QC of §3.3.1, GM20 refits in Stage L only); as-published settings, native cells and tuning grids |
| tuning | The grids of §5.12; objective CAE (F_const) at Escalda P0; tuned rows at the other elevation sites |
| truth_elev | Vaklodingen all epochs per cell; survey-matched default (nearest the window centre, ties to the later survey); 20 m cell support at the Dutch sites (≥ 2 of 4 pixels mapped); PNOA ground-only with the flight-level mask (+0.15 m) and constant-patch water (≥ 50 px within 1 mm); RTK via `load_rtk()` defaults; range filter [q_0.001(h_B) − 0.25, q_0.999(h_B) + 0.25] over the 2017-2025 1-min series; stable = ≥ 2 surveys within window ± 1 year and range ≤ 0.20 m; strata of §7.1 with their class limits |
| truth_error | σ²_s (Escalda intercept of D(Δt), conservative of the isotonic and power-law fits; Wadden and Ems from same-year overlaps, else Escalda; PNOA stated accuracy), D_W(Y) (from Escalda, noise counted once; transfer by the D(Δt) ratio), λ̂ = 1 − σ²_T/Var(t); sensitivity × {0.5, 1.5} |
| truth_lag | Per window: (a), τ_ab, (b), HW/LW, τ_all, crossing deciles, amplitude ratio; centred gauge series; G-time instants; 200 month-block bootstraps (seed 20261011) |
| judges | v13 rules, coverage per window; headline set of §6.1; the development-judge list |
| metrics_v2 | ΔMSE_c, δ_m, b*, D_b with the non-regular rule; the §7.2 panel; HPE (h_ref = LG series at judged sites, else h_B; per-cell margin); HFI (1000 quantiles; joint verdicts); CAE (F_const and F_freq, per-part median centring; agreement rule); split-half by alternate scenes; curve fit E² = s² + v/n with s_method; TTT targets 0.20 m, 0.30 m and 1.1 × best |
| analysis_v2 | Families P1 (16), P2 (128), P3 (98), L-P1 (5 per truth), L-P2 (1 per truth); Romano-Wolf step-down with intersection-union over views; Holm cross-check; BY for S; B 20,000/1,000 with the attainability assertion; block size from map-difference variograms on the lattice {0.25, ..., 16} km; ≥ 25 blocks; band-level test (2-band moving blocks, ≥ 6 bands); verdict order and margins of §9.3; the headline rule of §9.7; a fixed regret comparator set; meta-analysis over the Vaklodingen sites (REML + HKSJ); the LOSO rung with internal parameters |
| positive_control | The linear-mixing gating world and the misspecified worlds of §10.2; tolerances (v13 D22 amended) and per-method expected offsets; the fault rule; seed 20261013 |
| threads | Per method: BLAS and process counts; the worker-count rule of §13.4; Granadeiro BLAS 1 + 16 R workers |
| timing | Standard workloads of §8.2; 1 warm-up + 2 repeats; single-thread up to 10⁵ px; production configuration on all |

---

## Appendix B. References

Methods and data:
- Bell PS, Bird CO, Plater AJ (2016) Coastal Engineering 107:84-101, doi:10.1016/j.coastaleng.2015.09.009
- Bishop-Taylor R, Sagar S, Lymburner L, Beaman RJ (2019) ECSS 223:115-128, doi:10.1016/j.ecss.2019.03.006 (NIDEM); code github.com/GeoscienceAustralia/nidem
- Bishop-Taylor R et al. (2019) Remote Sensing 11:2984, doi:10.3390/rs11242984 (sub-pixel waterlines)
- Bishop-Taylor R, Phillips C, Sagar S, Newey V (2026) RSE 347:115663, doi:10.1016/j.rse.2026.115663; dataset doi:10.26186/149403; code github.com/GeoscienceAustralia/dea-intertidal (Apache-2.0); product page https://knowledge.dea.ga.gov.au/data/product/dea-intertidal
- Bishop-Taylor R, Sagar S, Phillips C, Newey V (2026) IJRS 47(12):5268-5287, doi:10.1080/01431161.2026.2666912
- Bishop-Taylor R et al. (2025) eo-tides, JOSS, doi:10.21105/joss.07786
- Bué I, Catalão J, Semedo Á (2020) Remote Sensing 12(8):1311, doi:10.3390/rs12081311
- Cai H, Savenije HHG, Toffolon M (2012) JGR 117:C09023, doi:10.1029/2012JC008000
- Chen Y-C, Wang S-F (2025) JAG 139:104561, doi:10.1016/j.jag.2025.104561
- Copernicus Marine Service: NWSHELF_ANALYSISFORECAST_PHY_004_013 (AMM15) and IBI_ANALYSISFORECAST_PHY_005_001, https://data.marine.copernicus.eu (dataset ids and archive depth checked by `fetch cmems --dry-run`)
- Costa WLL, Bryan KR, Coco G (2023) NHESS 23:3125-3146, doi:10.5194/nhess-23-3125-2023; code doi:10.5281/zenodo.8357351
- Fassoni-Andrade AC, Paiva RCD, Fleischmann AS (2020) WRR 56:e2019WR026362, doi:10.1029/2019WR026362; Fassoni-Andrade et al. (2021) RSE 251:112065, doi:10.1016/j.rse.2020.112065; code doi:10.17632/jwxxjc5kyf.1
- Fitton JM, Rennie AF, Hansom JD, Muir FME (2021) RSASE 22:100499, doi:10.1016/j.rsase.2021.100499
- Geyman EC, Maloof AC (2020) Earth and Space Science 7:e2019EA000958, doi:10.1029/2019EA000958; data doi:10.1594/PANGAEA.911255
- Granadeiro JP, Belo J, Henriques M, Catalão J, Catry T (2021) Remote Sensing 13(2):320, doi:10.3390/rs13020320; data doi:10.5281/zenodo.4662640
- Hart-Davis MG, Monahan T, Vos K, Andersen OB (2026) Commun. Earth Environ., doi:10.1038/s43247-026-03943-9; preprint doi:10.21203/rs.3.rs-8967674/v1
- Hart-Davis MG et al. (2021) EOT20, ESSD 13:3869-3884, doi:10.5194/essd-13-3869-2021
- Khan MJU et al. (2019) Remote Sensing 11(24):2888, doi:10.3390/rs11242888; code github.com/jamal919/pyIntertidalDEM (Apache-2.0)
- Liu Y, Li M, Zhou M, Yang K, Mao L (2013) Remote Sensing 5(11):6138-6158, doi:10.3390/rs5116138
- Mason DC et al. (1995) GRL 22(23):3187-3190, doi:10.1029/95GL03168; Mason DC, Scott TR, Dance SL (2010) ECSS 87(3):487-496, doi:10.1016/j.ecss.2010.01.015
- Pool L et al. (2025) Environmental Modelling & Software 193:106600, doi:10.1016/j.envsoft.2025.106600
- Ryu J-H, Won J-S, Min KD (2002) RSE 83:442-456, doi:10.1016/S0034-4257(02)00059-7
- Sagar S, Roberts D, Bala B, Lymburner L (2017) RSE 195:153-169, doi:10.1016/j.rse.2017.04.009
- Sagar S, Phillips C, Bala B, Roberts D, Lymburner L (2018) Remote Sensing 10(3):480, doi:10.3390/rs10030480 (tidal sampling bias of sun-synchronous sensors)
- Salameh E, Frappart F, Turki I, Laignel B (2020) ISPRS J. Photogramm. Remote Sens. 163:98-120, doi:10.1016/j.isprsjprs.2020.03.003
- Savenije HHG, Toffolon M, Haas J, Veling EJM (2008) JGR 113:C10025, doi:10.1029/2007JC004408
- Tseng K-H et al. (2017) ISPRS J. 131:92-103, doi:10.1016/j.isprsjprs.2017.07.008
- Vos K et al. (2019) CoastSat, Environmental Modelling & Software 122:104528, doi:10.1016/j.envsoft.2019.104528

Sentinel-2 data, timing and geometry:
- Copernicus Data Space Ecosystem, Sentinel-2 L2A processing baselines (Collection-1), https://documentation.dataspace.copernicus.eu/Data/Others/Sentinel2_L2A_baseline.html (read 2026-10-09)
- Copernicus Data Space Ecosystem, Sentinel-2 L2A bands including sunZenithAngles and sunAzimuthAngles, https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html (read 2026-10-09)
- Sentinel-2 product naming (datatake sensing start), https://sentiwiki.copernicus.eu/web/s2-products
- Reda I, Andreas A (2004) Solar position algorithm for solar radiation applications, Solar Energy 76:577-589, doi:10.1016/j.solener.2003.12.003
- Rhodes B, Skyfield, https://rhodesmill.org/skyfield/ (SGP4 propagation); historical TLEs from https://www.space-track.org

Indices, masks and detection:
- Baetens L, Desjardins C, Hagolle O (2019) Remote Sensing 11:433, doi:10.3390/rs11040433
- Brown CF et al. (2022) Dynamic World, Scientific Data 9:251, doi:10.1038/s41597-022-01307-4
- Cao H et al. (2019) Water 11:786, doi:10.3390/w11040786 (Bmax Otsu)
- Donchyts G et al. (2016) Remote Sensing 8(5):386, doi:10.3390/rs8050386; Markert KN et al. (2020) Remote Sensing 12:2469, doi:10.3390/rs12152469
- Du Y et al. (2016) Remote Sensing 8(4):354, doi:10.3390/rs8040354
- Feyisa GL et al. (2014) RSE 140:23-35, doi:10.1016/j.rse.2013.08.029
- Fisher A, Flood N, Danaher T (2016) RSE 175:167-182, doi:10.1016/j.rse.2015.12.055
- McFeeters SK (1996) IJRS 17:1425-1432, doi:10.1080/01431169608948714
- Otsu N (1979) IEEE TSMC 9:62-66, doi:10.1109/TSMC.1979.4310076
- Sen2Cor L2A ATBD v2.10, https://step.esa.int/thirdparties/sen2cor/2.10.0/docs/S2-PDGS-MPC-L2A-ATBD-V2.10.0.pdf
- Shen L, Li C (2010) doi:10.1109/GEOINFORMATICS.2010.5567762
- Skakun S et al. (2022) Cloud Mask Intercomparison eXercise (CMIX), RSE 274:112990, doi:10.1016/j.rse.2022.112990
- Wright N et al. (2025) OmniCloudMask, RSE, doi:10.1016/j.rse.2025.114694
- Xu H (2006) IJRS 27:3025-3033, doi:10.1080/01431160600589179

Design, metrics and statistics:
- Benjamini Y, Yekutieli D (2001) Ann. Stat. 29:1165-1188, doi:10.1214/aos/1013699998
- Berger RL (1982) Multiparameter hypothesis testing and acceptance sampling, Technometrics 24:295-300, doi:10.1080/00401706.1982.10487759
- Chow CK (1970) IEEE Trans. Inf. Theory 16:41-46, doi:10.1109/TIT.1970.1054406
- Corani G et al. (2017) Mach. Learn. 106:1817-1837, doi:10.1007/s10994-017-5641-9
- Dehghani M et al. (2022) The Efficiency Misnomer, arXiv:2110.12894
- Diebold FX, Mariano RS (1995) Comparing predictive accuracy, J. Bus. Econ. Stat. 13:253-263, doi:10.1080/07350015.1995.10524599
- Figueroa RL et al. (2012) BMC Med. Inform. Decis. Mak. 12:8, doi:10.1186/1472-6947-12-8
- Fuller WA (1987) Measurement Error Models, Wiley, doi:10.1002/9780470316665
- Hering AS, Genton MG (2011) Technometrics 53:414-425, doi:10.1198/TECH.2011.10136
- Hoefler T, Belli R (2015) SC15, doi:10.1145/2807591.2807644
- Holm S (1979) Scand. J. Stat. 6:65-70, https://www.jstor.org/stable/4615733
- IntHout J, Ioannidis JPA, Borm GF (2014) BMC Med. Res. Methodol. 14:25, doi:10.1186/1471-2288-14-25
- Lenth RV (1989) Technometrics 31:469-473, doi:10.1080/00401706.1989.10488595
- Lin LI (1989) Biometrics 45:255-268, doi:10.2307/2532051
- Qian PZG (2012) JASA 107:393-399, doi:10.1080/01621459.2011.644132
- Roberts DR et al. (2017) Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure, Ecography 40:913-929, doi:10.1111/ecog.02881
- Romano JP, Wolf M (2005) Stepwise multiple testing as formalized data snooping, Econometrica 73:1237-1282, doi:10.1111/j.1468-0262.2005.00615.x
- Saltelli A et al. (2008) Global Sensitivity Analysis: The Primer, doi:10.1002/9780470725184
- Schuirmann DJ (1987) J. Pharmacokinet. Biopharm. 15:657-680, doi:10.1007/BF01068419
- Shrout PE, Fleiss JL (1979) Psychol. Bull. 86:420-428, doi:10.1037/0033-2909.86.2.420
- Taylor KE (2001) JGR 106:7183-7192, doi:10.1029/2000JD900719
- Vallender SS (1974) Theory Probab. Appl. 18:784-786, doi:10.1137/1118101
- Willmott CJ, Robeson SM, Matsuura K (2012) Int. J. Climatol. 32:2088-2094, doi:10.1002/joc.2419
