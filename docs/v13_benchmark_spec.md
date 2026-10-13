# v13 — Controlled benchmark of intertidal lag and elevation methods: design specification (revision 2)

> **AUTHOR DECISIONS (final, 2026-10-08) — these override anything below.**
> - **D1 (cloud mask): PRIMARY = `marea_clear`** (clear iff SCL in {4, 5, 6, 7}, MAREA's production per-pixel mask, `marea.CLEAR`); `blocklist` (SCL 0, 3, 8, 9, 10 and no data masked) is the SENSITIVITY. The authors' words were "con nuestra máscara de nubes" and they were told this means MAREA's production mask. Consequences: `obs.MASK_RULE = "marea_clear"`; the package domain U_rec is still computed under `blocklist` + floor0, which is a superset (every count under `marea_clear` is ≤ the count under `blocklist`), so the same package serves both rules; every place that says "blocklist (primary)" reads "marea_clear (primary)"; the Tier-1 sensitivity `--sens mask=marea_clear` becomes `--sens mask=blocklist`.
> - **D6 / Granadeiro scope = option (a)**: controlled N = all at all five sites; A-mode curves at Escalda and Villaviciosa (N ∈ {20, 45, 90, 180, 330} × 3 reps); run after the cheap groups.
> - **All other decisions D2–D23: the recommended defaults of §11** (floor0 primary with raw as sensitivity; zero-lag fallback for every lag method; MAREA floor 40; N grid with 180/240/330; exact h_B and the common −45..+180 window for Granadeiro's controlled row with `eq1_90` and `calon` as declared variants; scorer read-outs; three co-primary lag truths; method-independent judge rules; variogram blocks; Holm primary family; etc.). Tier 2 variants are implemented (code paths and CLI) but scheduled later.
> - Implementation is delivered in milestones: **M1** foundation (`config`, `obs`, `levels`, `package`, `domain`, `judges`, `truth`, `product`, runner core, CLI `check-env`/`prepare`/`truth`/`status`) so that the user can start `prepare`/`truth` on the server; **M2** cheap methods + scoring + gates + positive control + `lock`/`run`/`score`/`report`; **M3** Granadeiro.

Repository `C:\Users\Jorge\sketch_fitton\Intertidal_analysis`, branch `validacion-10m`. Paths are relative to the repository root. Local Python: `C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe`; server Python: `$PY` after `source env.sh` (project at `/media/HDD/jalias/Intertidal_analysis`; the user runs every server command). Written 2026-10-08 in a read-only phase and revised the same day after two adversarial reviews (fairness, feasibility). Nothing in the repository was modified. §13 maps every review issue to its resolution.

---

## 0. Scope

### 0.1 Goal

Every method receives the same initial conditions, and only the estimator changes:

- the same scenes;
- the same per-pixel cloud mask;
- the same band preprocessing, from which each method computes its own index;
- the same boundary tide series, given to every method as exact levels;
- the same date subsets, under one definition of "N images";
- the same lag search window wherever a method searches a window.

Everything around the estimators is defined once, in the harness, without reference to any method: the scoring domain, the lag read-outs, the failure policy, the truths and the inference. One scorer then scores every product under identical conditions, at N = all scenes and along learning curves in N.

### 0.2 Decisions in force

The user confirmed these on 2026-10-08 ("escenas comunes, con nuestra máscara de nubes, índice espectral propio"):

- **C1. Common scenes.** Every cube scene of 2023-01-01..2025-12-31 with an overpass time and a finite boundary level enters, identical for every method. There is no scene-level cloud rule in the controlled rows.
- **C2. Our per-pixel cloud mask**, applied identically to every method: SCL 3, 8, 9 and 10 (`pyintertidal.water.BAD_CLASSES`) and no data (SCL 0, fill) are masked per pixel (§2.3).
- **C3. Each method's own index, from the same L2A bands and the same band preprocessing**: NDWI for no delay, MAREA, NIDEM, GM20 and HD26; NIR (B08) for Granadeiro et al. (2021), as in their paper.

Constraints fixed by the brief:

- MAREA's lag model by 3-km bands is fixed.
- One row per method; methods are never combined.
- Only 10 m numbers are reported; 20 m is for local testing.
- Existing results files, `products_*` folders and notebooks are never modified.
- Code is in English, in the repository's style. Commits are made later by the authors, without assistant attribution.
- The RTK survey is read only through `pyintertidal.rtk.load_rtk()` with default arguments; the reserved split stays sealed.

Recommended defaults that still need author confirmation are marked **[Dk]** and listed in §11.

### 0.3 Controlled rows

| Row | Produces | Lag | Elevation | Code reused |
|---|---|---|---|---|
| no delay | elevation | 0 | MAREA inversion at τ = 0 | `pyintertidal/marea.py` |
| MAREA | elevation + lag | m2a band lags (3-km bands), smoothed per pixel | MAREA inversion at τ_px | `marea.py`, `tide_estimators.py` |
| NIDEM | elevation | none (by design) | ITEM interval composites → contours → TIN | `waterline_contour.py`, `experiments/v7_waterline_contour.py` |
| Granadeiro | elevation + lag | rising/ebbing 4PL lag + GAM | 4PL inflection at the GAM lag | `granadeiro2021.py`, `r/granadeiro_fit.R`, `experiments/v8_granadeiro2021.py` |
| GM20 (Geyman & Maloof 2020) | lag only | per-pixel 12.42-h sinusoid phase | – | new |
| HD26 (Hart-Davis et al. 2026) | lag only | per-band M2 harmonic phase | – | new |
| zero lag | lag baseline | 0 | – | – |
| EOT20 / GOT5.6 at the gauge | lag baselines (lower bounds) | global model at the gauge vs h_B | – | `boundary.PyTMDBoundary` |

Variants, sensitivities and reference rows ("as published", scene-rule matrix, tuning budget) are listed in §3.9. They are never mixed into the controlled rows.

### 0.4 What is not edited

No estimator code changes:

- `pyintertidal/{marea, tide_estimators, elevation, waterline_contour, granadeiro2021, boundary, rtk, geometry}.py`
- `pyintertidal/r/granadeiro_fit.R`
- `experiments/v7_waterline_contour.py`, `experiments/v8_granadeiro2021.py`

The harness imports their functions. Where a driver (v7, v8) hard-wires a scene rule, a pixel mask or an output folder, the harness re-sequences the driver's calls inside its own adapter and proves equivalence with a gate (§4.1). Importing v7 or v8 runs `os.chdir(ROOT)` and `sys.path.insert`; this is harmless because every command runs from the repository root.

### 0.5 Two meanings of "N images" (one definition per curve, the same for every method)

| | **A-mode, "the archive has N images"** (primary) | **F-mode, "N images chosen from the full archive"** (diagnostic) |
|---|---|---|
| Question | how many images does a user need | how the estimator degrades when the domain is known |
| Domain and geometry steps | recomputed on the N scenes with each method's own rule | taken from the full record for every method |
| MAREA, no delay | `keep_sel`, `sea_sel`, `s_km`, bands from `marea.extract`'s rule (> 30 clear, 0.10 < wf < 0.90) and the mouth seeds on the subset; with fewer than 3,000 intertidal pixels the run stops (`reconstruct`'s rule); with ≤ 5,000 finite s, one band and τ = 0 (`reconstruct`'s no-anchor path) | full-record `keep_full`, `sea_full`, `s_km`, bands |
| GM20, HD26 | the same `keep_sel` and subset bands | `keep_full` and full bands |
| NIDEM | no domain step: identical in both modes (one run serves both) | same |
| Granadeiro | SD(NDWI) mask, [calibration reference], preliminary DEM, pool, all on the subset | SD mask, converged set and pool from the N = all controlled run; lag fits, GAM and final fits on the subset |
| Everything | the estimator on the subset | the estimator on the subset |

In both modes the scoring domain, the band strata, the judges' neighbourhoods and the truths stay fixed. At N = all the two modes coincide. Every curve is labelled with its mode.

Expected consequence (light check, §2.5): about 30 % of observations are clear, so `marea.extract`'s "more than 30 clear observations" rule needs N ≳ 100. In A-mode the MAREA family, GM20 and HD26 are expected to be NOT COMPUTABLE at N ≤ 60 and partly computable at N = 90-120, depending on how many clear scenes a subset holds. This is the honest answer of MAREA's own pipeline to "how many images", and the F-mode curve shows the estimator alone.

### 0.6 Validation ladder and printing rule

1. Unit tests (§8), synthetic, local.
2. Gates (§4.1): the adapters reproduce the production code on production inputs.
3. Positive control (§4.2): a synthetic cube with a planted DEM and lag, run through every adapter and the same scorer, with tolerances fixed in FROZEN before any real-data score.
4. Granadeiro projection and real-data smoke (§4.3).
5. Real runs and scoring.

A method's controlled row is printed only when its gates and its positive control have passed. Otherwise the table row reads "implementation not validated" and carries no numbers.

The analysis plan is frozen before the first real-data score (§2.11, "pre-registration lock").

---

## 1. Module layout and public interfaces

```
experiments/v13_benchmark.py            CLI (the only module that chdirs to ROOT)
experiments/benchmark/__init__.py
experiments/benchmark/config.py         sites, constants, seeds, FROZEN (sectioned), paths
experiments/benchmark/obs.py            THE mask, band-preprocessing and index functions
experiments/benchmark/levels.py         provider, exact level table, 1-min/10-min series, marks (raw and low-pass)
experiments/benchmark/package.py        common input package: two-pass build, load, manifest
experiments/benchmark/domain.py         full-record domain and the A-mode subset domain (extraction rule, geometry, bands)
experiments/benchmark/judges.py         in-estuary gauges: series, geometry, neighbourhoods, inclusion rules
experiments/benchmark/truth.py          frame truth, scoring domains, 20 m cells; lag truths (a), tau_ab, (b), (c); global-model baselines
experiments/benchmark/product.py        Unit, Product, statuses, atomic IO, keys
experiments/benchmark/readout.py        lag read-outs on applied lag fields (called by the scorer only), circular statistics
experiments/benchmark/inference.py      variogram block size, block bootstrap, Holm, margins, sign tests, rep-spread check
experiments/benchmark/scoring.py        metrics, views, strata, support-matched cells, accuracy-coverage, lag scores, curves
experiments/benchmark/control.py        positive control: synthetic cube, planted truth, verdict
experiments/benchmark/runner.py         unit queue, claims, resume, logging, resources, gates/control check
experiments/benchmark/report.py         CSV, LaTeX rows (v10 style), numbers.json, figures
experiments/benchmark/adapters/__init__.py   REGISTRY
experiments/benchmark/adapters/marea_pair.py no delay + MAREA (one unit, two products)
experiments/benchmark/adapters/nidem.py
experiments/benchmark/adapters/granadeiro.py
experiments/benchmark/adapters/geyman.py     GM20
experiments/benchmark/adapters/hartdavis.py  HD26
experiments/benchmark/adapters/baselines.py  zero lag, EOT20 and GOT5.6 at the gauge
experiments/benchmark/adapters/published.py  as-published reference rows (read-only loaders)
tests/test_benchmark_{obs,package,domain,marea,nidem,granadeiro,lagmethods,truth,scoring,inference,control,runner}.py
```

**Style** (as v7, v8, v9, v12): a module docstring that states the question and the pre-registered rules; `log()` prints with `flush=True`; numpy-style docstrings; no import-time side effects; comments say why, not what; only the CLI does `os.chdir(ROOT)`; names follow the repository (`keep`, `sel`, `h0`, `tau_px`, `band_of`, `centers`, `s_km`, `z_keep`).

### 1.1 `config.py`

```python
SITES = ("escalda", "wadden", "ems", "ferrol", "villaviciosa")       # + "pc" (positive control, §4.2)
SITE_ID = {"escalda": 1, "wadden": 2, "ems": 3, "ferrol": 4, "villaviciosa": 5, "pc": 9}
PERIOD = ("2023-01-01", "2025-12-31")
RES_CHOICES = (10, 20)            # 20 only for escalda/wadden/ems/pc, local tests

@dataclass(frozen=True)
class SiteConfig:
    site: str; res_m: int; pixel_m: float
    cube: str; extract: str; overpass: str
    boundary: dict        # {"kind": "make_boundary", "tide_model": "EOT20", "n_gauges": 2, "exclude": ("trnz",)}
                          # | {"kind": "pytmd", "model": "EOT20", "at": "aoi.centroid"}; pc uses the Wadden spec
    mouth_side: str
    elevation_truth: tuple   # ("vaklodingen",) | ("lidar", "rtk") | ("planted",) | ()
    judges: tuple            # station codes in data_v4/gauges/estuary_truth/stations.json
    published: dict          # as-published folders (may not exist)
    transition_result: str   # result.json holding the 10 m transition-zone record_dates

def site_config(site: str, res_m: int) -> SiteConfig
def out_root(res_m: int) -> str                  # "results/v13_benchmark_{res_m}m"
def common_dir(site: str, res_m: int) -> str     # out_root/common/<site>
N_GRID = (10, 20, 30, 45, 60, 90, 120, 180, 240, 330)       # plus "all" [D5]
GRAN_N_GRID = (20, 45, 90, 180, 330)                       # [D6]
REPS = {"rand_A": 10, "rand_F": 5, "win_A": 5, "floor": 5, "nidem": 10, "gran_A": 3, "gran_F": 3}   # [D6]
GRAN_CURVE_SITES = ("escalda", "villaviciosa")             # [D6]
SEEDS = dict(subsets=20261008, block_boot=20261009, gm20=20261010, truth_boot=20261011,
             windows=20261012, pc=20261013, variogram=20261014,
             m2=20260819, granadeiro=20210119)      # m2 and granadeiro are the production seeds
FROZEN = {...}                                      # Appendix A, in sections; each section has its own sha256
def frozen_sha(*sections) -> str                    # sha256 of json.dumps({s: FROZEN[s]}, sort_keys=True)
```

### 1.2 `obs.py` (the shared observation functions)

```python
BLOCK_SCL = (0, 3, 8, 9, 10)       # == (0, *pyintertidal.water.BAD_CLASSES), asserted at import
CLEAR_SCL = (4, 5, 6, 7)           # == pyintertidal.marea.CLEAR, asserted at import
SCL_NODATA_U8 = 255                # fill / NaN SCL in the uint8 package arrays
MASK_RULES = ("blocklist", "marea_clear")
INDEX_MODES = ("floor0", "raw")
MASK_RULE = "marea_clear"          # C2 (authors, final): production MAREA mask; "blocklist" = sensitivity
INDEX_MODE = "floor0"              # [D2]; "raw" = production behaviour, sensitivity

def decode(raw, fill) -> np.ndarray                     # int16 -> float32, fill -> NaN (== xarray default decoding)
def scl_u8(scl_raw, fill) -> np.ndarray[uint8]          # fill/NaN -> 255
def mask_valid(scl_u8, rule=MASK_RULE) -> np.ndarray[bool]
def bands(b03_raw, b08_raw, fill, mode=INDEX_MODE) -> tuple[np.ndarray, np.ndarray]   # float32 g, n; floor0: np.maximum(x, 0), NaN kept
def ndwi(g, n) -> np.ndarray[float32]                   # np.where(g + n != 0, (g - n)/(g + n), nan)  (== marea.extract)
def observe(b03_raw, b08_raw, scl_u8, fill, rule=MASK_RULE, mode=INDEX_MODE) -> tuple   # (ndwi, V), V = mask & isfinite(ndwi)
def wet(ndwi, V) -> np.ndarray[bool]                    # V & (ndwi > 0)
def floor_raw(raw, fill) -> np.ndarray[int16]           # np.where(raw == fill, fill, np.maximum(raw, 0)): Granadeiro's int16 path
def nir(b08_raw, fill, mode=INDEX_MODE) -> np.ndarray[float64]   # g21.to_reflectance(floor_raw(...) if floor0 else raw)
def mask_raw_bands(g_raw, n_raw, V, fill) -> int        # in place: G, N := fill where not V; returns the count
def item_valid(scl_u8) -> np.ndarray[bool]              # == wc.pixel_valid(decoded scl, rule="item"); gates only
def source_sha256() -> str
```

### 1.3 `levels.py`

```python
def make_provider(cfg: SiteConfig)       # make_boundary(...) / PyTMDBoundary(...) exactly as production (§2.7)
def ioc_chunks(cfg) -> list[dict]          # every data_v4/gauges/ioc_<code>_*.json the provider reads, with sha256

class LevelTable:                          # exact h_B(t_i - tau) for integer tau at the record instants
    shifts: np.ndarray                     # (S,) int, -90 .. tau_tab_max
    H: np.ndarray                          # (S, T) float64
    batch_invariant: bool                  # G-levels outcome (§4.1)
    def at(self, tau_min: int, sel=None) -> np.ndarray     # table row; outside the table: live per-shift provider call (logged)
    def bank(self, sel=None, taus=marea.BANK_TAUS) -> "te.ShiftBank"

def build_level_table(provider, t, tau_tab_max, chunk=150_000) -> LevelTable
def boundary_series_1min(provider, name, cache_path) -> tuple    # (t_min, h, cached) = v8.boundary_series (3 values)
def lowpass(h_1min, width=61) -> np.ndarray                      # centred running mean; NaN unless all 61 samples finite
def marks(t_min, h) -> dict                                      # = g21.tide_marks(t_min, h)
def series_10min(t_min, h) -> tuple[pd.DatetimeIndex, np.ndarray]
```

### 1.4 `package.py`

```python
STEPS = ("record", "series", "pass1", "geometry", "judges", "levels", "pass2", "subsets", "manifest")   # levels need K_full

def build_record(cfg, provider, ext_dates) -> dict
def pass1(cfg, record, out_dir) -> dict          # frame counters, packed frame validity, scene stats
def pass2(cfg, record, dom, out_dir) -> None     # raw B03/B08 (int16) and SCL (uint8) at the package domain
def build_subsets(T, site) -> dict               # perm (10, T), window starts, scene-rule subsets
def prepare(site, res_m, steps=STEPS, force=False) -> str        # returns package_key
def write_manifest(common_dir) -> str
def verify_manifest(common_dir, full=False) -> str               # full=True rehashes; else size + mtime_ns

class CommonPackage:
    site; res_m; cfg; dir; keys                   # keys: package_key, obs_sha, frozen section shas, git (provenance)
    dates; time_index; t; T; h0                   # record, sorted by time
    shape; H; W; fill
    dom                                           # (P_dom,) int64 flat indices, ascending (§2.5)
    B03; B08                                      # np.memmap (T, P_dom) int16, raw cube values
    SCL                                           # np.memmap (T, P_dom) uint8
    frame_valid                                   # wc.PackedRows (T, H*W), primary rule and mode
    clr_n; wet_n                                  # (H, W) int16 full-record counters, primary rule and mode
    full                                          # domain.Domain of the full record (§2.6)
    keep_prod; sea_prod; inter_prod               # production extraction (gates only)
    levels; series1; marks_lp; marks_raw; series10; judges; scene_stats   # lazy
    def positions(self, flat) -> np.ndarray       # flat indices -> positions in dom (raises if absent)
    def observe(self, sel, pos, rule=MASK_RULE, mode=INDEX_MODE) -> tuple   # (ndwi float32 (n, m), V bool (n, m))
    def blocks(self, sel, pos, size=200_000) -> Iterator                  # (a, b, ndwi, V) column blocks
    def subset(self, family, N, rep) -> np.ndarray                         # sorted record positions
    def subset_key(sel) -> str                    # sha256(sel.astype("<i8").tobytes())
    def provider(self)                            # live provider (prepare, truth, gates, beyond-table shifts)
def load(site, res_m) -> CommonPackage            # verify_manifest(full=False), once per process and site
```

### 1.5 `domain.py`

```python
EXTRACT_RULE = dict(min_clear=31, wf_lo=0.10, wf_hi=0.90)   # marea.extract: clr_n > 30, 0.10 < wf < 0.90, sea wf >= 0.90
MIN_INTERTIDAL = 3000                                         # marea.reconstruct: fewer -> "sin_intermareal"
MIN_FINITE_S = 5000                                           # marea.reconstruct: interior tide only above this

@dataclass
class Domain:
    mode: str                  # "A" | "F"
    keep: np.ndarray           # flat indices, ascending
    pos: np.ndarray            # positions in pkg.dom
    sea: np.ndarray            # (H, W) bool
    inter: np.ndarray          # (H, W) bool
    s_km: np.ndarray           # (len(keep),) float64, NaN = isolated
    band_of: np.ndarray; centers: np.ndarray; edges: np.ndarray; K: int
    status: str                # "ok" | "no_intertidal" | "no_anchor"

def full(pkg) -> Domain
def for_subset(pkg, sel) -> Domain                  # A-mode; for_subset(arange(T)) == full(pkg) bitwise (tested)
def get(pkg, unit) -> Domain                        # A -> for_subset; F -> full; cached per (subset_key, mode)
def band_map_frame(full_domain, s_frame) -> np.ndarray   # band of any frame pixel with finite s (strata, §2.6)
```

### 1.6 `judges.py`

```python
def stations(site) -> list[dict]          # cfg.judges from estuary_truth/stations.json
def read_series(station) -> pd.Series     # CSV as delivered (QC applied upstream); naive UTC; sorted, unique
def centred_10min(series, sampling_min) -> pd.Series
def geometry(pkg, station) -> dict        # gx, gy, d_min_km, k_g, s_g_km, R_g_m, nbhd (frame flat), n_keep_in_nbhd, flags
def inclusion(pkg, station, series) -> dict   # method-independent rules (§2.8): coverage, frame, boundary member, dry
def build(pkg) -> list[dict]              # cached in common/<site>/judges.json
```

### 1.7 `truth.py`

```python
def vaklodingen_frame(pkg) -> dict        # z_raw (H, W), year, cell_id   (v1.truth_grid + nearest-cell lookup)
def lidar_frame(pkg) -> dict              # z_raw (H, W)                   (v10.lidar_truth(rtk.DEFAULT_GRID))
def rtk_points(pkg) -> dict               # flat, z (elev_alicante), block150
def range_filter(pkg, z_raw) -> tuple     # [min h0 - 0.25, max h0 + 0.25] over the full record; counts
def build_elevation_truth(pkg) -> str     # truth/<site>/truth_elev.npz + .json; support.npy (§5.1)
def E_curve(LEV, g_at_instants, taus) -> np.ndarray
def truth_s2(pkg, station, g) -> dict       # (a) and tau_ab, with tolerance intervals
def truth_m2(pkg, station, g10) -> dict     # (b) with jackknife SE
def frame_truths(pkg, g_s2) -> dict         # (c), diagnostics
def model_at_gauge(pkg, station, model) -> dict   # EOT20 / GOT5.6 baselines for (a), tau_ab, (b)
def build_lag_truth(pkg) -> str
```

### 1.8 `product.py`, `readout.py`

```python
STATUSES = ("ok", "empty", "not_computable", "r_failed", "budget", "error")

@dataclass
class Unit:
    site: str; res_m: int; group: str; variant: str
    family: str          # "rand" | "win" | "rule-tz" | "rule-g21" | "rule-ard"
    mode: str            # "A" | "F"
    N: int | str; rep: int
    sens: dict           # {"mask": ..., "index": ..., "floor": ..., ...}; {} for the controlled setting
    sel: np.ndarray; subset_key: str

@dataclass
class Product:
    method: str; variant: str; unit: Unit
    status: str; stage: str | None; reason: str | None
    z: np.ndarray | None          # (n_S,) float32 on the site support S (§5.1); NaN = unmapped; own quality rule applied
    z_raw: np.ndarray | None      # (n_S,) float32 before the quality rule (MAREA family: bracket 0; NIDEM: unfiltered; Granadeiro: = z)
    conf: np.ndarray | None       # (n_S,) float32, the method's own confidence, higher = more confident
    clear_obs: np.ndarray | None  # (n_S,) int16 valid observations behind each pixel (NIDEM: min over intervals)
    lag: np.ndarray | None        # (n_S,) float32 applied lag (min) where the method defines one, NaN elsewhere
    lag_kind: str                 # "linear" | "phase"
    method_bands: dict | None     # the method's own bands: edges, centres, lag, identified, bound hits, beyond-table
    fallback: str | None          # §3.0
    domain: dict                  # mode, n_keep_sel, n_keep_sel_outside_S, status
    diag: dict; keys: dict; timings: dict; peak_rss_mb: float; smoke: bool = False

def unit_dir(root, unit, method, variant) -> str   # runs/<site>/<method>[__<variant>]/<family>-<mode>/N<NNN>_r<RR>[__<senstag>]
def save(product, out_dir) -> None                 # temp dir; an existing dir is moved to <dir>.old-<utc>; then os.replace
def load(out_dir) -> Product
def inputs_key(pkg, unit, method, variant, config_key, code_key) -> str
class NotComputable(Exception): ...               # stage, reason, diag
```

```python
P_M2_MIN = 745.2361                          # M2 period in minutes
def wrap(minutes, lo=-120.0, period=P_M2_MIN)
def circ_median(minutes, period=P_M2_MIN) -> float     # unwrap around the circular mean, then linear median
def local(lag_S, nbhd_S, kind, min_px=30) -> tuple      # (value, n, flags)
def band(lag_S, strata_band_S, k, kind, min_px=30) -> tuple
def band_interp(centers, band_values, s_g, kind) -> float
def readouts(pkg, product, judges) -> dict
```

### 1.9 `inference.py`, `scoring.py`, `control.py`, `runner.py`, `report.py`

```python
def block_ids(rows, cols, size_m, pixel_m, band=None) -> np.ndarray
def variogram_range(x, y, v, seed, max_px=20_000, max_lag_m=5000, n_bins=25) -> dict    # exponential fit, practical range
def block_size(pkg, contrasts) -> dict          # lattice-rounded size, n_blocks, fits
def boot_weights(n_blocks, B, seed_seq) -> Iterator[np.ndarray]
def holm(pvals) -> np.ndarray
def sign_test(a_abs_err, b_abs_err) -> dict     # exact two-sided binomial on non-ties
def rep_spread(deltas_by_rep, min_share=0.9) -> dict
```

```python
MIN_N = {"vaklodingen": 30, "lidar": 30, "rtk": 20, "planted": 30}
def stats(z, t, min_n) -> dict      # n, rmse_c, rmse_meanc, rmse_raw, bias, mean_err, slope, intercept, r, std_z, std_t
def wmedian(sorted_vals, w_sorted) -> float
class BlockSums: ...
def views(pkg, truth, products) -> dict          # own_D, own_K, common_q, common_raw, pair:<a>|<b>, cells, survey_ge_2023
def score_elevation(pkg, truth, products, with_boot) -> tuple[pd.DataFrame, ...]
def accuracy_coverage(pkg, truth, product, steps=20) -> pd.DataFrame
def score_lags(pkg, lag_truth, products) -> pd.DataFrame     # read-outs from readout.py, interval errors, fallbacks
def curves(points, lags) -> tuple[pd.DataFrame, pd.DataFrame]
def reproduction_gate(site, res_m) -> dict
```

```python
def synth(base_pkg, scenario, out_dir, seed=SEEDS["pc"]) -> dict   # cube, overpass, planted z and tau, site config
def verdict(scores, lag_scores) -> dict                             # PASS/FAIL per method with the numbers
```

```python
GROUPS = ("marea_pair", "nidem", "granadeiro", "gm20", "hd26", "baselines", "published")
def plan(sites, groups, families, modes, N_list, reps, variants, sens, res_m) -> list[Unit]
def claim(path) -> bool; def heartbeat(path) -> None; def release(path) -> None
def run_unit(unit, opts) -> list[Product]     # NotComputable -> status; R crash -> "r_failed"; other -> "error" + traceback
def run_queue(units, opts) -> dict
def peak_rss_mb(runner_stats=None) -> float   # self ru_maxrss; for Granadeiro RRunner.stats["python_plus_r_peak_mb"]
def validated(site, res_m, method) -> bool    # gates and positive control passed
```

```python
def elevation_rows(points, truth_kind) -> list[str]
def lag_rows(lag_scores) -> list[str]
def curve_rows(curves) -> list[str]
def figures(root) -> list[str]
def write(root, res_m, paper=False) -> None
```

### 1.10 Adapter contract

Every module in `adapters/` exposes `GROUP`, `METHODS`, `VARIANTS`, `CONFIG` (a frozen dict) and:

- `run(pkg, unit, out_root, **opts) -> list[Product]`;
- `gate(pkg, out_root, **opts) -> dict`.

`config_key = sha256(CONFIG, FROZEN[<method section>], unit.mode, unit.sens, variant)`. `adapters/__init__.py` holds `REGISTRY = {GROUP: module}`.

CLI subcommands: `check-env`, `prepare`, `truth`, `gate`, `control`, `run`, `status`, `score`, `report`, `lock` (§7.4).

---

## 2. The common input package (per site and resolution)

### 2.1 Sites

| | escalda | wadden | ems | ferrol | villaviciosa |
|---|---|---|---|---|---|
| Cube (10 m) | `ndwi_cube_escalda_2023-2025_10m.nc` (server) | `..._wadden_..._10m.nc` (server) | `..._ems_..._10m.nc` (server) | `ndwi_cube_ferrol_2023-2025_10m.nc` | `ndwi_cube_villaviciosa_grande_10y.nc` |
| Cube (20 m, local) | `..._20m.nc` | `..._20m.nc` | `..._20m.nc` | – | – |
| Production extraction | `products_escalda_10m/marea_extract.npz` | `products_wadden_10m/...` | `products_ems_10m/...` | `products_ferrol/...` | `products_marea/marea_extract.npz` |
| Overpass | `products_<site>_10m/overpass_times.json` | same | same | `products_ferrol/overpass_times.json` | `products_marea/overpass_times.json` |
| Boundary | `make_boundary(aoi, "EOT20", n_gauges=2, period=PERIOD, exclude=["trnz"])` | n=1, exclude harl | n=1, exclude delf | n=1, exclude fer2 | `PyTMDBoundary("EOT20", *aoi.centroid, directory="tide_models")` |
| Mouth side | west | north+west | north+west | west | any |
| Frame H × W (10 m) | 1467 × 2101 | 3425 × 2776 | 3206 × 3413 | 905 × 1707 | 915 × 915 |
| Record T (expected) | 464 | 462 | 462 | 461 | 465 (of 1,379 cube dates) |
| Production keep (10 m) | 291,799 | 1,455,129 | 1,886,417 | 103,759 | 32,120 |
| Package domain P_dom (est.) | ≈ 0.41 M | ≈ 2.0 M | ≈ 2.6 M | ≈ 0.15 M | ≈ 0.05 M |
| Production bands K (10 m) | 8 | 8 | 13 | 7 | 3 |
| Elevation truth | Vaklodingen | Vaklodingen | Vaklodingen (epoch ≈ 2020, flagged) | none (lag only, [D15]) | PNOA LiDAR + RTK |
| Lag judges | gatvanborssele, terneuzen, terneuzen.westsluis.buiten, borssele | terschelling.west, harlingen.waddenzee | emshoern_9340010, eemshaven.waddenzee, eemshaven.haven, dukegat_3990020, knock_3990010, delfzijl, emden_neue_seeschleuse_3970010, pogum_3950020, nieuwestatenzijl.dollard, uithuizerwad | fer2, fer4_3212 | none |

Every station is listed; the method-independent inclusion rules of §2.8 decide which ones are scored. `aoi = pyintertidal.sites.get(site)`. Every boundary call is preceded by `pyintertidal.net.use_system_certificates()`, with `directory="tide_models"` and `cache_dir="data_v4/gauges"`. The local 20 m Escalda test has T = 464 and a production keep of 73,764 px. Site `pc` is the synthetic positive control (§4.2).

### 2.2 Record (C1)

The same rule as v12's "all" record and v7's primary record:

```python
ext = np.load(cfg.extract, allow_pickle=True)          # header arrays only: dates, keep, shape, sea, inter, s_km?
ext_dates = np.array([str(d) for d in ext["dates"]])   # == cube dates (asserted via v7.cube_grid)
over = {k: v8.to_utc_naive(v) for k, v in json.load(open(cfg.overpass)).items()}
cand = np.flatnonzero([(PERIOD[0] <= d <= PERIOD[1]) and d in over for d in ext_dates])
t = pd.DatetimeIndex([over[d] for d in ext_dates[cand]]).tz_localize(None)
assert t.is_monotonic_increasing
h = np.asarray(provider.levels(t), float)               # ONE call, sorted instants (PyTMDBoundary returns time order)
time_index = cand[np.isfinite(h)]
```

The record gate (G-record, §4.1) checks it against stored products at all five sites; a mismatch means the IOC caches changed and stops `prepare`.

### 2.3 Per-pixel cloud mask (C2)

```python
def mask_valid(scl_u8, rule=MASK_RULE):
    present = scl_u8 != SCL_NODATA_U8
    if rule == "blocklist":                   # the brief: SCL 0, 3, 8, 9, 10 and no data masked
        return present & ~np.isin(scl_u8, BLOCK_SCL)
    if rule == "marea_clear":                 # production MAREA (marea.CLEAR), sensitivity only
        return present & np.isin(scl_u8, CLEAR_SCL)
    raise ValueError(rule)
```

- **Primary: `blocklist`**, as decided (C2). It equals ITEM's PQ mask except for SCL 1, which does not occur in the local cubes (light check: SCL 0 and 1 absent; SCL 2 in 3,601 and SCL 11 in 10,024 of 168 M frame observations at Escalda 20 m). `check-env` counts SCL 1 on the server cubes.
- **Sensitivity: `marea_clear`** at N = all for every method (§3.9).
- The package stores SCL, not a mask, so the rule is a run-time option and never baked into the package.

### 2.4 Band preprocessing, index and the observation (C3, [D2])

All cube bands are int16 with `_FillValue = -32768`, no `scale_factor` or `add_offset`, holding offset-harmonised L2A values (value = DN − 1000), so reflectance can be negative. `check-env` asserts the encoding and that no non-fill value is below −1000 (v8.load_bands' invariant).

**Common band preprocessing (primary, `floor0`):** every method receives B03 and B08 floored at 0 (non-fill values only) before any index is computed. This is a physical bound applied identically to every method; it removes |NDWI| > 1 and the sign flips that turn dark water with B08 < 0 into "land", which NIDEM's 0.25 and Granadeiro's 0.2 SD thresholds (defined on NDWI in [−1, 1]) cannot absorb. Light check on six chunk windows of Escalda 20 m (464 scenes, 54.6 M blocklist-valid observations):

| | share of valid observations |
|---|---|
| B08 < 0 | 4.2·10⁻³ |
| B03 < 0 | 3.1·10⁻⁴ |
| B03 ≤ 0 and B08 ≤ 0 (NDWI undefined under floor0) | 1.7·10⁻⁴ |
| raw \|NDWI\| > 1 | 4.2·10⁻³ |
| raw NDWI < −1 | 4.5·10⁻⁴ |
| wet/dry differs between raw and floor0 | 3.6·10⁻⁴ |

**Index and observation definition, the same for every method:**

- NDWI = `obs.ndwi(g, n)`, float32, with g, n the decoded (and, under floor0, floored) bands; the expression is bit-identical to `marea.extract` and `wc.scene_ndwi`.
- **Valid observation `V` = mask_valid(SCL) & isfinite(NDWI)** for every method. MAREA's `nan_to_num` never fires on a valid observation, and NIDEM, GM20 and HD26 drop exactly the same observations.
- **Wet = V & (NDWI > 0).** Under floor0 this equals `marea.extract`'s own wet test (B03 > B08) wherever NDWI is defined.
- **NIR (Granadeiro):** `obs.nir` = `g21.to_reflectance` of the floored int16 values (float64, fill → NaN). In common mode, observations outside V are set to the int16 fill before conversion (`obs.mask_raw_bands`), so every Granadeiro step treats them as no data, as it already treats fill. Granadeiro's SD(NDWI) mask uses Eq. 2 on the same floored reflectances (float64), as in v8.
- **Sensitivity `raw`:** the production behaviour (no floor) for every method at N = all (§3.9). Gates that reproduce production run in raw mode.

### 2.5 Package domain and stored arrays

The package stores raw values, not derived observations, so the mask rule and the index mode are run-time options:

- `B03`, `B08`: (T, P_dom) int16 raw cube values at the package domain;
- `SCL`: (T, P_dom) uint8 (fill → 255);
- `frame_valid`: packed bits (T, ⌈H·W/8⌉) of V under the primary rule and mode, for the whole frame;
- `clr_n`, `wet_n`: (H, W) int16 full-record counters under the primary rule and mode.

**Package domain `dom` = keep_prod ∪ U_rec ∪ NB**, sorted:

- `keep_prod`: the production extraction keep (gates);
- `U_rec = {clr_n ≥ 31, wet_n ≥ 4, dry_n ≥ 4}` over the full record (dry = clear − wet). It is an **exact superset of every A-mode `keep_sel`**: 0.10 < wf < 0.90 with ≥ 31 clear observations needs at least 4 wet and 4 dry observations in the subset, hence in the record;
- `NB`: frame pixels inside the judges' neighbourhoods (§2.8), for the GM20 `own` variant.

Light check (six windows of Escalda 20 m, primary rule and mode): |U_rec| / |keep_full| = 1.21-2.12 per window, 1.34 weighted by keep; keep_full is within 1 % of the production keep in every window. Hence P_dom ≈ 1.4 × keep (table §2.1).

**A-mode sea mask without frame values.** A pixel outside `U_rec` with ≥ 31 clear observations has either ≤ 3 dry observations ("sea-like": in any subset with ≥ 31 clear observations its wf ≥ 28/31 > 0.90, so it is sea exactly when its subset clear count is ≥ 31) or ≤ 3 wet observations (never sea or intertidal). So `sea_sel` is exact from `frame_valid` alone for those pixels and from the stored values inside `dom`.

**Two streaming passes** over the record scenes, one scene at a time on the full frame, reading raw int16 (`mask_and_scale=False`):

```python
# pass 1: counters, frame validity, scene statistics
for i, t in enumerate(record.time_index):
    g, n, s = (ds[v].isel({t_dim: int(t)}).values for v in ("B03", "B08", "SCL"))
    s8 = obs.scl_u8(s, fill); nd, V = obs.observe(g, n, s8, fill)      # primary rule and mode
    clr_n += V; wet_n += obs.wet(nd, V)
    frame_valid[i] = np.packbits(V.ravel())
    stats.append(scene_stats(g, n, s8, V, keep_prod))
# geometry (§2.6), judges (§2.8) -> dom
# pass 2: raw values at dom
for i, t in enumerate(record.time_index):
    g, n, s = (ds[v].isel({t_dim: int(t)}).values.ravel() for v in ("B03", "B08", "SCL"))
    B03[i], B08[i], SCL[i] = g[dom], n[dom], obs.scl_u8(s, fill)[dom]
```

`scene_stats.csv` per record scene: SCL counts 0..11 and fill on the frame and on keep_prod; g21 frame cloud and no-data shares (`g21.scene_frame_shares`); load_ard good fraction (`wc.frame_good_fraction`); V share on the frame and on keep_prod; negative-band and raw |NDWI| > 1 counts; h0; rising (low-pass marks).

### 2.6 Full-record domain, geometry and strata

`domain.full(pkg)` applies the extraction rule to the full record under the primary rule and mode:

- `keep_full` = clr_n > 30 & 0.10 < wf < 0.90; `sea_full` = clr_n > 30 & wf ≥ 0.90; `inter_full` = keep_full as a frame mask;
- `s_km = geometry.along_distance(sea_full | inter_full, geometry.mouth_seeds(sea_full, side=mouth_side), pixel_m).ravel()[keep_full] / 1000` (float64; `ValueError` from `mouth_seeds` gives NaN everywhere, as in `reconstruct`);
- bands: `te.make_bands_fixed(s_km, 3.0, 500)` when more than 5,000 values of s are finite, otherwise one band with τ = 0 (as `reconstruct`); NaN s gets band −1.

These define the F-mode domain, the scoring strata ("band:k"), the judges' bands and the neighbourhood rule. They are fixed for every method, mode and N.

**Strata outside keep_full.** The method-neutral scoring domain (§5.1) also holds pixels outside keep_full. For them, `s_frame = along_distance(sea_full | inter_full, seeds)` is read where finite, and `domain.band_map_frame` maps its raw band index ⌊s/3⌋ through the merge of `make_bands_fixed` (the merged band of the keep pixels in that raw band; for a raw band with no keep pixel, the merged band of the nearest lower raw band that has one). Pixels with NaN s form the stratum "no s".

**Coordinates and blocks:** `grid = v7.cube_grid(cube)`; `rows, cols = flat // W, flat % W`; `x, y = grid["xs"][cols], grid["ys"][rows]`. Block ids are computed on the fly by `inference.block_ids` for any size (§5.4).

**Production geometry (gate only):** `s_km` recomputed from the production `sea_prod`, `inter_prod` and `keep_prod` equals the stored `ex["s_km"]` (Dutch, Ferrol) or `results/server_10m/products_marea/marea_eot20_g0_2023-2025/marea.npz["s_km"]` (Villaviciosa) as float32, NaN equal to NaN (G-keep).

### 2.7 Boundary series, level table and marks (one provider per site)

The provider is built once per site by `levels.make_provider(cfg)` with the production call (table §2.1). Every tide quantity of every method derives from it.

1. **Exact level table** `LEV` (S × T, float64; built after the geometry step), integer τ ∈ [−90, τ_tab_max] with `τ_tab_max = max(240, 180·(K_full − 1))` (m2a's monotone increments are each bounded by 180 min, so a band can reach 180·k; Ems K = 13 gives 2,160 min, S = 2,251, 8.3 MB).
   - Built from sorted provider calls on the sorted unique union of `t − τ`, in chunks of at most 150,000 instants, mapped back with `searchsorted`.
   - Assert every row is finite and `LEV[τ=0] == h0` bitwise.
   - **G-levels (batch invariance):** every `marea.BANK_TAUS` row and every 60th row equal per-shift `provider.levels(t − τ)` calls bitwise. If any differs, the table is rebuilt by per-shift calls, `batch_invariant = False` is recorded, and `LevelTable.at(τ, sel)` then calls the provider per shift on `t[sel]`, exactly as `reconstruct` does.
   - **Beyond the table**, `at()` makes a live per-shift call on `t[sel]` (identical to `reconstruct`) and logs it. A lag is never capped.
   - MAREA's m2a bank is `te.ShiftBank(BANK_TAUS, [LEV[τ][sel] for τ in BANK_TAUS])`, the estimator's own 15-min bank with its own clipping at 180 min.
2. **1-min series** over PERIOD ± 2 days by `v8.boundary_series(provider, name, cache_path)`, which returns `(t_min, h, cached)`; cached as `common/<site>/boundary_1min_cache.npz` with v8's tag.
3. **10-min series** `hB10`: every 10th sample of (2) on the grid 2023-01-01 00:00 .. 2025-12-31 23:50.
4. **Marks, two versions:**
   - `marks_lp` (shared by every method that uses marks in the controlled rows): `g21.tide_marks` on `levels.lowpass(h, 61)`, a centred 61-min running mean of the 1-min h_B (NaN unless all 61 samples are finite; `tide_marks` drops NaN). It removes the sub-hour noise that gave 105,287 turning points at the Ems and keeps M4/M6 (amplitude factor 0.99 at M2, 0.90 at 4 h).
   - `marks_raw`: `g21.tide_marks(t_min, h)` on the raw 1-min series, exactly as v8 (faithful variants and the paper gate).
   - Users: Granadeiro's rising/ebbing split and h_c (controlled: `marks_lp`), GM20's HW fold reference (`marks_lp`). Per site, the report gives |Eq. 1 − h_B| at the record instants for both versions and the number of marks.

`ioc_chunks(cfg)` records every IOC chunk the provider reads, with its sha256. Adapters never build a provider; they read `pkg.levels`, which calls the live provider only beyond the table.

### 2.8 Judges (in-estuary gauges)

- **Inputs:** `data_v4/gauges/estuary_truth/stations.json` and one CSV per station (`time_utc,level_m`, naive UTC, QC applied upstream). Only centred quantities are compared, because datums differ.
- **Geometry** on the run grid (`judges.geometry`), never the 20 m bands in `stations.json`:

```python
gx, gy = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True).transform(lon, lat)
d_keep = np.hypot(x_keep_full - gx, y_keep_full - gy)
near200 = np.argsort(d_keep)[:200]
k_g = int(np.bincount(band_of_full[near200][band_of_full[near200] >= 0]).argmax())
s_g = float(np.nanmedian(s_km_full[near200]))
R = max(500.0, d_keep.min() + 300.0)
while (d_keep <= R).sum() < 100 and R < 2000.0:
    R += 250.0
nbhd = frame pixels with hypot(x - gx, y - gy) <= R          # flag "sparse" if < 100 keep_full px at 2 km
```

- **Inclusion, by method-independent rules only** [D10] (FROZEN `judges`):
  1. inside the frame;
  2. a gauge value within 6 min for at least 50 % of the record instants (excludes borssele, data only from 2025-10-28, and eemshaven.haven, January-April 2023);
  3. not within 3 km of a boundary member (excludes fer4, about 1 km from fer1);
  4. not a dry gauge: at most 10 % of its record-instant values within 1 cm of its minimum (uithuizerwad fails this and rule 1).
- The two Terneuzen gauges, 0.7 km apart, count as one location (mean of their two errors).
- Band-0 ("mouth") judges stay in the headline set. A "without mouth judges" sensitivity (s_g < 3 km) is shown for every method.

### 2.9 Subsets (identical for every method)

```python
perm[r] = np.random.default_rng([SEEDS["subsets"], SITE_ID[site], r]).permutation(T)       # r = 0..9
sel_rand(N, r) = np.sort(perm[r][:N])                     # nested in N within a rep; N == "all" -> arange(T)
start(N, r) = np.random.default_rng([SEEDS["windows"], SITE_ID[site], N, r]).integers(0, T - N + 1)
sel_win(N, r) = np.arange(start, start + N)               # contiguous windows (sensitivity)
sel_rule(name) = record positions passing the scene rule: tz (MAREA transition zone, 10 m result.json record_dates),
                 g21 (cloud < 0.10 and no data <= 0.02, frame shares), ard (load_ard good fraction >= 0.5)
```

Stored in `common/<site>/subsets.npz` with T and the seeds. `subset_key = sha256(sel.astype("<i8"))`. The same `(family, N, r)` gives the same dates at 20 m and 10 m. A group with fewer reps takes reps 0..R−1. The scorer refuses to compare products with different subset keys within one `(family, mode, N, rep)`.

### 2.10 Files, formats and memory plan

`results/v13_benchmark_<res>m/common/<site>/`:

| File | Content |
|---|---|
| `record.json` | dates, time_index, UTC instants, boundary spec and name, `ioc_chunks`, exclusions, grid summary |
| `levels.npz` | shifts, H (S, T), h0, `batch_invariant` |
| `boundary_1min_cache.npz` | v8 format `{tag, h}` |
| `marks.npz` | `marks_lp`, `marks_raw` and their bookkeeping |
| `dom.npy` | (P_dom,) int64 |
| `b03.npy`, `b08.npy` | (T, P_dom) int16, `np.lib.format.open_memmap` |
| `scl.npy` | (T, P_dom) uint8 |
| `frame_valid.npy` | (T, ⌈HW/8⌉) uint8 packed bits |
| `counts.npz` | clr_n, wet_n (H, W) int16 |
| `pixels.npz` | keep_full, sea_full, s_km, band_of, centers, edges, K, s_frame (float32), band_frame (int16), keep_prod, sea_prod, inter_prod |
| `scene_stats.csv` | §2.5 |
| `judges.json` | §2.8 |
| `subsets.npz` | §2.9 |
| `manifest.json` | §2.11 |

Disk at 10 m (estimates): raw arrays T·P_dom·5 B — Escalda 0.95 GB, Wadden 4.7 GB, Ems 6.1 GB, Ferrol 0.35 GB, Villaviciosa 0.10 GB; `frame_valid` 0.18 / 0.55 / 0.63 / 0.09 / 0.05 GB; about 14 GB in all. Prepare peak RAM < 1 GB (one frame of three bands plus counters). Workers memory-map the arrays and read column blocks; no worker holds a full (T, P_dom) array. A compressed npz is never used for the per-pixel arrays.

Full-frame methods (NIDEM, Granadeiro) stream only their subset's scenes from the per-scene-chunked cube, (1, 256, 256), and never cache full frames.

### 2.11 Provenance, keys and the pre-registration lock

- **`manifest.json`**: sha256 of every package file, of the cube, the extraction, the overpass file and each IOC chunk; `obs_sha`; the shas of the FROZEN sections `record`, `package`, `obs`, `domain`, `levels`, `subsets` (pass 1 uses the primary observation and the extraction thresholds); git HEAD and dirty flag (provenance only, never verified).
- **`package_key`** = sha256 of exactly those entries. It does not contain the mask rule, the index mode, any method, scoring or curve parameter, or git.
- **Per unit:** `config_key` (adapter CONFIG + its FROZEN section + mode, sens, variant); `code_key` (sha256 of the adapter module, `obs.py`, `domain.py`, `levels.py`, the `pyintertidal` modules it imports, and `r/granadeiro_fit.R` for Granadeiro); `inputs_key = sha256(package_key, subset_key, method, variant, config_key, code_key)`. `run.json` also records the Python, numpy, scipy, xarray and pyTMD versions, R and nplr/lmodel2/mgcv versions and the R BLAS (`Rscript -e 'cat(La_library(), extSoftVersion()["BLAS"])'`), the BLAS/OMP thread settings, git HEAD.
- **Scores** record the shas of the FROZEN sections `truth_elev`, `truth_lag`, `judges`, `scoring`, `inference`, `curves`; changing them reruns `score` only.
- **Verification cost:** `prepare`, `gate keep` and `score` rehash everything (`verify_manifest(full=True)`); every other process checks size and `mtime_ns` once per site.

| Change after a unit is done | Invalidates |
|---|---|
| `git pull` that leaves the hashed files unchanged | nothing |
| `obs.py`, `domain.py`, `levels.py`, the record, package, obs, domain, levels or subsets sections of FROZEN | every unit of that site (new package_key or code_key) |
| an adapter module or its FROZEN section | that adapter's units |
| adding N values, reps, variants or sensitivities | nothing: new units only |
| `truth_*`, `judges`, `scoring`, `inference`, `curves` sections | `score` and `report` only |

- **Pre-registration lock.** `v13_benchmark lock --res 10` refuses to run while the working tree is dirty in `experiments/benchmark`, `pyintertidal` or `tests`; it writes `results/v13_benchmark_10m/LOCK.json` with git HEAD, every FROZEN section sha and the positive-control verdicts. `score` refuses to score real-data 10 m products before the lock exists (20 m test runs, the positive control and gates are exempt). Any later change to a locked section is logged in `LOCK.json` (`amendments`, with date and reason) and reruns every affected method, never only some.

---

## 3. Adapters

### 3.0 Common behaviour

- **Input:** `(pkg, unit)`; `unit.sel` is the sorted record subset; `domain.get(pkg, unit)` gives the domain for the unit's mode.
- **Output:** a `Product` per method on the site support S (§5.1), saved to `runs/<site>/<method>[__<variant>]/<family>-<mode>/N<NNN>_r<RR>[__<senstag>]/{product.npz, run.json, log.txt}`. At N = all, full-frame rasters are also written for the maps.
- **Statuses:**
  - `ok` / `empty`: the product exists and maps at least / fewer than `MIN_N` scoring pixels (or gives / gives no lag field).
  - `not_computable`: the method's own rule stops it (`record_floor`, `no_intertidal`, `empty_interval`, `preflight`, `no_valid_pixel`, `scenes_lt_8`, `harmonic_all_bands`).
  - `r_failed`: an R job other than the GAM returned rc ≠ 0; the R log tail is stored.
  - `budget`: stopped by `--max-r-hours` after the projection (§4.3); resumable.
  - `error`: any other exception, with the traceback.
- **One failure policy for lag estimation** [D3]:
  - MAREA: with no anchor (K = 1) or no band identified, MAREA applies τ = 0 (its own behaviour), so its map equals no delay; `fallback = "no_anchor" | "unidentified_all"`. Bands with fewer than 50 usable columns are flagged unidentified in `method_bands`; their applied lag is whatever m2a returned (with monotone increments, the previous band's lag).
  - Granadeiro: an empty pool, fewer than 60 unique located lag pixels, or a GAM R failure → lag 0 everywhere, final DEM = the preliminary DEM (heights at lag 0); `fallback = "pool_empty" | "gam_lt_60" | "gam_failed"`.
  - GM20, HD26: no lag field where every pixel or band fails QC; the scorer then reads a fallback of 0 (§5.6).
  - NIDEM has no lag; its interval guards stay NOT COMPUTABLE.
  - **Primary** scoring uses the products as produced (fallbacks included and flagged). The **"identified only"** sensitivity treats every fallback, product or read-out, as NOT COMPUTABLE. The computable fraction and the fallback fraction are reported for every method.
- Any R crash outside the GAM is `r_failed`. At N = all it blocks `report` like an error; in curve units up to 10 % per (method, N) is tolerated and reported, more blocks `report`.
- A subset is never redrawn because a method fails on it.
- Method caches (`runs/<site>/<group>/_cache/`) are keyed by `package_key` and are never part of the package.
- **Threads:** the CLI sets `OMP/OPENBLAS/MKL_NUM_THREADS` before importing numpy. The BLAS thread count is fixed per group in FROZEN (marea_pair, nidem, gm20, hd26: 4; granadeiro: 1, so the R workers inherit one thread each) and is part of `config_key`, because a multithreaded `dgemv` partitions columns by thread and the last bits of `invert_series` depend on that partition; gates run with the same setting. `--blas-threads` exists only for tests and local smoke runs. Every replay that used joblib in production runs with an explicit pool (`--boot-threads`, default 1). Settings go into `run.json`.

### 3.1 Domain step (`domain.py`)

`for_subset(pkg, sel)`:

```python
clr_d, wet_d = counts over sel of V and wet at dom (pkg.blocks, primary rule and mode)
wf_d = np.where(clr_d > 30, wet_d / np.maximum(clr_d, 1), np.nan)
inter_d = (wf_d > 0.10) & (wf_d < 0.90); sea_d = wf_d >= 0.90
sea_like = (clr_n >= 31) & (clr_n - wet_n <= 3) & ~in_dom               # static, from the full-record counters
clr_out = frame_valid rows sel summed over sea_like pixels (unpacked in pixel blocks)
inter_sel = frame mask of dom[inter_d]; sea_sel = frame mask of dom[sea_d] | (sea_like & (clr_out > 30))   # exact, §2.5
keep_sel = dom[inter_d]                                                    # ascending flat indices
if len(keep_sel) < MIN_INTERTIDAL: status = "no_intertidal"; return        # marea.reconstruct returns None
try:
    s = geometry.along_distance(sea_sel | inter_sel, geometry.mouth_seeds(sea_sel, side=mouth_side), pixel_m).ravel()[keep_sel] / 1000
except ValueError:                                                          # no seed: reconstruct's except branch
    s = np.full(len(keep_sel), np.nan)
if np.isfinite(s).sum() > MIN_FINITE_S: edges, centers, band_of = te.make_bands_fixed(s, 3.0, 500)
else: centers = np.array([0.0]); band_of = np.zeros(len(keep_sel), int); K = 1; status = "no_anchor"
```

`for_subset(arange(T))` equals `full(pkg)` bitwise (test). The A-mode domain is computed once per `subset_key` and cached in `runs/<site>/_domain/`. Its cost is one pass over the subset's rows of the package plus one geodesic distance on the frame (about 1 GB and under a minute at the Ems).

### 3.2 `marea_pair`: no delay and MAREA (one unit, two products)

| Step | No delay | MAREA |
|---|---|---|
| FAITHFUL | `marea.invert_series` (closed-form a, b over a 50-point μ grid on [min h0, max h0] of the subset; σ grid `marea.SG_GRID`; guards min_obs 8, 0.15 < b < 2.5, \|a\| < 2.5); quality rule `min_bracket = marea.QUALITY_MIN_BRACKET = 3`; h = h_B(t) at the overpass instants | everything `marea.reconstruct` does: extraction-rule domain (A-mode); 3-km bands; `te.m2a_rasch` with configs/m2.yaml (`sigma0 = 0.20`, `sg_grid = (0.06, 0.1, 0.15, 0.22, 0.32, 0.45)`, `z_points = 60`, `max_px_band = 1200`), `tau_bounds = marea.TAU_BOUNDS = (-45, 180)`, `monotone = True`, band 0 pinned at 0, columns drawn with `default_rng(20260819 + 31)`; causality clip; `te.smooth_lag_profile(centers, tau_used, s_km, 3.0)`; rounded to the minute; isolated pixels τ = 0; inversion per distinct τ_px with the exact h_B(t − τ_px), the same lo/hi and quality rule |
| Shared rules | record floor 40 scenes for both products [D4] (sensitivity 80 and 0, curves in both modes); fewer than 3,000 intertidal px → NOT COMPUTABLE for both | |
| ADAPTED | common record, mask and floor0 preprocessing; exact levels from the frozen table (live beyond it, identical to `reconstruct`); no hypsometry or figure; the scene bootstrap is a diagnostic only (§5.7) | |

```python
M2 = yaml.safe_load(open("configs/m2.yaml"))
FIT_KW = dict(sigma0=M2["sigma0_m"], sg_grid=tuple(M2["sigma_perfil_m"]), tau_bounds=marea.TAU_BOUNDS,
              z_points=M2["z_puntos"], max_px_band=M2["max_px_banda"]["m2a"], monotone=True)
SLICE = 200_000                                               # 50 x invert_series' own chunk of 4000

def invert_group(Y, C, h, lo, hi):                            # one tau group, columns in reconstruct's order
    out = [marea.invert_series(np.nan_to_num(Y[:, a:a + SLICE], nan=0.0), C[:, a:a + SLICE], h, lo, hi,
                               min_bracket=0) for a in range(0, Y.shape[1], SLICE)]
    return concat(z), concat(sigma)                           # chunk partition == one call on the group (bitwise)

def bracket_min(C, h, z):                                     # == invert_series' own bracket count, at mu = z
    below = (C & (h[:, None] < z[None, :])).sum(0); above = (C & (h[:, None] > z[None, :])).sum(0)
    return np.minimum(below, above)

def run(pkg, unit, floor=40):
    sel, d = unit.sel, domain.get(pkg, unit)
    if len(sel) < floor: raise NotComputable("record_floor")             # both products
    if d.status == "no_intertidal": raise NotComputable("no_intertidal")
    h0 = pkg.levels.at(0, sel); lo, hi = float(h0.min()), float(h0.max())
    Y, C = pkg.observe(sel, d.pos)                                       # column blocks in practice
    # no delay: one group = d.keep in ascending order
    z_raw_nd, sg_nd = invert_group(Y, C, h0, lo, hi)
    br_nd = bracket_min(C, h0, z_raw_nd); z_nd = np.where(br_nd >= 3, z_raw_nd, np.nan)
    # MAREA
    if d.K > 1:
        cols = m2a_columns(d.band_of, d.K, M2["seed"] + 31, FIT_KW["max_px_band"])   # replay of m2a_rasch's draw
        Yz, Cc = np.nan_to_num(Y[:, cols], nan=0.0), C[:, cols]
        r = te.m2a_rasch(Cc & (Yz > 0), Cc, pkg.levels.bank(sel), d.band_of[cols], d.centers, d.K,
                         rng=np.random.default_rng(M2["seed"] + 31), **FIT_KW)        # no redraw: <= max_px_band per band
        tau = np.asarray(r["tau"], float); usable = per-band count of (Cc.sum(0) >= 6); identified = usable >= 50
    else:
        tau = np.array([0.0]); fallback = "no_anchor"
    tau_used = np.where(np.abs(tau) > marea.TAU_DETECT_MIN, tau, 0.0); tau_used = np.where(tau_used < 0, 0.0, tau_used)
    tau_s = te.smooth_lag_profile(d.centers, tau_used, d.s_km, 3.0) if d.K > 1 else np.zeros(len(d.keep))
    tau_px = np.where(np.isfinite(tau_s), np.round(tau_s), 0.0)
    for tv in np.unique(tau_px):
        g = np.flatnonzero(tau_px == tv)
        h = h0 if tv == 0 else pkg.levels.at(int(tv), sel)              # table row or live per-shift call
        z_raw_m[g], sg_m[g] = invert_group(Y[:, g], C[:, g], h, lo, hi)
        br_m[g] = bracket_min(C[:, g], h, z_raw_m[g])
    z_m = np.where(br_m >= 3, z_raw_m, np.nan)
    # products: z, z_raw, conf = bracket_min (ties broken by -sigma), clear_obs = C.sum(0), lag = tau_px (applied,
    # 0 at isolated pixels), method_bands = {edges, centers, tau, tau_used, identified, usable,
    # bound_hit = tau >= 179.5 or an increment at 3 h, beyond_table}
```

`z_m == np.where(bracket, z_raw, nan)` is exactly `invert_series(min_bracket=3)`: the bracket count uses the fitted μ, which equals `z_raw` wherever the guards pass, and z is NaN elsewhere in both. Under the bitwise column-slice rule the G-marea gate stays bitwise (§4.1); a test covers a group size that is not a multiple of 4000 and single-column groups.

**Variants and sensitivities:** `tau120` (`tau_bounds = (-45, 120)` and the bank restricted to BANK_TAUS ≤ 120, the setting before 2026-09-16); bracket 0 is `z_raw` itself; floors 80 and 0 (curves in both modes); mask and index sensitivities (§3.9).

**MAREA's scene bootstrap** (30 draws × 600 px/band, `default_rng(20260819)`), replayed sequentially with the draws of `te.bootstrap_lags`, is kept at N = all as a diagnostic of MAREA's own uncertainty. It is not used in any comparison (§5.7).

### 3.3 `nidem`

| | NIDEM |
|---|---|
| FAITHFUL | ITEM intervals (`wc.interval_assignment`) over the subset's own observed tidal range (all-observations rule); Z_j medians, U_j SDs; per-interval median NDWI composites over the full frame; R = land flags; 2-iteration gap fill; contours at 0.5..8.5 keyed by level; linear TIN in pixel space; unfiltered = R in 1..8; **filtered = unfiltered where the mean per-interval SD ≤ 0.25** (the row); empty-interval stop and the 1 % pre-flight floor = NOT COMPUTABLE (never relaxed) |
| ADAPTED | pixel mask = common mask; NDWI from floor0 bands (so the 0.25 threshold applies to NDWI in [−1, 1], as ITEM defined it); record = common; pre-flight counted on keep_full (v7 counted on the production keep; the 1 % guard fires only when an interval's scenes are overcast, so the pixel set does not change its outcome) |

```python
def run(pkg, unit, rule=obs.MASK_RULE, mode=obs.INDEX_MODE):
    sel = unit.sel; ti = pkg.time_index[sel]; h = pkg.h0[sel]
    try:
        j, edges, Z, U, sizes = wc.interval_assignment(h)
    except wc.EmptyIntervalError as e:
        raise NotComputable("empty_interval", str(e))
    cnt, _ = wc.clear_counts_by_interval(obs_keep_full(pkg, sel, rule, mode), None, j)   # PackedRows, cached
    cov = wc.interval_coverage(cnt, raise_error=False)                                   # no cube read before this
    if not cov["passed"]:
        raise NotComputable("preflight", cov)
    comp = composites(pkg.cfg.cube, ti, j, reader(rule, mode), check=KeepCheck(pkg, sel, rule, mode))
    assert np.array_equal(comp["count"].reshape(9, -1)[:, pkg.full.keep], cnt)
    try:
        res = wc.nidem(comp, Z, U)
    except wc.NoClearObservationsInIntervalError as e:
        raise NotComputable("no_valid_pixel", str(e))
    # products: z = filtered, z_raw = unfiltered, conf = -sd_mean, clear_obs = min_j count_j,
    # diag = v7.diagnostics(res, comp, pkg.full.keep, Z, U, sizes, edges, h); N == "all": v7.write_rasters(new dir)
```

`composites(...)` copies the row-block loop of `wc.stream_composites` (`waterline_contour.py` L861-907) with the cube's y-chunk (256 rows) and calls the private `wc._composite_rows(read, j, r1 - r0, W)` with:

```python
def read(i, r0=r0, r1=r1):
    sl = {t_dim: int(ti[i]), "y": slice(r0, r1)}
    g, n, s = (ds_raw[v].isel(**sl).values for v in ("B03", "B08", "SCL"))     # mask_and_scale=False
    nd, v = obs.observe(g, n, obs.scl_u8(s, fill), fill, rule, mode)          # gate: raw mode + obs.item_valid
    check(int(ti[i]), r0, r1, nd, v)
    return np.where(v, nd, np.nan).astype(np.float32)
```

Blocks are stitched as `stream_composites` stitches them (`wc._merge_max` for `max_abs_ndwi`). The private call is pinned by a test against `wc.stream_composites(rule="item")` on a synthetic cube. If the authors prefer a public API, the minimal backward-compatible alternative is an optional `valid_fn=None` argument to `wc.stream_composites` and `wc.scene_ndwi`.

**KeepCheck** (every scene and row block): at the dom positions inside the block, the reader's V and NDWI equal the package's (computed from the stored raw arrays with the same rule and mode), bitwise, NaN equal to NaN. In gate mode the check uses `item_valid` and raw NDWI on both sides.

NIDEM has no domain step, so one run serves both modes. Expected at N = all: Escalda, Wadden and Villaviciosa compute; the Ems is NOT COMPUTABLE (interval 1 = [2024-12-10], overcast). NIDEM is not run at Ferrol (no elevation truth, no lag). Learning curves: N = 10 is never computable; about 26-44 % of draws compute at N = 45 and 83-94 % at N = 90; at the Ems subsets that omit 2024-12-10 often compute.

**NIDEM confidence threshold** (tuning budget, §3.9): the filtered product at thresholds {0.15, 0.20, 0.25, 0.30} is recomputed from the stored `sd_mean` without a rerun.

### 3.4 `granadeiro`

| | Granadeiro et al. (2021) |
|---|---|
| FAITHFUL (v8 FROZEN) | [1] B03/B08 int16 / 10000; [11] intertidal mask SD(NDWI) > 0.2 (Eq. 2-3, ddof 0); [12-15] Eq. 1 heights from tide-table marks, rising/ebbing split at lag 0, h_c; [16] `convertToProp` per pixel; [17-18] lean nplr 4PL preliminary DEM on every mask pixel; [19-20] pool \|prelim − h_c\| ≤ 0.25 m, sample min(50,000, pool), `default_rng(20210119)`; [21] lags −90..+90 step 5, first argmin of \|h_rising − h_ebbing\|; [22] `mgcv::gam(lag ~ s(lon, lat))` defaults, `set.seed(20210119)`; [23] final fit at the GAM lag on converged pixels; [24] exposure (saved, not scored); lean-vs-`nplr()` proofs at N = all |
| CONTROLLED (`ctl`, the row) | common record/subset; common mask via `mask_raw_bands`; floor0 bands; inter-calibration OFF (identity coefficients, an exact identity of `g21.calibrate`); **tide input = exact h_B**: heights `LEV[0][sel]`, lag shifts `LEV[ℓ][sel]` for ℓ in the **common lag window −45..+180 step 5 (46 lags)**, final heights `LEV[round(gam_lag)][sel]` (rounded to the minute, as MAREA's τ_px); `marks_lp` only for the rising/ebbing split and h_c; failure policy §3.0; row-block loader |

**Variants** (§3.9 for sites and tiers):

| Variant | Differs from `ctl` by | Cost |
|---|---|---|
| `ctl_unpinned` | fits whose inflection lies within 5 mm of nplr's start value `(min x_used + max x_used)/2` are non-converged: removed from the pool and the final domain, and their lag fits count as failed in `choose_lags`; the GAM is refitted on the ctl sample minus pinned pixels (a uniform sample of the unpinned pool); final fits on scoring pixels | derived inside the ctl unit: no new prelim or lag fits |
| `eq1_90` (faithful tide and range) | Eq. 1 heights from `marks_raw` (v8), lag grid ±90 (37 lags), `marks_raw` for the split and h_c | full |
| `calon` ("native preprocessing") | v8 inter-calibration ON, with two declared rules: the reference is the paper criterion among the subset's paper-rule scenes, else the scene with the largest valid frame share; a scene with fewer than 1,000 sea or 1,000 land pairs, or an NA major axis, gets identity coefficients and is counted | full |
| `paper` | gate only: paper scene rule, no per-pixel mask, raw bands, calibration ON, Eq. 1, ±90 | full |

```python
def run(pkg, unit, variant="ctl", workers=16, tmpdir=None, rscript=None, max_r_hours=None, smoke=None):
    sel = unit.sel if variant != "paper" else sel_paper_rule(pkg)        # v8.scan_scenes + g21.scene_rule
    if len(sel) < 8: raise NotComputable("scenes_lt_8")
    tide = TideInput(pkg, sel, variant)          # ctl/unpinned/calon: exact LEV rows, LAG_CTL, marks_lp
                                                 # eq1_90/paper: g21.water_height / lag_heights on marks_raw, LAG_GRID_MIN
    L = MaskedBands(pkg, sel, mode="common" if variant != "paper" else "paper",
                    index=obs.INDEX_MODE if variant != "paper" else "raw", check=KeepCheck(pkg, sel))
    sd_unc = L.frame_ndwi_sd()                   # flat sub-blocks aligned to v8's 25,000 partition: bitwise
    coef = identity(len(sel)) if variant in ("ctl", "eq1_90") else calibration(L, sd_unc, rules)
    sd = sd_unc if coef is identity else L.frame_ndwi_sd(coef)
    mask_idx = np.flatnonzero(sd > SD_THRESHOLD)                       # 0.2 (tuning budget: 0.15, 0.30)
    N_pm = L.pixel_major("B08", mask_idx)                              # (M, T) int16, as v8's N_pm
    proj = project_r_hours(mask_idx, N_pm, tide, workers)              # §4.3; logged; may stop with "budget"
    # v8 main L1029-1183, re-sequenced with tide.h0 / tide.H_lag / tide.rising / tide.h_c / tide.grid:
    # prelim fit_logistic(x = tide.h0) -> proof (N = all) -> pool -> sample (default_rng(SEED)) -> lag_fits(tide.H_lag)
    # -> choose_lags(lag_grid = tide.grid) -> [< 60 unique located lags: fallback] -> gam_lag_surface at conv_idx
    #    (GAM RuntimeError: fallback "gam_failed") -> final fit with x_fn = tide.final_heights(gam_pred)
    # curve units: final fit only on conv ∩ scoring pixels (per-pixel identical values)
```

**F-mode** (diagnostic curves): `mask_idx`, the converged set and the pool are loaded from the site's N = all `ctl` unit (whose `product.npz` keeps them, with the prelim statuses and the lag sample, under v8's key names); the preliminary stage is skipped; the lag sample is drawn from that pool with `default_rng(20210119)`; lag fits, GAM and final fits use the subset's scenes. The runner orders F-mode units after the N = all unit of their site.

**`MaskedBands`** reads B03, B08 and SCL raw for the subset scenes in row blocks, applies `obs.floor_raw` (floor0) and, in common mode, `obs.mask_raw_bands(G, N, V)`. `frame_ndwi_sd` processes flat sub-blocks aligned to multiples of 25,000 over [0, H·W) (carrying partial sub-blocks across row blocks), so the partition equals v8's whole-frame call for any geometry, including the width-1 tail that changes `np.nanstd`'s summation path. `band_ranges` (calon/paper) is computed scene by scene. `pixel_major` gathers the mask columns. An in-RAM path identical to `v8.load_bands` is allowed when 4·T·H·W bytes is under 25 % of free RAM; both paths are tested equal.

**KeepCheck** (every scene and row block, at the dom positions inside the block):

- common mode: `(G_masked != fill) == V_pkg` and `(N_masked != fill) == V_pkg` (V already requires both bands and a finite NDWI), and the reader's float32 NDWI equals the package NDWI where V (bitwise);
- paper mode: only the raw NDWI against the package raw NDWI, NaN equal to NaN.

**Lag field:** `lag` = the GAM prediction at the converged pixels (the pixels where Granadeiro applies it); the GAM is predicted at exactly `conv_idx`, as v8. No extra prediction points.

**Diagnostics:** mask share of the frame and of keep_full; the share of fits pinned at nplr's start (prelim, lag, final) and how many pinned fits entered the pool and the sample; status counts; lag edge shares on the window used; GAM R², edf, n_unique; |Eq. 1 − h_B| at the record instants; calon: slopes and identity-coefficient scenes.

**Reused from v8:** `LazyRows`, `StageCache` (`--resume`), `wait_for_ram(label, need_gb, args)` with `args = SimpleNamespace(no_wait=..., min_free_gb=...)`, `auto_chunk`, `compare_fits`, `proof_selection`, `cube_grid`, `band_encoding`, `scan_scenes`, `minutes_since_epoch`, `to_utc_naive`; `g21.RRunner(workers, tmpdir, rscript)`; on the server `--rscript "$RSCRIPT"` and `--tmpdir $PWD/tmp`.

### 3.5 `geyman` (GM20, lag only)

| | Geyman & Maloof (2020) |
|---|---|
| FAITHFUL | per-pixel least-squares sinusoid of period 12.42 h (12.4206012) in hours since the reference station's high water; phase offset = tidal lag; uncertainty = IQR over 20 refits on random 70 % subsets; validation within 500 m of gauges |
| ADAPTED | intensity = NDWI (floor0) on the domain's keep pixels (A: keep_sel; F: keep_full); reference HW = HW marks of `marks_lp`; closed-form OLS (same optimum as Nelder-Mead); per-pixel valid instants V; primary lag referred to h_B's own fold phase on the same instants (`rel`); pre-registered QC; no spring-neap scaling in the primary |

```python
HW = pkg.marks_lp["t"][pkg.marks_lp["is_hw"]]
x = (t_min[sel] - HW[np.searchsorted(HW, t_min[sel], "right") - 1]) / 60.0
c, s = np.cos(2*np.pi*x/12.4206012), np.sin(2*np.pi*x/12.4206012)
phi_Y, amp, R2, n = fold_phase(Y, V)                     # closed form per column, chunks of 100,000 px
phi_B = fold_phase(np.broadcast_to(h0[sel][:, None], Y.shape), V)
tau_rel = wrap(phi_Y - phi_B); tau_raw = wrap(phi_Y)
qc = (n >= 20) & (sd(Y, ddof=0) > 0.2) & (0.05 < wet_share < 0.95) & (R2 >= 0.1)
lag = np.where(qc, tau_rel, np.nan)                      # lag_kind = "phase"
```

Variants: `rel` (primary); `raw` (scored against truth (c)); `scaled` (spring-neap scaling); `wet` (binary W); `own` (N = all: every package pixel inside the judges' neighbourhoods, sea included, with the QC minus the wet-share rule, for GM20's own use case on permanently submerged pixels). Curve units fit only the read-out pixels (nbhd ∩ domain, plus a seeded sample of at most 5,000 px per full-record band); per-pixel fits are identical either way.

### 3.6 `hartdavis` (HD26, lag only)

| | Hart-Davis et al. (2026) |
|---|---|
| FAITHFUL | harmonic least squares with nodal corrections for M2, N2, K2, O1 and Q1 of a level-proxy series after a linear detrend and a 3σ outlier filter; M2 phase; validated against gauge harmonic constants |
| ADAPTED | proxy = wet fraction ζ_k of I_k = domain keep ∩ domain band k (A-mode: subset bands; F-mode: full bands); a scene enters band k when at least 80 % of I_k is valid [D16; sensitivities 0.5 and 0.9]; primary lag = M2 phase of ζ_k minus that of h0 on the same instants (`rel`); variants `faithful` (minus the continuous M2 phase of h_B), `local` (I = nbhd ∩ keep) |

```python
def m2_phase(t_days, y):
    y = y - OLS line; keep |y - mean| <= 3 sd (one pass)
    if len(y) < 30 or span < 365 d: return NaN, flag            # per band; the unit fails only if every band fails
    amp, g = pyTMD.solve.constants(t_days, y - y.mean(), ["m2", "n2", "k2", "o1", "q1"], corrections="GOT")
    return g[0], se_g, VIF(M2), max|corr|
tau_k = wrap180(g_zeta - g_h0) * 60 / 28.9841042
lag = tau_k[band_of] on the domain keep pixels (NaN for band -1); lag_kind = "phase"
```

A VIF of the M2 pair above 2 is flagged, not gated.

### 3.7 `baselines`

- **Zero lag:** τ = 0 at every judge (the lag of no delay and NIDEM).
- **Global models at the gauge**, labelled lower bounds: EOT20 and GOT5.6 (`PyTMDBoundary(model, lat_g, lon_g, directory="tide_models")`; GOT5.6 only if `check-env` can load it; FES2022 and regional models are not available locally). τ_model by the (a), τ_ab and (b) machinery with the model's levels in place of the gauge; reported with `tidecheck.nearest_ocean_km` (20-33 km for the inner Ems and Westerschelde gauges, degenerate there by construction). An improvement over them is never quoted as evidence for a method.

### 3.8 As-published reference rows (`published`)

Read-only loaders, label "as published (own rules)": MAREA and no delay (`products_<site>_10m/comparison_<tag>/products.npz` and `marea_<tag>/`; Villaviciosa `products_marea/external_products.npz`), NIDEM primary (`v9.load_nidem`; NOT COMPUTABLE at the Ems), Granadeiro v8 (`v9.load_granadeiro`; Villaviciosa, Escalda, Ems). Scored by the benchmark scorer on the benchmark truth; never part of the controlled tables or their common sets.

### 3.9 Variants, sensitivities and tiers

Every row below runs every listed method under one change from the controlled setting; all other settings stay controlled. Tier 1 is required for the headline; Tier 2 is robustness, run in the listed order as server time allows [D20].

| Setting | Methods | N / modes | Sites | Tier |
|---|---|---|---|---|
| Controlled (primary) | all | all + curves (§6) | all | 1 |
| Positive control | all elevation and lag methods | all (+ N = 120, rep 0, A-mode) | pc | 1 |
| Scene-rule matrix: tz, g21, ard (common mask) | cheap groups | all, A-mode | all | 1 |
| Scene-rule matrix | Granadeiro ctl | all, A-mode | escalda, villaviciosa | 1 |
| Scene-rule matrix | Granadeiro ctl | all, A-mode | wadden, ems | 2 |
| Mask `marea_clear` | cheap groups | all, domain held at keep_full | all | 1 |
| Index `raw` | cheap groups | all, domain held at keep_full | all | 1 |
| Mask / index | Granadeiro ctl | all | escalda | 2 |
| MAREA `tau120`, bracket 0 (= z_raw), floors 80 / 0 | MAREA family | all; floors in A- and F-mode curves | all | 1 |
| Granadeiro `ctl_unpinned` | Granadeiro | all | all with ctl | 1 |
| Granadeiro `eq1_90` | Granadeiro | all | escalda, villaviciosa (then wadden, ems) | 2 |
| Granadeiro `calon` | Granadeiro | all | escalda, villaviciosa (then wadden, ems, ferrol) | 2 |
| GM20 `raw`, `scaled`, `wet`, `own`; HD26 `faithful`, `local`, 0.5 / 0.9 | lag-only | all | all with judges | 1 |
| Contiguous windows | cheap groups | curves, A-mode | all | 1 |
| F-mode curves | Granadeiro | GRAN_N_GRID × 3 | escalda | 2 |
| Granadeiro N = 330 × 3 reps | Granadeiro | A-mode | wadden, ems | 2 |
| Symmetric tuning budget (§5.4) | NIDEM (free), Granadeiro | all | escalda (development), then scored sites | NIDEM 1, Granadeiro 2 |

The sensitivities that change only the observation (mask, index) hold the domain at keep_full, so only the observation changes. The scene-rule matrix is A-mode, so each rule is "the archive" for every method.

---

## 4. Validation before scoring

### 4.1 Gates (reproduce production)

| Gate | Where | Criterion |
|---|---|---|
| G-record | all sites | Escalda, Wadden, Villaviciosa: record dates and h equal v7's primary `nidem.npz` (h bitwise) — locally `products_escalda/nidem_eot20_g2` (20 m); server `products_escalda_10m/nidem_eot20_g2`, `products_wadden_10m/nidem_eot20_g1`, `products_marea/nidem`. Ems: `products_ems_10m/nidem_eot20_g1/nidem_preflight.json` (n_scenes, LOT, HOT, edges, Z_j, U_j, the 5 lowest and highest h) bitwise. Ferrol (and all sites): [min, max] of h0 over the tz subset equals `marea_<tag>/result.json` `rango_marea_m` bitwise (to 1e-12 if G-levels found the provider not batch-invariant). Villaviciosa T = 465 |
| G-levels | all sites | §2.7 batch invariance; outcome recorded, fallback automatic |
| G-keep | all sites | at keep_prod, raw-mode NDWI from the stored arrays equals `ex["Y"][time_index]` and the `marea_clear` mask equals `ex["C"][time_index]`, bitwise (streamed by `v7.npz_rows`, one member at a time); production s_km recomputed (§2.6); package files rehashed |
| G-marea | Escalda, Ferrol, Villaviciosa at N = all; Wadden, Ems at N = 180 rep 0 (A-mode; at N = 120 a subset can hold too few clear scenes for a keep) | the adapter equals `marea.reconstruct(cube, out_dir=gates/<site>/marea_reference, pixel_m, boundary=pkg.provider(), extraction={"Y": ndwi[sel] at d.keep, "C": V, "keep": d.keep, "sea": d.sea, "inter": d.inter, "dates", "shape", "bbox" (from the production extraction, else `pyintertidal.sites.get(site).bbox`), "crs_wkt"}, mouth_side, min_scenes=min(40, N), overpass_times=over, n_boot=0)` bitwise on `tau_min`, `tau_px_min`, `z` (float32); no delay equals one `invert_series(min_bracket=3)` call. Same BLAS threads as the adapter; run in a separate process (reconstruct's float64 copies, about 5.5 GB at the Ems) |
| G-marea-legacy | local 20 m | the stored `products_escalda/marea_eot20_g2/{result.json, marea.npz}` and `comparison_eot20_g2/products.npz` on their 66-scene transition record, raw mode, `marea_clear`, `min_bracket = 0` (the scratchpad `repro_escalda20.py` logic, which matched bit for bit) |
| G-nidem | as G-record | the adapter in gate mode (`item_valid`, raw NDWI, pre-flight on keep_prod) reproduces v7's primary `nidem.npz` bitwise on `filtered_keep`, `unfiltered_keep`, `R_keep`, `conf_keep`, `count_keep`, `Z`, `U`, `edges`; at the Ems it reproduces the pre-flight stop (interval 1 = [2024-12-10]; 104 of 1,886,417 px valid at 10 m, 27 of 477,745 at 20 m) |
| G-granadeiro | Villaviciosa (server), paper mode, 98 scenes, 16 workers (~40 min) | against `products_marea/granadeiro2021/granadeiro2021.npz`: **exact** on the discrete outputs (`mask`, `mask_idx`, prelim statuses, pool size, `lag_sample_idx`, `lag_sample_lag` on the 5-min grid, final statuses) and on the Python-side calibration (`cal_B03`, `cal_B08`); **≤ 1e-9** on `prelim`, `final` (m) and `gam_lag` (min). The GAM is predicted at exactly v8's `conv_idx`. The run records both Rscript paths (the stored run used `/media/HDD/jalias/r-env/bin/Rscript`; the benchmark uses env.sh's `$RSCRIPT`) and R, nplr, lmodel2, mgcv and BLAS versions. If a discrete output differs, the authors may rerun the gate once with the stored run's Rscript (outside the project folder, so only with their permission) to separate code drift from platform drift; every benchmark Granadeiro run then uses one R, env.sh's |
| G-score | §5.8 | published metrics reproduced to 1e-9 |

`runner.validated(site, res_m, method)` requires the method's gates and its positive control.

### 4.2 Positive control (synthetic, every adapter, the same scorer)

`control.synth` builds site `pc` at 20 m from the Wadden 20 m record and provider (real record instants, the real h_B and its 1-min series, the real SCL pattern), with everything else planted (FROZEN `positive_control`, seed `SEEDS["pc"]`):

- **Frame** 120 × 640 px (2.4 × 12.8 km): open sea in columns 0-19; a channel (rows 52-67) from the sea eastward; tidal flats in rows 12-51 and 68-107, z rising linearly from the channel edge to the land edge over [LO + 0.15 R, HI − 0.15 R] of the record's h_B range; land in rows 0-11 and 108-119 (z = HI + 1 m); sea and channel z = LO − 1 m (always wet). Only the flats pass the range filter, so they are D_truth. Mouth side west; s ≈ along-channel distance.
- **Planted lag**, two scenarios: `zero` (τ ≡ 0) and `profile` (τ(s) = round(120 · min(s / 12 km, 1)) min).
- **Water:** local level h_loc(t_i, x) = LEV[τ(s(x))][i] (exact table rows); wet iff h_loc > z + e, e ~ N(0, 0.04 m) per observation.
- **Reflectance (DN):** water B03 ~ N(600, 150), B08 ~ N(80, 120) (turbid, about 25 % negative); exposed flat B03 ~ N(900, 150), B08 ~ N(1500, 300); land B03 ~ N(700, 100), B08 ~ N(3000, 400).
- **Clouds:** SCL = the Wadden 20 m cube's SCL on a fixed 120 × 640 window for the record scenes; SCL 8/9/10 → both bands ~ N(4000, 800); SCL 3 → bands × 0.3; SCL fill → band fill. **Missed haze:** 2 % of SCL-valid observations get +400 DN on both bands.
- **Outputs:** an int16 netCDF cube with the production layout (fill −32768, chunks (1, 120, 256)), `overpass_times.json`, the planted z (truth "planted") and τ (lag truth per band), and a site config whose boundary is the Wadden provider. `synth` builds the rows LEV[0..120] it needs with `levels.build_level_table` on the Wadden record instants; `prepare pc` then builds the package as for any site, and `control` asserts that both tables agree bitwise. `truth`, `run` and `score` run unchanged on `pc`.

**Pass criteria (pre-registered):**

- Elevation, on the in-range planted pixels: coverage ≥ 0.5, median |z − z_true| ≤ 0.10 m, |median bias| ≤ 0.05 m, OLS slope b ∈ [0.8, 1.2].
  - `zero`: no delay, MAREA, NIDEM, Granadeiro ctl must pass.
  - `profile`: MAREA and Granadeiro ctl must pass; no delay and NIDEM are reported only (the bias they show is the effect the benchmark measures).
- Lag, `profile`: MAREA, Granadeiro ctl, GM20 rel and HD26 rel: median over bands of |band read-out − planted band mean τ| ≤ 10 min; `zero`: every lag method's band read-outs within ±10 min of 0.
- Both scenarios at N = all (A = F) in the controlled setting; the `profile` scenario also at N = 120 rep 0 in A-mode (exercises the domain step; reported, not pass/fail).

A method that fails prints "implementation not validated" in every controlled table until the cause is found; the fix and the rerun are logged in `LOCK.json`. The synthetic cube is built once, on the laptop (`control --synth-only`), and its folder (`results/v13_benchmark_20m/control/`, about 0.1 GB) is copied to the server, so both machines score the same cube. The control runs locally for every method except Granadeiro, whose full control runs on the server (`--from-existing`, about 1-2 h with 16 workers); locally Granadeiro runs as a smoke with the lag sample capped at 1,000 (not a verdict). The server reruns every method's control for `LOCK.json`.

### 4.3 Granadeiro projection and real-data smoke

- **Projection, before any R stage, in every Granadeiro unit:** mask size, the median valid observations per mask pixel, the expected pool (from a 20,000-px seeded pre-sample of the preliminary fit when `--project-only`), and projected R hours = Σ stages (fits × seconds per fit at that observation count / workers), with seconds per fit from the logged costs (prelim ≈ 11-24 ms, lag ≈ 14 ms per fit per worker at 33-98 scenes, scaled linearly with observations per fit). Logged and saved in `run.json`; `--max-r-hours` stops with status `budget` (resumable).
- **Local real-data smoke (before phase C):** Granadeiro ctl end to end on a Villaviciosa 10 m window (about 300 × 400 px with flats, channel and land, recorded in `run.json`), 2 R workers, `--no-wait`. Not reportable; the authors read its diagnostics (pinned-fit share, SD-mask share, GAM R², LiDAR metrics on the window) before the server runs.

---

## 5. Truth and scoring

### 5.1 Elevation truth and scoring domains (built once per site by `truth`)

- **Vaklodingen** (Escalda, Wadden, Ems): `cells = v1.truth_grid(site)` (latest finite survey per 20 m RD cell in 2019-2025, NAP); every frame pixel centre transformed to EPSG:28992; a nearest-cell lookup with the arithmetic of `v1.sample_truth` (which returns only `(z, year)`) extended to return a cell id (sheet, iy, ix); a test pins z and year to `v1.sample_truth`.
- **LiDAR** (Villaviciosa): `truth, gt, gshape, water_share = v10.lidar_truth(rtk.DEFAULT_GRID)`; assert `gshape == (915, 915)`; peak about 1.2-1.5 GB; needs `data_v4/truth/pnoa/MDT02-WGS84-0015-3-COB2.tif` on the server.
- **RTK** (Villaviciosa): `dev = rtk.load_rtk()` (no arguments); its grid equals the cube grid (asserted, as v8 does); `flat = dev["row"] * W + dev["col"]`; z = `dev["elev_alicante"]` (`check-env` asserts `data/villaviciosa_rtk_gnss_alicante.csv` exists); blocks = `dev["block"]`. Not range-filtered (continuity).
- **Planted** (pc): the planted z.
- **Range filter** (method-independent, fixed for every method and N) [D11]: [min h0 − 0.25, max h0 + 0.25] over the full record.
- **Domains:**
  - **D_truth** (primary for coverage and own accuracy): frame pixels whose truth passes the range filter (RTK: every development point in the frame);
  - **D_keep** (secondary): D_truth ∩ keep_full;
  - common views are cut from D_truth (§5.3).
- **Support S** = D_truth ∪ keep_full ∪ ⋃ nbhd_g ∪ RTK pixels: every product is stored on S (`truth/<site>/support.npy`).
- **Files:** `truth/<site>/truth_elev.npz` (S-aligned `truth_raw`, `truth`, `year` (−1 none), `cell_id`, `band` (strata, §2.6), LO, HI, RTK arrays) and `truth_elev.json` (sources with sha256, counts excluded below and above, years).

### 5.2 Elevation metrics

```python
def stats(z, t, min_n):
    m = isfinite(z) & isfinite(t); z, t = z[m], t[m]; n = m.sum()
    zc, tc = z - np.median(z), t - np.median(t)
    rmse_c = sqrt(mean((zc - tc)**2))      # continuity with v9/v10 (== v7.vscore rmse_m)
    rmse_meanc = std(z - t)                # used for inference: immune to MAREA's mu-grid quantisation
    rmse_raw = sqrt(mean((z - t)**2))      # datum offset included (every method shares h_B's datum)
    bias = np.median(z - t); mean_err = mean(z - t)
    slope, intercept = np.polyfit(tc, zc, 1); r = corrcoef(tc, zc)[0, 1]
```

- Tables show coverage, rmse_c, rmse_meanc, **bias and rmse_raw with the same prominence**, b and r [D12].
- `MIN_N`: 30 for grid truths, 20 for RTK. Tests pin `stats == v7.vscore` and `== elevation_validation.ols_against_truth` to 1e-12.
- **Coverage** = |D ∩ finite z| / |D| on D_truth and D_keep. Reported with the clear observations behind the mapped pixels (median and p10 of `clear_obs`; NIDEM: the minimum per-interval count).
- **Accuracy-coverage curves** (N = all, descriptive): pixels of D_truth ∩ finite z_raw added in each method's own confidence order (`conf`); rmse_meanc against the cumulative share of D_truth at 5 % steps.

### 5.3 Views and strata

- **Views:**
  - `own_D` (D_truth ∩ finite z_X) and `own_K` (D_keep ∩ finite z_X);
  - **`common_q`**: D_truth ∩ finite z of every member, each with its own quality rule;
  - **`common_raw`** (co-primary): the same with z_raw (MAREA family with bracket 0, NIDEM unfiltered, Granadeiro all converged fits);
  - `pair:<A>|<B>`: every pair of elevation methods, both quality-ruled;
  - `cells` (co-primary at the Vaklodingen sites): each 20 m RD cell counted once; a product's value is the mean of its finite pixels in the cell when at least half of the cell's D_truth pixels are mapped; truth is the cell value;
  - `survey_ge_2023` (co-primary wherever n ≥ MIN_N and ≥ 25 blocks): Vaklodingen cells surveyed in 2023 or later.
  - Members are the elevation methods computable at that site and `(family, mode, N, rep)`; a common view without Granadeiro is shown while it is under review.
- **Strata:** `site`; `band:k` (§2.6 strata, `centers[k]` carried); `no_s`; survey year (Vaklodingen). RTK has a site stratum only.
- **Ems:** survey epoch about 2020 against 2023-2025 imagery; every Ems elevation row carries the flag "epoch-mismatched"; elevation numbers are never pooled across sites.

### 5.4 Elevation inference (pre-registered)

- **Block size from the data, per site** [D13]:
  1. for each primary contrast (below), e(x) = |d_A(x)| − |d_B(x)| on the `common_q` pixels;
  2. empirical semivariance of e on at most 20,000 seeded pixels (`SEEDS["variogram"]`), 25 distance bins up to 5 km;
  3. exponential model γ(h) = c0 + c1·(1 − exp(−3h/r)) fitted by least squares, r ∈ [pixel_m, 10 km];
  4. block size = the largest practical range r over the site's contrasts, rounded up to {100, 250, 500, 1000, 2000, 4000} m; 1 km if a fit fails, 4 km if r exceeds it.

  That size is used for every inferential statement at the site. 100 m blocks remain descriptive. Every CI reports n_blocks; with fewer than 25 blocks no CI is given (Villaviciosa and RTK are likely descriptive only). Band strata use blocks inside the band (block id × band).
- **Bootstrap:** `w_b = np.bincount(rng.integers(0, nb, nb), minlength=nb)` with `rng = default_rng([SEEDS["block_boot"], SITE_ID, truth_id, stratum_id, block_m])`; the same weights for every method and view, so comparisons are paired. B = 1000 for descriptive CIs and B = 5000 for the primary family (so that the smallest attainable p, 2/(B + 1), is below Holm's first threshold 0.05/48). Per method and view, `BlockSums` pre-computes block sums of 1, d = z − t, d², z, t, z², t², zt and the sorted orders of z, t and d with block labels; per draw, `n_w = w·cnt`, `rmse_c² = E_w[d²] − 2c·E_w[d] + c²` with `c = wmedian(z) − wmedian(t)`, `rmse_meanc` from E_w[d] and E_w[d²], slope and r from the weighted moments, bias = `wmedian(d)`, coverage = weighted mapped count / weighted domain count. `wmedian` equals `np.median` of the multiset expanded by the integer weights (`cw = cumsum(w)`, ranks `(W−1)//2` and `W//2` located with `searchsorted(cw, rank, "right")`), so unit weights reproduce the point estimate exactly.
- **Primary family** (the only inferential claims), N = all:
  - sites: Escalda, Wadden, Ems (flagged) — Vaklodingen; Villaviciosa — LiDAR;
  - contrasts: MAREA − no delay, MAREA − NIDEM, MAREA − Granadeiro (absent where a method is NOT COMPUTABLE or not validated);
  - metrics: rmse_meanc and |1 − b|;
  - views: `common_q` and `common_raw`.

  At most 48 tests; two-sided bootstrap p = min(1, 2·(1 + #{δ*_b on the far side of 0})/(B + 1)); **Holm** at α = 0.05 over the whole family.
- **A headline elevation result** for A over B at a site requires all of:
  1. Holm-significant in both views, same sign;
  2. |Δ̂| ≥ the practical margin in both views: 2 cm for rmse_meanc, 0.05 for |1 − b|;
  3. the same sign in ≥ 90 % of A-mode reps at N = 330 where both methods have reps (otherwise flagged "scene-sampling spread not assessed": no headline);
  4. the same sign in every scene-rule matrix row available at that site (otherwise "flips across scene rules": no headline);
  5. both methods validated (§4).

  Everything else is descriptive (CIs printed, no significance marks). Every paired CI is printed next to the scene-sampling spread of the same difference over the A-mode reps at N = 330 (share of reps with the same sign, IQR), because a block CI of a fixed product ignores which 3-year record was observed.
- **Symmetric tuning budget** (Tier: NIDEM 1, Granadeiro 2) [D21]: Escalda is the development site for the competitors, as these sites were for MAREA. NIDEM's confidence threshold ∈ {0.15, 0.20, 0.25, 0.30, unfiltered} and Granadeiro's SD threshold ∈ {0.15, 0.20, 0.30} × lag window ∈ {±90, −45..180} are chosen by Escalda's rmse_meanc on `common_raw`; the chosen settings run as "tuned" rows at the other sites. MAREA's site-informed settings are shown by its own sensitivities (`tau120`, bracket 0, floors 80 / 0, transition-zone record) and stated in the table notes: TAU_BOUNDS was widened after the Harlingen comparison, the bracket rule followed Wadden failures, and the 3-km bands, the 40-scene floor and the transition-zone rule were chosen on these sites.
- **Confirmatory evaluation:** the benchmark runs on MAREA's development data. A confirmatory comparison needs data outside development: the sealed RTK split, when the authors decide to open it (only they run that script), or a new site. It is planned, not run.

### 5.5 Lag truths (per judge, fixed at the full record)

`g` = the judge's series as delivered. **(a), τ_ab and (b) are co-primary**; (c) is diagnostic [D9].

**(a) Best constant lag at the record's S2 instants** (as `lag_estimator_agreement.py` L441-484):

```python
g_s2 = g.reindex(pkg.t, method="nearest", tolerance=pd.Timedelta("6min"))
E(tau) = sqrt(mean_i (d_i - median(d))**2), d_i = LEV[tau][i] - g_s2[i]       # tau in -60..240, 1 min
tau_a = argmin E;  I_a = [min, max] of {tau : E(tau) <= E(tau_a) + 0.01}
```

**τ_ab (gain-aware):** argmin over τ of min over (a, b) of ‖g_s2 − a − b·LEV[τ]‖ (RMS, median-centred as E); I_ab from its own 1-cm tolerance.

**(b) M2 harmonic phase lag:** centred 10-min series (RWS as delivered; 1-min → centred 10-min mean, ≥ 6 samples; 5-min → ≥ 2); `pyTMD.solve.constants` with CONS_B (52 constituents, GOT corrections, order 1) on the instants where both gauge and hB10 are finite; τ_b = wrap180(φ_g,M2 − φ_B,M2)·60/28.9841042; SE by jackknife over 12 quarter-years; **I_b = τ_b ± 1.96·SE**.

Also recorded: 200 month-block bootstraps of (a) (`SEEDS["truth_boot"]`), edge flags, τ_a with gaps ≤ 20 min interpolated, τ_all over every 10-min sample, E0, Emin, A_M2 of gauge and h_B, M4/M2, HW-only and LW-only lags. **(c)** GM20 frame (raw fold phase of g_s2 and minus that of h0) and HD26 oracle (M2 phase of g_s2 minus that of h0 on the same instants).

Light-check values (20 m, regression references): Harlingen (a) 82 / (b) 91.3; Terschelling W 45 / 44.3; Gat van Borssele 10 / 9.8; Terneuzen 18 / 18.7; Delfzijl 61 / 68.4.

### 5.6 Lag read-outs (in the scorer, one rule for every method) [D8]

Every adapter outputs its applied lag field on S; `readout.py` derives the read-outs identically for every method (circular statistics for `lag_kind = "phase"`):

| Support | Definition |
|---|---|
| **local** (primary) | median of the lag over nbhd_g ∩ {lag defined}; at least 30 px, else no read-out |
| band | median over the strata band k_g ∩ {lag defined} |
| band_interp | linear interpolation in s at s_g between the strata-band medians |
| s_profile (MAREA only) | `smooth_lag_profile(centers, tau_used, [s_g], 3.0)`, a continuity column with v10, never ranked |

- **Applied lag:** MAREA = τ_px (0 at isolated pixels, which is what it applies; the isolated share of the read-out pixels is a flag); Granadeiro = the GAM at its converged pixels; GM20 = the QC'd field; HD26 = τ_k on its band pixels.
- **No read-out** (no defined pixel, or fewer than 30) → **fallback 0** in the primary scores, flagged.
- **Identified-only sensitivity:** a read-out is excluded when it is a fallback, or when more than half of its pixels lie in a MAREA band flagged unidentified, or the product carries a `fallback`.
- **Window flags:** the share of read-out pixels at the edge of the method's window (Granadeiro: first or last lag of the grid; MAREA: `bound_hit` bands; GM20/HD26: outside [−45, 180]) is reported for every method.

### 5.7 Lag metrics and inference

Per site × judge × method × variant × support × truth:

- `err = τ_m − τ_truth` (wrapped to ±P_M2/2 for phase-type lags) and **`err_int`** = 0 if τ_m ∈ I_truth, else the distance to the nearest end of I_truth;
- `E_m = E(τ_m)`, `dE`, `explained = 1 − (E_m/E0)²` (descriptive, on the (a) curve);
- `n_px`, `n_scenes`, flags (fallback, window edge, isolated share, sparse).

**Aggregates:** per site MAE_int, bias, max |err|, n; **pooled = the mean of the site MAEs** (each site weight 1), shown next to the per-site rows; never pooled across different judge sets. Judge sets: `headline` (§2.8 inclusion; Terneuzen pair as one location; expected 13 locations: Escalda 2, Wadden 2, Ems 8, Ferrol 1), `all`, `without_mouth`.

**Inference:** for MAREA against each of zero lag, EOT20, Granadeiro, GM20 and HD26, an exact two-sided sign test across headline locations on |err_int| (ties dropped), Holm across the five comparisons, separately under (a), τ_ab and (b). **A ranking is stated only when it holds, Holm-significant and in the same direction, under all three truths**; otherwise "no ranking". With 13 locations a single comparison needs 11 of 13 (two-sided p = 0.022), and under Holm over five comparisons the smallest p must be at most 0.01, i.e. 12 of 13. Locations within one estuary share the boundary and are not independent (the Ems holds 8 of the 13), so every test prints its wins per site next to the pooled count. No method-side CIs enter any comparison: Granadeiro has no affordable one, so for symmetry no method's does; MAREA's scene bootstrap is a diagnostic file only. Truth-side uncertainty enters through `err_int`.

**Continuity column** (optional): v10's gauge-pair E(τ) for the IOC pairs (brsk/trnz, ters/harl, bork/delf, fer1/fer2).

### 5.8 Scorer reproduction gate (G-score)

Rescore the published products with the legacy filter (z_plain ± 0.25), `validation_grid.thin_mask` and `v7.vscore` / `ols_against_truth`, and assert equality to 1e-9 with: `products_<site>_10m/comparison_<tag>/vaklodingen_metrics.csv` and `vaklodingen_by_band.csv`; `vaklodingen_metrics_with_nidem.csv`; `results/method_comparison/<site>.csv` (v9); `products_marea/rtk_onsite_metrics.csv`; v10 `numbers.json` LiDAR entries; locally `products_escalda/comparison_eot20_g2/vaklodingen_metrics.csv` and `..._with_nidem.csv`. Only then are the benchmark domains and filter used.

---

## 6. Learning curves

- **N grid:** 10, 20, 30, 45, 60, 90, 120, 180, 240, 330 and all [D5]. Granadeiro: 20, 45, 90, 180, 330 [D6].
- **Families and modes:**
  - random nested subsets (§2.9): A-mode 10 reps (primary), F-mode 5 reps (diagnostic), for marea_pair, GM20 and HD26; NIDEM 10 reps (mode-independent);
  - contiguous windows: A-mode, 5 reps, cheap groups (sensitivity);
  - Granadeiro: A-mode 3 reps at Escalda and Villaviciosa (Tier 1); F-mode 3 reps at Escalda and N = 330 × 3 reps at Wadden and Ems (Tier 2);
  - MAREA family floors 80 and 0: both modes, 5 reps (cheap: only N < 80 is affected, and in A-mode the extraction rule usually stops those runs first).
- **Re-estimated at each unit:** the domain (A-mode, §3.1) and everything that depends on scenes (inversion grid, m2a lags, τ_px; NIDEM intervals, composites, TIN; Granadeiro mask, prelim, pool, sample, lag fits, GAM, final; GM20 and HD26 fits).
- **Fixed:** the scoring domains, strata, judges and neighbourhoods, truths, range filter, level table (rows indexed by `sel`).
- **Own thresholds unchanged at every N:** MAREA family: extraction rule, 3,000 px, floor 40, min_obs 8, the 3/3 bracket, m2a's 6 clear observations and 50 columns; NIDEM: empty interval and the 1 % pre-flight; Granadeiro: 8 scenes, its failure policy; GM20: n ≥ 20 per pixel; HD26: 30 points over 365 days per band. Nothing is relaxed or imputed. NOT COMPUTABLE counts as coverage 0.
- **Granadeiro pixel economy:** per-pixel steps (final fit) may be evaluated only on scoring pixels (identical values); population steps (SD mask, prelim over the whole mask, pool) always run on the full domain.
- **Aggregation [D17]:**
  - per (method, N): k computable reps of R; a point is shown when **k/R ≥ 0.5** (the same fraction for every method);
  - coverage: median and IQR over all reps (NOT COMPUTABLE = 0);
  - accuracy (pairwise and common views): median and IQR over the reps where **all compared methods** are computable, with that count;
  - IQRs are conditional on the 2023-2025 record (overlapping subsets near T shrink them) and are labelled so;
  - lag curves: MAE_int against N per truth, per site, fallback-as-zero (primary) and identified-only;
  - curves are never pooled across sites or judge sets.
- **Views:** elevation `own_D` (coverage and own accuracy), every `pair:`, and `common_q` with fixed membership (all elevation methods of the site; reps where one is not computable are dropped from that view); `clear_obs_median` next to N.
- **Figures** are labelled "A-mode: the archive has N images" or "F-mode: N images from the full archive".

---

## 7. Outputs, CLI, server commands, local test plan

### 7.1 Output tree (new folders only)

```
results/v13_benchmark_<res>m/
  FROZEN.json, LOCK.json
  common/<site>/...                         §2.10
  truth/<site>/truth_elev.{npz,json}, support.npy, truth_lag.json, truth_lag_curves.npz
  gates/<site>/<gate>.json                  PASS/FAIL + details
  control/<scenario>/...                    synthetic package, runs, scores, verdict.json
  runs/<site>/<method>[__<variant>]/<family>-<mode>/N<NNN>_r<RR>[__<senstag>]/{product.npz, run.json, log.txt}
  runs/<site>/_domain/, runs/<site>/<group>/_cache/
  scores/*.csv; tables/*_rows.tex, tables/numbers.json; figures/*.png; logs/
```

`report --paper` (10 m only) also copies `tables/` to the new folder `docs/paper/tables/benchmark/`; it never touches v10's tables.

### 7.2 CSV schemas (long format)

| File | Columns |
|---|---|
| `status.csv` | site, res_m, group, method, variant, family, mode, N, rep, sens, status, stage, reason, fallback, n_scenes, n_keep_sel, clear_obs_median, seconds, projected_r_hours, peak_rss_mb, package_key, subset_key, code_key |
| `elev_points.csv` | site, truth, view, members, stratum, band_s_km, method, variant, family, mode, N, rep, sens, status, fallback, n, n_domain, coverage, clear_obs_med, clear_obs_p10, rmse_c, rmse_meanc, rmse_raw, bias, mean_err, slope, intercept, r, std_z, std_t |
| `elev_boot.csv` | site, truth, view, stratum, method, variant, block_m, block_rule, n_blocks, B, metric, estimate, ci_lo, ci_hi |
| `elev_primary.csv` | site, truth, view, contrast, metric, delta, ci_lo, ci_hi, p_boot, p_holm, margin, passes_margin, rep_share_same_sign, rule_matrix_same_sign, validated, headline |
| `elev_paired.csv` | site, truth, view, stratum, block_m, method_a, method_b, metric, delta, ci_lo, ci_hi, n_common, n_blocks, rep_share_same_sign_N330, rep_iqr_N330 (descriptive) |
| `acc_cov.csv` | site, truth, method, variant, share_of_D, rmse_meanc, n |
| `variogram.csv` | site, truth, contrast, n_px, c0, c1, range_m, fit_ok, block_m_chosen |
| `lag_truth.csv` | site, judge, location, included, exclusion_rule, flags, k_g, s_g_km, d_min_km, R_g_m, n_nbhd, n_instants, tau_a, I_a_lo, I_a_hi, tau_ab, I_ab_lo, I_ab_hi, tau_b, se_tau_b, E0_m, Emin_m, tau_all, A_M2_gauge, A_M2_B, tau_c_raw, tau_c_rel, tau_hd26_oracle, eot20_*, got56_*, ocean_km |
| `lag_scores.csv` | site, judge, location, method, variant, family, mode, N, rep, sens, support, truth, tau_method, tau_truth, err, err_int, E_method_m, dE_m, explained, n_px, fallback, window_edge_share, isolated_share, flags |
| `lag_summary.csv` | scope, method, variant, support, truth, judge_set, view (primary / identified_only), N, n_locations, MAE_int, bias, max_abs |
| `lag_signtests.csv` | truth, judge_set, method_a, method_b, n_a_better, n_b_better, n_ties, p, p_holm, ranking_all_truths |
| `curves_elev.csv` | site, truth, view, method, variant, family, mode, N, R, k, k_joint, coverage_{med,q25,q75}, rmse_c_{...}, rmse_meanc_{...}, bias_{...}, slope_{...}, r_{...}, clear_obs_median |
| `curves_lag.csv` | site, method, variant, family, mode, support, truth, view, N, R, k, MAE_int_{med,q25,q75}, bias_{...}, fallback_share |
| `control.csv` | scenario, method, criterion, value, tolerance, pass |

### 7.3 LaTeX rows and figures (v10 style)

- Helpers `fmt`, `fmt_n`, `scenes` from `experiments/v10_validation_report.py`.
- Group headers carry the site, the common-set size and members; status cells print "not computable (<stage>)", "fallback (<reason>)" or "implementation not validated".
- Tables:
  - `bench_vaklodingen_rows.tex`, `bench_lidar_rows.tex`: method & scenes & n & coverage (D_truth) & RMSE_c & RMSE_meanc & bias & RMSE_raw & b & r \\, then the common_q and common_raw blocks; notes state MAREA's development history (§5.4);
  - `bench_rtk_rows.tex` (descriptive);
  - `bench_primary_rows.tex`: the primary family with Holm p, margins and the headline verdict;
  - `bench_lag_rows.tex`: one row per location: τ_a [I_a], τ_ab, τ_b ± SE, then err_int of zero / EOT20 / MAREA / Granadeiro / GM20 / HD26 under (a); then per-site MAE_int rows per truth and the sign-test verdicts;
  - `bench_matrix_rows.tex`: method × scene rule (all, tz, g21, ard) at each site;
  - `bench_curves_rows.tex`: per site × method: coverage and pairwise ΔRMSE at N = 120, 180, 330 and all, A-mode, with k/R.
- `numbers.json` holds every number the text quotes.
- Figures (`experiments/paper_style.py` colours): `curves_elev_<site>_<mode>.png`, `curves_lag_<site>_<mode>.png`, `acc_cov_<site>.png`, `lag_profiles_<site>.png` (band lags vs s per method, judges' I_a and τ_b), `elev_maps_<site>.png` (optional), `control_<scenario>.png`.

### 7.4 CLI (`python -X utf8 -m experiments.v13_benchmark <command>`)

| Command | Arguments | Does |
|---|---|---|
| `check-env` | `--res` | versions; Rscript and R packages (absent → warning, Granadeiro group skipped); GOT5.6 loadable; TMPDIR inside the project; free space on the project disk and in TMPDIR (≥ 20 GB + package size); every input present (cubes, extractions, overpass, IOC chunks, estuary_truth, Vaklodingen sheets, MDT02, RTK files incl. the Alicante CSV); cube encodings; 5 sampled scenes per cube: no non-fill value below −1000, SCL codes in 0..11, SCL 1 count; git HEAD |
| `prepare` | `site\|all --res [--steps ...] [--force]` | §2 (resumable per step, atomic writes) |
| `truth` | `site\|all --res` | §5.1, §5.5 |
| `gate` | `site\|all --res --gate record,levels,keep,marea,marea_legacy,nidem,granadeiro,score [--rscript P]` | §4.1 |
| `control` | `--scenario zero,profile [--groups ...] [--r-workers k] [--smoke] [--synth-only] [--from-existing]` | §4.2: synth → prepare → truth → run → score → verdict |
| `lock` | `--res` | §2.11 |
| `run` | `site\|all --res --groups ... [--variant ...] --family rand\|win\|rule-tz\|rule-g21\|rule-ard (comma list allowed) --mode A\|F --N all\|grid\|gran\|10,20 --reps 0-9 [--sens mask=marea_clear\|index=raw\|floor=80\|...] [--worker-id w] [--r-workers 16] [--rscript P] [--tmpdir D] [--resume] [--max-r-hours h] [--project-only] [--smoke] [--window r0:r1,c0:c1] [--no-wait] [--boot-threads 1] [--steal-stale]` (and `--blas-threads`, tests and smoke only) | queue of units |
| `status` | `[site] --res` | done, not computable, fallback, r_failed, error, budget, claimed, pending |
| `score` | `site\|all --res [--no-boot] [--allow-smoke]` | §5 → `scores/` |
| `report` | `--res [--paper] [--granadeiro-status TEXT]` | §7.3 |

**Resumability:** a unit is skipped when its `run.json` exists with status in {ok, empty, not_computable} and an equal `inputs_key`. Units are written to `….tmp-<host>-<pid>/`; an existing unit directory (a failed attempt) is first renamed to `<dir>.old-<utc>`, then the temporary directory is published with `os.replace`. Claim files `.claim` (`O_CREAT | O_EXCL`, host, pid, start; heartbeat every 60 s); a claim older than 30 min whose pid is gone is stale and `--steal-stale` takes it. Inside Granadeiro units v8's `StageCache` (`--resume`) keeps finished R stages. `run` refuses a group whose gates have not passed, and `score` refuses real-data products before `LOCK.json` exists.

**Logging:** per unit `log.txt` (stage timings, peak RSS: Linux `ru_maxrss` of the process, Windows `psutil` `peak_wset`; for Granadeiro `RRunner.stats` tree peaks); every command also prints to stdout, captured with `tee` into `logs/`.

### 7.5 Server commands (the user runs them; nothing is run by the assistant)

```bash
# 0. on the laptop: the authors commit and push the code; copy the gitignored inputs
rsync -av data_v4/gauges/estuary_truth/ <server>:/media/HDD/jalias/Intertidal_analysis/data_v4/gauges/estuary_truth/
rsync -av data_v4/truth/pnoa/           <server>:/media/HDD/jalias/Intertidal_analysis/data_v4/truth/pnoa/
rsync -av results/v13_benchmark_20m/control/ <server>:/media/HDD/jalias/Intertidal_analysis/results/v13_benchmark_20m/control/

# 1. on the server
tmux new -s v13
cd /media/HDD/jalias/Intertidal_analysis && git pull && source env.sh
mkdir -p tmp logs/v13 && export TMPDIR=$PWD/tmp
$PY -X utf8 -m pytest tests/test_benchmark_*.py -q 2>&1 | tee logs/v13/pytest.log
$PY -X utf8 -m experiments.v13_benchmark check-env --res 10 2>&1 | tee logs/v13/check_env.log
for s in escalda ferrol villaviciosa wadden ems; do
  $PY -X utf8 -m experiments.v13_benchmark prepare $s --res 10 2>&1 | tee -a logs/v13/prepare_$s.log
  $PY -X utf8 -m experiments.v13_benchmark truth   $s --res 10 2>&1 | tee -a logs/v13/truth_$s.log
done

# 2. gates, the Granadeiro gate, and the positive control of every method (Granadeiro alone on the machine)
$PY -X utf8 -m experiments.v13_benchmark gate all --res 10 --gate record,levels,keep,marea,nidem,score 2>&1 | tee -a logs/v13/gates.log
$PY -X utf8 -m experiments.v13_benchmark gate villaviciosa --res 10 --gate granadeiro --r-workers 16 \
    --rscript "$RSCRIPT" --tmpdir $PWD/tmp 2>&1 | tee -a logs/v13/gate_granadeiro.log
$PY -X utf8 -m experiments.v13_benchmark control --from-existing --scenario zero,profile --groups marea_pair,nidem,gm20,hd26,granadeiro --r-workers 16 \
    --rscript "$RSCRIPT" --tmpdir $PWD/tmp 2>&1 | tee -a logs/v13/control_granadeiro.log
$PY -X utf8 -m experiments.v13_benchmark lock --res 10 2>&1 | tee -a logs/v13/lock.log

# 3. phase A: N = all, controlled + Tier-1 sensitivities of the cheap groups (3 panes, worker-id 1..3)
$PY -X utf8 -m experiments.v13_benchmark run all --res 10 --groups marea_pair,nidem,gm20,hd26,baselines,published \
    --family rand --mode A --N all --worker-id 1 2>&1 | tee -a logs/v13/phaseA_w1.log
#    then: --family rule-tz / rule-g21 / rule-ard; --sens mask=marea_clear; --sens index=raw; --variant tau120
$PY -X utf8 -m experiments.v13_benchmark score all --res 10 2>&1 | tee -a logs/v13/score_A.log

# 4. phase B: curves of the cheap groups (5 panes, worker-id 1..5, overnight)
$PY -X utf8 -m experiments.v13_benchmark run all --res 10 --groups marea_pair,nidem,gm20,hd26 --family rand --mode A \
    --N grid --reps 0-9 --worker-id 1 2>&1 | tee -a logs/v13/phaseB_w1.log
#    then: --mode F --reps 0-4; --family win --mode A --reps 0-4; --sens floor=80 / floor=0 (both modes, reps 0-4)

# 5. phase C: Granadeiro N = all, one site at a time, alone (Tier 1)
for s in escalda villaviciosa ferrol wadden ems; do
  $PY -X utf8 -m experiments.v13_benchmark run $s --res 10 --groups granadeiro --variant ctl --family rand --mode A \
      --N all --r-workers 16 --rscript "$RSCRIPT" --tmpdir $PWD/tmp --resume 2>&1 | tee -a logs/v13/gran_$s.log
done

# 6. phase D: Granadeiro curves (A-mode, 3 reps) and scene-rule matrix at escalda and villaviciosa (Tier 1)
$PY -X utf8 -m experiments.v13_benchmark run escalda,villaviciosa --res 10 --groups granadeiro --variant ctl \
    --family rand --mode A --N gran --reps 0-2 --r-workers 16 --rscript "$RSCRIPT" --tmpdir $PWD/tmp \
    --resume 2>&1 | tee -a logs/v13/gran_curves.log
$PY -X utf8 -m experiments.v13_benchmark run escalda,villaviciosa --res 10 --groups granadeiro --variant ctl \
    --family rule-tz,rule-g21,rule-ard --mode A --N all --r-workers 16 --rscript "$RSCRIPT" --tmpdir $PWD/tmp \
    --resume 2>&1 | tee -a logs/v13/gran_matrix.log

# 7. phase E (Tier 2, in the order of §3.9 as time allows); 8. final score and report
$PY -X utf8 -m experiments.v13_benchmark score all --res 10 && $PY -X utf8 -m experiments.v13_benchmark report --res 10 --paper
```

### 7.6 Local test plan (laptop, 20 m; Escalda first)

The laptop has 7.4 GB of RAM with about 1.5 GB free. **R is installed**, contrary to the brief: `g21.find_rscript()` returns `C:/Program Files/R/R-4.4.3/bin/Rscript.exe` (nplr 0.1-8, lmodel2 1.7-4, mgcv 1.9-1). Without an Rscript, `check-env` warns and the Granadeiro group is skipped. One heavy unit at a time.

```powershell
$PY = "C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe"
& $PY -X utf8 -m pytest tests/test_benchmark_*.py -q                                   # synthetic, < 5 min
& $PY -X utf8 -m experiments.v13_benchmark check-env --res 20
& $PY -X utf8 -m experiments.v13_benchmark prepare escalda --res 20                    # two passes, ~10 min; ~0.25 GB
& $PY -X utf8 -m experiments.v13_benchmark truth escalda --res 20
& $PY -X utf8 -m experiments.v13_benchmark gate escalda --res 20 --gate record,levels,keep,marea_legacy,nidem,score
& $PY -X utf8 -m experiments.v13_benchmark gate escalda --res 20 --gate marea           # separate process, N = 180 rep 0
& $PY -X utf8 -m experiments.v13_benchmark prepare wadden --res 20 --steps record,series   # base of the control
& $PY -X utf8 -m experiments.v13_benchmark control --synth-only --scenario zero,profile
& $PY -X utf8 -m experiments.v13_benchmark control --from-existing --scenario zero,profile --groups marea_pair,nidem,gm20,hd26
& $PY -X utf8 -m experiments.v13_benchmark control --from-existing --scenario profile --groups granadeiro --smoke --r-workers 2
& $PY -X utf8 -m experiments.v13_benchmark run escalda --res 20 --groups marea_pair,nidem,gm20,hd26,baselines --N all
& $PY -X utf8 -m experiments.v13_benchmark run escalda --res 20 --groups marea_pair,gm20,hd26 --family rand --mode A --N grid --reps 0-9
& $PY -X utf8 -m experiments.v13_benchmark run escalda --res 20 --groups marea_pair --family rand --mode F --N grid --reps 0-4
& $PY -X utf8 -m experiments.v13_benchmark run escalda --res 20 --groups nidem --N 45,90,180 --reps 0-2
& $PY -X utf8 -m experiments.v13_benchmark score escalda --res 20 --allow-smoke
& $PY -X utf8 -m experiments.v13_benchmark report --res 20                               # never writes to docs/
```

Then at 10 m (plumbing, not reportable): Villaviciosa (`prepare`, `run --groups marea_pair,nidem,gm20,hd26 --N all`: mouth `any`, the 10-year extraction slice, `PyTMDBoundary`; LiDAR truth only with about 1.5 GB free) and Ferrol (`run --groups marea_pair,gm20,hd26,baselines --N all`). **Granadeiro locally:** pure-Python parts without R (masked loader, aligned SD sub-blocks, identity calibration, TideInput, pool/sample logic, pinned-fit logic, failure policy before R); R parts on synthetic pixels; the Villaviciosa window smoke (§4.3). **Smoke mode** caps the lag sample, uses `--no-wait` (v8's RAM guard defaults to 2.5 GB free), marks `run.json` with `smoke: true`, and the scorer refuses smoke products without `--allow-smoke`.

---

## 8. Tests (pytest, synthetic, no real cube unless marked)

Style as `tests/test_waterline_contour.py`: tiny netCDF cubes in `tmp_path` (int16, `_FillValue = -32768`, chunks (1, 8, 8)); `needs_r = pytest.mark.skipif(not _have_r(), ...)` as in `test_granadeiro2021.py`. `tests/test_r1_guard.py` already scans `experiments/` and therefore guards the new package against the reserved-split keyword.

| File | Tests |
|---|---|
| `test_benchmark_obs.py` | raw-mode `ndwi` equals `marea.extract` and `wc.scene_ndwi` bitwise (random int16 with fill, zero sum, negative bands); floor0: NDWI in [−1, 1], both-≤0 → NaN → not valid, fill preserved by `floor_raw`; `mask_valid` truth table for SCL 0..11 and 255 under both rules; `BLOCK_SCL`/`CLEAR_SCL` asserts; `decode` == xarray decoding; V == mask & finite; wet under floor0 == (B03 > B08) where defined; `mask_raw_bands`; `nir` == `g21.to_reflectance` on floored values |
| `test_benchmark_package.py` | record rule; level table == per-shift calls with a fake provider returning time-ordered values; beyond-table `at()` calls the provider per shift; non-batch-invariant fake provider → fallback; two passes reproduce `marea.extract` (raw, marea_clear) bitwise at keep_prod; `U_rec` ⊇ `keep_sel` for 200 random subsets of a synthetic cube (property test); the manifest key changes with any data byte and not with a scoring-section edit; size+mtime check; `load()` refuses a tampered file |
| `test_benchmark_domain.py` | `for_subset(arange(T)) == full` bitwise; A-mode sea from frame bits == sea from full frame values; `no_intertidal` and `no_anchor` statuses; `band_map_frame` merge rule; A-mode equals a direct `marea.extract`-rule computation on the subset's frames |
| `test_benchmark_marea.py` | adapter == `marea.reconstruct(n_boot=0)` bitwise (tau, tau_px, z) on a synthetic extraction with a fake provider; τ groups of 1 column and of 4,001 columns (slice rule); `z == where(bracket, z_raw)` == `invert_series(min_bracket=3)`; `m2a_columns` replay == `m2a_rasch`'s draw; planted band lag of 250 min uses a live per-shift call, never capped; planted lags (0/30/60/90 min) recovered within 5 min; unidentified flags; floor and `no_intertidal` give `not_computable` for both products; fallback flags |
| `test_benchmark_nidem.py` | adapter with `item_valid`/raw == `wc.stream_composites(rule="item")` bitwise; common mask/floor0 == `wc.composites_from_arrays`; pre-flight stop without opening the cube; empty interval; KeepCheck raises on a one-pixel mismatch, in both modes |
| `test_benchmark_granadeiro.py` | identity calibration is exact; aligned SD sub-blocks == whole-frame `frame_ndwi_sd` bitwise, including a width-1 tail; masked loader == `v8.load_bands` + `mask_raw_bands`; KeepCheck passes on a band-fill observation with valid SCL (feasibility blocker case) and in paper mode; TideInput: ctl heights == LEV rows, lag window −45..180, rising from `marks_lp`; pinned-fit detection; failure policy (pool empty, < 60 unique, GAM RuntimeError → fallback; other R rc ≠ 0 → `r_failed`), all before or around a fake RRunner; [needs_r] paper mode on a tiny synthetic site == `v8.main` on the same site (monkeypatched `v8.SITES`, output to tmp) |
| `test_benchmark_lagmethods.py` | GM20 `rel` and HD26 `rel` recover a planted 75-min lag within ±5 min on a synthetic tide at S2-like instants; GM20 sign and wrap; HD26 per-band failure → NaN + flag, unit fails only if all bands fail |
| `test_benchmark_truth.py` | (a) and τ_ab on h_B(t − 75) + offset (+ gain for τ_ab) + noise → 75 ± 1 with intervals containing 75; (b) on a 10-min delayed series → 75.0 ± 0.2 (pyTMD pin); nearest-within-6-min; inclusion rules on synthetic gauges (coverage, boundary member, dry); geometry and neighbourhoods; nearest-cell lookup == `v1.sample_truth` (z, year) plus cell ids; RTK frame mapping without IndexError at the frame edge |
| `test_benchmark_scoring.py` | `stats` == `v7.vscore` and `ols_against_truth` (1e-12); `wmedian`; block-sum metrics at unit weights == `stats`; identical products → paired deltas 0 in every draw; views (common_q vs common_raw membership, cells with the half-mapped rule); read-outs computed only in the scorer, circular for phase; fallback-as-zero and identified-only; interval error; site-weighted pooling; k/R ≥ 0.5 and joint-computable reps |
| `test_benchmark_inference.py` | variogram fit recovers a planted exponential range (±30 %); lattice rounding and 1-km default; no CI below 25 blocks; Holm; exact sign test; rep-spread rule |
| `test_benchmark_control.py` | synthesis is deterministic (seeded) and matches the planted geometry; a perfect fake product passes and a biased one fails each criterion |
| `test_benchmark_runner.py` | atomic publish over an existing failed unit; resume by `inputs_key`; claim files; `lock` refuses a dirty tree and `score` refuses real data before the lock; invalidation table of §2.11; a full synthetic pipeline writes only under its root; smoke products refused |

---

## 9. Compute budget (server, 10 m) and order of runs

Scaled from the v7, v8 and MAREA logs. Granadeiro per-fit costs from the paper-mode runs (16 workers): prelim 11-24 ms, lag 14 ms per fit; in the controlled setting each fit sees about 140 valid observations instead of 33-98, and the lag window has 46 lags instead of 37.

| Step | Escalda | Wadden | Ems | Ferrol | Villaviciosa |
|---|---|---|---|---|---|
| prepare (two passes, levels, series, geometry) | ~40 min | ~60 min | ~80 min | ~25 min | ~25 min |
| truth (frame truth, lag truths, baselines) | 5 min | 12 min | 15 min | 2 min | 5 min (LiDAR 1.5 GB peak) |
| gates | ~30 min | ~35 min | ~35 min | ~10 min | ~10 min + Granadeiro paper ~40 min |
| controlled N = all, cheap groups | ~15 min | ~45 min | ~50 min | ~5 min | ~6 min |
| scene-rule matrix + mask + index + tau120, cheap groups | ~45 min | ~2.5 h | ~3 h | ~20 min | ~20 min |
| curves, marea_pair (A 10 + F 5 + windows 5 + floors) | ~3 h | ~13 h | ~16 h | ~1.3 h | ~0.5 h |
| curves, NIDEM (10 + windows 5) | ~1.5 h | ~5 h | ~5 h | – | ~0.3 h |
| curves, GM20 + HD26 | ~0.5 h | ~1.5 h | ~2 h | ~0.3 h | ~0.1 h |
| Granadeiro ctl N = all (+ unpinned) | 2.5-4 h | 7-11 h | 8-13 h | 2-3.5 h | 2-3.5 h |
| Granadeiro curves A (5 N × 3 reps), Tier 1 | 12-16 h | – | – | – | 10-14 h |
| Granadeiro scene-rule matrix, Tier 1 | 3-6 h | – | – | – | 3-6 h |
| scoring (variograms, B = 1000, views, curves) | 0.5 h | 1.5 h | 2 h | 0.2 h | 0.3 h |

**Totals.** Cheap groups ≈ 60 single-worker hours, about 1-1.5 days of wall time with 3-5 workers (at most 3-4 concurrent cube streams). Granadeiro Tier 1 (gate, control, five N = all runs, unpinned, curves and matrix at two sites) ≈ 2.5-3.5 days, alone on the machine (16 R workers; one cheap worker with its 4 BLAS threads may run beside it). **Tier 1 ≈ 4-5 days.** Tier 2 (§3.9, mostly Granadeiro: eq1_90 and calon at two then four sites, the matrix at Wadden and Ems, F-mode curves, N = 330 reps at Wadden and Ems, the tuning budget) ≈ 4.5-8 days more.

**Memory and disk:** MAREA pair at the Ems with N = all ≈ 1 GB (column blocks); NIDEM at the Ems ≈ 2.5 GB; Granadeiro at the Ems ≈ 5-6 GB in Python plus 16 R workers (0.24-1.8 GB each); the A-mode domain step at the Ems ≈ 1 GB (MCP on 10.9 M px). Disk: the package ≈ 14 GB, products ≤ 40 GB (S-aligned, compressed), TMPDIR ≥ 5 GB for R exchange files. `check-env` reports free space for both.

**Order of runs** (cheapest information first; Granadeiro alone):

1. Laptop: implement and pass the tests; Escalda 20 m end to end; build the synthetic cube and run the positive control for every method except Granadeiro's full control; Villaviciosa and Ferrol 10 m plumbing; the Granadeiro Villaviciosa-window smoke.
2. Server: copy the gitignored inputs and the control folder; `check-env`; `prepare` and `truth` for all sites (two sites in parallel at most).
3. Gates; the Granadeiro gate; the positive control of every method on the copied cube; `lock`.
4. Phase A (N = all and the Tier-1 sensitivities of the cheap groups) and a first `score`.
5. Phase B, curves of the cheap groups, overnight.
6. Phase C, Granadeiro ctl at five sites, alone.
7. Phase D, Granadeiro curves and scene-rule matrix at Escalda and Villaviciosa.
8. Phase E, Tier 2 as time allows, in the order of §3.9.
9. Final `score` and `report`.

---

## 10. Reporting rules (summary)

- A controlled row needs its method validated (§4); otherwise "implementation not validated".
- Headline elevation claims: §5.4 (Holm, margins, both common views, rep spread, scene-rule robustness).
- Headline lag claims: §5.7 (all three truths, Holm-significant sign tests).
- Everything else is descriptive and labelled as such; EOT20/GOT5.6 are lower bounds; the Ems elevations carry the epoch flag; A-mode and F-mode curves are labelled; the table notes state that MAREA's settings were developed on these sites and judges.

---

## 11. Decisions needing the authors

Each item gives the recommended default first. "(rev.)" marks a default changed by the reviews.

- **D1. Cloud mask.** FINAL (authors): `marea_clear` (SCL in {4,5,6,7}) primary, `blocklist` as sensitivity.
- **D2. Band preprocessing.** `floor0` (B03 and B08 floored at 0 for every method before any index) as the common preprocessing; `raw` (production) as a sensitivity for every method. (rev.)
- **D3. Failure policy.** Zero-lag fallback for every lag method, labelled (MAREA → its no-delay map, Granadeiro → its preliminary DEM; GM20/HD26 read-outs → 0), with "identified only" as the sensitivity. (rev.)
- **D4. MAREA floor.** 40 scenes for both MAREA and no delay; 80 and 0 as sensitivity curves in both modes. (rev.: now applied to both.)
- **D5. N grid.** Add 180, 240 and 330 for every method (needed for the rep-spread rule).
- **D6. Curves.** A-mode primary, F-mode diagnostic; Granadeiro A-mode at Escalda and Villaviciosa, N ∈ {20, 45, 90, 180, 330} × 3 reps; F-mode and Wadden/Ems reps in Tier 2. (rev.)
- **D7. Granadeiro tide input and window.** Controlled row on exact h_B with the common window −45..+180 (MAREA's TAU_BOUNDS); `marks_lp` (61-min low-pass) for the split and h_c; Eq. 1 and ±90 as the `eq1_90` variant. (rev.)
- **D8. Lag read-out.** The scorer's local median over the gauge neighbourhood on every method's applied lag field (primary); band and band_interp secondary. (rev.)
- **D9. Lag truths.** (a), τ_ab and (b) co-primary with interval errors; a ranking only when it holds under all three. (rev.)
- **D10. Judges.** Method-independent inclusion rules only; band-0 judges stay in the headline set; Terneuzen pair as one location; "without mouth" sensitivity. (rev.)
- **D11. Truth range filter.** [min h0 − 0.25, max h0 + 0.25] over the full record; RTK unfiltered.
- **D12. Elevation views and metrics.** D_truth primary, D_keep secondary; `common_q` and `common_raw` co-primary; Vaklodingen cells and survey ≥ 2023 co-primary where n allows; inference on rmse_meanc and |1 − b|; bias and rmse_raw shown with rmse_c. (rev.)
- **D13. Inference.** Variogram-based block size per site (1 km default), ≥ 25 blocks, Holm over the 48-test primary family, margins 2 cm and 0.05. (rev.)
- **D14. Domain.** The production keep is used only by gates; the benchmark's full-record keep is the extraction rule on the common observations. (rev.)
- **D15. Lag-only sites.** Ferrol lag only; Villaviciosa lags descriptive.
- **D16. Lag-only variants.** GM20 `rel`, HD26 `rel` with the 0.8 band rule; the listed variants as sensitivities; GM20 `own` on sea pixels near gauges.
- **D17. Curve aggregation.** k/R ≥ 0.5 for every method; accuracy on jointly computable reps; IQRs labelled conditional on the record. (rev.)
- **D18. Scene rules.** The full method × rule matrix (all, tz, g21, ard) with the common mask; Granadeiro at Wadden and Ems in Tier 2; a conclusion that flips is not a headline. (rev.)
- **D19. As-published reference rows.** Reuse the existing 10 m runs; no new v8 runs at Wadden or Ferrol.
- **D20. Tiers.** Tier 1 (≈ 4-5 server days) is required; Tier 2 (≈ 4.5-8 days) runs in the order of §3.9 as the authors decide.
- **D21. Symmetric tuning budget.** Escalda as the competitors' development site (NIDEM free, Granadeiro Tier 2); confirmatory evaluation on the sealed RTK split or a new site, when the authors decide.
- **D22. Positive-control tolerances.** Coverage ≥ 0.5, median |Δz| ≤ 0.10 m, |bias| ≤ 0.05 m, b ∈ [0.8, 1.2]; lag ≤ 10 min (median over bands).
- **D23. Granadeiro gate R.** env.sh's `$RSCRIPT` for every run; one optional gate run with the stored run's Rscript (outside the project folder) only with the authors' permission.

---

## 12. Risks

1. **A-mode curves start late for the MAREA family:** with ~30 % clear observations, its extraction rule needs N ≳ 100, so MAREA, no delay, GM20 and HD26 are expected NOT COMPUTABLE at N ≤ 60. This changes the "how many images" message, and it is the reason both modes are shown.
2. **Granadeiro dominates time and risk:** about 2.5-3.5 Tier-1 days; the controlled setting is untested at full scale (only the 400-px calibration-off check exists); R platform drift may fail the strict gate, in which case the authors decide with the optional second gate run; the positive control may fail for method reasons rather than bugs, and the row then stays unprinted until explained.
3. **floor0 changes MAREA relative to production** (the controlled MAREA numbers will not equal the published ones; the raw sensitivity and the as-published rows show the difference).
4. **Weak lag evidence:** 13 locations, 8 of them in the Ems; a Holm-corrected sign test needs 12 of 13, and the three truths may disagree, giving "no ranking". The decisive test remains the elevation gain.
5. **Few blocks:** variogram ranges of kilometres leave Villaviciosa, RTK and some band strata without inferential CIs.
6. **Ems:** NIDEM NOT COMPUTABLE at N = all (3-way common set there) and truth epoch ≈ 2020.
7. **Boundary drift and batch invariance:** server CPU vectorisation may break bitwise batch invariance of pyTMD predictions; the fallback is automatic but slower. IOC cache drift is caught by G-record at all five sites.
8. **Development data:** MAREA's settings were chosen on these sites and judges; only a confirmatory dataset can remove this bias.
9. **Disk and I/O:** package ≈ 14 GB, products ≤ 40 GB, two-pass prepare; concurrent cube streams contend on `/media/HDD`.
10. **Re-sequenced driver code** (v8 main L1029-1183, `wc._composite_rows`): pinned by gates and tests, but the Granadeiro paper gate is server-only.
11. **pyTMD versions** (venv and cluster 2.2.9.1, `requirements.txt` 3.0.8): truth (b) and HD26 depend on `solve.constants`; pinned by a planted-phase test and recorded.
12. **Isolated pixels** (NaN s, 1.5-12 % of keep) get τ = 0; MAREA read-outs at harbour gauges may include them (flagged).
13. **Laptop limits:** about 1.5 GB free RAM; only Escalda 20 m, the positive control, Villaviciosa and Ferrol run locally.

---

## 13. Changes after review

| Review issue | Resolution |
|---|---|
| F-blocker: no single definition of N | §0.5 A-mode (primary) / F-mode (diagnostic), every domain step per mode, exact A-mode superset domain (§2.5), floors 40/80/0 |
| F-blocker: no positive control | §4.2 synthetic control through every adapter and the same scorer, pre-registered tolerances, printing rule; Granadeiro window smoke and pinned-fit variant (§3.4, §4.3) |
| Feas-blocker: default mask contradicts the brief | `blocklist` primary (§2.3); SCL stored, mask is a run-time option |
| Feas-blocker: Granadeiro KeepCheck raises on real data | V = mask & finite NDWI for every method; per-mode checks (§3.4) |
| Feas-blocker: bitwise Granadeiro gate across R builds | discrete outputs exact, continuous ≤ 1e-9, GAM at v8's points, R identities recorded, optional second gate run (§4.1) |
| F: MAREA-defined scoring domain | D_truth primary, D_keep secondary (§5.1) |
| F: raw NDWI penalises NIDEM and Granadeiro | `floor0` common preprocessing, `raw` sensitivity for all (§2.4) |
| F: Granadeiro gets less tide information | exact h_B rows; shared low-pass marks; Eq. 1 as a variant (§2.7, §3.4) |
| F: different lag windows | common window −45..+180; ±90 as a variant; edge shares for all (§3.4, §5.6) |
| F: lag truth (a) is MAREA's estimand | (a), τ_ab, (b) co-primary with interval errors (§5.5, §5.7) |
| F: band-0 judges dropped | method-independent inclusion only; mouth sensitivity (§2.8) |
| F: method-specific read-outs | applied lag fields, read out by the scorer for all (§5.6) |
| F: opposite failure policies | one zero-lag fallback policy + identified-only sensitivity (§3.0) |
| F: common view trimmed by MAREA's rule | `common_raw` co-primary, all pairwise views, accuracy-coverage curves (§5.2-5.3) |
| F: scene record not neutral | full method × scene-rule matrix (§3.9) |
| F: 100 m blocks anti-conservative | variogram block size, ≥ 25 blocks (§5.4) |
| F: multiplicity | 48-test primary family, Holm, margins (§5.4) |
| F: thin, asymmetric lag inference | site-weighted pooling, exact sign tests, no method-side CIs (§5.7) |
| F: development-data advantage | MAREA sensitivities, symmetric tuning budget, confirmatory plan, table notes (§5.4) |
| Feas: MAREA lags beyond the table | table to 180·(K−1), live per-shift beyond, never capped (§2.7) |
| Feas: chunking breaks the bitwise MAREA gate | per-τ-group column slices in multiples of 4000 (§3.2) |
| Feas: unhandled R GAM failures | ≥ 60 unique locations, GAM failure → fallback, other R crashes → `r_failed` (§3.0, §3.4) |
| Feas: keys invalidate finished units | sectioned FROZEN, narrow package_key, invalidation table, lock (§2.11) |

Minor issues adopted: verification by size+mtime (§2.11); reused-call signatures (`boundary_series` 3 values, `wait_for_ram` namespace, `sample_truth` cell ids, RTK mapping) (§3.4, §5.1); budget rows for N 180-330, the scene-rule matrix and TMPDIR (§9); gate-mode KeepChecks (§3.3, §3.4); record gates for the Ems and Ferrol (§4.1); explicit thread pools (§3.0); floor for both MAREA products (§3.2); Granadeiro projection (§4.3); aligned SD sub-blocks (§3.4); atomic publish over failed units and tree RSS (§7.4); local path without R (§7.6); BOA-offset and Alicante checks (§7.4); HD26 per-band failures (§3.6); coverage with clear-observation counts (§5.2); Vaklodingen cells (§5.3); calon labelled native preprocessing (§3.4); GM20 `own` (§3.5); joint-computable curve aggregation, contiguous windows (§6); HD26 0.5/0.9 (§3.6); global models as lower bounds (§3.7); survey ≥ 2023 co-primary and Ems flag (§5.3); rep-spread rule (§5.4).

---

## Appendix A. FROZEN (sections; each section's sha256 is recorded where §2.11 says)

| Section | Value |
|---|---|
| record | cube dates in 2023-01-01..2025-12-31 with an overpass time and a finite h_B(t); one provider call on sorted instants |
| package | raw B03/B08 int16 and SCL uint8 at dom = keep_prod ∪ U_rec ∪ NB; U_rec = {clr ≥ 31, wet ≥ 4, dry ≥ 4} under blocklist + floor0; frame validity bits; two passes |
| obs | mask rules blocklist (primary) / marea_clear; index modes floor0 (primary) / raw; V = mask & finite NDWI; wet = V & NDWI > 0; NIR = floored B08/10000 |
| domain | extraction rule clr > 30, 0.10 < wf < 0.90, sea wf ≥ 0.90; 3,000 px; 5,000 finite s; `make_bands_fixed(s, 3.0, 500)`; mouth sides per site |
| levels | integer τ ∈ [−90, max(240, 180·(K_full − 1))], chunks ≤ 150,000, batch-invariance rows, live per-shift beyond; 1-min PERIOD ± 2 d; 10-min = every 10th; marks_lp = 61-min centred mean; marks_raw as v8 |
| subsets | rand: `default_rng([20261008, SITE_ID, r]).permutation(T)[:N]` sorted, 10 reps; win: start `default_rng([20261012, SITE_ID, N, r])`; rules tz / g21 / ard |
| marea | m2.yaml values, TAU_BOUNDS, monotone, seed 20260819 + 31, smoothing 3 km, bracket 3, floor 40 (both products), slices of 200,000; variants tau120; bootstrap diagnostic 30 × 600 px |
| nidem | ITEM rules unchanged; filtered at 0.25; pre-flight 0.01 on keep_full; never relaxed |
| granadeiro | v8 FROZEN steps; ctl: identity calibration, exact h_B, window −45..180 step 5, marks_lp, round(GAM lag); fallback rules (pool empty, < 60 unique, GAM failure); pinned = within 0.005 m of (min + max)/2 of x_used; seed 20210119; variants eq1_90, calon, paper |
| gm20 | 12.4206012 h; marks_lp HW; rel primary; QC n ≥ 20, SD > 0.2, 0.05 < wet < 0.95, R² ≥ 0.1; wrap [−120, 625); 20 × 70 % IQR; own variant |
| hd26 | M2, N2, K2, O1, Q1, GOT corrections; detrend + one-pass 3σ; band scene ≥ 0.8 valid (0.5, 0.9 sensitivities); ≥ 30 points over ≥ 365 d per band; rel primary |
| baselines | zero; EOT20; GOT5.6 if loadable |
| judges | inclusion: in frame, ≥ 50 % instants within 6 min, > 3 km from boundary members, ≤ 10 % of values within 1 cm of the minimum; Terneuzen pair one location; neighbourhood R = max(500, d_min + 300) widened by 250 m to ≥ 100 keep_full px, cap 2,000 m; mouth sensitivity s_g < 3 km |
| truth_elev | Vaklodingen 2019-2025 latest per cell; PNOA ground-only (`v10.lidar_truth`); RTK `load_rtk()` defaults, `elev_alicante`; range filter [min h0 − 0.25, max h0 + 0.25]; D_truth, D_keep, S |
| truth_lag | (a) nearest within 6 min, τ ∈ [−60, 240] at 1 min, median-centred E, 1-cm interval; τ_ab; (b) CONS_B, GOT, order 1, identical instants, quarter-year jackknife, ±1.96 SE; 200 month-block bootstraps (seed 20261011) |
| scoring | MIN_N 30/30/20/30; views and strata of §5.3; cells half-mapped rule; read-outs ≥ 30 px, local primary, circular for phase; fallback 0 primary, identified-only sensitivity |
| inference | variogram (20,000 px, 25 bins to 5 km, exponential, lattice {100, 250, 500, 1000, 2000, 4000}, default 1 km, seed 20261014); ≥ 25 blocks; B = 1000 descriptive and 5000 for the primary family (seed 20261009); Holm α = 0.05; margins 0.02 m and 0.05; rep spread ≥ 90 % at N = 330; lag sign tests, Holm over five comparisons per truth, all three truths |
| curves | N grid; reps per family and mode; k/R ≥ 0.5; joint-computable accuracy |
| threads | BLAS threads per group: marea_pair, nidem, gm20, hd26 = 4; granadeiro = 1 (part of config_key); replay pools explicit |
| positive_control | base wadden 20 m; frame 120 × 640; planted z range [LO + 0.15 R, HI − 0.15 R]; scenarios zero / profile (120 min at 12 km); noise and reflectance laws of §4.2; SCL window; haze 2 % +400 DN; tolerances of D22; seed 20261013 |

## Appendix B. Reuse map

| Existing function | Used by |
|---|---|
| `marea.invert_series`, `QUALITY_MIN_BRACKET`, `BANK_TAUS`, `TAU_BOUNDS`, `TAU_DETECT_MIN`, `CLEAR`, `SG_GRID`, `reconstruct` (gate) | marea_pair, domain |
| `tide_estimators.ShiftBank`, `make_bands_fixed`, `m2a_rasch`, `smooth_lag_profile`, `bootstrap_lags` (test reference) | domain, marea_pair |
| `geometry.mouth_seeds`, `along_distance` | domain, package |
| `boundary.make_boundary`, `PyTMDBoundary`; `net.use_system_certificates`; `tidecheck.nearest_ocean_km` | levels, baselines |
| `waterline_contour.interval_assignment`, `clear_counts_by_interval`, `interval_coverage`, `PackedRows`, `_composite_rows`, `_merge_max`, `nidem`, `pixel_valid` (gate), `frame_good_fraction` (scene rule), `stream_composites` (test reference), `EmptyIntervalError`, `NoClearObservationsInIntervalError` | nidem, package |
| `v7.cube_grid`, `at_keep`, `diagnostics`, `write_rasters` (new dirs only), `vscore` (test reference), `npz_rows`, `ExtractionCheck` (pattern) | package, nidem, scoring |
| `granadeiro2021.*` (to_reflectance, calibrate, band_ranges, reference_scene, calibration_pixels, major_axis, frame_ndwi_sd, ndwi_sd_block, sd_histogram_valleys, tide_marks, water_height, lag_heights, scene_cycles_hours, convert_to_prop, choose_lags, exposure_eq5, RRunner, fit_logistic, lag_fits, lmodel2_rows, gam_lag_surface, r_versions, scene_frame_shares, scene_rule, find_rscript) | granadeiro, levels, package |
| `v8.boundary_series` (3 values), `LazyRows`, `StageCache`, `wait_for_ram` (args namespace), `auto_chunk`, `compare_fits`, `proof_selection`, `band_encoding`, `scan_scenes`, `minutes_since_epoch`, `to_utc_naive`, `load_bands` (test reference) | levels, granadeiro |
| `v1.truth_grid`, `sample_truth` (test reference); `v10.lidar_truth`, `fmt`, `fmt_n`, `scenes`; `rtk.load_rtk()`, `rtk.DEFAULT_GRID`; `elevation_validation.ols_against_truth` (test reference); `experiments/validation_grid.dutch_cube`, `dutch_products`, `thin_mask` (G-score) | truth, report, scoring |
| `v9.load_nidem`, `load_granadeiro` | published |
| `pyTMD.solve.constants`, `pyTMD.arguments.arguments` | truth, hartdavis |
| Scratchpad prototypes (reference only): `repro_escalda20.py`, `lagspec/*.py`, `g21check/*.py`, `benchmark/light_{ndwi,sd}_check.py`; the domain and band-sign light check of §2.4-2.5 (six chunk-aligned 256 × 256 windows of the Escalda 20 m cube, all 464 scenes, per-pixel counters only) is to be re-implemented as `tests/test_benchmark_package.py`'s real-cube check (marked, skipped without the cube) | implementation reference and regression numbers |
