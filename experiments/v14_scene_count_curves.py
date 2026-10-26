# -*- coding: utf-8 -*-
"""V14 - elevation error against the number of scenes used (MAREA, 10 m).

Question (co-author meeting, 2026-10-20)
----------------------------------------
How does the error of the 10 m MAREA elevation change when the record holds
fewer scenes, and is it the NUMBER of scenes or their SCREENING that matters?

Curves
------
MAREA's production inversion is re-run on random subsets of N scenes drawn
from one of two pools:

* ``screened``  MAREA's own screened record (``record_dates`` of the
                production run's result.json): villaviciosa 97, escalda 68,
                wadden 71, ems 54 scenes;
* ``all``       every cube scene with an overpass time and a finite boundary
                level (NIDEM's record, built by
                :func:`experiments.v7_waterline_contour.build_record`;
                Villaviciosa 2023-2025 only): 465 / 464 / 462 / 462 scenes.
                The per-pixel clear-sky mask (SCL 4-7) still applies, as in
                production.

Each N is drawn ``--reps`` times (``--reps-large`` above N = 100), seeded and
reproducible: one SeedSequence per (seed, site, pool, N, replicate). N = the
whole pool is one deterministic run. The figure shows the median over the
replicates and the 10-90 % band. Comparing the pools at equal N separates
quantity from quality; the median number of clear observations per scored
pixel is reported for every run, because many scenes of the ``all`` pool are
cloudy.

One fixed pixel set per site
----------------------------
F = finite truth (after the production range filter) AND finite STORED
production MAREA, i.e. exactly the published 'all' set (194 RTK points on 120
pixels / 248,430 / 1,385,016 / 1,614,155 px). Every N is scored on F where
its estimate is finite (median-centred RMSE, slope, r and bias = median
(product - truth), the functions of v9); ``coverage`` = the share of F with an
estimate, reported separately. Where F is larger than ``--max-px`` a seeded
random subsample of F is scored (declared in the JSON and on the figure;
subsamples of different sizes are nested). Truth only chooses which pixels
are scored: the inversion is per pixel, and the lag-fit pixels are
production's own seeded draw over all intertidal pixels.

Production inversion, exactly
-----------------------------
:func:`pyintertidal.marea.invert_series` with its defaults (mu_points 50,
min_obs 8, ...); levels = the boundary h_B read tau_px whole minutes earlier
(one boundary call over the sorted unique instants, checked against the
stored products); lo/hi = the subset's own zero-lag level range, as
production takes them from its record (so a small subset clamps elevations to
the range it sampled). The quality rule is scored both ways from the same
fit: ``mb`` 0 is the rule of the stored and published 10 m products (made
before the rule existed), ``mb`` 3 the current production rule
(:data:`pyintertidal.marea.QUALITY_MIN_BRACKET`). The rule only removes
estimates, so it is applied as a post-filter on z (invert_series lines
137-142), asserted bitwise against ``invert_series(min_bracket=3)`` before any
curve.

Lags (``--lags``)
-----------------
* ``fixed``  every pixel keeps its production lag (marea.npz tau_px_min: the
             band lags fitted on the full screened record). Declared on the
             figure.
* ``refit``  the band lags are re-fitted on every subset as
             :func:`pyintertidal.marea.reconstruct` does (m2a_rasch on
             production's seeded <= 1200 px per band with configs/m2.yaml,
             negative bands clipped to 0, the smoothed per-pixel profile
             rounded to the minute); no bootstrap (it only gives the interval).

Gates (before any curve; a failed gate stops the run)
-----------------------------------------------------
* the zero-lag level table equals build_record's levels and NIDEM's stored
  ``h`` (primary and sensitivity runs, where present);
* the screened record's level range equals result.json ``rango_marea_m``;
* the stored product scored on F reproduces the published MAREA 'all' row
  (results/method_comparison_same_scenes/<site>.csv, else
  results/method_comparison/<site>.csv);
* the full screened record with mb 0 and the production lags reproduces the
  stored z on F (same valid set, |dz| <= 1e-6 m) and its scores;
* the mb-3 post-filter equals invert_series(min_bracket=3) bitwise.
With ``--lags refit`` the refit on the full screened record is compared with
result.json ``tau_min`` (reported, not a stop).

Budget
------
``--max-minutes`` caps the process wall time. Before the curves the plan is
costed (the gate's own inversion timing scales the server's cost model); if
it does not fit, replicates are reduced first (down to 3), then the N grid is
thinned, and the JSON says so. Jobs run replicate-major (the full pools, then
replicate 0 of every N, replicate 1, ...), so a run stopped by the clock has
lost replicates, not values of N. The run is resumable: one CSV row per
(pool, N, replicate, mb) is appended as each job finishes, and a re-run with
the same settings skips the jobs already there.

Outputs (new files only, under ``--out``, default results/scene_count_curves)
------------------------------------------------------------------------------
<site>_<lags>.csv            N, pool, replicate, mb, rmse, bias, slope, r,
                             coverage, n, ... (one row per job and rule)
<site>_<lags>_settings.json  settings, gates, reference points (MAREA as
                             published, NIDEM as published and on MAREA's
                             record, scored on the same pixel set), shrink
                             notes, progress
<site>_<lags>_draws.npz      the drawn scene indices of every finished job
scene_count_<site>_<lags>_mb<k>.{png,pdf}, scene_count_all_sites_<lags>_mb<k>.{png,pdf}
                             (the ``plot`` command)

Run from the repository root:
    python -m experiments.v14_scene_count_curves run villaviciosa --lags fixed
    python -m experiments.v14_scene_count_curves plot --lags fixed
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # BLAS threads per process (one process per site beside other jobs on the
    # server); must be set before numpy loads its BLAS. A value already in the
    # environment wins.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = "results/scene_count_curves"
POOLS = ("screened", "all")
POOL_IDS = {"screened": 1, "all": 2}
POOL_TEXT = {
    "screened": "MAREA's screened record (record_dates of the production result.json)",
    "all": ("every cube scene with an overpass time and a finite boundary level "
            "(NIDEM's record, v7.build_record); the per-pixel clear-sky mask (SCL 4-7) "
            "still applies"),
}
GRID_BASE = {"screened": (10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90),
             "all": (10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400)}
LARGE_N = 100
MIN_REPS = 3
DEFAULT_SEED = 20261019
MAX_PX = {"fixed": 200_000, "refit": 100_000}
#: columns per invert_series call: a multiple of its chunk (4000), so the
#: chunks are exactly those of one call per lag group
BLOCK = 40_000
#: server prior for one inversion: (12 + 0.38 N) microseconds per pixel
#: (production logs); scaled by the gate's own timing before planning
COST_A_S, COST_B_S = 12e-6, 0.38e-6
JOB_OVERHEAD_S = 0.2
TOL_LEVEL = 1e-9
TOL_Z = 1e-6
TOL_SCORE = 1e-6
#: share of F allowed to differ in validity from the stored product at the
#: full-record gate (BLAS rounding can flip a near-tied grid choice)
TOL_FLIP_SHARE = 1e-5
CSV_COLUMNS = ("site", "lags", "pool", "pool_size", "N", "replicate", "mb",
               "rmse", "bias", "slope", "r", "coverage", "n", "n_set",
               "median_clear_obs", "lo_m", "hi_m", "tau_bands_min",
               "tau_px_median_min", "tau_px_max_min", "seconds", "draw_sha1")
SITE_NAMES = {"villaviciosa": "Ría de Villaviciosa", "escalda": "Westerschelde",
              "wadden": "Wadden Sea (Vlie)", "ems": "Ems-Dollard"}
SITE_ORDER = ("villaviciosa", "escalda", "wadden", "ems")
DUTCH_TAGS = {"escalda": "eot20_g2", "wadden": "eot20_g1", "ems": "eot20_g1"}
#: path components an output directory may never contain (production
#: products, published comparison tables, inputs)
FORBIDDEN_OUT = ("products", "method_comparison", "server_10m", "data", "data_v4",
                 "tide_models", "pyintertidal")

_T0 = time.time()


def log(msg):
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} +{(time.time() - _T0) / 60:6.1f} min] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  inputs
# ─────────────────────────────────────────────────────────────────────────────

def site_config(site):
    """Inputs of one site, relative to the repository root (the server's
    layout; ``run`` takes overrides for a laptop mirror)."""
    if site == "villaviciosa":
        return dict(name=SITE_NAMES[site], truth="rtk",
                    extract="products_marea/marea_extract.npz",
                    product="products_marea/marea_eot20_g0_2023-2025",
                    comparison=None, cube=None, nidem="products_marea/nidem")
    if site in DUTCH_TAGS:
        from experiments.validation_grid import dutch_cube, dutch_products
        tag, p = DUTCH_TAGS[site], dutch_products(site)
        return dict(name=SITE_NAMES[site], truth="vaklodingen", tag=tag,
                    extract=f"{p}/marea_extract.npz", product=f"{p}/marea_{tag}",
                    comparison=f"{p}/comparison_{tag}", cube=dutch_cube(site),
                    nidem=f"{p}/nidem_{tag}")
    raise SystemExit(f"unknown site {site!r}; choose from {list(SITE_NAMES)}")


def published_csvs(site):
    """Where the published MAREA 'all' row of a site is read (first found)."""
    return [f"results/method_comparison_same_scenes/{site}.csv",
            f"results/method_comparison/{site}.csv"]


def _read_exact(f, n):
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            raise EOFError("truncated npy member")
        parts.append(b)
        got += len(b)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def _npy_header(f):
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    return np.lib.format._read_array_header(f, version)  # noqa: SLF001


def stream_columns(path, key, rows, col_sets):
    """``[member[rows][:, c] for c in col_sets]`` for the 2-D C-order member
    ``key`` of the npz at ``path``, read one row at a time: the member (2.7 to
    3.5 GB of NDWI at the Dutch sites) is never held whole."""
    rows = np.asarray(rows, np.int64)
    if len(rows) == 0 or np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be non-empty, sorted and unique")
    cs = [np.asarray(c, np.int64) for c in col_sets]
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as f:
        shape, fortran, dtype = _npy_header(f)
        if fortran or len(shape) != 2:
            raise ValueError(f"{path}:{key}: unexpected layout {shape} (fortran={fortran})")
        T, P = (int(v) for v in shape)
        if rows[-1] >= T:
            raise IndexError(f"{path}:{key}: row {rows[-1]} >= {T}")
        for c in cs:
            if len(c) and (c.min() < 0 or c.max() >= P):
                raise IndexError(f"{path}:{key}: column outside 0..{P - 1}")
        outs = [np.empty((len(rows), len(c)), dtype) for c in cs]
        nrow = P * dtype.itemsize
        want = {int(t): q for q, t in enumerate(rows)}
        for t in range(int(rows[-1]) + 1):
            buf = _read_exact(f, nrow)
            q = want.get(t)
            if q is None:
                continue
            r = np.frombuffer(buf, dtype)
            for o, c in zip(outs, cs):
                o[q] = r[c]
    return outs


class LevelTable:
    """Boundary levels h_B(t - tau) at the chronological instants ``times``
    (the 'all' pool) for whole-minute lags tau, in as few boundary calls as
    possible: the missing lags of a request go into ONE call over the sorted
    unique shifted instants (PyTMDBoundary returns its values sorted by time
    and the consensus averages that with gauges read in input order, so the
    instants must be sorted); each lag is then a positional lookup."""

    def __init__(self, boundary, times):
        import pandas as pd

        self.boundary = boundary
        self.times = pd.DatetimeIndex(times)
        if not self.times.is_monotonic_increasing:
            raise ValueError("times must be chronological")
        self.L = {}
        self.calls = []

    def ensure(self, taus):
        import pandas as pd

        vals = [float(t) for t in np.atleast_1d(np.asarray(taus, float))]
        if any(v != round(v) for v in vals):
            raise ValueError("lags must be whole minutes")
        want = sorted({int(round(v)) for v in vals} - set(self.L))
        if not want:
            return
        t0 = time.time()
        shifted = [self.times - pd.Timedelta(minutes=tv) for tv in want]
        inst = pd.DatetimeIndex(np.unique(np.concatenate([s.values for s in shifted])))
        lev = np.asarray(self.boundary.levels(inst), float)
        if lev.shape != (len(inst),):
            raise RuntimeError(f"boundary returned {lev.shape} levels for {len(inst)} instants")
        for tv, s in zip(want, shifted):
            self.L[tv] = lev[np.searchsorted(inst.values, s.values)]
        self.calls.append({"lags_min": [want[0], want[-1]], "n_lags": len(want),
                           "instants": int(len(inst)), "seconds": round(time.time() - t0, 2)})
        log(f"levels: {len(want)} lag(s) {want[0]}..{want[-1]} min x {len(self.times)} scenes "
            f"in one boundary call over {len(inst):,} instants [{time.time() - t0:.1f} s]")

    def __getitem__(self, tau):
        tv = float(tau)
        if tv != round(tv):
            raise ValueError("lags must be whole minutes")
        tv = int(round(tv))
        if tv not in self.L:
            self.ensure([tv])
        return self.L[tv]


# ─────────────────────────────────────────────────────────────────────────────
#  subsets, plan, budget
# ─────────────────────────────────────────────────────────────────────────────

def site_id(site):
    """A stable integer per site name (Python's hash() is salted per run)."""
    return zlib.crc32(str(site).encode("utf-8")) & 0x7FFFFFFF


def draw(seed, site, pool, N, rep, pool_size):
    """The scenes of replicate ``rep`` at size ``N``: sorted indices into the
    pool. A pure function of (seed, site, pool, N, rep), independent of the
    order in which jobs run, so a resumed run draws the same subsets."""
    N, pool_size = int(N), int(pool_size)
    if N >= pool_size:
        return np.arange(pool_size)
    ss = np.random.SeedSequence([int(seed), site_id(site), POOL_IDS[pool], N, int(rep)])
    return np.sort(np.random.default_rng(ss).choice(pool_size, N, replace=False))


def draw_sha1(idx):
    return hashlib.sha1(np.asarray(idx, "<i8").tobytes()).hexdigest()[:12]


def parse_n_grid(text):
    """'10,20,40' (both pools) or 'screened=10,20;all=10,50,100'."""
    if not text:
        return None
    out = {}
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        key, vals = part.split("=", 1) if "=" in part else ("*", part)
        key = key.strip()
        if key != "*" and key not in POOLS:
            raise SystemExit(f"--n-grid: unknown pool {key!r}")
        out[key] = [int(v) for v in vals.split(",") if v.strip()]
    return out


def pool_grid(pool, size, n_grid=None):
    """The N values of a pool: the base (or ``--n-grid``) values below the
    pool size, plus the whole pool."""
    base = GRID_BASE[pool]
    if n_grid:
        base = n_grid.get(pool, n_grid.get("*", base))
    return sorted({int(n) for n in base if 2 <= int(n) < int(size)}) + [int(size)]


def reps_for(N, size, reps, reps_large):
    if N >= size:
        return 1
    return reps_large if N > LARGE_N else reps


def make_jobs(grids, sizes, reps, reps_large):
    """Replicate-major job list: the whole pools first, then replicate 0 of
    every N of every pool, then replicate 1, ..."""
    jobs = [(p, int(sizes[p]), 0) for p in grids]
    for rep in range(max(reps, reps_large, 1)):
        for p, grid in grids.items():
            for N in grid:
                if N < sizes[p] and rep < reps_for(N, sizes[p], reps, reps_large):
                    jobs.append((p, int(N), rep))
    return jobs


def thin_grid(grid):
    """Every second interior value dropped (the smallest N and the whole
    pool stay)."""
    grid = list(grid)
    if len(grid) <= 2:
        return grid
    inner = grid[1:-1]
    return [grid[0]] + (inner[1::2] if len(inner) > 1 else []) + [grid[-1]]


def shrink_plan(grids, sizes, reps, reps_large, cost, budget_s):
    """Fit the plan into ``budget_s`` seconds: fewer replicates first (down to
    MIN_REPS), then a thinner N grid. ``cost(pool, N)`` is the predicted
    seconds of one job. Returns (grids, reps, reps_large, jobs, notes,
    predicted seconds of the original plan, predicted seconds of the plan)."""
    def total(g, r, rl):
        j = make_jobs(g, sizes, r, rl)
        return j, float(sum(cost(p, N) for p, N, _ in j))

    grids = {p: list(v) for p, v in grids.items()}
    jobs, t = total(grids, reps, reps_large)
    pred0, notes = t, []
    if t <= budget_s:
        return grids, reps, reps_large, jobs, notes, pred0, t
    floor_r = min(MIN_REPS, reps)
    r, rl = reps, reps_large
    while r > floor_r and t > budget_s:
        r -= 1
        rl = max(1, min(reps_large, int(math.ceil(r * reps_large / max(reps, 1)))))
        jobs, t = total(grids, r, rl)
    if (r, rl) != (reps, reps_large):
        notes.append(f"replicates reduced from {reps} (N > {LARGE_N}: {reps_large}) to {r} ({rl}): "
                     f"the full plan was predicted at {pred0 / 60:.1f} min, {budget_s / 60:.1f} min left "
                     f"of --max-minutes")
    while t > budget_s:
        g2 = {p: thin_grid(v) for p, v in grids.items()}
        if g2 == grids:
            break
        grids = g2
        jobs, t = total(grids, r, rl)
        notes.append(f"N grid thinned to {grids} (predicted {t / 60:.1f} min)")
    if t > budget_s:
        notes.append(f"the smallest plan is still predicted at {t / 60:.1f} min > {budget_s / 60:.1f} min: "
                     "it runs anyway and stops at --max-minutes (replicate-major order: the last "
                     "replicates are the ones lost)")
    return grids, r, rl, jobs, notes, pred0, t


# ─────────────────────────────────────────────────────────────────────────────
#  the production inversion on a subset
# ─────────────────────────────────────────────────────────────────────────────

def lag_groups(tau_px):
    """[(lag in whole minutes, columns with that lag)]."""
    tau_px = np.asarray(tau_px, float)
    return [(int(round(tv)), np.flatnonzero(tau_px == tv)) for tv in np.unique(tau_px)]


def bracket_filter(z, clear, h, mb):
    """The per-pixel quality rule of :func:`pyintertidal.marea.invert_series`
    (``min_bracket``; its lines 137-142) applied to the z it returned with
    ``min_bracket=0``: z is kept only with at least ``mb`` clear observations
    whose level is below it and as many above it. The fit is the same, the
    rule only removes estimates, so this is bitwise the result of
    ``invert_series(..., min_bracket=mb)`` (asserted by the gate and the
    tests)."""
    z = np.asarray(z, float)
    if not mb:
        return z
    cl = np.asarray(clear) > 0
    hh = np.asarray(h, float)[:, None]
    zz = z[None, :]
    below = (cl & (hh < zz)).sum(axis=0)
    above = (cl & (hh > zz)).sum(axis=0)
    return np.where((below >= mb) & (above >= mb), z, np.nan)


def invert_subset(Y0, C, levels, rows, groups, mbs, check_mb=False):
    """MAREA's production inversion of the scenes ``rows`` (sorted indices
    into the rows of ``Y0``/``C``) on every column, each column with the
    levels of its lag group, lo/hi = the subset's zero-lag level range.

    ``Y0`` (T, P) float32 NDWI with NaN -> 0, ``C`` (T, P) bool clear flags.
    Returns ({mb: z}, clear observations per column, (lo, hi), seconds spent
    in the mb-0 inversion, number of mb>0 values that differ from
    ``invert_series(min_bracket=mb)`` when ``check_mb``)."""
    from pyintertidal import marea

    rows = np.asarray(rows)
    h0 = np.asarray(levels[0], float)[rows]
    lo, hi = float(h0.min()), float(h0.max())
    P = Y0.shape[1]
    z = {mb: np.full(P, np.nan) for mb in mbs}
    nclear = np.zeros(P, np.int64)
    t_inv, mismatch = 0.0, 0
    for tv, cols in groups:
        if not len(cols):
            continue
        h = np.asarray(levels[tv], float)[rows]
        for b0 in range(0, len(cols), BLOCK):
            cb = cols[b0:b0 + BLOCK]
            Yg = Y0[np.ix_(rows, cb)]
            Cg = C[np.ix_(rows, cb)]
            t1 = time.time()
            z0, _ = marea.invert_series(Yg, Cg, h, lo, hi, min_bracket=0)
            t_inv += time.time() - t1
            nclear[cb] = Cg.sum(axis=0)
            for mb in mbs:
                z[mb][cb] = bracket_filter(z0, Cg, h, mb)
                if check_mb and mb:
                    zr, _ = marea.invert_series(Yg, Cg, h, lo, hi, min_bracket=mb)
                    same = (zr == z[mb][cb]) | (np.isnan(zr) & np.isnan(z[mb][cb]))
                    mismatch += int((~same).sum())
    return z, nclear, (lo, hi), t_inv, mismatch


def refit_lags(sd, rows):
    """The band lags re-fitted on the scenes ``rows`` exactly as
    :func:`pyintertidal.marea.reconstruct` does (its lines 312-371, without
    the bootstrap) and the per-pixel lag of every inverted column. The
    lag-fit arrays hold only production's seeded draw (<= max_px_band per
    band), so m2a_rasch draws nothing and fits the same pixels."""
    from pyintertidal import marea
    from pyintertidal import tide_estimators as te

    m = sd.m2a
    rows = np.asarray(rows)
    Cm = m["C"][rows]
    wet = Cm & (m["Y"][rows] > 0)
    bank = te.ShiftBank(marea.BANK_TAUS, [sd.levels[t][rows] for t in marea.BANK_TAUS])
    r = te.m2a_rasch(wet, Cm, bank, m["band"], m["centers"], m["nb"],
                     rng=np.random.default_rng(m["seed"] + 31), **m["fit_kw"])
    tau = np.asarray(r["tau"], float)
    tau_used = np.where(np.abs(tau) > marea.TAU_DETECT_MIN, tau, 0.0)
    tau_used = np.where(tau_used < 0, 0.0, tau_used)
    if m["nb"] > 1:
        tp = te.smooth_lag_profile(m["centers"], tau_used, sd.s_km, bandwidth_km=m["band_km"])
    else:
        tp = np.zeros(len(sd.s_km))
    tp = np.where(np.isfinite(tp), np.round(tp), 0.0)
    return tau, tau_used, tp


def score(kind, truth, p):
    """The published metrics of one product on the scored units: 'rtk'
    (pyintertidal.elevation_validation.ols_against_truth, as v9 at
    Villaviciosa, None below 20 points) or 'vaklodingen' (v7.vscore, as v9 at
    the Dutch sites, None below 30 px). Coverage = share of the units with a
    finite estimate."""
    p = np.asarray(p, float)
    truth = np.asarray(truth, float)
    fin = np.isfinite(p) & np.isfinite(truth)
    out = dict(rmse=np.nan, bias=np.nan, slope=np.nan, r=np.nan, n=int(fin.sum()),
               coverage=float(np.isfinite(p).mean()) if len(p) else np.nan)
    if kind == "rtk":
        from pyintertidal import elevation_validation as ev
        f = ev.ols_against_truth(truth, p)
    elif kind == "vaklodingen":
        from experiments.v7_waterline_contour import vscore
        f = vscore(p, truth)
    else:
        raise ValueError(kind)
    if f:
        out.update(rmse=float(f["rmse_m"]), bias=float(f["bias_m"]), slope=float(f["slope"]),
                   r=float(f["r"]), n=int(f["n"]))
    return out


class SiteData:
    """What the curves need for one site, in memory.

    ``Y0``/``C``: (T, P) NDWI (NaN -> 0, float32) and clear flags (bool) of
    the P inverted pixels on the T scenes of the 'all' pool; ``levels``: a
    :class:`LevelTable` on those scenes' instants; ``pools``: {name: sorted
    row indices}; ``unit_col``: the inverted column of every scored unit
    (an RTK point or a Vaklodingen pixel); ``truth``: truth per unit;
    ``kind``: 'rtk' or 'vaklodingen'; ``tau_fixed``: production lag of every
    inverted pixel (whole minutes); ``s_km``: mouth distance of every inverted
    pixel (refit); ``m2a``: the lag-fit arrays and settings (refit)."""

    def __init__(self, site, Y0, C, levels, pools, unit_col, truth, kind, tau_fixed,
                 s_km=None, m2a=None):
        self.site, self.Y0, self.C, self.levels = site, Y0, C, levels
        self.pools = {k: np.asarray(v, np.int64) for k, v in pools.items()}
        self.unit_col = np.asarray(unit_col, np.int64)
        self.truth = np.asarray(truth, float)
        self.kind = kind
        self.tau_fixed = np.asarray(tau_fixed, float)
        self.groups_fixed = lag_groups(self.tau_fixed)
        self.s_km = None if s_km is None else np.asarray(s_km, float)
        self.m2a = m2a
        self.refit_cache = {}
        assert Y0.shape == C.shape and Y0.shape[1] == len(self.tau_fixed)
        assert len(self.unit_col) == len(self.truth)
        for k, v in self.pools.items():
            assert len(v) and np.all(np.diff(v) > 0) and v[-1] < Y0.shape[0], k


def run_job(sd, pool, N, rep, seed, lags, mbs, check_mb=False):
    """One (pool, N, replicate): draw, (refit,) invert, score every rule on
    the fixed set. Returns (CSV rows, details)."""
    size = len(sd.pools[pool])
    idx = draw(seed, sd.site, pool, N, rep, size)
    rows = sd.pools[pool][idx]
    sha = draw_sha1(rows)
    t0 = time.time()
    tau_b = None
    if lags == "refit":
        if sha not in sd.refit_cache:
            sd.refit_cache[sha] = refit_lags(sd, rows)
        tau_b, _, tau_px = sd.refit_cache.pop(sha)
        sd.levels.ensure(np.unique(tau_px))
        groups = lag_groups(tau_px)
    elif lags == "fixed":
        tau_px, groups = sd.tau_fixed, sd.groups_fixed
    else:
        raise ValueError(lags)
    z, nclear, (lo, hi), t_inv, mism = invert_subset(sd.Y0, sd.C, sd.levels, rows, groups,
                                                     mbs, check_mb=check_mb)
    clear_units = nclear[sd.unit_col]
    seconds = round(time.time() - t0, 3)
    out = []
    for mb in mbs:
        s = score(sd.kind, sd.truth, z[mb][sd.unit_col])
        out.append(dict(site=sd.site, lags=lags, pool=pool, pool_size=size, N=int(len(idx)),
                        replicate=int(rep), mb=int(mb), **s, n_set=int(len(sd.unit_col)),
                        median_clear_obs=float(np.median(clear_units)), lo_m=lo, hi_m=hi,
                        tau_bands_min=(";".join(f"{v:.3f}" for v in tau_b) if tau_b is not None else ""),
                        tau_px_median_min=float(np.median(tau_px)),
                        tau_px_max_min=float(np.max(tau_px)), seconds=seconds,
                        draw_sha1=draw_sha1(idx)))
    return out, dict(z=z, t_inv=t_inv, mismatch=mism, idx=idx, rows=rows, tau_bands=tau_b,
                     tau_px=tau_px, seconds=time.time() - t0)


# ─────────────────────────────────────────────────────────────────────────────
#  resumable CSV, JSON
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "nan" if not np.isfinite(v) else repr(v)
    return str(v)


class CsvLog:
    """The per-site CSV, appended job by job (one write and fsync per job),
    and the set of jobs it already holds. A job counts as done only with a
    row for every rule in ``mbs``; the rows of a job interrupted mid-write
    (a partial last line, or some of its rules only) are dropped on load so
    the job is redone cleanly."""

    def __init__(self, path, mbs):
        self.path = path
        self.mbs = {int(m) for m in mbs}
        self.done = {}
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def _load(self):
        with open(self.path, "rb") as f:
            data = f.read()
        if not data.endswith(b"\n"):
            # a job interrupted mid-write: drop its partial last line
            cut = data.rfind(b"\n") + 1
            with open(self.path, "r+b") as f:
                f.truncate(cut)
            log(f"{self.path}: dropped a partial last line ({len(data) - cut} bytes)")
        with open(self.path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            if tuple(header or ()) != CSV_COLUMNS:
                raise SystemExit(f"{self.path}: unexpected header; refusing to append "
                                 "(use another --out)")
            rows = [r for r in rd if len(r) == len(CSV_COLUMNS)]
        for row in rows:
            d = dict(zip(CSV_COLUMNS, row))
            self.done.setdefault((d["pool"], int(d["N"]), int(d["replicate"])), set()).add(int(d["mb"]))
        partial = {k for k, s in self.done.items() if not self.mbs <= s}
        if partial:
            keep = [r for r in rows if (r[2], int(r[4]), int(r[5])) not in partial]
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(CSV_COLUMNS)
                w.writerows(keep)
            os.replace(tmp, self.path)
            for k in partial:
                del self.done[k]
            log(f"{self.path}: dropped the rows of {len(partial)} job(s) written only in part: {sorted(partial)}")

    def is_done(self, pool, N, rep, mbs):
        return set(mbs) <= self.done.get((pool, int(N), int(rep)), set())

    def append(self, rows):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([_fmt(r[c]) for c in CSV_COLUMNS])
            f.flush()
            os.fsync(f.fileno())
        for r in rows:
            self.done.setdefault((r["pool"], int(r["N"]), int(r["replicate"])), set()).add(int(r["mb"]))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _clean(o):
    """NaN/inf -> None, recursively (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=1, default=_json_default, ensure_ascii=False)
    os.replace(tmp, path)


def array_sha1(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha1(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()[:16]


def file_info(path):
    if not path or not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    info = {"path": path, "exists": True, "bytes": int(st.st_size),
            "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    if st.st_size < 50e6:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def guard_out(out):
    """Refuse an output folder inside the repository whose path (relative to
    the repository root) names production products, published tables or
    inputs. A folder outside the repository (a scratch area) is allowed."""
    try:
        rel = os.path.relpath(os.path.abspath(out), ROOT)
    except ValueError:                       # another drive (Windows): outside
        return
    if rel == ".." or rel.startswith(".." + os.sep):
        return
    parts = [p.lower() for p in rel.split(os.sep)]
    bad = [p for p in parts if any(p.startswith(f) for f in FORBIDDEN_OUT)]
    if bad or rel == ".":
        raise SystemExit(f"refusing to write into {out!r} ({repr(bad[0]) if bad else 'the repository root'}): "
                         "outputs go to new paths only, e.g. results/scene_count_curves")


# ─────────────────────────────────────────────────────────────────────────────
#  loading a site
# ─────────────────────────────────────────────────────────────────────────────

def gauge_cache(site):
    """The IOC gauges make_boundary will use at a Dutch site and those whose
    cache file is missing (the run refuses to download: production used the
    cache)."""
    import pyintertidal as pit
    from pyintertidal.boundary import nearest_gauges
    from experiments import v7_waterline_contour as v7

    c = v7.SITES[site]
    lat, lon = pit.sites.get(site).centroid
    excl = {str(x).lower() for x in c["judge"]}
    near = [g for g in nearest_gauges(lat, lon, c["n_gauges"] + len(excl))
            if g[0].lower() not in excl][:int(c["n_gauges"])]
    t0, t1 = v7.TIME_EXTENT
    codes = [g[0] for g in near]
    missing = [k for k in codes if not os.path.exists(f"data_v4/gauges/ioc_{k}_{t0}_{t1}.json")]
    return codes, missing


def rtk_units(keep, W, z_stored):
    """The RTK development points (pyintertidal.rtk.load_rtk(), default
    arguments only) on the extraction pixels: (keep index per point, truth,
    info). Truth = REDNAP orthometric heights (Alicante) as v9's tables."""
    from pyintertidal import rtk

    dev = rtk.load_rtk()
    flat = np.asarray(dev["row"], np.int64) * W + np.asarray(dev["col"], np.int64)
    pos = np.clip(np.searchsorted(keep, flat), 0, len(keep) - 1)
    in_keep = keep[pos] == flat
    if "elev_alicante" in dev:
        zt, datum = np.asarray(dev["elev_alicante"], float), "Alicante (REDNAP orthometric)"
    else:
        zt, datum = np.asarray(dev["elev"], float), "ETRS89 ellipsoidal"
    zs = np.where(in_keep, z_stored[pos], np.nan)
    F = np.flatnonzero(np.isfinite(zt) & in_keep & np.isfinite(zs))
    info = {"truth": "RTK-GNSS development split (pyintertidal.rtk.load_rtk(), default arguments)",
            "datum": datum, "points_dev": int(len(zt)), "points_in_keep": int(in_keep.sum()),
            "n_reserved_hidden": int(dev.get("n_reserved_hidden", -1)),
            "points_in_F": int(len(F)), "pixels_in_F": int(len(np.unique(pos[F])))}
    return pos[F], zt[F], info


def dutch_truth(site, cfg, keep, H, W, z_plain):
    """Vaklodingen at every extraction pixel with the production range filter,
    exactly as v9.score_dutch: (truth per keep pixel, info)."""
    import pyproj
    from experiments import v1_vaklodingen_validate as v1
    from experiments import v7_waterline_contour as v7

    grid = v7.cube_grid(cfg["cube"])
    if (H, W) != (grid["H"], grid["W"]):
        raise RuntimeError(f"cube grid {grid['H']}x{grid['W']} != extraction {H}x{W}")
    rows_k, cols_k = keep // W, keep % W
    to_rd = pyproj.Transformer.from_crs(grid["crs"], "EPSG:28992", always_xy=True)
    xq, yq = to_rd.transform(grid["xs"][cols_k], grid["ys"][rows_k])
    cells = v1.truth_grid(site)
    truth_raw, _ = v1.sample_truth(cells, xq, yq)
    lo_, hi_ = float(np.nanmin(z_plain)), float(np.nanmax(z_plain))
    in_range = (truth_raw >= lo_ - 0.25) & (truth_raw <= hi_ + 0.25)
    truth = np.where(in_range, truth_raw, np.nan)
    info = {"truth": f"Rijkswaterstaat Vaklodingen, latest survey per 20 m RD cell in {v1.YEARS} (NAP), "
                     "nearest cell (v1.truth_grid / sample_truth)",
            "range_filter_m": [lo_ - 0.25, hi_ + 0.25],
            "range_filter_source": "production comparison products.npz z_plain range +- 0.25 m (fixed for every N)",
            "truth_px": int(np.isfinite(truth_raw).sum()), "truth_in_range": int(np.isfinite(truth).sum()),
            "pixel_m": abs(float(grid["xs"][1] - grid["xs"][0]))}
    return truth, info


def published_row(site, paths=None):
    """The published MAREA 'all' row (n, rmse_m, slope, r, bias_m) and its
    source (the first of ``paths``, default :func:`published_csvs`, that has
    it), or (None, None)."""
    import pandas as pd

    for f in (paths or published_csvs(site)):
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        a = d[(d["product"] == "MAREA") & (d["subset"] == "all")]
        if len(a):
            a = a.iloc[0]
            return {k: float(a[k]) for k in ("n", "rmse_m", "slope", "r", "bias_m")}, f
    return None, None


def load_site(site, cfg, lags, max_px, seed, pools_wanted):
    """Build the :class:`SiteData` of a site from the stored production files.
    Returns (SiteData, meta, extras) where extras holds what the gates need."""
    import yaml
    from pyintertidal import marea
    from pyintertidal import tide_estimators as te
    from experiments import v7_waterline_contour as v7

    meta = {"notes": [], "inputs": {}}
    t0 = time.time()
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        keep = np.asarray(ex["keep"])
        ext_dates = np.asarray(ex["dates"]).astype(str)
        H, W = (int(v) for v in ex["shape"])
    shapeY = None
    with zipfile.ZipFile(cfg["extract"]) as zf, zf.open("Y.npy") as f:
        shapeY = tuple(int(v) for v in _npy_header(f)[0])
    log(f"[{site}] extraction {cfg['extract']}: {len(ext_dates)} scenes "
        f"({ext_dates[0]}..{ext_dates[-1]}), {len(keep):,} px, grid {H}x{W}, Y {shapeY}")
    meta["inputs"]["extract"] = {"path": cfg["extract"], "Y_shape": shapeY, "grid": [H, W],
                                 "n_keep": int(len(keep))}

    pdir = cfg["product"]
    res_path, mp_path = f"{pdir}/result.json", f"{pdir}/marea.npz"
    res = json.load(open(res_path, encoding="utf-8"))
    meta["inputs"]["result_json"] = file_info(res_path)
    mp = None
    if os.path.exists(mp_path):
        mp = dict(np.load(mp_path, allow_pickle=False))
        if not np.array_equal(mp["keep"], keep):
            raise RuntimeError(f"{mp_path}: keep differs from the extraction")
    meta["inputs"]["marea_npz"] = file_info(mp_path)
    if "quality_min_bracket" in res:
        meta["notes"].append(f"result.json records quality_min_bracket = {res['quality_min_bracket']}")
    else:
        meta["notes"].append("result.json has no quality_min_bracket key: the stored product predates the "
                             "rule (min_bracket 0), so the published point is the mb-0 rule")
    comp = None
    if cfg.get("comparison"):
        cp = f"{cfg['comparison']}/products.npz"
        meta["inputs"]["comparison_products"] = file_info(cp)
        if os.path.exists(cp):
            with np.load(cp, allow_pickle=False) as P:
                comp = {k: P[k] for k in P.files}
            if not np.array_equal(comp["keep"], keep):
                raise RuntimeError(f"{cp}: keep differs from the extraction")
            if "record_dates" in comp and [str(d) for d in comp["record_dates"]] != \
                    [str(d)[:10] for d in res["record_dates"]]:
                raise RuntimeError(f"{cp}: record_dates differ from {res_path}")

    # the stored production z (the published product) and its lags
    if comp is not None:
        z_stored = comp["z_marea"].astype(float)
        z_src = f"{cfg['comparison']}/products.npz z_marea (the array v9 scores)"
        if mp is not None:
            same = np.array_equal(np.isfinite(mp["z"]), np.isfinite(z_stored))
            dz = float(np.nanmax(np.abs(mp["z"].astype(float) - z_stored)))
            meta["notes"].append(f"marea.npz z vs products.npz z_marea: same valid set {same}, "
                                 f"max |dz| {dz:.2e} m")
    elif mp is not None:
        z_stored = mp["z"].astype(float)
        z_src = f"{mp_path} z"
    else:
        raise SystemExit(f"[{site}] neither {mp_path} nor a comparison products.npz: no stored product")
    centers = np.asarray(res["centros_km"], float)
    band_km = float(mp["band_km"][0]) if mp is not None else float(res.get("band_km", 3.0))
    if comp is not None and "s_km" in comp:
        s_km_all, s_src = comp["s_km"].astype(float), "comparison products.npz s_km (float64)"
    elif mp is not None:
        s_km_all, s_src = mp["s_km"].astype(float), "marea.npz s_km (stored float32)"
    else:
        s_km_all, s_src = None, None
    if mp is not None:
        tau_px_all, tau_src = mp["tau_px_min"].astype(float), f"{mp_path} tau_px_min (production)"
        band_all = mp["band"].astype(int)
    else:
        if s_km_all is None or len(centers) < 2:
            raise SystemExit(f"[{site}] no marea.npz and no s_km to rebuild the production lags")
        tp = te.smooth_lag_profile(centers, np.asarray(res["tau_usado_min"], float), s_km_all,
                                   bandwidth_km=band_km)
        tau_px_all = np.where(np.isfinite(tp), np.round(tp), 0.0)
        tau_src = ("REBUILT (no marea.npz): smooth_lag_profile(result.json centros_km, tau_usado_min, "
                   f"{s_src}, bandwidth {band_km} km), rounded")
        band_all = comp["band"].astype(int) if comp is not None and "band" in comp else None
        meta["notes"].append("production lags rebuilt from result.json (marea.npz absent)")
    if np.any(tau_px_all != np.round(tau_px_all)):
        raise RuntimeError("production lags are not whole minutes")

    # the pools: build_record gives NIDEM's all-scenes record and its boundary
    if site != "villaviciosa":
        codes, missing = gauge_cache(site)
        meta["inputs"]["gauges"] = codes
        if missing:
            raise SystemExit(f"[{site}] gauge cache missing for {missing} in data_v4/gauges: "
                             "refusing to download (production used the cache)")
    t1 = time.time()
    rec = v7.build_record(site, v7.SITES[site], ext_dates)
    rows_all = np.asarray(rec["time_index"], np.int64)
    dates_all = np.asarray(rec["dates"]).astype(str)
    t_all = rec["t_real"]
    log(f"[{site}] pool 'all': {len(rows_all)} scenes ({rec['n_without_time']} without overpass time, "
        f"{rec['n_nonfinite_level']} without a finite level), boundary {rec['boundary']} "
        f"[{time.time() - t1:.1f} s]")
    rec_dates = [str(d)[:10] for d in res["record_dates"]]
    pos = {d: i for i, d in enumerate(dates_all)}
    lost = [d for d in rec_dates if d not in pos]
    if lost:
        raise RuntimeError(f"{len(lost)} record date(s) not in the 'all' pool: {lost[:5]}")
    idx_scr = np.array([pos[d] for d in rec_dates], np.int64)
    if np.any(np.diff(idx_scr) <= 0):
        raise RuntimeError("record_dates are not ascending")
    # both pools are always loaded (the gates use the screened record); the
    # plan draws only from the pools asked for
    pools = {"screened": idx_scr, "all": np.arange(len(rows_all))}
    meta["pools"] = {
        "screened": {"size": int(len(idx_scr)), "definition": POOL_TEXT["screened"],
                     "dates": rec_dates, "used": "screened" in pools_wanted},
        "all": {"size": int(len(rows_all)), "definition": POOL_TEXT["all"], "rule": rec["rule"],
                "dates": dates_all.tolist(), "used": "all" in pools_wanted,
                "n_without_time": rec["n_without_time"], "n_nonfinite_level": rec["n_nonfinite_level"]},
    }
    meta["boundary"] = rec["boundary"]

    # the fixed scoring set F, its truth, and the inverted pixels
    t1 = time.time()
    if cfg["truth"] == "rtk":
        unit_keep, unit_truth, tinfo = rtk_units(keep, W, z_stored)
        kind = "rtk"
    else:
        if comp is None:
            raise SystemExit(f"[{site}] the range filter needs the comparison products.npz z_plain")
        truth_px, tinfo = dutch_truth(site, cfg, keep, H, W, comp["z_plain"])
        F = np.flatnonzero(np.isfinite(truth_px) & np.isfinite(z_stored))
        unit_keep, unit_truth = F, truth_px[F]
        kind = "vaklodingen"
    nF = len(unit_keep)
    if max_px and nF > max_px:
        perm = np.random.default_rng(np.random.SeedSequence([int(seed), site_id(site), 7])).permutation(nF)
        sel = np.sort(perm[:int(max_px)])
        sub = {"n": int(max_px), "of": int(nF), "share": float(max_px / nF),
               "rule": ("seeded random subsample of F: the first max_px of one permutation "
                        "(SeedSequence([seed, site_id, 7])), sorted; nested across max_px")}
        log(f"[{site}] F has {nF:,} units > --max-px {max_px:,}: scoring a seeded random subsample")
    else:
        sel, sub = np.arange(nF), None
    units_keep, units_truth = unit_keep[sel], unit_truth[sel]
    px = np.unique(units_keep)
    unit_col = np.searchsorted(px, units_keep)
    meta["fixed_set"] = {
        "definition": ("F = finite truth (after the production range filter) AND finite stored "
                       f"production MAREA ({z_src}); the published 'all' set"),
        "n_units_F": int(nF), "n_units_scored": int(len(units_keep)), "n_pixels_inverted": int(len(px)),
        "unit": "RTK point" if kind == "rtk" else f"{tinfo.get('pixel_m', 10):g} m pixel", "subsample": sub,
        **tinfo,
        "coverage": "share of the scored units with a finite estimate at that N",
    }
    log(f"[{site}] fixed set: {len(units_keep):,} scored {meta['fixed_set']['unit']}s on {len(px):,} "
        f"inverted pixels (F = {nF:,}) [{time.time() - t1:.1f} s]")

    # lag-fit pixels (refit): production's own seeded draw over ALL keep pixels
    m2a = None
    m2a_cols = np.zeros(0, np.int64)
    if lags == "refit":
        if band_all is None:
            raise SystemExit(f"[{site}] refit needs the production bands (marea.npz band)")
        if s_km_all is None:
            raise SystemExit(f"[{site}] refit needs s_km")
        cfg2 = yaml.safe_load(open(os.path.join(ROOT, "configs", "m2.yaml"), encoding="utf-8"))
        maxb = int(cfg2["max_px_banda"]["m2a"])
        nb = len(centers)
        rng = np.random.default_rng(cfg2["seed"] + 31)
        sel_b = []
        for k in range(nb):
            c = np.where(band_all == k)[0]
            if len(c) > maxb:
                c = np.sort(rng.choice(c, maxb, replace=False))
            sel_b.append(c)
        m2a_cols = np.concatenate(sel_b).astype(np.int64)
        band_red = np.concatenate([np.full(len(c), k) for k, c in enumerate(sel_b)])
        fit_kw = dict(sigma0=cfg2["sigma0_m"], sg_grid=tuple(cfg2["sigma_perfil_m"]),
                      tau_bounds=marea.TAU_BOUNDS, z_points=cfg2["z_puntos"],
                      max_px_band=maxb, monotone=True)
        m2a = dict(band=band_red, centers=centers, nb=nb, band_km=band_km, fit_kw=fit_kw,
                   seed=int(cfg2["seed"]))
        meta["lag_fit"] = {
            "function": "pyintertidal.tide_estimators.m2a_rasch (as reconstruct L312-371, no bootstrap)",
            "pixels": (f"production's seeded draw: default_rng({cfg2['seed']} + 31), <= {maxb} px per band "
                       "over ALL extraction pixels by band (marea.npz band); truth never chooses them"),
            "n_pixels": int(len(m2a_cols)), "per_band": [int(len(c)) for c in sel_b],
            "fit_kw": {k: (list(v) if isinstance(v, tuple) else v) for k, v in fit_kw.items()},
            "bank_taus_min": list(marea.BANK_TAUS), "centers_km": centers.tolist(),
            "band_km": band_km, "tau_detect_min": marea.TAU_DETECT_MIN,
            "causality": "negative band lags clipped to 0", "s_km": s_src,
            "per_pixel": "smooth_lag_profile(centers, tau_used, s_km, bandwidth band_km), rounded, NaN -> 0",
        }

    # stream the needed columns of the 'all' pool's rows
    t1 = time.time()
    col_sets = [px] + ([m2a_cols] if len(m2a_cols) else [])
    Ys = stream_columns(cfg["extract"], "Y", rows_all, col_sets)
    Cs = stream_columns(cfg["extract"], "C", rows_all, col_sets)
    Y0 = np.nan_to_num(Ys[0], copy=False, nan=0.0).astype(np.float32, copy=False)
    C0 = Cs[0].astype(bool, copy=False)
    if m2a is not None:
        m2a["Y"] = np.nan_to_num(Ys[1], copy=False, nan=0.0).astype(np.float32, copy=False)
        m2a["C"] = Cs[1].astype(bool, copy=False)
    del Ys, Cs
    log(f"[{site}] streamed {len(rows_all)} rows x {len(px):,} inverted + {len(m2a_cols):,} lag-fit "
        f"columns [{time.time() - t1:.1f} s]")

    # levels: one boundary call for every lag the run needs
    levels = LevelTable(rec["boundary_obj"], t_all)
    tau_fixed = tau_px_all[px]
    need = {0} | {int(v) for v in np.unique(tau_fixed)}
    if lags == "refit":
        need |= {int(v) for v in marea.BANK_TAUS} | set(range(0, 181))
    levels.ensure(sorted(need))

    sd = SiteData(site, Y0, C0, levels, pools, unit_col, units_truth, kind, tau_fixed,
                  s_km=(s_km_all[px] if s_km_all is not None else None), m2a=m2a)
    meta["lags_production"] = {
        "source": tau_src, "distinct_on_inverted_px": int(len(np.unique(tau_fixed))),
        "median_min": float(np.median(tau_fixed)), "max_min": float(np.max(tau_fixed)),
        "band_tau_min": list(map(float, res.get("tau_min", []))),
        "band_tau_used_min": list(map(float, res.get("tau_usado_min", []))),
        "centers_km": centers.tolist()}
    extras = dict(res=res, rec=rec, keep=keep, z_stored=z_stored, unit_keep_full=unit_keep,
                  unit_truth_full=unit_truth, units_keep=units_keep, px=px, kind=kind,
                  idx_scr=idx_scr, dates_all=dates_all, rec_dates=rec_dates, tau_px_all=tau_px_all,
                  load_seconds=time.time() - t0)
    return sd, meta, extras


# ─────────────────────────────────────────────────────────────────────────────
#  gates and reference points
# ─────────────────────────────────────────────────────────────────────────────

def reference_points(site, cfg, sd, extras):
    """MAREA as published and NIDEM (as published; on MAREA's record), scored
    on the same units as the curves (and on the whole F where it was
    subsampled). NIDEM's published values sit on its own pixels; these are
    the comparable ones."""
    from experiments import v9_method_comparison as v9

    keep, kind = extras["keep"], extras["kind"]
    sub = len(extras["unit_keep_full"]) != len(extras["units_keep"])
    pts = []

    def add(label, key, scenes, pool, values):
        on = score(kind, sd.truth, values[extras["units_keep"]])
        row = {"label": label, "key": key, "scenes": scenes, "pool": pool,
               "on_scored_units": on}
        if sub:
            row["on_full_F"] = score(kind, extras["unit_truth_full"], values[extras["unit_keep_full"]])
        pts.append(row)

    add("MAREA as published (full screened record, mb 0, production lags)", "marea_published",
        len(extras["rec_dates"]), "screened", extras["z_stored"])
    nid = {"primary": cfg["nidem"], "sensitivity": os.path.join(cfg["nidem"], "sensitivity_marea_record")}
    for key, label, pool in (("primary", "NIDEM as published (all scenes)", "all"),
                             ("sensitivity", "NIDEM on MAREA's record", "screened")):
        v, n_sc, st = v9.load_nidem(nid[key], keep)
        if v is None:
            pts.append({"label": label, "key": f"nidem_{key}", "scenes": None, "pool": pool,
                        "status": st, "path": nid[key]})
            continue
        add(label, f"nidem_{key}", int(n_sc) if n_sc is not None else None, pool, v)
        pts[-1]["path"] = nid[key]
    return pts


def run_gates(site, cfg, sd, extras, mbs, seed, lags, out_rows):
    """The gates before any curve. Appends the full screened-record job
    (fixed lags) to ``out_rows`` and returns (gates dict, ok, timing)."""
    g, ok = {}, True
    rec, res, idx_scr = extras["rec"], extras["res"], extras["idx_scr"]
    L0 = sd.levels[0]

    d = float(np.max(np.abs(L0 - np.asarray(rec["h"], float))))
    g["levels_vs_build_record"] = {"max_abs_diff_m": d, "ok": d <= TOL_LEVEL}
    # NIDEM's stored levels: the primary on the 'all' pool, the sensitivity
    # on MAREA's record
    for key, f, rows_ in (("primary", os.path.join(cfg["nidem"], "nidem.npz"), np.arange(len(L0))),
                          ("sensitivity", os.path.join(cfg["nidem"], "sensitivity_marea_record", "nidem.npz"),
                           idx_scr)):
        if not os.path.exists(f):
            g[f"levels_vs_nidem_{key}"] = {"path": f, "checked": False, "reason": "no nidem.npz"}
            continue
        with np.load(f, allow_pickle=False) as z:
            if "h" not in z.files or "dates" not in z.files:
                g[f"levels_vs_nidem_{key}"] = {"path": f, "checked": False, "reason": "no h/dates"}
                continue
            nd = np.asarray(z["dates"]).astype(str)
            nh = np.asarray(z["h"], float)
        same_dates = bool(len(nd) == len(rows_) and np.array_equal(nd, extras["dates_all"][rows_]))
        dd = float(np.max(np.abs(nh - L0[rows_]))) if same_dates else None
        okk = same_dates and dd is not None and dd <= TOL_LEVEL
        g[f"levels_vs_nidem_{key}"] = {"path": f, "checked": True, "same_dates": same_dates,
                                       "max_abs_diff_m": dd, "ok": okk}
    h0 = L0[idx_scr]
    lohi = [float(h0.min()), float(h0.max())]
    dr = float(np.max(np.abs(np.asarray(lohi) - np.asarray(res["rango_marea_m"], float))))
    g["screened_level_range_vs_result_json"] = {"mine": lohi, "stored": res["rango_marea_m"],
                                                "max_abs_diff_m": dr, "ok": dr <= TOL_LEVEL}

    # the stored product on F reproduces the published row
    pub, src = published_row(site, cfg.get("published"))
    st = score(extras["kind"], extras["unit_truth_full"], extras["z_stored"][extras["unit_keep_full"]])
    if pub is None:
        g["stored_product_vs_published_row"] = {"checked": False, "reason": "no published CSV found",
                                                "mine": st}
    else:
        dd = max(abs(st[k] - pub[c]) for k, c in (("rmse", "rmse_m"), ("slope", "slope"),
                                                   ("r", "r"), ("bias", "bias_m")))
        okk = int(st["n"]) == int(pub["n"]) and dd <= TOL_SCORE
        g["stored_product_vs_published_row"] = {"checked": True, "source": src, "published": pub,
                                                "mine": st, "max_abs_diff": dd, "ok": okk}

    # the full screened record, mb 0 and 3, production lags: reproduces the
    # stored z; the mb-3 post-filter is invert_series(min_bracket=3)
    t0 = time.time()
    rows_j, info = run_job(sd, "screened", len(idx_scr), 0, seed, "fixed", mbs, check_mb=True)
    t_job = time.time() - t0
    z0 = info["z"][0]
    zs_px = extras["z_stored"][extras["px"]]
    u = sd.unit_col
    fin_m, fin_s = np.isfinite(z0[u]), np.isfinite(zs_px[u])
    flips = int(np.sum(fin_m != fin_s))
    both = fin_m & fin_s
    dz = float(np.max(np.abs(z0[u][both] - zs_px[u][both]))) if both.any() else float("nan")
    st_s = score(extras["kind"], sd.truth, zs_px[u])
    r0 = [r for r in rows_j if r["mb"] == 0][0]
    dsc = max(abs(r0[k] - st_s[k]) for k in ("rmse", "slope", "r", "bias"))
    okk = (flips <= TOL_FLIP_SHARE * len(u)) and (dz <= TOL_Z) and (dsc <= TOL_SCORE or flips > 0)
    g["full_screened_mb0_vs_stored_z"] = {
        "units": int(len(u)), "validity_flips": flips, "max_abs_dz_m": dz,
        "scores_mine": {k: r0[k] for k in ("n", "rmse", "slope", "r", "bias", "coverage")},
        "scores_stored": st_s, "max_abs_score_diff": dsc, "ok": bool(okk)}
    g["mb_post_filter_vs_invert_series"] = {"values_differing": int(info["mismatch"]),
                                            "ok": info["mismatch"] == 0}
    tim = {"gate_job_seconds": round(t_job, 2), "gate_inversion_seconds_mb0": round(info["t_inv"], 3),
           "gate_N": int(len(idx_scr)), "inverted_px": int(sd.Y0.shape[1])}
    if lags == "fixed":
        out_rows.extend(rows_j)

    if lags == "refit":
        t0 = time.time()
        rows = sd.pools["screened"]
        fit = refit_lags(sd, rows)
        t_fit = time.time() - t0
        sd.refit_cache[draw_sha1(rows)] = fit
        tau = fit[0]
        stored = np.asarray(res.get("tau_min", []), float)
        dtau = float(np.max(np.abs(tau - stored))) if len(stored) == len(tau) else None
        ndiff = int(np.sum(fit[2] != sd.tau_fixed))
        g["refit_full_screened_vs_result_json_tau_min"] = {
            "mine_min": tau.tolist(), "stored_min": stored.tolist(), "max_abs_diff_min": dtau,
            "per_pixel_lags_differing_from_production": ndiff, "of": int(len(sd.tau_fixed)),
            "ok": dtau is not None and dtau <= 1e-6, "stop_if_failed": False}
        tim["refit_seconds_full_screened"] = round(t_fit, 2)
    for k, v in g.items():
        if isinstance(v, dict) and v.get("ok") is False and v.get("stop_if_failed", True):
            ok = False
    return g, ok, tim


# ─────────────────────────────────────────────────────────────────────────────
#  run
# ─────────────────────────────────────────────────────────────────────────────

def git_head():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def blas_info():
    info = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = [{k: d.get(k) for k in ("internal_api", "num_threads")}
                               for d in threadpool_info()]
    except Exception:  # noqa: BLE001
        pass
    return info


def run_jobs(sd, jobs, csvlog, seed, lags, mbs, deadline, cost, mb_primary=None, on_progress=None):
    """Run, in order, the jobs (pool, N, replicate) not yet in ``csvlog``,
    appending each one's rows as it finishes; stop before a job whose
    predicted time (``cost`` scaled by the measured/predicted ratio so far)
    would pass ``deadline`` (epoch seconds). Returns (jobs done now, jobs
    skipped because already in the CSV, stop info or None)."""
    mb_primary = max(mbs) if mb_primary is None else int(mb_primary)
    todo = [j for j in jobs if not csvlog.is_done(*j, mbs)]
    n_skip = len(jobs) - len(todo)
    ratios, n_done, stopped = [], 0, None
    left = float(sum(cost(p, N) for p, N, _ in todo))
    for i, (pool, N, rep) in enumerate(todo):
        k_on = float(np.median(ratios)) if ratios else 1.0
        pred_j = cost(pool, N) * k_on
        if time.time() + pred_j > deadline:
            stopped = {"at_job": [pool, int(N), int(rep)], "jobs_left": len(todo) - i,
                       "reason": f"predicted {pred_j:.0f} s would pass --max-minutes"}
            log(f"[{sd.site}] STOP: --max-minutes reached before {pool} N={N} rep {rep} "
                f"({len(todo) - i} job(s) left)")
            break
        rows_out, info = run_job(sd, pool, N, rep, seed, lags, mbs)
        csvlog.append(rows_out)
        n_done += 1
        ratios.append(info["seconds"] / max(cost(pool, N), 1e-6))
        left -= cost(pool, N)
        r = {row["mb"]: row for row in rows_out}
        rp = r[mb_primary]
        eta = left * float(np.median(ratios)) / 60
        log(f"[{sd.site}] {pool:8s} N={N:4d} rep {rep:2d}: mb{mb_primary} RMSE {rp['rmse']:.4f} n {rp['n']:,} "
            f"cov {rp['coverage']:.3f} | mb0 RMSE {r[0]['rmse']:.4f} cov {r[0]['coverage']:.3f} | clear obs "
            f"{rp['median_clear_obs']:.0f}" + (f" | tau {rp['tau_bands_min']}" if lags == "refit" else "")
            + f" | {info['seconds']:.1f} s | {i + 1}/{len(todo)}, ETA {eta:.1f} min")
        if on_progress is not None and n_done % 10 == 0:
            on_progress(n_done, n_skip)
    return n_done, n_skip, stopped


def cost_model(k_inv, P, lags, fit_s, N1):
    """Predicted seconds of one job: inversion (the server prior scaled by
    ``k_inv``) on P pixels, plus a lag fit when refitting."""
    def cost(pool, N):
        c = k_inv * (COST_A_S + COST_B_S * N) * P + JOB_OVERHEAD_S
        if lags == "refit":
            c += fit_s * (0.4 + 0.6 * N / max(N1, 1))
        return c
    return cost


def run(args):
    from pyintertidal import marea

    site, lags = args.site, args.lags
    out = args.out
    guard_out(out)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"{site}_{lags}")
    csv_path, json_path, draws_path = f"{stem}.csv", f"{stem}_settings.json", f"{stem}_draws.npz"
    deadline = _T0 + float(args.max_minutes) * 60.0
    cfg = site_config(site)
    for k in ("extract", "product", "comparison", "nidem", "cube"):
        v = getattr(args, f"{k}_override", None)
        if v:
            cfg[k] = v
    if args.published_csv:
        cfg["published"] = [args.published_csv]
    overrides = {k: getattr(args, f"{k}_override") for k in ("extract", "product", "comparison", "nidem", "cube")
                 if getattr(args, f"{k}_override", None)}
    if args.published_csv:
        overrides["published_csv"] = args.published_csv
    mb_primary = int(args.min_bracket)
    mbs = sorted({0, mb_primary})
    pools_wanted = [p.strip() for p in args.pools.split(",") if p.strip()]
    bad = [p for p in pools_wanted if p not in POOLS]
    if bad:
        raise SystemExit(f"--pools: unknown {bad}; choose from {POOLS}")
    max_px = MAX_PX[lags] if args.max_px is None else int(args.max_px)
    n_grid = parse_n_grid(args.n_grid)
    reps = int(args.reps)
    reps_large = int(args.reps_large) if args.reps_large is not None else max(1, int(math.ceil(reps / 2)))
    log(f"[{site}] v14 scene-count curves: lags {lags}, pools {pools_wanted}, rules mb {mbs} "
        f"(primary {mb_primary}), seed {args.seed}, reps {reps} (N > {LARGE_N}: {reps_large}), "
        f"max {args.max_minutes} min, max px {max_px:,}; threads {blas_info()}")

    sd, meta, extras = load_site(site, cfg, lags, max_px, int(args.seed), pools_wanted)
    compat = {"version": 1, "site": site, "lags": lags, "seed": int(args.seed), "mbs": mbs,
              "pools": {p: int(len(v)) for p, v in sd.pools.items()},
              "n_units_scored": int(len(sd.unit_col)), "n_units_F": int(len(extras["unit_keep_full"])),
              "units_sha1": array_sha1(extras["units_keep"]), "z_stored_sha1": array_sha1(extras["z_stored"]),
              "tau_fixed_sha1": array_sha1(sd.tau_fixed)}
    prev = None
    if os.path.exists(json_path):
        prev = json.load(open(json_path, encoding="utf-8"))
        if prev.get("compat") != _clean(compat):
            raise SystemExit(f"{json_path} was written with other settings:\n  {prev.get('compat')}\n"
                             f"now:\n  {compat}\nrefusing to mix them: use another --out")
    csvlog = CsvLog(csv_path, mbs)
    if csvlog.done:
        log(f"[{site}] resuming: {len(csvlog.done)} job(s) already in {csv_path}")

    settings = prev or {}
    settings.update({
        "site": site, "site_name": SITE_NAMES.get(site, site), "lags": lags, "compat": compat,
        "status": "running", "git_head": git_head(),
        "script": file_info(os.path.relpath(os.path.abspath(__file__), ROOT)),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "blas": blas_info(), "seed": int(args.seed),
        "rules": {"mb_scored": mbs, "mb_primary": mb_primary,
                  "mb_0": "rule of the stored / published 10 m products (made before the rule existed)",
                  f"mb_{marea.QUALITY_MIN_BRACKET}": "pyintertidal.marea.QUALITY_MIN_BRACKET, the current "
                                                     "production rule",
                  "how": "one invert_series fit with min_bracket=0, the rule applied as its own post-filter "
                         "(asserted bitwise against invert_series(min_bracket=mb))"},
        "inversion": {"function": "pyintertidal.marea.invert_series", "kwargs": "defaults (mu_points 50, "
                      "chunk 4000, min_obs 8, min_b 0.15, max_b 2.5, max_a 2.5)",
                      "levels": "boundary h_B(t - tau_px), whole minutes, one call over sorted unique instants",
                      "lo_hi": "the subset's own zero-lag level range (as production takes them from its record)",
                      "y": "raw NDWI (B03-B08)/(B03+B08), NaN -> 0", "c": "SCL in {4,5,6,7}"},
        "lags_mode": {"mode": lags,
                      "text": ("fixed: every pixel keeps its production lag (band lags fitted on the full "
                               "screened record, not re-estimated per subset)" if lags == "fixed" else
                               "refit: band lags re-fitted on every subset (m2a_rasch as reconstruct), "
                               "per-pixel profile re-smoothed")},
        "max_minutes": float(args.max_minutes), "max_px": max_px,
        "input_overrides": overrides,
        **meta,
    })
    settings.setdefault("runs", [])
    this_run = {"started": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:]}
    settings["runs"].append(this_run)

    # gates
    gate_rows = []
    gates, ok, tim = run_gates(site, cfg, sd, extras, mbs, int(args.seed), lags, gate_rows)
    settings["gates"] = gates
    settings["timing"] = {"load_seconds": round(extras["load_seconds"], 1), **tim,
                          "level_calls": sd.levels.calls}
    for k, v in gates.items():
        log(f"[{site}] gate {k}: {'OK' if v.get('ok', True) else 'FAILED'} "
            f"{ {kk: vv for kk, vv in v.items() if kk in ('max_abs_diff_m', 'max_abs_diff', 'validity_flips', 'max_abs_dz_m', 'values_differing', 'max_abs_diff_min', 'checked')} }")
    if not ok and args.soft_gates:
        failed = [k for k, v in gates.items() if isinstance(v, dict) and v.get("ok") is False
                  and v.get("stop_if_failed", True)]
        settings["gates_overridden"] = failed
        log(f"[{site}] WARNING: gate(s) {failed} failed and --soft-gates is set: continuing; the JSON says so")
    elif not ok:
        settings["status"] = "gate_failed"
        this_run["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        write_json(json_path, settings)
        raise SystemExit(f"[{site}] a gate failed; see {json_path}")
    if gate_rows and "screened" in pools_wanted and \
            not csvlog.is_done("screened", len(sd.pools["screened"]), 0, mbs):
        csvlog.append(gate_rows)                # the gate job IS the full screened-record job
    settings["reference_points"] = reference_points(site, cfg, sd, extras)
    for p in settings["reference_points"]:
        o = p.get("on_scored_units")
        log(f"[{site}] reference: {p['label']}: " + (f"{p['scenes']} scenes, n {o['n']:,}, coverage "
            f"{o['coverage']:.3f}, RMSE {o['rmse']:.4f}, slope {o['slope']:.3f}, r {o['r']:.3f}, bias "
            f"{o['bias']:.3f}" if o else f"not available ({p.get('status')})"))

    # plan and budget
    sizes = {p: int(len(v)) for p, v in sd.pools.items()}
    grids = {p: pool_grid(p, sizes[p], n_grid) for p in POOLS if p in pools_wanted}
    N1 = len(sd.pools["screened"])
    prior = (COST_A_S + COST_B_S * tim["gate_N"]) * tim["inverted_px"]
    k_inv = max(tim["gate_inversion_seconds_mb0"] / max(prior, 1e-9), 0.05)
    fit_s = float(tim.get("refit_seconds_full_screened", 0.0))
    cost = cost_model(k_inv, tim["inverted_px"], lags, fit_s, N1)
    budget = deadline - time.time()
    grids_eff, reps_eff, rl_eff, jobs, notes, pred0, pred = shrink_plan(
        grids, sizes, reps, reps_large, cost, budget)
    settings["plan"] = {
        "grid_planned": grids, "reps_planned": reps, "reps_large_planned": reps_large,
        "large_n": LARGE_N, "grid": grids_eff, "reps": reps_eff, "reps_large": rl_eff,
        "jobs": len(jobs), "predicted_minutes_planned": round(pred0 / 60, 2),
        "predicted_minutes": round(pred / 60, 2), "budget_minutes_left": round(budget / 60, 2),
        "shrink_notes": notes, "order": "replicate-major (whole pools, then replicate 0 of every N, ...)",
        "cost_model": {"inversion_s_per_px": f"{k_inv:.3f} x ({COST_A_S * 1e6:g} + {COST_B_S * 1e6:g} N) us",
                       "refit_s": (f"{fit_s:.1f} x (0.4 + 0.6 N / {N1})" if lags == "refit" else None)},
    }
    for n_ in notes:
        log(f"[{site}] PLAN: {n_}")
    log(f"[{site}] plan: {len(jobs)} jobs, predicted {pred / 60:.1f} min, {budget / 60:.1f} min left; "
        f"grid {grids_eff}, reps {reps_eff} (N > {LARGE_N}: {rl_eff})")
    write_json(json_path, settings)

    # jobs
    def progress(n_done_, n_skip_):
        settings["progress"] = {"jobs_done_this_run": n_done_, "jobs_skipped_resume": n_skip_,
                                "jobs_planned": len(jobs)}
        write_json(json_path, settings)

    n_done, n_skip, stopped = run_jobs(sd, jobs, csvlog, int(args.seed), lags, mbs, deadline, cost,
                                       mb_primary=mb_primary, on_progress=progress)

    # finish
    done_jobs = [j for j in jobs if csvlog.is_done(*j, mbs)]
    draws = {}
    for pool, N, rep in done_jobs:
        draws[f"{pool}__N{N}__rep{rep}"] = draw(int(args.seed), site, pool, N, rep, sizes[pool])
    np.savez_compressed(draws_path, **draws,
                        pool_screened_dates=np.asarray(extras["rec_dates"]),
                        pool_all_dates=np.asarray(extras["dates_all"]))
    settings["status"] = ("complete" if len(done_jobs) == len(jobs) else
                          "stopped_by_time" if stopped else "incomplete")
    settings["progress"] = {"jobs_planned": len(jobs), "jobs_done": len(done_jobs),
                            "jobs_done_this_run": n_done, "jobs_skipped_resume": n_skip,
                            "stopped": stopped,
                            "replicates_done": _reps_done(done_jobs, grids_eff, sizes),
                            "elapsed_minutes": round((time.time() - _T0) / 60, 2)}
    this_run["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    this_run["status"] = settings["status"]
    write_json(json_path, settings)
    log(f"[{site}] {settings['status']}: {len(done_jobs)}/{len(jobs)} jobs -> {csv_path}, {json_path}, "
        f"{draws_path}")


def _reps_done(done_jobs, grids, sizes):
    out = {}
    for p, grid in grids.items():
        out[p] = {str(N): sum(1 for q, n, _ in done_jobs if q == p and n == N) for N in grid}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  plot
# ─────────────────────────────────────────────────────────────────────────────

#: the poster palette (poster/preamble/colors.tex)
PAL = {"deep": "#0B3B5A", "teal": "#1B7F86", "sand": "#C08A3E", "slate": "#4A5C6B",
       "ink": "#16202A", "rule": "#C9D3DA", "mist": "#EEF2F5", "sand_text": "#8A5F22"}
POOL_STYLE = {
    "screened": dict(color=PAL["deep"], marker="o", label="MAREA, N scenes drawn from its screened record"),
    "all": dict(color=PAL["teal"], marker="s", label="MAREA, N scenes drawn from all scenes"),
}
REF_STYLE = {
    "marea_published": dict(color=PAL["deep"], marker="D", ms=9, label="MAREA as published (full record)"),
    "nidem_primary": dict(color=PAL["sand"], marker="^", ms=10, label="NIDEM as published (all scenes)"),
    "nidem_sensitivity": dict(color=PAL["sand"], marker="v", ms=10, label="NIDEM on MAREA's record"),
}
SUMMARY_CSV = "results/method_comparison_same_scenes/same_scenes_summary.csv"
SUMMARY_METHODS = {"marea_published": "MAREA",
                   "nidem_primary": "NIDEM (all observations, as published)",
                   "nidem_sensitivity": "NIDEM (on MAREA's record)"}


def summarize(df, pool):
    """Per N of one pool: replicates, median and 10-90 % of RMSE (over the
    replicates with a score), median coverage and clear observations."""
    import pandas as pd

    rows = []
    for N, g in df[df["pool"] == pool].groupby("N"):
        v = g["rmse"].to_numpy(float)
        v = v[np.isfinite(v)]
        k = len(g)
        cov = g["coverage"].to_numpy(float)
        rows.append(dict(
            N=int(N), k=k, k_valid=len(v),
            rmse=float(np.median(v)) if len(v) and len(v) >= math.ceil(k / 2) else np.nan,
            rmse_lo=float(np.quantile(v, 0.1)) if len(v) >= 3 else np.nan,
            rmse_hi=float(np.quantile(v, 0.9)) if len(v) >= 3 else np.nan,
            cov=float(np.median(cov)), cov_lo=float(np.quantile(cov, 0.1)) if k >= 3 else np.nan,
            cov_hi=float(np.quantile(cov, 0.9)) if k >= 3 else np.nan,
            clear=float(np.median(g["median_clear_obs"].to_numpy(float)))))
    return pd.DataFrame(rows).sort_values("N") if rows else pd.DataFrame(
        columns=["N", "k", "k_valid", "rmse", "rmse_lo", "rmse_hi", "cov", "cov_lo", "cov_hi", "clear"])


def plot_inputs(out, site, lags, mb):
    """Curves, reference points and notes of one site for the figures."""
    import pandas as pd

    df = pd.read_csv(os.path.join(out, f"{site}_{lags}.csv"))
    df = df[df["mb"] == int(mb)]
    jp = os.path.join(out, f"{site}_{lags}_settings.json")
    st = json.load(open(jp, encoding="utf-8")) if os.path.exists(jp) else {}
    curves = {p: summarize(df, p) for p in POOLS if (df["pool"] == p).any()}
    refs, notes = [], []
    pts = {p.get("key"): p for p in st.get("reference_points", [])}
    summ = None
    # tonight's summary holds the production (10 m) values: it may stand in
    # for a missing point only when this run read the production inputs
    if os.path.exists(SUMMARY_CSV) and not st.get("input_overrides"):
        summ = pd.read_csv(SUMMARY_CSV)
        summ = summ[summ["site"] == site]
    for key in ("marea_published", "nidem_primary", "nidem_sensitivity"):
        p = pts.get(key)
        o = (p or {}).get("on_scored_units") or {}
        if o.get("rmse") is not None and np.isfinite(float(o["rmse"])):
            refs.append(dict(key=key, N=p["scenes"], rmse=o["rmse"], cov=o["coverage"], n=o["n"],
                             source="same pixel set"))
            continue
        row = None if summ is None else summ[summ["method"] == SUMMARY_METHODS[key]]
        if row is not None and len(row):
            r0 = row.iloc[0]
            try:
                n_sc = int(float(r0["scenes"]))
                if np.isfinite(float(r0["rmse_m"])):
                    refs.append(dict(key=key, N=n_sc, rmse=float(r0["rmse_m"]), cov=np.nan,
                                     n=int(r0["n"]), source="own pixels"))
                    notes.append(f"{REF_STYLE[key]['label']}: from {SUMMARY_CSV} (its own pixels, "
                                 f"n = {int(r0['n']):,})")
                    continue
            except (TypeError, ValueError):
                pass
            notes.append(f"{REF_STYLE[key]['label']}: {r0['scenes']} (no point)")
        elif p and p.get("status"):
            notes.append(f"{REF_STYLE[key]['label']}: {p['status']} (no point)")
        else:
            notes.append(f"{REF_STYLE[key]['label']}: not available (no point)")
    return dict(df=df, curves=curves, refs=refs, notes=notes, settings=st, site=site, lags=lags, mb=mb)


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["rule"])
    ax.tick_params(colors=PAL["slate"], labelcolor=PAL["ink"], length=3)
    ax.grid(True, color=PAL["rule"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _log_n_axis(ax, nmax, nmin=10.0):
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
    ax.set_xscale("log")
    lo = min(8.0, 0.8 * float(nmin))
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 400) if lo <= t <= nmax * 1.05]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, nmax * 1.25)


#: medians whose median coverage is below this share of the set are drawn
#: with an open marker: their RMSE rests on a minority of the set
LOW_COVERAGE = 0.5


def _draw_curves(ax_r, ax_c, D, x="N", refs=True, labels=True):
    """RMSE (median, 10-90 % band) and coverage against N (or clear
    observations) for both pools, and the reference points. Returns legend
    handles (proxies)."""
    from matplotlib.lines import Line2D

    handles = {}
    for pool, cv in D["curves"].items():
        if cv.empty:
            continue
        stl = POOL_STYLE[pool]
        xs = cv[x].to_numpy(float) if x == "N" else cv["clear"].to_numpy(float)
        ok = np.isfinite(cv["rmse_lo"].to_numpy(float))
        if ok.any():
            ax_r.fill_between(xs[ok], cv["rmse_lo"].to_numpy(float)[ok], cv["rmse_hi"].to_numpy(float)[ok],
                              color=stl["color"], alpha=0.14, lw=0)
        rm = cv["rmse"].to_numpy(float)
        low = cv["cov"].to_numpy(float) < LOW_COVERAGE
        ax_r.plot(xs, rm, color=stl["color"], lw=2.0, solid_capstyle="round", zorder=3)
        ax_r.plot(xs[~low], rm[~low], ls="none", marker=stl["marker"], ms=6.5, color=stl["color"],
                  mec="white", mew=1.2, zorder=4)
        ax_r.plot(xs[low], rm[low], ls="none", marker=stl["marker"], ms=6.5, mfc="white",
                  mec=stl["color"], mew=1.6, zorder=4)
        handles[pool] = Line2D([], [], color=stl["color"], lw=2.0, marker=stl["marker"], ms=6.5,
                               mec="white", mew=1.2, label=stl["label"])
        if low.any() and np.isfinite(rm[low]).any():
            handles["low"] = Line2D([], [], ls="none", marker="o", ms=6.5, mfc="white", mec=PAL["slate"],
                                    mew=1.6, label=f"open marker: under {LOW_COVERAGE:.0%} of the set has "
                                                   "an estimate")
        if ax_c is not None:
            okc = np.isfinite(cv["cov_lo"].to_numpy(float))
            if okc.any():
                ax_c.fill_between(xs[okc], 100 * cv["cov_lo"].to_numpy(float)[okc],
                                  100 * cv["cov_hi"].to_numpy(float)[okc], color=stl["color"], alpha=0.14, lw=0)
            ax_c.plot(xs, 100 * cv["cov"], color=stl["color"], lw=2.0, marker=stl["marker"], ms=5.5,
                      mec="white", mew=1.0, zorder=3)
        if labels and x == "N" and len(xs) and np.isfinite(cv["rmse"].to_numpy(float)[-1]):
            # the whole pool's size at the curve's end (the axis ticks do not carry it),
            # with its RMSE when there is room (labels="value")
            end = int(cv["N"].iloc[-1])
            txt = f"{end} scenes, {cv['rmse'].iloc[-1]:.3f} m" if labels == "value" else f"{end}"
            # right-aligned just under the end marker: never runs into the next panel
            ax_r.annotate(txt, (xs[-1], cv["rmse"].iloc[-1]), xytext=(5, -9), textcoords="offset points",
                          fontsize=9.5, color=PAL["slate"], ha="right", va="top")
    if refs and x == "N":
        for r in D["refs"]:
            if r["N"] is None or not np.isfinite(r["rmse"]):
                continue
            stl = REF_STYLE[r["key"]]
            h, = ax_r.plot([r["N"]], [r["rmse"]], ls="none", marker=stl["marker"], ms=stl["ms"],
                           color=stl["color"], mec="white", mew=1.5, zorder=5, label=stl["label"])
            handles[r["key"]] = h
            if ax_c is not None and np.isfinite(r["cov"]):
                ax_c.plot([r["N"]], [100 * r["cov"]], ls="none", marker=stl["marker"], ms=stl["ms"] - 2,
                          color=stl["color"], mec="white", mew=1.2, zorder=5)
    return handles


def _set_line(D):
    st = D["settings"]
    fs = st.get("fixed_set", {})
    unit = fs.get("unit", "unit")
    n = fs.get("n_units_scored")
    nF = fs.get("n_units_F")
    if fs.get("subsample"):
        return f"{n:,} {unit}s (random subsample of the {nF:,} of the published set)"
    if unit == "RTK point":
        return f"{n:,} RTK points on {fs.get('n_pixels_inverted', 0):,} pixels (the published set)"
    return f"{n:,} {unit}s (the published set)" if n else "the published set"


def _res_m(D):
    """The grid resolution of a run (m): the cube's pixel at the Dutch sites, 10 m at Villaviciosa."""
    return float(D["settings"].get("fixed_set", {}).get("pixel_m") or 10.0)


def _truth_line(site):
    return ("RTK-GNSS (development split, Alicante datum)" if site == "villaviciosa"
            else "Rijkswaterstaat Vaklodingen (NAP)")


def _reps_line(D):
    p = D["settings"].get("plan", {})
    st = D["settings"].get("status", "?")
    r, rl = p.get("reps"), p.get("reps_large")
    s = (f"median and 10-90 % band over {r} random subsets per N ({rl} above N = {LARGE_N})"
         if r else "median and 10-90 % band over the random subsets per N")
    if st != "complete":
        s += f"; run status: {st}"
    return s


def _lag_line(lags):
    return ("Lags fixed at the production band lags (fitted on the full screened record, not re-estimated "
            "per subset)" if lags == "fixed" else "Band lags re-fitted on every subset, as production fits them")


def site_figure(D):
    import matplotlib.pyplot as plt

    site, lags, mb = D["site"], D["lags"], D["mb"]
    nmax = max([int(cv["N"].max()) for cv in D["curves"].values() if not cv.empty] + [20])
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.25], width_ratios=[1.45, 1.0],
                          hspace=0.10, wspace=0.16, left=0.07, right=0.985, top=0.775, bottom=0.175)
    ar = fig.add_subplot(gs[0, 0])
    ac = fig.add_subplot(gs[1, 0], sharex=ar)
    br = fig.add_subplot(gs[0, 1], sharey=ar)
    bc = fig.add_subplot(gs[1, 1], sharex=br, sharey=ac)
    handles = _draw_curves(ar, ac, D, x="N", labels="value")
    _draw_curves(br, bc, D, x="clear", refs=False, labels=False)
    for ax in (ar, ac, br, bc):
        _style_axes(ax)
    lo_y, hi_y = ar.get_ylim()                  # room for the end labels under the lowest point
    ar.set_ylim(lo_y - 0.07 * (hi_y - lo_y), hi_y)
    _log_n_axis(ar, nmax)
    clear = [cv["clear"].to_numpy(float) for cv in D["curves"].values() if not cv.empty]
    clear = np.concatenate(clear) if clear else np.array([10.0, 20.0])
    clear = clear[np.isfinite(clear) & (clear > 0)]
    _log_n_axis(br, float(clear.max()) if len(clear) else 20.0,
                nmin=float(clear.min()) if len(clear) else 10.0)
    ar.set_ylabel("Elevation RMSE (m)", color=PAL["ink"], fontsize=11.5)
    ac.set_ylabel("Coverage (%)", color=PAL["ink"], fontsize=11.5)
    ac.set_xlabel("Number of scenes used, N", color=PAL["ink"], fontsize=11.5)
    bc.set_xlabel("Median clear observations per scored pixel", color=PAL["ink"], fontsize=11.5)
    ac.set_ylim(0, 104)
    plt.setp(ar.get_xticklabels(), visible=False)
    plt.setp(br.get_xticklabels(), visible=False)
    plt.setp(br.get_yticklabels(), visible=False)
    plt.setp(bc.get_yticklabels(), visible=False)
    ar.set_title("against the number of scenes", loc="left", fontsize=11, color=PAL["slate"])
    br.set_title("against the clear observations per pixel", loc="left", fontsize=11, color=PAL["slate"])
    fig.text(0.07, 0.955, f"{SITE_NAMES.get(site, site)}: elevation error vs number of scenes "
             f"(MAREA, {_res_m(D):g} m)",
             fontsize=16, color=PAL["deep"], weight="bold", ha="left")
    fig.text(0.07, 0.915, f"Truth: {_truth_line(site)} | scored on one fixed set: {_set_line(D)}",
             fontsize=10.5, color=PAL["slate"], ha="left")
    order = [k for k in ("screened", "all", "marea_published", "nidem_primary", "nidem_sensitivity", "low")
             if k in handles]
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order], loc="upper left",
               bbox_to_anchor=(0.065, 0.905), ncol=3, frameon=False, fontsize=10, handlelength=2.2,
               columnspacing=1.6)
    foot = [f"{_lag_line(lags)}; quality rule min_bracket = {mb}"
            + (" (the published products used 0)" if mb else " (the rule of the published products)") + ".",
            f"{_reps_line(D)[0].upper()}{_reps_line(D)[1:]}. RMSE, median-centred, over the part of the set "
            "with an estimate; coverage = share of the set with an estimate.",
            "Pool 'all scenes' = every cube scene with an overpass time and a boundary level; the per-pixel "
            "clear-sky mask (SCL 4-7) still applies, so many of its scenes add few observations."]
    same = [r for r in D["refs"] if r["source"] == "same pixel set" and r["key"] != "marea_published"]
    if same:
        foot.append("NIDEM = ITEM/NIDEM waterline-contour DEM (v7), placed at the number of scenes it used and "
                    "scored on the same set; MAREA as published = the stored product (full record, rule 0).")
    foot += D["notes"]
    fig.text(0.07, 0.015, "\n".join(foot), fontsize=8.6, color=PAL["slate"], ha="left", va="bottom",
             linespacing=1.35)
    return fig


def combined_figure(Ds, lags, mb):
    """One column per site, always the four sites in SITE_ORDER (a site
    without results yet keeps an empty, labelled column) so the layout is the
    same whatever has finished."""
    import matplotlib.pyplot as plt

    by_site = {D["site"]: D for D in Ds}
    cols = list(SITE_ORDER) + [s for s in by_site if s not in SITE_ORDER]
    n = len(cols)
    fig = plt.figure(figsize=(4.3 * n + 1.0, 8.2))
    gs = fig.add_gridspec(2, n, height_ratios=[3.0, 1.25], hspace=0.10, wspace=0.22,
                          left=0.05, right=0.99, top=0.765, bottom=0.175)
    handles = {}
    for j, site in enumerate(cols):
        ar = fig.add_subplot(gs[0, j])
        ac = fig.add_subplot(gs[1, j], sharex=ar)
        for ax in (ar, ac):
            _style_axes(ax)
        ar.text(0.0, 1.10, SITE_NAMES.get(site, site), transform=ar.transAxes, fontsize=13.5,
                color=PAL["deep"], weight="bold", va="bottom", ha="left")
        ac.set_xlabel("Number of scenes, N", color=PAL["ink"], fontsize=11)
        if j == 0:
            ar.set_ylabel("Elevation RMSE (m)", color=PAL["ink"], fontsize=11.5)
            ac.set_ylabel("Coverage (%)", color=PAL["ink"], fontsize=11.5)
        ac.set_ylim(0, 104)
        D = by_site.get(site)
        if D is None:
            _log_n_axis(ar, 465)
            ar.text(0.5, 0.5, "no results yet", transform=ar.transAxes, ha="center", va="center",
                    fontsize=12, color=PAL["slate"])
            ar.set_yticks([])
            plt.setp(ar.get_xticklabels(), visible=False)
            continue
        h = _draw_curves(ar, ac, D, x="N", labels=True)
        for k, v in h.items():
            handles.setdefault(k, v)
        nmax = max([int(cv["N"].max()) for cv in D["curves"].values() if not cv.empty] + [20])
        _log_n_axis(ar, nmax)
        plt.setp(ar.get_xticklabels(), visible=False)
        st = D["settings"].get("fixed_set", {})
        unit = "points" if D["site"] == "villaviciosa" else "px"
        ar.text(0.0, 1.025, f"{_truth_line(D['site']).split(' (')[0]}, n = {st.get('n_units_scored', 0):,} "
                f"{unit}", transform=ar.transAxes, fontsize=9.5, color=PAL["slate"], va="bottom", ha="left")
    res = sorted({_res_m(D) for D in Ds})
    fig.text(0.05, 0.955, "Elevation error vs number of scenes used (MAREA, "
             + "/".join(f"{r:g}" for r in res) + " m)", fontsize=17,
             color=PAL["deep"], weight="bold", ha="left")
    order = [k for k in ("screened", "all", "marea_published", "nidem_primary", "nidem_sensitivity", "low")
             if k in handles]
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order], loc="upper left",
               bbox_to_anchor=(0.045, 0.94), ncol=3, frameon=False, fontsize=10.5, handlelength=2.2,
               columnspacing=1.8)
    notes = []
    for D in Ds:
        for t in D["notes"]:
            notes.append(f"{SITE_NAMES.get(D['site'], D['site'])}: {t}")
        stt = D["settings"].get("status")
        if stt and stt != "complete":
            notes.append(f"{SITE_NAMES.get(D['site'], D['site'])}: run status {stt}")
        fs = D["settings"].get("fixed_set", {})
        if fs.get("subsample"):
            notes.append(f"{SITE_NAMES.get(D['site'], D['site'])}: scored on a random subsample of "
                         f"{fs['n_units_scored']:,} of the {fs['n_units_F']:,} pixels of the published set")
    reps = {D["site"]: (D["settings"].get("plan", {}).get("reps"), D["settings"].get("plan", {}).get("reps_large"))
            for D in Ds}
    if len(set(reps.values())) == 1:
        r, rl = next(iter(reps.values()))
        reps_txt = f"{r} random subsets per N ({rl} above N = {LARGE_N})"
    else:
        reps_txt = "random subsets per N: " + ", ".join(
            f"{SITE_NAMES.get(s, s)} {r} ({rl} above N = {LARGE_N})" for s, (r, rl) in reps.items())
    foot = [f"{_lag_line(lags)}; quality rule min_bracket = {mb}"
            + (" (the published products used 0)." if mb else " (the rule of the published products)."),
            f"Median and 10-90 % band over {reps_txt}. Each site is scored on one fixed set (truth & published "
            "MAREA); RMSE, median-centred, over the part of the set with an estimate; coverage = share of the "
            "set with an estimate.",
            "Pool 'all scenes' keeps the per-pixel clear-sky mask (SCL 4-7). NIDEM points are scored on the "
            "same set unless noted."] + notes
    fig.text(0.05, 0.012, "\n".join(foot), fontsize=8.6, color=PAL["slate"], ha="left", va="bottom",
             linespacing=1.35)
    return fig


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    sites = [s for s in (args.sites or SITE_ORDER)
             if os.path.exists(os.path.join(out, f"{s}_{args.lags}.csv"))]
    if not sites:
        raise SystemExit(f"no {out}/<site>_{args.lags}.csv to plot")
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.titlepad": 6})
    Ds = []
    for s in sites:
        D = plot_inputs(out, s, args.lags, args.mb)
        if D["df"].empty:
            log(f"{s}: no rows with mb {args.mb}; skipped")
            continue
        Ds.append(D)
        fig = site_figure(D)
        stem = os.path.join(out, f"scene_count_{s}_{args.lags}_mb{args.mb}")
        fig.savefig(stem + ".png", dpi=200)
        fig.savefig(stem + ".pdf")
        plt.close(fig)
        log(f"-> {stem}.png/.pdf")
    if Ds:
        fig = combined_figure(Ds, args.lags, args.mb)
        stem = os.path.join(out, f"scene_count_all_sites_{args.lags}_mb{args.mb}")
        fig.savefig(stem + ".png", dpi=200)
        fig.savefig(stem + ".pdf")
        plt.close(fig)
        log(f"-> {stem}.png/.pdf")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    from pyintertidal import marea

    ap = argparse.ArgumentParser(prog="python -m experiments.v14_scene_count_curves",
                                 description="MAREA elevation error against the number of scenes used.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="curves of one site (resumable)")
    r.add_argument("site", choices=list(SITE_NAMES))
    r.add_argument("--pools", default="screened,all", help="screened,all (default both)")
    r.add_argument("--n-grid", default=None,
                   help="N values: '10,20,40' for both pools or 'screened=10,20;all=10,50'; the whole pool "
                        "is always added (default: the built-in grids)")
    r.add_argument("--reps", type=int, default=10, help="replicates per N (default 10)")
    r.add_argument("--reps-large", type=int, default=None,
                   help=f"replicates per N above N = {LARGE_N} (default: half of --reps)")
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--lags", choices=("fixed", "refit"), default="fixed")
    r.add_argument("--min-bracket", type=int, default=marea.QUALITY_MIN_BRACKET,
                   help="primary quality rule (default QUALITY_MIN_BRACKET = 3); 0 is always scored too")
    r.add_argument("--max-minutes", type=float, default=60.0, help="wall-time cap of this process")
    r.add_argument("--max-px", type=int, default=None,
                   help=f"subsample the fixed set above this many units (default fixed {MAX_PX['fixed']:,}, "
                        f"refit {MAX_PX['refit']:,}; 0 = never)")
    r.add_argument("--soft-gates", action="store_true",
                   help="a failed gate is logged and recorded in the JSON instead of stopping the run "
                        "(only after reading why it failed)")
    r.add_argument("--out", default=OUT)
    r.add_argument("--product-dir", dest="product_override", default=None,
                   help="folder with the production marea.npz/result.json (e.g. a laptop mirror)")
    r.add_argument("--extract", dest="extract_override", default=None)
    r.add_argument("--comparison-dir", dest="comparison_override", default=None)
    r.add_argument("--nidem-dir", dest="nidem_override", default=None)
    r.add_argument("--cube", dest="cube_override", default=None)
    r.add_argument("--published-csv", default=None,
                   help="CSV with the published MAREA 'all' row for the gate (default: "
                        "results/method_comparison_same_scenes/<site>.csv, else results/method_comparison/<site>.csv)")
    p = sub.add_parser("plot", help="per-site and combined figures from the CSVs")
    p.add_argument("--out", default=OUT)
    p.add_argument("--lags", choices=("fixed", "refit"), default="fixed")
    p.add_argument("--mb", type=int, default=marea.QUALITY_MIN_BRACKET,
                   help="which quality rule to draw (default 3; 0 = the rule of the published products)")
    p.add_argument("--sites", nargs="*", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
