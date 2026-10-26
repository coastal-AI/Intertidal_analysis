# -*- coding: utf-8 -*-
"""V16 - MAREA elevation error against the LENGTH of the time window (10 m).

Question (first author, 2026-10-20)
-----------------------------------
Does a longer archive give MAREA a better elevation, and how short a window
still works? One piece of the case for MAREA as a method in its own right
(next to the fewer-images curves of v14/v15 and the settings factorial of
V3).

Windows
-------
* nested windows that END on 2025-12-31: W1 = 2025, W2 = 2024-2025,
  W3 = 2023-2025 at the four sites; at Villaviciosa, whose extraction covers
  2016-2025, also W5 (2021-2025), W7 (2019-2025) and W10 (2016-2025);
* single calendar years 2023, 2024, 2025 (``--years``). Y2025 holds the
  same scenes as W1: it is computed once and written under both labels
  (``same_scenes_as``).
A window holds EVERY extraction scene inside it with an overpass time and a
finite boundary level (v14's 'all' pool, i.e. NIDEM's record, restricted to
the window's dates); the per-pixel clear-sky mask (SCL 4-7) still applies.
A window runs only where the record has scenes in every one of its years
(the Dutch records are 2023-2025, so W5/W7/W10 are listed as not covered).

Boundary levels
---------------
Dutch sites: v14's level table (v7.build_record's boundary: EOT20 + the
production gauges, period 2023-2025). Villaviciosa: the production
provider, PyTMDBoundary EOT20 at the site centroid (the very object
v7.build_record builds), evaluated at the 2016-2022 overpass times too. It
is a global harmonic model, so it covers every year; the run checks it per
year (scenes, scenes with an overpass time, scenes with a finite level,
level range) and declares it in the JSON; a year left without levels would
make its windows not covered. The 2023-2025 part of the extended record
must equal v7.build_record's record (gate).

Lags and inversion: exactly v14's jobs (``--lags refit`` or ``fixed``)
---------------------------------------------------------------------
Every window is v14's own job on exactly its scenes
(:func:`experiments.v14_scene_count_curves.run_job` on a pool that IS the
window, so N = the whole pool and nothing is drawn). ``refit`` (default):
band lags re-fitted with m2a_rasch on production's seeded lag-fit pixels
(configs/m2.yaml, <= 1200 px per band, over all intertidal pixels),
negative bands clipped to 0, the per-pixel profile re-smoothed and rounded
to the minute. ``fixed``: every pixel keeps its production lag (marea.npz
tau_px_min), so a window changes only the scenes the elevation is fitted
on: the pair refit / fixed separates the lag-estimation error from the
elevation-estimation error. Production inversion (invert_series defaults),
lo/hi = the window's own zero-lag level range. Both quality rules come from
one fit: mb 0 (the rule of the published 10 m products) and mb 3 (the
current production rule, QUALITY_MIN_BRACKET). The fixed variant writes
the stem <site>_fixed.* and is scored on v14's fixed set (200,000 px at the
Dutch sites, which holds the refit run's 100,000: v14's subsamples are
nested).

One fixed set per site
----------------------
v14's set F (finite truth after the production range filter AND finite
published MAREA) with v14's seeded random subsample at the Dutch sites
(``--max-px`` 100,000, v14's refit default): every window is scored on the
same units as v14's curves. Metrics as v14 (RTK: ols_against_truth; Dutch:
v7.vscore, median-centred RMSE); coverage = share of the set with an
estimate. Reported per window and rule: RMSE, bias, slope, r, coverage, n,
the median clear observations per scored unit (v14's definition) and per
inverted pixel, the level range, the band lags (fitted and applied), the
per-pixel lag median and maximum, seconds. F is selected by the published
MAREA, so the coverage is also bounded on a METHOD-NEUTRAL set N0 (every
keep pixel with truth after the range filter; every RTK point in keep):
F holds n_F of its n_N0 units, so coverage on N0 lies in
[cov_F n_F / n_N0, cov_F n_F / n_N0 + 1 - n_F / n_N0] (columns
coverage_neutral_lo / _hi; F is 99.6-100 % of N0 at the four sites).

Confidence intervals (``ci``)
-----------------------------
Paired cluster bootstrap (experiments/paired_bootstrap.py) of every window
against W3 (RMSE, bias, slope, coverage) and, where both lag modes ran, of
refit against fixed lags on the same window, on the units finite in both:
clusters = the pixel of each RTK point (Villaviciosa, ~120 pixels:
indicative) or 1 km blocks of the grid (Dutch sites); 2,000 resamples, a
fixed seed, 95 % percentile intervals -> <site>_window_ci.csv; ``plot``
draws them as bars (the interval of the difference from W3, around each
window's value).

Gates (a failed gate stops the run; ``--soft-gates`` continues and says so)
--------------------------------------------------------------------------
* v14's own gates (level tables, the published row, the stored z
  reproduced with mb 0, the mb-3 post-filter == invert_series(min_bracket=3));
* F is v14's: n and sha1 of the scored units, of the stored z and of the
  production lags equal the compat block of v14's refit settings JSON;
* the windows are nested and the single years partition W3;
* W3's scenes are v14's 'all' pool, and W3 REPRODUCES v14's 'all'-pool
  full-record point (the same lag mode, same F) read from v14's
  <site>_<lags>.csv: N, n, coverage, clear observations, level range and
  band lags exactly, RMSE / bias / slope / r to 1e-9 (bitwise equality is
  reported: the laptop and the server already differ in the 15th digit of
  r for the same job);
* Villaviciosa extension: the 2023-2025 part of the extended record equals
  v7.build_record's (dates, rows, times, levels), and the streamed NDWI and
  clear flags (inverted and lag-fit pixels) and the level table on those
  scenes equal v14's, bitwise. Per year the extension also reports a
  radiometric proxy read from the extraction itself (share of clear
  observations; 5th / 50th / 95th percentiles of the NDWI of the clear
  observations of the lag-fit pixels): a year whose percentiles move by more
  than 0.10 from the 2023-2025 median is flagged (an L2A +1000 DN offset
  applied in some years and not others compresses the NDWI by far more), in
  the JSON and on the figure. The satellites are not recorded in the
  extraction; 2016 is S2A only.

NIDEM hook
----------
If experiments/v15_nidem_scene_curves.py exists and exposes one of
NIDEM_HOOK_NAMES (or the function named by ``--nidem-func``), NIDEM is
computed on the same windows (the same dates) and scored on the same set;
its rows go to <site>_nidem.csv and the figure draws them. Otherwise the
JSON records the TODO (NIDEM_TODO) and the figure draws MAREA alone. The
hook is called with the keywords its signature accepts among: site, dates,
rows (extraction row indices), t_real, h (zero-lag boundary levels),
units_keep (keep index of every scored unit), px_keep (keep index of every
inverted pixel), out_dir, truth and kind (the units' truth and v14.score's
kind, for the hook's own gate) and pixel_rule = 'item' (v15's production
NIDEM, ITEM's pixel mask, as before the hook had a rule; v19's E1 asks for
the common mask instead). A hook that refuses an input (SystemExit) is
recorded as an error of that window, never raised. v15 exposes nidem_on_scenes (2026-10-26):
v15's own NIDEM path on the window's scenes, gated on MAREA's record against
the stored sensitivity product. It returns NIDEM's elevation per scored unit, per
inverted pixel or per extraction keep pixel: an array, or a dict with
'z_units', 'z_px' or 'z_keep'; a NOT COMPUTABLE window (an EmptyIntervalError
of pyintertidal.waterline_contour, or a dict whose 'status' starts with
'NOT COMPUTABLE') is written as such (n 0, coverage 0). Each NIDEM row is
also scored, with MAREA, on their COMMON units (finite MAREA mb-k and finite
NIDEM: n_common, rmse_common, marea_rmse_common, ...), one row per rule k,
so the two methods are compared on the same units. Any other runner can
instead read <site>_windows.json and write <site>_nidem.csv (NIDEM_COLUMNS):
``plot`` draws whatever that file holds.

Outputs (new files only, under ``--out``, default results/window_curves)
------------------------------------------------------------------------
<site>.csv                 one row per (window, rule)  [<site>_fixed.* with --lags fixed]
<site>_settings.json       settings, record and boundary coverage, windows,
                           gates, v14's reference points, timing, memory
<site>_units.npz           every window's estimate on the scored units
                           (z__<window>__mb<k>, float64) with units_keep,
                           units_flat, shape, truth, unit_col, kind: saved
                           before the window's CSV rows, for ``ci``
<site>_windows.json        the windows' scenes: dates, extraction rows and,
                           inside the 'all' pool, pool indices + draw_sha1
                           (the hand-off for NIDEM on the same windows)
<site>_nidem.csv           NIDEM per window and rule (hook or another runner)
<site>_window_ci.csv       ``ci``: paired bootstrap intervals
window_curves_<lang>_mb<k>.{png,pdf}, window_curves_summary.csv (``plot``)

Run from the repository root:
    python -X utf8 -m experiments.v16_window_curves run villaviciosa
    python -X utf8 -m experiments.v16_window_curves run villaviciosa --lags fixed
    python -X utf8 -m experiments.v16_window_curves ci
    python -X utf8 -m experiments.v16_window_curves plot
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # BLAS threads per process (one process per site); v14's server runs used
    # 2, and the W3 gate compares with them. A value already set wins.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402  (also chdir to ROOT)
from experiments import paired_bootstrap as pbs  # noqa: E402

OUT = "results/window_curves"
LAGS = ("refit", "fixed")
END_YEAR = 2025
NESTED_YEARS = (1, 2, 3, 5, 7, 10)
SINGLE_YEARS = (2023, 2024, 2025)
#: the window that must reproduce v14's 'all'-pool full-record point
GATE_WINDOW = "W3"
#: sites whose record may be extended before v14's 2023-2025 'all' pool
#: (the boundary is a global model; the Dutch boundaries use gauges cached
#: for 2023-2025 only)
EXTENDABLE = ("villaviciosa",)
#: where v14's refit CSV is looked for (first found): the server's own
#: output, then the laptop copy of the 2026-10-20 server run
V14_REF_DIRS = ("results/scene_count_curves", "results/scene_count_curves/server_2026-10-20")
TOL_W3 = 1e-9
W3_EXACT = ("N", "n", "n_set", "coverage", "median_clear_obs", "lo_m", "hi_m", "tau_bands_min")
W3_TOL = ("rmse", "bias", "slope", "r")
COMPAT_KEYS = ("seed", "n_units_scored", "n_units_F", "units_sha1", "z_stored_sha1", "tau_fixed_sha1")
#: safety factor on a window's predicted seconds (W3's, scaled by N)
COST_SAFETY = 1.3
CSV_COLUMNS = ("site", "window", "kind", "n_years", "start", "end", "N", "mb", "lags",
               "rmse", "bias", "slope", "r", "coverage", "coverage_neutral_lo", "coverage_neutral_hi",
               "n", "n_set", "median_clear_obs", "median_clear_obs_px", "lo_m", "hi_m",
               "tau_bands_min", "tau_used_bands_min", "tau_px_median_min", "tau_px_max_min",
               "inversion_seconds", "seconds", "scenes_sha1", "same_scenes_as")
#: one NIDEM row per (window, MAREA rule mb): NIDEM's own scores on F, then
#: both methods on their common units (finite NIDEM and finite MAREA mb)
NIDEM_COLUMNS = ("site", "window", "kind", "n_years", "start", "end", "N", "mb", "product", "status",
                 "rmse", "bias", "slope", "r", "coverage", "coverage_neutral_lo", "coverage_neutral_hi",
                 "n", "n_set", "n_common", "rmse_common", "bias_common", "slope_common",
                 "marea_rmse_common", "marea_bias_common", "marea_slope_common",
                 "seconds", "scenes_sha1", "same_scenes_as", "hook")
INT_COLS = {"n_years", "N", "mb", "n", "n_set", "n_common"}
STR_COLS = {"site", "window", "kind", "start", "end", "tau_bands_min", "tau_used_bands_min",
            "scenes_sha1", "same_scenes_as", "product", "hook", "status", "lags"}
#: a NIDEM window that cannot be computed (pre-flight or an empty interval)
NOT_COMPUTABLE = "NOT COMPUTABLE"
#: the radiometric proxy of the extension: flag a year whose NDWI percentile
#: (5th or 95th of the clear observations) moves by more than this from the
#: 2023-2025 median
NDWI_SHIFT_FLAG = 0.10
CI_COLUMNS = ("site", "comparison", "lags_a", "window_a", "lags_b", "window_b", "mb", "stat",
              "n_units", "n_clusters", "cluster_rule", "a", "b", "diff", "ci_lo", "ci_hi",
              "share_diff_below_0", "n_valid_resamples", "n_boot", "seed", "level", "indicative")
NIDEM_HOOK_MODULE = "experiments.v15_nidem_scene_curves"
NIDEM_HOOK_FILE = os.path.join(ROOT, "experiments", "v15_nidem_scene_curves.py")
NIDEM_HOOK_NAMES = ("nidem_on_scenes", "nidem_on_subset", "run_nidem_on_scenes",
                    "run_nidem_subset", "nidem_subset")
NIDEM_TODO = ("TODO (NIDEM on the same windows). experiments/v15_nidem_scene_curves.py (another workflow; "
              "as of 2026-10-20) runs NIDEM on subsets of the 'all' pool only inside its run() "
              "(build_subsets(extra=...), one pass over the cube, stage2), so it exposes no function for a "
              "given scene subset. Either (a) v15 exposes one of " + ", ".join(NIDEM_HOOK_NAMES) + " taking "
              "any of site, dates, rows, t_real, h, units_keep, px_keep, out_dir, truth, kind and returning "
              "NIDEM per "
              "scored unit, inverted pixel or keep pixel; then re-run 'run' (the MAREA windows are resumed, "
              "only NIDEM is computed) and 'plot'; or (b) v15 reads <out>/<site>_windows.json (pool indices "
              "and their draw_sha1 per window) as extra subsets and writes <out>/<site>_nidem.csv with "
              "NIDEM_COLUMNS; then re-run 'plot'. W5-W10 at Villaviciosa lie outside that pool (2023-2025).")

log = v14.log
_T0 = time.time()


# ─────────────────────────────────────────────────────────────────────────────
#  windows
# ─────────────────────────────────────────────────────────────────────────────

def parse_years(text):
    """'2023,2024,2025', '2016-2025' or a mix -> sorted unique years."""
    out = set()
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if b < a:
                raise SystemExit(f"--years: empty range {part!r}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_ints(text):
    return sorted({int(v) for v in str(text or "").split(",") if v.strip()})


def window_specs(record_years, nested=NESTED_YEARS, years=SINGLE_YEARS, end_year=END_YEAR):
    """The windows a record supports: nested windows of k years ending on
    <end_year>-12-31 (``W<k>``) and single calendar years (``Y<year>``).
    A window is kept only where the record has scenes in every one of its
    years; the others are returned in ``skipped`` with the reason."""
    have = {int(y) for y in record_years}
    keep, skipped = [], []

    def add(spec, span):
        miss = [int(y) for y in span if int(y) not in have]
        if miss:
            skipped.append({**spec, "reason": f"the record has no scenes in {miss}"})
        else:
            keep.append(spec)

    for k in sorted({int(v) for v in nested}):
        if k < 1:
            raise ValueError("window lengths are whole years >= 1")
        y0 = int(end_year) - k + 1
        add(dict(label=f"W{k}", kind="nested", n_years=k, first_year=y0, last_year=int(end_year),
                 start=f"{y0}-01-01", end=f"{int(end_year)}-12-31"), range(y0, int(end_year) + 1))
    for y in sorted({int(v) for v in years}):
        add(dict(label=f"Y{y}", kind="year", n_years=1, first_year=y, last_year=y,
                 start=f"{y}-01-01", end=f"{y}-12-31"), [y])
    return keep, skipped


def window_rows(dates, start, end):
    """Sorted indices of the scenes whose date (YYYY-MM-DD) is in [start, end]."""
    d = np.asarray(dates).astype(str)
    return np.flatnonzero((d >= str(start)) & (d <= str(end))).astype(np.int64)


def check_windows(specs, rows_by):
    """Nested windows grow by inclusion; single years are disjoint and, where
    all of a window's years are single years too, they partition it."""
    nested = sorted((s for s in specs if s["kind"] == "nested"), key=lambda s: s["n_years"])
    years = [s for s in specs if s["kind"] == "year"]
    out = {"nested_by_inclusion": True, "years_disjoint": True, "years_partition": {}}
    for a, b in zip(nested, nested[1:]):
        if not set(rows_by[a["label"]].tolist()) <= set(rows_by[b["label"]].tolist()):
            out["nested_by_inclusion"] = False
    for i, a in enumerate(years):
        for b in years[i + 1:]:
            if set(rows_by[a["label"]].tolist()) & set(rows_by[b["label"]].tolist()):
                out["years_disjoint"] = False
    ylab = {s["first_year"]: s["label"] for s in years}
    for s in nested:
        span = range(s["first_year"], s["last_year"] + 1)
        if all(y in ylab for y in span):
            union = np.sort(np.concatenate([rows_by[ylab[y]] for y in span]))
            out["years_partition"][s["label"]] = bool(np.array_equal(union, rows_by[s["label"]]))
    out["empty"] = [s["label"] for s in specs if not len(rows_by[s["label"]])]
    out["ok"] = bool(out["nested_by_inclusion"] and out["years_disjoint"]
                     and all(out["years_partition"].values()) and not out["empty"])
    return out


def job_order(specs):
    """W3 first (the gate), then the other nested windows by length, then
    the single years by date."""
    gate = [s for s in specs if s["label"] == GATE_WINDOW]
    nested = sorted((s for s in specs if s["kind"] == "nested" and s["label"] != GATE_WINDOW),
                    key=lambda s: s["n_years"])
    years = sorted((s for s in specs if s["kind"] == "year"), key=lambda s: s["first_year"])
    return gate + nested + years


def site_stem(site, lags="refit"):
    """The output stem of a site: <site> (refit lags) or <site>_<lags>."""
    return site if lags == "refit" else f"{site}_{lags}"


def per_year(dates):
    out = {}
    for d in np.asarray(dates).astype(str):
        out[d[:4]] = out.get(d[:4], 0) + 1
    return dict(sorted(out.items()))


# ─────────────────────────────────────────────────────────────────────────────
#  the extended record (Villaviciosa, 2016-2025)
# ─────────────────────────────────────────────────────────────────────────────

def load_overpass(path):
    """{date: datetime} exactly as v7.build_record reads it."""
    import pandas as pd

    with open(path, encoding="utf-8") as f:
        return {k: pd.Timestamp(v).to_pydatetime() for k, v in json.load(f).items()}


def extended_record(ext_dates, overpass, boundary, start, end):
    """v7.build_record's Villaviciosa rule on [start, end]: every extraction
    scene in the window with an overpass time and a finite level of
    ``boundary`` at that time (one call, chronological instants). Returns
    the record (time_index, dates, t_real, h, ...) and, per year, the
    scenes, those with an overpass time and those with a finite level."""
    import pandas as pd

    d = np.asarray(ext_dates).astype(str)
    in_win = (d >= str(start)) & (d <= str(end))
    have = np.array([x in overpass for x in d], bool)
    idx = np.flatnonzero(in_win & have)
    if not len(idx):
        raise RuntimeError(f"no extraction scene with an overpass time in [{start}, {end}]")
    t_real = pd.DatetimeIndex(pd.to_datetime([overpass[x] for x in d[idx]])).tz_localize(None)
    if not t_real.is_monotonic_increasing:
        raise RuntimeError("overpass times are not chronological")
    h = np.asarray(boundary.levels(t_real), float)
    if h.shape != (len(idx),):
        raise RuntimeError(f"boundary returned {h.shape} levels for {len(idx)} instants")
    ok = np.isfinite(h)
    yr = np.array([int(x[:4]) for x in d], int)
    yr_idx = yr[idx]
    cov = {}
    for y in range(int(str(start)[:4]), int(str(end)[:4]) + 1):
        m = in_win & (yr == y)
        hf = h[(yr_idx == y) & ok]
        cov[str(y)] = {"scenes": int(m.sum()), "with_overpass_time": int((m & have).sum()),
                       "finite_level": int(((yr_idx == y) & ok).sum()),
                       "level_min_m": float(hf.min()) if len(hf) else None,
                       "level_max_m": float(hf.max()) if len(hf) else None,
                       "level_sd_m": float(hf.std()) if len(hf) else None}
    return dict(time_index=idx[ok], dates=d[idx[ok]], t_real=t_real[ok], h=h[ok],
                boundary=getattr(boundary, "name", type(boundary).__name__), boundary_obj=boundary,
                n_without_time=int((in_win & ~have).sum()), n_nonfinite_level=int((~ok).sum()),
                coverage=cov, start=str(start), end=str(end))


def compare_records(base, ext, start, end):
    """The part of the extended record in [start, end] equals the base
    record (v7.build_record's): same scenes and rows, the same overpass
    instants, the same levels."""
    import pandas as pd

    pos = window_rows(ext["dates"], start, end)
    bd = np.asarray(base["dates"]).astype(str)
    same_dates = bool(len(pos) == len(bd) and np.array_equal(np.asarray(ext["dates"])[pos], bd))
    out = {"scenes_base": int(len(bd)), "scenes_ext_in_span": int(len(pos)), "same_dates": same_dates,
           "same_rows": False, "same_times": False, "max_abs_level_diff_m": None}
    if same_dates:
        out["same_rows"] = bool(np.array_equal(np.asarray(ext["time_index"])[pos], np.asarray(base["time_index"])))
        out["same_times"] = bool(np.array_equal(pd.DatetimeIndex(ext["t_real"])[pos].values,
                                                pd.DatetimeIndex(base["t_real"]).values))
        out["max_abs_level_diff_m"] = float(np.max(np.abs(np.asarray(ext["h"])[pos] - np.asarray(base["h"]))))
    out["ok"] = bool(same_dates and out["same_rows"] and out["same_times"]
                     and out["max_abs_level_diff_m"] is not None and out["max_abs_level_diff_m"] <= v14.TOL_LEVEL)
    return out, pos


def radiometry_by_year(dates, Y, C, ref_start, ref_end, flag=NDWI_SHIFT_FLAG):
    """A per-year proxy of the radiometric convention of the record, read
    from the extraction itself (``Y`` raw NDWI with NaN, ``C`` clear flags,
    rows = ``dates``): the share of clear observations and the 5th / 50th /
    95th percentiles of the NDWI of the clear observations. An L2A +1000 DN
    offset present in some years and not in others compresses |NDWI| by far
    more than ``flag`` (water 0.5 -> about 0.15); a year whose 5th or 95th
    percentile moves by more than ``flag`` from the median of the years in
    [ref_start, ref_end] is flagged. The satellites are not in the
    extraction."""
    d = np.asarray(dates).astype(str)
    yr = np.array([int(x[:4]) for x in d], int)
    Y = np.asarray(Y, float)
    C = np.asarray(C, bool)
    per = {}
    for y in sorted(set(yr.tolist())):
        q = yr == y
        Cy, Yy = C[q], Y[q]
        v = Yy[Cy & np.isfinite(Yy)]
        per[str(y)] = {"scenes": int(q.sum()), "clear_share": float(Cy.mean()) if Cy.size else None,
                       "n_obs": int(v.size),
                       **({f"ndwi_p{k:02d}": float(np.percentile(v, k)) for k in (5, 50, 95)} if v.size else
                          {"ndwi_p05": None, "ndwi_p50": None, "ndwi_p95": None})}
    ref = [y for y in per if str(ref_start)[:4] <= y <= str(ref_end)[:4] and per[y]["ndwi_p95"] is not None]
    out = {"sample": "the lag-fit pixels of production (<= 1200 per band), every clear observation",
           "rule": (f"flag a year whose NDWI p05 or p95 differs by more than {flag} from the median over "
                    f"{str(ref_start)[:4]}-{str(ref_end)[:4]}"),
           "per_year": per, "flagged_years": [], "satellites": "not recorded in the extraction (2016: S2A only)"}
    if ref:
        r05 = float(np.median([per[y]["ndwi_p05"] for y in ref]))
        r95 = float(np.median([per[y]["ndwi_p95"] for y in ref]))
        out["reference"] = {"years": ref, "ndwi_p05": r05, "ndwi_p95": r95}
        for y, s in per.items():
            if s["ndwi_p95"] is None:
                continue
            s["d_p05"], s["d_p95"] = s["ndwi_p05"] - r05, s["ndwi_p95"] - r95
            s["flag"] = bool(abs(s["d_p05"]) > flag or abs(s["d_p95"]) > flag)
            if s["flag"]:
                out["flagged_years"].append(int(y))
    return out


def lag_fit_columns(product_dir, centers):
    """Production's seeded lag-fit pixel draw over all keep pixels by band,
    as v14.load_site / marea.reconstruct draw it (default_rng(seed + 31),
    <= max_px_banda per band): (columns, band of each column)."""
    import yaml

    with open(os.path.join(ROOT, "configs", "m2.yaml"), encoding="utf-8") as f:
        cfg2 = yaml.safe_load(f)
    maxb = int(cfg2["max_px_banda"]["m2a"])
    with np.load(os.path.join(product_dir, "marea.npz"), allow_pickle=False) as mp:
        band_all = mp["band"].astype(int)
    rng = np.random.default_rng(cfg2["seed"] + 31)
    sel = []
    for k in range(len(centers)):
        c = np.where(band_all == k)[0]
        if len(c) > maxb:
            c = np.sort(rng.choice(c, maxb, replace=False))
        sel.append(c)
    cols = np.concatenate(sel).astype(np.int64)
    band = np.concatenate([np.full(len(c), k) for k, c in enumerate(sel)])
    return cols, band


def compare_extension(base, ext, pos, band=None):
    """The extended table on the base scenes (rows ``pos``) equals the base
    table: NDWI and clear flags of the inverted and lag-fit pixels bitwise,
    and the level table at every lag the base holds."""
    g = {"ndwi": bool(np.array_equal(ext.Y0[pos], base.Y0)),
         "clear": bool(np.array_equal(ext.C[pos], base.C))}
    if base.m2a is not None:
        g["lagfit_ndwi"] = bool(np.array_equal(ext.m2a["Y"][pos], base.m2a["Y"]))
        g["lagfit_clear"] = bool(np.array_equal(ext.m2a["C"][pos], base.m2a["C"]))
        if band is not None:
            g["lagfit_bands"] = bool(np.array_equal(np.asarray(band), np.asarray(base.m2a["band"])))
    lags = sorted(base.levels.L)
    ext.levels.ensure(lags)
    dl = max((float(np.max(np.abs(ext.levels.L[t][pos] - base.levels.L[t]))) for t in lags), default=0.0)
    g["levels_max_abs_diff_m"] = dl
    g["n_lags_compared"] = len(lags)
    g["ok"] = bool(all(v for k, v in g.items() if isinstance(v, bool)) and dl <= v14.TOL_LEVEL)
    return g


def extend_site(site, cfg, sd14, extras, start, end):
    """v14's site data over the extended record [start, end]: the same
    inverted pixels, units, truth, lag-fit pixels and boundary provider,
    more scenes. Returns (SiteData, record, gates, info)."""
    from experiments import v7_waterline_contour as v7

    t0 = time.time()
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        ext_dates = np.asarray(ex["dates"]).astype(str)
    rec14 = extras["rec"]
    overpass = load_overpass(v7.SITES[site]["overpass"])
    rec = extended_record(ext_dates, overpass, rec14["boundary_obj"], start, end)
    rec["boundary"] = rec14["boundary"]                 # the same object: v7.build_record's name for it
    log(f"[{site}] extended record {start}..{end}: {len(rec['dates'])} scenes "
        f"({rec['n_without_time']} without overpass time, {rec['n_nonfinite_level']} without a finite level), "
        f"boundary {rec['boundary']} [{time.time() - t0:.1f} s]")
    gates = {}
    g, pos = compare_records(rec14, rec, *v7.TIME_EXTENT)
    gates["extended_record_vs_build_record"] = g
    # production's lag-fit pixels: the refit's inputs, and in both modes the
    # sample of the per-year radiometric proxy
    cols, band = lag_fit_columns(cfg["product"], np.asarray(extras["res"]["centros_km"], float))
    px = np.asarray(extras["px"], np.int64)
    t1 = time.time()
    Ys = v14.stream_columns(cfg["extract"], "Y", rec["time_index"], [px, cols])
    Cs = v14.stream_columns(cfg["extract"], "C", rec["time_index"], [px, cols])
    proxy = radiometry_by_year(rec["dates"], Ys[1], Cs[1].astype(bool), *v7.TIME_EXTENT)
    Y0 = np.nan_to_num(Ys[0], copy=False, nan=0.0).astype(np.float32, copy=False)
    C0 = Cs[0].astype(bool, copy=False)
    m2a = None
    if sd14.m2a is not None:                       # refit; the fixed mode has no lag fit
        m2a = dict(sd14.m2a)
        m2a["Y"] = np.nan_to_num(Ys[1], copy=False, nan=0.0).astype(np.float32, copy=False)
        m2a["C"] = Cs[1].astype(bool, copy=False)
    del Ys, Cs
    log(f"[{site}] streamed {len(rec['time_index'])} rows x {len(px):,} inverted + {len(cols):,} lag-fit "
        f"columns [{time.time() - t1:.1f} s]")
    if proxy["flagged_years"]:
        log(f"[{site}] WARNING radiometric proxy: NDWI percentiles of {proxy['flagged_years']} move by more than "
            f"{NDWI_SHIFT_FLAG} from the 2023-2025 median (an L2A offset state change?); see the JSON")
    levels = v14.LevelTable(rec["boundary_obj"], rec["t_real"])
    sd = v14.SiteData(site, Y0, C0, levels, {"all": np.arange(len(rec["time_index"]))}, sd14.unit_col,
                      sd14.truth, sd14.kind, sd14.tau_fixed, s_km=sd14.s_km, m2a=m2a)
    if g["ok"]:
        gates["extended_arrays_vs_v14"] = compare_extension(sd14, sd, pos, band)
    else:
        levels.ensure(sorted(sd14.levels.L))
        gates["extended_arrays_vs_v14"] = {"ok": False, "reason": "the extended record differs from "
                                           "build_record's on 2023-2025; arrays not compared"}
    cov = rec["coverage"]
    lost = {y: c["scenes"] - c["finite_level"] for y, c in cov.items() if c["scenes"] - c["finite_level"]}
    info = {"extended": True, "span": [start, end], "scenes": int(len(rec["dates"])),
            "rule": ("v7.build_record's Villaviciosa rule on the extended span: every extraction scene with an "
                     "overpass time (products_marea/overpass_times.json) and a finite boundary level"),
            "boundary": rec["boundary"],
            "provider": ("the production provider: the PyTMDBoundary object v7.build_record builds (EOT20 at "
                         "aoi.centroid, tide_models/), evaluated at every overpass time of the span"),
            "n_without_time": rec["n_without_time"], "n_nonfinite_level": rec["n_nonfinite_level"],
            "coverage_per_year": cov, "radiometry_proxy": proxy,
            "covers_every_year": bool(all(c["finite_level"] > 0 for c in cov.values())),
            "declaration": ("the provider gives a finite level at every scene with an overpass time in "
                            "every year of the span" if not lost and rec["n_without_time"] == 0 else
                            f"scenes lost per year (no overpass time or no finite level): {lost}; "
                            f"{rec['n_without_time']} without overpass time"),
            "seconds": round(time.time() - t0, 1)}
    return sd, rec, gates, info


# ─────────────────────────────────────────────────────────────────────────────
#  one window: v14's refit job on exactly its scenes
# ─────────────────────────────────────────────────────────────────────────────

def neutral_bounds(coverage, share_f):
    """Coverage on the method-neutral set N0 from the coverage on F, when F
    holds the share ``share_f`` of N0: [cov share_f, cov share_f + 1 -
    share_f] (nothing is known of N0 outside F)."""
    c, s = float(coverage), float(share_f) if share_f is not None else float("nan")
    if not (np.isfinite(c) and np.isfinite(s)):
        return float("nan"), float("nan")
    return c * s, c * s + (1.0 - s)


def marea_window(sd, spec, rows, mbs, seed, lags="refit", share_f=None):
    """v14's job (:func:`v14.run_job`, lags 'refit' or 'fixed') on exactly
    the scenes ``rows`` of the table: the window is passed as a pool of its
    own and run whole (N = the pool, so v14.draw draws nothing). Returns (CSV
    rows, one per rule, and the job's details). ``share_f`` = n_F / n_N0
    for the method-neutral coverage bounds."""
    from pyintertidal import marea

    if lags not in LAGS:
        raise ValueError(f"lags {lags!r} not in {LAGS}")
    rows = np.asarray(rows, np.int64)
    T = sd.Y0.shape[0]
    if not len(rows) or np.any(np.diff(rows) <= 0) or rows[0] < 0 or rows[-1] >= T:
        raise ValueError(f"{spec['label']}: rows must be non-empty, sorted, unique and inside the table")
    key = f"window:{spec['label']}"
    sd.pools[key] = rows
    try:
        v14_rows, info = v14.run_job(sd, key, len(rows), 0, seed, lags, mbs)
    finally:
        sd.pools.pop(key, None)
    if not np.array_equal(info["rows"], rows):
        raise RuntimeError(f"{spec['label']}: v14.run_job ran other scenes")
    if lags == "refit":
        tau = np.asarray(info["tau_bands"], float)
        tau_used = np.where(np.abs(tau) > marea.TAU_DETECT_MIN, tau, 0.0)
        tau_used = np.where(tau_used < 0, 0.0, tau_used)
        tau_used_txt = ";".join(f"{v:.3f}" for v in tau_used)
    else:
        tau_used_txt = ""                       # production's per-pixel lags (tau_px_* columns)
    clear_px = float(np.median(np.count_nonzero(sd.C[rows], axis=0)))
    sha = v14.draw_sha1(rows)
    out = []
    for r in v14_rows:
        lo_n, hi_n = neutral_bounds(r["coverage"], share_f)
        out.append(dict(site=sd.site, window=spec["label"], kind=spec["kind"], n_years=int(spec["n_years"]),
                        start=spec["start"], end=spec["end"], N=int(r["N"]), mb=int(r["mb"]), lags=lags,
                        rmse=r["rmse"], bias=r["bias"], slope=r["slope"], r=r["r"], coverage=r["coverage"],
                        coverage_neutral_lo=lo_n, coverage_neutral_hi=hi_n,
                        n=int(r["n"]), n_set=int(r["n_set"]), median_clear_obs=r["median_clear_obs"],
                        median_clear_obs_px=clear_px, lo_m=r["lo_m"], hi_m=r["hi_m"],
                        tau_bands_min=r["tau_bands_min"], tau_used_bands_min=tau_used_txt,
                        tau_px_median_min=r["tau_px_median_min"], tau_px_max_min=r["tau_px_max_min"],
                        inversion_seconds=round(float(info["t_inv"]), 3), seconds=r["seconds"],
                        scenes_sha1=sha, same_scenes_as=""))
    return out, dict(info, v14_rows=v14_rows)


def alias_rows(src, spec):
    """The rows of a window holding the same scenes as an already computed
    one (W1 and Y2025): copied, relabelled, no compute."""
    out = []
    for mb in sorted(src):
        r = dict(src[mb])
        r.update(window=spec["label"], kind=spec["kind"], n_years=int(spec["n_years"]), start=spec["start"],
                 end=spec["end"], same_scenes_as=src[mb]["window"], seconds=0.0)
        if "inversion_seconds" in r:
            r["inversion_seconds"] = 0.0
        out.append(r)
    return out


def typed(d):
    """A CSV row (strings) back to numbers."""
    out = {}
    for k, v in d.items():
        if k in STR_COLS:
            out[k] = v
        elif k in INT_COLS:
            out[k] = int(v)
        else:
            out[k] = float(v) if v not in ("", None) else float("nan")
    return out


def save_units(path, units_keep, truth, unit_col, z_store, units_flat=None, shape=None, kind=None,
               lags=None, pixel_m=None):
    """Every window's estimate on the scored units (float64: a re-score
    reproduces the CSV), with what ``ci`` needs to cluster them (the flat
    pixel index of every unit, the grid shape, the truth kind, the pixel
    size); written atomically."""
    arrays = {"units_keep": np.asarray(units_keep, np.int64), "truth": np.asarray(truth, float),
              "unit_col": np.asarray(unit_col, np.int64)}
    if units_flat is not None:
        arrays["units_flat"] = np.asarray(units_flat, np.int64)
    if shape is not None:
        arrays["shape"] = np.asarray(shape, np.int64)
    if kind is not None:
        arrays["kind"] = np.array(str(kind))
    if lags is not None:
        arrays["lags"] = np.array(str(lags))
    if pixel_m is not None:
        arrays["pixel_m"] = np.array(float(pixel_m))
    for lab, by_mb in sorted(z_store.items()):
        for mb, z in sorted(by_mb.items()):
            arrays[f"z__{lab}__mb{int(mb)}"] = np.asarray(z, float)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_units(path, units_keep):
    """({label: {mb: z}}, note) of an earlier run's units file, reused only
    when it holds the same units."""
    out = {}
    if not os.path.exists(path):
        return out, "no units file yet"
    with np.load(path, allow_pickle=False) as f:
        if "units_keep" not in f.files or not np.array_equal(f["units_keep"], np.asarray(units_keep)):
            return out, f"{path} holds other units: not reused"
        for k in f.files:
            if k.startswith("z__"):
                _, lab, mbk = k.split("__")
                out.setdefault(lab, {})[int(mbk[2:])] = np.asarray(f[k], float)
    return out, "reused"


def windows_handoff(site, specs, rows_by, table, pool_dates, nidem_csv):
    """The scene subsets of the windows in a stable, machine-readable form,
    for NIDEM on the same windows (v15 or any other runner): per window its
    dates, extraction rows and, where every scene is in the 'all' pool of
    v14/v15, its pool indices with their v14.draw_sha1 (the key v15 gives a
    subset)."""
    pool_dates = np.asarray(pool_dates).astype(str)
    pos = {d: i for i, d in enumerate(pool_dates)}
    ws = []
    for s in job_order(specs):
        r = rows_by[s["label"]]
        ds = [str(d) for d in np.asarray(table["dates"])[r]]
        inside = all(d in pos for d in ds)
        pidx = np.array([pos[d] for d in ds], np.int64) if inside else None
        ws.append({"label": s["label"], "kind": s["kind"], "n_years": int(s["n_years"]), "start": s["start"],
                   "end": s["end"], "N": int(len(r)), "scenes_sha1": v14.draw_sha1(r),
                   "pool_index": pidx.tolist() if inside else None,
                   "pool_index_sha1": v14.draw_sha1(pidx) if inside else None,
                   "outside_pool": not inside,
                   "extraction_rows": [int(x) for x in np.asarray(table["time_index"])[r]], "dates": ds})
    return {"site": site,
            "purpose": "the scene subsets of the v16 window curves, for NIDEM on the same windows",
            "pool": {"definition": ("pool 'all' of v14/v15: v7.build_record, every cube scene with an overpass "
                                    "time and a finite boundary level (2023-2025)"),
                     "size": int(len(pool_dates)), "first": str(pool_dates[0]), "last": str(pool_dates[-1]),
                     "dates_sha1": hashlib.sha1("|".join(pool_dates).encode()).hexdigest()[:16]},
            "windows": ws, "nidem_csv": nidem_csv, "nidem_csv_columns": list(NIDEM_COLUMNS),
            "scoring": "v14.score on v14's units (units_keep of <site>_units.npz)"}


# ─────────────────────────────────────────────────────────────────────────────
#  resumable CSV
# ─────────────────────────────────────────────────────────────────────────────

class ResumableCsv:
    """A per-site CSV appended job by job (one write and fsync per job). A
    job (one value of ``key``) counts as done only with a row for every
    value in ``need`` of ``need_field``; a partial last line and the rows of
    a job written only in part are dropped on load, so the job is redone."""

    def __init__(self, path, columns, key="window", need_field=None, need=()):
        self.path, self.columns, self.key = path, tuple(columns), key
        self.need_field = need_field
        self.need = {v14._fmt(v) for v in need} if need_field else set()
        self.rows = {}
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.columns)

    def _complete(self, group):
        if not self.need_field:
            return bool(group)
        return self.need <= {g[self.need_field] for g in group}

    def _load(self):
        with open(self.path, "rb") as f:
            data = f.read()
        if not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            with open(self.path, "r+b") as f:
                f.truncate(cut)
            log(f"{self.path}: dropped a partial last line ({len(data) - cut} bytes)")
        with open(self.path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            if tuple(header or ()) != self.columns:
                raise SystemExit(f"{self.path}: unexpected header; refusing to append (use another --out)")
            rows = [dict(zip(self.columns, r)) for r in rd if len(r) == len(self.columns)]
        groups = {}
        for d in rows:
            groups.setdefault(d[self.key], []).append(d)
        partial = sorted(k for k, g in groups.items() if not self._complete(g))
        if partial:
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self.columns)
                for d in rows:
                    if d[self.key] not in partial:
                        w.writerow([d[c] for c in self.columns])
            os.replace(tmp, self.path)
            log(f"{self.path}: dropped the rows of {len(partial)} job(s) written only in part: {partial}")
        self.rows = {k: g for k, g in groups.items() if k not in partial}

    def is_done(self, key):
        return str(key) in self.rows

    def get(self, key):
        return list(self.rows.get(str(key), []))

    def append(self, rows):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([v14._fmt(r[c]) for c in self.columns])
            f.flush()
            os.fsync(f.fileno())
        for r in rows:
            self.rows.setdefault(str(r[self.key]), []).append({c: v14._fmt(r[c]) for c in self.columns})


# ─────────────────────────────────────────────────────────────────────────────
#  the gates against v14
# ─────────────────────────────────────────────────────────────────────────────

def find_v14_reference(site, path=None, lags="refit"):
    """v14's 'all'-pool full-record rows {mb: row} of the lag mode ``lags``
    and the compat block of its settings JSON, from ``path`` or the first of
    V14_REF_DIRS that has <site>_<lags>.csv; None when there is none."""
    cands = [path] if path else [os.path.join(d, f"{site}_{lags}.csv") for d in V14_REF_DIRS]
    for p in cands:
        if not p or not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        full = [r for r in rows if r.get("pool") == "all" and r.get("lags", "refit") == lags
                and int(r["N"]) == int(r["pool_size"]) and int(r["replicate"]) == 0]
        js = p[:-4] + "_settings.json"
        compat = None
        if os.path.exists(js):
            with open(js, encoding="utf-8") as f:
                compat = json.load(f).get("compat")
        return {"path": p, "rows": {int(r["mb"]): r for r in full}, "compat": compat,
                "settings_path": js if os.path.exists(js) else None}
    return None


def _same(a, b):
    a, b = float(a), float(b)
    return (a == b) or (math.isnan(a) and math.isnan(b))


def w3_gate(mine, ref, mbs, tol=TOL_W3):
    """W3 against v14's 'all'-pool full-record point: N, n, coverage, clear
    observations, level range and band lags exactly; RMSE, bias, slope and
    r to ``tol``; bitwise equality reported."""
    if ref is None:
        return {"checked": False, "ok": False,
                "reason": ("no v14 CSV of this lag mode (" + ", ".join(f"{d}/<site>_<lags>.csv" for d in V14_REF_DIRS)
                           + "): pass --v14-csv")}
    per, ok, bit = {}, True, True
    for mb in mbs:
        r, m = ref["rows"].get(int(mb)), mine.get(int(mb))
        if r is None or m is None:
            per[str(mb)] = {"missing": "v14 row" if r is None else "v16 row"}
            ok = bit = False
            continue
        ex = {}
        for k in W3_EXACT:
            eq = (str(m[k]) == str(r[k])) if k == "tau_bands_min" else _same(m[k], r[k])
            ex[k] = {"v16": m[k], "v14": r[k], "equal": bool(eq)}
        tl = {}
        for k in W3_TOL:
            a, b = float(m[k]), float(r[k])
            both_nan = math.isnan(a) and math.isnan(b)
            d = 0.0 if both_nan else (abs(a - b) if not (math.isnan(a) or math.isnan(b)) else math.inf)
            tl[k] = {"v16": a, "v14": b, "abs_diff": d, "bitwise": bool(both_nan or a == b)}
        ok_mb = all(e["equal"] for e in ex.values()) and all(t["abs_diff"] <= tol for t in tl.values())
        bit_mb = ok_mb and all(t["bitwise"] for t in tl.values())
        per[str(mb)] = {"exact": ex, "within_tolerance": tl, "ok": bool(ok_mb), "bitwise": bool(bit_mb)}
        ok &= ok_mb
        bit &= bit_mb
    diffs = [t["abs_diff"] for p in per.values() for t in p.get("within_tolerance", {}).values()]
    return {"checked": True, "reference": ref["path"], "tolerance": tol, "per_mb": per,
            "max_abs_diff": max(diffs) if diffs else None, "bitwise": bool(bit), "ok": bool(ok)}


def fixed_set_gate(mine, ref):
    """F and the production inputs are v14's: the compat block of v14's
    refit settings JSON."""
    if ref is None or not ref.get("compat"):
        return {"checked": False, "ok": True, "stop_if_failed": False,
                "reason": "no v14 refit settings JSON next to the CSV (the W3 gate still compares the point)"}
    rc = ref["compat"]
    diff = {k: {"v16": mine.get(k), "v14": rc.get(k)} for k in COMPAT_KEYS if mine.get(k) != rc.get(k)}
    return {"checked": True, "reference": ref["settings_path"], "keys": list(COMPAT_KEYS), "differing": diff,
            "ok": not diff}


# ─────────────────────────────────────────────────────────────────────────────
#  the window loop
# ─────────────────────────────────────────────────────────────────────────────

def peak_rss_mb():
    """Peak resident memory of this process (MB), or None."""
    try:
        import resource  # POSIX
        r = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return r / 2 ** 20 if sys.platform == "darwin" else r / 1024.0
    except ImportError:
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 20
        except Exception:  # noqa: BLE001
            return None


def run_windows(sd, specs, rows_by, csvlog, mbs, seed, deadline, gate=None, on_job=None, z_store=None,
                lags="refit", share_f=None, on_units=None):
    """Run the windows in job order (W3 first): resume what ``csvlog``
    holds, copy a window whose scenes were already computed, stop before a
    window predicted (W3's seconds scaled by N) to pass ``deadline``.
    ``gate(label, rows)`` is called after W3; False stops the run.
    ``z_store`` ({label: {mb: z per scored unit}}) receives the estimates
    of every window computed or copied now; ``on_units()`` is called after
    each one enters ``z_store`` and BEFORE its CSV rows are written, so a
    CSV row always has its saved estimate. Returns ({label: {mb: row}},
    info)."""
    results, by_sha, status = {}, {}, {}
    stopped, w3_s = None, None
    mb_hi = max(mbs)
    n_gate = len(rows_by.get(GATE_WINDOW, ())) or None
    for spec in job_order(specs):
        lab, rows = spec["label"], rows_by[spec["label"]]
        sha = v14.draw_sha1(rows)
        if csvlog.is_done(lab):
            got = {r["mb"]: r for r in (typed(d) for d in csvlog.get(lab))}
            if any(r["scenes_sha1"] != sha for r in got.values()):
                raise SystemExit(f"{csvlog.path}: window {lab} was computed on other scenes; use another --out")
            results[lab] = got
            by_sha.setdefault(sha, got[mb_hi]["same_scenes_as"] or lab)
            status[lab] = "resumed"
        elif sha in by_sha:
            out = alias_rows(results[by_sha[sha]], spec)
            if z_store is not None and by_sha[sha] in z_store:
                z_store[lab] = z_store[by_sha[sha]]
                if on_units is not None:
                    on_units()
            csvlog.append(out)
            results[lab] = {r["mb"]: r for r in out}
            status[lab] = f"same scenes as {by_sha[sha]}"
            log(f"[{sd.site}] {lab:5s} N={len(rows):5d}: the scenes of {by_sha[sha]} (copied, no compute)")
        else:
            pred = COST_SAFETY * w3_s * len(rows) / n_gate if (w3_s and n_gate) else 0.0
            if time.time() + pred > deadline:
                stopped = {"before": lab, "predicted_seconds": round(pred, 1),
                           "reason": "--max-minutes would be passed (resumable)"}
                log(f"[{sd.site}] STOP before {lab}: predicted {pred:.0f} s would pass --max-minutes")
                break
            out, info = marea_window(sd, spec, rows, mbs, seed, lags=lags, share_f=share_f)
            if z_store is not None:
                z_store[lab] = {int(mb): np.asarray(info["z"][mb], float)[sd.unit_col] for mb in mbs}
                if on_units is not None:
                    on_units()
            csvlog.append(out)
            results[lab] = {r["mb"]: r for r in out}
            by_sha[sha] = lab
            status[lab] = "computed"
            r = results[lab]
            mem = peak_rss_mb()
            log(f"[{sd.site}] {lab:5s} N={len(rows):5d} ({spec['start']}..{spec['end']}): mb{mb_hi} RMSE "
                f"{r[mb_hi]['rmse']:.4f} n {r[mb_hi]['n']:,} cov {r[mb_hi]['coverage']:.3f} | mb0 RMSE "
                f"{r[0]['rmse']:.4f} cov {r[0]['coverage']:.3f} | clear obs {r[mb_hi]['median_clear_obs']:.0f} "
                f"| tau {r[mb_hi]['tau_bands_min']} | {r[mb_hi]['seconds']:.1f} s"
                + (f" | peak {mem / 1024:.2f} GB" if mem else ""))
        if lab == GATE_WINDOW:
            w3_s = max(float(results[lab][mb_hi]["seconds"]), 0.1)
            if gate is not None and not gate(lab, results[lab]):
                stopped = {"after": lab, "reason": "gate failed"}
                break
        if on_job is not None:
            on_job(lab, status[lab])
    return results, {"status": status, "stopped": stopped}


# ─────────────────────────────────────────────────────────────────────────────
#  NIDEM hook (experiments/v15_nidem_scene_curves.py, another workflow)
# ─────────────────────────────────────────────────────────────────────────────

def find_nidem_hook(name=None, path=None, module=None):
    """(function, info) of the v15 NIDEM-on-a-scene-subset function, or
    (None, info) with the TODO when v15 does not exist or exposes none."""
    path = NIDEM_HOOK_FILE if path is None else path
    module = NIDEM_HOOK_MODULE if module is None else module
    names = (name,) if name else NIDEM_HOOK_NAMES
    if not os.path.exists(path):
        return None, {"status": "todo", "reason": f"{os.path.relpath(path, ROOT)} does not exist yet",
                      "todo": NIDEM_TODO}
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        return None, {"status": "error", "reason": f"import of {module} failed: {e!r}", "todo": NIDEM_TODO}
    for nm in names:
        fn = getattr(mod, nm, None)
        if callable(fn):
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "?"
            return fn, {"status": "found", "function": f"{module}.{nm}", "signature": sig}
    return None, {"status": "todo", "reason": f"{module} exposes none of {list(names)}", "todo": NIDEM_TODO}


def call_hook(fn, avail):
    """Call ``fn`` with the keywords of ``avail`` its signature accepts (all
    of them if it takes **kwargs); refuse when it needs one v16 lacks."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kw = dict(avail)
    else:
        kw = {k: v for k, v in avail.items() if k in params}
    need = [p.name for p in params.values() if p.default is p.empty and p.name not in kw
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)]
    if need:
        raise TypeError(f"the NIDEM hook needs {need}; v16 provides {sorted(avail)}")
    return fn(**kw)


def hook_values(res, n_units, unit_col, units_keep, n_px, n_keep):
    """NIDEM per scored unit from what the hook returned: an array per unit,
    per inverted pixel or per keep pixel, or a dict holding one."""
    kind = None
    if isinstance(res, dict):
        for key, k in (("z_units", "units"), ("z_px", "px"), ("z_keep", "keep"), ("filtered_keep", "keep")):
            if key in res:
                arr, kind = np.asarray(res[key], float), k
                break
        else:
            raise ValueError(f"the hook returned a dict without z_units / z_px / z_keep: {sorted(res)}")
    else:
        arr = np.asarray(res, float)
    if kind is None:
        kind = ("units" if arr.shape == (n_units,) else "px" if arr.shape == (n_px,)
                else "keep" if arr.shape == (n_keep,) else None)
    want = {"units": n_units, "px": n_px, "keep": n_keep}
    if kind is None or arr.shape != (want[kind],):
        raise ValueError(f"the hook returned shape {arr.shape}; expected ({n_units},) per unit, "
                         f"({n_px},) per inverted pixel or ({n_keep},) per keep pixel")
    if kind == "units":
        return arr
    if kind == "px":
        return arr[np.asarray(unit_col)]
    return arr[np.asarray(units_keep)]


def _not_computable_reason(res):
    """The reason when the hook says NOT COMPUTABLE by its return value."""
    if isinstance(res, dict) and str(res.get("status", "")).upper().startswith(NOT_COMPUTABLE):
        return str(res.get("reason") or res.get("status"))
    return None


def nidem_rows(sd, spec, rows, z, z_marea, mbs, hook, seconds, share_f=None, status="ok"):
    """NIDEM's CSV rows of one window, one per MAREA rule mb: NIDEM's own
    scores on F (v14.score), then NIDEM and MAREA (mb) on their common units
    (both finite). ``z`` None = NOT COMPUTABLE (n 0, coverage 0)."""
    nan = float("nan")
    base = dict(site=sd.site, window=spec["label"], kind=spec["kind"], n_years=int(spec["n_years"]),
                start=spec["start"], end=spec["end"], N=int(len(rows)), product="NIDEM", status=status,
                n_set=int(len(sd.unit_col)), seconds=round(float(seconds), 2), scenes_sha1=v14.draw_sha1(rows),
                same_scenes_as="", hook=hook)
    out = []
    if z is None:
        for mb in mbs:
            lo_n, hi_n = neutral_bounds(0.0, share_f)
            out.append(dict(base, mb=int(mb), rmse=nan, bias=nan, slope=nan, r=nan, coverage=0.0,
                            coverage_neutral_lo=lo_n, coverage_neutral_hi=hi_n, n=0, n_common=0,
                            rmse_common=nan, bias_common=nan, slope_common=nan, marea_rmse_common=nan,
                            marea_bias_common=nan, marea_slope_common=nan))
        return out
    s = v14.score(sd.kind, sd.truth, z)
    lo_n, hi_n = neutral_bounds(s["coverage"], share_f)
    for mb in mbs:
        zm = None if z_marea is None else z_marea.get(int(mb))
        com = dict(n_common=0, rmse_common=nan, bias_common=nan, slope_common=nan, marea_rmse_common=nan,
                   marea_bias_common=nan, marea_slope_common=nan)
        if zm is not None:
            m = np.isfinite(z) & np.isfinite(zm) & np.isfinite(sd.truth)
            a = v14.score(sd.kind, sd.truth[m], z[m])
            b = v14.score(sd.kind, sd.truth[m], np.asarray(zm, float)[m])
            com.update(n_common=int(m.sum()), rmse_common=a["rmse"], bias_common=a["bias"],
                       slope_common=a["slope"], marea_rmse_common=b["rmse"], marea_bias_common=b["bias"],
                       marea_slope_common=b["slope"])
        out.append(dict(base, mb=int(mb), rmse=s["rmse"], bias=s["bias"], slope=s["slope"], r=s["r"],
                        coverage=s["coverage"], coverage_neutral_lo=lo_n, coverage_neutral_hi=hi_n,
                        n=int(s["n"]), **com))
    return out


def nidem_windows(fn, hook_info, sd, specs, rows_by, table, csvlog, out, deadline, mbs=(0, 3), z_store=None,
                  share_f=None):
    """NIDEM through the hook on the same windows (same dates), scored on
    the same set with v14.score, and with MAREA on their common units (one
    row per MAREA rule). ``table`` = dict(dates, time_index, t_real,
    units_keep, px_keep, n_keep). A window NIDEM cannot compute (an
    EmptyIntervalError of pyintertidal.waterline_contour, or a returned
    {'status': 'NOT COMPUTABLE ...'}) gets NOT COMPUTABLE rows (n 0,
    coverage 0); other errors are recorded, never raised: MAREA's rows
    stand alone. Returns {label: status}."""
    from pyintertidal import waterline_contour as wc

    status, by_sha = {}, {}
    hook = hook_info.get("function", "?")
    for spec in job_order(specs):
        lab, rows = spec["label"], rows_by[spec["label"]]
        sha = v14.draw_sha1(rows)
        if csvlog.is_done(lab):
            status[lab] = "resumed"
            by_sha.setdefault(sha, lab)
            continue
        if sha in by_sha:
            copies = []
            for d in csvlog.get(by_sha[sha]):
                src = typed(d)
                src.update(window=lab, kind=spec["kind"], n_years=int(spec["n_years"]), start=spec["start"],
                           end=spec["end"], same_scenes_as=by_sha[sha], seconds=0.0)
                copies.append(src)
            csvlog.append(copies)
            status[lab] = f"same scenes as {by_sha[sha]}"
            continue
        if time.time() > deadline:
            status[lab] = "not run: --max-minutes"
            continue
        avail = dict(site=sd.site, dates=[str(d) for d in np.asarray(table["dates"])[rows]],
                     rows=np.asarray(table["time_index"], np.int64)[rows],
                     t_real=table["t_real"][rows], h=np.asarray(sd.levels[0], float)[rows],
                     units_keep=np.asarray(table["units_keep"], np.int64),
                     px_keep=np.asarray(table["px_keep"], np.int64),
                     out_dir=os.path.join(out, "nidem_hook", sd.site, lab),
                     # the truth of the units, for the hook's own gate (v15 compares the scores of MAREA's
                     # record with the stored product's on the same units)
                     truth=np.asarray(sd.truth, float), kind=sd.kind,
                     # v16's NIDEM is v15's production NIDEM: ITEM's pixel mask (the hook's default, said
                     # explicitly; v19's common-mask column is another figure)
                     pixel_rule="item")
        t0 = time.time()
        reason = None
        try:
            res = call_hook(fn, avail)
            reason = _not_computable_reason(res)
            z = None if reason else hook_values(res, len(sd.unit_col), sd.unit_col, table["units_keep"],
                                                sd.Y0.shape[1], int(table["n_keep"]))
        except wc.EmptyIntervalError as e:                # NoClearObservationsInIntervalError included
            reason, z = f"{type(e).__name__}: {e}", None
        except (Exception, SystemExit) as e:  # noqa: BLE001  (the hook refuses an input with SystemExit)
            status[lab] = f"error: {e!r}"
            log(f"[{sd.site}] NIDEM {lab}: hook failed: {e!r}")
            continue
        st = "ok" if reason is None else f"{NOT_COMPUTABLE}: {reason}"[:300]
        rows_out = nidem_rows(sd, spec, rows, z, (z_store or {}).get(lab), list(mbs), hook, time.time() - t0,
                              share_f, st)
        csvlog.append(rows_out)
        by_sha[sha] = lab
        status[lab] = "computed" if reason is None else NOT_COMPUTABLE
        r0 = rows_out[-1]
        log(f"[{sd.site}] NIDEM {lab:5s} N={len(rows):5d}: "
            + (f"RMSE {r0['rmse']:.4f} n {r0['n']:,} cov {r0['coverage']:.3f}; on the units common with MAREA mb"
               f"{r0['mb']}: n {r0['n_common']:,} NIDEM {r0['rmse_common']:.4f} vs MAREA {r0['marea_rmse_common']:.4f}"
               if reason is None else f"{NOT_COMPUTABLE} ({reason})") + f" | {r0['seconds']:.1f} s")
    return status


# ─────────────────────────────────────────────────────────────────────────────
#  run
# ─────────────────────────────────────────────────────────────────────────────

def _stop_failed(gates):
    return [k for k, v in gates.items() if isinstance(v, dict) and v.get("ok") is False
            and v.get("stop_if_failed", True)]


def run(args):
    from pyintertidal import marea
    from experiments import v7_waterline_contour as v7

    site, out = args.site, args.out
    lags = getattr(args, "lags", "refit") or "refit"
    v14.guard_out(out)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    stem = os.path.join(out, site_stem(site, lags))
    csv_path, json_path = f"{stem}.csv", f"{stem}_settings.json"
    # NIDEM does not depend on MAREA's lags: one file per site, written by the refit run
    nidem_path = os.path.join(out, f"{site}_nidem.csv")
    units_path, windows_path = f"{stem}_units.npz", f"{stem}_windows.json"
    deadline = _T0 + float(args.max_minutes) * 60.0
    cfg = v14.site_config(site)
    overrides = {}
    for k in ("extract", "product", "comparison", "nidem", "cube"):
        v = getattr(args, f"{k}_override", None)
        if v:
            cfg[k] = v
            overrides[k] = v
    if args.published_csv:
        cfg["published"] = [args.published_csv]
        overrides["published_csv"] = args.published_csv
    mb_primary = int(args.min_bracket)
    mbs = sorted({0, mb_primary})
    seed = int(args.seed)
    max_px = v14.MAX_PX[lags] if args.max_px is None else int(args.max_px)
    nested = sorted(set(parse_ints(args.nested)) | {int(GATE_WINDOW[1:])})
    years = parse_years(args.years)
    soft = bool(args.soft_gates)
    log(f"[{site}] v16 window curves: nested {nested} years ending {END_YEAR}-12-31, single years {years}, "
        f"rules mb {mbs}, lags {lags}, seed {seed}, max px {max_px:,}, max {args.max_minutes} min; "
        f"threads {v14.blas_info()}")

    # 1. v14's site, loaded as v14 loads it for this lag mode: the 'all' pool, F and
    #    its subsample, the lag-fit pixels (refit), the levels
    t0 = time.time()
    sd14, meta, extras = v14.load_site(site, cfg, lags, max_px, seed, ["all"])
    load_s = time.time() - t0
    t0 = time.time()
    gates14, ok14, tim14 = v14.run_gates(site, cfg, sd14, extras, mbs, seed, lags, [])
    sd14.refit_cache.clear()
    refs = v14.reference_points(site, cfg, sd14, extras)
    gates14_s = time.time() - t0
    for k, v in gates14.items():
        log(f"[{site}] v14 gate {k}: {'OK' if v.get('ok', True) else 'FAILED'}")

    # 2. the record of the windows (extended at Villaviciosa when the windows reach before 2023)
    dates14 = np.asarray(extras["dates_all"]).astype(str)
    req, _ = window_specs(range(1900, 2101), nested, years)
    need_start = min(s["start"] for s in req)
    gates = {}
    if need_start < dates14[0] and site in EXTENDABLE:
        sd, rec, ext_gates, rec_info = extend_site(site, cfg, sd14, extras, need_start, f"{END_YEAR}-12-31")
        rec_info["extension_years"] = [int(need_start[:4]), int(dates14[0][:4]) - 1]
        gates.update(ext_gates)
        table = dict(dates=np.asarray(rec["dates"]).astype(str), time_index=np.asarray(rec["time_index"]),
                     t_real=rec["t_real"])
    else:
        sd = sd14
        rec_info = {"extended": False, "scenes": int(len(dates14)),
                    "rule": ("v14's 'all' pool: v7.build_record, every cube scene with an overpass time and a "
                             "finite boundary level"),
                    "boundary": meta.get("boundary"),
                    "note": (f"windows before {dates14[0][:4]} are not covered at this site (the record is "
                             f"{dates14[0][:4]}-{dates14[-1][:4]}; its boundary gauges are cached for "
                             f"{v7.TIME_EXTENT[0][:4]}-{v7.TIME_EXTENT[1][:4]})"
                             if need_start < dates14[0] else "")}
        table = dict(dates=dates14, time_index=np.asarray(extras["rec"]["time_index"]),
                     t_real=extras["rec"]["t_real"])
    table.update(units_keep=np.asarray(extras["units_keep"], np.int64), px_keep=np.asarray(extras["px"], np.int64),
                 n_keep=int(len(extras["keep"])))
    rec_info["per_year"] = per_year(table["dates"])
    rec_info["first"], rec_info["last"] = str(table["dates"][0]), str(table["dates"][-1])
    specs, skipped = window_specs([int(d[:4]) for d in table["dates"]], nested, years)
    rows_by = {s["label"]: window_rows(table["dates"], s["start"], s["end"]) for s in specs}
    gates["windows_nested"] = check_windows(specs, rows_by)
    w3 = rows_by.get(GATE_WINDOW)
    same_pool = w3 is not None and bool(np.array_equal(table["dates"][w3], dates14))
    gates["w3_scenes_are_v14_all_pool"] = {"ok": same_pool, "N_w3": None if w3 is None else int(len(w3)),
                                           "N_v14_all": int(len(dates14))}
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        ext_dates = np.asarray(ex["dates"]).astype(str)
    windows_meta = []
    for s in specs:
        r = rows_by[s["label"]]
        windows_meta.append({**s, "N": int(len(r)), "first_date": str(table["dates"][r[0]]) if len(r) else None,
                             "last_date": str(table["dates"][r[-1]]) if len(r) else None,
                             "extraction_scenes": int(len(window_rows(ext_dates, s["start"], s["end"]))),
                             "scenes_per_year": per_year(table["dates"][r]),
                             "scenes_sha1": v14.draw_sha1(r)})

    # 3. compat, resume, settings
    mine = {"seed": seed, "n_units_scored": int(len(sd14.unit_col)), "n_units_F": int(len(extras["unit_keep_full"])),
            "units_sha1": v14.array_sha1(extras["units_keep"]), "z_stored_sha1": v14.array_sha1(extras["z_stored"]),
            "tau_fixed_sha1": v14.array_sha1(sd14.tau_fixed)}
    compat = {"version": 2, "site": site, "lags": lags, "mbs": mbs, "max_px": max_px, **mine,
              "record_scenes": int(len(table["dates"])),
              "record_sha1": v14.array_sha1(np.asarray(table["time_index"], np.int64))}
    # the method-neutral set N0 (every keep pixel with truth after the range filter /
    # every RTK point in keep): F, chosen by the published MAREA, holds share_f of it
    fs = meta.get("fixed_set") or {}
    n_F = int(len(extras["unit_keep_full"]))
    n_N0 = fs.get("truth_in_range") if sd14.kind == "vaklodingen" else fs.get("points_in_keep")
    share_f = (n_F / float(n_N0)) if n_N0 else None
    neutral = {"n_F": n_F, "n_N0": n_N0, "share_F_of_N0": share_f,
               "N0": ("every keep pixel with Vaklodingen truth after the production range filter "
                      "(v14.dutch_truth: fixed_set.truth_in_range)" if sd14.kind == "vaklodingen" else
                      "every RTK development point inside the extraction (fixed_set.points_in_keep)"),
               "bounds": ("coverage on N0 in [cov_F share_F, cov_F share_F + 1 - share_F]: nothing is known of "
                          "N0 outside F (the method-neutral coverage of both MAREA and NIDEM)")}
    H_, W_ = (int(v) for v in meta["inputs"]["extract"]["grid"])
    keep_all = np.asarray(extras["keep"], np.int64)
    units_flat = keep_all[np.asarray(extras["units_keep"], np.int64)]
    pixel_m = float(fs.get("pixel_m") or 10.0)
    prev = None
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("compat") != v14._clean(compat):
            raise SystemExit(f"{json_path} was written with other settings:\n  {prev.get('compat')}\nnow:\n  "
                             f"{compat}\nrefusing to mix them: use another --out")
    csvlog = ResumableCsv(csv_path, CSV_COLUMNS, key="window", need_field="mb", need=mbs)
    if csvlog.rows:
        log(f"[{site}] resuming: {len(csvlog.rows)} window(s) already in {csv_path}")
    z_store, units_note = load_units(units_path, extras["units_keep"])
    v14.write_json(windows_path, windows_handoff(site, specs, rows_by, table, dates14, nidem_path))
    settings = prev or {}
    settings.update({
        "site": site, "site_name": v14.SITE_NAMES.get(site, site), "status": "running", "compat": compat,
        "git_head": v14.git_head(), "script": v14.file_info(os.path.relpath(os.path.abspath(__file__), ROOT)),
        "v14_script": v14.file_info(os.path.relpath(os.path.abspath(v14.__file__), ROOT)),
        "python": sys.version.split()[0], "numpy": np.__version__, "blas": v14.blas_info(), "seed": seed,
        "question": ("MAREA elevation error against the length of the time window ("
                     + ("band lags refitted on every window" if lags == "refit" else
                        "production lags fixed: the window changes only the scenes of the elevation fit") + ")"),
        "lags": lags, "neutral_set": neutral,
        "rules": {"mb_scored": mbs, "mb_primary": mb_primary,
                  "mb_0": "rule of the stored / published 10 m products (made before the rule existed)",
                  f"mb_{marea.QUALITY_MIN_BRACKET}": "pyintertidal.marea.QUALITY_MIN_BRACKET, the current rule",
                  "how": "one invert_series fit with min_bracket=0, the rule as v14's post-filter"},
        "job": (f"v14.run_job(lags='{lags}') on a pool that is the window, run whole (N = the pool, no draw): "
                + ("m2a_rasch band lags on production's seeded lag-fit pixels, negative bands clipped to 0, "
                   "per-pixel profile smoothed and rounded" if lags == "refit" else
                   "every pixel keeps its production lag (marea.npz tau_px_min)")
                + "; production invert_series; lo/hi = the window's zero-lag level range"),
        "windows_rule": ("every extraction scene in the window with an overpass time and a finite boundary "
                         "level (v14's 'all' pool restricted to the window); per-pixel clear mask SCL 4-7"),
        "windows": windows_meta, "windows_not_covered": skipped, "record": rec_info,
        "files": {"csv": csv_path, "units": units_path, "windows": windows_path, "nidem_csv": nidem_path,
                  "units_note": units_note,
                  "units_content": ("z__<window>__mb<k>: every window's estimate on the scored units "
                                    "(float64), with units_keep, units_flat, shape, truth, unit_col, kind; "
                                    "saved before the window's CSV rows")},
        "max_minutes": float(args.max_minutes), "max_px": max_px, "input_overrides": overrides,
        "v14_gates": gates14, "gates": gates, "reference_points_v14": refs,
        "caveat": ("truth is one survey per site: a longer window also averages real bed change over its "
                   "years (most for W5-W10 at Villaviciosa)"),
        **{k: meta[k] for k in ("fixed_set", "lag_fit", "lags_production", "boundary", "inputs", "notes")
           if k in meta},
    })
    settings.setdefault("runs", [])
    this_run = {"started": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:]}
    settings["runs"].append(this_run)
    settings["timing"] = {"load_seconds": round(load_s, 1), "v14_gates_seconds": round(gates14_s, 1),
                          "extension_seconds": rec_info.get("seconds"), **tim14}

    # 4. the gates before any window
    ref = find_v14_reference(site, args.v14_csv, lags)
    gates["fixed_set_is_v14s"] = fixed_set_gate(mine, ref)
    failed = ([f"v14:{k}" for k in _stop_failed(gates14)] if not ok14 else []) + _stop_failed(gates)
    for k, v in gates.items():
        log(f"[{site}] gate {k}: {'OK' if v.get('ok', True) else 'FAILED'}")

    def finish(status):
        settings["status"] = status
        settings["timing"]["level_calls"] = sd.levels.calls
        if sd is not sd14:
            settings["timing"]["level_calls_v14_table"] = sd14.levels.calls
        settings["timing"]["peak_rss_mb"] = peak_rss_mb()
        settings["timing"]["elapsed_minutes"] = round((time.time() - _T0) / 60, 2)
        this_run["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        this_run["status"] = status
        v14.write_json(json_path, settings)

    if failed and not soft:
        finish("gate_failed")
        raise SystemExit(f"[{site}] gate(s) {failed} failed; see {json_path}")
    if failed:
        settings["gates_overridden"] = failed
        log(f"[{site}] WARNING: gate(s) {failed} failed and --soft-gates is set: continuing; the JSON says so")
    v14.write_json(json_path, settings)

    # 5. the windows, W3 first (gated against v14's point)
    def gate(lab, rows):
        g = w3_gate(rows, ref, mbs)
        gates["w3_reproduces_v14_all_point"] = g
        log(f"[{site}] gate w3_reproduces_v14_all_point: {'OK' if g['ok'] else 'FAILED'} "
            f"(max |diff| {g.get('max_abs_diff')}, bitwise {g.get('bitwise')}, reference {g.get('reference')})")
        if g["ok"]:
            return True
        if soft:
            settings.setdefault("gates_overridden", []).append("w3_reproduces_v14_all_point")
            log(f"[{site}] WARNING: the W3 gate failed and --soft-gates is set: continuing")
            return True
        return False

    def on_units():
        # before the window's CSV rows: a CSV row always has its saved estimate
        save_units(units_path, extras["units_keep"], sd.truth, sd.unit_col, z_store, units_flat=units_flat,
                   shape=(H_, W_), kind=sd.kind, lags=lags, pixel_m=pixel_m)

    def on_job(lab, st):
        settings["progress"] = {"last": lab, "status": st}
        v14.write_json(json_path, settings)

    results, info = run_windows(sd, specs, rows_by, csvlog, mbs, seed, deadline, gate=gate, on_job=on_job,
                                z_store=z_store, lags=lags, share_f=share_f, on_units=on_units)
    missing_z = sorted(lab for lab in results if lab not in z_store)
    if missing_z:
        settings["files"]["units_missing"] = missing_z
        log(f"[{site}] no saved estimates for the resumed window(s) {missing_z} (units file {units_note})")
    for w in settings["windows"]:
        w["status"] = info["status"].get(w["label"], "not run")
    if info["stopped"] and info["stopped"].get("reason") == "gate failed":
        finish("gate_failed")
        raise SystemExit(f"[{site}] W3 does not reproduce v14's 'all'-pool point; see {json_path}")

    # 6. NIDEM on the same windows, when v15 exposes the function
    if args.nidem == "off":
        settings["nidem"] = {"status": "off", "reason": "--nidem off"}
    elif lags != "refit":
        settings["nidem"] = {"status": "off", "reason": ("the fixed-lag variant: NIDEM does not depend on MAREA's "
                                                         f"lags; the refit run writes {nidem_path}")}
    else:
        fn, hinfo = find_nidem_hook(args.nidem_func)
        settings["nidem"] = hinfo
        if fn is None:
            log(f"[{site}] NIDEM: {hinfo['reason']}; {NIDEM_TODO}")
        else:
            log(f"[{site}] NIDEM through {hinfo['function']}{hinfo['signature']}")
            ncsv = ResumableCsv(nidem_path, NIDEM_COLUMNS, key="window", need_field="mb", need=mbs)
            hinfo["windows"] = nidem_windows(fn, hinfo, sd, specs, rows_by, table, ncsv, out, deadline, mbs=mbs,
                                             z_store=z_store, share_f=share_f)
            hinfo["csv"] = nidem_path

    status = "complete" if all(w["label"] in results for w in windows_meta) else (
        "stopped_by_time" if info["stopped"] else "incomplete")
    settings["progress"] = {"windows_planned": len(specs), "windows_done": len(results),
                            "stopped": info["stopped"]}
    finish(status)
    log(f"[{site}] {status}: {len(results)}/{len(specs)} windows -> {csv_path}, {json_path}")
    for s in job_order(specs):
        r = results.get(s["label"])
        if r:
            a, b = r[max(mbs)], r[0]
            log(f"[{site}]   {s['label']:5s} N={a['N']:5d}  mb{max(mbs)} RMSE {a['rmse']:.4f} cov {a['coverage']:.3f}"
                f" | mb0 RMSE {b['rmse']:.4f} cov {b['coverage']:.3f} | clear obs/px {a['median_clear_obs_px']:.0f}"
                f" | tau {a['tau_bands_min']}")


# ─────────────────────────────────────────────────────────────────────────────
#  plot
# ─────────────────────────────────────────────────────────────────────────────

PAL = v14.PAL
METHOD_STYLE = {"MAREA": dict(color=PAL["deep"], marker="o"), "NIDEM": dict(color=PAL["sand"], marker="^")}
#: x tick labels (the mini-table) and their pad below the axis, in points;
#: v14._style_axes sets the tick length to 3
TICK_FS, TICK_PAD = 10.0, 3.5
SITE_NAMES_ES = {"villaviciosa": "Ría de Villaviciosa", "escalda": "Escalda Occidental",
                 "wadden": "Mar de Frisia (Vlie)", "ems": "Ems-Dollard"}
TEXT = {
    "en": dict(
        title="MAREA: elevation error vs length of the time window ({res} m)",
        leg_marea_nested="MAREA, windows ending 31 Dec 2025 (W1 = 2025, W2 = 2024-25, ...)",
        leg_marea_years="MAREA, single calendar years (hollow; line = their range)",
        leg_nidem_nested="NIDEM, same windows", leg_nidem_years="NIDEM, single years",
        rmse="Elevation RMSE (m)", cov="Coverage (%)",
        rows="years\nscenes\nclear obs./px", xlabel="Window length (years)",
        no_results="no results yet",
        truth_rtk="RTK-GNSS (development split), n = {n} points",
        truth_vak="Vaklodingen, n = {n} px{sub}", sub=" (random subsample of {nF})",
        rule0=" (the rule of the published products)", rule=" (the published products used 0)",
        foot_scenes=("Each window uses every scene in it with an overpass time and a finite boundary level; the "
                     "per-pixel clear-sky mask (SCL 4-7) applies. Band lags re-fitted on each window as production "
                     "fits them; production inversion; quality rule min_bracket = {mb}{rule}."),
        foot_set=("Each site is scored on one fixed set (finite truth and finite published MAREA; Dutch sites: the "
                  "seeded random subsample of the scene-count curves). RMSE median-centred over the part of the "
                  "set with an estimate; coverage = share of the set with an estimate. Under each tick: scenes in "
                  "the window and median clear observations per pixel."),
        foot_gate="W3 (2023-2025) reproduces the all-scenes full-record point of the scene-count curves: {txt}.",
        gate_ok="passed", gate_bad="FAILED", gate_none="not checked",
        foot_ext="{site}{yy}: the production boundary (EOT20 at the site centroid), finite at every scene.",
        foot_ext_lost="{site}{yy}: the production boundary (EOT20 at the site centroid); {decl}.",
        foot_years="Single calendar years drawn: {txt}.",
        foot_nidem_todo=("NIDEM on the same windows: not computed yet (experiments/v15_nidem_scene_curves.py does "
                         "not expose a scene-subset function); MAREA alone."),
        foot_nidem="NIDEM on the same windows through {fn}, scored on the same set.",
        foot_caveat="Truth is one survey per site: a longer window also averages real bed change.",
        end="{v} m"),
    "es": dict(
        title="MAREA: error de cota según la longitud de la ventana temporal ({res} m)",
        leg_marea_nested="MAREA, ventanas que terminan el 31-12-2025 (W1 = 2025, W2 = 2024-25, ...)",
        leg_marea_years="MAREA, años naturales sueltos (huecos; línea = su rango)",
        leg_nidem_nested="NIDEM, mismas ventanas", leg_nidem_years="NIDEM, años sueltos",
        rmse="RMSE de cota (m)", cov="Cobertura (%)",
        rows="años\nescenas\nobs. desp./px", xlabel="Longitud de la ventana (años)",
        no_results="sin resultados todavía",
        truth_rtk="RTK-GNSS (partición de desarrollo), n = {n} puntos",
        truth_vak="Vaklodingen, n = {n} px{sub}", sub=" (submuestra aleatoria de {nF})",
        rule0=" (la regla de los productos publicados)", rule=" (los productos publicados usaron 0)",
        foot_scenes=("Cada ventana usa todas sus escenas con hora de paso y nivel de contorno finito; se aplica la "
                     "máscara de nubes por píxel (SCL 4-7). Desfases por banda reajustados en cada ventana como en "
                     "producción; inversión de producción; regla de calidad min_bracket = {mb}{rule}."),
        foot_set=("Cada sitio se evalúa sobre un conjunto fijo (verdad finita y MAREA publicado finito; sitios "
                  "neerlandeses: la submuestra aleatoria con semilla de las curvas por número de escenas). RMSE "
                  "centrado en la mediana sobre la parte del conjunto con estimación; cobertura = fracción del "
                  "conjunto con estimación. Bajo cada marca: escenas de la ventana y mediana de observaciones "
                  "despejadas por píxel."),
        foot_gate=("W3 (2023-2025) reproduce el punto de registro completo con todas las escenas de las curvas "
                   "por número de escenas: {txt}."),
        gate_ok="superado", gate_bad="NO SUPERADO", gate_none="sin comprobar",
        foot_ext=("{site}{yy}: el contorno de producción (EOT20 en el centroide del sitio), finito en todas "
                  "las escenas."),
        foot_ext_lost="{site}{yy}: el contorno de producción (EOT20 en el centroide del sitio); {decl}.",
        foot_years="Años naturales sueltos dibujados: {txt}.",
        foot_nidem_todo=("NIDEM en las mismas ventanas: aún sin calcular (experiments/v15_nidem_scene_curves.py no "
                         "expone una función por subconjunto de escenas); solo MAREA."),
        foot_nidem="NIDEM en las mismas ventanas mediante {fn}, evaluado sobre el mismo conjunto.",
        foot_caveat=("La verdad es un único levantamiento por sitio: una ventana más larga también promedia "
                     "cambios reales del fondo."),
        end="{v} m"),
}


def site_name(site, lang):
    return (SITE_NAMES_ES if lang == "es" else v14.SITE_NAMES).get(site, site)


def fmt_num(v, nd, lang):
    s = f"{float(v):.{int(nd)}f}"
    return s.replace(".", ",") if lang == "es" else s


def fmt_int(n, lang):
    s = f"{int(round(float(n))):,}"
    return s.replace(",", ".") if lang == "es" else s


def plot_inputs(out, site):
    """The rows, settings and NIDEM rows (if any) of one site."""
    import pandas as pd

    df = pd.read_csv(os.path.join(out, f"{site}.csv"))
    jp = os.path.join(out, f"{site}_settings.json")
    st = {}
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            st = json.load(f)
    npath = os.path.join(out, f"{site}_nidem.csv")
    nd = pd.read_csv(npath) if os.path.exists(npath) else None
    return dict(site=site, df=df, settings=st, nidem=nd)


def _decimals(ax):
    t = np.asarray(ax.get_yticks(), float)
    if len(t) < 2:
        return 2
    step = float(np.min(np.diff(t)))
    return int(max(0, math.ceil(-math.log10(step) - 1e-9))) if step < 1 else 0


def _year_labels(ax, xs, vals, years, lang, color, max_all=3):
    """Year labels beside the single-year markers, merged where two would
    touch (never nudged apart); with more than ``max_all`` years only the
    lowest and the highest are labelled."""
    lo, hi = ax.get_ylim()
    pts = sorted((float(v), str(y)) for v, y in zip(vals, years) if np.isfinite(v))
    if len(pts) > max_all:
        pts = [pts[0], pts[-1]]
    groups = []
    for v, y in pts:
        if groups and (v - groups[-1][-1][0]) < 0.045 * (hi - lo):
            groups[-1].append((v, y))
        else:
            groups.append([(v, y)])
    for g in groups:
        v = float(np.mean([p[0] for p in g]))
        txt = ", ".join(p[1] for p in g)
        ax.annotate(txt, (xs, v), xytext=(-9, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.6, color=color)


def draw_site(ar, ac, D, mb, lang):
    """RMSE (top) and coverage (bottom) against the window length of one
    site: nested windows as a line, single years as hollow markers at one
    year. Returns legend handles."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullLocator

    T = TEXT[lang]
    handles = {}
    series = [("MAREA", D["df"][D["df"]["mb"] == int(mb)])]
    if D.get("nidem") is not None and len(D["nidem"]):
        series.append(("NIDEM", D["nidem"]))
    ticks = None
    covs, yr_marks = [], []
    for method, df in series:
        stl = METHOD_STYLE[method]
        nest = df[df["kind"] == "nested"].sort_values("n_years")
        yrs = df[df["kind"] == "year"].sort_values("start")
        x = nest["n_years"].to_numpy(float)
        y = nest["rmse"].to_numpy(float)
        c = 100 * nest["coverage"].to_numpy(float)
        ar.plot(x, y, color=stl["color"], lw=2.0, marker=stl["marker"], ms=7.5, mec="white", mew=1.4,
                zorder=4, solid_capstyle="round", solid_joinstyle="round")
        ac.plot(x, c, color=stl["color"], lw=2.0, marker=stl["marker"], ms=6.5, mec="white", mew=1.2, zorder=4)
        covs.extend(c[np.isfinite(c)].tolist())
        key = method.lower()
        handles[f"{key}_nested"] = Line2D([], [], color=stl["color"], lw=2.0, marker=stl["marker"], ms=7.5,
                                          mec="white", mew=1.4, label=T[f"leg_{key}_nested"])
        if len(yrs):
            yy = yrs["rmse"].to_numpy(float)
            cc = 100 * yrs["coverage"].to_numpy(float)
            fin = np.isfinite(yy)
            if fin.sum() >= 2:
                ar.vlines(1.0, np.nanmin(yy), np.nanmax(yy), color=stl["color"], lw=1.3, alpha=0.45, zorder=2)
            ar.plot(np.ones(len(yy)), yy, ls="none", marker=stl["marker"], ms=7.5, mfc="white", mec=stl["color"],
                    mew=1.6, zorder=3)
            ac.plot(np.ones(len(cc)), cc, ls="none", marker=stl["marker"], ms=6.5, mfc="white", mec=stl["color"],
                    mew=1.4, zorder=3)
            covs.extend(cc[np.isfinite(cc)].tolist())
            handles[f"{key}_years"] = Line2D([], [], ls="none", marker=stl["marker"], ms=7.5, mfc="white",
                                             mec=stl["color"], mew=1.6, label=T[f"leg_{key}_years"])
            if method == "MAREA":
                yr_marks = (yy, [s[:4] for s in yrs["start"].astype(str)])
        if method == "MAREA":
            ticks = nest
            # the value at both ends of the nested line (selective labels)
            fin = np.flatnonzero(np.isfinite(y))
            for i in ({int(fin[0]), int(fin[-1])} if len(fin) else set()):
                ar.annotate(T["end"].format(v=fmt_num(y[i], 3, lang)), (x[i], y[i]), xytext=(7, 7),
                            textcoords="offset points", ha="left", va="bottom", fontsize=9.2, color=PAL["ink"])
    ar.set_xscale("log")
    xs = [int(k) for k in ticks["n_years"]] if ticks is not None and len(ticks) else [1, 2, 3]
    ar.xaxis.set_major_locator(FixedLocator(xs))
    ar.xaxis.set_minor_locator(NullLocator())
    if ticks is not None and len(ticks):
        labels = [f"{k}\n{fmt_int(n, lang)}\n{fmt_int(c, lang)}" for k, n, c in
                  zip(xs, ticks["N"], ticks["median_clear_obs_px"])]
    else:
        labels = [str(k) for k in xs]
    ac.xaxis.set_major_formatter(FixedFormatter(labels))
    ar.set_xlim(0.66, max(xs) * 1.28)
    plt.setp(ar.get_xticklabels(), visible=False)
    lo, hi = ar.get_ylim()
    ar.set_ylim(lo - 0.08 * (hi - lo), hi + 0.10 * (hi - lo))
    nd = _decimals(ar)
    ar.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_num(v, nd, lang)))
    cmin = min(covs) if covs else 100.0
    ac.set_ylim(min(50.0, 10.0 * math.floor(cmin / 10.0)), 102.0)
    ac.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_num(v, 0, lang)))
    if len(yr_marks):
        _year_labels(ar, 1.0, yr_marks[0], yr_marks[1], lang, PAL["slate"])
    return handles


def _truth_line(D, lang):
    T = TEXT[lang]
    fs = D["settings"].get("fixed_set", {})
    n = fs.get("n_units_scored")
    if n is None:
        return ""
    if fs.get("unit") == "RTK point" or D["site"] == "villaviciosa":
        return T["truth_rtk"].format(n=fmt_int(n, lang))
    sub = T["sub"].format(nF=fmt_int(fs["n_units_F"], lang)) if fs.get("subsample") else ""
    return T["truth_vak"].format(n=fmt_int(n, lang), sub=sub)


def _gate_text(data, lang):
    T = TEXT[lang]
    parts = []
    for site, D in data.items():
        g = (D["settings"].get("gates") or {}).get("w3_reproduces_v14_all_point")
        if not g:
            st = T["gate_none"]
        elif not g.get("ok"):
            st = T["gate_bad"]
        elif g.get("bitwise"):
            st = T["gate_ok"] + (" (bit a bit)" if lang == "es" else " (bitwise)")
        elif g.get("max_abs_diff") is not None:
            st = T["gate_ok"] + f" (|Δ| ≤ {float(g['max_abs_diff']):.0e})"
        else:
            st = T["gate_ok"]
        parts.append(f"{site_name(site, lang)} {st}")
    return "; ".join(parts) if parts else T["gate_none"]


def window_figure(data, mb, lang):
    """One column per site (always the four sites of v14.SITE_ORDER; a site
    without results keeps a labelled empty column)."""
    import matplotlib.pyplot as plt

    T = TEXT[lang]
    cols = list(v14.SITE_ORDER) + [s for s in data if s not in v14.SITE_ORDER]
    widths = [1.5 if s == "villaviciosa" else 1.0 for s in cols]
    fig = plt.figure(figsize=(3.75 * sum(widths) + 1.3, 8.8))
    gs = fig.add_gridspec(2, len(cols), height_ratios=[3.0, 1.25], width_ratios=widths, hspace=0.10,
                          wspace=0.26, left=0.075, right=0.99, top=0.80, bottom=0.31)
    handles = {}
    for j, site in enumerate(cols):
        ar = fig.add_subplot(gs[0, j])
        ac = fig.add_subplot(gs[1, j], sharex=ar)
        for ax in (ar, ac):
            v14._style_axes(ax)
        ar.text(0.0, 1.10, site_name(site, lang), transform=ar.transAxes, fontsize=13.5, color=PAL["deep"],
                weight="bold", va="bottom", ha="left")
        if j == 0:
            ar.set_ylabel(T["rmse"], color=PAL["ink"], fontsize=11.5)
            ac.set_ylabel(T["cov"], color=PAL["ink"], fontsize=11.5)
        D = data.get(site)
        if D is None or D["df"][D["df"]["mb"] == int(mb)].empty:
            ar.text(0.5, 0.5, T["no_results"], transform=ar.transAxes, ha="center", va="center", fontsize=12,
                    color=PAL["slate"])
            for ax in (ar, ac):
                ax.set_xticks([])
                ax.set_yticks([])
            continue
        for k, v in draw_site(ar, ac, D, mb, lang).items():
            handles.setdefault(k, v)
        ar.text(0.0, 1.025, _truth_line(D, lang), transform=ar.transAxes, fontsize=9.5, color=PAL["slate"],
                va="bottom", ha="left")
        ac.tick_params(axis="x", labelsize=TICK_FS, pad=TICK_PAD)
        ac.set_xlabel(T["xlabel"], color=PAL["ink"], fontsize=11)
        if j == 0:
            # row headers of the mini-table under the ticks, on the tick labels' own lines
            ac.annotate(T["rows"], xy=(0, 0), xycoords="axes fraction", xytext=(-12, -(TICK_PAD + 3)),
                        textcoords="offset points", ha="right", va="top", fontsize=TICK_FS,
                        color=PAL["slate"], linespacing=1.2)
    res = sorted({float(D["settings"].get("fixed_set", {}).get("pixel_m") or 10.0) for D in data.values()})
    fig.text(0.075, 0.955, T["title"].format(res="/".join(f"{r:g}" for r in res)), fontsize=17,
             color=PAL["deep"], weight="bold", ha="left")
    order = [k for k in ("marea_nested", "marea_years", "nidem_nested", "nidem_years") if k in handles]
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order], loc="upper left",
               bbox_to_anchor=(0.07, 0.94), ncol=2, frameon=False, fontsize=10.5, handlelength=2.4,
               columnspacing=2.0)
    nid = [D for D in data.values() if D.get("nidem") is not None and len(D["nidem"])]
    hooks = sorted({str(h) for D in nid for h in D["nidem"].get("hook", [])})
    foot = [T["foot_scenes"].format(mb=int(mb), rule=T["rule"] if int(mb) else T["rule0"]),
            T["foot_set"], T["foot_gate"].format(txt=_gate_text(data, lang))]
    yrs_txt = []
    for site, D in data.items():
        ys = sorted({int(str(s)[:4]) for s in D["df"][D["df"]["kind"] == "year"]["start"]})
        if ys:
            span = f"{ys[0]}-{ys[-1]}" if ys == list(range(ys[0], ys[-1] + 1)) and len(ys) > 1 else \
                ", ".join(map(str, ys))
            yrs_txt.append(f"{site_name(site, lang)} {span}")
    if yrs_txt:
        foot.append(T["foot_years"].format(txt="; ".join(yrs_txt)))
    for site, D in data.items():
        rec = D["settings"].get("record") or {}
        if rec.get("extended"):
            y0, y1 = (rec.get("extension_years") or [None, None])[:2]
            clean = not rec.get("n_nonfinite_level") and not rec.get("n_without_time")
            foot.append((T["foot_ext"] if clean else T["foot_ext_lost"]).format(
                site=site_name(site, lang), yy=f" {y0}-{y1}" if y0 is not None else "",
                decl=rec.get("declaration", "")))
    foot.append(T["foot_nidem"].format(fn=", ".join(hooks)) if nid else T["foot_nidem_todo"])
    foot.append(T["foot_caveat"])
    import textwrap
    # ~15 characters per inch at 8.6 pt over the text block's width
    width_chars = int(0.91 * fig.get_figwidth() * 15)
    wrapped = [textwrap.fill(t, width=width_chars, break_on_hyphens=False) for t in foot]
    fig.text(0.075, 0.012, "\n".join(wrapped), fontsize=8.6, color=PAL["slate"], ha="left", va="bottom",
             linespacing=1.35)
    return fig


def write_summary(out, data):
    """All sites, both methods, every window and rule in one CSV."""
    import pandas as pd

    parts = []
    for D in data.values():
        a = D["df"].copy()
        a.insert(0, "method", "MAREA")
        parts.append(a)
        if D.get("nidem") is not None and len(D["nidem"]):
            b = D["nidem"].copy()
            b.insert(0, "method", "NIDEM")
            parts.append(b)
    if not parts:
        return None
    cols = ["method", "site", "window", "kind", "n_years", "start", "end", "N", "mb", "rmse", "bias", "slope", "r",
            "coverage", "n", "n_set", "median_clear_obs", "median_clear_obs_px", "tau_bands_min",
            "tau_used_bands_min", "same_scenes_as"]
    s = pd.concat(parts, ignore_index=True)
    s = s[[c for c in cols if c in s.columns]]
    path = os.path.join(out, "window_curves_summary.csv")
    s.to_csv(path, index=False)
    return path


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    sites = [s for s in (args.sites or v14.SITE_ORDER) if os.path.exists(os.path.join(out, f"{s}.csv"))]
    if not sites:
        raise SystemExit(f"no {out}/<site>.csv to plot")
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.titlepad": 6})
    data = {s: plot_inputs(out, s) for s in sites}
    p = write_summary(out, data)
    if p:
        log(f"-> {p}")
    paths = []
    for mb in parse_ints(args.mb):
        for lang in [x.strip() for x in str(args.lang).split(",") if x.strip()]:
            if lang not in TEXT:
                raise SystemExit(f"--lang: {lang!r} not in {sorted(TEXT)}")
            fig = window_figure(data, mb, lang)
            stem = os.path.join(out, f"window_curves_{lang}_mb{mb}")
            fig.savefig(stem + ".png", dpi=200)
            fig.savefig(stem + ".pdf")
            plt.close(fig)
            paths.append(stem)
            log(f"-> {stem}.png/.pdf")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    from pyintertidal import marea

    ap = argparse.ArgumentParser(prog="python -m experiments.v16_window_curves",
                                 description="MAREA elevation error against the length of the time window.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the windows of one site (resumable)")
    r.add_argument("site", choices=list(v14.SITE_NAMES))
    r.add_argument("--nested", default=",".join(map(str, NESTED_YEARS)),
                   help=f"nested window lengths in years, ending {END_YEAR}-12-31 (default 1,2,3,5,7,10; "
                        "3 is always run: it is the gate; lengths the record does not cover are listed, not run)")
    r.add_argument("--years", default=",".join(map(str, SINGLE_YEARS)),
                   help="single calendar years, e.g. 2023,2024,2025 or 2016-2025 (default 2023-2025)")
    r.add_argument("--seed", type=int, default=v14.DEFAULT_SEED, help="v14's seed (the subsample of F)")
    r.add_argument("--min-bracket", type=int, default=marea.QUALITY_MIN_BRACKET,
                   help="primary quality rule (default QUALITY_MIN_BRACKET = 3); 0 is always scored too")
    r.add_argument("--max-minutes", type=float, default=60.0, help="wall-time cap of this process")
    r.add_argument("--max-px", type=int, default=None,
                   help=f"subsample F above this many units (default v14's refit {v14.MAX_PX['refit']:,}; "
                        "another value breaks the W3 gate)")
    r.add_argument("--v14-csv", default=None,
                   help="v14's <site>_refit.csv for the W3 gate (default: first of "
                        + ", ".join(f"{d}/<site>_refit.csv" for d in V14_REF_DIRS) + ")")
    r.add_argument("--soft-gates", action="store_true",
                   help="a failed gate is logged and recorded instead of stopping (only after reading why)")
    r.add_argument("--nidem", choices=("auto", "off"), default="auto",
                   help="auto: NIDEM on the same windows when experiments/v15_nidem_scene_curves.py exposes it")
    r.add_argument("--nidem-func", default=None, help="name of the v15 function (default: the first of "
                                                      + ", ".join(NIDEM_HOOK_NAMES) + ")")
    r.add_argument("--out", default=OUT)
    r.add_argument("--product-dir", dest="product_override", default=None,
                   help="folder with the production marea.npz/result.json (e.g. a laptop mirror)")
    r.add_argument("--extract", dest="extract_override", default=None)
    r.add_argument("--comparison-dir", dest="comparison_override", default=None)
    r.add_argument("--nidem-dir", dest="nidem_override", default=None)
    r.add_argument("--cube", dest="cube_override", default=None)
    r.add_argument("--published-csv", default=None)
    p = sub.add_parser("plot", help="the figure (Spanish and English) and the summary CSV")
    p.add_argument("--out", default=OUT)
    p.add_argument("--mb", default=f"{marea.QUALITY_MIN_BRACKET},0",
                   help="quality rule(s) to draw (default 3,0)")
    p.add_argument("--lang", default="es,en", help="es,en (default both)")
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
