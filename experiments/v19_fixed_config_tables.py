# -*- coding: utf-8 -*-
"""V19 - elevation methods under ONE fixed external configuration, per site and
number of screened images, every window ending on its site's truth date (the
first author's design, 2026-10-20; truths and window ends, 2026-10-26).

Design
------
One configuration EXTERNAL to the methods, MAREA's production one exactly as
it is, given to every method:

* scene filter: the production transition-zone screening
  (pyintertidal.stability.reference_and_clouds + filter_dates(0.10) with the
  production reference map), then an overpass time and a finite production
  boundary level h_B. This is the screened list S of a site:
  - Villaviciosa: the notebook's 305 usable dates 2016-2025
    (v7.villaviciosa_usable_dates, or --usable-json) that are extraction
    dates with an overpass time and a finite h_B;
  - Dutch sites: result.json record_dates (68 / 71 / 54 scenes 2023-2025),
    extended before 2023 only from an extra extraction store (below);
* per-pixel mask: the extraction's C = SCL in {4, 5, 6, 7};
* index: raw NDWI = (B03 - B08) / (B03 + B08), NaN where undefined;
* wet: W = C & (NDWI > 0);
* pixels: the production extraction's keep pixels;
* levels: the production h_B of the site (Villaviciosa PyTMDBoundary EOT20 at
  the site centroid; Dutch sites the EOT20 + gauges consensus) at the
  overpass label instants. MAREA keeps its lag model (3-km bands, monotone,
  tau(0) = 0, refitted on every window exactly as v14's refit mode); every
  other method reads h_B(t) without lag (common conditions).

Truth and window end (the user's decisions of 2026-10-20 and 2026-10-26)
------------------------------------------------------------------------
Every window ends on the date of its site's truth (the validation survey):

* Dutch sites (Westerschelde, Wadden, Ems): Rijkswaterstaat Vaklodingen, the
  MOST RECENT survey inside the image archive 2023-2025, i.e. 2025: the truth
  is the 20 m cells surveyed in 2025 only, a cell-level selection
  (:func:`truth_grid_year`: v1's grid restricted to one survey year; v1's
  nearest-cell sampling; v14's production range filter). Vaklodingen gives
  the survey YEAR only (time stamps YYYY-01-01), so the window ends on
  2025-12-31, the end of the survey year (declared in the settings JSON and
  under every table). 2025 surveys: 5 of 5 sheets at the Westerschelde, 6 of
  15 at the Wadden, 3 of 12 at the Ems (most of the Ems is 2020).
* Wadden: survey-year groups (the user's choice, 2026-10-26;
  :data:`SURVEY_YEAR_GROUPS`). Per keep pixel, the most recent survey year
  in 2023-2025 of the Vaklodingen cell under it (v1's nearest-cell sampling
  within each year; :func:`dutch_truth_groups`) and that survey's value;
  the pixels are grouped by that year: rows 'wadden-2025', 'wadden-2024',
  'wadden-2023' (with their sheet and cell counts), each with its own truth
  date (the end of its year) and its own windows (the n most recent
  screened images on or before it); a group whose window lacks screened
  images is 'not reachable' with the count. The same F rules inside every
  group (the production range filter, no product decides F; the common set
  of a group row is its pixels where every headline method has a value).
  A pooled row 'wadden' only at the n where EVERY group's window is
  reachable: each pixel scored with its own group's estimate (estimator
  seconds: the sum of the groups'). The production record is the gate
  window of the 2025 group only. Declared in the settings JSON
  ('truth_groups') and under every table.
* Villaviciosa: the PNOA LiDAR, never the RTK. A GeoTIFF (``--vva-truth``;
  the repository holds the second-coverage MDT02,
  data_v4/truth/pnoa/MDT02-WGS84-0015-3-COB2.tif, flown 2015-2017, exact date
  unknown) sampled to the 10 m grid as experiments/benchmark/truth.py's
  pnoa_truth does for PNOA (ground only: its pnoa_ground constant-patch water
  mask, no flight-level mask; the 10 m mean with >= 95 % of the area valid and
  >= 80 % of it ground); range-filtered as the other sites (the production
  no-delay product's range +- 0.25 m: products_marea/external_products.npz
  'no_delay', whose 'MAREA' must equal the stored product). The window ends on
  the flight date (``--vva-truth-date``). Heights are orthometric: the RMSE
  stays median-centred, the bias is reported but carries the datum. Without
  ``--vva-truth`` and ``--vva-truth-date`` the Villaviciosa row is 'pending
  truth'. The archive starts 2016-01-01: a 2015-2017 flight leaves at most 36
  screened scenes, so no n >= 40 is reachable (the tables say so, with the
  number of screened images up to the date).

F = the keep pixels (the production extraction's pixels, which every method
receives) with a truth value inside the range filter, decided by NO product
(review of 2026-10-26: neither MAREA's validity nor any other method's), with
v14's seeded random subsample above ``--max-px``. v14.load_site builds it with
its unit function v14.rtk_units replaced for that one call by this design's
(:func:`dutch_units` at the Dutch sites, :func:`raster_units` at
Villaviciosa; v14 runs its 'rtk' branch, which takes the units as given; v14's
file is not edited and its sha256 gate holds; the RTK survey is never read).
Every unit is a 10 m pixel and every site is scored like a raster truth
(v7.vscore through v14.score: median-centred RMSE, slope, r, bias =
median(product - truth)). MAREA's coverage is measured on F like every other
method's (no longer 100 % by construction).

Windows: Window(site, n) = the n most recent screened scenes on or before the
site's truth date, for n in 125, 100, 75, 60, 50, 40 (``--ns``): a time window
flexible per site, starting where it holds n screened scenes. Nested and
deterministic (one realisation). A cell is 'not reachable' when fewer than n
screened scenes lie on or before the truth date. The full production record
(n = 97 / 68 / 71 / 54, the newest scenes of S) is the gate window: tabulated
when it ends on or before the truth date (the Dutch sites), otherwise
computed for the gate only (MAREA and no delay; never tabulated).

Columns (methods)
-----------------
* E0b MAREA: v14.run_job(lags='refit') on exactly the window's scenes; mb 0
  (the rule of the published products) is the headline, mb 3 (the current
  production rule) a variant from the same fit;
* E0a no delay: v14.invert_subset at tau = 0 (internal reference, not in the
  headline tables);
* every competitor of experiments/fixedcfg/methods.py (METHODS = {id: (label,
  run, card)}, ``run(obs, cfg) -> dict(z, sigma, meta)``), imported lazily; a
  method whose code is not built yet is logged and marked 'pending'. A card
  may set ``pixel_set`` ('keep', the default, or 'scored') or ``per_pixel``
  (True = 'scored'): a per-pixel method then runs on the scored pixels only;
* E1 NIDEM through the v15 hook (NIDEM_HOOK_NAMES in
  experiments/v15_nidem_scene_curves.py: nidem_on_scenes, v15's own path)
  under the COMMON per-pixel mask (pixel_rule 'common': SCL in {4, 5, 6, 7},
  the extraction's C, applied to the whole frame its contours and TIN need,
  checked equal to the extraction's C and Y at every keep pixel); a headline
  column. E1i: the same NIDEM on the same scenes and h under ITEM's mask (SCL
  in {2, 4, 5, 6, 7, 11}, v15's production rule), a declared sensitivity:
  scored on the common set, never part of it, never 'best'. A hook that
  takes ``subsets`` is called ONCE per run for every E1 and E1i window to do
  (one read of the cube for both rules; ``--io-lock``, ``--nidem-cache-dir``)
  and gates itself (ITEM) on MAREA's record against the stored sensitivity
  product. A window holding pre-2023 extra-store scenes is 'pending' for
  NIDEM (the extra store holds keep pixels only, no full-frame cube). The
  rule the hook reports it applied must be the column's, else 'error'.
  Without a hook: a registry 'E1', else 'pending';
* E2 DEA Intertidal 2.5.3 (experiments/fixedcfg/dea_intertidal.py, registered
  by methods.py when its side interpreter exists).
The catalogue methods with no code (docs/validation_v2_spec.md §3.2) are
'not implemented'.

The common inputs of a window, ``obs`` (types.SimpleNamespace): Y float32
(T, P) raw NDWI (NaN = no data), C bool (T, P), W bool (T, P) = C & (Y > 0),
h float64 (T,) = h_B(t) without lag, t pandas.DatetimeIndex (T,), keep int64
(P,) flat frame indices, shape (H, W), pixel_m, site; plus dates, n, window,
pixel_set and the frame builders ``frame(v, fill)`` ((P,) or (T, P) onto the
(H, W) grid, ``fill`` off keep) and ``unframe(img)``. Arrays are read-only.

Scoring
-------
One fixed unit set per site: F above (--max-px, default 200,000; the
subsamples are nested). Each (n, method) is scored (a) on the COMMON set, the
units where EVERY headline method computed at that n has a value (its share
of F is the row's common coverage), and (b) on its OWN valid units of F.
Metrics, both scorings, in the CSVs and the tables: cRMSE (RMSE after
removing each series' median, v7.vscore through v14.score), RMSE
(uncentred), bias = median(product - truth) (so RMSE^2 is not cRMSE^2 +
bias^2), OLS slope, r, n and coverage.

Timing (asked by the co-author, 2026-10-26)
--------------------------------------------
Per (site, n) and method, the wall time of the estimator call only
(time.perf_counter; no I/O, no truth loading): MAREA v14.run_job's own
seconds (lag refit and inversion, before v14 scores it), no delay
v14.invert_subset, a competitor run(obs, cfg) (or the estimator seconds the
adapter itself reports: DEA's per-pixel steps and edge cleaning, without its
side interpreter's start-up, imports and I/O; DEA runs its own pool of
processes, recorded), NIDEM the hook's own time of the subset (phase 2 plus
its share of the phase-1 R codes by number of scenes; the cube read is not
counted). The shared preprocessing once per
(site, n): the site's load (extraction streams, screening, levels; truth
loading and the gates excluded), the window table and the window selection.
With the host name and the BLAS thread settings: <site>.csv columns
seconds_estimator / seconds_shared / host / threads, tables/table_<n>.csv
columns seconds_<method> / seconds_shared / host / threads,
tables/timing_summary.json and a 'time (s)' row in the tables. Timing never
enters a computation: the results are the same with or without it.

Gates (a failed gate stops the run; --soft-gates continues and records it)
---------------------------------------------------------------------------
v14's sha256 (never soft); v14's own gates (its 'stored product vs published
row' is not applicable: the published point is scored on another truth); the
screened list's 2023-2025 part equals record_dates and its zero-lag level
range equals rango_marea_m; the window table equals v14's arrays and level
table bitwise on the shared scenes; the record window (MAREA, refitted lags,
mb 0) reproduces the stored MAREA on the units; at Villaviciosa the no-delay
product of the range filter belongs to the stored production (its 'MAREA'
equals the stored z) and the raster truth is on the extraction grid; the
NIDEM hook's own gates (MAREA's record reproduces the stored
sensitivity_marea_record product under ITEM's rule; the common mask is the
extraction's C and Y at keep); DEA Intertidal's temp root lies outside the
repository (the adapter refuses one inside it; checked before any job).

Extra extraction store (pre-2023 scenes; another builder writes it)
--------------------------------------------------------------------
results/fixedcfg/extra/<site>_extract_<years>.npz with the keys of
marea_extract.npz (Y, C, keep == the main keep, dates, shape) and, in the npz
or in the sidecar <stem>.json: ``overpass`` (date -> ISO label instant),
``cloud_pct`` (date -> production transition-zone %, against the production
reference map) or ``usable`` (dates; with both, ``usable`` must be exactly
``cloud_pct / 100 <= 0.10``), ``reference_map`` (the provenance of the map
the screening used; at the Dutch sites its class counts must be the
production map's, :data:`PRODUCTION_REFERENCE_CLASSES`) and ``gate_2023``
(dict with ``ok``: the bitwise re-download gate). Only dates before the main
extraction's first date are used. At a site whose h_B uses gauges (the Dutch
sites, gauges cached 2023-2025 only) the store is used only with
``--pre2023-levels production``: the production boundary object as it is,
i.e. EOT20 alone where no gauge has data (declared and counted per window).
The default refuses (the user's choice between the three options of the
investigation). A window holding store scenes is keyed by its dates AND each
store's (and sidecar's) bytes and sha256 and the level policy, so a rebuilt
store never reuses old jobs.

Outputs (new files only, under --out, default results/fixedcfg_truthdate)
-------------------------------------------------------------------------
<site>.csv             one row per (window, method, variant, scoring set);
                       at the Wadden the column 'site' holds the group row
                       (wadden-2025 / -2024 / -2023) or the pooled 'wadden'
<site>_settings.json   settings, truth, inputs, screening, windows, gates,
                       methods, reference points, timings, memory, versions
<site>_windows.json    the screened list and every window's scenes
<site>_units.npz       the scored units (keep index, truth, unit_col; at the
                       Wadden unit_group, the survey year of every unit;
                       the group windows' jobs are keyed 'y<year>_n040__E0b')
<site>_timing.json     host, threads and the shared preprocessing times
                       (<site>_timing__<tag>.json for a --tag group)
<site>_jobs/           one <key>.json + <key>.npz per (n, method): the
                       resumable state (the CSV is rebuilt from it)
tables/                per n: table_<n>.csv / .md / _es.tex, all_tables.md,
                       all_tables_es.tex, timing_summary.json (``tables``)
curves_<scoring>_<lang>.{png,pdf}, curves_<site>_<scoring>_<lang>.{png,pdf}
                       (``curves``)
A site's CSV enters the tables and curves only when its settings JSON holds
this design's truth block (TRUTH_DESIGN): results of earlier truths (the RTK,
all-years Vaklodingen) are never mixed in.

Run from the repository root:
    python -X utf8 -m experiments.v19_fixed_config_tables plan escalda
    python -X utf8 -m experiments.v19_fixed_config_tables run escalda
    python -X utf8 -m experiments.v19_fixed_config_tables run villaviciosa \\
        --vva-truth data_v4/truth/pnoa/MDT02-WGS84-0015-3-COB2.tif --vva-truth-date YYYY-MM-DD
    python -X utf8 -m experiments.v19_fixed_config_tables tables
    python -X utf8 -m experiments.v19_fixed_config_tables curves
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # BLAS threads per process (one process per site on the server; v14's
    # runs used 2). A value already in the environment wins.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402  (also chdir to ROOT)

#: a new folder for the truth-date design: earlier runs (RTK, all-years
#: Vaklodingen, windows ending 2025-12-31) stay in results/fixedcfg
OUT = "results/fixedcfg_truthdate"
EXTRA_DIR = "results/fixedcfg/extra"
NS = (125, 100, 75, 60, 50, 40)
#: quality rules scored from one MAREA fit: 0 = the published products' rule
#: (the headline), 3 = pyintertidal.marea.QUALITY_MIN_BRACKET (a variant)
MBS = (0, 3)
#: the rule that heads the tables (the user's decision; 0 = the rule of the
#: published and stored products, which the record gate reproduces). Every
#: role, column, label and curve below follows it; the other rules are
#: variants from the same fit
MB_HEADLINE = 0
#: MAREA's headline variant key and its other rules (variants)
HEAD_VAR = f"mb{MB_HEADLINE}"
MB_OTHERS = tuple(mb for mb in MBS if mb != MB_HEADLINE)
DEFAULT_MAX_PX = 200_000
DEFAULT_SEED = v14.DEFAULT_SEED
#: v14 as reused (uncommitted on 2026-10-20): imported, never edited
V14_SHA256 = "ef0f9a2e7b4f9b02f78d19fb22bed07a3b91ed5e8a4a59615e8333cfd4de5caf"
#: the production rule of filter_dates: usable iff cloud_pct / 100 <= 0.10
USABLE_CLOUD_MAX = 0.10
#: the span of the production records (v7.TIME_EXTENT)
RECORD_SPAN = ("2023-01-01", "2025-12-31")
#: the default window end (the end of the production records)
WINDOW_END = "2025-12-31"
#: the truths and window ends of the user's decisions (2026-10-20, 2026-10-26);
#: a settings JSON without this design's truth block is never tabulated
TRUTH_DESIGN = "truth-date design 2026-10-26"
#: Dutch sites: the most recent Vaklodingen survey inside the image archive
VAK_TRUTH_YEAR = 2025
DUTCH_WINDOW_END = f"{VAK_TRUTH_YEAR}-12-31"
DUTCH_END_RULE = (f"Vaklodingen gives the survey year only (time stamps YYYY-01-01): the truth is the 20 m cells "
                  f"surveyed in {VAK_TRUTH_YEAR} (the most recent survey inside the image archive 2023-2025) and "
                  f"the window ends on {DUTCH_WINDOW_END}, the end of the survey year")
#: Dutch sites whose truth is grouped by survey year (the user's choice of
#: 2026-10-26 at the Wadden, where 6 of 15 sheets hold a 2025 survey): the
#: years of the groups, the most recent first
SURVEY_YEAR_GROUPS = {"wadden": (2025, 2024, 2023)}
GROUP_END_RULE = ("Vaklodingen gives the survey year only (time stamps YYYY-01-01): per keep pixel the most recent "
                  "survey year in 2023-2025 of the 20 m cell under it (v1's nearest-cell sampling within each year), "
                  "the pixels grouped by that year, each group with its truth date at the end of its survey year and "
                  "its own windows (the n most recent screened images on or before it); a pooled row only at the n "
                  "where every group's window is reachable, each pixel scored with its own group's estimate")
VVA_END_RULE = "the PNOA LiDAR flight date given by --vva-truth-date (the window holds scenes on or before it)"
#: Villaviciosa: the second-coverage MDT02 the repository holds (flown 2015-2017, exact date unknown)
VVA_COB2 = "data_v4/truth/pnoa/MDT02-WGS84-0015-3-COB2.tif"
#: Villaviciosa's production no-delay product (the range filter) and its MAREA (must be the stored z)
VVA_NODELAY = "products_marea/external_products.npz"
#: the range filter's margin around the production no-delay range (v14.dutch_truth's 0.25 m)
RANGE_PAD_M = 0.25
#: the fixed unit set of every site (review of 2026-10-26: no product decides it)
F_RULE = ("F = the production extraction's keep pixels (every method's pixels) with a truth value inside the range "
          "filter (the production no-delay range +- 0.25 m); no product decides it, neither MAREA's validity nor any "
          "other method's: every method's coverage, MAREA's included, is measured on the same set")
#: the production reference map of the Dutch sites (classes 0 transition, 1
#: water, 2 land): the counts the notebooks printed, as the extra-store
#: builder's EXPECTED_REFERENCE (experiments/light_to_extraction.py; copied,
#: never imported). An extra store's sidecar must name a map with these counts
PRODUCTION_REFERENCE_CLASSES = {
    "escalda": {0: 777766, 1: 990308, 2: 1314093},
    "wadden": {0: 3051651, 1: 5025552, 2: 1430597},
    "ems": {0: 3401933, 1: 972618, 2: 6567527},
}
#: the logic of the MAREA / no-delay jobs; bump to invalidate cached jobs
JOB_VERSION = 1
#: how the runner calls a competitor and reads its result (part of every
#: competitor job's fingerprint); bump to invalidate cached competitor jobs
COMPETITOR_JOB_VERSION = 2
ROLES = ("headline", "variant", "internal")
REGISTRY_MODULE = "experiments.fixedcfg.methods"
NIDEM_HOOK_MODULE = "experiments.v15_nidem_scene_curves"
NIDEM_HOOK_NAMES = ("nidem_fixed_config", "nidem_on_scenes", "nidem_on_subset", "run_nidem_on_scenes",
                    "run_nidem_subset", "nidem_subset")
COMMON_MASK = "extraction C = SCL in {4, 5, 6, 7}"
#: NIDEM's two columns, one hook call and one cube read for both: E1 under the
#: common per-pixel mask (the design: SCL in {4, 5, 6, 7}; a headline column)
#: and E1i under ITEM's mask (v15's production rule, SCL in {2, 4, 5, 6, 7,
#: 11}): a declared sensitivity on the same scenes and h, scored on the common
#: set and never part of it
NIDEM_RULE = {"E1": "common", "E1i": "item"}
NIDEM_VARIANTS = {"E1": {"main": {"role": "headline", "label": "NIDEM"}},
                  "E1i": {"main": {"role": "variant", "label": "NIDEM, ITEM pixel mask (sensitivity)"}}}
NIDEM_RULE_NOTE = {
    "common": ("common mask: SCL in {4, 5, 6, 7} & finite NDWI, on the whole frame (checked equal to the "
               "extraction's C and Y at keep)"),
    "item": "DECLARED SENSITIVITY: ITEM's pixel mask, SCL in {2, 4, 5, 6, 7, 11} (v15's production rule)",
}
#: declared sensitivities with a column next to their method ((id, variant,
#: the method it sits after)): never 'best', never in the common set
SENSITIVITY_COLUMNS = (("E1i", "main", "E1"),)
#: where v14's settings JSONs are looked for (the F gate), first found
V14_SETTINGS_DIRS = ("results/scene_count_curves", "results/scene_count_curves/server_2026-10-20")

#: the elevation methods of docs/validation_v2_spec.md §3.2, in table order
CATALOGUE = (
    ("E0b", "MAREA"),
    ("E1", "NIDEM"),
    ("E2", "DEA Intertidal"),
    ("E3", "Granadeiro et al. 2021"),
    ("E4", "LogiTide2DEM"),
    ("E5a", "Flood2Topo"),
    ("E5b", "Tseng linear scaling"),
    ("E5c", "Sampled exceedance"),
    ("E6a", "Costa waterline"),
    ("E6b", "Optical S2 waterline"),
    ("E7", "Coast X-Ray"),
    ("E8", "Bué et al. 2020"),
    ("E12", "Wet/dry bracket"),
)
CATALOGUE_IDS = tuple(c[0] for c in CATALOGUE)
LABELS_ES = {"E0b": "MAREA", "E0a": "sin retardo", "E1": "NIDEM", "E2": "DEA Intertidal",
             "E3": "Granadeiro et al. 2021", "E4": "LogiTide2DEM", "E5a": "Flood2Topo",
             "E5b": "Tseng (escalado lineal)", "E5c": "Excedencia muestreada", "E6a": "Línea de agua (Costa)",
             "E6b": "Línea de agua óptica S2", "E7": "Coast X-Ray", "E8": "Bué et al. 2020",
             "E12": "Horquilla seco/mojado", "E1i": "NIDEM (máscara ITEM)"}
#: column labels of the methods that are not catalogue entries (English)
COLUMN_LABELS = {"E1i": "NIDEM (ITEM mask)"}
#: the methods the runner computes itself
RUNNER_IDS = ("E0b", "E0a")

OK, PENDING, NOT_COMPUTABLE, NOT_REACHABLE, NOT_IMPLEMENTED, ERROR = (
    "ok", "pending", "not computable", "not reachable", "not implemented", "error")
#: a site whose truth is not given yet (Villaviciosa without --vva-truth-date)
PENDING_TRUTH = "pending truth"
#: statuses whose job is not recomputed on resume (a deterministic outcome)
CACHEABLE = (OK, NOT_COMPUTABLE)

CSV_COLUMNS = ("site", "n_scenes", "is_record", "first_date", "last_date", "lower_bound_open", "years",
               "scenes_sha1", "method", "label", "variant", "role", "status", "scoring", "n_set", "n_common",
               "n_units", "n_px", "coverage", "rmse", "rmse_total", "bias", "slope", "r", "lo_m", "hi_m",
               "median_clear_obs", "seconds", "source", "fidelity", "notes", "reason",
               "seconds_estimator", "seconds_shared", "host", "threads")
SITE_ORDER = v14.SITE_ORDER
SITE_NAMES = {"villaviciosa": "Villaviciosa", "escalda": "Westerschelde", "wadden": "Wadden (Vlie)",
              "ems": "Ems-Dollard"}
#: the default truth labels (the settings JSON of a run holds the exact ones)
TRUTH_NAMES = {"villaviciosa": "PNOA LiDAR", "escalda": f"Vaklodingen {VAK_TRUTH_YEAR}",
               "wadden": "Vaklodingen 2023-2025 by survey year", "ems": f"Vaklodingen {VAK_TRUTH_YEAR}"}

COMMON_CONFIG = {
    "scene_filter": ("production transition-zone screening: pyintertidal.stability.reference_and_clouds(threshold "
                     "0.0, bad SCL {3, 8, 9, 10}, clean frame share <= 5 %, stable 95 %, transition buffer 10 px) "
                     "with the production reference map; usable iff cloud_pct <= 10 % (filter_dates); then an "
                     "overpass time and a finite production boundary level"),
    "pixel_mask": "C = SCL in {4, 5, 6, 7}: the production extraction's clear flags, the only per-pixel mask",
    "index": "raw NDWI = (B03 - B08) / (B03 + B08); NaN where undefined (MAREA reads NaN as 0, as production)",
    "wet": "W = C & (NDWI > 0)",
    "pixels": ("the production extraction's keep pixels (more than 30 clear observations over all cube scenes, "
               "0.10 < wet fraction < 0.90)"),
    "levels": ("the production boundary h_B of the site at the overpass label instants; MAREA reads "
               "h_B(t - tau_px) with its band lags refitted on the window; every other method reads h_B(t), "
               "no lag"),
    "window": ("the n most recent screened scenes on or before the site's truth date (Dutch sites 2025-12-31, the "
               "end of the Vaklodingen survey year; at the Wadden per survey-year group the end of its year, "
               "2023-12-31, 2024-12-31 or 2025-12-31; Villaviciosa the LiDAR flight date); nested, one realisation"),
}

log = v14.log
_T0 = time.time()


# ─────────────────────────────────────────────────────────────────────────────
#  small utilities
# ─────────────────────────────────────────────────────────────────────────────

def rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), ROOT).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def v14_sha_gate(path=None, expected=V14_SHA256):
    """v14 is reused by import only and gated by its sha256 (uncommitted)."""
    path = path or v14.__file__
    got = sha256_file(path)
    return {"path": rel(path), "sha256": got, "expected": expected, "ok": got == expected}


def scenes_sha1(dates):
    """A key for a scene list: sha1 of its dates (YYYY-MM-DD) in order."""
    return hashlib.sha1("|".join(str(d)[:10] for d in dates).encode("utf-8")).hexdigest()[:12]


def per_year(dates):
    out = {}
    for d in np.asarray(dates).astype(str):
        out[d[:4]] = out.get(d[:4], 0) + 1
    return dict(sorted(out.items()))


def years_text(years):
    return ";".join(f"{y}:{k}" for y, k in sorted(years.items()))


def group_row(site, year):
    """The table row (CSV 'site') of a survey-year group: 'wadden-2024'."""
    return f"{site}-{int(year)}"


def split_row(name):
    """(base site, survey year or None) of a table row: 'wadden-2024' ->
    ('wadden', 2024); 'wadden' (a site, or a grouped site's pooled row) ->
    ('wadden', None)."""
    s = str(name)
    base, _, y = s.rpartition("-")
    if base and len(y) == 4 and y.isdigit():
        return base, int(y)
    return s, None


def parse_ns(text):
    """'125,100,75' -> [125, 100, 75] (unique, descending)."""
    vals = sorted({int(v) for v in str(text or "").replace(";", ",").split(",") if v.strip()}, reverse=True)
    if not vals or any(v < 2 for v in vals):
        raise SystemExit(f"--ns: need integers >= 2, got {text!r}")
    return vals


def parse_list(text):
    return [v.strip() for v in str(text or "").replace(";", ",").split(",") if v.strip()]


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


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def load_avg():
    """The host's 1 / 5 / 15-minute load average (timing provenance on a
    shared server), or None where the OS has none."""
    try:
        return [round(float(v), 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def host_info():
    """The host name and the BLAS thread settings of this process (timing
    provenance): {'host', 'threads' (text), 'thread_env'}."""
    env = {k: os.environ.get(k) for k in THREAD_VARS}
    text = ";".join(f"{k.split('_')[0]}={v}" for k, v in env.items() if v is not None) or "unset"
    return {"host": socket.gethostname(), "threads": text, "thread_env": env}


def atomic_text(path, text):
    """Write ``text`` atomically, through a temp name of this process (--tag
    groups of one site rewrite the same CSV in parallel)."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    """v14.write_json (strict JSON, NaN -> None) through a temp name of this
    process (parallel --tag groups of one site)."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(v14._clean(obj), f, indent=1, default=v14._json_default, ensure_ascii=False)
    os.replace(tmp, path)


def load_overpass(path):
    """{date: datetime} exactly as v7.build_record reads overpass_times.json."""
    import pandas as pd

    with open(path, encoding="utf-8") as f:
        return {k: pd.Timestamp(v).to_pydatetime() for k, v in json.load(f).items()}


def site_paths(out, site, tag=""):
    t = f"__{tag}" if tag else ""
    return {"csv": os.path.join(out, f"{site}.csv"), "settings": os.path.join(out, f"{site}_settings{t}.json"),
            "windows": os.path.join(out, f"{site}_windows.json"), "units": os.path.join(out, f"{site}_units.npz"),
            "jobs": os.path.join(out, f"{site}_jobs"), "work": os.path.join(out, "work", site),
            "plan": os.path.join(out, f"{site}_plan.json"), "timing": os.path.join(out, f"{site}_timing{t}.json")}


# ─────────────────────────────────────────────────────────────────────────────
#  the screened list S
# ─────────────────────────────────────────────────────────────────────────────

def load_usable_dates(path=None):
    """Villaviciosa's production usable dates (transition-zone screening of
    the 2016-2025 cube): v7's recovery from the notebook's cell-23 HTML
    (asserts 1379 rows / 305 usable), or a JSON list (``--usable-json``: a
    list of dates, of {'date': ...} rows, or a dict holding one under
    'usable' / 'dates' / 'screened')."""
    if path:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            d = d.get("usable") or d.get("dates") or d.get("screened") or []
        out = sorted({(x["date"] if isinstance(x, dict) else str(x))[:10] for x in d})
        return out, f"{rel(path)} ({len(out)} dates)"
    from experiments import v7_waterline_contour as v7
    return v7.villaviciosa_usable_dates(), ("v7.villaviciosa_usable_dates(): cell-23 HTML of "
                                            f"{v7.VVA_NOTEBOOK} (1379 rows, 305 usable)")


def screen_candidates(dates, usable, overpass, levels_fn):
    """The screened scenes among the extraction ``dates``: production-usable
    AND an overpass time AND a finite level of ``levels_fn`` (one call over
    the chronological overpass instants; a boundary returns its levels
    sorted by time). Returns the rows into ``dates``, their dates, instants
    and zero-lag levels, and what was lost at each step."""
    import pandas as pd

    d = np.asarray(dates).astype(str)
    if len(d) > 1 and np.any(d[1:] <= d[:-1]):
        raise ValueError("extraction dates must be sorted and unique")
    us = {str(x)[:10] for x in usable}
    in_us = np.array([x in us for x in d], bool)
    have = np.array([x in overpass for x in d], bool)
    idx = np.flatnonzero(in_us & have)
    t = pd.DatetimeIndex(pd.to_datetime([overpass[x] for x in d[idx]]))
    if t.tz is not None:
        t = t.tz_localize(None)
    if len(t) > 1 and not t.is_monotonic_increasing:
        raise RuntimeError("overpass times are not chronological")
    h = np.asarray(levels_fn(t), float) if len(t) else np.zeros(0)
    if h.shape != (len(idx),):
        raise RuntimeError(f"the boundary returned {h.shape} levels for {len(idx)} instants")
    ok = np.isfinite(h)
    lost = {"usable_not_in_extraction": sorted(us - set(d.tolist())),
            "usable_without_overpass_time": d[in_us & ~have].tolist(),
            "usable_without_finite_level": d[idx[~ok]].tolist()}
    return {"idx": idx[ok].astype(np.int64), "dates": d[idx[ok]], "t": t[ok], "h0": h[ok], "lost": lost}


def screened_list(site, cfg, ext_dates, res, rec, rec_dates, usable_json=None):
    """S of a site from the main extraction: dict(dates, src (0 = main), row,
    t, h0) chronological, and its description."""
    import pandas as pd
    from experiments import v7_waterline_contour as v7

    if site == "villaviciosa":
        usable, src = load_usable_dates(usable_json)
        overpass = load_overpass(cfg.get("overpass") or v7.SITES[site]["overpass"])
        c = screen_candidates(ext_dates, usable, overpass, rec["boundary_obj"].levels)
        info = {"rule": ("production usable dates (transition-zone screening of the 2016-2025 cube) that are "
                         "extraction dates with an overpass time and a finite production boundary level"),
                "usable_source": src, "n_usable": int(len(usable)),
                "lost": {k: v for k, v in c["lost"].items()},
                "n_lost": {k: len(v) for k, v in c["lost"].items()}}
    else:
        d_all = np.asarray(rec["dates"]).astype(str)
        pos = {d: i for i, d in enumerate(d_all)}
        miss = [d for d in rec_dates if d not in pos]
        if miss:
            raise RuntimeError(f"[{site}] {len(miss)} record date(s) not in build_record's record: {miss[:5]}")
        k = np.array([pos[d] for d in rec_dates], np.int64)
        c = {"idx": np.asarray(rec["time_index"], np.int64)[k], "dates": d_all[k],
             "t": pd.DatetimeIndex(rec["t_real"])[k], "h0": np.asarray(rec["h"], float)[k]}
        info = {"rule": ("result.json record_dates: the production transition-zone screening of the 2023-2025 "
                         "cube, every one with an overpass time and a finite production boundary level"),
                "usable_source": "result.json record_dates", "n_usable": int(len(rec_dates))}
    S = {"dates": np.asarray(c["dates"]).astype(str), "src": np.zeros(len(c["dates"]), np.int64),
         "row": np.asarray(c["idx"], np.int64), "t": pd.DatetimeIndex(c["t"]),
         "h0": np.asarray(c["h0"], float)}
    info["n_main"] = int(len(S["dates"]))
    info["boundary"] = rec.get("boundary")
    return S, info


def record_part_gates(S, rec_dates, rango, span=RECORD_SPAN):
    """S's part inside the production span equals record_dates, and its
    zero-lag level range equals result.json rango_marea_m."""
    d = np.asarray(S["dates"]).astype(str)
    m = (d >= span[0]) & (d <= span[1])
    part = d[m].tolist()
    same = part == [str(x)[:10] for x in rec_dates]
    g = {"screened_record_part_equals_record_dates": {
        "n_part": len(part), "n_record": len(rec_dates), "equal": bool(same), "ok": bool(same)}}
    if m.any():
        lohi = [float(S["h0"][m].min()), float(S["h0"][m].max())]
        dr = float(np.max(np.abs(np.asarray(lohi) - np.asarray(rango, float))))
    else:
        lohi, dr = None, float("inf")
    g["screened_record_level_range_vs_rango_marea_m"] = {"mine": lohi, "stored": list(rango),
                                                         "max_abs_diff_m": dr, "ok": dr <= v14.TOL_LEVEL}
    # the record must be the newest part of S: the record window is S[-n_record:]
    tail = d[-len(rec_dates):].tolist() if len(rec_dates) and len(d) >= len(rec_dates) else []
    g["record_is_the_newest_part_of_S"] = {"ok": tail == [str(x)[:10] for x in rec_dates]}
    return g


# ── the extra extraction store (pre-2023 scenes) ────────────────────────────

def find_extra_stores(site, extra_dir=EXTRA_DIR):
    return sorted(p for p in glob.glob(os.path.join(extra_dir, f"{site}_extract_*.npz"))
                  if ".tmp" not in os.path.basename(p))


def reference_map_check(site, ref):
    """(problem or None, note) of an extra store's ``reference_map``
    provenance: present, and at a site whose production map is known
    (:data:`PRODUCTION_REFERENCE_CLASSES`) with exactly its class counts."""
    if not isinstance(ref, dict):
        return ("no reference_map provenance in the store's sidecar: its screening is not shown to use the "
                "production reference map"), None
    want = PRODUCTION_REFERENCE_CLASSES.get(site)
    if want is None:
        return None, "provenance present; no production class counts known for this site (counts not checked)"
    try:
        got = {int(k): int(v) for k, v in (ref.get("classes") or {}).items()}
    except (TypeError, ValueError, AttributeError):
        got = None
    if got != want:
        return (f"the reference map's class counts {ref.get('classes')} are not the production map's {want}"), None
    return None, "class counts equal the production map's (the notebooks' print)"


def _aligned(value, dates):
    """{date: v} from a dict, or a sequence aligned with ``dates``."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k)[:10]: v for k, v in value.items()}
    arr = list(np.asarray(value).tolist())
    if len(arr) != len(dates):
        raise ValueError(f"{len(arr)} values for {len(dates)} dates")
    return dict(zip([str(x) for x in dates], arr))


def load_extra_store(path, keep, shape, main_dates, site=None, soft=False):
    """One extra extraction: its dates before the main extraction's first
    date with their overpass times and production screening, or a refusal.
    The sidecar's ``reference_map`` must be the production map's
    (:func:`reference_map_check`) and, when both are given, ``usable`` must be
    exactly ``cloud_pct / 100 <= 0.10`` (``soft``: recorded, not refused).
    Returns (store dict or None, info)."""
    import pandas as pd

    info = {"path": rel(path), "used": False}
    req = {"Y", "C", "keep", "dates", "shape"}
    with np.load(path, allow_pickle=False) as ex:
        files = set(ex.files)
        if req - files:
            info["reason"] = f"missing keys {sorted(req - files)} (needs those of marea_extract.npz)"
            return None, info
        if not np.array_equal(np.asarray(ex["keep"]), np.asarray(keep)):
            info["reason"] = "keep differs from the main extraction"
            return None, info
        if tuple(int(v) for v in ex["shape"]) != tuple(int(v) for v in shape):
            info["reason"] = f"grid {tuple(ex['shape'])} differs from the main extraction {tuple(shape)}"
            return None, info
        dates = np.asarray(ex["dates"]).astype(str)
        inner = {k: ex[k] for k in ("overpass", "cloud_pct", "usable") if k in files}
        gate_npz = str(ex["gate_2023"]) if "gate_2023" in files else None
    side = {}
    sp = path[:-4] + ".json"
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            side = json.load(f)
    if len(dates) > 1 and np.any(dates[1:] <= dates[:-1]):
        info["reason"] = "dates are not sorted and unique"
        return None, info
    try:
        over = _aligned(side.get("overpass", inner.get("overpass")), dates)
        cloud = _aligned(side.get("cloud_pct", inner.get("cloud_pct")), dates)
        usable = side.get("usable")
        if usable is None and "usable" in inner:
            usable = [d for d, u in zip(dates, np.asarray(inner["usable"]).astype(bool)) if u]
    except ValueError as e:
        info["reason"] = f"sidecar values do not align with the dates: {e}"
        return None, info
    gate = side.get("gate_2023")
    if gate is None and gate_npz:
        try:
            gate = json.loads(gate_npz)
        except ValueError:
            gate = {"text": gate_npz}
    info["gate_2023"] = gate
    if over is None:
        info["reason"] = "no overpass times (npz key or sidecar 'overpass')"
        return None, info
    if cloud is None and usable is None:
        info["reason"] = "no production screening ('cloud_pct' or 'usable'): the scenes cannot be filtered"
        return None, info
    first_main = str(np.asarray(main_dates).astype(str)[0])
    early = dates < first_main
    info["dates_in_store"] = int(len(dates))
    info["dates_before_main"] = int(early.sum())
    info["dates_ignored_main_wins"] = int((~early).sum())
    # the production reference map, and a 'usable' list that is the production rule on cloud_pct
    info["reference_map"] = side.get("reference_map")
    problems = []
    why, note = reference_map_check(site, side.get("reference_map"))
    info["reference_map_check"] = why or note
    if why:
        problems.append(why)
    from_cloud = None
    if cloud is not None:
        from_cloud = {d for d, v in cloud.items() if v is not None and np.isfinite(float(v))
                      and float(v) / 100.0 <= USABLE_CLOUD_MAX}
    if usable is not None and from_cloud is not None:
        us_given = {str(x)[:10] for x in usable}
        if us_given != from_cloud:
            problems.append(f"the sidecar's 'usable' list is not cloud_pct / 100 <= {USABLE_CLOUD_MAX}: "
                            f"{len(us_given - from_cloud)} extra, {len(from_cloud - us_given)} missing")
    if problems:
        info["screening_problems"] = problems
        if not soft:
            info["reason"] = "; ".join(problems) + " (--soft-gates overrides)"
            return None, info
    if from_cloud is not None:
        us = from_cloud
        rule = f"cloud_pct / 100 <= {USABLE_CLOUD_MAX} (filter_dates)"
    else:
        us = {str(x)[:10] for x in usable}
        rule = "sidecar/npz 'usable' (no cloud_pct to check it against)"
    ov = {}
    for d, v in over.items():
        if v is not None and str(v) not in ("", "nan", "None"):
            ov[d] = pd.Timestamp(v).to_pydatetime()
    info["screening_rule"] = rule
    store = {"path": path, "dates": dates, "early": early, "usable": us, "overpass": ov, "gate_2023": gate}
    return store, info


def store_identity(path):
    """The bytes and sha256 of an extra store and of its sidecar (the key of
    the windows that hold its scenes)."""
    out = {"npz": {"path": rel(path), "bytes": int(os.path.getsize(path)), "sha256": sha256_file(path)}}
    sp = path[:-4] + ".json"
    if os.path.exists(sp):
        out["sidecar"] = {"path": rel(sp), "bytes": int(os.path.getsize(sp)), "sha256": sha256_file(sp)}
    return out


def extend_with_extra(S, site, extra_dir, keep, shape, main_dates, boundary, policy, soft=False):
    """S with the screened scenes of the extra stores prepended (dates before
    the main extraction). Returns (S, info list, per-scene level note)."""
    import pandas as pd

    infos = []
    paths = find_extra_stores(site, extra_dir)
    if not paths:
        return S, [], None
    uses_gauges = type(boundary).__name__ != "PyTMDBoundary"
    parts = []
    for k, p in enumerate(paths, start=1):
        st, info = load_extra_store(p, keep, shape, main_dates, site=site, soft=soft)
        infos.append(info)
        if st is None:
            log(f"[{site}] extra store {rel(p)} NOT used: {info['reason']}")
            continue
        g = st["gate_2023"]
        if not (isinstance(g, dict) and g.get("ok") is True) and not soft:
            info["reason"] = ("no passed 2023 re-download gate ('gate_2023' with ok = true): the new source is not "
                              "shown equal to the legacy cube (--soft-gates overrides)")
            log(f"[{site}] extra store {rel(p)} NOT used: {info['reason']}")
            continue
        if uses_gauges and policy != "production":
            info["reason"] = ("the site's h_B uses gauges cached for 2023-2025 only: pre-2023 levels need the "
                              "user's choice (--pre2023-levels production = the production boundary as it is, EOT20 "
                              "alone where no gauge has data)")
            log(f"[{site}] extra store {rel(p)} NOT used: {info['reason']}")
            continue
        d = st["dates"]
        cand = np.flatnonzero(st["early"])
        c = screen_candidates(d[cand], st["usable"], st["overpass"], boundary.levels)
        rows = cand[c["idx"]]
        info.update(used=True, source_index=k, screened=int(len(rows)),
                    lost={kk: len(v) for kk, v in c["lost"].items()},
                    levels=("production boundary as it is (EOT20 alone where no gauge has data)" if uses_gauges
                            else "production boundary (global model)"),
                    pre2023_levels=policy, identity=store_identity(p))
        parts.append({"dates": c["dates"], "src": np.full(len(rows), k, np.int64), "row": rows.astype(np.int64),
                      "t": c["t"], "h0": c["h0"]})
        log(f"[{site}] extra store {rel(p)}: {len(rows)} screened scenes before {main_dates[0]} added")
    if not parts:
        return S, infos, None
    allp = parts + [S]
    dates = np.concatenate([q["dates"] for q in allp])
    order = np.argsort(dates, kind="stable")
    if len(np.unique(dates)) != len(dates):
        raise RuntimeError(f"[{site}] the extra stores repeat a date")
    out = {"dates": dates[order], "src": np.concatenate([q["src"] for q in allp])[order],
           "row": np.concatenate([q["row"] for q in allp])[order],
           "t": pd.DatetimeIndex(np.concatenate([pd.DatetimeIndex(q["t"]).values for q in allp])[order]),
           "h0": np.concatenate([q["h0"] for q in allp])[order]}
    if not out["t"].is_monotonic_increasing:
        raise RuntimeError(f"[{site}] the extended list is not chronological")
    return out, infos, "pre-2023 scenes from the extra store(s)"


# ─────────────────────────────────────────────────────────────────────────────
#  the truth of each site and its window end (the user's decisions, 2026-10-26)
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(text, what):
    """'2016-08-15' -> '2016-08-15' (a valid calendar date), else SystemExit."""
    try:
        return _dt.date.fromisoformat(str(text).strip()[:10]).isoformat()
    except ValueError:
        raise SystemExit(f"{what}: {text!r} is not a date YYYY-MM-DD") from None


def truth_definition(site, args):
    """The truth of ``site`` under this design and its window end: a dict
    (design, status 'ready' or 'pending', kind, source, label_en/_es, date,
    end, end_rule, ...). Villaviciosa without ``--vva-truth`` and
    ``--vva-truth-date`` is 'pending' (the RTK is never used instead)."""
    if site == "villaviciosa":
        path = getattr(args, "vva_truth", None)
        date = getattr(args, "vva_truth_date", None)
        base = {"design": TRUTH_DESIGN, "site": site, "kind": "raster", "source": "PNOA LiDAR (a GeoTIFF DTM)",
                "end_rule": VVA_END_RULE, "scored_as": "a raster truth: v14.score 'vaklodingen' (v7.vscore)",
                "datum": ("orthometric heights (PNOA): the RMSE is median-centred; the bias is reported but carries "
                          "the datum offset"),
                "never": "the RTK survey (pyintertidal.rtk) is not read at Villaviciosa"}
        missing = [o for o, v in (("--vva-truth", path), ("--vva-truth-date", date)) if not v]
        if missing:
            return dict(base, status="pending", label_en="PNOA LiDAR", label_es="LiDAR PNOA",
                        reason=(f"PNOA LiDAR truth: {' and '.join(missing)} not given (the RTK is never used "
                                f"instead); e.g. --vva-truth {VVA_COB2} --vva-truth-date <its flight date>"))
        if not os.path.exists(path):
            raise SystemExit(f"--vva-truth {path}: no such file")
        d = parse_date(date, "--vva-truth-date")
        return dict(base, status="ready", path=rel(path), date=d, end=d, label_en=f"PNOA LiDAR, flight {d}",
                    label_es=f"LiDAR PNOA, vuelo {d}",
                    nodelay=rel(getattr(args, "vva_nodelay", None) or VVA_NODELAY),
                    file={"path": rel(path), "bytes": int(os.path.getsize(path)), "sha256": sha256_file(path)})
    base = {"design": TRUTH_DESIGN, "site": site, "status": "ready", "kind": "vaklodingen",
            "source": "Rijkswaterstaat Vaklodingen (20 m RD cells, NAP)", "year": VAK_TRUTH_YEAR,
            "date": DUTCH_WINDOW_END, "end": DUTCH_WINDOW_END, "end_rule": DUTCH_END_RULE,
            "label_en": f"Vaklodingen {VAK_TRUTH_YEAR}", "label_es": f"Vaklodingen {VAK_TRUTH_YEAR}",
            "scored_as": "v14.score 'vaklodingen' (v7.vscore)",
            "datum": "NAP; the products sit on the boundary's datum: the RMSE is median-centred"}
    if site not in SURVEY_YEAR_GROUPS:
        return base
    years = sorted({int(y) for y in SURVEY_YEAR_GROUPS[site]}, reverse=True)
    groups = [{"name": group_row(site, y), "year": y, "date": f"{y}-12-31", "end": f"{y}-12-31",
               "label_en": f"Vaklodingen {y}", "label_es": f"Vaklodingen {y}",
               "rule": (f"the keep pixels whose most recent Vaklodingen survey in {years[-1]}-{years[0]} is of {y}; "
                        f"the window ends on {y}-12-31, the end of that survey year")} for y in years]
    span = f"{years[-1]}-{years[0]}"
    return dict(base, year=years[0], years=years, groups=groups, date=groups[0]["end"], end=groups[0]["end"],
                end_rule=GROUP_END_RULE, label_en=f"Vaklodingen {span} by survey year",
                label_es=f"Vaklodingen {span} por año de levantamiento",
                grouping=("survey-year groups (the user's choice, 2026-10-26): " + "; ".join(
                    f"{g['name']}: {g['rule']}" for g in groups)
                    + "; the pooled row only at the n where every group's window is reachable, each pixel scored "
                      "with its own group's estimate; the production record is the gate window of the "
                      f"{years[0]} group only"))


def truth_grid_year(site, year=VAK_TRUTH_YEAR, truth_dir=None, sheets=None):
    """v1.truth_grid restricted to ONE survey year: per sheet of
    v0.sheets_for(site), in v1's order, the value of every 20 m RD cell in a
    survey of ``year`` (the latest stamp of that year, as v1 keeps the
    latest), in v1's cell format {sheet: (x, y, z, year)} for
    v1.sample_truth; a sheet without a survey of that year is listed, not
    used. Returns (cells, info with the sheets and their cell counts)."""
    import xarray as xr
    from experiments import v1_vaklodingen_validate as v1

    if sheets is None:
        from experiments.v0_vaklodingen_fetch import sheets_for
        sheets = [s[0] for s in sheets_for(site)]
    tdir = truth_dir or v1.TRUTH_DIR
    cells, rows = {}, []
    for name in sheets:
        path = os.path.join(tdir, name)
        row = {"sheet": name}
        rows.append(row)
        if not os.path.exists(path):
            row.update(used=False, reason="sheet file missing")
            log(f"[{site}] Vaklodingen sheet {name}: missing")
            continue
        ds = xr.open_dataset(path)
        try:
            zvar = "z" if "z" in ds else [v for v in ds.data_vars if ds[v].ndim == 3][0]
            yrs = np.asarray(ds["time"].dt.year.values, int)
            row["survey_years"] = sorted({int(y) for y in yrs if v1.YEARS[0] <= y <= v1.YEARS[1]})
            sel = np.flatnonzero(yrs == int(year))
            if not len(sel):
                row.update(used=False, reason=f"no survey in {year}")
                continue
            z = np.asarray(ds[zvar].isel(time=sel).values, float)
            x, y = ds["x"].values, ds["y"].values
            row["stamps"] = [str(np.datetime64(v, "D")) for v in ds["time"].values[sel]]
        finally:
            ds.close()
        zt = np.full(z.shape[1:], np.nan)
        yt = np.full(z.shape[1:], -1)
        for k in range(len(sel)):                     # time order: a later stamp of the year wins
            m = np.isfinite(z[k])
            zt[m] = z[k][m]
            yt[m] = int(year)
        cells[name] = (x, y, zt, yt)
        row.update(used=True, cells=int(np.isfinite(zt).sum()))
    info = {"year": int(year), "truth_dir": rel(tdir), "sheets": rows, "n_sheets": len(rows),
            "n_sheets_with_year": int(sum(1 for r in rows if r.get("used"))),
            "cells_with_year": int(sum(r.get("cells", 0) for r in rows))}
    log(f"[{site}] Vaklodingen {year}: {info['n_sheets_with_year']} of {info['n_sheets']} sheets, "
        f"{info['cells_with_year']:,} cells")
    return cells, info


def sample_cells(cells, xq, yq):
    """v1.sample_truth (the nearest cell; the first sheet with a finite value
    wins) and, per point, the id of the cell it reads (sheet index * 2^40 +
    row * 2^20 + column; -1 where none). The values are v1.sample_truth's
    (asserted). Returns (value, year, cell id)."""
    from experiments import v1_vaklodingen_validate as v1

    xq, yq = np.asarray(xq, float), np.asarray(yq, float)
    out, yr = v1.sample_truth(cells, xq, yq)
    got = np.full(len(xq), np.nan)
    cid = np.full(len(xq), -1, np.int64)
    for si, (x, y, zt, _) in enumerate(cells.values()):
        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        ix = np.round((xq - x[0]) / dx).astype(int)
        iy = np.round((yq - y[0]) / dy).astype(int)
        ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
        v = np.full(len(xq), np.nan)
        v[ok] = zt[iy[ok], ix[ok]]
        take = np.isfinite(v) & ~np.isfinite(got)
        got[take] = v[take]
        cid[take] = (np.int64(si) << 40) | (iy[take].astype(np.int64) << 20) | ix[take].astype(np.int64)
    if not np.array_equal(got, out, equal_nan=True):
        raise RuntimeError("the cell ids do not follow v1.sample_truth")
    return out, yr, cid


def dutch_truth_year(site, cfg, keep, H, W, z_plain, year=VAK_TRUTH_YEAR, truth_dir=None, holder=None):
    """v14.dutch_truth with the truth restricted to ONE survey year (the
    cells of :func:`truth_grid_year`): the same grid and RD transform, v1's
    nearest-cell sampling and the production range filter (z_plain's range
    +- 0.25 m). Same arguments and return as v14.dutch_truth ((truth per keep
    pixel, info)), so that v14.load_site can call it; ``holder`` (a dict)
    receives the cell id of every keep pixel, the sheets and the seconds."""
    import pyproj
    from experiments import v7_waterline_contour as v7

    t0 = time.perf_counter()
    grid = v7.cube_grid(cfg["cube"])
    if (H, W) != (grid["H"], grid["W"]):
        raise RuntimeError(f"cube grid {grid['H']}x{grid['W']} != extraction {H}x{W}")
    keep = np.asarray(keep, np.int64)
    rows_k, cols_k = keep // W, keep % W
    to_rd = pyproj.Transformer.from_crs(grid["crs"], "EPSG:28992", always_xy=True)
    xq, yq = to_rd.transform(grid["xs"][cols_k], grid["ys"][rows_k])
    cells, sinfo = truth_grid_year(site, year, truth_dir)
    truth_raw, _, cid = sample_cells(cells, xq, yq)
    lo_, hi_ = float(np.nanmin(z_plain)), float(np.nanmax(z_plain))
    in_range = (truth_raw >= lo_ - RANGE_PAD_M) & (truth_raw <= hi_ + RANGE_PAD_M)
    truth = np.where(in_range, truth_raw, np.nan)
    fin = np.isfinite(truth)
    info = {"truth": (f"Rijkswaterstaat Vaklodingen, the 20 m RD cells surveyed in {year} only (NAP), nearest cell "
                      "(v19.truth_grid_year = v1.truth_grid restricted to one survey year; v1.sample_truth)"),
            "truth_rule": DUTCH_END_RULE, "survey_year": int(year), "sheets": sinfo["sheets"],
            "n_sheets": sinfo["n_sheets"], "n_sheets_with_year": sinfo["n_sheets_with_year"],
            "cells_with_year_on_the_sheets": sinfo["cells_with_year"],
            "range_filter_m": [lo_ - RANGE_PAD_M, hi_ + RANGE_PAD_M],
            "range_filter_source": "production comparison products.npz z_plain range +- 0.25 m (v14.dutch_truth's)",
            "truth_px": int(np.isfinite(truth_raw).sum()), "truth_in_range": int(fin.sum()),
            "cells_under_keep": int(len(np.unique(cid[np.isfinite(truth_raw)]))),
            "cells_in_range": int(len(np.unique(cid[fin]))),
            "pixel_m": abs(float(grid["xs"][1] - grid["xs"][0]))}
    if holder is not None:
        holder.update(cell_id=cid, sheets=sinfo, seconds=time.perf_counter() - t0)
    return truth, info


def dutch_truth_groups(site, cfg, keep, H, W, z_plain, years, truth_dir=None, holder=None):
    """The Dutch truth grouped by survey year (:data:`SURVEY_YEAR_GROUPS`):
    per keep pixel the most recent year of ``years`` whose survey has a value
    at the pixel's nearest 20 m cell (:func:`truth_grid_year` of each year,
    v1's sampling: the first sheet in v1's order with a value of that year
    wins) and that survey's value; then the production range filter
    (z_plain's range +- 0.25 m) alike in every group. A pixel whose most
    recent survey falls outside the range filter is out of F; it never falls
    back to an older survey. Same grid, RD transform and return as
    :func:`dutch_truth_year` ((truth per keep pixel, info)); ``holder``
    receives 'group_year' (per keep pixel, 0 where no survey of ``years``)
    and, per year, the cell ids and the sheets ('groups'), and the seconds."""
    import pyproj
    from experiments import v7_waterline_contour as v7

    t0 = time.perf_counter()
    years = sorted({int(y) for y in years}, reverse=True)
    grid = v7.cube_grid(cfg["cube"])
    if (H, W) != (grid["H"], grid["W"]):
        raise RuntimeError(f"cube grid {grid['H']}x{grid['W']} != extraction {H}x{W}")
    keep = np.asarray(keep, np.int64)
    rows_k, cols_k = keep // W, keep % W
    to_rd = pyproj.Transformer.from_crs(grid["crs"], "EPSG:28992", always_xy=True)
    xq, yq = to_rd.transform(grid["xs"][cols_k], grid["ys"][rows_k])
    raw = np.full(len(keep), np.nan)
    gy = np.zeros(len(keep), np.int64)
    per = {}
    for y in years:
        cells, sinfo = truth_grid_year(site, y, truth_dir)
        val, _, cid = sample_cells(cells, xq, yq)
        take = np.isfinite(val) & (gy == 0)            # the most recent survey with a value wins
        raw[take] = val[take]
        gy[take] = y
        per[y] = {"sheets": sinfo, "cell_id": np.where(take, cid, -1).astype(np.int64)}
    lo_, hi_ = float(np.nanmin(z_plain)), float(np.nanmax(z_plain))
    in_range = (raw >= lo_ - RANGE_PAD_M) & (raw <= hi_ + RANGE_PAD_M)
    truth = np.where(in_range, raw, np.nan)
    fin = np.isfinite(truth)
    ginfo = {}
    for y in years:
        g, c = gy == y, per[y]["cell_id"]
        s = per[y]["sheets"]
        ginfo[str(y)] = {"survey_year": y, "truth_px": int(g.sum()), "truth_in_range": int((g & fin).sum()),
                         "cells_under_keep": int(len(np.unique(c[g]))),
                         "cells_in_range": int(len(np.unique(c[g & fin]))),
                         "n_sheets": s["n_sheets"], "n_sheets_with_year": s["n_sheets_with_year"],
                         "cells_with_year_on_the_sheets": s["cells_with_year"]}
    info = {"truth": (f"Rijkswaterstaat Vaklodingen by survey year: per keep pixel the most recent survey in "
                      f"{years[-1]}-{years[0]} of the 20 m RD cell under it (NAP), nearest cell "
                      "(v19.truth_grid_year of each year; v1.sample_truth), grouped by that year"),
            "truth_rule": GROUP_END_RULE, "survey_years": years, "groups": ginfo,
            "range_filter_m": [lo_ - RANGE_PAD_M, hi_ + RANGE_PAD_M],
            "range_filter_source": "production comparison products.npz z_plain range +- 0.25 m (v14.dutch_truth's)",
            "truth_px": int(np.isfinite(raw).sum()), "truth_in_range": int(fin.sum()),
            "pixel_m": abs(float(grid["xs"][1] - grid["xs"][0]))}
    if holder is not None:
        holder.update(group_year=gy, groups=per, seconds=time.perf_counter() - t0)
    return truth, info


def vva_raster_truth(path, keep, H, W, cube=None, grid_path=None):
    """The raster truth (a DTM GeoTIFF) on the extraction grid, exactly as
    experiments/benchmark/truth.py's pnoa_truth builds the PNOA truth: the
    2 m ground pixels (its pnoa_ground: no constant-value patch of >= 50
    pixels equal at 1 mm, 4-connected = flattened water; no flight-level
    mask, the flight water level is not known), averaged to 10 m where >= 95
    % of the area is valid and >= 80 % of it ground. ``grid_path``: a
    GeoTIFF on the frame grid (default pnoa_truth's, the product grid the
    extraction uses; checked against the extraction shape and, when the
    cube is there, its transform). Returns (truth per keep pixel before the
    range filter, info, seconds)."""
    import rasterio
    from experiments.benchmark import truth as btruth

    t0 = time.perf_counter()
    t = btruth.pnoa_truth(grid_path=grid_path, mdt_path=path, flight_level=None)
    z = np.asarray(t["z_raw"], float)
    if z.shape != (int(H), int(W)):
        raise RuntimeError(f"the raster truth grid {z.shape} is not the extraction grid {(H, W)}")
    gpath = t["sources"][1]
    with rasterio.open(gpath) as g:
        gt = tuple(g.transform)[:6]
    grid_ok = None
    if cube and os.path.exists(cube):
        from experiments import v7_waterline_contour as v7
        ct = tuple(v7.cube_grid(cube)["transform"])[:6]
        grid_ok = bool(max(abs(float(a) - float(b)) for a, b in zip(gt, ct)) < 1e-6)
    info = {"truth": (f"PNOA LiDAR {rel(path)} on the 10 m extraction grid: experiments/benchmark/truth.py "
                      f"pnoa_truth, ground only (pnoa_ground: constant patches >= {btruth.PATCH_MIN_PX} px equal at "
                      f"{btruth.PATCH_TOL_M} m are water; no flight-level mask), 10 m mean with >= "
                      f"{100 * btruth.PNOA_MIN_VALID:.0f} % of the area valid and >= "
                      f"{100 * btruth.PNOA_MIN_GROUND:.0f} % of it ground"),
            "grid": rel(gpath), "grid_transform": [float(v) for v in gt], "grid_is_the_cube_grid": grid_ok,
            "water_patch_share_2m": t["water_patch_share_2m"], "ground_share_2m": t["ground_share_2m"],
            "patches": {k: t["patches"].get(k) for k in ("n_patches", "n_px")},
            "flight_level": None, "pixel_m": abs(float(gt[0]))}
    return z.reshape(-1)[np.asarray(keep, np.int64)], info, time.perf_counter() - t0


def vva_nodelay_range(path, keep, z_stored):
    """The range filter of the raster truth: the production no-delay
    product's range at keep (v14.dutch_truth's rule), from ``path``
    ('no_delay'), whose 'MAREA' must be the stored product (same valid set,
    |dz| <= TOL_Z). Returns (lo, hi, gate)."""
    keep = np.asarray(keep, np.int64)
    z_stored = np.asarray(z_stored, float)
    with np.load(path, allow_pickle=False) as E:
        nd = np.asarray(E["no_delay"], float).reshape(-1)[keep]
        ma = np.asarray(E["MAREA"], float).reshape(-1)[keep]
    fa, fb = np.isfinite(ma), np.isfinite(z_stored)
    both = fa & fb
    dz = float(np.max(np.abs(ma[both] - z_stored[both]))) if both.any() else float("inf")
    same = bool(np.array_equal(fa, fb))
    gate = {"path": rel(path), "same_valid_set": same, "max_abs_dz_m": dz, "ok": bool(same and dz <= v14.TOL_Z),
            "rule": ("the no-delay product of the range filter belongs to the stored production: its MAREA is the "
                     "stored z")}
    return float(np.nanmin(nd)), float(np.nanmax(nd)), gate


def raster_units(truth_raw, lo, hi, info, holder=None):
    """A function with v14.rtk_units' arguments and return ((keep index per
    unit, truth per unit, info)) that gives v14.load_site the raster truth's
    F: the range filter [lo - 0.25, hi + 0.25] m and a finite truth; no
    product decides it (:data:`F_RULE`; the stored MAREA it is given is only
    recorded for the gate). The RTK survey is not read."""
    truth_raw = np.asarray(truth_raw, float)

    def units(keep, W, z_stored):
        truth = np.where((truth_raw >= lo - RANGE_PAD_M) & (truth_raw <= hi + RANGE_PAD_M), truth_raw, np.nan)
        F = np.flatnonzero(np.isfinite(truth)).astype(np.int64)
        zs = np.asarray(z_stored, float)
        out = dict(info, range_filter_m=[lo - RANGE_PAD_M, hi + RANGE_PAD_M],
                   range_filter_source="production no-delay product range +- 0.25 m (v14.dutch_truth's rule)",
                   truth_px=int(np.isfinite(truth_raw).sum()), truth_in_range=int(np.isfinite(truth).sum()),
                   units_in_F=int(len(F)), units_in_F_with_stored_marea=int(np.isfinite(zs[F]).sum()),
                   F_rule=F_RULE)
        if holder is not None:
            holder["z_stored_seen"] = zs
        return F, truth[F], out

    return units


def dutch_units(site, cfg, H, z_plain, year=VAK_TRUTH_YEAR, truth_dir=None, holder=None):
    """A function with v14.rtk_units' arguments and return ((keep index per
    unit, truth per unit, info)) that gives v14.load_site the Dutch F:
    :func:`dutch_truth_year` (the survey year's 20 m cells, v1's sampling,
    the production range filter on ``z_plain``) at every keep pixel, F = its
    finite values; no product decides it (:data:`F_RULE`). ``year`` a tuple
    of years: the survey-year groups (:func:`dutch_truth_groups`; the same F
    rules in every group). ``holder`` receives the cell ids (per group), the
    sheets, the group year per keep pixel and the seconds (spent inside
    load_site)."""
    years = tuple(int(y) for y in year) if isinstance(year, (tuple, list)) else None

    def units(keep, W, z_stored):
        if years:
            truth, info = dutch_truth_groups(site, cfg, keep, H, W, z_plain, years, truth_dir, holder)
        else:
            truth, info = dutch_truth_year(site, cfg, keep, H, W, z_plain, year, truth_dir, holder)
        F = np.flatnonzero(np.isfinite(truth)).astype(np.int64)
        zs = np.asarray(z_stored, float)
        out = dict(info, units_in_F=int(len(F)), units_in_F_with_stored_marea=int(np.isfinite(zs[F]).sum()),
                   F_rule=F_RULE)
        if holder is not None:
            holder["z_stored_seen"] = zs
            holder["in_load_site"] = True
        return F, truth[F], out

    return units


@contextlib.contextmanager
def truth_swap(name, fn):
    """v14.<name> replaced by ``fn`` for the duration of the block (one call
    of v14.load_site). v14's file is not edited; its sha256 gate holds."""
    orig = getattr(v14, name)
    setattr(v14, name, fn)
    try:
        yield
    finally:
        setattr(v14, name, orig)


def window_end_reason(avail, end, n, S0, rate):
    """The reason of a window that is not reachable (parsed by the tables)."""
    miss = int(n - avail)
    yrs = miss / rate if rate else None
    return (f"S holds {avail} screened scenes up to {end}: {miss} missing"
            + (f" (about {yrs:.1f} years before {S0[:4]} at {rate:.1f} screened scenes per year)"
               if yrs is not None and avail else ""))


# ─────────────────────────────────────────────────────────────────────────────
#  windows
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(S_dates, h0, ns, n_record=None, all_dates=None, end=WINDOW_END):
    """Window(n) = the n most recent screened scenes on or before ``end``
    (the site's truth date) for every n in ``ns``, i.e. the time window
    [S_end[-n], ``end``] with S_end the scenes of S on or before ``end``;
    plus the record window (the newest ``n_record`` scenes of S, the gate):
    tabulated when S ends on or before ``end``, else ``gate_only`` (label
    'rec<n>', computed for the gate, never tabulated). Each window:
    reachability, the screened scenes available up to the end, its first and
    last screened date, the open lower bound (any start in (that, first]
    gives the same scenes), scenes per year, the span in days (first date to
    the end, inclusive), the cube scenes inside it and the pass share, the
    zero-lag level range, the scenes' sha1; a window not reachable says how
    many screened scenes it lacks (and about how many years at the site's
    screened rate)."""
    S = np.asarray(S_dates).astype(str)
    h0 = np.asarray(h0, float)
    T = len(S)
    end = str(end)[:10]
    avail = int(np.searchsorted(S, end, side="right")) if T else 0
    nyears = len({d[:4] for d in S[:avail]})
    rate = avail / nyears if nyears else 0.0
    alld = None if all_dates is None else np.unique(np.asarray(all_dates).astype(str))
    req = {int(v) for v in ns}
    specs = {(n, False) for n in req}
    if n_record:
        specs.add((int(n_record), avail < T))         # the record ends after the end: the gate only
    out = []
    for n, gate_only in sorted(specs, key=lambda x: (-x[0], x[1])):
        is_rec = bool(n_record and n == int(n_record) and (gate_only or avail == T))
        w = {"n": int(n), "label": f"rec{n}" if gate_only else f"n{n}", "is_record": is_rec,
             "requested": int(n) in req and not gate_only, "gate_only": bool(gate_only), "end": end,
             "available": avail, "screened_total": int(T)}
        stop = T if gate_only else avail
        if n > stop:
            miss = int(n - stop)
            yrs = miss / rate if rate else None
            w.update(reachable=False, missing=miss, missing_years=None if yrs is None else round(yrs, 1),
                     reason=window_end_reason(stop, end if not gate_only else S[-1] if T else end, n,
                                              S[0] if T else "", rate))
            out.append(w)
            continue
        a = stop - n
        d = S[a:stop]
        first, last = d[0], d[-1]
        end_date = str(last) if gate_only else max(end, str(last))
        w.update(reachable=True, start=int(a), stop=int(stop), first_date=str(first), last_date=str(last),
                 end_date=end_date, lower_bound_open=str(S[a - 1]) if a > 0 else None, years=per_year(d),
                 span_days=int((np.datetime64(end_date) - np.datetime64(first)).astype(int)) + 1,
                 scenes_sha1=scenes_sha1(d), lo_m=float(h0[a:stop].min()), hi_m=float(h0[a:stop].max()))
        if gate_only:
            w["reason"] = (f"the production record ({first}..{last}) ends after the truth date {end}: computed only "
                           "as the reproduction gate, not tabulated")
        if alld is not None:
            k = int(((alld >= first) & (alld <= end_date)).sum())
            w.update(cube_scenes=k, pass_share=float(n / k) if k else None)
        out.append(w)
    return out


def table_windows(windows):
    """The windows of the tables: reachable and not gate-only."""
    return [w for w in windows if w.get("reachable") and not w.get("gate_only")]


def check_nested(windows):
    """The tabulated windows are nested and end on the same scene (by
    construction; checked); with survey-year groups, within each group."""
    def one(ws):
        rw = sorted(ws, key=lambda w: w["n"])
        ok = all(a["start"] >= b["start"] and a["stop"] == b["stop"] for a, b in zip(rw, rw[1:]))
        return {"ok": bool(ok), "sizes": [w["n"] for w in rw]}

    tw = table_windows(windows)
    groups = sorted({w.get("group") for w in tw if w.get("group") is not None}, reverse=True)
    if not groups:
        return one(tw)
    per = {str(g): one([w for w in tw if w.get("group") == g]) for g in groups}
    return {"ok": all(v["ok"] for v in per.values()), "sizes": [w["n"] for w in sorted(tw, key=lambda w: w["n"])],
            "groups": per}


def group_windows(S_dates, h0, ns, n_record, all_dates, groups):
    """The windows of a site whose truth is grouped by survey year
    (:func:`truth_definition`'s 'groups', the most recent first): per group
    :func:`make_windows` on its own truth date (the end of its survey year),
    labelled 'y<year>_<label>' and tagged with the group ('group', the
    table row 'row_site', the job-key prefix 'key_prefix'). The production
    record is the gate window of the most recent group only: in an earlier
    group it ends after the group's date and is a 'not reachable' record row
    (never computed twice)."""
    out = []
    for k, g in enumerate(groups):
        ws = make_windows(S_dates, h0, ns, n_record, all_dates, end=g["end"])
        for w in ws:
            w.update(group=int(g["year"]), row_site=g["name"], key_prefix=f"y{int(g['year'])}",
                     label=f"y{int(g['year'])}_{w['label']}")
            if k > 0 and w.get("is_record") and w.get("reachable"):
                first, last = w.get("first_date"), w.get("last_date")
                w.update(reachable=False, gate_only=False, requested=False,
                         reason=(f"the production record ({first}..{last}) ends after the truth date {g['end']} of "
                                 f"{g['name']}: not tabulated for this group (the reproduction gate runs on the "
                                 f"{groups[0]['name']} record window)"))
        out += ws
    return out


def rekey_windows(windows, S, x_info):
    """Key every reachable window that holds extra-store scenes by its dates
    AND, per store it reads, the store's and sidecar's bytes and sha256 and
    the pre-2023 level policy, so that a rebuilt store (or another level
    policy) never reuses old jobs; a window of main-extraction scenes keeps
    :func:`scenes_sha1` of its dates. Changes the windows in place and
    returns them."""
    ids = {int(x["source_index"]): x for x in (x_info or []) if x.get("used")}
    src = np.asarray(S["src"])
    dates = np.asarray(S["dates"]).astype(str)
    for w in windows:
        if not w.get("reachable"):
            continue
        sl = slice(int(w["start"]), int(w["stop"]))
        used = sorted({int(s) for s in src[sl] if int(s) != 0})
        if not used:
            continue
        parts = [scenes_sha1(dates[sl])]
        for k in used:
            x = ids.get(k) or {}
            parts.append(f"{k}:{json.dumps(x.get('identity') or {}, sort_keys=True)}:{x.get('pre2023_levels')}")
        w["scenes_sha1_dates"] = scenes_sha1(dates[sl])
        w["scenes_sha1"] = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
        w["extra_sources"] = used
    return windows


# ─────────────────────────────────────────────────────────────────────────────
#  the window table (v14's SiteData on the screened list)
# ─────────────────────────────────────────────────────────────────────────────

def lag_fit_columns(cfg, centers):
    """Production's seeded lag-fit pixel draw over all keep pixels by band,
    as v14.load_site / marea.reconstruct draw it (configs/m2.yaml,
    default_rng(seed + 31), <= max_px_banda per band): (columns, band)."""
    import yaml

    with open(os.path.join(ROOT, "configs", "m2.yaml"), encoding="utf-8") as f:
        cfg2 = yaml.safe_load(f)
    maxb = int(cfg2["max_px_banda"]["m2a"])
    mp = os.path.join(cfg["product"], "marea.npz")
    if os.path.exists(mp):
        with np.load(mp, allow_pickle=False) as z:
            band_all = z["band"].astype(int)
    else:
        with np.load(os.path.join(cfg["comparison"], "products.npz"), allow_pickle=False) as z:
            band_all = z["band"].astype(int)
    rng = np.random.default_rng(cfg2["seed"] + 31)
    sel = []
    for k in range(len(centers)):
        c = np.where(band_all == k)[0]
        if len(c) > maxb:
            c = np.sort(rng.choice(c, maxb, replace=False))
        sel.append(c)
    return (np.concatenate(sel).astype(np.int64),
            np.concatenate([np.full(len(c), k) for k, c in enumerate(sel)]))


def build_table(site, sources, S, px, m2a_cols, keep_n, boundary, sd14, need_keep, pos14=None):
    """v14's SiteData on the scenes of S (chronological): the inverted pixels
    ``px`` and the lag-fit columns, NaN -> 0 as production, a LevelTable on
    S's instants; with ``need_keep`` also the raw NDWI and clear flags of
    every keep pixel (``Yk``/``Ck``, for the competitors). Rows of v14's pool
    (``pos14``: date -> row of sd14) are copied from sd14 when nothing else
    is needed; the other rows are streamed from their source (``sources[k]``
    = npz path of source k). Returns (table, info)."""
    from pyintertidal import marea

    T = len(S["dates"])
    t0 = time.time()
    Y0 = np.zeros((T, len(px)), np.float32)
    C0 = np.zeros((T, len(px)), bool)
    mY = np.zeros((T, len(m2a_cols)), np.float32)
    mC = np.zeros((T, len(m2a_cols)), bool)
    Yk = Ck = None
    if need_keep:
        log(f"[{site}] keep arrays for the competitors: {T} scenes x {keep_n:,} px = "
            f"{T * keep_n * 5 / 2 ** 30:.2f} GB (NDWI float32 + clear flags)")
    filled = np.zeros(T, bool)
    copied = 0
    if not need_keep and pos14:
        for q, d in enumerate(S["dates"]):
            if int(S["src"][q]) == 0 and d in pos14:
                i = pos14[d]
                Y0[q], C0[q] = sd14.Y0[i], sd14.C[i]
                mY[q], mC[q] = sd14.m2a["Y"][i], sd14.m2a["C"][i]
                filled[q] = True
                copied += 1
    streamed = {}
    for src in sorted({int(s) for s in S["src"]}):
        qs = np.flatnonzero((np.asarray(S["src"]) == src) & ~filled)
        if not len(qs):
            continue
        rows = np.asarray(S["row"], np.int64)[qs]
        if np.any(np.diff(rows) <= 0):
            raise RuntimeError(f"[{site}] source {src}: rows not ascending with the dates")
        path = sources[src]
        cols = [np.arange(keep_n, dtype=np.int64)] if need_keep else [np.asarray(px), np.asarray(m2a_cols)]
        t1 = time.time()
        Ys = v14.stream_columns(path, "Y", rows, cols)
        Cs = v14.stream_columns(path, "C", rows, cols)
        if need_keep:
            if len(qs) == T:                        # one source holds every scene: no second copy
                Yk, Ck = Ys[0], Cs[0].astype(bool, copy=False)
            else:
                if Yk is None:
                    Yk = np.full((T, keep_n), np.nan, np.float32)
                    Ck = np.zeros((T, keep_n), bool)
                Yk[qs] = Ys[0]
                Ck[qs] = Cs[0].astype(bool, copy=False)
            Y0[qs] = np.nan_to_num(Ys[0][:, px], nan=0.0).astype(np.float32, copy=False)
            C0[qs] = Cs[0][:, px].astype(bool, copy=False)
            mY[qs] = np.nan_to_num(Ys[0][:, m2a_cols], nan=0.0).astype(np.float32, copy=False)
            mC[qs] = Cs[0][:, m2a_cols].astype(bool, copy=False)
        else:
            Y0[qs] = np.nan_to_num(Ys[0], nan=0.0).astype(np.float32, copy=False)
            C0[qs] = Cs[0].astype(bool, copy=False)
            mY[qs] = np.nan_to_num(Ys[1], nan=0.0).astype(np.float32, copy=False)
            mC[qs] = Cs[1].astype(bool, copy=False)
        del Ys, Cs
        streamed[rel(path)] = {"rows": int(len(rows)), "columns": int(sum(len(c) for c in cols)),
                               "seconds": round(time.time() - t1, 1)}
        log(f"[{site}] streamed {len(rows)} rows x {sum(len(c) for c in cols):,} columns of {rel(path)} "
            f"[{time.time() - t1:.1f} s]")
    levels = v14.LevelTable(boundary, S["t"])
    need = {0} | {int(v) for v in np.unique(sd14.tau_fixed)} | {int(v) for v in marea.BANK_TAUS} | set(range(0, 181))
    levels.ensure(sorted(need))
    m2a = dict(sd14.m2a)
    m2a["Y"], m2a["C"] = mY, mC
    table = v14.SiteData(site, Y0, C0, levels, {"S": np.arange(T)}, sd14.unit_col, sd14.truth, sd14.kind,
                         sd14.tau_fixed, s_km=sd14.s_km, m2a=m2a)
    if need_keep:
        if Yk is None or Ck is None:
            raise RuntimeError(f"[{site}] the keep arrays were not filled")
        Yk.setflags(write=False)
        Ck.setflags(write=False)
    table.Yk, table.Ck = Yk, Ck
    table.S_dates = np.asarray(S["dates"]).astype(str)
    table.S_t = S["t"]
    info = {"scenes": int(T), "inverted_px": int(len(px)), "lagfit_px": int(len(m2a_cols)),
            "keep_arrays": bool(need_keep), "keep_px": int(keep_n) if need_keep else None,
            "rows_copied_from_v14": int(copied), "streamed": streamed, "seconds": round(time.time() - t0, 1)}
    return table, info


def compare_with_v14(table, sd14, S, pos14):
    """The window table equals v14's on the shared scenes: NDWI and clear
    flags of the inverted and lag-fit pixels bitwise, and the level table at
    every lag v14 holds."""
    q, i = [], []
    for k, d in enumerate(S["dates"]):
        if int(S["src"][k]) == 0 and d in pos14:
            q.append(k)
            i.append(pos14[d])
    q, i = np.asarray(q, np.int64), np.asarray(i, np.int64)
    g = {"shared_scenes": int(len(q))}
    if not len(q):
        g.update(ok=False, reason="no scene shared with v14's pool")
        return g
    g["ndwi"] = bool(np.array_equal(table.Y0[q], sd14.Y0[i]))
    g["clear"] = bool(np.array_equal(table.C[q], sd14.C[i]))
    g["lagfit_ndwi"] = bool(np.array_equal(table.m2a["Y"][q], sd14.m2a["Y"][i]))
    g["lagfit_clear"] = bool(np.array_equal(table.m2a["C"][q], sd14.m2a["C"][i]))
    lags = sorted(sd14.levels.L)
    table.levels.ensure(lags)
    dl = max((float(np.max(np.abs(table.levels.L[t][q] - sd14.levels.L[t][i]))) for t in lags), default=0.0)
    g["levels_max_abs_diff_m"] = dl
    g["n_lags_compared"] = len(lags)
    g["ok"] = bool(all(v for k, v in g.items() if isinstance(v, bool)) and dl <= v14.TOL_LEVEL)
    return g


def level_provenance(boundary, t):
    """Per scene: is every member of an ensemble boundary finite there (the
    consensus), or does the mean rest on fewer members (e.g. EOT20 alone
    where no gauge has data)? None for a single provider."""
    providers = getattr(boundary, "providers", None)
    if not providers:
        return None
    fin = []
    names = []
    for p in providers:
        names.append(getattr(p, "name", type(p).__name__))
        fin.append(np.isfinite(np.asarray(p.levels(t), float)))
    fin = np.vstack(fin)
    allf = fin.all(axis=0)
    return {"members": names, "finite_per_member": [int(f.sum()) for f in fin], "all_members": allf,
            "n_all_members": int(allf.sum()), "n_partial": int((~allf).sum())}


def clear_stats(C):
    """Clear observations per pixel of a (T, P) flag array: median, p10 and
    the pixels below MAREA's min_obs (8)."""
    if C is None or not C.size:
        return None
    k = np.count_nonzero(C, axis=0)
    return {"median": float(np.median(k)), "p10": float(np.quantile(k, 0.1)), "below_8": int((k < 8).sum()),
            "pixels": int(len(k))}


# ─────────────────────────────────────────────────────────────────────────────
#  methods: the registry, the v15 NIDEM hook, the common inputs
# ─────────────────────────────────────────────────────────────────────────────

def module_path(module):
    return os.path.join(ROOT, *module.split(".")) + ".py"


def load_registry(module=REGISTRY_MODULE, variants=False):
    """{id: dict(label, run, card, variant)} of experiments/fixedcfg/methods.py
    (METHODS, and its declared sensitivities VARIANTS when ``variants``) and
    its status, with the registry's own REGISTRY_ERRORS. A missing package or
    a failing import is logged, not raised: the run continues with MAREA and
    no delay."""
    path = module_path(module)
    info = {"module": module, "path": rel(path)}
    if not os.path.exists(path):
        return {}, dict(info, status="absent", reason=f"{rel(path)} does not exist yet")
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        return {}, dict(info, status="error", reason=f"import failed: {e!r}")
    M = getattr(mod, "METHODS", None)
    if not isinstance(M, dict):
        return {}, dict(info, status="error", reason="the module has no METHODS dict")
    regs = [(M, False)] + ([(getattr(mod, "VARIANTS", None) or {}, True)] if variants else [])
    out, bad = {}, {}
    for reg, is_var in regs:
        for mid, entry in reg.items():
            try:
                label, run, card = entry
            except (TypeError, ValueError):
                bad[str(mid)] = "entry is not (label, run, card)"
                continue
            out[str(mid)] = {"label": str(label), "run": run, "card": dict(card or {}), "variant": is_var}
    info.update(status="found", file=v14.file_info(rel(path)), methods=[m for m in out if not out[m]["variant"]],
                variants=[m for m in out if out[m]["variant"]], invalid=bad,
                registry_errors=dict(getattr(mod, "REGISTRY_ERRORS", {}) or {}),
                registry_skipped=dict(getattr(mod, "REGISTRY_SKIPPED", {}) or {}))
    return out, info


def resolve_run(run):
    """(callable, None) or (None, reason): ``run`` may be a callable, a
    'module:function' string imported now, or None (not built yet)."""
    if callable(run):
        return run, None
    if isinstance(run, str) and ":" in run:
        mod, fn = run.split(":", 1)
        try:
            f = getattr(importlib.import_module(mod), fn)
        except Exception as e:  # noqa: BLE001
            return None, f"not built yet ({run}: {e!r})"
        return (f, None) if callable(f) else (None, f"{run} is not callable")
    if run is None:
        return None, "not built yet (run is None)"
    return None, f"run is not callable ({type(run).__name__})"


def sibling_imports(path):
    """The files of the same package directory that the module at ``path``
    imports, anywhere in its source (``from <package> import x``, ``from
    <package>.x import y``, ``from . import x``, ``import <package>.x``), and
    theirs, transitively; sorted, ``path`` excluded."""
    import ast

    here = os.path.dirname(os.path.abspath(path))
    pkg = rel(here).replace("/", ".")
    seen, todo = set(), [os.path.abspath(path)]
    while todo:
        p = todo.pop()
        try:
            tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level == 1 or mod == pkg:
                    if mod and node.level == 1:
                        names.add(mod.split(".")[0])
                    names.update(a.name for a in node.names)
                elif mod.startswith(pkg + "."):
                    names.add(mod[len(pkg) + 1:].split(".")[0])
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(pkg + "."):
                        names.add(a.name[len(pkg) + 1:].split(".")[0])
        for nm in names:
            f = os.path.join(here, nm + ".py")
            if os.path.exists(f) and f not in seen and f != os.path.abspath(path):
                seen.add(f)
                todo.append(f)
    return sorted(seen)


def code_fingerprint(fn, extra=""):
    """sha256 (16 hex) of the file defining ``fn`` and, after a '+', of the
    sibling files it imports (:func:`sibling_imports`: for DEA, the worker,
    which holds DEA_VERSION; for the built-ins, elev_base and level_series):
    a changed method recomputes its cached jobs."""
    try:
        f = inspect.getsourcefile(fn)
        if f and os.path.exists(f):
            h = sha256_file(f)[:16]
            deps = sibling_imports(f)
            if deps:
                h += "+" + hashlib.sha256("".join(sha256_file(p) for p in deps).encode()).hexdigest()[:12]
        else:
            h = "nofile"
    except (TypeError, OSError):
        h = "builtin"
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}:{h}{extra}"


def dea_tmp_problem(cfg=None):
    """None, or why DEA Intertidal (E2) would fail on every window: the temp
    root its adapter would use (cfg['dea_tmp'], else $MAREA_DEA_TMP, else the
    system temp dir) lies inside the repository, which the adapter refuses
    (the server's env.sh sets TMPDIR inside the repository)."""
    try:
        from pathlib import Path

        from experiments.fixedcfg import dea_intertidal as dea
        root = Path(dea._options(cfg or {}, 1)["tmp_root"]).expanduser()
        if dea.worker._inside(root, dea.REPO_ROOT):
            return (f"DEA Intertidal's temp root {root} is inside the repository {dea.REPO_ROOT}: the adapter refuses "
                    f"it, so every E2 job would be an error. Set {dea.ENV_TMP} to a folder outside the repository "
                    "(or leave E2 out with --methods)")
    except Exception as e:  # noqa: BLE001
        log(f"DEA temp-root check skipped ({e!r})")
    return None


def pixel_set_of(card):
    """'keep' (every keep pixel: the default, right for any method) or
    'scored' (the inverted pixels of the scored units only: cheaper, valid
    for a strictly per-pixel method) from the card's ``pixel_set`` or
    ``per_pixel``."""
    ps = str(card.get("pixel_set") or "").lower()
    if ps in ("keep", "scored"):
        return ps
    return "scored" if card.get("per_pixel") is True else "keep"


def fidelity_tier(text):
    """'F3 (re-implemented ...)' -> 'F3'."""
    s = str(text or "").strip()
    return s.split()[0].rstrip(",;:") if s else ""


def find_nidem_hook(name=None, module=NIDEM_HOOK_MODULE):
    """(function, info) of the v15 NIDEM-on-given-scenes function, or (None,
    info) while v15 does not exist or exposes none of NIDEM_HOOK_NAMES."""
    path = module_path(module)
    names = (name,) if name else NIDEM_HOOK_NAMES
    if not os.path.exists(path):
        return None, {"status": "absent", "reason": f"{rel(path)} does not exist yet"}
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        return None, {"status": "error", "reason": f"import of {module} failed: {e!r}"}
    for nm in names:
        fn = getattr(mod, nm, None)
        if callable(fn):
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "?"
            return fn, {"status": "found", "function": f"{module}.{nm}", "signature": sig,
                        "file": v14.file_info(rel(path))}
    return None, {"status": "absent", "reason": f"{module} exposes none of {list(names)}",
                  "file": v14.file_info(rel(path))}


def call_accepting(fn, avail):
    """``fn(**kw)`` with the keywords of ``avail`` its signature accepts (all
    of them when it takes **kwargs); refuse when it needs one not given."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kw = dict(avail)
    else:
        kw = {k: v for k, v in avail.items() if k in params}
    need = [p.name for p in params.values() if p.default is p.empty and p.name not in kw
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)]
    if need:
        raise TypeError(f"the hook needs {need}; the runner provides {sorted(avail)}")
    return fn(**kw), sorted(kw)


def frame_builders(keep_flat, shape):
    """frame(v, fill) puts (P,) or (T, P) keep values on the (H, W) grid
    (``fill`` off keep; False for bool); unframe(img) reads them back."""
    H, W = (int(v) for v in shape)
    keep_flat = np.asarray(keep_flat, np.int64)

    def frame(v, fill=np.nan):
        v = np.asarray(v)
        isb = v.dtype == bool
        dt = bool if isb else (v.dtype if np.issubdtype(v.dtype, np.floating) else np.float32)
        f = False if isb else fill
        if v.ndim == 1:
            out = np.full(H * W, f, dt)
            out[keep_flat] = v
            return out.reshape(H, W)
        out = np.full((v.shape[0], H * W), f, dt)
        out[:, keep_flat] = v
        return out.reshape(v.shape[0], H, W)

    def unframe(img):
        img = np.asarray(img)
        return img.reshape(-1)[keep_flat] if img.ndim == 2 else img.reshape(img.shape[0], -1)[:, keep_flat]

    return frame, unframe


def window_obs(table, ctx, w, pixel_set="keep"):
    """The common inputs of window ``w`` (see the module docstring); the
    arrays are read-only views where possible."""
    if table.Yk is None:
        raise RuntimeError("the keep arrays were not loaded (no competitor was expected)")
    sl = window_slice(w)
    if pixel_set == "keep":
        Y, C = table.Yk[sl], table.Ck[sl]
        keep_flat = np.asarray(ctx.keep_flat, np.int64)
    else:
        Y, C = table.Yk[sl][:, ctx.px], table.Ck[sl][:, ctx.px]
        keep_flat = np.asarray(ctx.keep_flat, np.int64)[ctx.px]
        Y.setflags(write=False)
        C.setflags(write=False)
    with np.errstate(invalid="ignore"):
        W = C & (Y > 0)                         # NaN NDWI is never wet
    W.setflags(write=False)
    h = np.array(table.levels[0][sl], float)
    h.setflags(write=False)
    frame, unframe = frame_builders(keep_flat, ctx.shape)
    return types.SimpleNamespace(
        Y=Y, C=C, W=W, h=h, t=table.S_t[sl], keep=keep_flat, shape=tuple(int(v) for v in ctx.shape),
        pixel_m=float(ctx.pixel_m), site=ctx.site, dates=[str(d) for d in table.S_dates[sl]], n=int(w["n"]),
        window=w["label"], pixel_set=pixel_set, frame=frame, unframe=unframe)


def window_slice(w):
    """The rows of window ``w`` in the window table (the scenes of S)."""
    return slice(int(w["start"]), int(w["stop"]))


def window_rows(w):
    return np.arange(int(w["start"]), int(w["stop"]), dtype=np.int64)


def level_end(w):
    """The end (exclusive) of a window's 1-min level series: 00:00 of the day
    after its end date (the Dutch windows: 2026-01-01)."""
    import pandas as pd
    return (pd.Timestamp(str(w.get("end_date") or w.get("end") or WINDOW_END)[:10])
            + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def method_cfg(ctx, w, mid, pixel_set, needs=()):
    """The second argument of run(obs, cfg): the common configuration, where
    a method may write scratch files and, for a method whose card ``needs``
    it, ``level_series`` = h_B without lag at 1 min from 00:00 of the
    window's first scene day to the day after the window's end
    (:func:`level_end`, exclusive)."""
    work = os.path.join(ctx.paths["work"], w["label"], mid)
    cfg = {"site": ctx.site, "method": mid, "n": int(w["n"]), "window": w["label"], "first_date": w["first_date"],
           "last_date": w["last_date"], "window_end": w.get("end_date"), "is_record": bool(w["is_record"]),
           "seed": int(ctx.seed), "out_dir": work, "threads": os.environ.get("OMP_NUM_THREADS"),
           "pixel_set": pixel_set, "boundary": ctx.boundary_name, "level_source": "production h_B(t), no lag",
           "mask": COMMON_MASK, "index": "NDWI", "wet_threshold": 0.0, "common": dict(COMMON_CONFIG)}
    if getattr(ctx, "dea_tmp", None):
        cfg["dea_tmp"] = str(ctx.dea_tmp)              # DEA Intertidal's temp root (--dea-tmp), outside the repository
    if "level_series" in needs:
        full = getattr(ctx, "level_series", None)
        if full is not None:
            import pandas as pd
            t0 = pd.Timestamp(ctx.table.S_t[int(w["start"])]).normalize()
            cfg["level_series"] = full.slice(t0, level_end(w))
    return cfg


#: the 1-min level series of a window ending 2025-12-31 ends here (exclusive)
LEVEL_SERIES_END = "2026-01-01"


def level_series_for(ctx, boundary, windows, path):
    """The site's 1-min h_B series (no lag) from 00:00 of the earliest first
    scene day of the tabulated windows to the latest :func:`level_end`, from
    the cache at ``path`` when it covers that span with the same boundary,
    else built with experiments.fixedcfg.level_series.LevelSeries.
    from_boundary and cached. Nested windows slice it."""
    import pandas as pd
    from experiments.fixedcfg.level_series import LevelSeries

    rw = table_windows(windows)
    if not rw:
        return None, None
    start = pd.Timestamp(ctx.table.S_t[min(int(w["start"]) for w in rw)]).normalize()
    end = pd.Timestamp(max(level_end(w) for w in rw))
    src = f"{getattr(boundary, 'name', type(boundary).__name__)}, 1-min, no lag"
    if os.path.exists(path):
        try:
            ls = LevelSeries.load(path)
            if ls.t is not None and ls.t[0] <= start and ls.t[-1] >= end - pd.Timedelta(minutes=1) \
                    and ls.source == src:
                # the boundary keeps its name when its gauges or members change: spot-check the levels
                k = np.unique(np.linspace(0, len(ls.t) - 1, 64).astype(int))
                fresh = np.asarray(boundary.levels(ls.t[k]), float)
                ok = fresh.shape == (len(k),) and bool(np.allclose(fresh, ls.h[k], rtol=0, atol=v14.TOL_LEVEL,
                                                                   equal_nan=True))
                if ok:
                    return ls, {"cache": rel(path), "reused": True, "spot_check": {"instants": int(len(k)),
                                                                                    "ok": True}, **ls.summary()}
                log(f"[{ctx.site}] level series cache {rel(path)}: the boundary gives other levels at the "
                    f"spot-checked instants: rebuilt")
        except Exception as e:  # noqa: BLE001
            log(f"[{ctx.site}] level series cache {rel(path)} unreadable ({e!r}): rebuilt")
    t0 = time.time()
    ls = LevelSeries.from_boundary(boundary, start, end, source=src.split(",")[0])
    tmp = f"{path[:-4]}.tmp{os.getpid()}.npz"          # atomic: parallel runs of a site may race
    ls.save(tmp)
    os.replace(tmp, path)
    info = {"cache": rel(path), "reused": False, "seconds": round(time.time() - t0, 1), **ls.summary()}
    log(f"[{ctx.site}] level series {start}..{end} at 1 min: {len(ls.h):,} levels [{info['seconds']} s]")
    return ls, info


def to_units(z, pixel_set, ctx):
    """A method's (P,) values on the scored units."""
    z = np.asarray(z, float)
    return z[ctx.units_keep] if pixel_set == "keep" else z[ctx.unit_col]


def parse_method_result(res, P):
    """(status, z, sigma, meta, reason) from what run(obs, cfg) returned: a
    dict(z, sigma, meta) whose status is ``meta['status']`` (or a top-level
    'status'), 'ok' or 'not computable' with a 'reason'; or a bare array."""
    if isinstance(res, dict):
        meta = dict(res.get("meta") or {})
        st = str(res.get("status") or meta.get("status") or OK).strip().lower().replace("_", " ")
        if st.startswith("not comp"):
            return NOT_COMPUTABLE, None, None, meta, str(res.get("reason") or meta.get("reason") or "")
        if "z" not in res:
            raise ValueError(f"the result has no 'z' (keys {sorted(res)})")
        z, sig = res["z"], res.get("sigma")
    else:
        z, sig, meta = res, None, {}
    z = np.asarray(z, float)
    if z.shape != (P,):
        raise ValueError(f"z has shape {z.shape}; expected ({P},)")
    if sig is not None:
        sig = np.asarray(sig, float)
        if sig.shape != (P,):
            raise ValueError(f"sigma has shape {sig.shape}; expected ({P},)")
    if not np.isfinite(z).any():
        return NOT_COMPUTABLE, None, None, meta, "no finite estimate"
    return OK, z, sig, meta, ""


def hook_values(res, n_units, unit_col, units_keep, n_px, n_keep):
    """NIDEM on the scored units from what the hook returned: an array per
    unit, per inverted pixel or per keep pixel, or a dict holding one."""
    kind = None
    if isinstance(res, dict):
        for key, k in (("z_units", "units"), ("z_px", "px"), ("z_keep", "keep"), ("filtered_keep", "keep"),
                       ("z", None)):
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
        raise ValueError(f"the hook returned shape {arr.shape}; expected ({n_units},) per unit, ({n_px},) per "
                         f"inverted pixel or ({n_keep},) per keep pixel")
    if kind == "units":
        return arr
    if kind == "px":
        return arr[np.asarray(unit_col)]
    return arr[np.asarray(units_keep)]


# ─────────────────────────────────────────────────────────────────────────────
#  the resumable job store
# ─────────────────────────────────────────────────────────────────────────────

def job_key(n, mid, gate=False):
    """'n060__E0b'; the gate-only record window: 'r097__E0b'."""
    return f"{'r' if gate else 'n'}{int(n):03d}__{mid}"


def wkey(w, mid):
    """The job key of window ``w`` (a survey-year group's window: prefixed by
    its group, 'y2024_n040__E0b')."""
    k = job_key(w["n"], mid, bool(w.get("gate_only")))
    return f"{w['key_prefix']}_{k}" if w.get("key_prefix") else k


class JobStore:
    """One <key>.json (status, provenance, scores' inputs) and, for a
    computed job, <key>.npz (the estimate of every variant on the scored
    units) per (n, method). The npz is written before the json, both
    atomically: a job counts as done only when its json says so."""

    def __init__(self, root, create=True):
        self.root = root
        if create:
            os.makedirs(root, exist_ok=True)
        self._z = {}

    def _p(self, key):
        return os.path.join(self.root, key + ".json"), os.path.join(self.root, key + ".npz")

    def keys(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(f[:-5] for f in os.listdir(self.root) if f.endswith(".json") and ".tmp" not in f)

    def meta(self, key):
        jp, _ = self._p(key)
        if not os.path.exists(jp):
            return None
        try:
            with open(jp, encoding="utf-8") as f:
                return json.load(f)
        except ValueError:
            return None

    def arrays(self, key):
        _, zp = self._p(key)
        if not os.path.exists(zp):
            return None
        st = os.stat(zp)
        tag = (st.st_mtime_ns, st.st_size)
        hit = self._z.get(key)
        if hit and hit[0] == tag:
            return hit[1]
        with np.load(zp, allow_pickle=False) as f:
            out = {k: np.asarray(f[k]) for k in f.files}
        self._z[key] = (tag, out)
        return out

    def save(self, key, meta, arrays=None):
        jp, zp = self._p(key)
        if arrays:
            tmp = zp[:-4] + f".tmp{os.getpid()}.npz"
            np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
            os.replace(tmp, zp)
        elif os.path.exists(zp):
            os.remove(zp)
        self._z.pop(key, None)
        write_json(jp, meta)


def is_current(meta, scenes, units, fingerprint):
    return bool(meta and meta.get("scenes_sha1") == scenes and meta.get("units_sha1") == units
                and meta.get("fingerprint") == fingerprint)


def done(store, key, scenes, units, fingerprint):
    """A cached job is reused only with the same scenes, units and code, a
    cacheable status and (computed) its arrays."""
    m = store.meta(key)
    if not is_current(m, scenes, units, fingerprint) or m.get("status") not in CACHEABLE:
        return None
    if m["status"] == OK:
        z = store.arrays(key)
        if z is None or any(f"z__{v}" not in z for v in m.get("variants", {})):
            return None
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  the jobs
# ─────────────────────────────────────────────────────────────────────────────

def marea_job(table, rows, seed):
    """E0b: v14's refit job (v14.run_job, lags 'refit') on exactly the
    scenes ``rows`` of the window table, passed as a pool of its own and run
    whole (N = the pool: v14.draw draws nothing). Returns ({variant: z per
    unit}, info)."""
    from pyintertidal import marea

    rows = np.asarray(rows, np.int64)
    key = "__fixedcfg_window__"
    table.pools[key] = rows
    try:
        t_call = time.perf_counter()                     # timing only, never a result
        v14_rows, info = v14.run_job(table, key, len(rows), 0, seed, "refit", list(MBS))
        t_call = time.perf_counter() - t_call
    finally:
        table.pools.pop(key, None)
    # the estimator: v14's own seconds of the job (lag refit + inversion), measured before v14 scores it
    t_est = float(v14_rows[0]["seconds"]) if v14_rows and v14_rows[0].get("seconds") is not None else t_call
    if not np.array_equal(info["rows"], rows):
        raise RuntimeError("v14.run_job ran other scenes")
    tau = np.asarray(info["tau_bands"], float)
    tau_used = np.where(np.abs(tau) > marea.TAU_DETECT_MIN, tau, 0.0)
    tau_used = np.where(tau_used < 0, 0.0, tau_used)
    z = {f"mb{mb}": np.asarray(info["z"][mb], float)[table.unit_col] for mb in MBS}
    r0 = v14_rows[0]
    meta = {"tau_bands_min": tau.tolist(), "tau_used_bands_min": tau_used.tolist(),
            "tau_px_median_min": float(np.median(info["tau_px"])), "tau_px_max_min": float(np.max(info["tau_px"])),
            "lo_m": r0["lo_m"], "hi_m": r0["hi_m"], "median_clear_obs": r0["median_clear_obs"],
            "inversion_seconds": round(float(info["t_inv"]), 3),
            "v14_scores": {int(r["mb"]): {k: r[k] for k in ("rmse", "bias", "slope", "r", "coverage", "n")}
                           for r in v14_rows},
            "seconds_estimator": float(t_est), "seconds_call_with_v14_scoring": float(t_call),
            "function": "experiments.v14_scene_count_curves.run_job(lags='refit') on the window's scenes"}
    return z, meta


def nodelay_job(table, rows):
    """E0a: v14.invert_subset at tau = 0 on every inverted pixel (the same
    inversion, lo/hi = the window's zero-lag range)."""
    rows = np.asarray(rows, np.int64)
    groups = [(0, np.arange(table.Y0.shape[1]))]
    t_est = time.perf_counter()                          # the estimator call only (timing, never a result)
    zz, nclear, (lo, hi), t_inv, _ = v14.invert_subset(table.Y0, table.C, table.levels, rows, groups, list(MBS))
    t_est = time.perf_counter() - t_est
    z = {f"mb{mb}": np.asarray(zz[mb], float)[table.unit_col] for mb in MBS}
    meta = {"lo_m": lo, "hi_m": hi, "median_clear_obs": float(np.median(nclear[table.unit_col])),
            "inversion_seconds": round(float(t_inv), 3), "seconds_estimator": float(t_est),
            "function": "experiments.v14_scene_count_curves.invert_subset at tau = 0"}
    return z, meta


#: the variants of the runner's own jobs, from one fit each: MAREA's headline
#: rule (MB_HEADLINE) and its other rules (variants); no delay is internal
VARIANTS = {
    "E0b": {f"mb{mb}": ({"role": "headline", "label": "MAREA"} if mb == MB_HEADLINE else
                        {"role": "variant", "label": f"MAREA, quality rule mb {mb}"}) for mb in MBS},
    "E0a": {f"mb{mb}": {"role": "internal", "label": "no delay" if mb == MB_HEADLINE else
                        f"no delay, quality rule mb {mb}"} for mb in MBS},
}


def is_marea_variant(mid, var):
    """MAREA under a quality rule other than the headline one (a variant
    column from the same fit)."""
    return mid == "E0b" and var != HEAD_VAR


def is_side_column(mid, var):
    """A column that is not a headline method: MAREA's other rules and the
    declared sensitivities (never 'best', never in the common set)."""
    return is_marea_variant(mid, var) or any((mid, var) == (s[0], s[1]) for s in SENSITIVITY_COLUMNS)


class Context(types.SimpleNamespace):
    """What the jobs of one site share."""


def job_meta(ctx, w, mid, status, label, source, fidelity="", variants=None, reason="", seconds=0.0,
             fingerprint="", cacheable=True, info=None, seconds_estimator=None):
    """A job's json. ``seconds``: the job's wall time (inputs included);
    ``seconds_estimator``: the estimator call only (time.perf_counter), with
    the host and BLAS threads it ran on."""
    hi = host_info()
    out = {"key": wkey(w, mid), "site": ctx.site, "n": int(w["n"]), "window": w["label"],
           "is_record": bool(w["is_record"]), "gate_only": bool(w.get("gate_only")), "method": mid, "label": label,
           "status": status, "reason": reason, "source": source, "fidelity": fidelity, "variants": variants or {},
           "scenes_sha1": w["scenes_sha1"], "units_sha1": ctx.units_sha1, "fingerprint": fingerprint,
           "cacheable": bool(cacheable), "seconds": round(float(seconds), 2),
           "seconds_estimator": None if seconds_estimator is None else round(float(seconds_estimator), 4),
           "host": hi["host"], "threads": hi["threads"], "loadavg": load_avg(),
           "finished": _dt.datetime.now().isoformat(timespec="seconds"), "peak_rss_mb": peak_rss_mb(),
           "info": info or {}}
    if w.get("group") is not None:
        out.update(group=int(w["group"]), row_site=w.get("row_site"))
    return out


def note_window_select(ctx, w, seconds):
    """The window-selection time of ``w`` (building its common inputs), the
    fastest of its jobs: part of the shared preprocessing of (site, n)."""
    d = ctx.__dict__.setdefault("window_select_s", {})
    d[w["label"]] = min(float(seconds), d.get(w["label"], float("inf")))


def run_one(ctx, w, mid):
    """Compute one (window, method): returns (meta, arrays or None). The
    estimator call alone is timed (time.perf_counter; job meta
    'seconds_estimator'); timing never enters a computation."""
    t_sel = time.perf_counter()
    rows = window_rows(w)
    note_window_select(ctx, w, time.perf_counter() - t_sel)
    t0 = time.time()
    if mid in ("E0b", "E0a"):
        fp = f"v14:{V14_SHA256[:16]}|job:{JOB_VERSION}"
        if mid == "E0b":
            z, info = marea_job(ctx.table, rows, ctx.seed)
            src = "v14.run_job(lags='refit'): production inversion, band lags refitted on the window"
        else:
            z, info = nodelay_job(ctx.table, rows)
            src = "v14.invert_subset at tau = 0 (internal reference)"
        meta = job_meta(ctx, w, mid, OK, VARIANTS[mid][HEAD_VAR]["label"], src, "F0", VARIANTS[mid],
                        seconds=time.time() - t0, fingerprint=fp, info=info,
                        seconds_estimator=info.get("seconds_estimator"))
        return meta, {f"z__{k}": v for k, v in z.items()}
    if mid in NIDEM_RULE and ctx.nidem_fn is not None:
        return nidem_one(ctx, w, rows, t0, mid)
    ent = ctx.registry.get(mid)
    if ent is None:
        if mid in NIDEM_RULE:
            return job_meta(ctx, w, mid, PENDING, NIDEM_VARIANTS[mid]["main"]["label"], "v15 hook",
                            variants=NIDEM_VARIANTS[mid],
                            reason=f"v15 hook: {ctx.nidem_info.get('reason', 'absent')}", fingerprint="none",
                            cacheable=False), None
        return job_meta(ctx, w, mid, PENDING, mid, "registry", reason="not registered", fingerprint="none",
                        cacheable=False), None
    card = ent["card"]
    label = ent["label"]
    fid = fidelity_tier(card.get("fidelity", ""))
    fn, why = resolve_run(ent["run"])
    if fn is None:
        return job_meta(ctx, w, mid, PENDING, label, "registry", fid, reason=why, fingerprint="none",
                        cacheable=False), None
    ps = pixel_set_of(card)
    fp = code_fingerprint(fn, f"|{ps}|runner:{COMPETITOR_JOB_VERSION}")
    needs = list(card.get("needs") or [])
    t_sel = time.perf_counter()
    obs = window_obs(ctx.table, ctx, w, ps)
    note_window_select(ctx, w, time.perf_counter() - t_sel)
    cfg = method_cfg(ctx, w, mid, ps, needs)
    # every METHODS entry is a column (headline); the registry's VARIANTS are
    # declared sensitivities. The card's own 'role' text (e.g. "catalogue
    # variant ... reported descriptively") is kept and marked in the tables.
    role = "variant" if ent.get("variant") else "headline"
    variants = {"main": {"role": role, "label": label}}
    if card.get("role"):
        variants["main"]["catalogue_role"] = str(card["role"])
    info = {"pixel_set": ps, "card": card, "P": int(len(obs.keep)), "needs": needs}
    if "level_series" in cfg:
        info["level_series"] = cfg["level_series"].summary()
    t_start, t_est = time.perf_counter(), None
    try:
        res = fn(obs, cfg)                           # the estimator call (timed alone)
        t_est = time.perf_counter() - t_start
        st, z, sig, rmeta, reason = parse_method_result(res, len(obs.keep))
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ == "NotComputable":      # experiments.fixedcfg.elev_base.NotComputable
            return job_meta(ctx, w, mid, NOT_COMPUTABLE, label, "registry", fid, variants, str(e)[:500],
                            time.time() - t0, fp, info=info,
                            seconds_estimator=time.perf_counter() - t_start if t_est is None else t_est), None
        tb = traceback.format_exc(limit=6)
        log(f"[{ctx.site}] {w['label']} {mid}: ERROR {e!r}\n{tb}")
        return job_meta(ctx, w, mid, ERROR, label, "registry", fid, reason=repr(e)[:500], seconds=time.time() - t0,
                        fingerprint=fp, cacheable=False, info=dict(info, traceback=tb[-4000:])), None
    info["method_meta"] = {k: v for k, v in rmeta.items() if k != "extra"}
    # an adapter that moves the inputs to another process (DEA's side interpreter) reports that I/O and its
    # imports, or its own estimator seconds: neither the I/O nor the start-up is counted
    sec = rmeta.get("seconds") if isinstance(rmeta, dict) else None
    io_s = 0.0
    if isinstance(sec, dict):
        io_s = sum(float(sec.get(k) or 0.0) for k in ("write_input", "read_output"))
        io_s += float((sec.get("worker_detail") or {}).get("import") or 0.0)
    est_ad = _f(rmeta.get("seconds_estimator")) if isinstance(rmeta, dict) else float("nan")
    if np.isfinite(est_ad):
        rule = ("seconds_estimator = the estimator seconds the adapter reports (meta['seconds_estimator']: DEA's "
                "per-pixel steps and edge cleaning, without the side interpreter's start-up, imports and I/O)")
    else:
        rule = "seconds_estimator = the run(obs, cfg) call minus the I/O and imports the adapter reports"
    info["timing"] = {"call_s": round(float(t_est), 4), "adapter_io_s": round(io_s, 4),
                      "adapter_estimator_s": None if not np.isfinite(est_ad) else round(est_ad, 4),
                      "processes": (rmeta.get("max_workers") if isinstance(rmeta, dict) else None) or 1,
                      "rule": rule}
    t_est = est_ad if np.isfinite(est_ad) else max(0.0, float(t_est) - io_s)
    if st != OK:
        return job_meta(ctx, w, mid, st, label, "registry", fid, variants, reason, time.time() - t0, fp,
                        info=info, seconds_estimator=t_est), None
    arrays = {"z__main": to_units(z, ps, ctx)}
    if sig is not None:
        arrays["sigma__main"] = to_units(sig, ps, ctx)
    info["finite_share_of_P"] = float(np.isfinite(z).mean())
    return job_meta(ctx, w, mid, OK, label, "registry", fid, variants, "", time.time() - t0, fp, info=info,
                    seconds_estimator=t_est), arrays


def hook_params(fn):
    """The keyword names ``fn`` accepts, or None when it takes **kwargs."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def nidem_base_avail(ctx):
    """What the runner offers every NIDEM hook call besides the scenes and
    the pixel rule (per subset, or ``pixel_rule`` for a one-window hook)."""
    return {"site": ctx.site, "units_keep": np.asarray(ctx.units_keep, np.int64),
            "px_keep": np.asarray(ctx.px, np.int64),
            "truth": np.asarray(ctx.table.truth, float), "kind": ctx.table.kind,
            "inputs": dict(getattr(ctx, "hook_inputs", None) or {}), "io_lock": getattr(ctx, "io_lock", None),
            "cache_dir": getattr(ctx, "nidem_cache_dir", None), "soft_gates": bool(getattr(ctx, "soft", False))}


def window_subset(ctx, w, rule=None):
    """The scenes of window ``w`` for a hook: dates, extraction rows (when
    every scene is from the main extraction, else None), the zero-lag levels
    of the window table and, with ``rule``, the pixel rule to apply."""
    rows = window_rows(w)
    main = bool(np.all(np.asarray(ctx.S["src"])[rows] == 0))
    out = {"dates": [str(d) for d in np.asarray(ctx.table.S_dates)[rows]],
           "rows": np.asarray(ctx.S["row"], np.int64)[rows] if main else None,
           "h": np.array(np.asarray(ctx.table.levels[0], float)[rows], float)}
    if rule:
        out["pixel_rule"] = rule
    return out


def extra_scenes(ctx, w):
    """How many scenes of window ``w`` come from an extra store (no cube)."""
    return int(np.count_nonzero(np.asarray(ctx.S["src"])[window_rows(w)] != 0))


def applied_rule(res, offered_common=False):
    """The pixel rule a hook's answer says it applied ('item' or 'common'):
    its 'mask', else the first word of its 'pixel_rule'; a hook that says
    nothing but was given the common inputs (obs / C) applied the common
    mask; else None (unknown: never labelled)."""
    if isinstance(res, dict):
        m = str(res.get("mask") or "").strip().lower()
        if m in ("item", "common"):
            return m
        t = str(res.get("pixel_rule") or "").strip().lower()
        w = t.split()[0].strip(",;:()") if t else ""
        if w in ("item", "common"):
            return w
    return "common" if offered_common else None


def _batch_key(w, mid):
    """The hook's subset key of (window, NIDEM column): 'n040|common'."""
    return f"{w['label']}|{NIDEM_RULE[mid]}"


def _site_level(e):
    """An error of the hook that is the same for every window of the call (a
    refused input, a failed gate): retrying window by window cannot help."""
    return isinstance(e, SystemExit) or "gate" in type(e).__name__.lower()


def nidem_from_batch(ctx, w, fn, mid):
    """The hook's answer for (window ``w``, column ``mid``) from ONE call over
    every NIDEM job still to do (ctx.nidem_todo, (window, column) pairs; this
    one added when missing): one read of the cube for E1 (common mask) and
    E1i (ITEM's) of every window. A window holding extra-store scenes is not
    sent (the store has keep pixels only; NIDEM needs the full-frame cube):
    its answer is 'pending'. When the batched call fails for a reason that is
    not site-level, each job is retried alone. Returns a dict: 'answer' and
    'used' (keywords), or 'pending' (the reason), or '__error__'."""
    store = ctx.__dict__.setdefault("nidem_batch", {})
    key = (w["label"], mid)
    if key not in store:
        todo = [(x, m) for x, m in (getattr(ctx, "nidem_todo", None) or []) if (x["label"], m) not in store]
        if all((x["label"], m) != key for x, m in todo):
            todo = [(w, mid)] + todo
        send = []
        for x, m in todo:
            k = extra_scenes(ctx, x)
            if k:
                store[(x["label"], m)] = {"pending": (
                    f"NIDEM reads the full-frame cube of the main extraction; {k} of the window's {int(x['n'])} "
                    "scenes come from an extra store (keep pixels only, no cube): not sent to the hook"), "used": []}
            else:
                send.append((x, m))
        if send:
            _nidem_batch_call(ctx, fn, send, store)
    return store[key]


def _nidem_batch_call(ctx, fn, items, store):
    """One hook call for ``items`` [(window, column)], answers into ``store``;
    a failed call is retried job by job unless the failure is site-level."""
    def call(its):
        subsets = {_batch_key(x, m): window_subset(ctx, x, NIDEM_RULE[m]) for x, m in its}
        avail = dict(nidem_base_avail(ctx), subsets=subsets)
        res, used = call_accepting(fn, avail)
        missing = [k for k in subsets if not isinstance(res, dict) or k not in res]
        if missing:
            raise RuntimeError(f"the hook returned no answer for {missing}")
        return {(x["label"], m): res[_batch_key(x, m)] for x, m in its}, used

    names = [_batch_key(x, m) for x, m in items]
    log(f"[{ctx.site}] NIDEM hook: one call for {names}")
    t1 = time.perf_counter()
    try:
        answers, used = call(items)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        err, tb = e, traceback.format_exc(limit=8)
    else:
        for k, a in answers.items():
            store[k] = {"answer": a, "used": used}
        ctx.nidem_batch_info = {"windows": names, "wall_seconds": round(time.perf_counter() - t1, 2),
                                "keywords": used, "retried_one_by_one": False}
        return
    if _site_level(err) or len(items) == 1:
        for x, m in items:
            store[(x["label"], m)] = {"__error__": repr(err)[:500], "__traceback__": tb[-4000:]}
        log(f"[{ctx.site}] NIDEM hook: the call failed ({err!r}); every job of it is an error")
        return
    log(f"[{ctx.site}] NIDEM hook: the batched call failed ({err!r}); one call per job")
    for x, m in items:
        try:
            answers, used = call([(x, m)])
            store[(x["label"], m)] = {"answer": answers[(x["label"], m)], "used": used}
        except (Exception, SystemExit) as e:  # noqa: BLE001
            store[(x["label"], m)] = {"__error__": repr(e)[:500],
                                      "__traceback__": traceback.format_exc(limit=8)[-4000:]}
    ctx.nidem_batch_info = {"windows": names, "wall_seconds": round(time.perf_counter() - t1, 2),
                            "retried_one_by_one": True, "batch_error": repr(err)[:500]}


def nidem_one(ctx, w, rows, t0, mid="E1"):
    """NIDEM column ``mid`` (E1: common mask; E1i: ITEM's) through the v15
    hook on the window's scenes. A hook that takes ``subsets`` is called once
    for every NIDEM job to do (:func:`nidem_from_batch`); another one per
    window, with the keywords its signature accepts (``pixel_rule``; the
    common inputs ``obs`` / ``C`` / ``Y`` / ``frame`` are built only for a
    hook that takes them). The rule the hook says it applied must be the
    column's, else the job is an error (never mislabelled)."""
    fn, hinfo = ctx.nidem_fn, ctx.nidem_info
    want = NIDEM_RULE[mid]
    variants = NIDEM_VARIANTS[mid]
    label = variants["main"]["label"]
    fp = method_fingerprint(ctx, mid)
    src = f"v15 hook {hinfo.get('function')} (pixel rule {want})"
    params = hook_params(fn)
    timing_note = "the hook's own time of the subset (its cube read excluded)"
    offered_common = False
    try:
        if params is not None and "subsets" in params:
            got = nidem_from_batch(ctx, w, fn, mid)
            if "pending" in got:
                return job_meta(ctx, w, mid, PENDING, label, src, "F2", variants, got["pending"],
                                time.time() - t0, "none", cacheable=False), None
            if "__error__" in got:
                raise RuntimeError(f"the batched hook call failed: {got['__error__']}")
            res, used = got["answer"], got["used"]
            t_est = res.get("seconds_estimator") if isinstance(res, dict) else None
        else:
            if extra_scenes(ctx, w):
                return job_meta(ctx, w, mid, PENDING, label, src, "F2", variants,
                                f"{extra_scenes(ctx, w)} of the window's scenes come from an extra store (no cube)",
                                time.time() - t0, "none", cacheable=False), None
            if mid == "E1i" and params is not None and "pixel_rule" not in params:
                return job_meta(ctx, w, mid, PENDING, label, src, "F2", variants,
                                "the hook takes no pixel_rule: it cannot apply ITEM's mask on request",
                                time.time() - t0, "none", cacheable=False), None
            sub = window_subset(ctx, w)
            avail = dict(nidem_base_avail(ctx), dates=sub["dates"], rows=sub["rows"], h=sub["h"],
                         t_real=ctx.table.S_t[window_slice(w)], n=int(w["n"]), pixel_rule=want,
                         out_dir=os.path.join(ctx.paths["work"], w["label"], mid))
            if params is None or {"obs", "frame", "C", "Y"} & params:
                obs = window_obs(ctx.table, ctx, w, "keep")
                avail.update(obs=obs, frame=obs.frame, C=obs.C, Y=obs.Y)
            t1 = time.perf_counter()
            res, used = call_accepting(fn, avail)
            t_est = time.perf_counter() - t1
            offered_common = bool({"obs", "C"} & set(used))
            if isinstance(res, dict) and res.get("seconds_estimator") is not None:
                t_est = float(res["seconds_estimator"])
            else:
                timing_note = "the hook call (whatever it reads included)"
        got_rule = applied_rule(res, offered_common)
        info = {"keywords": used, "timing": timing_note, "pixel_rule_asked": want, "pixel_rule_applied": got_rule,
                "hook_meta": {k: v for k, v in (res.items() if isinstance(res, dict) else [])
                              if not isinstance(v, np.ndarray)}}
        if got_rule != want:
            return job_meta(ctx, w, mid, ERROR, label, src, "F2", variants,
                            f"the hook applied pixel rule {got_rule!r}; {mid} needs {want!r}", time.time() - t0, fp,
                            cacheable=False, info=info, seconds_estimator=t_est), None
        info["mask"] = NIDEM_RULE_NOTE[want] + ("; the common inputs obs / C were passed" if offered_common else "")
        if isinstance(res, dict) and str(res.get("status", "ok")).strip().upper().startswith("NOT"):
            return job_meta(ctx, w, mid, NOT_COMPUTABLE, label, src, "F2", variants, str(res.get("reason", "")),
                            time.time() - t0, fp, info=info, seconds_estimator=t_est), None
        z = hook_values(res, len(ctx.unit_col), ctx.unit_col, ctx.units_keep, len(ctx.px), len(ctx.keep_flat))
    except (Exception, SystemExit) as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        log(f"[{ctx.site}] {w['label']} {mid} NIDEM hook: ERROR {e!r}")
        return job_meta(ctx, w, mid, ERROR, label, src, "F2", variants, repr(e)[:500], time.time() - t0, fp,
                        cacheable=False, info={"traceback": tb[-4000:]}), None
    if isinstance(res, dict) and isinstance(res.get("gate"), dict):
        ctx.__dict__.setdefault("nidem_gate", res["gate"])
    if not np.isfinite(z).any():
        return job_meta(ctx, w, mid, NOT_COMPUTABLE, label, src, "F2", variants, "no finite estimate on the units",
                        time.time() - t0, fp, info=info, seconds_estimator=t_est), None
    return (job_meta(ctx, w, mid, OK, label, src, "F2", variants, "", time.time() - t0, fp, info=info,
                     seconds_estimator=t_est), {"z__main": z})


def method_fingerprint(ctx, mid):
    """The fingerprint a cached job must carry to be reused (None = never
    cached: not built / pending). NIDEM's carries its pixel rule, so an ITEM
    result is never resumed as E1."""
    if mid in ("E0b", "E0a"):
        return f"v14:{V14_SHA256[:16]}|job:{JOB_VERSION}"
    if mid in NIDEM_RULE and ctx.nidem_fn is not None:
        return (f"{ctx.nidem_info.get('function')}:{(ctx.nidem_info.get('file') or {}).get('sha256', '?')[:16]}"
                f"|rule:{NIDEM_RULE[mid]}")
    ent = ctx.registry.get(mid)
    if ent is None:
        return None
    fn, _ = resolve_run(ent["run"])
    if fn is None:
        return None
    return code_fingerprint(fn, f"|{pixel_set_of(ent['card'])}|runner:{COMPETITOR_JOB_VERSION}")


def job_plan(windows, methods, record_window="all"):
    """(window label, method) in run order: MAREA on the record window first
    (the gate), then method-major (MAREA on every window, no delay, each
    competitor, NIDEM), each over the windows from the largest n down. The
    record window is computed for every method unless ``record_window`` is
    'marea' (then only MAREA and no delay); a gate-only record window (it
    ends after the truth date) only for MAREA and no delay."""
    rw = [w for w in windows if w["reachable"]]
    rec = [w for w in rw if w["is_record"]]
    others = sorted((w for w in rw if not w["is_record"] and not w.get("gate_only")), key=lambda w: -w["n"])
    plan = [(rec[0]["label"], "E0b")] if rec else []
    for mid in methods:
        ws = list(others)
        if rec:
            r0 = rec[0]
            if r0.get("gate_only"):
                if mid in RUNNER_IDS:
                    ws = [r0] + ws
            elif mid in RUNNER_IDS or record_window == "all" or r0["requested"]:
                ws = [r0] + ws
        for w in ws:
            if (w["label"], mid) not in plan:
                plan.append((w["label"], mid))
    return plan


# ─────────────────────────────────────────────────────────────────────────────
#  scoring and the per-site CSV (rebuilt from the job store)
# ─────────────────────────────────────────────────────────────────────────────

def score_values(kind, truth, z, unit_col=None):
    """v14.score (median-centred RMSE, OLS slope, r, bias = median(product -
    truth), coverage = finite share of the units) plus the total
    (uncentred) RMSE and the distinct pixels scored."""
    truth = np.asarray(truth, float)
    z = np.asarray(z, float)
    s = v14.score(kind, truth, z)
    fin = np.isfinite(z) & np.isfinite(truth)
    min_n = 20 if kind == "rtk" else 30
    s["rmse_total"] = float(np.sqrt(np.mean((z[fin] - truth[fin]) ** 2))) if fin.sum() >= min_n else float("nan")
    s["n_px"] = int(len(np.unique(np.asarray(unit_col)[fin]))) if unit_col is not None else int(fin.sum())
    return s


def _window_cols(site, w):
    return {"site": site, "n_scenes": int(w["n"]), "is_record": int(bool(w.get("is_record"))),
            "first_date": w.get("first_date", ""), "last_date": w.get("last_date", ""),
            "lower_bound_open": w.get("lower_bound_open") or "", "years": years_text(w.get("years", {})),
            "scenes_sha1": w.get("scenes_sha1", ""), "lo_m": w.get("lo_m", float("nan")),
            "hi_m": w.get("hi_m", float("nan"))}


def _blank(**kw):
    r = {c: "" for c in CSV_COLUMNS}
    for c in ("n_set", "n_common", "n_units", "n_px"):
        r[c] = 0
    for c in ("coverage", "rmse", "rmse_total", "bias", "slope", "r", "median_clear_obs", "seconds",
              "seconds_estimator", "seconds_shared"):
        r[c] = float("nan")
    r.update(kw)
    return r


def shared_seconds(timing, w):
    """The shared preprocessing seconds of (site, n): the site's (load,
    screening, window table; truth loading and gates excluded) plus the
    window's selection; NaN without a timing record."""
    if not timing or timing.get("site_shared_s") is None:
        return float("nan")
    return float(timing["site_shared_s"]) + float((timing.get("window_select_s") or {}).get(w.get("label"), 0.0))


def _job_timing(m):
    """The timing columns of a job's rows."""
    se = m.get("seconds_estimator")
    return {"seconds_estimator": float("nan") if se is None else float(se), "host": m.get("host", ""),
            "threads": m.get("threads", "")}


def _notes(mid, meta):
    info = meta.get("info") or {}
    if mid == "E0b" and info.get("tau_used_bands_min") is not None:
        return "tau bands " + "/".join(f"{v:.1f}" for v in info["tau_used_bands_min"]) + " min"
    if mid in NIDEM_RULE and info.get("mask"):
        return str(info["mask"])
    out = []
    if info.get("pixel_set"):
        out.append(f"pixels: {info['pixel_set']}")
    v0 = next(iter((meta.get("variants") or {}).values()), {})
    cr = v0.get("catalogue_role") or (v0.get("role") if v0.get("role") not in ROLES else None)
    if cr:
        out.append(f"catalogue role: {cr}")
    return "; ".join(out)


def descriptive_ids(df):
    """Methods whose card calls them a catalogue variant (reported
    descriptively): marked with a dagger in the tables."""
    d = df[df["notes"].str.contains("catalogue role:", na=False)]
    return sorted({m for m, n in zip(d["method"], d["notes"]) if "variant" in n.split("catalogue role:")[-1]})


def _members(jobs, store, n):
    """(method, variant, variant info, z per unit, job meta) of every computed
    job of a window whose arrays hold ``n`` units."""
    out = []
    for mid, m in jobs.items():
        if m["status"] != OK:
            continue
        arr = store.arrays(m["key"]) or {}
        for var, vinfo in (m.get("variants") or {}).items():
            z = arr.get(f"z__{var}")
            if z is None or z.shape != (n,):
                continue
            vinfo = dict(vinfo)
            if vinfo.get("role") not in ROLES:          # a card's descriptive text: a column
                vinfo.setdefault("catalogue_role", vinfo.get("role"))
                vinfo["role"] = "headline"
            out.append((mid, var, vinfo, np.asarray(z, float), m))
    return out


def _scored_rows(wc, kind, truth, unit_col, members, nonok, have, labels, unavailable, note_all=""):
    """The rows of one tabulated window (or a pooled row) on its units
    (``truth`` / ``unit_col`` / the members' z, already restricted to them):
    the common set (every headline method with a value) and each member's
    scores on it and on its own units; a row per job that is not computed
    (``nonok`` = [(method, job meta)]); per catalogue method without a job
    (not in ``have``) 'pending' (``unavailable``) or 'not implemented'."""
    rows = []
    # a headline method with no value on F at all cannot be scored; it
    # is left out of the intersection (and says so) instead of emptying it
    head = [x for x in members if x[2].get("role") == "headline" and np.isfinite(x[3]).any()]
    empty = {(x[0], x[1]) for x in members if x[2].get("role") == "headline" and not np.isfinite(x[3]).any()}
    common = np.isfinite(truth)
    for x in head:
        common &= np.isfinite(x[3])
    n_common = int(common.sum())
    members_in = ";".join(f"{x[0]}:{x[1]}" for x in head)
    for mid, var, vinfo, z, m in members:
        note = _notes(mid, m)
        if (mid, var) in empty:
            note = (note + "; " if note else "") + "no value on F: left out of the common set"
        if note_all:
            note = (note + "; " if note else "") + note_all
        base = dict(wc, method=mid, label=vinfo.get("label", m.get("label", mid)), variant=var,
                    role=vinfo.get("role", ""), status=OK, n_set=int(len(truth)), n_common=n_common,
                    seconds=m.get("seconds", float("nan")), source=m.get("source", ""),
                    fidelity=m.get("fidelity", ""), notes=note, reason=f"common set of {members_in}",
                    **_job_timing(m))
        mco = (m.get("info") or {}).get("median_clear_obs")
        for scoring, sel in (("common", common), ("own", None)):
            if sel is None:
                s = score_values(kind, truth, z, unit_col)
            else:
                s = score_values(kind, truth[sel], z[sel], unit_col[sel])
            rows.append(_blank(**base, scoring=scoring, n_units=int(s["n"]), n_px=int(s["n_px"]),
                               coverage=s["coverage"], rmse=s["rmse"], rmse_total=s["rmse_total"],
                               bias=s["bias"], slope=s["slope"], r=s["r"],
                               median_clear_obs=mco if mco is not None else float("nan")))
    for mid, m in nonok:
        rows.append(_blank(**wc, method=mid, label=m.get("label", mid), status=m["status"],
                           n_set=int(len(truth)), n_common=n_common, seconds=m.get("seconds", float("nan")),
                           source=m.get("source", ""), fidelity=m.get("fidelity", ""), reason=m.get("reason", ""),
                           role=next(iter((m.get("variants") or {}).values()), {}).get("role", "headline"),
                           notes=note_all, **_job_timing(m)))
    for mid in CATALOGUE_IDS:
        if mid not in have:
            why = (unavailable or {}).get(mid)
            rows.append(_blank(**wc, method=mid, label=labels.get(mid, mid),
                               status=PENDING if why else NOT_IMPLEMENTED,
                               role="headline", n_set=int(len(truth)), n_common=n_common,
                               reason=why or "no code under the common configuration yet", notes=note_all))
    return rows


def window_jobs(metas, w):
    """{method: job meta} of window ``w``: its n, scenes and label."""
    return {m["method"]: m for m in metas.values()
            if int(m["n"]) == int(w["n"]) and m.get("scenes_sha1") == w["scenes_sha1"]
            and not m.get("gate_only") and m.get("window", w["label"]) == w["label"]}


def build_rows(site, kind, truth, unit_col, windows, store, known=(), timing=None, unavailable=None,
               unit_group=None):
    """Every row of the per-site CSV from the job store: per tabulated
    window the common set (units where every headline method computed at
    that n has a value) and each method's scores on it and on its own
    units; rows for pending / not computable / error jobs and for the
    catalogue methods with no job: 'pending' when the method's code exists
    but was not run (``unavailable`` = {id: why}, e.g. DEA without its side
    interpreter on that host, :func:`unavailable_methods`), else 'not
    implemented'; per unreachable or gate-only window one 'not reachable'
    row per method with its reason. Every computed row carries the estimator
    seconds of its job, the shared seconds of (site, n) (``timing``: the
    <site>_timing.json record), the host and the threads.

    Survey-year groups (``unit_group``: the survey year of every unit;
    windows tagged by :func:`group_windows`): a group's window is scored on
    the group's units only (the same rules, its own common set), its rows
    under the group's row ('wadden-2024'); then :func:`pooled_rows`."""
    truth = np.asarray(truth, float)
    unit_col = np.asarray(unit_col, np.int64)
    metas = {k: store.meta(k) for k in store.keys()}
    metas = {k: m for k, m in metas.items() if m}
    seen = {m["method"]: m.get("label", m["method"]) for m in metas.values()}
    seen.update({k: v for k, v in known})
    labels = dict(CATALOGUE)
    labels.update({"E0a": "no delay"})
    labels.update(seen)
    method_ids = list(dict.fromkeys(["E0b", "E0a", *CATALOGUE_IDS, *sorted(seen)]))
    grouped = unit_group is not None and any(w.get("group") is not None for w in windows)
    if grouped:
        unit_group = np.asarray(unit_group, np.int64)
        if unit_group.shape != truth.shape:
            raise ValueError(f"unit_group has {unit_group.shape} values for {truth.shape} units")
    rows, done_w = [], {}
    for w in sorted(windows, key=lambda w: (-int(w["n"]), -int(w.get("group") or 0), bool(w.get("gate_only")))):
        wc = _window_cols(w.get("row_site") or site, w)
        if not w.get("reachable") or w.get("gate_only"):
            for mid in method_ids:
                rows.append(_blank(**wc, method=mid, label=labels.get(mid, mid), status=NOT_REACHABLE,
                                   reason=w.get("reason", ""),
                                   notes=GATE_ONLY_NOTE if w.get("gate_only") else ""))
            continue
        wc["seconds_shared"] = shared_seconds(timing, w)
        jobs = window_jobs(metas, w)
        members = _members(jobs, store, len(truth))
        nonok = [(mid, m) for mid, m in jobs.items() if m["status"] != OK]
        sel = (unit_group == int(w["group"])) if grouped and w.get("group") is not None else None
        if grouped:
            done_w[w["label"]] = (w, jobs, members)
        if sel is None:
            rows += _scored_rows(wc, kind, truth, unit_col, members, nonok, set(jobs), labels, unavailable)
        else:
            rows += _scored_rows(wc, kind, truth[sel], unit_col[sel],
                                 [(a, b, c, z[sel], m) for a, b, c, z, m in members], nonok, set(jobs), labels,
                                 unavailable)
    if grouped:
        rows += pooled_rows(site, kind, truth, unit_col, unit_group, windows, done_w, labels, unavailable, timing)
    return rows


def pooled_rows(site, kind, truth, unit_col, unit_group, windows, done_w, labels, unavailable=None, timing=None):
    """The pooled row of a site with survey-year groups (CSV 'site' = the
    site, e.g. 'wadden'), only at the n where EVERY group's window is
    reachable and tabulated: each unit scored with the estimate of its own
    group's window (every unit of F belongs to one group); the common set
    and the F rules as in any row. A method computed in every group is
    pooled (estimator seconds: the sum of the groups'); otherwise its row
    takes the status of the first group where it is not computed."""
    groups = sorted({int(w["group"]) for w in windows if w.get("group") is not None}, reverse=True)
    names = {int(w["group"]): w.get("row_site") or group_row(site, w["group"]) for w in windows
             if w.get("group") is not None}
    # a group without any unit of F has no pixel to estimate: it never holds the pooled row back
    groups = [g for g in groups if np.any(unit_group == g)]
    if not groups:
        return []
    note_all = ("pooled: each pixel scored with its own survey-year group's estimate ("
                + ", ".join(names[g] for g in groups) + "); seconds = the sum of the groups'")
    sels = {g: unit_group == g for g in groups}
    rows = []
    for n in sorted({int(w["n"]) for w in windows if w.get("requested")}, reverse=True):
        ws = {g: next((w for w in windows if w.get("group") == g and int(w["n"]) == n and w.get("requested")), None)
              for g in groups}
        if not all(w is not None and w.get("reachable") and not w.get("gate_only") and w["label"] in done_w
                   for w in ws.values()):
            continue
        gw = [ws[g] for g in groups]
        wc = _window_cols(site, {"n": n, "is_record": False, "first_date": min(w["first_date"] for w in gw),
                                 "last_date": max(w["last_date"] for w in gw), "lower_bound_open": "", "years": {},
                                 "scenes_sha1": hashlib.sha1("|".join(w["scenes_sha1"] for w in gw).encode())
                                 .hexdigest()[:12], "lo_m": min(w["lo_m"] for w in gw),
                                 "hi_m": max(w["hi_m"] for w in gw)})
        shared = [shared_seconds(timing, w) for w in gw]
        site_s = float((timing or {}).get("site_shared_s") or 0.0)
        wc["seconds_shared"] = ((site_s + sum(s - site_s for s in shared)) if all(np.isfinite(shared))
                                else float("nan"))
        mem = {g: {(a, b): (c, z, m) for a, b, c, z, m in done_w[ws[g]["label"]][2]} for g in groups}
        jobs = {g: done_w[ws[g]["label"]][1] for g in groups}
        members, nonok, have = [], [], set()
        keys = list(dict.fromkeys(k for g in groups for k in mem[g]))
        for mid, var in keys:
            if not all((mid, var) in mem[g] for g in groups):
                continue
            z = np.full(len(truth), np.nan)
            for g in groups:
                z[sels[g]] = mem[g][(mid, var)][1][sels[g]]
            ms = [mem[g][(mid, var)][2] for g in groups]
            est = [m.get("seconds_estimator") for m in ms]
            m = {"method": mid, "label": ms[0].get("label", mid), "status": OK, "variants": ms[0].get("variants"),
                 "source": ms[0].get("source", ""), "fidelity": ms[0].get("fidelity", ""),
                 "seconds": float(sum(float(x.get("seconds") or 0.0) for x in ms)),
                 "seconds_estimator": None if any(v is None for v in est) else float(sum(est)),
                 "host": _uniq(x.get("host", "") for x in ms), "threads": _uniq(x.get("threads", "") for x in ms),
                 "info": {}}
            members.append((mid, var, mem[groups[0]][(mid, var)][0], z, m))
            have.add(mid)
        for mid in dict.fromkeys(k for g in groups for k in jobs[g]):
            if mid in have:
                continue
            have.add(mid)
            for g in groups:
                m = jobs[g].get(mid)
                if m is None or m["status"] != OK:
                    st = dict(m or {"method": mid, "label": labels.get(mid, mid), "status": PENDING,
                                    "reason": "not computed yet", "variants": {}})
                    st["reason"] = f"{names[g]}: {st.get('reason', '')}"
                    nonok.append((mid, st))
                    break
        rows += _scored_rows(wc, kind, truth, unit_col, members, nonok, have, labels, unavailable, note_all)
    return rows


def unavailable_methods(out, site):
    """{method id: why} of the registry methods whose code exists but did not
    run at ``site`` (read from every <site>_settings*.json): skipped by the
    registry on that host (e.g. DEA Intertidal without its side interpreter),
    an adapter that failed to import, or left out by --methods. Such a method
    is 'pending' in the tables, never 'not implemented'."""
    why = {}
    for p in sorted(glob.glob(os.path.join(out, f"{site}_settings*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, ValueError):
            continue
        m = st.get("methods") or {}
        reg = m.get("registry") or {}
        host = st.get("host", "?")
        for k, v in (reg.get("registry_skipped") or {}).items():
            why.setdefault(str(k), f"not run on {host}: {v}"[:300])
        errs = reg.get("registry_errors") or {}
        if "dea_intertidal" in errs:
            why.setdefault("E2", f"adapter error on {host}: {errs['dea_intertidal']}"[:300])
        order = set(m.get("order") or [])
        for k in reg.get("methods") or []:
            if order and k not in order:
                why.setdefault(str(k), "not run (left out by --methods)")
    return why


def write_csv(path, rows):
    import io

    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(CSV_COLUMNS)
    for r in rows:
        wr.writerow([v14._fmt(r.get(c, "")) for c in CSV_COLUMNS])
    atomic_text(path, buf.getvalue())


def save_units(path, units_keep, truth, unit_col, px, kind, extra=None):
    arrays = {"units_keep": np.asarray(units_keep, np.int64), "truth": np.asarray(truth, float),
              "unit_col": np.asarray(unit_col, np.int64), "px": np.asarray(px, np.int64),
              "kind": np.asarray(str(kind))}
    for k, v in (extra or {}).items():
        arrays[k] = np.asarray(v)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_units(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: np.asarray(f[k]) for k in f.files}


def rescore_site(out, site, store=None):
    """Rebuild <site>.csv from <site>_units.npz, <site>_windows.json and the
    job store (``store``: a running JobStore, whose array cache saves
    re-reading every job). Returns the rows (None when the site has no run
    yet)."""
    p = site_paths(out, site)
    if not (os.path.exists(p["units"]) and os.path.exists(p["windows"])):
        return None
    u = load_units(p["units"])
    if str(u.get("truth_design", "")) != TRUTH_DESIGN:
        log(f"{site}: {p['units']} was written before the {TRUTH_DESIGN} (another truth): not rescored")
        return None
    with open(p["windows"], encoding="utf-8") as f:
        wj = json.load(f)
    timing = read_timing(out, site)
    store = store if store is not None else JobStore(p["jobs"], create=False)
    rows = build_rows(site, str(u["kind"]), u["truth"], u["unit_col"], wj["windows"], store, timing=timing,
                      unavailable=unavailable_methods(out, site), unit_group=u.get("unit_group"))
    write_csv(p["csv"], rows)
    return rows


def read_timing(out, site):
    """The shared preprocessing times of a site from its <site>_timing*.json
    (one per --tag group, the same preprocessing): the fastest site share and,
    per window, the fastest selection; None without a file."""
    docs = []
    for path in sorted(glob.glob(os.path.join(out, f"{site}_timing*.json"))):
        if ".tmp" in os.path.basename(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                docs.append(json.load(f))
        except (OSError, ValueError):
            continue
    if not docs:
        return None
    shared = [float(d["site_shared_s"]) for d in docs if d.get("site_shared_s") is not None]
    sel = {}
    for d in docs:
        for k, v in (d.get("window_select_s") or {}).items():
            if v is not None:
                sel[k] = min(float(v), sel.get(k, float("inf")))
    out_doc = dict(docs[0])
    out_doc.update(site_shared_s=min(shared) if shared else None, window_select_s=sel, files=len(docs))
    return out_doc


def write_timing(path, ctx, site_parts):
    """<site>_timing.json: host, threads and the shared preprocessing times
    (the site's, by part, and each window's selection)."""
    hi = host_info()
    doc = {"site": ctx.site, **hi, "loadavg": load_avg(), "site_shared_s": float(sum(site_parts.values())),
           "site_shared_parts_s": {k: round(float(v), 3) for k, v in site_parts.items()},
           "window_select_s": {k: round(float(v), 6) for k, v in (getattr(ctx, "window_select_s", None) or {}).items()},
           "rule": ("shared = the site's load (v14.load_site without the truth loading: extraction streams, screening "
                    "of the pool, levels), the screened list and windows, the window table (and the 1-min level "
                    "series when a method needs it), plus the window's selection (its common inputs); the gates "
                    "are not counted"),
           "updated": _dt.datetime.now().isoformat(timespec="seconds")}
    write_json(path, doc)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
#  gates
# ─────────────────────────────────────────────────────────────────────────────

def record_gate(site, kind, truth, unit_col, z0, z_stored, tau_bands=None, tau_stored=None, published=None,
                full_F=False, published_note=None):
    """The record window, MAREA with refitted lags and mb 0, reproduces the
    stored MAREA on the units (same valid set up to v14's flip tolerance,
    |dz| <= TOL_Z, the same scores) and, where the units are the whole F and
    a published point on the same truth is given, that point (n, RMSE,
    slope, r, bias). ``published_note``: why no published point applies."""
    z0, zs = np.asarray(z0, float), np.asarray(z_stored, float)
    fa, fb = np.isfinite(z0), np.isfinite(zs)
    flips = int(np.sum(fa != fb))
    both = fa & fb
    dz = float(np.max(np.abs(z0[both] - zs[both]))) if both.any() else float("nan")
    sm = score_values(kind, truth, z0, unit_col)
    ss = score_values(kind, truth, zs, unit_col)
    dsc = max(abs(sm[k] - ss[k]) for k in ("rmse", "slope", "r", "bias"))
    ok = (flips <= v14.TOL_FLIP_SHARE * len(z0)) and (dz <= v14.TOL_Z) and (dsc <= v14.TOL_SCORE or flips > 0)
    g = {"units": int(len(z0)), "validity_flips": flips, "max_abs_dz_m": dz,
         "scores_mine": {k: sm[k] for k in ("n", "rmse", "slope", "r", "bias", "coverage")},
         "scores_stored": {k: ss[k] for k in ("n", "rmse", "slope", "r", "bias", "coverage")},
         "max_abs_score_diff": dsc}
    if tau_bands is not None and tau_stored is not None and len(tau_stored) == len(tau_bands):
        g["tau_bands_vs_result_json_max_abs_diff_min"] = float(np.max(np.abs(np.asarray(tau_bands, float)
                                                                            - np.asarray(tau_stored, float))))
    if full_F and published is not None:
        dp = max(abs(sm[k] - published[c]) for k, c in (("rmse", "rmse_m"), ("slope", "slope"), ("r", "r"),
                                                        ("bias", "bias_m")))
        g["published_row"] = published
        g["max_abs_diff_vs_published"] = dp
        g["n_equals_published"] = int(sm["n"]) == int(published["n"])
        ok = ok and dp <= v14.TOL_SCORE and g["n_equals_published"]
    elif full_F:
        g["published_row"] = None
        g["note"] = published_note or "no published CSV found: compared with the stored product only"
    else:
        g["note"] = ("the units are a seeded subsample of F: compared with the stored product on them (v14's gate "
                     "ties the stored product to the published row on the whole F)")
    g["ok"] = bool(ok)
    return g


def find_v14_settings(site, n_units, dirs=V14_SETTINGS_DIRS):
    """The first v14 settings JSON of the site whose compat block holds the
    same number of scored units, or None."""
    for d in dirs:
        for lags in ("refit", "fixed"):
            p = os.path.join(d, f"{site}_{lags}_settings.json")
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    c = json.load(f).get("compat") or {}
            except ValueError:
                continue
            if int(c.get("n_units_scored", -1)) == int(n_units):
                return p, c
    return None, None


def fixed_set_gate(site, mine, truth=None):
    """F against a v14 settings JSON with the same number of units. Not
    applicable under this design's truths (F differs from v14's by design:
    the 2025 cells only / the PNOA raster), recorded as such."""
    if truth is not None and truth.get("design") == TRUTH_DESIGN:
        return {"checked": False, "ok": True, "stop_if_failed": False, "applicable": False,
                "reason": (f"F is built on this design's truth ({truth.get('label_en')}), not on v14's: no v14 "
                           "settings JSON can match it")}
    p, c = find_v14_settings(site, mine["n_units_scored"])
    if p is None:
        return {"checked": False, "ok": True, "stop_if_failed": False,
                "reason": f"no v14 settings JSON with {mine['n_units_scored']:,} scored units in {V14_SETTINGS_DIRS}"}
    keys = ("seed", "n_units_scored", "n_units_F", "units_sha1", "z_stored_sha1", "tau_fixed_sha1")
    diff = {k: {"v19": mine.get(k), "v14": c.get(k)} for k in keys if mine.get(k) != c.get(k)}
    return {"checked": True, "reference": rel(p), "keys": list(keys), "differing": diff, "ok": not diff}


def stop_failed(gates):
    return [k for k, v in gates.items() if isinstance(v, dict) and v.get("ok") is False
            and v.get("stop_if_failed", True)]


# ─────────────────────────────────────────────────────────────────────────────
#  run
# ─────────────────────────────────────────────────────────────────────────────

def site_cfg(args):
    cfg = v14.site_config(args.site)
    overrides = {}
    for k in ("extract", "product", "comparison", "nidem", "cube"):
        v = getattr(args, f"{k}_override", None)
        if v:
            cfg[k] = v
            overrides[k] = v
    if getattr(args, "published_csv", None):
        cfg["published"] = [args.published_csv]
        overrides["published_csv"] = args.published_csv
    from experiments import v7_waterline_contour as v7
    cfg["overpass"] = v7.SITES[args.site]["overpass"]
    return cfg, overrides


def windows_doc(site, S, s_info, windows, extra_info, provenance, truth=None):
    T = len(S["dates"])
    ws = []
    for w in windows:
        d = dict(w)
        if w["reachable"]:
            sl = window_slice(w)
            d["dates"] = [str(x) for x in S["dates"][sl]]
            d["sources"] = {str(k): int(v) for k, v in zip(*np.unique(np.asarray(S["src"])[sl], return_counts=True))}
            d["extraction_rows_main"] = [int(r) for r, s in zip(S["row"][sl], S["src"][sl]) if int(s) == 0]
            if provenance is not None:
                pa = provenance["all_members"][sl]
                d["levels_all_members"] = int(pa.sum())
                d["levels_partial"] = int((~pa).sum())
        ws.append(d)
    return {"site": site, "purpose": ("the screened list S and the windows (the n most recent screened scenes on or "
                                      "before the truth date) of the fixed-configuration tables"),
            "window_end": None if truth is None else dict(
                {"date": truth.get("end"), "rule": truth.get("end_rule")},
                **({"groups": [{"name": g["name"], "year": g["year"], "date": g["end"]} for g in truth["groups"]]}
                   if truth.get("groups") else {})),
            "S": {"n": int(T), "first": str(S["dates"][0]) if T else None, "last": str(S["dates"][-1]) if T else None,
                  "per_year": per_year(S["dates"]), "dates": [str(x) for x in S["dates"]],
                  "source": [int(s) for s in S["src"]], "row": [int(r) for r in S["row"]],
                  "t": [str(x) for x in S["t"]], "h0_m": [float(x) for x in S["h0"]],
                  "sha1": scenes_sha1(S["dates"]), "rule": s_info.get("rule")},
            "extra_stores": extra_info, "windows": ws}


def write_pending(args, site, out, paths, tdef, ns):
    """Villaviciosa without its truth: settings and a CSV whose rows say
    'pending truth' (one per n); nothing is loaded, the RTK is never used.
    Refuses an --out that holds a run of the site with a truth date."""
    if os.path.exists(paths["settings"]):
        with open(paths["settings"], encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("status") != "pending_truth":
            raise SystemExit(f"{paths['settings']} holds a run of {site} with its truth ({prev.get('status')}): "
                             "give --vva-truth and --vva-truth-date, or use another --out")
    rows = [_blank(site=site, n_scenes=int(n), is_record=0, method="E0b", label="MAREA", status=PENDING_TRUTH,
                   role="", reason=tdef["reason"]) for n in ns]
    write_csv(paths["csv"], rows)
    settings = {"site": site, "site_name": SITE_NAMES.get(site, site), "status": "pending_truth", "truth": tdef,
                "ns": ns, "git_head": v14.git_head(), "script": v14.file_info(rel(__file__)), **host_info(),
                "runs": [{"at": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
                          "status": "pending_truth"}]}
    v14.write_json(paths["settings"], settings)
    log(f"[{site}] PENDING TRUTH: {tdef['reason']} -> {paths['csv']}, {paths['settings']}")
    return settings


def prepare_truth(site, cfg, tdef, args, gates):
    """The unit function v14.load_site will call under this design: (name of
    the v14 function it replaces, always 'rtk_units', whose branch takes the
    units as given; the replacement; a holder that receives what the truth
    leaves behind: cell ids, seconds, the stored z it saw). F is decided by
    no product (:data:`F_RULE`): :func:`dutch_units` at the Dutch sites (the
    range filter reads the production comparison's z_plain), the raster's
    :func:`raster_units` at Villaviciosa, sampled here, before load_site (its
    gates go into ``gates``)."""
    holder = {"seconds": 0.0}
    if site != "villaviciosa":
        with np.load(cfg["extract"], allow_pickle=False) as ex:
            H = int(np.asarray(ex["shape"]).ravel()[0])
        cp = os.path.join(cfg.get("comparison") or "", "products.npz")
        if not cfg.get("comparison") or not os.path.exists(cp):
            raise SystemExit(f"[{site}] the range filter needs the comparison products.npz z_plain ({cp})")
        with np.load(cp, allow_pickle=False) as P:
            z_plain = np.asarray(P["z_plain"], float)
        # the survey-year groups (the Wadden): every year of the groups, the most recent survey per pixel
        year = tuple(int(g["year"]) for g in tdef["groups"]) if tdef.get("groups") else VAK_TRUTH_YEAR
        return "rtk_units", dutch_units(site, cfg, H, z_plain, year, getattr(args, "vak_dir", None), holder), holder
    from experiments import v7_waterline_contour as v7

    with np.load(cfg["extract"], allow_pickle=False) as ex:
        keep = np.asarray(ex["keep"], np.int64)
        H, W = (int(v) for v in ex["shape"])
    mp = os.path.join(cfg["product"], "marea.npz")
    with np.load(mp, allow_pickle=False) as z:
        z_pre = np.asarray(z["z"], float)
    lo, hi, g = vva_nodelay_range(tdef["nodelay"], keep, z_pre)
    gates["vva_nodelay_is_the_stored_production"] = g
    truth_raw, rinfo, secs = vva_raster_truth(tdef["path"], keep, H, W, cube=v7.SITES[site]["cube"],
                                              grid_path=getattr(args, "vva_grid", None))
    holder.update(seconds=secs, z_pre=z_pre)
    gates["vva_raster_truth_on_the_extraction_grid"] = {
        "grid": rinfo["grid"], "shape_equal": True, "cube_transform_equal": rinfo["grid_is_the_cube_grid"],
        "ok": rinfo["grid_is_the_cube_grid"] is not False}
    log(f"[{site}] PNOA truth {tdef['path']}: {int(np.isfinite(truth_raw).sum()):,} of {len(keep):,} keep px, "
        f"water patches {100 * rinfo['water_patch_share_2m']:.1f} % of the 2 m area [{secs:.1f} s]")
    return "rtk_units", raster_units(truth_raw, lo, hi, dict(rinfo, nodelay=tdef["nodelay"]), holder), holder


def truth_counts(site, tdef, holder, extras, meta):
    """The truth actually used, per site: sheets and 20 m cells under F and
    under the scored units (Dutch), pixels (all)."""
    fs = meta.get("fixed_set") or {}
    out = {"truth_px_at_keep": fs.get("truth_px"), "truth_px_in_range": fs.get("truth_in_range"),
           "units_F": int(len(extras["unit_keep_full"])), "units_scored": int(len(extras["units_keep"])),
           "range_filter_m": fs.get("range_filter_m")}
    cid = holder.get("cell_id")
    if cid is not None:
        def cells(idx):
            c = np.asarray(cid)[np.asarray(idx, np.int64)]
            c = c[c >= 0]
            sheets = sorted({int(v >> 40) for v in np.unique(c)})
            names = list((holder.get("sheets") or {}).get("sheets") or [])
            used = [r["sheet"] for r in names if r.get("used")]
            return {"cells": int(len(np.unique(c))), "sheets": [used[k] for k in sheets if k < len(used)]}
        f, s = cells(extras["unit_keep_full"]), cells(extras["units_keep"])
        sh = holder.get("sheets") or {}
        out.update(survey_year=tdef.get("year"), sheets_total=sh.get("n_sheets"),
                   sheets_with_year=sh.get("n_sheets_with_year"),
                   sheets_without_year=[r["sheet"] for r in sh.get("sheets", []) if not r.get("used")],
                   cells_with_year_on_the_sheets=sh.get("cells_with_year"), cells_in_F=f["cells"],
                   sheets_in_F=f["sheets"], cells_in_scored_units=s["cells"])
    if tdef.get("groups"):
        out["groups"] = group_counts(tdef, holder, extras)
    return out


def group_counts(tdef, holder, extras):
    """Per survey-year group (its row name): the year, its truth date, the
    keep pixels whose most recent survey is of that year, its units of F and
    scored units, and the 20 m cells and sheets under them."""
    gy = np.asarray(holder.get("group_year"), np.int64)
    F = np.asarray(extras["unit_keep_full"], np.int64)
    U = np.asarray(extras["units_keep"], np.int64)
    out = {}
    for g in tdef["groups"]:
        y = int(g["year"])
        hg = holder.get("groups") or {}
        per = hg.get(y) or hg.get(str(y)) or {}
        sh = per.get("sheets") or {}
        used = [r["sheet"] for r in sh.get("sheets", []) if r.get("used")]
        cid = per.get("cell_id")

        def cells(idx):
            if cid is None:
                return {"cells": None, "sheets": []}
            c = np.asarray(cid)[np.asarray(idx, np.int64)]
            c = c[c >= 0]
            k = sorted({int(v >> 40) for v in np.unique(c)})
            return {"cells": int(len(np.unique(c))), "sheets": [used[i] for i in k if i < len(used)]}

        fF, fU = F[gy[F] == y], U[gy[U] == y]
        cf, cu = cells(fF), cells(fU)
        out[g["name"]] = {"survey_year": y, "date": g["end"], "keep_px_with_this_survey": int((gy == y).sum()),
                          "units_F": int(len(fF)), "units_scored": int(len(fU)), "cells_in_F": cf["cells"],
                          "sheets_in_F": cf["sheets"], "cells_in_scored_units": cu["cells"],
                          "sheets_with_year": sh.get("n_sheets_with_year"), "sheets_total": sh.get("n_sheets"),
                          "sheets_without_year": [r["sheet"] for r in sh.get("sheets", []) if not r.get("used")],
                          "cells_with_year_on_the_sheets": sh.get("cells_with_year")}
    return out


def run(args):
    from pyintertidal import marea

    site, out = args.site, args.out
    v14.guard_out(out)
    os.makedirs(out, exist_ok=True)
    paths = site_paths(out, site, args.tag)
    deadline = _T0 + float(args.max_minutes) * 60.0
    soft = bool(args.soft_gates)
    ns = parse_ns(args.ns)
    tdef = truth_definition(site, args)
    if tdef["status"] == "pending":
        return write_pending(args, site, out, paths, tdef, ns)
    gates = {}
    g = v14_sha_gate()
    gates["v14_sha256"] = dict(g, stop_if_failed=True)
    if not g["ok"]:
        raise SystemExit(f"v14 changed: sha256 {g['sha256']} != {V14_SHA256}; refusing to run (never soft)")
    cfg, overrides = site_cfg(args)
    seed = int(args.seed)
    max_px = DEFAULT_MAX_PX if args.max_px is None else int(args.max_px)
    hinfo0 = host_info()
    log(f"[{site}] v19 fixed-configuration tables: n {ns}, truth {tdef['label_en']}, window end {tdef['end']}, "
        f"rules mb {list(MBS)} (headline mb {MB_HEADLINE}), seed {seed}, max px {max_px:,}, max {args.max_minutes} "
        f"min; host {hinfo0['host']}, threads {v14.blas_info()}")

    # 1. v14's site with this design's truth (its 'rtk' branch takes F as the unit function gives it: no
    #    product decides F), v14's gates and reference points
    swap_name, swap_fn, holder = prepare_truth(site, cfg, tdef, args, gates)
    t0 = time.perf_counter()
    with truth_swap(swap_name, swap_fn):
        sd14, meta, extras = v14.load_site(site, dict(cfg, truth="rtk"), "refit", max_px, seed, ["screened"])
    load_s = time.perf_counter() - t0
    # the Dutch truth is sampled inside load_site (not part of the shared preprocessing); the raster before it
    truth_s = float(holder.get("seconds") or 0.0) if holder.get("in_load_site") else 0.0
    # 10 m pixels scored like a raster truth (v7.vscore) at every site, never like RTK points
    sd14.kind = "vaklodingen"
    extras["kind"] = "vaklodingen"
    fs = meta.setdefault("fixed_set", {})
    fs.update(unit=f"{float(fs.get('pixel_m') or 10.0):g} m pixel", definition=F_RULE,
              v14_definition_replaced=fs.get("definition"))
    if site == "villaviciosa":
        seen = holder.get("z_stored_seen")
        if seen is None or not np.array_equal(seen, holder["z_pre"], equal_nan=True) or \
                not np.array_equal(np.asarray(extras["z_stored"], float), holder["z_pre"], equal_nan=True):
            raise RuntimeError(f"[{site}] the stored z of the range-filter gate is not load_site's")
    t1 = time.perf_counter()
    gates14, ok14, tim14 = v14.run_gates(site, cfg, sd14, extras, list(MBS), seed, "refit", [])
    sd14.refit_cache.clear()
    refs = v14.reference_points(site, cfg, sd14, extras)
    gates14_s = time.perf_counter() - t1
    pub = gates14.get("stored_product_vs_published_row")
    if isinstance(pub, dict):
        pub.update(applicable=False, stop_if_failed=False,
                   note=(f"not applicable under {TRUTH_DESIGN}: the published MAREA point is scored on another truth "
                         "(RTK / all-years Vaklodingen); the stored product is reproduced by the other gates"))
    for k, v in gates14.items():
        log(f"[{site}] v14 gate {k}: {'OK' if v.get('ok', True) else 'FAILED'}"
            + (" (not applicable)" if v.get("applicable") is False else ""))

    # 2. the scored units, the truth used, compat
    units_keep = np.asarray(extras["units_keep"], np.int64)
    counts = truth_counts(site, tdef, holder, extras, meta)
    truth_doc = dict(tdef, counts=counts, fixed_set_info={k: v for k, v in (meta.get("fixed_set") or {}).items()
                                                          if k not in ("sheets",)},
                     truth_seconds=round(float(holder.get("seconds") or 0.0), 2))
    log(f"[{site}] truth {tdef['label_en']}: F = {counts['units_F']:,} px ({counts['units_scored']:,} scored)"
        + (f", {counts.get('cells_in_F'):,} cells of {tdef.get('year')} on sheets {counts.get('sheets_in_F')} "
           f"({counts.get('sheets_with_year')} of {counts.get('sheets_total')} sheets with a {tdef.get('year')} survey)"
           if counts.get("cells_in_F") is not None else ""))
    # the survey-year groups (the Wadden): the year of every unit (its most recent survey)
    groups = tdef.get("groups") or None
    unit_group = None
    if groups:
        gy = holder.get("group_year")
        if gy is None or len(gy) != len(extras["keep"]):
            raise RuntimeError(f"[{site}] the grouped truth gave no survey year per keep pixel")
        unit_group = np.asarray(gy, np.int64)[units_keep]
        bad = sorted(set(np.unique(unit_group).tolist()) - {int(g["year"]) for g in groups})
        if bad:
            raise RuntimeError(f"[{site}] units of F outside the survey-year groups: years {bad}")
        for g in groups:
            c = counts["groups"][g["name"]]
            log(f"[{site}] group {g['name']} (truth date {g['end']}): F = {c['units_F']:,} px "
                f"({c['units_scored']:,} scored), {c['cells_in_F'] or 0:,} cells on {len(c['sheets_in_F'])} sheet(s) "
                f"{c['sheets_in_F']}; {c['sheets_with_year']} of {c['sheets_total']} sheets with a {g['year']} survey")
    mine = {"seed": seed, "n_units_scored": int(len(sd14.unit_col)), "n_units_F": int(len(extras["unit_keep_full"])),
            "units_sha1": v14.array_sha1(units_keep), "z_stored_sha1": v14.array_sha1(extras["z_stored"]),
            "tau_fixed_sha1": v14.array_sha1(sd14.tau_fixed)}
    if groups:
        mine["unit_group_sha1"] = v14.array_sha1(unit_group)
    compat = {"version": 3, "site": site, "max_px": max_px, "mbs": list(MBS), "mb_headline": MB_HEADLINE, **mine,
              "F": "truth inside the range filter; no product decides it",
              "truth": {"design": TRUTH_DESIGN, "kind": tdef["kind"], "date": tdef["date"], "end": tdef["end"],
                        "file_sha256": (tdef.get("file") or {}).get("sha256")}}
    if groups:
        compat["truth"]["groups"] = [int(g["year"]) for g in groups]
    gates["fixed_set_is_v14s"] = fixed_set_gate(site, mine, tdef)
    for p in (paths["settings"], os.path.join(out, f"{site}_settings.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                prev = json.load(f).get("compat")
            if prev is not None and prev != v14._clean(compat):
                raise SystemExit(f"{p} was written with other settings:\n  {prev}\nnow:\n  {compat}\n"
                                 "refusing to mix them: use another --out")
    z_stored_units = np.asarray(extras["z_stored"], float)[units_keep]
    uextra = {"z_stored_units": z_stored_units, "truth_design": TRUTH_DESIGN, "truth_label": tdef["label_en"]}
    if groups:
        uextra["unit_group"] = unit_group
    save_units(paths["units"], units_keep, sd14.truth, sd14.unit_col, extras["px"], sd14.kind, extra=uextra)

    # 3. the screened list S and the windows (ending on the truth date)
    t1 = time.perf_counter()
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        ext_dates = np.asarray(ex["dates"]).astype(str)
        shape = tuple(int(v) for v in ex["shape"])
    res = extras["res"]
    rec_dates = list(extras["rec_dates"])
    boundary = extras["rec"]["boundary_obj"]
    S, s_info = screened_list(site, cfg, ext_dates, res, extras["rec"], rec_dates, args.usable_json)
    gates.update(record_part_gates(S, rec_dates, res["rango_marea_m"]))
    S, x_info, x_note = extend_with_extra(S, site, args.extra_dir, extras["keep"], shape, ext_dates, boundary,
                                          args.pre2023_levels, soft=soft)
    all_dates = list(ext_dates)
    sources = {0: cfg["extract"]}
    for xi in x_info:
        if xi.get("used"):
            sources[int(xi["source_index"])] = os.path.join(ROOT, xi["path"])
            with np.load(sources[int(xi["source_index"])], allow_pickle=False) as ex:
                all_dates += [d for d in np.asarray(ex["dates"]).astype(str) if d < ext_dates[0]]
    if groups:
        # per survey-year group its own truth date and windows; the record is the most recent group's gate
        windows = group_windows(S["dates"], S["h0"], ns, len(rec_dates), all_dates, groups)
    else:
        windows = make_windows(S["dates"], S["h0"], ns, len(rec_dates), all_dates, end=tdef["end"])
    rekey_windows(windows, S, x_info)          # a window of store scenes: keyed by the stores too
    screening_s = time.perf_counter() - t1
    gates["windows_nested"] = check_nested(windows)
    avail = windows[0]["available"] if windows else 0
    log(f"[{site}] window end {tdef['end']} ({tdef['end_rule']}): {avail} of {len(S['dates'])} screened scenes "
        "on or before it")
    group_avail = {}
    for g in groups or []:
        wg = [w for w in windows if w.get("group") == int(g["year"])]
        group_avail[g["name"]] = int(wg[0]["available"]) if wg else 0
        log(f"[{site}] group {g['name']}: window end {g['end']}, {group_avail[g['name']]} screened scenes on or "
            "before it")
    for w in windows:
        log(f"[{site}] window {w['label']:6s}" + (" (record)" if w["is_record"] else "")
            + (" (gate only)" if w.get("gate_only") else "") + ": "
            + (f"{w['first_date']}..{w['last_date']}, {w['years']}, {w.get('cube_scenes')} cube scenes"
               if w["reachable"] else f"NOT REACHABLE: {w['reason']}"))
    tabulated = table_windows(windows)

    # 4. methods
    registry, reg_info = ({}, {"status": "off"}) if args.registry == "off" else \
        load_registry(variants=args.with_variants)
    log(f"[{site}] registry {REGISTRY_MODULE}: {reg_info.get('status')} "
        f"{reg_info.get('methods') or reg_info.get('reason', '')}"
        + (f", variants {reg_info.get('variants')}" if reg_info.get("variants") else ""))
    for k, v in (reg_info.get("registry_errors") or {}).items():
        log(f"[{site}] registry error in {k}: {v}")
    for k, v in (reg_info.get("registry_skipped") or {}).items():
        log(f"[{site}] registry: {k} skipped ({v})")
    nidem_fn, nidem_info = (None, {"status": "off", "reason": "--nidem off"}) if args.nidem == "off" \
        else find_nidem_hook(args.nidem_func)
    if nidem_fn is None and "E1" in registry:
        nidem_info = dict(nidem_info, fallback="registry E1")
    log(f"[{site}] NIDEM: {nidem_info.get('status')} {nidem_info.get('function') or nidem_info.get('reason', '')}")
    comps = [m for m in registry if m not in RUNNER_IDS and m not in NIDEM_RULE]
    methods = ["E0b", "E0a"] + comps + list(NIDEM_RULE)
    if args.methods:
        want = set(parse_list(args.methods))
        methods = [m for m in methods if m in want]
    runnable = [m for m in methods if m not in RUNNER_IDS and
                ((m in NIDEM_RULE and nidem_fn is not None) or
                 (m in registry and resolve_run(registry[m]["run"])[0] is not None))]
    hp = hook_params(nidem_fn) if nidem_fn is not None else set()
    keep_for = [m for m in runnable if m not in NIDEM_RULE or hp is None or bool({"obs", "frame", "C", "Y"} & hp)]
    need_keep = bool(keep_for) and bool(tabulated)
    # DEA Intertidal refuses a temp root inside the repository (the server's env.sh sets TMPDIR there): every
    # E2 job would end as an error, so the run stops before any job instead (--dea-tmp or $MAREA_DEA_TMP: a
    # folder outside the repository)
    dea_tmp = getattr(args, "dea_tmp", None)
    if "E2" in runnable and tabulated:
        why = dea_tmp_problem({"dea_tmp": dea_tmp} if dea_tmp else None)
        if why:
            raise SystemExit(f"[{site}] {why}")

    # 5. the window table
    t1 = time.perf_counter()
    pos14 = {d: i for i, d in enumerate(np.asarray(extras["dates_all"]).astype(str))}
    m2a_cols, m2a_band = lag_fit_columns(cfg, np.asarray(res["centros_km"], float))
    gates["lagfit_bands_are_v14s"] = {"ok": bool(np.array_equal(m2a_band, sd14.m2a["band"]))}
    table, t_info = build_table(site, sources, S, extras["px"], m2a_cols, len(extras["keep"]), boundary, sd14,
                                need_keep, pos14)
    table_s = time.perf_counter() - t1
    gates["window_table_vs_v14"] = compare_with_v14(table, sd14, S, pos14)
    dl = float(np.max(np.abs(table.levels[0] - S["h0"]))) if len(S["h0"]) else 0.0
    gates["window_table_levels_vs_screening"] = {"max_abs_diff_m": dl, "ok": dl <= v14.TOL_LEVEL}
    prov = level_provenance(boundary, S["t"])
    write_json(paths["windows"], windows_doc(site, S, s_info, windows, x_info, prov, tdef))
    for w in windows:
        if w["reachable"]:
            sl = window_slice(w)
            w["clear_obs_inverted_px"] = clear_stats(table.C[sl])
            w["clear_obs_keep_px"] = clear_stats(table.Ck[sl]) if table.Ck is not None else None
            if prov is not None:
                w["levels_all_members"] = int(prov["all_members"][sl].sum())
                w["levels_partial"] = int((~prov["all_members"][sl]).sum())
    sd14.Y0 = sd14.C = None                     # v14's copies are no longer needed
    sd14.m2a = None

    ctx = Context(site=site, seed=seed, table=table, S=S, paths=paths, registry=registry, nidem_fn=nidem_fn,
                  nidem_info=nidem_info, units_keep=units_keep, unit_col=np.asarray(sd14.unit_col, np.int64),
                  px=np.asarray(extras["px"], np.int64), keep_flat=np.asarray(extras["keep"], np.int64),
                  shape=shape, pixel_m=float(meta.get("fixed_set", {}).get("pixel_m") or 10.0),
                  units_sha1=mine["units_sha1"], boundary_name=str(extras["rec"].get("boundary")),
                  level_series=None, soft=soft, io_lock=args.io_lock, nidem_cache_dir=args.nidem_cache_dir,
                  hook_inputs={k: cfg[k] for k in ("extract", "product", "comparison", "nidem", "cube")
                               if cfg.get(k)},
                  window_select_s={}, dea_tmp=dea_tmp, unit_group=unit_group)
    store = JobStore(paths["jobs"])
    ls_info = None
    ls_s = 0.0
    if tabulated and any("level_series" in (registry[m]["card"].get("needs") or []) for m in runnable
                         if m in registry):
        t1 = time.perf_counter()
        ctx.level_series, ls_info = level_series_for(ctx, boundary, windows,
                                                     os.path.join(out, f"{site}_level_series.npz"))
        ls_s = time.perf_counter() - t1
    site_parts = {"load_without_truth": max(0.0, load_s - truth_s), "screening_and_windows": screening_s,
                  "window_table": table_s, "level_series": ls_s}
    write_timing(paths["timing"], ctx, site_parts)

    # 6. settings
    settings = {
        "site": site, "site_name": SITE_NAMES.get(site, site), "status": "running", "compat": compat,
        "question": ("elevation methods under one fixed external configuration (MAREA's production), per site "
                     "and number of screened images n, each window ending on the site's truth date"),
        "truth": truth_doc,
        "window_end": {"date": tdef["end"], "rule": tdef["end_rule"], "screened_scenes_up_to_it": int(avail),
                       "screened_total": int(len(S["dates"]))},
        "truth_groups": (None if not groups else {
            "rule": GROUP_END_RULE, "grouping": tdef.get("grouping"),
            "groups": [dict(g, screened_scenes_up_to_it=group_avail.get(g["name"]),
                            counts=counts["groups"][g["name"]],
                            windows=[w["label"] for w in windows if w.get("group") == int(g["year"])],
                            tabulated=[w["label"] for w in tabulated if w.get("group") == int(g["year"])])
                       for g in groups],
            "pooled": ("a row '" + site + "' only at the n where every group's window is reachable: each pixel "
                       "scored with its own group's estimate; the common set = the pixels where every headline "
                       "method has a value; estimator seconds = the sum of the groups'"),
            "pooled_ns": sorted({int(w["n"]) for w in tabulated if w.get("requested")
                                 and all(any(x.get("group") == int(g["year"]) and int(x["n"]) == int(w["n"])
                                             and x.get("requested") for x in tabulated) for g in groups
                                         if np.any(unit_group == int(g["year"])))}, reverse=True),
            "pooled_rule_empty_group": "a group without any unit of F never holds the pooled row back",
            "F": "the same F rules in every group (the production range filter; no product decides F)",
            "record": f"the production record is the gate window of {groups[0]['name']} only"}),
        "git_head": v14.git_head(), "script": v14.file_info(rel(__file__)),
        "v14_script": v14.file_info(rel(v14.__file__)), "python": sys.version.split()[0], "numpy": np.__version__,
        "blas": v14.blas_info(), "host": hinfo0["host"], "threads": hinfo0["threads"], "seed": seed,
        "max_px": max_px, "ns": ns, "input_overrides": overrides,
        "common_configuration": COMMON_CONFIG,
        "rules": {"mb_scored": list(MBS), "mb_headline": MB_HEADLINE,
                  "mb_0": "rule of the stored / published 10 m products (made before the rule existed): headline",
                  f"mb_{marea.QUALITY_MIN_BRACKET}": "pyintertidal.marea.QUALITY_MIN_BRACKET, the current production "
                                                     "rule: variant from the same fit"},
        "marea": ("v14.run_job(lags='refit') on the window: m2a_rasch band lags on production's seeded lag-fit "
                  "pixels (configs/m2.yaml, <= 1200 px per band, 3-km bands), monotone, tau(0) = 0, negative bands "
                  "set to 0, smooth_lag_profile, rounded; invert_series defaults; lo/hi = the window's zero-lag "
                  "level range"),
        "screening": s_info, "extra_stores": x_info, "extra_note": x_note,
        "pre2023_levels_policy": args.pre2023_levels,
        "S": {"n": int(len(S["dates"])), "per_year": per_year(S["dates"]), "first": str(S["dates"][0]),
              "last": str(S["dates"][-1]), "sha1": scenes_sha1(S["dates"])},
        "level_provenance": (None if prov is None else {k: v for k, v in prov.items() if k != "all_members"}),
        "windows": windows, "table": t_info, "methods": {
            "order": methods, "runnable_competitors": runnable, "registry": reg_info,
            "cards": {m: e["card"] for m, e in registry.items()}, "nidem": nidem_info,
            "pixel_sets": {m: pixel_set_of(registry[m]["card"]) for m in registry},
            "level_series": ls_info,
            "level_series_rule": ("h_B without lag at 1 min from 00:00 of the window's first scene day to 00:00 of "
                                  "the day after the window's end (exclusive), sliced from one series per site"),
            "dea_tmp_root": (dea_tmp or os.environ.get("MAREA_DEA_TMP")
                             or "the system temp dir (no --dea-tmp, no $MAREA_DEA_TMP)")},
        "scoring": {"F": meta.get("fixed_set"), "F_rule": F_RULE,
                    "common": ("units of F (subsample) where every headline method computed at that n has a value; "
                               "its share of F is the common coverage"),
                    "own": "each method's own finite units of F (subsample)",
                    "metrics": ("v14.score: cRMSE = RMSE after removing each series' median (v7.vscore), OLS slope, "
                                "r, bias = median(product - truth); rmse_total = the uncentred RMSE; coverage = "
                                "finite share of the scored units")},
        "v14_gates": gates14, "gates": gates, "reference_points_v14": refs,
        "timing": {"load_seconds": round(load_s, 1), "truth_seconds": round(float(holder.get("seconds") or 0.0), 1),
                   "v14_gates_seconds": round(gates14_s, 1), **tim14, "table_seconds": t_info["seconds"],
                   "shared_site_parts_s": {k: round(float(v), 2) for k, v in site_parts.items()},
                   "rule": ("seconds_estimator = the estimator call alone (time.perf_counter); seconds_shared = the "
                            "site's shared preprocessing (truth loading and gates excluded) + the window's "
                            "selection; see <site>_timing.json")},
        "inputs": meta.get("inputs"), "lag_fit": meta.get("lag_fit"), "boundary": meta.get("boundary"),
        "caveats": [
            "n is confounded with time span and season: nested deterministic windows, one realisation each",
            "small windows shrink the level range (lo/hi), which clamps MAREA and shifts interval methods",
            "F is decided by no product (truth inside the range filter): MAREA's coverage is measured on it like "
            "every other method's",
            "MAREA reads a clear observation with undefined NDWI as 0 (dry); other methods see NaN",
            "NIDEM (E1) applies the common mask to the whole frame its contours and TIN need; E1i is the same NIDEM "
            "under ITEM's mask (a declared sensitivity)",
            "every method but MAREA reads the production h_B at the scene instants without lag, in place of its own "
            "tide source (DEA Intertidal: per-pixel EOT20 tides; Coast X-Ray: POLPRED)",
            "label instants (datatake start) bias every level by 0-10 min (kept by the user on 2026-10-14)",
            f"the truth is dated: {tdef['end_rule']}",
        ] + ([f"survey-year groups: {tdef.get('grouping')}"] if groups else []),
        "runs": [],
    }
    if os.path.exists(paths["settings"]):
        with open(paths["settings"], encoding="utf-8") as f:
            settings["runs"] = json.load(f).get("runs", [])
    this_run = {"started": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:], **hinfo0}
    settings["runs"].append(this_run)

    def finish(status):
        settings["status"] = status
        settings["timing"]["level_calls"] = table.levels.calls
        settings["timing"]["peak_rss_mb"] = peak_rss_mb()
        settings["timing"]["elapsed_minutes"] = round((time.time() - _T0) / 60, 2)
        this_run.update(finished=_dt.datetime.now().isoformat(timespec="seconds"), status=status)
        v14.write_json(paths["settings"], settings)

    failed = ([f"v14:{k}" for k in stop_failed(gates14)] if not ok14 else []) + stop_failed(gates)
    for k, v in gates.items():
        log(f"[{site}] gate {k}: {'OK' if v.get('ok', True) else 'FAILED'}")
    if failed and not soft:
        finish("gate_failed")
        raise SystemExit(f"[{site}] gate(s) {failed} failed; see {paths['settings']}")
    if failed:
        settings["gates_overridden"] = failed
        log(f"[{site}] WARNING: gate(s) {failed} failed and --soft-gates is set: continuing; the JSON says so")
    v14.write_json(paths["settings"], settings)

    # 7. the jobs
    plan = job_plan(windows, methods, args.record_window)
    by_label = {w["label"]: w for w in windows if w["reachable"]}
    if not tabulated:
        log(f"[{site}] no window of n in {ns} is reachable on or before {tdef['end']}: only the record gate runs")
    # every NIDEM job (E1 and E1i of every window) still to do: one hook call, one cube read for all
    ctx.nidem_todo = []
    if nidem_fn is not None:
        for lab, mid in plan:
            if mid in NIDEM_RULE and done(store, wkey(by_label[lab], mid), by_label[lab]["scenes_sha1"],
                                          ctx.units_sha1, method_fingerprint(ctx, mid)) is None:
                ctx.nidem_todo.append((by_label[lab], mid))
    status, timing = {}, {}
    stopped = None
    for lab, mid in plan:
        w = by_label[lab]
        key = wkey(w, mid)
        fp = method_fingerprint(ctx, mid)
        m = done(store, key, w["scenes_sha1"], ctx.units_sha1, fp) if fp else None
        if m is not None:
            status[key] = "resumed"
        else:
            if time.time() > deadline:
                stopped = {"before": key, "reason": "--max-minutes passed (resumable)"}
                log(f"[{site}] STOP before {key}: --max-minutes passed")
                break
            m, arrays = run_one(ctx, w, mid)
            store.save(key, m, arrays)
            status[key] = m["status"]
            timing[key] = {"seconds": m["seconds"], "seconds_estimator": m.get("seconds_estimator")}
            write_timing(paths["timing"], ctx, site_parts)
            rescore_site(out, site, store)
            msg = f"[{site}] {w['label']:6s} {mid:5s}: {m['status']}"
            if m["status"] == OK and arrays:
                # a survey-year group's window: on its group's units
                gsel = (np.asarray(unit_group) == int(w["group"])) if unit_group is not None and \
                    w.get("group") is not None else slice(None)
                for var in m["variants"]:
                    s = score_values(ctx.table.kind, np.asarray(ctx.table.truth)[gsel],
                                     np.asarray(arrays[f"z__{var}"])[gsel], ctx.unit_col[gsel])
                    msg += f" | {var} RMSE {s['rmse']:.4f} cov {s['coverage']:.3f} n {s['n']:,}"
            elif m.get("reason"):
                msg += f" ({m['reason'][:160]})"
            se = m.get("seconds_estimator")
            log(msg + f" | {m['seconds']:.1f} s" + (f" (estimator {se:.1f} s)" if se is not None else "")
                + (f" | peak {peak_rss_mb() / 1024:.2f} GB" if peak_rss_mb() else ""))
        if w["is_record"] and mid == "E0b":
            z0 = store.arrays(key)["z__mb0"]
            gg = record_gate(site, ctx.table.kind, ctx.table.truth, ctx.unit_col, z0, z_stored_units,
                             m["info"].get("tau_bands_min"), res.get("tau_min"), None,
                             full_F=len(extras["unit_keep_full"]) == len(units_keep),
                             published_note=(f"the published MAREA point is scored on another truth (RTK / all-years "
                                             f"Vaklodingen); under {TRUTH_DESIGN} the record is compared with the "
                                             "stored product on the units"))
            gates["record_window_marea_mb0_reproduces_published"] = gg
            log(f"[{site}] gate record_window_marea_mb0_reproduces_published: {'OK' if gg['ok'] else 'FAILED'} "
                f"(flips {gg['validity_flips']}, max |dz| {gg['max_abs_dz_m']:.2e} m, max |score diff| "
                f"{gg['max_abs_score_diff']:.2e})")
            if not gg["ok"]:
                if not soft:
                    finish("gate_failed")
                    raise SystemExit(f"[{site}] the record window does not reproduce the stored MAREA; see "
                                     f"{paths['settings']}")
                settings.setdefault("gates_overridden", []).append("record_window_marea_mb0_reproduces_published")
            z_plain = None
            if cfg.get("comparison") and os.path.exists(os.path.join(cfg["comparison"], "products.npz")):
                with np.load(os.path.join(cfg["comparison"], "products.npz"), allow_pickle=False) as P:
                    if "z_plain" in P.files:
                        z_plain = np.asarray(P["z_plain"], float)[units_keep]
            settings["gates"] = gates
            if z_plain is not None:
                settings["no_delay_reference"] = {"z_plain_units": int(np.isfinite(z_plain).sum())}
            ctx.z_plain = z_plain
        if w["is_record"] and mid == "E0a" and getattr(ctx, "z_plain", None) is not None:
            za = store.arrays(key)["z__mb0"]
            fa, fb = np.isfinite(za), np.isfinite(ctx.z_plain)
            both = fa & fb
            gates["record_window_no_delay_vs_stored_z_plain"] = {
                "validity_flips": int(np.sum(fa != fb)), "stop_if_failed": False,
                "max_abs_dz_m": float(np.max(np.abs(za[both] - ctx.z_plain[both]))) if both.any() else None,
                "ok": bool(np.sum(fa != fb) <= v14.TOL_FLIP_SHARE * len(za)
                           and (not both.any() or np.max(np.abs(za[both] - ctx.z_plain[both])) <= v14.TOL_Z))}
            settings["gates"] = gates
        if mid in NIDEM_RULE and getattr(ctx, "nidem_gate", None) is not None and "nidem_hook" not in gates:
            gates["nidem_hook"] = dict(ctx.nidem_gate, stop_if_failed=False,
                                       note=("the hook's own gates (they raise when they fail, unless --soft-gates): "
                                             "the ITEM implementation against the stored sensitivity product; E1's "
                                             "common mask against the extraction's C and Y at keep (never soft)"))
            settings["gates"] = gates
        settings["progress"] = {"last": key, "status": status[key], "jobs_planned": len(plan),
                                "jobs_seen": len(status)}
        settings["timing"]["jobs_seconds"] = timing
        if getattr(ctx, "nidem_batch_info", None):
            settings["methods"]["nidem_batch"] = ctx.nidem_batch_info
        v14.write_json(paths["settings"], settings)
    write_timing(paths["timing"], ctx, site_parts)
    rows = rescore_site(out, site, store) or []
    settings["timing_per_window"] = timing_per_window(rows)
    n_ok = sum(1 for v in status.values() if v in ("resumed", OK))
    st = "stopped_by_time" if stopped else "complete"
    settings["progress"] = {"jobs_planned": len(plan), "jobs_done_or_resumed": n_ok, "stopped": stopped,
                            "status_per_job": status}
    finish(st)
    log(f"[{site}] {st}: {len(plan)} jobs planned, {n_ok} ok/resumed -> {paths['csv']} ({len(rows)} rows), "
        f"{paths['settings']}")
    head = [r for r in rows if r["scoring"] == "common" and r["role"] == "headline" and r["status"] == OK]
    for r in sorted(head, key=lambda r: (-int(r["n_scenes"]), str(r["site"]), r["method"])):
        log(f"[{r['site']}]   n={int(r['n_scenes']):4d} {r['method']:5s} common RMSE {r['rmse']:.4f} bias "
            f"{r['bias']:+.3f} (n {int(r['n_units']):,} of {int(r['n_set']):,}) | {r['seconds_estimator']:.1f} s")


def timing_per_window(rows):
    """{n: {'seconds_shared', 'host', 'threads', 'methods': {method: estimator seconds}}} from CSV rows (the
    computed headline methods and MAREA's no-delay reference)."""
    out = {}
    for r in rows:
        if r.get("status") != OK or r.get("scoring") != "common" or r.get("variant") not in ("main", HEAD_VAR):
            continue
        # a survey-year group's row: keyed by its row too ('wadden-2024|40')
        k = str(int(r["n_scenes"]))
        if split_row(r.get("site", ""))[1] is not None:
            k = f"{r['site']}|{k}"
        d = out.setdefault(k, {"seconds_shared": r.get("seconds_shared"), "methods": {},
                               "host": r.get("host"), "threads": r.get("threads")})
        d["methods"][r["method"]] = r.get("seconds_estimator")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  plan (light: dates, boundary levels and windows; no pixel data)
# ─────────────────────────────────────────────────────────────────────────────

def plan(args):
    from experiments import v7_waterline_contour as v7

    site, out = args.site, args.out
    v14.guard_out(out)
    os.makedirs(out, exist_ok=True)
    paths = site_paths(out, site)
    cfg, overrides = site_cfg(args)
    ns = parse_ns(args.ns)
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        ext_dates = np.asarray(ex["dates"]).astype(str)
        keep = np.asarray(ex["keep"])
        shape = tuple(int(v) for v in ex["shape"])
    with open(os.path.join(cfg["product"], "result.json"), encoding="utf-8") as f:
        res = json.load(f)
    if site != "villaviciosa":
        codes, missing = v14.gauge_cache(site)
        if missing:
            raise SystemExit(f"[{site}] gauge cache missing for {missing}: refusing to download")
    rec = v7.build_record(site, v7.SITES[site], ext_dates)
    rec_dates = [str(d)[:10] for d in res["record_dates"]]
    S, s_info = screened_list(site, cfg, ext_dates, res, rec, rec_dates, args.usable_json)
    gates = record_part_gates(S, rec_dates, res["rango_marea_m"])
    S, x_info, _ = extend_with_extra(S, site, args.extra_dir, keep, shape, ext_dates, rec["boundary_obj"],
                                     args.pre2023_levels, soft=args.soft_gates)
    tdef = truth_definition(site, args)
    end = tdef.get("end") or WINDOW_END
    if tdef.get("groups"):
        windows = group_windows(S["dates"], S["h0"], ns, len(rec_dates), list(ext_dates), tdef["groups"])
    else:
        windows = make_windows(S["dates"], S["h0"], ns, len(rec_dates), list(ext_dates), end=end)
    rekey_windows(windows, S, x_info)
    doc = windows_doc(site, S, s_info, windows, x_info, level_provenance(rec["boundary_obj"], S["t"]), tdef)
    doc.update(gates=gates, screening=s_info, input_overrides=overrides, truth=tdef)
    v14.write_json(paths["plan"], doc)
    print(f"\n{site}: S = {len(S['dates'])} screened scenes {S['dates'][0]}..{S['dates'][-1]} "
          f"{per_year(S['dates'])}")
    if tdef["status"] == "pending":
        print(f"  truth: PENDING ({tdef['reason']}); the windows below end on {end} only for the plan")
    else:
        print(f"  truth: {tdef['label_en']}; window end {end} ({tdef['end_rule']}); "
              f"{windows[0]['available'] if windows else 0} screened scenes on or before it")
    for k, v in gates.items():
        print(f"  gate {k}: {'OK' if v.get('ok') else 'FAILED'}")
    for w in windows:
        tag = " (record)" if w["is_record"] else ""
        tag += " (gate only)" if w.get("gate_only") else ""
        if w.get("row_site"):
            tag = f" {w['row_site']} (to {w.get('end')})" + tag
        if w["reachable"]:
            print(f"  n={w['n']:4d}{tag}: {w['first_date']}..{w['last_date']} (open bound {w['lower_bound_open']}), "
                  f"{w['years']}, {w['span_days']} d, {w.get('cube_scenes')} cube scenes, h_B "
                  f"[{w['lo_m']:.3f}, {w['hi_m']:.3f}] m")
        else:
            print(f"  n={w['n']:4d}{tag}: NOT REACHABLE - {w['reason']}")
    print(f"-> {paths['plan']}")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
#  tables
# ─────────────────────────────────────────────────────────────────────────────

TXT = {
    "en": {"site": "Site (truth)", "from": "From", "n": "n", "common": "Common units", "points": "points",
           "px": "px", "coverage": "coverage of F", "pending": "pending", "not computable": "not computable",
           "error": "error", "not implemented": "not implemented", "no results": "no results yet",
           "not reachable": "not reachable", "variant": "MAREA (rule mb {mb})",
           "missing": "{k} screened scenes missing (about {y} years before {y0})",
           "available": "{k} screened images up to {d} ({m} missing)",
           "gate_only": "the production record ends after the truth date {d} (computed only as the gate)",
           "pending truth": "pending truth", "time": "time (s)", "shared": "shared",
           "of_F": "of F", "row_rmse_total": "RMSE (uncentred), common", "row_slope_r": "slope / r, common",
           "row_n_common": "n (share of the common units)", "row_own": "own units: cRMSE (bias)",
           "row_own_n": "own units: n / coverage of F",
           "title_n": "n = {n} screened images per site", "title_rec": "the full production record per site"},
    "es": {"site": "Sitio \\sub{(verdad)}", "from": "Desde", "n": "n", "common": "Comunes", "points": "puntos",
           "px": "px", "coverage": "cobertura de F", "pending": "pendiente", "not computable": "no calculable",
           "error": "error", "not implemented": "sin implementar", "no results": "sin resultados todav\\'ia",
           "not reachable": "no alcanzable", "variant": "MAREA \\sub{{regla {mb}}}",
           "missing": "faltan {k} escenas cribadas (unos {y} a\\~nos antes de {y0})",
           "available": "{k} im\\'agenes cribadas hasta el {d} (faltan {m})",
           "gate_only": ("el registro de producci\\'on termina despu\\'es de la fecha de la verdad, {d} (solo "
                         "comprobaci\\'on)"),
           "pending truth": "verdad pendiente", "time": "tiempo (s)", "shared": "com\\'un",
           "of_F": "de F", "row_rmse_total": "RMSE sin centrar (comunes)", "row_slope_r": "pendiente OLS / r (comunes)",
           "row_n_common": "n (fracci\\'on de los comunes)", "row_own": "propios: cRMSE (sesgo)",
           "row_own_n": "propios: n / cobertura de F",
           "title_n": "n = {n} im\\'agenes cribadas por sitio", "title_rec": "el registro de producci\\'on completo"},
}
#: the same texts without LaTeX escapes (the figures)
PLAIN = {
    "en": {k: TXT["en"][k] for k in ("not reachable", "available", "gate_only", "missing")},
    "es": {"not reachable": "no alcanzable", "available": "{k} imágenes cribadas hasta el {d} (faltan {m})",
           "gate_only": "el registro de producción termina después de la fecha de la verdad, {d} (solo comprobación)",
           "missing": "faltan {k} escenas cribadas (unos {y} años antes de {y0})"},
}


def site_settings(out, site):
    """The settings JSON of a site under ``out`` (or None):
    <site>_settings.json, else (a run made only of --tag groups) the first
    <site>_settings__<tag>.json. With several files they must all hold the
    same compat block, else None (runs of other settings are never mixed)."""
    docs = {}
    for p in sorted(glob.glob(os.path.join(out, f"{site}_settings*.json"))):
        if ".tmp" in os.path.basename(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                docs[p] = json.load(f)
        except (OSError, ValueError):
            continue
    if not docs:
        return None
    compats = {json.dumps(d.get("compat"), sort_keys=True) for d in docs.values() if d.get("compat") is not None}
    if len(compats) > 1:
        log(f"{site}: its settings files hold different compat blocks ({sorted(docs)}): not tabulated")
        return None
    main = os.path.join(out, f"{site}_settings.json")
    return docs.get(main) or docs[sorted(docs)[0]]


def site_truth(out, site):
    """The truth block of a site's settings when it is this design's, else
    None (an earlier truth: never tabulated)."""
    st = site_settings(out, site)
    t = (st or {}).get("truth")
    return t if isinstance(t, dict) and t.get("design") == TRUTH_DESIGN else None


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def read_results(out, sites):
    """The per-site CSVs of ``sites`` whose settings hold this design's truth
    block; a CSV of an earlier truth (RTK, all-years Vaklodingen) is left
    out, with a log line."""
    import pandas as pd

    parts = []
    for s in sites:
        p = os.path.join(out, f"{s}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            if site_truth(out, s) is None:
                log(f"{s}: {p} is not of the {TRUTH_DESIGN} (no truth block in {s}_settings.json): left out")
                continue
            parts.append(pd.read_csv(p, keep_default_na=False, na_values=["nan", "NaN", ""]))
    if not parts:
        return pd.DataFrame(columns=list(CSV_COLUMNS))
    df = pd.concat(parts, ignore_index=True)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = float("nan") if c in ("seconds_estimator", "seconds_shared") else ""
    for c in ("site", "first_date", "last_date", "lower_bound_open", "years", "scenes_sha1", "method", "label",
              "variant", "role", "status", "scoring", "source", "fidelity", "notes", "reason", "host", "threads"):
        df[c] = df[c].fillna("").astype(str)
    return df


def table_columns(df):
    """The method columns of every table (the same in all): MAREA (its
    headline rule), then the other headline methods in catalogue order
    (registry ids not in the catalogue after them) that have at least one row
    other than 'not implemented'; each declared sensitivity with a computed
    row right after its method (:data:`SENSITIVITY_COLUMNS`: NIDEM with
    ITEM's mask after NIDEM); MAREA's other quality rules last. Returns
    [(method, variant)]."""
    have = df[(df["role"] == "headline") & ~df["status"].isin([NOT_IMPLEMENTED, NOT_REACHABLE, PENDING_TRUTH])]
    ids = set(have["method"])
    order = [m for m in CATALOGUE_IDS if m in ids] + sorted(m for m in ids if m not in CATALOGUE_IDS)
    cols = [("E0b", HEAD_VAR)] + [(m, "main") for m in order if m != "E0b"]
    for mid, var, after in SENSITIVITY_COLUMNS:
        d = df[(df["method"] == mid) & df["variant"].isin([var, ""])]
        if (~d["status"].isin([PENDING, NOT_REACHABLE, PENDING_TRUTH, NOT_IMPLEMENTED])).any():
            pos = next((i + 1 for i, c in enumerate(cols) if c[0] == after), len(cols))
            cols.insert(pos, (mid, var))
    for mb in MB_OTHERS:
        if ((df["method"] == "E0b") & (df["variant"] == f"mb{mb}")).any():
            cols.append(("E0b", f"mb{mb}"))
    return cols


def not_implemented(df):
    ni = df[(df["status"] == NOT_IMPLEMENTED)]["method"].unique().tolist()
    done_ = set(df[(df["status"] != NOT_IMPLEMENTED) & (df["status"] != NOT_REACHABLE)
                   & (df["status"] != PENDING_TRUTH)]["method"])
    return [m for m in CATALOGUE_IDS if m in ni and m not in done_]


def pending_methods(df, cols):
    """{method: why} of the table columns that are 'pending' in every row of
    the tables (the code exists but did not run: e.g. DEA Intertidal without
    its side interpreter on that host)."""
    out = {}
    for mid, var in cols:
        d = df[(df["method"] == mid) & ~df["status"].isin([NOT_REACHABLE, PENDING_TRUTH])]
        if len(d) and (d["status"] == PENDING).all():
            reasons = [str(v) for v in d["reason"] if str(v) not in ("", "nan", "None")]
            out[mid] = reasons[0][:240] if reasons else ""
    return out


#: the note of the rows of a gate-only record window (never in an n table)
GATE_ONLY_NOTE = "gate-only record window"
#: the reason of a site that has no results and no truth yet
VVA_PENDING_REASON = (f"PNOA LiDAR truth: run with --vva-truth <GeoTIFF> --vva-truth-date <flight date> (the RTK is "
                      f"never used; the repository holds {VVA_COB2})")


def _uniq(values):
    return ";".join(sorted({str(v) for v in values if str(v) not in ("", "nan", "None")}))


def _scores(row):
    """The scores of one CSV row: cRMSE ('rmse'), RMSE ('rmse_total'), bias,
    slope, r, n and coverage."""
    n = _f(row["n_units"])
    return {"rmse": _f(row["rmse"]), "rmse_total": _f(row["rmse_total"]), "bias": _f(row["bias"]),
            "slope": _f(row["slope"]), "r": _f(row["r"]), "n_units": int(n) if np.isfinite(n) else 0,
            "coverage": _f(row["coverage"])}


def table_data(df, sites, key, cols):
    """One table: per site its window, the common units (and their share of
    F, the common coverage) and per column, for both scorings, the scores
    (cRMSE, RMSE, bias, slope, r, n, coverage: 'common' on the common units,
    'own' on the method's own units of F) and the estimator seconds, or the
    status; plus the shared seconds, host and threads of (site, n). 'best' =
    the lowest common cRMSE among the headline columns. A site without
    results is 'pending truth' at Villaviciosa (no truth given yet), else 'no
    results'."""
    rows = []
    for site in sites:
        d = df[df["site"] == site]
        if key == "record":
            d = d[d["is_record"].astype(int) == 1]
        else:
            d = d[(d["n_scenes"].astype(int) == int(key)) & (d["notes"] != GATE_ONLY_NOTE)]
        r = {"site": site, "cells": {}}
        if d.empty:
            if site == "villaviciosa" and not (df["site"] == site).any():
                r.update(state="pending truth", reason=VVA_PENDING_REASON)
            else:
                r["state"] = "no results"
            rows.append(r)
            continue
        if (d["status"] == PENDING_TRUTH).all():
            r.update(state="pending truth", reason=str(d["reason"].iloc[0]))
            rows.append(r)
            continue
        if (d["status"] == NOT_REACHABLE).all():
            r["state"] = "not reachable"
            r["reason"] = str(d["reason"].iloc[0])
            r["n"] = int(d["n_scenes"].iloc[0])
            rows.append(r)
            continue
        r["state"] = "ok"
        r["n"] = int(d["n_scenes"].iloc[0])
        r["first_date"] = str(d["first_date"].iloc[0])
        r["last_date"] = str(d["last_date"].iloc[0])
        r["n_common"] = int(_f(d["n_common"].max()) if len(d) else 0)
        r["n_set"] = int(_f(d["n_set"].max()) if len(d) else 0)
        r["common_coverage"] = r["n_common"] / r["n_set"] if r["n_set"] else float("nan")
        sh = [v for v in (_f(x) for x in d["seconds_shared"]) if np.isfinite(v)]
        r["seconds_shared"] = sh[0] if sh else float("nan")
        ok_rows = d[d["status"] == OK]
        r["host"], r["threads"] = _uniq(ok_rows["host"]), _uniq(ok_rows["threads"])
        for mid, var in cols:
            dm = d[(d["method"] == mid) & ((d["variant"] == var) | (d["variant"] == ""))]
            c = {"status": "absent"}
            if not dm.empty:
                okc = dm[(dm["status"] == OK) & (dm["scoring"] == "common") & (dm["variant"] == var)]
                own = dm[(dm["status"] == OK) & (dm["scoring"] == "own") & (dm["variant"] == var)]
                if len(okc):
                    cm = _scores(okc.iloc[0])
                    ow = _scores(own.iloc[0]) if len(own) else None
                    c = {"status": OK, "common": cm, "own": ow, "seconds": _f(okc.iloc[0]["seconds_estimator"]),
                         # the common-set scores and the own coverage of F, flat (the curves and the time rows)
                         "rmse": cm["rmse"], "bias": cm["bias"], "slope": cm["slope"], "r": cm["r"],
                         "rmse_total": cm["rmse_total"], "n_units": cm["n_units"],
                         "coverage": ow["coverage"] if ow else float("nan"), "n_own": ow["n_units"] if ow else 0,
                         "rmse_own": ow["rmse"] if ow else float("nan")}
                else:
                    c = {"status": str(dm["status"].iloc[0]), "reason": str(dm["reason"].iloc[0]),
                         "seconds": _f(dm["seconds_estimator"].iloc[0])}
            r["cells"][(mid, var)] = c
        heads = [(k, c) for k, c in r["cells"].items() if not is_side_column(*k) and c["status"] == OK
                 and np.isfinite(c["rmse"])]
        r["best"] = min(heads, key=lambda kc: kc[1]["rmse"])[0] if len(heads) >= 2 else None
        rows.append(r)
    return rows


def col_label(mid, var, lang, labels):
    if is_marea_variant(mid, var):
        return TXT[lang]["variant"].format(mb=var[2:])
    if lang == "es":
        return tex_escape(LABELS_ES.get(mid, labels.get(mid, mid)))
    return dict(CATALOGUE).get(mid) or COLUMN_LABELS.get(mid) or labels.get(mid, mid)


def tex_escape(s):
    rep = {"á": "\\'a", "é": "\\'e", "í": "\\'i", "ó": "\\'o", "ú": "\\'u", "Á": "\\'A", "É": "\\'E",
           "Í": "\\'I", "Ó": "\\'O", "Ú": "\\'U", "ñ": "\\~n", "Ñ": "\\~N", "ü": '\\"u', "&": "\\&", "%": "\\%",
           "_": "\\_", "#": "\\#"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def fmt_int(n, lang):
    s = f"{int(n):,}"
    return s.replace(",", "\\,") if lang == "es" else s


def fmt_cov(c):
    if not np.isfinite(c):
        return "-"
    return "100" if c >= 0.9995 else f"{100 * c:.1f}"


def fmt_m(v, signed=False):
    """A metre value for a table cell (3 decimals; signed with 2), '-' when
    unknown."""
    v = _f(v)
    if not np.isfinite(v):
        return "-"
    return f"{v:+.2f}" if signed else f"{v:.3f}"


def fmt_slope_r(c):
    s, r = _f((c or {}).get("slope")), _f((c or {}).get("r"))
    return f"{s:.2f} / {r:.2f}" if np.isfinite(s) and np.isfinite(r) else "-"


def _missing_text(reason, lang, plain=False):
    """The table text of a window that is not reachable, from its reason:
    'S holds 36 screened scenes up to 2017-12-31: 4 missing ...' -> 'not
    reachable: 36 screened images up to 2017-12-31 (4 missing)'; a gate-only
    record window; the older '... 57 missing (about 2.5 years before
    2023 ...' form. ``plain``: without LaTeX escapes (the figures)."""
    import re

    T = PLAIN[lang] if plain else TXT[lang]
    reason = reason or ""
    m = re.search(r"S holds (\d+) screened scenes up to (\d{4}-\d\d-\d\d): (\d+) missing", reason)
    if m:
        return f"{T['not reachable']}: " + T["available"].format(k=m.group(1), d=m.group(2), m=m.group(3))
    m = re.search(r"ends after the truth date (\d{4}-\d\d-\d\d)", reason)
    if m:
        return T["gate_only"].format(d=m.group(1))
    m = re.search(r"(\d+) missing(?: \(about ([\d.]+) years before (\d{4}))?", reason)
    if not m:
        return T["not reachable"]
    k, y, y0 = m.group(1), m.group(2), m.group(3)
    if y is None:
        return f"{T['not reachable']}: {k}"
    return f"{T['not reachable']}: " + T["missing"].format(k=k, y=y, y0=y0)


def fmt_seconds(v):
    """Seconds for a table cell: 3 significant digits, '-' when unknown."""
    v = _f(v)
    if not np.isfinite(v):
        return "-"
    return f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"


def row_title(site_row, lang="en"):
    """The name of a table row (plain text): the site ('Wadden (Vlie)'), a
    survey-year group with its row id ('Wadden (Vlie), wadden-2024') or a
    grouped site's pooled row ('Wadden (Vlie), pooled')."""
    base, year = split_row(site_row)
    name = SITE_NAMES.get(base, base)
    if year is not None:
        return f"{name}, {site_row}"
    if base in SURVEY_YEAR_GROUPS:
        return f"{name}, {'pooled' if lang == 'en' else 'conjunto'}"
    return name


def truth_label(site, meta, lang="en", plain=False):
    """The truth of a table row (its settings' label, else the default
    name); a survey-year group's row: its year, sheets, 20 m cells and truth
    date (from the settings' counts). ``plain``: no LaTeX escapes."""
    base, year = split_row(site)
    t = ((meta or {}).get("truth") or {}).get(base) or {}
    if year is not None:
        c = ((t.get("counts") or {}).get("groups") or {}).get(site) or {}
        if c:
            k, cells = len(c.get("sheets_in_F") or []), int(c.get("cells_in_F") or 0)
            ce = f"{cells:,}".replace(",", " ") if plain else fmt_int(cells, "es")
            lab = (f"Vaklodingen {year}: {k} sheet(s), {cells:,} cells, truth date {c.get('date')}" if lang == "en"
                   else f"Vaklodingen {year}: {k} hoja(s), {ce} celdas, verdad al {c.get('date')}")
        else:
            lab = f"Vaklodingen {year}"
    else:
        lab = t.get("label_en" if lang == "en" else "label_es") or TRUTH_NAMES.get(site, "")
        if base in SURVEY_YEAR_GROUPS and t.get("groups"):
            lab += (", each pixel with its own group's estimate" if lang == "en"
                    else ", cada píxel con la estimación de su grupo")
    return lab if lang == "en" or plain else tex_escape(lab)


def table_sites(df, sites, key=None):
    """The rows of a table (or the panels of the curves) for the ``sites``:
    a site as it is; a site with survey-year groups expands to its group
    rows, the most recent first, then its pooled row when the CSV has one
    (only at the n where every group's window is reachable; ``key``: that
    table, None = any)."""
    names = set(df["site"].astype(str)) if len(df) else set()
    out = []
    for s in sites:
        groups = sorted((x for x in names if split_row(x)[0] == s and split_row(x)[1] is not None),
                        key=lambda x: -split_row(x)[1])
        if not groups:
            out.append(s)
            continue
        out += groups
        d = df[df["site"] == s]
        if key is not None and len(d):
            d = d[d["is_record"].astype(int) == 1] if key == "record" else d[d["n_scenes"].astype(int) == int(key)]
        if len(d):
            out.append(s)
    return out


#: what the scores of the tables are (under every table)
METRICS_NOTE = {
    "en": ("Common block: the units of F where every headline method has a value (their share of F is the common "
           "coverage): cRMSE (bias), RMSE, slope / r and n; a side column (MAREA's other quality rules, NIDEM with "
           "ITEM's mask) is scored on the common units where it has a value, with its n and share of them, and "
           "never defines them. Own block: each method on its own units of F: cRMSE (bias), n and coverage of F. "
           "cRMSE = RMSE after removing each series' median (v7.vscore); bias = median(product - truth); RMSE = "
           "uncentred (so RMSE^2 is not cRMSE^2 + bias^2); slope and r of the OLS fit of the centred series."),
    "es": ("Bloque com\\'un: los p\\'ixeles de F en los que todos los m\\'etodos principales tienen valor (su "
           "fracci\\'on de F es la cobertura com\\'un): cRMSE (sesgo), RMSE, pendiente OLS / r y n; una columna "
           "lateral (otras reglas de calidad de MAREA, NIDEM con la m\\'ascara de ITEM) se eval\\'ua en los comunes "
           "en que tiene valor, con su n y su fracci\\'on, y nunca los define. Bloque propio: cada m\\'etodo en sus "
           "p\\'ixeles de F con valor: cRMSE (sesgo), n y cobertura de F. cRMSE: RMSE tras quitar la mediana de "
           "cada serie (v7.vscore); sesgo: mediana de producto $-$ verdad; RMSE: sin centrar (RMSE$^2$ no es "
           "cRMSE$^2$ + sesgo$^2$); pendiente y r del ajuste OLS de las series centradas."),
}


def table_notes(rows, meta, lang="en"):
    """The notes under a table: the scores, F, the window end of each truth
    (Vaklodingen gives the survey year only), the Villaviciosa truth or its
    pending state, the truth used per site, the common conditions (h_B
    without lag for every method but MAREA, NIDEM's whole frame, E1i) and
    the timing rule with the hosts."""
    meta = meta or {}
    truths = meta.get("truth") or {}
    sites = [r["site"] for r in rows]
    out = []
    grouped = sorted({split_row(s)[0] for s in sites if split_row(s)[1] is not None})
    # the Dutch rows with the 2025 cells only (a site with survey-year groups has its own note)
    dutch = [s for s in sites if s != "villaviciosa" and split_row(s)[0] not in grouped]
    if lang == "en":
        out.append(METRICS_NOTE["en"])
        out.append(f"{F_RULE}.")
        out.append("Window: the n most recent screened images on or before each site's truth date.")
        if dutch:
            which = f" ({', '.join(SITE_NAMES.get(s, s) for s in dutch)})" if grouped else ""
            out.append(f"Dutch sites{which}: {DUTCH_END_RULE}.")
        for b in grouped:
            gs = [s for s in sites if split_row(s)[0] == b and split_row(s)[1] is not None]
            out.append(f"{SITE_NAMES.get(b, b)}: survey-year groups (the user's choice): per pixel the most recent "
                       "Vaklodingen survey in 2023-2025 of its 20 m cell; the rows " + ", ".join(gs) + " hold the "
                       "pixels of each survey year, each with its truth date at the end of that year and its own "
                       "windows (the n most recent screened images on or before it; 'not reachable' with the count "
                       "when fewer lie before it), the same F rules and its own common set; the row "
                       f"'{b}' pools them only at the n where every group's window is reachable, each pixel scored "
                       "with its own group's estimate (time: the sum of the groups'). The production record is the "
                       "gate window of the most recent group only.")
        if "villaviciosa" in sites:
            t = truths.get("villaviciosa")
            r = next(x for x in rows if x["site"] == "villaviciosa")
            if t and t.get("status") == "ready":
                out.append(f"Villaviciosa: {t.get('label_en')} ({t.get('path')}); the window ends on the flight date; "
                           f"{t.get('datum')}.")
            else:
                why = (t or {}).get("reason") or r.get("reason") or VVA_PENDING_REASON
                out.append(f"Villaviciosa: pending truth ({why}).")
        parts = []
        for s in sites:
            base, year = split_row(s)
            c = (truths.get(base) or {}).get("counts") or {}
            if year is not None:
                c = (c.get("groups") or {}).get(s) or {}
            if not c:
                continue
            txt = f"{row_title(s)}: F = {c.get('units_F', 0):,} px ({c.get('units_scored', 0):,} scored)"
            if year is not None:
                txt += (f", {int(c.get('cells_in_F') or 0):,} cells (20 m) whose most recent survey is of {year}, on "
                        f"{len(c.get('sheets_in_F') or [])} sheet(s); {c.get('sheets_with_year')} of "
                        f"{c.get('sheets_total')} sheets hold a {year} survey; truth date {c.get('date')}")
            elif c.get("cells_in_F") is not None:
                txt += (f", {c['cells_in_F']:,} cells (20 m) surveyed in {c.get('survey_year')} on "
                        f"{len(c.get('sheets_in_F') or [])} sheet(s); {c.get('sheets_with_year')} of "
                        f"{c.get('sheets_total')} sheets hold a {c.get('survey_year')} survey")
            parts.append(txt)
        if parts:
            out.append("Truth used: " + "; ".join(parts) + ".")
        out.append("MAREA: band lags refitted on each window; every other method reads the production h_B at the scene "
                   "instants without lag, in place of its own tide source (DEA Intertidal: per-pixel EOT20 tides; "
                   "Coast X-Ray: POLPRED). NIDEM reads the whole frame (contours, TIN) with the same mask rule off "
                   "keep: the common mask (SCL 4-7) for NIDEM, ITEM's (SCL 2, 4-7, 11) for 'NIDEM (ITEM mask)', a "
                   "declared sensitivity on the same scenes and h.")
        hosts = sorted({f"{r['host']} ({r['threads']})" for r in rows if r.get("state") == "ok" and r.get("host")})
        out.append("Row 'time (s)': the wall time of each method's estimator call at that n (time.perf_counter: no "
                   "I/O, no truth loading; MAREA: v14's refit and inversion; DEA Intertidal: its per-pixel steps "
                   "and edge cleaning, run in its own pool of worker processes (default 4); NIDEM: its phase 2 plus "
                   "its share of the phase-1 R codes by number of scenes, the cube read excluded); 'shared': the "
                   "site's common preprocessing (extraction load, screening, window table; truth loading and gates "
                   "excluded) plus the window's selection"
                   + (f". Host (threads): {', '.join(hosts)}." if hosts else "."))
        return out
    out.append(METRICS_NOTE["es"])
    out.append("F: los p\\'ixeles de la extracci\\'on con verdad dentro del filtro de rango; ning\\'un producto lo "
               "decide (tampoco MAREA): la cobertura de todos los m\\'etodos se mide en el mismo conjunto.")
    out.append("Ventana: las n im\\'agenes cribadas m\\'as recientes hasta la fecha de la verdad de cada sitio.")
    if dutch:
        which = f" ({', '.join(tex_escape(SITE_NAMES.get(s, s)) for s in dutch)})" if grouped else ""
        out.append(f"Sitios holandeses{which}: Vaklodingen solo da el a\\~no del levantamiento; la verdad son las "
                   f"celdas de 20 m levantadas en {VAK_TRUTH_YEAR} y la ventana termina el {DUTCH_WINDOW_END}, fin "
                   "de ese a\\~no.")
    for b in grouped:
        gs = [s for s in sites if split_row(s)[0] == b and split_row(s)[1] is not None]
        out.append(f"{tex_escape(SITE_NAMES.get(b, b))}: grupos por a\\~no de levantamiento (elecci\\'on del "
                   "usuario): cada p\\'ixel con el levantamiento Vaklodingen m\\'as reciente de 2023-2025 de su celda "
                   "de 20 m; las filas " + ", ".join(tex_escape(s) for s in gs) + " agrupan los p\\'ixeles de cada "
                   "a\\~no, cada una con la fecha de su verdad al final de ese a\\~no y sus propias ventanas (las n "
                   "im\\'agenes cribadas m\\'as recientes hasta esa fecha; no alcanzable, con el n\\'umero, si hay "
                   "menos), las mismas reglas de F y su propio conjunto com\\'un; la fila "
                   f"{tex_escape(b)} los junta solo con los n en que todas las ventanas de los grupos se alcanzan, "
                   "cada p\\'ixel evaluado con la estimaci\\'on de su grupo (tiempo: la suma de los grupos). El "
                   "registro de producci\\'on es la ventana de comprobaci\\'on del grupo m\\'as reciente.")
    if "villaviciosa" in sites:
        t = truths.get("villaviciosa")
        if t and t.get("status") == "ready":
            out.append(f"Villaviciosa: {tex_escape(t.get('label_es', ''))}; la ventana termina en la fecha del vuelo; "
                       "alturas ortom\\'etricas: RMSE sin sesgo, el sesgo incluye el datum.")
        else:
            out.append("Villaviciosa: verdad pendiente (LiDAR PNOA: falta --vva-truth-date).")
    out.append("NIDEM lee el marco completo (contornos, TIN) con la misma regla de m\\'ascara fuera de keep: la "
               "com\\'un (SCL 4-7) para NIDEM y la de ITEM (SCL 2, 4-7, 11) para NIDEM (m\\'ascara ITEM), una "
               "sensibilidad declarada con las mismas escenas y $h_B$.")
    out.append("Fila de tiempo: segundos de la llamada al estimador de cada m\\'etodo (sin E/S ni carga de la "
               "verdad; DEA Intertidal con su propio grupo de procesos, 4 por defecto); com\\'un: preproceso "
               "compartido del sitio y selecci\\'on de la ventana.")
    return out


def _pending_note(pend, lang):
    if not pend:
        return ""
    if lang == "en":
        return ("Pending (code present, not run): "
                + "; ".join(f"{m} {dict(CATALOGUE).get(m, COLUMN_LABELS.get(m, m))}" + (f" ({why})" if why else "")
                            for m, why in pend.items()) + ".")
    return (" Pendientes (c\\'odigo presente, sin ejecutar): "
            + "; ".join(tex_escape(LABELS_ES.get(m, m)) + (f" ({tex_escape(why)})" if why else "")
                        for m, why in pend.items()) + ".")


def table_markdown(rows, cols, key, labels, ni, desc=(), meta=None, pend=None):
    T = TXT["en"]
    title = T["title_rec"] if key == "record" else T["title_n"].format(n=key)
    head = ["Site (truth)", "From", "Common units"] + [
        col_label(m, v, "en", labels) + (" †" if m in desc and not is_side_column(m, v) else "") for m, v in cols]
    if key == "record":
        head.insert(1, "n")
    out = [f"### {title[0].upper()}{title[1:]}", "",
           "Per site: the common block (first row: cRMSE (bias) in m on the common units of F, where every headline "
           "method has a value; then RMSE, slope / r and n), the own block (each method on its own units of F) and "
           "each method's estimator time. **Bold**: the lowest common cRMSE of the site among the headline methods.",
           "", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        name = f"{row_title(r['site'])} ({truth_label(r['site'], meta, 'en')})"
        if r["state"] != "ok":
            txt = (_missing_text(r.get("reason"), "en") if r["state"] == "not reachable"
                   else f"{T['pending truth']}: {r.get('reason', '')}" if r["state"] == "pending truth"
                   else T["no results"])
            pre = [name] + ([str(r.get("n", ""))] if key == "record" else [])
            cells = pre + [txt] + [""] * (len(head) - len(pre) - 1)
            out.append("| " + " | ".join(cells) + " |")
            continue
        unit = T["px"]
        rec = [""] if key == "record" else []

        def lab(text, extra=""):
            return [f"  {text}"] + rec + ["", extra]

        base = [name] + ([str(r["n"])] if key == "record" else []) + [
            r["first_date"], f"{r['n_common']:,} {unit} ({fmt_cov(r['common_coverage'])} % {T['of_F']})"]
        lines = {k: lab(T[k]) for k in ("row_rmse_total", "row_slope_r", "row_n_common", "row_own", "row_own_n")}
        tim = lab(T["time"], f"{T['shared']} {fmt_seconds(r.get('seconds_shared'))}")
        for k in cols:
            c = r["cells"].get(k, {"status": "absent"})
            if c["status"] == OK:
                cm, ow = c["common"], c.get("own") or {}
                if not np.isfinite(cm["rmse"]):
                    base.append(f"n = {cm['n_units']} (too few)")
                else:
                    v = f"{fmt_m(cm['rmse'])} ({fmt_m(cm['bias'], True)})"
                    base.append(f"**{v}**" if r.get("best") == k else v)
                lines["row_rmse_total"].append(fmt_m(cm["rmse_total"]))
                lines["row_slope_r"].append(fmt_slope_r(cm))
                share = cm["n_units"] / r["n_common"] if r["n_common"] else float("nan")
                lines["row_n_common"].append(f"{cm['n_units']:,} ({fmt_cov(share)} %)")
                lines["row_own"].append(f"{fmt_m(ow.get('rmse'))} ({fmt_m(ow.get('bias'), True)})" if ow else "-")
                lines["row_own_n"].append(f"{int(ow.get('n_units', 0)):,} / {fmt_cov(_f(ow.get('coverage')))} %"
                                          if ow else "-")
            else:
                base.append(T.get(c["status"], c["status"]) if c["status"] != "absent" else "-")
                for v in lines.values():
                    v.append("")
            tim.append(fmt_seconds(c.get("seconds")) if c["status"] != "absent" else "")
        out.append("| " + " | ".join(base) + " |")
        for k in ("row_rmse_total", "row_slope_r", "row_n_common", "row_own", "row_own_n"):
            out.append("| " + " | ".join(lines[k]) + " |")
        out.append("| " + " | ".join(tim) + " |")
    if any(m in desc for m, _ in cols):
        out += ["", "† a catalogue variant, reported descriptively (docs/validation_v2_spec.md §3.2)."]
    if ni:
        out += ["", "Not implemented under the common configuration yet: "
                + ", ".join(f"{m} {dict(CATALOGUE).get(m, m)}" for m in ni) + "."]
    if pend:
        out += ["", _pending_note(pend, "en")]
    for note in table_notes(rows, meta, "en"):
        out += ["", note]
    return "\n".join(out) + "\n"


def tex_stack(label, width=12):
    """A header on two centred lines (split at the space nearest the middle)
    when longer than ``width`` characters; a label holding braces (a macro)
    is kept on one line."""
    words = str(label).split(" ")
    if len(str(label)) <= width or len(words) < 2 or "{" in str(label):
        return str(label)
    cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return ("\\begin{tabular}[b]{@{}c@{}}" + " ".join(words[:cut]) + "\\\\" + " ".join(words[cut:])
            + "\\end{tabular}")


def table_latex(rows, cols, key, labels, ni, desc=(), meta=None, pend=None):
    """One Spanish booktabs table (the style of docs/informe_fer): rows =
    sites (the common block: cRMSE with the bias in brackets, RMSE, slope /
    r, n; the own block: cRMSE (bias), n / coverage, with the window's first
    scene; then the estimator seconds), columns = methods; scaled down to the
    text width when wider."""
    T = TXT["es"]
    ncol = 2 + len(cols)
    spec = "@{}lr" + "c" * len(cols) + "@{}"
    heads = []
    for m, v in cols:
        if is_marea_variant(m, v):
            heads.append(f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}MAREA\\\\\\sub{{regla {v[2:]}}}\\end{{tabular}}")
        else:
            heads.append(tex_stack(col_label(m, v, "es", labels)) + ("$^\\dagger$" if m in desc else ""))
    head = [T["site"], T["common"]] + heads
    lines = ["% generado por experiments/v19_fixed_config_tables.py tables (no editar a mano)",
             "% requiere booktabs, xcolor, graphicx y float; define \\up, \\sub y los colores si faltan",
             "\\providecolor{marea}{RGB}{47,115,80}\\providecolor{muted}{RGB}{86,101,95}",
             "\\providecommand{\\up}[1]{\\textcolor{marea}{\\textbf{#1}}}",
             "\\providecommand{\\sub}[1]{{\\footnotesize\\color{muted}#1}}",
             "\\begin{table}[H]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]

    def sgn(v):
        return fmt_m(v, True).replace("+", "$+$").replace("-", "$-$") if np.isfinite(_f(v)) else "---"

    for r in rows:
        extra = f", n = {r['n']}" if key == "record" and r.get("n") else ""
        name = (f"{tex_escape(row_title(r['site'], 'es'))} "
                f"\\sub{{({truth_label(r['site'], meta, 'es')}{extra})}}")
        if r["state"] != "ok":
            txt = (_missing_text(r.get("reason"), "es") if r["state"] == "not reachable"
                   else T["pending truth"] if r["state"] == "pending truth" else T["no results"])
            lines.append(f"{name} & \\multicolumn{{{ncol - 1}}}{{l}}{{\\sub{{{txt}}}}} \\\\")
            continue
        unit = T["px"]
        base = [name, f"{fmt_int(r['n_common'], 'es')} \\sub{{{unit} ({fmt_cov(r['common_coverage'])}\\,\\% "
                      f"{T['of_F']})}}"]
        rows_ = {k: [f"\\quad\\sub{{{T[k]}}}", ""] for k in ("row_rmse_total", "row_slope_r", "row_n_common",
                                                             "row_own")}
        rows_["row_own_n"] = [f"\\quad\\sub{{desde {r['first_date']}; {T['row_own_n']}}}", ""]
        tim = [f"\\quad\\sub{{{T['time']}}}", f"\\sub{{{T['shared']} {fmt_seconds(r.get('seconds_shared'))}}}"]
        for k in cols:
            c = r["cells"].get(k, {"status": "absent"})
            if c["status"] == OK:
                cm, ow = c["common"], c.get("own") or {}
                if not np.isfinite(cm["rmse"]):
                    base.append(f"\\sub{{n = {cm['n_units']}}}")
                else:
                    v = fmt_m(cm["rmse"])
                    base.append((f"\\up{{{v}}}" if r.get("best") == k else v) + f" \\sub{{({sgn(cm['bias'])})}}")
                rows_["row_rmse_total"].append(f"\\sub{{{fmt_m(cm['rmse_total'])}}}")
                rows_["row_slope_r"].append(f"\\sub{{{fmt_slope_r(cm)}}}")
                share = cm["n_units"] / r["n_common"] if r["n_common"] else float("nan")
                rows_["row_n_common"].append(f"\\sub{{{fmt_int(cm['n_units'], 'es')} ({fmt_cov(share)}\\,\\%)}}")
                rows_["row_own"].append(f"\\sub{{{fmt_m(ow.get('rmse'))} ({sgn(ow.get('bias'))})}}" if ow else "")
                rows_["row_own_n"].append(f"\\sub{{{fmt_int(ow.get('n_units', 0), 'es')} / "
                                          f"{fmt_cov(_f(ow.get('coverage')))}\\,\\%}}" if ow else "")
            else:
                base.append(f"\\sub{{{T.get(c['status'], c['status'])}}}" if c["status"] != "absent" else "\\sub{---}")
                for v in rows_.values():
                    v.append("")
            tim.append(f"\\sub{{{fmt_seconds(c.get('seconds'))}}}" if c["status"] != "absent" else "")
        lines.append(" & ".join(base) + " \\\\")
        for k in ("row_rmse_total", "row_slope_r", "row_n_common", "row_own", "row_own_n"):
            lines.append(" & ".join(rows_[k]) + " \\\\")
        lines.append(" & ".join(tim) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    ns_rec = [str(r["n"]) for r in rows if r.get("state") == "ok" and r.get("n")]
    what = ((T["title_rec"] + (f" ({' / '.join(ns_rec)} im\\'agenes)" if ns_rec else "")) if key == "record"
            else T["title_n"].format(n=key))
    cap = (f"Error de la cota con una misma configuraci\\'on externa a los m\\'etodos (la de producci\\'on de "
           f"MAREA), {what}. Primera fila de cada sitio: cRMSE (m) en los p\\'ixeles de F en los que todos los "
           "m\\'etodos principales tienen valor y, entre par\\'entesis, el sesgo (mediana de producto $-$ verdad); "
           "debajo, el resto del bloque com\\'un, el bloque propio (cobertura: fracci\\'on de F con valor de cada "
           "m\\'etodo; primera escena de la ventana, las n escenas cribadas m\\'as recientes hasta la fecha de la "
           "verdad de cada sitio) y el tiempo. MAREA con retardo reajustado en cada ventana y la regla de calidad "
           f"{MB_HEADLINE}; el resto, con $h_B$ sin retardo. En verde, el menor error de cada sitio. "
           + " ".join(table_notes(rows, meta, "es")))
    if any(m in desc for m, _ in cols):
        cap += " $^\\dagger$ Variante del cat\\'alogo, descriptiva."
    if ni:
        cap += (" Sin implementar todav\\'ia: " + ", ".join(tex_escape(LABELS_ES.get(m, m)) for m in ni) + ".")
    if pend:
        cap += _pending_note(pend, "es")
    lines += [f"\\caption{{{cap}}}", f"\\label{{tab:fijo-{key}}}", "\\end{table}", ""]
    return "\n".join(lines)


#: the scores of every cell in the per-n table CSVs, for both scorings
TABLE_METRICS = ("rmse", "rmse_total", "bias", "slope", "r", "n_units", "coverage")


def table_csv(rows, cols, key, path):
    """The per-n table as CSV: per site its window, the common units and
    their share of F (common_coverage), and per column its status and, for
    each scoring (common, own), cRMSE ('rmse'), RMSE ('rmse_total'), bias,
    slope, r, n and coverage: <method>_<variant>_<scoring>_<metric>; then the
    estimator seconds per method, the shared seconds, host and threads."""
    import io

    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    hdr = ["site", "state", "n_scenes", "first_date", "last_date", "n_common", "n_set", "common_coverage", "reason"]
    for m, v in cols:
        hdr += [f"{m}_{v}_status"] + [f"{m}_{v}_{s}_{x}" for s in ("common", "own") for x in TABLE_METRICS]
    # timing: the estimator seconds per method (one fit gives MAREA's rules), the shared seconds, host, threads
    tcols = [(m, v) for m, v in cols if not is_marea_variant(m, v)]
    hdr += [f"seconds_{m}" for m, _ in tcols] + ["seconds_shared", "host", "threads"]
    wr.writerow(hdr)
    for r in rows:
        line = [r["site"], r["state"], r.get("n", ""), r.get("first_date", ""), r.get("last_date", ""),
                r.get("n_common", ""), r.get("n_set", ""), v14._fmt(r.get("common_coverage", float("nan"))),
                r.get("reason", "")]
        for k in cols:
            c = r.get("cells", {}).get(k, {"status": "absent"})
            line.append(c["status"])
            for s in ("common", "own"):
                sc = c.get(s) or {}
                line += [v14._fmt(sc.get(x, float("nan"))) if x != "n_units" else (sc.get(x, "") if sc else "")
                         for x in TABLE_METRICS]
        line += [v14._fmt(r.get("cells", {}).get(k, {}).get("seconds", float("nan"))) for k in tcols]
        line += [v14._fmt(r.get("seconds_shared", float("nan"))), r.get("host", ""), r.get("threads", "")]
        wr.writerow(line)
    atomic_text(path, buf.getvalue())


def timing_summary(tables_rows):
    """The timing of every (site, n, method) of the tables: {'rows': [...],
    'hosts': {site: host (threads)}, 'rule'}."""
    out, hosts = [], {}
    for key, rows, cols in tables_rows:
        for r in rows:
            if r.get("state") != "ok":
                continue
            hosts[r["site"]] = f"{r.get('host', '')} ({r.get('threads', '')})"
            for (m, v) in cols:
                if is_marea_variant(m, v):
                    continue
                c = r["cells"].get((m, v), {})
                if c.get("status", "absent") == "absent":
                    continue
                out.append({"site": r["site"], "table": str(key), "n_scenes": r.get("n"), "method": m,
                            "status": c.get("status"), "seconds_estimator": c.get("seconds"),
                            "seconds_shared": r.get("seconds_shared"), "host": r.get("host"),
                            "threads": r.get("threads")})
    return {"rule": ("seconds_estimator: the wall time of the method's estimator call alone (time.perf_counter; no "
                     "I/O, no truth loading; MAREA: v14's refit and inversion; DEA Intertidal: its per-pixel steps "
                     "and edge cleaning in its own process pool; NIDEM: phase 2 of the subset plus its share of the "
                     "phase-1 R codes by number of scenes, the cube read excluded); seconds_shared: the site's "
                     "common preprocessing (extraction load, screening, window table; truth loading and gates "
                     "excluded) plus the window's selection"),
            "hosts": hosts, "rows": out}


LATEX_DOC = r"""\documentclass[11pt,a4paper]{article}
\usepackage[spanish,es-noshorthands,es-nolists,es-tabla]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[a4paper,top=2.2cm,bottom=2.2cm,left=1.8cm,right=1.8cm]{geometry}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\definecolor{marea}{RGB}{47,115,80}
\definecolor{muted}{RGB}{86,101,95}
\newcommand{\up}[1]{\textcolor{marea}{\textbf{#1}}}
\newcommand{\sub}[1]{{\footnotesize\color{muted}#1}}
\begin{document}
%s
\end{document}
"""


def tables(args):
    out = args.out
    sites = args.sites or list(SITE_ORDER)
    if not args.no_rescore:
        for s in sites:
            if rescore_site(out, s) is not None:
                log(f"{s}: {out}/{s}.csv rebuilt from the job store")
    df = read_results(out, sites)
    if df.empty:
        raise SystemExit(f"no {out}/<site>.csv to tabulate")
    labels = {}
    for _, r in df[(df["variant"].isin(["main", HEAD_VAR]))].iterrows():
        labels.setdefault(r["method"], r["label"])
    cols = table_columns(df)
    ni = not_implemented(df)
    desc = descriptive_ids(df)
    pend = pending_methods(df, cols)
    tdir = os.path.join(out, "tables")
    os.makedirs(tdir, exist_ok=True)
    ns = parse_ns(args.ns)
    keys = [n for n in ns if (df["n_scenes"].astype(int) == n).any()]
    if (df["is_record"].astype(int) == 1).any():
        keys.append("record")
    meta = {"truth": {s: site_truth(out, s) for s in sites if site_truth(out, s) is not None}}
    md_all, tex_all, paths, trows = [], [], [], []
    for key in keys:
        # a site with survey-year groups: its group rows, then its pooled row where it exists at that n
        rows = table_data(df, table_sites(df, sites, key), key, cols)
        stem = os.path.join(tdir, f"table_{key if key == 'record' else f'n{int(key):03d}'}")
        table_csv(rows, cols, key, stem + ".csv")
        md = table_markdown(rows, cols, key, labels, ni, desc, meta, pend)
        tex = table_latex(rows, cols, key, labels, ni, desc, meta, pend)
        atomic_text(stem + ".md", md)
        atomic_text(stem + "_es.tex", tex)
        md_all.append(md)
        tex_all.append(tex)
        paths.append(stem)
        trows.append((key, rows, cols))
        log(f"-> {stem}.csv/.md/_es.tex")
    atomic_text(os.path.join(tdir, "all_tables.md"), "# Fixed-configuration tables\n\n" + "\n".join(md_all))
    atomic_text(os.path.join(tdir, "all_tables_es.tex"), LATEX_DOC % "\n".join(tex_all))
    write_json(os.path.join(tdir, "timing_summary.json"), timing_summary(trows))
    log(f"-> {tdir}/all_tables.md, {tdir}/all_tables_es.tex, {tdir}/timing_summary.json")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  curves
# ─────────────────────────────────────────────────────────────────────────────

PAL = {"deep": "#0B3B5A", "teal": "#1B7F86", "sand": "#C08A3E", "slate": "#4A5C6B", "ink": "#16202A",
       "rule": "#C9D3DA", "surface": "#FFFFFF"}
#: one fixed style per method id (colour follows the entity, never its
#: rank); the poster palette has four hues, so methods sharing a hue differ
#: by marker and dash (secondary encoding); MAREA is always PDeep
STYLE = {
    "E0b": dict(color=PAL["deep"], marker="o", ls="-"),
    "E1": dict(color=PAL["sand"], marker="^", ls="-"),
    "E3": dict(color=PAL["teal"], marker="s", ls="-"),
    "E2": dict(color=PAL["teal"], marker="D", ls="--"),
    "E4": dict(color=PAL["slate"], marker="v", ls="-"),
    "E5a": dict(color=PAL["sand"], marker="s", ls="--"),
    "E5b": dict(color=PAL["slate"], marker="P", ls="-."),
    "E5c": dict(color=PAL["teal"], marker="X", ls=":"),
    "E6a": dict(color=PAL["sand"], marker="h", ls=":"),
    "E6b": dict(color=PAL["slate"], marker="<", ls=":"),
    "E7": dict(color=PAL["teal"], marker=">", ls="-."),
    "E8": dict(color=PAL["sand"], marker="p", ls=(0, (1, 1, 4, 1))),
    "E12": dict(color=PAL["sand"], marker="*", ls="-."),
    # a declared sensitivity: its method's hue, dashed, hollow markers
    "E1i": dict(color=PAL["sand"], marker="^", ls=(0, (4, 2)), mfc="white"),
}
VARIANT_STYLE = dict(color=PAL["deep"], marker="o", ls=(0, (4, 2)), mfc="white")
_SPARE = [dict(color=c, marker=m, ls=l) for c in (PAL["teal"], PAL["sand"], PAL["slate"])
          for m, l in (("d", "--"), ("H", ":"), ("8", "-."))]


def style_for(mid, var):
    if is_marea_variant(mid, var):
        return dict(VARIANT_STYLE)
    if mid in STYLE:
        return dict(STYLE[mid])
    return dict(_SPARE[zlib.crc32(str(mid).encode()) % len(_SPARE)])


CTEXT = {
    "en": dict(title="Elevation error vs number of screened images, one fixed configuration ({res} m)",
               rmse="Elevation RMSE (m)", cov="Coverage of F (%)", x="Screened images in the window, n",
               no="no results yet", rec="record", unreach="n > {n}: not reachable",
               variant="MAREA, quality rule mb {mb} (variant)",
               sensitivity={"E1i": "NIDEM, ITEM pixel mask (sensitivity)"},
               truth="{t}, {n} px", pending_truth="pending truth (PNOA LiDAR flight date)",
               foot=["One configuration external to the methods (MAREA's production): transition-zone scene "
                     "screening, per-pixel mask SCL 4-7, NDWI, wet = NDWI > 0, the site's production h_B; window = "
                     "the n most recent screened scenes on or before the site's truth date (nested). Vaklodingen "
                     "gives the survey year only: the Dutch truth is the cells surveyed in 2025 and their windows "
                     "end on 31 Dec 2025; at the Wadden, per survey-year group (each pixel's most recent survey in "
                     "2023-2025) the end of its year, a panel per group and a pooled panel where every group's "
                     "window is reachable; Villaviciosa: the PNOA LiDAR flight date.",
                     "MAREA: band lags refitted on each window, quality rule {mb} (the published products' is 0); "
                     "every other method reads h_B without lag, in place of its own tide source. NIDEM: the common "
                     "mask on the whole frame; dashed with hollow markers, a declared sensitivity (NIDEM under "
                     "ITEM's mask, MAREA under another quality rule), never in the common set. {scoring}",
                     "Coverage = each method's own share of the fixed set F (the truth inside the range filter; no "
                     "product decides it)."],
               common=("RMSE median-centred on the common set: the units of F where every method computed at "
                       "that n has a value."),
               own="RMSE median-centred on each method's own valid units of F.",
               pending="Pending: {m}.", notimpl="Not implemented yet: {m}."),
    "es": dict(title="Error de la cota frente al número de imágenes cribadas, con una misma configuración ({res} m)",
               rmse="RMSE de la cota (m)", cov="Cobertura de F (%)", x="Imágenes cribadas en la ventana, n",
               no="sin resultados todavía", rec="registro", unreach="n > {n}: no alcanzable",
               variant="MAREA, regla de calidad mb {mb} (variante)",
               sensitivity={"E1i": "NIDEM, máscara de ITEM (sensibilidad)"},
               truth="{t}, {n} px", pending_truth="verdad pendiente (fecha del vuelo LiDAR PNOA)",
               foot=["Una configuración externa a los métodos (la de producción de MAREA): cribado de escenas en la "
                     "zona de transición, máscara por píxel SCL 4-7, NDWI, mojado = NDWI > 0, h_B de producción del "
                     "sitio; ventana = las n escenas cribadas más recientes hasta la fecha de la verdad del sitio "
                     "(anidadas). Vaklodingen solo da el año del levantamiento: la verdad holandesa son las celdas "
                     "levantadas en 2025 y sus ventanas terminan el 31-12-2025; en Wadden, por grupo de año de "
                     "levantamiento (el más reciente de 2023-2025 de cada píxel) el final de ese año, un panel por "
                     "grupo y uno conjunto donde se alcanzan todas las ventanas; Villaviciosa: la fecha del vuelo "
                     "LiDAR PNOA.",
                     "MAREA: retardos por banda reajustados en cada ventana, regla de calidad {mb} (la de los "
                     "productos publicados es 0); el resto de métodos usa h_B sin retardo en lugar de su propia "
                     "marea. NIDEM: la máscara común en todo el marco; a trazos y con marcadores huecos, una "
                     "sensibilidad declarada (NIDEM con la máscara de ITEM, MAREA con otra regla de calidad), nunca "
                     "en el conjunto común. {scoring}",
                     "Cobertura = fracción del conjunto fijo F (la verdad dentro del filtro de rango; ningún producto "
                     "lo decide) con valor de cada método."],
               common=("RMSE sin sesgo en el conjunto común: los píxeles de F en los que todos los métodos "
                       "calculados con ese n tienen valor."),
               own="RMSE sin sesgo en los píxeles válidos de F de cada método.",
               pending="Pendientes: {m}.", notimpl="Sin implementar todavía: {m}."),
}


def curve_series(d, scoring="common"):
    """{(method, variant): rows sorted by n with rmse (on ``scoring``),
    coverage (own) and whether the window is the record}."""
    out = {}
    ok = d[d["status"] == OK]
    for (mid, var), g in ok.groupby(["method", "variant"]):
        role = str(g["role"].iloc[0])
        # the headline methods, MAREA's other rules and the declared
        # sensitivities with a column (NIDEM under ITEM's mask); never the
        # internal reference or the registry's own sensitivities
        if role != "headline" and not is_side_column(mid, var):
            continue
        pts = []
        for n, gn in g.groupby("n_scenes"):
            a = gn[gn["scoring"] == scoring]
            o = gn[gn["scoring"] == "own"]
            if a.empty:
                continue
            pts.append({"n": int(n), "rmse": _f(a["rmse"].iloc[0]), "bias": _f(a["bias"].iloc[0]),
                        "cov": _f(o["coverage"].iloc[0]) if len(o) else float("nan"),
                        "is_record": int(a["is_record"].iloc[0]) == 1})
        if pts:
            out[(mid, var)] = {"pts": sorted(pts, key=lambda p: p["n"]), "role": role,
                               "label": str(g["label"].iloc[0])}
    return out


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["rule"])
    ax.tick_params(colors=PAL["slate"], labelcolor=PAL["ink"], length=3)
    ax.grid(True, color=PAL["rule"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def method_label(mid, var, lang, label):
    if is_marea_variant(mid, var):
        return CTEXT[lang]["variant"].format(mb=var[2:])
    if mid in CTEXT[lang]["sensitivity"]:
        return CTEXT[lang]["sensitivity"][mid]
    if lang == "es":
        return LABELS_ES.get(mid, label)
    return label if mid not in dict(CATALOGUE) or mid == "E0b" else dict(CATALOGUE)[mid]


def _series_order(k):
    """MAREA first (its headline rule, then its variants), then catalogue
    order, a sensitivity right after its method."""
    after = {s[0]: s[2] for s in SENSITIVITY_COLUMNS}
    base = after.get(k[0], k[0])
    return (base != "E0b", is_marea_variant(*k), CATALOGUE_IDS.index(base) if base in CATALOGUE_IDS else 99,
            base, k[0] in after, k[0])


def draw_site(ar, ac, d, lang, scoring, handles):
    from matplotlib.lines import Line2D

    T = CTEXT[lang]
    series = curve_series(d, scoring)
    order = sorted(series, key=_series_order)
    ends = []
    xs_all = set()
    for k in order:
        s = series[k]
        st = style_for(*k)
        x = np.array([p["n"] for p in s["pts"]], float)
        y = np.array([p["rmse"] for p in s["pts"]], float)
        c = np.array([100 * p["cov"] for p in s["pts"]], float)
        xs_all.update(int(v) for v in x)
        lw = 1.6 if is_side_column(*k) else 2.0
        mfc = st.get("mfc", st["color"])
        ar.plot(x, y, color=st["color"], ls=st["ls"], lw=lw, marker=st["marker"], ms=7.5, mfc=mfc,
                mec="white" if mfc != "white" else st["color"], mew=1.3, zorder=4 if k[0] == "E0b" else 3,
                solid_capstyle="round")
        ac.plot(x, c, color=st["color"], ls=st["ls"], lw=lw, marker=st["marker"], ms=6.0, mfc=mfc,
                mec="white" if mfc != "white" else st["color"], mew=1.1, zorder=3)
        lab = method_label(k[0], k[1], lang, s["label"])
        handles.setdefault(k, Line2D([], [], color=st["color"], ls=st["ls"], lw=lw, marker=st["marker"], ms=7.5,
                                     mfc=mfc, mec="white" if mfc != "white" else st["color"], mew=1.3, label=lab))
        fin = np.flatnonzero(np.isfinite(y))
        if len(fin) and not is_side_column(*k):
            ends.append((float(x[fin[-1]]), float(y[fin[-1]]), lab))
    return ends, sorted(xs_all)


def _end_labels(ax, ends, max_labels=4):
    """Selective direct labels at the right end of each line, in ink; none
    when they would collide (the legend carries identity)."""
    if not ends or len(ends) > max_labels:
        return
    lo, hi = ax.get_ylim()
    ys = sorted(e[1] for e in ends)
    if any(b - a < 0.06 * (hi - lo) for a, b in zip(ys, ys[1:])):
        return
    for x, y, lab in ends:
        ax.annotate(lab, (x, y), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=8.8,
                    color=PAL["ink"])


def curves_figure(df, sites, lang, scoring="common", end_labels=None, meta=None):
    """RMSE (top, on ``scoring``) and coverage of F (bottom) against n, one
    column per site, one line per method (MAREA always PDeep). Direct end
    labels only on a single-site figure (room on the right; the legend
    carries identity everywhere)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    import textwrap

    T = CTEXT[lang]
    n = len(sites)
    one = n == 1
    # a single site: the legend in a column on the right (identity without
    # crowding end labels); several sites: the legend in rows under the title
    lay = (dict(figsize=(11.0, 8.6), left=0.085, right=0.66, top=0.84, bottom=0.235, title_y=0.955, title_fs=14.0,
                leg=dict(loc="upper left", bbox_to_anchor=(0.68, 0.84), ncol=1), foot_w=0.86)
           if one else
           dict(figsize=(4.3 * n + 1.2, 8.4), left=0.06, right=0.985, top=0.80, bottom=0.25, title_y=0.955,
                title_fs=16.5, leg=dict(loc="upper left", bbox_to_anchor=(0.055, 0.94), ncol=4), foot_w=0.92))
    end_labels = False if end_labels is None else bool(end_labels)
    x_hi = 158 if end_labels else 133
    fig = plt.figure(figsize=lay["figsize"])
    gs = fig.add_gridspec(2, n, height_ratios=[3.0, 1.25], hspace=0.10, wspace=0.26, left=lay["left"],
                          right=lay["right"], top=lay["top"], bottom=lay["bottom"])
    handles = {}
    pend, notimpl = set(), set()
    for j, site in enumerate(sites):
        ar = fig.add_subplot(gs[0, j])
        ac = fig.add_subplot(gs[1, j], sharex=ar)
        for ax in (ar, ac):
            _style_axes(ax)
        ar.text(0.0, 1.10, row_title(site, lang), transform=ar.transAxes, fontsize=13.5, color=PAL["deep"],
                weight="bold", va="bottom", ha="left")
        if j == 0:
            ar.set_ylabel(T["rmse"], color=PAL["ink"], fontsize=11.5)
            ac.set_ylabel(T["cov"], color=PAL["ink"], fontsize=11.5)
        ac.set_xlabel(T["x"], color=PAL["ink"], fontsize=10.5)
        d = df[df["site"] == site]
        if d.empty or not (d["status"] == OK).any():
            pend_truth = (site == "villaviciosa" and (d.empty or (d["status"] == PENDING_TRUTH).all()))
            unreach = not d.empty and (d["status"] == NOT_REACHABLE).all()
            msg = (T["pending_truth"] if pend_truth else
                   _missing_text(str(d["reason"].iloc[0]), lang, plain=True) if unreach else T["no"])
            ar.text(0.5, 0.5, textwrap.fill(msg, 34, break_on_hyphens=False), transform=ar.transAxes, ha="center",
                    va="center", fontsize=12, color=PAL["slate"], wrap=True)
            for ax in (ar, ac):
                ax.set_xticks([])
                ax.set_yticks([])
            continue
        pend.update(d[d["status"] == PENDING]["method"].unique().tolist())
        notimpl.update(d[d["status"] == NOT_IMPLEMENTED]["method"].unique().tolist())
        ends, xs = draw_site(ar, ac, d, lang, scoring, handles)
        rec = d[(d["is_record"].astype(int) == 1) & (d["status"] == OK)]
        ticks = sorted(set(int(v) for v in NS) | set(xs))
        rec_n = set(rec["n_scenes"].astype(int))
        ar.xaxis.set_major_locator(FixedLocator([t for t in ticks if t in NS or (t in xs and t not in rec_n)]))
        ar.set_xlim(32, x_hi)
        if len(rec):
            nr = int(rec["n_scenes"].iloc[0])
            ar.axvline(nr, color=PAL["rule"], lw=1.0, zorder=1)
            ar.text(nr, 0.98, f"{T['rec']} ({nr})", transform=ar.get_xaxis_transform(), rotation=90, ha="right",
                    va="top", fontsize=8.5, color=PAL["slate"])
            ac.axvline(nr, color=PAL["rule"], lw=1.0, zorder=1)
        unr = d[d["status"] == NOT_REACHABLE]["n_scenes"].astype(int)
        if len(unr):
            reach = d[d["status"] != NOT_REACHABLE]["n_scenes"].astype(int)
            ar.text(0.98, 0.02, T["unreach"].format(n=int(reach.max()) if len(reach) else 0),
                    transform=ar.transAxes, ha="right", va="bottom", fontsize=8.8, color=PAL["slate"])
        covs = [100 * _f(v) for v in d[(d["scoring"] == "own") & (d["status"] == OK)
                                         & (d["role"] != "internal")]["coverage"]]
        covs = [v for v in covs if np.isfinite(v)] or [100.0]
        ac.set_ylim(min(50.0, 10.0 * math.floor(max(0.0, min(covs)) / 10.0)), 103.0)
        plt.setp(ar.get_xticklabels(), visible=False)
        lo, hi = ar.get_ylim()
        ar.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
        if end_labels:
            _end_labels(ar, ends)
        nset = int(_f(d["n_set"].max()))
        tname = truth_label(site, meta, lang, plain=True)
        ar.text(0.0, 1.025, T["truth"].format(t=tname, n=f"{nset:,}"), transform=ar.transAxes, fontsize=9.5,
                color=PAL["slate"], va="bottom", ha="left")
        ar.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    title = T["title"].format(res="10")
    # bold DejaVu Sans is about 0.6 em per character: wrap to the figure width
    per_line = max(30, int((1.0 - lay["left"] - 0.02) * fig.get_figwidth() * 72 / (0.6 * lay["title_fs"])))
    fig.text(lay["left"], lay["title_y"] + 0.02, textwrap.fill(title, per_line), fontsize=lay["title_fs"],
             color=PAL["deep"], weight="bold", ha="left", va="top", linespacing=1.15)
    order = sorted(handles, key=_series_order)
    if order:
        leg = dict(lay["leg"])
        leg["ncol"] = min(leg["ncol"], len(order))
        fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order], frameon=False,
                   fontsize=10.2, handlelength=2.6, columnspacing=1.8, **leg)
    foot = [t.format(scoring=T[scoring], mb=MB_HEADLINE) for t in T["foot"]]
    pend = [m for m in pend if m not in set(df[df["status"] == OK]["method"])]
    if pend:
        foot.append(T["pending"].format(m=", ".join(method_label(m, "main", lang, m) for m in sorted(pend))))
    ni = [m for m in CATALOGUE_IDS if m in notimpl and m not in set(df[df["status"] == OK]["method"])]
    if ni:
        foot.append(T["notimpl"].format(m=", ".join(method_label(m, "main", lang, m) for m in ni)))
    width = int(lay["foot_w"] * fig.get_figwidth() * 15)
    fig.text(lay["left"], 0.012, "\n".join(textwrap.fill(t, width=width, break_on_hyphens=False) for t in foot),
             fontsize=8.6, color=PAL["slate"], ha="left", va="bottom", linespacing=1.35)
    return fig


def curves(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    sites = args.sites or list(SITE_ORDER)
    if not args.no_rescore:
        for s in sites:
            rescore_site(out, s)
    df = read_results(out, sites)
    if df.empty:
        raise SystemExit(f"no {out}/<site>.csv to draw")
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    meta = {"truth": {s: site_truth(out, s) for s in sites if site_truth(out, s) is not None}}
    paths = []
    for lang in parse_list(args.lang):
        if lang not in CTEXT:
            raise SystemExit(f"--lang: {lang!r} not in {sorted(CTEXT)}")
        # a site with survey-year groups: one panel per group row, then its pooled row where it exists
        panels = table_sites(df, sites)
        fig = curves_figure(df, panels, lang, args.scoring, meta=meta)
        stem = os.path.join(out, f"curves_{args.scoring}_{lang}")
        fig.savefig(stem + ".png", dpi=200)
        fig.savefig(stem + ".pdf")
        plt.close(fig)
        paths.append(stem)
        log(f"-> {stem}.png/.pdf")
        for s in panels:
            if (df["site"] == s).any() and (df[df["site"] == s]["status"] == OK).any():
                fig = curves_figure(df, [s], lang, args.scoring, meta=meta)
                st = os.path.join(out, f"curves_{s}_{args.scoring}_{lang}")
                fig.savefig(st + ".png", dpi=200)
                fig.savefig(st + ".pdf")
                plt.close(fig)
                paths.append(st)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _inputs(p):
    p.add_argument("--ns", default=",".join(map(str, NS)), help="window sizes n (default 125,100,75,60,50,40)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--usable-json", default=None,
                   help="Villaviciosa: the production usable dates as JSON (default: v7's notebook recovery)")
    p.add_argument("--extra-dir", default=EXTRA_DIR,
                   help="where the <site>_extract_<years>.npz extra stores are looked for")
    p.add_argument("--pre2023-levels", choices=("refuse", "production"), default="refuse",
                   help="pre-2023 h_B at gauge sites: refuse (default; the user's choice) or production = the "
                        "production boundary as it is (EOT20 alone where no gauge has data; declared)")
    p.add_argument("--soft-gates", action="store_true",
                   help="a failed gate is recorded instead of stopping (only after reading why)")
    p.add_argument("--product-dir", dest="product_override", default=None,
                   help="folder with the production marea.npz/result.json (e.g. a laptop mirror)")
    p.add_argument("--extract", dest="extract_override", default=None)
    p.add_argument("--comparison-dir", dest="comparison_override", default=None)
    p.add_argument("--nidem-dir", dest="nidem_override", default=None)
    p.add_argument("--cube", dest="cube_override", default=None)
    p.add_argument("--published-csv", default=None)
    p.add_argument("--vva-truth", default=None,
                   help=f"Villaviciosa: the PNOA LiDAR DTM GeoTIFF of the truth (the repository holds {VVA_COB2}); "
                        "without it and --vva-truth-date the row is 'pending truth' (never the RTK)")
    p.add_argument("--vva-truth-date", default=None,
                   help="Villaviciosa: the LiDAR flight date YYYY-MM-DD, the end of every window")
    p.add_argument("--vva-nodelay", default=None,
                   help=f"Villaviciosa: the production no-delay product of the range filter (default {VVA_NODELAY})")
    p.add_argument("--vva-grid", default=None,
                   help="Villaviciosa: a GeoTIFF on the extraction grid (default: pnoa_truth's, the product grid)")
    p.add_argument("--vak-dir", default=None, help="the Vaklodingen sheets (default v1.TRUTH_DIR)")


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m experiments.v19_fixed_config_tables",
                                 description="Elevation methods under one fixed configuration, per site and n.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="every window and method of one site (resumable)")
    r.add_argument("site", choices=list(SITE_NAMES))
    _inputs(r)
    r.add_argument("--seed", type=int, default=DEFAULT_SEED, help="v14's seed (the subsample of F)")
    r.add_argument("--max-px", type=int, default=None,
                   help=f"subsample F above this many units (default {DEFAULT_MAX_PX:,}; 100000 = the units of the "
                        "v14 refit / v15 / v16 runs)")
    r.add_argument("--methods", default=None,
                   help="only these method ids (e.g. E0b,E0a,E3); MAREA on the record window always runs (gate)")
    r.add_argument("--record-window", choices=("all", "marea"), default="all",
                   help="methods on the full-record window: all (default) or only MAREA and no delay")
    r.add_argument("--registry", choices=("auto", "off"), default="auto",
                   help=f"auto: the competitors of {REGISTRY_MODULE}; off: none")
    r.add_argument("--with-variants", action="store_true",
                   help="also run the registry's VARIANTS (declared sensitivities; role 'variant', not headline)")
    r.add_argument("--nidem", choices=("auto", "off"), default="auto")
    r.add_argument("--nidem-func", default=None, help="the v15 function (default: first of "
                                                      + ", ".join(NIDEM_HOOK_NAMES) + ")")
    r.add_argument("--io-lock", default=None,
                   help="NIDEM: a lock file shared by processes reading cubes on one disk (v15's --io-lock)")
    r.add_argument("--nidem-cache-dir", default=None,
                   help="NIDEM: a folder OUTSIDE the repository for the hook's phase-1 arrays (resumable)")
    r.add_argument("--dea-tmp", default=None,
                   help="DEA Intertidal: its temp root, a folder OUTSIDE the repository (default $MAREA_DEA_TMP, "
                        "else the system temp dir; the server's env.sh sets TMPDIR inside the repository, which the "
                        "adapter refuses: the run then stops before any job)")
    r.add_argument("--max-minutes", type=float, default=600.0, help="wall-time cap (resumable)")
    r.add_argument("--tag", default="", help="suffix of the settings JSON (parallel method groups on one site)")
    p = sub.add_parser("plan", help="the screened list and the windows of one site (no pixel data)")
    p.add_argument("site", choices=list(SITE_NAMES))
    _inputs(p)
    t = sub.add_parser("tables", help="one table per n (CSV, markdown, Spanish LaTeX)")
    t.add_argument("--out", default=OUT)
    t.add_argument("--ns", default=",".join(map(str, NS)))
    t.add_argument("--sites", nargs="*", default=None)
    t.add_argument("--no-rescore", action="store_true", help="use the CSVs as they are")
    c = sub.add_parser("curves", help="error vs n per site, one line per method (Spanish and English)")
    c.add_argument("--out", default=OUT)
    c.add_argument("--lang", default="es,en")
    c.add_argument("--scoring", choices=("common", "own"), default="common")
    c.add_argument("--sites", nargs="*", default=None)
    c.add_argument("--no-rescore", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"run": run, "plan": plan, "tables": tables, "curves": curves}[args.cmd](args)


if __name__ == "__main__":
    main()
