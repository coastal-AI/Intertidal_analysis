# -*- coding: utf-8 -*-
"""V19 - elevation methods under ONE fixed external configuration, per site and
number of screened images (the first author's design, 2026-10-20).

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

Windows: Window(site, n) = S[-n:], the n most recent screened scenes, for
n in 125, 100, 75, 60, 50, 40 (``--ns``): a time window flexible per site,
ending 2025-12-31, starting where it holds n screened scenes. Plus the full
production record (n = 97 / 68 / 71 / 54), which is also S[-n:] and is the
gate window. Nested and deterministic (one realisation). A cell is 'not
reachable' when S holds fewer than n scenes.

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
  experiments/v15_nidem_scene_curves.py), else a registry 'E1', else
  'pending'.
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
One fixed unit set per site: v14's F (finite truth after the production range
filter AND finite published MAREA; Villaviciosa: RTK points via
pyintertidal.rtk.load_rtk() with default arguments, inside v14.rtk_units)
with v14's seeded random subsample at the Dutch sites (--max-px, default
200,000; 100,000 gives the units of the v14 refit / v15 / v16 runs; the
subsamples are nested). Each (n, method) is scored (a) on the COMMON set,
the units where EVERY headline method computed at that n has a value, and
(b) on its OWN valid units. Metrics: coverage, median-centred RMSE
(v14.score), total RMSE (uncentred), bias = median(product - truth), OLS
slope, r, n.

Gates (a failed gate stops the run; --soft-gates continues and records it)
---------------------------------------------------------------------------
v14's sha256 (never soft); v14's own gates; F is v14's (units, stored z and
production lags equal the compat block of a v14 settings JSON with the same
units, where one exists); the screened list's 2023-2025 part equals
record_dates and its zero-lag level range equals rango_marea_m; the window
table equals v14's arrays and level table bitwise on the shared scenes; the
record window (MAREA, refitted lags, mb 0) reproduces the stored MAREA on the
units and, at Villaviciosa, the published MAREA point on F.

Extra extraction store (pre-2023 scenes; another builder writes it)
--------------------------------------------------------------------
results/fixedcfg/extra/<site>_extract_<years>.npz with the keys of
marea_extract.npz (Y, C, keep == the main keep, dates, shape) and, in the npz
or in the sidecar <stem>.json: ``overpass`` (date -> ISO label instant),
``cloud_pct`` (date -> production transition-zone %, against the production
reference map) or ``usable`` (dates), and ``gate_2023`` (dict with ``ok``: the
bitwise re-download gate). Only dates before the main extraction's first date
are used. At a site whose h_B uses gauges (the Dutch sites, gauges cached
2023-2025 only) the store is used only with ``--pre2023-levels production``:
the production boundary object as it is, i.e. EOT20 alone where no gauge has
data (declared and counted per window). The default refuses (the user's
choice between the three options of the investigation).

Outputs (new files only, under --out, default results/fixedcfg)
---------------------------------------------------------------
<site>.csv             one row per (window, method, variant, scoring set)
<site>_settings.json   settings, inputs, screening, windows, gates, methods,
                       reference points, timings, memory, versions
<site>_windows.json    the screened list and every window's scenes
<site>_units.npz       the scored units (keep index, truth, unit_col)
<site>_jobs/           one <key>.json + <key>.npz per (n, method): the
                       resumable state (the CSV is rebuilt from it)
tables/                per n: table_<n>.csv / .md / _es.tex, all_tables.md,
                       all_tables_es.tex (``tables``)
curves_<scoring>_<lang>.{png,pdf}, curves_<site>_<scoring>_<lang>.{png,pdf}
                       (``curves``)

Run from the repository root:
    python -X utf8 -m experiments.v19_fixed_config_tables plan villaviciosa
    python -X utf8 -m experiments.v19_fixed_config_tables run villaviciosa
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
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402  (also chdir to ROOT)

OUT = "results/fixedcfg"
EXTRA_DIR = "results/fixedcfg/extra"
NS = (125, 100, 75, 60, 50, 40)
#: quality rules scored from one MAREA fit: 0 = the published products' rule
#: (the headline), 3 = pyintertidal.marea.QUALITY_MIN_BRACKET (a variant)
MBS = (0, 3)
MB_HEADLINE = 0
DEFAULT_MAX_PX = 200_000
DEFAULT_SEED = v14.DEFAULT_SEED
#: v14 as reused (uncommitted on 2026-10-20): imported, never edited
V14_SHA256 = "ef0f9a2e7b4f9b02f78d19fb22bed07a3b91ed5e8a4a59615e8333cfd4de5caf"
#: the production rule of filter_dates: usable iff cloud_pct / 100 <= 0.10
USABLE_CLOUD_MAX = 0.10
#: the span of the production records (v7.TIME_EXTENT)
RECORD_SPAN = ("2023-01-01", "2025-12-31")
#: every window ends here (the scenes counted, the span)
WINDOW_END = "2025-12-31"
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
             "E12": "Horquilla seco/mojado"}
#: the methods the runner computes itself
RUNNER_IDS = ("E0b", "E0a")

OK, PENDING, NOT_COMPUTABLE, NOT_REACHABLE, NOT_IMPLEMENTED, ERROR = (
    "ok", "pending", "not computable", "not reachable", "not implemented", "error")
#: statuses whose job is not recomputed on resume (a deterministic outcome)
CACHEABLE = (OK, NOT_COMPUTABLE)

CSV_COLUMNS = ("site", "n_scenes", "is_record", "first_date", "last_date", "lower_bound_open", "years",
               "scenes_sha1", "method", "label", "variant", "role", "status", "scoring", "n_set", "n_common",
               "n_units", "n_px", "coverage", "rmse", "rmse_total", "bias", "slope", "r", "lo_m", "hi_m",
               "median_clear_obs", "seconds", "source", "fidelity", "notes", "reason")
SITE_ORDER = v14.SITE_ORDER
SITE_NAMES = {"villaviciosa": "Villaviciosa", "escalda": "Westerschelde", "wadden": "Wadden (Vlie)",
              "ems": "Ems-Dollard"}
TRUTH_NAMES = {"villaviciosa": "RTK", "escalda": "Vaklodingen", "wadden": "Vaklodingen", "ems": "Vaklodingen"}

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
    "window": "the n most recent screened scenes, S[-n:], ending 2025-12-31; nested, one realisation",
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


def atomic_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
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
            "plan": os.path.join(out, f"{site}_plan.json")}


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
                  if not p.endswith(".tmp.npz"))


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


def load_extra_store(path, keep, shape, main_dates):
    """One extra extraction: its dates before the main extraction's first
    date with their overpass times and production screening, or a refusal.
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
    if usable is not None:
        us = {str(x)[:10] for x in usable}
        rule = "sidecar/npz 'usable'"
    else:
        us = {d for d, v in cloud.items() if v is not None and np.isfinite(float(v))
              and float(v) / 100.0 <= USABLE_CLOUD_MAX}
        rule = f"cloud_pct / 100 <= {USABLE_CLOUD_MAX} (filter_dates)"
    ov = {}
    for d, v in over.items():
        if v is not None and str(v) not in ("", "nan", "None"):
            ov[d] = pd.Timestamp(v).to_pydatetime()
    info.update(screening_rule=rule, reference_map=side.get("reference_map"))
    store = {"path": path, "dates": dates, "early": early, "usable": us, "overpass": ov, "gate_2023": gate}
    return store, info


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
        st, info = load_extra_store(p, keep, shape, main_dates)
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
                            else "production boundary (global model)"))
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
#  windows
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(S_dates, h0, ns, n_record=None, all_dates=None, end=WINDOW_END):
    """Window(n) = S[-n:] for every n in ``ns`` and the record size, i.e.
    the time window [S[-n], ``end``]. Each window: reachability, its first
    and last screened date, the open lower bound S[-n-1] (any start in
    (S[-n-1], S[-n]] gives the same scenes), scenes per year, the span in
    days (first date to ``end``, inclusive), the cube scenes inside it and
    the pass share, the zero-lag level range, the scenes' sha1; a window
    not reachable says how many screened scenes are missing (and about how
    many years at the site's screened rate)."""
    S = np.asarray(S_dates).astype(str)
    h0 = np.asarray(h0, float)
    T = len(S)
    nyears = len({d[:4] for d in S})
    rate = T / nyears if nyears else 0.0
    alld = None if all_dates is None else np.unique(np.asarray(all_dates).astype(str))
    sizes = {int(v) for v in ns} | ({int(n_record)} if n_record else set())
    out = []
    for n in sorted(sizes, reverse=True):
        w = {"n": int(n), "label": f"n{n}", "is_record": bool(n_record and n == int(n_record)),
             "requested": int(n) in {int(v) for v in ns}}
        if n > T:
            miss = int(n - T)
            yrs = miss / rate if rate else None
            w.update(reachable=False, missing=miss, missing_years=None if yrs is None else round(yrs, 1),
                     reason=(f"S holds {T} screened scenes: {miss} missing"
                             + (f" (about {yrs:.1f} years before {S[0][:4]} at {rate:.1f} screened scenes per "
                                "year)" if yrs is not None and T else "")))
        else:
            a = T - n
            d = S[a:]
            first, last = d[0], d[-1]
            stop = max(str(end), str(last))
            w.update(reachable=True, start=int(a), stop=int(T), first_date=str(first), last_date=str(last),
                     end_date=stop, lower_bound_open=str(S[a - 1]) if a > 0 else None, years=per_year(d),
                     span_days=int((np.datetime64(stop) - np.datetime64(first)).astype(int)) + 1,
                     scenes_sha1=scenes_sha1(d), lo_m=float(h0[a:].min()), hi_m=float(h0[a:].max()))
            if alld is not None:
                k = int(((alld >= first) & (alld <= stop)).sum())
                w.update(cube_scenes=k, pass_share=float(n / k) if k else None)
        out.append(w)
    return out


def check_nested(windows):
    """Reachable windows are nested suffixes of S (by construction; checked)."""
    rw = sorted((w for w in windows if w["reachable"]), key=lambda w: w["n"])
    ok = all(a["start"] >= b["start"] and a["stop"] == b["stop"] for a, b in zip(rw, rw[1:]))
    return {"ok": bool(ok), "sizes": [w["n"] for w in rw]}


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
                registry_errors=dict(getattr(mod, "REGISTRY_ERRORS", {}) or {}))
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


def code_fingerprint(fn, extra=""):
    """sha256 (16 hex) of the file defining ``fn``: a changed method
    recomputes its cached jobs."""
    try:
        f = inspect.getsourcefile(fn)
        h = sha256_file(f)[:16] if f and os.path.exists(f) else "nofile"
    except (TypeError, OSError):
        h = "builtin"
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}:{h}{extra}"


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
    T = len(table.S_dates)
    sl = slice(T - int(w["n"]), T)
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


def method_cfg(ctx, w, mid, pixel_set, needs=()):
    """The second argument of run(obs, cfg): the common configuration, where
    a method may write scratch files and, for a method whose card ``needs``
    it, ``level_series`` = h_B without lag at 1 min from 00:00 of the
    window's first scene day to LEVEL_SERIES_END (exclusive)."""
    work = os.path.join(ctx.paths["work"], w["label"], mid)
    cfg = {"site": ctx.site, "method": mid, "n": int(w["n"]), "window": w["label"], "first_date": w["first_date"],
           "last_date": w["last_date"], "is_record": bool(w["is_record"]), "seed": int(ctx.seed),
           "out_dir": work, "threads": os.environ.get("OMP_NUM_THREADS"), "pixel_set": pixel_set,
           "boundary": ctx.boundary_name, "level_source": "production h_B(t), no lag", "mask": COMMON_MASK,
           "index": "NDWI", "wet_threshold": 0.0, "common": dict(COMMON_CONFIG)}
    if "level_series" in needs:
        full = getattr(ctx, "level_series", None)
        if full is not None:
            import pandas as pd
            T = len(ctx.table.S_dates)
            t0 = pd.Timestamp(ctx.table.S_t[T - int(w["n"])]).normalize()
            cfg["level_series"] = full.slice(t0, LEVEL_SERIES_END)
    return cfg


#: the 1-min level series of a window ends here (exclusive): windows end 2025-12-31
LEVEL_SERIES_END = "2026-01-01"


def level_series_for(ctx, boundary, windows, path):
    """The site's 1-min h_B series (no lag) from 00:00 of the longest
    reachable window's first scene day to LEVEL_SERIES_END, from the cache
    at ``path`` when it covers that span with the same boundary, else built
    with experiments.fixedcfg.level_series.LevelSeries.from_boundary and
    cached. Nested windows slice it."""
    import pandas as pd
    from experiments.fixedcfg.level_series import LevelSeries

    rw = [w for w in windows if w["reachable"]]
    if not rw:
        return None, None
    T = len(ctx.table.S_dates)
    start = pd.Timestamp(ctx.table.S_t[T - max(w["n"] for w in rw)]).normalize()
    end = pd.Timestamp(LEVEL_SERIES_END)
    src = f"{getattr(boundary, 'name', type(boundary).__name__)}, 1-min, no lag"
    if os.path.exists(path):
        try:
            ls = LevelSeries.load(path)
            if ls.t is not None and ls.t[0] <= start and ls.t[-1] >= end - pd.Timedelta(minutes=1) \
                    and ls.source == src:
                return ls, {"cache": rel(path), "reused": True, **ls.summary()}
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

def job_key(n, mid):
    return f"n{int(n):03d}__{mid}"


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
        return sorted(f[:-5] for f in os.listdir(self.root) if f.endswith(".json") and not f.endswith(".tmp.json"))

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
            tmp = zp[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
            os.replace(tmp, zp)
        elif os.path.exists(zp):
            os.remove(zp)
        self._z.pop(key, None)
        v14.write_json(jp, meta)


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
        v14_rows, info = v14.run_job(table, key, len(rows), 0, seed, "refit", list(MBS))
    finally:
        table.pools.pop(key, None)
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
            "function": "experiments.v14_scene_count_curves.run_job(lags='refit') on the window's scenes"}
    return z, meta


def nodelay_job(table, rows):
    """E0a: v14.invert_subset at tau = 0 on every inverted pixel (the same
    inversion, lo/hi = the window's zero-lag range)."""
    rows = np.asarray(rows, np.int64)
    groups = [(0, np.arange(table.Y0.shape[1]))]
    zz, nclear, (lo, hi), t_inv, _ = v14.invert_subset(table.Y0, table.C, table.levels, rows, groups, list(MBS))
    z = {f"mb{mb}": np.asarray(zz[mb], float)[table.unit_col] for mb in MBS}
    meta = {"lo_m": lo, "hi_m": hi, "median_clear_obs": float(np.median(nclear[table.unit_col])),
            "inversion_seconds": round(float(t_inv), 3),
            "function": "experiments.v14_scene_count_curves.invert_subset at tau = 0"}
    return z, meta


VARIANTS = {
    "E0b": {"mb0": {"role": "headline", "label": "MAREA"},
            "mb3": {"role": "variant", "label": "MAREA, quality rule mb 3"}},
    "E0a": {"mb0": {"role": "internal", "label": "no delay"},
            "mb3": {"role": "internal", "label": "no delay, quality rule mb 3"}},
}


class Context(types.SimpleNamespace):
    """What the jobs of one site share."""


def job_meta(ctx, w, mid, status, label, source, fidelity="", variants=None, reason="", seconds=0.0,
             fingerprint="", cacheable=True, info=None):
    return {"key": job_key(w["n"], mid), "site": ctx.site, "n": int(w["n"]), "window": w["label"],
            "is_record": bool(w["is_record"]), "method": mid, "label": label, "status": status, "reason": reason,
            "source": source, "fidelity": fidelity, "variants": variants or {}, "scenes_sha1": w["scenes_sha1"],
            "units_sha1": ctx.units_sha1, "fingerprint": fingerprint, "cacheable": bool(cacheable),
            "seconds": round(float(seconds), 2), "finished": _dt.datetime.now().isoformat(timespec="seconds"),
            "peak_rss_mb": peak_rss_mb(), "info": info or {}}


def run_one(ctx, w, mid):
    """Compute one (window, method): returns (meta, arrays or None)."""
    T = len(ctx.table.S_dates)
    rows = np.arange(T - int(w["n"]), T, dtype=np.int64)
    t0 = time.time()
    if mid in ("E0b", "E0a"):
        fp = f"v14:{V14_SHA256[:16]}|job:{JOB_VERSION}"
        if mid == "E0b":
            z, info = marea_job(ctx.table, rows, ctx.seed)
            src = "v14.run_job(lags='refit'): production inversion, band lags refitted on the window"
        else:
            z, info = nodelay_job(ctx.table, rows)
            src = "v14.invert_subset at tau = 0 (internal reference)"
        meta = job_meta(ctx, w, mid, OK, VARIANTS[mid]["mb0"]["label"], src, "F0", VARIANTS[mid],
                        seconds=time.time() - t0, fingerprint=fp, info=info)
        return meta, {f"z__{k}": v for k, v in z.items()}
    if mid == "E1" and ctx.nidem_fn is not None:
        return nidem_one(ctx, w, rows, t0)
    ent = ctx.registry.get(mid)
    if ent is None:
        if mid == "E1":
            return job_meta(ctx, w, mid, PENDING, "NIDEM", "v15 hook",
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
    obs = window_obs(ctx.table, ctx, w, ps)
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
    try:
        res = fn(obs, cfg)
        st, z, sig, rmeta, reason = parse_method_result(res, len(obs.keep))
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ == "NotComputable":      # experiments.fixedcfg.elev_base.NotComputable
            return job_meta(ctx, w, mid, NOT_COMPUTABLE, label, "registry", fid, variants, str(e)[:500],
                            time.time() - t0, fp, info=info), None
        tb = traceback.format_exc(limit=6)
        log(f"[{ctx.site}] {w['label']} {mid}: ERROR {e!r}\n{tb}")
        return job_meta(ctx, w, mid, ERROR, label, "registry", fid, reason=repr(e)[:500], seconds=time.time() - t0,
                        fingerprint=fp, cacheable=False, info=dict(info, traceback=tb[-4000:])), None
    info["method_meta"] = {k: v for k, v in rmeta.items() if k != "extra"}
    if st != OK:
        return job_meta(ctx, w, mid, st, label, "registry", fid, variants, reason, time.time() - t0, fp,
                        info=info), None
    arrays = {"z__main": to_units(z, ps, ctx)}
    if sig is not None:
        arrays["sigma__main"] = to_units(sig, ps, ctx)
    info["finite_share_of_P"] = float(np.isfinite(z).mean())
    return job_meta(ctx, w, mid, OK, label, "registry", fid, variants, "", time.time() - t0, fp, info=info), arrays


def nidem_one(ctx, w, rows, t0):
    """E1 through the v15 hook on the window's scenes."""
    fn, hinfo = ctx.nidem_fn, ctx.nidem_info
    fp = f"{hinfo.get('function')}:{(hinfo.get('file') or {}).get('sha256', '?')[:16]}"
    src = f"v15 hook {hinfo.get('function')}"
    obs = window_obs(ctx.table, ctx, w, "keep")
    S = ctx.S
    main = bool(np.all(np.asarray(S["src"])[rows] == 0))
    avail = {"site": ctx.site, "dates": obs.dates,
             "rows": np.asarray(S["row"], np.int64)[rows] if main else None,
             "t_real": obs.t, "h": np.asarray(obs.h, float), "units_keep": np.asarray(ctx.units_keep, np.int64),
             "px_keep": np.asarray(ctx.px, np.int64), "out_dir": os.path.join(ctx.paths["work"], w["label"], "E1"),
             "mask": COMMON_MASK, "obs": obs, "frame": obs.frame, "n": int(w["n"]), "C": obs.C, "Y": obs.Y}
    try:
        res, used = call_accepting(fn, avail)
        if isinstance(res, dict) and str(res.get("status", "ok")).strip().upper().startswith("NOT"):
            return job_meta(ctx, w, "E1", NOT_COMPUTABLE, "NIDEM", src, "F2", {"main": {"role": "headline",
                            "label": "NIDEM"}}, str(res.get("reason", "")), time.time() - t0, fp,
                            info={"keywords": used}), None
        z = hook_values(res, len(ctx.unit_col), ctx.unit_col, ctx.units_keep, len(ctx.px), len(ctx.keep_flat))
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        log(f"[{ctx.site}] {w['label']} NIDEM hook: ERROR {e!r}")
        return job_meta(ctx, w, "E1", ERROR, "NIDEM", src, "F2", reason=repr(e)[:500], seconds=time.time() - t0,
                        fingerprint=fp, cacheable=False, info={"traceback": tb[-4000:]}), None
    mask_note = ("the common mask was passed (mask/C/obs)" if any(k in used for k in ("mask", "C", "obs"))
                 else "DECLARED DEVIATION: the hook takes no mask; it applies its own pixel rule")
    info = {"keywords": used, "mask": mask_note,
            "hook_meta": {k: v for k, v in (res.items() if isinstance(res, dict) else [])
                          if not isinstance(v, np.ndarray)}}
    if not np.isfinite(z).any():
        return job_meta(ctx, w, "E1", NOT_COMPUTABLE, "NIDEM", src, "F2", reason="no finite estimate on the units",
                        seconds=time.time() - t0, fingerprint=fp, info=info), None
    return (job_meta(ctx, w, "E1", OK, "NIDEM", src, "F2", {"main": {"role": "headline", "label": "NIDEM"}}, "",
                     time.time() - t0, fp, info=info), {"z__main": z})


def method_fingerprint(ctx, mid):
    """The fingerprint a cached job must carry to be reused (None = never
    cached: not built / pending)."""
    if mid in ("E0b", "E0a"):
        return f"v14:{V14_SHA256[:16]}|job:{JOB_VERSION}"
    if mid == "E1" and ctx.nidem_fn is not None:
        return f"{ctx.nidem_info.get('function')}:{(ctx.nidem_info.get('file') or {}).get('sha256', '?')[:16]}"
    ent = ctx.registry.get(mid)
    if ent is None:
        return None
    fn, _ = resolve_run(ent["run"])
    if fn is None:
        return None
    return code_fingerprint(fn, f"|{pixel_set_of(ent['card'])}|runner:{COMPETITOR_JOB_VERSION}")


def job_plan(windows, methods, record_window="all"):
    """(n, method) in run order: MAREA on the record window first (the
    gate), then method-major (MAREA on every window, no delay, each
    competitor, NIDEM), each over the windows from the largest n down. The
    record window is computed for every method unless ``record_window`` is
    'marea' (then only MAREA and no delay)."""
    rw = [w for w in windows if w["reachable"]]
    rec = [w for w in rw if w["is_record"]]
    others = sorted((w for w in rw if not w["is_record"]), key=lambda w: -w["n"])
    plan = [(rec[0]["n"], "E0b")] if rec else []
    for mid in methods:
        ws = list(others)
        if rec and (mid in RUNNER_IDS or record_window == "all" or rec[0]["requested"]):
            ws = rec + ws
        for w in ws:
            if (w["n"], mid) not in plan:
                plan.append((w["n"], mid))
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
    for c in ("coverage", "rmse", "rmse_total", "bias", "slope", "r", "median_clear_obs", "seconds"):
        r[c] = float("nan")
    r.update(kw)
    return r


def _notes(mid, meta):
    info = meta.get("info") or {}
    if mid == "E0b" and info.get("tau_used_bands_min") is not None:
        return "tau bands " + "/".join(f"{v:.1f}" for v in info["tau_used_bands_min"]) + " min"
    if mid == "E1" and info.get("mask"):
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


def build_rows(site, kind, truth, unit_col, windows, store, known=()):
    """Every row of the per-site CSV from the job store: per reachable
    window the common set (units where every headline method computed at
    that n has a value) and each method's scores on it and on its own
    units; rows for pending / not computable / error jobs and for the
    catalogue methods with no job ('not implemented'); per unreachable
    window one 'not reachable' row per method."""
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
    rows = []
    for w in sorted(windows, key=lambda w: -int(w["n"])):
        wc = _window_cols(site, w)
        if not w.get("reachable"):
            for mid in method_ids:
                rows.append(_blank(**wc, method=mid, label=labels.get(mid, mid), status=NOT_REACHABLE,
                                   reason=w.get("reason", "")))
            continue
        jobs = {m["method"]: m for m in metas.values()
                if int(m["n"]) == int(w["n"]) and m.get("scenes_sha1") == w["scenes_sha1"]}
        members = []
        for mid, m in jobs.items():
            if m["status"] != OK:
                continue
            arr = store.arrays(m["key"]) or {}
            for var, vinfo in (m.get("variants") or {}).items():
                z = arr.get(f"z__{var}")
                if z is None or z.shape != truth.shape:
                    continue
                vinfo = dict(vinfo)
                if vinfo.get("role") not in ROLES:          # a card's descriptive text: a column
                    vinfo.setdefault("catalogue_role", vinfo.get("role"))
                    vinfo["role"] = "headline"
                members.append((mid, var, vinfo, np.asarray(z, float), m))
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
            base = dict(wc, method=mid, label=vinfo.get("label", m.get("label", mid)), variant=var,
                        role=vinfo.get("role", ""), status=OK, n_set=int(len(truth)), n_common=n_common,
                        seconds=m.get("seconds", float("nan")), source=m.get("source", ""),
                        fidelity=m.get("fidelity", ""), notes=note, reason=f"common set of {members_in}")
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
        for mid, m in jobs.items():
            if m["status"] == OK:
                continue
            rows.append(_blank(**wc, method=mid, label=m.get("label", mid), status=m["status"],
                               n_set=int(len(truth)), n_common=n_common, seconds=m.get("seconds", float("nan")),
                               source=m.get("source", ""), fidelity=m.get("fidelity", ""), reason=m.get("reason", ""),
                               role=next(iter((m.get("variants") or {}).values()), {}).get("role", "headline")))
        for mid in CATALOGUE_IDS:
            if mid not in jobs:
                rows.append(_blank(**wc, method=mid, label=labels.get(mid, mid), status=NOT_IMPLEMENTED,
                                   role="headline", n_set=int(len(truth)), n_common=n_common,
                                   reason="no code under the common configuration yet"))
    return rows


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
    tmp = path[:-4] + ".tmp.npz"
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
    with open(p["windows"], encoding="utf-8") as f:
        wj = json.load(f)
    store = store if store is not None else JobStore(p["jobs"], create=False)
    rows = build_rows(site, str(u["kind"]), u["truth"], u["unit_col"], wj["windows"], store)
    write_csv(p["csv"], rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  gates
# ─────────────────────────────────────────────────────────────────────────────

def record_gate(site, kind, truth, unit_col, z0, z_stored, tau_bands=None, tau_stored=None, published=None,
                full_F=False):
    """The record window, MAREA with refitted lags and mb 0, reproduces the
    stored MAREA on the units (same valid set up to v14's flip tolerance,
    |dz| <= TOL_Z, the same scores) and, where the units are the whole F
    (Villaviciosa), the published MAREA point (n, RMSE, slope, r, bias)."""
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
        g["note"] = "no published CSV found: compared with the stored product only"
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


def fixed_set_gate(site, mine):
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


def windows_doc(site, S, s_info, windows, extra_info, provenance):
    T = len(S["dates"])
    ws = []
    for w in windows:
        d = dict(w)
        if w["reachable"]:
            a = w["start"]
            d["dates"] = [str(x) for x in S["dates"][a:]]
            d["sources"] = {str(k): int(v) for k, v in zip(*np.unique(np.asarray(S["src"])[a:], return_counts=True))}
            d["extraction_rows_main"] = [int(r) for r, s in zip(S["row"][a:], S["src"][a:]) if int(s) == 0]
            if provenance is not None:
                pa = provenance["all_members"][a:]
                d["levels_all_members"] = int(pa.sum())
                d["levels_partial"] = int((~pa).sum())
        ws.append(d)
    return {"site": site, "purpose": "the screened list S and the windows S[-n:] of the fixed-configuration tables",
            "S": {"n": int(T), "first": str(S["dates"][0]) if T else None, "last": str(S["dates"][-1]) if T else None,
                  "per_year": per_year(S["dates"]), "dates": [str(x) for x in S["dates"]],
                  "source": [int(s) for s in S["src"]], "row": [int(r) for r in S["row"]],
                  "t": [str(x) for x in S["t"]], "h0_m": [float(x) for x in S["h0"]],
                  "sha1": scenes_sha1(S["dates"]), "rule": s_info.get("rule")},
            "extra_stores": extra_info, "windows": ws}


def run(args):
    from pyintertidal import marea

    site, out = args.site, args.out
    v14.guard_out(out)
    os.makedirs(out, exist_ok=True)
    paths = site_paths(out, site, args.tag)
    deadline = _T0 + float(args.max_minutes) * 60.0
    soft = bool(args.soft_gates)
    gates = {}
    g = v14_sha_gate()
    gates["v14_sha256"] = dict(g, stop_if_failed=True)
    if not g["ok"]:
        raise SystemExit(f"v14 changed: sha256 {g['sha256']} != {V14_SHA256}; refusing to run (never soft)")
    cfg, overrides = site_cfg(args)
    ns = parse_ns(args.ns)
    seed = int(args.seed)
    max_px = DEFAULT_MAX_PX if args.max_px is None else int(args.max_px)
    log(f"[{site}] v19 fixed-configuration tables: n {ns}, rules mb {list(MBS)} (headline mb {MB_HEADLINE}), "
        f"seed {seed}, max px {max_px:,}, max {args.max_minutes} min; threads {v14.blas_info()}")

    # 1. v14's site, gates and reference points
    t0 = time.time()
    sd14, meta, extras = v14.load_site(site, cfg, "refit", max_px, seed, ["screened"])
    load_s = time.time() - t0
    t0 = time.time()
    gates14, ok14, tim14 = v14.run_gates(site, cfg, sd14, extras, list(MBS), seed, "refit", [])
    sd14.refit_cache.clear()
    refs = v14.reference_points(site, cfg, sd14, extras)
    gates14_s = time.time() - t0
    for k, v in gates14.items():
        log(f"[{site}] v14 gate {k}: {'OK' if v.get('ok', True) else 'FAILED'}")

    # 2. the scored units, compat
    units_keep = np.asarray(extras["units_keep"], np.int64)
    mine = {"seed": seed, "n_units_scored": int(len(sd14.unit_col)), "n_units_F": int(len(extras["unit_keep_full"])),
            "units_sha1": v14.array_sha1(units_keep), "z_stored_sha1": v14.array_sha1(extras["z_stored"]),
            "tau_fixed_sha1": v14.array_sha1(sd14.tau_fixed)}
    compat = {"version": 1, "site": site, "max_px": max_px, "mbs": list(MBS), **mine}
    gates["fixed_set_is_v14s"] = fixed_set_gate(site, mine)
    for p in (paths["settings"], os.path.join(out, f"{site}_settings.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                prev = json.load(f).get("compat")
            if prev is not None and prev != v14._clean(compat):
                raise SystemExit(f"{p} was written with other settings:\n  {prev}\nnow:\n  {compat}\n"
                                 "refusing to mix them: use another --out")
    z_stored_units = np.asarray(extras["z_stored"], float)[units_keep]
    save_units(paths["units"], units_keep, sd14.truth, sd14.unit_col, extras["px"], sd14.kind,
               extra={"z_stored_units": z_stored_units})

    # 3. the screened list S and the windows
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
    windows = make_windows(S["dates"], S["h0"], ns, len(rec_dates), all_dates)
    gates["windows_nested"] = check_nested(windows)
    for w in windows:
        log(f"[{site}] window {w['label']:5s}" + (" (record)" if w["is_record"] else "") + ": "
            + (f"{w['first_date']}..{w['last_date']}, {w['years']}, {w.get('cube_scenes')} cube scenes"
               if w["reachable"] else f"NOT REACHABLE: {w['reason']}"))

    # 4. methods
    registry, reg_info = ({}, {"status": "off"}) if args.registry == "off" else \
        load_registry(variants=args.with_variants)
    log(f"[{site}] registry {REGISTRY_MODULE}: {reg_info.get('status')} "
        f"{reg_info.get('methods') or reg_info.get('reason', '')}"
        + (f", variants {reg_info.get('variants')}" if reg_info.get("variants") else ""))
    for k, v in (reg_info.get("registry_errors") or {}).items():
        log(f"[{site}] registry error in {k}: {v}")
    nidem_fn, nidem_info = (None, {"status": "off", "reason": "--nidem off"}) if args.nidem == "off" \
        else find_nidem_hook(args.nidem_func)
    if nidem_fn is None and "E1" in registry:
        nidem_info = dict(nidem_info, fallback="registry E1")
    log(f"[{site}] NIDEM: {nidem_info.get('status')} {nidem_info.get('function') or nidem_info.get('reason', '')}")
    comps = [m for m in registry if m not in RUNNER_IDS and m != "E1"]
    methods = ["E0b", "E0a"] + comps + ["E1"]
    if args.methods:
        want = set(parse_list(args.methods))
        methods = [m for m in methods if m in want]
    runnable = [m for m in methods if m not in RUNNER_IDS and
                ((m == "E1" and nidem_fn is not None) or
                 (m in registry and resolve_run(registry[m]["run"])[0] is not None))]
    need_keep = bool(runnable)

    # 5. the window table
    pos14 = {d: i for i, d in enumerate(np.asarray(extras["dates_all"]).astype(str))}
    m2a_cols, m2a_band = lag_fit_columns(cfg, np.asarray(res["centros_km"], float))
    gates["lagfit_bands_are_v14s"] = {"ok": bool(np.array_equal(m2a_band, sd14.m2a["band"]))}
    table, t_info = build_table(site, sources, S, extras["px"], m2a_cols, len(extras["keep"]), boundary, sd14,
                                need_keep, pos14)
    gates["window_table_vs_v14"] = compare_with_v14(table, sd14, S, pos14)
    dl = float(np.max(np.abs(table.levels[0] - S["h0"]))) if len(S["h0"]) else 0.0
    gates["window_table_levels_vs_screening"] = {"max_abs_diff_m": dl, "ok": dl <= v14.TOL_LEVEL}
    prov = level_provenance(boundary, S["t"])
    v14.write_json(paths["windows"], windows_doc(site, S, s_info, windows, x_info, prov))
    for w in windows:
        if w["reachable"]:
            T = len(S["dates"])
            sl = slice(T - w["n"], T)
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
                  level_series=None)
    store = JobStore(paths["jobs"])
    ls_info = None
    if any("level_series" in (registry[m]["card"].get("needs") or []) for m in runnable if m in registry):
        ctx.level_series, ls_info = level_series_for(ctx, boundary, windows,
                                                     os.path.join(out, f"{site}_level_series.npz"))

    # 6. settings
    settings = {
        "site": site, "site_name": SITE_NAMES.get(site, site), "status": "running", "compat": compat,
        "question": ("elevation methods under one fixed external configuration (MAREA's production), per site "
                     "and number of screened images n"),
        "git_head": v14.git_head(), "script": v14.file_info(rel(__file__)),
        "v14_script": v14.file_info(rel(v14.__file__)), "python": sys.version.split()[0], "numpy": np.__version__,
        "blas": v14.blas_info(), "seed": seed, "max_px": max_px, "ns": ns, "input_overrides": overrides,
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
            "level_series_rule": (f"h_B without lag at 1 min from 00:00 of the window's first scene day to "
                                  f"{LEVEL_SERIES_END} (exclusive), sliced from one series per site")},
        "scoring": {"F": meta.get("fixed_set"), "common": ("units of F (subsample) where every headline method "
                                                           "computed at that n has a value"),
                    "own": "each method's own finite units of F (subsample)",
                    "metrics": ("v14.score: median-centred RMSE, OLS slope, r, bias = median(product - truth); "
                                "rmse_total uncentred; coverage = finite share of the scored units")},
        "v14_gates": gates14, "gates": gates, "reference_points_v14": refs,
        "timing": {"load_seconds": round(load_s, 1), "v14_gates_seconds": round(gates14_s, 1), **tim14,
                   "table_seconds": t_info["seconds"]},
        "inputs": meta.get("inputs"), "lag_fit": meta.get("lag_fit"), "boundary": meta.get("boundary"),
        "caveats": [
            "n is confounded with time span and season: nested deterministic windows, one realisation each",
            "small windows shrink the level range (lo/hi), which clamps MAREA and shifts interval methods",
            "F is conditioned on the published MAREA (finite stored z)",
            "MAREA reads a clear observation with undefined NDWI as 0 (dry); other methods see NaN",
            "label instants (datatake start) bias every level by 0-10 min (kept by the user on 2026-10-14)",
        ],
        "runs": [],
    }
    if os.path.exists(paths["settings"]):
        with open(paths["settings"], encoding="utf-8") as f:
            settings["runs"] = json.load(f).get("runs", [])
    this_run = {"started": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:]}
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
    by_n = {w["n"]: w for w in windows if w["reachable"]}
    n_rec = len(rec_dates)
    status, timing = {}, {}
    stopped = None
    for n, mid in plan:
        w = by_n[n]
        key = job_key(n, mid)
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
            timing[key] = m["seconds"]
            rescore_site(out, site, store)
            msg = f"[{site}] {w['label']:5s} {mid:5s}: {m['status']}"
            if m["status"] == OK and arrays:
                for var in m["variants"]:
                    s = score_values(ctx.table.kind, ctx.table.truth, arrays[f"z__{var}"], ctx.unit_col)
                    msg += f" | {var} RMSE {s['rmse']:.4f} cov {s['coverage']:.3f} n {s['n']:,}"
            elif m.get("reason"):
                msg += f" ({m['reason'][:160]})"
            log(msg + f" | {m['seconds']:.1f} s" + (f" | peak {peak_rss_mb() / 1024:.2f} GB" if peak_rss_mb() else ""))
        if n == n_rec and mid == "E0b":
            z0 = store.arrays(key)["z__mb0"]
            gg = record_gate(site, ctx.table.kind, ctx.table.truth, ctx.unit_col, z0, z_stored_units,
                             m["info"].get("tau_bands_min"), res.get("tau_min"),
                             v14.published_row(site, cfg.get("published"))[0],
                             full_F=len(extras["unit_keep_full"]) == len(units_keep))
            gates["record_window_marea_mb0_reproduces_published"] = gg
            log(f"[{site}] gate record_window_marea_mb0_reproduces_published: {'OK' if gg['ok'] else 'FAILED'} "
                f"(flips {gg['validity_flips']}, max |dz| {gg['max_abs_dz_m']:.2e} m, max |score diff| "
                f"{gg['max_abs_score_diff']:.2e}" + (f", vs published {gg['max_abs_diff_vs_published']:.2e}"
                                                    if "max_abs_diff_vs_published" in gg else "") + ")")
            if not gg["ok"]:
                if not soft:
                    finish("gate_failed")
                    raise SystemExit(f"[{site}] the record window does not reproduce the published MAREA; see "
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
        if n == n_rec and mid == "E0a" and getattr(ctx, "z_plain", None) is not None:
            za = store.arrays(key)["z__mb0"]
            fa, fb = np.isfinite(za), np.isfinite(ctx.z_plain)
            both = fa & fb
            gates["record_window_no_delay_vs_stored_z_plain"] = {
                "validity_flips": int(np.sum(fa != fb)), "stop_if_failed": False,
                "max_abs_dz_m": float(np.max(np.abs(za[both] - ctx.z_plain[both]))) if both.any() else None,
                "ok": bool(np.sum(fa != fb) <= v14.TOL_FLIP_SHARE * len(za)
                           and (not both.any() or np.max(np.abs(za[both] - ctx.z_plain[both])) <= v14.TOL_Z))}
            settings["gates"] = gates
        settings["progress"] = {"last": key, "status": status[key], "jobs_planned": len(plan),
                                "jobs_seen": len(status)}
        settings["timing"]["jobs_seconds"] = timing
        v14.write_json(paths["settings"], settings)
    rows = rescore_site(out, site, store) or []
    n_ok = sum(1 for v in status.values() if v in ("resumed", OK))
    st = "stopped_by_time" if stopped else "complete"
    settings["progress"] = {"jobs_planned": len(plan), "jobs_done_or_resumed": n_ok, "stopped": stopped,
                            "status_per_job": status}
    finish(st)
    log(f"[{site}] {st}: {len(plan)} jobs planned, {n_ok} ok/resumed -> {paths['csv']} ({len(rows)} rows), "
        f"{paths['settings']}")
    head = [r for r in rows if r["scoring"] == "common" and r["role"] == "headline" and r["status"] == OK]
    for r in sorted(head, key=lambda r: (-int(r["n_scenes"]), r["method"])):
        log(f"[{site}]   n={int(r['n_scenes']):4d} {r['method']:5s} common RMSE {r['rmse']:.4f} bias "
            f"{r['bias']:+.3f} (n {int(r['n_units']):,} of {int(r['n_set']):,})")


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
    windows = make_windows(S["dates"], S["h0"], ns, len(rec_dates), list(ext_dates))
    doc = windows_doc(site, S, s_info, windows, x_info, level_provenance(rec["boundary_obj"], S["t"]))
    doc.update(gates=gates, screening=s_info, input_overrides=overrides)
    v14.write_json(paths["plan"], doc)
    print(f"\n{site}: S = {len(S['dates'])} screened scenes {S['dates'][0]}..{S['dates'][-1]} "
          f"{per_year(S['dates'])}")
    for k, v in gates.items():
        print(f"  gate {k}: {'OK' if v.get('ok') else 'FAILED'}")
    for w in windows:
        tag = " (record)" if w["is_record"] else ""
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
           "not reachable": "not reachable", "variant": "MAREA, rule mb 3",
           "missing": "{k} screened scenes missing (about {y} years before {y0})",
           "title_n": "n = {n} screened images per site", "title_rec": "the full production record per site"},
    "es": {"site": "Sitio \\sub{(verdad)}", "from": "Desde", "n": "n", "common": "Comunes", "points": "puntos",
           "px": "px", "coverage": "cobertura de F", "pending": "pendiente", "not computable": "no calculable",
           "error": "error", "not implemented": "sin implementar", "no results": "sin resultados todav\\'ia",
           "not reachable": "no alcanzable", "variant": "MAREA \\sub{regla 3}",
           "missing": "faltan {k} escenas cribadas (unos {y} a\\~nos antes de {y0})",
           "title_n": "n = {n} im\\'agenes cribadas por sitio", "title_rec": "el registro de producci\\'on completo"},
}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def read_results(out, sites):
    import pandas as pd

    parts = []
    for s in sites:
        p = os.path.join(out, f"{s}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            parts.append(pd.read_csv(p, keep_default_na=False, na_values=["nan", "NaN", ""]))
    if not parts:
        return pd.DataFrame(columns=list(CSV_COLUMNS))
    df = pd.concat(parts, ignore_index=True)
    for c in ("site", "first_date", "last_date", "lower_bound_open", "years", "scenes_sha1", "method", "label",
              "variant", "role", "status", "scoring", "source", "fidelity", "notes", "reason"):
        df[c] = df[c].fillna("").astype(str)
    return df


def table_columns(df):
    """The method columns of every table (the same in all): MAREA, then the
    other headline methods in catalogue order (registry ids not in the
    catalogue after them) that have at least one row other than 'not
    implemented'; the MAREA mb-3 variant last. Returns [(method, variant)]."""
    have = df[(df["role"] == "headline") & (df["status"] != NOT_IMPLEMENTED) & (df["status"] != NOT_REACHABLE)]
    ids = set(have["method"])
    order = [m for m in CATALOGUE_IDS if m in ids] + sorted(m for m in ids if m not in CATALOGUE_IDS)
    cols = [("E0b", "mb0")] + [(m, "main") for m in order if m != "E0b"]
    if ((df["method"] == "E0b") & (df["variant"] == "mb3")).any():
        cols.append(("E0b", "mb3"))
    return cols


def not_implemented(df):
    ni = df[(df["status"] == NOT_IMPLEMENTED)]["method"].unique().tolist()
    done_ = set(df[(df["status"] != NOT_IMPLEMENTED) & (df["status"] != NOT_REACHABLE)]["method"])
    return [m for m in CATALOGUE_IDS if m in ni and m not in done_]


def table_data(df, sites, key, cols):
    """One table: per site its window, the common units and per column the
    common-set RMSE and bias, the own coverage, or the status."""
    rows = []
    for site in sites:
        d = df[df["site"] == site]
        if key == "record":
            d = d[d["is_record"].astype(int) == 1]
        else:
            d = d[d["n_scenes"].astype(int) == int(key)]
        r = {"site": site, "cells": {}}
        if d.empty:
            r["state"] = "no results"
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
        for mid, var in cols:
            dm = d[(d["method"] == mid) & ((d["variant"] == var) | (d["variant"] == ""))]
            c = {"status": "absent"}
            if not dm.empty:
                okc = dm[(dm["status"] == OK) & (dm["scoring"] == "common") & (dm["variant"] == var)]
                own = dm[(dm["status"] == OK) & (dm["scoring"] == "own") & (dm["variant"] == var)]
                if len(okc):
                    a = okc.iloc[0]
                    c = {"status": OK, "rmse": _f(a["rmse"]), "bias": _f(a["bias"]), "slope": _f(a["slope"]),
                         "r": _f(a["r"]), "rmse_total": _f(a["rmse_total"]), "n_units": int(_f(a["n_units"])),
                         "coverage": _f(own.iloc[0]["coverage"]) if len(own) else float("nan"),
                         "n_own": int(_f(own.iloc[0]["n_units"])) if len(own) else 0,
                         "rmse_own": _f(own.iloc[0]["rmse"]) if len(own) else float("nan")}
                else:
                    c = {"status": str(dm["status"].iloc[0]), "reason": str(dm["reason"].iloc[0])}
            r["cells"][(mid, var)] = c
        heads = [(k, c) for k, c in r["cells"].items() if k[1] != "mb3" and c["status"] == OK
                 and np.isfinite(c["rmse"])]
        r["best"] = min(heads, key=lambda kc: kc[1]["rmse"])[0] if len(heads) >= 2 else None
        rows.append(r)
    return rows


def col_label(mid, var, lang, labels):
    if (mid, var) == ("E0b", "mb3"):
        return TXT[lang]["variant"] if lang == "es" else "MAREA (rule mb 3)"
    if lang == "es":
        return tex_escape(LABELS_ES.get(mid, labels.get(mid, mid)))
    return dict(CATALOGUE).get(mid) or labels.get(mid, mid)


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


def _missing_text(reason, lang):
    """'... 57 missing (about 2.5 years before 2023 ...' -> the table text."""
    import re

    m = re.search(r"(\d+) missing(?: \(about ([\d.]+) years before (\d{4}))?", reason or "")
    if not m:
        return TXT[lang]["not reachable"]
    k, y, y0 = m.group(1), m.group(2), m.group(3)
    if y is None:
        return f"{TXT[lang]['not reachable']}: {k}"
    return f"{TXT[lang]['not reachable']}: " + TXT[lang]["missing"].format(k=k, y=y, y0=y0)


def table_markdown(rows, cols, key, labels, ni, desc=()):
    T = TXT["en"]
    title = T["title_rec"] if key == "record" else T["title_n"].format(n=key)
    head = ["Site (truth)", "From", "Common units"] + [
        col_label(m, v, "en", labels) + (" †" if m in desc and v != "mb3" else "") for m, v in cols]
    if key == "record":
        head.insert(1, "n")
    out = [f"### {title[0].upper()}{title[1:]}", "",
           "Cell: RMSE (m, median-centred) on the common units of F where every method has a value; in brackets "
           "the bias (median product - truth). Row 'coverage of F': each method's own share of F with a value. "
           "**Bold**: the lowest error of the site.", "",
           "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        name = f"{SITE_NAMES.get(r['site'], r['site'])} ({TRUTH_NAMES.get(r['site'], '')})"
        if r["state"] != "ok":
            txt = (_missing_text(r.get("reason"), "en") if r["state"] == "not reachable" else T["no results"])
            pre = [name] + ([str(r.get("n", ""))] if key == "record" else [])
            cells = pre + [txt] + [""] * (len(head) - len(pre) - 1)
            out.append("| " + " | ".join(cells) + " |")
            continue
        unit = T["points"] if r["site"] == "villaviciosa" else T["px"]
        base = [name] + ([str(r["n"])] if key == "record" else []) + [r["first_date"], f"{r['n_common']:,} {unit}"]
        cov = ["  coverage of F"] + ([""] if key == "record" else []) + ["", ""]
        for k in cols:
            c = r["cells"].get(k, {"status": "absent"})
            if c["status"] == OK and not np.isfinite(c["rmse"]):
                base.append(f"n = {c['n_units']} (too few)")
                cov.append(fmt_cov(c["coverage"]) + " %")
            elif c["status"] == OK:
                v = f"{c['rmse']:.3f} ({c['bias']:+.2f})"
                base.append(f"**{v}**" if r.get("best") == k else v)
                cov.append(fmt_cov(c["coverage"]) + " %")
            else:
                base.append(T.get(c["status"], c["status"]) if c["status"] != "absent" else "-")
                cov.append("")
        out.append("| " + " | ".join(base) + " |")
        out.append("| " + " | ".join(cov) + " |")
    if any(m in desc for m, _ in cols):
        out += ["", "† a catalogue variant, reported descriptively (docs/validation_v2_spec.md §3.2)."]
    if ni:
        out += ["", "Not implemented under the common configuration yet: "
                + ", ".join(f"{m} {dict(CATALOGUE).get(m, m)}" for m in ni) + "."]
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


def table_latex(rows, cols, key, labels, ni, desc=()):
    """One Spanish booktabs table (the style of docs/informe_fer): rows =
    sites (RMSE with the bias in brackets, then a coverage row that also
    carries the window's first scene), columns = methods; scaled down to the
    text width when wider."""
    T = TXT["es"]
    ncol = 2 + len(cols)
    spec = "@{}lr" + "c" * len(cols) + "@{}"
    heads = []
    for m, v in cols:
        if (m, v) == ("E0b", "mb3"):
            heads.append("\\begin{tabular}[b]{@{}c@{}}MAREA\\\\\\sub{regla 3}\\end{tabular}")
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
    for r in rows:
        extra = f", n = {r['n']}" if key == "record" and r.get("n") else ""
        name = f"{tex_escape(SITE_NAMES.get(r['site'], r['site']))} \\sub{{({TRUTH_NAMES.get(r['site'], '')}{extra})}}"
        if r["state"] != "ok":
            txt = (_missing_text(r.get("reason"), "es") if r["state"] == "not reachable" else T["no results"])
            lines.append(f"{name} & \\multicolumn{{{ncol - 1}}}{{l}}{{\\sub{{{txt}}}}} \\\\")
            continue
        unit = T["points"] if r["site"] == "villaviciosa" else T["px"]
        base = [name, f"{fmt_int(r['n_common'], 'es')} \\sub{{{unit}}}"]
        cov = [f"\\quad\\sub{{desde {r['first_date']}; {T['coverage']}}}", ""]
        for k in cols:
            c = r["cells"].get(k, {"status": "absent"})
            if c["status"] == OK and not np.isfinite(c["rmse"]):
                base.append(f"\\sub{{n = {c['n_units']}}}")
                cov.append(f"\\sub{{{fmt_cov(c['coverage'])}\\,\\%}}")
            elif c["status"] == OK:
                v = f"{c['rmse']:.3f}"
                b = f"{c['bias']:+.2f}".replace("+", "$+$").replace("-", "$-$")
                base.append((f"\\up{{{v}}}" if r.get("best") == k else v) + f" \\sub{{({b})}}")
                cov.append(f"\\sub{{{fmt_cov(c['coverage'])}\\,\\%}}")
            else:
                base.append(f"\\sub{{{T.get(c['status'], c['status'])}}}" if c["status"] != "absent" else "\\sub{---}")
                cov.append("")
        lines.append(" & ".join(base) + " \\\\")
        lines.append(" & ".join(cov) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    what = (T["title_rec"] + " (97 / 68 / 71 / 54 im\\'agenes)") if key == "record" else T["title_n"].format(n=key)
    cap = (f"Error de la cota con una misma configuraci\\'on externa a los m\\'etodos (la de producci\\'on de "
           f"MAREA), {what}. Celda: RMSE sin sesgo (m) en los p\\'ixeles de F en los que todos los m\\'etodos "
           "tienen valor; entre par\\'entesis, el sesgo (mediana de producto $-$ verdad). Fila de cobertura: "
           "fracci\\'on de F con valor de cada m\\'etodo y primera escena de la ventana (las n escenas cribadas "
           "m\\'as recientes hasta el 31-12-2025). MAREA con retardo reajustado en cada ventana y la regla de "
           "calidad publicada (0); el resto, con $h_B$ sin retardo. En verde, el menor error de cada sitio.")
    if any(m in desc for m, _ in cols):
        cap += " $^\\dagger$ Variante del cat\\'alogo, descriptiva."
    if ni:
        cap += (" Sin implementar todav\\'ia: " + ", ".join(tex_escape(LABELS_ES.get(m, m)) for m in ni) + ".")
    lines += [f"\\caption{{{cap}}}", f"\\label{{tab:fijo-{key}}}", "\\end{table}", ""]
    return "\n".join(lines)


def table_csv(rows, cols, key, path):
    import io

    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    hdr = ["site", "state", "n_scenes", "first_date", "last_date", "n_common", "n_set", "reason"]
    for m, v in cols:
        hdr += [f"{m}_{v}_{x}" for x in ("status", "rmse", "bias", "slope", "r", "rmse_total", "coverage", "n_own")]
    wr.writerow(hdr)
    for r in rows:
        line = [r["site"], r["state"], r.get("n", ""), r.get("first_date", ""), r.get("last_date", ""),
                r.get("n_common", ""), r.get("n_set", ""), r.get("reason", "")]
        for k in cols:
            c = r.get("cells", {}).get(k, {"status": "absent"})
            line += [c["status"]] + [v14._fmt(c.get(x, float("nan"))) for x in
                                     ("rmse", "bias", "slope", "r", "rmse_total", "coverage")] + [c.get("n_own", "")]
        wr.writerow(line)
    atomic_text(path, buf.getvalue())


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
    for _, r in df[(df["variant"].isin(["main", "mb0"]))].iterrows():
        labels.setdefault(r["method"], r["label"])
    cols = table_columns(df)
    ni = not_implemented(df)
    desc = descriptive_ids(df)
    tdir = os.path.join(out, "tables")
    os.makedirs(tdir, exist_ok=True)
    ns = parse_ns(args.ns)
    keys = [n for n in ns if (df["n_scenes"].astype(int) == n).any()]
    if (df["is_record"].astype(int) == 1).any():
        keys.append("record")
    md_all, tex_all, paths = [], [], []
    for key in keys:
        rows = table_data(df, sites, key, cols)
        stem = os.path.join(tdir, f"table_{key if key == 'record' else f'n{int(key):03d}'}")
        table_csv(rows, cols, key, stem + ".csv")
        md = table_markdown(rows, cols, key, labels, ni, desc)
        tex = table_latex(rows, cols, key, labels, ni, desc)
        atomic_text(stem + ".md", md)
        atomic_text(stem + "_es.tex", tex)
        md_all.append(md)
        tex_all.append(tex)
        paths.append(stem)
        log(f"-> {stem}.csv/.md/_es.tex")
    atomic_text(os.path.join(tdir, "all_tables.md"), "# Fixed-configuration tables\n\n" + "\n".join(md_all))
    atomic_text(os.path.join(tdir, "all_tables_es.tex"), LATEX_DOC % "\n".join(tex_all))
    log(f"-> {tdir}/all_tables.md, {tdir}/all_tables_es.tex")
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
}
VARIANT_STYLE = dict(color=PAL["deep"], marker="o", ls=(0, (4, 2)), mfc="white")
_SPARE = [dict(color=c, marker=m, ls=l) for c in (PAL["teal"], PAL["sand"], PAL["slate"])
          for m, l in (("d", "--"), ("H", ":"), ("8", "-."))]


def style_for(mid, var):
    if mid == "E0b" and var == "mb3":
        return dict(VARIANT_STYLE)
    if mid in STYLE:
        return dict(STYLE[mid])
    return dict(_SPARE[zlib.crc32(str(mid).encode()) % len(_SPARE)])


CTEXT = {
    "en": dict(title="Elevation error vs number of screened images, one fixed configuration ({res} m)",
               rmse="Elevation RMSE (m)", cov="Coverage of F (%)", x="Screened images in the window, n",
               no="no results yet", rec="record", unreach="n > {n}: not reachable",
               variant="MAREA, quality rule mb 3 (variant)",
               truth={"rtk": "RTK-GNSS, {n} points", "vaklodingen": "Vaklodingen, {n} px"},
               foot=["One configuration external to the methods (MAREA's production): transition-zone scene "
                     "screening, per-pixel mask SCL 4-7, NDWI, wet = NDWI > 0, the site's production h_B; window = "
                     "the n most recent screened scenes up to 31 Dec 2025 (nested).",
                     "MAREA: band lags refitted on each window, quality rule 0 (the published products'); every "
                     "other method reads h_B without lag. {scoring}",
                     "Coverage = each method's own share of the fixed set F (finite truth and published MAREA)."],
               common=("RMSE median-centred on the common set: the units of F where every method computed at "
                       "that n has a value."),
               own="RMSE median-centred on each method's own valid units of F.",
               pending="Pending: {m}.", notimpl="Not implemented yet: {m}."),
    "es": dict(title="Error de la cota frente al número de imágenes cribadas, con una misma configuración ({res} m)",
               rmse="RMSE de la cota (m)", cov="Cobertura de F (%)", x="Imágenes cribadas en la ventana, n",
               no="sin resultados todavía", rec="registro", unreach="n > {n}: no alcanzable",
               variant="MAREA, regla de calidad mb 3 (variante)",
               truth={"rtk": "RTK-GNSS, {n} puntos", "vaklodingen": "Vaklodingen, {n} px"},
               foot=["Una configuración externa a los métodos (la de producción de MAREA): cribado de escenas en la "
                     "zona de transición, máscara por píxel SCL 4-7, NDWI, mojado = NDWI > 0, h_B de producción del "
                     "sitio; ventana = las n escenas cribadas más recientes hasta el 31-12-2025 (anidadas).",
                     "MAREA: retardos por banda reajustados en cada ventana, regla de calidad 0 (la de los productos "
                     "publicados); el resto de métodos usa h_B sin retardo. {scoring}",
                     "Cobertura = fracción del conjunto fijo F (verdad finita y MAREA publicado finito) con valor de "
                     "cada método."],
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
        # the headline methods and MAREA's mb-3 variant; never the internal
        # reference or the registry's declared sensitivities
        if role != "headline" and (mid, var) != ("E0b", "mb3"):
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
    if (mid, var) == ("E0b", "mb3"):
        return CTEXT[lang]["variant"]
    if lang == "es":
        return LABELS_ES.get(mid, label)
    return label if mid not in dict(CATALOGUE) or mid == "E0b" else dict(CATALOGUE)[mid]


def draw_site(ar, ac, d, lang, scoring, handles):
    from matplotlib.lines import Line2D

    T = CTEXT[lang]
    series = curve_series(d, scoring)
    order = sorted(series, key=lambda k: (k[0] != "E0b", k[1] == "mb3",
                                          CATALOGUE_IDS.index(k[0]) if k[0] in CATALOGUE_IDS else 99, k[0]))
    ends = []
    xs_all = set()
    for k in order:
        s = series[k]
        st = style_for(*k)
        x = np.array([p["n"] for p in s["pts"]], float)
        y = np.array([p["rmse"] for p in s["pts"]], float)
        c = np.array([100 * p["cov"] for p in s["pts"]], float)
        xs_all.update(int(v) for v in x)
        lw = 1.6 if k[1] == "mb3" else 2.0
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
        if len(fin) and k[1] != "mb3":
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


def curves_figure(df, sites, lang, scoring="common", end_labels=None):
    """RMSE (top, on ``scoring``) and coverage of F (bottom) against n, one
    column per site, one line per method (MAREA always PDeep). Direct end
    labels only on a single-site figure (room on the right; the legend
    carries identity everywhere)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

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
        ar.text(0.0, 1.10, SITE_NAMES.get(site, site), transform=ar.transAxes, fontsize=13.5, color=PAL["deep"],
                weight="bold", va="bottom", ha="left")
        if j == 0:
            ar.set_ylabel(T["rmse"], color=PAL["ink"], fontsize=11.5)
            ac.set_ylabel(T["cov"], color=PAL["ink"], fontsize=11.5)
        ac.set_xlabel(T["x"], color=PAL["ink"], fontsize=10.5)
        d = df[df["site"] == site]
        if d.empty or not (d["status"] == OK).any():
            ar.text(0.5, 0.5, T["no"], transform=ar.transAxes, ha="center", va="center", fontsize=12,
                    color=PAL["slate"])
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
        kind = "rtk" if site == "villaviciosa" else "vaklodingen"
        nset = int(_f(d["n_set"].max()))
        ar.text(0.0, 1.025, T["truth"][kind].format(n=f"{nset:,}"), transform=ar.transAxes, fontsize=9.5,
                color=PAL["slate"], va="bottom", ha="left")
        ar.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    import textwrap
    title = T["title"].format(res="10")
    # bold DejaVu Sans is about 0.6 em per character: wrap to the figure width
    per_line = max(30, int((1.0 - lay["left"] - 0.02) * fig.get_figwidth() * 72 / (0.6 * lay["title_fs"])))
    fig.text(lay["left"], lay["title_y"] + 0.02, textwrap.fill(title, per_line), fontsize=lay["title_fs"],
             color=PAL["deep"], weight="bold", ha="left", va="top", linespacing=1.15)
    order = sorted(handles, key=lambda k: (k[0] != "E0b", k[1] == "mb3",
                                           CATALOGUE_IDS.index(k[0]) if k[0] in CATALOGUE_IDS else 99, k[0]))
    if order:
        leg = dict(lay["leg"])
        leg["ncol"] = min(leg["ncol"], len(order))
        fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order], frameon=False,
                   fontsize=10.2, handlelength=2.6, columnspacing=1.8, **leg)
    foot = [t.format(scoring=T[scoring]) for t in T["foot"]]
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
    paths = []
    for lang in parse_list(args.lang):
        if lang not in CTEXT:
            raise SystemExit(f"--lang: {lang!r} not in {sorted(CTEXT)}")
        fig = curves_figure(df, sites, lang, args.scoring)
        stem = os.path.join(out, f"curves_{args.scoring}_{lang}")
        fig.savefig(stem + ".png", dpi=200)
        fig.savefig(stem + ".pdf")
        plt.close(fig)
        paths.append(stem)
        log(f"-> {stem}.png/.pdf")
        for s in sites:
            if (df["site"] == s).any() and (df[df["site"] == s]["status"] == OK).any():
                fig = curves_figure(df, [s], lang, args.scoring)
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
