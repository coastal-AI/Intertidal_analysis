# -*- coding: utf-8 -*-
"""Pre-2023 scenes of the Dutch sites from the Zarr stores, as v19's extra extraction.

Why
---
The fixed-configuration tables (``experiments/v19_fixed_config_tables.py``:
windows of the n = 125, 100, 75, 60, 50, 40 most recent screened scenes,
ending on the 2025 Vaklodingen survey) need scenes before 2023 at the
Westerschelde (``escalda``), the Wadden and the Ems: the legacy 10 m netCDF
cubes cover 2023-2025 only (68 / 71 / 54 screened scenes).
``experiments.download_cube_zarr`` fetches the missing years from the CDSE
openEO backend as yearly Zarr stores (B03, B08 and SCL only for the light
stores, ``cubes_light/<site>/<year>.zarr``; the 9-band stores of 2023-2025
stay in ``cubes/``). This script turns the stores of the years asked for
into ONE extraction per site,
``results/fixedcfg/extra/<site>_extract_<y0>-<y1>.npz`` plus its sidecar
``<site>_extract_<y0>-<y1>.json``, in the contract of v19's extra extraction
store (v19 docstring, ``load_extra_store`` and ``extend_with_extra``), and
gates it on 2023, the year that is both in the production extraction and in
a store. Nothing is downloaded; the stores, the products and the legacy
cubes are only read.

The npz (production semantics; nothing recomputed that production fixed)
------------------------------------------------------------------------
* ``Y`` float32 (T, P), C order: raw NDWI = (B03 - B08) / (B03 + B08) at the
  production keep pixels exactly as ``pyintertidal.marea.extract`` computes
  it (the int16 digital numbers as stored, no rescaling and no offset, the
  fill -32768 read as NaN like xarray's CF decoding of the legacy cubes;
  float32 arithmetic; NaN where B03 + B08 = 0 or a band has no data).
* ``C`` bool (T, P): SCL in {4, 5, 6, 7} (``marea.CLEAR``) at the keep pixels.
* ``keep``, ``shape``, ``sea``, ``inter``: copied byte for byte from the
  production extraction ``products_<site>_10m/marea_extract.npz`` (keep comes
  from the 2023-2025 scenes and is never recomputed). Not copied: ``s_km`` (v19
  reads the main extraction's), ``bbox`` (an object array) and ``crs_wkt``.
* ``dates`` '<U10' 'YYYY-MM-DD': EVERY scene of the stores of those years,
  sorted and unique (v19 counts cube scenes per window from them).
The rows are streamed into the npz (deflate, zip64; C packed in memory while
Y is written), and the first, middle and last rows are read back and compared
before the file is renamed into place.

The sidecar (the keys v19 reads, plus provenance)
-------------------------------------------------
* ``overpass``: {date: 'YYYY-MM-DDTHH:MM:SS'}, the datatake-start label
  instants (``pyintertidal.overpass.get_overpass_times``, first product of
  the day, as production) from ``<root>/<site>/overpass_times.json``, which
  ``download_cube_zarr.update_overpass`` writes (and ``--overpass`` files).
  A date without a time is left out (v19 drops it and counts it).
* ``cloud_pct``: {date: %} of every scene, the production transition-zone
  screening (below). v19 applies ``cloud_pct / 100 <= 0.10`` itself; no
  top-level ``usable`` key is written (v19 would let it take precedence).
* ``gate_2023``: the 2023 gate (below), a dict with ``ok``.
* ``reference_map``: the production reference map used (path, sha256, classes).
* provenance: the stores (registration, verification, .zmetadata sha256),
  the input fingerprints, per-date screening detail, per-row CRC32 of Y and
  C, the code version and sha256.
* ``radiometry_probe`` (a GATE since 2026-10-26, below): per year and per
  input processing-baseline group, the median B03 and the median share of
  low B08 digital numbers of the clear keep pixels against the gate year's.
* ``levels_pre2023``: the production boundary h_B at the overpass instants of
  the store's scenes, with the IOC gauges of those years (below).
* ``n_target``: the screened scenes the extra store adds against the
  ``N_TARGET`` = 125 of the largest table (the stop rule of the purchases).

The screening (production's rule, production's fixed reference map)
-------------------------------------------------------------------
``pyintertidal.stability.reference_and_clouds`` pass 2 with the notebooks'
parameters (cell 12: bad SCL {3, 8, 9, 10}, no data counts as not bad; g =
bad share of the whole H x W frame; g <= 0.05 -> cloud_pct = round(100 g, 4),
else round(100 x bad transition pixels / transition pixels, 4); usable iff
cloud_pct / 100 <= 0.10, ``filter_dates``) on the FIXED production reference
map (pass 1 depends on the 2023-2025 scenes and is never redone on new
years). :func:`screen_scene` is that pass for one scene; the tests check it
against ``reference_and_clouds`` itself and ``reference`` checks it on every
scene of the legacy cube. The map is the production one: ``reference``
recomputes it exactly as the notebooks did (``reference_and_clouds`` on the
legacy 10 m cube; the notebooks never wrote it) or imports a file
(``--from-map``), and gates it on the class counts the notebooks printed
(``EXPECTED_REFERENCE``) and on reproducing result.json ``record_dates``.

Pre-registered rule, the static no-data of the legacy cubes: the legacy 10 m
cubes hold pixels with no data on EVERY date (Ferrol measured: a 13 x 301 px
block), where the Zarr stores have data. Production's map labels them
transition and its screening never counted them bad. ``reference`` stores
the mask of the pixels whose legacy SCL is no data (fill or class 0) on every
2023-2025 date; the screening sets them to no data in every store's SCL
before pass 2. The keep pixels never touch the mask (checked).

Gates
-----
``reference`` (the map and the mask): class counts equal the notebooks'
print and the cube has the printed number of scenes (``--allow-reference-
mismatch`` records an override); when the map is recomputed,
:func:`screen_scene` gives production's cloud_pct on every legacy scene,
bit for bit; usable dates with a production overpass time = result.json
record_dates; keep outside the static mask.

``gate`` (2023, no credits once a 2023 store exists): the 2023 store (the
light one when complete, else the 9-band one) against the production
extraction on the 2023 dates: the same dates; Y bitwise (float32 exactness,
NaN positions equal; the documented tolerance is none, since identical
digital numbers give identical float32 NDWI) and C bitwise on every keep
pixel; the screening of the store's 2023 SCL (static mask applied) gives
exactly the 2023 part of record_dates (usable and with a production overpass
time); per-date cloud_pct against the legacy reproduction (``reference``)
when there is one. Differences are attributed (NaN only in production =
the legacy's no-data; values; signed zeros; C where production has no data)
and listed per date, with their SHARE of the pixel-dates; the store's B03 /
B08 digital numbers are also compared with the legacy cube's at the keep
pixels (``legacy_dn``: valid pixel-dates differing, their share, max |d| DN;
the Ems 9-band 2023 store differs from the legacy cube on 0.74 % of its
valid B03 / B08 pixel-dates, max 5192 DN). A gate that fails on extraction
values ONLY is accepted solely by the user's explicit flag: ``gate
--accept-gate-differences`` records the acceptance in the report (the
downloader then lets that site's pre-2023 years be bought) and ``build``
honours it, or ``build --accept-gate-differences`` (recorded); nothing
accepts a date, screening or static-mask difference.

``build`` refuses: a gate that did not pass or was computed with other inputs
or another code version (``--allow-failed-gate`` embeds it failed, so v19
refuses it unless --soft-gates; ``--accept-gate-differences`` accepts, as a
recorded decision, a gate whose dates and screening reproduce production and
whose only differences are extraction values, e.g. the Ems three-tile
mosaic); a store that is not complete, lacks B03/B08/SCL as int16 with fill
-32768, is not on the production grid, or whose registration is not
established: method 'netcdf' verified, or '--offset-px' / 'sibling store'
confirmed by the median composites, at the site's known offset and delivered
shape (``SITE_OFFSET``, ``SITE_DELIVERED_SHAPE``; ``--allow-unconfirmed-
registration`` records an override); light stores when the downloader's
light-path gate (``cubes_light/escalda/2023.vs_nine_band.json``, the band
subset against the nine-band store of the same year, bitwise) FAILED (the
same override; an absent report is recorded and warned about); a store
bought with the downloader's ``--without-light-gate`` while that report is
absent or did not pass (the same override); years that
reach the main extraction's
first year; and another extra store of the site in the extra folder (v19
uses every ``<site>_extract_*.npz`` and stops on a repeated date): one this
script wrote and that overlaps is moved aside with ``--replace`` (never
deleted), one of another builder is never touched (``--allow-other-stores``
accepts a non-overlapping one).

Radiometry (``build``, 2026-10-26): the gate year (2023, baselines 05.09-05.11)
holds no BOA offset in the values, and nothing else checks the older
baselines (e.g. 2022 changes to 04.00 on 01-25; reprocessed N05.xx products
can sit among N02-N03 originals). Per store year and per input-baseline group
of the year (``input_baselines`` of the store), on the dates with clear keep
pixels holding data on at least ``RADIOMETRY_MIN_VALID`` of the keep pixels:
a group of at least ``RADIOMETRY_MIN_DATES`` such dates whose median share
of low B08 (1 <= DN < ``LOW_DN``) is below ``LOW_SHARE_OFFSET`` while the gate
year's is above ``LOW_SHARE_NO_OFFSET`` (a +1000 offset left in the values
removes every low DN), or whose median B03 is more than
``OFFSET_SUSPECT_DN`` from the gate year's, refuses the build;
``--accept-radiometry`` takes it (recorded). The dates before baseline 04.00
are counted per year. NDWI > 0 is offset-invariant; NDWI magnitudes (DEA
Intertidal at 0.1, CoastXray at 0.2, MAREA's inversion) are not.

Boundary levels before 2023 (``build``, 2026-10-26)
---------------------------------------------------
Production's h_B at the Dutch sites is the consensus of EOT20 and the IOC
gauges vlis + brsk (Westerschelde), ters (Wadden), bork (Ems)
(``make_boundary``, ``PRODUCTION_BOUNDARY``), whose cached chunks in
``data_v4/gauges`` cover 2023-2025 only: before 2023 that object gives EOT20
ALONE. ``build`` therefore fetches the free IOC data of the store years with
the production downloader (``pyintertidal.boundary.fetch_ioc``, yearly
chunks) into a SEPARATE cache, ``PRE2023_GAUGE_CACHE`` (never into
``data_v4/gauges``: ``load_cached_ioc`` reads every chunk there and
``make_boundary`` demeans each gauge by their median, so a pre-2023 chunk
there would silently change the published 2023-2025 h_B), or stops with a
clear message (``--no-fetch``: no network, missing years refuse). A fetched
year counts when every month holds data on at least
``GAUGE_MIN_DAYS_PER_MONTH`` days (an outage leaves ``[]``: moved aside as
``*.incomplete-<UTC>``, never deleted, and fetched once more); a real gap
refuses unless ``--accept-gauge-gaps`` (recorded). The levels: production's
own boundary object (its EOT20 member, its inverse-distance weights) with
each gauge member rebuilt from the pre-2023 record by the production recipe
(``load_cached_ioc`` + ``despike``) and demeaned by the SAME constant as
production (the median of its 2023-2025 record, reproduced bit for bit from
``data_v4/gauges``, so the datum does not change at 2023; the pre-2023
median is recorded beside it, not used). A usable scene whose level would
rest on EOT20 alone (no gauge with data within an hour) refuses the build
unless ``--accept-gauge-gaps``. The sidecar's ``levels_pre2023`` holds h_B and
each member per scene. v19 (2026-10-26) does not read it: it computes h_B
from the production boundary object, i.e. EOT20 alone before 2023 with
``--pre2023-levels production``; ``build`` and ``check`` say so (v19's
owner decides how the levels reach the tables).

Commands (from the repository root; server: ``source env.sh`` first)
--------------------------------------------------------------------
    $PY -X utf8 -m experiments.light_to_extraction reference escalda
    $PY -X utf8 -m experiments.light_to_extraction gate escalda
    $PY -X utf8 -m experiments.light_to_extraction gate ems --accept-gate-differences   # only if the user decides
    $PY -X utf8 -m experiments.light_to_extraction build escalda --years 2020-2022 --dry-run
    $PY -X utf8 -m experiments.light_to_extraction build escalda --years 2020-2022
    $PY -X utf8 -m experiments.light_to_extraction check escalda

Outputs: ``results/fixedcfg/light/<site>_reference.{npz,json}`` (map, static
mask, legacy grid; provenance and gates), ``<site>_gate_2023.json``, and the
store pair in ``results/fixedcfg/extra`` (a ``.gitignore`` holding ``*`` is
written into each of those folders on its first write: the npz files reach
GBs and the repository goes public); the pre-2023 IOC chunks in
``data_v4/gauges_pre2023``. Importing this module does nothing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import sys
import time
import zipfile
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
#  constants
# ─────────────────────────────────────────────────────────────────────────────

#: the logic of the rows, the screening and the static rule; a gate computed
#: with another version is not accepted by ``build`` (bump on any change).
#: 2 (2026-10-26): the gate adds the keep pixels' B08 statistics (radiometry
#: gate), the B03 / B08 comparison with the legacy cube and the user's
#: recorded acceptance
CODE_VERSION = 2
BUILDER = "experiments/light_to_extraction.py"
TARGET_SITES = ("escalda", "wadden", "ems")
FILL = -32768
BANDS = ("B03", "B08", "SCL")
NINE_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL", "CLD")
#: pyintertidal.marea.CLEAR and pyintertidal.water.BAD_CLASSES (the tests
#: assert both against the library)
CLEAR = (4, 5, 6, 7)
BAD = (3, 8, 9, 10)
#: the notebooks' screening call (tide_boundary_comparison_<site>.ipynb,
#: cells 2, 10 and 12)
CLEAN_SCENE_MAX_BAD = 0.05
CLOUD_THRESHOLD = 0.10
STABLE_THRESHOLD = 0.95
TRANSITION_BUFFER_PX = 10
NDWI_THRESHOLD = 0.0
TIME_EXTENT = ("2023-01-01", "2025-12-31")
GATE_YEAR = 2023
#: production boundary tags of the Dutch products (v14.DUTCH_TAGS)
DUTCH_TAGS = {"escalda": "eot20_g2", "wadden": "eot20_g1", "ems": "eot20_g1"}
#: production's 10 m reference maps: class counts 0 / 1 / 2 the notebooks
#: printed (server runs 2026-10-01..05), scenes of the legacy cube and usable
#: dates (as experiments/download_cube_stac.EXPECTED_REFERENCE)
EXPECTED_REFERENCE = {
    "escalda": {"classes": {0: 777766, 1: 990308, 2: 1314093}, "cube_scenes": 464, "usable": 68},
    "wadden": {"classes": {0: 3051651, 1: 5025552, 2: 1430597}, "cube_scenes": 462, "usable": 71},
    "ems": {"classes": {0: 3401933, 1: 972618, 2: 6567527}, "cube_scenes": 462, "usable": 54},
}
#: where the frame sits in the backend's padded deliveries, and their shape,
#: measured on the 9-band stores (registration 'netcdf', verified;
#: results/server_logs_2026-10-19/zarr_verification/verify_table.json)
SITE_OFFSET = {"escalda": (5, 1), "wadden": (0, 5), "ems": (288, 271)}
SITE_DELIVERED_SHAPE = {"escalda": (1536, 2176), "wadden": (3456, 2816), "ems": (3840, 3968)}
SITE_FRAME_SHAPE = {"escalda": (1467, 2101), "wadden": (3425, 2776), "ems": (3206, 3413)}
#: registration records (download_cube_zarr.register_delivery) whose offset
#: is established for an extraction
REG_CONFIRMED = "confirmed by the median composites"
EXTRA_DIR = "results/fixedcfg/extra"
WORK_DIR = "results/fixedcfg/light"
LIGHT_ROOT = "cubes_light"
GATE_ROOTS = ("cubes_light", "cubes")
#: dates per read block (the stores' time chunk)
T_BLOCK = 16
#: the downloader's light-path gate (download_cube_zarr.nine_band_gate: the
#: band subset's store of a year against the nine-band store of that year,
#: bitwise; only the Westerschelde 2023 has both)
LIGHT_CHECK = "cubes_light/escalda/2023.vs_nine_band.json"
#: per-date differences listed in a gate report
GATE_LIST = 25
#: radiometry gate (module docstring): a group whose median B03 over the clear
#: keep pixels is farther than this from the gate year's (+1000 offset: 1000 DN)
OFFSET_SUSPECT_DN = 500
#: a low digital number (1 <= DN < LOW_DN; download_cube_zarr.LOW_DN): water
#: without the +1000 BOA offset; never with it (a reflectance below -0.05)
LOW_DN = 500
#: the share of low B08 among the clear keep pixels with data: a group below
#: LOW_SHARE_OFFSET while the gate year is above LOW_SHARE_NO_OFFSET holds the
#: offset (download_cube_zarr.radiometry_summary's thresholds)
LOW_SHARE_OFFSET = 0.002
LOW_SHARE_NO_OFFSET = 0.01
#: a date counts for the radiometry when its clear keep pixels with data are
#: at least this share of the keep pixels; a group needs this many such dates
RADIOMETRY_MIN_VALID = 0.01
RADIOMETRY_MIN_DATES = 5
#: the n of the largest table (v19: windows of 125, 100, 75, 60, 50, 40 scenes)
N_TARGET = 125
#: production's boundary h_B of the Dutch sites (experiments/v7_waterline_
#: contour.py SITES, copied; the tests assert it): make_boundary(aoi, 'EOT20',
#: n_gauges, period 2023-2025, exclude the judge), and the gauges it picks
PRODUCTION_BOUNDARY = {
    "escalda": {"tide_model": "EOT20", "n_gauges": 2, "exclude": ("trnz",), "gauges": ("vlis", "brsk")},
    "wadden": {"tide_model": "EOT20", "n_gauges": 1, "exclude": ("harl",), "gauges": ("ters",)},
    "ems": {"tide_model": "EOT20", "n_gauges": 1, "exclude": ("delf",), "gauges": ("bork",)},
}
#: production's IOC cache (2023-2025 chunks; only READ here) and the separate
#: cache of the pre-2023 yearly chunks this script fetches (never the former)
PRODUCTION_GAUGE_CACHE = "data_v4/gauges"
PRE2023_GAUGE_CACHE = "data_v4/gauges_pre2023"
#: a yearly IOC chunk counts when every month holds data on at least this many
#: days (validation v2's rule, experiments/benchmark/fetch/ioc.py)
GAUGE_MIN_DAYS_PER_MONTH = 20

_T0 = time.time()


class Refused(RuntimeError):
    """A rule of this script refuses the inputs (nothing is written)."""


def log(msg):
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} +{(time.time() - _T0) / 60:6.1f} min] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  small utilities
# ─────────────────────────────────────────────────────────────────────────────

def resolve(path):
    """A path from the repository root unless absolute."""
    p = os.fspath(path)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), ROOT).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def utcnow():
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def array_sha256(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def zip_fingerprint(path):
    """A cheap fingerprint of an npz: size, mtime and the sha256 of its
    members' names, CRC-32s and sizes (the zip directory; no member read)."""
    with zipfile.ZipFile(path) as zf:
        items = sorted((i.filename, int(i.CRC), int(i.file_size)) for i in zf.infolist())
    st = os.stat(path)
    return {"path": rel(path), "size": int(st.st_size),
            "mtime_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc)
            .replace(tzinfo=None).isoformat(timespec="seconds"),
            "members_sha256": hashlib.sha256(json.dumps(items).encode()).hexdigest(),
            "members": {n: [c, s] for n, c, s in items}}


def file_info(path):
    st = os.stat(path)
    return {"path": rel(path), "size": int(st.st_size), "sha256": sha256_file(path)}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_json(path, obj):
    """``obj`` as indented JSON at ``path``, through ``path.tmp``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=_jsonable, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_ignored(folder):
    """``<folder>/.gitignore`` holding ``*`` (never overwritten): this
    script's outputs (npz files of GBs) never show up in git, as the cube
    roots' (``download_cube_zarr.ensure_gitignore``)."""
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, ".gitignore")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("# outputs of experiments/light_to_extraction.py (large npz files): never in git\n*\n")
    return p


def _attr_json(v):
    """A store attribute that holds JSON text, decoded (or as it is)."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def parse_years(text):
    """'2020-2022', '2022' or '2019,2020,2021' -> [2019, ...] (contiguous)."""
    s = str(text).strip()
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", s)
    if m:
        y0, y1 = int(m.group(1)), int(m.group(2))
        years = list(range(y0, y1 + 1))
    else:
        try:
            years = sorted({int(v) for v in s.replace(";", ",").split(",") if v.strip()})
        except ValueError:
            raise argparse.ArgumentTypeError(f"years must be YYYY, YYYY-YYYY or a list, not {text!r}") from None
    if not years:
        raise argparse.ArgumentTypeError(f"no years in {text!r}")
    if years != list(range(years[0], years[-1] + 1)):
        raise argparse.ArgumentTypeError(f"the years must be contiguous (the output is named <y0>-<y1>): {years}")
    return years


def iso_instant(v):
    """'YYYY-MM-DDTHH:MM:SS' (naive UTC) of an overpass value."""
    import pandas as pd

    ts = pd.Timestamp(v)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  the two production operations: one extraction row, one scene's screening
# ─────────────────────────────────────────────────────────────────────────────

def keep_subframe(keep, shape):
    """``((r0, r1, c0, c1), rows, cols)``: the bounding box of the keep pixels
    in the (H, W) frame and their row / column indices inside it."""
    H, W = (int(v) for v in shape)
    rows, cols = np.divmod(np.asarray(keep, np.int64), W)
    if len(rows) and (rows.min() < 0 or rows.max() >= H):
        raise ValueError("keep pixels outside the frame")
    r0, r1 = (int(rows.min()), int(rows.max()) + 1) if len(rows) else (0, 0)
    c0, c1 = (int(cols.min()), int(cols.max()) + 1) if len(cols) else (0, 0)
    return (r0, r1, c0, c1), rows - r0, cols - c0


def extraction_row(b03, b08, scl, rows, cols):
    """(Y, C) of one scene at the keep pixels, as ``pyintertidal.marea.extract``
    computes them from the legacy netCDF: the int16 bands decoded to float32
    with NaN at the fill (xarray's CF decoding), NDWI = where(g + n != 0,
    (g - n) / (g + n), NaN) in float32, C = SCL in {4, 5, 6, 7}."""
    gi = b03[rows, cols]
    ni = b08[rows, cols]
    g = gi.astype(np.float32)
    n = ni.astype(np.float32)
    g[gi == FILL] = np.nan
    n[ni == FILL] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(g + n != 0, (g - n) / (g + n), np.nan)
    c = np.isin(scl[rows, cols], CLEAR)
    return y.astype(np.float32, copy=False), c


def keep_b03_median(b03, c, rows, cols):
    """Median B03 (digital numbers) over the clear keep pixels with data, or
    None: the radiometry probe. B03 is nearly the same over wet and dry flats
    (B08 is not), so the yearly median hardly depends on the tides sampled,
    and an L2A +1000 BOA offset left in the values moves it by 1000 DN (NDWI
    magnitudes, not its sign, depend on that offset)."""
    gi = b03[rows, cols]
    ok = c & (gi != FILL)
    if not ok.any():
        return None
    return float(np.median(gi[ok]))


def keep_b08_stats(b08, c, rows, cols):
    """The clear keep pixels with B08 data as a share of the keep pixels
    (``keep_valid_share``) and the share of them with a low B08 digital
    number, 1 <= DN < LOW_DN (``keep_b08_low_share``; None without data):
    wet flats without the +1000 BOA offset, never with it."""
    ni = b08[rows, cols]
    ok = c & (ni != FILL)
    nv = int(ok.sum())
    low = int((ok & (ni >= 1) & (ni < LOW_DN)).sum())
    return {"keep_valid_share": round(nv / len(rows), 6) if len(rows) else 0.0,
            "keep_b08_low_share": round(low / nv, 6) if nv else None}


def _median(items, key):
    v = [x[key] for x in items if x.get(key) is not None]
    return float(np.median(v)) if v else None


def _radiometry_dates(items):
    return [x for x in items if (x.get("keep_valid_share") or 0.0) >= RADIOMETRY_MIN_VALID]


def radiometry_probe(per_date, years, reference, baselines=None):
    """The radiometry GATE of a build (module docstring): per store year and
    per input processing-baseline group of the year (``baselines``: {date:
    [baselines]}, the stores' ``input_baselines``; 'unrecorded' without),
    on the dates whose clear keep pixels with data reach RADIOMETRY_MIN_VALID
    of the keep pixels, the median ``keep_b03_median_dn`` and
    ``keep_b08_low_share`` against the gate year's (``reference``: the gate
    report's {date: detail}). A group of at least RADIOMETRY_MIN_DATES such
    dates is SUSPECT when its low-B08 share is below LOW_SHARE_OFFSET while the
    gate year's is above LOW_SHARE_NO_OFFSET, or its B03 is more than
    OFFSET_SUSPECT_DN from the gate year's. Returns the record (``suspect``:
    the group names; ``suspect_years``; per year the dates before 04.00)."""
    baselines = baselines or {}
    ref_items = _radiometry_dates(list((reference or {}).values()))
    ref_b03, ref_low = _median(ref_items, "keep_b03_median_dn"), _median(ref_items, "keep_b08_low_share")
    out = {"rule": (f"per store year and per input-baseline group of the year, on the dates whose clear keep "
                    f"pixels with data are at least {RADIOMETRY_MIN_VALID} of the keep pixels: a group of at least "
                    f"{RADIOMETRY_MIN_DATES} such dates is suspect when its median share of low B08 (1 <= DN < "
                    f"{LOW_DN}) is below {LOW_SHARE_OFFSET} while the gate year's is above {LOW_SHARE_NO_OFFSET}, "
                    f"or its median B03 is more than {OFFSET_SUSPECT_DN} DN from the gate year's (an L2A +1000 BOA "
                    "offset left in the values); a suspect group refuses the build (--accept-radiometry, "
                    "recorded)"),
           "gate_year": ref_b03, "gate_year_b08_low_share": ref_low, "gate_year_dates": len(ref_items),
           "years": {}, "groups": {}, "suspect": [], "suspect_years": [], "pre_04_dates": {}}
    groups = {}
    for d, v in sorted(per_date.items()):
        if d[:4] not in {str(y) for y in years}:
            continue
        bl = baselines.get(d)
        name = "+".join(sorted(str(b) for b in bl)) if bl else "unrecorded"
        groups.setdefault((d[:4], "all"), []).append(v)
        groups.setdefault((d[:4], name), []).append(v)
        if bl and any(_baseline_before_04(b) for b in bl):
            out["pre_04_dates"][d[:4]] = out["pre_04_dates"].get(d[:4], 0) + 1
    for y in years:
        out["years"][str(y)] = _median(_radiometry_dates(groups.get((str(y), "all"), [])), "keep_b03_median_dn")
        out["pre_04_dates"].setdefault(str(y), 0)
    for (y, name), items in sorted(groups.items()):
        use = _radiometry_dates(items)
        b03, low = _median(use, "keep_b03_median_dn"), _median(use, "keep_b08_low_share")
        why = []
        decided = len(use) >= RADIOMETRY_MIN_DATES
        if decided and low is not None and ref_low is not None and ref_low > LOW_SHARE_NO_OFFSET \
                and low < LOW_SHARE_OFFSET:
            why.append(f"low-B08 share {low:.5f} < {LOW_SHARE_OFFSET} while the gate year's is {ref_low:.5f}")
        if decided and b03 is not None and ref_b03 is not None and abs(b03 - ref_b03) > OFFSET_SUSPECT_DN:
            why.append(f"median B03 {b03:.0f} DN vs {ref_b03:.0f} in the gate year")
        key = f"{y} {name}"
        out["groups"][key] = {"dates": len(items), "dates_used": len(use), "decided": decided,
                              "keep_b03_median_dn": b03, "keep_b08_low_share": low, "suspect": why}
        if why:
            out["suspect"].append(key)
            if y not in out["suspect_years"]:
                out["suspect_years"].append(y)
    return out


def _baseline_before_04(b):
    try:
        return float(b) < 4.0
    except (TypeError, ValueError):
        return False


def screen_scene(scl, transition, static=None, n_transition=None,
                 clean_max_bad=CLEAN_SCENE_MAX_BAD, cloud_threshold=CLOUD_THRESHOLD):
    """Production's transition-zone screening of ONE scene on the fixed
    reference map (``transition`` = map == 0): pass 2 of
    ``pyintertidal.stability.reference_and_clouds`` then ``filter_dates``,
    operation for operation (SCL with no data set to 0 and cast to int16;
    bad = SCL in {3, 8, 9, 10}; g = the mean over the whole frame; clean
    when g <= 0.05: round(g * 100, 4), else round(bad in the transition /
    transition pixels * 100, 4); usable iff cloud_pct / 100 <= 0.10).
    ``static``: the pre-registered static no-data mask of the legacy cube,
    set to no data before the pass (None: nothing masked)."""
    s = np.asarray(scl)
    nodata = np.isnan(s) if s.dtype.kind == "f" else (s == FILL)
    if s.dtype.kind == "f":
        s = np.nan_to_num(s, nan=0.0).astype(np.int16)
    else:
        s = np.where(nodata, 0, s).astype(np.int16)
    masked_bad = 0
    if static is not None:
        masked_bad = int(np.isin(s[static], BAD).sum())
        s[static] = 0
    bad = np.isin(s, list(BAD))
    g = bad.reshape(1, -1).mean(axis=1)[0]
    n_t = max(int(transition.sum()), 1) if n_transition is None else int(n_transition)
    n_bad_t = int(bad[transition].sum())
    clean = bool(g <= clean_max_bad)
    if clean:
        pct = round(float(g) * 100, 4)
    else:
        pct = round(float(n_bad_t) / n_t * 100, 4)
    return {"cloud_pct": pct, "usable": bool(pct / 100.0 <= cloud_threshold), "clean": clean,
            "g": float(g), "n_bad_frame": int(bad.sum()), "n_bad_transition": n_bad_t,
            "nodata_share": round(float(nodata.mean()), 6), "static_bad_masked": masked_bad}


# ─────────────────────────────────────────────────────────────────────────────
#  npz members: headers, row streaming, streamed writing
# ─────────────────────────────────────────────────────────────────────────────

def _npy_header(f):
    from numpy.lib import format as fmt

    v = fmt.read_magic(f)
    if v == (1, 0):
        return fmt.read_array_header_1_0(f)
    if v == (2, 0):
        return fmt.read_array_header_2_0(f)
    return fmt._read_array_header(f, v)  # noqa: SLF001


def _read_exact(f, n):
    if n == 0:
        return b""
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            raise EOFError("truncated npy member")
        parts.append(b)
        got += len(b)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def member_layout(path, key):
    """(shape, fortran, dtype) of the member ``key`` of an npz (header only)."""
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as f:
        return _npy_header(f)


def iter_rows(path, key, rows):
    """Yield ``(row, member[row])`` for the sorted unique ``rows`` of the 2-D
    C-order member ``key``, streamed one row at a time (never held whole)."""
    want = [int(r) for r in rows]
    if any(b <= a for a, b in zip(want, want[1:])):
        raise ValueError("rows must be sorted and unique")
    if not want:
        return
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as f:
        shape, fortran, dtype = _npy_header(f)
        if fortran or len(shape) != 2:
            raise ValueError(f"{path}:{key}: layout {shape} (fortran={fortran})")
        T, P = (int(v) for v in shape)
        if want[-1] >= T:
            raise IndexError(f"{path}:{key}: row {want[-1]} >= {T}")
        nrow = P * dtype.itemsize
        k = 0
        for t in range(want[-1] + 1):
            buf = _read_exact(f, nrow)
            if t == want[k]:
                yield t, np.frombuffer(buf, dtype).copy()
                k += 1


def write_npz(path, small, row_members, compresslevel=1):
    """An npz at ``path`` whose 2-D members are written row by row: ``small``
    {key: array}; ``row_members`` {key: (dtype, (T, P), iterator of rows)}
    in order (an iterator is consumed only when its member is written).
    ``compresslevel`` None: stored, else deflate at that level."""
    from numpy.lib import format as fmt

    comp = zipfile.ZIP_STORED if compresslevel is None else zipfile.ZIP_DEFLATED
    kw = {} if compresslevel is None else {"compresslevel": int(compresslevel)}
    with zipfile.ZipFile(path, "w", compression=comp, allowZip64=True, **kw) as zf:
        for k, (dtype, shape, rows) in row_members.items():
            dtype = np.dtype(dtype)
            with zf.open(k + ".npy", "w", force_zip64=True) as f:
                fmt.write_array_header_1_0(f, {"descr": fmt.dtype_to_descr(dtype), "fortran_order": False,
                                               "shape": tuple(int(v) for v in shape)})
                n = 0
                for r in rows:
                    r = np.ascontiguousarray(r, dtype=dtype)
                    if r.shape != (int(shape[1]),):
                        raise ValueError(f"{k}: row shape {r.shape} != ({shape[1]},)")
                    f.write(r.tobytes())
                    n += 1
                if n != int(shape[0]):
                    raise ValueError(f"{k}: {n} rows written, {shape[0]} declared")
        for k, a in small.items():
            with zf.open(k + ".npy", "w", force_zip64=True) as f:
                fmt.write_array(f, np.asanyarray(a), allow_pickle=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  inputs: paths, production extraction, record, overpass
# ─────────────────────────────────────────────────────────────────────────────

def site_inputs(site, args):
    """The input and output paths of a site (the server layout; every one
    can be overridden on the command line)."""
    from experiments.validation_grid import dutch_cube, dutch_products

    if site not in TARGET_SITES:
        raise SystemExit(f"unknown site {site!r}: this converter serves {list(TARGET_SITES)}")
    p = dutch_products(site)
    g = lambda k: getattr(args, k, None)  # noqa: E731
    work = resolve(g("work") or WORK_DIR)
    return {"cube": resolve(g("cube") or dutch_cube(site)),
            "extraction": resolve(g("extraction") or f"{p}/marea_extract.npz"),
            "result": resolve(g("result_json") or f"{p}/marea_{DUTCH_TAGS[site]}/result.json"),
            "overpass": resolve(g("prod_overpass") or f"{p}/overpass_times.json"),
            "work": work, "extra_dir": resolve(g("extra_dir") or EXTRA_DIR),
            "reference_npz": os.path.join(work, f"{site}_reference.npz"),
            "reference_json": os.path.join(work, f"{site}_reference.json")}


def gate_path(work, site, year=GATE_YEAR):
    return os.path.join(work, f"{site}_gate_{int(year)}.json")


def load_production(path):
    """The production extraction's small members (never Y / C whole):
    keep, shape, dates, sea / inter, the Y / C layouts and a fingerprint."""
    if not os.path.isfile(path):
        raise Refused(f"the production extraction {rel(path)} is not on this machine")
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        need = {"Y", "C", "keep", "dates", "shape"}
        if need - files:
            raise Refused(f"{rel(path)}: missing members {sorted(need - files)}")
        small = {"keep": np.asarray(z["keep"]), "shape": np.asarray(z["shape"])}
        for k in ("sea", "inter"):
            if k in files:
                small[k] = np.asarray(z[k])
        dates = np.asarray(z["dates"]).astype(str)
    keep = small["keep"]
    shape = tuple(int(v) for v in small["shape"])
    if keep.ndim != 1 or keep.dtype.kind not in "iu":
        raise Refused(f"{rel(path)}: keep is {keep.dtype} {keep.shape}, not 1-D integers")
    if len(dates) > 1 and np.any(dates[1:] <= dates[:-1]):
        raise Refused(f"{rel(path)}: dates are not sorted and unique")
    lay = {}
    for k, want in (("Y", np.float32), ("C", np.bool_)):
        shp, fortran, dt = member_layout(path, k)
        if fortran or tuple(int(v) for v in shp) != (len(dates), len(keep)) or dt != np.dtype(want):
            raise Refused(f"{rel(path)}:{k}: {dt} {shp} fortran={fortran}, expected {np.dtype(want)} "
                          f"({len(dates)}, {len(keep)}) C order")
        lay[k] = [int(v) for v in shp]
    for k in ("sea", "inter"):
        if k in small and (small[k].shape != shape or small[k].dtype != np.bool_):
            raise Refused(f"{rel(path)}:{k}: {small[k].dtype} {small[k].shape}, expected bool {shape}")
    return {"path": path, "keep": keep, "shape": shape, "dates": dates, "date_list": [str(d) for d in dates],
            "small": small, "layout": lay, "files": sorted(files), "fingerprint": zip_fingerprint(path),
            "keep_sha256": array_sha256(keep)}


def load_record_dates(path):
    if not os.path.isfile(path):
        raise Refused(f"the production result.json {rel(path)} is not on this machine")
    res = read_json(path)
    d = [str(x)[:10] for x in res.get("record_dates") or []]
    if not d:
        raise Refused(f"{rel(path)}: no record_dates")
    if any(b <= a for a, b in zip(d, d[1:])):
        raise Refused(f"{rel(path)}: record_dates are not ascending")
    return d


def load_overpass_file(path, required=True):
    """{date: 'YYYY-MM-DDTHH:MM:SS'} of an overpass file (pipeline format)."""
    if not os.path.isfile(path):
        if required:
            raise Refused(f"the overpass file {rel(path)} is not on this machine")
        return {}
    raw = read_json(path)
    out, bad = {}, []
    for k, v in raw.items():
        if v is None or str(v) in ("", "nan", "None"):
            continue
        iso = iso_instant(v)
        if iso[:10] != str(k)[:10]:
            bad.append((k, v))
        out[str(k)[:10]] = iso
    if bad:
        raise Refused(f"{rel(path)}: overpass times on another day than their date: {bad[:5]}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the stores
# ─────────────────────────────────────────────────────────────────────────────

def find_store(site, year, roots):
    """The first complete ``<root>/<site>/<year>.zarr`` in ``roots`` order."""
    from experiments import zarr_cubes as zc

    tried = []
    for r in roots:
        p = zc.store_path(site, year, resolve(r))
        if zc.is_complete(p):
            return str(p), r
        tried.append(f"{rel(p)}: " + ("not complete" if p.exists() else "absent"))
    raise Refused(f"{site} {year}: no complete store ({'; '.join(tried)})")


def store_kind(bands):
    b = set(bands)
    if b == set(BANDS):
        return "light"
    if set(NINE_BANDS) <= b:
        return "9-band"
    return "other"


def registration_status(attrs, site, frame_shape):
    """The store's registration record (download_cube_zarr.register_delivery)
    against this script's rule: method 'netcdf' and verified, or
    '--offset-px' / 'sibling store' confirmed by the median composites; at
    the site's known offset, delivered shape and frame."""
    reg = _attr_json(attrs.get("registration")) or {}
    if not isinstance(reg, dict):
        reg = {}
    method = reg.get("method") or attrs.get("registration_method")
    offset = reg.get("offset", _attr_json(attrs.get("delivered_offset_px")))
    dshape = reg.get("delivered_shape", _attr_json(attrs.get("delivered_shape_px")))
    checked = str(reg.get("checked") or "")
    verified = reg.get("verified") is True
    problems = []
    if method == "netcdf":
        if not verified:
            problems.append("registered against the netCDF but not verified")
    elif method == "expert --offset-px":
        if not checked.startswith(REG_CONFIRMED):
            problems.append(f"--offset-px not {REG_CONFIRMED} ({checked or 'no check recorded'})")
    elif method == "sibling store":
        pass                     # taken by the downloader only when the composites confirm it
    else:
        problems.append(f"registration method {method!r} does not establish the offset")
    want_off, want_shape = SITE_OFFSET.get(site), SITE_DELIVERED_SHAPE.get(site)
    if offset is None or [int(v) for v in offset] != list(want_off):
        problems.append(f"offset {offset} is not the site's {list(want_off)}")
    if dshape is None or [int(v) for v in dshape] != list(want_shape):
        problems.append(f"delivered shape {dshape} is not the site's {list(want_shape)}")
    fs = reg.get("frame_shape")
    if fs is not None and [int(v) for v in fs] != [int(v) for v in frame_shape]:
        problems.append(f"registration frame {fs} is not the production frame {list(frame_shape)}")
    return {"method": method, "offset": offset, "delivered_shape": dshape, "verified": verified,
            "checked": checked or None, "reference": reg.get("reference") or attrs.get("registration_reference"),
            "why": str(reg.get("why") or "")[:800], "ok": not problems, "problems": problems}


def open_checked_store(path, site, frame, year):
    """(dataset raw, dates, info, problems, attrs) of one yearly store: B03 /
    B08 / SCL int16 with fill -32768, the production grid (the legacy cube's
    x / y / CRS), dates of its year. The dataset is lazy (dask); the
    registration rule is :func:`registration_status`'s."""
    from experiments import zarr_cubes as zc

    attrs = zc.store_attrs(path)
    info = {"path": rel(path), "year": int(year)}
    problems = []
    try:
        ds = zc.open_store(path, bands=list(BANDS), masked=False)
    except (KeyError, FileNotFoundError) as e:
        raise Refused(f"{rel(path)}: {e}") from None
    bands_attr = _attr_json(attrs.get("band_subset") or attrs.get("bands"))
    if isinstance(bands_attr, list):
        names = [str(b) for b in bands_attr]
    else:
        full = zc.open_store(path)
        names = zc.band_names(full)
        full.close()
    info.update(bands=names, kind=store_kind(names), band_subset=_attr_json(attrs.get("band_subset")),
                nine_band_root=attrs.get("nine_band_root"),
                zmetadata_sha256=sha256_file(os.path.join(path, ".zmetadata")),
                job_id=attrs.get("job_id"), job_title=attrs.get("job_title"),
                processing_date=attrs.get("processing_date"), written_by=attrs.get("written_by"),
                input_baselines=zc.store_baselines(attrs), radiometry=zc.store_radiometry(attrs),
                verification=_attr_json(attrs.get("verification")),
                light_gate=_attr_json(attrs.get("light_gate")))
    for b in BANDS:
        dt, f = str(ds[b].dtype), zc.fill_of(ds[b])
        if dt != "int16" or f != FILL:
            problems.append(f"{b} is {dt} with fill {f}, expected int16 with fill {FILL}")
    H, W = frame["shape"]
    if (ds.sizes.get("y"), ds.sizes.get("x")) != (H, W):
        problems.append(f"grid {ds.sizes.get('y')} x {ds.sizes.get('x')} is not the production frame {H} x {W}")
    else:
        same = (np.array_equal(ds["x"].values, frame["x"]) and np.array_equal(ds["y"].values, frame["y"]))
        wkt = ds["crs"].attrs.get("crs_wkt") if "crs" in ds.variables else None
        crs_ok = zc.crs_equal(wkt, frame["crs_wkt"])
        if not same or not crs_ok:
            problems.append(f"x / y / CRS are not the legacy cube's (axes equal {same}, CRS equal {crs_ok})")
    dates = zc.dates(ds)
    if any(b <= a for a, b in zip(dates, dates[1:])):
        problems.append("dates are not strictly increasing")
    other = [d for d in dates if d[:4] != str(year)]
    if other:
        problems.append(f"{len(other)} date(s) outside {year}: {other[:4]}")
    if not dates:
        problems.append("no dates")
    info["dates"] = len(dates)
    info["first_date"], info["last_date"] = (dates[0], dates[-1]) if dates else (None, None)
    return ds, dates, info, problems, attrs


def checked_store(site, year, roots, frame, allow_unconfirmed=False):
    """One yearly store found in ``roots``, opened and checked; refused on a
    problem (registration problems: unless ``allow_unconfirmed``, recorded)."""
    path, root = find_store(site, year, roots)
    ds, dates, info, problems, attrs = open_checked_store(path, site, frame, year)
    info["root"] = root
    reg = registration_status(attrs, site, frame["shape"])
    info["registration"] = reg
    if problems:
        ds.close()
        raise Refused(f"{rel(path)}: " + "; ".join(problems))
    if not reg["ok"]:
        if not allow_unconfirmed:
            ds.close()
            raise Refused(f"{rel(path)}: the registration is not established for an extraction: "
                          + "; ".join(reg["problems"]) + " (--allow-unconfirmed-registration records an override)")
        info["registration_override"] = "--allow-unconfirmed-registration: " + "; ".join(reg["problems"])
        log(f"WARNING {rel(path)}: registration not established ({'; '.join(reg['problems'])}); "
            "taken by --allow-unconfirmed-registration (recorded)")
    return {"ds": ds, "dates": dates, "info": info, "path": path}


def scene_stream(stores, keep, shape, transition, static, t_block=T_BLOCK, want_dn=False):
    """Yield ``(date, Y row, C row, screening)`` for every date of ``stores``
    (in order), reading each block of ``t_block`` dates once: the full-frame
    SCL (the screening) and B03 / B08 over the keep pixels' bounding box.
    The screening carries the radiometry statistics of the keep pixels
    (``keep_b03_median_dn``, ``keep_b08_stats``) and, with ``want_dn``, the
    raw B03 / B08 at the keep pixels as ``dn`` (the gate's legacy check)."""
    H, W = (int(v) for v in shape)
    (r0, r1, c0, c1), rr, cc = keep_subframe(keep, shape)
    n_t = max(int(transition.sum()), 1)
    for st in stores:
        ds, dates = st["ds"], st["dates"]
        T = len(dates)
        t0 = time.time()
        for a in range(0, T, int(t_block)):
            b = min(T, a + int(t_block))
            scl = np.asarray(ds["SCL"].isel(t=slice(a, b)).values)
            g = np.asarray(ds["B03"].isel(t=slice(a, b), y=slice(r0, r1), x=slice(c0, c1)).values)
            n = np.asarray(ds["B08"].isel(t=slice(a, b), y=slice(r0, r1), x=slice(c0, c1)).values)
            if scl.shape != (b - a, H, W) or g.shape != (b - a, r1 - r0, c1 - c0) or n.shape != g.shape:
                raise RuntimeError(f"{st['info']['path']}: block {a}..{b} has shapes {scl.shape} {g.shape}")
            if scl.dtype != np.int16 or g.dtype != np.int16 or n.dtype != np.int16:
                raise RuntimeError(f"{st['info']['path']}: raw bands are not int16")
            for j in range(b - a):
                y, c = extraction_row(g[j], n[j], scl[j, r0:r1, c0:c1], rr, cc)
                s = screen_scene(scl[j], transition, static, n_transition=n_t)
                s["keep_b03_median_dn"] = keep_b03_median(g[j], c, rr, cc)
                s.update(keep_b08_stats(n[j], c, rr, cc))
                if want_dn:
                    s["dn"] = (g[j][rr, cc].copy(), n[j][rr, cc].copy())
                yield dates[a + j], y, c, s
            del scl, g, n
            log(f"{st['info']['path']}: {b} of {T} dates [{time.time() - t0:.0f} s]")


# ─────────────────────────────────────────────────────────────────────────────
#  the production reference map, the static no-data mask, the legacy grid
# ─────────────────────────────────────────────────────────────────────────────

def legacy_grid(cube_path):
    """x, y, CRS, dates and the time dimension of the legacy netCDF cube."""
    import xarray as xr

    if not os.path.isfile(cube_path):
        raise Refused(f"the legacy cube {rel(cube_path)} is not on this machine")
    with xr.open_dataset(cube_path, mask_and_scale=False) as ds:
        tdim = "t" if "t" in ds["SCL"].dims else ("time" if "time" in ds["SCL"].dims else
                                                  [d for d in ds["SCL"].dims if d not in ("y", "x")][0])
        wkt = None
        for v in ("crs", "spatial_ref"):
            if v in ds.variables:
                wkt = ds[v].attrs.get("crs_wkt") or ds[v].attrs.get("spatial_ref")
                if wkt:
                    break
        dates = [str(np.datetime64(v, "D")) for v in ds[tdim].values]
        return {"x": np.asarray(ds["x"].values, np.float64), "y": np.asarray(ds["y"].values, np.float64),
                "crs_wkt": str(wkt or ""), "dates": dates, "tdim": tdim,
                "shape": (int(ds.sizes["y"]), int(ds.sizes["x"])),
                "scl_fill": None if ds["SCL"].attrs.get("_FillValue") is None
                else int(np.asarray(ds["SCL"].attrs["_FillValue"]).item()),
                "bands": {b: {k: str(v) for k, v in ds[b].attrs.items()
                              if k in ("_FillValue", "scale_factor", "add_offset", "units")}
                          for b in BANDS if b in ds.variables}}


def production_reference(site, cube_path):
    """The production reference map and cloud_pct, recomputed exactly as the
    notebooks' cell 12 (``reference_and_clouds`` on the cached cube)."""
    import pyintertidal as pit

    cube = pit.SentinelCube(pit.sites.get(site), TIME_EXTENT, water="ndwi", cache_path=cube_path,
                            resolution=10)
    ref, cloud = pit.reference_and_clouds(
        cube, threshold=NDWI_THRESHOLD, bad_classes=pit.water.BAD_CLASSES,
        clean_scene_max_bad=CLEAN_SCENE_MAX_BAD, stable_threshold=STABLE_THRESHOLD,
        transition_buffer_px=TRANSITION_BUFFER_PX)
    return np.asarray(ref).astype(np.uint8), {str(k)[:10]: float(v) for k, v in dict(cloud).items()}


def load_map_file(path, shape):
    if str(path).lower().endswith(".npy"):
        ref = np.load(path, allow_pickle=False)
    else:
        import rasterio
        with rasterio.open(path) as src:
            ref = src.read(1)
    ref = np.asarray(ref)
    if tuple(ref.shape) != tuple(shape):
        raise Refused(f"{rel(path)}: map {ref.shape} is not the frame {tuple(shape)}")
    if not set(np.unique(ref).tolist()) <= {0, 1, 2}:
        raise Refused(f"{rel(path)}: values {np.unique(ref)[:10]} are not the classes 0 / 1 / 2")
    return ref.astype(np.uint8)


def legacy_scl_pass(cube_path, grid, transition, t_block=T_BLOCK):
    """One pass over the legacy SCL (raw int16): the static no-data mask
    (no valid SCL, i.e. fill or class 0, on every date), the same per year,
    and :func:`screen_scene`'s cloud_pct of every scene on the map."""
    import xarray as xr

    H, W = grid["shape"]
    fill = grid.get("scl_fill")
    any_valid = np.zeros((H, W), bool)
    per_year = {}
    cloud, detail = {}, {}
    n_t = max(int(transition.sum()), 1)
    with xr.open_dataset(cube_path, mask_and_scale=False) as ds:
        da = ds["SCL"].transpose(grid["tdim"], "y", "x")
        dates = grid["dates"]
        T = len(dates)
        t0 = time.time()
        for a in range(0, T, int(t_block)):
            b = min(T, a + int(t_block))
            blk = np.asarray(da.isel({grid["tdim"]: slice(a, b)}).values)
            if blk.dtype.kind == "f":
                valid = np.isfinite(blk) & (blk != 0)
            else:
                valid = (blk != FILL) & (blk != 0)
                if fill is not None and fill != FILL:
                    valid &= blk != fill
                    blk = np.where(blk == fill, FILL, blk)
            any_valid |= valid.any(axis=0)
            for j in range(b - a):
                d = dates[a + j]
                yv = per_year.setdefault(d[:4], np.zeros((H, W), bool))
                yv |= valid[j]
                s = screen_scene(blk[j], transition, None, n_transition=n_t)
                cloud[d] = s["cloud_pct"]
                detail[d] = {k: s[k] for k in ("clean", "g", "n_bad_transition", "nodata_share")}
            del blk, valid
            log(f"legacy SCL pass: {b} of {T} dates [{time.time() - t0:.0f} s]")
    static = ~any_valid
    return static, {y: int((~v).sum()) for y, v in sorted(per_year.items())}, cloud, detail


def load_reference(site, inputs, need_ok=True):
    """The reference npz (map, static mask, legacy grid) checked against its
    JSON (sha256) and, by default, refused unless its gates passed."""
    jp, zp = inputs["reference_json"], inputs["reference_npz"]
    if not (os.path.isfile(jp) and os.path.isfile(zp)):
        raise Refused(f"no reference for {site} in {rel(inputs['work'])}: run "
                      f"`python -X utf8 -m experiments.light_to_extraction reference {site}` first")
    J = read_json(jp)
    sha = sha256_file(zp)
    if sha != (J.get("npz") or {}).get("sha256"):
        raise Refused(f"{rel(zp)}: sha256 {sha[:12]} is not the one its JSON records")
    if need_ok and not J.get("ok"):
        raise Refused(f"{rel(jp)}: the reference gates did not pass ({J.get('failed')})")
    with np.load(zp, allow_pickle=False) as z:
        ref = np.asarray(z["reference_map"])
        static = np.asarray(z["static_nodata"]).astype(bool)
        frame = {"x": np.asarray(z["x"], np.float64), "y": np.asarray(z["y"], np.float64),
                 "crs_wkt": str(z["crs_wkt"]), "shape": tuple(int(v) for v in ref.shape)}
    return {"ref": ref, "transition": ref == 0, "static": static, "frame": frame, "json": J, "sha256": sha,
            "npz": zp, "json_path": jp,
            "summary": {"path": rel(zp), "sha256": sha, "json_sha256": sha256_file(jp),
                        "classes": (J.get("map") or {}).get("classes"),
                        "equal_to_production_print": (J.get("map") or {}).get("equal_to_production_print"),
                        "source": (J.get("map") or {}).get("source"),
                        "static_nodata_px": (J.get("static_nodata") or {}).get("px")}}


def cmd_reference(args):
    """The production reference map (recomputed with the production function
    on the legacy cube, or imported), the static no-data mask and the legacy
    grid, gated; into ``<work>/<site>_reference.{npz,json}``."""
    site = args.site
    I = site_inputs(site, args)
    prod = load_production(I["extraction"])
    rec_dates = load_record_dates(I["result"])
    over = load_overpass_file(I["overpass"])
    grid = legacy_grid(I["cube"])
    if grid["shape"] != prod["shape"]:
        raise Refused(f"the legacy cube {grid['shape']} is not the production frame {prod['shape']}")
    if prod["date_list"] != list(grid["dates"]):
        a, b = set(prod["date_list"]), set(grid["dates"])
        raise Refused(f"the production extraction's dates are not the legacy cube's (only in the extraction "
                      f"{sorted(a - b)[:5]}, only in the cube {sorted(b - a)[:5]})")
    t0 = time.time()
    if args.from_map:
        ref = load_map_file(resolve(args.from_map), grid["shape"])
        cloud_prod = None
        source = f"imported {rel(resolve(args.from_map))} (sha256 {sha256_file(resolve(args.from_map))})"
        log(f"[{site}] reference map imported from {rel(resolve(args.from_map))}")
    else:
        log(f"[{site}] recomputing the production reference map: reference_and_clouds on {rel(I['cube'])} "
            f"({len(grid['dates'])} scenes; the notebooks' call)")
        ref, cloud_prod = production_reference(site, I["cube"])
        source = ("recomputed: pyintertidal.stability.reference_and_clouds(SentinelCube(sites.get(site), "
                  f"{TIME_EXTENT}, water='ndwi', cache_path={rel(I['cube'])}), threshold={NDWI_THRESHOLD}, "
                  f"bad_classes={BAD}, clean_scene_max_bad={CLEAN_SCENE_MAX_BAD}, stable_threshold="
                  f"{STABLE_THRESHOLD}, transition_buffer_px={TRANSITION_BUFFER_PX}) as the notebooks' cell 12")
        log(f"[{site}] reference_and_clouds done [{time.time() - t0:.0f} s]")
    classes = {int(c): int((ref == c).sum()) for c in (0, 1, 2)}
    exp = EXPECTED_REFERENCE[site]
    trans = ref == 0
    static, per_year, cloud_mine, detail = legacy_scl_pass(I["cube"], grid, trans, args.t_block)
    keep_in_static = int(static.ravel()[prod["keep"]].sum())
    gates = {}
    over_ok = bool(args.allow_reference_mismatch)
    g_cls = classes == exp["classes"]
    gates["classes_equal_production_print"] = {"ok": g_cls or over_ok, "equal": g_cls, "mine": classes,
                                               "printed": exp["classes"],
                                               **({"overridden": True} if over_ok and not g_cls else {})}
    g_n = len(grid["dates"]) == exp["cube_scenes"]
    gates["cube_scenes_equal_production_print"] = {"ok": g_n or over_ok, "equal": g_n, "mine": len(grid["dates"]),
                                                   "printed": exp["cube_scenes"],
                                                   **({"overridden": True} if over_ok and not g_n else {})}
    cmp_prod = None
    if cloud_prod is not None:
        same_keys = sorted(cloud_prod) == sorted(cloud_mine)
        diffs = {d: [cloud_prod[d], cloud_mine.get(d)] for d in cloud_prod if cloud_prod[d] != cloud_mine.get(d)}
        cmp_prod = {"dates": len(cloud_prod), "same_dates": same_keys, "differ": len(diffs),
                    "first_differences": dict(list(diffs.items())[:GATE_LIST])}
        gates["screen_scene_equals_reference_and_clouds"] = {"ok": bool(same_keys and not diffs), **cmp_prod}
    cloud = cloud_prod if cloud_prod is not None else cloud_mine
    usable = sorted(d for d, p in cloud.items() if p / 100.0 <= CLOUD_THRESHOLD)
    pool = [d for d in usable if d in over]
    gates["usable_with_overpass_equals_record_dates"] = {
        "ok": pool == rec_dates, "n_usable": len(usable), "n_pool": len(pool), "n_record": len(rec_dates),
        "only_pool": sorted(set(pool) - set(rec_dates)), "only_record": sorted(set(rec_dates) - set(pool)),
        "usable_without_overpass": [d for d in usable if d not in over]}
    gates["keep_outside_static_mask"] = {"ok": keep_in_static == 0, "keep_px_inside": keep_in_static}
    ok = all(g["ok"] for g in gates.values())
    ensure_ignored(I["work"])
    tmp = I["reference_npz"] + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, reference_map=ref.astype(np.uint8), static_nodata=static, x=grid["x"],
                            y=grid["y"], crs_wkt=np.array(grid["crs_wkt"]))
    os.replace(tmp, I["reference_npz"])
    st = os.stat(I["cube"])
    J = {"site": site, "utc": utcnow(), "ok": ok, "failed": [k for k, g in gates.items() if not g["ok"]],
         "code": {"version": CODE_VERSION, "sha256": sha256_file(__file__)},
         "cube": {"path": rel(I["cube"]), "size": int(st.st_size), "dates": len(grid["dates"]),
                  "first": grid["dates"][0], "last": grid["dates"][-1], "grid": list(grid["shape"]),
                  "bands": grid["bands"], "scl_fill": grid["scl_fill"]},
         "map": {"source": source, "classes": classes, "expected": exp["classes"],
                 "equal_to_production_print": g_cls,
                 "params": {"threshold": NDWI_THRESHOLD, "bad_classes": list(BAD),
                            "clean_scene_max_bad": CLEAN_SCENE_MAX_BAD, "stable_threshold": STABLE_THRESHOLD,
                            "transition_buffer_px": TRANSITION_BUFFER_PX, "time_extent": list(TIME_EXTENT)}},
         "static_nodata": {"rule": ("pixels whose legacy SCL is no data (fill or class 0) on EVERY date of the "
                                    "legacy cube; set to no data in every store's SCL before the screening "
                                    "(pre-registered; production never counted them bad)"),
                           "px": int(static.sum()), "share": round(float(static.mean()), 6),
                           "transition_px_inside": int((static & trans).sum()),
                           "per_year_no_valid_px": per_year, "keep_px_inside": keep_in_static},
         "screening": {"cloud_pct_reference_and_clouds": cloud_prod, "cloud_pct_screen_scene": cloud_mine,
                       "screen_scene_vs_reference_and_clouds": cmp_prod, "usable": usable,
                       "pool_with_overpass": pool, "per_date": detail},
         "inputs": {"production_extraction": prod["fingerprint"], "keep_sha256": prod["keep_sha256"],
                    "result_json": file_info(I["result"]), "production_overpass": file_info(I["overpass"])},
         "gates": gates, "npz": {"path": rel(I["reference_npz"]), "sha256": sha256_file(I["reference_npz"])}}
    write_json(I["reference_json"], J)
    log(f"[{site}] reference map classes {classes} (printed {exp['classes']}: "
        f"{'equal' if g_cls else 'DIFFERENT'}); static no-data {int(static.sum()):,} px "
        f"({100 * static.mean():.3f} %), {keep_in_static} keep px inside")
    for k, g in gates.items():
        log(f"[{site}] gate {k}: {'ok' if g['ok'] else 'FAILED'}"
            + ("" if g["ok"] else f" {json.dumps({kk: v for kk, v in g.items() if kk != 'ok'}, default=str)[:600]}"))
    log(f"[{site}] -> {rel(I['reference_npz'])}, {rel(I['reference_json'])}: {'OK' if ok else 'NOT OK'}")
    return 0 if ok else 2


# ─────────────────────────────────────────────────────────────────────────────
#  the 2023 gate
# ─────────────────────────────────────────────────────────────────────────────

class RowCompare:
    """Accumulated comparison of store rows with production rows, attributed."""

    KEYS = ("pixels", "nan_both", "nan_only_production", "nan_only_store", "bit_identical",
            "value_differs", "signed_zero_only", "c_differs", "c_store_true_production_false",
            "c_store_false_production_true", "c_differs_where_production_nan")

    def __init__(self):
        self.n = dict.fromkeys(self.KEYS, 0)
        self.max_abs = 0.0
        self.rows = 0
        self.per_date = {}

    def add(self, date, ys, cs, yp, cp):
        ns, npn = np.isnan(ys), np.isnan(yp)
        fin = ~ns & ~npn
        bits = fin & (ys.view(np.uint32) != yp.view(np.uint32))
        vals = fin & (ys != yp)
        cd = cs != cp
        d = {"nan_both": int((ns & npn).sum()), "nan_only_production": int((npn & ~ns).sum()),
             "nan_only_store": int((ns & ~npn).sum()), "bit_identical": int((fin & ~bits).sum()),
             "value_differs": int(vals.sum()), "signed_zero_only": int((bits & ~vals).sum()),
             "c_differs": int(cd.sum()), "c_store_true_production_false": int((cs & ~cp).sum()),
             "c_store_false_production_true": int((~cs & cp).sum()),
             "c_differs_where_production_nan": int((cd & npn).sum())}
        self.rows += 1
        self.n["pixels"] += int(ys.size)
        for k, v in d.items():
            self.n[k] += v
        if d["value_differs"]:
            dv = np.abs(ys[vals].astype(np.float64) - yp[vals].astype(np.float64))
            self.max_abs = max(self.max_abs, float(dv.max()))
        bad = d["nan_only_production"] + d["nan_only_store"] + d["value_differs"] + d["signed_zero_only"] \
            + d["c_differs"]
        if bad:
            self.per_date[date] = {k: v for k, v in d.items() if v and k not in ("nan_both", "bit_identical")}

    def report(self):
        n = self.n
        y_eq = not (n["nan_only_production"] or n["nan_only_store"] or n["value_differs"] or n["signed_zero_only"])
        worst = sorted(self.per_date.items(), key=lambda kv: -sum(kv[1].values()))
        fin = n["bit_identical"] + n["value_differs"] + n["signed_zero_only"]
        pct = (lambda a, b: round(100.0 * a / b, 4) if b else 0.0)
        return {"rows_compared": self.rows, **n, "finite_both": fin,
                "Y_equal": bool(y_eq), "C_equal": n["c_differs"] == 0,
                "max_abs_diff": self.max_abs, "dates_with_differences": len(self.per_date),
                "shares_pct": {"Y_value_differs_of_finite_both": pct(n["value_differs"], fin),
                               "Y_nan_mismatch_of_pixels": pct(n["nan_only_production"] + n["nan_only_store"],
                                                               n["pixels"]),
                               "C_differs_of_pixels": pct(n["c_differs"], n["pixels"])},
                "worst_dates": dict(worst[:GATE_LIST]),
                "tolerance": ("none: float32 exactness (bit-identical finite values, NaN at the same "
                              "pixels); identical digital numbers give identical float32 NDWI")}


class DnCompare:
    """The store's raw B03 / B08 against the legacy cube's at the keep pixels
    (the digital numbers behind production's Y): valid pixel-dates (data in
    both), how many differ and by how much, and no data on one side only."""

    BANDS = ("B03", "B08")

    def __init__(self, legacy_fills):
        self.fills = dict(legacy_fills)
        self.per = {b: dict.fromkeys(("valid_both", "differ", "nodata_only_legacy", "nodata_only_store",
                                      "nodata_both"), 0) for b in self.BANDS}
        self.max_abs = dict.fromkeys(self.BANDS, 0)
        self.per_date = {}

    def add(self, date, store, legacy):
        row = {}
        for b, s, lg in zip(self.BANDS, store, legacy):
            s = np.asarray(s).astype(np.int32)
            lg = np.asarray(lg).astype(np.int32)
            vs, vl = s != FILL, lg != int(self.fills.get(b, FILL))
            both = vs & vl
            dif = both & (s != lg)
            p = self.per[b]
            p["valid_both"] += int(both.sum())
            p["differ"] += int(dif.sum())
            p["nodata_only_legacy"] += int((vs & ~vl).sum())
            p["nodata_only_store"] += int((~vs & vl).sum())
            p["nodata_both"] += int((~vs & ~vl).sum())
            if dif.any():
                self.max_abs[b] = max(self.max_abs[b], int(np.abs(s[dif] - lg[dif]).max()))
                row[b] = int(dif.sum())
        if row:
            self.per_date[date] = row

    def report(self):
        tot = {k: sum(p[k] for p in self.per.values()) for k in next(iter(self.per.values()))}
        pct = (lambda a, b: round(100.0 * a / b, 4) if b else 0.0)
        per = {b: dict(p, differ_pct=pct(p["differ"], p["valid_both"]), max_abs_dn=self.max_abs[b])
               for b, p in self.per.items()}
        worst = sorted(self.per_date.items(), key=lambda kv: -sum(kv[1].values()))
        return {"what": ("the store's raw B03 / B08 digital numbers against the legacy 10 m cube's at the keep "
                         "pixels, on the common dates (production's Y is computed from the legacy ones)"),
                "per_band": per, **tot, "differ_pct_of_valid_both": pct(tot["differ"], tot["valid_both"]),
                "max_abs_dn": max(self.max_abs.values()), "equal": tot["differ"] == 0
                and tot["nodata_only_legacy"] == 0 and tot["nodata_only_store"] == 0,
                "dates_with_differences": len(self.per_date), "worst_dates": dict(worst[:GATE_LIST])}


def legacy_keep_dn(cube_path, grid, dates, keep, shape, t_block=T_BLOCK):
    """Yield ``(date, (B03, B08))``: the legacy cube's raw int16 digital
    numbers at the keep pixels for ``dates`` (in order), read in blocks of
    ``t_block`` dates over the keep pixels' bounding box."""
    import xarray as xr

    (r0, r1, c0, c1), rr, cc = keep_subframe(keep, shape)
    pos = {d: i for i, d in enumerate(grid["dates"])}
    missing = [d for d in dates if d not in pos]
    if missing:
        raise Refused(f"the legacy cube {rel(cube_path)} lacks the dates {missing[:5]}")
    tdim = grid["tdim"]
    with xr.open_dataset(cube_path, mask_and_scale=False) as ds:
        das = {b: ds[b].transpose(tdim, "y", "x") for b in DnCompare.BANDS}
        for a in range(0, len(dates), int(t_block)):
            sel = list(dates[a:a + int(t_block)])
            idx = [pos[d] for d in sel]
            blk = {b: np.asarray(v.isel({tdim: idx, "y": slice(r0, r1), "x": slice(c0, c1)}).values)
                   for b, v in das.items()}
            for j, d in enumerate(sel):
                yield d, tuple(blk[b][j][rr, cc] for b in DnCompare.BANDS)
            del blk


def legacy_fills(grid):
    """The legacy cube's no-data value of B03 / B08 (its ``_FillValue``)."""
    out = {}
    for b in DnCompare.BANDS:
        v = (grid.get("bands") or {}).get(b, {}).get("_FillValue")
        try:
            out[b] = int(float(v)) if v is not None else FILL
        except ValueError:
            out[b] = FILL
    return out


def only_extraction_differences(g):
    """True when a gate's dates and screening reproduce production and the
    static mask holds: its only differences, if any, are extraction values
    (the case the user may accept, e.g. the Ems three-tile mosaic)."""
    return bool((g.get("dates") or {}).get("equal") and (g.get("screening") or {}).get("equal_record_dates")
                and (g.get("extraction") or {}).get("keep_px_in_static_mask") == 0)


def compute_gate(site, inputs, roots, year=GATE_YEAR, t_block=T_BLOCK, allow_unconfirmed=False):
    """The gate report of ``year`` (a dict; ``ok`` = dates, Y, C and the
    screened list all reproduce production)."""
    R = load_reference(site, inputs)
    prod = load_production(inputs["extraction"])
    rec_dates = load_record_dates(inputs["result"])
    over = load_overpass_file(inputs["overpass"])
    if R["frame"]["shape"] != prod["shape"]:
        raise Refused(f"the reference frame {R['frame']['shape']} is not the production frame {prod['shape']}")
    grid = legacy_grid(inputs["cube"])
    if tuple(grid["shape"]) != tuple(prod["shape"]):
        raise Refused(f"the legacy cube {grid['shape']} is not the production frame {prod['shape']}")
    st = checked_store(site, year, roots, R["frame"], allow_unconfirmed)
    s_dates = st["dates"]
    p_idx = {d: i for i, d in enumerate(prod["date_list"]) if d[:4] == str(year)}
    common = sorted(set(s_dates) & set(p_idx))
    rows = [p_idx[d] for d in common]
    log(f"[{site}] gate {year}: {st['info']['kind']} store {st['info']['path']} ({len(s_dates)} dates) vs the "
        f"production extraction ({len(p_idx)} dates of {year}, {len(prod['keep']):,} keep px) and the legacy "
        f"cube's B03 / B08 ({rel(inputs['cube'])})")
    yit = iter_rows(prod["path"], "Y", rows)
    cit = iter_rows(prod["path"], "C", rows)
    lit = legacy_keep_dn(inputs["cube"], grid, common, prod["keep"], prod["shape"], t_block)
    cmp_ = RowCompare()
    dn = DnCompare(legacy_fills(grid))
    cloud, detail = {}, {}
    try:
        for d, ys, cs, s in scene_stream([st], prod["keep"], prod["shape"], R["transition"], R["static"], t_block,
                                         want_dn=True):
            raw = s.pop("dn")
            cloud[d] = s["cloud_pct"]
            detail[d] = {k: s[k] for k in ("clean", "g", "n_bad_transition", "nodata_share", "static_bad_masked",
                                           "keep_b03_median_dn", "keep_valid_share", "keep_b08_low_share")}
            if d in p_idx:
                (ty, yp), (tc, cp) = next(yit), next(cit)
                if ty != p_idx[d] or tc != p_idx[d]:
                    raise RuntimeError(f"production rows out of step at {d}")
                cmp_.add(d, ys, cs, yp, cp.astype(bool))
                tl, leg = next(lit)
                if tl != d:
                    raise RuntimeError(f"legacy rows out of step at {d} ({tl})")
                dn.add(d, raw, leg)
    finally:
        st["ds"].close()
        yit.close()
        cit.close()
        lit.close()
    usable = sorted(d for d, p in cloud.items() if p / 100.0 <= CLOUD_THRESHOLD)
    screened = [d for d in usable if d in over]
    rec_y = [d for d in rec_dates if d[:4] == str(year)]
    legacy = (R["json"].get("screening") or {})
    leg = legacy.get("cloud_pct_reference_and_clouds") or legacy.get("cloud_pct_screen_scene") or {}
    vs = None
    if leg:
        both = [d for d in common if d in leg]
        diff = {d: [leg[d], cloud[d]] for d in both if leg[d] != cloud[d]}
        flips = [d for d in both if (leg[d] / 100.0 <= CLOUD_THRESHOLD) != (cloud[d] / 100.0 <= CLOUD_THRESHOLD)]
        vs = {"source": ("reference_and_clouds on the legacy cube" if legacy.get("cloud_pct_reference_and_clouds")
                         else "screen_scene on the legacy SCL"),
              "compared": len(both), "equal": len(both) - len(diff), "differ": len(diff),
              "max_abs_diff": max((abs(a - b) for a, b in diff.values()), default=0.0),
              "usable_flips": flips, "first_differences": dict(list(diff.items())[:GATE_LIST])}
    dates_rep = {"store": len(s_dates), "production": len(p_idx), "common": len(common),
                 "only_store": sorted(set(s_dates) - set(p_idx)), "only_production": sorted(set(p_idx) - set(s_dates))}
    dates_rep["equal"] = not dates_rep["only_store"] and not dates_rep["only_production"]
    ext = cmp_.report()
    ext["keep_px_in_static_mask"] = int(R["static"].ravel()[prod["keep"]].sum())
    scr = {"rule": ("production pass 2 on the fixed reference map (screen_scene), static no-data mask "
                    "applied; usable iff cloud_pct / 100 <= 0.10; screened = usable with a production "
                    "overpass time (the record rule; every usable date had a finite level at these sites)"),
           "static_mask_px": int(R["static"].sum()), "cloud_pct": cloud, "usable": usable, "screened": screened,
           "record_dates_year": rec_y, "equal_record_dates": screened == rec_y,
           "only_screened": sorted(set(screened) - set(rec_y)), "only_record": sorted(set(rec_y) - set(screened)),
           "usable_without_overpass": [d for d in usable if d not in over],
           "cloud_pct_all_equal_to_legacy": None if vs is None else bool(vs["differ"] == 0
                                                                         and vs["compared"] == len(common)),
           "vs_legacy_reproduction": vs, "per_date": detail}
    strict = bool(dates_rep["equal"] and ext["Y_equal"] and ext["C_equal"] and scr["equal_record_dates"]
                  and ext["keep_px_in_static_mask"] == 0)
    g = {"ok": strict, "strict_ok": strict, "site": site, "year": int(year), "utc": utcnow(),
         "rule": ("the converter's rows of the year's store equal the production extraction's rows on the "
                  "same dates at every keep pixel (Y bitwise with NaN positions equal, C bitwise), and the "
                  "screening of the store reproduces the year's part of result.json record_dates"),
         "store": st["info"], "dates": dates_rep, "extraction": ext, "legacy_dn": dn.report(), "screening": scr,
         "reference": R["summary"],
         "inputs": {"production_extraction": prod["fingerprint"], "keep_sha256": prod["keep_sha256"],
                    "result_json": file_info(inputs["result"]),
                    "production_overpass": file_info(inputs["overpass"]),
                    "reference_npz_sha256": R["sha256"]},
         "code": {"version": CODE_VERSION, "sha256": sha256_file(__file__)}}
    g["only_extraction_differences"] = (not strict) and only_extraction_differences(g)
    g["decision"] = ("passed: nothing to decide" if strict else
                     ("FAILED on extraction values only: the user's decision. `gate --accept-gate-differences` "
                      "(recorded here; the downloader then buys this site's pre-2023 years) or `build "
                      "--accept-gate-differences` (recorded) accepts them; otherwise the site keeps its 2023-2025 "
                      "scenes only") if g["only_extraction_differences"] else
                     "FAILED on the dates, the screening or the static mask: no flag accepts it")
    return g


def gate_lines(g):
    e, s, d = g["extraction"], g["screening"], g["dates"]
    out = [f"gate {g['year']} {g['site']}: {'PASSED' if g['ok'] else 'FAILED'} ({g['store']['kind']} store "
           f"{g['store']['path']})",
           f"  dates: store {d['store']}, production {d['production']}, common {d['common']}; only store "
           f"{d['only_store'][:6]}, only production {d['only_production'][:6]}",
           f"  Y: {e['bit_identical']:,} bit-identical, {e['nan_both']:,} NaN in both; NaN only in production "
           f"{e['nan_only_production']:,}, only in the store {e['nan_only_store']:,}; values differ "
           f"{e['value_differs']:,} (max |d| {e['max_abs_diff']:g}), signed zeros {e['signed_zero_only']:,}",
           f"  C: {e['c_differs']:,} differ ({e['c_differs_where_production_nan']:,} where production has no "
           f"NDWI); {e['dates_with_differences']} date(s) with differences; keep px in the static mask "
           f"{e['keep_px_in_static_mask']}",
           f"  screening: {len(s['screened'])} screened vs {len(s['record_dates_year'])} record dates "
           f"({'equal' if s['equal_record_dates'] else 'DIFFERENT'}); only screened {s['only_screened'][:6]}, "
           f"only record {s['only_record'][:6]}"]
    v = s.get("vs_legacy_reproduction")
    if v:
        out.append(f"  cloud_pct vs the legacy reproduction: {v['equal']} of {v['compared']} equal, max |d| "
                   f"{v['max_abs_diff']:g} %, usable flips {v['usable_flips'][:6]}")
    sh = e.get("shares_pct") or {}
    if sh:
        out.append(f"  shares: Y values differ on {sh.get('Y_value_differs_of_finite_both', 0):g} % of the "
                   f"pixel-dates finite in both, NaN mismatch {sh.get('Y_nan_mismatch_of_pixels', 0):g} %, C "
                   f"differs {sh.get('C_differs_of_pixels', 0):g} %")
    dn = g.get("legacy_dn")
    if dn:
        out.append(f"  B03 / B08 vs the legacy cube at the keep pixels: {dn['differ']:,} of {dn['valid_both']:,} "
                   f"valid pixel-dates differ ({dn['differ_pct_of_valid_both']:g} %; B03 "
                   f"{dn['per_band']['B03']['differ_pct']:g} %, B08 {dn['per_band']['B08']['differ_pct']:g} %), max "
                   f"|d| {dn['max_abs_dn']} DN; no data only in the legacy {dn['nodata_only_legacy']:,}, only in the "
                   f"store {dn['nodata_only_store']:,}")
    if g.get("decision"):
        out.append(f"  decision: {g['decision']}")
    if g.get("accepted"):
        out.append(f"  ACCEPTED by {g['accepted'].get('by')} ({g['accepted'].get('utc')})")
    return out


def acceptance_record(g, by):
    """The user's recorded acceptance of a gate failing on extraction values."""
    e, dn = g.get("extraction") or {}, g.get("legacy_dn") or {}
    return {"by": by, "utc": utcnow(),
            "differences": {k: e.get(k) for k in RowCompare.KEYS + ("max_abs_diff", "shares_pct")},
            "legacy_dn": {k: dn.get(k) for k in ("valid_both", "differ", "differ_pct_of_valid_both", "max_abs_dn",
                                                 "nodata_only_legacy", "nodata_only_store")}}


def cmd_gate(args):
    site = args.site
    I = site_inputs(site, args)
    roots = [r for r in (args.roots or ",".join(GATE_ROOTS)).split(",") if r.strip()]
    g = compute_gate(site, I, roots, args.year, args.t_block, args.allow_unconfirmed_registration)
    out = args.out or gate_path(I["work"], site, args.year)
    rc = 0 if g["ok"] else 2
    if args.accept_gate_differences and not g["ok"]:
        if g["only_extraction_differences"]:
            g["accepted"] = acceptance_record(g, "light_to_extraction gate --accept-gate-differences (the user's "
                                                 "decision: extraction values only)")
            rc = 0
        else:
            log(f"[{site}] --accept-gate-differences accepts extraction-value differences only; this gate also "
                "fails on the dates, the screening or the static mask: NOT accepted")
    if os.path.dirname(os.path.abspath(out)) == os.path.abspath(I["work"]):
        ensure_ignored(I["work"])
    write_json(out, g)
    for line in gate_lines(g):
        log(line)
    log(f"[{site}] -> {rel(out)}")
    return rc


def gate_for_build(site, inputs, R, prod, args):
    """The gate to embed (refusals and the recorded acceptance rules)."""
    p = args.gate or gate_path(inputs["work"], site)
    if not os.path.isfile(p):
        raise Refused(f"no gate report {rel(p)}: run `python -X utf8 -m experiments.light_to_extraction gate {site}`")
    g = read_json(p)
    stale = []
    if (g.get("code") or {}).get("version") != CODE_VERSION:
        stale.append(f"code version {(g.get('code') or {}).get('version')} != {CODE_VERSION}")
    if (g.get("inputs") or {}).get("reference_npz_sha256") != R["sha256"]:
        stale.append("another reference map / static mask")
    pf = ((g.get("inputs") or {}).get("production_extraction") or {}).get("members_sha256")
    if pf != prod["fingerprint"]["members_sha256"]:
        stale.append("another production extraction")
    gs = g.get("store") or {}
    zm = os.path.join(resolve(gs.get("path") or ""), ".zmetadata")
    if not os.path.isfile(zm) or sha256_file(zm) != gs.get("zmetadata_sha256"):
        stale.append(f"the gate's store {gs.get('path')} is gone or was rewritten since")
    if stale:
        raise Refused(f"{rel(p)} was computed with other inputs ({'; '.join(stale)}): run the gate again")
    g = dict(g, path=rel(p), sha256=sha256_file(p))
    if g.get("ok") is True:
        return g
    only_extraction = only_extraction_differences(g)
    if only_extraction and isinstance(g.get("accepted"), dict):
        log(f"WARNING [{site}] gate FAILED on the extraction values; accepted by the user at gate time "
            f"({g['accepted'].get('by')}, {g['accepted'].get('utc')}; recorded)")
        return dict(g, ok=True, strict_ok=False)
    if args.accept_gate_differences:
        if not only_extraction:
            raise Refused(f"{rel(p)}: --accept-gate-differences accepts extraction differences only; this gate "
                          "also fails on the dates, the screening or the static mask")
        e = g["extraction"]
        log(f"WARNING [{site}] gate FAILED on the extraction values (Y equal {e['Y_equal']}, C equal "
            f"{e['C_equal']}); accepted by --accept-gate-differences (recorded)")
        return dict(g, ok=True, strict_ok=False,
                    accepted=acceptance_record(g, "build --accept-gate-differences (the user's decision: "
                                                  "extraction values only)"))
    if args.allow_failed_gate:
        log(f"WARNING [{site}] the gate FAILED; embedded as failed by --allow-failed-gate: v19 refuses the store "
            "unless --soft-gates")
        return dict(g, ok=False, embedded_failed="--allow-failed-gate")
    raise Refused(f"{rel(p)}: the {g.get('year')} gate failed ("
                  + "; ".join(gate_lines(g)[1:]).replace("  ", " ")[:900]
                  + "); --accept-gate-differences (extraction values only) or --allow-failed-gate")


# ─────────────────────────────────────────────────────────────────────────────
#  boundary levels before 2023: the IOC gauges of the store years
# ─────────────────────────────────────────────────────────────────────────────

def gauge_chunk(cache_dir, code, year):
    """The yearly chunk ``fetch_ioc`` writes for ``code`` and ``year``."""
    return os.path.join(cache_dir, f"ioc_{code}_{int(year)}-01-01_{int(year)}-12-31.json")


def fetch_gauge_year(code, year, cache_dir):
    """One calendar year of an IOC gauge into ``cache_dir`` with the
    production downloader (``pyintertidal.boundary.fetch_ioc``; free, no
    credits). It swallows request errors and writes ``[]`` after an outage:
    :func:`gauge_year_coverage` decides whether the year counts."""
    from pyintertidal import net
    from pyintertidal.boundary import fetch_ioc

    net.use_system_certificates()
    fetch_ioc(code, f"{int(year)}-01-01", f"{int(year)}-12-31", cache_dir=cache_dir)


def gauge_year_coverage(path, year):
    """Samples and days with data per month of a yearly IOC chunk (records
    with a time in ``year`` and a numeric level); ``ok`` when every month has
    data on at least GAUGE_MIN_DAYS_PER_MONTH days."""
    import pandas as pd

    raw = read_json(path)
    if not isinstance(raw, list) or not raw:
        days = {}
        n = 0
    else:
        df = pd.DataFrame(raw)
        if "stime" not in df or "slevel" not in df:
            raise Refused(f"{rel(path)}: not an IOC chunk (no stime / slevel)")
        t = pd.to_datetime(df["stime"], errors="coerce")
        v = pd.to_numeric(df["slevel"], errors="coerce")
        ok = t.notna() & v.notna() & (t.dt.year == int(year))
        t = t[ok]
        n = int(ok.sum())
        days = {f"{m:02d}": int(k) for m, k in t.dt.normalize().drop_duplicates().dt.month.value_counts().items()}
    thin = [f"{m:02d}" for m in range(1, 13) if days.get(f"{m:02d}", 0) < GAUGE_MIN_DAYS_PER_MONTH]
    return {"samples": n, "days_per_month": dict(sorted(days.items())), "thin_months": thin, "ok": not thin}


def _same_or_inside(a, b):
    a, b = os.path.realpath(a), os.path.realpath(b)
    return a == b or a.startswith(b.rstrip(os.sep) + os.sep)


def ensure_gauge_years(site, years, cache_dir, prod_cache, fetch=True, dry_run=False, accept_gaps=False):
    """The yearly IOC chunks of the site's production gauges for ``years`` in
    the SEPARATE pre-2023 cache: present, else fetched (``fetch``; never in a
    dry run), each assessed (:func:`gauge_year_coverage`; a just-fetched thin
    year, e.g. ``[]`` after an outage, is moved aside as
    ``*.incomplete-<UTC>``, never deleted, and fetched once more). Refused:
    the cache is production's (or inside it); a year missing (``--no-fetch``)
    or failing to download; a thin year unless ``accept_gaps`` (recorded).
    Returns the record."""
    if _same_or_inside(cache_dir, prod_cache) or _same_or_inside(prod_cache, cache_dir):
        raise Refused(f"the pre-2023 gauge cache {rel(cache_dir)} is (or holds, or lies in) production's "
                      f"{rel(prod_cache)}: load_cached_ioc reads every chunk there and make_boundary demeans "
                      "each gauge by their median, so pre-2023 chunks there would change the published 2023-2025 "
                      "h_B; give another --gauge-cache")
    codes = list(PRODUCTION_BOUNDARY[site]["gauges"])
    rec = {"cache": rel(cache_dir), "production_cache": rel(prod_cache), "gauges": codes,
           "rule": (f"a yearly chunk counts when every month holds data on at least {GAUGE_MIN_DAYS_PER_MONTH} "
                    "days; fetched with pyintertidal.boundary.fetch_ioc (free)"),
           "years": {}, "missing": [], "thin": [], "fetched": [], "moved_aside": []}
    if not dry_run:
        os.makedirs(cache_dir, exist_ok=True)
    for code in codes:
        for y in years:
            p = gauge_chunk(cache_dir, code, y)
            fetched = False
            cov = None
            for attempt in range(2):
                if not os.path.isfile(p):
                    if dry_run or not fetch:
                        break
                    log(f"[{site}] fetching the IOC gauge {code} {y} (free) into {rel(cache_dir)}")
                    try:
                        fetch_gauge_year(code, y, cache_dir)
                    except Exception as e:  # noqa: BLE001 - any failure stops with the message below
                        raise Refused(f"fetching the IOC gauge {code} {y} failed ({type(e).__name__}: {e}): the "
                                      f"pre-2023 levels of {site} need it (EOT20 alone is never used silently); "
                                      "run where the IOC service is reachable") from None
                    fetched = True
                    rec["fetched"].append(f"{code} {y}")
                if not os.path.isfile(p):
                    break
                cov = gauge_year_coverage(p, y)
                if cov["ok"] or not fetched or attempt == 1:
                    break
                aside = f"{p}.incomplete-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}"
                os.replace(p, aside)
                rec["moved_aside"].append(rel(aside))
                log(f"[{site}] {rel(p)}: thin ({cov['thin_months']}), moved aside to {rel(aside)} (kept); "
                    "fetching once more")
                cov = None
            key = f"{code} {y}"
            if cov is None:
                rec["missing"].append(key)
                rec["years"][key] = {"path": rel(p), "present": False}
                continue
            rec["years"][key] = dict(cov, path=rel(p), present=True, sha256=sha256_file(p))
            if not cov["ok"]:
                rec["thin"].append(key)
    rec["ok"] = not rec["missing"] and not rec["thin"]
    if dry_run:
        return rec
    if rec["missing"]:
        raise Refused(f"the IOC gauge years {rec['missing']} of {site} are not in {rel(cache_dir)}"
                      + (" (--no-fetch)" if not fetch else "") + ": the pre-2023 levels need them (EOT20 alone is "
                      "never used silently); build without --no-fetch where the IOC service is reachable")
    if rec["thin"]:
        if not accept_gaps:
            raise Refused(f"the IOC gauge years {rec['thin']} of {site} hold months with data on fewer than "
                          f"{GAUGE_MIN_DAYS_PER_MONTH} days ("
                          + "; ".join(f"{k}: {rec['years'][k]['thin_months']}" for k in rec["thin"])
                          + "): a real gauge gap leaves those scenes' levels on fewer members; "
                            "--accept-gauge-gaps takes it (recorded)")
        rec["accepted_gaps"] = {"by": "--accept-gauge-gaps (the user's decision)", "utc": utcnow()}
        log(f"WARNING [{site}] thin IOC gauge years {rec['thin']} taken by --accept-gauge-gaps (recorded)")
    return rec


class _cwd:
    """Run a block from ``path`` (make_boundary reads data_v4/... relative)."""

    def __init__(self, path):
        self.path, self.old = path, None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc):
        os.chdir(self.old)


def production_boundary(site, prod_cache):
    """Production's boundary object of a Dutch site, built as
    experiments/v7_waterline_contour.py builds it (make_boundary(aoi, 'EOT20',
    n_gauges, period 2023-2025, exclude the judge), the gauges from
    ``prod_cache``)."""
    import pyintertidal as pit
    from pyintertidal import net
    from pyintertidal.boundary import make_boundary

    cfg = PRODUCTION_BOUNDARY[site]
    net.use_system_certificates()
    with _cwd(ROOT):
        return make_boundary(pit.sites.get(site), tide_model=cfg["tide_model"], n_gauges=cfg["n_gauges"],
                             period=TIME_EXTENT, cache_dir=prod_cache, exclude=list(cfg["exclude"]),
                             verbose=False)


def pre2023_boundary(site, prod, prod_cache, pre_cache):
    """Production's boundary with its gauges extended before 2023:
    ``EnsembleBoundary([production's tide model member, InterpolatedGauges(
    members, production's weights)])`` where each member is the pre-2023 IOC
    record (``load_cached_ioc`` of ``pre_cache`` + ``despike``) demeaned by
    production's constant: the median of its despiked 2023-2025 record,
    reproduced here bit for bit from ``prod_cache`` (refused otherwise).
    Returns (boundary, record)."""
    from pyintertidal.boundary import EnsembleBoundary, GaugeBoundary, InterpolatedGauges
    from pyintertidal.gauges import despike, load_cached_ioc

    cfg = PRODUCTION_BOUNDARY[site]
    provs = list(getattr(prod, "providers", None) or [])
    if len(provs) != 2 or type(provs[1]).__name__ != "InterpolatedGauges":
        raise Refused(f"the production boundary of {site} ({getattr(prod, 'name', prod)}) is not the consensus of "
                      "a tide model and InterpolatedGauges")
    tide, ig = provs
    names = [getattr(g, "name", None) for g in ig.gauges]
    if names != list(cfg["gauges"]):
        raise Refused(f"the production boundary of {site} uses the gauges {names}, not {list(cfg['gauges'])}")
    members, info = [], {}
    with _cwd(ROOT):
        for g in ig.gauges:
            df = load_cached_ioc(g.name, cache_dir=prod_cache)
            if df is None:
                raise Refused(f"no production chunk of {g.name} in {rel(prod_cache)}")
            df = despike(df)
            lev = df["level_m"].to_numpy(float)
            med = float(np.nanmedian(lev))
            again = GaugeBoundary(df["time"], lev - med, name=g.name)
            if not (np.array_equal(again.values, g.values) and again.times.equals(g.times)):
                raise Refused(f"the production gauge member {g.name} is not reproduced from {rel(prod_cache)} "
                              "(load_cached_ioc + despike - median): its demeaning constant is unknown")
            pre = load_cached_ioc(g.name, cache_dir=pre_cache)
            if pre is None or len(pre) < 1000:
                raise Refused(f"the pre-2023 record of {g.name} in {rel(pre_cache)} holds "
                              f"{0 if pre is None else len(pre)} samples (make_boundary skips a gauge below 1000)")
            n_raw = len(pre)
            pre = despike(pre)
            lp = pre["level_m"].to_numpy(float)
            members.append(GaugeBoundary(pre["time"], lp - med, name=g.name))
            info[g.name] = {"production_median_m": med, "pre2023_median_m": float(np.nanmedian(lp)),
                            "pre2023_minus_production_median_m": float(np.nanmedian(lp)) - med,
                            "samples": int(len(pre)), "spikes_dropped": int(n_raw - len(pre)),
                            "first": str(pre["time"].min()), "last": str(pre["time"].max()),
                            "production_member_reproduced": True}
    ig_pre = InterpolatedGauges(members, np.ones(len(members)), name=getattr(ig, "name", "gauges"))
    ig_pre.weights = np.array(ig.weights, dtype=float).copy()       # production's inverse-distance weights
    name = f"pre-2023: {getattr(prod, 'name', 'production boundary')} (gauges from {rel(pre_cache)})"
    return EnsembleBoundary([tide, ig_pre], name=name), {
        "boundary": name, "production_boundary": getattr(prod, "name", None),
        "tide_member": getattr(tide, "name", type(tide).__name__),
        "weights": [float(w) for w in ig_pre.weights], "gauges": info,
        "datum": ("each gauge demeaned by production's constant (the median of its despiked 2023-2025 record), so "
                  "h_B keeps production's datum across 2023; the pre-2023 median is recorded, not used")}


def scene_levels(boundary, over, dates):
    """{date: h_B and each member at the overpass instant} for the ``dates``
    with a time in ``over``; ``members``: 'all', 'partial' (a gauge without
    data within its max gap), 'EOT20 alone' (no gauge) or 'none'."""
    import pandas as pd

    have = [d for d in dates if d in over]
    if not have:
        return {}
    t = pd.DatetimeIndex(pd.to_datetime([over[d] for d in have]))
    if t.tz is not None:
        t = t.tz_localize(None)
    tide, ig = boundary.providers
    h = np.asarray(boundary.levels(t), float)
    e = np.asarray(tide.levels(t), float)
    gl = {g.name: np.asarray(g.levels(t), float) for g in ig.gauges}
    tname = getattr(tide, "name", "tide model")
    out = {}
    for i, d in enumerate(have):
        fin = [bool(np.isfinite(v[i])) for v in gl.values()]
        kind = ("none" if not np.isfinite(h[i]) else "all" if all(fin) and np.isfinite(e[i]) else
                "EOT20 alone" if not any(fin) else "partial")
        out[d] = {"t": over[d], "h_B": float(h[i]) if np.isfinite(h[i]) else None,
                  tname: float(e[i]) if np.isfinite(e[i]) else None,
                  **{k: float(v[i]) if np.isfinite(v[i]) else None for k, v in gl.items()},
                  "gauges_with_data": int(sum(fin)), "members": kind}
    return out


V19_LEVELS_NOTE = ("v19 (experiments/v19_fixed_config_tables.py, 2026-10-26) computes h_B itself from the "
                   "production boundary object, whose gauges are cached for 2023-2025 only: with --pre2023-levels "
                   "production these scenes get EOT20 ALONE, and by default v19 refuses the store. levels_pre2023 "
                   "holds the production consensus with the gauges of those years; v19 does not read it (its "
                   "owner's change)")


def n_target_summary(rec_count, screened_per_year, years):
    """The stop rule of the purchases: the screened scenes before 2023 (usable
    with an overpass time) against N_TARGET - the production record."""
    need = max(N_TARGET - int(rec_count), 0)
    have = int(sum(screened_per_year.values()))
    short = max(need - have, 0)
    nxt = int(min(years)) - 1 if years else None
    return {"n_target": N_TARGET, "production_record": int(rec_count), "needed_before_2023": need,
            "screened_before_2023": have, "reaches": short == 0, "short_by": short,
            "advice": ("enough for n = %d" % N_TARGET if short == 0 else
                       f"short by {short} for n = {N_TARGET}: the next older year is {nxt} (buy it only while short; "
                       "state in the tables when a site does not reach n)")}


# ─────────────────────────────────────────────────────────────────────────────
#  the build
# ─────────────────────────────────────────────────────────────────────────────

def extra_paths(extra_dir, site, y0, y1):
    stem = os.path.join(extra_dir, f"{site}_extract_{y0}-{y1}")
    return stem + ".npz", stem + ".json"


def existing_stores(extra_dir, site):
    return sorted(p for p in glob.glob(os.path.join(extra_dir, f"{site}_extract_*.npz"))
                  if not p.endswith(".tmp.npz"))


def supersede_plan(extra_dir, site, out_npz, new_dates, replace, allow_other):
    """The stores already in the extra folder: those to move aside (this
    script's, overlapping or of the same name, with ``replace``), refusals."""
    aside, others = [], []
    new = set(new_dates)
    for p in existing_stores(extra_dir, site):
        side = p[:-4] + ".json"
        builder = None
        if os.path.isfile(side):
            try:
                builder = read_json(side).get("builder")
            except ValueError:
                builder = None
        with np.load(p, allow_pickle=False) as z:
            d = set(np.asarray(z["dates"]).astype(str).tolist()) if "dates" in z.files else set()
        same = os.path.abspath(p) == os.path.abspath(out_npz)
        overlap = len(d & new)
        if same or overlap:
            if builder != BUILDER:
                raise Refused(f"{rel(p)} ({'same name' if same else f'{overlap} common dates'}) was written by "
                              f"{builder or 'another builder'}: not touched; v19 would stop on repeated dates. "
                              "Move it out of the extra folder first")
            if not replace:
                raise Refused(f"{rel(p)} ({'same name' if same else f'{overlap} common dates'}) is there: "
                              "--replace moves it aside (kept, never deleted)")
            aside.append(p)
        else:
            if not allow_other:
                if builder == BUILDER:
                    raise Refused(f"{rel(p)} is another store of this script for {site}: build ONE range holding "
                                  "every year (e.g. --years 2019-2022 --replace), or --allow-other-stores")
                raise Refused(f"{rel(p)} ({builder or 'unknown builder'}) is another extra store of {site}: v19 "
                              "uses every one of them (sources mixed); --allow-other-stores accepts it")
            others.append({"path": rel(p), "builder": builder, "dates": len(d)})
    return aside, others


def move_aside(path):
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest, k = f"{path}.superseded-{stamp}", 1
    while os.path.exists(dest):
        dest, k = f"{path}.superseded-{stamp}-{k}", k + 1
    os.replace(path, dest)
    log(f"{rel(path)} moved aside to {rel(dest)} (kept, never deleted)")
    return dest


def merged_overpass(site, roots_used, extra_files):
    """{date: ISO} from the stores' roots' overpass files then ``extra_files``;
    a date with two different times refuses."""
    from experiments import zarr_cubes as zc

    out, src, conflicts = {}, {}, []
    files = [str(zc.overpass_path(site, resolve(r))) for r in roots_used] + [resolve(f) for f in extra_files or []]
    used = []
    for f in files:
        if not os.path.isfile(f):
            continue
        d = load_overpass_file(f, required=False)
        used.append(file_info(f))
        for k, v in d.items():
            if k in out and out[k] != v:
                conflicts.append((k, out[k], v, rel(f)))
            out.setdefault(k, v)
            src.setdefault(k, rel(f))
    if conflicts:
        raise Refused(f"overpass files disagree on {len(conflicts)} date(s): {conflicts[:4]}")
    return out, used


def cmd_build(args):
    site = args.site
    years = list(args.years)
    I = site_inputs(site, args)
    R = load_reference(site, I)
    prod = load_production(I["extraction"])
    if R["frame"]["shape"] != prod["shape"]:
        raise Refused(f"the reference frame {R['frame']['shape']} is not the production frame {prod['shape']}")
    first_main = str(prod["dates"][0])
    if years[-1] >= int(first_main[:4]):
        raise Refused(f"years {years} reach the main extraction (first date {first_main}): the extra store holds "
                      "the years before it only (2023 is the gate's)")
    gate = gate_for_build(site, I, R, prod, args)
    roots = [r for r in (args.roots or LIGHT_ROOT).split(",") if r.strip()]
    stores = []
    try:
        for y in years:
            stores.append(checked_store(site, y, roots, R["frame"], args.allow_unconfirmed_registration))
        dates = [d for st in stores for d in st["dates"]]
        if any(b <= a for a, b in zip(dates, dates[1:])):
            raise Refused("the stores' dates are not strictly increasing across the years")
        if dates and dates[-1] >= first_main:
            raise Refused(f"store dates reach the main extraction's first date {first_main}")
        over, over_files = merged_overpass(site, sorted({st["info"]["root"] for st in stores}), args.overpass)
        T, P = len(dates), len(prod["keep"])
        if not T or not P:
            raise Refused(f"nothing to extract ({T} scenes x {P} keep px)")
        out_npz, out_json = extra_paths(I["extra_dir"], site, years[0], years[-1])
        aside, others = supersede_plan(I["extra_dir"], site, out_npz, dates, args.replace, args.allow_other_stores)
        no_time = [d for d in dates if d not in over]
        per_year = {str(st["info"]["year"]): st["info"]["dates"] for st in stores}
        g_kind = (gate.get("store") or {}).get("kind")
        kinds = sorted({st["info"]["kind"] for st in stores})
        lc = light_check(args.light_check)
        if "light" in kinds and lc["passed"] is False:
            if not args.allow_unconfirmed_registration:
                raise Refused(f"the downloader's light-path gate FAILED ({lc['path']}: {lc.get('why')}): the light "
                              "stores' layout is not proven (--allow-unconfirmed-registration records an override)")
            log(f"WARNING [{site}] the light-path gate FAILED ({lc['path']}); taken by "
                "--allow-unconfirmed-registration (recorded)")
        elif "light" in kinds and lc["passed"] is not True:
            log(f"WARNING [{site}] no light-path gate report at {lc['path']} (the downloader's light vs nine-band "
                "comparison of one year): recorded as absent")
        unproven = [st["info"]["path"] for st in stores if isinstance(st["info"].get("light_gate"), dict)
                    and st["info"]["light_gate"].get("without_light_gate")]
        if unproven and lc["passed"] is not True:
            why = (f"{unproven} were bought with the downloader's --without-light-gate and the light-path gate "
                   f"report {lc['path']} " + ("FAILED" if lc["passed"] is False else "is absent"))
            if not args.allow_unconfirmed_registration:
                raise Refused(f"{why}: their layout is not proven (--allow-unconfirmed-registration records an "
                              "override)")
            for st in stores:
                if st["info"]["path"] in unproven:
                    st["info"]["light_gate_override"] = f"--allow-unconfirmed-registration: {why}"
            log(f"WARNING [{site}] {why}; taken by --allow-unconfirmed-registration (recorded)")
        kind_note = None
        if kinds != [g_kind]:
            kind_note = (f"the gate ran on a {g_kind} store, these stores are {kinds}: the {g_kind} vs "
                         f"{'/'.join(kinds)} equality rests on the downloader's light-path gate "
                         f"({lc['path']}: passed {lc['passed']})")
            log(f"WARNING [{site}] {kind_note}")
        log(f"[{site}] build {years[0]}-{years[-1]}: {T} scenes {per_year} x {P:,} keep px from "
            + ", ".join(f"{st['info']['path']} ({st['info']['kind']}, {st['info']['registration']['method']})"
                        for st in stores)
            + f"; {len(no_time)} without an overpass time; gate {gate.get('year')} "
            + ("passed" if gate.get("strict_ok") else ("ACCEPTED (differences)" if gate.get("ok") else "FAILED"))
            + f"; -> {rel(out_npz)}" + (f" (moving aside {[rel(p) for p in aside]})" if aside else ""))
        # the boundary levels of those years: the production gauges' IOC data, fetched or refused
        gauge_cache, prod_cache = resolve(args.gauge_cache), resolve(args.production_gauge_cache)
        gauges = ensure_gauge_years(site, years, gauge_cache, prod_cache, fetch=not args.no_fetch,
                                    dry_run=args.dry_run, accept_gaps=args.accept_gauge_gaps)
        if args.dry_run:
            log(f"[{site}] dry run: IOC gauges {gauges['gauges']} in {gauges['cache']}: "
                + ("all years present" if not gauges["missing"] else f"to fetch {gauges['missing']}")
                + (f"; thin {gauges['thin']}" if gauges["thin"] else ""))
            log(f"[{site}] dry run: Y {T * P * 4 / 2 ** 30:.2f} GiB + C {T * P / 2 ** 30:.2f} GiB uncompressed "
                f"(C packed in memory: {T * ((P + 7) // 8) / 2 ** 20:.0f} MiB); nothing written")
            return 0
        bnd, binfo = pre2023_boundary(site, production_boundary(site, prod_cache), prod_cache, gauge_cache)
        with _cwd(ROOT):                     # the tide model reads tide_models/ relative
            levels = scene_levels(bnd, over, dates)
        rec_dates = load_record_dates(I["result"])
        log(f"[{site}] levels: {binfo['boundary']}; {len(levels)} scenes with a time: "
            + ", ".join(f"{k} {sum(1 for v in levels.values() if v['members'] == k)}"
                        for k in ("all", "partial", "EOT20 alone", "none")))
        ensure_ignored(I["extra_dir"])
        packed = np.zeros((T, (P + 7) // 8), np.uint8)
        screening, cloud = {}, {}
        crc_y, crc_c = [], []
        probe = sorted({0, T // 2, T - 1}) if T else []
        kept = {}
        counter = {"i": 0}

        def y_rows():
            for d, y, c, s in scene_stream(stores, prod["keep"], prod["shape"], R["transition"], R["static"],
                                           args.t_block):
                i = counter["i"]
                if d != dates[i]:
                    raise RuntimeError(f"scene order: {d} != {dates[i]}")
                packed[i] = np.packbits(c)
                crc_y.append(zlib.crc32(y.tobytes()) & 0xFFFFFFFF)
                crc_c.append(zlib.crc32(c.view(np.uint8).tobytes()) & 0xFFFFFFFF)
                cloud[d] = s["cloud_pct"]
                screening[d] = {"cloud_pct": s["cloud_pct"], "usable": s["usable"], "clean": s["clean"],
                                "g": s["g"], "n_bad_transition": s["n_bad_transition"],
                                "nodata_share": s["nodata_share"], "static_bad_masked": s["static_bad_masked"],
                                "clear_keep_px": int(c.sum()), "finite_keep_px": int(np.isfinite(y).sum()),
                                "keep_b03_median_dn": s["keep_b03_median_dn"],
                                "keep_valid_share": s["keep_valid_share"],
                                "keep_b08_low_share": s["keep_b08_low_share"]}
                if i in probe:
                    kept[i] = (y.copy(), c.copy())
                counter["i"] = i + 1
                yield y

        def c_rows():
            for i in range(T):
                yield np.unpackbits(packed[i], count=P).astype(bool)

        small = {"keep": prod["small"]["keep"], "dates": np.array(dates, dtype="<U10"),
                 "shape": prod["small"]["shape"]}
        for k in ("sea", "inter"):
            if k in prod["small"]:
                small[k] = prod["small"][k]
        tmp_npz = out_npz + ".tmp"
        t0 = time.time()
        try:
            write_npz(tmp_npz, small, {"Y": (np.float32, (T, P), y_rows()), "C": (np.bool_, (T, P), c_rows())},
                      compresslevel=None if args.no_compress else 1)
        except BaseException:
            if os.path.exists(tmp_npz):          # this run's own incomplete file (never a store or a product)
                os.remove(tmp_npz)
            raise
        if counter["i"] != T:
            raise RuntimeError(f"{counter['i']} rows computed, {T} expected")
        log(f"[{site}] npz written [{time.time() - t0:.0f} s]; reading back rows {probe}")
        back = readback(tmp_npz, probe, kept, small)
    finally:
        for st in stores:
            st["ds"].close()
    usable = [d for d in dates if screening[d]["usable"]]
    # the radiometry gate, per year and input-baseline group
    bl = {}
    for st in stores:
        bl.update(((st["info"].get("input_baselines") or {}).get("per_date")) or {})
    probe_rad = radiometry_probe(screening, years, (gate.get("screening") or {}).get("per_date"), bl)
    refusals = []
    for k in probe_rad["suspect"]:
        log(f"WARNING [{site}] radiometry: {k}: {'; '.join(probe_rad['groups'][k]['suspect'])} (an L2A +1000 "
            "offset in the values?)")
    if probe_rad["suspect"]:
        if args.accept_radiometry:
            probe_rad["accepted"] = {"by": "--accept-radiometry (the user's decision)", "utc": utcnow()}
        else:
            refusals.append(f"the radiometry of {probe_rad['suspect']} is not the gate year's ("
                            + "; ".join(f"{k}: {'; '.join(probe_rad['groups'][k]['suspect'])}"
                                        for k in probe_rad["suspect"])
                            + "): --accept-radiometry takes it (recorded)")
    # the levels of the usable scenes with a time: never EOT20 alone silently
    screened = [d for d in usable if d in over]
    alone = [d for d in screened if levels.get(d, {}).get("members") in ("EOT20 alone", "none")]
    lev_rec = {"rule": ("h_B of production's boundary object with its gauge members extended before 2023 by the "
                        "IOC records of these years (pre2023_boundary), at the overpass label instants"),
               **binfo, "fetch": gauges, "per_date": levels,
               "screened_members": {k: sum(1 for d in screened if levels.get(d, {}).get("members") == k)
                                    for k in ("all", "partial", "EOT20 alone", "none")},
               "screened_eot20_alone": alone, "v19_note": V19_LEVELS_NOTE}
    if alone:
        if args.accept_gauge_gaps:
            lev_rec["accepted_eot20_alone"] = {"by": "--accept-gauge-gaps (the user's decision)", "utc": utcnow()}
            log(f"WARNING [{site}] {len(alone)} screened scene(s) with EOT20 alone (no gauge within an hour): "
                f"{alone[:6]}; taken by --accept-gauge-gaps (recorded)")
        else:
            refusals.append(f"{len(alone)} usable scene(s) with an overpass time would have h_B from EOT20 alone (no "
                            f"gauge with data within an hour): {alone[:8]}; --accept-gauge-gaps takes them (recorded)")
    if refusals:
        os.remove(tmp_npz)                   # this run's own unfinished output, never renamed into place
        raise Refused("; ".join(refusals))
    per_year_screened = {str(y): sum(1 for d in screened if d[:4] == str(y)) for y in years}
    target = n_target_summary(len(rec_dates), per_year_screened, years)
    side = {
        "builder": BUILDER, "site": site, "years": years, "utc": utcnow(),
        "overpass": {d: over[d] for d in dates if d in over},
        "cloud_pct": cloud,
        "gate_2023": gate,
        "reference_map": R["summary"],
        "contract": ("experiments/v19_fixed_config_tables.py extra extraction store: npz Y, C, keep, dates, shape "
                     "(+ sea, inter) as marea_extract.npz; sidecar overpass, cloud_pct, gate_2023"),
        "n_dates": T, "n_keep": P, "dates_per_year": per_year,
        "usable_per_year": {str(y): sum(1 for d in usable if d[:4] == str(y)) for y in years},
        "usable_with_overpass_per_year": {str(y): sum(1 for d in usable if d[:4] == str(y) and d in over)
                                          for y in years},
        "dates_without_overpass": no_time,
        "extraction": {"rule": ("Y = (B03 - B08) / (B03 + B08) in float32 on the int16 digital numbers as stored "
                                "(fill -32768 -> NaN; NaN where B03 + B08 = 0), C = SCL in {4, 5, 6, 7}, at the "
                                "production keep pixels (pyintertidal.marea.extract)"),
                       "copied_from_production": sorted(k for k in small if k != "dates"),
                       "not_copied": [k for k in prod["files"] if k not in small and k not in ("Y", "C")],
                       "dtypes": {"Y": "float32", "C": "bool", "dates": "<U10"},
                       "readback": back, "row_crc32": {"Y": crc_y, "C": crc_c}},
        "screening": {"rule": ("production transition-zone screening (pyintertidal.stability.reference_and_clouds "
                               "pass 2 with the notebooks' parameters, then filter_dates 0.10) on the production "
                               "reference map, the legacy cube's static no-data set to no data first"),
                      "per_date": screening, "static_mask_px": int(R["static"].sum())},
        "stores": {str(st["info"]["year"]): st["info"] for st in stores},
        "radiometry_probe": probe_rad,
        "levels_pre2023": lev_rec,
        "n_target": target,
        "light_path_check": lc,
        "gate_store_kind_note": kind_note,
        "other_stores_accepted": others,
        "inputs": {"production_extraction": prod["fingerprint"], "keep_sha256": prod["keep_sha256"],
                   "overpass_files": over_files, "reference": R["summary"],
                   "gate": {"path": gate.get("path"), "sha256": gate.get("sha256")}},
        "code": {"version": CODE_VERSION, "sha256": sha256_file(__file__)},
    }
    tmp_json = out_json + ".tmp.json"
    write_json(tmp_json, side)
    for p in aside:
        move_aside(p)
        if os.path.isfile(p[:-4] + ".json"):
            move_aside(p[:-4] + ".json")
    os.replace(tmp_json, out_json)
    os.replace(tmp_npz, out_npz)
    log(f"[{site}] -> {rel(out_npz)} ({os.path.getsize(out_npz) / 2 ** 20:.0f} MiB) and {rel(out_json)}: {T} scenes, "
        f"{len(usable)} usable ({side['usable_with_overpass_per_year']} with an overpass time)")
    log(f"[{site}] n = {N_TARGET}: {target['screened_before_2023']} screened before 2023 + {target['production_record']} "
        f"in production; {target['advice']}")
    log(f"[{site}] levels: {V19_LEVELS_NOTE}")
    return 0


def readback(path, probe, kept, small):
    """The written npz read back: small members equal, layouts, and the
    probe rows of Y and C equal to what was written."""
    with np.load(path, allow_pickle=False) as z:
        for k, a in small.items():
            got = z[k]
            if got.dtype != np.asarray(a).dtype or not np.array_equal(got, a):
                raise RuntimeError(f"{path}:{k} does not read back as written")
    for k, dt in (("Y", np.float32), ("C", np.bool_)):
        shp, fortran, d = member_layout(path, k)
        if fortran or d != np.dtype(dt) or len(shp) != 2:
            raise RuntimeError(f"{path}:{k}: layout {d} {shp} fortran={fortran}")
    for (ty, y), (tc, c) in zip(iter_rows(path, "Y", probe), iter_rows(path, "C", probe)):
        ey, ec = kept[ty]
        if not (np.array_equal(y.view(np.uint32), ey.view(np.uint32)) and np.array_equal(c, ec)):
            raise RuntimeError(f"{path}: row {ty} does not read back as written")
    return {"rows": list(probe), "small_members": sorted(small), "ok": True}


def light_check(path):
    """The downloader's light-path gate report (``<year>.vs_nine_band.json``:
    ``passed``, ``why``), recorded; ``passed`` None when absent or unreadable."""
    p = resolve(path or LIGHT_CHECK)
    if not os.path.isfile(p):
        return {"path": rel(p), "present": False, "passed": None}
    try:
        r = read_json(p)
    except ValueError:
        return {"path": rel(p), "present": True, "readable": False, "passed": None}
    cmp_ = r.get("comparison") if isinstance(r.get("comparison"), dict) else {}
    return {"path": rel(p), "present": True, "passed": r.get("passed"), "why": str(r.get("why") or "")[:600],
            "verdict": cmp_.get("verdict"), "store": r.get("store"), "nine_band_store": r.get("nine_band_store"),
            "checked_utc": r.get("checked_utc"), "sha256": sha256_file(p)}


# ─────────────────────────────────────────────────────────────────────────────
#  check: the stores as v19 reads them
# ─────────────────────────────────────────────────────────────────────────────

def cmd_check(args):
    """Every extra store of the site through v19's own ``load_extra_store``
    (with the site, so its production-map check runs as in ``v19 run``), and
    what the levels and the n target say."""
    import inspect

    from experiments import v19_fixed_config_tables as v19

    site = args.site
    I = site_inputs(site, args)
    prod = load_production(I["extraction"])
    paths = v19.find_extra_stores(site, I["extra_dir"])
    if not paths:
        log(f"[{site}] no extra store in {rel(I['extra_dir'])}")
        return 2
    kw = {"site": site} if "site" in inspect.signature(v19.load_extra_store).parameters else {}
    bad = 0
    for p in paths:
        st, info = v19.load_extra_store(p, prod["keep"], prod["shape"], prod["dates"], **kw)
        g = (info.get("gate_2023") or {}) if isinstance(info.get("gate_2023"), dict) else {}
        ok = st is not None and g.get("ok") is True
        bad += not ok
        log(f"[{site}] {rel(p)}: " + (f"readable by v19 ({info.get('dates_in_store')} dates, "
                                      f"{info.get('dates_before_main')} before the main extraction, "
                                      f"{len(st['usable'])} usable, {len(st['overpass'])} with a time; "
                                      f"{info.get('screening_rule')}; reference map: "
                                      f"{info.get('reference_map_check')}); gate_2023 ok {g.get('ok')}"
                                      if st is not None else f"REFUSED by v19: {info.get('reason')}"))
        side = p[:-4] + ".json"
        S = read_json(side) if os.path.isfile(side) else {}
        lev, tgt = S.get("levels_pre2023") or {}, S.get("n_target") or {}
        if lev:
            log(f"[{site}] {rel(p)}: levels of the screened scenes {lev.get('screened_members')}; NOTE: "
                f"{V19_LEVELS_NOTE}")
        else:
            log(f"[{site}] {rel(p)}: NO levels_pre2023 in the sidecar: v19 would give these scenes EOT20 alone "
                "(--pre2023-levels production)")
            bad += 1
        if tgt:
            log(f"[{site}] {rel(p)}: n = {tgt.get('n_target')}: {tgt.get('advice')}")
    return 0 if not bad else 2


# ─────────────────────────────────────────────────────────────────────────────
#  command line
# ─────────────────────────────────────────────────────────────────────────────

def make_parser():
    p = argparse.ArgumentParser(prog="python -m experiments.light_to_extraction",
                                description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("site", choices=TARGET_SITES)
        sp.add_argument("--work", default=None, help=f"reference / gate folder (default {WORK_DIR})")
        sp.add_argument("--extra-dir", default=None, help=f"v19's extra store folder (default {EXTRA_DIR})")
        sp.add_argument("--extraction", default=None, help="production extraction (default products_<site>_10m/"
                                                           "marea_extract.npz)")
        sp.add_argument("--result-json", default=None, help="production result.json (record_dates)")
        sp.add_argument("--prod-overpass", default=None, help="production overpass_times.json")
        sp.add_argument("--cube", default=None, help="legacy 10 m netCDF cube (default ndwi_cube_<site>_2023-2025_10m.nc)")
        sp.add_argument("--t-block", type=int, default=T_BLOCK, help=f"dates per read block (default {T_BLOCK})")
        sp.add_argument("--gauge-cache", default=PRE2023_GAUGE_CACHE,
                        help=f"build: the pre-2023 IOC chunks (default {PRE2023_GAUGE_CACHE}; never production's)")
        sp.add_argument("--production-gauge-cache", default=PRODUCTION_GAUGE_CACHE,
                        help=f"build: production's IOC chunks, only read (default {PRODUCTION_GAUGE_CACHE})")

    r = sub.add_parser("reference", help="the production reference map, the static no-data mask and the grid")
    common(r)
    r.add_argument("--from-map", default=None, help="import the production map (.tif / .npy) instead of "
                                                     "recomputing it")
    r.add_argument("--allow-reference-mismatch", action="store_true",
                   help="class / scene counts may differ from the notebooks' print (recorded)")
    g = sub.add_parser("gate", help="the 2023 gate: a store against the production extraction and record")
    common(g)
    g.add_argument("--roots", default=None, help=f"store roots searched in order (default {','.join(GATE_ROOTS)})")
    g.add_argument("--year", type=int, default=GATE_YEAR)
    g.add_argument("--out", default=None, help="gate report path (default <work>/<site>_gate_<year>.json)")
    g.add_argument("--allow-unconfirmed-registration", action="store_true")
    g.add_argument("--accept-gate-differences", action="store_true",
                   help="the user's decision: record in the report the acceptance of a gate that fails on "
                        "extraction values only (never dates, screening or the static mask); build and the "
                        "downloader's purchase of this site's pre-2023 years then honour it")
    b = sub.add_parser("build", help="the extra extraction store of the years given")
    common(b)
    b.add_argument("--years", required=True, type=parse_years,
                   help="YYYY-YYYY (contiguous, before the main extraction)")
    b.add_argument("--roots", default=None, help=f"store roots searched in order (default {LIGHT_ROOT})")
    b.add_argument("--gate", default=None, help="gate report (default <work>/<site>_gate_2023.json)")
    b.add_argument("--overpass", action="append", default=[], help="extra overpass file(s) {date: ISO}")
    b.add_argument("--replace", action="store_true", help="move this script's overlapping store aside")
    b.add_argument("--allow-other-stores", action="store_true",
                   help="accept a non-overlapping extra store of another builder in the folder")
    b.add_argument("--allow-unconfirmed-registration", action="store_true")
    b.add_argument("--allow-failed-gate", action="store_true",
                   help="embed a failed gate (v19 then refuses the store unless --soft-gates)")
    b.add_argument("--accept-gate-differences", action="store_true",
                   help="accept a gate whose only differences are extraction values (recorded)")
    b.add_argument("--light-check", default=None, help=f"light-vs-9-band report recorded (default {LIGHT_CHECK})")
    b.add_argument("--no-fetch", action="store_true",
                   help="never fetch the IOC gauge years (no network): a missing year refuses")
    b.add_argument("--accept-gauge-gaps", action="store_true",
                   help="take thin IOC gauge months and usable scenes whose h_B would rest on EOT20 alone "
                        "(recorded; the user's decision)")
    b.add_argument("--accept-radiometry", action="store_true",
                   help="take a year or baseline group the radiometry gate finds suspect (recorded)")
    b.add_argument("--no-compress", action="store_true", help="store the members uncompressed")
    b.add_argument("--dry-run", action="store_true", help="check the inputs and print the plan; write nothing")
    c = sub.add_parser("check", help="the site's extra stores through v19's load_extra_store")
    common(c)
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    fn = {"reference": cmd_reference, "gate": cmd_gate, "build": cmd_build, "check": cmd_check}[args.cmd]
    try:
        return fn(args)
    except Refused as e:
        log(f"REFUSED: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
