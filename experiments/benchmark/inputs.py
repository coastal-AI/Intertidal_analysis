# -*- coding: utf-8 -*-
"""Validation v2 — the scene inputs every method reads (spec §1.2, §1.10, §1.11).

Question
--------
What exactly are the scenes of a site: which dates, at which instants, on
which grid, with which radiometric offset, and how are they read so that a
Zarr chunk is decoded once? The harness never reads the stores directly: it
reads them through a :class:`SceneSource` (the Zarr stores of
``experiments.zarr_cubes``, or the legacy netCDF cubes for the reproduction
gates only).

Pre-registered rules (docs/validation_v2_spec.md §1.2-1.4, §1.10, §1.11;
user decisions of 2026-10-13 and 2026-10-14)
--------------------------------------------------------------------------
* Record: the store dates in ``[start, end)`` (end exclusive) that have a
  valid instant, in time order; positions are record positions of the
  source (open it over the union period 2017-01-01..2026-01-01 for global
  positions; a window is a selection, :meth:`SceneSource.window`). An
  instant must fall on its label's UTC date. The finite-h_B condition is
  the package's (``keep_dates``).
* Instants (user decision of 2026-10-14): the DEFAULT scene instant is the
  catalogue / product-name time, i.e. the datatake start of
  ``overpass_times.json`` (``instants="catalogue"``; "labels" is an alias),
  exactly as in every production result; using it raises no warning and
  sets no flag. A ``sensing_times.json`` (the optional solar-angle
  diagnostic of ``sensing_time.py``) is read ONLY when its path is given
  explicitly (``instants=<path>``; CLI ``--sensing-times <file>``); no
  file beside the stores is ever picked up implicitly.
* Raw values: int16 spectral bands exactly as stored (no CF decoding);
  SCL and CLD as uint8 after a range check (``obs.scl_u8``, ``obs.cld_u8``).
* Reading: chunk-aligned blocks of 16 dates x 256 rows (the stores' chunks),
  in chunk order, through an LRU cache of decoded chunk blocks
  (:class:`BlockCache`), so a chunk is decoded once when the cache holds
  the bands of one 16-date block (the default 1 GB does for row blocks; a
  smaller cache only costs time). With ``pair20=True`` the row blocks start
  on the 20 m lattice rows, so the 2 x 2 blocks of the 20 m rule never
  straddle two row blocks.
* ``lattice20``: the phase (py, px) of the native 20 m lattice on the 10 m
  grid, measured from the 2 x 2 replication of B8A, B11, B12 and SCL on up
  to 8 scenes with data (:func:`measure_lattice20`): a phase is exact when
  every pair of rows (columns) it pairs is equal in every band and scene;
  None when no phase is exact (box means at (0, 0), declared). Ambiguity
  (both phases exact, uniform data) is resolved by the grid geometry (20 m
  lattice lines at multiples of 20 m), else None.
* Per-scene BOA offset o_s in {0, 1000} (:func:`decide_offsets`, the frozen
  rule written before lock-A, :data:`OFFSET_RULE`), never by date: (1)
  negative values in the delivered bands -> 0; (2) all processing
  baselines < 04.00 -> 0 (unclear if the water says otherwise); (3) at least
  1,000 SCL-water pixels with valid B12: a share of B12 DN < 500 >= 0.5 ->
  0, <= 0.02 -> 1000, else unclear; (4) the frame share of B12 DN in
  [1, 500) > 0.01 (on >= 1 % of the frame valid) -> 0; (5) otherwise the
  regime of the year's strongly decided scenes if they agree, else the
  store's year ``offset_state``, else unclear. 'unclear' is an error for
  the observation (:class:`OffsetUnclear`) and fails the inputs check.
  The legacy netCDF cubes are offset-harmonised (value = DN - 1000 >=
  -1000): o_s = 0 for every scene.
* Fingerprint of a store-year: sha256 of the consolidated metadata plus the
  sorted (chunk key, size) list; ``full=True`` adds a content hash.
* Scene metadata sidecar ``cubes/<site>/scene_meta.json``
  (:data:`SCENE_META_FORMAT`): per date the STAC items (one per MGRS tile
  and product) with platform, orbits, processing version, cloud cover,
  mean sun and view angles; built by :func:`scene_meta_from_items` from the
  CDSE STAC features (the fetch step does the network query).
* The inputs report (``results/validation_v2/common/inputs_report.json``,
  :func:`inputs_report`): the §1.11 checks; it writes no package and is
  safe to repeat.

Memory: a row block of 16 dates x 256 rows x the frame width per band, and
the cache (``cache_mb``); nothing holds a whole cube.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import zarr_cubes as zc
from experiments.benchmark import obs, settings

#: the union period of the stores (end exclusive)
UNION_PERIOD = settings.UNION_PERIOD
#: dates per decoded time block (the stores' time chunk)
T_BLOCK = int(zc.CHUNKS["t"])
#: rows per row block (the stores' row chunk)
ROW_BLOCK = int(zc.CHUNKS["y"])
#: bands of the Zarr stores and of the legacy netCDF cubes
ZARR_BANDS = tuple(zc.BANDS)
LEGACY_BANDS = ("B03", "B08", "SCL")
#: bands decoded to uint8 after a range check
U8_BANDS = ("SCL", "CLD")
#: the bands lattice20 is measured on
LATTICE_BANDS = ("B8A", "B11", "B12", "SCL")
#: scenes sampled for lattice20
LATTICE_SCENES = 8
#: default cache budget (MB)
CACHE_MB = 1024
#: the cache of the inputs scan (every chunk is read once)
SCAN_CACHE_MB = 128
#: file names beside the stores
SCENE_META_NAME = "scene_meta.json"
#: the optional solar-angle diagnostic's file name (read only when its path
#: is given explicitly; user decision of 2026-10-14)
SENSING_NAME = "sensing_times.json"
#: the default scene instants: the catalogue / product-name time (the
#: datatake start), as every production result (user decision 2026-10-14)
CATALOGUE = "catalogue"
CATALOGUE_ALIASES = ("catalogue", "labels")
SCENE_META_FORMAT = "pyintertidal-scene-meta/1"
INPUTS_REPORT_FORMAT = "validation-v2-inputs-report/1"
INPUTS_REPORT = "results/validation_v2/common/inputs_report.json"
#: the key of an instant in sensing_times.json
INSTANT_KEY = "instant"

#: the frozen per-scene offset rule (§1.10, §2.3); written before lock-A
OFFSET_RULE = {
    "rule": ("(1) negative values -> 0; (2) all baselines < 04.00 -> 0 (unclear if the water "
             "B12 says +1000); (3) >= 1,000 SCL-water px with valid B12: share of B12 DN < 500 "
             ">= 0.5 -> 0, <= 0.02 -> 1000, else unclear; (4) frame share of B12 DN in [1, 500) "
             "> 0.01 on >= 1 % of the frame valid -> 0; (5) otherwise the regime of the year's "
             "strongly decided scenes if they agree, else the store's year offset_state, else "
             "unclear"),
    "low_dn": 500,
    "offset_baseline": "04.00",
    "water_scl": 6,
    "water_min_px": 1000,
    "water_low_none": 0.5,
    "water_low_offset": 0.02,
    "frame_min_valid_share": 0.01,
    "frame_low_none": 0.01,
}
OFFSET_STATES = ("+1000 in the values", "no offset", "unclear")


def log(msg):
    print(msg, flush=True)


class InputsError(RuntimeError):
    """An input violates the reader contract."""


class OffsetUnclear(InputsError):
    """A scene's BOA offset could not be decided (§1.10): an error."""


class GridMismatch(InputsError):
    """Two grids that must be identical differ."""


# ─────────────────────────────────────────────────────────────────────────────
#  the grid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class Grid:
    """Pixel centres (x ascending, y descending, metres) and the CRS."""

    xs: np.ndarray
    ys: np.ndarray
    crs_wkt: str

    def __post_init__(self):
        xs = np.array(self.xs, np.float64)
        ys = np.array(self.ys, np.float64)
        xs.flags.writeable = False
        ys.flags.writeable = False
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)

    @property
    def H(self):
        return int(len(self.ys))

    @property
    def W(self):
        return int(len(self.xs))

    @property
    def shape(self):
        return self.H, self.W

    @property
    def epsg(self):
        return zc._epsg_from_wkt(self.crs_wkt)

    def same_as(self, other):
        """Identical shape, coordinates (exact) and CRS."""
        return (self.shape == other.shape and np.array_equal(self.xs, other.xs)
                and np.array_equal(self.ys, other.ys) and zc.crs_equal(self.crs_wkt, other.crs_wkt))

    def describe(self):
        dx = float(self.xs[1] - self.xs[0]) if self.W > 1 else None
        dy = float(self.ys[1] - self.ys[0]) if self.H > 1 else None
        return {"H": self.H, "W": self.W, "x0": float(self.xs[0]), "y0": float(self.ys[0]),
                "dx": dx, "dy": dy, "epsg": self.epsg, "sha256": self.sha256()}

    def sha256(self):
        h = hashlib.sha256(self.xs.astype("<f8").tobytes())
        h.update(self.ys.astype("<f8").tobytes())
        h.update(str(self.epsg or self.crs_wkt).encode("utf-8"))
        return h.hexdigest()

    def lonlat_centre(self):
        """(lon, lat) of the frame centre."""
        from pyproj import CRS, Transformer

        tf = Transformer.from_crs(CRS.from_user_input(self.crs_wkt), "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(float(self.xs.mean()), float(self.ys.mean()))
        return float(lon), float(lat)


def grid_of(ds):
    """The :class:`Grid` of an xarray dataset with ``x``, ``y`` and a ``crs``
    (or ``spatial_ref``) variable."""
    return Grid(ds["x"].values, ds["y"].values, zc.grid_info(ds)["crs_wkt"] or "")


def grid_from_geotiff(path):
    """The :class:`Grid` of a north-up GeoTIFF (pixel centres)."""
    import rasterio

    with rasterio.open(path) as r:
        tr, H, W = r.transform, r.height, r.width
        if tr.b != 0 or tr.d != 0:
            raise InputsError(f"{path}: rotated transform")
        xs = tr.c + tr.a * (np.arange(W) + 0.5)
        ys = tr.f + tr.e * (np.arange(H) + 0.5)
        return Grid(xs, ys, r.crs.to_wkt() if r.crs else "")


def assert_same_grid(a, b, what):
    """Raise :class:`GridMismatch` unless ``a`` and ``b`` are identical."""
    if not a.same_as(b):
        raise GridMismatch(f"{what}: grids differ ({a.describe()} vs {b.describe()})")


def expected_lattice20(grid):
    """The 20 m lattice phase implied by the coordinates: Sentinel-2 20 m
    pixel edges lie on multiples of 20 m in the tile's UTM CRS, so the phase
    is 0 when the frame's first 10 m pixel starts a 20 m pixel, else 1."""
    dx = float(grid.xs[1] - grid.xs[0]) if grid.W > 1 else 10.0
    dy = float(grid.ys[0] - grid.ys[1]) if grid.H > 1 else 10.0
    left = float(grid.xs[0]) - dx / 2.0
    top = float(grid.ys[0]) + dy / 2.0
    px = 0 if abs(((left + 1e-6) % 20.0)) < 1e-3 else 1
    py = 0 if abs(((top + 1e-6) % 20.0)) < 1e-3 else 1
    return py, px


# ─────────────────────────────────────────────────────────────────────────────
#  lattice20
# ─────────────────────────────────────────────────────────────────────────────

def lattice_votes(a):
    """``{"y": {p: [equal, pairs]}, "x": {p: [equal, pairs]}}`` of one 2-D
    array: for phase p, the pairs of rows (columns) r, r + 1 with r = p mod 2
    and how many of their elements are equal."""
    a = np.asarray(a)
    out = {"y": {}, "x": {}}
    for ax, n in (("y", a.shape[0]), ("x", a.shape[1])):
        for p in (0, 1):
            r = np.arange(p, n - 1, 2)
            if ax == "y":
                eq = int(np.count_nonzero(a[r, :] == a[r + 1, :]))
                tot = int(r.size * a.shape[1])
            else:
                eq = int(np.count_nonzero(a[:, r] == a[:, r + 1]))
                tot = int(r.size * a.shape[0])
            out[ax][p] = [eq, tot]
    return out


def measure_lattice20(arrays, expected=None):
    """The native 20 m lattice phase from replicated 20 m bands.

    Parameters
    ----------
    arrays : iterable of (band, 2-D array)
        Raw values of the 20 m bands on the 10 m grid (any number of scenes;
        each row block must start on an even row).
    expected : (py, px), optional
        :func:`expected_lattice20`, used only to resolve ambiguity.

    Returns
    -------
    dict
        ``phase`` ((py, px) or None), ``exact`` (bool), ``votes`` per axis
        and phase, ``ambiguous`` axes, ``expected``, ``agrees_with_grid``.
    """
    tot = {"y": {0: [0, 0], 1: [0, 0]}, "x": {0: [0, 0], 1: [0, 0]}}
    bands = set()
    for band, a in arrays:
        bands.add(band)
        v = lattice_votes(a)
        for ax in ("y", "x"):
            for p in (0, 1):
                tot[ax][p][0] += v[ax][p][0]
                tot[ax][p][1] += v[ax][p][1]
    phase, amb = [], []
    for k, ax in enumerate(("y", "x")):
        exact = [p for p in (0, 1) if tot[ax][p][1] > 0 and tot[ax][p][0] == tot[ax][p][1]]
        if len(exact) == 1:
            phase.append(exact[0])
        elif len(exact) == 2:
            amb.append(ax)
            phase.append(expected[k] if expected is not None else None)
        else:
            phase.append(None)
    ph = tuple(phase) if None not in phase else None
    return {"phase": list(ph) if ph else None, "exact": ph is not None,
            "votes": {ax: {str(p): tot[ax][p] for p in (0, 1)} for ax in tot},
            "ambiguous": amb, "bands": sorted(bands),
            "expected": list(expected) if expected is not None else None,
            "agrees_with_grid": (None if expected is None or ph is None
                                 else list(ph) == list(expected))}


# ─────────────────────────────────────────────────────────────────────────────
#  the block cache
# ─────────────────────────────────────────────────────────────────────────────

class BlockCache:
    """LRU cache of decoded chunk blocks, bounded in bytes.

    Keys are ``(source part, band, time block, row chunk)``; values are raw
    arrays of one time block x one row chunk x the frame width. ``loads``
    counts decodes, ``hits`` reuses; a block larger than the budget is
    returned without being kept.
    """

    def __init__(self, max_mb=CACHE_MB):
        self.max_bytes = int(float(max_mb) * 1e6)
        self._d = collections.OrderedDict()
        self.nbytes = 0
        self.loads = 0
        self.hits = 0
        self.evictions = 0

    def get(self, key, load):
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        v = load()
        self.loads += 1
        if v.nbytes <= self.max_bytes:
            while self._d and self.nbytes + v.nbytes > self.max_bytes:
                _, old = self._d.popitem(last=False)
                self.nbytes -= old.nbytes
                self.evictions += 1
            self._d[key] = v
            self.nbytes += v.nbytes
        return v

    def clear(self):
        self._d.clear()
        self.nbytes = 0

    def stats(self):
        return {"loads": self.loads, "hits": self.hits, "evictions": self.evictions,
                "held_mb": round(self.nbytes / 1e6, 1), "max_mb": round(self.max_bytes / 1e6, 1)}


# ─────────────────────────────────────────────────────────────────────────────
#  instants
# ─────────────────────────────────────────────────────────────────────────────

def _naive_utc(v):
    if v is None:
        return pd.NaT
    ts = pd.Timestamp(v)
    if ts is pd.NaT:
        return ts
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def read_sensing_times(src):
    """``{date: Timestamp (naive UTC) or NaT}`` from ``sensing_times.json``.

    Accepts the G-time format ``{"dates": {date: {"instant": iso or null,
    ...}}}``, ``{date: {"instant": ...}}`` and the label format ``{date:
    iso}`` (``overpass_times.json``); ``src`` may be a path or a mapping.
    """
    if isinstance(src, (str, os.PathLike)):
        with open(src, encoding="utf-8") as f:
            src = json.load(f)
    d = src.get("dates", src) if isinstance(src, dict) else src
    out = {}
    for date, v in dict(d).items():
        if isinstance(v, dict):
            v = v.get(INSTANT_KEY)
        out[str(date)[:10]] = _naive_utc(v)
    return out


def resolve_instants(instants, labels):
    """The scene instants and where they come from (user decision of
    2026-10-14).

    Parameters
    ----------
    instants : None, "catalogue", "labels", a path or a mapping
        None, "catalogue" or its alias "labels": the catalogue times
        (``labels()``, the datatake start of the product names: the default,
        no warning, no flag). A path: a ``sensing_times.json`` given
        explicitly (the optional diagnostic). A mapping: ``{date: instant}``.
    labels : callable
        Returns the catalogue ``{date: time}`` (raises when absent).

    Returns
    -------
    (dict, str, str or None)
        ``{date: naive UTC Timestamp or NaT}``, the source ("catalogue",
        "sensing_times" or "given") and the file read (None for a mapping).
    """
    if instants is None or (isinstance(instants, str) and instants in CATALOGUE_ALIASES):
        lab, path = labels()
        return {str(d)[:10]: _naive_utc(t) for d, t in lab.items()}, CATALOGUE, path
    if isinstance(instants, str) and instants in ("sensing", "auto"):
        raise ValueError(f"instants={instants!r} is no longer a mode (user decision "
                         "2026-10-14): the scene instants are the catalogue times by default; "
                         "a sensing_times.json is used only when its PATH is given "
                         "(--sensing-times <file>)")
    if isinstance(instants, (str, os.PathLike)):
        if not os.path.isfile(instants):
            raise FileNotFoundError(f"{instants}: no such sensing_times.json")
        return read_sensing_times(instants), "sensing_times", str(instants).replace("\\", "/")
    return ({str(k)[:10]: _naive_utc(v) for k, v in dict(instants).items()}, "given", None)


def _check_instants(instants):
    """Instants must fall on their label's UTC date."""
    bad = [d for d, t in instants.items()
           if t is not pd.NaT and not pd.isna(t) and t.strftime("%Y-%m-%d") != d]
    if bad:
        raise InputsError(f"instants not on their label date: {bad[:5]}")


def instants_sha(instants):
    """sha256 of a resolved ``{date: instant or NaT}`` mapping (ISO instants,
    None for NaT, dates sorted): the identity of a record's instants."""
    canon = {str(d)[:10]: (None if (v is None or v is pd.NaT or pd.isna(v))
                           else pd.Timestamp(v).isoformat())
             for d, v in sorted(dict(instants).items())}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode("utf-8")).hexdigest()


def catalogue_labels(site, root=None):
    """The catalogue-times reader of a site's Zarr stores (for
    :func:`resolve_instants`): ``<root>/<site>/overpass_times.json``."""
    def labels():
        p = zc.overpass_path(site, root)
        lab = zc.read_overpass(site, root)
        if not lab:
            raise FileNotFoundError(f"{p}: no catalogue times (the downloader writes "
                                    "overpass_times.json beside the stores)")
        return lab, str(p).replace("\\", "/")
    return labels


def file_labels(labels_path):
    """The catalogue-times reader of a legacy cube (its production
    ``overpass_times.json``)."""
    def labels():
        if labels_path is None:
            raise ValueError("the catalogue instants need labels_path (overpass_times.json)")
        return read_sensing_times(labels_path), str(labels_path).replace("\\", "/")
    return labels


def instants_identity(instants, labels, years=None):
    """What a run's instants are, without opening a store: ``{"source",
    "path", "sha256"}`` (:func:`resolve_instants`, :func:`instants_sha`).
    The package compares it with the record's: another source or file is
    refused, other content of the same file rebuilds the record.
    ``years``: hash only the dates of these calendar years (the record's),
    so dates the downloader adds for other years change nothing."""
    d, source, path = resolve_instants(instants, labels)
    if years is not None:
        ys = {str(int(y)) for y in years}
        d = {k: v for k, v in d.items() if str(k)[:4] in ys}
    return {"source": source, "path": path, "sha256": instants_sha(d)}


# ─────────────────────────────────────────────────────────────────────────────
#  the per-scene offset (§1.10, §2.3)
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_class(versions):
    """"offset" (every version >= 04.00), "none" (every one < 04.00),
    "mixed", or None (unknown)."""
    vs = []
    for v in versions or ():
        try:
            vs.append(float(str(v).lstrip("N")))
        except ValueError:
            continue
    if not vs:
        return None
    lim = float(OFFSET_RULE["offset_baseline"])
    if all(v >= lim for v in vs):
        return "offset"
    if all(v < lim for v in vs):
        return "none"
    return "mixed"


def scene_offset_stats(b12_raw, scl_u8, b_neg_raw=(), fill=obs.FILL):
    """The per-scene statistics of the offset rule (one scene or a block of
    scenes, (..., H, W)).

    Returns a dict of int arrays over the leading axes: ``n_px``,
    ``n_valid`` (B12 not fill, not 0), ``n_low`` (valid with 1 <= DN <
    500), ``n_water`` (SCL 6 with valid B12), ``n_water_low``, ``n_neg``
    (values < 0, fill excluded, in B12 and every array of ``b_neg_raw``).
    """
    b = np.asarray(b12_raw)
    s = np.asarray(scl_u8)
    low = int(OFFSET_RULE["low_dn"])
    ax = (-2, -1)
    valid = (b != fill) & (b != 0)
    lowv = valid & (b >= 1) & (b < low)
    water = valid & (s == int(OFFSET_RULE["water_scl"]))
    neg = ((b < 0) & (b != fill)).sum(axis=ax)
    for a in b_neg_raw:
        a = np.asarray(a)
        neg = neg + ((a < 0) & (a != fill)).sum(axis=ax)
    return {"n_px": np.full(b.shape[:-2], b.shape[-2] * b.shape[-1], np.int64),
            "n_valid": valid.sum(axis=ax), "n_low": lowv.sum(axis=ax),
            "n_water": water.sum(axis=ax), "n_water_low": (water & lowv).sum(axis=ax),
            "n_neg": neg}


def _strong(st, bl):
    """The strong decision of one scene, or (None, reason)."""
    r = OFFSET_RULE
    if st.get("n_neg") is not None and st["n_neg"] > 0:
        return 0, "negative values in the delivered bands: no offset"
    nw = st.get("n_water")
    wshare = (st["n_water_low"] / nw) if nw else None
    water_offset = nw is not None and nw >= r["water_min_px"] and wshare <= r["water_low_offset"]
    if bl == "none":
        if water_offset:
            return "unclear", "baseline < 04.00 but the water B12 is >= 500 DN"
        return 0, "processing baseline < 04.00"
    if nw is not None and nw >= r["water_min_px"]:
        if wshare >= r["water_low_none"]:
            return 0, f"water B12 below {r['low_dn']} DN (share {wshare:.3f})"
        if wshare <= r["water_low_offset"]:
            if bl == "mixed":
                return "unclear", "mixed baselines with the water B12 >= 500 DN"
            return 1000, f"water B12 at or above {r['low_dn']} DN (share {wshare:.3f})"
        return "unclear", f"water B12 share below {r['low_dn']} DN is {wshare:.3f}"
    nv, npx = st.get("n_valid"), st.get("n_px")
    if nv is not None and npx and nv >= r["frame_min_valid_share"] * npx and nv > 0:
        if st["n_low"] / nv > r["frame_low_none"]:
            return 0, f"frame: {st['n_low'] / nv:.3f} of B12 below {r['low_dn']} DN"
    return None, "no strong evidence"


def decide_offsets(stats, baselines=None, year_state=None):
    """The frozen per-scene offset decision (:data:`OFFSET_RULE`).

    Parameters
    ----------
    stats : dict
        ``{date: {"n_px", "n_valid", "n_low", "n_water", "n_water_low",
        "n_neg"}}`` (:func:`scene_offset_stats`; keys may be missing, e.g.
        from the store's ``radiometry`` counts only).
    baselines : dict, optional
        ``{date: [processing versions]}`` (scene_meta.json, else the
        store's ``input_baselines``).
    year_state : dict, optional
        ``{year: offset_state}`` of the stores' ``radiometry`` attribute.

    Returns
    -------
    pandas.DataFrame
        Indexed by date: ``offset`` (0, 1000 or NaN), ``state`` (one of
        :data:`OFFSET_STATES`), ``reason``, ``strong`` (bool).
    """
    baselines = baselines or {}
    year_state = {str(k): v for k, v in (year_state or {}).items()}
    rows = {}
    for d in sorted(stats):
        bl = _baseline_class(baselines.get(d))
        val, why = _strong(dict(stats[d]), bl)
        rows[d] = {"value": val, "reason": why, "baseline": bl}
    regime = {}
    for d, r in rows.items():
        if r["value"] in (0, 1000) and r["baseline"] != "none":
            regime.setdefault(d[:4], set()).add(r["value"])
    out = []
    for d, r in rows.items():
        val, why, strong = r["value"], r["reason"], r["value"] in (0, 1000)
        if val is None:
            reg = regime.get(d[:4], set())
            ys = year_state.get(d[:4])
            if len(reg) == 1:
                val = next(iter(reg))
                why = f"the regime of the year's decided scenes ({val})"
            elif len(reg) > 1:
                val, why = "unclear", "the year's decided scenes disagree"
            elif ys == "+1000 in the values":
                val, why = 1000, "the store's year offset_state"
            elif ys == "no offset":
                val, why = 0, "the store's year offset_state"
            else:
                val, why = "unclear", f"no evidence (year state {ys!r})"
        state = ("unclear" if val == "unclear" else
                 "+1000 in the values" if val == 1000 else "no offset")
        out.append({"date": d, "offset": np.nan if val == "unclear" else float(val),
                    "state": state, "reason": why, "strong": bool(strong)})
    df = pd.DataFrame(out, columns=["date", "offset", "state", "reason", "strong"])
    return df.set_index("date")


def radiometry_stats(attrs):
    """``{date: {"n_valid", "n_low"}}`` from a store's ``radiometry``
    attribute (the downloader's per-date counts of B12, else B11), plus the
    year's ``offset_state`` and the band used."""
    rad = zc.store_radiometry(attrs) or {}
    per = rad.get("per_date_valid_low") or {}
    return ({d: {"n_valid": int(v), "n_low": int(lo)} for d, (v, lo) in per.items()},
            rad.get("offset_state"), rad.get("band"))


# ─────────────────────────────────────────────────────────────────────────────
#  fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def _sha_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def store_fingerprint(path, full=False):
    """Fingerprint of one store: sha256 of the consolidated metadata and of
    the sorted ``(chunk key, size)`` list of every file below the band
    folders; ``full=True`` adds ``content_sha256`` over (key, file sha256).

    Returns
    -------
    dict
        ``fingerprint``, ``meta_sha256``, ``chunks_sha256``, ``n_files``,
        ``bytes`` (and ``content_sha256``).
    """
    path = Path(path)
    meta = (path / ".zmetadata").read_bytes()
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.parent != path:
            files.append((p.relative_to(path).as_posix(), p.stat().st_size))
    files.sort()
    ch = hashlib.sha256("".join(f"{k}\t{s}\n" for k, s in files).encode("utf-8")).hexdigest()
    ms = hashlib.sha256(meta).hexdigest()
    out = {"fingerprint": hashlib.sha256(f"{ms}\n{ch}".encode()).hexdigest(),
           "meta_sha256": ms, "chunks_sha256": ch, "n_files": len(files),
           "bytes": int(sum(s for _, s in files))}
    if full:
        h = hashlib.sha256()
        for k, _ in files:
            h.update(f"{k}\t{_sha_file(path / k)}\n".encode("utf-8"))
        out["content_sha256"] = h.hexdigest()
    return out


def netcdf_fingerprint(path, full=False):
    """Fingerprint of a legacy netCDF cube: sha256 of its header (dims,
    variables, dtypes, attributes, chunking) and size; ``full=True`` adds
    the sha256 of the file."""
    import xarray as xr

    path = Path(path)
    with xr.open_dataset(path, mask_and_scale=False, decode_times=False) as ds:
        head = {"dims": {k: int(v) for k, v in ds.sizes.items()},
                "vars": {k: {"dtype": str(v.dtype), "dims": list(v.dims),
                             "attrs": {a: str(x) for a, x in v.attrs.items()},
                             "chunks": str(v.encoding.get("chunksizes"))}
                         for k, v in ds.variables.items()},
                "attrs": {a: str(x) for a, x in ds.attrs.items()}}
    hs = hashlib.sha256(json.dumps(head, sort_keys=True).encode("utf-8")).hexdigest()
    size = path.stat().st_size
    out = {"fingerprint": hashlib.sha256(f"{hs}\n{size}".encode()).hexdigest(),
           "header_sha256": hs, "bytes": int(size)}
    if full:
        out["content_sha256"] = _sha_file(path)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the scene source
# ─────────────────────────────────────────────────────────────────────────────

Block = collections.namedtuple("Block", "positions rows data lattice")


class SceneSource:
    """The scenes of one site (the protocol of spec §1.2).

    Members (subclasses fill ``_dates_all``, ``_parts`` and the readers):

    * ``dates``, ``instants``: the record (dates with a valid instant);
    * ``grid``: :class:`Grid`; ``lattice20``: (py, px) or None;
    * ``offset(q)``, ``offsets``: the per-scene BOA offset (0 or 1000);
    * ``position(q)``: record position -> (part, local index);
    * ``read_scene(q, bands, rows=None)``, ``read_block(qs, bands, rows)``;
    * ``row_blocks(bands, block_rows=256, positions=None, pair20=False)``;
    * ``fingerprint(part=None, full=False)``;
    * ``window(start, end)``: record positions of a window.
    """

    kind = "abstract"
    bands = ()

    def __init__(self, site, instants, keep_dates=None, cache_mb=CACHE_MB, lattice="measure",
                 offsets=None):
        self.site = site
        self.cache = BlockCache(cache_mb)
        inst = dict(instants)
        _check_instants(inst)
        self._instants_all = inst
        sel = []
        keep = None if keep_dates is None else {str(d)[:10] for d in keep_dates}
        for i, d in enumerate(self._dates_all):
            t = inst.get(d, pd.NaT)
            if t is pd.NaT or pd.isna(t):
                continue
            if keep is not None and d not in keep:
                continue
            sel.append(i)
        self._sel = np.asarray(sel, np.int64)
        self.dates = [self._dates_all[i] for i in self._sel]
        self.instants = pd.DatetimeIndex([inst[d] for d in self.dates])
        if len(self.instants) > 1 and not (np.diff(self.instants.values) > np.timedelta64(0)).all():
            raise InputsError(f"{site}: record instants are not strictly increasing")
        kept = set(self.dates)
        self.dropped = [d for d in self._dates_all if d not in kept]
        self._lattice = lattice
        self._lattice_report = None
        self._frozen_offsets = None if offsets is None else {str(k)[:10]: v
                                                             for k, v in dict(offsets).items()}
        self._offset_table = None

    # -- record --------------------------------------------------------------
    @property
    def T(self):
        return len(self.dates)

    @property
    def instants_sha256(self):
        """:func:`instants_sha` of the instants of the source's store dates
        (NaT where a date has none): the record's instants, recorded."""
        return instants_sha({d: self._instants_all.get(d, pd.NaT) for d in self._dates_all})

    def __len__(self):
        return self.T

    def window(self, start, end):
        """Record positions of the dates in ``[start, end)``."""
        t0, t1 = zc.parse_date(start), zc.parse_date(end)
        d = pd.DatetimeIndex(pd.to_datetime(self.dates))
        return np.flatnonzero((d >= t0) & (d < t1)).astype(np.int64)

    def position(self, q):
        """Record position ``q`` -> ``(part, local index)`` (part: the year
        of a Zarr store, None for a netCDF cube)."""
        return self._position_of_all(int(self._sel[int(q)]))

    def _check_bands(self, bands):
        bands = [bands] if isinstance(bands, str) else list(bands)
        bad = [b for b in bands if b not in self.bands]
        if bad:
            raise KeyError(f"{self.kind} source of {self.site}: bands {bad} not available "
                           f"(has {self.bands})")
        return bands

    # -- reading -------------------------------------------------------------
    def _chunk(self, part, band, tb, rc):
        """Raw (nt, rows of row chunk rc, W) of one chunk block (cached)."""
        return self.cache.get((part, band, int(tb), int(rc)),
                              lambda: self._load(part, band, int(tb), int(rc)))

    def _rows_of(self, rows):
        if rows is None:
            return 0, self.grid.H
        if isinstance(rows, slice):
            r0, r1, st = rows.indices(self.grid.H)
            if st != 1:
                raise ValueError("row slices must be contiguous")
            return r0, r1
        r0, r1 = (int(v) for v in rows)
        return r0, min(r1, self.grid.H)

    def _assemble(self, part, band, local, r0, r1):
        """Raw values of local indices ``local`` (sorted, one time block or
        more) and rows [r0, r1)."""
        local = np.asarray(local, np.int64)
        out = None
        for tb in np.unique(local // T_BLOCK):
            m = local // T_BLOCK == tb
            li = local[m] - tb * T_BLOCK
            pieces = []
            for rc in range(r0 // ROW_BLOCK, (r1 - 1) // ROW_BLOCK + 1):
                c = self._chunk(part, band, tb, rc)
                a0 = max(r0, rc * ROW_BLOCK) - rc * ROW_BLOCK
                a1 = min(r1, (rc + 1) * ROW_BLOCK) - rc * ROW_BLOCK
                pieces.append(c[li, a0:a1, :])
            blk = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
            if out is None:
                out = np.empty((len(local),) + blk.shape[1:], blk.dtype)
            out[np.flatnonzero(m)] = blk
        return out

    def _finish(self, band, a):
        if band == "SCL":
            return obs.scl_u8(a, self.fill.get(band, obs.FILL))
        if band == "CLD":
            return obs.cld_u8(a, self.fill.get(band, obs.FILL))
        return np.ascontiguousarray(a)

    def read_block(self, qs, bands, rows=None):
        """``{band: (len(qs), rows, W)}`` raw values of record positions
        ``qs`` (int16; SCL and CLD as checked uint8)."""
        bands = self._check_bands(bands)
        qs = np.atleast_1d(np.asarray(qs, np.int64))
        r0, r1 = self._rows_of(rows)
        allpos = [self.position(q) for q in qs]
        out = {}
        for b in bands:
            res = None
            for part in dict.fromkeys(p for p, _ in allpos):
                idx = [k for k, (p, _) in enumerate(allpos) if p == part]
                local = np.asarray([allpos[k][1] for k in idx], np.int64)
                order = np.argsort(local, kind="stable")
                vals = self._assemble(part, b, local[order], r0, r1)
                if res is None:
                    res = np.empty((len(qs),) + vals.shape[1:], vals.dtype)
                res[np.asarray(idx)[order]] = vals
            out[b] = self._finish(b, res)
        return out

    def read_scene(self, q, bands, rows=None):
        """``{band: (rows, W)}`` raw values of record position ``q``."""
        return {b: a[0] for b, a in self.read_block([int(q)], bands, rows).items()}

    def row_blocks(self, bands, block_rows=ROW_BLOCK, positions=None, pair20=False):
        """Chunk-aligned blocks in chunk order: for each time block of 16
        dates (per part), each row block of ``block_rows`` rows.

        Yields
        ------
        Block
            ``positions`` (record positions), ``rows`` (slice), ``data``
            ({band: (n, rows, W)}), ``lattice`` (with ``pair20``: the 20 m
            phase relative to the block's first row; else None). With
            ``pair20`` the row boundaries move to the lattice rows (r = py
            mod 2), so no 20 m block straddles two row blocks and the 20 m
            indices of a block equal the frame's; the cache still decodes
            each chunk once.
        """
        bands = self._check_bands(bands)
        if pair20 and int(block_rows) % 2:
            raise ValueError(f"block_rows {block_rows} is odd: with pair20 a 20 m block would "
                             "straddle two row blocks (use an even number)")
        lat = self.lattice20 if pair20 else None
        H = self.grid.H
        bounds = list(range(0, H, int(block_rows))) + [H]
        if pair20 and lat is not None and lat[0] == 1:
            bounds = [0] + [min(H, b + 1) for b in bounds[1:-1]] + [H]
            bounds = sorted(set(bounds))
        for qs in self.time_blocks(positions):
            for r0, r1 in zip(bounds[:-1], bounds[1:]):
                data = self.read_block(qs, bands, slice(r0, r1))
                local_lat = None if lat is None else ((lat[0] - r0) % 2, lat[1])
                yield Block(np.asarray(qs, np.int64), slice(r0, r1), data, local_lat)

    def time_blocks(self, positions=None):
        """Record positions grouped by stored time block (the 16 dates of one
        chunk of one part), in chunk order: reading one group at once decodes
        each chunk once, whatever the cache size.

        Returns
        -------
        list of int64 arrays
        """
        pos = np.arange(self.T, dtype=np.int64) if positions is None \
            else np.asarray(positions, np.int64)
        if len(pos) > 1 and not (np.diff(pos) > 0).all():
            raise ValueError("positions must be strictly increasing (chunk order)")
        groups = collections.OrderedDict()
        for q in pos:
            part, local = self.position(q)
            groups.setdefault((part, int(local) // T_BLOCK), []).append(int(q))
        return [np.asarray(qs, np.int64) for qs in groups.values()]

    # -- lattice20 -----------------------------------------------------------
    def _sample_positions(self, n=LATTICE_SCENES):
        T = self.T
        if T == 0:
            return []
        return sorted(set(np.linspace(0, T - 1, min(n, T)).round().astype(int).tolist()))

    def measure_lattice20(self, n_scenes=LATTICE_SCENES, max_rows=2 * ROW_BLOCK):
        """Measure the 20 m lattice on up to ``n_scenes`` scenes spread over
        the record (rows of a central, even-aligned band of ``max_rows``)."""
        bands = [b for b in LATTICE_BANDS if b in self.bands]
        if not bands:
            return {"phase": None, "exact": False, "reason": "no 20 m band"}
        H = self.grid.H
        r0 = max(0, (H // 2 - max_rows // 2)) // 2 * 2
        r1 = min(H, r0 + max_rows)
        arrays = []
        for q in self._sample_positions(n_scenes):
            sc = self.read_scene(q, bands, slice(r0, r1))
            arrays += [(b, sc[b]) for b in bands]
        rep = measure_lattice20(arrays, expected_lattice20(self.grid))
        rep.update(rows=[int(r0), int(r1)], n_scenes=len(self._sample_positions(n_scenes)))
        return rep

    @property
    def lattice20(self):
        """The native 20 m lattice phase (py, px), or None (§2.4)."""
        if isinstance(self._lattice, str) and self._lattice == "measure":
            rep = self.measure_lattice20()
            self._lattice_report = rep
            self._lattice = tuple(rep["phase"]) if rep.get("phase") else None
        return self._lattice

    @property
    def lattice_report(self):
        _ = self.lattice20
        return self._lattice_report

    # -- offsets -------------------------------------------------------------
    def offset_table(self):
        """The per-scene offset decisions (:func:`decide_offsets`) of the
        record; frozen ones when given at construction."""
        if self._offset_table is None:
            self._offset_table = self._offsets_decided()
        return self._offset_table

    def _offsets_decided(self):
        if self._frozen_offsets is not None:
            rows = []
            for d in self.dates:
                v = self._frozen_offsets.get(d)
                ok = v is not None and not (isinstance(v, float) and np.isnan(v)) \
                    and int(v) in settings.OFFSETS
                rows.append({"date": d, "offset": float(v) if ok else np.nan,
                             "state": ("unclear" if not ok else "+1000 in the values"
                                       if int(v) == 1000 else "no offset"),
                             "reason": "frozen" if ok else "missing from the frozen offsets",
                             "strong": bool(ok)})
            return pd.DataFrame(rows, columns=["date", "offset", "state", "reason",
                                               "strong"]).set_index("date")
        return self._offsets_provisional()

    def _offsets_provisional(self):
        raise NotImplementedError

    def offset(self, q):
        """The BOA offset (0 or 1000 DN) of record position ``q``;
        :class:`OffsetUnclear` when undecided."""
        d = self.dates[int(q)]
        row = self.offset_table().loc[d]
        if row["state"] == "unclear":
            raise OffsetUnclear(f"{self.site} {d}: BOA offset unclear ({row['reason']})")
        return int(row["offset"])

    @property
    def offsets(self):
        """``(T,)`` int array of the record's offsets (raises if any is unclear)."""
        return np.asarray([self.offset(q) for q in range(self.T)], np.int64)

    # -- misc ----------------------------------------------------------------
    def describe(self):
        return {"site": self.site, "kind": self.kind, "T": self.T,
                "first": self.dates[0] if self.dates else None,
                "last": self.dates[-1] if self.dates else None,
                "dropped_without_instant": len(self.dropped), "bands": list(self.bands),
                "grid": self.grid.describe(), "cache": self.cache.stats()}

    def close(self):
        self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ZarrSource(SceneSource):
    """:class:`SceneSource` over the per-year Zarr stores of a site.

    Parameters
    ----------
    site : str
    start, end : str, optional
        The period (end exclusive); default :data:`UNION_PERIOD`, clipped
        to the years present (``missing="skip"``).
    root : path, optional
        The cubes root (``zarr_cubes.cubes_root``).
    instants : "catalogue" (default), "labels", a path or a mapping
        The catalogue times of ``<root>/<site>/overpass_times.json`` (the
        datatake start of the product names, the default of every run;
        "labels" is an alias); an explicit ``sensing_times.json`` path (the
        optional diagnostic); or ``{date: instant}`` (:func:`resolve_instants`).
    keep_dates : iterable, optional
        Restrict the record (e.g. dates with a finite h_B).
    offsets : mapping, optional
        The frozen ``{date: 0 or 1000}`` (inputs report); without it the
        decision is provisional, from the stores' attributes and the
        scene metadata.
    scene_meta : dict or path, optional
        :func:`read_scene_meta` (default ``cubes/<site>/scene_meta.json`` if
        present).
    lattice : "measure", (py, px) or None
    cache_mb : float
    """

    kind = "zarr"
    bands = ZARR_BANDS

    def __init__(self, site, start=None, end=None, root=None, instants=CATALOGUE,
                 keep_dates=None, offsets=None, scene_meta=None, lattice="measure",
                 cache_mb=CACHE_MB):
        import zarr

        self.root = root
        start = start or UNION_PERIOD[0]
        end = end or UNION_PERIOD[1]
        with warnings.catch_warnings():
            # years that differ in their offset regime are expected: the
            # per-scene offset handles them (kept in radiometry_warning)
            warnings.filterwarnings("ignore", message=".*disagree on the L2A.*")
            ds = zc.open_cube(site, start, end, root=root, missing="skip")
        try:
            self.grid = grid_of(ds)
            self.years = [int(y) for y in ds.attrs.get("years", [])]
            self.missing_years = [int(y) for y in ds.attrs.get("missing_years", [])]
            self.fill = {b: (zc.fill_of(ds[b]) if zc.fill_of(ds[b]) is not None else obs.FILL)
                         for b in zc.band_names(ds)}
            all_dates = zc.dates(ds)
            self.radiometry_warning = ds.attrs.get("radiometry_warning")
        finally:
            ds.close()
        bad = {b: f for b, f in self.fill.items() if f != obs.FILL}
        if bad:
            raise InputsError(f"{site}: unexpected fills {bad}")
        self.bands = tuple(b for b in ZARR_BANDS if b in self.fill)
        self._paths = {y: zc.store_path(site, y, root) for y in self.years}
        self._groups = {y: zarr.open_group(str(p), mode="r", zarr_format=2)
                        for y, p in self._paths.items()}
        self._local = {}
        part_of = []
        for y in self.years:
            with zc.open_store(self._paths[y], bands=["SCL"]) as dy:
                dd = zc.dates(dy)
            self._local[y] = {d: i for i, d in enumerate(dd)}
        for d in all_dates:
            y = int(d[:4])
            if d not in self._local.get(y, {}):
                raise InputsError(f"{site} {d}: not in the {y} store's dates")
            part_of.append((y, self._local[y][d]))
        self._part_of = part_of
        self._dates_all = list(all_dates)
        self._attrs = {y: zc.store_attrs(p) for y, p in self._paths.items()}
        if isinstance(scene_meta, (str, os.PathLike)):
            scene_meta = read_scene_meta(scene_meta)
        elif scene_meta is None:
            p = zc.site_dir(site, root) / SCENE_META_NAME
            scene_meta = read_scene_meta(p) if p.is_file() else None
        self.scene_meta = scene_meta
        instants, self.instant_source, self.instant_path = resolve_instants(
            instants, catalogue_labels(site, root))
        super().__init__(site, instants, keep_dates, cache_mb, lattice, offsets)

    def _position_of_all(self, i):
        return self._part_of[i]

    def _load(self, year, band, tb, rc):
        arr = self._groups[year][band]
        n = arr.shape[0]
        t0, t1 = tb * T_BLOCK, min(n, (tb + 1) * T_BLOCK)
        r0, r1 = rc * ROW_BLOCK, min(self.grid.H, (rc + 1) * ROW_BLOCK)
        return np.asarray(arr[t0:t1, r0:r1, :])

    def fingerprint(self, part=None, full=False):
        """:func:`store_fingerprint` of one year (``part``) or ``{year: ...}``."""
        if part is not None:
            return store_fingerprint(self._paths[int(part)], full)
        return {str(y): store_fingerprint(p, full) for y, p in self._paths.items()}

    def baselines(self):
        """``{date: [processing versions]}``: the scene metadata if present,
        else the stores' ``input_baselines``."""
        if self.scene_meta:
            return {d: list(v.get("processing_baselines", []))
                    for d, v in self.scene_meta.get("dates", {}).items()}
        out = {}
        for a in self._attrs.values():
            out.update((zc.store_baselines(a) or {}).get("per_date", {}))
        return out

    def year_states(self):
        return {str(y): radiometry_stats(a)[1] for y, a in self._attrs.items()}

    def _offsets_provisional(self):
        stats = {}
        for y, a in self._attrs.items():
            per, _, _ = radiometry_stats(a)
            H, W = self.grid.shape
            for d, st in per.items():
                stats[d] = dict(st, n_px=H * W)
        for d in self.dates:
            stats.setdefault(d, {})
        tab = decide_offsets(stats, self.baselines(), self.year_states())
        return tab.loc[[d for d in self.dates]]

    def angles(self, lon=None, lat=None):
        """The openEO angle bands at a point (frame centre by default) for
        the record dates (diagnostic only: those angles are quantised;
        sensing times come from the STAC sun angles); None without angles
        stores."""
        if not zc.list_angle_years(self.site, self.root):
            return None
        if lon is None or lat is None:
            lon, lat = self.grid.lonlat_centre()
        start, end = self.dates[0], (pd.Timestamp(self.dates[-1]) + pd.Timedelta(days=1))
        df = zc.angles_at(self.site, lon, lat, start, str(end.date()), root=self.root,
                          missing="skip")
        df.index = df.index.strftime("%Y-%m-%d")
        return df.reindex(self.dates)


class NetCDFSource(SceneSource):
    """:class:`SceneSource` over a legacy netCDF cube (B03, B08, SCL): the
    reproduction gates and the local runs on the 20 m cubes (``--res 20``).
    Values are offset-harmonised (o_s = 0 for every scene,
    ``obs.check_boa``); the lattice is measured on SCL alone (pass
    ``lattice=None`` on a 20 m grid, which has no 20 m sub-lattice).

    Parameters
    ----------
    path : path
    site : str
    instants : "catalogue" (default), "labels", a path or a mapping
        The catalogue times of ``labels_path`` (the production
        overpass_times.json: the datatake start of the product names); an
        explicit ``sensing_times.json`` path; or ``{date: instant}``.
    start, end : str, optional
        Restrict the dates (end exclusive).
    labels_path : path, optional
        The catalogue times (required for "catalogue").
    """

    kind = "netcdf"
    bands = LEGACY_BANDS

    def __init__(self, path, site, instants=CATALOGUE, start=None, end=None, keep_dates=None,
                 labels_path=None, lattice="measure", cache_mb=CACHE_MB):
        import xarray as xr

        self.path = Path(path)
        self._ds = xr.open_dataset(self.path, mask_and_scale=False, decode_times=True)
        ds = self._ds
        self._tdim = zc._time_dim(ds)
        for b in LEGACY_BANDS:
            if b not in ds.data_vars:
                raise InputsError(f"{path}: band {b} missing")
            if ds[b].dims != (self._tdim, "y", "x"):
                raise InputsError(f"{path}: {b} dims {ds[b].dims} are not (t, y, x)")
            if str(ds[b].dtype) != "int16":
                raise InputsError(f"{path}: {b} is {ds[b].dtype}, not int16")
            if any(k in ds[b].attrs or k in ds[b].encoding for k in ("scale_factor", "add_offset")):
                raise InputsError(f"{path}: {b} carries scale/offset attributes")
        self.fill = {b: int(zc.fill_of(ds[b])) for b in LEGACY_BANDS}
        self.grid = grid_of(ds)
        dates = [str(d)[:10] for d in pd.DatetimeIndex(ds[self._tdim].values).strftime("%Y-%m-%d")]
        idx = np.arange(len(dates))
        if start is not None or end is not None:
            d = pd.DatetimeIndex(pd.to_datetime(dates))
            m = np.ones(len(d), bool)
            if start is not None:
                m &= d >= zc.parse_date(start)
            if end is not None:
                m &= d < zc.parse_date(end)
            idx = idx[m]
        self._dates_all = [dates[i] for i in idx]
        self._cube_index = idx

        read_labels = file_labels(labels_path)

        def labels():
            try:
                return read_labels()
            except ValueError:
                self._ds.close()
                raise

        instants, self.instant_source, self.instant_path = resolve_instants(instants, labels)
        super().__init__(site, instants, keep_dates, cache_mb, lattice, offsets=None)
        self.years = sorted({int(d[:4]) for d in self._dates_all})
        self.missing_years = []

    def _position_of_all(self, i):
        return None, int(self._cube_index[i])

    def _load(self, part, band, tb, rc):
        n = self._ds.sizes[self._tdim]
        t0, t1 = tb * T_BLOCK, min(n, (tb + 1) * T_BLOCK)
        r0, r1 = rc * ROW_BLOCK, min(self.grid.H, (rc + 1) * ROW_BLOCK)
        return np.asarray(self._ds[band].isel({self._tdim: slice(t0, t1),
                                               "y": slice(r0, r1)}).values)

    def _offsets_provisional(self):
        return pd.DataFrame({"date": self.dates, "offset": 0.0, "state": "no offset",
                             "reason": "legacy cube: offset-harmonised values (DN - 1000)",
                             "strong": True}).set_index("date")

    def fingerprint(self, part=None, full=False):
        return netcdf_fingerprint(self.path, full)

    def close(self):
        super().close()
        self._ds.close()


# ─────────────────────────────────────────────────────────────────────────────
#  the scene metadata sidecar (§1.10)
# ─────────────────────────────────────────────────────────────────────────────

def scene_meta_path(site, root=None):
    return zc.site_dir(site, root) / SCENE_META_NAME


def _item_record(f):
    p = f.get("properties", {})
    bbox = f.get("bbox")
    cen = None
    if bbox and len(bbox) >= 4:
        cen = [round((bbox[0] + bbox[2]) / 2.0, 6), round((bbox[1] + bbox[3]) / 2.0, 6)]
    tile = str(p.get("grid:code") or p.get("s2:mgrs_tile") or "")
    return {"id": f.get("id"), "tile": tile.replace("MGRS-", ""),
            "datetime": p.get("datetime") or p.get("start_datetime"),
            "platform": p.get("platform"),
            "processing_baseline": p.get("processing:version") or p.get("s2:processing_baseline"),
            "timeliness": p.get("product:timeliness_category") or p.get("product:timeliness"),
            "cloud_cover": p.get("eo:cloud_cover"), "snow_cover": p.get("eo:snow_cover"),
            "relative_orbit": p.get("sat:relative_orbit"),
            "absolute_orbit": p.get("sat:absolute_orbit"),
            "orbit_state": p.get("sat:orbit_state"),
            "sun_azimuth": p.get("view:sun_azimuth"), "sun_elevation": p.get("view:sun_elevation"),
            "view_azimuth": p.get("view:azimuth"), "view_incidence": p.get("view:incidence_angle"),
            "datatake_id": p.get("eopf:datatake_id"), "bbox": bbox, "centre_lonlat": cen}


def scene_meta_from_items(features, site=None, query=None):
    """The scene metadata sidecar from CDSE STAC features (``sentinel-2-l2a``).

    Per date (the UTC date of the item's datetime): every item record
    (:func:`_item_record`), and a summary: platforms, relative orbits,
    processing baselines, tiles, the mean tile cloud cover, and
    ``double_overpass`` (more than one relative orbit on the date).
    """
    dates = {}
    for f in features:
        r = _item_record(f)
        if not r["datetime"]:
            continue
        d = str(_naive_utc(r["datetime"]).date())
        dates.setdefault(d, {"items": []})["items"].append(r)
    for d, v in dates.items():
        it = sorted(v["items"], key=lambda r: (str(r["tile"]), str(r["id"])))
        v["items"] = it
        v["platforms"] = sorted({str(r["platform"]) for r in it if r["platform"]})
        v["relative_orbits"] = sorted({int(r["relative_orbit"]) for r in it
                                       if r["relative_orbit"] is not None})
        v["processing_baselines"] = sorted({str(r["processing_baseline"]) for r in it
                                            if r["processing_baseline"]})
        v["tiles"] = sorted({str(r["tile"]) for r in it if r["tile"]})
        cc = [float(r["cloud_cover"]) for r in it if r["cloud_cover"] is not None]
        v["cloud_cover_mean"] = round(float(np.mean(cc)), 4) if cc else None
        v["double_overpass"] = len(v["relative_orbits"]) > 1
        ids = [r["id"] for r in it]
        v["duplicate_tiles"] = sorted({t for t in v["tiles"]
                                       if sum(1 for r in it if r["tile"] == t) > 1})
        v["n_items"] = len(ids)
    return {"format": SCENE_META_FORMAT, "site": site, "query": query or {},
            "dates": dict(sorted(dates.items()))}


def write_scene_meta(path, meta):
    """Write the sidecar (sorted keys, LF, atomically)."""
    _write_json(path, meta)


def read_scene_meta(path):
    """Read and check a scene metadata sidecar."""
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format") != SCENE_META_FORMAT:
        raise InputsError(f"{path}: not a scene metadata sidecar ({meta.get('format')!r})")
    return meta


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(obj, sort_keys=True, indent=1, default=_jsonable, allow_nan=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data + "\n")
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
#  the inputs report (§1.11)
# ─────────────────────────────────────────────────────────────────────────────

def check_store(path):
    """§1.11 item 1 for one store: complete, FORMAT_ID, dims, dtype and
    ``_FillValue`` per band, no active scale or offset, axes and chunks.

    Returns
    -------
    dict
        ``ok``, ``problems`` (list of str), ``bands`` (dtype, fill, chunks),
        ``offset_state``, ``delivered_attrs`` (the ``delivered_*`` names).
    """
    path = Path(path)
    probs = []
    attrs = zc.store_attrs(path)
    if attrs.get("complete") != 1:
        probs.append("complete != 1")
    if attrs.get("format") != zc.FORMAT_ID:
        probs.append(f"format {attrs.get('format')!r} != {zc.FORMAT_ID!r}")
    meta = (zc._consolidated(path) or {}).get("metadata", {})
    bands = {}
    delivered = {}
    ds = zc.open_store(path)
    try:
        for b in ZARR_BANDS:
            if b not in ds.data_vars:
                probs.append(f"{b} missing")
                continue
            da_ = ds[b]
            za = meta.get(f"{b}/.zattrs", {})
            zr = meta.get(f"{b}/.zarray", {})
            if da_.dims != ("t", "y", "x"):
                probs.append(f"{b} dims {da_.dims}")
            if str(da_.dtype) != "int16":
                probs.append(f"{b} dtype {da_.dtype}")
            f = zc.fill_of(da_)
            if f != obs.FILL:
                probs.append(f"{b} _FillValue {f}")
            for k in ("scale_factor", "add_offset"):
                if k in za or k in da_.attrs or k in da_.encoding:
                    probs.append(f"{b} carries an active {k}")
            delivered[b] = sorted(k for k in za if k.startswith("delivered_"))
            want = list(zc.chunk_shape(da_.shape))
            if zr.get("chunks") and list(zr["chunks"]) != want:
                probs.append(f"{b} chunks {zr['chunks']} != {want}")
            bands[b] = {"dtype": str(da_.dtype), "fill": f, "chunks": zr.get("chunks")}
        x, y = ds["x"].values, ds["y"].values
        if len(x) > 1 and not np.all(np.diff(x) > 0):
            probs.append("x not ascending")
        if len(y) > 1 and not np.all(np.diff(y) < 0):
            probs.append("y not descending")
        if len(x) > 1 and not np.allclose(np.diff(x), 10.0):
            probs.append("x spacing is not 10 m")
        if len(y) > 1 and not np.allclose(np.diff(y), -10.0):
            probs.append("y spacing is not 10 m")
    finally:
        ds.close()
    _, state, band = radiometry_stats(attrs)
    if state is None:
        probs.append("no radiometry attribute")
    return {"ok": not probs, "problems": probs, "bands": bands, "offset_state": state,
            "radiometry_band": band, "delivered_attrs": delivered}


def scan_source(src, positions=None, log_every=50):
    """§1.11 item 2 by streaming B08, B12, SCL and CLD of every scene (row
    blocks of 16 dates): SCL codes (``obs.scl_u8`` raises outside 0..11),
    the SCL histogram, CLD range (``obs.cld_u8``), the offset statistics.

    Returns
    -------
    dict
        ``{date: {"scl": [13 counts], "cld_fill", "cld_max", **offset stats}}``.
    """
    bands = [b for b in ("B08", "B12", "SCL", "CLD") if b in src.bands]
    out = {}
    t0 = time.time()
    done = 0
    for blk in src.row_blocks(bands, positions=positions):
        if blk.rows.stop == src.grid.H:
            done += len(blk.positions)
            if log_every and done % log_every < len(blk.positions):
                log(f"  [inputs] {src.site}: {done}/{src.T if positions is None else len(positions)}"
                    f" scenes scanned ({time.time() - t0:.0f} s)")
        s8 = blk.data["SCL"]
        st = (scene_offset_stats(blk.data["B12"], s8, [blk.data["B08"]], src.fill.get("B12",
                                                                                         obs.FILL))
              if "B12" in blk.data else None)
        for k, q in enumerate(blk.positions):
            d = src.dates[int(q)]
            e = out.setdefault(d, {"scl": [0] * 13, "cld_fill": 0, "cld_max": None})
            cnt = obs.scl_counts(s8[k])
            e["scl"] = [a + int(b) for a, b in zip(e["scl"], cnt)]
            if "CLD" in blk.data:
                c = blk.data["CLD"][k]
                e["cld_fill"] += int(np.count_nonzero(c == obs.CLD_NODATA_U8))
                pres = c[c != obs.CLD_NODATA_U8]
                if pres.size:
                    m = int(pres.max())
                    e["cld_max"] = m if e["cld_max"] is None else max(e["cld_max"], m)
            if st is not None:
                for key, arr in st.items():            # summed over row blocks
                    e[key] = e.get(key, 0) + int(arr[k])
    return out


def _iter_files(top):
    """Files under ``top``, depth first in sorted order, lazily (a chunk
    tree is never listed in full)."""
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def disk_check(paths, read_from=None, read_mb=64):
    """§1.11 item 8: free space per path and the sequential read
    throughput of up to ``read_mb`` MB of the files under ``read_from``
    (page cache not dropped: an upper bound, declared). The files are
    walked lazily and the walk stops after ``read_mb``: pass one store
    (or one band of it), never the whole cubes root."""
    out = {"free_gb": {}}
    for p in paths:
        try:
            out["free_gb"][str(p)] = round(shutil.disk_usage(p).free / 1e9, 1)
        except OSError as exc:
            out["free_gb"][str(p)] = f"error: {exc}"
    if read_from is not None:
        n, k, t0 = 0, 0, time.time()
        for f in _iter_files(read_from):
            with open(f, "rb") as fh:
                n += len(fh.read())
            k += 1
            if n >= read_mb * 1e6:
                break
        dt = max(time.time() - t0, 1e-6)
        out["read"] = {"from": str(read_from), "mb": round(n / 1e6, 1), "files": k,
                       "mb_per_s": round(n / 1e6 / dt, 1), "note": "page cache not dropped"}
    return out


def disk_sample_dir(site, root=None):
    """The directory :func:`disk_check` reads at a site: the B08 array of its
    latest store (a few thousand chunk files), or None without a store."""
    years = zc.list_years(site, root)
    if not years:
        return None
    p = zc.store_path(site, max(years), root) / "B08"
    return str(p) if p.is_dir() else str(zc.store_path(site, max(years), root))


def _instants_doc(src):
    """Where a source's instants come from (for the reports)."""
    out = {"source": src.instant_source, "path": getattr(src, "instant_path", None)}
    p = out["path"]
    if p and os.path.isfile(p):
        out["sha256"] = _sha_file(p)
    if src.instant_source == CATALOGUE:
        out["note"] = ("catalogue / product-name time (datatake start), the default of every "
                       "run (user decision 2026-10-14)")
    return out


def inputs_report(site, root=None, start=None, end=None, legacy=None, rtk_grid=None,
                  instants=CATALOGUE, scan=True, full_fingerprint=False, disk_paths=None,
                  out=None, cache_mb=SCAN_CACHE_MB):
    """The §1.11 checks of one site; returns the report (and writes it to
    ``out`` merged into the existing file's ``sites`` mapping, if given).

    Parameters
    ----------
    site : str
    root : path, optional
        The cubes root.
    start, end : str, optional
        The period (default :data:`UNION_PERIOD`).
    legacy : path, optional
        The legacy netCDF cube (grid equality: the whole grid, or the
        window of a store that is an extract of it, ``extract.py``; G-zarr
        reads the stores' ``<year>.verify.json``).
    rtk_grid : path, optional
        Villaviciosa's ``rtk.DEFAULT_GRID`` GeoTIFF (grid equality).
    instants : see :class:`ZarrSource`; default the catalogue times (no
        warning); a ``sensing_times.json`` only when its path is given.
    scan : bool
        Stream the stores for codes, ranges and the offset statistics.
    cache_mb : float
        The block cache of the scan (each chunk is read once: a small cache
        keeps the memory low).
    """
    t0 = time.time()
    rep = {"format": INPUTS_REPORT_FORMAT, "site": site, "problems": [], "warnings": [],
           "offset_rule": OFFSET_RULE, "source": {"kind": "zarr", "root": str(
               zc.cubes_root(root)).replace("\\", "/")}}
    years = zc.list_years(site, root)
    rep["years"] = years
    rep["stores"] = {str(y): check_store(zc.store_path(site, y, root)) for y in years}
    for y, c in rep["stores"].items():
        rep["problems"] += [f"{y}: {p}" for p in c["problems"]]
    extracts = {}
    for y in years:
        e = zc.store_attrs(zc.store_path(site, y, root)).get("extract")
        if e:
            extracts[str(y)] = json.loads(e) if isinstance(e, str) else e
    if extracts:
        rep["extract"] = extracts
        rep["warnings"].append("these stores are a local EXTRACT (real B03, B08, SCL; synthetic "
                               "B02, B04, B8A, B11, B12, CLD): software checks only, never "
                               "results")
    src = ZarrSource(site, start, end, root=root, instants=instants, cache_mb=cache_mb)
    try:
        rep["instants"] = _instants_doc(src)
        rep["record"] = {"T": src.T, "dropped_without_instant": src.dropped,
                         "instant_source": src.instant_source,
                         "missing_years": src.missing_years}
        rep["grid"] = src.grid.describe()
        if legacy is not None and Path(legacy).is_file():
            import xarray as xr

            with xr.open_dataset(legacy, mask_and_scale=False) as lg:
                lgrid = grid_of(lg)
            wins = {tuple(e["window"]) for e in extracts.values() if e.get("window")}
            if len(wins) == 1:
                r0, r1, c0, c1 = next(iter(wins))
                lgrid = Grid(lgrid.xs[c0:c1], lgrid.ys[r0:r1], lgrid.crs_wkt)
                rep["grid"]["legacy_window"] = [r0, r1, c0, c1]
            same = lgrid.same_as(src.grid)
            rep["grid"]["legacy_equal"] = bool(same)
            if not same:
                rep["problems"].append("the Zarr grid differs from the legacy netCDF grid")
        if rtk_grid is not None and Path(rtk_grid).is_file():
            same = grid_from_geotiff(rtk_grid).same_as(src.grid)
            rep["grid"]["rtk_equal"] = bool(same)
            if not same:
                rep["problems"].append("the Zarr grid differs from rtk.DEFAULT_GRID")
        rep["lattice20"] = src.lattice_report
        if not rep["lattice20"] or not rep["lattice20"].get("exact"):
            rep["warnings"].append("lattice20 is None: 20 m indices use box means at (0, 0)")
        rep["fingerprints"] = src.fingerprint(full=full_fingerprint)
        stats = scan_source(src) if scan else {}
        rep["scenes"] = stats
        if stats:
            scl11 = {}
            for d, e in stats.items():
                scl11[d[:4]] = scl11.get(d[:4], 0) + e["scl"][11]
            rep["scl11_per_year"] = scl11
            bad_cld = [d for d, e in stats.items() if e["cld_max"] is not None and e["cld_max"] > 100]
            if bad_cld:
                rep["problems"].append(f"CLD above 100 on {bad_cld[:5]}")
        off_stats = {}
        for y, a in src._attrs.items():
            per, _, _ = radiometry_stats(a)
            for d, st in per.items():
                off_stats[d] = dict(st, n_px=src.grid.H * src.grid.W)
        for d, e in stats.items():
            off_stats[d] = {k: e[k] for k in ("n_px", "n_valid", "n_low", "n_water",
                                              "n_water_low", "n_neg") if k in e}
        tab = decide_offsets({d: off_stats.get(d, {}) for d in src.dates}, src.baselines(),
                             src.year_states())
        rep["offsets"] = {d: (None if np.isnan(r.offset) else int(r.offset))
                          for d, r in tab.iterrows()}
        rep["offset_decisions"] = {d: {"state": r.state, "reason": r.reason,
                                       "strong": bool(r.strong)} for d, r in tab.iterrows()}
        unclear = [d for d, r in tab.iterrows() if r.state == "unclear"]
        if unclear:
            rep["problems"].append(f"BOA offset unclear on {len(unclear)} scenes: {unclear[:5]}")
        rep["offset_counts"] = tab["state"].value_counts().to_dict()
        gz = {}
        for y in years:
            p = zc.sidecar_path(site, y, "verify.json", root)
            if p.is_file():
                with open(p, encoding="utf-8") as f:
                    v = json.load(f)
                gz[str(y)] = {"verdict": v.get("verdict"), "dates": v.get("dates_with_differences")}
            else:
                gz[str(y)] = {"verdict": "no verify.json"}
        rep["g_zarr"] = gz
        meta = src.scene_meta
        rep["scene_meta"] = {"present": meta is not None}
        if meta:
            dm = meta.get("dates", {})
            rep["scene_meta"].update(
                missing_dates=[d for d in src.dates if d not in dm],
                double_overpass_dates=sorted(d for d, v in dm.items() if v.get("double_overpass")),
                baselines_per_year=_per_year(dm, "processing_baselines"),
                platforms_per_year=_per_year(dm, "platforms"),
                relative_orbits_per_year=_per_year(dm, "relative_orbits"))
        else:
            rep["warnings"].append("no scene_meta.json: baselines from the stores' attributes")
        if disk_paths:
            rep["disk"] = disk_check(disk_paths, read_from=zc.site_dir(site, root))
        rep["cache"] = src.cache.stats()
    finally:
        src.close()
    rep["ok"] = not rep["problems"]
    rep["seconds"] = round(time.time() - t0, 1)
    if out is not None:
        _merge_report(out, site, rep)
    return rep


class ReportLock:
    """An exclusive lock file ``<path>.lock`` (O_CREAT | O_EXCL) around the
    read-merge-write of the shared inputs report, so two ``inputs``
    processes never drop each other's site. A lock left by a dead process
    of this host is taken over; another holder is waited for up to
    ``timeout`` seconds."""

    def __init__(self, path, timeout=600.0):
        self.path = f"{path}.lock"
        self.timeout = float(timeout)

    def _stale(self):
        import socket
        try:
            with open(self.path, encoding="utf-8") as f:
                info = json.load(f)
            if info.get("host") != socket.gethostname():
                return False
            import psutil
            return not psutil.pid_exists(int(info["pid"]))
        except Exception:                        # unreadable or no psutil: not stale
            return False

    def __enter__(self):
        import socket
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    os.remove(self.path)
                    continue
                if time.time() - t0 > self.timeout:
                    raise InputsError(f"{self.path} is held by another process (delete it if "
                                      "no inputs command is running)")
                time.sleep(0.5)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"host": socket.gethostname(), "pid": os.getpid()}, f)
            return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def _merge_report(out, site, rep):
    """Write one site's report into the shared report file, under
    :class:`ReportLock` (read, merge, write)."""
    with ReportLock(out):
        doc = {"format": INPUTS_REPORT_FORMAT, "sites": {}}
        if Path(out).is_file():
            with open(out, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("format") == INPUTS_REPORT_FORMAT:
                doc = old
        doc["sites"][site] = rep
        _write_json(out, doc)


def legacy_inputs_report(site, cube, labels_path, res_m, instants=CATALOGUE, scan=True,
                         full_fingerprint=False, out=None, cache_mb=SCAN_CACHE_MB):
    """The data checks of a LEGACY netCDF cube (B03, B08, SCL), the source of
    the local runs on the 20 m cubes (``--res 20``; never reported numbers).

    The reader contract (int16, fill -32768, no scale or offset attributes:
    :class:`NetCDFSource` refuses otherwise), the record (catalogue times by
    default), the grid and its pixel size, the netCDF fingerprint, and with
    ``scan`` the SCL histogram of every scene (codes outside 0..11 refused).
    The legacy values are offset-harmonised: every scene's offset is 0.
    The 20 m sub-lattice does not exist on a 20 m grid (``lattice20`` None).

    Returns the report (written to ``out`` merged by site, if given).
    """
    t0 = time.time()
    rep = {"format": INPUTS_REPORT_FORMAT, "site": site, "problems": [], "warnings": [],
           "source": {"kind": "netcdf", "path": str(cube).replace("\\", "/"),
                      "res_m": int(res_m), "bands": list(LEGACY_BANDS)},
           "offset_rule": "legacy cube: offset-harmonised values (DN - 1000 delivered as "
                          "BOA x 10000), o_s = 0 for every scene"}
    lat = None if int(res_m) >= 20 else "measure"
    try:
        src = NetCDFSource(cube, site, instants, labels_path=labels_path, lattice=lat,
                           cache_mb=cache_mb)
    except (InputsError, ValueError, FileNotFoundError) as e:
        rep["problems"].append(f"{type(e).__name__}: {e}")
        rep["ok"] = False
        if out is not None:
            _merge_report(out, site, rep)
        return rep
    try:
        rep["instants"] = _instants_doc(src)
        rep["record"] = {"T": src.T, "dropped_without_instant": src.dropped,
                         "instant_source": src.instant_source, "years": src.years,
                         "first": src.dates[0] if src.dates else None,
                         "last": src.dates[-1] if src.dates else None}
        g = src.grid.describe()
        rep["grid"] = g
        if g["dx"] is not None and abs(abs(g["dx"]) - float(res_m)) > 1e-6:
            rep["problems"].append(f"pixel size {g['dx']} m is not {res_m} m")
        rep["lattice20"] = src.lattice_report if lat else {
            "phase": None, "note": f"{res_m} m grid: no 20 m sub-lattice (indices per pixel)"}
        rep["fingerprints"] = {"netcdf": src.fingerprint(full=full_fingerprint)}
        stats = scan_source(src) if scan else {}
        rep["scenes"] = stats
        if stats:
            scl11 = {}
            for d, e in stats.items():
                scl11[d[:4]] = scl11.get(d[:4], 0) + e["scl"][11]
            rep["scl11_per_year"] = scl11
        tab = src.offset_table()
        rep["offsets"] = {d: int(r.offset) for d, r in tab.iterrows()}
        rep["offset_counts"] = tab["state"].value_counts().to_dict()
        rep["cache"] = src.cache.stats()
        if not src.T:
            rep["problems"].append("no scene has a catalogue time")
    finally:
        src.close()
    rep["warnings"].append(f"legacy {res_m} m netCDF source: local software checks only; the "
                           "reported numbers come from the 10 m Zarr stores")
    rep["ok"] = not rep["problems"]
    rep["seconds"] = round(time.time() - t0, 1)
    if out is not None:
        _merge_report(out, site, rep)
    return rep


def _per_year(dm, key):
    out = {}
    for d, v in dm.items():
        out.setdefault(d[:4], {})
        for x in v.get(key, []):
            out[d[:4]][str(x)] = out[d[:4]].get(str(x), 0) + 1
    return out


def frozen_offsets(report):
    """``{date: 0 or 1000}`` from an inputs report (one site's dict, or a
    file with ``sites``); raises :class:`OffsetUnclear` if any is None."""
    off = report.get("offsets", {})
    bad = [d for d, v in off.items() if v is None]
    if bad:
        raise OffsetUnclear(f"{report.get('site')}: offsets unclear on {bad[:5]}")
    return {d: int(v) for d, v in off.items()}
