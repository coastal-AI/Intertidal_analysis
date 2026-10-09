# -*- coding: utf-8 -*-
"""Zarr validation cubes: the store format, the reader and the checks.

The validation cubes of the five sites (villaviciosa, ferrol, escalda,
wadden, ems) are kept as ONE ZARR STORE PER SITE AND CALENDAR YEAR,
``cubes/<site>/<year>.zarr``, written by ``experiments.download_cube_zarr``.
The temporal window of a validation run is a setting (1, 2, 3, 5, 9 years
...), so a window must never require rewriting a store: :func:`open_cube`
opens the yearly stores it needs, lazily, concatenates them along time and
selects the window. Years can be added later (2023-2025 first, 2017-2022
afterwards) without touching the stores already there.

This module depends on numpy, pandas, xarray, zarr and numcodecs only (no
pyintertidal, no openEO), so a notebook can use it anywhere the stores are.

Layout of every store (``FORMAT_ID``)
-------------------------------------
* Zarr v2, consolidated metadata (``.zmetadata``). v2 because xarray 2025.1
  with zarr-python 3.2 reads it reliably (zarr-python 3.2 v3 stores trip
  xarray 2025.1) and because GDAL, R and older zarr read it as well.
* dims ``t, y, x``. ``t``: the acquisition date as openEO labels it
  (00:00 UTC; the overpass time is in ``overpass_times.json`` beside the
  stores). ``y`` descending, ``x`` ascending: pixel CENTRES in metres in the
  site's UTM CRS, 10 m apart, the grid of the existing netCDF cubes.
* One variable per band, ``B02 B03 B04 B08 B8A B11 B12 SCL CLD``, with the
  values EXACTLY as the backend delivered them (int16 digital numbers: 10000
  x reflectance for the spectral bands, class codes for SCL, cloud
  probability in % for CLD); ``_FillValue`` -32768 = no data. Nothing is ever
  rescaled; scale attributes a backend may attach are kept under
  ``delivered_*`` names so no reader applies them by accident.
* ``crs``: grid-mapping variable (``crs_wkt``, ``spatial_ref``,
  ``GeoTransform``), named as in the netCDF cubes, so
  ``pyintertidal.cube.grid_from_dataset`` reads these datasets too.
* chunks ``{t: 16, y: 256, x: 256}`` (2 MiB of int16), Blosc/Zstd level 5
  with byte shuffle. See ``CHUNKS`` for why.
* global attributes: provenance (openEO job id, backend, processing date,
  bands, resampling, process graph, delivered layout) and ``complete = 1``,
  written last: a store without it is not a finished store.

Reading
-------
>>> from experiments import zarr_cubes as zc
>>> ds = zc.open_cube("ems", "2023-01-01", "2026-01-01")          # 3 years
>>> ds = zc.open_cube("ems", "2025-06-01", "2025-07-01", bands=["B03", "B08", "SCL"])

``end`` is EXCLUSIVE, as in openEO and in the yearly windows of the
downloads ([YYYY-01-01, YYYY+1-01-01)). Data come back raw (int16 with the
fill in ``attrs['_FillValue']``) and lazy (dask); ``masked=True`` gives
float32 with NaN instead.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: repository root (this file lives in <root>/experiments/)
REPO_ROOT = Path(__file__).resolve().parents[1]
#: environment variable that moves the cubes root (default <repo>/cubes)
ENV_ROOT = "ZARR_CUBES_ROOT"

#: the five validation sites
SITES = ("villaviciosa", "ferrol", "escalda", "wadden", "ems")
#: bands of the validation cubes, in store order (spectral, then the masks)
BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL", "CLD")
#: native Sentinel-2 L2A resolution of each band (m); 20 m bands reach the
#: 10 m grid by nearest neighbour (see download_cube_zarr.RESAMPLING)
NATIVE_RES_M = {"B02": 10, "B03": 10, "B04": 10, "B08": 10,
                "B8A": 20, "B11": 20, "B12": 20, "SCL": 20, "CLD": 20}
#: no-data value of the int16 bands (the openEO backend's, kept as delivered)
FILL = -32768
#: SCL classes the production per-pixel mask keeps as clear (MAREA)
CLEAR_SCL = (4, 5, 6, 7)
#: identifier of the store layout described in the module docstring
FORMAT_ID = "pyintertidal-zarr-cube/1"

#: Chunk shape of every band. One chunk is 16 x 256 x 256 int16 = 2 MiB.
#: Reading cost of the two access patterns this format exists for, at the
#: largest site (ems, ~3206 x 3413 px, ~145-173 dates a year):
#:   one date, one band (maps, RGB): 16 x the image = ~350 MB decoded
#:     ({t: 32, y: 512, x: 512} would read 700 MB);
#:   one pixel, 9 years (time series): ~1300 dates x 256 x 256 = ~170 MB
#:     decoded ({t: 32, y: 512, x: 512} would read 680 MB).
#: A streaming pass over whole images in time blocks (the pipeline's
#: pattern) reads every chunk once when its block is a multiple of 16; a
#: pass over spatial tiles with all dates reads every chunk once when tiles
#: are multiples of 256. Measured on the Ferrol 10 m cube, Blosc/Zstd-5
#: compresses these chunks as well as 16 MiB ones (B03 1.80x, B08 1.76x, SCL
#: ~100x; netCDF zlib-6 gets 1.48x/1.46x), and a site-year stays at a few
#: thousand files. Smaller chunks would multiply files and dask tasks for no
#: gain in compression.
CHUNKS = {"t": 16, "y": 256, "x": 256}
#: Blosc/Zstd level 5, byte shuffle: best of {zlib 6, zstd 3, blosc-zstd
#: 3/5/7 x shuffle/bitshuffle/none, blosc-lz4} on real B03/B08/SCL blocks;
#: level 7 adds <1 % on reflectance at twice the time
COMPRESSOR = {"cname": "zstd", "clevel": 5, "shuffle": "shuffle"}

#: SCL class codes -> (name, colour) for maps and legends
SCL_CLASSES = {
    0: ("no data", "#000000"), 1: ("saturated / defective", "#ff0000"),
    2: ("dark area pixels", "#2f2f2f"), 3: ("cloud shadows", "#643200"),
    4: ("vegetation", "#00a000"), 5: ("not vegetated", "#ffe65a"),
    6: ("water", "#0000ff"), 7: ("unclassified", "#808080"),
    8: ("cloud medium prob.", "#c0c0c0"), 9: ("cloud high prob.", "#ffffff"),
    10: ("thin cirrus", "#64c8ff"), 11: ("snow / ice", "#ff96ff"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  paths
# ─────────────────────────────────────────────────────────────────────────────

def cubes_root(root=None):
    """The folder holding ``<site>/<year>.zarr``.

    ``root`` (or ``$ZARR_CUBES_ROOT``) when given, else ``<repo>/cubes``.
    A relative path is taken from the repository root, so a notebook in a
    sub-folder and a script run from the root see the same stores.
    """
    p = Path(root) if root else Path(os.environ.get(ENV_ROOT) or REPO_ROOT / "cubes")
    return p if p.is_absolute() else REPO_ROOT / p


def site_dir(site, root=None):
    return cubes_root(root) / str(site)


def store_path(site, year, root=None):
    """``<root>/<site>/<year>.zarr``."""
    return site_dir(site, root) / f"{int(year)}.zarr"


def sidecar_path(site, year, kind, root=None):
    """Files kept beside a store: ``<year>.verify.json`` (comparison with the
    netCDF cube), ``<year>.format.txt`` (what the backend delivered),
    ``<year>.job.json`` (the openEO job and its result metadata)."""
    return site_dir(site, root) / f"{int(year)}.{kind}"


def overpass_path(site, root=None):
    """``<root>/<site>/overpass_times.json``: ``{date: ISO UTC time}`` of
    every scene of the stores (same format as the pipeline's files)."""
    return site_dir(site, root) / "overpass_times.json"


def year_window(year):
    """The download window of a year, end-exclusive like openEO:
    ``("YYYY-01-01", "YYYY+1-01-01")``."""
    y = int(year)
    return f"{y:04d}-01-01", f"{y + 1:04d}-01-01"


def parse_date(value):
    """A window bound as a Timestamp: ``'YYYY-MM-DD'``, a Timestamp, or an
    int year (``2023`` -> 2023-01-01)."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return pd.Timestamp(f"{int(value):04d}-01-01")
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


# ─────────────────────────────────────────────────────────────────────────────
#  store inventory
# ─────────────────────────────────────────────────────────────────────────────

def _consolidated(path):
    p = Path(path) / ".zmetadata"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def store_attrs(path):
    """Global attributes of a store, from its consolidated metadata (cheap)."""
    meta = _consolidated(path)
    if not meta:
        return {}
    return dict(meta.get("metadata", {}).get(".zattrs", {}))


def is_complete(path):
    """True for a finished store: consolidated, this layout, ``complete = 1``.

    The downloader writes a store under ``<year>.zarr.tmp`` and renames it
    only after every block was read back and compared, so a store that is
    there is normally complete; this also rejects a copy cut short.
    """
    attrs = store_attrs(path)
    return attrs.get("complete") == 1 and attrs.get("format") == FORMAT_ID


def list_years(site, root=None):
    """Years with a complete store for ``site``, sorted."""
    d = site_dir(site, root)
    if not d.is_dir():
        return []
    years = []
    for p in d.iterdir():
        m = re.fullmatch(r"(\d{4})\.zarr", p.name)
        if m and p.is_dir() and is_complete(p):
            years.append(int(m.group(1)))
    return sorted(years)


def available(root=None):
    """``{site: [years]}`` of every complete store under the root."""
    base = cubes_root(root)
    if not base.is_dir():
        return {}
    out = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and not d.name.startswith((".", "_")):
            years = list_years(d.name, root)
            if years:
                out[d.name] = years
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  encoding (used by the writer; defined here so reader and writer agree)
# ─────────────────────────────────────────────────────────────────────────────

def compressor():
    """The numcodecs compressor of every band (see ``COMPRESSOR``)."""
    from numcodecs import Blosc

    shuffle = {"shuffle": Blosc.SHUFFLE, "bitshuffle": Blosc.BITSHUFFLE,
               "noshuffle": Blosc.NOSHUFFLE}[COMPRESSOR["shuffle"]]
    return Blosc(cname=COMPRESSOR["cname"], clevel=COMPRESSOR["clevel"], shuffle=shuffle)


def chunk_shape(shape, chunks=None):
    """``CHUNKS`` clipped to an array of ``shape`` (t, y, x)."""
    c = chunks or CHUNKS
    return tuple(int(max(1, min(c[d], n))) for d, n in zip(("t", "y", "x"), shape))


def encoding(ds, bands, fills, chunks=None):
    """``to_zarr`` encoding of a canonical dataset: band chunks, compressor,
    fill (``fills[band]``, None for none) and dtype; the time axis as integer
    days when every label is at 00:00 (as in the netCDF cubes), else integer
    seconds. The fill goes in the encoding only: xarray refuses a
    ``_FillValue`` in both the attributes and the encoding."""
    enc = {}
    for b in bands:
        da_ = ds[b]
        e = {"chunks": chunk_shape(da_.shape, chunks), "compressor": compressor(),
             "dtype": da_.dtype}
        fill = fills.get(b)
        if fill is not None:
            e["_FillValue"] = np.asarray(fill, dtype=da_.dtype)[()]
        enc[b] = e
    t = pd.DatetimeIndex(ds["t"].values)
    if len(t) and bool((t == t.normalize()).all()):
        enc["t"] = {"units": "days since 1990-01-01", "calendar": "proleptic_gregorian",
                    "dtype": "int32"}
    else:
        enc["t"] = {"units": "seconds since 1970-01-01", "calendar": "proleptic_gregorian",
                    "dtype": "int64"}
    return enc


# ─────────────────────────────────────────────────────────────────────────────
#  reading
# ─────────────────────────────────────────────────────────────────────────────

def open_store(path, bands=None, masked=False, chunks=None):
    """One yearly store, lazily (dask, on-disk chunks unless ``chunks``).

    ``masked=False`` (default): raw values, dtype as stored (int16), the fill
    in ``attrs['_FillValue']``. ``masked=True``: float32 with NaN for no data.
    """
    path = Path(path)
    if not (path / ".zmetadata").is_file():
        raise FileNotFoundError(f"{path}: not a consolidated Zarr store (missing .zmetadata)")
    ds = xr.open_zarr(path, consolidated=True, chunks={} if chunks is None else chunks,
                      mask_and_scale=bool(masked), decode_times=True)
    if bands is not None:
        bands = [bands] if isinstance(bands, str) else list(bands)
        missing = [b for b in bands if b not in ds.data_vars]
        if missing:
            raise KeyError(f"{path}: bands {missing} not in the store; it has "
                           f"{[v for v in ds.data_vars if v != 'crs']}")
    else:
        bands = band_names(ds)            # the store lists them alphabetically
    keep = bands + (["crs"] if "crs" in ds.data_vars else [])
    return ds[keep]


def band_names(ds):
    """The band variables of a cube dataset (everything but ``crs``), in the
    order of ``BANDS`` (others after, alphabetically)."""
    names = [v for v in ds.data_vars if v != "crs" and set(ds[v].dims) >= {"y", "x"}]
    rank = {b: i for i, b in enumerate(BANDS)}
    return sorted(names, key=lambda b: (rank.get(b, len(rank)), str(b)))


def _same_grid(a, b):
    if a.sizes.get("y") != b.sizes.get("y") or a.sizes.get("x") != b.sizes.get("x"):
        return False
    if not (np.array_equal(a["x"].values, b["x"].values)
            and np.array_equal(a["y"].values, b["y"].values)):
        return False
    wa = a["crs"].attrs.get("crs_wkt") if "crs" in a.variables else None
    wb = b["crs"].attrs.get("crs_wkt") if "crs" in b.variables else None
    return crs_equal(wa, wb)


def open_cube(site, start, end=None, bands=None, root=None, masked=False, chunks=None,
              missing="raise"):
    """The cube of ``site`` over ``[start, end)``, lazily, across years.

    Opens only the yearly stores that overlap the window, checks they share
    the grid, concatenates them along ``t`` and selects the window: any
    window is a cheap selection and nothing is read until values are asked
    for. ``end`` is exclusive (``None``: up to the last stored date).
    ``missing="raise"`` refuses a window that needs a year not downloaded;
    ``"skip"`` returns what is there (and says which years were missing in
    ``attrs['missing_years']``).
    """
    t0 = parse_date(start)
    t1 = parse_date(end) if end is not None else None
    if t1 is not None and t1 <= t0:
        raise ValueError(f"empty window [{t0.date()}, {t1.date()})")
    have = list_years(site, root)
    if not have:
        raise FileNotFoundError(f"no complete Zarr store for {site!r} under {site_dir(site, root)}")
    last = (t1 - pd.Timedelta(1, "ns")).year if t1 is not None else max(have)
    wanted = list(range(t0.year, last + 1))
    absent = [y for y in wanted if y not in have]
    if absent and missing == "raise":
        raise FileNotFoundError(
            f"{site}: the window [{t0.date()}, {t1.date() if t1 is not None else 'end'}) needs "
            f"years {absent}, not downloaded (complete stores: {have}); download them with "
            f"`python -m experiments.download_cube_zarr {site} --years "
            f"{','.join(str(y) for y in absent)}` or pass missing='skip'")
    years = [y for y in wanted if y in have]
    if not years:
        raise FileNotFoundError(f"{site}: no stored year overlaps the window (have {have})")
    parts = [open_store(store_path(site, y, root), bands=bands, masked=masked, chunks=chunks)
             for y in years]
    for y, p in zip(years[1:], parts[1:]):
        if not _same_grid(parts[0], p):
            raise ValueError(f"{site}: the {y} store is not on the grid of the {years[0]} store")
    sources = [{"year": y, "store": str(store_path(site, y, root)),
                "job_id": p.attrs.get("job_id"), "processing_date": p.attrs.get("processing_date")}
               for y, p in zip(years, parts)]
    if len(parts) == 1:
        ds = parts[0]
    else:
        ds = xr.concat(parts, dim="t", data_vars="minimal", coords="minimal",
                       compat="override", join="exact", combine_attrs="override")
    t = ds["t"].values
    keep = t >= np.datetime64(t0)
    if t1 is not None:
        keep &= t < np.datetime64(t1)
    ds = ds.isel(t=np.flatnonzero(keep))
    tt = ds["t"].values
    if len(tt) > 1 and not np.all(np.diff(tt) > np.timedelta64(0, "ns")):
        raise ValueError(f"{site}: dates not strictly increasing across the yearly stores")
    attrs = {k: v for k, v in ds.attrs.items()
             if k in ("format", "collection", "backend", "bands", "resampling", "grid_epsg",
                      "Conventions")}
    attrs.update({"site": site, "window_start": str(t0.date()),
                  "window_end_exclusive": str(t1.date()) if t1 is not None else "",
                  "years": years, "missing_years": absent,
                  "sources": json.dumps(sources)})
    ds.attrs = attrs
    return ds


def dates(ds):
    """Acquisition dates of a cube dataset as ``'YYYY-MM-DD'`` strings."""
    return [str(d)[:10] for d in pd.DatetimeIndex(ds["t"].values).strftime("%Y-%m-%d")]


def fill_of(da_):
    """The no-data value of a raw band (None when it has none)."""
    f = da_.attrs.get("_FillValue", da_.encoding.get("_FillValue"))
    if f is None:
        return None
    f = np.asarray(f).item()
    return None if isinstance(f, float) and np.isnan(f) else f


def as_float(da_):
    """A raw band as float32 with NaN for no data (lazy)."""
    f = fill_of(da_)
    out = da_.astype("float32")
    if f is not None:
        out = out.where(da_ != f)
    out.attrs = {k: v for k, v in da_.attrs.items() if k != "_FillValue"}
    return out


def normalized_difference(a, b):
    """``(a - b) / (a + b)`` of two raw bands, float32, NaN where either has
    no data or the sum is 0 (e.g. NDWI = nd(B03, B08), MNDWI = nd(B03, B11))."""
    fa, fb = as_float(a), as_float(b)
    s = fa + fb
    return ((fa - fb) / s.where(s != 0)).astype("float32")


def clear_mask(scl, classes=CLEAR_SCL):
    """Boolean clear-sky mask from SCL (the production rule: 4, 5, 6, 7)."""
    return scl.isin(list(classes))


def read_overpass(site, root=None):
    """``{date: Timestamp (UTC, naive)}`` from the stores' overpass file."""
    p = overpass_path(site, root)
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: parse_date(v) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _epsg_from_wkt(wkt):
    if not wkt:
        return None
    try:
        from pyproj import CRS

        return CRS.from_user_input(wkt).to_epsg()
    except Exception:
        m = re.findall(r'AUTHORITY\["EPSG",\s*"(\d+)"\]', str(wkt))
        return int(m[-1]) if m else None


def crs_equal(a, b):
    """Same CRS? (pyproj when available, else EPSG codes, else the text)."""
    if not a or not b:
        return a == b
    if a == b:
        return True
    try:
        from pyproj import CRS

        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:
        ea, eb = _epsg_from_wkt(a), _epsg_from_wkt(b)
        return ea is not None and ea == eb


def grid_info(ds):
    """Shape, spacing, CRS and GDAL-style transform of a cube dataset."""
    x = np.asarray(ds["x"].values, dtype="float64")
    y = np.asarray(ds["y"].values, dtype="float64")
    dx = float(x[1] - x[0]) if len(x) > 1 else float("nan")
    dy = float(y[1] - y[0]) if len(y) > 1 else float("nan")
    wkt = None
    for v in ("crs", "spatial_ref"):
        if v in ds.variables:
            wkt = ds[v].attrs.get("crs_wkt") or ds[v].attrs.get("spatial_ref")
            if wkt:
                break
    return {"shape": [int(len(y)), int(len(x))], "dx": dx, "dy": dy,
            "x_first": float(x[0]) if len(x) else None, "x_last": float(x[-1]) if len(x) else None,
            "y_first": float(y[0]) if len(y) else None, "y_last": float(y[-1]) if len(y) else None,
            "epsg": _epsg_from_wkt(wkt), "crs_wkt": wkt,
            "transform": [float(x[0] - dx / 2), dx, 0.0, float(y[0] - dy / 2), 0.0, dy]
            if len(x) > 1 and len(y) > 1 else None}


def _overlap(a, b):
    """Index ranges of the common coordinate values of two regular axes.

    Returns ``(a0, a1, b0, b1)`` (end-exclusive) or None. Exact matching to
    1 mm, since both grids are pixel centres of the same 10 m lattice.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    ia = np.flatnonzero(np.isclose(a[:, None], b[None, :], rtol=0, atol=1e-3).any(axis=1)) \
        if a.size * b.size <= 25_000_000 else None
    if ia is None:                       # very large axes: sorted search
        order = np.argsort(b)
        pos = np.clip(np.searchsorted(b[order], a), 0, len(b) - 1)
        near = np.abs(b[order][pos] - a) < 1e-3
        ia = np.flatnonzero(near)
    if ia.size == 0:
        return None
    a0, a1 = int(ia[0]), int(ia[-1]) + 1
    if a1 - a0 != ia.size:
        return None
    jb = int(np.argmin(np.abs(b - a[a0])))
    b0, b1 = jb, jb + (a1 - a0)
    if b1 > len(b) or not np.allclose(a[a0:a1], b[b0:b1], rtol=0, atol=1e-3):
        return None
    return a0, a1, b0, b1


# ─────────────────────────────────────────────────────────────────────────────
#  verification against the netCDF cubes
# ─────────────────────────────────────────────────────────────────────────────

def _time_dim(ds):
    for d in ("t", "time"):
        if d in ds.dims:
            return d
    raise ValueError(f"no time dimension in {list(ds.dims)}")


def _dates_of(values):
    return [str(d) for d in pd.DatetimeIndex(values).strftime("%Y-%m-%d")]


def compare_with_netcdf(store, nc_path, bands=("B03", "B08", "SCL"), start=None, end=None,
                        ref_period=None, t_block=None, row_block=None, log=None,
                        max_dates_listed=60):
    """Compare a Zarr store with a netCDF cube of the same site, pixel by pixel.

    Parameters
    ----------
    store : path or Dataset
        A canonical store (or a dataset opened raw from one).
    nc_path : path
        The existing netCDF cube (int16 on disk, fill -32768).
    bands : bands compared (those in both).
    start, end : the window compared, end-exclusive (default: the store's
        ``temporal_extent`` attribute, else its first..last date).
    ref_period : ``(start, end)`` the netCDF cube was REQUESTED for,
        end-exclusive. Dates of the store outside it are listed apart
        (``outside_reference_period``) instead of as missing from the
        reference: a 2025 store holds 2025-12-31, which a cube requested up
        to 2025-12-31 (exclusive) cannot have.

    Returns a JSON-able dict: grid check (CRS, identical axes or the common
    window and any half-pixel offset), the date sets (common, only in the
    store, only in the reference, outside the reference period) and, per
    band over the common dates and pixels, the counts of equal values,
    differing values, no-data in one only, the maximum absolute difference
    and the share of differing pixels, plus the dates where something
    differs. ``verdict``: ``identical``, ``differs`` or ``not comparable``.
    """
    say = log or (lambda msg: None)
    t_start = time.time()
    z = open_store(store) if isinstance(store, (str, Path)) else store
    r = xr.open_dataset(nc_path, mask_and_scale=False, decode_times=True)
    try:
        rt = _time_dim(r)
        if rt != "t":
            r = r.rename({rt: "t"})
        zt = pd.DatetimeIndex(z["t"].values)
        rtt = pd.DatetimeIndex(r["t"].values)
        if start is None or end is None:
            ext = z.attrs.get("temporal_extent")
            if isinstance(ext, str):
                try:
                    ext = json.loads(ext)
                except ValueError:
                    ext = None
            if ext and len(ext) == 2:
                start = start or ext[0]
                end = end or ext[1]
        w0 = parse_date(start) if start is not None else (zt.min() if len(zt) else None)
        w1 = parse_date(end) if end is not None else (
            zt.max() + pd.Timedelta(days=1) if len(zt) else None)
        if ref_period is not None:
            p0, p1 = parse_date(ref_period[0]), parse_date(ref_period[1])
        else:
            p0 = rtt.min() if len(rtt) else w0
            p1 = rtt.max() + pd.Timedelta(days=1) if len(rtt) else w1
        zsel = (zt >= w0) & (zt < w1) if w0 is not None else np.zeros(len(zt), bool)
        rsel = (rtt >= max(w0, p0)) & (rtt < min(w1, p1)) if w0 is not None else np.zeros(len(rtt), bool)
        zd = {d: i for d, i in zip(_dates_of(zt), range(len(zt))) if zsel[i]}
        rd = {d: i for d, i in zip(_dates_of(rtt), range(len(rtt))) if rsel[i]}
        in_ref = {d for d in zd if p0 <= pd.Timestamp(d) < p1}
        common = sorted(set(zd) & set(rd))
        rep = {
            "store": str(store) if isinstance(store, (str, Path)) else z.attrs.get("store", ""),
            "reference": str(nc_path),
            "window": [str(w0.date()) if w0 is not None else None,
                       str(w1.date()) if w1 is not None else None],
            "reference_period": [str(p0.date()), str(p1.date())],
            "dates": {
                "store": len(zd), "reference": len(rd), "common": len(common),
                "only_store": sorted(in_ref - set(rd)),
                "only_reference": sorted(set(rd) - set(zd)),
                "outside_reference_period": sorted(set(zd) - in_ref),
            },
        }
        dup_z = len(set(_dates_of(zt))) != len(zt)
        dup_r = len(set(_dates_of(rtt))) != len(rtt)
        if dup_z or dup_r:
            rep["dates"]["duplicated_dates"] = {"store": dup_z, "reference": dup_r}

        # ── grid ──────────────────────────────────────────────────────────
        gz, gr = grid_info(z), grid_info(r)
        grid = {"store": {k: gz[k] for k in ("shape", "dx", "dy", "epsg", "transform")},
                "reference": {k: gr[k] for k in ("shape", "dx", "dy", "epsg", "transform")},
                "crs_equal": crs_equal(gz["crs_wkt"], gr["crs_wkt"])}
        same_axes = (gz["shape"] == gr["shape"]
                     and np.array_equal(z["x"].values, r["x"].values)
                     and np.array_equal(z["y"].values, r["y"].values))
        grid["identical"] = bool(same_axes and grid["crs_equal"])
        ox = _overlap(z["x"].values, r["x"].values)
        oy = _overlap(z["y"].values, r["y"].values)
        if ox is None or oy is None:
            # not on one lattice (or disjoint): the fractional offset in
            # pixels, in (-0.5, 0.5]; 0.5 means pixel corners against centres
            for name, a, b, d in (("x", z["x"].values, r["x"].values, gz["dx"]),
                                  ("y", z["y"].values, r["y"].values, gz["dy"])):
                if len(a) and len(b) and d and np.isfinite(d):
                    frac = ((float(a[0]) - float(b[0])) / abs(d)) % 1.0
                    grid[f"{name}_offset_px"] = round(frac if frac <= 0.5 else frac - 1.0, 4)
            grid["common_window"] = None
        else:
            grid["common_window"] = {"store_rows": [oy[0], oy[1]], "store_cols": [ox[0], ox[1]],
                                     "reference_rows": [oy[2], oy[3]],
                                     "reference_cols": [ox[2], ox[3]]}
        rep["grid"] = grid

        cmp_bands = [b for b in bands if b in z.data_vars and b in r.data_vars]
        rep["bands_compared"] = cmp_bands
        rep["bands_not_in_reference"] = [b for b in band_names(z) if b not in r.data_vars]
        if not common or grid["common_window"] is None or not cmp_bands or not grid["crs_equal"]:
            why = ("no common dates" if not common else
                   "no common grid window" if grid["common_window"] is None else
                   "different CRS" if not grid["crs_equal"] else "no common bands")
            rep["verdict"] = "not comparable"
            rep["reason"] = why
            rep["seconds"] = round(time.time() - t_start, 1)
            return rep

        cw = grid["common_window"]
        zy0, zy1 = cw["store_rows"]
        zx0, zx1 = cw["store_cols"]
        ry0, _ = cw["reference_rows"]
        rx0, rx1 = cw["reference_cols"]
        H = zy1 - zy0
        tb = int(t_block or CHUNKS["t"])
        rb = int(row_block or CHUNKS["y"])
        stats = {b: {"pixels": 0, "equal": 0, "value_differs": 0, "nodata_only_store": 0,
                     "nodata_only_reference": 0, "nodata_both": 0, "max_abs_diff": 0.0}
                 for b in cmp_bands}
        per_date = {}
        zfill = {b: fill_of(z[b]) for b in cmp_bands}
        rfill = {b: fill_of(r[b]) for b in cmp_bands}
        zi_all = [zd[d] for d in common]
        ri_all = [rd[d] for d in common]
        for k in range(0, len(common), tb):
            zi = zi_all[k:k + tb]
            ri = ri_all[k:k + tb]
            for row in range(0, H, rb):
                n = min(rb, H - row)
                for b in cmp_bands:
                    a = np.asarray(z[b].isel(t=zi, y=slice(zy0 + row, zy0 + row + n),
                                             x=slice(zx0, zx1)).values)
                    c = np.asarray(r[b].isel(t=ri, y=slice(ry0 + row, ry0 + row + n),
                                             x=slice(rx0, rx1)).values)
                    fa = np.isnan(a) if a.dtype.kind == "f" else np.zeros(a.shape, bool)
                    fc = np.isnan(c) if c.dtype.kind == "f" else np.zeros(c.shape, bool)
                    if zfill[b] is not None:
                        fa |= a == zfill[b]
                    if rfill[b] is not None:
                        fc |= c == rfill[b]
                    valid = ~fa & ~fc
                    diff = valid & (a != c)
                    s = stats[b]
                    s["pixels"] += a.size
                    s["nodata_both"] += int((fa & fc).sum())
                    s["nodata_only_store"] += int((fa & ~fc).sum())
                    s["nodata_only_reference"] += int((fc & ~fa).sum())
                    nd = int(diff.sum())
                    s["value_differs"] += nd
                    s["equal"] += int((valid & ~diff).sum())
                    if nd:
                        d_ = np.abs(a[diff].astype("float64") - c[diff].astype("float64"))
                        s["max_abs_diff"] = max(s["max_abs_diff"], float(d_.max()))
                    bad = (diff | (fa ^ fc)).sum(axis=(1, 2))
                    for j, cnt in enumerate(bad):
                        if cnt:
                            dd = common[k + j]
                            per_date.setdefault(dd, {}).setdefault(b, 0)
                            per_date[dd][b] += int(cnt)
            say(f"compared {min(k + tb, len(common))}/{len(common)} common dates")
        for b, s in stats.items():
            bad = s["value_differs"] + s["nodata_only_store"] + s["nodata_only_reference"]
            s["differing_pixels"] = bad
            s["share_differing"] = bad / s["pixels"] if s["pixels"] else None
            valid = s["equal"] + s["value_differs"]
            s["share_value_differs_of_valid"] = s["value_differs"] / valid if valid else None
        rep["bands"] = stats
        worst = sorted(per_date.items(), key=lambda kv: -sum(kv[1].values()))
        rep["dates_with_differences"] = len(per_date)
        rep["dates_with_differences_listed"] = dict(worst[:max_dates_listed])
        clean = (not any(s["differing_pixels"] for s in stats.values())
                 and not rep["dates"]["only_store"] and not rep["dates"]["only_reference"])
        if clean and grid["identical"]:
            rep["verdict"] = "identical"
        elif clean:
            rep["verdict"] = "identical on the common window"
            rep["note"] = ("same dates and values wherever both grids have pixels; the grids "
                           "differ in extent (expected for the --test window)")
        else:
            rep["verdict"] = "differs"
        rep["seconds"] = round(time.time() - t_start, 1)
        return rep
    finally:
        r.close()


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_json(obj, path):
    """Write ``obj`` as indented JSON, atomically (numpy types allowed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_jsonable, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def report_lines(rep):
    """A few log lines summarising :func:`compare_with_netcdf`'s report."""
    out = [f"verification vs {Path(rep.get('reference', '?')).name}: {rep.get('verdict', '?').upper()}"
           + (f" ({rep['reason']})" if rep.get("reason") else "")]
    d = rep.get("dates", {})
    out.append(f"  dates: store {d.get('store')}, reference {d.get('reference')}, common "
               f"{d.get('common')}; only in store {d.get('only_store', [])[:10]}"
               f"{' ...' if len(d.get('only_store', [])) > 10 else ''}; only in reference "
               f"{d.get('only_reference', [])[:10]}"
               f"{' ...' if len(d.get('only_reference', [])) > 10 else ''}; outside the "
               f"reference period {d.get('outside_reference_period', [])}")
    g = rep.get("grid", {})
    if g:
        out.append(f"  grid: identical={g.get('identical')} crs_equal={g.get('crs_equal')} "
                   f"store {g.get('store', {}).get('shape')} reference "
                   f"{g.get('reference', {}).get('shape')} common window {g.get('common_window')}"
                   + (f" offsets(px) x={g.get('x_offset_px')} y={g.get('y_offset_px')}"
                      if "x_offset_px" in g else ""))
    for b, s in (rep.get("bands") or {}).items():
        share = s.get("share_differing")
        out.append(f"  {b}: {s['differing_pixels']} of {s['pixels']} pixel-dates differ "
                   f"({(share or 0) * 100:.4f} %): values {s['value_differs']} (max |diff| "
                   f"{s['max_abs_diff']:g}), no data only in store {s['nodata_only_store']}, "
                   f"only in reference {s['nodata_only_reference']}")
    if rep.get("dates_with_differences"):
        first = list(rep.get("dates_with_differences_listed", {}).items())[:5]
        out.append(f"  dates with differences: {rep['dates_with_differences']} (worst {first})")
    return out
