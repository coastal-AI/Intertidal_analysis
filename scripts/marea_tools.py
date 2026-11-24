# -*- coding: utf-8 -*-
"""Helpers shared by the MAREA scripts and tutorials.

No code in the library builds the record, writes the emersion map, exports
the GeoTIFFs or checks a reproduction; this module does, by calling public
pyintertidal functions only. It adds no algorithm.

Who uses it: ``scripts/reproduce_villaviciosa.py``, ``replication/build_kit.py``,
notebooks 01 and 03, and ``scripts/quickstart.py``, whose
``map_estuary(cube_path, tide, out_dir, ...)`` is the one-call route for a
new estuary (it runs steps 2-11 with these helpers). Import it with
``sys.path.insert(0, "scripts"); import marea_tools as mt``; importing it puts
the repository root on ``sys.path``, so ``import pyintertidal`` then finds
this repository's package.

What is here, by step
---------------------
* (3) :func:`screen` — the reference map and the usable scenes.
* (5) :func:`build_record` (with :func:`record_rows`) — the record.
* (6) :func:`site_boundary`, :class:`TableBoundary` (the replication kit's
  tabulated tide), :class:`TideSeries` (a series, regular or not).
* (11) :func:`emersion_time`, :func:`export_geotiffs`.
* Checks: :func:`compare_products`, :func:`overlap_summary`,
  :func:`qa_table`, :func:`coverage`; environment and provenance:
  :func:`check_env`, :func:`provenance`.
* Files: :func:`load_site` (a site YAML file), :func:`load_overpass`,
  :func:`load_extraction` / :func:`save_extraction`, :func:`write_npz`,
  :func:`write_json`.

The 11 processing steps (the same numbers and names in the guide, the
scripts and the code comments of notebook 03; the headings of the notebooks
number their own sections):

 (1) Download the Sentinel-2 cube
 (2) Get the acquisition time of every scene
 (3) Screen the scenes
 (4) Extract the looks and the intertidal mask
 (5) Build the record
 (6) Read the tide at the mouth
 (7) Find the mouth, the distance along the water and the 3-km sections
 (8) Fit one delay per section, with its interval
 (9) Give every pixel its delay
 (10) Fit each pixel's elevation and width on its delayed tide
 (11) Make the products

Conventions: times are UTC (the datatake start); heights are metres on the
scale of the tide at the mouth; the delay tau is in minutes, positive when
the water inside is later than at the mouth; sections are numbered from 0
at the mouth (in the code and the products a section is a "band"); a look
is wet when NDWI > 0.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

#: the repository root (the folder above scripts/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # so that ``import pyintertidal`` finds this repository's package,
    # wherever the calling script or notebook runs from
    sys.path.insert(0, str(ROOT))

#: the 11 processing steps, verbatim
STEPS = (
    "Download the Sentinel-2 cube: B03, B08, SCL at 10 m",
    "Get the acquisition time of every scene",
    "Screen the scenes",
    "Extract the looks and the intertidal mask",
    "Build the record",
    "Read the tide at the mouth",
    "Find the mouth, the distance along the water and the 3-km sections",
    "Fit one delay per section, with its interval",
    "Give every pixel its delay",
    "Fit each pixel's elevation and width on its delayed tide",
    "Make the products: hypsometry, GeoTIFFs, emersion time",
)

#: folders no script may write into (the published products)
PROTECTED = (ROOT / "products_marea", ROOT / "products_villaviciosa")

#: packages reported by check_env: (distribution name, role)
PACKAGES = (
    ("numpy", "core"), ("scipy", "core"), ("pandas", "core"),
    ("xarray", "core"), ("pyTMD", "tide models"), ("eo-tides", "tide models"),
    ("scikit-image", "core"), ("PyYAML", "core"), ("pyproj", "core"),
    ("netCDF4", "core"), ("h5netcdf", "core"), ("joblib", "core"),
    ("rasterio", "core"), ("shapely", "core"), ("matplotlib", "core"),
    ("requests", "core"), ("openeo", "download"),
    ("nbformat", "tutorials"), ("nbclient", "tutorials"),
    ("ipykernel", "tutorials"),
)
#: the full pinned environment
PINNED_FILE = ROOT / "requirements.txt"
#: the Python version the product was made and checked with
PYTHON_PINNED = "3.13.3"

#: file name of each layer in export_geotiffs, when it differs from the layer
LAYER_FILE = {"tau_px_min": "tau_px"}
#: layers written as int16 with -1 as no-data (the others: float32, NaN)
INT_LAYERS = ("band", "n_clear")


def step(k):
    """Heading of step ``k`` (1-11): ``'(k) <name>'``."""
    return f"({k}) {STEPS[k - 1]}"


# ─────────────────────────────────────────────────────────────────────────────
#  Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def sha256(path):
    """SHA-256 of a file's bytes (hex), read in 1 MiB chunks.

    The same digest as ``pyintertidal.seal._sha256``, ``sha256sum`` and
    ``certutil -hashfile <file> SHA256``.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    """``path`` relative to the repository root when it is inside it."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def short_path(path):
    """``path`` relative to the working directory when inside it, else absolute.

    Passed as ``out_dir`` to ``marea.reconstruct``, which prints it, so the
    console shows ``runs/<site>`` and not the full local path.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(p)


def resolve(path):
    """A path from a site file: absolute as given, else from the repo root.

    ``~`` is expanded; None gives None. Returns a ``Path``. :func:`load_site`
    leaves the paths of a site file as written, so pass them through this.
    """
    if path is None:
        return None
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else (ROOT / p)


def guard_out_dir(out_dir):
    """Refuse an output folder inside products_marea/ or products_villaviciosa/.

    Returns ``out_dir`` as an absolute ``Path`` (not created); exits with
    ``SystemExit`` and a message for a folder inside a published product.
    """
    out = Path(out_dir).resolve()
    for p in PROTECTED:
        if out == p or p in out.parents:
            raise SystemExit(f"refusing to write into {rel(p)}/: the published "
                             f"product is read-only; choose a folder under runs/")
    return out


def format_table(rows, columns):
    """Rows (sequences) as a fixed-width text table with a header line.

    ``rows`` are sequences of values (None prints empty); ``columns`` the
    header names, one per value. Returns the table as one string.
    """
    rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    width = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(columns)]
    line = lambda r: "  ".join(v.ljust(w) for v, w in zip(r, width)).rstrip()
    out = [line(columns), line(["-" * w for w in width])]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _json_default(o):
    """What ``json.dump`` cannot write itself: numpy values, arrays, paths."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def write_json(path, obj):
    """JSON with UTF-8 and indentation, numpy values converted."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_json_default)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Site configuration
# ─────────────────────────────────────────────────────────────────────────────

#: every key of a site file and its default (None = required or unset).
#: In this edition only some keys are acted on: scripts/reproduce_villaviciosa.py
#: and notebook 01 read key, overpass_times, out_dir, tide.model, screening.*,
#: record.* and marea.* (except region_mask); site_boundary() also reads
#: polygon, bbox, period and the other tide.* keys, but no script calls it.
#: resolution, cube, per_year, job_options, marea.region_mask and every
#: export.* key are validated, so a site file keeps one format, but nothing
#: here acts on them (setting export.cog to true, for example, writes nothing).
SITE_DEFAULTS = {
    "site": None,
    "key": None,
    "polygon": None,
    "bbox": None,
    "period": None,
    "resolution": 10,
    "cube": None,
    "per_year": True,
    "job_options": None,
    "overpass_times": None,
    "tide": {"model": "EOT20", "directory": "tide_models", "point": "centroid",
             "n_gauges": None, "exclude": [], "gauge_csv": None,
             "cache_dir": None},
    "screening": {"threshold": 0.0, "bad_classes": [3, 8, 9, 10],
                  "clean_scene_max_bad": 0.05, "stable_threshold": 0.95,
                  "transition_buffer_px": 10, "cloud_threshold": 0.10},
    "record": {"epoch_years": 3, "which": "latest"},
    "marea": {"mouth_side": "any", "min_scenes": 40, "quality_min_bracket": 3,
              "band_km": 3.0, "min_px_band": 500, "n_boot": 30,
              "region_mask": None},
    "export": {"grid_reference": None, "emersion_step_min": 10,
               "emersion_period": None,
               "layers": ["z", "sigma", "tau_px_min", "s_km", "band",
                          "emersion", "n_clear"],
               "cog": False, "stac": False},
    "out_dir": None,
}
_SIDES = {"any", "north", "south", "east", "west"}


def load_site(yaml_path):
    """Read a site file, fill the defaults and validate it.

    Parameters
    ----------
    yaml_path : str or Path
        A file like ``scripts/sites/villaviciosa.yaml``.

    Returns
    -------
    dict
        Every key of :data:`SITE_DEFAULTS`, plus ``_yaml`` (the file read).
        Raises ``ValueError`` naming the key for any invalid or unknown key.

    Paths inside the file (``cube``, ``overpass_times``, ``out_dir``, ...)
    are returned AS WRITTEN, not resolved: they are meant relative to the
    repository root, so pass them through :func:`resolve` (code that runs
    from the repository root, as the notebooks do, can use them directly).
    ``period`` values come back as strings. See :data:`SITE_DEFAULTS` for
    which keys this edition acts on.
    """
    import pandas as pd
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ValueError(f"{yaml_path}: expected a mapping of keys")
    cfg = copy.deepcopy(SITE_DEFAULTS)
    for k, v in user.items():
        if k not in SITE_DEFAULTS:
            raise ValueError(f"{yaml_path}: unknown key {k!r}; known keys: "
                             f"{', '.join(SITE_DEFAULTS)}")
        if isinstance(SITE_DEFAULTS[k], dict):
            if v is None:
                continue
            if not isinstance(v, dict):
                raise ValueError(f"{yaml_path}: {k!r} must be a mapping")
            for kk, vv in v.items():
                if kk not in SITE_DEFAULTS[k]:
                    raise ValueError(f"{yaml_path}: unknown key {k}.{kk}; known: "
                                     f"{', '.join(SITE_DEFAULTS[k])}")
                cfg[k][kk] = vv
        else:
            cfg[k] = v
    cfg["_yaml"] = str(Path(yaml_path).resolve())

    def bad(key, why):
        raise ValueError(f"{yaml_path}: {key}: {why}")

    if not cfg["site"]:
        bad("site", "a short name is required (used in file names)")
    if not (cfg["key"] or cfg["polygon"] or cfg["bbox"]):
        bad("key", "give a registry key, a polygon or a bbox")
    if cfg["bbox"] is not None and len(cfg["bbox"]) != 4:
        bad("bbox", "expected [W, S, E, N] in degrees")
    if cfg["bbox"] is not None:
        w, s, e, n = (float(v) for v in cfg["bbox"])
        if not (w < e and s < n):
            bad("bbox", "expected [W, S, E, N] with W < E and S < N")
    if not cfg["period"] or len(cfg["period"]) != 2:
        bad("period", "expected [start, end] (ISO dates, end exclusive)")
    p0, p1 = (pd.Timestamp(str(v)) for v in cfg["period"])
    if not p0 < p1:
        bad("period", "start must be before end")
    cfg["period"] = [str(v) for v in cfg["period"]]
    if not cfg["cube"]:
        bad("cube", "the path of the netCDF cube is required")
    if not cfg["out_dir"]:
        bad("out_dir", "an output folder (under runs/) is required")
    pt = cfg["tide"]["point"]
    if not (pt in ("centroid", "bbox_centre")
            or (isinstance(pt, (list, tuple)) and len(pt) == 2)):
        bad("tide.point", "centroid, bbox_centre or [lat, lon]")
    if cfg["record"]["which"] not in ("latest", "all"):
        bad("record.which", "latest or all")
    sides = {s.strip().lower() for s in str(cfg["marea"]["mouth_side"]).split("+")}
    if not sides <= _SIDES:
        bad("marea.mouth_side", "any, or north/south/east/west joined by '+'")
    if int(cfg["marea"]["quality_min_bracket"]) < 0:
        bad("marea.quality_min_bracket", "0 (off) or a positive count")
    if cfg["export"]["emersion_period"] is not None:
        if len(cfg["export"]["emersion_period"]) != 2:
            bad("export.emersion_period", "expected [start, end] (end exclusive)")
        cfg["export"]["emersion_period"] = [str(v) for v in
                                            cfg["export"]["emersion_period"]]
    return cfg


def site_aoi(cfg):
    """The frame of a site as an AOI: polygon, else bbox, else the registry.

    ``cfg`` is :func:`load_site`'s dict; the AOI is named after ``site``.
    """
    import pyintertidal as pit

    name = cfg["site"]
    if cfg.get("polygon"):
        return pit.AOI.from_polygon([tuple(v) for v in cfg["polygon"]], name=name)
    if cfg.get("bbox"):
        return pit.AOI.from_bbox(*[float(v) for v in cfg["bbox"]], name=name)
    return pit.sites.get(cfg["key"], name=name)


def tide_point(cfg):
    """``(lat, lon)`` of the tide at the mouth.

    ``centroid`` is the registry polygon's centroid when ``key`` is given
    (as for the published Villaviciosa product), else the frame's centroid;
    ``bbox_centre`` is the centre of the frame's bounding box (what
    reconstruct uses when no boundary is passed); ``[lat, lon]`` as given.

    ``cfg`` is :func:`load_site`'s dict; returns floats in degrees, latitude
    first.
    """
    import pyintertidal as pit

    pt = cfg["tide"]["point"]
    if isinstance(pt, (list, tuple)):
        return float(pt[0]), float(pt[1])
    if pt == "centroid":
        if cfg.get("key"):
            return tuple(float(v) for v in pit.sites.get(cfg["key"]).centroid)
        return tuple(float(v) for v in site_aoi(cfg).centroid)
    b = site_aoi(cfg).bbox
    return 0.5 * (b["south"] + b["north"]), 0.5 * (b["west"] + b["east"])


def site_boundary(cfg, out_dir=None, tide_dir=None, verbose=True):
    """The boundary provider a site file asks for (step 6).

    * ``tide.gauge_csv``: your own gauge record (``GaugeBoundary``), columns
      ``time`` (ISO 8601 UTC) and ``level_m`` (m); leading ``#`` lines are
      read as comments (name the datum there).
    * ``tide.n_gauges``: the model plus the N nearest IOC gauges
      (``make_boundary``), downloaded into ``tide.cache_dir`` or
      ``<out_dir>/gauges`` so the run folder keeps them; with
      ``tide.model: null``, the gauges alone. Not usable in this edition as
      shipped: it needs the IOC station list
      ``data_v4/gauges/ioc_stations.json``, which the repository does not
      include, and raises ``FileNotFoundError`` without it.
    * otherwise the model at :func:`tide_point` (``PyTMDBoundary``).

    ``cfg`` is :func:`load_site`'s dict; ``out_dir`` the run folder (for the
    gauge downloads); ``tide_dir`` overrides ``tide.directory`` (the folder
    holding ``EOT20/ocean_tides/``). Downloading IOC gauges needs a network.
    No script or notebook of this edition calls this function: the
    reproduction builds its boundary directly (``PyTMDBoundary`` or
    :class:`TableBoundary`).
    """
    import pyintertidal as pit
    from pyintertidal.boundary import (EnsembleBoundary, GaugeBoundary,
                                       PyTMDBoundary, make_boundary)

    t = cfg["tide"]
    directory = str(tide_dir or resolve(t["directory"]))
    if t.get("gauge_csv"):
        path = resolve(t["gauge_csv"])
        with open(path, encoding="utf-8") as f:
            skip = 0
            for line in f:
                if not line.startswith("#"):
                    break
                skip += 1
        return GaugeBoundary.from_file(str(path), name=Path(path).stem,
                                       time_column="time", level_column="level_m",
                                       skiprows=skip)
    lat, lon = tide_point(cfg)
    if not t.get("n_gauges"):
        if not t.get("model"):
            raise ValueError("tide: set a model, n_gauges or gauge_csv")
        return PyTMDBoundary(t["model"], lat, lon, directory=directory)
    cache = resolve(t.get("cache_dir")) if t.get("cache_dir") else (
        Path(out_dir) / "gauges")
    os.makedirs(cache, exist_ok=True)
    period = tuple(cfg["period"])
    if t["point"] == "centroid" and cfg.get("key"):
        aoi = pit.sites.get(cfg["key"])
        return make_boundary(aoi, t["model"], int(t["n_gauges"]), period,
                             directory=directory, cache_dir=str(cache),
                             exclude=tuple(t.get("exclude") or ()),
                             verbose=verbose)
    # a point other than the registry centroid: the model there, plus the
    # gauges nearest to it, averaged as make_boundary does
    eps = 1e-6
    aoi = pit.AOI.from_bbox(lon - eps, lat - eps, lon + eps, lat + eps, name="tide point")
    gauges = make_boundary(aoi, None, int(t["n_gauges"]), period,
                           directory=directory, cache_dir=str(cache),
                           exclude=tuple(t.get("exclude") or ()), verbose=verbose)
    if not t.get("model"):
        return gauges
    model = PyTMDBoundary(t["model"], lat, lon, directory=directory)
    return EnsembleBoundary([model, gauges],
                            name=f"consensus {model.name} + {gauges.name}")


def load_overpass(path):
    """Acquisition times from an ``overpass_times.json`` file (step 2), as written.

    ``path`` is a JSON object ``{"YYYY-MM-DD": "YYYY-MM-DD HH:MM:SS"}`` of
    UTC instants (the datatake start of each scene), as
    ``products_marea/overpass_times.json``. Returns
    ``{'YYYY-MM-DD': pandas.Timestamp}``, sorted by date.

    This is the function that reads ``products_marea/overpass_times.json``
    in this repository: ``scripts/reproduce_villaviciosa.py``,
    ``replication/build_kit.py``, notebook 01, the example of the package
    docstring and ``scripts/quickstart.py --overpass-times`` call it.
    ``pyintertidal.overpass.load_overpass`` reads the same format and
    differs in three ways:

    * it puts right the known errors listed in
      ``configs/overpass_corrections.json`` for a file whose path ends with
      the listed name (one scene of ``products_urdaibai_10m/overpass_times.json``,
      a file that is not in this repository), and raises ``ValueError`` when
      that file holds, on the listed date, another instant than the wrong
      one listed; this function applies no correction;
    * it returns ``(times, changed)``, with ``changed`` ``{date: (wrong,
      right)}``; this function returns the dict only;
    * it keeps the file's order; this function sorts by date.

    No correction is listed for Villaviciosa, so both give the same 1,380
    times for ``products_marea/overpass_times.json``.
    """
    import pandas as pd

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): pd.Timestamp(v) for k, v in sorted(raw.items())}


# ─────────────────────────────────────────────────────────────────────────────
#  Cube, screening and record (steps 1, 3, 4, 5)
# ─────────────────────────────────────────────────────────────────────────────

def cube_info(cube_path):
    """Dates, shape, grid and dtypes of a netCDF cube, read lazily.

    Returns a dict with ``dates`` (list of 'YYYY-MM-DD'), ``shape`` (H, W),
    ``transform``, ``crs``, ``bands`` {name: (stored dtype, fill)}, ``bbox``
    (the frame in degrees, from the pixel-centre extremes as
    marea.extract computes it) and ``institution`` (the backend that wrote it).

    ``cube_path`` is an openEO netCDF cube with a ``B03`` band and a ``crs``
    variable (as the Copernicus back-end writes them).
    """
    import pyproj
    import xarray as xr
    from pyintertidal.cube import grid_from_dataset

    with xr.open_dataset(cube_path) as ds:
        t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
        dates = [str(np.datetime64(v, "D")) for v in ds[t_dim].values]
        transform, crs = grid_from_dataset(ds)
        bands = {v: (str(ds[v].encoding.get("dtype", ds[v].dtype)),
                     ds[v].encoding.get("_FillValue"))
                 for v in ds.data_vars if v != "crs"}
        wkt = ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref")
        tr = pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(wkt), 4326,
                                         always_xy=True)
        xs, ys = ds["x"].values, ds["y"].values
        lons, lats = tr.transform([xs.min(), xs.max()], [ys.min(), ys.max()])
        return {"dates": dates, "shape": (ds.sizes["y"], ds.sizes["x"]),
                "transform": transform, "crs": crs, "bands": bands,
                "bbox": {"west": float(min(lons)), "south": float(min(lats)),
                         "east": float(max(lons)), "north": float(max(lats))},
                "institution": ds.attrs.get("institution", "")}


def open_cached_cube(cube_path, name=None):
    """A ``SentinelCube`` over an existing netCDF file (never downloads).

    The cube is opened with ``water="ndwi"`` (bands B03, B08, SCL) and an AOI
    built from the file's own frame, named ``name`` (default: the file
    name). Raises ``FileNotFoundError`` when the file cannot be read.
    """
    import pyintertidal as pit

    info = cube_info(cube_path)
    aoi = pit.AOI.from_bbox(info["bbox"]["west"], info["bbox"]["south"],
                            info["bbox"]["east"], info["bbox"]["north"],
                            name=name or Path(cube_path).stem)
    cube = pit.SentinelCube(aoi, (info["dates"][0], info["dates"][-1]),
                            water="ndwi", cache_path=str(cube_path))
    if not cube.cached:
        raise FileNotFoundError(f"{cube_path}: not a readable cube")
    return cube


def screen(cube, threshold=0.0, bad_classes=(3, 8, 9, 10),
           clean_scene_max_bad=0.05, stable_threshold=0.95,
           transition_buffer_px=10, cloud_threshold=0.10):
    """Step 3: the reference map and the scenes clear over the transition zone.

    Parameters
    ----------
    cube : SentinelCube or str
        The cube, or the path of a cached netCDF cube.
    threshold : float
        NDWI above which a clear look is wet (dimensionless).
    bad_classes : tuple of int
        SCL classes counted as cloud or shadow.
    clean_scene_max_bad : float
        A scene votes in the reference map when at most this share (0-1) of
        the frame is in ``bad_classes``.
    stable_threshold : float
        Share of votes (0-1) that makes a pixel stable water or land.
    transition_buffer_px : int
        Dilation (pixels) around the transition class.
    cloud_threshold : float
        A scene is usable when at most this share (0-1) of the transition
        zone is cloudy.

    Returns
    -------
    (usable_dates, reference_map, cloud_pct)
        Sorted 'YYYY-MM-DD' strings; the uint8 map (0 transition, 1 stable
        water, 2 stable land); ``{date: cloud percentage}``.
    """
    import pyintertidal as pit

    if not hasattr(cube, "stream"):
        cube = open_cached_cube(cube)
    ref, cloud_pct = pit.reference_and_clouds(
        cube, threshold=threshold, bad_classes=tuple(bad_classes),
        clean_scene_max_bad=clean_scene_max_bad,
        stable_threshold=stable_threshold,
        transition_buffer_px=transition_buffer_px)
    usable = pit.filter_dates(cloud_pct, cloud_threshold=cloud_threshold)
    return usable, ref, cloud_pct


def record_rows(dates, usable_dates, times, boundary=None, epoch_years=3,
                which="latest"):
    """The rows of an extraction that form the record (the rule of step 5).

    The record is the usable dates that have an acquisition time and a
    finite boundary level; with ``which='latest'`` only those in the latest
    ``epoch_years``-year epoch (``elevation.epochs(...)[-1]``, anchored at
    the end of the record). This is the rule the Villaviciosa product was
    made with.

    Returns ``(rows, record_dates, label)``: row indices into ``dates``, the
    dates themselves and the epoch label (e.g. '2023-2025').

    ``dates`` are the extraction's dates (one per row); ``usable_dates`` the
    scenes kept by :func:`screen`; ``times`` ``{date: timestamp}``;
    ``boundary`` any object with ``levels(times)`` (None skips the level
    check); ``which`` is "latest" or "all" (every usable date). With no
    usable date: empty rows and label None.
    """
    import pandas as pd
    from pyintertidal import epochs

    dates = np.array([str(d) for d in dates])
    pool = [d for d in sorted({str(d) for d in usable_dates}) if d in times]
    if boundary is not None and pool:
        t = pd.DatetimeIndex(pd.to_datetime([times[d] for d in pool])).tz_localize(None)
        lv = np.asarray(boundary.levels(t), float)
        pool = [d for d, v in zip(pool, lv) if np.isfinite(v)]
    if not pool:
        return np.array([], int), [], None
    if which == "latest":
        ep = epochs(pool, epoch_years=int(epoch_years))[-1]
        chosen, label = set(ep["dates"]), ep["label"]
    else:
        chosen, label = set(pool), f"{pool[0][:4]}-{pool[-1][:4]}"
    in_rec = np.array([d in chosen and d in times for d in dates])
    rows = np.flatnonzero(in_rec)
    if boundary is not None and len(rows):
        t = pd.DatetimeIndex(pd.to_datetime([times[d] for d in dates[rows]])).tz_localize(None)
        rows = rows[np.isfinite(np.asarray(boundary.levels(t), float))]
    return rows, [str(d) for d in dates[rows]], label


def build_record(ex, usable_dates, times, boundary, epoch_years=3, which="latest"):
    """Step 5: restrict an extraction to the record.

    Parameters
    ----------
    ex : dict
        What ``pyintertidal.marea.extract`` returns (Y, C, dates, ...).
    usable_dates : iterable of str
        The scenes kept by step 3.
    times : dict
        ``{date: timestamp}`` acquisition times (step 2).
    boundary : BoundaryProvider
        The tide at the mouth (step 6); a date without a finite level is
        dropped.
    epoch_years : int
        Length of an epoch in years; the latest one is kept.
    which : str
        "latest" (the latest epoch, as the product) or "all".

    Returns
    -------
    (ex_rec, record_dates)
        The extraction with Y, C and dates restricted to the record rows
        (``ex_rec['record_epoch']`` holds the epoch label), and the dates.
        For Villaviciosa it returns exactly ``result.json`` record_dates.
        The other keys of ``ex`` (keep, sea, inter, bbox, ...) are passed on
        unchanged; pass ``ex_rec`` to ``marea.reconstruct`` as
        ``extraction``.
    """
    rows, rec, label = record_rows(ex["dates"], usable_dates, times, boundary,
                                   epoch_years, which)
    ex_rec = {**ex, "Y": ex["Y"][rows], "C": ex["C"][rows],
              "dates": np.asarray(ex["dates"])[rows], "record_epoch": label}
    return ex_rec, rec


def save_extraction(ex, path):
    """Cache an extraction (step 4) as a compressed npz.

    ``ex`` is ``marea.extract``'s dict; ``path`` the ``.npz`` to write
    (returned). :func:`load_extraction` reads it back. No script or notebook
    of this edition calls it; use it to avoid a second pass over a large
    cube.
    """
    np.savez_compressed(path, Y=ex["Y"], C=ex["C"], keep=ex["keep"],
                        dates=np.asarray(ex["dates"]), shape=np.asarray(ex["shape"]),
                        sea=ex["sea"], inter=ex["inter"],
                        bbox=np.array([ex["bbox"][k] for k in
                                       ("west", "south", "east", "north")]),
                        crs_wkt=np.array(ex.get("crs_wkt", "")))
    return path


def load_extraction(path):
    """Read an extraction saved by :func:`save_extraction` or a kit record.

    Returns the dict ``pyintertidal.marea.reconstruct`` accepts as
    ``extraction`` (``shape`` a tuple, ``bbox`` a dict in degrees).

    ``path`` e.g. ``replication/villaviciosa_record_2023-2025.npz``: ``Y``
    (T, P) float32 NDWI, ``C`` (T, P) bool clear looks, ``keep`` (P,) flat
    pixel indices, ``dates``, ``shape``, ``sea``/``inter`` (H, W) masks,
    ``bbox``, ``crs_wkt`` (and, in the kit, ``source_cube_sha256``).
    """
    with np.load(path, allow_pickle=False) as d:
        ex = {k: d[k] for k in d.files}
    ex["shape"] = tuple(int(v) for v in ex["shape"])
    if "bbox" in ex:
        w, s, e, n = (float(v) for v in ex["bbox"])
        ex["bbox"] = {"west": w, "south": s, "east": e, "north": n,
                      "crs": "EPSG:4326"}
    for k in ("crs_wkt", "source_cube_sha256"):
        if k in ex:
            ex[k] = str(ex[k])
    return ex


def n_clear(ex_rec):
    """Clear looks of each pixel in the record: (P,) int16.

    ``ex_rec`` is :func:`build_record`'s extraction; the result is the
    ``n_clear`` layer of :func:`export_geotiffs` (pass it in ``extra``).
    """
    return np.asarray(ex_rec["C"]).sum(axis=0).astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
#  Tide at the mouth, from tabulated values (the offline kit)
# ─────────────────────────────────────────────────────────────────────────────

# The two classes below are boundary providers (anything with ``levels`` and
# ``harmonics``); they subclass the library's base class when it imports.
try:
    from pyintertidal.boundary import BoundaryProvider as _Provider
except Exception:                                  # pragma: no cover
    _Provider = object


class TableBoundary(_Provider):
    """A boundary provider that reads tabulated tide levels (no tide model).

    ``levels(times)`` returns the stored values at exactly those instants,
    in the order given, and raises ``KeyError`` for any instant the table
    does not hold: it never interpolates. The replication kit stores EOT20
    at the 97 record instants of Villaviciosa for every whole-minute shift
    from -45 to 180 min, which is every instant reconstruct asks for.

    Parameters
    ----------
    npz_path : str or Path
        A file with ``times_ns`` (T,) int64 UTC instants, ``shifts_min``
        (S,) and ``levels_m`` (S, T) float64: the level at
        ``times - shift``.

    Attributes ``name`` (the model, e.g. "EOT20"), ``lat``, ``lon`` (the
    tabulated point), ``times_ns`` and ``shifts_min`` come from the file.
    """

    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=False) as d:
            t_ns = d["times_ns"].astype(np.int64)
            shifts = d["shifts_min"].astype(np.int64)
            lev = d["levels_m"].astype(np.float64)
            self.name = str(d["model"]) if "model" in d.files else "table"
            self.lat = float(d["lat"]) if "lat" in d.files else None
            self.lon = float(d["lon"]) if "lon" in d.files else None
        inst = t_ns[None, :] - shifts[:, None] * 60_000_000_000
        self.times_ns = t_ns
        self.shifts_min = shifts
        self._table = dict(zip(inst.ravel().tolist(), lev.ravel().tolist()))
        if len(self._table) != inst.size:
            raise ValueError(f"{npz_path}: repeated instants in the table")

    def levels(self, times):
        """Levels (m) at exactly these instants, in the order given.

        ``times``: any datetime-like sequence (naive = UTC). ``KeyError``
        for an instant not in the table.
        """
        import pandas as pd

        idx = pd.DatetimeIndex(pd.to_datetime(times))
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        ns = idx.asi8.tolist()
        missing = [v for v in ns if v not in self._table]
        if missing:
            raise KeyError(
                f"TableBoundary: {len(missing)} of {len(ns)} instants are not in "
                f"the table (first {pd.Timestamp(missing[0])}); it holds only the "
                f"record instants shifted by whole minutes "
                f"{int(self.shifts_min.min())}..{int(self.shifts_min.max())}")
        return np.array([self._table[v] for v in ns], float)

    def harmonics(self, constituents=None):
        """None: a table of levels has no harmonic description."""
        return None


class TideSeries(_Provider):
    """A tide series, regular or not, read by linear interpolation.

    ``levels(times)`` interpolates between samples (across a gap of any
    length) and raises ``ValueError`` for an instant outside the series.
    Used for emersion time, where the tide is needed every 10 minutes over
    years, and as the tide at the mouth of ``scripts/quickstart.py
    --tide-csv`` (``quickstart.read_tide_csv`` returns one).

    ``times`` (N,) datetime-like and ``levels`` (N,) metres, in any order
    (sorted here); ``name`` labels the series. Here and in :meth:`levels` an
    instant with an offset or a time zone is converted to UTC and a naive
    one is taken as UTC (a list that mixes offsets raises ``ValueError``).

    ``pyintertidal.boundary.GaugeBoundary`` reads the same kind of record
    differently: it drops an offset without converting it, returns NaN
    outside the record instead of raising, and returns NaN at an instant
    farther than ``max_gap`` from both neighbouring samples instead of
    bridging the gap.
    """

    def __init__(self, times, levels, name="series"):
        import pandas as pd

        idx = pd.DatetimeIndex(pd.to_datetime(times))
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        order = np.argsort(idx.asi8)
        self._x = idx.asi8[order].astype(np.float64)
        self._y = np.asarray(levels, float)[order]
        self.name = name

    @classmethod
    def from_npz(cls, path):
        """Read ``times_ns`` and ``levels_m`` from an npz (the kit's 10-min series)."""
        with np.load(path, allow_pickle=False) as d:
            name = str(d["model"]) if "model" in d.files else Path(path).stem
            return cls(d["times_ns"].astype("datetime64[ns]"), d["levels_m"], name=name)

    @property
    def span(self):
        """``(first, last)`` instants of the series, as pandas Timestamps (UTC)."""
        import pandas as pd
        return pd.Timestamp(int(self._x[0])), pd.Timestamp(int(self._x[-1]))

    def levels(self, times):
        """Levels (m) at these instants, by linear interpolation, in the order given.

        ``ValueError`` when an instant lies outside :attr:`span`.
        """
        import pandas as pd

        idx = pd.DatetimeIndex(pd.to_datetime(times))
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        q = idx.asi8.astype(np.float64)
        if q.size and (q.min() < self._x[0] or q.max() > self._x[-1]):
            a, b = self.span
            raise ValueError(f"TideSeries {self.name!r} covers {a} .. {b}; "
                             f"asked for {pd.Timestamp(int(q.min()))} .. "
                             f"{pd.Timestamp(int(q.max()))}")
        return np.interp(q, self._x, self._y)

    def harmonics(self, constituents=None):
        """None: the series is used as it is, not as harmonics."""
        return None


def _level_source(src):
    """A level source from anything with ``levels(times)``, a series npz path, or (times, levels)."""
    if hasattr(src, "levels"):
        return src
    if isinstance(src, (str, Path)):
        return TideSeries.from_npz(src)
    times, levels = src
    return TideSeries(times, levels)


def emersion_time(z, tau_px, boundary_or_series, start, end, step_min=10):
    """Emersion time of every pixel: the share of time it is out of the water.

    For each distinct delay in ``tau_px`` the tide at the mouth is read that
    many minutes earlier on a regular grid from ``start`` to ``end``
    (end exclusive), and the emersion is ``1 - hydroperiod(z, h)``: the
    share of instants with the delayed tide at or below ``z``.

    Parameters
    ----------
    z : (P,) array
        Elevation (m, scale of the tide at the mouth); NaN gives NaN.
    tau_px : (P,) array
        Delay of each pixel (min).
    boundary_or_series : provider, path or (times, levels)
        A boundary provider (exact levels), a series npz such as
        ``replication/villaviciosa_eot20_10min_2023-2025.npz`` or a
        ``(times, levels)`` pair (both read by linear interpolation).
    start, end : str
        ISO dates or instants (UTC).
    step_min : int
        Grid step (min).

    Returns
    -------
    (P,) float32 in 0-1 (multiply by 24 for hours per day).

    The source must cover the grid minus the largest delay: the kit's 10-min
    series starts 180 min before 2023-01-01 for that reason.
    """
    import pandas as pd
    from pyintertidal.hydroperiod import hydroperiod

    src = _level_source(boundary_or_series)
    grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end),
                         freq=f"{int(step_min)}min", inclusive="left")
    z = np.asarray(z, float)
    tau = np.asarray(tau_px, float)
    out = np.full(z.shape, np.nan, np.float32)
    ok = np.isfinite(z)
    for tv in np.unique(tau[ok]):
        cols = np.flatnonzero(ok & (tau == tv))
        h = np.asarray(src.levels(grid - pd.Timedelta(minutes=float(tv))), float)
        out[cols] = (1.0 - hydroperiod(z[cols], h)).astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Products (step 11)
# ─────────────────────────────────────────────────────────────────────────────

def grid_reference(ref):
    """``(transform, crs, (H, W))`` from a GeoTIFF, a netCDF cube or a tuple.

    A path ending in ".nc" is read with :func:`cube_info`, any other path
    with ``pyintertidal.raster.grid_of``; a 3-tuple is returned as it is.
    """
    if isinstance(ref, (tuple, list)) and len(ref) == 3:
        return ref
    ref = str(ref)
    if ref.lower().endswith(".nc"):
        info = cube_info(ref)
        return info["transform"], info["crs"], info["shape"]
    from pyintertidal.raster import grid_of
    return grid_of(ref)


def export_geotiffs(product_dir, grid_ref, out_dir, site, period,
                    layers=("z", "sigma", "tau_px_min", "s_km", "band"),
                    extra=None):
    """Write product layers as GeoTIFFs on the cube grid.

    Parameters
    ----------
    product_dir : str or Path
        Folder with ``marea.npz``.
    grid_ref : str, Path or tuple
        A GeoTIFF or netCDF cube on the product's grid, or
        ``(transform, crs, (H, W))``.
    out_dir : str or Path
        Where to write (never products_marea/ or products_villaviciosa/).
    site, period : str
        Used in the names ``marea_<site>_<period>_<layer>.tif``. Pass the
        record's epoch label as ``period`` — ``ex_rec['record_epoch']`` from
        :func:`build_record` (e.g. ``2023-2025``), as
        ``scripts/quickstart.py`` does — so the files are named
        ``marea_<site>_<epoch>_<layer>.tif`` as in the guide.
    layers : sequence of str
        Arrays of marea.npz or keys of ``extra``. ``tau_px_min`` is written
        as ``tau_px`` (minutes).
    extra : dict, optional
        More (P,) layers, e.g. ``{'emersion': E, 'n_clear': n}``.

    Returns
    -------
    list of paths. Floats are float32 with NaN no-data; ``band`` and
    ``n_clear`` are int16 with -1 no-data (outside the intertidal pixels,
    and ``band`` -1 where water cannot reach).
    """
    from pyintertidal.raster import write_geotiff

    out_dir = guard_out_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with np.load(Path(product_dir) / "marea.npz") as d:
        prod = {k: d[k] for k in d.files}
    H, W = (int(v) for v in prod["shape"])
    keep = prod["keep"]
    transform, crs, shape = grid_reference(grid_ref)
    if tuple(shape) != (H, W):
        raise ValueError(f"grid {tuple(shape)} is not the product frame {(H, W)}")
    extra = extra or {}
    paths = []
    for layer in layers:
        arr = extra[layer] if layer in extra else prod[layer]
        name = LAYER_FILE.get(layer, layer)
        path = out_dir / f"marea_{site}_{period}_{name}.tif"
        if layer in INT_LAYERS:
            img = np.full(H * W, -1, np.int16)
            img[keep] = np.asarray(arr).astype(np.int16)
            write_geotiff(str(path), img.reshape(H, W), transform, crs,
                          dtype="int16", nodata=-1)
        else:
            img = np.full(H * W, np.nan, np.float32)
            img[keep] = np.asarray(arr, np.float32)
            write_geotiff(str(path), img.reshape(H, W), transform, crs)
        paths.append(path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  Checks
# ─────────────────────────────────────────────────────────────────────────────

#: arrays of marea.npz that a replication must reproduce bit for bit
IDENTICAL = ("keep", "shape", "z", "sigma", "s_km", "band", "tau_px_min",
             "hyp_z", "hyp_area", "hyp_lo", "hyp_hi", "band_km")
#: (array, tolerance in minutes): the section delays, fitted and applied
CLOSE = (("tau_min", 1e-6), ("tau_usado_min", 1e-6))
#: (array, tolerance in minutes): the bootstrap delays and their interval,
#: skipped by a --fast run (n_boot 0)
BOOT = (("tau_boot_min", 0.01), ("tau_lo_min", 0.01), ("tau_hi_min", 0.01))
#: result.json keys compared exactly (floats within 1e-9). The keys are partly
#: Spanish (sitio = site, estado = status, n_escenas = scenes in the record,
#: n_px_cota = pixels with an elevation, centros_km = section centres,
#: tau_usado_min = the delay applied, con_operador = a delay was applied,
#: veto_fisico = a negative delay was set to 0, rango_marea_m = the mouth tide's
#: range over the record, hipsometria = the hypsometry); the full glossary is in
#: guide/DATA.md (result.json).
RESULT_KEYS = ("sitio", "estado", "n_escenas", "n_px", "record_dates",
               "min_scenes", "overpass_source", "n_px_cota", "s_max_km",
               "band_km", "min_px_band", "centros_km", "tau_px_median_min",
               "tau_px_max_min", "con_operador", "veto_fisico", "rango_marea_m",
               "hipsometria")


def _short(v, n=48):
    """``v`` as text of at most ``n`` characters, for a check table."""
    s = json.dumps(v, default=_json_default) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n - 3] + "..."


def _same(a, b, tol=1e-9):
    """Equal JSON values, numbers within ``tol`` (nested dicts and lists too)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _has_float(v):
    """True when a JSON value holds a float anywhere (its check gets a tolerance)."""
    if isinstance(v, bool):
        return False
    if isinstance(v, float):
        return True
    if isinstance(v, dict):
        return any(_has_float(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_has_float(x) for x in v)
    return False


def compare_products(dir_a, dir_b, expected_cube_sha=None, fast=False):
    """Check a MAREA run against a reference product.

    Parameters
    ----------
    dir_a : str or Path
        The run to check (marea.npz, result.json).
    dir_b : str or Path
        The reference, e.g. products_marea/.
    expected_cube_sha : str, optional
        The sha256 that ``inputs_sha.cube`` of the run must report (the kit
        record in Route A, the authors' cube in Route B; the routes are those
        of guide/REPRODUCE.md).
    fast : bool
        The run had no bootstrap (``n_boot`` 0): the intervals are skipped.

    Returns
    -------
    (rows, passed)
        rows of (check, expected, found, tolerance, status); ``passed`` is
        True when no row FAILs. Arrays in :data:`IDENTICAL` must be equal
        with the same dtype (NaN equal to NaN); tau_min and tau_usado_min
        within 1e-6 min; tau_boot_min, tau_lo_min, tau_hi_min within 0.01 min.
    """
    A = np.load(Path(dir_a) / "marea.npz")
    B = np.load(Path(dir_b) / "marea.npz")
    ra = json.loads((Path(dir_a) / "result.json").read_text(encoding="utf-8"))
    rb = json.loads((Path(dir_b) / "result.json").read_text(encoding="utf-8"))
    rows = []

    def add(check, expected, found, tol, ok):
        rows.append((check, expected, found, tol,
                     ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")))

    for k in IDENTICAL:
        if k not in B.files:
            continue
        if k not in A.files:
            add(f"marea.npz {k}", "present", "missing", "identical", False)
            continue
        a, b = A[k], B[k]
        same = (a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
        if same:
            found = "identical"
        elif a.dtype != b.dtype or a.shape != b.shape:
            found = f"{a.dtype}{a.shape}"
        else:
            eq = a == b
            if a.dtype.kind == "f":
                eq |= np.isnan(a) & np.isnan(b)
            found = f"{int((~eq).sum()):,} values differ"
        add(f"marea.npz {k}", f"{b.dtype}{b.shape}", found, "identical", same)
    for k, tol in CLOSE:
        a, b = A[k], B[k]
        d = float(np.max(np.abs(a - b))) if a.shape == b.shape else np.inf
        add(f"marea.npz {k}", _short(np.round(b, 2).tolist()),
            f"max |diff| {d:.2e} min", f"{tol:g} min", d <= tol)
    for k, tol in BOOT:
        if fast:
            add(f"marea.npz {k}", "-", "not computed (--fast)", f"{tol:g} min", "SKIP")
            continue
        if k not in A.files:
            add(f"marea.npz {k}", "present", "missing", f"{tol:g} min", False)
            continue
        a, b = A[k], B[k]
        d = (float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else np.inf)
        add(f"marea.npz {k}", f"{b.dtype}{b.shape}", f"max |diff| {d:.2e} min",
            f"{tol:g} min", d <= tol)
    for k in RESULT_KEYS:
        if k not in rb:
            continue
        add(f"result.json {k}", _short(rb[k]), _short(ra.get(k, "missing")),
            "1e-9" if _has_float(rb[k]) else "exact",
            k in ra and _same(ra[k], rb[k]))
    if fast:
        add("result.json n_boot", rb.get("n_boot"), ra.get("n_boot"), "-", "SKIP")
    else:
        add("result.json n_boot", rb.get("n_boot"), ra.get("n_boot"), "exact",
            ra.get("n_boot") == rb.get("n_boot"))
    for k in ("tau_min", "tau_usado_min"):
        if k in rb:
            ok = k in ra and np.allclose(ra[k], rb[k], rtol=0, atol=1e-6)
            add(f"result.json {k}", _short([round(v, 2) for v in rb[k]]),
                _short([round(v, 2) for v in ra.get(k, [])]), "1e-6 min", ok)
    extra = sorted(set(ra) - set(rb))
    expected_extra = ["quality_min_bracket"] if "quality_min_bracket" not in rb else []
    add("result.json extra keys", _short(expected_extra), _short(extra), "exact",
        extra == expected_extra)
    if "quality_min_bracket" not in rb:
        add("result.json quality_min_bracket", 0, ra.get("quality_min_bracket"),
            "exact", ra.get("quality_min_bracket") == 0)
    if expected_cube_sha is not None:
        found = ra.get("inputs_sha", {}).get("cube")
        add("result.json inputs_sha.cube", expected_cube_sha[:12] + "...",
            (found or "missing")[:12] + "...", "exact", found == expected_cube_sha)
    passed = all(r[4] != "FAIL" for r in rows)
    return rows, passed


def overlap_summary(dir_a, dir_b):
    """Share of identical elevations and the largest |dz| on the common pixels.

    For a run from a re-downloaded cube (Route C): no verdict, only facts.
    Returns a dict.

    ``dir_a`` (the run) and ``dir_b`` (the reference) hold ``marea.npz``;
    pixels are matched by their flat index (``keep``) and |dz| is in metres.
    """
    A = np.load(Path(dir_a) / "marea.npz")
    B = np.load(Path(dir_b) / "marea.npz")
    common, ia, ib = np.intersect1d(A["keep"], B["keep"], return_indices=True)
    za, zb = A["z"][ia], B["z"][ib]
    both = np.isfinite(za) & np.isfinite(zb)
    dz = np.abs(za[both] - zb[both])
    return {"pixels_run": int(A["keep"].size), "pixels_reference": int(B["keep"].size),
            "pixels_common": int(common.size), "z_both_finite": int(both.sum()),
            "z_identical_share": float((dz == 0).mean()) if dz.size else float("nan"),
            "z_max_abs_diff_m": float(dz.max()) if dz.size else float("nan"),
            "z_median_abs_diff_m": float(np.median(dz)) if dz.size else float("nan")}


def qa_table(run_dir):
    """The quality checks of a run (the quality table of notebook 03), from its outputs.

    Returns rows of (check, where, found, fine when). Facts about the run
    only; nothing here compares with a measurement.

    ``run_dir`` is a ``marea.reconstruct`` output folder (result.json,
    marea.npz, and coverage.json when :func:`coverage` wrote one). A run
    that did not end "ok" gives the status row only.
    """
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    rows = [("estado", "result.json estado", res.get("estado"), "ok")]
    if res.get("estado") != "ok" or not (run_dir / "marea.npz").exists():
        return rows
    d = np.load(run_dir / "marea.npz")
    z, sg, s, band = d["z"], d["sigma"], d["s_km"], d["band"]
    lo, hi = res["rango_marea_m"]
    fin = np.isfinite(z)
    n_s = int(np.isfinite(s).sum())
    rows += [
        ("pixels with a distance s", "marea.npz s_km", f"{n_s:,}",
         "> 5,000 (else no delay is fitted)"),
        ("share of band -1", "marea.npz band", f"{(band == -1).mean():.1%}",
         "small (pixels water cannot reach)"),
        ("sections", "result.json centros_km",
         f"{len(res['centros_km'])} at " + ", ".join(f"{c:.2f}" for c in res["centros_km"]) + " km",
         "one per 3 km of estuary"),
    ]
    # 179.5 min: the 180-min end of marea.TAU_BOUNDS and of the shift bank (a
    # delay there was not measured); 0.03 and 0.65 m: the ends of marea.SG_GRID
    # (a sigma pinned at an end was not resolved)
    lo_i, hi_i = np.asarray(res["tau_lo_min"], float), np.asarray(res["tau_hi_min"], float)
    width =", ".join("-" if not np.isfinite(a) else f"{b - a:.1f}"
                      for a, b in zip(lo_i, hi_i))
    rows += [
        ("interval width per section (min)", "result.json tau_lo_min, tau_hi_min",
         width, "narrow compared with the delay"),
        ("section at the 180-min bound", "result.json tau_usado_min",
         str(bool(np.any(np.asarray(res["tau_usado_min"]) >= 179.5))), "False"),
        ("veto_fisico (a negative section clipped to 0)", "result.json veto_fisico",
         str(res.get("veto_fisico")), "False"),
        ("elevations", "result.json n_px_cota / n_px",
         f"{res['n_px_cota']:,} of {res['n_px']:,} ({res['n_px_cota'] / res['n_px']:.0%})",
         "most pixels"),
        ("share of z at the edges of rango_marea_m", "marea.npz z",
         f"{(np.isclose(z[fin], lo, atol=1e-6) | np.isclose(z[fin], hi, atol=1e-6)).mean():.1%}",
         "small"),
        ("share of sigma at 0.03 or 0.65 m", "marea.npz sigma",
         f"{(np.isclose(sg[fin], 0.03) | np.isclose(sg[fin], 0.65)).mean():.1%}",
         "small"),
    ]
    cov = run_dir / "coverage.json"
    if cov.exists():
        v = json.loads(cov.read_text(encoding="utf-8"))
        rows.append(("coverage_verdict", "coverage.json suitable",
                     str(v.get("suitable")) + ("" if v.get("suitable") else
                                               " (" + "; ".join(v.get("reasons", [])) + ")"),
                     "True"))
    return rows


def coverage(record_levels, boundary, start, end, out_dir=None, every_hours=1):
    """Tide coverage of the record: coverage_metrics and coverage_verdict.

    ``record_levels`` are the levels at the record instants; the reference
    is the boundary every ``every_hours`` hours from ``start`` to ``end``.
    Writes ``coverage.json`` in ``out_dir`` when given; returns the verdict.

    A wrapper of the module ``pyintertidal.coverage`` (same name, imported
    here as ``cv``), with that module's default thresholds. No script or
    notebook of this edition calls it; :func:`qa_table` reads the
    coverage.json it writes. ``boundary`` is any object with
    ``levels(times)``; ``end`` is exclusive.
    """
    import pandas as pd
    from pyintertidal import coverage as cv

    grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end),
                         freq=f"{int(every_hours)}h", inclusive="left")
    ref = np.asarray(boundary.levels(grid), float)
    m = cv.coverage_metrics(np.asarray(record_levels, float), ref)
    v = cv.coverage_verdict(m)
    if out_dir is not None:
        write_json(Path(out_dir) / "coverage.json",
                   {"suitable": bool(v["suitable"]), "checks": v["checks"],
                    "reasons": v["reasons"], "metrics": m,
                    "reference": f"{boundary.name} every {every_hours} h, "
                                 f"{start} to {end} (end exclusive)"})
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  Environment and provenance
# ─────────────────────────────────────────────────────────────────────────────

def _pins():
    """``{package (lower case): version}`` from the ``name==version`` lines of requirements.txt."""
    pins = {}
    if PINNED_FILE.exists():
        for line in PINNED_FILE.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                name, ver = line.split("==", 1)
                pins[name.strip().lower()] = ver.strip()
    return pins


def _installed(dist):
    """The installed version of a distribution, or None."""
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def check_env():
    """Installed against pinned versions (the full pinned environment).

    Returns
    -------
    (rows, ok)
        rows of (package, required, installed, status); ``ok`` is False
        when a core package is missing. A different version is reported but
        does not fail: the arrays were checked identical with the pins.
    """
    pins = _pins()
    py = platform.python_version()
    rows = [("python", PYTHON_PINNED, py, "ok" if py == PYTHON_PINNED else "other version")]
    ok = True
    for dist, role in PACKAGES:
        req = pins.get(dist.lower(), "-")
        got = _installed(dist)
        if got is None:
            status = "MISSING" if role == "core" else f"missing (needed for {role})"
            ok &= role != "core"
        elif req != "-" and got != req:
            status = "other version"
        else:
            status = "ok"
        rows.append((dist, req, got or "-", status))
    return rows, ok


def _git(*args):
    """The output of ``git -C <repository root> <args>``, or None when git cannot run."""
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def provenance(out_dir, inputs, params, started=None):
    """Write ``provenance.json`` (git commit, versions, input sha256, parameters).

    Parameters
    ----------
    out_dir : str, Path or None
        Folder to write into; None only returns the dict.
    inputs : dict
        ``{label: path}`` or ``{label: {'path': ..., 'sha256': ...}}`` when
        the hash is already known (a 3 GB cube need not be read twice).
    params : dict
        The parameters of the run.
    started : datetime, optional
        UTC start of the run (default: now).

    Returns
    -------
    dict
        ``git_commit``, ``git_dirty`` (tracked files changed since that
        commit; None outside a git checkout), ``python``, ``os``,
        ``packages``, ``inputs``, ``parameters``, ``started_utc``,
        ``ended_utc``. Input paths inside the repository are written
        relative to its root.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    started = started or now
    inp = {}
    for label, v in (inputs or {}).items():
        if isinstance(v, dict):
            p, h = v.get("path"), v.get("sha256")
        else:
            p, h = v, None
        if p is not None and Path(p).exists():
            inp[label] = {"path": rel(p), "bytes": Path(p).stat().st_size,
                          "sha256": h or sha256(p)}
        else:
            inp[label] = {"path": None if p is None else str(p), "sha256": h}
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    # The executed tutorials rewrite themselves while they are built, which is
    # not a code change. NOTE: notebooks/tutorials/ does not exist in this
    # edition (the notebooks are in notebooks/), so this filter matches nothing
    # here and a notebook re-executed in place marks the run git_dirty.
    if status:
        status ="\n".join(l for l in status.splitlines()
                           if not re.search(r"notebooks/tutorials/[^/]+\.ipynb$", l))
    prov = {
        "git_commit": commit or None,
        "git_dirty": (bool(status) if status is not None and commit else None),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "packages": {d: _installed(d) for d, _ in PACKAGES},
        "inputs": inp,
        "parameters": params,
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if out_dir is not None:
        write_json(Path(out_dir) / "provenance.json", prov)
    return prov


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic npz (the replication kit)
# ─────────────────────────────────────────────────────────────────────────────

def write_npz(path, arrays):
    """Write arrays as a compressed npz whose bytes depend only on the data.

    ``numpy.savez_compressed`` stamps the current time into the zip, so the
    sha256 of a rebuilt file changes; this writes a fixed timestamp. The
    file reads with ``numpy.load`` as usual (no pickle).

    ``path`` the ``.npz`` to write (returned); ``arrays`` ``{name: array}``
    — object arrays are refused.
    """
    from numpy.lib import format as npf

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for key, arr in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as f:
                npf.write_array(f, np.asanyarray(arr), allow_pickle=False)
    return path
