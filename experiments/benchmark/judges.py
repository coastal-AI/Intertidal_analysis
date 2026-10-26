# -*- coding: utf-8 -*-
"""v13 benchmark — the in-estuary gauges that judge every lag (§1.6, §2.8).

Question
--------
Which tide gauges inside each estuary judge the methods' lags, where does
each one sit on the run grid, which pixels form its neighbourhood, and which
of them are scored, by rules that look at no method?

Pre-registered rules (FROZEN['judges']; FROZEN['package']['NB'] for the
package domain)
-----------------------------------------------------------------------
* Inputs: ``data_v4/gauges/estuary_truth/stations.json`` and one CSV per
  station (``time_utc,level_m``, naive UTC, QC applied upstream), read as
  delivered. Datums differ between providers (NAP, NHN, the raw IOC
  reference, a record median), so only centred quantities are compared.
* Every station of ``config.JUDGES[site]`` is listed; the rules below, not a
  method, decide which ones are scored.
* Geometry on the run grid, never the 20 m fields of stations.json: the
  station is projected to the cube CRS; d_min is its distance to the nearest
  keep_full pixel; the 200 nearest keep_full pixels give the modal band k_g
  (bands >= 0) and the median s_g; the neighbourhood radius is
  R = max(500, d_min + 300) m, widened by 250 m while fewer than 100
  keep_full pixels lie within R, the widening stopping at 2,000 m; the
  neighbourhood nbhd_g is every frame pixel within R. A station with fewer
  than 100 keep_full pixels within 2 km is flagged "sparse".
* Inclusion, by method-independent rules only (D10), every rule evaluated
  for every station:
  1. ``in_frame``: the station lies inside the frame;
  2. ``coverage``: a gauge value within 6 min of at least 50 % of the record
     instants;
  3. ``boundary_member``: more than 3 km from every gauge member of the
     site's boundary (the IOC gauges whose record enters h_B; the tide-model
     point is not a gauge);
  4. ``dry``: at most 10 % of its record-instant values within 1 cm of the
     minimum of those values.
* The two Terneuzen gauges, 0.7 km apart, are one location (the scorer
  averages their errors). Judges with s_g < 3 km are "mouth" judges; they
  stay in the headline set and a sensitivity drops them.
* Centred 10-min series (for the harmonic truth and the all-sample lag):
  10-min series (RWS F007 means) are used as delivered on the 10-min grid;
  finer series are averaged over the closed window [T - 5 min, T + 5 min]
  around each grid instant T (symmetric about T, so no timing shift),
  requiring at least 6 samples of a 1-min series and 2 of a 5-min series.

Two copies of the judge table exist: ``common/<site>/judges.json``, written
by ``prepare`` (its neighbourhoods feed the package domain through
:func:`nb_flat`), and ``truth/<site>/judges.json``, written by ``truth``,
which is the table every truth and score uses. Both are built by
:func:`build`; the neighbourhood pixel lists are not stored, they are
recomputed from (gx, gy, R_g) by :func:`load` and checked against a stored
sha256.

Validation v2 (docs/validation_v2_spec.md §1.7, §6.1; the section at the
end of this module; everything above is unchanged)
-------------------------------------------------------------------------
* The coverage and dry rules are evaluated PER WINDOW, on the window's
  record (:func:`window_inclusion`, :func:`coverage_by_window`); the
  geometry stays the one of the union-record domain under P0.
* The headline set of §6.1 (:data:`HEADLINE`, 13 locations) is a check of
  what the rules give (:func:`check_headline`), never an override.
* The development-judge list (:data:`DEVELOPMENT_JUDGES`: Harlingen, plus
  what the authors add in LOCK.json; :data:`DEVELOPMENT_CANDIDATES` are to
  be confirmed) defines the "development-free" sensitivity; "without mouth"
  drops s_g < 3 km (:func:`judge_sets`).
* Time stamps: each provider's averaging window and stamp position
  (:data:`TIMESTAMP_CONVENTIONS`); series are centred before use
  (:func:`centred`) and a declared sensitivity shifts them by +- half the
  window (:func:`half_window_min`). An unknown provider is refused.
* The scene instants the judges are read at are the package record's: the
  catalogue / product-name times (datatake start) by default, as every
  production result (user decision of 2026-10-14); the solar-angle
  sensing time is an optional diagnostic used only when a run is given
  ``--sensing-times``.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from experiments.benchmark import config, levels, product

_J = config.FROZEN["judges"]
#: neighbourhood rule of the score side (the package NB uses its own copy of
#: the same numbers, FROZEN['package']['NB'])
NEIGHBOURHOOD = _J["neighbourhood"]
INCLUSION = _J["inclusion"]
SPARSE = _J["sparse_flag"]
MOUTH_S_KM = float(_J["mouth_s_km"])
#: the inclusion rules, in the order they are reported
RULES = ("in_frame", "coverage", "boundary_member", "dry")
#: half-width of the centred 10-min window (min) and the samples it needs
TEN_MIN_HALF = 5
TEN_MIN_MIN_SAMPLES = {1: 6, 5: 2}
JUDGES_FORMAT = "v13-judges-1"
#: station fields copied from stations.json
STATION_KEYS = ("code", "name", "provider", "lon", "lat", "datum", "sampling_min",
                "file", "period", "site", "in_boundary")
NS_PER_MIN = 60_000_000_000
EARTH_RADIUS_KM = 6371.0          # as pyintertidal.boundary.nearest_gauges
del _J


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  stations and series
# ─────────────────────────────────────────────────────────────────────────────

def stations(site, path=None):
    """The listed stations of ``site`` (``config.JUDGES`` order).

    Parameters
    ----------
    site : str
    path : str, optional
        ``stations.json`` (default ``config.STATIONS_JSON``).

    Returns
    -------
    list of dict
        The :data:`STATION_KEYS` fields of each station.
    """
    codes = config.JUDGES.get(site, ())
    if not codes:
        return []
    path = path or config.STATIONS_JSON
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    by_code = {r["code"]: r for r in rows}
    missing = [c for c in codes if c not in by_code]
    if missing:
        raise KeyError(f"{path} lacks the judges {missing} of {site}")
    out = []
    for code in codes:
        r = by_code[code]
        if r.get("site") != site:
            raise ValueError(f"{code} belongs to {r.get('site')!r}, not {site!r}")
        if r.get("in_boundary"):
            # a boundary member judging the boundary would score 0 by construction
            raise ValueError(f"{code} is marked in_boundary in {path}")
        out.append({k: r.get(k) for k in STATION_KEYS})
    return out


def series_path(station, base=None):
    return os.path.join(base or config.ESTUARY_TRUTH, station["file"])


def read_series(station, base=None):
    """The station's CSV as delivered: a float64 Series on a naive-UTC,
    strictly increasing ns DatetimeIndex (QC was applied upstream; absent
    rows are the gaps).

    A non-finite value or a repeated instant would be a delivery error: the
    first is dropped and the second keeps its first value, both logged.
    """
    path = series_path(station, base)
    df = pd.read_csv(path, dtype={"level_m": np.float64})
    if list(df.columns[:2]) != ["time_utc", "level_m"]:
        raise ValueError(f"{path}: expected columns time_utc,level_m, got {list(df.columns)}")
    idx = pd.DatetimeIndex(pd.to_datetime(df["time_utc"], format="%Y-%m-%d %H:%M:%S"))
    idx = idx.as_unit("ns")
    v = df["level_m"].to_numpy(np.float64)
    ok = np.isfinite(v)
    if not ok.all():
        log(f"[judges] {station['code']}: {int((~ok).sum())} non-finite values dropped")
    s = pd.Series(v[ok], index=idx[ok], name=station["code"])
    if not s.index.is_monotonic_increasing:
        s = s.sort_index(kind="stable")
    dup = s.index.duplicated(keep="first")
    if dup.any():
        log(f"[judges] {station['code']}: {int(dup.sum())} repeated instants dropped")
        s = s[~dup]
    return s


def at_instants(series, t, tol_min=None):
    """The gauge value nearest each instant of ``t`` within ``tol_min``
    minutes (NaN beyond): pandas' ``reindex(method="nearest", tolerance)``,
    as the lag agreement study read its judges."""
    tol = float(INCLUSION["tolerance_min"] if tol_min is None else tol_min)
    t = pd.DatetimeIndex(t)
    return series.reindex(t, method="nearest",
                          tolerance=pd.Timedelta(minutes=tol)).to_numpy(np.float64)


def ten_min_grid():
    """The 10-min grid of the benchmark (FROZEN['levels']['series_10min'])."""
    sp = config.FROZEN["levels"]["series_10min"]
    return pd.date_range(sp["start"], sp["end"], freq=f"{int(sp['every'])}min").as_unit("ns")


def centred_10min(series, sampling_min, grid=None):
    """The gauge on the 10-min grid, centred on each grid instant.

    Parameters
    ----------
    series : pd.Series
        As :func:`read_series` returns it.
    sampling_min : float
        The station's sampling (stations.json ``sampling_min``).
    grid : DatetimeIndex, optional
        Default :func:`ten_min_grid`.

    Returns
    -------
    pd.Series on ``grid``
        10-min series: the delivered values at the grid instants (exact
        times; RWS F007 values are already centred 10-min means). Finer
        series: the mean of the samples in [T - 5, T + 5] min, NaN with
        fewer than :data:`TEN_MIN_MIN_SAMPLES` samples.
    """
    grid = ten_min_grid() if grid is None else pd.DatetimeIndex(grid).as_unit("ns")
    step = float(sampling_min)
    if step >= 10:
        return series.reindex(grid).astype(np.float64)
    need = TEN_MIN_MIN_SAMPLES.get(int(round(step)))
    if need is None or abs(step - round(step)) > 1e-9:
        raise ValueError(f"no centred 10-min rule for a {sampling_min}-min series")
    x = series.index.asi8
    v = series.to_numpy(np.float64)
    g = grid.asi8
    half = TEN_MIN_HALF * NS_PER_MIN
    lo = np.searchsorted(x, g - half, side="left")
    hi = np.searchsorted(x, g + half, side="right")
    n = hi - lo
    # window sums from a cumulative sum of the median-removed values (the
    # removal keeps the running sum small: the error stays below 1e-9 m)
    ref = float(np.median(v)) if len(v) else 0.0
    c = np.concatenate([[0.0], np.cumsum(v - ref)])
    mean = (c[hi] - c[lo]) / np.maximum(n, 1) + ref
    return pd.Series(np.where(n >= need, mean, np.nan), index=grid, name=series.name)


# ─────────────────────────────────────────────────────────────────────────────
#  the run grid
# ─────────────────────────────────────────────────────────────────────────────

def frame_shape(pkg):
    """(H, W) of a package (``pkg.shape``, or ``pkg.H``/``pkg.W``)."""
    shp = getattr(pkg, "shape", None)
    if shp is not None:
        return int(shp[0]), int(shp[1])
    return int(pkg.H), int(pkg.W)


def grid_of(pkg):
    """The cube grid (``v7.cube_grid``: xs, ys pixel centres, crs, transform)
    of a package: ``pkg.grid`` when the package carries it, else read from
    ``pkg.cfg.cube`` (coordinates only, no band is loaded)."""
    g = getattr(pkg, "grid", None)
    if g is None:
        # imported here: v7 chdirs to the repository root on import, which
        # is harmless for the CLI (it runs there) but must not happen at
        # this module's import
        from experiments.v7_waterline_contour import cube_grid
        g = cube_grid(pkg.cfg.cube)
    H, W = frame_shape(pkg)
    if (len(g["ys"]), len(g["xs"])) != (H, W):
        raise ValueError(f"grid {len(g['ys'])} x {len(g['xs'])} differs from the frame {H} x {W}")
    return g


def project(lon, lat, crs):
    """(x, y) of a WGS84 point in the grid CRS."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(float(lon), float(lat))
    return float(x), float(y)


def pixel_of(grid, gx, gy):
    """(row, col) of the frame pixel that holds (gx, gy), None outside.
    Pixel edges are half a step from the centres (v7.cube_grid's transform)."""
    xs, ys = np.asarray(grid["xs"], float), np.asarray(grid["ys"], float)
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    col = int(np.floor((gx - (xs[0] - dx / 2)) / dx))
    row = int(np.floor((gy - (ys[0] - dy / 2)) / dy))
    if 0 <= row < len(ys) and 0 <= col < len(xs):
        return row, col
    return None


def nbhd_flat(grid, gx, gy, R):
    """Frame flat indices (ascending) of every pixel centre within ``R`` m of
    (gx, gy): ``hypot(x - gx, y - gy) <= R``, evaluated on the bounding box
    only (|dx| <= R is implied, so the result equals the full-frame test)."""
    xs, ys = np.asarray(grid["xs"], float), np.asarray(grid["ys"], float)
    W = len(xs)
    c = np.flatnonzero(np.abs(xs - gx) <= R)
    r = np.flatnonzero(np.abs(ys - gy) <= R)
    if not len(c) or not len(r):
        return np.empty(0, np.int64)
    d = np.hypot(xs[c][None, :] - gx, ys[r][:, None] - gy)
    rr, cc = np.nonzero(d <= R)
    return r[rr].astype(np.int64) * W + c[cc].astype(np.int64)


def neighbourhood_radius(d_sorted, params=None):
    """R of the neighbourhood rule from the ascending keep_full distances.

    R = max(r0, d_min + pad); while fewer than ``min_keep_px`` distances are
    <= R and R < cap: R = min(R + step, cap). A start beyond the cap (a
    station far from the keep) is kept as it is.
    """
    p = NEIGHBOURHOOD if params is None else params
    d_sorted = np.asarray(d_sorted, float)
    R = max(float(p["r0_m"]), float(d_sorted[0]) + float(p["pad_m"]))
    cap = float(p["cap_m"])
    while (np.searchsorted(d_sorted, R, side="right") < int(p["min_keep_px"])
           and R < cap):
        R = min(R + float(p["step_m"]), cap)
    return float(R)


def station_geometry(lon, lat, grid, keep, s_km, band_of, centers=None, params=None):
    """Where a station sits on the run grid (§2.8).

    Parameters
    ----------
    lon, lat : float
        WGS84 position.
    grid : dict
        ``xs``, ``ys`` (pixel centres) and ``crs``.
    keep, s_km, band_of : arrays
        The full-record domain: keep_full (frame flat indices), its s (km,
        NaN isolated) and band (-1 for NaN s).
    centers : array, optional
        Band centres (km), to report the centre of k_g.
    params : mapping, optional
        The neighbourhood rule (default the judges' one).

    Returns
    -------
    dict
        gx, gy, row, col, in_frame, d_min_km, k_g, k_g_share, centre_k_km,
        s_g_km, R_g_m, nbhd (frame flat, ascending), n_nbhd, n_keep_in_nbhd,
        n_keep_2km, nbhd_sha256, flags.
    """
    p = NEIGHBOURHOOD if params is None else params
    keep = np.asarray(keep, np.int64)
    if not len(keep):
        raise ValueError("keep_full is empty: no geometry")
    xs, ys = np.asarray(grid["xs"], float), np.asarray(grid["ys"], float)
    W = len(xs)
    gx, gy = project(lon, lat, grid["crs"])
    d = np.hypot(xs[keep % W] - gx, ys[keep // W] - gy)
    # a stable sort breaks the many equal distances of a regular grid by
    # pixel order, so the 200 nearest are the same on every platform
    order = np.argsort(d, kind="stable")
    d_sorted = d[order]
    near = order[:int(p["n_nearest"])]
    band_of = np.asarray(band_of)
    bn = band_of[near]
    bpos = bn[bn >= 0]
    if len(bpos):
        k_g = int(np.bincount(bpos).argmax())
        k_share = float(np.mean(bn == k_g))
    else:
        k_g, k_share = -1, 0.0
    s_near = np.asarray(s_km, float)[near]
    s_g = float(np.nanmedian(s_near)) if np.isfinite(s_near).any() else float("nan")
    R = neighbourhood_radius(d_sorted, p)
    nb = nbhd_flat(grid, gx, gy, R)
    n_keep_R = int(np.searchsorted(d_sorted, R, side="right"))
    n_keep_2km = int(np.searchsorted(d_sorted, float(SPARSE["radius_m"]), side="right"))
    pix = pixel_of(grid, gx, gy)
    centre = (float(centers[k_g]) if (centers is not None and 0 <= k_g < len(centers))
              else float("nan"))
    flags = []
    if pix is None:
        flags.append("outside_frame")
    if n_keep_2km < int(SPARSE["min_keep_px"]):
        flags.append("sparse")
    if k_g < 0:
        flags.append("no_band")
    if not np.isfinite(s_g):
        flags.append("no_s")
    if n_keep_R < int(p["min_keep_px"]):
        flags.append("few_keep_in_nbhd")
    return {"gx": gx, "gy": gy,
            "row": None if pix is None else int(pix[0]),
            "col": None if pix is None else int(pix[1]),
            "in_frame": pix is not None,
            "d_min_km": float(d_sorted[0]) / 1000.0,
            "k_g": k_g, "k_g_share": k_share, "centre_k_km": centre,
            "s_g_km": s_g, "R_g_m": R, "nbhd": nb, "n_nbhd": int(len(nb)),
            "n_keep_in_nbhd": n_keep_R, "n_keep_2km": n_keep_2km,
            "nbhd_sha256": config.sha256_bytes(nb.astype("<i8").tobytes()),
            "flags": flags}


def _full(pkg):
    """keep, s_km, band_of, centers of the package's full-record domain."""
    full = pkg.full
    return (np.asarray(full.keep, np.int64), np.asarray(full.s_km, float),
            np.asarray(full.band_of), getattr(full, "centers", None))


def geometry(pkg, station, params=None, grid=None):
    """:func:`station_geometry` of ``station`` on the package's grid and
    full-record domain (``pkg.full``)."""
    keep, s_km, band_of, centers = _full(pkg)
    return station_geometry(station["lon"], station["lat"],
                            grid if grid is not None else grid_of(pkg),
                            keep, s_km, band_of, centers, params)


# ─────────────────────────────────────────────────────────────────────────────
#  inclusion
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km), the formula of ``boundary.nearest_gauges``."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def boundary_members(cfg, provider=None, stations_path=None):
    """The gauge members of the site's boundary, with their positions.

    ``make_boundary`` takes the ``n_gauges`` nearest IOC stations of
    ``ioc_stations.json`` after removing the held-out codes (the same rule
    as ``levels.ioc_chunks``); with ``provider`` the list is cut to the
    gauges the provider actually holds. A tide-model-only boundary has none.

    Returns
    -------
    list of dict
        ``{"code", "lat", "lon"}``.
    """
    b = cfg.boundary
    if b["kind"] != "make_boundary" or not b.get("n_gauges"):
        return []
    from pyintertidal import sites
    from pyintertidal.boundary import nearest_gauges

    stations_path = stations_path or config.IOC_STATIONS
    lat, lon = sites.get(b["aoi"]).centroid
    excl = {str(c).lower() for c in b["exclude"]}
    near = [g for g in nearest_gauges(lat, lon, int(b["n_gauges"]) + len(excl),
                                      stations_path=stations_path)
            if g[0].lower() not in excl][:int(b["n_gauges"])]
    codes = [g[0] for g in near]
    if provider is not None:
        used = set(levels.describe_provider(provider)["gauges"])
        codes = [c for c in codes if c in used]
    with open(stations_path, encoding="utf-8") as f:
        rows = json.load(f)
    rows = rows if isinstance(rows, list) else rows.get("stations", rows)
    pos = {r["Code"]: (float(r["Lat"]), float(r["Lon"])) for r in rows
           if "Code" in r and "Lat" in r and "Lon" in r}
    return [{"code": c, "lat": pos[c][0], "lon": pos[c][1]} for c in codes]


def inclusion_rules(station, g_s2, in_frame, members, rules=None):
    """The method-independent inclusion rules (§2.8) of one station.

    Parameters
    ----------
    station : dict
        With ``lat`` and ``lon``.
    g_s2 : (T,) float
        The gauge at the record instants (nearest within 6 min, NaN else).
    in_frame : bool
    members : list of dict
        Boundary gauge members (``code``, ``lat``, ``lon``).
    rules : mapping, optional
        FROZEN['judges']['inclusion'].

    Returns
    -------
    dict
        n_record, n_with_gauge, coverage, dry_share, nearest_member,
        nearest_member_km, passed {rule: bool}, included, exclusion (failed
        rules in :data:`RULES` order).
    """
    r = INCLUSION if rules is None else rules
    g = np.asarray(g_s2, float)
    fin = np.isfinite(g)
    n, T = int(fin.sum()), int(len(g))
    coverage = n / T if T else 0.0
    if n:
        vals = g[fin]
        dry_share = float(np.mean(vals <= vals.min() + float(r["dry_tol_m"])))
    else:
        dry_share = float("nan")
    dist = sorted(((haversine_km(station["lat"], station["lon"], m["lat"], m["lon"]), m["code"])
                   for m in members))
    near_km, near_code = dist[0] if dist else (float("inf"), None)
    passed = {"in_frame": bool(in_frame),
              "coverage": bool(coverage >= float(r["coverage_min"])),
              "boundary_member": bool(near_km > float(r["boundary_member_min_km"])),
              "dry": bool(n > 0 and dry_share <= float(r["dry_max_share"]))}
    return {"n_record": T, "n_with_gauge": n, "coverage": float(coverage),
            "dry_share": dry_share, "nearest_member": near_code,
            "nearest_member_km": float(near_km), "passed": passed,
            "included": all(passed.values()),
            "exclusion": [k for k in RULES if not passed[k]]}


def inclusion(pkg, station, series, members=None, geom=None):
    """:func:`inclusion_rules` of ``station`` on the package's record."""
    if members is None:
        members = boundary_members(pkg.cfg)
    if geom is None:
        geom = geometry(pkg, station)
    g = at_instants(series, pkg.t)
    return inclusion_rules(station, g, geom["in_frame"], members)


def location_of(code):
    """The location a station counts as (the first code of its
    ``config.SAME_LOCATION`` group, else itself)."""
    for group in config.SAME_LOCATION:
        if code in group:
            return group[0]
    return code


# ─────────────────────────────────────────────────────────────────────────────
#  the judge table
# ─────────────────────────────────────────────────────────────────────────────

def judges_path(pkg):
    """``common/<site>/judges.json`` of a package (the prepare-time copy)."""
    d = getattr(pkg, "dir", None) or pkg.cfg.common_dir
    return os.path.join(d, "judges.json").replace("\\", "/")


def _sha_arrays(*arrays):
    import hashlib

    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype.str).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _file_id(path):
    st = os.stat(path)
    return {"path": path.replace("\\", "/"), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns)}


def inputs_of(pkg, stas=None, grid=None, members=None):
    """What a judge table depends on: the record instants, the full-record
    domain, the grid, the station files, the boundary members, the FROZEN
    judges section and this module. Two equal dicts mean the same table."""
    keep, s_km, band_of, _ = _full(pkg)
    grid = grid if grid is not None else grid_of(pkg)
    stas = stations(pkg.cfg.site) if stas is None else stas
    t = pd.DatetimeIndex(pkg.t).as_unit("ns")
    return {"t_sha": _sha_arrays(t.asi8.astype("<i8")),
            "domain_sha": _sha_arrays(keep.astype("<i8"), s_km.astype("<f8"),
                                      band_of.astype("<i8")),
            "grid_sha": _sha_arrays(np.asarray(grid["xs"], "<f8"), np.asarray(grid["ys"], "<f8")),
            "stations_json": (_file_id(config.STATIONS_JSON) if stas else None),
            "series": {s["code"]: _file_id(series_path(s)) for s in stas},
            "members": [m["code"] for m in (members or [])],
            "frozen_judges": config.frozen_sha("judges"),
            "code": config.sha256_source(os.path.abspath(__file__))}


def build(pkg, path=None, provider=None, write=True, members=None):
    """The judge table of a package (§2.8): geometry, inclusion, location.

    Parameters
    ----------
    pkg : CommonPackage
        Reads ``cfg``, ``t`` (record instants), ``full`` (keep, s_km,
        band_of, centers), the frame shape and the grid (:func:`grid_of`).
    path : str, optional
        Where to write the table (default :func:`judges_path`, the package
        copy; ``truth`` passes ``truth/<site>/judges.json``).
    provider : boundary provider, optional
        Cuts the boundary members to the gauges it holds.
    write : bool
        Write the JSON (atomically).
    members : list of dict, optional
        Boundary members (default :func:`boundary_members`).

    Returns
    -------
    list of dict
        One record per listed station, with its ``nbhd`` array.
    """
    cfg = pkg.cfg
    stas = stations(cfg.site)
    grid = grid_of(pkg)
    if members is None:
        members = boundary_members(cfg, provider)
    out = []
    for st in stas:
        series = read_series(st)
        geom = geometry(pkg, st, grid=grid)
        inc = inclusion(pkg, st, series, members, geom)
        rec = {**st, "location": location_of(st["code"]),
               "series_n": int(len(series)),
               "series_first": str(series.index[0]) if len(series) else None,
               "series_last": str(series.index[-1]) if len(series) else None,
               **geom, **inc,
               "mouth": bool(np.isfinite(geom["s_g_km"]) and geom["s_g_km"] < MOUTH_S_KM)}
        out.append(rec)
        log(f"[judges] {cfg.site} {st['code']}: "
            f"{'INCLUDED' if inc['included'] else 'excluded (' + ', '.join(inc['exclusion']) + ')'}; "
            f"k_g {geom['k_g']}, s_g {geom['s_g_km']:.2f} km, d_min {geom['d_min_km']:.2f} km, "
            f"R {geom['R_g_m']:.0f} m ({geom['n_keep_in_nbhd']} keep px), "
            f"coverage {inc['coverage']:.2f}, dry {inc['dry_share']:.3f}, nearest member "
            f"{inc['nearest_member']} {inc['nearest_member_km']:.1f} km"
            + (f"; flags {geom['flags']}" if geom["flags"] else ""))
    if write:
        save(path or judges_path(pkg), pkg, out, grid=grid, members=members, stas=stas)
    return out


def table_document(pkg, table, grid=None, members=None, stas=None):
    """The JSON document of a judge table (nbhd arrays replaced by their
    size and sha256; :func:`load` recomputes them)."""
    recs = []
    for r in table:
        rec = {k: v for k, v in r.items() if k != "nbhd"}
        recs.append(rec)
    locs = {}
    for r in table:
        locs.setdefault(r["location"], []).append(r["code"])
    headline = sorted({r["location"] for r in table if r["included"]})
    return {"format": JUDGES_FORMAT, "site": pkg.cfg.site, "res_m": int(pkg.cfg.res_m),
            "scenario": getattr(pkg.cfg, "scenario", None),
            "n_record": int(len(pkg.t)),
            "neighbourhood": config.frozen_plain("judges")["judges"]["neighbourhood"],
            "inclusion_rules": config.frozen_plain("judges")["judges"]["inclusion"],
            "boundary_members": list(members or []),
            "inputs": inputs_of(pkg, stas=stas, grid=grid, members=members),
            "locations": locs, "headline_locations": headline,
            "stations": recs}


def save(path, pkg, table, grid=None, members=None, stas=None):
    """Write a judge table as JSON (atomically)."""
    product.write_json(path, table_document(pkg, table, grid=grid, members=members, stas=stas))
    log(f"[judges] wrote {path} ({len(table)} stations)")
    return path


def read(path):
    """The JSON document of a judge table (no neighbourhood arrays)."""
    doc = product.read_json(path)
    if doc.get("format") != JUDGES_FORMAT:
        raise ValueError(f"{path}: not a {JUDGES_FORMAT} file")
    return doc


def load(path, grid):
    """A judge table with its neighbourhoods recomputed on ``grid`` and
    checked against the stored sha256."""
    doc = read(path)
    out = []
    for rec in doc["stations"]:
        rec = dict(rec)
        nb = nbhd_flat(grid, rec["gx"], rec["gy"], rec["R_g_m"])
        sha = config.sha256_bytes(nb.astype("<i8").tobytes())
        if sha != rec["nbhd_sha256"]:
            raise ValueError(f"{path}: the neighbourhood of {rec['code']} does not match "
                             "the grid it was built on")
        rec["nbhd"] = nb
        out.append(rec)
    return out


def load_for(pkg, path=None):
    """:func:`load` on the package's grid (default the package copy)."""
    return load(path or judges_path(pkg), grid_of(pkg))


def nb_flat(pkg, params=None, grid=None):
    """NB of the package domain (§2.5): every frame pixel within R_g of a
    listed station that lies inside the frame, R_g by the neighbourhood rule
    with the PACKAGE's parameters (FROZEN['package']['NB']), so the package
    key never depends on the score-side judges section.

    Returns
    -------
    (n,) int64, ascending unique frame flat indices.
    """
    p = config.FROZEN["package"]["NB"] if params is None else params
    grid = grid if grid is not None else grid_of(pkg)
    parts = []
    for st in stations(pkg.cfg.site):
        geom = geometry(pkg, st, params=p, grid=grid)
        if geom["in_frame"]:
            parts.append(geom["nbhd"])
    if not parts:
        return np.empty(0, np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64)


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION v2 (docs/validation_v2_spec.md §1.7, §6.1)
# ═════════════════════════════════════════════════════════════════════════════
#
# Added for v2 without changing anything above: the coverage rule per window,
# the headline set of §6.1 (a check, never an override of the rules), the
# development-judge list frozen in LOCK.json, the judge sets of the two
# lag sensitivities, and the providers' time-stamp conventions.

#: the 13 headline locations of §6.1 (the inclusion rules decide; this list
#: is what they are expected to give and is compared, never imposed)
HEADLINE = {
    "escalda": ("gatvanborssele", "terneuzen"),
    "wadden": ("terschelling.west", "harlingen.waddenzee"),
    "ems": ("emshoern_9340010", "eemshaven.waddenzee", "dukegat_3990020", "knock_3990010",
            "delfzijl", "emden_neue_seeschleuse_3970010", "pogum_3950020",
            "nieuwestatenzijl.dollard"),
    "ferrol": ("fer2",),
    "villaviciosa": (),
}

#: judges whose comparison influenced a MAREA parameter (§1.7). Harlingen is
#: required by the spec (TAU_BOUNDS was widened after it); the authors
#: complete the list before lock-A and LOCK.json freezes it.
DEVELOPMENT_JUDGES = (
    ("harlingen.waddenzee", "TAU_BOUNDS was widened after the Harlingen comparison (§1.7)"),
)
#: candidates the authors must confirm or reject before lock-A: the gauges
#: held out of the production boundaries as the notebooks' JUDGE_GAUGES
#: (config.HELD_OUT), whose comparisons the authors saw during development
DEVELOPMENT_CANDIDATES = (
    ("terneuzen", "held out of the Escalda production boundary (IOC trnz)"),
    ("delfzijl", "held out of the Ems production boundary (IOC delf)"),
    ("fer2", "held out of the Ferrol production boundary"),
)

#: providers' time-stamp conventions (§1.7): the averaging window (min) and
#: where in it the stamp sits; ``status`` says whether it is documented by the
#: provider (or by the project's own download script) or assumed pending the
#: metadata check of ``fetch judges``
TIMESTAMP_CONVENTIONS = {
    "RWS": {"averaging_min": 10.0, "stamp": "centre", "status": "documented",
            "source": ("RWS DD-API WATHTE, WaardeBepalingsMethode F007: the mean of the "
                       "previous 5 and the next 5 minutes (estuary_truth build_outputs.py "
                       "keeps F007 only)")},
    "WSV": {"averaging_min": 1.0, "stamp": "centre", "status": "assumed",
            "source": ("PEGELONLINE WASSERSTAND ROHDATEN, unchecked raw 1-min values; the "
                       "stamping is not stated in the download; treated as instantaneous")},
    "IOC": {"averaging_min": 1.0, "stamp": "centre", "status": "assumed",
            "source": ("IOC SLSMF raw 1-min record (majority sensor); the sensor's "
                       "averaging is not stated per station; treated as instantaneous")},
    "PdE": {"averaging_min": 5.0, "stamp": "centre", "status": "documented",
            "source": ("Puertos del Estado REDMAR 2 Hz samples averaged by "
                       "estuary_truth/scripts/pde_download.py into centred 5-min bins "
                       "[t - 2.5, t + 2.5) min")},
}
#: shift (in units of the averaging window) that centres a stamp
STAMP_SHIFT = {"start": 0.5, "centre": 0.0, "end": -0.5}


def development_list(extra=()):
    """The development judges (codes, sorted): :data:`DEVELOPMENT_JUDGES`
    plus the codes the authors add (``extra``, from LOCK.json)."""
    return sorted({c for c, _ in DEVELOPMENT_JUDGES} | {str(c) for c in extra})


def conventions_doc(conventions=None):
    """The conventions in force (JSON-ready)."""
    return {k: dict(v) for k, v in (conventions or TIMESTAMP_CONVENTIONS).items()}


def conventions_sha(conventions=None):
    """sha256 of :func:`conventions_doc` (enters the lag-truth key)."""
    return config.sha256_json(conventions_doc(conventions))


def read_conventions(path):
    """Conventions written by ``fetch judges`` (``{"providers": {...}}``),
    falling back to :data:`TIMESTAMP_CONVENTIONS` for a missing file or
    provider."""
    out = conventions_doc()
    if path and os.path.exists(path):
        doc = product.read_json(path)
        for k, v in (doc.get("providers") or {}).items():
            out[k] = dict(v)
    return out


def stamp_shift_min(rec, conventions=None):
    """Minutes to add to a station's stamps to put them at the centre of
    its averaging window (0 for centred or instantaneous values)."""
    conv = (conventions or TIMESTAMP_CONVENTIONS).get(str(rec.get("provider")))
    if conv is None:
        raise KeyError(f"no time-stamp convention for provider {rec.get('provider')!r} "
                       f"of {rec.get('code')}")
    if conv["stamp"] not in STAMP_SHIFT:
        raise ValueError(f"unknown stamp position {conv['stamp']!r}")
    return STAMP_SHIFT[conv["stamp"]] * float(conv["averaging_min"])


def half_window_min(rec, conventions=None):
    """Half the averaging window (min): the declared timing sensitivity
    shifts each series by +- this much (§1.7)."""
    conv = (conventions or TIMESTAMP_CONVENTIONS)[str(rec.get("provider"))]
    return 0.5 * float(conv["averaging_min"])


def shift_series(series, minutes):
    """The series with its stamps moved by ``minutes`` (values unchanged)."""
    if not minutes:
        return series
    out = series.copy()
    out.index = series.index + pd.Timedelta(minutes=float(minutes))
    return out


def centred(series, rec, conventions=None, shift_min=0.0):
    """The series centred with its provider's convention, plus the
    sensitivity shift ``shift_min`` (min)."""
    return shift_series(series, stamp_shift_min(rec, conventions) + float(shift_min))


def window_inclusion(rec, series, t_window, wid=None, members=None, rules=None,
                     conventions=None):
    """The inclusion rules of one station on one window's record (§1.7).

    ``in_frame`` and ``boundary_member`` do not depend on the window: they
    are read from the judge-table record (``in_frame``,
    ``nearest_member_km``) when it has them, else evaluated with
    ``members``. ``coverage`` and ``dry`` are evaluated on the gauge at the
    window's instants.

    ``conventions``: the time-stamp conventions to centre the series with
    first (:func:`centred`; "default" for :data:`TIMESTAMP_CONVENTIONS`),
    as the lag truths read it; None takes the series as given (a caller
    that centred it already).

    Returns
    -------
    dict
        :func:`inclusion_rules`' fields plus ``window``.
    """
    if conventions is not None:
        series = centred(series, rec, None if conventions == "default" else conventions)
    g = at_instants(series, t_window)
    out = inclusion_rules(rec, g, bool(rec.get("in_frame", True)), members or [], rules)
    if members is None and rec.get("nearest_member_km") is not None:
        r = INCLUSION if rules is None else rules
        km = float(rec["nearest_member_km"])
        out["nearest_member"], out["nearest_member_km"] = rec.get("nearest_member"), km
        out["passed"]["boundary_member"] = bool(km > float(r["boundary_member_min_km"]))
        out["included"] = all(out["passed"].values())
        out["exclusion"] = [k for k in RULES if not out["passed"][k]]
    out["window"] = wid
    return out


def coverage_by_window(table, t, windows, series_by_code=None, conventions=None):
    """``{code: {window: window_inclusion}}`` for every station of a table.

    Parameters
    ----------
    table : list of dict
        Judge-table records.
    t : DatetimeIndex
        The record instants (union record).
    windows : sequence of str
    series_by_code : dict, optional
        Pre-read series (default :func:`read_series` of each record).
    conventions : mapping or "default", optional
        Centre each series first (:func:`window_inclusion`).
    """
    from experiments.benchmark import settings

    t = pd.DatetimeIndex(t)
    out = {}
    for rec in table:
        s = (series_by_code or {}).get(rec["code"])
        s = read_series(rec) if s is None else s
        out[rec["code"]] = {}
        for wid in windows:
            w = settings.window(wid)
            tw = t[(t >= pd.Timestamp(w.start)) & (t < pd.Timestamp(w.end))]
            out[rec["code"]][wid] = window_inclusion(rec, s, tw, wid, conventions=conventions)
    return out


def judge_sets(table, window_incl=None, wid=None, development=None):
    """The location sets of the lag tables (§6.1): ``headline`` (every
    included location), ``without_mouth`` (s_g >= 3 km, v13 D10) and
    ``development_free`` (no development judge at the location).

    Parameters
    ----------
    table : list of dict
    window_incl : dict, optional
        :func:`coverage_by_window`; with ``wid``, inclusion is the window's.
    development : sequence, optional
        Codes (default :func:`development_list`).
    """
    dev = set(development_list() if development is None else development)
    inc = {}
    for r in table:
        ok = r.get("included", False)
        if window_incl is not None and wid is not None:
            ok = window_incl.get(r["code"], {}).get(wid, {}).get("included", False)
        inc[r["code"]] = ok
    locs = {}
    for r in table:
        locs.setdefault(r.get("location") or location_of(r["code"]), []).append(r)
    head, nomouth, devfree = [], [], []
    for loc, recs in sorted(locs.items()):
        if not any(inc[r["code"]] for r in recs):
            continue
        head.append(loc)
        if not any(r.get("mouth") for r in recs if inc[r["code"]]):
            nomouth.append(loc)
        if not any(r["code"] in dev for r in recs):
            devfree.append(loc)
    return {"headline": head, "without_mouth": nomouth, "development_free": devfree}


def check_headline(site, table):
    """The table's included locations against :data:`HEADLINE` (a report:
    the rules decide).

    Returns
    -------
    dict
        expected, got, missing, unexpected, equal.
    """
    want = sorted(HEADLINE.get(site, ()))
    got = sorted({r.get("location") or location_of(r["code"])
                  for r in table if r.get("included")})
    return {"expected": want, "got": got, "missing": sorted(set(want) - set(got)),
            "unexpected": sorted(set(got) - set(want)), "equal": want == got}
