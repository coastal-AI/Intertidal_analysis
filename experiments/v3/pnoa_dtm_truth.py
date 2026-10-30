# -*- coding: utf-8 -*-
"""The PNOA DTM truth of a validation v3 site on its 10 m grid, from one or several GeoTIFF sheets.

What
----
The elevation truth of one site and coverage of the design of 2026-10-29
(experiments/validation_v3_cantabrian_2026-10-29.md, "Sites and truth" and its amendment): the PNOA LiDAR terrain
model sheets the second author listed, MDT02 (2 m, 2nd coverage) or MDT50CM (0.5 m, 3rd coverage), in ETRS89 / UTM
29N (EPSG:25829, Foz and Eo) or 30N (EPSG:25830), averaged to the site's 10 m Sentinel-2 grid, whatever the grid's
CRS. The processing is validation v2's (experiments/benchmark/truth.py: ``pnoa_ground`` and ``aggregate_ground``,
called unchanged, as ``pnoa_truth`` calls them for its one sheet), run on the mosaic of the sheets and with the
water-patch rule as an area.

How
---
1. The sheets (:func:`lattice_offsets`): one CRS, one cell size, north-up, their origins whole cells apart (one
   lattice); otherwise the build stops. A truth is made of the sheets of one product and one zone.
2. The mosaic (:func:`mosaic_sheets`): the window of the grid's bounds in the sheets' CRS, on the first sheet's
   lattice, computed as ``pnoa_truth`` computes it for its sheet (``transform_bounds`` with 21 points per edge,
   offsets and lengths rounded). The sheets' cells are copied in, in the order given, where the mosaic has no value
   yet: where sheets overlap, the first sheet with a value is used, and each sheet's overlap with the earlier ones is
   counted and compared. Cells that no sheet covers are NaN, so a 10 m pixel across two sheets uses the cells of
   both, and a pixel partly outside every sheet fails the valid rule.
3. ``truth.pnoa_ground`` on the mosaic: water = 4-connected patches of at least :data:`PATCH_MIN_AREA_M2` = 200 m2
   whose values round to the same mm (``min_px`` = :func:`patch_min_cells`: 50 cells at 2 m, pnoa_ground's own
   default, and 800 at 0.5 m) and, when a flight level is given, every cell below it + 0.15 m. A patch cut by a sheet
   edge is whole in the mosaic; a patch cut by the grid's bounds counts its cells inside them, as in pnoa_truth.
4. ``truth.aggregate_ground`` to the grid: the area-weighted mean of the ground cells (GDAL's average, which also
   reprojects); a pixel needs more than 95 % of its area valid and at least 80 % of that ground.

With one 2 m sheet, z is pnoa_truth's z bit for bit (tests/test_v3_pnoa_dtm_truth.py).

Outputs of ``build``
--------------------
* ``<out>.tif``: float32 GeoTIFF on the grid (its CRS, transform and shape), NaN where there is no truth;
* ``<out>.json`` next to it: the sources with their sha256 (the sheets, the grid), the sheets (file, sheet name,
  product, zone, CRS, cell size, the cells each gave the mosaic, its overlap with the earlier sheets), the CRS and
  resolution of sheets and grid, the masks' shares of the valid cells, the grid's pixels with a value and why the
  others have none, the rule and its constants, the truth date and its rule, the code and library versions.

Run from the repository root::

    python -X utf8 -m experiments.v3.pnoa_dtm_truth build --sheets a.tif b.tif --grid grid.tif \\
        --truth-date 2020-07-15 --out data_v4/truth/pnoa_v3/<site>_<coverage>_10m.tif

``--flight-level`` (m, in the sheets' heights) adds the flight-level mask once a flight time is known. The truth date
is written with its rule, by default the design's provisional one (15 July of the coverage year); another date needs
``--truth-date-rule`` (the dated amendment that replaces it).

Memory: ``pnoa_ground`` peaks at about 120 bytes per mosaic cell when most cells are flat sea (measured on 9 million
cells of the Villaviciosa MDT02, about 60 % of them in patches); a 0.5 m mosaic holds 16 times the cells of a 2 m
one over the same grid.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
import time
from collections.abc import Mapping

import numpy as np

from experiments.benchmark import config, product, truth

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DESIGN = "experiments/validation_v3_cantabrian_2026-10-29.md"
FORMAT = "v3-pnoa-dtm-truth-1"
#: the water-patch rule as an area (the design's amendment of 2026-10-29): the 50 cells of the MDT02 rule at 2 m
PATCH_MIN_AREA_M2 = 200.0
#: the provisional truth date's rule (the design, "Truth date": the second author's rule)
TRUTH_DATE_RULE = "provisional: 15 July of the coverage year"
#: sheets share a lattice when their origins are whole cells apart within this fraction of a cell
LATTICE_TOL = 1e-3
#: a PNOA DTM file name: product, UTM zone, MTN50 sheet and its part, e.g. MDT02-ETRS89-HU30-0015-3-COB2.tif
#: (MDT02, 30, "0015-3") or MDT50CM-ETRS89-H30-0038-4-5-COB3.tif (MDT50CM, 30, "0038-4-5")
SHEET_RE = re.compile(r"(MDT\d+(?:CM)?)[-_ ]+(?:ETRS89|WGS84)[-_ ]+HU?(\d{2})[-_ ]+(\d{4})[-_ ]+(\d+(?:[-_]\d+)?)",
                      re.I)
DATUM = {
    "horizontal": "the sheets' ETRS89 / UTM zone; GDAL reprojects to the grid's CRS while it averages",
    "vertical": "orthometric heights (REDNAP), metres, as in the sheets",
}


def log(msg):
    print(msg, flush=True)


def rel(path):
    """``path`` relative to the repository root when it lies inside it, else absolute (forward slashes)."""
    p = os.path.abspath(os.fspath(path))
    try:
        r = os.path.relpath(p, ROOT)
    except ValueError:                                      # another drive
        r = p
    return (p if r.startswith("..") else r).replace("\\", "/")


# ── the rule ────────────────────────────────────────────────────────────────

def patch_min_cells(cell_w, cell_h=None, area_m2=PATCH_MIN_AREA_M2):
    """The fewest cells of a water patch: ``area_m2`` over the cell area, rounded up (a quotient within 1e-6 of a
    whole number is that number): 50 at 2 m, 800 at 0.5 m."""
    cell_h = cell_w if cell_h is None else cell_h
    n = float(area_m2) / abs(float(cell_w) * float(cell_h))
    k = round(n)
    return int(k) if abs(n - k) <= 1e-6 * max(n, 1.0) else int(math.ceil(n))


def ground_cells(src, cell_w, cell_h=None, flight_level=None):
    """The ground mask of the sheets' cells: ``truth.pnoa_ground`` with ``min_px`` from the area rule
    (:func:`patch_min_cells`); at 2 m that is pnoa_ground with its defaults.

    Returns
    -------
    pnoa_ground's dict: valid, water_patch, below_flight (outside the patches), ground (bool arrays), patches.
    """
    return truth.pnoa_ground(src, flight_level, min_px=patch_min_cells(cell_w, cell_h))


# ── sheets and grid ─────────────────────────────────────────────────────────

def parse_sheet_name(path):
    """{product, zone, sheet} of a PNOA DTM file name (:data:`SHEET_RE`), or None when the name does not parse."""
    m = SHEET_RE.search(os.path.basename(os.fspath(path)))
    if not m:
        return None
    prod, zone, mtn, part = m.groups()
    return {"product": prod.upper(), "zone": int(zone), "sheet": f"{mtn}-{part.replace('_', '-')}"}


def grid_of_cube(path):
    """(transform, crs, (H, W)) of a netCDF cube's 10 m grid: its x / y cell centres (regular, north-up) and the CRS
    of its ``crs`` variable (``crs_wkt`` or ``spatial_ref``, as ``pyintertidal.marea.extract`` reads it)."""
    import xarray as xr
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    with xr.open_dataset(path) as ds:
        x, y = np.asarray(ds["x"].values, np.float64), np.asarray(ds["y"].values, np.float64)
        attrs = ds["crs"].attrs
        wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    dx, dy = np.diff(x), np.diff(y)
    if not (len(x) > 1 and len(y) > 1 and np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])
            and dx[0] > 0 and dy[0] < 0):
        raise ValueError(f"{path}: x / y are not a regular north-up grid")
    if not wkt:
        raise ValueError(f"{path}: its crs variable has no crs_wkt / spatial_ref")
    a, e = float(dx[0]), float(dy[0])
    return Affine(a, 0.0, float(x[0]) - a / 2, 0.0, e, float(y[0]) - e / 2), CRS.from_wkt(wkt), (len(y), len(x))


def read_grid(grid):
    """(transform, crs, (H, W), path) of the target grid.

    ``grid`` is a GeoTIFF on it, a netCDF cube on it (:func:`grid_of_cube`), a (transform, crs, shape) tuple or a
    mapping with those keys (path None); the transform an Affine or its six coefficients, the CRS anything
    rasterio's CRS.from_user_input reads.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    if isinstance(grid, (str, os.PathLike)):
        if os.fspath(grid).lower().endswith(".nc"):
            return (*grid_of_cube(grid), os.fspath(grid))
        with rasterio.open(grid) as g:
            return g.transform, g.crs, (int(g.height), int(g.width)), os.fspath(grid)
    if isinstance(grid, Mapping):
        transform, crs, shape = grid["transform"], grid["crs"], grid["shape"]
    else:
        transform, crs, shape = grid
    if not isinstance(transform, Affine):
        transform = Affine(*[float(v) for v in list(transform)[:6]])
    return transform, CRS.from_user_input(crs), (int(shape[0]), int(shape[1])), None


def sheet_header(path):
    """The georeferencing of a sheet: path, crs, transform, shape, nodata, dtype, bounds. A sheet needs a CRS and a
    north-up grid."""
    import rasterio

    with rasterio.open(path) as s:
        head = {"path": os.fspath(path), "crs": s.crs, "transform": s.transform,
                "shape": (int(s.height), int(s.width)),
                "nodata": None if s.nodata is None else float(s.nodata), "dtype": str(s.dtypes[0]),
                "bounds": [float(v) for v in s.bounds]}
    t = head["transform"]
    if head["crs"] is None:
        raise ValueError(f"{path}: the sheet has no CRS")
    if not (t.b == 0 and t.d == 0 and t.a > 0 and t.e < 0):
        raise ValueError(f"{path}: not a north-up grid (transform {tuple(t)[:6]})")
    return head


def same_frame(a, b, bounds, tol_m=1e-3):
    """True when CRS ``a`` and ``b`` are equal, or give the same coordinates (within ``tol_m``) at the corners and
    centre of ``bounds`` (in ``a``). The CNIG labels sheets of one ETRS89 / UTM zone with two EPSG codes, 25830 and
    3042 (29N: 25829 and 3041), which differ only in the axis order they declare; rasterio's transforms read x as
    the easting under both, so their cells coincide."""
    if a == b:
        return True
    from rasterio.warp import transform

    w, s, e, n = bounds
    xs, ys = [w, e, w, e, (w + e) / 2], [s, s, n, n, (s + n) / 2]
    try:
        tx, ty = transform(a, b, xs, ys)
    except Exception:  # noqa: BLE001 - not transformable: not the same frame
        return False
    return max(abs(p - q) for p, q in zip(xs + ys, list(tx) + list(ty))) < tol_m


def lattice_offsets(heads):
    """(row, col) of every sheet's origin, in cells of the first sheet's lattice.

    The sheets must share the CRS (:func:`same_frame`) and the cell size, and their origins must be whole cells
    apart (within :data:`LATTICE_TOL` of a cell); otherwise ValueError.
    """
    t0, crs0, p0 = heads[0]["transform"], heads[0]["crs"], heads[0]["path"]
    out = []
    for h in heads:
        t = h["transform"]
        if not same_frame(h["crs"], crs0, h["bounds"]):
            raise ValueError(f"{h['path']}: CRS {h['crs'].to_string()} differs from {crs0.to_string()} of {p0}; "
                             "a truth is made of the sheets of one zone")
        if abs(t.a - t0.a) > 1e-9 * t0.a or abs(t.e - t0.e) > 1e-9 * abs(t0.e):
            raise ValueError(f"{h['path']}: cell {t.a} x {-t.e} m differs from {t0.a} x {-t0.e} m of {p0}")
        dc, dr = (t.c - t0.c) / t0.a, (t.f - t0.f) / t0.e
        if max(abs(dc - round(dc)), abs(dr - round(dr))) > LATTICE_TOL:
            raise ValueError(f"{h['path']}: origin {dr:.4f} rows, {dc:.4f} columns from that of {p0}, not on its "
                             "lattice")
        out.append((int(round(dr)), int(round(dc))))
    return out


def mosaic_sheets(sheets, grid):
    """The sheets' cells over the grid's bounds, on the first sheet's lattice (module docstring, step 2).

    Parameters
    ----------
    sheets : sequence of str
        The sheets, in priority order where they overlap.
    grid : see :func:`read_grid`

    Returns
    -------
    src (h, w) float32 (NaN where no sheet has a value), its transform and CRS, and per sheet a dict: path, file,
    product, zone, sheet (:func:`parse_sheet_name`; None when the name does not parse), crs, cell_m, shape, bounds,
    nodata, dtype, offset_cells (rows, columns from the first sheet's origin), cells_in_window, cells_valid,
    cells_used (copied into the mosaic), cells_overlap (valid where an earlier sheet gave a value),
    overlap_cells_differ, overlap_max_abs_diff_m (None without an overlap).
    """
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import transform as window_transform

    gt, gcrs, (H, W), _ = read_grid(grid)
    heads = [sheet_header(p) for p in sheets]
    offsets = lattice_offsets(heads)
    t0, crs = heads[0]["transform"], heads[0]["crs"]
    w = from_bounds(*transform_bounds(gcrs, crs, *array_bounds(H, W, gt), densify_pts=21), transform=t0)
    w = w.round_offsets().round_lengths()
    r0, c0, h, wd = int(w.row_off), int(w.col_off), int(w.height), int(w.width)
    src = np.full((h, wd), np.nan, np.float32)
    info = []
    for head, (dr, dc) in zip(heads, offsets):
        t = head["transform"]
        e = {"path": head["path"], "file": os.path.basename(head["path"]),
             **(parse_sheet_name(head["path"]) or {"product": None, "zone": None, "sheet": None}),
             "crs": head["crs"].to_string(), "cell_m": [float(t.a), float(-t.e)], "shape": list(head["shape"]),
             "bounds": head["bounds"], "nodata": head["nodata"], "dtype": head["dtype"], "offset_cells": [dr, dc],
             "cells_in_window": 0, "cells_valid": 0, "cells_used": 0, "cells_overlap": 0,
             "overlap_cells_differ": 0, "overlap_max_abs_diff_m": None}
        rs, cs = r0 - dr, c0 - dc                           # the window's first cell in this sheet
        ra, rb = max(rs, 0), min(rs + h, head["shape"][0])
        ca, cb = max(cs, 0), min(cs + wd, head["shape"][1])
        if rb > ra and cb > ca:
            with rasterio.open(head["path"]) as s:
                a = s.read(1, window=Window(ca, ra, cb - ca, rb - ra)).astype("float32")
            if head["nodata"] is not None:
                a[a == head["nodata"]] = np.nan
            sub = src[ra - rs:rb - rs, ca - cs:cb - cs]     # a view: filled in place
            v = np.isfinite(a)
            have = np.isfinite(sub)
            both, new = v & have, v & ~have
            if both.any():
                d = np.abs(a[both].astype(np.float64) - sub[both].astype(np.float64))
                e.update(cells_overlap=int(both.sum()), overlap_cells_differ=int((d > 0).sum()),
                         overlap_max_abs_diff_m=float(d.max()))
            sub[new] = a[new]
            e.update(cells_in_window=int(a.size), cells_valid=int(v.sum()), cells_used=int(new.sum()))
        info.append(e)
    return src, window_transform(w, t0), crs, info


# ── the truth ───────────────────────────────────────────────────────────────

def build_truth(sheets, grid, flight_level=None):
    """The PNOA DTM truth of one site and coverage on its grid (module docstring).

    Parameters
    ----------
    sheets : str or sequence of str
        The MDT02 or MDT50CM GeoTIFF sheets (one CRS, one cell size, one lattice; where they overlap, the first one
        with a value is used).
    grid : str, (transform, crs, shape) or mapping
        The site's 10 m grid (:func:`read_grid`): a GeoTIFF on it, or its georeferencing.
    flight_level : float, optional
        The water level at flight time (m, in the sheets' heights); None (or NaN): the patch mask only.

    Returns
    -------
    dict
        z (H, W) float32, NaN where there is no truth; cell_m (the sheets' cell, x and y), patch_min_area_m2,
        patch_min_cells; flight_level, flight_mask_applied; the mosaic's cells_valid, cells_water_patch,
        cells_below_flight (outside the patches), cells_ground and their shares of the valid cells
        (water_patch_share, flight_masked_share, ground_share: pnoa_truth's shares); patches (pnoa_ground's patch
        info); pixels (total, with_value, valid_share_too_low: at most 95 % of the area valid, ground_share_too_low:
        valid but under 80 % ground, outside_sheets: no valid cell); reprojected (the sheets' CRS is not the
        grid's); grid and mosaic (crs, transform, shape, cell_m; the grid's path, the mosaic's cells); sheets (the
        per-sheet info of :func:`mosaic_sheets`).
    """
    t_start = time.time()
    sheets = [os.fspath(sheets)] if isinstance(sheets, (str, os.PathLike)) else [os.fspath(p) for p in sheets]
    if not sheets:
        raise ValueError("no sheet given")
    keys = [os.path.normcase(os.path.abspath(p)) for p in sheets]
    if len(set(keys)) < len(keys):
        raise ValueError(f"a sheet is listed twice: {sheets}")
    gt, gcrs, gshape, gpath = read_grid(grid)
    crs0 = sheet_header(sheets[0])["crs"]
    if not crs0.is_projected:
        raise ValueError(f"{sheets[0]}: {crs0.to_string()} is not projected; the patch rule is an area in m2 "
                         "(use the ETRS89 / UTM sheets, not the WGS84 ones)")
    src, st, scrs, info = mosaic_sheets(sheets, (gt, gcrs, gshape))
    cell = [float(st.a), float(-st.e)]
    n_min = patch_min_cells(*cell)
    nv = int(np.isfinite(src).sum())
    log(f"[pnoa_v3] {len(sheets)} sheet(s) {scrs.to_string()} at {cell[0]:g} m -> grid {gshape[0]} x {gshape[1]} "
        f"at {gt.a:g} m {gcrs.to_string()}; mosaic {src.shape[0]:,} x {src.shape[1]:,} cells, {nv:,} valid")
    for e in info:
        log(f"  {e['file']}: {e['cells_used']:,} cells used, {e['cells_overlap']:,} in an earlier sheet's "
            f"overlap ({e['overlap_cells_differ']:,} differ"
            + (f", max {e['overlap_max_abs_diff_m']:.3f} m)" if e["overlap_max_abs_diff_m"] is not None else ")"))
    if not nv:
        raise ValueError(f"no sheet has a value inside the grid's bounds: {sheets}")
    gr = ground_cells(src, cell[0], cell[1], flight_level)
    z, gshare, vshare = truth.aggregate_ground(src, st, scrs, gr["ground"], gr["valid"], gt, gcrs, gshape)
    n = {k: int(gr[k].sum()) for k in ("water_patch", "below_flight", "ground")}
    mshape = [int(v) for v in src.shape]
    del gr["valid"], gr["water_patch"], gr["below_flight"], gr["ground"], src
    ok_valid = vshare > float(truth.PNOA_MIN_VALID)
    fin = np.isfinite(z)
    applied = flight_level is not None and bool(np.isfinite(flight_level))
    out = {
        "z": z.astype(np.float32),
        "cell_m": cell, "patch_min_area_m2": PATCH_MIN_AREA_M2, "patch_min_cells": n_min,
        "flight_level": None if flight_level is None else float(flight_level), "flight_mask_applied": applied,
        "cells_valid": nv, "cells_water_patch": n["water_patch"], "cells_below_flight": n["below_flight"],
        "cells_ground": n["ground"],
        "water_patch_share": float(n["water_patch"] / nv), "flight_masked_share": float(n["below_flight"] / nv),
        "ground_share": float(n["ground"] / nv),
        "patches": gr["patches"],
        "pixels": {"total": int(z.size), "with_value": int(fin.sum()),
                   "valid_share_too_low": int((~ok_valid).sum()),
                   "ground_share_too_low": int((ok_valid & ~fin).sum()),
                   "outside_sheets": int((vshare <= 0).sum())},
        "reprojected": bool(scrs != gcrs),
        "grid": {"path": gpath, "crs": gcrs.to_string(), "transform": [float(v) for v in list(gt)[:6]],
                 "shape": [int(gshape[0]), int(gshape[1])], "cell_m": [float(gt.a), float(-gt.e)]},
        "mosaic": {"crs": scrs.to_string(), "transform": [float(v) for v in list(st)[:6]], "shape": mshape,
                   "cell_m": cell, "cells": mshape[0] * mshape[1]},
        "sheets": info,
    }
    log(f"[pnoa_v3] water patches {out['water_patch_share']:.2%} of the valid cells ({gr['patches']['n_patches']} "
        f"patches of >= {n_min} cells = {PATCH_MIN_AREA_M2:g} m2), below the flight level "
        f"{out['flight_masked_share']:.2%}{'' if applied else ' (no flight level)'}; "
        f"{out['pixels']['with_value']:,} of {out['pixels']['total']:,} pixels with a value "
        f"[{time.time() - t_start:.1f} s]")
    return out


# ── outputs ─────────────────────────────────────────────────────────────────

def _versions():
    import rasterio

    return {"numpy": np.__version__, "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
            "python": sys.version.split()[0]}


def build(sheets, grid, out, truth_date, truth_date_rule=TRUTH_DATE_RULE, flight_level=None):
    """:func:`build_truth`, written: ``out`` (float32 GeoTIFF on the grid, NaN without truth) and ``out`` with
    ``.json`` (module docstring, "Outputs"). Both through a temporary file. Returns the JSON record."""
    import rasterio

    dt.date.fromisoformat(str(truth_date))                  # a calendar date, or ValueError
    res = build_truth(sheets, grid, flight_level)
    gt, gcrs, gshape, gpath = read_grid(grid)
    z = res.pop("z")
    out = os.fspath(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp.tif"
    with rasterio.open(tmp, "w", driver="GTiff", height=gshape[0], width=gshape[1], count=1, dtype="float32",
                       crs=gcrs, transform=gt, nodata=float("nan"), compress="deflate") as d:
        d.write(z, 1)
    os.replace(tmp, out)
    sheets = [os.fspath(sheets)] if isinstance(sheets, (str, os.PathLike)) else [os.fspath(p) for p in sheets]
    rec = {
        "format": FORMAT, "design": DESIGN, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "truth_date": str(truth_date), "truth_date_rule": truth_date_rule,
        "output": {"path": rel(out), **product.file_entry(out)},
        "sources": {"sheets": [{"path": rel(p), **product.file_entry(p)} for p in sheets],
                    "grid": None if gpath is None else {"path": rel(gpath), **product.file_entry(gpath)}},
        "rule": {"patch_min_area_m2": PATCH_MIN_AREA_M2, "patch_tol_m": truth.PATCH_TOL_M,
                 "flight_margin_m": truth.FLIGHT_MARGIN_M, "min_valid_share": truth.PNOA_MIN_VALID,
                 "min_ground_share": truth.PNOA_MIN_GROUND,
                 "aggregation": "area-weighted mean of the ground cells (GDAL average), reprojected to the grid"},
        "datum": DATUM,
        "z_m": {k: (float(f(z[np.isfinite(z)])) if np.isfinite(z).any() else None)
                for k, f in (("min", np.min), ("median", np.median), ("max", np.max))},
        **res,
        "code": {"module": rel(__file__), "sha256": config.sha256_source(__file__),
                 "truth_py_sha256": config.sha256_source(truth.__file__), "versions": _versions()},
    }
    json_path = os.path.splitext(out)[0] + ".json"
    product.write_json(json_path, rec)
    log(f"[pnoa_v3] -> {rel(out)} and {rel(json_path)}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the truth of one site and coverage on its grid")
    b.add_argument("--sheets", nargs="+", required=True, help="the MDT02 / MDT50CM GeoTIFF sheets (ETRS89 / UTM)")
    b.add_argument("--grid", required=True, help="a GeoTIFF or the netCDF cube on the site's 10 m grid")
    b.add_argument("--truth-date", required=True, help="YYYY-MM-DD")
    b.add_argument("--truth-date-rule", default=TRUTH_DATE_RULE)
    b.add_argument("--flight-level", type=float, default=None, help="m, in the sheets' heights")
    b.add_argument("--out", required=True, help="the GeoTIFF to write (its .json beside it)")
    a = ap.parse_args(argv)
    build(a.sheets, a.grid, a.out, a.truth_date, a.truth_date_rule, a.flight_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
