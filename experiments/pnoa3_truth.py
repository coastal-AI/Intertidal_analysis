# -*- coding: utf-8 -*-
"""The Spanish sites' elevation truth from the PNOA-LiDAR 3rd coverage, on each site's 10 m frame.

What
----
The PNOA-LiDAR 3rd coverage (CNIG / IGN, series LIDA3) flew Asturias and Galicia between May 2024 and
February 2026. For both regions only the point clouds of the first classification level are published
(NPC01, the initial automatic classification; IGN status of September 2026), as 1 x 1 km LAZ tiles; no
MDT50cm (the coverage's terrain model) exists yet for the sheets of the two frames. This script

* ``tiles <site>``: lists the tiles that cover the site's intertidal zone (the production extraction's
  pixels ``inter``, with a 30 m margin) with their catalogue rows (``experiments/pnoa3_catalogue.csv``),
  into ``data_v4/truth/pnoa/<site>/tiles_to_download.csv``: what the first author downloads by hand
  (the Centro de Descargas refuses scripted clients);
* ``grid <site>``: reads every LAZ tile found in ``data_v4/truth/pnoa/<site>/`` and writes, on the
  site's 10 m frame (its CRS, transform and shape),

  - ``data_v4/truth/pnoa/<site>_pnoa3_10m.tif``: float32, NaN nodata, the MEDIAN z of the ground points
    (class 2, withheld points excluded) in each 10 m cell; NaN where a cell has fewer than 3;
  - ``<site>_pnoa3_10m_count.tif``: int32, the ground points of each cell (0: none);
  - ``<site>_pnoa3_10m_time.tif``: float64, the median acquisition instant of each cell's ground points,
    UTC POSIX seconds (NaN: no ground point), for the water level at flight time;
  - ``<site>_pnoa3_10m.json``: the datum, the licence, the rule, every tile (file, size, sha256, catalogue
    id, class histogram, flight days from the GPS time), the point and cell counts, the intertidal pixels
    with a value, the flight period of the tiles under the intertidal and the scored pixels, and the
    truth date;
  - ``data_v4/truth/pnoa/<site>/manifest.json``: the raw files (URL of the catalogue entry, size, sha256,
    tile, flight days).

The rule (fixed before any tile was read)
-----------------------------------------
* Only class 2 (ground) is used. NPC01 has the classes 1 (unclassified), 2, 3-5 (vegetation), 6 (building),
  7 (noise) and 12 (overlap) and no water class; a water class (9), if present, is counted and never used.
  Withheld points are excluded. Water does not return the LiDAR pulse as ground (the near-infrared pulse
  does not see through water), so intertidal cells that were under water at the flight have no ground
  point; NPC01's automatic classification may still call a water-surface return ground: the time raster
  lets the water level at the flight be compared with such cells (not done here).
* A cell's value is the median of the z of its ground points (the mean of the two middle ones for an even
  count, numpy's median); NaN with fewer than 3 points. Cells are the frame's 10 m cells; the frames are on
  multiples of 10 m, so every cell lies in one 1 x 1 km tile (a point on a tile edge that falls in a cell of
  the neighbouring tile is merged exactly: the raw points of the cells along every tile edge are kept for
  that).
* The coordinates are ETRS89 / UTM (EPSG:258zz) and are used as the frame's WGS 84 / UTM (EPSG:326zz)
  without a transformation (the two differ by less than 1 m, a tenth of a cell), as the production
  notebooks used the CNIG rasters. The heights are orthometric (H = h - N with the geoid EGM08-REDNAP:
  the REDNAP datum, mean sea level at Alicante).
* The flight days come from the points' GPS time (LAS 1.4, adjusted standard GPS time: GPS seconds - 1e9;
  UTC = GPS - 18 s). A tile's flight period is the first and last UTC day of all its points.
* The truth date (where the n-table windows end) is the LAST flight day of the tiles that cover the scored
  pixels: the intertidal pixels (the extraction's keep) with a value inside the production no-delay
  product's range +- 0.25 m (the notebook's F). On that day the whole scored truth exists, and it is the
  convention of the Dutch rows (a Vaklodingen cell is dated to the end of its survey year).

Run (laptop venv; reading LAZ needs ``laspy`` with the ``lazrs`` backend, which the server venv lacks)::

    python -X utf8 -m experiments.pnoa3_truth tiles villaviciosa
    python -X utf8 -m experiments.pnoa3_truth grid villaviciosa

Tiles are cached one by one in ``data_v4/truth/pnoa/<site>/cache/`` (keyed by the LAZ sha256), so a site can
be gridded in batches when the disk holds only part of its tiles: the merge uses every cached tile.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = "data_v4/truth/pnoa"
CATALOGUE = "experiments/pnoa3_catalogue.csv"
DETAIL_URL = "https://centrodedescargas.cnig.es/CentroDescargas/detalleArchivo?sec={sec}"
SERIES_URL = "https://centrodedescargas.cnig.es/CentroDescargas/lidar-tercera-cobertura"
FORMAT = "pnoa3-truth-1"

CELL_M = 10.0
MIN_POINTS = 3            # fewer ground points in a cell: NaN
GROUND = 2                # ASPRS class 2
WATER = 9                 # ASPRS class 9: counted, never used
RANGE_PAD_M = 0.25        # the notebook's pad around the production no-delay range (its F)
MARGIN_PX = 3             # the tiles listed for download: the intertidal pixels dilated by 30 m
BORDER_M = 10.0           # raw points kept for the cells whose centre is within this distance of a tile edge
GPS_EPOCH_UNIX = 315964800.0        # 1980-01-06T00:00:00Z
GPS_UTC_LEAP_S = 18.0               # GPS - UTC since 2017-01-01
CHUNK = 4_000_000

SITES = {
    "villaviciosa": dict(grid=("tif", "products_villaviciosa/elevation.tif"),
                         extract="products_marea/marea_extract.npz",
                         nodelay=("products_marea/external_products.npz", "no_delay"),
                         zone=30, region="AST", sheets=["0015", "0030"]),
    "ferrol": dict(grid=("nc", "ndwi_cube_ferrol_2023-2025_10m.nc"),
                   extract="products_ferrol/marea_extract.npz",
                   nodelay=("products_ferrol/comparison_eot20_g1/products.npz", "z_plain"),
                   zone=29, region="GAL", sheets=["0021", "0022", "0006", "0007"]),
}

DATUM = {
    "horizontal": ("ETRS89 / UTM zone {zone}N (EPSG:258{zone}), used as the frame's WGS 84 / UTM zone {zone}N "
                   "(EPSG:326{zone}) without a transformation (they differ by less than 1 m)"),
    "vertical": ("orthometric heights H = h - N, geoid model EGM08-REDNAP (REDNAP vertical datum: mean sea level "
                 "at the Alicante tide gauge), metres"),
}
SOURCE = {
    "product": "PNOA-LiDAR, 3rd coverage (2022-2025), classified point clouds NPC01 (initial automatic "
               "classification), 1 x 1 km LAZ tiles, 5 points/m2, RGBI-coloured",
    "producer": "Instituto Geografico Nacional (IGN) / Centro Nacional de Informacion Geografica (CNIG)",
    "series": "LIDA3",
    "catalogue": SERIES_URL,
    "licence": ("Orden FOM/2807/2015: free use under the IGN licence compatible with CC BY 4.0, which obliges "
                "the user to acknowledge and cite the origin and ownership of the data"),
    "attribution": "LiDAR-PNOA 3rd coverage (2024) CC-BY 4.0 scne.es (IGN / CNIG); check the wording with the "
                   "CNIG selector 'Como mencionar el origen y propiedad de los datos'",
    "classes_npc01": "1 unclassified, 2 ground, 3-5 vegetation, 6 building, 7 noise, 12 overlap (no water class)",
}

TILE_RE = re.compile(r"PNOA[-_](\d{4})[-_]([A-Z]+)[-_](\d{3})-(\d{4})[-_]H(\d{2})[-_](NPC\d{2})", re.I)


def log(msg):
    print(msg, flush=True)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def rel(path):
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


# ── names, tiles and the frame ──────────────────────────────────────────────

def parse_tile_name(name):
    """{year, region, x_km, y_km, zone, level, key} of a PNOA-LiDAR tile name (hyphens or underscores, any
    extension case), or None. x_km, y_km: the tile's upper-left corner in km."""
    m = TILE_RE.search(os.path.basename(name))
    if not m:
        return None
    year, region, x, y, zone, level = m.groups()
    return dict(year=int(year), region=region.upper(), x_km=int(x), y_km=int(y), zone=int(zone),
                level=level.upper(), key=f"{int(x)}-{int(y)}")


def tile_key_of(x, y):
    """The key "<x_km>-<y_km>" (upper-left corner) of the 1 x 1 km tile holding the points x, y (arrays)."""
    return np.floor(np.asarray(x) / 1000.0).astype(int), np.ceil(np.asarray(y) / 1000.0).astype(int)


def tile_extent(x_km, y_km):
    """(xmin, ymin, xmax, ymax) of a tile from its upper-left corner in km (the 3rd coverage's naming)."""
    return x_km * 1000.0, (y_km - 1) * 1000.0, (x_km + 1) * 1000.0, y_km * 1000.0


def frame(site):
    """(transform, crs, (H, W), inter, keep) of the site's 10 m frame and the production extraction's pixels."""
    import rasterio
    P = SITES[site]
    kind, path = P["grid"]
    if kind == "tif":
        with rasterio.open(resolve(path)) as r:
            transform, crs, shape = r.transform, r.crs, (r.height, r.width)
    else:
        import xarray as xr
        from pyintertidal.cube import grid_from_dataset
        with xr.open_dataset(resolve(path)) as ds:       # x, y and crs only; the bands are not loaded
            transform, crs = grid_from_dataset(ds)
            shape = (ds.sizes["y"], ds.sizes["x"])
    with np.load(resolve(P["extract"])) as z:
        inter, keep = z["inter"].astype(bool), z["keep"]
    assert inter.shape == shape, f"{P['extract']}: inter {inter.shape} is not the frame {shape}"
    assert abs(transform.a - CELL_M) < 1e-9 and abs(transform.e + CELL_M) < 1e-9, "the frame is not 10 m"
    assert transform.c % CELL_M == 0 and transform.f % CELL_M == 0, "the frame is not on multiples of 10 m"
    return transform, crs, shape, inter, keep


def cell_centres(transform, rows, cols):
    return transform.c + CELL_M * (np.asarray(cols) + 0.5), transform.f - CELL_M * (np.asarray(rows) + 0.5)


def tiles_of_pixels(transform, shape, flat):
    """{key: number of pixels} of the tiles holding the frame pixels `flat` (flat indices)."""
    rows, cols = np.divmod(np.asarray(flat), shape[1])
    xk, yk = tile_key_of(*cell_centres(transform, rows, cols))
    keys, n = np.unique(np.char.add(np.char.add(xk.astype(str), "-"), yk.astype(str)), return_counts=True)
    return dict(zip(keys.tolist(), n.tolist()))


def read_catalogue(path=CATALOGUE):
    import pandas as pd
    cat = pd.read_csv(resolve(path), comment="#", dtype={"sheet": str})
    info = [parse_tile_name(n) for n in cat["name"]]
    cat["key"] = [i["key"] for i in info]
    cat["zone"] = [i["zone"] for i in info]
    return cat


def needed_tiles(site, margin_px=MARGIN_PX):
    """[(key, intertidal pixels in the tile)] of the tiles under the intertidal pixels dilated by margin_px."""
    from scipy.ndimage import binary_dilation
    transform, _, shape, inter, _ = frame(site)
    wide = binary_dilation(inter, iterations=margin_px) if margin_px else inter
    rows, cols = np.nonzero(wide)
    xc, yc = cell_centres(transform, rows, cols)
    keys = set()
    for dx in (-CELL_M / 2, CELL_M / 2 - 1e-6):          # the cells' corners: a cell touching a tile counts it
        for dy in (-CELL_M / 2 + 1e-6, CELL_M / 2):
            xk, yk = tile_key_of(xc + dx, yc + dy)
            keys |= set(f"{a}-{b}" for a, b in zip(xk, yk))
    inside = tiles_of_pixels(transform, shape, np.flatnonzero(inter))
    return sorted(((k, inside.get(k, 0)) for k in keys), key=lambda kv: tuple(map(int, kv[0].split("-"))))


def cmd_tiles(site):
    """The download list of a site: data_v4/truth/pnoa/<site>/tiles_to_download.csv."""
    import pandas as pd
    P = SITES[site]
    cat = read_catalogue()
    cat = cat[cat.zone == P["zone"]].drop_duplicates("key").set_index("key")
    rows = []
    for key, n_inter in needed_tiles(site):
        r = dict(key=key, intertidal_px=n_inter)
        if key in cat.index:
            c = cat.loc[key]
            r.update(name=c["name"], sheet=c["sheet"], size_mb=float(c["size_mb"]), sec=int(c["sec"]),
                     url=DETAIL_URL.format(sec=int(c["sec"])), listed=True)
        else:
            r.update(name=f"PNOA-2024-{P['region']}-{key}-H{P['zone']}-NPC01.LAZ (not in the listings read)",
                     listed=False)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["sec"] = df["sec"].astype("Int64")                 # an id, not a float (the unlisted rows have none)
    out = resolve(f"{OUT_ROOT}/{site}/tiles_to_download.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    listed = df[df.listed]
    log(f"[{site}] {len(df)} tiles under the intertidal pixels (+{MARGIN_PX * 10:.0f} m): {len(listed)} listed, "
        f"{listed.size_mb.sum():,.0f} MB; {int((~df.listed).sum())} not in the listings read "
        f"({int(df[~df.listed].intertidal_px.sum()):,} intertidal px) -> {rel(out)}")
    return df


# ── one tile ────────────────────────────────────────────────────────────────

def gps_to_unix(t, adjusted=True):
    """UTC POSIX seconds of LAS GPS times (adjusted standard GPS time: GPS seconds - 1e9)."""
    if not adjusted:
        raise ValueError("GPS week time: the points carry no date")
    return np.asarray(t, np.float64) + 1e9 + GPS_EPOCH_UNIX - GPS_UTC_LEAP_S


def unix_to_iso(s):
    return dt.datetime.fromtimestamp(float(s), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def unix_day(s):
    return dt.datetime.fromtimestamp(float(s), dt.timezone.utc).strftime("%Y-%m-%d")


def group_median(cell, v):
    """(cells, counts, medians) of v grouped by cell; the median is numpy's (mean of the two middle values for
    an even count)."""
    if not len(cell):
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float64)
    o = np.lexsort((v, cell))
    cs, vs = cell[o], v[o]
    u, start, cnt = np.unique(cs, return_index=True, return_counts=True)
    hi = start + cnt // 2
    lo = np.where(cnt % 2 == 1, hi, hi - 1)
    return u, cnt, (vs[lo] + vs[hi]) / 2.0


def read_tile(path, transform, shape):
    """Every ground point of a LAZ tile that falls in the frame, as frame cells, and the tile's statistics."""
    import laspy
    info = parse_tile_name(path) or {}
    H, W = shape
    x0, y0 = transform.c, transform.f
    hist = np.zeros(256, np.int64)
    cells, zs, ts = [], [], []
    t_lo, t_hi, days = np.inf, -np.inf, {}
    n_pts = n_withheld = n_overlap_flag = n_ground = n_ground_frame = 0
    with laspy.open(path) as f:
        hdr = f.header
        adjusted = int(hdr.global_encoding.gps_time_type) == 1
        dims = set(hdr.point_format.dimension_names)
        for pts in f.chunk_iterator(CHUNK):
            cls = np.asarray(pts.classification, np.int64)
            n = len(cls)
            n_pts += n
            hist += np.bincount(cls, minlength=256)[:256]
            withheld = np.asarray(pts.withheld, bool) if "withheld" in dims else np.zeros(n, bool)
            n_withheld += int(withheld.sum())
            if "overlap" in dims:
                n_overlap_flag += int(np.asarray(pts.overlap, bool).sum())
            if adjusted:                                  # GPS week time carries no date: no flight days
                t = gps_to_unix(pts.gps_time)
                t_lo, t_hi = min(t_lo, float(t.min())), max(t_hi, float(t.max()))
                d, c = np.unique(np.floor(t / 86400.0).astype(np.int64), return_counts=True)
                for di, ci in zip(d.tolist(), c.tolist()):
                    days[di] = days.get(di, 0) + ci
            else:
                t = np.full(n, np.nan)
            g = (cls == GROUND) & ~withheld
            n_ground += int(g.sum())
            x, y, z = np.asarray(pts.x)[g], np.asarray(pts.y)[g], np.asarray(pts.z)[g]
            col = np.floor((x - x0) / CELL_M).astype(np.int64)
            row = np.floor((y0 - y) / CELL_M).astype(np.int64)
            inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
            n_ground_frame += int(inside.sum())
            cells.append(row[inside] * W + col[inside])
            zs.append(np.asarray(z[inside], np.float64))
            ts.append(t[g][inside])
        crs = None
        try:
            c = hdr.parse_crs()
            crs = c.to_string() if c is not None else None
        except Exception:                                 # a VLR laspy cannot parse: recorded as unknown
            crs = None
        dated = bool(n_pts) and np.isfinite(t_lo)
        stats = dict(
            file=os.path.basename(path), key=info.get("key"),
            year=info.get("year"), level=info.get("level"), las_version=str(hdr.version),
            point_format=int(hdr.point_format.id), crs=crs, header_points=int(hdr.point_count),
            bounds=[float(v) for v in (*hdr.mins, *hdr.maxs)], gps_time_adjusted=adjusted,
            points=n_pts, withheld=n_withheld, overlap_flag=n_overlap_flag, ground=n_ground,
            ground_in_frame=n_ground_frame, water_class_points=int(hist[WATER]),
            classes={str(k): int(v) for k, v in enumerate(hist) if v},
            first_utc=unix_to_iso(t_lo) if dated else None, last_utc=unix_to_iso(t_hi) if dated else None,
            flight_days={dt.datetime.fromtimestamp(d * 86400.0, dt.timezone.utc).strftime("%Y-%m-%d"): n
                         for d, n in sorted(days.items())})
    cat = lambda a, dtype: np.concatenate(a).astype(dtype) if a else np.array([], dtype)
    return cat(cells, np.int64), cat(zs, np.float64), cat(ts, np.float64), stats


def grid_tile(path, transform, shape):
    """A tile's per-cell result: (cells, count, median z, median time), the raw points of its edge cells, and
    the statistics (cells owned by another tile, counted)."""
    cell, z, t, stats = read_tile(path, transform, shape)
    u, cnt, zmed = group_median(cell, z)
    _, _, tmed = group_median(cell, t)
    rows, cols = np.divmod(u, shape[1])
    xc, yc = cell_centres(transform, rows, cols)
    info = parse_tile_name(path)
    if info:
        xmin, ymin, xmax, ymax = tile_extent(info["x_km"], info["y_km"])
        edge = ((xc - xmin < BORDER_M) | (xmax - xc < BORDER_M) | (yc - ymin < BORDER_M) | (ymax - yc < BORDER_M))
        outside = (xc < xmin) | (xc > xmax) | (yc < ymin) | (yc > ymax)
    else:                                                 # no tile name: every cell is an edge cell
        edge, outside = np.ones(len(u), bool), np.zeros(len(u), bool)
    stats["cells_with_ground"] = int(len(u))
    stats["cells_outside_tile"] = int(outside.sum())
    e = np.isin(cell, u[edge])
    return dict(cell=u, count=cnt, z=zmed, t=tmed, edge_cell=cell[e], edge_z=z[e], edge_t=t[e]), stats


def cached_tile(path, sha, transform, shape, cache_dir):
    """grid_tile, cached by the LAZ sha256 (a re-run or a later batch reads the cache)."""
    os.makedirs(cache_dir, exist_ok=True)
    npz = os.path.join(cache_dir, os.path.basename(path).rsplit(".", 1)[0] + ".npz")
    if os.path.exists(npz):
        with np.load(npz, allow_pickle=False) as c:
            stats = json.loads(str(c["stats"]))
            if stats.get("sha256") == sha and stats.get("frame") == [*transform[:6], *shape]:
                return {k: c[k] for k in c.files if k != "stats"}, stats
    res, stats = grid_tile(path, transform, shape)
    stats.update(sha256=sha, size_bytes=os.path.getsize(path), frame=[*transform[:6], *shape])
    np.savez_compressed(npz, stats=np.array(json.dumps(stats)), **res)
    return res, stats


def merge_tiles(results, shape):
    """The frame's count, median z and median time from the per-tile results; a cell with points from more
    than one tile (a point on a tile edge) is recomputed from the raw points the tiles kept for their edge
    cells."""
    H, W = shape
    count = np.zeros(H * W, np.int64)
    zmed = np.full(H * W, np.nan)
    tmed = np.full(H * W, np.nan)
    allc = np.concatenate([r["cell"] for r in results]) if results else np.array([], np.int64)
    u, n = np.unique(allc, return_counts=True)
    shared = u[n > 1]
    for r in results:
        own = ~np.isin(r["cell"], shared)
        count[r["cell"][own]] = r["count"][own]
        zmed[r["cell"][own]] = r["z"][own]
        tmed[r["cell"][own]] = r["t"][own]
    if len(shared):
        for r in results:
            missing = np.setdiff1d(np.intersect1d(r["cell"], shared), r["edge_cell"])
            assert not len(missing), f"{len(missing)} cells shared between tiles lie far from a tile edge"
        ec = np.concatenate([r["edge_cell"] for r in results])
        ez = np.concatenate([r["edge_z"] for r in results])
        et = np.concatenate([r["edge_t"] for r in results])
        k = np.isin(ec, shared)
        c2, n2, z2 = group_median(ec[k], ez[k])
        _, _, t2 = group_median(ec[k], et[k])
        count[c2], zmed[c2], tmed[c2] = n2, z2, t2
    z = np.where(count >= MIN_POINTS, zmed, np.nan)
    return count.reshape(H, W), z.reshape(H, W), tmed.reshape(H, W), int(len(shared))


def write_tif(path, arr, transform, crs, dtype, nodata):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata, compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256) as d:
        d.write(arr.astype(dtype), 1)


def flight_period(tile_stats, keys):
    t = [tile_stats[k] for k in sorted(keys) if k in tile_stats and tile_stats[k].get("first_utc")]
    if not t:
        return None
    return dict(first=min(s["first_utc"] for s in t)[:10], last=max(s["last_utc"] for s in t)[:10],
                tiles=sorted(s["key"] for s in t),
                days=sorted({d for s in t for d in s["flight_days"]}))


def cmd_grid(site, laz_dir=None, out_root=OUT_ROOT):
    """Grid every LAZ tile of data_v4/truth/pnoa/<site>/ on the site's frame (module docstring)."""
    import laspy
    P = SITES[site]
    transform, crs, shape, inter, keep = frame(site)
    H, W = shape
    laz_dir = resolve(laz_dir or f"{out_root}/{site}")
    files = sorted(set(glob.glob(os.path.join(laz_dir, "*.laz")) + glob.glob(os.path.join(laz_dir, "*.LAZ"))))
    cache_dir = os.path.join(laz_dir, "cache")
    cached = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    cat = read_catalogue()
    cat = cat[cat.zone == P["zone"]].drop_duplicates("key").set_index("key")
    log(f"[{site}] frame {H}x{W} @ 10 m {crs.to_string()}; {len(files)} LAZ files in {rel(laz_dir)}, "
        f"{len(cached)} cached tiles")

    results, tile_stats, manifest = [], {}, []
    for k, path in enumerate(files):
        info = parse_tile_name(path)
        assert info and info["zone"] == P["zone"], f"{path}: not a PNOA tile of zone {P['zone']}"
        assert info["key"] not in tile_stats, f"two files for the tile {info['key']}: {path}"
        sha = sha256_file(path)
        res, stats = cached_tile(path, sha, transform, shape, cache_dir)
        stats["raw_file_present"] = True
        results.append(res)
        tile_stats[stats["key"]] = stats
        log(f"  [{k + 1}/{len(files)}] {stats['file']}: {stats['points']:,} points, {stats['ground_in_frame']:,} "
            f"ground in the frame, {stats['cells_with_ground']:,} cells; flown {stats['first_utc']} -> "
            f"{stats['last_utc']}")
    seen = {s["sha256"] for s in tile_stats.values()}
    for npz in cached:                                    # tiles gridded in an earlier batch, raw file moved away
        with np.load(npz, allow_pickle=False) as c:
            stats = json.loads(str(c["stats"]))
            if stats["sha256"] in seen or stats.get("frame") != [*transform[:6], *shape]:
                continue
            if stats["key"] in tile_stats:                # an older cache of a tile whose file was replaced
                log(f"  skipped {rel(npz)}: an older cache of {stats['key']}")
                continue
            results.append({kk: c[kk] for kk in c.files if kk != "stats"})
            stats["raw_file_present"] = False
            tile_stats[stats["key"]] = stats
            log(f"  cached only: {stats['file']} (sha256 {stats['sha256'][:12]})")
    for key, s in sorted(tile_stats.items()):
        c = cat.loc[key] if key in cat.index else None
        s["catalogue_name"] = None if c is None else c["name"]
        s["sec"] = None if c is None else int(c["sec"])
        s["url"] = None if c is None else DETAIL_URL.format(sec=int(c["sec"]))
        s["listed_size_mb"] = None if c is None else float(c["size_mb"])
        manifest.append({k: s.get(k) for k in ("file", "key", "catalogue_name", "url", "sec", "size_bytes",
                                                 "listed_size_mb", "sha256", "first_utc", "last_utc",
                                                 "flight_days", "raw_file_present")})
    write_json(os.path.join(laz_dir, "manifest.json"),
               dict(format=FORMAT + "-manifest", site=site, source=SOURCE, created_utc=unix_to_iso(
                   dt.datetime.now(dt.timezone.utc).timestamp()), files=manifest))

    count, z, tmed, n_shared = merge_tiles(results, shape)
    stem = os.path.join(resolve(out_root), f"{site}_pnoa3_10m")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    write_tif(stem + ".tif", z, transform, crs, "float32", np.nan)
    write_tif(stem + "_count.tif", count, transform, crs, "int32", None)
    write_tif(stem + "_time.tif", tmed, transform, crs, "float64", np.nan)

    zf, cf = z.ravel(), count.ravel()
    have = np.isfinite(zf[keep])
    path_nd, key_nd = P["nodelay"]
    with np.load(resolve(path_nd)) as zz:
        z_nodelay = zz[key_nd]
    lo, hi = float(np.nanmin(z_nodelay)) - RANGE_PAD_M, float(np.nanmax(z_nodelay)) + RANGE_PAD_M
    scored = keep[have & (zf[keep] >= lo) & (zf[keep] <= hi)]
    needed = dict(needed_tiles(site, margin_px=0))
    missing = {k: n for k, n in needed.items() if k not in tile_stats and n > 0}
    fp_inter = flight_period(tile_stats, tiles_of_pixels(transform, shape, keep[have]))
    fp_scored = flight_period(tile_stats, tiles_of_pixels(transform, shape, scored))
    per_tile_inter = tiles_of_pixels(transform, shape, keep)
    per_tile_value = tiles_of_pixels(transform, shape, keep[have])
    for key, s in tile_stats.items():
        s["intertidal_px"] = int(per_tile_inter.get(key, 0))
        s["intertidal_px_with_value"] = int(per_tile_value.get(key, 0))
    tq = tmed.ravel()[keep[have]]
    code = os.path.abspath(__file__)
    doc = dict(
        format=FORMAT, site=site, created_utc=unix_to_iso(dt.datetime.now(dt.timezone.utc).timestamp()),
        code=dict(path=rel(code), sha256=sha256_file(code), laspy=laspy.__version__,
                  numpy=np.__version__, python=sys.version.split()[0]),
        source=SOURCE, datum={k: v.format(zone=P["zone"]) for k, v in DATUM.items()},
        frame=dict(crs=crs.to_string(), transform=list(transform[:6]), shape=[H, W], cell_m=CELL_M,
                   grid=P["grid"][1]),
        rule=(f"median z of the class-{GROUND} (ground) points of each 10 m cell, withheld points excluded; NaN "
              f"where a cell has fewer than {MIN_POINTS} ground points; no other class is used (a water class, "
              f"{WATER}, is counted and never used); ETRS89 / UTM used as the frame's WGS 84 / UTM"),
        files=dict(elevation=rel(stem + ".tif"), count=rel(stem + "_count.tif"), time=rel(stem + "_time.tif"),
                   manifest=rel(os.path.join(laz_dir, "manifest.json"))),
        totals=dict(tiles=len(tile_stats), points=sum(s["points"] for s in tile_stats.values()),
                    ground_points=sum(s["ground"] for s in tile_stats.values()),
                    ground_points_in_frame=sum(s["ground_in_frame"] for s in tile_stats.values()),
                    water_class_points=sum(s["water_class_points"] for s in tile_stats.values()),
                    cells_with_ground=int((cf > 0).sum()), cells_with_value=int(np.isfinite(zf).sum()),
                    cells_below_min_points=int(((cf > 0) & (cf < MIN_POINTS)).sum()),
                    cells_shared_between_tiles=n_shared),
        intertidal=dict(
            pixels=int(len(keep)), with_value=int(have.sum()), share_with_value=float(have.mean()),
            with_any_ground_point=int((cf[keep] > 0).sum()),
            in_tiles_gridded=int(sum(per_tile_inter.get(k, 0) for k in tile_stats)),
            in_missing_tiles=int(sum(missing.values())),
            ground_points_per_cell_median=float(np.median(cf[keep][have])) if have.any() else None,
            elevation_m=dict(zip(("p01", "p50", "p99"), np.nanpercentile(zf[keep], [1, 50, 99]).tolist()))
            if have.any() else None,
            flight_utc=dict(first=unix_to_iso(np.nanmin(tq)), last=unix_to_iso(np.nanmax(tq)))
            if np.isfinite(tq).any() else None,
            note="the extraction's pixels (keep, = inter); LiDAR does not see through water: cells under water "
                 "at the flight have no ground point"),
        scored=dict(rule=f"intertidal pixels with a value inside the production no-delay range +- {RANGE_PAD_M} m "
                         "(the n-table notebook's F)", nodelay=f"{path_nd}:{key_nd}", range_m=[lo, hi],
                    pixels=int(len(scored))),
        flight=dict(intertidal_with_value=fp_inter, scored=fp_scored),
        truth_date=fp_scored["last"] if fp_scored else None,
        truth_date_rule=("the last flight day (UTC) of the tiles that cover the scored pixels: the first day on "
                         "which the whole scored truth exists, as a Dutch row is dated to the end of its survey "
                         "year; the windows hold the images on or before it"),
        missing_tiles=[dict(key=k, intertidal_px=n) for k, n in sorted(missing.items())],
        tiles=[tile_stats[k] for k in sorted(tile_stats, key=lambda kk: tuple(map(int, kk.split("-"))))],
    )
    write_json(stem + ".json", doc)
    log(f"[{site}] {doc['totals']['cells_with_value']:,} cells with a value; intertidal pixels with a value "
        f"{have.sum():,} of {len(keep):,} ({100 * have.mean():.1f} %), scored {len(scored):,}; "
        f"{len(missing)} needed tiles missing ({sum(missing.values()):,} intertidal px); "
        f"flight of the scored tiles {fp_scored and fp_scored['first']} -> {fp_scored and fp_scored['last']}; "
        f"truth date {doc['truth_date']} -> {rel(stem)}.tif / .json")
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["tiles", "grid"])
    p.add_argument("site", choices=sorted(SITES))
    p.add_argument("--laz-dir", help="where the LAZ tiles are (default data_v4/truth/pnoa/<site>/)")
    a = p.parse_args(argv)
    sys.path.insert(0, ROOT)
    if a.command == "tiles":
        cmd_tiles(a.site)
    else:
        cmd_grid(a.site, a.laz_dir)


if __name__ == "__main__":
    main()
