"""
mosaic.py — Stitch the tiles back into one map
===============================================

Regional mapping runs one AOI at a time (see :meth:`pyintertidal.aoi.AOI.tile`)
and then has to put the pieces back together. This module merges the
per-tile GeoTIFFs into a single raster, and — more importantly — checks that
the merge is legitimate.

Three failure modes it guards against
-------------------------------------
1. **Grid mismatch.** Tiles processed with different resolutions or CRSs
   cannot be merged meaningfully. :func:`check_tiles` refuses instead of
   silently resampling.
2. **Datum drift.** Each tile predicts its tide at its own centroid, so
   neighbouring tiles can sit on slightly different vertical references and
   show a visible step at the seam. :func:`seam_offsets` measures those
   steps, and :func:`merge_tiles` can level them.
3. **Double counting.** With overlapping halos, naive summing counts the
   overlap twice. The merge averages overlaps by default.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge


def check_tiles(paths, verbose=True):
    """Verify that a set of tiles can be merged, and report their layout.

    Returns a dict with the common CRS and resolution, the union bounds, and
    a list of any problems found (different CRS, different pixel size).
    """
    problems, metas = [], []
    for path in paths:
        with rasterio.open(path) as src:
            metas.append({"path": path, "crs": src.crs,
                          "res": (abs(src.transform.a), abs(src.transform.e)),
                          "bounds": src.bounds, "shape": src.shape,
                          "dtype": src.dtypes[0]})
    if not metas:
        raise ValueError("no tiles given")

    crs0, res0 = metas[0]["crs"], metas[0]["res"]
    for m in metas[1:]:
        if m["crs"] != crs0:
            problems.append(f"{os.path.basename(m['path'])}: CRS {m['crs']} "
                            f"≠ {crs0}")
        if not np.allclose(m["res"], res0, rtol=1e-6):
            problems.append(f"{os.path.basename(m['path'])}: resolution "
                            f"{m['res']} ≠ {res0}")

    bounds = [m["bounds"] for m in metas]
    union = (min(b.left for b in bounds), min(b.bottom for b in bounds),
             max(b.right for b in bounds), max(b.top for b in bounds))
    info = {"n_tiles": len(metas), "crs": crs0, "resolution": res0,
            "bounds": union, "problems": problems, "tiles": metas}
    if verbose:
        print(f"[mosaic] {len(metas)} tiles · {res0[0]:g} m · {crs0}")
        print(f"[mosaic] extent {(union[2] - union[0]) / 1000:.1f} × "
              f"{(union[3] - union[1]) / 1000:.1f} km")
        for problem in problems:
            print(f"[mosaic] PROBLEM {problem}")
    return info


def seam_offsets(paths, verbose=True):
    """Median elevation difference between overlapping tiles, pair by pair.

    A large offset means the two tiles are on different vertical references —
    usually because their tide predictions came from different points. Values
    of a few centimetres are normal; tens of centimetres warrant levelling
    (or re-running both tiles with a shared tide location).

    Returns ``[{tile_a, tile_b, offset_m, n_overlap}, …]``.
    """
    from itertools import combinations
    from rasterio.warp import reproject, Resampling

    results = []
    for a, b in combinations(paths, 2):
        with rasterio.open(a) as sa, rasterio.open(b) as sb:
            if not (sa.bounds.left < sb.bounds.right
                    and sb.bounds.left < sa.bounds.right
                    and sa.bounds.bottom < sb.bounds.top
                    and sb.bounds.bottom < sa.bounds.top):
                continue                                  # no overlap
            arr_a = sa.read(1).astype(float)
            arr_b = np.full(sa.shape, np.nan, dtype=np.float32)
            reproject(source=rasterio.band(sb, 1), destination=arr_b,
                      src_transform=sb.transform, src_crs=sb.crs,
                      dst_transform=sa.transform, dst_crs=sa.crs,
                      resampling=Resampling.nearest)
        if sa.nodata is not None:
            arr_a = np.where(arr_a == sa.nodata, np.nan, arr_a)
        common = np.isfinite(arr_a) & np.isfinite(arr_b)
        if common.sum() < 30:
            continue
        offset = float(np.median(arr_b[common] - arr_a[common]))
        results.append({"tile_a": os.path.basename(a),
                        "tile_b": os.path.basename(b),
                        "offset_m": round(offset, 4),
                        "n_overlap": int(common.sum())})
        if verbose:
            print(f"[mosaic] seam {os.path.basename(a)} ↔ "
                  f"{os.path.basename(b)}: {offset:+.3f} m "
                  f"({int(common.sum()):,} px)")
    return results


def merge_tiles(paths, out_path, method="average", level_seams=False,
                verbose=True):
    """Merge tile rasters into one GeoTIFF.

    Parameters
    ----------
    method : {"average", "first", "max", "min"}
        How to resolve overlapping pixels. ``"average"`` (default) is the
        right choice for halos: it blends the two estimates instead of
        letting one tile win arbitrarily.
    level_seams : bool
        Remove each tile's median offset relative to the first tile before
        merging, hiding datum steps between tiles. Cosmetic — it changes
        absolute elevations, so state it when you use it.
    """
    info = check_tiles(paths, verbose=verbose)
    if info["problems"]:
        raise ValueError("tiles are not compatible: "
                         + "; ".join(info["problems"]))

    if method == "average":
        # rasterio's merge has no mean method: accumulate sum and count.
        datasets = [rasterio.open(p) for p in paths]
        try:
            total, transform = rio_merge(datasets, method="first")
            shape = total.shape[1:]
            acc = np.zeros(shape, dtype=np.float64)
            cnt = np.zeros(shape, dtype=np.float64)
            from rasterio.warp import reproject, Resampling
            for i, src in enumerate(datasets):
                buf = np.full(shape, np.nan, dtype=np.float32)
                reproject(source=rasterio.band(src, 1), destination=buf,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=src.crs,
                          resampling=Resampling.nearest)
                if src.nodata is not None:
                    buf = np.where(buf == src.nodata, np.nan, buf)
                if level_seams and i > 0:
                    common = np.isfinite(buf) & (cnt > 0)
                    if common.sum() > 30:
                        buf = buf - float(np.median(buf[common]
                                                    - (acc[common] / cnt[common])))
                good = np.isfinite(buf)
                acc[good] += buf[good]
                cnt[good] += 1
            merged = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            profile = datasets[0].profile
        finally:
            for src in datasets:
                src.close()
    else:
        datasets = [rasterio.open(p) for p in paths]
        try:
            stack, transform = rio_merge(datasets, method=method)
            merged = stack[0].astype(np.float32)
            profile = datasets[0].profile
        finally:
            for src in datasets:
                src.close()

    profile.update({"height": merged.shape[0], "width": merged.shape[1],
                    "transform": transform, "count": 1, "dtype": "float32",
                    "nodata": np.nan, "compress": "deflate"})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(merged.astype(np.float32), 1)
    if verbose:
        valid = int(np.isfinite(merged).sum())
        px = abs(transform.a * transform.e)
        print(f"[mosaic] wrote {out_path} · {merged.shape[0]}×{merged.shape[1]} "
              f"px · {valid * px / 1e6:.2f} km² with data")
    return out_path
