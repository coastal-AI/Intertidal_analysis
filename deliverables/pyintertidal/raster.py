"""
raster.py — GeoTIFF primitives
==============================

The single home for reading and writing rasters. Every other module imports
from here rather than reimplementing IO, so there is exactly one definition
of "write a product" and one of "read a scene" in the package.

Two families:

* **Products** — :func:`write_geotiff` / :func:`read_band` handle the
  single-band rasters the pipeline produces (elevation, water frequency,
  masks…).
* **Scenes** — :func:`read_rgb`, :func:`read_scl` and :func:`is_valid_tif`
  handle the per-date Sentinel-2 files downloaded by
  :mod:`pyintertidal.scenes` for visual inspection.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio


# ─────────────────────────────────────────────────────────────────────────────
#  Products
# ─────────────────────────────────────────────────────────────────────────────

def write_geotiff(path, array, transform, crs, dtype="float32", nodata=np.nan):
    """Write a single-band, deflate-compressed GeoTIFF.

    Use ``nodata=None`` for class/uint8 rasters (writes no nodata tag).
    Returns the path, so it composes: ``plot(write_geotiff(...))``.
    """
    array = np.asarray(array)
    profile = {
        "driver": "GTiff", "height": array.shape[0], "width": array.shape[1],
        "count": 1, "dtype": dtype, "transform": transform, "crs": crs,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    return path


def read_band(path, band=1, nodata_to_nan=True):
    """Read one band as float32 plus its georeferencing.

    Returns ``(array, transform, crs)``. The file's nodata value becomes NaN
    unless ``nodata_to_nan=False``.
    """
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        if nodata_to_nan and src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        return arr, src.transform, src.crs


def grid_of(path):
    """``(transform, crs, shape)`` of a raster — handy to align products."""
    with rasterio.open(path) as src:
        return src.transform, src.crs, src.shape


# ─────────────────────────────────────────────────────────────────────────────
#  Scenes (per-date Sentinel-2 files)
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_tif(path):
    """True when a downloaded scene actually contains data.

    OpenEO occasionally returns a technically valid GeoTIFF that is entirely
    zeros or NaN (e.g. the AOI fell outside the granule). Checking this
    before analysis avoids silently averaging empty scenes into a product.
    """
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
    except Exception:
        return False
    if data.size == 0:
        return False
    finite = data[np.isfinite(data)]
    return finite.size > 0 and not np.all(finite == 0)


def percentile_normalize(band, low=2, high=98):
    """Stretch a band to [0, 1] between two percentiles.

    Percentile (rather than min–max) stretching is what makes satellite RGB
    look natural: a handful of specular or cloud pixels would otherwise
    compress everything else into darkness.
    """
    band = np.asarray(band, dtype=np.float32)
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(band)
    return np.clip((band - lo) / (hi - lo), 0, 1)


def read_rgb(path, normalize=True, low=2, high=98):
    """Read a 3-band (B04, B03, B02) scene as an ``(H, W, 3)`` image.

    With ``normalize=True`` each band is percentile-stretched for display.
    Returns ``None`` when the file is missing or empty.
    """
    if not os.path.exists(path) or not is_valid_tif(path):
        return None
    with rasterio.open(path) as src:
        bands = [src.read(i + 1).astype(np.float32) for i in range(min(3, src.count))]
    if len(bands) < 3:
        return None
    if normalize:
        bands = [percentile_normalize(b, low, high) for b in bands]
    return np.dstack(bands)


def read_scl(path):
    """Read a Scene Classification band as an integer class array.

    Returns ``None`` when the file is missing or empty. See
    :mod:`pyintertidal.scl` for what the class numbers mean.
    """
    if not os.path.exists(path) or not is_valid_tif(path):
        return None
    with rasterio.open(path) as src:
        return np.nan_to_num(src.read(1), nan=0).astype(np.int16)


def scene_coverage(path):
    """Fraction of the scene that actually has data (0–1).

    Sentinel-2 granules cut across AOIs, so an "available" date may only
    cover part of the study area. Use this to drop partial scenes before
    they bias a composite. Pixels that are NaN or exactly zero count as
    no-data (that is how OpenEO pads outside the granule).
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    if data.size == 0:
        return 0.0
    return float((np.isfinite(data) & (data != 0)).mean())


def tifs_to_png(paths, out_dir, low=2, high=98, scl=False):
    """Convert scenes to plain PNGs (for datasets, slideshows or ML input).

    RGB files are percentile-stretched; with ``scl=True`` the class raster is
    written as-is (nearest-neighbour, no stretching) so classes stay exact.
    Returns the list of written paths.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for path in paths:
        arr = read_scl(path) if scl else read_rgb(path, True, low, high)
        if arr is None:
            continue
        img = (arr if scl else (arr * 255)).astype(np.uint8)
        out = os.path.join(out_dir,
                           os.path.splitext(os.path.basename(path))[0] + ".png")
        Image.fromarray(img).save(out, optimize=True)
        written.append(out)
    return written
