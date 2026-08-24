"""
terrain.py — Derivatives of any elevation model
===============================================

Slope, aspect, hillshade, contours and roughness. These work on ANY DEM on
the analysis grid — the HSR product, the step product, or an external LiDAR
raster — which is why they live in the core rather than next to one
particular reconstruction method.

Intertidal specifics worth knowing:

* Slopes are tiny (often < 1°), so :func:`slope` returns degrees and it is
  normal for the whole map to sit in the first degree; use a percentile
  colour scale when plotting.
* Every function is NaN-aware. Intertidal DEMs are full of holes (subtidal
  channels, land, low-confidence pixels) and a naive gradient would smear
  those holes across their neighbours.
"""

from __future__ import annotations

import numpy as np


def _gradients(elevation, pixel_size):
    """NaN-aware central-difference gradients ``(dz/dx, dz/dy)``."""
    z = np.asarray(elevation, dtype=float)
    gy, gx = np.gradient(z, pixel_size, pixel_size)
    invalid = ~np.isfinite(z)
    gx[invalid] = np.nan
    gy[invalid] = np.nan
    return gx, gy


def slope(elevation, pixel_size=10.0, degrees=True):
    """Terrain slope. Degrees by default, or the raw gradient magnitude.

    On tidal flats expect values well under 1°; the informative structure is
    the CONTRAST between the flats and the channel banks.
    """
    gx, gy = _gradients(elevation, pixel_size)
    magnitude = np.hypot(gx, gy)
    return np.degrees(np.arctan(magnitude)) if degrees else magnitude


def aspect(elevation, pixel_size=10.0):
    """Slope direction in degrees clockwise from North (0–360).

    In an estuary this maps which bank a pixel belongs to, and highlights the
    drainage direction of tidal creeks.
    """
    gx, gy = _gradients(elevation, pixel_size)
    ang = np.degrees(np.arctan2(-gx, gy))
    return np.where(np.isfinite(ang), (ang + 360.0) % 360.0, np.nan)


def hillshade(elevation, pixel_size=10.0, azimuth=315.0, altitude=45.0,
              vertical_exaggeration=1.0):
    """Classical GIS hillshade in [0, 1].

    ``vertical_exaggeration`` matters here more than in normal terrain work:
    with centimetric relief a true-scale hillshade is uniformly grey, so
    values of 5–20 are reasonable for intertidal DEMs.
    """
    z = np.asarray(elevation, dtype=float) * float(vertical_exaggeration)
    gx, gy = _gradients(z, pixel_size)
    slope_rad = np.arctan(np.hypot(gx, gy))
    aspect_rad = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    shaded = (np.sin(alt) * np.cos(slope_rad)
              + np.cos(alt) * np.sin(slope_rad) * np.cos(az - aspect_rad))
    return np.clip(shaded, 0, 1)


def roughness(elevation, window=3):
    """Local standard deviation of elevation — small-scale surface texture.

    Complements the HSR ``sigma`` product: ``sigma`` is relief measured
    WITHIN a pixel from the tidal signal, this is relief measured BETWEEN
    neighbouring pixels from the DEM itself. Disagreement between the two is
    informative (it usually means the DEM is smoother than reality).
    """
    from scipy.ndimage import uniform_filter

    z = np.asarray(elevation, dtype=float)
    valid = np.isfinite(z).astype(float)
    filled = np.nan_to_num(z, nan=0.0)
    mean = uniform_filter(filled, window) / np.maximum(
        uniform_filter(valid, window), 1e-6)
    mean_sq = uniform_filter(filled ** 2, window) / np.maximum(
        uniform_filter(valid, window), 1e-6)
    var = np.maximum(mean_sq - mean ** 2, 0.0)
    return np.where(np.isfinite(z), np.sqrt(var), np.nan)


def drop_small_regions(elevation, min_region_px=25, connectivity=2):
    """Blank isolated specks: keep only connected regions of some size.

    A per-pixel fit is a per-pixel decision, so a handful of pixels far from
    the estuary — a wet field, a roof, a pond — can pass the tests on their
    own and scatter confetti across the map. Real intertidal flats are
    CONTIGUOUS, which is information the fit never uses and this does.

    Deliberately a separate, explicit step rather than something the plots do
    quietly: it deletes data, so it belongs in the analysis where it shows up
    in the pixel count, not in the rendering where it would only flatter the
    figure.

    Parameters
    ----------
    elevation : 2-D array
        Values with NaN outside the mapped area.
    min_region_px : int
        Regions smaller than this are set to NaN. At 10 m, 25 px = 0.25 ha.
    connectivity : int
        1 = edge neighbours only, 2 = diagonals count too.

    Returns
    -------
    2-D array, same shape.
    """
    from scipy import ndimage

    z = np.asarray(elevation, dtype=float)
    valid = np.isfinite(z)
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, n = ndimage.label(valid, structure=structure)
    if n == 0:
        return z.copy()
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = sizes >= int(min_region_px)
    return np.where(keep[labels], z, np.nan)


def contours(elevation, transform, crs, interval=0.5, levels=None,
             min_vertices=5):
    """Elevation contours as a GeoDataFrame of LineStrings.

    Contours of an intertidal DEM are effectively modelled waterlines: the
    ``0 m`` contour is mean sea level, and the spacing of the others shows
    how quickly the flat rises. Fragments shorter than ``min_vertices`` are
    dropped as noise.
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import LineString
    from skimage.measure import find_contours

    z = np.asarray(elevation, dtype=float)
    finite = np.isfinite(z)
    if not finite.any():
        return gpd.GeoDataFrame({"elevation": []}, geometry=[], crs=crs)

    if levels is None:
        lo = np.floor(np.nanmin(z) / interval) * interval
        hi = np.ceil(np.nanmax(z) / interval) * interval
        levels = np.arange(lo, hi + interval, interval)

    # find_contours cannot handle NaN: fill with a value far below every
    # level so contours simply do not cross the holes.
    filled = np.where(finite, z, np.nanmin(z) - 1e6)

    geoms, values = [], []
    for level in levels:
        for path in find_contours(filled, level):
            if len(path) < min_vertices:
                continue
            rows, cols = path[:, 0], path[:, 1]
            valid = finite[np.clip(np.round(rows).astype(int), 0, z.shape[0] - 1),
                           np.clip(np.round(cols).astype(int), 0, z.shape[1] - 1)]
            if valid.mean() < 0.5:          # mostly outside real data
                continue
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            geoms.append(LineString(list(zip(xs, ys))))
            values.append(float(level))
    return gpd.GeoDataFrame({"elevation": values}, geometry=geoms, crs=crs)
