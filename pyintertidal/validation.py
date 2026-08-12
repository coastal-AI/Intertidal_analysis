"""
validation.py — Independent references, error metrics, field transects
======================================================================

Validation is deliberately SEPARATE from production: tiles must not depend
on a national reference service, and a LiDAR survey is a single moment in
time while satellite products represent an epoch. This module provides:

* :func:`download_mdt_ign` — clip of the Spanish IGN 5 m LiDAR DTM (WCS),
* :func:`reproject_to_grid` — put any reference on the analysis grid,
* :func:`compare_dems` — score several DEMs against one reference,
* :func:`transect_profile` — sample any raster along a GPS transect (built
  for field campaigns: walk the line, compare measured vs predicted).

Datum note (important): satellite elevations live in the tide-model MSL
datum; LiDAR is orthometric. Their offset shows up as a BIAS (median
difference) that says nothing about method skill, so metrics are reported
after removing it (and the bias itself is reported separately).
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

IGN_MDT_WCS = "https://servicios.idee.es/wcs-inspire/mdt"


def download_mdt_ign(bbox, out_path, coverage="Elevacion4258_5",
                     timeout=120, force=False):
    """Download a clip of the Spanish IGN 5 m LiDAR DTM via WCS.

    ``bbox`` is WGS84 ``west/south/east/north`` (add ``crs`` for other input
    systems). Idempotent: an existing ``out_path`` is reused unless
    ``force=True`` — but note the file is NOT checked against the bbox, so
    use one file per study area.
    """
    import requests
    from pyproj import Transformer

    if (not force) and os.path.exists(out_path):
        return out_path

    src_crs = bbox.get("crs", "EPSG:4326") if isinstance(bbox, dict) else "EPSG:4326"
    tf = Transformer.from_crs(src_crs, "EPSG:4258", always_xy=True)
    west, south = tf.transform(bbox["west"], bbox["south"])
    east, north = tf.transform(bbox["east"], bbox["north"])

    params = [
        ("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
        ("coverageId", coverage),
        ("subset", f"Lat({south},{north})"),
        ("subset", f"Long({west},{east})"),
        ("format", "image/tiff"),
    ]
    resp = requests.get(IGN_MDT_WCS, params=params, timeout=timeout)
    resp.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path


def reproject_to_grid(src_path, dst_transform, dst_crs, dst_shape,
                      resampling=Resampling.bilinear, nodata_below=-1000.0):
    """Reproject band 1 of a raster onto the analysis grid.

    Values below ``nodata_below`` become NaN (WCS servers often encode
    nodata as large negative numbers).
    """
    out = np.full(dst_shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=resampling,
        )
    return np.where(out < nodata_below, np.nan, out)


def compare_dems(dems, reference, mask=None, remove_bias=True):
    """Score several DEMs against one reference on their common pixels.

    Parameters
    ----------
    dems : dict {label: 2-D array}
        The DEMs to score, all on the SAME grid as ``reference``.
    reference : 2-D array
        The truth raster (e.g. LiDAR via :func:`reproject_to_grid`).
    remove_bias : bool
        Subtract each DEM's median offset before RMSE/MAE (datum offset, see
        module docstring). The raw bias is always reported.

    Returns
    -------
    list of dicts (n, bias_m, rmse_m, mae_m, pearson_r, spearman) — one per
    DEM, ready for ``pandas.DataFrame(rows)``.
    """
    from scipy.stats import pearsonr, spearmanr

    ref = np.asarray(reference, dtype=float)
    rows = []
    for label, dem in dems.items():
        pred = np.asarray(dem, dtype=float)
        m = np.isfinite(pred) & np.isfinite(ref)
        if mask is not None:
            m &= mask
        n = int(m.sum())
        if n < 50:
            rows.append({"method": label, "n": n})
            continue
        d = pred[m] - ref[m]
        bias = float(np.median(d))
        dd = d - bias if remove_bias else d
        rows.append({
            "method": label, "n": n, "bias_m": round(bias, 3),
            "rmse_m": round(float(np.sqrt((dd ** 2).mean())), 3),
            "mae_m": round(float(np.abs(dd).mean()), 3),
            "pearson_r": round(float(pearsonr(pred[m], ref[m])[0]), 3),
            "spearman": round(float(spearmanr(pred[m], ref[m])[0]), 3),
        })
    return rows


def transect_profile(raster, transform, crs, start_lonlat, end_lonlat, n=200):
    """Sample a raster along a straight transect between two WGS84 points.

    Built for field campaigns: define the transect by the GPS coordinates you
    will actually walk, plot the predicted profile of every method
    (:func:`pyintertidal.viz.plot_transect`), and compare with what you
    measure on site.

    Returns ``(distances_m, values)`` — distance along the line and the
    sampled values (NaN outside data).
    """
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = tf.transform(*start_lonlat)
    x1, y1 = tf.transform(*end_lonlat)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    inv = ~transform
    cols, rows_ = inv * (xs, ys)
    rows_ = np.round(rows_).astype(int)
    cols = np.round(cols).astype(int)
    values = np.full(n, np.nan, dtype=float)
    inside = ((rows_ >= 0) & (rows_ < raster.shape[0]) &
              (cols >= 0) & (cols < raster.shape[1]))
    values[inside] = np.asarray(raster, dtype=float)[rows_[inside], cols[inside]]
    distances = np.hypot(xs - x0, ys - y0)
    return distances, values


def wf_vs_elevation(water_frequency, elevation, mask=None, z_range=None):
    """Correlate water frequency against a reference elevation.

    A physical consistency check that needs NO elevation model of our own:
    higher ground floods less often, so water frequency and elevation must be
    NEGATIVELY correlated. The more negative the Spearman coefficient, the
    more topographically consistent the water-frequency map — which is how we
    compared water indices (NDWI/MNDWI/AWEI/SCL) before any DEM existed.

    Parameters
    ----------
    water_frequency : 2-D array or dict {label: array}
        One map, or several to rank against each other.
    elevation : 2-D array
        Reference elevation on the same grid (e.g. LiDAR).
    mask : bool array, optional
        Restrict to a region (normally the intertidal mask).
    z_range : (low, high), optional
        Keep only reference elevations in this band — excludes deep water
        and high land, where the relationship saturates and dilutes the test.

    Returns
    -------
    dict or list of dicts with ``spearman``, ``pearson`` and ``n``.
    """
    from scipy.stats import pearsonr, spearmanr

    maps = (water_frequency if isinstance(water_frequency, dict)
            else {"wf": water_frequency})
    ref = np.asarray(elevation, dtype=float)
    rows = {}
    for label, wf in maps.items():
        wf = np.asarray(wf, dtype=float)
        m = np.isfinite(wf) & np.isfinite(ref)
        if mask is not None:
            m &= mask
        if z_range is not None:
            m &= (ref >= z_range[0]) & (ref <= z_range[1])
        n = int(m.sum())
        if n < 30:
            rows[label] = {"spearman": np.nan, "pearson": np.nan, "n": n}
            continue
        rows[label] = {
            "spearman": round(float(spearmanr(wf[m], ref[m])[0]), 4),
            "pearson": round(float(pearsonr(wf[m], ref[m])[0]), 4),
            "n": n,
        }
    return rows if isinstance(water_frequency, dict) else rows["wf"]


def mask_agreement(masks):
    """Intersection-over-Union between every pair of masks.

    Quantifies how much two methods actually disagree about WHERE the
    intertidal zone is — a difference that summary areas (km²) can hide, since
    two maps of identical area can sit in different places.

    ``masks`` is ``{label: bool array}``; returns ``{(a, b): iou}``.
    """
    import itertools

    out = {}
    for a, b in itertools.combinations(masks, 2):
        ma = np.asarray(masks[a], dtype=bool)
        mb = np.asarray(masks[b], dtype=bool)
        h = min(ma.shape[0], mb.shape[0])
        w = min(ma.shape[1], mb.shape[1])
        ma, mb = ma[:h, :w], mb[:h, :w]
        union = int((ma | mb).sum())
        out[(a, b)] = float((ma & mb).sum() / union) if union else 0.0
    return out
