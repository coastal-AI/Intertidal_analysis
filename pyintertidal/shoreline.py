"""
shoreline.py — Waterlines and shoreline change
===============================================

A waterline is the boundary between wet and dry in ONE image — the sea's
edge at that instant, at that tide level. Two uses, deliberately kept apart:

**As topography.** With the tide height attached, a waterline is a contour of
known elevation. This is the classic waterline method that our elevation
estimators superseded (:mod:`pyintertidal.elevation`), but the vectors remain
useful for QA: overlaying a few waterlines on a DEM shows immediately whether
the modelled surface has the right shape.

**As a change indicator.** Extracted at a CONSISTENT tide level across years,
shoreline position becomes the standard coastal-erosion metric — the thing
managers actually track. :func:`shoreline_change` measures it along transects
perpendicular to the coast, the same way DSAS-style analyses do.

The catch, stated plainly: a waterline moves because the tide moved, because
the beach moved, or because the water index was noisy that day. Only the
second is signal. Always compare waterlines extracted at a similar tide
level (``tide_tolerance``) and always report how many dates went in.
"""

from __future__ import annotations

import numpy as np

from . import water as W


def extract_waterline(index, threshold, transform, crs, tide_height=None,
                      min_vertices=8, smooth=True):
    """Vectorise the wet/dry boundary of a single scene.

    Parameters
    ----------
    index : 2-D array
        Water index of one date (NDWI recommended).
    threshold : float
        The wet/dry cut — the same value the rest of the pipeline uses.
    tide_height : float, optional
        Attached to every feature; this is what turns a line into a contour
        of known elevation.

    Returns a GeoDataFrame of LineStrings with ``tide_height`` and
    ``length_m``.
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import LineString
    from skimage.measure import find_contours

    field = np.asarray(index, dtype=float)
    if smooth:
        from scipy.ndimage import uniform_filter
        valid = np.isfinite(field).astype(float)
        num = uniform_filter(np.nan_to_num(field), 3)
        den = uniform_filter(valid, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            field = np.where(den > 0.3, num / den, np.nan)

    filled = np.where(np.isfinite(field), field, threshold - 10.0)
    geoms = []
    for path in find_contours(filled, threshold):
        if len(path) < min_vertices:
            continue
        xs, ys = rasterio.transform.xy(transform, path[:, 0], path[:, 1])
        geoms.append(LineString(list(zip(xs, ys))))

    gdf = gpd.GeoDataFrame({"tide_height": [tide_height] * len(geoms)},
                           geometry=geoms, crs=crs)
    if len(gdf):
        gdf["length_m"] = gdf.geometry.length
        gdf = gdf.sort_values("length_m", ascending=False).reset_index(drop=True)
    return gdf


def waterline_series(cube, tide_heights, dates=None, threshold=0.0,
                     target_tide=None, tide_tolerance=0.25, max_lines=40,
                     verbose=True):
    """Extract one waterline per date, optionally at a fixed tide level.

    Parameters
    ----------
    target_tide : float, optional
        Only use dates whose tide is within ``tide_tolerance`` of this value.
        REQUIRED for change analysis: comparing waterlines from different
        tide levels measures the tide, not the coast.
    max_lines : int
        Cap on the number of dates processed (each one is a streamed read).

    Returns ``{date: GeoDataFrame}``.
    """
    dates = sorted(dates if dates is not None else tide_heights.keys())
    if target_tide is not None:
        dates = [d for d in dates
                 if d in tide_heights
                 and abs(tide_heights[d] - target_tide) <= tide_tolerance]
        if verbose:
            print(f"[shoreline] {len(dates)} dates within "
                  f"{tide_tolerance:.2f} m of {target_tide:+.2f} m")
    if len(dates) > max_lines:
        keep = np.linspace(0, len(dates) - 1, max_lines).round().astype(int)
        dates = [dates[i] for i in keep]

    transform, crs = cube.grid
    all_dates = cube.dates
    wanted = set(dates)
    out = {}
    for sl, blocks in cube.stream():
        for j in range(sl.stop - sl.start):
            date = all_dates[sl.start + j]
            if date not in wanted:
                continue
            block = {b: blocks[b][j:j + 1] for b in cube.bands}
            index = W.index_block(cube.water, block)
            if index is None:
                continue
            out[date] = extract_waterline(index[0], threshold, transform, crs,
                                          tide_height=tide_heights.get(date))
    if verbose:
        print(f"[shoreline] extracted {len(out)} waterlines")
    return dict(sorted(out.items()))


def shoreline_change(waterlines, baseline, transects, crs):
    """Shoreline position along transects, and its rate of change.

    Parameters
    ----------
    waterlines : dict {date: GeoDataFrame}
        Output of :func:`waterline_series`, ideally at a fixed tide level.
    baseline, transects : GeoDataFrames
        A landward baseline and the transects crossing the shoreline
        (the DSAS convention). Distance is measured from the baseline
        intersection to the waterline intersection.

    Returns a DataFrame with one row per transect and date, plus the linear
    rate of change (metres per year) fitted per transect.
    """
    import pandas as pd

    rows = []
    for date, lines in waterlines.items():
        if not len(lines):
            continue
        union = lines.geometry.union_all() if hasattr(lines.geometry, "union_all") \
            else lines.geometry.unary_union
        for tid, transect in enumerate(transects.geometry):
            hit = transect.intersection(union)
            if hit.is_empty:
                continue
            point = hit if hit.geom_type == "Point" else list(hit.geoms)[0]
            origin = transect.intersection(baseline.geometry.iloc[0]) \
                if len(baseline) else None
            start = origin if (origin is not None and not origin.is_empty
                               and origin.geom_type == "Point") \
                else type(point)(transect.coords[0])
            rows.append({"date": date, "transect": tid,
                         "distance_m": float(start.distance(point))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    rates = []
    for tid, group in df.groupby("transect"):
        if len(group) < 3:
            continue
        years = (group["date"] - group["date"].min()).dt.days / 365.25
        slope = np.polyfit(years, group["distance_m"], 1)[0]
        rates.append({"transect": tid, "rate_m_per_year": round(float(slope), 3),
                      "n_dates": len(group),
                      "trend": "accreting" if slope > 0 else "eroding"})
    df.attrs["rates"] = rates
    return df
