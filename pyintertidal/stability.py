"""
stability.py — Coastal stability: the reference map and the cloud filter
========================================================================

Two products from two streaming passes over the cube:

1. **Reference map** — every pixel is voted wet/dry across the CLEAN scenes
   of the archive and labelled:

   * ``0`` transition (sometimes wet, sometimes dry → intertidal candidate),
   * ``1`` stable water (wet in ≥ ``stable_threshold`` of votes),
   * ``2`` stable land,

   with a safety buffer dilated around the transition so narrow fringes are
   not missed.

2. **Per-date cloudiness** — for every scene, the percentage of cloudy
   pixels: measured GLOBALLY for clean scenes and INSIDE the transition zone
   for cloudy ones. This enables "date recovery": a scene whose clouds miss
   the transition zone is still perfectly usable for intertidal work even if
   the rest of the frame is overcast — on cloudy coasts this can multiply
   the usable archive (:func:`date_recovery_gain` reports by how much).

How it fits: this is step 3 of MAREA (``scripts/marea_tools.screen`` calls
:func:`reference_and_clouds` then :func:`filter_dates`); the usable dates
then restrict the record (step 5).

Things a reader trips on
------------------------
* Units: the values of ``cloud_pct`` are PERCENT (0-100), while
  ``clean_scene_max_bad`` and ``cloud_threshold`` are FRACTIONS (0-1).
* A pixel never seen in a clean scene has no votes and is labelled 0
  (transition); the buffer around the transition overwrites any stable
  class it reaches.
* The votes count SCL classes :data:`pyintertidal.water.CLEAR_CLASSES`
  (4, 5, 6) as clear, not the SCL 4-7 of MAREA's looks
  (``marea.CLEAR``). The published Villaviciosa screening was made this
  way; changing either changes the product.
"""

from __future__ import annotations

import numpy as np

from . import water as W


def reference_and_clouds(
    cube,
    threshold=0.0,
    bad_classes=W.BAD_CLASSES,
    clear_classes=W.CLEAR_CLASSES,
    clean_scene_max_bad=0.05,
    stable_threshold=0.95,
    transition_buffer_px=10,
    chunk=None,
):
    """Reference map + per-date cloud statistics (two streaming passes).

    Parameters
    ----------
    cube : SentinelCube
        The cached cube to stream.
    threshold : float
        Water-index threshold (ignored by the categorical ``"scl"`` detector).
    bad_classes : tuple of int
        SCL classes counted as cloud or shadow (default 3, 8, 9, 10).
    clear_classes : tuple of int
        SCL classes whose looks vote wet/dry (default
        :data:`pyintertidal.water.CLEAR_CLASSES`).
    clean_scene_max_bad : float
        A scene votes in the reference map only if its cloudy fraction is
        at most this (a fraction; default 0.05 = 5 %).
    stable_threshold : float
        Fraction of votes needed to call a pixel stable (default 95 %).
    transition_buffer_px : int
        Safety dilation (pixels) around the transition class.
    chunk : int, optional
        Dates per streamed block (:func:`pyintertidal.cube.auto_chunk`).

    Returns
    -------
    (reference_map, cloud_pct)
        ``reference_map`` — (H, W) uint8 raster (0/1/2 as documented above);
        ``cloud_pct`` — ``{date: cloud percentage, 0-100}`` where the
        percentage is global for clean dates and transition-zone-only for
        cloudy ones.
    """
    from scipy.ndimage import binary_dilation

    H, Wd = cube.shape
    dates = cube.dates
    T = len(dates)

    # ── pass 1: wet/dry votes over the clean scenes ──────────────────────────
    wet_votes = np.zeros((H, Wd), np.int64)
    dry_votes = np.zeros((H, Wd), np.int64)
    global_bad = np.zeros(T, np.float64)

    for sl, blocks in cube.stream(chunk):
        scl = np.nan_to_num(blocks["SCL"], nan=0.0).astype(np.int16)
        bad = np.isin(scl, list(bad_classes))
        gf = bad.reshape(bad.shape[0], -1).mean(axis=1)
        global_bad[sl] = gf
        clean = gf <= clean_scene_max_bad
        if clean.any():
            wet, dry, _ = W.water_land_masks(cube.water, blocks, threshold,
                                             clear_classes)
            wet_votes += wet[clean].sum(0)
            dry_votes += dry[clean].sum(0)

    votes = wet_votes + dry_votes
    votes[votes == 0] = 1          # no votes: neither stable class -> transition
    stable_wet = (wet_votes / votes) >= stable_threshold
    stable_dry = (dry_votes / votes) >= stable_threshold
    transition = ~(stable_wet | stable_dry)
    if transition_buffer_px > 0:
        transition = binary_dilation(transition,
                                     iterations=int(transition_buffer_px))
    reference = np.zeros((H, Wd), np.uint8)
    reference[stable_wet] = 1
    reference[stable_dry] = 2
    reference[transition] = 0      # last: the buffer overwrites stable classes

    # ── pass 2: per-date cloudiness (global or transition-zone) ──────────────
    tmask = reference == 0
    n_t = max(int(tmask.sum()), 1)
    cloud_pct = {}
    for sl, blocks in cube.stream(chunk):
        gf = global_bad[sl]
        scl = None
        if np.any(gf > clean_scene_max_bad):
            scl = np.nan_to_num(blocks["SCL"], nan=0.0).astype(np.int16)
        for j, g in enumerate(gf):
            d = dates[sl.start + j]
            if g <= clean_scene_max_bad:
                cloud_pct[d] = round(float(g) * 100, 4)
            else:
                bad = np.isin(scl[j], list(bad_classes))
                cloud_pct[d] = round(float(bad[tmask].sum()) / n_t * 100, 4)
    return reference, cloud_pct


def filter_dates(cloud_pct, cloud_threshold=0.10):
    """Dates whose cloudiness is at most the threshold.

    Explicit and tiny on purpose: the notebook shows the threshold and can
    report how many dates were recovered versus a global-only filter
    (:func:`date_recovery_gain`).

    Parameters
    ----------
    cloud_pct : dict
        ``{date: percent}`` from :func:`reference_and_clouds` (global for
        clean scenes, transition zone for the others), in PERCENT.
    cloud_threshold : float
        A FRACTION: 0.10 keeps dates with at most 10 % cloud.

    Returns sorted ``'YYYY-MM-DD'`` strings.
    """
    return sorted(d for d, p in cloud_pct.items()
                  if p / 100.0 <= cloud_threshold)


def date_recovery_gain(cloud_pct, clean_scene_max_bad=0.05,
                       cloud_threshold=0.10, verbose=True):
    """How many dates the transition-zone filter rescues, versus a global one.

    A scene can be 80 % cloudy overall and perfectly clear over the estuary.
    Judging dates by cloudiness INSIDE the transition zone therefore recovers
    imagery that a whole-scene filter would discard — on the Cantabrian coast
    this typically multiplies the usable archive several-fold, which directly
    improves every downstream product.

    ``cloud_pct`` is :func:`reference_and_clouds`'s dict (PERCENT);
    ``clean_scene_max_bad`` and ``cloud_threshold`` are FRACTIONS, as there.
    Returns counts and the relative gain, and prints a short report.

    Note: ``n_reference_quality`` counts the dates whose stored percentage is
    at most ``clean_scene_max_bad``. For a cloudy scene that percentage is
    the transition zone's, so a cloudy scene with a clear transition zone is
    counted there too: the reported gain is a lower bound of what the
    transition-zone filter adds to a whole-scene filter.
    """
    total = len(cloud_pct)
    reference = [d for d, p in cloud_pct.items()
                 if p / 100.0 <= clean_scene_max_bad]
    usable = [d for d, p in cloud_pct.items() if p / 100.0 <= cloud_threshold]
    recovered = sorted(set(usable) - set(reference))
    gain = (100.0 * len(recovered) / len(reference)) if reference else float("inf")
    out = {
        "n_total": total, "n_reference_quality": len(reference),
        "n_usable": len(usable), "n_recovered": len(recovered),
        "recovered_dates": recovered,
        "gain_relative_pct": gain,
        "coverage_before_pct": 100.0 * len(reference) / total if total else 0.0,
        "coverage_after_pct": 100.0 * len(usable) / total if total else 0.0,
    }
    if verbose:
        print(f"[dates] {total} acquisitions → "
              f"{len(reference)} clear scenes "
              f"({out['coverage_before_pct']:.0f}%) → "
              f"{len(usable)} usable after the transition-zone filter "
              f"({out['coverage_after_pct']:.0f}%)")
        print(f"        {len(recovered)} dates recovered "
              f"(+{gain:.0f}% more observations)")
    return out


def transition_geometry(reference_map, transform, crs, min_area_px=10):
    """Vectorise the transition zone into polygons (GeoDataFrame).

    Useful to hand the study area to GIS users, to clip other datasets, or to
    compute per-polygon statistics (e.g. one marsh vs the main channel).
    Polygons smaller than ``min_area_px`` pixels are dropped as speckle.

    ``reference_map`` is :func:`reference_and_clouds`'s (H, W) map;
    ``transform`` and ``crs`` are the cube's grid (``cube.grid``). Returns
    polygons in that CRS, with ``area_km2``, largest first.
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    mask = (np.asarray(reference_map) == 0).astype(np.uint8)
    pixel_area = abs(transform.a * transform.e)
    geoms = []
    for geom, value in shapes(mask, mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue
        poly = shape(geom)
        if poly.area / pixel_area >= min_area_px:
            geoms.append(poly)
    gdf = gpd.GeoDataFrame({"class": ["transition"] * len(geoms)},
                           geometry=geoms, crs=crs)
    gdf["area_km2"] = gdf.geometry.area / 1e6
    return gdf.sort_values("area_km2", ascending=False).reset_index(drop=True)
