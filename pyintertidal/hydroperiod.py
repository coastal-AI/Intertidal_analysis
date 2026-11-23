"""
hydroperiod.py — How long is each pixel under water?
=====================================================

Elevation is a means, not an end. What organises intertidal life is the
**hydroperiod**: the fraction of time a given spot is submerged. Marsh
species, seagrass, shellfish beds and feeding windows for wading birds all
sort themselves along that axis, so a hydroperiod map is the bridge between
our topography and habitat.

The computation is simple once a DEM exists: a pixel at elevation *z* is
submerged whenever the tide exceeds *z*, so

.. math:: \\text{hydroperiod}(z) = P(\\text{tide} > z)

evaluated over a CONTINUOUS tide series (e.g. from
:meth:`pyintertidal.boundary.PyTMDBoundary.levels`), not over the satellite dates.
That distinction matters: the satellite samples the tide unevenly (see
:mod:`pyintertidal.coverage`), so using acquisition dates would inherit that
bias, while an hourly model series represents the real tidal regime.

How it fits: MAREA's emersion time (step 11) is ``1 - hydroperiod(z, h)``,
computed by ``scripts/marea_tools.emersion_time`` for each pixel's delay,
with h the tide at the mouth read that many minutes earlier every 10 minutes.

Do not confuse this with water frequency
----------------------------------------
Water frequency (:mod:`pyintertidal.frequency`) is the fraction of
OBSERVATIONS in which a pixel looked wet — an observational quantity,
biased by cloud cover and orbit. Hydroperiod is the fraction of TIME the
pixel is really submerged. They correlate, but only the second one is a
physical habitat variable.
"""

from __future__ import annotations

import numpy as np


def hydroperiod(elevation, tide_heights):
    """Fraction of time each pixel is submerged (0–1).

    Parameters
    ----------
    elevation : array, any shape
        Elevations (m) in the same vertical datum as the tide series
        (MAREA's z is on the scale of the tide at the mouth, as its tide
        series are). ``marea_tools.emersion_time`` passes 1-D pixel arrays.
    tide_heights : array
        Continuous tide series, ideally hourly over a full year or more so
        that spring/neap cycles are represented.

    Returns
    -------
    Float array of ``elevation``'s shape: the share of the series strictly
    above each elevation; NaN where the elevation is NaN.
    """
    z = np.asarray(elevation, dtype=float)
    tides = np.asarray(tide_heights, dtype=float)
    tides = np.sort(tides[np.isfinite(tides)])
    if tides.size == 0:
        raise ValueError("the tide series is empty")
    # Fraction of the series above each elevation, via the sorted series.
    idx = np.searchsorted(tides, z.ravel(), side="right")
    frac = 1.0 - idx / tides.size
    out = frac.reshape(z.shape)
    return np.where(np.isfinite(z), out, np.nan)


def exposure_time(elevation, tide_heights, hours_per_sample=1.0,
                  per="year"):
    """Hours per year (or per day) that each pixel is EXPOSED to air.

    The complement of the hydroperiod, in the units ecologists usually work
    in. Desiccation stress — the main control on the upper limit of most
    intertidal species — scales with this.

    ``elevation`` and ``tide_heights`` as in :func:`hydroperiod`. ``per``:
    "year" (8760 h), "month" (730 h) or "day" (24 h). ``hours_per_sample``
    is NOT used: the result depends only on the share of the series above
    each elevation, so the sampling interval does not enter.
    """
    submerged = hydroperiod(elevation, tide_heights)
    exposed = 1.0 - submerged
    total_hours = {"year": 8760.0, "day": 24.0, "month": 730.0}[per]
    return exposed * total_hours


def zonation(elevation, tide_heights, bands=None):
    """Classify pixels into ecological tidal zones by hydroperiod.

    The default bands follow the classical intertidal zonation, expressed as
    submergence fractions:

    ===================  ==================  ===============================
    zone                 submerged           typical community
    ===================  ==================  ===============================
    subtidal             > 90 %              permanently wet, seagrass
    lower intertidal     50–90 %             mudflats, high productivity
    mid intertidal       20–50 %             the classic flat
    upper intertidal      5–20 %             pioneer marsh (*Salicornia*)
    supratidal           < 5 %               high marsh, splash zone
    ===================  ==================  ===============================

    Returns ``(classes, labels)`` where ``classes`` is an integer raster
    (NaN → 0) and ``labels`` maps the codes to names.

    ``elevation`` and ``tide_heights`` as in :func:`hydroperiod`; ``bands``
    a list of ``(lo, hi, name)`` submergence fractions, each zone being
    ``lo < hydroperiod <= hi``; codes follow the list order from 1.
    ``classes`` is uint8 of ``elevation``'s shape.
    """
    if bands is None:
        bands = [(0.90, 1.01, "subtidal"), (0.50, 0.90, "lower intertidal"),
                 (0.20, 0.50, "mid intertidal"), (0.05, 0.20, "upper intertidal"),
                 (-0.01, 0.05, "supratidal")]
    hp = hydroperiod(elevation, tide_heights)
    classes = np.zeros(hp.shape, dtype=np.uint8)
    labels = {0: "no data"}
    for code, (lo, hi, name) in enumerate(bands, start=1):
        classes[np.isfinite(hp) & (hp > lo) & (hp <= hi)] = code
        labels[code] = name
    return classes, labels


def zone_areas(classes, labels, transform):
    """Area of each ecological zone in km² and as a percentage.

    The headline table of a habitat study: how much of the estuary is
    mudflat, how much is pioneer marsh, and how those shares change between
    epochs.

    ``classes`` and ``labels`` from :func:`zonation`; ``transform`` the
    grid's affine transform (only the pixel area is used). Percentages are
    of the classified pixels (code > 0). Returns one dict per zone.
    """
    px_km2 = abs(transform.a * transform.e) / 1e6
    total = int((classes > 0).sum())
    rows = []
    for code, name in labels.items():
        if code == 0:
            continue
        n = int((classes == code).sum())
        rows.append({"zone": name, "code": int(code), "pixels": n,
                     "area_km2": round(n * px_km2, 3),
                     "percent": round(100.0 * n / total, 1) if total else 0.0})
    return rows


def inundation_curve(elevation, tide_heights, n_levels=50):
    """Flooded area as a function of tide height — the estuary's rating curve.

    For each level in the observed tidal range, the area of the DEM below it.
    Practical uses: reading off how much habitat a storm surge would cover,
    and comparing the flooding behaviour of different estuaries.

    Returns ``(levels, n_pixels)`` — NOT an area: ``levels`` (n_levels,) m,
    evenly spaced over the finite range of ``tide_heights``; ``n_pixels``
    (n_levels,) float, the number of DEM pixels at or below each level.
    Multiply by the pixel area (1e-4 km² at 10 m) to get km².
    """
    z = np.asarray(elevation, dtype=float)
    valid = np.isfinite(z)
    tides = np.asarray(tide_heights, dtype=float)
    tides = tides[np.isfinite(tides)]
    levels = np.linspace(float(tides.min()), float(tides.max()), int(n_levels))
    zs = z[valid]
    counts = np.array([(zs <= lvl).sum() for lvl in levels], dtype=float)
    return levels, counts
