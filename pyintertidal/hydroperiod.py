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

evaluated over a CONTINUOUS tide series (from
:meth:`pyintertidal.tides.TideService.series`), not over the satellite dates.
That distinction matters: the satellite samples the tide unevenly (see
:mod:`pyintertidal.coverage`), so using acquisition dates would inherit that
bias, while an hourly model series represents the real tidal regime.

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
    elevation : 2-D array
        DEM in the same vertical datum as the tide series (the output of
        :mod:`pyintertidal.elevation` already is).
    tide_heights : array
        Continuous tide series, ideally hourly over a full year or more so
        that spring/neap cycles are represented.

    Returns
    -------
    2-D float array, NaN where the DEM is NaN.
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

    Returns ``(levels, area_km2)``.
    """
    z = np.asarray(elevation, dtype=float)
    valid = np.isfinite(z)
    tides = np.asarray(tide_heights, dtype=float)
    tides = tides[np.isfinite(tides)]
    levels = np.linspace(float(tides.min()), float(tides.max()), int(n_levels))
    zs = z[valid]
    counts = np.array([(zs <= lvl).sum() for lvl in levels], dtype=float)
    return levels, counts
