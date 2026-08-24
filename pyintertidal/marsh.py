"""
marsh.py — Flooded vegetation (salt marsh) detection
====================================================

Salt-marsh plants (*Spartina*, *Salicornia*, *Zostera* beds) grow in the
upper intertidal and are regularly submerged, but ESA's Scene Classification
sees a green canopy and labels them **vegetation** — dry land. Left
uncorrected, the entire marsh disappears from the intertidal zone, which is
precisely the habitat that matters most ecologically.

The fix is spectral: vegetation whose water index is high is vegetation
standing in water. Those pixels are relabelled **class 12**, which is why
``12`` appears in :data:`pyintertidal.water.CLEAR_CLASSES` and
:data:`pyintertidal.water.SCL_WATER_CLASSES` throughout the package.

MNDWI (B03/B11) is the default index here — unlike open water, where NDWI
wins, short-wave infrared penetrates a thin canopy better and reveals the
water beneath it. That is a different physical problem from the open-water
detection in :mod:`pyintertidal.water`, which is why this module exists
separately rather than as another "water index".

This correction is optional and NOT part of the validated elevation
benchmark; it matters for habitat mapping and for studies where marsh extent
is the target.
"""

from __future__ import annotations

import numpy as np

from . import water as W

#: SCL class used for detected flooded vegetation.
MARSH_CLASS = 12
#: SCL classes that can be reclassified as marsh (vegetation and bare soil).
VEGETATION_CLASSES = (4, 5)


def detect_flooded_vegetation(scl, index, threshold=0.3,
                              vegetation_classes=VEGETATION_CLASSES):
    """Boolean mask of vegetation pixels that are standing in water.

    Parameters
    ----------
    scl : array
        Scene Classification band (any shape; ``(t, y, x)`` works too).
    index : array
        Water index of the same scene(s) — MNDWI recommended, see the module
        docstring; compute it with
        :func:`pyintertidal.water.index_block`.
    threshold : float
        Index value above which a vegetated pixel counts as flooded. 0.3 is
        conservative for MNDWI (0.0 would catch merely damp soil); lower it
        if the marsh is sparse, raise it if turbid water bleeds in.

    Returns
    -------
    Boolean array, True where vegetation is flooded.
    """
    scl = np.nan_to_num(np.asarray(scl), nan=0.0).astype(np.int16)
    index = np.asarray(index, dtype=np.float32)
    return (np.isin(scl, list(vegetation_classes))
            & np.isfinite(index) & (index > threshold))


def correct_scl(scl, flooded_mask, marsh_class=MARSH_CLASS):
    """Return a copy of the SCL band with flooded vegetation relabelled.

    The rest of the package treats ``marsh_class`` as both a clear
    observation and a water observation, so after this correction marsh
    pixels contribute to the water frequency instead of being counted as
    permanent land.
    """
    out = np.nan_to_num(np.asarray(scl), nan=0.0).astype(np.int16).copy()
    out[np.asarray(flooded_mask, dtype=bool)] = int(marsh_class)
    return out


def marsh_frequency(cube, dates=None, threshold=0.3, min_obs=8, chunk=None):
    """How often each pixel is flooded vegetation, across the archive.

    Streams the cube like every other product and returns the fraction of
    clear observations in which a pixel was detected as marsh — a direct
    habitat-extent indicator (stable marsh ≈ high values, transitional
    fringe ≈ intermediate).

    Requires a cube containing B11 (``water="mndwi"`` or ``"awei"``); with an
    NDWI-only cube the SWIR band is unavailable and NDWI is used instead,
    which under-detects marsh under dense canopy.
    """
    index_name = "mndwi" if "B11" in cube.bands else "ndwi"
    H, Wd = cube.shape
    all_dates = cube.dates
    keep = set(dates) if dates is not None else None

    marsh_votes = np.zeros((H, Wd), np.int64)
    clear_votes = np.zeros((H, Wd), np.int64)
    for sl, blocks in cube.stream(chunk):
        if keep is not None:
            idx = [j for j in range(sl.stop - sl.start)
                   if all_dates[sl.start + j] in keep]
            if not idx:
                continue
        index = W.index_block(index_name, blocks)
        scl = np.nan_to_num(blocks["SCL"], nan=0.0).astype(np.int16)
        flooded = detect_flooded_vegetation(scl, index, threshold)
        clear = np.isin(scl, list(W.CLEAR_CLASSES))
        if keep is not None:
            flooded, clear = flooded[idx], clear[idx]
        marsh_votes += flooded.sum(0)
        clear_votes += clear.sum(0)

    safe = np.where(clear_votes == 0, 1, clear_votes).astype(np.float32)
    freq = (marsh_votes / safe).astype(np.float32)
    freq[clear_votes < max(1, int(min_obs))] = np.nan
    return freq
