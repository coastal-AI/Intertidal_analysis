"""
frequency.py — Water frequency and the intertidal window
========================================================

**Water frequency** is the fraction of clear observations in which a pixel
is wet. It is the workhorse product of intertidal mapping: stable water sits
near 1, stable land near 0, and the intertidal zone spans the middle, ordered
by elevation (higher flats flood less often).

The **intertidal window** ``low ≤ WF ≤ high`` (applied inside the transition
zone of the reference map) removes the residual noise at both ends. The
window can be pinned by hand or calibrated from the data with
:func:`multiotsu_window` — the notebook decides, visibly.

How it fits: this module serves ``elevation.fit_hsr``/``fit_step`` and
exploratory maps. MAREA does not use it: ``marea.extract`` computes its own
water frequency and window (pixels with more than 30 clear looks, wet in
more than 10 % and less than 90 % of them).
"""

from __future__ import annotations

import numpy as np

from . import water as W


def water_frequency(
    cube,
    dates=None,
    threshold=0.0,
    clear_classes=W.CLEAR_CLASSES,
    min_obs=8,
    chunk=None,
):
    """Fraction of clear observations in which each pixel is wet (streaming).

    Parameters
    ----------
    cube : SentinelCube
        The cached cube to stream.
    dates : list, optional
        Restrict to the cloud-filtered dates (recommended: pass the output of
        :func:`pyintertidal.stability.filter_dates`).
    threshold : float
        Index value above which a clear look is wet.
    clear_classes : tuple of int
        SCL classes counted as clear.
    min_obs : int
        Pixels with fewer clear observations than this are NaN — not enough
        evidence to state a frequency.
    chunk : int, optional
        Dates per streamed block.

    Returns
    -------
    (H, W) float32 raster in [0, 1] with NaN where evidence is insufficient.
    """
    H, Wd = cube.shape
    all_dates = cube.dates
    keep_set = set(dates) if dates is not None else None

    wet_votes = np.zeros((H, Wd), np.int64)
    clear_votes = np.zeros((H, Wd), np.int64)
    for sl, blocks in cube.stream(chunk):
        if keep_set is not None:
            keep = [j for j in range(sl.stop - sl.start)
                    if all_dates[sl.start + j] in keep_set]
            if not keep:
                continue
        wet, _, clear = W.water_land_masks(cube.water, blocks, threshold,
                                           clear_classes)
        if keep_set is not None:
            wet, clear = wet[keep], clear[keep]
        wet_votes += wet.sum(0)
        clear_votes += clear.sum(0)

    safe = np.where(clear_votes == 0, 1, clear_votes).astype(np.float32)
    wf = (wet_votes / safe).astype(np.float32)
    wf[clear_votes < max(1, int(min_obs))] = np.nan
    return wf


def multiotsu_window(wf, transition_mask,
                     bounds_low=(0.01, 0.25), bounds_high=(0.60, 0.99)):
    """Data-driven intertidal window via a 3-class Otsu split.

    Inside the transition zone the WF histogram mixes three populations —
    almost-never-wet (supratidal noise), truly intertidal, and
    almost-always-wet (subtidal). Multi-Otsu finds the two valleys that
    separate them; the result is clipped to sane bounds as a guard against
    degenerate histograms. Returns ``(low, high)``.

    ``wf`` is :func:`water_frequency`'s raster; ``transition_mask`` an
    (H, W) bool (``reference_map == 0``); ``bounds_low``/``bounds_high`` the
    ranges each threshold is clipped to.

    Falls back to ``(0.05, 0.85)`` WITHOUT a warning when fewer than 500
    finite values lie in the transition zone, or when multi-Otsu fails: check
    the sample size before trusting the window.
    """
    from skimage.filters import threshold_multiotsu

    vals = wf[transition_mask & np.isfinite(wf)]
    if vals.size < 500:
        return 0.05, 0.85          # conservative fallback for tiny samples
    try:
        t1, t2 = threshold_multiotsu(vals, classes=3)
    except Exception:
        return 0.05, 0.85
    return (float(np.clip(t1, *bounds_low)),
            float(np.clip(t2, *bounds_high)))


def intertidal_mask(wf, reference_map, low, high, aoi_mask=None):
    """The intertidal zone: transition pixels whose WF is inside the window.

    ``aoi_mask`` (optional) additionally clips to the study polygon so
    contiguous production tiles never overlap.

    ``wf`` (H, W) water frequency; ``reference_map`` (H, W) 0/1/2 from
    :func:`pyintertidal.stability.reference_and_clouds`; ``low``/``high`` the
    window (both ends included). Returns an (H, W) bool mask.
    """
    out = (reference_map == 0) & (wf >= low) & (wf <= high)
    if aoi_mask is not None:
        out = out & aoi_mask
    return out
