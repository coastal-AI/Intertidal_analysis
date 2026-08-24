"""
morphodynamics.py — What changed between two epochs
====================================================

Once elevation is fitted per epoch (:func:`pyintertidal.elevation.epochs`),
subtracting two DEMs turns a static map into a change study: where the flat
accreted, where it eroded, how much sediment moved, and whether the channels
migrated.

The one rule that makes this honest
-----------------------------------
**Never subtract two DEMs without their uncertainties.** Both epochs carry a
per-pixel error, so a difference is only meaningful when it exceeds what
those errors can produce by chance. :func:`dem_difference` therefore returns
a *significance mask* built from the propagated uncertainty

.. math:: \\sigma_\\Delta = \\sqrt{\\sigma_1^2 + \\sigma_2^2}

and every volume in :func:`sediment_budget` is computed over significant
pixels only. Without that test, a change map mostly shows noise — and noise
is what makes erosion "findings" evaporate under review.

A second caution: a satellite DEM is referenced to a tide-model datum. If the
two epochs used different tide models or a different prediction point, part
of the difference is datum, not sediment. Keep the configuration identical
between epochs, or remove the median offset explicitly.
"""

from __future__ import annotations

import numpy as np


def dem_difference(dem_early, dem_late, sigma_early=None, sigma_late=None,
                   confidence=1.96, remove_median_offset=False):
    """Elevation change between two epochs, with a significance test.

    Parameters
    ----------
    dem_early, dem_late : 2-D arrays
        DEMs of the two epochs, same grid. The result is late − early, so
        POSITIVE means accretion.
    sigma_early, sigma_late : 2-D arrays, optional
        Per-pixel uncertainties (``ElevationResult.sigma_mu``). Without them
        no significance test is possible and every finite pixel is reported —
        interpret those results with care.
    confidence : float
        Multiplier of the propagated uncertainty a change must exceed to be
        called significant (1.96 ≈ 95 %).
    remove_median_offset : bool
        Subtract the median difference first, absorbing a datum shift between
        epochs. Do this only if you have reason to believe the datums differ,
        and say so — it also removes any genuine basin-wide signal.

    Returns
    -------
    dict with ``change`` (m), ``significant`` (bool mask), ``sigma`` (the
    propagated uncertainty) and ``offset_removed``.
    """
    a = np.asarray(dem_early, dtype=float)
    b = np.asarray(dem_late, dtype=float)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]

    change = b - a
    offset = 0.0
    if remove_median_offset:
        common = np.isfinite(change)
        if common.any():
            offset = float(np.median(change[common]))
            change = change - offset

    if sigma_early is not None and sigma_late is not None:
        sa = np.asarray(sigma_early, dtype=float)[:h, :w]
        sb = np.asarray(sigma_late, dtype=float)[:h, :w]
        sigma = np.sqrt(np.nan_to_num(sa, nan=np.inf) ** 2
                        + np.nan_to_num(sb, nan=np.inf) ** 2)
        significant = np.isfinite(change) & (np.abs(change) > confidence * sigma)
    else:
        sigma = np.full(change.shape, np.nan)
        significant = np.isfinite(change)

    return {"change": change.astype(np.float32),
            "significant": significant,
            "sigma": sigma.astype(np.float32),
            "offset_removed": offset,
            "confidence": confidence}


def sediment_budget(difference, transform, mask=None):
    """Volumes of erosion and accretion (m³) over the significant pixels.

    Returns gross accretion, gross erosion, the net budget and the areas
    involved. The net figure is what tells you whether the system is gaining
    or losing sediment; the gross figures tell you how ACTIVE it is — a flat
    can be violently reworked and still net to zero.
    """
    change = np.asarray(difference["change"], dtype=float)
    sig = np.asarray(difference["significant"], dtype=bool)
    if mask is not None:
        sig = sig & mask[:sig.shape[0], :sig.shape[1]]
    px_area = abs(transform.a * transform.e)

    gain = change > 0
    loss = change < 0
    acc_px = int((sig & gain).sum())
    ero_px = int((sig & loss).sum())
    accretion = float(np.nansum(change[sig & gain]) * px_area)
    erosion = float(np.nansum(change[sig & loss]) * px_area)
    return {
        "accretion_m3": round(accretion, 1),
        "erosion_m3": round(erosion, 1),
        "net_m3": round(accretion + erosion, 1),
        "accretion_area_km2": round(acc_px * px_area / 1e6, 3),
        "erosion_area_km2": round(ero_px * px_area / 1e6, 3),
        "significant_px": int(sig.sum()),
        "mean_change_m": (round(float(np.nanmean(change[sig])), 3)
                          if sig.any() else float("nan")),
        "regime": ("accreting" if accretion + erosion > 0 else
                   "eroding" if accretion + erosion < 0 else "balanced"),
    }


def change_by_zone(difference, classes, labels, transform):
    """Sediment budget broken down by ecological zone.

    Erosion concentrated in the lower flat and accretion in the marsh mean
    something very different from the reverse, so a per-zone budget is often
    more informative than the basin total. Pair it with
    :func:`pyintertidal.hydroperiod.zonation`.
    """
    rows = []
    for code, name in labels.items():
        if code == 0:
            continue
        zone_mask = np.asarray(classes) == code
        budget = sediment_budget(difference, transform, mask=zone_mask)
        budget["zone"] = name
        rows.append(budget)
    return rows


def summarize(difference, transform, name="change", verbose=True):
    """Print a readable summary of a change analysis."""
    budget = sediment_budget(difference, transform)
    change = difference["change"]
    total = int(np.isfinite(change).sum())
    sig = budget["significant_px"]
    if verbose:
        print(f"[morphodynamics] {name}")
        print(f"  {sig:,} of {total:,} pixels changed significantly "
              f"({100 * sig / max(total, 1):.0f}%) at "
              f"{difference['confidence']:.2f}σ")
        if difference["offset_removed"]:
            print(f"  datum offset removed: "
                  f"{difference['offset_removed']:+.3f} m")
        print(f"  accretion {budget['accretion_m3']:+,.0f} m³ over "
              f"{budget['accretion_area_km2']:.2f} km²")
        print(f"  erosion   {budget['erosion_m3']:+,.0f} m³ over "
              f"{budget['erosion_area_km2']:.2f} km²")
        print(f"  net       {budget['net_m3']:+,.0f} m³ → {budget['regime']}")
    return budget
