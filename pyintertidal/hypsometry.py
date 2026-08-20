"""
hypsometry.py — The area–elevation signature of an estuary
===========================================================

The hypsometric curve — how much surface lies below each elevation — is the
compact morphological fingerprint of a tidal system. It answers questions
that a map cannot at a glance:

* is this estuary **convex** (most area high, a narrow low channel: typically
  accreting, sediment-rich) or **concave** (most area low: typically
  erosional or sediment-starved)?
* how would the flooded area change under sea-level rise?
* did the system change shape between two epochs, even where the mean
  elevation did not move?

Because it is a distribution rather than a picture, it also lets you compare
estuaries of completely different sizes, and compare one estuary with itself
through time (:mod:`pyintertidal.morphodynamics`).
"""

from __future__ import annotations

import numpy as np


def hypsometric_curve(elevation, transform, n_bins=100, mask=None):
    """Cumulative area below each elevation.

    Returns a dict with ``levels`` (bin edges, m), ``area_km2`` (cumulative
    area below each level), ``fraction`` (the same, normalised 0–1) and
    ``hist_km2`` (area IN each bin — the density rather than the cumulative).
    """
    z = np.asarray(elevation, dtype=float)
    valid = np.isfinite(z)
    if mask is not None:
        valid &= mask
    if not valid.any():
        raise ValueError("no valid elevation pixels")
    zs = z[valid]
    px_km2 = abs(transform.a * transform.e) / 1e6

    levels = np.linspace(float(zs.min()), float(zs.max()), int(n_bins) + 1)
    hist, _ = np.histogram(zs, bins=levels)
    cumulative = np.cumsum(hist) * px_km2
    return {
        "levels": levels[1:],
        "area_km2": cumulative,
        "fraction": cumulative / cumulative[-1] if cumulative[-1] else cumulative,
        "hist_km2": hist * px_km2,
        "total_km2": float(zs.size * px_km2),
        "z_min": float(zs.min()), "z_max": float(zs.max()),
        "z_mean": float(zs.mean()), "z_median": float(np.median(zs)),
    }


def curve_with_uncertainty(z_keep, pixel_m=10.0, sigma_z=None, n_bins=80,
                           n_draws=30, seed=20260820):
    """MAREA-era hypsometric curve: 1-D elevations + a per-pixel sigma.

    The v4 product carries an uncertainty for every pixel, so its hypsometry
    can carry one too: the curve is recomputed ``n_draws`` times with each
    elevation jittered by its own sigma, and the 5-95 % envelope is
    reported. Pixels without elevation are simply absent — the curve
    describes what was measured, never what was filled.

    Returns dict with ``z`` (levels), ``area_km2`` (cumulative below),
    ``area_lo``/``area_hi`` (the envelope), ``integral`` (Strahler),
    ``area_total_km2``.
    """
    z = np.asarray(z_keep, float)
    ok = np.isfinite(z)
    zs = z[ok]
    if zs.size < 100:
        raise ValueError("demasiado pocos pixeles con cota")
    px_km2 = (float(pixel_m) ** 2) / 1e6
    levels = np.linspace(float(np.percentile(zs, 0.5)),
                         float(np.percentile(zs, 99.5)), int(n_bins))

    def cum(v):
        return np.searchsorted(np.sort(v), levels) * px_km2

    area = cum(zs)
    if sigma_z is not None:
        sg = np.asarray(sigma_z, float)[ok]
        sg = np.where(np.isfinite(sg), sg, np.nanmedian(sg))
        rng = np.random.default_rng(seed)
        draws = np.stack([cum(zs + sg * rng.standard_normal(zs.size))
                          for _ in range(int(n_draws))])
        lo = np.percentile(draws, 5, axis=0)
        hi = np.percentile(draws, 95, axis=0)
    else:
        lo = hi = area
    frac = area / area[-1] if area[-1] else area
    span = levels[-1] - levels[0]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    integral = float(integrate(1.0 - frac, (levels - levels[0]) / span)) \
        if span > 0 else float("nan")
    return {"z": levels, "area_km2": area, "area_lo": lo, "area_hi": hi,
            "integral": integral,
            "area_total_km2": float(zs.size * px_km2)}


def hypsometric_integral(curve):
    """The hypsometric integral (0–1): the shape of the curve in one number.

    Strahler's classic index, applied to a tidal flat. Values above ~0.5
    indicate a convex profile (most area high, e.g. a well-fed accreting
    flat); below ~0.5 a concave one (most area low, e.g. a scoured or
    sediment-starved system). Comparable between sites and between epochs.
    """
    levels = np.asarray(curve["levels"], dtype=float)
    fraction = np.asarray(curve["fraction"], dtype=float)
    span = levels[-1] - levels[0]
    if span <= 0:
        return float("nan")
    relative_height = (levels - levels[0]) / span
    # Area under the "fraction of area ABOVE this level" curve.
    # np.trapz was removed in NumPy 2; np.trapezoid is the current name.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(1.0 - fraction, relative_height))


def describe(curve, name="site"):
    """Readable summary of a hypsometric curve, with its interpretation."""
    hi = hypsometric_integral(curve)
    shape = ("convex — area concentrated high (accretional signature)"
             if hi > 0.55 else
             "concave — area concentrated low (erosional signature)"
             if hi < 0.45 else "near-linear — evenly graded profile")
    print(f"[hypsometry] {name}")
    print(f"  intertidal area   {curve['total_km2']:.2f} km²")
    print(f"  elevation range   {curve['z_min']:+.2f} … {curve['z_max']:+.2f} m "
          f"(median {curve['z_median']:+.2f})")
    print(f"  hypsometric integral {hi:.3f} → {shape}")
    return {"hypsometric_integral": hi, "shape": shape,
            "total_km2": curve["total_km2"]}


def compare_curves(curves, reference=None):
    """Compare hypsometric curves between sites or epochs.

    ``curves`` is ``{label: curve}``. Returns one row per curve with its
    area, median elevation and hypsometric integral, plus the difference in
    area against ``reference`` when given — the quickest way to see whether a
    system gained or lost intertidal surface, and at which elevations.
    """
    rows = []
    for label, curve in curves.items():
        row = {"label": label,
               "total_km2": round(curve["total_km2"], 3),
               "z_median": round(curve["z_median"], 3),
               "hypsometric_integral": round(hypsometric_integral(curve), 3)}
        if reference is not None and label != reference:
            base = curves[reference]
            row["area_change_km2"] = round(curve["total_km2"]
                                           - base["total_km2"], 3)
            row["z_median_change_m"] = round(curve["z_median"]
                                             - base["z_median"], 3)
        rows.append(row)
    return rows


def plot_curves(curves, title="Hypsometric curves", figsize=(7, 6), ax=None):
    """Plot one or several hypsometric curves (elevation on the y axis).

    Elevation goes on the vertical axis because that is how the estuary is
    stacked in reality; the horizontal axis is cumulative area.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    items = curves.items() if isinstance(curves, dict) else [("curve", curves)]
    for label, curve in items:
        ax.plot(curve["area_km2"], curve["levels"], lw=1.8, label=label)
    ax.set_xlabel("Cumulative area below level (km²)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    if isinstance(curves, dict) and len(curves) > 1:
        ax.legend()
    return fig, ax
