"""
tidecheck.py — What an estuary does to the tide, and what a satellite can see
=============================================================================

Every intertidal elevation method, ours and the published ones alike, takes
the water level as a known input from a global tide model. Measured against an
RTK GNSS survey of the Ría de Villaviciosa, that input — not the estimator —
is the limiting error: HSR and the faithful step baseline tie, while relief
comes out compressed (slope 0.7–0.9 against surveyed elevation instead of 1)
and the compression worsens ~1.0 of slope per kilometre upstream, *identically
in both methods*. Something shared is wrong, and the only thing they share is
the tide.

This module is the set of checks that follow from that.

What is recoverable, and what is not
------------------------------------
The sigmoid of :mod:`pyintertidal.elevation` is **invariant under affine maps
of the tide axis**: replacing ``h`` with ``α·h + c`` merely remaps every ``μ``
and ``σ`` by the same amounts and leaves the residual untouched. So:

============================  =========================  ====================
Estuarine effect              Recoverable from imagery?  Villaviciosa
============================  =========================  ====================
Datum offset                  No — needs a gauge         ---
Constant amplification ``α``  No — needs ground truth    ---
Phase lag ``τ``               Yes, :func:`phase_lag`     ≈ 0
Damping that varies with      Yes,                       **0.25 m**
tidal range                   :func:`range_dependence`
============================  =========================  ====================

That first pair is a theorem, not a limitation of effort: no amount of
cleverness recovers a scale the data cannot distinguish. Correcting
amplification in an estuary requires a tide gauge, a pressure sensor, or a
field survey — and saying so is more useful than an estimator that appears to
do it.

Everything that is NOT affine does leave a signature, which is why the last
two rows are measurable. Real estuaries deform the wave in exactly those
non-affine ways.

A caution about the published ensemble
--------------------------------------
``eo-tides`` offers an ``"ensemble"`` model that blends the three
best-performing models at each location. Its ranking dataset is 32 936 points
and **all of them are in Australian waters**. At 15 000 km the inverse-distance
weighting returns near-identical weights, and the ensemble degenerates to an
unweighted mean of its members — measured difference 0.0000 m over 1379 dates
at Villaviciosa. It is not wrong, it simply carries no local information
outside its calibration area. Building the equivalent ranking for another
coast, against tide gauges there, is a real piece of work someone should do.
"""

from __future__ import annotations

import numpy as np

from .elevation import _fit_block

__all__ = ["nearest_ocean_km", "phase_lag", "range_dependence"]

_SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


# ─────────────────────────────────────────────────────────────────────────────
#  Is the tide even predicted near this site?
# ─────────────────────────────────────────────────────────────────────────────

def nearest_ocean_km(lat, lon, model="EOT20", directory="./tide_models",
                     search_deg=0.5, step_deg=0.02,
                     when="2024-01-01T12:00:00", verbose=True):
    """Distance to the closest cell where ``model`` actually predicts a tide.

    Tide models are defined on an ocean grid. An estuary usually falls on a
    LAND cell, where the model returns nothing, so every implementation —
    ours included — searches outward until it finds water. That search
    succeeds silently, which is the problem: at the Ría de Villaviciosa,
    ``GOT4.10`` on its 0.5° grid has no ocean cell within **34 km**, so every
    elevation ever fitted there used a tide imported from open water, and
    nothing in the output said so. ``EOT20`` at 1/8° brings it to **13 km**.
    (Both measured with the default 2 km probe; a coarser probe overstates
    the EOT20 figure by nearly a factor of two, which is why ``step_deg``
    exists.)

    Call this before trusting a site, and record the answer beside the
    products. A large distance does not invalidate a fit — the tide is
    coherent over tens of kilometres — but it is a term in the error budget
    that deserves to be visible.

    Parameters
    ----------
    lat, lon : float
        Point of interest, usually :func:`pyintertidal.tides.water_centroid`.
    model : str
        pyTMD model name. ``"GOT4.10"`` must be spelled ``"GOT4.10_nc"`` for
        pyTMD itself; this function handles that.
    search_deg : float
        Half-width of the probe grid, in degrees.
    step_deg : float
        Probe spacing. This sets the ANSWER'S RESOLUTION, so it is a spacing
        and not a point count: with a fixed number of points the reported
        distance depends on where those points happen to land, and the same
        site gives 36 km or 40 km depending on the grid. At the default
        0.02° (~2 km) the answer is stable.

    Returns
    -------
    dict with ``distance_km`` (``inf`` if the whole box is land),
    ``n_water`` / ``n_probed``, ``step_km``, and the ``lat``/``lon`` of the
    closest hit.
    """
    import pyTMD

    name = "GOT4.10_nc" if model == "GOT4.10" else model
    n = max(3, int(round(2 * search_deg / step_deg)) + 1)
    lats = np.linspace(lat - search_deg, lat + search_deg, n)
    lons = np.linspace(lon - search_deg, lon + search_deg, n)
    LO, LA = np.meshgrid(lons, lats)
    stamp = np.array([np.datetime64(when)] * LO.size, dtype="datetime64[s]")

    # pyTMD's own parameters are upper-case (MODEL, DIRECTORY, EPSG, TIME).
    # The lower-case aliases exist but do not map reliably across versions —
    # passing `model=` silently leaves MODEL as None and fails with
    # "Unlisted tide model None", which reads like a missing data file.
    h = pyTMD.compute.tide_elevations(
        LO.ravel(), LA.ravel(), stamp, MODEL=name, DIRECTORY=directory,
        EPSG="4326", TIME="datetime")
    h = np.ma.filled(np.asarray(h, float).ravel(), np.nan)
    ok = np.isfinite(h)

    out = {"model": model, "n_probed": int(LO.size), "n_water": int(ok.sum()),
           "step_km": float(step_deg * 111.32),
           "distance_km": float("inf"), "lat": None, "lon": None}
    if ok.any():
        km = np.hypot((LO.ravel()[ok] - lon) * 111.32 * np.cos(np.radians(lat)),
                      (LA.ravel()[ok] - lat) * 111.32)
        i = int(np.argmin(km))
        out.update(distance_km=float(km[i]),
                   lat=float(LA.ravel()[ok][i]), lon=float(LO.ravel()[ok][i]))
    if verbose:
        d = out["distance_km"]
        where = "nowhere in the search box" if not np.isfinite(d) \
            else f"{d:.1f} km away"
        print(f"[tidecheck] {model}: nearest ocean cell {where} "
              f"({out['n_water']}/{out['n_probed']} probe points wet)")
        if np.isfinite(d) and d > 15:
            print(f"[tidecheck] that is far for an estuary — the tide used is "
                  f"open-water, not the level inside the ria")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Phase lag: does the wave arrive late?
# ─────────────────────────────────────────────────────────────────────────────

def phase_lag(Y, C, tide_at, flood, ebb, lags=range(-120, 121, 20),
              mu_points=60, sg_grid=_SG_GRID, min_obs=8, min_b=0.15,
              verbose=True):
    """Estimate the estuary's tidal lag from the imagery alone.

    A time lag is not an affine map of the tide axis — each scene moves by a
    different amount depending on where in the cycle it fell — so unlike
    amplification it *is* identifiable. The signature: if the estuary lags the
    ocean, then on a RISING tide the real level is below the model and on a
    FALLING tide above it. Fit the rising scenes and the falling scenes
    separately and their elevations disagree; the lag that makes them agree is
    the estuary's lag.

    Parameters
    ----------
    Y, C : (T, P) arrays
        NDWI and a usable-observation mask, as returned by
        :func:`pyintertidal.elevation.extract_intertidal_ndwi`.
    tide_at : callable
        ``tide_at(minutes) -> (T,)`` tide height with the series shifted by
        that many minutes. Positive means the estuary lags the ocean.
    flood, ebb : (T,) bool
        Rising and falling scenes. Compute these ONCE from the unshifted ocean
        model and hold them fixed, so subset membership does not change as the
        lag varies — otherwise the two effects are confounded.

    Returns
    -------
    dict with ``curve`` (list of ``(lag_min, median_difference_m)``),
    ``best_grid`` and ``crossing`` (the interpolated zero, in minutes).

    Notes
    -----
    Measured at Villaviciosa on the 2023–2025 epoch: **−7.6 min**, i.e. no
    meaningful lag — the ria is in phase with the open-ocean model to within
    ten minutes, which is finer than the 20-minute grid step. Split into bands
    along the estuary axis the crossing trends at +7.3 min/km (r = 0.879),
    which matches the shallow-water travel time — but applying that gradient
    made the elevation *worse* against the field survey and did not reduce the
    compression. Report the number; do not correct with it on this evidence.
    """
    curve = []
    for lag in lags:
        t = tide_at(int(lag))
        mu_f = _fit_subset(Y, C, t, flood, mu_points, sg_grid, min_obs, min_b)
        mu_e = _fit_subset(Y, C, t, ebb, mu_points, sg_grid, min_obs, min_b)
        both = np.isfinite(mu_f) & np.isfinite(mu_e)
        if both.sum() < 50:
            continue
        d = float(np.median(mu_e[both] - mu_f[both]))
        curve.append((int(lag), d, int(both.sum())))
        if verbose:
            print(f"[tidecheck] lag {lag:+5d} min  "
                  f"median(mu_ebb - mu_flood) = {d:+.4f} m  "
                  f"({int(both.sum())} px)")

    if not curve:
        return {"curve": [], "best_grid": None, "crossing": None}
    xs = np.array([c[0] for c in curve], float)
    ys = np.array([c[1] for c in curve], float)
    crossing = None
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            crossing = float(xs[i] - ys[i] * (xs[i + 1] - xs[i])
                             / (ys[i + 1] - ys[i]))
            break
    best = float(xs[np.argmin(np.abs(ys))])
    if verbose:
        print(f"[tidecheck] lag on the grid {best:+.0f} min; "
              + (f"zero crossing {crossing:+.1f} min" if crossing is not None
                 else "no zero crossing in the range probed"))
        print("[tidecheck] positive = the ria lags the ocean")
    return {"curve": curve, "best_grid": best, "crossing": crossing}


# ─────────────────────────────────────────────────────────────────────────────
#  Range dependence: are springs damped differently from neaps?
# ─────────────────────────────────────────────────────────────────────────────

def range_dependence(Y, C, tide, day_range, mu_points=60, sg_grid=_SG_GRID,
                     min_obs=6, min_b=0.15, n_bins=10, edge_pad=0.1,
                     seed=99, verbose=True):
    """Does the estuary treat big tides differently from small ones?

    A *constant* amplification is invisible to the fit (it is affine). An
    amplification that DEPENDS on tidal range is not, and it is what a real
    estuary does — friction damps large tides more, convergence amplifies
    them, and which wins is a property of the channel's shape.

    Fit the spring scenes and the neap scenes separately and compare. Two
    corrections decide whether the answer means anything, and both are applied
    here:

    1. **A common tide window.** Springs reach lower water than neaps and
       neaps reach higher, so each subset would otherwise get a ``μ`` grid
       shifted the other way and the comparison would measure the grids. Both
       subsets are clipped to the range they share.
    2. **Matched tide-height histograms.** Even inside that window springs
       cluster at the extremes and neaps in the middle. Bin the window and
       take equal numbers from each subset per bin, so the only thing left
       differing is the range of the tide that produced each scene.

    Without (1) the raw Villaviciosa answer is −0.23 m and is pure artefact;
    with (1) alone it is −0.18 m; with both it is **−0.25 m**, consistent in
    sign across every band of the estuary and flat along its axis. Negative
    means the spring fit places pixels lower — the real level during springs
    ran higher than the model, i.e. the ria AMPLIFIES big tides, which is what
    a convergent, funnel-shaped estuary does.

    Parameters
    ----------
    Y, C : (T, P) arrays
        NDWI and usable-observation mask.
    tide : (T,) array
        Tide height at each scene.
    day_range : (T,) array
        Peak-to-peak tide range of the DAY each scene falls on — a property of
        the date, not of the scene's own height. Model a dense grid either
        side of the overpass and take max minus min.

    Returns
    -------
    dict with ``difference_m`` (spring minus neap), ``n_pixels``,
    ``n_spring`` / ``n_neap`` after matching, and the mean tidal range of each.
    """
    rng = np.random.default_rng(seed)
    finite = np.isfinite(tide) & np.isfinite(day_range)
    med = float(np.nanmedian(day_range[finite]))
    spring = finite & (day_range >= med)
    neap = finite & (day_range < med)

    lo = max(np.nanmin(tide[spring]), np.nanmin(tide[neap]))
    hi = min(np.nanmax(tide[spring]), np.nanmax(tide[neap]))
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    who = np.digitize(tide, edges)

    ms, mn = np.zeros_like(spring), np.zeros_like(neap)
    for k in range(1, len(edges) + 1):
        si = np.flatnonzero(spring & (who == k) & (tide >= lo) & (tide <= hi))
        ni = np.flatnonzero(neap & (who == k) & (tide >= lo) & (tide <= hi))
        take = min(len(si), len(ni))
        if take:
            ms[rng.choice(si, take, replace=False)] = True
            mn[rng.choice(ni, take, replace=False)] = True

    grid = np.linspace(lo, hi, mu_points)
    pad = edge_pad * (hi - lo)

    def fit(mask):
        mu = _fit_subset(Y, C, tide, mask, mu_points, sg_grid, min_obs, min_b,
                         grid=grid)
        # Only pixels whose transition sits well inside the shared window: at
        # the edges mu is pinned by the grid rather than measured.
        return np.where((mu > lo + pad) & (mu < hi - pad), mu, np.nan)

    a, b = fit(ms), fit(mn)
    both = np.isfinite(a) & np.isfinite(b)
    diff = float(np.median(a[both] - b[both])) if both.sum() else float("nan")

    out = {"difference_m": diff, "n_pixels": int(both.sum()),
           "n_spring": int(ms.sum()), "n_neap": int(mn.sum()),
           "window_m": (float(lo), float(hi)),
           "range_spring_m": float(np.nanmean(day_range[ms])),
           "range_neap_m": float(np.nanmean(day_range[mn]))}
    if verbose:
        print(f"[tidecheck] common tide window {lo:+.2f}..{hi:+.2f} m; "
              f"{out['n_spring']} spring and {out['n_neap']} neap scenes "
              f"after matching")
        print(f"[tidecheck] mean daily range: spring "
              f"{out['range_spring_m']:.2f} m, neap {out['range_neap_m']:.2f} m")
        print(f"[tidecheck] spring minus neap elevation: {diff:+.3f} m "
              f"over {out['n_pixels']:,} pixels")
        if np.isfinite(diff) and abs(diff) > 0.05:
            sense = "amplifies" if diff < 0 else "damps"
            print(f"[tidecheck] the estuary {sense} large tides relative to "
                  f"the model — an error no single-alpha correction removes")
    return out


# ─────────────────────────────────────────────────────────────────────────────

def _fit_subset(Y, C, tide, mask, mu_points, sg_grid, min_obs, min_b,
                grid=None):
    """Fit the sigmoid on a subset of scenes; NaN where it is not resolved."""
    m = np.asarray(mask, bool) & np.isfinite(tide)
    t = tide[m]
    if t.size < 3:
        return np.full(Y.shape[1], np.nan)
    g = np.linspace(t.min(), t.max(), mu_points) if grid is None else grid
    a, b, mu, sg, rmse, N = _fit_block(
        np.nan_to_num(Y[m], nan=0.0).astype(np.float64),
        C[m].astype(np.float64), t, g, sg_grid)
    return np.where((N >= min_obs) & (b > min_b), mu, np.nan)
