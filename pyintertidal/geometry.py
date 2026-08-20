"""
geometry.py — The estuary's shape, measured from the archive itself (fase M1)
=============================================================================

Everything the interior-tide estimators need to know about the basin's shape
comes from four objects, all derived from the imagery with no external map:

* the ALONG-ESTUARY coordinate ``s``: geodesic distance from the mouth,
  travelling only through pixels water can occupy. This is not the distance
  to the nearest channel — it is "how far into the estuary am I", the axis
  every transfer profile lives on;
* the THALWEG: the skeleton of the permanent-water mask, for display and for
  sanity checks (s must grow monotonically along it);
* WIDTH(s, h): how wide the wetted estuary is at each along-coordinate for
  each tide level — the lateral-storage observable that plan v4 notes is
  measured, not inferred, which is what breaks the circularity with the
  target bathymetry;
* the CONVERGENCE γ_c(s) = −d ln(width)/ds: how fast the funnel narrows,
  the quantity hydraulic closures trade against friction.

Level masks come from tide-binned composites of the archive: a pixel counts
as wet at a level if it was wet in at least half of its usable observations
within that tide bin. Uncertainty is by bootstrap over SCENES (not pixels —
scenes are the independent unit here).

Hard rules honoured: R2 (no survey data anywhere), R7 (seeds explicit).
"""

from __future__ import annotations

import numpy as np


def mouth_seeds(sea):
    """The seaward opening: permanent-water pixels touching the image border.

    Verified for Villaviciosa at adoption: the open-sea component reaches the
    border (canal.py, 2026-08-18); an estuary clipped away from the sea would
    yield no seeds, which is a loud error and should be.
    """
    sea = np.asarray(sea, bool)
    border = np.zeros_like(sea)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    seeds = sea & border
    if not seeds.any():
        raise ValueError("el mar permanente no toca el borde de la imagen: "
                         "no hay boca desde la que medir s")
    return seeds


def along_distance(reachable, seeds, pixel_m):
    """Geodesic distance from the mouth through water-occupiable ground (m).

    NaN where water cannot reach at all — hydraulically isolated pixels,
    which is a finding, not a nuisance.
    """
    from skimage.graph import MCP_Geometric

    costs = np.where(np.asarray(reachable, bool), 1.0, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True)
    dist, _ = mcp.find_costs(starts=[tuple(ix) for ix in np.argwhere(seeds)])
    dist = np.asarray(dist, float) * float(pixel_m)
    dist[~np.isfinite(dist)] = np.nan
    return dist


def thalweg(sea):
    """Skeleton of the permanent-water mask — the channel's centreline."""
    from skimage.morphology import skeletonize

    return skeletonize(np.asarray(sea, bool))


def level_masks(wet_frac_by_bin, sea, min_frac=0.5):
    """Wet mask per tide bin: permanent water plus reliably-wet intertidal."""
    return [np.asarray(sea, bool) | (wf >= min_frac)
            for wf in wet_frac_by_bin]


def wet_fraction_by_bin(store_keep, wet_counts, obs_counts, shape,
                        min_obs=3):
    """Per-pixel wet fraction on the full grid from per-bin counts."""
    out = np.full(shape, np.nan, np.float32)
    frac = np.where(obs_counts >= min_obs,
                    wet_counts / np.maximum(obs_counts, 1), np.nan)
    out.ravel()[store_keep] = frac
    return out


def width_profile(mask, s_m, band_m=250.0, s_max=None):
    """Wetted width (m) per along-coordinate band.

    Width = wetted area in the band / band length — the standard estuarine
    definition, robust to a sinuous channel.
    """
    mask = np.asarray(mask, bool)
    s = np.asarray(s_m, float)
    if s_max is None:
        s_max = np.nanmax(np.where(mask, s, np.nan))
    edges = np.arange(0.0, s_max + band_m, band_m)
    centers, widths = [], []
    for a, b in zip(edges, edges[1:]):
        sel = mask & (s >= a) & (s < b)
        centers.append(0.5 * (a + b))
        widths.append(float(sel.sum()) * 100.0 / band_m)   # px de 10 m
    return np.asarray(centers), np.asarray(widths)


def flood_threshold(z_map, sea, valid):
    """Level at which each pixel first floods, given water needs a PATH.

    Minimax elevation over paths from the sea — exactly what grayscale
    depression filling returns. A pixel in a pit gets a threshold ABOVE its
    own elevation; the bathtub model cannot express that. Promoted from the
    2026-08-18 lamina prototype into the package for phase B6: the spill
    surface is what the wet/dry archive actually OBSERVES for pit pixels,
    so the DEM must declare terrain-vs-spill per pixel.
    """
    from skimage.morphology import reconstruction

    zz = np.where(valid, np.asarray(z_map, float), np.nan)
    hi = np.nanmax(zz) + 1.0
    mask = np.where(np.isfinite(zz), zz, hi)
    seed = np.full_like(mask, hi)
    seed[np.asarray(sea, bool)] = mask[np.asarray(sea, bool)]
    filled = reconstruction(seed, mask, method="erosion")
    return np.where(valid, filled, np.nan)


def convergence(s_centers_m, width_m, smooth=None):
    """γ_c(s) = −d ln(width)/ds via a smoothing spline; e-folding length.

    scipy's smoothing spline is the penalised-spline tool the stack allows
    (R5). Returns (gamma_per_km, efold_km, spline).
    """
    from scipy.interpolate import UnivariateSpline

    ok = np.isfinite(width_m) & (width_m > 0)
    s_km = np.asarray(s_centers_m, float)[ok] / 1000.0
    lw = np.log(np.asarray(width_m, float)[ok])
    if smooth is None:
        smooth = len(s_km) * np.var(lw) * 0.1
    sp = UnivariateSpline(s_km, lw, s=smooth, k=3)
    gamma = -sp.derivative()(s_km)
    with np.errstate(divide="ignore"):
        efold = np.where(gamma > 1e-6, 1.0 / gamma, np.inf)
    return gamma, efold, sp
