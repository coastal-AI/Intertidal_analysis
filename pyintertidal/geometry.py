"""
geometry.py — Where the mouth is and how far each pixel is from it
==================================================================

MAREA needs one coordinate per pixel: ``s``, the distance from the mouth
travelling only through ground water can occupy (permanent water and
intertidal pixels). It is not the distance to the nearest channel — it is
"how far into the estuary am I". The delay of the tide is measured section
by section along s and applied to every pixel along s. Everything here is
derived from the images; no map or survey is used.

Used by MAREA (``marea.reconstruct``, step 7)
---------------------------------------------
:func:`mouth_seeds`
    The mouth: the permanent-water pixels on the image border.
:func:`along_distance`
    s, in metres, from those seeds (NaN where water cannot reach).

Tools for describing an estuary's shape (not used by MAREA)
-----------------------------------------------------------
:func:`water_pieces`, :func:`pieces_by_size`
    The connected pieces of water-occupiable ground, and which reach the
    mouth.
:func:`thalweg`
    The centreline (skeleton) of permanent water, for display and checks
    (s should grow along it).
:func:`width_profile`, :func:`convergence`
    How wide the wetted estuary is along s, and how fast it narrows:
    γ_c(s) = −d ln(width)/ds.
:func:`level_masks`, :func:`wet_fraction_by_bin`
    Wet masks per tide level, from looks binned by tide: a pixel counts as
    wet at a level when it was wet in at least half of its looks in that bin.
:func:`flood_threshold`
    The level at which each pixel first floods along a path from the sea.

Grids are 2-D ``(H, W)`` arrays on the cube grid, rows north to south.
"""

from __future__ import annotations

import numpy as np


def mouth_seeds(sea, side="any"):
    """The seaward opening: permanent-water pixels touching the image border.

    Every permanent-water pixel on the chosen edge(s) counts as mouth. An
    estuary clipped away from the sea yields no seeds, which is a loud error
    and should be (at Villaviciosa the open sea reaches the border).

    Parameters
    ----------
    sea : (H, W) bool
        Permanent water (``marea.extract``: wet in at least 90 % of the
        clear looks).
    side : str
        Which borders count: ``"any"`` (all four), or ``"west"``,
        ``"east"``, ``"north"``, ``"south"``, several joined by ``"+"``
        (e.g. ``"west+north"``). Use one side for a box the estuary crosses —
        the Westerschelde leaves the frame on the east too, and measuring
        distance from both ends would put the inner reach next to a second
        "mouth".

    Returns
    -------
    (H, W) bool, True on the mouth pixels. Raises ``ValueError`` (the
    message is in Spanish: "the permanent sea does not touch the image
    border: there is no mouth to measure s from") when no permanent water
    touches the chosen border.
    """
    sea = np.asarray(sea, bool)
    border = np.zeros_like(sea)
    sides = {s.strip().lower() for s in str(side).split("+")}
    if "any" in sides:
        sides = {"north", "south", "west", "east"}
    if "north" in sides:
        border[0, :] = True
    if "south" in sides:
        border[-1, :] = True
    if "west" in sides:
        border[:, 0] = True
    if "east" in sides:
        border[:, -1] = True
    seeds = sea & border
    if not seeds.any():
        raise ValueError("el mar permanente no toca el borde de la imagen: "
                         "no hay boca desde la que medir s")
    return seeds


def along_distance(reachable, seeds, pixel_m):
    """Geodesic distance from the mouth through water-occupiable ground (m).

    NaN where water cannot reach at all — hydraulically isolated pixels,
    which is a finding, not a nuisance.

    Parameters
    ----------
    reachable : (H, W) bool
        Ground water can occupy (MAREA: permanent water | intertidal).
    seeds : (H, W) bool
        The mouth (:func:`mouth_seeds`).
    pixel_m : float
        Pixel size in metres (10 for MAREA).

    Returns
    -------
    (H, W) float, metres: the shortest path through ``reachable`` moving
    between pixels that share an edge or a corner (8-connected; a diagonal
    step counts √2 pixels; scikit-image ``MCP_Geometric``); NaN where no
    such path exists (a pond, a marsh behind a gate, a channel narrower
    than a pixel).
    """
    from skimage.graph import MCP_Geometric

    costs = np.where(np.asarray(reachable, bool), 1.0, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True)
    dist, _ = mcp.find_costs(starts=[tuple(ix) for ix in np.argwhere(seeds)])
    dist = np.asarray(dist, float) * float(pixel_m)
    dist[~np.isfinite(dist)] = np.nan
    return dist


def pieces_by_size(image, connectivity=8):
    """The connected pieces of a 2-D boolean image, numbered by size.

    Pixels join through edges and corners (``connectivity=8``) or through
    edges only (4). Returns ``(pieces, n)``: ``pieces`` int64, of the
    image's shape, 0 outside its True pixels and the pieces 1, 2, ..., n
    by size, the largest first (equal sizes: the one whose first pixel,
    row by row, comes first). The one numbering of :func:`water_pieces`.
    """
    from scipy import ndimage

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity is 4 (through edges) or 8 (edges "
                         f"and corners), not {connectivity}")
    through = ndimage.generate_binary_structure(     # the cross, or the
        2, 1 if connectivity == 4 else 2)            # 3 x 3 block
    lab, n = ndimage.label(np.asarray(image, bool), structure=through)
    flat = lab.ravel()
    on = np.flatnonzero(flat)                       # row by row
    size = np.bincount(flat, minlength=n + 1)[1:]
    first = np.zeros(n, np.int64)
    found, at = np.unique(flat[on], return_index=True)
    first[found - 1] = on[at]                       # each piece's first pixel
    number = np.zeros(n + 1, np.int64)
    number[np.lexsort((first, -size)) + 1] = np.arange(1, n + 1)
    return number[lab], n


def water_pieces(reachable, seeds):
    """The pieces of water-occupiable ground, and which reach the mouth.

    :func:`along_distance` travels between pixels that touch at an edge or
    a corner (its MCP is fully connected), so its distance is finite
    exactly on the 8-connected pieces of ``reachable`` that hold a seed.
    Every other piece has no water path to the mouth, and no distance: a
    pond, a marsh behind a gate, or water the mask lost where a channel is
    narrower than a pixel.

    Returns ``(pieces, to_mouth)``. ``pieces`` (H, W) int64: 0 outside
    ``reachable``, the pieces 1, 2, ... by size, the largest first (equal
    sizes: the one whose first pixel, row by row, comes first;
    :func:`pieces_by_size`). ``to_mouth``: one bool per piece number,
    True for a piece that holds a seed (index 0, outside ``reachable``,
    is False), so that ``to_mouth[pieces]`` is
    ``np.isfinite(along_distance(reachable, seeds, pixel_m))``; a seed
    outside ``reachable`` joins no piece (there along_distance gives that
    pixel alone a distance of 0).
    """
    reachable = np.asarray(reachable, bool)
    seeds = np.asarray(seeds, bool)
    if reachable.ndim != 2 or seeds.shape != reachable.shape:
        raise ValueError(f"reachable and seeds are one (H, W) grid, not "
                         f"{reachable.shape} and {seeds.shape}")
    pieces, n = pieces_by_size(reachable, connectivity=8)
    to_mouth = np.zeros(n + 1, bool)
    to_mouth[pieces[seeds & reachable]] = True
    to_mouth[0] = False
    return pieces, to_mouth


def thalweg(sea):
    """Skeleton of the permanent-water mask — the channel's centreline.

    ``sea`` (H, W) bool; returns an (H, W) bool line one pixel wide.
    """
    from skimage.morphology import skeletonize

    return skeletonize(np.asarray(sea, bool))


def level_masks(wet_frac_by_bin, sea, min_frac=0.5):
    """Wet mask per tide bin: permanent water plus reliably-wet intertidal.

    ``wet_frac_by_bin`` is a list of (H, W) wet fractions, one per tide bin
    (:func:`wet_fraction_by_bin`); ``sea`` (H, W) bool; a pixel is wet in a
    bin when its fraction is at least ``min_frac``. Returns a list of (H, W)
    bool masks.
    """
    return [np.asarray(sea, bool) | (wf >= min_frac)
            for wf in wet_frac_by_bin]


def wet_fraction_by_bin(store_keep, wet_counts, obs_counts, shape,
                        min_obs=3):
    """Per-pixel wet fraction on the full grid from per-bin counts.

    Parameters
    ----------
    store_keep : (P,) int
        Flat indices (``row * W + col``) of the P pixels the counts belong
        to (like ``marea.extract``'s ``keep``).
    wet_counts, obs_counts : (P,) int
        Wet looks and clear looks of each pixel in one tide bin.
    shape : (H, W)
        The full grid.
    min_obs : int
        Fewer clear looks than this gives NaN.

    Returns an (H, W) float32 grid, NaN outside the P pixels.
    """
    out = np.full(shape, np.nan, np.float32)
    frac = np.where(obs_counts >= min_obs,
                    wet_counts / np.maximum(obs_counts, 1), np.nan)
    out.ravel()[store_keep] = frac
    return out


def width_profile(mask, s_m, band_m=250.0, s_max=None):
    """Wetted width (m) per along-coordinate band.

    Width = wetted area in the band / band length — the standard estuarine
    definition, robust to a sinuous channel.

    The pixel area is FIXED at 100 m², i.e. a 10 m grid: on a 20 m grid
    multiply the widths by 4.

    Parameters
    ----------
    mask : (H, W) bool
        The wetted area (e.g. one of :func:`level_masks`).
    s_m : (H, W) float
        Distance from the mouth, metres (:func:`along_distance`).
    band_m : float
        Length of each band along s, metres.
    s_max : float, optional
        Upper end of the profile, metres (default: the largest s in mask).

    Returns
    -------
    (centres_m, widths_m) : two 1-D float arrays, metres.
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
        widths.append(float(sel.sum()) * 100.0 / band_m)   # assumes 10 m pixels (100 m² each)
    return np.asarray(centers), np.asarray(widths)


def flood_threshold(z_map, sea, valid):
    """Level at which each pixel first floods, given water needs a PATH.

    Minimax elevation over paths from the sea — exactly what grayscale
    depression filling returns. A pixel in a pit gets a threshold ABOVE its
    own elevation; the bathtub model cannot express that. For a pit pixel
    the wet/dry archive observes this spill level, not the ground, which is
    why a DEM made from it should say, pixel by pixel, which of the two it
    holds.

    Parameters
    ----------
    z_map : (H, W) float
        Elevations, metres.
    sea : (H, W) bool
        Where water enters (the seeds of the filling).
    valid : (H, W) bool
        Pixels with an elevation; the others are treated as higher than
        every elevation (+1 m above the maximum).

    Returns an (H, W) float grid, metres, NaN outside ``valid``.
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

    scipy's smoothing spline is the penalised-spline tool used here.
    Returns (gamma_per_km, efold_km, spline).

    ``s_centers_m`` and ``width_m`` are :func:`width_profile`'s output
    (metres); only bands with a finite, positive width are used, so the
    returned arrays may be shorter than the input. ``smooth`` is the
    spline's smoothing factor (default ``n × var(ln width) × 0.1``). The
    spline maps s in km to ln(width); ``efold_km`` is infinite where the
    estuary does not narrow.
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
