# -*- coding: utf-8 -*-
"""V14 of the addendum of 2026-10-28 (experiments/variants_addendum_2026-10-28.md): the water threshold learnt from
each scene, a factor applied to every wet/dry estimator ("+ Otsu").

For each scene s, t_s is the Otsu threshold of the NDWI of the scene's clear extraction pixels with a defined NDWI,
on 256 bins of [-1, 1]; when t_s falls outside [-0.2, 0.3] (a scene without two classes, e.g. all water at high
tide), or the scene has fewer than 2 such pixels, t_s = 0, the common configuration's threshold. Wet = clear and
NDWI > t_s; a clear look with an undefined NDWI is dry, as with NDWI > 0. MAREA-B, MAREA-BL and MAREA-M take that wet
matrix itself (`wet_from_threshold`); the competitors (LT2D, sampled exceedance, the wet/dry bracket) take NDWI - t_s
(`shift_for_threshold`), so that their own rule wet = NDWI > 0 is wet = NDWI > t_s without a change to their code.
Both are computed in float64, and for a float64 a and a finite b, a - b > 0 exactly when a > b (rounding keeps the
sign of a difference, and with gradual underflow a difference of two distinct floats is never rounded to 0), so the
two routes give the same wet matrix bit for bit; the shift leaves NaN NaN and a finite NDWI finite, so the
competitors' valid looks (clear with a defined NDWI) do not change either. MAREA's delay fit is unchanged (V0's,
with NDWI > 0).

    otsu_threshold       t of one set of values
    scene_thresholds     t_s of every scene (the rows of Y), one scene at a time, and where the fallback was used
    shift_for_threshold  Y - t_s per scene: the competitors' input
    wet_from_threshold   C & (Y > t_s) per scene: the explicit wet matrix

In the attribution notebook (Y_S, C_S: every scene of S at every extraction pixel; w, px: a window's scenes and
pixels):

    t, fell_back = scene_thresholds(Y_S, C_S)                              # once per site
    wet = wet_from_threshold(Y_S[np.ix_(w, px)], C_S[np.ix_(w, px)], t[w])   # MAREA-B, -BL, -M + Otsu
    Y_otsu = shift_for_threshold(Y_S[np.ix_(w, px)], t[w])                  # obs.Y of E4, E5c, E12 + Otsu, with
                                                                           # obs.W = C & (Y_otsu > 0), as make_obs

The histogram, and the point of the bin returned
------------------------------------------------
The edges are e_j = lo + j (hi - lo)/bins, j = 0 .. bins (np.linspace), and bin k holds e_k < v <= e_{k+1}: closed
on the right, bin 0 also holding lo. A value below lo counts in bin 0 and one above hi in the last bin, so that every
pixel with a defined NDWI enters the histogram (an NDWI lies in [-1, 1] unless a reflectance is negative). A split
after bin k makes bins 0 .. k the dry class and bins k+1 .. bins-1 the wet class. With the bin index i as Otsu's
grey level, n_i the counts, N = sum n_i, S = sum i n_i, w_k = sum_{i<=k} n_i and s_k = sum_{i<=k} i n_i, the
between-class variance of the split is (Otsu 1979)

    sigma_B^2(k) = (S w_k - N s_k)^2 / (N^2 w_k (N - w_k)),

maximised over the splits that leave both classes non-empty (the bin centres are an affine map of i: that scales
sigma_B^2 by a constant and does not move its maximum). The threshold returned is the UPPER EDGE of the last dry bin,
t = e_{k*+1}. With bins closed on the right, NDWI > t is then exactly the wet class of the split, value by value, so
the rule applied to the pixels is the partition Otsu chose; a bin centre (scikit-image's choice) would put the upper
half of the last dry bin on the wet side of the rule. The edge also makes the common threshold one of Otsu's
candidates: 0 is the edge e_128 of the default binning, and an NDWI of exactly 0 (B03 = B08) lies in the bin below
it, dry, as NDWI > 0 has it. The default thresholds are multiples of 1/128, exact in binary.

Ties: when the classes are separated by empty bins, every split inside the gap has the same classes and the same
sigma_B^2, bit for bit, and the first maximum (the lowest split, as scikit-image and OpenCV take it) is returned.
Every split in the gap classifies the scene's own pixels the same way; only the range check of `scene_thresholds`
can tell them apart. With fewer than 2 values, or every value in one bin, no split leaves two non-empty classes:
NaN, and `scene_thresholds` falls back.

Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and
Cybernetics 9(1), 62-66. doi:10.1109/TSMC.1979.4310076
"""
from __future__ import annotations

import numpy as np

BINS = 256                  # the addendum's histogram: 256 bins on [LO, HI]
LO, HI = -1.0, 1.0
VALID = (-0.2, 0.3)         # a t_s outside this closed interval: the scene has no two classes, the fallback is used
FALLBACK = 0.0              # the common configuration's threshold, wet = NDWI > 0
MIN_PIXELS = 2              # a scene with fewer clear pixels with a defined NDWI: the fallback


def _edges(bins, lo, hi):
    """The bins + 1 edges of the histogram on [lo, hi]."""
    if int(bins) != bins or bins < 2:
        raise ValueError(f"bins must be an integer >= 2 (one split at least), got {bins!r}")
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"the histogram needs finite lo < hi, got [{lo}, {hi}]")
    return np.linspace(lo, hi, int(bins) + 1)


def _bin_index(v, edges):
    """The bin of every value: k with e_k < v <= e_{k+1} (bins closed on the right, bin 0 also holding lo); below lo,
    bin 0; above hi, the last bin. The values are compared with the edges themselves, so the rule v > e_{k+1} and the
    bin agree exactly."""
    return np.clip(np.searchsorted(edges, v, side="left") - 1, 0, len(edges) - 2)


def _otsu_split(counts):
    """k*, the last bin of the dry class of the split that maximises the between-class variance of the histogram
    `counts` (the first maximum: the lowest such split); -1 when no split leaves both classes non-empty."""
    n = np.asarray(counts, np.float64)
    i = np.arange(n.size, dtype=np.float64)
    w = np.cumsum(n)[:-1]                       # pixels in bins 0..k, for the splits k = 0 .. bins-2
    s = np.cumsum(n * i)[:-1]                   # the sum of their bin indices
    N, S = n.sum(), float(n @ i)
    both = (w > 0) & (w < N)                    # a dry and a wet pixel at least
    if not both.any():
        return -1
    # N^2 sigma_B^2(k). The counts and sums are integers, so d is exact in float64 while N^2 (bins - 1) < 2^53
    # (N below 5.9 million values at 256 bins); with both classes non-empty their means differ and d is not 0
    d = S * w - N * s
    crit = np.where(both, d * d / np.where(both, w * (N - w), 1.0), -1.0)
    return int(np.argmax(crit))                 # argmax: the first maximum


def otsu_threshold(values, bins=BINS, lo=LO, hi=HI):
    """The Otsu threshold t of `values` (any shape; NaN and +-inf are left out) on `bins` bins of [lo, hi]: the upper
    edge of the last bin of the lower class of the split that maximises the between-class variance, the first
    maximum on a tie, so that values > t is exactly that split's upper class (the module docstring says why).
    NaN with fewer than 2 values or every value in one bin (no split has two classes). float64 arithmetic."""
    edges = _edges(bins, lo, hi)
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    k = _otsu_split(np.bincount(_bin_index(v, edges), minlength=len(edges) - 1))
    return float(edges[k + 1]) if k >= 0 else np.nan


def scene_thresholds(Y, C, bins=BINS, lo=LO, hi=HI, valid=VALID, fallback=FALLBACK):
    """V14's t_s of every scene: the Otsu threshold (`otsu_threshold`) of the NDWI of its clear pixels with a defined
    NDWI, or `fallback` when the scene has fewer than 2 such pixels or its Otsu threshold is not inside the closed
    interval `valid` (NaN, no split, is not). Y (scenes x pixels) NDWI with NaN where undefined, the columns the
    extraction's pixels; C (scenes x pixels) the clear looks. One scene at a time, so the extraction (at the Wadden
    ~150 scenes x 1.2 million pixels, float32) is never copied whole. Returns (t float64 (scenes,), used_fallback
    bool (scenes,))."""
    shape = tuple(np.shape(Y))                  # np.shape reads .shape: a lazy or memory-mapped Y is not loaded
    if len(shape) != 2 or tuple(np.shape(C)) != shape:
        raise ValueError(f"Y and C must be scenes x pixels of the same shape, got {shape} and {np.shape(C)}")
    _edges(bins, lo, hi)                        # a bad binning is an error even without a scene to bin
    v_lo, v_hi = (float(x) for x in valid)
    t = np.full(shape[0], float(fallback))
    used = np.ones(shape[0], bool)
    for s in range(shape[0]):
        y = np.asarray(Y[s])                    # one scene: (pixels,)
        v = y[np.asarray(C[s], bool) & np.isfinite(y)]
        if v.size < MIN_PIXELS:
            continue
        ts = otsu_threshold(v, bins, lo, hi)
        if v_lo <= ts <= v_hi:                  # False for NaN
            t[s], used[s] = ts, False
    return t, used


def _per_scene(t, Y):
    """t as float64 (rows of Y,): one finite threshold per scene (a scalar is every scene's)."""
    if Y.ndim != 2:
        raise ValueError(f"Y must be looks x pixels, got shape {Y.shape}")
    t = np.asarray(t, np.float64)
    if t.ndim == 0:
        t = np.full(Y.shape[0], float(t))
    if t.shape != (Y.shape[0],):
        raise ValueError(f"one threshold per look: t has shape {t.shape}, Y has {Y.shape[0]} looks")
    if not np.all(np.isfinite(t)):
        raise ValueError("every threshold must be finite (scene_thresholds falls back instead of giving NaN)")
    return t


def shift_for_threshold(Y, t):
    """Y - t[:, None] in float64: the competitors' input for V14, so that their rule wet = NDWI > 0 is NDWI > t_s
    scene by scene (rows of Y: the window's looks; t: their thresholds, or one scalar for every look). NaN stays NaN
    and a finite NDWI stays finite. With t = 0 it is Y itself (as float64), bit for bit."""
    Y = np.asarray(Y)
    t = _per_scene(t, Y)
    return np.subtract(Y, t[:, None], dtype=np.float64)


def wet_from_threshold(Y, C, t):
    """The explicit wet matrix of V14, C & (Y > t[:, None]) (rows of Y: the window's looks; t: their thresholds, or
    one scalar for every look): a clear look with an undefined NDWI is dry. Compared in float64, so it is
    C & (shift_for_threshold(Y, t) > 0) bit for bit; with t = 0 it is the common configuration's C & (Y > 0)."""
    Y, C = np.asarray(Y), np.asarray(C, bool)
    t = _per_scene(t, Y)
    if C.shape != Y.shape:
        raise ValueError(f"C has shape {C.shape}, Y has {Y.shape}")
    wet = np.empty(Y.shape, bool)
    with np.errstate(invalid="ignore"):
        np.greater(Y, t[:, None], out=wet)      # t is a float64 array: compared in float64; NaN > t is False
    wet &= C
    return wet
