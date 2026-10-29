# -*- coding: utf-8 -*-
"""V11 MAREA-M (monotone): the elevation read where a pixel's isotonic wet probability reaches 0.5, with no curve
shape (experiments/variants_addendum_2026-10-28.md, row V11; V11d is the same call with the levels read without
delay).

What it computes. Per pixel, the wet indicator of its clear looks is regressed on the level h by isotonic
(non-decreasing) least squares, the pool-adjacent-violators fit, with the looks at equal h pooled first; p, the fitted
wet probability, is a step function of h. h_hi is the lowest of the pixel's levels with p >= 0.5, h_lo the highest of
its levels strictly below h_hi, and the elevation is z = (h_lo + h_hi)/2. When every level has p >= 0.5, z is the
pixel's lowest level (censored low, -1); when none has, its highest level (censored high, +1); a censored pixel keeps
its value and is flagged. "Level" is always one of the pixel's own clear looks: no grid is used. The guards are V2's:
at least 8 clear looks, with at least one wet and one dry. wet = clear and NDWI > 0, a clear look with an undefined
NDWI being dry, unless the caller gives the wet looks (V14).

How, without a pool-adjacent-violators loop per pixel: the level sets of the isotonic fit. Take the looks in h order,
equal levels kept together, and a maximal run of them sharing one fitted value v. Because the fit is optimal, every
lower part of the run (its first levels) has a mean wet indicator of at least v and every upper part (its last
levels) a mean of at most v; otherwise lowering that lower part, or raising that upper part, a little would cut the
squared error without breaking the order. So, for any c, the looks with p >= c are the longest suffix (whole levels,
from one level to the top) that maximises S = the sum of (wet - c) over the suffix. A longer suffix adds runs with
fitted value below c, whole or their upper part, whose mean is then below c too: S falls strictly. A shorter one
drops runs with fitted value >= c, whole or their lower part, whose mean is then >= c: S does not rise. With c = 0.5,
2S is the number of wet looks minus dry looks in the suffix, an exact integer. One reverse cumulative sum over the
looks in h order, read at each level's first look, gives 2S at every level for every pixel at once; its first maximum
(np.argmax, the empty suffix with 2S = 0 counted last) is the longest maximising suffix, and that suffix's lowest
level is h_hi. The levels h are common to every pixel of a call (one delay group, as in run_marea_b), so the looks
are sorted and pooled once per call. tests/test_variants_monotone.py checks the read-off against a plain
pool-adjacent-violators on many random pixels.

The notebook calls it as it calls invert_bernoulli: once per delay group, with Y = Y_S[np.ix_(w, px)],
C = C_S[np.ix_(w, px)] and h = level_at(tau)[w] (V11d: h = LEV[0][w] for every pixel).
"""
import numpy as np


def invert_monotone(Y, C, h, wet=None, min_looks=8, chunk=4000):
    """V11: per pixel, the isotonic (non-decreasing) regression of the wet indicator of its clear looks on the levels
    h, looks at equal h pooled first, read where the fitted wet probability p reaches 0.5: z = (h_lo + h_hi)/2, where
    h_hi is the lowest of the pixel's levels with p >= 0.5 and h_lo its highest level strictly below h_hi. When every
    level has p >= 0.5, z = the pixel's lowest level and censored = -1; when none has, z = its highest level and
    censored = +1.

    Y (T, P) NDWI with NaN where undefined; C (T, P) the clear looks; h (T,) the levels, finite (read with the delay
    of the call's pixels). wet (T, P), optional: the wet looks (V14); by default C & (Y > 0), so an undefined NDWI is
    never wet; a wet look outside C is ignored, and Y is not read when wet is given. z is NaN, and censored 0, where
    the pixel has fewer than min_looks clear looks or lacks a wet or a dry one. Works on blocks of `chunk` pixels, as
    invert_series and invert_bernoulli. Returns (z, censored, n_wet, n_dry): z float64, the other three int. n_wet
    and n_dry, the pixel's wet and dry clear looks, are given for every pixel, also where z is NaN."""
    h = np.asarray(h, np.float64)
    T, P = np.shape(C)
    if h.shape != (T,):
        raise ValueError(f"h has shape {h.shape}; the looks are {T}")
    if not np.isfinite(h).all():
        raise ValueError("every level h must be finite: a look without a level cannot be placed in h order")
    z, censored = np.full(P, np.nan), np.zeros(P, int)
    n_wet, n_dry = np.zeros(P, int), np.zeros(P, int)
    if T == 0:
        return z, censored, n_wet, n_dry

    order = np.argsort(h, kind="stable")
    hs = h[order]
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])     # where each distinct level begins in h order
    levels = hs[starts]                                         # the window's distinct levels, ascending: (L,)
    L = len(levels)
    rank = np.arange(L)[:, None]
    for j in range(0, P, chunk):
        s = slice(j, min(j + chunk, P))
        clear = np.asarray(C[:, s], bool)
        if wet is None:
            with np.errstate(invalid="ignore"):
                w = clear & (np.asarray(Y[:, s]) > 0)            # an undefined NDWI is never wet
        else:
            w = clear & np.asarray(wet[:, s], bool)
        nc, nw = clear.sum(axis=0), w.sum(axis=0)

        # 2S of the suffix from each look to the top, in h order: +1 per wet look, -1 per dry one, 0 if not clear
        cs, ws = clear[order], w[order]
        S = np.cumsum((ws.astype(np.int8) * np.int8(2) - cs)[::-1], axis=0, dtype=np.int32)[::-1]
        if L < T:                       # tied levels: a suffix starts at a level's first look and keeps it whole
            S, has = S[starts], np.logical_or.reduceat(cs, starts, axis=0)
        else:
            has = cs
        # has: the levels of the pixel's own clear looks. A suffix starting at a level the pixel has no look at holds
        # the same looks as the one starting at its next level, so only the pixel's levels are candidates; the empty
        # suffix (2S = 0) counts last, so it is taken only when every candidate is negative
        S = np.where(has, S, np.iinfo(np.int32).min)
        k = np.argmax(S, axis=0)                                # the first maximum: the longest maximising suffix
        k = np.where(S[k, np.arange(len(k))] >= 0, k, L)
        first = np.argmax(has, axis=0)                          # the pixel's lowest level,
        top = L - 1 - np.argmax(has[::-1], axis=0)              # its highest,
        below = L - 1 - np.argmax((has & (rank < k))[::-1], axis=0)    # and its highest below level k
        low, high = k == first, k == L                          # every level has p >= 0.5 / none has
        h_hi = levels[np.minimum(k, L - 1)]
        zb = np.where(high, levels[top], np.where(low, h_hi, 0.5 * (levels[below] + h_hi)))

        ok = (nc >= min_looks) & (nw > 0) & (nw < nc)          # V2's guards
        z[s] = np.where(ok, zb, np.nan)
        censored[s] = np.where(ok, np.where(low, -1, np.where(high, 1, 0)), 0)
        n_wet[s], n_dry[s] = nw, nc - nw
    return z, censored, n_wet, n_dry
