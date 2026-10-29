# -*- coding: utf-8 -*-
"""V4, MAREA's robust fit (declared 2026-10-27; run as experiments/variants_addendum_2026-10-28.md fixes it).

V0's per-pixel fit NDWI = a + b Phi((h - z)/sigma), on the grid of `marea.invert_series` (mu_points levels between
the window's lowest and highest level, sigma in `marea.SG_GRID`), with a Huber loss on the NDWI residuals
(delta = 0.10) in place of least squares and weight 0.5 for the looks of SCL class 7. The Huber fit is computed by
iteratively reweighted least squares on `pyintertidal.elevation._fit_block`, which takes a float weight per look:

    w = C s                            the start: V0's least-squares fit, weighted (s = 0.5 for SCL 7, else 1)
    r = Y - (a + b Phi((h - mu)/sigma))  the residuals of the current fit, on the clear looks
    w = C s min(1, delta/|r|)          Huber's weights (w = C s where r = 0), and the weighted fit again

until the pixel's grid point (mu, sigma) and (a, b) repeat exactly, or after max_iter refits. Every refit is a full
`_fit_block` sweep of the (mu, sigma) grid, so only the pixels whose fit has not repeated are refitted. Two readings
of the final fit, with `invert_series`' guards (at least min_obs clear looks, min_b < b < max_b, |a| < max_a):

* V4 = the midpoint z = mu (as V0);
* V4-0 = the level where the fitted NDWI crosses 0 (as V1, the builder's `marea_fit`, notebook 01's
  `invert_ndwi0`): z0 = mu + sigma Phi^-1(-a/b), no value unless a < 0 < a + b.

The inputs are V0's, as `run_marea` passes them: Y (looks x pixels) NDWI, a clear look with an undefined NDWI
entering as 0 (a NaN left in Y is read as 0 here too); C the clear looks; h the levels read with MAREA's delay;
lo, hi the window's lowest and highest level without delay (the grid bounds). With delta = inf and no SCL 7 every
weight is the clear mask, the first refit repeats the start, and V4 is V0 (`invert_series`) and V4-0 is V1, bit for
bit.

Floating point: `_fit_block`'s sums go through BLAS, whose rounding of one pixel's sums can change in the last bit
with the other columns of its block. The first refit is therefore always done on the whole block, as the start was
(so the delta = inf limit is exact); later refits gather the pixels still moving, and a pixel there can, rarely, take
one more refit to repeat itself exactly.
"""
import numpy as np
from scipy.special import erf, ndtri

from pyintertidal import marea
from pyintertidal.elevation import _fit_block

DELTA = 0.10              # Huber's threshold on the NDWI residual (the variants file, 2026-10-27)
SCL7_WEIGHT = 0.5         # the weight of a look of SCL class 7 (unclassified, the likeliest missed haze)
MAX_ITER = 20             # the cap on the refits of a pixel (the addendum)


def huber_weights(Y, base, h, a, b, mu, sg, delta=DELTA):
    """Huber's IRLS weights of the looks of every pixel under its current fit: base min(1, delta/|r|), with
    r = Y - (a + b Phi((h - mu)/sigma)) and base = C s the starting weights (0 off the clear looks). A pixel without
    a fitted grid point (sigma = 0, `_fit_block`'s mark) keeps its base weights."""
    fitted = sg > 0
    u = (h[:, None] - mu[None, :]) / (np.where(fitted, sg, 1.0)[None, :] * np.sqrt(2.0))
    r = Y - (a[None, :] + b[None, :] * (0.5 * (1.0 + erf(u))))        # Phi as _fit_block computes it
    ar = np.where(fitted[None, :], np.abs(r), 0.0)
    out = ar > delta                                                     # never with delta = inf
    return base * np.where(out, delta / np.where(out, ar, 1.0), 1.0)    # factor 1 where |r| <= delta, r = 0 too


def huber_block(Y, base, h, grid, delta=DELTA, max_iter=MAX_ITER):
    """The Huber fit of one block of pixels by IRLS on `_fit_block` (Y float64 with no NaN, base = C s).

    Returns ``(a, b, mu, sigma, n_iter)`` per pixel: the last fit, as `_fit_block` gives it (a = b = mu = sigma = 0
    where no grid point could be fitted), and n_iter the number of refits that changed it: 0 when the first refit
    repeats the start (no residual beyond delta, or delta = inf), max_iter when the cap stopped the pixel before
    its fit repeated. A pixel left without a fitted grid point by a refit stops there."""
    a, b, mu, sg, _, _ = _fit_block(Y, base, h, grid, marea.SG_GRID)
    n_iter = np.zeros(Y.shape[1], int)
    live = np.arange(Y.shape[1])               # the first refit takes the whole block: the start's columns
    for k in range(1, max_iter + 1):
        if not live.size:
            break
        Yl = Y if live.size == Y.shape[1] else np.ascontiguousarray(Y[:, live])
        bl = base if live.size == Y.shape[1] else np.ascontiguousarray(base[:, live])
        w = huber_weights(Yl, bl, h, a[live], b[live], mu[live], sg[live], delta)
        a_k, b_k, mu_k, sg_k, _, _ = _fit_block(Yl, w, h, grid, marea.SG_GRID)
        same = (mu_k == mu[live]) & (sg_k == sg[live]) & (a_k == a[live]) & (b_k == b[live])
        a[live], b[live], mu[live], sg[live] = a_k, b_k, mu_k, sg_k
        n_iter[live[~same]] = k
        live = live[~same & (sg_k > 0)]
    return a, b, mu, sg, n_iter


def invert_huber(Y, C, h, lo=None, hi=None, scl7=None, delta=DELTA, scl7_weight=SCL7_WEIGHT, max_iter=MAX_ITER,
                 mu_points=50, chunk=4000, min_obs=8, min_b=0.15, max_b=2.5, max_a=2.5):
    """V4 and V4-0: per pixel, the Huber fit of NDWI = a + b Phi((h - mu)/sigma) on `invert_series`' grid.

    Y (T, P) NDWI with V0's convention (an undefined NDWI on a clear look is 0; NaN is read as 0), C (T, P) the
    clear looks, h (T,) the levels, lo/hi the grid bounds (h's own range when None), scl7 (T, P) True on the looks
    of SCL class 7 (None: no such look), weighted scl7_weight. Blocks of `chunk` pixels, as `invert_series`.

    Returns ``(z, z0, sigma, a, b, n_iter)``: z the midpoint mu and z0 the NDWI = 0 crossing mu + sigma
    Phi^-1(-a/b), NaN where `invert_series`' guards reject (fewer than min_obs clear looks with a weight, b outside
    (min_b, max_b), |a| >= max_a) and z0 also unless a < 0 < a + b; sigma, a and b of the final fit before the
    guards (as the builder's marea_fit keeps V0's; 0 where no grid point could be fitted); n_iter the refits that
    changed the fit (see `huber_block`; max_iter: stopped by the cap)."""
    h = np.asarray(h, np.float64)
    lo = float(np.min(h)) if lo is None else lo
    hi = float(np.max(h)) if hi is None else hi
    grid = np.linspace(lo, hi, mu_points)
    P = Y.shape[1]
    z, z0 = np.full(P, np.nan), np.full(P, np.nan)
    sigma, a, b = np.full(P, np.nan), np.full(P, np.nan), np.full(P, np.nan)
    n_iter = np.zeros(P, int)
    for j in range(0, P, chunk):
        s = slice(j, min(j + chunk, P))
        Yb = np.asarray(Y[:, s], np.float64)
        Yb = np.where(np.isnan(Yb), 0.0, Yb)                     # V0's inputs: an undefined NDWI enters as 0
        clear = np.asarray(C[:, s], bool)
        s7 = 1.0 if scl7 is None else np.where(np.asarray(scl7[:, s], bool), float(scl7_weight), 1.0)
        base = np.where(clear, s7, 0.0)                          # C s: the starting weights
        ab, bb, mu, sg, n_iter[s] = huber_block(Yb, base, h, grid, delta, max_iter)
        N = (base > 0).sum(axis=0)                               # the clear looks that enter the fit
        ok = (N >= min_obs) & (bb > min_b) & (bb < max_b) & (np.abs(ab) < max_a)     # invert_series' guards
        cross = ok & (ab < 0) & (ab + bb > 0)
        z[s] = np.where(ok, mu, np.nan)
        z0[s] = np.where(cross, mu + sg * ndtri(np.where(cross, -ab / np.where(cross, bb, 1.0), 0.5)), np.nan)
        sigma[s], a[s], b[s] = sg, ab, bb
    return z, z0, sigma, a, b, n_iter
