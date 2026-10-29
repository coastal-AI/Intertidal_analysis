# -*- coding: utf-8 -*-
"""V12 MAREA-P (depth-aware): MAREA's least-squares fit of the NDWI magnitude with the rise of NDWI with water depth
as a term of its own, so that the elevation is the level where the pixel is half covered and not the midpoint of the
NDWI magnitude (experiments/variants_addendum_2026-10-28.md, row V12, and its amendment of 2026-10-28).

What it computes. Per pixel, over its clear looks, the weighted least squares of

    NDWI = a + Phi(u) (b + c (h - z)),    u = (h - z)/sigma,    c >= 0,

with V0's inputs, as run_marea passes them to marea.invert_series: Y the NDWI with an undefined value already set to
0 by the caller (a clear look with an undefined NDWI enters as 0), C the clear looks as weights. Phi(u) is the share
of the pixel under water (MAREA's sigmoid), a the NDWI of the dry pixel, b the step of the covered part at the
waterline and c its rise per metre of depth h - z. (a, b, c) have a closed form at each (z, sigma) of invert_series'
grid (50 levels between the window's lowest and highest level, sigma in marea.SG_GRID); the grid point of least loss
is kept, ties as _fit_block (the first); the elevation is its z, the level where Phi = 0.5, and the guards are
invert_series' (N >= 8, 0.15 < b < 2.5, |a| < 2.5). MAREA (V0) fits the same model with c = 0, and its sigmoid
takes the rise of NDWI with depth as part of the transition, which moves its midpoint up; here that rise is c.

c >= 0. The loss is a convex quadratic in (a, b, c), so when its unconstrained minimum at a grid point has c < 0, the
minimum over c >= 0 lies on c = 0: that grid point takes the c = 0 fit, MAREA's, computed with _fit_block's own
lines. With depth=False every grid point takes it, and fit_depth_block and invert_depth return _fit_block's and
invert_series' results bit for bit (with c = 0 V12 is V0; the tests check it).

How. At a grid point the regressors are 1, phi = Phi((h - z)/sigma) and phi (h - z). The third is taken as
q = phi (h - r), r the middle of the z grid, the same at every grid point: phi (h - z) = q + (r - z) phi, so both
bases give the same fits and the same c, and b = b_q + c (z - r), b_q being phi's coefficient next to q. Two grid
points with the same phi on every look (erf saturates there, between two of the pixel's levels far apart) then have
the same sums and the same loss bit for bit, and the first is kept, as in _fit_block; with phi (h - z) their losses
would differ by rounding. The 3 x 3 normal equations are solved with c last: (a0, b0) and loss0 are _fit_block's
c = 0 fit; q's regression on (1, phi), with the same 2 x 2 inverse, leaves D, the weighted sum of squares of q's
residual (the Schur complement, det3/det2); rho is the c = 0 residual's weighted product with q. Then c = rho/D,
(a, b_q) = (a0, b0) - c (q's coefficients on 1 and phi), and the loss falls by c rho = rho^2/D. So c > 0 exactly
when rho > 0; the grid point keeps the c = 0 fit when rho <= 0, or when q adds nothing to (1, phi) (D at most Q_TOL
of q's own sum of squares, e.g. every look with phi > 0 at one level: the c = 0 fit is then a least-squares solution
of the three-term model). The sums with q come from one matrix product per block, over the distinct rows of phi;
_fit_block's sums keep its matrix-vector products, so that the c = 0 arithmetic is its own.

The notebook calls it as run_marea calls invert_series: once per delay group, with
Y = np.nan_to_num(Y_S[np.ix_(w, px)], nan=0.0), C = C_S[np.ix_(w, px)], h = level_at(tau)[w] and lo, hi the window's
lowest and highest level without delay (h0.min(), h0.max()). tests/test_variants_depth_model.py checks it against
bounded least squares by plain loops.
"""
import numpy as np
from scipy.special import erf

from pyintertidal.marea import SG_GRID

#: q = phi (h - r) is taken as adding nothing to (1, phi) when its residual sum of squares on them is at most this
#: share of its own (the counterpart of _fit_block's det > 1e-9): the grid point keeps the c = 0 fit
Q_TOL = 1e-9


def fit_depth_block(Y, C, tide, mu_grid, sg_grid, depth=True):
    """V12's fit for a block of pixels: _fit_block with the depth term.

    ``Y`` (T, P) NDWI with an undefined value already 0, ``C`` (T, P) the clear looks as weights (bool or float, as
    _fit_block takes them), ``tide`` (T,) the levels, ``mu_grid`` and ``sg_grid`` the grid, swept z first and sigma
    inside, as _fit_block. Per pixel and grid point, the weighted least squares of
    Y = a + phi (b + c (tide - mu)), phi = Phi((tide - mu)/sigma), with c >= 0; the grid point of least loss is
    kept, the first on ties. ``depth=False`` fixes c = 0 at every grid point: _fit_block's result bit for bit.

    Returns ``(a, b, c, mu, sigma, rmse, n_obs)``, each of length P. As in _fit_block, a pixel with no grid point
    whose phi varies over its clear looks (e.g. no clear look) keeps zeros, and its rmse is inf.
    """
    T, P = Y.shape
    # _fit_block's sums, line for line
    Cf = C.astype(np.float64)
    Yc = np.where(Cf > 0, Y, 0.0).astype(np.float64) * Cf

    N = Cf.sum(axis=0)
    Sy = Yc.sum(axis=0)
    Syy = (np.where(Cf > 0, Y, 0.0) * Yc).sum(axis=0)

    if depth:
        # phi at every grid point, in the sweep's order and with the loop's own expression; grid points with the
        # same phi share one row of the q sums, so that their losses are equal bit for bit
        Phi = np.array([0.5 * (1.0 + erf((tide - mu) / (sg * np.sqrt(2.0))))
                        for mu in mu_grid for sg in sg_grid]).reshape(len(mu_grid) * len(sg_grid), T)
        Phi_u, row = np.unique(Phi, axis=0, return_inverse=True)
        row = np.asarray(row).reshape(-1)
        r = 0.5 * (float(np.min(mu_grid)) + float(np.max(mu_grid)))
        q = Phi_u * (np.asarray(tide, np.float64) - r)                 # q = phi (h - r): (distinct phi, T)
        Sq, Spq, Sqq = (np.concatenate([q, Phi_u * q, q * q]) @ Cf).reshape(3, len(Phi_u), P)
        Syq = q @ Yc

    best_loss = np.full(P, np.inf)
    best = np.zeros((5, P))

    g = 0
    for mu in mu_grid:
        for sg in sg_grid:
            # MAREA's c = 0 fit at this grid point: _fit_block's lines
            phi = 0.5 * (1.0 + erf((tide - mu) / (sg * np.sqrt(2.0))))    # (T,)
            Sp = Cf.T @ phi
            Spp = Cf.T @ (phi * phi)
            Syp = Yc.T @ phi
            det = N * Spp - Sp * Sp
            ok = det > 1e-9
            b = np.where(ok, (N * Syp - Sp * Sy) / np.where(ok, det, 1.0), 0.0)
            a = np.where(N > 0, (Sy - b * Sp) / np.where(N > 0, N, 1.0), 0.0)
            loss = (Syy - 2 * a * Sy - 2 * b * Syp + a * a * N
                    + 2 * a * b * Sp + b * b * Spp)
            c = np.zeros(P)

            if depth:
                k = row[g]
                # q's regression on (1, phi), with the c = 0 fit's 2 x 2 inverse
                beta = np.where(ok, (N * Spq[k] - Sp * Sq[k]) / np.where(ok, det, 1.0), 0.0)
                alpha = np.where(N > 0, (Sq[k] - beta * Sp) / np.where(N > 0, N, 1.0), 0.0)
                D = Sqq[k] - alpha * Sq[k] - beta * Spq[k]          # what q adds to (1, phi)
                rho = Syq[k] - a * Sq[k] - b * Spq[k]               # the c = 0 residual against q
                free = ok & (D > Q_TOL * Sqq[k]) & (rho > 0)        # the unconstrained c = rho / D is > 0
                c = np.where(free, rho / np.where(free, D, 1.0), 0.0)
                a = np.where(free, a - c * alpha, a)
                b = np.where(free, b - c * beta + c * (mu - r), b)  # b_q -> b, the step at the waterline z = mu
                loss = np.where(free, loss - c * rho, loss)
            g += 1

            upd = ok & (loss < best_loss)
            best_loss = np.where(upd, loss, best_loss)
            best[0] = np.where(upd, a, best[0])
            best[1] = np.where(upd, b, best[1])
            best[2] = np.where(upd, c, best[2])
            best[3] = np.where(upd, mu, best[3])
            best[4] = np.where(upd, sg, best[4])

    rmse = np.sqrt(np.maximum(best_loss, 0.0) / np.maximum(N, 1.0))
    return best[0], best[1], best[2], best[3], best[4], rmse, N


def invert_depth(Y, C, h, lo=None, hi=None, mu_points=50, chunk=4000,
                 min_obs=8, min_b=0.15, max_b=2.5, max_a=2.5, depth=True):
    """V12 MAREA-P: per-pixel elevation from NDWI series under a level series ``h``, invert_series with the depth
    term: fit_depth_block on blocks of ``chunk`` pixels, over invert_series' grid (``mu_points`` levels in [lo, hi],
    by default the lowest and highest of h, and sigma in marea.SG_GRID), with its guards (N >= min_obs,
    min_b < b < max_b, |a| < max_a, b being the step at the waterline).

    ``Y`` and ``C`` as invert_series takes them, V0's inputs: an undefined NDWI already 0, C the clear looks as
    weights. Returns ``(z, sigma, c)``, NaN where the guards reject. ``depth=False`` fixes c = 0 at every grid point:
    invert_series' (z, sigma) bit for bit (its min_bracket = 0, as run_marea calls it).
    """
    lo = float(np.min(h)) if lo is None else lo
    hi = float(np.max(h)) if hi is None else hi
    z = np.full(Y.shape[1], np.nan)
    sg_out = np.full(Y.shape[1], np.nan)
    c_out = np.full(Y.shape[1], np.nan)
    grid = np.linspace(lo, hi, mu_points)
    for j in range(0, Y.shape[1], chunk):
        s = slice(j, min(j + chunk, Y.shape[1]))
        a, b, c, mu, sg, _, N = fit_depth_block(np.asarray(Y[:, s], np.float64),
                                                np.asarray(C[:, s], np.float64),
                                                h, grid, SG_GRID, depth=depth)
        ok = ((N >= min_obs) & (b > min_b) & (b < max_b)
              & (np.abs(a) < max_a))
        z[s] = np.where(ok, mu, np.nan)
        sg_out[s] = np.where(ok, sg, np.nan)
        c_out[s] = np.where(ok, c, np.nan)
    return z, sg_out, c_out
