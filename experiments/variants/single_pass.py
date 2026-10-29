# -*- coding: utf-8 -*-
"""V15 MAREA-S, one model and one search (experiments/variants_addendum_2026-10-28.md, amendment of 2026-10-29).

MAREA fits its delay by the profiled Bernoulli likelihood of the wet/dry record (`te.m2a_rasch`, whose `_band_nll`
maximises every pixel's likelihood over (z, sigma) and keeps only the maximum), then discards those z and fits the
elevation again by least squares on the NDWI. V15 keeps the delay fit's own z: at the window's band lags, the
argmax of the very likelihood `_band_nll` maximises,

    l(z, sigma) = sum_wet log p + sum_dry log(1 - p),   p = clip(Phi((h - z)/sigma), 1e-4, 1 - 1e-4),

with wet = clear and NDWI > 0 (an undefined NDWI is dry), z on the delay fit's grid and sigma on its atoms, for every
pixel. Ties (the plateau of complete separation): the median of the z levels whose profiled log-likelihood is within
tie_tol of the maximum (MAREA-B's rule); sigma is the best atom at the tied level nearest z. Guards: at least
min_looks clear looks (the delay fit's 6) with at least one wet and one dry.

`invert_single` also returns each pixel's maximum log-likelihood, the number `_band_nll` adds up, so that the
consistency with the delay fit can be checked (tests/test_variants_single_pass.py).
"""
import numpy as np
from scipy.special import erf

CLIP = 1e-4             # the delay fit's clipping of p (tide_estimators._band_nll)
MIN_LOOKS = 6           # the delay fit's least number of clear looks of a scored pixel
TIE_TOL = 1e-9          # MAREA-B's tolerance for tied z levels


def _phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))          # tide_estimators._phi


def invert_single(Y, C, h, z_grid, sg_grid, clip=CLIP, min_looks=MIN_LOOKS, tie_tol=TIE_TOL, chunk=4000):
    """V15 per pixel: Y (T, P) NDWI (NaN where undefined), C (T, P) clear looks, h (T,) the levels at the band's lag,
    z_grid and sg_grid the delay fit's grid and atoms. Returns ``(z, sigma, max_ll)``: z the median of the tied z
    levels, sigma the best atom at the tied level nearest z (NaN where the guards reject), and max_ll the maximum of
    the profiled log-likelihood (finite for every pixel, as `_band_nll` computes it)."""
    h = np.asarray(h, np.float64)
    z_grid, sg = np.asarray(z_grid, np.float64), np.asarray(sg_grid, np.float64)
    T, P = Y.shape
    # (looks, levels) log p and log(1 - p) for every atom, as _band_nll: p clipped, then log and log1p
    logp, log1mp = [], []
    for s0 in sg:
        p = np.clip(_phi((h[:, None] - z_grid[None, :]) / s0), clip, 1 - clip)
        logp.append(np.log(p))
        log1mp.append(np.log1p(-p))
    z, s_out, mx = np.full(P, np.nan), np.full(P, np.nan), np.full(P, np.nan)
    for j in range(0, P, chunk):
        s = slice(j, min(j + chunk, P))
        clear = np.asarray(C[:, s], bool)
        with np.errstate(invalid="ignore"):
            wet = clear & (np.asarray(Y[:, s]) > 0)
        W, D = wet.astype(np.float32), (clear & ~wet).astype(np.float32)      # _band_nll's float32 indicators
        ll = np.stack([W.T @ lp + D.T @ l1 for lp, l1 in zip(logp, log1mp)], axis=2)   # (pixels, levels, atoms)
        prof = ll.max(axis=2)
        best = prof.max(axis=1)
        tied = prof >= best[:, None] - tie_tol
        zb = np.nanmedian(np.where(tied, z_grid[None, :], np.nan), axis=1)
        near = np.argmin(np.where(tied, np.abs(z_grid[None, :] - zb[:, None]), np.inf), axis=1)
        sb = sg[ll[np.arange(len(near)), near].argmax(axis=1)]
        ok = (clear.sum(axis=0) >= min_looks) & wet.any(axis=0) & (clear & ~wet).any(axis=0)
        z[s], s_out[s], mx[s] = np.where(ok, zb, np.nan), np.where(ok, sb, np.nan), best
    return z, s_out, mx
