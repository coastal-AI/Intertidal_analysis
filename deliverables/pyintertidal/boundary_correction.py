"""
boundary_correction.py — Fase M3: el contorno, corregible e intercambiable
==========================================================================

The transfer operator of phase M4 is only as good as the level it is anchored
to. Phase M3 makes that anchor honest in three moves, none of which touches a
survey label (R2) or a tide gauge (the operator must never need one):

1. **Decompose** the boundary series into harmonics at the astronomical
   frequencies plus a residual. Only ALIASED-RESOLVABLE constituents enter
   the design matrix (alias_table decides; S2's column at a sun-synchronous
   hour is collinear with the mean and would be pure noise).

2. **Correct** per constituent: gain ``(1+gamma_k)`` and lag ``dt_k``
   (minutes) on the estimable constituents only. The overall datum and the
   frozen constituents stay pinned to the prior — that is the affine
   theorem acting as a specification, not a nuisance: imagery cannot see a
   datum, so no correction of it is ever attempted.

3. **Judge from the mouth**: the mouth band is where the ocean model should
   already be right (the anchor of M2a — alpha=1, tau=0 by construction), so
   whatever gain/lag correction the mouth pixels' wet/dry record demands is
   attributable to the BOUNDARY, not to estuarine transfer. Fitted by
   profiled Bernoulli likelihood, adopted only if it also wins on held-out
   scenes; otherwise the correction is zero and says so.

The class :class:`ClimatologyBoundary` closes the provider family: the
harmonic reconstruction alone — a tide "climatology" usable offline anywhere
once fitted, with the explicit caveat that it carries no weather.
"""

from __future__ import annotations

import numpy as np

from .boundary import (PERIODS_H, BoundaryProvider, fit_harmonics,
                       hours_since_epoch)
from .tide_estimators import _band_nll


def decompose(times, levels, constituents):
    """Harmonic part + residual of a boundary series.

    Returns ``(harm, offset, resid)`` with ``harm`` from
    :func:`boundary.fit_harmonics` restricted to ``constituents``,
    ``offset`` the mean of what the harmonics do not explain, and ``resid``
    the zero-mean leftover (weather, model error, everything non-tidal).
    """
    harm = fit_harmonics(times, levels, constituents=list(constituents))
    if harm is None:
        raise ValueError("serie demasiado corta para descomponer")
    t_h = hours_since_epoch(times)
    base = np.zeros_like(t_h, dtype=float)
    for k, (amp, ph, _) in harm.items():
        w = 2 * np.pi / PERIODS_H[k]
        base += amp * np.cos(w * t_h + ph)
    resid = np.asarray(levels, float) - base
    offset = float(np.nanmean(resid))
    return harm, offset, resid - offset


def corrected_series(times, levels, gamma, dt_min, constituents):
    """Apply per-constituent gain/lag corrections to a boundary series.

    ``gamma``/``dt_min``: dicts constituent -> value (missing = no change).
    A lag of ``dt`` minutes evaluates the constituent at ``t - dt``, i.e.
    phase shift ``-w * dt/60`` — kept in time units so signs stay physical
    (positive = the true water is LATE relative to the prior).

    Everything not corrected — frozen constituents, datum, weather residual —
    passes through untouched.
    """
    harm, offset, resid = decompose(times, levels, constituents)
    t_h = hours_since_epoch(times)
    out = offset + resid.copy()
    for k, (amp, ph, _) in harm.items():
        w = 2 * np.pi / PERIODS_H[k]
        g = 1.0 + float(gamma.get(k, 0.0))
        dphi = -w * float(dt_min.get(k, 0.0)) / 60.0
        out += g * amp * np.cos(w * t_h + ph + dphi)
    return out


class ClimatologyBoundary(BoundaryProvider):
    """The harmonic reconstruction alone: a tide climatology, offline.

    Built from any fitted series; serves levels at arbitrary instants with
    no network and no model files. Carries NO weather — its honest error is
    the residual sigma of the fit it came from, stored as ``resid_sigma``.
    """

    name = "climatology"

    def __init__(self, harm, offset, resid_sigma=None):
        self.harm = dict(harm)
        self.offset = float(offset)
        self.resid_sigma = resid_sigma

    @classmethod
    def from_series(cls, times, levels, constituents):
        harm, offset, resid = decompose(times, levels, constituents)
        return cls(harm, offset, float(np.nanstd(resid)))

    def levels(self, times):
        t_h = hours_since_epoch(times)
        out = np.full_like(t_h, self.offset, dtype=float)
        for k, (amp, ph, _) in self.harm.items():
            w = 2 * np.pi / PERIODS_H[k]
            out += amp * np.cos(w * t_h + ph)
        return out


def fit_correction_from_mouth(wet_mouth, clear_mouth, times, levels,
                              constituent="M2", gamma_grid=None,
                              dt_grid_min=None, sigma0=0.20, z_points=60,
                              test_every=3, sg_grid=None):
    """Estimate the boundary's PHASE error from mouth-band wet/dry alone.

    Grid-searches the lag ``dt`` of one constituent, scoring the profiled
    Bernoulli likelihood of the mouth band under the corrected series. The
    winner is ADOPTED only if it also beats zero on held-out scenes (every
    ``test_every``-th scene, an interleaved split so both halves see all
    seasons); otherwise returns zeros with ``adopted=False`` — a correction
    that cannot prove itself out of sample does not enter the pipeline.

    GAIN is deliberately not searched by default (``gamma_grid=[0]``). The
    M3 v1 gate measured why: the dominant constituent's gain is nearly a
    global rescale of the series, which the affine theorem hides from
    binary data — the fitted gamma landed at the grid edge with the wrong
    sign, driven by sigma misfit (the same collapse as alpha in M2 v1).
    The honest gain prior is the ENSEMBLE SPREAD between independent ocean
    models; imagery audits phase, the ensemble audits gain.

    ``sg_grid``: profile sigma per pixel (recommended) — without it scale
    misfit contaminates the phase estimate too.
    """
    gamma_grid = np.asarray([0.0] if gamma_grid is None else gamma_grid,
                            float)
    dt_grid_min = np.asarray(
        [-24, -16, -10, -5, 0, 5, 10, 16, 24]
        if dt_grid_min is None else dt_grid_min, float)

    idx = np.arange(len(times))
    te_mask = (idx % test_every) == 0
    tr_mask = ~te_mask
    h_all = np.asarray(levels, float)
    z_grid = np.linspace(h_all.min() - 0.3, h_all.max() + 0.3, z_points)

    def nll_of(gamma, dt, mask):
        h = corrected_series(times, levels, {constituent: gamma},
                             {constituent: dt}, [constituent] +
                             [k for k in ("N2", "O1") if k != constituent])
        v, _ = _band_nll(wet_mouth[mask], clear_mouth[mask], h[mask],
                         z_grid, sigma0, sg_grid=sg_grid)
        return v

    scores = np.array([[nll_of(g, dtv, tr_mask) for dtv in dt_grid_min]
                       for g in gamma_grid])
    i, j = np.unravel_index(np.argmin(scores), scores.shape)
    g_hat, dt_hat = float(gamma_grid[i]), float(dt_grid_min[j])

    # the out-of-sample judge: does the winner beat "no correction"?
    nll_zero = nll_of(0.0, 0.0, te_mask)
    nll_win = nll_of(g_hat, dt_hat, te_mask)
    adopted = bool(nll_win < nll_zero)
    return {"constituent": constituent, "gamma": g_hat if adopted else 0.0,
            "dt_min": dt_hat if adopted else 0.0, "adopted": adopted,
            "gamma_raw": g_hat, "dt_min_raw": dt_hat,
            "nll_test_zero": float(nll_zero), "nll_test_win": float(nll_win),
            "grid_scores": scores.tolist(),
            "gamma_grid": gamma_grid.tolist(),
            "dt_grid_min": dt_grid_min.tolist()}
