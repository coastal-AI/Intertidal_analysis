"""
tide_estimators.py — Interior-tide estimators of phase M2 (plan v4)
====================================================================

The four spec'd estimators. All consume the M1 along-mouth coordinate and
the archive alone (R2: zero survey labels); all take the boundary series
through a :class:`ShiftBank` so the lag axis is continuous and the boundary
provider stays interchangeable (phase M3).

A structural fact shapes the family — the AFFINE THEOREM measured earlier in
this project: any estimator built on the ORDER of scenes (ranks of flooded
area, concordance of wet/dry outcomes) is invariant to gain and datum of the
level, so it can measure the LAG tau(s) but is blind to the amplitude
alpha(s). Amplitude is only visible to the likelihood, through how wide the
wet/dry transition is against the boundary clock. Hence:

    ============  =================================  ==========  =========
    estimator     mechanism                          estimates   blind to
    ============  =================================  ==========  =========
    M2a Rasch     Bernoulli likelihood, z profiled   alpha, tau  —
    M2b copulas   per-pixel wet-h concordance        tau         alpha
    M2c waterline flooded-area rank correlation      tau         alpha
    M2d Granadeiro flood/ebb topography discrepancy  tau         alpha
    ============  =================================  ==========  =========

**M2d — the literature baseline (Granadeiro-style).** The published route to
cotidal lags from the archive: for each along-estuary band, find the single
lag that makes the topography inferred from FLOOD scenes agree with the one
inferred from EBB scenes. If the assumed tide leads the local water, flood
fits and ebb fits disagree in opposite directions; the lag that zeroes the
disagreement is the cotidal lag. Assumes symmetry between limbs — which is
exactly what our hysteresis evidence questions, and why M2d is the baseline
to beat rather than the method.

**M2a — the main engine (Rasch/IRT).** Scene = examinee with ability
h(s, t); pixel = item with difficulty z; response = wet/dry; Bernoulli
likelihood. The ability is the boundary tide warped by two smooth profiles
pinned at the mouth (the anchor items of the spec — the datum comes from the
boundary, deviations are measured relative to it):

    h(s, t) = alpha(s) * h_boundary(t - tau(s))

with ``alpha`` and ``tau`` piecewise-linear on a few knots, ``alpha(0)=1``,
``tau(0)=0``. Item difficulties z are PROFILED per pixel by a vectorised 1-D
grid search (they never enter the global parameter vector), and the handful
of global parameters go to L-BFGS-B — the R5-sanctioned optimiser.

Continuous lags use linear interpolation over a precomputed shift grid of the
boundary series, so the objective stays smooth for the optimiser.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .elevation import _fit_block

SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


def _phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


class ShiftBank:
    """h_boundary(t - tau) for continuous tau, by interpolating a shift grid."""

    def __init__(self, shifts_min, series):
        order = np.argsort(shifts_min)
        self.taus = np.asarray(shifts_min, float)[order]
        self.H = np.asarray(series, float)[order]      # (n_shift, T)

    def at(self, tau_min):
        t = float(np.clip(tau_min, self.taus[0], self.taus[-1]))
        j = int(np.searchsorted(self.taus, t) - 1)
        j = max(0, min(j, len(self.taus) - 2))
        w = (t - self.taus[j]) / (self.taus[j + 1] - self.taus[j])
        return (1 - w) * self.H[j] + w * self.H[j + 1]


def make_bands(s_km, n_bands):
    """Quantile bands of the along-mouth coordinate; returns (edges, centers,
    band_of_pixel)."""
    fin = np.isfinite(s_km)
    qs = np.nanquantile(s_km, np.linspace(0, 1, n_bands + 1))
    qs[0] -= 1e-9
    band = np.full(len(s_km), -1, int)
    centers = []
    for k, (a, b) in enumerate(zip(qs, qs[1:])):
        m = fin & (s_km > a) & (s_km <= b)
        band[m] = k
        centers.append(float(np.median(s_km[m])) if m.any() else np.nan)
    return qs, np.asarray(centers), band


# ─────────────────────────────────────────────────────────────────────────────
#  M2d — flood/ebb discrepancy minimisation (the baseline)
# ─────────────────────────────────────────────────────────────────────────────

def m2d_flood_ebb(Y, C, bank, rising, band_of, n_bands, tau_grid,
                  mu_points=50, min_obs=6, min_b=0.15, rng=None,
                  max_px_band=2500):
    """Cotidal lag per band by minimising flood-vs-ebb topography disagreement.

    Returns ``tau_hat`` (n_bands,) minutes, NaN where undetermined.
    """
    rng = rng or np.random.default_rng(0)
    tau_hat = np.full(n_bands, np.nan)
    for k in range(n_bands):
        cols = np.where(band_of == k)[0]
        if len(cols) < 100:
            continue
        if len(cols) > max_px_band:
            cols = np.sort(rng.choice(cols, max_px_band, replace=False))
        Yb, Cb = Y[:, cols], C[:, cols]
        disc = []
        for tau in tau_grid:
            h = bank.at(tau)
            grid = np.linspace(h.min(), h.max(), mu_points)
            zs = {}
            for lab, sel in (("f", rising), ("e", ~rising)):
                a, b, mu, sg, _, N = _fit_block(Yb[sel], Cb[sel], h[sel],
                                                grid, SG_GRID)
                ok = (N >= min_obs) & (b > min_b) & (b < 2.5)
                zs[lab] = np.where(ok, mu, np.nan)
            both = np.isfinite(zs["f"]) & np.isfinite(zs["e"])
            disc.append(np.median(np.abs(zs["f"] - zs["e"])[both])
                        if both.sum() >= 50 else np.nan)
        disc = np.asarray(disc)
        if np.isfinite(disc).sum() < 3:
            continue
        j = int(np.nanargmin(disc))
        if 0 < j < len(tau_grid) - 1 and np.isfinite(disc[j - 1:j + 2]).all():
            d0, d1, d2 = disc[j - 1], disc[j], disc[j + 1]
            den = d0 - 2 * d1 + d2
            off = 0.5 * (d0 - d2) / den if abs(den) > 1e-12 else 0.0
            step = tau_grid[1] - tau_grid[0]
            tau_hat[k] = tau_grid[j] + np.clip(off, -1, 1) * step
        else:
            tau_hat[k] = tau_grid[j]
    return tau_hat


# ─────────────────────────────────────────────────────────────────────────────
#  B5 — hysteresis: one clock per tide limb (flood water arrives, ebb water
#  LEAVES — and ponding makes leaving slower than arriving)
# ─────────────────────────────────────────────────────────────────────────────

def hysteresis_oos(Y, C, bank, rising, band_of, n_bands, tau_up_grid,
                   tau_dn_grid, test_every=3, rng=None, max_px_band=2000,
                   mu_points=50, min_obs=6, min_b=0.15):
    """B5 v2: limb clocks selected by OUT-OF-SAMPLE continuous prediction.

    The binary profiled likelihood cannot judge a two-clock model — split
    the limbs and it rewards separating them until the limb itself predicts
    wetness (measured: clocks ran to opposite bounds even with no hysteresis
    planted). The judge that works is the b2 prototype's: fit the per-pixel
    sigmoids on TRAIN scenes under each candidate clock pair, score the
    prediction of held-out CONTINUOUS NDWI, adopt the pair only if it beats
    the best single clock. Pathological separations predict held-out scenes
    WORSE, so the degeneracy dies at the judge.

    Returns per-band ``tau_up``, ``tau_dn``, ``adopted``, and the OOS RMSE
    of the winner and of the best single clock.
    """
    rng = rng or np.random.default_rng(4)
    idx = np.arange(Y.shape[0])
    te_m = (idx % test_every) == 0
    tr_m = ~te_m
    tau_up = np.zeros(n_bands)
    tau_dn = np.zeros(n_bands)
    tau_single = np.zeros(n_bands)
    adopted = np.zeros(n_bands, bool)
    r_two = np.full(n_bands, np.nan)
    r_one = np.full(n_bands, np.nan)

    singles = sorted(set(tau_up_grid) | set(tau_dn_grid))

    for k in range(n_bands):
        cols = np.where(band_of == k)[0]
        if len(cols) < 100:
            continue
        if len(cols) > max_px_band:
            cols = np.sort(rng.choice(cols, max_px_band, replace=False))
        Yk = Y[:, cols].astype(np.float64)
        Ck = C[:, cols].astype(np.float64)

        def oos(tu, td):
            h = np.where(rising, bank.at(tu), bank.at(td))
            grid = np.linspace(h.min(), h.max(), mu_points)
            a, b, mu, sg, _, N = _fit_block(Yk[tr_m], Ck[tr_m], h[tr_m],
                                            grid, SG_GRID)
            ok = (N >= min_obs) & (b > min_b) & (b < 2.5) & (np.abs(a) < 2.5)
            pred = np.clip(a[None, :] + b[None, :] * _phi(
                (h[te_m][:, None] - mu[None, :])
                / np.maximum(sg[None, :], 1e-3)), -2.0, 2.0)
            w = (Ck[te_m] > 0) & ok[None, :]
            if w.sum() < 200:
                return np.nan
            return float(np.sqrt(np.mean(
                (pred[w] - Yk[te_m][w]) ** 2)))

        best1, tau1 = np.inf, 0.0
        for tv in singles:
            r = oos(tv, tv)
            if np.isfinite(r) and r < best1:
                best1, tau1 = r, float(tv)
        best2, tu2, td2 = best1, tau1, tau1
        for tu in tau_up_grid:
            for td in tau_dn_grid:
                if tu == td:
                    continue
                r = oos(tu, td)
                if np.isfinite(r) and r < best2:
                    best2, tu2, td2 = r, float(tu), float(td)
        adopted[k] = best2 < best1
        tau_up[k], tau_dn[k] = (tu2, td2) if adopted[k] else (tau1, tau1)
        tau_single[k] = tau1
        r_two[k], r_one[k] = best2, best1
    return {"tau_up": tau_up, "tau_dn": tau_dn, "tau_single": tau_single,
            "adopted": adopted, "oos_two": r_two, "oos_one": r_one}


def m2a_hysteresis(wet, clear, bank, rising, band_of, n_bands,
                   tau_grid=None, sigma0=0.20, sg_grid=None, z_points=60,
                   rng=None, max_px_band=1200, n_rounds=2):
    """DEPRECATED for estimation — kept as the documented failure (B5 v1).

    Splitting scenes by limb opens a degenerate direction the profiled
    Bernoulli likelihood loves: separate the limbs' levels until the limb
    itself predicts wetness (measured: clocks ran to opposite bounds with
    AND without planted hysteresis). Use :func:`hysteresis_oos` — the
    out-of-sample continuous judge — for anything beyond one clock.
    """
    rng = rng or np.random.default_rng(3)
    tau_grid = np.asarray([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0,
                           45.0, 60.0] if tau_grid is None else tau_grid)
    h0 = bank.at(0.0)
    z_grid = np.linspace(h0.min() - 0.3, h0.max() + 0.3, z_points)
    tau_up = np.zeros(n_bands)
    tau_dn = np.zeros(n_bands)
    nll_tot = 0.0
    for k in range(n_bands):
        cols = np.where(band_of == k)[0]
        if len(cols) < 100:
            tau_up[k] = tau_dn[k] = np.nan
            continue
        if len(cols) > max_px_band:
            cols = np.sort(rng.choice(cols, max_px_band, replace=False))
        Wk, Ck = wet[:, cols], clear[:, cols]

        def nll_of(tu, td):
            h = np.where(rising, bank.at(tu), bank.at(td))
            v, _ = _band_nll(Wk, Ck, h, z_grid, sigma0, sg_grid=sg_grid)
            return v

        tu = td = 0.0
        if k > 0:                      # mouth band stays the anchor (0, 0)
            for _ in range(n_rounds):
                sc = np.asarray([nll_of(tu, tv) for tv in tau_grid])
                td = _parabolic_argmax(-sc, tau_grid)   # min NLL en tau_dn
                sc = np.asarray([nll_of(tv, td) for tv in tau_grid])
                tu = _parabolic_argmax(-sc, tau_grid)   # min NLL en tau_up
        tau_up[k], tau_dn[k] = tu, td
        nll_tot += nll_of(tu, td)
    return {"tau_up": tau_up, "tau_dn": tau_dn,
            "nll": float(nll_tot / max(n_bands, 1))}


# ─────────────────────────────────────────────────────────────────────────────
#  M2b — copula/concordance: which clock explains each pixel's wet sequence
# ─────────────────────────────────────────────────────────────────────────────

def m2b_concordance(wet, clear, bank, band_of, n_bands, tau_grid,
                    min_obs=10, rng=None, max_px_band=1500):
    """Lag per band by per-pixel point-biserial concordance.

    For each candidate lag, correlate every pixel's wet/dry sequence with the
    shifted boundary series over its clear scenes; the band's score is the
    MEDIAN correlation over its pixels (robust to the always-wet/always-dry
    tails, which carry no clock information). The lag that maximises the
    score is the band's clock. Rank-like, hence affine-blind: tau only.
    """
    rng = rng or np.random.default_rng(2)
    tau_hat = np.full(n_bands, np.nan)
    for k in range(n_bands):
        cols = np.where(band_of == k)[0]
        if len(cols) < 50:
            continue
        if len(cols) > max_px_band:
            cols = np.sort(rng.choice(cols, max_px_band, replace=False))
        W = wet[:, cols].astype(np.float64)
        Cl = clear[:, cols].astype(np.float64)
        n = Cl.sum(axis=0)
        ok_n = n >= min_obs
        score = []
        for tau in tau_grid:
            h = bank.at(tau)
            mh = (Cl * h[:, None]).sum(0) / np.maximum(n, 1)
            vh = (Cl * h[:, None] ** 2).sum(0) / np.maximum(n, 1) - mh ** 2
            mw = (W * Cl).sum(0) / np.maximum(n, 1)
            cov = (W * Cl * h[:, None]).sum(0) / np.maximum(n, 1) - mw * mh
            den = np.sqrt(np.maximum(mw * (1 - mw) * vh, 1e-12))
            r = np.where(ok_n & (mw > 0.02) & (mw < 0.98), cov / den, np.nan)
            score.append(float(np.nanmedian(r)))
        tau_hat[k] = _parabolic_argmax(np.asarray(score), tau_grid)
    return tau_hat


# ─────────────────────────────────────────────────────────────────────────────
#  M2c — differential waterline kinematics (flooded-area ranks)
# ─────────────────────────────────────────────────────────────────────────────

def m2c_waterline(wet, clear, bank, band_of, n_bands, tau_grid,
                  min_cover=0.3, min_scenes=60):
    """Lag per band by rank correlation of flooded area against the shifted
    boundary — the binary-gauging idea already validated externally on the
    Scheldt (imagery gradient 0.9 min/km = the gauges'). Affine-blind by the
    same theorem: ranks of area survive any monotone gain/datum, so tau only.
    """
    from scipy.stats import spearmanr

    tau_hat = np.full(n_bands, np.nan)
    for k in range(n_bands):
        cols = np.where(band_of == k)[0]
        if len(cols) < 50:
            continue
        Cl = clear[:, cols]
        cover = Cl.mean(axis=1)
        use = cover >= min_cover
        if use.sum() < min_scenes:
            continue
        A = (wet[:, cols] & Cl).sum(axis=1)[use] / Cl.sum(axis=1)[use]
        score = np.asarray([spearmanr(A, bank.at(tau)[use]).statistic
                            for tau in tau_grid])
        tau_hat[k] = _parabolic_argmax(score, tau_grid)
    return tau_hat


def _parabolic_argmax(score, tau_grid):
    """Sub-grid argmax by parabola through the top three points."""
    if np.isfinite(score).sum() < 3:
        return np.nan
    j = int(np.nanargmax(score))
    if 0 < j < len(tau_grid) - 1 and np.isfinite(score[j - 1:j + 2]).all():
        d0, d1, d2 = score[j - 1], score[j], score[j + 1]
        den = d0 - 2 * d1 + d2
        off = 0.5 * (d0 - d2) / den if abs(den) > 1e-12 else 0.0
        step = tau_grid[1] - tau_grid[0]
        return float(tau_grid[j] - np.clip(off, -1, 1) * step)
    return float(tau_grid[j])


# ─────────────────────────────────────────────────────────────────────────────
#  M2a — Rasch/IRT with profiled difficulties
# ─────────────────────────────────────────────────────────────────────────────

def _band_nll(wet, clear, h, z_grid, sigma0, sg_grid=None):
    """Mean Bernoulli NLL of a band, difficulties profiled on ``z_grid``.

    ``wet``/``clear``: (T, P) boolean. Vectorised as two matmuls; the z that
    maximises each pixel's likelihood is chosen and never leaves this
    function — that is the profiling.

    ``sg_grid``: optionally profile the transition width per pixel too, over
    the archive's own sigma atoms. Added after the v1 gate: a single fixed
    sigma0 lets scale misfit masquerade as gain/lag (measured: alpha pinned
    at its bound), whereas profiling sigma spends the scale degeneracy where
    it belongs — inside the pixel, not inside the transfer.
    """
    W = (wet & clear).astype(np.float32)
    D = (~wet & clear).astype(np.float32)
    best = None
    for s0 in ([sigma0] if sg_grid is None else sg_grid):
        p = np.clip(_phi((h[:, None] - z_grid[None, :]) / s0),
                    1e-4, 1 - 1e-4)                   # (T, Z)
        LL = W.T @ np.log(p) + D.T @ np.log1p(-p)     # (P, Z)
        m = LL.max(axis=1)
        best = m if best is None else np.maximum(best, m)
    n = clear.sum(axis=0)
    ok = n >= 6
    return -float(best[ok].sum() / max(n[ok].sum(), 1)), int(ok.sum())


def alpha_nll_profile(wet, clear, bank, tau, alpha_grid, z_grid, sg_grid):
    """NLL(alpha) with (z, sigma) profiled per pixel — the degeneracy meter.

    The affine theorem says binary data cannot see gain: rescaling
    (alpha, z, sigma) TOGETHER leaves every likelihood term unchanged. The
    profiling grids therefore scale with alpha — otherwise their fixed
    ceilings penalise large alpha and manufacture identifiability out of
    quantisation (measured on the first v2 run: a spurious 0.18 NLL slope,
    entirely from sigma being capped at its top atom). With scaled grids a
    flat curve verifies the implementation adds no false identification;
    curvature would mean a coding error, not physics.
    """
    out = []
    for al in alpha_grid:
        al = float(al)
        h = al * bank.at(tau)
        v, _ = _band_nll(wet, clear, h, al * np.asarray(z_grid, float),
                         None, sg_grid=[al * s for s in sg_grid])
        out.append(v)
    return np.asarray(out)


def m2a_rasch(wet, clear, bank, band_of, centers_km, n_bands, sigma0=0.20,
              alpha_bounds=(0.7, 1.4), tau_bounds=(-30.0, 120.0),
              z_points=60, rng=None, max_px_band=1200, verbose=False,
              bank_by_band=None, fit_alpha=False, sg_grid=None):
    """Fit alpha(s), tau(s) on band centres by profiled Bernoulli likelihood.

    Mouth band pinned to (alpha=1, tau=0): the anchor. Free parameters are
    the profiles at the remaining band centres — 2*(n_bands-1) numbers for
    L-BFGS-B, well inside the spec's 20-40 budget.

    ``bank_by_band``: optional list of per-band ShiftBanks. Used by the M4
    contraction gate: feed each band the OPERATOR's own level series and the
    refit must come back (alpha=1, tau=0) — the fixed point.

    ``fit_alpha``: **False by default since the v1 gate.** The affine theorem
    makes per-band gain unidentifiable from binary data — the v1 gate showed
    the optimiser spending alpha on sigma misfit (pinned at its bound). The
    default is the honest phase-only model; ``alpha_nll_profile`` exists to
    demonstrate the flatness rather than hide it. ``fit_alpha=True`` is kept
    for degeneracy studies only.

    ``sg_grid``: per-pixel sigma atoms to profile over (recommended: the
    archive's own grid) — without it, scale misfit leaks into tau as well.

    Returns dict with ``alpha``, ``tau`` (per band, mouth included) and the
    achieved mean NLL.
    """
    from scipy.optimize import minimize

    rng = rng or np.random.default_rng(1)
    cols_by_band = []
    for k in range(n_bands):
        cols = np.where(band_of == k)[0]
        if len(cols) > max_px_band:
            cols = np.sort(rng.choice(cols, max_px_band, replace=False))
        cols_by_band.append(cols)

    h0 = bank.at(0.0)
    z_grid = np.linspace(h0.min() - 0.3, h0.max() + 0.3, z_points)
    nf = n_bands - 1

    def unpack(theta):
        if fit_alpha:
            alpha = np.concatenate([[1.0], theta[:nf]])
            tau = np.concatenate([[0.0], theta[nf:] * 60.0])
        else:
            alpha = np.ones(n_bands)
            tau = np.concatenate([[0.0], theta * 60.0])
        return alpha, tau

    def objective(theta):
        alpha, tau = unpack(theta)
        total, nn = 0.0, 0
        for k in range(n_bands):
            cols = cols_by_band[k]
            if len(cols) < 50:
                continue
            bk = bank if bank_by_band is None else bank_by_band[k]
            h = alpha[k] * bk.at(tau[k])
            nll, npx = _band_nll(wet[:, cols], clear[:, cols], h,
                                 z_grid, sigma0, sg_grid=sg_grid)
            total += nll * npx
            nn += npx
        return total / max(nn, 1)

    # tau is optimised in HOURS so every free parameter is O(1) and one
    # finite-difference step suits them all
    tb = (tau_bounds[0] / 60.0, tau_bounds[1] / 60.0)
    if fit_alpha:
        x0 = np.concatenate([np.ones(nf), np.zeros(nf)])
        bounds = [alpha_bounds] * nf + [tb] * nf
    else:
        x0 = np.zeros(nf)
        bounds = [tb] * nf
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 60, "eps": 1e-3})
    alpha, tau = unpack(res.x)
    if verbose:
        print(f"    m2a: NLL {res.fun:.5f} · {res.nit} iter · "
              f"alpha {np.round(alpha, 3)} · tau {np.round(tau, 1)}")
    return {"alpha": alpha, "tau": tau, "nll": float(res.fun),
            "centers_km": centers_km, "converged": bool(res.success)}
