"""
tide_estimators.py — The delay of the tide along the estuary
============================================================

Inside an estuary the tide arrives later than at the mouth. This module
measures that delay, ``tau`` (minutes, positive = later than at the mouth),
section by section, from the wet/dry looks of the record alone: no survey
and no gauge inside the estuary. In the code a section is called a "band"
(``band_km``, ``band_of``, ``n_bands``).

What MAREA uses (``marea.reconstruct``, steps 7-9)
--------------------------------------------------
:class:`ShiftBank`
    The tide at the mouth read ``tau`` minutes earlier, for any ``tau`` in
    the tabulated range (MAREA: ``marea.BANK_TAUS``, −45 to 180 min every
    15). A delay outside that range is clamped SILENTLY to its nearest end.
:func:`make_bands_fixed`
    Sections of fixed length (3 km) along the distance from the mouth, thin
    ones merged. Use its ``band_of`` and ``centers``; its ``edges`` are the
    raw edges from BEFORE the merge.
:func:`m2a_rasch`
    One delay per section, by maximum likelihood (below).
:func:`bootstrap_lags`
    The 95 % interval of each section's delay, by resampling the looks.
:func:`smooth_lag_profile`
    The delay of every pixel: the section delays smoothed along the
    distance, anchored at (0 km, 0 min), never decreasing inland and never
    negative. ``marea.reconstruct`` then rounds it to the minute.

How :func:`m2a_rasch` works. Section k has one delay ``tau_k``; the mouth
section is fixed at 0. A pixel at elevation z is wet in a look with
probability Φ((h(t − tau_k) − z) / σ), h being the tide at the mouth and σ
the width of the pixel's wet/dry transition. For trial delays, each pixel's
z (and σ, when ``sg_grid`` is given, as MAREA does) is picked on a grid and
never enters the optimiser ("profiled"); L-BFGS-B moves only the few delays
to minimise the mean Bernoulli negative log-likelihood. With
``monotone=True`` (MAREA) the free parameters are the per-section
INCREMENTS of the delay, each in [0, ``tau_bounds[1]``], so the delay never
decreases inland.

Why no gain is fitted. Wet/dry data cannot tell a larger tide from a lower
pixel with a wider transition: scaling (gain, z, σ) together leaves every
likelihood term unchanged. The default therefore fits the delay only (``fit_alpha=False``);
:func:`alpha_nll_profile` shows the flat likelihood. For the same reason the
rank-based estimators below can see the delay but not the gain.

Development alternatives (kept for comparison, not used by MAREA)
-----------------------------------------------------------------
The four estimators are labelled M2a (:func:`m2a_rasch`, the one MAREA uses)
to M2d.

:func:`m2d_flood_ebb`
    M2d, the published Granadeiro-style approach: the delay that makes the
    elevations fitted on FLOOD-tide looks agree with those fitted on
    EBB-tide looks. It assumes flood and ebb behave alike.
:func:`m2b_concordance`
    M2b: the delay that best correlates each pixel's wet/dry series with the
    tide.
:func:`m2c_waterline`
    M2c: the delay that best rank-correlates each section's flooded area
    with the tide.
:func:`hysteresis_oos`
    Separate flood and ebb delays, adopted only when they predict held-out
    looks better than one delay.
:func:`m2a_hysteresis`
    Deprecated: the two-delay likelihood that fails, kept as a record.
:func:`make_bands`
    Quantile sections (equal pixel counts); MAREA uses fixed-length ones.
:func:`alpha_nll_profile`
    The likelihood as a function of the gain (a check, see above).

Shapes and units
----------------
T looks (scenes), P pixels. ``wet``, ``clear`` (T, P) bool; ``Y`` (T, P)
NDWI; ``C`` (T, P) clear looks; ``rising`` (T,) bool, the look was taken on
a rising tide; ``band_of`` (P,) int, the section of each pixel, −1 = none;
``s_km`` (P,) km. Levels are metres on the scale of the tide at the mouth;
delays are minutes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .elevation import _fit_block

#: Transition widths σ (m) the per-pixel fits of the development estimators
#: choose from (m2d_flood_ebb, hysteresis_oos). The same values as
#: marea.SG_GRID: keep the two equal.
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


def _phi(x):
    """Standard normal cumulative distribution Φ(x)."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


class ShiftBank:
    """The tide at the mouth read ``tau`` minutes earlier, for any ``tau``.

    Parameters
    ----------
    shifts_min : (S,) array of float
        The tabulated delays, minutes, in any order (MAREA:
        ``marea.BANK_TAUS``, −45 to 180 every 15).
    series : (S, T) array of float
        ``series[j]`` = the mouth level (m) at each of the T look instants
        minus ``shifts_min[j]`` minutes.

    ``at(tau_min)`` returns the (T,) levels for any delay, by linear
    interpolation between the two nearest tabulated delays. A delay outside
    the tabulated range is CLAMPED to its nearest end, silently: in MAREA,
    any delay above 180 min reads the 180-min series.

    Example
    -------
    >>> bank = ShiftBank([0.0, 15.0], [h0, h15])   # h0, h15: (T,) levels
    >>> bank.at(6.0)                               # 0.6 * h0 + 0.4 * h15
    """

    def __init__(self, shifts_min, series):
        order = np.argsort(shifts_min)
        self.taus = np.asarray(shifts_min, float)[order]
        self.H = np.asarray(series, float)[order]      # (n_shift, T)

    def at(self, tau_min):
        """(T,) mouth levels (m) read ``tau_min`` minutes earlier (clamped to the bank's range)."""
        t = float(np.clip(tau_min, self.taus[0], self.taus[-1]))
        j = int(np.searchsorted(self.taus, t) - 1)
        j = max(0, min(j, len(self.taus) - 2))
        w = (t - self.taus[j]) / (self.taus[j + 1] - self.taus[j])
        return (1 - w) * self.H[j] + w * self.H[j + 1]


def make_bands(s_km, n_bands):
    """Quantile bands of the along-mouth coordinate; returns (edges, centers,
    band_of_pixel).

    Development alternative to :func:`make_bands_fixed` (MAREA uses that
    one): ``n_bands`` sections with equal numbers of pixels. ``s_km`` (P,)
    km, NaN allowed; ``edges`` (n_bands + 1,) km; ``centers`` (n_bands,)
    median s of each section; ``band_of`` (P,) int, −1 where s is NaN.
    """
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

    Development alternative (M2d), not used by MAREA. For each candidate
    delay, each pixel's elevation is fitted twice, on the rising-tide looks
    and on the falling-tide looks; the section's delay is the one where the
    median |difference| is smallest (refined between grid points by a
    parabola).

    Parameters
    ----------
    Y, C : (T, P)
        NDWI and clear looks.
    bank : ShiftBank
    rising : (T,) bool
        True for looks taken on a rising tide.
    band_of : (P,) int; n_bands : int
    tau_grid : (k,) float
        Candidate delays, minutes, evenly spaced.
    mu_points, min_obs, min_b, max_px_band, rng
        Elevation grid size, the per-pixel fit's acceptance rules, and the
        seeded pixel subsample per section.

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
#  Hysteresis (development alternative): one delay per tide limb (flood water
#  arrives, ebb water LEAVES — and ponding makes leaving slower than arriving)
# ─────────────────────────────────────────────────────────────────────────────

def hysteresis_oos(Y, C, bank, rising, band_of, n_bands, tau_up_grid,
                   tau_dn_grid, test_every=3, rng=None, max_px_band=2000,
                   mu_points=50, min_obs=6, min_b=0.15):
    """Flood and ebb delays per section, chosen by OUT-OF-SAMPLE prediction.

    Development alternative, not used by MAREA. The wet/dry likelihood
    cannot judge a two-delay model — split the limbs and it rewards
    separating them until the limb itself predicts wetness (measured: the
    delays ran to opposite bounds even on synthetic data with no
    hysteresis). The judge that works: fit the per-pixel sigmoids on
    TRAINING looks under each candidate pair of delays, score how well they
    predict the held-out looks' continuous NDWI, and adopt the pair only if
    it beats the best single delay. Pathological separations predict
    held-out looks WORSE, so the degeneracy dies at the judge.

    Parameters
    ----------
    Y, C : (T, P)
        NDWI and clear looks.
    bank : ShiftBank; rising : (T,) bool; band_of : (P,) int; n_bands : int
    tau_up_grid, tau_dn_grid : sequences of float
        Candidate rising-tide and falling-tide delays, minutes.
    test_every : int
        Every ``test_every``-th look (0, n, 2n, ...) is held out.
    rng, max_px_band, mu_points, min_obs, min_b
        As in :func:`m2d_flood_ebb`.

    Returns per-band ``tau_up``, ``tau_dn``, ``adopted``, and the OOS RMSE
    of the winner and of the best single clock (``oos_two``, ``oos_one``;
    also ``tau_single``).
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
    """DEPRECATED for estimation — kept as the documented failure.

    The first two-delay attempt (one delay for rising-tide looks, one for
    falling-tide looks), by the same profiled Bernoulli likelihood as
    :func:`m2a_rasch`. Splitting the looks by limb opens a degenerate
    direction the likelihood loves: separate the limbs' levels until the
    limb itself predicts wetness (measured: the delays ran to opposite
    bounds with AND without hysteresis planted in synthetic data). Use
    :func:`hysteresis_oos` — the out-of-sample judge — for anything beyond
    one delay. Arguments as in :func:`m2a_rasch` and :func:`hysteresis_oos`;
    returns ``{"tau_up", "tau_dn", "nll"}``.
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
                td = _parabolic_argmax(-sc, tau_grid)   # min NLL over tau_dn
                sc = np.asarray([nll_of(tv, td) for tv in tau_grid])
                tu = _parabolic_argmax(-sc, tau_grid)   # min NLL over tau_up
        tau_up[k], tau_dn[k] = tu, td
        nll_tot += nll_of(tu, td)
    return {"tau_up": tau_up, "tau_dn": tau_dn,
            "nll": float(nll_tot / max(n_bands, 1))}


# ─────────────────────────────────────────────────────────────────────────────
#  M2b (development alternative) — concordance: which delay explains each
#  pixel's wet sequence
# ─────────────────────────────────────────────────────────────────────────────

def m2b_concordance(wet, clear, bank, band_of, n_bands, tau_grid,
                    min_obs=10, rng=None, max_px_band=1500):
    """Lag per band by per-pixel point-biserial concordance.

    Development alternative (M2b), not used by MAREA. For each candidate
    lag, correlate every pixel's wet/dry sequence with the shifted boundary
    series over its clear scenes; the band's score is the MEDIAN correlation
    over its pixels (robust to the always-wet/always-dry tails, which carry
    no clock information). The lag that maximises the score is the band's
    delay. It sees the delay but not the gain (see the module docstring).

    ``wet``, ``clear`` (T, P) bool; ``tau_grid`` evenly spaced candidate
    delays (min); pixels with fewer than ``min_obs`` clear looks, or wet in
    under 2 % or over 98 % of them, are left out. Returns (n_bands,)
    minutes, NaN where undetermined.
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
#  M2c (development alternative) — waterline: flooded-area ranks
# ─────────────────────────────────────────────────────────────────────────────

def m2c_waterline(wet, clear, bank, band_of, n_bands, tau_grid,
                  min_cover=0.3, min_scenes=60):
    """Lag per band by rank correlation of flooded area against the shifted
    boundary — the binary-gauging idea, which reproduced the gauges' lag
    gradient on the Scheldt. Affine-blind by the
    same theorem: ranks of area survive any monotone gain/datum, so tau only.

    Development alternative (M2c), not used by MAREA. A look counts in a
    section when at least ``min_cover`` of the section's pixels are clear;
    a section needs ``min_scenes`` such looks. ``wet``, ``clear`` (T, P)
    bool; ``tau_grid`` evenly spaced candidate delays (min). Returns
    (n_bands,) minutes, NaN where undetermined.
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
    """Sub-grid argmax by parabola through the top three points.

    ``score`` (k,) and an evenly spaced ``tau_grid`` (k,). The refinement
    moves at most one grid step; at the ends of the grid, or next to a NaN,
    the grid point itself is returned; with fewer than 3 finite scores, NaN.
    """
    if np.isfinite(score).sum() < 3:
        return np.nan
    j = int(np.nanargmax(score))
    if 0 < j < len(tau_grid) - 1 and np.isfinite(score[j - 1:j + 2]).all():
        d0, d1, d2 = score[j - 1], score[j], score[j + 1]
        den = d0 - 2 * d1 + d2
        off = 0.5 * (d0 - d2) / den if abs(den) > 1e-12 else 0.0
        step = tau_grid[1] - tau_grid[0]
        # vertex of the parabola through (-1, d0), (0, d1), (1, d2) is at +off
        return float(tau_grid[j] + np.clip(off, -1, 1) * step)
    return float(tau_grid[j])


# ─────────────────────────────────────────────────────────────────────────────
#  M2a — the estimator MAREA uses: Bernoulli likelihood with profiled
#  elevations (in item-response terms: a Rasch model, looks = examinees,
#  pixels = items with difficulty z)
# ─────────────────────────────────────────────────────────────────────────────

def _band_nll(wet, clear, h, z_grid, sigma0, sg_grid=None, min_looks=6):
    """Mean Bernoulli NLL of a band, difficulties profiled on ``z_grid``.

    ``wet``/``clear``: (T, P) boolean. Vectorised as two matmuls; the z that
    maximises each pixel's likelihood is chosen and never leaves this
    function — that is the profiling.

    ``h`` (T,) the (delayed) mouth levels, m; ``z_grid`` candidate
    elevations, m; ``sigma0`` the transition width (m) used without
    ``sg_grid``. ``sg_grid``: optionally profile the transition width per
    pixel too, over the archive's own sigma values. With one fixed sigma0 a
    scale misfit shows up as a false gain or delay (measured: the gain
    pinned at its bound); profiling sigma keeps it inside the pixel.

    Returns ``(nll, n_pixels)``: the NLL per clear look over the pixels with
    at least ``min_looks`` clear looks, and how many such pixels there are.
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
    ok = n >= min_looks
    return -float(best[ok].sum() / max(n[ok].sum(), 1)), int(ok.sum())


def alpha_nll_profile(wet, clear, bank, tau, alpha_grid, z_grid, sg_grid):
    """NLL(alpha) with (z, sigma) profiled per pixel — the degeneracy meter.

    Wet/dry data cannot see gain (see "Why no gain is fitted" in the module
    docstring): rescaling (alpha, z, sigma) TOGETHER leaves every likelihood term
    unchanged. The profiling grids therefore scale with alpha — otherwise
    their fixed ceilings penalise large alpha and manufacture
    identifiability out of quantisation (measured once: a spurious 0.18 NLL
    slope, entirely from sigma being capped at its top value). With scaled
    grids a flat curve verifies the implementation adds no false
    identification; curvature would mean a coding error, not physics.

    ``wet``, ``clear`` (T, P) of one section; ``tau`` its delay (min);
    ``alpha_grid`` the gains tried; ``z_grid`` and ``sg_grid`` the elevation
    and transition-width grids at gain 1 (m). Returns the NLL per gain.
    Development check, not used by MAREA.
    """
    out = []
    for al in alpha_grid:
        al = float(al)
        h = al * bank.at(tau)
        v, _ = _band_nll(wet, clear, h, al * np.asarray(z_grid, float),
                         None, sg_grid=[al * s for s in sg_grid])
        out.append(v)
    return np.asarray(out)


def make_bands_fixed(s_km, length_km=3.0, min_px=500):
    """Bands of fixed length along the mouth distance, merged forward when
    too thin; returns (edges, centers, band_of_pixel) like :func:`make_bands`.

    Band k spans [k L, (k+1) L) km. A band with fewer than ``min_px`` pixels
    is merged with the next one (the last thin band with the previous), so
    every band has enough pixels to fix its own clock; the number of bands
    is set by the estuary's length, not chosen. Centres are the median s of
    each band's pixels.

    Parameters
    ----------
    s_km : (P,) float
        Distance of each pixel from the mouth along the water, km; NaN
        where water cannot reach.
    length_km : float
        Section length L, km (MAREA: 3.0, ``band_km``).
    min_px : int
        Thinner sections are merged (MAREA: 500, ``min_px_band``).

    Returns
    -------
    edges : (n_raw + 1,) float
        The raw edges 0, L, 2L, ... km from BEFORE the merge: after a merge
        ``len(edges) - 1`` is larger than the number of sections, so do not
        use them to find a pixel's section — use ``band_of``.
    centers : (n,) float
        Median s (km) of each final section, from the mouth inward.
    band_of : (P,) int
        Section of each pixel, 0 at the mouth; −1 where s is NaN.
    """
    s = np.asarray(s_km, float)
    fin = np.isfinite(s)
    n_raw = int(np.floor(np.nanmax(s) / length_km)) + 1
    # NaN s cast to int raises a harmless RuntimeWarning ("invalid value
    # encountered in cast"); np.where then sets those pixels to -1
    raw = np.where(fin, np.floor(s / length_km).astype(int), -1)
    raw = np.clip(raw, -1, n_raw - 1)
    counts = np.bincount(raw[raw >= 0], minlength=n_raw)
    # forward merge of thin bands
    label = np.arange(n_raw)
    k = 0
    while k < n_raw:
        j = k
        while counts[label == label[k]].sum() < min_px and j + 1 < n_raw:
            j += 1
            label[label == label[j]] = label[k]
        k = j + 1
    # a thin last band joins the previous one
    uniq = np.unique(label)
    if len(uniq) > 1 and counts[label == uniq[-1]].sum() < min_px:
        label[label == uniq[-1]] = uniq[-2]
    uniq = np.unique(label)
    remap = {u: i for i, u in enumerate(uniq)}
    band = np.full(len(s), -1, int)
    band[raw >= 0] = [remap[label[r]] for r in raw[raw >= 0]]
    edges = np.array([k * length_km for k in range(n_raw + 1)])   # pre-merge
    centers =np.array([float(np.median(s[band == b])) for b in range(len(uniq))])
    return edges, centers, band


def smooth_lag_profile(centers_km, tau_min, s_query_km, bandwidth_km=3.0):
    """The lag as a continuous function of the mouth distance: Gaussian
    kernel smoothing of the band lags, anchored at tau = 0 at s = 0 and
    made monotone non-decreasing along s.

    In order: (1) a Gaussian-kernel average of the section delays plus an
    anchor point (0 km, 0 min); (2) a running maximum along s, so the delay
    never decreases inland; (3) clipped at 0, never negative.
    ``marea.reconstruct`` then rounds the result to whole minutes (and
    gives 0 to the pixels without a distance).

    Parameters
    ----------
    centers_km : (n,) float
        Section centres, km (:func:`make_bands_fixed`).
    tau_min : (n,) float
        Section delays, minutes.
    s_query_km : (P,) float
        Distance of each pixel from the mouth, km; NaN allowed.
    bandwidth_km : float
        Standard deviation of the kernel, km (MAREA: the section length).

    Returns tau (min) at every ``s_query_km``; NaN where s is NaN.
    """
    c = np.concatenate([[0.0], np.asarray(centers_km, float)])
    t = np.concatenate([[0.0], np.asarray(tau_min, float)])
    s = np.asarray(s_query_km, float)
    out = np.full(s.shape, np.nan)
    fin = np.isfinite(s)
    w = np.exp(-0.5 * ((s[fin][:, None] - c[None, :]) / bandwidth_km) ** 2)
    out[fin] = (w * t[None, :]).sum(axis=1) / w.sum(axis=1)
    # monotone along s (the kernel of a monotone sequence is monotone up to
    # numerical spacing effects; enforce it exactly)
    order = np.argsort(s[fin])
    vals = out[fin][order]
    out_fin = out[fin]
    out_fin[order] = np.maximum.accumulate(vals)
    out[fin] = np.maximum(out_fin, 0.0)
    return out


def bootstrap_lags(wet, clear, bank, band_of, centers_km, n_bands, n_boot=30,
                   seed=7, **kwargs):
    """Scene-bootstrap of the band lags: refit :func:`m2a_rasch` on
    ``n_boot`` resamples of the scenes (with replacement) and return the
    (n_boot, n_bands) array of lags. The 2.5 and 97.5 percentiles of each
    column are the 95 % confidence interval of that band's lag.

    Parameters
    ----------
    wet, clear : (T, P) bool
    bank : ShiftBank
        Its series are resampled with the looks.
    band_of, centers_km, n_bands
        As for :func:`m2a_rasch`.
    n_boot : int
        Number of resamples (MAREA: 30). With ``n_boot=0`` MAREA does not
        call this function at all (the intervals are then NaN).
    seed : int
        Seeds the resamples; each refit also gets its own seed
        (``seed + 100 + b``), so the result does not depend on how many
        threads ran.
    **kwargs
        Passed to :func:`m2a_rasch` (MAREA passes its fit settings with
        ``max_px_band=600``).

    ``np.nanpercentile(out, [2.5, 97.5], axis=0)`` gives MAREA's
    ``tau_lo_min`` and ``tau_hi_min``. Runs in threads (joblib, up to the
    number of CPUs minus 2) when joblib is installed, else in a loop.
    """
    rng = np.random.default_rng(seed)
    T = wet.shape[0]
    draws = [rng.integers(0, T, T) for _ in range(n_boot)]

    def one(b):
        idx = draws[b]
        bk = ShiftBank(bank.taus, bank.H[:, idx])
        r = m2a_rasch(wet[idx], clear[idx], bk, band_of, centers_km, n_bands,
                      rng=np.random.default_rng(seed + 100 + b), **kwargs)
        return r["tau"]

    # the band likelihood is two matmuls that release the GIL, so threads
    # give real parallelism without duplicating the record in memory
    try:
        import os
        from joblib import Parallel, delayed
        n_jobs = max(1, min(n_boot, (os.cpu_count() or 2) - 2))
        taus = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(b) for b in range(n_boot))
    except ImportError:
        taus = [one(b) for b in range(n_boot)]
    return np.asarray(taus, float)


def m2a_rasch(wet, clear, bank, band_of, centers_km, n_bands, sigma0=0.20,
              alpha_bounds=(0.7, 1.4), tau_bounds=(-30.0, 120.0),
              z_points=60, rng=None, max_px_band=1200, verbose=False,
              bank_by_band=None, fit_alpha=False, sg_grid=None,
              monotone=True, min_looks=6):
    """Fit alpha(s), tau(s) on band centres by profiled Bernoulli likelihood.

    The estimator MAREA uses (steps 7-8): one delay per section. The mouth
    section is pinned to (alpha=1, tau=0): the anchor. The free parameters
    are the delays of the other sections — ``n_bands - 1`` numbers (twice
    that with ``fit_alpha``) for L-BFGS-B, started at 0 and stopped after at
    most 60 iterations.

    Parameters
    ----------
    wet, clear : (T, P) bool
        Whether each look is wet (NDWI > 0) and whether it is clear.
    bank : ShiftBank
        The tide at the mouth for any delay (levels in m).
    band_of : (P,) int
        Section of each pixel (:func:`make_bands_fixed`); −1 = none.
    centers_km : (n_bands,) float
        Section centres; only copied into the result.
    n_bands : int
        Number of sections; section 0 is the mouth.
    sigma0 : float
        Transition width (m) used when ``sg_grid`` is None.
    alpha_bounds : (lo, hi)
        Bounds of the gain, used only with ``fit_alpha=True``.
    tau_bounds : (lo, hi)
        Minutes. With ``monotone=True`` (MAREA) the free parameters are the
        per-section INCREMENTS of the delay, each in [0, ``tau_bounds[1]``],
        and ``tau_bounds[0]`` is not used; the total delay is their running
        sum, so it is not bounded by ``tau_bounds[1]`` — but the bank clamps
        any delay to its own range (MAREA: 180 min), so beyond that the
        likelihood is flat: a delay at or above the bank's end is a delay the
        record could not measure. With ``monotone=False`` each delay lies in
        ``tau_bounds``.
    z_points : int
        Points of the elevation grid, spanning the mouth tide's range ± 0.3 m.
    rng : numpy Generator, optional
        Seeds the pixel subsample (default seed 1).
    max_px_band : int
        At most this many pixels per section enter the likelihood.
    verbose : bool
        Print one line with the result.
    bank_by_band : list of ShiftBank, optional
        One bank per section instead of ``bank``: a development check
        (feed each section its own level series and the refit must come back
        at alpha=1, tau=0).
    fit_alpha : bool
        **False by default**, and MAREA keeps it so: wet/dry data cannot see
        a per-section gain (the module docstring), and when it was fitted
        the optimiser spent the gain on a misfit of sigma (pinned at its
        bound). :func:`alpha_nll_profile` shows the flat likelihood.
        ``fit_alpha=True`` is kept for degeneracy studies only.
    sg_grid : sequence of float, optional
        Transition widths (m) to profile per pixel (MAREA:
        ``configs/m2.yaml`` ``sigma_perfil_m``) — without it, a misfit of
        scale leaks into tau as well.
    monotone : bool
        True: the delay never decreases inland (a propagation delay
        accumulates).
    min_looks : int
        A pixel with fewer clear looks does not enter the likelihood. With
        every pixel below it the objective is flat and every lag stays at 0.

    A section with fewer than 50 pixels does not enter the likelihood
    either, so its own pixels say nothing about its delay; with
    ``monotone=True`` its delay ends between its neighbours'.

    Returns
    -------
    dict
        ``alpha`` (n_bands,) — ones unless ``fit_alpha``; ``tau``
        (n_bands,) minutes, ``tau[0] = 0``; ``nll`` the mean negative
        log-likelihood per clear look (sections weighted by their pixels);
        ``centers_km``; ``converged`` (L-BFGS-B's success flag).
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
        th_tau = theta[nf:] if fit_alpha else theta
        # monotone: the free parameters are the INCREMENTS of the lag from
        # one band to the next, each >= 0, so the profile can only grow
        # inland (a propagation delay accumulates; causality is built in)
        lags = np.cumsum(th_tau) if monotone else th_tau
        tau = np.concatenate([[0.0], lags * 60.0])
        alpha = np.concatenate([[1.0], theta[:nf]]) if fit_alpha else np.ones(n_bands)
        return alpha, tau

    def objective(theta):
        alpha, tau = unpack(theta)
        total, nn = 0.0, 0
        for k in range(n_bands):
            cols = cols_by_band[k]
            if len(cols) < 50:
                continue                  # too few pixels: out of the likelihood
            bk = bank if bank_by_band is None else bank_by_band[k]
            h = alpha[k] * bk.at(tau[k])
            nll, npx = _band_nll(wet[:, cols], clear[:, cols], h,
                                 z_grid, sigma0, sg_grid=sg_grid, min_looks=min_looks)
            total += nll * npx
            nn += npx
        return total / max(nn, 1)

    # tau is optimised in HOURS so every free parameter is O(1) and one
    # finite-difference step suits them all
    tb = ((0.0, tau_bounds[1] / 60.0) if monotone
          else (tau_bounds[0] / 60.0, tau_bounds[1] / 60.0))
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
