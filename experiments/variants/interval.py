# -*- coding: utf-8 -*-
"""V13 of the addendum of 2026-10-28 (experiments/variants_addendum_2026-10-28.md): MAREA-B's likelihood interval
and abstention rule, the error-coverage curves of every method, and the block bootstrap of the judging rule.

bernoulli_interval
    MAREA-B (V2) per pixel, as the attribution notebook's ``invert_bernoulli``
    (notebooks/validation/_build_05_attribution.py) computes it, bit for bit: the probit likelihood of the pixel's
    wet/dry record on its clear looks, P(wet) = Phi((h - z)/sigma), on ``marea.invert_series``' grid (50 levels
    between the window's lowest and highest level, sigma in ``marea.SG_GRID``), z = the median of the tied levels.
    The same pass keeps the profile log-likelihood lp(z) = max over sigma, and the 95 % interval is the set of grid
    levels with lp(z) >= max lp - chi2_1(0.95)/2 = max lp - 1.9207 (Wilks' likelihood-ratio interval). The interval
    is *bracketed* when it touches neither end of the grid; V13a is V2's z where it is, no value elsewhere (there
    the window's levels do not bound the elevation on one side).
error_coverage
    every method's RMSE, bias and centred RMSE (notebook 01's ``scores()``) on the most reliable share q of the
    pixels, for q = 100, 95, 90, 85 and 80 %. The reliability is one number per pixel that no method chooses (the
    addendum: r = min(wet looks, dry looks) in the window), so a method that abstains is compared on the same
    ranking as one that maps everything; each method is scored on its pixels with a value in the share and on the
    share's pixels where every method has a value.
block_bootstrap
    the paired difference variant - reference of RMSE and centred RMSE on the pixels both have, with the 95 %
    percentile interval of 1,000 resamples of 32 x 32-pixel blocks of the grid (320 m), seed 20261028. A resample
    is the multiset of the drawn blocks' pixels and its statistic is ``scores()``'s on it, each median the exact
    median of the multiset (``experiments.paired_bootstrap.WeightedMedian``).

The inputs are those of the attribution notebook's estimators, per window and delay group: Y (looks x pixels) NDWI
with NaN where undefined, C (looks x pixels) the clear looks, h (looks,) the levels read with MAREA's delay, lo and
hi the window's lowest and highest level without delay (the grid bounds). float64 arithmetic, blocks of 4,000
pixels as ``marea.invert_series``.
"""
import numpy as np
import pandas as pd
from scipy.special import log_ndtr                    # log Phi of the standard normal
from scipy.stats import chi2

from experiments.paired_bootstrap import WeightedMedian
from pyintertidal import marea


# ─────────────────────────────────────────────────────────────────────────────
#  MAREA-B and its profile-likelihood interval
# ─────────────────────────────────────────────────────────────────────────────

def bernoulli_interval(Y, C, h, lo, hi, wet=None, min_looks=8, tie_tol=1e-9, mu_points=50, chunk=4000, level=0.95):
    """MAREA-B (V2) and its profile-likelihood interval (V13), per pixel.

    z and sigma are the attribution notebook's ``invert_bernoulli``, bit for bit (its arithmetic: the same log_ndtr
    terms, matrix products, blocks of ``chunk`` pixels, tie rule and guards): the probit likelihood of the wet/dry
    record on the clear looks against the levels h, P(wet) = Phi((h - z)/sigma), maximised on invert_series' grid
    (mu_points levels in [lo, hi], sigma in marea.SG_GRID); z = the median of the tied z levels (profile
    log-likelihood within tie_tol of its maximum), sigma = the best sigma at the tied level nearest z.

    The interval: lp(z) = the maximum over sigma of that log-likelihood at each grid level (the profile); z_lo and
    z_hi = the lowest and the highest grid level with lp(z) >= max lp - chi2.ppf(level, 1)/2 (1.9207 for 0.95). It
    holds z (every tied level is in it); were lp to dip below the cut between two levels above it, [z_lo, z_hi] is
    the hull of the set. bracketed = z_lo > grid[0] and z_hi < grid[-1]: the window's levels bound the elevation on
    both sides.

    wet: None, wet = C & (Y > 0) (an undefined NDWI is never wet, a clear look that is not wet is dry); or an
    explicit (looks x pixels) bool matrix, taken on the clear looks only (V14's per-scene threshold), and then Y is
    not read. NaN (bracketed False) with fewer than min_looks clear looks or without a wet and a dry one. Returns
    (z, sigma, z_lo, z_hi, bracketed).
    """
    grid, sgs = np.linspace(lo, hi, mu_points), np.asarray(marea.SG_GRID, float)
    u = (np.asarray(h, float)[:, None, None] - grid[None, :, None]) / sgs[None, None, :]
    lw, ld = log_ndtr(u).reshape(len(h), -1), log_ndtr(-u).reshape(len(h), -1)   # (looks, levels x sigmas)
    drop = chi2.ppf(level, 1) / 2.0                     # the likelihood-ratio cut: 1.9207 for level 0.95
    P = C.shape[1]
    z, sg = np.full(P, np.nan), np.full(P, np.nan)
    z_lo, z_hi, bracketed = np.full(P, np.nan), np.full(P, np.nan), np.zeros(P, bool)
    for j in range(0, P, chunk):
        s = slice(j, min(j + chunk, P))
        clear = np.asarray(C[:, s], bool)
        if wet is None:
            with np.errstate(invalid="ignore"):
                w_s = clear & (np.asarray(Y[:, s]) > 0)     # an undefined NDWI is never wet
        else:
            w_s = clear & np.asarray(wet[:, s], bool)
        dry = clear & ~w_s
        # from here to `z[s], sg[s] = ...`, invert_bernoulli's lines (its `wet` is w_s)
        ll = (w_s.T.astype(float) @ lw + dry.T.astype(float) @ ld).reshape(-1, len(grid), len(sgs))
        prof = ll.max(axis=2)                           # the profile over sigma: (pixels, levels)
        tied = prof >= prof.max(axis=1, keepdims=True) - tie_tol
        zb = np.nanmedian(np.where(tied, grid[None, :], np.nan), axis=1)
        near = np.argmin(np.where(tied, np.abs(grid[None, :] - zb[:, None]), np.inf), axis=1)
        sb = sgs[ll[np.arange(len(near)), near].argmax(axis=1)]
        ok = (clear.sum(axis=0) >= min_looks) & w_s.any(axis=0) & dry.any(axis=0)
        z[s], sg[s] = np.where(ok, zb, np.nan), np.where(ok, sb, np.nan)
        # the interval: the grid levels whose profile is within `drop` of its maximum, from the first to the last
        inside = prof >= prof.max(axis=1, keepdims=True) - drop
        first, last = inside.argmax(axis=1), len(grid) - 1 - inside[:, ::-1].argmax(axis=1)
        has = ok & inside.any(axis=1)                   # (a NaN level leaves no level inside: no interval)
        z_lo[s], z_hi[s] = np.where(has, grid[first], np.nan), np.where(has, grid[last], np.nan)
        bracketed[s] = has & (grid[first] > grid[0]) & (grid[last] < grid[-1])
    return z, sg, z_lo, z_hi, bracketed


# ─────────────────────────────────────────────────────────────────────────────
#  notebook 01's scores, and the error-coverage curves
# ─────────────────────────────────────────────────────────────────────────────

SCORE_KEYS = ("n_px", "rmse", "bias", "crmse")


def rmse_bias_crmse(p, t, min_px=30):
    """n_px, rmse, bias and crmse of notebook 01's scores(p, t), with its arithmetic: on the pixels where p and t are
    finite, RMSE = sqrt(mean((p - t)^2)), bias = median(p - t), centred RMSE = sqrt(mean(((p - median p) -
    (t - median t))^2)); NaN on fewer than min_px pixels (scores(): 30)."""
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = np.isfinite(p) & np.isfinite(t)
    out = dict(n_px=int(ok.sum()), rmse=np.nan, bias=np.nan, crmse=np.nan)
    if ok.sum() >= max(min_px, 1):
        d = p[ok] - t[ok]
        pc, tc = p[ok] - np.median(p[ok]), t[ok] - np.median(t[ok])
        out.update(rmse=float(np.sqrt(np.mean(d ** 2))), bias=float(np.median(d)),
                   crmse=float(np.sqrt(np.mean((pc - tc) ** 2))))
    return out


def reliability_ranking(truth, reliability):
    """The pixels with a finite truth (F, as positions in the arrays), most reliable first: reliability descending,
    ties by pixel index (a NaN reliability ranks last)."""
    F = np.flatnonzero(np.isfinite(np.asarray(truth, float)))
    r = np.asarray(reliability, float)[F]
    key = np.where(np.isfinite(r), -r, np.inf)
    return F[np.argsort(key, kind="stable")]           # F is in index order: the stable sort breaks ties by index


def error_coverage(truth, estimates, reliability, coverages=(1.0, 0.95, 0.90, 0.85, 0.80), min_px=30):
    """The error-coverage curves of V13: every method scored on the most reliable share q of the pixels.

    F = the pixels with a finite truth, ranked by ``reliability`` (``reliability_ranking``: largest first, ties by
    pixel index). At each coverage q the subset is the first ceil(q |F|) of them, and every method of ``estimates``
    ({name: elevation per pixel}, on the pixels of ``truth``) gets notebook 01's scores (``rmse_bias_crmse``) on its
    pixels with a value in the subset (n_px, rmse, bias, crmse) and on the subset's pixels where every method of
    ``estimates`` has a value (n_px_common, rmse_common, bias_common, crmse_common). The pixels are scored in index
    order, so q = 1 gives the notebook's scores(p, truth), bit for bit.

    Returns a DataFrame with one row per coverage and method, in the order given: coverage, method, n_subset
    (= ceil(q |F|)) and the eight scores.
    """
    truth = np.asarray(truth, float)
    est = {m: np.asarray(p, float) for m, p in estimates.items()}
    ranked = reliability_ranking(truth, reliability)
    rows = []
    for q in coverages:
        # ceil(q |F|); the rounding drops the float error of q (0.07 * 100 is 7.000000000000001)
        n_keep = int(np.ceil(round(q * len(ranked), 9)))
        keep = np.zeros(len(truth), bool)
        keep[ranked[:n_keep]] = True
        common = keep.copy()
        for p in est.values():
            common &= np.isfinite(p)
        for m, p in est.items():
            own, com = rmse_bias_crmse(p[keep], truth[keep], min_px), rmse_bias_crmse(p[common], truth[common], min_px)
            rows.append(dict(coverage=float(q), method=m, n_subset=n_keep, **own,
                             **{f"{k}_common": v for k, v in com.items()}))
    return pd.DataFrame(rows, columns=["coverage", "method", "n_subset", *SCORE_KEYS,
                                       *(f"{k}_common" for k in SCORE_KEYS)])


# ─────────────────────────────────────────────────────────────────────────────
#  the block bootstrap of the paired difference
# ─────────────────────────────────────────────────────────────────────────────

def block_index(rows, cols, block=32):
    """(block number per pixel, number of blocks): the blocks of block x block grid pixels (rows // block,
    cols // block) that hold a pixel, numbered in the order (block row, block column)."""
    rb, cb = np.asarray(rows, np.int64) // block, np.asarray(cols, np.int64) // block
    if not len(rb):
        return np.zeros(0, np.int64), 0
    rb, cb = rb - rb.min(), cb - cb.min()
    labels, cl = np.unique(rb * (int(cb.max()) + 1) + cb, return_inverse=True)
    return cl.astype(np.int64).ravel(), int(len(labels))


def block_bootstrap(truth, p_variant, p_reference, rows, cols, block=32, n_boot=1000, seed=20261028, min_px=30):
    """The bootstrap of the judging rule: the paired difference variant - reference of RMSE and centred RMSE on the
    pixels where truth, variant and reference are all finite, and its 95 % percentile interval.

    The observed values are notebook 01's scores (``rmse_bias_crmse``) on those pixels as they are. The pixels go in
    blocks of block x block grid pixels (``block_index``; rows and cols are the pixels' positions in the grid: 32
    pixels of 10 m, 320 m). Each of the n_boot resamples draws as many blocks as there are, with replacement, the
    same draw for both estimates (paired): numpy's default_rng(seed), one integers(0, B, B) per resample. A resample
    is the multiset of the drawn blocks' pixels (a pixel counts as many times as its block was drawn); its RMSE is
    the weighted root mean square of p - t, its centred RMSE that of (p - t) - (median p - median t) with each
    median the exact median of the multiset (numpy.median of the resampled values: paired_bootstrap's
    WeightedMedian), so a resample's statistic is what scores() gives on the concatenation of its blocks; the sums
    are taken from per-block sums, no pixel is copied. The interval is the 2.5 and 97.5 percentiles of the n_boot
    differences. NaN with fewer than min_px pixels.

    Returns dict(n_px, n_blocks, rmse_variant, rmse_reference, d_rmse, d_rmse_lo, d_rmse_hi, crmse_variant,
    crmse_reference, d_crmse, d_crmse_lo, d_crmse_hi).
    """
    t, pv, pr = (np.asarray(x, float) for x in (truth, p_variant, p_reference))
    both = np.isfinite(t) & np.isfinite(pv) & np.isfinite(pr)
    t, pv, pr = t[both], pv[both], pr[both]
    cl, B = block_index(np.asarray(rows)[both], np.asarray(cols)[both], block)
    sv, sr = rmse_bias_crmse(pv, t, min_px), rmse_bias_crmse(pr, t, min_px)
    out = dict(n_px=int(both.sum()), n_blocks=B,
               rmse_variant=sv["rmse"], rmse_reference=sr["rmse"], d_rmse=sv["rmse"] - sr["rmse"],
               d_rmse_lo=np.nan, d_rmse_hi=np.nan,
               crmse_variant=sv["crmse"], crmse_reference=sr["crmse"], d_crmse=sv["crmse"] - sr["crmse"],
               d_crmse_lo=np.nan, d_crmse_hi=np.nan)
    if not np.isfinite(out["d_rmse"]):                  # fewer than min_px pixels: no comparison
        return out
    n_b = np.bincount(cl, minlength=B).astype(float)  # pixels per block

    def block_sums(d):
        """Per block, the sums a resample's scores are made of: of d^2 (the RMSE), and of d - m and (d - m)^2 about
        the mean m of d (the centred RMSE; shifted so that its terms do not cancel)."""
        m = float(d.mean())
        e = d - m
        return np.bincount(cl, d * d, B), m, np.bincount(cl, e, B), np.bincount(cl, e * e, B)

    def centred(k, n, c, m, s, q):
        """The root mean square over the multiset of d - c, c = median p - median t:
        sum (d - c)^2 = sum (d - m)^2 - 2 (c - m) sum (d - m) + (c - m)^2 n."""
        e = c - m
        return np.sqrt(max(k @ q - 2.0 * e * (k @ s) + e * e * n, 0.0) / n)

    (q_v, m_v, s_v, q2_v), (q_r, m_r, s_r, q2_r) = block_sums(pv - t), block_sums(pr - t)
    med_t, med_v, med_r = WeightedMedian(t, cl), WeightedMedian(pv, cl), WeightedMedian(pr, cl)
    rng = np.random.default_rng(seed)
    d_rmse, d_crmse = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        k = np.bincount(rng.integers(0, B, B), minlength=B)                # times each block is drawn
        n, mt = float(k @ n_b), med_t(k)
        d_rmse[i] = np.sqrt(k @ q_v / n) - np.sqrt(k @ q_r / n)
        d_crmse[i] = (centred(k, n, med_v(k) - mt, m_v, s_v, q2_v)
                      - centred(k, n, med_r(k) - mt, m_r, s_r, q2_r))
    out["d_rmse_lo"], out["d_rmse_hi"] = (float(v) for v in np.quantile(d_rmse, [0.025, 0.975]))
    out["d_crmse_lo"], out["d_crmse_hi"] = (float(v) for v in np.quantile(d_crmse, [0.025, 0.975]))
    return out
