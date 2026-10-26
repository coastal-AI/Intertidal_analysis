# -*- coding: utf-8 -*-
"""Validation v2 — lag inference: truth-side bootstrap weights, exact sign-flip
tests, Holm, family L-P1, ICC(2,1), Lin's CCC and the attainability report
(spec §6.4-§6.5, §9.1-§9.2, §9.8).

Question
--------
How uncertain is each lag metric because of the truth, is MAREA's lag closer
to the in-estuary gauges than each pre-registered competitor's, and can each
frozen family reject at all?

Rules (spec section in brackets; every number in :data:`INFERENCE_V2`, or
read from FROZEN where the value exists)
------------------------------------------------------------------------
* Truth-side month-block bootstrap (§6.5 item 4, scoring spec §4): calendar
  months drawn with replacement over the months present among a window's S2
  instants with a finite gauge value; an instant enters with the
  multiplicity of its month (:func:`month_block_weights`). The draw loop is
  ``truth.month_boot``'s (``rng.integers(0, n_months, n_months)`` per draw),
  so with the truth's own seed the weights reproduce its 200 draws; the
  scorer seeds ``[*truth.window_boot_seed(site, code, w), 1]`` (a stream of
  its own) and shares one weight matrix by every product (paired). B = 1000
  (FROZEN inference.boot.B_descriptive). Statistics are evaluated exactly as
  on the replicated sample: the weighted median is the median of the
  replicated values (:func:`weighted_median`), the median-centred RMS is
  :func:`weighted_mcrms`. Percentile intervals 2.5-97.5, labelled "truth
  uncertainty only".
* Family L-P1 (§6.5 item 1): per location d_l = |e_MAREA,l| - |e_X,l|; the
  statistic is the mean over the judged sites of the site-mean d (each site
  weight 1); an EXACT two-sided sign-flip test over all 2^n location sign
  vectors, p = #{|S*| >= |S| - 1e-12} / 2^n, with k_zero exactly-zero
  differences and the attainable minimum p_min = 2 / 2^(n - k_zero)
  (:func:`sign_flip`); Holm within each truth over the family
  (:func:`holm`, alpha = FROZEN inference.holm_alpha); confirmatory iff Holm
  rejects AND the sign of the site-mean d agrees with S in at least 3 of the
  4 judged sites; a ranking only if it holds under all three truths
  (:func:`ranking_claims`). Pooled location tests are description only,
  labelled "pseudo-replicated" (:func:`pooled_sign_test`).
* ICC(2,1), two-way random effects, absolute agreement, single measure
  (Shrout & Fleiss 1979, doi:10.1037/0033-2909.86.2.420), on complete rows
  (:func:`icc21`); Lin's concordance correlation with population moments
  (Lin 1989, doi:10.2307/2532051; :func:`lin_ccc`).
* Attainability (§9.8, :func:`attain_report`): ``config.attainability()``
  (2 / (B + 1) against alpha / m for every bootstrap family); the smallest
  exact sign-flip p of L-P1 and L-P2 given the included locations per judged
  site at W3 (from the truth documents, under the window-year guard) against
  the Holm thresholds; the expected number of square blocks per site at
  1, 4 and 16 km and the 3-km bands holding truth, from the D_truth geometry
  of the package support (no product is read); the denominators and the
  rules for untestable members (§9.2). Written to ``<root>/attain.json``.

This module imports numpy and pandas and the benchmark's config at import
time; ``attain_report`` imports the package module lazily.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments.benchmark import config

_INF = config.FROZEN["inference"]

#: every constant of the lag inference (lock.py records the sha256 of it)
INFERENCE_V2 = {
    "boot": {
        "B_descriptive": int(_INF["boot"]["B_descriptive"]),
        "B_primary": int(_INF["boot"]["B_primary"]),
        "block": "calendar month",
        "draw": "rng.integers(0, n_months, n_months) per draw (truth.month_boot's loop)",
        "seed": "[*truth.window_boot_seed(site, code, window), 1]",
        "seed_extra": 1,
        "ci_percentiles": [2.5, 97.5],
        "label": "truth uncertainty only",
        "paired": "one weight matrix per (site, judge, window), shared by every product",
        "statistic": "weighted = evaluated on the replicated sample (weighted median, "
                     "median-centred RMS)",
    },
    "sign_flip": {"exact": True, "two_sided": True, "tie_tol": 1e-12, "max_n": 30,
                  "chunk": 1 << 16,
                  "statistic": "mean over judged sites of the site-mean d"},
    "holm_alpha": float(_INF["holm_alpha"]),
    "lp1": {
        "marea": "L01",
        "comparisons": ["L17", "L14", "L16", "L08", "L09"],
        "comparison_names": {"L17": "zero lag", "L14": "primary global model",
                             "L16": "regional model", "L08": "GM20", "L09": "HD26-H"},
        "truths": ["a", "ab", "b"],
        "judged_sites": ["escalda", "wadden", "ems", "ferrol"],
        "min_agree_sites": 3,
        "window": "W3",
        "descriptive_windows": ["W3", "W2", "W1", "Y2025", "Y2024", "Y2023"],
        "descriptive_label": "descriptive, mixed windows, not the pre-registered L-P1",
        "confirmatory_label": "confirmatory L-P1 (W3, LOCK phase A)",
        "pooled_label": "pseudo-replicated (description only)",
        "d": "|e_MAREA,l| - |e_X,l| (circular |e| for phase-type outputs), location = mean "
             "over its gauges",
    },
    "icc": "ICC(2,1): two-way random, absolute agreement, single measure (Shrout & Fleiss "
           "1979); complete rows only",
    "ccc": "Lin (1989) with population (1/n) moments",
    "attain": {
        "block_sizes_km": [1, 4, 16],
        "min_blocks": int(_INF["min_blocks"]),
        "band_km": float(config.FROZEN["domain"]["bands"]["length_km"]),
        "min_bands_p1": 6,
        "min_truth_px_band": 30,
        "lp1_m": [5, 4],
        "lp2_m": 1,
        "rules": [
            "family membership is frozen at lock-A as the nominal sets (m per family)",
            "the truth-free 'score --blocks' step removes members that cannot be tested "
            "(fewer than 25 blocks on the pair view; fewer than 6 bands with truth for P1 "
            "accuracy) and logs the removal; the Holm / Romano-Wolf denominators are then "
            "the nominal m minus the logged removals, fixed before any truth is read",
            "a competitor that fails validation with a documented fault is removed before "
            "any score",
            "L-P1 has 5 comparisons per truth (4 when the regional model L16 is not "
            "available at lock-A); L-P2 one per truth",
            "a smallest attainable p not below Holm's first threshold alpha / m makes a "
            "family unable to reject: the report says so",
        ],
    },
}
del _INF

ALPHA = INFERENCE_V2["holm_alpha"]


# ─────────────────────────────────────────────────────────────────────────────
#  truth-side month-block bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def month_labels(t):
    """``year * 12 + month`` of each instant (the label of truth.month_boot)."""
    t = pd.DatetimeIndex(t)
    return (t.year * 12 + t.month).to_numpy(np.int64)


def month_block_weights(t, finite, B, seed):
    """Calendar-month block bootstrap multiplicities (scoring spec §4).

    Parameters
    ----------
    t : DatetimeIndex (n,)
        The window's S2 instants.
    finite : (n,) bool
        Instants with a finite gauge value; the months drawn are the months
        present among them, every other instant gets weight 0.
    B : int
        Number of draws.
    seed : int or sequence of int
        ``np.random.default_rng(seed)``.

    Returns
    -------
    (B, n) int64
        ``W[b, i]`` = how often instant i's month was drawn in draw b.
    """
    t = pd.DatetimeIndex(t)
    finite = np.asarray(finite, bool)
    if finite.shape != (len(t),):
        raise ValueError(f"finite has shape {finite.shape}, expected ({len(t)},)")
    B = int(B)
    W = np.zeros((max(B, 0), len(t)), np.int64)
    if B <= 0 or not finite.any():
        return W
    lab = month_labels(t)[finite]
    um, inv = np.unique(lab, return_inverse=True)
    cols = np.flatnonzero(finite)
    rng = np.random.default_rng(seed)
    n_m = len(um)
    for b in range(B):
        draw = rng.integers(0, n_m, n_m)
        W[b, cols] = np.bincount(draw, minlength=n_m)[inv]
    return W


def weighted_median(values, W):
    """The median of each replicated sample: row b of ``W`` repeats value i
    ``W[b, i]`` times; the result equals ``np.median`` of that sample
    (the mean of the two middle values for an even count).

    Parameters
    ----------
    values : (n,) float
    W : (B, n) non-negative integer weights

    Returns
    -------
    (B,) float64 (NaN where a row has no weight)
    """
    v = np.asarray(values, np.float64)
    W = np.atleast_2d(np.asarray(W))
    order = np.argsort(v, kind="stable")
    vs = v[order]
    Wc = np.cumsum(W[:, order], axis=1)
    tot = Wc[:, -1] if Wc.shape[1] else np.zeros(W.shape[0])
    out = np.full(W.shape[0], np.nan)
    ok = tot > 0
    if not ok.any():
        return out
    lo = (tot[ok] - 1) // 2
    hi = tot[ok] // 2
    Wk = Wc[ok]
    jlo = (Wk <= lo[:, None]).sum(axis=1)
    jhi = (Wk <= hi[:, None]).sum(axis=1)
    out[ok] = 0.5 * (vs[jlo] + vs[jhi])
    return out


def weighted_mcrms(D, W):
    """Weighted median-centred RMS of each row of ``D`` for each weight row.

    ``sqrt(sum_i w_i (d_i - med_w d)^2 / sum_i w_i)``, the statistic E of
    ``truth.e_rows`` on the replicated sample (``med_w`` =
    :func:`weighted_median`). With unit weights it equals ``truth.e_rows`` to
    rounding (about 1e-15 m): the squares are summed as
    ``sum w d^2 - 2 m sum w d + m^2 sum w`` on d centred by its median.

    Parameters
    ----------
    D : (R, n) or (n,) float
        Rows of differences d_i (finite).
    W : (B, n) non-negative weights

    Returns
    -------
    (B, R) float64
    """
    D = np.atleast_2d(np.asarray(D, np.float64))
    W = np.atleast_2d(np.asarray(W))
    if D.shape[1] != W.shape[1]:
        raise ValueError(f"D has {D.shape[1]} columns, W {W.shape[1]}")
    Wf = W.astype(np.float64)
    tot = Wf.sum(axis=1)
    out = np.full((W.shape[0], D.shape[0]), np.nan)
    ok = tot > 0
    if not ok.any() or not D.shape[1]:
        return out
    Wk = Wf[ok]
    tk = tot[ok]
    for r in range(D.shape[0]):
        # E is shift-invariant: centre d first so that the expansion
        # sum w (d - m)^2 = sum w d^2 - 2 m sum w d + m^2 sum w is well
        # conditioned (two BLAS products instead of a B x n residual array)
        d = D[r] - np.median(D[r])
        med = weighted_median(d, W)[ok]
        s1 = Wk @ d
        s2 = Wk @ (d * d)
        var = (s2 - 2.0 * med * s1 + med * med * tk) / tk
        out[ok, r] = np.sqrt(np.maximum(var, 0.0))
    return out


def percentile_ci(draws, q=None):
    """``(lo, hi)`` percentiles of the finite draws (NaN when fewer than half
    of them are finite or there are none)."""
    q = INFERENCE_V2["boot"]["ci_percentiles"] if q is None else q
    d = np.asarray(draws, np.float64).ravel()
    fin = d[np.isfinite(d)]
    if not len(fin) or len(fin) < 0.5 * len(d):
        return float("nan"), float("nan")
    lo, hi = np.percentile(fin, q)
    return float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
#  exact sign-flip test and Holm
# ─────────────────────────────────────────────────────────────────────────────

def _finite_by_site(d_by_site):
    out = {}
    for site, d in d_by_site.items():
        a = np.asarray(d, np.float64).ravel()
        a = a[np.isfinite(a)]
        if len(a):
            out[str(site)] = a
    return out


def site_weighted_stat(d_by_site):
    """The L-P1 statistic: the mean over sites of the site-mean d."""
    fs = _finite_by_site(d_by_site)
    if not fs:
        return float("nan")
    return float(np.mean([a.mean() for a in fs.values()]))


def sign_flip(d_by_site, tol=None, max_n=None):
    """Exact two-sided sign-flip test of the site-weighted mean (§6.5).

    Parameters
    ----------
    d_by_site : dict {site: array of location differences d_l}
        NaN differences are dropped; a site without a finite d drops out of
        the mean over sites.
    tol : float
        |S*| >= |S| - tol counts as at least as extreme (default 1e-12).
    max_n : int
        Refuse more locations than this (2^n sign vectors are enumerated).

    Returns
    -------
    dict
        stat (S), p, n (locations), k_zero, p_min = min(1, 2 / 2^(n - k_zero)),
        n_sites, site_means {site: mean d}, n_by_site.
    """
    tol = INFERENCE_V2["sign_flip"]["tie_tol"] if tol is None else float(tol)
    max_n = INFERENCE_V2["sign_flip"]["max_n"] if max_n is None else int(max_n)
    fs = _finite_by_site(d_by_site)
    nan = float("nan")
    out = {"stat": nan, "p": nan, "n": 0, "k_zero": 0, "p_min": nan, "n_sites": len(fs),
           "site_means": {s: float(a.mean()) for s, a in fs.items()},
           "n_by_site": {s: int(len(a)) for s, a in fs.items()}}
    if not fs:
        return out
    d = np.concatenate(list(fs.values()))
    w = np.concatenate([np.full(len(a), 1.0 / (len(fs) * len(a))) for a in fs.values()])
    n = int(len(d))
    if n > max_n:
        raise ValueError(f"sign_flip: {n} locations exceed max_n = {max_n} (2^n vectors)")
    S = site_weighted_stat(fs)
    wd = w * d
    total = 1 << n
    chunk = int(INFERENCE_V2["sign_flip"]["chunk"])
    bits = np.arange(n, dtype=np.int64)
    count = 0
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
        signs = 1.0 - 2.0 * ((idx[:, None] >> bits[None, :]) & 1)
        Sstar = signs @ wd
        count += int(np.count_nonzero(np.abs(Sstar) >= abs(S) - tol))
    k_zero = int(np.count_nonzero(d == 0.0))
    out.update(stat=float(S), p=count / total, n=n, k_zero=k_zero,
               p_min=float(min(1.0, 2.0 / 2.0 ** (n - k_zero))))
    return out


def holm(pvals, alpha=None):
    """Holm's step-down procedure (Holm 1979) over one family.

    Parameters
    ----------
    pvals : dict {name: p}
        A NaN p (no data) counts as 1 in the adjustment: the family keeps its
        frozen size and that member never rejects.
    alpha : float
        Default FROZEN inference.holm_alpha.

    Returns
    -------
    dict {name: {"p", "p_adj", "reject", "threshold", "rank"}}
        ``p_adj`` = the Holm-adjusted p (running max of (m - j) p_(j),
        capped at 1); ``reject`` = step-down rejection at ``alpha``
        (equivalently p_adj <= alpha); ``threshold`` = alpha / (m - j) of the
        member's rank j (0-based).
    """
    alpha = ALPHA if alpha is None else float(alpha)
    names = list(pvals)
    p = np.array([float(pvals[k]) for k in names], np.float64)
    pa = np.where(np.isfinite(p), p, 1.0)
    m = len(names)
    order = np.argsort(pa, kind="stable")
    adj = np.empty(m)
    rej = np.zeros(m, bool)
    thr = np.empty(m)
    running = 0.0
    going = True
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pa[i]))
        adj[i] = running
        thr[i] = alpha / (m - rank)
        if going and pa[i] <= thr[i] and np.isfinite(p[i]):
            rej[i] = True
        else:
            going = False
    return {k: {"p": float(p[i]), "p_adj": float(adj[i]), "reject": bool(rej[i]),
                "threshold": float(thr[i]), "rank": int(np.flatnonzero(order == i)[0])}
            for i, k in enumerate(names)}


# ─────────────────────────────────────────────────────────────────────────────
#  family L-P1
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_COLUMNS = ("truth", "comparison", "n", "k_zero", "S", "p", "p_min", "holm_p", "reject",
                  "site_signs", "agree_sites", "confirmatory", "label")


def _finite_mean(x):
    """The mean of the finite values (NaN only when none is finite): one
    gauge without a value does not drop its location (the Terneuzen pair)."""
    v = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), np.float64)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def location_abs_e(points):
    """Location-level |e| (the mean over the location's gauges with a
    finite value, scoring spec §1, as the summaries): a DataFrame with site,
    window, location, product, truth, abs_e, e."""
    df = pd.DataFrame(points)
    if df.empty:
        return pd.DataFrame(columns=["site", "window", "location", "product", "truth",
                                     "abs_e", "e"])
    keys = [k for k in ("site", "window", "location", "product", "truth") if k in df.columns]
    g = df.groupby(keys, sort=True, dropna=False)
    out = g.agg(abs_e=("abs_e", _finite_mean), e=("e", _finite_mean)).reset_index()
    return out


def _sign(x):
    if not np.isfinite(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def family_lp1(points, truth, comparisons, judged_sites, *, marea=None, alpha=None,
               min_agree=None, verdict=True, label=None):
    """Family L-P1 for one truth (§6.5 item 1).

    Parameters
    ----------
    points : DataFrame or records
        Judge-level (or location-level) rows with site, location, product,
        truth, abs_e (and window). Pass ONE window per site: the caller
        selects W3 (confirmatory) or each site's available window
        (descriptive); a site with several windows is refused.
    truth : str
        "a", "ab" or "b".
    comparisons : sequence of product ids
        The family members X (MAREA is ``marea``, default "L01").
    judged_sites : sequence of str
        The sites the statistic averages over (and site agreement counts).
    verdict : bool
        False: the descriptive version (pre-lock or mixed windows) — p and
        Holm p are computed, ``reject`` and ``confirmatory`` are left empty
        (no verdict is issued).
    label : str
        The label column.

    Returns
    -------
    DataFrame with :data:`FAMILY_COLUMNS`, one row per comparison.
    """
    lp = INFERENCE_V2["lp1"]
    marea = lp["marea"] if marea is None else marea
    min_agree = lp["min_agree_sites"] if min_agree is None else int(min_agree)
    df = pd.DataFrame(points)
    cols = list(FAMILY_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[df["truth"] == truth]
    if "window" in df.columns:
        per_site = df.groupby("site")["window"].nunique()
        many = per_site[per_site > 1]
        if len(many):
            raise ValueError(f"family_lp1: several windows at {list(many.index)}; pass one "
                             "window per site")
    loc = location_abs_e(df)
    rows, stats = [], {}
    for X in comparisons:
        d_by_site, windows = {}, {}
        for site in judged_sites:
            a = loc[(loc["site"] == site) & (loc["product"] == marea)].set_index("location")
            b = loc[(loc["site"] == site) & (loc["product"] == X)].set_index("location")
            common = sorted(set(a.index) & set(b.index))
            d = np.array([a.loc[c, "abs_e"] - b.loc[c, "abs_e"] for c in common], float)
            d = d[np.isfinite(d)]
            if len(d):
                d_by_site[site] = d
                if "window" in a.columns and len(a):
                    windows[site] = str(a["window"].iloc[0])
        sf = sign_flip(d_by_site)
        S = sf["stat"]
        signs = {s: _sign(m) for s, m in sf["site_means"].items()}
        agree = sum(1 for s in judged_sites if s in signs and signs[s] == _sign(S) != 0)
        site_signs = ";".join(f"{s}@{windows.get(s, '')}={signs[s]:+d}" if s in signs
                              else f"{s}=NA" for s in judged_sites)
        stats[X] = sf
        rows.append({"truth": truth, "comparison": X, "n": sf["n"], "k_zero": sf["k_zero"],
                     "S": S, "p": sf["p"], "p_min": sf["p_min"], "holm_p": np.nan,
                     "reject": None, "site_signs": site_signs, "agree_sites": int(agree),
                     "confirmatory": None, "label": label or ""})
    h = holm({r["comparison"]: r["p"] for r in rows}, alpha)
    for r in rows:
        hh = h[r["comparison"]]
        r["holm_p"] = hh["p_adj"]
        if verdict:
            r["reject"] = bool(hh["reject"])
            r["confirmatory"] = bool(hh["reject"] and r["agree_sites"] >= min_agree)
    return pd.DataFrame(rows, columns=cols)


def ranking_claims(family, truths=None):
    """The ranking rule (§6.5): a direction 'MAREA better' (S < 0) or 'MAREA
    worse' (S > 0) against X is claimed only if X is confirmatory with that
    sign under every one of ``truths`` (default a, ab, b).

    Returns
    -------
    dict {comparison: "MAREA better" | "MAREA worse" | None}
    """
    truths = INFERENCE_V2["lp1"]["truths"] if truths is None else list(truths)
    df = pd.DataFrame(family)
    out = {}
    if df.empty:
        return out
    for X, g in df.groupby("comparison", sort=False):
        claim = None
        ok = True
        sgn = None
        for t in truths:
            r = g[g["truth"] == t]
            c = r["confirmatory"].iloc[0] if len(r) == 1 else None
            if c is None or (isinstance(c, float) and np.isnan(c)) or not bool(c):
                ok = False
                break
            s = _sign(float(r["S"].iloc[0]))
            if s == 0 or (sgn is not None and s != sgn):
                ok = False
                break
            sgn = s
        if ok and sgn is not None:
            claim = "MAREA better" if sgn < 0 else "MAREA worse"
        out[X] = claim
    return out


def pooled_sign_test(points, truth, comparison, *, marea=None, label=None):
    """The pooled location sign-flip test (every location of every site in
    one sample, unweighted mean): description only, "pseudo-replicated"."""
    lp = INFERENCE_V2["lp1"]
    marea = lp["marea"] if marea is None else marea
    df = pd.DataFrame(points)
    if df.empty:
        d = np.array([])
    else:
        loc = location_abs_e(df[df["truth"] == truth])
        a = loc[loc["product"] == marea].set_index(["site", "location"])
        b = loc[loc["product"] == comparison].set_index(["site", "location"])
        common = sorted(set(a.index) & set(b.index))
        d = np.array([a.loc[c, "abs_e"] - b.loc[c, "abs_e"] for c in common], float)
    sf = sign_flip({"pooled": d})
    return {"truth": truth, "comparison": comparison, "n": sf["n"], "k_zero": sf["k_zero"],
            "S": sf["stat"], "p": sf["p"], "p_min": sf["p_min"], "holm_p": np.nan,
            "reject": None, "site_signs": "", "agree_sites": np.nan, "confirmatory": None,
            "label": label or lp["pooled_label"]}


# ─────────────────────────────────────────────────────────────────────────────
#  reliability and concordance
# ─────────────────────────────────────────────────────────────────────────────

def icc21(Y):
    """ICC(2,1) of Shrout & Fleiss (1979): two-way random effects, absolute
    agreement, single measure.

    ICC = (MSR - MSE) / (MSR + (k - 1) MSE + k (MSC - MSE) / n), with n
    targets (rows; here bands) and k raters (columns; here calendar years or
    orbits). Rows with a non-finite value are dropped; NaN with fewer than 2
    complete rows or 2 columns, or a zero denominator.
    """
    Y = np.asarray(Y, np.float64)
    if Y.ndim != 2:
        raise ValueError("icc21 needs a 2-D (targets x raters) array")
    Y = Y[np.isfinite(Y).all(axis=1)]
    n, k = Y.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = Y.mean()
    rows = Y.mean(axis=1)
    cols = Y.mean(axis=0)
    ssr = k * np.sum((rows - grand) ** 2)
    ssc = n * np.sum((cols - grand) ** 2)
    sst = np.sum((Y - grand) ** 2)
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float((msr - mse) / den)


def lin_ccc(x, y):
    """Lin's concordance correlation coefficient (population moments) on the
    finite pairs; NaN with fewer than 2 pairs or a zero denominator."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x, y = x[m], y[m]
    mx, my = x.mean(), y.mean()
    sxy = np.mean((x - mx) * (y - my))
    den = np.var(x) + np.var(y) + (mx - my) ** 2
    return float(2 * sxy / den) if den > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
#  attainability (§9.8)
# ─────────────────────────────────────────────────────────────────────────────

ATTAIN_FORMAT = "v2-attain-1"


def count_blocks(flat, shape, pixel_m, size_m):
    """Square blocks of ``size_m`` (anchored at the frame's first pixel)
    holding at least one of the frame flat indices ``flat``."""
    flat = np.asarray(flat, np.int64)
    if not len(flat):
        return 0
    H, W = int(shape[0]), int(shape[1])
    bs = max(1, int(round(float(size_m) / float(pixel_m))))
    r, c = flat // W, flat % W
    nbc = -(-W // bs)
    return int(len(np.unique((r // bs) * nbc + (c // bs))))


def bands_with_truth(flat, band_frame, min_px=None):
    """``(n_bands with >= min_px truth pixels, n_bands with any)`` from the
    package's band map (``geometry.npz`` band_frame, -1 outside bands)."""
    min_px = INFERENCE_V2["attain"]["min_truth_px_band"] if min_px is None else int(min_px)
    flat = np.asarray(flat, np.int64)
    if not len(flat):
        return 0, 0
    b = np.asarray(band_frame).ravel()[flat]
    b = b[b >= 0].astype(np.int64)
    if not len(b):
        return 0, 0
    cnt = np.bincount(b)
    return int(np.sum(cnt >= min_px)), int(np.sum(cnt > 0))


def _w3_locations(doc, years):
    """Included locations at W3 of a truth document (window-year guard:
    none when a W3 year is missing from the package)."""
    from experiments.benchmark import judges, settings

    if doc is None:
        return [], "no lag truth document"
    miss = [y for y in settings.window("W3").years if years is not None and y not in years]
    if miss:
        return [], f"store years {miss} missing (window-year guard)"
    locs = set()
    for st in doc.get("stations", []):
        e = st.get("windows", {}).get("W3", {})
        if e.get("inclusion", {}).get("included"):
            locs.add(st.get("location") or judges.location_of(st["code"]))
    return sorted(locs), None


def _site_inputs(root, site, cubes_root=None):
    """D_truth, band map, shape, pixel size and years of a site's package."""
    from experiments.benchmark import package

    site2 = package.v2_site(site, root=root, cubes_root=cubes_root)
    pkg = package.V2Package(site2)
    sup = pkg.support
    has = bool(np.asarray(sup.get("has_d_truth", False)))
    d = np.asarray(sup["d_truth"], np.int64) if has else None
    return {"d_truth": d, "band_frame": np.asarray(pkg.band_frame), "shape": pkg.shape,
            "pixel_m": float(pkg.pixel_m), "years": list(pkg.years)}


def attain_report(root="results/validation_v2", sites=None, *, inputs=None, truth_docs=None,
                  cubes_root=None, write=True):
    """``pilot attain`` (§9.8): the attainability and power check, truth-free
    except for the judges' inclusion at W3 (which reads no truth value).

    Parameters
    ----------
    root : str
        The validation v2 root; ``<root>/attain.json`` is written.
    sites : sequence, optional
        Default every site of the benchmark.
    inputs : dict, optional
        ``{site: {"d_truth", "band_frame", "shape", "pixel_m", "years"}}``
        instead of reading the packages (tests).
    truth_docs : dict, optional
        ``{site: truth_lag_v2 document}`` instead of reading them.

    Returns
    -------
    dict (the JSON written)
    """
    from experiments.benchmark import product

    sites = list(config.SITES if sites is None else sites)
    at = INFERENCE_V2["attain"]
    lp = INFERENCE_V2["lp1"]
    fams = config.attainability()
    B = INFERENCE_V2["boot"]["B_primary"]
    boot_ok = all(f["attainable"] for f in fams.values() if f["kind"] == "bootstrap")
    out = {"format": ATTAIN_FORMAT, "created_utc": product.utc_stamp(), "alpha": ALPHA,
           "B_primary": B, "B_descriptive": INFERENCE_V2["boot"]["B_descriptive"],
           "p_min_bootstrap": 2.0 / (B + 1), "families": fams,
           "bootstrap_families_attainable": bool(boot_ok),
           "sites": {}, "lag_families": {}, "rules": list(at["rules"]), "warnings": []}
    n_loc_total = 0
    per_site_locs = {}
    for site in sites:
        e = {"site": site}
        try:
            inp = (inputs or {}).get(site)
            if inp is None and inputs is None:
                inp = _site_inputs(root, site, cubes_root)
            if inp is None:
                raise FileNotFoundError("no inputs given")
        except Exception as exc:          # a missing package is reported, not fatal
            e["status"] = f"not available: {type(exc).__name__}: {exc}"
            out["warnings"].append(f"{site}: {e['status']}")
            inp = None
        years = None if inp is None else inp.get("years")
        if inp is not None:
            d = inp.get("d_truth")
            if d is None:
                e["status"] = "no D_truth yet (run truth <site>)"
                e["n_truth_px"] = 0
            else:
                d = np.asarray(d, np.int64)
                e["status"] = "ok"
                e["n_truth_px"] = int(len(d))
                blocks = {}
                for km in at["block_sizes_km"]:
                    nb = count_blocks(d, inp["shape"], inp["pixel_m"], 1000.0 * km)
                    blocks[f"{km}km"] = {"n_blocks": nb,
                                         "testable": bool(nb >= at["min_blocks"])}
                e["blocks"] = blocks
                nb_min, nb_any = bands_with_truth(d, inp["band_frame"])
                e["bands_with_truth"] = nb_min
                e["bands_with_any_truth"] = nb_any
                e["p1_band_test_possible"] = bool(nb_min >= at["min_bands_p1"])
                e["expected_confirmatory"] = {k: v["testable"] for k, v in blocks.items()}
        doc = None
        if truth_docs is not None:
            doc = truth_docs.get(site)
        else:
            p = os.path.join(root, "truth", site, "truth_lag_v2.json")
            if os.path.exists(p):
                doc = product.read_json(p)
        if site in lp["judged_sites"]:
            locs, why = _w3_locations(doc, years)
            e["lp1_locations_W3"] = locs
            if why:
                e["lp1_note"] = why
            per_site_locs[site] = locs
            n_loc_total += len(locs)
        out["sites"][site] = e
    for name, ms in (("L-P1", at["lp1_m"]), ("L-P2", [at["lp2_m"]])):
        n = int(n_loc_total)
        p_min = float(min(1.0, 2.0 / 2.0 ** n)) if n else float("nan")
        fam = {"n_locations": n, "by_site": {s: len(v) for s, v in per_site_locs.items()},
               "p_min_exact": p_min, "by_m": {}}
        for m in ms:
            thr = [ALPHA / (m - j) for j in range(m)]
            fam["by_m"][str(m)] = {"holm_thresholds": thr,
                                   "attainable": bool(np.isfinite(p_min) and p_min < thr[0])}
        n_head = int(config.FROZEN["inference"]["families"]["headline_locations"])
        if n < n_head:
            fam["note"] = (f"{n} of the {n_head} headline locations are included at W3 with "
                           "complete store years")
        out["lag_families"][name] = fam
    if write:
        product.write_json(os.path.join(root, "attain.json"), out)
    return out
