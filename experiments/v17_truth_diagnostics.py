# -*- coding: utf-8 -*-
"""V17 - truth-side diagnostics of the published 10 m MAREA and NIDEM products.

Question (first author, 2026-10-20)
-----------------------------------
To decide whether MAREA can be published as a method in its own right: has it
an advantage over NIDEM on the TRUTH side? The published 10 m products are
compared from their stored files only (nothing is re-inverted), on their
common pixels at the four sites:

* ``MAREA``                      the published 10 m product (comparison
                                 products.npz ``z_marea``; Villaviciosa:
                                 external_products.npz ``MAREA``, the arrays v9
                                 scores);
* ``NIDEM``                      NIDEM as published: ITEM all-observations rule,
                                 experiments/v7_waterline_contour.py primary
                                 (NOT COMPUTABLE at the Ems, reported as such);
* ``NIDEM (on MAREA's record)``  v7's labelled sensitivity, NIDEM on exactly
                                 the scenes MAREA scored;
* ``no delay``                   MAREA's inversion with the lag set to zero: a
                                 REFERENCE row only (analyses 1 and 3). It is
                                 not part of the common set and has no
                                 uncertainty; it is scored on the part of the
                                 set where it has an estimate (n reported).

Three analyses
--------------
1. Bias (median of product - truth) and the published median-centred RMSE
   (with slope, r and the total RMSE) by along-channel 3-km band (MAREA's band
   map: products.npz / marea.npz ``band``, centres and applied lags from the
   production result.json; band -1 = pixels without a mouth distance, to
   which production applies lag 0, reported as their own row) and by
   elevation class: the lower, middle and upper third (equal width) of the
   truth's range over the common set. The classes are read from the truth,
   never from a product, so no product selects its own pixels. Per class the
   share of estimates at each method's ceiling or floor is also given (MAREA
   and no delay: within half a step of the inversion grid's ends, the
   record's zero-lag level range; NIDEM: at or beyond its highest / lowest
   interval height Z_j), because the upper third is bounded by them.
2. Calibration of each method's own per-pixel uncertainty: MAREA ``sigma``
   (marea.npz; the width of the fitted wet/dry transition, from the SG_GRID of
   pyintertidal.marea) and NIDEM ``uncertainty_keep`` (nidem.npz; the TIN of
   the interval tide-height SDs U_j). After removing the product's site bias
   b = median(p - t) on the calibration set, e = p - t - b:
   share of |e| <= sigma and <= 2 sigma (68.3 / 95.4 % for calibrated Gaussian
   errors); Spearman rho(sigma, |e|), also on v9's thinned subset (one unit per
   100 m block, validation_grid.thin_mask); k68 = the 68.27th percentile of
   |e| / sigma (the factor sigma would need to cover 68 % of the errors); the
   RMS of e / sigma; ENCE (Levi et al. 2019, mean over bins of
   |RMV - RMSE| / RMV); and a reliability curve: per sigma bin (one bin per
   distinct value when there are at most 12, i.e. MAREA's sigma grid; else 10
   quantile bins, ties kept together) the 68.27th percentile of |e| and the
   RMSE of e against the bin's sigma. The figure draws the percentile (robust
   to the heavy tails, and on the 1:1 line exactly when 68 % of the bin is
   within its sigma); the CSV holds both, and the slope (and intercept) of
   the per-bin q68 against sigma (``reliability_slope``): sigma is called
   calibrated only where that slope lies in 0.8-1.25. Neither uncertainty is
   a standard error of z: MAREA's sigma is the fitted transition width (one
   of the SG_GRID values, it does not shrink with more observations), NIDEM's
   U the tide-height SD of an interval. The truth has its own error (survey
   error, the 20 m truth cell against a 10 m pixel, bed change between survey
   and images; at Villaviciosa an RTK point against its pixel), which counts
   most against the smaller stated uncertainty; so the shares and k68 are
   also given against s_eff = sqrt(sigma^2 + sigma_t^2) for sigma_t = 0,
   0.05, 0.10, 0.15 m and for the site's own sigma_t: at Villaviciosa the
   pooled within-pixel SD of the RTK heights (pixels holding >= 2 points),
   at the Dutch sites 0.10 m ASSUMED (not estimable from one survey).
3. The total (uncentred) RMSE sqrt(mean((p - t)^2)) next to the published
   median-centred RMSE and the bias, on ``all`` (each product on every unit it
   resolves: the published 'all' rows), ``common``, ``pair_published`` (MAREA &
   NIDEM) and ``pair_same_scenes`` (MAREA & NIDEM on MAREA's record). Both
   methods sit on the same boundary levels, so the difference of their total
   RMSEs is a method effect; the absolute total RMSE also carries the vertical
   datum offset between the boundary's datum and the truth's, common to all
   products.
4. ``bootstrap``: paired cluster bootstrap of the differences MAREA - NIDEM,
   MAREA - NIDEM on MAREA's record and MAREA - no delay (total RMSE, centred
   RMSE, |bias|, bias; within 1 sigma for the pairs with uncertainties) on
   the common units, from <site>_units.npz (experiments/paired_bootstrap.py):
   1 km blocks of the grid at the Dutch sites, the 120 pixels holding the RTK
   points at Villaviciosa (too few clusters for more than an indicative
   interval), 2,000 resamples, a fixed seed, 95 % percentile intervals.

Sets
----
common (C) = finite truth & MAREA & every computable NIDEM (both NIDEM runs
where both exist; at the Ems MAREA & NIDEM on MAREA's record).
calibration set = C & every compared product's uncertainty finite and > 0.

Truth and scoring, exactly as v9_method_comparison
--------------------------------------------------
Dutch sites: Rijkswaterstaat Vaklodingen through v1.truth_grid / sample_truth
at the keep pixel centres, with the production range filter (comparison
products.npz z_plain range +- 0.25 m): v14.dutch_truth, the v9 code;
median-centred RMSE, slope, r and bias by v7.vscore (None below 30 px).
Villaviciosa: the RTK-GNSS development split through
pyintertidal.rtk.load_rtk() with its default arguments, heights in the
national datum (Alicante) as in the v9 tables, every product sampled at the
points' pixels; pyintertidal.elevation_validation.ols_against_truth (None
below 20 points). The total RMSE and the calibration use the same finite
pairs.

Gates (before any table; a failed gate stops the site, ``--soft-gates``
records it in the JSON and goes on)
------------------------------------------------------------------------
* ``published_all_rows``: every product scored on 'all' reproduces its 'all'
  row of results/method_comparison_same_scenes/<site>.csv (n exact; RMSE,
  slope, r and bias within 1e-9): the arrays and the truth are the published
  ones;
* ``published_common_subsets``: v9's common_published / common_same_scenes /
  common4 rows, rebuilt with v9.same_scene_rows, are reproduced too;
* ``sigma_belongs_to_scored_marea``: the z stored next to MAREA's sigma
  (marea.npz) is the scored MAREA on every unit with truth (same valid set,
  |dz| <= 1e-6 m);
* ``same_scenes_record``: the sensitivity NIDEM's scenes are exactly MAREA's
  scored record;
* ``band_map`` : products.npz band == marea.npz band (Dutch sites) and every
  band of the common set but -1 (no mouth distance, lag 0 in production) has
  a centre in result.json;
* ``record_dates`` (Dutch sites): products.npz record_dates == result.json
  record_dates.

Outputs (new files only, under ``--out``, default results/truth_diagnostics)
------------------------------------------------------------------------------
run:       <site>_by_band.csv, <site>_by_class.csv, <site>_calibration.csv,
           <site>_reliability.csv, <site>_totals.csv, <site>_settings.json,
           <site>_units.npz (every unit with truth: truth, products, sigmas,
           band, class, row, col, the common and calibration masks)
bootstrap: <site>_bootstrap.csv, <site>_bootstrap.json
report:    truth_{by_band,by_class,calibration,reliability,totals,gates,bootstrap}.csv,
           truth_diagnostics.md (English), truth_diagnostics.es.md (Spanish),
           fig1_bias_rmse_by_band_class_{es,en}.{png,pdf},
           fig2_uncertainty_calibration_{es,en}.{png,pdf},
           fig3_total_vs_centred_rmse_{es,en}.{png,pdf}

Run from the repository root (one process per site, the bootstrap, then the
report):
    python -X utf8 -m experiments.v17_truth_diagnostics run villaviciosa
    python -X utf8 -m experiments.v17_truth_diagnostics run escalda
    python -X utf8 -m experiments.v17_truth_diagnostics bootstrap --sites villaviciosa escalda
    python -X utf8 -m experiments.v17_truth_diagnostics report
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # BLAS threads per process (one process per site beside other jobs); a
    # value already in the environment wins
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402
from experiments import v9_method_comparison as v9  # noqa: E402
from experiments import paired_bootstrap as pbs  # noqa: E402
from experiments.validation_grid import thin_mask  # noqa: E402

OUT = "results/truth_diagnostics"
#: v9's product names (the published CSVs use them)
SENS = v9.SENS                                   # "NIDEM (on MAREA's record)"
SENS_DIR = "sensitivity_marea_record"            # v7.SENSITIVITY_TAG
COMPARED = ("MAREA", "NIDEM", SENS)
REFERENCE = "no delay"
PRODUCTS = COMPARED + (REFERENCE,)
#: v9's minimum number of scored units (ols_against_truth 20, vscore 30)
MIN_N = {"rtk": 20, "vaklodingen": 30}
TOL_SCORE = 1e-9
TOL_Z = 1e-6
#: P(|Z| <= 1), P(|Z| <= 2) for a standard normal Z
GAUSS_1 = 0.6826894921370859
GAUSS_2 = 0.9544997361036416
MAX_DISTINCT_BINS = 12
N_QBINS = 10
#: the truth's own error added to each stated uncertainty (m): the sweep
SIGMA_TS = (0.0, 0.05, 0.10, 0.15)
#: the Dutch sites' sigma_t: ASSUMED (Vaklodingen survey error, the 20 m cell
#: against a 10 m pixel, bed change between survey and images; one survey per
#: cell, so not estimable from the data)
SIGMA_T_DUTCH = 0.10
#: the per-bin 68th percentile of |e| follows sigma when the slope of q68 on
#: sigma over the scored bins lies here
RELIABILITY_OK = (0.8, 1.25)
#: band of the pixels without a mouth distance (production applies lag 0)
NO_MOUTH_BAND = -1
CLASS_KEYS = ("lower", "middle", "upper")
SUBSETS = ("all", "common", "pair_published", "pair_same_scenes")
SITE_ORDER = v14.SITE_ORDER
PAL = v14.PAL
FIG_STEMS = {"bands": "fig1_bias_rmse_by_band_class", "calibration": "fig2_uncertainty_calibration",
             "totals": "fig3_total_vs_centred_rmse"}

log = v14.log


# ─────────────────────────────────────────────────────────────────────────────
#  metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(kind, truth, p, sel=None):
    """The published metrics of one product (v14.score: v7.vscore for
    'vaklodingen', ols_against_truth for 'rtk'; NaN below MIN_N) plus, of
    d = p - t, the total RMSE sqrt(mean(d^2)), the mean and the SD (ddof 0),
    so that total^2 = mean^2 + SD^2 exactly, on the units ``sel`` (a boolean
    mask or indices; all when None) where both are finite. Selecting keeps
    the order, so this equals scoring ``np.where(sel, p, nan)`` as v9 does.

    The published 'centred' RMSE subtracts the median of p from p and the
    median of t from t (v7.vscore, ols_against_truth): it is not the RMSE
    about the bias, and it can exceed the total RMSE when the slope departs
    from 1."""
    t = np.asarray(truth, float)
    v = np.asarray(p, float)
    if sel is not None:
        t, v = t[sel], v[sel]
    s = v14.score(kind, t, v)
    fin = np.isfinite(v) & np.isfinite(t)
    out = {"n": int(fin.sum()), "bias_m": s["bias"], "rmse_centred_m": s["rmse"],
           "rmse_total_m": np.nan, "mean_diff_m": np.nan, "sd_diff_m": np.nan, "slope": s["slope"], "r": s["r"]}
    if np.isfinite(s["rmse"]):
        d = v[fin] - t[fin]
        out["rmse_total_m"] = float(np.sqrt(np.mean(d * d)))
        out["mean_diff_m"] = float(np.mean(d))
        out["sd_diff_m"] = float(np.std(d))
    return out


def build_sets(S):
    """The scored sets of a site (boolean masks over its units) and the
    compared products present / with an uncertainty."""
    t, P = S["truth"], S["prods"]
    have = [p for p in COMPARED if p in P]
    tf = np.isfinite(t)
    common = tf.copy()
    for p in have:
        common &= np.isfinite(P[p])
    sets = {"all": tf, "common": common}
    if "NIDEM" in P and "MAREA" in P:
        sets["pair_published"] = tf & np.isfinite(P["MAREA"]) & np.isfinite(P["NIDEM"])
    if SENS in P and "MAREA" in P:
        sets["pair_same_scenes"] = tf & np.isfinite(P["MAREA"]) & np.isfinite(P[SENS])
    sig = [p for p in have if S["sigma"].get(p) is not None]
    cal = common.copy()
    for p in sig:
        s = np.asarray(S["sigma"][p], float)
        with np.errstate(invalid="ignore"):
            cal &= np.isfinite(s) & (s > 0)
    sets["calibration"] = cal
    return sets, have, sig


def truth_classes(truth, sel):
    """Class 0/1/2 (lower/middle/upper third, equal width, of the truth's
    range over ``sel``), -1 outside ``sel``; and the edges [lo, e1, e2, hi]."""
    t = np.asarray(truth, float)
    sel = np.asarray(sel, bool) & np.isfinite(t)
    cls = np.full(len(t), -1, np.int8)
    if not sel.any():
        return cls, [np.nan] * 4
    lo, hi = float(t[sel].min()), float(t[sel].max())
    e1, e2 = lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0
    tt = np.where(sel, t, np.nan)
    with np.errstate(invalid="ignore"):
        cls[sel & (tt < e1)] = 0
        cls[sel & (tt >= e1) & (tt < e2)] = 1
        cls[sel & (tt >= e2)] = 2
    return cls, [lo, e1, e2, hi]


def spearman(a, b):
    """Spearman rank correlation (average ranks for ties); NaN when either
    side is constant or there are fewer than 3 pairs."""
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = spearmanr(a, b)
    return float(getattr(r, "statistic", r[0]))


def sigma_bins(s):
    """Bin index of every sigma and the bins' (lo, hi): one bin per distinct
    value when there are at most MAX_DISTINCT_BINS (MAREA's sigma grid), else
    N_QBINS quantile bins (equal values never split; a value on an inner edge
    goes to the upper bin). Returns (index, bounds, rule)."""
    s = np.asarray(s, float)
    u = np.unique(s)
    if len(u) <= MAX_DISTINCT_BINS:
        return np.searchsorted(u, s), [(float(x), float(x)) for x in u], "one bin per distinct value"
    q = np.unique(np.quantile(s, np.linspace(0.0, 1.0, N_QBINS + 1)))
    idx = np.clip(np.searchsorted(q, s, side="right") - 1, 0, len(q) - 2)
    return idx, [(float(q[k]), float(q[k + 1])) for k in range(len(q) - 1)], \
        f"{len(q) - 1} quantile bins"


def t_key(sigma_t):
    """Column suffix of a truth-error value: 0.05 m -> 't005'."""
    return f"t{int(round(float(sigma_t) * 100)):03d}"


def reliability_verdict(slope, n_bins):
    """Words for the per-bin q68-on-sigma slope."""
    if n_bins < 2 or not np.isfinite(slope):
        return "fewer than 2 scored sigma bins: shape not tested"
    lo, hi = RELIABILITY_OK
    if lo <= slope <= hi:
        return f"q68 follows sigma (slope {slope:.2f} in {lo}-{hi})"
    return f"q68 does not follow sigma (slope {slope:.2f}, outside {lo}-{hi})"


def calibrate(kind, truth, p, sigma, sel, thin=None, sigma_ts=SIGMA_TS, sigma_t_site=None):
    """Calibration of one product's own uncertainty on the units ``sel``
    (truth, p and sigma > 0 finite there). The shares within 1 and 2 sigma
    and k68 are also given against s_eff = sqrt(sigma^2 + sigma_t^2) for
    every truth error of ``sigma_ts`` (columns ..._t005 etc.) and for the
    site's ``sigma_t_site`` (columns ..._tsite). Returns (summary dict,
    reliability rows)."""
    t = np.asarray(truth, float)
    v = np.asarray(p, float)
    s_all = np.asarray(sigma, float)
    with np.errstate(invalid="ignore"):
        m = np.asarray(sel, bool) & np.isfinite(v) & np.isfinite(t) & np.isfinite(s_all) & (s_all > 0)
    n = int(m.sum())
    out = {"n": n, "bias_removed_m": np.nan, "sigma_median_m": np.nan, "sigma_rmv_m": np.nan,
           "rmse_e_m": np.nan, "q68_abs_e_m": np.nan, "within_1sigma": np.nan, "within_2sigma": np.nan,
           "spearman_rho": np.nan, "spearman_rho_thinned": np.nan, "n_thinned": 0, "k68": np.nan,
           "z_rms": np.nan, "ence": np.nan, "n_bins_scored": 0, "bin_rule": "",
           "reliability_slope": np.nan, "reliability_intercept_m": np.nan, "reliability_verdict": ""}
    site_t = (float(sigma_t_site) if sigma_t_site is not None and np.isfinite(float(sigma_t_site)) else None)
    for st in tuple(sigma_ts) + (("site",) if site_t is not None else ()):
        k_ = "tsite" if st == "site" else t_key(st)
        out.update({f"within_1sigma_{k_}": np.nan, f"within_2sigma_{k_}": np.nan, f"k68_{k_}": np.nan})
    out["sigma_t_site_m"] = site_t if site_t is not None else np.nan
    if n < MIN_N[kind]:
        return out, []
    d_all = v - t
    b = float(np.median(d_all[m]))
    e = d_all[m] - b
    s = s_all[m]
    ae = np.abs(e)
    z = ae / s
    out.update(bias_removed_m=b, sigma_median_m=float(np.median(s)),
               sigma_rmv_m=float(np.sqrt(np.mean(s * s))), rmse_e_m=float(np.sqrt(np.mean(e * e))),
               q68_abs_e_m=float(np.quantile(ae, GAUSS_1)), within_1sigma=float(np.mean(ae <= s)),
               within_2sigma=float(np.mean(ae <= 2.0 * s)), spearman_rho=spearman(s, ae),
               k68=float(np.quantile(z, GAUSS_1)), z_rms=float(np.sqrt(np.mean(z * z))))
    for st in tuple(sigma_ts) + (("site",) if site_t is not None else ()):
        k_ = "tsite" if st == "site" else t_key(st)
        stv = site_t if st == "site" else float(st)
        se = np.sqrt(s * s + stv * stv)
        out.update({f"within_1sigma_{k_}": float(np.mean(ae <= se)),
                    f"within_2sigma_{k_}": float(np.mean(ae <= 2.0 * se)),
                    f"k68_{k_}": float(np.quantile(ae / se, GAUSS_1))})
    if thin is not None:
        mt = m & np.asarray(thin, bool)
        out["n_thinned"] = int(mt.sum())
        if mt.sum() >= MIN_N[kind]:
            out["spearman_rho_thinned"] = spearman(s_all[mt], np.abs(d_all[mt] - b))
    idx, bounds, rule = sigma_bins(s)
    out["bin_rule"] = rule
    rows, ence = [], []
    for k, (lo, hi) in enumerate(bounds):
        q = idx == k
        nk = int(q.sum())
        if not nk:
            continue
        sk, ek = s[q], e[q]
        rmv = float(np.sqrt(np.mean(sk * sk)))
        rmse = float(np.sqrt(np.mean(ek * ek)))
        rows.append({"bin": k, "sigma_lo_m": lo, "sigma_hi_m": hi, "n": nk,
                     "sigma_median_m": float(np.median(sk)), "rmv_m": rmv, "rmse_e_m": rmse,
                     "q68_abs_e_m": float(np.quantile(np.abs(ek), GAUSS_1)),
                     "within_1sigma": float(np.mean(np.abs(ek) <= sk)), "scored": nk >= MIN_N[kind]})
        if nk >= MIN_N[kind]:
            ence.append(abs(rmv - rmse) / rmv)
    out["n_bins_scored"] = len(ence)
    out["ence"] = float(np.mean(ence)) if ence else np.nan
    # does the per-bin q68 follow sigma? (slope of q68 on sigma over the scored bins)
    sc = [r for r in rows if r["scored"]]
    xs = np.array([r["sigma_median_m"] for r in sc], float)
    n_eff = len(sc) if (len(sc) >= 2 and float(np.ptp(xs)) > 0) else 0
    if n_eff:
        slope, icpt = np.polyfit(xs, np.array([r["q68_abs_e_m"] for r in sc], float), 1)
        out["reliability_slope"], out["reliability_intercept_m"] = float(slope), float(icpt)
    out["reliability_verdict"] = reliability_verdict(out["reliability_slope"], n_eff)
    return out, rows


# ─────────────────────────────────────────────────────────────────────────────
#  the analyses on one site's units
# ─────────────────────────────────────────────────────────────────────────────

def role_of(p):
    return "reference (not in the common set)" if p == REFERENCE else "compared"


def clamp_shares(p, sel, clamp):
    """Shares of the finite estimates on ``sel`` at the method's ceiling
    (>= hi - tol) and floor (<= lo + tol); NaN without a clamp or an
    estimate."""
    v = np.asarray(p, float)
    q = np.asarray(sel, bool) & np.isfinite(v)
    if clamp is None or not q.any():
        return {"share_at_upper_clamp": np.nan, "share_at_lower_clamp": np.nan}
    x = v[q]
    return {"share_at_upper_clamp": float(np.mean(x >= clamp["hi"] - clamp["tol"])),
            "share_at_lower_clamp": float(np.mean(x <= clamp["lo"] + clamp["tol"]))}


def analyse(S):
    """Analyses 1-3 on the units of one site. ``S`` holds: site, kind, truth,
    prods {name: values}, sigma {name: values or None}, band (int, -1 none),
    s_km, centers, tau_band, thin, scenes, status. Returns (tables, info)."""
    import pandas as pd

    site, kind, t = S["site"], S["kind"], np.asarray(S["truth"], float)
    P = S["prods"]
    sets, have, sig = build_sets(S)
    C = sets["common"]
    nC = int(C.sum())
    if "MAREA" not in P or nC < MIN_N[kind]:
        raise RuntimeError(f"[{site}] the common set has {nC} units (< {MIN_N[kind]}): nothing to compare")
    scored = have + ([REFERENCE] if REFERENCE in P else [])

    # 1a. by along-channel band (MAREA's band map)
    band = np.asarray(S["band"], np.int64)
    centers = list(S.get("centers") or [])
    taus = list(S.get("tau_band") or [])
    s_km = np.asarray(S["s_km"], float) if S.get("s_km") is not None else np.full(len(t), np.nan)
    bands = sorted(set(band[C].tolist()))
    rows_b = []
    for p in scored:
        rows_b.append({"site": site, "product": p, "role": role_of(p), "band": "all", "s_km_centre": np.nan,
                       "s_km_median": np.nan, "tau_applied_min": np.nan, "n_set": nC,
                       **metrics(kind, t, P[p], C)})
        for k in bands:
            q = C & (band == k)
            ok = 0 <= k < len(centers)
            sk = s_km[q]
            sk = sk[np.isfinite(sk)]
            # band -1: no mouth distance, production's lag 0
            tau_k = (0.0 if k == NO_MOUTH_BAND else float(taus[k]) if 0 <= k < len(taus) else np.nan)
            rows_b.append({"site": site, "product": p, "role": role_of(p), "band": int(k),
                           "s_km_centre": float(centers[k]) if ok else np.nan,
                           "s_km_median": float(np.median(sk)) if len(sk) else np.nan,
                           "tau_applied_min": tau_k, "n_set": int(q.sum()), **metrics(kind, t, P[p], q)})

    # 1b. by elevation class (thirds of the truth's range on C), with the
    # share of each product's estimates at its ceiling / floor
    cls, edges = truth_classes(t, C)
    clamps = S.get("clamp") or {}
    rows_c = []
    for p in scored:
        rows_c.append({"site": site, "product": p, "role": role_of(p), "class": "all", "z_lo_m": edges[0],
                       "z_hi_m": edges[3], "n_set": nC, **metrics(kind, t, P[p], C),
                       **clamp_shares(P[p], C, clamps.get(p))})
        for k, key in enumerate(CLASS_KEYS):
            q = cls == k
            rows_c.append({"site": site, "product": p, "role": role_of(p), "class": key,
                           "z_lo_m": edges[k], "z_hi_m": edges[k + 1], "n_set": int(q.sum()),
                           **metrics(kind, t, P[p], q), **clamp_shares(P[p], q, clamps.get(p))})

    # 2. calibration of each method's own uncertainty
    cal = sets["calibration"]
    rows_k, rows_r = [], []
    for p in COMPARED:
        if p not in P:
            rows_k.append({"site": site, "product": p, "status": S["status"].get(p, "missing"), "n": 0})
            continue
        if p not in sig:
            rows_k.append({"site": site, "product": p, "status": "no uncertainty stored", "n": 0})
            continue
        summ, bins = calibrate(kind, t, P[p], S["sigma"][p], cal, S.get("thin"),
                               sigma_t_site=(S.get("truth_error") or {}).get("sigma_t_m"))
        rows_k.append({"site": site, "product": p, "status": "ok",
                       "sigma_source": S.get("sigma_source", {}).get(p, ""), "n_common": nC, **summ})
        for r in bins:
            rows_r.append({"site": site, "product": p, **r})

    # 3. total vs centred RMSE
    members = {"all": scored, "common": scored, "pair_published": ["MAREA", "NIDEM"],
               "pair_same_scenes": ["MAREA", SENS]}
    rows_t = []
    for sub in SUBSETS:
        if sub not in sets:
            continue
        for p in members[sub]:
            if p not in P:
                continue
            rows_t.append({"site": site, "product": p, "role": role_of(p), "subset": sub,
                           "scenes": S["scenes"].get(p), "status": "ok", "n_set": int(sets[sub].sum()),
                           **metrics(kind, t, P[p], sets[sub])})
    for p in COMPARED:
        if p not in P:
            rows_t.append({"site": site, "product": p, "role": role_of(p), "subset": "all",
                           "scenes": S["scenes"].get(p), "status": S["status"].get(p, "missing"), "n_set": 0,
                           "n": 0})

    tables = {k: pd.DataFrame(v) for k, v in (("by_band", rows_b), ("by_class", rows_c),
                                              ("calibration", rows_k), ("reliability", rows_r),
                                              ("totals", rows_t))}
    info = {"sets": {k: int(np.sum(v)) for k, v in sets.items()},
            "common_members": ["truth"] + have, "compared_present": have, "with_uncertainty": sig,
            "reference_on_common": (int(np.sum(C & np.isfinite(P[REFERENCE]))) if REFERENCE in P else None),
            "classes": {"edges_m": edges, "rule": "equal-width thirds of the truth's range [min, max] over the "
                                                  "common set; lower [lo, e1), middle [e1, e2), upper [e2, hi]",
                        "counts": [int(np.sum(cls == k)) for k in range(3)]},
            "bands": {"present_in_common": [int(b) for b in bands], "centres_km": centers,
                      "tau_applied_min": taus,
                      "n_common_without_mouth_distance": int(np.sum(C & (band == NO_MOUTH_BAND))),
                      "band_minus_1": "no mouth distance (s_km NaN): production applies lag 0"},
            "clamps": {p: v for p, v in clamps.items()},
            "truth_error": S.get("truth_error")}
    return tables, info


# ─────────────────────────────────────────────────────────────────────────────
#  inputs
# ─────────────────────────────────────────────────────────────────────────────

def site_inputs(site, overrides=None):
    """The stored files of a site, relative to the repository root (the
    server's layout, as v9 / v14 read them); ``overrides`` replaces any."""
    c = dict(v14.site_config(site))
    c["nidem_sensitivity"] = os.path.join(c["nidem"], SENS_DIR)
    c["external"] = v9.SITES["villaviciosa"]["products"] if site == "villaviciosa" else None
    c["published"] = f"results/method_comparison_same_scenes/{site}.csv"
    for k, v in (overrides or {}).items():
        if v:
            c[k] = v
    if overrides and overrides.get("nidem") and not overrides.get("nidem_sensitivity"):
        c["nidem_sensitivity"] = os.path.join(c["nidem"], SENS_DIR)
    return c


def load_nidem_run(path, keep):
    """A NIDEM run on the keep pixels: (filtered DEM, uncertainty, scenes,
    status, dates, interval heights Z_j). The DEM through v9.load_nidem (the
    primary product, the NOT COMPUTABLE verdict read first); the uncertainty
    (``uncertainty_keep``) and Z from the same nidem.npz."""
    v, n_sc, st = v9.load_nidem(path, keep)
    if v is None:
        return None, None, n_sc, st, None, None
    u, dates, Z = None, None, None
    with np.load(os.path.join(path, "nidem.npz"), allow_pickle=False) as z:
        if "uncertainty_keep" in z.files:
            u = z["uncertainty_keep"].astype(float)
        elif "uncertainty" in z.files:
            u = np.asarray(z["uncertainty"]).ravel()[keep].astype(float)
        if "dates" in z.files:
            dates = [str(d)[:10] for d in z["dates"]]
        if "Z" in z.files:
            Z = np.asarray(z["Z"], float)
    if u is not None and u.shape != v.shape:
        raise RuntimeError(f"{path}: uncertainty {u.shape} != DEM {v.shape}")
    return v, u, n_sc, st, dates, Z


def marea_clamp(res):
    """MAREA's ceiling and floor: the record's zero-lag level range
    (result.json rango_marea_m), the ends of invert_series' mu grid; an
    estimate within half a grid step of an end is at the clamp."""
    import inspect

    from pyintertidal import marea

    lo, hi = (float(x) for x in res["rango_marea_m"])
    mu = int(inspect.signature(marea.invert_series).parameters["mu_points"].default)
    step = (hi - lo) / (mu - 1)
    return {"lo": lo, "hi": hi, "tol": 0.5 * step,
            "rule": (f"MAREA's inversion grid np.linspace(lo, hi, {mu}) over the record's zero-lag level range "
                     f"(result.json rango_marea_m [{lo:.3f}, {hi:.3f}] m): at the clamp = within half a step "
                     f"({0.5 * step:.4f} m) of an end")}


def nidem_clamp(Z):
    """NIDEM's ceiling and floor: its highest and lowest interval height."""
    if Z is None or not np.isfinite(Z).any():
        return None
    lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    return {"lo": lo, "hi": hi, "tol": 1e-6,
            "rule": f"NIDEM's 9 interval heights Z_j span [{lo:.3f}, {hi:.3f}] m: at the clamp = at or beyond an end"}


def rtk_truth_error(truth, rows, cols, keep, W):
    """The RTK truth's own spread at the scale of a pixel: the pooled
    within-pixel SD of the heights of the points that share a pixel
    (pixels in keep holding >= 2 points with a height). Read from the truth
    only."""
    t = np.asarray(truth, float)
    pix = np.asarray(rows, np.int64) * int(W) + np.asarray(cols, np.int64)
    ok = np.isfinite(t) & np.isin(pix, np.asarray(keep, np.int64))
    u, inv, cnt = np.unique(pix[ok], return_inverse=True, return_counts=True)
    tt = t[ok]
    ss, dof = 0.0, 0
    for g in np.flatnonzero(cnt >= 2):
        x = tt[inv == g]
        ss += float(np.sum((x - x.mean()) ** 2))
        dof += len(x) - 1
    sd = float(np.sqrt(ss / dof)) if dof else np.nan
    return {"sigma_t_m": sd, "n_pixels_with_repeats": int(np.sum(cnt >= 2)),
            "n_points_in_them": int(np.sum(cnt[cnt >= 2])), "dof": int(dof),
            "source": ("estimated: pooled within-pixel SD of the RTK heights (pixels of the extraction holding "
                       ">= 2 development points)")}


def _agreement(a, b, sel=None):
    """Validity flips and max |a - b| where both are finite (on ``sel``)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if sel is not None:
        a, b = a[sel], b[sel]
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    return {"units": int(len(a)), "validity_flips": int(np.sum(fa != fb)),
            "max_abs_diff_m": float(np.max(np.abs(a[both] - b[both]))) if both.any() else None}


def _nidems(S, cfg, keep, on_units):
    """Both NIDEM runs into S (``on_units`` maps keep values to the units)."""
    for name, key in (("NIDEM", "nidem"), (SENS, "nidem_sensitivity")):
        v, u, n_sc, st, dates, Z = load_nidem_run(cfg[key], keep)
        S["meta"]["inputs"][f"nidem_{'primary' if name == 'NIDEM' else 'sensitivity'}"] = \
            v14.file_info(os.path.join(cfg[key], "nidem.npz"))
        S["status"][name] = st
        S["scenes"][name] = int(n_sc) if (v is not None and n_sc is not None) else st
        if v is None:
            S["meta"]["notes"].append(f"{name}: {st} ({cfg[key]})")
            continue
        S["prods"][name] = on_units(v)
        S["sigma"][name] = on_units(u) if u is not None else None
        S["sigma_source"][name] = f"{cfg[key]}/nidem.npz uncertainty_keep (TIN of the interval tide SDs U_j)"
        S["checks"][f"dates_{name}"] = dates
        cl = nidem_clamp(Z)
        if cl is not None:
            S["clamp"][name] = cl


def load_villaviciosa(cfg):
    """The RTK development points and the published products at their pixels,
    as v9.score_villaviciosa samples them."""
    from pyintertidal import rtk

    S = _empty_site("villaviciosa", "rtk")
    keep, (H, W) = v9.load_keep(cfg)
    S["meta"]["inputs"]["extract"] = {"path": cfg["extract"], "n_keep": int(len(keep)), "grid": [H, W]}
    with np.load(cfg["external"], allow_pickle=False) as E:
        ras = {"no delay": E["no_delay"].astype(float), "MAREA": E["MAREA"].astype(float)}
    for name, r in ras.items():
        if r.shape != (H, W):
            raise RuntimeError(f"{cfg['external']} {name}: {r.shape} != grid {(H, W)}")
    S["meta"]["inputs"]["external_products"] = v14.file_info(cfg["external"])
    mdir = cfg["product"]
    with np.load(os.path.join(mdir, "marea.npz"), allow_pickle=False) as M:
        if not np.array_equal(M["keep"], keep):
            raise RuntimeError(f"{mdir}/marea.npz: keep differs from the extraction")
        mz, msig = M["z"].astype(float), M["sigma"].astype(float)
        mband, ms = M["band"].astype(np.int64), M["s_km"].astype(float)
    res = json.load(open(os.path.join(mdir, "result.json"), encoding="utf-8"))
    S["meta"]["inputs"]["marea_npz"] = v14.file_info(os.path.join(mdir, "marea.npz"))
    S["meta"]["inputs"]["result_json"] = v14.file_info(os.path.join(mdir, "result.json"))

    dev = rtk.load_rtk()
    r_, c_ = np.asarray(dev["row"], np.int64), np.asarray(dev["col"], np.int64)
    if "elev_alicante" in dev:
        S["truth"], datum = np.asarray(dev["elev_alicante"], float), "Alicante (REDNAP orthometric)"
    else:
        S["truth"], datum = np.asarray(dev["elev"], float), "ETRS89 ellipsoidal"

    def on_units(vals, fill=np.nan, dtype=float):
        g = np.full(H * W, fill, dtype)
        g[keep] = vals
        return g.reshape(H, W)[r_, c_]

    S["prods"] = {name: r[r_, c_] for name, r in ras.items()}
    S["sigma"]["MAREA"] = on_units(msig)
    S["sigma_source"]["MAREA"] = f"{mdir}/marea.npz sigma (fitted transition width)"
    S["band"] = on_units(mband, fill=-1, dtype=np.int64)
    S["s_km"] = on_units(ms)
    S["centers"] = [float(x) for x in res.get("centros_km", [])]
    S["tau_band"] = [float(x) for x in res.get("tau_usado_min", [])]
    rec = [str(d)[:10] for d in res.get("record_dates", [])]
    S["scenes"].update({"MAREA": len(rec) or None, "no delay": len(rec) or None})
    S["status"].update({"MAREA": "ok", "no delay": "ok"})
    S["checks"]["marea_record_dates"] = rec
    S["checks"]["marea_z_units"] = on_units(mz)
    S["checks"]["marea_z_vs_scored_on_keep"] = _agreement(
        mz, ras["MAREA"].ravel()[keep])
    with np.errstate(invalid="ignore"):
        S["checks"]["marea_sigma_finite_where_z_finite"] = float(
            np.mean(np.isfinite(msig[np.isfinite(mz)]))) if np.isfinite(mz).any() else None
    if "rango_marea_m" in res:
        S["clamp"]["MAREA"] = S["clamp"]["no delay"] = marea_clamp(res)
    _nidems(S, cfg, keep, lambda v: on_units(v))
    S["rows"], S["cols"], S["pixel_m"], S["grid_w"] = r_, c_, 10.0, W
    S["thin"] = thin_mask(r_, c_, 10.0)
    S["truth_error"] = rtk_truth_error(S["truth"], r_, c_, keep, W)
    S["meta"]["truth"] = {"truth": "RTK-GNSS development split (pyintertidal.rtk.load_rtk(), default arguments)",
                          "datum": datum, "points_dev": int(len(S["truth"])),
                          "n_reserved_hidden": int(dev.get("n_reserved_hidden", -1)),
                          "points_in_keep": int(np.isin(r_ * W + c_, keep).sum()),
                          "unit": "RTK point"}
    return S


def load_dutch(site, cfg):
    """The keep pixels with their Vaklodingen truth (v14.dutch_truth: v1
    sampling and the production range filter, the v9 code) and the published
    products on them."""
    S = _empty_site(site, "vaklodingen")
    keep, (H, W) = v9.load_keep(cfg)
    S["meta"]["inputs"]["extract"] = {"path": cfg["extract"], "n_keep": int(len(keep)), "grid": [H, W]}
    cp = os.path.join(cfg["comparison"], "products.npz")
    with np.load(cp, allow_pickle=False) as P:
        if not np.array_equal(P["keep"], keep):
            raise RuntimeError(f"{cp}: keep differs from the extraction")
        z_plain, z_marea = P["z_plain"].astype(float), P["z_marea"].astype(float)
        band_p = P["band"].astype(np.int64)
        s_km = P["s_km"].astype(float)
        rec_p = [str(d)[:10] for d in P["record_dates"]] if "record_dates" in P.files else None
    S["meta"]["inputs"]["comparison_products"] = v14.file_info(cp)
    mdir = cfg["product"]
    with np.load(os.path.join(mdir, "marea.npz"), allow_pickle=False) as M:
        if not np.array_equal(M["keep"], keep):
            raise RuntimeError(f"{mdir}/marea.npz: keep differs from the extraction")
        mz, msig, band_m = M["z"].astype(float), M["sigma"].astype(float), M["band"].astype(np.int64)
    res = json.load(open(os.path.join(mdir, "result.json"), encoding="utf-8"))
    S["meta"]["inputs"]["marea_npz"] = v14.file_info(os.path.join(mdir, "marea.npz"))
    S["meta"]["inputs"]["result_json"] = v14.file_info(os.path.join(mdir, "result.json"))

    truth, tinfo = v14.dutch_truth(site, cfg, keep, H, W, z_plain)
    S["truth"] = np.asarray(truth, float)
    S["prods"] = {"no delay": z_plain, "MAREA": z_marea}
    S["sigma"]["MAREA"] = msig
    S["sigma_source"]["MAREA"] = f"{mdir}/marea.npz sigma (fitted transition width)"
    S["band"], S["s_km"] = band_p, s_km
    S["centers"] = [float(x) for x in res.get("centros_km", [])]
    S["tau_band"] = [float(x) for x in res.get("tau_usado_min", [])]
    rec = [str(d)[:10] for d in res.get("record_dates", [])]
    S["scenes"].update({"MAREA": len(rec_p or rec) or None, "no delay": len(rec_p or rec) or None})
    S["status"].update({"MAREA": "ok", "no delay": "ok"})
    S["checks"]["marea_record_dates"] = rec_p or rec
    S["checks"]["record_dates_products_vs_result"] = (None if rec_p is None else rec_p == rec)
    S["checks"]["band_products_vs_marea"] = bool(np.array_equal(band_p, band_m))
    S["checks"]["marea_z_units"] = mz
    S["checks"]["marea_z_vs_scored_on_keep"] = _agreement(mz, z_marea)
    with np.errstate(invalid="ignore"):
        S["checks"]["marea_sigma_finite_where_z_finite"] = float(
            np.mean(np.isfinite(msig[np.isfinite(mz)]))) if np.isfinite(mz).any() else None
    if "rango_marea_m" in res:
        S["clamp"]["MAREA"] = S["clamp"]["no delay"] = marea_clamp(res)
    _nidems(S, cfg, keep, lambda v: v)
    px_m = float(tinfo.get("pixel_m") or 10.0)
    rows, cols = keep // W, keep % W
    S["rows"], S["cols"], S["pixel_m"], S["grid_w"] = rows, cols, px_m, W
    S["thin"] = thin_mask(rows, cols, px_m)
    S["meta"]["truth"] = {**tinfo, "unit": f"{px_m:g} m pixel"}
    S["truth_error"] = {"sigma_t_m": SIGMA_T_DUTCH,
                        "source": ("ASSUMED: Vaklodingen survey error, the 20 m truth cell against a 10 m pixel and "
                                   "bed change between survey and images; one survey per cell, so not estimable "
                                   "from the data")}
    return S


def _empty_site(site, kind):
    return {"site": site, "kind": kind, "truth": None, "prods": {}, "sigma": {}, "sigma_source": {},
            "band": None, "s_km": None, "centers": [], "tau_band": [], "thin": None, "scenes": {},
            "status": {}, "checks": {}, "clamp": {}, "truth_error": None,
            "meta": {"inputs": {}, "notes": []}}


# ─────────────────────────────────────────────────────────────────────────────
#  gates
# ─────────────────────────────────────────────────────────────────────────────

def _close(a, b, tol=TOL_SCORE):
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b)), 0.0
    return abs(a - b) <= tol, abs(a - b)


def _row_check(mine, ref, keys):
    """(ok, max |diff|) of one row against its published row; n exact."""
    ok = int(mine["n"]) == int(float(ref["n"]))
    worst = 0.0
    for a, b in keys:
        o, d = _close(mine.get(a, np.nan), ref.get(b, np.nan))
        ok &= o
        worst = max(worst, d)
    return bool(ok), worst


def run_gates(S, published_csv):
    """The gates of one site (dict name -> {ok, ...})."""
    import pandas as pd

    g = {}
    kind, t, P, site = S["kind"], np.asarray(S["truth"], float), S["prods"], S["site"]
    pub = pd.read_csv(published_csv) if published_csv and os.path.exists(published_csv) else None

    # the published 'all' rows
    if pub is None:
        g["published_all_rows"] = {"checked": False, "ok": False,
                                   "reason": f"no published CSV {published_csv}"}
    else:
        rows, ok_all, worst = [], True, 0.0
        for p in P:
            r = pub[(pub["product"] == p) & (pub["subset"] == "all")]
            mine = metrics(kind, t, P[p])
            if r.empty:
                rows.append({"product": p, "ok": False, "reason": "no published 'all' row"})
                ok_all = False
                continue
            ok, d = _row_check(mine, r.iloc[0].to_dict(), (("rmse_centred_m", "rmse_m"), ("slope", "slope"),
                                                           ("r", "r"), ("bias_m", "bias_m")))
            rows.append({"product": p, "n": mine["n"], "n_published": int(float(r.iloc[0]["n"])),
                         "max_abs_diff": d, "ok": ok})
            ok_all &= ok
            worst = max(worst, d)
        g["published_all_rows"] = {"checked": True, "source": published_csv, "rows": rows,
                                   "max_abs_diff": worst, "ok": bool(ok_all)}

    # v9's same-scenes common subsets, rebuilt with v9's own code
    if pub is None:
        g["published_common_subsets"] = {"checked": False, "ok": False, "reason": "no published CSV"}
    elif REFERENCE not in P or "MAREA" not in P:
        g["published_common_subsets"] = {"checked": False, "ok": False, "reason": "no delay / MAREA missing"}
    else:
        if kind == "rtk":
            from pyintertidal import elevation_validation as ev

            def score_fn(v):
                return ev.ols_against_truth(t, v)
        else:
            from experiments.v7_waterline_contour import vscore

            def score_fn(v):
                return vscore(v, t)
        records = {p: ("", S["scenes"].get(p)) for p in P}
        extra, info = v9.same_scene_rows(site, P, records, np.isfinite(t), score_fn)
        rows, ok_all, worst = [], True, 0.0
        for r in extra:
            ref = pub[(pub["product"] == r["product"]) & (pub["subset"] == r["subset"])]
            if ref.empty:
                rows.append({"product": r["product"], "subset": r["subset"], "ok": False,
                             "reason": "no published row"})
                ok_all = False
                continue
            mine = {"n": r["n"], "rmse_m": r["rmse_m"], "slope": r["slope"], "r": r["r"], "bias_m": r["bias_m"]}
            ok, d = _row_check(mine, ref.iloc[0].to_dict(), (("rmse_m", "rmse_m"), ("slope", "slope"),
                                                             ("r", "r"), ("bias_m", "bias_m")))
            rows.append({"product": r["product"], "subset": r["subset"], "n": int(r["n"]), "max_abs_diff": d,
                         "ok": ok})
            ok_all &= ok
            worst = max(worst, d)
        g["published_common_subsets"] = {"checked": True, "subsets": {k: v["n"] for k, v in info.items()},
                                         "rows": rows, "max_abs_diff": worst, "ok": bool(ok_all)}

    # MAREA's sigma belongs to the scored MAREA
    agr = _agreement(S["checks"]["marea_z_units"], P["MAREA"], np.isfinite(t))
    d = agr["max_abs_diff_m"]
    g["sigma_belongs_to_scored_marea"] = {
        **agr, "on_keep": S["checks"].get("marea_z_vs_scored_on_keep"),
        "sigma_finite_where_z_finite_share": S["checks"].get("marea_sigma_finite_where_z_finite"),
        "ok": bool(agr["validity_flips"] == 0 and (d is None or d <= TOL_Z))}

    # the sensitivity's scenes are MAREA's scored record
    sd, md = S["checks"].get(f"dates_{SENS}"), S["checks"].get("marea_record_dates")
    if SENS not in P:
        g["same_scenes_record"] = {"checked": False, "ok": True, "reason": f"{SENS}: {S['status'].get(SENS)}"}
    else:
        same = sd is not None and md is not None and list(sd) == list(md)
        g["same_scenes_record"] = {"checked": True, "n_sensitivity": len(sd or []), "n_marea": len(md or []),
                                   "ok": bool(same)}

    # the band map (band -1: no mouth distance, lag 0 in production: allowed)
    sets, _, _ = build_sets(S)
    b = np.asarray(S["band"], np.int64)[sets["common"]]
    nb = len(S.get("centers") or [])
    out_of = sorted({int(x) for x in np.unique(b) if x != NO_MOUTH_BAND and not 0 <= x < nb})
    eq = S["checks"].get("band_products_vs_marea", True)
    g["band_map"] = {"n_bands_result_json": nb, "bands_without_centre_in_common": out_of,
                     "n_common_without_mouth_distance": int(np.sum(b == NO_MOUTH_BAND)),
                     "products_band_equals_marea_band": eq, "ok": bool(eq and not out_of)}

    if "record_dates_products_vs_result" in S["checks"]:
        v = S["checks"]["record_dates_products_vs_result"]
        g["record_dates"] = {"checked": v is not None, "ok": bool(v is None or v)}
    return g


def gates_ok(g):
    return all(v.get("ok", True) for v in g.values() if isinstance(v, dict))


# ─────────────────────────────────────────────────────────────────────────────
#  run one site
# ─────────────────────────────────────────────────────────────────────────────

def peak_memory_mb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)   # Linux: KiB
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        return None


def write_csv(df, path):
    df.to_csv(path, index=False)


def run_site(site, out=OUT, overrides=None, soft_gates=False):
    """Load, gate and analyse one site; write its tables and settings."""
    t0 = time.time()
    v14.guard_out(out)
    os.makedirs(out, exist_ok=True)
    cfg = site_inputs(site, overrides)
    log(f"[{site}] v17 truth diagnostics: inputs {json.dumps({k: cfg.get(k) for k in ('extract', 'product', 'comparison', 'nidem', 'nidem_sensitivity', 'external', 'cube', 'published')})}")
    S = load_villaviciosa(cfg) if site == "villaviciosa" else load_dutch(site, cfg)
    t_load = time.time() - t0
    log(f"[{site}] loaded {len(S['truth']):,} units ({int(np.isfinite(S['truth']).sum()):,} with truth), "
        f"products {sorted(S['prods'])}, uncertainties {sorted(k for k, v in S['sigma'].items() if v is not None)} "
        f"[{t_load:.1f} s]")
    settings = {
        "site": site, "site_name": v14.SITE_NAMES.get(site, site), "status": "running",
        "started": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
        "git_head": v14.git_head(), "script": v14.file_info(os.path.relpath(os.path.abspath(__file__), ROOT)),
        "reused": {k: v14.file_info(f"experiments/{k}.py") for k in ("v14_scene_count_curves",
                                                                       "v9_method_comparison",
                                                                       "v7_waterline_contour")},
        "python": sys.version.split()[0], "numpy": np.__version__, "blas": v14.blas_info(),
        "kind": S["kind"], "inputs_cfg": {k: v for k, v in cfg.items() if isinstance(v, (str, type(None)))},
        "overrides": {k: v for k, v in (overrides or {}).items() if v},
        "truth": S["meta"]["truth"], "inputs": S["meta"]["inputs"], "notes": S["meta"]["notes"],
        "scenes": S["scenes"], "status_products": S["status"], "sigma_source": S["sigma_source"],
        "definitions": {
            "bias_m": "median(product - truth) (v9)",
            "rmse_centred_m": "median-centred RMSE of the published tables (v7.vscore / ols_against_truth)",
            "rmse_total_m": "sqrt(mean((product - truth)^2)), uncentred: carries the bias",
            "common": "finite truth & MAREA & every computable NIDEM",
            "calibration": "common & every compared product's uncertainty finite and > 0",
            "e": "product - truth - median(product - truth) on the calibration set",
            "k68": "68.27th percentile of |e| / sigma", "ence": "Levi et al. 2019: mean over sigma bins "
                                                               "(n >= MIN_N) of |RMV - RMSE| / RMV",
            "thinned": "validation_grid.thin_mask: one unit per 100 m block (v9's thinned subset)",
            "min_n": MIN_N},
    }
    t1 = time.time()
    g = run_gates(S, cfg["published"])
    settings["gates"] = g
    for k, v in g.items():
        log(f"[{site}] gate {k}: {'OK' if v.get('ok', True) else 'FAILED'} "
            f"{ {kk: vv for kk, vv in v.items() if kk in ('max_abs_diff', 'validity_flips', 'max_abs_diff_m', 'checked', 'n_sensitivity', 'n_marea', 'bands_without_centre_in_common', 'n_common_without_mouth_distance')} }")
    jpath = os.path.join(out, f"{site}_settings.json")
    if not gates_ok(g):
        failed = [k for k, v in g.items() if not v.get("ok", True)]
        if not soft_gates:
            settings.update(status="gate_failed", failed_gates=failed,
                            finished=_dt.datetime.now().isoformat(timespec="seconds"))
            v14.write_json(jpath, settings)
            raise SystemExit(f"[{site}] gate(s) {failed} failed; see {jpath}")
        settings["gates_overridden"] = failed
        log(f"[{site}] WARNING: gate(s) {failed} failed and --soft-gates is set; the JSON says so")
    t_g = time.time() - t1

    t2 = time.time()
    tables, info = analyse(S)
    for name, df in tables.items():
        write_csv(df, os.path.join(out, f"{site}_{name}.csv"))
    upath = save_units(os.path.join(out, f"{site}_units.npz"), S)
    t_a = time.time() - t2
    settings.update(info)
    settings["units_file"] = {"path": upath, "bytes": os.path.getsize(upath),
                              "content": ("every unit with truth: truth, the products (p__*), the uncertainties "
                                          "(s__*), band, class, row, col, the common and calibration masks; floats "
                                          "stored as float32 only where that is lossless")}
    settings["status"] = "complete" if gates_ok(g) else "complete_gates_overridden"
    settings["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    settings["timing"] = {"load_seconds": round(t_load, 2), "gates_seconds": round(t_g, 2),
                          "analysis_seconds": round(t_a, 2), "total_seconds": round(time.time() - t0, 2),
                          "peak_memory_mb": peak_memory_mb()}
    v14.write_json(jpath, settings)
    tot = tables["totals"]
    c = tot[(tot["subset"] == "common")]
    for _, r in c.iterrows():
        log(f"[{site}] common n {int(r['n']):,}: {r['product']:28s} bias {r['bias_m']:+.3f} centred RMSE "
            f"{r['rmse_centred_m']:.3f} total RMSE {r['rmse_total_m']:.3f}")
    for _, r in tables["calibration"].iterrows():
        if r.get("status") == "ok":
            log(f"[{site}] calibration {r['product']:28s} n {int(r['n']):,}: within 1 sigma "
                f"{r['within_1sigma']:.1%}, 2 sigma {r['within_2sigma']:.1%}, Spearman {r['spearman_rho']:.2f}, "
                f"k68 {r['k68']:.2f}")
    log(f"[{site}] {settings['status']} -> {out}/{site}_*.csv, {jpath} "
        f"[{time.time() - t0:.1f} s, peak {settings['timing']['peak_memory_mb']} MB]")
    return tables, settings


# ─────────────────────────────────────────────────────────────────────────────
#  the units file and the paired bootstrap
# ─────────────────────────────────────────────────────────────────────────────

#: product name -> key in the units file
UNIT_KEYS = {"MAREA": "MAREA", "NIDEM": "NIDEM", SENS: "NIDEM_marea_record", REFERENCE: "no_delay"}
#: the paired comparisons of ``bootstrap`` (a - b)
BOOT_PAIRS = (("MAREA", "NIDEM"), ("MAREA", SENS), ("MAREA", REFERENCE))
BOOT_STATS = ("rmse_total", "rmse", "abs_bias", "bias", "within_1sigma", "within_1sigma_tsite")
TOL_BOOT_POINT = 1e-9


def _compact(a):
    """float64 -> float32 where that is lossless (NaN kept), else unchanged."""
    a = np.asarray(a)
    if a.dtype == np.float64:
        b = a.astype(np.float32)
        if np.array_equal(b.astype(np.float64), a, equal_nan=True):
            return b
    return a


def save_units(path, S):
    """Every unit with truth (RTK point / pixel): truth, products, sigmas,
    band, class, row, col, the common and calibration masks; written
    atomically. Floats are float32 only where that is lossless, so a
    re-score reproduces the tables exactly."""
    t = np.asarray(S["truth"], float)
    sets, _, _ = build_sets(S)
    cls, _ = truth_classes(t, sets["common"])
    u = np.flatnonzero(np.isfinite(t))
    arr = {"truth": _compact(t[u]), "band": np.asarray(S["band"], np.int64)[u].astype(np.int16),
           "class": cls[u].astype(np.int8), "row": np.asarray(S["rows"], np.int64)[u].astype(np.int32),
           "col": np.asarray(S["cols"], np.int64)[u].astype(np.int32),
           "mask_common": sets["common"][u], "mask_calibration": sets["calibration"][u],
           "unit_index": u.astype(np.int64), "pixel_m": np.array(float(S.get("pixel_m") or 10.0)),
           "kind": np.array(S["kind"]), "site": np.array(S["site"]),
           "sigma_t_site_m": np.array(float((S.get("truth_error") or {}).get("sigma_t_m", np.nan)))}
    for p, v in S["prods"].items():
        arr[f"p__{UNIT_KEYS[p]}"] = _compact(np.asarray(v, float)[u])
    for p, v in S["sigma"].items():
        if v is not None and p in UNIT_KEYS:
            arr[f"s__{UNIT_KEYS[p]}"] = _compact(np.asarray(v, float)[u])
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, path)
    return path


def load_units(path):
    """The units file back as float64 arrays: {'truth', 'prods' {name: z},
    'sigma' {name: s}, 'band', 'class', 'row', 'col', 'mask_common',
    'mask_calibration', 'kind', 'site', 'pixel_m', 'sigma_t_site_m'}."""
    names = {v: k for k, v in UNIT_KEYS.items()}
    with np.load(path, allow_pickle=False) as f:
        D = {"prods": {}, "sigma": {}}
        for k in f.files:
            if k.startswith("p__"):
                D["prods"][names[k[3:]]] = f[k].astype(float)
            elif k.startswith("s__"):
                D["sigma"][names[k[3:]]] = f[k].astype(float)
            elif k in ("kind", "site"):
                D[k] = str(f[k])
            elif k in ("pixel_m", "sigma_t_site_m"):
                D[k] = float(f[k])
            elif k == "truth":
                D[k] = f[k].astype(float)
            else:
                D[k] = np.asarray(f[k])
    return D


def bootstrap_site(site, out=OUT, n_boot=pbs.N_BOOT, seed=pbs.SEED, level=pbs.LEVEL, block_m=pbs.BLOCK_M):
    """Paired cluster bootstrap of MAREA - NIDEM, MAREA - NIDEM on MAREA's
    record and MAREA - no delay on the common units of one site (from
    <site>_units.npz). Writes <site>_bootstrap.csv / .json; returns the
    rows."""
    import pandas as pd

    t0 = time.time()
    v14.guard_out(out)
    upath = os.path.join(out, f"{site}_units.npz")
    if not os.path.exists(upath):
        raise SystemExit(f"[{site}] no {upath}: run 'run {site}' first")
    D = load_units(upath)
    kind, t = D["kind"], D["truth"]
    C = D["mask_common"].astype(bool)
    cal = D["mask_calibration"].astype(bool)
    labels, rule = pbs.cluster_labels(kind, D["row"], D["col"], D["pixel_m"], block_m)
    st_site = D.get("sigma_t_site_m", np.nan)
    tot_path = os.path.join(out, f"{site}_totals.csv")
    tot = pd.read_csv(tot_path) if os.path.exists(tot_path) else None
    rows, checks = [], []
    for a, b in BOOT_PAIRS:
        if a not in D["prods"] or b not in D["prods"]:
            continue
        pa, pb_ = D["prods"][a], D["prods"][b]
        u = C & np.isfinite(pa) & np.isfinite(pb_)
        if int(u.sum()) < MIN_N[kind]:
            continue
        ci, B = pbs.cluster_index(labels[u])
        tu, au, bu = t[u], pa[u], pb_[u]
        counts = pbs.resample_counts(B, n_boot, seed, key=f"v17:{site}:{a}-{b}")
        mt = pbs.WeightedMedian(tu, ci)
        fns = {"a": pbs.MetricEngine(tu, au, ci, B, med_t=mt), "b": pbs.MetricEngine(tu, bu, ci, B, med_t=mt)}
        pts = {"a": pbs.point_metrics(tu, au, kind), "b": pbs.point_metrics(tu, bu, kind)}
        # within 1 sigma: on the pair's units where both methods state an uncertainty
        sa, sb = D["sigma"].get(a), D["sigma"].get(b)
        uc = None
        if sa is not None and sb is not None:
            with np.errstate(invalid="ignore"):
                uc = cal[u] & np.isfinite(sa[u]) & (sa[u] > 0) & np.isfinite(sb[u]) & (sb[u] > 0)
            if int(uc.sum()) >= MIN_N[kind]:
                for tag, stv in (("within_1sigma", 0.0), ("within_1sigma_tsite", st_site)):
                    if not np.isfinite(stv):
                        continue
                    ea = np.sqrt(sa[u][uc] ** 2 + stv ** 2)
                    eb = np.sqrt(sb[u][uc] ** 2 + stv ** 2)
                    fns[f"a_{tag}"] = pbs.WithinSigmaEngine(tu[uc], au[uc], ea, ci[uc])
                    fns[f"b_{tag}"] = pbs.WithinSigmaEngine(tu[uc], bu[uc], eb, ci[uc])
                    pts[f"a_{tag}"] = pbs.point_within(tu[uc], au[uc], ea)
                    pts[f"b_{tag}"] = pbs.point_within(tu[uc], bu[uc], eb)
            else:
                uc = None
        t1 = time.time()
        reps = pbs.replicate(counts, fns)
        secs = time.time() - t1
        for stat in BOOT_STATS:
            if stat.startswith("within"):
                if f"a_{stat}" not in fns:
                    continue
                pa_, pb2 = pts[f"a_{stat}"], pts[f"b_{stat}"]
                ra, rb = reps[f"a_{stat}"], reps[f"b_{stat}"]
                n_u = int(uc.sum())
            else:
                pa_, pb2 = pts["a"][stat], pts["b"][stat]
                ra, rb = reps[f"a.{stat}"], reps[f"b.{stat}"]
                n_u = int(u.sum())
            iv = pbs.paired_interval(pa_, pb2, ra, rb, level)
            rows.append({"site": site, "comparison": f"{a} - {b}", "a": a, "b": b, "stat": stat, "n_units": n_u,
                         "n_clusters": int(B), "cluster_rule": rule, "a_value": iv["a"], "b_value": iv["b"],
                         "diff": iv["diff"], "ci_lo": iv["ci_lo"], "ci_hi": iv["ci_hi"],
                         "share_diff_below_0": iv["share_diff_below_0"],
                         "n_valid_resamples": iv["n_valid_resamples"], "n_boot": int(n_boot), "seed": int(seed),
                         "level": float(level), "indicative": bool(kind == "rtk" or B < 30),
                         "sigma_t_m": (float(st_site) if stat == "within_1sigma_tsite" else
                                       0.0 if stat == "within_1sigma" else np.nan),
                         "seconds": round(secs, 2)})
        # the point estimates are the totals of the run (same units, same functions)
        if tot is not None:
            for name, key in ((a, "a"), (b, "b")):
                r = tot[(tot["subset"] == "common") & (tot["product"] == name)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if int(r["n"]) != int(u.sum()):
                    # MAREA in the pair with no delay: scored on the part of C where no delay
                    # has an estimate, not on C itself (no published row to compare)
                    continue
                d = max(abs(float(r["rmse_total_m"]) - pts[key]["rmse_total"]),
                        abs(float(r["rmse_centred_m"]) - pts[key]["rmse"]),
                        abs(float(r["bias_m"]) - pts[key]["bias"]))
                checks.append({"comparison": f"{a} - {b}", "product": name, "n_totals": int(r["n"]),
                               "n_bootstrap": int(u.sum()), "max_abs_diff": d, "ok": bool(d <= TOL_BOOT_POINT)})
        log(f"[{site}] bootstrap {a} - {b}: {int(u.sum()):,} units in {B:,} clusters, {n_boot} resamples "
            f"[{secs:.1f} s]")
    df = pd.DataFrame(rows)
    write_csv(df, os.path.join(out, f"{site}_bootstrap.csv"))
    ok = all(c["ok"] for c in checks) if checks else None
    v14.write_json(os.path.join(out, f"{site}_bootstrap.json"), {
        "site": site, "units_file": v14.file_info(upath), "n_boot": int(n_boot), "seed": int(seed),
        "level": float(level), "cluster_rule": rule, "block_m": float(block_m), "kind": kind,
        "indicative": ("RTK: the clusters are the pixels holding the points (about 120): an indicative interval"
                       if kind == "rtk" else "1 km blocks"),
        "method": ("paired cluster bootstrap (experiments/paired_bootstrap.py): the same cluster draw for both "
                   "methods; exact multiset medians; percentile interval of the replicate differences a - b"),
        "gate_points_reproduce_totals": {"rows": checks, "ok": ok},
        "script": v14.file_info(os.path.relpath(os.path.abspath(__file__), ROOT)),
        "seconds": round(time.time() - t0, 1), "peak_memory_mb": peak_memory_mb()})
    if ok is False:
        raise SystemExit(f"[{site}] the bootstrap's point estimates do not reproduce {tot_path}; see "
                         f"{site}_bootstrap.json")
    for r in rows:
        if r["stat"] in ("rmse_total", "rmse", "abs_bias", "within_1sigma"):
            log(f"[{site}] {r['comparison']:34s} {r['stat']:14s} diff {r['diff']:+.4f} "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]" + (" (indicative)" if r["indicative"] else ""))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  report: combined tables, markdown, figures
# ─────────────────────────────────────────────────────────────────────────────

TXT = {
    "en": {
        "prod": {"MAREA": "MAREA", "NIDEM": "NIDEM as published (all scenes)",
                 SENS: "NIDEM on MAREA's scenes", REFERENCE: "MAREA without lag (reference)"},
        "short": {"MAREA": "MAREA", "NIDEM": "NIDEM", SENS: "NIDEM, MAREA's scenes", REFERENCE: "no lag"},
        "site": dict(v14.SITE_NAMES),
        "truth": {"rtk": "RTK-GNSS, development split (Alicante datum)", "vaklodingen": "Vaklodingen (NAP)"},
        "unit": {"rtk": "RTK points", "vaklodingen": "px"},
        "classes": ("lower", "middle", "upper"),
        "class_word": "third",
        "t1": "Truth-side errors of the published 10 m products, by along-channel band and by elevation class",
        "s1": ("Common pixels of MAREA and NIDEM at each site. Bias = median (product − truth); "
               "RMSE median-centred, as in the published tables."),
        "cols1": ("Bias by band (m)", "Centred RMSE by band (m)", "Bias by elevation third (m)",
                  "Centred RMSE by elevation third (m)"),
        "xband": "Distance from the mouth, centre of the 3 km band (km)",
        "xclass": "Third of the truth range (m)",
        "foot1": ("Bands: MAREA's along-channel band map (3 km). Elevation thirds: equal-width thirds of the "
                  "truth's range on the common pixels, read from the truth. Points with fewer than {rtk} RTK points "
                  "/ {px} px are not drawn.\nThe upper third is bounded by each method's ceiling (MAREA: the "
                  "record's highest zero-lag level, the end of its inversion grid; NIDEM: its highest interval "
                  "height), and classes read from a truth with its own error regress to the mean: its negative "
                  "bias is not a defect of one method alone (share at the clamp: truth_by_class.csv)."),
        "t2": "Does each method's own uncertainty predict its error?",
        "s2": ("e = product − truth − median (product − truth) at the site, on the common pixels where every "
               "uncertainty exists. Stated spreads, not standard errors:\nMAREA: width of the fitted wet/dry "
               "transition (a grid of 8 values); NIDEM: its uncertainty raster (tide-height SD of the interval)."),
        "relx": "Stated uncertainty σ (m)", "rely": "68th percentile of |e| (m)",
        "one": "1:1",
        "covx": "Truths within ±1σ and ±2σ (%)", "w1": "within ±1σ", "w2": "within ±2σ",
        "w_t": "with the truth's error σₜ added (hollow)",
        "gauss": "Gaussian 68 % and 95 %",
        "foot2": ("Top: one point per σ bin (MAREA: each value of its σ grid; NIDEM: deciles), bins with fewer than "
                  "{rtk} RTK points / {px} px omitted. Bottom: filled, the truth taken as exact; hollow, against "
                  "√(σ² + σₜ²) with the truth's own error σₜ ({st}).\nρ = Spearman correlation of σ with |e| "
                  "(in brackets: one unit per 100 m block); k68 = factor σ would need to hold 68 % of the errors; "
                  "slope = q68 of |e| on σ over the bins (σ follows the errors when it lies in 0.8-1.25)."),
        "st_rtk": "Villaviciosa: {v} m, the pooled spread of RTK heights inside a pixel",
        "st_vak": "Dutch sites: {v} m assumed (survey error, 20 m cell, bed change)",
        "t3": "Total versus median-centred RMSE on the common pixels",
        "s3": ("Ring: median-centred RMSE (the published figure). Dot: total RMSE, "
               "√mean (product − truth)², which also carries the bias."),
        "x3": "RMSE (m)", "bias": "bias", "centred": "centred RMSE", "total": "total RMSE",
        "foot3": ("Both methods use the same boundary levels, so the gap between their total RMSEs is a method "
                  "effect; the total RMSE also holds the datum offset between the boundary and the truth, common "
                  "to every product.\nThe published centred RMSE aligns the medians of product and truth (v9), so "
                  "it can exceed the total RMSE when the slope departs from 1. MAREA without lag: scored where it "
                  "has an estimate."),
        "nc": "not computable", "nores": "no results yet", "common_n": "common", "n": "n",
        "md_title": "Truth-side diagnostics: published 10 m MAREA and NIDEM",
        "md_gen": "Generated {date} by experiments/v17_truth_diagnostics.py from {out}.",
        "md_intro": ("Products compared on their common pixels (truth & MAREA & every computable NIDEM): MAREA, "
                     "NIDEM as published (ITEM all-observations rule) and NIDEM on MAREA's record (v7 "
                     "sensitivity); MAREA without lag is a reference row, scored on the common pixels where it "
                     "has an estimate. Truth: Vaklodingen (NAP; v1 sampling and the v9 range filter) and the RTK "
                     "development split (Alicante; pyintertidal.rtk.load_rtk() defaults). Bias = median (product − "
                     "truth); centred RMSE = the published median-centred RMSE (v9); total RMSE = √mean (product − "
                     "truth)²."),
        "md_gates": "Gates", "md_s1": "1. Bias and centred RMSE by band and by elevation third (common pixels)",
        "md_s1_note": ("Each product cell: bias / centred RMSE / total RMSE (m); -- below the minimum n. Band −1: "
                       "pixels without a mouth distance, to which production applies lag 0."),
        "md_clamp_note": ("Share of each product's estimates at its ceiling in the upper third (MAREA and no lag: "
                          "within half a step of the top of the inversion grid, the record's highest zero-lag "
                          "level; NIDEM: at its highest interval height): {txt}. The upper third is bounded by "
                          "these ceilings and, as its class is read from a truth with its own error, regresses to "
                          "the mean; its negative bias is not a defect of one method alone."),
        "md_band": "By along-channel band", "md_class": "By elevation third",
        "no_mouth": "−1 (no mouth distance, lag 0)",
        "md_s2": "2. Calibration of each method's own uncertainty",
        "md_s2_note": ("On the common pixels where every uncertainty exists; e = product − truth − its site bias. "
                       "Neither uncertainty is a standard error of the elevation: MAREA's σ is the fitted width of "
                       "the wet/dry transition (one of 8 grid values; it does not shrink with more observations), "
                       "NIDEM's the tide-height SD of its interval. So the first question is whether σ RANKS the "
                       "errors (Spearman ρ, also on one unit per 100 m block) and whether the per-bin 68th "
                       "percentile of |e| FOLLOWS σ (slope of q68 on σ; 'follows' in 0.8-1.25); only then is the "
                       "level (within 1σ, 68.3 % for Gaussian errors; within 2σ, 95.4 %; k68 = 1) a calibration. "
                       "The truth has its own error, which costs most to the smaller σ: the last columns add it, "
                       "√(σ² + σₜ²), with σₜ {st}; the CSV holds σₜ = 0, 0.05, 0.10 and 0.15 m."),
        "md_verdict": "{site}, {prod}: ρ {rho} ({rho_t} thinned); {verdict}.",
        "md_s4": "4. Paired differences with block-bootstrap intervals",
        "md_s4_note": ("MAREA minus each comparator on the same units, resampled together (paired cluster bootstrap, "
                       "experiments/paired_bootstrap.py): {rules}; {n_boot} resamples, 95 % percentile interval. "
                       "A negative difference favours MAREA for the RMSEs and |bias|, a positive one for the share "
                       "within 1σ. Villaviciosa's interval is indicative (its clusters are the ~120 pixels holding "
                       "the RTK points)."),
        "h_boot": ("site", "comparison", "statistic", "units", "clusters", "MAREA", "other", "difference",
                   "95 % interval", "share < 0"),
        "boot_stat": {"rmse_total": "total RMSE (m)", "rmse": "centred RMSE (m)", "abs_bias": "|bias| (m)",
                      "bias": "bias (m)", "within_1sigma": "within 1σ", "within_1sigma_tsite": "within 1σ (+σₜ)"},
        "md_s3": "3. Total versus centred RMSE",
        "md_s3_note": ("all = each product on every unit it resolves (the published rows); common = the common "
                       "pixels; pair_published = MAREA & NIDEM; pair_same_scenes = MAREA & NIDEM on MAREA's record. "
                       "d = product − truth; total RMSE² = (mean d)² + (SD d)² exactly. The published centred RMSE "
                       "subtracts the product's median from the product and the truth's median from the truth (v9): "
                       "it is not the RMSE about the bias, and it can exceed the total RMSE when the slope departs "
                       "from 1."),
        "md_missing": "Sites without results",
        "h_band": ("band", "s (km)", "lag (min)", "n"), "h_class": ("third", "truth range (m)", "n"),
        "h_cal": ("site", "product", "n", "Spearman ρ (thinned)", "slope q68/σ", "site bias (m)", "σ median (m)",
                  "within 1σ", "within 2σ", "k68", "RMS e/σ", "ENCE", "σₜ (m)", "within 1σ (+σₜ)",
                  "within 2σ (+σₜ)", "k68 (+σₜ)"),
        "h_tot": ("site", "product", "subset", "scenes", "n", "centred RMSE (m)", "total RMSE (m)", "bias (m)",
                  "mean d (m)", "SD d (m)", "slope", "r"),
        "h_gate": ("site", "gate", "result", "detail"),
        "ok": "OK", "failed": "FAILED", "ref_note": "MAREA without lag scored on {n} of the {N} common units.",
        "lang": "en", "all": "all",
    },
    "es": {
        "prod": {"MAREA": "MAREA", "NIDEM": "NIDEM publicado (todas las escenas)",
                 SENS: "NIDEM con las escenas de MAREA", REFERENCE: "MAREA sin retardo (referencia)"},
        "short": {"MAREA": "MAREA", "NIDEM": "NIDEM", SENS: "NIDEM, escenas MAREA", REFERENCE: "sin retardo"},
        "site": {"villaviciosa": "Ría de Villaviciosa", "escalda": "Escalda occidental (Westerschelde)",
                 "wadden": "Mar de Wadden (Vlie)", "ems": "Ems-Dollard"},
        "truth": {"rtk": "RTK-GNSS, mitad de desarrollo (datum de Alicante)", "vaklodingen": "Vaklodingen (NAP)"},
        "unit": {"rtk": "puntos RTK", "vaklodingen": "px"},
        "classes": ("inferior", "medio", "superior"),
        "class_word": "tercio",
        "t1": "Errores frente a la verdad de los productos publicados a 10 m, por tramo del canal y por cota",
        "s1": ("Píxeles comunes de MAREA y NIDEM en cada sitio. Sesgo = mediana (producto − verdad); "
               "RMSE centrado en la mediana, como en las tablas publicadas."),
        "cols1": ("Sesgo por tramo (m)", "RMSE centrado por tramo (m)", "Sesgo por tercio de cota (m)",
                  "RMSE centrado por tercio de cota (m)"),
        "xband": "Distancia a la desembocadura, centro del tramo de 3 km (km)",
        "xclass": "Tercio del rango de la verdad (m)",
        "foot1": ("Tramos: el mapa de tramos de 3 km a lo largo del canal de MAREA. Tercios de cota: tercios de "
                  "igual anchura del rango de la verdad en los píxeles comunes, leídos de la verdad. No se dibujan "
                  "los puntos con menos de {rtk} puntos RTK / {px} px.\nEl tercio superior está acotado por el techo "
                  "de cada método (MAREA: el nivel sin retardo más alto del registro, el extremo de su rejilla de "
                  "inversión; NIDEM: su altura de intervalo más alta) y unas clases leídas de una verdad con error "
                  "propio regresan a la media: su sesgo negativo no es un defecto de un solo método (fracción en el "
                  "techo: truth_by_class.csv)."),
        "t2": "¿Predice la incertidumbre propia de cada método su error?",
        "s2": ("e = producto − verdad − mediana (producto − verdad) en el sitio, en los píxeles comunes con todas "
               "las incertidumbres. Dispersiones declaradas, no errores típicos:\nMAREA: anchura de la transición "
               "seco/mojado ajustada (una rejilla de 8 valores); NIDEM: su ráster de incertidumbre (desviación "
               "típica de la marea en el intervalo)."),
        "relx": "Incertidumbre declarada σ (m)", "rely": "Percentil 68 de |e| (m)",
        "one": "1:1",
        "covx": "Verdades dentro de ±1σ y ±2σ (%)", "w1": "dentro de ±1σ", "w2": "dentro de ±2σ",
        "w_t": "con el error propio de la verdad σₜ (huecos)",
        "gauss": "gaussiana: 68 % y 95 %",
        "foot2": ("Arriba: un punto por clase de σ (MAREA: cada valor de su rejilla de σ; NIDEM: deciles); se omiten "
                  "las clases con menos de {rtk} puntos RTK / {px} px. Abajo: llenos, la verdad tomada como exacta; "
                  "huecos, frente a √(σ² + σₜ²) con el error propio de la verdad σₜ ({st}).\nρ = correlación de "
                  "Spearman entre σ y |e| (entre paréntesis: una unidad por bloque de 100 m); k68 = factor que "
                  "necesitaría σ para abarcar el 68 % de los errores; pendiente = q68 de |e| frente a σ por clases "
                  "(σ sigue a los errores entre 0,8 y 1,25)."),
        "st_rtk": "Villaviciosa: {v} m, la dispersión conjunta de las cotas RTK dentro de un píxel",
        "st_vak": "sitios neerlandeses: {v} m supuesto (error del levantamiento, celda de 20 m, cambio del fondo)",
        "t3": "RMSE total frente al RMSE centrado en los píxeles comunes",
        "s3": ("Anillo: RMSE centrado en la mediana (la cifra publicada). Punto: RMSE total, "
               "√media (producto − verdad)², que incluye también el sesgo."),
        "x3": "RMSE (m)", "bias": "sesgo", "centred": "RMSE centrado", "total": "RMSE total",
        "foot3": ("Los dos métodos usan los mismos niveles de frontera, así que la diferencia entre sus RMSE totales "
                  "es un efecto del método; el RMSE total incluye además el desfase de datum entre la frontera y la "
                  "verdad, común a todos los productos.\nEl RMSE centrado publicado alinea las medianas del producto "
                  "y de la verdad (v9), así que puede superar al RMSE total cuando la pendiente se aleja de 1. MAREA "
                  "sin retardo: puntuado donde tiene estimación."),
        "nc": "no computable", "nores": "sin resultados todavía", "common_n": "comunes", "n": "n",
        "md_title": "Diagnóstico frente a la verdad: MAREA y NIDEM publicados a 10 m",
        "md_gen": "Generado el {date} por experiments/v17_truth_diagnostics.py a partir de {out}.",
        "md_intro": ("Productos comparados en sus píxeles comunes (verdad y MAREA y cada NIDEM computable): MAREA, "
                     "NIDEM publicado (regla ITEM de todas las observaciones) y NIDEM con el registro de MAREA "
                     "(sensibilidad de v7); MAREA sin retardo es una fila de referencia, puntuada en los píxeles "
                     "comunes donde tiene estimación. Verdad: Vaklodingen (NAP; muestreo de v1 y filtro de rango de "
                     "v9) y la mitad de desarrollo del RTK (Alicante; pyintertidal.rtk.load_rtk() con sus valores "
                     "por defecto). Sesgo = mediana (producto − verdad); RMSE centrado = el RMSE centrado en la "
                     "mediana publicado (v9); RMSE total = √media (producto − verdad)²."),
        "md_gates": "Controles", "md_s1": "1. Sesgo y RMSE centrado por tramo y por tercio de cota (píxeles comunes)",
        "md_s1_note": ("Cada celda de producto: sesgo / RMSE centrado / RMSE total (m); -- por debajo del n mínimo. "
                       "Tramo −1: píxeles sin distancia a la desembocadura, a los que producción aplica retardo 0."),
        "md_clamp_note": ("Fracción de las estimaciones de cada producto en su techo dentro del tercio superior "
                          "(MAREA y sin retardo: a menos de medio paso del extremo superior de la rejilla de "
                          "inversión, el nivel sin retardo más alto del registro; NIDEM: en su altura de intervalo "
                          "más alta): {txt}. El tercio superior está acotado por esos techos y, como su clase se lee "
                          "de una verdad con error propio, regresa a la media; su sesgo negativo no es un defecto de "
                          "un solo método."),
        "md_band": "Por tramo a lo largo del canal", "md_class": "Por tercio de cota",
        "no_mouth": "−1 (sin distancia a la desembocadura, retardo 0)",
        "md_s2": "2. Calibración de la incertidumbre propia de cada método",
        "md_s2_note": ("En los píxeles comunes con todas las incertidumbres; e = producto − verdad − su sesgo en el "
                       "sitio. Ninguna de las dos incertidumbres es un error típico de la cota: la σ de MAREA es la "
                       "anchura ajustada de la transición seco/mojado (uno de 8 valores de rejilla; no disminuye con "
                       "más observaciones), la de NIDEM la desviación típica de la marea en su intervalo. Por eso lo "
                       "primero es si σ ORDENA los errores (ρ de Spearman, también con una unidad por bloque de "
                       "100 m) y si el percentil 68 de |e| por clases SIGUE a σ (pendiente de q68 frente a σ; «sigue» "
                       "entre 0,8 y 1,25); solo entonces el nivel (dentro de 1σ, 68,3 % para errores gaussianos; "
                       "dentro de 2σ, 95,4 %; k68 = 1) es una calibración. La verdad tiene su propio error, que "
                       "penaliza más a la σ menor: las últimas columnas lo suman, √(σ² + σₜ²), con σₜ {st}; el CSV "
                       "tiene σₜ = 0, 0,05, 0,10 y 0,15 m."),
        "md_verdict": "{site}, {prod}: ρ {rho} ({rho_t} aclarado); {verdict}.",
        "verdicts": {"follows": "q68 sigue a σ (pendiente {s})", "not": "q68 no sigue a σ (pendiente {s})",
                     "few": "menos de 2 clases de σ puntuadas: forma sin comprobar"},
        "md_s4": "4. Diferencias pareadas con intervalos bootstrap por bloques",
        "md_s4_note": ("MAREA menos cada comparador en las mismas unidades, remuestreadas juntas (bootstrap pareado "
                       "por conglomerados, experiments/paired_bootstrap.py): {rules}; {n_boot} remuestreos, intervalo "
                       "de percentiles al 95 %. Una diferencia negativa favorece a MAREA en los RMSE y en |sesgo|, una "
                       "positiva en la fracción dentro de 1σ. El intervalo de Villaviciosa es orientativo (sus "
                       "conglomerados son los ~120 píxeles con puntos RTK)."),
        "h_boot": ("sitio", "comparación", "estadístico", "unidades", "conglomerados", "MAREA", "otro",
                   "diferencia", "intervalo 95 %", "fracción < 0"),
        "boot_stat": {"rmse_total": "RMSE total (m)", "rmse": "RMSE centrado (m)", "abs_bias": "|sesgo| (m)",
                      "bias": "sesgo (m)", "within_1sigma": "dentro de 1σ", "within_1sigma_tsite": "dentro de 1σ (+σₜ)"},
        "md_s3": "3. RMSE total frente al centrado",
        "md_s3_note": ("all = cada producto en todas las unidades que resuelve (las filas publicadas); common = los "
                       "píxeles comunes; pair_published = MAREA y NIDEM; pair_same_scenes = MAREA y NIDEM con el "
                       "registro de MAREA. d = producto − verdad; RMSE total² = (media d)² + (DT d)² exactamente. El "
                       "RMSE centrado publicado resta al producto su mediana y a la verdad la suya (v9): no es el RMSE "
                       "alrededor del sesgo y puede superar al RMSE total cuando la pendiente se aleja de 1."),
        "md_missing": "Sitios sin resultados",
        "h_band": ("tramo", "s (km)", "retardo (min)", "n"), "h_class": ("tercio", "rango de la verdad (m)", "n"),
        "h_cal": ("sitio", "producto", "n", "ρ de Spearman (aclarado)", "pendiente q68/σ", "sesgo del sitio (m)",
                  "σ mediana (m)", "dentro de 1σ", "dentro de 2σ", "k68", "RMS e/σ", "ENCE", "σₜ (m)",
                  "dentro de 1σ (+σₜ)", "dentro de 2σ (+σₜ)", "k68 (+σₜ)"),
        "h_tot": ("sitio", "producto", "subconjunto", "escenas", "n", "RMSE centrado (m)", "RMSE total (m)",
                  "sesgo (m)", "media d (m)", "DT d (m)", "pendiente", "r"),
        "h_gate": ("sitio", "control", "resultado", "detalle"),
        "ok": "OK", "failed": "FALLO", "ref_note": "MAREA sin retardo puntuado en {n} de las {N} unidades comunes.",
        "lang": "es", "all": "todos",
    },
}
TXT["en"]["verdicts"] = {"follows": "q68 follows σ (slope {s})", "not": "q68 does not follow σ (slope {s})",
                         "few": "fewer than 2 scored σ bins: shape not tested"}

STYLE = {
    "MAREA": dict(color=PAL["deep"], marker="o", ls="-", mfc=PAL["deep"]),
    "NIDEM": dict(color=PAL["sand"], marker="^", ls="-", mfc=PAL["sand"]),
    SENS: dict(color=PAL["sand"], marker="v", ls=(0, (4, 2)), mfc="white"),
    REFERENCE: dict(color=PAL["slate"], marker="s", ls=(0, (1, 1.5)), mfc=PAL["slate"]),
}


#: digit grouping of the Spanish outputs (RAE / SI: a narrow no-break space
#: between groups of three, only for numbers of five or more digits; decimal
#: comma). The same rule in v16, v17 and v18.
NNBSP = " "


def _num(v, d=3, sign=False, lang="en"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(v):
        return "--"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    s = s.replace("-", "−")
    return s.replace(".", ",") if lang == "es" else s


def _pct(v, lang="en"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(v):
        return "--"
    s = f"{100 * v:.1f} %"
    return s.replace(".", ",") if lang == "es" else s


def _int(v, lang="en"):
    """An integer, thousands grouped (English: comma; Spanish: RAE, a narrow
    no-break space from five digits on); '--' when missing."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v) if v is not None else "--"
    if not np.isfinite(v):
        return "--"
    n = int(round(v))
    digits = str(abs(n))
    if lang == "es":
        s = digits if len(digits) <= 4 else f"{abs(n):,}".replace(",", NNBSP)
    else:
        s = f"{abs(n):,}"
    return ("−" if n < 0 else "") + s


def _comma_axis(ax, lang, which=("x", "y")):
    """Decimal commas on the tick labels of a Spanish figure (linear axes)."""
    if lang != "es":
        return
    from matplotlib.ticker import ScalarFormatter

    class _Comma(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")

    for w in which:
        axis = ax.xaxis if w == "x" else ax.yaxis
        if axis.get_scale() == "linear" and isinstance(axis.get_major_formatter(), ScalarFormatter):
            axis.set_major_formatter(_Comma())


def load_report_inputs(out, sites=None):
    """{site: {tables..., settings}} for the sites whose run completed."""
    import pandas as pd

    D, missing = {}, {}
    for site in (sites or SITE_ORDER):
        jp = os.path.join(out, f"{site}_settings.json")
        if not os.path.exists(jp):
            missing[site] = "not run"
            continue
        st = json.load(open(jp, encoding="utf-8"))
        if not str(st.get("status", "")).startswith("complete"):
            missing[site] = f"status {st.get('status')}"
            continue
        d = {"settings": st}
        for k in ("by_band", "by_class", "calibration", "reliability", "totals"):
            f = os.path.join(out, f"{site}_{k}.csv")
            d[k] = pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 1 else pd.DataFrame()
        D[site] = d
    return D, missing


def _kind(D, site):
    return D[site]["settings"].get("kind", "rtk" if site == "villaviciosa" else "vaklodingen")


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10.0, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.titlepad": 5, "axes.unicode_minus": True})
    return plt


class _Sheet:
    """A figure laid out in inches from its top-left corner."""

    def __init__(self, plt, W, H):
        self.W, self.H = float(W), float(H)
        self.fig = plt.figure(figsize=(self.W, self.H))

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1.0 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("va", "top")
        kw.setdefault("ha", "left")
        return self.fig.text(x / self.W, 1.0 - y / self.H, s, **kw)

    def legend(self, handles, x, y, **kw):
        return self.fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x / self.W, 1.0 - y / self.H),
                               frameon=False, **kw)


def _site_head(sh, x, y, D, site, T, two_lines=False):
    """Site name and its truth / common-n line(s), the name's top at ``y``
    in (``two_lines``: truth and n on separate lines, for narrow columns)."""
    sh.text(x, y, T["site"].get(site, site), fontsize=12.5, color=PAL["deep"], weight="bold")
    if site in D:
        k = _kind(D, site)
        n = D[site]["settings"].get("sets", {}).get("common")
        nn = f"{T['common_n']} n = {_int(n, T['lang'])} {T['unit'][k]}"
        if two_lines:
            sh.text(x, y + 0.25, T["truth"][k], fontsize=9.0, color=PAL["slate"])
            sh.text(x, y + 0.43, nn, fontsize=9.0, color=PAL["slate"])
        else:
            sh.text(x, y + 0.24, f"{T['truth'][k]} · {nn}", fontsize=9.0, color=PAL["slate"])


def _empty(ax, T):
    ax.text(0.5, 0.5, T["nores"], transform=ax.transAxes, ha="center", va="center", fontsize=11,
            color=PAL["slate"])
    ax.set_xticks([])
    ax.set_yticks([])


def _legend_handles(T, products, extra=()):
    from matplotlib.lines import Line2D
    hs = []
    for p in products:
        stl = STYLE[p]
        hs.append(Line2D([], [], color=stl["color"], ls=stl["ls"], lw=1.8, marker=stl["marker"], ms=7,
                         mfc=stl["mfc"], mec=stl["color"], mew=1.3, label=T["prod"][p]))
    return hs + list(extra)


def _band_xlim(D, site, bb):
    """0 to the outer edge of the site's last band (3 km bands): the whole
    channel, whichever bands hold truth."""
    c = list((D[site]["settings"].get("bands") or {}).get("centres_km") or [])
    if not c:
        c = [float(v) for v in bb["s_km_centre"].to_numpy(float) if np.isfinite(v)]
    return (0.0, (max(c) if c else 3.0) + 1.5)


def fig_bands_classes(D, lang):
    """Analysis 1: bias and centred RMSE by band and by elevation third, one
    row per site."""
    plt = _setup_mpl()
    T = TXT[lang]
    sites = list(SITE_ORDER)
    left, gap, widths = 0.78, 0.70, (3.35, 3.35, 2.75, 2.75)
    head_in, ax_h, below_in, top_in = 0.70, 1.85, 0.66, 1.62
    W = left + sum(widths) + gap * 3 + 0.22
    foot, foot_in = _wrap_foot(T["foot1"].format(rtk=MIN_N["rtk"], px=MIN_N["vaklodingen"]), W - left - 0.3)
    H = top_in + len(sites) * (head_in + ax_h + below_in) + foot_in
    sh = _Sheet(plt, W, H)
    present = set()
    xs = [left + sum(widths[:j]) + gap * j for j in range(4)]
    for i, site in enumerate(sites):
        y0 = top_in + i * (head_in + ax_h + below_in)
        _site_head(sh, left, y0, D, site, T)
        axs = [sh.axes(xs[j], y0 + head_in, widths[j], ax_h) for j in range(4)]
        axs[2].sharey(axs[0])
        axs[3].sharey(axs[1])
        for j, ax in enumerate(axs):
            v14._style_axes(ax)
            ax.set_title(T["cols1"][j], loc="left", fontsize=9.5, color=PAL["slate"])
        if site not in D:
            for ax in axs:
                _empty(ax, T)
            continue
        bb = D[site]["by_band"]
        bc = D[site]["by_class"]
        prods = [p for p in COMPARED if p in set(bb["product"])]
        present |= set(prods)
        absent = [p for p in COMPARED if p not in prods]
        st_p = D[site]["settings"].get("status_products", {})
        if absent:
            sh.text(xs[2], y0 + 0.24, "; ".join(f"{T['prod'][p]}: "
                                                f"{T['nc'] if str(st_p.get(p, '')).upper().startswith('NOT') else '--'}"
                                                for p in absent), fontsize=9.0, color=PAL["slate"])
        bb = bb[bb["band"].astype(str) != "all"].copy()
        bb["band_i"] = bb["band"].astype(int)
        for p in prods:
            g = bb[(bb["product"] == p) & (bb["band_i"] >= 0)].sort_values("s_km_centre")
            stl = STYLE[p]
            x = g["s_km_centre"].to_numpy(float)
            for ax, col in ((axs[0], "bias_m"), (axs[1], "rmse_centred_m")):
                ax.plot(x, g[col].to_numpy(float), color=stl["color"], ls=stl["ls"], lw=1.8, marker=stl["marker"],
                        ms=6.5, mfc=stl["mfc"], mec=stl["color"], mew=1.3, zorder=3)
        cl = bc[bc["class"].isin(CLASS_KEYS)]
        offs = {p: (k - (len(prods) - 1) / 2.0) * 0.16 for k, p in enumerate(prods)}
        for p in prods:
            g = cl[cl["product"] == p].set_index("class").reindex(CLASS_KEYS)
            stl = STYLE[p]
            x = np.arange(3) + offs[p]
            for ax, col in ((axs[2], "bias_m"), (axs[3], "rmse_centred_m")):
                ax.plot(x, g[col].to_numpy(float), color=stl["color"], ls=stl["ls"], lw=1.5, marker=stl["marker"],
                        ms=7, mfc=stl["mfc"], mec=stl["color"], mew=1.3, zorder=3)
        for ax in (axs[0], axs[2]):
            ax.axhline(0.0, color=PAL["ink"], lw=0.8, zorder=2)
        for ax in (axs[1], axs[3]):
            ax.set_ylim(bottom=0.0)
        for ax in (axs[0], axs[1]):
            ax.set_xlim(*_band_xlim(D, site, bb))
            ax.set_xlabel(T["xband"], fontsize=8.8, color=PAL["ink"])
        g0 = cl[cl["product"] == prods[0]].set_index("class").reindex(CLASS_KEYS) if prods else None
        labels = []
        for k in range(3):
            lo = g0["z_lo_m"].iloc[k] if g0 is not None else np.nan
            hi = g0["z_hi_m"].iloc[k] if g0 is not None else np.nan
            labels.append(f"{T['classes'][k]}\n{_num(lo, 2, lang=lang)}…{_num(hi, 2, lang=lang)}")
        for ax in (axs[2], axs[3]):
            ax.set_xticks(range(3))
            ax.set_xticklabels(labels, fontsize=8.3, color=PAL["ink"])
            ax.set_xlim(-0.55, 2.55)
            ax.set_xlabel(T["xclass"], fontsize=8.8, color=PAL["ink"])
        for ax in axs:
            _comma_axis(ax, lang)
    sh.text(left, 0.28, T["t1"], fontsize=15, color=PAL["deep"], weight="bold")
    sh.text(left, 0.70, T["s1"], fontsize=10, color=PAL["slate"])
    hs = _legend_handles(T, [p for p in COMPARED if p in present] or list(COMPARED))
    sh.legend(hs, left - 0.08, 1.02, ncol=3, fontsize=10, handlelength=2.6, columnspacing=2.0)
    sh.text(left, H - foot_in + 0.12, foot, fontsize=8.6, color=PAL["slate"], linespacing=1.35)
    return sh.fig


def _wrap_foot(text, width_in, fontsize=8.6):
    """The footnote wrapped to ``width_in`` inches (about 15 characters per
    inch at 8.6 pt; paragraphs kept) and the height it needs (in)."""
    import textwrap

    per_in = 15.0 * 8.6 / float(fontsize)
    width = max(40, int(width_in * per_in))
    lines = []
    for par in str(text).split("\n"):
        lines += textwrap.wrap(par, width=width, break_on_hyphens=False) or [""]
    return "\n".join(lines), 0.26 + 0.168 * len(lines) * float(fontsize) / 8.6


def fig_calibration(D, lang):
    """Analysis 2: reliability curves (top) and shares within 1 and 2 sigma
    with Spearman rho and k68 (bottom), one column per site."""
    plt = _setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    T = TXT[lang]
    sites = list(SITE_ORDER)
    left, col_w, gap = 1.80, 3.25, 0.62
    top_in, head_in, rel_h, mid_in, cov_h = 1.62, 0.74, 3.0, 0.78, 1.75
    W = left + len(sites) * col_w + (len(sites) - 1) * gap + 0.25
    foot, foot_in = _wrap_foot(T["foot2"].format(rtk=MIN_N["rtk"], px=MIN_N["vaklodingen"],
                                                 st=_sigma_t_text(D, T, lang)), W - left - 0.3, fontsize=8.5)
    foot_in += 0.62                          # the x tick labels and label of the bottom row above it
    H = top_in + head_in + rel_h + mid_in + cov_h + foot_in
    sh = _Sheet(plt, W, H)
    present = set()
    order = list(COMPARED)
    ys = {p: len(order) - 1 - i for i, p in enumerate(order)}
    for j, site in enumerate(sites):
        x0 = left + j * (col_w + gap)
        _site_head(sh, x0, top_in, D, site, T, two_lines=True)
        ar = sh.axes(x0, top_in + head_in, rel_h, rel_h)
        ac = sh.axes(x0, top_in + head_in + rel_h + mid_in, col_w, cov_h)
        for ax in (ar, ac):
            v14._style_axes(ax)
        ac.set_ylim(-0.6, len(order) - 0.25)
        ac.set_yticks([ys[p] for p in order])
        ac.set_yticklabels([T["short"][p] for p in order] if j == 0 else [""] * len(order), fontsize=9.0,
                           color=PAL["ink"])
        ac.set_xlim(0, 100)
        ac.set_xlabel(T["covx"], fontsize=9.3, color=PAL["ink"])
        ar.set_xlabel(T["relx"], fontsize=9.3, color=PAL["ink"])
        if j == 0:
            ar.set_ylabel(T["rely"], fontsize=9.3, color=PAL["ink"])
        if site not in D:
            _empty(ar, T)
            ac.text(0.5, 0.5, T["nores"], transform=ac.transAxes, ha="center", va="center", fontsize=11,
                    color=PAL["slate"])
            continue
        k = _kind(D, site)
        rel, cal = D[site]["reliability"], D[site]["calibration"]
        lo_v, hi_v = [], []
        for p in order:
            g = rel[(rel["product"] == p) & (rel["n"] >= MIN_N[k])].sort_values("sigma_median_m") if len(rel) else rel
            if not len(g):
                continue
            present.add(p)
            stl = STYLE[p]
            x, y = g["sigma_median_m"].to_numpy(float), g["q68_abs_e_m"].to_numpy(float)
            ar.plot(x, y, color=stl["color"], ls=stl["ls"], lw=1.6, marker=stl["marker"], ms=6.5, mfc=stl["mfc"],
                    mec=stl["color"], mew=1.3, zorder=3)
            lo_v += [x.min(), y.min()]
            hi_v += [x.max(), y.max()]
        if lo_v:
            lo, hi = max(min(lo_v) / 1.5, 0.005), max(hi_v) * 1.5
            ar.set_xscale("log")
            ar.set_yscale("log")
            ar.set_xlim(lo, hi)
            ar.set_ylim(lo, hi)
            ar.plot([lo, hi], [lo, hi], color=PAL["slate"], lw=1.0, ls=(0, (4, 3)), zorder=2)
            ticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0) if lo <= t <= hi]
            for axis in (ar.xaxis, ar.yaxis):
                axis.set_major_locator(FixedLocator(ticks))
                axis.set_major_formatter(FuncFormatter(
                    lambda v, _p: f"{v:g}".replace(".", ",") if lang == "es" else f"{v:g}"))
                axis.set_minor_locator(NullLocator())
        for gref in (GAUSS_1, GAUSS_2):
            ac.axvline(100 * gref, color=PAL["slate"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        for p in order:
            yy = ys[p]
            r = cal[cal["product"] == p] if len(cal) else cal
            if not len(r) or r.iloc[0].get("status") != "ok":
                st = str(r.iloc[0].get("status")) if len(r) else "--"
                ac.text(1.5, yy, T["nc"] if st.upper().startswith("NOT") else st, fontsize=8.6, color=PAL["slate"],
                        va="center", ha="left")
                continue
            r = r.iloc[0]
            stl = STYLE[p]
            w1, w2 = 100 * float(r["within_1sigma"]), 100 * float(r["within_2sigma"])
            ac.plot([w1, w2], [yy, yy], color=stl["color"], lw=1.4, zorder=2)
            ac.plot([w1], [yy], ls="none", marker="o", ms=7.5, color=stl["color"], mec="white", mew=1.2, zorder=3)
            ac.plot([w2], [yy], ls="none", marker="D", ms=6.5, color=stl["color"], mec="white", mew=1.2, zorder=3)
            # against sqrt(sigma^2 + sigma_t^2): the truth's own error added (hollow)
            t1, t2 = (float(r.get(c, np.nan)) for c in ("within_1sigma_tsite", "within_2sigma_tsite"))
            if np.isfinite(t1) and np.isfinite(t2):
                yt = yy - 0.22
                ac.plot([100 * t1, 100 * t2], [yt, yt], color=stl["color"], lw=0.9, ls=(0, (2, 2)), zorder=2)
                ac.plot([100 * t1], [yt], ls="none", marker="o", ms=6.5, mfc="white", mec=stl["color"], mew=1.3,
                        zorder=3)
                ac.plot([100 * t2], [yt], ls="none", marker="D", ms=5.8, mfc="white", mec=stl["color"], mew=1.3,
                        zorder=3)
            slope_w = "slope" if lang == "en" else "pendiente"
            ac.text(1.5, yy + 0.30, f"ρ {_num(r['spearman_rho'], 2, lang=lang)} "
                                    f"({_num(r.get('spearman_rho_thinned'), 2, lang=lang)}) · "
                                    f"k68 {_num(r['k68'], 2, lang=lang)} · "
                                    f"{slope_w} {_num(r.get('reliability_slope'), 2, lang=lang)}",
                    fontsize=8.2, color=PAL["ink"], va="bottom", ha="left")
    sh.text(left, 0.25, T["t2"], fontsize=15, color=PAL["deep"], weight="bold")
    sh.text(left, 0.64, T["s2"], fontsize=9.6, color=PAL["slate"], linespacing=1.35)
    extra = [Line2D([], [], color=PAL["slate"], lw=1.0, ls=(0, (4, 3)), label=f"{T['one']} · {T['gauss']}"),
             Line2D([], [], ls="none", marker="o", ms=7, color=PAL["slate"], mec="white", label=T["w1"]),
             Line2D([], [], ls="none", marker="D", ms=6, color=PAL["slate"], mec="white", label=T["w2"]),
             Line2D([], [], ls="none", marker="o", ms=6.5, mfc="white", mec=PAL["slate"], mew=1.3, label=T["w_t"])]
    hs = _legend_handles(T, [p for p in COMPARED if p in present] or list(COMPARED), extra)
    sh.legend(hs, left - 0.08, 1.12, ncol=4, fontsize=9.4, handlelength=2.4, columnspacing=1.5)
    sh.text(left, H - foot_in + 0.66, foot, fontsize=8.5, color=PAL["slate"], linespacing=1.35)
    return sh.fig


def _sigma_t_text(D, T, lang):
    """The truth errors used for the hollow markers, in words."""
    parts = []
    for site in SITE_ORDER:
        if site not in D:
            continue
        te = D[site]["settings"].get("truth_error") or {}
        v = te.get("sigma_t_m")
        if v is None:
            continue
        key = "st_rtk" if _kind(D, site) == "rtk" else "st_vak"
        txt = T[key].format(v=_num(v, 3 if key == "st_rtk" else 2, lang=lang))
        if txt not in parts:
            parts.append(txt)
    return "; ".join(parts) if parts else "--"


def fig_totals(D, lang):
    """Analysis 3: centred (ring) and total (dot) RMSE on the common pixels,
    with the bias, one panel per site."""
    plt = _setup_mpl()
    from matplotlib.lines import Line2D
    T = TXT[lang]
    sites = list(SITE_ORDER)
    order = list(PRODUCTS)                    # MAREA, NIDEM, NIDEM (MAREA's), no delay
    left, col_w, gap = 2.75, 3.45, 0.32
    top_in, head_in, ax_h, foot_in = 1.12, 0.74, 2.55, 1.2
    W = left + len(sites) * col_w + (len(sites) - 1) * gap + 0.25
    H = top_in + head_in + ax_h + foot_in
    sh = _Sheet(plt, W, H)
    xmax = 0.0
    for site in sites:
        if site in D and len(D[site]["totals"]):
            t = D[site]["totals"]
            c = t[(t["subset"] == "common")]
            if len(c):
                xmax = max(xmax, float(np.nanmax(c[["rmse_centred_m", "rmse_total_m"]].to_numpy(float))))
    xmax = (xmax or 1.0) * 1.45
    ys = {p: len(order) - 1 - i for i, p in enumerate(order)}
    for j, site in enumerate(sites):
        x0 = left + j * (col_w + gap)
        _site_head(sh, x0, top_in, D, site, T, two_lines=True)
        ax = sh.axes(x0, top_in + head_in, col_w, ax_h)
        v14._style_axes(ax)
        ax.grid(False, axis="y")
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_yticks([ys[p] for p in order])
        ax.set_yticklabels([T["prod"][p] for p in order] if j == 0 else [""] * len(order), fontsize=9.2,
                           color=PAL["ink"])
        ax.set_xlim(0, xmax)
        ax.set_xlabel(T["x3"], fontsize=9.5, color=PAL["ink"])
        _comma_axis(ax, lang, ("x",))
        if site not in D:
            ax.text(0.5, 0.5, T["nores"], transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    color=PAL["slate"])
            continue
        t = D[site]["totals"]
        c = t[(t["subset"] == "common")]
        for p in order:
            r = c[c["product"] == p]
            y = ys[p]
            if r.empty or not np.isfinite(float(r.iloc[0]["rmse_total_m"])):
                st = t[(t["product"] == p) & (t["subset"] == "all")]
                why = T["nc"] if (len(st) and str(st.iloc[0].get("status", "")).upper().startswith("NOT")) else "--"
                ax.text(0.02 * xmax, y, why, fontsize=8.8, color=PAL["slate"], va="center", ha="left")
                continue
            r = r.iloc[0]
            stl = STYLE[p]
            cen, tot, b = float(r["rmse_centred_m"]), float(r["rmse_total_m"]), float(r["bias_m"])
            ax.plot([cen, tot], [y, y], color=stl["color"], lw=2.0, solid_capstyle="round", zorder=2)
            ax.plot([tot], [y], ls="none", marker="o", ms=6.5, color=stl["color"], mec="white", mew=1.0, zorder=3)
            ax.plot([cen], [y], ls="none", marker="o", ms=11, mfc="none", mec=stl["color"], mew=1.6, zorder=4)
            ax.text(max(cen, tot) + 0.035 * xmax, y, f"{T['bias']} {_num(b, 2, sign=True, lang=lang)} m",
                    fontsize=8.6, color=PAL["ink"], va="center", ha="left")
    sh.text(left, 0.25, T["t3"], fontsize=15, color=PAL["deep"], weight="bold")
    sh.text(left, 0.64, T["s3"], fontsize=9.6, color=PAL["slate"])
    hs = [Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=PAL["slate"], mew=1.6, label=T["centred"]),
          Line2D([], [], ls="none", marker="o", ms=6.5, color=PAL["slate"], mec="white", label=T["total"])]
    sh.legend(hs, W - 4.2, 0.22, ncol=2, fontsize=9.6)
    sh.text(left, H - foot_in + 0.52, _wrap_foot(T["foot3"], W - left - 0.3, fontsize=8.5)[0], fontsize=8.5,
            color=PAL["slate"], linespacing=1.35)
    return sh.fig


FIGURES = {"bands": fig_bands_classes, "calibration": fig_calibration, "totals": fig_totals}


def md_report(D, missing, gates, lang, out, boot=None):
    """The markdown tables of the analyses (and the gates; the paired
    bootstrap when ``boot`` holds its rows)."""
    T = TXT[lang]

    def num(v, d=3, sign=False):
        return _num(v, d, sign, lang)

    L = [f"# {T['md_title']}", "",
         T["md_gen"].format(date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), out=out), "",
         T["md_intro"], ""]
    # gates
    L += [f"## {T['md_gates']}", "", "| " + " | ".join(T["h_gate"]) + " |", "|---|---|---|---|"]
    for site in SITE_ORDER:
        for k, v in (gates.get(site) or {}).items():
            det = {kk: vv for kk, vv in v.items() if kk in ("max_abs_diff", "validity_flips", "max_abs_diff_m",
                                                             "n_sensitivity", "n_marea", "reason",
                                                             "n_common_without_mouth_distance")}
            det = ", ".join(f"{kk} {(f'{vv:.1e}' if isinstance(vv, float) else vv)}" for kk, vv in det.items())
            L.append(f"| {T['site'].get(site, site)} | {k} | {T['ok'] if v.get('ok', True) else T['failed']} | "
                     f"{det} |")
    L.append("")
    # 1
    L += [f"## {T['md_s1']}", "", T["md_s1_note"], ""]
    clamp_txt = []
    for site in SITE_ORDER:
        if site not in D:
            continue
        st = D[site]["settings"]
        bb, bc = D[site]["by_band"], D[site]["by_class"]
        prods = [p for p in PRODUCTS if p in set(bb["product"])]
        L += [f"### {T['site'].get(site, site)}", ""]
        nC = st.get("sets", {}).get("common")
        ref_n = st.get("reference_on_common")
        if ref_n is not None and nC:
            L += [T["ref_note"].format(n=_int(ref_n, lang), N=_int(nC, lang)), ""]
        L += [f"**{T['md_band']}**", "",
              "| " + " | ".join(T["h_band"]) + " | " + " | ".join(T["prod"][p] for p in prods) + " |",
              "|" + "---|" * (len(T["h_band"]) + len(prods))]
        ref = bb[bb["product"] == prods[0]]
        for _, r0 in ref.iterrows():
            key = str(r0["band"])
            cells = []
            for p in prods:
                r = bb[(bb["product"] == p) & (bb["band"].astype(str) == key)]
                r = r.iloc[0] if len(r) else None
                cells.append("--" if r is None else
                             f"{num(r['bias_m'], 3, True)} / {num(r['rmse_centred_m'])} / {num(r['rmse_total_m'])}")
            label = T["all"] if key == "all" else T["no_mouth"] if key == str(NO_MOUTH_BAND) else key
            L.append(f"| {label} | {num(r0['s_km_centre'], 1)} | "
                     f"{num(r0['tau_applied_min'], 1)} | {_int(r0['n_set'], lang)} | " + " | ".join(cells) + " |")
        L += ["", f"**{T['md_class']}**", "",
              "| " + " | ".join(T["h_class"]) + " | " + " | ".join(T["prod"][p] for p in prods) + " |",
              "|" + "---|" * (len(T["h_class"]) + len(prods))]
        ref = bc[bc["product"] == prods[0]]
        for _, r0 in ref.iterrows():
            key = str(r0["class"])
            name = (T["classes"][CLASS_KEYS.index(key)] if key in CLASS_KEYS else T["all"] if key == "all" else key)
            cells = []
            for p in prods:
                r = bc[(bc["product"] == p) & (bc["class"].astype(str) == key)]
                r = r.iloc[0] if len(r) else None
                cells.append("--" if r is None else
                             f"{num(r['bias_m'], 3, True)} / {num(r['rmse_centred_m'])} / {num(r['rmse_total_m'])}")
            L.append(f"| {name} | {num(r0['z_lo_m'], 2)} … {num(r0['z_hi_m'], 2)} | {_int(r0['n_set'], lang)} | "
                     + " | ".join(cells) + " |")
        up = bc[bc["class"].astype(str) == "upper"]
        if "share_at_upper_clamp" in up.columns:
            parts = [f"{T['short'][r['product']]} {_pct(r['share_at_upper_clamp'], lang)}"
                     for _, r in up.iterrows() if np.isfinite(float(r.get("share_at_upper_clamp", np.nan)))]
            if parts:
                clamp_txt.append(f"{T['site'].get(site, site)}: " + ", ".join(parts))
        L.append("")
    if clamp_txt:
        L += [T["md_clamp_note"].format(txt="; ".join(clamp_txt)), ""]
    # 2
    L += [f"## {T['md_s2']}", "", T["md_s2_note"].format(st=_sigma_t_text(D, T, lang)), "",
          "| " + " | ".join(T["h_cal"]) + " |", "|" + "---|" * len(T["h_cal"])]
    verdicts = []
    for site in SITE_ORDER:
        if site not in D:
            continue
        for _, r in D[site]["calibration"].iterrows():
            if r.get("status") != "ok":
                L.append(f"| {T['site'].get(site, site)} | {T['prod'][r['product']]} | -- | "
                         f"{T['nc'] if str(r.get('status')).upper().startswith('NOT') else r.get('status')} |"
                         + " -- |" * (len(T["h_cal"]) - 4))
                continue
            g = r.get
            L.append(f"| {T['site'].get(site, site)} | {T['prod'][r['product']]} | {_int(r['n'], lang)} | "
                     f"{num(r['spearman_rho'], 2)} ({num(g('spearman_rho_thinned'), 2)}) | "
                     f"{num(g('reliability_slope'), 2)} | {num(r['bias_removed_m'], 3, True)} | "
                     f"{num(r['sigma_median_m'])} | {_pct(r['within_1sigma'], lang)} | "
                     f"{_pct(r['within_2sigma'], lang)} | {num(r['k68'], 2)} | {num(r['z_rms'], 2)} | "
                     f"{num(r['ence'], 2)} | {num(g('sigma_t_site_m'), 3)} | {_pct(g('within_1sigma_tsite'), lang)} | "
                     f"{_pct(g('within_2sigma_tsite'), lang)} | {num(g('k68_tsite'), 2)} |")
            slope = float(g("reliability_slope", np.nan))
            lo_ok, hi_ok = RELIABILITY_OK
            vkey = ("few" if not np.isfinite(slope) else "follows" if lo_ok <= slope <= hi_ok else "not")
            verdicts.append(T["md_verdict"].format(
                site=T["site"].get(site, site), prod=T["prod"][r["product"]], rho=num(r["spearman_rho"], 2),
                rho_t=num(g("spearman_rho_thinned"), 2), verdict=T["verdicts"][vkey].format(s=num(slope, 2))))
    L.append("")
    if verdicts:
        L += [f"* {v}" for v in verdicts] + [""]
    # 3
    L += [f"## {T['md_s3']}", "", T["md_s3_note"], "", "| " + " | ".join(T["h_tot"]) + " |",
          "|" + "---|" * len(T["h_tot"])]
    for site in SITE_ORDER:
        if site not in D:
            continue
        t = D[site]["totals"]
        for sub in SUBSETS:
            for p in PRODUCTS:
                r = t[(t["subset"] == sub) & (t["product"] == p)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if str(r.get("status")) != "ok":
                    L.append(f"| {T['site'].get(site, site)} | {T['prod'][p]} | {sub} | "
                             f"{T['nc'] if str(r.get('status')).upper().startswith('NOT') else r.get('status')} |"
                             + " -- |" * (len(T["h_tot"]) - 4))
                    continue
                L.append(f"| {T['site'].get(site, site)} | {T['prod'][p]} | {sub} | {_int(r['scenes'], lang)} | "
                         f"{_int(r['n'], lang)} | {num(r['rmse_centred_m'])} | {num(r['rmse_total_m'])} | "
                         f"{num(r['bias_m'], 3, True)} | {num(r['mean_diff_m'], 3, True)} | {num(r['sd_diff_m'])} | "
                         f"{num(r['slope'], 2)} | {num(r['r'], 2)} |")
    L.append("")
    # 4
    if boot is not None and len(boot):
        rules = sorted({str(x) for x in boot["cluster_rule"].dropna()})
        nb = sorted({int(x) for x in boot["n_boot"].dropna()})
        L += [f"## {T['md_s4']}", "", T["md_s4_note"].format(rules="; ".join(rules),
                                                            n_boot="/".join(_int(x, lang) for x in nb)), "",
              "| " + " | ".join(T["h_boot"]) + " |", "|" + "---|" * len(T["h_boot"])]
        for site in SITE_ORDER:
            for _, r in boot[boot["site"] == site].iterrows():
                st = str(r["stat"])
                if st.startswith("within"):
                    def fa(v):
                        return _pct(v, lang)

                    def fd(v):
                        return _num(100 * float(v), 1, True, lang) + " pp"
                else:
                    def fa(v):
                        return num(v, 3)

                    def fd(v):
                        return num(v, 3, True)
                ind = " *" if bool(r.get("indicative")) else ""
                L.append(f"| {T['site'].get(site, site)}{ind} | {T['short'][r['a']]} − {T['short'][r['b']]} | "
                         f"{T['boot_stat'].get(st, st)} | {_int(r['n_units'], lang)} | "
                         f"{_int(r['n_clusters'], lang)} | {fa(r['a_value'])} | {fa(r['b_value'])} | "
                         f"{fd(r['diff'])} | [{fd(r['ci_lo'])}, {fd(r['ci_hi'])}] | "
                         f"{_pct(r['share_diff_below_0'], lang)} |")
        L.append("")
    if missing:
        L += [f"## {T['md_missing']}", ""] + [f"* {T['site'].get(s, s)}: {why}" for s, why in missing.items()] + [""]
    return "\n".join(L)


def report(out=OUT, sites=None, dpi=200, formats=("png", "pdf"), langs=("es", "en")):
    """Combined CSVs, the markdown in both languages and the three figures in
    both languages (the bootstrap section when <site>_bootstrap.csv exist)."""
    import pandas as pd

    v14.guard_out(out)
    D, missing = load_report_inputs(out, sites)
    if not D:
        raise SystemExit(f"no completed site in {out} ({missing})")
    for k in ("by_band", "by_class", "calibration", "reliability", "totals"):
        frames = [d[k] for d in D.values() if len(d[k])]
        if frames:
            write_csv(pd.concat(frames, ignore_index=True), os.path.join(out, f"truth_{k}.csv"))
    gates = {s: d["settings"].get("gates", {}) for s, d in D.items()}
    rows = [{"site": s, "gate": k, "ok": v.get("ok", True), **{kk: vv for kk, vv in v.items()
                                                              if isinstance(vv, (int, float, str, bool))}}
            for s, g in gates.items() for k, v in g.items()]
    write_csv(pd.DataFrame(rows), os.path.join(out, "truth_gates.csv"))
    bframes = []
    for s in D:
        bp = os.path.join(out, f"{s}_bootstrap.csv")
        if os.path.exists(bp) and os.path.getsize(bp) > 1:
            bframes.append(pd.read_csv(bp))
    boot = pd.concat(bframes, ignore_index=True) if bframes else None
    if boot is not None:
        write_csv(boot, os.path.join(out, "truth_bootstrap.csv"))
    written = []
    for lang in langs:
        name = "truth_diagnostics.md" if lang == "en" else f"truth_diagnostics.{lang}.md"
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(md_report(D, missing, gates, lang, out, boot) + "\n")
        written.append(name)
    plt = _setup_mpl()
    for key, fn in FIGURES.items():
        for lang in langs:
            fig = fn(D, lang)
            stem = os.path.join(out, f"{FIG_STEMS[key]}_{lang}")
            for ext in formats:
                fig.savefig(f"{stem}.{ext}", dpi=dpi)
                written.append(os.path.basename(f"{stem}.{ext}"))
            plt.close(fig)
    log(f"report: sites {sorted(D)} (missing {missing}; bootstrap for "
        f"{sorted(set(boot['site'])) if boot is not None else []}) -> {out}/: {', '.join(written)}")
    return D, missing


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

OVERRIDE_KEYS = ("extract", "product", "comparison", "nidem", "nidem_sensitivity", "external", "cube", "published")


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m experiments.v17_truth_diagnostics",
                                 description="Truth-side diagnostics of the published 10 m MAREA and NIDEM.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one site: gates, then the tables")
    r.add_argument("site", choices=list(SITE_ORDER))
    r.add_argument("--out", default=OUT)
    r.add_argument("--soft-gates", action="store_true",
                   help="a failed gate is recorded in the JSON instead of stopping (only after reading why)")
    r.add_argument("--extract", default=None, help="extraction npz (keep, shape)")
    r.add_argument("--marea-dir", dest="product", default=None,
                   help="production MAREA folder with marea.npz and result.json")
    r.add_argument("--comparison-dir", dest="comparison", default=None, help="Dutch comparison_<tag> folder")
    r.add_argument("--nidem-dir", dest="nidem", default=None, help="NIDEM primary folder (sensitivity inside)")
    r.add_argument("--nidem-sensitivity-dir", dest="nidem_sensitivity", default=None)
    r.add_argument("--external", default=None, help="Villaviciosa external_products.npz")
    r.add_argument("--cube", default=None, help="Dutch cube (grid of the truth sampling)")
    r.add_argument("--published-csv", dest="published", default=None,
                   help="the same-scenes CSV of v9 (default results/method_comparison_same_scenes/<site>.csv)")
    b = sub.add_parser("bootstrap", help="paired block bootstrap of MAREA - NIDEM (and - no delay) per site")
    b.add_argument("--out", default=OUT)
    b.add_argument("--sites", nargs="*", default=None, help="default: every site with a <site>_units.npz")
    b.add_argument("--n-boot", type=int, default=pbs.N_BOOT)
    b.add_argument("--seed", type=int, default=pbs.SEED)
    b.add_argument("--block-m", type=float, default=pbs.BLOCK_M, help="block side at the Dutch sites (m)")
    p = sub.add_parser("report", help="combined tables, markdown and figures from the per-site outputs")
    p.add_argument("--out", default=OUT)
    p.add_argument("--sites", nargs="*", default=None)
    p.add_argument("--dpi", type=int, default=200)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        ov = {k: getattr(args, k, None) for k in OVERRIDE_KEYS}
        run_site(args.site, out=args.out, overrides=ov, soft_gates=args.soft_gates)
    elif args.cmd == "bootstrap":
        sites = args.sites or [s for s in SITE_ORDER if os.path.exists(os.path.join(args.out, f"{s}_units.npz"))]
        if not sites:
            raise SystemExit(f"no <site>_units.npz in {args.out}: run the sites first")
        for site in sites:
            bootstrap_site(site, out=args.out, n_boot=args.n_boot, seed=args.seed, block_m=args.block_m)
    else:
        report(out=args.out, sites=args.sites, dpi=args.dpi)


if __name__ == "__main__":
    main()
