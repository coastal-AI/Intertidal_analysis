# -*- coding: utf-8 -*-
"""Paired cluster bootstrap of the published elevation metrics.

Shared by experiments/v16_window_curves.py (one window against W3, refit
against fixed lags) and experiments/v17_truth_diagnostics.py (MAREA against
NIDEM on the truth side).

Two estimates ``a`` and ``b`` of the SAME units (RTK points or pixels)
against one truth ``t`` are resampled together. The units are grouped in
clusters: an RTK point goes with the pixel that holds it (points sharing a
pixel are not independent), a pixel goes with its 1 km block of the image
grid (most of the spatial correlation of the errors stays inside a block).
Every resample draws as many clusters as there are, with replacement; the
SAME draw is used for ``a`` and ``b`` (paired), so what varies is the
difference between the two methods on a resampled site, not each method
alone.

Metrics, as published (experiments/v7_waterline_contour.vscore and
pyintertidal.elevation_validation.ols_against_truth): the median-centred
RMSE (product and truth each centred by its own median), the OLS slope of
the centred product on the centred truth, Pearson r, the bias median(p - t);
and also the total RMSE sqrt(mean((p - t)^2)), |bias|, the share of units
with an estimate (coverage) and the share of |e| <= sigma with
e = p - t - median(p - t) (calibration of a stated uncertainty).

A resample is a multiset of units, so each median is its exact median (the
two middle order statistics averaged, as numpy.median), from one argsort per
variable and the cluster multiplicities; means are weighted sums. The point
estimates use the published formulas on the units as they are; the interval
is the percentile interval of the replicate differences a - b.
"""
from __future__ import annotations

import zlib

import numpy as np

N_BOOT = 2000
SEED = 20261020
BLOCK_M = 1000.0
LEVEL = 0.95
#: minimum units of the published metrics (ols_against_truth 20, vscore 30)
MIN_N = {"rtk": 20, "vaklodingen": 30}


# ─────────────────────────────────────────────────────────────────────────────
#  clusters and draws
# ─────────────────────────────────────────────────────────────────────────────

def cluster_labels(kind, rows, cols, pixel_m=10.0, block_m=BLOCK_M):
    """A cluster label per unit and the rule in words. ``kind`` 'rtk': the
    pixel holding the point; otherwise the ``block_m`` block of the image
    grid (``rows // b``, ``cols // b`` with b = block_m / pixel_m pixels)."""
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    if kind == "rtk":
        span = int(cols.max()) + 1 if len(cols) else 1
        return rows * span + cols, "the pixel holding each RTK point (points sharing a pixel move together)"
    b = max(1, int(round(float(block_m) / float(pixel_m))))
    rb, cb = rows // b, cols // b
    span = int(cb.max()) + 1 if len(cb) else 1
    return rb * span + cb, (f"{float(block_m) / 1000:g} km blocks of the image grid ({b} x {b} pixels of "
                            f"{float(pixel_m):g} m)")


def cluster_index(labels):
    """(index 0..B-1 per unit, B) of arbitrary cluster labels."""
    u, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64).ravel(), int(len(u))


def resample_counts(n_clusters, n_boot=N_BOOT, seed=SEED, key=""):
    """(n_boot, n_clusters) multiplicities: each row draws ``n_clusters``
    clusters with replacement. Reproducible from (seed, key, n_clusters)."""
    B = int(n_clusters)
    salt = zlib.crc32(str(key).encode("utf-8")) & 0x7FFFFFFF
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), salt, B]))
    out = np.empty((int(n_boot), B), np.int32)
    for i in range(int(n_boot)):
        out[i] = np.bincount(rng.integers(0, B, B), minlength=B)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  weighted (multiset) statistics
# ─────────────────────────────────────────────────────────────────────────────

class WeightedMedian:
    """Exact median of the multiset in which unit i appears k[cl[i]] times
    (numpy.median of the resampled values, without building them)."""

    def __init__(self, x, cl):
        x = np.asarray(x, float)
        o = np.argsort(x, kind="stable")
        self.x = x[o]
        self.cl = np.asarray(cl, np.int64)[o]

    def __call__(self, k):
        if not len(self.x):
            return np.nan
        c = np.cumsum(np.asarray(k)[self.cl], dtype=np.int64)
        N = int(c[-1])
        if N <= 0:
            return np.nan
        i0, i1 = np.searchsorted(c, [(N - 1) // 2 + 1, N // 2 + 1])
        return 0.5 * (self.x[i0] + self.x[i1])


class MetricEngine:
    """Replicates of the published metrics of one product ``p`` against the
    truth ``t`` on fixed units (both finite on every unit) in clusters ``cl``
    (0..B-1). ``med_t`` may be shared between products on the same units."""

    def __init__(self, t, p, cl, B, med_t=None):
        t, p = np.asarray(t, float), np.asarray(p, float)
        if not (np.isfinite(t).all() and np.isfinite(p).all()):
            raise ValueError("MetricEngine needs finite truth and product on every unit")
        self.cl, self.B = np.asarray(cl, np.int64), int(B)
        self.d = p - t
        self.med_t = med_t if med_t is not None else WeightedMedian(t, self.cl)
        self.med_p = WeightedMedian(p, self.cl)
        self.med_d = WeightedMedian(self.d, self.cl)
        tc = t - (t.mean() if len(t) else 0.0)
        pc = p - (p.mean() if len(p) else 0.0)

        def bc(w=None):
            return np.bincount(self.cl, weights=w, minlength=self.B).astype(float)
        self.n_b = bc()
        self.s_t, self.s_p = bc(tc), bc(pc)
        self.s_tt, self.s_pp, self.s_tp = bc(tc * tc), bc(pc * pc), bc(tc * pc)
        self.s_dd = bc(self.d * self.d)

    def __call__(self, k):
        kf = np.asarray(k, float)
        S = float(kf @ self.n_b)
        nan = dict(rmse=np.nan, rmse_total=np.nan, bias=np.nan, abs_bias=np.nan, slope=np.nan, r=np.nan, n=S)
        if S <= 0:
            return nan
        St, Sp = kf @ self.s_t, kf @ self.s_p
        vt = kf @ self.s_tt - St * St / S
        vp = kf @ self.s_pp - Sp * Sp / S
        ctp = kf @ self.s_tp - St * Sp / S
        mt, mp, b = self.med_t(k), self.med_p(k), self.med_d(k)
        w = np.asarray(k)[self.cl]
        e = self.d - (mp - mt)
        return dict(rmse=float(np.sqrt(np.dot(w, e * e) / S)),
                    rmse_total=float(np.sqrt(max(kf @ self.s_dd, 0.0) / S)),
                    bias=float(b), abs_bias=float(abs(b)),
                    slope=float(ctp / vt) if vt > 0 else np.nan,
                    r=float(ctp / np.sqrt(vt * vp)) if vt > 0 and vp > 0 else np.nan, n=S)


class WithinSigmaEngine:
    """Replicates of the share of units with |p - t - median(p - t)| <= f*sigma
    (``sigma`` finite and > 0 on every unit)."""

    def __init__(self, t, p, sigma, cl, factor=1.0):
        self.d = np.asarray(p, float) - np.asarray(t, float)
        self.s = float(factor) * np.asarray(sigma, float)
        self.cl = np.asarray(cl, np.int64)
        self.med_d = WeightedMedian(self.d, self.cl)

    def __call__(self, k):
        w = np.asarray(k)[self.cl]
        S = float(w.sum())
        if S <= 0:
            return np.nan
        b = self.med_d(k)
        return float(np.dot(w, np.abs(self.d - b) <= self.s) / S)


class ShareEngine:
    """Replicates of the share of units whose flag is set (e.g. coverage:
    a finite estimate)."""

    def __init__(self, flag, cl, B):
        cl = np.asarray(cl, np.int64)
        self.n_b = np.bincount(cl, minlength=int(B)).astype(float)
        self.f_b = np.bincount(cl, weights=np.asarray(flag, float), minlength=int(B))

    def __call__(self, k):
        kf = np.asarray(k, float)
        S = float(kf @ self.n_b)
        return float(kf @ self.f_b / S) if S > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  point estimates (the published formulas) and the paired interval
# ─────────────────────────────────────────────────────────────────────────────

def point_metrics(t, p, kind="vaklodingen"):
    """The published metrics on the finite pairs, exactly as v7.vscore /
    ols_against_truth compute them (NaN below their minimum n), plus the
    total RMSE and |bias|."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(p)
    n = int(m.sum())
    out = dict(rmse=np.nan, rmse_total=np.nan, bias=np.nan, abs_bias=np.nan, slope=np.nan, r=np.nan, n=n)
    if n < MIN_N.get(kind, 30):
        return out
    pc, tc = p[m] - np.median(p[m]), t[m] - np.median(t[m])
    d = p[m] - t[m]
    b = float(np.median(d))
    out.update(rmse=float(np.sqrt(np.mean((pc - tc) ** 2))), rmse_total=float(np.sqrt(np.mean(d * d))),
               bias=b, abs_bias=abs(b), slope=float(np.polyfit(tc, pc, 1)[0]),
               r=float(np.corrcoef(tc, pc)[0, 1]))
    return out


def point_within(t, p, sigma, factor=1.0):
    """Share of |p - t - median(p - t)| <= factor * sigma (v17.calibrate)."""
    d = np.asarray(p, float) - np.asarray(t, float)
    if not len(d):
        return np.nan
    return float(np.mean(np.abs(d - np.median(d)) <= float(factor) * np.asarray(sigma, float)))


def replicate(counts, fns):
    """{name: callable(k) -> float or dict} over every row of ``counts``.
    Returns {name: array} (a dict-valued callable gives {name.key: array})."""
    out = {}
    for i in range(counts.shape[0]):
        k = counts[i]
        for name, fn in fns.items():
            v = fn(k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out.setdefault(f"{name}.{kk}", np.empty(counts.shape[0]))[i] = vv
            else:
                out.setdefault(name, np.empty(counts.shape[0]))[i] = v
    return out


def paired_interval(a, b, reps_a, reps_b, level=LEVEL):
    """Point difference a - b with the percentile interval of the replicate
    differences and the share of replicates below 0."""
    d = np.asarray(reps_a, float) - np.asarray(reps_b, float)
    ok = np.isfinite(d)
    q = (1.0 - float(level)) / 2.0
    if ok.sum() >= 2:
        lo, hi = (float(v) for v in np.quantile(d[ok], [q, 1.0 - q]))
        below = float(np.mean(d[ok] < 0))
    else:
        lo = hi = below = np.nan
    a, b = float(a), float(b)
    return dict(a=a, b=b, diff=a - b, ci_lo=lo, ci_hi=hi, share_diff_below_0=below,
                n_valid_resamples=int(ok.sum()), level=float(level))
