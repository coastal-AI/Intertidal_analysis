# -*- coding: utf-8 -*-
"""Validation of elevation products against a truth: the shared arithmetic.

Used by the site notebooks for the on-site validation (RTK-GNSS points:
:func:`ols_against_truth`, :func:`transect_order`) and the external one
(a LiDAR or bathymetric grid: :func:`score_products`,
:func:`lidar_diagnostic`). Every score is median-centred first: products
sit on the tide model's datum and truths on their own, and the datum is
reported separately, never mixed into the error.
"""
from __future__ import annotations

import numpy as np


def ols_against_truth(truth, product, min_n=20):
    """Median-centred comparison of one product with the truth on their
    common finite samples.

    Returns a dict with ``n``, ``rmse_m`` (centred), ``slope`` and
    ``intercept`` of the OLS line *product = a + b * truth*, ``r`` (Pearson),
    ``bias_m`` (median product - truth, the datum offset), plus the centred
    arrays ``x`` (truth) and ``y`` (product) for plotting. ``None`` when
    fewer than ``min_n`` samples are common.
    """
    t, p = np.asarray(truth, float), np.asarray(product, float)
    m = np.isfinite(t) & np.isfinite(p)
    if m.sum() < min_n:
        return None
    x, y = t[m] - np.median(t[m]), p[m] - np.median(p[m])
    b, a = np.polyfit(x, y, 1)
    return {"n": int(m.sum()), "rmse_m": float(np.sqrt(np.mean((y - x) ** 2))),
            "slope": float(b), "intercept": float(a),
            "r": float(np.corrcoef(x, y)[0, 1]),
            "bias_m": float(np.median(p[m] - t[m])), "x": x, "y": y}


def score_products(products, truth, mask=None, min_n=50):
    """Score several products against one truth on the same grid.

    ``products`` maps label -> array; ``truth`` an array on the same grid;
    ``mask`` (optional) restricts the comparison. Returns two dicts of rows:
    ``all`` (each product on every pixel it resolves) and ``common`` (all
    products on the pixels every one of them resolves), each row being the
    output of :func:`ols_against_truth` without the arrays.
    """
    t = np.asarray(truth, float)
    base = np.isfinite(t) if mask is None else (np.asarray(mask, bool) & np.isfinite(t))
    common = base.copy()
    for p in products.values():
        common &= np.isfinite(np.asarray(p, float))

    def row(p, sel):
        r = ols_against_truth(np.where(sel, t, np.nan), np.where(sel, np.asarray(p, float), np.nan), min_n)
        if r:
            r = {k: v for k, v in r.items() if k not in ("x", "y")}
        return r
    return ({k: row(p, base) for k, p in products.items()},
            {k: row(p, common) for k, p in products.items()},
            int(common.sum()))


def transect_order(east, north):
    """Order survey points along their principal axis.

    Returns ``(order, distance_m)``: the indices that sort the points along
    the first principal component of their positions, and the distance of
    each sorted point from the first one along that axis.
    """
    e, n = np.asarray(east, float), np.asarray(north, float)
    e, n = e - e.mean(), n - n.mean()
    w, vec = np.linalg.eigh(np.cov(np.vstack([e, n])))
    axis = vec[:, np.argmax(w)]
    d = e * axis[0] + n * axis[1]
    order = np.argsort(d)
    return order, d[order] - d[order][0]


def lidar_diagnostic(lidar, mask, tol_m=0.05):
    """How much of a LiDAR grid over the intertidal zone is one flat value.

    Over the flats a LiDAR flown at high water returns the water surface, a
    single elevation repeated everywhere. Returns the number of masked cells,
    the number of distinct values, the modal value and the share of cells
    within ``tol_m`` of it: the share that is a water surface, not a bed.
    """
    v = np.asarray(lidar, float)[np.asarray(mask, bool) & np.isfinite(np.asarray(lidar, float))]
    if v.size == 0:
        return None
    vals, counts = np.unique(np.round(v, 2), return_counts=True)
    mode = float(vals[np.argmax(counts)])
    return {"n": int(v.size), "distinct_values": int(len(vals)), "mode_m": mode,
            "share_within_tol_of_mode": float(np.mean(np.abs(v - mode) <= tol_m))}
