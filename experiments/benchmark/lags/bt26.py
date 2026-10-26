# -*- coding: utf-8 -*-
"""Validation v2, Stage L -- L12 BT26-shift: the per-pixel lag that best correlates Y with h_B.

Question
--------
Bishop-Taylor et al. (2026, IJRS 47(12):5268-5287) rank tide models by how
well their modelled heights correlate with what the satellite sees. The same
criterion, applied to delayed copies of the boundary tide instead of to
different models, gives each pixel a lag: the tau that maximises the
correlation of its index with h_B(t - tau).

Rules (spec §3.3 row L12, §3.3.1 "L12 BT26", §4.2; frozen in CONFIG)
------------------------------------------------------------------
* per pixel, over its valid scenes V: Pearson r(Y, h_B(t - tau)) for tau in
  ``common.TAU_SCAN`` (-45..180 min, step 5), with h_B(t - tau) the exact
  level rows (``inst.levels.stack``); Y is the setting's standardised,
  water-high index;
* tau = the vertex of the argmax: ``te._parabolic_argmax`` through
  ``common.scan_vertex``, vectorised over pixels (:func:`vertex_rows`, equal
  to ``scan_vertex`` row by row, bit for bit: tested); a best grid point on
  the window edge returns the edge value (the scorer counts it as an edge
  share);
* QC: at least 20 valid scenes (spec §4.2 "L12 ... >= 20 per pixel") and
  r_max >= 0.3 (spec §3.3.1); otherwise NaN;
* output: the per-pixel field (kind 'field', lag_kind 'linear'); the band
  value is the median of the field, taken by the scorer (spec §6.3).
* BT26-rank (variant 'rank'): the model with the highest median r per band,
  once at least 3 tide models are installed; a stub in V2 that is NOT
  COMPUTABLE ("fewer than 3 tide models") until then, and out of scope of V2
  beyond that (spec "Tier 2/3 lag rows, not scheduled").

Memory: ``inst.obs.iter_blocks(100_000)`` in chunks of about 2.5 million
cells (T x columns, 256 to 20,000 columns); the correlations of a chunk with
all 46 lags come from three matrix products.
"""
from __future__ import annotations

import numpy as np

from experiments.benchmark import product
from experiments.benchmark.lags import aggregate, common

CONFIG = {
    "score": "Pearson r(Y, h_B(t - tau)) over the pixel's valid scenes",
    "scan": [float(common.SCAN_LO), float(common.SCAN_HI), 5.0],
    "vertex": "te._parabolic_argmax (common.scan_vertex), vectorised",
    "qc": {"min_n": 20, "min_r_max": 0.3},
    "rank": {"min_models": 3, "status": "stub: NotComputable below 3 installed models"},
    "block": 100_000,
    "chunk_cells": 2_500_000,
    "chunk_cols": [256, 20_000],
}

CODE_DEPS = ("experiments/benchmark/lags/aggregate.py", "pyintertidal/tide_estimators.py")

CARDS = {"L12": {
    "id": "L12",
    "name": "BT26-shift: per-pixel r(Y, h_B(t - tau)) maximised over tau",
    "family": "order statistic",
    "kind": "lag",
    "tier": "1",
    "fidelity": "F4",
    "evidence": ("F4 adaptation (new code): unit tests on synthetic tides "
                 "(tests/test_benchmark_lags_bt26.py: planted 0/45/120 min recovered, edge at "
                 "200 min, r_max < 0.3 gives NaN, the vectorised vertex equals scan_vertex) "
                 "and the positive control of spec §10.2"),
    "references": [
        {"cite": "Bishop-Taylor R, Sagar S, Phillips C, Newey V (2026) IJRS 47(12):5268-5287 "
                 "(tide-model ranking by the correlation of satellite observations with "
                 "modelled heights)", "doi_or_url": "doi:10.1080/01431161.2026.2666912"},
        {"cite": "Bishop-Taylor R et al. (2025) eo-tides, JOSS", "doi_or_url":
         "doi:10.21105/joss.07786"},
    ],
    "output": "field",
    "lag_kind": "linear",
    "signature": aggregate.card_signature("L12"),
    "scan_window": [float(common.SCAN_LO), float(common.SCAN_HI), 5.0],
    "published_window": None,
    "as_published": ("the published criterion ranks candidate tide MODELS by the correlation "
                     "of satellite observations with each model's heights (BT26-rank, once 3 "
                     "models are installed); the lag scan over delayed copies of one boundary "
                     "tide is this study's adaptation"),
    "native_cell": None,
    "common_conditions": ("Y of the instance's setting on the harness domain keep; exact h_B "
                          "rows at TAU_SCAN; vertex of the argmax; QC n >= 20, r_max >= 0.3"),
    "deviations": [
        {"paper": "candidate heights from several tide models",
         "ours": "candidate heights h_B(t - tau), tau in -45..180 min",
         "why": "turns the model-selection criterion into a lag estimator",
         "rows": "L12", "sensitivity": "BT26-rank (variant 'rank', >= 3 models; stub in V2)"},
    ],
    "qc": {"min_n": 20, "min_r_max": 0.3},
    "n_min": 20,
    "tuning": [],
    "control_offset_min": 0.0,
    "control_offset_why": ("0 min: in the gating world (spec §10.2) each pixel's index is a "
                           "monotone function of h_B(t - tau_k) plus noise, so r peaks at tau_k"),
    "cost": aggregate.card_cost("L12"),
    "variants": {"rank": "BT26-rank: the model of highest median r per band (>= 3 installed "
                         "models; stub, not scheduled)"},
}}


def vertex_rows(S, grid=common.TAU_SCAN):
    """``common.scan_vertex`` of every row of ``S`` (m, L), vectorised:
    ``(tau (m,), edge (m,))``, bit for bit the same as the row-by-row call
    (fewer than 3 finite scores: NaN and False)."""
    S = np.asarray(S, np.float64)
    g = np.asarray(grid, np.float64)
    m, L = S.shape
    if L != g.size:
        raise ValueError(f"vertex_rows: {L} scores per row, grid of {g.size}")
    fin = np.isfinite(S)
    ok = fin.sum(axis=1) >= 3
    j = np.argmax(np.where(fin, S, -np.inf), axis=1)
    rows = np.arange(m)
    jj = np.clip(j, 1, L - 2)
    d0, d1, d2 = S[rows, jj - 1], S[rows, jj], S[rows, jj + 1]
    inner = (j > 0) & (j < L - 1) & np.isfinite(d0) & np.isfinite(d1) & np.isfinite(d2)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        den = d0 - 2 * d1 + d2
        off = np.where(np.abs(den) > 1e-12, 0.5 * (d0 - d2) / den, 0.0)
    step = g[1] - g[0]
    refined = g[j] + np.clip(off, -1, 1) * step
    tau = np.where(ok, np.where(inner, refined, g[j]), np.nan)
    edge = ok & ((j == 0) | (j == L - 1))
    return tau, edge


def corr_block(Y, V, Hc):
    """Pearson r of every column of Y (over its valid rows) with every row of
    ``Hc`` (L, T): returns (m, L) and n (m,)."""
    V = np.asarray(V, bool)
    Vf = V.astype(np.float64)
    Y64 = np.asarray(Y, np.float64)
    Yv = np.where(V, Y64, 0.0)
    n = Vf.sum(axis=0)
    # a pixel whose valid values are all equal has no correlation (its centred
    # values are rounding noise)
    varies = (np.max(np.where(V, Y64, -np.inf), axis=0)
              > np.min(np.where(V, Y64, np.inf), axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = Yv.sum(axis=0) / np.maximum(n, 1.0)
        Yc = np.where(V, Yv - ybar[None, :], 0.0)
        vy = np.sum(Yc * Yc, axis=0)                               # (m,)
        cov = Yc.T @ Hc.T                                          # (m, L)
        Sh = Vf.T @ Hc.T
        Shh = Vf.T @ (Hc * Hc).T
        vh = Shh - Sh * Sh / np.maximum(n, 1.0)[:, None]
        den = np.sqrt(vy[:, None] * vh)
        ok = (varies & (vy > 0) & (n >= 3))[:, None] & (vh > 0)
        r = np.where(ok, cov / den, np.nan)
    return r, n


def run_unit(inst, unit, opts=None):
    """L12 BT26-shift on the instance's keep pixels (module docstring)."""
    aggregate.require_domain(inst)
    products = aggregate.unit_products(unit, ("L12",))
    if "rank" in aggregate.variant_parts(unit):
        models = tuple(getattr(inst, "installed_models", ()) or ())
        if len(models) < int(CONFIG["rank"]["min_models"]):
            raise product.NotComputable("rank", "fewer than 3 tide models",
                                        {"installed_models": list(models)})
        raise NotImplementedError("BT26-rank is not scheduled in milestone V2 "
                                  "(Tier 2/3 lag rows)")
    q = CONFIG["qc"]
    n_keep = aggregate._n_keep(inst)
    with aggregate._stage(inst, "levels"):
        H = np.asarray(inst.levels.stack(np.asarray(np.round(common.TAU_SCAN), np.int64)),
                       np.float64)
    Hc = H - H.mean(axis=1, keepdims=True)
    tau = np.full(n_keep, np.nan, np.float32)
    fails = {"n": 0, "r_max": 0}
    n_enough = 0
    n_edge_lo = n_edge_hi = 0
    rmax_kept = []
    chunk = aggregate.chunk_cols(inst.obs.T, CONFIG["chunk_cells"], CONFIG["chunk_cols"])
    with aggregate._stage(inst, "scan"):
        for cols, ob in inst.obs.iter_blocks(int(CONFIG["block"])):
            cols = np.asarray(cols, np.int64)
            for a in range(0, len(cols), chunk):
                b = min(a + chunk, len(cols))
                r, n = corr_block(ob.Y[:, a:b], ob.V[:, a:b], Hc)
                fin = np.isfinite(r)
                r_max = np.where(fin.any(axis=1), np.max(np.where(fin, r, -np.inf), axis=1),
                                 np.nan)
                t_px, edge = vertex_rows(r)
                ok_n = n >= q["min_n"]
                with np.errstate(invalid="ignore"):
                    ok_r = r_max >= q["min_r_max"]
                good = ok_n & ok_r & np.isfinite(t_px)
                fails["n"] += int(np.sum(~ok_n))
                fails["r_max"] += int(np.sum(ok_n & ~(ok_r & np.isfinite(t_px))))
                n_enough += int(ok_n.sum())
                n_edge_lo += int(np.sum(good & edge & (t_px <= common.SCAN_LO)))
                n_edge_hi += int(np.sum(good & edge & (t_px >= common.SCAN_HI)))
                tau[cols[a:b][good]] = t_px[good].astype(np.float32)
                rmax_kept.append(r_max[good])
            aggregate._progress(inst, len(cols))
    rmax_kept = np.concatenate(rmax_kept) if rmax_kept else np.zeros(0)
    n_def = int(np.isfinite(tau).sum())
    diag = {"n_keep": n_keep, "n_defined": n_def, "n_with_min_n": n_enough,
            "qc_fail_first": fails, "qc": dict(q),
            "n_edge_lo": n_edge_lo, "n_edge_hi": n_edge_hi,
            "median_r_max_defined": float(np.median(rmax_kept)) if rmax_kept.size
            else float("nan")}
    if n_def:
        res = common.lag_result("L12", "ok", n_keep=n_keep, kind="field", lag_kind="linear",
                                field=tau, diag=diag)
    elif n_enough == 0:
        res = common.lag_result("L12", "not_computable", n_keep=n_keep, kind="field",
                                lag_kind="linear", stage=f"n_lt_{q['min_n']}",
                                reason="no keep pixel has 20 valid scenes", diag=diag)
    else:
        res = common.lag_result("L12", "empty", n_keep=n_keep, kind="field", lag_kind="linear",
                                stage="qc", reason="every pixel failed the QC", diag=diag)
    return [res for _ in products]
