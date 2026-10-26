# -*- coding: utf-8 -*-
"""Validation v2, Stage L -- L11 PPL: the phase-plane logistic lag of each pixel (F4, new).

Question
--------
A pixel is wet when the local level exceeds its elevation. If the local
level is the boundary M2 tide delayed by tau, then with x1 = h_B(t) - mean
and x2 = (dh_B/dt)/omega (the boundary's state in its phase plane, both in
metres)

    h_B(t - tau) = x1 cos(omega tau) - x2 sin(omega tau),

so the pixel's wet/dry boundary is a straight line in the (x1, x2) plane
whose normal points at the angle -omega tau. A logistic regression of the
wet layer on (x1, x2) finds that line without any lag scan:
tau = atan2(-b2, b1) / omega.

Rules (spec §3.3 row L11, §3.3.1 "L11 PPL", §4.2; frozen in CONFIG)
------------------------------------------------------------------
* x1 = h0 - mean(h0 over the instance's scenes); x2 = (dh_B/dt at t_i) /
  omega_M2, the rate a central difference of the 1-min boundary series over
  +-1 min (``inst.levels.series_at``), omega_M2 = 2 pi / 745.236 min
  (``common.OMEGA_M2_PER_MIN``);
* per pixel, over its valid scenes V: penalised logistic regression of W on
  [1, x1, x2] by Newton (IRLS) steps: the objective is LL(b) - (lambda/2)
  (b1^2 + b2^2) with lambda = 1e-2 n (n = the pixel's valid scenes; b0 is
  not penalised), vectorised over pixels with 3 x 3 solves, step halving on
  the penalised objective, at most 50 iterations, converged when every
  |step| < 1e-8;
* tau = atan2(-b2, b1) / omega_M2 minutes, a phase lag (circular; period one
  M2 cycle; the scorer wraps it about 67.5 min and the band value is the
  circular median of the field, ``common.circ_median``);
* QC: n >= 20; 0.05 < wet share < 0.95 (pixels outside are not fitted);
  b1 > 0 (wet when the level is high); likelihood-ratio p < 0.01 (chi^2, 2
  df, of 2 (LL(b) - LL_0), LL_0 the intercept-only maximum, LL unpenalised
  at the penalised estimate); converged. Otherwise NaN.

Consequence of b1 > 0, stated on the card: |tau| < P/4 = 186 min. A pixel
whose lag exceeds a quarter period fails the QC rather than wrapping.

Memory and speed: ``inst.obs.iter_blocks(100_000)``, fitted in chunks of
about 30,000 cells (T x columns, 16 to 2,000 columns: about a dozen
float64 arrays of ~240 kB alive, small enough to stay in cache and below the
allocator's large-block threshold; measured 2.5 x faster than 500 columns);
no (T, n_keep) array is held.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from experiments.benchmark.lags import aggregate, common

CONFIG = {
    "x1": "h0 - mean(h0 over the instance)",
    "x2": "(dh_B/dt at t_i) / omega_M2; central difference of series1 over +-1 min",
    "rate_dt_min": 1.0,
    "omega_per_min": float(common.OMEGA_M2_PER_MIN),
    "ridge_per_n": 1e-2,
    "ridge_on": ["b1", "b2"],
    "objective": "LL(b) - (ridge_per_n * n / 2) (b1^2 + b2^2)",
    "max_iter": 50,
    "tol": 1e-8,
    "max_halvings": 30,
    "tau": "atan2(-b2, b1) / omega_M2",
    "qc": {"min_n": 20, "wet_lo": 0.05, "wet_hi": 0.95, "b1_positive": True,
           "lr_p_max": 0.01, "lr_df": 2, "converged": True},
    "block": 100_000,
    "chunk_cells": 30_000,
    "chunk_cols": [16, 2_000],
}

CODE_DEPS = ("experiments/benchmark/lags/aggregate.py",)

CARDS = {"L11": {
    "id": "L11",
    "name": "PPL: phase-plane logistic (the angle of the wet/dry line in (h, dh/dt / omega))",
    "family": "harmonic",
    "kind": "lag",
    "tier": "1",
    "fidelity": "F4",
    "evidence": ("F4 adaptation (new code): unit tests on synthetic tides "
                 "(tests/test_benchmark_lags_ppl.py: the sign convention on a planted lag, "
                 "planted 0/45/120 min recovered, each QC rule) and the positive control of "
                 "spec §10.2"),
    "references": [
        {"cite": "Validation v2 specification §3.3 (row L11) and §3.3.1: in-house (F4)",
         "doi_or_url": "docs/validation_v2_spec.md"},
        {"cite": "le Cessie S, van Houwelingen JC (1992) Ridge estimators in logistic "
                 "regression. J. R. Stat. Soc. C 41(1):191-201",
         "doi_or_url": "doi:10.2307/2347628"},
    ],
    "output": "field",
    "lag_kind": "phase",
    "signature": aggregate.card_signature("L11"),
    "scan_window": None,
    "published_window": None,
    "as_published": "no published configuration (in-house)",
    "native_cell": None,
    "common_conditions": ("W & V of the instance's setting on the harness domain keep; x1 from "
                          "the exact h0 rows, x2 from the 1-min h_B; band value = circular "
                          "median of the field (scorer)"),
    "deviations": [],
    "qc": {"min_n": 20, "wet_lo": 0.05, "wet_hi": 0.95, "b1_positive": True,
           "lr_p_max": 0.01, "converged": True,
           "consequence": "b1 > 0 limits |tau| to a quarter M2 period (186 min)"},
    "n_min": 20,
    "tuning": [],
    "control_offset_min": 0.0,
    "control_offset_why": ("0 min for an M2-dominated tide: the local level of the gating world "
                           "(spec §10.2) is h_B(t - tau_k), whose phase-plane angle is "
                           "-omega tau_k; other constituents add scatter, not a bias"),
    "cost": aggregate.card_cost("L11"),
    "variants": {},
}}


def phase_plane(inst, dt=None):
    """``(X, ok)``: the design [1, x1, x2] (T, 3) of the instance, zero rows
    where x2 is not finite, and the finite mask."""
    dt = float(CONFIG["rate_dt_min"] if dt is None else dt)
    h0 = np.asarray(inst.h0, np.float64)
    t = aggregate.t_minutes(inst.t)
    up = np.asarray(inst.levels.series_at(t + dt), np.float64)
    dn = np.asarray(inst.levels.series_at(t - dt), np.float64)
    x1 = h0 - h0.mean()
    x2 = (up - dn) / (2.0 * dt) / float(common.OMEGA_M2_PER_MIN)
    ok = np.isfinite(x1) & np.isfinite(x2)
    X = np.column_stack([np.ones_like(x1), np.where(ok, x1, 0.0), np.where(ok, x2, 0.0)])
    X[~ok] = 0.0
    return X, ok


def _softplus_e(eta, e):
    """log(1 + exp(eta)) from ``e = exp(-|eta|)`` (stable)."""
    return np.maximum(eta, 0.0) + np.log1p(e)


def _sigmoid_e(eta, e):
    """expit(eta) from ``e = exp(-|eta|)`` (stable)."""
    d = 1.0 / (1.0 + e)
    return np.where(eta >= 0.0, d, e * d)


def _pll(eta, y, Vf, e=None):
    """Unpenalised Bernoulli log-likelihood per column."""
    if e is None:
        e = np.exp(-np.abs(eta))
    return np.sum(Vf * (y * eta - _softplus_e(eta, e)), axis=0)


def _xx(X):
    return np.column_stack([X[:, 0] * X[:, 0], X[:, 0] * X[:, 1], X[:, 0] * X[:, 2],
                            X[:, 1] * X[:, 1], X[:, 1] * X[:, 2], X[:, 2] * X[:, 2]])


def logistic_fit(X, W, V, ridge_per_n=None, max_iter=None, tol=None, max_halvings=None):
    """Penalised logistic fits of every column (module docstring).

    Damped Newton (IRLS) on LL(b) - (lambda/2)(b1^2 + b2^2), lambda =
    ridge_per_n * n: each iteration solves (X' W X + diag(0, lambda, lambda))
    step = gradient per column; a step that lowers the penalised objective is
    halved (eta is linear in b, so the halving needs no new product); a
    column has converged when every |step| < tol; converged columns leave
    the active set.

    Parameters
    ----------
    X : (T, 3) design [1, x1, x2] (zero rows at unusable instants)
    W, V : (T, m) bool, wet and valid (V already AND-ed with the usable rows)

    Returns
    -------
    dict of (m,) arrays: beta (m, 3), ll, ll0, n, wet, converged, iters.
    Columns are fitted only where n >= 1 and 0 < wet share < 1; others NaN.
    """
    ridge_per_n = float(CONFIG["ridge_per_n"] if ridge_per_n is None else ridge_per_n)
    max_iter = int(CONFIG["max_iter"] if max_iter is None else max_iter)
    tol = float(CONFIG["tol"] if tol is None else tol)
    max_halvings = int(CONFIG["max_halvings"] if max_halvings is None else max_halvings)
    X = np.asarray(X, np.float64)
    V = np.asarray(V, bool)
    n = V.sum(axis=0).astype(np.float64)
    nw = (np.asarray(W, bool) & V).sum(axis=0).astype(np.float64)
    m = V.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        wet = np.where(n > 0, nw / np.maximum(n, 1.0), np.nan)
    beta = np.full((m, 3), np.nan)
    ll = np.full(m, np.nan)
    ll0 = np.full(m, np.nan)
    conv = np.zeros(m, bool)
    iters = np.zeros(m, np.int64)
    fit_idx = np.flatnonzero((n >= 1) & (wet > 0) & (wet < 1))
    if not fit_idx.size:
        return {"beta": beta, "ll": ll, "ll0": ll0, "n": n.astype(np.int64), "wet": wet,
                "converged": conv, "iters": iters}
    pw = wet[fit_idx]
    ll0[fit_idx] = n[fit_idx] * (pw * np.log(pw) + (1.0 - pw) * np.log(1.0 - pw))
    XX = _xx(X)
    # the active set: its columns of the masks and the state
    idx = fit_idx.copy()
    Vs = V[:, idx].astype(np.float64)
    ys = (np.asarray(W, bool)[:, idx] & V[:, idx]).astype(np.float64)
    lam = ridge_per_n * n[idx]
    b = np.zeros((idx.size, 3))
    b[:, 0] = np.log(pw / (1.0 - pw))
    eta = X @ b.T
    e = np.exp(-np.abs(eta))
    obj = ll0[idx].copy()                                   # b1 = b2 = 0: no penalty
    it_n = np.zeros(idx.size, np.int64)
    for it in range(max_iter):
        if not idx.size:
            break
        if it == 0:
            # the null start: p is the column's wet share at every instant, so
            # the first Newton step needs products only
            g = (X.T @ ys).T - pw[:, None] * (X.T @ Vs).T
            S = (pw * (1.0 - pw))[:, None] * (Vs.T @ XX)
        else:
            p = _sigmoid_e(eta, e)
            g = (X.T @ (Vs * (ys - p))).T
            g[:, 1:] -= lam[:, None] * b[:, 1:]
            S = (Vs * p * (1.0 - p)).T @ XX
            del p
        A = np.empty((idx.size, 3, 3))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = S[:, 0], S[:, 1], S[:, 2]
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = S[:, 1], S[:, 3] + lam, S[:, 4]
        A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = S[:, 2], S[:, 4], S[:, 5] + lam
        step = np.linalg.solve(A, g[:, :, None])[:, :, 0]
        d_eta = X @ step.T
        b_c = b + step
        eta_c = eta + d_eta
        e_c = np.exp(-np.abs(eta_c))
        obj_c = _pll(eta_c, ys, Vs, e_c) - 0.5 * lam * (b_c[:, 1] ** 2 + b_c[:, 2] ** 2)
        ts = np.ones(idx.size)
        worse = obj_c < obj - 1e-12 * np.maximum(1.0, np.abs(obj))
        for _h in range(max_halvings):
            if not worse.any():
                break
            j = np.flatnonzero(worse)
            ts[j] *= 0.5
            b_c[j] = b[j] + ts[j, None] * step[j]
            eta_c[:, j] = eta[:, j] + ts[j][None, :] * d_eta[:, j]
            e_c[:, j] = np.exp(-np.abs(eta_c[:, j]))
            obj_c[j] = (_pll(eta_c[:, j], ys[:, j], Vs[:, j], e_c[:, j])
                        - 0.5 * lam[j] * (b_c[j, 1] ** 2 + b_c[j, 2] ** 2))
            worse[j] = obj_c[j] < obj[j] - 1e-12 * np.maximum(1.0, np.abs(obj[j]))
        b, eta, e, obj = b_c, eta_c, e_c, obj_c
        it_n += 1
        done = np.max(np.abs(ts[:, None] * step), axis=1) < tol
        if done.any():
            out_i = idx[done]
            beta[out_i] = b[done]
            ll[out_i] = _pll(eta[:, done], ys[:, done], Vs[:, done], e[:, done])
            conv[out_i] = True
            iters[out_i] = it_n[done]
            keep = ~done
            idx, b, obj, lam, it_n = idx[keep], b[keep], obj[keep], lam[keep], it_n[keep]
            pw = pw[keep]
            if idx.size:
                eta, e, Vs, ys = eta[:, keep], e[:, keep], Vs[:, keep], ys[:, keep]
    if idx.size:                                           # not converged in max_iter
        beta[idx] = b
        ll[idx] = _pll(eta, ys, Vs, e)
        iters[idx] = it_n
    return {"beta": beta, "ll": ll, "ll0": ll0, "n": n.astype(np.int64), "wet": wet,
            "converged": conv, "iters": iters}


def tau_of(beta):
    """tau = atan2(-b2, b1) / omega_M2 (min)."""
    return np.arctan2(-beta[:, 2], beta[:, 1]) / float(common.OMEGA_M2_PER_MIN)


def qc_mask(fit):
    """PPL's QC on a :func:`logistic_fit` result: (good, first failing rule
    counts as a dict of boolean arrays)."""
    q = CONFIG["qc"]
    with np.errstate(invalid="ignore"):
        lr = 2.0 * (fit["ll"] - fit["ll0"])
        p = chi2.sf(np.maximum(lr, 0.0), int(q["lr_df"]))
        rules = {"n": fit["n"] >= q["min_n"],
                 "wet": (fit["wet"] > q["wet_lo"]) & (fit["wet"] < q["wet_hi"]),
                 "converged": fit["converged"],
                 "b1": fit["beta"][:, 1] > 0,
                 "lr": np.isfinite(lr) & (p < q["lr_p_max"])}
    return rules, p


def run_unit(inst, unit, opts=None):
    """L11 PPL on the instance's keep pixels (module docstring)."""
    aggregate.require_domain(inst)
    products = aggregate.unit_products(unit, ("L11",))
    n_keep = aggregate._n_keep(inst)
    X, x_ok = phase_plane(inst)
    q = CONFIG["qc"]
    tau = np.full(n_keep, np.nan, np.float32)
    fails = {"n": 0, "wet": 0, "converged": 0, "b1": 0, "lr": 0}
    n_enough = 0
    iters = []
    chunk = aggregate.chunk_cols(inst.obs.T, CONFIG["chunk_cells"], CONFIG["chunk_cols"])
    with aggregate._stage(inst, "fit"):
        for cols, ob in inst.obs.iter_blocks(int(CONFIG["block"])):
            cols = np.asarray(cols, np.int64)
            for a in range(0, len(cols), chunk):
                b = min(a + chunk, len(cols))
                V = np.asarray(ob.V[:, a:b], bool) & x_ok[:, None]
                W = np.asarray(ob.W[:, a:b], bool)
                n = V.sum(axis=0)
                wet = np.where(n > 0, (W & V).sum(axis=0) / np.maximum(n, 1), np.nan)
                pre = (n >= q["min_n"]) & (wet > q["wet_lo"]) & (wet < q["wet_hi"])
                n_enough += int(np.sum(n >= q["min_n"]))
                idx = np.flatnonzero(pre)
                good = np.zeros(b - a, bool)
                t_blk = np.full(b - a, np.nan)
                if idx.size:
                    fit = logistic_fit(X, W[:, idx], V[:, idx])
                    rules, _ = qc_mask(fit)
                    g = np.ones(idx.size, bool)
                    for name in ("n", "wet", "converged", "b1", "lr"):
                        fails[name] += int(np.sum(g & ~rules[name]))
                        g &= rules[name]
                    good[idx[g]] = True
                    t_blk[idx] = tau_of(fit["beta"])
                    iters.append(fit["iters"])
                fails["n"] += int(np.sum(n < q["min_n"]))
                fails["wet"] += int(np.sum((n >= q["min_n"]) & ~pre))
                tau[cols[a:b][good]] = t_blk[good].astype(np.float32)
                del V, W
            aggregate._progress(inst, len(cols))
    iters = np.concatenate(iters) if iters else np.zeros(0, np.int64)
    n_def = int(np.isfinite(tau).sum())
    diag = {"n_keep": n_keep, "n_defined": n_def, "n_with_min_n": n_enough,
            "qc_fail_first": fails, "qc": dict(q),
            "median_iterations": float(np.median(iters)) if iters.size else float("nan"),
            "max_iterations": int(iters.max()) if iters.size else 0,
            "n_instants_without_rate": int(np.sum(~x_ok))}
    if n_def:
        res = common.lag_result("L11", "ok", n_keep=n_keep, kind="field", lag_kind="phase",
                                field=tau, edge_share=0.0, diag=diag)
    elif n_enough == 0:
        res = common.lag_result("L11", "not_computable", n_keep=n_keep, kind="field",
                                lag_kind="phase", stage=f"n_lt_{q['min_n']}",
                                reason="no keep pixel has 20 valid scenes", diag=diag)
    else:
        res = common.lag_result("L11", "empty", n_keep=n_keep, kind="field", lag_kind="phase",
                                stage="qc", reason="every pixel failed the QC", diag=diag)
    return [res for _ in products]
