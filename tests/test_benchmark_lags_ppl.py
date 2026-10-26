# -*- coding: utf-8 -*-
"""Validation v2, package B: L11 PPL (phase-plane logistic), on the synthetic
world of tests/test_benchmark_lags_aggregate.py.

Run: python -X utf8 -m pytest -q tests/test_benchmark_lags_ppl.py
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.benchmark import product
from experiments.benchmark.lags import common, ppl
from tests.test_benchmark_lags_aggregate import band_values, lag_errors, make_world, unit


def run(inst):
    return ppl.run_unit(inst, unit("L11", ("L11",)), {})[0]


def test_tau_of_the_separating_line():
    w = common.OMEGA_M2_PER_MIN
    for tau in (-40.0, 0.0, 45.0, 120.0, 180.0):
        beta = np.array([[0.3, 2.0 * np.cos(w * tau), -2.0 * np.sin(w * tau)]])
        assert ppl.tau_of(beta)[0] == pytest.approx(tau, abs=1e-9)


@pytest.mark.parametrize("lag", [45.0, -30.0])
def test_sign_convention_on_a_planted_lag(lag):
    """A local tide LATER than h_B gives a positive lag (pure M2: exact geometry)."""
    inst, w = make_world(lags=(lag,) * 3, amps={"M2": 0.8}, seed=41)
    r = run(inst)
    v = band_values(r, w, circular=True)
    assert np.all(np.sign(v) == np.sign(lag))
    assert np.all(lag_errors(v, w["lags"], circular=True) <= 3.0), v


def test_recovers_planted_lags():
    inst, w = make_world(seed=42)
    r = run(inst)
    assert r["status"] == "ok" and r["kind"] == "field" and r["lag_kind"] == "phase"
    v = band_values(r, w, circular=True)
    err = lag_errors(v, w["lags"], circular=True)
    assert np.median(err) <= 10.0 and np.all(err <= 15.0), v
    assert r["edge_share"] == 0.0
    assert r["diag"]["max_iterations"] < ppl.CONFIG["max_iter"]


def test_quarter_period_limit_of_b1_positive():
    """|tau| > P/4 (186 min) means b1 < 0: such pixels fail the QC (stated on the card)."""
    inst, w = make_world(lags=(200.0, 200.0, 45.0), seed=43)
    r = run(inst)
    f = r["field"]
    share_200 = np.isfinite(f[w["band_of"] < 2]).mean()
    share_45 = np.isfinite(f[w["band_of"] == 2]).mean()
    assert share_200 < 0.5 * share_45
    assert r["diag"]["qc_fail_first"]["b1"] > 0


def test_qc_rules_at_their_thresholds():
    inst, w = make_world(seed=44, days=400)
    X, ok = ppl.phase_plane(inst)
    T = X.shape[0]
    rng = np.random.default_rng(3)
    h_loc = inst.levels.at(30.0)
    z = 0.1
    good = h_loc > z
    cols = {"good": good, "n19": good, "n20": good, "wet98": good,
            "inverted": h_loc < z, "random": rng.random(T) < 0.5}
    W = np.stack(list(cols.values()), 1)
    V = np.repeat(ok[:, None], W.shape[1], 1)
    idx = np.flatnonzero(ok)
    V[:, 1] = False
    V[idx[:19], 1] = True
    V[:, 2] = False
    V[idx[:20], 2] = True
    W[:, 3] = True
    W[idx[: int(round(0.02 * len(idx)))], 3] = False
    W &= V
    fit = ppl.logistic_fit(X, W, V)
    rules, p = ppl.qc_mask(fit)
    passed = np.logical_and.reduce([rules[k] for k in rules])
    assert list(passed) == [True, False, True, False, False, False], (rules, p)
    assert not rules["n"][1] and rules["n"][2]
    assert fit["wet"][3] >= 0.95 and not rules["wet"][3]
    assert fit["beta"][4, 1] < 0 and not rules["b1"][4]
    assert p[5] >= 0.01 and not rules["lr"][5]
    assert fit["converged"][[0, 2, 4, 5]].all()


def test_ridge_keeps_separable_pixels_finite():
    inst, w = make_world(seed=45, days=300)
    X, ok = ppl.phase_plane(inst)
    h_loc = inst.levels.at(60.0)
    W = (h_loc[:, None] > np.array([-0.3, 0.0, 0.4])[None, :]) & ok[:, None]
    V = np.repeat(ok[:, None], 3, 1)
    fit = ppl.logistic_fit(X, W, V)
    assert np.isfinite(fit["beta"]).all() and fit["converged"].all()
    assert np.all(np.abs(fit["beta"]) < 1e3)
    tau = ppl.tau_of(fit["beta"])
    assert np.all(np.abs(common.circ_diff(tau, 60.0)) < 15.0), tau


def test_qc_failures_are_nan_in_the_product():
    inst, w = make_world(seed=46)
    V, W = inst.obs.V, inst.obs.W
    rows = np.flatnonzero(V[:, 0])[:19]
    V[:, 0] = False
    V[rows, 0] = True
    W[:, 0] &= V[:, 0]
    W[:, 1] = V[:, 1]                                       # always wet
    W[:, 2] = V[:, 2] & (np.random.default_rng(0).random(w["T"]) < 0.5)   # no tide signal
    r = run(inst)
    assert np.isnan(r["field"][:3]).all()


def test_streams_blocks_statuses_and_card():
    inst, w = make_world(seed=47, days=300, allow_read=False)
    ppl.CONFIG["block"], old = 150, ppl.CONFIG["block"]
    try:
        r = run(inst)
    finally:
        ppl.CONFIG["block"] = old
    assert r["status"] == "ok" and max(inst.obs.block_sizes) <= 150
    inst, w = make_world(seed=47, days=90)                  # 18 scenes
    r = run(inst)
    assert r["status"] == "not_computable" and r["stage"] == "n_lt_20"
    inst, w = make_world(seed=47, days=300)
    inst.obs.W[:] = False                                    # never wet: every pixel fails
    r = run(inst)
    assert r["status"] == "empty"
    inst, w = make_world(domain_status="no_intertidal", days=60)
    with pytest.raises(product.NotComputable):
        run(inst)
    c = ppl.CARDS["L11"]
    assert common.check_card(c) == [] and c["id"] == "L11" and c["lag_kind"] == "phase"
    from experiments.benchmark import lags
    assert lags.load("L11") is ppl.run_unit and lags.card("L11") is c
