# -*- coding: utf-8 -*-
"""Validation v2, package B: L12 BT26-shift, on the synthetic world of
tests/test_benchmark_lags_aggregate.py.

Run: python -X utf8 -m pytest -q tests/test_benchmark_lags_bt26.py
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.benchmark import product
from experiments.benchmark.lags import bt26, common
from tests.test_benchmark_lags_aggregate import band_values, lag_errors, make_world, unit


def run(inst, variant=""):
    return bt26.run_unit(inst, unit("L12", ("L12",), variant), {})[0]


def test_vertex_rows_equals_scan_vertex_row_by_row():
    rng = np.random.default_rng(0)
    S = rng.normal(0, 1, (400, 46))
    S[rng.random(S.shape) < 0.15] = np.nan
    S[0] = np.nan                                          # nothing finite
    S[1, 2:] = np.nan                                      # two finite values
    S[2] = np.linspace(0, 1, 46)                           # best at the last point
    S[3] = np.linspace(1, 0, 46)                           # best at the first point
    S[4] = 0.5                                             # flat: first index, den 0
    S[5, 10:13] = [0.9, 1.0, np.nan]                       # neighbour missing
    S[6] = -np.abs(np.arange(46) - 20.3)                   # an interior vertex
    tau, edge = bt26.vertex_rows(S)
    for i in range(S.shape[0]):
        t_ref, e_ref = common.scan_vertex(S[i])
        assert (np.isnan(tau[i]) and np.isnan(t_ref)) or tau[i] == t_ref, i
        assert bool(edge[i]) == e_ref, i
    assert edge[2] and tau[2] == common.SCAN_HI and edge[3] and tau[3] == common.SCAN_LO


def test_corr_block_is_the_masked_pearson_r():
    inst, w = make_world(seed=51, days=300)
    H = inst.levels.stack(np.arange(-45, 181, 5))
    Hc = H - H.mean(axis=1, keepdims=True)
    r, n = bt26.corr_block(inst.obs.Y[:, :30], inst.obs.V[:, :30], Hc)
    for j in range(30):
        m = inst.obs.V[:, j]
        for i in (0, 17, 45):
            ref = np.corrcoef(inst.obs.Y[m, j].astype(float), H[i, m])[0, 1]
            assert r[j, i] == pytest.approx(ref, abs=1e-10)
        assert n[j] == m.sum()


def test_recovers_planted_lags():
    inst, w = make_world(seed=52)
    r = run(inst)
    assert r["status"] == "ok" and r["kind"] == "field" and r["lag_kind"] == "linear"
    v = band_values(r, w)
    err = lag_errors(v, w["lags"])
    assert np.median(err) <= 10.0 and np.all(err <= 15.0), v


def test_edge_at_200_min():
    inst, w = make_world(lags=(200.0, 200.0, 45.0), seed=53)
    r = run(inst)
    f = r["field"]
    assert np.median(f[(w["band_of"] < 2) & np.isfinite(f)]) == common.SCAN_HI
    assert r["diag"]["n_edge_hi"] > 0.5 * np.isfinite(f[w["band_of"] < 2]).sum()
    assert r["edge_share"] > 0.5


def test_qc_r_max_and_n():
    inst, w = make_world(seed=54)
    rng = np.random.default_rng(2)
    inst.obs.Y[:, 0] = rng.normal(0, 0.5, w["T"]).astype(np.float32)   # r_max < 0.3
    V = inst.obs.V
    rows = np.flatnonzero(V[:, 1])[:19]
    V[:, 1] = False
    V[rows, 1] = True                                                   # n = 19
    rows = np.flatnonzero(V[:, 2])[:20]
    V[:, 2] = False
    V[rows, 2] = True                                                   # n = 20
    r = run(inst)
    assert np.isnan(r["field"][0]) and np.isnan(r["field"][1])
    H = inst.levels.stack(np.arange(-45, 181, 5))
    rr, _ = bt26.corr_block(inst.obs.Y[:, [0]], inst.obs.V[:, [0]],
                            H - H.mean(axis=1, keepdims=True))
    assert np.nanmax(rr) < 0.3
    assert np.isfinite(r["field"][2])
    assert r["diag"]["qc_fail_first"]["n"] >= 1 and r["diag"]["qc_fail_first"]["r_max"] >= 1


def test_rank_variant_is_a_stub():
    inst, w = make_world(seed=55, days=60)
    with pytest.raises(product.NotComputable) as e:
        run(inst, "rank")
    assert "fewer than 3 tide models" in str(e.value)
    inst.installed_models = ("EOT20", "GOT4.10c", "FES2022")
    with pytest.raises(NotImplementedError):
        run(inst, "rank")


def test_streams_blocks_statuses_and_card():
    inst, w = make_world(seed=56, days=300, allow_read=False)
    bt26.CONFIG["block"], old = 200, bt26.CONFIG["block"]
    try:
        r = run(inst)
    finally:
        bt26.CONFIG["block"] = old
    assert r["status"] == "ok" and max(inst.obs.block_sizes) <= 200
    inst, w = make_world(seed=56, days=90)                     # 18 scenes
    assert run(inst)["status"] == "not_computable"
    inst, w = make_world(seed=56, days=300)
    inst.obs.Y[:] = np.float32(0.2)                            # no variance: QC fails
    assert run(inst)["status"] == "empty"
    inst, w = make_world(domain_status="no_intertidal", days=60)
    with pytest.raises(product.NotComputable):
        run(inst)
    c = bt26.CARDS["L12"]
    assert common.check_card(c) == [] and c["id"] == "L12"
    assert c["scan_window"] == [-45.0, 180.0, 5.0] and c["qc"] == {"min_n": 20, "min_r_max": 0.3}
    from experiments.benchmark import lags
    assert lags.load("L12") is bt26.run_unit and lags.card("L12") is c
