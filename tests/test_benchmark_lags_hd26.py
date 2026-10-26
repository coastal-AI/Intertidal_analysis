# -*- coding: utf-8 -*-
"""Validation v2, package B: the L09 unit (L09 HD26-H and L10 HD26-R2), on the
synthetic world of tests/test_benchmark_lags_aggregate.py (750 days of scenes,
so the 365-day rule of HD26-H is met).

Run: python -X utf8 -m pytest -q tests/test_benchmark_lags_hd26.py
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.benchmark import config, product, truth
from experiments.benchmark.lags import aggregate, common, hd26
from tests.test_benchmark_lags_aggregate import lag_errors, make_world, unit


def run(inst, products=("L09", "L10"), variant=""):
    return hd26.run_unit(inst, unit("L09", products, variant), {})


@pytest.mark.parametrize("lags", [(0.0, 45.0, 120.0), (200.0, 45.0, 0.0)])
def test_recovers_planted_lags(lags):
    inst, w = make_world(lags=lags, seed=31)
    r9, r10 = run(inst)
    assert (r9["product"], r10["product"]) == ("L09", "L10")
    assert r9["status"] == "ok" and r9["kind"] == "band" and r9["lag_kind"] == "phase"
    err9 = lag_errors(r9["band"]["tau"], w["lags"], circular=True)
    assert np.median(err9) <= 10.0 and np.all(err9 <= 15.0), r9["band"]["tau"]
    assert not r9["band"]["edge"].any()                        # a phase lag has no window
    assert np.allclose(r9["field"], r9["band"]["tau"][w["band_of"]].astype(np.float32))
    assert r10["status"] == "ok" and r10["lag_kind"] == "linear"
    if 200.0 in lags:
        assert bool(r10["band"]["edge"][0]) and r10["band"]["tau"][0] == common.SCAN_HI
        err10 = lag_errors(r10["band"]["tau"][1:], w["lags"][1:])
    else:
        assert not r10["band"]["edge"].any()
        err10 = lag_errors(r10["band"]["tau"], w["lags"])
    assert np.median(err10) <= 10.0 and np.all(err10 <= 15.0), r10["band"]["tau"]


def test_rel_is_wrap180_of_the_two_m2_phases():
    inst, w = make_world(seed=32)
    r9 = run(inst, ("L09",))[0]
    d = r9["diag"]
    tau = truth.wrap180(d["g_proxy_deg"] - d["g_h0_deg"]) * 60.0 / 28.9841042
    assert np.allclose(r9["band"]["tau"], tau)
    agg = aggregate.band_aggregates(inst)
    use = agg["cover"][:, 1] >= 0.8
    ref = truth.hd26_m2_phase(truth.days_since_1992(inst.t[use]), agg["wetfrac"][use, 1])
    assert d["g_proxy_deg"][1] == ref["phase_deg"] and d["n_points"][1] == ref["n"]
    assert np.all(d["vif_m2"] >= 1.0) and not d["vif_flagged"].any()
    assert np.all(np.isfinite(d["tau_index_proxy_min"]))


def test_l09_not_computable_under_365_days_while_l10_runs():
    inst, w = make_world(days=300, step_days=3, seed=33)       # ~100 scenes over 300 days
    r9, r10 = run(inst)
    assert r9["status"] == "not_computable" and r9["stage"] == "span_lt_365d"
    assert r9["field"] is None and set(r9["diag"]["band_reason"]) == {"span_lt_365d"}
    assert np.all(r9["diag"]["span_days"] < 365)
    assert r10["status"] == "ok"


def test_l09_band_rules():
    t_days = np.arange(0, 800, 5.0)
    rng = np.random.default_rng(0)
    y = np.cos(2 * np.pi * t_days * 24 / 12.4206012) + rng.normal(0, 0.1, t_days.size)
    h0 = np.cos(2 * np.pi * t_days * 24 / 12.4206012)
    assert hd26.l09_band(t_days[:29], y[:29], h0[:29])["reason"] == "span_lt_365d"
    assert hd26.l09_band(t_days[::6][:29], y[::6][:29], h0[::6][:29])["reason"] == "points_lt_30"
    assert hd26.l09_band(t_days[:2], y[:2], h0[:2])["reason"] == "points_lt_30"
    ok = hd26.l09_band(t_days, y, h0)
    assert ok["reason"] == "ok" and abs(ok["tau"]) < 2.0
    flat = hd26.l09_band(t_days, np.full(t_days.size, 0.3), h0)
    assert flat["reason"] == "proxy_constant" and np.isnan(flat["tau"])
    assert hd26.l09_band(t_days[:29], np.full(29, 0.3), h0[:29])["reason"] == "span_lt_365d"


def test_both_products_not_computable_on_a_short_record():
    inst, w = make_world(days=200, seed=34)                    # 40 scenes, 200 days
    r9, r10 = run(inst)
    assert r9["status"] == "not_computable" and r10["status"] == "not_computable"
    assert r10["stage"] == "scenes_lt_60"


def test_products_in_the_unit_order_and_shared_aggregates():
    inst, w = make_world(seed=35, days=500)
    out = run(inst, ("L10", "L09"))
    assert [r["product"] for r in out] == ["L10", "L09"]
    assert "aggregate.band" in inst.cache
    assert sum(1 for s in w["stages"] if s == "aggregate") >= 2   # built once, timed twice


def test_cards_external_inputs_and_variants():
    assert set(hd26.CARDS) == {"L09", "L10"}
    for pid, c in hd26.CARDS.items():
        assert common.check_card(c) == [] and c["id"] == pid
    assert hd26.CARDS["L09"]["fidelity"] == "F2" and hd26.CARDS["L09"]["lag_kind"] == "phase"
    assert hd26.CARDS["L10"]["fidelity"] == "F4" and hd26.CARDS["L10"]["n_min"] == 60
    assert hd26.CONFIG["L09"]["min_span_days"] == config.FROZEN["hd26"]["min_span_days"]
    ext = hd26.external_inputs(None, None)
    assert set(ext) == {"pyTMD"} and len(ext["pyTMD"]) == 64
    inst, w = make_world(seed=36, days=60)
    with pytest.raises(ValueError):
        run(inst, variant="faithful")
    inst, w = make_world(domain_status="no_intertidal", days=60)
    with pytest.raises(product.NotComputable):
        run(inst)
    from experiments.benchmark import lags
    assert lags.load("L09") is hd26.run_unit
    assert lags.card("L09") is hd26.CARDS["L09"] and lags.card("L10") is hd26.CARDS["L10"]


def test_vif_grows_when_the_record_is_short():
    t_long = np.arange(0, 1095, 5.0) + 0.45
    t_short = np.arange(0, 120, 5.0) + 0.45
    v_long, v_short = hd26.vif_m2(t_long), hd26.vif_m2(t_short)
    assert 1.0 <= v_long < 2.0 and v_short > v_long
    assert np.isnan(hd26.vif_m2(t_long[:10]))


def test_l09_fits_h0_on_the_scenes_the_proxy_kept():
    """rel = h0 at the same scenes: the proxy's 3-sigma pass drops an
    outlier scene, and h0 is fitted without it (the V2 build fitted h0 on
    every usable scene)."""
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 760.0, 120)) + 11000.0
    w = 2 * np.pi * 24.0 * 28.9841042 / 360.0
    h0 = 1.6 * np.cos(w * t) + 0.3 * np.cos(2 * np.pi * 24.0 * 13.9430356 / 360.0 * t)
    proxy = 0.5 + 0.3 * np.cos(w * t - 0.4) + rng.normal(0, 0.02, len(t))
    proxy[7] = 5.0                                        # one wild scene
    kept = hd26.proxy_kept(t, proxy)
    assert not kept[7] and kept.sum() == len(t) - 1
    r = hd26.l09_band(t, proxy, h0)
    want = truth.hd26_m2_phase(t[kept], h0[kept])
    assert r["g_h0"] == pytest.approx(want["phase_deg"])
    assert r["n_h0_dropped"] == 0 and r["reason"] == "ok"
    every = truth.hd26_m2_phase(t, h0)
    assert every["n"] == len(t)                           # what the old fit used
