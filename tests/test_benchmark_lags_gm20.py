# -*- coding: utf-8 -*-
"""Validation v2, package B: L08 GM20 (Geyman & Maloof 2020), on the synthetic
world of tests/test_benchmark_lags_aggregate.py.

Run: python -X utf8 -m pytest -q tests/test_benchmark_lags_gm20.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.benchmark import config, product, truth
from experiments.benchmark.lags import common, gm20
from tests.test_benchmark_lags_aggregate import (EPOCH_MIN, PHASES, SPEEDS, band_values,
                                                 lag_errors, make_world, unit)


def run(inst, variant="", opts=None):
    return gm20.run_unit(inst, unit("L08", ("L08",), variant), opts or {})[0]


def m2_hw_marks(t0_min, t1_min, shift_min=0.0):
    """Exact HW instants of the pure M2 tide of the synthetic world."""
    w = np.deg2rad(SPEEDS["M2"])                         # rad/h
    th0, th1 = (t0_min - EPOCH_MIN) / 60.0, (t1_min - EPOCH_MIN) / 60.0
    k = np.arange(np.floor((w * th0 + PHASES["M2"]) / (2 * np.pi)) - 1,
                  np.ceil((w * th1 + PHASES["M2"]) / (2 * np.pi)) + 2)
    t = EPOCH_MIN + 60.0 * (2 * np.pi * k - PHASES["M2"]) / w + shift_min
    return {"t": t, "h": np.zeros_like(t), "is_hw": np.ones(t.shape, bool),
            "sample_index": np.zeros(t.shape, np.int64)}


def test_closed_form_equals_truth_fold_phase_on_single_pixels():
    inst, w = make_world(seed=21, days=300)
    x = gm20.hours_since_hw(w["t_min"], inst.marks_lp)
    C, ok = gm20.fold_design(x)
    Y = inst.obs.Y[:, :60]
    V = inst.obs.V[:, :60] & ok[:, None]
    fit = gm20.fold_fit(C, Y, V, h0=inst.h0, W=inst.obs.W[:, :60])
    for j in range(60):
        m = V[:, j]
        ref = truth.fold_phase(x[m], Y[m, j].astype(np.float64))
        assert fit["n"][j] == ref["n"]
        assert fit["phase_y"][j] == pytest.approx(ref["phase_min"], abs=1e-7)
        assert fit["amp"][j] == pytest.approx(ref["amp"], rel=1e-9, abs=1e-12)
        assert fit["r2"][j] == pytest.approx(ref["r2"], abs=1e-9)
        refh = truth.fold_phase(x[m], inst.h0[m])
        assert fit["phase_h"][j] == pytest.approx(refh["phase_min"], abs=1e-7)
        sd = float(np.std(Y[m, j].astype(np.float64)))
        assert fit["sd"][j] == pytest.approx(sd, rel=1e-9)


def test_hours_since_the_latest_hw_mark():
    marks = {"t": np.array([100.0, 845.0, 1590.0, 470.0]),
             "is_hw": np.array([True, True, True, False])}
    x = gm20.hours_since_hw(np.array([50.0, 100.0, 400.0, 845.0, 2000.0]), marks)
    assert np.isnan(x[0])
    assert np.allclose(x[1:], [0.0, 5.0, 0.0, (2000.0 - 1590.0) / 60.0])


def test_rel_cancels_a_common_phase():
    inst, w = make_world(amps={"M2": 0.8}, seed=22, days=400)
    t0, t1 = w["t_min"].min() - 2000, w["t_min"].max() + 2000
    inst.marks_lp = m2_hw_marks(t0, t1)
    r0 = run(inst)
    inst.marks_lp = m2_hw_marks(t0, t1, shift_min=40.0)
    r1 = run(inst)
    f0, f1 = r0["field"].astype(float), r1["field"].astype(float)
    assert np.array_equal(np.isfinite(f0), np.isfinite(f1))
    fin = np.isfinite(f0)
    assert fin.sum() > 300
    assert np.max(np.abs(common.circ_diff(f0[fin], f1[fin]))) < 1e-3
    # the raw fold phases moved by -40 min, pixel and h0 alike
    x0 = gm20.hours_since_hw(w["t_min"], m2_hw_marks(t0, t1))
    x1 = gm20.hours_since_hw(w["t_min"], m2_hw_marks(t0, t1, 40.0))
    V = inst.obs.V[:, :20]
    a = gm20.fold_fit(gm20.fold_design(x0)[0], inst.obs.Y[:, :20], V, h0=inst.h0)
    b = gm20.fold_fit(gm20.fold_design(x1)[0], inst.obs.Y[:, :20], V, h0=inst.h0)
    assert np.allclose(common.circ_diff(b["phase_y"], a["phase_y"]), -40.0, atol=1e-6)
    assert np.allclose(common.circ_diff(b["phase_h"], a["phase_h"]), -40.0, atol=1e-6)


@pytest.mark.parametrize("lags", [(0.0, 45.0, 120.0), (200.0, 45.0, 0.0)])
def test_recovers_planted_lags_circularly(lags):
    inst, w = make_world(lags=lags, seed=23)
    r = run(inst)
    assert r["status"] == "ok" and r["kind"] == "field" and r["lag_kind"] == "phase"
    assert r["band"] is None and r["field"].dtype == np.float32
    v = band_values(r, w, circular=True)
    err = lag_errors(v, w["lags"], circular=True)
    assert np.median(err) <= 10.0 and np.all(err <= 15.0), v
    f = r["field"][np.isfinite(r["field"])]
    assert f.min() >= -120.0 and f.max() < -120.0 + common.PERIOD_M2_MIN    # wrap_gm20
    assert r["edge_share"] == 0.0 and r["interval95"] is None


def test_qc_rules_at_their_thresholds():
    inst, w = make_world(seed=24, days=400)
    x = gm20.hours_since_hw(w["t_min"], inst.marks_lp)
    C, ok = gm20.fold_design(x)
    T = len(x)
    rng = np.random.default_rng(0)
    wave = np.cos(2 * np.pi * (np.where(ok, x, 0) - 1.0) / gm20.CONFIG["period_h"])
    Y = np.stack([0.6 * wave + rng.normal(0, 0.05, T),          # 0 good
                  0.6 * wave + rng.normal(0, 0.05, T),          # 1 n = 19
                  0.6 * wave + rng.normal(0, 0.05, T),          # 2 n = 20
                  0.6 * wave + rng.normal(0, 0.05, T),          # 3 wet share 0.98
                  rng.normal(0, 0.5, T),                        # 4 R^2 < 0.1
                  0.1 * wave + rng.normal(0, 0.01, T)], 1)      # 5 SD < 0.2
    V = np.repeat(ok[:, None], 6, axis=1)
    idx = np.flatnonzero(ok)
    V[:, 1] = False
    V[idx[:19], 1] = True
    V[:, 2] = False
    V[idx[:20], 2] = True
    W = V & (Y > 0)
    W[:, 3] = V[:, 3]
    W[idx[: int(round(0.02 * len(idx)))], 3] = False             # 98 % wet
    fit = gm20.fold_fit(C, Y, V, h0=inst.h0, W=W)
    q = gm20.qc_mask(fit)
    good = q["n"] & q["sd"] & q["wet"] & q["r2"] & q["solved"]
    assert list(good) == [True, False, True, False, False, False]
    assert not q["n"][1] and q["n"][2]
    assert fit["wet"][3] >= 0.95 and not q["wet"][3]
    assert fit["r2"][4] < 0.1 and not q["r2"][4] and q["sd"][4]
    assert fit["sd"][5] <= 0.2 and not q["sd"][5] and q["r2"][5]


def test_qc_failures_are_nan_in_the_product():
    inst, w = make_world(seed=25)
    V, W = inst.obs.V, inst.obs.W
    rng = np.random.default_rng(1)
    few = 0                                   # n < 20
    keep_rows = np.flatnonzero(V[:, few])[:19]
    V[:, few] = False
    V[keep_rows, few] = True
    W[:, few] &= V[:, few]
    W[:, 1] = V[:, 1]                          # always wet
    inst.obs.Y[:, 2] = rng.normal(0, 0.5, w["T"]).astype(np.float32)   # no tide signal
    W[:, 2] = V[:, 2] & (inst.obs.Y[:, 2] > 0)
    r = run(inst)
    assert np.isnan(r["field"][:3]).all()
    assert r["diag"]["qc_fail_first"]["n"] >= 1
    assert r["diag"]["n_defined"] == int(np.isfinite(r["field"]).sum())


def test_refits_give_the_nominal_95_interval():
    inst, w = make_world(seed=26, days=500)
    r = run(inst, "boot30", {"refits": 20})
    iv = r["interval95"]
    assert iv["level"] == "pixel" and iv["nominal"] == gm20.NOMINAL
    assert iv["nominal"] == "iqr20x1.96/0.674xsqrt(m/(n-m))"
    tau = r["field"].astype(float)
    fin = np.isfinite(tau)
    assert np.isfinite(iv["lo"][fin]).mean() > 0.95
    ok = fin & np.isfinite(iv["lo"])
    assert np.all(iv["lo"][ok] <= tau[ok]) and np.all(iv["hi"][ok] >= tau[ok])
    assert np.isnan(iv["lo"][~fin]).all()
    assert r["diag"]["n_refits"] == 20 and r["diag"]["refit_subset_size"] == math.ceil(0.7 * w["T"])
    # one pixel by hand: the IQR of the circular refit differences, rescaled
    j = int(np.flatnonzero(ok)[0])
    masks, n_sub = gm20.refit_masks(w["T"], "wadden", "W3", 20)
    x = gm20.hours_since_hw(w["t_min"], inst.marks_lp)
    C, okx = gm20.fold_design(x)
    Vj = inst.obs.V[:, [j]] & okx[:, None]
    full = gm20.fold_fit(C, inst.obs.Y[:, [j]], Vj, h0=inst.h0, stats=False)
    rel = truth.wrap_gm20(full["phase_y"] - full["phase_h"])
    d = []
    for k in range(20):
        fk = gm20.fold_fit(C, inst.obs.Y[:, [j]], Vj & masks[k][:, None], h0=inst.h0,
                           stats=False)
        d.append(common.circ_diff(truth.wrap_gm20(fk["phase_y"] - fk["phase_h"]), rel)[0])
    iqr = np.percentile(d, 75) - np.percentile(d, 25)
    # the m-out-of-n subsample correction: the refits on m of T scenes spread
    # sqrt((T - m)/m) times less than the full-sample error
    fac = math.sqrt(n_sub / (w["T"] - n_sub))
    assert r["diag"]["subsample_factor"] == pytest.approx(fac)
    assert iv["hi"][j] - iv["lo"][j] == pytest.approx(iqr * 1.96 / 0.674 * fac, rel=1e-4,
                                                      abs=1e-3)
    # the refit fields are stored for the interval at the scorer's read-out
    R = r["diag"]["refit_tau"]
    assert R.shape == (20, len(tau)) and R.dtype == np.float32
    assert np.isnan(R[:, ~fin]).all() and np.isfinite(R[:, j]).all()
    dj = common.circ_diff(R[:, j].astype(float), tau[j])
    assert np.percentile(dj, 75) - np.percentile(dj, 25) == pytest.approx(iqr, abs=1e-3)
    assert r["diag"]["T"] == w["T"]


def test_refit_subsets_follow_the_seed_rule():
    masks, n_sub = gm20.refit_masks(463, "ems", "Y2024", 20)
    assert n_sub == 325 and masks.shape == (20, 463) and np.all(masks.sum(1) == 325)
    for k in (0, 7, 19):
        perm = np.random.default_rng([config.SEEDS["gm20"], config.SITE_ID["ems"], 2024,
                                      k]).permutation(463)
        expect = np.zeros(463, bool)
        expect[perm[:325]] = True
        assert np.array_equal(masks[k], expect)
    other, _ = gm20.refit_masks(463, "ems", "Y2024", 20, seed_offset=2)
    assert not np.array_equal(other, masks)


def test_no_half_period_for_nir_type_indices():
    inst, w = make_world(seed=27, days=300)
    a = run(inst)
    inst.nir_type = True
    b = run(inst)
    assert np.array_equal(a["field"], b["field"], equal_nan=True)
    assert b["diag"]["half_period_shift_applied"] is False and b["diag"]["nir_type"] is True


def test_streams_blocks_and_never_reads_the_whole_domain():
    inst, w = make_world(seed=28, days=300, allow_read=False)
    gm20.CONFIG["block"], old = 100, gm20.CONFIG["block"]
    try:
        r = run(inst)
    finally:
        gm20.CONFIG["block"] = old
    assert r["status"] == "ok"
    assert max(inst.obs.block_sizes) <= 100 and sum(inst.obs.block_sizes) == w["n"]


def test_statuses_and_refused_variants():
    inst, w = make_world(seed=29, days=90)                 # 18 scenes: n < 20 everywhere
    r = run(inst)
    assert r["status"] == "not_computable" and r["stage"] == "n_lt_20"
    inst, w = make_world(seed=29, days=300)
    inst.obs.Y[:] = np.float32(0.5)                         # SD 0: every pixel fails the QC
    r = run(inst)
    assert r["status"] == "empty" and r["stage"] == "qc"
    for v in ("raw", "scaled", "wet", "own"):
        with pytest.raises(ValueError):
            run(inst, v)
    inst, w = make_world(domain_status="no_intertidal", days=60)
    with pytest.raises(product.NotComputable):
        run(inst)


def test_card():
    c = gm20.CARDS["L08"]
    assert common.check_card(c) == [] and c["id"] == "L08"
    assert c["fidelity"] == "F3" and c["output"] == "field" and c["lag_kind"] == "phase"
    assert c["qc"] == {"min_n": 20, "min_sd": 0.2, "wet_lo": 0.05, "wet_hi": 0.95, "min_r2": 0.1}
    assert any("P/2" in d["paper"] for d in c["deviations"])
    from experiments.benchmark import lags
    assert lags.load("L08") is gm20.run_unit and lags.card("L08") is c
