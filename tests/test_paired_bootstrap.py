# -*- coding: utf-8 -*-
"""experiments/paired_bootstrap.py on tiny synthetic data: exact multiset
medians, the replicate metrics against the published formulas on the
expanded resample, the clusters, the draws and the paired interval.

Run:  python -m pytest -q tests/test_paired_bootstrap.py
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import paired_bootstrap as pbs  # noqa: E402
from experiments import v14_scene_count_curves as v14  # noqa: E402


def _site(n=600, B=40, seed=0):
    rng = np.random.default_rng(seed)
    cl = np.sort(rng.integers(0, B, n))
    t = rng.uniform(-1.5, 1.2, n)
    shift = rng.normal(0, 0.1, B)[cl]                                    # an error shared inside a cluster
    a = 0.85 * t + 0.1 + shift + rng.normal(0, 0.10, n)
    b = 0.80 * t - 0.2 + shift + rng.normal(0, 0.25, n)
    return t, a, b, cl, B


def test_weighted_median_is_the_median_of_the_resampled_multiset():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 101)
    cl = rng.integers(0, 13, 101)
    wm = pbs.WeightedMedian(x, cl)
    for _ in range(50):
        k = rng.integers(0, 4, 13)
        k[rng.integers(0, 13)] = 0                                       # clusters drawn 0 times too
        expanded = np.repeat(x, k[cl])
        assert wm(k) == np.median(expanded) if len(expanded) else np.isnan(wm(k))
    assert wm(np.ones(13, int)) == np.median(x)
    assert np.isnan(pbs.WeightedMedian(x, cl)(np.zeros(13, int)))


def test_replicate_metrics_equal_the_published_formulas_on_the_expanded_resample():
    t, a, b, cl, B = _site()
    mt = pbs.WeightedMedian(t, cl)
    eng = pbs.MetricEngine(t, a, cl, B, med_t=mt)
    # all ones: the point estimate, which is v7.vscore / ols_against_truth exactly
    pt = pbs.point_metrics(t, a, "vaklodingen")
    sv = v14.score("vaklodingen", t, a)
    sr = v14.score("rtk", t, a)
    for k_, s_ in (("rmse", "rmse"), ("bias", "bias"), ("slope", "slope"), ("r", "r")):
        assert pt[k_] == sv[s_] == sr[s_]
    one = eng(np.ones(B, int))
    for k_ in ("rmse", "rmse_total", "bias", "abs_bias", "slope", "r"):
        assert one[k_] == pytest.approx(pt[k_], rel=0, abs=1e-12), k_
    rng = np.random.default_rng(2)
    for _ in range(20):
        k = np.bincount(rng.integers(0, B, B), minlength=B)
        w = k[cl]
        te, ae = np.repeat(t, w), np.repeat(a, w)
        ref = pbs.point_metrics(te, ae, "vaklodingen")
        got = eng(k)
        for k_ in ("rmse", "rmse_total", "bias", "abs_bias", "slope", "r"):
            assert got[k_] == pytest.approx(ref[k_], rel=0, abs=1e-10), k_
        s = 0.05 + 0.2 * np.abs(np.sin(t))
        ws = pbs.WithinSigmaEngine(t, a, s, cl)
        assert ws(k) == pytest.approx(pbs.point_within(te, ae, np.repeat(s, w)), abs=1e-12)
        fl = np.isfinite(np.where(t > 0, np.nan, 1.0))
        assert pbs.ShareEngine(fl, cl, B)(k) == pytest.approx(np.mean(np.repeat(fl, w)), abs=1e-12)
    with pytest.raises(ValueError):
        pbs.MetricEngine(t, np.where(t > 0, np.nan, a), cl, B)


def test_clusters_draws_and_the_paired_interval():
    lab, rule = pbs.cluster_labels("vaklodingen", np.array([0, 99, 100, 250]), np.array([0, 5, 0, 450]), 10.0)
    assert lab[0] == lab[1] != lab[2] and "1 km" in rule                # 100 px of 10 m per block
    lab, rule = pbs.cluster_labels("rtk", np.array([3, 3, 4]), np.array([7, 7, 7]))
    assert lab[0] == lab[1] != lab[2] and "pixel" in rule
    idx, B = pbs.cluster_index(np.array([50, 7, 50, 9]))
    assert B == 3 and idx.tolist() == [2, 0, 2, 1]
    c1 = pbs.resample_counts(17, 30, seed=5, key="x")
    assert c1.shape == (30, 17) and (c1.sum(axis=1) == 17).all()
    assert np.array_equal(c1, pbs.resample_counts(17, 30, seed=5, key="x"))
    assert not np.array_equal(c1, pbs.resample_counts(17, 30, seed=5, key="y"))
    # a planted difference: a has the smaller error; the paired interval excludes 0
    t, a, b, cl, B = _site(n=2000, B=60, seed=3)
    counts = pbs.resample_counts(B, 300, seed=1, key="t")
    mt = pbs.WeightedMedian(t, cl)
    reps = pbs.replicate(counts, {"a": pbs.MetricEngine(t, a, cl, B, med_t=mt),
                                  "b": pbs.MetricEngine(t, b, cl, B, med_t=mt)})
    pa, pb = pbs.point_metrics(t, a), pbs.point_metrics(t, b)
    iv = pbs.paired_interval(pa["rmse"], pb["rmse"], reps["a.rmse"], reps["b.rmse"])
    assert iv["diff"] < 0 and iv["ci_lo"] <= iv["diff"] <= iv["ci_hi"] < 0 and iv["share_diff_below_0"] == 1.0
    # the same estimate twice: a zero-width interval at 0
    z = pbs.paired_interval(pa["rmse"], pa["rmse"], reps["a.rmse"], reps["a.rmse"])
    assert z["diff"] == 0 and z["ci_lo"] == 0 == z["ci_hi"]
    # the shared cluster error is common to both, so the paired interval is narrower than either alone
    wa = np.quantile(reps["a.bias"], 0.975) - np.quantile(reps["a.bias"], 0.025)
    wd = np.quantile(reps["a.bias"] - reps["b.bias"], 0.975) - np.quantile(reps["a.bias"] - reps["b.bias"], 0.025)
    assert wd < wa
