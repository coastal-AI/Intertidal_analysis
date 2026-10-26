# -*- coding: utf-8 -*-
"""experiments/v14_scene_count_curves.py on tiny synthetic data: seeded
subsets, the production inversion on a subset, the fixed scoring set,
coverage, resumability, the budget shrink, and the RTK guard.

Run:  python -m pytest -q tests/test_scene_count_curves.py
"""
import ast
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402
from pyintertidal import marea  # noqa: E402

M2_H = 12.4206012


class FakeBoundary:
    """A boundary like PyTMDBoundary: levels returned sorted by time, and
    every call counted."""

    def __init__(self, amp=1.4):
        self.amp, self.calls, self.instants = amp, 0, 0

    def levels(self, times):
        t = pd.DatetimeIndex(times)
        self.calls += 1
        self.instants += len(t)
        hrs = (t.sort_values().values.astype("datetime64[s]").astype(np.int64) / 3600.0)
        return self.amp * np.sin(2 * np.pi * hrs / M2_H) + 0.2 * np.sin(2 * np.pi * hrs / 12.0)


def synthetic_site(T=60, P=48, clear_p=0.7, seed=0, lags=(0, 5, 10), units_rtk=False):
    """A site whose pixels flood when the lagged boundary level tops them."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01 11:00")
    times = pd.DatetimeIndex(sorted(t0 + pd.to_timedelta(rng.uniform(0, 900, T), unit="D")))
    B = FakeBoundary()
    lev = v14.LevelTable(B, times)
    tau = np.array([lags[i % len(lags)] for i in range(P)], float)
    lev.ensure(np.unique(tau))
    z_true = rng.uniform(-1.0, 1.0, P)
    H = np.stack([lev[tv] for tv in tau], axis=1)                      # (T, P)
    Y = np.where(H > z_true[None, :], 0.45, -0.30) + rng.normal(0, 0.05, (T, P))
    C = rng.random((T, P)) < clear_p
    Y = np.where(C, Y, np.nan).astype(np.float32)
    Y0 = np.nan_to_num(Y, nan=0.0).astype(np.float32)
    pools = {"screened": np.sort(rng.choice(T, T // 2, replace=False)), "all": np.arange(T)}
    if units_rtk:
        unit_col = np.repeat(np.arange(P), 2)                           # two points per pixel
        truth = z_true[unit_col] + rng.normal(0, 0.03, len(unit_col))
        kind = "rtk"
    else:
        unit_col = np.arange(P)
        truth = z_true + rng.normal(0, 0.03, P)
        kind = "vaklodingen"
    sd = v14.SiteData("synthetic", Y0, C, lev, pools, unit_col, truth, kind, tau)
    return sd, B, z_true


# ── subsets ──────────────────────────────────────────────────────────────────

def test_subsets_reproducible_from_seed():
    a = v14.draw(7, "escalda", "all", 30, 3, 464)
    b = v14.draw(7, "escalda", "all", 30, 3, 464)
    assert np.array_equal(a, b)
    assert len(a) == 30 and np.all(np.diff(a) > 0) and a.min() >= 0 and a.max() < 464
    # another replicate, pool, site or seed draws another subset
    for other in (v14.draw(7, "escalda", "all", 30, 4, 464), v14.draw(7, "escalda", "screened", 30, 3, 464),
                  v14.draw(7, "wadden", "all", 30, 3, 464), v14.draw(8, "escalda", "all", 30, 3, 464)):
        assert not np.array_equal(a, other)
    # the whole pool is the whole pool, whatever the replicate
    assert np.array_equal(v14.draw(7, "ems", "screened", 54, 0, 54), np.arange(54))
    assert np.array_equal(v14.draw(7, "ems", "screened", 60, 5, 54), np.arange(54))
    # independent of the order jobs run in (no shared generator state)
    v14.draw(1, "x", "all", 10, 0, 100)
    assert np.array_equal(v14.draw(7, "escalda", "all", 30, 3, 464), a)
    assert v14.site_id("escalda") == v14.site_id("escalda") != v14.site_id("wadden")


def test_job_order_is_replicate_major_and_complete():
    grids = {"screened": [10, 20, 54], "all": [10, 150, 462]}
    sizes = {"screened": 54, "all": 462}
    jobs = v14.make_jobs(grids, sizes, reps=4, reps_large=2)
    assert jobs[:2] == [("screened", 54, 0), ("all", 462, 0)]          # whole pools first
    assert len(jobs) == len(set(jobs))
    assert sum(1 for j in jobs if j[:2] == ("screened", 10)) == 4
    assert sum(1 for j in jobs if j[:2] == ("all", 150)) == 2           # N > 100: reps_large
    reps_seen = [r for p, n, r in jobs[2:]]
    assert reps_seen == sorted(reps_seen)                                # replicate-major
    assert v14.pool_grid("screened", 54) == [10, 15, 20, 25, 30, 40, 50, 54]
    assert v14.pool_grid("all", 462, v14.parse_n_grid("screened=10;all=10,500")) == [10, 462]


# ── the production inversion on a subset ────────────────────────────────────

def test_full_record_subset_reproduces_full_inversion():
    sd, _, _ = synthetic_site()
    rows = v14.draw(1, "synthetic", "all", 10 ** 6, 0, len(sd.pools["all"]))   # the whole pool
    z, nclear, (lo, hi), _, mism = v14.invert_subset(sd.Y0, sd.C, sd.levels, rows, sd.groups_fixed, [0, 3],
                                                     check_mb=True)
    h0 = sd.levels[0]
    assert (lo, hi) == (float(h0.min()), float(h0.max()))
    ref0 = np.full(sd.Y0.shape[1], np.nan)
    ref3 = np.full(sd.Y0.shape[1], np.nan)
    for tv, cols in sd.groups_fixed:                                     # the production loop, direct
        ref0[cols], _ = marea.invert_series(sd.Y0[:, cols], sd.C[:, cols].astype(np.float32), sd.levels[tv],
                                            lo, hi)
        ref3[cols], _ = marea.invert_series(sd.Y0[:, cols], sd.C[:, cols].astype(np.float32), sd.levels[tv],
                                            lo, hi, min_bracket=3)
    assert np.array_equal(z[0], ref0, equal_nan=True)                    # bitwise
    assert np.array_equal(z[3], ref3, equal_nan=True)                    # post-filter == min_bracket=3
    assert mism == 0
    assert np.isfinite(z[0]).sum() > 0.8 * len(ref0)
    assert np.array_equal(nclear, sd.C.sum(axis=0))


def test_blocks_reproduce_one_call_bitwise(monkeypatch):
    # more columns than one block: the blocks are whole invert_series chunks
    rng = np.random.default_rng(3)
    T, P = 30, 9000
    h = np.sort(rng.uniform(-1.5, 1.5, T))
    z_true = rng.uniform(-1, 1, P)
    C = rng.random((T, P)) < 0.8
    Y0 = np.where(C, np.where(h[:, None] > z_true[None, :], 0.4, -0.3), 0.0).astype(np.float32)

    class Flat:
        def __getitem__(self, tau):
            return h

    monkeypatch.setattr(v14, "BLOCK", 4000)
    z, *_ = v14.invert_subset(Y0, C, Flat(), np.arange(T), [(0, np.arange(P))], [0, 3])
    ref, _ = marea.invert_series(Y0, C, h, float(h.min()), float(h.max()))
    assert np.array_equal(z[0], ref, equal_nan=True)


def test_bracket_filter_is_the_quality_rule():
    rng = np.random.default_rng(5)
    h = rng.uniform(-1, 1, 80)
    zt = np.array([0.0, 0.93, -0.95, 0.4])
    Y = np.where(h[:, None] > zt[None, :], 0.4, -0.3) + rng.normal(0, 0.05, (80, 4))
    C = rng.random((80, 4)) < 0.9
    z0, _ = marea.invert_series(Y, C.astype(float), h, -1, 1)
    for mb in (1, 3, 5):
        z_mb, _ = marea.invert_series(Y, C.astype(float), h, -1, 1, min_bracket=mb)
        assert np.array_equal(v14.bracket_filter(z0, C, h, mb), z_mb, equal_nan=True)
    assert np.array_equal(v14.bracket_filter(z0, C, h, 0), z0, equal_nan=True)


def test_level_table_one_call_equals_per_lag_calls():
    rng = np.random.default_rng(2)
    times = pd.DatetimeIndex(sorted(pd.Timestamp("2023-03-01") + pd.to_timedelta(rng.uniform(0, 300, 40), "D")))
    B = FakeBoundary()
    lev = v14.LevelTable(B, times)
    lev.ensure([0, 7, 30, 180])
    assert B.calls == 1                                                  # one call for every lag
    lev.ensure([7, 30])
    assert B.calls == 1                                                  # nothing new to fetch
    for tv in (0, 7, 30, 180):
        direct = FakeBoundary().levels(times - pd.Timedelta(minutes=tv))
        assert np.array_equal(lev[tv], direct)
    with pytest.raises(ValueError):
        lev.ensure([2.5])


def test_stream_columns_matches_np_load(tmp_path):
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(9, 13)).astype(np.float32)
    C = rng.random((9, 13)) < 0.5
    p = tmp_path / "ex.npz"
    np.savez_compressed(p, Y=Y, C=C)
    rows, c1, c2 = [1, 4, 8], [0, 5, 12], [3]
    a, b = v14.stream_columns(str(p), "Y", rows, [c1, c2])
    assert np.array_equal(a, Y[rows][:, c1]) and np.array_equal(b, Y[rows][:, c2])
    (c,) = v14.stream_columns(str(p), "C", rows, [c1])
    assert c.dtype == bool and np.array_equal(c, C[rows][:, c1])
    with pytest.raises(ValueError):
        v14.stream_columns(str(p), "Y", [4, 1], [c1])


# ── the fixed set, coverage, resumability ───────────────────────────────────

def test_fixed_pixel_set_used_at_every_N_and_coverage_reported(tmp_path, monkeypatch):
    sd, _, _ = synthetic_site(units_rtk=True)
    seen = []
    real_score = v14.score

    def spy(kind, truth, p):
        seen.append((truth.copy(), len(p)))
        return real_score(kind, truth, p)

    monkeypatch.setattr(v14, "score", spy)
    grids = {"screened": [6, 15, 30], "all": [6, 30, 60]}
    sizes = {p: len(v) for p, v in sd.pools.items()}
    jobs = v14.make_jobs(grids, sizes, reps=2, reps_large=1)
    log = v14.CsvLog(str(tmp_path / "s.csv"), [0, 3])
    n_done, n_skip, stopped = v14.run_jobs(sd, jobs, log, 11, "fixed", [0, 3], time.time() + 600,
                                           lambda p, N: 0.01)
    assert (n_done, n_skip, stopped) == (len(jobs), 0, None)
    # the same units, the same truth, every N and every rule
    assert len(seen) == 2 * len(jobs)
    for truth, n in seen:
        assert n == len(sd.unit_col) and np.array_equal(truth, sd.truth)
    df = pd.read_csv(tmp_path / "s.csv")
    assert set(v14.CSV_COLUMNS) <= set(df.columns)
    assert len(df) == 2 * len(jobs) and set(df["mb"]) == {0, 3}
    assert (df["n_set"] == len(sd.unit_col)).all()
    assert df["coverage"].between(0, 1).all()
    assert np.allclose(df["coverage"] * df["n_set"], df["n"])           # truth finite on every unit
    full = df[(df["pool"] == "all") & (df["N"] == 60) & (df["mb"] == 0)]["coverage"].iloc[0]
    tiny = df[(df["pool"] == "all") & (df["N"] == 6) & (df["mb"] == 0)]["coverage"].max()
    assert full > 0.9 and tiny < full                                    # < min_obs clear scenes at N=6
    assert (df["mb"] == 3).sum() == (df["mb"] == 0).sum()
    m = df.pivot_table(index=["pool", "N", "replicate"], columns="mb", values="coverage")
    assert (m[3] <= m[0] + 1e-12).all()                                  # the rule only removes estimates
    # resumable: a second pass appends nothing
    log2 = v14.CsvLog(str(tmp_path / "s.csv"), [0, 3])
    n_done2, n_skip2, _ = v14.run_jobs(sd, jobs, log2, 11, "fixed", [0, 3], time.time() + 600,
                                       lambda p, N: 0.01)
    assert (n_done2, n_skip2) == (0, len(jobs))
    assert len(pd.read_csv(tmp_path / "s.csv")) == len(df)


def test_partial_last_line_is_dropped_and_job_redone(tmp_path):
    sd, _, _ = synthetic_site()
    jobs = [("all", 60, 0), ("screened", 10, 0)]
    path = str(tmp_path / "p.csv")
    log = v14.CsvLog(path, [0, 3])
    v14.run_jobs(sd, jobs, log, 3, "fixed", [0, 3], time.time() + 600, lambda p, N: 0.01)
    ref = pd.read_csv(path)
    text = open(path, encoding="utf-8").read()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text[:-15])                         # killed mid-write: the last job has one rule only
    log2 = v14.CsvLog(path, [0, 3])
    assert log2.is_done("all", 60, 0, [0, 3]) and not log2.is_done("screened", 10, 0, [0, 3])
    assert len(pd.read_csv(path)) == 2              # the half-written job's rows are gone
    n_done, n_skip, _ = v14.run_jobs(sd, jobs, log2, 3, "fixed", [0, 3], time.time() + 600,
                                     lambda p, N: 0.01)
    assert (n_done, n_skip) == (1, 1)
    df = pd.read_csv(path)
    assert len(df) == 4 and df.groupby(["pool", "N", "replicate"]).size().eq(2).all()
    cols = ["pool", "N", "replicate", "mb", "rmse", "coverage", "n", "draw_sha1"]
    pd.testing.assert_frame_equal(df[cols].reset_index(drop=True), ref[cols].reset_index(drop=True))


def test_deadline_stops_before_a_job(tmp_path):
    sd, _, _ = synthetic_site()
    jobs = v14.make_jobs({"all": [10, 30, 60]}, {"all": 60}, reps=3, reps_large=3)
    log = v14.CsvLog(str(tmp_path / "d.csv"), [0])
    n_done, _, stopped = v14.run_jobs(sd, jobs, log, 3, "fixed", [0], time.time() + 1.0,
                                      lambda p, N: 30.0)                # predicted past the deadline
    assert n_done == 0 and stopped is not None and stopped["jobs_left"] == len(jobs)
    assert stopped["at_job"] == list(jobs[0])
    n_done, _, stopped = v14.run_jobs(sd, jobs, log, 3, "fixed", [0], time.time() + 600,
                                      lambda p, N: 0.01)
    assert n_done == len(jobs) and stopped is None


def test_shrink_reduces_replicates_first_then_the_grid():
    grids = {"screened": [10, 20, 30, 40, 50, 60, 68], "all": [10, 20, 40, 80, 150, 300, 464]}
    sizes = {"screened": 68, "all": 464}
    cost = lambda p, N: float(N)                                         # noqa: E731
    full = sum(cost(p, N) for p, N, _ in v14.make_jobs(grids, sizes, 10, 5))
    g, r, rl, jobs, notes, pred0, pred = v14.shrink_plan(grids, sizes, 10, 5, cost, full + 1)
    assert (g, r, rl, notes) == (grids, 10, 5, []) and pred0 == pred == full
    g, r, rl, jobs, notes, _, pred = v14.shrink_plan(grids, sizes, 10, 5, cost, 0.6 * full)
    assert g == grids and 3 <= r < 10 and pred <= 0.6 * full and "replicates reduced" in notes[0]
    g, r, rl, jobs, notes, _, pred = v14.shrink_plan(grids, sizes, 10, 5, cost, 0.12 * full)
    assert r == 3 and any("thinned" in n for n in notes)
    for p in g:
        assert g[p][0] == grids[p][0] and g[p][-1] == grids[p][-1]       # smallest N and whole pool kept
        assert len(g[p]) < len(grids[p])


# ── lag refit plumbing ───────────────────────────────────────────────────────

def test_refit_recovers_a_planted_band_lag():
    rng = np.random.default_rng(8)
    T, per = 90, 120
    times = pd.DatetimeIndex(sorted(pd.Timestamp("2024-01-01 11:00") + pd.to_timedelta(rng.uniform(0, 900, T), "D")))
    lev = v14.LevelTable(FakeBoundary(amp=1.6), times)
    lev.ensure([0, 40])
    band = np.repeat([0, 1], per)
    z = rng.uniform(-1.2, 1.2, 2 * per)
    H = np.where(band[None, :] == 0, lev[0][:, None], lev[40][:, None])
    Ym = np.where(H > z[None, :], 0.4, -0.3).astype(np.float32)
    Cm = np.ones_like(Ym, bool)
    cfg_fit = dict(sigma0=0.2, sg_grid=(0.06, 0.1, 0.15, 0.22, 0.32, 0.45), tau_bounds=marea.TAU_BOUNDS,
                   z_points=60, max_px_band=1200, monotone=True)
    m2a = dict(Y=Ym, C=Cm, band=band, centers=np.array([1.5, 4.5]), nb=2, band_km=3.0, fit_kw=cfg_fit,
               seed=20260819)
    s_km = np.array([0.5, 1.5, 3.0, 4.5, 6.0, np.nan])
    sd = v14.SiteData("synthetic", np.zeros((T, 6), np.float32), np.ones((T, 6), bool), lev,
                      {"all": np.arange(T)}, np.arange(6), np.zeros(6), "rtk", np.zeros(6), s_km=s_km, m2a=m2a)
    tau, tau_used, tp = v14.refit_lags(sd, np.arange(T))
    assert tau[0] == 0.0 and abs(tau[1] - 40.0) < 8.0                   # mouth pinned, lag recovered
    assert np.all(tau_used >= 0)
    assert np.all(tp == np.round(tp)) and tp[-1] == 0.0 and np.all(np.diff(tp[:5]) >= 0)
    tau2, _, tp2 = v14.refit_lags(sd, np.arange(T))
    assert np.array_equal(tau, tau2) and np.array_equal(tp, tp2)        # deterministic


# ── the RTK guard ────────────────────────────────────────────────────────────

def test_rtk_guard_passes():
    from tests import test_r1_guard
    test_r1_guard.test_static_no_allow_reserved()                       # scans experiments/, this script too
    tree = ast.parse(open(v14.__file__, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "attr", None) == "load_rtk" or getattr(n.func, "id", None) == "load_rtk")]
    assert calls, "the script reads the survey through pyintertidal.rtk.load_rtk"
    assert all(not c.args and not c.keywords for c in calls), "load_rtk must be called with default arguments only"


def test_output_guard_refuses_production_paths():
    for bad in ("products_marea/x", "products_escalda_10m/marea_eot20_g2",
                "results/method_comparison_same_scenes"):
        with pytest.raises(SystemExit):
            v14.guard_out(bad)
    v14.guard_out("results/scene_count_curves")
