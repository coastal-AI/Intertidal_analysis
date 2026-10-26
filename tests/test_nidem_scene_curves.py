# -*- coding: utf-8 -*-
"""experiments/v15_nidem_scene_curves.py on tiny synthetic data: the subsets
are v14's draws, the sort-free R equals the library's, the whole pool
reproduces wc.nidem_from_arrays exactly, not-computable subsets are recorded
with coverage 0 (never dropped), the scoring set is the same at every N,
resumability, the figure, and the RTK guard.

Run:  python -m pytest -q tests/test_nidem_scene_curves.py
"""
import ast
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402
from experiments import v15_nidem_scene_curves as v15  # noqa: E402
from pyintertidal import waterline_contour as wc  # noqa: E402

SEED = 20261019


def synthetic_cube(T=72, H=36, W=44, seed=0, cloud=0.35, swap=0.12):
    """Integer reflectances over a sloping flat: water (B03 > B08) where the
    tide tops the bed; clouds (SCL 9); some observations with B03/B08 swapped
    so that +x / -x NDWI pairs give exact median ties."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    elev = (-1.1 + 2.2 * xx / (W - 1) + 0.25 * np.sin(yy / 5.0)).astype(np.float32)
    h = rng.uniform(-1.3, 1.3, T)
    wet = h[:, None, None] > elev[None]
    g = rng.integers(200, 2500, (T, H, W)).astype(np.float32)
    n = np.where(wet, g * rng.uniform(0.3, 0.9, (T, H, W)), g * rng.uniform(1.1, 3.0, (T, H, W)))
    n = np.round(n).astype(np.float32)
    sw = rng.random((T, H, W)) < swap
    b03, b08 = np.where(sw, n, g), np.where(sw, g, n)
    scl = np.where(rng.random((T, H, W)) < cloud, 9.0, 4.0).astype(np.float32)
    ndwi, valid = wc.scene_ndwi(b03, b08, scl, rule="item")
    return dict(ndwi=ndwi, valid=valid, D=np.where(valid, ndwi, np.nan).astype(np.float32), h=h, elev=elev,
                T=T, H=H, W=W)


def make_ctx(cube, subsets, keep, units_keep, truth, kind="vaklodingen", row_block=16):
    T, H, W = cube["T"], cube["H"], cube["W"]
    D = cube["D"]
    units_flat = keep[units_keep]
    unit_px = np.unique(units_flat)
    p1 = v15.phase1(lambda r0, r1: D[:, r0:r1].copy(), (T, H, W), subsets, keep, unit_px, row_block)
    return dict(site="synthetic", size=T, subsets=subsets, p1=p1, keep=keep, units_keep=units_keep,
                units_flat=units_flat, unit_px=unit_px, unit_col=np.searchsorted(unit_px, units_flat),
                truth=np.asarray(truth, float), kind=kind, H=H, W=W, time_index=np.arange(T))


def keep_and_units(cube, seed=0, share=0.6, n_units=160):
    rng = np.random.default_rng(seed)
    HW = cube["H"] * cube["W"]
    keep = np.sort(rng.choice(HW, int(share * HW), replace=False))
    units_keep = np.sort(rng.choice(len(keep), n_units, replace=False))
    truth = cube["elev"].ravel()[keep[units_keep]].astype(float) + rng.normal(0, 0.05, n_units)
    return keep, units_keep, truth


# ── subsets ──────────────────────────────────────────────────────────────────

def test_subsets_are_v14_draws_in_v14_order():
    h = np.random.default_rng(1).uniform(-1, 1, 120)
    grid = [10, 30, 60, 120]
    jobs = v14.make_jobs({"all": grid}, {"all": 120}, 4, 2)
    msettings = {"plan": {"grid": {"all": grid}, "reps": 4, "reps_large": 2}, "pools": {"all": {"size": 120}}}
    grid2, reps, reps_large, size, jobs2 = v15.plan_from_settings(msettings)
    assert (grid2, reps, reps_large, size, jobs2) == (grid, 4, 2, 120, jobs)
    subs = v15.build_subsets(SEED, "escalda", 120, jobs, h, extra=[("marea_record", np.arange(0, 120, 3))])
    assert len(subs) == len(jobs) + 1
    for (pool, N, rep), s in zip(jobs, subs):
        idx = v14.draw(SEED, "escalda", "all", N, rep, 120)          # the same function, the same arguments
        assert np.array_equal(s["idx"], idx) and s["sha"] == v14.draw_sha1(idx)
        assert (s["N"], s["rep"]) == (len(idx), rep)
    assert subs[0]["N"] == 120 and subs[0]["gate"]                    # the whole pool is a gate subset
    assert subs[-1]["key"] == "marea_record" and subs[-1]["rep"] == -1 and subs[-1]["gate"]


def test_server_villaviciosa_draws_reproduced():
    """The laptop copy of v14's Villaviciosa refit run: every 'all'-pool job
    and its draw_sha1 (no heavy data needed)."""
    d = os.path.join(ROOT, "results", "scene_count_curves", "server_2026-10-20")
    if not os.path.exists(os.path.join(d, "villaviciosa_refit.csv")):
        pytest.skip("no laptop copy of v14's server outputs")
    st = json.load(open(os.path.join(d, "villaviciosa_refit_settings.json"), encoding="utf-8"))
    grid, reps, reps_large, size, jobs = v15.plan_from_settings(st)
    rows = v15.marea_rows(os.path.join(d, "villaviciosa_refit.csv"))
    want = {(int(r.N), int(r.replicate)): r.draw_sha1 for r in rows.itertuples()}
    have = {(len(idx), rep): v14.draw_sha1(idx) for idx, rep in
            ((v14.draw(st["seed"], "villaviciosa", "all", N, rep, size), rep) for _, N, rep in jobs)}
    assert have == want and len(have) == 101


def test_interval_sizes_is_the_library_binning():
    rng = np.random.default_rng(7)
    seen_empty = seen_ok = 0
    for _ in range(60):
        h = rng.uniform(-1, 1, int(rng.integers(5, 40)))
        sizes = v15.interval_sizes(h)
        try:
            _, _, _, _, lib = wc.interval_assignment(h)
        except wc.EmptyIntervalError:
            assert (sizes == 0).any()
            seen_empty += 1
            continue
        assert np.array_equal(sizes, lib)
        seen_ok += 1
    assert seen_empty and seen_ok


# ── the sort-free R ──────────────────────────────────────────────────────────

def _flags(D):
    Df = D.reshape(D.shape[0], -1)
    with np.errstate(invalid="ignore"):
        return Df, np.isfinite(Df).view(np.uint8), (Df > 0).view(np.uint8)


def test_fast_codes_equal_library_R():
    c = synthetic_cube(T=90, H=20, W=30, seed=2)
    D, h = c["D"], c["h"]
    Df, FIN, POS = _flags(D)
    rng = np.random.default_rng(3)
    n_tie = n_checked = 0
    for N in (25, 45, 90):
        for _ in range(3):
            idx = np.sort(rng.choice(90, N, replace=False))
            try:
                j, *_ = wc.interval_assignment(h[idx])
            except wc.EmptyIntervalError:
                continue
            code, _, _, nt = v15.fast_codes(FIN, POS, Df, [idx[j == q] for q in range(1, 10)])
            comp = wc.composites_from_arrays(D[idx], np.ones((N, 20, 30), bool), j)
            R_lib = wc.relative_extents(comp["median"], comp["count"])[0]
            R_fast = np.where(code != 255, code, np.nan).astype(np.float32).reshape(20, 30)
            assert np.array_equal(R_lib, R_fast, equal_nan=True)
            n_tie += nt
            n_checked += 1
    assert n_checked >= 5 and n_tie > 0                              # the tie branch was exercised


def _one_interval_code(values):
    """Code of one pixel whose interval 1 holds ``values`` and intervals 2..9
    one water observation each: 1 if interval 1 is land, else 0."""
    v = np.asarray(values, np.float32)
    D = np.concatenate([v, np.full(8, 0.5, np.float32)])[:, None]
    Df, FIN, POS = _flags(D)
    members = [np.arange(len(v))] + [np.array([len(v) + q]) for q in range(8)]
    code = v15.fast_codes(FIN, POS, Df, members)[0][0]
    # the library on the same stack
    med, _, cnt = wc.interval_statistics(v.copy()[:, None, None])
    lib_land = bool(wc.land_flags(med)[0, 0])
    assert int(code) == int(lib_land), (values, code, med)
    return int(code)


def test_explicit_ties_and_large_intervals():
    up = float(np.nextafter(np.float32(0.2), np.float32(1)))
    assert _one_interval_code([-0.2, 0.2]) == 1                      # median exactly 0: land
    assert _one_interval_code([0.0, -0.0]) == 1
    assert _one_interval_code([-0.2, up]) == 0                       # median just above 0: water
    assert _one_interval_code([np.nan, 0.5]) == 0
    assert _one_interval_code([-0.1, 0.2, 0.3]) == 0
    assert _one_interval_code([-0.1, -0.2, 0.3]) == 1
    assert _one_interval_code([-0.1, 0.1, 0.2, -0.3]) == 1           # tie: (-0.1 + 0.1) / 2 = 0
    assert _one_interval_code([-0.3, 0.31, 0.2, -0.1, np.nan]) == 0  # tie: (-0.1 + 0.2) / 2 > 0
    assert _one_interval_code([0.3]) == 0 and _one_interval_code([-0.3]) == 1
    rng = np.random.default_rng(5)
    for n_pos, n_neg in ((150, 150), (151, 150), (140, 160), (200, 100)):   # > 255 members: uint16 counts
        vals = np.concatenate([rng.uniform(0.01, 1, n_pos), -rng.uniform(0, 1, n_neg)])
        _one_interval_code(rng.permutation(vals))


# ── the whole pool reproduces the library NIDEM ─────────────────────────────

def test_full_pool_reproduces_nidem_from_arrays():
    c = synthetic_cube(T=72, H=36, W=44, seed=3)
    keep, units_keep, truth = keep_and_units(c, seed=4)
    h = c["h"]
    rec_idx = np.arange(0, 72, 2)
    subs = v15.build_subsets(1, "synthetic", 72, [("all", 72, 0)], h, extra=[("marea_record", rec_idx)])
    assert subs[0]["status"] == subs[1]["status"] == "assigned"
    ctx = make_ctx(c, subs, keep, units_keep, truth, row_block=16)    # 3 blocks of rows
    for k, idx in ((0, np.arange(72)), (1, rec_idx)):
        s = subs[k]
        ref, (j, edges, Z, U, sizes) = wc.nidem_from_arrays(c["ndwi"][idx], c["valid"][idx], h[idx])
        row, res = v15.stage2(ctx, k, keep_conf=ctx["p1"]["gate"][s["key"]])
        assert row["status"] == "ok" and ref["n_vertices"] > 50
        for key in ("R", "valid", "R_filled", "dem", "unfiltered", "uncertainty"):        # full frame
            assert np.array_equal(res[key], ref[key], equal_nan=True), key
        for key in ("filtered", "conf", "filtered_bounded", "conf_bounded"):              # at keep
            assert np.array_equal(res[key].ravel()[keep], ref[key].ravel()[keep], equal_nan=True), key
        rc, lv = wc.contour_vertices(res["contours"])
        rc0, lv0 = wc.contour_vertices(ref["contours"])
        assert np.array_equal(rc, rc0) and np.array_equal(lv, lv0)
        for a, b in ((s["j"], j), (s["Z"], Z), (s["U"], U), (s["sizes"], sizes), (s["edges"], edges)):
            assert np.array_equal(a, b, equal_nan=True)
        # the phase-1 library composites at keep are the library's composites there
        comp = wc.composites_from_arrays(c["ndwi"][idx], c["valid"][idx], j)
        g = ctx["p1"]["gate"][s["key"]]
        assert np.array_equal(g["median"], comp["median"].reshape(9, -1)[:, keep], equal_nan=True)
        assert np.array_equal(g["count"], comp["count"].reshape(9, -1)[:, keep])
        # a curve job (confidence at the scored pixels only) has the same values at the units
        row2, res2 = v15.stage2(ctx, k)
        fu = ctx["units_flat"]
        assert np.array_equal(res2["filtered"].ravel()[fu], ref["filtered"].ravel()[fu], equal_nan=True)
        sc = v14.score("vaklodingen", truth, ref["filtered"].ravel()[fu])
        for key in ("rmse", "bias", "slope", "r", "n", "coverage"):
            assert row2[key] == row[key] == sc[key]
        assert 0.1 < row["coverage"] <= 1.0 and np.isfinite(row["rmse"])   # something is scored


# ── not computable, coverage, the fixed set ─────────────────────────────────

def _site_with_a_lonely_cloudy_low_scene(T=60, seed=4):
    """The Ems case: the lowest scene sits alone in the lowest interval of any
    set that holds it, and it is overcast."""
    c = synthetic_cube(T=T, H=24, W=30, seed=seed)
    lo = int(np.argmin(c["h"]))
    rest = np.delete(c["h"], lo)
    a, b = rest.min(), rest.max()
    c["h"][lo] = a - 1.3 * (b - a) / 9                 # below e_1 of the pool, the rest above it
    c["valid"][lo] = False
    c["D"][lo] = np.nan
    return c, lo


def test_not_computable_recorded_with_coverage_zero(tmp_path):
    c, lo = _site_with_a_lonely_cloudy_low_scene()
    keep, units_keep, truth = keep_and_units(c, seed=6, n_units=120)
    jobs = v14.make_jobs({"all": [8, 20, 40, 60]}, {"all": 60}, 4, 2)
    subs = v15.build_subsets(5, "synthetic", 60, jobs, c["h"])
    ctx = make_ctx(c, subs, keep, units_keep, truth)
    log = v15.RowLog(str(tmp_path / "x.csv"))
    n_done, stop = v15.run_jobs(ctx, list(range(len(jobs))), log, 1, time.time() + 600, 1.0)
    assert stop is None and n_done == len(jobs)
    df = pd.read_csv(tmp_path / "x.csv")
    assert len(df) == len(jobs) and not df.duplicated(["N", "replicate"]).any()        # nobody dropped
    nc = df[df["status"].isin(v15.NOT_COMPUTABLE)]
    assert (nc["coverage"] == 0).all() and (nc["n"] == 0).all() and nc["rmse"].isna().all()
    assert nc["reason"].str.len().gt(0).all()
    full = df[df["N"] == 60].iloc[0]
    assert full["status"] == v15.STATUS_PREFLIGHT and full["keep_valid_share"] == 0.0
    assert (df["status"] == v15.STATUS_EMPTY).any()
    assert (df["status"] == v15.STATUS_OK).any()        # draws without the lonely scene can be computable
    ok = df[df["status"] == v15.STATUS_OK]
    assert ok["coverage"].between(0, 1).all()
    # computability is not monotone in N here: the pool stops, smaller draws do not
    assert ok["N"].max() < 60
    # the pre-flight numbers are wc.interval_coverage's
    D = c["D"].reshape(60, -1)
    for s in subs:
        if s["status"] != "assigned":
            continue
        counts = np.stack([np.isfinite(D[s["members"][q]][:, keep]).sum(0) for q in range(9)])
        lib = wc.interval_coverage(counts, raise_error=False)
        mine = v15.preflight_numbers(ctx["p1"]["n_has"][s["slot"]], ctx["p1"]["n_valid"][s["slot"]], len(keep))
        for key in ("px", "share_px_with_ge1_clear_obs", "intervals_with_zero_clear_obs",
                    "intervals_below_floor", "expected_valid_px", "expected_valid_share", "passed"):
            assert mine[key] == lib[key], key
    # the summary per N counts every draw
    cnt = v15.nc_counts(log.done.values(), [8, 20, 40, 60])
    assert sum(v["draws"] for v in cnt.values()) == len(jobs)
    assert cnt["60"]["not_computable_preflight"] == 1
    # every pre-flight stop gets the most it could have covered without the stop
    stops = v15.preflight_stop_units(ctx)
    pre_rows = df[df["status"] == v15.STATUS_PREFLIGHT]
    assert {(int(r.N), int(r.replicate)) for r in pre_rows.itertuples()} == {(v["N"], v["rep"]) for v in stops.values()}
    for v in stops.values():
        assert 0 <= v["units_R_1_8"] <= v["units_valid_R"] <= v["n_units"] == len(truth)
        assert v["max_coverage_without_stop"] == v["units_R_1_8"] / v["n_units"]
    assert next(v for v in stops.values() if v["N"] == 60)["units_valid_R"] == 0   # the lonely scene is overcast


def test_full_frame_audit_and_cache(tmp_path, monkeypatch):
    """Phase 1 checks the fast R of the audited curve draws against the
    library's R on every pixel (0 differing; a code wrong at one pixel that
    no unit covers is caught), and the phase-1 cache round-trips, keyed by
    the audit choice and the phase-1 code."""
    c = synthetic_cube(T=72, H=36, W=44, seed=3)
    keep, units_keep, truth = keep_and_units(c, seed=4)
    jobs = v14.make_jobs({"all": [20, 40, 72]}, {"all": 72}, 3, 2)
    subs = v15.build_subsets(5, "synthetic", 72, jobs, c["h"])
    audit = v15.audit_choice(subs)
    cand = [s for s in subs if s["status"] == "assigned" and not s["gate"] and s["rep"] >= 0]
    chosen = [next(s for s in subs if s["key"] == a) for a in audit]
    assert len(audit) == 2 and len(set(audit)) == 2 and all(s in cand for s in chosen)
    # the draw closest to N = 40 whose sparsest interval is fullest (then the lowest replicate),
    # then the closest to 80 among the rest; chosen from the plan alone
    assert chosen[0] is min(cand, key=lambda s: (abs(s["N"] - 40), s["N"], -int(s["sizes"].min()), s["rep"]))
    assert abs(chosen[1]["N"] - 80) == min(abs(s["N"] - 80) for s in cand if s is not chosen[0])
    near = [s for s in cand if s["N"] == chosen[0]["N"]]
    assert chosen[0]["sizes"].min() == max(s["sizes"].min() for s in near)
    D = c["D"]
    unit_px = np.unique(keep[units_keep])
    p1 = v15.phase1(lambda r0, r1: D[:, r0:r1].copy(), (72, 36, 44), subs, keep, unit_px, 16, audit=audit)
    assert sorted(p1["audit"]) == sorted(audit)
    for a in p1["audit"].values():
        assert a["px"] == 36 * 44 and a["differing_px"] == 0 and a["valid_px_library"] == a["valid_px_fast"] > 0
    # the cache: the same arrays and audit back, only under the same key
    info = {"path": "cube.nc", "bytes": 1, "mtime": "t"}
    key = v15.phase1_key("synthetic", info, np.arange(72), c["h"], keep, unit_px, subs, audit)
    path = str(tmp_path / "p1.npz")
    v15.save_phase1(path, p1, key)
    back = v15.load_phase1(path, key, subs)
    assert back["audit"] == p1["audit"] and np.array_equal(back["codes"], p1["codes"])
    assert np.array_equal(back["Du"], p1["Du"], equal_nan=True)
    assert v15.load_phase1(path, "another key", subs) is None
    assert v15.phase1_key("synthetic", info, np.arange(72), c["h"], keep, unit_px, subs, audit[:0]) != key
    assert len(v15.phase1_code_digest()) == 64
    # a fast code wrong at one pixel that no unit covers: the audit sees it
    target = next(s for s in subs if s["key"] == audit[0])
    px = next(p for p in range(16 * 44) if p not in set(unit_px.tolist()))
    real = v15.fast_codes
    seen = []

    def wrong(FIN, POS, Df, members, kloc=None):
        out = real(FIN, POS, Df, members, kloc)
        if members is target["members"] and not seen:                # the first block of the target only
            seen.append(1)
            code = out[0].copy()
            code[px] = 3 if code[px] != 3 else 4
            out = (code,) + tuple(out[1:])
        return out

    monkeypatch.setattr(v15, "fast_codes", wrong)
    p1w = v15.phase1(lambda r0, r1: D[:, r0:r1].copy(), (72, 36, 44), subs, keep, unit_px, 16, audit=audit)
    assert p1w["audit"][audit[0]]["differing_px"] == 1


def test_scoring_set_is_the_same_at_every_N(tmp_path, monkeypatch):
    c = synthetic_cube(T=60, H=24, W=30, seed=8)
    keep, units_keep, truth = keep_and_units(c, seed=9, n_units=140)
    jobs = v14.make_jobs({"all": [12, 30, 60]}, {"all": 60}, 3, 2)
    subs = v15.build_subsets(5, "synthetic", 60, jobs, c["h"])
    ctx = make_ctx(c, subs, keep, units_keep, truth)
    seen = []
    real = v14.score

    def spy(kind, t, p):
        seen.append((kind, np.array(t, copy=True), len(p)))
        return real(kind, t, p)

    monkeypatch.setattr(v14, "score", spy)
    log = v15.RowLog(str(tmp_path / "s.csv"))
    v15.run_jobs(ctx, list(range(len(jobs))), log, 1, time.time() + 600, 1.0)
    df = pd.read_csv(tmp_path / "s.csv")
    assert len(seen) == int((df["status"] == "ok").sum()) > 0
    for kind, t, n in seen:
        assert kind == "vaklodingen" and n == len(truth) and np.array_equal(t, truth)
    assert (df["n_set"] == len(truth)).all()
    ok = df[df["status"] == "ok"]
    assert np.allclose(ok["coverage"] * ok["n_set"], ok["n"])        # truth is finite on every unit
    assert ok.loc[ok["N"] == 60, "coverage"].iloc[0] >= ok["coverage"].median()


def test_parallel_phase2_equals_serial_and_deadline(tmp_path):
    """The pool path (a concurrent.futures thread pool here; the fork
    ProcessPoolExecutor on the server, same submit/wait code) gives the
    serial rows; a passed deadline starts no NIDEM but still writes the rows
    that need none."""
    from concurrent.futures import ThreadPoolExecutor as ThreadPool

    c = synthetic_cube(T=60, H=24, W=30, seed=8)
    keep, units_keep, truth = keep_and_units(c, seed=9, n_units=140)
    jobs = v14.make_jobs({"all": [12, 30, 60]}, {"all": 60}, 3, 2)
    subs = v15.build_subsets(5, "synthetic", 60, jobs, c["h"])
    ctx = make_ctx(c, subs, keep, units_keep, truth)
    todo = list(range(len(jobs)))
    a, b = v15.RowLog(str(tmp_path / "a.csv")), v15.RowLog(str(tmp_path / "b.csv"))
    v15.run_jobs(ctx, todo, a, 1, time.time() + 600, 1.0)
    n, stop = v15.run_jobs(ctx, todo, b, 3, time.time() + 600, 1.0, pool_factory=ThreadPool)
    assert stop is None and n == len(jobs)
    cols = [k for k in v15.CSV_COLUMNS if k != "seconds"]

    def table(p):
        return pd.read_csv(p).sort_values(["N", "replicate"])[cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(table(tmp_path / "a.csv"), table(tmp_path / "b.csv"))
    n_heavy = int((table(tmp_path / "a.csv")["status"] == "ok").sum())
    assert n_heavy > 0
    for workers, factory in ((1, None), (3, ThreadPool)):
        p = tmp_path / f"late{workers}.csv"
        n, stop = v15.run_jobs(ctx, todo, v15.RowLog(str(p)), workers, time.time() - 1, 1.0, pool_factory=factory)
        late = pd.read_csv(p)
        assert stop is not None and stop["jobs_left"] == n_heavy
        assert len(late) == len(jobs) - n_heavy and (late["status"] != "ok").all()


def test_pool_error_stops_loudly(tmp_path, monkeypatch):
    """A job that fails in the pool (as a dead worker's BrokenProcessPool
    would) raises in the parent instead of hanging; the rows already written
    stay, so a rerun resumes."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    c = synthetic_cube(T=60, H=24, W=30, seed=8)
    keep, units_keep, truth = keep_and_units(c, seed=9, n_units=140)
    jobs = v14.make_jobs({"all": [12, 30, 60]}, {"all": 60}, 3, 2)
    subs = v15.build_subsets(5, "synthetic", 60, jobs, c["h"])
    ctx = make_ctx(c, subs, keep, units_keep, truth)
    real = v15._pool_job
    calls = []

    def dying(k):
        calls.append(k)
        if len(calls) == 1:
            raise BrokenProcessPool("a worker died")
        return real(k)

    monkeypatch.setattr(v15, "_pool_job", dying)
    log = v15.RowLog(str(tmp_path / "p.csv"))
    with pytest.raises(BrokenProcessPool):
        v15.run_jobs(ctx, list(range(len(jobs))), log, 2, time.time() + 600, 1.0, pool_factory=ThreadPoolExecutor)
    kept = pd.read_csv(tmp_path / "p.csv")
    assert len(kept) < len(jobs) and not kept.duplicated(["N", "replicate"]).any()
    monkeypatch.setattr(v15, "_pool_job", real)
    log2 = v15.RowLog(str(tmp_path / "p.csv"))                      # resume: only the missing jobs run
    todo = [i for i, s in enumerate(subs) if not log2.is_done(s["N"], s["rep"])]
    v15.run_jobs(ctx, todo, log2, 2, time.time() + 600, 1.0, pool_factory=ThreadPoolExecutor)
    done = pd.read_csv(tmp_path / "p.csv")
    assert len(done) == len(jobs) and not done.duplicated(["N", "replicate"]).any()


def test_run_lock_refuses_a_second_run(tmp_path):
    pytest.importorskip("fcntl")
    p = str(tmp_path / "x.run.lock")
    fd = v15.acquire_run_lock(p)
    assert fd is not None
    with pytest.raises(SystemExit):
        v15.acquire_run_lock(p)
    os.close(fd)
    os.close(v15.acquire_run_lock(p))                                 # free again once the first run is gone


def test_rows_resume_and_partial_line(tmp_path):
    p = str(tmp_path / "r.csv")
    log = v15.RowLog(p)
    base = dict(site="s", pool="all", pool_size=60, status="ok", reason="", rmse=0.1, bias=0.0, slope=1.0, r=0.9,
                coverage=0.5, n=10, n_set=20, median_valid_obs=5.0, keep_valid_share=0.4, sizes="1;2",
                n_vertices=10, seconds=0.1, draw_sha1="x")
    assert log.append(dict(base, N=10, replicate=0)) and log.append(dict(base, N=10, replicate=1))
    assert not log.append(dict(base, N=10, replicate=1))             # never twice
    text = open(p, encoding="utf-8").read()
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text[:-7])                                           # killed mid-write
    log2 = v15.RowLog(p)
    assert log2.is_done(10, 0) and not log2.is_done(10, 1)
    assert len(pd.read_csv(p)) == 1


# ── the figure ───────────────────────────────────────────────────────────────

def _plot_inputs_on_disk(tmp_path, name="a", status="complete", drop=(), bad_draw=None, marea_sha=None):
    """A MAREA refit CSV and a NIDEM CSV of 'escalda' with the same draws;
    ``drop``: (N, rep) missing from NIDEM; ``bad_draw``: (N, rep) whose NIDEM
    draw_sha1 differs; ``marea_sha``: the MAREA sha256 the NIDEM settings
    claim (default: the true one)."""
    md, nd = tmp_path / f"m{name}", tmp_path / f"n{name}"
    md.mkdir()
    nd.mkdir()
    rng = np.random.default_rng(0)
    rows, nrows = [], []
    for N, k in ((10, 5), (30, 5), (100, 5), (300, 3), (464, 1)):
        for rep in range(k):
            sha = f"d{N}_{rep}"
            rows.append(dict(site="escalda", lags="refit", pool="all", pool_size=464, N=N, replicate=rep, mb=0,
                             rmse=0.4 + 0.3 / np.sqrt(N) + 0.01 * rng.normal(), coverage=min(1.0, N / 40),
                             median_clear_obs=N / 3, draw_sha1=sha))
            if (N, rep) in drop:
                continue
            nc = N <= 30 and rep < 4
            nrows.append(dict(site="escalda", pool="all", pool_size=464, N=N, replicate=rep,
                              status="not_computable_preflight" if nc else "ok",
                              rmse=np.nan if nc else 0.37 + 0.2 / np.sqrt(N), coverage=0.0 if nc else 0.97,
                              median_valid_obs=N / 2, draw_sha1="other" if (N, rep) == bad_draw else sha))
    pd.DataFrame(rows).to_csv(md / "escalda_refit.csv", index=False)
    with open(md / "escalda_refit_settings.json", "w", encoding="utf-8") as f:
        json.dump({"fixed_set": {"n_units_scored": 100000}, "plan": {"reps": 5, "reps_large": 3}}, f)
    pd.DataFrame(nrows).to_csv(nd / "escalda_nidem_all.csv", index=False)
    with open(nd / "escalda_nidem_all_settings.json", "w", encoding="utf-8") as f:
        json.dump({"status": status, "compat": {"marea_csv_sha256": marea_sha or
                                                v15.sha256_file(str(md / "escalda_refit.csv"))}}, f)
    return md, nd


def test_plot_both_versions(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md, nd = _plot_inputs_on_disk(tmp_path)
    D = v15.plot_inputs("escalda", [str(md)], [str(nd)])
    assert D["nc"][10] == (4, 5) and D["nc"][464] == (0, 1)
    assert np.isnan(D["nidem"].set_index("N").loc[10, "rmse"])       # under half the draws scored: no point
    assert D["nidem"].set_index("N").loc[10, "cov"] == 0.0
    assert D["marea"]["k"].sum() == D["nidem"]["k"].sum() == 19      # the same draws for both
    # a MAREA median whose median coverage is under half of the set is neither drawn nor joined
    fig, (ax, axc) = plt.subplots(2)
    v15._draw_site(ax, axc, D, "es", v15.LAYOUT["es"]["fs"])
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()         # MAREA's median line
    assert D["marea"].set_index("N").loc[10, "cov"] < v14.LOW_COVERAGE
    assert np.isnan(y[list(x).index(10)]) and np.isfinite(y[list(x).index(30)])
    assert len(ax.lines[1].get_xdata()) == int(np.isfinite(y).sum())  # markers only where the line is
    plt.close(fig)
    args = types.SimpleNamespace(marea_dir=str(md), nidem_dir=str(nd), lang="es,en", dpi=60,
                                 out_es=str(tmp_path / "f_es"), out_en=str(tmp_path / "f_en"))
    v15.plot(args)
    for stem in ("f_es", "f_en"):
        for ext in (".png", ".pdf"):
            assert (tmp_path / (stem + ext)).stat().st_size > 1000
    assert "logarítmica" in v15.TXT["es"]["x"]
    for lang in ("es", "en"):
        assert set(v15.TXT[lang]["status"]) >= {"running", "crashed", "stopped_by_time", "incomplete",
                                                "gate_failed", "gates_only", None}
    v14.guard_out(os.path.dirname(v15.FIG_ES))                       # the default outputs are allowed paths
    v14.guard_out(os.path.dirname(v15.FIG_EN))


def test_plot_refuses_unlike_inputs_and_matches_partial_runs(tmp_path):
    md, nd = _plot_inputs_on_disk(tmp_path, "b", bad_draw=(100, 2))
    with pytest.raises(SystemExit):                                   # a NIDEM draw that is not MAREA's
        v15.plot_inputs("escalda", [str(md)], [str(nd)])
    md, nd = _plot_inputs_on_disk(tmp_path, "c", marea_sha="0" * 64)
    with pytest.raises(SystemExit):                                   # not the MAREA CSV NIDEM was gated against
        v15.plot_inputs("escalda", [str(md)], [str(nd)])
    drop = {(100, 3), (100, 4), (300, 2)}
    md, nd = _plot_inputs_on_disk(tmp_path, "d", status="stopped_by_time", drop=drop)
    D = v15.plot_inputs("escalda", [str(md)], [str(nd)])              # a partial run: MAREA on NIDEM's draws
    km, kn = D["marea"].set_index("N")["k"], D["nidem"].set_index("N")["k"]
    assert km.to_dict() == kn.to_dict() and km.loc[100] == 3 and km.loc[300] == 2
    assert D["n_draws_nidem"] == 16 and D["n_draws_marea"] == 19 and D["nsettings"]["status"] == "stopped_by_time"


# ── the RTK guard ────────────────────────────────────────────────────────────

def test_rtk_guard_passes():
    from tests import test_r1_guard
    static = [getattr(test_r1_guard, n) for n in dir(test_r1_guard) if n.startswith("test_static")]
    assert static
    for f in static:
        f()                                                          # scans experiments/, this script too
    src = open(v15.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "attr", None) == "load_rtk" or getattr(n.func, "id", None) == "load_rtk")]
    assert all(not c.args and not c.keywords for c in calls)
    assert "v14.rtk_units(" in src                                   # RTK truth only through v14
    kws = {kw.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for kw in n.keywords if kw.arg}
    assert not any(k.endswith("reserved") for k in kws)
