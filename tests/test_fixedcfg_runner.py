# -*- coding: utf-8 -*-
"""experiments/v19_fixed_config_tables.py on tiny synthetic data: the windows
(nested suffixes, reachability), the screening helpers and gates, the extra
store contract, the MAREA and no-delay jobs (= v14's code), the common
inputs given to a competitor, the registry statuses, the NIDEM hook, the
resumable job store, the common / own scoring, the record gate, the tables
(CSV, markdown, Spanish LaTeX), the curves in both languages, and the
guards.

Run:  python -m pytest -q tests/test_fixedcfg_runner.py
"""
import argparse
import ast
import json
import math
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402
from experiments import v19_fixed_config_tables as v19  # noqa: E402
from pyintertidal import marea  # noqa: E402

M2_H = 12.4206012
SEED = 20261019


class FakeBoundary:
    """Like PyTMDBoundary: levels returned sorted by time; optionally NaN at
    one instant."""

    name = "fake boundary"

    def __init__(self, amp=1.4, nan_at=None):
        self.amp, self.nan_at, self.calls = amp, nan_at, 0

    def levels(self, times):
        t = pd.DatetimeIndex(times).sort_values()
        self.calls += 1
        hrs = t.values.astype("datetime64[s]").astype(np.int64) / 3600.0
        out = self.amp * np.sin(2 * np.pi * hrs / M2_H) + 0.2 * np.sin(2 * np.pi * hrs / 12.0)
        if self.nan_at is not None:
            out[np.asarray(t == pd.Timestamp(self.nan_at))] = np.nan
        return out


def make_dates(years, per_year, rng):
    dates, times = [], []
    for y in years:
        for d in np.sort(rng.choice(np.arange(0, 360), per_year, replace=False)):
            ts = (pd.Timestamp(f"{y}-01-01 11:00") + pd.Timedelta(days=int(d))
                  + pd.Timedelta(seconds=int(rng.integers(0, 1800))))
            times.append(ts)
            dates.append(ts.strftime("%Y-%m-%d"))
    return np.array(dates), pd.DatetimeIndex(times)


def synthetic_table(T_years=(2023, 2024, 2025), per_year=17, Pk=40, n_units=24, seed=0, lag1=30):
    """A window table like v19.build_table's: v14's SiteData on the screened
    scenes with the raw keep arrays (Yk, Ck); pixels flood when the level,
    read at their lag, tops them; two lag-fit bands, the second lagging
    ``lag1`` minutes. Returns (table, S, ctx-like fields)."""
    rng = np.random.default_rng(seed)
    dates, times = make_dates(T_years, per_year, rng)
    T = len(dates)
    lev = v14.LevelTable(FakeBoundary(), times)
    lev.ensure(sorted({0, *(int(t) for t in marea.BANK_TAUS), *range(0, lag1 + 1)}))
    s_km = np.linspace(0.3, 6.0, Pk)
    band = (s_km > 3.0).astype(int)
    tau_px_true = np.where(band == 1, float(lag1), 0.0)
    z = rng.uniform(-1.0, 1.0, Pk)
    H = np.stack([lev[int(t)] for t in tau_px_true], axis=1)
    Ck = rng.random((T, Pk)) < 0.85
    Yk = (np.where(H > z[None, :], 0.45, -0.3) + rng.normal(0, 0.05, (T, Pk))).astype(np.float32)
    Yk[~Ck & (rng.random((T, Pk)) < 0.5)] = np.nan            # some no-data observations
    px = np.arange(n_units, dtype=np.int64)                    # inverted pixels = the first keep columns
    Y0 = np.nan_to_num(Yk[:, px], nan=0.0).astype(np.float32)
    C0 = Ck[:, px]
    fit_kw = dict(sigma0=0.2, sg_grid=(0.06, 0.1, 0.15, 0.22, 0.32, 0.45), tau_bounds=marea.TAU_BOUNDS,
                  z_points=60, max_px_band=1200, monotone=True)
    m2a = dict(Y=np.nan_to_num(Yk, nan=0.0).astype(np.float32), C=Ck.copy(), band=band, centers=np.array([1.5, 4.5]),
               nb=2, band_km=3.0, fit_kw=fit_kw, seed=20260819)
    truth = z[px] + rng.normal(0, 0.03, n_units)
    table = v14.SiteData("synthetic", Y0, C0, lev, {"S": np.arange(T)}, np.arange(n_units), truth, "vaklodingen",
                         np.zeros(n_units), s_km=s_km[px], m2a=m2a)
    Yk.setflags(write=False)
    Ck.setflags(write=False)
    table.Yk, table.Ck = Yk, Ck
    table.S_dates, table.S_t = dates, times
    S = {"dates": dates, "src": np.zeros(T, np.int64), "row": np.arange(T, dtype=np.int64), "t": times,
         "h0": np.asarray(lev[0], float)}
    keep_flat = np.arange(Pk, dtype=np.int64) * 2 + 1          # flat indices on an 8 x 10 grid
    return table, S, keep_flat, px


def make_ctx(tmp_path, table, S, keep_flat, px, registry=None, nidem_fn=None, nidem_info=None, level_series=None):
    return v19.Context(site="synthetic", seed=SEED, table=table, S=S, paths=v19.site_paths(str(tmp_path), "synthetic"),
                       registry=registry or {}, nidem_fn=nidem_fn, nidem_info=nidem_info or {"reason": "absent"},
                       units_keep=px.copy(), unit_col=np.asarray(table.unit_col, np.int64), px=px,
                       keep_flat=keep_flat, shape=(8, 10), pixel_m=10.0, units_sha1="units0",
                       boundary_name="fake boundary", level_series=level_series)


def windows_for(S, ns=(40, 30), n_record=51):
    return v19.make_windows(S["dates"], S["h0"], ns, n_record)


# ── windows, screening, gates ───────────────────────────────────────────────

def test_windows_are_nested_suffixes_with_reachability():
    rng = np.random.default_rng(1)
    dates, _ = make_dates(range(2019, 2026), 10, rng)            # 70 screened scenes
    h0 = np.linspace(-1, 1, len(dates))
    all_dates = np.concatenate([dates, ["2025-12-30", "2025-12-31", "2018-05-05"]])
    ws = v19.make_windows(dates, h0, v19.NS, 68, all_dates)
    by = {w["n"]: w for w in ws}
    assert [w["n"] for w in ws] == [125, 100, 75, 68, 60, 50, 40]
    for n in (125, 100, 75):
        assert not by[n]["reachable"] and by[n]["missing"] == n - 70
        assert "missing" in by[n]["reason"] and "years before 2019" in by[n]["reason"]
    w60 = by[60]
    assert w60["reachable"] and w60["first_date"] == dates[-60] and w60["lower_bound_open"] == dates[-61]
    assert sum(w60["years"].values()) == 60 and w60["last_date"] == dates[-1]
    assert w60["end_date"] == "2025-12-31"
    # cube scenes counted to the window end (2025-12-31), not to the last screened scene
    assert w60["cube_scenes"] == int(((np.unique(all_dates) >= dates[-60]) & (np.unique(all_dates) <= "2025-12-31")).sum())
    assert by[68]["is_record"] and not by[68]["requested"] and by[60]["requested"]
    assert w60["lo_m"] == pytest.approx(h0[-60:].min()) and w60["hi_m"] == pytest.approx(h0[-60:].max())
    assert v19.check_nested(ws)["ok"]
    # the same scenes give the same key; nested windows differ
    assert by[60]["scenes_sha1"] == v19.scenes_sha1(dates[-60:]) != by[50]["scenes_sha1"]
    assert v19.parse_ns("40, 125;60") == [125, 60, 40]
    with pytest.raises(SystemExit):
        v19.parse_ns("1")


def test_screen_candidates_keeps_usable_with_time_and_level():
    rng = np.random.default_rng(2)
    dates, times = make_dates([2024, 2025], 6, rng)
    over = {d: t.to_pydatetime() for d, t in zip(dates, times)}
    del over[dates[3]]                                          # no overpass time
    usable = set(dates[::2]) | {dates[3], "2020-01-01"}         # one usable date not in the extraction
    b = FakeBoundary(nan_at=times[4])                           # dates[4] usable, but no level
    c = v19.screen_candidates(dates, usable, over, b.levels)
    want = [d for i, d in enumerate(dates) if d in usable and i not in (3, 4)]
    assert c["dates"].tolist() == want and np.all(np.isfinite(c["h0"]))
    assert c["lost"]["usable_not_in_extraction"] == ["2020-01-01"]
    assert c["lost"]["usable_without_overpass_time"] == [dates[3]]
    assert c["lost"]["usable_without_finite_level"] == [dates[4]]
    assert np.array_equal(c["idx"], np.array([i for i, d in enumerate(dates) if d in want]))
    with pytest.raises(ValueError):
        v19.screen_candidates(dates[::-1], usable, over, b.levels)


def test_record_part_gates():
    rng = np.random.default_rng(3)
    dates, times = make_dates(range(2020, 2026), 5, rng)
    h0 = np.sin(np.arange(len(dates)))
    S = {"dates": dates, "h0": h0}
    rec = [d for d in dates if d >= "2023-01-01"]
    m = np.asarray(dates) >= "2023-01-01"
    g = v19.record_part_gates(S, rec, [h0[m].min(), h0[m].max()])
    assert all(v["ok"] for v in g.values())
    g = v19.record_part_gates(S, rec[1:], [h0[m].min(), h0[m].max()])
    assert not g["screened_record_part_equals_record_dates"]["ok"]
    g = v19.record_part_gates(S, rec, [h0[m].min() - 0.01, h0[m].max()])
    assert not g["screened_record_level_range_vs_rango_marea_m"]["ok"]


def test_extra_store_contract(tmp_path):
    rng = np.random.default_rng(4)
    keep = np.arange(6, dtype=np.int64)
    main_dates, main_t = make_dates([2023, 2024], 4, rng)
    S = {"dates": main_dates, "src": np.zeros(8, np.int64), "row": np.arange(8, dtype=np.int64), "t": main_t,
         "h0": FakeBoundary().levels(main_t)}
    xd, xt = make_dates([2021, 2022], 3, rng)
    xd = np.concatenate([xd, [main_dates[0]]])                   # one date the main extraction already has
    xt = xt.append(pd.DatetimeIndex([main_t[0]]))
    Y = rng.normal(0, 0.3, (len(xd), 6)).astype(np.float32)
    C = rng.random((len(xd), 6)) < 0.8
    d = str(tmp_path)

    def store(name, keep_=keep, side=None):
        p = os.path.join(d, f"synthetic_extract_{name}.npz")
        np.savez(p, Y=Y, C=C, keep=keep_, dates=xd, shape=np.array([2, 3]))
        if side is not None:
            with open(p[:-4] + ".json", "w", encoding="utf-8") as f:
                json.dump(side, f)
        return p

    over = {a: str(b) for a, b in zip(xd, xt)}
    cloud = {a: (5.0 if i % 2 == 0 else 50.0) for i, a in enumerate(xd)}
    st, info = v19.load_extra_store(store("a", keep_=keep[::-1].copy()), keep, (2, 3), main_dates)
    assert st is None and "keep differs" in info["reason"]
    st, info = v19.load_extra_store(store("b", side={"overpass": over}), keep, (2, 3), main_dates)
    assert st is None and "screening" in info["reason"]
    for p in os.listdir(d):
        os.remove(os.path.join(d, p))
    store("2021-2022", side={"overpass": over, "cloud_pct": cloud, "gate_2023": {"ok": True}})
    b = FakeBoundary()
    # a boundary with gauges (any provider that is not PyTMDBoundary): refused unless the user chose
    S2, infos, note = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "refuse")
    assert S2 is S and not infos[0]["used"] and "pre-2023 levels" in infos[0]["reason"]
    S2, infos, note = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "production")
    assert infos[0]["used"] and infos[0]["dates_ignored_main_wins"] == 1
    early_usable = [a for i, a in enumerate(xd[:-1]) if i % 2 == 0]
    assert S2["dates"].tolist() == sorted(early_usable) + main_dates.tolist()
    assert S2["t"].is_monotonic_increasing and set(S2["src"][:len(early_usable)]) == {1}
    # no passed re-download gate: refused (soft gates override)
    os.remove(os.path.join(d, "synthetic_extract_2021-2022.json"))
    store("2021-2022", side={"overpass": over, "cloud_pct": cloud})
    S3, infos, _ = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "production")
    assert S3 is S and "gate" in infos[0]["reason"]
    S3, infos, _ = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "production", soft=True)
    assert infos[0]["used"]


def test_screened_list_dutch_branch_is_the_record():
    rng = np.random.default_rng(7)
    dates, times = make_dates([2023, 2024, 2025], 6, rng)
    rec = {"time_index": np.arange(len(dates)) + 3, "dates": dates, "t_real": times,
           "h": np.linspace(-1, 1, len(dates)), "boundary": "fake"}
    rec_dates = list(dates[::3])
    S, info = v19.screened_list("escalda", {}, dates, {}, rec, rec_dates)
    k = np.arange(0, len(dates), 3)
    assert S["dates"].tolist() == rec_dates and np.array_equal(S["row"], k + 3)
    assert np.array_equal(S["h0"], rec["h"][k]) and S["t"].equals(times[k]) and set(S["src"]) == {0}
    assert info["n_main"] == len(rec_dates) and "record_dates" in info["usable_source"]
    with pytest.raises(RuntimeError):
        v19.screened_list("escalda", {}, dates, {}, rec, rec_dates + ["2026-01-05"])


def test_build_table_copies_or_streams_and_equals_v14(tmp_path):
    rng = np.random.default_rng(8)
    dates, times = make_dates([2021, 2022, 2023, 2024, 2025], 6, rng)      # 30 extraction scenes
    Ta, Pk = len(dates), 12
    Y = rng.normal(0, 0.4, (Ta, Pk)).astype(np.float32)
    Y[rng.random((Ta, Pk)) < 0.1] = np.nan
    C = rng.random((Ta, Pk)) < 0.8
    npz = str(tmp_path / "extract.npz")
    np.savez(npz, Y=Y, C=C)
    px, cols = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7, 9, 11])
    pool = np.arange(12, Ta)                                                 # v14's pool: the newest 18 scenes
    b = FakeBoundary()
    lev14 = v14.LevelTable(b, times[pool])
    lev14.ensure([0, 5, 10])
    fit_kw = dict(sigma0=0.2, sg_grid=(0.1, 0.2), tau_bounds=marea.TAU_BOUNDS, z_points=20, max_px_band=1200,
                  monotone=True)
    m2a = dict(Y=np.nan_to_num(Y[pool][:, cols], nan=0.0), C=C[pool][:, cols], band=np.array([0, 0, 0, 1, 1, 1]),
               centers=np.array([1.5, 4.5]), nb=2, band_km=3.0, fit_kw=fit_kw, seed=1)
    sd14 = v14.SiteData("synthetic", np.nan_to_num(Y[pool][:, px], nan=0.0).astype(np.float32), C[pool][:, px],
                        lev14, {"all": np.arange(len(pool))}, np.arange(4), np.zeros(4), "vaklodingen",
                        np.array([0.0, 0.0, 5.0, 10.0]), s_km=np.linspace(0, 6, 4), m2a=m2a)
    pos14 = {d: i for i, d in enumerate(dates[pool])}
    rows = np.array([2, 5, 9, 12, 15, 20, 25, 29])                          # 3 older scenes, 5 inside the pool
    S = {"dates": dates[rows], "src": np.zeros(len(rows), np.int64), "row": rows, "t": times[rows],
         "h0": b.levels(times[rows])}
    for need_keep in (False, True):
        table, info = v19.build_table("synthetic", {0: npz}, S, px, cols, Pk, b, sd14, need_keep, pos14)
        assert info["rows_copied_from_v14"] == (0 if need_keep else 5)
        assert np.array_equal(table.Y0, np.nan_to_num(Y[rows][:, px], nan=0.0))
        assert np.array_equal(table.C, C[rows][:, px])
        assert np.array_equal(table.m2a["Y"], np.nan_to_num(Y[rows][:, cols], nan=0.0))
        assert np.array_equal(table.m2a["C"], C[rows][:, cols])
        g = v19.compare_with_v14(table, sd14, S, pos14)
        assert g["ok"] and g["shared_scenes"] == 5 and g["levels_max_abs_diff_m"] <= v14.TOL_LEVEL
        assert np.allclose(table.levels[0], S["h0"], atol=1e-12)
        if need_keep:
            assert np.array_equal(table.Yk, Y[rows], equal_nan=True) and np.array_equal(table.Ck, C[rows])
            assert not table.Yk.flags.writeable
        else:
            assert table.Yk is None
    # a different scene at a shared date is caught
    bad = dict(m2a, Y=m2a["Y"].copy())
    bad["Y"][0, 0] += 1.0
    sd_bad = v14.SiteData("synthetic", sd14.Y0, sd14.C, lev14, {"all": np.arange(len(pool))}, np.arange(4),
                          np.zeros(4), "vaklodingen", sd14.tau_fixed, s_km=sd14.s_km, m2a=bad)
    assert not v19.compare_with_v14(table, sd_bad, S, pos14)["ok"]
    assert v19.level_provenance(b, S["t"]) is None                        # a single provider


# ── the jobs: v14's code on the window ──────────────────────────────────────

def test_marea_job_is_v14s_refit_job_and_nodelay_is_invert_subset():
    table, S, keep_flat, px = synthetic_table()
    T = len(S["dates"])
    rows = np.arange(T - 40, T)
    z, meta = v19.marea_job(table, rows, SEED)
    table.pools["ref"] = rows
    ref_rows, info = v14.run_job(table, "ref", len(rows), 0, SEED, "refit", [0, 3])
    table.pools.pop("ref")
    for mb in v19.MBS:
        assert np.array_equal(z[f"mb{mb}"], np.asarray(info["z"][mb], float)[table.unit_col], equal_nan=True)
    assert meta["tau_bands_min"] == list(np.asarray(info["tau_bands"], float))
    assert all(t >= 0 for t in meta["tau_used_bands_min"]) and meta["tau_used_bands_min"][0] == 0.0
    assert "S" in table.pools and len(table.pools) == 1           # the temporary pool is gone
    zn, mn = v19.nodelay_job(table, rows)
    zz, _, (lo, hi), _, _ = v14.invert_subset(table.Y0, table.C, table.levels, rows,
                                              [(0, np.arange(table.Y0.shape[1]))], [0, 3])
    for mb in v19.MBS:
        assert np.array_equal(zn[f"mb{mb}"], np.asarray(zz[mb], float)[table.unit_col], equal_nan=True)
    assert (mn["lo_m"], mn["hi_m"]) == (lo, hi)
    assert lo == pytest.approx(float(np.min(table.levels[0][rows])))


def test_competitor_gets_the_common_inputs_and_statuses(tmp_path):
    table, S, keep_flat, px = synthetic_table()
    T = len(S["dates"])
    seen = {}

    def good(obs, cfg):
        seen["obs"], seen["cfg"] = obs, cfg
        f = (obs.W & np.isfinite(obs.Y)).sum(0) / np.maximum((obs.C & np.isfinite(obs.Y)).sum(0), 1)
        z = np.quantile(obs.h, 1 - np.clip(f, 0, 1)).astype(np.float32)
        z[0] = np.nan
        return {"z": z, "sigma": None, "meta": {"status": "ok", "reason": ""}}

    def nc(obs, cfg):
        return {"z": np.full(len(obs.keep), np.nan, np.float32), "meta": {"status": "not computable",
                                                                          "reason": "too few images"}}

    class NotComputable(Exception):
        pass

    def raises_nc(obs, cfg):
        raise NotComputable("needs cfg['level_series']")

    def broken(obs, cfg):
        raise ValueError("boom")

    def per_pixel(obs, cfg):
        seen["pp"] = obs
        return {"z": np.asarray(obs.h.mean() + 0 * obs.Y[0], np.float32), "meta": {"status": "ok"}}

    card = {"reference": "r", "doi": "d", "fidelity": "F3 (test)", "deviations": [], "needs_full_frame": False}
    reg = {"EXA": {"label": "good", "run": good, "card": dict(card, needs=["level_series"]), "variant": False},
           "EXB": {"label": "nc", "run": nc, "card": card, "variant": False},
           "EXC": {"label": "raises", "run": raises_nc, "card": card, "variant": False},
           "EXD": {"label": "broken", "run": broken, "card": card, "variant": False},
           "EXE": {"label": "not built", "run": None, "card": card, "variant": False},
           "EXF": {"label": "per pixel", "run": per_pixel, "card": dict(card, per_pixel=True), "variant": True}}

    class Series:
        def __init__(self):
            self.cuts = []

        def slice(self, a, b):
            self.cuts.append((a, b))
            return self

        def summary(self):
            return {"n": 0}

    ls = Series()
    ctx = make_ctx(tmp_path, table, S, keep_flat, px, registry=reg, level_series=ls)
    w = [x for x in windows_for(S) if x["n"] == 40][0]
    m, arr = v19.run_one(ctx, w, "EXA")
    obs = seen["obs"]
    sl = slice(T - 40, T)
    assert m["status"] == v19.OK and m["fidelity"] == "F3" and m["cacheable"]
    assert obs.Y.shape == (40, len(keep_flat)) and obs.Y.dtype == np.float32 and np.isnan(obs.Y).any()
    assert np.shares_memory(obs.Y, table.Yk) and not obs.Y.flags.writeable and not obs.W.flags.writeable
    assert np.array_equal(obs.W, obs.C & (obs.Y > 0)) and obs.C.dtype == bool
    assert np.array_equal(obs.h, table.levels[0][sl]) and obs.h.dtype == np.float64
    assert len(obs.t) == 40 and obs.t.equals(S["t"][sl]) and obs.dates == list(S["dates"][sl])
    assert np.array_equal(obs.keep, keep_flat) and obs.shape == (8, 10) and obs.pixel_m == 10.0
    img = obs.frame(obs.Y[0])
    assert img.shape == (8, 10) and np.array_equal(obs.unframe(img), obs.Y[0], equal_nan=True)
    assert np.isnan(img.ravel()[0]) and obs.frame(obs.C[:2]).dtype == bool
    assert seen["cfg"]["level_series"] is ls and ls.cuts[0][1] == v19.LEVEL_SERIES_END
    assert ls.cuts[0][0] == pd.Timestamp(S["t"][T - 40]).normalize()
    assert seen["cfg"]["mask"] == v19.COMMON_MASK and seen["cfg"]["wet_threshold"] == 0.0
    assert arr["z__main"].shape == (len(px),) and np.isnan(arr["z__main"][0])     # keep values -> units
    assert m["variants"]["main"]["role"] == "headline"
    m, arr = v19.run_one(ctx, w, "EXB")
    assert m["status"] == v19.NOT_COMPUTABLE and arr is None and "too few" in m["reason"] and m["cacheable"]
    m, _ = v19.run_one(ctx, w, "EXC")
    assert m["status"] == v19.NOT_COMPUTABLE and "level_series" in m["reason"]
    m, _ = v19.run_one(ctx, w, "EXD")
    assert m["status"] == v19.ERROR and not m["cacheable"] and "boom" in m["reason"]
    m, _ = v19.run_one(ctx, w, "EXE")
    assert m["status"] == v19.PENDING and not m["cacheable"] and "not built" in m["reason"]
    m, arr = v19.run_one(ctx, w, "EXF")
    assert seen["pp"].Y.shape == (40, len(px)) and m["variants"]["main"]["role"] == "variant"
    assert m["info"]["pixel_set"] == "scored" and arr["z__main"].shape == (len(px),)
    # NIDEM without a hook or a registry entry: pending, never cached
    m, _ = v19.run_one(ctx, w, "E1")
    assert m["status"] == v19.PENDING and not m["cacheable"] and m["label"] == "NIDEM"
    # parse_method_result rules
    with pytest.raises(ValueError):
        v19.parse_method_result({"z": np.zeros(3)}, 4)
    assert v19.parse_method_result(np.zeros(4), 4)[0] == v19.OK
    assert v19.parse_method_result({"z": np.full(4, np.nan)}, 4)[0] == v19.NOT_COMPUTABLE
    assert v19.fidelity_tier("F2, port") == "F2" and v19.pixel_set_of({"pixel_set": "scored"}) == "scored"


def test_nidem_hook_absent_present_and_mapped(tmp_path, monkeypatch):
    fn, info = v19.find_nidem_hook(module="experiments.no_such_v15_module")
    assert fn is None and info["status"] == "absent"
    mod_dir = tmp_path / "hookpkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("", encoding="utf-8")
    (mod_dir / "fake_v15.py").write_text(
        "import numpy as np\n"
        "CALLS = []\n"
        "def nidem_on_scenes(site, dates, h, units_keep, mask, obs):\n"
        "    CALLS.append(dict(site=site, n=len(dates), mask=mask))\n"
        "    return {'z_keep': np.asarray(obs.h.mean() + 0.0 * obs.Y[0], float)}\n"
        "def nidem_subset(site, dates):\n"
        "    return {'status': 'NOT COMPUTABLE', 'reason': 'empty interval'}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(v19, "module_path", lambda m: str(tmp_path / (m.replace(".", os.sep) + ".py")))
    fn, info = v19.find_nidem_hook(module="hookpkg.fake_v15")
    assert info["status"] == "found" and info["function"].endswith("nidem_on_scenes")
    table, S, keep_flat, px = synthetic_table()
    ctx = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=fn, nidem_info=info)
    w = [x for x in windows_for(S) if x["n"] == 40][0]
    m, arr = v19.run_one(ctx, w, "E1")
    assert m["status"] == v19.OK and arr["z__main"].shape == (len(px),)
    assert "common mask was passed" in m["info"]["mask"]
    import hookpkg.fake_v15 as hk
    assert hk.CALLS[-1] == {"site": "synthetic", "n": 40, "mask": v19.COMMON_MASK}
    fn2, info2 = v19.find_nidem_hook(name="nidem_subset", module="hookpkg.fake_v15")
    ctx.nidem_fn, ctx.nidem_info = fn2, info2
    m, arr = v19.run_one(ctx, w, "E1")
    assert m["status"] == v19.NOT_COMPUTABLE and "empty interval" in m["reason"] and arr is None
    # the per-unit / per-pixel / per-keep readings of a hook's answer
    uk, uc = np.array([3, 5, 7]), np.array([0, 1, 1])
    assert np.array_equal(v19.hook_values(np.arange(10.0), 3, uc, uk, 2, 10), [3.0, 5.0, 7.0])
    assert np.array_equal(v19.hook_values({"z_px": np.array([1.0, 2.0])}, 3, uc, uk, 2, 10), [1.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        v19.hook_values(np.zeros(4), 3, uc, uk, 2, 10)


# ── the job store, scoring and the record gate ──────────────────────────────

def _job(store, w, mid, status, variants=None, arrays=None, fp="fp", cache=True, label=None):
    meta = {"key": v19.job_key(w["n"], mid), "site": "synthetic", "n": w["n"], "window": w["label"],
            "is_record": w["is_record"], "method": mid, "label": label or mid, "status": status, "reason": "r",
            "source": "s", "fidelity": "F3", "variants": variants or {}, "scenes_sha1": w["scenes_sha1"],
            "units_sha1": "units0", "fingerprint": fp, "cacheable": cache, "seconds": 1.0, "info": {}}
    store.save(meta["key"], meta, arrays)
    return meta


def test_job_store_resume_rules(tmp_path):
    _, S, _, _ = synthetic_table()
    w = windows_for(S)[0]
    store = v19.JobStore(str(tmp_path / "jobs"))
    z = np.arange(5.0)
    _job(store, w, "EXA", v19.OK, {"main": {"role": "headline"}}, {"z__main": z})
    assert v19.done(store, v19.job_key(w["n"], "EXA"), w["scenes_sha1"], "units0", "fp") is not None
    assert v19.done(store, v19.job_key(w["n"], "EXA"), "other", "units0", "fp") is None          # other scenes
    assert v19.done(store, v19.job_key(w["n"], "EXA"), w["scenes_sha1"], "units1", "fp") is None  # other units
    assert v19.done(store, v19.job_key(w["n"], "EXA"), w["scenes_sha1"], "units0", "fp2") is None  # other code
    _job(store, w, "EXP", v19.PENDING, fp="none", cache=False)
    assert v19.done(store, v19.job_key(w["n"], "EXP"), w["scenes_sha1"], "units0", "none") is None
    _job(store, w, "EXN", v19.NOT_COMPUTABLE, fp="fp")
    assert v19.done(store, v19.job_key(w["n"], "EXN"), w["scenes_sha1"], "units0", "fp") is not None
    os.remove(os.path.join(store.root, v19.job_key(w["n"], "EXA") + ".npz"))                      # arrays lost
    assert v19.done(store, v19.job_key(w["n"], "EXA"), w["scenes_sha1"], "units0", "fp") is None
    assert v19.JobStore(store.root).arrays(v19.job_key(w["n"], "EXA")) is None
    # a json left half-written by a crash is not a job
    with open(os.path.join(store.root, "n001__EXZ.json.tmp"), "w", encoding="utf-8") as f:
        f.write("{")
    assert "n001__EXZ" not in store.keys()


def test_common_and_own_scoring_rows(tmp_path):
    _, S, _, _ = synthetic_table()
    ws = windows_for(S, ns=(125, 40))
    w = [x for x in ws if x["n"] == 40][0]
    rng = np.random.default_rng(5)
    n = 60
    truth = rng.uniform(-1, 1, n)
    unit_col = np.arange(n) // 2
    store = v19.JobStore(str(tmp_path / "jobs"))
    zm = truth + rng.normal(0, 0.1, n)
    zm3 = zm.copy()
    zm3[:5] = np.nan
    _job(store, w, "E0b", v19.OK, v19.VARIANTS["E0b"], {"z__mb0": zm, "z__mb3": zm3}, label="MAREA")
    za = truth + rng.normal(0, 0.2, n)
    za[50:] = np.nan
    _job(store, w, "E0a", v19.OK, v19.VARIANTS["E0a"], {"z__mb0": zm, "z__mb3": zm})
    _job(store, w, "E4", v19.OK, {"main": {"role": "headline", "label": "LogiTide2DEM"}}, {"z__main": za})
    _job(store, w, "E7", v19.OK, {"main": {"role": "headline"}}, {"z__main": np.full(n, np.nan)})
    _job(store, w, "E5b", v19.OK, {"main": {"role": "catalogue variant (descriptive)"}}, {"z__main": zm + 0.1})
    _job(store, w, "E1", v19.PENDING, fp="none", cache=False, label="NIDEM")
    rows = v19.build_rows("synthetic", "vaklodingen", truth, unit_col, ws, store)
    df = pd.DataFrame(rows)
    ok = df[(df["status"] == v19.OK) & (df["n_scenes"] == 40)]
    common = ok[ok["scoring"] == "common"]
    # the common set: finite truth & every headline method with a value (E7 has none: left out and noted)
    assert set(common["n_common"]) == {50}
    e4 = common[common["method"] == "E4"].iloc[0]
    assert e4["coverage"] == 1.0 and e4["n_units"] == 50
    s = v19.score_values("vaklodingen", truth[:50], za[:50], unit_col[:50])
    assert e4["rmse"] == pytest.approx(s["rmse"]) and e4["rmse_total"] == pytest.approx(
        math.sqrt(np.mean((za[:50] - truth[:50]) ** 2)))
    own = ok[(ok["scoring"] == "own") & (ok["method"] == "E4")].iloc[0]
    assert own["coverage"] == pytest.approx(50 / 60) and own["n_px"] == 25
    assert "left out of the common set" in ok[ok["method"] == "E7"].iloc[0]["notes"]
    # variants and the internal reference are scored on the common set but do not define it
    mb3 = common[(common["method"] == "E0b") & (common["variant"] == "mb3")].iloc[0]
    assert mb3["role"] == "variant" and mb3["n_units"] == 45 and mb3["coverage"] == pytest.approx(45 / 50)
    assert set(common[common["method"] == "E0a"]["role"]) == {"internal"}
    assert common[common["method"] == "E5b"].iloc[0]["role"] == "headline"     # a card's text -> a column
    pend = df[(df["method"] == "E1") & (df["n_scenes"] == 40)].iloc[0]
    assert pend["status"] == v19.PENDING and pend["role"] == "headline"
    ni = df[(df["status"] == v19.NOT_IMPLEMENTED) & (df["n_scenes"] == 40)]
    assert {"E2", "E3", "E8"} <= set(ni["method"]) and "E4" not in set(ni["method"])
    assert set(df[df["n_scenes"] == 125]["status"]) == {v19.NOT_REACHABLE}
    # the CSV round trip
    p = str(tmp_path / "synthetic.csv")
    v19.write_csv(p, rows)
    back = pd.read_csv(p)
    assert list(back.columns) == list(v19.CSV_COLUMNS) and len(back) == len(rows)


def test_record_gate():
    rng = np.random.default_rng(6)
    truth = rng.uniform(-1, 1, 50)
    zs = truth + rng.normal(0, 0.1, 50)
    uc = np.arange(50)
    pub = v19.score_values("rtk", truth, zs, uc)
    published = {"n": pub["n"], "rmse_m": pub["rmse"], "slope": pub["slope"], "r": pub["r"], "bias_m": pub["bias"]}
    g = v19.record_gate("x", "rtk", truth, uc, zs + 1e-9, zs, [0.0, 5.0], [0.0, 5.0], published, full_F=True)
    assert g["ok"] and g["validity_flips"] == 0 and g["max_abs_diff_vs_published"] < 1e-6
    z2 = zs.copy()
    z2[3] = np.nan
    assert not v19.record_gate("x", "rtk", truth, uc, z2, zs, published=published, full_F=True)["ok"]
    assert not v19.record_gate("x", "rtk", truth, uc, zs + 1e-3, zs)["ok"]
    bad = dict(published, rmse_m=published["rmse_m"] + 0.01)
    assert not v19.record_gate("x", "rtk", truth, uc, zs, zs, published=bad, full_F=True)["ok"]
    assert v19.record_gate("x", "rtk", truth, uc, zs, zs, published=bad, full_F=False)["ok"]   # subsample: stored


def test_job_plan_order():
    _, S, _, _ = synthetic_table()
    ws = windows_for(S, ns=(125, 40, 30), n_record=51)
    plan = v19.job_plan(ws, ["E0b", "E0a", "E4", "E1"])
    assert plan[0] == (51, "E0b")
    assert plan[1:3] == [(40, "E0b"), (30, "E0b")]
    assert (51, "E4") in plan and (125, "E4") not in plan                     # unreachable: never planned
    assert [p for p in plan if p[1] == "E4"] == [(51, "E4"), (40, "E4"), (30, "E4")]
    plan = v19.job_plan(ws, ["E0b", "E0a", "E4"], record_window="marea")
    assert (51, "E4") not in plan and (51, "E0a") in plan


# ── tables and curves ───────────────────────────────────────────────────────

def _fake_site(out, site, kind, ns_ok, ns_bad=(), seed=0, with_variant_card=False):
    rng = np.random.default_rng(seed)
    dates, times = make_dates(range(2021, 2026), 25, rng)
    S = {"dates": dates, "h0": np.sin(np.arange(len(dates)))}
    ws = v19.make_windows(S["dates"], S["h0"], tuple(ns_ok) + tuple(ns_bad), n_record=max(ns_ok) - 3)
    ws = [w for w in ws if w["n"] in set(ns_ok) | set(ns_bad) | {max(ns_ok) - 3}]
    for w in ws:
        if w["n"] in ns_bad:
            w.update(reachable=False, missing=w["n"] - 1, reason=f"S holds 1 screened scenes: {w['n'] - 1} missing "
                                                                  "(about 2.5 years before 2023 at 22.7 screened scenes per year)")
    n = 80
    truth = rng.uniform(-1, 1, n)
    store = v19.JobStore(os.path.join(out, f"{site}_jobs"))
    for w in ws:
        if not w["reachable"]:
            continue
        k = w["n"]
        zm = truth + rng.normal(0, 0.05 + 2.0 / k, n)
        _job(store, w, "E0b", v19.OK, v19.VARIANTS["E0b"], {"z__mb0": zm, "z__mb3": zm}, label="MAREA")
        _job(store, w, "E4", v19.OK, {"main": {"role": "headline", "label": "LogiTide2DEM (Chen & Wang 2025)"}},
             {"z__main": truth + rng.normal(0.05, 0.08, n)})
        v5 = {"main": {"role": "headline", "label": "Tseng"}}
        if with_variant_card:
            v5["main"]["catalogue_role"] = "catalogue variant of the frequency family"
        _job(store, w, "E5b", v19.OK, v5, {"z__main": truth + rng.normal(0.2, 0.2, n)})
        _job(store, w, "E1", v19.PENDING, fp="none", cache=False, label="NIDEM")
    rows = v19.build_rows(site, kind, truth, np.arange(n), ws, store)
    v19.write_csv(os.path.join(out, f"{site}.csv"), rows)


def test_tables_csv_markdown_and_spanish_latex(tmp_path):
    out = str(tmp_path)
    _fake_site(out, "villaviciosa", "rtk", (125, 100, 75, 60, 50, 40), seed=1, with_variant_card=True)
    _fake_site(out, "escalda", "vaklodingen", (60, 50, 40), ns_bad=(125, 100, 75), seed=2)
    paths = v19.tables(argparse.Namespace(out=out, sites=None, ns="125,100,75,60,50,40", no_rescore=True))
    assert any(p.endswith("table_n125") for p in paths) and any(p.endswith("table_record") for p in paths)
    tdir = os.path.join(out, "tables")
    md = open(os.path.join(tdir, "table_n125.md"), encoding="utf-8").read()
    assert "| Villaviciosa (RTK) |" in md and "not reachable: 124 screened scenes missing" in md
    assert "**" in md and "pending" in md and "Tseng linear scaling †" in md
    assert "Not implemented under the common configuration yet" in md and "E3 Granadeiro" in md
    tex = open(os.path.join(tdir, "table_n060_es.tex"), encoding="utf-8").read()
    for k in ("\\toprule", "\\midrule", "\\bottomrule", "Sitio", "cobertura de F", "\\up{", "pendiente",
              "\\caption{", "\\label{tab:fijo-60}", "$^\\dagger$", "\\providecommand{\\sub}", "\\resizebox"):
        assert k in tex, k
    tex125 = open(os.path.join(tdir, "table_n125_es.tex"), encoding="utf-8").read()
    assert "no alcanzable: faltan 124 escenas cribadas" in tex125
    doc = open(os.path.join(tdir, "all_tables_es.tex"), encoding="utf-8").read()
    assert doc.startswith("\\documentclass") and doc.count("\\begin{table}") == len(paths)
    t = pd.read_csv(os.path.join(tdir, "table_n060.csv"))
    assert set(t["site"]) >= {"villaviciosa", "escalda"} and "E0b_mb0_rmse" in t.columns
    assert {"E0b_mb0_rmse", "E4_main_rmse", "E1_main_status"} <= set(t.columns)
    assert t.set_index("site").loc["escalda", "E1_main_status"] == "pending"


def test_curves_both_languages_marea_always_deep(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    out = str(tmp_path)
    _fake_site(out, "villaviciosa", "rtk", (125, 100, 75, 60, 50, 40), seed=1)
    _fake_site(out, "escalda", "vaklodingen", (60, 50, 40), ns_bad=(125, 100, 75), seed=2)
    paths = v19.curves(argparse.Namespace(out=out, lang="es,en", scoring="common", sites=None, no_rescore=True))
    for lang in ("es", "en"):
        assert os.path.exists(os.path.join(out, f"curves_common_{lang}.png"))
        assert os.path.exists(os.path.join(out, f"curves_common_{lang}.pdf"))
        assert os.path.exists(os.path.join(out, f"curves_villaviciosa_common_{lang}.png"))
    df = v19.read_results(out, ["villaviciosa", "escalda"])
    for lang, word in (("es", "imágenes cribadas"), ("en", "screened images")):
        fig = v19.curves_figure(df, list(v19.SITE_ORDER), lang)
        texts = [t.get_text() for t in fig.texts] + [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any(word in t for t in texts)
        panels = [ax for ax in fig.axes if ax.lines]
        assert panels
        for ax in panels:
            colours = {to_hex(ln.get_color()).upper() for ln in ax.lines}
            assert v19.PAL["deep"].upper() in colours                    # MAREA in every panel with data
        plt.close(fig)
    assert v19.style_for("E0b", "mb0")["color"] == v19.PAL["deep"] == v19.style_for("E0b", "mb3")["color"]
    assert v19.style_for("E99", "main") == v19.style_for("E99", "main")   # stable for unknown ids


# ── guards ──────────────────────────────────────────────────────────────────

def test_guards():
    from tests import test_r1_guard

    static = [n for n in dir(test_r1_guard) if n.startswith("test_static")]
    assert static
    for n in static:
        getattr(test_r1_guard, n)()                                      # scans experiments/, v19 included
    src = open(v19.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and "load_rtk" in (getattr(c.func, "attr", None), getattr(c.func, "id", None))]
    assert not calls                                                     # the survey comes via v14.rtk_units()
    g = v19.v14_sha_gate()
    assert g["ok"] == (g["sha256"] == v19.V14_SHA256)
    assert not v19.v14_sha_gate(expected="0" * 64)["ok"]
    for bad in ("products_marea/x", "results/method_comparison_same_scenes", "products_escalda_10m/y"):
        with pytest.raises(SystemExit):
            v14.guard_out(bad)
    v14.guard_out(v19.OUT)
    # v19 writes nothing at import and edits no reused module
    assert "def run_job" not in src and "def invert_series" not in src
