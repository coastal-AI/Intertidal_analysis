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
    ref = {"path": "work/synthetic_reference.npz", "sha256": "0" * 64, "classes": {"0": 3, "1": 2, "2": 1}}
    st, info = v19.load_extra_store(store("a", keep_=keep[::-1].copy()), keep, (2, 3), main_dates)
    assert st is None and "keep differs" in info["reason"]
    st, info = v19.load_extra_store(store("b", side={"overpass": over}), keep, (2, 3), main_dates)
    assert st is None and "screening" in info["reason"]
    st, info = v19.load_extra_store(store("c", side={"overpass": over, "cloud_pct": cloud}), keep, (2, 3), main_dates)
    assert st is None and "reference_map" in info["reason"]                      # no map provenance: refused
    for p in os.listdir(d):
        os.remove(os.path.join(d, p))
    store("2021-2022", side={"overpass": over, "cloud_pct": cloud, "gate_2023": {"ok": True}, "reference_map": ref})
    b = FakeBoundary()
    # a boundary with gauges (any provider that is not PyTMDBoundary): refused unless the user chose
    S2, infos, note = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "refuse")
    assert S2 is S and not infos[0]["used"] and "pre-2023 levels" in infos[0]["reason"]
    S2, infos, note = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "production")
    assert infos[0]["used"] and infos[0]["dates_ignored_main_wins"] == 1
    assert "not checked" in infos[0]["reference_map_check"]                     # no production counts for this site
    idn = infos[0]["identity"]
    assert idn["npz"]["sha256"] == v19.sha256_file(os.path.join(d, "synthetic_extract_2021-2022.npz"))
    assert idn["sidecar"]["bytes"] > 0 and infos[0]["pre2023_levels"] == "production"
    early_usable = [a for i, a in enumerate(xd[:-1]) if i % 2 == 0]
    assert S2["dates"].tolist() == sorted(early_usable) + main_dates.tolist()
    assert S2["t"].is_monotonic_increasing and set(S2["src"][:len(early_usable)]) == {1}
    # the windows that hold store scenes are keyed by the store too; the others keep the dates' key
    ws = v19.make_windows(S2["dates"], S2["h0"], (10, 6), None)
    keys0 = {w["n"]: w["scenes_sha1"] for w in ws}
    v19.rekey_windows(ws, S2, infos)
    by = {w["n"]: w for w in ws}
    assert by[6]["scenes_sha1"] == keys0[6] == v19.scenes_sha1(S2["dates"][-6:])     # main scenes only
    assert by[10]["scenes_sha1"] != keys0[10] and by[10]["scenes_sha1_dates"] == keys0[10]
    assert by[10]["extra_sources"] == [1]
    other = [dict(infos[0], identity=dict(idn, npz=dict(idn["npz"], sha256="f" * 64)))]
    ws2 = v19.make_windows(S2["dates"], S2["h0"], (10, 6), None)
    v19.rekey_windows(ws2, S2, other)
    assert {w["n"]: w["scenes_sha1"] for w in ws2}[10] not in (by[10]["scenes_sha1"], keys0[10])   # a rebuilt store
    # no passed re-download gate: refused (soft gates override)
    os.remove(os.path.join(d, "synthetic_extract_2021-2022.json"))
    store("2021-2022", side={"overpass": over, "cloud_pct": cloud, "reference_map": ref})
    S3, infos, _ = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "production")
    assert S3 is S and "gate" in infos[0]["reason"]
    S3, infos, _ = v19.extend_with_extra(S, "synthetic", d, keep, (2, 3), main_dates, b, "production", soft=True)
    assert infos[0]["used"]


def test_extra_store_reference_map_and_usable_rule(tmp_path):
    """A Dutch store's sidecar must name a reference map with the production
    map's class counts, and a 'usable' list given with cloud_pct must be
    exactly cloud_pct / 100 <= 0.10 (both refused unless --soft-gates)."""
    rng = np.random.default_rng(14)
    keep = np.arange(6, dtype=np.int64)
    main_dates, _ = make_dates([2023], 4, rng)
    xd, xt = make_dates([2021, 2022], 3, rng)
    p = str(tmp_path / "escalda_extract_2021-2022.npz")
    np.savez(p, Y=np.zeros((len(xd), 6), np.float32), C=np.ones((len(xd), 6), bool), keep=keep, dates=xd,
             shape=np.array([2, 3]))
    over = {a: str(b) for a, b in zip(xd, xt)}
    cloud = {a: (5.0 if i % 2 == 0 else 50.0) for i, a in enumerate(xd)}
    good = {"path": "x", "sha256": "0" * 64, "classes": {str(k): v for k, v in
                                                          v19.PRODUCTION_REFERENCE_CLASSES["escalda"].items()}}

    def side(**kw):
        with open(p[:-4] + ".json", "w", encoding="utf-8") as f:
            json.dump(dict({"overpass": over, "cloud_pct": cloud, "gate_2023": {"ok": True}}, **kw), f)

    side(reference_map=good)
    st, info = v19.load_extra_store(p, keep, (2, 3), main_dates, site="escalda")
    assert st is not None and "equal the production" in info["reference_map_check"]
    assert st["usable"] == {a for i, a in enumerate(xd) if i % 2 == 0} and "cloud_pct" in info["screening_rule"]
    side(reference_map=dict(good, classes={"0": 1, "1": 2, "2": 3}))
    st, info = v19.load_extra_store(p, keep, (2, 3), main_dates, site="escalda")
    assert st is None and "not the production map" in info["reason"]
    st, info = v19.load_extra_store(p, keep, (2, 3), main_dates, site="escalda", soft=True)
    assert st is not None and info["screening_problems"]                       # soft: used, the problem recorded
    side(reference_map=good, usable=list(xd))                                   # 'usable' is not the 10 % rule
    st, info = v19.load_extra_store(p, keep, (2, 3), main_dates, site="escalda")
    assert st is None and "usable" in info["reason"]
    side(reference_map=good, usable=[a for i, a in enumerate(xd) if i % 2 == 0])
    st, info = v19.load_extra_store(p, keep, (2, 3), main_dates, site="escalda")
    assert st is not None


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
        "def nidem_on_scenes(site, dates, h, units_keep, pixel_rule, obs):\n"
        "    CALLS.append(dict(site=site, n=len(dates), pixel_rule=pixel_rule))\n"
        "    return {'z_keep': np.asarray(obs.h.mean() + 0.0 * obs.Y[0], float)}\n"
        "def nidem_subset(site, dates, pixel_rule='item'):\n"
        "    return {'status': 'NOT COMPUTABLE', 'reason': 'empty interval', 'mask': pixel_rule}\n"
        "def nidem_silent(site, dates):\n"
        "    return {'z_units': np.zeros(24)}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(v19, "module_path", lambda m: str(tmp_path / (m.replace(".", os.sep) + ".py")))
    fn, info = v19.find_nidem_hook(module="hookpkg.fake_v15")
    assert info["status"] == "found" and info["function"].endswith("nidem_on_scenes")
    table, S, keep_flat, px = synthetic_table()
    ctx = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=fn, nidem_info=info)
    w = [x for x in windows_for(S) if x["n"] == 40][0]
    m, arr = v19.run_one(ctx, w, "E1")
    assert m["status"] == v19.OK and arr["z__main"].shape == (len(px),)
    assert "common mask" in m["info"]["mask"] and "obs / C were passed" in m["info"]["mask"]
    import hookpkg.fake_v15 as hk
    assert hk.CALLS[-1] == {"site": "synthetic", "n": 40, "pixel_rule": "common"}
    # a hook given the common inputs that does not say which rule it applied: E1 (common) only, never E1i
    m, arr = v19.run_one(ctx, w, "E1i")
    assert m["status"] == v19.ERROR and "needs 'item'" in m["reason"] and arr is None
    fn2, info2 = v19.find_nidem_hook(name="nidem_subset", module="hookpkg.fake_v15")
    ctx.nidem_fn, ctx.nidem_info = fn2, info2
    m, arr = v19.run_one(ctx, w, "E1")
    assert m["status"] == v19.NOT_COMPUTABLE and "empty interval" in m["reason"] and arr is None
    m, arr = v19.run_one(ctx, w, "E1i")                              # the hook applied the rule it was asked for
    assert m["status"] == v19.NOT_COMPUTABLE and m["info"]["pixel_rule_applied"] == "item"
    # a hook that says nothing about its mask and gets no common inputs: never labelled
    fn3, info3 = v19.find_nidem_hook(name="nidem_silent", module="hookpkg.fake_v15")
    ctx.nidem_fn, ctx.nidem_info = fn3, info3
    m, arr = v19.run_one(ctx, w, "E1")
    assert m["status"] == v19.ERROR and "applied pixel rule None" in m["reason"] and not m["cacheable"]
    m, arr = v19.run_one(ctx, w, "E1i")
    assert m["status"] == v19.PENDING and "pixel_rule" in m["reason"] and not m["cacheable"]
    # the per-unit / per-pixel / per-keep readings of a hook's answer
    uk, uc = np.array([3, 5, 7]), np.array([0, 1, 1])
    assert np.array_equal(v19.hook_values(np.arange(10.0), 3, uc, uk, 2, 10), [3.0, 5.0, 7.0])
    assert np.array_equal(v19.hook_values({"z_px": np.array([1.0, 2.0])}, 3, uc, uk, 2, 10), [1.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        v19.hook_values(np.zeros(4), 3, uc, uk, 2, 10)


# ── the job store, scoring and the record gate ──────────────────────────────

def _job(store, w, mid, status, variants=None, arrays=None, fp="fp", cache=True, label=None, seconds_estimator=0.75):
    meta = {"key": v19.job_key(w["n"], mid), "site": "synthetic", "n": w["n"], "window": w["label"],
            "is_record": w["is_record"], "method": mid, "label": label or mid, "status": status, "reason": "r",
            "source": "s", "fidelity": "F3", "variants": variants or {}, "scenes_sha1": w["scenes_sha1"],
            "units_sha1": "units0", "fingerprint": fp, "cacheable": cache, "seconds": 1.0, "info": {},
            "seconds_estimator": seconds_estimator, "host": "testhost", "threads": "OMP=2;OPENBLAS=2;MKL=2"}
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
    assert plan[0] == ("n51", "E0b")
    assert plan[1:3] == [("n40", "E0b"), ("n30", "E0b")]
    assert ("n51", "E4") in plan and ("n125", "E4") not in plan                 # unreachable: never planned
    assert [p for p in plan if p[1] == "E4"] == [("n51", "E4"), ("n40", "E4"), ("n30", "E4")]
    plan = v19.job_plan(ws, ["E0b", "E0a", "E4"], record_window="marea")
    assert ("n51", "E4") not in plan and ("n51", "E0a") in plan
    # a record that ends after the truth date: the gate window, MAREA and no delay only, its own job keys
    end = str(S["dates"][30])
    ws = v19.make_windows(S["dates"], S["h0"], (40, 30, 20), 51, end=end)
    rec = [w for w in ws if w["is_record"]][0]
    assert rec["gate_only"] and rec["label"] == "rec51" and rec["stop"] == len(S["dates"])
    plan = v19.job_plan(ws, ["E0b", "E0a", "E4", "E1"])
    assert plan[0] == ("rec51", "E0b") and ("rec51", "E0a") in plan
    assert not [p for p in plan if p[0] == "rec51" and p[1] in ("E4", "E1")]
    assert [p[0] for p in plan if p[1] == "E4"] == ["n30", "n20"]             # n40 > 31 scenes up to the end
    assert v19.wkey(rec, "E0b") == "r051__E0b" and v19.job_key(51, "E0b") == "n051__E0b"


# ── tables and curves ───────────────────────────────────────────────────────

def _fake_truth(out, site):
    """A settings JSON with this design's truth block (the tables read only such sites)."""
    lab = "PNOA LiDAR, flight 2016-08-01" if site == "villaviciosa" else "Vaklodingen 2025"
    t = {"design": v19.TRUTH_DESIGN, "status": "ready", "label_en": lab, "label_es": lab, "end": "2025-12-31",
         "datum": "orthometric", "path": "x.tif",
         "counts": {"units_F": 80, "units_scored": 80, "cells_in_F": 40, "survey_year": 2025, "sheets_in_F": ["a"],
                    "sheets_with_year": 1, "sheets_total": 2}}
    v14.write_json(os.path.join(out, f"{site}_settings.json"), {"site": site, "truth": t})


def _fake_site(out, site, kind, ns_ok, ns_bad=(), seed=0, with_variant_card=False, truth_block=True):
    rng = np.random.default_rng(seed)
    dates, times = make_dates(range(2021, 2026), 25, rng)
    S = {"dates": dates, "h0": np.sin(np.arange(len(dates)))}
    ws = v19.make_windows(S["dates"], S["h0"], tuple(ns_ok) + tuple(ns_bad), n_record=max(ns_ok) - 3)
    ws = [w for w in ws if w["n"] in set(ns_ok) | set(ns_bad) | {max(ns_ok) - 3}]
    for w in ws:
        if w["n"] in ns_bad:
            w.update(reachable=False, missing=w["n"] - 1,
                     reason=v19.window_end_reason(1, "2025-12-31", w["n"], "2023-01-05", 22.7))
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
    timing = {"site_shared_s": 12.5, "window_select_s": {w["label"]: 0.25 for w in ws}}
    rows = v19.build_rows(site, kind, truth, np.arange(n), ws, store, timing=timing)
    v19.write_csv(os.path.join(out, f"{site}.csv"), rows)
    if truth_block:
        _fake_truth(out, site)


def test_tables_csv_markdown_and_spanish_latex(tmp_path):
    out = str(tmp_path)
    _fake_site(out, "villaviciosa", "rtk", (125, 100, 75, 60, 50, 40), seed=1, with_variant_card=True)
    _fake_site(out, "escalda", "vaklodingen", (60, 50, 40), ns_bad=(125, 100, 75), seed=2)
    paths = v19.tables(argparse.Namespace(out=out, sites=None, ns="125,100,75,60,50,40", no_rescore=True))
    assert any(p.endswith("table_n125") for p in paths) and any(p.endswith("table_record") for p in paths)
    tdir = os.path.join(out, "tables")
    md = open(os.path.join(tdir, "table_n125.md"), encoding="utf-8").read()
    assert "| Villaviciosa (PNOA LiDAR, flight 2016-08-01) |" in md and "RTK" not in md
    assert "not reachable: 1 screened images up to 2025-12-31 (124 missing)" in md      # the images up to the date
    assert "**" in md and "pending" in md and "Tseng linear scaling †" in md
    assert "Not implemented under the common configuration yet" in md and "E3 Granadeiro" in md
    assert "Vaklodingen gives the survey year only" in md and "time (s)" in md and "shared 12.8" in md
    tex = open(os.path.join(tdir, "table_n060_es.tex"), encoding="utf-8").read()
    for k in ("\\toprule", "\\midrule", "\\bottomrule", "Sitio", "cobertura de F", "\\up{", "pendiente",
              "\\caption{", "\\label{tab:fijo-60}", "$^\\dagger$", "\\providecommand{\\sub}", "\\resizebox"):
        assert k in tex, k
    tex125 = open(os.path.join(tdir, "table_n125_es.tex"), encoding="utf-8").read()
    assert "no alcanzable: 1 im\\'agenes cribadas hasta el 2025-12-31 (faltan 124)" in tex125
    assert "tiempo (s)" in tex and "solo da el a\\~no del levantamiento" in tex
    doc = open(os.path.join(tdir, "all_tables_es.tex"), encoding="utf-8").read()
    assert doc.startswith("\\documentclass") and doc.count("\\begin{table}") == len(paths)
    t = pd.read_csv(os.path.join(tdir, "table_n060.csv"))
    assert set(t["site"]) >= {"villaviciosa", "escalda"} and "E0b_mb0_common_rmse" in t.columns
    assert {"E0b_mb0_common_rmse", "E4_main_common_rmse", "E1_main_status", "common_coverage"} <= set(t.columns)
    assert t.set_index("site").loc["escalda", "E1_main_status"] == "pending"
    # every metric, both scorings: cRMSE, RMSE, bias, slope, r, n, coverage
    for s in ("common", "own"):
        assert {f"E4_main_{s}_{x}" for x in v19.TABLE_METRICS} <= set(t.columns)
    e = t.set_index("site").loc["escalda"]
    assert e["E4_main_common_coverage"] == 1.0 and 0 < e["E4_main_own_coverage"] <= 1
    assert e["common_coverage"] == e["n_common"] / e["n_set"] and e["E4_main_common_rmse_total"] > 0
    for k in ("RMSE (uncentred), common", "slope / r, common", "own units: cRMSE (bias)",
              "own units: n / coverage of F", "cRMSE = RMSE after removing each series' median", "of F)"):
        assert k in md, k
    for k in ("RMSE sin centrar", "propios: cRMSE", "cobertura com\\'un"):
        assert k in tex, k


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


# ── the truth-date design (2026-10-26) ──────────────────────────────────────

def _sheet(path, x0, y0, nx, ny, stamps, values):
    """A tiny Vaklodingen-like sheet: z (time, y, x) on a 20 m RD grid."""
    import xarray as xr
    t = pd.to_datetime(stamps)
    z = np.stack([np.full((ny, nx), np.nan) if v is None else np.asarray(v, float) for v in values])
    xr.Dataset({"z": (("time", "y", "x"), z)},
               coords={"time": t, "x": x0 + 20.0 * np.arange(nx), "y": y0 - 20.0 * np.arange(ny)}).to_netcdf(path)


def test_truth_grid_year_is_v1_restricted_to_one_year(tmp_path, monkeypatch):
    """The Dutch truth: only the cells surveyed in 2025, per sheet in v1's
    order; v1's nearest-cell sampling (values asserted equal to
    v1.sample_truth) with the cell of every point; v14.dutch_truth's range
    filter; a sheet without a 2025 survey is listed, not used."""
    from experiments import v1_vaklodingen_validate as v1
    from experiments import v7_waterline_contour as v7

    a = np.arange(12.0).reshape(3, 4) / 10.0
    _sheet(tmp_path / "A.nc", 1000.0, 5000.0, 4, 3, ["2023-01-01", "2025-01-01"], [a - 5.0, a])
    b = np.full((3, 4), 0.9)
    b[0, 0] = np.nan
    _sheet(tmp_path / "B.nc", 1080.0, 5000.0, 4, 3, ["2024-01-01"], [b])        # no 2025 survey
    cells, info = v19.truth_grid_year("escalda", 2025, str(tmp_path), sheets=["A.nc", "B.nc", "C.nc"])
    assert list(cells) == ["A.nc"] and np.array_equal(cells["A.nc"][2], a)
    used = {r["sheet"]: r for r in info["sheets"]}
    assert used["A.nc"]["used"] and used["A.nc"]["cells"] == 12 and used["A.nc"]["stamps"] == ["2025-01-01"]
    assert not used["B.nc"]["used"] and "no survey in 2025" in used["B.nc"]["reason"]
    assert "missing" in used["C.nc"]["reason"] and info["n_sheets_with_year"] == 1 and info["cells_with_year"] == 12
    # v1's grid over 2019-2025 would take the 2024 survey of sheet B: the one-year grid never does
    xq = np.array([1000.0, 1021.0, 1060.0, 1100.0, 1500.0])
    yq = np.array([5000.0, 4980.0, 4960.0, 5000.0, 5000.0])
    val, yr, cid = v19.sample_cells(cells, xq, yq)
    ref, _ = v1.sample_truth(cells, xq, yq)
    assert np.array_equal(val, ref, equal_nan=True) and np.isnan(val[3]) and np.isnan(val[4])
    assert val[0] == a[0, 0] and val[1] == a[1, 1] and cid[0] != cid[1] and cid[4] == -1
    # dutch_truth_year: v14.dutch_truth's signature, grid transform, sampling and range filter
    W = 3
    grid = {"H": 2, "W": W, "xs": np.array([1000.0, 1021.0, 1060.0]), "ys": np.array([5000.0, 4980.0]),
            "crs": "EPSG:28992"}
    monkeypatch.setattr(v7, "cube_grid", lambda cube: grid)
    monkeypatch.setattr(v19, "truth_grid_year", lambda site, year, tdir: (cells, info))
    holder = {}
    keep = np.array([0, 2, 4, 5], np.int64)                    # (0,0) (0,2) (1,1) (1,2) of a 2 x 3 grid
    z_plain = np.array([0.0, 0.3])
    truth, tinfo = v19.dutch_truth_year("escalda", {"cube": "x.nc"}, keep, 2, W, z_plain, 2025, None, holder)
    raw = np.array([a[0, 0], a[0, 3], a[1, 1], a[1, 3]])        # the nearest 20 m cells
    want = np.where((raw >= -0.25) & (raw <= 0.3 + 0.25), raw, np.nan)
    assert np.array_equal(truth, want, equal_nan=True) and tinfo["survey_year"] == 2025
    assert tinfo["range_filter_m"] == pytest.approx([-0.25, 0.55]) and tinfo["truth_px"] == 4
    assert tinfo["truth_in_range"] == 3 and tinfo["cells_in_range"] == 3 and tinfo["n_sheets_with_year"] == 1
    assert holder["cell_id"].shape == keep.shape and holder["seconds"] >= 0


def test_truth_swap_is_scoped_and_villaviciosa_never_reads_the_rtk(tmp_path, monkeypatch):
    from pyintertidal import rtk

    orig = v14.dutch_truth
    with v19.truth_swap("dutch_truth", lambda *a: "mine"):
        assert v14.dutch_truth() == "mine"
    assert v14.dutch_truth is orig
    with pytest.raises(RuntimeError):
        with v19.truth_swap("rtk_units", lambda *a: None):
            raise RuntimeError("boom")
    assert v14.rtk_units.__name__ == "rtk_units"                   # restored after an error too
    # the raster units: v14.rtk_units' arguments and return, F = in range & finite truth; no product decides
    # it (the stored MAREA, NaN at unit 4, is only recorded)
    monkeypatch.setattr(rtk, "load_rtk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RTK read")))
    truth_raw = np.array([0.1, np.nan, 2.0, -0.4, 0.3, -0.5])       # 2.0 and -0.5 outside [-0.45, 1.25]
    holder = {}
    fn = v19.raster_units(truth_raw, -0.2, 1.0, {"truth": "PNOA"}, holder)
    F, t, info = fn(np.arange(6), 10, np.array([1.0, 1.0, 1.0, 1.0, np.nan, 1.0]))
    assert F.tolist() == [0, 3, 4] and t.tolist() == [0.1, -0.4, 0.3] and info["truth_in_range"] == 3
    assert info["units_in_F_with_stored_marea"] == 2 and info["F_rule"] == v19.F_RULE
    assert info["range_filter_m"] == [-0.45, 1.25] and "z_stored_seen" in holder
    # Villaviciosa without the date: 'pending truth', nothing loaded
    a = argparse.Namespace(vva_truth=None, vva_truth_date=None)
    t = v19.truth_definition("villaviciosa", a)
    assert t["status"] == "pending" and "--vva-truth-date" in t["reason"] and "RTK is never used" in t["reason"]
    f = tmp_path / "dtm.tif"
    f.write_bytes(b"x")
    t = v19.truth_definition("villaviciosa", argparse.Namespace(vva_truth=str(f), vva_truth_date="2016-08-15",
                                                                vva_nodelay=None))
    assert t["status"] == "ready" and t["end"] == "2016-08-15" and t["file"]["sha256"] and t["kind"] == "raster"
    with pytest.raises(SystemExit):
        v19.truth_definition("villaviciosa", argparse.Namespace(vva_truth=str(f), vva_truth_date="2016-13-01"))
    d = v19.truth_definition("wadden", a)
    assert d["end"] == "2025-12-31" and "survey year only" in d["end_rule"] and d["year"] == 2025


def test_pnoa_raster_truth_and_range_gate(tmp_path):
    """A tiny DTM: a planar ground and a flattened water patch (equal values,
    >= 50 px), sampled to a 10 m grid by benchmark/truth.py's pnoa_truth."""
    import rasterio
    from rasterio.transform import from_origin

    mdt, grid = tmp_path / "mdt.tif", tmp_path / "grid.tif"
    yy, xx = np.mgrid[0:40, 0:40]
    z = (0.01 * xx + 0.02 * yy).astype("float32")
    z[0:10, 0:10] = 1.234                                    # 100 px of one value: water
    prof = dict(driver="GTiff", height=40, width=40, count=1, dtype="float32", crs="EPSG:32630", nodata=-32767.0,
                transform=from_origin(300000.0, 4800080.0, 2.0, 2.0))
    with rasterio.open(mdt, "w", **prof) as dst:
        dst.write(z, 1)
    prof.update(height=8, width=8, transform=from_origin(300000.0, 4800080.0, 10.0, 10.0))
    with rasterio.open(grid, "w", **prof) as dst:
        dst.write(np.zeros((8, 8), "float32"), 1)
    keep = np.arange(64, dtype=np.int64)
    truth, info, secs = v19.vva_raster_truth(str(mdt), keep, 8, 8, cube=None, grid_path=str(grid))
    t = truth.reshape(8, 8)
    assert np.isnan(t[0:2, 0:2]).all() and np.isfinite(t[3:, 3:]).all()     # the patch is masked, the ground kept
    assert abs(t[5, 5] - (0.01 * 27 + 0.02 * 27)) < 1e-4 and info["pixel_m"] == 10.0   # the 10 m mean of the plane
    assert info["patches"]["n_patches"] == 1 and secs >= 0
    # the range filter's no-delay product must be the stored production
    np.savez(tmp_path / "ext.npz", no_delay=np.linspace(-2, 1, 64).reshape(8, 8), MAREA=np.ones((8, 8)))
    lo, hi, g = v19.vva_nodelay_range(str(tmp_path / "ext.npz"), keep, np.ones(64))
    assert (lo, hi) == (-2.0, 1.0) and g["ok"]
    lo, hi, g = v19.vva_nodelay_range(str(tmp_path / "ext.npz"), keep, np.full(64, 2.0))
    assert not g["ok"]


def test_windows_end_on_the_truth_date_and_say_what_is_available():
    rng = np.random.default_rng(3)
    dates, _ = make_dates(range(2016, 2026), 20, rng)          # 200 screened scenes 2016-2025
    h0 = np.sin(np.arange(len(dates)))
    end = "2017-06-30"
    ws = v19.make_windows(dates, h0, v19.NS, 97, end=end)
    k = int((dates <= end).sum())
    by = {w["label"]: w for w in ws}
    assert k < 40 and all(not by[f"n{n}"]["reachable"] for n in v19.NS)
    txt = v19._missing_text(by["n40"]["reason"], "en")
    assert txt == f"not reachable: {k} screened images up to {end} ({40 - k} missing)"
    assert v19._missing_text(by["n40"]["reason"], "es").startswith(f"no alcanzable: {k} im")
    rec = by["rec97"]
    assert rec["gate_only"] and rec["reachable"] and rec["first_date"] == dates[-97]
    assert "after the truth date" in v19._missing_text(rec["reason"], "en")
    assert v19.table_windows(ws) == [] and v19.check_nested(ws)["ok"]
    # a later date: windows of scenes on or before it, nested, the span to the date
    end = "2024-06-30"
    ws = v19.make_windows(dates, h0, v19.NS, 97, end=end)
    tw = v19.table_windows(ws)
    k = int((dates <= end).sum())
    assert [w["n"] for w in tw] == list(v19.NS) and all(w["stop"] == k and w["end_date"] == end for w in tw)
    assert all(w["last_date"] <= end for w in tw) and v19.check_nested(ws)["ok"]
    assert v19.level_end(tw[0]) == "2024-07-01" and v19.window_slice(tw[0]) == slice(k - 125, k)


def test_pending_truth_rows_and_tables(tmp_path, monkeypatch):
    from pyintertidal import rtk

    monkeypatch.setattr(rtk, "load_rtk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RTK read")))
    out = str(tmp_path)
    args = argparse.Namespace(site="villaviciosa", out=out, tag="", max_minutes=1.0, soft_gates=False,
                              ns="125,100,75,60,50,40", vva_truth=None, vva_truth_date=None)
    st = v19.run(args)
    assert st["status"] == "pending_truth"
    df = pd.read_csv(os.path.join(out, "villaviciosa.csv"))
    assert (df["status"] == v19.PENDING_TRUTH).all() and len(df) == 6
    _fake_site(out, "escalda", "vaklodingen", (60, 50, 40), ns_bad=(125, 100, 75), seed=2)
    v19.tables(argparse.Namespace(out=out, sites=None, ns="125,100,75,60,50,40", no_rescore=False))
    md = open(os.path.join(out, "tables", "table_n060.md"), encoding="utf-8").read()
    assert "| Villaviciosa (PNOA LiDAR) | pending truth:" in md and "--vva-truth-date" in md
    # a CSV of an earlier truth (no truth block in the settings) is never tabulated
    _fake_site(out, "wadden", "vaklodingen", (60, 50, 40), seed=4, truth_block=False)
    df = v19.read_results(out, ["escalda", "wadden"])
    assert set(df["site"]) == {"escalda"}


def test_timing_columns_exist_and_are_non_negative(tmp_path):
    """The co-author's compute times: the estimator seconds per (site, n,
    method), the shared seconds of (site, n), the host and threads, in the
    per-site CSV, the per-n table CSV, the JSON summary and the markdown."""
    table, S, keep_flat, px = synthetic_table()
    ctx = make_ctx(tmp_path, table, S, keep_flat, px)
    w = [x for x in windows_for(S) if x["n"] == 40][0]
    m, arr = v19.run_one(ctx, w, "E0b")
    assert m["seconds_estimator"] >= 0 and m["host"] and "threads" in m and ctx.window_select_s["n40"] >= 0
    z_ref, _ = v19.marea_job(table, v19.window_rows(w), SEED)                 # timing never changes a result
    assert all(np.array_equal(arr[f"z__{k}"], z_ref[k], equal_nan=True) for k in z_ref)
    out = str(tmp_path / "o")
    os.makedirs(out)
    _fake_site(out, "escalda", "vaklodingen", (60, 50, 40), seed=2)
    rows = pd.read_csv(os.path.join(out, "escalda.csv"))
    ok = rows[rows["status"] == v19.OK]
    assert (ok["seconds_estimator"] >= 0).all() and (ok["seconds_shared"] >= 0).all()
    assert (ok["host"] == "testhost").all() and ok["threads"].str.contains("OMP=2").all()
    v19.tables(argparse.Namespace(out=out, sites=["escalda"], ns="60,50,40", no_rescore=True))
    t = pd.read_csv(os.path.join(out, "tables", "table_n040.csv"))
    sec = [c for c in t.columns if c.startswith("seconds_")]
    assert {"seconds_E0b", "seconds_E4", "seconds_shared"} <= set(sec) and "host" in t.columns
    vals = t.loc[t["state"] == "ok", sec].to_numpy(float)
    assert np.isfinite(vals).any() and (vals[np.isfinite(vals)] >= 0).all()
    js = json.load(open(os.path.join(out, "tables", "timing_summary.json"), encoding="utf-8"))
    assert js["rows"] and all(r["seconds_estimator"] is None or r["seconds_estimator"] >= 0 for r in js["rows"])
    assert "time (s)" in open(os.path.join(out, "tables", "table_n040.md"), encoding="utf-8").read()


def _batch_hook(calls, wrong_rule=False, fail_key=None, exc=None):
    """A fake v15 hook taking subsets: answers every subset with its own pixel
    rule (``wrong_rule``: the other one), fails on the subsets whose key
    starts with ``fail_key`` (only when asked for more than one subset with
    ``exc`` = 'batch'), or raises ``exc``."""
    def nidem_on_scenes(site, units_keep, subsets, truth=None, kind=None, soft_gates=False):
        calls.append(sorted(subsets))
        if exc is not None and exc != "batch":
            raise exc
        if fail_key and any(k.startswith(fail_key) for k in subsets) and (exc != "batch" or len(subsets) > 1):
            raise ValueError(f"subset {fail_key!r}: 9 date(s) are not extraction (cube) dates")
        out = {}
        for k, spec in subsets.items():
            assert len(spec["dates"]) == len(spec["h"]) and spec["rows"] is not None
            rule = spec["pixel_rule"]
            if wrong_rule:
                rule = "item" if rule == "common" else "common"
            out[k] = {"status": "ok", "z_units": np.asarray(truth) + (0.1 if rule == "common" else 0.2),
                      "seconds_estimator": 1.5, "mask": rule, "pixel_rule": f"{rule} (fake)",
                      "gate": {"ok": True, "bitwise": True}}
        return out
    return nidem_on_scenes


def test_nidem_batch_hook_one_call_for_every_window_and_rule(tmp_path):
    """E1 (common mask, headline) and E1i (ITEM's mask, a sensitivity) of
    every window from ONE hook call; the rule the hook says it applied must be
    the column's."""
    table, S, keep_flat, px = synthetic_table()
    calls = []
    ctx = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=_batch_hook(calls),
                   nidem_info={"function": "fake.nidem_on_scenes", "file": {"sha256": "0" * 64}})
    ws = {w["n"]: w for w in windows_for(S, ns=(40, 30))}
    ctx.nidem_todo = [(ws[40], "E1"), (ws[30], "E1"), (ws[40], "E1i"), (ws[30], "E1i")]
    m40, a40 = v19.run_one(ctx, ws[40], "E1")
    m30, a30 = v19.run_one(ctx, ws[30], "E1")
    i40, b40 = v19.run_one(ctx, ws[40], "E1i")
    assert calls == [["n30|common", "n30|item", "n40|common", "n40|item"]]
    assert m40["status"] == m30["status"] == i40["status"] == v19.OK and m40["seconds_estimator"] == 1.5
    assert np.allclose(a30["z__main"], table.truth + 0.1) and np.allclose(b40["z__main"], table.truth + 0.2)
    assert "common mask" in m40["info"]["mask"] and "DECLARED SENSITIVITY" in i40["info"]["mask"]
    assert m40["variants"]["main"]["role"] == "headline" and i40["variants"]["main"]["role"] == "variant"
    assert m40["fingerprint"].endswith("|rule:common") and i40["fingerprint"].endswith("|rule:item")
    assert ctx.nidem_gate["ok"] and ctx.nidem_batch_info["windows"] == ["n40|common", "n30|common", "n40|item",
                                                                        "n30|item"]
    # a hook that applied the other rule: an error, never cached, never labelled
    ctx2 = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=_batch_hook([], wrong_rule=True),
                    nidem_info={"function": "fake.nidem_on_scenes", "file": {"sha256": "0" * 64}})
    ctx2.nidem_todo = [(ws[40], "E1")]
    m, arr = v19.run_one(ctx2, ws[40], "E1")
    assert m["status"] == v19.ERROR and "applied pixel rule 'item'" in m["reason"] and arr is None
    assert not m["cacheable"]


def test_nidem_windows_with_extra_store_scenes_are_pending_and_the_rest_computed(tmp_path):
    """A window holding pre-2023 extra-store scenes (no cube) is not sent to
    the hook: 'pending'; the windows of main-extraction scenes come from one
    call and are computed."""
    table, S, keep_flat, px = synthetic_table()
    S["src"][:8] = 1                                        # the 8 oldest scenes from an extra store
    calls = []
    ctx = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=_batch_hook(calls),
                   nidem_info={"function": "fake.nidem_on_scenes", "file": {"sha256": "0" * 64}})
    ws = {w["n"]: w for w in windows_for(S, ns=(45, 40, 30))}            # 51 scenes: n45 holds 2 store scenes
    ctx.nidem_todo = [(ws[n], m) for m in ("E1", "E1i") for n in (45, 40, 30)]
    out = {(n, m): v19.run_one(ctx, ws[n], m) for m in ("E1", "E1i") for n in (45, 40, 30)}
    assert calls == [["n30|common", "n30|item", "n40|common", "n40|item"]]          # one call, cube windows only
    for m in ("E1", "E1i"):
        meta, arr = out[(45, m)]
        assert meta["status"] == v19.PENDING and not meta["cacheable"] and arr is None
        assert "2 of the window's 45 scenes come from an extra store" in meta["reason"]
        assert out[(40, m)][0]["status"] == out[(30, m)][0]["status"] == v19.OK


def test_nidem_batch_failure_is_retried_per_job_unless_site_level(tmp_path):
    table, S, keep_flat, px = synthetic_table()
    ws = {w["n"]: w for w in windows_for(S, ns=(40, 30))}
    info = {"function": "fake.nidem_on_scenes", "file": {"sha256": "0" * 64}}
    # one bad window: the batched call fails, each job is retried alone: the bad one is an error, the others
    # are computed
    calls = []
    ctx = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=_batch_hook(calls, fail_key="n40"), nidem_info=info)
    ctx.nidem_todo = [(ws[40], "E1"), (ws[30], "E1")]
    m30, _ = v19.run_one(ctx, ws[30], "E1")
    m40, _ = v19.run_one(ctx, ws[40], "E1")
    assert calls == [["n30|common", "n40|common"], ["n40|common"], ["n30|common"]]
    assert m30["status"] == v19.OK and m40["status"] == v19.ERROR and "not extraction" in m40["reason"]
    assert ctx.nidem_batch_info["retried_one_by_one"]
    # a batch that fails only as a batch: every job computed on its own
    calls = []
    ctx = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=_batch_hook(calls, fail_key="n40", exc="batch"),
                   nidem_info=info)
    ctx.nidem_todo = [(ws[40], "E1"), (ws[30], "E1")]
    assert all(v19.run_one(ctx, ws[n], "E1")[0]["status"] == v19.OK for n in (30, 40)) and len(calls) == 3
    # a site-level failure (a refused input: SystemExit; a failed gate) is not retried, and never stops the run
    for exc in (SystemExit("[synthetic] cube dates differ from the extraction dates"),
                type("HookGateError", (RuntimeError,), {})("NIDEM hook gate(s) ['x'] failed")):
        calls = []
        ctx = make_ctx(tmp_path, table, S, keep_flat, px, nidem_fn=_batch_hook(calls, exc=exc), nidem_info=info)
        ctx.nidem_todo = [(ws[40], "E1"), (ws[30], "E1")]
        ms = [v19.run_one(ctx, ws[n], "E1")[0] for n in (40, 30)]
        assert len(calls) == 1 and all(m["status"] == v19.ERROR and not m["cacheable"] for m in ms)


def test_the_real_v15_hook_is_found_and_takes_subsets_and_pixel_rule():
    fn, info = v19.find_nidem_hook()
    assert info["status"] == "found" and info["function"] == "experiments.v15_nidem_scene_curves.nidem_on_scenes"
    p = v19.hook_params(fn)
    assert "subsets" in p and "pixel_rule" in p and not {"mask", "obs", "C", "Y"} & p   # the rule per subset
    need = [k for k, q in __import__("inspect").signature(fn).parameters.items() if q.default is q.empty]
    assert need == ["site"]


def _fake_v14_site(tmp_path, monkeypatch, site, years=(2023, 2024, 2025), per_year=17):
    """run()'s inputs on a tiny synthetic site: an extraction npz, v14.load_site
    / run_gates / reference_points replaced by synthetic ones (load_site
    calls the unit function the swap put in place for cfg['truth'] == 'rtk',
    as v14's does), the production record = every scene, the stored z =
    MAREA's own output on it at every keep pixel (so the record gate must
    pass on any F)."""
    table, S, keep_flat, px = synthetic_table(T_years=years, per_year=per_year, Pk=60, n_units=40)
    full, _, _, _ = synthetic_table(T_years=years, per_year=per_year, Pk=60, n_units=60)      # the same scenes
    T, Pk = len(S["dates"]), len(keep_flat)
    assert np.array_equal(full.Yk, table.Yk, equal_nan=True)
    Yk, Ck = np.array(table.Yk), np.array(table.Ck)
    ext = tmp_path / "marea_extract.npz"
    np.savez(ext, Y=Yk, C=Ck, keep=keep_flat, dates=S["dates"], shape=np.array([12, 10]))
    b = table.levels.boundary
    rec = {"time_index": np.arange(T), "dates": S["dates"], "t_real": S["t"], "h": np.asarray(table.levels[0]),
           "boundary": "fake boundary", "boundary_obj": b, "rule": "all"}
    res = {"record_dates": [str(d) for d in S["dates"]], "centros_km": [1.5, 4.5], "tau_min": [0.0, 30.0],
           "rango_marea_m": [float(np.min(rec["h"])), float(np.max(rec["h"]))]}
    z_stored = np.asarray(v19.marea_job(full, np.arange(T), SEED)[0]["mb0"], float)   # MAREA's record product
    seen = {}

    def load_site(site_, cfg, lags, max_px, seed, pools):
        seen["cfg_truth"] = cfg.get("truth")
        if cfg.get("truth") == "rtk":
            F, truth_u, tinfo = v14.rtk_units(keep_flat, 10, z_stored)         # the swapped unit function
            seen["rtk_units"] = tinfo
            units = np.asarray(F, np.int64)
        else:
            truth_px, tinfo = v14.dutch_truth(site_, cfg, keep_flat, 12, 10, np.array([-1.0, 1.0]))
            seen["dutch_truth"] = tinfo
            units = np.flatnonzero(np.isfinite(truth_px) & np.isfinite(z_stored))
            truth_u = truth_px[units]
        pxs = np.unique(units)
        sd = v14.SiteData(site_, full.Y0[:, pxs], full.C[:, pxs], v14.LevelTable(b, S["t"]),
                          {"screened": np.arange(T), "all": np.arange(T)}, np.searchsorted(pxs, units), truth_u,
                          "rtk" if cfg.get("truth") == "rtk" else "vaklodingen", np.zeros(len(pxs)),
                          s_km=full.s_km[pxs], m2a=dict(table.m2a))
        sd.levels.ensure(sorted({0, *(int(t) for t in marea.BANK_TAUS), *range(0, 181)}))
        extras = dict(res=res, rec=rec, keep=keep_flat, z_stored=z_stored, unit_keep_full=units,
                      unit_truth_full=truth_u, units_keep=units, px=pxs, kind=sd.kind, idx_scr=np.arange(T),
                      dates_all=S["dates"], rec_dates=list(res["record_dates"]))
        meta = {"fixed_set": dict(tinfo, pixel_m=10.0, definition="v14's F (finite stored MAREA)",
                                  unit="RTK point"), "inputs": {}}
        return sd, meta, extras

    monkeypatch.setattr(v14, "load_site", load_site)
    monkeypatch.setattr(v14, "run_gates", lambda *a, **k: (
        {"stored_product_vs_published_row": {"ok": False, "checked": True}, "levels": {"ok": True}}, False, {}))
    monkeypatch.setattr(v14, "reference_points", lambda *a, **k: [])
    monkeypatch.setattr(v19, "lag_fit_columns", lambda cfg, centers: (np.arange(Pk), table.m2a["band"]))
    over = tmp_path / "overpass_times.json"                         # Villaviciosa's screening reads these two
    with open(over, "w", encoding="utf-8") as f:
        json.dump({d: t.isoformat() for d, t in zip(S["dates"], S["t"])}, f)
    with open(tmp_path / "usable.json", "w", encoding="utf-8") as f:
        json.dump([str(d) for d in S["dates"]], f)
    cfg = {"extract": str(ext), "product": str(tmp_path), "comparison": None, "nidem": str(tmp_path), "cube": None,
           "overpass": str(over)}
    monkeypatch.setattr(v19, "site_cfg", lambda args: (dict(cfg), {}))
    calls = []

    def nidem_on_scenes(site, units_keep, subsets, truth=None, kind=None, io_lock=None, inputs=None):
        calls.append(list(subsets))
        return {k: {"status": "ok", "z_units": np.asarray(truth) + 0.05, "seconds_estimator": 0.3,
                    "mask": spec["pixel_rule"], "pixel_rule": f"{spec['pixel_rule']} (fake)",
                    "gate": {"ok": True, "bitwise": True}} for k, spec in subsets.items()}

    monkeypatch.setattr(v19, "find_nidem_hook", lambda name=None: (
        nidem_on_scenes, {"status": "found", "function": "fake.nidem_on_scenes", "file": {"sha256": "1" * 64}}))
    return table, S, keep_flat, px, z_stored, seen, calls


def _run_args(out, site, **kw):
    a = dict(site=site, out=out, tag="", max_minutes=30.0, soft_gates=False, ns="60,50,40", seed=SEED, max_px=None,
             usable_json=None, extra_dir=os.path.join(out, "no_extra"), pre2023_levels="refuse", registry="off",
             with_variants=False, nidem="auto", nidem_func=None, methods=None, record_window="all", io_lock=None,
             nidem_cache_dir=None, vva_truth=None, vva_truth_date=None, vva_nodelay=None, vva_grid=None, vak_dir=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_run_end_to_end_on_a_fake_dutch_site(tmp_path, monkeypatch):
    table, S, keep_flat, px, z_stored, seen, calls = _fake_v14_site(tmp_path, monkeypatch, "escalda")
    holder = {"seconds": 0.25, "cell_id": np.arange(len(keep_flat), dtype=np.int64),
              "sheets": {"sheets": [{"sheet": "A.nc", "used": True}, {"sheet": "B.nc", "used": False}],
                         "n_sheets": 2, "n_sheets_with_year": 1, "cells_with_year": 60}}
    truth_px = np.where(np.arange(len(keep_flat)) < 50, np.linspace(-1, 1, len(keep_flat)), np.nan)

    def fake_units(keep, W, z_stored_):
        holder["in_load_site"] = True
        F = np.flatnonzero(np.isfinite(truth_px))                  # no product decides F
        return F, truth_px[F], {"truth": "Vaklodingen 2025 (fake)", "truth_px": 50, "truth_in_range": 50,
                                "units_in_F": int(len(F))}

    monkeypatch.setattr(v19, "prepare_truth", lambda site, cfg, tdef, args, gates: ("rtk_units", fake_units, holder))
    out = str(tmp_path / "out")
    v19.run(_run_args(out, "escalda"))
    st = json.load(open(os.path.join(out, "escalda_settings.json"), encoding="utf-8"))
    assert st["status"] == "complete" and "rtk_units" in seen and seen["cfg_truth"] == "rtk"
    assert v14.rtk_units.__name__ == "rtk_units" and v14.dutch_truth.__name__ == "dutch_truth"
    assert st["truth"]["design"] == v19.TRUTH_DESIGN and st["window_end"]["date"] == "2025-12-31"
    assert "survey year only" in st["window_end"]["rule"] and st["truth"]["counts"]["cells_in_F"] > 0
    assert st["truth"]["counts"]["units_F"] == 50                      # every truth pixel in range, whatever MAREA
    fs = st["scoring"]["F"]
    assert fs["definition"] == v19.F_RULE and fs["unit"] == "10 m pixel" and "stored MAREA" in fs[
        "v14_definition_replaced"] and st["compat"]["version"] == 3
    assert st["truth"]["counts"]["sheets_in_F"] == ["A.nc"] and st["v14_gates"]["stored_product_vs_published_row"][
        "applicable"] is False
    assert st["gates"]["record_window_marea_mb0_reproduces_published"]["ok"]
    assert st["gates"]["fixed_set_is_v14s"]["applicable"] is False and st["gates"]["nidem_hook"]["ok"]
    # one NIDEM call for every window and both pixel rules
    assert len(calls) == 1 and sorted(calls[0]) == sorted(f"{w}|{r}" for w in ("n51", "n50", "n40")
                                                          for r in ("common", "item"))
    assert st["host"] and st["timing"]["shared_site_parts_s"]["load_without_truth"] >= 0
    tim = json.load(open(os.path.join(out, "escalda_timing.json"), encoding="utf-8"))
    assert tim["site_shared_s"] >= 0 and set(tim["window_select_s"]) == {"n51", "n50", "n40"}
    df = pd.read_csv(os.path.join(out, "escalda.csv"))
    ok = df[df["status"] == v19.OK]
    assert set(ok["n_scenes"]) == {51, 50, 40} and set(df[df["n_scenes"] == 60]["status"]) == {v19.NOT_REACHABLE}
    assert (ok["seconds_estimator"] >= 0).all() and (ok["seconds_shared"] >= 0).all() and ok["host"].str.len().all()
    assert "51 screened scenes up to 2025-12-31" in df[df["n_scenes"] == 60]["reason"].iloc[0]
    assert set(st["timing_per_window"]) == {"51", "50", "40"}
    # E1 (common mask) is a headline column, E1i (ITEM's) a variant scored on the common set
    roles = {(r.method, r.variant): r.role for r in ok.itertuples()}
    assert roles[("E1", "main")] == "headline" and roles[("E1i", "main")] == "variant"
    # the tables of that run: the Dutch note, the truth, the time row; Villaviciosa pending (no run)
    v19.tables(argparse.Namespace(out=out, sites=None, ns="60,50,40", no_rescore=False))
    md = open(os.path.join(out, "tables", "table_n040.md"), encoding="utf-8").read()
    assert "Westerschelde (Vaklodingen 2025)" in md and "Villaviciosa (PNOA LiDAR) | pending truth" in md
    assert "time (s)" in md and "survey year only" in md and "| NIDEM | NIDEM (ITEM mask) |" in md
    # a rerun resumes every job (the same scenes, units and code)
    v19.run(_run_args(out, "escalda"))
    st = json.load(open(os.path.join(out, "escalda_settings.json"), encoding="utf-8"))
    assert set(st["progress"]["status_per_job"].values()) == {"resumed"} and len(calls) == 1


def test_run_end_to_end_villaviciosa_raster_truth_before_the_record(tmp_path, monkeypatch):
    """A PNOA flight date before every window: nothing is tabulated, the
    record (after the date) runs as the gate only, the RTK is never read."""
    from pyintertidal import rtk

    monkeypatch.setattr(rtk, "load_rtk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RTK read")))
    table, S, keep_flat, px, z_stored, seen, calls = _fake_v14_site(tmp_path, monkeypatch, "villaviciosa")
    holder = {"seconds": 0.5, "z_pre": z_stored}
    truth_raw = np.linspace(-1, 1, len(keep_flat))
    fn = v19.raster_units(truth_raw, -1.0, 1.0, {"truth": "PNOA (fake)", "pixel_m": 10.0}, holder)
    monkeypatch.setattr(v19, "prepare_truth", lambda site, cfg, tdef, args, gates: ("rtk_units", fn, holder))
    f = tmp_path / "dtm.tif"
    f.write_bytes(b"x")
    out = str(tmp_path / "out")
    date = str(S["dates"][20])
    v19.run(_run_args(out, "villaviciosa", vva_truth=str(f), vva_truth_date=date,
                      usable_json=str(tmp_path / "usable.json")))
    st = json.load(open(os.path.join(out, "villaviciosa_settings.json"), encoding="utf-8"))
    assert st["status"] == "complete" and "rtk_units" in seen and v14.rtk_units.__name__ == "rtk_units"
    assert st["truth"]["kind"] == "raster" and st["window_end"]["date"] == date
    assert st["window_end"]["screened_scenes_up_to_it"] == 21 and st["scoring"]["F"]["unit"] == "10 m pixel"
    assert st["truth"]["counts"]["units_F"] == len(keep_flat)              # the raster in range, no product
    assert set(st["progress"]["status_per_job"]) == {"r051__E0b", "r051__E0a"} and calls == []
    assert st["gates"]["record_window_marea_mb0_reproduces_published"]["ok"]
    df = pd.read_csv(os.path.join(out, "villaviciosa.csv"))
    assert (df["status"] == v19.NOT_REACHABLE).all()
    r40 = df[df["n_scenes"] == 40]["reason"].iloc[0]
    assert f"S holds 21 screened scenes up to {date}: 19 missing" in r40
    v19.tables(argparse.Namespace(out=out, sites=["villaviciosa"], ns="60,50,40", no_rescore=False))
    md = open(os.path.join(out, "tables", "table_n040.md"), encoding="utf-8").read()
    assert f"not reachable: 21 screened images up to {date} (19 missing)" in md
    assert f"PNOA LiDAR, flight {date}" in md and "RTK" not in md.replace("never the RTK", "")
    md = open(os.path.join(out, "tables", "table_record.md"), encoding="utf-8").read()
    assert "ends after the truth date" in md
    # the curves: a site whose every window is not reachable says so (not 'no results yet')
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dfr = v19.read_results(out, ["villaviciosa"])
    for lang, word in (("en", f"not reachable: 21 screened images up to {date}"), ("es", "no alcanzable: 21 imágenes")):
        fig = v19.curves_figure(dfr, ["villaviciosa"], lang)
        texts = " ".join(t.get_text().replace("\n", " ") for ax in fig.axes for t in ax.texts)
        assert word in texts and "\\'" not in texts
        plt.close(fig)


# ── the 2026-10-26 review fixes ─────────────────────────────────────────────

def test_dutch_units_F_is_decided_by_no_product(monkeypatch):
    """The Dutch F: every keep pixel with a truth in range, the stored MAREA
    only recorded (v14's 'rtk' branch takes the units as given)."""
    truth = np.array([0.1, np.nan, 0.5, -0.2, 0.3])

    def fake_year(site, cfg, keep, H, W, z_plain, year, truth_dir, holder):
        holder.update(seconds=0.5, cell_id=np.arange(len(keep)))
        return truth, {"truth": "fake", "truth_px": 4, "truth_in_range": 4}

    monkeypatch.setattr(v19, "dutch_truth_year", fake_year)
    holder = {}
    fn = v19.dutch_units("escalda", {}, 3, np.array([-1.0, 1.0]), 2025, None, holder)
    F, t, info = fn(np.arange(5), 10, np.array([1.0, 1.0, np.nan, np.nan, 2.0]))
    assert F.tolist() == [0, 2, 3, 4] and t.tolist() == [0.1, 0.5, -0.2, 0.3]
    assert info["units_in_F"] == 4 and info["units_in_F_with_stored_marea"] == 2 and info["F_rule"] == v19.F_RULE
    assert holder["in_load_site"] and holder["seconds"] == 0.5


def test_unavailable_methods_are_pending_not_not_implemented(tmp_path):
    """A registry method whose code exists but did not run on that host (DEA
    without its side interpreter) is 'pending' with the reason, never 'not
    implemented'; the tables say so in a footnote."""
    out = str(tmp_path)
    _fake_site(out, "escalda", "vaklodingen", (60, 50, 40), seed=2)
    st = json.load(open(os.path.join(out, "escalda_settings.json"), encoding="utf-8"))
    st.update(host="laptop", methods={"order": ["E0b", "E0a", "E4", "E1", "E1i"], "registry": {
        "methods": ["E4", "E5b"], "registry_skipped": {"E2": "DEA Intertidal 2.5.3: no side interpreter (x)"},
        "registry_errors": {}}})
    v14.write_json(os.path.join(out, "escalda_settings.json"), st)
    why = v19.unavailable_methods(out, "escalda")
    assert "no side interpreter" in why["E2"] and "laptop" in why["E2"] and "--methods" in why["E5b"]
    store = v19.JobStore(os.path.join(out, "escalda_jobs"))
    _, S, _, _ = synthetic_table()
    ws = windows_for(S, ns=(40,))
    rng = np.random.default_rng(3)
    truth = rng.uniform(-1, 1, 30)
    rows = v19.build_rows("escalda", "vaklodingen", truth, np.arange(30), ws, store, unavailable=why)
    df = pd.DataFrame(rows)
    e2 = df[(df["method"] == "E2") & (df["n_scenes"] == 40)].iloc[0]
    assert e2["status"] == v19.PENDING and "no side interpreter" in e2["reason"]
    assert "E2" not in v19.not_implemented(df) and "E3" in v19.not_implemented(df)
    rows = v19.build_rows("escalda", "vaklodingen", truth, np.arange(30), ws, store)
    assert "E2" in v19.not_implemented(pd.DataFrame(rows))                    # no reason given: not implemented
    # the tables: a pending column and its footnote
    df = v19.read_results(out, ["escalda"])
    df = df[df["method"] != "E2"]
    df = pd.concat([df, df[df["method"] == "E4"].assign(method="E2", label="DEA Intertidal", status=v19.PENDING,
                                                         variant="", role="headline", reason=why["E2"])],
                   ignore_index=True)
    cols = v19.table_columns(df)
    assert ("E2", "main") in cols
    pend = v19.pending_methods(df, cols)
    assert "no side interpreter" in pend["E2"]
    md = v19.table_markdown(v19.table_data(df, ["escalda"], 40, cols), cols, 40, {}, v19.not_implemented(df), (),
                            None, pend)
    note = next(ln for ln in md.splitlines() if ln.startswith("Pending (code present, not run): "))
    assert "E2 DEA Intertidal (not run on laptop" in note
    tex = v19.table_latex(v19.table_data(df, ["escalda"], 40, cols), cols, 40, {}, [], (), None, pend)
    assert "Pendientes (c\\'odigo presente, sin ejecutar): " in tex and "DEA Intertidal (not run on laptop" in tex


def test_sensitivity_and_marea_rule_columns_follow_their_roles(tmp_path):
    """NIDEM with ITEM's mask (E1i) sits after NIDEM, is scored on the common
    set without defining it and is never 'best'; MAREA's other rules follow
    MB_HEADLINE; both are drawn dashed."""
    _, S, _, _ = synthetic_table()
    ws = windows_for(S, ns=(40,))
    w = [x for x in ws if x["n"] == 40][0]
    rng = np.random.default_rng(9)
    n = 60
    truth = rng.uniform(-1, 1, n)
    store = v19.JobStore(str(tmp_path / "jobs"))
    zm = truth + rng.normal(0, 0.2, n)
    _job(store, w, "E0b", v19.OK, v19.VARIANTS["E0b"], {"z__mb0": zm, "z__mb3": zm}, label="MAREA")
    z1 = truth + rng.normal(0, 0.15, n)
    z1[:6] = np.nan
    _job(store, w, "E1", v19.OK, v19.NIDEM_VARIANTS["E1"], {"z__main": z1}, label="NIDEM")
    zi = truth + rng.normal(0, 0.01, n)                                        # the best numbers, a sensitivity
    zi[50:] = np.nan
    _job(store, w, "E1i", v19.OK, v19.NIDEM_VARIANTS["E1i"], {"z__main": zi}, label="NIDEM, ITEM")
    rows = v19.build_rows("x", "vaklodingen", truth, np.arange(n), ws, store)
    df = pd.DataFrame(rows)
    common = df[(df["status"] == v19.OK) & (df["scoring"] == "common") & (df["n_scenes"] == 40)]
    assert set(common["n_common"]) == {54}                                     # E1i (NaN at 50..59) does not define it
    ei = common[common["method"] == "E1i"].iloc[0]
    assert ei["role"] == "variant" and ei["n_units"] == 44 and ei["coverage"] == pytest.approx(44 / 54)
    for c in v19.CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df.fillna({"notes": "", "reason": "", "variant": "", "role": ""})
    cols = v19.table_columns(df)
    assert cols.index(("E1i", "main")) == cols.index(("E1", "main")) + 1 and cols[-1] == ("E0b", "mb3")
    assert cols[0] == ("E0b", v19.HEAD_VAR) and v19.is_marea_variant("E0b", "mb3")
    assert not v19.is_marea_variant("E0b", v19.HEAD_VAR) and v19.is_side_column("E1i", "main")
    assert v19.VARIANTS["E0b"][v19.HEAD_VAR]["role"] == "headline"
    assert all(v19.VARIANTS["E0b"][f"mb{mb}"]["role"] == "variant" for mb in v19.MB_OTHERS)
    r = v19.table_data(df, ["x"], 40, cols)[0]
    assert r["best"] in (("E0b", "mb0"), ("E1", "main"))                       # never the sensitivity
    series = v19.curve_series(df[df["site"] == "x"], "common")
    assert ("E1i", "main") in series and ("E0b", "mb3") in series and ("E0a", "mb0") not in series
    assert v19.style_for("E1i", "main")["mfc"] == "white" and v19.style_for("E0b", "mb3")["mfc"] == "white"
    assert sorted(series, key=v19._series_order)[:2] == [("E0b", "mb0"), ("E0b", "mb3")]
    lab = v19.method_label("E1i", "main", "es", "x")
    assert "ITEM" in lab and v19.method_label("E0b", "mb3", "en", "x").startswith("MAREA, quality rule mb 3")


def test_level_series_cache_is_spot_checked_against_the_boundary(tmp_path):
    table, S, keep_flat, px = synthetic_table(T_years=(2025,), per_year=30)
    ctx = make_ctx(tmp_path, table, S, keep_flat, px)
    ws = v19.make_windows(S["dates"], S["h0"], (20,), None)
    p = str(tmp_path / "ls.npz")
    a = FakeBoundary(amp=1.4)
    ls, info = v19.level_series_for(ctx, a, ws, p)
    assert not info["reused"]
    ls, info = v19.level_series_for(ctx, a, ws, p)
    assert info["reused"] and info["spot_check"]["ok"]
    b = FakeBoundary(amp=1.1)                                                  # the same name, other levels
    ls, info = v19.level_series_for(ctx, b, ws, p)
    assert not info["reused"] and np.allclose(ls.h[:5], b.levels(ls.t[:5]))


def test_code_fingerprint_covers_the_sibling_modules():
    from experiments.fixedcfg import dea_intertidal, elev_coastxray, elev_freq
    from experiments.fixedcfg import methods as M

    deps = {os.path.basename(f) for f in v19.sibling_imports(elev_coastxray.__file__)}
    assert {"elev_base.py", "level_series.py"} <= deps
    assert "dea_worker.py" in {os.path.basename(f) for f in v19.sibling_imports(dea_intertidal.__file__)}
    assert M.METHODS["E5a"][1].__module__ == elev_freq.__name__
    fp = v19.code_fingerprint(M.METHODS["E5a"][1], "|keep")
    assert "+" in fp.split(":")[1] and fp.endswith("|keep")


def test_tag_groups_share_files_safely(tmp_path):
    """--tag groups of one site: per-process temp names; the tables read any
    settings file of the site when all hold the same compat; the timing files
    are merged."""
    out = str(tmp_path)
    p = os.path.join(out, "x.csv")
    v19.atomic_text(p, "a\n")
    v19.write_json(os.path.join(out, "y.json"), {"v": float("nan")})
    assert open(p, encoding="utf-8").read() == "a\n" and json.load(open(os.path.join(out, "y.json")))["v"] is None
    assert not [f for f in os.listdir(out) if ".tmp" in f]
    assert v19.site_paths(out, "s", "g1")["timing"].endswith("s_timing__g1.json")
    t = {"design": v19.TRUTH_DESIGN, "status": "ready"}
    v14.write_json(os.path.join(out, "s_settings__g1.json"), {"compat": {"v": 1}, "truth": t})
    v14.write_json(os.path.join(out, "s_settings__g2.json"), {"compat": {"v": 1}, "truth": t})
    assert v19.site_truth(out, "s") == t                                        # only tag groups ran
    v14.write_json(os.path.join(out, "s_settings__g3.json"), {"compat": {"v": 2}, "truth": t})
    assert v19.site_settings(out, "s") is None                                  # other settings: never mixed
    v14.write_json(os.path.join(out, "s_timing__g1.json"), {"site_shared_s": 5.0, "window_select_s": {"n40": 0.3}})
    v14.write_json(os.path.join(out, "s_timing__g2.json"), {"site_shared_s": 4.0, "window_select_s": {"n40": 0.5,
                                                                                                     "n50": 0.1}})
    tm = v19.read_timing(out, "s")
    assert tm["site_shared_s"] == 4.0 and tm["window_select_s"] == {"n40": 0.3, "n50": 0.1} and tm["files"] == 2


def test_dea_temp_root_inside_the_repository_is_caught(monkeypatch, tmp_path):
    from experiments.fixedcfg import dea_intertidal as dea

    monkeypatch.setenv(dea.ENV_TMP, os.path.join(ROOT, ".tmp"))
    why = v19.dea_tmp_problem()
    assert why and "inside the repository" in why and dea.ENV_TMP in why
    monkeypatch.setenv(dea.ENV_TMP, str(tmp_path))
    assert v19.dea_tmp_problem() is None


def test_estimator_seconds_of_adapters_marea_and_hook_errors(tmp_path):
    """The adapter's own estimator seconds win over the call's (DEA: its
    per-pixel steps, not its start-up, imports or I/O); else the reported
    I/O and imports are taken off; MAREA's are v14's job seconds; a hook
    that raises SystemExit gives an error job, never stops the run."""
    table, S, keep_flat, px = synthetic_table()

    def with_own(obs, cfg):
        return {"z": np.zeros(len(obs.keep), np.float32), "meta": {
            "seconds_estimator": 0.125, "max_workers": 4,
            "seconds": {"write_input": 1.0, "read_output": 1.0, "worker_detail": {"import": 2.0}}}}

    def with_io(obs, cfg):
        import time as _t
        _t.sleep(0.05)
        return {"z": np.zeros(len(obs.keep), np.float32), "meta": {
            "seconds": {"write_input": 0.01, "read_output": 0.01, "worker_detail": {"import": 0.02}}}}

    card = {"reference": "r", "doi": "d", "fidelity": "F1", "deviations": [], "needs_full_frame": False}
    reg = {"EXA": {"label": "own", "run": with_own, "card": card, "variant": False},
           "EXB": {"label": "io", "run": with_io, "card": card, "variant": False}}

    def raising(site, dates, pixel_rule, units_keep):
        raise SystemExit("[synthetic] gauge cache missing")

    ctx = make_ctx(tmp_path, table, S, keep_flat, px, registry=reg, nidem_fn=raising,
                   nidem_info={"function": "fake.raising", "file": {"sha256": "2" * 64}})
    w = [x for x in windows_for(S) if x["n"] == 40][0]
    m, _ = v19.run_one(ctx, w, "EXA")
    assert m["seconds_estimator"] == 0.125 and m["info"]["timing"]["processes"] == 4
    assert m["info"]["timing"]["adapter_estimator_s"] == 0.125
    m, _ = v19.run_one(ctx, w, "EXB")
    t = m["info"]["timing"]
    assert t["adapter_io_s"] == pytest.approx(0.04) and m["seconds_estimator"] == pytest.approx(t["call_s"] - 0.04,
                                                                                                abs=1e-3)
    m, _ = v19.run_one(ctx, w, "E1")
    assert m["status"] == v19.ERROR and "gauge cache missing" in m["reason"] and not m["cacheable"]
    z, meta = v19.marea_job(table, v19.window_rows(w), SEED)
    assert 0 <= meta["seconds_estimator"] <= meta["seconds_call_with_v14_scoring"] + 1e-3


def test_record_caption_counts_come_from_the_rows(tmp_path):
    out = str(tmp_path)
    _fake_site(out, "escalda", "vaklodingen", (60, 50, 40), seed=2)
    df = v19.read_results(out, ["escalda"])
    cols = v19.table_columns(df)
    rows = v19.table_data(df, ["villaviciosa", "escalda"], "record", cols)
    tex = v19.table_latex(rows, cols, "record", {}, [], (), None)
    n = [r["n"] for r in rows if r["state"] == "ok"]
    assert f"({' / '.join(str(v) for v in n)} im\\'agenes)" in tex and "97 / 68" not in tex


# ── the Wadden survey-year groups (the user's choice, 2026-10-26) ──────────

def test_truth_definition_of_the_wadden_groups():
    t = v19.truth_definition("wadden", argparse.Namespace())
    assert [g["name"] for g in t["groups"]] == ["wadden-2025", "wadden-2024", "wadden-2023"]
    assert [g["end"] for g in t["groups"]] == ["2025-12-31", "2024-12-31", "2023-12-31"]
    assert t["end"] == "2025-12-31" and t["year"] == 2025 and t["years"] == [2025, 2024, 2023]
    assert "survey year only" in t["end_rule"] and "pooled" in t["grouping"]
    for s in ("escalda", "ems"):                                        # the other Dutch sites: 2025 only
        d = v19.truth_definition(s, argparse.Namespace())
        assert "groups" not in d and d["end"] == "2025-12-31"
    assert v19.split_row("wadden-2024") == ("wadden", 2024) and v19.split_row("wadden") == ("wadden", None)
    assert v19.group_row("wadden", 2023) == "wadden-2023"


def test_dutch_truth_groups_take_each_pixel_most_recent_survey(tmp_path, monkeypatch):
    """Per keep pixel the most recent survey year 2023-2025 of its cell (v1's
    sampling within each year) and its value; the range filter alike in
    every group; a most recent survey out of range never falls back to an
    older one."""
    from experiments import v7_waterline_contour as v7

    a23 = np.full((3, 4), 0.1)
    a25 = np.arange(12.0).reshape(3, 4) / 100.0
    a25[0, 0] = np.nan                                       # cell A(0,0): no 2025 value -> its 2023 survey
    a25[1, 1] = 2.0                                          # cell A(1,1): 2025 out of range -> out of F, group 2025
    _sheet(tmp_path / "A.nc", 1000.0, 5000.0, 4, 3, ["2023-01-01", "2025-01-01"], [a23, a25])
    b24 = np.full((3, 4), 0.3)
    _sheet(tmp_path / "B.nc", 1080.0, 5000.0, 4, 3, ["2024-01-01"], [b24])
    orig = v19.truth_grid_year
    monkeypatch.setattr(v19, "truth_grid_year",
                        lambda site, year, tdir: orig(site, year, str(tmp_path), sheets=["A.nc", "B.nc"]))
    grid = {"H": 2, "W": 5, "xs": np.array([1000.0, 1060.0, 1021.0, 1100.0, 1500.0]),
            "ys": np.array([5000.0, 4980.0]), "crs": "EPSG:28992"}
    monkeypatch.setattr(v7, "cube_grid", lambda cube: grid)
    keep = np.array([0, 1, 7, 3, 4], np.int64)                # A(0,0) A(0,3) A(1,1) B(0,1) and a point on no sheet
    holder = {}
    truth, info = v19.dutch_truth_groups("wadden", {"cube": "x.nc"}, keep, 2, 5, np.array([0.0, 0.3]),
                                         (2023, 2025, 2024), None, holder)
    assert holder["group_year"].tolist() == [2023, 2025, 2025, 2024, 0]
    assert np.array_equal(truth, [0.1, a25[0, 3], np.nan, 0.3, np.nan], equal_nan=True)
    assert info["survey_years"] == [2025, 2024, 2023] and info["truth_px"] == 4 and info["truth_in_range"] == 3
    assert info["groups"]["2025"]["truth_px"] == 2 and info["groups"]["2025"]["truth_in_range"] == 1
    assert info["groups"]["2024"]["cells_in_range"] == 1 and info["groups"]["2023"]["n_sheets_with_year"] == 1
    assert (holder["groups"][2024]["cell_id"] >= 0).tolist() == [False, False, False, True, False]
    # through dutch_units (a tuple of years): F = the finite truth, the group year of every keep pixel
    holder = {}
    fn = v19.dutch_units("wadden", {"cube": "x.nc"}, 2, np.array([0.0, 0.3]), (2025, 2024, 2023), None, holder)
    F, t, uinfo = fn(keep, 5, np.full(5, np.nan))
    assert F.tolist() == [0, 1, 3] and holder["in_load_site"] and uinfo["F_rule"] == v19.F_RULE
    assert holder["group_year"][F].tolist() == [2023, 2025, 2024]


def _groups_case(tmp_path, ns=(40, 30, 15)):
    """Survey-year group windows on a synthetic site (17 screened scenes a
    year 2023-2025, the record = all 51) and a job store with MAREA, a
    competitor and a pending NIDEM on every tabulated window, each group's
    estimate offset by its own constant."""
    _, S, _, _ = synthetic_table()
    tdef = v19.truth_definition("wadden", argparse.Namespace())
    ws = v19.group_windows(S["dates"], S["h0"], ns, 51, None, tdef["groups"])
    rng = np.random.default_rng(12)
    n = 90
    truth = rng.uniform(-1, 1, n)
    ug = np.repeat([2025, 2024, 2023], 30)
    store = v19.JobStore(str(tmp_path / "wadden_jobs"))
    off = {2025: 0.01, 2024: 0.02, 2023: 0.03}

    def gjob(w, mid, status, variants, arrays, seconds_estimator):
        meta = {"key": v19.wkey(w, mid), "site": "wadden", "n": w["n"], "window": w["label"],
                "is_record": w["is_record"], "method": mid, "label": mid, "status": status, "reason": "r",
                "source": "s", "fidelity": "F3", "variants": variants, "scenes_sha1": w["scenes_sha1"],
                "units_sha1": "units0", "fingerprint": "fp", "cacheable": status == v19.OK, "seconds": 1.0,
                "info": {}, "seconds_estimator": seconds_estimator, "host": "testhost", "threads": "OMP=2"}
        store.save(meta["key"], meta, arrays)

    for w in v19.table_windows(ws):                          # every job under its group's own key
        z = truth + off[w["group"]]
        gjob(w, "E0b", v19.OK, v19.VARIANTS["E0b"], {"z__mb0": z + rng.normal(0, 0.05, n), "z__mb3": z}, 1.0)
        gjob(w, "E4", v19.OK, {"main": {"role": "headline", "label": "LogiTide2DEM"}}, {"z__main": z}, 0.5)
        gjob(w, "E1", v19.PENDING, {}, None, None)
    return S, tdef, ws, truth, ug, store, off


def test_group_windows_rows_and_the_pooled_row(tmp_path):
    S, tdef, ws, truth, ug, store, off = _groups_case(tmp_path)
    by = {w["label"]: w for w in ws}
    # each group its own truth date and windows; the record is the 2025 group's gate window only
    assert by["y2025_n51"]["is_record"] and by["y2025_n51"]["reachable"] and v19.wkey(by["y2025_n51"], "E0b") == \
        "y2025_n051__E0b"
    assert not by["y2024_rec51"]["reachable"] and "ends after the truth date 2024-12-31" in by["y2024_rec51"]["reason"]
    assert [w["label"] for w in v19.table_windows(ws) if w["group"] == 2024] == ["y2024_n30", "y2024_n15"]
    assert [w["label"] for w in v19.table_windows(ws) if w["group"] == 2023] == ["y2023_n15"]
    assert all(w["stop"] == 34 for w in v19.table_windows(ws) if w["group"] == 2024)
    assert "S holds 17 screened scenes up to 2023-12-31: 13 missing" in by["y2023_n30"]["reason"]
    nest = v19.check_nested(ws)
    assert nest["ok"] and set(nest["groups"]) == {"2025", "2024", "2023"}
    plan = v19.job_plan(ws, ["E0b", "E4"])
    assert plan[0] == ("y2025_n51", "E0b") and ("y2024_rec51", "E0b") not in plan and ("y2024_n30", "E4") in plan
    assert "y2024_n030__E4" in store.keys() and "y2023_n015__E0b" in store.keys()
    timing = {"site_shared_s": 10.0, "window_select_s": {w["label"]: 0.5 for w in ws}}
    rows = v19.build_rows("wadden", "vaklodingen", truth, np.arange(len(truth)), ws, store, timing=timing,
                          unit_group=ug)
    df = pd.DataFrame(rows)
    ok = df[(df["status"] == v19.OK) & (df["scoring"] == "common")]
    assert set(df["site"]) == {"wadden-2025", "wadden-2024", "wadden-2023", "wadden"}
    # a group row is scored on its group's units only, with its own common set
    g24 = ok[(ok["site"] == "wadden-2024") & (ok["n_scenes"] == 30) & (ok["method"] == "E4")].iloc[0]
    assert g24["n_set"] == 30 and g24["n_units"] == 30
    assert g24["bias"] == pytest.approx(0.02) and g24["rmse_total"] == pytest.approx(0.02)
    # unreachable groups: 'not reachable' rows with the count
    r23 = df[(df["site"] == "wadden-2023") & (df["n_scenes"] == 30)]
    assert (r23["status"] == v19.NOT_REACHABLE).all() and "17 screened scenes up to 2023-12-31" in r23["reason"].iloc[0]
    # the pooled row: only at the n where every group's window is reachable, each pixel with its own group's estimate
    pooled = df[df["site"] == "wadden"]
    assert set(pooled["n_scenes"]) == {15}
    p4 = pooled[(pooled["method"] == "E4") & (pooled["scoring"] == "own")].iloc[0]
    zp = truth + np.select([ug == 2025, ug == 2024], [0.01, 0.02], 0.03)
    s = v19.score_values("vaklodingen", truth, zp, np.arange(len(truth)))
    assert p4["status"] == v19.OK and p4["n_units"] == 90 and p4["rmse_total"] == pytest.approx(s["rmse_total"])
    assert p4["seconds_estimator"] == pytest.approx(1.5) and "pooled" in p4["notes"]
    assert p4["seconds_shared"] == pytest.approx(10.0 + 3 * 0.5)
    e1 = pooled[pooled["method"] == "E1"].iloc[0]
    assert e1["status"] == v19.PENDING and e1["reason"].startswith("wadden-2025:")
    # a group without any unit of F (here 2023) never holds the pooled row back
    ug2 = np.repeat([2025, 2024], 45)
    d2 = pd.DataFrame(v19.build_rows("wadden", "vaklodingen", truth, np.arange(len(truth)), ws, store, unit_group=ug2))
    assert set(d2[d2["site"] == "wadden"]["n_scenes"]) == {30, 15}
    assert set(d2[(d2["site"] == "wadden-2023") & (d2["status"] == v19.OK)]["n_set"]) == {0}
    # the CSV, then the tables: one row per group (sheets and cells), the pooled row only at n = 15
    out = str(tmp_path)
    v19.write_csv(os.path.join(out, "wadden.csv"), rows)
    counts = {"units_F": 90, "units_scored": 90, "groups": {
        f"wadden-{y}": {"survey_year": y, "date": f"{y}-12-31", "units_F": 30, "units_scored": 30,
                        "cells_in_F": 10 + y % 10, "sheets_in_F": [f"S{y}.nc"], "sheets_with_year": 1,
                        "sheets_total": 3} for y in (2025, 2024, 2023)}}
    v14.write_json(os.path.join(out, "wadden_settings.json"), {"site": "wadden", "truth": dict(tdef, counts=counts)})
    v19.tables(argparse.Namespace(out=out, sites=["wadden"], ns="40,30,15", no_rescore=True))
    md15 = open(os.path.join(out, "tables", "table_n015.md"), encoding="utf-8").read()
    md30 = open(os.path.join(out, "tables", "table_n030.md"), encoding="utf-8").read()
    for y, c in ((2025, 15), (2024, 14), (2023, 13)):
        assert f"| Wadden (Vlie), wadden-{y} (Vaklodingen {y}: 1 sheet(s), {c} cells, truth date {y}-12-31) |" in md15
    assert "| Wadden (Vlie), pooled (" in md15 and "| Wadden (Vlie), pooled (" not in md30
    assert "not reachable: 17 screened images up to 2023-12-31 (13 missing)" in md30
    assert "survey-year groups" in md30 and "pools them only at the n where every group's window" in md30
    tex = open(os.path.join(out, "tables", "table_n015_es.tex"), encoding="utf-8").read()
    assert "wadden-2024" in tex and "Wadden (Vlie), conjunto" in tex and "grupos por a\\~no de levantamiento" in tex
    rec = open(os.path.join(out, "tables", "table_record.md"), encoding="utf-8").read()
    assert "wadden-2025" in rec and "ends after the truth date 2024-12-31" in rec and "Wadden (Vlie), pooled" not in rec
    # the curves: a panel per group row and the pooled one
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dfr = v19.read_results(out, ["wadden"])
    panels = v19.table_sites(dfr, ["wadden"])
    assert panels == ["wadden-2025", "wadden-2024", "wadden-2023", "wadden"]
    fig = v19.curves_figure(dfr, panels, "en", meta={"truth": {"wadden": dict(tdef, counts=counts)}})
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "Wadden (Vlie), wadden-2024" in texts and "Wadden (Vlie), pooled" in texts
    assert "Vaklodingen 2024: 1 sheet(s), 14 cells" in texts
    plt.close(fig)


def test_run_end_to_end_wadden_survey_year_groups(tmp_path, monkeypatch):
    """run() at the Wadden: per keep pixel its survey year (the fake truth),
    the group windows (their own truth dates), the record gate on the 2025
    group, one NIDEM call for every group window, group-keyed jobs, the
    pooled row only at n = 15, everything declared, and a resume."""
    table, S, keep_flat, px, z_stored, seen, calls = _fake_v14_site(tmp_path, monkeypatch, "wadden")
    Pk = len(keep_flat)
    gy = np.repeat([2025, 2024, 2023], Pk // 3)
    holder = {"seconds": 0.25, "group_year": gy, "groups": {
        y: {"cell_id": np.where(gy == y, np.arange(Pk), -1),
            "sheets": {"sheets": [{"sheet": f"S{y}.nc", "used": True}, {"sheet": "X.nc", "used": False}],
                       "n_sheets": 2, "n_sheets_with_year": 1, "cells_with_year": Pk // 3}}
        for y in (2025, 2024, 2023)}}
    truth_px = np.linspace(-1, 1, Pk)

    def fake_units(keep, W, z_stored_):
        holder["in_load_site"] = True
        F = np.flatnonzero(np.isfinite(truth_px))
        return F, truth_px[F], {"truth": "Vaklodingen by survey year (fake)", "truth_px": Pk, "truth_in_range": Pk,
                                "units_in_F": int(len(F))}

    monkeypatch.setattr(v19, "prepare_truth", lambda site, cfg, tdef, args, gates: ("rtk_units", fake_units, holder))
    out = str(tmp_path / "out")
    v19.run(_run_args(out, "wadden", ns="40,30,15"))
    st = json.load(open(os.path.join(out, "wadden_settings.json"), encoding="utf-8"))
    assert st["status"] == "complete" and st["gates"]["record_window_marea_mb0_reproduces_published"]["ok"]
    tg = st["truth_groups"]
    assert [g["name"] for g in tg["groups"]] == ["wadden-2025", "wadden-2024", "wadden-2023"]
    assert [g["screened_scenes_up_to_it"] for g in tg["groups"]] == [51, 34, 17] and tg["pooled_ns"] == [15]
    assert tg["groups"][1]["tabulated"] == ["y2024_n30", "y2024_n15"] and tg["groups"][1]["counts"]["units_F"] == 20
    assert st["compat"]["truth"]["groups"] == [2025, 2024, 2023] and "unit_group_sha1" in st["compat"]
    assert st["gates"]["windows_nested"]["ok"] and any("survey-year groups" in c for c in st["caveats"])
    jobs = set(st["progress"]["status_per_job"])
    assert "y2025_n051__E0b" in jobs and "y2024_n030__E1i" in jobs and "y2023_n015__E0a" in jobs
    assert not [k for k in jobs if not k.startswith("y")] and not [k for k in jobs if k.startswith("y2023_n030")]
    assert len(calls) == 1 and "y2024_n15|common" in calls[0] and "y2025_n51|item" in calls[0]
    u = v19.load_units(os.path.join(out, "wadden_units.npz"))
    assert sorted(set(u["unit_group"].tolist())) == [2023, 2024, 2025]
    m = json.load(open(os.path.join(out, "wadden_jobs", "y2024_n030__E0b.json"), encoding="utf-8"))
    assert m["group"] == 2024 and m["row_site"] == "wadden-2024" and "loadavg" in m
    df = pd.read_csv(os.path.join(out, "wadden.csv"))
    assert set(df["site"]) == {"wadden-2025", "wadden-2024", "wadden-2023", "wadden"}
    assert set(df[df["site"] == "wadden"]["n_scenes"]) == {15}
    ok = df[(df["status"] == v19.OK) & (df["scoring"] == "own")]
    assert set(ok[ok["site"] == "wadden-2024"]["n_set"]) == {20}
    assert set(ok[ok["site"] == "wadden"]["n_set"]) == {60}
    assert set(st["timing_per_window"]) >= {"wadden-2025|40", "wadden-2024|30", "15"}
    v19.tables(argparse.Namespace(out=out, sites=["wadden"], ns="40,30,15", no_rescore=False))
    md = open(os.path.join(out, "tables", "table_n015.md"), encoding="utf-8").read()
    assert "Wadden (Vlie), wadden-2023 (Vaklodingen 2023: 1 sheet(s), 20 cells, truth date 2023-12-31)" in md
    assert "Wadden (Vlie), pooled" in md
    v19.run(_run_args(out, "wadden", ns="40,30,15"))
    st = json.load(open(os.path.join(out, "wadden_settings.json"), encoding="utf-8"))
    assert set(st["progress"]["status_per_job"].values()) == {"resumed"} and len(calls) == 1


def test_dea_temp_root_option_and_the_cfg(tmp_path, monkeypatch):
    """--dea-tmp: a temp root inside the repository is refused before any
    job; one outside goes to DEA's cfg['dea_tmp']."""
    from experiments.fixedcfg import dea_intertidal as dea

    monkeypatch.delenv(dea.ENV_TMP, raising=False)
    assert "inside the repository" in v19.dea_tmp_problem({"dea_tmp": os.path.join(ROOT, "results", "x")})
    assert v19.dea_tmp_problem({"dea_tmp": str(tmp_path)}) is None
    table, S, keep_flat, px = synthetic_table()
    ctx = make_ctx(tmp_path, table, S, keep_flat, px)
    w = windows_for(S)[0]
    assert "dea_tmp" not in v19.method_cfg(ctx, w, "E2", "keep")
    ctx.dea_tmp = str(tmp_path)
    assert v19.method_cfg(ctx, w, "E2", "keep")["dea_tmp"] == str(tmp_path)
    a = v19.build_parser().parse_args(["run", "escalda", "--dea-tmp", str(tmp_path)])
    assert a.dea_tmp == str(tmp_path)


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
