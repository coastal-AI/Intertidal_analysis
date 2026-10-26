# -*- coding: utf-8 -*-
"""experiments/v16_window_curves.py on tiny synthetic data: the windows, the
window job (= v14's refit job on exactly the window's scenes), the W3 gate
against a v14 CSV, the extended record and arrays, resumability, the NIDEM
hook, the figure in both languages, and the RTK/output guards.

Run:  python -m pytest -q tests/test_window_curves.py
"""
import argparse
import ast
import math
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
from experiments import v16_window_curves as v16  # noqa: E402
from pyintertidal import marea  # noqa: E402

M2_H = 12.4206012
MBS = [0, 3]


class FakeBoundary:
    """A boundary like PyTMDBoundary: levels returned sorted by time; an
    optional instant without a finite level."""

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


def make_times(years, per_year, rng):
    """per_year chronological scenes in each year, near 11:00 UTC."""
    dates, times = [], []
    for y in years:
        for d in np.sort(rng.choice(np.arange(0, 360), per_year, replace=False)):
            ts = (pd.Timestamp(f"{y}-01-01 11:00") + pd.Timedelta(days=int(d))
                  + pd.Timedelta(seconds=int(rng.integers(0, 1800))))     # whole seconds, as overpass times
            times.append(ts)
            dates.append(ts.strftime("%Y-%m-%d"))
    return np.array(dates), pd.DatetimeIndex(times)


def synthetic_site(years=(2023, 2024, 2025), per_year=16, P=36, Pm=40, seed=0, clear_p=0.85, lag1=30):
    """A site whose pixels flood when the boundary level, read at their lag,
    tops them; two lag-fit bands, the inner one lagging ``lag1`` minutes."""
    rng = np.random.default_rng(seed)
    dates, times = make_times(years, per_year, rng)
    T = len(dates)
    lev = v14.LevelTable(FakeBoundary(), times)
    lev.ensure(sorted({0, *(int(t) for t in marea.BANK_TAUS), *range(0, 61)}))
    band = np.repeat([0, 1], Pm)
    zm = rng.uniform(-1.1, 1.1, 2 * Pm)
    H = np.where(band[None, :] == 0, lev[0][:, None], lev[int(lag1)][:, None])
    Cm = rng.random((T, 2 * Pm)) < clear_p
    Ym = np.where(Cm, np.where(H > zm[None, :], 0.4, -0.3) + rng.normal(0, 0.05, (T, 2 * Pm)), 0.0)
    fit_kw = dict(sigma0=0.2, sg_grid=(0.06, 0.1, 0.15, 0.22, 0.32, 0.45), tau_bounds=marea.TAU_BOUNDS,
                  z_points=60, max_px_band=1200, monotone=True)
    m2a = dict(Y=Ym.astype(np.float32), C=Cm, band=band, centers=np.array([1.5, 4.5]), nb=2, band_km=3.0,
               fit_kw=fit_kw, seed=20260819)
    s_km = np.linspace(0.3, 6.0, P)
    tau_px = np.round(np.interp(s_km, [0.0, 1.5, 4.5], [0.0, 0.0, float(lag1)]))
    z = rng.uniform(-1.0, 1.0, P)
    Hp = np.stack([lev[t] for t in tau_px], axis=1)
    C = rng.random((T, P)) < clear_p
    Y0 = np.where(C, np.where(Hp > z[None, :], 0.45, -0.3) + rng.normal(0, 0.05, (T, P)), 0.0)
    truth = z + rng.normal(0, 0.03, P)
    sd = v14.SiteData("synthetic", Y0.astype(np.float32), C, lev, {"all": np.arange(T)}, np.arange(P), truth,
                      "vaklodingen", np.zeros(P), s_km=s_km, m2a=m2a)
    return sd, dates, times


def same(a, b):
    a, b = float(a), float(b)
    return a == b or (math.isnan(a) and math.isnan(b))


def windows_of(dates, years=(2023, 2024, 2025)):
    specs, skipped = v16.window_specs(sorted({int(d[:4]) for d in dates}), years=years)
    return specs, skipped, {s["label"]: v16.window_rows(dates, s["start"], s["end"]) for s in specs}


# ── windows ──────────────────────────────────────────────────────────────────

def test_windows_nested_years_partition_and_order():
    keep, skipped = v16.window_specs(range(2016, 2026))
    assert [s["label"] for s in keep] == ["W1", "W2", "W3", "W5", "W7", "W10", "Y2023", "Y2024", "Y2025"]
    assert not skipped
    w = {s["label"]: s for s in keep}
    assert (w["W1"]["start"], w["W1"]["end"]) == ("2025-01-01", "2025-12-31")
    assert w["W3"]["start"] == "2023-01-01" and w["W10"]["start"] == "2016-01-01"
    # a 2023-2025 record (the Dutch sites): the long windows are listed, not run
    keep, skipped = v16.window_specs([2023, 2024, 2025])
    assert [s["label"] for s in keep] == ["W1", "W2", "W3", "Y2023", "Y2024", "Y2025"]
    assert [s["label"] for s in skipped] == ["W5", "W7", "W10"] and "2021" in skipped[0]["reason"]
    # rows: nested by inclusion, the single years partition W1-W3, Y2025 = W1
    dates, _ = make_times(range(2016, 2026), 5, np.random.default_rng(1))
    keep, _, rows = windows_of(dates)
    assert len(rows["W10"]) == len(dates) and len(rows["W1"]) == 5 and len(rows["W3"]) == 15
    assert np.array_equal(rows["W1"], rows["Y2025"])
    chk = v16.check_windows(keep, rows)
    assert chk["ok"] and chk["years_partition"] == {"W1": True, "W2": True, "W3": True}
    bad = dict(rows, W2=rows["W2"][1:])
    assert not v16.check_windows(keep, bad)["ok"]
    order = [s["label"] for s in v16.job_order(keep)]
    assert order == ["W3", "W1", "W2", "W5", "W7", "W10", "Y2023", "Y2024", "Y2025"]
    assert v16.parse_years("2016-2018, 2025") == [2016, 2017, 2018, 2025]
    args = v16.build_parser().parse_args(["run", "villaviciosa", "--years", "2016-2025"])
    assert v16.parse_years(args.years) == list(range(2016, 2026)) and args.max_px is None


# ── the window job is v14's refit job ───────────────────────────────────────

def test_window_job_is_v14s_refit_job():
    sd, dates, _ = synthetic_site()
    specs, _, rows_by = windows_of(dates)
    w = {s["label"]: s for s in specs}
    T = len(dates)
    mine, _ = v16.marea_window(sd, w["W3"], np.arange(T), MBS, 11)
    ref, _ = v14.run_job(sd, "all", T, 0, 11, "refit", MBS)
    for a, b in zip(mine, ref):
        assert a["mb"] == b["mb"] and a["N"] == b["N"] == T
        for k in ("rmse", "bias", "slope", "r", "coverage", "median_clear_obs", "lo_m", "hi_m",
                  "tau_px_median_min", "tau_px_max_min"):
            assert same(a[k], b[k]), k
        assert a["n"] == b["n"] and a["n_set"] == b["n_set"] and a["tau_bands_min"] == b["tau_bands_min"]
    assert set(sd.pools) == {"all"}                                     # the window's pool is removed
    assert np.isfinite(mine[1]["rmse"]) and mine[1]["coverage"] > 0.9
    # a sub-window equals the refit, the inversion and the score done directly
    rows = rows_by["Y2024"]
    mine, _ = v16.marea_window(sd, w["Y2024"], rows, MBS, 11)
    tau, tau_used, tp = v14.refit_lags(sd, rows)
    sd.levels.ensure(np.unique(tp))
    z, _, (lo, hi), _, _ = v14.invert_subset(sd.Y0, sd.C, sd.levels, rows, v14.lag_groups(tp), MBS)
    for row in mine:
        s = v14.score(sd.kind, sd.truth, z[row["mb"]][sd.unit_col])
        assert same(row["rmse"], s["rmse"]) and row["n"] == s["n"] and same(row["coverage"], s["coverage"])
        assert row["N"] == len(rows) and (row["lo_m"], row["hi_m"]) == (lo, hi)
        assert row["tau_bands_min"] == ";".join(f"{v:.3f}" for v in tau)
        assert row["tau_used_bands_min"] == ";".join(f"{v:.3f}" for v in tau_used)
        assert row["median_clear_obs_px"] == float(np.median(sd.C[rows].sum(axis=0)))
        assert row["scenes_sha1"] == v14.draw_sha1(rows) and row["window"] == "Y2024"
    with pytest.raises(ValueError):
        v16.marea_window(sd, w["Y2024"], rows[::-1], MBS, 11)


# ── the W3 gate and the fixed-set gate ───────────────────────────────────────

def test_w3_gate_against_a_v14_csv(tmp_path):
    sd, dates, _ = synthetic_site(seed=2)
    T = len(dates)
    w3 = [s for s in windows_of(dates)[0] if s["label"] == "W3"][0]
    ref_rows, _ = v14.run_job(sd, "all", T, 0, 11, "refit", MBS)
    p = tmp_path / "synthetic_refit.csv"
    v14.CsvLog(str(p), MBS).append(ref_rows)                           # v14's own CSV format
    ref = v16.find_v14_reference("synthetic", str(p))
    assert set(ref["rows"]) == {0, 3} and ref["compat"] is None
    mine_rows, _ = v16.marea_window(sd, w3, np.arange(T), MBS, 11)
    mine = {r["mb"]: r for r in mine_rows}
    g = v16.w3_gate(mine, ref, MBS)
    assert g["ok"] and g["bitwise"] and g["max_abs_diff"] == 0.0
    # through our own CSV: repr floats round-trip exactly
    path = str(tmp_path / "mine.csv")
    v16.ResumableCsv(path, v16.CSV_COLUMNS, need_field="mb", need=MBS).append(mine_rows)
    back = {r["mb"]: r for r in map(v16.typed, v16.ResumableCsv(path, v16.CSV_COLUMNS, need_field="mb",
                                                                 need=MBS).get("W3"))}
    assert v16.w3_gate(back, ref, MBS)["bitwise"]
    # last-digit noise (laptop vs server) passes, not bitwise; real differences fail
    noisy = {mb: dict(r, r=r["r"] + 2e-15) for mb, r in mine.items()}
    g = v16.w3_gate(noisy, ref, MBS)
    assert g["ok"] and not g["bitwise"]
    for change in (dict(rmse=mine[3]["rmse"] + 1e-6), dict(n=mine[3]["n"] - 1),
                   dict(tau_bands_min="0.000;1.000"), dict(coverage=mine[3]["coverage"] - 0.01),
                   dict(lo_m=mine[3]["lo_m"] + 1e-12)):
        assert not v16.w3_gate({0: mine[0], 3: dict(mine[3], **change)}, ref, MBS)["ok"], change
    assert not v16.w3_gate({0: mine[0]}, ref, MBS)["ok"]                # a rule missing
    g = v16.w3_gate(mine, None, MBS)
    assert g["checked"] is False and g["ok"] is False
    assert v16.find_v14_reference("synthetic", str(tmp_path / "none.csv")) is None
    compat = {"seed": 11, "n_units_scored": 36, "n_units_F": 36, "units_sha1": "a", "z_stored_sha1": "b",
              "tau_fixed_sha1": "c"}
    ref2 = dict(ref, compat=dict(compat), settings_path="x_settings.json")
    assert v16.fixed_set_gate(compat, ref2)["ok"]
    assert not v16.fixed_set_gate(dict(compat, units_sha1="z"), ref2)["ok"]
    assert v16.fixed_set_gate(compat, ref)["checked"] is False


# ── the extended record (Villaviciosa 2016-2025) ─────────────────────────────

def test_extended_record_coverage_and_match_with_the_base():
    rng = np.random.default_rng(3)
    dates, times = make_times(range(2021, 2026), 8, rng)
    overpass = {d: t.to_pydatetime() for d, t in zip(dates, times)}
    lost = dates[3]                                                     # 2021: no overpass time
    del overpass[lost]
    ext = v16.extended_record(dates, overpass, FakeBoundary(nan_at=times[10]), "2021-01-01", "2025-12-31")
    assert ext["n_without_time"] == 1 and ext["n_nonfinite_level"] == 1     # 2022: no finite level
    assert len(ext["dates"]) == len(dates) - 2 and lost not in set(ext["dates"])
    cov = ext["coverage"]
    assert (cov["2021"]["scenes"], cov["2021"]["with_overpass_time"], cov["2021"]["finite_level"]) == (8, 7, 7)
    assert cov["2022"]["finite_level"] == 7 and cov["2025"]["finite_level"] == 8
    assert np.all(np.diff(ext["time_index"]) > 0) and ext["t_real"].is_monotonic_increasing
    base = v16.extended_record(dates, overpass, FakeBoundary(), "2023-01-01", "2025-12-31")
    g, pos = v16.compare_records(base, ext, "2023-01-01", "2025-12-31")
    assert g["ok"] and g["max_abs_level_diff_m"] == 0.0 and len(pos) == 24
    assert not v16.compare_records(dict(base, h=base["h"] + 1e-6), ext, "2023-01-01", "2025-12-31")[0]["ok"]
    assert not v16.compare_records(dict(base, time_index=base["time_index"] + 1), ext,
                                   "2023-01-01", "2025-12-31")[0]["ok"]
    assert not v16.compare_records(dict(base, dates=base["dates"][1:]), ext, "2023-01-01", "2025-12-31")[0]["ok"]


def test_extended_arrays_match_the_base_table():
    sd, dates, times = synthetic_site(years=(2021, 2022, 2023, 2024, 2025), per_year=10, seed=7)
    pos = v16.window_rows(dates, "2023-01-01", "2025-12-31")
    lev = v14.LevelTable(FakeBoundary(), times[pos])
    lev.ensure(sorted(sd.levels.L))
    base = v14.SiteData("synthetic", sd.Y0[pos], sd.C[pos], lev, {"all": np.arange(len(pos))}, sd.unit_col,
                        sd.truth, sd.kind, sd.tau_fixed, s_km=sd.s_km,
                        m2a=dict(sd.m2a, Y=sd.m2a["Y"][pos], C=sd.m2a["C"][pos]))
    g = v16.compare_extension(base, sd, pos, band=sd.m2a["band"])
    assert g["ok"] and g["levels_max_abs_diff_m"] == 0.0 and g["n_lags_compared"] >= 16
    sd.m2a["C"][pos[0], 0] = ~sd.m2a["C"][pos[0], 0]
    assert not v16.compare_extension(base, sd, pos)["ok"]


# ── the window loop: aliases, resume, stops ─────────────────────────────────

def test_window_loop_copies_same_scenes_and_resumes(tmp_path, monkeypatch):
    sd, dates, _ = synthetic_site(seed=4)
    specs, _, rows_by = windows_of(dates)
    calls, real = [], v14.run_job

    def spy(*a, **k):
        calls.append(a[1])
        return real(*a, **k)

    monkeypatch.setattr(v14, "run_job", spy)
    path = str(tmp_path / "w.csv")
    seen = []

    def gate(lab, rows):
        seen.append((lab, sorted(rows)))
        return True

    res, info = v16.run_windows(sd, specs, rows_by, v16.ResumableCsv(path, v16.CSV_COLUMNS, need_field="mb",
                                                                     need=MBS), MBS, 11, time.time() + 600,
                                gate=gate)
    assert info["stopped"] is None and set(res) == {s["label"] for s in specs}
    assert len(calls) == 5 and calls[0] == "window:W3"                  # Y2025 = W1's scenes, not refitted
    assert seen == [("W3", MBS)] and info["status"]["Y2025"] == "same scenes as W1"
    df = pd.read_csv(path, keep_default_na=False, na_values=["nan"])
    assert len(df) == 2 * len(specs) and set(df["mb"]) == set(MBS)
    a = df[df["window"] == "W1"].sort_values("mb")
    b = df[df["window"] == "Y2025"].sort_values("mb")
    for k in ("rmse", "bias", "slope", "r", "coverage", "n", "tau_bands_min", "scenes_sha1"):
        assert list(a[k]) == list(b[k]), k
    assert (b["same_scenes_as"] == "W1").all() and (b["seconds"] == 0).all()
    assert (df["n_set"] == len(sd.unit_col)).all() and df["coverage"].between(0, 1).all()
    w = df[df["mb"] == 3].set_index("window")
    assert w.loc["W3", "N"] == len(dates) and w.loc["W1", "N"] < w.loc["W2", "N"] < w.loc["W3", "N"]
    m = df.pivot_table(index="window", columns="mb", values="coverage")
    assert (m[3] <= m[0] + 1e-12).all()                                  # the rule only removes estimates
    # resumable: a second pass computes nothing and reads the same values
    calls.clear()
    res2, info2 = v16.run_windows(sd, specs, rows_by, v16.ResumableCsv(path, v16.CSV_COLUMNS, need_field="mb",
                                                                       need=MBS), MBS, 11, time.time() + 600)
    assert not calls and set(info2["status"].values()) == {"resumed"}
    assert all(same(res2[k][mb]["rmse"], res[k][mb]["rmse"]) for k in res for mb in MBS)
    # killed mid-write: the half-written last window is dropped, then copied again (no compute)
    text = open(path, encoding="utf-8").read()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text[:-20])
    log3 = v16.ResumableCsv(path, v16.CSV_COLUMNS, need_field="mb", need=MBS)
    assert log3.is_done("Y2024") and not log3.is_done("Y2025")
    _, info3 = v16.run_windows(sd, specs, rows_by, log3, MBS, 11, time.time() + 600)
    assert not calls and info3["status"]["Y2025"] == "same scenes as W1"
    assert len(pd.read_csv(path)) == 2 * len(specs)


def test_window_loop_stops_on_the_clock_and_on_the_gate(tmp_path, monkeypatch):
    sd, dates, _ = synthetic_site(seed=5)
    specs, _, rows_by = windows_of(dates)

    def fresh(name):
        return v16.ResumableCsv(str(tmp_path / name), v16.CSV_COLUMNS, need_field="mb", need=MBS)

    res, info = v16.run_windows(sd, specs, rows_by, fresh("a.csv"), MBS, 11, time.time() - 1)
    assert res == {} and info["stopped"]["before"] == "W3"
    res, info = v16.run_windows(sd, specs, rows_by, fresh("b.csv"), MBS, 11, time.time() + 600,
                                gate=lambda lab, rows: False)
    assert set(res) == {"W3"} and info["stopped"] == {"after": "W3", "reason": "gate failed"}
    monkeypatch.setattr(v16, "COST_SAFETY", 1e9)                        # W3's cost says W1 cannot fit
    res, info = v16.run_windows(sd, specs, rows_by, fresh("c.csv"), MBS, 11, time.time() + 600)
    assert set(res) == {"W3"} and info["stopped"]["before"] == "W1"


# ── the NIDEM hook ───────────────────────────────────────────────────────────

def test_nidem_hook_absent_present_and_failing(tmp_path, monkeypatch):
    fn, info = v16.find_nidem_hook(path=str(tmp_path / "v15_missing.py"))
    assert fn is None and info["status"] == "todo" and "does not exist" in info["reason"]
    assert info["todo"].startswith("TODO")
    sd, dates, times = synthetic_site(seed=6)
    specs, _, rows_by = windows_of(dates)
    got = []

    def nidem_on_scenes(site, dates, units_keep):
        got.append((site, list(dates), np.asarray(units_keep).copy()))
        return sd.truth + 0.1                                            # one value per scored unit

    mod = types.ModuleType("fake_v15_hook")
    mod.nidem_on_scenes = nidem_on_scenes
    monkeypatch.setitem(sys.modules, "fake_v15_hook", mod)
    f = tmp_path / "fake_v15_hook.py"
    f.write_text("# stands for experiments/v15_nidem_scene_curves.py\n", encoding="utf-8")
    fn, info = v16.find_nidem_hook(path=str(f), module="fake_v15_hook")
    assert fn is nidem_on_scenes and info["status"] == "found"
    assert info["function"] == "fake_v15_hook.nidem_on_scenes"
    table = dict(dates=dates, time_index=np.arange(len(dates)) + 100, t_real=times,
                 units_keep=np.arange(len(sd.unit_col)) + 5, px_keep=np.arange(sd.Y0.shape[1]) + 5, n_keep=200)
    log = v16.ResumableCsv(str(tmp_path / "n.csv"), v16.NIDEM_COLUMNS, key="window")
    st = v16.nidem_windows(fn, info, sd, specs, rows_by, table, log, str(tmp_path), time.time() + 600)
    assert st["Y2025"] == "same scenes as W1" and sum(v == "computed" for v in st.values()) == 5
    assert len(got) == 5 and got[0][0] == "synthetic"
    assert got[0][1] == [str(d) for d in dates[rows_by["W3"]]]          # the window's own dates, W3 first
    assert np.array_equal(got[0][2], table["units_keep"])
    df = pd.read_csv(tmp_path / "n.csv")
    # one NIDEM row per (window, MAREA rule mb): 6 windows x rules 0 and 3 (NIDEM_COLUMNS carries mb)
    assert len(df) == 12 and (df.groupby("window").size() == 2).all() and set(df["mb"]) == {0, 3}
    assert (df["product"] == "NIDEM").all() and (df["n_set"] == len(sd.unit_col)).all()
    assert np.allclose(df["bias"], 0.1) and (df["coverage"] == 1).all()
    assert set(df["window"]) == {"W1", "W2", "W3", "Y2023", "Y2024", "Y2025"}
    # the real v15 now exposes the hook: v16 finds it by default and can call it with its keywords
    real, rinfo = v16.find_nidem_hook()
    assert rinfo["status"] == "found" and rinfo["function"] == "experiments.v15_nidem_scene_curves.nidem_on_scenes"
    import inspect
    required = [k for k, p in inspect.signature(real).parameters.items() if p.default is p.empty]
    v16_keys = {"site", "dates", "rows", "t_real", "h", "units_keep", "px_keep", "out_dir", "truth", "kind"}
    assert set(required) <= v16_keys and {"dates", "rows", "h", "units_keep", "truth", "kind"} <= \
        set(inspect.signature(real).parameters)
    # only the keywords a signature accepts are passed; **kwargs gets them all
    seen = {}

    def kw_hook(**kw):
        seen.update(kw)

    v16.call_hook(kw_hook, {"site": "s", "dates": [], "rows": np.arange(3)})
    assert set(seen) == {"site", "dates", "rows"}

    def needs_more(site, cube_path):
        return None

    with pytest.raises(TypeError):
        v16.call_hook(needs_more, {"site": "s"})
    # the return shapes: per unit, per inverted pixel, per keep pixel, or a dict
    uc, uk = np.array([0, 0, 1, 2]), np.array([10, 10, 11, 12])
    per_px, per_keep = np.array([1.0, 2.0, 3.0]), np.arange(20, dtype=float)
    assert np.array_equal(v16.hook_values(per_px, 4, uc, uk, 3, 20), [1, 1, 2, 3])
    assert np.array_equal(v16.hook_values(per_keep, 4, uc, uk, 3, 20), [10, 10, 11, 12])
    assert np.array_equal(v16.hook_values({"z_units": [5, 6, 7, 8]}, 4, uc, uk, 3, 20), [5, 6, 7, 8])
    assert np.array_equal(v16.hook_values({"z_keep": per_keep}, 4, uc, uk, 3, 20), [10, 10, 11, 12])
    with pytest.raises(ValueError):
        v16.hook_values(np.zeros(7), 4, uc, uk, 3, 20)
    # a failing hook is recorded, never raised
    def broken(site):
        raise RuntimeError("no cube here")

    log2 = v16.ResumableCsv(str(tmp_path / "n2.csv"), v16.NIDEM_COLUMNS, key="window")
    st = v16.nidem_windows(broken, {"function": "broken"}, sd, specs, rows_by, table, log2, str(tmp_path),
                           time.time() + 600)
    assert all(v.startswith("error") for v in st.values()) and len(pd.read_csv(tmp_path / "n2.csv")) == 0


def test_nidem_hook_gets_item_rule_and_a_refusal_is_recorded(tmp_path):
    """v16's NIDEM is v15's production NIDEM: the hook is asked for ITEM's
    pixel rule explicitly; a hook that refuses an input (SystemExit) is an
    error of that window, never a stop of the run."""
    sd, dates, times = synthetic_site(seed=7)
    specs, _, rows_by = windows_of(dates)
    table = dict(dates=dates, time_index=np.arange(len(dates)), t_real=times,
                 units_keep=np.arange(len(sd.unit_col)), px_keep=np.arange(sd.Y0.shape[1]), n_keep=200)
    rules = []

    def with_rule(site, dates, units_keep, pixel_rule="common"):
        rules.append(pixel_rule)
        return sd.truth + 0.1

    log = v16.ResumableCsv(str(tmp_path / "r.csv"), v16.NIDEM_COLUMNS, key="window")
    st = v16.nidem_windows(with_rule, {"function": "f"}, sd, specs, rows_by, table, log, str(tmp_path),
                           time.time() + 600)
    assert rules and set(rules) == {"item"} and any(v == "computed" for v in st.values())

    def refuses(site, dates):
        raise SystemExit("[synthetic] cube dates differ from the extraction dates")

    log2 = v16.ResumableCsv(str(tmp_path / "r2.csv"), v16.NIDEM_COLUMNS, key="window")
    st = v16.nidem_windows(refuses, {"function": "refuses"}, sd, specs, rows_by, table, log2, str(tmp_path),
                           time.time() + 600)
    assert st and all(v.startswith("error") and "cube dates differ" in v for v in st.values())


# ── the figure ───────────────────────────────────────────────────────────────

def _fake_results(out, site, years):
    """Plausible rows and settings for a site (no compute); Villaviciosa with
    every single year of its record, as the server command runs it."""
    specs, _ = v16.window_specs(years, years=years if site == "villaviciosa" else v16.SINGLE_YEARS)
    rows = []
    for s in specs:
        N = 150 * s["n_years"] + (20 if s["label"] in ("W1", "Y2025") else 0)
        for mb in MBS:
            rmse = 0.32 - 0.03 * math.log(s["n_years"]) + (0.01 if s["label"] == "Y2023" else 0.0) + 0.002 * mb
            rows.append(dict(site=site, window=s["label"], kind=s["kind"], n_years=s["n_years"], start=s["start"],
                             end=s["end"], N=N, mb=mb, rmse=rmse, bias=0.05, slope=0.8, r=0.9,
                             coverage=min(1.0, 0.9 + 0.02 * s["n_years"]) - 0.01 * mb, n=190, n_set=194,
                             median_clear_obs=N * 0.33, median_clear_obs_px=N * 0.33, lo_m=-2.0, hi_m=1.3,
                             tau_bands_min="0.000;5.000", tau_used_bands_min="0.000;5.000",
                             tau_px_median_min=5.0, tau_px_max_min=6.0, inversion_seconds=0.1, seconds=2.0,
                             scenes_sha1="x", same_scenes_as="W1" if s["label"] == "Y2025" else ""))
    pd.DataFrame(rows, columns=v16.CSV_COLUMNS).to_csv(os.path.join(out, f"{site}.csv"), index=False)
    fs = ({"unit": "RTK point", "n_units_scored": 194, "n_units_F": 194, "pixel_m": 10.0, "subsample": None}
          if site == "villaviciosa" else
          {"unit": "10 m pixel", "n_units_scored": 100000, "n_units_F": 248430, "pixel_m": 10.0,
           "subsample": {"n": 100000}})
    rec = {"extended": True, "extension_years": [2016, 2022], "n_nonfinite_level": 0, "n_without_time": 0}         if site == "villaviciosa" else {"extended": False}
    st = {"fixed_set": fs, "record": rec,
          "gates": {"w3_reproduces_v14_all_point": {"ok": True, "bitwise": site != "villaviciosa",
                                                    "max_abs_diff": 1.7e-15}},
          "nidem": {"status": "todo"}}
    v14.write_json(os.path.join(out, f"{site}_settings.json"), st)


def test_figure_in_both_languages_and_summary(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = str(tmp_path)
    _fake_results(out, "villaviciosa", range(2016, 2026))
    _fake_results(out, "escalda", [2023, 2024, 2025])
    v16.plot(argparse.Namespace(out=out, mb="3", lang="en", sites=None))
    assert os.path.exists(os.path.join(out, "window_curves_en_mb3.png"))
    assert os.path.exists(os.path.join(out, "window_curves_en_mb3.pdf"))
    summ = pd.read_csv(os.path.join(out, "window_curves_summary.csv"))
    assert set(summ["site"]) == {"villaviciosa", "escalda"} and set(summ["method"]) == {"MAREA"}
    data = {s: v16.plot_inputs(out, s) for s in ("villaviciosa", "escalda")}
    for lang, title, empty in (("es", "ventana temporal", "sin resultados"), ("en", "time window", "no results")):
        fig = v16.window_figure(data, 3, lang)
        texts = [t.get_text() for t in fig.texts] + [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any(title in t for t in texts) and any(empty in t for t in texts)
        assert any("v15_nidem_scene_curves" in t for t in texts)        # the NIDEM TODO is on the figure
        assert any("2016-2022" in t and "EOT20" in t for t in texts)     # the extension's boundary line
        assert any("2016-2025" in t for t in texts)                       # the single years drawn
        labels = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels()]
        assert any(t.startswith("10\n") for t in labels)                 # W10 at Villaviciosa
        plt.close(fig)
    assert v16.fmt_num(0.1905, 3, "es") == "0,191" and v16.fmt_int(100000, "es") == "100.000"


# ── guards ───────────────────────────────────────────────────────────────────

def test_rtk_and_output_guards():
    from tests import test_r1_guard

    static = [n for n in dir(test_r1_guard) if n.startswith("test_static")]
    assert static
    for n in static:
        getattr(test_r1_guard, n)()                                      # scans experiments/, v16 included
    tree = ast.parse(open(v16.__file__, encoding="utf-8").read())
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and "load_rtk" in (getattr(c.func, "attr", None), getattr(c.func, "id", None))]
    assert all(not c.args and not c.keywords for c in calls)            # the survey comes via v14.rtk_units()
    for bad in ("products_marea/x", "results/method_comparison_same_scenes", "products_escalda_10m/y"):
        with pytest.raises(SystemExit):
            v14.guard_out(bad)
    v14.guard_out(v16.OUT)
