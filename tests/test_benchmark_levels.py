"""Tests of experiments.benchmark.levels: the boundary tide of the v13 benchmark.

They pin (spec §8, the levels part of the package tests): the exact level
table equals per-shift provider calls bitwise with a fake provider that
answers in time order (as PyTMDBoundary); the batched calls receive sorted
unique instants in chunks; subset rows equal subset calls; beyond the table
``at()`` makes ONE live per-shift call and never caps the lag; a
batch-variant provider triggers the per-shift fallback; MAREA's bank built
from the table equals the bank ``marea.reconstruct`` builds; the 1-min
series and its cache (v8's tag, written atomically), the 10-min grid, the
61-min low-pass and both mark versions; the production provider calls; the
IOC chunk inventory; the byte-stable npz files behind the package key.
Validation v2 (reviews of 2026-10-14): a union IOC year counts by its data
(an empty or thin download is renamed, fetched again and blocks the marker
unless accepted), the CLI exits 1 while the cache is not ready; the CMEMS
campaign per (site, window) never substitutes a dataset, passes dataset
ids (not the chooser's tuple), measures the hourly-mean stamp convention
and maps statuses to exit codes; LG centres each judge's stamps as the lag
truths do and drops judges without a finite positive s_g.

Run:  python -X utf8 -m pytest tests/test_benchmark_levels.py -q
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, levels, product  # noqa: E402
from pyintertidal import marea  # noqa: E402
from pyintertidal import tide_estimators as te  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  fake providers
# ─────────────────────────────────────────────────────────────────────────────

class FakeTide:
    """A smooth multi-constituent tide evaluated elementwise from the
    instants' int64 nanoseconds. Like PyTMDBoundary it answers in TIME ORDER
    whatever the input order. ``batch_dependent`` adds a last-bit term that
    depends on the batch size (a provider that is not batch-invariant);
    ``nan_before`` returns NaN before that instant."""

    def __init__(self, batch_dependent=False, nan_before=None):
        self.batch_dependent = batch_dependent
        self.nan_before = None if nan_before is None else pd.Timestamp(nan_before).value
        self.calls = []

    def value(self, ns):
        x = np.asarray(ns, np.int64) / 3.6e12                    # hours since 1970
        return (1.7 * np.cos(2 * np.pi * x / 12.4206012 + 0.3)
                + 0.45 * np.cos(2 * np.pi * x / 12.0 + 1.1)
                + 0.2 * np.cos(2 * np.pi * x / 25.8193417 - 0.4)
                + 0.12 * np.cos(2 * np.pi * x / 6.2103006 + 2.0))

    def levels(self, times):
        idx = pd.DatetimeIndex(pd.to_datetime(times))
        ns = np.sort(idx.asi8)                                  # time order
        self.calls.append(idx.asi8.copy())
        h = self.value(ns)
        if self.batch_dependent:
            h = h + 1e-9 * len(ns)
        if self.nan_before is not None:
            h = np.where(ns < self.nan_before, np.nan, h)
        return h


def _record(T=30, seed=4):
    """Record instants as the cubes have them: about one per 2.5 days near
    10:50 UTC with whole seconds, strictly increasing."""
    rng = np.random.default_rng(seed)
    days = np.cumsum(rng.integers(1, 5, T))
    sec = rng.integers(0, 3600, T)
    t = (pd.Timestamp("2023-01-03 10:20:00") + pd.to_timedelta(days, "D")
         + pd.to_timedelta(sec, "s"))
    return pd.DatetimeIndex(t)


def _per_shift(provider, t, tau):
    return np.asarray(provider.levels(t - pd.Timedelta(minutes=float(tau))), float)


def _bits(a):
    return np.ascontiguousarray(a, np.float64).view(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  the exact level table
# ─────────────────────────────────────────────────────────────────────────────

def test_tau_tab_max():
    assert [config.tau_tab_max(K) for K in (1, 2, 3, 8, 13)] == [240, 240, 360, 1260, 2160]
    assert levels.tau_tab_max is config.tau_tab_max
    with pytest.raises(ValueError):
        config.tau_tab_max(0)
    f = config.FROZEN["levels"]
    assert (levels.TAU_MIN, levels.CHUNK, levels.CHECK_EVERY, levels.LOWPASS_WIDTH) == \
        (f["tau_min"], f["chunk"], f["check_every"], f["marks_lp"]["width_min"]) == \
        (-90, 150_000, 60, 61)


def test_table_equals_per_shift_calls_bitwise():
    p = FakeTide()
    t = _record()
    h0 = _per_shift(p, t, 0)
    lev = levels.build_level_table(p, t, 300, chunk=1000, h0=h0)
    assert lev.batch_invariant
    assert lev.shifts.tolist() == list(range(-90, 301))
    assert lev.H.shape == (391, len(t)) and lev.H.dtype == np.float64
    for k, tau in enumerate(lev.shifts):                       # EVERY row, bitwise
        assert np.array_equal(_bits(lev.H[k]), _bits(_per_shift(p, t, tau))), tau
    assert np.array_equal(_bits(lev.h0), _bits(h0))
    # G-levels checked the bank rows, every 60th row and the last row
    want = ({int(x) for x in marea.BANK_TAUS} | set(range(-90, 301, 60)) | {300})
    assert set(lev.checked.tolist()) == want
    assert lev.meta["mismatch"] == []
    # read-only: nobody edits the frozen levels
    with pytest.raises(ValueError):
        lev.H[0, 0] = 1.0


def test_batched_calls_get_sorted_unique_instants_in_chunks():
    p = FakeTide()
    t = _record(T=25)
    lev = levels.build_level_table(p, t, 240, chunk=700)
    n_batched = lev.meta["n_batched_calls"]
    batched = p.calls[:n_batched]
    assert n_batched == int(np.ceil(lev.meta["n_unique_instants"] / 700)) > 1
    allv = np.concatenate(batched)
    assert all(len(c) <= 700 for c in batched)
    assert np.all(np.diff(allv) > 0)                           # sorted and unique overall
    want = np.unique(t.asi8[None, :] - np.arange(-90, 241)[:, None] * 60_000_000_000)
    assert np.array_equal(allv, want)


def test_subset_rows_equal_subset_calls():
    p = FakeTide()
    t = _record(T=40)
    lev = levels.build_level_table(p, t, 400, chunk=5000)
    rng = np.random.default_rng(7)
    for _ in range(5):
        sel = np.sort(rng.choice(len(t), 13, replace=False))
        tau = int(rng.integers(-90, 401))
        n0 = len(p.calls)
        got = lev.at(tau, sel)
        assert len(p.calls) == n0                                # a table read, no call
        assert np.array_equal(_bits(got), _bits(_per_shift(p, t[sel], tau)))
    # a boolean mask selects the same rows; integer-valued floats are accepted
    m = np.zeros(len(t), bool)
    m[[1, 5, 9]] = True
    assert np.array_equal(lev.at(15.0, m), lev.at(15, [1, 5, 9]))
    for bad in (7.5, np.nan):
        with pytest.raises(ValueError):
            lev.at(bad)
    with pytest.raises(ValueError):
        lev.at(0, [3, 1])                                        # unsorted


def test_beyond_the_table_is_one_live_per_shift_call_never_capped():
    p = FakeTide()
    t = _record(T=20)
    lev = levels.build_level_table(p, t, 240, chunk=4000)
    sel = np.array([0, 3, 4, 11, 19])
    for tau in (250, 2160, -120):
        n0 = len(p.calls)
        got = lev.at(tau, sel)
        assert len(p.calls) == n0 + 1                            # exactly one call
        assert np.array_equal(p.calls[-1], (t[sel] - pd.Timedelta(minutes=tau)).asi8)
        assert np.array_equal(_bits(got), _bits(_per_shift(p, t[sel], tau)))
        assert lev.live_calls[-1] == {"tau": tau, "n": 5, "why": "beyond the table"}
    # not capped at the last row
    assert not np.array_equal(lev.at(250, sel), lev.at(240, sel))
    # without a provider a live read is an error, not a silent cap
    bare = levels.LevelTable(lev.shifts, lev.H, lev.t)
    with pytest.raises(RuntimeError):
        bare.at(300)
    # a factory is resolved on first use
    lazy = levels.LevelTable(lev.shifts, lev.H, lev.t, provider=lambda: p)
    assert np.array_equal(lazy.at(300, sel), lev.at(300, sel))


def test_batch_variant_provider_falls_back_to_per_shift_calls():
    p = FakeTide(batch_dependent=True)
    t = _record(T=20)
    lev = levels.build_level_table(p, t, 240, chunk=3000)
    assert not lev.batch_invariant
    assert lev.meta["mismatch"]
    for k, tau in enumerate(lev.shifts):                       # rows = per-shift calls
        assert np.array_equal(_bits(lev.H[k]), _bits(_per_shift(p, t, tau)))
    # the full record reads the table; a proper subset calls the provider
    n0 = len(p.calls)
    assert np.array_equal(lev.at(30), lev.H[120])
    assert len(p.calls) == n0
    sel = np.arange(0, 20, 3)
    got = lev.at(30, sel)
    assert len(p.calls) == n0 + 1
    assert np.array_equal(_bits(got), _bits(_per_shift(p, t[sel], 30)))
    assert lev.live_calls[-1]["why"] == "batch-variant provider, subset"


def test_h0_must_equal_the_table():
    p = FakeTide()
    t = _record(T=12)
    h0 = _per_shift(p, t, 0)
    h0[3] += 1e-12
    with pytest.raises(RuntimeError):
        levels.build_level_table(p, t, 240, chunk=10_000, h0=h0)


def test_nonfinite_levels_raise_before_any_per_shift_call():
    t = _record(T=10)
    p = FakeTide(nan_before=t[0] - pd.Timedelta(minutes=100))   # tau > 100 is NaN at t[0]
    with pytest.raises(ValueError, match="non-finite"):
        levels.build_level_table(p, t, 240, chunk=10_000)
    assert len(p.calls) == 1                                   # the one batched call only


def test_record_instants_validated():
    p = FakeTide()
    t = _record(T=10)
    with pytest.raises(ValueError):
        levels.build_level_table(p, t[::-1], 240)
    with pytest.raises(ValueError):
        levels.build_level_table(p, t.tz_localize("UTC"), 240)


def test_bank_equals_the_bank_reconstruct_builds():
    p = FakeTide()
    t = _record(T=35)
    lev = levels.build_level_table(p, t, 360, chunk=8000)
    sel = np.sort(np.random.default_rng(3).choice(len(t), 17, replace=False))
    t_real = t[sel]
    # marea.reconstruct: one boundary.levels call per BANK_TAUS shift
    ref = te.ShiftBank(marea.BANK_TAUS, [
        np.asarray(p.levels(t_real - pd.Timedelta(minutes=tv)), float)
        for tv in marea.BANK_TAUS])
    bank = lev.bank(sel)
    assert np.array_equal(_bits(bank.H), _bits(ref.H))
    assert np.array_equal(bank.taus, ref.taus)
    assert np.array_equal(_bits(bank.at(0.0)), _bits(lev.at(0, sel)))
    # tau120: the bank restricted to the shifts <= 120
    b120 = lev.bank(sel, taus=[x for x in marea.BANK_TAUS if x <= 120])
    assert b120.taus[-1] == 120.0 and b120.H.shape == (12, len(sel))
    # Granadeiro ctl and the truths read stacks of exact rows
    st = lev.stack(range(-45, 181, 5), sel)
    assert st.shape == (46, len(sel)) and np.array_equal(st[9], lev.at(0, sel))


def test_save_and_load_round_trip(tmp_path):
    p = FakeTide()
    t = _record(T=15)
    lev = levels.build_level_table(p, t, 240, chunk=10_000)
    path = tmp_path / "levels.npz"
    lev.save(path)
    b1 = path.read_bytes()
    lev.save(path)
    assert path.read_bytes() == b1                             # byte-stable file
    # a rebuild of the same table gives the same bytes (a keyed package file:
    # re-running the step must not change the package key)
    again = levels.build_level_table(FakeTide(), t, 240, chunk=10_000)
    again.save(tmp_path / "again.npz")
    assert (tmp_path / "again.npz").read_bytes() == b1
    assert again.build_seconds is not None and "seconds" not in again.meta
    back = levels.LevelTable.load(path, provider=p)
    assert np.array_equal(back.shifts, lev.shifts)
    assert np.array_equal(_bits(back.H), _bits(lev.H))
    assert back.t.equals(lev.t) and back.batch_invariant
    assert back.meta["checked"] == lev.meta["checked"]
    assert np.array_equal(back.at(300, [1, 2]), lev.at(300, [1, 2]))


# ─────────────────────────────────────────────────────────────────────────────
#  series and marks
# ─────────────────────────────────────────────────────────────────────────────

def test_lowpass():
    rng = np.random.default_rng(1)
    h = rng.normal(size=500)
    lp = levels.lowpass(h, 61)
    assert np.isnan(lp[:30]).all() and np.isnan(lp[-30:]).all()
    ref = np.array([h[i - 30:i + 31].mean() for i in range(30, 470)])
    assert np.allclose(lp[30:470], ref, rtol=0, atol=1e-14)
    h[200] = np.nan                                            # one gap
    lp = levels.lowpass(h, 61)
    assert np.isnan(lp[170:231]).all() and np.isfinite(lp[169]) and np.isfinite(lp[231])
    assert np.array_equal(levels.lowpass(np.full(100, 2.5), 61)[30:70], np.full(40, 2.5))
    for w in (0, 60, -1):
        with pytest.raises(ValueError):
            levels.lowpass(h, w)
    assert np.isnan(levels.lowpass(np.ones(10), 61)).all()


def test_marks_lp_removes_subhour_noise_and_keeps_the_tide(tmp_path):
    p = FakeTide()
    idx = pd.date_range("2024-03-01", "2024-03-31", freq="1min", inclusive="left")
    t_min = levels.minutes_since_epoch(idx)
    h_true = p.value(idx.asi8)
    rng = np.random.default_rng(5)
    h = h_true + 0.03 * rng.standard_normal(len(idx))           # sub-hour wiggles
    mk = levels.build_marks(t_min, h)
    lp, raw = mk["marks_lp"], mk["marks_raw"]
    assert lp["lowpass_width"] == 61
    assert raw["n_turning_points"] > 20 * lp["n_turning_points"]
    # the low-passed marks sit on the true extremes (M4 and M6 kept)
    true = levels.marks(t_min, h_true)
    hw_true = true["t"][true["is_hw"]]
    hw_lp = lp["t"][lp["is_hw"]]
    d = np.abs(hw_lp[:, None] - hw_true[None, :]).min(axis=1)
    assert np.median(d) <= 3.0
    assert lp["n_marks"] == true["n_marks"]
    # persistence
    levels.save_marks(tmp_path / "marks.npz", **mk)
    back = levels.load_marks(tmp_path / "marks.npz")
    assert set(back) == {"marks_lp", "marks_raw"}
    for name in back:
        for k in levels.MARK_ARRAYS:
            assert np.array_equal(back[name][k], mk[name][k])
        assert back[name]["n_marks"] == mk[name]["n_marks"]
    assert back["marks_lp"]["is_hw"].dtype == bool


def test_minutes_since_epoch_is_v8s():
    from experiments import v8_granadeiro2021 as v8

    idx = pd.date_range("2022-12-30", periods=5000, freq="1min")
    assert np.array_equal(levels.minutes_since_epoch(idx), v8.minutes_since_epoch(idx))


def test_boundary_series_1min_cache_and_10min_grid(tmp_path):
    from experiments import v8_granadeiro2021 as v8

    p = FakeTide()
    cache = str(tmp_path / "common" / "boundary_1min_cache.npz")
    t_min, h, cached = levels.boundary_series_1min(p, "fake boundary", cache)
    assert not cached and os.path.exists(cache)
    assert not [f for f in os.listdir(tmp_path / "common") if ".tmp-" in f]
    idx = pd.date_range("2022-12-30", "2026-01-03", freq="1min", inclusive="left")
    assert len(t_min) == len(h) == len(idx)
    assert np.array_equal(t_min, v8.minutes_since_epoch(idx))
    assert np.array_equal(_bits(h), _bits(p.value(idx.asi8)))
    with np.load(cache) as z:
        assert str(z["tag"]) == ("fake boundary|2022-12-30 00:00:00|"
                                 "2026-01-03 00:00:00|1min")
    n0 = len(p.calls)
    t2, h2, cached2 = levels.boundary_series_1min(p, "fake boundary", cache)
    assert cached2 and len(p.calls) == n0 and np.array_equal(h2, h)
    # the package reads the cache without a provider
    t4, h4 = levels.load_series_1min(cache, name="fake boundary")
    assert np.array_equal(t4, t_min) and np.array_equal(h4, h)
    assert levels.series_1min_index().equals(idx)
    with pytest.raises(ValueError):
        levels.load_series_1min(cache, name="another boundary")
    # another provider name: the old cache is kept aside and rebuilt
    t3, h3, cached3 = levels.boundary_series_1min(p, "other boundary", cache)
    assert not cached3
    assert [f for f in os.listdir(tmp_path / "common") if ".stale-" in f]
    # the 10-min series: every 10th sample on 2023-01-01 00:00 .. 2025-12-31 23:50
    i10, h10 = levels.series_10min(t_min, h)
    assert i10[0] == pd.Timestamp("2023-01-01 00:00") and i10[-1] == pd.Timestamp("2025-12-31 23:50")
    assert len(i10) == 1096 * 144
    assert np.array_equal(h10, h[2880:2880 + 10 * len(i10):10])
    with pytest.raises(ValueError):
        levels.series_10min(t_min + 0.5, h)             # grid not on the series
    with pytest.raises(ValueError):
        levels.series_10min(t_min[:-3000], h[:-3000])   # series ends too early


# ─────────────────────────────────────────────────────────────────────────────
#  sites and the production provider
# ─────────────────────────────────────────────────────────────────────────────

def test_site_configs_and_paths():
    c = config.site_config("escalda", 10)
    assert (c.cube, c.extract, c.overpass) == (
        "ndwi_cube_escalda_2023-2025_10m.nc", "products_escalda_10m/marea_extract.npz",
        "products_escalda_10m/overpass_times.json")
    assert (c.pixel_m, c.mouth_side, c.tag) == (10.0, "west", "eot20_g2")
    assert c.common_dir == config.common_dir("escalda", 10) == \
        "results/v13_benchmark_10m/common/escalda"
    c20 = config.site_config("escalda", 20)
    assert (c20.cube, c20.extract, c20.pixel_m) == (
        "ndwi_cube_escalda_2023-2025_20m.nc", "products_escalda/marea_extract.npz", 20.0)
    # the rule-tz record is the 10 m transition record at both grids
    assert c20.transition_result == c.transition_result == \
        "products_escalda_10m/marea_eot20_g2/result.json"
    assert config.out_root(20) == "results/v13_benchmark_20m"
    v = config.site_config("villaviciosa", 10)
    assert (v.cube, v.mouth_side, v.boundary["kind"], v.elevation_truth) == (
        "ndwi_cube_villaviciosa_grande_10y.nc", "any", "pytmd", ("lidar", "rtk"))
    f = config.site_config("ferrol", 10)
    assert (f.boundary["exclude"], f.elevation_truth, f.judges) == (("fer2",), (), ("fer2", "fer4_3212"))
    assert config.site_config("ems", 10).mouth_side == "north+west"
    assert len(config.site_config("ems", 10).judges) == 10
    pc = config.site_config("pc", 20, "profile")
    assert pc.root == "results/v13_benchmark_20m/control/profile"
    assert pc.common_dir == "results/v13_benchmark_20m/control/profile/common/pc"
    assert pc.boundary["aoi"] == "wadden" and pc.extract is None
    for bad in (("ferrol", 20), ("villaviciosa", 20), ("pc", 10), ("tagus", 10),
                ("escalda", 30)):
        with pytest.raises(ValueError):
            config.site_config(*bad)
    with pytest.raises(ValueError):
        config.site_config("pc", 20)                          # needs a scenario
    assert [config.SITE_ID[s] for s in config.ALL_SITES] == [1, 2, 3, 4, 5, 9]
    doc = config.frozen_document()
    assert set(doc["sections"]) == set(doc["sha256"]) == set(config.FROZEN_SECTIONS)
    assert doc["package_key_sections"] == ["record", "package", "obs", "domain", "levels",
                                           "subsets"]
    # the package's neighbourhood superset NB and the judges' rule are one rule
    nb = dict(config.FROZEN["package"]["NB"])
    assert {k: nb[k] for k in config.FROZEN["judges"]["neighbourhood"]} == \
        dict(config.FROZEN["judges"]["neighbourhood"])

def test_make_provider_uses_the_production_calls(monkeypatch):
    import pyintertidal.boundary as bmod
    import pyintertidal.net as net
    from pyintertidal import sites

    seen = {"certs": 0}
    monkeypatch.setattr(net, "use_system_certificates",
                        lambda: seen.__setitem__("certs", seen["certs"] + 1) or True)

    def fake_make_boundary(aoi, **kw):
        seen["make"] = (aoi, kw)
        return "consensus"

    class FakePyTMD:
        def __init__(self, model, lat, lon, directory="tide_models"):
            seen["pytmd"] = (model, lat, lon, directory)

    monkeypatch.setattr(bmod, "make_boundary", fake_make_boundary)
    monkeypatch.setattr(bmod, "PyTMDBoundary", FakePyTMD)
    for site, n, excl in (("escalda", 2, ["trnz"]), ("wadden", 1, ["harl"]),
                          ("ems", 1, ["delf"]), ("ferrol", 1, ["fer2"])):
        assert levels.make_provider(config.site_config(site, 10)) == "consensus"
        aoi, kw = seen["make"]
        assert aoi.centroid == sites.get(site).centroid
        assert kw == dict(tide_model="EOT20", n_gauges=n, period=("2023-01-01", "2025-12-31"),
                          directory="tide_models", cache_dir="data_v4/gauges", exclude=excl)
    levels.make_provider(config.site_config("pc", 20, "profile"))
    assert seen["make"][0].centroid == sites.get("wadden").centroid
    assert seen["make"][1]["exclude"] == ["harl"]
    levels.make_provider(config.site_config("villaviciosa", 10))
    lat, lon = sites.get("villaviciosa").centroid
    assert seen["pytmd"] == ("EOT20", lat, lon, "tide_models")
    assert seen["certs"] == 6
    for s in config.ALL_SITES:
        cfg = config.site_config(s, 20 if s == "pc" else 10, "zero" if s == "pc" else None)
        assert levels.check_boundary_spec(cfg)


def test_provider_name_is_v8s_and_describe_provider():
    from experiments import v8_granadeiro2021 as v8
    from pyintertidal.boundary import (EnsembleBoundary, GaugeBoundary, InterpolatedGauges,
                                       PyTMDBoundary)

    cfg = config.site_config("villaviciosa", 10)
    b, name = v8.build_boundary("villaviciosa", v8.SITES["villaviciosa"])
    assert levels.provider_name(cfg, b) == name
    tt = pd.date_range("2023-01-01", periods=5, freq="h")
    g1 = GaugeBoundary(tt, np.zeros(5), name="vlis")
    g2 = GaugeBoundary(tt, np.zeros(5), name="brsk")
    ens = EnsembleBoundary([PyTMDBoundary("EOT20", 51.4, 3.9),
                            InterpolatedGauges([g1, g2], [10.0, 20.0], name="2 gauges")],
                           name="consensus EOT20 + 2 gauges")
    d = levels.describe_provider(ens)
    assert d["gauges"] == ["vlis", "brsk"]
    assert d["models"] == [{"model": "EOT20", "lat": 51.4, "lon": 3.9}]
    assert levels.provider_name(config.site_config("escalda", 10), ens) == ens.name


def test_ioc_chunks_lists_every_file_the_provider_reads(tmp_path):
    from pyintertidal import sites

    lat, lon = sites.get("escalda").centroid
    cache = tmp_path / "gauges"
    cache.mkdir()
    stations = [  # trnz nearest but held out; vlis, brsk the members; far1 unused
        {"Code": "trnz", "Lat": lat + 0.01, "Lon": lon},
        {"Code": "vlis", "Lat": lat + 0.05, "Lon": lon},
        {"Code": "brsk", "Lat": lat + 0.10, "Lon": lon},
        {"Code": "far1", "Lat": lat + 0.50, "Lon": lon},
    ]
    st_path = cache / "ioc_stations.json"
    st_path.write_text(json.dumps(stations))
    files = {"ioc_vlis_2023-01-01_2025-12-31.json": "[1]",
             "ioc_vlis_2025-09-27_2025-12-31.json": "[2]",
             "ioc_brsk_2023-01-01_2025-12-31.json": "[3]",
             "ioc_trnz_2023-01-01_2025-12-31.json": "[4]",
             "ioc_far1_2023-01-01_2025-12-31.json": "[5]"}
    for k, v in files.items():
        (cache / k).write_text(v)
    cfg = config.site_config("escalda", 20)
    out = levels.ioc_chunks(cfg, cache_dir=str(cache), stations_path=str(st_path))
    assert [(e["code"], os.path.basename(e["path"])) for e in out] == [
        ("ioc_stations", "ioc_stations.json"),
        ("vlis", "ioc_vlis_2023-01-01_2025-12-31.json"),
        ("vlis", "ioc_vlis_2025-09-27_2025-12-31.json"),
        ("brsk", "ioc_brsk_2023-01-01_2025-12-31.json")]
    for e in out:
        path = os.path.normpath(os.path.join(ROOT, e["path"]))   # relative to the root
        assert e["sha256"] == product.sha256_file(path)
        assert e["size"] == os.path.getsize(path)
    assert levels.ioc_chunks(config.site_config("villaviciosa", 10)) == []


# ─────────────────────────────────────────────────────────────────────────────
#  byte-stable files behind the package key
# ─────────────────────────────────────────────────────────────────────────────

def test_npz_files_are_byte_stable_and_content_hashed(tmp_path):
    a = {"x": np.arange(10, dtype=np.int16), "y": np.linspace(0, 1, 7)}
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    product.save_npz(p1, a)
    product.save_npz(p2, a)
    assert p1.read_bytes() == p2.read_bytes()
    back = product.load_npz(p1)
    assert np.array_equal(back["x"], a["x"]) and back["x"].dtype == np.int16
    # np.savez stamps the write time into the zip; the content hash ignores it
    p3 = tmp_path / "c.npz"
    np.savez(p3, **a)
    assert product.sha256_file(p3) == product.sha256_file(p1)
    b = dict(a, y=a["y"] + 1e-15)
    product.save_npz(p2, b)
    assert product.sha256_file(p2) != product.sha256_file(p1)
    # the package key: data bytes enter, judges.json and git do not
    man = {"files": {"levels.npz": {"sha256": product.sha256_file(p1)},
                     "judges.json": {"sha256": "j1"}},
           "inputs": {"cube": {"sha256": "c1"}}, "obs_sha": "o1",
           "frozen": config.frozen_shas(config.PACKAGE_SECTIONS + ("scoring",)),
           "git": {"head": "h1", "dirty": False}}
    k = product.package_key(man)
    man2 = json.loads(json.dumps(man))
    man2["files"]["judges.json"]["sha256"] = "j2"
    man2["git"] = {"head": "h2", "dirty": True}
    man2["frozen"]["scoring"] = "other"
    assert product.package_key(man2) == k
    man2["files"]["levels.npz"]["sha256"] = product.sha256_file(p2)
    assert product.package_key(man2) != k


# ═════════════════════════════════════════════════════════════════════════════
#  validation v2: the union-period boundary and the water-level sources
#  (docs/validation_v2_spec.md §1.3, §1.5, §1.6, §4.3, §10.3)
# ═════════════════════════════════════════════════════════════════════════════

from experiments.benchmark import settings  # noqa: E402
from experiments.benchmark.fetch import cmems  # noqa: E402
from experiments.benchmark.fetch import ioc as fioc  # noqa: E402
from experiments.benchmark.fetch import tides as ftides  # noqa: E402

UTM31 = "EPSG:32631"
STATIONS = os.path.join(ROOT, config.IOC_STATIONS)


def _union_record(T=60, seed=11):
    """Instants spread over 2017-2025 near 10:50 UTC, strictly increasing."""
    rng = np.random.default_rng(seed)
    days = np.sort(rng.choice(np.arange(1, 3285), T, replace=False))
    sec = rng.integers(0, 1800, T)
    return pd.DatetimeIndex(pd.Timestamp("2017-01-01 10:40") + pd.to_timedelta(days, "D")
                            + pd.to_timedelta(sec, "s"))


def _grid(nx=100, ny=40, res=50.0, x0=540000.0, y0=5700000.0):
    """A small UTM 31N frame near the Westerschelde mouth (pixel centres)."""
    return {"xs": x0 + res / 2 + res * np.arange(nx), "ys": y0 - res / 2 - res * np.arange(ny),
            "crs": UTM31}


# ─────────────────────────────────────────────────────────────────────────────
#  LB: the provider over the union period and the exact table
# ─────────────────────────────────────────────────────────────────────────────

def test_v2_exact_table_over_the_union_period_equals_per_shift_calls():
    assert levels.UNION_LEVEL_PERIOD == levels._inclusive(settings.UNION_PERIOD) == \
        fioc.UNION_FETCH_PERIOD == ("2017-01-01", "2025-12-31")
    p = FakeTide()
    t = _union_record()
    assert t[0].year == 2017 and t[-1].year == 2025
    lev = levels.build_level_table(p, t, 240, chunk=3000, h0=_per_shift(p, t, 0))
    assert lev.batch_invariant and lev.meta["n_batched_calls"] > 1
    for k, tau in enumerate(lev.shifts):                       # EVERY row, bitwise
        assert np.array_equal(_bits(lev.H[k]), _bits(_per_shift(p, t, tau))), tau
    # LB is the table's tau = 0 row, the same at every pixel and subset
    lb = levels.level_lb(lev, "escalda")
    assert lb.computable and lb.G == 1 and lb.pix is None
    sel = np.array([0, 7, 33, 59])
    got = lb.pixel_levels([5, 9, 1000], sel=sel)
    assert got.shape == (3, 4)
    for row in got:
        assert np.array_equal(_bits(row), _bits(_per_shift(p, t[sel], 0)))
    assert lb.unit_status()["status"] == "ok"
    bands, ser = lb.band_series(np.array([3, 4, 8]), np.array([0, 1, 1]), sel=sel)
    assert bands.tolist() == [0, 1] and np.array_equal(ser[0], ser[1])


def test_v2_union_series_1min_keeps_v8s_grid_and_tag(tmp_path):
    idx = levels.series_1min_index(levels.UNION_LEVEL_PERIOD)
    assert idx[0] == pd.Timestamp("2016-12-30 00:00") and idx[-1] == pd.Timestamp("2026-01-02 23:59")
    assert levels.series_1min_tag("b", levels.UNION_LEVEL_PERIOD) == \
        "b|2016-12-30 00:00:00|2026-01-03 00:00:00|1min"
    # v13 default unchanged
    assert levels.series_1min_index().equals(
        pd.date_range("2022-12-30", "2026-01-03", freq="1min", inclusive="left"))
    p = FakeTide()
    period = ("2019-03-01", "2019-06-30")                       # > one 92-day chunk
    cache = str(tmp_path / "boundary_1min_union.npz")
    t_min, h, cached = levels.boundary_series_1min(p, "fake", cache, period=period)
    want = levels.series_1min_index(period)
    assert not cached and len(h) == len(want) == (122 + 4) * 1440
    assert np.array_equal(_bits(h), _bits(p.value(want.asi8)))
    assert len(p.calls) == int(np.ceil(len(want) / (92 * 1440)))
    with np.load(cache) as z:
        assert str(z["tag"]) == levels.series_1min_tag("fake", period)
    n0 = len(p.calls)
    t2, h2, cached2 = levels.boundary_series_1min(p, "fake", cache, period=period)
    assert cached2 and len(p.calls) == n0 and np.array_equal(h2, h)
    t3, h3 = levels.load_series_1min(cache, name="fake", period=period)
    assert np.array_equal(h3, h) and np.array_equal(t3, t_min)
    with pytest.raises(ValueError):
        levels.load_series_1min(cache, name="fake")             # another period


def _ioc_json(path, year_lo, year_hi, code, daily=False, skip_months=()):
    """A fake IOC chunk: three samples per year (production-like thin
    chunks), or with ``daily`` one sample per day (a downloaded year that
    passes the coverage rule) except in ``skip_months``."""
    rows = []
    for y in range(year_lo, year_hi + 1):
        if daily:
            for d in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D"):
                if d.month not in skip_months:
                    rows.append({"stime": f"{d:%Y-%m-%d} 10:00:00",
                                 "slevel": str(0.01 * d.dayofyear), "sensor": "rad"})
            continue
        for k in range(3):
            rows.append({"stime": f"{y}-0{k + 1}-15 10:0{k}:00", "slevel": str(0.1 * k + y % 7),
                         "sensor": "rad"})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def test_v2_fetch_ioc_fills_its_own_cache_and_union_provider_needs_it(tmp_path, monkeypatch):
    import pyintertidal.boundary as bmod
    import pyintertidal.net as net
    from pyintertidal.gauges import load_cached_ioc

    cfg = config.site_config("escalda", 10)
    codes = levels.boundary_codes(cfg, STATIONS)
    assert len(codes) == 2 and "trnz" not in codes
    assert set(codes) <= set(fioc.UNION_CODES)
    prod, union = tmp_path / "gauges", tmp_path / "gauges" / "ioc_2017-2025"
    prod.mkdir()
    for c in codes:
        _ioc_json(prod / f"ioc_{c}_2023-01-01_2025-12-31.json", 2023, 2025, c)
    _ioc_json(prod / f"ioc_{codes[0]}x_2017-01-01_2017-12-31.json", 2017, 2017, "x")   # not ours
    # union_provider refuses an incomplete cache (it would download the union at once)
    with pytest.raises(RuntimeError, match="fetch ioc"):
        levels.union_provider(cfg, cache_dir=str(union), stations_path=STATIONS)
    calls = []

    def fake_fetch(code, t0, t1, cache_dir):
        calls.append((code, t0, t1))
        _ioc_json(os.path.join(cache_dir, f"ioc_{code}_{t0}_{t1}.json"),
                  int(t0[:4]), int(t1[:4]), code, daily=True)

    rep = fioc.fetch_union(codes=codes, cache_dir=str(union), source_cache=str(prod),
                           fetch=fake_fetch)
    assert rep["ready"] and not rep["problems"]
    for c in codes:          # downloaded years assessed; the thin production copy reported
        yrs = rep["codes"][c]["years"]
        assert all(yrs[str(y)]["ok"] and yrs[str(y)]["kind"] == "downloaded"
                   for y in range(2017, 2023))
        assert rep["codes"][c]["production_gaps"] == [2023, 2024, 2025]
        assert yrs["2019"]["samples"] == 365 and yrs["2020"]["days"]["02"] == 29
    assert sorted(calls) == sorted((c, f"{y}-01-01", f"{y}-12-31")
                                   for c in codes for y in range(2017, 2023))
    for c in codes:                    # production's bytes, copied, never re-downloaded
        name = f"ioc_{c}_2023-01-01_2025-12-31.json"
        assert (union / name).read_bytes() == (prod / name).read_bytes()
        assert (union / fioc.marker_name(c)).read_text() == "[]"
        assert rep["codes"][c]["copied"] == [name]
        df = load_cached_ioc(c, cache_dir=str(union))           # the marker adds nothing
        assert sorted(set(df["time"].dt.year)) == list(range(2017, 2026))
        n_days = sum(len(pd.date_range(f"{y}-01-01", f"{y}-12-31")) for y in range(2017, 2023))
        assert len(df) == n_days + 3 * 3
    assert os.path.isfile(union / fioc.MANIFEST_NAME)
    assert not [f for f in os.listdir(prod) if f.startswith("ioc_") and "2017-01-01_2025" in f]
    rep2 = fioc.fetch_union(codes=codes, cache_dir=str(union), source_cache=str(prod),
                            fetch=fake_fetch)
    assert len(calls) == 12 and rep2["ready"]                    # nothing downloaded again
    with pytest.raises(ValueError):
        fioc.fetch_union(codes=codes, cache_dir=str(prod), source_cache=str(prod),
                         fetch=fake_fetch)
    seen = {}
    monkeypatch.setattr(net, "use_system_certificates", lambda: True)
    monkeypatch.setattr(bmod, "make_boundary", lambda aoi, **kw: seen.update(kw) or "consensus")
    assert levels.union_provider(cfg, cache_dir=str(union), stations_path=STATIONS) == "consensus"
    assert seen == dict(tide_model="EOT20", n_gauges=2, period=("2017-01-01", "2025-12-31"),
                        directory="tide_models", cache_dir=str(union), exclude=["trnz"])
    # the v13 provider call is unchanged
    levels.make_provider(cfg)
    assert seen["period"] == ("2023-01-01", "2025-12-31") and seen["cache_dir"] == "data_v4/gauges"


def test_v2_fetch_ioc_refuses_empty_or_thin_years_unless_accepted(tmp_path, monkeypatch, capsys):
    """A year counts as covered by its DATA, not by the file name (fetch_ioc
    writes its yearly file even after an outage): an empty or thin
    downloaded year is renamed *.incomplete-<utc>, downloaded again, and
    without --accept-gaps the cache is not ready (no marker, CLI exit 1)."""
    prod, union = tmp_path / "gauges", tmp_path / "union"
    prod.mkdir()
    codes = ("aaa",)
    _ioc_json(prod / "ioc_aaa_2023-01-01_2025-12-31.json", 2023, 2025, "aaa")
    calls = []

    def outage(code, t0, t1, cache_dir):          # 2019: an outage, then a real gap
        calls.append(int(t0[:4]))
        path = os.path.join(cache_dir, f"ioc_{code}_{t0}_{t1}.json")
        if t0.startswith("2019"):
            if calls.count(2019) == 1:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("[]")
            else:
                _ioc_json(path, 2019, 2019, code, daily=True, skip_months=(5,))
        else:
            _ioc_json(path, int(t0[:4]), int(t1[:4]), code, daily=True)

    rep = fioc.fetch_union(codes=codes, cache_dir=str(union), source_cache=str(prod),
                           fetch=outage)
    r = rep["codes"]["aaa"]
    assert calls.count(2019) == 2 and not rep["ready"]
    assert [e["year"] for e in r["incomplete"]] == [2019, 2019]
    assert r["incomplete"][0]["samples"] == 0 and r["incomplete"][1]["thin"] == ["05"]
    assert r["missing"] == [2019] and r["marker"] is None
    assert not os.path.exists(union / fioc.marker_name("aaa"))
    left = sorted(f for f in os.listdir(union) if ".incomplete-" in f)
    assert len(left) == 2 and all(f.startswith("ioc_aaa_2019-01-01_2019-12-31.json") for f in left)
    assert all(".incomplete" not in p for p in fioc.chunks("aaa", str(union)))
    ready, problems = fioc.union_ready(codes, str(union))
    assert not ready and any("2019" in p for p in problems)
    # the gap is real: accepted after looking at the counts (recorded)
    rep2 = fioc.fetch_union(codes=codes, cache_dir=str(union), source_cache=str(prod),
                            fetch=outage, accept_gaps=True)
    r2 = rep2["codes"]["aaa"]
    assert rep2["ready"] and r2["accepted_gaps"] == [2019] and rep2["accept_gaps"]
    assert r2["years"]["2019"]["thin"] == ["05"] and r2["marker"] == fioc.marker_name("aaa")
    assert r2["production_gaps"] == [2023, 2024, 2025]
    assert fioc.union_ready(codes, str(union))[0]
    # a cache never assessed by this fetch (an older manifest) is not ready
    man = fioc.read_manifest(str(union))
    del man["codes"]["aaa"]["years"]
    assert not fioc.union_ready(codes, str(union), manifest=man)[0]
    # the CLI exits 1 while the union cache is not ready
    from experiments import validation_v2 as cli

    class FakeIOC:
        UNION_CODES, UNION_CACHE, MANIFEST_NAME = ("aaa",), str(union), fioc.MANIFEST_NAME

        @staticmethod
        def fetch_union(accept_gaps=False):
            return {"codes": {"aaa": {"years": {"2019": {"samples": 0, "days": {}, "ok": False,
                                                          "thin": ["01"]}},
                                      "missing": [2019]}},
                    "ready": False, "problems": ["aaa: years [2019] not in the cache"]}

    monkeypatch.setattr(cli, "_module", lambda name: FakeIOC)
    import argparse
    assert cli._fetch_ioc(argparse.Namespace(dry_run=False, accept_gaps=False)) == 1
    assert "NOT READY" in capsys.readouterr().out


def test_v2_boundary_coverage_counts_model_only_instants():
    from pyintertidal.boundary import EnsembleBoundary, GaugeBoundary, InterpolatedGauges

    class Model:
        model = "EOT20"

        def levels(self, times):
            return np.zeros(len(pd.DatetimeIndex(times)))

    tt = pd.date_range("2024-01-01", periods=24 * 60, freq="min")
    g = GaugeBoundary(tt[:600], np.ones(600), name="vlis")       # only the first 10 h
    ens = EnsembleBoundary([Model(), InterpolatedGauges([g], [5.0])])
    t = pd.DatetimeIndex(["2024-01-01 02:00", "2024-01-01 09:00", "2024-01-01 15:00",
                          "2024-01-01 20:00"])
    cov = levels.boundary_coverage(ens, t, windows={"a": [0, 1], "b": [2, 3]})
    assert cov["model_only"] == 2 and cov["share"] == 0.5
    assert cov["windows"] == {"a": 0.0, "b": 1.0}
    assert levels.boundary_coverage(Model(), t)["gauge_parts"] == 0


def test_v2_record_instants_are_the_catalogue_times_unless_a_file_is_given(tmp_path, capsys):
    """User decision of 2026-10-14: the catalogue (product-name) times are the
    default, silently; a sensing_times.json beside the stores is NOT picked
    up; it is used only when its path is given."""
    site_dir = tmp_path / "escalda"
    site_dir.mkdir()
    labels = {"2024-03-01": "2024-03-01T10:47:31Z", "2024-03-04": "2024-03-04T10:57:29Z"}
    (site_dir / "overpass_times.json").write_text(json.dumps(labels))
    sens = {"dates": {"2024-03-01": {"instant": "2024-03-01T10:52:02.5Z"},
                      "2024-03-04": {"instant": None}}}
    (site_dir / "sensing_times.json").write_text(json.dumps(sens))
    r = levels.record_instants("escalda", root=str(tmp_path))
    assert r["source"] == "catalogue" and "flag" not in r
    assert r["path"].endswith("escalda/overpass_times.json")
    out = capsys.readouterr().out
    assert "LABEL" not in out and "!!!" not in out
    assert not hasattr(levels, "LABEL_FLAG")
    assert list(r["t"]) == [pd.Timestamp("2024-03-01 10:47:31"), pd.Timestamp("2024-03-04 10:57:29")]
    r2 = levels.record_instants("escalda", root=str(tmp_path), instants="labels")
    assert r2["source"] == "catalogue"
    r = levels.record_instants("escalda", root=str(tmp_path),
                               instants=str(site_dir / "sensing_times.json"))
    assert r["source"] == "sensing_times"
    assert r["dates"] == ["2024-03-01"] and r["dropped"] == ["2024-03-04"]
    assert r["t"][0] == pd.Timestamp("2024-03-01 10:52:02.5")
    for mode in ("sensing", "auto"):
        with pytest.raises(ValueError, match="no longer a mode"):
            levels.record_instants("escalda", root=str(tmp_path), instants=mode)
    with pytest.raises(FileNotFoundError, match="catalogue"):
        levels.record_instants("ems", root=str(tmp_path))


def test_v2_check_use_keeps_level_sources_out_of_lags_and_lg_out_of_families():
    assert levels.check_use("LB", "lag") and levels.check_use("LG", "elevation")
    for src in ("LT", "LM", "LG"):
        with pytest.raises(ValueError):
            levels.check_use(src, "lag")
    with pytest.raises(ValueError):
        levels.check_use("LG", "confirmatory")
    assert levels.check_use("LM", "confirmatory")
    assert set(levels.SOURCES) == set(settings.LEVEL)
    lg = settings.LEVEL["LG"]
    assert lg.upper_bound and not lg.confirmatory
    # the registry's impl names resolve to this module's builders
    for e in settings.LEVEL.values():
        for name in e.impl:
            assert callable(getattr(levels, name.rsplit(".", 1)[1]))


# ─────────────────────────────────────────────────────────────────────────────
#  LT: eo-tides nodes
# ─────────────────────────────────────────────────────────────────────────────

def test_v2_lt_models_and_rule():
    local = ["EOT20", "GOT4.10_SAL", "GOT4.10_nc", "GOT4.8_nc"]
    m = ftides.lt_models(available=local)
    assert m == [("EOT20", "EOT20"), ("GOT4.10", "GOT4.10_nc")]          # GOT4.8 superseded
    assert ftides.lt_rule(m) == "primary"
    m3 = ftides.lt_models(available=local + ["FES2022"])
    assert [x[0] for x in m3] == ["EOT20", "FES2022", "GOT4.10"]
    assert ftides.lt_rule(m3) == "ensemble"
    with pytest.raises(RuntimeError):
        ftides.lt_models(available=["GOT4.10_nc"])
    assert levels.LEVEL_RULES["LT"]["top_n"] == ftides.TOP_N
    assert levels.LEVEL_RULES["LT"]["node_km"] == settings.LEVEL["LT"].params["node_km"] == \
        ftides.NODE_M / 1000
    assert levels.LEVEL_RULES["LM"]["bin_km"] == settings.LEVEL["LM"].params["bin_km"]
    assert levels.LEVEL_RULES["LM"]["unresolved_km"] == settings.LEVEL["LM"].params["unresolved_km"]


def test_v2_lt_on_a_tiny_grid_equals_pytmd_at_the_nodes():
    """Real eo-tides with EOT20 on a 1.5 x 1.2 km frame at the Westerschelde mouth."""
    pytest.importorskip("eo_tides")
    if not os.path.isdir(os.path.join(ROOT, "tide_models", "EOT20")):
        pytest.skip("EOT20 is not installed")
    grid = _grid(nx=150, ny=120, res=10.0)
    t = pd.DatetimeIndex(["2023-03-01 10:50:12", "2024-06-02 10:57:01", "2025-01-05 11:01:00"])
    pix = np.arange(0, 150 * 120, 7, dtype=np.int64)
    lt = levels.level_lt(grid, t, pix, site="escalda", models=[("EOT20", "EOT20")],
                         directory=os.path.join(ROOT, "tide_models"))
    assert lt.computable and lt.flags["rule"] == "primary" and not lt.demean
    xn, yn = lt.groups["node_x"], lt.groups["node_y"]
    # every pixel takes its nearest node (brute force over the used nodes)
    W = 150
    px, py = grid["xs"][pix % W], grid["ys"][pix // W]
    d = np.hypot(px[:, None] - xn[None, :], py[:, None] - yn[None, :])
    assert np.allclose(d[np.arange(len(pix)), lt.g0], d.min(axis=1))
    assert np.all(np.diff(np.unique(np.round(xn))) >= 999)        # 1 km nodes
    L = lt.pixel_levels(pix[:5])
    assert L.shape == (5, 3) and np.isfinite(L).all() and np.abs(L).max() < 4
    gate = ftides.check_points(np.c_[xn, yn], UTM31, t, lt.H, "EOT20",
                               directory=os.path.join(ROOT, "tide_models"), n_points=2)
    assert gate["pass"], gate


def _fake_pixel_tides(values):
    """A pixel_tides stand-in with eo-tides' node grid; ``values[model]`` gives
    the level as a function of (x, y, hours since 2024)."""
    import odc.geo.xr  # noqa: F401
    import xarray as xr
    from odc.geo.geobox import GeoBox

    calls = []

    def pixel_tides(gb, time, model, directory, resample, resolution, buffer, extrapolate,
                    cutoff, parallel):
        calls.append(dict(n=len(time), model=list(model), resample=resample,
                          resolution=resolution, buffer=buffer, extrapolate=extrapolate,
                          cutoff=cutoff, parallel=parallel))
        rg = GeoBox.from_bbox(bbox=gb.buffered(buffer).boundingbox, resolution=resolution)
        z = odc.geo.xr.xr_zeros(rg)
        x, y = z["x"].values, z["y"].values
        hrs = (pd.DatetimeIndex(time) - pd.Timestamp("2024-01-01")) / pd.Timedelta(hours=1)
        X, Y = np.meshgrid(x, y)
        arr = np.stack([np.stack([values[m](X, Y, h) for h in hrs]) for m in model])
        return xr.DataArray(arr.astype(np.float32), dims=("tide_model", "time", "y", "x"),
                            coords={"tide_model": list(model), "time": pd.DatetimeIndex(time),
                                    "y": y, "x": x})

    return pixel_tides, calls


def test_v2_lt_ensemble_needs_a_ranking_then_combines_members():
    vals = {"EOT20": lambda X, Y, h: 1.0 + 0 * X + 0.01 * h,
            "FES2022": lambda X, Y, h: 2.0 + (X - 540000) / 1e5 + 0 * h,
            "GOT4.10_nc": lambda X, Y, h: 3.0 + 0 * X + 0 * h}
    pt, calls = _fake_pixel_tides(vals)
    grid = _grid(nx=100, ny=40, res=50.0)
    t = pd.DatetimeIndex(["2024-01-01 10:00", "2024-01-02 11:00", "2024-01-05 10:30"])
    pix = np.arange(0, 4000, 3, dtype=np.int64)
    models = [("EOT20", "EOT20"), ("FES2022", "FES2022"), ("GOT4.10", "GOT4.10_nc")]
    lt = levels.level_lt(grid, t, pix, site="escalda", models=models, pixel_tides=pt)
    assert calls[0]["resample"] is False and calls[0]["resolution"] == 1000.0 and \
        calls[0]["buffer"] == 1000.0 and calls[0]["cutoff"] == np.inf and \
        calls[0]["extrapolate"] is True and calls[0]["parallel"] is False
    assert lt.status == "not computable" and "ranking" in lt.reason
    assert lt.unit_status()["status"] == "NOT COMPUTABLE"
    with pytest.raises(levels.NotComputable):
        lt.pixel_levels(pix[:3])
    # BT26-rank: the model choice (top 1)
    bt = levels.lt_with_ranking(lt, {"kind": "bt26", "ranks": {"EOT20": 2, "FES2022": 1,
                                                               "GOT4.10": 3}})
    assert bt.computable and bt.flags["top_n"] == 1
    want = lt.extra["members"][1]
    assert np.allclose(bt.H, want)
    # eo-tides' rule: the equal-weight mean of the top 3
    eo = levels.lt_with_ranking(lt, {"kind": "eo-tides", "ranks": {"EOT20": 1, "FES2022": 2,
                                                                   "GOT4.10": 3}})
    assert np.allclose(eo.H, lt.extra["members"].mean(axis=0))
    # per-node ranks
    G = lt.extra["members"].shape[1]
    r = np.tile([1.0, 2.0, 3.0], (G, 1))
    r[0] = [3.0, 2.0, 1.0]
    pn = levels.lt_with_ranking(lt, {"kind": "bt26", "ranks": r})
    assert np.allclose(pn.H[0], lt.extra["members"][2][0])
    assert np.allclose(pn.H[1:], lt.extra["members"][0][1:])
    with pytest.raises(ValueError):
        levels.lt_with_ranking(lt, {"kind": "median", "ranks": {}})
    # time chunks do not change the values
    a = ftides.node_tides(grid, t, models[:1], pixel_tides=pt, time_chunk=1)
    b = ftides.node_tides(grid, t, models[:1], pixel_tides=pt, time_chunk=256)
    assert np.array_equal(a["H"], b["H"])


# ─────────────────────────────────────────────────────────────────────────────
#  LM: the regional SSH through a mocked toolbox
# ─────────────────────────────────────────────────────────────────────────────

LM_DS = "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT1H-i"
LM_T0 = pd.Timestamp("2025-01-01")


def _zos(lon, lat, tt):
    """A cubic in time per cell (the 4-point stencil reproduces it exactly)."""
    u = np.asarray((pd.DatetimeIndex(tt) - LM_T0) / pd.Timedelta(days=1), float)
    a = 0.5 + 3.0 * (lon - 3.5)
    return (a[None] + 0.01 * u[:, None, None] + 1e-4 * u[:, None, None] ** 2
            - 2e-7 * u[:, None, None] ** 3 + 0 * lat[None])


class FakeToolbox:
    """copernicusmarine stand-in: ``subset`` writes a NetCDF on a 0.02 x 0.0135 deg
    grid covering the box; cells east of ``land_lon`` and north of ``land_lat``
    are land (NaN)."""

    __version__ = "fake-2.4.1"

    def __init__(self, land_lon=3.61, land_lat=51.46, raise_exc=None):
        self.calls, self.land_lon, self.land_lat, self.raise_exc = [], land_lon, land_lat, raise_exc

    def subset(self, **kw):
        import xarray as xr

        self.calls.append(kw)
        if self.raise_exc is not None:
            raise self.raise_exc
        lon = np.arange(np.floor(kw["minimum_longitude"] / 0.02) * 0.02,
                        kw["maximum_longitude"] + 0.02, 0.02)
        lat = np.arange(np.floor(kw["minimum_latitude"] / 0.0135) * 0.0135,
                        kw["maximum_latitude"] + 0.0135, 0.0135)
        tt = pd.date_range(kw["start_datetime"], kw["end_datetime"], freq="1h")
        LON, LAT = np.meshgrid(lon, lat)
        z = _zos(LON, LAT, tt)
        z[:, (LON > self.land_lon) | (LAT > self.land_lat)] = np.nan
        ds = xr.Dataset({"zos": (("time", "latitude", "longitude"), z.astype(np.float64))},
                        coords={"time": tt, "latitude": lat, "longitude": lon})
        ds.to_netcdf(os.path.join(kw["output_directory"], kw["output_filename"]))

    def login(self, **kw):
        return True


def _credentials(path, user="u", pwd="p"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"[credentials]\nusername={user}\npassword={pwd}\n")
    return path


def test_v2_cmems_catalogue_archive_depth_and_choice():
    assert set(cmems.SITE_MODEL) == set(config.SITES)
    # AMM15 starts 2024-08-04: not W3 at the Dutch sites, W1 (2025) yes, at 15 min
    did, why = cmems.choose_dataset("escalda", "2023-01-01", "2026-01-01")
    assert did is None and "2024-08-04" in why and "NOT AVAILABLE" in why
    assert cmems.choose_dataset("ems", "2025-01-01", "2026-01-01")[0] == \
        "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT15M-i"
    # AMM7 (7 km) is never substituted unless the user lets it in
    assert cmems.choose_dataset("escalda", "2023-01-01", "2026-01-01",
                                include_ineligible=True)[0] == \
        "cmems_mod_nws_phy-ssh_my_7km-2D_PT1H-i"
    # the rías: IBI reanalysis (hourly means) for W3, IBI 15-min for W1
    assert cmems.choose_dataset("villaviciosa", "2023-01-01", "2026-01-01")[0] == \
        "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m"
    assert cmems.choose_dataset("ferrol", "2025-01-01", "2026-01-01")[0] == \
        "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT15M-i"
    assert cmems.DEGENERATE == ("villaviciosa",)


def test_v2_cmems_credentials_missing_is_not_computable_never_zero(tmp_path):
    grid = _grid()
    t = pd.DatetimeIndex(["2025-03-01 10:50", "2025-03-04 11:00"])
    tb = FakeToolbox()
    cred = str(tmp_path / ".config" / "copernicusmarine" / ".copernicusmarine-credentials")
    assert cmems.CREDENTIALS_FILE == ".config/copernicusmarine/.copernicusmarine-credentials"
    man = cmems.fetch_site("escalda", grid, [2025], LM_DS, str(tmp_path / "cmems"),
                           toolbox=tb, credentials_file=cred)
    assert man["status"] == "credentials missing" and "does not exist" in man["reason"]
    assert tb.calls == []                                      # no request, no prompt
    assert cmems.read_manifest(str(tmp_path / "cmems"))["status"] == "credentials missing"
    lm = levels.level_lm("escalda", t, np.arange(10), np.linspace(0, 4, 10), grid, [],
                         np.zeros(2), LM_DS, manifest=man)
    assert lm.status == "credentials missing" and lm.H is None
    assert lm.unit_status() == {"source": "LM", "status": "NOT COMPUTABLE",
                                "reason": man["reason"]}
    with pytest.raises(levels.NotComputable):
        lm.pixel_levels(np.arange(3))
    # a file without a password
    _credentials(cred, pwd="")
    assert cmems.check_credentials(cred)["status"] == "credentials missing"
    # rejected by the server (the toolbox's exception class name)

    class InvalidUsernameOrPassword(Exception):
        pass

    _credentials(cred)
    bad = FakeToolbox(raise_exc=InvalidUsernameOrPassword("no"))
    man2 = cmems.fetch_site("escalda", grid, [2025], LM_DS, str(tmp_path / "cmems2"),
                            toolbox=bad, credentials_file=cred)
    assert man2["status"] == "credentials missing" and len(bad.calls) == 1
    net = FakeToolbox(raise_exc=ConnectionError("down"))
    man3 = cmems.fetch_site("escalda", grid, [2025], LM_DS, str(tmp_path / "cmems3"),
                            toolbox=net, credentials_file=cred)
    assert man3["status"] == "not available"
    lm3 = levels.level_lm("escalda", t, np.arange(3), np.ones(3), grid, [], np.zeros(2), LM_DS,
                          manifest=man3)
    assert lm3.unit_status()["status"] == "NOT AVAILABLE"


def test_v2_lm_bins_cells_interpolates_and_demeans(tmp_path):
    from pyproj import Transformer

    grid = _grid()                                             # 5 x 2 km, 50 m pixels
    W = len(grid["xs"])
    pix = np.arange(len(grid["xs"]) * len(grid["ys"]), dtype=np.int64)
    s_km = (grid["xs"][pix % W] - 540000.0) / 1000.0           # mouth on the west edge
    s_km[::97] = np.nan                                        # pixels without s
    rng = np.random.default_rng(3)
    t = pd.DatetimeIndex(sorted(pd.Timestamp("2025-01-01 10:30") + pd.to_timedelta(
        rng.choice(np.arange(360), 20, replace=False), "D")
        + pd.to_timedelta(rng.integers(0, 3600, 20), "s")))
    ref = 0.3 * np.sin(np.arange(20))
    cred = _credentials(str(tmp_path / "cred.ini"))
    tb = FakeToolbox()
    out = str(tmp_path / "common" / "escalda" / "cmems")
    man = cmems.fetch_site("escalda", grid, [2025], LM_DS, out, toolbox=tb,
                           credentials_file=cred)
    assert man["status"] == "ok" and len(tb.calls) == 1
    kw = tb.calls[0]
    assert kw["variables"] == ["zos"] and kw["coordinates_selection_method"] == "outside"
    assert kw["start_datetime"] == "2024-12-31T22:00:00" and kw["end_datetime"] == \
        "2026-01-01T02:00:00"
    assert kw["credentials_file"] == cred
    # cached: a second fetch makes no request
    man_b = cmems.fetch_site("escalda", grid, [2025], LM_DS, out, toolbox=tb,
                             credentials_file=cred)
    assert len(tb.calls) == 1 and man_b["files"] == man["files"]
    files = cmems.manifest_files(out, man)
    lm = levels.level_lm("escalda", t, pix, s_km, grid, files, ref, LM_DS, manifest=man)
    assert lm.computable and lm.demean and not lm.flags["degenerate"]
    # independent expectation: cells, channel matching, bins, analytic cubic
    f = cmems.read_subset(files[0])
    LON, LAT = np.meshgrid(f["lon"], f["lat"])
    wet = np.isfinite(f["zos"]).all(axis=0)
    cx, cy = Transformer.from_crs("EPSG:4326", UTM31, always_xy=True).transform(LON[wet], LAT[wet])
    has = np.isfinite(s_km)
    P = np.c_[grid["xs"][pix[has] % W], grid["ys"][pix[has] // W]]
    bins_px = np.floor(s_km[has]).astype(int)
    exact = _zos(LON[wet][None, :], LAT[wet][None, :], t)[:, 0, :]          # (T, C)
    assert lm.groups["bin"].tolist() == sorted(set(bins_px.tolist())) == [0, 1, 2, 3, 4]
    for k, b in enumerate(lm.groups["bin"]):
        own = []
        for c in range(len(cx)):
            d = np.hypot(P[:, 0] - cx[c], P[:, 1] - cy[c])
            if d.min() <= 1000.0 and bins_px[np.argmin(d)] == b:
                own.append(c)
        mb = bins_px == b
        dpc = np.hypot(P[mb][:, None, 0] - cx[None], P[mb][:, None, 1] - cy[None])
        assert np.isclose(lm.groups["d_wet_km"][k], dpc.min() / 1000.0)
        assert lm.groups["n_cells"][k] == len(own)
        want = exact[:, own].mean(axis=1) if own else exact[:, np.unravel_index(
            np.argmin(dpc), dpc.shape)[1]]
        assert np.allclose(lm.H[k], want, atol=1e-9), b
        assert lm.groups["unresolved"][k] == (dpc.min() > 1000.0)
    # the fixture holds bins with channel cells, a resolved bin without one and
    # unresolved bins (land east of 3.61 E)
    assert (lm.groups["n_cells"] > 0).any() and (lm.groups["n_cells"] == 0).any()
    assert lm.groups["unresolved"].any() and not lm.groups["unresolved"].all()
    assert (~lm.groups["unresolved"] & (lm.groups["n_cells"] == 0)).any()
    # demeaned against h_B over the window, at read time
    win = np.arange(0, 20, 2)
    R = lm.rows(sel=[1, 2], window=win)
    off = (lm.H[:, win] - ref[win]).mean(axis=1)
    assert np.allclose(R, lm.H[:, [1, 2]] - off[:, None])
    with pytest.raises(ValueError):
        lm.rows(sel=[1, 2])                                      # no window
    L = lm.pixel_levels(pix[:200], sel=[0], window=win)
    assert np.isnan(L[0]).all()                                  # pixel 0 has no s
    assert lm.groups["bin"][0] == 0 and L[1, 0] == lm.rows([0], win)[0, 0]   # s 0.075 km
    # an instant outside the archive (2024) is NOT AVAILABLE, never zero
    t2 = t.insert(0, pd.Timestamp("2024-06-01 10:50"))
    lm2 = levels.level_lm("escalda", t2, pix, s_km, grid, files, np.r_[0.0, ref], LM_DS,
                          manifest=man)
    assert lm2.unit_status(window=np.arange(21))["status"] == "NOT AVAILABLE"
    assert lm2.unit_status(window=np.arange(1, 21))["status"] == "ok"
    with pytest.raises(levels.NotAvailable):
        lm2.rows([0, 1], np.arange(21))
    # hourly means need a measured stamp convention
    lm3 = levels.level_lm("villaviciosa", t, pix, s_km, grid, files, ref,
                          "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m")
    assert lm3.status == "not computable" and "stamp" in lm3.reason
    # round trip
    p = str(tmp_path / "levels_LM.npz")
    lm.save(p)
    back = levels.SourceLevels.load(p)
    assert np.array_equal(back.H, lm.H) and np.array_equal(back.g0, lm.g0)
    assert np.array_equal(back.rows([3, 4], win), lm.rows([3, 4], win))
    lm.save(str(tmp_path / "again.npz"))
    assert product.sha256_file(p) == product.sha256_file(str(tmp_path / "again.npz"))


def test_v2_cubic_at_reproduces_cubics_and_marks_gaps():
    tm = pd.date_range("2025-01-01", periods=48, freq="15min")
    u = np.arange(48) / 4.0
    Z = np.c_[1 + 2 * u - 0.3 * u ** 2 + 0.01 * u ** 3, np.sin(u)]
    q = pd.DatetimeIndex(["2025-01-01 00:07", "2025-01-01 03:00", "2025-01-01 05:52:30",
                          "2024-12-31 23:00", "2025-01-01 11:50"])
    out = levels.cubic_at(tm, Z, q)
    uq = np.asarray((q - tm[0]) / pd.Timedelta(hours=1), float)
    poly = 1 + 2 * uq - 0.3 * uq ** 2 + 0.01 * uq ** 3
    assert np.isnan(out[0, 0])                                  # stencil leaves the axis
    assert np.allclose(out[0, 1:3], poly[1:3], atol=1e-12)
    assert out[0, 1] == Z[12, 0]                                 # on a step: the sample
    assert np.isnan(out[:, 3]).all() and np.isnan(out[:, 4]).all()
    # a missing step becomes NaN for the stencils that need it
    keep = np.ones(48, bool)
    keep[20] = False
    out2 = levels.cubic_at(tm[keep], Z[keep], pd.DatetimeIndex(["2025-01-01 04:40",
                                                                "2025-01-01 08:20"]))
    assert np.isnan(out2[0, 0]) and np.isfinite(out2[0, 1])


def test_v2_measure_stamp_shift_of_hourly_means():
    t15 = pd.date_range("2025-01-01", "2025-01-10", freq="15min")
    h = np.asarray((t15 - t15[0]) / pd.Timedelta(hours=1), float)
    v = 1.5 * np.cos(2 * np.pi * h / 12.42)
    s = pd.Series(v, index=t15)
    # means over [stamp, stamp + 1 h] (stamped at the interval START): shift +30
    stamps = pd.date_range("2025-01-02", "2025-01-08", freq="1h")
    w = np.array([0.5, 1, 1, 1, 0.5]) / 4.0
    mv = np.array([sum(wk * s[st + pd.Timedelta(minutes=15 * k)] for k, wk in enumerate(w))
                   for st in stamps])
    r = cmems.measure_stamp_shift(stamps, mv, t15, v)
    assert r["shift_min"] == 30 and r["rms_m"][30] < 1e-9 < r["rms_m"][0]


# ─────────────────────────────────────────────────────────────────────────────
#  LG: linear in s through the judges (labelled upper bound)
# ─────────────────────────────────────────────────────────────────────────────

def test_v2_lg_interpolates_along_s_reproduces_each_judge_and_fills_gaps(tmp_path):
    T = 30
    t = pd.date_range("2024-02-01 10:50", periods=T, freq="3D")
    ref = 1.8 * np.cos(np.arange(T) * 0.7)
    j1 = 1.2 * ref + 0.35                                         # another datum
    j2 = 1.5 * np.roll(ref, 1) - 0.2
    j1m = j1.copy()
    j1m[4] = np.nan                                               # missing at one instant
    knots = [{"code": "far", "s_km": 8.0, "values": j2},
             {"code": "near", "s_km": 3.0, "values": j1m}]
    s = np.array([-1.0, 0.0, 1.5, 3.0, 5.5, 8.0, 12.0, np.nan])
    pix = np.arange(len(s), dtype=np.int64) * 10
    lg = levels.level_lg("escalda", t, ref, knots, pix, s)
    assert lg.computable and lg.flags["upper_bound"] and not lg.flags["confirmatory"]
    assert lg.groups["s_km"].tolist() == [0.0, 3.0, 8.0]
    win = np.arange(T)
    off1 = np.nanmean(j1m - ref)
    off2 = np.mean(j2 - ref)
    L = lg.pixel_levels(window=win)
    J1, J2 = j1 - off1, j2 - off2
    ok = np.arange(T) != 4
    assert np.array_equal(L[0], ref) and np.array_equal(L[1], ref)          # s <= 0: h_B
    assert np.array_equal(L[3][ok], J1[ok])                                  # at s_g: the judge
    assert np.array_equal(L[5], J2) and np.array_equal(L[6], J2)             # constant beyond
    assert np.allclose(L[2][ok], 0.5 * ref[ok] + 0.5 * J1[ok])
    assert np.allclose(L[4][ok], 0.5 * J1[ok] + 0.5 * J2[ok])
    assert np.isnan(L[7]).all()                                              # no s
    # the missing judge is filled by the same interpolation (h_B -> far judge)
    assert np.isclose(L[3][4], ref[4] + 3.0 / 8.0 * (J2[4] - ref[4]))
    # native datum: no demeaning
    lgn = levels.level_lg("escalda", t, ref, knots, pix, s, demean=False)
    assert np.array_equal(lgn.pixel_levels(sel=[0, 1])[3], j1[[0, 1]])
    # per-band polygon series
    bands, ser = lg.band_series(pix[1:7], np.array([0, 0, 1, 1, 2, 4]), window=win)
    assert bands.tolist() == [0, 1, 2, 4]
    assert np.allclose(ser[0], (L[1] + L[2]) / 2) and np.allclose(ser[3], L[6])
    # not applicable outside the judged Dutch sites
    lf = levels.level_lg("ferrol", t, ref, knots, pix, s)
    assert lf.unit_status()["status"] == "not_applicable"
    with pytest.raises(ValueError):
        levels.level_lg("escalda", t, ref, knots + [{"code": "dup", "s_km": 3.0,
                                                     "values": j1}], pix, s)
    p = str(tmp_path / "lg.npz")
    lg.save(p)
    back = levels.SourceLevels.load(p)
    assert np.array_equal(back.pixel_levels(window=win), L, equal_nan=True)


def test_v2_gauge_at_interpolates_within_the_gap_rule():
    idx = pd.DatetimeIndex(["2024-01-01 10:00", "2024-01-01 10:10", "2024-01-01 10:20",
                            "2024-01-01 10:50"])
    ser = pd.Series([1.0, 2.0, np.nan, 4.0], index=idx)
    t = pd.DatetimeIndex(["2024-01-01 10:05", "2024-01-01 10:10", "2024-01-01 10:15",
                          "2024-01-01 10:30", "2024-01-01 09:55"])
    got = levels.gauge_at(ser, t, max_gap_min=10)
    assert got[0] == 1.5 and got[1] == 2.0
    assert np.isnan(got[2]) and np.isnan(got[3]) and np.isnan(got[4])      # gap / outside
    assert np.isclose(levels.gauge_at(ser, t, max_gap_min=40)[3], 2.0 + 2.0 * 20 / 40)


def test_v2_source_levels_validation():
    t = pd.date_range("2025-01-01", periods=3, freq="D")
    with pytest.raises(ValueError):
        levels.SourceLevels("LX", "s", t, H=np.zeros((1, 3)))
    with pytest.raises(ValueError):
        levels.SourceLevels("LT", "s", t, H=np.zeros((2, 3)))                 # uniform needs 1 row
    with pytest.raises(ValueError):
        levels.SourceLevels("LM", "s", t, H=np.zeros((1, 3)), pix=[0, 1], g0=[0, 0],
                            demean=True)                                       # no ref
    with pytest.raises(ValueError):
        levels.SourceLevels.unavailable("LM", "s", t, "ok", "")
    src = levels.SourceLevels("LT", "s", t, H=np.arange(6.0).reshape(2, 3), pix=[2, 5],
                              g0=[1, 0])
    assert src.pixel_levels([5, 2]).tolist() == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(KeyError):
        src.pixel_levels([3])
    with pytest.raises(ValueError):
        src.rows(sel=[2, 1])


# ─────────────────────────────────────────────────────────────────────────────
#  the reviews of 2026-10-14: the CMEMS campaign per (site, window), LG centring
# ─────────────────────────────────────────────────────────────────────────────

class CampaignToolbox(FakeToolbox):
    """FakeToolbox with the dataset's own time step: 15-min instantaneous
    values for the PT15M datasets and, for an hourly-MEAN dataset, the mean
    over [stamp, stamp + 1 h] (stamped at the interval START: shift +30)."""

    def subset(self, **kw):
        import xarray as xr

        self.calls.append(kw)
        ds_id = kw["dataset_id"]
        assert isinstance(ds_id, str) and ds_id in cmems.DATASETS, ds_id
        lon = np.arange(np.floor(kw["minimum_longitude"] / 0.02) * 0.02,
                        kw["maximum_longitude"] + 0.02, 0.02)
        lat = np.arange(np.floor(kw["minimum_latitude"] / 0.0135) * 0.0135,
                        kw["maximum_latitude"] + 0.0135, 0.0135)
        step = "15min" if "PT15M" in ds_id else "1h"
        tt = pd.date_range(kw["start_datetime"], kw["end_datetime"], freq=step)
        LON, LAT = np.meshgrid(lon, lat)
        if ds_id.endswith("-m"):
            w = (0.5, 1.0, 1.0, 1.0, 0.5)
            z = sum(wk / 4.0 * _zos(LON, LAT, tt + pd.Timedelta(minutes=15 * k))
                    for k, wk in enumerate(w))
        else:
            z = _zos(LON, LAT, tt)
        ds = xr.Dataset({"zos": (("time", "latitude", "longitude"), z.astype(np.float64))},
                        coords={"time": tt, "latitude": lat, "longitude": lon})
        ds.to_netcdf(os.path.join(kw["output_directory"], kw["output_filename"]))


def test_v2_cmems_plan_per_window_never_substitutes():
    p = cmems.plan_site("escalda")
    ds15 = "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT15M-i"
    assert {w: e["dataset"] for w, e in p["windows"].items()} == {
        "W3": None, "W2": None, "W1": ds15, "Y2023": None, "Y2024": None, "Y2025": ds15}
    assert "never substituted" in p["windows"]["W3"]["reason"]
    assert p["fetch"] == {ds15: [2025]} and p["stamp"] == {}
    f = cmems.plan_site("ferrol")
    my, an = "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m", "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT15M-i"
    assert f["windows"]["W3"]["dataset"] == my and f["windows"]["W1"]["dataset"] == an
    assert f["fetch"] == {my: [2023, 2024, 2025], an: [2025]}
    assert f["stamp"] == {my: an}                    # hourly means: measured against 15 min


def test_v2_cmems_campaign_fetches_per_dataset_and_measures_the_stamp(tmp_path, monkeypatch):
    """The CLI passed choose_dataset's (id, reason) TUPLE as the dataset id
    and downloaded nothing (review of 2026-10-14): the campaign passes the id,
    one folder and manifest per dataset, and measures the hourly-mean stamp
    convention (+30 min here) on the overlap month."""
    monkeypatch.setattr(cmems, "PAD_KM", 0.5)
    grid = _grid(nx=10, ny=10)
    cred = _credentials(str(tmp_path / "cred.ini"))
    tb = CampaignToolbox()
    root = str(tmp_path / "common")
    res = cmems.fetch_campaign("ferrol", grid, ("Y2024",), root=root, toolbox=tb,
                               credentials_file=cred)
    my, an = "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m", "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT15M-i"
    assert res["status"] == "ok"
    assert {(c["dataset_id"], c["output_filename"][-7:-3]) for c in tb.calls} == {
        (my, "2024"), (my, "2025"), (an, "2025")}
    for ds in (my, an):
        d = cmems.site_cache_dir("ferrol", root, ds)
        assert res["datasets"][ds]["dir"] == d and cmems.read_manifest(d)["dataset"] == ds
    st = res["stamp_shift"][my]
    assert st["status"] == "measured" and st["shift_min"] == 30 and st["rms_m"][30] < 1e-9
    d_my = cmems.site_cache_dir("ferrol", root, my)
    assert cmems.read_stamp_shift(d_my) == 30.0
    # a re-fetched mean file invalidates the measurement
    man = cmems.read_manifest(d_my)
    man["files"]["2025"]["sha256"] = "0" * 64
    with open(os.path.join(d_my, cmems.MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(man, f)
    assert cmems.read_stamp_shift(d_my) is None
    # without credentials: no request, the status says so, no stale measurement kept
    tb2 = CampaignToolbox()
    res2 = cmems.fetch_campaign("ferrol", grid, ("Y2024",), root=root, toolbox=tb2,
                                credentials_file=str(tmp_path / "missing.ini"))
    assert res2["status"] == "credentials missing" and tb2.calls == []
    assert res2["stamp_shift"][my]["shift_min"] is None
    assert not os.path.exists(os.path.join(d_my, cmems.STAMP_SHIFT_NAME))


def test_v2_cli_fetch_cmems_exit_codes(tmp_path, monkeypatch, capsys):
    import argparse

    from experiments import validation_v2 as cli
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs

    class DS:
        def close(self):
            pass

    monkeypatch.setenv("VALIDATION_V2_ROOT", str(tmp_path / "v2"))
    monkeypatch.setattr(zc, "open_cube", lambda *a, **k: DS())
    monkeypatch.setattr(inputs, "grid_of", lambda ds: _grid())
    monkeypatch.setattr(cmems, "check_credentials", lambda *a, **k: {"status": "ok"})
    seen = []

    def campaign(status):
        def run(site, grid, windows, root=None, **kw):
            seen.append((site, tuple(windows), root))
            return {"status": status, "datasets": {"d": {"years": [2025], "status": status,
                                                         "reason": "", "dir": "x"}}}
        return run

    args = argparse.Namespace(sites="ferrol", years=None, dry_run=False, cubes_root=None)
    for status, rc in (("ok", 0), ("not available", 1), ("credentials missing", 3)):
        monkeypatch.setattr(cmems, "fetch_campaign", campaign(status))
        assert cli._fetch_cmems(args) == rc, status
    site, wins, root = seen[0]
    assert site == "ferrol" and wins == cmems.CAMPAIGN1_WINDOWS
    assert root.replace("\\", "/").endswith("v2/common")
    out = capsys.readouterr().out
    assert "ferrol W3: cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m" in out
    args = argparse.Namespace(sites="escalda", years="2023-2024", dry_run=False, cubes_root=None)
    monkeypatch.setattr(cmems, "fetch_campaign", campaign("ok"))
    assert cli._fetch_cmems(args) == 0
    assert seen[-1][1] == ("Y2023", "Y2024")
    assert "escalda Y2023: NOT AVAILABLE" in capsys.readouterr().out


def test_v2_lg_knots_centre_the_stamps_and_drop_judges_without_s(tmp_path, monkeypatch):
    """LG reads each judge exactly as the lag truths do: centred with its
    provider's time-stamp convention before the 10-min centring; a judge
    whose s_g is not a finite positive distance is dropped."""
    from experiments.benchmark import judges
    monkeypatch.setitem(config.JUDGES, "escalda", ("ja", "jb"))
    rows = [{"code": c, "name": c, "provider": "END2", "lon": 3.6, "lat": 51.4, "datum": "x",
             "sampling_min": 1, "file": f"{c}.csv", "period": "2024", "site": "escalda",
             "in_boundary": False} for c in ("ja", "jb")]
    stations = tmp_path / "stations.json"
    stations.write_text(json.dumps(rows), encoding="utf-8")
    t0 = pd.Timestamp("2024-03-01 00:00")
    idx = pd.date_range(t0, periods=6 * 60, freq="1min")
    for c in ("ja", "jb"):
        pd.DataFrame({"time_utc": idx.strftime("%Y-%m-%d %H:%M:%S"),
                      "level_m": np.arange(len(idx), dtype=float)}).to_csv(
            tmp_path / f"{c}.csv", index=False)
    conv = {"END2": {"averaging_min": 2.0, "stamp": "end", "status": "test", "source": "t"}}
    t = pd.DatetimeIndex(["2024-03-01 01:00:00", "2024-03-01 02:33:00", "2024-03-01 04:10:30"])
    knots = levels.lg_knots("escalda", t, {"ja": 2.5, "jb": float("nan")}, base=str(tmp_path),
                            path=str(stations), conventions=conv)
    assert [k["code"] for k in knots] == ["ja"]
    minutes = np.asarray((t - t0) / pd.Timedelta(minutes=1), float)
    # an end stamp of a 2-min mean moves 1 min back: the value at t is minutes(t) + 1
    assert np.allclose(knots[0]["values"], minutes + 1.0, atol=1e-9)
    assert judges.stamp_shift_min(rows[0], conv) == -1.0
    with pytest.raises(KeyError, match="time-stamp convention"):
        levels.lg_knots("escalda", t, {"ja": 2.5}, base=str(tmp_path), path=str(stations))
    lg = levels.level_lg("escalda", t, np.zeros(3), knots + [
        {"code": "jz", "s_km": float("nan"), "values": np.zeros(3)}], np.arange(4),
        np.array([0.0, 1.0, 2.5, 4.0]))
    assert lg.flags["dropped_knots"] == ["jz"] and lg.flags["n_knots"] == 1


def test_v2_gauge_band_lags_place_the_judges_lags_along_s():
    """User decision of 2026-10-15: the E0g band lags are the judges'
    measured lags placed along s as LG places their levels: tau(0) = 0,
    linear between judges, constant beyond the last one, NaN kept."""
    r = levels.gauge_band_lags({"a": 20.0, "c": 60.0, "b": 40.0},
                               {"a": 10.0, "b": 20.0, "c": 30.0},
                               [-2.0, 0.0, 5.0, 10.0, 15.0, 25.0, 30.0, 45.0, np.nan])
    assert r["status"] == "ok" and r["codes"] == ["a", "b", "c"]
    assert not r["monotone_adjusted"] and r["knots_tau_min"] == [20.0, 40.0, 60.0]
    np.testing.assert_allclose(r["tau_band"][:-1], [0, 0, 10, 20, 30, 50, 60, 60])
    assert np.isnan(r["tau_band"][-1])
    assert r["rule"] is levels.GAUGE_LAG_RULE


def test_v2_gauge_band_lags_monotone_pairs_and_dropped_judges():
    # a decreasing pair is pooled (isotonic), a negative lag is clipped at the anchor
    r = levels.gauge_band_lags({"n": -10.0, "a": 30.0, "b": 20.0, "c": 60.0},
                               {"n": 5.0, "a": 10.0, "b": 20.0, "c": 30.0}, [5.0, 10.0, 20.0])
    assert r["monotone_adjusted"] and r["raw_tau_min"] == [-10.0, 30.0, 20.0, 60.0]
    np.testing.assert_allclose(r["knots_tau_min"], [0.0, 25.0, 25.0, 60.0])
    np.testing.assert_allclose(r["tau_band"], [0.0, 25.0, 25.0])
    fit = np.asarray(r["knots_tau_min"])
    assert np.all(np.diff(fit) >= 0) and fit[0] >= 0
    # the isotonic fit is the least-squares non-decreasing fit (brute check)
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.normal(0, 10, 6)
        f = levels._isotonic(y)
        assert np.all(np.diff(f) >= -1e-12) and abs(f.sum() - y.sum()) < 1e-9
        for _ in range(50):
            g = np.sort(f + rng.normal(0, 0.5, 6))
            assert ((y - f) ** 2).sum() <= ((y - g) ** 2).sum() + 1e-9
    # a same-location pair is one knot (mean s, mean lag); bad judges are dropped
    pair = config.SAME_LOCATION[0]
    r = levels.gauge_band_lags({pair[0]: 18.0, pair[1]: 22.0, "x": 50.0, "y": float("nan")},
                               {pair[0]: 21.0, pair[1]: 23.0, "x": None, "y": 5.0}, [22.0])
    assert r["codes"] == ["+".join(pair)] and r["knots_s_km"] == [22.0]
    np.testing.assert_allclose(r["tau_band"], [20.0])
    assert r["dropped"] == {"x": "no positive finite s_g", "y": "no finite lag"}
    # nothing to place, and judges at the same s
    none = levels.gauge_band_lags({"x": 50.0}, {"x": -1.0}, [1.0, 2.0])
    assert none["status"] == "not available" and np.all(np.isnan(none["tau_band"]))
    with pytest.raises(ValueError):
        levels.gauge_band_lags({"a": 10.0, "b": 20.0}, {"a": 5.0, "b": 5.0}, [1.0])
