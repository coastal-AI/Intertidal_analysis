"""Tests of experiments.benchmark.truth, the truths every method is scored against.

On the synthetic site of ``tests/benchmark_site.py`` (a package built with
the fake boundary h_B; the judge ``st_in`` reads offset + h_B(t - 75 min) +
3 mm noise every 5 min over 2023-2025) the tests pin the rules of the module
docstring (v13 spec §5.1, §5.5, §3.7):

* lag truth (a) at the record's S2 instants recovers the planted 75 min
  with an interval holding it; tau_ab recovers it with a planted gain too
  (E_ab <= E at every lag); the month-block bootstrap is seeded per judge;
* truth (b), the M2 harmonic phase lag of the centred 10-min series,
  recovers 75 min within 0.2 min with I_b holding it; the all-sample lag
  recovers it exactly; the gauge is read nearest within 6 min and
  interpolated only across gaps of at most 20 min;
* a global model at the gauge is scored through the same machinery, and an
  unloadable model is reported, not fatal;
* the elevation truth: the nearest-cell lookup equals ``v1.sample_truth``
  (z, year) and adds the cell ids; the range filter [min h0 - 0.25,
  max h0 + 0.25]; the support S = D_truth ∪ keep_full ∪ the
  neighbourhoods; a planted truth end to end (arrays, strata bands,
  resume by inputs key); RTK frame mapping drops points off the frame
  without an IndexError at the edge;
* the lag truth of a site end to end (truth_lag.json, lag_truth.csv,
  resume by inputs key).

Validation v2 (the ``test_v2_*`` tests, spec §1.7-§1.9, §6.2, §7.1, §10.3):

* multi-epoch Vaklodingen: the survey-matched truth (nearest epoch, ties to
  the later, fallback to the next survey), v13's latest view equal to
  ``v1.sample_truth``, the first sheet with any survey wins, a sheet's
  same-year stamps merged; the 20 m cell support on a 10 m UTM frame (every
  pixel centre inside its cell, the >= 2 and >= half rule against brute
  force); the stable-cell rule;
* the range filter from the 1-min h_B (quantiles, counts per band, the local
  LG bounds);
* PNOA: constant-value patches (mm rounding, 4-connectivity, >= 50 px), the
  flight-level mask, the ground averaging to 10 m against brute force, the
  flight water level with the gauge residual; the RTK epoch from the time
  column only;
* the truth-error model on synthetic random-walk surveys recovers sigma2_s,
  D_W(Y) inside and outside the window and the transfer ratio;
* the strata (along-s interpolation, classes, hydroperiod zones equal to
  ``hydroperiod.zonation``, slope, distances, terciles), the low-tide dry
  medians by lattice-aligned row blocks equal to the whole-frame
  computation, the channel-anchor medians;
* ``fetch vak-meta`` (source methods, survey dates, overrides) and ``fetch
  pnoa-meta`` (ISO 19139 dates and accuracy, zoned flight table);
* per-window lag truths on the synthetic site recover the planted 75 min in
  (a), tau_ab, (b), tau_all and HW/LW, with the amplitude ratio 1, a month
  jackknife in 1-year windows and window-specific bootstrap seeds; the
  per-window coverage rule and the end-to-end v2 truth files with resume.

Run:  python -X utf8 -m pytest tests/test_benchmark_truth.py -q
"""
import dataclasses
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, judges, product, truth  # noqa: E402
from tests.benchmark_site import (  # noqa: E402
    GAUGES, H, SITE, W, FakeTide, elevation, gauge_series, make_site, use_stations,
    write_gauges)

#: a small constituent set for truth (b) in tests (the fake tide holds M2,
#: S2 and O1; the frozen 52 constituents are for the real series)
CONS = ("m2", "s2", "o1", "k1", "n2", "m4")
LAG = GAUGES["st_in"]["lag_min"]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    s = make_site(tmp_path_factory.mktemp("v13_truth"), mp)
    use_stations(mp, s.inp)
    s.gauges = write_gauges(s.inp)
    s.stations = {st["code"]: st for st in judges.stations(SITE)}
    s.series = {c: judges.read_series(st) for c, st in s.stations.items()}
    s.H = s.pkg.levels.stack(truth.TAUS_A)
    yield s
    s.pkg.close()
    mp.undo()


class ShiftedTide:
    """A 'global model' that reads the fake tide ``lag_min`` late (time order)."""

    def __init__(self, lag_min):
        self.lag = int(lag_min) * 60_000_000_000

    def levels(self, times):
        ns = np.sort(pd.DatetimeIndex(times).asi8)
        return FakeTide.value(ns - self.lag)


# ─────────────────────────────────────────────────────────────────────────────
#  small helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_wraps_and_days():
    assert truth.wrap180(190.0) == -170.0 and truth.wrap180(180.0) == -180.0
    assert truth.wrap180(-180.0) == -180.0 and truth.wrap180(359.0) == -1.0
    P = truth.GM20_PERIOD_MIN
    assert truth.wrap_gm20(-121.0) == pytest.approx(P - 121.0)
    assert truth.wrap_gm20(-120.0) == -120.0 and truth.wrap_gm20(P - 120.0) == -120.0
    assert truth.days_since_1992(pd.to_datetime(["1992-01-02 12:00"]))[0] == 1.5


def test_lookup_never_indexes_past_the_end():
    pos, found = truth.lookup([1, 5, 9], [9, 10, 0, 5])
    assert found.tolist() == [True, False, False, True] and pos[[0, 3]].tolist() == [2, 1]
    pos, found = truth.lookup([], [3])
    assert not found.any()


def test_interp_at_bridges_only_short_gaps():
    idx = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:10", "2023-01-01 00:40"])
    s = pd.Series([0.0, 1.0, 4.0], index=idx).astype(float)
    s.index = s.index.as_unit("ns")
    q = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:20",
                        "2023-01-01 00:40", "2023-01-01 00:41", "2022-12-31 23:59"])
    got = truth.interp_at(s, q)
    assert np.array_equal(got, [0.0, 0.5, np.nan, 4.0, np.nan, np.nan], equal_nan=True)
    assert truth.interp_at(s, q, max_gap_min=30)[2] == pytest.approx(2.0)
    assert np.isnan(truth.interp_at(s.iloc[:0], q)).all()


def test_best_lag_and_e_rows():
    taus = np.arange(-3, 4)
    E = np.array([5.0, 3.0, 1.005, 1.0, 1.009, 2.0, np.nan])
    r = truth.best_lag(taus, E, tol=0.01)
    assert (r["tau"], r["lo"], r["hi"], r["Emin_m"]) == (0.0, -1.0, 1.0, 1.0)
    assert r["contiguous"] and not r["edge"] and not r["interval_edge"]
    r = truth.best_lag(taus, np.array([1.0, 2, 3, 4, 1.005, 6, 7]), tol=0.01)
    assert r["edge"] and r["interval_edge"] and not r["contiguous"]
    r = truth.best_lag(taus, np.full(7, np.nan))
    assert np.isnan(r["tau"]) and not r["edge"]
    Hm = np.array([[1.0, 2.0, 3.0, 9.0], [1.0, 2.0, 3.0, 4.0]])
    g = np.array([0.0, 1.0, 2.0, np.nan])
    assert np.allclose(truth.e_rows(Hm, g), [0.0, 0.0])          # a constant offset
    assert np.isnan(truth.e_rows(Hm, np.full(4, np.nan))).all()


def test_gain_aware_is_never_worse_than_a():
    rng = np.random.default_rng(0)
    Hm = rng.normal(size=(9, 60))
    g = 1.7 * Hm[4] + 0.3 + rng.normal(0, 0.01, 60)
    g[::7] = np.nan
    E = truth.e_rows(Hm, g)
    Eab, B = truth.gain_aware(Hm, g)
    assert (Eab <= E + 1e-12).all()
    assert int(np.argmin(Eab)) == 4 and B[4] == pytest.approx(1.7, abs=0.01)
    Eab, B = truth.gain_aware(Hm, np.full(60, np.nan))
    assert np.isnan(Eab).all() and np.isnan(B).all()


def test_fold_phase_reads_the_crest_time():
    x = np.linspace(0, 3 * truth.GM20_PERIOD_H, 400)
    y = 0.5 + 2.0 * np.cos(2 * np.pi * (x - 2.0) / truth.GM20_PERIOD_H)
    r = truth.fold_phase(x, y)
    assert r["phase_min"] == pytest.approx(120.0, abs=1e-6)
    assert r["amp"] == pytest.approx(2.0) and r["r2"] == pytest.approx(1.0)
    assert np.isnan(truth.fold_phase(x[:3], y[:3])["phase_min"])


# ─────────────────────────────────────────────────────────────────────────────
#  lag truths on the synthetic site
# ─────────────────────────────────────────────────────────────────────────────

def test_truth_a_and_ab_recover_the_planted_lag(site):
    st = site.stations["st_in"]
    r = truth.truth_s2(site.pkg, st, site.series["st_in"], H=site.H)
    a, ab = r["a"], r["ab"]
    assert a["n_instants"] == site.pkg.T and a["coverage"] == 1.0
    assert abs(a["tau"] - LAG) <= 1 and a["lo"] <= LAG <= a["hi"]
    # the gauge is read as delivered: the nearest 5-min sample is up to 2.5 min
    # from the instant, about 1.5 cm of RMS on this tide
    assert a["Emin_m"] < 0.03 and a["E0_m"] > 0.1 and not a["edge"]
    assert a["interp"]["Emin_m"] < a["Emin_m"]
    assert abs(ab["tau"] - LAG) <= 1 and ab["lo"] <= LAG <= ab["hi"]
    assert ab["gain"] == pytest.approx(1.0, abs=0.02) and not ab["gain_edge"]
    assert (r["curves"]["E_ab"] <= r["curves"]["E_a"] + 1e-12).all()
    assert abs(a["interp"]["tau"] - LAG) <= 1
    assert a["boot"]["n"] == truth.N_BOOT and a["boot"]["ci_lo"] <= LAG <= a["boot"]["ci_hi"]
    # the bootstrap is seeded per (site, judge): the same draws twice
    again = truth.truth_s2(site.pkg, st, site.series["st_in"], H=site.H)
    assert np.array_equal(again["curves"]["boot_tau"], r["curves"]["boot_tau"])
    assert np.array_equal(r["g_s2"], judges.at_instants(site.series["st_in"], site.pkg.t, 6))


def test_tau_ab_recovers_a_planted_gain(site):
    spec = dict(GAUGES["st_in"], gain=1.3, offset_m=-0.4)
    t, v = gauge_series(spec, seed=5)
    g = pd.Series(v, index=t)
    r = truth.truth_s2(site.pkg, {"code": "gain"}, g, H=site.H, boot=False)
    assert abs(r["ab"]["tau"] - LAG) <= 1 and r["ab"]["lo"] <= LAG <= r["ab"]["hi"]
    assert r["ab"]["gain"] == pytest.approx(1.3, abs=0.02)
    assert r["ab"]["Emin_m"] < 0.03 and r["a"]["Emin_m"] > 0.1
    assert "boot" not in r["a"]


def test_truth_b_m2_phase_lag(site):
    st = site.stations["st_in"]
    idx10, _ = truth.series10(site.pkg)
    g10 = judges.centred_10min(site.series["st_in"], st["sampling_min"], grid=idx10)
    b = truth.truth_m2(site.pkg, st, g10.to_numpy(), constituents=CONS)
    assert b["tau"] == pytest.approx(LAG, abs=0.2), b
    assert b["lo"] <= LAG <= b["hi"] and b["se"] < 0.5
    assert b["n_quarters"] == 12 and b["flags"] == []
    assert b["A_M2_gauge"] == pytest.approx(b["A_M2_B"], rel=0.01)
    assert b["n_samples"] == int(np.isfinite(g10.to_numpy()).sum())
    # tau_all: E over every 10-min sample from the exact 1-min series
    al, E = truth.tau_all(site.pkg, g10.to_numpy())
    assert al["tau"] == LAG and al["Emin_m"] < 0.01 < al["E0_m"]
    # too few samples: flagged, no lag
    short = np.full(len(idx10), np.nan)
    short[:10] = g10.to_numpy()[:10]
    bs = truth.truth_m2(site.pkg, st, short, constituents=CONS)
    assert np.isnan(bs["tau"]) and "too_few_samples" in bs["flags"]
    with pytest.raises(ValueError):
        truth.truth_m2(site.pkg, st, g10.to_numpy()[:-1], constituents=CONS)


def test_hw_lw_and_frame_truths_are_finite_and_near_the_lag(site):
    st = site.stations["st_in"]
    idx10, _ = truth.series10(site.pkg)
    g10 = judges.centred_10min(site.series["st_in"], st["sampling_min"], grid=idx10)
    hl = truth.hw_lw_lags(site.pkg, g10.to_numpy())
    for k in ("hw", "lw"):
        assert hl[k]["n"] > 1000 and abs(hl[k]["median_min"] - LAG) <= 15, hl
    g_s2 = judges.at_instants(site.series["st_in"], site.pkg.t, 6)
    c = truth.frame_truths(site.pkg, g_s2)
    assert c["n"] == site.pkg.T
    assert abs(c["hd26_oracle"] - LAG) <= 10, c
    assert np.isfinite(c["gm20_rel"]) and np.isfinite(c["gm20_raw"])


def test_model_at_gauge(site, monkeypatch):
    st = site.stations["st_in"]
    monkeypatch.setattr(truth, "model_provider", lambda model, lat, lon: ShiftedTide(30))
    r = truth.model_at_gauge(site.pkg, st, "FAKE", H=site.H, ocean=False, constituents=CONS)
    assert r["available"] and r["a"]["tau"] == 30 and r["ab"]["tau"] == 30
    assert r["b"]["tau"] == pytest.approx(30.0, abs=0.2)
    assert r["ab"]["gain"] == pytest.approx(1.0, abs=0.01)

    def broken(model, lat, lon):
        raise FileNotFoundError("no model files")

    monkeypatch.setattr(truth, "model_provider", broken)
    r = truth.model_at_gauge(site.pkg, st, "GONE", H=site.H, ocean=False)
    assert not r["available"] and "FileNotFoundError" in r["reason"]


def test_build_lag_truth_end_to_end(site, monkeypatch):
    monkeypatch.setattr(truth, "model_provider", lambda model, lat, lon: ShiftedTide(30))
    monkeypatch.setattr(truth, "nearest_ocean_km", lambda lat, lon, model: 1.25)
    # st_in (a full record) and st_quay (a short, drying one) keep it short
    table = [r for r in judges.build(site.pkg, path=f"{site.cfg.truth_dir}/judges.json")
             if r["code"] != "st_out"]
    path = truth.build_lag_truth(site.pkg, table=table, models=("FAKE",), constituents=CONS)
    doc = truth.load_lag_truth(site.cfg)
    assert path.endswith("truth_lag.json") and doc["inputs_key"]
    by = {s["code"]: s for s in doc["stations"]}
    assert list(by) == ["st_in", "st_quay"] and not by["st_quay"]["included"]
    s = by["st_in"]
    assert abs(s["a"]["tau"] - LAG) <= 1 and s["b"]["tau"] == pytest.approx(LAG, abs=0.2)
    assert s["all"]["tau"] == LAG and s["included"]
    assert s["models"]["FAKE"]["a"]["tau"] == 30 and s["models"]["FAKE"]["ocean_km"] == 1.25
    assert doc["headline_locations"] == ["st_in"]
    csv = pd.read_csv(f"{site.cfg.truth_dir}/lag_truth.csv")
    assert csv["judge"].tolist() == list(by) and "fake_tau_a" in csv.columns
    assert csv.loc[csv["judge"] == "st_in", "tau_a"].item() == s["a"]["tau"]
    curves = product.load_npz(f"{site.cfg.truth_dir}/truth_lag_curves.npz")
    assert np.array_equal(curves["taus"], truth.TAUS_A)
    assert "st_in__E_a" in curves and "st_in__fake__E_ab" in curves
    # an unchanged rebuild does nothing; force rebuilds
    m0 = os.stat(path).st_mtime_ns
    truth.build_lag_truth(site.pkg, table=table, models=("FAKE",), constituents=CONS)
    assert os.stat(path).st_mtime_ns == m0


# ─────────────────────────────────────────────────────────────────────────────
#  elevation truth
# ─────────────────────────────────────────────────────────────────────────────

def _sheet(x0, y0, nx, ny, step, rng, nan_share=0.0, year=2021):
    x = x0 + step * np.arange(nx)
    y = y0 - step * np.arange(ny)
    z = rng.normal(size=(ny, nx))
    z[rng.random(z.shape) < nan_share] = np.nan
    yr = np.full((ny, nx), year)
    yr[:, nx // 2:] = year + 2
    return x, y, z, yr


def test_sample_cells_equals_v1_sample_truth_plus_cell_ids():
    from experiments import v1_vaklodingen_validate as v1

    rng = np.random.default_rng(8)
    cells = {"A": _sheet(0.0, 1000.0, 30, 25, 20.0, rng, nan_share=0.3),
             "B": _sheet(300.0, 900.0, 30, 30, 20.0, rng, year=2019)}
    xq = rng.uniform(-50, 950, 3000)
    yq = rng.uniform(250, 1050, 3000)
    xq[:5] = np.nan
    z, yr, sheet, iy, ix = truth.sample_cells(cells, xq, yq)
    z1, yr1 = v1.sample_truth(cells, xq, yq)
    assert np.array_equal(z, z1, equal_nan=True) and np.array_equal(yr, yr1)
    names = list(cells)
    fin = np.isfinite(z)
    assert fin.any() and (~fin).any() and (sheet[fin] == 1).any() and (sheet[fin] == 0).any()
    for k in np.flatnonzero(fin):
        zt = cells[names[sheet[k]]][2]
        assert zt[iy[k], ix[k]] == z[k]
    assert (sheet[~fin] == -1).all() and (iy[~fin] == -1).all() and (yr[~fin] == -1).all()
    ids = truth.cell_ids(sheet, iy, ix)
    assert (ids[~fin] == -1).all()
    want = (sheet[fin].astype(np.int64) * 10**12 + iy[fin].astype(np.int64) * 10**6
            + ix[fin].astype(np.int64))
    assert ids.dtype == np.int64 and np.array_equal(ids[fin], want)


def test_map_points_drops_points_off_the_frame():
    row = np.array([0, H - 1, H, -1, 5])
    col = np.array([W - 1, 0, 3, 3, W])
    r = truth.map_points(row, col, np.arange(5.0), np.arange(5.0) + 50, np.arange(5), (H, W))
    assert r["n_outside"] == 3
    assert r["flat"].tolist() == [W - 1, (H - 1) * W] and r["z"].tolist() == [0.0, 1.0]
    assert r["z_ellipsoidal"].tolist() == [50.0, 51.0] and r["block150"].tolist() == [0, 1]
    pos, found = truth.lookup(np.arange(H * W), r["flat"])
    assert found.all()


def test_range_filter_and_support():
    pkg = types.SimpleNamespace(h0=np.array([-1.0, 0.5, 2.0]))
    z = np.array([[-1.26, -1.25, 0.0], [2.25, 2.2501, np.nan]])
    tr, c = truth.range_filter(pkg, z)
    assert np.array_equal(tr, [[np.nan, -1.25, 0.0], [2.25, np.nan, np.nan]], equal_nan=True)
    assert (c["LO"], c["HI"], c["lo"], c["hi"]) == (-1.0, 2.0, -1.25, 2.25)
    assert (c["n_finite"], c["n_below"], c["n_above"], c["n_in_range"]) == (5, 1, 1, 3)
    with pytest.raises(ValueError):
        truth.record_range(types.SimpleNamespace(h0=np.array([0.0, np.nan])))
    S = truth.support([7, 3], [3, 9], nbhds=[np.array([1]), np.array([], np.int64)],
                      rtk_flat=[20, 7])
    assert S.tolist() == [1, 3, 7, 9, 20] and S.dtype == np.int64


def test_planted_elevation_truth_end_to_end(site, tmp_path):
    pkg = site.pkg
    z = elevation()
    tau = np.round(30 + 4.0 * np.arange(W))[None, :] * np.ones((H, 1))
    planted = str(tmp_path / "planted.npz")
    np.savez(planted, z=z, tau=tau)
    cfg = dataclasses.replace(pkg.cfg, elevation_truth=("planted",),
                              root=str(tmp_path / "out").replace("\\", "/"),
                              extra=types.MappingProxyType({"planted": planted}))
    proxy = types.SimpleNamespace(cfg=cfg, h0=pkg.h0, full=pkg.full, shape=pkg.shape,
                                  band_frame=pkg.band_frame, keys=pkg.keys, grid=pkg.grid)
    nb = [np.array([0, 1, W], np.int64), pkg.nb["nbhd"]["st_in"]]
    table = [{"nbhd": a} for a in nb]
    path = truth.build_elevation_truth(proxy, table=table)
    arrays, doc = truth.load_elevation_truth(cfg)
    lo, hi = pkg.h0.min() - 0.25, pkg.h0.max() + 0.25
    inr = np.isfinite(z) & (z >= lo) & (z <= hi)
    d_truth = np.flatnonzero(inr.ravel())
    S = np.unique(np.concatenate([d_truth, pkg.full.keep, *nb]))
    assert np.array_equal(arrays["support"], S)
    assert np.array_equal(truth.load_support(cfg), S)
    assert np.array_equal(arrays["truth_raw"], z.ravel()[S])
    assert np.array_equal(arrays["truth"], np.where(inr.ravel()[S], z.ravel()[S], np.nan),
                          equal_nan=True)
    assert np.array_equal(arrays["in_D_truth"], inr.ravel()[S])
    assert np.array_equal(arrays["in_keep"], np.isin(S, pkg.full.keep))
    assert np.array_equal(arrays["band"], pkg.band_frame.ravel()[S])
    assert (arrays["year"] == -1).all() and (arrays["cell_id"] == -1).all()
    assert doc["n_D_truth"] == len(d_truth) and doc["n_S"] == len(S)
    assert doc["grid_kind"] == "planted" and doc["range"]["n_in_range"] == len(d_truth)
    # 0 < z of the pond is in range, the sea and land are not
    assert inr[3, 42] and not inr[0, 0] and not inr[0, W - 1]
    m0 = os.stat(path).st_mtime_ns
    truth.build_elevation_truth(proxy, table=table)
    assert os.stat(path).st_mtime_ns == m0
    truth.build_elevation_truth(proxy, table=table, force=True)
    assert os.stat(path).st_mtime_ns != m0
    assert product.read_json(path)["inputs_key"] == doc["inputs_key"]
    # the planted lag per stratum band
    pl = truth.planted_lag(proxy)
    keep, band = pkg.full.keep, pkg.full.band_of
    for b, t_mean, n in zip(pl["bands"], pl["tau_mean_min"], pl["n_px"]):
        assert t_mean == pytest.approx(tau.ravel()[keep][band == b].mean())
        assert n == int((band == b).sum())
    with pytest.raises(FileNotFoundError):
        truth.planted_frame(types.SimpleNamespace(
            cfg=dataclasses.replace(cfg, extra=types.MappingProxyType({})), shape=pkg.shape))


def test_frozen_constants():
    assert truth.TAUS_A[0] == -60 and truth.TAUS_A[-1] == 240 and len(truth.TAUS_A) == 301
    assert truth.TOL_MIN == 6.0 and truth.INTERVAL_M == 0.01 and truth.RANGE_PAD_M == 0.25
    assert len(truth.CONS_B) == 52 and len(set(truth.CONS_B)) == 52
    assert truth.GAIN_GRID[0] == 0.0 and truth.GAIN_GRID[-1] == 4.0 and 1.0 in truth.GAIN_GRID
    assert tuple(truth.MODELS) == tuple(config.FROZEN["baselines"]["models"])
    r1 = truth.boot_rng(SITE, "st_in").integers(0, 1 << 30, 3)
    r2 = truth.boot_rng(SITE, "st_in").integers(0, 1 << 30, 3)
    r3 = truth.boot_rng(SITE, "st_quay").integers(0, 1 << 30, 3)
    assert np.array_equal(r1, r2) and not np.array_equal(r1, r3)


# ═════════════════════════════════════════════════════════════════════════════
#  validation v2 (docs/validation_v2_spec.md §1.7-§1.9, §6.2, §7.1)
# ═════════════════════════════════════════════════════════════════════════════

#: the synthetic site's judges use provider "synthetic": its stamps are centred
SYN_CONV = {"synthetic": {"averaging_min": 5.0, "stamp": "centre", "status": "test",
                          "source": "tests"}}


def _epoch_sheet(x0, y0, nx, ny, years, rng, step=20.0, nan_share=0.0):
    """A read_sheet_epochs-like dict with independent surveys per year."""
    x = x0 + step * np.arange(nx)
    y = y0 - step * np.arange(ny)
    z = rng.normal(size=(len(years), ny, nx))
    z[rng.random(z.shape) < nan_share] = np.nan
    src = np.where(np.isfinite(z), np.arange(len(years))[:, None, None], -1).astype(np.int16)
    return {"x": x, "y": y, "years": np.asarray(years, np.int64), "z": z, "src": src,
            "stamps": {int(v): [f"{v}-01-01"] for v in years},
            "sources": [f"_{v}0101_s{v}tt20.asc" for v in years]}


def test_v2_survey_matched_nearest_with_ties_to_the_later():
    years = np.array([2016, 2019, 2022, 2025])
    Z = np.array([[1.0, 2.0, 3.0, 4.0],
                  [1.0, 2.0, 3.0, np.nan],
                  [1.0, np.nan, np.nan, np.nan],
                  [np.nan] * 4])
    z, yr, gap = truth.survey_matched(Z, years, 2024.5)       # W3
    assert z.tolist()[:3] == [4.0, 3.0, 1.0] and np.isnan(z[3])
    assert yr.tolist() == [2025, 2022, 2016, -1]
    assert gap[:3].tolist() == [1.0, 2.0, 8.0] and np.isnan(gap[3])
    # 2021.0 is 1.5 years from both 2019 (epoch 2019.5) and 2022 (2022.5): the later wins
    z, yr, _ = truth.survey_matched(Z, years, 2021.0)
    assert yr.tolist()[:2] == [2022, 2022]
    z, yr, _ = truth.survey_matched(Z, years, truth.window_of("V2019L3").centre)
    assert yr.tolist()[:3] == [2019, 2019, 2016]
    zl, yl = truth.latest_survey(Z, years)
    assert zl.tolist()[:2] == [4.0, 3.0] and np.isnan(zl[2]) and yl.tolist() == [2025, 2022, -1, -1]
    assert np.array_equal(truth.survey_in_year(Z, years, 2019), Z[:, 1], equal_nan=True)
    assert np.isnan(truth.survey_in_year(Z, years, 2020)).all()


def test_v2_cells_latest_view_equals_v1_and_the_first_sheet_wins():
    from experiments import v1_vaklodingen_validate as v1

    rng = np.random.default_rng(11)
    years = [2016, 2018, 2019, 2021, 2023, 2025]
    A = _epoch_sheet(0.0, 1000.0, 30, 25, years, rng, nan_share=0.4)
    B = _epoch_sheet(300.0, 900.0, 30, 30, years[1:], rng, nan_share=0.2)
    sheets = {"A": A, "B": B}
    xq = rng.uniform(-50, 950, 4000)
    yq = rng.uniform(250, 1050, 4000)
    xq[:3] = np.nan
    C = truth.vaklodingen_cells(xq, yq, sheets)
    assert C["years"].tolist() == years and C["Z"].shape == (len(C["cell_id"]), len(years))
    assert (np.diff(C["cell_id"]) > 0).all() and (C["point_cell"][:3] == -1).all()
    pc = C["point_cell"]
    assert C["n_px"].sum() == (pc >= 0).sum()
    # every point takes the first sheet whose cell has any survey, with every year
    for k in np.flatnonzero(pc >= 0)[:300]:
        c = pc[k]
        sh = sheets["AB"[C["sheet"][c]]]
        col = [list(sh["years"]).index(y) if y in sh["years"] else None for y in years]
        want = [sh["z"][j][C["iy"][c], C["ix"][c]] if j is not None else np.nan for j in col]
        assert np.array_equal(C["Z"][c], want, equal_nan=True)
        assert C["x_rd"][c] == sh["x"][C["ix"][c]] and C["y_rd"][c] == sh["y"][C["iy"][c]]
        if C["sheet"][c] == 1:              # B only where A's cell has no survey at all
            fx = np.round((xq[k] - A["x"][0]) / 20.0)
            fy = np.round((yq[k] - A["y"][0]) / 20.0)
            if 0 <= fx < 30 and 0 <= fy < 25:
                assert np.isnan(A["z"][:, int(fy), int(fx)]).all()
    # v13's view (latest 2019-2025) equals v1.sample_truth on v1-style merged cells
    merged = {}
    for name, sh in sheets.items():
        zt = np.full(sh["z"].shape[1:], np.nan)
        yt = np.full(zt.shape, -1)
        for j, y in enumerate(sh["years"]):
            if 2019 <= y <= 2025:
                m = np.isfinite(sh["z"][j])
                zt[m], yt[m] = sh["z"][j][m], y
        merged[name] = (sh["x"], sh["y"], zt, yt)
    z1, y1 = v1.sample_truth(merged, xq, yq)
    zl, yl = truth.latest_survey(C["Z"], C["years"])
    z2 = np.where(pc >= 0, zl[np.maximum(pc, 0)], np.nan)
    y2 = np.where(pc >= 0, yl[np.maximum(pc, 0)], -1)
    # equal wherever the first sheet with a survey also has one in 2019-2025
    same = np.isfinite(z2) | ~np.isfinite(z1)
    assert same.mean() > 0.95
    assert np.array_equal(z1[same], z2[same], equal_nan=True)
    assert np.array_equal(y1[same], y2[same])


def test_v2_cell_support_on_a_10m_frame():
    pyproj = pytest.importorskip("pyproj")
    x0, y0 = 560_000.0, 5_700_000.0                       # EPSG:32631, inside the RD domain
    grid = {"xs": x0 + 10.0 * np.arange(40), "ys": y0 - 10.0 * np.arange(30),
            "crs": pyproj.CRS.from_epsg(32631)}
    XQ, YQ = truth.rd_coords(grid, (30, 40), block_rows=7)
    tr = pyproj.Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    xa, ya = tr.transform(grid["xs"][[0, 39]], grid["ys"][[0, 29]])
    assert XQ[0] == pytest.approx(xa[0]) and YQ[-1] == pytest.approx(ya[1])
    rng = np.random.default_rng(3)
    x = np.arange(np.floor(XQ.min() / 20) * 20 - 30, XQ.max() + 40, 20.0)
    y = np.arange(np.ceil(YQ.max() / 20) * 20 + 30, YQ.min() - 40, -20.0)
    sh = {"x": x, "y": y, "years": np.array([2023, 2025]),
          "z": rng.normal(size=(2, len(y), len(x)))}
    C = truth.vaklodingen_cells(XQ, YQ, {"S": sh})
    pc = C["point_cell"]
    assert (pc >= 0).all()
    # every pixel centre lies inside its 20 m cell; cells hold 2-6 centres, 4 on average
    assert (np.abs(XQ - C["x_rd"][pc]) <= 10.0 + 1e-6).all()
    assert (np.abs(YQ - C["y_rd"][pc]) <= 10.0 + 1e-6).all()
    assert C["n_px"].min() >= 1 and C["n_px"].max() <= 6
    assert 3.0 < C["n_px"].mean() < 4.5
    # the cell value: the mean of its mapped pixels, scored with >= 2 and >= half mapped
    vals = rng.normal(size=pc.size)
    vals[rng.random(pc.size) < 0.45] = np.nan
    mean, cnt, ok = truth.cell_aggregate(vals, pc, len(C["cell_id"]), C["n_px"])
    for c in range(len(C["cell_id"])):
        v = vals[pc == c]
        n = int(np.isfinite(v).sum())
        assert cnt[c] == n
        want = n >= 2 and n >= 0.5 * C["n_px"][c]
        assert ok[c] == want
        if want:
            assert mean[c] == pytest.approx(np.nanmean(v), abs=1e-12)
        else:
            assert np.isnan(mean[c])
    mean2, _, ok2 = truth.cell_aggregate(vals, pc, len(C["cell_id"]), C["n_px"],
                                         mapped=np.zeros(pc.size, bool))
    assert not ok2.any() and np.isnan(mean2).all()


def test_v2_stable_cells_rule():
    years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025])
    Z = np.full((4, 7), np.nan)
    Z[0, [3, 6]] = [1.00, 1.15]                    # 2022 and 2025: range 0.15
    Z[1, [3, 4]] = [1.00, 1.30]                    # range 0.30
    Z[2, [4]] = [1.0]                              # one survey only
    Z[3, [0, 1]] = [1.0, 1.05]                     # outside 2022..2026
    st = truth.stable_cells(Z, years, truth.window_years("W3"))
    assert st["years"] == [2022, 2026]
    assert st["stable"].tolist() == [True, False, False, False]
    assert st["n"].tolist() == [2, 2, 1, 0]
    assert st["range_m"][0] == pytest.approx(0.15) and np.isnan(st["range_m"][2])
    assert truth.window_years("Y2024") == (2024, 2024) and truth.window_years("W9") == (2017, 2025)


def test_v2_range_filter_from_the_1min_series():
    rng = np.random.default_rng(5)
    h = np.sin(np.linspace(0, 400 * np.pi, 200_001)) * 2.0
    h[::1000] = np.nan
    r = truth.range_from_series(h)
    fin = h[np.isfinite(h)]
    assert r["q_lo"] == pytest.approx(np.quantile(fin, 0.001))
    assert r["hi"] == pytest.approx(np.quantile(fin, 0.999) + 0.25)
    assert r["n_finite"] == len(fin) and r["q"] == [0.001, 0.999]
    with pytest.raises(ValueError):
        truth.range_from_series(np.full(5, np.nan))
    z = rng.uniform(-4, 4, (20, 20))
    z[0, 0] = np.nan
    band = np.repeat(np.arange(4), 100).reshape(20, 20)
    tr, c = truth.apply_range(z, r["lo"], r["hi"], band=band)
    inr = np.isfinite(z) & (z >= r["lo"]) & (z <= r["hi"])
    assert np.array_equal(np.isfinite(tr), inr)
    assert c["n_below"] + c["n_above"] + c["n_in_range"] == c["n_finite"] == 399
    assert sum(v["n_in_range"] for v in c["by_band"].values()) == c["n_in_range"]
    assert c["by_band"][1]["n_finite"] == 100
    # the LG sensitivity: per-bin bounds
    hb = np.vstack([h, 1.5 * h, np.full(len(h), np.nan)])
    lo, hi = truth.local_range(hb)
    assert hi[1] > hi[0] and np.isnan(lo[2])
    zz = np.array([2.3, 2.3, 2.3, 0.0])
    got = truth.apply_local_range(zz, [0, 1, 2, -1], lo, hi)
    assert np.isnan(got[0]) and got[1] == 2.3 and np.isnan(got[2]) and np.isnan(got[3])


def _pnoa_scene(rng):
    """A 2 m rough intertidal slope (mm resolution) with a flattened pond of
    120 px at -0.56 m and a 30-px constant patch that is too small."""
    H, W = 40, 60
    z = np.round((-1.5 + 3.0 * np.arange(W) / W)[None, :] + rng.normal(0, 0.01, (H, W)), 3)
    z[5:15, 5:17] = -0.56                     # 120 px pond
    z[30:35, 40:46] = 0.802                   # 30 px: not water
    z[38:, :] = np.nan
    return z


def test_v2_constant_patches_and_the_flight_mask():
    rng = np.random.default_rng(9)
    z = _pnoa_scene(rng)
    water, info = truth.constant_patches(z)
    assert water[5:15, 5:17].all() and not water[30:35, 40:46].any()
    assert info["n_patches"] >= 1 and info["values"][0] == [-0.56, 120]
    assert water.sum() - 120 < 0.01 * np.isfinite(z).sum()  # rough ground rarely chains
    # a patch needs equal values at 1 mm: 4 mm steps never form one
    ramp = np.round(np.arange(100)[None, :] * 0.004 + np.zeros((3, 1)), 3)
    assert not truth.constant_patches(ramp, min_px=4)[0].any()       # columns of 3 only
    # vertical neighbours count (4-connectivity), diagonals do not
    d = np.arange(64.0).reshape(8, 8)
    d[np.arange(8), np.arange(8)] = -1.0
    assert not truth.constant_patches(d, min_px=2)[0].any()
    fm = truth.flight_mask(z, 0.10)
    assert np.array_equal(fm, np.isfinite(z) & (z < 0.10 + 0.15))
    assert not truth.flight_mask(z, None).any() and not truth.flight_mask(z, np.nan).any()
    g = truth.pnoa_ground(z, flight_level=0.10)
    assert np.array_equal(g["ground"], np.isfinite(z) & ~water & ~fm)
    assert not (g["below_flight"] & g["water_patch"]).any()
    g0 = truth.pnoa_ground(z)
    assert np.array_equal(g0["ground"], np.isfinite(z) & ~water)


def test_v2_aggregate_ground_to_10m():
    pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    rng = np.random.default_rng(4)
    z = _pnoa_scene(rng)
    g = truth.pnoa_ground(z, flight_level=-0.9)
    crs = CRS.from_epsg(32630)
    st = from_origin(400_000.0, 4_800_000.0, 2.0, 2.0)
    dt = from_origin(400_000.0, 4_800_000.0, 10.0, 10.0)
    out, gshare, vshare = truth.aggregate_ground(z, st, crs, g["ground"], g["valid"], dt, crs,
                                                 (8, 12))
    for r in range(8):
        for c in range(12):
            blk = (slice(5 * r, 5 * r + 5), slice(5 * c, 5 * c + 5))
            gr, va = g["ground"][blk], g["valid"][blk]
            share = gr.sum() / max(va.sum(), 1)
            if abs(share - 0.8) < 1e-6 or abs(va.mean() - 0.95) < 1e-6:
                continue                              # exactly on a rule's limit
            ok = va.mean() > 0.95 and share >= 0.8
            if ok:
                assert out[r, c] == pytest.approx(z[blk][gr].mean(), abs=1e-9)
            else:
                assert np.isnan(out[r, c])
    assert np.isnan(out[1:3, 1:3]).all()                  # the pond is masked


def test_v2_flight_water_level_adds_the_gauge_residual():
    t0 = pd.Timestamp("2016-07-02 11:00")

    def tide(t):
        m = (pd.DatetimeIndex(t) - t0).total_seconds().to_numpy() / 60.0
        return 1.5 * np.cos(2 * np.pi * m / 745.2)

    def gauge_tide(t):
        return tide(pd.DatetimeIndex(t) - pd.Timedelta(minutes=12))

    idx = pd.date_range(t0 - pd.Timedelta(days=20), t0 + pd.Timedelta(days=20), freq="1min")
    surge = np.where(np.abs(idx - t0) <= pd.Timedelta(hours=6), 0.30, 0.0)
    g = pd.Series(7.0 + gauge_tide(idx) + surge, index=idx)     # datum 7 m, surge 0.3 m
    r = truth.flight_water_level([t0, t0 + pd.Timedelta(hours=3)], tide, g, gauge_tide)
    assert r["tide"][0] == pytest.approx(1.5)
    assert r["residual"][0] == pytest.approx(0.30, abs=1e-9)
    assert r["level"] == pytest.approx(1.8, abs=1e-9) and r["n_residual"] > 1000
    r0 = truth.flight_water_level(t0, tide)
    assert r0["level"] == pytest.approx(1.5) and r0["residual"] == [0.0]


def test_v2_rtk_epoch_reads_only_the_time_column(tmp_path):
    p = tmp_path / "rtk.csv"
    pd.DataFrame({"Name": [1, 2], "Averaging start": ["2026-08-14 11:07:37.0 UTC+02:00",
                                                        "2026-08-14 13:10:00.0 UTC+02:00"],
                  "Ellipsoidal height": [1.0, 2.0]}).to_csv(p, index=False)
    r = truth.rtk_epoch(str(p))
    assert r["first_utc"] == "2026-08-14 09:07:37" and r["last_utc"] == "2026-08-14 11:10:00"
    assert r["dates"] == ["2026-08-14"] and r["n_points"] == 2 and 2026.6 < r["epoch"] < 2026.7


def _surveys(rng, years, n, sigma_s, q, s0_sd=1.0):
    """Random-walk surfaces (increment variance q per year) plus survey noise."""
    steps = rng.normal(0, np.sqrt(q), (n, len(years)))
    steps[:, 0] = rng.normal(0, s0_sd, n)
    s = np.cumsum(steps, axis=1)
    return s + rng.normal(0, sigma_s, s.shape)


def test_v2_truth_error_model_recovers_noise_and_mismatch():
    rng = np.random.default_rng(2026)
    years = np.arange(2016, 2026)
    sig, q = 0.12, 0.01
    Z = _surveys(rng, years, 40_000, sig, q)
    Z[rng.random(Z.shape) < 0.05] = np.nan
    pr = truth.pair_variances(Z, years)
    dt, D, n = truth.d_curve(pr)
    assert dt.tolist() == list(range(1, 10))
    assert np.allclose(D, 2 * sig ** 2 + q * dt, rtol=0.05)
    noise = truth.survey_noise(dt, D, n)
    assert noise["sigma2_s"] == pytest.approx(sig ** 2, rel=0.10)
    assert noise["iso_intercept"] == pytest.approx(2 * sig ** 2, rel=0.15)
    # D_W(Y): Y outside the window (32 q / 9) and inside it (2 q / 9)
    out = truth.window_mismatch(Z, years, (2023, 2025), 2020, sig ** 2)
    assert out["D_W"] == pytest.approx(32 * q / 9, rel=0.10) and not out["Y_in_window"]
    inn = truth.window_mismatch(Z, years, (2023, 2025), 2024, sig ** 2)
    assert inn["D_W"] == pytest.approx(2 * q / 9, rel=0.25) and inn["Y_in_window"]
    assert inn["noise"] == pytest.approx(sig ** 2 * 2 / 3)
    miss = truth.window_mismatch(Z, years, (2026, 2027), 2020, sig ** 2)
    assert np.isnan(miss["D_W"]) and "reason" in miss
    # a site whose surface changes twice as fast: ratio 2
    Zw = _surveys(rng, years, 40_000, sig, 2 * q)
    model = truth.truth_error_model({"Z": Z, "years": years},
                                    sites={"wadden": {"Z": Zw, "years": years}},
                                    pnoa={"rmse_m": 0.15, "random_share": 0.5},
                                    windows=("W3",), survey_years=[2020, 2024])
    assert model["sites"]["wadden"]["ratio"] == pytest.approx(2.0, rel=0.15)
    assert model["sites"]["wadden"]["sigma2_s"] == model["sites"]["escalda"]["sigma2_s"]
    e = model["D_W"]["escalda"]["W3"]["2020"]
    assert model["D_W"]["wadden"]["W3"]["2020"]["D_W"] == pytest.approx(
        e["D_W"] * model["sites"]["wadden"]["ratio"])
    s2 = model["sites"]["escalda"]["sigma2_s"]
    assert truth.sigma2_T(model, "escalda", "W3", 2020) == pytest.approx(s2 + e["D_W"])
    assert truth.sigma2_T(model, "escalda", "W3", 2020, mult=1.5) == pytest.approx(
        1.5 * (s2 + e["D_W"]))
    assert truth.sigma2_T(model, "villaviciosa") == pytest.approx(
        0.0225 * 0.5 + 0.0225 * 0.5 / 25)
    assert truth.reliability(1.0, 0.2) == pytest.approx(0.8)
    assert len(model["sha256"]) == 64 and model["sensitivity"] == [0.5, 1.0, 1.5]
    # same-year overlapping sheets give the noise directly
    a = {"x": np.arange(50.0), "y": np.arange(40.0), "years": np.array([2020]),
         "z": rng.normal(0, 1, (1, 40, 50))}
    b = {"x": np.arange(30.0, 80.0), "y": np.arange(40.0), "years": np.array([2020, 2021]),
         "z": np.concatenate([a["z"][:, :, 30:50], rng.normal(size=(1, 40, 20))], axis=2)}
    b["z"] = np.stack([b["z"][0] + rng.normal(0, 0.1, b["z"][0].shape),
                       rng.normal(size=b["z"][0].shape)])
    ov = truth.overlap_noise(a, b)
    assert ov["n"] == 800 and ov["sigma2_s"] == pytest.approx(0.01 / 2, rel=0.2)
    with pytest.raises(ValueError):
        truth.pnoa_noise(0.1, 1.5)


def test_v2_isotonic_and_intercepts():
    assert truth.isotonic([1.0, 3.0, 2.0, 4.0]).tolist() == [1.0, 2.5, 2.5, 4.0]
    dt = np.arange(1, 9.0)
    assert truth.isotonic_intercept(dt, 0.05 + 0.02 * dt) == pytest.approx(0.05)
    assert truth.powerlaw_intercept(dt, 0.05 + 0.03 * dt ** 0.7) == pytest.approx(0.05, abs=1e-4)
    assert truth.isotonic_intercept(dt, np.full(8, 0.04)) == pytest.approx(0.04)
    assert np.isnan(truth.powerlaw_intercept(dt[:2], dt[:2]))


def test_v2_strata_arrays():
    from pyintertidal import hydroperiod

    s = np.array([0.0, 4.0, 10.0, 20.0, 40.0, np.nan])
    tau, cls = truth.reference_lag(s, [20.0, 10.0, np.nan], [60.0, 20.0, 5.0])
    assert tau[:5].tolist() == [0.0, 8.0, 20.0, 60.0, 60.0] and np.isnan(tau[5])
    assert cls.tolist() == [0, 0, 1, 3, 3, -1]
    amp, acl = truth.amplification(s, [10.0, 20.0], [1.1, 1.2])
    assert amp[1] == pytest.approx(1.04) and acl.tolist() == [1, 1, 2, 3, 3, -1]
    assert truth.classify([0.1, 0.0999, 2.0], truth.SLOPE_EDGES_PCT).tolist() == [1, 0, 3]
    rng = np.random.default_rng(0)
    h = 2.0 * np.sin(np.linspace(0, 300, 20_000))
    z = rng.uniform(-2.5, 2.5, (12, 10))
    z[0, 0] = np.nan
    zone, hp = truth.hydroperiod_zone(z, h)
    want, _ = hydroperiod.zonation(z, h)
    assert np.array_equal(zone, want) and np.isnan(hp[0, 0])
    pos = truth.sampled_range_position(np.array([-3.0, 0.0, 3.0, np.nan]), [-1.0, 1.0, np.nan])
    assert pos.tolist() == [-1, 0, 1, -2]
    yy, xx = np.mgrid[0:30, 0:40]
    plane = 0.01 * 10.0 * xx                         # 1 % along x at 10 m pixels
    sl = truth.slope_percent(plane, 10.0)
    assert np.allclose(sl[5:-5, 5:-5], 1.0, atol=1e-5)
    plane_nan = plane.copy()
    plane_nan[10, 10] = np.nan
    assert np.isnan(truth.slope_percent(plane_nan, 10.0)[10, 10])
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    d = truth.distance_to(m, 10.0)
    assert d[2, 2] == 0 and d[2, 4] == 20.0 and d[0, 0] == pytest.approx(np.hypot(20, 20))
    assert np.isinf(truth.distance_to(np.zeros((2, 2), bool), 10.0)).all()
    zz = np.linspace(-1, 1, 9)
    st = truth.strata_arrays(zz, s_km=np.linspace(0, 40, 9), band=np.arange(9) // 3,
                             judges_s=[10.0, 20.0], judges_tau=[20.0, 70.0],
                             judges_ratio=[1.0, 1.2], h_ref=h, h_at_instants=h[::500],
                             slope_pct=np.linspace(0, 3, 9), rho11=np.arange(9.0),
                             ndvi=np.linspace(-0.2, 0.5, 9),
                             turbidity_by_band={0: 0.01, 1: 0.05, 2: 0.2},
                             n_clear=np.arange(9) * 10, epoch_gap=np.linspace(0, 6, 9))
    assert st["tau_ref_class"].tolist() == [0, 1, 1, 2, 3, 3, 3, 3, 3]
    assert st["substrate_class"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert st["turbidity_class"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert st["epoch_gap_class"].tolist() == [0, 0, 1, 2, 2, 2, 3, 3, 3]
    assert st["vegetation_class"].tolist()[0] == 0 and st["vegetation_class"].tolist()[-1] == 2
    assert set(st["edges"]) >= {"tau_ref_class", "amp_class", "zone", "slope_class",
                                "substrate_class", "turbidity_class", "n_clear_class"}
    bare = truth.strata_arrays(zz)
    assert bare == {"edges": {}}
    assert truth.low_tide_positions([3.0, 0.0, 1.0, 2.0, np.nan]).tolist() == [1]


class _FakeSource:
    """The SceneSource surface the strata helpers read."""

    def __init__(self, data, offsets, lattice):
        self.data = data
        T, H, W = data["SCL"].shape
        self.grid = types.SimpleNamespace(H=H, W=W)
        self.offsets = np.asarray(offsets, np.int64)
        self.lattice20 = lattice
        self.calls = []

    def read_block(self, qs, bands, rows=None):
        rows = slice(None) if rows is None else rows
        self.calls.append((tuple(int(q) for q in qs), rows.start, rows.stop))
        return {b: self.data[b][np.asarray(qs)][:, rows] for b in bands}


def _fake_scenes(rng, T=7, H=9, W=8):
    b03 = rng.integers(200, 1500, (T, H, W)).astype(np.int16)
    b08 = rng.integers(200, 1500, (T, H, W)).astype(np.int16)
    b04 = rng.integers(100, 1500, (T, H, W)).astype(np.int16)
    b11 = rng.integers(100, 3000, (T, H, W)).astype(np.int16)
    scl = rng.choice(np.array([4, 5, 6, 8, 9], np.int16), (T, H, W))
    b03[0, 0, 0] = -32768
    return {"B03": b03, "B04": b04, "B08": b08, "B11": b11, "SCL": scl}


def test_v2_low_tide_dry_medians_equal_the_frame_computation():
    from experiments.benchmark import obs

    rng = np.random.default_rng(21)
    data = _fake_scenes(rng)
    offs = [0, 1000, 0, 0, 1000, 0, 0]
    for k in (1, 4):                                  # the delivered +1000 offset
        for b in ("B03", "B04", "B08", "B11"):
            data[b][k] = np.where(data[b][k] == -32768, -32768, data[b][k] + 1000)
    src = _FakeSource(data, offs, (1, 0))
    pos = np.array([0, 1, 3, 4, 6])
    whole = truth.low_tide_dry_medians(src, pos, block_rows=100)
    blocks = truth.low_tide_dry_medians(src, pos, block_rows=2)
    for k in ("rho11", "ndvi", "n_dry"):
        assert np.array_equal(whole[k], blocks[k], equal_nan=True), k
    assert all(c[1] in (0, 1, 3, 5, 7) for c in src.calls)  # bounds on lattice rows
    # brute force on the whole frame
    raw = {b: data[b][pos] for b in data}
    o = obs.observe_setting(raw, "C1.floor0.ndwi.D1", offset=np.asarray(offs)[pos],
                            lattice=(1, 0))
    dry = o.V & ~o.W
    b11 = obs.prep_band(raw["B11"], np.asarray(offs)[pos], "floor0")
    b11 = obs.expand20(obs.block_mean20(b11, (1, 0)), (1, 0), b11.shape[-2:]) / np.float32(1e4)
    for r in range(9):
        for c in range(8):
            v = b11[:, r, c][dry[:, r, c]]
            if len(v):
                assert whole["rho11"][r, c] == pytest.approx(np.median(v), rel=1e-6)
            else:
                assert np.isnan(whole["rho11"][r, c])
    assert whole["n_dry"].sum() == dry.sum()
    empty = truth.low_tide_dry_medians(src, [])
    assert np.isnan(empty["rho11"]).all() and empty["n_dry"].sum() == 0


def test_v2_anchor_band_medians():
    rng = np.random.default_rng(8)
    data = _fake_scenes(rng)
    data["SCL"][:] = 6
    src = _FakeSource(data, [0] * 7, (0, 0))
    W = 8
    anchors = np.array([0 * W + 1, 2 * W + 3, 8 * W + 7, 5 * W + 5])
    band = np.array([0, 0, 1, 1])
    clear = np.array([True, False, True, True, True, False, True])
    pos = np.arange(7)
    got = truth.anchor_band_medians(src, pos, anchors, band, clear, block_rows=3)
    for b in (0, 1):
        a = anchors[band == b]
        per = [np.median(data["B04"][p].ravel()[a] / 1e4) for p in pos[clear]]
        assert got[b] == pytest.approx(np.median(per), rel=1e-6)
    none = truth.anchor_band_medians(src, pos, anchors, band, np.zeros(7, bool))
    assert all(np.isnan(v) for v in none.values())


def test_v2_dated_epochs_and_vak_meta_tables():
    from experiments.benchmark.fetch import vak_meta

    years = np.array([2021, 2022])
    SRC = np.array([[0, 1], [0, 2], [-1, 1]], np.int16)
    sheet = np.array([0, 0, 1])
    ep = truth.dated_epochs(SRC, years, sheet, [{1: 2022.41}, {1: 2022.9}])
    assert ep.tolist() == [[2021.5, 2022.41], [2021.5, 2022.5], [2021.5, 2022.9]]
    assert np.array_equal(truth.dated_epochs(SRC, years, sheet, None),
                          [[2021.5, 2022.5]] * 3)
    p = vak_meta.parse_source_name
    assert p("_20160101_Westerschelde_-_binnen_2016lo.asc", "2016-01-01")["method"] == "sounding"
    assert p("_20180101_wsch18tt20.asc")["method"] == "combined"
    assert p("_20190101_x_2019la20.asc")["method"] == "lidar"
    assert p("_20240101_BodemWS.asc")["method"] == "unknown"
    r = p("_20220101_WesterscheldeMonding_22_5_30.asc", "2022-01-01")
    assert (r["survey_date"], r["date_status"]) == ("2022-05-30", "filename (probable)")
    r = p("_20210101_WesterscheldeBi_2021_3_1.asc", "2021-01-01")
    assert r["survey_date"] is None                      # a part number, not a date
    r = p("KB113_4948_20050302.asc", "2005-03-02")
    assert (r["survey_date"], r["date_status"]) == ("2005-03-02", "stamp")
    assert p("_20230101_x_23_5_30.asc", "2022-01-01")["survey_date"] is None
    meta = {"sheets": {"A": {"stamps": [{"sources": [
        {"index": 0, "method": "sounding", "survey_date": None},
        {"index": 1, "method": "combined", "survey_date": "2022-05-30"}]}]},
        "B": {"stamps": [{"sources": [{"index": 1, "method": "lidar", "survey_date": None}]}]}}}
    sd = vak_meta.src_dates(meta, ["A", "B"])
    assert sd[0][0] is None and sd[0][1] == pytest.approx(2022.41, abs=0.01)
    tabs = vak_meta.class_tables(meta, ["A", "B"])
    assert tabs == [{0: 1, 1: 3}, {1: 2}]
    cls = vak_meta.cell_source_class(SRC, sheet, tabs)
    assert cls.tolist() == [[1, 3], [1, 0], [-1, 2]]
    assert vak_meta.decimal_year("2021-01-01") == pytest.approx(2021 + 0.5 / 365)


def _write_sheet_nc(path, xr):
    t = pd.to_datetime(["2015-01-01", "2020-01-01", "2022-01-01", "2022-05-30", "2024-01-01"])
    z = np.full((5, 3, 4), np.nan)
    z[0] = 9.0
    z[1, :, :2] = 1.0
    z[2] = 2.0
    z[3, 0, :] = 3.0                                  # a later 2022 stamp, partial
    z[4, 1:, :] = 4.0
    isrc = np.where(np.isfinite(z), np.arange(5)[:, None, None], np.nan).astype("float32")
    ds = xr.Dataset({"z": (("time", "y", "x"), z), "isource": (("time", "y", "x"), isrc)},
                    coords={"time": t, "y": [100.0, 80.0, 60.0], "x": [0.0, 20.0, 40.0, 60.0]})
    ds["isource"].attrs["flag_meanings"] = ("KB1_20150101.asc _20200101_a_2020lo20.asc "
                                            "_20220101_b_2022tt20.asc _20220101_c_22_5_30.asc "
                                            "_20240101_BodemWS.asc")
    ds.to_netcdf(path)
    return path


def test_v2_read_sheet_epochs_and_vak_meta_build(tmp_path):
    xr = pytest.importorskip("xarray")
    path = _write_sheet_nc(str(tmp_path / "vaklodingenKB1.nc"), xr)
    sh = truth.read_sheet_epochs(path)
    assert sh["years"].tolist() == [2020, 2022, 2024]
    assert np.isnan(sh["z"][0][:, 2:]).all() and (sh["z"][0][:, :2] == 1.0).all()
    assert (sh["z"][1][0] == 3.0).all() and (sh["z"][1][1:] == 2.0).all()   # later stamp wins
    assert (sh["src"][1][0] == 3).all() and (sh["src"][1][1:] == 2).all()
    assert sh["stamps"][2022] == ["2022-01-01", "2022-05-30"] and len(sh["sources"]) == 5
    from experiments.benchmark.fetch import vak_meta

    ov = tmp_path / "ov.json"
    product.write_json(str(ov), {"_20220101_b_2022tt20.asc": {"transect_spacing_m": 200}})
    out = vak_meta.build("escalda", out_dir=str(tmp_path / "vm"), sheets={"KB1": path},
                         overrides_path=str(ov))
    meta = product.read_json(out)
    st = meta["sheets"]["KB1"]["stamps"]
    assert [s["date"] for s in st] == ["2020-01-01", "2022-01-01", "2022-05-30", "2024-01-01"]
    s22 = st[1]["sources"][0]
    assert s22["method"] == "combined" and s22["transect_spacing_m"] == 200 and s22["override"]
    assert st[2]["sources"][0]["survey_date"] == "2022-05-30"
    assert st[2]["sources"][0]["date_status"] == "stamp"
    assert meta["cells_by_method"]["sounding"] == 6 and meta["overrides"]["sha256"]


def test_v2_pnoa_meta_parsing(tmp_path):
    from experiments.benchmark.fetch import pnoa_meta

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<gmd:MD_Metadata xmlns:gmd="http://www.isotc211.org/2005/gmd"
  xmlns:gco="http://www.isotc211.org/2005/gco" xmlns:gml="http://www.opengis.net/gml">
 <gmd:dateStamp><gco:Date>2021-03-01</gco:Date></gmd:dateStamp>
 <gmd:identificationInfo><gmd:citation><gmd:title>
   <gco:CharacterString>MDT02 0015-3 COB2</gco:CharacterString></gmd:title></gmd:citation>
 <gmd:extent><gmd:EX_Extent><gmd:temporalElement><gmd:EX_TemporalExtent><gmd:extent>
  <gml:TimePeriod><gml:beginPosition>2016-07-02</gml:beginPosition>
  <gml:endPosition>2016-07-03</gml:endPosition></gml:TimePeriod>
 </gmd:extent></gmd:EX_TemporalExtent></gmd:temporalElement></gmd:EX_Extent></gmd:extent>
 </gmd:identificationInfo>
 <gmd:dataQualityInfo><gmd:report><gmd:DQ_AbsoluteExternalPositionalAccuracy>
  <gmd:nameOfMeasure><gco:CharacterString>RMSE Z (vertical)</gco:CharacterString></gmd:nameOfMeasure>
  <gmd:result><gmd:DQ_QuantitativeResult><gmd:value><gco:Record>
   <gco:Decimal>0.15</gco:Decimal></gco:Record></gmd:value></gmd:DQ_QuantitativeResult></gmd:result>
 </gmd:DQ_AbsoluteExternalPositionalAccuracy></gmd:report></gmd:dataQualityInfo>
</gmd:MD_Metadata>"""
    xp = tmp_path / "meta.xml"
    xp.write_text(xml, encoding="utf-8")
    d = pnoa_meta.parse_iso19139(xml)
    assert (d["begin"], d["end"]) == ("2016-07-02", "2016-07-03")
    assert d["vertical_rmse_m"] == 0.15 and d["title"] == "MDT02 0015-3 COB2"
    doc = pnoa_meta.build([str(xp)], out=str(tmp_path / "pm.json"))
    assert not doc["flight_known"] and doc["flight_dates"] == ["2016-07-02", "2016-07-03"]
    assert doc["flights"][0]["status"] == "date only" and pnoa_meta.flight_instants(doc) == []
    tab = tmp_path / "flights.csv"
    pd.DataFrame({"strip": ["P1", "P2"],
                  "start_utc": ["2016-07-02T10:00:00+02:00", "2016-07-02 10:30"],
                  "end_utc": ["2016-07-02T10:20:00+02:00", "2016-07-02 10:50"]}).to_csv(
        tab, index=False)
    doc = pnoa_meta.build([str(xp)], table_path=str(tab), out=str(tmp_path / "pm.json"))
    assert doc["flight_known"] and len(doc["sources"]) == 2
    usable = [f for f in doc["flights"] if f["usable"]]
    assert [f["strip"] for f in usable] == ["P1"]
    assert usable[0]["start_utc"] == "2016-07-02 08:00:00"
    assert [f["status"] for f in doc["flights"] if f["strip"] == "P2"] == ["no time zone"]
    assert pnoa_meta.flight_instants(doc) == [pd.Timestamp("2016-07-02 08:10")]
    assert pnoa_meta.load(str(tmp_path / "pm.json"))["sha256"] == doc["sha256"]
    assert pnoa_meta.load(str(tmp_path / "absent.json")) is None


def test_v2_lag_truths_per_window_recover_the_planted_lag(site):
    st = site.stations["st_in"]
    g = site.series["st_in"]
    seen = {}
    for wid in ("W3", "Y2024"):
        r, cv = truth.lag_truths_window(site.pkg, st, g, wid, H=site.H, constituents=CONS,
                                        conventions=SYN_CONV)
        pos = truth.window_positions(site.pkg.t, wid)
        assert r["a"]["n_record"] == len(pos) and r["a"]["coverage"] == 1.0
        assert abs(r["a"]["tau"] - LAG) <= 1 and r["a"]["lo"] <= LAG <= r["a"]["hi"]
        assert abs(r["ab"]["tau"] - LAG) <= 1 and r["ab"]["gain"] == pytest.approx(1.0, abs=0.03)
        assert r["b"]["tau"] == pytest.approx(LAG, abs=0.3) and r["b"]["lo"] <= LAG <= r["b"]["hi"]
        assert r["amp_ratio"] == pytest.approx(1.0, abs=0.01)
        assert r["all"]["tau"] == LAG
        hl = r["hw_lw"]
        for k in ("hw", "lw"):
            assert hl[k]["n"] > (1000 if wid == "W3" else 300)
            assert abs(hl[k]["median_min"] - LAG) <= 15, hl
        assert abs(hl["mean_min"] - LAG) <= 15 and abs(hl["distortion_min"]) <= 20
        seen[wid] = (r, cv)
    assert seen["W3"][0]["b"]["group"] == "quarter" and seen["Y2024"][0]["b"]["group"] == "month"
    assert seen["Y2024"][0]["b"]["n_groups"] == 12 and seen["W3"][0]["b"]["n_groups"] == 12
    s3, s24 = seen["W3"][0]["a"]["boot"]["seed"], seen["Y2024"][0]["a"]["boot"]["seed"]
    assert s3[:3] == s24[:3] and (s3[3], s24[3]) == (3, 2024)
    # Y2024's record is the 2024 instants only
    t24 = pd.DatetimeIndex(site.pkg.t)[truth.window_positions(site.pkg.t, "Y2024")]
    assert (t24.year == 2024).all() and len(t24) > 20
    # the declared timing sensitivity moves (a) by the shift
    r, _ = truth.lag_truths_window(site.pkg, st, g, "W3", H=site.H, constituents=CONS,
                                   conventions=SYN_CONV, shift_min=-10.0, boot=False)
    assert abs(r["a"]["tau"] - (LAG - 10)) <= 1 and r["shift_min"] == -10.0
    with pytest.raises(KeyError):
        truth.lag_truths_window(site.pkg, st, g, "W3", H=site.H, constituents=CONS)


def test_v2_build_lag_truth_per_window_end_to_end(site, tmp_path):
    table = [r for r in judges.build(site.pkg, path=f"{site.cfg.truth_dir}/judges.json")
             if r["code"] != "st_out"]
    out = str(tmp_path / "lag")
    path = truth.build_lag_truth_v2(site.pkg, windows=("W3", "Y2023", "Y2024"), table=table,
                                    constituents=CONS, out=out, conventions=SYN_CONV)
    doc = truth.load_lag_truth_v2(site.cfg, out=out)
    assert doc["development_judges"] == ["harlingen.waddenzee"]
    by = {s["code"]: s for s in doc["stations"]}
    for wid in ("W3", "Y2023", "Y2024"):
        e = by["st_in"]["windows"][wid]
        assert e["inclusion"]["included"] and abs(e["a"]["tau"] - LAG) <= 1
        assert e["inclusion"]["window"] == wid
    q = by["st_quay"]["windows"]
    assert "coverage" in q["Y2024"]["inclusion"]["exclusion"]       # ends 2024-02-15
    assert q["Y2024"]["inclusion"]["coverage"] < 0.5 and "a" not in q["Y2024"]
    assert q["Y2023"]["inclusion"]["coverage"] > 0.9               # covered in 2023
    csv = pd.read_csv(f"{out}/lag_truth_v2.csv")
    assert set(csv["window"][csv["judge"] == "st_in"]) == {"W3", "Y2023", "Y2024"}
    assert list(csv.columns) == list(truth.LAG_V2_CSV_COLUMNS)
    s, tau, amp = truth.judge_nodes(doc, "W3")
    assert len(s) == 1 and abs(tau[0] - LAG) <= 1 and amp[0] == pytest.approx(1.0, abs=0.01)
    m0 = os.stat(path).st_mtime_ns
    truth.build_lag_truth_v2(site.pkg, windows=("W3", "Y2023", "Y2024"), table=table,
                             constituents=CONS, out=out, conventions=SYN_CONV)
    assert os.stat(path).st_mtime_ns == m0


def test_v2_build_elevation_truth_end_to_end(site, tmp_path):
    pkg = site.pkg
    grid = judges.grid_of(pkg)
    XQ, YQ = truth.rd_coords(grid, (H, W))
    rng = np.random.default_rng(12)
    x = np.arange(np.floor(XQ.min() / 20) * 20 - 40, XQ.max() + 60, 20.0)
    y = np.arange(np.ceil(YQ.max() / 20) * 20 + 40, YQ.min() - 60, -20.0)
    years = np.array([2019, 2022, 2025])
    z = rng.uniform(-3.0, 3.0, (3, len(y), len(x)))
    z[0, :, : len(x) // 3] = np.nan                     # part never surveyed in 2019
    sheets = {"S1": {"x": x, "y": y, "years": years, "z": z}}
    h1 = 1.4 * np.sin(np.linspace(0, 900 * np.pi, 300_000))
    cfg = dataclasses.replace(pkg.cfg, elevation_truth=("vaklodingen",),
                              root=str(tmp_path / "out").replace("\\", "/"))
    proxy = types.SimpleNamespace(cfg=cfg, full=pkg.full, shape=pkg.shape, grid=pkg.grid,
                                  band_frame=pkg.band_frame, keys=pkg.keys)
    table = [{"nbhd": pkg.nb["nbhd"]["st_in"]}]
    out = str(tmp_path / "elev")
    path = truth.build_elevation_truth_v2(proxy, windows=("W3", "V2019L3"), h1=h1,
                                          table=table, sheets=sheets, out=out)
    arrays, doc = truth.load_elevation_truth_v2(cfg, out=out)
    rng_ = truth.range_from_series(h1)
    C = truth.vaklodingen_cells(XQ, YQ, sheets)
    assert np.array_equal(arrays["cell_id"], C["cell_id"])
    for wid, want_year in (("W3", 2025), ("V2019L3", 2019)):
        zz, yr, gap = truth.survey_matched(C["Z"], C["years"], truth.window_of(wid).centre)
        tr, _ = truth.apply_range(zz, rng_["lo"], rng_["hi"])
        assert np.array_equal(arrays[f"{wid}.truth"], tr, equal_nan=True)
        assert np.array_equal(arrays[f"{wid}.year"], yr)
        assert (yr[np.isfinite(zz)] == want_year).mean() > 0.5
    # V2019L3 falls back to 2022 where 2019 was not surveyed (2022 is nearer than 2025)
    yv = arrays["V2019L3.year"]
    assert set(np.unique(yv[yv >= 0]).tolist()) == {2019, 2022}
    pc = C["point_cell"]
    inD = arrays["cell_in_D_union"]
    want_d = np.flatnonzero((pc >= 0) & inD[np.maximum(pc, 0)])
    assert np.array_equal(arrays["d_truth_union"], want_d)
    S = np.unique(np.concatenate([want_d, pkg.full.keep, pkg.nb["nbhd"]["st_in"]]))
    assert np.array_equal(arrays["support"], S)
    assert np.array_equal(arrays["support_cell"], pc[S])
    assert doc["views"]["W3"]["n_cells_D_truth"] == int(np.isfinite(arrays["W3.truth"]).sum())
    assert doc["range"]["lo"] == pytest.approx(rng_["lo"]) and "latest_2019_2025" in doc["views"]
    assert doc["years_kept"] == [2019, 2022, 2025]
    m0 = os.stat(path).st_mtime_ns
    truth.build_elevation_truth_v2(proxy, windows=("W3", "V2019L3"), h1=h1, table=table,
                                   sheets=sheets, out=out)
    assert os.stat(path).st_mtime_ns == m0
    with pytest.raises(ValueError):
        truth.build_elevation_truth_v2(proxy, windows=("W3",), table=table, sheets=sheets,
                                       out=out, force=True)


def test_v2_frozen_truth_sections():
    plain = truth.truth_v2_plain()
    assert set(plain) == {"truth_elev", "strata", "truth_lag", "truth_error"}
    assert plain["truth_elev"]["range"]["quantiles"] == [0.001, 0.999]
    assert plain["truth_elev"]["pnoa"]["patch_min_px"] == 50
    assert plain["truth_elev"]["stable"] == {"min_surveys": 2, "pad_years": 1,
                                             "max_range_m": 0.2}
    assert plain["strata"]["reference_lag_edges_min"] == [10.0, 30.0, 60.0, 120.0]
    assert truth.truth_v2_sha() == truth.truth_v2_sha() and len(truth.truth_v2_sha("strata")) == 64
    with pytest.raises(TypeError):
        truth.TRUTH_V2["strata"]["slope_edges_pct"] = (1,)


def test_v2_default_truth_windows_cover_every_campaign_window():
    """The CLI builds the truths over every Campaign-1 window by default
    (review of 2026-10-14): D_truth, hence S0 and dom, is the union over the
    windows the elevation units are scored in, and freeze_dom refuses a
    truth that lacks one."""
    import inspect

    from experiments.benchmark import designs, package
    elev = inspect.signature(truth.build_elevation_truth_v2).parameters["windows"].default
    lag = inspect.signature(truth.build_lag_truth_v2).parameters["windows"].default
    assert tuple(elev) == truth.ELEV_WINDOWS_1A == package.FREEZE_TRUTH_WINDOWS
    assert tuple(lag) == truth.LAG_WINDOWS_1A
    used = {u.window for d in designs.QUEUE_1A for u in designs.build(d)
            if u.status == "planned" and u.design != "lag"
            and designs.METHODS[u.method].kind in ("elev", "both")}
    assert used <= set(elev)
    assert set(designs.LAG_WINDOWS) | {"Y2025"} <= set(lag)
