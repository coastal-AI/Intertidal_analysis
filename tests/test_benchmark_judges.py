"""Tests of experiments.benchmark.judges, the in-estuary gauges that judge the lags.

On the synthetic site of ``tests/benchmark_site.py`` (three listed judges:
``st_in`` on the flats, ``st_quay`` on the quay at the landward head with a
short, drying record, ``st_out`` 20 km west of the frame; one unlisted
station) the tests pin the rules of the module docstring (v13 spec §1.6,
§2.8):

* stations: listed order, the copied fields, and the refusals (a missing
  judge, another site's station, a boundary member);
* series as delivered: sorted, non-finite values and repeated instants
  dropped; the nearest value within 6 min; the centred 10-min series (as
  delivered at 10 min; the mean over [T - 5, T + 5] min with at least 2
  samples of a 5-min and 6 of a 1-min series) equal brute force;
* geometry on the run grid: the pixel holding a point, the neighbourhood
  (every pixel centre within R) equal brute force, the radius rule
  (R = max(500, d_min + 300), +250 m while fewer than 100 keep_full pixels,
  never beyond the 2,000 m cap), the modal band and median s of the 200
  nearest keep_full pixels; the neighbourhoods equal the package's NB;
* the method-independent inclusion rules (in frame, coverage >= 50 %,
  > 3 km from every boundary member, <= 10 % of values within 1 cm of the
  minimum) and their exclusion order;
* the judge table: build, save, load with the neighbourhoods recomputed
  and checked against the stored sha256; a tampered table is refused.

Validation v2 (the ``test_v2_*`` tests, spec §1.7, §6.1): the coverage rule
per window (a judge whose record ends in February 2024 judges 2023 but not
2024 or 2025), the window-independent rules read from the table record; the
headline, without-mouth and development-free location sets; the 13 headline
locations as a check; the providers' time-stamp conventions (centring,
the +- half-window shift, refusal of an unknown provider, the conventions
file read back); and ``fetch judges`` (script arguments, coverage per year,
the extension plan and its files).

Run:  python -X utf8 -m pytest tests/test_benchmark_judges.py -q
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, judges, product  # noqa: E402
from tests.benchmark_site import (  # noqa: E402
    GAUGES, H, JUDGES, PIXEL_M, SITE, STATION_PX, W, X0, Y0, gauge_series, make_site,
    station_lonlat, station_record, use_stations, write_gauges, write_series_csv)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    s = make_site(tmp_path_factory.mktemp("v13_judges"), mp)
    use_stations(mp, s.inp)
    s.gauges = write_gauges(s.inp)
    s.stations = {st["code"]: st for st in judges.stations(SITE)}
    yield s
    s.pkg.close()
    mp.undo()


def _grid():
    xs = X0 + PIXEL_M * np.arange(W)
    ys = Y0 - PIXEL_M * np.arange(H)
    return {"xs": xs, "ys": ys}


def _brute_nbhd(grid, gx, gy, R):
    xs, ys = grid["xs"], grid["ys"]
    rr, cc = np.mgrid[0:len(ys), 0:len(xs)]
    return np.flatnonzero((np.hypot(xs[cc] - gx, ys[rr] - gy) <= R).ravel())


# ─────────────────────────────────────────────────────────────────────────────
#  stations and series
# ─────────────────────────────────────────────────────────────────────────────

def test_stations_listed_in_order_with_their_fields(site):
    st = judges.stations(SITE)
    assert [s["code"] for s in st] == list(JUDGES)
    assert set(st[0]) == set(judges.STATION_KEYS)
    assert st[0]["sampling_min"] == GAUGES["st_in"]["sampling_min"]
    assert judges.stations("villaviciosa") == []          # no judges listed


def test_stations_refusals(site, tmp_path, monkeypatch):
    recs = [station_record(c, *STATION_PX[c]) for c in JUDGES]
    path = tmp_path / "stations.json"

    def write(rows):
        path.write_text(json.dumps(rows), encoding="utf-8")
        return str(path)

    with pytest.raises(KeyError):
        judges.stations(SITE, write(recs[:2]))
    bad = [dict(r) for r in recs]
    bad[1]["site"] = "wadden"
    with pytest.raises(ValueError):
        judges.stations(SITE, write(bad))
    bad = [dict(r) for r in recs]
    bad[0]["in_boundary"] = True
    with pytest.raises(ValueError):
        judges.stations(SITE, write(bad))


def test_read_series_sorts_and_drops_delivery_errors(tmp_path):
    t = pd.to_datetime(["2023-01-01 00:10:00", "2023-01-01 00:00:00", "2023-01-01 00:20:00",
                        "2023-01-01 00:20:00", "2023-01-01 00:30:00"])
    v = np.array([1.0, 0.5, np.nan, 2.0, 3.0])
    write_series_csv(tmp_path / "x.csv", t, v)
    s = judges.read_series({"code": "x", "file": "x.csv"}, base=str(tmp_path))
    assert s.index.is_monotonic_increasing and s.index.dtype == "datetime64[ns]"
    assert s.index.tz is None and s.dtype == np.float64
    assert s.index.strftime("%H:%M").tolist() == ["00:00", "00:10", "00:20", "00:30"]
    assert s.tolist() == [0.5, 1.0, 2.0, 3.0]            # NaN dropped first, then 2.0 kept
    t2 = pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:00:00"])
    write_series_csv(tmp_path / "d.csv", t2, [4.0, 5.0])
    assert judges.read_series({"code": "d", "file": "d.csv"},
                              base=str(tmp_path)).tolist() == [4.0]
    pd.DataFrame({"t": ["2023-01-01 00:00:00"], "level": [1.0]}).to_csv(
        tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        judges.read_series({"code": "bad", "file": "bad.csv"}, base=str(tmp_path))


def test_at_instants_is_the_nearest_value_within_6_min():
    idx = pd.date_range("2023-01-01 00:00", periods=4, freq="20min")
    s = pd.Series([0.0, 1.0, 2.0, 3.0], index=idx)
    q = pd.to_datetime(["2023-01-01 00:06", "2023-01-01 00:07", "2023-01-01 00:14",
                        "2023-01-01 00:35", "2023-01-01 01:06", "2023-01-01 01:07",
                        "2022-12-31 23:54"])
    got = judges.at_instants(s, q)
    assert np.array_equal(got, [0.0, np.nan, 1.0, 2.0, 3.0, np.nan, 0.0], equal_nan=True)
    assert np.array_equal(judges.at_instants(s, q[:2], tol_min=10), [0.0, 0.0])


def test_centred_10min_equals_brute_force():
    grid = pd.date_range("2023-03-01 00:00", periods=40, freq="10min").as_unit("ns")
    rng = np.random.default_rng(4)
    for step, need in ((1, 6), (5, 2)):
        t = pd.date_range("2023-02-28 23:50", "2023-03-01 07:00", freq=f"{step}min")
        keep = rng.random(len(t)) > (0.5 if step == 1 else 0.4)      # gaps
        s = pd.Series(10.0 + rng.normal(size=len(t)), index=t)[keep]
        got = judges.centred_10min(s, step, grid=grid)
        assert got.index.equals(grid)
        x = s.index.asi8
        want = []
        for g in grid.asi8:
            m = (x >= g - 5 * judges.NS_PER_MIN) & (x <= g + 5 * judges.NS_PER_MIN)
            want.append(s.to_numpy()[m].mean() if m.sum() >= need else np.nan)
        assert np.allclose(got.to_numpy(), want, rtol=0, atol=1e-9, equal_nan=True)
        assert np.isnan(got.to_numpy()).any() and np.isfinite(got.to_numpy()).any()
    # a 10-min series is used as delivered, at the grid instants exactly
    t10 = grid[::2].append(pd.DatetimeIndex([grid[1] + pd.Timedelta(minutes=1)]))
    s10 = pd.Series(np.arange(len(t10), dtype=float), index=t10).sort_index()
    got = judges.centred_10min(s10, 10, grid=grid)
    assert np.array_equal(got.to_numpy()[::2], np.arange(20.0))
    assert np.isnan(got.to_numpy()[1::2]).all()
    for bad in (2, 2.5, 7):
        with pytest.raises(ValueError):
            judges.centred_10min(s10, bad, grid=grid)


# ─────────────────────────────────────────────────────────────────────────────
#  geometry
# ─────────────────────────────────────────────────────────────────────────────

def test_pixel_of_uses_half_step_edges():
    g = _grid()
    assert judges.pixel_of(g, X0, Y0) == (0, 0)
    assert judges.pixel_of(g, X0 + 99.9, Y0 - 99.9) == (0, 0)
    assert judges.pixel_of(g, X0 + 100.0, Y0 - 100.0) == (1, 1)
    assert judges.pixel_of(g, X0 - 100.1, Y0) is None
    assert judges.pixel_of(g, X0 + PIXEL_M * (W - 0.5) + 0.1, Y0) is None
    assert judges.pixel_of(g, X0, Y0 - PIXEL_M * (H - 0.5) - 0.1) is None


def test_nbhd_flat_equals_the_full_frame_test():
    g = _grid()
    rng = np.random.default_rng(2)
    for _ in range(30):
        gx = X0 + rng.uniform(-1500, PIXEL_M * W + 1500)
        gy = Y0 - rng.uniform(-1500, PIXEL_M * H + 1500)
        R = rng.uniform(0, 2500)
        got = judges.nbhd_flat(g, gx, gy, R)
        assert got.dtype == np.int64 and np.array_equal(got, _brute_nbhd(g, gx, gy, R))
    assert judges.nbhd_flat(g, X0 - 1e5, Y0, 500).size == 0
    assert np.array_equal(judges.nbhd_flat(g, X0, Y0, 200.0), [0, 1, W])   # boundary in


def test_neighbourhood_radius_rule():
    p = dict(config.FROZEN["judges"]["neighbourhood"])
    assert (p["r0_m"], p["pad_m"], p["step_m"], p["cap_m"], p["min_keep_px"]) == \
        (500, 300, 250, 2000, 100)
    # many pixels close by: R = max(500, d_min + 300)
    assert judges.neighbourhood_radius(np.arange(0, 1000, 1.0)) == 500.0
    assert judges.neighbourhood_radius(np.arange(400, 2000, 1.0)) == 700.0
    # widened by 250 m until 100 pixels lie within R
    d = np.concatenate([np.full(40, 100.0), np.full(40, 700.0), np.full(40, 1100.0)])
    assert judges.neighbourhood_radius(np.sort(d)) == 1250.0
    # the widening stops at the cap, even when a step would overshoot it
    d = np.sort(np.concatenate([[1530.0], np.full(5, 1600.0)]))
    assert judges.neighbourhood_radius(d) == 2000.0
    # a start beyond the cap (a station far from the keep) is kept as it is
    assert judges.neighbourhood_radius(np.array([20_000.0, 20_001.0])) == 20_300.0


def test_station_geometry_on_the_package(site):
    pkg = site.pkg
    grid = pkg.grid
    keep, s_km, band_of = pkg.full.keep, pkg.full.s_km, pkg.full.band_of
    for code in ("st_in", "st_quay"):
        st = site.stations[code]
        g = judges.geometry(pkg, st, grid=grid)
        lon, lat = station_lonlat(*STATION_PX[code])
        assert (st["lon"], st["lat"]) == (lon, lat)
        gx, gy = g["gx"], g["gy"]
        assert gx == pytest.approx(X0 + PIXEL_M * STATION_PX[code][1], abs=1e-3)
        assert gy == pytest.approx(Y0 - PIXEL_M * STATION_PX[code][0], abs=1e-3)
        assert g["in_frame"] and (g["row"], g["col"]) == tuple(
            int(np.floor(v + 0.5)) for v in STATION_PX[code])
        d = np.hypot(grid["xs"][keep % W] - gx, grid["ys"][keep // W] - gy)
        order = np.argsort(d, kind="stable")
        near = order[:200]
        bn = band_of[near]
        assert g["k_g"] == int(np.bincount(bn[bn >= 0]).argmax())
        assert g["k_g_share"] == pytest.approx(np.mean(bn == g["k_g"]))
        assert g["s_g_km"] == pytest.approx(np.nanmedian(s_km[near]))
        assert g["d_min_km"] == pytest.approx(d.min() / 1000.0)
        assert g["R_g_m"] == judges.neighbourhood_radius(np.sort(d))
        assert np.array_equal(g["nbhd"], _brute_nbhd(grid, gx, gy, g["R_g_m"]))
        assert g["n_keep_in_nbhd"] == int((d <= g["R_g_m"]).sum()) >= 100
        assert g["nbhd_sha256"] == config.sha256_bytes(g["nbhd"].astype("<i8").tobytes())
        # the package NB used the same rule on the same keep_full
        assert np.array_equal(g["nbhd"], pkg.nb["nbhd"][code])
        assert g["centre_k_km"] == pkg.full.centers[g["k_g"]]
    out = judges.geometry(pkg, site.stations["st_out"], grid=grid)
    assert not out["in_frame"] and out["row"] is None and "outside_frame" in out["flags"]
    assert out["d_min_km"] > 19.0 and out["R_g_m"] > 2000.0
    assert np.array_equal(judges.nb_flat(pkg), pkg.nb["nb"])
    with pytest.raises(ValueError):
        judges.station_geometry(0.0, 0.0, grid, [], [], [])


# ─────────────────────────────────────────────────────────────────────────────
#  inclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_haversine_and_location():
    assert judges.haversine_km(52.0, 4.0, 53.0, 4.0) == pytest.approx(111.195, abs=1e-3)
    assert judges.haversine_km(52.0, 4.0, 52.0, 4.0) == 0.0
    assert judges.location_of("terneuzen.westsluis.buiten") == "terneuzen"
    assert judges.location_of("terneuzen") == "terneuzen"
    assert judges.location_of("borssele") == "borssele"


def test_inclusion_rules():
    st = {"lat": 51.40, "lon": 3.80}
    g = np.linspace(0.0, 2.0, 100)
    near = {"code": "near", "lat": 51.40 + 2.9 / 111.195, "lon": 3.80}
    far = {"code": "far", "lat": 51.40 + 3.1 / 111.195, "lon": 3.80}
    r = judges.inclusion_rules(st, g, True, [far])
    assert r["included"] and r["exclusion"] == [] and r["coverage"] == 1.0
    assert r["nearest_member"] == "far" and r["nearest_member_km"] == pytest.approx(3.1, 1e-3)
    assert r["dry_share"] == pytest.approx(1 / 100)
    r = judges.inclusion_rules(st, g, True, [far, near])
    assert not r["included"] and r["exclusion"] == ["boundary_member"]
    assert judges.inclusion_rules(st, g, True, [])["nearest_member_km"] == float("inf")
    # coverage: exactly 50 % passes, below fails
    half = g.copy()
    half[50:] = np.nan
    assert judges.inclusion_rules(st, half, True, [])["passed"]["coverage"]
    half[49] = np.nan
    assert judges.inclusion_rules(st, half, True, [])["exclusion"] == ["coverage"]
    # dry: more than 10 % of the values within 1 cm of their minimum
    dry = np.maximum(g, 0.3)                      # 15 % at the floor
    r = judges.inclusion_rules(st, dry, False, [near])
    assert r["dry_share"] == pytest.approx(0.16)
    assert r["exclusion"] == ["in_frame", "boundary_member", "dry"]
    ok = np.maximum(g, 0.15)                      # 8 % at the floor
    assert judges.inclusion_rules(st, ok, True, [])["passed"]["dry"]
    none = judges.inclusion_rules(st, np.full(5, np.nan), True, [])
    assert none["exclusion"] == ["coverage", "dry"] and np.isnan(none["dry_share"])


def test_judge_table_build_save_load(site, tmp_path):
    pkg = site.pkg
    path = str(tmp_path / "judges.json")
    table = judges.build(pkg, path=path)
    by = {r["code"]: r for r in table}
    assert list(by) == list(JUDGES)
    assert by["st_in"]["included"] and by["st_in"]["exclusion"] == []
    assert by["st_in"]["coverage"] == 1.0
    assert "coverage" in by["st_quay"]["exclusion"] and by["st_quay"]["coverage"] < 0.5
    t_end = pd.Timestamp(GAUGES["st_quay"]["end"])
    assert by["st_quay"]["n_with_gauge"] == int((pkg.t <= t_end + pd.Timedelta(minutes=6)).sum())
    assert by["st_out"]["exclusion"][0] == "in_frame" and not by["st_out"]["included"]
    # the dry share is the rule on the gauge at the record instants
    t, v = gauge_series(GAUGES["st_quay"], seed=100 + sorted(GAUGES).index("st_quay"))
    gq = judges.at_instants(pd.Series(v, index=t), pkg.t)
    fin = gq[np.isfinite(gq)]
    assert by["st_quay"]["dry_share"] == pytest.approx(np.mean(fin <= fin.min() + 0.01))
    doc = judges.read(path)
    assert doc["headline_locations"] == ["st_in"] and doc["n_record"] == pkg.T
    assert doc["inputs"] == judges.inputs_of(pkg, members=[])
    assert all("nbhd" not in r for r in doc["stations"])
    back = judges.load(path, pkg.grid)
    for a, b in zip(table, back):
        assert np.array_equal(a["nbhd"], b["nbhd"]) and a["code"] == b["code"]
    # a table that does not match the grid is refused
    doc["stations"][0]["R_g_m"] += 250.0
    product.write_json(path, doc)
    with pytest.raises(ValueError):
        judges.load(path, pkg.grid)
    doc["format"] = "v12-judges"
    product.write_json(path, doc)
    with pytest.raises(ValueError):
        judges.read(path)


def test_judges_path_is_the_package_copy(site):
    want = os.path.join(site.cfg.common_dir, "judges.json").replace("\\", "/")
    assert judges.judges_path(site.pkg) == want
    assert judges.frame_shape(site.pkg) == (H, W)
    assert judges.boundary_members(site.cfg) == []          # a fake boundary has none


# ═════════════════════════════════════════════════════════════════════════════
#  validation v2 (docs/validation_v2_spec.md §1.7, §6.1)
# ═════════════════════════════════════════════════════════════════════════════

def test_v2_coverage_rule_per_window(site, tmp_path):
    pkg = site.pkg
    table = judges.build(pkg, path=str(tmp_path / "judges.json"))
    by = {r["code"]: r for r in table}
    t = pd.DatetimeIndex(pkg.t)
    series = {c: judges.read_series(site.stations[c]) for c in ("st_in", "st_quay")}
    cov = judges.coverage_by_window([by["st_in"], by["st_quay"]], t,
                                    ("W3", "Y2023", "Y2024", "Y2025"), series)
    for wid in ("W3", "Y2023", "Y2024", "Y2025"):
        e = cov["st_in"][wid]
        assert e["included"] and e["coverage"] == 1.0 and e["window"] == wid
        n_w = int(((t >= f"{wid[1:]}-01-01") & (t < f"{int(wid[1:]) + 1}-01-01")).sum()) \
            if wid.startswith("Y") else len(t)
        assert e["n_record"] == n_w
    q = cov["st_quay"]
    t_end = pd.Timestamp(GAUGES["st_quay"]["end"]) + pd.Timedelta(minutes=6)
    t24 = t[(t >= "2024-01-01") & (t < "2025-01-01")]
    assert q["Y2024"]["n_with_gauge"] == int((t24 <= t_end).sum())
    assert "coverage" in q["Y2024"]["exclusion"] and "coverage" in q["Y2025"]["exclusion"]
    assert q["Y2023"]["coverage"] == 1.0 and "coverage" not in q["Y2023"]["exclusion"]
    assert q["W3"]["coverage"] == pytest.approx(by["st_quay"]["coverage"])
    # the window-independent rules come from the table record
    rec = dict(by["st_in"], nearest_member_km=2.5, nearest_member="m1")
    w = judges.window_inclusion(rec, series["st_in"], t, "W3")
    assert w["exclusion"] == ["boundary_member"] and w["nearest_member"] == "m1"
    rec = dict(by["st_in"], in_frame=False)
    assert judges.window_inclusion(rec, series["st_in"], t, "W3")["exclusion"] == ["in_frame"]
    # explicit members are evaluated instead
    st = site.stations["st_in"]
    near = {"code": "near", "lat": st["lat"] + 1.0 / 111.195, "lon": st["lon"]}
    w = judges.window_inclusion(st, series["st_in"], t, "W3", members=[near])
    assert w["exclusion"] == ["boundary_member"] and w["nearest_member"] == "near"


def test_v2_judge_sets_and_the_development_list():
    assert judges.development_list() == ["harlingen.waddenzee"]
    assert judges.development_list(["delfzijl"]) == ["delfzijl", "harlingen.waddenzee"]
    assert {c for c, _ in judges.DEVELOPMENT_CANDIDATES} == {"terneuzen", "delfzijl", "fer2"}
    table = [
        {"code": "harlingen.waddenzee", "location": "harlingen.waddenzee", "included": True,
         "mouth": False},
        {"code": "terschelling.west", "location": "terschelling.west", "included": True,
         "mouth": False},
        {"code": "emshoern_9340010", "location": "emshoern_9340010", "included": True,
         "mouth": True},
        {"code": "terneuzen", "location": "terneuzen", "included": False, "mouth": False},
        {"code": "terneuzen.westsluis.buiten", "location": "terneuzen", "included": True,
         "mouth": False},
    ]
    s = judges.judge_sets(table)
    assert s["headline"] == ["emshoern_9340010", "harlingen.waddenzee", "terneuzen",
                             "terschelling.west"]
    assert s["without_mouth"] == ["harlingen.waddenzee", "terneuzen", "terschelling.west"]
    assert s["development_free"] == ["emshoern_9340010", "terneuzen", "terschelling.west"]
    s = judges.judge_sets(table, development=["terneuzen"])
    assert "terneuzen" not in s["development_free"]
    wi = {"terschelling.west": {"Y2024": {"included": False}},
          "harlingen.waddenzee": {"Y2024": {"included": True}}}
    s = judges.judge_sets(table, window_incl=wi, wid="Y2024")
    assert s["headline"] == ["harlingen.waddenzee"]
    # the headline set of §6.1: 13 locations; a table is compared, never overridden
    assert sum(len(v) for v in judges.HEADLINE.values()) == 13
    assert set(judges.HEADLINE) == set(config.SITES)
    for site_, codes in judges.HEADLINE.items():
        assert set(codes) <= set(config.FROZEN["judges"]["stations"][site_])
        assert all(judges.location_of(c) == c for c in codes)
    wad = judges.check_headline("wadden", table[:2])
    assert wad["equal"] and wad["missing"] == []
    esc = judges.check_headline("escalda", table[3:])
    assert esc["missing"] == ["gatvanborssele"] and not esc["equal"]


def test_v2_timestamp_conventions(tmp_path):
    conv = judges.TIMESTAMP_CONVENTIONS
    assert set(conv) == {"RWS", "WSV", "IOC", "PdE"}
    assert conv["RWS"]["status"] == "documented" and conv["RWS"]["averaging_min"] == 10.0
    assert conv["PdE"]["averaging_min"] == 5.0
    rws = {"code": "x", "provider": "RWS"}
    assert judges.stamp_shift_min(rws) == 0.0 and judges.half_window_min(rws) == 5.0
    custom = {"RWS": dict(conv["RWS"], stamp="start"), "X": dict(conv["RWS"], stamp="end")}
    assert judges.stamp_shift_min(rws, custom) == 5.0
    assert judges.stamp_shift_min({"code": "y", "provider": "X"}, custom) == -5.0
    with pytest.raises(KeyError):
        judges.stamp_shift_min({"code": "z", "provider": "nobody"})
    with pytest.raises(ValueError):
        judges.stamp_shift_min(rws, {"RWS": dict(conv["RWS"], stamp="middle")})
    idx = pd.date_range("2024-01-01", periods=4, freq="10min")
    s = pd.Series([0.0, 1.0, 2.0, 3.0], index=idx)
    c = judges.centred(s, rws, custom)
    assert (c.index == idx + pd.Timedelta(minutes=5)).all() and c.tolist() == s.tolist()
    assert (judges.centred(s, rws, shift_min=-5.0).index == idx - pd.Timedelta(minutes=5)).all()
    assert judges.shift_series(s, 0) is s
    assert s.index.equals(idx)                                  # the input is untouched
    # conventions written by fetch judges are read back; the rest stays the built-in table
    p = tmp_path / "conv.json"
    product.write_json(str(p), {"providers": {"WSV": dict(conv["WSV"], status="documented")}})
    back = judges.read_conventions(str(p))
    assert back["WSV"]["status"] == "documented" and back["RWS"] == dict(conv["RWS"])
    assert judges.read_conventions(str(tmp_path / "absent.json")) == judges.conventions_doc()
    assert judges.conventions_sha() != judges.conventions_sha(back)


def test_v2_fetch_judges_plan_and_conventions(site, tmp_path, monkeypatch):
    from experiments.benchmark.fetch import judges as fj

    rec = {"code": "knock_3990010", "provider": "WSV", "provider_code": "knock_3990010"}
    assert fj.script_argument(rec) == "3990010"
    assert fj.script_argument({"code": "fer4_3212", "provider": "PdE"}) == "fer4"
    assert fj.script_argument({"code": "delfzijl", "provider": "RWS"}) == "delfzijl"
    with pytest.raises(ValueError):
        fj.script_argument({"code": "x_y", "provider": "WSV"})
    argv = fj.command({"code": "delfzijl", "provider": "RWS"}, python="py")
    assert argv[0] == "py" and argv[-2].endswith("scripts/rws_download.py")
    assert argv[-1] == "delfzijl"
    s = judges.read_series(site.stations["st_quay"])
    cov = fj.coverage_by_year(s, 10, years=(2022, 2023, 2024))
    assert cov[2022] == 0.0 and cov[2023] == pytest.approx(1.0, abs=1e-4)
    assert 0.1 < cov[2024] < 0.15                           # ends 2024-02-15
    # the synthetic stations' provider has no convention or script
    stas = judges.stations(SITE)
    doc = fj.conventions_document(stas)
    assert all(v["status"] == "no convention (refused)" for v in doc["stations"].values())
    doc = fj.conventions_document([{"code": "fer2", "provider": "IOC", "sampling_min": 1.0}])
    assert doc["stations"]["fer2"]["half_window_min"] == 0.5
    monkeypatch.setitem(fj.SCRIPTS, "synthetic", "fake_download.py")
    monkeypatch.setitem(fj.SCRIPT_YEARS, "synthetic", (2023, 2025))
    p = fj.plan([SITE], years=(2022, 2023, 2024, 2025))
    e = p["stations"]["st_quay"]
    assert e["missing_years"] == [2022, 2024, 2025] and not e["script_covers_missing"]
    assert p["stations"]["st_in"]["missing_years"] == [2022]
    paths = fj.run([SITE], out_dir=str(tmp_path))
    assert product.read_json(paths["plan"])["ran"] == []
    assert product.read_json(paths["conventions"])["format"] == fj.CONVENTIONS_FORMAT


def test_v2_window_inclusion_reads_the_series_centred_like_the_lag_truths():
    """The per-window inclusion centres the gauge with its provider's
    time-stamp convention first, exactly as the lag truths read it (review
    of 2026-10-14); an unknown provider is refused, never taken as centred."""
    idx = pd.date_range("2024-05-01", periods=24 * 6, freq="10min")
    series = pd.Series(np.sin(np.arange(len(idx)) / 7.0), index=idx)
    rec = {"code": "x", "provider": "P", "in_frame": True}
    conv = {"P": {"averaging_min": 10.0, "stamp": "end", "status": "test", "source": "t"}}
    t = pd.DatetimeIndex(["2024-05-01 03:07", "2024-05-01 09:41", "2024-05-01 17:55"])
    a = judges.window_inclusion(rec, series, t, "W3", conventions=conv)
    b = judges.window_inclusion(rec, judges.centred(series, rec, conv), t, "W3")
    assert a == b
    with pytest.raises(KeyError, match="time-stamp convention"):
        judges.window_inclusion(rec, series, t, "W3", conventions="default")
    rws = dict(rec, provider="RWS")
    assert judges.window_inclusion(rws, series, t, "W3", conventions="default") == \
        judges.window_inclusion(rws, series, t, "W3")              # RWS F007 is centred
