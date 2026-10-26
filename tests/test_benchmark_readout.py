"""Tests of experiments.benchmark.readout, the lag read-outs of spec §6.3.

On a tiny UTM frame (30 x 50 pixels of 100 m) with a synthetic instance
domain (s = distance from the west edge, 1-km bands) the tests pin:

* the ladder: step 1 (median over nbhd ∩ keep with >= 30 defined pixels,
  exactly at the threshold), step 2 (the instance's modal band from
  judges.station_geometry, flagged ``step2``), step 3 (0, flagged
  ``fallback0``), with tau_band, tau_step1 and the read-out pixel count;
* the edge share (band edges for band methods; the scan bounds -45 / 180
  for linear pixel methods) and the ``edge`` flag above 0.5;
* phase fields: wrapping about the truth-free 67.5 min and the circular
  median across the period boundary;
* band_interp: linear between band centres, clamped at the end bands, and
  unwrapped for phase medians;
* point and group read-outs, and the interval at the gauge (band and pixel
  levels).

Run:  python -X utf8 -m pytest tests/test_benchmark_readout.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import judges, readout as rd  # noqa: E402
from experiments.benchmark.lags import common as lc  # noqa: E402

CRS = "EPSG:32631"
X0, Y0, PIX = 500_000.0, 5_700_000.0, 100.0
H, W = 30, 50
P = lc.PERIOD_M2_MIN


def _grid():
    return {"xs": X0 + PIX * np.arange(W), "ys": Y0 - PIX * np.arange(H), "crs": CRS}


def _domain():
    rr, cc = np.meshgrid(np.arange(3, 27), np.arange(2, 48), indexing="ij")
    keep = np.sort((rr * W + cc).ravel()).astype(np.int64)
    col = keep % W
    s_km = (col - 2) * 0.1 + 0.05
    band_of = np.floor(s_km / 1.0).astype(np.int64)
    K = int(band_of.max()) + 1
    centers = np.array([np.median(s_km[band_of == k]) for k in range(K)])
    return keep, s_km, band_of, centers


def _judge(row, col, R=400.0, code="j"):
    from pyproj import Transformer

    g = _grid()
    gx, gy = float(g["xs"][col]), float(g["ys"][row])
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(gx, gy)
    return {"code": code, "lon": lon, "lat": lat, "gx": gx, "gy": gy,
            "nbhd": judges.nbhd_flat(g, gx, gy, R), "s_g_km": (col - 2) * 0.1 + 0.05}


@pytest.fixture(scope="module")
def dom():
    keep, s_km, band_of, centers = _domain()
    return {"keep": keep, "s_km": s_km, "band_of": band_of, "centers": centers,
            "grid": _grid(), "judge": _judge(14, 22)}


def _ro(dom, field, **kw):
    return rd.readout(field, dom["keep"], dom["band_of"], dom["s_km"], dom["centers"],
                      dom["judge"], dom["grid"], **kw)


def test_nbhd_positions_intersect_with_keep(dom):
    keep = dom["keep"]
    nb = np.array([keep[0] - 1, keep[3], keep[10], keep[-1], keep[-1] + 5], np.int64)
    pos = rd.nbhd_positions(nb, keep)
    assert pos.tolist() == [3, 10, len(keep) - 1]
    assert rd.nbhd_positions(np.array([], np.int64), keep).size == 0


def test_step1_median_of_the_neighbourhood(dom):
    f = np.full(len(dom["keep"]), 30.0)
    nbpos = rd.nbhd_positions(dom["judge"]["nbhd"], dom["keep"])
    assert len(nbpos) >= 40
    f[nbpos] = np.linspace(10, 20, len(nbpos))
    r = _ro(dom, f)
    assert r["step"] == 1 and r["n_px"] == len(nbpos)
    assert r["tau"] == pytest.approx(float(np.median(f[nbpos])))
    assert r["tau_step1"] == r["tau"]
    geom = rd.judge_geometry(dom["judge"], dom["grid"], dom["keep"], dom["s_km"],
                             dom["band_of"], dom["centers"])
    assert r["band"] == geom["k_g"] == 2
    band = f[dom["band_of"] == geom["k_g"]]
    assert r["tau_band"] == pytest.approx(float(np.median(band)))
    assert "fallback0" not in r["flags"] and "step2" not in r["flags"]


def test_threshold_of_step1_is_30_defined_pixels(dom):
    nbpos = rd.nbhd_positions(dom["judge"]["nbhd"], dom["keep"])
    f = np.full(len(dom["keep"]), 50.0)
    f[nbpos] = np.nan
    f[nbpos[:30]] = 10.0
    r = _ro(dom, f)
    assert r["step"] == 1 and r["tau"] == 10.0 and r["n_px"] == 30
    f[nbpos[29]] = np.nan
    r = _ro(dom, f)
    assert r["step"] == 2 and "step2" in r["flags"] and np.isnan(r["tau_step1"])
    band = (dom["band_of"] == r["band"]) & np.isfinite(f)
    assert r["tau"] == pytest.approx(float(np.median(f[band]))) == r["tau_band"]
    assert r["n_px"] == int(band.sum())
    # a smaller min_px makes the same field a step-1 read-out
    assert _ro(dom, f, min_px=29)["step"] == 1


def test_step3_zero_fallback(dom):
    f = np.full(len(dom["keep"]), 40.0)
    geom = rd.judge_geometry(dom["judge"], dom["grid"], dom["keep"], dom["s_km"],
                             dom["band_of"], dom["centers"])
    nbpos = rd.nbhd_positions(dom["judge"]["nbhd"], dom["keep"])
    f[nbpos] = np.nan
    f[dom["band_of"] == geom["k_g"]] = np.nan
    r = _ro(dom, f, geom=geom)
    assert r["step"] == 3 and r["tau"] == 0.0 and "fallback0" in r["flags"]
    assert r["n_px"] == 0 and np.isnan(r["tau_band"]) and np.isnan(r["tau_step1"])
    assert np.isnan(r["edge_share"])
    # the interpolated band read-out still comes from the other bands
    assert np.isfinite(r["tau_band_interp"])


def test_edge_share_band_and_pixel_rules(dom):
    K = len(dom["centers"])
    tau_b = np.arange(K) * 10.0
    f = lc.broadcast_band(tau_b, dom["band_of"])
    edge = np.zeros(K, bool)
    r = _ro(dom, f, band_edge=edge)
    assert r["edge_share"] == 0.0 and "edge" not in r["flags"]
    edge[2] = True                           # the gauge's band is on the scan edge
    r = _ro(dom, f, band_edge=edge)
    nbpos = rd.nbhd_positions(dom["judge"]["nbhd"], dom["keep"])
    share = float(np.mean(dom["band_of"][nbpos] == 2))
    assert 0.5 < share < 1.0                 # the neighbourhood straddles bands 1 and 2
    assert r["edge_share"] == pytest.approx(share) and "edge" in r["flags"]
    edge[1] = True
    assert _ro(dom, f, band_edge=edge)["edge_share"] == 1.0
    edge[:] = False
    edge[1] = True
    r = _ro(dom, f, band_edge=edge)
    assert r["edge_share"] == pytest.approx(1 - share) and "edge" not in r["flags"]
    # linear pixel method: values on the scan bounds
    g = np.full(len(dom["keep"]), 30.0)
    nbpos = rd.nbhd_positions(dom["judge"]["nbhd"], dom["keep"])
    g[nbpos[: len(nbpos) * 3 // 4]] = lc.SCAN_HI
    r = _ro(dom, g)
    assert r["edge_share"] == pytest.approx(len(nbpos[: len(nbpos) * 3 // 4]) / len(nbpos))
    assert "edge" in r["flags"]
    # phase fields have no scan bounds
    assert np.isnan(_ro(dom, g, lag_kind="phase")["edge_share"])


def test_wrap_about_the_truth_free_centre():
    x = np.array([-400.0, -305.0, 0.0, 67.5, 440.0, 441.0, 1000.0])
    w = lc.wrap_phase(x)
    assert (w >= 67.5 - P / 2).all() and (w < 67.5 + P / 2).all()
    assert np.allclose(lc.circ_diff(w, x), 0.0)
    assert w[3] == 67.5 and w[2] == pytest.approx(0.0) and w[4] == pytest.approx(440.0)
    assert w[5] == pytest.approx(441.0 - P)


def test_circular_median_across_the_period_boundary(dom):
    nbpos = rd.nbhd_positions(dom["judge"]["nbhd"], dom["keep"])
    f = np.full(len(dom["keep"]), 20.0)
    half = len(nbpos) // 2
    f[nbpos[:half]] = P - 4.0            # = -4 min on the circle
    f[nbpos[half:]] = 4.0 + 2 * P        # = +4 min
    f[nbpos[-1]] = 4.0                   # odd count: the median is +4 or -4
    r = _ro(dom, f, lag_kind="phase")
    assert r["step"] == 1 and abs(r["tau"]) == pytest.approx(4.0)
    assert np.median(f[nbpos]) > 300           # a linear median would be wrong
    # the band median is circular too, and the value is wrapped about 67.5
    g = np.full(len(dom["keep"]), 100.0 + P)
    r = _ro(dom, g, lag_kind="phase")
    assert r["tau"] == pytest.approx(100.0) and r["tau_band"] == pytest.approx(100.0)
    assert rd.median_of([P - 5.0, 5.0, 2 * P], "phase") == pytest.approx(0.0, abs=1e-9)


def test_band_interp_is_linear_and_clamped():
    c = np.array([1.5, 4.5, 7.5])
    v = np.array([10.0, 40.0, 70.0])
    assert rd.band_interp(v, c, 3.0) == pytest.approx(25.0)
    assert rd.band_interp(v, c, 0.2) == 10.0            # clamped at the first band
    assert rd.band_interp(v, c, 30.0) == 70.0           # and at the last
    assert rd.band_interp(np.array([10.0, np.nan, 70.0]), c, 4.5) == pytest.approx(40.0)
    assert np.isnan(rd.band_interp(v, c, np.nan))
    assert np.isnan(rd.band_interp(np.full(3, np.nan), c, 2.0))
    # phase medians are unwrapped along s before interpolating
    vp = np.array([350.0, 350.0 + 40.0 - P, 350.0 + 80.0 - P])
    out = rd.band_interp(vp, c, 3.0, "phase")
    assert lc.circ_diff(out, 370.0) == pytest.approx(0.0, abs=1e-9)


def test_band_medians_and_interp_from_the_field(dom):
    K = len(dom["centers"])
    tau_b = 10.0 * np.arange(K)
    f = lc.broadcast_band(tau_b, dom["band_of"]).astype(float)
    m = rd.band_medians(f, dom["band_of"], K)
    assert np.allclose(m, tau_b)
    r = _ro(dom, f)
    s_g = r["s_g_km"]
    assert r["tau_band_interp"] == pytest.approx(np.interp(s_g, dom["centers"], tau_b))
    assert np.isfinite(s_g) and abs(s_g - dom["judge"]["s_g_km"]) < 0.3


def test_point_readout():
    point = {"g1": {"a": 12.0, "ab": None, "b": float("nan"), "hwlw": "x"},
             "g2": {"a": 3}}
    assert rd.point_readout(point, "g1", "a") == 12.0
    assert np.isnan(rd.point_readout(point, "g1", "ab"))
    assert np.isnan(rd.point_readout(point, "g1", "b"))
    assert np.isnan(rd.point_readout(point, "g1", "hwlw"))
    assert rd.point_readout(point, "g2", "a") == 3.0
    assert np.isnan(rd.point_readout(point, "missing", "a"))
    assert np.isnan(rd.point_readout(None, "g1", "a"))


def test_group_readouts(dom):
    K = len(dom["centers"])
    groups = {"orbit037": {"tau": 5.0 * np.arange(K), "n_scenes": 80, "status": "ok"},
              "orbit137": {"tau": np.full(K, 33.0), "n_scenes": 70, "status": "ok"},
              "orbit080": {"tau": np.full(K, np.nan), "n_scenes": 9,
                           "status": "not_computable", "reason": "too few scenes"}}
    out = rd.group_readouts(groups, dom["keep"], dom["band_of"], dom["s_km"], dom["centers"],
                            dom["judge"], dom["grid"])
    assert set(out) == set(groups)
    assert out["orbit037"]["tau"] == 10.0 and out["orbit037"]["step"] == 1
    assert out["orbit137"]["tau"] == 33.0 and out["orbit137"]["n_scenes"] == 70
    assert out["orbit080"]["status"] == "not_computable" and np.isnan(out["orbit080"]["tau"])


def test_interval_readout_band_and_pixel_levels(dom):
    K = len(dom["centers"])
    f = lc.broadcast_band(10.0 * np.arange(K), dom["band_of"]).astype(float)
    r = _ro(dom, f)
    iv = {"level": "band", "lo": 10.0 * np.arange(K) - 3, "hi": 10.0 * np.arange(K) + 4,
          "nominal": "bootstrap30"}
    assert rd.interval_readout(iv, r) == (17.0, 24.0)
    n = len(dom["keep"])
    ivp = {"level": "pixel", "lo": f - 6.0, "hi": f + 6.0, "nominal": "iqr20x1.96/0.674"}
    lo, hi = rd.interval_readout(ivp, r)
    assert lo == pytest.approx(r["tau"] - 6.0) and hi == pytest.approx(r["tau"] + 6.0)
    assert all(np.isnan(rd.interval_readout(None, r)))
    # a step-3 read-out has no interval at the gauge
    r3 = dict(r, step=3)
    assert all(np.isnan(rd.interval_readout(iv, r3)))
    # phase pixel intervals: circular centre, median half-width
    ph = np.full(n, P - 2.0)
    ivq = {"level": "pixel", "lo": ph - 5.0, "hi": ph + 5.0, "nominal": "iqr20x1.96/0.674"}
    rp = _ro(dom, ph, lag_kind="phase")
    lo, hi = rd.interval_readout(ivq, rp, "phase")
    assert lo == pytest.approx(-7.0) and hi == pytest.approx(3.0)


def test_field_must_be_aligned_with_keep(dom):
    with pytest.raises(ValueError):
        _ro(dom, np.zeros(len(dom["keep"]) - 1))
    with pytest.raises(ValueError):
        _ro(dom, np.zeros(len(dom["keep"])), lag_kind="circular")
