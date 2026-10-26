# -*- coding: utf-8 -*-
"""Competitors of the fixed-configuration tables (experiments/fixedcfg):
E4 LogiTide2DEM, E5a Flood2Topo, E5b Tseng, E5c sampled exceedance, E7 Coast
X-Ray, E12 wet/dry bracket. Tiny synthetic worlds with planted elevations;
every method recovers a planted ramp within its expected precision, and the
published steps are checked one by one."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.fixedcfg import elev_base as base  # noqa: E402
from experiments.fixedcfg import elev_bracket, elev_coastxray, elev_freq, elev_logitide  # noqa: E402
from experiments.fixedcfg import methods as M  # noqa: E402
from experiments.fixedcfg.level_series import (LevelSeries, harmonic_constants,  # noqa: E402
                                               rayleigh_select, window_period)

M2_H, S2_H, K1_H, O1_H = 12.4206012, 12.0, 23.9344697, 25.8193417
#: the competitors of this builder (other modules of the package may add more)
MINE = ["E4", "E5a", "E5b", "E5c", "E7", "E12"]
MINE_VARIANTS = ["E4_firth", "E4_cap25", "E7_t0", "E7_corrected"]


def mine(variants=True):
    ids = MINE + (MINE_VARIANTS if variants else [])
    return [(mid,) + tuple(M.get(mid)) for mid in ids]


# ─────────────────────────────────────────────────────────────────────────────
#  synthetic worlds
# ─────────────────────────────────────────────────────────────────────────────

def tide_series(days=60, start="2024-03-01", z0=0.02, a_m2=1.2, a_s2=0.45, a_k1=0.0, a_o1=0.0):
    from pyintertidal.boundary import hours_since_epoch

    t = pd.date_range(start, periods=int(days * 1440), freq="1min")
    th = hours_since_epoch(t)
    h = (z0 + a_m2 * np.cos(2 * np.pi / M2_H * th + 0.3) + a_s2 * np.cos(2 * np.pi / S2_H * th + 1.1)
         + a_k1 * np.cos(2 * np.pi / K1_H * th + 0.7) + a_o1 * np.cos(2 * np.pi / O1_H * th + 2.0))
    return LevelSeries(h, t, source="synthetic")


def make_obs(h, t, z_true, rng, clear=0.85, nan_share=0.01, noise=0.05, p_wet=None):
    """Step world: wet iff h > z (or Bernoulli(p_wet)); NDWI +-0.3 with noise
    that never flips the sign; a cloud mask; a few undefined NDWI."""
    T, P = len(h), len(z_true)
    wet = (h[:, None] > z_true[None, :]) if p_wet is None else (rng.random((T, P)) < p_wet)
    Y = (np.where(wet, 0.3, -0.3) + rng.uniform(-noise, noise, (T, P))).astype(np.float32)
    C = rng.random((T, P)) < clear
    Y[C & (rng.random((T, P)) < nan_share)] = np.nan
    W = C & (Y > 0)
    return types.SimpleNamespace(Y=Y, C=C, W=W, h=np.asarray(h, float), t=pd.DatetimeIndex(t),
                                 keep=np.arange(P, dtype=np.int64), shape=(1, P), pixel_m=10.0,
                                 site="synthetic")


def regular_scenes(series, n, stride=577):
    """n scenes at a regular stride (minutes) through the series: they
    sample the level distribution of the series almost uniformly."""
    idx = (np.arange(n) * int(stride)) % len(series.h)
    idx = np.sort(np.unique(idx))
    return series.t[idx], series.h[idx]


def random_scenes(series, n, rng):
    idx = np.sort(rng.choice(len(series.h), n, replace=False))
    return series.t[idx], series.h[idx]


@pytest.fixture(scope="module")
def series():
    return tide_series()


@pytest.fixture(scope="module")
def ramp():
    return np.linspace(-1.2, 1.2, 61)


# ─────────────────────────────────────────────────────────────────────────────
#  registry and interface
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_ids_cards_and_order():
    assert list(M.METHODS)[:len(MINE)] == MINE                 # built-ins first, catalogue order
    assert set(MINE_VARIANTS) <= set(M.VARIANTS)
    assert not set(M.REGISTRY_ERRORS) & set(M.BUILTIN)
    for mid, label, run, card in mine():
        assert callable(run) and label and card["label"] == label
        assert base.check_card(card)
        assert card["deviations"], mid
        assert card["needs_full_frame"] is False
        assert card["fidelity"][:2] in ("F0", "F2", "F3")
    assert [m for m in MINE if M.needs_level_series(m)] == ["E5a", "E5b", "E7"]
    assert M.get("E4_firth")[0].startswith("LogiTide2DEM")
    with pytest.raises(KeyError):
        M.get("E99")


def test_registry_screens_modules_by_source(tmp_path):
    yes = tmp_path / "a.py"
    yes.write_text("import os\nMETHODS = {}\n", encoding="utf-8")
    no = tmp_path / "b.py"
    no.write_text("def f():\n    METHODS = {}\n    return METHODS\n", encoding="utf-8")
    bad = tmp_path / "c.py"
    bad.write_text("def (:\n", encoding="utf-8")
    assert M._defines_methods(str(yes)) and not M._defines_methods(str(no))
    assert not M._defines_methods(str(bad))


def test_check_card_rejects_incomplete_deviation():
    card = dict(M.METHODS["E12"][2])
    card["deviations"] = [{"paper": "x", "here": "y", "why": "", "sensitivity": "z"}]
    with pytest.raises(ValueError):
        base.check_card(card)


def test_every_method_honours_the_interface(series, ramp):
    rng = np.random.default_rng(1)
    t, h = random_scenes(series, 120, rng)
    obs = make_obs(h, t, ramp, rng)
    for mid, _, run, _ in mine():
        out = run(obs, {"level_series": series})
        assert set(out) == {"z", "sigma", "meta"}, mid
        z = out["z"]
        assert z.dtype == np.float32 and z.shape == (len(ramp),), mid
        assert out["sigma"] is None
        m = out["meta"]
        assert m["status"] == base.STATUS_OK, (mid, m.get("reason"))
        assert m["method"] == mid and m["n_scenes"] == len(h) and m["n_px"] == len(ramp)
        assert m["n_finite"] == int(np.isfinite(z).sum()) > 0
        for k, v in m.get("extra", {}).items():
            assert np.asarray(v).shape == (len(ramp),), (mid, k)


def test_blocking_does_not_change_values(series, ramp):
    rng = np.random.default_rng(2)
    t, h = random_scenes(series, 90, rng)
    obs = make_obs(h, t, ramp, rng)
    for mid, _, run, _ in mine(variants=False):
        a = run(obs, {"level_series": series})["z"]
        b = run(obs, {"level_series": series, "block_px": 7})["z"]
        assert np.array_equal(a, b, equal_nan=True), mid


def test_inputs_are_checked(series, ramp):
    rng = np.random.default_rng(3)
    t, h = random_scenes(series, 40, rng)
    obs = make_obs(h, t, ramp, rng)
    run = M.METHODS["E12"][1]
    bad = types.SimpleNamespace(**vars(obs))
    bad.W = obs.W.copy()
    bad.W[0, 0] = ~bad.W[0, 0]
    with pytest.raises(ValueError, match="C & \\(Y > 0\\)"):
        run(bad, None)
    bad = types.SimpleNamespace(**vars(obs))
    bad.h = obs.h.copy()
    bad.h[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run(bad, None)
    with pytest.raises(ValueError, match="unknown option"):
        M.METHODS["E7"][1](obs, {"level_series": series, "E7": {"treshold": 0.1}})


def test_series_is_required_and_must_span_the_scenes(series, ramp):
    rng = np.random.default_rng(4)
    t, h = random_scenes(series, 60, rng)
    obs = make_obs(h, t, ramp, rng)
    for mid in ("E5a", "E5b", "E7"):
        out = M.METHODS[mid][1](obs, None)
        assert out["meta"]["status"] == base.STATUS_NOT_COMPUTABLE
        assert "level_series" in out["meta"]["reason"]
        assert np.isnan(out["z"]).all()
    short = series.slice(series.t[0], series.t[len(series.t) // 2])
    with pytest.raises(ValueError, match="does not span"):
        M.METHODS["E5a"][1](obs, {"level_series": short})
    # a bare array of levels is enough for E5a / E5b, not for E7 (needs times)
    assert M.METHODS["E5a"][1](obs, {"level_series": series.h})["meta"]["status"] == "ok"
    out = M.METHODS["E7"][1](obs, {"level_series": series.h})
    assert out["meta"]["status"] == base.STATUS_NOT_COMPUTABLE and "times" in out["meta"]["reason"]


# ─────────────────────────────────────────────────────────────────────────────
#  E12 bracket: identity with the legacy UDF source
# ─────────────────────────────────────────────────────────────────────────────

def _udf_namespace():
    """BATHYMETRY_UDF_NDWI executed from its source with a stub openeo.udf."""
    from pyintertidal.legacy import udf

    class XarrayDataCube:
        def __init__(self, array):
            self.array = array

        def get_array(self):
            return self.array

    stub_pkg, stub_udf = types.ModuleType("openeo"), types.ModuleType("openeo.udf")
    stub_udf.XarrayDataCube = XarrayDataCube
    stub_pkg.udf = stub_udf
    saved = {k: sys.modules.get(k) for k in ("openeo", "openeo.udf")}
    sys.modules["openeo"], sys.modules["openeo.udf"] = stub_pkg, stub_udf
    try:
        ns = {}
        exec(compile(udf.BATHYMETRY_UDF_NDWI, "BATHYMETRY_UDF_NDWI", "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns, XarrayDataCube


def test_bracket_equals_the_udf_bitwise():
    xr = pytest.importorskip("xarray")
    ns, Cube = _udf_namespace()
    rng = np.random.default_rng(5)
    T, H, Wd = 30, 4, 6
    dates = pd.date_range("2024-01-01", periods=T, freq="3D")
    h = 1.5 * np.sin(np.arange(T) * 2.1) + rng.normal(0, 0.05, T)
    zt = np.linspace(-2.5, 2.5, H * Wd).reshape(H, Wd)          # includes never-wet / never-dry
    wet = h[:, None, None] > zt[None]
    near = np.abs(h[:, None, None] - zt[None]) < 0.4
    flip = near & (rng.random((T, H, Wd)) < 0.3)                  # inconsistent pixels
    wet = wet ^ flip
    b08 = rng.uniform(300, 1500, (T, H, Wd)).astype(np.float32)
    b03 = np.where(wet, b08 * rng.uniform(1.2, 2.0, wet.shape), b08 * rng.uniform(0.4, 0.9, wet.shape))
    b03 = b03.astype(np.float32)
    zero = rng.random((T, H, Wd)) < 0.02                         # undefined NDWI (B03 + B08 = 0)
    b03[zero], b08[zero] = 0.0, 0.0
    scl = rng.choice(np.array([4, 5, 6, 7, 3, 8, 9, 10, 12]), size=(T, H, Wd),
                     p=[.3, .2, .2, .1, .05, .05, .05, .03, .02]).astype(np.float32)
    arr = xr.DataArray(np.stack([b03, b08, scl], axis=1), dims=("t", "bands", "y", "x"),
                       coords={"t": dates.values, "bands": ["B03", "B08", "SCL"]})
    ctx = {"tide_heights": {str(d.date()): float(v) for d, v in zip(dates, h)}, "valid_dates": None,
           "clear_classes": [4, 5, 6, 7], "ndwi_threshold": 0.0, "method": "pixels"}
    ref = ns["apply_datacube"](Cube(arr), ctx).get_array().sel(bands="Elevation").values
    # the common inputs, built as marea.extract builds them
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(b03 + b08 != 0, (b03 - b08) / (b03 + b08), np.nan).astype(np.float32)
    C = np.isin(scl.astype(np.int16), [4, 5, 6, 7])
    Y2, C2 = Y.reshape(T, -1), C.reshape(T, -1)
    obs = types.SimpleNamespace(Y=Y2, C=C2, W=C2 & (Y2 > 0), h=h, t=dates, keep=np.arange(H * Wd),
                                shape=(H, Wd), pixel_m=10.0, site="synthetic")
    out = elev_bracket.run(obs, None)
    z = out["z"].reshape(H, Wd)
    assert z.dtype == ref.dtype == np.float32
    assert np.array_equal(z, ref, equal_nan=True)
    c = out["meta"]["counts"]
    assert c["never_dry"] > 0 and c["never_wet"] > 0 and c["inconsistent_kept"] > 0


def test_bracket_recovers_the_ramp(series, ramp):
    rng = np.random.default_rng(6)
    t, h = random_scenes(series, 200, rng)
    out = elev_bracket.run(make_obs(h, t, ramp, rng), None)
    z, ex = out["z"].astype(float), out["meta"]["extra"]
    lo, hi = ex["zmin"].astype(float), ex["zmax"].astype(float)
    assert np.all(np.isfinite(z))
    assert np.all((lo <= ramp + 1e-6) & (ramp <= hi + 1e-6))          # the truth is bracketed
    assert np.all(np.abs(z - ramp) <= 0.5 * (hi - lo) + 1e-6)
    assert np.median(np.abs(z - ramp)) < 0.02


# ─────────────────────────────────────────────────────────────────────────────
#  E4 LogiTide2DEM
# ─────────────────────────────────────────────────────────────────────────────

def test_r_logit_link_clamps():
    eta = np.array([-40.0, -30.0, 0.0, 30.0, 40.0])
    mu = elev_logitide.linkinv(eta)
    eps = np.finfo(float).eps
    assert mu[0] == eps / (1 + eps) and mu[-1] == (1 / eps) / (1 + 1 / eps)
    assert mu[2] == 0.5 and mu[1] == np.exp(-30.0) / (1 + np.exp(-30.0))
    me = elev_logitide.mu_eta(eta)
    assert me[0] == eps and me[-1] == eps and me[2] == 0.25


def test_logitide_separated_pixels_take_the_gap_midpoint(series, ramp):
    rng = np.random.default_rng(7)
    t, h = random_scenes(series, 150, rng)
    obs = make_obs(h, t, ramp, rng)
    e4 = elev_logitide.run(obs, None)
    e12 = elev_bracket.run(obs, None)
    assert e4["meta"]["counts"]["separated"] == len(ramp)
    assert np.allclose(e4["z"], e12["z"], atol=1e-6, rtol=0)
    assert np.all(e4["meta"]["extra"]["msr"] == 1.0)
    assert np.median(np.abs(e4["z"] - ramp)) < 0.02


def _logistic_world(series, rng, n=150, s=0.08):
    t, h = random_scenes(series, n, rng)
    z = np.linspace(-1.0, 1.0, 41)
    p = 1.0 / (1.0 + np.exp(-(h[:, None] - z[None, :]) / s))
    return make_obs(h, t, z, rng, clear=0.9, nan_share=0.0, p_wet=p), z


def test_logitide_irls_equals_unpenalised_reference_fit(series):
    rng = np.random.default_rng(8)
    obs, z = _logistic_world(series, rng)
    out = elev_logitide.run(obs, {"E4": {"msr_min": 0.0}})
    case = out["meta"]["extra"]["case"]
    fitted = np.flatnonzero((case == 3) & np.isfinite(out["z"]))
    assert len(fitted) >= 20
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:            # independent fallback: BFGS on the 2 parameters
        LogisticRegression = None
    from scipy.optimize import minimize
    for p in fitted[:25]:
        v = obs.C[:, p] & np.isfinite(obs.Y[:, p])
        hv = obs.h[v]
        y = (~obs.W[v, p]).astype(float)                       # land = 1
        c = hv.mean()
        if LogisticRegression is not None:
            try:       # scikit-learn >= 1.8: no penalty is C = inf
                m = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-14, max_iter=1000)
                m.fit((hv - c)[:, None], y)
            except (TypeError, ValueError):
                m = LogisticRegression(penalty=None, solver="newton-cholesky", tol=1e-14, max_iter=1000)
                m.fit((hv - c)[:, None], y)
            b0, b1 = float(m.intercept_[0]), float(m.coef_[0, 0])
        else:
            def nll(b):
                e = b[0] + b[1] * (hv - c)
                return float(np.sum(np.logaddexp(0, e) - y * e))
            b0, b1 = minimize(nll, [0.0, -1.0], method="BFGS", options={"gtol": 1e-12}).x
        Z_ref = c - b0 / b1
        assert abs(out["z"][p] - Z_ref) <= 1e-6 + 1e-6 * abs(Z_ref), p     # float32 output


def test_logitide_recovers_a_planted_logistic_ramp(series):
    rng = np.random.default_rng(9)
    obs, z = _logistic_world(series, rng)
    out = elev_logitide.run(obs, None)
    fin = np.isfinite(out["z"])
    assert fin.mean() >= 0.9
    err = out["z"][fin] - z[fin]
    assert np.median(np.abs(err)) < 0.04 and np.max(np.abs(err)) < 0.2
    assert out["meta"]["counts"]["fitted"] > 0 and out["meta"]["counts"]["nonconverged"] == 0


def test_logitide_filters_and_the_27_image_rule(series, ramp):
    Z = np.array([0.0, 0.5, 2.0, -2.0, 0.1, np.nan])
    msr = np.array([0.9, 0.59, 0.9, 0.9, 0.6, 0.9])
    keep, drop = elev_logitide.apply_filters(Z, msr, np.full(6, -1.5), np.full(6, 1.5), 0.6)
    assert keep.tolist() == [True, False, False, False, True, False]
    assert drop["msr"].tolist() == [False, True, False, False, False, False]
    assert drop["range"].tolist() == [False, False, True, True, False, False]
    # a pixel whose class does not depend on the tide has MSR ~ 0.5 and is dropped
    rng = np.random.default_rng(10)
    t, h = random_scenes(series, 200, rng)
    p = np.full((200, 1), 0.5)
    obs = make_obs(h, t, np.zeros(1), rng, clear=1.0, nan_share=0.0, p_wet=p)
    out = elev_logitide.run(obs, None)
    assert np.isnan(out["z"][0]) and out["meta"]["extra"]["msr"][0] < 0.6
    # fewer than 27 images: not computable
    t, h = random_scenes(series, 26, rng)
    out = elev_logitide.run(make_obs(h, t, ramp, rng), None)
    assert out["meta"]["status"] == base.STATUS_NOT_COMPUTABLE and "27" in out["meta"]["reason"]
    assert np.isnan(out["z"]).all()


def test_logitide_firth_equals_brute_force_penalised_maximum(series):
    from scipy.optimize import minimize

    rng = np.random.default_rng(11)
    obs, z = _logistic_world(series, rng, n=80, s=0.05)
    out = elev_logitide.run_firth(obs, {"E4": {"msr_min": 0.0}})
    assert out["meta"]["method"] == "E4_firth"
    case = out["meta"]["extra"]["case"]
    picks = list(np.flatnonzero(case == 1)[:3]) + list(np.flatnonzero(case == 3)[:3])
    assert len(picks) >= 4
    for p in picks:
        v = obs.C[:, p] & np.isfinite(obs.Y[:, p])
        hv = obs.h[v]
        y = (~obs.W[v, p]).astype(float)
        c = hv.mean()
        x = hv - c

        def npll(b):
            e = b[0] + b[1] * x
            mu = 1.0 / (1.0 + np.exp(-e))
            w = mu * (1 - mu)
            det = w.sum() * (w * x * x).sum() - (w * x).sum() ** 2
            return -(np.sum(y * e - np.logaddexp(0, e)) + 0.5 * np.log(det))
        r = minimize(npll, [0.0, -5.0], method="Nelder-Mead",
                     options={"xatol": 1e-11, "fatol": 1e-15, "maxiter": 40000, "maxfev": 80000})
        Z_ref = c - r.x[0] / r.x[1]
        assert np.isfinite(out["z"][p])
        assert abs(out["z"][p] - Z_ref) < 1e-5, (p, out["z"][p], Z_ref)


def test_logitide_cap25_stays_inside_the_gap(series, ramp):
    rng = np.random.default_rng(12)
    t, h = random_scenes(series, 120, rng)
    obs = make_obs(h, t, ramp, rng)
    out = elev_logitide.run_cap25(obs, None)
    e12 = elev_bracket.run(obs, None)["meta"]["extra"]
    fin = np.isfinite(out["z"])
    assert fin.mean() > 0.9
    assert np.all(out["z"][fin] >= e12["zmin"][fin] - 1e-5) and np.all(out["z"][fin] <= e12["zmax"][fin] + 1e-5)


# ─────────────────────────────────────────────────────────────────────────────
#  E5 frequency methods
# ─────────────────────────────────────────────────────────────────────────────

def test_flood2topo_follows_interp1():
    levels = np.array([0.0, 3.0, 1.0, 2.0])            # sorted descending: 3, 2, 1, 0; Freq 25..100
    f = np.array([0.0, 0.1, 0.25, 0.375, 0.5, 1.0, np.nan])
    z = elev_freq.flood2topo(f, levels)
    assert np.isnan(z[0]) and np.isnan(z[1]) and np.isnan(z[-1])   # f = 0; 100 f < Freq_1; NaN
    assert z[2] == 3.0 and z[3] == 2.5 and z[4] == 2.0 and z[5] == 0.0


def test_flood2topo_is_the_lowest_wet_level_when_the_series_is_the_scenes(ramp):
    rng = np.random.default_rng(13)
    s = tide_series(days=20)
    t, h = random_scenes(s, 150, rng)
    obs = make_obs(h, t, ramp, rng, clear=1.0, nan_share=0.0)
    out = elev_freq.run_e5a(obs, {"level_series": LevelSeries(h, t)})
    wet = obs.W
    expect = np.array([h[wet[:, p]].min() if wet[:, p].any() else np.nan for p in range(len(ramp))])
    assert np.isfinite(expect).sum() > 40
    assert np.allclose(out["z"], expect, atol=1e-6, rtol=0, equal_nan=True)


def test_flood2topo_recovers_the_ramp_when_scenes_sample_the_series(series, ramp):
    rng = np.random.default_rng(14)
    t, h = regular_scenes(series, 300)
    out = elev_freq.run_e5a(make_obs(h, t, ramp, rng, clear=0.9), {"level_series": series})
    err = out["z"] - ramp
    assert np.all(np.isfinite(err))
    assert np.median(np.abs(err)) < 0.04 and np.max(np.abs(err)) < 0.15


def test_tseng_is_linear_and_exact_for_a_uniform_level_distribution(ramp):
    # a triangle-wave tide has a uniform level distribution, so the linear
    # scaling is the exceedance curve itself
    t = pd.date_range("2024-01-01", periods=30 * 1440, freq="1min")
    ph = (np.arange(len(t)) % 745) / 745.0
    tri = LevelSeries(1.5 * (4 * np.abs(ph - 0.5) - 1.0), t)
    rng = np.random.default_rng(15)
    tt, h = regular_scenes(tri, 300)
    obs = make_obs(h, tt, ramp, rng, clear=0.9)
    out = elev_freq.run_e5b(obs, {"level_series": tri})
    f = out["meta"]["extra"]["f"].astype(float)
    lo, hi = tri.h.min(), tri.h.max()
    assert out["meta"]["H_lo_m"] == lo and out["meta"]["H_hi_m"] == hi
    assert np.allclose(out["z"], (hi - f * (hi - lo)).astype(np.float32), atol=1e-6)
    assert np.median(np.abs(out["z"] - ramp)) < 0.04


def test_sampled_exceedance_is_the_numpy_quantile_and_inside_the_bracket(series, ramp):
    rng = np.random.default_rng(16)
    t, h = random_scenes(series, 100, rng)
    obs = make_obs(h, t, ramp, rng)
    out = elev_freq.run_e5c(obs, None)
    for p in range(len(ramp)):
        v = obs.C[:, p] & np.isfinite(obs.Y[:, p])
        f = obs.W[v, p].mean()
        assert abs(out["z"][p] - np.quantile(h[v], 1 - f)) < 1e-6
    e12 = elev_bracket.run(obs, None)["meta"]["extra"]
    assert np.all(out["z"] >= e12["zmin"] - 1e-6) and np.all(out["z"] <= e12["zmax"] + 1e-6)
    assert np.median(np.abs(out["z"] - ramp)) < 0.03
    # always wet -> lowest sampled level; always dry -> highest
    obs2 = make_obs(h, t, np.array([-9.0, 9.0]), rng, clear=1.0, nan_share=0.0)
    z2 = elev_freq.run_e5c(obs2, None)["z"]
    assert z2[0] == np.float32(h.min()) and z2[1] == np.float32(h.max())


# ─────────────────────────────────────────────────────────────────────────────
#  E7 Coast X-Ray
# ─────────────────────────────────────────────────────────────────────────────

def test_harmonic_analysis_recovers_planted_constants():
    from pyintertidal.boundary import fit_harmonics

    s = tide_series(days=40, z0=0.07, a_m2=1.3, a_s2=0.42, a_k1=0.08, a_o1=0.06)
    hc = harmonic_constants(s)
    assert "K2" not in hc["constituents"] and "P1" not in hc["constituents"]      # Rayleigh, 40 d
    assert abs(hc["Z0"] - 0.07) < 1e-8
    for c, a in (("M2", 1.3), ("S2", 0.42), ("K1", 0.08), ("O1", 0.06)):
        assert abs(hc["amplitudes"][c] - a) < 1e-7, c
    sub = s.slice(s.t[0], s.t[0] + pd.Timedelta(days=20))
    dur_h = (sub.t[-1] - sub.t[0]).total_seconds() / 3600.0
    ref = fit_harmonics(sub.t, sub.h, rayleigh_select(dur_h))
    mine = harmonic_constants(sub)
    assert mine["constituents"] == list(ref)
    for c, (amp, _, _) in ref.items():
        assert abs(mine["amplitudes"][c] - amp) < 1e-9
    ten_days = rayleigh_select(10 * 24.0)
    assert "M2" in ten_days and "S2" not in ten_days           # M2/S2 need 14.8 d


def test_stage_formula_and_bins():
    mhws, mlws = 1.7, -1.3                                   # Z0 = 0.2
    te = np.array([-1.3, 0.2, 1.7])
    printed = elev_coastxray.stage_percent(te, mhws, mlws, "printed")
    corrected = elev_coastxray.stage_percent(te, mhws, mlws, "corrected")
    assert np.allclose(corrected, [0.0, 50.0, 100.0])
    assert np.allclose(printed - corrected, 100.0 * (mhws + mlws) / (mhws - mlws))
    b = elev_coastxray.stage_bins(np.array([-10.01, -10.0, -0.01, 0.0, 99.99, 100.0, 109.99, 110.0, 250.0]))
    assert b.tolist() == [0, 1, 1, 2, 11, 12, 12, 13, 13]
    assert len(elev_coastxray.STAGE_EDGES) + 1 == 14
    with pytest.raises(base.NotComputable):
        elev_coastxray.stage_percent(te, 1.0, 1.0)


def _coastxray_reference(Y, V, h, mhws, mlws, thr, min_images):
    """Independent per-pixel loop of the Coast X-Ray rule."""
    stage = (h + mhws) / (mhws - mlws) * 100.0
    q = np.array([int(np.sum(s >= np.array(elev_coastxray.STAGE_EDGES))) for s in stage])
    out = np.full(Y.shape[1], np.nan)
    for p in range(Y.shape[1]):
        best = np.inf
        for b in np.unique(q):
            rows = np.flatnonzero(q == b)
            if len(rows) < min_images:
                continue
            vals = Y[rows, p][V[rows, p]]
            if len(vals) and np.median(vals.astype(np.float64)) > thr:
                best = min(best, float(np.mean(h[rows])))
        out[p] = best if np.isfinite(best) else np.nan
    return out


def test_coastxray_equals_an_independent_loop(series, ramp):
    rng = np.random.default_rng(17)
    t, h = random_scenes(series, 140, rng)
    obs = make_obs(h, t, ramp, rng, clear=0.7, noise=0.25)       # noisy NDWI around 0.2
    for thr, run in ((0.2, elev_coastxray.run), (0.0, elev_coastxray.run_t0)):
        out = run(obs, {"level_series": series})
        m = out["meta"]
        V = obs.C & np.isfinite(obs.Y)
        ref = _coastxray_reference(obs.Y, V, h, m["MHWS_m"], m["MLWS_m"], thr, 3)
        assert np.allclose(out["z"], ref, equal_nan=True, atol=1e-6, rtol=0)


def test_coastxray_minimum_reducer_offset_is_up_to_one_bin(series, ramp):
    rng = np.random.default_rng(18)
    t, h = regular_scenes(series, 400)
    out = elev_coastxray.run(make_obs(h, t, ramp, rng, clear=0.9), {"level_series": series})
    m = out["meta"]
    assert abs(m["M2_m"] - 1.2) < 1e-6 and abs(m["S2_m"] - 0.45) < 1e-6 and abs(m["Z0_m"] - 0.02) < 1e-6
    bin_m = 0.1 * (m["MHWS_m"] - m["MLWS_m"])
    inner = (ramp > m["MLWS_m"] + bin_m) & (ramp < m["MHWS_m"] - bin_m)
    err = (out["z"][inner] - ramp[inner]) / bin_m
    assert np.all(np.isfinite(err))
    # upward by about half a bin; up to ~one bin, more only where the levels
    # inside a bin are skewed (the theoretical ceiling is two bins)
    assert np.all(err > -0.25) and np.all(err < 1.5)
    assert 0.3 < np.median(err) < 0.7 and np.quantile(err, 0.9) < 1.0


def test_coastxray_skips_bins_with_fewer_than_three_images(series):
    rng = np.random.default_rng(19)
    hc = harmonic_constants(series)
    a = hc["amplitudes"]["M2"] + hc["amplitudes"]["S2"]
    mhws, mlws = hc["Z0"] + a, hc["Z0"] - a
    stages = np.r_[np.repeat([5.0, 25.0, 45.0, 65.0, 85.0], 4), [105.0, 106.0]]      # bin 12 holds 2
    h = stages / 100.0 * (mhws - mlws) - mhws                         # Eq. 1 as printed, inverted
    t = series.t[np.sort(rng.choice(len(series.t), len(h), replace=False))]
    z_px = np.array([0.0, mhws - 0.12])        # the second pixel is wet only in the 2-image bin
    obs = make_obs(h, t, z_px, rng, clear=1.0, nan_share=0.0, noise=0.0)
    out = elev_coastxray.run(obs, {"level_series": series})
    assert 12 not in out["meta"]["bins_used"] and out["meta"]["images_per_bin"][12] == 2
    assert np.isfinite(out["z"][0]) and np.isnan(out["z"][1])
    # with min_images = 2 the bin is used
    out2 = elev_coastxray.run(obs, {"level_series": series, "E7": {"min_images": 2}})
    assert 12 in out2["meta"]["bins_used"] and np.isfinite(out2["z"][1])


# ─────────────────────────────────────────────────────────────────────────────
#  the level series helper
# ─────────────────────────────────────────────────────────────────────────────

class _FakeBoundary:
    name = "fake"

    def __init__(self):
        self.calls = []

    def levels(self, times):
        idx = pd.DatetimeIndex(times)
        assert idx.is_monotonic_increasing
        self.calls.append(len(idx))
        return np.sin(idx.asi8 / 3.6e12)


def test_level_series_from_boundary_chunks_slices_and_round_trips(tmp_path):
    fb = _FakeBoundary()
    s = LevelSeries.from_boundary(fb, "2024-01-01", "2024-01-02", chunk=500)
    assert fb.calls == [500, 500, 440] and len(s.h) == 1440 and s.step == pd.Timedelta(minutes=1)
    one = LevelSeries.from_boundary(_FakeBoundary(), "2024-01-01", "2024-01-02", chunk=10_000)
    assert np.array_equal(s.h, one.h) and s.t.equals(one.t)
    part = s.slice("2024-01-01 06:00", "2024-01-01 07:00")
    assert len(part.h) == 60 and part.t[0] == pd.Timestamp("2024-01-01 06:00")
    s.save(tmp_path / "ls.npz")
    back = LevelSeries.load(tmp_path / "ls.npz")
    assert np.array_equal(back.h, s.h) and back.t.equals(s.t) and back.source == s.source
    start, end = window_period(pd.DatetimeIndex(["2024-03-05 11:02", "2024-04-01 10:50"]))
    assert start == pd.Timestamp("2024-03-05") and end == pd.Timestamp("2024-04-02")
    assert LevelSeries.coerce(None) is None and LevelSeries.coerce(s) is s
    assert np.array_equal(LevelSeries.coerce((s.t, s.h)).h, s.h)
    assert LevelSeries.coerce({"h": s.h}).t is None
