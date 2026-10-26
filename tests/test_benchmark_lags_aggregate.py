# -*- coding: utf-8 -*-
"""Validation v2, package B: the band aggregates, L04 XC, L07 LOOP and L10 HD26-R2.

This module also holds the synthetic world shared by the five lag test files
of package B (``make_world``; imported by test_benchmark_lags_gm20/hd26/ppl/
bt26). Instances are ``types.SimpleNamespace`` objects with the LagInstance
attributes of the plan's INTERFACES §4:

* tide: an analytic M2 + S2 + O1 tide of about 2.3 m range;
* instants: one scene every ``step_days`` days at a fixed hour (+-2 min
  jitter), as Sentinel-2 samples a site;
* LevelView: the exact h_B(t - tau) of the analytic tide (``at``, ``stack``,
  ``series_at``); marks_lp: the HW/LW turning points of the 1-min series;
* pixels: K bands of ``n_px`` pixels with elevations z; the band's local
  level is h_B(t - tau_k); Y = 2 expit((h - z) / 0.05) - 1 + noise (water
  high), W = V & (Y > 0) (the D1 detector), V with ~30 % cloudy scenes; Y
  holds random values where V is False, so a method that ignores V fails.

Run: python -X utf8 -m pytest -q tests/test_benchmark_lags_aggregate.py
"""
from __future__ import annotations

import contextlib
import types

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from experiments.benchmark import product, truth
from experiments.benchmark.lags import aggregate, common
from experiments.benchmark.obs import Observation

# ─────────────────────────────────────────────────────────────────────────────
#  the synthetic world (shared by the package-B lag tests)
# ─────────────────────────────────────────────────────────────────────────────

SPEEDS = {"M2": 28.9841042, "S2": 30.0, "O1": 13.9430356}       # deg/h
AMPS = {"M2": 0.80, "S2": 0.25, "O1": 0.10}                      # m
PHASES = {"M2": 0.3, "S2": 1.1, "O1": 2.0}                       # rad
EPOCH_MIN = float(pd.Timestamp("2023-01-01").value) / 6e10       # minutes since 1970


def tide_fn(amps=None):
    amps = dict(AMPS if amps is None else amps)

    def h(t_min):
        th = (np.asarray(t_min, np.float64) - EPOCH_MIN) / 60.0
        out = np.zeros(th.shape)
        for c, a in amps.items():
            if a:
                out = out + a * np.cos(np.deg2rad(SPEEDS[c]) * th + PHASES[c])
        return out
    return h


class LevelView:
    """Exact h_B(t_i - tau) of an analytic tide at the record instants."""

    def __init__(self, t_min, h):
        self.t_min = np.asarray(t_min, np.float64)
        self.h = h
        self.calls = []

    def at(self, tau):
        self.calls.append(float(tau))
        return self.h(self.t_min - float(tau))

    def stack(self, taus):
        return np.vstack([self.at(t) for t in taus])

    def series_at(self, t_min):
        return self.h(np.asarray(t_min, np.float64))


class ObsView:
    """The instance's observations (T, n): read(cols) and iter_blocks."""

    MAX_CELLS = 250_000_000

    def __init__(self, Y, V, W, allow_read=True):
        self.Y = np.asarray(Y, np.float32)
        self.V = np.asarray(V, bool)
        self.W = np.asarray(W, bool)
        self.T, self.n = self.Y.shape
        self.allow_read = allow_read
        self.read_sizes = []
        self.block_sizes = []

    def _obs(self, cols):
        return Observation(self.Y[:, cols], self.Y[:, cols], self.V[:, cols], self.W[:, cols])

    def read(self, cols=None):
        if not self.allow_read:
            raise AssertionError("this method must stream through iter_blocks")
        cols = np.arange(self.n) if cols is None else np.asarray(cols, np.int64)
        assert cols.ndim == 1 and np.all(np.diff(cols) > 0), "cols must be sorted unique"
        if self.T * len(cols) > self.MAX_CELLS:
            raise ValueError("too many cells")
        self.read_sizes.append(len(cols))
        return self._obs(cols)

    def iter_blocks(self, block=100_000):
        for a in range(0, self.n, block):
            cols = np.arange(a, min(a + block, self.n), dtype=np.int64)
            self.block_sizes.append(len(cols))
            yield cols, self._obs(cols)


def hw_marks(t0_min, t1_min, h):
    """Turning points of the 1-min series of ``h`` on [t0, t1)."""
    tg = np.arange(np.floor(t0_min), np.ceil(t1_min), 1.0)
    hg = h(tg)
    d = np.diff(hg)
    i = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    j = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    k = np.sort(np.concatenate([i, j]))
    return {"t": tg[k], "h": hg[k], "is_hw": np.isin(k, i), "sample_index": k}


def make_world(lags=(0.0, 45.0, 120.0), n_px=240, days=750, step_days=5, seed=7,
               cloudy_share=0.3, noise=0.05, z_range=(-0.9, 0.9), amps=None,
               start="2023-01-03 10:56", allow_read=True, site="wadden", window="W3",
               domain_status="ok"):
    """A synthetic instance (module docstring). Returns ``(inst, truth_dict)``."""
    rng = np.random.default_rng(seed)
    T = int(days // step_days)
    jitter = rng.uniform(-2.0, 2.0, T)
    t = pd.DatetimeIndex(pd.Timestamp(start) + pd.to_timedelta(np.arange(T) * step_days, "D")
                         + pd.to_timedelta(jitter, "min"))
    t_min = aggregate.t_minutes(t)
    h = tide_fn(amps)
    lv = LevelView(t_min, h)
    K = len(lags)
    n = K * n_px
    band_of = np.repeat(np.arange(K), n_px).astype(np.int64)
    z = rng.uniform(z_range[0], z_range[1], n)
    tau_px = np.asarray(lags, np.float64)[band_of]
    h_loc = h(t_min[:, None] - tau_px[None, :])
    Y = 2.0 * expit((h_loc - z[None, :]) / 0.05) - 1.0 + rng.normal(0.0, noise, (T, n))
    cloudy = rng.random(T) < cloudy_share
    V = np.where(cloudy[:, None], rng.random((T, n)) < 0.15, rng.random((T, n)) < 0.98)
    W = V & (Y > 0)
    Y = np.where(V, Y, rng.uniform(-1.0, 1.0, (T, n))).astype(np.float32)
    obs = ObsView(Y, V, W, allow_read=allow_read)
    s_km = (band_of * 3.0 + rng.uniform(0.1, 2.9, n)).astype(np.float64)
    domain = types.SimpleNamespace(
        keep=np.arange(n, dtype=np.int64) * 3 + 11, pos=np.arange(n, dtype=np.int64),
        s_km=s_km, band_of=band_of, centers=np.arange(K) * 3.0 + 1.5,
        edges=np.arange(K + 1) * 3.0, K=K, status=domain_status, n_sel=T,
        sea=np.zeros((4, 4), bool), inter=np.zeros((4, 4), bool))
    marks = hw_marks(t_min.min() - 2 * 1440, t_min.max() + 2 * 1440, h)
    stages = []
    progress = []

    @contextlib.contextmanager
    def stage(name):
        stages.append(name)
        yield

    inst = types.SimpleNamespace(
        site=site, window=window, setting="C1.floor0.ndwi.D1", family="rand", mode="A",
        N="all", rep=0, instance_variant="", sel=np.arange(T, dtype=np.int64), t=t,
        dates=np.array([d.strftime("%Y-%m-%d") for d in t]), year=t.year.to_numpy(np.int64),
        h0=lv.at(0), rising=None, orbit=np.full(T, -1, np.int64), orbit_source="phase10",
        platform=np.array(["S2A"] * T), levels=lv, series1=None, series10=None,
        marks_lp=marks, domain=domain, obs=obs, index="ndwi",
        index_params={"t_k": 0.0, "s_k": 1.0}, nir_type=False, pixel_m=10.0, shape=(4, 4),
        mouth_side="west", grid={}, judges=[], boundary_members=[], paths={},
        installed_models=("EOT20", "GOT4.10c"), seed=0, stage=stage,
        progress=lambda k=1: progress.append(k), cache={}, pkg=None)
    inst.rising = np.asarray(lv.series_at(t_min + 1) > lv.series_at(t_min - 1))
    world = {"lags": np.asarray(lags, float), "band_of": band_of, "z": z, "Y": Y, "V": V,
             "W": W, "T": T, "n": n, "stages": stages, "progress": progress, "h": h,
             "t_min": t_min}
    return inst, world


def unit(method, products, variant=""):
    return {"method": method, "variant": variant, "products": tuple(products),
            "design": "lag", "unit_id": "test", "site": "wadden", "window": "W3",
            "setting": "C1.floor0.ndwi.D1", "family": "rand", "mode": "A", "N": "all",
            "rep": 0}


def band_values(res, world, circular=False):
    """Per-band value of a result: band tau, or the (circular) median of the field."""
    if res["band"] is not None:
        return np.asarray(res["band"]["tau"], float)
    f = np.asarray(res["field"], float)
    out = []
    for k in range(len(world["lags"])):
        v = f[world["band_of"] == k]
        if circular:
            out.append(common.circ_median(v))
        else:
            v = v[np.isfinite(v)]
            out.append(float(np.median(v)) if v.size else np.nan)
    return np.asarray(out)


def lag_errors(values, lags, circular=False):
    d = np.asarray(values, float) - np.asarray(lags, float)
    return np.abs(common.circ_diff(values, lags)) if circular else np.abs(d)


# ─────────────────────────────────────────────────────────────────────────────
#  tests: band aggregates
# ─────────────────────────────────────────────────────────────────────────────

def brute(world):
    Y, V, W, b = world["Y"], world["V"], world["W"], world["band_of"]
    K = len(world["lags"])
    T = world["T"]
    nv = np.zeros((T, K), np.int64)
    nw = np.zeros((T, K), np.int64)
    med = np.full((T, K), np.nan, np.float32)
    for k in range(K):
        m = b == k
        nv[:, k] = V[:, m].sum(1)
        nw[:, k] = (W & V)[:, m].sum(1)
        for t in range(T):
            v = Y[t, m][V[t, m]]
            if v.size:
                med[t, k] = np.median(v)
    return nv, nw, med


def test_band_aggregates_equal_brute_force_single_read():
    inst, w = make_world(seed=1)
    agg = aggregate.band_aggregates(inst)
    nv, nw, med = brute(w)
    assert agg["path"] == ["single"] * 3
    assert np.array_equal(agg["n_valid"], nv) and agg["n_valid"].dtype == np.int32
    assert np.array_equal(agg["n_wet"], nw) and agg["n_wet"].dtype == np.int32
    assert agg["median_Y"].dtype == np.float32
    assert np.array_equal(agg["median_Y"], med, equal_nan=True)
    n_px = np.bincount(w["band_of"])
    assert np.array_equal(agg["n_px"], n_px)
    assert np.allclose(agg["cover"], nv / n_px[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        wf = np.where(nv > 0, nw / np.maximum(nv, 1), np.nan)
    assert np.array_equal(agg["wetfrac"], wf, equal_nan=True)
    assert inst.cache["aggregate.band"] is agg                    # memoised
    assert aggregate.band_aggregates(inst) is agg


def test_band_aggregates_two_pass_selection_is_exact():
    inst, w = make_world(seed=2, n_px=101, days=200, cloudy_share=0.5)
    # ties, both zeros and repeated values make the selection non-trivial
    Y = inst.obs.Y
    rng = np.random.default_rng(5)
    Y[:, ::7] = 0.0
    Y[:, 3::11] = np.float32(-0.0)
    Y[:, 5::13] = np.round(Y[:, 5::13], 1)
    Y[rng.random(Y.shape) < 0.05] = np.float32(0.25)
    w["Y"] = Y
    agg = aggregate.band_aggregates(inst, read_cols=17, cache=False)
    assert agg["path"] == ["two-pass"] * 3
    assert max(inst.obs.read_sizes) <= 17
    nv, nw, med = brute(w)
    assert np.array_equal(agg["n_valid"], nv)
    assert np.array_equal(agg["n_wet"], nw)
    assert np.array_equal(agg["median_Y"], med, equal_nan=True)
    assert "aggregate.band" not in inst.cache                    # cache=False


def test_f32_keys_sort_like_values_and_round_trip():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 1, 500), [0.0, -0.0, 1e-38, -1e-38, 3e38, -3e38,
                                                 np.inf, -np.inf]]).astype(np.float32)
    k = aggregate.f32_keys(x)
    assert k.dtype == np.uint32
    o = np.argsort(k, kind="stable")
    assert np.all(np.diff(x[o].astype(np.float64)) >= 0)
    assert np.array_equal(aggregate.keys_to_f32(k).view(np.uint32), x.view(np.uint32))


def test_reads_respect_the_cell_ceiling():
    assert aggregate.read_cols_for(460) == 100_000
    assert aggregate.read_cols_for(1340) == 50_000_000 // 1340
    assert aggregate.read_cols_for(460, read_cols=250_000) == 50_000_000 // 460


# ─────────────────────────────────────────────────────────────────────────────
#  tests: L04 XC
# ─────────────────────────────────────────────────────────────────────────────

def test_l04_recovers_planted_band_lags():
    inst, w = make_world()
    out = aggregate.run_l04(inst, unit("L04", ("L04w", "L04i")), {})
    assert [r["product"] for r in out] == ["L04w", "L04i"]
    for r in out:
        assert r["status"] == "ok" and r["kind"] == "band" and r["lag_kind"] == "linear"
        err = lag_errors(r["band"]["tau"], w["lags"])
        assert np.median(err) <= 10.0, (r["product"], r["band"]["tau"])
        assert np.all(err <= 15.0), (r["product"], r["band"]["tau"])
        assert not r["band"]["edge"].any()
        f = r["field"]
        assert f.dtype == np.float32 and f.shape == (w["n"],)
        assert np.allclose(f, r["band"]["tau"][w["band_of"]].astype(np.float32))
        assert r["diag"]["score"].shape == (3, 46)
    assert "aggregate" in w["stages"] and "levels" in w["stages"]
    assert sum(w["progress"]) >= w["n"]


def test_l04_edge_flag_at_200_min():
    inst, w = make_world(lags=(200.0, 200.0, 45.0))
    out = aggregate.run_l04(inst, unit("L04", ("L04w", "L04i")), {})
    for r in out:
        assert list(r["band"]["edge"]) == [True, True, False]
        assert np.all(r["band"]["tau"][:2] == common.SCAN_HI)
        assert r["edge_share"] == pytest.approx(2.0 / 3.0)


def test_l04_coverage_and_usable_scene_rules():
    # xc_band: 29 usable scenes is too few, 30 is enough
    inst, w = make_world(seed=4)
    H = aggregate.levels_scan(inst)
    p = np.linspace(0.0, 1.0, inst.obs.T)
    assert aggregate.xc_band(p[:29], H[:, :29])["reason"] == "usable_lt_30"
    assert np.isfinite(aggregate.xc_band(p[:30], H[:, :30])["tau"])
    # a band whose every scene is below 80 % valid is undefined
    inst, w = make_world(seed=4)
    V = inst.obs.V
    b1 = np.flatnonzero(w["band_of"] == 1)
    V[:, b1] &= (np.arange(240) % 5 != 0)[None, :]          # at most 192/240 = 0.8 valid ...
    V[:, b1[1]] = False                                     # ... and now below 0.8
    inst.obs.W &= V
    r = aggregate.run_l04(inst, unit("L04", ("L04w",)), {})[0]
    assert r["status"] == "ok"
    assert np.isnan(r["band"]["tau"][1]) and np.isfinite(r["band"]["tau"][[0, 2]]).all()
    assert r["diag"]["band_reason"][1] == "usable_lt_30"
    assert r["diag"]["n_usable"][1] == 0
    # the cover0.5 variant accepts those scenes again
    inst.cache.clear()
    r5 = aggregate.run_l04(inst, unit("L04", ("L04w",), "cover0.5"), {})[0]
    assert np.isfinite(r5["band"]["tau"][1]) and r5["diag"]["min_cover"] == 0.5


def test_l04_not_computable_when_no_band_has_30_usable_scenes():
    inst, w = make_world(days=140, seed=6)                 # 28 scenes in all
    out = aggregate.run_l04(inst, unit("L04", ("L04w", "L04i")), {})
    for r in out:
        assert r["status"] == "not_computable" and r["stage"] == "usable_lt_30"
        assert r["field"] is None


def test_l04_empty_when_every_score_is_undefined():
    inst, w = make_world(seed=8)
    inst.obs.W[:] = False                                   # a constant wet fraction
    r = aggregate.run_l04(inst, unit("L04", ("L04w",)), {})[0]
    assert r["status"] == "empty" and r["stage"] == "qc"
    assert set(r["diag"]["band_reason"]) == {"score_undefined"}


# ─────────────────────────────────────────────────────────────────────────────
#  tests: L07 LOOP
# ─────────────────────────────────────────────────────────────────────────────

def test_l07_recovers_planted_band_lags_and_flags_the_edge():
    inst, w = make_world()
    r = aggregate.run_l07(inst, unit("L07", ("L07",)), {})[0]
    assert r["status"] == "ok" and r["product"] == "L07"
    err = lag_errors(r["band"]["tau"], w["lags"])
    assert np.median(err) <= 10.0 and np.all(err <= 15.0), r["band"]["tau"]
    inst, w = make_world(lags=(0.0, 200.0, 45.0), seed=9)
    r = aggregate.run_l07(inst, unit("L07", ("L07",)), {})[0]
    assert list(r["band"]["edge"]) == [False, True, False]
    assert r["band"]["tau"][1] == common.SCAN_HI


def test_l07_limb_rule():
    rng = np.random.default_rng(0)
    n = 40
    x = rng.uniform(-1, 1, (46, n))
    p = np.clip(x[10] + rng.normal(0, 0.05, n), 0, 1)
    R = np.where(np.arange(n) < 25, 1.0, -1.0)[None, :].repeat(46, 0)
    ok = aggregate.loop_band(p, x, R)
    assert np.isfinite(ok["tau"]) and ok["n_rise"][0] == 25 and ok["n_fall"][0] == 15
    R14 = np.where(np.arange(n) < 26, 1.0, -1.0)[None, :].repeat(46, 0)     # 14 falling
    bad = aggregate.loop_band(p, x, R14)
    assert np.isnan(bad["tau"]) and bad["reason"] == "limb_lt_15"
    assert np.all(np.isnan(bad["score"]))
    few = aggregate.loop_band(p[:29], x[:, :29], R[:, :29])
    assert few["reason"] == "usable_lt_30"


def test_l07_score_is_one_isotonic_fit_through_both_limbs():
    rng = np.random.default_rng(1)
    h = rng.uniform(-1, 1, 50)
    y = np.clip(0.5 + 0.4 * h + rng.normal(0, 0.1, 50), 0, 1)
    o = np.argsort(h)
    fit = truth.isotonic(y[o])
    assert aggregate.isotonic_sse(y, h) == pytest.approx(float(np.sum((y[o] - fit) ** 2)))
    assert np.all(np.diff(fit) >= -1e-15)


# ─────────────────────────────────────────────────────────────────────────────
#  tests: L10 HD26-R2 (the band part; the unit is tested in test_benchmark_lags_hd26)
# ─────────────────────────────────────────────────────────────────────────────

def test_l10_recovers_planted_lags_and_reports_s():
    inst, w = make_world()
    r = aggregate.l10_hd26r2(inst)
    assert r["product"] == "L10" and r["status"] == "ok"
    err = lag_errors(r["band"]["tau"], w["lags"])
    assert np.median(err) <= 10.0 and np.all(err <= 15.0), r["band"]["tau"]
    assert set(r["diag"]["s_best_min"]) <= {0.0, 15.0, 30.0, 60.0}
    assert r["diag"]["sse_min_by_s"].shape == (3, 4)


def test_l10_at_s0_equals_the_plain_isotonic_scan():
    inst, w = make_world(seed=11)
    r = aggregate.l10_hd26r2(inst, s_grid=(0,))
    agg = aggregate.band_aggregates(inst)
    H = aggregate.levels_scan(inst)
    for k in range(3):
        use = agg["cover"][:, k] >= 0.8
        p = agg["wetfrac"][use, k]
        sse = np.array([float(np.sum((p[np.argsort(H[j, use], kind="stable")]
                                      - truth.isotonic(p[np.argsort(H[j, use], kind="stable")]))
                                     ** 2)) for j in range(46)])
        tau, edge = common.scan_vertex(-sse)
        assert r["band"]["tau"][k] == tau and bool(r["band"]["edge"][k]) == edge
    assert np.all(r["diag"]["s_best_min"] == 0.0)


def test_l10_needs_60_scenes():
    inst, w = make_world(days=400, seed=12, cloudy_share=0.5)        # ~40 usable
    r = aggregate.l10_hd26r2(inst)
    assert r["status"] == "not_computable" and r["stage"] == "scenes_lt_60"
    assert all(x == "scenes_lt_60" for x in r["diag"]["band_reason"])


def test_smear_is_the_gaussian_sum_of_the_one_minute_series():
    inst, w = make_world(days=60, seed=13)
    U = aggregate.smeared_scan(inst, 15)
    m, wts = aggregate.gaussian_weights(15)
    assert m[0] == -45 and m[-1] == 45 and wts.sum() == pytest.approx(1.0)
    j = 20                                                   # tau = 55 min
    direct = sum(wk * w["h"](w["t_min"] - common.TAU_SCAN[j] - mk) for mk, wk in zip(m, wts))
    assert np.allclose(U[j], direct, atol=1e-12)
    assert aggregate.smeared_scan(inst, 0) is aggregate.levels_scan(inst)


# ─────────────────────────────────────────────────────────────────────────────
#  tests: cards, domain status, registry
# ─────────────────────────────────────────────────────────────────────────────

def test_cards_pass_check_card():
    for pid, c in {**aggregate.CARDS, "L10": aggregate.CARD_L10}.items():
        assert common.check_card(c) == [], pid
        assert c["id"] == pid
        assert c["scan_window"] == [-45.0, 180.0, 5.0]
        assert c["tuning"] == [] and isinstance(c["control_offset_min"], float)


def test_no_intertidal_domain_is_not_computable():
    inst, w = make_world(domain_status="no_intertidal", days=60)
    for fn in (aggregate.run_l04, aggregate.run_l07):
        with pytest.raises(product.NotComputable) as e:
            fn(inst, unit("L04", ("L04w", "L04i")) if fn is aggregate.run_l04
               else unit("L07", ("L07",)), {})
        assert e.value.stage == "no_intertidal"


def test_registry_loads_these_run_functions_and_cards():
    from experiments.benchmark import lags
    assert lags.load("L04") is aggregate.run_l04
    assert lags.load("L07") is aggregate.run_l07
    for pid in ("L04w", "L04i", "L07"):              # only this package's modules imported
        c = lags.card(pid)
        assert common.check_card(c) == [] and c["id"] == pid, pid
    assert lags.card("L04i")["id"] == "L04i"


def test_unknown_products_are_refused():
    inst, w = make_world(days=60)
    with pytest.raises(ValueError):
        aggregate.run_l04(inst, unit("L04", ("L04w", "L07")), {})


def test_card_copies_of_the_unit_signature_and_cost_match_designs():
    from experiments.benchmark import designs
    for method in ("L04", "L07", "L08", "L09", "L11", "L12"):
        m = designs.METHODS[method]
        assert aggregate.card_signature(method) == {"reads": sorted(m.reads),
                                                    "domain": bool(m.domain)}, method
        assert aggregate.card_cost(method) == {"wh_cell": float(m.cost_wh),
                                               "ram_gb": float(m.ram_gb)}, method


def test_constant_proxy_is_undefined_not_an_edge():
    inst, w = make_world(seed=14, days=400)
    H = aggregate.levels_scan(inst)
    R = aggregate.rate_scan(inst)
    p = np.full(inst.obs.T, 0.4)
    r = aggregate.loop_band(p, H, R)
    assert np.isnan(r["tau"]) and not r["edge"] and r["reason"] == "proxy_constant"
    r = aggregate.l10_band(p, [H], [0.0])
    assert np.isnan(r["tau"]) and r["reason"] == "proxy_constant"
    assert np.isnan(aggregate.xc_band(p, H)["tau"])


def test_every_result_revalidates_unchanged():
    """The queue re-validates each dict with lag_result(**d, n_keep=, K=)."""
    from experiments.benchmark.lags import bt26, gm20, hd26, ppl
    inst, w = make_world(seed=15, days=500)
    out = (aggregate.run_l04(inst, unit("L04", ("L04w", "L04i")), {})
           + aggregate.run_l07(inst, unit("L07", ("L07",)), {})
           + hd26.run_unit(inst, unit("L09", ("L09", "L10")), {})
           + gm20.run_unit(inst, unit("L08", ("L08",), "boot30"), {"refits": 5})
           + ppl.run_unit(inst, unit("L11", ("L11",)), {})
           + bt26.run_unit(inst, unit("L12", ("L12",)), {}))
    for r in out:
        d = dict(r)
        again = common.lag_result(d.pop("product"), d.pop("status"), n_keep=w["n"], K=3, **d)
        assert again.keys() == r.keys()
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(again[k], v, equal_nan=True), (r["product"], k)
        assert again["status"] == "ok" and again["edge_share"] == r["edge_share"]


def test_configs_cards_and_code_deps_are_hashable_repository_facts():
    """CONFIG and CARDS enter the config key as JSON; CODE_DEPS are files."""
    import os

    from experiments.benchmark import config
    from experiments.benchmark.lags import bt26, gm20, hd26, ppl
    for mod in (aggregate, gm20, hd26, ppl, bt26):
        assert len(config.sha256_json(mod.CONFIG)) == 64, mod.__name__
        assert len(config.sha256_json(mod.CARDS)) == 64, mod.__name__
        for path in mod.CODE_DEPS:
            assert os.path.isfile(os.path.join(config.ROOT, path)), (mod.__name__, path)



def test_l07_scores_the_loop_gap_not_one_pooled_fit():
    """L07 is the hysteresis-loop collapse: the score is minus the extra
    residual of forcing the rising and falling limbs onto one isotonic curve
    (>= 0, 0 when they collapse). The V2 build's -SSE_pooled equalled L10 at
    s = 0 (code review 2026-10-20)."""
    rng = np.random.default_rng(2)
    n = 80
    h = rng.uniform(-1.0, 1.0, n)
    rate = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    p = np.clip(0.5 + 0.4 * h + rng.normal(0, 0.02, n), 0, 1)
    gap0 = aggregate.loop_gap(p, h, rate)
    assert 0.0 <= gap0 < 0.01                              # collapsed limbs
    loop = np.clip(0.5 + 0.4 * (h + 0.3 * rate) + rng.normal(0, 0.02, n), 0, 1)
    gap1 = aggregate.loop_gap(loop, h, rate)
    assert gap1 > 10 * max(gap0, 1e-6)                     # a loop: the limbs separate
    up, dn = rate > 0, rate < 0
    want = (aggregate.isotonic_sse(loop, h) - aggregate.isotonic_sse(loop[up], h[up])
            - aggregate.isotonic_sse(loop[dn], h[dn]))
    assert gap1 == pytest.approx(want)
    # on the same scenes the L07 and L10(s = 0) scores now differ
    H = np.vstack([h + d for d in np.linspace(-0.5, 0.5, 46)])
    R = np.repeat(rate[None, :], 46, axis=0)
    r7 = aggregate.loop_band(loop, H, R)
    r10 = aggregate.l10_band(loop, [H], [0.0], min_scenes=30)
    assert not np.allclose(r7["score"], -aggregate.isotonic_scan(loop, H), equal_nan=True)
    assert np.isfinite(r7["tau"]) and np.isfinite(r10["tau"])
    assert aggregate.CONFIG["L07"]["score"].startswith("-(SSE_pooled - SSE_rise - SSE_fall)")
    assert aggregate.CARDS["L07"]["deviations"][0]["rows"] == "L07"
