"""Validation v2, Stage L: MAREA's lag family (lags/marea_family.py).

L01's lag stage is te.m2a_rasch exactly as marea.reconstruct calls it,
with the column draw replayed and only the drawn columns read (bitwise
equal to the full-matrix call, bands above and below max_px_band); its
bootstrap is te.bootstrap_lags draw for draw; L01v fits each relative
orbit; L02 and L03 equal te.m2b_concordance and te.m2c_waterline on the
full matrices. Tiny synthetic instances (SimpleNamespace objects with the
LagInstance attributes): at most 120 scenes, 3 bands of at most 400 px.
"""
import contextlib
import types

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark import config, obs
from experiments.benchmark.lags import common
from experiments.benchmark.lags import marea_family as mf
from experiments.benchmark.product import NotComputable
from pyintertidal import marea
from pyintertidal import tide_estimators as te
from tests.benchmark_site import FakeTide


# ─────────────────────────────────────────────────────────────────────────────
#  a synthetic instance (shared with the E0 and gate tests)
# ─────────────────────────────────────────────────────────────────────────────

class FakeLevels:
    """h_B(t - tau) of FakeTide at the instants (exact for any tau)."""

    def __init__(self, t):
        self.t = pd.DatetimeIndex(t)
        self.calls = []

    def at(self, tau):
        self.calls.append(float(tau))
        return FakeTide.value((self.t - pd.Timedelta(minutes=float(tau))).asi8)

    def stack(self, taus):
        return np.vstack([self.at(x) for x in taus])

    def bank(self, taus=None):
        taus = list(marea.BANK_TAUS if taus is None else taus)
        return te.ShiftBank(taus, [self.at(x) for x in taus])


class FakeObs:
    """The ObsView of a synthetic instance; records the columns it reads."""

    def __init__(self, X, V, W):
        self.X, self.V, self.W = X, V, W
        self.T, self.n = X.shape
        self.reads = []

    def read(self, cols=None):
        cols = np.arange(self.n) if cols is None else np.asarray(cols, np.int64)
        self.reads.append(cols.copy())
        return obs.Observation(self.X[:, cols], self.X[:, cols], self.V[:, cols],
                               self.W[:, cols])

    def iter_blocks(self, block=100_000):
        for a in range(0, self.n, block):
            cols = np.arange(a, min(a + block, self.n))
            yield cols, self.read(cols)

    def read_union(self):
        return np.unique(np.concatenate(self.reads)) if self.reads else np.zeros(0, int)


def make_data(T=120, npx=(300, 400, 350), taus=(0.0, 60.0, 120.0), seed=3, clear_p=0.85,
              n_nan_s=6, noise=0.03):
    """Wet iff h_B(t - tau_k) > z + noise per band k; NDWI-like X."""
    rng = np.random.default_rng(seed)
    t = (pd.date_range("2023-01-05", periods=T, freq="5D") + pd.Timedelta(hours=10, minutes=50)
         + pd.to_timedelta(rng.integers(0, 900, T), "s"))
    lev = FakeLevels(t)
    K = len(npx)
    band_of = np.concatenate([np.full(n, k, np.int64) for k, n in enumerate(npx)])
    perm = rng.permutation(len(band_of))
    band_of = band_of[perm]
    s_km = 3.0 * band_of + rng.uniform(0.05, 2.95, len(band_of))
    if n_nan_s:
        iso = rng.choice(len(band_of), n_nan_s, replace=False)
        s_km[iso] = np.nan
        band_of[iso] = -1
    n = len(band_of)
    h0 = lev.at(0)
    R = h0.max() - h0.min()
    z = rng.uniform(h0.min() + 0.15 * R, h0.max() - 0.15 * R, n)
    Hk = np.stack([lev.at(x) for x in taus])
    local = Hk[np.where(band_of >= 0, band_of, 0)].T
    wet_true = local > z[None] + rng.normal(0.0, noise, (T, n))
    V = rng.random((T, n)) < clear_p
    X = np.where(wet_true, rng.normal(0.3, 0.08, (T, n)),
                 rng.normal(-0.3, 0.08, (T, n))).astype(np.float32)
    W = V & (X > 0)
    centers = np.array([float(np.median(s_km[band_of == k])) for k in range(K)])
    return dict(t=t, lev=lev, band_of=band_of, s_km=s_km, z=z, X=X, V=V, W=W,
                centers=centers, K=K, T=T, n=n, taus=np.asarray(taus, float))


def make_inst(d, status="ok", orbit=None, site="escalda", setting="C1.floor0.ndwi.D1",
              sel=None, pkg=None):
    """A SimpleNamespace with the LagInstance attributes the MAREA family reads."""
    T = d["T"]
    sel = np.arange(T) if sel is None else np.asarray(sel)
    dom = types.SimpleNamespace(keep=np.arange(d["n"], dtype=np.int64),
                                pos=np.arange(d["n"], dtype=np.int64), s_km=d["s_km"],
                                band_of=d["band_of"], centers=d["centers"],
                                edges=3.0 * np.arange(d["K"] + 1), K=d["K"], status=status,
                                n_keep=d["n"], sea=None, inter=None, n_sel=T)
    if status in ("no_anchor", "no_intertidal"):
        dom.band_of = np.zeros(d["n"], np.int64)
        dom.centers = np.array([0.0])
        dom.K = 1
    lev = d["lev"]
    return types.SimpleNamespace(
        site=site, window="W3", setting=setting, family="rand", mode="A", N="all", rep=0,
        instance_variant="", sel=sel, t=d["t"], h0=lev.at(0), levels=lev, domain=dom,
        obs=FakeObs(d["X"], d["V"], d["W"]),
        orbit=np.full(T, -1, np.int64) if orbit is None else np.asarray(orbit, np.int64),
        orbit_source="scene_meta", cache={}, pkg=pkg, paths={}, judges=[],
        stage=lambda name: contextlib.nullcontext(), progress=lambda n=1: None)


def _bits(a):
    return np.ascontiguousarray(a, np.float64).view(np.int64)


def bit_equal(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(
        _bits(np.nan_to_num(a, nan=0.0)), _bits(np.nan_to_num(b, nan=0.0)))


# ─────────────────────────────────────────────────────────────────────────────
#  L01: the lag stage
# ─────────────────────────────────────────────────────────────────────────────

SMALL_DRAWS = {"MAX_PX_BAND": 330, "BOOT_MAX_PX": 120}


@pytest.fixture(scope="module")
def planted():
    """One lag stage with the bootstrap (3 draws) and the full-matrix
    references, with max_px_band lowered so bands 0 (300 px) sits below and
    bands 1-2 (400, 350 px) above it."""
    with pytest.MonkeyPatch.context() as mp:
        for k, v in SMALL_DRAWS.items():
            mp.setattr(mf, k, v)
        d = make_data()
        inst = make_inst(d)
        st = mf.lag_stage(inst, n_boot=3, boot_threads=2)
        bank = d["lev"].bank()
        kw = mf.fit_kw()
        ref = te.m2a_rasch(d["W"], d["V"], bank, d["band_of"], d["centers"], d["K"],
                           rng=np.random.default_rng(mf.COLUMN_SEED), **kw)
        boot = te.bootstrap_lags(d["W"], d["V"], bank, d["band_of"], d["centers"], d["K"],
                                 n_boot=3, seed=mf.BOOT_SEED,
                                 **{**kw, "max_px_band": mf.BOOT_MAX_PX})
        reads = inst.obs.read_union()
        main = mf.draw_columns(d["band_of"], d["K"], mf.MAX_PX_BAND,
                               np.random.default_rng(mf.COLUMN_SEED))
        bcols = [mf.draw_columns(d["band_of"], d["K"], mf.BOOT_MAX_PX,
                                 np.random.default_rng(mf.BOOT_SEED + 100 + b)) for b in range(3)]
        counts = [int((d["band_of"] == k).sum()) for k in range(d["K"])]
        max_px = mf.MAX_PX_BAND
    return types.SimpleNamespace(d=d, inst=inst, st=st, ref=ref, boot=boot, reads=reads,
                                 main=main, bcols=bcols, kw=kw, counts=counts, max_px=max_px)


def test_lag_stage_equals_m2a_on_full_matrices_bitwise(planted):
    st, ref = planted.st, planted.ref
    assert bit_equal(st["tau_raw"], ref["tau"])
    assert st["nll"] == ref["nll"] and st["converged"] == ref["converged"]
    # the draw: band 0 (< max_px_band) is read whole, bands 1 and 2 are drawn
    sizes = [len(c) for c in planted.main]
    c, m = planted.counts, planted.max_px
    assert c[0] <= m < min(c[1], c[2])
    assert sizes == [c[0], m, m]
    assert list(st["n_cols"]) == sizes
    # only the drawn columns (main fit and bootstrap draws) were read
    want = mf._union(planted.main, *planted.bcols)
    assert np.array_equal(planted.reads, want)
    assert len(want) < planted.d["n"]
    # isolated pixels (NaN s, band -1) are never read nor fitted
    assert not np.isin(np.flatnonzero(planted.d["band_of"] < 0), want).any()


def test_bootstrap_equals_bootstrap_lags_draw_by_draw(planted):
    st, boot = planted.st, planted.boot
    assert st["boot"].shape == (3, planted.d["K"])
    for b in range(3):
        assert bit_equal(st["boot"][b], boot[b]), b
    with np.errstate(invalid="ignore"):
        assert bit_equal(st["lo"], np.nanpercentile(boot, 2.5, axis=0))
        assert bit_equal(st["hi"], np.nanpercentile(boot, 97.5, axis=0))
    res = mf.l01_result(st)
    assert res["interval95"]["level"] == "band" and res["interval95"]["nominal"] == "bootstrap3"
    assert bit_equal(res["band"]["lo"], st["lo"])


def test_planted_profile_recovered_and_applied_lag(planted):
    st, d = planted.st, planted.d
    err = np.abs(st["tau_used"] - d["taus"])
    assert np.median(err) <= 10.0, (st["tau_used"], d["taus"])
    # causality clip and the smoothed, rounded applied lag of reconstruct
    tau_used, veto = mf.causality(st["tau_raw"])
    assert bit_equal(st["tau_used"], tau_used) and not veto
    sm = te.smooth_lag_profile(d["centers"], tau_used, d["s_km"], 3.0)
    want = np.where(np.isfinite(sm), np.round(sm), 0.0)
    assert st["tau_px"].dtype == np.float32
    assert np.array_equal(st["tau_px"].astype(np.float64), want)
    assert np.all(st["tau_px"][d["band_of"] < 0] == 0)
    # the L01 product: band tau = tau_used, field = the applied lag
    res = mf.l01_result(st)
    assert res["kind"] == "band" and res["lag_kind"] == "linear"
    assert bit_equal(res["band"]["tau"], st["tau_used"])
    assert np.array_equal(res["field"], st["tau_px"])
    assert bit_equal(res["diag"]["tau_raw"], st["tau_raw"])
    assert list(res["band"]["n_px"]) == [int((d["band_of"] == k).sum()) for k in range(3)]


def test_lag_stage_is_memoised_per_instance(planted):
    inst = planted.inst
    n = len(inst.obs.reads)
    again = mf.lag_stage(inst, n_boot=3, boot_threads=1)
    assert again is planted.st and len(inst.obs.reads) == n


def test_threaded_bootstrap_advances_progress_per_draw():
    """--boot-threads > 1: the progress counter moves with every finished
    draw (never one lump at the end), so a long bootstrap never looks hung to
    the queue's monitor (spec §13.5)."""
    with pytest.MonkeyPatch.context() as mp:
        for k, v in SMALL_DRAWS.items():
            mp.setattr(mf, k, v)
        d = make_data()
        inst = make_inst(d)
        ticks = []
        inst.progress = lambda n=1: ticks.append(int(n))
        st = mf.lag_stage(inst, n_boot=3, boot_threads=2)
    assert st["boot"].shape == (3, d["K"])
    assert ticks.count(1) >= 5 and 3 not in ticks       # read, m2a, then one per draw


def test_causality_clip_and_smoothing_helpers():
    tau_used, veto = mf.causality([0.0, -3.0, 5.0, -0.0])
    assert tau_used.tolist() == [0.0, 0.0, 5.0, 0.0] and veto
    s = np.array([0.5, 4.0, np.nan, 7.2])
    out = mf.applied_lag(np.array([1.0, 4.0, 7.0]), np.array([0.0, 10.0, 31.0]), s)
    sm = te.smooth_lag_profile(np.array([1.0, 4.0, 7.0]), np.array([0.0, 10.0, 31.0]), s, 3.0)
    assert out[2] == 0.0 and np.array_equal(out[[0, 1, 3]], np.round(sm[[0, 1, 3]]))
    assert np.array_equal(mf.applied_lag(np.array([0.0]), np.array([0.0]), s), np.zeros(4))
    assert mf.m2a_edge([0.0, 179.6, 200.0]).tolist() == [False, True, True]
    assert mf.m2a_edge([0.0, 20.0, 199.6]).tolist() == [False, False, True]
    assert mf.scan_edge([-45.0, 0.0, 180.0, np.nan]).tolist() == [True, False, True, False]


def test_seed_offset_changes_only_the_column_draw():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mf, "MAX_PX_BAND", 200)
        d = make_data(T=90, npx=(250, 300, 260), seed=11)
        inst = make_inst(d)
        st = mf.lag_stage(inst, n_boot=0, seed_offset=2)
        ref = te.m2a_rasch(d["W"], d["V"], d["lev"].bank(), d["band_of"], d["centers"], d["K"],
                           rng=np.random.default_rng(mf.COLUMN_SEED + 2000), **mf.fit_kw())
        assert bit_equal(st["tau_raw"], ref["tau"]) and st["seed_offset"] == 2
        assert st["lo"] is None and st["boot"] is None
        want = mf._union(mf.draw_columns(d["band_of"], 3, 200,
                                         np.random.default_rng(mf.COLUMN_SEED + 2000)))
        other = mf._union(mf.draw_columns(d["band_of"], 3, 200,
                                          np.random.default_rng(mf.COLUMN_SEED)))
        assert np.array_equal(inst.obs.read_union(), want) and not np.array_equal(want, other)
        assert ("marea_family.lag_stage" in inst.cache
                and (0, 2, False) in inst.cache["marea_family.lag_stage"])


def test_zero_profile_gives_small_lags():
    d = make_data(T=110, npx=(260, 300, 280), taus=(0.0, 0.0, 0.0), seed=21)
    st = mf.lag_stage(make_inst(d))
    assert np.all(np.abs(st["tau_raw"]) <= 5.0), st["tau_raw"]


def test_no_anchor_single_band_no_intertidal_and_floor():
    d = make_data(T=60, npx=(120, 120), seed=5, n_nan_s=0)
    inst = make_inst(d, status="no_anchor")
    st = mf.lag_stage(inst, n_boot=30)
    assert st["tau_raw"].tolist() == [0.0] and st["tau_used"].tolist() == [0.0]
    assert np.all(st["tau_px"] == 0) and np.isnan(st["lo"]).all() and st["boot"] is None
    assert "no_anchor" in st["flags"] and inst.obs.reads == []
    res = mf.l01_result(st)
    assert res["band"]["tau"].tolist() == [0.0] and res["status"] == "ok"
    # one band with an anchor: tau = 0 (m2a has no free parameter)
    one = make_inst(d)
    one.domain.band_of = np.zeros(d["n"], np.int64)
    one.domain.centers, one.domain.K = np.array([1.0]), 1
    st1 = mf.lag_stage(one)
    assert st1["tau_raw"].tolist() == [0.0] and "single_band" in st1["flags"]
    with pytest.raises(NotComputable) as e:
        mf.lag_stage(make_inst(d, status="no_intertidal"))
    assert e.value.stage == "no_intertidal"
    short = make_inst(d, sel=np.arange(39))
    short.h0 = short.h0[:39]
    with pytest.raises(NotComputable) as e:
        mf.lag_stage(short)
    assert e.value.stage == "record_floor" and e.value.diag["floor"] == 40


# ─────────────────────────────────────────────────────────────────────────────
#  L01v
# ─────────────────────────────────────────────────────────────────────────────

def test_l01v_fits_each_relative_orbit_and_lists_small_groups():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mf, "MAX_PX_BAND", 220)
        d = make_data(T=100, npx=(200, 260, 240), seed=8)
        orbit = np.array([8] * 55 + [51] * 30 + [-1] * 15)
        orbit = orbit[np.random.default_rng(1).permutation(100)]
        inst = make_inst(d, orbit=orbit)
        (res,) = mf.run_l01v(inst, {"method": "L01v", "products": ("L01v",)}, {})
        assert res["status"] == "ok" and res["kind"] == "groups"
        g = res["groups"]
        assert list(g) == ["orbit8", "orbit51"]
        assert g["orbit51"]["status"] == "not_computable" and g["orbit51"]["n_scenes"] == 30
        assert np.isnan(g["orbit51"]["tau"]).all()
        idx = np.flatnonzero(orbit == 8)
        bank = d["lev"].bank()
        seed = [mf.COLUMN_SEED, config.SITE_ID["escalda"], 8]
        ref = te.m2a_rasch(d["W"][idx], d["V"][idx], te.ShiftBank(bank.taus, bank.H[:, idx]),
                           d["band_of"], d["centers"], 3, rng=np.random.default_rng(seed),
                           **mf.fit_kw())
        assert bit_equal(g["orbit8"]["tau_raw"], ref["tau"])
        assert bit_equal(g["orbit8"]["tau"], mf.causality(ref["tau"])[0])
        assert g["orbit8"]["n_scenes"] == 55
        assert res["diag"]["n_unknown_orbit"] == 15 and res["diag"]["orbit_source"] == "scene_meta"
        assert res["diag"]["seeds"]["orbit8"]["seed"] == seed


def test_l01v_without_orbits_or_large_groups_is_not_computable():
    d = make_data(T=60, npx=(120, 120), seed=9, n_nan_s=0)
    with pytest.raises(NotComputable) as e:
        mf.run_l01v(make_inst(d), {}, {})
    assert e.value.stage == "no_orbit"
    (res,) = mf.run_l01v(make_inst(d, orbit=np.arange(60) % 3), {}, {})
    assert res["status"] == "not_computable" and res["stage"] == "orbit_floor"
    assert all(e["status"] == "not_computable" for e in res["groups"].values())


# ─────────────────────────────────────────────────────────────────────────────
#  L02 and L03
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def scan_data():
    return make_data(T=100, npx=(40, 260, 300), taus=(0.0, 35.0, 80.0), seed=13)


def test_l02_equals_m2b_on_full_matrices(scan_data):
    d = scan_data
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mf, "M2B_MAX_PX", 280)
        inst = make_inst(d)
        (res,) = mf.run_l02(inst, {"method": "L02"}, {})
        grid = np.asarray(common.TAU_SCAN, float)
        bank = te.ShiftBank(grid, d["lev"].stack(grid))
        ref = te.m2b_concordance(d["W"], d["V"], bank, d["band_of"], d["K"], grid,
                                 min_obs=10, max_px_band=280)
        assert bit_equal(res["band"]["tau"], ref)
        # band 0 has 40 px (< 50): skipped before any draw; band 2 (300) drawn to 280
        assert np.isnan(res["band"]["tau"][0]) and np.isfinite(res["band"]["tau"][1:]).all()
        c = [int((d["band_of"] == k).sum()) for k in range(3)]
        assert c[0] < 50 and c[1] <= 280 < c[2]
        assert res["diag"]["n_cols"].tolist() == [0, c[1], 280]
        reads = inst.obs.read_union()
        assert len(reads) == c[1] + 280
        assert not np.isin(np.flatnonzero(d["band_of"] == 0), reads).any()
    assert res["status"] == "ok" and res["kind"] == "band" and res["lag_kind"] == "linear"
    assert np.array_equal(res["field"], common.broadcast_band(ref, d["band_of"]),
                          equal_nan=True)
    assert res["band"]["edge"].tolist() == mf.scan_edge(ref).tolist()
    assert abs(res["band"]["tau"][2] - 80.0) <= 10.0 and abs(res["band"]["tau"][1] - 35.0) <= 10


def test_l03_equals_m2c_band_by_band(scan_data):
    d = scan_data
    inst = make_inst(d)
    (res,) = mf.run_l03(inst, {"method": "L03"}, {})
    grid = np.asarray(common.TAU_SCAN, float)
    bank = te.ShiftBank(grid, d["lev"].stack(grid))
    ref = te.m2c_waterline(d["W"], d["V"], bank, d["band_of"], d["K"], grid,
                           min_cover=0.3, min_scenes=60)
    assert bit_equal(res["band"]["tau"], ref)
    # one band's columns per read; the 40-px band is never read
    assert len(inst.obs.reads) == 2
    assert [sorted(r.tolist()) for r in inst.obs.reads] == [
        np.flatnonzero(d["band_of"] == k).tolist() for k in (1, 2)]
    assert res["diag"]["n_scenes_used"][0] == 0 and res["diag"]["n_scenes_used"][1] > 60
    assert np.isnan(res["band"]["tau"][0])
    short = make_inst(d, sel=np.arange(59))
    with pytest.raises(NotComputable) as e:
        mf.run_l03(short, {}, {})
    assert e.value.stage == "min_scenes"


def test_l02_l03_empty_and_no_intertidal():
    d = make_data(T=30, npx=(30, 40), seed=2, n_nan_s=0)
    (res,) = mf.run_l02(make_inst(d), {}, {})
    assert res["status"] == "empty" and np.isnan(res["field"]).all() and res["reason"]
    with pytest.raises(NotComputable):
        mf.run_l02(make_inst(d, status="no_intertidal"), {}, {})


def test_cards_registry_and_config():
    import json

    from experiments.benchmark import lags
    for pid, card in mf.CARDS.items():
        assert common.check_card(card) == [], pid
        assert card["id"] == pid and card["fidelity"] == "F0" and card["tuning"] == []
    assert mf.CARDS["L02"]["scan_window"] == [-45.0, 180.0, 5.0]
    assert "asbuilt" in mf.CARDS["L02"]["variants"] and "asbuilt" in mf.CARDS["L03"]["variants"]
    assert mf.CARDS["L01v"]["output"] == "groups"
    json.dumps(mf.CONFIG)
    json.dumps(mf.CARDS)
    assert lags.load("L02") is mf.run_l02 and lags.load("L03") is mf.run_l03
    assert lags.load("L01v") is mf.run_l01v
    assert lags.card("L02") is mf.CARDS["L02"]
    assert mf.check_m2_yaml()["max_px_banda_m2a"] == 1200
    kw = mf.fit_kw()
    assert kw["tau_bounds"] == tuple(marea.TAU_BOUNDS) and kw["monotone"] and not kw["fit_alpha"]


def test_l01v_inputs_key_holds_the_scene_meta_and_the_orbit_source(tmp_path):
    """A sidecar fetched later reruns L01v (its sha256 and the orbit source
    enter the inputs key); the other units of the module have none; the
    groups of the date-phase fallback are named phase5-<k>."""
    import json

    from experiments.benchmark import inputs
    cubes = str(tmp_path / "cubes")
    inst = types.SimpleNamespace(site="escalda", orbit_source="phase5",
                                 site2=types.SimpleNamespace(cubes_root=cubes))
    assert mf.external_inputs(inst, {"method": "L02"}) == {}
    a = mf.external_inputs(inst, {"method": "L01v"})
    assert a == {"scene_meta": None, "orbit_source": "phase5"}
    p = inputs.scene_meta_path("escalda", cubes)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"format": inputs.SCENE_META_FORMAT, "dates": {}}), encoding="utf-8")
    b = mf.external_inputs(types.SimpleNamespace(site="escalda", orbit_source="scene_meta",
                                                 site2=inst.site2), {"method": "L01v"})
    assert b["scene_meta"] and b["orbit_source"] == "scene_meta" and b != a
    assert mf.group_name(3, "phase5") == "phase5-3" and mf.group_name(51, "scene_meta") == "orbit51"
    names = sorted(["phase5-4", "phase5-0", "phase5-2"], key=mf._group_order)
    assert names == ["phase5-0", "phase5-2", "phase5-4"]
    assert sorted(["orbit137", "orbit037"], key=mf._group_order) == ["orbit037", "orbit137"]
