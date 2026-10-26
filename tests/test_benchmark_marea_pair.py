"""Validation v2: the E0 unit (methods/elev_marea_pair.py).

No delay (E0a) and MAREA (E0b) equal marea.invert_series on the full
matrices, per tau group, when the frozen slice and invert_series' chunk are
shrunk (slice 8, chunk 4): the slices align with the chunks, so the bits
are production's. The L01 product shares E0b's lag stage; the 'gaugelag'
variant (E0g) inverts with the judges' band lags (lags.references stubbed);
'nodelay' is V3; production_compat takes the pixel cloud rule as the clear
mask. Tiny synthetic instances.
"""
import os
import sys
import types

import numpy as np
import pytest

from experiments.benchmark import config
from experiments.benchmark.lags import common
from experiments.benchmark.lags import marea_family as mf
from experiments.benchmark.methods import elev_marea_pair as emp
from experiments.benchmark.product import NotComputable
from pyintertidal import marea
from tests.test_benchmark_lags_marea import bit_equal, make_data, make_inst

UNIT = {"method": "E0", "variant": "", "products": ("E0a", "E0b", "L01"), "design": "lag",
        "unit_id": "u", "site": "escalda", "window": "W3", "setting": "C1.floor0.ndwi.D1",
        "family": "rand", "mode": "A", "N": "all", "rep": 0}
OPTS = {"n_boot": 0, "refits": 0, "seed_offset": 0, "threads": 1, "boot_threads": 1,
        "smoke": False}


def _small(mp):
    mp.setattr(emp, "SLICE", 8)
    mp.setattr(emp, "CHUNK", 4)


def reference_inversion(d, tau_px, Y=None, C=None, chunk=4):
    """reconstruct's loop on full matrices: one invert_series call per
    unique tau over that tau's columns."""
    Y = np.nan_to_num(d["X"], nan=0.0).astype(np.float64) if Y is None else Y
    C = d["V"].astype(np.float64) if C is None else C
    h0 = d["lev"].at(0)
    lo, hi = float(h0.min()), float(h0.max())
    z = np.full(d["n"], np.nan)
    sg = np.full(d["n"], np.nan)
    for tv in np.unique(tau_px):
        cols = np.where(tau_px == tv)[0]
        h = h0 if not tv else d["lev"].at(int(tv))
        z[cols], sg[cols] = marea.invert_series(Y[:, cols], C[:, cols], h, lo, hi, chunk=chunk,
                                                min_bracket=3)
    return z, sg


@pytest.fixture(scope="module")
def e0():
    with pytest.MonkeyPatch.context() as mp:
        _small(mp)
        # bands of >= 50 px (m2a skips smaller bands), sizes not multiples of 4
        d = make_data(T=64, npx=(53, 61, 57), taus=(0.0, 45.0, 95.0), seed=31, n_nan_s=3)
        inst = make_inst(d)
        res = emp.run_unit(inst, UNIT, OPTS)
        st = inst.cache["marea_family.lag_stage"][(0, 0, False)]
    return types.SimpleNamespace(d=d, inst=inst, res=res, st=st)


def test_e0a_equals_one_invert_series_call(e0):
    d = e0.d
    a, b, lag = e0.res
    assert [r["product"] for r in e0.res] == ["E0a", "E0b", "L01"]
    h0 = d["lev"].at(0)
    z, sg = marea.invert_series(np.nan_to_num(d["X"]).astype(np.float64),
                                d["V"].astype(np.float64), h0, float(h0.min()), float(h0.max()),
                                chunk=4, min_bracket=3)
    assert a["kind"] == "elev" and a["z"].dtype == np.float32
    assert np.array_equal(a["z"], z.astype(np.float32), equal_nan=True)
    assert bit_equal(a["z"], z.astype(np.float32)) and bit_equal(a["sigma"], sg.astype(np.float32))
    assert np.isfinite(a["z"]).sum() > 0.5 * d["n"]
    assert np.array_equal(a["clear_obs"], d["V"].sum(axis=0).astype(np.int16))
    assert np.all(a["tau_px"] == 0) and a["diag"]["min_bracket"] == 3
    # 171 columns in slices of 8 with chunk 4 (171 is not a multiple of 4)
    assert d["n"] % 4 and d["n"] > 8


def test_e0b_equals_reconstructs_loop_per_tau_group(e0):
    d, st = e0.d, e0.st
    a, b, lag = e0.res
    tau_px = st["tau_px"].astype(np.float64)
    groups = np.unique(tau_px)
    assert len(groups) >= 3                         # several groups, sizes not multiples of 4
    z, sg = reference_inversion(d, tau_px)
    assert bit_equal(b["z"], z.astype(np.float32)) and bit_equal(b["sigma"], sg.astype(np.float32))
    assert np.array_equal(b["tau_px"], st["tau_px"])
    assert sum(b["diag"]["groups"].values()) == d["n"]
    # L01 shares the stage: band tau = tau_used, field = the applied lag
    assert lag["product"] == "L01" and lag["kind"] == "band"
    assert bit_equal(lag["band"]["tau"], st["tau_used"]) and lag["interval95"] is None
    assert np.array_equal(lag["field"], st["tau_px"])
    # pixels at tau 0 are inverted in their own group: not E0a's partition
    assert np.isfinite(b["z"]).sum() > 0.5 * d["n"]


def test_slice_must_be_a_multiple_of_the_chunk(monkeypatch):
    d = make_data(T=50, npx=(12, 12), seed=4, n_nan_s=0)
    monkeypatch.setattr(emp, "SLICE", 6)
    monkeypatch.setattr(emp, "CHUNK", 4)
    with pytest.raises(ValueError, match="multiple"):
        emp.run_unit(make_inst(d), dict(UNIT, products=("E0a",)), OPTS)


def test_boot30_and_seed_variants_reach_the_lag_stage(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(mf, "BOOT_MAX_PX", 10)
    d = make_data(T=60, npx=(20, 25), taus=(0.0, 30.0), seed=6, n_nan_s=0)
    inst = make_inst(d)
    a, b, lag = emp.run_unit(inst, dict(UNIT, variant="boot30"), dict(OPTS, n_boot=2))
    assert lag["interval95"]["nominal"] == "bootstrap2" and lag["diag"]["boot"].shape == (2, 2)
    inst2 = make_inst(d)
    out = emp.run_unit(inst2, dict(UNIT, variant="seed3", products=("E0b", "L01")),
                       {"n_boot": 0})
    assert [r["product"] for r in out] == ["E0b", "L01"]
    assert (0, 3, False) in inst2.cache["marea_family.lag_stage"]
    assert out[1]["diag"]["seed_offset"] == 3


def test_floor_no_intertidal_and_no_anchor(monkeypatch):
    _small(monkeypatch)
    d = make_data(T=45, npx=(15, 15), seed=7, n_nan_s=0)
    short = make_inst(d, sel=np.arange(39))
    short.h0 = short.h0[:39]
    with pytest.raises(NotComputable) as e:
        emp.run_unit(short, UNIT, OPTS)
    assert e.value.stage == "record_floor"
    with pytest.raises(NotComputable) as e:
        emp.run_unit(make_inst(d, status="no_intertidal"), UNIT, OPTS)
    assert e.value.stage == "no_intertidal"
    a, b, lag = emp.run_unit(make_inst(d, status="no_anchor"), UNIT, OPTS)
    assert bit_equal(a["z"], b["z"]) and np.all(b["tau_px"] == 0)
    assert lag["band"]["tau"].tolist() == [0.0]


def test_nodelay_is_v3_and_unknown_products_refused():
    d = make_data(T=45, npx=(15, 15), seed=7, n_nan_s=0)
    with pytest.raises(NotComputable) as e:
        emp.run_unit(make_inst(d), dict(UNIT, variant="nodelay", products=("E0a",)), OPTS)
    assert "V3" in e.value.reason
    with pytest.raises(ValueError):
        emp.run_unit(make_inst(d), dict(UNIT, products=("E0a", "E1")), OPTS)


class FakeRefs(types.ModuleType):
    def __init__(self, answer):
        super().__init__("experiments.benchmark.lags.references")
        self.answer = answer
        self.calls = []

    def gauge_band_taus(self, inst, window, band_s_km=None):
        self.calls.append((window, np.asarray(band_s_km, float).copy()))
        return self.answer


def _stub_refs(monkeypatch, answer):
    import experiments.benchmark.lags as lags_pkg
    fake = FakeRefs(answer)
    monkeypatch.setitem(sys.modules, "experiments.benchmark.lags.references", fake)
    monkeypatch.setattr(lags_pkg, "references", fake, raising=False)
    return fake


def test_gaugelag_writes_e0g_with_the_gauge_band_lags(monkeypatch):
    _small(monkeypatch)
    d = make_data(T=50, npx=(18, 22, 20), seed=12, n_nan_s=2)
    fake = _stub_refs(monkeypatch, {"status": "ok", "reason": "", "tau_band": [0.0, 30.0, 70.0],
                                    "codes": ["a", "b"]})
    inst = make_inst(d)
    unit = dict(UNIT, variant="gaugelag", products=("E0g",))
    (g,) = emp.run_unit(inst, unit, OPTS)
    assert g["product"] == "E0g" and g["kind"] == "elev"
    window, band_s = fake.calls[0]
    assert window == "W3"
    assert bit_equal(band_s, mf.band_mean_s(d["s_km"], d["band_of"], 3))
    tau_px = mf.applied_lag(d["centers"], np.array([0.0, 30.0, 70.0]), d["s_km"])
    assert np.array_equal(g["tau_px"].astype(np.float64), tau_px)
    z, sg = reference_inversion(d, tau_px)
    assert bit_equal(g["z"], z.astype(np.float32))
    assert g["diag"]["gauge"]["codes"] == ["a", "b"]
    assert "marea_family.lag_stage" not in inst.cache          # no archive lag
    _stub_refs(monkeypatch, {"status": "not available", "reason": "no included judge"})
    with pytest.raises(NotComputable) as e:
        emp.run_unit(make_inst(d), unit, OPTS)
    assert e.value.stage == "gauge_lags" and "judge" in e.value.reason
    _stub_refs(monkeypatch, {"status": "ok", "tau_band": [0.0, np.nan, 3.0]})
    with pytest.raises(NotComputable):
        emp.run_unit(make_inst(d), unit, OPTS)
    with pytest.raises(NotComputable) as e:
        emp.run_unit(make_inst(d, status="no_anchor"), unit, OPTS)
    assert e.value.stage == "no_anchor"


class FakePkg:
    """The package's stored SCL at the instance's pixels (production_compat)."""

    def __init__(self, scl):
        self.scl = scl
        self.reads = 0

    def read(self, names, sel=None, pos=None):
        self.reads += 1
        return {"SCL": self.scl[np.asarray(sel)][:, np.asarray(pos)]}


def test_production_compat_takes_the_cloud_rule_as_clear(monkeypatch):
    _small(monkeypatch)
    d = make_data(T=55, npx=(20, 20), seed=14, n_nan_s=0)
    rng = np.random.default_rng(0)
    scl = np.where(d["V"], np.uint8(5), np.uint8(9))
    # clear pixels whose NDWI is NaN (B03 + B08 == 0): production keeps them as dry
    nan_at = (rng.random(d["X"].shape) < 0.05) & d["V"]
    d["X"] = np.where(nan_at, np.float32(np.nan), d["X"])
    d["V"] = d["V"] & np.isfinite(d["X"])
    d["W"] = d["V"] & (np.nan_to_num(d["X"]) > 0)
    inst = make_inst(d, setting="C2.raw.ndwi.D1", pkg=FakePkg(scl))
    out = emp.compute(inst, production_compat=True, products=("E0a",))
    C = (scl == 5).astype(np.float64)
    z, _ = marea.invert_series(np.nan_to_num(d["X"]).astype(np.float64), C, d["lev"].at(0),
                               float(d["lev"].at(0).min()), float(d["lev"].at(0).max()), chunk=4,
                               min_bracket=3)
    assert bit_equal(out["z_a"], z) and inst.pkg.reads > 0
    assert np.array_equal(out["clear_obs"], (scl == 5).sum(axis=0))
    normal = emp.compute(make_inst(d), products=("E0a",))
    assert not bit_equal(normal["z_a"], out["z_a"])           # the clear mask matters
    # reconstruct's wet = C & (Y > 0) in the lag stage too
    arr = mf.read_columns(inst, np.arange(d["n"]), ("wet", "clear"), production_compat=True)
    assert np.array_equal(arr["clear"], scl == 5)
    assert np.array_equal(arr["wet"], (scl == 5) & (np.nan_to_num(d["X"]) > 0))
    for bad in ("C3.raw.ndwi.D1", "C1.floor0.ndwi.D1", "C2.raw.mndwi.D1"):
        with pytest.raises(ValueError):
            mf.read_columns(make_inst(d, setting=bad, pkg=FakePkg(scl)), np.arange(3),
                            production_compat=True)


def test_cards_registry_config_and_external_inputs(tmp_path):
    import json

    from experiments.benchmark import lags
    for pid in ("E0a", "E0b", "L01", "E0g"):
        card = emp.CARDS[pid]
        assert common.check_card(card) == [], pid
        assert card["id"] == pid
    assert emp.CARDS["L01"] is mf.CARDS["L01"]
    assert lags.load("E0") is emp.run_unit and lags.card("E0b") is emp.CARDS["E0b"]
    json.dumps(emp.CONFIG)
    json.dumps(emp.CARDS)
    assert emp.CONFIG["invert_series"]["chunk"] == 4000 and emp.SLICE == 200000
    assert emp.SLICE % emp.CHUNK == 0 and emp.MIN_BRACKET == marea.QUALITY_MIN_BRACKET == 3
    assert "experiments/benchmark/lags/marea_family.py" in emp.CODE_DEPS
    for p in emp.CODE_DEPS:
        assert os.path.exists(os.path.join(config.ROOT, p)), p
    inst = types.SimpleNamespace(paths={"truth_dir": str(tmp_path)})
    assert emp.external_inputs(inst, {"variant": "boot30"}) == {}
    (tmp_path / "truth_lag_v2.json").write_text("{}", encoding="utf-8")
    ext = emp.external_inputs(inst, {"variant": "gaugelag"})
    assert len(ext["truth/truth_lag_v2.json"]) == 64 and ext["truth/judges.json"] is None
    assert emp.REFERENCES_PY in ext
