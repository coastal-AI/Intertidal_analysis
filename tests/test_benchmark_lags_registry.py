# -*- coding: utf-8 -*-
"""Validation v2, Stage L: the lag-method registry and the shared helpers
(``experiments/benchmark/lags/__init__.py``, ``lags/common.py``; plan
INTERFACES §2-§3)."""
import sys
import types

import numpy as np
import pytest

from experiments.benchmark import lags
from experiments.benchmark.lags import common
from pyintertidal import tide_estimators as te


# ─────────────────────────────────────────────────────────────────────────────
#  the registry
# ─────────────────────────────────────────────────────────────────────────────

def test_every_registry_string_parses():
    for method, target in lags.REGISTRY.items():
        module, sep, func = target.partition(":")
        assert sep == ":" and module.startswith("experiments.benchmark.") and func
        assert lags.module_of(method) == module
    # every product maps to a registered method, and E0 writes its four products
    assert set(lags.PRODUCT_METHOD.values()) <= set(lags.REGISTRY)
    assert {p for p, m in lags.PRODUCT_METHOD.items() if m == "E0"} == {"E0a", "E0b", "L01",
                                                                         "E0g"}
    assert lags.PRODUCT_METHOD["L10"] == "L09" and lags.PRODUCT_METHOD["L15"] == "L14"


def test_registry_covers_the_lag_units_of_the_catalogue():
    from experiments.benchmark import designs
    for m in ("E0", "L01v") + designs.LAG_UNITS:
        assert m in lags.REGISTRY, m
        for p in designs.METHODS[m].products:
            assert lags.PRODUCT_METHOD[p] == m


def test_package_import_imports_no_adapter():
    import subprocess
    code = ("import sys, experiments.benchmark.lags as l; "
            "print(sorted(m for m in sys.modules if m.startswith('experiments.benchmark.lags.') "
            "or m.startswith('experiments.benchmark.methods')))")
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True,
                         text=True, cwd=str(__import__("pathlib").Path(__file__).parents[1]),
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_load_of_a_missing_module_is_a_clear_import_error(monkeypatch):
    monkeypatch.setitem(lags.REGISTRY, "LX", "experiments.benchmark.lags.no_such_module:run")
    with pytest.raises(ImportError, match="module not built: "
                                          "experiments.benchmark.lags.no_such_module"):
        lags.load("LX")
    with pytest.raises(lags.ModuleNotBuilt):
        lags.import_module("LX")
    with pytest.raises(KeyError, match="not in the registry"):
        lags.load("NOPE")


def test_load_finds_the_function_and_reports_a_missing_one(monkeypatch):
    mod = types.ModuleType("fake_lag_mod")
    mod.run_x = lambda inst, unit, opts: []
    mod.CARDS = {"LX": {"id": "LX"}}
    monkeypatch.setitem(sys.modules, "fake_lag_mod", mod)
    monkeypatch.setitem(lags.REGISTRY, "LX", "fake_lag_mod:run_x")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "LX", "LX")
    assert lags.load("LX") is mod.run_x
    assert lags.card("LX") == {"id": "LX"}
    monkeypatch.setitem(lags.REGISTRY, "LY", "fake_lag_mod:run_y")
    with pytest.raises(ImportError, match="no run function 'run_y'"):
        lags.load("LY")


def test_an_import_error_inside_a_module_is_not_disguised(monkeypatch, tmp_path):
    (tmp_path / "fake_broken_lag_mod.py").write_text(
        "import some_dependency_that_does_not_exist" + chr(10), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(lags.REGISTRY, "LZ", "fake_broken_lag_mod:run")
    with pytest.raises(ModuleNotFoundError) as ei:
        lags.import_module("LZ")
    assert not isinstance(ei.value, lags.ModuleNotBuilt)
    assert ei.value.name == "some_dependency_that_does_not_exist"
    sys.modules.pop("fake_broken_lag_mod", None)


def test_check_all_cards_reports_unbuilt_modules_and_bad_cards(monkeypatch):
    mod = types.ModuleType("fake_card_mod")
    good = {k: None for k in common.CARD_KEYS}
    good.update(id="LG1", family="harmonic", kind="lag", tier="1", fidelity="F3",
                output="field", lag_kind="phase")
    bad = dict(good, id="LG2", family="astrology", tier=1)
    mod.CARDS = {"LG1": good, "LG2": bad}
    mod.run = lambda *a: []
    monkeypatch.setitem(sys.modules, "fake_card_mod", mod)
    monkeypatch.setitem(lags.REGISTRY, "LG", "fake_card_mod:run")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "LG1", "LG")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "LG2", "LG")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "LG3", "LG")
    monkeypatch.setitem(lags.REGISTRY, "LQ", "experiments.benchmark.lags.never_built:run")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "LQ1", "LQ")
    res = lags.check_all_cards()
    assert res["LG1"] == []
    assert any("family" in p for p in res["LG2"]) and any("tier" in p for p in res["LG2"])
    assert res["LG3"][0].startswith("no card")
    assert res["LQ1"][0].startswith("module not built")
    cards = lags.all_cards()
    assert "LG1" in cards and "LQ1" not in cards and "LG3" not in cards


def test_module_attrs_defaults(monkeypatch):
    mod = types.ModuleType("fake_attrs_mod")
    mod.run = lambda *a: []
    monkeypatch.setitem(sys.modules, "fake_attrs_mod", mod)
    monkeypatch.setitem(lags.REGISTRY, "LA", "fake_attrs_mod:run")
    a = lags.module_attrs("LA")
    assert a["CONFIG"] == {} and a["CARDS"] == {} and a["CODE_DEPS"] == ()
    assert a["external_inputs"] is None and a["module"] == "fake_attrs_mod"


# ─────────────────────────────────────────────────────────────────────────────
#  common constants and helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_constants():
    from experiments.benchmark import truth
    assert len(common.TAU_SCAN) == 46
    assert common.TAU_SCAN[0] == common.SCAN_LO == -45.0
    assert common.TAU_SCAN[-1] == common.SCAN_HI == 180.0
    assert np.allclose(np.diff(common.TAU_SCAN), 5.0)
    assert common.PERIOD_M2_MIN == pytest.approx(745.236072)
    assert common.OMEGA_M2_PER_MIN == pytest.approx(2 * np.pi / 745.236072)
    assert common.MIN_PER_DEG_M2 == truth.MIN_PER_DEG_M2
    assert common.WRAP_CENTRE_MIN == 67.5 == (common.SCAN_LO + common.SCAN_HI) / 2


@pytest.mark.parametrize("seed", range(6))
def test_scan_vertex_equals_parabolic_argmax_plus_the_edge_flag(seed):
    rng = np.random.default_rng(seed)
    score = -(common.TAU_SCAN - rng.uniform(-60, 200)) ** 2 + rng.normal(0, 50, 46)
    score[rng.integers(0, 46, 3)] = np.nan
    v, edge = common.scan_vertex(score)
    assert v == te._parabolic_argmax(score, common.TAU_SCAN)
    j = int(np.nanargmax(score))
    assert edge == (j in (0, 45))


def test_scan_vertex_edges_and_too_few_points():
    s = np.linspace(0, 1, 46)                    # best at the last point
    assert common.scan_vertex(s) == (180.0, True)
    assert common.scan_vertex(s[::-1]) == (-45.0, True)
    s = np.full(46, np.nan)
    s[:2] = 1.0
    v, e = common.scan_vertex(s)
    assert np.isnan(v) and e is False
    with pytest.raises(ValueError):
        common.scan_vertex(np.zeros(5))


def test_wrap_phase_range_and_identity():
    P = common.PERIOD_M2_MIN
    rng = np.random.default_rng(1)
    x = rng.uniform(-5000, 5000, 1000)
    w = common.wrap_phase(x)
    assert np.all(w >= 67.5 - P / 2 - 1e-9) and np.all(w < 67.5 + P / 2 + 1e-9)
    k = (x - w) / P
    assert np.allclose(k, np.round(k))
    inside = np.array([-200.0, 0.0, 67.5, 300.0])
    assert np.allclose(common.wrap_phase(inside), inside)
    assert np.isnan(common.wrap_phase([np.nan]))[0]


def test_circ_diff_antisymmetry_and_range():
    P = common.PERIOD_M2_MIN
    rng = np.random.default_rng(2)
    a, b = rng.uniform(-2000, 2000, (2, 500))
    d = common.circ_diff(a, b)
    assert np.all(d >= -P / 2) and np.all(d < P / 2)
    assert np.allclose(d, -common.circ_diff(b, a))
    assert common.circ_diff(10.0, 10.0 + P) == pytest.approx(0.0)
    assert common.circ_diff(700.0, 0.0) == pytest.approx(700.0 - P)


def test_circ_median():
    P = common.PERIOD_M2_MIN
    assert common.circ_median([10.0, 20.0, 30.0 + P, np.nan]) == pytest.approx(20.0)
    assert np.isnan(common.circ_median([np.nan, np.nan]))
    assert np.isnan(common.circ_median([]))


def test_broadcast_band_nan_rules():
    tau = np.array([5.0, np.nan, 30.0])
    band_of = np.array([0, 1, 2, -1, 0, 2])
    f = common.broadcast_band(tau, band_of)
    assert f.dtype == np.float32 and f.shape == (6,)
    assert np.array_equal(np.isnan(f), [False, True, False, True, False, False])
    assert f[0] == 5.0 and f[5] == 30.0
    with pytest.raises(ValueError):
        common.broadcast_band(tau, np.array([3]))
    with pytest.raises(ValueError):
        common.broadcast_band(tau, np.array([0.0]))


def _band(K=3):
    return {"tau": np.arange(K, dtype=float), "n_px": np.full(K, 10), "edge": np.zeros(K, bool)}


def test_lag_result_defaults_and_casts():
    r = common.lag_result("L02", kind="band", band=_band(), field=np.zeros(5), n_keep=5, K=3)
    assert r["status"] == "ok" and r["lag_kind"] == "linear" and r["kind"] == "band"
    assert r["field"].dtype == np.float32 and r["band"]["tau"].dtype == np.float64
    assert r["band"]["n_px"].dtype == np.int64 and r["band"]["edge"].dtype == bool
    assert r["n_defined"] == 5 and r["diag"] == {} and r["interval95"] is None
    assert r["edge_share"] == 0.0
    again = common.lag_result(**r, n_keep=5, K=3)            # idempotent
    for k in ("status", "kind", "lag_kind", "n_defined", "edge_share"):
        assert again[k] == r[k]
    assert np.array_equal(again["field"], r["field"])


def test_lag_result_rejects_wrong_lengths_kinds_and_dtypes():
    with pytest.raises(ValueError, match="length"):
        common.lag_result("L08", kind="field", field=np.zeros(4), n_keep=5)
    with pytest.raises(ValueError, match="kind"):
        common.lag_result("L08", kind="pixel", field=np.zeros(5), n_keep=5)
    with pytest.raises(ValueError, match="numeric"):
        common.lag_result("L08", kind="field", field=np.array(["a"] * 5), n_keep=5)
    with pytest.raises(ValueError, match="numeric"):
        common.lag_result("L08", kind="field", field=np.zeros(5, bool), n_keep=5)
    with pytest.raises(ValueError, match="numeric"):
        common.lag_result("L08", kind="field", field=np.zeros(5, complex), n_keep=5)
    with pytest.raises(ValueError, match="integer"):
        common.lag_result("L02", kind="band", band=dict(_band(), n_px=np.zeros(3)), K=3)
    with pytest.raises(ValueError, match="length"):
        common.lag_result("L02", kind="band", band=_band(2), K=3)
    with pytest.raises(ValueError, match="status"):
        common.lag_result("L02", status="maybe")
    with pytest.raises(ValueError, match="lag_kind"):
        common.lag_result("L02", kind="field", field=np.zeros(5), lag_kind="circular")
    with pytest.raises(ValueError, match="payload"):
        common.lag_result("L02", kind="band")
    with pytest.raises(ValueError, match="unknown fields"):
        common.lag_result("L02", kind="field", field=np.zeros(5), feild=1)
    with pytest.raises(ValueError, match="1-D"):
        common.lag_result("L02", kind="field", field=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="interval95"):
        common.lag_result("L01", kind="band", band=_band(), K=3,
                          interval95={"level": "site", "lo": np.zeros(3), "hi": np.zeros(3),
                                      "nominal": "bootstrap30"})
    with pytest.raises(ValueError, match="length"):
        common.lag_result("L01", kind="band", band=_band(), K=3,
                          interval95={"level": "band", "lo": np.zeros(2), "hi": np.zeros(2),
                                      "nominal": "bootstrap30"})


def test_lag_result_not_computable_and_point_and_groups():
    nc = common.lag_result("L03", "not_computable", stage="scenes", kind="band")
    assert nc["reason"] == "unspecified" and nc["n_defined"] == 0 and nc["band"] is None
    pt = common.lag_result("L14", kind="point", point={"st": {"a": 50, "b": 52.5, "model":
                                                                "EOT20", "flags": ["x"]}})
    e = pt["point"]["st"]
    assert e["a"] == 50.0 and e["b"] == 52.5 and np.isnan(e["ab"]) and e["models"] is None
    assert e["flags"] == ["x"] and pt["n_defined"] == 1
    with pytest.raises(ValueError):
        common.lag_result("L14", kind="point", point={"st": {"a": "fifty"}})
    g = common.lag_result("L01v", kind="groups", K=2,
                          groups={"orbit51": {"tau": [1.0, 2.0], "n_scenes": 40},
                                  "orbit94": {"tau": [np.nan, 3.0], "n_scenes": 12,
                                              "status": "not_computable", "reason": "few"}})
    assert g["groups"]["orbit51"]["tau"].dtype == np.float64 and g["n_defined"] == 1
    with pytest.raises(ValueError, match="length"):
        common.lag_result("L01v", kind="groups", K=3, groups={"o": {"tau": [1.0, 2.0]}})


def test_lag_result_edge_share_defaults():
    f = np.array([-45.0, 0.0, 180.0, np.nan], np.float32)
    r = common.lag_result("L12", kind="field", field=f, n_keep=4)
    assert r["edge_share"] == pytest.approx(2 / 3) and r["n_defined"] == 3
    b = {"tau": [1.0, 2.0, np.nan], "n_px": [10, 30, 5], "edge": [True, False, True]}
    r = common.lag_result("L02", kind="band", band=b, K=3, field=np.zeros(3))
    assert r["edge_share"] == pytest.approx(0.25)


def test_elev_result():
    e = common.elev_result("E0b", z=np.zeros(4), sigma=np.ones(4), clear_obs=np.arange(4),
                           tau_px=np.zeros(4), n_keep=4)
    assert e["kind"] == "elev" and e["z"].dtype == np.float32
    assert e["clear_obs"].dtype == np.int16 and e["n_defined"] == 4
    with pytest.raises(ValueError, match="length"):
        common.elev_result("E0b", z=np.zeros(4), sigma=np.ones(3))
    with pytest.raises(ValueError, match="needs z"):
        common.elev_result("E0a")
    with pytest.raises(ValueError, match="integer"):
        common.elev_result("E0a", z=np.zeros(2), clear_obs=np.zeros(2))
    nc = common.elev_result("E0g", "not_computable", stage="no judges")
    assert nc["z"] is None and nc["reason"] == "unspecified"


def _card(**kw):
    c = {k: None for k in common.CARD_KEYS}
    c.update(id="L08", name="GM20", family="harmonic", kind="lag", tier="1", fidelity="F3",
             output="field", lag_kind="phase", references=[], deviations=[], qc={}, tuning=[],
             variants={}, signature={"reads": [], "domain": True}, scan_window=None,
             control_offset_min=0.0, cost={"wh_cell": 0.2, "ram_gb": 1.5})
    c.update(kw)
    return c


def test_check_card():
    assert common.check_card(_card()) == []
    probs = common.check_card({"id": "x"})
    assert len(probs) == len(common.CARD_KEYS) - 1
    assert any("family" in p for p in common.check_card(_card(family="vibes")))
    assert any("tier" in p for p in common.check_card(_card(tier="4")))
    assert any("tier" in p for p in common.check_card(_card(tier=1)))
    assert any("fidelity" in p for p in common.check_card(_card(fidelity="F9")))
    assert any("output" in p for p in common.check_card(_card(output="elev")))
    assert any("lag_kind" in p for p in common.check_card(_card(lag_kind=None)))
    assert any("kind" in p for p in common.check_card(_card(kind="both")))
    assert common.check_card(_card(kind="elev", output="elev", lag_kind=None,
                                   family="elevation")) == []
    assert common.check_card(_card(fidelity="-")) == []
    assert common.check_card("not a dict")
