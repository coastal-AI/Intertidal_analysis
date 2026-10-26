# -*- coding: utf-8 -*-
"""L20 (placeholder; the scorer computes it) and the E0g gauge band lags."""
import json
import types

import numpy as np
import pytest

from experiments.benchmark import judges, levels, truth
from experiments.benchmark.lags import common, references
from tests.test_benchmark_lags_models import make_instance

#: (code, s_g in judges.json, {window: (included, tau_a, tau_b)})
STATIONS = [
    ("gatvanborssele", 14.6, {"W3": (True, 10.0, 9.8), "Y2024": (True, 12.0, 11.0)}),
    ("terneuzen", 21.7, {"W3": (True, 18.0, 18.5), "Y2024": (False, None, None)}),
    ("terneuzen.westsluis.buiten", 22.1, {"W3": (True, 20.0, 19.0)}),
    ("borssele", 12.0, {"W3": (False, None, None)}),
    ("inner", 28.0, {"W3": (True, 16.0, 25.0)}),          # a decrease: the monotone fit
]


def write_truth(tmp_path):
    st, tab = [], []
    for code, s, wins in STATIONS:
        w = {}
        for wid, (inc, ta, tb) in wins.items():
            e = {"inclusion": {"included": inc, "exclusion": None if inc else ["coverage"],
                               "window": wid}}
            if inc:
                e.update(a={"tau": ta}, ab={"tau": ta + 1.0}, b={"tau": tb},
                         hw_lw={"mean_min": ta + 2.0})
            w[wid] = e
        st.append({"code": code, "location": judges.location_of(code), "s_g_km": s + 0.5,
                   "windows": w})
        tab.append({"code": code, "s_g_km": s, "location": judges.location_of(code)})
    (tmp_path / "truth_lag_v2.json").write_text(json.dumps(
        {"format": truth.TRUTH_LAG_V2_FORMAT, "stations": st}), encoding="utf-8")
    (tmp_path / "judges.json").write_text(json.dumps(
        {"format": judges.JUDGES_FORMAT, "stations": tab}), encoding="utf-8")


def domain():
    s = np.linspace(0.1, 32.0, 400)
    return types.SimpleNamespace(keep=np.arange(400), s_km=s,
                                 band_of=np.floor(s / 3.0).astype(np.int64), K=11)


def test_l20_is_a_placeholder_computed_by_the_scorer():
    (r,) = references.run_l20(make_instance(), {"products": ("L20",)}, {})
    assert r["product"] == "L20" and r["status"] == "ok" and r["kind"] == "point"
    assert r["point"] == {} and "scorer" in r["diag"]["computed_by"]
    assert common.check_card(references.CARDS["L20"]) == []
    assert references.CARDS["L20"]["family"] == "reference"


def test_gauge_band_taus_equals_levels_gauge_band_lags(tmp_path):
    write_truth(tmp_path)
    inst = make_instance("W3", paths={"truth_dir": str(tmp_path)})
    inst.domain = domain()
    res = references.gauge_band_taus(inst, "W3")
    lags = {"gatvanborssele": 10.0, "terneuzen": 18.0, "terneuzen.westsluis.buiten": 20.0,
            "inner": 16.0}
    s_g = {"gatvanborssele": 14.6, "terneuzen": 21.7, "terneuzen.westsluis.buiten": 22.1,
           "inner": 28.0}                          # judges.json, not the truth doc's s
    band_s = references.band_mean_s(inst.domain)
    ref = levels.gauge_band_lags(lags, s_g, band_s)
    np.testing.assert_array_equal(res["tau_band"], ref["tau_band"])
    assert res["status"] == "ok" and res["lags_min"] == lags and res["s_g_km"] == s_g
    # the Terneuzen pair is one knot; the excluded judge is dropped
    assert res["codes"] == ["gatvanborssele", "terneuzen+terneuzen.westsluis.buiten", "inner"]
    assert res["knots_s_km"] == pytest.approx([14.6, 21.9, 28.0])
    assert "borssele" in res["excluded"] and "borssele" not in res["lags_min"]
    assert res["monotone_adjusted"]                  # 19 -> 16 pooled by the isotonic fit
    assert res["tau_band"][0] == pytest.approx(band_s[0] / 14.6 * 10.0)
    assert len(res["source_shas"]["truth_lag_v2.json"]) == 64
    assert len(res["source_shas"]["judges.json"]) == 64


def test_gauge_band_taus_per_window_and_truth_key(tmp_path):
    write_truth(tmp_path)
    inst = make_instance("Y2024", paths={"truth_dir": str(tmp_path)})
    inst.domain = domain()
    res = references.gauge_band_taus(inst, "Y2024")
    assert res["lags_min"] == {"gatvanborssele": 12.0}
    assert set(res["excluded"]) == {"terneuzen", "terneuzen.westsluis.buiten", "borssele",
                                    "inner"}
    rb = references.gauge_band_taus(inst, "W3", truth_key="b")
    assert rb["lags_min"]["inner"] == 25.0 and rb["truth_key"] == "b"
    band_s = np.array([1.5, 4.5, 40.0, np.nan])
    r = references.gauge_band_taus(inst, "W3", band_s_km=band_s)
    assert np.isnan(r["tau_band"][3]) and r["tau_band"][2] == pytest.approx(r["knots_tau_min"][-1])
    with pytest.raises(ValueError):
        references.gauge_band_taus(inst, "W3", truth_key="nope")
    # no included judge: not available (never zero lags)
    r = references.gauge_band_taus(inst, "Y2023")
    assert r["status"] == "not available" and np.isnan(r["tau_band"]).all()


def test_gauge_band_taus_without_a_truth_is_not_available(tmp_path):
    inst = make_instance("W3", paths={"truth_dir": str(tmp_path / "none")})
    inst.domain = domain()
    r = references.gauge_band_taus(inst, "W3")
    assert r["status"] == "not available" and "truth" in r["reason"]
    assert np.isnan(r["tau_band"]).all() and len(r["tau_band"]) == inst.domain.K
    assert r["missing_input"].endswith("truth_lag_v2.json")
    assert r["source_shas"] == {"truth_lag_v2.json": None, "judges.json": None}
