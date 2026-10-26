# -*- coding: utf-8 -*-
"""L14 / L15 (global tide models at the gauges), on a synthetic instance.

No tide model file is read: ``truth.model_provider`` is replaced by an
analytic tide that reads the boundary tide ``LAG`` minutes late at the
gauge and on time at the boundary member, and ``models.ocean_point`` by a
stub. The instance is a ``types.SimpleNamespace`` with the attributes of
the plan's LagInstance (INTERFACES §4) that the point read-outs use.
"""
import contextlib
import types

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark import judges, levels, settings, truth
from experiments.benchmark.lags import common, models
from experiments.benchmark.product import NotComputable

CONS = ("m2", "s2", "o1", "k1", "n2", "m4")
LAG = 40.0
GAUGE = {"code": "jA", "lat": 51.40, "lon": 3.80, "in_frame": True, "s_g_km": 10.0,
         "location": "jA", "included": True}
MEMBER = {"code": "bnd", "lat": 51.44, "lon": 3.40, "kind": "ioc"}
NS_MIN = 60_000_000_000
_MARKS = {}


# ─────────────────────────────────────────────────────────────────────────────
#  the synthetic instance (shared with the regional, physics and reference tests)
# ─────────────────────────────────────────────────────────────────────────────

def tide(ns):
    """The boundary tide h_B (M2, S2, O1 at their exact periods)."""
    x = np.asarray(ns, np.int64) / 3.6e12
    return (1.7 * np.cos(2 * np.pi * x / 12.4206012 + 0.3)
            + 0.45 * np.cos(2 * np.pi * x / 12.0 + 1.1)
            + 0.2 * np.cos(2 * np.pi * x / 25.8193417 - 0.4))


class LevelView:
    """``levels`` of a LagInstance: h_B(t - tau) at the instance's instants."""

    def __init__(self, t):
        self.t = pd.DatetimeIndex(t).as_unit("ns")

    def at(self, tau):
        return tide(self.t.asi8 - int(round(float(tau) * NS_MIN)))

    def stack(self, taus):
        return np.vstack([self.at(x) for x in taus])


def instants(wid, n_per_year=100):
    """About ``n_per_year`` instants a year at 10:50 UTC +- 2.5 min."""
    w = settings.window(wid)
    out = []
    for y in w.years:
        d = pd.Timestamp(f"{y}-01-02") + pd.to_timedelta(np.round(np.arange(n_per_year)
                                                                  * 365.0 / n_per_year), "D")
        jit = pd.to_timedelta((np.arange(n_per_year) * 37) % 300 - 150, "s")
        out.append((d + pd.Timedelta(hours=10, minutes=50) + jit).values)
    return pd.DatetimeIndex(np.concatenate(out)).as_unit("ns")


def marks_for(wid):
    """The boundary's low-passed marks over the window +- 2 days (cached)."""
    if wid not in _MARKS:
        w = settings.window(wid)
        g = pd.date_range(pd.Timestamp(w.start) - pd.Timedelta(days=2),
                          pd.Timestamp(w.end) + pd.Timedelta(days=2), freq="1min",
                          inclusive="left")
        _MARKS[wid] = levels.build_marks(levels.minutes_since_epoch(g), tide(g.asi8))["marks_lp"]
    return _MARKS[wid]


def series1(wid):
    w = settings.window(wid)
    g = pd.date_range(pd.Timestamp(w.start) - pd.Timedelta(days=2),
                      pd.Timestamp(w.end) + pd.Timedelta(days=2), freq="1min", inclusive="left")
    return levels.minutes_since_epoch(g), tide(g.asi8)


def make_instance(wid="Y2024", judges_=None, members=None, installed=("EOT20", "GOT4.10_nc"),
                  site="escalda", n_per_year=100, **kw):
    t = instants(wid, n_per_year)
    idx10 = pd.date_range("2023-01-01", "2025-12-31 23:50", freq="10min")
    ns = types.SimpleNamespace(
        site=site, window=wid, setting="C1.floor0.ndwi.D1", family="rand", mode="A", N="all",
        rep=0, instance_variant="", t=t, sel=np.arange(len(t)), levels=LevelView(t),
        series10=(idx10, tide(idx10.asi8)), marks_lp=marks_for(wid),
        judges=[dict(GAUGE)] if judges_ is None else judges_,
        boundary_members=[dict(MEMBER)] if members is None else members,
        installed_models=tuple(installed), paths={}, cache={}, seed=0,
        stage=lambda name: contextlib.nullcontext(), progress=lambda n=1: None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class ShiftProvider:
    """A 'model' that reads the boundary tide ``lag`` minutes late, times
    ``gain``, answering in time order (as PyTMDBoundary)."""

    def __init__(self, lag, gain=1.0):
        self.lag, self.gain = float(lag), float(gain)

    def levels(self, times):
        ns = np.sort(pd.DatetimeIndex(times).as_unit("ns").asi8)
        return self.gain * tide(ns - int(round(self.lag * NS_MIN)))


def _provider(model, lat, lon, gains=None):
    if model.startswith("FES2014"):
        raise FileNotFoundError("no FES2014 files")
    gain = (gains or {}).get(model, 1.0)
    at_gauge = abs(lat - GAUGE["lat"]) < 1e-9 and abs(lon - GAUGE["lon"]) < 1e-9
    return ShiftProvider(LAG if at_gauge else 0.0, gain)


@pytest.fixture
def fake_models(monkeypatch):
    """truth.model_provider -> analytic shifted tides; ocean probes stubbed."""
    state = {"gains": {"EOT20": 1.0, "GOT4.10_nc": 1.2, "FES2022": 1.0}}
    monkeypatch.setattr(truth, "model_provider",
                        lambda model, lat, lon: _provider(model, lat, lon, state["gains"]))
    monkeypatch.setattr(models, "ocean_point",
                        lambda label, name, lat, lon, directory=None:
                        {"distance_km": 2.5, "lat": float(lat), "lon": float(lon)})
    monkeypatch.setattr(models, "CONSTITUENTS", CONS)

    def no_gauge(*a, **k):
        raise AssertionError("L14-L18 must never read a gauge series")
    monkeypatch.setattr(judges, "read_series", no_gauge)
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wid", ["Y2024", "W3"])
def test_l14_recovers_a_planted_model_lag_per_window(fake_models, wid):
    inst = make_instance(wid)
    r14, r15 = models.run_l14(inst, {"products": ("L14", "L15")}, {})
    assert r14["product"] == "L14" and r14["status"] == "ok" and r14["kind"] == "point"
    e = r14["point"]["jA"]
    assert e["model"] == "EOT20"
    for k in ("a", "ab", "hwlw", "internal_a"):
        assert abs(e[k] - LAG) <= 1.0, (k, e[k])
    for k in ("b", "internal_b"):
        assert abs(e[k] - LAG) <= 0.5, (k, e[k])
    assert e["dist_km"] == 2.5 and "degenerate" not in e["flags"]
    assert e["gain_ab"] == pytest.approx(1.0, abs=0.02)
    assert r14["n_defined"] == 1 and r14["diag"]["primary"] == "EOT20"
    if wid == "Y2024":
        # (b) is the tau of truth.m2_lag on the same arrays (the truth machinery)
        ctx = models.ReadoutContext(inst)
        m10 = ShiftProvider(LAG).levels(ctx.idx_w)
        ref = truth.m2_lag(ctx.idx_w, ctx.h_w, m10, constituents=CONS)
        assert e["b"] == pytest.approx(ref["tau"], abs=1e-9)


def test_unloadable_model_is_recorded_and_never_substituted(fake_models):
    inst = make_instance(installed=("EOT20", "GOT4.10_nc", "FES2014"))
    r14, _ = models.run_l14(inst, {"products": ("L14", "L15")}, {})
    mm = r14["point"]["jA"]["models"]
    assert set(mm) == {"EOT20", "GOT4.10c", "FES2014"}
    assert mm["FES2014"]["available"] is False
    assert "FileNotFoundError" in mm["FES2014"]["reason"] and np.isnan(mm["FES2014"]["a"])
    assert mm["EOT20"]["available"] and mm["GOT4.10c"]["available"]
    assert abs(mm["GOT4.10c"]["a"] - LAG) <= 1 and mm["GOT4.10c"]["name"] == "GOT4.10_nc"
    assert r14["diag"]["available"] == {"EOT20": True, "GOT4.10c": True, "FES2014": False}


def test_primary_model_rule(fake_models):
    assert models.primary_label(["EOT20", "FES2022"]) == "FES2022"
    assert models.primary_label(["GOT4.10c", "EOT20"]) == "EOT20"
    assert models.primary_label(["GOT4.10c"]) is None
    assert models.installed(("GOT4.10_nc", "EOT20", "unknown")) == [("EOT20", "EOT20"),
                                                                   ("GOT4.10c", "GOT4.10_nc")]
    assert models.installed(("FES2022_extrapolated",)) == [("FES2022", "FES2022_extrapolated")]
    inst = make_instance(installed=("EOT20", "FES2022"))
    r14, _ = models.run_l14(inst, {"products": ("L14", "L15")}, {})
    assert r14["point"]["jA"]["model"] == "FES2022" and r14["diag"]["primary"] == "FES2022"
    # no primary installed: L14 is NOT COMPUTABLE (never another model), L15 still runs
    inst = make_instance(installed=("GOT4.10_nc", "GOT5.6"))
    r14, r15 = models.run_l14(inst, {"products": ("L14", "L15")}, {})
    assert r14["status"] == "not_computable" and r14["stage"] == "primary"
    assert r15["status"] == "ok"


def test_l15_is_the_mean_of_two_fakes(fake_models):
    inst = make_instance(installed=("EOT20", "GOT4.10_nc"))
    r14, r15 = models.run_l14(inst, {"products": ("L14", "L15")}, {})
    e15 = r15["point"]["jA"]
    assert r15["status"] == "ok" and e15["model"] == "ensemble(EOT20+GOT4.10c)"
    assert e15["members"] == ["EOT20", "GOT4.10c"]
    mm = r14["point"]["jA"]["models"]
    assert mm["EOT20"]["gain_ab"] == pytest.approx(1.0, abs=0.01)
    assert mm["GOT4.10c"]["gain_ab"] == pytest.approx(1.2, abs=0.01)
    # the equal-weight mean of a gain-1.0 and a gain-1.2 model is a gain-1.1 model
    assert e15["gain_ab"] == pytest.approx(1.1, abs=0.01)
    assert e15["amp_ratio"] == pytest.approx(1.1, abs=0.01)
    assert abs(e15["a"] - LAG) <= 1 and abs(e15["b"] - LAG) <= 0.5
    # read-outs of the mean series itself
    ctx = models.ReadoutContext(inst)
    mean = 0.5 * (ShiftProvider(LAG, 1.0).levels(ctx.times_gauge)
                  + ShiftProvider(LAG, 1.2).levels(ctx.times_gauge))
    ref = models.readouts(ctx, *ctx.gauge_parts(mean))
    for k in ("a", "ab", "b", "hwlw", "gain_ab"):
        assert e15[k] == pytest.approx(ref[k], abs=1e-9)


def test_fewer_than_two_models_makes_l15_not_computable(fake_models):
    inst = make_instance(installed=("EOT20",))
    r14, r15 = models.run_l14(inst, {"products": ("L14", "L15")}, {})
    assert r14["status"] == "ok"
    assert r15["status"] == "not_computable" and r15["stage"] == "ensemble"
    assert r15["diag"]["missing_input"] == "a second tide model"
    # an unloadable second model does not count
    inst = make_instance(installed=("EOT20", "FES2014"))
    _, r15 = models.run_l14(inst, {"products": ("L14", "L15")}, {})
    assert r15["status"] == "not_computable"


def test_no_judge_is_empty_and_no_model_is_not_computable(fake_models):
    r14, r15 = models.run_l14(make_instance(judges_=[]), {"products": ("L14", "L15")}, {})
    assert r14["status"] == r15["status"] == "empty" and r14["point"] == {}
    out = dict(GAUGE, code="out", in_frame=False)
    r14, _ = models.run_l14(make_instance(judges_=[out]), {"products": ("L14", "L15")}, {})
    assert r14["status"] == "empty"
    with pytest.raises(NotComputable) as ei:
        models.run_l14(make_instance(installed=()), {"products": ("L14", "L15")}, {})
    assert ei.value.stage == "models" and "missing_input" in ei.value.diag


def test_degenerate_when_gauge_and_member_share_a_model_cell(fake_models):
    near = dict(MEMBER, lat=GAUGE["lat"] + 0.01, lon=GAUGE["lon"] + 0.01)
    r14, r15 = models.run_l14(make_instance(members=[near]), {"products": ("L14", "L15")}, {})
    e = r14["point"]["jA"]
    assert "degenerate" in e["flags"]
    assert models.same_or_adjacent(models.model_cell("EOT20", GAUGE["lat"], GAUGE["lon"]),
                                   models.model_cell("EOT20", near["lat"], near["lon"]))
    r14, _ = models.run_l14(make_instance(), {"products": ("L14", "L15")}, {})
    assert "degenerate" not in r14["point"]["jA"]["flags"]
    # the coarse GOT4.10c (0.5 deg) cannot separate the default member: degenerate
    assert "degenerate" in r14["point"]["jA"]["models"]["GOT4.10c"]["flags"]
    # without a boundary member the internal read-out is undefined, never zero
    r14, _ = models.run_l14(make_instance(members=[]), {"products": ("L14", "L15")}, {})
    e = r14["point"]["jA"]
    assert np.isnan(e["internal_a"]) and np.isnan(e["internal_b"]) and abs(e["a"] - LAG) <= 1


def test_cards_config_and_inputs(fake_models, tmp_path):
    import json
    for pid in ("L14", "L15"):
        assert common.check_card(models.CARDS[pid]) == [], pid
        assert models.CARDS[pid]["id"] == pid and models.CARDS[pid]["family"] == "models"
    json.dumps(models.CONFIG)
    inst = make_instance(paths={"truth_dir": str(tmp_path), "tide_dir": str(tmp_path)})
    ext = models.external_inputs(inst, {"method": "L14"})
    assert ext["judges.json"] is None and set(ext) >= {"tide_model:EOT20", "tide_model:GOT4.10c"}
    (tmp_path / "judges.json").write_text("{}", encoding="utf-8")
    assert len(models.external_inputs(inst, {"method": "L14"})["judges.json"]) == 64


def test_unresolved_models_are_flagged_and_carried_onto_the_ensemble():
    f = models.resolution_flag
    assert f("EOT20", 2.0) is None                        # within the 1/8 deg spacing
    assert f("EOT20", 25.0).startswith("unresolved (nearest wet EOT20 cell 25.0 km")
    assert f("GOT4.10c", float("inf")).startswith("unresolved (no wet GOT4.10c cell")
    assert f("GOT4.10c", float("nan"), "probe failed") == ("resolution unknown (no "
                                                            "nearest-ocean probe)")
    assert f("GOT4.10c", float("nan")) is None
    # the ensemble entry carries its members' flags (the estimate is unchanged)
    per = {"EOT20": {"flags": [], "dist_km": 2.0},
           "GOT4.10c": {"flags": ["unresolved (no wet GOT4.10c cell within the 0.5 deg probe)",
                                  "degeneracy unknown (no nearest-ocean probe)"],
                        "dist_km": float("inf")}}

    class Ctx:
        def gauge_parts(self, v):
            return v, v, v

        def boundary_parts(self, v):
            return v, v

    calls = {}

    def fake_readouts(ctx, m_t, m10, m_hw, mb_lag, mb10, ref_key=None):
        calls["m"] = m_t
        return {"a": 1.0, "ab": 1.0, "b": 1.0, "hwlw": 1.0, "internal_a": 1.0,
                "internal_b": 1.0, "flags": []}

    real = models.readouts
    models.readouts = fake_readouts
    try:
        e = models._ensemble_entry(Ctx(), {"EOT20": np.ones(3), "GOT4.10c": 3 * np.ones(3)},
                                   {}, per)
    finally:
        models.readouts = real
    assert "members unresolved: GOT4.10c" in e["flags"]
    assert any(x.startswith("members with unknown degeneracy") for x in e["flags"])
    assert np.allclose(calls["m"], 2.0) and e["a"] == 1.0
