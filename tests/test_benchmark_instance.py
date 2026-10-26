# -*- coding: utf-8 -*-
"""Validation v2, Stage L: the lag instance (``experiments/benchmark/instance.py``;
plan INTERFACES §4, §6)."""
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark import domain, levels, settings
from experiments.benchmark import instance as I
from experiments.benchmark.obs import Observation


# ─────────────────────────────────────────────────────────────────────────────
#  fakes
# ─────────────────────────────────────────────────────────────────────────────

class FakePkg:
    """The record part of a V2Package: dates over ``years``, windows, the
    partial-year guard and a deterministic scene rule."""

    def __init__(self, years=(2023, 2024, 2025), n_per_year=40, site="escalda"):
        d = []
        for y in years:
            d += list(pd.date_range(f"{y}-01-03", periods=n_per_year,
                                    freq=f"{max(1, 360 // n_per_year)}D"))
        self.dates = np.array([str(x.date()) for x in d])
        self.years = list(years)
        self.site = site
        self.T = len(d)
        self.observed = []

    def window(self, wid):
        w = settings.window(wid)
        t = pd.to_datetime(self.dates)
        return np.flatnonzero((t >= pd.Timestamp(w.start)) & (t < pd.Timestamp(w.end)))

    def window_years(self, wid):
        w = settings.window(wid)
        miss = [y for y in w.years if y not in self.years]
        if miss:
            raise RuntimeError(f"window {wid} needs the shard years {miss}")
        return list(w.years)

    def rule_subset(self, rule, wid):
        return self.window(wid)[::3]

    def observe(self, setting, sel, pos=None):
        self.observed.append((setting, np.asarray(sel), np.asarray(pos)))
        X = (np.asarray(sel)[:, None] * 1000 + np.asarray(pos)[None, :]).astype(np.float32)
        return Observation(X, X / 2, X > 0, (X.astype(np.int64) % 2) == 0)


def _row(**kw):
    r = {"design": "curves-k1", "site": "escalda", "window": "W3",
         "setting_id": settings.P0.id, "method": "L02", "variant": "", "family": "rand",
         "mode": "A", "N": "all", "rep": 0, "status": "planned", "products": "L02",
         "unit_id": "u", "est_hours": "0.01", "est_ram_gb": "1.0"}
    r.update(kw)
    return r


# ─────────────────────────────────────────────────────────────────────────────
#  rows, keys, names
# ─────────────────────────────────────────────────────────────────────────────

def test_norm_row_and_instance_key():
    r = I.norm_row(_row(N="45", rep="3", setting_id="C1.floor0.ndwi.D1@LT",
                        variant="bnd-eot20", products="E0a E0b L01"))
    assert r["N"] == 45 and r["rep"] == 3 and r["setting"] == "C1.floor0.ndwi.D1"
    assert r["setting_id"] == "C1.floor0.ndwi.D1@LT" and r["products"] == ("E0a", "E0b", "L01")
    assert r["instance_variant"] == "bnd-eot20" and r["est_hours"] == 0.01
    k = I.instance_key(r)
    assert k == ("escalda", "W3", "C1.floor0.ndwi.D1", "rand", "A", "45", 3, "bnd-eot20")
    assert I.instance_key(_row(variant="boot30"))[-1] == ""
    assert I.instance_key(_row(variant="boot30")) == I.instance_key(_row(variant="seed2"))
    assert len(I.instance_id(k)) == 16 and I.instance_id(k) == I.instance_id(list(k))
    assert I.instance_seed(k) == I.instance_seed(k) and 0 <= I.instance_seed(k) < 2 ** 32
    assert I.parse_N("rule:C2") == "rule:C2" and I.parse_N(20) == 20
    with pytest.raises(ValueError):
        I.parse_N("many")
    with pytest.raises(ValueError):
        I.norm_row(_row(mode="Z"))


def test_leaf_names_and_curve_R():
    assert I.leaf_name("all", 0) == "Nall_r00"
    assert I.leaf_name(45, 3) == "N045_r03"
    assert I.leaf_name("rule:C2", 1) == "NruleC2_r01"
    assert I.curve_R("curves-k1", "W3", 45) == 5
    assert I.curve_R("curves-k1", "W3", 330) == 2
    with pytest.raises(ValueError):
        I.curve_R("curves-k1", "W3", 46)
    with pytest.raises(ValueError, match="disagree"):
        I.curve_R("other", "W3", 180)                 # A-mode: K1 and K3 R=2, K7 R=5
    assert I.curve_R("other", "W3", 180, mode="F") == 3


# ─────────────────────────────────────────────────────────────────────────────
#  the scenes of an instance
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N,rep", [(20, 1), (45, 4), (90, 2), (330, 1)])
def test_selection_equals_rand_subsets(N, rep):
    pkg = FakePkg(n_per_year=150)
    win = pkg.window("W3")
    R = I.curve_R("curves-k1", "W3", N)
    _, subs = settings.rand_subsets("escalda", "W3", len(win), N, R)
    got = I.instance_selection(pkg, _row(N=str(N), rep=rep))
    assert np.array_equal(got, win[subs[rep]]) and got.dtype == np.int64
    assert np.all(np.diff(got) > 0) and len(got) == N


def test_selection_equals_matched_subsets_halves_and_rules():
    pkg = FakePkg()
    win = pkg.window("W3")
    n_rule = len(pkg.rule_subset("C2", "W3"))
    for d in range(3):
        got = I.instance_selection(pkg, _row(family="matched-C2", N="rule:C2", rep=d))
        want = settings.matched_subsets("escalda", "W3", win, n_rule, draws=3)[d]
        assert np.array_equal(got, want) and len(got) == n_rule
    for h in (0, 1):
        assert np.array_equal(I.instance_selection(pkg, _row(family="half", rep=h)),
                              win[h::2])
    assert np.array_equal(I.instance_selection(pkg, _row()), win)
    rule = I.instance_selection(pkg, _row(setting_id="C2.floor0.ndwi.D1", family="rule"))
    assert np.array_equal(rule, pkg.rule_subset("C2", "W3"))
    y = I.instance_selection(pkg, _row(window="Y2024"))
    assert np.array_equal(y, pkg.window("Y2024"))
    with pytest.raises(ValueError):
        I.instance_selection(pkg, _row(family="half", rep=2))


def test_partial_year_guard_raises_instance_unavailable():
    pkg = FakePkg(years=(2023,))
    assert len(pkg.window("W3")) == 40                # window() alone is silently partial
    with pytest.raises(I.InstanceUnavailable, match=r"\[2024, 2025\]"):
        I.instance_selection(pkg, _row())
    assert len(I.instance_selection(pkg, _row(window="Y2023"))) == 40
    with pytest.raises(I.InstanceUnavailable):
        I.instance_selection(pkg, _row(window="Y2024"))


def test_unknown_families_are_unavailable_not_silent():
    pkg = FakePkg()
    with pytest.raises(I.InstanceUnavailable, match="summer"):
        I.instance_selection(pkg, _row(family="summer"))


# ─────────────────────────────────────────────────────────────────────────────
#  levels and observations
# ─────────────────────────────────────────────────────────────────────────────

class LinTide:
    name = "lin"

    def levels(self, idx):
        m = levels.minutes_since_epoch(pd.DatetimeIndex(idx))
        return np.sin(m / 120.0)


def _table(T=8, provider=None):
    t = pd.date_range("2024-01-01 10:00", periods=T, freq="17h")
    shifts = np.arange(-5, 11)
    p = provider or LinTide()
    H = np.vstack([p.levels(t - pd.Timedelta(minutes=int(k))) for k in shifts])
    return levels.LevelTable(shifts, H, t, provider=p), t


def test_level_view_integer_rows_and_interpolation():
    tab, t = _table()
    sel = np.array([1, 3, 4, 7])
    lv = I.LevelView(tab, sel)
    for k in (-5, 0, 3, 10):
        assert np.array_equal(lv.at(k), tab.at(k, sel))
    assert np.array_equal(lv.at(0.0), tab.at(0, sel))
    w = 0.25
    assert np.allclose(lv.at(2.25), (1 - w) * tab.at(2, sel) + w * tab.at(3, sel), atol=0,
                       rtol=1e-15)
    assert np.array_equal(lv.at(-4.5), 0.5 * tab.at(-5, sel) + 0.5 * tab.at(-4, sel))
    st = lv.stack([0, 2.25, 3])
    assert st.shape == (3, 4) and np.array_equal(st[1], lv.at(2.25))
    live = lv.at(25)                                   # beyond the table: a live call
    want = LinTide().levels(t[sel] - pd.Timedelta(minutes=25))
    assert np.array_equal(live, want) and tab.live_calls[-1]["tau"] == 25
    with pytest.raises(ValueError):
        lv.at(np.nan)


def test_level_view_bank_and_series():
    tab, t = _table()
    sel = np.array([0, 2, 5])
    lv = I.LevelView(tab, sel, series1=lambda: (np.array([0.0, 10.0, 20.0]),
                                                 np.array([0.0, 1.0, 3.0])))
    from pyintertidal import marea
    bank = lv.bank(taus=[0, 5, 10])
    ref = tab.bank(sel, [0, 5, 10])
    assert np.array_equal(bank.H, ref.H) and np.array_equal(bank.taus, ref.taus)
    assert np.allclose(lv.series_at([5.0, 15.0]), [0.5, 2.0])
    assert np.all(np.isnan(lv.series_at([-1.0, 25.0])))
    assert marea.BANK_TAUS is not None


def test_obs_view_read_and_blocks():
    pkg = FakePkg()
    sel = np.array([2, 5, 9])
    n = 23
    dom = types.SimpleNamespace(keep=np.arange(n) * 3, pos=np.arange(n) + 100)
    ov = I.ObsView(pkg, settings.P0.id, sel, dom)
    assert (ov.n, ov.T) == (n, 3)
    o = ov.read([1, 4, 5])
    ref = pkg.observe(settings.P0.id, sel, pos=dom.pos[[1, 4, 5]])
    for a, b in zip(o, ref):
        assert np.array_equal(a, b)
    full = ov.read()
    seen = []
    for cols, ob in ov.iter_blocks(block=7):
        seen.append(cols)
        assert np.array_equal(ob.X, full.X[:, cols])
    assert np.array_equal(np.concatenate(seen), np.arange(n))
    with pytest.raises(ValueError, match="sorted"):
        ov.read([4, 1])
    with pytest.raises(ValueError, match="MAX_CELLS"):
        I.ObsView(pkg, settings.P0.id, sel, dom, max_cells=10).read()


def test_orbits_phase5_fallback_and_scene_meta(tmp_path):
    """Without the sidecar the 5-day phase of the date groups the scenes: S2A
    and S2B fly each relative orbit 5 days apart, so one residue is one
    relative orbit of the constellation (the 10-day phase split each orbit
    into its two satellites)."""
    from experiments.benchmark import inputs
    dates = ["2024-01-03", "2024-01-08", "2024-01-13"]
    s2 = types.SimpleNamespace(site="escalda", cubes_root=str(tmp_path / "cubes"))
    orb, dbl, plat, src = I.scene_orbits(s2, dates)
    assert src == "phase5" and I.ORBIT_PHASE_DAYS == 5
    assert list(orb) == [pd.Timestamp(d).date().toordinal() % 5 for d in dates]
    assert len(set(orb)) == 1                              # one track, both satellites
    assert not dbl.any() and list(plat) == ["", "", ""]
    meta = {"format": inputs.SCENE_META_FORMAT, "site": "escalda", "query": {},
            "dates": {"2024-01-03": {"relative_orbits": [51], "platforms": ["sentinel-2a"],
                                     "double_overpass": False},
                      "2024-01-08": {"relative_orbits": [8, 51], "platforms": ["sentinel-2b"],
                                     "double_overpass": True}}}
    p = inputs.scene_meta_path("escalda", s2.cubes_root)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(meta), encoding="utf-8")
    orb, dbl, plat, src = I.scene_orbits(s2, dates)
    assert src == "scene_meta" and list(orb) == [51, 8, -1]
    assert list(dbl) == [False, True, False] and list(plat) == ["sentinel-2a", "sentinel-2b", ""]


# ─────────────────────────────────────────────────────────────────────────────
#  the small v2 package (tests/benchmark_zarr_site.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def zsite(tmp_path_factory):
    from tests import benchmark_zarr_site as zs
    mp = pytest.MonkeyPatch()
    s = zs.make_site(tmp_path_factory.mktemp("v2_instance"), mp)
    yield s
    s.pkg.close()
    mp.undo()


def test_p0_w3_instance_on_the_small_package(zsite, tmp_path):
    pkg = zsite.pkg
    inst = I.build_instance(zsite.site2, _row(design="lag"), pkg=pkg, provider=zsite.provider)
    win = pkg.window("W3")
    assert np.array_equal(inst.sel, win) and inst.T == len(win) == pkg.T
    assert inst.t.equals(pkg.t[win]) and np.array_equal(inst.h0, pkg.h0[win])
    assert np.array_equal(inst.levels.at(0), inst.h0)
    assert inst.orbit_source == "phase5" and inst.orbit.shape == (inst.T,)
    assert inst.index == "ndwi" and inst.nir_type is False
    assert inst.shape == (pkg.H, pkg.W) and inst.pixel_m == 10.0
    assert set(inst.package_keys["shards"]) == {"2023", "2024", "2025"}
    assert inst.package_keys["package_key"] == pkg.keys["package_key"]
    assert inst.judges == []                             # no truth/<site>/judges.json
    assert inst.boundary_members == []                   # the fake boundary
    assert inst.paths["truth_dir"] == zsite.site2.truth_dir
    # L18 reads the tiles where 'fetch emodnet' writes them (<root>/common/<site>/emodnet)
    from experiments.benchmark.fetch import emodnet
    assert inst.paths["emodnet_dir"] == emodnet.site_dir("escalda", zsite.site2.root)
    assert "domain" not in inst._lazy                    # the domain is lazy
    d = inst.domain
    assert d.equals(domain.get_v2(pkg, settings.P0, "W3"))
    o = inst.obs.read(np.arange(5))
    assert o.X.shape == (inst.T, 5)
    ref = pkg.observe(settings.P0.id, win, pos=d.pos[:5])
    assert np.array_equal(o.W, ref.W) and np.array_equal(o.V, ref.V)
    with pytest.raises(I.InstanceUnavailable, match="series"):
        inst.series1                                     # no 'series' step in the fixture
    root = str(tmp_path / "root")
    idir, sha = I.write_instance_npz(root, inst)
    assert idir.endswith("runs/escalda/W3/C1.floor0.ndwi.D1/_instance/rand-A/Nall_r00")
    back = I.read_instance(idir)
    assert np.array_equal(back["keep"], d.keep) and back["K"] == d.K
    assert back["band_of"].dtype == np.int16 and back["s_km"].dtype == np.float32
    assert back["json"]["format"] == I.INSTANCE_FORMAT and back["json"]["T"] == inst.T
    assert back["json"]["subset_key"] == inst.subset_key
    m0 = os.stat(os.path.join(idir, "instance.npz")).st_mtime_ns
    idir2, sha2 = I.write_instance_npz(root, inst)
    assert sha2 == sha and os.stat(os.path.join(idir, "instance.npz")).st_mtime_ns == m0


def test_subset_instances_on_the_small_package(zsite):
    pkg = zsite.pkg
    a = I.build_instance(zsite.site2, _row(N="20", rep=1), pkg=pkg)
    win = pkg.window("W3")
    _, subs = settings.rand_subsets("escalda", "W3", len(win), 20, 5)
    assert np.array_equal(a.sel, win[subs[1]])
    assert a.domain.mode == "A" and a.domain.n_sel == 20
    f = I.build_instance(zsite.site2, _row(N="20", rep=1, mode="F", design="other"), pkg=pkg)
    assert f.domain.mode == "F" and f.domain.n_sel == len(win)
    h = I.build_instance(zsite.site2, _row(family="half", rep=1, design="base"), pkg=pkg)
    assert np.array_equal(h.sel, win[1::2])
    y = I.build_instance(zsite.site2, _row(window="Y2024"), pkg=pkg)
    assert np.array_equal(y.sel, pkg.window("Y2024"))
    assert set(y.package_keys["shards"]) == {"2024"}
    assert a.instance_id != f.instance_id != h.instance_id


def test_build_instance_unavailable_paths(zsite, monkeypatch):
    pkg = zsite.pkg

    def no_years(wid):
        raise RuntimeError("window W3 needs the shard years [2025]")

    monkeypatch.setattr(pkg, "window_years", no_years)
    with pytest.raises(I.InstanceUnavailable, match="2025"):
        I.build_instance(zsite.site2, _row(), pkg=pkg)
    monkeypatch.undo()
    with pytest.raises(I.InstanceUnavailable, match="bnd-eot20"):
        I.build_instance(zsite.site2, _row(variant="bnd-eot20"), pkg=pkg)
    monkeypatch.setattr(pkg, "dom_version", None)
    with pytest.raises(I.InstanceUnavailable, match="no dom"):
        I.build_instance(zsite.site2, _row(), pkg=pkg)


# ─────────────────────────────────────────────────────────────────────────────
#  pilot reps (pilots.py): the truth-free computability of the curve points
# ─────────────────────────────────────────────────────────────────────────────

def test_pilot_reps_on_the_small_package(zsite, tmp_path, capsys):
    from experiments.benchmark import pilots
    doc = pilots.pilot_reps([zsite.site2], design="curves-k1", root=str(tmp_path), max_px=500)
    assert doc["truth_free"] and os.path.exists(tmp_path / "common" / "pilot_reps.json")
    e = doc["sites"]["escalda"]
    pts = {(p["N"], p["rep"]): p for p in e["points"]}
    assert set(pts) == {(n, r) for n, R in ((20, 5), (45, 5), (90, 3), (180, 2), (330, 2))
                        for r in range(R)}
    p = pts[(20, 1)]
    assert p["status"] == "ok" and p["n_scenes"] == 20 and p["subset_kind"] == "nested"
    assert p["n_px_counted"] == 500 and p["n_clear_p10"] <= p["n_clear_median"] <= 20
    assert p["computable"]["L03"]["computable"] is False           # m2c needs 60 scenes
    assert "60" in p["computable"]["L03"]["reason"]
    assert p["computable"]["E0a"]["computable"] is False           # the 40-scene floor
    assert pts[(90, 0)]["status"] == "invalid"                     # N > T = 64
    assert e["kR"]["W3|20"]["L03"]["flag"] is True
    out = capsys.readouterr().out
    assert "[pilot reps] design curves-k1" in out and "INVALID" in out
