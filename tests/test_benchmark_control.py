# -*- coding: utf-8 -*-
"""The positive control: the synthetic site 'pc' in the Zarr store format.

A micro world (12 x 200 px, one year, 24 scenes) is written once per module;
the physics of the misspecified worlds is checked scene by scene without
writing stores. The package build on the micro world (record, pass 1,
geometry) is slow and runs only with V2_CONTROL_FULL=1.
"""
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

from experiments import zarr_cubes as zc
from experiments.benchmark import control, inputs, obs, settings

MICRO = {"H": 12, "W": 200, "scenes_per_year": 24}
FILL = -32768


@pytest.fixture(scope="module")
def micro(tmp_path_factory):
    out = tmp_path_factory.mktemp("pc_micro")
    return control.write_world(str(out), "gating", "profile", years=(2024,), frame=MICRO)


@pytest.fixture(scope="module")
def micro_zero(tmp_path_factory):
    out = tmp_path_factory.mktemp("pc_zero")
    return control.write_world(str(out), "gating", "zero", years=(2024,),
                               frame={"H": 8, "W": 40, "scenes_per_year": 6})


def test_micro_world_stores_open_with_zarr_cubes(micro):
    root = micro["cubes_root"]
    doc = micro["world_doc"]
    ds = zc.open_cube("pc", "2024-01-01", "2025-01-01", root=root)
    try:
        assert dict(ds.sizes) == {"t": 24, "y": 12, "x": 200}
        assert zc.band_names(ds) == list(zc.BANDS) and zc.dates(ds) == doc["dates"]
        assert all(str(ds[b].dtype) == "int16" and zc.fill_of(ds[b]) == FILL for b in zc.BANDS)
    finally:
        ds.close()
    chk = inputs.check_store(zc.store_path("pc", 2024, root))
    assert chk["ok"], chk["problems"]
    assert chk["offset_state"] == "no offset"
    for kind in ("verify.json", "format.txt", "job.json"):
        assert zc.sidecar_path("pc", 2024, kind, root).is_file()
    lab = zc.read_overpass("pc", root)
    assert sorted(lab) == doc["dates"] and all(t.strftime("%H") == "10" for t in lab.values())
    meta = inputs.read_scene_meta(inputs.scene_meta_path("pc", root))
    assert sorted(meta["dates"]) == doc["dates"]
    assert {o for v in meta["dates"].values() for o in v["relative_orbits"]} == {8, 51}
    assert {b for v in meta["dates"].values() for b in v["processing_baselines"]} == {"05.11"}
    # the harness's reader: the world's instants, frozen offsets, measured lattice
    src = inputs.ZarrSource("pc", "2024-01-01", "2025-01-01", root=root,
                            instants=doc["instants"], offsets=doc["offsets"],
                            lattice="measure", cache_mb=16)
    try:
        assert src.T == 24 and tuple(src.lattice20) == (0, 0)
        assert np.all(src.offsets == 0)
        assert src.grid.shape == (12, 200) and src.grid.epsg == control.EPSG
    finally:
        src.close()
    assert doc["format"] == control.WORLD_FORMAT and doc["gating"] and doc["years"] == [2024]
    assert json.loads(open(doc["paths"]["world"], encoding="utf-8").read())["world"] == "gating"


def test_20m_bands_are_exact_replicas_and_codes_in_range(micro):
    ds = zc.open_cube("pc", "2024-01-01", "2025-01-01", root=micro["cubes_root"])
    try:
        for b in ("B8A", "B11", "B12", "SCL", "CLD"):
            a = ds[b].values
            assert np.array_equal(a[:, 0::2, :], a[:, 1::2, :]), b
            assert np.array_equal(a[:, :, 0::2], a[:, :, 1::2]), b
        scl = ds["SCL"].values
        cld = ds["CLD"].values
        assert set(np.unique(scl[scl != FILL]).tolist()) <= set(range(12))
        assert {5, 6}.issubset(set(np.unique(scl).tolist()))
        v = cld[cld != FILL]
        assert v.min() >= 0 and v.max() <= 100
        assert np.array_equal(scl == FILL, cld == FILL)          # one no-data mask
        for b in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12"):
            x = ds[b].values
            assert x[x != FILL].min() >= 1                          # no BOA offset, no negatives
        # NDWI: water (the channel) high, the dykes low, on clear scenes
        g = ds["B03"].values.astype(float)
        n = ds["B08"].values.astype(float)
        clear = obs.mask_valid(obs.scl_u8(scl), "marea_clear")
        with np.errstate(invalid="ignore", divide="ignore"):
            nd = np.where(clear & (g > 0) & (n > 0), (g - n) / (g + n), np.nan)
        rows = np.nanmedian(nd, axis=(0, 2))
        assert rows[5] > 0.3 and rows[6] > 0.3 and rows[0] < -0.1 and rows[11] < -0.1
    finally:
        ds.close()


def test_haze_share_is_about_two_percent():
    df = control.scene_schedule((2024, 2025), 104)
    assert len(df) == 208 and df["haze"].sum() == round(0.02 * 208)
    assert 0.01 <= df["haze"].mean() <= 0.03
    assert set(df["orbit"]) == {8, 51} and df["date"].is_unique
    inst = pd.DatetimeIndex(df["instant"])
    assert ((inst - inst.normalize()) / pd.Timedelta(minutes=1)).min() >= 10 * 60 + 47
    assert ((inst - inst.normalize()) / pd.Timedelta(minutes=1)).max() <= 10 * 60 + 53
    big = control.scene_schedule((2024, 2025), 250)
    assert len(big) == 500 and big["haze"].sum() == 10 and big["nodata_strip"].sum() == 15
    with pytest.raises(ValueError):
        control.scene_schedule((2024,), 400)
    # haze adds 400 DN to every spectral band, nothing else
    dem = control.world_dem(12, 200)
    row = dict(df.iloc[3])
    row.update(haze=False, nodata_strip=False, band_fill=False)
    a = control._scene(dem, row, "gating", "profile", control.TIDE, False, control.SEED)
    b = control._scene(dem, dict(row, haze=True), "gating", "profile", control.TIDE, False,
                       control.SEED)
    for k in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12"):
        m = a[k] > 1                                 # not clipped at the 1 DN floor
        assert m.mean() > 0.9 and np.all(b[k][m].astype(int) - a[k][m].astype(int) == 400), k
    assert np.array_equal(a["SCL"], b["SCL"]) and np.all(b["CLD"] >= a["CLD"])


def test_planted_tau_is_monotone_in_s_and_zero_in_the_zero_scenario(micro, micro_zero):
    with np.load(micro["planted_path"]) as z:
        tau, s, amp, zt = z["tau"], z["s_true_km"], z["amp"], z["z_true"]
        assert set(z.files) >= {"z_true", "tau", "amp", "z"}
        assert np.array_equal(z["z"], z["z_true"]) and zt.dtype == np.float32
    assert tau.shape == (12, 200) and tau.dtype == np.float32
    assert np.all(np.diff(tau, axis=1) >= 0) and np.all(np.diff(s, axis=1) >= 0)
    np.testing.assert_allclose(tau, (120.0 * np.minimum(s / 12.0, 1.0)).astype(np.float32),
                               rtol=1e-6)
    assert tau[:, 0].max() == 0 and np.all(amp == 1.0)
    with np.load(micro_zero["planted_path"]) as z:
        assert np.all(z["tau"] == 0)
    # the profile reaches 120 min at 12 km and stays there
    assert control.lag_profile([0.0, 6.0, 12.0, 14.0], "profile").tolist() == [0, 60, 120, 120]
    assert control.amp_profile([0.0, 6.0, 30.0], "amplification").tolist() == [1.0, 1.1, 1.2]
    assert control.amp_profile([6.0], "gating").tolist() == [1.0]
    # the DEM: sea and channel below the lowest tide, dykes above the highest
    dem = control.planted_dem(12, 200)
    lo = -sum(c[2] for c in control.TIDE["constituents"]) * control.AMPLIFICATION["inland_ratio"]
    assert dem["z_true"][dem["zone"] <= 1].max() < lo
    assert dem["z_true"][dem["zone"] == 5].min() > -lo
    assert {0, 1, 3, 5} <= set(np.unique(dem["zone"]).tolist())     # banks are sub-pixel


def test_planted_provider_is_deterministic(micro):
    doc = micro["world_doc"]
    p1, p2 = control.planted_provider(doc), control.planted_provider(json.loads(json.dumps(doc)))
    t = pd.date_range("2024-03-01", periods=500, freq="7min")
    assert np.array_equal(p1.levels(t), p2.levels(t))
    assert np.array_equal(p1.levels(t[::-1]), p1.levels(t))                  # time order
    assert np.array_equal(p1.levels(t), control.tide_levels(t.asi8))
    assert "planted tide" in p1.name and np.ptp(p1.levels(t)) > 2.0


def test_planted_band_lags_on_a_given_domain(micro):
    with np.load(micro["planted_path"]) as z:
        tau = z["tau"].ravel().astype(float)
    keep = np.arange(200, 2200, 3)
    band = np.where(keep % 200 < 100, 0, 1)
    band[::50] = -1
    d = types.SimpleNamespace(keep=keep, band_of=band)
    r = control.planted_band_lags(micro["planted_path"], d)
    assert r["bands"] == [0, 1]
    for i, b in enumerate((0, 1)):
        assert r["tau_mean_min"][i] == pytest.approx(tau[keep[band == b]].mean())
        assert r["n_px"][i] == int((band == b).sum())
    with pytest.raises(ValueError):
        control.planted_band_lags(micro["planted_path"], types.SimpleNamespace(
            keep=keep, band_of=band[:-1]))


def test_expected_offset_reads_the_card(monkeypatch):
    from experiments.benchmark import lags

    cards = {"LX": {"control_offset_min": 5.0}, "LY": {"control_offset_min": None}, "LZ": {}}
    monkeypatch.setattr(lags, "card", lambda pid: cards[pid])
    assert control.expected_offset("LX") == 5.0
    assert control.expected_offset("LY") == 0.0 and control.expected_offset("LZ") == 0.0
    with pytest.raises(KeyError):
        control.expected_offset("nope")


def test_control_site2_is_the_v2_site_of_the_world(micro):
    doc = micro["world_doc"]
    out = doc["paths"]["out_root"]
    s2 = control.control_site2(out, doc)
    assert s2.site == "pc" and s2.cfg.mouth_side == "west" and s2.cfg.judges == ()
    assert s2.lattice == (0, 0) and s2.years == (2024,) and s2.keys == (settings.P0.id,)
    assert set(s2.offsets.values()) == {0} and sorted(s2.offsets) == doc["dates"]
    assert s2.cubes_root == micro["cubes_root"] and s2.dir == f"{out}/common/pc"
    assert s2.cfg.extra["planted"] == micro["planted_path"] and s2.cfg.site_id == 9
    assert s2.observation_keys() == (settings.P0.id,)
    assert s2.cfg.boundary["kind"] == "planted"


def test_misspecified_worlds_change_only_their_mechanism():
    dem = control.world_dem(12, 200)
    df = control.scene_schedule((2024,), 60)
    tide = control.TIDE
    s = control.s_of_x(dem["x_m"], dem["geom"])
    tau = control.lag_profile(s, "profile")

    def limb(row):
        t = pd.Timestamp(row["instant"]).value - np.round(tau * 6e10).astype(np.int64)
        d = control.tide_levels(t, tide) - control.tide_levels(t - 20 * 60_000_000_000, tide)
        return "fall" if np.all(d < 0) else "rise" if np.all(d >= 0) else "mixed"

    rows = [dict(r, haze=False, nodata_strip=False, band_fill=False, cloud_cover=0.0,
                 cloud_class="clear") for r in df.to_dict("records")]
    fall = [r for r in rows if limb(r) == "fall"]
    rise = [r for r in rows if limb(r) == "rise"]
    assert fall and rise
    for r in rise[:2]:            # rising limb: the hysteresis world is the gating world
        g = control._scene(dem, r, "gating", "profile", tide, False, control.SEED)
        h = control._scene(dem, r, "hysteresis", "profile", tide, False, control.SEED)
        assert all(np.array_equal(g[b], h[b]) for b in g)
    wet_g = wet_h = 0
    for r in fall[:4]:            # falling limb: drained 20 min later -> more water
        g = control._scene(dem, r, "gating", "profile", tide, False, control.SEED)
        h = control._scene(dem, r, "hysteresis", "profile", tide, False, control.SEED)
        wet_g += int((g["SCL"] == 6).sum())
        wet_h += int((h["SCL"] == 6).sum())
        assert h["B08"].astype(float).mean() <= g["B08"].astype(float).mean() + 1e-9
    assert wet_h > wet_g
    # MNAR: dark mud labelled SCL 2/3 on the low flats at low tide only
    low = min(rows, key=lambda r: control.tide_levels(pd.Timestamp(r["instant"]).value, tide))
    m_low = control._scene(dem, low, "mnar", "profile", tide, True, control.SEED)
    g_low = control._scene(dem, low, "gating", "profile", tide, True, control.SEED)
    assert np.isin(m_low["SCL"], (2, 3)).sum() > 0 and (g_low["SCL"] == 2).sum() == 0
    m_hi = control._scene(dem, low, "mnar", "profile", tide, False, control.SEED)
    assert (m_hi["SCL"] == 2).sum() == 0
    assert m_low["B08"].astype(float).mean() < g_low["B08"].astype(float).mean()
    with pytest.raises(ValueError):
        control.write_world("unused", "nope")
    with pytest.raises(ValueError):
        control._frame({"H": 11, "W": 200})


@pytest.mark.skipif(os.environ.get("V2_CONTROL_FULL") != "1",
                    reason="slow: set V2_CONTROL_FULL=1 to build the control package")
def test_package_prepare_v2_on_the_micro_world(micro, monkeypatch):
    from experiments.benchmark import domain, package

    for k, v in {"MIN_INTERTIDAL": 40, "MIN_FINITE_S": 60, "MIN_PX_BAND": 60,
                 "MIN_CLEAR": 10}.items():
        monkeypatch.setattr(domain, k, v)
    monkeypatch.setattr(package, "SPACE_MARGIN_GB", 0.0)
    doc = micro["world_doc"]
    s2 = control.control_site2(doc["paths"]["out_root"], doc)
    ran = package.prepare_v2(s2, steps=("record", "pass1", "geometry"),
                             provider=control.planted_provider(doc))
    assert {"record", "pass1:2024", "geometry"} <= set(ran)
    pkg = control.open_package(s2, doc)
    try:
        assert pkg.T == 24 and list(pkg.dates) == doc["dates"]
        assert np.all(pkg.offsets == 0) and pkg.lattice == (0, 0)
        np.testing.assert_allclose(pkg.h0, control.tide_levels(pkg.t.asi8))
        full = pkg.full
        assert full.status == "ok" and full.n_keep > 300
        s = full.s_km[np.isfinite(full.s_km)]
        assert s.min() < 0.3 and s.max() > 1.5
        r = control.planted_band_lags(doc["paths"]["planted"], full)
        assert r["bands"] and 0 < r["tau_mean_min"][0] < 20
        assert pkg.provider() is pkg._provider
    finally:
        pkg.close()
