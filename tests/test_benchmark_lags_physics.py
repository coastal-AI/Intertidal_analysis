# -*- coding: utf-8 -*-
"""L17 (zero lag) and L18 (shallow-water celerity) on a synthetic frame.

The frame is 10 x 80 pixels of 100 m (8 km along s): land rows, flats
(the keep) and a permanently wet channel (wf = 1); the package geometry
(wf, s_frame) is given directly and the EMODnet DTM is a local GeoTIFF in
EPSG:4326 with its manifest, so nothing is fetched.
"""
import json
import os
import types

import numpy as np
import pytest

from experiments.benchmark import judges
from experiments.benchmark.fetch import emodnet
from experiments.benchmark.lags import common, physics
from experiments.benchmark.product import NotComputable
from tests.test_benchmark_lags_models import GAUGE, make_instance, series1

H, W, PX = 10, 80, 100.0
X0, Y0 = 640_000.0, 5_900_000.0
CRS = "EPSG:32631"
CHANNEL_ROWS = (4, 5)
FLAT_ROWS = (2, 3, 6, 7)


def frame():
    xs = X0 + PX / 2 + PX * np.arange(W)
    ys = Y0 - PX / 2 - PX * np.arange(H)
    s = np.broadcast_to(((np.arange(W) + 0.5) * PX / 1000.0)[None, :], (H, W)).copy()
    wf = np.zeros((H, W))
    wf[list(CHANNEL_ROWS)] = 1.0
    wf[list(FLAT_ROWS)] = 0.5
    s[[0, 1, 8, 9]] = np.nan                         # land: no along-water distance
    return xs, ys, s, wf


def dtm(path, value, nodata_box=None):
    """A constant-elevation DTM (EPSG:4326, 0.002 deg) covering the frame."""
    from pyproj import Transformer
    from rasterio.transform import from_origin

    xs, ys, _, _ = frame()
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([xs[0], xs[-1], xs[0], xs[-1]], [ys[0], ys[0], ys[-1], ys[-1]])
    lon0, lat1 = min(lon) - 0.02, max(lat) + 0.02
    nx = int((max(lon) - min(lon) + 0.04) / 0.002) + 1
    ny = int((max(lat) - min(lat) + 0.04) / 0.002) + 1
    a = np.full((ny, nx), float(value), np.float32)
    data = emodnet._bytes_of_geotiff(a, from_origin(lon0, lat1, 0.002, 0.002))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    man = {"format": emodnet.MANIFEST_FORMAT, "status": "ok", "coverage_id": emodnet.COVERAGE_ID,
           "file": {"name": emodnet.DTM_NAME, "sha256": emodnet.sha256_file(path),
                    "size": os.path.getsize(path)}}
    with open(os.path.join(os.path.dirname(path), emodnet.MANIFEST_NAME), "w",
              encoding="utf-8") as f:
        json.dump(man, f)


def instance(tmp_path, wf=None, judges_=None):
    xs, ys, s, wf0 = frame()
    wf = wf0 if wf is None else wf
    keep = np.flatnonzero(np.isin(np.arange(H * W) // W, FLAT_ROWS))
    s_keep = s.ravel()[keep]
    dom = types.SimpleNamespace(keep=keep, s_km=s_keep, band_of=np.floor(s_keep / 3.0).astype(int),
                                K=3, status="ok", sea=wf >= 0.9, s_frame=s)
    inst = make_instance("Y2024", judges_=judges_)
    inst.domain = dom
    inst.shape = (H, W)
    inst.grid = {"xs": xs, "ys": ys, "crs": CRS}
    inst.geometry = {"wf": wf.astype(np.float32), "s_frame": s.astype(np.float32)}
    inst.series1 = series1("Y2024")
    # fetch emodnet writes <root>/common/<site>/emodnet (DTM_DIR below), which L18
    # finds first (instance.instance_paths gives that folder since the
    # integration); a site-less emodnet_dir, as an older instance gave, still works
    inst.paths = {"root": str(tmp_path), "emodnet_dir": f"{tmp_path}/common/emodnet"}
    return inst


def dtm_path(tmp_path):
    return os.path.join(emodnet.site_dir("escalda", str(tmp_path)), emodnet.DTM_NAME)


@pytest.fixture(autouse=True)
def no_gauge(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("L17/L18 must never read a gauge series")
    monkeypatch.setattr(judges, "read_series", fail)


def test_l17_is_zero_everywhere(tmp_path):
    inst = instance(tmp_path)
    (r,) = physics.run_l17(inst, {"products": ("L17",)}, {})
    assert r["status"] == "ok" and r["kind"] == "point" and r["product"] == "L17"
    assert r["field"].dtype == np.float32 and len(r["field"]) == len(inst.domain.keep)
    assert np.all(r["field"] == 0)
    e = r["point"]["jA"]
    assert e["a"] == e["ab"] == e["b"] == e["hwlw"] == 0.0
    assert np.isnan(e["internal_a"]) and r["edge_share"] == 0.0
    (r,) = physics.run_l17(instance(tmp_path, judges_=[]), {"products": ("L17",)}, {})
    assert r["status"] == "ok" and r["point"] == {} and np.all(r["field"] == 0)


@pytest.mark.parametrize("h", [9.0, 2.5])
def test_l18_constant_depth_is_exact(tmp_path, h):
    inst = instance(tmp_path)
    offset, mean_h, lat_ref, _ = physics.lat_offset(inst)
    _, h1 = inst.series1
    assert offset == pytest.approx(mean_h - lat_ref) and lat_ref < mean_h
    dtm(dtm_path(tmp_path), offset - h)   # depth below MSL = h
    (r,) = physics.run_l18(inst, {"products": ("L18",)}, {})
    assert r["status"] == "ok" and r["kind"] == "point"
    h = offset - float(np.float32(offset - h))      # the depth the float32 DTM holds
    want = inst.domain.s_km * 1000.0 / np.sqrt(9.81 * h) / 60.0
    np.testing.assert_allclose(r["field"], want.astype(np.float32), rtol=1e-6)
    e = r["point"]["jA"]
    s_g = GAUGE["s_g_km"]
    assert e["a"] == e["ab"] == e["b"] == e["hwlw"]
    assert e["a"] == pytest.approx(s_g * 1000.0 / np.sqrt(9.81 * h) / 60.0, rel=1e-12)
    assert "s_g beyond the wet channel (depth extrapolated)" in e["flags"]   # 10 km > 8 km
    assert np.allclose(r["diag"]["hbar_m"], h)


def test_l18_depth_floor_and_empty_bins(tmp_path):
    xs, ys, s, wf = frame()
    wf2 = wf.copy()
    wf2[list(CHANNEL_ROWS), 20:30] = 0.5           # s in [2, 3) km: no permanent water
    inst = instance(tmp_path, wf=wf2)
    offset = physics.lat_offset(inst)[0]
    dtm(dtm_path(tmp_path), offset - 0.4)   # 0.4 m: floored to 1 m
    (r,) = physics.run_l18(inst, {"products": ("L18",)}, {})
    # bin 2 has no permanent water; bins 8-10 lie beyond the channel (the judge at 10 km)
    assert r["diag"]["interpolated_bins"] == [2, 8, 9, 10] and 0 in r["diag"]["floored_bins"]
    np.testing.assert_allclose(r["diag"]["hbar_m"], 1.0)
    want = inst.domain.s_km * 1000.0 / np.sqrt(9.81 * 1.0) / 60.0
    np.testing.assert_allclose(r["field"], want.astype(np.float32), rtol=1e-6)


def test_bin_depths_and_the_integral():
    s = np.array([0.2, 0.7, 1.5, 3.2, 3.9, 4.5])
    d = np.array([4.0, 4.0, 9.0, 16.0, 16.0, 0.5])
    bd = physics.bin_depths(s, d, n_bins=6)
    # bins 0, 1, 3 from the data; 2 interpolated between 1 and 3; 4 floored; 5 constant
    np.testing.assert_allclose(bd["hbar"], [4.0, 9.0, 12.5, 16.0, 1.0, 1.0])
    assert bd["interpolated"].tolist() == [False, False, True, False, False, True]
    assert bd["floored"].tolist() == [False, False, False, False, True, False]
    tau = physics.tau_of_s(np.array([0.0, 1.5, 7.0, np.nan]), [4.0, 9.0])
    sec = 1000.0 / np.sqrt(9.81 * 4.0) + 500.0 / np.sqrt(9.81 * 9.0)
    assert tau[0] == 0.0 and tau[1] == pytest.approx(sec / 60.0, rel=1e-12)
    # beyond the last bin: the last bin's depth
    far = (1000.0 / np.sqrt(9.81 * 4.0) + 6000.0 / np.sqrt(9.81 * 9.0)) / 60.0
    assert tau[2] == pytest.approx(far, rel=1e-12) and np.isnan(tau[3])
    with pytest.raises(ValueError):
        physics.bin_depths(s, np.full(6, np.nan), n_bins=3)


def test_l18_missing_or_changed_tiles_are_not_computable(tmp_path):
    inst = instance(tmp_path)
    with pytest.raises(NotComputable) as ei:
        physics.run_l18(inst, {"products": ("L18",)}, {})
    assert ei.value.stage == "no_depths" and "missing_input" in ei.value.diag
    p = dtm_path(tmp_path)
    dtm(p, -5.0)
    with open(p, "ab") as f:
        f.write(b"x")
    with pytest.raises(NotComputable) as ei:
        physics.run_l18(inst, {"products": ("L18",)}, {})
    assert ei.value.stage == "no_depths" and "changed" in ei.value.reason


def test_l18_variants_and_fallback_channel(tmp_path, monkeypatch):
    from tests.test_benchmark_lags_models import CONS
    from experiments.benchmark.lags import models
    monkeypatch.setattr(models, "CONSTITUENTS", CONS)
    inst = instance(tmp_path)
    offset = physics.lat_offset(inst)[0]
    dtm(dtm_path(tmp_path), offset - 4.0)
    (base,) = physics.run_l18(inst, {"products": ("L18",)}, {})
    (hw,) = physics.run_l18(inst, {"products": ("L18",), "variant": "hw"}, {})
    (lw,) = physics.run_l18(inst, {"products": ("L18",), "variant": "lw"}, {})
    (sm,) = physics.run_l18(inst, {"products": ("L18",), "variant": "site_mean"}, {})
    a = hw["diag"]["a_m2"]
    # the M2 amplitude of the test tide (1.7 m), divided by the M2 nodal factor the
    # GOT-corrected fit assumes (the synthetic tide has no nodal modulation)
    assert a == pytest.approx(1.7, rel=0.05)
    assert hw["point"]["jA"]["a"] < base["point"]["jA"]["a"] < lw["point"]["jA"]["a"]
    np.testing.assert_allclose(hw["diag"]["hbar_m"], 4.0 + a)
    assert sm["point"]["jA"]["a"] == pytest.approx(base["point"]["jA"]["a"], rel=1e-9)
    with pytest.raises(ValueError):
        physics.run_l18(inst, {"products": ("L18",), "variant": "nope"}, {})
    # without the package geometry: the instance's sea mask (declared, recorded)
    del inst.geometry
    (r,) = physics.run_l18(inst, {"products": ("L18",)}, {})
    assert r["diag"]["channel"].startswith("domain.sea")
    assert r["point"]["jA"]["a"] == pytest.approx(base["point"]["jA"]["a"], rel=1e-9)


def test_cards_and_inputs(tmp_path):
    for pid in ("L17", "L18"):
        assert common.check_card(physics.CARDS[pid]) == [], pid
    json.dumps(physics.CONFIG)
    inst = instance(tmp_path)
    assert physics.external_inputs(inst, {"method": "L17"}) == {"judges.json": None}
    dtm(dtm_path(tmp_path), -5.0)
    ext = physics.external_inputs(inst, {"method": "L18", "products": ("L18",)})
    assert len(ext["emodnet_manifest"]) == 64 and len(ext["emodnet_dtm"]) == 64
