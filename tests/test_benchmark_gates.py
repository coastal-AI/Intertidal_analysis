"""Validation v2: the reproduction and identity gates (experiments/benchmark/gates.py).

G-record, G-keep and G-marea run on a tiny synthetic legacy site written as
production's inputs are (an int16 netCDF cube, its overpass labels, the
production extraction made by ``marea.extract``) with a FakeTide boundary;
the 'production' MAREA products are made by ``marea.reconstruct`` itself on
the transition-zone record (n_boot 2: the frozen 30 is lowered in the
module fixture). Each gate passes, then fails on a one-byte change with
its first difference. The thin gates (G-zarr, G-offset, G-level, G-ocm)
read stand-in V1 outputs; every gate is NOT RUN when its inputs are absent.
"""
import dataclasses
import json
import os
import shutil
import types

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark import config, domain, gates, inputs, levels, package, product
from experiments.benchmark import settings
from experiments.benchmark.methods import elev_marea_pair as emp
from pyintertidal import geometry, marea
from tests.benchmark_site import FakeTide

FILL = -32768
SITE = "escalda"                 # a real site name (SITE_ID); every input is synthetic
H, W, PX = 60, 110, 50.0         # 5.5 km across: two 3-km bands, > 5,000 intertidal px
X0, Y0 = 500_000.0, 5_700_000.0
KEYS = ("C1.floor0.ndwi.D1", "C1.raw.ndwi.D1")
N_BOOT = 2


class OffsetTide(FakeTide):
    """A provider whose levels differ by a last bit (G-record's FAIL case)."""

    def levels(self, times):
        return super().levels(times) + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
#  the synthetic legacy site
# ─────────────────────────────────────────────────────────────────────────────

def write_legacy_inputs(root, seed=5):
    """An int16 netCDF cube (B03, B08, SCL) over 2023-2025 with labels for
    every date, and the production extraction (``marea.extract`` + s_km)."""
    import pyproj
    import xarray as xr

    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-03", "2025-12-28", freq="12D")
    T = len(dates)
    inst = (dates + pd.Timedelta(hours=10, minutes=50)
            + pd.to_timedelta(rng.integers(0, 1200, T), "s"))
    cols = np.arange(W)
    z = np.empty((H, W))
    z[:] = -1.6 + 3.2 * (cols - 5) / 98.0
    z += 0.1 * np.sin(np.arange(H) / 4.0)[:, None]
    z[:, :5] = -4.0
    z[:, 104:] = 5.0
    h = FakeTide.value(inst.asi8)
    wet = h[:, None, None] > z[None] + rng.normal(0.0, 0.05, (T, H, W))
    land = np.broadcast_to(z >= 4.9, wet.shape)
    b03 = np.where(wet, rng.normal(600, 100, wet.shape), rng.normal(900, 100, wet.shape))
    b08 = np.where(wet, rng.normal(100, 50, wet.shape), rng.normal(1500, 200, wet.shape))
    b03 = np.where(land, rng.normal(700, 80, wet.shape), b03)
    b08 = np.where(land, rng.normal(3000, 300, wet.shape), b08)
    scl = np.where(wet, 6, rng.choice([4, 5], wet.shape))
    scl = np.where(land, 4, scl)
    cloudy = rng.choice(T, 10, replace=False)
    for k in cloudy[:7]:                                     # cloud patches
        r, c = rng.integers(0, H - 25), rng.integers(0, W - 40)
        scl[k, r:r + 25, c:c + 40] = 9
    for k in cloudy[7:]:                                     # overcast scenes
        scl[k] = 8
    cloud = np.isin(scl, (8, 9))
    b03 = np.where(cloud, rng.normal(4000, 300, wet.shape), b03)
    b08 = np.where(cloud, rng.normal(4000, 300, wet.shape), b08)
    b03 = np.clip(np.round(b03), 1, 15000).astype(np.int16)
    b08 = np.clip(np.round(b08), 1, 15000).astype(np.int16)
    # clear observations whose NDWI is NaN (B03 + B08 == 0, or a fill) on mid-flat
    # pixels: production keeps them as dry (Y = 0); production_compat must too
    for k in range(12):
        t_, r_, c_ = int(rng.integers(0, T)), int(rng.integers(5, H - 5)), int(rng.integers(40, 70))
        if scl[t_, r_, c_] in (4, 5, 6):
            if k % 2:
                b03[t_, r_, c_] = b08[t_, r_, c_] = 0
            else:
                b03[t_, r_, c_] = FILL
    scl = scl.astype(np.int16)
    xs = X0 + PX * np.arange(W)
    ys = Y0 - PX * np.arange(H)
    crs = pyproj.CRS.from_epsg(32631)
    ds = xr.Dataset({v: (("t", "y", "x"), a) for v, a in
                     (("B03", b03), ("B08", b08), ("SCL", scl))},
                    coords={"t": dates, "y": ys, "x": xs})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": crs.to_wkt()})
    enc = {v: {"dtype": "int16", "_FillValue": FILL, "chunksizes": (1, 20, 55), "zlib": True}
           for v in ("B03", "B08", "SCL")}
    cube = os.path.join(root, "cube.nc")
    ds.to_netcdf(cube, encoding=enc)
    ex = marea.extract(cube)
    seeds = geometry.mouth_seeds(ex["sea"], side="west")
    s_km = geometry.along_distance(ex["sea"] | ex["inter"], seeds,
                                   PX).ravel()[ex["keep"]] / 1000.0
    extract = os.path.join(root, "marea_extract.npz")
    np.savez_compressed(extract, Y=ex["Y"], C=ex["C"], keep=ex["keep"], dates=ex["dates"],
                        shape=np.array(ex["shape"]), sea=ex["sea"], inter=ex["inter"],
                        s_km=s_km.astype(np.float32), crs_wkt=ex["crs_wkt"])
    labels = {str(d.date()): t.strftime("%Y-%m-%dT%H:%M:%S") for d, t in zip(dates, inst)}
    lab = os.path.join(root, "overpass_times.json")
    with open(lab, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    return {"root": root, "cube": cube, "extract": extract, "labels": lab, "ex": ex,
            "dates": dates, "inst": inst, "T": T, "labels_map": labels}


def legacy_site2(inp, out_root, prod_dir, comp_dir=None):
    cfg = dataclasses.replace(
        config.site_config(SITE, 20), cube=inp["cube"], extract=inp["extract"],
        overpass=inp["labels"], pixel_m=PX,
        boundary=types.MappingProxyType({"kind": "fake", "name": "fake tide"}),
        mouth_side="west", judges=(), elevation_truth=(), transition_result=None,
        published=types.MappingProxyType({"marea": prod_dir, "comparison": comp_dir}),
        extra=types.MappingProxyType({}))
    return package.V2Site(site=SITE, cfg=cfg, root=f"{out_root}/{gates.LEGACY_SUBDIR}",
                          source="netcdf", boundary="production", cube=inp["cube"],
                          labels=inp["labels"], years=(2023, 2024, 2025), keys=KEYS,
                          cache_mb=64, block_rows=20)


def run_production(inp, tz_dates, out_dir, n_boot=N_BOOT):
    """Production MAREA as it ran: reconstruct on the transition-zone record
    with the full-record extraction domain."""
    ex = dict(inp["ex"])
    idx = np.flatnonzero(np.isin(ex["dates"], tz_dates))
    ex["Y"], ex["C"], ex["dates"] = ex["Y"][idx], ex["C"][idx], ex["dates"][idx]
    return marea.reconstruct(inp["cube"], out_dir, name="synthetic", pixel_m=PX,
                             boundary=FakeTide(), extraction=ex, mouth_side="west",
                             verbose=False, band_km=3.0, min_px_band=500, n_boot=n_boot,
                             min_scenes=40, overpass_times=inp["labels_map"])


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    """The legacy package of the synthetic site (every G-marea step), its
    C2 record and the production products made on it."""
    base = tmp_path_factory.mktemp("gates")
    mp = pytest.MonkeyPatch()
    mp.setattr(package, "SPACE_MARGIN_GB", 0.0)
    mp.setattr(gates, "GATE_N_BOOT", N_BOOT)
    mp.setattr(gates.product, "git_state", lambda root=None: {"head": "test", "dirty": None})
    try:
        inp = write_legacy_inputs(str(base / "inputs"))
        root = str(base / "root").replace("\\", "/")
        prod = str(base / "production" / "marea")
        s2 = legacy_site2(inp, root, prod)
        package.prepare_v2(s2, steps=list(gates.GATE_STEPS["marea"]), provider=FakeTide())
        pkg = package.V2Package(s2)
        tz = [str(d) for d in np.asarray(pkg.dates)[pkg.rule_subset("C2", "W3")]]
        pkg.close()
        result = run_production(inp, tz, prod)
        assert result is not None and result["estado"] == "ok"
        yield types.SimpleNamespace(inp=inp, root=root, prod=prod, s2=s2, tz=tz, base=base,
                                    result=result)
    finally:
        mp.undo()


def test_fixture_is_a_real_two_band_production_run(legacy):
    r = legacy.result
    assert len(r["centros_km"]) == 2 and r["n_px"] > 5000
    assert 40 <= r["n_escenas"] < legacy.inp["T"] and r["n_boot"] == N_BOOT
    assert r["record_dates"] == legacy.tz and r["quality_min_bracket"] == 3


# ─────────────────────────────────────────────────────────────────────────────
#  G-record
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_record_pass_and_fail(legacy):
    res = gates.gate_record(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                            reference_provider=FakeTide(), check_zarr=False)
    assert res["verdict"] == "PASS", res
    c = res["compared"]
    assert c["T_production"] == c["T_v2_window"] == legacy.inp["T"]
    assert [e["check"] for e in c["checks"]] == ["dates", "t_ns", "h0"]
    bad = gates.gate_record(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                            reference_provider=OffsetTide(), check_zarr=False)
    assert bad["verdict"] == "FAIL" and bad["first_difference"]["check"] == "h0"
    assert bad["first_difference"]["index"] == 0


def test_gate_record_checks_the_record_the_lag_rows_use(legacy, tmp_path):
    """'zarr_record': the record of the site's Zarr package (the lag rows'
    record) against the production record: NOT RUN without it; t_ns bitwise
    on the common dates; every date difference explained, else FAIL; h0
    differences are descriptive."""
    import shutil
    kw = dict(root=legacy.root, site2=legacy.s2, provider=FakeTide(),
              reference_provider=FakeTide())
    none = gates.gate_record(SITE, **kw)            # no Zarr package at this root yet
    assert none["verdict"] == "NOT RUN"
    assert none["compared"]["checks"][-1]["check"] == "zarr_record"
    # the legacy record stands in for the Zarr record: identical, PASS
    ok = gates.gate_record(SITE, zarr_site2=legacy.s2, **kw)
    assert ok["verdict"] == "PASS", ok
    zr = ok["compared"]["zarr_record"]
    assert zr["n_common"] == legacy.inp["T"] and zr["n_t_ns_differ"] == 0
    assert zr["h0_max_abs_diff_m"] == 0.0
    # a record with other instants (one second later) FAILs
    zdir = tmp_path / "zarr_site"
    zdir.mkdir()
    rec = product.read_json(os.path.join(legacy.s2.dir, "record.json"))
    rec2 = dict(rec, t_ns=[int(t) + 10 ** 9 for t in rec["t_ns"]])
    product.write_json(str(zdir / "record.json"), rec2)
    zs = types.SimpleNamespace(site=SITE, dir=str(zdir))
    bad = gates.gate_record(SITE, zarr_site2=zs, **kw)
    assert bad["verdict"] == "FAIL" and bad["first_difference"]["what"] == "t_ns"
    # a date only the v2 record holds: NOT RUN without verify.json, FAIL when
    # G-zarr's verify.json does not list it, PASS when it lists it store-only
    assert rec["dates"][0] != "2023-01-01"
    rec3 = dict(rec, dates=["2023-01-01"] + rec["dates"],
                t_ns=[int(rec["t_ns"][0]) - 86_400 * 10 ** 9] + rec["t_ns"],
                h0=[0.0] + rec["h0"])
    product.write_json(str(zdir / "record.json"), rec3)
    cubes = str(tmp_path / "cubes")
    kw2 = dict(kw, cubes_root=cubes)
    assert gates.gate_record(SITE, zarr_site2=zs, **kw2)["verdict"] == "NOT RUN"
    vp = inputs.zc.sidecar_path(SITE, 2023, inputs.VERIFY_NAME, cubes)
    vp.parent.mkdir(parents=True)
    vp.write_text(json.dumps({"verdict": "equal", "dates": {"only_store": [],
                                                            "only_reference": []}}),
                  encoding="utf-8")
    bad = gates.gate_record(SITE, zarr_site2=zs, **kw2)
    assert bad["verdict"] == "FAIL" and bad["first_difference"]["unexplained"] == ["2023-01-01"]
    vp.write_text(json.dumps({"verdict": "equal", "dates": {"only_store": ["2023-01-01"],
                                                            "only_reference": []}}),
                  encoding="utf-8")
    ok = gates.gate_record(SITE, zarr_site2=zs, **kw2)
    assert ok["verdict"] == "PASS", ok
    assert ok["compared"]["zarr_record"]["explained"] == {"2023-01-01": "store-only date (G-zarr)"}
    shutil.rmtree(str(zdir))


# ─────────────────────────────────────────────────────────────────────────────
#  G-keep
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_keep_pass_and_fail(legacy, tmp_path):
    res = gates.gate_keep(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide())
    assert res["verdict"] == "PASS", res
    c = res["compared"]
    assert c["s_km"].startswith("the extraction's stored")
    assert c["dates"]["only_extraction"] == [] and c["dates"]["only_record"] == []
    assert c["p0_keep_differences"]["n_keep_p0"] > 0
    assert res["package_key"] is not None
    # one keep pixel dropped from a copy of the extraction: FAIL at keep
    z = dict(np.load(legacy.inp["extract"], allow_pickle=False))
    drop = int(z["keep"][100])
    z["keep"] = np.delete(z["keep"], 100)
    z["inter"] = z["inter"].copy()
    z["inter"].ravel()[drop] = False
    z["s_km"] = np.delete(z["s_km"], 100)
    p = tmp_path / "marea_extract.npz"
    np.savez_compressed(p, **z)
    s2 = dataclasses.replace(legacy.s2, cfg=dataclasses.replace(legacy.s2.cfg, extract=str(p)))
    bad = gates.gate_keep(SITE, root=legacy.root, site2=s2, provider=FakeTide(), build=False)
    assert bad["verdict"] == "FAIL" and bad["first_difference"]["check"] == "keep"
    assert "inter" in bad["reason"] and bad["compared"]["s_km"].startswith("not compared")


# ─────────────────────────────────────────────────────────────────────────────
#  G-marea
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_marea_reproduces_production_bitwise(legacy):
    res = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide())
    assert res["verdict"] == "PASS", (res["reason"], res["first_difference"])
    checks = {e["check"]: e["ok"] for e in res["compared"]["checks"]}
    for k in ("record", "c2_rule", "keep", "s_km", "band", "tau_min", "tau_usado_min",
              "tau_lo_min", "tau_hi_min", "tau_boot_min", "tau_px_min", "z", "sigma",
              "result.n_px_cota", "result.tau_lo_min", "result.rango_marea_m"):
        assert checks[k] is True, k
    assert res["compared"]["nodelay"] == "not compared"
    assert res["compared"]["lag"]["K"] == 2
    assert "equal to the production extraction" in res["compared"]["domain_source"]


def test_gate_marea_fails_on_one_byte_and_reports_it(legacy, tmp_path):
    prod = tmp_path / "marea"
    shutil.copytree(legacy.prod, prod)
    z = dict(np.load(prod / "marea.npz", allow_pickle=False))
    i = int(np.flatnonzero(np.isfinite(z["z"]))[7])
    raw = z["z"].view(np.uint8).copy()
    raw[4 * i] ^= 1                                            # one byte of one float32
    z["z"] = raw.view(np.float32)
    np.savez_compressed(prod / "marea.npz", **z)
    bad = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                           production_dir=str(prod), build=False)
    assert bad["verdict"] == "FAIL"
    fd = bad["first_difference"]
    assert fd["check"] == "z" and fd["index"] == i and fd["n_differ"] == 1


def test_gate_marea_runs_on_the_extraction_when_the_domain_rule_differs(legacy, monkeypatch):
    """A v2 domain that differs from the extraction (G-keep's FAIL) leaves
    G-marea on the production extraction's masks: the adapter is still
    compared bit for bit."""
    real = domain.for_setting

    def shifted(pkg, setting, wid, sel=None, with_frame=False):
        d = real(pkg, setting, wid, sel, with_frame)
        if setting == gates.KEEP_SETTING:
            keep = d.keep[1:]
            inter = d.inter.copy()
            inter.ravel()[d.keep[0]] = False
            d = domain.assemble("F", keep, d.sea, inter, pkg.pixel_m, pkg.cfg.mouth_side,
                                dom=pkg.dom)
        return d

    monkeypatch.setattr(domain, "for_setting", shifted)
    res = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                           build=False)
    assert res["verdict"] == "PASS", (res["reason"], res["first_difference"])
    assert res["compared"]["domain_source"].startswith("the production extraction's")
    assert "1 keep pixels" in res["compared"]["domain_source"]


def test_gate_marea_refuses_another_configuration_and_missing_products(legacy, tmp_path):
    prod = tmp_path / "marea"
    shutil.copytree(legacy.prod, prod)
    rj = product.read_json(prod / "result.json")
    rj.pop("quality_min_bracket")
    with open(prod / "result.json", "w", encoding="utf-8") as f:
        json.dump(rj, f)
    res = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                           production_dir=str(prod), build=False)
    assert res["verdict"] == "NOT RUN" and "quality_min_bracket" in res["reason"]
    res = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2,
                           production_dir=str(tmp_path / "nothing"), build=False)
    assert res["verdict"] == "NOT RUN" and "missing" in res["reason"]


def test_gate_marea_compares_the_no_delay_map(legacy, tmp_path):
    pkg = package.V2Package(legacy.s2)
    pkg._provider = FakeTide()
    try:
        where = {d: i for i, d in enumerate(pkg.dates)}
        sel = np.array([where[d] for d in legacy.tz])
        dom = domain.for_setting(pkg, gates.KEEP_SETTING, "W3")
        inst = gates.gate_instance(pkg, gates.MAREA_SETTING, "W3", sel, dom)
        out = emp.compute(inst, production_compat=True, products=("E0a",))
        # the fixture exercises production_compat: clear pixels whose NDWI is NaN
        # enter production as dry observations, so the v2 observation (C = V) differs
        inst_v = gates.gate_instance(pkg, gates.MAREA_SETTING, "W3", sel, dom)
        normal = emp.compute(inst_v, production_compat=False, products=("E0a", "L01"))
        assert (normal["clear_obs"] <= out["clear_obs"]).all()
        assert (normal["clear_obs"] < out["clear_obs"]).any()
        # ... which moves the continuous m2a lag (z sits on the 50-point mu grid), so
        # the bitwise G-marea PASS depends on production_compat
        compat = emp.compute(inst, production_compat=True, products=("L01",))["lag"]
        assert not np.array_equal(normal["lag"]["tau_raw"], compat["tau_raw"])
    finally:
        pkg.close()
    comp = tmp_path / "comparison"
    comp.mkdir()
    np.savez_compressed(comp / "products.npz", keep=dom.keep, z_plain=out["z_a"],
                        record_dates=np.array(legacy.tz))
    res = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                           comparison_dir=str(comp), build=False)
    assert res["verdict"] == "PASS" and res["compared"]["nodelay"].startswith("compared")
    # a map made before the quality rule: equal where both exist, more pixels
    zp = out["z_a"].copy()
    nan = np.flatnonzero(np.isnan(zp))
    assert len(nan)
    zp[nan[:3]] = 0.25
    np.savez_compressed(comp / "products.npz", keep=dom.keep, z_plain=zp,
                        record_dates=np.array(legacy.tz))
    res = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                           comparison_dir=str(comp), build=False)
    assert res["verdict"] == "PASS" and "predates" in res["compared"]["nodelay"]
    zp = out["z_a"].copy()
    fin = np.flatnonzero(np.isfinite(zp))
    zp[fin[0]] += 1e-9
    np.savez_compressed(comp / "products.npz", keep=dom.keep, z_plain=zp,
                        record_dates=np.array(legacy.tz))
    res = gates.gate_marea(SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                           comparison_dir=str(comp), build=False)
    assert res["verdict"] == "FAIL" and res["first_difference"]["check"] == "nodelay_z"


def test_adapters_run_alike_on_the_instance_module_views(legacy):
    """The E0 and L02 adapters give the same bits through the queue's
    LevelView / ObsView (experiments.benchmark.instance) as through the
    gate's own minimal views."""
    from experiments.benchmark import instance as inst_mod
    from experiments.benchmark.lags import marea_family as mf
    pkg = package.V2Package(legacy.s2)
    pkg._provider = FakeTide()
    try:
        sel = pkg.window("W3")
        dom = domain.for_setting(pkg, gates.P0_SETTING, "W3")
        mine = gates.gate_instance(pkg, gates.P0_SETTING, "W3", sel, dom)
        theirs = types.SimpleNamespace(**vars(mine))
        theirs.levels = inst_mod.LevelView(pkg.levels, sel)
        theirs.obs = inst_mod.ObsView(pkg, gates.P0_SETTING, sel, dom)
        theirs.h0 = theirs.levels.at(0)
        theirs.cache = {}
        a = emp.compute(mine, products=("E0a",))
        b = emp.compute(theirs, products=("E0a",))
        assert np.array_equal(a["z_a"].view(np.int64), b["z_a"].view(np.int64))
        (l2a,) = mf.run_l02(mine, {"method": "L02"}, {})
        (l2b,) = mf.run_l02(theirs, {"method": "L02"}, {})
        assert np.array_equal(l2a["band"]["tau"], l2b["band"]["tau"], equal_nan=True)
        assert np.isfinite(l2a["band"]["tau"]).all()
    finally:
        pkg.close()


# ─────────────────────────────────────────────────────────────────────────────
#  run_gate, verdict files, NOT RUN on missing inputs
# ─────────────────────────────────────────────────────────────────────────────

DOC_KEYS = {"format", "gate", "site", "verdict", "reason", "package_key", "compared",
            "first_difference", "code_sha256", "utc", "git"}


def test_run_gate_writes_the_verdict_schema(legacy, monkeypatch):
    monkeypatch.setattr(gates.product, "git_state", lambda root=None: {"head": "x", "dirty": None})
    doc = gates.run_gate("record", SITE, root=legacy.root, site2=legacy.s2, provider=FakeTide(),
                         reference_provider=FakeTide(), zarr_site2=legacy.s2)
    assert doc["code_sha256"] == gates.gate_code_shas("record")
    assert gates.gate_code_stale(doc, "record") is None
    assert "no code hashes" in gates.gate_code_stale(dict(doc, code_sha256=None), "record")
    assert set(doc) == DOC_KEYS and doc["format"] == "v2-gate-1" and doc["verdict"] == "PASS"
    on_disk = gates.read_gate(legacy.root, SITE, "record")
    assert on_disk["verdict"] == "PASS" and set(on_disk) == DOC_KEYS
    assert gates.gate_passed(legacy.root, SITE, "record")
    assert gates.gate_passed(legacy.root, SITE, "record", package_key=doc["package_key"])
    assert not gates.gate_passed(legacy.root, SITE, "record", package_key="other")
    assert not gates.gate_passed(legacy.root, SITE, "nothing")
    # a gate that raises is NOT RUN with the error, never PASS
    monkeypatch.setitem(gates.GATES, "boom", lambda site, **kw: 1 / 0)
    doc = gates.run_gate("boom", SITE, root=legacy.root)
    assert doc["verdict"] == "NOT RUN" and "ZeroDivisionError" in doc["reason"]
    assert doc["compared"]["traceback"]
    with pytest.raises(ZeroDivisionError):
        gates.run_gate("boom", SITE, root=legacy.root, raise_errors=True)
    with pytest.raises(KeyError):
        gates.run_gate("nope", SITE, root=legacy.root)
    assert gates.GATES["lagtruth"] == "experiments.benchmark.scoring:gate_lagtruth"


def test_legacy_gates_are_not_run_without_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(gates.product, "git_state", lambda root=None: {"head": "x", "dirty": None})
    root = str(tmp_path / "root")
    s2 = gates.legacy_site(SITE, root)
    assert s2.source == "netcdf" and s2.boundary == "production"
    assert s2.dir.replace("\\", "/").endswith("gates/_legacy/common/escalda")
    cfg = dataclasses.replace(s2.cfg, cube=str(tmp_path / "none.nc"),
                              extract=str(tmp_path / "none.npz"),
                              overpass=str(tmp_path / "none.json"),
                              published=types.MappingProxyType({}))
    s2 = dataclasses.replace(s2, cfg=cfg, cube=None, labels=None)
    for name in ("record", "keep", "marea"):
        doc = gates.run_gate(name, SITE, root=root, site2=s2)
        assert doc["verdict"] == "NOT RUN", (name, doc)
        assert set(doc) == DOC_KEYS
    empty = str(tmp_path / "no_cubes")
    os.makedirs(empty)
    for name in ("zarr", "offset", "level", "ocm"):
        doc = gates.run_gate(name, SITE, root=root, cubes_root=empty)
        assert doc["verdict"] == "NOT RUN", (name, doc)
        assert set(doc) == DOC_KEYS and doc["first_difference"] is None
        assert gates.read_gate(root, SITE, name)["verdict"] == "NOT RUN"


# ─────────────────────────────────────────────────────────────────────────────
#  thin gates over V1 outputs
# ─────────────────────────────────────────────────────────────────────────────

def _verify(equal=10, differs=0):
    b = {"equal": equal, "value_differs": differs, "nodata_only_store": 0,
         "nodata_only_reference": 0, "differing_pixels": differs}
    return {"verdict": "identical" if not differs else "differs",
            "bands": {k: dict(b) for k in ("B03", "B08", "SCL")},
            "dates": {"only_store": ["2024-01-01"], "only_reference": []},
            "dates_with_differences": 1 if differs else 0,
            "dates_with_differences_listed": {"2024-02-03": {"B03": differs}} if differs else {}}


def test_gate_zarr(tmp_path):
    from experiments import zarr_cubes as zc
    cubes = str(tmp_path / "cubes")
    for y, v in ((2023, _verify()), (2024, _verify())):
        p = zc.sidecar_path(SITE, y, inputs.VERIFY_NAME, cubes)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(v), encoding="utf-8")
    root = str(tmp_path / "root")
    res = gates.gate_zarr(SITE, root=root, cubes_root=cubes, years=[2023, 2024])
    assert res["verdict"] == "PASS", res
    p = zc.sidecar_path(SITE, 2024, inputs.VERIFY_NAME, cubes)
    p.write_text(json.dumps(_verify(differs=3)), encoding="utf-8")
    res = gates.gate_zarr(SITE, root=root, cubes_root=cubes, years=[2023, 2024])
    assert res["verdict"] == "FAIL" and res["first_difference"]["first_date"] == "2024-02-03"
    assert "pre-registered response" in res["reason"]
    res = gates.gate_zarr(SITE, root=root, cubes_root=cubes, years=[2023, 2025])
    assert res["verdict"] == "NOT RUN"
    assert gates.gate_zarr(SITE, root=root, cubes_root=cubes)["verdict"] == "NOT RUN"


def test_gate_offset(tmp_path):
    root = str(tmp_path / "root")
    assert gates.gate_offset(SITE, root=root)["verdict"] == "NOT RUN"
    rep = {"site": SITE, "source": {"kind": "zarr"}, "offset_rule_sha256":
           inputs.offset_rule_sha256(), "offsets": {"2023-01-05": 0, "2023-01-10": 1000},
           "offset_counts": {"no offset": 1, "+1000 in the values": 1}}
    path = os.path.join(root, "common", "inputs_report.json")
    os.makedirs(os.path.dirname(path))

    def write(r):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": inputs.INPUTS_REPORT_FORMAT, "sites": {SITE: r}}, f)

    write(rep)
    res = gates.gate_offset(SITE, root=root)
    assert res["verdict"] == "PASS" and "not in the report" in res["compared"]["anchor_step"]
    write(dict(rep, offsets={"2023-01-05": 0, "2023-01-10": None}))
    res = gates.gate_offset(SITE, root=root)
    assert res["verdict"] == "FAIL" and res["first_difference"]["dates"] == ["2023-01-10"]
    write(dict(rep, offset_rule_sha256="0" * 64))
    assert gates.gate_offset(SITE, root=root)["verdict"] == "NOT RUN"
    assert gates.gate_offset("wadden", root=root)["verdict"] == "NOT RUN"


def _write_sources(sdir, t, with_lt=False, corrupt=False):
    rng = np.random.default_rng(3)
    ref = np.sin(np.arange(len(t)) / 3.0)
    knots = [{"code": "g1", "s_km": 4.0, "values": ref + 0.1 + rng.normal(0, 0.01, len(t))},
             {"code": "g2", "s_km": 9.0, "values": ref - 0.05}]
    pix = np.arange(5, dtype=np.int64)
    lg = levels.level_lg(SITE, t, ref, knots, pix, np.array([0.5, 4.0, 6.0, 9.0, 12.0]))
    lb = levels.SourceLevels("LB", SITE, t, H=ref[None, :], pix=None)
    files = {}
    for name, src in (("levels_LB.npz", lb), ("levels_LG.npz", lg)):
        p = os.path.join(sdir, name)
        src.save(p)
        files[name] = {"source": src.source, "status": src.status, "reason": src.reason,
                       "unit_status": levels.UNIT_STATUS[src.status], "groups": int(src.G),
                       "flags": src.flags, "sha256": product.sha256_file(p)}
    if with_lt:
        lt = levels.SourceLevels("LT", SITE, t, H=np.vstack([ref, ref]), pix=pix,
                                 g0=np.array([0, 0, 1, 1, 1]))
        p = os.path.join(sdir, "levels_LT.npz")
        lt.save(p)
        files["levels_LT.npz"] = {"source": "LT", "status": "ok", "reason": "",
                                  "unit_status": "ok", "groups": 2, "flags": {},
                                  "sha256": product.sha256_file(p)}
    if corrupt:
        files["levels_LG.npz"]["sha256"] = "0" * 64
    product.write_json(os.path.join(sdir, "sources.json"),
                       {"format": package.SOURCES_FORMAT, "site": SITE, "files": files})


def test_gate_level(tmp_path):
    root = str(tmp_path / "root")
    assert gates.gate_level(SITE, root=root)["verdict"] == "NOT RUN"
    sdir = os.path.join(root, "common", SITE)
    os.makedirs(sdir)
    t = pd.date_range("2023-01-01", periods=40, freq="5D")
    _write_sources(sdir, t)
    res = gates.gate_level(SITE, root=root)
    assert res["verdict"] == "PASS", res
    note = {e["check"]: e.get("note") for e in res["compared"]["checks"]}
    assert "reproduced" in note["levels_LG.npz"]
    _write_sources(sdir, t, with_lt=True)
    res = gates.gate_level(SITE, root=root)
    assert res["verdict"] == "NOT RUN" and "levels_LT.npz" in res["reason"]
    _write_sources(sdir, t, corrupt=True)
    res = gates.gate_level(SITE, root=root)
    assert res["verdict"] == "FAIL" and "sha256" in res["first_difference"]["what"]


def test_gate_ocm(tmp_path):
    root = str(tmp_path / "root")
    res = gates.gate_ocm(SITE, root=root, years=[2023, 2024])
    assert res["verdict"] == "NOT RUN"
    card = json.loads(json.dumps(dict(settings.CLOUD["C11"].params),
                                 default=product._json_default))
    prov = {"package": "omnicloudmask", "version": None, "card": card, "injected": False,
            "predictor": "omnicloudmask.predict_from_array", "weights": [{"sha256": "a"}]}
    for y in (2023, 2024):
        d = os.path.join(root, "common", SITE, "years", str(y), "ocm")
        os.makedirs(d)
        product.write_json(os.path.join(d, "ocm.json"), {"year": y, "provenance": prov})
    res = gates.gate_ocm(SITE, root=root, years=[2023, 2024])
    assert res["verdict"] == "PASS", res
    assert "one-scene" in res["compared"]["not_checked"]
    d = os.path.join(root, "common", SITE, "years", "2024", "ocm", "ocm.json")
    product.write_json(d, {"year": 2024, "provenance": dict(prov, injected=True,
                                                            predictor="FakeOCM")})
    res = gates.gate_ocm(SITE, root=root, years=[2023, 2024])
    assert res["verdict"] == "FAIL" and "injected" in res["first_difference"]["what"]
    product.write_json(d, {"year": 2024, "provenance": dict(prov, weights=[{"sha256": "b"}])})
    res = gates.gate_ocm(SITE, root=root, years=[2023, 2024])
    assert res["verdict"] == "FAIL" and res["first_difference"]["check"] == "consistency"


def test_first_difference_semantics():
    assert gates.first_difference("a", np.array([1.0, np.nan]), np.array([1.0, np.nan])) is None
    d = gates.first_difference("a", np.array([1.0, -0.0]), np.array([1.0, 0.0]))
    assert d["index"] == 1                                      # bitwise: -0.0 != 0.0
    d = gates.first_difference("b", np.zeros(3), np.zeros(4))
    assert d["what"] == "shape"
    d = gates.first_difference("c", np.array([[1, 2], [3, 5]]), np.array([[1, 2], [3, 4]]))
    assert d["index"] == [1, 1] and d["got"] == 5 and d["expected"] == 4
