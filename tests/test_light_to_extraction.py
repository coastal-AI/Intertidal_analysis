# -*- coding: utf-8 -*-
"""experiments/light_to_extraction.py offline, on tiny synthetic data.

One synthetic Westerschelde-like world (24 x 64 px: land, an intertidal strip,
water; a legacy-cube block with no data on every date where the stores have
data): a legacy 2023 netCDF cube and production's extraction of it
(pyintertidal.marea.extract, saved as the notebooks save it), production's
record (reference_and_clouds + filter_dates + overpass times), and Zarr
stores of 2021, 2022 and 2023 in the downloader's layout. Checked: the
screening against pyintertidal.stability.reference_and_clouds, the rows
against pyintertidal.marea.extract (bitwise), the reference command, the
2023 gate (passing, and failing with attributed differences, accepted only by
the user's flag), the build (contract, read-back, v19's own loader and
extender, v14's streaming), the refusals (registration, years, other stores,
replace, radiometry), and the pre-2023 boundary levels (IOC gauge years
fetched into their own cache with the production recipe, or refused).
Production's boundary and the IOC downloader are replaced by fakes on
synthetic gauge records (no network, no tide model).

Run:  python -m pytest -q tests/test_light_to_extraction.py
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import light_to_extraction as lte  # noqa: E402
from experiments import zarr_cubes as zc  # noqa: E402

FILL = lte.FILL
EPSG = 32631
X0, Y0 = 538205.0, 5701335.0          # pixel centres of a 10 m lattice
H, W = 24, 64
STATIC = (slice(0, 4), slice(28, 37))  # no data on every legacy date, data in the stores
SITE = "escalda"


def wkt():
    from pyproj import CRS
    return CRS.from_epsg(EPSG).to_wkt("WKT1_GDAL")


def axes():
    return X0 + 10.0 * np.arange(W), Y0 - 10.0 * np.arange(H)


# ── the synthetic world ───────────────────────────────────────────────────────

def scene(rng, kind):
    """B03, B08, SCL of one scene. Land cols < 28, intertidal 28..36 (the
    waterline moves with a random tide), water >= 37. Clouds by ``kind``:
    'clean'; 'land' (frame > 5 % bad, transition clear: usable by the
    transition-zone rule); 'trans' (clouds over the transition); 'static'
    (46 bad px elsewhere plus 36 over the static block: clean and usable in
    the legacy cube, cloudy and not usable in a store unless masked)."""
    yy, xx = np.mgrid[:H, :W]
    wl = 37 - int(rng.integers(0, 10))
    wet = xx >= wl
    b03 = np.where(wet, 900, 600) + rng.integers(-40, 40, (H, W))
    b08 = np.where(wet, 300, 2500) + rng.integers(-40, 40, (H, W))
    scl = np.where(wet, 6, np.where(xx < 28, 4, 5))
    if kind == "land":
        scl[0:9, 0:13] = 9
    elif kind == "trans":
        scl[5:16, 26:39] = 8
    elif kind == "static":
        scl[10:14, 30:40] = 9          # 40 px in the transition
        scl[0:2, 0:3] = 3              # 6 px elsewhere
        scl[STATIC] = 9                # 36 px over the static block (stores only)
    return b03.astype(np.int16), b08.astype(np.int16), scl.astype(np.int16)


def year_bands(dates, kinds, seed):
    rng = np.random.default_rng(seed)
    out = [scene(rng, k) for k in kinds]
    return {b: np.stack([o[i] for o in out]) for i, b in enumerate(lte.BANDS)}


def legacy_of(bands):
    """The legacy cube's view of a store's bands: the static block without data."""
    leg = {b: v.copy() for b, v in bands.items()}
    for v in leg.values():
        v[(slice(None),) + STATIC] = FILL
    return leg


def write_nc(path, bands, dates):
    import xarray as xr

    x, y = axes()
    ds = xr.Dataset({b: (("t", "y", "x"), bands[b]) for b in lte.BANDS},
                    coords={"t": pd.DatetimeIndex(dates), "x": x, "y": y})
    ds["crs"] = xr.DataArray(np.array(b"", dtype="S1"), attrs={"crs_wkt": wkt(), "spatial_ref": wkt()})
    enc = {b: {"dtype": "int16", "_FillValue": FILL, "zlib": True} for b in lte.BANDS}
    enc["t"] = {"units": "days since 1990-01-01", "dtype": "int32"}
    ds.to_netcdf(path, encoding=enc)
    return str(path)


def registration(year, method="netcdf", checked=None, offset=None, shape=None):
    off = list(offset or lte.SITE_OFFSET[SITE])
    rec = {"method": method, "offset": off, "verified": method == "netcdf", "checked": checked,
           "delivered_shape": list(shape or lte.SITE_DELIVERED_SHAPE[SITE]), "frame_shape": [H, W],
           "why": f"synthetic {year}"}
    return {"registration": json.dumps(rec), "registration_method": method,
            "delivered_offset_px": json.dumps(off), "delivered_shape_px": json.dumps(rec["delivered_shape"])}


def write_store(root, year, bands, dates, attrs=None, band_names=lte.BANDS):
    """A complete store in the downloader's layout (zarr_cubes.FORMAT_ID)."""
    import xarray as xr

    x, y = axes()
    path = zc.store_path(SITE, year, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({b: (("t", "y", "x"), bands[b]) for b in band_names},
                    coords={"t": pd.DatetimeIndex(dates), "y": y, "x": x})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": wkt(), "spatial_ref": wkt()})
    ds.attrs = {"format": zc.FORMAT_ID, "complete": 1, "site": SITE, "year": int(year),
                "bands": json.dumps(list(band_names)), "job_id": f"j-{year}",
                "verification": json.dumps({"accepted": True, "how": "synthetic"}), **(attrs or {})}
    if tuple(band_names) == lte.BANDS:              # as download_cube_zarr --bands writes a subset
        ds.attrs.update(band_subset=json.dumps(list(band_names)), nine_band_root="cubes")
    enc = zc.encoding(ds, list(band_names), {b: FILL for b in band_names})
    ds.to_zarr(str(path), mode="w", encoding=enc, zarr_format=2, consolidated=True)
    return path


class Boundary:
    """A gauge-like boundary for v19's extender: finite levels everywhere."""

    def levels(self, t):
        return np.zeros(len(t))


# ── synthetic IOC gauges: production's chunks (2023) and the pre-2023 years ──

GAUGES = ("vlis", "brsk")
SHIFT = {"vlis": 0.05, "brsk": -0.03}          # pre-2023 records sit off production's median


def tide_m(t):
    h = (pd.DatetimeIndex(t).asi8 / 3.6e12) % 1e6
    return 1.8 * np.cos(2 * np.pi * h / 12.42)


def ioc_records(t0, t1, shift=0.0, drop=()):
    """Hourly IOC-like records (stime, slevel, sensor); none within 3 h of ``drop``."""
    t = pd.date_range(t0, t1, freq="1h")
    keep = np.ones(len(t), bool)
    for d in drop:
        keep &= np.abs((t - pd.Timestamp(d)).total_seconds()) > 3 * 3600
    t = t[keep]
    return [{"stime": s.strftime("%Y-%m-%d %H:%M:%S"), "slevel": round(float(v), 4), "sensor": "rad"}
            for s, v in zip(t, tide_m(t) + shift)]


class FakeTide:
    name = "EOT20"

    def levels(self, t):
        return np.full(len(t), 0.25)


def fake_production_boundary(prod_cache, bump=0.0):
    """Production's object as make_boundary builds it (EOT20 + gauges demeaned
    by their median); ``bump`` spoils the demeaning (not reproducible)."""
    from pyintertidal.boundary import EnsembleBoundary, GaugeBoundary, InterpolatedGauges
    from pyintertidal.gauges import despike, load_cached_ioc

    members = []
    for code in GAUGES:
        df = despike(load_cached_ioc(code, cache_dir=str(prod_cache)))
        lev = df["level_m"].to_numpy(float)
        members.append(GaugeBoundary(df["time"], lev - np.nanmedian(lev) + bump, name=code))
    return EnsembleBoundary([FakeTide(), InterpolatedGauges(members, [10.0, 20.0], name="2 gauges")],
                            name="consensus EOT20 + 2 gauges")


@pytest.fixture(autouse=True)
def gauges(world, monkeypatch):
    """The IOC downloader and production's boundary replaced by fakes."""
    state = {"calls": [], "empty": False, "drop": [], "bump": 0.0}

    def fetch(code, year, cache_dir):
        state["calls"].append((code, int(year), str(cache_dir)))
        recs = [] if state["empty"] else ioc_records(f"{year}-01-01", f"{year}-12-31 23:00", SHIFT[code],
                                                     state["drop"])
        with open(lte.gauge_chunk(cache_dir, code, year), "w") as f:
            json.dump(recs, f)

    monkeypatch.setattr(lte, "fetch_gauge_year", fetch)
    monkeypatch.setattr(lte, "production_boundary",
                        lambda site, prod_cache: fake_production_boundary(prod_cache, state["bump"]))
    return state


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import pyintertidal as pit
    from pyintertidal import marea

    base = tmp_path_factory.mktemp("lte")
    d23 = [str(d.date()) for d in pd.date_range("2023-01-02", periods=40, freq="5D")]
    k23 = (["clean"] * 4 + ["land", "trans", "static", "clean"]) * 5
    b23 = year_bands(d23, k23, 1)
    leg23 = legacy_of(b23)
    nc = write_nc(base / "legacy_2023.nc", leg23, d23)
    # production's extraction, saved as the notebooks save it (object bbox included)
    ex = marea.extract(nc)
    assert 50 < len(ex["keep"]) < H * W
    exp = base / "prod" / "marea_extract.npz"
    exp.parent.mkdir()
    np.savez_compressed(exp, Y=ex["Y"], C=ex["C"], keep=ex["keep"], dates=ex["dates"],
                        shape=np.array(ex["shape"]), sea=ex["sea"], inter=ex["inter"],
                        s_km=np.zeros(len(ex["keep"]), np.float32),
                        bbox=np.array(ex["bbox"], dtype=object), crs_wkt=np.array(ex["crs_wkt"]))
    # production's record: the screening of the legacy cube, usable dates with an overpass time
    cube = pit.SentinelCube(pit.sites.get(SITE), lte.TIME_EXTENT, water="ndwi", cache_path=nc, resolution=10)
    ref, cloud = pit.reference_and_clouds(cube, threshold=0.0, bad_classes=pit.water.BAD_CLASSES,
                                          clean_scene_max_bad=0.05, stable_threshold=0.95,
                                          transition_buffer_px=10)
    usable = pit.filter_dates(cloud, cloud_threshold=0.10)
    no_time = usable[1]
    over23 = {d: f"{d}T10:5{i % 10}:31" for i, d in enumerate(d23) if d != no_time}
    rec = [d for d in usable if d in over23]
    kinds = dict(zip(d23, k23))
    assert {kinds[d] for d in rec} >= {"clean", "land", "static"} and "trans" not in {kinds[d] for d in rec}
    with open(base / "prod" / "overpass_times.json", "w") as f:
        json.dump(over23, f)
    with open(base / "prod" / "result.json", "w") as f:
        json.dump({"record_dates": rec, "rango_marea_m": [-1.0, 1.0]}, f)
    # the stores: 2021 and 2022 light (registered by --offset-px against 2023), 2023 light (netCDF)
    light = base / "cubes_light"
    d21 = [str(d.date()) for d in pd.date_range("2021-03-01", periods=6, freq="10D")]
    d22 = [str(d.date()) for d in pd.date_range("2022-03-03", periods=6, freq="10D")]
    k_old = ["clean", "land", "trans", "static", "clean", "land"]
    b21, b22 = year_bands(d21, k_old, 2), year_bands(d22, k_old, 3)
    conf = lte.REG_CONFIRMED + " against the registered 2023 store"
    write_store(light, 2021, b21, d21, registration(2021, "expert --offset-px", conf))
    write_store(light, 2022, b22, d22, registration(2022, "expert --offset-px", conf))
    write_store(light, 2023, b23, d23, registration(2023))
    over_old = {d: f"{d}T10:4{i % 10}:11" for i, d in enumerate(d21 + d22) if d != d22[4]}
    with open(light / SITE / "overpass_times.json", "w") as f:
        json.dump(over_old, f)
    gprod = base / "gauges_prod"                    # production's IOC chunks (only read)
    gprod.mkdir()
    for code in GAUGES:
        with open(gprod / f"ioc_{code}_2023-01-01_2025-12-31.json", "w") as f:
            json.dump(ioc_records("2023-01-01", "2023-03-31"), f)
    paths = ["--work", str(base / "work"), "--extra-dir", str(base / "extra"), "--extraction", str(exp),
             "--result-json", str(base / "prod" / "result.json"),
             "--prod-overpass", str(base / "prod" / "overpass_times.json"), "--cube", nc,
             "--gauge-cache", str(base / "gauges_pre2023"), "--production-gauge-cache", str(gprod)]
    assert lte.main(["reference", SITE, *paths, "--allow-reference-mismatch"]) == 0
    return dict(base=base, nc=nc, exp=str(exp), ex=ex, ref=ref, cloud=cloud, rec=rec, d23=d23, k23=k23,
                b23=b23, d21=d21, d22=d22, b21=b21, b22=b22, light=light, over_old=over_old, paths=paths,
                no_time_old=d22[4], gprod=gprod)


# ── the two production operations ─────────────────────────────────────────────

class FakeCube:
    water = "ndwi"

    def __init__(self, bands, dates):
        self.b, self.dates = bands, dates
        self.shape = bands["SCL"].shape[1:]

    def stream(self, chunk=None):
        T = len(self.dates)
        for a in range(0, T, 3):
            sl = slice(a, min(a + 3, T))
            yield sl, {k: np.where(v[sl] == FILL, np.nan, v[sl]).astype(np.float32) for k, v in self.b.items()}


def test_constants_are_the_librarys():
    from pyintertidal import marea, water

    from experiments import v7_waterline_contour as v7

    assert tuple(marea.CLEAR) == lte.CLEAR and tuple(water.BAD_CLASSES) == lte.BAD
    assert lte.SITE_FRAME_SHAPE["ems"] == (3206, 3413) and lte.SITE_OFFSET["wadden"] == (0, 5)
    assert tuple(v7.TIME_EXTENT) == lte.TIME_EXTENT
    for s, cfg in lte.PRODUCTION_BOUNDARY.items():        # production's boundary call (v7)
        assert (cfg["n_gauges"], list(cfg["exclude"])) == (v7.SITES[s]["n_gauges"], v7.SITES[s]["judge"])


def test_screen_scene_is_reference_and_clouds_pass_2():
    from pyintertidal import stability, water

    dates = [str(d.date()) for d in pd.date_range("2023-01-01", periods=16, freq="5D")]
    kinds = ["clean", "land", "trans", "static"] * 4
    leg = legacy_of(year_bands(dates, kinds, 9))
    leg["SCL"][3, 10:12, :] = FILL                   # partial coverage on one date
    ref, cloud = stability.reference_and_clouds(
        FakeCube(leg, dates), threshold=0.0, bad_classes=water.BAD_CLASSES, clean_scene_max_bad=0.05,
        stable_threshold=0.95, transition_buffer_px=10)
    usable = set(stability.filter_dates(cloud, cloud_threshold=0.10))
    trans = ref == 0
    assert 0 < trans.sum() < trans.size
    seen = set()
    for t, d in enumerate(dates):
        raw = lte.screen_scene(leg["SCL"][t], trans)
        dec = lte.screen_scene(np.where(leg["SCL"][t] == FILL, np.nan, leg["SCL"][t]).astype(np.float32), trans)
        assert raw["cloud_pct"] == dec["cloud_pct"] == cloud[d], (d, raw, cloud[d])
        assert raw["usable"] == (d in usable)
        seen.add((raw["clean"], raw["usable"]))
    assert {(True, True), (False, True), (False, False)} <= seen     # incl. the transition-zone recovery


def test_static_mask_gives_the_legacy_screening():
    dates = [str(d.date()) for d in pd.date_range("2023-01-01", periods=8, freq="5D")]
    st = year_bands(dates, ["clean", "static"] * 4, 4)
    leg = legacy_of(st)
    ref = np.where(np.arange(W)[None] < 18, 2, np.where(np.arange(W)[None] > 46, 1, 0)).repeat(H, 0)
    ref[STATIC] = 0
    trans = ref == 0
    static = np.zeros((H, W), bool)
    static[STATIC] = True
    flips = 0
    for t in range(len(dates)):
        want = lte.screen_scene(leg["SCL"][t], trans)
        got = lte.screen_scene(st["SCL"][t], trans, static)
        assert got["cloud_pct"] == want["cloud_pct"] and got["usable"] == want["usable"]
        flips += lte.screen_scene(st["SCL"][t], trans)["usable"] != want["usable"]
    assert flips == 4                                   # without the mask the 'static' scenes flip


def test_extraction_row_is_marea_extract_bitwise(world):
    ex, b = world["ex"], legacy_of(world["b23"])
    rows, cols = np.divmod(ex["keep"], W)
    b["B08"][3].flat[ex["keep"][0]] = -b["B03"][3].flat[ex["keep"][0]]      # B03 + B08 = 0 -> NaN
    rng = np.random.default_rng(7)
    for t in range(len(world["d23"])):
        y, c = lte.extraction_row(b["B03"][t], b["B08"][t], b["SCL"][t], rows, cols)
        if t == 3:
            assert np.isnan(y[0])
            continue
        assert y.dtype == np.float32 and c.dtype == bool
        assert np.array_equal(y.view(np.uint32)[~np.isnan(y)], ex["Y"][t].view(np.uint32)[~np.isnan(y)])
        assert np.array_equal(np.isnan(y), np.isnan(ex["Y"][t])) and np.array_equal(c, ex["C"][t])
    g = rng.integers(-900, 3000, (5, 7)).astype(np.int16)
    n = rng.integers(-900, 3000, (5, 7)).astype(np.int16)
    g[1, 1], n[2, 2] = FILL, FILL
    y, _ = lte.extraction_row(g, n, np.full((5, 7), 4, np.int16), *np.divmod(np.arange(35), 7))
    assert np.isnan(y[8]) and np.isnan(y[16]) and np.isfinite(y[0])


# ── reference, gate, build ────────────────────────────────────────────────────

def test_reference_reproduces_production(world):
    J = json.load(open(world["base"] / "work" / f"{SITE}_reference.json"))
    assert J["ok"] and J["gates"]["screen_scene_equals_reference_and_clouds"]["ok"]
    assert J["gates"]["usable_with_overpass_equals_record_dates"]["ok"]
    assert J["gates"]["classes_equal_production_print"].get("overridden")       # tiny world, recorded
    assert J["static_nodata"]["px"] == 36 and J["static_nodata"]["keep_px_inside"] == 0
    with np.load(world["base"] / "work" / f"{SITE}_reference.npz") as z:
        assert np.array_equal(z["reference_map"], world["ref"])
        st = np.zeros((H, W), bool)
        st[STATIC] = True
        assert np.array_equal(z["static_nodata"], st)
    assert J["screening"]["cloud_pct_screen_scene"] == {k: float(v) for k, v in world["cloud"].items()}


def test_gate_2023_passes(world):
    out = world["base"] / "work" / f"{SITE}_gate_2023.json"
    rc = lte.main(["gate", SITE, *world["paths"], "--roots", str(world["light"])])
    g = json.load(open(out))
    assert rc == 0 and g["ok"] and g["strict_ok"]
    assert g["store"]["kind"] == "light" and g["store"]["registration"]["ok"]
    e = g["extraction"]
    assert e["rows_compared"] == 40 and e["Y_equal"] and e["C_equal"] and e["bit_identical"] > 0
    s = g["screening"]
    assert s["equal_record_dates"] and s["screened"] == world["rec"]
    assert s["vs_legacy_reproduction"]["differ"] == 0 and s["usable_without_overpass"]
    assert sum(v["static_bad_masked"] for v in s["per_date"].values()) == 5 * 36
    dn = g["legacy_dn"]                     # the digital numbers behind Y, against the legacy cube
    assert dn["equal"] and dn["valid_both"] == 2 * 40 * len(world["ex"]["keep"]) and dn["differ"] == 0
    assert g["decision"].startswith("passed") and open(world["base"] / "work" / ".gitignore").read().endswith("*\n")


def test_build_end_to_end_and_v19_contract(world, monkeypatch):
    from experiments import v14_scene_count_curves as v14
    from experiments import v19_fixed_config_tables as v19

    lte.main(["gate", SITE, *world["paths"], "--roots", str(world["light"])])
    extra = world["base"] / "extra"
    if extra.exists():
        shutil.rmtree(extra)
    rc = lte.main(["build", SITE, "--years", "2021-2022", *world["paths"], "--roots", str(world["light"])])
    assert rc == 0
    npz, side = extra / f"{SITE}_extract_2021-2022.npz", extra / f"{SITE}_extract_2021-2022.json"
    ex = world["ex"]
    dates = world["d21"] + world["d22"]
    with np.load(npz, allow_pickle=False) as z:
        assert set(z.files) == {"Y", "C", "keep", "dates", "shape", "sea", "inter"}
        assert z["Y"].dtype == np.float32 and z["C"].dtype == bool and z["dates"].dtype.str == "<U10"
        assert z["dates"].tolist() == dates and np.array_equal(z["keep"], ex["keep"])
        assert z["keep"].dtype == np.asarray(ex["keep"]).dtype and tuple(z["shape"]) == (H, W)
        assert np.array_equal(z["sea"], ex["sea"]) and np.array_equal(z["inter"], ex["inter"])
        Y, C = z["Y"], z["C"]
    rows, cols = np.divmod(ex["keep"], W)
    allb = {b: np.concatenate([world["b21"][b], world["b22"][b]]) for b in lte.BANDS}
    for t in range(len(dates)):
        g = allb["B03"][t][rows, cols].astype(np.float32)
        n = allb["B08"][t][rows, cols].astype(np.float32)
        want = np.where(g + n != 0, (g - n) / (g + n), np.nan).astype(np.float32)
        assert np.array_equal(Y[t], want, equal_nan=True)
        assert np.array_equal(C[t], np.isin(allb["SCL"][t][rows, cols], (4, 5, 6, 7)))
    S = json.load(open(side))
    assert S["builder"] == lte.BUILDER and S["gate_2023"]["ok"] is True and "usable" not in S
    assert S["overpass"] == {d: v for d, v in world["over_old"].items()}
    assert S["dates_without_overpass"] == [world["no_time_old"]]
    with np.load(world["base"] / "work" / f"{SITE}_reference.npz") as z:
        trans, static = z["reference_map"] == 0, z["static_nodata"]
    for t, d in enumerate(dates):
        assert S["cloud_pct"][d] == lte.screen_scene(allb["SCL"][t], trans, static)["cloud_pct"]
    assert S["extraction"]["readback"]["ok"] and S["extraction"]["not_copied"] == ["bbox", "crs_wkt", "s_km"]
    rad = S["radiometry_probe"]
    assert rad["suspect"] == [] and rad["gate_year"] is not None and rad["gate_year_b08_low_share"] > 0.01
    assert rad["groups"]["2021 unrecorded"]["decided"] and rad["pre_04_dates"] == {"2021": 0, "2022": 0}
    lev = S["levels_pre2023"]                       # every screened scene has both gauges
    assert lev["screened_members"]["all"] == len([d for d in dates if d in world["over_old"]
                                                  and S["cloud_pct"][d] / 100.0 <= 0.10])
    assert "EOT20 ALONE" in lev["v19_note"] and S["n_target"]["needed_before_2023"] == 125 - len(world["rec"])
    assert open(extra / ".gitignore").read().splitlines()[-1] == "*"
    # v14's streaming reader and v19's own loader / extender read it as the contract says (the tiny
    # world's map stands for production's: its class counts are the ones v19 checks)
    monkeypatch.setitem(v19.PRODUCTION_REFERENCE_CLASSES, SITE,
                        {int(k): v for k, v in S["reference_map"]["classes"].items()})
    ys = v14.stream_columns(str(npz), "Y", [0, 5, 11], [np.arange(len(ex["keep"]))])[0]
    assert np.array_equal(ys, Y[[0, 5, 11]], equal_nan=True)
    st, info = v19.load_extra_store(str(npz), ex["keep"], (H, W), ex["dates"], site=SITE)
    assert st is not None, info
    usable = {d for d in dates if S["cloud_pct"][d] / 100.0 <= 0.10}
    assert st["usable"] == usable and info["dates_before_main"] == 12 and info["gate_2023"]["ok"] is True
    main = world["rec"]
    pos = {d: i for i, d in enumerate(ex["dates"])}
    S0 = {"dates": np.array(main), "src": np.zeros(len(main), np.int64),
          "row": np.array([pos[d] for d in main], np.int64),
          "t": pd.DatetimeIndex([f"{d}T11:00:00" for d in main]), "h0": np.zeros(len(main))}
    out, infos, note = v19.extend_with_extra(S0, SITE, str(extra), ex["keep"], (H, W), ex["dates"], Boundary(),
                                             "production")
    pre = sorted(d for d in usable if d in world["over_old"])
    assert out["dates"].tolist() == pre + main and infos[0]["used"] and infos[0]["screened"] == len(pre)
    assert assert_rows(out, npz, Y)
    assert lte.main(["check", SITE, *world["paths"]]) == 0
    monkeypatch.setitem(v19.PRODUCTION_REFERENCE_CLASSES, SITE, {0: 1, 1: 2, 2: 3})
    assert lte.main(["check", SITE, *world["paths"]]) == 2         # check runs v19's map check too


def assert_rows(out, npz, Y):
    """The extended list points into the extra store at the rows of its dates."""
    with np.load(npz) as z:
        d = z["dates"].tolist()
    for date, src, row in zip(out["dates"], out["src"], out["row"]):
        if src == 1 and d[row] != date:
            return False
    return True


def test_build_refusals_and_replace(world):
    lte.main(["gate", SITE, *world["paths"], "--roots", str(world["light"])])
    extra = world["base"] / "extra"
    if extra.exists():
        shutil.rmtree(extra)
    args = ["build", SITE, *world["paths"], "--roots", str(world["light"])]
    assert lte.main(args + ["--years", "2022-2023"]) == 3              # reaches the main extraction
    assert lte.main(args + ["--years", "2022"]) == 0
    assert lte.main(args + ["--years", "2022"]) == 3                   # the same store is there
    assert lte.main(args + ["--years", "2021-2022"]) == 3              # overlaps it
    assert lte.main(args + ["--years", "2021-2022", "--replace"]) == 0
    names = sorted(os.listdir(extra))
    assert f"{SITE}_extract_2021-2022.npz" in names and f"{SITE}_extract_2022-2022.npz" not in names
    assert any(n.startswith(f"{SITE}_extract_2022-2022.npz.superseded-") for n in names)
    # another builder's store is never moved, overlapping or not
    other = extra / f"{SITE}_extract_2018-2018.npz"
    np.savez(other, dates=np.array(["2018-05-01"]))
    with open(extra / f"{SITE}_extract_2018-2018.json", "w") as f:
        json.dump({"builder": "experiments/download_cube_stac.py"}, f)
    assert lte.main(args + ["--years", "2021-2022", "--replace"]) == 3
    np.savez(other, dates=np.array(["2021-03-01"]))
    assert lte.main(args + ["--years", "2021-2022", "--replace", "--allow-other-stores"]) == 3
    assert os.path.exists(other)


def test_registration_rule(world, tmp_path):
    lte.main(["gate", SITE, *world["paths"], "--roots", str(world["light"])])
    light = tmp_path / "cubes_light"
    shutil.copytree(world["light"], light)
    b, d = world["b21"], world["d21"]
    args = ["build", SITE, "--years", "2021", *world["paths"], "--roots", str(light),
            "--extra-dir", str(tmp_path / "extra")]
    write_store(light, 2021, b, d, registration(2021, "expert --offset-px",
                                                 "NOT confirmed by the composites against the registered 2023 store"))
    assert lte.main(args) == 3
    assert lte.main(args + ["--allow-unconfirmed-registration"]) == 0
    S = json.load(open(tmp_path / "extra" / f"{SITE}_extract_2021-2021.json"))
    assert "--allow-unconfirmed-registration" in S["stores"]["2021"]["registration_override"]
    write_store(light, 2021, b, d, registration(2021, "expert --offset-px", lte.REG_CONFIRMED, offset=(6, 1)))
    assert lte.main(args + ["--replace"]) == 3                       # not the site's offset
    write_store(light, 2021, b, d, registration(2021, "expert --offset-px", lte.REG_CONFIRMED,
                                                 shape=(1536, 2048)))
    assert lte.main(args + ["--replace"]) == 3                       # another delivered shape
    reg = lte.registration_status(json.loads(json.dumps(registration(2023))), SITE, (H, W))
    assert reg["ok"] and reg["method"] == "netcdf"
    wrong = {b_: v.astype(np.int32) for b_, v in b.items()}
    write_store(light, 2021, wrong, d, registration(2021, "expert --offset-px", lte.REG_CONFIRMED))
    assert lte.main(args + ["--replace", "--allow-unconfirmed-registration"]) == 3     # not int16


def test_light_path_gate_report(world, tmp_path):
    lte.main(["gate", SITE, *world["paths"], "--roots", str(world["light"])])
    rep = tmp_path / "2023.vs_nine_band.json"
    args = ["build", SITE, "--years", "2021", *world["paths"], "--roots", str(world["light"]),
            "--light-check", str(rep)]
    with open(rep, "w") as f:
        json.dump({"passed": False, "why": "SCL 12 differing pixel-dates", "comparison": {"verdict": "differs"}}, f)
    assert lte.main(args + ["--extra-dir", str(tmp_path / "a")]) == 3
    assert lte.main(args + ["--extra-dir", str(tmp_path / "b"), "--allow-unconfirmed-registration"]) == 0
    with open(rep, "w") as f:
        json.dump({"passed": True, "why": "146 = 146 dates, bitwise", "comparison": {"verdict": "identical"}}, f)
    assert lte.main(args + ["--extra-dir", str(tmp_path / "c")]) == 0
    S = json.load(open(tmp_path / "c" / f"{SITE}_extract_2021-2021.json"))
    assert S["light_path_check"]["passed"] is True and S["light_path_check"]["verdict"] == "identical"
    assert S["stores"]["2021"]["band_subset"] == list(lte.BANDS) and S["gate_store_kind_note"] is None


def test_radiometry_gate_refuses_an_offset_year(world, tmp_path):
    """+1000 left in 2021 (one baseline group): refused, unless --accept-radiometry (recorded)."""
    lte.main(["gate", SITE, *world["paths"], "--roots", str(world["light"])])
    light = tmp_path / "cubes_light"
    shutil.copytree(world["light"], light)
    b = {k: (v + 1000 if k != "SCL" else v).astype(np.int16) for k, v in world["b21"].items()}
    write_store(light, 2021, b, world["d21"], {**registration(2021, "expert --offset-px", lte.REG_CONFIRMED),
                                               "input_baselines": json.dumps({d: ["03.01"] for d in world["d21"]})})
    args = ["build", SITE, "--years", "2021-2022", *world["paths"], "--roots", str(light)]
    assert lte.main(args + ["--extra-dir", str(tmp_path / "x")]) == 3
    assert os.listdir(tmp_path / "x") == [".gitignore"]                    # nothing kept
    assert lte.main(args + ["--extra-dir", str(tmp_path / "y"), "--accept-radiometry"]) == 0
    S = json.load(open(tmp_path / "y" / f"{SITE}_extract_2021-2022.json"))
    p = S["radiometry_probe"]
    assert p["suspect"] == ["2021 03.01", "2021 all"] and p["suspect_years"] == ["2021"]
    assert p["groups"]["2021 all"]["keep_b08_low_share"] == 0.0 and len(p["groups"]["2021 all"]["suspect"]) == 2
    assert p["years"]["2021"] - p["gate_year"] > 900 and p["pre_04_dates"]["2021"] == 6
    assert abs(p["years"]["2022"] - p["gate_year"]) < lte.OFFSET_SUSPECT_DN and "--accept" in p["accepted"]["by"]


def test_gate_attributes_differences(world, tmp_path):
    light = tmp_path / "cubes_light"
    shutil.copytree(world["light"], light)
    b = {k: v.copy() for k, v in world["b23"].items()}
    k0 = int(world["ex"]["keep"][0])
    b["B03"][0].flat[k0] += 7                                       # one value
    b["B08"][1].flat[k0] = FILL                                     # no data only in the store
    write_store(light, 2023, b, world["d23"], registration(2023))
    gp = tmp_path / "gate.json"
    paths = world["paths"] + ["--roots", str(light)]
    assert lte.main(["gate", SITE, *paths, "--out", str(gp)]) == 2
    g = json.load(open(gp))
    e = g["extraction"]
    assert not g["ok"] and e["value_differs"] == 1 and e["nan_only_store"] == 1 and e["c_differs"] == 0
    assert sorted(e["worst_dates"]) == world["d23"][:2] and g["screening"]["equal_record_dates"]
    assert e["shares_pct"]["Y_value_differs_of_finite_both"] > 0 and g["only_extraction_differences"]
    dn = g["legacy_dn"]
    assert dn["differ"] == 1 and dn["per_band"]["B03"]["differ"] == 1 and dn["max_abs_dn"] == 7
    assert dn["nodata_only_store"] == 1 and dn["nodata_only_legacy"] == 0 and dn["differ_pct_of_valid_both"] > 0
    assert "the user's decision" in g["decision"] and "accepted" not in g
    bargs = ["build", SITE, "--years", "2021-2022", *world["paths"], "--roots", str(world["light"]),
             "--gate", str(gp), "--extra-dir", str(tmp_path / "x1")]
    assert lte.main(bargs) == 3
    assert lte.main(bargs + ["--accept-gate-differences"]) == 0
    S = json.load(open(tmp_path / "x1" / f"{SITE}_extract_2021-2022.json"))
    assert S["gate_2023"]["ok"] is True and S["gate_2023"]["strict_ok"] is False
    assert "--accept-gate-differences" in S["gate_2023"]["accepted"]["by"]
    assert S["gate_2023"]["accepted"]["legacy_dn"]["max_abs_dn"] == 7
    # the same decision taken at gate time, recorded in the report: build then honours it
    assert lte.main(["gate", SITE, *paths, "--out", str(gp), "--accept-gate-differences"]) == 0
    g = json.load(open(gp))
    assert g["ok"] is False and g["accepted"]["by"].startswith("light_to_extraction gate --accept-gate-differences")
    assert lte.main(bargs[:-1] + [str(tmp_path / "x3")]) == 0
    S = json.load(open(tmp_path / "x3" / f"{SITE}_extract_2021-2022.json"))
    assert S["gate_2023"]["ok"] is True and S["gate_2023"]["accepted"]["by"].startswith("light_to_extraction gate")
    # a screening difference is never accepted by that flag
    b = {k: v.copy() for k, v in world["b23"].items()}
    t = world["d23"].index(world["rec"][0])
    b["SCL"][t][5:16, 26:39] = 8                                    # clouds over the transition
    write_store(light, 2023, b, world["d23"], registration(2023))
    assert lte.main(["gate", SITE, *paths, "--out", str(gp)]) == 2
    g = json.load(open(gp))
    assert not g["screening"]["equal_record_dates"] and g["screening"]["only_record"] == [world["rec"][0]]
    assert lte.main(["gate", SITE, *paths, "--out", str(gp), "--accept-gate-differences"]) == 2
    assert "accepted" not in json.load(open(gp)) and "no flag accepts it" in json.load(open(gp))["decision"]
    assert lte.main(bargs + ["--accept-gate-differences"]) == 3
    assert lte.main(bargs[:-2] + ["--extra-dir", str(tmp_path / "x2"), "--allow-failed-gate"]) == 0
    S = json.load(open(tmp_path / "x2" / f"{SITE}_extract_2021-2022.json"))
    assert S["gate_2023"]["ok"] is False


def test_build_refuses_a_stale_gate(world, tmp_path):
    light = tmp_path / "cubes_light"
    shutil.copytree(world["light"], light)
    gp = tmp_path / "gate.json"
    build = ["build", SITE, "--years", "2021", *world["paths"], "--roots", str(world["light"]),
             "--gate", str(gp), "--extra-dir", str(tmp_path / "x")]
    assert lte.main(["gate", SITE, *world["paths"], "--roots", str(light), "--out", str(gp)]) == 0
    g = json.load(open(gp))
    g["code"]["version"] = lte.CODE_VERSION + 1
    with open(gp, "w") as f:
        json.dump(g, f)
    assert lte.main(build) == 3                                      # another code version
    assert lte.main(["gate", SITE, *world["paths"], "--roots", str(light), "--out", str(gp)]) == 0
    shutil.copy(light / SITE / "2023.zarr" / ".zmetadata", tmp_path / "zm")
    with open(light / SITE / "2023.zarr" / ".zmetadata", "a") as f:
        f.write(" ")                                                 # the store rewritten since
    assert lte.main(build) == 3
    shutil.copy(tmp_path / "zm", light / SITE / "2023.zarr" / ".zmetadata")
    assert lte.main(["gate", SITE, *world["paths"], "--roots", str(light), "--out", str(gp)]) == 0
    assert lte.main(build) == 0


def test_pre2023_levels_fetch_the_gauges_or_stop(world, gauges, tmp_path):
    """The gauge years are fetched once into their own cache (production's is
    only read), production's member is reproduced and its median kept as the
    datum; a missing, empty or production cache refuses; a scene whose level
    would rest on EOT20 alone refuses unless --accept-gauge-gaps."""
    lte.main(["gate", SITE, *world["paths"], "--roots", str(world["light"])])
    before = {p.name: p.read_bytes() for p in world["gprod"].iterdir()}
    pre = tmp_path / "g"
    args = ["build", SITE, "--years", "2021", *world["paths"], "--roots", str(world["light"]), "--gauge-cache", str(pre)]
    assert lte.main(args + ["--extra-dir", str(tmp_path / "a"), "--no-fetch"]) == 3 and not gauges["calls"]
    assert lte.main(args + ["--extra-dir", str(tmp_path / "b"), "--gauge-cache", str(world["gprod"])]) == 3
    assert lte.main(args + ["--extra-dir", str(tmp_path / "c")]) == 0
    assert sorted(c[:2] for c in gauges["calls"]) == [("brsk", 2021), ("vlis", 2021)]
    assert lte.main(args + ["--extra-dir", str(tmp_path / "d")]) == 0 and len(gauges["calls"]) == 2   # cached
    assert {p.name: p.read_bytes() for p in world["gprod"].iterdir()} == before                     # only read
    lev = json.load(open(tmp_path / "c" / f"{SITE}_extract_2021-2021.json"))["levels_pre2023"]
    gi = lev["gauges"]["vlis"]
    assert gi["production_member_reproduced"] and abs(gi["pre2023_minus_production_median_m"] - SHIFT["vlis"]) < 0.03
    d = world["d21"][0]
    r = lev["per_date"][d]
    w = np.array([1 / 10.0 ** 2, 1 / 20.0 ** 2]) / (1 / 10.0 ** 2 + 1 / 20.0 ** 2)
    assert r["members"] == "all" and r["EOT20"] == 0.25
    assert r["h_B"] == pytest.approx(np.mean([0.25, w[0] * r["vlis"] + w[1] * r["brsk"]]), abs=1e-12)
    want = tide_m([r["t"]])[0] + SHIFT["vlis"] - gi["production_median_m"]      # production's datum
    assert r["vlis"] == pytest.approx(want, abs=0.07)
    # no production member reproduced: the demeaning constant is unknown
    gauges["bump"] = 0.01
    assert lte.main(args + ["--extra-dir", str(tmp_path / "e")]) == 3
    gauges["bump"] = 0.0
    # an outage leaves []: moved aside (kept), fetched once more, still empty -> refused
    gauges["empty"] = True
    assert lte.main(args + ["--extra-dir", str(tmp_path / "f"), "--gauge-cache", str(tmp_path / "g2")]) == 3
    assert len([n for n in os.listdir(tmp_path / "g2") if ".incomplete-" in n]) == 2
    gauges["empty"] = False
    # both gauges silent at a usable scene's overpass: EOT20 alone, refused unless the user takes it
    gauges["drop"] = [world["over_old"][d]]
    gap = args + ["--gauge-cache", str(tmp_path / "g3")]
    assert lte.main(gap + ["--extra-dir", str(tmp_path / "h")]) == 3
    assert lte.main(gap + ["--extra-dir", str(tmp_path / "i"), "--accept-gauge-gaps"]) == 0
    lev = json.load(open(tmp_path / "i" / f"{SITE}_extract_2021-2021.json"))["levels_pre2023"]
    assert lev["screened_eot20_alone"] == [d] and lev["per_date"][d]["h_B"] == 0.25
    assert "--accept-gauge-gaps" in lev["accepted_eot20_alone"]["by"]


def test_helpers():
    assert lte.parse_years("2020-2022") == [2020, 2021, 2022] and lte.parse_years("2019") == [2019]
    assert lte.parse_years("2021,2019,2020") == [2019, 2020, 2021]
    with pytest.raises(Exception):
        lte.parse_years("2019,2021")
    assert lte.store_kind(lte.BANDS) == "light" and lte.store_kind(lte.NINE_BANDS) == "9-band"
    assert lte.iso_instant("2021-03-01T10:41:11Z") == "2021-03-01T10:41:11"


def test_no_reserved_keyword_in_my_files():
    token = "allow_" + "reserved"
    for f in (lte.__file__, __file__):
        assert token not in open(f, encoding="utf-8").read()
