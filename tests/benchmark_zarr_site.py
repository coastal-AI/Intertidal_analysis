"""A synthetic validation-v2 site in Zarr stores (a helper, not a test module).

Three yearly stores (2023-2025, all nine bands, 22 dates a year) written by
``download_cube_zarr.write_store`` on a 48 x 64 frame of 10 m pixels, as the
real stores are: a sea on the west edge, a channel, tidal flats rising
eastward whose wet/dry state follows :class:`tests.benchmark_site.FakeTide`,
land on the east edge. The 20 m bands (B8A, B11, B12, SCL, CLD) are exact
2 x 2 replicas on lattice phase (1, 0). Clouds (SCL 8/9, bright, CLD high),
shadows (SCL 3), dark wet mud labelled 2, snow 11, a no-data patch, one
fully cloudy scene; 2025 is delivered with +1000 in the spectral values.
One judge stands on the flats. :func:`fake_ocm` is a deterministic stand-in
for OmniCloudMask (thick cloud where green is bright, a sparse thin-cloud
lattice), recording what it received; :func:`fake_pixel_tides` stands in for
eo-tides in the level-sources step (no tide model file is read) and
:func:`write_d_truth` for the v2 elevation truth (D_truth and the windows it
registers).
"""
import dataclasses
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

from experiments import zarr_cubes as zc
from experiments.benchmark import config, domain, obs, package
from tests.benchmark_site import FakeTide

FILL = -32768
SITE = "escalda"           # a real site name (SITE_ID, seeds); every input is synthetic
H, W = 48, 64
X_EDGE, Y_EDGE = 500_000.0, 4_800_010.0
PHASE = (1, 0)
YEARS = (2023, 2024, 2025)
N_PER_YEAR = 22
SMALL = {"MIN_INTERTIDAL": 40, "MIN_FINITE_S": 60, "MIN_PX_BAND": 60}
#: package geometry scaled to the tiny frame (FROZEN: 300 m, 300 m, 1 km)
SMALL_PKG = {"S0_KEEP_DILATION_M": 30.0, "OMEGA_RING_M": 30.0, "REGION_PAD_M": 20.0,
             "SPACE_MARGIN_GB": 0.0}
SMALL_NB = {"r0_m": 30.0, "pad_m": 20.0, "step_m": 10.0, "min_keep_px": 10, "cap_m": 100.0,
            "n_nearest": 200}
STATION_PX = (20.4, 30.3)
KEYS = ("C1.floor0.ndwi.D1", "C1.raw.ndwi.D1", "C1.floor0.mndwi.D1", "C1.floor0.ndwi20.D1",
        "C1.floor0.awei_sh.D1", "C1.floor0.ndwi.D5", "C1d.floor0.ndwi.D1", "C0.floor0.ndwi.D1",
        "C4.floor0.ndwi.D1", "C3.floor0.ndwi.D1", "C11.floor0.ndwi.D1", "C11.floor0.mndwi.D1",
        "C1.floor0.nir.D1", "C1.floor0.ndwi.D2", "C1.floor0.ndwi.D3", "C1.floor0.mndwi.D3",
        "C4.floor0.ndwi.D3", "C11.floor0.ndwi.D3")


def wkt():
    from pyproj import CRS
    return CRS.from_epsg(32631).to_wkt("WKT1_GDAL")


def coords():
    return X_EDGE + 5.0 + 10.0 * np.arange(W), Y_EDGE - 5.0 - 10.0 * np.arange(H)


def dates(year):
    return pd.date_range(f"{year}-01-05", periods=N_PER_YEAR, freq="16D")


def instants(year):
    d = dates(year)
    sec = (np.arange(len(d)) * 37 + year) % 600
    return d + pd.Timedelta(hours=10, minutes=48) + pd.to_timedelta(sec, "s")


def elevation():
    cols = np.arange(W)
    z = np.empty((H, W))
    z[:] = -1.7 + 3.4 * (cols - 6) / 48.0
    z += 0.1 * np.sin(np.arange(H) / 3.0)[:, None]
    z[:, :6] = -4.0
    z[22:26, :50] = -4.0
    z[:, 56:] = 5.0
    return z


def _rep20(blocks):
    return obs.expand20(blocks, PHASE, (H, W))


def _take20(a):
    return obs.block_take20(a, PHASE)


def year_bands(year, seed):
    """Raw int16 bands (T, H, W) of one year, and the planted wet state."""
    rng = np.random.default_rng(seed)
    T = N_PER_YEAR
    t = instants(year)
    h = FakeTide.value(t.asi8)
    z = elevation()
    wet = h[:, None, None] > z[None] + rng.normal(0.0, 0.05, (T, H, W))
    land = np.broadcast_to(z >= 4.9, wet.shape)
    wet20 = _take20(wet)
    land20 = _take20(np.broadcast_to(land, wet.shape))
    Hb, Wb = wet20.shape[-2:]

    def mix(shape, w_rng, d_rng, l_rng, wmask, lmask):
        out = np.where(wmask, rng.integers(*w_rng, shape), rng.integers(*d_rng, shape))
        return np.where(lmask, rng.integers(*l_rng, shape), out)

    d = {}
    for b, wr, dr, lr in (("B02", (300, 600), (600, 1000), (400, 800)),
                          ("B03", (500, 900), (700, 1100), (600, 1000)),
                          ("B04", (300, 700), (800, 1300), (500, 900)),
                          ("B08", (40, 300), (1500, 2500), (2800, 3800))):
        d[b] = mix((T, H, W), wr, dr, lr, wet, land)
    for b, wr, dr, lr in (("B8A", (30, 250), (1400, 2400), (2700, 3700)),
                          ("B11", (20, 200), (1200, 2200), (1500, 2500)),
                          ("B12", (10, 150), (800, 1600), (900, 1700))):
        d[b] = _rep20(mix((T, Hb, Wb), wr, dr, lr, wet20, land20))
    scl20 = np.where(wet20, 6, rng.choice([4, 5], (T, Hb, Wb)))
    scl20 = np.where(land20, 4, scl20)
    cld20 = rng.integers(0, 15, (T, Hb, Wb))
    u = rng.random((T, Hb, Wb))
    scl20 = np.where((u < 0.04) & wet20, 2, scl20)               # dark wet mud
    scl20 = np.where((u > 0.995), 11, scl20)
    cloud20 = np.zeros((T, Hb, Wb), bool)
    for k in range(T):
        for _ in range(rng.integers(0, 3)):
            r, c = rng.integers(0, Hb - 4), rng.integers(0, Wb - 6)
            cloud20[k, r:r + rng.integers(2, 6), c:c + rng.integers(3, 9)] = True
    cloud20[7] = True                                            # a fully cloudy scene
    scl20 = np.where(cloud20, rng.choice([8, 9], (T, Hb, Wb)), scl20)
    cld20 = np.where(cloud20, rng.integers(60, 100, (T, Hb, Wb)), cld20)
    shadow20 = np.zeros((T, Hb, Wb), bool)
    shadow20[3, 5:8, 10:16] = True
    scl20 = np.where(shadow20, 3, scl20)
    cld20[11, 2:6, 2:6] = 30                                     # CLD between the two p
    d["SCL"] = _rep20(scl20)
    d["CLD"] = _rep20(cld20)
    cloud = _rep20(cloud20)
    for b in ("B02", "B03", "B04", "B08"):
        d[b] = np.where(cloud, rng.integers(3000, 4500, (T, H, W)), d[b])
    for b in ("B8A", "B11", "B12"):
        d[b] = np.where(cloud, _rep20(rng.integers(2800, 4000, (T, Hb, Wb))), d[b])
    shadow = _rep20(shadow20)
    for b in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12"):
        d[b] = np.where(shadow, (0.4 * d[b]).astype(np.int64), d[b])
    d = {b: a.astype(np.int16) for b, a in d.items()}
    d["B03"][2, 10, 10] = FILL
    d["B08"][4, 30, 20] = FILL
    d["SCL"][5, 1:5, :6] = FILL                                  # a no-data patch (block-aligned)
    d["CLD"][5, 1:5, :6] = FILL
    if year == 2025:
        for b in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12"):
            m = d[b] != FILL
            d[b][m] = (d[b][m] + 1000).astype(np.int16)
    return d, wet


def write_year(root, year, seed):
    import xarray as xr
    from experiments import download_cube_zarr as dcz

    x, y = coords()
    data, wet = year_bands(year, seed)
    ds = xr.Dataset({b: (("t", "y", "x"), data[b], {"long_name": b}) for b in zc.BANDS},
                    coords={"t": dates(year), "y": y, "x": x})
    w = wkt()
    ds["crs"] = xr.DataArray(np.int32(0), attrs={
        "crs_wkt": w, "spatial_ref": w, "GeoTransform": f"{X_EDGE} 10 0 {Y_EDGE} 0 -10"})
    out = zc.store_path(SITE, year, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    dcz.write_store(ds, {b: FILL for b in zc.BANDS}, out, {"site": SITE},
                    log=lambda m: None, verify=False)
    return data, wet


def station_lonlat(row, col):
    import pyproj
    inv = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(32631), "EPSG:4326", always_xy=True)
    x, y = coords()
    lon, lat = inv.transform(x[0] + 10.0 * col, y[0] - 10.0 * row)
    return float(lon), float(lat)


class FakeOCM:
    """OmniCloudMask stand-in: class 1 (thick cloud) where green > 2500 DN,
    class 2 (thin cloud) on a sparse diagonal lattice, else 0; records the
    inputs it received."""

    description = "tests.benchmark_zarr_site.FakeOCM"

    def __init__(self):
        self.calls = []

    def __call__(self, stack):
        stack = np.asarray(stack)
        self.calls.append(stack.copy())
        h, w = stack.shape[1:]
        out = np.zeros((1, h, w), np.uint8)
        rr, cc = np.mgrid[0:h, 0:w]
        out[0][(rr + cc) % 13 == 0] = 2
        out[0][stack[1] > 2500] = 1
        return out


def fake_ocm_classes(raw_scene, offset):
    """What :class:`FakeOCM` gives for one full scene (direct reference)."""
    stack, nodata = obs.ocm_input(raw_scene, offset)
    out = FakeOCM()(stack)[0]
    out[nodata] = obs.OCM_NODATA_U8
    return out


def make_inputs(root, seed=11):
    """Stores, instants, offsets, stations; returns a dict of paths and arrays."""
    os.makedirs(root, exist_ok=True)
    cubes = os.path.join(root, "cubes")
    data, wet, inst, offs = {}, {}, {}, {}
    for k, y in enumerate(YEARS):
        data[y], wet[y] = write_year(cubes, y, seed + k)
        for d, t in zip(dates(y), instants(y)):
            inst[str(d.date())] = t.isoformat()
            offs[str(d.date())] = 1000 if y == 2025 else 0
    no_inst = str(dates(2024)[6].date())
    inst[no_inst] = None
    nan_instant = pd.Timestamp(inst[str(dates(2023)[9].date())])
    lon, lat = station_lonlat(*STATION_PX)
    stations = os.path.join(root, "stations.json")
    with open(stations, "w", encoding="utf-8") as f:
        json.dump([{"code": "st_flat", "name": "st flat", "provider": "synthetic", "lon": lon,
                    "lat": lat, "datum": "none", "sampling_min": 10, "file": "st_flat.csv",
                    "period": "2023-2025", "site": SITE, "in_boundary": False}], f)
    return {"root": root, "cubes": cubes, "data": data, "wet": wet, "instants": inst,
            "offsets": offs, "no_instant": no_inst, "nan_instant": nan_instant,
            "stations": stations}


def site2_for(inp, out_root, **kw):
    cfg = dataclasses.replace(
        config.site_config(SITE, 10),
        boundary=types.MappingProxyType({"kind": "fake", "name": "fake tide"}),
        mouth_side="west", judges=("st_flat",),
        extra=types.MappingProxyType({"stations_json": inp["stations"]}))
    args = dict(site=SITE, cfg=cfg, root=str(out_root).replace("\\", "/"),
                cubes_root=inp["cubes"], instants=inp["instants"], offsets=inp["offsets"],
                lattice="measure", block_rows=20, keys=KEYS, cache_mb=64)
    args.update(kw)
    return package.V2Site(**args)


def small(mp):
    for k, v in SMALL.items():
        mp.setattr(domain, k, v)
    for k, v in SMALL_PKG.items():
        mp.setattr(package, k, v)
    mp.setattr(package, "NB_RULE", dict(SMALL_NB))


def write_d_truth(site2, flat, windows=None):
    """A stand-in v2 elevation truth: D_truth and the windows it registers
    (default every Campaign-1 elevation window, as freeze_dom requires)."""
    os.makedirs(site2.truth_dir, exist_ok=True)
    from experiments.benchmark import product
    product.save_npz(os.path.join(site2.truth_dir, package.TRUTH_ELEV_V2),
                     {"d_truth_union": np.asarray(flat, np.int64)})
    product.write_json(os.path.join(site2.truth_dir, "truth_elev_v2.json"),
                       {"windows": list(package.FREEZE_TRUTH_WINDOWS if windows is None
                                        else windows)})


def fake_pixel_tides(level=0.25):
    """An eo-tides ``pixel_tides`` stand-in on eo-tides' 1 km node grid (a
    level rising 0.01 m per hour since 2024 plus a west-east slope), so the
    sources step never reads the tide model files in tests."""
    import odc.geo.xr  # noqa: F401
    import xarray as xr
    from odc.geo.geobox import GeoBox

    def pixel_tides(gb, time, model, directory, resample, resolution, buffer, extrapolate,
                    cutoff, parallel):
        rg = GeoBox.from_bbox(bbox=gb.buffered(buffer).boundingbox, resolution=resolution)
        z = odc.geo.xr.xr_zeros(rg)
        x, y = z["x"].values, z["y"].values
        hrs = (pd.DatetimeIndex(time) - pd.Timestamp("2024-01-01")) / pd.Timedelta(hours=1)
        X, _ = np.meshgrid(x, y)
        arr = np.stack([np.stack([level + 0.01 * h + (X - X_EDGE) / 1e5 for h in hrs])
                        for _ in model])
        return xr.DataArray(arr.astype(np.float32), dims=("tide_model", "time", "y", "x"),
                            coords={"tide_model": list(model), "time": pd.DatetimeIndex(time),
                                    "y": y, "x": x})
    return pixel_tides


def record_frames(inp, pkg):
    """``{band: (T, H, W)}`` raw arrays of the record scenes (SCL, CLD as
    the checked uint8), and the offsets."""
    out = {}
    pos = {}
    for y in YEARS:
        for i, d in enumerate(dates(y)):
            pos[str(d.date())] = (y, i)
    for b in zc.BANDS:
        arr = np.stack([inp["data"][pos[d][0]][b][pos[d][1]] for d in pkg.dates])
        if b == "SCL":
            arr = obs.scl_u8(arr, FILL)
        elif b == "CLD":
            arr = obs.cld_u8(arr, FILL)
        out[b] = arr
    return out


def make_site(base, mp, predictor=None, build=True, d_truth=True):
    """Inputs and a built v2 package; ``mp`` is a pytest.MonkeyPatch the
    caller undoes."""
    small(mp)
    inp = make_inputs(str(base / "inputs"))
    site2 = site2_for(inp, base / "out")
    provider = FakeTide(nan_at=[inp["nan_instant"]])
    predictor = predictor or FakeOCM()
    ns = types.SimpleNamespace(inp=inp, site2=site2, provider=provider, predictor=predictor,
                               base=base, ran=None, pkg=None)
    if build:
        ns.ran = package.prepare_v2(site2, steps=("record", "pass1", "geometry"),
                                    provider=provider, predictor=predictor)
        if d_truth:
            g = package._geometry(site2)
            write_d_truth(site2, g["keep_full"][::3])
        ns.sources_kw = sources_kw()
        ns.ran.update(package.prepare_v2(site2, provider=provider, predictor=predictor,
                                         sources_kw=ns.sources_kw))
        ns.pkg = package.V2Package(site2)
    return ns


def sources_kw():
    """The sources step's injections of the synthetic site (fake tides)."""
    return {"lt_kw": {"models": [("EOT20", "EOT20")], "pixel_tides": fake_pixel_tides()}}
