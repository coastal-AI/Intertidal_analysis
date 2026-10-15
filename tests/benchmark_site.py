"""The synthetic site shared by the benchmark tests (a helper, not a test module).

A tiny site written as the production inputs are, so that the package,
domain, judges and truth tests build the same common package:

* an int16 cube (_FillValue -32768, no scale, chunks (1, 8, 8), a crs
  variable) with a sea on the west edge, a channel, tidal flats rising
  eastward, land on the east edge and an isolated pond behind a dyke; SCL
  with every code, fill and fully cloudy scenes; bands with fills, zero sums
  and negative values;
* a production extraction made by ``marea.extract`` itself;
* an overpass file with one date missing; a fake boundary provider
  (:class:`FakeTide`) with one non-finite level;
* three listed judges and one unlisted station (stations.json with every
  field ``judges.stations`` reads): ``st_in`` in the middle of the flats,
  ``st_quay`` on the quay at the landward head (its neighbourhood reaches
  land beyond the 3-px dilation of U_rec, so the package domain test sees
  the NB term of v2 spec §2.10), and ``st_out`` 20 km west of the frame;
* a transition-zone record (every 5th in-period date plus one date that is
  not in the record).

The judge series (:func:`write_gauges`) are written only by the tests that
need them; :func:`use_stations` points ``judges.stations`` and
``judges.read_series`` at them.
"""
import dataclasses
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark import config, domain, package
from pyintertidal import geometry

FILL = -32768
SITE = "escalda"          # a real site name (SITE_ID, seeds); every input is synthetic
RES = 20
H, W = 36, 60
PIXEL_M = 200.0           # 12 km across: several 3-km bands on a tiny frame
N_DATES = 84
CRS_EPSG = 32631
X0, Y0 = 500_000.0, 5_700_000.0
#: domain thresholds scaled to the tiny frame (FROZEN: 3000, 5000, 500)
SMALL = {"MIN_INTERTIDAL": 40, "MIN_FINITE_S": 60, "MIN_PX_BAND": 60}
#: the listed judges and where they stand, in pixel steps (row, col) from the
#: centre of the first pixel: st_in on the flats (pixel (10, 30)), st_quay on
#: the quay at the landward head (pixel (26, 53), land), st_out 20 km west
STATION_PX = {"st_in": (9.6, 30.3), "st_quay": (26.4, 52.6), "st_out": (9.0, -100.0)}
JUDGES = ("st_in", "st_quay", "st_out")
#: the judge series of :func:`write_gauges`: sampling (min), period, the
#: planted lag (min), datum offset (m), gain and the level a drying gauge
#: bottoms out at (m, None = never dry)
GAUGES = {
    "st_in": dict(sampling_min=5, start="2023-01-01 00:00", end="2025-12-31 23:55",
                  lag_min=75, offset_m=0.37, gain=1.0, floor_m=None, noise_m=0.003),
    "st_quay": dict(sampling_min=10, start="2023-01-01 00:00", end="2024-02-15 00:00",
                    lag_min=20, offset_m=-1.2, gain=1.0, floor_m=-0.6, noise_m=0.0),
    "st_out": dict(sampling_min=10, start="2023-01-01 00:00", end="2025-12-31 23:50",
                   lag_min=10, offset_m=0.0, gain=1.0, floor_m=None, noise_m=0.003),
}


# ─────────────────────────────────────────────────────────────────────────────
#  a fake boundary provider and bit views
# ─────────────────────────────────────────────────────────────────────────────

class FakeTide:
    """A smooth tide evaluated elementwise from the instants' nanoseconds
    (M2, S2 and O1 at their exact periods). Like PyTMDBoundary it answers in
    TIME ORDER whatever the input order. NaN at the instants of ``nan_at``;
    ``batch_dependent`` adds a last-bit term that depends on the batch size
    (a provider that is not batch-invariant)."""

    name = "fake tide"

    def __init__(self, nan_at=(), batch_dependent=False):
        self.nan_at = np.asarray(pd.DatetimeIndex(list(nan_at)).asi8, np.int64)
        self.batch_dependent = batch_dependent
        self.calls = []

    @staticmethod
    def value(ns):
        x = np.asarray(ns, np.int64) / 3.6e12                    # hours since 1970
        return (1.7 * np.cos(2 * np.pi * x / 12.4206012 + 0.3)
                + 0.45 * np.cos(2 * np.pi * x / 12.0 + 1.1)
                + 0.2 * np.cos(2 * np.pi * x / 25.8193417 - 0.4))

    def levels(self, times):
        ns = np.sort(pd.DatetimeIndex(times).asi8)
        self.calls.append(ns.copy())
        h = self.value(ns)
        if self.batch_dependent:
            h = h + 1e-9 * len(ns)
        if self.nan_at.size:
            h = np.where(np.isin(ns, self.nan_at), np.nan, h)
        return h


def _bits(a):
    return np.ascontiguousarray(a, np.float64).view(np.int64)


def _bits32(a):
    return np.ascontiguousarray(a, np.float32).view(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
#  the synthetic site
# ─────────────────────────────────────────────────────────────────────────────

def elevation():
    """Sea on the west edge, a channel from it into the frame, flats rising
    eastward, land on the east edge, and an isolated intertidal pond."""
    cols = np.arange(W)
    z = np.empty((H, W))
    z[:] = -1.6 + 3.2 * (cols - 6) / 45.0
    z += 0.15 * np.sin(np.arange(H) / 3.0)[:, None]
    z[:, :6] = -4.0                       # open sea
    z[16:20, :51] = -4.0                  # channel (stops before the land)
    z[:, 52:] = 5.0                       # land: no sea on the east edge
    z[1:7, 39:46] = 5.0                   # a dyke ring ...
    z[2:6, 40:45] = 0.0                   # ... around an intertidal pond
    return z


def station_lonlat(row, col):
    """WGS84 (lon, lat) of a point given in pixel steps from the first centre."""
    import pyproj

    inv = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(CRS_EPSG), "EPSG:4326",
                                      always_xy=True)
    lon, lat = inv.transform(X0 + PIXEL_M * col, Y0 - PIXEL_M * row)
    return float(lon), float(lat)


def station_record(code, row, col, sampling_min=10, site=SITE):
    """A stations.json entry with every field ``judges.stations`` copies."""
    lon, lat = station_lonlat(row, col)
    return {"code": code, "name": code.replace("_", " "), "provider": "synthetic",
            "lon": lon, "lat": lat, "datum": "none", "sampling_min": sampling_min,
            "file": f"{code}.csv", "period": "2023-2025", "site": site,
            "in_boundary": False}


def make_inputs(root, seed=7):
    """Write the synthetic inputs under ``root``; return their paths and the
    raw arrays."""
    xr = pytest.importorskip("xarray")
    pyproj = pytest.importorskip("pyproj")
    from pyintertidal import marea

    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-12-10", periods=N_DATES, freq="14D")
    sec = rng.integers(0, 1200, N_DATES)
    inst = dates + pd.Timedelta(hours=10, minutes=50) + pd.to_timedelta(sec, "s")
    z = elevation()
    h = FakeTide.value(inst.asi8)
    wet = h[:, None, None] > z[None] + rng.normal(0.0, 0.05, (N_DATES, H, W))
    land = np.broadcast_to(z >= 4.9, wet.shape)
    b03 = np.where(wet, rng.normal(600, 150, wet.shape), rng.normal(900, 150, wet.shape))
    b08 = np.where(wet, rng.normal(80, 120, wet.shape), rng.normal(1500, 300, wet.shape))
    b03 = np.where(land, rng.normal(700, 100, wet.shape), b03)
    b08 = np.where(land, rng.normal(3000, 400, wet.shape), b08)
    u = rng.random(wet.shape)
    scl = np.select([u < 0.76, u < 0.82, u < 0.86, u < 0.89, u < 0.92, u < 0.94, u < 0.95,
                     u < 0.96],
                    [rng.choice([4, 5, 6, 7], wet.shape), rng.choice([8, 9, 10], wet.shape),
                     np.full(wet.shape, 3), np.full(wet.shape, 2), np.full(wet.shape, 11),
                     np.full(wet.shape, 1), np.full(wet.shape, 0), np.full(wet.shape, 4)],
                    FILL).astype(np.int16)
    scl[10] = 9                           # a fully cloudy scene
    scl[11, : H // 2] = FILL              # half the frame without data
    cloud = np.isin(scl, (8, 9, 10))
    b03 = np.where(cloud, rng.normal(4000, 800, wet.shape), b03)
    b08 = np.where(cloud, rng.normal(4000, 800, wet.shape), b08)
    shadow = scl == 3
    b03 = np.where(shadow, 0.3 * b03, b03)
    b08 = np.where(shadow, 0.3 * b08, b08)
    b03 = np.clip(np.round(b03), -999, 15000).astype(np.int16)
    b08 = np.clip(np.round(b08), -999, 15000).astype(np.int16)
    b03[20, 10, 10], b08[20, 10, 10] = 5, -5          # zero sum
    b03[21, 10, 11], b08[21, 10, 11] = -100, -50      # both < 0
    b03[22, 10, 12], b08[22, 10, 12] = 0, -200        # floor0: 0 / 0
    b03[23, 10, 13], b08[23, 10, 13] = 300, -280      # raw |NDWI| > 1
    fill_b = rng.random(wet.shape) < 0.01
    b03[fill_b] = FILL
    b08[rng.random(wet.shape) < 0.01] = FILL
    scl[20, 10, 10] = scl[21, 10, 11] = scl[22, 10, 12] = scl[23, 10, 13] = 4

    # the cube, written as the production cubes are
    xs = X0 + PIXEL_M * np.arange(W)
    ys = Y0 - PIXEL_M * np.arange(H)
    crs = pyproj.CRS.from_epsg(CRS_EPSG)
    ds = xr.Dataset({v: (("t", "y", "x"), a) for v, a in
                     (("B03", b03), ("B08", b08), ("SCL", scl))},
                    coords={"t": dates, "y": ys, "x": xs})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": crs.to_wkt()})
    enc = {v: {"dtype": "int16", "_FillValue": FILL, "chunksizes": (1, 8, 8), "zlib": True}
           for v in ("B03", "B08", "SCL")}
    cube = os.path.join(root, "cube.nc")
    ds.to_netcdf(cube, encoding=enc)

    # the production extraction, by the production code (cell 19 adds s_km)
    ex = marea.extract(cube)
    seeds = geometry.mouth_seeds(ex["sea"], side="west")
    s_km = geometry.along_distance(ex["sea"] | ex["inter"], seeds,
                                   PIXEL_M).ravel()[ex["keep"]] / 1000.0
    extract = os.path.join(root, "marea_extract.npz")
    np.savez_compressed(extract, Y=ex["Y"], C=ex["C"], keep=ex["keep"], dates=ex["dates"],
                        shape=np.array(ex["shape"]), sea=ex["sea"], inter=ex["inter"],
                        s_km=s_km.astype(np.float32),
                        bbox=np.array(ex["bbox"], dtype=object), crs_wkt=ex["crs_wkt"])

    # overpass times: one in-period date has none
    no_time = str(dates[12].date())
    over = {str(d.date()): ts.strftime("%Y-%m-%dT%H:%M:%S")
            for d, ts in zip(dates, inst) if str(d.date()) != no_time}
    overpass = os.path.join(root, "overpass_times.json")
    with open(overpass, "w", encoding="utf-8") as f:
        json.dump(over, f)
    nan_instant = inst[30]                # a non-finite boundary level there

    # the listed judges and one unlisted station (at st_in's place)
    recs = [station_record(c, *STATION_PX[c], sampling_min=GAUGES[c]["sampling_min"])
            for c in JUDGES]
    recs.append(station_record("unlisted", *STATION_PX["st_in"]))
    stations = os.path.join(root, "stations.json")
    with open(stations, "w", encoding="utf-8") as f:
        json.dump(recs, f)

    # a transition-zone record: every 5th in-period date, plus one date not
    # in the record (the one without an overpass time)
    in_period = [str(d.date()) for d in dates if "2023-01-01" <= str(d.date()) <= "2025-12-31"]
    tz_dates = sorted(set(in_period[::5]) | {no_time})
    tz = os.path.join(root, "result.json")
    with open(tz, "w", encoding="utf-8") as f:
        json.dump({"record_dates": tz_dates}, f)
    return {"root": root, "cube": cube, "extract": extract, "overpass": overpass,
            "stations": stations, "tz": tz, "tz_dates": tz_dates, "dates": dates,
            "inst": inst, "no_time": no_time, "nan_instant": nan_instant, "b03": b03,
            "b08": b08, "scl": scl, "z": z, "ex": ex, "s_km_ex": s_km}


def site_cfg(inp, out_root, **over):
    """A SiteConfig for the synthetic site (outputs under ``out_root``)."""
    kw = dict(cube=inp["cube"], extract=inp["extract"], overpass=inp["overpass"],
              pixel_m=PIXEL_M, boundary=types.MappingProxyType({"kind": "fake",
                                                                "name": "fake tide"}),
              mouth_side="west", judges=JUDGES, transition_result=inp["tz"],
              root=str(out_root).replace("\\", "/"),
              extra=types.MappingProxyType({"stations_json": inp["stations"]}))
    kw.update(over)
    return dataclasses.replace(config.site_config(SITE, RES), **kw)


def small_thresholds(mp):
    for k, v in SMALL.items():
        mp.setattr(domain, k, v)


def build(inp, out_root, provider=None, **kw):
    """Prepare a package of the synthetic site; return (cfg, key, provider)."""
    cfg = site_cfg(inp, out_root)
    provider = provider or FakeTide(nan_at=[inp["nan_instant"]])
    key = package.prepare(SITE, RES, cfg=cfg, provider=provider, **kw)
    return cfg, key, provider


def make_site(base, mp):
    """Inputs, a built and loaded package (domain thresholds lowered through
    ``mp``, a pytest.MonkeyPatch the caller undoes): a SimpleNamespace with
    inp, cfg, key, provider, pkg and base."""
    small_thresholds(mp)
    inp = make_inputs(str(base / "inputs"))
    cfg, key, prov = build(inp, base / "out")
    pkg = package.load(SITE, RES, cfg=cfg, provider=prov)
    return types.SimpleNamespace(inp=inp, cfg=cfg, key=key, provider=prov, pkg=pkg,
                                 base=base)


def record_frames(inp, pkg):
    """The cube's raw arrays of the record scenes."""
    ti = pkg.time_index
    return inp["b03"][ti], inp["b08"][ti], inp["scl"][ti]


# ─────────────────────────────────────────────────────────────────────────────
#  judge series
# ─────────────────────────────────────────────────────────────────────────────

def gauge_series(spec, seed=0):
    """``(DatetimeIndex, level)`` of a synthetic judge: offset + gain *
    h_B(t - lag) + noise, floored at ``floor_m`` for a drying gauge."""
    step = f"{int(spec['sampling_min'])}min"
    t = pd.date_range(spec["start"], spec["end"], freq=step).as_unit("ns")
    lag = pd.Timedelta(minutes=float(spec["lag_min"]))
    v = spec["offset_m"] + spec["gain"] * FakeTide.value((t - lag).asi8)
    if spec.get("noise_m"):
        v = v + np.random.default_rng(seed).normal(0.0, spec["noise_m"], len(t))
    if spec.get("floor_m") is not None:
        v = np.maximum(v, spec["offset_m"] + spec["floor_m"])
    return t, v


def write_series_csv(path, t, v):
    """A judge CSV as delivered: ``time_utc,level_m``, naive UTC seconds."""
    stamps = np.char.replace(np.datetime_as_string(pd.DatetimeIndex(t).values, unit="s"),
                             "T", " ")
    pd.DataFrame({"time_utc": stamps, "level_m": np.asarray(v, np.float64)}).to_csv(
        path, index=False, lineterminator="\n")
    return path


def write_gauges(inp, specs=None):
    """Write every judge series of ``specs`` (default :data:`GAUGES`) next to
    stations.json; return ``{code: path}``."""
    specs = GAUGES if specs is None else specs
    out = {}
    for k, (code, spec) in enumerate(sorted(specs.items())):
        t, v = gauge_series(spec, seed=100 + k)
        out[code] = write_series_csv(os.path.join(inp["root"], f"{code}.csv"), t, v)
    return out


def use_stations(mp, inp):
    """Point ``judges.stations`` and ``judges.read_series`` at the synthetic
    stations: config.JUDGES[SITE], STATIONS_JSON and ESTUARY_TRUTH."""
    mp.setitem(config.JUDGES, SITE, JUDGES)
    mp.setattr(config, "STATIONS_JSON", inp["stations"])
    mp.setattr(config, "ESTUARY_TRUTH", inp["root"])
