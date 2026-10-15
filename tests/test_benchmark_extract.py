"""Tests of experiments.benchmark.extract, the small Zarr extract of a legacy cube (local runs).

They pin: the window with the most mask pixels; the synthetic bands follow
their frozen rules (B02, B04 per pixel; B8A, B11, B12 and CLD exact 2 x 2
replicas on the legacy SCL lattice; fill wherever a real input is fill);
the extract is a canonical store whose real bands read back through
``inputs.ZarrSource`` bit for bit as the legacy values, at the catalogue
times written beside it; its attributes flag the synthetic bands; the
inputs report compares its grid with the legacy window; nothing is ever
written under ``cubes/``.

Run:  python -X utf8 -m pytest tests/test_benchmark_extract.py -q
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import zarr_cubes as zc  # noqa: E402
from experiments.benchmark import extract, inputs, obs  # noqa: E402

FILL = -32768
SITE = "ferrol"
H, W = 40, 50
PHASE = (1, 0)
X0, Y0 = 560_000.0, 4_815_010.0          # 10 m past a 20 m line in y: py = 1


def _wkt():
    from pyproj import CRS
    return CRS.from_epsg(32629).to_wkt("WKT1_GDAL")


def _legacy(path, n=30):
    """A legacy-layout 10 m netCDF (B03, B08, SCL int16, fill -32768) of
    ``n`` dates in 2023 and 2 in 2024; SCL exact 2 x 2 replicas on PHASE."""
    import xarray as xr

    rng = np.random.default_rng(5)
    t = pd.DatetimeIndex(list(pd.date_range("2023-01-03", periods=n, freq="5D"))
                         + list(pd.date_range("2024-01-02", periods=2, freq="5D")))
    T = len(t)
    water = np.zeros((H, W), bool)
    water[:, :20] = True
    b03 = np.where(water, rng.integers(400, 900, (T, H, W)), rng.integers(600, 1300, (T, H, W)))
    b08 = np.where(water, rng.integers(20, 300, (T, H, W)), rng.integers(1800, 3500, (T, H, W)))
    Hb = obs._block_index(H, PHASE[0])[1]
    Wb = obs._block_index(W, PHASE[1])[1]
    scl20 = rng.choice([4, 5, 6, 8, 9, 10, 3], (T, Hb, Wb), p=[.3, .2, .3, .05, .05, .05, .05])
    scl = obs.expand20(scl20, PHASE, (H, W))
    b03 = b03.astype(np.int16)
    b08 = b08.astype(np.int16)
    scl = scl.astype(np.int16)
    # fills inside the extract window (rows 6..26, cols 9..33) on extract dates
    # (every 3rd 2023 date): a B03 pixel, a B08 pixel, one whole 20 m SCL block
    b03[3, 10, 12] = FILL
    b08[6, 15, 21] = FILL
    scl[9, 7:9, 10:12] = FILL
    x = X0 + 5.0 + 10.0 * np.arange(W)
    y = Y0 - 5.0 - 10.0 * np.arange(H)
    ds = xr.Dataset({"B03": (("t", "y", "x"), b03), "B08": (("t", "y", "x"), b08),
                     "SCL": (("t", "y", "x"), scl)}, coords={"t": t, "y": y, "x": x})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": _wkt()})
    enc = {v: {"dtype": "int16", "_FillValue": FILL, "chunksizes": (1, 16, 16), "zlib": True}
           for v in ("B03", "B08", "SCL")}
    ds.to_netcdf(path, encoding=enc)
    labels = {str(d.date()): (d + pd.Timedelta(hours=11, minutes=20)).strftime(
        "%Y-%m-%dT%H:%M:%S") for d in t}
    return {"b03": b03, "b08": b08, "scl": scl, "t": t, "labels": labels}


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("extract")
    nc = tmp / "legacy.nc"
    leg = _legacy(nc)
    lp = tmp / "overpass_times.json"
    lp.write_text(json.dumps(leg["labels"]), encoding="utf-8")
    out = tmp / "scratch_cubes"
    desc = extract.legacy_extract(SITE, out, cube=str(nc), labels=str(lp), year=2023, n_dates=10,
                                  size=(20, 24), origin=(6, 9))
    return {"tmp": tmp, "nc": nc, "leg": leg, "labels": lp, "out": out, "desc": desc}


def test_best_window_maximises_the_mask():
    m = np.zeros((30, 40), bool)
    m[10:16, 25:33] = True
    m[0, 0] = True
    assert extract.best_window(m, (6, 8)) == (10, 25)
    with pytest.raises(ValueError):
        extract.best_window(m, (31, 8))


def test_extract_reads_back_as_the_legacy_window_at_the_catalogue_times(made):
    d = made["desc"]
    leg = made["leg"]
    r0, r1, c0, c1 = d["window"]
    assert (r0, r1, c0, c1) == (6, 26, 9, 33)
    idx23 = np.flatnonzero(leg["t"].year == 2023)
    step = max(1, len(idx23) // 10)
    idx = idx23[::step][:10]
    assert d["dates"] == [str(x.date()) for x in leg["t"][idx]]
    assert d["lattice20"] == [(PHASE[0] - r0) % 2, (PHASE[1] - c0) % 2]
    with inputs.ZarrSource(SITE, root=str(made["out"])) as src:
        assert src.instant_source == "catalogue" and src.dates == d["dates"]
        assert src.instants[0] == pd.Timestamp(made["leg"]["labels"][d["dates"][0]])
        raw = src.read_block(np.arange(src.T), list(zc.BANDS))
        assert src.lattice20 == tuple(d["lattice20"])
    sub = (idx, slice(r0, r1), slice(c0, c1))
    assert np.array_equal(raw["B03"], leg["b03"][sub])
    assert np.array_equal(raw["B08"], leg["b08"][sub])
    assert np.array_equal(raw["SCL"], obs.scl_u8(leg["scl"][sub], FILL))
    # the synthetic bands follow their rules
    b03 = leg["b03"][sub].astype(float)
    b08 = leg["b08"][sub].astype(float)
    bad = (leg["b03"][sub] == FILL) | (leg["b08"][sub] == FILL)
    want02 = np.where(bad, FILL, np.rint(0.92 * b03 + 20))
    assert np.array_equal(raw["B02"], want02.astype(np.int16))
    lat = tuple(d["lattice20"])
    n8 = np.where(leg["b08"][sub] == FILL, np.nan, b08)
    m = obs.block_mean20(n8, lat)
    b8a = np.where(np.isfinite(m), np.rint(m), FILL)
    assert np.array_equal(raw["B8A"], obs.expand20(b8a.astype(np.int16), lat, (r1 - r0, c1 - c0)))
    for b in ("B8A", "B11", "B12", "CLD"):
        a = raw[b]
        assert np.array_equal(a, obs.expand20(obs.block_take20(a, lat), lat, a.shape[1:])), b
    s8 = obs.block_take20(raw["SCL"], lat)
    c8 = obs.block_take20(raw["CLD"], lat)
    assert np.all(c8[s8 == 9] == 90) and np.all(c8[s8 == 8] == 60)
    assert np.all(c8[s8 == 4] == 2) and np.all(c8[s8 == obs.SCL_NODATA_U8] == obs.CLD_NODATA_U8)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (b03 - b08) / (b03 + b08)
        mndwi = (b03 - raw["B11"]) / (b03 + raw["B11"])
    ok = ~bad & (raw["B11"] != FILL) & np.isfinite(ndwi) & np.isfinite(mndwi)
    assert np.mean(np.sign(ndwi[ok]) == np.sign(mndwi[ok])) > 0.95


def test_extract_is_flagged_and_compared_with_its_legacy_window(made, tmp_path):
    a = zc.store_attrs(zc.store_path(SITE, 2023, made["out"]))
    assert a["complete"] == 1 and a["purpose"] == extract.PURPOSE
    assert a["synthetic_bands"] == list(extract.SYNTHETIC_BANDS)
    info = extract.extract_info(SITE, 2023, made["out"])
    assert info["window"] == made["desc"]["window"] and info["synthetic_rules"]
    assert zc.sidecar_path(SITE, 2023, "extract.json", made["out"]).is_file()
    rep = inputs.inputs_report(SITE, root=str(made["out"]), legacy=str(made["nc"]), scan=False)
    assert rep["grid"]["legacy_equal"] is True
    assert rep["grid"]["legacy_window"] == made["desc"]["window"]
    assert rep["extract"]["2023"]["synthetic_bands"] == list(extract.SYNTHETIC_BANDS)
    assert any("EXTRACT" in w for w in rep["warnings"])
    assert rep["instants"]["source"] == "catalogue"
    with pytest.raises(extract.ExtractError, match="exists"):
        extract.legacy_extract(SITE, made["out"], cube=str(made["nc"]), labels=str(made["labels"]),
                               n_dates=10, size=(20, 24), origin=(6, 9))
    with pytest.raises(extract.ExtractError, match="never cubes"):
        extract.legacy_extract(SITE, zc.cubes_root(None), cube=str(made["nc"]),
                               labels=str(made["labels"]))
    with pytest.raises(extract.ExtractError, match="outside"):
        extract.legacy_extract(SITE, tmp_path / "x", cube=str(made["nc"]),
                               labels=str(made["labels"]), size=(20, 24), origin=(30, 0))
