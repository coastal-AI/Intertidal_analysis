# -*- coding: utf-8 -*-
"""Zarr validation cubes: downloader normalisation, reader and checks (no network).

Fake backend results are written with zarr-python in every layout the
openEO backend might deliver (directory store, zipped store, one asset per
file, Zarr v2 big-endian as a JVM writer might make it, Zarr v3 with a
``bands`` dimension, sharded v3, no dimension names, no group metadata,
assets split by band or by date, netCDF), normalised into canonical yearly
stores, and read back through ``open_cube``; the verification is checked on
synthetic netCDF cubes. Jobs run against a fake connection.
"""
import ast
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from experiments import download_cube_zarr as dcz
from experiments import zarr_cubes as zc

BANDS = zc.BANDS
FILL = zc.FILL
X0, Y0 = 301025.0, 4824835.0          # pixel centres on the Villaviciosa 10 m lattice


@pytest.fixture(scope="module")
def wkt():
    from pyproj import CRS

    return CRS.from_epsg(32630).to_wkt("WKT1_GDAL")


def synth(year, n=6, H=24, W=40, seed=0, x0=X0, y0=Y0, start=None):
    """A synthetic year: dates, axes and int16 bands with a no-data patch."""
    rng = np.random.default_rng(seed + int(year))
    t = pd.date_range(start or f"{year}-01-03", periods=n, freq="17D")
    x = x0 + 10.0 * np.arange(W)
    y = y0 - 10.0 * np.arange(H)
    data = {}
    for b in BANDS:
        hi = 12 if b == "SCL" else 101 if b == "CLD" else 10000
        a = rng.integers(0, hi, (n, H, W)).astype("int16")
        a[0, :3, :5] = FILL
        a[-1, -2:, :] = FILL
        data[b] = a
    return t, y, x, data


def _days(t):
    return (pd.DatetimeIndex(t) - pd.Timestamp("1990-01-01")).days.values.astype("int32")


def write_raw_v2(path, t, y, x, data, wkt, *, big_endian=True, dims=True, group_marker=True,
                 y_ascending=False, chunks=(2, 12, 16), names=None, fill_value=FILL):
    """A Zarr v2 store as a JVM writer might make it: one array per band.
    ``fill_value=None``: zarr's default (0), as a writer that declares none."""
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    yy = y[::-1] if y_ascending else y
    for b, a in data.items():
        name = (names or {}).get(b, b)
        kw = {} if fill_value is None else {"fill_value": fill_value}
        arr = g.create_array(name, shape=a.shape, dtype=">i2" if big_endian else "<i2",
                             chunks=chunks, **kw)
        arr[:] = a[:, ::-1, :] if y_ascending else a
        attrs = {"grid_mapping": "crs"}
        if dims:
            attrs["_ARRAY_DIMENSIONS"] = ["t", "y", "x"]
        arr.attrs.update(attrs)
    for name, vals, attrs in (("t", _days(t), {"units": "days since 1990-01-01"}),
                              ("x", np.asarray(x), {}), ("y", np.asarray(yy), {})):
        arr = g.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
        arr[:] = vals
        a2 = dict(attrs)
        if dims:
            a2["_ARRAY_DIMENSIONS"] = [name]
        arr.attrs.update(a2)
    c = g.create_array("crs", shape=(), dtype="int32")
    c.attrs.update({"crs_wkt": wkt, "spatial_ref": wkt, **({"_ARRAY_DIMENSIONS": []} if dims else {})})
    if not group_marker:
        os.remove(Path(path) / ".zgroup")
    return Path(path)


def write_raw_v3_bands(path, t, y, x, data, labels=True, chunks=(1, 1, 8, 8)):
    """Zarr v3, one (time, bands, y, x) array, CRS as an EPSG root attribute."""
    g = zarr.open_group(str(path), mode="w", zarr_format=3, attributes={"crs": "EPSG:32630"})
    cube = np.stack([data[b] for b in BANDS], axis=1)
    arr = g.create_array("cube", shape=cube.shape, dtype="int16", chunks=chunks,
                         fill_value=FILL, dimension_names=("time", "bands", "y", "x"))
    arr[:] = cube
    if labels:
        lab = g.create_array("bands", shape=(len(BANDS),), dtype=str, dimension_names=("bands",))
        lab[:] = np.array(BANDS)
    for name, vals, attrs in (("time", _days(t), {"units": "days since 1990-01-01"}),
                              ("x", np.asarray(x), {}), ("y", np.asarray(y), {})):
        a = g.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape,
                           dimension_names=(name,), attributes=attrs)
        a[:] = vals
    return Path(path)


def write_raw_v3_sharded(path, t, y, x, data, wkt):
    """Zarr v3, sharded, zarr fill_value 0 but a declared nodata of -32768."""
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    for b, a in data.items():
        arr = g.create_array(b, shape=a.shape, dtype="int16", chunks=(1, 8, 8), shards=(2, 16, 16),
                             fill_value=0, dimension_names=("t", "y", "x"),
                             attributes={"nodata": FILL, "grid_mapping": "spatial_ref"})
        arr[:] = a
    for name, vals, attrs in (("t", _days(t), {"units": "days since 1990-01-01"}),
                              ("x", np.asarray(x), {}), ("y", np.asarray(y), {})):
        a = g.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape,
                           dimension_names=(name,), attributes=attrs)
        a[:] = vals
    sr = g.create_array("spatial_ref", shape=(), dtype="int32", attributes={"crs_wkt": wkt})
    sr[...] = 0
    return Path(path)


def write_nc(path, t, y, x, data, wkt, bands=("B03", "B08", "SCL")):
    """A netCDF cube as the openEO backend writes them (int16, fill -32768)."""
    ds = xr.Dataset({b: (("t", "y", "x"), data[b], {"long_name": b, "grid_mapping": "crs"})
                     for b in bands}, coords={"t": pd.DatetimeIndex(t), "y": y, "x": x})
    ds["crs"] = xr.DataArray(np.array(b"", dtype="S1"), attrs={"crs_wkt": wkt, "spatial_ref": wkt})
    enc = {b: {"dtype": "int16", "_FillValue": np.int16(FILL), "zlib": True, "complevel": 1,
               "chunksizes": (1, len(y), len(x))} for b in bands}
    enc["t"] = {"units": "days since 1990-01-01", "dtype": "int32"}
    ds.to_netcdf(path, encoding=enc)
    return Path(path)


def zip_store(store_dir, zip_base):
    """Zip a store as a backend might (shutil.make_archive on its root)."""
    return Path(shutil.make_archive(str(zip_base), "zip", root_dir=str(store_dir)))


def make_request(root, year=2023, site="villaviciosa", start=None, end=None, test=False):
    s, e = zc.year_window(year)
    return dcz.Request(site=site, year=year, start=start or s, end=end or e,
                       bbox=dict(dcz.SITE_CONFIG["villaviciosa"]["bbox"]), root=Path(root),
                       test=test)


def place(req, items, datetimes=None):
    """Put fake downloaded assets in ``<part>/assets`` and mark them complete.

    ``items``: (relative name, source file or folder, media type);
    ``datetimes``: {relative name: STAC datetime} of its files."""
    raw = Path(req.part) / "assets"
    raw.mkdir(parents=True, exist_ok=True)
    files = {}
    for rel, src, typ in items:
        dst = raw / rel
        extra = {"datetime": datetimes[rel]} if datetimes and rel in datetimes else {}
        if Path(src).is_dir():
            shutil.copytree(src, dst)
            for f in dst.rglob("*"):
                if f.is_file():
                    files[f.relative_to(raw).as_posix()] = dict(
                        {"size": f.stat().st_size, "type": typ}, **extra)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files[rel] = dict({"size": dst.stat().st_size, "type": typ}, **extra)
    zc.write_json({"job_id": "j-test", "complete": True, "files": files},
                  Path(req.part) / dcz.ASSETS_DONE)


def normalise_and_write(req, ref_grid=None, **kw):
    lines = []
    ds, interp, rep = dcz.normalise(req, lines.append, job_info={"id": "j-test", "title": "t"},
                                    results_meta={}, ref_grid=ref_grid, **kw)
    flat = dcz.build_process(req).flat_graph()
    attrs = dcz.provenance(req, {"id": "j-test", "title": "t"}, {}, interp, dcz.Context(), flat,
                           ref_grid=ref_grid)
    dcz.write_store(ds, interp.fills, req.store, attrs, log=lines.append)
    return interp, rep, lines


def check_store(store, t, y, x, data, bands=BANDS, job_id="j-test"):
    assert zc.is_complete(store)
    ds = zc.open_store(store)
    assert list(ds["t"].values) == list(pd.DatetimeIndex(t).values)
    np.testing.assert_array_equal(ds["x"].values, x)
    np.testing.assert_array_equal(ds["y"].values, y)
    assert sorted(zc.band_names(ds)) == sorted(bands)
    for b in bands:
        assert ds[b].dims == ("t", "y", "x")
        assert ds[b].dtype == np.int16
        assert zc.fill_of(ds[b]) == FILL
        np.testing.assert_array_equal(ds[b].values, data[b])
    meta = json.loads((Path(store) / "B03" / ".zarray").read_text())
    assert meta["compressor"]["id"] == "blosc" and meta["compressor"]["cname"] == "zstd"
    assert meta["compressor"]["clevel"] == 5 and meta["compressor"]["shuffle"] == 1
    assert meta["chunks"] == list(zc.chunk_shape(ds["B03"].shape))
    assert meta["fill_value"] == FILL and meta["dtype"] == "<i2"
    assert zc.grid_info(ds)["epsg"] == 32630
    assert ds.attrs["job_id"] == job_id and ds.attrs["format"] == zc.FORMAT_ID
    assert json.loads(ds.attrs["bands"]) == list(BANDS)
    assert zc.band_names(ds) == [b for b in BANDS if b in bands]
    ds.close()


# ─────────────────────────────────────────────────────────────────────────────
#  frames on the Villaviciosa lattice, and deliveries like the backend's Zarr
#  of the format test of 2026-10-13
# ─────────────────────────────────────────────────────────────────────────────

#: (west, south, east, north) of ndwi_cube_villaviciosa_grande_10y.nc (EPSG:32630)
VILLA_FRAME = (301020.0, 4815690.0, 310170.0, 4824840.0)
#: a 300 x 300 px frame on that lattice around the --test centre: a light reference cube
TEST_FRAME = (305000.0, 4820000.0, 308000.0, 4823000.0)
#: the --test window the downloader puts in either frame (2,040 m, corners on 60 m)
TEST_WIN = (305400.0, 4820340.0, 307440.0, 4822380.0)


def axes_of(frame, res=10.0):
    """Pixel-centre axes (x ascending, y descending) of a (west, south, east, north) frame."""
    west, south, east, north = frame
    nx, ny = int(round((east - west) / res)), int(round((north - south) / res))
    return west + res / 2 + res * np.arange(nx), north - res / 2 - res * np.arange(ny)


def frame_dict(frame, wkt, epsg=32630, res=10.0):
    """A frame as the downloader's frame helpers return it."""
    west, south, east, north = frame
    return {"west": west, "south": south, "east": east, "north": north, "res": res, "epsg": epsg,
            "crs_wkt": wkt, "source": "a synthetic frame", "kind": "netcdf", "origin": "synthetic"}


def _mix(*ints):
    """A 64-bit integer hash (splitmix64 rounds) of integer arrays, elementwise:
    pseudo-random values of the position with no periodic structure."""
    h = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for v in ints:
            h = (h ^ np.asarray(v).astype(np.int64).astype(np.uint64)) * np.uint64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> np.uint64(31))) * np.uint64(0x94D049BB133111EB)
            h = h ^ (h >> np.uint64(29))
    return h


def lattice_values(b, n, x, y, scl_clear=False):
    """Deterministic int16 values of band ``b`` on ``n`` dates at pixel
    centres ``x`` (columns) by ``y`` (rows), a function of the UTM position
    only, so any two grids on the lattice agree where they overlap. The 20 m
    bands are constant on the 20 m cells of the lattice (nearest neighbour
    from the native grid); date 0 has no data on two 20 m rows (y 4821005..35).
    A hash of the position (``_mix``), not a linear congruence: the field of
    before 2026-10-19 ((X * 7919 + Y * 104729 + ...) mod 10000) repeated itself
    exactly at some shifts ((32, 33) px), which a registration rightly refuses
    as not unique. ``scl_clear``: SCL in the clear classes 4/5/6 only (the
    registration chooses its dates by them)."""
    res = float(zc.NATIVE_RES_M[b])
    xv, yv = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    X = np.floor(xv / res).astype("int64")[None, None, :]
    Y = np.floor(yv / res).astype("int64")[None, :, None]
    k = np.arange(n, dtype="int64")[:, None, None]
    h = _mix(X, Y, k, BANDS.index(b) + 1)
    if b == "SCL" and scl_clear:
        out = (4 + h % np.uint64(3)).astype("int16")
    else:
        hi = 12 if b == "SCL" else 101 if b == "CLD" else 10000
        out = (h % np.uint64(hi)).astype("int16")
    out[0, (yv > 4821000) & (yv < 4821040), :] = FILL
    return out


def frame_nc(path, frame, wkt, n=1, bands=("B03",), start="2024-06-02", values=False):
    """A light reference netCDF cube on ``frame``: ``n`` dates (5 days apart)
    of ``bands``, constant values (compress to nothing) unless ``values``
    (then ``lattice_values``)."""
    x, y = axes_of(frame)
    t = pd.date_range(start, periods=n, freq="5D")
    data = {b: (lattice_values(b, n, x, y) if values else np.full((n, len(y), len(x)), 1000, "int16"))
            for b in bands}
    return write_nc(path, t, y, x, data, wkt, bands=tuple(bands))


def server_axes(west, north, n_cols, n_rows, res=10.0):
    """Cell centres of a delivery anchored on (west, north), as the backend's
    Zarr path does it, n_cols x n_rows (padded to whole chunks)."""
    return (west + res / 2 + res * np.arange(n_cols), north - res / 2 - res * np.arange(n_rows))


def f32_like_the_backend(x, y):
    """x / y as the backend's Zarr writer delivers them: float32 (the
    float64 axes cast, rounding to nearest: the 2026-10-13 delivery's y
    4822370.74 came as 4822370.5). Pixel centres on the lattice are whole
    metres, exact in float32, so a delivery on the lattice comes back exact;
    ``test_a_real_offset_is_still_refused`` covers values off it."""
    return (np.asarray(x, dtype="float64").astype("float32"),
            np.asarray(y, dtype="float64").astype("float32"))


def write_server_zarr(path, t, y, x, data, wkt, extent, chunks=(1, 128, 128), dtype=">i2",
                      fill_value=FILL):
    """A delivery like the backend's Zarr of 2026-10-13: Zarr v2, one
    (time, y, x) array per band (``_ARRAY_DIMENSIONS``), big-endian int16 with
    fill -32768, Blosc lz4 chunks [1, 128, 128]; time big-endian int64 in
    'milliseconds since 01/01/1970 - 00:00:00' (proleptic_gregorian); x / y as
    given (float32 for the backend's); root attributes 'extent' and '_CRS'
    (WKT), no grid-mapping variable."""
    import numcodecs

    comp = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    g.attrs.update({"extent": [float(v) for v in extent], "_CRS": wkt})
    for b, a in data.items():
        arr = g.create_array(b, shape=a.shape, dtype=dtype, chunks=chunks, fill_value=fill_value,
                             compressors=comp)
        arr[:] = a
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
    ms = ((pd.DatetimeIndex(t) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")).values
    for name, vals, attrs in (
            ("time", np.asarray(ms, dtype=">i8"),
             {"units": "milliseconds since 01/01/1970 - 00:00:00",
              "calendar": "proleptic_gregorian"}),
            ("x", np.asarray(x), {}), ("y", np.asarray(y), {})):
        dt_ = ">i8" if name == "time" else np.asarray(vals).dtype.newbyteorder(">")
        arr = g.create_array(name, shape=vals.shape, dtype=dt_, chunks=vals.shape, compressors=comp)
        arr[:] = vals
        arr.attrs.update(dict(attrs, _ARRAY_DIMENSIONS=[name]))
    return Path(path)


# ─────────────────────────────────────────────────────────────────────────────
#  normalisation of every plausible delivered layout
# ─────────────────────────────────────────────────────────────────────────────

def test_zipped_v2_store_big_endian(tmp_path, wkt):
    """The backend's default: one zip of a Zarr v2 store, per-band arrays."""
    t, y, x, data = synth(2023)
    raw = write_raw_v2(tmp_path / "src" / "openEO.zarr", t, y, x, data, wkt)
    z = zip_store(raw, tmp_path / "src" / "openEO.zarr")
    req = make_request(tmp_path / "cubes")
    place(req, [("openEO.zarr.zip", z, "application/zip")])
    interp, rep, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert "1 Zarr store(s)" in " ".join(rep)
    assert interp.crs_source.startswith("variable 'crs'")
    assert all(interp.fill_sources[b] == "Zarr fill_value" for b in BANDS)
    assert (Path(req.part) / "extracted" / "openEO.zarr" / ".zgroup").is_file()
    assert req.sidecar("format.txt").is_file()


def test_directory_store_as_many_assets(tmp_path, wkt):
    """A store delivered file by file (one asset per Zarr file)."""
    t, y, x, data = synth(2023, seed=3)
    raw = write_raw_v2(tmp_path / "src" / "openEO.zarr", t, y, x, data, wkt, big_endian=False)
    req = make_request(tmp_path / "cubes")
    place(req, [("openEO.zarr", raw, "application/octet-stream")])
    _, rep, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert any("downloaded files under" in ln for ln in rep)       # the folder tree is shown
    assert any(ln.lstrip().startswith("openEO.zarr") and "files" in ln for ln in rep)


def test_bands_dimension_v3_with_time_alias(tmp_path):
    t, y, x, data = synth(2023, seed=5)
    raw = write_raw_v3_bands(tmp_path / "src" / "cube.zarr", t, y, x, data)
    z = zip_store(raw, tmp_path / "src" / "cube.zarr")
    req = make_request(tmp_path / "cubes")
    place(req, [("cube.zarr.zip", z, "application/zip")])
    interp, rep, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert interp.delivered["B8A"] == "cube[bands=B8A]"
    assert "global attribute 'crs'" in interp.crs_source
    assert any("renamed" in n for n in interp.notes)


def test_bands_dimension_without_labels_uses_request_order(tmp_path):
    t, y, x, data = synth(2023, seed=6)
    raw = write_raw_v3_bands(tmp_path / "src" / "cube.zarr", t, y, x, data, labels=False)
    req = make_request(tmp_path / "cubes")
    place(req, [("cube.zarr", raw, "application/octet-stream")])
    interp, _, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert any("requested order" in w for w in interp.warnings)


def test_sharded_v3_declared_nodata_wins_over_zarr_fill(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=7)
    raw = write_raw_v3_sharded(tmp_path / "src" / "s.zarr", t, y, x, data, wkt)
    req = make_request(tmp_path / "cubes")
    place(req, [("s.zarr", raw, "application/octet-stream")])
    interp, _, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert interp.fill_sources["B03"] == "attribute 'nodata'"
    assert "spatial_ref" in interp.crs_source


@pytest.mark.parametrize("H,W", [(24, 40), (24, 24)])
def test_no_dimension_names_inferred_from_coordinates(tmp_path, wkt, H, W):
    t, y, x, data = synth(2023, H=H, W=W, seed=8)
    raw = write_raw_v2(tmp_path / "src" / "n.zarr", t, y, x, data, wkt, dims=False)
    req = make_request(tmp_path / "cubes")
    place(req, [("n.zarr", raw, "application/octet-stream")])
    interp, _, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    if H == W:
        assert any("conventional order" in n for n in interp.notes)


def test_missing_group_marker_and_ascending_y(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=9)
    raw = write_raw_v2(tmp_path / "src" / "g.zarr", t, y, x, data, wkt, group_marker=False,
                       y_ascending=True)
    req = make_request(tmp_path / "cubes")
    place(req, [("g.zarr", raw, "application/octet-stream")])
    interp, _, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)               # y descending again, rows in place
    assert any("flipped" in n for n in interp.notes)


def test_assets_split_by_time_and_by_band(tmp_path, wkt):
    t, y, x, data = synth(2023, n=8, seed=10)
    early = {b: a[:5] for b, a in data.items()}
    late = {b: a[5:] for b, a in data.items()}
    a1 = write_raw_v2(tmp_path / "src" / "p1.zarr", t[:5], y, x, early, wkt)
    a2 = write_raw_v2(tmp_path / "src" / "p2.zarr", t[5:], y, x, late, wkt)
    req = make_request(tmp_path / "cubes")
    place(req, [("p2.zarr", a2, "x"), ("p1.zarr", a1, "x")])
    interp, _, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert any("concatenated" in n for n in interp.notes)

    req2 = make_request(tmp_path / "cubes2")
    spectral = {b: data[b] for b in BANDS[:7]}
    masks = {b: data[b] for b in BANDS[7:]}
    b1 = write_raw_v2(tmp_path / "src" / "q1.zarr", t, y, x, spectral, wkt)
    b2 = write_raw_v2(tmp_path / "src" / "q2.zarr", t, y, x, masks, wkt)
    place(req2, [("q1.zarr", b1, "x"), ("q2.zarr", b2, "x")])
    interp2, _, _ = normalise_and_write(req2)
    check_store(req2.store, t, y, x, data)
    assert any("merged" in n for n in interp2.notes)


def test_band_names_in_other_spellings(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=11)
    names = {"B02": "b2", "B8A": "b8a", "SCL": "scl", "CLD": "SENTINEL2_L2A_CLD"}
    raw = write_raw_v2(tmp_path / "src" / "o.zarr", t, y, x, data, wkt, names=names)
    req = make_request(tmp_path / "cubes")
    place(req, [("o.zarr", raw, "x")])
    interp, _, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert interp.delivered["B02"] == "b2" and interp.delivered["CLD"] == "SENTINEL2_L2A_CLD"
    for name, key in (("SENTINEL2_L2A:B02", "B02"), ("B8a", "B8A"), ("b08", "B08"), ("B8", "B08"),
                      ("S2_SCL", "SCL"), ("band_3", "BAND_3"), ("sunAzimuthAngles", "SUNAZIMUTHANGLES")):
        assert dcz.band_key(name) == key


def test_netcdf_delivered_instead(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=12)
    nc = write_nc(tmp_path / "openEO.nc", t, y, x, data, wkt, bands=BANDS)
    req = make_request(tmp_path / "cubes")
    place(req, [("openEO.nc", nc, "application/x-netcdf")])
    _, rep, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert any("delivered netCDF" in ln for ln in rep)


def test_crs_from_the_reference_cube_when_the_store_has_none(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=13)
    raw = write_raw_v2(tmp_path / "src" / "c.zarr", t, y, x, data, wkt)
    shutil.rmtree(raw / "crs")
    for b in BANDS:
        zarr.open_array(store=str(raw / b), mode="r+").attrs.pop("grid_mapping", None)
    req = make_request(tmp_path / "cubes")
    place(req, [("c.zarr", raw, "x")])
    with pytest.raises(dcz.FormatError, match="no CRS found"):
        dcz.normalise(req, lambda m: None)
    ref = {"path": "ref.nc", "x": x, "y": y, "crs_wkt": wkt}
    interp, _, _ = normalise_and_write(req, ref_grid=ref)
    check_store(req.store, t, y, x, data)
    assert interp.crs_source.startswith("reference netCDF cube")


def test_write_store_in_blocks_aligned_to_the_delivered_chunks(tmp_path, wkt):
    t, y, x, data = synth(2023, n=9, seed=18)
    raw = write_raw_v2(tmp_path / "src" / "c.zarr", t, y, x, data, wkt, chunks=(3, 12, 16))
    req = make_request(tmp_path / "cubes")
    place(req, [("c.zarr", raw, "x")])
    ds, interp, _ = dcz.normalise(req, lambda m: None)
    msgs = []
    dcz.write_store(ds, interp.fills, req.store, {"job_id": "j-test", "bands": json.dumps(BANDS)},
                    log=msgs.append, chunks={"t": 2, "y": 8, "x": 8})
    assert any("blocks of 4 dates" in m for m in msgs)
    assert sum(m.startswith("block ") for m in msgs) == 3           # [0, 4), [4, 8), [8, 9)
    out = zc.open_store(req.store)
    for b in BANDS:
        np.testing.assert_array_equal(out[b].values, data[b])
    assert json.loads((req.store / "B12" / ".zarray").read_text())["chunks"] == [2, 8, 8]
    out.close()


def test_grid_off_the_reference_lattice_stops(tmp_path, wkt):
    """Corners delivered as centres, or another CRS: no store unless allowed."""
    t, y, x, data = synth(2023, seed=17)
    ref = {"path": "ref.nc", "x": x, "y": y, "crs_wkt": wkt}
    raw = write_raw_v2(tmp_path / "src" / "h.zarr", t, y + 5.0, x - 5.0, data, wkt)
    req = make_request(tmp_path / "cubes")
    place(req, [("h.zarr", raw, "x")])
    with pytest.raises(dcz.FormatError, match=r"offset by \(0.500, 0.500\) px"):
        dcz.normalise(req, lambda m: None, ref_grid=ref)
    assert not req.store.exists()
    _, interp, _ = dcz.normalise(req, lambda m: None, ref_grid=ref, allow_offset=True)
    assert any("--allow-grid-offset" in w for w in interp.warnings)
    # same lattice, smaller extent: a production year stops (the fill would read as
    # no-data inside the frame); --allow-padding puts it on the reference grid, warned
    sub = {b: a[:, 2:10, 3:20] for b, a in data.items()}
    raw2 = write_raw_v2(tmp_path / "src" / "s.zarr", t, y[2:10], x[3:20], sub, wkt)
    req2 = make_request(tmp_path / "c2")
    place(req2, [("s.zarr", raw2, "x")])
    with pytest.raises(dcz.FormatError, match=r"does not cover the whole reference grid.*"
                                              r"--allow-padding"):
        dcz.normalise(req2, lambda m: None, ref_grid=ref)
    assert not req2.store.exists()
    assert "NOT NORMALISABLE" in req2.sidecar("format.txt").read_text(encoding="utf-8")
    ds2, interp2, _ = dcz.normalise(req2, lambda m: None, ref_grid=ref, allow_padding=True)
    assert any("another extent" in w and "padded with the fill top 2" in w
               for w in interp2.warnings)
    np.testing.assert_array_equal(ds2["x"].values, x)
    np.testing.assert_array_equal(ds2["B03"].values[:, 2:10, 3:20], sub["B03"])
    assert (ds2["B03"].values[:, :2, :] == FILL).all()
    # --test: the window stays as delivered
    req2t = make_request(tmp_path / "c2t", test=True)
    place(req2t, [("s.zarr", raw2, "x")])
    ds2t, interp2t, _ = dcz.normalise(req2t, lambda m: None, ref_grid=ref)
    assert ds2t.sizes["x"] == 17 and not any("another extent" in w for w in interp2t.warnings)
    # another CRS
    from pyproj import CRS

    ref31 = dict(ref, crs_wkt=CRS.from_epsg(32631).to_wkt("WKT1_GDAL"))
    with pytest.raises(dcz.FormatError, match="is not the CRS"):
        dcz.normalise(req2, lambda m: None, ref_grid=ref31)


def test_output_format_report():
    desc = {"name": "Zarr", "title": "Zarr", "gis_data_types": ["raster"], "experimental": True,
            "parameters": {"chunk_size": {"type": "object", "default": None,
                                          "description": "chunking"}}}
    lines = dcz.report_output_format(desc)
    assert "'experimental': True" in lines[0] and any("chunk_size" in ln for ln in lines)


def test_unexpected_formats_fail_clearly(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=14)
    # a requested band is missing
    part = {b: a for b, a in data.items() if b != "CLD"}
    raw = write_raw_v2(tmp_path / "src" / "m.zarr", t, y, x, part, wkt)
    req = make_request(tmp_path / "c1")
    place(req, [("m.zarr", raw, "x")])
    with pytest.raises(dcz.FormatError, match=r"requested bands \['CLD'\] not found"):
        dcz.normalise(req, lambda m: None)
    assert "NOT NORMALISABLE" in req.sidecar("format.txt").read_text(encoding="utf-8")
    assert not req.store.exists()
    # nothing readable: a directory asset announced as application/x-zarr
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b'{"error": "not a file"}')
    req2 = make_request(tmp_path / "c2")
    place(req2, [("openEO.zarr", junk, "application/x-zarr")])
    with pytest.raises(dcz.FormatError, match="did not download as a store"):
        dcz.normalise(req2, lambda m: None)
    # not on the 10 m grid
    raw3 = write_raw_v2(tmp_path / "src" / "r.zarr", t, y, X0 + 20.0 * np.arange(len(x)), data, wkt)
    req3 = make_request(tmp_path / "c3")
    place(req3, [("r.zarr", raw3, "x")])
    with pytest.raises(dcz.FormatError, match="not the 10 m grid"):
        dcz.normalise(req3, lambda m: None)


def test_zip_members_never_leave_the_extraction_folder(tmp_path):
    import zipfile

    req = make_request(tmp_path / "cubes")
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../outside.txt", "x")
        zf.writestr("ok/.zgroup", '{"zarr_format": 2}')
    place(req, [("evil.zip", z, "application/zip")])
    out = dcz.extract_zips(req.part, log=lambda m: None)
    assert (out[0] / "outside.txt").is_file() and not (tmp_path / "outside.txt").exists()


def test_asset_paths():
    base = "https://s3.example.invalid/openeo/jobs/j-1/results/assets"
    assert dcz.asset_relpath("0.0.0", f"{base}/openEO.zarr/B03/0.0.0?X-Sig=abc") == \
        Path("openEO.zarr/B03/0.0.0")
    assert dcz.asset_relpath(".zarray", f"{base}/openEO.zarr/B03/.zarray") == \
        Path("openEO.zarr/B03/.zarray")
    assert dcz.asset_relpath("openEO.zarr.zip", f"{base}/openEO.zarr.zip") == Path("openEO.zarr.zip")
    assert dcz.asset_relpath("../../x.nc", "https://h/x.nc") == Path("x.nc")


class _FakeResults:
    def __init__(self, meta):
        self.meta = meta

    def get_metadata(self, force=False):
        return self.meta


class _FakeResultJob:
    def __init__(self, meta):
        self.job_id = "j-files"
        self.results = _FakeResults(meta)
        self.connection = None

    def get_results(self):
        return self.results


def test_store_delivered_one_asset_per_file(tmp_path, wkt, monkeypatch):
    """One asset per Zarr file under opaque keys: the hrefs place the files."""
    from openeo.rest.job import ResultAsset

    t, y, x, data = synth(2023, seed=15)
    raw = write_raw_v2(tmp_path / "src" / "openEO.zarr", t, y, x, data, wkt)
    base = "https://s3.example.invalid/jobs/j-files/results/assets"
    files, assets = {}, {}
    for k, f in enumerate(sorted(p for p in raw.rglob("*") if p.is_file())):
        href = f"{base}/{f.relative_to(raw.parent).as_posix()}?sig=1"
        files[href] = f
        assets[f"asset_{k}"] = {"href": href, "type": "application/octet-stream",
                                "file:size": f.stat().st_size}

    def fake_download(self, target=None, **kw):
        shutil.copy2(files[self.href], target)
        return Path(target)

    monkeypatch.setattr(ResultAsset, "download", fake_download)
    req = make_request(tmp_path / "cubes")
    dcz.download_results(_FakeResultJob({"assets": assets}), req, lambda m: None)
    assert (Path(req.part) / "assets" / "openEO.zarr" / "B03" / ".zarray").is_file()
    normalise_and_write(req)
    check_store(req.store, t, y, x, data)


def test_an_asset_without_bands_is_skipped(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=16)
    a1 = write_raw_v2(tmp_path / "src" / "cube.zarr", t, y, x, data, wkt)
    g = zarr.open_group(str(tmp_path / "src" / "meta.zarr"), mode="w", zarr_format=2)
    g.create_array("quality", shape=(3,), dtype="int16")[:] = 1
    req = make_request(tmp_path / "cubes")
    place(req, [("cube.zarr", a1, "x"), ("meta.zarr", tmp_path / "src" / "meta.zarr", "x")])
    _, rep, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert any("asset skipped" in ln for ln in rep)


# ─────────────────────────────────────────────────────────────────────────────
#  open_cube across yearly stores
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_years(tmp_path, wkt):
    """Canonical stores for 2023 and 2024 written by the real writer."""
    root = tmp_path / "cubes"
    out = {}
    for year in (2023, 2024):
        t, y, x, data = synth(year, n=7, seed=20)
        raw = write_raw_v2(tmp_path / f"src{year}" / "openEO.zarr", t, y, x, data, wkt)
        req = make_request(root, year=year)
        place(req, [("openEO.zarr.zip", zip_store(raw, tmp_path / f"src{year}" / "z"),
                     "application/zip")])
        normalise_and_write(req)
        out[year] = (t, y, x, data)
    return root, out


def test_open_cube_windows_across_years(two_years):
    root, src = two_years
    assert zc.list_years("villaviciosa", root) == [2023, 2024]
    assert zc.available(root) == {"villaviciosa": [2023, 2024]}
    all_t = np.concatenate([src[y][0].values for y in (2023, 2024)])
    ds = zc.open_cube("villaviciosa", "2023-03-01", "2024-03-01", root=root)
    want = all_t[(all_t >= np.datetime64("2023-03-01")) & (all_t < np.datetime64("2024-03-01"))]
    assert list(ds["t"].values) == list(want) and ds.attrs["years"] == [2023, 2024]
    assert sorted(zc.band_names(ds)) == sorted(BANDS)
    assert all(ds[b].dtype == np.int16 and zc.fill_of(ds[b]) == FILL for b in BANDS)
    # values come from the right year and date
    k = int(np.flatnonzero(src[2024][0].values == want[-1])[0])
    np.testing.assert_array_equal(ds["B11"].isel(t=-1).values, src[2024][3]["B11"][k])
    # one year only: the other store is not even opened
    one = zc.open_cube("villaviciosa", "2024-01-01", "2025-01-01", root=root, bands=["B03", "SCL"])
    assert one.attrs["years"] == [2024] and zc.band_names(one) == ["B03", "SCL"]
    assert len(one["t"]) == 7
    # end exclusive: a window ending on a scene date leaves it out
    d = pd.Timestamp(src[2023][0][2])
    ex = zc.open_cube("villaviciosa", src[2023][0][0], d, root=root)
    assert pd.Timestamp(ex["t"].values[-1]) < d and len(ex["t"]) == 2
    # int years and masked float view
    m = zc.open_cube("villaviciosa", 2023, 2024, root=root, bands=["B04"], masked=True)
    assert m["B04"].dtype == np.float32
    assert np.isnan(m["B04"].isel(t=0).values[:3, :5]).all()


def test_verify_only_cli(two_years, tmp_path, wkt, monkeypatch, capsys):
    """--verify-only: offline comparison with the netCDF cube, JSON beside the store."""
    root, src = two_years
    t, y, x, data = src[2023]
    nc = write_nc(tmp_path / "ref.nc", t, y, x, data, wkt)
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(nc),
        reference_period=("2023-01-01", "2024-01-01")))
    monkeypatch.chdir(tmp_path)
    zc.write_json({str(d.date()): f"{d.date()}T11:00:00" for d in t},
                  zc.overpass_path("villaviciosa", root))
    assert dcz.main(["villaviciosa", "--years", "2023", "--verify-only", "--root", str(root),
                     "--products", "main"]) == 0
    rep = json.loads(zc.sidecar_path("villaviciosa", 2023, "verify.json", root).read_text())
    assert rep["verdict"] == "identical" and rep["dates"]["common"] == len(t)
    assert "IDENTICAL" in capsys.readouterr().out
    # with the default products the angles of 2023 are missing
    assert dcz.main(["villaviciosa", "--years", "2023", "--verify-only", "--root", str(root)]) == 1
    # 2024 lacks overpass times and is outside the reference; 2025 is not downloaded
    assert dcz.main(["villaviciosa", "--years", "2024-2025", "--verify-only", "--root",
                     str(root), "--products", "main"]) == 1
    rep24 = json.loads(zc.sidecar_path("villaviciosa", 2024, "verify.json", root).read_text())
    assert rep24["verdict"] == "no reference"


def test_open_cube_refuses_missing_years_and_other_grids(two_years, tmp_path, wkt):
    root, src = two_years
    with pytest.raises(FileNotFoundError, match=r"needs years \[2025\]"):
        zc.open_cube("villaviciosa", "2024-06-01", "2025-06-01", root=root)
    part = zc.open_cube("villaviciosa", "2024-06-01", "2025-06-01", root=root, missing="skip")
    assert part.attrs["missing_years"] == [2025]
    with pytest.raises(FileNotFoundError):
        zc.open_cube("ferrol", "2023-01-01", "2024-01-01", root=root)
    # a 2025 store on a shifted grid cannot be concatenated
    t, y, x, data = synth(2025, n=3, seed=21, x0=X0 + 10.0)
    raw = write_raw_v2(tmp_path / "src25" / "openEO.zarr", t, y, x, data, wkt)
    req = make_request(root, year=2025)
    place(req, [("z.zip", zip_store(raw, tmp_path / "src25" / "z"), "application/zip")])
    normalise_and_write(req)
    with pytest.raises(ValueError, match="not on the grid"):
        zc.open_cube("villaviciosa", "2024-06-01", "2025-06-01", root=root)
    # a store without the completion mark is not listed
    bad = zc.store_path("villaviciosa", 2026, root)
    shutil.copytree(zc.store_path("villaviciosa", 2024, root), bad)
    meta = json.loads((bad / ".zmetadata").read_text())
    meta["metadata"][".zattrs"]["complete"] = 0
    (bad / ".zmetadata").write_text(json.dumps(meta))
    assert 2026 not in zc.list_years("villaviciosa", root)


# ─────────────────────────────────────────────────────────────────────────────
#  verification against a netCDF cube
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_with_netcdf_counts(tmp_path, wkt):
    t, y, x, data = synth(2023, n=8, seed=30)
    # reference: dates 0..6 (date 7 'missing'), plus a date the store lacks
    ref_t = list(t[:7]) + [pd.Timestamp("2023-12-20")]
    ref = {b: np.concatenate([data[b][:7], data[b][:1]]) for b in ("B03", "B08", "SCL")}
    nc = write_nc(tmp_path / "ref.nc", ref_t, y, x, ref, wkt)
    # store: alter 5 B08 values on date 2, put 3 SCL pixels to no data on date 3,
    # drop the 2023-12-20 scene, and add a 2023-12-31 scene outside the period
    st = {b: a.copy() for b, a in data.items()}
    st["B08"][2, 10, 10:15] += 1
    st["SCL"][3, 5, 5:8] = FILL
    st_t = list(t) + [pd.Timestamp("2023-12-31")]
    st = {b: np.concatenate([a, a[:1]]) for b, a in st.items()}
    raw = write_raw_v2(tmp_path / "src" / "s.zarr", st_t, y, x, st, wkt)
    req = make_request(tmp_path / "cubes")
    place(req, [("s.zarr", raw, "x")])
    normalise_and_write(req)
    rep = zc.compare_with_netcdf(req.store, nc, start="2023-01-01", end="2024-01-01",
                                 ref_period=("2023-01-01", "2023-12-31"))
    assert rep["grid"]["identical"] is True
    d = rep["dates"]
    assert d["common"] == 7
    assert d["only_store"] == [str(t[7].date())]
    assert d["only_reference"] == ["2023-12-20"]
    assert d["outside_reference_period"] == ["2023-12-31"]
    assert rep["bands"]["B08"]["value_differs"] == 5 and rep["bands"]["B08"]["max_abs_diff"] == 1
    assert rep["bands"]["SCL"]["nodata_only_store"] == 3
    assert rep["bands"]["B03"]["differing_pixels"] == 0
    n_px = 7 * len(y) * len(x)
    assert rep["bands"]["B08"]["share_differing"] == pytest.approx(5 / n_px)
    assert rep["verdict"] == "differs"
    assert set(rep["dates_with_differences_listed"]) == {str(t[2].date()), str(t[3].date())}
    assert rep["bands_not_in_reference"] == ["B02", "B04", "B8A", "B11", "B12", "CLD"]
    assert any("B08" in ln for ln in zc.report_lines(rep))


def test_compare_identical_subwindow_and_shifted(tmp_path, wkt):
    t, y, x, data = synth(2023, n=5, H=30, W=44, seed=31)
    nc = write_nc(tmp_path / "ref.nc", t, y, x, data, wkt)
    raw = write_raw_v2(tmp_path / "src" / "a.zarr", t, y, x, data, wkt)
    req = make_request(tmp_path / "c1")
    place(req, [("a.zarr", raw, "x")])
    normalise_and_write(req)
    rep = zc.compare_with_netcdf(req.store, nc)
    assert rep["verdict"] == "identical"
    # a test-like window inside the reference grid
    sub = {b: a[:, 4:20, 7:30] for b, a in data.items()}
    raw2 = write_raw_v2(tmp_path / "src" / "b.zarr", t, y[4:20], x[7:30], sub, wkt)
    req2 = make_request(tmp_path / "c2")
    place(req2, [("b.zarr", raw2, "x")])
    normalise_and_write(req2)
    rep2 = zc.compare_with_netcdf(req2.store, nc)
    assert rep2["verdict"] == "identical on the common window"
    assert rep2["grid"]["common_window"]["reference_rows"] == [4, 20]
    assert rep2["grid"]["common_window"]["reference_cols"] == [7, 30]
    # corners instead of centres: half a pixel off, not comparable
    raw3 = write_raw_v2(tmp_path / "src" / "c.zarr", t, y + 5.0, x - 5.0, data, wkt)
    req3 = make_request(tmp_path / "c3")
    place(req3, [("c.zarr", raw3, "x")])
    normalise_and_write(req3)
    rep3 = zc.compare_with_netcdf(req3.store, nc)
    assert rep3["verdict"] == "not comparable" and rep3["grid"]["x_offset_px"] == 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  the process, the CLI and the job flow
# ─────────────────────────────────────────────────────────────────────────────

def test_years_and_windows():
    assert dcz.parse_years(["2017-2022"]) == [2017, 2018, 2019, 2020, 2021, 2022]
    assert dcz.parse_years(["2023,2025", "2024"]) == [2023, 2024, 2025]
    with pytest.raises(ValueError):
        dcz.parse_years(["2025-2023"])
    with pytest.raises(ValueError):
        dcz.parse_years(["1999"])
    assert zc.year_window(2023) == ("2023-01-01", "2024-01-01")
    with pytest.raises(SystemExit):
        dcz.parse_args(["santander"])


def test_job_options_are_those_of_download_cube():
    src = (dcz.ROOT / "experiments" / "download_cube.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    vals = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(getattr(tg, "id", None) == "JOB_OPTIONS" for tg in n.targets)]
    assert vals == [dcz.JOB_OPTIONS]


def test_same_process_as_the_existing_cubes(tmp_path, monkeypatch):
    """SentinelCube.ensure's graph == ours, but for the THREE intended
    differences of the module docstring: the bands, the format, and the
    spatial extent (the site's frame in its UTM CRS, ``crs`` an EPSG integer,
    instead of the lon/lat box). ``--lonlat-extent`` gives the old extent
    back: then only the bands and the format differ."""
    import copy

    from openeo.rest.datacube import DataCube
    from openeo.rest.result import SaveResult

    import pyintertidal as pit
    from pyintertidal.cube import SentinelCube

    captured = {}

    class Stop(Exception):
        pass

    def fake_create_job(self, **kw):
        captured["graph"] = self.flat_graph()
        captured["kw"] = kw
        raise Stop

    class Conn:
        def load_collection(self, cid, **kw):
            return DataCube.load_collection(cid, connection=None, **kw)

    monkeypatch.setattr(SaveResult, "create_job", fake_create_job)
    cube = SentinelCube(pit.sites.get("ferrol"), ("2023-01-01", "2024-01-01"), water="ndwi",
                        cache_path=str(tmp_path / "x.nc"), resolution=10)
    with pytest.raises(Stop):
        cube.ensure(Conn(), job_options=dcz.JOB_OPTIONS, verbose=False)
    old = captured["graph"]
    assert old["loadcollection1"]["arguments"]["bands"] == ["B03", "B08", "SCL"]
    assert old["loadcollection1"]["arguments"]["spatial_extent"] == dcz.site_bbox("ferrol")
    assert captured["kw"]["job_options"] == dcz.JOB_OPTIONS
    root = tmp_path / "cubes"
    req = dcz.year_requests(["ferrol"], [2023], root)[0]
    new = dcz.build_process(req).flat_graph()
    lc = new["loadcollection1"]["arguments"]
    assert lc["bands"] == list(BANDS)
    assert new["saveresult1"]["arguments"]["format"] == "Zarr"
    # the extent: the frame of the site's reference grid (its 10 m netCDF cube where it is on
    # this machine, else the lon/lat box snapped outward to 60 m), in UTM zone 29
    ext = lc["spatial_extent"]
    assert ext == dcz.extent_of(dcz.site_frame("ferrol", root)) == req.extent
    assert sorted(ext) == ["crs", "east", "north", "south", "west"]
    assert ext["crs"] == 32629 and type(ext["crs"]) is int
    assert all(ext[k] % 10 == 0 for k in ("west", "south", "east", "north"))
    want = copy.deepcopy(old)
    want["loadcollection1"]["arguments"].update(bands=list(BANDS), spatial_extent=ext)
    want["saveresult1"]["arguments"]["format"] = "Zarr"
    assert want == new                                    # nothing else differs
    # --lonlat-extent: the request before 2026-10-13
    lreq = dcz.year_requests(["ferrol"], [2023], root, lonlat=True)[0]
    assert lreq.extent is None and lreq.spatial_extent == dcz.site_bbox("ferrol")
    want_ll = copy.deepcopy(old)
    want_ll["loadcollection1"]["arguments"]["bands"] = list(BANDS)
    want_ll["saveresult1"]["arguments"]["format"] = "Zarr"
    assert dcz.build_process(lreq).flat_graph() == want_ll
    assert dcz.job_title(lreq, "0" * 12) == dcz.job_title(req, "0" * 12)     # titles: by the hash


def test_predicted_grid_matches_the_existing_cubes():
    cases = [(dcz.SITE_CONFIG["villaviciosa"]["bbox"], "ndwi_cube_villaviciosa_grande_10y.nc", 10),
             ("ferrol", "ndwi_cube_ferrol_2023-2025_10m.nc", 10),
             ("escalda", "ndwi_cube_escalda_2023-2025_20m.nc", 20)]
    seen = 0
    for bbox, nc, res in cases:
        path = dcz.ROOT / nc
        if not path.is_file():
            continue
        bbox = dcz.site_bbox(bbox) if isinstance(bbox, str) else bbox
        g = dcz.predict_grid(bbox, res=res)
        with xr.open_dataset(path) as ds:
            assert g["shape"] == [ds.sizes["y"], ds.sizes["x"]]
            assert g["x_edges"][0] == float(ds["x"][0]) - res / 2
            assert g["y_edges"][1] == float(ds["y"][0]) + res / 2
        seen += 1
    if not seen:
        pytest.skip("no existing cube on this machine")
    tb = dcz.test_bbox()
    vb = dcz.SITE_CONFIG["villaviciosa"]["bbox"]
    assert vb["west"] < tb["west"] < tb["east"] < vb["east"]
    assert vb["south"] < tb["south"] < tb["north"] < vb["north"]
    assert abs(dcz.predict_grid(tb)["shape"][0] - 200) <= 8


def test_dry_run_is_offline(tmp_path, monkeypatch, capsys):
    import socket

    def no_net(*a, **k):
        raise AssertionError("network used in a dry run")

    monkeypatch.setattr(socket, "create_connection", no_net)
    monkeypatch.chdir(tmp_path)
    assert dcz.main(["--test", "--dry-run", "--root", str(tmp_path / "cubes")]) == 0
    out = capsys.readouterr().out
    for s in ('"load_collection"', '"Zarr"', '"B8A"', '"CLD"', '"near"', "TOTAL", "2025-06-16"):
        assert s in out
    # the UTM extents of the --test window (the same with the netCDF frame or the snapped box)
    for s in ("spatial extent: EPSG:32630 west 305400 south 4820340 east 307440 north 4822380 "
              "(204 x 204 px of 10 m)",
              "spatial extent: EPSG:32630 west 305400 south 4820100 east 307500 north 4822500 "
              "(8 x 7 cells of 300 m)", "SPATIAL EXTENTS requested", '"crs": 32630',
              "lon/lat box (kept in the provenance, not requested)"):
        assert s in out, s
    assert not (tmp_path / "cubes").exists()


def test_site_lock(tmp_path):
    with dcz.SiteLock(tmp_path):
        with pytest.raises(dcz.LockedError):
            with dcz.SiteLock(tmp_path):
                pass
    import socket

    (tmp_path / ".download.lock").write_text(json.dumps(
        {"host": socket.gethostname(), "pid": 2 ** 22 + 12345, "started_utc": "x"}))
    with dcz.SiteLock(tmp_path) as lk:                  # a dead holder is replaced
        assert lk.mine
    assert not (tmp_path / ".download.lock").exists()


class FakeJob:
    def __init__(self, conn, job_id):
        self.connection = self.conn = conn
        self.job_id = job_id

    def describe(self):
        from openeo.rest import OpenEoApiError

        j = self.conn.jobs.get(self.job_id)
        if j is None:
            raise OpenEoApiError(http_status_code=404, code="JobNotFound", message="gone")
        if j["status"] in ("queued", "running") and j["plan"]:
            j["status"] = j["plan"].pop(0)
        return {"id": self.job_id, "title": j["title"], "status": j["status"],
                "created": j["created"], "costs": 2.5, "usage": {"cpu": {"value": 9}}}

    def start(self):
        self.conn.starts.append(self.job_id)
        if self.conn.refuse:
            raise self.conn.refuse.pop(0)
        self.conn.jobs[self.job_id]["status"] = "queued"
        return self

    def logs(self, level=None):
        return [{"id": "1", "level": "error", "message": f"executor lost in {self.job_id}"}]


class FakeSaveResult:
    def __init__(self, conn, flat):
        self.conn, self.flat = conn, flat

    def flat_graph(self):
        return self.flat

    def create_job(self, title=None, job_options=None):
        jid = f"j-{len(self.conn.jobs) + 1}"
        plan = self.conn.plans.pop(0) if self.conn.plans else ["running", "finished"]
        self.conn.jobs[jid] = {"title": title, "status": "created", "plan": list(plan),
                               "created": f"2026-10-08T00:00:{len(self.conn.jobs):02d}Z",
                               "options": job_options}
        self.conn.created.append(jid)
        return FakeJob(self.conn, jid)


class FakeConn:
    def __init__(self, plans=None, refuse=None):
        self.jobs, self.created, self.starts = {}, [], []
        self.plans, self.refuse = list(plans or []), list(refuse or [])

    def job(self, job_id):
        return FakeJob(self, job_id)

    def list_jobs(self, limit=100):
        self.list_limit = limit
        return [{"id": k, "title": v["title"], "status": v["status"], "created": v["created"]}
                for k, v in self.jobs.items()][:limit]

    def list_file_formats(self):
        return {"input": {}, "output": {"ZARR": {"title": "Zarr", "experimental": True,
                                                 "gis_data_types": ["raster"], "parameters": {}}}}


@pytest.fixture
def fake_build(monkeypatch):
    real = dcz.build_process

    def build(req, connection=None, backend_format=dcz.BACKEND_FORMAT, format_options=None):
        return FakeSaveResult(connection, real(req, None, backend_format, format_options).flat_graph())

    monkeypatch.setattr(dcz, "build_process", build)


def _ctx(conn, retries=1):
    sleeps = []
    ctx = dcz.Context(None, connect=lambda: conn, sleep=sleeps.append)
    ctx.retries = retries
    ctx.poll_s = ctx.poll_max_s = 0.0
    return ctx, sleeps


def test_job_created_then_reattached_never_twice(tmp_path, fake_build):
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    req = make_request(tmp_path / "cubes")
    job, info = dcz.ensure_finished_job(ctx, req, "T", lambda m: None)
    assert info["status"] == "finished" and conn.created == ["j-1"] and conn.starts == ["j-1"]
    led = json.loads((Path(req.part) / "job.json").read_text())
    assert led["jobs"][-1]["job_id"] == "j-1" and led["jobs"][-1]["status"] == "finished"
    # a second call (e.g. the download failed) reuses the finished job
    dcz.ensure_finished_job(ctx, req, "T", lambda m: None)
    assert conn.created == ["j-1"] and conn.starts == ["j-1"]
    # a session killed while the job ran: the next run reattaches, no new job
    conn.jobs["j-1"].update(status="running", plan=["running", "finished"])
    dcz.ensure_finished_job(ctx, req, "T", lambda m: None)
    assert conn.created == ["j-1"] and conn.starts == ["j-1"]


def test_orphan_found_by_title(tmp_path, fake_build):
    conn = FakeConn()
    conn.jobs["j-77"] = {"title": "T", "status": "running", "plan": ["finished"],
                         "created": "2026-10-07T10:00:00Z"}
    ctx, _ = _ctx(conn)
    req = make_request(tmp_path / "cubes")
    job, _ = dcz.ensure_finished_job(ctx, req, "T", lambda m: None)
    assert job.job_id == "j-77" and conn.created == [] and conn.starts == []
    assert conn.list_limit == 1000                       # not the client's default 100
    led = json.loads((Path(req.part) / "job.json").read_text())
    assert led["jobs"][0]["source"] == "orphan"
    # --new-job: no orphan adoption, a paid fresh job
    conn.jobs["j-78"] = {"title": "T", "status": "finished", "plan": [],
                         "created": "2026-10-07T11:00:00Z"}
    ctx.new_job = True
    job2, _ = dcz.ensure_finished_job(ctx, make_request(tmp_path / "c2"), "T", lambda m: None)
    assert job2.job_id not in ("j-77", "j-78") and conn.created == [job2.job_id]


def test_failed_job_logs_kept_and_retried(tmp_path, fake_build):
    conn = FakeConn(plans=[["running", "error"], ["running", "finished"]])
    ctx, _ = _ctx(conn, retries=1)
    req = make_request(tmp_path / "cubes")
    msgs = []
    job, _ = dcz.ensure_finished_job(ctx, req, "T", msgs.append)
    assert job.job_id == "j-2" and conn.created == ["j-1", "j-2"]
    # the second job gets twice the executor memory; the ledger keeps what was submitted
    assert conn.jobs["j-1"]["options"] == dcz.JOB_OPTIONS
    assert conn.jobs["j-2"]["options"] == {"executor-memory": "8G", "executor-memoryOverhead": "8G"}
    led = json.loads((Path(req.part) / "job.json").read_text())
    assert led["jobs"][-1]["job_options"]["executor-memory"] == "8G"
    assert "loadcollection1" in led["jobs"][-1]["process_graph"]
    assert any("memory raised after 1 failed job" in m for m in msgs)
    errs = json.loads((Path(req.part) / "job_j-1.errors.json").read_text())
    assert "executor lost in j-1" in errs[0]["message"]
    assert any("backend error: executor lost in j-1" in m for m in msgs)
    conn2 = FakeConn(plans=[["running", "error"]])
    ctx2, _ = _ctx(conn2, retries=0)
    with pytest.raises(dcz.JobFailed):
        dcz.ensure_finished_job(ctx2, make_request(tmp_path / "c2"), "T", lambda m: None)
    assert conn2.created == ["j-1"]


def test_start_waits_while_the_backend_is_full(tmp_path, fake_build):
    from openeo.rest import OpenEoApiError

    conn = FakeConn(refuse=[OpenEoApiError(http_status_code=429, code="TooManyRequests",
                                           message="too many concurrent jobs")])
    ctx, sleeps = _ctx(conn)
    dcz.ensure_finished_job(ctx, make_request(tmp_path / "cubes"), "T", lambda m: None)
    assert conn.starts == ["j-1", "j-1"] and ctx.capacity_wait_s in sleeps
    conn2 = FakeConn(refuse=[OpenEoApiError(http_status_code=402, code="PaymentRequired",
                                            message="insufficient credits")])
    ctx2, _ = _ctx(conn2)
    with pytest.raises(dcz.JobFailed, match="credits"):
        dcz.ensure_finished_job(ctx2, make_request(tmp_path / "c2"), "T", lambda m: None)


def test_run_request_end_to_end_and_resume(tmp_path, wkt, monkeypatch, fake_build):
    """Job -> download -> store -> verification -> overpass file; then skipped."""
    t, y, x, data = synth(2023, n=6, seed=40)
    nc = write_nc(tmp_path / "ref.nc", t, y, x, data, wkt)
    pipe = tmp_path / "pipeline_overpass.json"
    pipe.write_text(json.dumps({str(d.date()): f"{d.date()} 11:24:52" for d in t[:5]}))
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", {
        "bbox": dict(dcz.SITE_CONFIG["villaviciosa"]["bbox"]), "reference": str(nc),
        "reference_period": ("2023-01-01", "2023-12-31"), "pipeline_overpass": str(pipe)})
    raw = write_raw_v2(tmp_path / "src" / "openEO.zarr", t, y, x, data, wkt)
    z = zip_store(raw, tmp_path / "src" / "openEO.zarr")

    def fake_download(job, req, log, workers=4):
        place(req, [("openEO.zarr.zip", z, "application/zip")])
        meta = {"type": "Collection", "assets": {"openEO.zarr.zip": {
            "href": "https://example.invalid/openEO.zarr.zip", "type": "application/zip",
            "roles": ["data"], "file:size": z.stat().st_size}}}
        zc.write_json(meta, Path(req.part) / "job-results.json")
        return meta

    monkeypatch.setattr(dcz, "download_results", fake_download)
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    ctx.network = False
    req = make_request(tmp_path / "cubes")
    msgs = []
    res = dcz.run_request(ctx, req, msgs.append)
    assert res["status"] == "done" and res["verdict"] == "identical" and res["dates"] == 6
    assert res["missing_overpass"] == [str(t[5].date())]
    check_store(req.store, t, y, x, data, job_id="j-1")
    ds = zc.open_store(req.store)
    assert ds.attrs["job_id"] == "j-1" and json.loads(ds.attrs["job_costs"]) == 2.5
    assert "resample_spatial" in ds.attrs["process_graph"]
    assert ds.attrs["process_graph_source"] == "submitted with job j-1"   # not rebuilt offline
    assert ds.attrs["grid_reference"].startswith("netcdf: ")
    assert json.loads(ds.attrs["job_options"]) == dcz.JOB_OPTIONS
    # a delivery of the frame's shape whose x/y agree: (0, 0) is the only offset (as before)
    assert ds.attrs["registration_method"] == "frame-sized"
    assert json.loads(ds.attrs["delivered_offset_px"]) == [0, 0]
    assert json.loads(ds.attrs["verification"])["how"] == "identical"
    assert res["accepted"] == "identical" and res["registration"] == "frame-sized [0, 0]"
    ds.close()
    ver = json.loads(req.sidecar("verify.json").read_text())
    assert ver["radiometry"]["offset_state"] in ("no offset", "+1000 in the values", "unclear")
    assert res["radiometry"] == ver["radiometry"]["offset_state"]
    over = json.loads(zc.overpass_path("villaviciosa", req.root).read_text())
    assert over[str(t[0].date())] == f"{t[0].date()}T11:24:52"
    assert json.loads(req.sidecar("verify.json").read_text())["verdict"] == "identical"
    assert req.sidecar("job.json").is_file() and req.sidecar("format.txt").is_file()
    assert not Path(req.part).exists()                  # raw removed once the store is whole
    # a second run downloads nothing
    res2 = dcz.run_request(ctx, req, msgs.append)
    assert res2["status"] == "skipped" and conn.created == ["j-1"]


def test_failed_normalisation_keeps_the_raw_for_the_next_run(tmp_path, wkt, monkeypatch,
                                                             fake_build):
    t, y, x, data = synth(2023, n=4, seed=41)
    nodata = {b: a for b, a in data.items() if b != "B12"}
    raw = write_raw_v2(tmp_path / "src" / "openEO.zarr", t, y, x, nodata, wkt)
    z = zip_store(raw, tmp_path / "src" / "openEO.zarr")
    calls = []

    def fake_download(job, req, log, workers=4):
        calls.append(job.job_id)
        place(req, [("openEO.zarr.zip", z, "application/zip")])
        return {}

    monkeypatch.setattr(dcz, "download_results", fake_download)
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    ctx.network = False
    req = make_request(tmp_path / "cubes")
    with pytest.raises(dcz.FormatError):
        dcz.run_request(ctx, req, lambda m: None)
    assert (Path(req.part) / "assets" / "openEO.zarr.zip").is_file()
    with pytest.raises(dcz.FormatError):                # the retry reuses the download
        dcz.run_request(ctx, req, lambda m: None)
    assert calls == ["j-1"] and conn.created == ["j-1"]


def test_format_test_run(tmp_path, wkt, monkeypatch, fake_build):
    """--test with the UTM extent, the delivery shaped like the backend's Zarr
    of 2026-10-13 (zipped v2, >i2, Blosc lz4 chunks [1, 128, 128], x/y
    float32, time in ms since 01/01/1970, arrays padded to 256 x 256 px with
    the fill): the store is the 204 x 204 px window, bitwise (axes = the
    lattice's float64 values, values = the delivered ones); report, a store
    opened through open_cube, comparison on the common window of the
    reference cube, raw download kept.

    Since 2026-10-19 the window is REGISTERED against the reference (the
    three dates, SCL clear: ``scl_clear``, so the registration has its dates):
    the delivered data sit at row 0, col 0 of the padded arrays here, so the
    registration finds (0, 0) and the store is the one the naive crop made
    (the golden of before), now established by the data, not the x/y."""
    t = pd.date_range("2025-06-02", periods=3, freq="5D")
    rx, ry = axes_of(TEST_FRAME)
    nc = write_nc(tmp_path / "ref.nc", t, ry, rx, {b: lattice_values(b, 3, rx, ry, scl_clear=True)
                                                  for b in ("B03", "B08", "SCL")}, wkt)
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(nc),
        reference_period=("2016-01-01", "2025-12-31"), pipeline_overpass=str(tmp_path / "none.json")))
    req = dcz.test_request(tmp_path / "cubes")
    assert req.extent == {"west": 305400, "south": 4820340, "east": 307440, "north": 4822380,
                          "crs": 32630}
    wx, wy = axes_of(TEST_WIN)
    assert len(wx) == len(wy) == 204
    dx, dy = server_axes(TEST_WIN[0], TEST_WIN[3], 256, 256)       # 2 x 2 chunks of 128 px
    data = {}
    for b in BANDS:
        a = np.full((3, 256, 256), FILL, "int16")
        a[:, :204, :204] = lattice_values(b, 3, wx, wy, scl_clear=True)
        data[b] = a
    xf, yf = f32_like_the_backend(dx, dy)
    raw = write_server_zarr(tmp_path / "src" / "out.zarr", t, yf, xf, data, wkt, extent=TEST_WIN)
    z = zip_store(raw, tmp_path / "src" / "out.zarr")

    def fake_download(job, req, log, workers=4):
        place(req, [("out.zarr.zip", z, "application/zip")])
        return {"assets": {"out.zarr.zip": {"href": "https://h/out.zarr.zip",
                                            "type": "application/zip"}}}

    monkeypatch.setattr(dcz, "download_results", fake_download)
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    ctx.network = False
    msgs = []
    res = dcz.run_request(ctx, req, msgs.append)
    text = "\n".join(msgs)
    assert res["status"] == "done" and res["verdict"] == "identical on the common window"
    assert "with the warnings above" not in text
    assert conn.jobs["j-1"]["title"].startswith("zarr_cube_test_villaviciosa_2025-06-01_2025-06-16_")
    for s_ in ("backend output format 'ZARR'", "FORMAT REPORT", "zip extracted", "open_cube(",
               "p1/p50/p99", "SUMMARY", "band naming: B02<-B02", "B11 100.0 %", "SCL 100.0 %",
               "extent EPSG:32630 west 305400 south 4820340 east 307440 north 4822380 (204 x 204 px",
               "is not the --test window", "cropped rows 0:204 cols 0:204",
               "cropped-away margin", ": only the fill",
               "the delivered 'extent' attribute [305400.0, 4820340.0, 307440.0, 4822380.0] is the "
               "requested extent",
               "x delivered as >f4 (float32)", "the lattice's exact float64 values are used",
               "'milliseconds since 01/01/1970 - 00:00:00' decoded explicitly",
               "global attribute '_CRS'", "=> normalisable: YES",
               "registration (netcdf): the frame (204 x 204 px) starts at row 0, col 0 of the "
               "delivered 256 x 256 px arrays (their x/y put it at row 0, col 0)",
               "at the registered offset (netcdf)", "marked complete"):
        assert s_ in text, s_
    assert "WARNING" not in text.split("=> normalisable")[0].split("interpretation:")[-1]
    # the store IS the window: the lattice's exact axes and the delivered values, bitwise
    ds = zc.open_store(req.store)
    assert ds["x"].dtype == np.float64 and ds["y"].dtype == np.float64
    np.testing.assert_array_equal(ds["x"].values, wx)
    np.testing.assert_array_equal(ds["y"].values, wy)
    assert [str(d) for d in pd.DatetimeIndex(ds["t"].values).date] == [str(d.date()) for d in t]
    for b in BANDS:
        assert ds[b].dtype == np.int16 and zc.fill_of(ds[b]) == FILL
        np.testing.assert_array_equal(ds[b].values, lattice_values(b, 3, wx, wy, scl_clear=True))
    assert json.loads(ds.attrs["delivered_offset_px"]) == [0, 0]
    assert ds.attrs["registration_method"] == "netcdf"
    assert ds.attrs["registration_reference"] == str(nc)
    assert len(json.loads(ds.attrs["registration_dates"])) == 3
    assert json.loads(ds.attrs["verification"])["how"] == "identical"
    assert json.loads(ds.attrs["spatial_extent"]) == req.extent
    assert json.loads(ds.attrs["spatial_extent_lonlat"]) == dcz.test_bbox()
    assert ds.attrs["spatial_extent_source"].startswith("the --test window, 2040 x 2040 m")
    assert '"crs": 32630' in ds.attrs["process_graph"]
    ds.close()
    cube = zc.open_cube("villaviciosa", "2025-06-01", "2025-08-01", root=tmp_path / "cubes" / "_test")
    assert cube.sizes["t"] == 3 and zc.band_names(cube) == list(BANDS)
    cube.close()
    assert Path(req.part).is_dir()                       # the test keeps the backend's files
    assert "FORMAT REPORT" in req.sidecar("format.txt").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  review fixes: the report on any error, no-data, reruns, grid, radiometry,
#  per-date stores, memory
# ─────────────────────────────────────────────────────────────────────────────

def test_any_reading_error_still_writes_the_format_report(tmp_path, wkt):
    """An unknown codec, or dates pandas cannot parse: the FORMAT REPORT
    (with the raw Zarr metadata, or the traceback) is written and printed, and
    a FormatError says where it is."""
    t, y, x, data = synth(2023, seed=50)
    raw = write_raw_v2(tmp_path / "src" / "u.zarr", t, y, x, data, wkt)
    meta = json.loads((raw / "B03" / ".zarray").read_text())
    meta["compressor"] = {"id": "some_new_codec", "level": 3}
    (raw / "B03" / ".zarray").write_text(json.dumps(meta))
    req = make_request(tmp_path / "c1")
    place(req, [("u.zarr", raw, "x")])
    msgs = []
    with pytest.raises(dcz.FormatError, match="nothing readable"):
        dcz.normalise(req, msgs.append)
    text = req.sidecar("format.txt").read_text(encoding="utf-8")
    for s_ in ("FORMAT REPORT", "zarr cannot read it (UnknownCodecError", "raw Zarr metadata",
               "some_new_codec", "array B12", "downloaded files under", "NOT NORMALISABLE"):
        assert s_ in text, s_
    assert any("NOT NORMALISABLE" in m for m in msgs) and any("some_new_codec" in m for m in msgs)
    assert not req.store.exists()
    # dates as text pandas cannot parse: a ValueError inside, reported with its traceback
    raw2 = write_raw_v2(tmp_path / "src" / "d.zarr", t, y, x, data, wkt)
    shutil.rmtree(raw2 / "t")
    g = zarr.open_group(str(raw2), mode="r+")
    ta = g.create_array("t", shape=(len(t),), dtype=str)
    ta[:] = np.array([f"day {k}" for k in range(len(t))])
    ta.attrs["_ARRAY_DIMENSIONS"] = ["t"]
    req2 = make_request(tmp_path / "c2")
    place(req2, [("d.zarr", raw2, "x")])
    with pytest.raises(dcz.FormatError, match="while reading the delivered output.*FORMAT REPORT in"):
        dcz.normalise(req2, lambda m: None)
    text2 = req2.sidecar("format.txt").read_text(encoding="utf-8")
    assert "traceback (last lines)" in text2 and "raised in _decode_time" in text2
    assert "B03            shape [6, 24, 40]" in text2          # what was read is still there


def test_zarr_default_fill_value_is_not_no_data(tmp_path, wkt):
    """No no-data attribute and zarr's default fill_value 0: -32768 in the
    data is the no-data, and 0 (clear sky in CLD) stays a value."""
    t, y, x, data = synth(2023, seed=51)
    data["CLD"][:, 4:8, 4:8] = 0
    raw = write_raw_v2(tmp_path / "src" / "z.zarr", t, y, x, data, wkt, fill_value=None)
    assert json.loads((raw / "CLD" / ".zarray").read_text())["fill_value"] == 0
    req = make_request(tmp_path / "cubes")
    place(req, [("z.zarr", raw, "x")])
    interp, rep, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)                 # fill -32768, values untouched
    assert all(interp.fill_sources[b].startswith("-32768 found in the data") for b in BANDS)
    assert any("no-data -32768" in ln for ln in rep)
    ds = zc.open_store(req.store)
    assert ds["CLD"].attrs["delivered_zarr_fill_value"] == 0
    ds.close()
    m = zc.open_store(req.store, masked=True)
    assert (m["CLD"].values[:, 4:8, 4:8] == 0).all()      # clear sky is not no data
    assert np.isnan(m["CLD"].values[0, :3, :5]).all()
    m.close()


def test_ambiguous_no_data_stops_a_year_but_not_the_test(tmp_path, wkt):
    t, y, x, data = synth(2023, seed=52)
    for a in data.values():
        a[a == FILL] = 7                                  # -32768 nowhere
    raw = write_raw_v2(tmp_path / "src" / "a.zarr", t, y, x, data, wkt, fill_value=None)
    req = make_request(tmp_path / "c1")
    place(req, [("a.zarr", raw, "x")])
    with pytest.raises(dcz.AmbiguousNoData, match="AMBIGUOUS NO-DATA"):
        dcz.normalise(req, lambda m: None)
    text = req.sidecar("format.txt").read_text(encoding="utf-8")
    assert "AMBIGUOUS NO-DATA" in text and "--nodata" in text and not req.store.exists()
    # the format test reports it (with the counts) and goes on
    treq = make_request(tmp_path / "c2", test=True)
    place(treq, [("a.zarr", raw, "x")])
    _, interp, lines = dcz.normalise(treq, lambda m: None)
    assert any("AMBIGUOUS NO-DATA" in ln for ln in lines)
    assert any(ln.strip().startswith("CLD: Zarr fill_value 0") and "-32768 x 0" in ln for ln in lines)
    assert any("AMBIGUOUS NO-DATA" in w for w in interp.warnings)
    # --nodata decides
    _, i3, _ = dcz.normalise(req, lambda m: None, nodata=FILL)
    assert all(i3.fill_sources[b] == "--nodata" and i3.fills[b] == FILL for b in BANDS)
    _, i4, _ = dcz.normalise(req, lambda m: None, nodata=0)
    assert i4.fills["CLD"] == 0
    assert dcz.parse_args(["--nodata", "-32768"]).nodata == FILL


def test_changed_request_or_new_job_never_reuses_the_old_download(tmp_path, wkt, monkeypatch,
                                                                  fake_build):
    t, y, x, data = synth(2023, n=4, seed=53)
    nob12 = {b: a for b, a in data.items() if b != "B12"}       # normalisation always fails
    raw = write_raw_v2(tmp_path / "src" / "openEO.zarr", t, y, x, nob12, wkt)
    z = zip_store(raw, tmp_path / "src" / "openEO.zarr")
    calls = []

    def fake_download(job, req, log, workers=4):
        calls.append(job.job_id)
        place(req, [("openEO.zarr.zip", z, "application/zip")])
        return {}

    monkeypatch.setattr(dcz, "download_results", fake_download)
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    ctx.network = False
    req = make_request(tmp_path / "cubes")
    msgs = []
    with pytest.raises(dcz.FormatError):
        dcz.run_request(ctx, req, msgs.append)
    title1 = json.loads((Path(req.part) / "job.json").read_text())["title"]
    # other save_result options: another graph and title -> the old part set aside, a new job
    ctx.format_options = {"chunking": {"t": 16}}
    with pytest.raises(dcz.FormatError):
        dcz.run_request(ctx, req, msgs.append)
    assert calls == ["j-1", "j-2"] and conn.created == ["j-1", "j-2"]
    led = json.loads((Path(req.part) / "job.json").read_text())
    assert led["title"] != title1 and [j["job_id"] for j in led["jobs"]] == ["j-2"]
    aside = dcz.superseded_parts(req)
    assert len(aside) == 1 and json.loads((aside[0] / "job.json").read_text())["title"] == title1
    assert any("the request changed" in m for m in msgs)
    # the part folder deleted: the finished job is found again by its title, nothing paid ...
    shutil.rmtree(req.part)
    with pytest.raises(dcz.FormatError):
        dcz.run_request(ctx, req, msgs.append)
    assert conn.created == ["j-1", "j-2"] and calls[-1] == "j-2"
    # ... --new-job pays for a fresh one, twice in a row without name clashes
    ctx.new_job = True
    for want in ("j-3", "j-4"):
        with pytest.raises(dcz.FormatError):
            dcz.run_request(ctx, req, msgs.append)
        assert conn.created[-1] == want and calls[-1] == want
    assert len(dcz.superseded_parts(req)) == 3


def test_complete_year_is_skipped_even_with_new_job(two_years, fake_build):
    root, _ = two_years
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    ctx.network, ctx.new_job = False, True
    res = dcz.run_request(ctx, make_request(root, year=2023), lambda m: None)
    assert res["status"] == "skipped" and conn.created == []


def test_a_new_job_never_mixes_with_files_of_an_old_one(tmp_path, monkeypatch):
    from openeo.rest.job import ResultAsset

    got = []

    def fake_download(self, target=None, **kw):
        jid = self.href.split("/")[-2]
        got.append((jid, self.name))
        Path(target).write_bytes(f"{jid}-{self.name}".encode())
        return Path(target)

    monkeypatch.setattr(ResultAsset, "download", fake_download)
    req = make_request(tmp_path / "cubes")

    def job(jid, names):
        j = _FakeResultJob({"assets": {n: {"href": f"https://h/{jid}/{n}", "type": "x"}
                                       for n in names}})
        j.job_id = jid
        return j

    dcz.download_results(job("j-1", ["a.nc", "b.nc"]), req, lambda m: None)
    dcz.download_results(job("j-1", ["a.nc", "b.nc"]), req, lambda m: None)   # resumed: kept
    assert got == [("j-1", "a.nc"), ("j-1", "b.nc")]
    msgs = []
    dcz.download_results(job("j-2", ["a.nc"]), req, msgs.append)            # results expired
    raw = Path(req.part) / "assets"
    assert got[-1] == ("j-2", "a.nc") and (raw / "a.nc").read_bytes() == b"j-2-a.nc"
    assert not (raw / "b.nc").exists()
    done = json.loads((Path(req.part) / dcz.ASSETS_DONE).read_text())
    assert done["job_id"] == "j-2" and done["files"] == {"a.nc": {"size": 8, "type": "x",
                                                                  "job_id": "j-2"}}
    assert any("files of job j-1" in m for m in msgs)


def test_other_extent_is_put_on_the_reference_grid(tmp_path, wkt):
    """Same 10 m lattice, another extent: cropped and padded with the fill
    onto the reference grid (an index shift; values untouched)."""
    t, y, x, data = synth(2023, H=24, W=40, seed=54)
    ref = {"path": "ref.nc", "kind": "netcdf", "x": x[2:36], "y": y[3:20], "crs_wkt": wkt}
    sub = {b: a[:, 0:22, 5:40] for b, a in data.items()}     # sticks out on three sides
    raw = write_raw_v2(tmp_path / "src" / "e.zarr", t, y[0:22], x[5:40], sub, wkt)
    req = make_request(tmp_path / "cubes")
    place(req, [("e.zarr", raw, "x")])
    with pytest.raises(dcz.FormatError, match="does not cover the whole reference grid"):
        dcz.normalise(req, lambda m: None, ref_grid=ref)          # 3 columns padded: stops
    interp, _, _ = normalise_and_write(req, ref_grid=ref, allow_padding=True)
    assert any("put on the reference grid" in w for w in interp.warnings)
    # the cropped margin holds data (pixels outside the frame): a note, not a warning
    assert not any("cropped-away margin" in w for w in interp.warnings)
    assert any("cropped-away margin" in n and "holds data" in n and "OUTSIDE the requested frame"
               in n for n in interp.notes)
    ds = zc.open_store(req.store)
    np.testing.assert_array_equal(ds["x"].values, ref["x"])
    np.testing.assert_array_equal(ds["y"].values, ref["y"])
    for b in ("B03", "SCL", "CLD"):
        got = ds[b].values
        assert got.dtype == np.int16
        np.testing.assert_array_equal(got[:, :, 3:], data[b][:, 3:20, 5:36])
        assert (got[:, :, :3] == FILL).all()                  # reference columns 2..4: padded
    ds.close()
    with pytest.raises(dcz.FormatError, match="does not overlap"):
        far = {"path": "far.nc", "x": x + 1000.0, "y": y, "crs_wkt": wkt}
        dcz.normalise(req, lambda m: None, ref_grid=far)


def test_reference_grid_falls_back_to_a_store_then_to_the_bbox(tmp_path, wkt, monkeypatch):
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(tmp_path / "absent.nc")))
    root = tmp_path / "cubes"
    msgs = []
    g = dcz.reference_grid("villaviciosa", root=root, log=msgs.append)
    assert g["kind"] == "predicted" and "GRID GUARD WEAKER" in msgs[-1]
    # the box snapped OUTWARD to multiples of 60 m: the netCDF cube's 915 x 915 px frame
    # (301020 / 4824840, its upper-left corner, already on 60 m) and 3 px more east and south
    assert "snapped outward to multiples of 60 m" in msgs[-1]
    assert (len(g["y"]), len(g["x"])) == (918, 918) and (g["x"][0], g["y"][0]) == (X0, Y0)
    assert (g["x"][-1] + 5.0, g["y"][-1] - 5.0) == (310200.0, 4815660.0)
    t, y, x, data = synth(2023, seed=55)
    raw = write_raw_v2(tmp_path / "src" / "s.zarr", t, y, x, data, wkt)
    req = make_request(root)
    place(req, [("s.zarr", raw, "x")])
    interp, _, _ = normalise_and_write(req, ref_grid=g, allow_padding=True)  # 24 x 40, padded
    ds = zc.open_store(req.store)
    assert ds.attrs["grid_reference"].startswith("predicted: ")
    ds.close()
    g2 = dcz.reference_grid("villaviciosa", root=root, log=msgs.append)
    assert g2["kind"] == "store" and g2["path"] == str(req.store) and "weaker" in msgs[-1]
    assert (len(g2["y"]), len(g2["x"])) == (918, 918)
    fr = dcz.site_frame("villaviciosa", root)             # the store's frame is then requested
    assert dcz.extent_of(fr) == {"west": 301020, "south": 4815660, "east": 310200,
                                 "north": 4824840, "crs": 32630}
    assert fr["origin"].startswith("complete store 2023.zarr")
    # the store's CRS is used when a delivery carries none
    shutil.rmtree(raw / "crs")
    for b in BANDS:
        zarr.open_array(store=str(raw / b), mode="r+").attrs.pop("grid_mapping", None)
    req2 = make_request(tmp_path / "c2")
    place(req2, [("s.zarr", raw, "x")])
    _, i2, _ = dcz.normalise(req2, lambda m: None, ref_grid=g2, allow_padding=True)
    assert i2.crs_source.startswith("reference store")


def test_open_cube_refuses_years_with_different_no_data(tmp_path, wkt):
    root = tmp_path / "cubes"
    t, y, x, data = synth(2023, n=4, seed=56)
    raw = write_raw_v2(tmp_path / "s23" / "a.zarr", t, y, x, data, wkt)
    req = make_request(root, year=2023)
    place(req, [("a.zarr", raw, "x")])
    normalise_and_write(req)
    t4, _, _, d4 = synth(2024, n=4, seed=57)
    for a in d4.values():
        a[a == FILL] = 0                                     # 2024 delivered with no-data 0
    raw4 = write_raw_v2(tmp_path / "s24" / "a.zarr", t4, y, x, d4, wkt, fill_value=None)
    req4 = make_request(root, year=2024)
    place(req4, [("a.zarr", raw4, "x")])
    ds, interp, _ = dcz.normalise(req4, lambda m: None, nodata=0)
    attrs = dcz.provenance(req4, {"id": "j-24"}, {}, interp, dcz.Context(),
                           dcz.build_process(req4).flat_graph())
    dcz.write_store(ds, interp.fills, req4.store, attrs, log=lambda m: None)
    with pytest.raises(ValueError, match="differ in dtype or no-data"):
        zc.open_cube("villaviciosa", 2023, 2025, root=root)
    m = zc.open_cube("villaviciosa", 2023, 2025, root=root, masked=True, bands=["B03"])
    v = m["B03"].values
    assert np.isnan(v[0, :3, :5]).all() and np.isnan(v[4, :3, :5]).all()     # both no-datas
    assert np.isnan(v[-1, -2:, :]).all()
    one = zc.open_cube("villaviciosa", 2024, 2025, root=root)                # one year: raw ok
    assert zc.fill_of(one["B03"]) == 0


def test_reports_use_the_dates_with_data(tmp_path, wkt):
    """The first date has no data at all: the value and resampling reports
    use the dates with most valid pixels, and say how many pairs."""
    t, y, x, data = synth(2025, n=5, H=20, W=30, seed=58, start="2025-06-02")
    iy = (np.floor(y / 20) - np.floor(y / 20).min()).astype(int)
    ix = (np.floor(x / 20) - np.floor(x / 20).min()).astype(int)
    for b in ("B8A", "B11", "B12", "SCL", "CLD"):
        data[b] = data[b][:, :iy.max() + 1, :ix.max() + 1][:, iy][:, :, ix]
    for a in data.values():
        a[0] = FILL
    raw = write_raw_v2(tmp_path / "src" / "r.zarr", t, y, x, data, wkt)
    req = make_request(tmp_path / "cubes", year=2025, test=True)
    place(req, [("r.zarr", raw, "x")])
    _, _, lines = dcz.normalise(req, lambda m: None)
    res = next(ln for ln in lines if ln.startswith("resampling check"))
    assert str(t[0].date()) not in res and str(t[1].date()) in res
    for b in ("B8A", "B11", "B12", "SCL", "CLD"):
        assert f"{b} 100.0 % (n=" in res and f"{b} 100.0 % (n=0)" not in res
    assert any("dates used:" in ln and f"{t[0].date()} 0 %" in ln for ln in lines)
    assert any(ln.startswith("  B12 ") and "DN in [1, 500)" in ln for ln in lines)


def write_raw_2d(path, y, x, arrays, wkt):
    """One date, no time dimension: (y, x) arrays per band."""
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    for b, a in arrays.items():
        arr = g.create_array(b, shape=a.shape, dtype="<i2", chunks=(12, 16), fill_value=FILL)
        arr[:] = a
        arr.attrs.update({"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": "crs"})
    for name, vals in (("x", np.asarray(x)), ("y", np.asarray(y))):
        arr = g.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
        arr[:] = vals
        arr.attrs["_ARRAY_DIMENSIONS"] = [name]
    c = g.create_array("crs", shape=(), dtype="int32")
    c.attrs.update({"crs_wkt": wkt, "_ARRAY_DIMENSIONS": []})
    return Path(path)


def test_one_store_per_date_without_time_dimension(tmp_path, wkt):
    t, y, x, data = synth(2023, n=3, seed=59)
    # the dates in the store names
    req = make_request(tmp_path / "c1")
    items = []
    for k, d in enumerate(t):
        p = write_raw_2d(tmp_path / "src1" / f"openEO_{d.date()}Z.zarr", y, x,
                         {b: a[k] for b, a in data.items()}, wkt)
        items.append((p.name, p, "x"))
    place(req, items[::-1])
    interp, _, _ = normalise_and_write(req)
    check_store(req.store, t, y, x, data)
    assert any("taken from the path" in n for n in interp.notes)
    # opaque names: the dates from the assets' STAC metadata
    req2 = make_request(tmp_path / "c2")
    items, when = [], {}
    for k, d in enumerate(t):
        p = write_raw_2d(tmp_path / "src2" / f"item{k}.zarr", y, x,
                         {b: a[k] for b, a in data.items()}, wkt)
        items.append((p.name, p, "x"))
        when[p.name] = f"{d.date()}T00:00:00Z"
    place(req2, items, datetimes=when)
    interp2, _, _ = normalise_and_write(req2)
    check_store(req2.store, t, y, x, data)
    assert any("taken from the job's STAC metadata" in n for n in interp2.notes)
    # no date anywhere: a clear refusal
    req3 = make_request(tmp_path / "c3")
    place(req3, [("item0.zarr", tmp_path / "src2" / "item0.zarr", "x")])
    with pytest.raises(dcz.FormatError, match="no 't' dimension .* no date"):
        dcz.normalise(req3, lambda m: None)


def test_collect_assets_keeps_the_item_datetime():
    class Resp:
        def __init__(self, js):
            self.js = js

        def json(self):
            return self.js

    class Conn:
        def get(self, href, expected_status=None):
            return Resp({"id": "i1", "properties": {"datetime": "2023-01-03T00:00:00Z"},
                         "assets": {"openEO.zarr.zip": {"href": "https://h/a.zip", "type": "x"}}})

    class Job:
        connection = Conn()

    got = dcz.collect_assets(Job(), {"links": [{"rel": "item", "href": "https://h/i1"}]})
    assert got[0][2]["item_datetime"] == "2023-01-03T00:00:00Z"
    assert dcz.asset_datetime(got[0][2]) == "2023-01-03T00:00:00Z"


def test_radiometry_recorded_and_checked_across_years(tmp_path, wkt):
    import warnings

    root = tmp_path / "cubes"
    for year, offset in ((2023, True), (2024, False)):
        t, y, x, data = synth(year, n=4, seed=60)
        if offset:
            v = data["B12"]
            data["B12"] = np.where(v == FILL, v, v % 9000 + 1000).astype("int16")
        raw = write_raw_v2(tmp_path / f"s{year}" / "a.zarr", t, y, x, data, wkt)
        req = make_request(root, year=year)
        place(req, [("a.zarr", raw, "x")])
        normalise_and_write(req)
    r23 = zc.store_radiometry(zc.store_attrs(zc.store_path("villaviciosa", 2023, root)))
    r24 = zc.store_radiometry(zc.store_attrs(zc.store_path("villaviciosa", 2024, root)))
    assert r23["offset_state"] == "+1000 in the values" and r24["offset_state"] == "no offset"
    assert r23["band"] == "B12" and len(r23["per_date_valid_low"]) == 4
    with pytest.warns(UserWarning, match=r"disagree on the L2A \+1000 offset"):
        ds = zc.open_cube("villaviciosa", 2023, 2025, root=root)
    assert "radiometry_warning" in ds.attrs
    assert json.loads(ds.attrs["radiometry"])["2023"]["offset_state"] == "+1000 in the values"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        one = zc.open_cube("villaviciosa", 2024, 2025, root=root)
    assert "radiometry_warning" not in one.attrs


def test_input_products_and_radiometry_summary():
    meta = {"links": [
        {"rel": "derived_from",
         "href": "https://x/S2B_MSIL2A_20210105T112359_N0214_R037_T30TUP_20210105T125917.SAFE"},
        {"rel": "derived_from",
         "href": "https://x/S2A_MSIL2A_20230110T112441_N0509_R037_T30TUP_20230110T141815.SAFE"},
        {"rel": "derived_from",
         "href": "https://x/S2A_MSIL2A_20230110T112441_N0510_R037_T30TUN_20230110T141815"}]}
    prods = dcz.input_products(meta)
    assert prods == {"2021-01-05": ["02.14"], "2023-01-10": ["05.09", "05.10"]}
    bl = zc.store_baselines({"input_baselines": json.dumps(prods)})
    assert bl["counts"] == {"02.14": 1, "05.09": 1, "05.10": 1} and bl["pre_04_dates"] == 1
    assert dcz.input_products({}) == {} and zc.store_baselines({"input_baselines": "{}"}) is None
    s = dcz.radiometry_summary({"d1": (1000, 0), "d2": (900, 1), "d3": (5, 5)}, 10000, "B12")
    assert s["offset_state"] == "+1000 in the values" and s["dates_used"] == 2
    assert dcz.radiometry_summary({"d1": (1000, 0), "d2": (900, 300)}, 10000, "B12")[
        "offset_state"] == "no offset"
    assert dcz.radiometry_summary({}, 10000, "B12")["offset_state"].startswith("unknown")


def test_retry_gets_more_executor_memory():
    assert dcz.escalate_job_options(dcz.JOB_OPTIONS, 0) == dcz.JOB_OPTIONS
    assert dcz.escalate_job_options(dcz.JOB_OPTIONS, 3) == {"executor-memory": "8G",
                                                             "executor-memoryOverhead": "8G"}
    assert dcz.escalate_job_options({"executor-memory": "3G", "executor-memoryOverhead": "512m",
                                     "driver-memory": "2G"}, 2) == {
        "executor-memory": "8G", "executor-memoryOverhead": "2G", "driver-memory": "2G"}


def test_bands_sharing_chunks_are_written_together_and_memory_is_checked(tmp_path, monkeypatch):
    t, y, x, data = synth(2023, n=8, seed=61)
    raw = write_raw_v3_bands(tmp_path / "src" / "b.zarr", t, y, x, data, chunks=(8, 9, 12, 16))
    req = make_request(tmp_path / "cubes")
    place(req, [("b.zarr", raw, "x")])
    ds, interp, lines = dcz.normalise(req, lambda m: None)
    assert interp.groups == [list(BANDS)]
    assert any(ln.startswith("largest delivered chunk: cube chunks [8, 9, 12, 16]") for ln in lines)
    H, W = len(y), len(x)
    small = {"t": 2, "y": 8, "x": 8}
    msgs = []
    dcz.write_store(ds, interp.fills, req.store, {"job_id": "j"}, log=msgs.append, chunks=small,
                    groups=interp.groups)
    assert sum(m.startswith("block ") for m in msgs) == 1      # the delivered chunk decoded once
    # little memory: blocks of the store's time chunk instead, the same values
    monkeypatch.setattr(dcz, "_available_memory", lambda: int(1.5 * 8 * H * W * 2 * 10))
    msgs = []
    dcz.write_store(ds, interp.fills, req.store, {"job_id": "j"}, log=msgs.append, chunks=small,
                    groups=interp.groups)
    assert any("blocks of 2 dates instead" in m for m in msgs)
    assert sum(m.startswith("block ") for m in msgs) == 4
    out = zc.open_store(req.store)
    for b in BANDS:
        np.testing.assert_array_equal(out[b].values, data[b])
    out.close()
    # not even that: stop before reading anything; the store already there is untouched
    monkeypatch.setattr(dcz, "_available_memory", lambda: 1000)
    with pytest.raises(MemoryError, match="free memory first"):
        dcz.write_store(ds, interp.fills, req.store, {"job_id": "j"}, log=lambda m: None,
                        chunks=small, groups=interp.groups)
    assert zc.is_complete(req.store)


def test_padding_onto_a_large_grid_stays_a_few_blocks():
    """dask's pad would cut a wide margin into edge-chunk-sized blocks (~10^5
    tasks for a small window on a site grid); the margins are one block each."""
    import dask.array as da

    a = np.arange(6 * 24 * 40, dtype="int16").reshape(6, 24, 40)
    d = da.from_array(a, chunks=(2, 12, 16))
    for pads in [(0, 891, 0, 875), (3, 4, 5, 6), (0, 0, 2, 0), (7, 0, 0, 0)]:
        p = dcz._pad_tyx(d, *pads, FILL)
        want = np.pad(a, ((0, 0), pads[:2], pads[2:]), constant_values=FILL)
        assert p.dtype == np.int16 and p.shape == want.shape
        np.testing.assert_array_equal(p.compute(), want)
        assert np.prod(p.numblocks) <= 3 * 4 * 5
    np.testing.assert_array_equal(dcz._pad_tyx(a, 1, 1, 1, 1, FILL),
                                  np.pad(a, ((0, 0), (1, 1), (1, 1)), constant_values=FILL))


# ─────────────────────────────────────────────────────────────────────────────
#  the UTM extent on the reference lattice (after the format test of
#  2026-10-13), the normalisation of a delivery like the backend's, the time
#  units, and a changed request on the server's _test folder
# ─────────────────────────────────────────────────────────────────────────────

#: the 2 km lon/lat test box reprojected to EPSG:32630 (the 'extent' attribute
#: of the server's delivery of 2026-10-13): the old request
OLD_TEST_BOX_UTM = (305408.66, 4820322.70, 307468.56, 4822375.74)


def _wkt_of(epsg):
    from pyproj import CRS

    return CRS.from_epsg(epsg).to_wkt("WKT1_GDAL")


def _ref_site(monkeypatch, nc, site="villaviciosa", **extra):
    """``nc`` as the reference netCDF of ``site`` (a missing path: none on this machine)."""
    monkeypatch.setitem(dcz.SITE_CONFIG, site, dict(dcz.SITE_CONFIG[site], reference=str(nc),
                                                    **extra))


def _edges(fr):
    return tuple(float(fr[k]) for k in ("west", "south", "east", "north"))


@pytest.mark.parametrize("frame,epsg", [
    (VILLA_FRAME, 32630),                                   # upper-left corner on 60 m
    ((551720.0, 4818000.0, 552330.0, 4818920.0), 32629),    # Ferrol-like x0 551725: west not on 60 m
    ((305030.0, 4820010.0, 305640.0, 4820490.0), 32630),    # odd origin: edges on 10 m only
])
def test_main_extent_is_the_reference_frame(tmp_path, monkeypatch, frame, epsg):
    """The main jobs request exactly the reference netCDF's frame (pixel
    centres +- 5 m) in its UTM CRS: ``crs`` an EPSG integer, edges whole
    metres, and the extent's axes are the reference axes bitwise."""
    nc = frame_nc(tmp_path / "ref.nc", frame, _wkt_of(epsg))
    _ref_site(monkeypatch, nc)
    root = tmp_path / "cubes"
    fr = dcz.site_frame("villaviciosa", root)
    ext = dcz.extent_of(fr)
    west, south, east, north = frame
    assert ext == {"west": int(west), "south": int(south), "east": int(east), "north": int(north),
                   "crs": epsg}
    assert all(type(v) is int for v in ext.values())
    assert fr["kind"] == "netcdf" and fr["origin"] == "reference netCDF ref.nc"
    assert "NOTE" not in fr["source"]
    rx, ry = axes_of(frame)
    x, y = dcz.frame_axes(ext, 10)
    np.testing.assert_array_equal(x, rx)
    np.testing.assert_array_equal(y, ry)
    req = dcz.year_requests(["villaviciosa"], [2024], root)[0]
    assert req.extent == ext and req.spatial_extent == ext and req.extent_source == fr["source"]
    assert dcz.build_process(req).flat_graph()["loadcollection1"]["arguments"]["spatial_extent"] == ext
    assert dcz.describe_extent(ext, 10).endswith(f"({len(ry)} x {len(rx)} px of 10 m)")
    assert dcz.extent_line(req).endswith("<- reference netCDF ref.nc")


def test_a_reference_off_the_native_lattice_is_requested_as_it_is(tmp_path, monkeypatch, wkt):
    """Edges not on multiples of 10 m: the reference is still the target
    (requested exactly), and the source says it is not the native lattice."""
    frame = (305025.0, 4820015.0, 305635.0, 4820485.0)
    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", frame, wkt))
    fr = dcz.site_frame("villaviciosa", tmp_path / "cubes")
    assert _edges(fr) == frame and "not the native Sentinel-2 lattice" in fr["source"]


def test_frame_without_a_10_m_reference_is_the_box_snapped_out_to_60_m(tmp_path, monkeypatch, wkt):
    """A 20 m cube (as the local Escalda one) is no 10 m reference: the
    frame is the lon/lat box projected to UTM and snapped OUTWARD to
    multiples of 60 m (a line of the 10, 20 and 60 m lattices)."""
    f20 = (301020.0, 4815660.0, 310200.0, 4824840.0)
    x20, y20 = axes_of(f20, res=20.0)
    nc = write_nc(tmp_path / "ref20.nc", pd.date_range("2024-06-02", periods=1), y20, x20,
                  {b: np.full((1, len(y20), len(x20)), 1000, "int16") for b in ("B03", "B08", "SCL")},
                  wkt)
    _ref_site(monkeypatch, nc)
    fr = dcz.site_frame("villaviciosa", tmp_path / "cubes")
    assert fr["kind"] == "predicted"
    assert fr["origin"].startswith("SNAPPING: the lon/lat box outward to multiples of 60 m")
    assert "ref20.nc is on a 20 m grid" in fr["source"]
    ext = dcz.extent_of(fr)
    assert ext == {"west": 301020, "south": 4815660, "east": 310200, "north": 4824840, "crs": 32630}
    assert all(ext[k] % 60 == 0 for k in ("west", "south", "east", "north"))
    g = dcz.predict_grid(dcz.site_bbox("villaviciosa"), res=10)    # the netCDF cubes' envelope
    (x0, x1), (y0, y1) = g["x_edges"], g["y_edges"]
    assert ext["west"] <= x0 < ext["west"] + 60 and ext["east"] - 60 < x1 <= ext["east"]
    assert ext["south"] <= y0 < ext["south"] + 60 and ext["north"] - 60 < y1 <= ext["north"]
    # frames of axes on other lattices, and the outward snapping
    f = dcz.frame_from_axes(x20, y20, wkt, 20)
    assert _edges(f) == f20 and f["epsg"] == 32630 and f["res"] == 20.0
    with pytest.raises(ValueError, match="not a regular 20 m axis"):
        dcz.frame_from_axes(np.r_[x20[:3], x20[3] + 1.0], y20, wkt, 20)
    with pytest.raises(ValueError, match="no EPSG code"):
        dcz.extent_of(dict(f, epsg=None))
    box = dict(zip(("west", "south", "east", "north"), OLD_TEST_BOX_UTM))
    assert _edges(dcz.snap_frame(box, 60)) == (305400.0, 4820280.0, 307500.0, 4822380.0)
    assert _edges(dcz.snap_frame(box, 10)) == (305400.0, 4820320.0, 307470.0, 4822380.0)
    on = dict(box, west=305400.0, east=307440.0)                 # lines stay where they are
    assert _edges(dcz.snap_frame(on, 60))[0::2] == (305400.0, 307440.0)


def test_the_test_window(wkt):
    """--test: 2,040 x 2,040 m (204 x 204 px) inside the frame, corners on
    multiples of 60 m, next to the old 2 km lon/lat box; the same window
    whether the frame is the netCDF's or the snapped box."""
    villa = frame_dict(VILLA_FRAME, wkt)
    win = dcz.test_window(villa)
    assert _edges(win) == TEST_WIN and dcz.frame_shape(win) == [204, 204]
    assert all(v % 60 == 0 for v in TEST_WIN) and dcz.TEST_SIZE_M % 60 == 0
    assert "corners on multiples of 60 m" in win["source"] and win["kind"] == "test"
    ow, os_, oe, on = OLD_TEST_BOX_UTM
    assert abs(win["west"] - ow) < 60 and abs(win["north"] - on) < 60
    assert abs(win["east"] - oe) < 60 and abs(win["south"] - os_) < 60
    snapped = frame_dict((301020.0, 4815660.0, 310200.0, 4824840.0), wkt)
    assert _edges(dcz.test_window(snapped)) == TEST_WIN
    # a frame with an odd origin (edges on the 10 m lattice only): corners still on 60 m
    odd = frame_dict((301030.0, 4815680.0, 310180.0, 4824830.0), wkt)
    assert _edges(dcz.test_window(odd)) == TEST_WIN
    # a frame whose lattice is not on multiples of 10 m: corners on its own lattice
    ph = frame_dict((301025.0, 4815695.0, 310175.0, 4824845.0), wkt)
    w5 = dcz.test_window(ph)
    assert "corners on the 10 m lattice of the frame" in w5["source"]
    assert (w5["west"] - 301025.0) % 10 == 0 and (w5["north"] - 4824845.0) % 10 == 0
    assert dcz.frame_shape(w5) == [204, 204]
    # a frame the window would stick out of: moved inside, still on 60 m
    narrow = frame_dict((301020.0, 4815690.0, 306000.0, 4824840.0), wkt)
    assert _edges(dcz.test_window(narrow)) == (303960.0, 4820340.0, 306000.0, 4822380.0)
    with pytest.raises(ValueError, match="does not fit"):
        dcz.test_window(frame_dict((305000.0, 4820000.0, 307000.0, 4822000.0), wkt))
    with pytest.raises(ValueError, match="whole number"):
        dcz.test_window(villa, size=2045)


def test_angles_extent_is_the_frame_snapped_out_to_300_m(tmp_path, monkeypatch, wkt):
    """The angles jobs request the 10 m frame snapped OUTWARD to multiples of
    300 m (the rule of the angles reference grid, ``coarse_axes``); --test
    requests its window snapped outward to the 300 m cells of the site."""
    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", VILLA_FRAME, wkt))
    root = tmp_path / "cubes"
    af = dcz.angles_frame("villaviciosa", root)
    aext = {"west": 300900, "south": 4815600, "east": 310200, "north": 4824900, "crs": 32630}
    assert dcz.extent_of(af) == aext and dcz.frame_shape(af, 300) == [31, 31]
    assert af["origin"] == "reference netCDF ref.nc, snapped outward to multiples of 300 m"
    ag = dcz.angles_reference_grid("villaviciosa", root)
    x, y = dcz.frame_axes(aext, 300)
    np.testing.assert_array_equal(x, ag["x"])
    np.testing.assert_array_equal(y, ag["y"])
    tw = dcz.test_angles_window(root)
    text = {"west": 305400, "south": 4820100, "east": 307500, "north": 4822500, "crs": 32630}
    assert dcz.extent_of(tw) == text and dcz.frame_shape(tw, 300) == [8, 7]
    main, ang = dcz.year_requests(["villaviciosa"], [2024], root, ["main", "angles"])
    assert ang.extent == aext and main.extent == dcz.extent_of(dcz.site_frame("villaviciosa", root))
    tm, ta = dcz.test_requests(root)
    assert _edges(tm.extent) == TEST_WIN and ta.extent == text
    assert tm.root == root / dcz.TEST_ROOT and ta.root == root / dcz.TEST_ROOT
    # a frame with an odd origin (Ferrol's): edges on 300 m, containing it
    ferrol = (551720.0, 4809870.0, 568790.0, 4818920.0)
    fr = dcz.snap_frame(frame_dict(ferrol, _wkt_of(32629), epsg=32629), 300)
    assert _edges(fr) == (551700.0, 4809600.0, 568800.0, 4819200.0)
    cx, cy = dcz.coarse_axes(*axes_of(ferrol))
    assert (cx[0] - 150, cy[-1] - 150, cx[-1] + 150, cy[0] + 150) == _edges(fr)
    # --lonlat-extent: no UTM extent, the lon/lat box (diagnosis), under _test_lonlat
    ll = dcz.test_requests(root, lonlat=True)
    assert all(q.extent is None and q.spatial_extent == dcz.test_bbox() for q in ll)
    assert all(q.root == root / dcz.TEST_ROOT_LONLAT for q in ll)


def test_time_units_of_the_backend_decoded_explicitly():
    """'milliseconds since 01/01/1970 - 00:00:00' (the backend's Zarr) and
    its variants, into exact dates; a reference whose day and month cannot
    be told apart is refused."""
    t0 = np.datetime64("1970-01-01T00:00:00", "ns")
    for units, ns, ref in (
            ("milliseconds since 01/01/1970 - 00:00:00", 10 ** 6, t0),
            ("milliseconds since 01/01/1970", 10 ** 6, t0),
            ("ms since 1970-01-01", 10 ** 6, t0),
            ("milliseconds since 1970-01-01T00:00:00Z", 10 ** 6, t0),
            ("seconds since 1970-01-01 00:00:00 UTC", 10 ** 9, t0),
            ("days since 1990-01-01", 86400 * 10 ** 9, np.datetime64("1990-01-01", "ns")),
            ("Milliseconds Since 13/01/1970 - 00:00:00", 10 ** 6, np.datetime64("1970-01-13", "ns")),
            ("hours since 01/13/1970 - 06:30:00", 3600 * 10 ** 9,
             np.datetime64("1970-01-13T06:30", "ns"))):
        assert dcz.parse_time_units(units) == (ns, ref), units
    for bad, msg in (("milliseconds since 02/03/1970 - 00:00:00", "ambiguous"),
                     ("fortnights since 1970-01-01", "not understood"),
                     ("milliseconds after 1970-01-01", "not '<unit> since <date>'"),
                     ("ms since 1970/01/01", "not understood")):
        with pytest.raises(ValueError, match=msg):
            dcz.parse_time_units(bad)
    days = pd.DatetimeIndex(["2025-06-02", "2025-06-07T10:56:31.024", "2025-06-12T11:06:29"])
    ms = ((days - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")).values.astype(">i8")
    units = "milliseconds since 01/01/1970 - 00:00:00"
    for vals in (ms, ms.astype("<i8"), ms.astype(">f8")):
        got = dcz.decode_time_values(vals, units)
        assert got.dtype == np.dtype("datetime64[ns]")
        np.testing.assert_array_equal(got, days.values)
    var = xr.Variable(("t",), ms, attrs={"units": units, "calendar": "proleptic_gregorian"})
    notes = []
    np.testing.assert_array_equal(dcz._decode_time(var, notes), days.values)
    assert notes == [f"time units {units!r} decoded explicitly (reference date read as "
                     f"1970-01-01T00:00:00, >i8 values)"]
    with pytest.raises(dcz.FormatError, match="ambiguous"):
        dcz._decode_time(xr.Variable(("t",), ms, attrs={"units": "ms since 02/03/1970"}), [])
    # 'since' in any case reaches the case-insensitive parser
    shout = xr.Variable(("t",), ms, attrs={"units": "Milliseconds SINCE 01/01/1970 - 00:00:00"})
    np.testing.assert_array_equal(dcz._decode_time(shout, []), days.values)
    with pytest.raises(dcz.FormatError, match="numeric without CF units"):
        dcz._decode_time(xr.Variable(("t",), ms, attrs={"units": "milliseconds sincere"}), [])
    # float32 times: refused where a float32 step is coarser than a second (ms since 1970:
    # 131072 ms), accepted where it resolves the second (hours since a near date)
    with pytest.raises(dcz.FormatError, match=r"one float32 step is 131\.072 s"):
        dcz._decode_time(xr.Variable(("t",), ms.astype(">f4"), attrs={"units": units}), [])
    hrs = np.array([24.0, 144.5, 264.25], dtype=">f4")
    got = dcz._decode_time(xr.Variable(("t",), hrs, attrs={"units": "hours since 2025-06-01"}), [])
    np.testing.assert_array_equal(got, (pd.Timestamp("2025-06-01")
                                        + pd.to_timedelta([24.0, 144.5, 264.25], "h")).values)


def test_a_delivery_like_the_backends_lands_bitwise_on_the_reference_grid(tmp_path, wkt,
                                                                          monkeypatch):
    """Production: the request is the reference frame (here a 300 x 300 px
    netCDF cube); the delivery is shaped as on 2026-10-13 (zipped Zarr v2,
    >i2 with fill -32768, Blosc lz4 chunks [1, 128, 128], float32 x/y (exact:
    centres in whole metres), time in ms since 01/01/1970, padded to whole
    chunks: 384 x 384 px): cropped onto the frame, the store's axes are the
    reference's float64 values and its values the delivered ones, bitwise,
    in native int16; the cropped margin held only the fill, so no warning.

    Since 2026-10-19 production REGISTERS the delivery (``registration_plan``
    -> ``register_delivery``) against the reference netCDF, which therefore
    holds the delivery's three dates (with clear SCL) instead of one
    constant 2024 date: the data sit at row 0, col 0 of the padded arrays
    here, the registration finds (0, 0), and the store is bitwise the one
    the coordinate crop of before made ("cropped rows 0:300 cols 0:300"); the
    note no longer speaks of padding (a registered crop never pads)."""
    t = pd.date_range("2025-06-02", periods=3, freq="5D")
    rx, ry = axes_of(TEST_FRAME)
    nc = write_nc(tmp_path / "ref.nc", t, ry, rx,
                  {b: lattice_values(b, 3, rx, ry, scl_clear=True) for b in ("B03", "B08", "SCL")},
                  wkt)
    _ref_site(monkeypatch, nc)
    root = tmp_path / "cubes"
    req = dcz.year_requests(["villaviciosa"], [2025], root)[0]
    assert req.extent == {"west": 305000, "south": 4820000, "east": 308000, "north": 4823000,
                          "crs": 32630}
    assert req.extent_origin == "reference netCDF ref.nc"
    data = {}
    for b in BANDS:
        a = np.full((3, 384, 384), FILL, "int16")
        a[:, :300, :300] = lattice_values(b, 3, rx, ry, scl_clear=True)
        data[b] = a
    xf, yf = f32_like_the_backend(*server_axes(TEST_FRAME[0], TEST_FRAME[3], 384, 384))
    assert xf.dtype == yf.dtype == np.float32
    raw = write_server_zarr(tmp_path / "src" / "out.zarr", t, yf, xf, data, wkt, extent=TEST_FRAME)
    place(req, [("out.zarr.zip", zip_store(raw, tmp_path / "src" / "out.zarr"), "application/zip")])
    ref = dcz.reference_grid("villaviciosa", root=root)
    plan = dcz.registration_plan(req, None, ref)
    assert plan.reference == str(nc) and plan.strict and plan.expert is None
    interp, rep, lines = normalise_and_write(req, ref_grid=ref, registration=plan)
    assert interp.warnings == []
    notes = "\n".join(interp.notes)
    for s_ in ("same 10 m lattice as ref.nc but another extent: 384 x 384 px",
               "at the registered offset (netcdf): cropped rows 0:300 cols 0:300",
               "cropped-away margin", ": only the fill (-32768) / NaN",
               "time units 'milliseconds since 01/01/1970 - 00:00:00' decoded explicitly",
               "x (delivered as >f4) 0.000 m (0.000 px) from the 10 m lattice of ref.nc: exactly "
               "its float32 values; the lattice's exact float64 values are used",
               "y (delivered as >f4) 0.000 m (0.000 px) from the 10 m lattice of ref.nc: exactly "
               "its float32 values",
               "the delivered 'extent' attribute [305000.0, 4820000.0, 308000.0, 4823000.0] is the "
               "requested extent"):
        assert s_ in notes, s_
    ds = zc.open_store(req.store)
    assert ds["x"].dtype == ds["y"].dtype == np.float64
    np.testing.assert_array_equal(ds["x"].values, ref["x"])
    np.testing.assert_array_equal(ds["y"].values, ref["y"])
    np.testing.assert_array_equal(ds["t"].values, t.values)
    for b in BANDS:
        assert ds[b].dtype == np.dtype("int16") and ds[b].dtype.byteorder in "=|<"
        assert zc.fill_of(ds[b]) == FILL
        np.testing.assert_array_equal(ds[b].values, lattice_values(b, 3, rx, ry, scl_clear=True))
    assert json.loads(ds.attrs["delivered_offset_px"]) == [0, 0]
    assert json.loads(ds.attrs["delivered_shape_px"]) == [384, 384]
    assert ds.attrs["registration_method"] == "netcdf"
    # clearest first: date 0 has two no-data 20 m rows, so it comes last
    assert json.loads(ds.attrs["registration_dates"]) == [str(d.date()) for d in t[[1, 2, 0]]]
    assert all(r == 1.0 for band in json.loads(ds.attrs["registration_r"]).values()
               for r in band.values())
    assert json.loads(ds.attrs["spatial_extent"]) == req.extent
    assert json.loads(ds.attrs["spatial_extent_lonlat"]) == dcz.site_bbox("villaviciosa")
    assert ds.attrs["spatial_extent_source"] == "the frame of the reference netCDF ref.nc (300 x 300 px)"
    ds.close()


def test_a_real_offset_is_still_refused(tmp_path, wkt):
    """The delivery of the lon/lat request of 2026-10-13, anchored on the
    reprojected box's upper-left corner with float32 x/y, is (0.134, 0.450)
    px off the lattice: refused unless --allow-grid-offset. ON the lattice
    means every float32 value is the float32 rounding of a lattice value
    (the lattice in whole metres is exact in float32); a uniform shift by a
    float32 step or more (0.0625 m in x, 0.5 m in y) is a real offset and
    refused, as is any offset of float64 axes; values off the lattice on
    some rows only leave the regular axis (their median) on it."""
    t = pd.date_range("2025-06-02", periods=1)
    rx, ry = axes_of(TEST_FRAME)
    ref = {"path": "ref.nc", "kind": "netcdf", "x": rx, "y": ry, "crs_wkt": wkt}

    def deliver(name, x, y):
        data = {b: np.full((1, len(y), len(x)), 1000, "int16") for b in BANDS}
        raw = write_server_zarr(tmp_path / "src" / name / "out.zarr", t, y, x, data, wkt,
                                extent=OLD_TEST_BOX_UTM)
        req = make_request(tmp_path / name, year=2025)
        place(req, [("out.zarr.zip", zip_store(raw, tmp_path / "src" / name / "out.zarr"),
                     "application/zip")])
        return req

    ox, oy = server_axes(OLD_TEST_BOX_UTM[0], OLD_TEST_BOX_UTM[3], 256, 256)
    req = deliver("old", ox.astype("float32"), oy.astype("float32"))
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.134, 0\.450\) px .* this is a real "
                                              r"offset; nothing written"):
        dcz.normalise(req, lambda m: None, ref_grid=ref)
    assert not req.store.exists()
    _, interp, _ = dcz.normalise(req, lambda m: None, ref_grid=ref, allow_offset=True)
    assert any("--allow-grid-offset" in w for w in interp.warnings)
    x0, y0 = server_axes(TEST_FRAME[0], TEST_FRAME[3], 300, 300)
    # exact float32: on the lattice, the lattice's float64 values
    ds, interp, _ = dcz.normalise(deliver("f32_exact", x0.astype("float32"), y0.astype("float32")),
                                  lambda m: None, ref_grid=ref)
    np.testing.assert_array_equal(ds["x"].values, rx)
    np.testing.assert_array_equal(ds["y"].values, ry)
    assert any(n.startswith("y (delivered as >f4) 0.000 m (0.000 px) from the 10 m lattice of "
                            "ref.nc: exactly its float32 values") for n in interp.notes)
    # uniform shifts of one or two float32 steps are no rounding: refused (before, the
    # 0.5 m tolerance relabelled them as rounding)
    xs = (x0 + 0.0625).astype("float32")                   # 2 float32 steps at x ~ 3e5 m
    assert np.all(xs.astype("float64") - x0 == 0.0625)
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.006, 0\.000\) px .* real offset"):
        dcz.normalise(deliver("f32_x2ulp", xs, y0.astype("float32")), lambda m: None, ref_grid=ref)
    ys = (y0 + 0.5).astype("float32")                      # 1 float32 step at y ~ 4.8e6 m
    assert np.all(ys.astype("float64") - y0 == 0.5)
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.000, 0\.050\) px .* real offset"):
        dcz.normalise(deliver("f32_y1ulp", x0.astype("float32"), ys), lambda m: None, ref_grid=ref)
    # 0.04 px in float32 (accepted as rounding before): a real offset
    x4 = (x0 + 0.4).astype("float32")
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.041, 0\.000\) px"):
        dcz.normalise(deliver("f32_004", x4, y0.astype("float32")), lambda m: None, ref_grid=ref)
    # one float32 step off on every 7th row only: the regular axis (median) is on the
    # lattice, so the store is; the note counts the odd values
    yj = y0.astype("float32")
    yj[::7] = np.nextafter(yj[::7], np.float32(np.inf))
    ds, interp, _ = dcz.normalise(deliver("f32_rows", x0.astype("float32"), yj), lambda m: None,
                                  ref_grid=ref)
    np.testing.assert_array_equal(ds["y"].values, ry)
    assert any("y (delivered as >f4): 43 of 300 values are not the float32 rounding" in n
               and "the lattice check decides on the regular axis" in n for n in interp.notes)
    assert interp.warnings == []
    # a lattice whose centres are not whole metres: float32 rounding within half a step
    rf = rx + 0.3
    ds, interp, _ = dcz.normalise(deliver("f32_frac", rf.astype("float32"), y0.astype("float32")),
                                  lambda m: None, ref_grid=dict(ref, x=rf))
    np.testing.assert_allclose(ds["x"].values, rf, rtol=0, atol=1e-6)
    assert any(n.startswith("x (delivered as >f4) 0.0") and "float32 rounding (at most half a "
               "float32 step, 0.016 m here)" in n for n in interp.notes)
    # 0.07 px in float32, and 0.04 px in float64: real offsets
    x7 = (x0 + 0.7).astype("float32")
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.069, 0\.000\) px"):
        dcz.normalise(deliver("f32_007", x7, y0.astype("float32")), lambda m: None, ref_grid=ref)
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.040, 0\.000\) px"):
        dcz.normalise(deliver("f64_004", x0 + 0.4, y0), lambda m: None, ref_grid=ref)


def test_the_first_utm_test_makes_a_new_job_and_keeps_the_old_part(tmp_path, wkt, monkeypatch,
                                                                   fake_build):
    """The server after the format test of 2026-10-13: <root>/_test/
    villaviciosa/2025.part holds the job and the download of the lon/lat
    request, refused by the grid guard. The --test of the UTM extent has
    another graph, so another job title: the old part is moved aside (kept;
    --test never deletes it), a NEW job is made and its delivery lands on the
    window. --lonlat-extent (under <root>/_test_lonlat) finds the old job
    again by its title: no new job, no credits."""
    t = pd.date_range("2025-06-02", periods=1)
    rx, ry = axes_of(TEST_FRAME)
    nc = write_nc(tmp_path / "ref.nc", t, ry, rx, {b: lattice_values(b, 1, rx, ry)
                                                  for b in ("B03", "B08", "SCL")}, wkt)
    _ref_site(monkeypatch, nc, reference_period=("2016-01-01", "2025-12-31"),
              pipeline_overpass=str(tmp_path / "none.json"))
    root = tmp_path / "cubes"
    # the old delivery: 256 x 256 px anchored on the reprojected lon/lat box, float32 x/y
    ox, oy = server_axes(OLD_TEST_BOX_UTM[0], OLD_TEST_BOX_UTM[3], 256, 256)
    old_raw = write_server_zarr(tmp_path / "src" / "old" / "out.zarr", t, oy.astype("float32"),
                                ox.astype("float32"),
                                {b: np.full((1, 256, 256), 1000, "int16") for b in BANDS}, wkt,
                                extent=OLD_TEST_BOX_UTM)
    # the new delivery: the window, padded to 256 x 256 px
    wx, wy = axes_of(TEST_WIN)
    data = {}
    for b in BANDS:
        a = np.full((1, 256, 256), FILL, "int16")
        a[:, :204, :204] = lattice_values(b, 1, wx, wy)
        data[b] = a
    nx_, ny_ = server_axes(TEST_WIN[0], TEST_WIN[3], 256, 256)
    new_raw = write_server_zarr(tmp_path / "src" / "new" / "out.zarr", t, ny_.astype("float32"),
                                nx_.astype("float32"), data, wkt, extent=TEST_WIN)
    deliveries = {"j-1": zip_store(old_raw, tmp_path / "src" / "old" / "out.zarr"),
                  "j-2": zip_store(new_raw, tmp_path / "src" / "new" / "out.zarr")}

    def fake_download(job, req, log, workers=4):
        place(req, [("out.zarr.zip", deliveries[job.job_id], "application/zip")])
        return {"assets": {"out.zarr.zip": {"href": "https://h/out.zarr.zip",
                                            "type": "application/zip"}}}

    monkeypatch.setattr(dcz, "download_results", fake_download)
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    ctx.network = False
    # 2026-10-13: the lon/lat request, under <root>/_test as the code of that day had it
    old = dcz.test_request(root, lonlat=True)
    old.root = root / dcz.TEST_ROOT
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.134, 0\.450\) px"):
        dcz.run_request(ctx, old, lambda m: None)
    part = Path(old.part)
    old_title = dcz.recorded_title(part)
    assert old_title and conn.created == ["j-1"] and (part / "assets" / "out.zarr.zip").is_file()
    kept = sorted(p.relative_to(part).as_posix() for p in part.rglob("*") if p.is_file())
    # today: the --test of the UTM extent, same part folder, another title
    new = dcz.test_request(root)
    assert Path(new.part) == part
    flat = dcz.build_process(new).flat_graph()
    assert dcz.job_title(new, dcz.graph_hash(flat)) != old_title
    msgs = []
    res = dcz.run_request(ctx, new, msgs.append)
    text = "\n".join(msgs)
    assert res["status"] == "done", text
    assert conn.created == ["j-1", "j-2"] and conn.jobs["j-2"]["title"] != old_title
    assert dcz.recorded_title(part) == conn.jobs["j-2"]["title"]
    aside = dcz.superseded_parts(new)
    assert len(aside) == 1 and aside[0].name.startswith("2025.part.superseded-")
    assert dcz.recorded_title(aside[0]) == old_title
    assert sorted(p.relative_to(aside[0]).as_posix() for p in aside[0].rglob("*")
                  if p.is_file()) == kept
    assert "the request changed since 2025.part was made" in text
    assert "kept: --test never deletes it" in text
    ds = zc.open_store(new.store)
    np.testing.assert_array_equal(ds["x"].values, wx)
    np.testing.assert_array_equal(ds["y"].values, wy)
    np.testing.assert_array_equal(ds["B03"].values, lattice_values("B03", 1, wx, wy))
    ds.close()
    # --lonlat-extent: the old request again, under _test_lonlat: its part is copied from
    # the folder set aside (same title), so no new job and no new download, even if the
    # results of 2026-10-13 have expired on the backend
    ll = dcz.test_request(root, lonlat=True)
    assert ll.root == root / dcz.TEST_ROOT_LONLAT
    downloads = []
    monkeypatch.setattr(dcz, "download_results", lambda *a, **k: downloads.append(a) or {})
    lmsgs = []
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.134, 0\.450\) px"):
        dcz.run_request(ctx, ll, lmsgs.append)
    assert conn.created == ["j-1", "j-2"] and downloads == []
    assert any(f"copied from {aside[0]}" in m and "no new job" in m for m in lmsgs)
    assert dcz.recorded_title(ll.part) == old_title
    assert dcz.superseded_parts(new) == aside and aside[0].is_dir()
    assert sorted(p.relative_to(aside[0]).as_posix() for p in aside[0].rglob("*")
                  if p.is_file()) == kept                        # the source is untouched
    # a second run finds its own part; nothing is copied again
    lmsgs.clear()
    with pytest.raises(dcz.FormatError, match=r"offset by \(0\.134, 0\.450\) px"):
        dcz.run_request(ctx, ll, lmsgs.append)
    assert not any("copied from" in m for m in lmsgs) and downloads == []


def test_dry_run_prints_each_sites_utm_extent_and_its_origin(tmp_path, monkeypatch, capsys, wkt):
    import re

    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", TEST_FRAME, wkt))
    _ref_site(monkeypatch, tmp_path / "missing.nc", site="ferrol")
    monkeypatch.chdir(tmp_path)
    cubes = str(tmp_path / "cubes")
    assert dcz.main(["villaviciosa", "ferrol", "--years", "2024", "--dry-run", "--root", cubes]) == 0
    out = capsys.readouterr().out
    assert '"crs": 32630' in out and '"crs": 32629' in out
    tail = out.split("SPATIAL EXTENTS requested")[1]
    rows = {tuple(ln.split()[:2]): ln for ln in tail.splitlines() if ln.startswith("  ")}
    assert ("EPSG:32630 west 305000 south 4820000 east 308000 north 4823000 (300 x 300 px of 10 m)"
            "  <- reference netCDF ref.nc") in rows[("villaviciosa", "main")]
    assert ("EPSG:32630 west 304800 south 4819800 east 308100 north 4823100 (11 x 11 cells of 300 m)"
            "  <- reference netCDF ref.nc, snapped outward to multiples of 300 m") in \
        rows[("villaviciosa", "angles")]
    fm = rows[("ferrol", "main")]
    assert "EPSG:32629" in fm and ("<- SNAPPING: the lon/lat box outward to multiples of 60 m "
                                   "(missing.nc is not on this machine, no complete store)") in fm
    edges = [int(v) for v in re.search(r"west (\d+) south (\d+) east (\d+) north (\d+)", fm).groups()]
    assert all(v % 60 == 0 for v in edges)
    assert "spatial extent: EPSG:32630 west 305000" in out and "origin: reference netCDF ref.nc" in out
    # the snapped rows are marked: a production run refuses them before any job
    assert fm.endswith("[REFUSED: no reference netCDF; --allow-snapped-frame]")
    assert rows[("ferrol", "angles")].endswith("[REFUSED: no reference netCDF; --allow-snapped-frame]")
    assert "REFUSED" not in rows[("villaviciosa", "main")] + rows[("villaviciosa", "angles")]
    assert (f"REFUSED before any job (production lands only on a reference frame): ferrol main, "
            f"ferrol angles. Missing here: {tmp_path / 'missing.nc'}") in out
    assert dcz.main(["ferrol", "--years", "2024", "--dry-run", "--root", cubes,
                     "--allow-snapped-frame"]) == 0
    assert "REFUSED" not in capsys.readouterr().out
    # --lonlat-extent: the old lon/lat box, said to be off the lattice
    assert dcz.main(["villaviciosa", "--years", "2024", "--dry-run", "--lonlat-extent",
                     "--root", cubes]) == 0
    out = capsys.readouterr().out
    assert "spatial extent: the lon/lat box" in out and "NOT on the lattice" in out
    assert '"crs"' not in out.split("SPATIAL EXTENTS requested")[0]


# ─────────────────────────────────────────────────────────────────────────────
#  review of the UTM extent: production refuses a snapped frame and padding,
#  the odd --test window, the angles' phase, the margin of a sample of dates
# ─────────────────────────────────────────────────────────────────────────────

def test_production_refuses_a_snapped_frame_before_any_job(tmp_path, monkeypatch, capsys,
                                                           fake_build):
    """A site whose 10 m netCDF is missing on the machine would be requested
    on the lon/lat box snapped to 60 m, which is not the netCDF frame (and
    would become the reference of the other years): a production run stops
    before any job, network or folder; --allow-snapped-frame lets it go;
    --test and --dry-run never refuse."""
    import socket

    def no_net(*a, **k):
        raise AssertionError("network used by a refused run")

    _ref_site(monkeypatch, tmp_path / "missing.nc", site="ferrol")
    monkeypatch.setattr(socket, "create_connection", no_net)
    monkeypatch.chdir(tmp_path)
    cubes = tmp_path / "cubes"
    assert dcz.main(["ferrol", "--years", "2024", "--root", str(cubes)]) == 2
    out = capsys.readouterr().out
    assert "REFUSED, nothing started (no job, no download):" in out
    assert "ferrol 2024: the extent EPSG:32629 west" in out and "ferrol 2024 angles:" in out
    assert "--allow-snapped-frame" in out and str(tmp_path / "missing.nc") in out
    assert not cubes.exists()
    # run_request refuses too (any caller), before the part folder or a job
    req = dcz.year_requests(["ferrol"], [2024], cubes)[0]
    assert dcz.snapped(req) and req.extent_origin.startswith("SNAPPING")
    conn = FakeConn()
    ctx, _ = _ctx(conn)
    with pytest.raises(dcz.FrameRefused, match="not the frame of the site's reference netCDF"):
        dcz.run_request(ctx, req, lambda m: None)
    assert conn.created == [] and not Path(req.part).exists()
    # --allow-snapped-frame: the refusal is lifted (the job is made)
    ctx.allow_snapped = True

    def stop(*a, **k):
        raise RuntimeError("stop here")

    monkeypatch.setattr(dcz, "download_results", stop)
    with pytest.raises(RuntimeError, match="stop here"):
        dcz.run_request(ctx, req, lambda m: None)
    assert conn.created == ["j-1"]
    assert dcz.parse_args(["ferrol", "--allow-snapped-frame"]).allow_snapped_frame
    # a reference netCDF frame, the --test window or a lon/lat request: never refused
    assert dcz.refuse_snapped(dcz.test_request(cubes)) is None
    lonlat = dcz.year_requests(["ferrol"], [2024], cubes, lonlat=True)[0]
    assert not dcz.snapped(lonlat) and dcz.refuse_snapped(lonlat) is None


def test_the_odd_test_window(tmp_path, monkeypatch, wkt):
    """--test --odd-window: 2,050 x 2,050 m (205 px, odd), west and south
    edges 10 m past a multiple of 20 m, north and east on 20 m, inside the
    Villaviciosa frame; main product only, under <root>/_test_odd, its own
    job title."""
    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", VILLA_FRAME, wkt))
    root = tmp_path / "cubes"
    win = dcz.test_odd_window(dcz.site_frame("villaviciosa", root))
    assert _edges(win) == (305410.0, 4820330.0, 307460.0, 4822380.0)
    assert dcz.frame_shape(win) == [205, 205]
    assert win["west"] % 20 == 10 and win["south"] % 20 == 10
    assert win["east"] % 20 == 0 and win["north"] % 20 == 0
    assert "west/south edge(s) 10 m past a multiple of 20 m" in win["source"]
    reqs = dcz.test_requests(root, odd=True)
    assert [r.product for r in reqs] == ["main"] and reqs[0].root == root / dcz.TEST_ROOT_ODD
    assert reqs[0].extent == {"west": 305410, "south": 4820330, "east": 307460, "north": 4822380,
                              "crs": 32630}
    std = dcz.test_request(root)
    t_std = dcz.job_title(std, dcz.graph_hash(dcz.build_process(std).flat_graph()))
    t_odd = dcz.job_title(reqs[0], dcz.graph_hash(dcz.build_process(reqs[0]).flat_graph()))
    assert t_odd != t_std and t_odd.startswith("zarr_cube_test_villaviciosa_2025-06-01_2025-06-16_")
    with pytest.raises(ValueError, match="main product of the UTM extent only"):
        dcz.test_request(root, "angles", odd=True)
    with pytest.raises(ValueError, match="not inside the frame"):
        dcz.test_odd_window(frame_dict((305400.0, 4820340.0, 307440.0, 4822380.0), wkt))
    for argv in (["--odd-window"], ["--test", "--odd-window", "--lonlat-extent"]):
        with pytest.raises(SystemExit):
            dcz.parse_args(argv)
    assert dcz.parse_args(["--test", "--odd-window"]).odd_window


def test_angles_keep_the_phase_of_a_frame_off_multiples_of_10_m(tmp_path, monkeypatch, wkt):
    """A reference frame whose edges are at ...5 m (not the native lattice,
    requested as it is): its 300 m angle cells keep that offset, so their
    edges are lines of its lattice (before, they sat on multiples of 300 m,
    half a pixel off it)."""
    frame = (305025.0, 4820015.0, 305635.0, 4820485.0)
    _ref_site(monkeypatch, frame_nc(tmp_path / "ref.nc", frame, wkt))
    root = tmp_path / "cubes"
    af = dcz.angles_frame("villaviciosa", root)
    assert _edges(af) == (304805.0, 4819805.0, 305705.0, 4820705.0)
    assert "plus (5, 5) m, the 10 m frame's own offset" in af["origin"]
    assert all((v - 5.0) % 10 == 0 for v in _edges(af))
    assert af["west"] <= frame[0] and af["east"] >= frame[2]
    assert af["south"] <= frame[1] and af["north"] >= frame[3]
    ag = dcz.angles_reference_grid("villaviciosa", root)
    x, y = dcz.frame_axes(dcz.extent_of(af), 300)
    np.testing.assert_array_equal(x, ag["x"])
    np.testing.assert_array_equal(y, ag["y"])
    assert dcz.lattice_phase(305025.0) == 5.0 and dcz.lattice_phase(301020.0) == 0.0
    assert dcz.lattice_phase(301019.9999999) == 0.0
    # the sites' frames (edges on multiples of 10 m): multiples of 300 m, as before
    cx, cy = dcz.coarse_axes(*axes_of(VILLA_FRAME))
    assert (cx[0] - 150, cy[0] + 150) == (300900.0, 4824900.0)


def test_the_margin_line_is_a_note_and_says_when_dates_are_sampled(tmp_path, wkt):
    """What the cropped-away margin held is a note either way (it lies
    outside the frame); on a sample of the dates the line says 'sampled'."""
    m = {"shape": [384, 384], "rows": [0, 300], "cols": [0, 300], "dates": 12, "of": 173,
         "other": 0, "zero": 0, "bands": {"B03": {"cells": 12 * 1000, "fill": 12 * 1000, "nan": 0,
                                                  "zero": 0, "other": 0, "min": None,
                                                  "max": None}}}
    line, clean = dcz.format_margin(m, {"B03": FILL})
    assert clean and "12 of 173 dates sampled)" in line
    assert line.endswith(": only the fill (-32768) / NaN on the sampled dates")
    whole = dict(m, dates=8, of=8)
    line, _ = dcz.format_margin(whole, {"B03": FILL})
    assert "sampled" not in line and line.endswith(": only the fill (-32768) / NaN")
    data = dict(m, other=5, bands={"B03": dict(m["bands"]["B03"], fill=11995, other=5, min=10.0,
                                               max=900.0)})
    line, clean = dcz.format_margin(data, {"B03": FILL})
    assert not clean and "holds data on the sampled dates" in line
    assert "pixels OUTSIDE the requested frame" in line and "nothing inside the frame is lost" in line
    # a production delivery larger than the frame, its margin holding data: cropped, notes only
    t, y, x, d = synth(2023, H=24, W=40, seed=61)
    ref = {"path": "ref.nc", "kind": "netcdf", "x": x[2:36], "y": y[3:20], "crs_wkt": wkt}
    raw = write_raw_v2(tmp_path / "src" / "m.zarr", t, y, x, d, wkt)
    req = make_request(tmp_path / "cubes")
    place(req, [("m.zarr", raw, "x")])
    ds, interp, lines = dcz.normalise(req, lambda m_: None, ref_grid=ref)
    assert interp.warnings == [] and "=> normalisable: YES" in lines
    assert any("cropped rows 3:20 cols 2:36" in n for n in interp.notes)
    assert any(n.startswith("cropped-away margin") and "holds data" in n for n in interp.notes)
    np.testing.assert_array_equal(ds["B03"].values, d["B03"][:, 3:20, 2:36])


# ─────────────────────────────────────────────────────────────────────────────
#  registration (2026-10-19): the backend's padded arrays carry x/y labelled
#  from the frame's corner while the frame's data start at row dy, col dx of
#  them; the store must land on the true pixels, or not be written
# ─────────────────────────────────────────────────────────────────────────────

#: a 64 x 64 px frame on the Villaviciosa lattice: the reference netCDF of these tests
FRAME64 = (305000.0, 4822360.0, 305640.0, 4823000.0)
D24 = pd.date_range("2024-06-02", periods=4, freq="5D")
D23 = pd.date_range("2023-06-03", periods=4, freq="5D")
D22 = pd.date_range("2022-06-04", periods=4, freq="5D")
D21 = pd.date_range("2021-06-05", periods=4, freq="5D")


#: the bands that reach 10 m by nearest neighbour from 20 m
TWENTY_M = ("B8A", "B11", "B12", "SCL", "CLD")


class World:
    """A synthetic piece of the Villaviciosa lattice: per band a STATIC field
    (the geography, the same on every date) plus a smaller part that changes
    with the date, all hashes of the UTM position (``_mix``: no periodic
    structure), so a reference frame, a padded delivery and another year cut
    from it agree wherever they overlap on one date, and correlate across
    dates. A 10 m band = a 20 m-cell field + 10 m detail + the date's part;
    B08 shares its 20 m field with B8A (the NIR: B8A against B08 r ~ 0.97),
    as the 20 m check of the registration expects of real data. The 20 m
    bands are constant on the 20 m cells of the lattice; SCL is clear
    (4/5/6) except on the ``cloudy`` dates (9 there); ``period`` > 0 repeats
    every band every ``period`` (even) pixels in x (a correlation without a
    unique peak)."""

    def __init__(self, seed=0, cloudy=(), period=0):
        self.seed, self.period = int(seed), int(period)
        self.cloudy = {str(pd.Timestamp(d).date()) for d in cloudy}

    def values(self, b, date, x, y):
        X10 = np.floor(np.asarray(x, dtype="float64") / 10.0).astype("int64")
        Y10 = np.floor(np.asarray(y, dtype="float64") / 10.0).astype("int64")
        if self.period:
            X10 = X10 % self.period
        X20, Y20 = X10 // 2, Y10 // 2                   # the 20 m cells (x, y multiples of 20 m)
        c, f = (X20[None, :], Y20[:, None]), (X10[None, :], Y10[:, None])
        bi, s = BANDS.index(b) + 1, self.seed
        day = pd.Timestamp(date).toordinal()
        u = np.uint64
        if b == "SCL":
            v = (np.full((len(Y20), len(X20)), 9) if str(pd.Timestamp(date).date()) in self.cloudy
                 else 4 + _mix(*c, s, bi) % u(3))
        elif b == "CLD":
            v = _mix(*c, s, bi) % u(20) + u(70 if str(pd.Timestamp(date).date()) in self.cloudy else 0)
        elif zc.NATIVE_RES_M[b] == 10:
            v = u(500) + _mix(*c, s, bi) % u(3000) + _mix(*f, s, bi) % u(600) + _mix(*f, s, bi, day) % u(300)
        elif b == "B8A":                                # the 20 m field of B08 (band 4)
            nir = BANDS.index("B08") + 1
            v = u(500) + _mix(*c, s, nir) % u(3000) + _mix(*c, s, bi) % u(300) + _mix(*c, s, bi, day) % u(300)
        else:
            v = u(500) + _mix(*c, s, bi) % u(4000) + _mix(*c, s, bi, day) % u(300)
        return np.asarray(v).astype("int16")

    def cube(self, b, dates, x, y):
        return np.stack([self.values(b, d, x, y) for d in dates])


def padded_delivery(world, dates, offset, shape, frame=FRAME64, fill_margin=False, offset20=None,
                    bands20=TWENTY_M):
    """A delivery of ``frame`` like the backend's of 2026-10: per band (t,
    rows, cols) arrays of ``shape`` whose frame starts at ``offset`` (dy, dx)
    (or a list, one per date), holding the true pixels of ``world`` there
    (data outside the frame too, or the fill with ``fill_margin``); and the
    x/y the backend labels them with: from the FRAME's corner, float32.
    ``offset20``: the ``bands20`` (default every 20 m band) at that offset
    instead (a 20 m layer displaced against the 10 m one)."""
    west, _, _, north = frame
    Hr, Wr = shape
    fx, fy = axes_of(frame)
    H, W = len(fy), len(fx)
    offs = list(offset) if isinstance(offset, list) else [offset] * len(dates)
    data = {b: np.empty((len(dates), Hr, Wr), "int16") for b in BANDS}
    for k, (d, off) in enumerate(zip(dates, offs)):
        for b in BANDS:
            dy, dx = offset20 if offset20 is not None and b in bands20 else off
            tx = west + 5.0 - 10.0 * dx + 10.0 * np.arange(Wr)
            ty = north - 5.0 + 10.0 * dy - 10.0 * np.arange(Hr)
            a = world.values(b, d, tx, ty)
            if fill_margin:
                m = np.full(a.shape, FILL, "int16")
                m[dy:dy + H, dx:dx + W] = a[dy:dy + H, dx:dx + W]
                a = m
            data[b][k] = a
    xf, yf = f32_like_the_backend(*server_axes(west, north, Wr, Hr))
    return data, xf, yf


def frame_values(world, dates, frame=FRAME64):
    """Every band of ``world`` on ``frame``: what a registered store holds."""
    fx, fy = axes_of(frame)
    return {b: world.cube(b, dates, fx, fy) for b in BANDS}


class Site:
    """villaviciosa with a reference netCDF on FRAME64 (B03 B08 SCL of
    ``world`` on ``ref_dates``, requested for ``period``; ``tweak(data)``
    may alter it), and production years run through ``run_request`` with a
    fake job and download (``deliver`` sets what a year's job delivers)."""

    def __init__(self, tmp_path, monkeypatch, wkt, world, ref_dates,
                 period=("2024-01-01", "2025-01-01"), tweak=None):
        self.tmp, self.wkt, self.root = tmp_path, wkt, tmp_path / "cubes"
        fx, fy = axes_of(FRAME64)
        ref = {b: world.cube(b, ref_dates, fx, fy) for b in ("B03", "B08", "SCL")}
        if tweak:
            tweak(ref)
        self.nc = write_nc(tmp_path / "ref.nc", ref_dates, fy, fx, ref, wkt)
        monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
            dcz.SITE_CONFIG["villaviciosa"], reference=str(self.nc), reference_period=period,
            pipeline_overpass=str(tmp_path / "none.json")))
        self.deliveries = {}
        self.conn = FakeConn()
        self.ctx, _ = _ctx(self.conn)
        self.ctx.network, self.ctx.keep_raw = False, True

        def fake_download(job, req, log, workers=4):
            place(req, [("out.zarr.zip", self.deliveries[req.year], "application/zip")])
            meta = {"assets": {"out.zarr.zip": {"href": "https://h/out.zarr.zip",
                                                "type": "application/zip"}}}
            zc.write_json(meta, Path(req.part) / "job-results.json")
            return meta

        monkeypatch.setattr(dcz, "download_results", fake_download)

    def deliver(self, year, world, dates, offset, shape, fill_margin=False, tweak=None, **kw):
        data, xf, yf = padded_delivery(world, dates, offset, shape, fill_margin=fill_margin, **kw)
        if tweak:
            tweak(data)
        src = self.tmp / "src" / str(year)
        raw = write_server_zarr(src / "out.zarr", dates, yf, xf, data, self.wkt, extent=FRAME64)
        self.deliveries[year] = zip_store(raw, src / "out.zarr")
        return data

    def request(self, year, root=None):
        return dcz.year_requests(["villaviciosa"], [year], root or self.root, ["main"])[0]

    def run(self, year, msgs=None, root=None):
        req = self.request(year, root)
        msgs = [] if msgs is None else msgs
        res = dcz.run_request(self.ctx, req, msgs.append)
        return req, res, "\n".join(msgs)


def _attrs(store):
    return zc.store_attrs(store)


@pytest.mark.parametrize("offset,shape,fill_margin", [
    ((72, 72), (192, 192), False),          # Villaviciosa's offset; data around the frame
    ((288, 271), (384, 384), False),        # Ems's
    ((0, 5), (128, 128), True),             # Wadden's; the margin holds the fill
])
def test_a_padded_delivery_is_registered_and_lands_bitwise(tmp_path, monkeypatch, wkt, fake_build,
                                                           offset, shape, fill_margin):
    """The frame's data start at ``offset`` of the padded arrays while the
    delivered x/y put it at (0, 0): the registration against the reference
    netCDF finds the offset (B08, B03 confirming, on the clear dates), every
    band is cropped there, and the store is bitwise the reference frame (and
    the true pixels of the bands the reference lacks); the naive crop of
    before would have been another place."""
    world = World(seed=1)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    data = site.deliver(2024, world, D24, offset, shape, fill_margin)
    want = frame_values(world, D24)
    dy, dx = offset
    if offset != (0, 0):
        assert not np.array_equal(data["B08"][:, :64, :64], want["B08"])     # raw[0:H, 0:W]
    req, res, text = site.run(2024)
    assert res["status"] == "done" and res["verdict"] == "identical", text
    assert res["accepted"] == "identical" and res["registration"] == f"netcdf [{dy}, {dx}]"
    ds = zc.open_store(req.store)
    fx, fy = axes_of(FRAME64)
    np.testing.assert_array_equal(ds["x"].values, fx)
    np.testing.assert_array_equal(ds["y"].values, fy)
    for b in BANDS:
        assert ds[b].dtype == np.int16
        np.testing.assert_array_equal(ds[b].values, want[b], err_msg=b)
    ds.close()
    a = _attrs(req.store)
    assert a["complete"] == 1 and json.loads(a["verification"])["how"] == "identical"
    assert json.loads(a["delivered_offset_px"]) == [dy, dx]
    assert json.loads(a["delivered_shape_px"]) == list(shape)
    assert a["registration_method"] == "netcdf" and a["registration_reference"] == str(site.nc)
    rec = json.loads(a["registration"])
    assert rec["coordinate_offset"] == [0, 0] and rec["verified"] is True
    assert rec["offsets_searched"] == [shape[0] - 63, shape[1] - 63]
    assert rec["rule"]["min_r"] == dcz.REGISTRATION_MIN_R
    assert sorted(json.loads(a["registration_dates"])) == [str(d.date()) for d in D24]
    r = json.loads(a["registration_r"])
    assert sorted(r) == ["B03", "B08"] and all(v == 1.0 for t_ in r.values() for v in t_.values())
    assert all(v == 1.0 for v in rec["netcdf"]["scl_equal"].values())     # the 20 m SCL too
    assert sorted(rec["netcdf"]["scl_confirmed"]) == sorted(r["B08"])     # best AT the offset
    per = rec["netcdf"]["per_date"]
    assert all(v["second"] < 0.2 and v["overlap"] == 64 * 64 and v["centred"]
               and max(abs(s) for s in v["subpixel"] if s is not None) < 0.1   # None: offset 0
               for d in per.values() for v in d.values())
    # the 20 m layer: B8A against the delivery's own B08 frame peaks at the offset; the 20 m
    # cells of the placed frame are at phase (0, 0) of the lattice
    assert sorted(rec["layer_20m"]["confirmed"]) == sorted(r["B08"])
    assert not rec["layer_20m"]["contradicted"]
    ph = rec["phase_20m"]
    assert ph["contradicted"] == [] and sorted(ph["bands"]) == ["B11", "B12", "B8A"]
    assert all(s["shares"]["00"] == 1.0 and max(v for k, v in s["shares"].items() if k != "00") < 0.6
               for s in ph["bands"].values())
    assert (f"registration (netcdf): the frame (64 x 64 px) starts at row {dy}, col {dx} of the "
            f"delivered {shape[0]} x {shape[1]} px arrays (their x/y put it at row 0, col 0)") in text
    fmt = req.sidecar("format.txt").read_text(encoding="utf-8")
    assert (f"at the registered offset (netcdf): cropped rows {dy}:{dy + 64} cols {dx}:{dx + 64}; "
            f"the delivered x/y put the frame at row 0, col 0: not used") in fmt
    assert ((": only the fill" in fmt) if fill_margin else
            ("holds data" in fmt and "cut off by the registered offset" in fmt))
    rep = json.loads(req.sidecar("verify.json").read_text())
    assert rep["acceptance"]["accepted"] and rep["registration"]["delivered_offset_px"] == [dy, dx]


def test_a_cloudy_delivery_is_refused_not_guessed(tmp_path, monkeypatch, wkt, fake_build):
    """No date of the delivery is clear enough (SCL 9 everywhere): the offset
    is not guessed; nothing is written, the raw stays, the FORMAT REPORT
    says why."""
    world = World(seed=2, cloudy=D24)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    with pytest.raises(dcz.RegistrationRefused, match="a cloudy delivery is refused, not guessed"):
        site.run(2024)
    req = site.request(2024)
    assert not req.store.exists() and (Path(req.part) / dcz.ASSETS_DONE).is_file()
    fmt = req.sidecar("format.txt").read_text(encoding="utf-8")
    assert "NOT NORMALISABLE" in fmt and "0 usable date(s) among the 4 in common" in fmt


def test_dates_disagreeing_on_the_offset_are_refused(tmp_path, monkeypatch, wkt, fake_build):
    world = World(seed=3)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, [(72, 72), (70, 72), (72, 72), (72, 72)], (192, 192))
    with pytest.raises(dcz.RegistrationRefused, match=r"disagree on the offset .*B08 \(70, 72\)"):
        site.run(2024)
    assert not site.request(2024).store.exists()


def test_an_offset_without_a_unique_peak_is_refused(tmp_path, monkeypatch, wkt, fake_build):
    """A scene repeating every 16 px in x correlates perfectly at several
    offsets: not clearly above the second best, refused."""
    world = World(seed=4, period=16)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    with pytest.raises(dcz.RegistrationRefused, match=r"not unique: r 1\.00000 at offset"):
        site.run(2024)


D20 = pd.date_range("2020-06-06", periods=5, freq="5D")
D19 = pd.date_range("2019-06-07", periods=4, freq="5D")


def test_a_year_without_reference_takes_a_registered_siblings_offset_and_checks_it(
        tmp_path, monkeypatch, wkt, fake_build):
    """2023 is not covered by the reference (2024 only): the offset of the
    registered 2024 store of the same request is taken once the MEDIAN
    COMPOSITES of the clear dates of the two years (this delivery's full
    arrays, the store's frame) put the correlation peak there for B08 and B03
    (r >= REGISTRATION_SIBLING_MIN_R, unique and centred); recorded as not
    verified. Composites, not pairs of single dates: a year whose dates sit
    1 px off one another (product geolocation) is still confirmed. A
    delivery at another offset, or of another delivered shape, is refused."""
    world = World(seed=5)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    req24, res24, _ = site.run(2024)
    assert res24["registration"] == "netcdf [72, 72]"
    site.deliver(2023, world, D23, (72, 72), (192, 192))
    req, res, text = site.run(2023)
    assert res["verdict"] == "no reference" and res["accepted"] == "not verified"
    assert res["registration"] == "sibling store [72, 72]"
    ds = zc.open_store(req.store)
    want = frame_values(world, D23)
    for b in BANDS:
        np.testing.assert_array_equal(ds[b].values, want[b], err_msg=b)
    ds.close()
    a = _attrs(req.store)
    assert a["registration_method"] == "sibling store"
    assert json.loads(a["delivered_offset_px"]) == [72, 72]
    assert a["registration_reference"] == str(req24.store)
    rec = json.loads(a["registration"])
    sib = rec["sibling"]
    assert rec["verified"] is False and sib["sibling_year"] == 2024 and sib["outcome"] == "confirmed"
    assert sib["dates"] == [str(d.date()) for d in D23]
    assert sib["sibling_dates"] == [str(d.date()) for d in D24]
    assert all(v["unique"] and v["centred"] for v in sib["per_band"].values())
    rs = json.loads(a["registration_r"])
    assert sorted(rs) == ["B03", "B08"]
    assert all(t_["composite"] >= dcz.REGISTRATION_SIBLING_MIN_R for t_ in rs.values())
    assert rec["layer_20m"]["confirmed"] and not rec["layer_20m"]["contradicted"]
    assert "registration (sibling store)" in text
    assert "confirmed by the median composites of 4 clear dates of this delivery and 4 of that store" \
        in text
    # dates 1 px off one another (2 of 5): single pairs would disagree, the composites do not
    site.deliver(2020, world, D20, [(72, 72), (72, 73), (72, 72), (72, 71), (72, 72)], (192, 192))
    _, res20, _ = site.run(2020)
    assert res20["registration"] == "sibling store [72, 72]"
    # another offset in 2022: the sibling's is not confirmed; the message says where the frame is
    site.deliver(2022, world, D22, (70, 72), (192, 192))
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"offset \(72, 72\) of the registered 2024 store is not confirmed .*"
                             r"the composites put the frame at \(70, 72\)"):
        site.run(2022)
    # another delivered shape in 2021: no sibling of the same request (2020, 2023: no reference)
    site.deliver(2021, world, D21, (72, 72), (256, 256))
    with pytest.raises(dcz.RegistrationRefused, match=r"2023: not registered against a reference "
                                                      r"netCDF; 2024: its delivery was \[192, 192\]"):
        site.run(2021)


def test_offset_px_is_recorded_and_checked(tmp_path, monkeypatch, wkt, fake_build):
    """--offset-px. With a reference covering the year the registration must
    agree (a wrong one is refused, a right one recorded as confirmed).
    Without one it is CHECKED against the median composites of a store of
    the site registered against its reference: refused when they put the
    frame elsewhere (the site's offset no longer holds), recorded as
    confirmed when they agree; when they cannot measure (a cloudy delivery)
    it is refused if the delivered arrays are not that store's shape, unless
    --accept-unverified-offset (then taken, recorded, with a warning).
    Outside the delivered arrays it is refused; without it, a year with
    neither a reference nor a registered store is refused with the hint."""
    world = World(seed=6)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    site.ctx.offset_px = (70, 72)
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"--offset-px 70,72 contradicts the registration against ref\.nc: "
                             r"\(72, 72\)"):
        site.run(2024)
    site.ctx.offset_px = (72, 72)
    req, res, _ = site.run(2024)
    rec = json.loads(_attrs(req.store)["registration"])
    assert rec["method"] == "netcdf" and rec["verified"] and rec["expert"] == [72, 72]
    assert "--offset-px 72,72 confirmed" in rec["why"]
    site.deliver(2023, world, D23, (72, 72), (192, 192))
    req23, res23, _ = site.run(2023)
    a = _attrs(req23.store)
    assert a["registration_method"] == "expert --offset-px" and res23["accepted"] == "not verified"
    rec23 = json.loads(a["registration"])
    assert rec23["verified"] is False and rec23["sibling"]["outcome"] == "confirmed"
    assert rec23["checked"] == "confirmed by the median composites against the registered 2024 store"
    assert "not verified against a reference netCDF (ref.nc covers [2024-01-01, 2025-01-01)" \
        in rec23["why"]
    np.testing.assert_array_equal(zc.open_store(req23.store)["B08"].values,
                                  frame_values(world, D23)["B08"])
    # the frame elsewhere in this year's delivery: the site's offset is refused, the right one taken
    site.deliver(2021, world, D21, (70, 72), (192, 192))
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"--offset-px 72,72 contradicts the median composites against the "
                             r"registered 2024 store, which put the frame at \(70, 72\)"):
        site.run(2021)
    assert not site.request(2021).store.exists()
    site.ctx.offset_px = (70, 72)
    req21, res21, _ = site.run(2021)
    assert res21["registration"] == "expert --offset-px [70, 72]"
    np.testing.assert_array_equal(zc.open_store(req21.store)["B08"].values,
                                  frame_values(world, D21)["B08"])
    # a cloudy year delivered in other arrays: nothing can check the offset -> refused ...
    cloudy = World(seed=6, cloudy=D19)
    site.deliver(2019, cloudy, D19, (72, 72), (256, 256))
    site.ctx.offset_px = (72, 72)
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"--offset-px 72,72 cannot be checked against the registered 2024 store "
                             r".*not that store's \(\[192, 192\]\).*--accept-unverified-offset"):
        site.run(2019)
    # ... unless --accept-unverified-offset: taken, recorded, a warning
    site.ctx.accept_unverified_offset = True
    req19, res19, text19 = site.run(2019)
    rec19 = json.loads(_attrs(req19.store)["registration"])
    assert rec19["method"] == "expert --offset-px" and rec19["sibling"]["outcome"] == "unmeasured"
    assert "--accept-unverified-offset" in rec19["checked"] and "NOT confirmed" in text19
    site.ctx.accept_unverified_offset = False
    site.deliver(2022, world, D22, (72, 72), (192, 192))
    site.ctx.offset_px = (200, 0)
    with pytest.raises(dcz.RegistrationRefused, match=r"puts the frame outside the delivered "
                                                      r"arrays \(offsets 0\.\.128 x 0\.\.128\)"):
        site.run(2022)
    site.ctx.offset_px = None
    with pytest.raises(dcz.RegistrationRefused, match=r"no registered store of villaviciosa from "
                                                      r"the same request \(none complete\).*"
                                                      r"--offset-px ROWS,COLS"):
        site.run(2022, root=tmp_path / "elsewhere")
    assert dcz.parse_args(["villaviciosa", "--offset-px", "72,72"]).offset_px == (72, 72)
    assert dcz.parse_args(["villaviciosa", "--offset-px", "72,72",
                           "--accept-unverified-offset"]).accept_unverified_offset
    for bad in (["--offset-px", "72,72"], ["villaviciosa", "--offset-px", "72"],
                ["villaviciosa", "--test", "--offset-px", "1,2"],
                ["villaviciosa", "ems", "--offset-px", "1,2"], ["--renormalise", "--test"],
                ["villaviciosa", "--renormalise", "--new-job"],
                ["villaviciosa", "--accept-unverified-offset"]):
        with pytest.raises(SystemExit):
            dcz.parse_args(bad)


def test_a_store_that_differs_from_the_reference_is_not_completed(tmp_path, monkeypatch, wkt,
                                                                  fake_build):
    """The registration passes (one reference date differs by +1 in B08: r
    is blind to it), but the verification after writing finds B08 equal on
    75 % of the pixel-dates: the store is NOT marked complete, the run stops
    with a message pointing at verify.json, the raw stays; verify.json says
    where the differences are (per block) and that every date is in place
    (the geometry). With --accept-differences the store already written is
    verified again (not rewritten: no second copy), completed and the
    acceptance recorded; the raw is kept although --keep-raw is off (the
    only way to register that year again)."""
    world = World(seed=7)

    def tweak(ref):
        ref["B08"][1] += 1

    site = Site(tmp_path, monkeypatch, wkt, world, D24, tweak=tweak)
    site.ctx.keep_raw = False
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    with pytest.raises(dcz.VerificationRefused, match=r"NOT marked complete.*2024\.verify\.json.*"
                                                      r"--accept-differences"):
        site.run(2024)
    req = site.request(2024)
    assert req.store.exists() and not zc.is_complete(req.store)
    assert zc.list_years("villaviciosa", site.root) == []
    assert _attrs(req.store)["verification"].startswith("pending")
    rep = json.loads(req.sidecar("verify.json").read_text())
    assert rep["verdict"] == "differs" and rep["acceptance"]["accepted"] is False
    why = rep["acceptance"]["why"]
    assert "B08 equal on 75.0000% of 16384 pixel-dates valid in either store (< 99.9%)" in why
    assert "geometry: the store's B08 is in place (peak at (0, 0) within 3 px" in why
    geo = rep["geometry"]
    assert geo["checked"] and geo["same"] == geo["dates_checked"] == 4 and not geo["displaced"]
    blk = rep["bands"]["B08"]["differing_by_block"]
    assert blk == {"block_px": 100, "columns": [0.25], "rows": [0.25]}
    assert (Path(req.part) / dcz.ASSETS_DONE).is_file()                  # the raw is kept
    site.ctx.accept_differences = True
    req, res, text = site.run(2024)
    assert res["status"] == "verified again" and zc.is_complete(req.store)
    assert res["accepted"] == "accepted with --accept-differences"
    assert "is verified again, NOT rewritten" in text
    acc = json.loads(_attrs(req.store)["verification"])
    assert acc["how"] == "accepted with --accept-differences" and "B08 equal on 75" in acc["why"]
    assert sorted(req.store.parent.glob("2024.zarr.superseded-*")) == []  # no second copy
    assert json.loads(req.sidecar("verify.json").read_text())["acceptance"]["accepted"] is True
    assert (Path(req.part) / dcz.ASSETS_DONE).is_file() and "raw download kept" in text


def test_dates_that_do_not_measure_are_set_aside(tmp_path, monkeypatch, wkt, fake_build):
    """Against the reference, a usable date that does not measure is SET ASIDE
    and the next clearest one taken. The reference here mosaics its six
    westernmost columns from another product on every date (Villaviciosa's
    strip: r below 1 at the true offset), strongly on one date (r between
    REGISTRATION_MIN_R_ANY and REGISTRATION_MIN_R: counted, not one of the
    REGISTRATION_MIN_DATES strong ones), has no data on another (set aside)
    and differs beyond REGISTRATION_MIN_R_ANY on a third (set aside): four
    dates count, three of them strong, all at the true offset: registered.
    The verification then refuses the values (B08 equal on ~75 %), with every
    checked date in place and the differences in the first 8-column block."""
    dates = pd.date_range("2024-06-02", periods=6, freq="5D")
    world = World(seed=13)
    rng = np.random.default_rng(13)

    def tweak(ref):
        for b in ("B08", "B03"):
            s = ref[b][:, :, :6]
            amp = np.array([150, 800, 0, 3000, 150, 150])[:, None, None]
            noise = rng.uniform(-1, 1, s.shape) * amp
            ref[b][:, :, :6] = np.clip(s + noise, 1, 30000).astype("int16")
        for b in ("B08", "B03", "SCL"):
            ref[b][2] = FILL                                     # no data on date 2

    site = Site(tmp_path, monkeypatch, wkt, world, dates, tweak=tweak)
    site.deliver(2024, world, dates, (72, 72), (192, 192))
    monkeypatch.setattr(zc, "DIFF_BLOCK_PX", 8)
    with pytest.raises(dcz.VerificationRefused, match=r"B08 equal on"):
        site.run(2024)
    req = site.request(2024)
    rec = json.loads(_attrs(req.store)["registration"])
    net = rec["netcdf"]
    d = [str(x.date()) for x in dates]
    assert rec["method"] == "netcdf" and rec["offset"] == [72, 72]
    assert net["dates"] == [d[0], d[1], d[4], d[5]] and net["dates_tried"] == 5
    assert sorted(net["set_aside"]) == [d[2], d[3]]
    assert "the reference frame holds data on 0%" in net["set_aside"][d[2]]
    assert re.search(r"B08 r 0\.\d+ < 0\.98", net["set_aside"][d[3]])
    assert dcz.REGISTRATION_MIN_R_ANY <= net["r"]["B08"][d[1]] < dcz.REGISTRATION_MIN_R
    assert net["dates_min_r"] == [d[0], d[4], d[5]]
    rep = json.loads(req.sidecar("verify.json").read_text())
    geo = rep["geometry"]
    assert geo["dates_checked"] == 5 and geo["skipped"] == 1 and not geo["displaced"]
    assert geo["same"] == 4 and list(geo["undetermined"]) == [d[3]]
    cols = rep["bands"]["B08"]["differing_by_block"]["columns"]
    assert cols[0] > 0.5 > max(cols[1:])                       # the strip: the first 8 columns


def test_accept_differences_never_accepts_a_displaced_date(tmp_path, monkeypatch, wkt, fake_build):
    """One reference date sits 2 px east of the delivery's (a date the
    registration does not use: its delivered SCL is cloudy): the
    verification finds that date DISPLACED (its peak at (0, -2), r 1) and
    refuses the store; --accept-differences does not accept it."""
    dates = pd.date_range("2024-06-02", periods=6, freq="5D")
    world = World(seed=14, cloudy=dates[5:])

    def tweak(ref):
        for b in ("B03", "B08", "SCL"):
            ref[b][5] = np.roll(ref[b][5], 2, axis=1)

    site = Site(tmp_path, monkeypatch, wkt, world, dates, tweak=tweak)
    site.deliver(2024, world, dates, (72, 72), (192, 192))
    with pytest.raises(dcz.VerificationRefused, match=r"GEOMETRY: DISPLACED on 1 date"):
        site.run(2024)
    req = site.request(2024)
    assert json.loads(_attrs(req.store)["registration"])["offset"] == [72, 72]
    rep = json.loads(req.sidecar("verify.json").read_text())
    assert rep["geometry"]["displaced"] == {str(dates[5].date()): [0, -2, 1.0]}
    site.ctx.accept_differences = True
    with pytest.raises(dcz.VerificationRefused, match=r"--accept-differences was NOT applied"):
        site.run(2024)
    assert not zc.is_complete(req.store)
    acc = json.loads(req.sidecar("verify.json").read_text())["acceptance"]
    assert acc["accepted"] is False and "NOT applied" in acc["accept_differences"]


def test_a_displaced_20m_layer_is_refused(tmp_path, monkeypatch, wkt, fake_build):
    """The offset measured on B08 / B03 must hold for the 20 m layer (B8A
    B11 B12 SCL CLD). Against the reference, the delivered SCL matching the
    reference's better 2 px away refuses the year; against a registered
    store, the delivery's B8A peaking 1 px away from its own B08 frame
    refuses it; and a 20 m band at another phase of the 20 m lattice than the
    placed frame refuses it whatever the source of the offset."""
    world = World(seed=11)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192), offset20=(72, 74))
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"the delivered SCL \(20 m\) matches the reference's better elsewhere "
                             r"than at \(72, 72\).* at \(72, 74\)"):
        site.run(2024)
    req24 = site.request(2024)
    assert not req24.store.exists()
    shutil.rmtree(req24.part)                                  # a right delivery of 2024
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    site.run(2024)
    site.deliver(2023, world, D23, (72, 72), (192, 192), offset20=(72, 73))
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"20 m layer DISPLACED: B8A against the delivery's own B08 frame peaks "
                             r"elsewhere \(2023-06-03 at \(72, 73\)"):
        site.run(2023)
    # no reference, no registered store: --offset-px; B8A in place, B11 and B12 1 px east
    site.deliver(2022, world, D22, (72, 72), (192, 192), offset20=(72, 73), bands20=("B11", "B12"))
    site.ctx.offset_px = (72, 72)
    with pytest.raises(dcz.RegistrationRefused, match=r"20 m phase CONTRADICTED for B11, B12"):
        site.run(2022, root=tmp_path / "elsewhere")
    assert not site.request(2022, root=tmp_path / "elsewhere").store.exists()


def test_a_frame_sized_delivery_without_reference_is_checked(tmp_path, monkeypatch, wkt,
                                                             fake_build):
    """A delivery of the frame's shape has no other offset, but in a year
    without reference its composites are searched within
    REGISTRATION_FRAME_SIZED_PX of (0, 0) against a registered store: in
    place, recorded as confirmed; its data 3 rows down, refused."""
    world = World(seed=12)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    site.run(2024)
    site.deliver(2023, world, D23, (0, 0), (64, 64))
    req, res, _ = site.run(2023)
    rec = json.loads(_attrs(req.store)["registration"])
    assert rec["method"] == "frame-sized" and res["registration"] == "frame-sized [0, 0]"
    assert rec["checked"].startswith("(0, 0) confirmed within 8 px by the median composites")
    site.deliver(2022, world, D22, (3, 0), (64, 64))
    with pytest.raises(dcz.RegistrationRefused,
                       match=r"composites against the registered 2024 store peak at \(3, 0\), not "
                             r"\(0, 0\)"):
        site.run(2022)


def test_a_near_identical_store_is_accepted(tmp_path, monkeypatch, wkt, fake_build):
    """One pixel-date of B08 differs (1 / 16384 < 0.1 %): near-identical,
    completed, and the decision says so. --verify-only applies the same rule
    (exit 0) and writes its decision into verify.json with the one made at
    writing."""
    world = World(seed=8)

    def tweak(ref):
        ref["B08"][2, 10, 10] += 1

    site = Site(tmp_path, monkeypatch, wkt, world, D24, tweak=tweak)
    site.deliver(2024, world, D24, (72, 72), (192, 192))
    req, res, _ = site.run(2024)
    assert res["verdict"] == "differs" and res["accepted"] == "near-identical"
    assert zc.is_complete(req.store)
    assert "B08 99.9939%" in json.loads(_attrs(req.store)["verification"])["why"]
    monkeypatch.setattr(dcz, "make_log", lambda r: (lambda m: None))
    zc.write_json({str(d.date()): f"{d.date()}T11:00:00" for d in D24},
                  zc.overpass_path("villaviciosa", site.root))       # --verify-only checks them too
    assert dcz.verify_only([req]) == 0
    acc = json.loads(req.sidecar("verify.json").read_text())["acceptance"]
    assert acc["accepted"] and acc["how"] == "near-identical" and acc["by"] == "--verify-only"
    assert acc["at_writing"]["how"] == "near-identical"


def test_verification_decision_rule():
    def rep(verdict, identical=True, dates=None, geometry=None, **shares):
        bands = {b: {"pixels": 1000, "differing_pixels": round((1 - s) * 1000)}
                 for b, s in shares.items()}
        return {"verdict": verdict, "grid": {"identical": identical}, "bands": bands,
                "reason": "why", "dates": dates or {}, "geometry": geometry or {}}

    assert dcz.verification_decision(rep("identical"))[:2] == (True, "identical")
    assert dcz.verification_decision(rep("identical on the common window"))[0]
    assert dcz.verification_decision(rep("no reference"))[:2] == (True, "not verified")
    ok = rep("differs", B03=1.0, B08=0.999, SCL=0.99)
    assert dcz.verification_decision(ok)[:2] == (True, "near-identical")
    for bad in (rep("differs", B03=1.0, B08=0.998, SCL=1.0),
                rep("differs", B03=1.0, B08=1.0, SCL=0.98), rep("differs", B03=1.0, B08=1.0),
                rep("differs", False, B03=1.0, B08=1.0, SCL=1.0), rep("not comparable")):
        assert dcz.verification_decision(bad)[0] is False, bad
    # the equal share counts the pixel-dates valid in either store (no data in both left out)
    half = rep("differs", B03=1.0, B08=1.0, SCL=1.0)
    half["bands"]["B08"].update(nodata_both=990, differing_pixels=2)
    acc, how, why = dcz.verification_decision(half)
    assert not acc and "B08 equal on 80.0000% of 10 pixel-dates valid in either store" in why
    # a reference date missing from the store, or a displaced date: refused, whatever the values
    miss = rep("differs", dates={"only_reference": ["2024-06-02"]}, B03=1.0, B08=1.0, SCL=1.0)
    assert dcz.verification_decision(miss)[0] is False
    assert "1 date(s) of the reference missing from the store" in dcz.verification_decision(miss)[2]
    geo = {"checked": True, "band": "B08", "search_px": 3, "min_r": 0.98, "dates_checked": 4,
           "same": 3, "displaced": {"2024-06-07": [0, -2, 1.0]}, "undetermined": {}, "skipped": 0}
    disp = rep("differs", geometry=geo, B03=1.0, B08=0.9995, SCL=1.0)
    acc, _, why = dcz.verification_decision(disp)
    assert not acc and why.startswith("GEOMETRY: DISPLACED on 1 date(s) (2024-06-07 by (0, -2)")
    assert dcz.geometry_blocks(geo) and not dcz.geometry_blocks(dict(geo, displaced={}))
    assert dcz.geometry_blocks(dict(geo, displaced={}, same=1))       # in place on < half


def test_masked_correlation_is_the_linear_one_at_every_offset():
    """The FFT surface is Pearson r over the pixels valid in both, at every
    offset of the frame inside the array (checked against the direct
    computation), and never a cyclic shift: a frame that only matches when
    wrapped around the array's edge is not found."""
    rng = np.random.default_rng(9)
    img = rng.integers(0, 3000, (23, 31)).astype("int16")
    tpl = rng.integers(0, 3000, (9, 12)).astype("int16")
    ok_i, ok_t = rng.random(img.shape) > 0.1, rng.random(tpl.shape) > 0.1
    r, n = dcz.masked_correlation(img, ok_i, tpl, ok_t)
    assert r.shape == n.shape == (15, 20)
    for dy in range(15):
        for dx in range(20):
            rd, nd = dcz.window_r(img, ok_i, tpl, ok_t, dy, dx)
            assert n[dy, dx] == nd and r[dy, dx] == pytest.approx(rd, abs=1e-9)
    wrapped = np.roll(img, (-5, -27), axis=(0, 1))[:9, :12]       # columns 27..30 then 0..7
    ok_w = np.ones(wrapped.shape, bool)
    r2, _ = dcz.masked_correlation(img, np.ones(img.shape, bool), wrapped, ok_w)
    assert np.nanmax(r2) < 0.6
    res = dcz.offset_of(img, np.ones(img.shape, bool), wrapped, ok_w, dcz.REGISTRATION_MIN_R)
    assert not res["pass"]


def test_the_peak_must_be_centred_and_clear_pixels_are_counted_date_by_date():
    """``_subpixel``: the parabola through the peak and its neighbours puts
    a symmetric peak at 0 and a flat top (two equal best offsets) half a
    pixel away, which ``offset_of`` refuses as not centred (the margin rule
    alone cannot see it). ``clear_counts`` reads one date at a time, in the
    window given, and equals the counts over the whole arrays;
    ``core_window`` is the part inside the frame at every offset."""
    import dask.array as da

    assert dcz._subpixel(np.array([[0, .5, 0], [.5, 1, .5], [0, .5, 0]]), 1, 1) == [0.0, 0.0]
    assert dcz._subpixel(np.array([[0, .5, 0], [.7, 1, .5], [0, .5, 0]]), 1, 1)[1] == -0.125
    assert dcz._subpixel(np.array([[0, .5, 0], [1, 1, .5], [0, .5, 0]]), 1, 1)[1] == -0.5
    assert dcz._subpixel(np.array([[1.0, 0.9]]), 0, 0) == [None, None]
    rng = np.random.default_rng(4)
    base = rng.integers(0, 3000, (40, 61)).astype("float64")
    img = (base[:, :-1] + base[:, 1:]) / 2.0              # every column the mean of two
    tpl = img[10:30, 20:40].copy()
    tpl[:, :] = (base[10:30, 20:40] + base[10:30, 21:41] + base[10:30, 22:42]) / 3.0
    res = dcz.offset_of(img, np.ones(img.shape, bool), tpl, np.ones(tpl.shape, bool), 0.5)
    assert res["unique"] and abs(res["subpixel"][1]) > dcz.REGISTRATION_MAX_SUBPIXEL
    assert not res["centred"] and not res["pass"] and "not centred" in res["why"]
    scl = rng.integers(0, 12, (7, 40, 56)).astype("int16")
    scl[2, :5] = FILL
    arr = xr.DataArray(da.from_array(scl, chunks=(1, 8, 8)), dims=("t", "y", "x"))
    nv, nc = dcz.clear_counts(arr, [0, 2, 6], FILL, slice(4, 30), slice(10, 50))
    sub = scl[[0, 2, 6], 4:30, 10:50]
    assert list(nv) == [int(((s != 0) & (s != FILL)).sum()) for s in sub]
    assert list(nc) == [int(np.isin(s, (4, 5, 6)).sum()) for s in sub]
    nv_all, _ = dcz.clear_counts(arr, range(7), FILL)
    assert list(nv_all) == [int(((s != 0) & (s != FILL)).sum()) for s in scl]
    assert dcz.core_window((1152, 1152), (915, 915)) == (slice(237, 915), slice(237, 915))
    assert dcz.core_window((3840, 3968), (3206, 3413)) == (slice(634, 3206), slice(555, 3413))
    assert dcz.core_window((192, 192), (64, 64)) == (slice(0, 192), slice(0, 192))


def _displaced_store(tmp_path, monkeypatch, wkt, seed):
    """The server's state on 2026-10-19: a COMPLETE store cut naively (by the
    delivered x/y) from a padded delivery kept with --keep-raw (its ledger
    made for the request's job title). Returns ``(site, req, world,
    displaced B08)``."""
    world = World(seed=seed)
    site = Site(tmp_path, monkeypatch, wkt, world, D24)
    data = site.deliver(2024, world, D24, (72, 72), (192, 192))
    req = site.request(2024)
    meta = dcz.download_results(None, req, lambda m: None)              # the kept raw
    title = dcz.job_title(req, dcz.graph_hash(dcz.build_process(req).flat_graph()))
    led = dcz.Ledger(req.part, title)
    led.update(led.add("j-7", "finished", "created"), info={"id": "j-7", "title": title})
    ref = dcz.reference_grid("villaviciosa", root=site.root)
    ds, interp, _ = dcz.normalise(req, lambda m: None, job_info={"id": "j-7"}, results_meta=meta,
                                  ref_grid=ref)                          # no registration: x/y
    attrs = dcz.provenance(req, {"id": "j-7"}, meta, interp, dcz.Context(),
                           dcz.build_process(req).flat_graph(), ref_grid=ref)
    dcz.write_store(ds, interp.fills, req.store, attrs, log=lambda m: None)
    zc.write_json({"verdict": "differs", "note": "the evidence of the bug"}, req.sidecar("verify.json"))
    displaced = zc.open_store(req.store)["B08"].values
    np.testing.assert_array_equal(displaced, data["B08"][:, :64, :64])  # raw[0:H, 0:W]: wrong
    assert zc.is_complete(req.store) and "registration" not in _attrs(req.store)
    return site, req, world, displaced


def test_renormalise_rebuilds_a_displaced_store_from_its_raw_offline(tmp_path, monkeypatch, wkt,
                                                                     fake_build, capsys):
    """--renormalise --dry-run measures the offset and says what it would
    move, changing nothing; --renormalise rebuilds the store from the part
    with the registration, without any network call or job (a connection
    that fails on any use), moves the old one aside (never deleted,
    untouched) once the new one is written, keeps the old sidecars beside
    it, verifies it and updates the sidecars; with the raw gone it refuses.
    A run that skips the complete store before says it was written before
    the registration."""
    import socket

    site, req, world, displaced = _displaced_store(tmp_path, monkeypatch, wkt, 10)
    msgs = []
    assert dcz.run_request(site.ctx, req, msgs.append)["pre_registration"] is True
    assert any("2024.zarr was written before the registration (2026-10-19)" in m for m in msgs)

    def no_network(*a, **k):
        raise AssertionError("network used by --renormalise")

    monkeypatch.setattr(dcz.Context, "connection", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.chdir(tmp_path)
    site_dir = zc.site_dir("villaviciosa", site.root)

    def snapshot():
        return sorted((p.relative_to(site_dir).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in site_dir.rglob("*") if p.is_file())

    before = snapshot()
    argv = ["villaviciosa", "--years", "2024", "--renormalise", "--root", str(site.root)]
    assert dcz.main(argv + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert ("DRY RUN: registration netcdf: offset [72, 72] of the frame [64, 64] in the delivered "
            "[192, 192] px (their x/y: [0, 0])") in out
    assert "DRY RUN: would move" in out and "2024.zarr (complete) to 2024.zarr.superseded-<UTC>" in out
    assert "DRY RUN: 20 m layer confirmed" in out and "20 m phase check passed" in out
    assert "needed beside the kept store" in out
    assert snapshot() == before                                        # nothing written
    assert dcz.main(argv) == 0
    aside = sorted(site_dir.glob("2024.zarr.superseded-*"))
    assert len(aside) == 1 and zc.is_complete(aside[0])
    np.testing.assert_array_equal(zc.open_store(aside[0])["B08"].values, displaced)   # untouched
    stamp = aside[0].name.split(".superseded-")[1]
    old_verify = site_dir / f"2024.verify.json.superseded-{stamp}"
    assert json.loads(old_verify.read_text())["note"] == "the evidence of the bug"
    assert (site_dir / f"2024.format.txt.superseded-{stamp}").is_file()
    new = zc.open_store(req.store)
    want = frame_values(world, D24)
    for b in BANDS:
        np.testing.assert_array_equal(new[b].values, want[b], err_msg=b)
    new.close()
    a = _attrs(req.store)
    assert a["complete"] == 1 and json.loads(a["delivered_offset_px"]) == [72, 72]
    assert a["registration_method"] == "netcdf" and a["job_id"] == "j-7"
    assert a["renormalised_from"] == str(req.part) and a["superseded_store"] == str(aside[0])
    rep = json.loads(req.sidecar("verify.json").read_text())
    assert rep["verdict"] == "identical" and rep["acceptance"]["accepted"] is True
    assert "cropped rows 72:136 cols 72:136" in req.sidecar("format.txt").read_text(encoding="utf-8")
    notes = json.loads((Path(req.part) / "job.json").read_text())["notes"]
    assert notes[-1]["action"] == "--renormalise" and notes[-1]["delivered_offset_px"] == [72, 72]
    assert notes[-1]["superseded_store"] == str(aside[0]) and notes[-1]["verification"]["accepted"]
    assert str(old_verify) in notes[-1]["sidecars_kept"]
    assert json.loads(req.sidecar("job.json").read_text())["ledger"]["notes"] == notes
    assert (Path(req.part) / dcz.ASSETS_DONE).is_file()                 # the raw is kept
    assert site.conn.created == []                                       # no job
    # the raw gone: refused, the store untouched
    shutil.rmtree(req.part)
    kept = snapshot()
    assert dcz.main(argv) == 1
    assert "no raw delivery to rebuild 2024.zarr" in capsys.readouterr().out
    assert [f for f in snapshot() if not f[0].endswith("download.log")] == \
        [f for f in kept if not f[0].endswith("download.log")]


def test_renormalise_never_leaves_a_year_without_its_store(tmp_path, monkeypatch, wkt, fake_build,
                                                          capsys):
    """--renormalise refuses a part made for another job title, and refuses
    to start without the disk space for a second copy: nothing changes. A
    failure while writing (here: memory) leaves the old store where it was,
    complete and untouched (it is moved aside only once the new one is
    written whole), and the ledger's note says it failed; the next run
    rebuilds the store."""
    site, req, world, displaced = _displaced_store(tmp_path, monkeypatch, wkt, 15)
    monkeypatch.chdir(tmp_path)
    site_dir = zc.site_dir("villaviciosa", site.root)
    argv = ["villaviciosa", "--years", "2024", "--renormalise", "--root", str(site.root)]

    def snapshot():
        return sorted((p.relative_to(site_dir).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in site_dir.rglob("*") if p.is_file() and p.name != "download.log")

    led_path = Path(req.part) / "job.json"
    good = led_path.read_text()
    led = json.loads(good)
    led["title"] = "zarr_cube_villaviciosa_2024_deadbeef"
    led_path.write_text(json.dumps(led))
    before = snapshot()
    assert dcz.main(argv) == 1
    assert "was made for the job title 'zarr_cube_villaviciosa_2024_deadbeef'" in capsys.readouterr().out
    assert snapshot() == before
    led_path.write_text(good)
    # a dry run never extracts nor removes: a zip without its '.extracted' flag stops it
    flag = Path(req.part) / "extracted" / "out.zarr.extracted"
    kept_flag = flag.read_text()
    flag.unlink()
    before = snapshot()
    assert dcz.main(argv + ["--dry-run"]) == 1
    assert "dry run: out.zarr.zip is not extracted" in capsys.readouterr().out
    assert snapshot() == before and (Path(req.part) / "extracted" / "out.zarr").is_dir()
    flag.write_text(kept_flag)
    real_usage = shutil.disk_usage
    monkeypatch.setattr(dcz.shutil, "disk_usage",
                        lambda p: real_usage(p)._replace(free=1000))
    before = snapshot()
    assert dcz.main(argv) == 1
    assert "not enough to write the new store beside the kept one" in capsys.readouterr().out
    assert snapshot() == before
    monkeypatch.setattr(dcz.shutil, "disk_usage", real_usage)

    def no_memory(*a, **k):
        raise MemoryError("no memory for the test")

    real_write = dcz.write_store
    monkeypatch.setattr(dcz, "write_store", no_memory)
    assert dcz.main(argv) == 1
    assert "MemoryError: no memory for the test" in capsys.readouterr().out
    assert zc.is_complete(req.store) and sorted(site_dir.glob("2024.zarr.superseded-*")) == []
    np.testing.assert_array_equal(zc.open_store(req.store)["B08"].values, displaced)
    note = json.loads(led_path.read_text())["notes"][-1]
    assert note["failed"].startswith("MemoryError") and note["superseded_store"] is None
    monkeypatch.setattr(dcz, "write_store", real_write)
    assert dcz.main(argv) == 0
    assert json.loads(_attrs(req.store)["delivered_offset_px"]) == [72, 72]
    assert len(sorted(site_dir.glob("2024.zarr.superseded-*"))) == 1
    # a store of this delivery left pending (e.g. refused, then --accept-differences): verified
    # again, not rewritten, and still without any network call
    import socket

    import zarr as zarr_

    def no_network(*a, **k):
        raise AssertionError("network used by --renormalise")

    monkeypatch.setattr(dcz.Context, "connection", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    g = zarr_.open_group(str(req.store), mode="r+")
    g.attrs["complete"], g.attrs["verification"] = 0, dcz.PENDING
    zarr_.consolidate_metadata(str(req.store))
    assert not zc.is_complete(req.store)
    capsys.readouterr()
    assert dcz.main(argv) == 0
    assert "is verified again, NOT rewritten" in capsys.readouterr().out
    assert zc.is_complete(req.store) and len(sorted(site_dir.glob("2024.zarr.superseded-*"))) == 1
    note = json.loads(led_path.read_text())["notes"][-1]
    assert note["action"].startswith("--renormalise (the pending store verified again")
    assert note["verification"]["accepted"] and "failed" not in note


def test_a_test_run_that_cannot_register_warns_and_keeps_the_coordinates(tmp_path, wkt,
                                                                          monkeypatch, fake_build):
    """--test is registered too, but a failure only warns: with no date clear
    enough (SCL of every class) the warning says NOT REGISTERED, the
    delivered x/y place the pixels (the 0:204 crop) and the store is
    written."""
    t = pd.date_range("2025-06-02", periods=3, freq="5D")
    rx, ry = axes_of(TEST_FRAME)
    nc = write_nc(tmp_path / "ref.nc", t, ry, rx, {b: lattice_values(b, 3, rx, ry)
                                                  for b in ("B03", "B08", "SCL")}, wkt)
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(nc),
        reference_period=("2016-01-01", "2025-12-31"), pipeline_overpass=str(tmp_path / "none.json")))
    req = dcz.test_request(tmp_path / "cubes")
    wx, wy = axes_of(TEST_WIN)
    dx, dy = server_axes(TEST_WIN[0], TEST_WIN[3], 256, 256)
    data = {}
    for b in BANDS:
        a = np.full((3, 256, 256), FILL, "int16")
        a[:, :204, :204] = lattice_values(b, 3, wx, wy)
        data[b] = a
    xf, yf = f32_like_the_backend(dx, dy)
    raw = write_server_zarr(tmp_path / "src" / "out.zarr", t, yf, xf, data, wkt, extent=TEST_WIN)
    z = zip_store(raw, tmp_path / "src" / "out.zarr")

    def fake_download(job, req, log, workers=4):
        place(req, [("out.zarr.zip", z, "application/zip")])
        return {}

    monkeypatch.setattr(dcz, "download_results", fake_download)
    ctx, _ = _ctx(FakeConn())
    ctx.network = False
    msgs = []
    res = dcz.run_request(ctx, req, msgs.append)
    text = "\n".join(msgs)
    assert res["status"] == "done" and res["registration"] == "none None"
    assert "WARNING: NOT REGISTERED: the registration against ref.nc: 0 usable date(s)" in text
    assert "the delivered x/y place the pixels (--test only; a production year is refused)" in text
    ds = zc.open_store(req.store)
    np.testing.assert_array_equal(ds["B08"].values, lattice_values("B08", 3, wx, wy))
    ds.close()
