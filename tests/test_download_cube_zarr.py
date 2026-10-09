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


def normalise_and_write(req, ref_grid=None):
    lines = []
    ds, interp, rep = dcz.normalise(req, lines.append, job_info={"id": "j-test", "title": "t"},
                                    results_meta={}, ref_grid=ref_grid)
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
    # same lattice, smaller extent: put on the reference grid (padded), with a warning
    sub = {b: a[:, 2:10, 3:20] for b, a in data.items()}
    raw2 = write_raw_v2(tmp_path / "src" / "s.zarr", t, y[2:10], x[3:20], sub, wkt)
    req2 = make_request(tmp_path / "c2")
    place(req2, [("s.zarr", raw2, "x")])
    ds2, interp2, _ = dcz.normalise(req2, lambda m: None, ref_grid=ref)
    assert any("another extent" in w for w in interp2.warnings)
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
    assert dcz.main(["villaviciosa", "--years", "2023", "--verify-only", "--root", str(root)]) == 0
    rep = json.loads(zc.sidecar_path("villaviciosa", 2023, "verify.json", root).read_text())
    assert rep["verdict"] == "identical" and rep["dates"]["common"] == len(t)
    assert "IDENTICAL" in capsys.readouterr().out
    # 2024 lacks overpass times and is outside the reference; 2025 is not downloaded
    assert dcz.main(["villaviciosa", "--years", "2024-2025", "--verify-only", "--root",
                     str(root)]) == 1
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
    """SentinelCube.ensure's graph == ours, but for the bands and the format."""
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
    assert captured["kw"]["job_options"] == dcz.JOB_OPTIONS
    req = dcz.Request(site="ferrol", year=2023, start="2023-01-01", end="2024-01-01",
                      bbox=dcz.site_bbox("ferrol"), root=tmp_path)
    new = dcz.build_process(req).flat_graph()
    assert new["loadcollection1"]["arguments"]["bands"] == list(BANDS)
    assert new["saveresult1"]["arguments"]["format"] == "Zarr"
    old["loadcollection1"]["arguments"]["bands"] = list(BANDS)
    old["saveresult1"]["arguments"]["format"] = "Zarr"
    assert old == new


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
    """--test: report, a store opened through open_cube, comparison on the
    common window of the reference cube, raw download kept."""
    t, y, x, data = synth(2025, n=4, H=30, W=44, seed=42, start="2025-06-02")
    iy = (np.floor(y / 20) - np.floor(y / 20).min()).astype(int)       # 20 m cells
    ix = (np.floor(x / 20) - np.floor(x / 20).min()).astype(int)
    for b in ("B8A", "B11", "B12", "SCL", "CLD"):                    # nearest from 20 m
        data[b] = data[b][:, :iy.max() + 1, :ix.max() + 1][:, iy][:, :, ix]
    nc = write_nc(tmp_path / "ref.nc", t, y, x, data, wkt)
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(nc),
        reference_period=("2016-01-01", "2025-12-31"), pipeline_overpass=str(tmp_path / "none.json")))
    sub = {b: a[:, 5:25, 4:30] for b, a in data.items()}
    raw = write_raw_v2(tmp_path / "src" / "openEO.zarr", t, y[5:25], x[4:30], sub, wkt)
    z = zip_store(raw, tmp_path / "src" / "openEO.zarr")

    def fake_download(job, req, log, workers=4):
        place(req, [("openEO.zarr.zip", z, "application/zip")])
        return {"assets": {"openEO.zarr.zip": {"href": "https://h/openEO.zarr.zip",
                                               "type": "application/zip"}}}

    monkeypatch.setattr(dcz, "download_results", fake_download)
    ctx, _ = _ctx(FakeConn())
    ctx.network = False
    req = dcz.test_request(tmp_path / "cubes")
    req.start, req.end = "2025-06-01", "2025-08-01"
    msgs = []
    res = dcz.run_request(ctx, req, msgs.append)
    text = "\n".join(msgs)
    assert res["status"] == "done" and res["verdict"] == "identical on the common window"
    for s_ in ("backend output format 'ZARR'", "FORMAT REPORT", "zip extracted", "open_cube(",
               "p1/p50/p99", "SUMMARY", "band naming: B02<-B02", "B11 100.0 %", "SCL 100.0 %",
               "=> normalisable: YES"):
        assert s_ in text, s_
    cube = zc.open_cube("villaviciosa", "2025-06-01", "2025-08-01", root=tmp_path / "cubes" / "_test")
    assert cube.sizes["t"] == 4 and zc.band_names(cube) == list(BANDS)
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
    interp, _, _ = normalise_and_write(req, ref_grid=ref)
    assert any("put on the reference grid" in w for w in interp.warnings)
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
    assert (len(g["y"]), len(g["x"])) == (915, 915) and (g["x"][0], g["y"][0]) == (X0, Y0)
    t, y, x, data = synth(2023, seed=55)
    raw = write_raw_v2(tmp_path / "src" / "s.zarr", t, y, x, data, wkt)
    req = make_request(root)
    place(req, [("s.zarr", raw, "x")])
    interp, _, _ = normalise_and_write(req, ref_grid=g)       # a 24 x 40 delivery, padded
    ds = zc.open_store(req.store)
    assert ds.attrs["grid_reference"].startswith("predicted: ")
    ds.close()
    g2 = dcz.reference_grid("villaviciosa", root=root, log=msgs.append)
    assert g2["kind"] == "store" and g2["path"] == str(req.store) and "weaker" in msgs[-1]
    assert (len(g2["y"]), len(g2["x"])) == (915, 915)
    # the store's CRS is used when a delivery carries none
    shutil.rmtree(raw / "crs")
    for b in BANDS:
        zarr.open_array(store=str(raw / b), mode="r+").attrs.pop("grid_mapping", None)
    req2 = make_request(tmp_path / "c2")
    place(req2, [("s.zarr", raw, "x")])
    _, i2, _ = dcz.normalise(req2, lambda m: None, ref_grid=g2)
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
