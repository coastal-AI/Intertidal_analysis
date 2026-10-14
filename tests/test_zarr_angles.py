# -*- coding: utf-8 -*-
"""The ANGLES product of the Zarr downloader, and the main product unchanged (no network).

Part 1 pins the MAIN product: fixed synthetic deliveries (no random numbers)
in several layouts are normalised and written, and a fingerprint of the
result (decoded values, dtype, no-data, chunks, compressor, band and store
attributes, the interpretation and the format report) is compared with the
one the downloader produced BEFORE the angles product existed. The raw chunk
bytes and the process graph are compared too when the library versions are
those the golden values were made with.

Part 2 tests the angles product: float angle deliveries in the plausible
layouts, the 300 m grid guard, the store format, the checks (dates against
the main store, physical ranges, spread inside the frame, the values at the
frame centre), ``zarr_cubes.open_angles`` / ``angles_at``, and the
``--products`` selection with its skipping logic.
"""
import hashlib
import json
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
from tests.test_download_cube_zarr import (X0, Y0, FakeConn, FakeSaveResult, make_request,
                                           normalise_and_write, place, write_nc, write_raw_v2,
                                           write_raw_v3_bands, zip_store)

BANDS = zc.BANDS
FILL = zc.FILL


@pytest.fixture(scope="module")
def wkt():
    from pyproj import CRS

    return CRS.from_epsg(32630).to_wkt("WKT1_GDAL")


# ─────────────────────────────────────────────────────────────────────────────
#  part 1: the main product is unchanged
# ─────────────────────────────────────────────────────────────────────────────

def det_cube(year, n=6, H=24, W=40, x0=X0, y0=Y0, start=None, seed=0):
    """A synthetic year like ``synth`` but without random numbers (the
    values are a fixed arithmetic pattern), so its fingerprint does not
    depend on numpy's generator."""
    t = pd.date_range(start or f"{year}-01-03", periods=n, freq="17D")
    x = x0 + 10.0 * np.arange(W)
    y = y0 - 10.0 * np.arange(H)
    k, i, j = np.meshgrid(np.arange(n), np.arange(H), np.arange(W), indexing="ij")
    data = {}
    for bi, b in enumerate(BANDS):
        hi = 12 if b == "SCL" else 101 if b == "CLD" else 10000
        a = ((k * 7919 + i * 104729 + j * 1299709 + (bi + 1) * 15485863 + seed * 32452843)
             % hi).astype("int16")
        a[0, :3, :5] = FILL
        a[-1, -2:, :] = FILL
        data[b] = a
    return t, y, x, data


def _canon(obj):
    """``obj`` with the variable lists of the 'layout' notes sorted: zarr
    lists the arrays of a v2 group in no fixed order, so the note (and the
    ``delivered_notes`` attribute) varies from run to run in its order only."""
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"(band-like variables: )\[([^\]]*)\]",
                      lambda m: m.group(1) + "[" + ", ".join(sorted(m.group(2).split(", "))) + "]",
                      obj)
    return obj


def _sha(obj):
    blob = json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _arr_sha(a):
    a = np.asarray(a)
    if a.dtype.kind == "M":
        a = a.astype("datetime64[ns]").astype("int64")
    digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    return f"{a.dtype.str}:{list(a.shape)}:{digest}"


#: global attributes whose values depend on the run, the machine or a library version
RUN_ATTRS = ("processing_date", "software", "delivered", "process_graph")
#: library versions the raw chunk bytes and the process graph were pinned with
VERSION_LIBS = ("zarr", "numcodecs", "numpy", "xarray", "openeo")


def _versions():
    import numcodecs
    import openeo

    return {"zarr": zarr.__version__, "numcodecs": numcodecs.__version__,
            "numpy": np.__version__, "xarray": xr.__version__, "openeo": openeo.__version__}


def main_fingerprint(store, interp, lines, flat):
    """``{"core", "bytes", "graph"}`` hashes of a written main store.

    core: everything the downloader decides (values, dtypes, no-data,
    chunks, compressor, the band attributes it sets, the global attributes
    but the run-dependent ones, the coordinates and their encoding, the
    interpretation, the format report from 'interpretation:' on, without
    the codec lines). bytes: the chunk files and the full Zarr metadata (a
    function of zarr / numcodecs / xarray too). graph: the process graph.
    """
    store = Path(store)
    meta = json.loads((store / ".zmetadata").read_text(encoding="utf-8"))["metadata"]
    gattrs = meta[".zattrs"]
    core = {"global_keys": sorted(gattrs),
            "global": {k: v for k, v in gattrs.items() if k not in RUN_ATTRS},
            "bands": {}, "coords": {}}
    ds = zc.open_store(store)
    for b in zc.band_names(ds):
        za = meta[f"{b}/.zarray"]
        zt = meta[f"{b}/.zattrs"]
        core["bands"][b] = {
            "values": _arr_sha(ds[b].values), "dtype": za["dtype"], "fill_value": za["fill_value"],
            "chunks": za["chunks"], "shape": za["shape"],
            "compressor": {k: za["compressor"].get(k) for k in ("id", "cname", "clevel", "shuffle")},
            "attrs": {k: v for k, v in zt.items() if k not in ("_ARRAY_DIMENSIONS", "coordinates")}}
    for c in ("t", "y", "x"):
        core["coords"][c] = {"values": _arr_sha(ds[c].values), "dtype": meta[f"{c}/.zarray"]["dtype"],
                             "attrs": {k: v for k, v in meta[f"{c}/.zattrs"].items()
                                       if k != "_ARRAY_DIMENSIONS"}}
    core["coords"]["crs"] = {k: v for k, v in meta["crs/.zattrs"].items()
                             if k in ("GeoTransform", "epsg_code", "long_name")}
    ds.close()
    core["interp"] = {"fills": {b: repr(v) for b, v in interp.fills.items()},
                      "fill_sources": interp.fill_sources, "delivered": interp.delivered,
                      "warnings": interp.warnings, "notes": interp.notes,
                      "nodata_lines": interp.nodata_lines, "groups": interp.groups,
                      "crs_source": interp.crs_source}
    k0 = next(i for i, ln in enumerate(lines) if ln.startswith("interpretation:"))
    core["report"] = [ln for ln in lines[k0:] if "codecs" not in ln]
    chunks = {}
    for f in sorted(p for p in store.rglob("*") if p.is_file()):
        parts = f.relative_to(store).parts           # chunk files; metadata is below
        if len(parts) == 2 and parts[0] in BANDS + ("t", "x", "y") and not parts[1].startswith("."):
            chunks["/".join(parts)] = hashlib.sha256(f.read_bytes()).hexdigest()
    full_meta = {k: v for k, v in meta.items() if k != ".zattrs"}
    full_meta[".zattrs"] = {k: v for k, v in gattrs.items() if k not in RUN_ATTRS}
    return {"core": _sha(core), "bytes": _sha({"chunks": chunks, "meta": full_meta}),
            "graph": _sha(flat), "_core": _canon(core)}


def _write_main_case(name, tmp, wkt):
    """Deliver, normalise and write one fixed main-product case; returns
    ``(store, interp, report lines, flat graph)``."""
    src = tmp / "src" / name
    src.mkdir(parents=True, exist_ok=True)
    t, y, x, data = det_cube(2023)
    kw, ref = {}, None
    req = make_request(tmp / "cubes" / name)
    if name == "zip_v2_big_endian":
        raw = write_raw_v2(src / "openEO.zarr", t, y, x, data, wkt)
        place(req, [("openEO.zarr.zip", zip_store(raw, src / "openEO.zarr"), "application/zip")])
    elif name == "v3_bands_dimension":
        raw = write_raw_v3_bands(src / "cube.zarr", t, y, x, data)
        place(req, [("cube.zarr", raw, "x")])
    elif name == "zarr_default_fill":
        data["CLD"][:, 4:8, 4:8] = 0
        raw = write_raw_v2(src / "z.zarr", t, y, x, data, wkt, fill_value=None)
        place(req, [("z.zarr", raw, "x")])
    elif name == "nodata_zero":
        for a in data.values():
            a[a == FILL] = 0
        raw = write_raw_v2(src / "a.zarr", t, y, x, data, wkt, fill_value=None)
        place(req, [("a.zarr", raw, "x")])
        kw["nodata"] = 0
    elif name == "padded_onto_reference":       # padded on one side: needs --allow-padding now
        ref = {"path": "ref.nc", "kind": "netcdf", "x": x[2:36], "y": y[3:20], "crs_wkt": wkt}
        kw["allow_padding"] = True
        sub = {b: a[:, 0:22, 5:40] for b, a in data.items()}
        raw = write_raw_v2(src / "e.zarr", t, y[0:22], x[5:40], sub, wkt)
        place(req, [("e.zarr", raw, "x")])
    elif name == "netcdf":
        nc = write_nc(src / "openEO.nc", t, y, x, data, wkt, bands=BANDS)
        place(req, [("openEO.nc", nc, "application/x-netcdf")])
    elif name == "test_ambiguous_nodata":
        for a in data.values():
            a[a == FILL] = 7
        raw = write_raw_v2(src / "a.zarr", t, y, x, data, wkt, fill_value=None)
        req = make_request(tmp / "cubes" / name, test=True)
        place(req, [("a.zarr", raw, "x")])
    else:
        raise ValueError(name)
    ds, interp, lines = dcz.normalise(req, lambda m: None, job_info={"id": "j-test", "title": "t"},
                                      results_meta={}, ref_grid=ref, **kw)
    flat = dcz.build_process(req).flat_graph()
    attrs = dcz.provenance(req, {"id": "j-test", "title": "t"}, {}, interp, dcz.Context(), flat,
                           ref_grid=ref)
    dcz.write_store(ds, interp.fills, req.store, attrs, log=lambda m: None, groups=interp.groups)
    return req.store, interp, lines, flat


MAIN_CASES = ("zip_v2_big_endian", "v3_bands_dimension", "zarr_default_fill", "nodata_zero",
              "padded_onto_reference", "netcdf", "test_ambiguous_nodata")

#: fingerprints of the main product made by the downloader of commit 57669c1
#: (before the angles product), with these library versions. One intended
#: change since: "padded_onto_reference" crops a margin that holds data, and
#: its report and ``delivered_notes`` now say so (the 'cropped-away margin'
#: line of ``margin_summary``, a note: the margin lies outside the frame);
#: it is padded on one side, which a production year now refuses, so it is
#: written with ``allow_padding`` (--allow-padding), whose warning is the
#: 57669c1 one. With ``margin_summary`` returning None its core and bytes
#: are still the 57669c1 ones (62179b9f..., ca75e004...)
MAIN_GOLDEN_VERSIONS = {"zarr": "3.2.1", "numcodecs": "0.16.5", "numpy": "2.4.6",
                        "xarray": "2025.1.1", "openeo": "0.50.0"}
_GRAPH = "208a9557f1c6566a017431cc1be74ff0f9905437db96bf794537485addbf5ba3"
MAIN_GOLDEN = {
    "zip_v2_big_endian": {
        "core": "be911025454425916391e0f1fb8d4bc26d167b8bc007a3bc5f5934317b1c38a7",
        "bytes": "be92744d2c9626ada438dc527042546749eb63c780a0b970c30e521f1c56a337", "graph": _GRAPH},
    "v3_bands_dimension": {
        "core": "3b5656cc589d7bbefee402ccd3487fbe9c479a47f0206da96358dc5457a33e25",
        "bytes": "7434a0c0490d1a650fec45df91d1b517d7ab89814018447ce6ef7caa21f69ee8", "graph": _GRAPH},
    "zarr_default_fill": {
        "core": "986d2644f1ee4ae37f843f2a76af5316293df748fabcf7ec32c37846e5f0eeb3",
        "bytes": "7f3896f9a0f64470156237f63796702bf13a9c39c1d003239e3fa578984b7ba6", "graph": _GRAPH},
    "nodata_zero": {
        "core": "20326e9b0098416f818d1c24cc1ace65116e745862e305f0ebf3ca09deea7f96",
        "bytes": "b75e8269f72ca60322ff9778b05c55b0cdf5f75dbc20f9226bd9168db096d2d4", "graph": _GRAPH},
    "padded_onto_reference": {
        "core": "aac5a7ba4dda3c37fa813fa849e289f2315e202019e4e051233839d7323e05fd",
        "bytes": "7b785e08fef2bc4995452188fe62ead83f947c8b8ac4e17e68dafde23693f486", "graph": _GRAPH},
    "netcdf": {
        "core": "c4d19bf140ba6fa420594d9abe7f04965f46b000fa19cf537abc5be839812f5b",
        "bytes": "859c307ef6e120aa1678decbda36bc28098ebbc56ef6b222647d104f0865812c", "graph": _GRAPH},
    "test_ambiguous_nodata": {
        "core": "160287c13959aea14ae15a3849394c9c7c9f7c4b640fc2bab086d068ac1204c8",
        "bytes": "3c83b4b21b93cc1e777c21736ecec05cdf9fde7ea4e7f6b19adbbee3e1c582a1", "graph": _GRAPH},
}


@pytest.mark.parametrize("case", MAIN_CASES)
def test_main_product_is_unchanged(case, tmp_path, wkt):
    """The main product, written by the generalised code, is the one the
    downloader wrote before the angles product existed: same values, dtype,
    no-data, chunks, compressor, attributes, interpretation and report (and,
    with the library versions of the golden values, the same chunk bytes and
    process graph)."""
    store, interp, lines, flat = _write_main_case(case, tmp_path, wkt)
    fp = main_fingerprint(store, interp, lines, flat)
    want = MAIN_GOLDEN[case]
    assert fp["core"] == want["core"], json.dumps(fp["_core"], indent=1, default=str)[:6000]
    if all(_versions()[k] == MAIN_GOLDEN_VERSIONS[k] for k in VERSION_LIBS):
        assert fp["bytes"] == want["bytes"]
        assert fp["graph"] == want["graph"]
    ds = zc.open_store(store)
    assert all(ds[b].dtype == np.int16 for b in zc.band_names(ds))
    assert ds.attrs["format"] == zc.FORMAT_ID and "product" not in ds.attrs
    ds.close()


def test_main_process_and_paths_unchanged(tmp_path):
    """The main job's graph, title, store, part and sidecar names are those
    of before; the graph is the one the old code built."""
    from openeo.rest.datacube import DataCube

    req = make_request(tmp_path / "cubes", year=2024)
    assert req.product == "main" and req.bands == BANDS and req.tag == "villaviciosa 2024"
    assert req.store.name == "2024.zarr" and Path(req.part).name == "2024.part"
    assert req.sidecar("verify.json").name == "2024.verify.json"
    old = (DataCube.load_collection("SENTINEL2_L2A", connection=None,
                                    spatial_extent=dict(req.bbox),
                                    temporal_extent=[req.start, req.end], bands=list(BANDS),
                                    max_cloud_cover=100)
           .resample_spatial(resolution=10, method="near").save_result(format="Zarr")).flat_graph()
    flat = dcz.build_process(req).flat_graph()
    assert flat == old
    assert dcz.job_title(req, dcz.graph_hash(flat)) == \
        f"zarr_cube_villaviciosa_2024_{dcz.graph_hash(old)[:8]}"
    treq = dcz.test_request(tmp_path / "cubes")
    assert treq.tag == "villaviciosa 2025 TEST" and treq.product == "main"
    assert dcz.job_title(treq, "abcdef0123456").startswith("zarr_cube_test_villaviciosa_2025-06-01_")


# ─────────────────────────────────────────────────────────────────────────────
#  part 2: the angles product
# ─────────────────────────────────────────────────────────────────────────────

ANGLES = zc.ANGLE_BANDS
RES = zc.ANGLES_RES_M
#: cell centres of the --test angle frame (the 2 km box of Villaviciosa: 8 x 7 cells)
AX0, AY0 = 305550.0, 4822350.0
#: per angle: value at (AX0, AY0) on date 0, change per date, d/dx, d/dy and
#: d2/dxdy per km: a bilinear field, which bilinear interpolation reproduces
FIELD = {"sunZenithAngles": (24.0, 1.5, 0.004, -0.009, 1e-4),
         "sunAzimuthAngles": (151.0, 0.7, 0.012, 0.003, -2e-4),
         "viewZenithMean": (6.0, 0.2, -0.05, 0.01, 5e-4),
         "viewAzimuthMean": (104.0, 30.0, 0.3, -0.2, 1e-3)}


def field_values(b, n, x, y, x0=AX0, y0=AY0):
    """The synthetic field of angle ``b`` on ``n`` dates, on the grid of
    columns ``x`` by rows ``y`` (float64, degrees)."""
    c0, ct, cx, cy, cxy = FIELD[b]
    k = np.arange(n, dtype="float64")[:, None, None]
    X = (np.atleast_1d(np.asarray(x, dtype="float64")) - x0)[None, None, :] / 1000.0
    Y = (np.atleast_1d(np.asarray(y, dtype="float64")) - y0)[None, :, None] / 1000.0
    v = c0 + ct * k + cx * X + cy * Y + cxy * X * Y
    return np.mod(v, 360.0) if b in zc.AZIMUTH_BANDS else v


def angle_cube(year, n=4, H=8, W=7, x0=AX0, y0=AY0, start=None, freq="5D"):
    """Synthetic angles: dates, 300 m axes and float32 bilinear fields."""
    t = pd.date_range(start or f"{year}-06-02", periods=n, freq=freq)
    x = x0 + float(RES) * np.arange(W)
    y = y0 - float(RES) * np.arange(H)
    return t, y, x, {b: field_values(b, n, x, y).astype("float32") for b in ANGLES}


def _days(t):
    return (pd.DatetimeIndex(t) - pd.Timestamp("1990-01-01")).days.values.astype("int32")


def write_angles_v2(path, t, y, x, data, wkt, *, dtype=">f4", fill_value=float("nan"),
                    names=None, attrs=None, time_dim=True):
    """A Zarr v2 store, one array per angle (``dtype``; ``fill_value`` None:
    zarr's default), the time in days, a ``crs`` variable. ``time_dim``
    False: one date, (y, x) arrays (a store per date)."""
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    dims = ["t", "y", "x"] if time_dim else ["y", "x"]
    for b, a in data.items():
        kw = {} if fill_value is None else {"fill_value": fill_value}
        arr = g.create_array((names or {}).get(b, b), shape=a.shape, dtype=dtype,
                             chunks=(2, 4, 4)[-a.ndim:], **kw)
        arr[:] = a.astype(np.dtype(dtype).newbyteorder("="))
        arr.attrs.update(dict({"_ARRAY_DIMENSIONS": dims, "grid_mapping": "crs"},
                              **(attrs or {}).get(b, {})))
    coords = [("x", np.asarray(x), {}), ("y", np.asarray(y), {})]
    if time_dim:
        coords.insert(0, ("t", _days(t), {"units": "days since 1990-01-01"}))
    for name, vals, at in coords:
        arr = g.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape)
        arr[:] = vals
        arr.attrs.update(dict(at, _ARRAY_DIMENSIONS=[name]))
    c = g.create_array("crs", shape=(), dtype="int32")
    c.attrs.update({"crs_wkt": wkt, "_ARRAY_DIMENSIONS": []})
    return Path(path)


def write_angles_v3_bands(path, t, y, x, data, labels=ANGLES):
    """Zarr v3, one (time, bands, y, x) float32 array with band labels as
    given (aliases allowed), the CRS as an EPSG root attribute."""
    g = zarr.open_group(str(path), mode="w", zarr_format=3, attributes={"crs": "EPSG:32630"})
    cube = np.stack([data[b] for b in ANGLES], axis=1)
    arr = g.create_array("cube", shape=cube.shape, dtype="float32", chunks=(2, 4, 4, 4),
                         fill_value=float("nan"), dimension_names=("time", "bands", "y", "x"))
    arr[:] = cube
    lab = g.create_array("bands", shape=(len(labels),), dtype=str, dimension_names=("bands",))
    lab[:] = np.array(labels)
    for name, vals in (("time", _days(t)), ("x", np.asarray(x)), ("y", np.asarray(y))):
        a = g.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape,
                           dimension_names=(name,),
                           attributes={"units": "days since 1990-01-01"} if name == "time" else {})
        a[:] = vals
    return Path(path)


def write_angles_nc(path, t, y, x, data, wkt):
    """A float32 netCDF with NaN as _FillValue (the fallback format)."""
    ds = xr.Dataset({b: (("t", "y", "x"), data[b], {"grid_mapping": "crs"}) for b in data},
                    coords={"t": pd.DatetimeIndex(t), "y": y, "x": x})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": wkt, "spatial_ref": wkt})
    enc = {b: {"dtype": "float32", "_FillValue": np.float32(np.nan)} for b in data}
    enc["t"] = {"units": "days since 1990-01-01", "dtype": "int32"}
    ds.to_netcdf(path, encoding=enc)
    return Path(path)


def angles_request(root, year=2025, test=False, start=None, end=None):
    s, e = zc.year_window(year)
    return dcz.Request(site="villaviciosa", year=year, start=start or s, end=end or e,
                       bbox=dict(dcz.SITE_CONFIG["villaviciosa"]["bbox"]), root=Path(root),
                       test=test, product="angles")


def write_angles_store(req, ref_grid=None, **kw):
    """normalise -> provenance -> write_store, as run_request does;
    returns ``(interpretation, report lines)``."""
    ds, interp, rep = dcz.normalise(req, lambda m: None, job_info={"id": "j-ang", "title": "a"},
                                    results_meta={}, ref_grid=ref_grid, **kw)
    flat = dcz.build_process(req).flat_graph()
    attrs = dcz.provenance(req, {"id": "j-ang", "title": "a"}, {}, interp, dcz.Context(), flat,
                           ref_grid=ref_grid)
    dcz.write_store(ds, dcz.store_fills(interp, req.spec), req.store, attrs, log=lambda m: None,
                    groups=interp.groups, chunks=req.spec.chunks, format_id=req.spec.format_id)
    return interp, rep


def check_angles_store(store, t, y, x, want, nan_mask=None):
    """A complete angles store: float32 degrees, NaN no-data, the values
    ``want`` (float32), the layout and provenance of the format."""
    assert zc.is_complete(store, zc.ANGLES_FORMAT_ID) and not zc.is_complete(store)
    ds = zc.open_store(store)
    assert list(ds["t"].values) == list(pd.DatetimeIndex(t).values)
    np.testing.assert_array_equal(ds["x"].values, x)
    np.testing.assert_array_equal(ds["y"].values, y)
    assert zc.band_names(ds) == list(ANGLES)
    for b in ANGLES:
        assert ds[b].dtype == np.float32 and ds[b].dims == ("t", "y", "x")
        assert zc.fill_of(ds[b]) is None                     # NaN
        w = np.asarray(want[b], dtype="float32").copy()
        if nan_mask is not None:
            w[nan_mask] = np.nan
        np.testing.assert_array_equal(ds[b].values, w)      # NaN == NaN here
        assert ds[b].attrs["units"] == "degree"
    meta = json.loads((Path(store) / "sunZenithAngles" / ".zarray").read_text())
    assert meta["dtype"] == "<f4" and meta["fill_value"] == "NaN"
    assert meta["compressor"]["cname"] == "zstd" and meta["compressor"]["clevel"] == 5
    assert meta["chunks"] == list(zc.chunk_shape(ds["sunZenithAngles"].shape, zc.ANGLES_CHUNKS))
    assert (Path(store) / ".zmetadata").is_file()
    a = ds.attrs
    assert a["format"] == zc.ANGLES_FORMAT_ID and a["product"] == "angles" and a["job_id"] == "j-ang"
    assert json.loads(a["bands"]) == list(ANGLES) and a["grid_resolution_m"] == RES
    assert "bilinear" in a["resampling"] and json.loads(a["chunks"]) == zc.ANGLES_CHUNKS
    assert a["units"] == "degree" and "radiometry" not in a
    assert ds["sunZenithAngles"].attrs["standard_name"] == "solar_zenith_angle"
    assert zc.grid_info(ds)["epsg"] == 32630 and zc.grid_info(ds)["dx"] == RES
    ds.close()


# ─────────────────────────────────────────────────────────────────────────────
#  normalisation of a float angles delivery in the plausible layouts
# ─────────────────────────────────────────────────────────────────────────────

ANGLE_LAYOUTS = ("zip_v2_big_endian_nan", "v3_bands_with_aliases", "zarr_default_fill_zero",
                 "float64_declared_nodata", "netcdf", "store_per_date", "split_by_band",
                 "packed_int16_cf")


@pytest.mark.parametrize("layout", ANGLE_LAYOUTS)
def test_angles_normalised_from_plausible_layouts(layout, tmp_path, wkt):
    t, y, x, data = angle_cube(2025)
    src = tmp_path / "src"
    src.mkdir()
    req = angles_request(tmp_path / "cubes")
    want = {b: a.copy() for b, a in data.items()}
    hole = np.zeros(data[ANGLES[0]].shape, bool)
    hole[1, :2, :3] = True                                  # no data in a corner of one date
    if layout == "zip_v2_big_endian_nan":
        for a in data.values():
            a[hole] = np.nan
        raw = write_angles_v2(src / "openEO.zarr", t, y, x, data, wkt)
        place(req, [("openEO.zarr.zip", zip_store(raw, src / "openEO.zarr"), "application/zip")])
    elif layout == "v3_bands_with_aliases":
        for a in data.values():
            a[hole] = np.nan
        raw = write_angles_v3_bands(src / "cube.zarr", t, y, x, data,
                                    labels=("SZA", "SAA", "approximateViewZenith", "VAA"))
        place(req, [("cube.zarr", raw, "x")])
    elif layout == "zarr_default_fill_zero":
        for a in data.values():
            a[hole] = 0.0                                   # what an unwritten chunk reads as
        raw = write_angles_v2(src / "z.zarr", t, y, x, data, wkt, dtype="<f4", fill_value=None)
        assert json.loads((raw / "sunZenithAngles" / ".zarray").read_text())["fill_value"] == 0.0
        place(req, [("z.zarr", raw, "x")])
    elif layout == "float64_declared_nodata":
        for a in data.values():
            a[hole] = -9999.0
        data = {b: a.astype("float64") for b, a in data.items()}
        raw = write_angles_v2(src / "d.zarr", t, y, x, data, wkt, dtype="<f8", fill_value=0.0,
                              attrs={b: {"nodata": -9999.0} for b in ANGLES})
        place(req, [("d.zarr", raw, "x")])
    elif layout == "netcdf":
        for a in data.values():
            a[hole] = np.nan
        place(req, [("openEO.nc", write_angles_nc(src / "openEO.nc", t, y, x, data, wkt),
                     "application/x-netcdf")])
    elif layout == "store_per_date":
        for a in data.values():
            a[hole] = np.nan
        items = []
        for k, d in enumerate(t):
            p = write_angles_v2(src / f"openEO_{d.date()}Z.zarr", t, y, x,
                                {b: a[k] for b, a in data.items()}, wkt, dtype="<f4",
                                time_dim=False)
            items.append((p.name, p, "x"))
        place(req, items[::-1])
    elif layout == "split_by_band":
        for a in data.values():
            a[hole] = np.nan
        sun = write_angles_v2(src / "sun.zarr", t, y, x, {b: data[b] for b in ANGLES[:2]}, wkt)
        view = write_angles_v2(src / "view.zarr", t, y, x, {b: data[b] for b in ANGLES[2:]}, wkt)
        place(req, [("sun.zarr", sun, "x"), ("view.zarr", view, "x")])
    elif layout == "packed_int16_cf":
        packed = {b: np.round(a.astype("float64") * 100.0).astype("int16") for b, a in data.items()}
        for a in packed.values():
            a[hole] = FILL
        raw = write_angles_v2(src / "p.zarr", t, y, x, packed, wkt, dtype="<i2", fill_value=FILL,
                              attrs={b: {"scale_factor": 0.01, "add_offset": 0.0} for b in ANGLES})
        place(req, [("p.zarr", raw, "x")])
        want = {b: (a.astype("float64") * 0.01).astype("float32") for b, a in packed.items()}
    interp, rep = write_angles_store(req)
    check_angles_store(req.store, t, y, x, want, nan_mask=hole)
    text = "\n".join(rep)
    for s_ in ("FORMAT REPORT villaviciosa 2025 angles", "resample_spatial(300 m, 'bilinear')",
               "values on", "angles check:", "at the frame centre", "dates over the bound: 0",
               "=> normalisable: YES"):
        assert s_ in text, s_
    assert req.sidecar("format.txt").name == "2025.angles.format.txt"
    assert req.sidecar("format.txt").is_file()
    if layout == "v3_bands_with_aliases":
        assert interp.delivered["viewZenithMean"] == "cube[bands=approximateViewZenith]"
    if layout == "zarr_default_fill_zero":
        assert interp.fills["sunZenithAngles"] == 0.0
        assert interp.fill_sources["sunZenithAngles"].startswith("Zarr fill_value 0.0 (angles")
    if layout == "float64_declared_nodata":
        assert interp.fills["viewAzimuthMean"] == -9999.0
        assert any("not float32: converted" in w for w in interp.warnings)
        ds = zc.open_store(req.store)
        assert ds["sunZenithAngles"].attrs["delivered_dtype"] == "float64"
        assert ds["sunZenithAngles"].attrs["delivered_nodata"] == "-9999.0"
        ds.close()
    if layout == "packed_int16_cf":
        assert any("DECODED (CF packing" in w for w in interp.warnings)
        ds = zc.open_store(req.store)
        assert ds["sunAzimuthAngles"].attrs["applied_scale_factor"] == 0.01
        ds.close()


def test_angles_out_of_physical_range_stop_a_year_not_the_test(tmp_path, wkt):
    """Sun angles that are not degrees (here x100 without a scale attribute)
    stop a production year before anything is written; --test reports them."""
    t, y, x, data = angle_cube(2025)
    scaled = {b: a * 100.0 for b, a in data.items()}
    raw = write_angles_v2(tmp_path / "src" / "s.zarr", t, y, x, scaled, wkt)
    req = angles_request(tmp_path / "c1")
    place(req, [("s.zarr", raw, "x")])
    with pytest.raises(dcz.FormatError, match="sun angles outside their physical range"):
        dcz.normalise(req, lambda m: None)
    assert not req.store.exists()
    assert "NOT NORMALISABLE" in req.sidecar("format.txt").read_text(encoding="utf-8")
    treq = angles_request(tmp_path / "c2", test=True)
    place(treq, [("s.zarr", raw, "x")])
    _, interp, lines = dcz.normalise(treq, lambda m: None)
    assert any("sun angles outside their physical range" in w for w in interp.warnings)
    assert any(ln.strip().startswith("sunZenithAngles") and "outside [0, 90]: 224" in ln
               for ln in lines)


# ─────────────────────────────────────────────────────────────────────────────
#  the 300 m grid: same CRS, cell edges on the 10 m lattice, the frame around it
# ─────────────────────────────────────────────────────────────────────────────

def test_angles_grid_guard(tmp_path, wkt, monkeypatch):
    from pyproj import CRS

    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(tmp_path / "absent.nc")))
    root = tmp_path / "cubes"
    msgs = []
    ref = dcz.angles_reference_grid("villaviciosa", root=root, log=msgs.append)
    # the 10 m frame (915 x 915 px from 301020 / 4824840) snapped outward to 300 m
    assert ref["kind"] == "predicted" and "snapped outward to the 300 m lattice" in msgs[-1]
    assert (len(ref["y"]), len(ref["x"])) == (31, 31)
    assert (ref["x"][0], ref["y"][0]) == (301050.0, 4824750.0)
    g300 = dcz.predict_grid(dcz.SITE_CONFIG["villaviciosa"]["bbox"], res=RES)
    assert g300["shape"] == [31, 31] and g300["x_edges"][0] == ref["x"][0] - RES / 2
    for axis in (ref["x"], ref["y"]):                     # cell edges on the 10 m lattice
        assert np.all(np.mod(axis - RES / 2, 10.0) == 0)
    fe = ref["fine_edges"]
    assert ref["x"][0] - RES / 2 <= fe[0] < fe[1] <= ref["x"][-1] + RES / 2
    assert ref["y"][-1] - RES / 2 <= fe[2] < fe[3] <= ref["y"][0] + RES / 2
    # on that lattice, a smaller extent: a production year stops (NaN inside the frame
    # would read as no data); --allow-padding puts it on the reference grid, padded with NaN
    t, y, x, data = angle_cube(2025, x0=ref["x"][3], y0=ref["y"][2])
    raw_a = write_angles_v2(tmp_path / "src" / "a.zarr", t, y, x, data, wkt)
    req = angles_request(root)
    place(req, [("a.zarr", raw_a, "x")])
    with pytest.raises(dcz.FormatError, match="does not cover the whole reference grid"):
        dcz.normalise(req, lambda m: None, ref_grid=ref)
    interp, _ = write_angles_store(req, ref_grid=ref, allow_padding=True)
    assert any("same 300 m lattice" in w and "another extent" in w for w in interp.warnings)
    ds = zc.open_store(req.store)
    np.testing.assert_array_equal(ds["x"].values, ref["x"])
    np.testing.assert_array_equal(ds["y"].values, ref["y"])
    got = ds["sunZenithAngles"].values
    np.testing.assert_array_equal(got[:, 2:10, 3:10], data["sunZenithAngles"])
    assert np.isnan(got[:, :2, :]).all() and np.isnan(got[:, :, :3]).all()
    ds.close()
    # the --test window stays as delivered
    treq = angles_request(tmp_path / "ct", test=True)
    place(treq, [("a.zarr", raw_a, "x")])
    ds_t, interp_t, _ = dcz.normalise(treq, lambda m: None, ref_grid=ref)
    assert ds_t.sizes["x"] == 7 and not any("another extent" in w for w in interp_t.warnings)
    # another phase of the 10 m lattice: kept as delivered against the prediction ...
    t, y, x, data = angle_cube(2024, x0=ref["x"][3] + 150.0, y0=ref["y"][2] - 60.0)
    raw_b = write_angles_v2(tmp_path / "src" / "b.zarr", t, y, x, data, wkt)
    req_b = angles_request(tmp_path / "cb", year=2024)
    place(req_b, [("b.zarr", raw_b, "x")])
    _, interp_b, _ = dcz.normalise(req_b, lambda m: None, ref_grid=ref)
    assert any("not at multiples of 300 m" in w for w in interp_b.warnings)
    assert any("does not cover the 10 m frame" in w for w in interp_b.warnings)
    np.testing.assert_array_equal(interp_b.x, x)
    # ... refused once a store of the site fixes the grid
    ref2 = dcz.angles_reference_grid("villaviciosa", root=root, log=msgs.append)
    assert ref2["kind"] == "store" and "complete angles store" in msgs[-1]
    with pytest.raises(dcz.FormatError, match="another phase of the 10 m lattice than the site's"):
        dcz.normalise(req_b, lambda m: None, ref_grid=ref2)
    # cell edges off the 10 m lattice: refused (unless --allow-grid-offset)
    t, y, x, data = angle_cube(2024, x0=ref["x"][3] + 5.0, y0=ref["y"][2])
    raw_c = write_angles_v2(tmp_path / "src" / "c.zarr", t, y, x, data, wkt)
    req_c = angles_request(tmp_path / "cc", year=2024)
    place(req_c, [("c.zarr", raw_c, "x")])
    with pytest.raises(dcz.FormatError, match="not lines of the 10 m lattice"):
        dcz.normalise(req_c, lambda m: None, ref_grid=ref)
    _, interp_c, _ = dcz.normalise(req_c, lambda m: None, ref_grid=ref, allow_offset=True)
    assert any("--allow-grid-offset" in w for w in interp_c.warnings)
    # another CRS
    ref31 = dict(ref, crs_wkt=CRS.from_epsg(32631).to_wkt("WKT1_GDAL"))
    with pytest.raises(dcz.FormatError, match="is not the CRS"):
        dcz.normalise(req, lambda m: None, ref_grid=ref31)
    # a 10 m delivery is not the 300 m grid
    t10 = pd.date_range("2024-06-02", periods=2, freq="5D")
    x10, y10 = X0 + 10.0 * np.arange(6), Y0 - 10.0 * np.arange(5)
    d10 = {b: np.full((2, 5, 6), 30.0, "float32") for b in ANGLES}
    raw_d = write_angles_v2(tmp_path / "src" / "d.zarr", t10, y10, x10, d10, wkt)
    req_d = angles_request(tmp_path / "cd", year=2024)
    place(req_d, [("d.zarr", raw_d, "x")])
    with pytest.raises(dcz.FormatError, match="not the 300 m grid"):
        dcz.normalise(req_d, lambda m: None)


# ─────────────────────────────────────────────────────────────────────────────
#  the checks: dates against the main store, ranges, spread, frame centre
# ─────────────────────────────────────────────────────────────────────────────

def test_angles_verification(tmp_path, wkt):
    root = tmp_path / "cubes"
    t, y, x, data = det_cube(2025, n=6, start="2025-06-02")
    mreq = make_request(root, year=2025)
    place(mreq, [("a.zarr", write_raw_v2(tmp_path / "m" / "a.zarr", t, y, x, data, wkt), "x")])
    normalise_and_write(mreq)
    # the angles: the main dates but the last, and one date the main store lacks
    ta = pd.DatetimeIndex(list(t[:-1]) + [pd.Timestamp("2025-09-30")])
    n = len(ta)
    _, ay, ax, _ = angle_cube(2025)
    adata = {b: field_values(b, n, ax, ay).astype("float32") for b in ANGLES}
    adata["sunAzimuthAngles"][2, :, 4:] += 4.0             # date 2: two datatakes in the frame
    adata["sunZenithAngles"][3] = np.nan                    # date 3: no sun angles
    adata["sunAzimuthAngles"][3] = np.nan
    areq = angles_request(root)
    place(areq, [("ang.zarr", write_angles_v2(tmp_path / "a" / "ang.zarr", ta, ay, ax, adata, wkt),
                  "x")])
    write_angles_store(areq)
    msgs = []
    rep = dcz.verify_angles(areq, msgs.append)
    d = rep["dates"]
    assert d["compared"] and d["angles"] == 6 and d["main"] == 6 and d["common"] == 5
    assert d["only_angles"] == ["2025-09-30"] and d["only_main"] == [str(t[-1].date())]
    assert all(r["out_of_range"] == 0 for r in rep["ranges"].values())
    assert rep["ranges"]["sunZenithAngles"]["range"] == [0.0, 90.0]
    assert list(rep["dates_over_bound"]) == [str(t[2].date())]
    spread, bound = rep["dates_over_bound"][str(t[2].date())]["sunAzimuthAngles"]
    assert spread > 4.0 > bound > rep["spread_bound"]["diagonal_deg"]
    assert rep["dates_without_sun"] == [str(t[3].date())]
    assert rep["spread"]["sunZenithAngles"]["max"] < rep["spread_bound"]["diagonal_deg"] + 0.1
    assert len(rep["problems"]) == 3 and rep["verdict"].startswith("dates differ from the main store")
    # the angles at the frame centre: the (bilinear) field there
    cx, cy = rep["centre"]["x"], rep["centre"]["y"]
    assert (cx, cy) == ((ax[0] + ax[-1]) / 2, (ay[0] + ay[-1]) / 2)
    assert rep["centre"]["lon"] == pytest.approx(-5.395, abs=0.01)
    v0 = rep["centre"]["per_date"][str(t[0].date())]
    for b in ANGLES:
        assert v0[b] == pytest.approx(field_values(b, n, cx, cy)[0, 0, 0], abs=2e-4)
    assert rep["centre"]["per_date"][str(t[3].date())]["sunZenithAngles"] is None
    assert json.loads(areq.sidecar("verify.json").read_text())["verdict"] == rep["verdict"]
    text = "\n".join(msgs)
    for s_ in ("angles verification: PROBLEMS: dates differ", "only in the angles ['2025-09-30']",
               "dates over the bound: 1", "WITHOUT sun angles", "at the frame centre (lon -5.39",
               "sunZenith  sunAzimuth  viewZenith  viewAzimuth"):
        assert s_ in text, s_
    # no complete main store: dates not compared, which is not a problem by itself
    shutil.rmtree(mreq.store)
    rep2 = dcz.verify_angles(areq, lambda m: None)
    assert rep2["dates"]["compared"] is False
    assert not any("dates differ" in p for p in rep2["problems"]) and len(rep2["problems"]) == 2


def test_sun_angles_written_as_zero_are_flagged(tmp_path, wkt):
    """A no-data written as 0 without being declared (the store's fill is
    NaN): kept as delivered, reported by the format report and the check."""
    t, y, x, data = angle_cube(2025)
    data["sunZenithAngles"][2, :3, :] = 0.0
    data["sunAzimuthAngles"][2, :3, :] = 0.0
    req = angles_request(tmp_path / "cubes", test=True)
    place(req, [("z.zarr", write_angles_v2(tmp_path / "src" / "z.zarr", t, y, x, data, wkt), "x")])
    interp, rep = write_angles_store(req)
    assert any("sun angles exactly 0" in w for w in interp.warnings)
    assert any("exactly 0: 21" in ln for ln in rep)
    assert any("CLD" not in ln and "0.0 x 21" in ln for ln in rep)       # the --test no-data counts
    chk = dcz.verify_angles(req, lambda m: None)
    assert any(p.startswith("sun angles exactly 0") for p in chk["problems"])
    assert chk["ranges"]["sunZenithAngles"]["zeros"] == 21


def test_the_spread_bound_is_geometric():
    """The bound grows with the frame: a 2 km frame allows ~0.13 deg of sun
    zenith, the Ems frame (~47 km diagonal) ~0.5 deg."""
    x = 300150.0 + 300.0 * np.arange(114)
    y = 5930250.0 - 300.0 * np.arange(108)
    t = pd.date_range("2024-06-01", periods=1)
    ds = xr.Dataset({"sunZenithAngles": (("t", "y", "x"), np.full((1, 108, 114), 35.0, "float32"))},
                    coords={"t": t, "y": y, "x": x})
    from pyproj import CRS

    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": CRS.from_epsg(32632).to_wkt("WKT1_GDAL")})
    chk = dcz.angle_checks(ds)
    assert 0.38 < chk["spread_bound"]["diagonal_deg"] < 0.45
    assert chk["dates_over_bound"] == {} and chk["spread"]["sunZenithAngles"]["max"] == 0.0
    ds["sunZenithAngles"][0, :, 60:] = 35.6                 # a step larger than the frame allows
    assert list(dcz.angle_checks(ds)["dates_over_bound"]) == ["2024-06-01"]


# ─────────────────────────────────────────────────────────────────────────────
#  zarr_cubes.open_angles / angles_at
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def angle_years(tmp_path, wkt):
    """Angles stores for 2024 and 2025 (bilinear fields on the --test frame);
    in 2025 the view azimuth of date 0 crosses 0/360 between columns 3 and
    4, and one sun zenith cell of date 1 has no data."""
    root = tmp_path / "cubes"
    for year in (2024, 2025):
        t, y, x, data = angle_cube(year)
        if year == 2025:
            data["viewAzimuthMean"][0] = np.where(np.arange(len(x)) <= 3, 359.8, 0.2)[None, :]
            data["sunZenithAngles"][1, 3, 3] = np.nan
        raw = write_angles_v2(tmp_path / f"s{year}" / "a.zarr", t, y, x, data, wkt)
        req = angles_request(root, year=year)
        place(req, [("a.zarr", raw, "x")])
        write_angles_store(req)
    return root, x, y


def _lonlat(px, py):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True).transform(px, py)


def test_open_angles_across_years(angle_years):
    root, x, y = angle_years
    assert zc.list_angle_years("villaviciosa", root) == [2024, 2025]
    assert zc.list_years("villaviciosa", root) == [] and zc.available(root) == {}
    ds = zc.open_angles("villaviciosa", "2024-06-05", "2025-06-10", root=root)
    assert zc.dates(ds) == ["2024-06-07", "2024-06-12", "2024-06-17", "2025-06-02", "2025-06-07"]
    assert ds.attrs["years"] == [2024, 2025] and ds.attrs["format"] == zc.ANGLES_FORMAT_ID
    assert zc.band_names(ds) == list(ANGLES) and all(ds[b].dtype == np.float32 for b in ANGLES)
    np.testing.assert_array_equal(ds["x"].values, x)
    one = zc.open_angles("villaviciosa", 2025, 2026, root=root, bands=["sunZenithAngles"])
    assert one.attrs["years"] == [2025] and zc.band_names(one) == ["sunZenithAngles"]
    with pytest.raises(FileNotFoundError, match=r"needs the angles of years \[2026\]"):
        zc.open_angles("villaviciosa", "2025-06-01", "2026-02-01", root=root)
    part = zc.open_angles("villaviciosa", "2025-06-01", "2026-02-01", root=root, missing="skip")
    assert part.attrs["missing_years"] == [2026] and len(part["t"]) == 4
    with pytest.raises(FileNotFoundError, match="no complete angles store for 'ferrol'"):
        zc.open_angles("ferrol", 2025, 2026, root=root)
    with pytest.raises(FileNotFoundError, match="no complete Zarr store"):
        zc.open_cube("villaviciosa", 2025, 2026, root=root)            # angles are not cubes


def test_angles_at_is_bilinear(angle_years):
    root, x, y = angle_years
    px, py = AX0 + 777.0, AY0 - 1234.0
    lon, lat = _lonlat(px, py)
    df = zc.angles_at("villaviciosa", lon, lat, "2024-01-01", "2025-01-01", root=root)
    assert list(df.columns) == list(ANGLES) and df.index.name == "date" and len(df) == 4
    assert isinstance(df.index, pd.DatetimeIndex) and str(df.index[0].date()) == "2024-06-02"
    assert df.attrs["x"] == pytest.approx(px, abs=1e-3) and df.attrs["y"] == pytest.approx(py, abs=1e-3)
    assert df.attrs["epsg"] == 32630 and df.attrs["years"] == [2024]
    for b in ANGLES:                           # a bilinear field is reproduced exactly
        np.testing.assert_allclose(df[b].values, field_values(b, 4, px, py)[:, 0, 0], atol=2e-4)
    # between the outermost centres and the frame edge the edge value is held
    ex = zc.angles_at("villaviciosa", *_lonlat(x[0] - 100.0, py), 2024, 2025, root=root)
    np.testing.assert_allclose(ex["sunZenithAngles"].values,
                               field_values("sunZenithAngles", 4, x[0], py)[:, 0, 0], atol=2e-4)
    with pytest.raises(ValueError, match="outside the frame"):
        zc.angles_at("villaviciosa", *_lonlat(x[0] - 200.0, py), 2024, 2025, root=root)
    # 2025: an azimuth across 0/360 is interpolated as an angle; a missing cell is left out
    mid = zc.angles_at("villaviciosa", *_lonlat((x[3] + x[4]) / 2, (y[3] + y[4]) / 2), 2025, 2026,
                       root=root)
    va = mid["viewAzimuthMean"].values[0]
    assert min(va, 360.0 - va) < 1e-3                       # 359.8 and 0.2 -> 0, not 180
    sz = mid["sunZenithAngles"].values[1]
    near = [field_values("sunZenithAngles", 4, x[j], y[i])[1, 0, 0]
            for i, j in ((3, 4), (4, 3), (4, 4))]
    assert np.isfinite(sz) and min(near) - 1e-4 <= sz <= max(near) + 1e-4
    assert np.isfinite(mid["sunZenithAngles"].values).all()


def test_sample_point_weights():
    """Direct check of the weights on a tiny dataset: corners, centre, NaN."""
    ds = xr.Dataset({"a": (("t", "y", "x"), np.array([[[0.0, 10.0], [20.0, 30.0]]], "float32"))},
                    coords={"t": pd.date_range("2025-01-01", periods=1), "y": [100.0, 0.0],
                            "x": [0.0, 100.0]})
    assert zc.sample_point(ds, 50.0, 50.0)["a"].iloc[0] == pytest.approx(15.0)
    assert zc.sample_point(ds, 0.0, 100.0)["a"].iloc[0] == 0.0
    assert zc.sample_point(ds, 75.0, 100.0)["a"].iloc[0] == pytest.approx(7.5)
    assert zc.sample_point(ds, 100.0, 25.0)["a"].iloc[0] == pytest.approx(25.0)   # 10 -> 30
    ds["a"][0, 0, 0] = np.nan
    assert zc.sample_point(ds, 50.0, 50.0)["a"].iloc[0] == pytest.approx(20.0)   # (10+20+30)/3
    with pytest.raises(ValueError, match="outside the frame"):
        zc.sample_point(ds, 151.0, 50.0)


# ─────────────────────────────────────────────────────────────────────────────
#  --products: selection, requests, and the skipping of finished products
# ─────────────────────────────────────────────────────────────────────────────

def test_products_option_and_requests(tmp_path):
    assert dcz.parse_args(["ems"]).products == ["main", "angles"]
    assert dcz.parse_args(["ems", "--products", "angles"]).products == ["angles"]
    assert dcz.parse_args(["--products", " angles , main "]).products == ["main", "angles"]
    for bad in ("angels", ","):
        with pytest.raises(SystemExit):
            dcz.parse_args(["--products", bad])
    reqs = dcz.year_requests(["villaviciosa"], [2023, 2024], tmp_path, ["main", "angles"])
    assert [(r.year, r.product) for r in reqs] == [(2023, "main"), (2023, "angles"),
                                                  (2024, "main"), (2024, "angles")]
    r = reqs[1]
    assert r.bands == zc.ANGLE_BANDS and r.tag == "villaviciosa 2023 angles"
    assert r.store == zc.angles_path("villaviciosa", 2023, tmp_path)
    assert r.store.name == "2023.angles.zarr" and Path(r.part).name == "2023.angles.part"
    assert r.sidecar("verify.json").name == "2023.angles.verify.json"
    assert dcz.superseded_parts(r) == []
    flat = dcz.build_process(r).flat_graph()
    lc = flat["loadcollection1"]["arguments"]
    main_lc = dcz.build_process(reqs[0]).flat_graph()["loadcollection1"]["arguments"]
    assert lc["bands"] == list(zc.ANGLE_BANDS)
    # the angles extent: the main (10 m) extent snapped OUTWARD to multiples of 300 m
    me, ae = main_lc["spatial_extent"], lc["spatial_extent"]
    assert me["crs"] == ae["crs"] == 32630 and r.extent == ae
    assert all(ae[k] % 300 == 0 for k in ("west", "south", "east", "north"))
    assert (ae["west"] <= me["west"] < ae["west"] + 300 and ae["south"] <= me["south"] < ae["south"] + 300
            and ae["east"] - 300 < me["east"] <= ae["east"]
            and ae["north"] - 300 < me["north"] <= ae["north"])
    assert ae == dcz.extent_of(dcz.snap_frame(dcz.site_frame("villaviciosa", tmp_path), 300))
    assert lc["temporal_extent"] == ["2023-01-01", "2024-01-01"]
    assert lc.get("properties") == main_lc.get("properties")          # the same scenes
    rs = flat["resamplespatial1"]["arguments"]
    assert (rs["resolution"], rs["method"]) == (300, "bilinear")
    assert flat["saveresult1"]["arguments"]["format"] == "Zarr"
    assert dcz.job_title(r, dcz.graph_hash(flat)).startswith("zarr_angles_villaviciosa_2023_")
    tr = dcz.test_requests(tmp_path)
    assert [q.product for q in tr] == ["main", "angles"] and tr[1].tag == "villaviciosa 2025 angles TEST"
    assert dcz.job_title(tr[1], "0123456789ab").startswith("zarr_angles_test_villaviciosa_2025-06-01_")
    with pytest.raises(ValueError, match="unknown product"):
        dcz.Request(site="ems", year=2023, start="2023-01-01", end="2024-01-01", bbox={},
                    root=tmp_path, product="bands")


@pytest.fixture
def fake_build(monkeypatch):
    real = dcz.build_process

    def build(req, connection=None, backend_format=dcz.BACKEND_FORMAT, format_options=None):
        return FakeSaveResult(connection, real(req, None, backend_format, format_options).flat_graph())

    monkeypatch.setattr(dcz, "build_process", build)


def test_a_year_with_its_main_store_fetches_only_the_angles(tmp_path, wkt, monkeypatch, fake_build):
    import socket

    def no_net(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "create_connection", no_net)
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(tmp_path / "absent.nc"),
        pipeline_overpass=str(tmp_path / "none.json")))
    root = tmp_path / "cubes"
    t, y, x, data = det_cube(2025, n=4, H=70, W=95, start="2025-06-02")
    mreq = make_request(root, year=2025)
    place(mreq, [("a.zarr", write_raw_v2(tmp_path / "m" / "a.zarr", t, y, x, data, wkt), "x")])
    normalise_and_write(mreq)
    zc.write_json({str(d.date()): f"{d.date()}T11:00:00" for d in t},
                  zc.overpass_path("villaviciosa", root))
    # what the backend delivers: the 300 m cells around that 10 m grid
    ax, ay = dcz.coarse_axes(x, y)
    assert (len(ay), len(ax)) == (3, 4) and np.all(np.mod(ax - RES / 2, RES) == 0)
    adata = {b: field_values(b, len(t), ax, ay).astype("float32") for b in ANGLES}
    araw = write_angles_v2(tmp_path / "a" / "openEO.zarr", t, ay, ax, adata, wkt)
    azip = zip_store(araw, tmp_path / "a" / "openEO.zarr")
    downloads = []

    def fake_download(job, req, log, workers=4):
        downloads.append((req.product, job.job_id))
        place(req, [("openEO.zarr.zip", azip, "application/zip")])
        return {"assets": {}}

    monkeypatch.setattr(dcz, "download_results", fake_download)
    conn = FakeConn(plans=[["finished"]] * 3)
    monkeypatch.setattr(dcz.Context, "connection", lambda self: conn)
    monkeypatch.chdir(tmp_path)
    assert dcz.main(["villaviciosa", "--years", "2025", "--root", str(root)]) == 0
    assert downloads == [("angles", "j-1")]               # the main store was not fetched again
    assert conn.created == ["j-1"]
    assert conn.jobs["j-1"]["title"].startswith("zarr_angles_villaviciosa_2025_")
    areq = angles_request(root)
    assert zc.is_complete(areq.store, zc.ANGLES_FORMAT_ID) and not Path(areq.part).exists()
    rep = json.loads(areq.sidecar("verify.json").read_text())
    assert rep["verdict"] == "ok" and rep["dates"]["common"] == len(t)
    ds = zc.open_store(areq.store)
    assert ds.attrs["grid_reference"].startswith("predicted: 300 m cells around 2025.zarr")
    np.testing.assert_array_equal(ds["x"].values, ax)
    ds.close()
    log = (zc.site_dir("villaviciosa", root) / "download.log").read_text(encoding="utf-8")
    assert "[villaviciosa 2025] " in log and "[villaviciosa 2025 angles] " in log
    assert "angles verification: OK" in log
    # again: both complete, nothing paid for; --verify-only checks both
    assert dcz.main(["villaviciosa", "--years", "2025", "--root", str(root)]) == 0
    assert conn.created == ["j-1"] and downloads == [("angles", "j-1")]
    assert dcz.main(["villaviciosa", "--years", "2025", "--verify-only", "--root", str(root)]) == 0
    assert dcz.main(["villaviciosa", "--years", "2024", "--verify-only", "--root", str(root),
                     "--products", "angles"]) == 1
    # --products main: the angles are not looked at
    shutil.rmtree(areq.store)
    assert dcz.main(["villaviciosa", "--years", "2025", "--root", str(root), "--products",
                     "main"]) == 0
    assert conn.created == ["j-1"] and not areq.store.exists()


def test_dry_run_covers_both_products(tmp_path, monkeypatch, capsys):
    import socket

    def no_net(*a, **k):
        raise AssertionError("network used in a dry run")

    monkeypatch.setattr(socket, "create_connection", no_net)
    monkeypatch.chdir(tmp_path)
    assert dcz.main(["--test", "--dry-run", "--root", str(tmp_path / "cubes")]) == 0
    out = capsys.readouterr().out
    for s_ in ("== villaviciosa 2025 TEST", "== villaviciosa 2025 angles TEST",
               "zarr_cube_test_villaviciosa", "zarr_angles_test_villaviciosa", '"sunZenithAngles"',
               '"viewAzimuthMean"', '"bilinear"', '"near"', "grid 8 x 7 cells of 300 m",
               "as float32 (4 angles)", "TOTAL main: 1 store(s)", "TOTAL angles: 1 store(s)",
               "2025.angles.zarr", "free under"):
        assert s_ in out, s_
    assert not (tmp_path / "cubes").exists()
    monkeypatch.setitem(dcz.SITE_CONFIG, "villaviciosa", dict(
        dcz.SITE_CONFIG["villaviciosa"], reference=str(tmp_path / "absent.nc")))
    assert dcz.main(["villaviciosa", "--years", "2024", "--products", "angles", "--dry-run",
                     "--root", str(tmp_path / "cubes")]) == 0
    out = capsys.readouterr().out
    assert "grid 31 x 31 cells of 300 m" in out and "TOTAL angles: 1 store(s)" in out
    assert "TOTAL main" not in out and "angles grid guard: reference grid = the 10 m grid" in out

