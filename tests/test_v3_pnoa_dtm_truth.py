# -*- coding: utf-8 -*-
"""experiments/v3/pnoa_dtm_truth.py: the PNOA DTM truth of a v3 site, on synthetic ETRS89 / UTM sheets.

* with one 2 m sheet, z is v2's pnoa_truth z bit for bit (its window, pnoa_ground with 50 cells, aggregate_ground);
* a sheet cut in two gives the same z as the whole sheet, a water patch across the cut included (whole in the mosaic);
* where sheets overlap the first one with a value is used, and the overlap is counted;
* the area rule: 200 m2 is 50 cells at 2 m and 800 at 0.5 m (799 cells of one value stay ground at 0.5 m);
* sheets off one lattice, of two zones or geographic are refused;
* the grid read from a netCDF cube is the GeoTIFF's; ``build`` writes the GeoTIFF and its JSON record.

Run:  python -m pytest -q tests/test_v3_pnoa_dtm_truth.py
"""
import json
import os
import sys

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import truth  # noqa: E402
from experiments.v3 import pnoa_dtm_truth as pt  # noqa: E402

X0, Y0 = 500000.0, 4810000.0
NODATA = -32767.0


def terrain(h, w, cell, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    z = 0.002 * xx * cell - 0.001 * yy * cell + rng.normal(0, 0.05, (h, w))
    return z.astype("float32")


def write_tif(path, a, x0, y0, cell, crs="EPSG:25830", nodata=NODATA):
    a = np.where(np.isfinite(a), a, nodata).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32",
                       crs=crs, transform=from_origin(x0, y0, cell, cell), nodata=nodata) as d:
        d.write(a, 1)
    return str(path)


def grid_tif(path, x0=X0 + 103.0, y0=Y0 - 97.0, shape=(60, 80), crs="EPSG:32630"):
    return write_tif(path, np.zeros(shape, "float32"), x0, y0, 10.0, crs=crs, nodata=0.0)


def sheet_2m(seed=0):
    z = terrain(400, 500, 2.0, seed)
    z[100:120, 240:270] = 1.234          # 600 cells of one value: water, across the cut at column 250
    z[300:306, 50:56] = 0.777            # 36 cells: under 50, ground
    z[:, :3] = np.nan                    # no data on the west edge
    return z


def pnoa_truth_reference(grid_path, mdt_path):
    """v2's experiments.benchmark.truth.pnoa_truth, its body with explicit paths (no flight level)."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(grid_path) as g:
        gt, gcrs, gshape, gb = g.transform, g.crs, g.shape, g.bounds
    with rasterio.open(mdt_path) as s:
        w = from_bounds(*transform_bounds(gcrs, s.crs, *gb, densify_pts=21), transform=s.transform)
        w = w.round_offsets().round_lengths()
        src = s.read(1, window=w, boundless=True, fill_value=s.nodata).astype("float32")
        src[src == s.nodata] = np.nan
        st, scrs = s.window_transform(w), s.crs
    gr = truth.pnoa_ground(src, None)
    z, _, _ = truth.aggregate_ground(src, st, scrs, gr["ground"], gr["valid"], gt, gcrs, gshape)
    return z


def test_the_area_rule_and_the_names():
    assert (pt.patch_min_cells(2.0), pt.patch_min_cells(0.5), pt.patch_min_cells(1.0)) == (50, 800, 200)
    assert pt.patch_min_cells(3.0) == 23
    assert pt.parse_sheet_name("MDT02-ETRS89-HU30-0015-3-COB2.TIF") == {"product": "MDT02", "zone": 30,
                                                                         "sheet": "0015-3"}
    assert pt.parse_sheet_name("MDT50CM-ETRS89-H30-0062-5-1-COB3-V1.TIF") == {"product": "MDT50CM", "zone": 30,
                                                                              "sheet": "0062-5-1"}
    assert pt.parse_sheet_name("elevation.tif") is None


def test_one_2m_sheet_is_pnoa_truth_bit_for_bit(tmp_path):
    s = write_tif(tmp_path / "MDT02-ETRS89-HU30-0001-1-COB2.tif", sheet_2m(), X0, Y0, 2.0)
    g = grid_tif(tmp_path / "grid.tif")
    res = pt.build_truth(s, g)
    ref = pnoa_truth_reference(g, s)
    assert np.isfinite(ref).sum() > 3000
    assert np.array_equal(res["z"], ref.astype("float32"), equal_nan=True)
    assert res["patch_min_cells"] == 50 and res["reprojected"]
    assert res["cells_water_patch"] == 600, "the 600-cell patch only; the 36-cell one is ground"


def test_a_sheet_cut_in_two_is_the_whole_sheet(tmp_path):
    z = sheet_2m()
    whole = write_tif(tmp_path / "whole.tif", z, X0, Y0, 2.0)
    west = write_tif(tmp_path / "west.tif", z[:, :250], X0, Y0, 2.0)
    east = write_tif(tmp_path / "east.tif", z[:, 250:], X0 + 500.0, Y0, 2.0)
    g = grid_tif(tmp_path / "grid.tif")
    a, b = pt.build_truth(whole, g), pt.build_truth([west, east], g)
    assert np.array_equal(a["z"], b["z"], equal_nan=True)
    assert b["cells_water_patch"] == 600, "a patch across the cut is one patch in the mosaic"
    assert [e["cells_overlap"] for e in b["sheets"]] == [0, 0]
    assert sum(e["cells_used"] for e in b["sheets"]) == b["cells_valid"] == a["cells_valid"]


def test_where_sheets_overlap_the_first_one_wins(tmp_path):
    z = sheet_2m()
    first = write_tif(tmp_path / "first.tif", z[:, :300], X0, Y0, 2.0)
    other = z[:, 200:].copy()
    other[:, :100] += 0.5                                   # the overlap (columns 200-299) differs by 0.5 m
    second = write_tif(tmp_path / "second.tif", other, X0 + 400.0, Y0, 2.0)
    g = grid_tif(tmp_path / "grid.tif")
    res = pt.build_truth([first, second], g)
    whole = pt.build_truth(write_tif(tmp_path / "whole.tif", z, X0, Y0, 2.0), g)
    assert np.array_equal(res["z"], whole["z"], equal_nan=True), "the first sheet's cells, not the second's"
    s2 = res["sheets"][1]
    assert s2["cells_overlap"] > 0 and s2["overlap_cells_differ"] == s2["cells_overlap"]
    assert abs(s2["overlap_max_abs_diff_m"] - 0.5) < 1e-6


def test_the_area_rule_at_half_a_metre(tmp_path):
    rng = np.random.default_rng(3)
    z = (rng.normal(0, 0.05, (400, 400)) + np.linspace(0, 1, 400)).astype("float32")
    z[10:50, 10:30] = 2.5                                   # 800 cells = 200 m2: water
    blob = np.zeros((400, 400), bool)
    blob[200:240, 200:220] = True
    blob[239, 219] = False                                  # 799 cells: ground
    z[blob] = 3.5
    s = write_tif(tmp_path / "MDT50CM-ETRS89-H30-0038-4-5-COB3-V1.tif", z, X0, Y0, 0.5)
    g = grid_tif(tmp_path / "grid.tif", x0=X0 + 3.0, y0=Y0 - 3.0, shape=(19, 19))
    res = pt.build_truth(s, g)
    assert res["patch_min_cells"] == 800 and res["cells_water_patch"] == 800


def test_the_two_epsg_labels_of_one_utm_zone_mosaic(tmp_path):
    # the CNIG labels sheets of one zone 25829 or 3041 (axis order only): the Eo's 0010-3 and 0010-4
    z = sheet_2m()
    g = grid_tif(tmp_path / "grid.tif", crs="EPSG:32629")
    whole = write_tif(tmp_path / "whole.tif", z, X0, Y0, 2.0, crs="EPSG:25829")
    west = write_tif(tmp_path / "west.tif", z[:, :250], X0, Y0, 2.0, crs="EPSG:25829")
    east = write_tif(tmp_path / "east.tif", z[:, 250:], X0 + 500.0, Y0, 2.0, crs="EPSG:3041")
    a, b = pt.build_truth(whole, g), pt.build_truth([west, east], g)
    assert np.array_equal(a["z"], b["z"], equal_nan=True)
    assert [e["crs"] for e in b["sheets"]] == ["EPSG:25829", "EPSG:3041"], "each sheet's own label is recorded"
    alone = pt.build_truth(write_tif(tmp_path / "alone.tif", z, X0, Y0, 2.0, crs="EPSG:3041"), g)
    assert np.array_equal(a["z"], alone["z"], equal_nan=True), "a 3041 sheet alone gives the 25829 truth"


def test_sheets_that_cannot_be_mosaicked_are_refused(tmp_path):
    z = sheet_2m()
    g = grid_tif(tmp_path / "grid.tif")
    a = write_tif(tmp_path / "a.tif", z[:, :250], X0, Y0, 2.0)
    off = write_tif(tmp_path / "off.tif", z[:, 250:], X0 + 501.0, Y0, 2.0)
    with pytest.raises(ValueError, match="lattice"):
        pt.build_truth([a, off], g)
    zone = write_tif(tmp_path / "zone.tif", z[:, 250:], X0 + 500.0, Y0, 2.0, crs="EPSG:25829")
    with pytest.raises(ValueError, match="CRS"):
        pt.build_truth([a, zone], g)
    geo = write_tif(tmp_path / "MDT02-WGS84-0015-3-COB2.tif", z, -5.5, 43.6, 1.8e-5, crs="EPSG:4326")
    with pytest.raises(ValueError, match="not projected"):
        pt.build_truth(geo, g)
    with pytest.raises(ValueError, match="twice"):
        pt.build_truth([a, a], g)


def test_the_grid_of_a_cube_and_build(tmp_path):
    xr = pytest.importorskip("xarray")
    g = grid_tif(tmp_path / "grid.tif")
    with rasterio.open(g) as r:
        t, shape = r.transform, r.shape
    xs = t.c + 10.0 * (np.arange(shape[1]) + 0.5)
    ys = t.f - 10.0 * (np.arange(shape[0]) + 0.5)
    ds = xr.Dataset({"B03": (("t", "y", "x"), np.zeros((1,) + shape, "int16"))},
                    coords={"t": np.array(["2020-07-01"], "datetime64[ns]"), "y": ys, "x": xs})
    ds["crs"] = xr.DataArray(0, attrs={"crs_wkt": rasterio.crs.CRS.from_epsg(32630).to_wkt()})
    cube = str(tmp_path / "cube.nc")
    ds.to_netcdf(cube)
    ct, ccrs, cshape = pt.grid_of_cube(cube)
    assert cshape == shape and ccrs.to_epsg() == 32630
    assert np.allclose(tuple(ct)[:6], tuple(t)[:6])
    s = write_tif(tmp_path / "MDT02-ETRS89-HU30-0001-1-COB2.tif", sheet_2m(), X0, Y0, 2.0)
    out = tmp_path / "out" / "site_cob2_10m.tif"
    rec = pt.build([s], cube, out, "2020-07-15")
    with rasterio.open(out) as r:
        z = r.read(1)
        assert r.shape == shape and r.crs.to_epsg() == 32630
    assert np.array_equal(z, pt.build_truth(s, g)["z"], equal_nan=True)
    j = json.load(open(str(out).replace(".tif", ".json"), encoding="utf-8"))
    assert j["truth_date"] == "2020-07-15" and j["truth_date_rule"].startswith("provisional")
    assert len(j["sources"]["sheets"][0]["sha256"]) == 64 and j["sources"]["grid"]["path"].endswith("cube.nc")
    assert j["pixels"]["with_value"] == int(np.isfinite(z).sum()) == rec["pixels"]["with_value"]
    with pytest.raises(ValueError):
        pt.build([s], cube, tmp_path / "x.tif", "2020-15-07")
