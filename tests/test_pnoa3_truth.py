"""experiments/pnoa3_truth.py: the PNOA-LiDAR 3rd coverage gridded to a site's 10 m frame.

* tile names (hyphens or underscores) give the upper-left corner in km; the catalogue CSV parses;
* the per-cell median is numpy's median (odd and even counts);
* on synthetic LAZ tiles (LAS 1.4, point format 6, adjusted standard GPS time): only class 2 is used (classes 1,
  7, 9 and withheld ground points never enter), NaN below 3 ground points with the count kept, points outside
  the frame dropped, a point on a tile edge that falls in the neighbouring tile's cell merged exactly, the flight
  days from the GPS time, and the truth date = the last flight day of the tiles under the scored pixels;
* a site gridded in two batches (the first tile's raw file moved away, its cache kept) gives the same rasters;
* the GeoTIFF is on the frame: pyintertidal's reproject_to_grid reads it back bit for bit.

Needs laspy with the lazrs backend (the laptop venv); skipped without it.
Run:  python -m pytest tests/test_pnoa3_truth.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

laspy = pytest.importorskip("laspy")
pytest.importorskip("lazrs")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from experiments import pnoa3_truth as pt  # noqa: E402

X0, Y0, H, W = 303000.0, 4820000.0, 100, 150          # the frame: tile 303-4820 and half of 304-4820
DAY1 = 1718013600.0                                    # 2024-06-10T10:00:00Z
DAY2 = 1719914400.0                                    # 2024-07-02T10:00:00Z


def test_tile_names_and_catalogue():
    a = pt.parse_tile_name("PNOA-2024-AST-303-4820-H30-NPC01.LAZ")
    b = pt.parse_tile_name("some/dir/PNOA_2024_AST_303-4820_H30_NPC01.laz")
    assert a == b == dict(year=2024, region="AST", x_km=303, y_km=4820, zone=30, level="NPC01", key="303-4820")
    assert pt.tile_extent(303, 4820) == (303000.0, 4819000.0, 304000.0, 4820000.0)
    xk, yk = pt.tile_key_of([303000.0, 303999.9], [4819000.1, 4819999.9])
    assert list(xk) == [303, 303] and list(yk) == [4820, 4820]
    cat = pt.read_catalogue()
    assert cat.key.is_unique and set(cat.zone) == {29, 30} and (cat.size_mb > 0).all()
    assert "303-4820" in set(cat.key) and "567-4818" in set(cat.key)


def test_group_median_is_numpy_median():
    rng = np.random.default_rng(5)
    cell = rng.integers(0, 40, 3000)
    v = rng.normal(0, 1, 3000).round(3)
    u, n, m = pt.group_median(cell, v)
    for c, k, med in zip(u, n, m):
        assert k == (cell == c).sum() and med == np.median(v[cell == c])


def gps(unix):
    return np.asarray(unix, float) - pt.GPS_EPOCH_UNIX + pt.GPS_UTC_LEAP_S - 1e9


def write_laz(path, x, y, z, cls, t, withheld=None):
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([300000.0, 4800000.0, 0.0])
    hdr.global_encoding.gps_time_type = laspy.header.GpsTimeType.STANDARD
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    las.classification = np.asarray(cls, np.uint8)
    las.gps_time = gps(t)
    if withheld is not None:
        las.withheld = np.asarray(withheld, np.uint8)
    las.write(path)


def centre(r, c):
    return X0 + 10 * c + 5, Y0 - 10 * r - 5


def synthetic_site(tmp):
    """A frame, its extraction (inter = rows 50-60), a no-delay product spanning [0, 1] m and two LAZ tiles.
    Returns (laz dir, the expected ground z of every cell as lists)."""
    rng = np.random.default_rng(7)
    tr = from_origin(X0, Y0, 10, 10)
    grid = os.path.join(tmp, "grid.tif")
    with rasterio.open(grid, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", crs="EPSG:32630",
                       transform=tr, nodata=np.nan) as d:
        d.write(np.zeros((H, W), np.float32), 1)
    inter = np.zeros((H, W), bool)
    inter[50:61, :] = True
    np.savez(os.path.join(tmp, "extract.npz"), inter=inter, keep=np.flatnonzero(inter))
    np.savez(os.path.join(tmp, "nodelay.npz"), z=np.array([0.0, 0.4, 1.0, np.nan]))
    pt.SITES["synthetic"] = dict(grid=("tif", grid), extract=os.path.join(tmp, "extract.npz"),
                                 nodelay=(os.path.join(tmp, "nodelay.npz"), "z"), zone=30, region="AST", sheets=[])
    laz = os.path.join(tmp, "laz")
    os.makedirs(laz)
    expect = {}
    pts = {"A": [], "B": []}                           # (x, y, z, class, t, withheld)

    def add(tile, r, c, zs, cls=2, t=DAY1, withheld=0, dx=0.0):
        x, y = centre(r, c)
        for z in zs:
            pts[tile].append((x + dx, y, z, cls, t, withheld))
            if cls == 2 and not withheld and 0 <= c < W:
                expect.setdefault((r, c), []).append(z)

    add("A", 10, 10, [1, 2, 3, 4, 100])                # odd count: median 3
    add("A", 10, 10, [50], cls=1)
    add("A", 10, 10, [-50], cls=7)
    add("A", 10, 10, [0.5], cls=9)                     # water: never used
    add("A", 10, 10, [-100], withheld=1)               # withheld ground: never used
    add("A", 20, 20, [1, 2, 3, 10])                    # even count: 2.5
    add("A", 30, 30, [7, 8])                           # 2 points: NaN, count 2
    for r in range(50, 61):                            # the intertidal rows: z in [0, 1] in A, [5, 6] in B
        for c in range(0, 100):
            add("A", r, c, rng.uniform(0, 1, 3).round(3))
        for c in range(100, 150):
            add("B", r, c, rng.uniform(5, 6, 3).round(3))
    add("B", 40, 100, [5, 6, 7])                       # a cell of B ...
    x, y = 304000.0, centre(40, 100)[1]                # ... and a point of A exactly on the edge, in B's cell
    pts["A"].append((x, y, 8.0, 2, DAY1, 0))
    expect[(40, 100)].append(8.0)
    add("B", 45, 120, [9, 9, 9], cls=1, t=DAY2)        # a later flight day in B, not ground
    for c in (170, 190):                               # in tile B but outside the frame: dropped
        add("B", 55, c, [999, 999, 999])
    for tile, name in (("A", "PNOA_2024_AST_303-4820_H30_NPC01.laz"), ("B", "PNOA_2024_AST_304-4820_H30_NPC01.laz")):
        a = np.array(pts[tile], float)
        write_laz(os.path.join(laz, name), a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(int), a[:, 4], a[:, 5])
    return laz, expect


def read(path):
    with rasterio.open(path) as r:
        return r.read(1), r.transform, r.crs


def check_rasters(out, expect):
    z, tr, crs = read(os.path.join(out, "synthetic_pnoa3_10m.tif"))
    n, _, _ = read(os.path.join(out, "synthetic_pnoa3_10m_count.tif"))
    t, _, _ = read(os.path.join(out, "synthetic_pnoa3_10m_time.tif"))
    assert z.dtype == np.float32 and n.dtype == np.int32 and t.dtype == np.float64
    assert tr == from_origin(X0, Y0, 10, 10) and crs.to_epsg() == 32630 and z.shape == (H, W)
    for (r, c), zs in expect.items():
        zs = np.round(np.asarray(zs, float), 3)
        assert n[r, c] == len(zs), (r, c)
        if len(zs) >= 3:
            assert abs(z[r, c] - np.float32(np.median(zs))) < 1e-6, (r, c, z[r, c], np.median(zs))
        else:
            assert np.isnan(z[r, c])
    assert z[10, 10] == 3.0 and n[10, 10] == 5 and z[20, 20] == 2.5 and np.isnan(z[30, 30]) and n[30, 30] == 2
    assert z[40, 100] == 6.5 and n[40, 100] == 4       # merged across the tile edge: 5, 6, 7 and 8
    assert np.isnan(z[45, 120]) and n[45, 120] == 0    # class 1 only
    have = np.zeros((H, W), bool)
    have[tuple(np.array([k for k, v in expect.items() if len(v) >= 3]).T)] = True
    assert np.array_equal(np.isfinite(z), have), "a cell without 3 ground points has a value, or the reverse"
    assert np.all(np.isfinite(t[n > 0])) and np.all(np.isnan(t[n == 0]))
    return z, n, t


def test_grid_synthetic_tiles_and_truth_date():
    import pyintertidal as pit
    with tempfile.TemporaryDirectory() as tmp:
        laz, expect = synthetic_site(tmp)
        out = os.path.join(tmp, "out")
        doc = pt.cmd_grid("synthetic", laz_dir=laz, out_root=out)
        z, n, _ = check_rasters(out, expect)
        back = pit.reproject_to_grid(os.path.join(out, "synthetic_pnoa3_10m.tif"), from_origin(X0, Y0, 10, 10),
                                     rasterio.crs.CRS.from_epsg(32630), (H, W))
        assert np.array_equal(np.isnan(back), np.isnan(z)) and np.array_equal(back[~np.isnan(z)], z[~np.isnan(z)])
        j = json.load(open(os.path.join(out, "synthetic_pnoa3_10m.json"), encoding="utf-8"))
        assert j["truth_date"] == doc["truth_date"] and j["format"] == pt.FORMAT
        tiles = {t["key"]: t for t in j["tiles"]}
        assert set(tiles) == {"303-4820", "304-4820"}
        a, b = tiles["303-4820"], tiles["304-4820"]
        ground_a = 5 + 4 + 2 + 11 * 100 * 3 + 1        # + 1 withheld ground point
        assert a["classes"] == {"1": 1, "2": ground_a + 1, "7": 1, "9": 1} and a["ground"] == ground_a
        assert a["withheld"] == 1 and a["water_class_points"] == 1 and a["gps_time_adjusted"]
        assert a["cells_outside_tile"] == 1                   # the edge point's cell is B's
        assert a["flight_days"] == {"2024-06-10": a["points"]} and set(b["flight_days"]) == {"2024-06-10", "2024-07-02"}
        assert b["ground"] == b["ground_in_frame"] + 6        # the 6 points beyond the frame's east edge
        assert j["totals"]["cells_shared_between_tiles"] == 1
        assert j["intertidal"]["pixels"] == 11 * 150 and j["intertidal"]["with_value"] == 11 * 150
        assert j["scored"]["range_m"] == [-0.25, 1.25] and j["scored"]["pixels"] == 11 * 100   # tile A only
        assert j["flight"]["intertidal_with_value"]["last"] == "2024-07-02"     # B flew on a later day too
        assert j["flight"]["scored"] == dict(first="2024-06-10", last="2024-06-10", tiles=["303-4820"],
                                             days=["2024-06-10"])
        assert j["truth_date"] == "2024-06-10" and j["missing_tiles"] == []
        m = json.load(open(os.path.join(laz, "manifest.json"), encoding="utf-8"))
        assert {f["key"] for f in m["files"]} == set(tiles)
        assert all(len(f["sha256"]) == 64 and f["size_bytes"] > 0 for f in m["files"])
        assert m["files"][0]["url"].startswith(pt.DETAIL_URL.split("{")[0])      # 303-4820 is in the catalogue


def test_two_batches_give_the_same_rasters():
    with tempfile.TemporaryDirectory() as tmp:
        laz, expect = synthetic_site(tmp)
        ref = os.path.join(tmp, "ref")
        pt.cmd_grid("synthetic", laz_dir=laz, out_root=ref)
        z0, n0, t0 = (read(os.path.join(ref, f"synthetic_pnoa3_10m{s}.tif"))[0] for s in ("", "_count", "_time"))
        shutil.rmtree(os.path.join(laz, "cache"))
        b = os.path.join(laz, "PNOA_2024_AST_304-4820_H30_NPC01.laz")
        aside = os.path.join(tmp, "aside")
        os.makedirs(aside)
        shutil.move(b, aside)                          # batch 1: tile A only
        out = os.path.join(tmp, "out")
        doc = pt.cmd_grid("synthetic", laz_dir=laz, out_root=out)
        assert [t["key"] for t in doc["missing_tiles"]] == ["304-4820"]
        assert doc["missing_tiles"][0]["intertidal_px"] == 11 * 50
        a = os.path.join(laz, "PNOA_2024_AST_303-4820_H30_NPC01.laz")
        shutil.move(a, aside)                          # batch 2: A's raw file moved away (its cache stays), B back
        shutil.move(os.path.join(aside, os.path.basename(b)), laz)
        doc = pt.cmd_grid("synthetic", laz_dir=laz, out_root=out)
        z1, n1, t1 = (read(os.path.join(out, f"synthetic_pnoa3_10m{s}.tif"))[0] for s in ("", "_count", "_time"))
        assert np.array_equal(n0, n1) and np.array_equal(z0, z1, equal_nan=True)
        assert np.array_equal(t0, t1, equal_nan=True)
        assert doc["missing_tiles"] == [] and doc["truth_date"] == "2024-06-10"
        assert [t.get("raw_file_present", True) for t in doc["tiles"]] == [False, True]
