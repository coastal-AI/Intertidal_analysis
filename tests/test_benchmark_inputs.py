"""Tests of experiments.benchmark.inputs, the scene inputs of validation v2.

They pin (spec §1.2, §1.10, §1.11, §10.3): a synthetic site of two yearly
Zarr stores with all nine bands (2024 delivered without the BOA offset, 2025
with +1000 in the values; the 20 m bands exact 2 x 2 replicas on lattice
phase (1, 0)), an angles store and a legacy netCDF cube on the same grid.
The record (dates with a valid instant, positions -> (year, local index),
windows); raw reads equal ``zarr_cubes.open_cube`` (SCL and CLD as checked
uint8); the 16-date block cache decodes every chunk once, also with the
lattice-aligned row blocks of the 20 m rule; the 20 m index of a block
equals the frame's; lattice20 measured, None for non-replicated bands and
ambiguity resolved by the grid; the per-scene offset rule (every branch)
and its provisional and frozen forms; fingerprints; grid assertions
(years, legacy netCDF, GeoTIFF); the NetCDF source; the scene metadata
sidecar from STAC features; the store checks and the inputs report. The
scene instants are the catalogue times by default (user decision of
2026-10-14): a sensing_times.json beside the stores is never picked up, only
an explicit path is read; the legacy-cube report of the local 20 m runs.

Run:  python -X utf8 -m pytest tests/test_benchmark_inputs.py -q
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

from experiments import zarr_cubes as zc  # noqa: E402
from experiments.benchmark import inputs, obs  # noqa: E402

FILL = -32768
SITE = "escalda"
H, W = 300, 24
#: left edge 500000 (a multiple of 20 m) -> px = 0; top edge 4800010 (10 m
#: past a multiple of 20 m) -> py = 1
X_EDGE, Y_EDGE = 500_000.0, 4_800_010.0
PHASE = (1, 0)
N_PER_YEAR = 18
WATER_COLS = 12


def _wkt():
    from pyproj import CRS

    return CRS.from_epsg(32631).to_wkt("WKT1_GDAL")


def _coords():
    x = X_EDGE + 5.0 + 10.0 * np.arange(W)
    y = Y_EDGE - 5.0 - 10.0 * np.arange(H)
    return x, y


def _dates(year):
    return pd.date_range(f"{year}-01-04", periods=N_PER_YEAR, freq="9D")


def _rep20(blocks):
    return obs.expand20(blocks, PHASE, (H, W))


def year_bands(year, offset, seed):
    """Raw int16 bands of one synthetic year: water on the left columns
    (SCL 6, dark B08/B11/B12), land on the right; scene 3 cloudy (SCL 9,
    CLD 90 everywhere); scene 5 has class 11; a fill pixel in B03 and in SCL;
    one negative B08 value in the offset-free year; the 20 m bands (B8A,
    B11, B12, SCL, CLD) exact 2 x 2 replicas on PHASE."""
    rng = np.random.default_rng(seed)
    T = N_PER_YEAR
    Hb, Wb = obs._block_index(H, PHASE[0])[1], obs._block_index(W, PHASE[1])[1]
    water10 = np.zeros((H, W), bool)
    water10[:, :WATER_COLS] = True
    water20 = obs.block_take20(water10, PHASE)
    d = {}
    d["B02"] = np.where(water10, rng.integers(300, 600, (T, H, W)),
                        rng.integers(500, 1200, (T, H, W)))
    d["B03"] = np.where(water10, rng.integers(500, 900, (T, H, W)),
                        rng.integers(600, 1300, (T, H, W)))
    d["B04"] = np.where(water10, rng.integers(300, 700, (T, H, W)),
                        rng.integers(700, 1500, (T, H, W)))
    d["B08"] = np.where(water10, rng.integers(50, 300, (T, H, W)),
                        rng.integers(1800, 3500, (T, H, W)))
    for b, wlo, whi, llo, lhi in (("B8A", 40, 250, 1800, 3400), ("B11", 20, 200, 1500, 2800),
                                  ("B12", 10, 150, 900, 2000)):
        blk = np.where(water20, rng.integers(wlo, whi, (T, Hb, Wb)),
                       rng.integers(llo, lhi, (T, Hb, Wb)))
        d[b] = _rep20(blk)
    scl20 = np.where(water20, 6, rng.choice([4, 5], (T, Hb, Wb)))
    cld20 = rng.integers(0, 15, (T, Hb, Wb))
    scl20[3] = 9
    cld20[3] = 90
    scl20[5, :3, :3] = 11
    scl20[7, 10:20, 6:12] = 8
    cld20[7, 10:20, 6:12] = 60
    d["SCL"] = _rep20(scl20)
    d["CLD"] = _rep20(cld20)
    d = {b: a.astype(np.int16) for b, a in d.items()}
    if offset:
        for b in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12"):
            d[b] = (d[b] + 1000).astype(np.int16)
    else:
        d["B08"][2, 0, 0] = -12                      # dark water below 0: no offset
    d["B03"][1, 5, 5] = FILL
    d["SCL"][4, 0, 1] = FILL
    return d


def write_year(root, year, offset, seed):
    import xarray as xr
    from experiments import download_cube_zarr as dcz

    x, y = _coords()
    data = year_bands(year, offset, seed)
    ds = xr.Dataset({b: (("t", "y", "x"), data[b], {"long_name": b}) for b in zc.BANDS},
                    coords={"t": _dates(year), "y": y, "x": x})
    wkt = _wkt()
    ds["crs"] = xr.DataArray(np.int32(0), attrs={
        "crs_wkt": wkt, "spatial_ref": wkt, "GeoTransform": f"{X_EDGE} 10 0 {Y_EDGE} 0 -10"})
    out = zc.store_path(SITE, year, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    dcz.write_store(ds, {b: FILL for b in zc.BANDS}, out, {"site": SITE},
                    log=lambda m: None, verify=False)
    return data


def write_angles(root, year):
    import xarray as xr
    from experiments import download_cube_zarr as dcz

    t = _dates(year)
    ax = X_EDGE + 150.0 + 300.0 * np.arange(2)
    ay = Y_EDGE - 150.0 - 300.0 * np.arange(11)
    shape = (len(t), len(ay), len(ax))
    vals = {"sunZenithAngles": 40.0, "sunAzimuthAngles": 160.0, "viewZenithMean": 5.0,
            "viewAzimuthMean": 100.0}
    ds = xr.Dataset({b: (("t", "y", "x"),
                         np.full(shape, v, np.float32) + np.arange(len(t), dtype=np.float32)[:, None, None],
                         {"units": "degree"}) for b, v in vals.items()},
                    coords={"t": t, "y": ay, "x": ax})
    wkt = _wkt()
    ds["crs"] = xr.DataArray(np.int32(0), attrs={
        "crs_wkt": wkt, "spatial_ref": wkt,
        "GeoTransform": f"{X_EDGE} 300 0 {Y_EDGE} 0 -300"})
    out = zc.angles_path(SITE, year, root)
    dcz.write_store(ds, {b: None for b in vals}, out, {"site": SITE}, log=lambda m: None,
                    verify=False, format_id=zc.ANGLES_FORMAT_ID)


def write_legacy(path, data, year):
    import xarray as xr

    x, y = _coords()
    ds = xr.Dataset({b: (("t", "y", "x"), data[b]) for b in ("B03", "B08", "SCL")},
                    coords={"t": _dates(year), "y": y, "x": x})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": _wkt()})
    enc = {v: {"dtype": "int16", "_FillValue": FILL, "chunksizes": (1, 64, 24), "zlib": True}
           for v in ("B03", "B08", "SCL")}
    ds.to_netcdf(path, encoding=enc)


def instants_for(dates, drop=()):
    return {d: (None if d in drop else f"{d}T10:5{i % 10}:00Z") for i, d in enumerate(dates)}


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zarr_site")
    root = tmp / "cubes"
    data = {2024: write_year(root, 2024, offset=False, seed=1),
            2025: write_year(root, 2025, offset=True, seed=2)}
    write_angles(root, 2025)
    dates = [str(d.date()) for y in (2024, 2025) for d in _dates(y)]
    drop = {dates[6], dates[20]}
    inst = instants_for(dates, drop)
    legacy = tmp / "legacy_2024.nc"
    write_legacy(legacy, data[2024], 2024)
    return {"root": root, "data": data, "dates": dates, "drop": drop, "instants": inst,
            "legacy": legacy, "tmp": tmp}


def _src(site, **kw):
    kw.setdefault("lattice", PHASE)
    return inputs.ZarrSource(SITE, root=site["root"], instants=site["instants"], **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  record, positions, raw reads
# ─────────────────────────────────────────────────────────────────────────────

def test_record_positions_and_windows(site):
    with _src(site) as src:
        want = [d for d in site["dates"] if d not in site["drop"]]
        assert src.dates == want and src.T == 34 == len(src.instants)
        assert src.dropped == sorted(site["drop"])
        assert src.years == [2024, 2025] and src.bands == zc.BANDS
        assert src.grid.shape == (H, W)
        assert all(t.strftime("%Y-%m-%d") == d for t, d in zip(src.instants, src.dates))
        # positions: (year, local index in that year's store)
        q = src.dates.index(site["dates"][7])
        assert src.position(q) == (2024, 7)
        q = src.dates.index(site["dates"][N_PER_YEAR + 3])
        assert src.position(q) == (2025, 3)
        w = src.window("2025-01-01", "2026-01-01")
        assert [src.dates[i] for i in w] == [d for d in want if d.startswith("2025")]
        # keep_dates restricts the record (the finite-h_B condition)
        sub = _src(site, keep_dates=want[:5])
        assert sub.dates == want[:5]
        sub.close()


def test_instants_must_fall_on_their_date(site):
    bad = dict(site["instants"])
    d = site["dates"][0]
    bad[d] = "2024-01-05T10:50:00Z"
    with pytest.raises(inputs.InputsError, match="not on their label date"):
        inputs.ZarrSource(SITE, root=site["root"], instants=bad, lattice=PHASE)


def test_read_sensing_times_formats(tmp_path):
    g = {"format": "x", "dates": {"2024-01-04": {"instant": "2024-01-04T10:51:12.5Z", "A": 1},
                                  "2024-01-13": {"instant": None}}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(g), encoding="utf-8")
    r = inputs.read_sensing_times(p)
    assert r["2024-01-04"] == pd.Timestamp("2024-01-04 10:51:12.5") and pd.isna(r["2024-01-13"])
    lab = inputs.read_sensing_times({"2024-01-04": "2024-01-04T10:40:00+00:00"})
    assert lab["2024-01-04"] == pd.Timestamp("2024-01-04 10:40")
    assert inputs.read_sensing_times({"2024-01-04": {"instant": "2024-01-04T11:00:00"}}) == \
        {"2024-01-04": pd.Timestamp("2024-01-04 11:00")}


def test_reads_equal_open_cube(site):
    with pytest.warns(UserWarning, match="disagree on the L2A"):
        ds = zc.open_cube(SITE, "2024-01-01", "2026-01-01", root=site["root"])
    all_dates = zc.dates(ds)
    with _src(site) as src:
        for q in (0, 5, 17, 18, 33):
            d = src.dates[q]
            i = all_dates.index(d)
            for rows in (None, slice(250, 270), slice(0, 3), slice(256, 300)):
                sc = src.read_scene(q, zc.BANDS, rows)
                r = slice(None) if rows is None else rows
                for b in zc.BANDS:
                    ref = np.asarray(ds[b].isel(t=i).values)[r]
                    if b == "SCL":
                        assert sc[b].dtype == np.uint8
                        assert np.array_equal(sc[b], obs.scl_u8(ref, FILL))
                    elif b == "CLD":
                        assert sc[b].dtype == np.uint8
                        assert np.array_equal(sc[b], obs.cld_u8(ref, FILL))
                    else:
                        assert sc[b].dtype == np.int16 and np.array_equal(sc[b], ref)
        # a block across the two years and both time blocks of 2024
        qs = [1, 16, 17, 20, 30]
        blk = src.read_block(qs, ["B03", "SCL"], slice(240, 280))
        for k, q in enumerate(qs):
            i = all_dates.index(src.dates[q])
            assert np.array_equal(blk["B03"][k], ds["B03"].isel(t=i).values[240:280])
        with pytest.raises(KeyError):
            src.read_scene(0, ["B05"])
    ds.close()


def test_scl_out_of_range_is_refused(site, tmp_path):
    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    import zarr

    g = zarr.open_group(str(zc.store_path(SITE, 2024, root)), mode="r+", zarr_format=2)
    a = g["SCL"]
    blk = a[0:1, 0:1, 0:1]
    blk[...] = 13
    a[0:1, 0:1, 0:1] = blk
    with inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE) as src:
        with pytest.raises(ValueError, match="outside the codes"):
            src.read_scene(0, ["SCL"])


# ─────────────────────────────────────────────────────────────────────────────
#  the block cache and row blocks
# ─────────────────────────────────────────────────────────────────────────────

def _expected_loads(src, bands):
    keys = set()
    for q in range(src.T):
        y, loc = src.position(q)
        for rc in range((H - 1) // inputs.ROW_BLOCK + 1):
            for b in bands:
                keys.add((y, b, loc // inputs.T_BLOCK, rc))
    return len(keys)


@pytest.mark.parametrize("pair20", [False, True])
def test_block_cache_decodes_each_chunk_once(site, pair20):
    with _src(site) as src:
        seen = []
        for blk in src.row_blocks(zc.BANDS, pair20=pair20):
            assert blk.data["B03"].shape == (len(blk.positions), blk.rows.stop - blk.rows.start, W)
            seen.append((tuple(blk.positions), blk.rows.start, blk.rows.stop))
        assert src.cache.loads == _expected_loads(src, zc.BANDS)
        # chunk order: time blocks in order, all row blocks of one time block
        firsts = [s[0][0] for s in seen]
        assert firsts == sorted(firsts)
        covered = {}
        for qs, r0, r1 in seen:
            for q in qs:
                covered.setdefault(q, []).append((r0, r1))
        assert sorted(covered) == list(range(src.T))
        for q, rr in covered.items():
            assert rr[0][0] == 0 and rr[-1][1] == H
            assert all(a[1] == b[0] for a, b in zip(rr[:-1], rr[1:]))
        bounds = sorted({r0 for _, r0, _ in seen})
        assert bounds == ([0, 257] if pair20 else [0, 256])
        if pair20:
            assert src.cache.hits > 0
    # a tiny cache still gives the same values (decoding more than once)
    with _src(site, cache_mb=0.01) as small, _src(site) as big:
        a = small.read_block([0, 1, 2], ["B11"])["B11"]
        assert np.array_equal(a, big.read_block([0, 1, 2], ["B11"])["B11"])


def test_pair20_blocks_give_the_frames_20m_index(site):
    with _src(site) as src:
        qs = [0, 1, 2]
        full = src.read_block(qs, ["B03", "B08", "B11", "SCL"])
        x_full = obs.compute_index(obs.prepare(full, ["B03", "B11"], 0, "floor0"), "mndwi", PHASE)
        parts = []
        for blk in src.row_blocks(["B03", "B11", "SCL"], positions=qs, pair20=True):
            assert blk.lattice == ((PHASE[0] - blk.rows.start) % 2, PHASE[1])
            parts.append(obs.compute_index(obs.prepare(blk.data, ["B03", "B11"], 0, "floor0"),
                                           "mndwi", blk.lattice))
        x_blocks = np.concatenate(parts, axis=1)
        assert np.array_equal(x_blocks.view(np.int32), x_full.view(np.int32))


# ─────────────────────────────────────────────────────────────────────────────
#  lattice20
# ─────────────────────────────────────────────────────────────────────────────

def test_lattice20_is_measured(site):
    with inputs.ZarrSource(SITE, root=site["root"], instants=site["instants"]) as src:
        assert src.lattice20 == PHASE
        rep = src.lattice_report
        assert rep["exact"] and rep["agrees_with_grid"] and rep["expected"] == list(PHASE)
        assert rep["bands"] == ["B11", "B12", "B8A", "SCL"]
    assert inputs.expected_lattice20(inputs.Grid(*_coords(), _wkt())) == PHASE


def test_measure_lattice20_rules():
    rng = np.random.default_rng(0)
    for phase in ((0, 0), (0, 1), (1, 0), (1, 1)):
        hb, wb = obs._block_index(22, phase[0])[1], obs._block_index(17, phase[1])[1]
        blk = rng.integers(0, 1000, (hb, wb))
        a = obs.expand20(blk, phase, (22, 17))
        r = inputs.measure_lattice20([("B11", a)])
        assert r["phase"] == list(phase) and r["exact"]
    # not replicated -> None
    r = inputs.measure_lattice20([("B11", rng.integers(0, 1000, (22, 17)))])
    assert r["phase"] is None and not r["exact"]
    # uniform data: both phases exact -> the grid decides, else None
    u = np.full((10, 10), 7)
    assert inputs.measure_lattice20([("SCL", u)], expected=(1, 0))["phase"] == [1, 0]
    assert inputs.measure_lattice20([("SCL", u)])["phase"] is None


# ─────────────────────────────────────────────────────────────────────────────
#  the per-scene offset
# ─────────────────────────────────────────────────────────────────────────────

def _st(**kw):
    base = {"n_px": 100_000, "n_valid": 90_000, "n_low": 0, "n_water": 0, "n_water_low": 0,
            "n_neg": 0}
    base.update(kw)
    return base


def test_offset_rule_branches():
    stats = {
        "2023-01-01": _st(n_neg=3),                                    # rule 1 -> 0
        "2023-01-02": _st(),                                           # weak
        "2023-01-03": _st(n_water=5000, n_water_low=4900),             # rule 3 -> 0
        "2022-01-04": _st(n_water=5000, n_water_low=10),               # rule 3 -> 1000
        "2022-01-05": _st(n_water=5000, n_water_low=1000),             # rule 3 -> unclear
        "2022-01-06": _st(),                                           # weak, year regime 1000
        "2021-01-07": _st(n_water=5000, n_water_low=0),                # baseline < 04 -> unclear
        "2021-01-08": _st(),                                           # baseline < 04 -> 0
        "2020-01-09": _st(n_low=5000),                                 # rule 4 -> 0
        "2020-01-10": _st(n_valid=10),                                 # weak, year state
        "2019-01-11": _st(n_water=5000, n_water_low=0),                # mixed baselines
        "2019-01-12": {},                                              # no stats, year state none
    }
    bl = {"2021-01-07": ["03.01"], "2021-01-08": ["02.09", "03.00"],
          "2022-01-04": ["05.10"], "2019-01-11": ["03.01", "05.00"]}
    tab = inputs.decide_offsets(stats, bl, {"2020": "+1000 in the values"})
    o = tab["offset"]
    assert o["2023-01-01"] == 0 and o["2023-01-03"] == 0
    assert o["2023-01-02"] == 0 and "regime" in tab.loc["2023-01-02", "reason"]
    assert o["2022-01-04"] == 1000 and o["2022-01-06"] == 1000
    assert tab.loc["2022-01-05", "state"] == "unclear"
    assert tab.loc["2021-01-07", "state"] == "unclear" and o["2021-01-08"] == 0
    # 2020: one strong 0 (rule 4) -> the weak scene follows the decided scenes,
    # not the store's year state
    assert o["2020-01-09"] == 0 and o["2020-01-10"] == 0
    assert tab.loc["2019-01-11", "state"] == "unclear"
    assert tab.loc["2019-01-12", "state"] == "unclear"
    assert set(tab["state"]) <= set(inputs.OFFSET_STATES)
    # the year state decides only when no scene of the year is decided
    t2 = inputs.decide_offsets({"2020-01-10": _st(n_valid=10)}, {}, {"2020": "+1000 in the values"})
    assert t2["offset"]["2020-01-10"] == 1000
    # disagreeing decided scenes leave the weak ones unclear
    t3 = inputs.decide_offsets({"2018-01-01": _st(n_water=5000, n_water_low=10),
                                "2018-01-02": _st(n_neg=1), "2018-01-03": _st()})
    assert t3.loc["2018-01-03", "state"] == "unclear"


def test_scene_offset_stats():
    b12 = np.array([[[0, 5, 499, 500], [FILL, 1200, -3, 100]]], np.int16)
    scl = np.array([[[6, 6, 6, 6], [6, 4, 6, 6]]], np.uint8)
    b08 = np.array([[[-1, 2, 3, 4], [5, 6, 7, FILL]]], np.int16)
    st = inputs.scene_offset_stats(b12, scl, [b08])
    assert st["n_valid"][0] == 6                      # 0 and the fill excluded
    assert st["n_low"][0] == 3                        # 5, 499, 100 (not -3, not 500)
    assert st["n_water"][0] == 5 and st["n_water_low"][0] == 3
    assert st["n_neg"][0] == 2 and st["n_px"][0] == 8


def test_source_offsets_provisional_and_frozen(site):
    with _src(site) as src:
        tab = src.offset_table()
        for q, d in enumerate(src.dates):
            want = 0 if d.startswith("2024") else 1000
            assert src.offset(q) == want, (d, tab.loc[d].to_dict())
        assert src.year_states() == {"2024": "no offset", "2025": "+1000 in the values"}
        offs = src.offsets
        assert offs.dtype == np.int64 and set(offs.tolist()) == {0, 1000}
    frozen = {d: (0 if d.startswith("2024") else 1000) for d in site["dates"]}
    frozen[site["dates"][0]] = None
    with _src(site, offsets=frozen) as src:
        with pytest.raises(inputs.OffsetUnclear):
            src.offset(0)
        assert src.offset(1) == 0 and src.offset(src.T - 1) == 1000
        with pytest.raises(inputs.OffsetUnclear):
            _ = src.offsets


def test_offset_makes_the_two_regimes_identical(site):
    """o_s removes the +1000: the 2025 store read with its offsets equals the
    same values delivered without it (floor0 and raw)."""
    d25 = year_bands(2025, offset=True, seed=2)
    d25n = year_bands(2025, offset=False, seed=2)
    for prep in ("floor0", "raw"):
        for b in ("B03", "B11"):
            a = obs.prep_band(d25[b][:4], 1000, prep)
            ref = obs.prep_band(d25n[b][:4], 0, prep)
            m = np.ones(a.shape, bool)
            m[2, 0, 0] = False                  # the negative B08 of the offset-free year
            assert np.array_equal(a[m], ref[m], equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────────
#  fingerprints and grids
# ─────────────────────────────────────────────────────────────────────────────

def test_fingerprints(site, tmp_path):
    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    p = zc.store_path(SITE, 2024, root)
    f0 = inputs.store_fingerprint(p, full=True)
    assert f0 == inputs.store_fingerprint(p, full=True)
    assert f0["fingerprint"] == inputs.store_fingerprint(zc.store_path(SITE, 2024, site["root"]),
                                                         full=True)["fingerprint"]
    with inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE) as src:
        fp = src.fingerprint()
        assert set(fp) == {"2024", "2025"} and fp["2024"]["fingerprint"] == f0["fingerprint"]
    # same size, different bytes: only the content hash sees it
    chunk = p / "B03" / "0.0.0"
    raw = bytearray(chunk.read_bytes())
    raw[-1] ^= 0xFF
    chunk.write_bytes(bytes(raw))
    f1 = inputs.store_fingerprint(p, full=True)
    assert f1["fingerprint"] == f0["fingerprint"] and f1["content_sha256"] != f0["content_sha256"]
    # a size change or a metadata change changes the fingerprint
    chunk.write_bytes(bytes(raw) + b"x")
    assert inputs.store_fingerprint(p)["fingerprint"] != f0["fingerprint"]
    meta = json.loads((p / ".zmetadata").read_text(encoding="utf-8"))
    meta["metadata"][".zattrs"]["site"] = "other"
    (p / ".zmetadata").write_text(json.dumps(meta), encoding="utf-8")
    assert inputs.store_fingerprint(p)["meta_sha256"] != f0["meta_sha256"]


def test_grid_assertions(site, tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    with _src(site) as src:
        g = src.grid
        assert g.same_as(inputs.Grid(*_coords(), _wkt()))
        # the legacy netCDF grid and a GeoTIFF on the same lattice
        import xarray as xr
        with xr.open_dataset(site["legacy"]) as lg:
            assert inputs.grid_of(lg).same_as(g)
        tif = tmp_path / "g.tif"
        with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                           crs="EPSG:32631", transform=from_origin(X_EDGE, Y_EDGE, 10, 10)) as r:
            r.write(np.zeros((1, H, W), np.float32))
        assert inputs.grid_from_geotiff(tif).same_as(g)
        shifted = inputs.Grid(g.xs + 10.0, g.ys, g.crs_wkt)
        with pytest.raises(inputs.GridMismatch):
            inputs.assert_same_grid(g, shifted, "test")
        other_crs = inputs.Grid(g.xs, g.ys, __import__("pyproj").CRS.from_epsg(32630).to_wkt())
        assert not g.same_as(other_crs)
    # a year on another grid is refused
    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    p = zc.store_path(SITE, 2025, root)
    meta = json.loads((p / ".zmetadata").read_text(encoding="utf-8"))
    import zarr
    g2 = zarr.open_group(str(p), mode="r+", zarr_format=2)
    xx = g2["x"][:]
    g2["x"][:] = xx + 10.0
    zarr.consolidate_metadata(str(p))
    del meta
    with pytest.raises(ValueError, match="not on the grid"):
        inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE)


# ─────────────────────────────────────────────────────────────────────────────
#  the legacy netCDF source
# ─────────────────────────────────────────────────────────────────────────────

def test_netcdf_source_equals_the_zarr_store(site):
    inst = {d: t for d, t in site["instants"].items() if d.startswith("2024")}
    with inputs.NetCDFSource(site["legacy"], SITE, inst) as nc, _src(site) as zs:
        assert nc.bands == ("B03", "B08", "SCL") and nc.T == N_PER_YEAR - 1
        assert nc.grid.same_as(zs.grid)
        for q in range(nc.T):
            d = nc.dates[q]
            qz = zs.dates.index(d)
            a, b = nc.read_scene(q, ["B03", "B08", "SCL"]), zs.read_scene(qz, ["B03", "B08", "SCL"])
            for k in a:
                assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])
            assert nc.offset(q) == 0
        assert nc.position(0) == (None, 0)
        with pytest.raises(KeyError):
            nc.read_scene(0, ["B11"])
        assert nc.lattice20 == PHASE                          # measured on SCL alone
        fp = nc.fingerprint()
        assert fp == nc.fingerprint() and fp["bytes"] == os.path.getsize(site["legacy"])
        sub = inputs.NetCDFSource(site["legacy"], SITE, inst, start="2024-03-01",
                                  end="2024-05-01", lattice=None)
        assert all("2024-03-01" <= d < "2024-05-01" for d in sub.dates) and sub.T > 0
        sub.close()


def test_netcdf_labels_need_a_path(site):
    with pytest.raises(ValueError, match="labels_path"):
        inputs.NetCDFSource(site["legacy"], SITE, "labels")


# ─────────────────────────────────────────────────────────────────────────────
#  the scene metadata sidecar
# ─────────────────────────────────────────────────────────────────────────────

def _feature(fid, dt, tile, orbit, pb, platform="sentinel-2a", cc=10.0):
    return {"id": fid, "bbox": [3.0, 51.0, 4.0, 52.0],
            "properties": {"datetime": dt, "grid:code": f"MGRS-{tile}", "platform": platform,
                           "processing:version": pb, "eo:cloud_cover": cc,
                           "sat:relative_orbit": orbit, "sat:absolute_orbit": 1000 + orbit,
                           "sat:orbit_state": "descending", "view:sun_azimuth": 151.48,
                           "view:sun_elevation": 59.1, "view:azimuth": 130.9,
                           "view:incidence_angle": 3.4,
                           "product:timeliness_category": "NRT"}}


def test_scene_meta_from_items(tmp_path):
    feats = [_feature("a", "2025-06-02T11:21:31.025Z", "31UES", 37, "05.11", cc=30.0),
             _feature("b", "2025-06-02T11:21:31.025Z", "31UET", 37, "05.11", cc=10.0),
             _feature("c", "2025-06-02T10:51:00Z", "31UES", 8, "05.11", "sentinel-2b"),
             _feature("d", "2024-01-05T10:50:00Z", "31UES", 8, "05.10"),
             _feature("e", "2024-01-05T10:50:00Z", "31UES", 8, "05.00")]
    m = inputs.scene_meta_from_items(feats, site=SITE, query={"collection": "sentinel-2-l2a"})
    assert list(m["dates"]) == ["2024-01-05", "2025-06-02"]
    d = m["dates"]["2025-06-02"]
    assert d["relative_orbits"] == [8, 37] and d["double_overpass"]
    assert d["platforms"] == ["sentinel-2a", "sentinel-2b"] and d["tiles"] == ["31UES", "31UET"]
    assert d["items"][0]["sun_azimuth"] == 151.48 and d["n_items"] == 3
    assert d["cloud_cover_mean"] == round((30 + 10 + 10) / 3, 4)
    e = m["dates"]["2024-01-05"]
    assert e["processing_baselines"] == ["05.00", "05.10"] and e["duplicate_tiles"] == ["31UES"]
    p = tmp_path / "scene_meta.json"
    inputs.write_scene_meta(p, m)
    assert inputs.read_scene_meta(p) == json.loads(json.dumps(m))
    assert b"\r\n" not in p.read_bytes()
    p.write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(inputs.InputsError):
        inputs.read_scene_meta(p)


def test_scene_meta_baselines_feed_the_offset_rule(site, tmp_path):
    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    feats = [_feature(f"f{i}", f"{d}T10:50:00Z", "31UES", 8, "05.10")
             for i, d in enumerate(site["dates"])]
    inputs.write_scene_meta(inputs.scene_meta_path(SITE, root),
                            inputs.scene_meta_from_items(feats, SITE))
    with inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE) as src:
        assert src.scene_meta is not None
        assert src.baselines()[site["dates"][0]] == ["05.10"]
        # 2024: baseline 05.10 but values without the offset (harmonised): 0
        assert src.offset(0) == 0 and src.offset(src.T - 1) == 1000


# ─────────────────────────────────────────────────────────────────────────────
#  store checks, the angles diagnostic and the inputs report
# ─────────────────────────────────────────────────────────────────────────────

def test_check_store(site, tmp_path):
    c = inputs.check_store(zc.store_path(SITE, 2024, site["root"]))
    assert c["ok"], c["problems"]
    assert c["offset_state"] == "no offset" and c["radiometry_band"] == "B12"
    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    p = zc.store_path(SITE, 2024, root)
    meta = json.loads((p / ".zmetadata").read_text(encoding="utf-8"))
    meta["metadata"]["B03/.zattrs"]["scale_factor"] = 0.0001
    meta["metadata"][".zattrs"]["complete"] = 0
    (p / ".zmetadata").write_text(json.dumps(meta), encoding="utf-8")
    c = inputs.check_store(p)
    assert not c["ok"]
    assert any("scale_factor" in x for x in c["problems"])
    assert any("complete" in x for x in c["problems"])


def test_angles_diagnostic(site):
    with _src(site) as src:
        df = src.angles()
        assert list(df.index) == src.dates
        d25 = [d for d in src.dates if d.startswith("2025")]
        assert df.loc[d25, "sunZenithAngles"].notna().all()
        assert df.loc[[d for d in src.dates if d.startswith("2024")], "sunZenithAngles"].isna().all()
        i = [str(t.date()) for t in _dates(2025)].index(d25[2])
        assert df.loc[d25[2], "sunZenithAngles"] == pytest.approx(40.0 + i)


def test_inputs_report(site, tmp_path):
    out = tmp_path / "inputs_report.json"
    rep = inputs.inputs_report(SITE, root=site["root"], legacy=site["legacy"],
                               instants=site["instants"], disk_paths=[str(tmp_path)], out=out)
    assert rep["ok"], rep["problems"]
    assert rep["years"] == [2024, 2025] and rep["record"]["T"] == 34
    assert rep["grid"]["legacy_equal"] is True
    assert rep["lattice20"]["phase"] == list(PHASE)
    assert set(rep["fingerprints"]) == {"2024", "2025"}
    # 3 x 3 blocks of class 11 at phase (1, 0): rows {0}, {1, 2}, {3, 4} x 6 columns
    assert rep["scl11_per_year"] == {"2024": 30, "2025": 30}
    offs = rep["offsets"]
    assert all(v == (0 if d.startswith("2024") else 1000) for d, v in offs.items())
    # the cloudy scene of 2025 (no water) follows its year's decided scenes
    cloudy = [str(t.date()) for t in _dates(2025)][3]
    assert rep["offset_decisions"][cloudy]["strong"] is False
    assert rep["g_zarr"]["2024"]["verdict"] == "no verify.json"
    assert rep["scene_meta"]["present"] is False and rep["warnings"]
    assert "free_gb" in rep["disk"] and rep["disk"]["read"]["mb"] > 0
    assert inputs.frozen_offsets(rep) == offs
    assert rep["instants"]["source"] == "given" and "sensing" not in rep
    # the file holds the site and is merged by a second site's call
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["format"] == inputs.INPUTS_REPORT_FORMAT and SITE in doc["sites"]
    doc["sites"]["ems"] = {"site": "ems"}
    out.write_text(json.dumps(doc), encoding="utf-8")
    inputs.inputs_report(SITE, root=site["root"], instants=site["instants"], scan=False, out=out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert set(doc["sites"]) == {SITE, "ems"}
    # frozen offsets refuse an unclear scene
    bad = dict(rep, offsets=dict(offs, **{cloudy: None}))
    with pytest.raises(inputs.OffsetUnclear):
        inputs.frozen_offsets(bad)


# ─────────────────────────────────────────────────────────────────────────────
#  the catalogue instants (user decision of 2026-10-14) and the legacy report
# ─────────────────────────────────────────────────────────────────────────────

def _catalogue_root(site, tmp_path):
    """A copy of the stores with overpass_times.json (the catalogue times)
    and, beside them, a sensing_times.json 8 minutes later."""
    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    lab = {d: t for d, t in site["instants"].items() if t is not None}
    (root / SITE / "overpass_times.json").write_text(json.dumps(lab), encoding="utf-8")
    sens = {"dates": {d: {"instant": (pd.Timestamp(t) + pd.Timedelta(minutes=8)).isoformat()}
                      for d, t in lab.items()}, "verdict": "pass"}
    sp = root / SITE / inputs.SENSING_NAME
    sp.write_text(json.dumps(sens), encoding="utf-8")
    return root, lab, sp


def test_catalogue_instants_are_the_default_and_a_sensing_file_only_when_given(site, tmp_path):
    root, lab, sp = _catalogue_root(site, tmp_path)
    with inputs.ZarrSource(SITE, root=root, lattice=PHASE) as src:
        assert src.instant_source == "catalogue" and src.instant_path.endswith(
            "overpass_times.json")
        assert src.T == len(lab)
        assert all(t == pd.Timestamp(lab[d]).tz_localize(None) for d, t in
                   zip(src.dates, src.instants))
    with inputs.ZarrSource(SITE, root=root, lattice=PHASE, instants="labels") as src:
        assert src.instant_source == "catalogue"
    with inputs.ZarrSource(SITE, root=root, lattice=PHASE, instants=str(sp)) as src:
        assert src.instant_source == "sensing_times"
        d0 = src.dates[0]
        shift = src.instants[0] - pd.Timestamp(lab[d0]).tz_localize(None)
        assert shift == pd.Timedelta(minutes=8)
    for mode in ("sensing", "auto"):
        with pytest.raises(ValueError, match="no longer a mode"):
            inputs.ZarrSource(SITE, root=root, lattice=PHASE, instants=mode)
    rep = inputs.inputs_report(SITE, root=root, scan=False)
    assert rep["instants"]["source"] == "catalogue" and rep["instants"]["sha256"]
    assert not any("sensing" in w or "label" in w for w in rep["warnings"])
    (root / SITE / "overpass_times.json").unlink()
    with pytest.raises(FileNotFoundError, match="no catalogue times"):
        inputs.ZarrSource(SITE, root=root, lattice=PHASE)


def test_legacy_inputs_report(site, tmp_path):
    lab = {d: t for d, t in site["instants"].items() if t is not None and d.startswith("2024")}
    lp = tmp_path / "overpass_times.json"
    lp.write_text(json.dumps(lab), encoding="utf-8")
    out = tmp_path / "report.json"
    rep = inputs.legacy_inputs_report(SITE, site["legacy"], lp, 10, out=out)
    assert rep["ok"], rep["problems"]
    assert rep["source"]["kind"] == "netcdf" and rep["source"]["bands"] == ["B03", "B08", "SCL"]
    assert rep["record"]["T"] == len(lab) and rep["instants"]["source"] == "catalogue"
    assert rep["lattice20"]["phase"] == list(PHASE)
    assert set(rep["offsets"].values()) == {0}
    assert rep["scl11_per_year"] == {"2024": 30}
    assert rep["fingerprints"]["netcdf"]["bytes"] == os.path.getsize(site["legacy"])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert SITE in doc["sites"] and doc["format"] == inputs.INPUTS_REPORT_FORMAT
    bad = inputs.legacy_inputs_report(SITE, site["legacy"], lp, 20, scan=False)
    assert not bad["ok"] and any("pixel size" in p for p in bad["problems"])
    assert bad["lattice20"]["phase"] is None
    miss = inputs.legacy_inputs_report(SITE, site["legacy"], None, 10, scan=False)
    assert not miss["ok"] and any("labels_path" in p for p in miss["problems"])


def test_shared_report_merge_is_locked_and_a_dead_holder_is_taken_over(tmp_path):
    """Two 'inputs' processes never drop each other's site: the read-merge-
    write of the shared report holds an O_EXCL lock; a lock left by a dead
    process of this host is taken over, a live one is waited for."""
    import socket
    out = str(tmp_path / "common" / "inputs_report.json")
    inputs._merge_report(out, "escalda", {"ok": True})
    inputs._merge_report(out, "ems", {"ok": False})
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)
    assert set(doc["sites"]) == {"escalda", "ems"} and not os.path.exists(out + ".lock")
    with open(out + ".lock", "w", encoding="utf-8") as f:
        json.dump({"host": socket.gethostname(), "pid": 999999999}, f)
    inputs._merge_report(out, "wadden", {"ok": True})             # dead holder: taken over
    assert not os.path.exists(out + ".lock")
    with open(out + ".lock", "w", encoding="utf-8") as f:
        json.dump({"host": socket.gethostname(), "pid": os.getpid()}, f)
    with pytest.raises(inputs.InputsError, match="held by another process"):
        with inputs.ReportLock(out, timeout=0.6):
            pass
    os.remove(out + ".lock")


def test_disk_check_reads_one_store_band_lazily(tmp_path):
    d = tmp_path / "B08"
    for k in range(5):
        (d / str(k)).mkdir(parents=True)
        (d / str(k) / "0").write_bytes(b"\0" * 400_000)
    r = inputs.disk_check([str(tmp_path)], read_from=str(d), read_mb=1)
    assert r["read"]["files"] == 3 and r["read"]["mb"] == 1.2
    assert isinstance(r["free_gb"][str(tmp_path)], float)
