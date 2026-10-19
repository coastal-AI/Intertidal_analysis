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
and its provisional and frozen forms, with the amendment of 2026-10-19
(rule 0, identity with the offset-harmonised legacy cube from the
store-year's verify.json, per date, self-checked per year; the water test
on CLEAR water, CLD <= 10 %, under a clear sky; dark pixels first; the
regime step that demotes an isolated '+1000' and flags contradictions; the
overcast scenes of the first run on the complete stores; the fallback of
statistics written before it; the rule's sha256 and prepare's refusal of a
report of another rule; the summary lines to paste back); fingerprints;
grid assertions (years, legacy netCDF, GeoTIFF); the NetCDF source; the
scene metadata sidecar from STAC features; the store checks and the inputs
report. The scene instants are the catalogue times by default (user
decision of 2026-10-14): a sensing_times.json beside the stores is never
picked up, only an explicit path is read; the legacy-cube report of the
local 20 m runs.

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

def _st(old=False, **kw):
    """Scene statistics. Since the amendment of 2026-10-19 the water test
    reads the CLEAR-water counts: unless given, they equal the SCL-water
    counts here (all water clear). ``old=True``: statistics written before
    the amendment (no clear-water counts). No SCL cloud counts unless given
    (:func:`_sky`): the cloud guard then does not apply."""
    base = {"n_px": 100_000, "n_valid": 90_000, "n_low": 0, "n_water": 0, "n_water_low": 0,
            "n_neg": 0}
    base.update(kw)
    if not old:
        base.setdefault("n_water_clear", base["n_water"])
        base.setdefault("n_water_clear_low", base["n_water_low"])
    return base


def _sky(cloud_share, n=100_000):
    """The SCL cloud counts of a frame of ``n`` SCL-valid pixels with the
    given SCL 8/9/10 share."""
    return {"n_scl_cloud": int(round(cloud_share * n)), "n_scl_valid": n}


def test_offset_rule_branches():
    """Every branch of the amended rule (2026-10-19), one year per regime;
    the rules are tested in the order (0), (1), (4), (2), (3), then (5)."""
    stats = {
        # 2023: strong 0 by (1), (3) and (4); the weak scenes follow them
        "2023-01-01": _st(n_neg=300),                                   # (1) -> 0
        "2023-01-02": _st(),                                            # weak -> regime 0
        "2023-01-03": _st(n_water=5000, n_water_low=4900),              # (3) dark -> 0
        "2023-01-04": _st(n_water=5000, n_water_low=600),               # (3) share 0.12 -> 0
        "2023-01-05": _st(n_water=5000, n_water_low=300),               # (3) share 0.06: weak
        "2023-01-06": _st(n_low=180),                                   # (4) share 0.002 -> 0
        "2023-01-07": _st(n_low=179),                                   # (4) below 0.002: weak
        "2023-01-08": _st(n_neg=99),                                    # (1) needs 100: weak
        # 2022: +1000 on clear water under a clear sky; the weak scenes follow
        "2022-01-04": _st(n_water=5000, n_water_low=10, **_sky(0.2)),   # (3) -> 1000
        "2022-01-05": _st(n_water=5000, n_water_low=10, **_sky(0.8)),   # under cloud: weak
        "2022-01-06": _st(),                                            # weak, regime 1000
        # 2021: baselines < 04.00
        "2021-01-07": _st(n_water=5000, n_water_low=0),                 # (2) vs water: unclear
        "2021-01-08": _st(),                                            # (2) -> 0
        "2021-01-09": _st(n_water=5000, n_water_low=0, **_sky(0.9)),    # under cloud: (2) -> 0
        # 2020: (4), then a weak scene that follows it (not the store's year state)
        "2020-01-09": _st(n_low=5000),                                  # (4) -> 0
        "2020-01-10": _st(n_valid=10),                                  # weak, regime 0
        # 2019: mixed baselines with bright clear water; no evidence at all
        "2019-01-11": _st(n_water=5000, n_water_low=0),                 # unclear
        "2019-01-12": {},                                               # no stats: unclear
    }
    bl = {"2021-01-07": ["03.01"], "2021-01-08": ["02.09", "03.00"], "2021-01-09": ["N0301"],
          "2022-01-04": ["05.10"], "2019-01-11": ["03.01", "05.00"]}
    tab = inputs.decide_offsets(stats, bl, {"2020": "+1000 in the values"})
    o, rule = tab["offset"], tab["rule"]
    assert set(tab["state"]) <= set(inputs.OFFSET_STATES)
    for d, tag in (("2023-01-01", "negatives"), ("2023-01-03", "dark water"),
                   ("2023-01-04", "dark water"), ("2023-01-06", "frame")):
        assert o[d] == 0 and tab.loc[d, "strong"] and rule[d] == tag, d
    for d in ("2023-01-02", "2023-01-05", "2023-01-07", "2023-01-08"):
        assert o[d] == 0 and not tab.loc[d, "strong"] and rule[d] == "regime", d
    assert tab.loc["2023-01-02", "reason"] == "the regime of the year's decided scenes (0)"
    assert tab.loc["2023-01-05", "reason"].startswith(
        "the regime of the year's decided scenes (0); clear water B12 partly below 500 DN "
        "(share 0.060 of 5000 px")
    assert (o["2022-01-04"], o["2022-01-05"], o["2022-01-06"]) == (1000, 1000, 1000)
    assert rule["2022-01-04"] == "bright water" and tab.loc["2022-01-04", "strong"]
    assert rule["2022-01-05"] == "regime" and "under cloud" in tab.loc["2022-01-05", "reason"]
    assert tab.loc["2021-01-07", "state"] == "unclear" and rule["2021-01-07"] == "baseline vs water"
    assert o["2021-01-08"] == 0 and o["2021-01-09"] == 0 and rule["2021-01-09"] == "baseline"
    assert o["2020-01-09"] == 0 and o["2020-01-10"] == 0 and rule["2020-01-10"] == "regime"
    assert tab.loc["2019-01-11", "state"] == "unclear" and rule["2019-01-11"] == "mixed baselines"
    assert tab.loc["2019-01-12", "state"] == "unclear" and rule["2019-01-12"] == "no evidence"
    # the reasons keep the 'kind (details)' shape
    assert tab.loc["2023-01-01", "reason"] == \
        "negative values in the delivered bands (300 px >= 100): no offset"
    assert tab.loc["2023-01-03", "reason"].startswith("clear water B12 below 500 DN (share 0.980")
    assert tab.loc["2023-01-06", "reason"] == \
        "frame B12 below 500 DN (share 0.0020 of 90000 valid px >= 0.002)"
    assert tab.loc["2022-01-04", "reason"] == (
        "clear water B12 at or above 500 DN (share 0.002 of 5000 px with CLD <= 10 %; frame SCL "
        "8/9/10 share 0.200)")
    assert tab.loc["2021-01-08", "reason"] == "processing baseline < 04.00"
    # the year state decides only when no scene of the year is decided
    t2 = inputs.decide_offsets({"2020-01-10": _st(n_valid=10)}, {}, {"2020": "+1000 in the values"})
    assert t2["offset"]["2020-01-10"] == 1000 and t2["rule"]["2020-01-10"] == "year state"


def test_dark_pixels_outrank_bright_water():
    """(4) before (3), and the thresholds of the amendment: a frame with dark
    B12 (impossible with +1000) is 0 even when its clear water is bright;
    one negative pixel is not enough; a middle water share is no strong
    evidence, never 'unclear' by itself."""
    glint = dict(n_water=20_000, n_water_low=100, **_sky(0.01))     # bright clear water
    assert inputs._strong(_st(n_low=50_000, **glint), "offset")[0] == 0         # 5 % dark
    assert inputs._strong(_st(n_low=179, **glint), "offset")[0] == 1000         # < 0.002
    assert inputs._strong(_st(n_low=180, **glint), "offset")[0] == 0            # 0.002
    assert inputs._strong(_st(n_neg=1, **glint), "offset")[0] == 1000           # 1 negative
    assert inputs._strong(_st(n_neg=100, **glint), "offset")[0] == 0
    # a frame below 1 % valid is no frame evidence
    assert inputs._strong(_st(n_valid=900, n_low=900), "offset") == (None, "no strong evidence")
    # the middle band: 0.3 -> 0 (>= 0.1), 0.05 -> weak; with dark frame pixels -> 0 by (4)
    assert inputs._strong(_st(n_water=20_000, n_water_low=6000), "offset")[0] == 0
    assert inputs._strong(_st(n_water=20_000, n_water_low=1000), "offset")[0] is None
    assert inputs._strong(_st(n_low=27_000, n_water=20_000, n_water_low=1000), "offset")[0] == 0
    tab = inputs.decide_offsets({"2025-03-01": _st(n_water=20_000, n_water_low=1000),
                                 "2025-03-02": _st(n_low=27_000)}, {}, {"2025": "no offset"})
    assert not (tab["state"] == "unclear").any() and (tab["offset"] == 0).all()
    # a mixed-baseline date with dark frame pixels is 0 by (4), as by negatives in V1
    assert inputs._strong(_st(n_low=20_000, n_water=40_000, n_water_low=0), "mixed")[0] == 0
    assert inputs._strong(_st(n_water=40_000, n_water_low=0), "mixed")[0] == "unclear"


def test_bright_water_under_cloud_is_no_strong_offset():
    """The cloud guard of (3) (independent of CLD): bright clear water in a
    frame whose SCL 8/9/10 share is above 0.5 is no '+1000' evidence; at or
    below 0.5 it is; without SCL counts the guard does not apply (declared in
    the reason)."""
    bright = dict(n_water=20_000, n_water_low=0)
    val, why = inputs._strong(_st(**bright, **_sky(0.51)), "offset")
    assert val is None and why.startswith("clear water B12 at or above 500 DN under cloud (share")
    assert "frame SCL 8/9/10 share 0.510 > 0.5): no strong evidence" in why
    assert inputs._strong(_st(**bright, **_sky(0.5)), "offset")[0] == 1000
    val, why = inputs._strong(_st(**bright), "offset")
    assert val == 1000 and why.endswith("frame SCL 8/9/10 share unknown)")
    # under cloud a pre-04.00 scene is 0 by (2), a mixed one is weak (not unclear)
    assert inputs._strong(_st(**bright, **_sky(0.9)), "none") == (0, "processing baseline < 04.00")
    assert inputs._strong(_st(**bright, **_sky(0.9)), "mixed")[0] is None
    # the counts from an SCL histogram (codes 0..11, then no data)
    sc = [5, 10, 0, 0, 300, 100, 200, 0, 40, 50, 10, 0, 999]
    assert inputs.scl_cloud_counts(sc) == {"n_scl_cloud": 100, "n_scl_valid": 710}
    assert inputs._cloud_share(inputs.scl_cloud_counts(sc)) == pytest.approx(100 / 710)
    assert inputs._cloud_share({}) is None and inputs._cloud_share(_sky(0.3, n=0)) is None


def test_regime_step_demotes_an_isolated_1000_and_flags_contradictions():
    """(5): one false '+1000' no longer spreads to its year. An isolated
    strong 1000 (<= 2 % of the strong decisions, >= 10 strong 0) is demoted
    to weak and follows the regime; otherwise the smaller side is 'unclear'
    (also a minority of strong 0 beside many 1000; both sides on a tie, with
    the weak scenes); a year of one value keeps it."""
    dark = _st(n_low=20_000)                                           # strong 0 (frame)
    glint = _st(n_water=5000, n_water_low=0, **_sky(0.05))            # a strong 1000
    weak = _st(n_valid=10)

    def year(n0, n1, nweak=1, y="2025"):
        days = lambda start, n: [str(t.date()) for t in pd.date_range(start, periods=n)]  # noqa
        st = {d: dark for d in days(f"{y}-01-01", n0)}                # up to 2025-04-10
        st.update({d: glint for d in days(f"{y}-06-01", n1)})
        st.update({d: weak for d in days(f"{y}-12-01", nweak)})
        return st

    # 100 strong 0 and 1 strong 1000: demoted, no 'unclear', the weak scene 0
    tab = inputs.decide_offsets(year(100, 1), {}, {"2025": "+1000 in the values"})
    assert not (tab["state"] == "unclear").any() and (tab["offset"] == 0).all()
    g = "2025-06-01"
    assert tab.loc[g, "rule"] == "demoted" and not tab.loc[g, "strong"]
    assert tab.loc[g, "reason"].startswith("the regime of the year's decided scenes (0); clear "
                                           "water B12 at or above 500 DN (share 0.000")
    assert tab.loc[g, "reason"].endswith("; demoted: 1 strong '+1000' beside the year's 100 "
                                         "strong 0 (<= 2% of 101)")
    assert tab.attrs["demoted"] == [g] and tab.attrs["contradicted"] == []
    assert tab.loc["2025-12-01", "rule"] == "regime"
    # 2 of 102 (<= 2 %) are both demoted; 3 of 103 are not: the three are 'unclear'
    assert len(inputs.decide_offsets(year(100, 2)).attrs["demoted"]) == 2
    tab = inputs.decide_offsets(year(100, 3))
    assert (tab.loc[[f"2025-06-0{k}" for k in (1, 2, 3)], "state"] == "unclear").all()
    assert tab.loc["2025-06-01", "rule"] == "contradiction"
    assert tab.loc["2025-06-01", "reason"].endswith("; contradicts the year's 100 strong 0")
    assert tab.loc["2025-12-01", "offset"] == 0                       # the weak scene: regime 0
    # fewer than 10 strong 0: no demotion, the 1000 is 'unclear'
    tab = inputs.decide_offsets(year(9, 1))
    assert tab.loc["2025-06-01", "state"] == "unclear" and tab.loc["2025-12-01", "offset"] == 0
    # a minority of strong 0 beside many 1000 is 'unclear' (never demoted); the 1000 stand
    tab = inputs.decide_offsets(year(1, 30))
    assert tab.loc["2025-01-01", "state"] == "unclear" and tab.loc["2025-01-01", "rule"] == \
        "contradiction"
    assert tab.loc["2025-01-01", "reason"].endswith("; contradicts the year's 30 strong 1000")
    assert (tab.loc[[f"2025-06-{k:02d}" for k in range(1, 31)], "offset"] == 1000).all()
    assert tab.loc["2025-12-01", "offset"] == 1000
    # a tie: both sides and the weak scenes 'unclear'
    tab = inputs.decide_offsets(year(2, 2))
    assert (tab["state"] == "unclear").all()
    assert tab.loc["2025-12-01", "reason"] == "the year's decided scenes disagree"
    # a year of one value keeps it; a year without strong decisions takes the year state
    tab = inputs.decide_offsets(dict(year(0, 20), **year(0, 0, 2, "2024")), {},
                                {"2024": "no offset"})
    assert (tab.loc[[d for d in tab.index if d[:4] == "2025"], "offset"] == 1000).all()
    assert (tab.loc[[d for d in tab.index if d[:4] == "2024"], "rule"] == "year state").all()
    # the scenes with baselines < 04.00 do not count: a regime change, no contradiction
    st = {f"2022-01-{k + 1:02d}": _st() for k in range(10)}
    st["2022-06-01"] = glint
    bl = {d: ["02.14"] for d in st if d < "2022-06"}
    bl["2022-06-01"] = ["04.00"]
    tab = inputs.decide_offsets(st, bl)
    assert (tab.loc[[d for d in st if d < "2022-06"], "offset"] == 0).all()
    assert tab.loc["2022-06-01", "offset"] == 1000 and tab.loc["2022-06-01", "strong"]


def test_scene_offset_stats():
    b12 = np.array([[[0, 5, 499, 500], [FILL, 1200, -3, 100]]], np.int16)
    scl = np.array([[[6, 6, 6, 6], [6, 4, 6, 6]]], np.uint8)
    b08 = np.array([[[-1, 2, 3, 4], [5, 6, 7, FILL]]], np.int16)
    st = inputs.scene_offset_stats(b12, scl, [b08])
    assert st["n_valid"][0] == 6                      # 0 and the fill excluded
    assert st["n_low"][0] == 3                        # 5, 499, 100 (not -3, not 500)
    assert st["n_water"][0] == 5 and st["n_water_low"][0] == 3
    assert st["n_neg"][0] == 2 and st["n_px"][0] == 8
    assert "n_water_clear" not in st                  # no CLD given: no clear-water counts
    # clear water (amendment of 2026-10-19): SCL 6, valid B12, CLD present and <= 10 %
    cld = np.array([[[0, 10, 11, obs.CLD_NODATA_U8], [3, 0, 50, 5]]], np.uint8)
    st = inputs.scene_offset_stats(b12, scl, [b08], cld_u8=cld)
    assert st["n_water"][0] == 5 and st["n_water_low"][0] == 3       # unchanged
    assert st["n_water_clear"][0] == 2                # 5 (CLD 10) and 100 (CLD 5)
    assert st["n_water_clear_low"][0] == 2            # not 499 (CLD 11), 500 (no CLD), -3 (CLD 50)
    raw = np.array([[[0, 10, 11, FILL], [3, 0, 50, 5]]], np.int16)  # raw CLD: decoded first
    st_raw = inputs.scene_offset_stats(b12, scl, [b08], cld_u8=raw)
    assert all(np.array_equal(st[k], st_raw[k]) for k in st)
    with pytest.raises(ValueError, match="CLD shape"):
        inputs.scene_offset_stats(b12, scl, cld_u8=cld[..., :3])


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
#  the amendment of 2026-10-19: rule (0) identity with the legacy cube, the
#  water test on CLEAR water under a clear sky, dark pixels first, the regime
# ─────────────────────────────────────────────────────────────────────────────

#: wadden 2025-12-15 as the first inputs run on the complete stores saw it (an
#: almost fully overcast winter scene): 7,240 SCL-water px, none below 500 DN;
#: 84 of 9,507,800 valid frame px with B12 in [1, 500); no negative value
WADDEN_1215 = {"n_px": 9_507_800, "n_valid": 9_507_800, "n_low": 84, "n_water": 7240,
               "n_water_low": 0, "n_neg": 0}
#: its SCL cloud counts as the 20 m legacy cube gives them (94.6 % SCL 9, 5.3 %
#: SCL 8): illustrative, the Zarr counts come with the rerun
WADDEN_1215_SKY = {"n_scl_cloud": 9_500_000, "n_scl_valid": 9_507_800}


def _overcast_block(cld):
    """Two scenes of 60 x 50 px: SCL water on 30 columns with a bright B12
    (900 DN, no offset: cloud or haze), cloud (SCL 9) elsewhere, no dark
    pixel; CLD ``cld`` everywhere."""
    b12 = np.full((2, 60, 50), 900, np.int16)
    scl = np.full(b12.shape, 9, np.uint8)
    scl[:, :, :30] = 6
    return b12, scl, np.full(b12.shape, cld, np.uint8)


def _bright_water_block(cld, other_scl):
    """One scene of 100 x 400 px: SCL water on 30 columns (3,000 px) with a
    bright B12 (900 DN) and CLD ``cld``; the rest SCL ``other_scl`` (9: thick
    cloud, 92.5 % of the frame; 4: a clear sky, e.g. glint on the water) with
    B12 1500; no dark pixel, no negative value. Returns its statistics."""
    b12 = np.full((1, 100, 400), 1500, np.int16)
    b12[:, :, :30] = 900
    scl = np.full(b12.shape, other_scl, np.uint8)
    scl[:, :, :30] = 6
    st = inputs.scene_offset_stats(b12, scl, cld_u8=np.full(b12.shape, cld, np.uint8))
    out = {k: int(v[0]) for k, v in st.items()}
    out.update(inputs.scl_cloud_counts(obs.scl_counts(scl[0])))
    return out


def test_overcast_scl_water_is_no_longer_a_strong_offset():
    """(a) the wadden 2025-12-15 case: SCL water that is thick cloud (bright
    B12), a frame almost without dark B12: no strong '+1000' any more, by the
    clear-water filter when CLD is high, and by the cloud guard when it is not
    (the residual risk: Sen2Cor gives its class 6 the low cloud
    probabilities). The V1 rule, reproduced by the pre-amendment statistics,
    made it one."""
    over = dict(WADDEN_1215, n_water_clear=0, n_water_clear_low=0)
    assert inputs._strong(over, "offset") == (None, "no strong evidence")
    val, why = inputs._strong(dict(WADDEN_1215), "offset")          # the V1 form
    assert val == 1000 and why.startswith("water B12 at or above 500 DN (share 0.000 of 7240 "
                                          "SCL-water px; statistics written before the "
                                          "2026-10-19 amendment")
    # CLD did not remove the cloud labelled water: the cloud guard does
    lowcld = dict(WADDEN_1215, n_water_clear=7240, n_water_clear_low=0, **WADDEN_1215_SKY)
    val, why = inputs._strong(lowcld, "offset")
    assert val is None and "under cloud" in why and "share 0.999 > 0.5" in why
    # from the pixels: the overcast water is SCL water but not clear water
    b12, scl, cld = _overcast_block(60)
    st = inputs.scene_offset_stats(b12, scl, cld_u8=cld)
    assert st["n_water"].tolist() == [1800, 1800] and st["n_water_low"].tolist() == [0, 0]
    assert st["n_water_clear"].tolist() == [0, 0] and st["n_low"].tolist() == [0, 0]
    one = {k: int(v[0]) for k, v in st.items()}
    assert inputs._strong(one, "offset") == (None, "no strong evidence")
    # the same with CLD 5 on the water, in a frame of thick cloud: the guard decides
    o5 = _bright_water_block(5, 9)
    assert o5["n_water_clear"] == 3000 and o5["n_water_clear_low"] == 0
    assert inputs._cloud_share(o5) == pytest.approx(0.925)
    assert inputs._strong(o5, "offset")[0] is None
    # alone in its year it follows the store's year state (or stays unclear without one)
    tab = inputs.decide_offsets({"2025-12-15": over}, {}, {"2025": "no offset"})
    assert tab.loc["2025-12-15", "offset"] == 0 and not tab.loc["2025-12-15", "strong"]
    assert inputs.decide_offsets({"2025-12-15": over}).loc["2025-12-15", "state"] == "unclear"


def test_bright_clear_water_under_a_clear_sky_is_left_to_the_identity_and_the_regime():
    """The residual case (code review): SCL 6 with CLD 5 and a bright B12 on
    3,000 px under a clear sky (glint, haze) is a strong '+1000' by its own
    pixels; identical to the legacy cube it is 0 (the reason keeps what the
    pixels say, the report lists it); isolated among its year's strong 0 it is
    demoted; beside a few it is 'unclear'."""
    g = _bright_water_block(5, 4)
    assert inputs._cloud_share(g) == 0.0
    val, why = inputs._strong(g, "offset")
    assert val == 1000 and why == ("clear water B12 at or above 500 DN (share 0.000 of 3000 px "
                                   "with CLD <= 10 %; frame SCL 8/9/10 share 0.000)")
    hard = _st(n_low=20_000)
    stats = {"2025-01-01": hard, "2025-01-02": hard, "2025-12-31": g}
    ident = {"2025-01-01": "B03 and B08 bitwise", "2025-12-31": "B03 and B08 bitwise"}
    tab = inputs.decide_offsets(stats, {}, None, ident)
    assert tab.loc["2025-12-31", "offset"] == 0 and tab.loc["2025-12-31", "rule"] == "identity"
    assert tab.loc["2025-12-31", "reason"].startswith(
        f"{inputs.IDENTITY_REASON} (B03 and B08 bitwise); its pixels alone say +1000: clear water "
        "B12 at or above 500 DN")
    assert tab.attrs["identity_pixels_1000"] == ["2025-12-31"]
    # without the identity: beside 2 strong 0 it contradicts its year
    tab = inputs.decide_offsets(stats)
    assert tab.loc["2025-12-31", "state"] == "unclear"
    assert tab.loc["2025-12-31", "rule"] == "contradiction"
    # beside >= 10 strong 0 (<= 2 %): demoted, weak, 0
    many = {str(t.date()): hard for t in pd.date_range("2025-01-01", periods=60)}
    many["2025-12-31"] = g
    tab = inputs.decide_offsets(many)
    assert tab.loc["2025-12-31", "offset"] == 0 and tab.loc["2025-12-31", "rule"] == "demoted"


def test_a_true_offset_on_clear_water_is_still_decided_1000():
    """(b) a scene delivered with +1000, its water clear (CLD < 8 %) under a
    clear sky: strong 1000, from the pixels to the decision."""
    rng = np.random.default_rng(5)
    b12 = (1000 + rng.integers(10, 150, (2, 60, 50))).astype(np.int16)
    b12[:, :, 30:] = (1000 + rng.integers(900, 2000, (2, 60, 20))).astype(np.int16)
    scl = np.full(b12.shape, 4, np.uint8)
    scl[:, :, :30] = 6
    cld = rng.integers(0, 8, b12.shape).astype(np.uint8)
    st = inputs.scene_offset_stats(b12, scl, cld_u8=cld)
    assert st["n_water_clear"].tolist() == [1800, 1800]
    assert st["n_water_clear_low"].tolist() == [0, 0]
    assert st["n_low"].tolist() == [0, 0] and st["n_neg"].tolist() == [0, 0]
    stats = {f"2025-01-0{k + 1}": dict({key: int(v[k]) for key, v in st.items()},
                                       **inputs.scl_cloud_counts(obs.scl_counts(scl[k])))
             for k in range(2)}
    tab = inputs.decide_offsets(stats, {d: ["05.11"] for d in stats})
    assert (tab["offset"] == 1000).all() and tab["strong"].all()
    assert (tab["rule"] == "bright water").all()
    assert tab["reason"].eq("clear water B12 at or above 500 DN (share 0.000 of 1800 px with "
                            "CLD <= 10 %; frame SCL 8/9/10 share 0.000)").all()
    # the clear-water share bands: >= 0.1 -> 0, between 0.02 and 0.1 no strong evidence
    assert inputs._strong(_st(n_water=5000, n_water_low=500, n_water_clear=2000,
                              n_water_clear_low=200), "offset")[0] == 0           # 0.1
    mid = _st(n_water=5000, n_water_low=500, n_water_clear=2000, n_water_clear_low=150)
    assert inputs._strong(mid, "offset")[0] is None                                # 0.075


def test_identity_with_the_legacy_cube_decides_no_offset_first():
    """(c) B03 and B08 identical to the offset-harmonised legacy cube: 0,
    strong, even when the clear water says +1000, under a baseline < 04.00
    or mixed baselines; never on a frame with B12 valid on < 1 %. A strong
    '+1000' left in a year the identity decides is 'unclear' (code review)."""
    bright = _st(n_water=5000, n_water_low=10)                   # the clear water says +1000
    note = "B03 and B08 bitwise, 2025.verify.json verdict 'identical'"
    stats = {"2025-01-01": bright, "2025-01-02": bright, "2025-01-03": bright,
             "2025-01-04": bright, "2025-01-05": _st(n_valid=0, n_low=0),
             "2025-01-06": _st(n_low=5000), "2025-01-07": _st(n_valid=900, n_low=0)}
    bl = {"2025-01-02": ["03.01"], "2025-01-03": ["03.01", "05.10"]}
    ident = {d: note for d in ("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-05",
                               "2025-01-06", "2025-01-07")}
    tab = inputs.decide_offsets(stats, bl, None, ident)
    assert tab.attrs["identity_years"]["2025"].startswith("identity used: 1 of its 6 identical")
    for d in ("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"):
        assert tab.loc[d, "offset"] == 0 and tab.loc[d, "strong"], d
        assert tab.loc[d, "reason"].startswith(f"{inputs.IDENTITY_REASON} ({note})")
        assert tab.loc[d, "rule"] == "identity"
    assert "its pixels alone say +1000" in tab.loc["2025-01-01", "reason"]
    assert "its pixels alone say unclear" in tab.loc["2025-01-02", "reason"]
    # not identical, bright clear water, beside the year's identity: 'unclear'
    assert tab.loc["2025-01-04", "state"] == "unclear"
    assert tab.loc["2025-01-04", "reason"].endswith(
        "; contradicts the year's 3 strong 0, 3 of them by the identity with the legacy cube")
    # no valid B12 pixel, or B12 valid on < 1 % of the frame: the identity proves nothing
    for d in ("2025-01-05", "2025-01-07"):
        assert not tab.loc[d, "strong"] and tab.loc[d, "offset"] == 0          # the regime
        assert tab.loc[d, "rule"] == "regime"
    # without the identity: 1000 / unclear (baseline < 04.00) / unclear (mixed baselines); the
    # one strong 0 left (01-06, dark frame) is the smaller side against two strong 1000
    t0 = inputs.decide_offsets(stats, bl)
    assert t0.loc["2025-01-02", "state"] == t0.loc["2025-01-03", "state"] == "unclear"
    assert t0.loc["2025-01-01", "offset"] == t0.loc["2025-01-04", "offset"] == 1000
    assert t0.loc["2025-01-06", "state"] == "unclear"
    assert t0.loc["2025-01-06", "rule"] == "contradiction"
    # an iterable of dates is accepted (default note): a list, a numpy array, a pandas Index
    hard = {"2025-01-01": _st(n_low=5000)}
    for it in (["2025-01-01"], np.array(["2025-01-01"]), pd.Index(["2025-01-01"])):
        t1 = inputs.decide_offsets(hard, identical=it)
        assert t1.loc["2025-01-01", "reason"] == f"{inputs.IDENTITY_REASON} (verify.json)"
    assert inputs._identity_map(None) == {} and inputs._identity_map([]) == {}


def test_identity_is_used_only_where_an_identical_date_shows_dark_or_negative_values():
    """(0) checks itself per store-year: the legacy cubes come from the same
    backend, so identity proves only o_s(Zarr) = o_s(legacy). A year whose
    identical dates show no value impossible with +1000 (negatives, dark
    frame pixels, dark clear water) does not use it; one such date is
    enough."""
    bright = _st(n_water=5000, n_water_low=0, **_sky(0.1))
    cloudy = _st(n_valid=10)
    stats = {"2024-03-01": bright, "2024-03-02": cloudy, "2024-03-03": bright}
    ident = dict.fromkeys(stats, "B03 and B08 bitwise")
    tab = inputs.decide_offsets(stats, {}, {"2024": "+1000 in the values"}, ident)
    assert tab.attrs["identity_years"]["2024"].startswith("identity NOT used: none of its 3")
    assert "both cubes could carry the +1000" in tab.attrs["identity_years"]["2024"]
    assert not (tab["rule"] == "identity").any() and (tab["offset"] == 1000).all()
    for hard, tag in ((_st(n_neg=500), "negatives"), (_st(n_low=900), "frame"),
                      (_st(n_water=4000, n_water_low=3000), "dark water")):
        tab = inputs.decide_offsets(dict(stats, **{"2024-03-04": hard}), {}, None,
                                    dict(ident, **{"2024-03-04": "x"}))
        assert tab.attrs["identity_years"]["2024"].startswith("identity used: 1 of its 4"), tag
        assert (tab["offset"] == 0).all() and tab.loc["2024-03-01", "rule"] == "identity", tag
    # another year's dark date does not lend its proof
    tab = inputs.decide_offsets(dict(stats, **{"2025-01-01": _st(n_low=900)}), {}, None,
                                dict(ident, **{"2025-01-01": "x"}))
    assert tab.attrs["identity_years"]["2024"].startswith("identity NOT used")
    assert tab.attrs["identity_years"]["2025"].startswith("identity used")


STORE4 = ["2025-01-01", "2025-01-10", "2025-01-20", "2025-01-30"]


def _grid(store=(20, 20), rows=(0, 20), cols=(0, 20), identical=True):
    """The ``grid`` check of a verify.json: the store's shape and the common
    window in store rows and columns."""
    return {"store": {"shape": list(store)}, "reference": {"shape": [20, 20]},
            "crs_equal": True, "identical": identical,
            "common_window": {"store_rows": list(rows), "store_cols": list(cols),
                              "reference_rows": [0, rows[1] - rows[0]],
                              "reference_cols": [0, cols[1] - cols[0]]}}


def _verify(**kw):
    """A 2025 verify.json of ``zarr_cubes.compare_with_netcdf`` over the 4
    dates of STORE4 (identical grids): 2025-01-10 differs in B03 and B08,
    2025-01-20 in SCL only; ``kw`` replaces top-level keys."""
    v = {"site": SITE, "year": 2025, "verdict": "differs", "grid": _grid(),
         "window": ["2025-01-01", "2026-01-01"], "reference_period": ["2023-01-01", "2025-12-31"],
         "dates": {"store": 4, "reference": 4, "common": 4, "only_store": [],
                   "only_reference": [], "outside_reference_period": []},
         "bands_compared": ["B03", "B08", "SCL"],
         "bands": {b: {"pixels": 400, "equal": 395, "value_differs": 5, "nodata_only_store": 0,
                       "nodata_only_reference": 0} for b in ("B03", "B08", "SCL")},
         "dates_with_differences": 2,
         "dates_with_differences_listed": {"2025-01-10": {"B03": 5, "B08": 5},
                                           "2025-01-20": {"SCL": 5}}}
    v.update(kw)
    return v


def test_identity_from_verify_per_date():
    """(d) the per-date identity from a verify.json: a complete list of
    differing dates grants the unlisted dates; a truncated one does not; a
    listed date differing in SCL only is identical in B03 and B08; the
    comparison must have covered the store's whole frame; the verification
    is bound to the store by its date counts and, when it kept them, its
    per-date B12 counts."""
    ident, note = inputs.identity_from_verify(_verify(), STORE4, 2025)
    assert sorted(ident) == ["2025-01-01", "2025-01-20", "2025-01-30"]
    assert "SCL only" in ident["2025-01-20"] and "not among the 2" in ident["2025-01-01"]
    assert "3 of 4 common dates" in note and "complete" in note
    assert note.endswith("(bound to the store by its date counts)")
    # truncated at max_dates_listed (3 dates differ, 2 listed): no identity for the unlisted
    ident, note = inputs.identity_from_verify(_verify(dates_with_differences=3), STORE4, 2025)
    assert sorted(ident) == ["2025-01-20"] and "truncated (2 of 3)" in note
    ident, _ = inputs.identity_from_verify(_verify(dates_with_differences=None), STORE4, 2025)
    assert sorted(ident) == ["2025-01-20"]                        # unknown count: truncated
    # no B03 or B08 difference over all dates (SCL only): every common date, even truncated
    clean = {b: {"pixels": 400, "equal": 400, "value_differs": 0, "nodata_only_store": 0,
                 "nodata_only_reference": 0} for b in ("B03", "B08")}
    clean["SCL"] = {"pixels": 400, "equal": 390, "value_differs": 10, "nodata_only_store": 0,
                    "nodata_only_reference": 0}
    ident, _ = inputs.identity_from_verify(
        _verify(bands=clean, dates_with_differences=7,
                dates_with_differences_listed={"2025-01-20": {"SCL": 10}}), STORE4, 2025)
    assert sorted(ident) == STORE4
    # 'identical': every common date; a store date outside the reference period never
    ok = _verify(verdict="identical", dates_with_differences=0, dates_with_differences_listed={},
                 reference_period=["2023-01-01", "2025-01-30"],
                 dates={"store": 4, "reference": 3, "common": 3, "only_store": [],
                        "only_reference": [], "outside_reference_period": ["2025-01-30"]})
    ident, note = inputs.identity_from_verify(ok, STORE4, 2025)
    assert sorted(ident) == STORE4[:3] and "3 common dates" in note
    # a date of the store missing from the reference is not common
    miss = dict(ok, dates=dict(ok["dates"], common=2, only_store=["2025-01-10"]))
    assert sorted(inputs.identity_from_verify(miss, STORE4, 2025)[0]) == ["2025-01-01",
                                                                          "2025-01-20"]
    # the per-date B12 counts the downloader keeps bind the verification to the store
    rad = {"per_date_valid_low": {d: [1000, 400 + k] for k, d in enumerate(STORE4)}}
    bound = dict(ok, radiometry=rad)
    ident, note = inputs.identity_from_verify(bound, STORE4, 2025, rad)
    assert sorted(ident) == STORE4[:3] and "per-date B12 counts" in note
    other = {"per_date_valid_low": dict(rad["per_date_valid_low"], **{STORE4[2]: [1000, 7]})}
    ident, note = inputs.identity_from_verify(bound, STORE4, 2025, other)
    assert ident == {} and "per-date B12 counts are not the store's" in note
    assert sorted(inputs.identity_from_verify(bound, STORE4, 2025, None)[0]) == STORE4[:3]
    # the comparison must have covered the store's whole frame: identical grids, or a common
    # window that is the whole store frame (a store inside the legacy cube, e.g. an extract);
    # the per-date counts cover the common window only
    inside = dict(ok, verdict="identical on the common window",
                  grid=_grid(store=(10, 12), rows=(0, 10), cols=(0, 12), identical=False))
    assert inputs._covers_store(ok) and inputs._covers_store(inside)
    assert sorted(inputs.identity_from_verify(inside, STORE4, 2025)[0]) == STORE4[:3]
    beyond = _grid(store=(24, 20), rows=(2, 22), cols=(0, 20), identical=False)
    for v in (dict(ok, verdict="identical on the common window", grid=beyond),
              _verify(grid=beyond),                                     # 'differs' as well
              {k: x for k, x in ok.items() if k != "grid"},             # no grid check
              dict(ok, grid=dict(_grid(), crs_equal=False, identical=False))):
        ident, note = inputs.identity_from_verify(v, STORE4, 2025)
        assert ident == {} and "did not cover the store's whole frame" in note, note
    # nothing is granted ...
    for bad, why in (
            (dict(ok, dates=dict(ok["dates"], store=5)), "another download"),   # stale
            (_verify(verdict="not comparable"), "verdict 'not comparable'"),
            (_verify(verdict="no reference"), "verdict 'no reference'"),
            (dict(ok, year=2024), "for the year 2024"),
            (dict(ok, bands_compared=["B03", "SCL"]), "not both compared"),
            (dict(ok, bands={b: dict(s, equal=0) for b, s in clean.items()}), "no valid"),
            (dict(ok, dates=dict(ok["dates"], duplicated_dates={"store": True})), "duplicated"),
            (dict(ok, window=None), "no window"),
            (dict(ok, dates=["2025-01-01"]), "malformed"),
            (_verify(dates_with_differences_listed=["2025-01-10"]), "malformed"),
            (dict(bound, radiometry={"per_date_valid_low": {STORE4[0]: "x"}}), "not the store's"),
            ("not a dict", "not a JSON object"),
            (None, "no verify.json"),
            ({"verdict": "unreadable verify.json", "error": "x"}, "unreadable")):
        ident, note = inputs.identity_from_verify(bad, STORE4, 2025, rad)
        assert ident == {} and why in note, (why, note)


def test_a_year_of_no_offset_with_overcast_scenes_has_no_unclear():
    """(e) a year whose decided scenes are all 0, plus overcast scenes (bright
    SCL water that is cloud) and a nearly empty one: no 'unclear', every scene
    0, the overcast ones weak. The same year read with pre-amendment
    statistics (no clear-water counts, no SCL cloud counts) puts the overcast
    scenes back as strong '+1000' evidence; the regime step now flags exactly
    them (the contradiction), not every weak scene of the year as V1 did."""
    stats = {f"2025-0{m}-01": _st(n_water=5000, n_water_low=4900) for m in range(1, 6)}
    over = [f"2025-1{k}-15" for k in range(3)]
    for k, d in enumerate(over):
        stats[d] = dict(WADDEN_1215, n_water_clear=300 * k, n_water_clear_low=0)
    stats["2025-12-20"] = _st(n_valid=10)                          # nearly empty
    tab = inputs.decide_offsets(stats, {}, {"2025": "no offset"})
    assert not (tab["state"] == "unclear").any() and (tab["offset"] == 0).all()
    assert not tab.loc[over + ["2025-12-20"], "strong"].any()
    assert tab.loc[over, "reason"].eq("the regime of the year's decided scenes (0)").all()
    old = {d: {k: v for k, v in s.items() if not k.startswith("n_water_clear")}
           for d, s in stats.items()}
    t1 = inputs.decide_offsets(old, {}, {"2025": "no offset"})
    assert (t1.loc[over, "state"] == "unclear").all() and (t1.loc[over, "rule"] ==
                                                           "contradiction").all()
    assert all("statistics written before the 2026-10-19 amendment" in t1.loc[d, "reason"]
               for d in over)
    assert t1.loc["2025-12-20", "offset"] == 0                     # the weak scene: regime 0
    # with the SCL cloud counts of an overcast frame the guard removes them even so
    old_sky = {d: (dict(s, **WADDEN_1215_SKY) if d in over else s) for d, s in old.items()}
    t2 = inputs.decide_offsets(old_sky, {}, {"2025": "no offset"})
    assert not (t2["state"] == "unclear").any() and (t2["offset"] == 0).all()


def test_statistics_written_before_the_amendment_fall_back_with_the_reason():
    """(f) statistics without the clear-water counts (an inputs report written
    before 2026-10-19) still decide, with the SCL-water counts, and every
    water reason says so."""
    old = {"2023-01-03": _st(old=True, n_water=5000, n_water_low=4900),
           "2023-01-05": _st(old=True, n_water=5000, n_water_low=1000),
           "2023-01-06": _st(old=True, n_neg=200),
           "2023-01-07": _st(old=True, n_low=5000),
           "2022-01-04": _st(old=True, n_water=5000, n_water_low=10),
           "2022-01-05": _st(old=True, n_water=5000, n_water_low=300)}
    assert "n_water_clear" not in old["2023-01-03"]
    tab = inputs.decide_offsets(old)
    assert (tab.loc[["2023-01-03", "2023-01-05", "2023-01-06", "2023-01-07"], "offset"] == 0).all()
    assert tab.loc["2022-01-04", "offset"] == 1000 and tab.loc["2022-01-05", "offset"] == 1000
    tail = ("SCL-water px; statistics written before the 2026-10-19 amendment: no clear-water "
            "counts, CLD not used")
    assert tab.loc["2022-01-04", "reason"] == \
        f"water B12 at or above 500 DN (share 0.002 of 5000 {tail}; frame SCL 8/9/10 share unknown)"
    for d in ("2023-01-03", "2023-01-05"):
        assert tab.loc[d, "reason"].startswith("water B12 below 500 DN") and \
            tail in tab.loc[d, "reason"]
    assert tail in tab.loc["2022-01-05", "reason"] and "partly below" in tab.loc["2022-01-05",
                                                                                 "reason"]
    # the reasons without water evidence do not mention it
    assert tab.loc["2023-01-06", "reason"] == \
        "negative values in the delivered bands (200 px >= 100): no offset"
    assert tab.loc["2023-01-07", "reason"].startswith("frame B12 below 500 DN (share 0.0556")
    # one clear count without the other (malformed): the clear counts are not used, the
    # SCL-water counts are (with the same note)
    half = _st(old=True, n_water=5000, n_water_low=10, n_water_clear=4000)
    n, n_low, what, note = inputs._water_counts(half)
    assert (n, n_low, what) == (5000, 10, "water") and "before the 2026-10-19" in note


def test_baselines_are_parsed_and_merged():
    """'0213' and 'N0213' are 02.13 (not 213); a date is 'mixed' when its
    versions fall on both sides of 04.00."""
    assert inputs._baseline_class(["N0213"]) == inputs._baseline_class(["0213"]) == "none"
    assert inputs._baseline_class(["N0511"]) == inputs._baseline_class(["05.11"]) == "offset"
    assert inputs._baseline_class(["5.1"]) == inputs._baseline_class([5.11]) == "offset"
    assert inputs._baseline_class(["05.10", "02.14"]) == "mixed"
    assert inputs._baseline_class(["x", None]) is None and inputs._baseline_class(None) is None
    assert inputs._baseline_version("N0400") == 4.0 and inputs._baseline_version("") is None


def test_offset_rule_hash_covers_parameters_and_code(monkeypatch):
    """The rule's sha256 (inputs report, prepare's staleness check) and the
    prep levels' hash cover OFFSET_RULE (as JSON) and every function of the
    decision: a changed parameter changes both."""
    from experiments.benchmark import settings
    assert set(settings.OFFSET_RULE_IMPL) <= set(settings.PREP["floor0"].impl)
    src = settings.impl_source("experiments.benchmark.inputs.OFFSET_RULE")
    assert src.startswith("<json>\n") and json.loads(src[7:]) == json.loads(
        json.dumps(inputs.OFFSET_RULE))
    assert not any(settings.impl_source(n).startswith("<missing")
                   for n in settings.OFFSET_RULE_IMPL)
    h0, p0 = inputs.offset_rule_sha256(), settings.entry_sha(settings.PREP["floor0"])
    assert h0 == inputs.offset_rule_sha256() and len(h0) == 64
    monkeypatch.setitem(inputs.OFFSET_RULE, "water_cld_max", 11)
    assert inputs.offset_rule_sha256() != h0
    assert settings.entry_sha(settings.PREP["floor0"]) != p0


def test_prepare_takes_offsets_only_from_a_report_of_the_current_rule(tmp_path):
    """prepare's frozen offsets come only from an inputs report written with
    the current offset rule (its sha256): a report of the V1 rule (no sha) or
    of another rule is 'provisional ... rerun inputs', which v2_record
    refuses without --allow-provisional-offsets (code review, staleness)."""
    import types

    from experiments.benchmark import package
    site2 = types.SimpleNamespace(site="wadden", root=str(tmp_path), offsets=None)
    p = tmp_path / "common" / "inputs_report.json"
    p.parent.mkdir(parents=True)
    offs = {"2025-01-01": 0, "2025-12-15": 0}
    cur = {"site": "wadden", "offsets": offs, "offset_rule": inputs.OFFSET_RULE,
           "offset_rule_sha256": inputs.offset_rule_sha256()}
    v1 = {"site": "wadden", "offsets": offs, "offset_rule": {"rule": inputs._OFFSET_RULE_V1}}
    other = dict(cur, offset_rule_sha256="0" * 64)
    assert inputs.offset_rule_status(cur) is None
    for rep, want in ((cur, None), (v1, "without a recorded sha256 (V1)"),
                      (other, "another offset rule (sha256 000000000000, revision 2026-10-19")):
        p.write_text(json.dumps({"sites": {"wadden": rep}}), encoding="utf-8")
        got, src = package._frozen_offsets_for(site2)
        if want is None:
            assert got == offs and src.startswith("inputs report")
        else:
            assert got is None and src.startswith("provisional (the inputs report") and \
                want in src and src.endswith("rerun inputs)"), src


def test_offset_summary_lines():
    """The lines the user pastes back: per site the counts, G-zarr and the
    identity, the warnings, the change of state against a previous report,
    and the scenes that are not plainly 'no offset' now or were '+1000'."""
    cur = {"sites": {"wadden": {
        "site": "wadden", "ok": False, "record": {"T": 3}, "offset_rule_sha256": "x",
        "offset_counts": {"no offset": 2, "unclear": 1},
        "offset_rule_counts": {"identity": 1, "contradiction": 1, "regime": 1},
        "g_zarr": {"2025": {"verdict": "identical", "scenes_identical": 1,
                            "identity_check": "identity used: 1 of its 1",
                            "identity": "2025.verify.json: verdict 'identical'"}},
        "identity_overrides": {"2025 no offset": 1},
        "problems": ["BOA offset unclear on 1 scenes: ['2025-12-31']"], "warnings": ["w"],
        "offset_decisions": {
            "2025-01-01": {"state": "no offset", "rule": "identity", "reason": "identical"},
            "2025-12-15": {"state": "no offset", "rule": "regime", "reason": "the regime"},
            "2025-12-31": {"state": "unclear", "rule": "contradiction", "reason": "c"}},
        "scenes": {"2025-12-15": dict(WADDEN_1215, n_water_clear=0, n_water_clear_low=0,
                                      **WADDEN_1215_SKY)}}}}
    prev = {"sites": {"wadden": {"offset_decisions": {
        "2025-01-01": {"state": "no offset"}, "2025-12-15": {"state": "+1000 in the values"},
        "2025-12-31": {"state": "unclear"}}}}}
    lines = inputs.offset_summary(cur, prev)
    text = "\n".join(lines)
    assert lines[0].startswith("== wadden | ok False | T 3 | offsets") and "NOT the current" \
        in lines[0]
    assert "G-zarr 2025: identical | 1 scenes by identity | identity used" in text
    assert "PROBLEM BOA offset unclear" in text and "warning w" in text
    assert "'+1000 in the values -> no offset': 1" in text and "'unclear -> unclear': 1" in text
    sc = [x for x in lines if x.strip().startswith("2025-12-15")]
    assert len(sc) == 1 and "before +1000 in the values" in sc[0] and "clear 0" in sc[0]
    assert "SCL 8/9/10 0.999" in sc[0] and "frame valid 9507800 low 84" in sc[0]
    assert not any(x.strip().startswith("2025-01-01") for x in lines)
    assert inputs.offset_summary(cur, sites=["ems"]) == ["== ems: not in the report"]


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
    # amendment of 2026-10-19: the scene statistics hold the SCL-water and the clear-water
    # counts (CLD read in the same pass); the +1000 of 2025 is decided on clear water
    assert all({"n_water", "n_water_low", "n_water_clear", "n_water_clear_low"} <= set(e)
               for e in rep["scenes"].values())
    assert all(e["n_water_clear"] <= e["n_water"] for e in rep["scenes"].values())
    strong25 = [d for d, e in rep["offset_decisions"].items() if d[:4] == "2025" and e["strong"]]
    assert len(strong25) == N_PER_YEAR - 2                        # the dropped and the cloudy
    assert all(rep["offset_decisions"][d]["reason"].startswith(
        "clear water B12 at or above 500 DN") for d in strong25)
    assert rep["offset_rule"]["revision"]["date"] == "2026-10-19"
    assert rep["offset_rule"]["water_cld_max"] == 10
    assert rep["g_zarr"]["2024"]["scenes_identical"] == 0
    assert rep["g_zarr"]["2024"]["identity"] == "no verify.json"
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


def _write_verify(root, year, nc, **kw):
    """``<year>.verify.json`` as the downloader writes it (``verify_request``:
    the site and the year, then ``zarr_cubes.compare_with_netcdf`` over the
    year's window)."""
    rep = zc.compare_with_netcdf(zc.store_path(SITE, year, root), nc, start=f"{year}-01-01",
                                 end=f"{year + 1}-01-01", **kw)
    rep = dict({"site": SITE, "year": year}, **rep)
    zc.write_json(rep, zc.sidecar_path(SITE, year, "verify.json", root))
    return rep


def test_identity_from_the_real_comparison_feeds_the_decisions(site, tmp_path):
    """Rule (0) end to end with ``zarr_cubes.compare_with_netcdf``: the 2024
    store equals its legacy cube (identity on every date: 0, strong, in the
    provisional table and in the report); the 2025 store carries +1000 and
    its offset-harmonised legacy cube differs by 1000 in B03 and B08 on every
    date (listed up to 5, truncated): no identity, the clear water decides
    1000. A stale verify.json grants nothing; SCL differences alone keep the
    identity, even with a truncated list."""
    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    assert _write_verify(root, 2024, site["legacy"])["verdict"] == "identical"
    d25 = site["data"][2025]
    harm = {b: (d25[b] if b == "SCL" else
                np.where(d25[b] != FILL, d25[b] - 1000, FILL).astype(np.int16))
            for b in ("B03", "B08", "SCL")}
    nc25 = tmp_path / "legacy_2025.nc"
    write_legacy(nc25, harm, 2025)
    v25 = _write_verify(root, 2025, nc25, max_dates_listed=5)
    assert v25["verdict"] == "differs" and v25["dates_with_differences"] == N_PER_YEAR
    assert len(v25["dates_with_differences_listed"]) == 5
    d24 = [str(t.date()) for t in _dates(2024)]
    with inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE) as src:
        ident, notes = src.legacy_identity()
        assert sorted(ident) == d24 and "verdict 'identical', 18 common dates" in notes["2024"]
        assert "truncated (5 of 18)" in notes["2025"]
        tab = src.offset_table()                # provisional (store attributes): rule 0 first
        for d in src.dates:
            assert tab.loc[d, "reason"].startswith(inputs.IDENTITY_REASON) == (d[:4] == "2024")
            assert tab.loc[d, "offset"] == (0 if d[:4] == "2024" else 1000), d
    rep = inputs.inputs_report(SITE, root=root, legacy=site["legacy"], instants=site["instants"])
    assert rep["ok"], rep["problems"]
    dec = rep["offset_decisions"]
    rec24 = [d for d in rep["offsets"] if d[:4] == "2024"]
    assert all(dec[d]["reason"].startswith(inputs.IDENTITY_REASON) and dec[d]["strong"]
               and rep["offsets"][d] == 0 for d in rec24)
    assert all(v == 1000 for d, v in rep["offsets"].items() if d[:4] == "2025")
    g = rep["g_zarr"]
    assert g["2024"]["verdict"] == "identical"
    assert g["2024"]["scenes_identical"] == len(rec24) == N_PER_YEAR - 1
    assert g["2025"]["verdict"] == "differs" and g["2025"]["dates"] == N_PER_YEAR
    assert g["2025"]["scenes_identical"] == 0 and "truncated" in g["2025"]["identity"]
    # a stale verify.json (another download: other date counts) grants nothing
    p = zc.sidecar_path(SITE, 2024, "verify.json", root)
    v = json.loads(p.read_text(encoding="utf-8"))
    v["dates"]["store"] -= 1
    zc.write_json(v, p)
    with inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE) as src:
        ident, notes = src.legacy_identity()
        assert not any(d[:4] == "2024" for d in ident) and "another download" in notes["2024"]
    # SCL differences alone keep B03 and B08 identical, even with a truncated list
    d24x = {b: np.array(site["data"][2024][b]) for b in ("B03", "B08", "SCL")}
    d24x["SCL"][[0, 4, 9], 100, :] = 4
    nc24x = tmp_path / "legacy_2024_scl.nc"
    write_legacy(nc24x, d24x, 2024)
    v24x = _write_verify(root, 2024, nc24x, max_dates_listed=1)
    assert v24x["verdict"] == "differs" and v24x["dates_with_differences"] == 3
    assert len(v24x["dates_with_differences_listed"]) == 1
    with inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE) as src:
        ident, notes = src.legacy_identity()
        assert sorted(d for d in ident if d[:4] == "2024") == d24
        assert "outside B03 and B08" in notes["2024"]


def test_overcast_scenes_in_a_store_no_longer_fail_the_inputs_check(site, tmp_path):
    """(a, e) end to end: three 2024 scenes made overcast, with no dark B12
    pixel in the frame: the cloudy scene 3 (SCL 9, CLD 90) with a bright B12,
    and scenes 8 and 9 whose SCL water (class 6) lies under CLD 80 with a
    bright B12 (900 DN, no offset). The amended rule decides them by the
    year's regime (0, weak) and the report is OK. Replayed on the same
    statistics without the clear-water and SCL cloud counts (an inputs report
    written before the amendment), the SCL-water test makes scenes 8 and 9
    strong '+1000' again: V1 then made every weak scene of the year
    'unclear'; the amended regime step flags these two only."""
    import zarr

    root = tmp_path / "cubes"
    shutil.copytree(site["root"], root)
    g = zarr.open_group(str(zc.store_path(SITE, 2024, root)), mode="r+", zarr_format=2)
    for k, cld in ((3, None), (8, 80), (9, 80)):
        for band, value in (("B12", 900), ("CLD", cld)):
            if value is None:
                continue
            a = g[band]
            blk = a[k:k + 1]
            blk[:, :, :WATER_COLS] = value
            a[k:k + 1] = blk
    rep = inputs.inputs_report(SITE, root=root, instants=site["instants"])
    assert rep["ok"], rep["problems"]
    d24 = [str(t.date()) for t in _dates(2024)]
    cloudy, over = d24[3], [d24[8], d24[9]]
    for d in over:
        e = rep["scenes"][d]
        assert e["n_water"] >= 1000 and e["n_water_low"] == 0 and e["n_water_clear"] == 0
        assert e["n_low"] == 0 and e["n_neg"] == 0
    for d in [cloudy] + over:
        dec = rep["offset_decisions"][d]
        assert rep["offsets"][d] == 0 and dec["strong"] is False
        assert dec["reason"] == "the regime of the year's decided scenes (0)"
    assert set(rep["offsets"][d] for d in rep["offsets"] if d[:4] == "2024") == {0}
    old = {d: {k: v for k, v in e.items() if k in ("n_px", "n_valid", "n_low", "n_water",
                                                    "n_water_low", "n_neg")}
           for d, e in rep["scenes"].items()}
    with inputs.ZarrSource(SITE, root=root, instants=site["instants"], lattice=PHASE) as src:
        t1 = inputs.decide_offsets({d: old[d] for d in src.dates}, src.baselines(),
                                   src.year_states())
    assert (t1.loc[over, "state"] == "unclear").all()
    assert (t1.loc[over, "rule"] == "contradiction").all()
    assert all("before the 2026-10-19 amendment" in t1.loc[d, "reason"] for d in over)
    # 17 record scenes in 2024 (one has no instant): 14 strong 0 beside the two
    assert all(t1.loc[d, "reason"].endswith("; contradicts the year's 14 strong 0") for d in over)
    assert t1.loc[cloudy, "offset"] == 0 and t1.loc[cloudy, "rule"] == "regime"
    # the scan keeps both counts for every scene, and the SCL cloud counts
    assert all(e["n_scl_valid"] == sum(e["scl"][1:12]) for e in rep["scenes"].values())
    assert rep["scenes"][cloudy]["n_scl_cloud"] == rep["scenes"][cloudy]["n_scl_valid"]


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
