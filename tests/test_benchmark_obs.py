"""Tests of experiments.benchmark.obs, THE observation of the v13 benchmark.

They pin (spec §8): raw-mode NDWI bit-identical to ``marea.extract`` and
``waterline_contour.scene_ndwi`` on a production-layout int16 cube (fill,
zero band sums, negative bands); the marea_clear mask equal to the
extraction's C; floor0 keeping NDWI in [-1, 1], making "both bands <= 0"
undefined (hence not valid) and preserving the fill; the mask truth tables
of both rules for SCL 0..11 and no data; the SCL classes equal to the
production constants; ``decode`` equal to xarray's decoding; V = mask &
finite NDWI; wet under floor0 equal to B03 > B08 where NDWI is defined;
``mask_raw_bands``; NIR equal to ``g21.to_reflectance`` of the floored
values; and the line-ending-proof source hash that keys every package.

Run:  python -X utf8 -m pytest tests/test_benchmark_obs.py -q
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, obs  # noqa: E402
from pyintertidal import granadeiro2021 as g21  # noqa: E402
from pyintertidal import marea, water  # noqa: E402
from pyintertidal import waterline_contour as wc  # noqa: E402

FILL = -32768


def _bits32(a):
    """float32 bit patterns: equality is bitwise, NaN payloads included."""
    return np.ascontiguousarray(a, np.float32).view(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
#  a production-layout cube
# ─────────────────────────────────────────────────────────────────────────────

def _bands(T=40, H=9, W=11, seed=3, clear_share=0.9):
    """Raw int16 B03, B08, SCL with every special case of the real cubes:
    fill in a band and in SCL, zero band sums, both bands negative, dark
    water with B08 < 0 (|NDWI| > 1 in raw mode), every SCL code."""
    rng = np.random.default_rng(seed)
    b03 = rng.integers(-300, 2500, (T, H, W)).astype(np.int16)
    b08 = rng.integers(-400, 2500, (T, H, W)).astype(np.int16)
    b03[0, 0, 0] = b08[0, 0, 0] = 0               # zero sum: NDWI undefined
    b03[1, 0, 1], b08[1, 0, 1] = 5, -5            # zero sum with valid values
    b03[2, 0, 2], b08[2, 0, 2] = -100, -50        # both < 0: raw defined, floor0 not
    b03[3, 0, 3], b08[3, 0, 3] = 0, -200          # floor0 -> 0 / 0
    b03[4, 0, 4], b08[4, 0, 4] = 300, -280        # raw |NDWI| = 29
    b03[5, 0, 5], b08[5, 0, 5] = -999, 400        # raw DN 1 in B03
    b03[6, 1, 1] = FILL                           # missing band
    b08[7, 1, 2] = FILL
    clear = rng.random((T, H, W)) < clear_share
    scl = np.where(clear, rng.choice([4, 5, 6, 7], (T, H, W)),
                   rng.choice([0, 1, 2, 3, 8, 9, 10, 11], (T, H, W))).astype(np.int16)
    scl[8, 2, 2] = FILL                           # SCL no data
    scl[9, 2, 3] = FILL
    scl[10, :, 0] = np.arange(H) % 12             # make sure every code occurs
    scl[11, 0, :] = np.arange(W) % 12
    return b03, b08, scl


def _cube(tmp_path, b03, b08, scl):
    """Write the bands as the production cubes are written: int16,
    _FillValue -32768, no scale, chunks (1, 8, 8), a crs variable."""
    xr = pytest.importorskip("xarray")
    pyproj = pytest.importorskip("pyproj")
    import pandas as pd

    T, H, W = b03.shape
    ds = xr.Dataset(
        {v: (("t", "y", "x"), a) for v, a in (("B03", b03), ("B08", b08), ("SCL", scl))},
        coords={"t": pd.date_range("2023-01-01", periods=T, freq="D"),
                "y": 5_700_000.0 - 10.0 * np.arange(H), "x": 500_000.0 + 10.0 * np.arange(W)})
    ds["crs"] = xr.DataArray(np.int32(0),
                             attrs={"crs_wkt": pyproj.CRS.from_epsg(32631).to_wkt()})
    enc = {v: {"dtype": "int16", "_FillValue": FILL, "chunksizes": (1, 8, 8), "zlib": True}
           for v in ("B03", "B08", "SCL")}
    path = str(tmp_path / "cube.nc")
    ds.to_netcdf(path, encoding=enc)
    return path


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    b03, b08, scl = _bands()
    path = _cube(tmp_path_factory.mktemp("obs_cube"), b03, b08, scl)
    return path, b03, b08, scl


# ─────────────────────────────────────────────────────────────────────────────
#  constants and the frozen plan
# ─────────────────────────────────────────────────────────────────────────────

def test_constants_are_the_production_ones():
    assert obs.BLOCK_SCL == (0, *water.BAD_CLASSES) == (0, 3, 8, 9, 10)
    assert obs.CLEAR_SCL == tuple(marea.CLEAR) == (4, 5, 6, 7)
    assert obs.ITEM_MASKED_SCL == tuple(wc.ITEM_PQ_MASKED_SCL)
    assert obs.MASK_RULE == "marea_clear"            # D1, authors' final decision
    assert obs.INDEX_MODE == "floor0"                # D2
    assert set(obs.MASK_RULES) == {"blocklist", "marea_clear"}
    assert set(obs.INDEX_MODES) == {"floor0", "raw"}
    f = config.FROZEN["obs"]
    assert (f["primary_mask"], f["primary_index"]) == (obs.MASK_RULE, obs.INDEX_MODE)
    assert f["encoding"]["fill"] == obs.FILL == FILL
    # the package domain is built under blocklist + floor0, a superset
    u = config.FROZEN["package"]["U_rec"]
    assert (u["mask"], u["index"], u["min_clear"], u["min_wet"], u["min_dry"]) == \
        ("blocklist", "floor0", 31, 4, 4)
    assert config.FROZEN["package"]["frame_valid"]["mask"] == obs.MASK_RULE


def test_frozen_sections_and_their_hash():
    assert config.FROZEN_SECTIONS == (
        "record", "package", "obs", "domain", "levels", "subsets", "marea", "nidem",
        "granadeiro", "gm20", "hd26", "baselines", "judges", "truth_elev", "truth_lag",
        "scoring", "inference", "curves", "threads", "positive_control")
    plain = config.frozen_plain("obs")
    want = hashlib.sha256(json.dumps(plain, sort_keys=True).encode()).hexdigest()
    assert config.frozen_sha("obs") == want
    assert config.frozen_sha("obs") != config.frozen_sha("domain")
    assert config.frozen_sha("obs", "domain") == hashlib.sha256(json.dumps(
        config.frozen_plain("obs", "domain"), sort_keys=True).encode()).hexdigest()
    with pytest.raises(TypeError):
        config.FROZEN["obs"]["primary_mask"] = "blocklist"     # read-only
    with pytest.raises(KeyError):
        config.frozen_sha("no_such_section")


# ─────────────────────────────────────────────────────────────────────────────
#  masks
# ─────────────────────────────────────────────────────────────────────────────

def test_mask_valid_truth_table():
    codes = np.array(list(range(12)) + [255], np.uint8)
    #                      0  1  2  3  4  5  6  7  8  9 10 11 no data
    marea_clear = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
    blocklist = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0], bool)
    item = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0], bool)
    assert obs.mask_valid(codes, "marea_clear").tolist() == marea_clear.tolist()
    assert obs.mask_valid(codes, "blocklist").tolist() == blocklist.tolist()
    assert obs.mask_valid(codes).tolist() == marea_clear.tolist()      # primary
    assert obs.item_valid(codes).tolist() == item.tolist()
    # marea_clear is a subset of blocklist (why one package serves both)
    assert not (marea_clear & ~blocklist).any()
    with pytest.raises(ValueError):
        obs.mask_valid(codes, "clear")
    with pytest.raises(TypeError):
        obs.mask_valid(codes.astype(np.int16), "blocklist")


def test_item_valid_equals_pixel_valid_item(cube):
    _, _, _, scl = cube
    dec = obs.decode(scl, FILL)
    assert np.array_equal(obs.item_valid(obs.scl_u8(scl, FILL)),
                          wc.pixel_valid(dec, rule="item"))


def test_scl_u8():
    raw = np.array([[0, 4, 11, FILL], [7, 3, FILL, 9]], np.int16)
    u8 = obs.scl_u8(raw, FILL)
    assert u8.dtype == np.uint8
    assert u8.tolist() == [[0, 4, 11, 255], [7, 3, 255, 9]]
    dec = np.array([0.0, 4.0, np.nan, 11.0], np.float32)          # xarray-decoded
    assert obs.scl_u8(dec).tolist() == [0, 4, 255, 11]
    for bad in (np.array([12], np.int16), np.array([-1], np.int16),
                np.array([2.5], np.float32)):
        with pytest.raises(ValueError):
            obs.scl_u8(bad, FILL)
    counts = obs.scl_counts(u8)
    assert counts.shape == (13,) and counts.sum() == u8.size
    assert counts[4] == 1 and counts[12] == 2


# ─────────────────────────────────────────────────────────────────────────────
#  decoding, NDWI and V against the production code
# ─────────────────────────────────────────────────────────────────────────────

def test_decode_equals_xarray_decoding(cube):
    xr = pytest.importorskip("xarray")
    path, b03, b08, scl = cube
    with xr.open_dataset(path) as ds, xr.open_dataset(path, mask_and_scale=False) as raw:
        for v, a in (("B03", b03), ("B08", b08), ("SCL", scl)):
            assert raw[v].dtype == np.int16
            assert np.array_equal(raw[v].values, a)
            dec = ds[v].values
            assert dec.dtype == np.float32
            assert obs.decode(raw[v].values, FILL).dtype == np.float32
            assert np.array_equal(obs.decode(raw[v].values, FILL), dec, equal_nan=True)
    with pytest.raises(TypeError):
        obs.decode(np.zeros(3, np.float32), FILL)


def test_raw_ndwi_and_clear_equal_marea_extract(cube):
    path, b03, b08, scl = cube
    ex = marea.extract(path)
    keep = ex["keep"]
    H, W = ex["shape"]
    assert len(keep) > 0.5 * H * W              # most pixels intertidal by design
    nd, V = obs.observe(b03, b08, obs.scl_u8(scl, FILL), FILL,
                        rule="marea_clear", mode="raw")
    rr, cc = keep // W, keep % W
    assert ex["Y"].dtype == nd.dtype == np.float32
    assert np.array_equal(_bits32(nd[:, rr, cc]), _bits32(ex["Y"]))    # bitwise, NaN too
    assert np.array_equal(obs.mask_valid(obs.scl_u8(scl, FILL), "marea_clear")[:, rr, cc],
                          ex["C"])
    # every special case is on keep, and raw NDWI is unbounded there
    assert np.isnan(ex["Y"]).any() and np.nanmax(np.abs(ex["Y"])) > 1


def test_raw_ndwi_equals_scene_ndwi(cube):
    _, b03, b08, scl = cube
    ref, valid = wc.scene_ndwi(obs.decode(b03, FILL), obs.decode(b08, FILL),
                               obs.decode(scl, FILL), rule="item")
    g, n = obs.bands(b03, b08, FILL, mode="raw")
    mine = obs.ndwi(g, n)
    assert mine.dtype == ref.dtype == np.float32
    assert np.array_equal(_bits32(mine), _bits32(ref))
    assert np.array_equal(valid, obs.item_valid(obs.scl_u8(scl, FILL)))


def test_ndwi_is_the_production_expression_bit_for_bit():
    """obs.ndwi uses fewer temporaries than the production one-liner; the
    float32 operations are the same, so every bit (NaN included) is too."""
    rng = np.random.default_rng(11)
    g = rng.integers(-1000, 20000, 400_000).astype(np.float32)
    n = rng.integers(-1000, 20000, 400_000).astype(np.float32)
    g[::97] = np.nan
    n[::89] = np.nan
    n[::7] = -g[::7]                            # zero sums
    g[5::11] = n[5::11] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(g + n != 0, (g - n) / (g + n), np.nan).astype(np.float32)
    assert np.array_equal(_bits32(obs.ndwi(g, n)), _bits32(ref))
    assert np.isnan(ref).sum() > 10_000


def test_floor0(cube):
    _, b03, b08, scl = cube
    s8 = obs.scl_u8(scl, FILL)
    nd, V = obs.observe(b03, b08, s8, FILL, rule="blocklist", mode="floor0")
    fin = np.isfinite(nd)
    assert np.all(np.abs(nd[fin]) <= 1.0)
    nd_raw, _ = obs.observe(b03, b08, s8, FILL, rule="blocklist", mode="raw")
    assert np.nanmax(np.abs(nd_raw)) > 1.0      # the case floor0 removes
    # both bands <= 0 (non-fill) -> NDWI undefined -> never a valid observation
    both = (b03 != FILL) & (b08 != FILL) & (b03 <= 0) & (b08 <= 0)
    assert both.any()
    assert np.isnan(nd[both]).all() and not V[both].any()
    assert np.isfinite(nd_raw[both & (b03 + b08.astype(int) != 0)]).all()
    # the fill survives the floor, in the float and in the int16 path
    for a in (b03, b08):
        fl = obs.floor_raw(a, FILL)
        assert fl.dtype == np.int16
        assert np.array_equal(fl == FILL, a == FILL)
        assert np.all(fl[a != FILL] >= 0)
        assert np.array_equal(fl[a > 0], a[a > 0])
    g, n = obs.bands(b03, b08, FILL, mode="floor0")
    assert np.array_equal(g, obs.decode(obs.floor_raw(b03, FILL), FILL), equal_nan=True)
    assert np.array_equal(n, obs.decode(obs.floor_raw(b08, FILL), FILL), equal_nan=True)
    with pytest.raises(ValueError):
        obs.bands(b03, b08, FILL, mode="clip")


@pytest.mark.parametrize("rule", ["blocklist", "marea_clear"])
@pytest.mark.parametrize("mode", ["floor0", "raw"])
def test_valid_is_mask_and_finite_ndwi(cube, rule, mode):
    _, b03, b08, scl = cube
    s8 = obs.scl_u8(scl, FILL)
    nd, V = obs.observe(b03, b08, s8, FILL, rule=rule, mode=mode)
    assert V.dtype == bool and nd.dtype == np.float32
    assert np.array_equal(V, obs.mask_valid(s8, rule) & np.isfinite(nd))
    assert np.array_equal(nd, obs.ndwi(*obs.bands(b03, b08, FILL, mode)), equal_nan=True)
    # a band fill under a valid SCL is not an observation (Granadeiro's
    # KeepCheck blocker case of the review)
    assert not V[6, 1, 1] and not V[7, 1, 2]
    # marea_clear observations are blocklist observations
    _, V_bl = obs.observe(b03, b08, s8, FILL, rule="blocklist", mode=mode)
    _, V_mc = obs.observe(b03, b08, s8, FILL, rule="marea_clear", mode=mode)
    assert not (V_mc & ~V_bl).any()
    # elementwise: a row block gives the frame's values
    nd_b, V_b = obs.observe(b03[:, 2:5], b08[:, 2:5], s8[:, 2:5], FILL, rule, mode)
    assert np.array_equal(nd_b, nd[:, 2:5], equal_nan=True) and np.array_equal(V_b, V[:, 2:5])
    # both rules from one NDWI (the package's first pass)
    nd_all, Vs = obs.observe_rules(b03, b08, s8, FILL, mode=mode)
    assert np.array_equal(_bits32(nd_all), _bits32(nd))
    assert set(Vs) == set(obs.MASK_RULES) and np.array_equal(Vs[rule], V)


def test_wet_under_floor0_is_b03_gt_b08(cube):
    _, b03, b08, scl = cube
    s8 = obs.scl_u8(scl, FILL)
    nd, V = obs.observe(b03, b08, s8, FILL, rule="blocklist", mode="floor0")
    w = obs.wet(nd, V)
    assert np.array_equal(w, V & (nd > 0))
    assert np.array_equal(w[V], (b03 > b08)[V])
    # and marea.extract's own wet test on the floored bands
    g, n = obs.bands(b03, b08, FILL, mode="floor0")
    with np.errstate(invalid="ignore", divide="ignore"):
        ext = (g - n) / np.maximum(g + n, 1) > 0
    assert np.array_equal(w[V], ext[V])
    assert not w[~V].any()


# ─────────────────────────────────────────────────────────────────────────────
#  raw-value paths (Granadeiro)
# ─────────────────────────────────────────────────────────────────────────────

def test_nir_is_to_reflectance_of_floored_values(cube):
    _, _, b08, _ = cube
    assert np.array_equal(obs.nir(b08, FILL), g21.to_reflectance(obs.floor_raw(b08, FILL), FILL),
                          equal_nan=True)
    assert np.array_equal(obs.nir(b08, FILL, mode="raw"), g21.to_reflectance(b08, FILL),
                          equal_nan=True)
    r = obs.nir(b08, FILL)
    assert r.dtype == np.float64
    assert np.array_equal(np.isnan(r), b08 == FILL)
    assert np.nanmin(r) >= 0.0 and np.nanmin(obs.nir(b08, FILL, mode="raw")) < 0.0
    # the int16 path equals the float path of bands() divided by 10000
    g, _ = obs.bands(b08, b08, FILL, mode="floor0")
    assert np.array_equal(obs.reflectance(b08, FILL), g.astype(np.float64) / 10000.0,
                          equal_nan=True)


def test_mask_raw_bands(cube):
    _, b03, b08, scl = cube
    s8 = obs.scl_u8(scl, FILL)
    _, V = obs.observe(b03, b08, s8, FILL)
    G, N = b03.copy(), b08.copy()
    held = (~V & ((G != FILL) | (N != FILL))).sum()
    removed = obs.mask_raw_bands(G, N, V, FILL)
    assert removed == held > 0
    assert np.array_equal(G == FILL, ~V | (b03 == FILL))
    assert np.array_equal(N == FILL, ~V | (b08 == FILL))
    assert np.array_equal(G[V], b03[V]) and np.array_equal(N[V], b08[V])
    # after masking, every surviving pair is a valid observation and NIR
    # treats the masked ones as no data
    assert np.array_equal(np.isnan(obs.nir(N, FILL)), ~V | (b08 == FILL))
    assert obs.mask_raw_bands(G, N, V, FILL) == 0                # idempotent
    ro = b03.copy()
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        obs.mask_raw_bands(ro, b08.copy(), V, FILL)
    with pytest.raises(ValueError):
        obs.mask_raw_bands(b03[:1].copy(), b08.copy(), V, FILL)


def test_check_boa():
    a = np.array([FILL, -1000, -999, 0, 3000], np.int16)
    assert obs.check_boa(a, FILL) == 4
    with pytest.raises(AssertionError):
        obs.check_boa(np.array([FILL, -1001], np.int16), FILL)


def test_rule_and_mode():
    assert obs.rule_and_mode() == ("marea_clear", "floor0")
    assert obs.rule_and_mode({}) == ("marea_clear", "floor0")
    assert obs.rule_and_mode({"mask": "blocklist"}) == ("blocklist", "floor0")
    assert obs.rule_and_mode({"index": "raw", "floor": 80}) == ("marea_clear", "raw")
    with pytest.raises(ValueError):
        obs.rule_and_mode({"mask": "item"})


# ─────────────────────────────────────────────────────────────────────────────
#  the source hash behind every package key
# ─────────────────────────────────────────────────────────────────────────────

def test_source_hash_ignores_line_endings(tmp_path):
    text = b"x = 1\nif x:\n    y = 2\n"
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(text)
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    assert config.sha256_source(lf) == config.sha256_source(crlf) == \
        hashlib.sha256(text).hexdigest()
    src = open(obs.__file__, "rb").read().replace(b"\r\n", b"\n")
    assert obs.source_sha256() == hashlib.sha256(src).hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
#  validation v2: the observation of a setting (spec §2.2-2.6, §10.3)
# ═════════════════════════════════════════════════════════════════════════════

from experiments.benchmark import inputs, settings  # noqa: E402


def _instants(dates):
    return {d: f"{d}T10:50:00Z" for d in dates}


def _same(a, b):
    if a.dtype == np.float32:
        return np.array_equal(_bits32(a), _bits32(b))
    return np.array_equal(a, b)


def test_v2_ndwi_path_is_bitwise_marea_extract(cube):
    """Through the NetCDF SceneSource and observe(setting): raw/C1 NDWI and
    C equal marea.extract bit for bit; floor0 equals the v13 path; Y is X;
    W is obs.wet."""
    path, b03, b08, scl = cube
    ex = marea.extract(path)
    keep = ex["keep"]
    H, W = ex["shape"]
    rr, cc = keep // W, keep % W
    with inputs.NetCDFSource(path, "escalda", _instants(list(ex["dates"])), lattice=None) as src:
        assert src.dates == list(ex["dates"])
        raw = src.read_block(np.arange(src.T), ["B03", "B08", "SCL"])
        o = obs.observe(raw, "C1.raw.ndwi.D1", offset=src.offsets)
        assert np.array_equal(_bits32(o.X[:, rr, cc]), _bits32(ex["Y"]))
        assert np.array_equal(obs.pixel_valid("C1", raw["SCL"])[:, rr, cc], ex["C"])
        assert np.array_equal(o.V, obs.mask_valid(raw["SCL"], "marea_clear") & np.isfinite(o.X))
        assert o.Y is o.X
        p0 = obs.observe_setting(raw, settings.P0, offset=src.offsets)
        nd, V = obs.observe(b03, b08, obs.scl_u8(scl, FILL), FILL, "marea_clear", "floor0")
        assert np.array_equal(_bits32(p0.X), _bits32(nd)) and np.array_equal(p0.V, V)
        assert np.array_equal(p0.W, obs.wet(nd, V)) and p0.Y is p0.X
        # the v2 form through observe() equals observe_setting
        o2 = obs.observe_setting(raw, "C1.raw.ndwi.D1", offset=0)
        assert all(_same(a, b) for a, b in zip(o, o2))
    with pytest.raises(TypeError):
        obs.observe(b03, b08, obs.scl_u8(scl, FILL), FILL, offset=0)


def scene9(T=3, H=40, W=36, seed=12, phase=(0, 0)):
    """Raw 9-band scenes: water (left 16 columns: the shore on a 20 m block
    edge, so no 20 m block mixes the classes) and land, a cloud blob (SCL 9,
    CLD 80) on scene 1, a fill in B03 and B08, SCL 0; 20 m bands replicated
    on ``phase``."""
    rng = np.random.default_rng(seed)
    water = np.zeros((H, W), bool)
    water[:, :16] = True
    hb, wb = obs._block_index(H, phase[0])[1], obs._block_index(W, phase[1])[1]
    w20 = obs.block_take20(water, phase)
    r = {}
    for b, (wl, wh, ll, lh) in {"B02": (400, 700, 500, 900), "B03": (600, 1000, 500, 900),
                                "B04": (300, 600, 700, 1200),
                                "B08": (50, 250, 2000, 3500)}.items():
        r[b] = np.where(water, rng.integers(wl, wh, (T, H, W)), rng.integers(ll, lh, (T, H, W)))
    for b, (wl, wh, ll, lh) in {"B8A": (40, 220, 2000, 3400), "B11": (20, 150, 1500, 2600),
                                "B12": (10, 100, 900, 1800)}.items():
        r[b] = obs.expand20(np.where(w20, rng.integers(wl, wh, (T, hb, wb)),
                                     rng.integers(ll, lh, (T, hb, wb))), phase, (H, W))
    scl = obs.expand20(np.where(w20, 6, rng.choice([4, 5], (T, hb, wb))), phase, (H, W)).copy()
    cld = obs.expand20(rng.integers(0, 10, (T, hb, wb)), phase, (H, W)).copy()
    scl[1, 10:24, 20:34] = 9
    cld[1, 10:24, 20:34] = 80
    scl[2, 0, :] = 0
    r = {b: a.astype(np.int16) for b, a in r.items()}
    r["B03"][0, 5, 5] = FILL
    r["B08"][0, 5, 5] = FILL
    r["SCL"] = scl.astype(np.int16)
    r["CLD"] = cld.astype(np.int16)
    return r, np.broadcast_to(water, (T, H, W))


def _params(index):
    return settings.index_params(index, t_nir=0.1 if index == "nir" else None,
                                 s_k=None if index == "ndwi" else 0.8)


@pytest.mark.parametrize("index", list(settings.INDEX))
@pytest.mark.parametrize("detect", list(settings.DETECT))
def test_v2_every_index_and_detector_on_bimodal_scenes(index, detect):
    raw, water = scene9()
    sid = f"C1.floor0.{index}.{detect}"
    p = _params(index)
    s8 = obs.scl_u8(raw["SCL"])
    thr = None
    if detect in ("D2", "D3"):
        o1 = obs.observe_setting(raw, f"C1.floor0.{index}.D1", params=p, lattice=(0, 0))
        log_hist = settings.INDEX[index].log_hist
        thr = []
        for k in range(o1.X.shape[0]):
            if detect == "D2":
                r = obs.d2_threshold(o1.X[k], o1.V[k], np.ones(o1.X.shape[1:], bool), p["t_k"],
                                     params={"min_px": 200}, log_hist=log_hist)
            else:
                r = obs.d3_threshold(o1.X[k], o1.V[k], p["t_k"],
                                     params={"min_buffer_px": 100, "min_edge_px": 10},
                                     log_hist=log_hist)
            thr.append(r["t"])
        thr = np.asarray(thr)
    X, Y, V, Wt = obs.observe_setting(raw, sid, params=p, lattice=(0, 0), threshold=thr)
    assert X.dtype == np.float32 and Y.dtype == np.float32 and V.dtype == bool and Wt.dtype == bool
    assert X.shape == V.shape == Wt.shape == raw["B03"].shape
    assert not (Wt & ~V).any() and not (V & ~obs.cloud_c1(s8)).any()
    assert not V[0, 5, 5] and not V[1, 15, 25] and not V[2, 0].any()
    if index == "ndwi":
        assert Y is X
    else:
        np.testing.assert_array_equal(Y, ((X - np.float32(p["t_k"])) / np.float32(p["s_k"])))
    # water is found: accuracy on the valid observations
    acc = np.mean(Wt[V] == water[V])
    assert acc > 0.97, (sid, acc)


def test_v2_offsets_cancel_bit_for_bit():
    raw, _ = scene9()
    off = {b: (np.where(a == FILL, a, a + 1000).astype(np.int16) if b in obs.SPECTRAL_BANDS
               else a) for b, a in raw.items()}
    for prep in ("floor0", "raw"):
        for index in settings.INDEX:
            sid = f"C1.{prep}.{index}.D5"
            a = obs.observe_setting(off, sid, offset=1000, params=_params(index))
            b = obs.observe_setting(raw, sid, offset=0, params=_params(index))
            assert all(_same(x, y) for x, y in zip(a, b)), sid
    # one offset per scene in a block of mixed regimes
    mixed = {b: (np.concatenate([raw[b][:1], off[b][1:]]) if b in obs.SPECTRAL_BANDS
                 else raw[b]) for b in raw}
    m = obs.observe_setting(mixed, "C1.floor0.mndwi.D1", offset=np.array([0, 1000, 1000]),
                            params=_params("mndwi"))
    r = obs.observe_setting(raw, "C1.floor0.mndwi.D1", params=_params("mndwi"))
    assert _same(m.X, r.X)
    with pytest.raises(ValueError, match="one offset per scene"):
        obs.observe_setting(raw, settings.P0, offset=np.array([0, 1000]))
    with pytest.raises(ValueError, match="offsets must be"):
        obs.observe_setting(raw, settings.P0, offset=500)


def test_v2_cloud_levels_in_the_observation():
    raw, _ = scene9()
    o1 = obs.observe_setting(raw, "C1.floor0.ndwi.D1")
    o3 = obs.observe_setting(raw, "C3.floor0.ndwi.D1")
    assert not o3.V[1, 15, 25] and not o3.V[1, 15, 19]          # blob + buffer
    assert np.array_equal(o3.V, obs.pixel_valid("C3", raw["SCL"], raw["CLD"])
                          & np.isfinite(o3.X))
    # a precomputed cloud mask (the package's stored bits) overrides the rule
    cv = np.ones(raw["SCL"].shape, bool)
    o = obs.observe_setting(raw, "C3.floor0.ndwi.D1", cloud_valid=cv)
    assert np.array_equal(o.V, np.isfinite(o.X))
    # scene rules observe through their pixel rule (C1)
    o2 = obs.observe_setting(raw, "C2.floor0.ndwi.D1")
    assert all(np.array_equal(a, b) for a, b in zip(o1[2:], o2[2:]))
    # C11 needs the classes; with them, V = C11 & finite X
    ocm = np.zeros(raw["SCL"].shape, np.uint8)
    ocm[:, :, :3] = 2
    o = obs.observe_setting(raw, "C11.floor0.ndwi.D1", ocm=ocm)
    assert not o.V[:, :, :3].any() and o.V[0, 10, 10]
    with pytest.raises(ValueError, match="C11 needs"):
        obs.observe_setting(raw, "C11.floor0.ndwi.D1")
    with pytest.raises(ValueError, match="per-scene thresholds"):
        obs.observe_setting(raw, "C1.floor0.ndwi.D2")
    with pytest.raises(ValueError, match="one threshold per scene"):
        obs.observe_setting(raw, "C1.floor0.ndwi.D2", threshold=np.zeros(2))
    with pytest.raises(ValueError, match="pass s_k"):
        obs.observe_setting(raw, "C1.floor0.mndwi.D1")


def test_v2_d4_and_d5_rules():
    V = np.ones(6, bool)
    V[5] = False
    assert obs.detect_d4(V, np.array([6, 4, 6, 255, 6, 6], np.uint8)).tolist() == \
        [True, False, True, False, True, False]
    nd = np.array([0.1, 0.1, -0.1, 0.1, np.nan, 0.1], np.float32)
    mn = np.array([0.1, -0.1, 0.1, -0.1, 0.2, 0.1], np.float32)
    aw = np.array([-0.1, 0.1, 0.1, -0.1, 0.2, 0.1], np.float32)
    assert obs.detect_d5(V, nd, mn, aw).tolist() == [True, True, True, False, True, False]


def test_v2_core_hash():
    h = obs.core_sha256()
    assert h == obs.core_sha256() and h != obs.source_sha256()
    assert set(obs.CORE_FUNCTIONS) <= set(dir(obs))
    # the registry entries' own functions are not in the core
    assert not {"cloud_c3", "index_mndwi", "d3_threshold", "ocm_classes"} & set(obs.CORE_FUNCTIONS)
