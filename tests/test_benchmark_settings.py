"""Tests of experiments.benchmark.settings and the obs registries behind it.

They pin (spec §2, §10.3): the registries and their ids; setting ids round
trip for every combination of levels; the truth table of every pixel cloud
rule for SCL 0..11 and no data; C3 and C5 openings and buffers against a
brute-force disk; C11 through a stub OmniCloudMask model; the scene rules
against the production functions (stability.filter_dates, g21, load_ard);
every index formula against ``water.index_block`` where one exists and a
float64 reference otherwise; the 20 m rule against brute force; Otsu against
skimage; the D2 guards; edge-Otsu on synthetic edges; the standardisation;
effective keys and per-entry hashes (adding a dummy level changes nothing
else); windows and the subset families with their frozen seeds.

Run:  python -X utf8 -m pytest tests/test_benchmark_settings.py -q
"""
import itertools
import json
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, obs, settings  # noqa: E402
from pyintertidal import granadeiro2021 as g21  # noqa: E402
from pyintertidal import stability, water  # noqa: E402
from pyintertidal import waterline_contour as wc  # noqa: E402

FILL = -32768
CODES = np.array(list(range(12)) + [255], np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  registries, ids, cells
# ─────────────────────────────────────────────────────────────────────────────

def test_registries_hold_the_spec_levels():
    assert tuple(settings.CLOUD) == ("C0", "C1", "C1d", "C1s", "C2", "C3", "C4", "C5", "C6",
                                     "C7", "C8", "C11")
    assert settings.SCENE_CLOUD == ("C2", "C6", "C7", "C8")
    assert all(settings.CLOUD[c].pixel_rule == "C1" for c in settings.SCENE_CLOUD)
    assert tuple(settings.PREP) == ("floor0", "raw")
    assert tuple(settings.INDEX) == ("ndwi", "mndwi", "nir", "awei_sh", "awei_nsh", "wri",
                                     "ndwi20")
    assert tuple(settings.DETECT) == ("D1", "D2", "D3", "D4", "D5")
    assert tuple(settings.LEVEL) == ("LB", "LT", "LM", "LG")
    assert settings.LEVEL["LG"].upper_bound and not settings.LEVEL["LG"].confirmatory
    # the base P0 and its tiers
    assert settings.P0.id == "C1.floor0.ndwi.D1"
    for f, k in (("cloud", "C1"), ("prep", "floor0"), ("index", "ndwi"), ("detect", "D1"),
                 ("level", "LB")):
        assert settings.REGISTRIES[f][k].tier == "P0"
    # the 20 m rule is declared on every index with a 20 m band
    for e in settings.INDEX.values():
        assert e.at20 == bool(e.bands20)
        assert e.bands == e.bands10 + e.bands20
    assert settings.INDEX["nir"].t_k is None and settings.INDEX["wri"].t_k == 1.0
    assert settings.INDEX["wri"].log_hist
    # C11: OmniCloudMask, its licence and version recorded
    p = settings.CLOUD["C11"].params
    assert (p["package"], p["licence"], p["clear_class"]) == ("omnicloudmask", "MIT", 0)
    assert tuple(p["bands"]) == ("B04", "B03", "B8A")
    assert settings.CLOUD["C3"].params["p"] == 20
    # read-only
    with pytest.raises(TypeError):
        settings.CLOUD["C9"] = None
    with pytest.raises(TypeError):
        settings.CLOUD["C3"].params["p"] = 35
    json.dumps(settings.frozen_sections())                 # JSON-ready


def test_setting_ids_round_trip_for_every_combination():
    n = 0
    for c, p, i, d, lv in itertools.product(settings.CLOUD, settings.PREP, settings.INDEX,
                                            settings.DETECT, settings.LEVEL):
        s = settings.Setting(c, p, i, d, lv)
        assert settings.Setting.parse(s.id) == s
        assert ("@" in s.id) == (lv != "LB")
        assert len(s.id) <= 25                     # short ids keep the paths short
        n += 1
    assert n == 12 * 2 * 7 * 5 * 4
    assert settings.setting_id("C11", "floor0", "mndwi", "D3") == "C11.floor0.mndwi.D3"
    assert settings.parse_setting_id("C1.floor0.ndwi.D1@LM").level == "LM"
    for bad in ("C1.floor0.ndwi", "C1.floor0.ndwi.D1@LB", "C9.floor0.ndwi.D1",
                "C1.floor1.ndwi.D1", "C1.floor0.ndvi.D1", "C1.floor0.ndwi.D7", "C1.floor0.ndwi.D1@LX"):
        with pytest.raises(ValueError):
            settings.Setting.parse(bad)
    assert settings.cell_id("C2.floor0.ndwi.D1", "W3") == ("C2.floor0.ndwi.D1", "W3")


def test_setting_bands_and_rules():
    s = settings.Setting.parse("C11.floor0.mndwi.D3")
    assert s.bands == ("B03", "B04", "B8A", "B11", "SCL")
    assert s.pixel_rule == "C11" and s.scene_rule is None
    s = settings.Setting.parse("C2.raw.ndwi.D5")
    assert s.pixel_rule == "C1" and s.scene_rule == "C2"
    assert s.bands == ("B03", "B08", "B11", "B12", "SCL")
    assert settings.Setting.parse("C3.floor0.nir.D4").bands == ("B08", "SCL", "CLD")
    assert obs.spectral_bands("C1.floor0.awei_sh.D1") == ("B02", "B03", "B08", "B11", "B12")


def test_effective_keys_project_cells_onto_signatures():
    sig_gran = settings.Signature(reads={"cloud", "prep", "index", "level", "window"})
    sig_khan = settings.Signature(reads={"cloud", "prep", "level", "window"})
    sig_marea = settings.Signature(reads={"cloud", "prep", "index", "window"}, domain=True)
    sig_lag = settings.Signature(reads={"cloud", "prep", "index", "detect", "window"})
    # Granadeiro does not read W: D1 and D3 cells share one unit
    k1 = settings.effective_key("C1.floor0.mndwi.D1", "W3", sig_gran)
    k3 = settings.effective_key("C1.floor0.mndwi.D3", "W3", sig_gran)
    assert k1 == k3 == "C1.floor0.mndwi.*@LB|W3"
    # Khan reads neither the index nor the detector
    assert settings.effective_key("C4.floor0.nir.D3", "W3", sig_khan) == \
        settings.effective_key("C4.floor0.ndwi.D1", "W3", sig_khan) == "C4.floor0.*.*@LB|W3"
    # MAREA depends on D through its domain, and is LB by design (no level read)
    assert settings.effective_key("C1.floor0.ndwi.D3", "W3", sig_marea) == \
        "C1.floor0.ndwi.D3@*|W3"
    assert settings.effective_key("C1.floor0.ndwi.D3@LM", "W3", sig_marea) == \
        settings.effective_key("C1.floor0.ndwi.D3", "W3", sig_marea)
    # lag products do not read the level source
    assert settings.effective_key("C1.floor0.ndwi.D1@LG", "W1", sig_lag).endswith("@*|W1")
    # G1' collapses to 12 effective cells for a method that does not read D
    g1 = [settings.setting_id(c, "floor0", i, d) for c in ("C1", "C2", "C4", "C11")
          for i in ("ndwi", "mndwi", "nir") for d in ("D1", "D3")]
    assert len(g1) == 24
    assert len({settings.effective_key(s, "W3", sig_gran) for s in g1}) == 12
    assert len({settings.effective_key(s, "W3", sig_marea) for s in g1}) == 24
    with pytest.raises(ValueError):
        settings.Signature(reads={"colour"})


def test_per_entry_hashes_and_a_dummy_level(monkeypatch):
    sig = settings.Signature()
    before = {s: settings.code_shas(s, "W3", sig)
              for s in ("C1.floor0.ndwi.D1", "C11.floor0.mndwi.D3", "C2.raw.awei_sh.D5@LT")}
    keys_before = {s: settings.effective_key(s, "W3", sig) for s in before}
    reg_before = settings.registry_shas("cloud")
    # a scene rule's code key includes its pixel rule; D5 includes its indices
    assert {"cloud:C1", "cloud:C2", "index:mndwi", "index:awei_nsh", "index:ndwi"} <= \
        set(before["C2.raw.awei_sh.D5@LT"])
    dummy = settings.CloudLevel(id="C99", rule="dummy", params={"x": 1},
                                impl=("experiments.benchmark.obs.cloud_c1",))
    cloud = types.MappingProxyType({**settings.CLOUD, "C99": dummy})
    monkeypatch.setattr(settings, "CLOUD", cloud)
    monkeypatch.setattr(settings, "REGISTRIES",
                        types.MappingProxyType({**settings.REGISTRIES, "cloud": cloud}))
    after = {s: settings.code_shas(s, "W3", sig) for s in before}
    assert after == before
    assert {s: settings.effective_key(s, "W3", sig) for s in before} == keys_before
    reg_after = settings.registry_shas("cloud")
    assert {k: v for k, v in reg_after.items() if k != "C99"} == reg_before
    assert settings.Setting("C99").id == "C99.floor0.ndwi.D1"


def test_entry_sha_covers_fields_and_source():
    a = settings.CloudLevel(id="Cx", rule="r", impl=("experiments.benchmark.obs.cloud_c1",))
    b = settings.CloudLevel(id="Cx", rule="r", impl=("experiments.benchmark.obs.cloud_c1d",))
    c = settings.CloudLevel(id="Cx", rule="r2", impl=("experiments.benchmark.obs.cloud_c1",))
    assert len({a.sha, b.sha, c.sha}) == 3 and a.sha == settings.entry_sha(a)
    m = settings.CloudLevel(id="Cx", rule="r", impl=("experiments.benchmark.obs.no_such",))
    assert settings.impl_source("experiments.benchmark.obs.no_such") == \
        "<missing experiments.benchmark.obs.no_such>"
    assert m.sha != a.sha
    # CRLF does not change a source hash
    assert "\r\n" not in settings.impl_source("experiments.benchmark.obs.cloud_c1")
    # the level builders exist (levels.py v2) and are hashed by their source
    for name in ("level_lt", "level_lm", "level_lg"):
        assert settings.impl_source(f"experiments.benchmark.levels.{name}").startswith(
            f"def {name}(")


# ─────────────────────────────────────────────────────────────────────────────
#  cloud rules: truth tables, buffers, OCM, scene rules
# ─────────────────────────────────────────────────────────────────────────────

def test_pixel_rule_truth_tables():
    want = {"C0": set(range(1, 12)), "C1": {4, 5, 6, 7}, "C1d": {2, 4, 5, 6, 7},
            "C1s": {4, 5, 6}, "C4": {2, 4, 5, 6, 7, 11}}
    for lv, ok in want.items():
        v = obs.pixel_valid(lv, CODES)
        assert {int(c) for c, x in zip(CODES, v) if x} == ok, lv
    assert np.array_equal(obs.cloud_c1(CODES), obs.mask_valid(CODES, "marea_clear"))
    assert np.array_equal(obs.cloud_c4(CODES), obs.item_valid(CODES))
    # raw int16 SCL is accepted (fill -> no data)
    raw = np.array([4, FILL, 0, 6], np.int16)
    assert obs.pixel_valid("C1", raw).tolist() == [True, False, False, True]
    with pytest.raises(ValueError, match="scene rule"):
        obs.pixel_valid("C2", CODES)
    with pytest.raises(ValueError, match="C3 needs CLD"):
        obs.pixel_valid("C3", CODES)


def _brute_within(mask, r):
    H, W = mask.shape
    out = np.zeros_like(mask)
    pts = np.argwhere(mask)
    for i in range(H):
        for j in range(W):
            if pts.size and (((pts - [i, j]) ** 2).sum(axis=1) <= r * r).any():
                out[i, j] = True
    return out


def test_disk_within_and_opening():
    rng = np.random.default_rng(1)
    m = rng.random((23, 19)) < 0.03
    for r in (2, 4, 8):
        assert np.array_equal(obs.within(m, r), _brute_within(m, r))
    assert obs.disk(2).sum() == 13 and not obs.within(np.zeros((4, 4), bool), 3).any()
    # opening removes specks, keeps blobs, and a blob at the crop edge survives
    a = np.zeros((30, 30), bool)
    a[3, 3] = True                          # speck
    a[10:17, 10:17] = True                  # 7 x 7 blob
    a[0:6, 24:30] = True                    # blob at the corner of the crop
    o = obs.binary_open(a, 2)
    assert not o[3, 3] and o[13, 13] and o[0, 29] and o[2, 26]


def test_c3_and_c5_buffers():
    H, W = 40, 40
    scl = np.full((H, W), 4, np.uint8)
    cld = np.zeros((H, W), np.uint8)
    cld[5, 5] = 50                           # isolated probable-cloud pixel: opened away
    cld[20:27, 20:27] = 20                   # CLD == p is bad (>=)
    scl[2, 35] = 3                           # shadow: buffered 4 px
    scl[38, 2] = 1                           # saturated: masked, not buffered
    scl[38, 38] = 255                        # no data: masked, not cloud
    cld[0, 0] = obs.CLD_NODATA_U8            # CLD fill: masked
    v = obs.pixel_valid("C3", scl, cld)
    assert v[5, 5] and v[5, 6]               # the speck does not mask
    assert not v[23, 23] and not v[23, 34] and v[23, 35]       # blob + 8 px buffer (34 = 26 + 8)
    assert not v[2, 31] and v[2, 30]         # shadow buffer 4 px
    assert not v[38, 2] and v[37, 2] and not v[38, 38] and v[37, 37] and not v[0, 0]
    bad = obs.binary_open((cld >= 20) & (cld != obs.CLD_NODATA_U8), 2)
    ref = ~(_brute_within(bad, 8) | _brute_within(scl == 3, 4) | np.isin(scl, (0, 1, 255))
            | (cld == obs.CLD_NODATA_U8))
    assert np.array_equal(v, ref)
    # p from the cross-tab
    v35 = obs.pixel_valid("C3", scl, cld, params={"p": 35})
    assert v35[23, 23]
    # SCL 10 counts as bad whatever CLD says
    s10 = scl.copy()
    s10[10:17, 30:37] = 10
    assert not obs.pixel_valid("C3", s10, cld)[13, 33]
    # stacked scenes equal scene by scene
    st = obs.pixel_valid("C3", np.stack([scl, s10]), np.stack([cld, cld]))
    assert np.array_equal(st[1], obs.pixel_valid("C3", s10, cld))
    # C5: the C4 blocklist plus opened/dilated SCL 8-10 and dilated SCL 3
    s5 = np.full((H, W), 5, np.uint8)
    s5[10:17, 10:17] = 9
    s5[30, 30] = 8                           # speck: opened away
    s5[2, 2] = 3
    s5[35, 5] = 11                           # snow kept by C4/C5
    v5 = obs.pixel_valid("C5", s5)
    assert not v5[13, 24] and v5[13, 25] and v5[30, 31] and not v5[30, 30]
    assert not v5[2, 6] and v5[2, 7] and v5[35, 5]


def test_c11_through_a_stub_model():
    rng = np.random.default_rng(2)
    H, W = 16, 20
    raw = {b: rng.integers(100, 3000, (H, W)).astype(np.int16) for b in ("B03", "B04", "B8A")}
    raw["B04"][:4, :] = 9000                 # "cloud" in red
    raw["B03"][10, 10] = FILL                # no data
    calls = []

    def stub(x):
        calls.append(x)
        assert x.shape == (3, H, W) and x.dtype == np.float32
        cls = np.zeros((1, H, W), np.uint8)
        cls[0, x[0] > 5000] = 1
        cls[0, 6, :] = 3
        return cls

    ocm = obs.ocm_classes(raw, offset=0, predictor=stub)
    x = calls[0]
    # order red, green, NIR, floor0 values, no data (any band) written as 0
    ok = np.ones((H, W), bool)
    ok[10, 10] = False
    for k, b in enumerate(("B04", "B03", "B8A")):
        assert np.array_equal(x[k][ok], obs.prep_band(raw[b])[ok])
    assert (x[:, 10, 10] == 0).all() and ocm[10, 10] == obs.OCM_NODATA_U8
    assert (ocm[:4] == 1).all() and (ocm[6] == 3).all()
    # the offset is removed before the model
    raw_off = {b: np.where(a == FILL, a, a + 1000).astype(np.int16) for b, a in raw.items()}
    obs.ocm_classes(raw_off, offset=1000, predictor=stub)
    assert np.array_equal(calls[1], calls[0])
    scl = np.full((H, W), 4, np.uint8)
    scl[0, 0] = 0
    v = obs.pixel_valid("C11", scl, ocm=ocm)
    assert np.array_equal(v, (ocm == 0) & (scl != 0))
    with pytest.raises(ValueError, match="C11 needs"):
        obs.pixel_valid("C11", scl)
    with pytest.raises(ValueError, match="classes outside"):
        obs.ocm_classes(raw, predictor=lambda a: np.full((1, H, W), 7))
    with pytest.raises(ValueError, match="returned"):
        obs.ocm_classes(raw, predictor=lambda a: np.zeros((H, W + 1)))


def test_scene_rules_equal_the_production_functions():
    rng = np.random.default_rng(3)
    fb = np.concatenate([rng.random(200) * 0.3, [0.05, 0.0500001, 0.1, 0.1000001]])
    tb = np.concatenate([rng.random(200) * 0.3, [0.2, 0.1, 0.0, 0.1]])
    keep = obs.scene_c2(fb, tb)
    # stability's own path: percentages from reference_and_clouds' rule
    pct = {f"d{i}": (round(float(f) * 100, 4) if f <= 0.05 else round(float(t) * 100, 4))
           for i, (f, t) in enumerate(zip(fb, tb))}
    kept = set(stability.filter_dates(pct, 0.10))
    assert [f"d{i}" in kept for i in range(len(fb))] == keep.tolist()
    assert obs.scene_c2(0.04, 0.9) is True and obs.scene_c2(0.06, 0.11) is False
    # C6, C7 on synthetic frames
    for _ in range(30):
        s = rng.choice(np.arange(12), (20, 20), p=np.r_[[0.01] * 3, [0.3, 0.3, 0.2, 0.06],
                                                       [0.02] * 3, [0.02], [0.03]] /
                       np.r_[[0.01] * 3, [0.3, 0.3, 0.2, 0.06], [0.02] * 3, [0.02], [0.03]].sum())
        s8 = s.astype(np.uint8)
        s8[0, :int(rng.integers(0, 3))] = 255
        cl, nd = g21.scene_frame_shares(s8, fill=255)
        assert obs.scene_c6(s8) == bool(cl < 0.10 and nd <= 0.02)
        assert obs.scene_c7(s8) == (wc.frame_good_fraction(s8) >= 0.5)
    # C8 on keep_full
    v = np.zeros((10, 10), bool)
    v[:5] = True
    region = np.zeros((10, 10), bool)
    region[3:7] = True
    assert obs.scene_c8(v, region) is True                       # 2 of 4 rows: 0.5
    assert obs.scene_c8(v, np.flatnonzero(region.ravel())) is True
    assert obs.scene_c8(v, region, 0.51) is False
    assert obs.scene_bad_share(np.full((4, 4), 9, np.uint8)) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  indices and the 20 m rule
# ─────────────────────────────────────────────────────────────────────────────

def _brute_block_mean(a, phase):
    H, W = a.shape
    py, px = phase
    by = (np.arange(H) + py) // 2
    bx = (np.arange(W) + px) // 2
    out = np.zeros((by.max() + 1, bx.max() + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = a[np.ix_(by == i, bx == j)].astype(np.float64).mean()
    return out.astype(np.float32)


@pytest.mark.parametrize("phase", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_block_mean_take_expand(phase):
    rng = np.random.default_rng(4)
    a = rng.integers(0, 5000, (13, 10)).astype(np.float32)
    assert np.array_equal(obs.block_mean20(a, phase), _brute_block_mean(a, phase))
    a2 = a.copy()
    a2[4, 4] = np.nan
    m = obs.block_mean20(a2, phase)
    by, bx = (4 + phase[0]) // 2, (4 + phase[1]) // 2
    assert np.isnan(m[by, bx]) and np.isfinite(m).sum() == m.size - 1
    blocks = rng.integers(0, 100, m.shape)
    rep = obs.expand20(blocks, phase, a.shape)
    assert np.array_equal(obs.block_take20(rep, phase), blocks)
    assert np.array_equal(obs.block_mean20(rep, phase), blocks.astype(np.float32))
    st = np.stack([a, a + 1])
    assert np.array_equal(obs.block_mean20(st, phase)[1], obs.block_mean20(a + 1, phase))


def _bands(seed=5, shape=(14, 12), phase=(1, 0)):
    rng = np.random.default_rng(seed)
    raw = {b: rng.integers(-200, 4000, shape).astype(np.int16)
           for b in ("B02", "B03", "B04", "B08")}
    hb = obs._block_index(shape[0], phase[0])[1]
    wb = obs._block_index(shape[1], phase[1])[1]
    for b in ("B8A", "B11", "B12"):
        raw[b] = obs.expand20(rng.integers(-100, 3500, (hb, wb)), phase, shape).astype(np.int16)
    raw["B03"][0, 0] = 0
    raw["B11"][0, 0] = 0                      # zero sum at a 20 m block? (B03 mean is not 0)
    raw["B08"][1, 1] = FILL
    return raw


@pytest.mark.parametrize("prep", ["floor0", "raw"])
def test_index_formulas(prep):
    phase = (1, 0)
    raw = _bands(phase=phase)
    pb = obs.prepare(raw, obs.SPECTRAL_BANDS, 0, prep)
    # NDWI: bitwise = water.index_block on the same bands and = obs.ndwi
    x = obs.compute_index(pb, "ndwi", phase)
    ref = water.index_block("ndwi", {"B03": pb["B03"], "B08": pb["B08"]})
    assert np.array_equal(x.view(np.int32), ref.view(np.int32))
    g, n = obs.bands(raw["B03"], raw["B08"], mode=prep)
    assert np.array_equal(x.view(np.int32), obs.ndwi(g, n).view(np.int32))
    # 20 m indices: on the blocks, then replicated
    blk = {b: obs.block_mean20(pb[b], phase) for b in ("B02", "B03", "B04", "B08")}
    blk.update({b: obs.block_take20(pb[b], phase) for b in ("B8A", "B11", "B12")})
    exp = lambda a: obs.expand20(a, phase, x.shape)        # noqa: E731
    m = water.index_block("mndwi", {"B03": blk["B03"], "B11": blk["B11"]})
    assert np.array_equal(obs.compute_index(pb, "mndwi", phase).view(np.int32),
                          exp(m).view(np.int32))
    n20 = water.index_block("ndwi", {"B03": blk["B03"], "B08": blk["B8A"]})
    assert np.array_equal(obs.compute_index(pb, "ndwi20", phase).view(np.int32),
                          exp(n20).view(np.int32))
    # AWEI_nsh = water.index_block's "awei" (its DN scale heuristic applies here)
    aw = water.index_block("awei", {"B03": blk["B03"], "B08": blk["B08"], "B11": blk["B11"],
                                    "B12": blk["B12"]})
    np.testing.assert_allclose(obs.compute_index(pb, "awei_nsh", phase), exp(aw),
                               rtol=1e-5, atol=1e-6, equal_nan=True)
    # float64 references for the others
    r = {b: v.astype(np.float64) / 10000 for b, v in blk.items()}
    sh = r["B02"] + 2.5 * r["B03"] - 1.5 * (r["B08"] + r["B11"]) - 0.25 * r["B12"]
    np.testing.assert_allclose(obs.compute_index(pb, "awei_sh", phase), exp(sh), rtol=1e-5,
                               atol=1e-6, equal_nan=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        wri = np.clip((r["B03"] + r["B04"]) / (r["B08"] + r["B11"]), 0, 10)
    np.testing.assert_allclose(obs.compute_index(pb, "wri", phase), exp(wri), rtol=1e-5,
                               equal_nan=True)
    nir = obs.compute_index(pb, "nir", phase)
    np.testing.assert_allclose(nir, -pb["B08"].astype(np.float64) / 10000, rtol=1e-6,
                               equal_nan=True)
    assert all(obs.compute_index(pb, i, phase).dtype == np.float32 for i in settings.INDEX)
    # the fill propagates: NaN index (no observation)
    assert np.isnan(x[1, 1]) and np.isnan(nir[1, 1])
    if prep == "floor0":
        assert np.nanmin(obs.compute_index(pb, "wri", phase)) >= 0
        assert np.nanmax(np.abs(obs.compute_index(pb, "mndwi", phase))) <= 1


def test_twenty_m_rule_without_a_lattice_uses_box_means_at_phase_00():
    raw = _bands(phase=(1, 0))
    pb = obs.prepare(raw, obs.SPECTRAL_BANDS, 0, "floor0")
    x = obs.compute_index(pb, "mndwi", None)
    b3 = obs.block_mean20(pb["B03"], (0, 0))
    b11 = obs.block_mean20(pb["B11"], (0, 0))             # a mean, not a take
    want = obs.expand20(obs.ndwi(b3, b11), (0, 0), x.shape)
    assert np.array_equal(x.view(np.int32), want.view(np.int32))
    assert not np.array_equal(np.nan_to_num(x), np.nan_to_num(obs.compute_index(pb, "mndwi", (1, 0))))
    with pytest.raises(ValueError):
        obs.compute_index(pb, "mndwi", (2, 0))


def test_standardisation_params():
    assert settings.index_params("ndwi") == {"t_k": 0.0, "s_k": 1.0}
    assert settings.index_params("nir", t_nir=0.12, s_k=0.4) == {"t_k": -0.12, "s_k": 0.4}
    assert settings.index_params("wri", s_k=2.0) == {"t_k": 1.0, "s_k": 2.0}
    with pytest.raises(ValueError, match="pass t_nir"):
        settings.index_params("nir", s_k=1.0)
    with pytest.raises(ValueError, match="pass s_k"):
        settings.index_params("mndwi")
    with pytest.raises(ValueError):
        settings.index_params("ndwi", s_k=2.0)
    with pytest.raises(ValueError):
        settings.index_params("mndwi", s_k=0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Otsu, D2, D3
# ─────────────────────────────────────────────────────────────────────────────

def _bimodal(n=20000, lo=-0.4, hi=0.3, sd=0.06, share=0.4, seed=6):
    rng = np.random.default_rng(seed)
    k = int(n * share)
    return np.concatenate([rng.normal(hi, sd, k), rng.normal(lo, sd, n - k)])


def test_otsu_equals_skimage():
    from skimage.filters import threshold_otsu

    for seed in range(5):
        v = _bimodal(seed=seed)
        t, eta, m = obs.otsu(v, 256)
        assert t == pytest.approx(float(threshold_otsu(v, nbins=256)), abs=1e-12)
        assert eta > 0.85 and m == pytest.approx(0.4, abs=0.02)
    u = np.random.default_rng(9).normal(0, 1, 20000)
    assert obs.otsu(u)[1] < 0.75
    assert np.isnan(obs.otsu(np.ones(10))[0]) and np.isnan(obs.otsu([])[0])
    # pooled per-year histograms give the pooled Otsu
    v1, v2 = _bimodal(seed=1), _bimodal(seed=2)
    edges = np.linspace(-1, 1, 2001)
    c = np.histogram(v1, edges)[0] + np.histogram(v2, edges)[0]
    tp, _, _ = obs.otsu_hist(c, (edges[:-1] + edges[1:]) / 2)
    # well-separated modes: any cut in the gap is optimal; Otsu takes the first
    assert -0.25 < tp < 0.1
    v = np.concatenate([v1, v2])
    assert np.mean((v > tp) != (v > -0.05)) < 0.002


def test_d2_guards():
    rng = np.random.default_rng(7)
    H, W = 100, 100
    x = np.where(np.arange(W)[None, :] < 40, rng.normal(0.25, 0.05, (H, W)),
                 rng.normal(-0.35, 0.05, (H, W))).astype(np.float32)
    V = np.ones((H, W), bool)
    om = np.ones((H, W), bool)
    r = obs.d2_threshold(x, V, om, 0.0)
    assert r["used"] and -0.25 < r["t"] < 0.1 and r["eta"] >= 0.75 and r["n"] == H * W
    assert np.mean((x > r["t"]) != (np.arange(W)[None, :] < 40)) < 0.002
    # too few valid pixels in Omega -> t_k
    small = np.zeros((H, W), bool)
    small[:20, :20] = True
    r = obs.d2_threshold(x, V, small, 0.0)
    assert not r["used"] and r["t"] == 0.0 and r["n"] == 400
    # minority below 0.15 -> t_k
    x2 = x.copy()
    x2[:, 5:40] = rng.normal(-0.35, 0.05, (H, 35))
    r = obs.d2_threshold(x2, V, om, 0.0)
    assert r["minority"] < 0.15 and not r["used"] and r["t"] == 0.0
    # unimodal -> eta low -> t_k
    r = obs.d2_threshold(rng.normal(0, 0.1, (H, W)), V, om, 0.0)
    assert r["eta"] < 0.75 and not r["used"]
    # WRI: Otsu on log(x), threshold back in x units
    xw = np.exp(np.where(np.arange(W)[None, :] < 40, rng.normal(0.8, 0.1, (H, W)),
                         rng.normal(-0.9, 0.1, (H, W))))
    r = obs.d2_threshold(xw, V, om, 1.0, log_hist=True)
    assert r["used"] and 0.5 < r["t"] < 2.0


def _edge_scene(seed=8, H=120, W=120, t_true=-0.2):
    """Water (x ~ t_true + 0.3 = 0.1, so t_k = 0 cuts its tail) left of a
    wavy shoreline, land (t_true - 0.3) right of it."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    shore = 60 + 8 * np.sin(yy / 9.0)
    x = np.where(xx < shore, t_true + 0.3, t_true - 0.3) + rng.normal(0, 0.05, (H, W))
    return x.astype(np.float32)


def test_d3_edge_otsu_on_synthetic_edges():
    x = _edge_scene()
    V = np.ones(x.shape, bool)
    V[:10, :] = False
    r = obs.d3_threshold(x, V, 0.0)
    assert r["used"] and not r["skipped"] and r["n_edges"] >= 50 and r["n_buffer"] >= 500
    # the cut falls in the gap between the modes (-0.5 and 0.1), below t_k = 0
    assert -0.45 < r["t"] < -0.02
    yy, xx = np.mgrid[:120, :120]
    water = xx < 60 + 8 * np.sin(yy / 9.0)
    err_d3, err_tk = np.mean((x > r["t"]) != water), np.mean((x > 0.0) != water)
    assert err_d3 < 0.002 and err_tk > 0.005       # t_k = 0 cuts the water's tail
    # few valid pixels -> skipped, t_k
    Vs = np.zeros(x.shape, bool)
    Vs[:20, :20] = True
    r = obs.d3_threshold(x, Vs, 0.0)
    assert r["skipped"] and r["t"] == 0.0 and not r["used"]
    # no edge (all land) -> t_k
    r = obs.d3_threshold(np.full(x.shape, -0.5, np.float32), np.ones(x.shape, bool), 0.0)
    assert not r["used"] and r["t"] == 0.0 and r["n_edges"] == 0
    # a short edge (< 50 px) is dropped: a small water pond
    xp = np.full((120, 120), -0.5, np.float32)
    xp[50:55, 50:55] = 0.4
    r = obs.d3_threshold(xp, np.ones(xp.shape, bool), 0.0)
    assert r["n_edges"] == 0 and not r["used"]


# ─────────────────────────────────────────────────────────────────────────────
#  windows and subsets
# ─────────────────────────────────────────────────────────────────────────────

def test_windows():
    w = settings.window("W3")
    assert (w.start, w.end, w.years, w.centre, w.num) == ("2023-01-01", "2026-01-01",
                                                          (2023, 2024, 2025), 2024.5, 3)
    assert settings.window("W9").start == "2017-01-01" and settings.window("W1").years == (2025,)
    assert settings.window("Y2019").end == "2020-01-01" and settings.window_num("Y2019") == 2019
    v = settings.window("V2021L5")
    assert (v.start, v.end, v.centre, v.sites) == ("2019-01-01", "2024-01-01", 2021.5,
                                                   ("escalda",))
    assert settings.window("V2019L3").sites == ("escalda", "wadden")
    assert settings.window_num("V2021L3") == 20213
    assert settings.survey_window("V2024L1").years == (2024,)
    with pytest.raises(ValueError):
        settings.survey_window("V2021L2")
    with pytest.raises(KeyError):
        settings.window("W4")
    nums = [w.num for w in settings.WINDOW.values()]
    assert len(nums) == len(set(nums))
    assert all(settings.UNION_PERIOD[0] <= w.start and w.end <= settings.UNION_PERIOD[1]
               for w in settings.WINDOW.values())


def test_curve_grids():
    k1 = settings.CURVES["K1"]
    assert k1.grid == ((20, 5, "d"), (45, 5, "d"), (90, 3, "d"), (180, 2, "d"), (330, 2, "n"),
                       ("all", 1, "all"))
    # the expected kinds hold at the W3 record (T = 465)
    for N, R, kind in k1.grid:
        assert settings.subset_kind(465, N, R)[0] == {"d": "d", "n": "n", "all": "a"}[kind]
    for N, R, kind in settings.CURVES["K4"].grid:
        assert settings.subset_kind(1300, N, R)[0] == {"d": "d", "n": "n", "all": "a"}[kind]
    assert settings.CURVES["K2"].mode == "F" and settings.CURVES["K9"].family == "win"
    # every rand grid's expected kind holds at its window's typical T
    T_of = {"W1": 172, "W2": 317, "W3": 465, "W5": 750, "W9": 1300}
    for c in settings.CURVES.values():
        if c.family != "rand":
            continue
        for N, R, kind in c.grid:
            assert settings.subset_kind(T_of[c.windows[0]], N, R)[0] == \
                {"d": "d", "n": "n", "all": "a"}[kind], (c.id, N, R)
        assert "experiments.benchmark.settings.rand_subsets" in c.impl
    with pytest.raises(ValueError):
        settings.subset_kind(100, 120, 1)


def test_rand_subsets_disjoint_and_nested():
    site, wid, T = "escalda", "W3", 465
    kind, sels = settings.rand_subsets(site, wid, T, 90, 3)
    assert kind == "disjoint" and len(sels) == 3
    perm0 = np.random.default_rng([20261008, config.SITE_ID[site], 3, 0]).permutation(T)
    for r, s in enumerate(sels):
        assert np.array_equal(s, np.sort(perm0[r * 90:(r + 1) * 90])) and s.dtype == np.int64
    assert len(np.unique(np.concatenate(sels))) == 270            # no overlap
    kind, sels = settings.rand_subsets(site, wid, T, 330, 2)
    assert kind == "nested"
    for r, s in enumerate(sels):
        pr = np.random.default_rng([20261008, config.SITE_ID[site], 3, r]).permutation(T)
        assert np.array_equal(s, np.sort(pr[:330]))
    kind, sels = settings.rand_subsets(site, wid, T, "all", 1)
    assert kind == "all" and np.array_equal(sels[0], np.arange(T))
    # windows and sites have their own draws
    assert not np.array_equal(settings.rand_subsets(site, "W1", T, 90, 1)[1][0],
                              settings.rand_subsets(site, "W3", T, 90, 1)[1][0])
    assert not np.array_equal(settings.rand_subsets("ems", wid, T, 90, 1)[1][0],
                              settings.rand_subsets(site, wid, T, 90, 1)[1][0])


def test_win_and_matched_subsets():
    s = settings.win_subset("ems", "W3", 465, 45, 2)
    st = int(np.random.default_rng([20261012, config.SITE_ID["ems"], 3, 45, 2]).integers(0, 421))
    assert np.array_equal(s, np.arange(st, st + 45))
    pos = np.arange(0, 400, 2)
    m = settings.matched_subsets("ems", "W3", pos, 37, draws=3)
    assert len(m) == 3 and all(len(x) == 37 and set(x) <= set(pos) for x in m)
    rng = np.random.default_rng([20261016, config.SITE_ID["ems"], 3, 37, 1])
    assert np.array_equal(m[1], np.sort(pos[rng.permutation(len(pos))[:37]]))
    assert settings.matched_subsets("ems", "W3", pos, 37, draws=1)[0].tolist() == m[0].tolist()
    with pytest.raises(ValueError):
        settings.matched_subsets("ems", "W3", pos, 500)
