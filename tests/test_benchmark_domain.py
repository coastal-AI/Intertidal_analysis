"""Tests of experiments.benchmark.domain, the pixel domain and the geometry.

On the synthetic site of ``tests/benchmark_site.py`` (a built package, the
domain thresholds lowered to the tiny frame) the tests pin the rules of the
module docstring (v13 spec §1.5, §2.6, §3.1; v2 spec §2.10):

* the extraction rule (clr > 30, 0.10 < wf < 0.90, sea wf >= 0.90) on
  counts equals ``marea.extract``'s formula, whatever the counts' dtype;
* ``for_subset(arange(T))`` equals the full-record domain bitwise, and that
  equals the copy the package stored;
* the A-mode domain of a subset equals a direct computation of the rule on
  the subset's own frames (keep, sea from the frame validity bits outside
  dom, s, bands); the frame clear counts equal a direct count;
* ``no_intertidal`` (fewer keep pixels than the floor) and ``no_anchor``
  (too few finite s: one band, band 0) statuses;
* the strata rule ``band_map_frame``: merged bands of raw bands, the
  nearest lower raw band for raw bands without keep pixels, band 0 below
  the first, clipping beyond the last, -1 for NaN s;
* the production geometry reproduces the s_km the production extraction
  stored (G-keep);
* ``get``: F-mode, observation sensitivities and the whole record return
  the full domain; A-mode subsets are cached in memory and on disk, and the
  disk cache is keyed by the package and this module.

Validation v2 (the domain of an instance, spec §2.9, §2.10), on the synthetic
Zarr site of ``tests/benchmark_zarr_site.py``: P0 over the whole W3 window is
keep_full; any pixel setting over a whole window uses its own full-frame
counters (keep ∩ dom, the dropped pixels counted); a subset (P0, C11)
equals the rule on the subset's own frames, sea exact outside dom; a
threshold detector counts at dom and takes P0's sea outside dom
(declared); a scene rule uses its kept record; selections are validated;
``get_v2`` caches by (setting, window, subset) in memory and on disk.

Run:  python -X utf8 -m pytest tests/test_benchmark_domain.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import domain, obs, product  # noqa: E402
from pyintertidal import geometry  # noqa: E402
from pyintertidal import tide_estimators as te  # noqa: E402
from tests.benchmark_site import (  # noqa: E402
    FILL, H, PIXEL_M, RES, SITE, W, _bits, make_site, record_frames)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    s = make_site(tmp_path_factory.mktemp("v13_domain"), mp)
    yield s
    s.pkg.close()
    mp.undo()


def _direct(inp, pkg, sel, rule=obs.MASK_RULE, mode=obs.INDEX_MODE):
    """Counts of the record scenes ``sel`` straight from the cube arrays."""
    b03, b08, scl = record_frames(inp, pkg)
    nd, V = obs.observe(b03[sel], b08[sel], obs.scl_u8(scl[sel], FILL), FILL, rule, mode)
    return V.sum(axis=0), obs.wet(nd, V).sum(axis=0)


def _extract_rule(clr, wet):
    """marea.extract's lines, verbatim in effect."""
    with np.errstate(invalid="ignore", divide="ignore"):
        wf = np.where(clr > 30, wet / np.maximum(clr, 1), np.nan)
    inter = np.isfinite(wf) & (wf > 0.10) & (wf < 0.90)
    sea = np.isfinite(wf) & (wf >= 0.90)
    return inter, sea


def _unit(pkg, sel, mode="A", sens=None, N=None):
    sel = np.asarray(sel, np.int64)
    return product.Unit(site=SITE, res_m=RES, group="marea_pair", variant="",
                        family="rand", mode=mode, N=N or len(sel), rep=0, sens=sens or {},
                        sel=sel)


# ─────────────────────────────────────────────────────────────────────────────
#  the rule
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_is_the_extraction_rule():
    rng = np.random.default_rng(1)
    clr = rng.integers(0, 80, (50, 40))
    wet = np.minimum(rng.integers(0, 80, (50, 40)), clr)
    clr[0, :4], wet[0, :4] = 31, [3, 4, 27, 28]           # the edges of the rule
    clr[1, :3], wet[1, :3] = 30, [3, 15, 30]               # not enough clear
    inter_w, sea_w = _extract_rule(clr, wet)
    for dt in (np.int16, np.int32, np.uint8, np.int64):
        inter, sea = domain.classify(clr.astype(dt), wet.astype(dt))
        assert np.array_equal(inter, inter_w) and np.array_equal(sea, sea_w), dt
    assert not inter[0, 0] and inter[0, 1] and inter[0, 2] and not inter[0, 3]
    assert sea[0, 3] and not inter[1, :3].any() and not sea[1, :3].any()
    assert domain.EXTRACT_RULE == {"min_clear": 31, "wf_lo": 0.10, "wf_hi": 0.90}


def test_positions_in_and_domain_validation():
    dom = np.array([2, 5, 9, 14])
    assert np.array_equal(domain.positions_in(dom, [5, 14, 2]), [1, 3, 0])
    assert domain.positions_in(dom, []).dtype == np.int64
    for bad in ([3], [15], [1, 5]):
        with pytest.raises(KeyError):
            domain.positions_in(dom, bad)
    z = np.zeros((2, 2), bool)
    with pytest.raises(ValueError):
        domain.Domain("X", [], None, z, z, [], [], [0.0], [], 1, "ok")
    with pytest.raises(ValueError):
        domain.Domain("A", [], None, z, z, [], [], [0.0], [], 1, "done")
    with pytest.raises(ValueError):
        domain.assemble("A", [3, 1], z, z, 20.0, "west")


def test_held_at_full():
    assert not domain.held_at_full({})
    assert not domain.held_at_full(None)
    assert not domain.held_at_full({"mask": obs.MASK_RULE, "index": obs.INDEX_MODE})
    assert domain.held_at_full({"mask": "blocklist"})
    assert domain.held_at_full({"index": "raw"})
    assert not domain.held_at_full({"floor": 80})


# ─────────────────────────────────────────────────────────────────────────────
#  full record and subsets on the synthetic package
# ─────────────────────────────────────────────────────────────────────────────

def test_full_record_and_whole_subset_are_the_stored_domain(site):
    pkg = site.pkg
    f = domain.full(pkg)
    assert f.status == "ok" and f.mode == "F" and f.n_sel == pkg.T and f.K >= 2
    assert f.equals(pkg.full, mode=True)
    a = domain.for_subset(pkg, np.arange(pkg.T))
    assert a.mode == "A" and a.equals(f) and not a.equals(f, mode=True)
    b = domain.for_subset(pkg, np.ones(pkg.T, bool))
    assert b.equals(a)
    # and both equal the rule on the full record's own frames
    clr, wet = _direct(site.inp, pkg, np.arange(pkg.T))
    inter, sea = _extract_rule(clr, wet)
    assert np.array_equal(f.keep, np.flatnonzero(inter.ravel()))
    assert np.array_equal(f.sea, sea) and np.array_equal(f.inter, inter)
    assert np.array_equal(pkg.dom[f.pos], f.keep)


@pytest.mark.parametrize("n,seed", [(70, 0), (60, 1), (45, 2)])
def test_a_mode_equals_the_rule_on_the_subset_frames(site, n, seed):
    pkg = site.pkg
    sel = np.sort(np.random.default_rng(seed).choice(pkg.T, n, replace=False))
    d = domain.for_subset(pkg, sel, with_frame=True)
    clr, wet = _direct(site.inp, pkg, sel)
    inter, sea = _extract_rule(clr, wet)
    keep = np.flatnonzero(inter.ravel())
    assert np.array_equal(d.keep, keep)
    assert np.array_equal(d.inter, inter)
    assert np.array_equal(d.sea, sea)                 # sea outside dom from the bits
    assert d.n_sel == n
    if len(keep) < domain.MIN_INTERTIDAL:
        assert d.status == "no_intertidal"
        return
    s_m = geometry.along_distance(sea | inter, geometry.mouth_seeds(sea, side="west"),
                                  PIXEL_M)
    s_km = s_m.ravel()[keep] / 1000.0
    assert np.array_equal(_bits(d.s_km), _bits(s_km))
    assert np.array_equal(_bits(d.s_frame), _bits(s_m / 1000.0))
    if np.isfinite(s_km).sum() > domain.MIN_FINITE_S:
        edges, centers, band_of = te.make_bands_fixed(s_km, 3.0, domain.MIN_PX_BAND)
        assert d.status == "ok"
        assert np.array_equal(d.band_of, band_of) and np.array_equal(_bits(d.centers),
                                                                     _bits(centers))
        assert np.array_equal(_bits(d.edges), _bits(np.asarray(edges, float)))
        assert d.K == len(centers)


def test_frame_clear_counts_equal_a_direct_count(site):
    pkg = site.pkg
    sel = np.arange(0, pkg.T, 3)
    clr, _ = _direct(site.inp, pkg, sel)
    for rows in (None, 1, 7):
        got = domain.frame_clear_counts(pkg, sel, rows_per_chunk=rows)
        assert got.dtype == np.int32 and np.array_equal(got, clr.ravel())


def test_subset_selection_is_validated(site):
    pkg = site.pkg
    for bad in ([3, 1], [0, 0], [-1, 2], [pkg.T], np.ones(pkg.T - 1, bool)):
        with pytest.raises(ValueError):
            domain.for_subset(pkg, np.asarray(bad))


def test_no_intertidal_and_no_anchor(site, monkeypatch):
    pkg = site.pkg
    small = domain.for_subset(pkg, np.arange(20))      # never more than 30 clear
    assert small.status == "no_intertidal" and small.n_keep == 0 and small.K == 1
    assert np.isnan(small.s_km).all() and not small.inter.any()
    # one pixel short of the floor: no geometry at all
    monkeypatch.setattr(domain, "MIN_INTERTIDAL", pkg.full.n_keep + 1)
    d = domain.full(pkg, with_frame=True)
    assert d.status == "no_intertidal" and d.n_keep == pkg.full.n_keep
    assert np.isnan(d.s_km).all() and np.isnan(d.s_frame).all()
    assert np.array_equal(d.band_of, np.zeros(d.n_keep)) and len(d.edges) == 0
    # enough keep pixels but too few finite s: one band, tau = 0
    monkeypatch.setattr(domain, "MIN_INTERTIDAL", 1)
    monkeypatch.setattr(domain, "MIN_FINITE_S", pkg.full.n_keep)
    d = domain.full(pkg)
    assert d.status == "no_anchor" and d.K == 1 and np.array_equal(d.centers, [0.0])
    assert (d.band_of == 0).all() and np.isfinite(d.s_km).any()
    assert np.array_equal(_bits(d.s_km), _bits(pkg.full.s_km))
    # no sea on the mouth side: NaN s everywhere, hence no anchor
    z = np.zeros((6, 8), bool)
    inter = z.copy()
    inter[2:4, 2:6] = True
    d = domain.assemble("A", np.flatnonzero(inter.ravel()), z, inter, 20.0, "west")
    assert d.status == "no_anchor" and np.isnan(d.s_km).all()


# ─────────────────────────────────────────────────────────────────────────────
#  strata
# ─────────────────────────────────────────────────────────────────────────────

def _toy(s_km, band_of, edges):
    n = len(s_km)
    z = np.zeros((1, max(n, 1)), bool)
    centers = np.zeros(max(band_of) + 1)
    return domain.Domain("F", np.arange(n), None, z, z, s_km=s_km, band_of=band_of,
                         centers=centers, edges=edges, K=len(centers), status="ok")


def test_band_map_frame_merge_rule():
    # raw bands 0..4 (edges every 3 km); raw 0 and 1 merged into band 0, raw 2
    # without keep pixels, raw 3 -> band 1, raw 4 -> band 2; one NaN s
    s = np.array([0.5, 2.9, 3.1, 5.0, 9.2, 11.0, 12.5, np.nan])
    b = np.array([0, 0, 0, 0, 1, 1, 2, -1])
    d = _toy(s, b, np.arange(0.0, 15.1, 3.0))
    assert np.array_equal(domain.raw_band_table(d), [0, 0, 0, 1, 2])
    q = np.array([[0.0, 2.99, 3.0, 6.5, 8.99], [9.0, 14.9, 40.0, np.nan, -0.5]])
    got = domain.band_map_frame(d, q)
    assert got.dtype == np.int16 and got.shape == q.shape
    assert got.tolist() == [[0, 0, 0, 0, 0], [1, 2, 2, -1, 0]]
    # the band of a keep pixel is its own band
    assert np.array_equal(domain.band_map_frame(d, s)[:-1], b[:-1])
    # a raw band without keep pixels below the first one maps to band 0
    d2 = _toy(np.array([7.0, 8.0, 10.0]), np.array([0, 0, 1]), np.arange(0.0, 12.1, 3.0))
    assert np.array_equal(domain.raw_band_table(d2), [0, 0, 0, 1])
    # a raw band split between two merged bands is refused
    d3 = _toy(np.array([1.0, 2.0]), np.array([0, 1]), np.array([0.0, 3.0]))
    with pytest.raises(AssertionError):
        domain.raw_band_table(d3)


def test_band_map_frame_without_bands():
    z = np.zeros((1, 3), bool)
    d = domain.Domain("F", [0, 1, 2], None, z, z, s_km=[1.0, 2.0, np.nan],
                      band_of=[0, 0, 0], centers=[0.0], edges=[], K=1, status="no_anchor")
    assert domain.band_map_frame(d, np.array([1.0, np.nan, 50.0])).tolist() == [0, -1, 0]
    with pytest.raises(ValueError):
        domain.raw_band_table(d)


def test_stored_band_frame_is_the_strata_rule(site):
    pkg = site.pkg
    f = domain.full(pkg, with_frame=True)
    bf = domain.band_map_frame(f, f.s_frame)
    assert np.array_equal(bf, pkg.band_frame)
    fin = np.isfinite(f.s_km)
    assert np.array_equal(bf.ravel()[f.keep][fin], f.band_of[fin])
    assert (bf[~np.isfinite(f.s_frame)] == -1).all()
    assert np.array_equal(pkg.s_frame, f.s_frame.astype(np.float32), equal_nan=True)


def test_production_geometry_reproduces_the_stored_s_km(site):
    """G-keep: s_km of the production extraction, recomputed from its keep,
    sea and inter, equals the float32 s_km the production run stored."""
    pkg, inp = site.pkg, site.inp
    p = domain.production(pkg)
    assert p.mode == "P" and np.array_equal(p.keep, inp["ex"]["keep"])
    assert np.array_equal(p.s_km.astype(np.float32).view(np.int32),
                          np.asarray(inp["s_km_ex"], np.float32).view(np.int32))
    assert np.array_equal(pkg.dom[p.pos], p.keep)


# ─────────────────────────────────────────────────────────────────────────────
#  get and its caches
# ─────────────────────────────────────────────────────────────────────────────

def test_get_returns_the_full_domain_when_the_rules_say_so(site):
    pkg = site.pkg
    sel = np.arange(0, pkg.T, 2)
    assert domain.get(pkg, _unit(pkg, sel, mode="F")) is pkg.full
    assert domain.get(pkg, _unit(pkg, sel, sens={"mask": "blocklist"})) is pkg.full
    assert domain.get(pkg, _unit(pkg, np.arange(pkg.T), N="all")) is pkg.full


def test_get_caches_a_mode_domains_in_memory_and_on_disk(site):
    pkg = site.pkg
    sel = np.sort(np.random.default_rng(11).choice(pkg.T, 66, replace=False))
    u = _unit(pkg, sel)
    pkg.__dict__.pop("_domain_cache", None)
    d = domain.get(pkg, u)
    path = domain.cache_path(pkg, u.subset_key)
    assert os.path.exists(path)
    assert domain.get(pkg, u) is d                     # memory
    assert d.equals(domain.for_subset(pkg, sel))
    pkg.__dict__.pop("_domain_cache", None)
    e = domain.get(pkg, u)                             # disk
    assert e is not d and e.equals(d) and np.array_equal(e.pos, d.pos)
    # the disk cache is keyed: another package key or a stale format is ignored
    meta = domain._cache_meta(pkg, u.subset_key, len(sel))
    assert domain.load_cached(path, meta, dom=pkg.dom).equals(d)
    assert domain.load_cached(path, dict(meta, package_key="other")) is None
    assert domain.load_cached(path, dict(meta, domain_sha="other")) is None
    assert domain.load_cached(path + ".absent", meta) is None
    with open(path, "wb") as f:                        # a damaged cache is rebuilt
        f.write(b"not a zip")
    assert domain.load_cached(path, meta) is None
    pkg.__dict__.pop("_domain_cache", None)
    assert domain.get(pkg, u).equals(d)
    pkg.__dict__.pop("_domain_cache", None)


def test_summary_is_json_ready(site):
    import json

    s = site.pkg.full.summary()
    assert json.loads(json.dumps(s)) == s
    assert s["n_keep"] == site.pkg.full.n_keep and s["K"] == site.pkg.full.K
    assert s["n_sea"] == int(site.pkg.full.sea.sum())
    assert H * W > s["n_keep"] > 0


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION V2: the harness domain of an instance (spec §2.9, §2.10)
# ═════════════════════════════════════════════════════════════════════════════

from experiments.benchmark import package, settings  # noqa: E402
from tests import benchmark_zarr_site as zs  # noqa: E402


@pytest.fixture(scope="module")
def zsite(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    s = zs.make_site(tmp_path_factory.mktemp("v2_domain"), mp)
    s.frames = zs.record_frames(s.inp, s.pkg)
    yield s
    s.pkg.close()
    mp.undo()


def _c11(zsite, sel):
    f, pkg = zsite.frames, zsite.pkg
    r0, r1, c0, c1 = (int(v) for v in pkg.support["ocm_region"])
    out = obs.cloud_c1(f["SCL"][sel]).copy()
    for k, q in enumerate(sel):
        raw = {b: f[b][q][r0:r1, c0:c1] for b in ("B04", "B03", "B8A")}
        cls = zs.fake_ocm_classes(raw, int(pkg.offsets[q]))
        out[k][r0:r1, c0:c1] = obs.cloud_c11(f["SCL"][q][r0:r1, c0:c1], cls)
    return out


def _counts(zsite, sid, sel):
    """Direct frame counts of a setting over record positions ``sel``."""
    f, pkg = zsite.frames, zsite.pkg
    s = settings.Setting.parse(sid)
    cv = _c11(zsite, sel) if s.pixel_rule == "C11" else None
    prm = pkg.index_params(s.index)
    thr = (pkg.scene_thresholds(s.detect, s.index, sel)
           if settings.DETECT[s.detect].per_scene else None)
    o = obs.observe_setting({b: a[sel] for b, a in f.items()},
                            s.replace(cloud=s.pixel_rule), offset=pkg.offsets[sel],
                            lattice=zs.PHASE, params=prm, threshold=thr, cloud_valid=cv)
    return o.V.sum(axis=0), o.W.sum(axis=0)


def test_v2_whole_window_p0_is_keep_full(zsite):
    pkg = zsite.pkg
    d = domain.for_setting(pkg, settings.P0, "W3")
    assert d.mode == "F" and d.equals(pkg.full)
    assert d.notes["counts"] == "full-frame uint8 year counters"
    assert d.notes["keep_outside_dom_dropped"] == 0


@pytest.mark.parametrize("sid", ["C4.floor0.ndwi.D1", "C1.floor0.mndwi.D1",
                                 "C1d.floor0.ndwi.D1", "C11.floor0.ndwi.D1",
                                 "C1.raw.ndwi.D1"])
def test_v2_whole_window_uses_the_settings_own_counts(zsite, sid):
    pkg = zsite.pkg
    for wid in ("W3", "Y2024"):
        sel = pkg.window(wid)
        clr, wet = _counts(zsite, sid, sel)
        inter, sea = _extract_rule(clr, wet)
        keep = np.intersect1d(np.flatnonzero(inter), pkg.dom)
        d = domain.for_setting(pkg, sid, wid)
        assert d.mode == "F" and np.array_equal(d.keep, keep), (sid, wid)
        assert np.array_equal(d.sea, sea)
        assert d.notes["keep_outside_dom_dropped"] == int(inter.sum()) - len(keep)
        assert np.array_equal(d.pos, pkg.positions(keep))


@pytest.mark.parametrize("sid", ["C1.floor0.ndwi.D1", "C11.floor0.ndwi.D1"])
def test_v2_subset_domain_is_the_rule_on_the_subset_frames(zsite, sid):
    pkg = zsite.pkg
    w3 = pkg.window("W3")
    rng = np.random.default_rng(4)
    for n in (40, 52):
        sel = np.sort(rng.choice(w3, n, replace=False))
        clr, wet = _counts(zsite, sid, sel)
        inter, sea = _extract_rule(clr, wet)
        d = domain.for_setting(pkg, sid, "W3", sel)
        assert d.mode == "A" and d.n_sel == n
        # U_rec of the window lies in dom: no subset keep pixel is lost
        assert np.array_equal(d.keep, np.flatnonzero(inter))
        assert np.array_equal(d.sea, sea)                       # exact outside dom too
        assert "exact" in d.notes["sea"]
        ref = domain.assemble("A", np.flatnonzero(inter), sea, inter, 10.0, "west",
                              dom=pkg.dom, n_sel=n)
        assert d.equals(ref)


def test_v2_threshold_setting_counts_at_dom(zsite):
    pkg = zsite.pkg
    sid = "C1.floor0.ndwi.D3"
    sel = pkg.window("W3")
    clr, wet = _counts(zsite, sid, sel)
    c = domain.setting_counts(pkg, sid, "W3")
    assert c["kind"] == "dom" and c["source"].startswith("at-dom year counters")
    assert np.array_equal(c["clr"], clr.ravel()[pkg.dom])
    assert np.array_equal(c["wet"], wet.ravel()[pkg.dom])
    d = domain.for_setting(pkg, sid, "W3")
    inter, sea = _extract_rule(clr.ravel()[pkg.dom], wet.ravel()[pkg.dom])
    assert np.array_equal(d.keep, pkg.dom[inter])
    sea_p0 = pkg.full.sea.ravel()
    outside = np.ones(pkg.H * pkg.W, bool)
    outside[pkg.dom] = False
    assert np.array_equal(d.sea.ravel()[outside], sea_p0[outside])
    assert "declared" in d.notes["sea"] and "exact" not in d.notes["sea"]


def test_v2_scene_rule_uses_its_kept_record(zsite):
    pkg = zsite.pkg
    sel = pkg.rule_subset("C6", "W3")
    assert 0 < len(sel) < len(pkg.window("W3"))
    clr, wet = _counts(zsite, "C1.floor0.ndwi.D1", sel)
    inter, sea = _extract_rule(clr, wet)
    d = domain.for_setting(pkg, "C6.floor0.ndwi.D1", "W3")
    assert d.mode == "A" and d.n_sel == len(sel)
    assert np.array_equal(d.keep, np.flatnonzero(inter)) and np.array_equal(d.sea, sea)


def test_v2_instance_selection_is_validated(zsite):
    pkg = zsite.pkg
    with pytest.raises(ValueError, match="outside window"):
        domain.instance_selection(pkg, settings.P0, "Y2024", pkg.window("Y2025")[:5])
    with pytest.raises(ValueError):
        domain.instance_selection(pkg, settings.P0, "W3", [5, 3])
    with pytest.raises(package.PackageError, match="needs the shard years"):
        domain.for_setting(pkg, settings.P0, "W5")


def test_v2_get_caches_by_setting_window_and_subset(zsite, tmp_path):
    import dataclasses
    import shutil
    shutil.copytree(zsite.site2.root, tmp_path / "copy")
    s2 = dataclasses.replace(zsite.site2, root=str(tmp_path / "copy").replace("\\", "/"))
    pkg = package.V2Package(s2)
    sel = pkg.window("W3")[::2]
    a = domain.get_v2(pkg, settings.P0, "W3", sel)
    assert domain.get_v2(pkg, settings.P0, "W3", sel) is a
    key = domain.cache_key_v2(settings.P0, "W3", product.subset_key(sel))
    path = os.path.join(pkg.dir, "_domain", pkg.dom_version, f"{key}.npz")
    assert os.path.exists(path)
    assert key != domain.cache_key_v2("C4.floor0.ndwi.D1", "W3", product.subset_key(sel))
    assert key != domain.cache_key_v2(settings.P0, "Y2024", product.subset_key(sel))
    pkg2 = package.V2Package(s2)                  # a new process: read from disk
    b = domain.get_v2(pkg2, settings.P0, "W3", sel)
    assert b is not a and b.equals(a) and b.mode == "A"
    whole = domain.get_v2(pkg2, settings.P0, "W3")
    assert whole.mode == "F" and whole.equals(pkg2.full)
    pkg.close()
    pkg2.close()
