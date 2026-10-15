"""Tests of experiments.benchmark.package, the common input package (spec §8).

The synthetic site of ``tests/benchmark_site.py`` is written as the
production inputs are: an int16 cube (_FillValue -32768, no scale, chunks
(1, 8, 8), a crs variable) with a sea on the west edge, a channel, tidal
flats rising eastward, land, and an isolated pond; SCL with every code, fill
and fully cloudy scenes; bands with fills, zero sums and negative values; a
production extraction made by ``marea.extract`` itself; an overpass file
with one date missing; a fake boundary provider with one non-finite level;
three judges (one mid-flat, one on the quay at the landward head, one
outside the frame); a transition-zone record. On it the tests pin:

* the record rule (period, overpass, finite level; ONE provider call);
* pass 1: the primary and U_rec counters, the frame validity bits and the
  scene statistics equal direct computations;
* the domains of validation v2 spec §2.10: S0 = NB ∪ dilate(keep_full,
  300 m) and dom = S0 ∪ dilate(U_rec, 3 px) ∪ keep_prod, recomputed by brute
  force, with every term exercised (the quay judge's neighbourhood reaches
  land beyond both dilations); the disk dilation itself; U_rec contains the
  A-mode keep of 200 random subsets under both mask rules (property test),
  and outside dom the A-mode sea is exactly "sea-like and > 30 clear";
* pass 2 stores the cube's raw values at dom, and the package reproduces
  marea.extract (raw, marea_clear) bitwise at keep_prod (G-keep);
* the level table equals per-shift calls, a lag beyond it is a live call,
  a batch-variant provider triggers the fallback;
* the subsets (formulas, nesting, stored draws, rule subsets);
* the manifest: any data byte changes the package key, a scoring-section
  edit does not, a package-section edit makes the package stale; the
  size+mtime check; load() refuses a tampered file; a full check finds a
  tampered file whose mtime was restored;
* resumable steps: a two-part build equals a one-shot build byte for byte,
  a finished package reruns nothing, a forced step makes the later ones
  stale;
* the defects of v2 spec §12.3 (milestone V0) stay fixed: both passes drop
  every memory map before they publish or clean up (WinError 32 on
  Windows), scene_stats.csv floats round-trip bit for bit (the g21 rule
  compares them with its thresholds), and load() re-checks the keyed files
  on every call.

Validation v2 (the year shards, spec §2.10, §13.1-13.4), on the synthetic
Zarr site of ``tests/benchmark_zarr_site.py`` (three yearly stores, 2025 with
+1000 in the values, 20 m bands on lattice phase (1, 0), a fake
OmniCloudMask): the record (instants, finite h_B, rising, offsets); the
uint8 pass-1 counters, frame bits and scene statistics equal the frame
observation of every key (C3 at both cut-offs, 20 m indices, D5); the
geometry, Omega, S0, the lattice-closed dom by brute force; dom_version
stable and frozen (refusals without D_truth or on a change); the OCM step
(the predictor's inputs, provenance, C11 inside the region and the C1
analog outside); the pixel-major tiles and cloud bits; the observation at
dom bitwise equal to the frame observation for 13 settings; the settings
pass (t_NIR, s_k, D2 and D3 per scene) against obs; the at-dom counters;
the MNAR table and the logistic slope; the scene rules, transition zone,
matched and rand subsets; shard manifests and keys; resume and staleness;
the large-dom refusal; the 20 m rule on columns. The legacy-cube source of
the local runs (B03, B08, SCL through NetCDFSource at the catalogue times)
builds the same record, counts only the keys of its bands, and gives the
Zarr package's P0 observation and C2 rule bit for bit.

The reviews of 2026-10-14 (before the server runs): a record is never
reused with other instants and rebuilds when its inputs change (instants of
its years, stores, offsets, IOC chunks), --years of an existing record only
selects per-year steps, missing stores and provisional offsets are refused
before anything is written; support, dom and pass 2 go stale when D_truth
or the OCM step arrive; freeze_dom refuses a partial or older D_truth; the
truths key on the package files and the level table on the record; the
free-space check, the fallocate reservation and the removal of a dead
run's temporaries; crops on whole 20 m blocks; the level sources (LM per
regional dataset and window, never zero); a killed OmniCloudMask year
resumes at its last time block with identical bits; superseded dom
versions are listed and pruned (never the current or a frozen one).

Run:  python -X utf8 -m pytest tests/test_benchmark_package.py -q
"""
import dataclasses
import json
import os
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, domain, levels, obs, package, product  # noqa: E402
from tests.benchmark_site import (  # noqa: E402
    CRS_EPSG, FILL, H, N_DATES, PIXEL_M, RES, SITE, W, X0, Y0, FakeTide, _bits, _bits32,
    make_inputs, make_site, record_frames, site_cfg, small_thresholds)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    s = make_site(tmp_path_factory.mktemp("v13_package"), mp)
    yield s
    s.pkg.close()
    mp.undo()


_record_frames = record_frames


def _direct_counts(b03, b08, scl, rule, mode="floor0"):
    nd, V = obs.observe(b03, b08, obs.scl_u8(scl, FILL), FILL, rule, mode)
    return V.sum(axis=0), obs.wet(nd, V).sum(axis=0), V


# ─────────────────────────────────────────────────────────────────────────────
#  the record
# ─────────────────────────────────────────────────────────────────────────────

def test_record_rule(site):
    inp, pkg = site.inp, site.pkg
    dates = np.array([str(d.date()) for d in inp["dates"]])
    in_period = (dates >= "2023-01-01") & (dates <= "2025-12-31")
    nan_date = str(inp["nan_instant"].date())
    want = np.flatnonzero(in_period & (dates != inp["no_time"]) & (dates != nan_date))
    assert np.array_equal(pkg.time_index, want)
    assert pkg.T == len(want) == N_DATES - 4 - 2 - 2
    assert np.array_equal(pkg.dates, dates[want])
    assert np.array_equal(pkg.t.asi8, inp["inst"][want].asi8)
    assert (np.diff(pkg.t.asi8) > 0).all() and pkg.t.tz is None
    assert np.array_equal(_bits(pkg.h0), _bits(FakeTide.value(pkg.t.asi8)))
    ex = pkg.record["exclusions"]
    assert ex["without_overpass"] == [inp["no_time"]]
    assert ex["nonfinite_level"] == [nan_date]
    assert ex["outside_period"] == dates[~in_period].tolist()
    # the levels came from ONE call on the sorted candidate instants
    cand = np.flatnonzero(in_period & (dates != inp["no_time"]))
    assert np.array_equal(site.provider.calls[0], inp["inst"][cand].asi8)
    assert pkg.record["boundary"]["name"] == "fake tide"
    assert pkg.record["grid"]["H"] == H and pkg.record["grid"]["W"] == W
    assert pkg.fill == FILL and pkg.shape == (H, W)


def test_record_refuses_unsorted_or_mismatched_dates(site, tmp_path):
    cfg = site.cfg
    with pytest.raises(ValueError):
        package.build_record(cfg, FakeTide(), ["2023-01-05", "2023-01-02"])
    with pytest.raises(ValueError):
        package.build_record(cfg, FakeTide(), ["2021-01-05"])     # nothing in the period


# ─────────────────────────────────────────────────────────────────────────────
#  pass 1
# ─────────────────────────────────────────────────────────────────────────────

def test_pass1_counters_bits_and_scene_stats(site):
    from pyintertidal import granadeiro2021 as g21
    from pyintertidal import waterline_contour as wc

    inp, pkg = site.inp, site.pkg
    b03, b08, scl = _record_frames(inp, pkg)
    clr, wet, V = _direct_counts(b03, b08, scl, "marea_clear")
    assert pkg.clr_n.dtype == np.int16
    assert np.array_equal(pkg.clr_n, clr) and np.array_equal(pkg.wet_n, wet)
    clr_b, wet_b, Vb = _direct_counts(b03, b08, scl, "blocklist")
    assert np.array_equal(pkg.clr_n_blocklist, clr_b)
    assert np.array_equal(pkg.wet_n_blocklist, wet_b)
    assert (clr_b >= clr).all() and (wet_b >= wet).all()
    # frame bits: V of each record scene, packed over the flat frame
    fv = pkg.frame_valid
    assert fv.shape == (pkg.T, H * W)
    for q in range(pkg.T):
        assert np.array_equal(fv[q], V[q].ravel())
    # scene statistics
    st = pkg.scene_stats
    assert list(st.columns) == package.SCENE_STATS_COLUMNS
    assert len(st) == pkg.T and np.array_equal(st["q"], np.arange(pkg.T))
    keep_prod = inp["ex"]["keep"]
    for q in (0, 7, int(np.flatnonzero(pkg.time_index == 10)[0]) if 10 in pkg.time_index
              else 1, pkg.T - 1):
        s8 = obs.scl_u8(scl[q], FILL)
        cnt = np.bincount(s8.ravel(), minlength=256)
        assert [st[f"scl_{c}"][q] for c in range(12)] == cnt[:12].tolist()
        assert st["scl_nodata"][q] == cnt[255]
        cnt_k = np.bincount(s8.ravel()[keep_prod], minlength=256)
        assert [st[f"keep_scl_{c}"][q] for c in range(12)] == cnt_k[:12].tolist()
        cloud, nodata = g21.scene_frame_shares(scl[q], FILL)
        assert st["g21_cloud"][q] == cloud and st["g21_nodata"][q] == nodata
        assert st["ard_good"][q] == wc.frame_good_fraction(
            np.where(scl[q] == FILL, np.nan, scl[q].astype(float)))
        assert st["v_frame"][q] == V[q].mean()
        assert st["v_keep_prod"][q] == V[q].ravel()[keep_prod].mean()
        assert st["v_frame_blocklist"][q] == Vb[q].mean()
        assert st["h0"][q] == pkg.h0[q]
        gp, np_ = b03[q] != FILL, b08[q] != FILL
        assert st["n_both_le0"][q] == np.count_nonzero(gp & np_ & (b03[q] <= 0)
                                                        & (b08[q] <= 0))
        nd_raw, _ = obs.observe(b03[q], b08[q], s8, FILL, "blocklist", "raw")
        assert st["n_raw_abs_ndwi_gt1"][q] == np.count_nonzero(np.abs(nd_raw) > 1)
    # rising from the low-pass marks (g21's bracket: previous mark a low water)
    rising = g21.eq1_bracket(levels.minutes_since_epoch(pkg.t), pkg.marks_lp)["rising"]
    assert np.array_equal(st["rising"].to_numpy().astype(bool), rising)
    assert 0 < rising.sum() < pkg.T


# ─────────────────────────────────────────────────────────────────────────────
#  the package domain
# ─────────────────────────────────────────────────────────────────────────────

def _disk_dilation(mask, radius_px):
    """Brute-force disk dilation on pixel centres, the reference of
    ``package.dilate_disk``: every pixel within ``radius_px`` pixel steps
    (Euclidean) of a True pixel."""
    mask = np.asarray(mask, bool)
    out = mask.copy()
    rr, cc = np.nonzero(mask)
    n = int(np.floor(radius_px))
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            if dr * dr + dc * dc <= radius_px * radius_px:
                r2, c2 = rr + dr, cc + dc
                ok = (r2 >= 0) & (r2 < mask.shape[0]) & (c2 >= 0) & (c2 < mask.shape[1])
                out[r2[ok], c2[ok]] = True
    return out


def _mask(flat, shape=(H, W)):
    m = np.zeros(shape[0] * shape[1], bool)
    m[np.asarray(flat, np.int64)] = True
    return m.reshape(shape)


def _flat(mask):
    return np.flatnonzero(np.asarray(mask).ravel()).astype(np.int64)


def _judge_nbhd(xs, ys, keep_full, lon, lat):
    """R and the neighbourhood of a judge by the package rule (v13 §2.8:
    R = max(500, d_min + 300), +250 m while < 100 keep_full px and R < 2 km)."""
    import pyproj

    tr = pyproj.Transformer.from_crs("EPSG:4326", pyproj.CRS.from_epsg(CRS_EPSG),
                                     always_xy=True)
    gx, gy = tr.transform(lon, lat)
    d_keep = np.hypot(xs[keep_full % W] - gx, ys[keep_full // W] - gy)
    R = max(500.0, d_keep.min() + 300.0)
    while (d_keep <= R).sum() < 100 and R < 2000.0:
        R += 250.0
    rr, cc = np.mgrid[0:H, 0:W]
    return R, np.flatnonzero((np.hypot(xs[cc] - gx, ys[rr] - gy) <= R).ravel())


def test_dom_is_s0_and_the_dilated_urec(site):
    """v2 spec §2.10 (defect 3 of §12.3): S0 = NB ∪ dilate(keep_full, 300 m)
    (D_truth joins S0 with the v2 truth module; Ω lies inside the dilation)
    and dom = S0 ∪ dilate(U_rec, 3 px) ∪ keep_prod, recomputed here from the
    cube with a brute-force disk. The fixture exercises every term: the quay
    judge's neighbourhood holds land beyond both dilations (the mid-flat
    judge's lies inside U_rec, which is why the v13 test could not pass),
    and the dilated U_rec reaches beyond S0."""
    inp, pkg = site.inp, site.pkg
    b03, b08, scl = _record_frames(inp, pkg)
    clr_b, wet_b, _ = _direct_counts(b03, b08, scl, "blocklist")
    U = (clr_b >= 31) & (wet_b >= 4) & (clr_b - wet_b >= 4)
    keep_prod = inp["ex"]["keep"]
    kf = pkg.full.keep
    xs = X0 + PIXEL_M * np.arange(W)
    ys = Y0 - PIXEL_M * np.arange(H)
    with open(inp["stations"], encoding="utf-8") as f:
        stations = {s["code"]: s for s in json.load(f)}
    R, nbhd = {}, {}
    for code in ("st_in", "st_quay"):
        R[code], nbhd[code] = _judge_nbhd(xs, ys, kf, stations[code]["lon"],
                                          stations[code]["lat"])
    NB = np.union1d(nbhd["st_in"], nbhd["st_quay"])
    keep_dil = _disk_dilation(_mask(kf), 300.0 / PIXEL_M)
    S0 = _flat(keep_dil | _mask(NB))
    U3 = _disk_dilation(U, 3.0)
    dom = _flat(U3 | _mask(S0) | _mask(keep_prod))
    assert np.array_equal(pkg.s0, S0) and pkg.s0.dtype == np.int64
    assert np.array_equal(pkg.dom, dom) and pkg.dom.dtype == np.int64
    # NB: the union of the neighbourhoods of the listed judges in the frame
    nb = pkg.nb
    assert np.array_equal(nb["nb"], NB)
    for code in ("st_in", "st_quay"):
        assert np.array_equal(nb["nbhd"][code], nbhd[code])
    assert len(nb["nbhd"]["st_out"]) == 0
    st = {s["code"]: s for s in nb["stations"]}
    assert st["st_in"]["in_frame"] and st["st_quay"]["in_frame"]
    assert not st["st_out"]["in_frame"] and "unlisted" not in st
    assert (st["st_in"]["row"], st["st_in"]["col"]) == (10, 30)
    assert (st["st_quay"]["row"], st["st_quay"]["col"]) == (26, 53)
    for code in ("st_in", "st_quay"):
        assert st[code]["R_m"] == pytest.approx(R[code], abs=1e-6)
        assert st[code]["n_keep_in_nbhd"] >= 100
    assert R["st_in"] == 1250.0 and R["st_quay"] < 2000.0
    # every term of the definition matters on this frame
    others = U3 | keep_dil | _mask(keep_prod)
    beyond = np.setdiff1d(NB, _flat(others))
    assert beyond.size > 0 and set((beyond % W).tolist()) <= set(range(55, W))
    assert np.setdiff1d(nbhd["st_in"], _flat(U)).size == 0
    assert np.setdiff1d(_flat(U3), S0).size > 0
    assert np.setdiff1d(S0, _flat(U)).size > 0
    # keep_full lies in U_rec; keep_prod and S0 lie in dom
    assert U.ravel()[kf].all()
    assert np.isin(keep_prod, pkg.dom).all() and np.isin(S0, pkg.dom).all()
    # the sizes are recorded with the domain
    meta = json.loads(str(product.load_npz(
        os.path.join(site.cfg.common_dir, package.NB_NPZ))["meta"]))
    assert (meta["n_s0"], meta["n_dom"], meta["n_nb"], meta["n_urec"]) ==         (len(S0), len(dom), len(NB), int(U.sum()))
    assert meta["s0_keep_dilation_m"] == 300.0 and meta["dom_urec_dilation_px"] == 3
    assert meta["n_keep_full"] == len(kf) and meta["n_keep_prod"] == len(keep_prod)
    # the package neighbourhood rule is the judges'
    assert {k: v for k, v in package.NB_RULE.items()} ==         dict(config.FROZEN["judges"]["neighbourhood"])


def test_dilate_disk_equals_a_brute_force_disk():
    rng = np.random.default_rng(5)
    m = None
    for shape, p in (((23, 31), 0.02), ((40, 17), 0.2), ((9, 9), 0.0), ((1, 12), 0.3)):
        m = rng.random(shape) < p
        for r in (0, 1, 1.5, 2, 2.2, 3, 7.5, 30.0):
            got = package.dilate_disk(m, r)
            assert got.dtype == bool and got.shape == m.shape
            assert np.array_equal(got, _disk_dilation(m, r)), (shape, p, r)
    assert package.dilate_disk(np.ones((5, 6), bool), 2).all()
    with pytest.raises(ValueError):
        package.dilate_disk(m, -1)


def test_support_and_domain_terms_at_10_m():
    """At 10 m the 300 m dilation of keep_full (30 px) reaches far beyond the
    3-px dilation of U_rec, D_truth joins S0, keep_prod joins dom."""
    shape = (90, 120)
    keep = np.zeros(shape, bool)
    keep[40:50, 50:70] = True
    U = keep.copy()
    U[38:52, 48:72] = True                       # U_rec: a superset of keep_full
    nb = np.array([5 * 120 + 5, 5 * 120 + 6], np.int64)
    d_truth = np.array([88 * 120 + 118, 2 * 120 + 110], np.int64)
    keep_prod = np.array([0, 89 * 120], np.int64)
    s0 = package.support_s0(shape, _flat(keep), nb, 10.0, d_truth=d_truth)
    want_s0 = _disk_dilation(keep, 30.0) | _mask(nb, shape) | _mask(d_truth, shape)
    assert np.array_equal(s0, _flat(want_s0)) and s0.dtype == np.int64
    assert np.array_equal(package.support_s0(shape, _flat(keep), nb, 10.0),
                          _flat(_disk_dilation(keep, 30.0) | _mask(nb, shape)))
    dom = package.package_domain(shape, s0, U, keep_prod)
    want = _disk_dilation(U, 3.0) | want_s0 | _mask(keep_prod, shape)
    assert np.array_equal(dom, _flat(want)) and dom.dtype == np.int64
    # each term adds pixels of its own
    assert np.setdiff1d(_flat(_disk_dilation(keep, 30.0)), _flat(_disk_dilation(U, 3.0))).size
    assert np.isin(d_truth, s0).all() and np.isin(nb, s0).all()
    assert np.isin(keep_prod, dom).all() and not np.isin(keep_prod, s0).any()


def _subset_rule(clr, wet):
    with np.errstate(invalid="ignore", divide="ignore"):
        wf = np.where(clr > 30, wet / np.maximum(clr, 1), np.nan)
    inter = np.isfinite(wf) & (wf > 0.10) & (wf < 0.90)
    sea = np.isfinite(wf) & (wf >= 0.90)
    return inter, sea


def test_urec_contains_every_subset_keep_property(site):
    """U_rec (blocklist + floor0, full record) contains the A-mode keep of
    any subset under either mask rule; outside dom, the A-mode sea is
    exactly "sea-like and more than 30 clear observations in the subset"."""
    inp, pkg = site.inp, site.pkg
    b03, b08, scl = _record_frames(inp, pkg)
    s8 = obs.scl_u8(scl, FILL)
    nd, Vs = obs.observe_rules(b03, b08, s8, FILL, mode="floor0")
    W_ = {r: obs.wet(nd, v) for r, v in Vs.items()}
    U = package.urec_mask(pkg.clr_n_blocklist, pkg.wet_n_blocklist)
    in_dom = np.zeros(H * W, bool)
    in_dom[pkg.dom] = True
    in_dom = in_dom.reshape(H, W)
    clr_n, wet_n = pkg.clr_n.astype(int), pkg.wet_n.astype(int)
    sea_like = (clr_n >= 31) & (clr_n - wet_n <= 3) & ~in_dom
    rng = np.random.default_rng(2026)
    n_keep_seen = n_sea_out = 0
    for _ in range(200):
        N = int(rng.integers(31, pkg.T + 1))
        sel = np.sort(rng.choice(pkg.T, N, replace=False))
        for rule in ("marea_clear", "blocklist"):
            clr = Vs[rule][sel].sum(axis=0)
            wet = W_[rule][sel].sum(axis=0)
            inter, sea = _subset_rule(clr, wet)
            assert not (inter & ~U).any(), (N, rule)
            n_keep_seen += int(inter.sum())
            if rule == "marea_clear":
                out = ~in_dom
                assert np.array_equal(sea[out], (sea_like & (clr >= 31))[out])
                n_sea_out += int((sea & out).sum())
    assert n_keep_seen > 0 and n_sea_out > 0         # the property was exercised


# ─────────────────────────────────────────────────────────────────────────────
#  pass 2 and the production extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_pass2_stores_the_raw_cube_values_at_dom(site):
    inp, pkg = site.inp, site.pkg
    b03, b08, scl = _record_frames(inp, pkg)
    flat = lambda a: a.reshape(len(a), -1)[:, pkg.dom]  # noqa: E731
    assert pkg.B03.dtype == np.int16 and pkg.SCL.dtype == np.uint8
    assert np.array_equal(np.asarray(pkg.B03), flat(b03))
    assert np.array_equal(np.asarray(pkg.B08), flat(b08))
    assert np.array_equal(np.asarray(pkg.SCL), obs.scl_u8(flat(scl), FILL))
    assert (np.asarray(pkg.SCL) == 255).any()                  # the fill became 255


def test_package_reproduces_marea_extract_at_keep_prod(site):
    """G-keep: raw-mode NDWI and the marea_clear mask from the stored arrays
    equal marea.extract's Y and C at its keep, bitwise (NaN payloads too)."""
    inp, pkg = site.inp, site.pkg
    ex = inp["ex"]
    assert np.array_equal(pkg.keep_prod, ex["keep"])
    assert np.array_equal(pkg.sea_prod, ex["sea"]) and np.array_equal(pkg.inter_prod, ex["inter"])
    pos = pkg.positions(ex["keep"])
    nd, V = pkg.observe(np.arange(pkg.T), pos, rule="marea_clear", mode="raw")
    Y, C = ex["Y"][pkg.time_index], ex["C"][pkg.time_index]
    assert np.array_equal(_bits32(nd), _bits32(Y))
    assert np.isnan(Y).any()                                   # NaN cases exercised
    m = obs.mask_valid(np.asarray(pkg.SCL)[:, pos], "marea_clear")
    assert np.array_equal(m, C)
    assert np.array_equal(V, C & np.isfinite(Y))
    out = package.check_extraction(pkg, tmpdir=str(site.base), block=37)
    assert out["n_keep_prod"] == len(ex["keep"]) and out["s_km"]["n"] == len(ex["keep"])
    # the production s_km recomputed by the domain module equals cell 19's
    prod = domain.production(pkg)
    assert np.array_equal(prod.s_km.astype(np.float32), inp["s_km_ex"].astype(np.float32),
                          equal_nan=True)


def test_check_extraction_detects_one_differing_value(site):
    pkg = site.pkg
    fake = types.SimpleNamespace(**{k: getattr(pkg, k) for k in
                                    ("cfg", "T", "time_index", "keep_prod", "SCL", "dom")})
    B03 = np.array(pkg.B03)
    p = int(pkg.positions(pkg.keep_prod[:1])[0])
    q = int(np.flatnonzero(B03[:, p] != FILL)[0])
    B03[q, p] += 1
    fake.positions = pkg.positions
    fake.observe = lambda sel, pos, rule, mode: obs.observe(
        B03[sel][:, pos], np.asarray(pkg.B08)[sel][:, pos], np.asarray(pkg.SCL)[sel][:, pos],
        FILL, rule, mode)
    with pytest.raises(AssertionError, match="raw NDWI differs"):
        package.check_extraction(fake, tmpdir=str(site.base))


# ─────────────────────────────────────────────────────────────────────────────
#  levels, series, marks
# ─────────────────────────────────────────────────────────────────────────────

def test_level_table_in_the_package(site):
    pkg, prov = site.pkg, site.provider
    lev = pkg.levels
    assert lev.batch_invariant
    assert lev.tau_lo == -90 and lev.tau_hi == config.tau_tab_max(pkg.full.K)
    assert np.array_equal(_bits(lev.h0), _bits(pkg.h0))
    for k, tau in enumerate(lev.shifts):                   # EVERY row, bitwise
        want = FakeTide.value((pkg.t - pd.Timedelta(minutes=int(tau))).asi8)
        assert np.array_equal(_bits(lev.H[k]), _bits(want)), tau
    # beyond the table: one live per-shift call through the package's provider
    n0 = len(prov.calls)
    sel = np.arange(0, pkg.T, 3)
    tau = lev.tau_hi + 37
    h = lev.at(tau, sel)
    assert len(prov.calls) == n0 + 1
    assert np.array_equal(prov.calls[-1], (pkg.t[sel] - pd.Timedelta(minutes=tau)).asi8)
    assert np.array_equal(_bits(h), _bits(FakeTide.value(prov.calls[-1])))
    # the 1-min and 10-min series and both marks load without a provider
    t_min, hh = pkg.series1
    assert len(t_min) == len(levels.series_1min_index())
    i10, h10 = pkg.series10
    assert len(i10) == 1096 * 144
    assert pkg.marks_lp["lowpass_width"] == 61 and pkg.marks_raw["n_marks"] > 4000


def test_batch_variant_provider_falls_back(tmp_path, monkeypatch):
    small_thresholds(monkeypatch)
    inp = make_inputs(str(tmp_path / "inputs"), seed=8)
    prov = FakeTide(batch_dependent=True)
    cfg = site_cfg(inp, tmp_path / "out")
    package.prepare(SITE, RES, steps=("record", "series", "pass1", "geometry", "levels"),
                    cfg=cfg, provider=prov)
    tab = levels.LevelTable.load(os.path.join(cfg.common_dir, package.LEVELS_NPZ))
    assert not tab.batch_invariant
    assert tab.meta["mismatch"]
    rec = product.read_json(os.path.join(cfg.common_dir, package.RECORD_JSON))
    assert np.array_equal(_bits(tab.h0), _bits(np.asarray(rec["h0"])))


# ─────────────────────────────────────────────────────────────────────────────
#  subsets
# ─────────────────────────────────────────────────────────────────────────────

def test_subsets(site):
    from pyintertidal import granadeiro2021 as g21

    pkg, inp = site.pkg, site.inp
    T, sid = pkg.T, config.SITE_ID[SITE]
    z = pkg.subsets_stored
    assert int(z["T"]) == T and z["perm"].shape == (10, T)
    for r in (0, 3, 9, 12):                   # rep 12 is drawn on the fly
        perm = np.random.default_rng([20261008, sid, r]).permutation(T)
        prev = np.zeros(0, np.int64)
        for N in (10, 20, 45, T):
            sel = pkg.subset("rand", N, r)
            assert np.array_equal(sel, np.sort(perm[:N]))
            assert np.isin(prev, sel).all()       # nested in N within a rep
            prev = sel
        start = np.random.default_rng([20261012, sid, 45, r]).integers(0, T - 45 + 1)
        assert np.array_equal(pkg.subset("win", 45, r), np.arange(start, start + 45))
    assert np.array_equal(pkg.subset("rand", "all", 0), np.arange(T))
    assert np.array_equal(pkg.subset("win", "all", 4), np.arange(T))
    with pytest.raises(ValueError):
        pkg.subset("rand", T + 1, 0)
    with pytest.raises(ValueError):
        pkg.subset("rule-tz", 20, 0)
    # scene rules
    tz = pkg.subset("rule-tz", "all")
    assert np.array_equal(tz, np.flatnonzero(np.isin(pkg.dates, inp["tz_dates"])))
    meta = json.loads(str(z["meta"]))
    assert meta["n_tz_dates_not_in_record"] == 1
    st = pkg.scene_stats
    g = pkg.subset("rule-g21", "all")
    assert np.array_equal(g, np.flatnonzero((st["g21_cloud"] < 0.10) & (st["g21_nodata"] <= 0.02)))
    assert np.array_equal(g, np.flatnonzero(g21.scene_rule(
        st["g21_cloud"], st["g21_nodata"], np.ones(T, bool), np.ones(T, bool))))
    a = pkg.subset("rule-ard", "all")
    assert np.array_equal(a, np.flatnonzero(st["ard_good"] >= 0.5))
    cloudy = int(np.flatnonzero(pkg.time_index == 10)[0])
    assert cloudy not in g and cloudy not in a
    # subset keys
    sel = pkg.subset("rand", 20, 1)
    assert pkg.subset_key(sel) == product.subset_key(sel)
    # the same draws for the same site and T whatever the grid
    assert np.array_equal(package.subset_positions(SITE, T, "rand", 20, 1), sel)


# ─────────────────────────────────────────────────────────────────────────────
#  manifest, keys, verification
# ─────────────────────────────────────────────────────────────────────────────

def _copy_package(site, dst):
    shutil.copytree(site.cfg.root, dst)
    return dataclasses.replace(site.cfg, root=str(dst).replace("\\", "/"))


def test_manifest_lists_every_keyed_file_and_the_inputs(site):
    man = product.read_json(os.path.join(site.cfg.common_dir, package.MANIFEST_JSON))
    assert set(man["files"]) == set(package.KEYED_FILES)
    assert {"cube", "extract", "overpass", "judges_coords", "transition_result"} <= \
        set(man["inputs"])
    assert man["obs_sha"] == obs.source_sha256()
    assert man["frozen"] == config.frozen_shas(config.PACKAGE_SECTIONS)
    assert man["package_key"] == site.key == product.package_key(man)
    assert site.pkg.keys["package_key"] == site.key
    assert package.status(SITE, RES, cfg=site.cfg)["manifest"] == "ok"
    # keyed files have LF line ends and no times
    rec = open(os.path.join(site.cfg.common_dir, package.RECORD_JSON), "rb").read()
    assert b"\r\n" not in rec
    assert b"\r\n" not in open(os.path.join(site.cfg.common_dir,
                                            package.SCENE_STATS_CSV), "rb").read()


def test_data_byte_changes_the_key_scoring_edit_does_not(site, tmp_path, monkeypatch):
    cfg = _copy_package(site, tmp_path / "copy")
    d = cfg.common_dir
    key0 = package.write_manifest(d)
    assert key0 == site.key                         # same contents, other folder
    # a scoring-section edit: same key
    frozen = dict(config.FROZEN)
    frozen["scoring"] = types.MappingProxyType({"rule": "edited after the package"})
    monkeypatch.setattr(config, "FROZEN", types.MappingProxyType(frozen))
    assert package.write_manifest(d) == key0
    assert package.verify_manifest(d) == key0
    # a package-section edit: the package is stale, and its key would change
    frozen["subsets"] = types.MappingProxyType({"rule": "edited"})
    monkeypatch.setattr(config, "FROZEN", types.MappingProxyType(frozen))
    with pytest.raises(package.StalePackageError):
        package.verify_manifest(d)
    assert package.write_manifest(d) != key0
    monkeypatch.undo()
    small_thresholds(monkeypatch)
    # one data byte: a different key
    p = os.path.join(d, package.B03_NPY)
    raw = bytearray(open(p, "rb").read())
    raw[-1] ^= 1
    open(p, "wb").write(bytes(raw))
    assert package.write_manifest(d, rehash=True) != key0


def test_load_refuses_a_tampered_file(site, tmp_path):
    cfg = _copy_package(site, tmp_path / "copy")
    d = cfg.common_dir
    package.verify_manifest(d, full=True)           # a copy: contents equal, times restamped
    pkg = package.load(SITE, RES, cfg=cfg)
    pkg.close()
    del pkg
    p = os.path.join(d, package.SCL_NPY)
    st = os.stat(p)
    raw = bytearray(open(p, "rb").read())
    raw[-3] ^= 0x10
    open(p, "wb").write(bytes(raw))
    with pytest.raises(package.PackageError, match="scl.npy"):
        package.load(SITE, RES, cfg=cfg)
    # the cheap check sees size and mtime only: a restored mtime passes it,
    # the full check (prepare, gate keep, score) does not
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert package.verify_manifest(d, full=False) == site.key
    with pytest.raises(package.PackageError, match="content differs"):
        package.verify_manifest(d, full=True)
    # a size change is always caught
    open(p, "ab").write(b"\0")
    with pytest.raises(package.PackageError, match="size"):
        package.verify_manifest(d, full=False)
    # a manifest edited by hand no longer gives its own key
    man = product.read_json(os.path.join(d, package.MANIFEST_JSON))
    man["files"][package.DOM_NPY]["sha256"] = "0" * 64
    package.write_json_lf(os.path.join(d, package.MANIFEST_JSON), man)
    with pytest.raises(package.PackageError, match="edited by hand"):
        package.verify_manifest(d)


def test_resume_is_byte_identical_and_runs_nothing_twice(site, tmp_path, monkeypatch):
    small_thresholds(monkeypatch)
    inp = site.inp
    prov = FakeTide(nan_at=[inp["nan_instant"]])
    cfg = site_cfg(inp, tmp_path / "two_parts")
    assert package.prepare(SITE, RES, steps=("record", "series", "pass1"), cfg=cfg,
                           provider=prov) is None
    st = package.step_status(cfg.common_dir)
    assert [st[s] for s in package.STEPS] == ["done"] * 3 + ["pending"] * 6
    with pytest.raises(package.PackageError, match="needs"):
        package.prepare(SITE, RES, steps=("levels",), cfg=cfg, provider=prov)
    key = package.prepare(SITE, RES, cfg=cfg, provider=prov)
    assert key == site.key                           # the one-shot build's key
    for name in package.KEYED_FILES:                 # byte for byte
        a = open(os.path.join(cfg.common_dir, name), "rb").read()
        b = open(os.path.join(site.cfg.common_dir, name), "rb").read()
        assert a == b or name.endswith(".npz") and \
            product.sha256_file(os.path.join(cfg.common_dir, name)) == \
            product.sha256_file(os.path.join(site.cfg.common_dir, name)), name
    # a finished package reruns nothing (no provider call at all)
    n0 = len(prov.calls)
    assert package.prepare(SITE, RES, cfg=cfg, provider=prov) == key
    assert len(prov.calls) == n0
    # a forced step makes the steps that read its files stale ...
    package.prepare(SITE, RES, steps=("pass1",), force=True, cfg=cfg, provider=prov)
    st = package.step_status(cfg.common_dir)
    assert st["pass1"] == "done" and st["record"] == "done"
    assert not os.path.exists(os.path.join(cfg.common_dir, package.MANIFEST_JSON))
    with pytest.raises(package.PackageError, match="no manifest"):
        package.load(SITE, RES, cfg=cfg)
    # ... but identical outputs keep them done, and the key is unchanged
    assert st["geometry"] == "done" and st["pass2"] == "done"
    assert package.prepare(SITE, RES, cfg=cfg, provider=prov) == key
    # a changed output (here a new frame bit) makes its readers stale
    p = os.path.join(cfg.common_dir, package.FRAME_VALID_NPY)
    fv = np.load(p)
    fv[0, 0] ^= 0x80
    with open(p, "wb") as f:
        np.lib.format.write_array(f, fv)
    st = package.step_status(cfg.common_dir)
    assert st["pass1"] == "stale" and st["pass2"] == "stale" and st["record"] == "done"


def test_prepare_refuses_a_locked_package(site, tmp_path):
    cfg = _copy_package(site, tmp_path / "copy")
    lock = os.path.join(cfg.common_dir, package.LOCK_FILE)
    with open(lock, "w") as f:
        json.dump({"host": "another-host", "pid": 1}, f)
    with pytest.raises(package.PackageError, match="another prepare"):
        package.prepare(SITE, RES, cfg=cfg, provider=FakeTide())


# ─────────────────────────────────────────────────────────────────────────────
#  reading the package
# ─────────────────────────────────────────────────────────────────────────────

def test_observe_blocks_count_and_positions(site):
    inp, pkg = site.inp, site.pkg
    b03, b08, scl = _record_frames(inp, pkg)
    sel = pkg.subset("rand", 45, 2)
    pos = pkg.positions(pkg.full.keep)
    nd, V = pkg.observe(sel, pos)
    flat = lambda a: a[sel].reshape(len(sel), -1)[:, pkg.full.keep]  # noqa: E731
    nd0, V0 = obs.observe(flat(b03), flat(b08), obs.scl_u8(flat(scl), FILL), FILL)
    assert np.array_equal(_bits32(nd), _bits32(nd0)) and np.array_equal(V, V0)
    got = list(pkg.blocks(sel, pos, size=17))
    assert [(a, b) for a, b, _, _ in got] == [(a, min(a + 17, len(pos)))
                                              for a in range(0, len(pos), 17)]
    assert np.array_equal(np.hstack([x[3] for x in got]), V)
    assert np.array_equal(_bits32(np.hstack([x[2] for x in got])), _bits32(nd))
    clr, wet = pkg.count(sel, pos, budget_mb=0.001)          # many row chunks
    assert np.array_equal(clr, V.sum(0)) and np.array_equal(wet, obs.wet(nd, V).sum(0))
    for rule in obs.MASK_RULES:                              # a run-time option
        for mode in obs.INDEX_MODES:
            n1, v1 = pkg.observe(sel, pos[::7], rule, mode)
            n2, v2 = obs.observe(flat(b03)[:, ::7], flat(b08)[:, ::7],
                                 obs.scl_u8(flat(scl)[:, ::7], FILL), FILL, rule, mode)
            assert np.array_equal(_bits32(n1), _bits32(n2)) and np.array_equal(v1, v2)
    outside = np.setdiff1d(np.arange(H * W), pkg.dom)[:3]
    with pytest.raises(KeyError):
        pkg.positions(outside)
    # the stored full domain is the recomputed one, and the grid is the cube's
    assert pkg.full.equals(domain.full(pkg))
    from experiments import v7_waterline_contour as v7
    g = v7.cube_grid(inp["cube"])
    assert np.array_equal(pkg.grid["xs"], g["xs"]) and np.array_equal(pkg.grid["ys"], g["ys"])
    assert tuple(pkg.grid["transform"]) == tuple(g["transform"])
    assert pkg.band_frame.dtype == np.int16 and pkg.s_frame.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
#  the defects of v2 spec §12.3 (milestone V0) stay fixed
# ─────────────────────────────────────────────────────────────────────────────

class _FlushFailed(Exception):
    """An injected failure (a distinct type, so that a PermissionError raised
    while cleaning up cannot pass for it)."""


def _tmp_leftovers(d):
    return sorted(f for f in os.listdir(d) if ".tmp-" in f)


def test_pass2_drops_every_memory_map(site, tmp_path, monkeypatch):
    """Defect 1: pass 2 kept its last map open through the flush loop's
    variable, so os.replace of a mapped file failed on Windows (WinError
    32). Both exits must drop every map first: after a run the published
    files can be moved at once, and after a failure in the flush loop the
    original error propagates and no temporary file is left behind."""
    pkg, cfg = site.pkg, site.cfg
    out = tmp_path / "p2"
    out.mkdir()
    shutil.copy(os.path.join(cfg.common_dir, package.FRAME_VALID_NPY), out)
    info = package.pass2(cfg, pkg.record, pkg.dom, str(out))
    assert (info["T"], info["P_dom"]) == (pkg.T, pkg.P)
    names = (package.B03_NPY, package.B08_NPY, package.SCL_NPY)
    for name, ref in zip(names, (pkg.B03, pkg.B08, pkg.SCL)):
        p = os.path.join(out, name)
        assert np.array_equal(np.load(p), np.asarray(ref))
        os.replace(p, p + ".moved")             # a mapped file cannot be moved on Windows
        os.remove(p + ".moved")
    assert _tmp_leftovers(out) == []
    real_flush = np.memmap.flush
    calls = []

    def flaky_flush(self):
        calls.append(1)
        if len(calls) == 2:
            raise _FlushFailed("disk full while flushing b08")
        return real_flush(self)

    with monkeypatch.context() as m:
        m.setattr(np.memmap, "flush", flaky_flush)
        with pytest.raises(_FlushFailed):
            package.pass2(cfg, pkg.record, pkg.dom, str(out))
    assert len(calls) == 2
    assert _tmp_leftovers(out) == []
    assert not any(os.path.exists(os.path.join(out, n)) for n in names)


def test_pass1_drops_its_memory_map(site, tmp_path, monkeypatch):
    """The frame validity bits of pass 1 are mapped too: a run publishes a
    movable file; a failure in the scene loop, or inside the flush (whose
    traceback frame holds the map), leaves no temporary file and the
    original error propagates."""
    pkg, cfg = site.pkg, site.cfg
    out = tmp_path / "p1"
    out.mkdir()
    package.pass1(cfg, pkg.record, str(out))
    p = os.path.join(out, package.FRAME_VALID_NPY)
    ref = np.load(os.path.join(cfg.common_dir, package.FRAME_VALID_NPY))
    assert np.array_equal(np.load(p), ref)
    os.replace(p, p + ".moved")
    os.remove(p + ".moved")
    real = package.read_scene
    calls = []

    def flaky(ds, t_dim, t):
        calls.append(t)
        if len(calls) == 4:
            raise _FlushFailed("cube unreadable")
        return real(ds, t_dim, t)

    with monkeypatch.context() as m:
        m.setattr(package, "read_scene", flaky)
        with pytest.raises(_FlushFailed):
            package.pass1(cfg, pkg.record, str(out))
    assert _tmp_leftovers(out) == [] and not os.path.exists(p)

    def failing_flush(self):
        raise _FlushFailed("disk full while flushing the frame bits")

    with monkeypatch.context() as m:
        m.setattr(np.memmap, "flush", failing_flush)
        with pytest.raises(_FlushFailed):
            package.pass1(cfg, pkg.record, str(out))
    assert _tmp_leftovers(out) == [] and not os.path.exists(p)


def test_scene_stats_floats_round_trip_bit_for_bit(tmp_path):
    """Defect 2: scene_stats.csv was read back with pandas' default float
    parser, which lands one ulp away from some of the shortest reprs that
    to_csv writes (73/720 = 0.10138888888888889 reads 0.1013888888888888;
    the double just above 0.02 reads 0.02), and the g21 rule compares these
    shares with 0.10 and 0.02. read_scene_stats parses them exactly."""
    above_002 = float(np.nextafter(0.02, 1.0))
    below_010 = float(np.nextafter(0.10, 0.0))
    shares = [  # (g21_cloud, g21_nodata, ard_good)
        (73 / 720, 0.0, 0.9),                    # cloud > 0.10: fails g21
        (0.05, above_002, 0.9),                  # no data just above 0.02: fails g21
        (below_010, 0.02, 0.9),                  # both just inside: passes g21
        (1 / 30, 1 / 70, float(np.nextafter(0.5, 0.0))),   # ard just below 0.5: fails ard
        (0.0, 0.0, 0.5),                         # ard exactly 0.5: passes ard
    ]
    rows = []
    for q, (c, nd, ag) in enumerate(shares):
        row = {k: 0 for k in package.SCENE_STATS_COLUMNS}
        row.update(q=q, t_index=q + 3, date=f"2023-01-{q + 1:02d}",
                   utc=f"2023-01-{q + 1:02d}T10:55:00", h0=0.1 * q + 1 / 3, rising=q % 2,
                   g21_cloud=c, g21_nodata=nd, ard_good=ag, v_frame=73 / 720,
                   v_keep_prod=float("nan") if q == 4 else 2 / 3, v_frame_blocklist=1 / 9)
        rows.append(row)
    p = os.path.join(tmp_path, package.SCENE_STATS_CSV)
    package._write_csv_lf(p, rows, package.SCENE_STATS_COLUMNS)
    st = package.read_scene_stats(p)
    assert list(st.columns) == package.SCENE_STATS_COLUMNS
    for col in ("h0", "g21_cloud", "g21_nodata", "ard_good", "v_frame", "v_keep_prod",
                "v_frame_blocklist"):
        want = np.array([r[col] for r in rows], np.float64)
        assert np.array_equal(_bits(st[col].to_numpy(np.float64)), _bits(want)), col
    rules = package.rule_subsets([r["date"] for r in rows], st)
    assert rules["g21"].tolist() == [2, 3, 4]
    assert rules["ard"].tolist() == [0, 1, 2, 4]
    assert rules["tz"] is None


def test_load_checks_the_files_on_every_call(site, tmp_path):
    """Defect 4: load() memoised its verification per manifest mtime, so a
    file changed after a first load in the same process was never checked
    again. Every call now re-stats every keyed file (size and mtime_ns)."""
    cfg = _copy_package(site, tmp_path / "copy")
    d = cfg.common_dir
    package.verify_manifest(d, full=True)          # the copy's times recorded once
    first = package.load(SITE, RES, cfg=cfg)       # loaded, and still open
    man = os.path.join(d, package.MANIFEST_JSON)
    man_mtime = os.stat(man).st_mtime_ns
    p = os.path.join(d, package.SCENE_STATS_CSV)   # a keyed file that is not mapped
    st = os.stat(p)
    original = open(p, "rb").read()
    j = original.index(b"\n") + 1                  # the first data row starts with q
    tampered = bytearray(original)
    tampered[j] = ord("7")
    with open(p, "wb") as f:
        f.write(bytes(tampered))
    with pytest.raises(package.PackageError, match="scene_stats.csv"):
        package.load(SITE, RES, cfg=cfg)
    assert os.stat(man).st_mtime_ns == man_mtime   # the manifest never changed
    # the original bytes and time back: accepted again
    with open(p, "wb") as f:
        f.write(original)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    package.load(SITE, RES, cfg=cfg).close()
    # a new time alone fails the cheap check; one full check records it
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    with pytest.raises(package.PackageError, match="mtime"):
        package.load(SITE, RES, cfg=cfg)
    assert package.verify_manifest(d, full=True) == site.key
    package.load(SITE, RES, cfg=cfg).close()
    first.close()


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION V2: year shards (spec §2.10, §13.1-13.4)
# ═════════════════════════════════════════════════════════════════════════════

from tests import benchmark_zarr_site as zs  # noqa: E402
from experiments.benchmark import settings  # noqa: E402


@pytest.fixture(scope="module")
def zsite(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    s = zs.make_site(tmp_path_factory.mktemp("v2_package"), mp)
    s.frames = zs.record_frames(s.inp, s.pkg)
    yield s
    s.pkg.close()
    mp.undo()


def _raw(frames, sel):
    return {b: a[sel] for b, a in frames.items()}


def _fixed(index):
    t = settings.INDEX[index].t_k
    return {"t_k": float(t), "s_k": 1.0}


def _c3(frames, sel, p):
    return obs.cloud_c3(frames["SCL"][sel], frames["CLD"][sel], p)


def _region_slices(region):
    r0, r1, c0, c1 = (int(v) for v in region)
    return slice(r0, r1), slice(c0, c1)


def _c11_valid(zsite, sel):
    """C11 on the whole frames: the fake OmniCloudMask inside the OCM region,
    C1 outside (declared)."""
    f, pkg = zsite.frames, zsite.pkg
    rs, cs = _region_slices(pkg.support["ocm_region"])
    out = obs.cloud_c1(f["SCL"][sel]).copy()
    for k, q in enumerate(sel):
        raw = {b: f[b][q][rs, cs] for b in ("B04", "B03", "B8A")}
        cls = zs.fake_ocm_classes(raw, int(pkg.offsets[q]))
        out[k][rs, cs] = obs.cloud_c11(f["SCL"][q][rs, cs], cls)
    return out


def test_v2_record_rule(zsite):
    pkg, inp = zsite.pkg, zsite.inp
    every = [str(d.date()) for y in zs.YEARS for d in zs.dates(y)]
    nan_date = str(inp["nan_instant"].date())
    want = [d for d in every if d not in (inp["no_instant"], nan_date)]
    assert list(pkg.dates) == want and pkg.T == 64
    assert pkg.record["nonfinite_level"] == [nan_date]
    assert pkg.record["year_slices"] == {"2023": [0, 21], "2024": [21, 42], "2025": [42, 64]}
    assert pkg.lattice == zs.PHASE
    assert set(pkg.offsets[:42].tolist()) == {0} and set(pkg.offsets[42:].tolist()) == {1000}
    assert np.array_equal(_bits(pkg.h0), _bits(FakeTide.value(pkg.t.asi8)))
    dt = pd.Timedelta(minutes=package.RISING_HALF_MIN)
    assert np.array_equal(pkg.rising, FakeTide.value((pkg.t + dt).asi8)
                          > FakeTide.value((pkg.t - dt).asi8))
    assert pkg.record["offset_source"] == "given"
    assert set(pkg.record["fingerprints"]) == {"2023", "2024", "2025"}


def test_v2_uint8_counters_refuse_a_year_over_the_limit(zsite, monkeypatch):
    monkeypatch.setattr(package, "YEAR_MAX_SCENES", 20)
    with pytest.raises(package.PackageError, match="uint8"):
        package.v2_record(zsite.site2, zsite.provider)


def test_v2_pass1_counters_bits_and_stats_equal_the_frames(zsite):
    from pyintertidal import granadeiro2021 as g21
    from pyintertidal import waterline_contour as wc
    pkg, f, s2 = zsite.pkg, zsite.frames, zsite.site2
    plan = package.pass1_plan(zs.KEYS)
    assert {n for n, _, _ in plan} >= {"C1.floor0.ndwi.D1", "C3.floor0.ndwi.D1~p20",
                                       "C3.floor0.ndwi.D1~p35", "C1.floor0.ndwi.D5"}
    assert not any(package.threshold_key(n.split("~")[0]) for n, _, _ in plan)
    for y in zs.YEARS:
        sel = pkg.window_years_safe(y)
        raw = _raw(f, sel)
        for name, s, p in plan:
            cv = _c3(f, sel, p) if p is not None else None
            o = obs.observe_setting(raw, s, offset=pkg.offsets[sel], lattice=zs.PHASE,
                                    params=_fixed(s.index), cloud_valid=cv)
            clr, wet = package.year_counters(s2, y, s.id, p)
            assert clr.dtype == np.uint8 and wet.dtype == np.uint8
            assert np.array_equal(clr, o.V.sum(axis=0)), (y, name)
            assert np.array_equal(wet, o.W.sum(axis=0)), (y, name)
        d = package._year_dir(s2, y, "pass1")
        o = obs.observe_setting(raw, settings.P0, offset=pkg.offsets[sel], lattice=zs.PHASE)
        bits = np.load(os.path.join(d, "bits_P0.npy"))
        assert np.array_equal(package._unpack_rows(bits, zs.W), o.V)
        bad = package._unpack_rows(np.load(os.path.join(d, "bits_bad.npy")), zs.W)
        assert np.array_equal(bad, np.isin(raw["SCL"], (3, 8, 9, 10)))
        for p in (20, 35):
            b3 = package._unpack_rows(np.load(os.path.join(d, f"bits_C3p{p}.npy")), zs.W)
            o3 = obs.observe_setting(raw, settings.P0.replace(cloud="C3"),
                                     offset=pkg.offsets[sel], lattice=zs.PHASE,
                                     cloud_valid=_c3(f, sel, p))
            assert np.array_equal(b3, o3.V)
        st = package.read_scene_stats_v2(s2, y)
        xt = np.load(os.path.join(d, "crosstab.npy"))
        assert (xt.sum(axis=(1, 2)) == zs.H * zs.W).all()
        for j, q in enumerate(sel):
            s8 = raw["SCL"][j]
            cnt = obs.scl_counts(s8)
            assert [st[f"scl_{k}"][j] for k in range(12)] + [st["scl_nodata"][j]] == \
                cnt.tolist()
            cl, nd = g21.scene_frame_shares(s8, fill=obs.SCL_NODATA_U8)
            assert st["g21_cloud"][j] == cl and st["g21_nodata"][j] == nd
            assert st["ard_good"][j] == wc.frame_good_fraction(s8)
            assert st["cld_fill"][j] == int(np.count_nonzero(raw["CLD"][j] == 255))
            assert bool(obs.scene_c6(s8)) == bool(g21.scene_rule(cl, nd, True, True))
    assert (np.asarray(zsite.ran["pass1:2025"]["T"]) == 22)


def test_v2_geometry_support_and_dom_by_brute_force(zsite):
    pkg, f, s2 = zsite.pkg, zsite.frames, zsite.site2
    sel = pkg.window("W3")
    o = obs.observe_setting(_raw(f, sel), settings.P0, offset=pkg.offsets[sel],
                            lattice=zs.PHASE)
    clr, wet = o.V.sum(axis=0), o.W.sum(axis=0)
    d = domain.from_counts(clr, wet, 10.0, "west")
    assert np.array_equal(pkg.full.keep, d.keep) and np.array_equal(pkg.full.sea, d.sea)
    assert pkg.full.mode == "F" and np.array_equal(pkg.full.pos, pkg.positions(d.keep))
    keep_m = np.zeros(zs.H * zs.W, bool)
    keep_m[d.keep] = True
    keep_m = keep_m.reshape(zs.H, zs.W)
    with np.errstate(invalid="ignore", divide="ignore"):
        wf = np.where(clr >= 31, wet / np.maximum(clr, 1), np.nan)
    ring = _disk_dilation(keep_m, 3.0) & ~keep_m
    omega = keep_m | (ring & ((wf >= 0.98) | (wf <= 0.02)))
    assert np.array_equal(pkg.omega, np.flatnonzero(omega))
    sup = pkg.support
    assert bool(sup["has_d_truth"]) and np.array_equal(sup["d_truth"], np.unique(d.keep[::3]))
    s0 = _disk_dilation(keep_m, 3.0).ravel() | omega.ravel()
    s0[sup["nb"]] = True
    s0[sup["d_truth"]] = True
    assert np.array_equal(pkg.s0, np.flatnonzero(s0))
    assert len(sup["nb"]) > 0
    # dom = S0 | dilate(U_rec(P0) | U_rec(C11) in the OCM region, 3 px), closed
    u = package.urec(clr, wet)
    c11 = package.window_counters(s2, (2023, 2024, 2025), "C11.floor0.ndwi.D1")
    rs, cs = _region_slices(sup["ocm_region"])
    uc = np.zeros_like(u)
    uc[rs, cs] = package.urec(*c11)[rs, cs]
    m = _disk_dilation(u | uc, 3.0).ravel()
    m[pkg.s0] = True
    m = package.close_lattice(m.reshape(zs.H, zs.W), zs.PHASE).ravel()
    assert np.array_equal(pkg.dom, np.flatnonzero(m))
    assert sorted(pkg.dom_meta["terms"]) == ["C1.floor0.ndwi.D1", "C11.floor0.ndwi.D1"]
    # the closure: every 20 m block touching dom lies inside it
    blocks = package.Blocks1D(pkg.dom, zs.W, zs.PHASE)
    assert np.array_equal(blocks.counts[blocks.inverse],
                          _block_sizes(pkg.dom, zs.H, zs.W, zs.PHASE))


def _block_sizes(flat, Hh, Ww, phase):
    """In-frame size of each pixel's 20 m block (brute force)."""
    py, px = phase
    rr, cc = np.asarray(flat) // Ww, np.asarray(flat) % Ww
    br, bc = (rr + py) // 2, (cc + px) // 2
    out = []
    for a, b in zip(br, bc):
        rows = [r for r in (2 * a - py, 2 * a - py + 1) if 0 <= r < Hh]
        cols = [c for c in (2 * b - px, 2 * b - px + 1) if 0 <= c < Ww]
        out.append(len(rows) * len(cols))
    return np.asarray(out, float)


def test_v2_dom_version_is_stable_and_frozen(zsite, tmp_path):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    rec = package._read_record(s2)
    dv0 = package.current_dom(s2)["dom_version"]
    outs, meta = package.v2_dom(s2, rec, use_c11=True)
    assert meta["dom_version"] == dv0
    cur = package.freeze_dom(s2)
    assert cur["frozen"] and cur["dom_version"] == dv0
    package.v2_dom(s2, rec, use_c11=True)                  # identical: accepted
    assert package.current_dom(s2)["frozen"]
    # a new D_truth would change S0 and dom: refused while frozen
    g = package._geometry(s2)
    zs.write_d_truth(s2, np.arange(0, zs.H * zs.W, 5))
    package.v2_support(s2, rec)
    with pytest.raises(package.DomFrozenError, match="frozen"):
        package.v2_dom(s2, rec, use_c11=True)
    outs, meta = package.v2_dom(s2, rec, use_c11=True, force=True)
    assert meta["dom_version"] != dv0
    assert g["keep_full"].size


def _copy_v2(zsite, dst):
    shutil.copytree(zsite.site2.root, dst)
    return dataclasses.replace(zsite.site2, root=str(dst).replace("\\", "/"))


def test_v2_freeze_refuses_without_d_truth(zsite, tmp_path):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    os.remove(os.path.join(s2.truth_dir, package.TRUTH_ELEV_V2))
    rec = package._read_record(s2)
    package.v2_support(s2, rec)
    package.v2_dom(s2, rec, use_c11=True)
    with pytest.raises(package.PackageError, match="D_truth"):
        package.freeze_dom(s2)


def test_v2_ocm_step_with_a_fake_predictor(zsite):
    pkg, f, s2 = zsite.pkg, zsite.frames, zsite.site2
    region = pkg.support["ocm_region"]
    rs, cs = _region_slices(region)
    assert (rs.stop - rs.start, cs.stop - cs.start) != (zs.H, zs.W)   # a real sub-region
    pred = zsite.predictor
    assert len(pred.calls) == pkg.T
    for q in (0, 30, 50, 63):
        stack = pred.calls[q]
        assert stack.shape == (3, rs.stop - rs.start, cs.stop - cs.start)
        for k, b in enumerate(("B04", "B03", "B8A")):
            want = obs.prep_band(f[b][q][rs, cs], int(pkg.offsets[q]), "floor0")
            want = np.where(np.isfinite(want), want, 0)
            assert np.array_equal(stack[k], want), (q, b)
    doc = product.read_json(package._year_dir(s2, 2025, "ocm", "ocm.json"))
    prov = doc["provenance"]
    assert prov["package"] == "omnicloudmask" and prov["injected"] is True
    assert "version" in prov and prov["licence"]
    assert doc["region"] == [int(v) for v in region]
    sel = pkg.window_years_safe(2024)
    cv = _c11_valid(zsite, sel)
    for key in ("C11.floor0.ndwi.D1", "C11.floor0.mndwi.D1"):
        s = settings.Setting.parse(key)
        o = obs.observe_setting(_raw(f, sel), s, offset=pkg.offsets[sel], lattice=zs.PHASE,
                                params=_fixed(s.index), cloud_valid=cv)
        clr, wet = package.year_counters(s2, 2024, key)
        assert np.array_equal(clr, o.V.sum(axis=0)) and np.array_equal(wet, o.W.sum(axis=0))
        c1 = package.year_counters(s2, 2024, s.replace(cloud="C1").id)
        out = np.ones((zs.H, zs.W), bool)
        out[rs, cs] = False
        assert np.array_equal(clr[out], c1[0][out])       # C1 analog outside the region
    bits = np.load(package._year_dir(s2, 2024, "ocm", "bits_C11.npy"))
    o = obs.observe_setting(_raw(f, sel), settings.P0.replace(cloud="C11"),
                            offset=pkg.offsets[sel], lattice=zs.PHASE, cloud_valid=cv)
    assert np.array_equal(package._unpack_rows(bits, zs.W), o.V)


def test_v2_tiles_hold_the_raw_values_at_dom(zsite):
    pkg, f = zsite.pkg, zsite.frames
    dom = pkg.dom
    for y in zs.YEARS:
        mm = pkg._tile(y, "B03")
        assert mm.shape == (pkg.n_tiles, len(pkg.window_years_safe(y)), package.TILE)
        assert mm.dtype == np.int16 and pkg._tile(y, "SCL").dtype == np.uint8
        if pkg.P % package.TILE:
            assert (np.asarray(mm[-1, :, pkg.P % package.TILE:]) == obs.FILL).all()
    allv = pkg.read(list(package.TILE_BANDS))
    for b in package.TILE_BANDS:
        assert np.array_equal(allv[b], f[b].reshape(pkg.T, -1)[:, dom]), b
    rng = np.random.default_rng(3)
    sel = np.sort(rng.choice(pkg.T, 17, replace=False))
    pos = np.sort(rng.choice(pkg.P, 300, replace=False))
    part = pkg.read(["B11", "CLD"], sel, pos)
    assert np.array_equal(part["B11"], f["B11"].reshape(pkg.T, -1)[sel][:, dom[pos]])
    cb = pkg.read("cloudbits")["cloudbits"]
    for p in (20, 35):
        want = _c3(f, np.arange(pkg.T), p).reshape(pkg.T, -1)[:, dom]
        assert np.array_equal((cb & package.CLOUDBITS[f"C3p{p}"]) > 0, want)
    rs, cs = _region_slices(pkg.support["ocm_region"])
    inreg = np.zeros((zs.H, zs.W), bool)
    inreg[rs, cs] = True
    assert np.array_equal((cb & package.CLOUDBITS["ocm_region"]) > 0,
                          np.broadcast_to(inreg.ravel()[dom], cb.shape))


V2_OBSERVED = ("C1.floor0.ndwi.D1", "C1.raw.ndwi.D1", "C1.floor0.mndwi.D1",
               "C1.floor0.ndwi20.D1", "C1.floor0.awei_sh.D1", "C1.floor0.ndwi.D5",
               "C4.floor0.ndwi.D1", "C11.floor0.ndwi.D1", "C3.floor0.ndwi.D1",
               "C1.floor0.nir.D1", "C1.floor0.ndwi.D3", "C1.floor0.ndwi.D2",
               "C1.floor0.mndwi.D3")


@pytest.mark.parametrize("sid", V2_OBSERVED)
def test_v2_observation_at_dom_equals_the_frame_observation(zsite, sid):
    pkg, f = zsite.pkg, zsite.frames
    s = settings.Setting.parse(sid)
    sel = np.arange(pkg.T)
    prm = pkg.index_params(s.index)
    thr = (pkg.scene_thresholds(s.detect, s.index, sel)
           if settings.DETECT[s.detect].per_scene else None)
    cv = None
    if s.pixel_rule == "C3":
        cv = _c3(f, sel, 20)
    elif s.pixel_rule == "C11":
        cv = _c11_valid(zsite, sel)
    o = obs.observe_setting(_raw(f, sel), s, offset=pkg.offsets, lattice=zs.PHASE, params=prm,
                            threshold=thr, cloud_valid=cv)
    rng = np.random.default_rng(len(sid))
    pos = np.sort(rng.choice(pkg.P, 700, replace=False))
    got = pkg.observe(s, sel, pos)
    flat = pkg.dom[pos]
    for name in ("X", "Y", "V", "W"):
        a = getattr(o, name).reshape(pkg.T, -1)[:, flat]
        b = getattr(got, name)
        if a.dtype == np.float32:
            assert np.array_equal(a.view(np.int32), b.view(np.int32)), name
        else:
            assert np.array_equal(a, b), name


def test_v2_settings_pass_equals_direct_computations(zsite):
    pkg, f = zsite.pkg, zsite.frames
    doc = pkg.settings_doc
    keep = pkg.full.keep
    hist = {}
    for i in ("ndwi", "nir", "mndwi", "awei_sh"):
        e = settings.INDEX[i]
        acc = 0
        for y in (2023, 2024, 2025):
            sel = pkg.window_years_safe(y)
            o = obs.observe_setting(_raw(f, sel), settings.Setting(index=i),
                                    offset=pkg.offsets[sel], lattice=zs.PHASE,
                                    params={"t_k": 0.0, "s_k": 1.0})
            x = o.X.reshape(len(sel), -1)[:, keep][o.V.reshape(len(sel), -1)[:, keep]]
            acc = acc + package._hist_fixed(e, x)
        hist[i] = acc
        lo, hi = e.hist_range
        assert doc["indices"][i]["q25"] == pytest.approx(
            package.hist_quantile(acc, lo, hi, 0.25), abs=0)
    e = settings.INDEX["nir"]
    edges = np.linspace(*e.hist_range, e.hist_bins + 1)
    t, _, _ = obs.otsu_hist(hist["nir"], (edges[:-1] + edges[1:]) / 2.0)
    assert doc["indices"]["nir"]["t_k"] == t and doc["indices"]["nir"]["t_nir"] == -t
    assert doc["indices"]["ndwi"]["s_k"] == 1.0
    assert doc["indices"]["mndwi"]["s_k"] == pytest.approx(
        doc["indices"]["mndwi"]["iqr"] / doc["indices"]["ndwi"]["iqr"])
    # D2 on Omega and D3 on the bounding box of dom plus the pad, per scene
    omega = np.zeros(zs.H * zs.W, bool)
    omega[pkg.omega] = True
    omega = omega.reshape(zs.H, zs.W)
    reg = package._bbox_region(package._flat_mask(pkg.dom, pkg.shape),
                               package.REGION_PAD_M / 10.0, pkg.shape)
    rs, cs = slice(reg[0], reg[1]), slice(reg[2], reg[3])
    n_used = 0
    for y in (2023, 2025):
        z = pkg.thresholds(y)
        sel = pkg.window_years_safe(y)
        for i in ("ndwi", "mndwi"):
            t_k = doc["indices"][i]["t_k"]
            o = obs.observe_setting(_raw(f, sel), settings.Setting(index=i),
                                    offset=pkg.offsets[sel], lattice=zs.PHASE,
                                    params={"t_k": t_k, "s_k": 1.0})
            for j in range(len(sel)):
                if i == "ndwi":
                    r = obs.d2_threshold(o.X[j], o.V[j], omega, t_k)
                    assert np.float32(r["t"]) == z["d2_t_ndwi"][j]
                    assert bool(r["used"]) == bool(z["d2_used_ndwi"][j])
                    n_used += int(r["used"])
                lat = ((zs.PHASE[0] - reg[0]) % 2, (zs.PHASE[1] - reg[2]) % 2)
                raw = {b: f[b][sel][j][None, rs, cs] for b in f}
                oo = obs.observe_setting(raw, settings.Setting(index=i),
                                         offset=int(pkg.offsets[sel][j]), lattice=lat,
                                         params={"t_k": t_k, "s_k": 1.0})
                r3 = obs.d3_threshold(oo.X[0], oo.V[0], t_k)
                assert np.float32(r3["t"]) == z[f"d3_t_{i}"][j]
                assert r3["n_buffer"] == z[f"d3_n_buffer_{i}"][j]
    assert n_used > 0


def test_v2_counters_at_dom_for_threshold_detectors(zsite):
    pkg, s2 = zsite.pkg, zsite.site2
    for key in package.dom_counter_plan(zs.KEYS):
        for y in (2024,):
            sel = pkg.window_years_safe(y)
            o = pkg.observe(key, sel)
            z = product.load_npz(package._shard_dir(s2, pkg.dom_version, y, "counters",
                                                    f"{key}.npz"))
            assert z["clr"].dtype == np.uint8
            assert np.array_equal(z["clr"], o.V.sum(axis=0))
            assert np.array_equal(z["wet"], o.W.sum(axis=0))


def test_v2_mnar_table(zsite):
    pkg, s2 = zsite.pkg, zsite.site2
    tab = pd.read_csv(package._p(s2, "mnar.csv"))
    fits = product.read_json(package._p(s2, "mnar.json"))["fits"]
    keep = pkg.full.keep
    for wid in ("W3", "Y2024"):
        sel = pkg.window(wid)
        scl = zsite.frames["SCL"][sel].reshape(len(sel), -1)[:, keep]
        for c in ("C1", "C4", "C1d", "C0"):
            t = tab[(tab.window == wid) & (tab.cloud == c)]
            assert t["n"].sum() == len(sel) * len(keep)
            want = int((~obs.pixel_valid(c, scl)).sum())
            assert t["n_masked"].sum() == want, (wid, c)
            assert fits[wid][c]["n_obs"] == len(sel) * len(keep)
        t = tab[(tab.window == wid) & (tab.cloud == "C1")]
        for code in (2, 3, 8, 9):
            sub = t[t.scl.astype(str) == str(code)]
            assert sub["n"].sum() == int((scl == code).sum())
            assert (sub["n_masked"] == sub["n"]).all()           # C1 masks these classes
        assert set(t.h_decile) <= set(range(10)) and set(t.rising) <= {0, 1}
    assert set(fits) >= {"W3", "Y2023", "Y2024", "Y2025"}
    assert "C11" in fits["W3"] and "C3" in fits["W3"]


def test_logistic_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(5)
    h = rng.uniform(-2, 2, 400)
    n = np.full(400, 2000)
    k = rng.binomial(n, 1 / (1 + np.exp(-(-1.0 + 0.8 * h))))
    b0, b1, se = package.logistic_slope(n, k, h)
    assert b1 == pytest.approx(0.8, abs=4 * se) and b0 == pytest.approx(-1.0, abs=0.05)
    assert np.isnan(package.logistic_slope(n[:2], k[:2], h[:2])[1])


def _tz_brute(f, sel, st):
    from scipy.ndimage import binary_dilation
    wv = np.zeros((zs.H, zs.W), int)
    dv = np.zeros((zs.H, zs.W), int)
    for j, q in enumerate(sel):
        if st["frame_bad"][q] > 0.05:
            continue
        s8 = f["SCL"][q]
        g = obs.prep_band(f["B03"][q], 0 if q < 42 else 1000, "raw")
        n = obs.prep_band(f["B08"][q], 0 if q < 42 else 1000, "raw")
        x = obs.ndwi(g, n)
        clear = np.isin(s8, (4, 5, 6)) & np.isfinite(x)
        wv += clear & (x > 0)
        dv += clear & (x <= 0)
    v = wv + dv
    v[v == 0] = 1
    tz = ~((wv / v >= 0.95) | (dv / v >= 0.95))
    return binary_dilation(tz, iterations=10)


def test_v2_subsets_and_scene_rules(zsite):
    pkg, f, s2 = zsite.pkg, zsite.frames, zsite.site2
    st = package._scene_stats_all(s2, pkg)
    for wid in ("W3", "Y2025"):
        z = package.load_subsets(s2, wid)
        sel = pkg.window(wid)
        assert np.array_equal(z["positions"], sel)
        c6 = np.array([obs.scene_c6(f["SCL"][q]) for q in sel])
        c7 = np.array([obs.scene_c7(f["SCL"][q]) for q in sel])
        keep = pkg.full.keep
        c8 = np.array([obs.scene_c8(obs.cloud_c1(f["SCL"][q]), keep) for q in sel])
        assert np.array_equal(z["rule_C6"], np.flatnonzero(c6))
        assert np.array_equal(z["rule_C7"], np.flatnonzero(c7))
        assert np.array_equal(z["rule_C8"], np.flatnonzero(c8))
        tz = _tz_brute(f, sel, st)
        bad = [float(np.count_nonzero(np.isin(f["SCL"][q], (3, 8, 9, 10)) & tz)) / tz.sum()
               for q in sel]
        assert np.array_equal(z["tz_bad"], bad)
        c2 = obs.scene_c2(st["frame_bad"].to_numpy()[sel], np.array(bad))
        assert np.array_equal(z["rule_C2"], np.flatnonzero(c2))
        assert len(z["rule_C6"]) < len(sel)                       # the cloudy scene goes
        T = len(sel)
        for r in ("C2", "C6"):
            n_rule = len(z[f"rule_{r}"])
            for d_, m in enumerate(settings.matched_subsets(zs.SITE, wid, np.arange(T),
                                                            n_rule, 3)):
                assert np.array_equal(z[f"matched_{r}_{d_}"], m)
        if wid == "W3":
            kind, subs = settings.rand_subsets(zs.SITE, "W3", T, 20, 5)
            assert kind == "nested"                              # T = 64 < 5 x 20
            for r_, sub in enumerate(subs):
                assert np.array_equal(z[f"rand_20_{r_}"], sub)
    assert np.array_equal(pkg.rule_subset("C6", "W3"),
                          pkg.window("W3")[package.load_subsets(s2, "W3")["rule_C6"]])


def test_v2_shard_manifests_and_keys(zsite, tmp_path):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    keys = package.shard_keys(s2)
    assert set(keys) == {2023, 2024, 2025} and len(set(keys.values())) == 3
    dv = package.current_dom(s2)["dom_version"]
    man = product.read_json(package._shard_dir(s2, dv, 2024, "manifest.json"))
    assert man["obs_core"] == obs.core_sha256() and man["dom_version"] == dv
    assert set(man["frozen"]) == set(package.SHARD_KEY_SECTIONS)
    assert set(man["files"]) >= {"tile_B03.npy", "tile_cloudbits.npy", "dom.npz"}
    # one data byte of 2024 changes 2024's key only
    p = package._shard_dir(s2, dv, 2024, "tile_B03.npy")
    b = bytearray(open(p, "rb").read())
    b[-3] ^= 0xFF
    open(p, "wb").write(bytes(b))
    package.v2_manifest(s2, package.V2Package(s2), 2024)
    k2 = package.shard_keys(s2)
    assert k2[2024] != keys[2024] and k2[2023] == keys[2023] and k2[2025] == keys[2025]
    mp = package._shard_dir(s2, dv, 2025, "manifest.json")
    m = product.read_json(mp)
    m["dom_version"] = "0" * 16
    product.write_json(mp, m)
    with pytest.raises(package.PackageError, match="package_key"):
        package.shard_keys(s2)


def test_v2_prepare_resumes_and_tracks_staleness(zsite, tmp_path):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    ran = package.prepare_v2(s2, provider=zsite.provider, predictor=zs.FakeOCM(),
                             sources_kw=zsite.sources_kw)
    assert ran == {}
    st = package.status_v2(s2)
    assert set(st["steps"].values()) == {"done"}
    ran = package.prepare_v2(s2, steps=("pass1",), years=(2024,), force=True,
                             provider=zsite.provider, predictor=zs.FakeOCM(),
                             sources_kw=zsite.sources_kw)
    assert list(ran) == ["pass1:2024"]
    st = package.status_v2(s2)["steps"]
    # same bytes (deterministic): nothing downstream became stale
    assert set(st.values()) == {"done"}
    # a changed counter makes geometry (which read it) stale
    p = package._counter_path(s2, 2024, "C1.floor0.ndwi.D1")
    z = product.load_npz(p)
    z["clr"][0, 0] += 1
    product.save_npz(p, z)
    st = package.status_v2(s2)["steps"]
    assert st["pass1:2024"] == "stale"
    ran = package.prepare_v2(s2, steps=("pass1", "geometry"), provider=zsite.provider,
                             predictor=zs.FakeOCM(), sources_kw=zsite.sources_kw)
    assert "pass1:2024" in ran
    assert package.status_v2(s2)["steps"]["geometry"] == "done"


def test_v2_pass2_refuses_a_large_dom(zsite, tmp_path, monkeypatch):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    rec = package._read_record(s2)
    dv, dom, meta = package.load_dom(s2)
    monkeypatch.setattr(package, "DOM_MAX_OVER_S0", 0.5)
    with pytest.raises(package.PackageError, match="refused"):
        package.v2_pass2(s2, rec, 2024, dv, dom)
    v = package.dom_verdict(meta)
    assert set(v) == {"ratio_ok", "ratio", "projected_w9_gb", "budget_gb_all_sites"}


@pytest.mark.parametrize("lattice", [(1, 0), (0, 1), None])
def test_index_on_columns_equals_the_frame_rule(lattice):
    rng = np.random.default_rng(9)
    Hh, Ww = 13, 17
    raw = {b: rng.integers(0, 3000, (3, Hh, Ww)).astype(np.float32)
           for b in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")}
    raw["B03"][1, 4, 4] = np.nan
    mask = rng.random((Hh, Ww)) < 0.2
    mask = package.close_lattice(mask, lattice)
    flat = np.flatnonzero(mask)
    blocks = package.Blocks1D(flat, Ww, lattice)
    for idx in ("ndwi", "mndwi", "ndwi20", "awei_sh", "awei_nsh", "wri", "nir"):
        e = settings.INDEX[idx]
        frame = obs.compute_index({b: raw[b] for b in e.bands}, idx, lattice)
        cols = package.compute_index_cols({b: raw[b].reshape(3, -1)[:, flat]
                                           for b in e.bands}, idx, blocks)
        a = frame.reshape(3, -1)[:, flat]
        assert np.array_equal(a.view(np.int32), np.asarray(cols, np.float32).view(np.int32)), idx


def test_v2_cli_pilot_dom_crosstab_and_status(zsite, tmp_path, monkeypatch, capsys):
    from experiments import validation_v2 as cli
    s2 = _copy_v2(zsite, tmp_path / "copy")
    monkeypatch.setenv("VALIDATION_V2_ROOT", s2.root)
    common = ["--sites", "escalda", "--cubes-root", zsite.inp["cubes"]]
    assert cli.main(["pilot", "dom", *common]) == 0
    out = capsys.readouterr().out
    assert "|S0| =" in out and "W9 package" in out
    doc = product.read_json(os.path.join(s2.root, "common", "pilot_dom.json"))
    r = doc["sites"]["escalda"]
    assert r["n_dom"] == zsite.pkg.P and r["dom_version"] == zsite.pkg.dom_version
    assert r["settings"]["C1.floor0.ndwi.D1"]["W3"]["n_keep"] == len(zsite.pkg.full.keep)
    assert cli.main(["pilot", "dom", "--freeze", *common]) == 0
    assert package.current_dom(s2)["frozen"]
    assert cli.main(["pilot", "crosstab", *common]) == 0
    x = product.read_json(os.path.join(s2.root, "common", package.CROSSTAB_JSON))
    assert x["p"] in (20, 35) and x["provisional"] is True
    yr = x["sites"]["escalda"]["years"]["2024"]["unknown"]
    assert yr["n_scenes"] == 21 and set(yr["scl"]) >= {"4", "6", "8"}
    assert yr["scl"]["8"]["q01"] >= 60                     # the synthetic clouds' CLD
    assert cli.main(["status", *common]) == 0
    assert "steps done" in capsys.readouterr().out


def test_v2_c11_without_the_ocm_step_is_refused(zsite, tmp_path):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    dv = package.current_dom(s2)["dom_version"]
    p = package._shard_dir(s2, dv, 2024, "shard.json")
    doc = product.read_json(p)
    doc["ocm"] = False
    product.write_json(p, doc)
    pkg = package.V2Package(s2)
    with pytest.raises(package.PackageError, match="C11 needs the OCM step"):
        pkg.observe("C11.floor0.ndwi.D1", pkg.window("Y2025"), np.arange(10))
    o = pkg.observe("C1.floor0.ndwi.D1", pkg.window("Y2025"), np.arange(10))
    assert o.V.shape == (22, 10)
    pkg.close()


def _write_legacy_v2(inp, path):
    """The synthetic Zarr site as a legacy cube: B03, B08, SCL of 2023-2025,
    offset-harmonised (the +1000 of 2025 removed), int16 with fill."""
    import xarray as xr
    x, y = zs.coords()
    t, arr = [], {b: [] for b in ("B03", "B08", "SCL")}
    for yy in zs.YEARS:
        t += list(zs.dates(yy))
        for b in arr:
            a = np.asarray(inp["data"][yy][b]).copy()
            if yy == 2025 and b != "SCL":
                m = a != zs.FILL
                a[m] = (a[m] - 1000).astype(np.int16)
            arr[b].append(a)
    ds = xr.Dataset({b: (("t", "y", "x"), np.concatenate(v).astype(np.int16))
                     for b, v in arr.items()}, coords={"t": pd.DatetimeIndex(t), "y": y, "x": x})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": zs.wkt()})
    enc = {b: {"dtype": "int16", "_FillValue": zs.FILL, "chunksizes": (1, 16, 16), "zlib": True}
           for b in arr}
    ds.to_netcdf(path, encoding=enc)


def test_v2_netcdf_source_builds_the_same_p0_package(zsite, tmp_path):
    """A legacy cube (B03, B08, SCL; offset-harmonised) builds the v2 package
    through NetCDFSource at the CATALOGUE times (the default instants, user
    decision of 2026-10-14): the record equals the Zarr one, only the keys of
    its bands are counted (the rest listed), OCM and C3 cannot run (dom =
    S0 ∪ dilate(U_rec of P0), declared), the tiles hold B03, B08 and SCL, and
    the P0 observation and the C2 scene rule at the common dom pixels equal
    the Zarr package's bit for bit."""
    inp = zsite.inp
    nc = tmp_path / "legacy.nc"
    _write_legacy_v2(inp, nc)
    lab = tmp_path / "overpass_times.json"
    lab.write_text(json.dumps({d: t for d, t in inp["instants"].items() if t is not None}),
                   encoding="utf-8")
    s2 = zs.site2_for(inp, tmp_path / "nc_out", source="netcdf", cube=str(nc),
                      labels=str(lab), instants="catalogue", keys=None)
    assert s2.observation_keys() and all(
        set(settings.Setting.parse(k).bands) <= {"B03", "B08", "SCL"}
        for k in s2.observation_keys())
    unav = s2.unavailable_keys()
    assert unav["C3.floor0.ndwi.D1"] == ["CLD"] and "B11" in unav["C1.floor0.mndwi.D1"]
    package.prepare_v2(s2, steps=("record", "pass1", "geometry"), provider=zsite.provider)
    zs.write_d_truth(s2, package._geometry(s2)["keep_full"][::3])
    ran = package.prepare_v2(s2, provider=zsite.provider, sources_kw=zs.sources_kw())
    assert not any(k.startswith("ocm:") for k in ran) and "levels" in ran
    rec = package._read_record(s2)
    assert rec["dates"] == list(zsite.pkg.dates) and rec["instants"] == "catalogue"
    assert rec["instants_path"].endswith("overpass_times.json") and rec["sensing_sha256"] is None
    assert rec["source"]["kind"] == "netcdf" and set(rec["offsets"]) == {0}
    assert rec["boundary"]["kind"] == "union" and "C3.floor0.ndwi.D1" in rec["keys_unavailable"]
    assert rec["t_ns"] == zsite.pkg.record["t_ns"] and rec["lattice20"]["phase"] == list(zs.PHASE)
    pkg = package.V2Package(s2)
    try:
        assert pkg.bands == ("B03", "B08", "SCL") and pkg.source == "netcdf"
        assert pkg.dom_meta["terms"] == [package.P0_KEY] and "note" in pkg.dom_meta
        shard = product.read_json(package._shard_dir(s2, pkg.dom_version, 2024, "shard.json"))
        assert shard["bands"] == ["B03", "B08", "SCL"] and shard["c3_p"] == []
        man = product.read_json(package._shard_dir(s2, pkg.dom_version, 2024, "manifest.json"))
        assert sorted(man["files"]) == ["dom.npz", "tile_B03.npy", "tile_B08.npy",
                                        "tile_SCL.npy", "tile_cloudbits.npy"]
        with pytest.raises(package.PackageError, match="not in this package"):
            pkg.read("B11", np.arange(3), np.arange(3))
        with pytest.raises(package.PackageError, match="C3 needs CLD"):
            pkg.observe("C3.floor0.ndwi.D1", np.arange(3), np.arange(3))
        assert "C3" not in product.read_json(os.path.join(s2.dir, "mnar.json"))["clouds"]
        common = np.intersect1d(pkg.dom, zsite.pkg.dom)
        assert len(common) > 0.5 * len(zsite.pkg.dom)
        a = pkg.observe(package.P0_KEY, None, pkg.positions(common))
        b = zsite.pkg.observe(package.P0_KEY, None, zsite.pkg.positions(common))
        for x, y in zip(a, b):
            assert np.array_equal(x, y, equal_nan=True)
        for wid in ("W3", "Y2024"):
            za, zb = package.load_subsets(s2, wid), package.load_subsets(zsite.site2, wid)
            assert np.array_equal(za["rule_C2"], zb["rule_C2"])
            assert np.array_equal(za["positions"], zb["positions"])
    finally:
        pkg.close()
    with pytest.raises(package.PackageError, match="boundary"):
        package.prepare_v2(dataclasses.replace(s2, boundary="production"), steps=("pass1",),
                           provider=zsite.provider)


def test_v2_series_step_serves_the_lag_truth_inputs(zsite, tmp_path):
    """The 'series' step (run by the truth command): the 1-min h_B over the
    provider's period (v8's grid and tag) and its marks; the package serves
    series1, series10 and marks_lp to the lag truths."""
    s2 = dataclasses.replace(_copy_v2(zsite, tmp_path / "copy"), boundary="production")
    rec = package._read_record(s2)
    outs, info = package.v2_series(s2, rec, zsite.provider)
    assert [os.path.basename(o) for o in outs] == [package.SERIES_V2, package.MARKS_V2,
                                                   package.SERIES_V2[:-4] + ".key.json"]
    pkg = package.V2Package(s2)
    try:
        t1, h1 = pkg.series1
        grid = levels.series_1min_index(config.PERIOD)
        assert len(h1) == len(grid) == info["n"]
        np.testing.assert_allclose(h1[:500], FakeTide.value(grid[:500].asi8), atol=1e-9)
        idx, h10 = pkg.series10
        assert len(idx) == len(h10) and np.isfinite(h10).all()
        assert pkg.marks_lp["n_marks"] > 1000 and pkg.marks_raw["n_marks"] > 1000
    finally:
        pkg.close()
    with pytest.raises(ValueError, match="unknown step"):
        package.prepare_v2(s2, steps=("nope",))


# ═════════════════════════════════════════════════════════════════════════════
#  the reviews of 2026-10-14 (milestone V1): staleness, refusals, disk, sources
# ═════════════════════════════════════════════════════════════════════════════

def test_v2_record_refuses_other_instants_and_rebuilds_on_new_content(zsite, tmp_path):
    """A record is never reused silently with other instants: another
    source (a sensing_times.json against the record's) is refused before
    any step runs; the same source with other content makes the record and
    everything downstream stale."""
    s2 = _copy_v2(zsite, tmp_path / "copy")
    rec = package._read_record(s2)
    assert rec["instants"] == "given" and len(rec["instants_sha256"]) == 64
    sens = tmp_path / "sensing_times.json"
    sens.write_text(json.dumps({"site": zs.SITE, "dates": {
        d: {"instant": t} for d, t in zsite.inp["instants"].items()}}), encoding="utf-8")
    before = product.read_json(os.path.join(s2.dir, package.V2_BOOK))
    with pytest.raises(package.PackageError, match="built with the given instants"):
        package.prepare_v2(dataclasses.replace(s2, instants=str(sens)), provider=zsite.provider,
                           predictor=zs.FakeOCM(), sources_kw=zsite.sources_kw)
    assert product.read_json(os.path.join(s2.dir, package.V2_BOOK)) == before
    with pytest.raises(package.PackageError, match="sensing_times"):
        package.check_record_instants(dataclasses.replace(s2, instants=str(sens)), rec)
    # a date the downloader adds for another year changes nothing
    more = dict(zsite.inp["instants"], **{"2026-01-05": "2026-01-05T10:57:00"})
    s2y = dataclasses.replace(s2, instants=more)
    assert package._V2Book(s2y).why_stale("record", [], package.record_inputs(s2y)) is None
    # the same source with one instant moved by a minute: the record reruns
    moved = dict(zsite.inp["instants"])
    d0 = rec["dates"][5]
    moved[d0] = (pd.Timestamp(moved[d0]) + pd.Timedelta(minutes=1)).isoformat()
    s2m = dataclasses.replace(s2, instants=moved)
    book = package._V2Book(s2m)
    assert "instants" in book.why_stale("record", [], package.record_inputs(s2m))
    ran = package.prepare_v2(s2m, steps=("record",), provider=zsite.provider)
    assert list(ran) == ["record"]
    rec2 = package._read_record(s2m)
    assert rec2["instants_sha256"] != rec["instants_sha256"]
    st = package.status_v2(s2m)["steps"]
    assert st["record"] == "done" and st["pass1:2023"] == "stale" and st["levels"] == "stale"


def test_v2_years_of_an_existing_record_select_per_year_steps_only(zsite, tmp_path, capsys):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    capsys.readouterr()
    ran = package.prepare_v2(dataclasses.replace(s2, years=(2024,)), steps=("record", "pass1"),
                             provider=zsite.provider)
    assert ran == {}
    out = capsys.readouterr().out
    assert "select the per-year steps only" in out and "pass1:2024: done" in out
    assert "pass1:2023" not in out
    assert package._read_record(s2)["years"] == ["2023", "2024", "2025"]


def test_v2_record_refuses_missing_stores_and_provisional_offsets(zsite, tmp_path):
    base = dataclasses.replace(zsite.site2, root=str(tmp_path / "fresh").replace("\\", "/"))
    with pytest.raises(package.PackageError, match=r"stores of \[2026\] are not complete"):
        package.prepare_v2(dataclasses.replace(base, years=(2023, 2024, 2025, 2026)),
                           steps=("record",), provider=zsite.provider)
    assert not os.path.exists(os.path.join(base.dir, "record.json"))
    nooff = dataclasses.replace(base, offsets=None)
    with pytest.raises(package.PackageError, match="inputs --sites escalda"):
        package.prepare_v2(nooff, steps=("record",), provider=zsite.provider)
    assert not os.path.exists(os.path.join(base.dir, "record.json"))
    ins = package.record_inputs(dataclasses.replace(nooff, allow_provisional_offsets=True))
    assert ins["allow_provisional_offsets"] and ins["offsets"]["sha256"] is None
    assert ins["offsets"]["source"].startswith("provisional")
    with pytest.raises(ValueError, match="even"):
        dataclasses.replace(base, block_rows=21)


def test_v2_steps_go_stale_when_their_inputs_change(zsite, tmp_path):
    """support reads D_truth, pass 2 reads the OCM step once it exists; the
    remedy freeze_dom prints (prepare --step support,dom) now runs them."""
    s2 = _copy_v2(zsite, tmp_path / "copy")
    book = package._V2Book(s2)
    dv = package.current_dom(s2)["dom_version"]
    key = f"pass2:{dv}:2024"
    assert book.why_stale(key, ["dom", "pass1:2024", "ocm:2024"]) is None
    assert "upstream steps changed" in book.why_stale(key, ["dom", "pass1:2024"])
    assert book.why_stale("support", None, package.support_inputs(s2)) is None
    g = package._geometry(s2)
    zs.write_d_truth(s2, g["keep_full"][1::2])                  # a new D_truth arrives
    assert "d_truth" in book.why_stale("support", None, package.support_inputs(s2))
    with pytest.raises(package.PackageError, match="older D_truth"):
        package.freeze_dom(s2)
    ran = package.prepare_v2(s2, steps=("support", "dom"), provider=zsite.provider,
                             predictor=zs.FakeOCM(), sources_kw=zsite.sources_kw)
    assert {"support", "dom"} <= set(ran)
    sup = package._support(s2)
    assert np.array_equal(np.sort(sup["d_truth"]), np.unique(g["keep_full"][1::2]))
    assert package.freeze_dom(s2)["frozen"]


def test_v2_freeze_refuses_a_truth_without_every_campaign_window(zsite, tmp_path):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    flat = package._support(s2)["d_truth"]
    zs.write_d_truth(s2, flat, windows=("W3",))
    with pytest.raises(package.PackageError, match="registers"):
        package.freeze_dom(s2)
    zs.write_d_truth(s2, flat)
    assert package.freeze_dom(s2)["frozen"]


def test_v2_package_keys_follow_its_files_and_levels_follow_the_record(zsite, tmp_path):
    from experiments.benchmark import truth
    s2 = _copy_v2(zsite, tmp_path / "copy")
    pkg = package.V2Package(s2)
    k = dict(pkg.keys)
    assert set(k) == set(package.V2Package.KEY_FILES) | {"package_key"}
    assert k["record"] == product.sha256_file(os.path.join(s2.dir, "record.json"))
    assert k["levels"] == product.sha256_file(os.path.join(s2.dir, "levels.npz"))
    assert truth._package_key(pkg) == k["package_key"]
    assert pkg.levels.tau_hi > 0
    pkg.close()
    rp = os.path.join(s2.dir, "record.json")
    rec = product.read_json(rp)
    rec["h0"][3] += 0.01                       # a record rebuilt on other levels
    package.write_json_lf(rp, rec)
    pkg2 = package.V2Package(s2)
    try:
        assert pkg2.keys["record"] != k["record"] and pkg2.keys["package_key"] != k["package_key"]
        with pytest.raises(package.PackageError, match="another record"):
            pkg2.levels
    finally:
        pkg2.close()


def test_v2_free_space_check_reserve_and_stale_temporaries(tmp_path, monkeypatch):
    import errno
    import socket
    d = tmp_path / "out"
    d.mkdir()
    assert package.memmap_bytes((3, 4, 5), np.int16) == 3 * 4 * 5 * 2 + 4096
    with pytest.raises(package.PackageError, match="GB are free"):
        package.check_space(str(d / "not" / "yet"), 10 ** 18, "a test step", margin_gb=0)
    assert package.check_space(str(d), 1, "a test step", margin_gb=0) > 0
    monkeypatch.setenv("VALIDATION_V2_SPACE_MARGIN_GB", "1e9")      # local override
    assert package.space_margin_gb() == 1e9
    with pytest.raises(package.PackageError, match="margin"):
        package.check_space(str(d), 1, "a test step")
    monkeypatch.delenv("VALIDATION_V2_SPACE_MARGIN_GB")
    assert package.space_margin_gb() == package.SPACE_MARGIN_GB
    f = d / "x.npy"
    f.write_bytes(b"\0" * 64)

    def full(fd, off, n):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(package.os, "posix_fallocate", full, raising=False)
    with pytest.raises(package.PackageError, match="no space left"):
        package.reserve_file(str(f))
    monkeypatch.setattr(package.os, "posix_fallocate", lambda fd, off, n: None, raising=False)
    assert package.reserve_file(str(f)) is True
    host = socket.gethostname()
    dead = d / f"tile_B03.npy.tmp-{host}-999999999"
    live = d / f"tile_B08.npy.tmp-{host}-{os.getpid()}"
    other = d / "tile_B11.npy.tmp-otherhost-1"
    for p in (dead, live, other):
        p.write_bytes(b"x")
    removed = package.clean_stale_tmps(str(d))
    assert [os.path.basename(p) for p in removed] == [dead.name]
    assert live.exists() and other.exists() and not dead.exists()


@pytest.mark.parametrize("lattice", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_bbox_region_moves_out_to_whole_lattice_blocks(lattice):
    H_, W_ = 31, 29
    mask = np.zeros((H_, W_), bool)
    mask[6:12, 9:15] = True
    plain = package._bbox_region(mask, 2, (H_, W_))
    r0, r1, c0, c1 = package._bbox_region(mask, 2, (H_, W_), lattice)
    py, px = lattice
    assert r0 <= plain[0] and r1 >= plain[1] and c0 <= plain[2] and c1 >= plain[3]
    assert (r0 - py) % 2 == 0 or r0 == 0
    assert (c0 - px) % 2 == 0 or c0 == 0
    assert (r1 - py) % 2 == 0 or r1 == H_
    assert (c1 - px) % 2 == 0 or c1 == W_
    assert r1 - plain[1] <= 1 and plain[0] - r0 <= 1


def test_v2_sources_step_writes_each_level_source_with_its_status(zsite, tmp_path):
    """LB from the table; LM per regional dataset and window (at the Dutch
    synthetic site: AMM15 for W1 = Y2025, NOT AVAILABLE for W3 and the
    earlier years), never zero; LG NOT COMPUTABLE before the judge table;
    a CMEMS manifest arriving later makes the step stale."""
    from experiments.benchmark.fetch import cmems
    s2 = _copy_v2(zsite, tmp_path / "copy")
    doc = product.read_json(os.path.join(s2.dir, package.SOURCES_JSON))
    files = doc["files"]
    assert files["levels_LB.npz"]["status"] == "ok"
    assert files["levels_LT.npz"]["status"] == "ok"
    assert files["levels_LG.npz"]["unit_status"] == "NOT COMPUTABLE"
    ds = "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT15M-i"
    assert doc["lm_windows"]["W1"]["dataset"] == ds == doc["lm_windows"]["Y2025"]["dataset"]
    for wid in ("W3", "Y2023", "Y2024"):
        e = doc["lm_windows"][wid]
        assert e["dataset"] is None and e["status"] == "NOT AVAILABLE" and e["file"] is None
        assert "never substituted" in e["reason"]
    assert "no CMEMS manifest" in files[doc["lm_windows"]["W1"]["file"]]["reason"]
    w3 = package.load_source(s2, "LM", "W3")
    assert w3.unit_status()["status"] == "NOT AVAILABLE"
    # a CMEMS manifest that appears later makes the sources step stale
    book = package._V2Book(s2)
    ups = ["record", "geometry", "dom", "levels"]
    root = str(tmp_path / "cmems")
    ex0 = package.sources_inputs(s2, [2023, 2024, 2025], root, zsite.sources_kw["lt_kw"])
    d = cmems.site_cache_dir(zs.SITE, root, ds)
    os.makedirs(d)
    product.write_json(os.path.join(d, cmems.MANIFEST_NAME), {"status": "ok", "files": {}})
    ex1 = package.sources_inputs(s2, [2023, 2024, 2025], root, zsite.sources_kw["lt_kw"])
    assert ex0["lm"][ds]["manifest"] is None and ex1["lm"][ds]["manifest"] is not None
    book.doc["steps"]["sources"]["extra_sha256"] = config.sha256_json(ex0)
    book.doc["steps"]["sources"]["extra"] = config._plain(ex0)
    assert "lm" in book.why_stale("sources", ups, ex1)
    # LM computed from injected fields (2025, 15 min) over the frame
    pkg = package.V2Package(s2)
    try:
        lo0, lo1, la0, la1 = cmems.bbox_lonlat(pkg.grid)
        lon = np.arange(lo0 - 0.01, lo1 + 0.02, 0.01)
        lat = np.arange(la0 - 0.01, la1 + 0.02, 0.01)
        tt = pd.date_range("2024-12-31 22:00", "2026-01-01 02:00", freq="15min")
        u = np.asarray((tt - tt[0]) / pd.Timedelta(hours=1), float)
        zos = (0.3 * np.cos(2 * np.pi * u / 12.42))[:, None, None] + 0 * lon[None, None, :] \
            + 0 * lat[None, :, None]
        fields = [{"t": tt, "lon": lon, "lat": lat, "zos": zos}]
        outs, info = package.v2_sources(s2, pkg, sources=("LM",), lm_fields={ds: fields},
                                        cmems_root=str(tmp_path / "cmems"))
    finally:
        pkg.close()
    doc = product.read_json(os.path.join(s2.dir, package.SOURCES_JSON))
    name = doc["lm_windows"]["W1"]["file"]
    assert doc["lm_windows"]["W1"]["status"] == "ok" and info[name] == "ok"
    lm = package.load_source(s2, "LM", "W1")
    assert lm.computable and lm.flags["dataset"] == ds


def test_v2_ocm_resumes_a_killed_year_at_its_last_time_block(zsite, tmp_path):
    """OmniCloudMask is the long pole of V1: a year interrupted after its
    first stored time block resumes there (same dates, region, plan, card
    and predictor) and ends with the same bits and counters as one run."""
    s2 = _copy_v2(zsite, tmp_path / "copy")
    rec = package._read_record(s2)
    keys = s2.observation_keys()
    region = [int(v) for v in package._support(s2)["ocm_region"]]
    d = package._year_dir(s2, 2024, "ocm")
    ref = {n: np.load(os.path.join(d, n)) for n in ("bits_C11.npy", "ocm_clear.npy")}
    ref_c = {f: product.load_npz(os.path.join(d, "counters", f))
             for f in os.listdir(os.path.join(d, "counters"))}

    class Killed(zs.FakeOCM):
        def __call__(self, stack):
            if len(self.calls) == 20:
                raise KeyboardInterrupt("killed")
            return super().__call__(stack)

    src = package.open_source(s2, rec)
    first = len(src.time_blocks(package.year_positions(rec, 2024))[0])  # one stored chunk
    src.close()
    assert 0 < first < 20
    with pytest.raises(KeyboardInterrupt):
        package.v2_ocm(s2, rec, 2024, keys, region, predictor=Killed())
    prog = product.read_json(os.path.join(d, package.OCM_PROGRESS))
    assert prog["done"] == first and os.path.exists(os.path.join(d, "bits_C11.npy.partial"))
    again = zs.FakeOCM()
    outs, info = package.v2_ocm(s2, rec, 2024, keys, region, predictor=again)
    assert info["resumed_at"] == first and len(again.calls) == info["T"] - first
    for n, a in ref.items():
        assert np.array_equal(np.load(os.path.join(d, n)), a), n
    for f, z in ref_c.items():
        got = product.load_npz(os.path.join(d, "counters", f))
        assert np.array_equal(got["clr"], z["clr"]) and np.array_equal(got["wet"], z["wet"]), f
    assert not os.path.exists(os.path.join(d, package.OCM_PROGRESS))
    assert not [f for f in os.listdir(d) if f.endswith(".partial")]
    meta = product.read_json(os.path.join(d, "ocm.json"))
    assert meta["resumed_at"] == first and meta["provenance"]["injected"]


def test_v2_status_lists_dom_versions_and_prune_keeps_current_and_frozen(zsite, tmp_path):
    s2 = _copy_v2(zsite, tmp_path / "copy")
    cur = package.current_dom(s2)["dom_version"]
    for dv in ("0" * 16, "1" * 16):                    # two superseded versions
        for sub in ("dom", "shards"):
            os.makedirs(os.path.join(s2.dir, sub, dv, "2024"))
            with open(os.path.join(s2.dir, sub, dv, "2024", "x.npy"), "wb") as f:
                f.write(b"\0" * 1000)
    doc = product.read_json(os.path.join(s2.dir, "dom", "current.json"))
    doc["frozen_history"] = ["1" * 16]                  # frozen once: never pruned
    package.write_json_lf(os.path.join(s2.dir, "dom", "current.json"), doc)
    st = package.status_v2(s2)["dom_versions"]
    assert st[cur]["current"] and not st["0" * 16]["current"] and st["1" * 16]["frozen"]
    assert package.prune_stale_shards(s2, dry_run=True) == {"0" * 16: 0.0}
    assert os.path.isdir(os.path.join(s2.dir, "shards", "0" * 16))
    freed = package.prune_stale_shards(s2)
    assert list(freed) == ["0" * 16]
    assert not os.path.exists(os.path.join(s2.dir, "shards", "0" * 16))
    assert os.path.isdir(os.path.join(s2.dir, "shards", "1" * 16))
    assert os.path.isdir(os.path.join(s2.dir, "shards", cur))
    assert package.status_v2(s2)["steps"][f"pass2:{cur}:2024"] == "done"
