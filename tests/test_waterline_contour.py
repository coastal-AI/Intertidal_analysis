"""Tests of pyintertidal.waterline_contour (ITEM/NIDEM re-implementation).

The regression test the method must pass before any real site is run: a
synthetic planar tidal flat, observed under a synthetic tide with and
without NDWI noise and clouds, is recovered within the contour spacing.
The other tests pin the frozen choices: interval edges and merging, the
tie rule, contours keyed by level when a level is empty, NaN-safe
contouring and the absence of a half-pixel shift. Since the revision of
2026-09-29 they also pin the ITEM pixel-quality mask mapped to SCL, the
literal load_ard scene rule and pixel mask of the variant (no data counted
as good), the truth-free pre-pass, and the Villaviciosa record switch.

Run:  python -m pytest tests/test_waterline_contour.py
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyintertidal import waterline_contour as wc  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  synthetic planar flat
# ─────────────────────────────────────────────────────────────────────────────

def planar_flat(H=48, W=64, T=240, noise=0.0, cloud=0.0, seed=0):
    """A tilted plane from -2.45 to +2.45 m (wider than the tide, so all
    nine waterlines exist) under a two-constituent tide of range ~4 m.
    NDWI is +0.45 over water and -0.35 over land, plus Gaussian noise;
    cloudy observations carry a garbage NDWI of +0.9 that the method must
    ignore."""
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[0:H, 0:W]
    z = -2.3 + 4.6 * cc / (W - 1) + 0.3 * (rr / (H - 1) - 0.5)
    t_h = np.sort(rng.uniform(0.0, 24.0 * 400, T))
    h = (1.6 * np.sin(2 * np.pi * t_h / 12.42)
         + 0.4 * np.sin(2 * np.pi * t_h / 12.0 + 1.0))
    wet = h[:, None, None] > z[None]
    ndwi = np.where(wet, 0.45, -0.35) + noise * rng.standard_normal((T, H, W))
    clear = rng.random((T, H, W)) >= cloud
    ndwi = np.where(clear, ndwi, 0.9).astype(np.float32)
    return z, h, ndwi, clear


def _recovery(noise, cloud, seed):
    z, h, ndwi, clear = planar_flat(noise=noise, cloud=cloud, seed=seed)
    res, (j, edges, Z, U, sizes) = wc.nidem_from_arrays(ndwi, clear, h)
    spacing = float(np.max(np.diff(Z)))
    return z, res, Z, spacing


@pytest.mark.parametrize("noise,cloud,seed", [(0.0, 0.0, 1), (0.15, 0.3, 2)])
def test_planar_flat_recovered_within_contour_spacing(noise, cloud, seed):
    z, res, Z, spacing = _recovery(noise, cloud, seed)
    assert res["empty_levels"] == []
    for name in ("unfiltered", "filtered"):
        d = res[name]
        ok = np.isfinite(d)
        err = d[ok] - z[ok]
        # the zone between the lowest and highest waterline must be mapped
        inside = (z > Z[0] + 0.05) & (z < Z[-1] - 0.05)
        assert ok[inside].mean() > 0.9, (name, ok[inside].mean())
        assert np.max(np.abs(err)) <= spacing, (name, np.max(np.abs(err)),
                                                spacing)
        assert np.sqrt(np.mean(err ** 2)) < 0.25 * spacing
    # the relative extents are ordinal in the true elevation
    R = res["R"]
    assert np.isfinite(R).mean() > 0.99
    ok = np.isfinite(R)
    assert np.corrcoef(R[ok], z[ok])[0, 1] > 0.95
    # the published confidence mask is honoured
    f = np.isfinite(res["filtered"])
    assert np.all(res["conf"][f] <= wc.CONFIDENCE_MAX)
    assert np.all(np.isfinite(res["unfiltered"][f]))


def test_planar_flat_errors_quoted():
    """Report the recovery errors (printed with -s) for the record."""
    for noise, cloud, seed in ((0.0, 0.0, 1), (0.15, 0.3, 2)):
        z, res, Z, spacing = _recovery(noise, cloud, seed)
        d = res["filtered"]
        ok = np.isfinite(d)
        err = d[ok] - z[ok]
        print(f"noise {noise} cloud {cloud}: spacing max {spacing:.3f} m, "
              f"filtered max|err| {np.max(np.abs(err)):.3f} m, "
              f"RMSE {np.sqrt(np.mean(err ** 2)):.4f} m, n {ok.sum()}")


# ─────────────────────────────────────────────────────────────────────────────
#  intervals
# ─────────────────────────────────────────────────────────────────────────────

def test_interval_edges_and_merging():
    h = np.array([0.0, 1.0, 1.5, 2.0, 3.0, 3.5, 4.2, 5.0, 6.0, 7.0, 8.0,
                  8.5, 9.0, 9.5, 10.0])
    j, edges, Z, U, sizes = wc.interval_assignment(h)
    np.testing.assert_allclose(edges, np.arange(11.0), atol=1e-12)
    # lowest interval closed [e0, e1]; others right-closed (e_{k-1}, e_k];
    # raw interval 10 merged into 9
    expected = [1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 9, 9]
    assert j.tolist() == expected
    assert sizes.tolist() == [2, 2, 1, 1, 2, 1, 1, 1, 4]
    # the merged top interval holds raw 80-90 % and 90-100 % scenes
    assert set(h[j == 9]) == {8.5, 9.0, 9.5, 10.0}
    np.testing.assert_allclose(Z[8], np.median([8.5, 9.0, 9.5, 10.0]))
    np.testing.assert_allclose(U[8], np.std([8.5, 9.0, 9.5, 10.0], ddof=1))
    np.testing.assert_allclose(Z[0], 0.5)


def test_empty_interval_stops():
    h = np.array([0.0, 0.05, 0.95, 1.0])      # nothing between 10 and 90 %
    with pytest.raises(wc.EmptyIntervalError):
        wc.interval_assignment(h)


def _flat_with_lone_low_scene(clear_patch=None, seed=9):
    """The planar flat plus one extra scene far below every other level, so
    that it alone forms interval 1 (the Ems storm set-down case). The scene
    is overcast everywhere, or clear only on ``clear_patch`` (a slice)."""
    z, h, ndwi, clear = planar_flat(H=24, W=32, T=160, seed=seed)
    h_low = h.min() - 0.15 * (h.max() - h.min())     # gap > OTR / 9
    h = np.append(h, h_low)
    ndwi = np.concatenate([ndwi, np.full((1,) + ndwi.shape[1:], 0.9,
                                         np.float32)])
    lone = np.zeros((1,) + clear.shape[1:], bool)
    if clear_patch is not None:
        lone[(0,) + clear_patch] = True
        ndwi[-1][clear_patch] = 0.45
    clear = np.concatenate([clear, lone])
    return z, h, ndwi, clear


def test_preflight_counts_equal_composite_counts():
    rng = np.random.default_rng(12)
    T, H, W = 40, 6, 7
    ndwi = rng.normal(0, 0.4, (T, H, W)).astype(np.float32)
    ndwi[rng.random(ndwi.shape) < 0.05] = np.nan      # clear but undefined
    clear = rng.random((T, H, W)) >= 0.35
    h = rng.permutation(np.linspace(-1.0, 1.0, T))
    j, *_ = wc.interval_assignment(h)
    comp = wc.composites_from_arrays(ndwi, clear, j)
    cnt, per_scene = wc.clear_counts_by_interval(
        clear.reshape(T, -1), ndwi.reshape(T, -1), j)
    assert np.array_equal(cnt, comp["count"].reshape(wc.N_INTERVALS, -1))
    assert np.array_equal(per_scene,
                          (clear & np.isfinite(ndwi)).reshape(T, -1).sum(axis=1))
    cov = wc.interval_coverage(cnt, raise_error=False)
    valid = np.all(comp["count"] >= 1, axis=0)
    assert cov["expected_valid_px"] == int(valid.sum())


def test_preflight_passes_on_a_normal_record():
    z, h, ndwi, clear = planar_flat(H=24, W=32, T=160, noise=0.1, cloud=0.3,
                                    seed=10)
    j, *_ = wc.interval_assignment(h)
    cnt, _ = wc.clear_counts_by_interval(clear.reshape(len(h), -1),
                                         ndwi.reshape(len(h), -1), j)
    cov = wc.interval_coverage(cnt)                    # does not raise
    assert cov["passed"] and cov["expected_valid_share"] > 0.99
    assert cov["intervals_below_floor"] == []


def test_lone_overcast_extreme_interval_stops():
    """Review round 2 (Ems): the only scene of interval 1 is overcast, so no
    pixel can be valid. The pre-flight and nidem() both stop and report
    instead of emitting an empty product."""
    z, h, ndwi, clear = _flat_with_lone_low_scene()
    j, edges, Z, U, sizes = wc.interval_assignment(h)
    assert sizes[0] == 1 and j[-1] == 1 and np.isnan(U[0])
    T = len(h)
    cnt, per_scene = wc.clear_counts_by_interval(clear.reshape(T, -1),
                                                 ndwi.reshape(T, -1), j)
    assert per_scene[-1] == 0
    with pytest.raises(wc.NoClearObservationsInIntervalError) as e:
        wc.interval_coverage(cnt)
    cov = e.value.coverage
    assert cov["intervals_with_zero_clear_obs"] == [1]
    assert cov["intervals_below_floor"] == [1]
    assert cov["expected_valid_px"] == 0 and not cov["passed"]
    assert cov["share_px_with_ge1_clear_obs"][1:] == [1.0] * 8
    # the same family as the empty-interval stop
    assert isinstance(e.value, wc.EmptyIntervalError)
    with pytest.raises(wc.NoClearObservationsInIntervalError):
        wc.nidem_from_arrays(ndwi, clear, h)


def test_lone_nearly_overcast_extreme_interval_stops_before_streaming():
    """A few clear pixels in the lone scene (27 of 2.7 M at Ems) leave nidem()
    runnable but the product (almost) empty; the pre-flight floor stops it."""
    z, h, ndwi, clear = _flat_with_lone_low_scene(
        clear_patch=(slice(10, 12), slice(14, 16)))   # 4 of 768 px
    j, *_ = wc.interval_assignment(h)
    T = len(h)
    cnt, per_scene = wc.clear_counts_by_interval(clear.reshape(T, -1),
                                                 ndwi.reshape(T, -1), j)
    assert per_scene[-1] == 4
    cov = wc.interval_coverage(cnt, raise_error=False)
    assert not cov["passed"] and cov["expected_valid_px"] == 4
    assert cov["intervals_below_floor"] == [1]
    assert cov["intervals_with_zero_clear_obs"] == []
    with pytest.raises(wc.NoClearObservationsInIntervalError):
        wc.interval_coverage(cnt)
    res, _ = wc.nidem_from_arrays(ndwi, clear, h)     # no guard inside nidem()
    assert res["valid"].sum() == 4
    assert np.isfinite(res["filtered"]).sum() <= 4


def test_single_scene_interval_has_nan_uncertainty():
    h = np.arange(10.0)                         # one scene per raw interval
    j, edges, Z, U, sizes = wc.interval_assignment(h)
    assert sizes.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2]
    assert np.all(np.isnan(U[:8])) and np.isfinite(U[8])


# ─────────────────────────────────────────────────────────────────────────────
#  composites
# ─────────────────────────────────────────────────────────────────────────────

def test_interval_statistics_match_numpy():
    rng = np.random.default_rng(5)
    for n in (1, 2, 7, 8):
        x = rng.normal(0, 0.3, (n, 9, 11)).astype(np.float32)
        x[rng.random(x.shape) < 0.3] = np.nan
        ref_med = _nanmedian(x)
        ref_sd = _nanstd(x)
        med, sd, cnt = wc.interval_statistics(x.copy())
        np.testing.assert_allclose(med, ref_med, rtol=0, atol=1e-6)
        np.testing.assert_allclose(sd, ref_sd, rtol=0, atol=1e-6)
        assert np.array_equal(cnt, np.isfinite(x).sum(axis=0))


def test_bounded_statistics_match_numpy():
    """Supplementary confidence: SD of NDWI clipped to [-1, 1], counts of
    out-of-range observations and the largest |NDWI| (L2A reflectances can
    be negative, so |NDWI| > 1 occurs)."""
    rng = np.random.default_rng(6)
    for n in (1, 2, 7, 8):
        x = rng.normal(0, 0.3, (n, 9, 11)).astype(np.float32)
        out = rng.random(x.shape) < 0.15
        x[out] = rng.choice([-1, 1], out.sum()) * rng.uniform(1.0, 40.0, out.sum())
        x[rng.random(x.shape) < 0.3] = np.nan
        x[:, 0, 0] = np.nan                                  # no observation
        s = x.copy()
        med, sd, cnt = wc.interval_statistics(s)
        sdb, above, below, mx = wc.bounded_statistics(s, cnt)
        np.testing.assert_allclose(sdb, _nanstd(np.clip(x, -1, 1)), rtol=0,
                                   atol=1e-6)
        assert np.isnan(sdb[0, 0]) and cnt[0, 0] == 0
        assert np.array_equal(above, np.sum(x > 1, axis=0))
        assert np.array_equal(below, np.sum(x < -1, axis=0))
        assert mx == pytest.approx(float(np.nanmax(np.abs(x))), abs=0)
        # the median was taken BEFORE clipping, and clipping is a contraction
        np.testing.assert_allclose(med, _nanmedian(x), rtol=0, atol=1e-6)
        ok = np.isfinite(sd)
        assert np.all(sdb[ok] <= sd[ok] + 1e-6)


def test_composite_median_uses_raw_ndwi():
    # interval 9 holds -1.5 and +2.0: raw median +0.25 (water); a clipped
    # median would be 0 (land). The frozen composite is on the raw NDWI.
    h, ndwi, clear = _one_scene_per_interval_cube(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1.5, 2.0])
    j, *_ = wc.interval_assignment(h)
    comp = wc.composites_from_arrays(ndwi, clear, j)
    assert comp["median"][8, 0, 0] == pytest.approx(0.25)
    assert not wc.land_flags(comp["median"])[8, 0, 0]
    assert comp["n_ndwi_above_bound"][0, 0] == 1
    assert comp["n_ndwi_below_minus_bound"][0, 0] == 1
    assert comp["max_abs_ndwi"] == 2.0
    # raw SD of interval 9 is 1.75, clipped SD is 1.0; the others are 0
    assert comp["sd_mean"][0, 0] == pytest.approx(1.75 / 9)
    assert comp["sd_mean_bounded"][0, 0] == pytest.approx(1.0 / 9)


def test_bounded_confidence_is_nested_between_filtered_and_unfiltered():
    z, h, ndwi, clear = planar_flat(noise=0.1, cloud=0.2, seed=7)
    rng = np.random.default_rng(8)
    bad = clear & (rng.random(ndwi.shape) < 0.03)       # impossible NDWI
    ndwi = ndwi.copy()
    ndwi[bad] = (rng.choice([-1, 1], bad.sum())
                 * rng.uniform(1.5, 60.0, bad.sum())).astype(np.float32)
    res, _ = wc.nidem_from_arrays(ndwi, clear, h)
    ok = np.isfinite(res["conf"])
    assert np.array_equal(ok, np.isfinite(res["conf_bounded"]))
    assert np.all(res["conf_bounded"][ok] <= res["conf"][ok] + 1e-6)
    f = np.isfinite(res["filtered"])
    fb = np.isfinite(res["filtered_bounded"])
    fu = np.isfinite(res["unfiltered"])
    assert np.all(fb[f]) and np.all(fu[fb])
    assert fb.sum() > f.sum()                  # the outliers did bind
    assert np.array_equal(res["filtered_bounded"][fb], res["dem"][fb])
    assert np.all(res["conf_bounded"][fb] <= wc.CONFIDENCE_MAX)


def _nanmedian(x):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(x.astype(np.float64), axis=0)


def _nanstd(x):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanstd(x.astype(np.float64), axis=0)


def _one_scene_per_interval_cube(values):
    """10 scenes, h = 0..9 -> intervals 1..8 one scene each, 9 two scenes.
    ``values`` (10,) NDWI of a 1 x 1 pixel."""
    h = np.arange(10.0)
    ndwi = np.asarray(values, np.float32).reshape(10, 1, 1)
    clear = np.ones_like(ndwi, bool)
    return h, ndwi, clear


def test_tie_counts_as_land():
    # interval 1 has a single observation of exactly 0 -> land
    h, ndwi, clear = _one_scene_per_interval_cube(
        [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    j, *_ = wc.interval_assignment(h)
    comp = wc.composites_from_arrays(ndwi, clear, j)
    land = wc.land_flags(comp["median"])[:, 0, 0]
    assert land.tolist() == [True] + [False] * 8
    # interval 9 has two observations -0.2 and +0.2: median 0 -> land
    h, ndwi, clear = _one_scene_per_interval_cube(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.2, 0.2])
    comp = wc.composites_from_arrays(ndwi, clear, j)
    assert comp["median"][8, 0, 0] == 0.0
    R, valid, non_mono = wc.relative_extents(comp["median"], comp["count"])
    assert R[0, 0] == 1 and valid[0, 0] and non_mono[0, 0]
    # a strictly positive median is water
    h, ndwi, clear = _one_scene_per_interval_cube([1e-6] * 10)
    comp = wc.composites_from_arrays(ndwi, clear, j)
    assert not wc.land_flags(comp["median"]).any()


def test_valid_needs_one_clear_observation_in_every_interval():
    h = np.arange(10.0)
    j, *_ = wc.interval_assignment(h)
    ndwi = np.full((10, 2, 2), -0.3, np.float32)
    clear = np.ones_like(ndwi, bool)
    clear[3, 0, 0] = False              # interval 4's only scene is cloudy
    ndwi[3, 0, 0] = 0.9                 # garbage under the cloud
    clear[8, 1, 1] = False              # one of interval 9's two scenes
    ndwi[8, 1, 1] = 0.9
    comp = wc.composites_from_arrays(ndwi, clear, j)
    R, valid, _ = wc.relative_extents(comp["median"], comp["count"])
    assert valid.tolist() == [[False, True], [True, True]]
    assert np.isnan(R[0, 0]) and R[1, 1] == 9
    assert comp["count"][8, 1, 1] == 1 and comp["count"][3, 0, 0] == 0
    assert np.isnan(comp["sd_mean"][0, 0])


def test_ndwi_matches_extraction_formula():
    g = np.array([[100, 0, np.nan, 300]], np.float32)
    n = np.array([[50, 0, 10, 300]], np.float32)
    scl = np.array([[4, 6, 8, np.nan]], np.float32)
    ndwi, clear = wc.scene_ndwi(g, n, scl)
    assert ndwi.dtype == np.float32
    assert ndwi[0, 0] == np.float32(50) / np.float32(150)
    assert np.isnan(ndwi[0, 1]) and np.isnan(ndwi[0, 2])
    assert ndwi[0, 3] == 0.0
    assert clear.tolist() == [[True, True, False, False]]


def _synthetic_cube(tmp_path, seed=11, T=30, H=17, W=13):
    """A small raw-band cube with every SCL code, NaN SCL (no data),
    negative L2A values, a zero band sum, a missing band and raw-DN-1
    (value -999) pixels."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(seed)
    b03 = rng.integers(-200, 2000, (T, H, W)).astype(np.float32)
    b08 = rng.integers(-300, 2000, (T, H, W)).astype(np.float32)  # L2A < 0
    b03[0, 0, 0] = b08[0, 0, 0] = 0.0          # NDWI undefined but valid SCL
    b03[1, 2, 3] = np.nan                       # missing band
    b08[rng.random(b08.shape) < 0.02] = -999.0  # raw DN 1
    scl = rng.choice(np.arange(1, 12), (T, H, W)).astype(np.float32)
    scl[rng.random(scl.shape) < 0.05] = np.nan  # SCL no data
    ds = xr.Dataset({v: (("t", "y", "x"), a) for v, a in
                     (("B03", b03), ("B08", b08), ("SCL", scl))},
                    coords={"t": np.arange(T), "y": np.arange(H)[::-1] * 10.0,
                            "x": np.arange(W) * 10.0})
    path = str(tmp_path / f"cube{seed}.nc")
    ds.to_netcdf(path)
    return path, b03, b08, scl


@pytest.mark.parametrize("rule", ["item", "load_ard"])
def test_streaming_equals_in_memory(tmp_path, rule):
    path, b03, b08, scl = _synthetic_cube(tmp_path)
    T, H, W = scl.shape
    rng = np.random.default_rng(12)
    time_index = np.arange(5, T)                          # a subset
    h = rng.permutation(np.linspace(-1.0, 1.0, len(time_index)))
    j, edges, Z, U, sizes = wc.interval_assignment(h)
    calls = []

    def check(t, r0, r1, nd, cl, sc):
        ref_nd, ref_cl = wc.scene_ndwi(b03[t, r0:r1], b08[t, r0:r1],
                                       scl[t, r0:r1], rule=rule)
        assert np.array_equal(nd, ref_nd, equal_nan=True)
        assert np.array_equal(cl, ref_cl)
        assert np.array_equal(sc, scl[t, r0:r1], equal_nan=True)
        calls.append((t, r0))

    s = wc.stream_composites(path, time_index, j, row_block=5, check=check,
                             verbose=False, rule=rule)
    nd, cl = wc.scene_ndwi(b03[time_index], b08[time_index], scl[time_index],
                           rule=rule)
    m = wc.composites_from_arrays(nd, cl, j)
    assert np.array_equal(s["median"], m["median"], equal_nan=True)
    assert np.array_equal(s["count"], m["count"])
    assert np.array_equal(s["sd_mean"], m["sd_mean"], equal_nan=True)
    assert np.array_equal(s["sd_mean_bounded"], m["sd_mean_bounded"],
                          equal_nan=True)
    for k in ("n_ndwi_above_bound", "n_ndwi_below_minus_bound"):
        assert np.array_equal(s[k], m[k])
    assert s["max_abs_ndwi"] == m["max_abs_ndwi"] > 1.0
    assert m["n_ndwi_above_bound"].sum() > 0 and m["n_ndwi_below_minus_bound"].sum() > 0
    assert len(calls) == len(time_index) * int(np.ceil(H / 5))


# ─────────────────────────────────────────────────────────────────────────────
#  contours and TIN
# ─────────────────────────────────────────────────────────────────────────────

def test_no_half_pixel_shift():
    H, W = 6, 10
    R = np.tile(np.arange(W, dtype=np.float32), (H, 1))     # R = column
    Z = 0.1 * np.arange(1, 10)
    U = np.full(9, 0.02)
    cont = wc.contours_by_level(R)
    assert sorted(cont) == list(wc.CONTOUR_LEVELS)
    for level, lines in cont.items():
        cols = np.concatenate([ln[:, 1] for ln in lines])
        np.testing.assert_allclose(cols, level, atol=1e-12)   # k - 0.5
    dem, unc, nv = wc.tin_dem(cont, Z, U, (H, W))
    # pixel column c (R = c) lies between levels c - 0.5 (Z_c) and
    # c + 0.5 (Z_{c+1}): exactly (Z_c + Z_{c+1}) / 2 = 0.1 c + 0.05
    for c in range(1, 9):
        np.testing.assert_allclose(dem[:, c], 0.1 * c + 0.05, atol=1e-9)
    np.testing.assert_allclose(unc[:, 1:9], 0.02, atol=1e-12)
    assert np.all(np.isnan(dem[:, 0])) and np.all(np.isnan(dem[:, 9]))


def test_degenerate_tin_gives_nan_not_an_error():
    # collinear vertices: Qhull cannot build a triangle; like < 3 vertices,
    # the DEM is NaN everywhere instead of a crash
    line = np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5], [1.0, 3.5]])
    dem, unc, nv = wc.tin_dem({0.5: [line], 1.5: [line + [0.0, 0.25]]},
                              0.1 * np.arange(1, 10), np.full(9, 0.01), (3, 5))
    assert nv == 8 and np.all(np.isnan(dem)) and np.all(np.isnan(unc))


def test_contours_keyed_by_level_when_a_level_is_empty():
    # region A: R 0..2, region B: R 5..9, separated by NaN: levels 2.5, 3.5
    # and 4.5 have no contour
    row = np.array([0, 1, 2, 2, np.nan, np.nan, np.nan, np.nan,
                    5, 6, 7, 8, 9, 9], np.float32)
    R = np.tile(row, (5, 1))
    cont = wc.contours_by_level(R)
    assert sorted(cont) == [0.5, 1.5, 5.5, 6.5, 7.5, 8.5]
    Z = 0.1 * np.arange(1, 10)
    U = 0.01 * np.arange(1, 10)
    dem, unc, _ = wc.tin_dem(cont, Z, U, R.shape)
    # R = 6 at column 9: between level 5.5 (Z_6) and 6.5 (Z_7)
    np.testing.assert_allclose(dem[:, 9], 0.65, atol=1e-9)
    np.testing.assert_allclose(unc[:, 9], 0.065, atol=1e-9)
    # the positional pairing of NIDEM_generation.py would have given
    # Z_3/Z_4 here (0.35); keyed by level it cannot
    assert not np.allclose(dem[:, 9], 0.35)
    # R = 1 at column 1: between level 0.5 (Z_1) and 1.5 (Z_2)
    np.testing.assert_allclose(dem[:, 1], 0.15, atol=1e-9)


def test_nan_safe_contouring():
    rng = np.random.default_rng(3)
    H, W = 40, 50
    rr, cc = np.mgrid[0:H, 0:W]
    R = np.clip(np.round(cc / 5.0), 0, 9).astype(np.float32)
    holes = rng.random((H, W)) < 0.08
    R[holes] = np.nan
    R[10:15, 20:30] = np.nan
    cont = wc.contours_by_level(R)
    assert cont, "no contour at all"
    for level, lines in cont.items():
        for ln in lines:
            assert np.all(np.isfinite(ln))
            assert len(ln) >= wc.MIN_VERTICES
            r0, c0 = np.floor(ln[:, 0]).astype(int), np.floor(ln[:, 1]).astype(int)
            r1, c1 = np.ceil(ln[:, 0]).astype(int), np.ceil(ln[:, 1]).astype(int)
            for a, b in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                assert np.all(np.isfinite(R[a, b])), "vertex touches NaN"
    # the full chain tolerates NaN holes: gap fill, contours, TIN, masks
    block = np.zeros((H, W), bool)
    block[10:15, 20:30] = True
    Rf = wc.gap_fill(R)
    assert np.isfinite(Rf[holes & ~block]).all()
    assert np.isnan(Rf[12, 25])              # > 2 px from any valid pixel
    dem, unc, _ = wc.tin_dem(wc.contours_by_level(Rf), 0.1 * np.arange(1, 10),
                             np.full(9, 0.01), R.shape)
    assert np.isfinite(dem).any()


def test_keyerror_retry(monkeypatch):
    calls = []
    real = wc.find_contours

    def flaky(image, level, mask=None):
        calls.append(level)
        if level == 0.5:
            raise KeyError("skimage issue 4830")
        return real(image, level, mask=mask)

    monkeypatch.setattr(wc, "find_contours", flaky)
    R = np.tile(np.arange(4, dtype=np.float32), (4, 1))
    cont = wc.contours_by_level(R)
    assert 0.5 in cont                       # keyed by the nominal level
    assert calls[:2] == [0.5, 0.5 + wc.LEVEL_EPS]


def test_gap_fill_rule():
    R = np.full((9, 9), np.nan, np.float32)
    R[4, 4] = 3.0
    Rf = wc.gap_fill(R)
    # 2 iterations of the default (cross) dilation: a diamond of radius 2
    rr, cc = np.mgrid[0:9, 0:9]
    diamond = (np.abs(rr - 4) + np.abs(cc - 4)) <= 2
    assert np.array_equal(np.isfinite(Rf), diamond)
    assert np.all(Rf[diamond] == 3.0)


def test_unfiltered_uses_R_before_gap_fill():
    z, h, ndwi, clear = planar_flat(H=20, W=30, T=200, seed=4)
    clear[:, 5, 5] = False                   # one pixel never observed
    res, _ = wc.nidem_from_arrays(ndwi, clear, h)
    assert np.isnan(res["R"][5, 5]) and np.isfinite(res["R_filled"][5, 5])
    assert np.isnan(res["unfiltered"][5, 5])
    assert np.isnan(res["filtered"][5, 5])
    # R = 0 and R = 9 pixels are never in the product
    for v in (0, 9):
        sel = res["R"] == v
        assert np.all(np.isnan(res["unfiltered"][sel]))


# ─────────────────────────────────────────────────────────────────────────────
#  PIXEL MASKS (revision of 2026-09-29): ITEM PQ mask (primary), load_ard
#  (variant)
# ─────────────────────────────────────────────────────────────────────────────

def test_item_pixel_mask_maps_the_pq_flags_to_scl():
    """ITEM masks cloud, cloud shadow, band saturation and contiguity
    (Sagar et al. 2017 p.155; ITEM v2.0 step 5). Mapped to SCL: 0 (no data;
    NaN in our cubes), 1, 3, 8, 9, 10 masked; 2, 4, 5, 6, 7, 11 valid."""
    assert wc.ITEM_PQ_MASKED_SCL == (0, 1, 3, 8, 9, 10)
    scl = np.array([np.nan] + list(range(12)), np.float32)
    valid = wc.pixel_valid(scl, rule="item")
    expected = {0: False, 1: False, 2: True, 3: False, 4: True, 5: True,
                6: True, 7: True, 8: False, 9: False, 10: False, 11: True}
    assert not valid[0]                                   # NaN = no data
    assert valid[1:].tolist() == [expected[c] for c in range(12)]
    # a superset of MAREA's clear set: nothing MAREA sees is removed
    assert set(wc.CLEAR) <= {c for c in range(12) if expected[c]}
    # the bands are not looked at by the ITEM mask; a NaN band gives NaN NDWI
    g = np.array([np.nan, -999.0, 500.0], np.float32)
    n = np.array([100.0, 100.0, 500.0], np.float32)
    nd, v = wc.scene_ndwi(g, n, np.array([4.0, 4.0, 2.0]), rule="item")
    assert v.tolist() == [True, True, True]
    assert np.isnan(nd[0]) and np.isfinite(nd[1]) and nd[2] == 0.0
    # integer SCL arrays work too
    assert wc.pixel_valid(np.arange(12), rule="item").tolist() == \
        [expected[c] for c in range(12)]
    with pytest.raises(ValueError):
        wc.pixel_valid(scl, rule="marea")


def test_load_ard_pixel_mask_is_literal():
    """load_ard: erase_bad(pq_mask) with SCL {1, 3, 8, 9, 10}; SCL no data is
    NOT masked; keep_good_only(ds_data > 1) on the raw DN of every band
    (cube value = DN - 1000, DN 0 = NaN, so value > -999)."""
    assert wc.DEAFRICA_S2_BAD_SCL == (1, 3, 8, 9, 10)
    assert wc.LOAD_ARD_MIN_VALUE == -999
    scl = np.array([np.nan] + list(range(12)), np.float32)
    ok = np.full(scl.shape, 500.0, np.float32)
    v = wc.pixel_valid(scl, ok, ok, rule="load_ard")
    bad = {1, 3, 8, 9, 10}
    assert v[0]                                           # NaN SCL: not masked
    assert v[1:].tolist() == [c not in bad for c in range(12)]
    # raw DN <= 1 in either band is masked; DN 2 (value -998) is kept
    g = np.array([-999.0, 500.0, -998.0, np.nan, 500.0], np.float32)
    n = np.array([500.0, -999.0, -998.0, 500.0, np.nan], np.float32)
    s = np.full(5, 4.0, np.float32)
    assert wc.pixel_valid(s, g, n, rule="load_ard").tolist() == \
        [False, False, True, False, False]
    with pytest.raises(ValueError):
        wc.pixel_valid(s, rule="load_ard")                # needs the bands


def test_ndwi_is_unchanged_under_both_masks():
    g = np.array([[100, 0, 300, -50]], np.float32)
    n = np.array([[50, 0, 300, 150]], np.float32)
    scl = np.array([[2, 4, 11, 1]], np.float32)
    nd_i, v_i = wc.scene_ndwi(g, n, scl, rule="item")
    nd_l, v_l = wc.scene_ndwi(g, n, scl, rule="load_ard")
    assert np.array_equal(nd_i, nd_l, equal_nan=True)
    assert nd_i[0, 0] == np.float32(50) / np.float32(150)
    assert np.isnan(nd_i[0, 1]) and nd_i[0, 2] == 0.0
    assert nd_i[0, 3] == np.float32(-200) / np.float32(100)
    assert v_i.tolist() == [[True, True, True, False]]
    assert v_l.tolist() == [[True, True, True, False]]


# ─────────────────────────────────────────────────────────────────────────────
#  VARIANT "NIDEM, Sentinel-2 scene rule" (DE Africa load_ard min_gooddata)
# ─────────────────────────────────────────────────────────────────────────────

def test_s2_scene_rule_constants_are_the_reference_values():
    # Intertidal_elevation_S2 cell 15: load_ard(..., min_gooddata=0.5, ...);
    # load_ard's default categories_to_mask_s2 -> SCL codes 9, 8, 10, 3, 1
    assert wc.S2_SCENE_MIN_GOODDATA == 0.5
    assert wc.DEAFRICA_S2_BAD_SCL == (1, 3, 8, 9, 10)
    assert wc.CLEAR == (4, 5, 6, 7)
    assert wc.BOA_ADD_OFFSET == -1000 and wc.LOAD_ARD_MIN_RAW_DN_EXCLUSIVE == 1


def test_literal_good_fraction_counts_no_data_as_good():
    """load_ard: data_perc = (~pq_mask).sum / (all frame pixels); no data
    (SCL 0 there, NaN here) is not a masked category, so it counts as GOOD,
    and so do 2 and 11."""
    scl = np.full((10, 10), 9.0, np.float32)          # high cloud everywhere
    scl.ravel()[:50] = [4, 5, 6, 7, 4] * 10           # 50 of 100 clear
    assert wc.frame_good_fraction(scl) == 0.5
    scl2 = np.array([[np.nan, 0, 2, 11, 3, 1, 10, 8, 4, 6]], np.float32)
    # good: NaN, 0, 2, 11, 4, 6 -> 6 of 10
    assert wc.frame_good_fraction(scl2) == pytest.approx(0.6)
    # the superseded statistic (information only): SCL in {4,5,6,7}
    assert wc.clear_scl_fraction(scl2) == pytest.approx(0.2)
    # a scene mostly outside the swath: 70 % no data, 30 % cloud -> KEPT by
    # the literal rule (0.7), dropped by the superseded one (0.0)
    scl3 = np.full((10, 10), np.nan, np.float32)
    scl3[:3] = 9
    assert wc.frame_good_fraction(scl3) == pytest.approx(0.7)
    assert wc.clear_scl_fraction(scl3) == 0.0
    assert wc.s2_scene_rule([wc.frame_good_fraction(scl3)]).tolist() == [True]
    # a frame of 40 % no data and 60 % cloud shadow: 0.4 -> dropped
    scl4 = np.full((5, 5), 3.0, np.float32)
    scl4.ravel()[:10] = np.nan
    assert wc.frame_good_fraction(scl4) == pytest.approx(0.4)
    with pytest.raises(wc.EmptyIntervalError):
        wc.s2_scene_rule([wc.frame_good_fraction(scl4)])
    # in-memory helper on the load_ard good flag equals the frame function
    stack = np.stack([scl2.reshape(2, 5), np.full((2, 5), 8.0)])
    np.testing.assert_allclose(
        wc.good_fractions_from_arrays(wc.load_ard_good(stack)),
        [wc.frame_good_fraction(stack[0]), wc.frame_good_fraction(stack[1])])
    # the good flag looks only at the SCL (a valid pixel with an undefined
    # NDWI or a DN-1 band still counts, as in load_ard's data_perc)
    assert wc.load_ard_good(np.array([4.0, np.nan, 3.0])).tolist() == \
        [True, True, False]


def test_s2_scene_rule_drops_below_threshold_and_keeps_exactly_half():
    keep = wc.s2_scene_rule([0.0, 0.4999, 0.5, 0.51, 1.0])
    assert keep.tolist() == [False, False, True, True, True]
    with pytest.raises(wc.EmptyIntervalError):
        wc.s2_scene_rule([0.1, 0.2])
    with pytest.raises(ValueError):
        wc.s2_scene_rule([np.nan, 0.7])
    with pytest.raises(ValueError):
        wc.s2_scene_rule([1.2, 0.7])


def test_s2_scene_rule_uses_the_frame_not_the_intertidal_zone():
    """A scene clear on the whole tidal flat but overcast on the rest of the
    frame is DROPPED; a scene overcast on the flat but good on more than
    half of the frame is KEPT (load_ard divides by the whole loaded area)."""
    T, H, W = 4, 10, 10
    scl = np.full((T, H, W), 4.0, np.float32)
    flat = (slice(0, 3), slice(0, 10))                    # 30 % of the frame
    scl[1] = 9.0
    scl[1][flat] = 4.0                                    # flat only -> 0.30
    scl[2][flat] = 8.0                                    # all but flat -> 0.70
    g = wc.good_fractions_from_arrays(wc.load_ard_good(scl))
    np.testing.assert_allclose(g, [1.0, 0.3, 0.7, 1.0])
    assert wc.s2_scene_rule(g).tolist() == [True, False, True, True]


def test_s2_scene_rule_recomputes_intervals_on_retained_scenes():
    """The Ems case: the lone lowest scene is valid on 4 of 768 px. Under the
    ITEM rule the pre-flight stops. The variant drops that scene FIRST, so
    the OTR, the edges, Z_j, U_j and |S_j| are those of the retained scenes,
    and the product equals the method run on the retained scenes alone."""
    z, h, ndwi, clear = _flat_with_lone_low_scene(
        clear_patch=(slice(10, 12), slice(14, 16)))
    T = len(h)
    scl = np.where(clear, 6.0, 9.0).astype(np.float32)   # SCL of the frames
    # primary (ITEM rule): not computable
    j0, edges0, Z0, U0, sizes0 = wc.interval_assignment(h)
    cnt0, _ = wc.clear_counts_by_interval(clear.reshape(T, -1),
                                          ndwi.reshape(T, -1), j0)
    with pytest.raises(wc.NoClearObservationsInIntervalError):
        wc.interval_coverage(cnt0)
    # variant: the lone scene has a frame good fraction of 4 / 768
    g = np.array([wc.frame_good_fraction(s) for s in scl])
    assert g[-1] == pytest.approx(4 / 768) and np.all(g[:-1] == 1.0)
    keep = wc.s2_scene_rule(g)
    assert keep.tolist() == [True] * (T - 1) + [False]
    with pytest.raises(ValueError):                       # needs the fractions
        wc.nidem_from_arrays(ndwi, clear, h,
                             min_gooddata=wc.S2_SCENE_MIN_GOODDATA)
    res_v, (jv, edges_v, Zv, Uv, sizes_v) = wc.nidem_from_arrays(
        ndwi, clear, h, min_gooddata=wc.S2_SCENE_MIN_GOODDATA, good_fraction=g)
    # intervals come from the retained scenes, not from the full record
    j1, edges1, Z1, U1, sizes1 = wc.interval_assignment(h[keep])
    assert np.array_equal(jv, j1) and len(jv) == T - 1
    np.testing.assert_array_equal(edges_v, edges1)
    np.testing.assert_array_equal(Zv, Z1)
    assert np.array_equal(Uv, U1, equal_nan=True)
    np.testing.assert_array_equal(sizes_v, sizes1)
    assert edges_v[0] == h[keep].min() > h.min() == edges0[0]
    assert edges_v[-1] == pytest.approx(edges0[-1], abs=1e-12)  # same HOT
    assert not np.allclose(edges_v, edges0)
    assert sizes_v[0] > 1 and sizes0[0] == 1
    # the retained record passes the pre-flight and recovers the flat
    cnt1, _ = wc.clear_counts_by_interval(clear[keep].reshape(T - 1, -1),
                                          ndwi[keep].reshape(T - 1, -1), j1)
    assert wc.interval_coverage(cnt1)["passed"]
    ref, _ = wc.nidem_from_arrays(ndwi[keep], clear[keep], h[keep])
    for k in ("R", "filtered", "unfiltered", "conf", "uncertainty"):
        assert np.array_equal(res_v[k], ref[k], equal_nan=True), k
    d = res_v["filtered"]
    ok = np.isfinite(d)
    assert ok.mean() > 0.5
    assert np.max(np.abs(d[ok] - z[ok])) <= float(np.max(np.diff(Zv)))


def test_s2_scene_rule_off_changes_nothing():
    z, h, ndwi, clear = planar_flat(H=24, W=32, T=160, noise=0.1, cloud=0.3,
                                    seed=13)
    res0, a0 = wc.nidem_from_arrays(ndwi, clear, h)
    res1, a1 = wc.nidem_from_arrays(ndwi, clear, h, min_gooddata=None)
    for k in ("R", "filtered", "unfiltered", "conf", "conf_bounded",
              "uncertainty", "valid"):
        assert np.array_equal(res0[k], res1[k], equal_nan=True), k
    for x, y in zip(a0, a1):
        assert np.array_equal(x, y, equal_nan=True)
    # and with the rule ON but no scene below 0.5 (30 % random cloud), the
    # variant is the primary: nothing dropped, identical product
    g = wc.good_fractions_from_arrays(clear)
    assert np.all(g >= 0.5)
    res2, a2 = wc.nidem_from_arrays(ndwi, clear, h, min_gooddata=0.5,
                                    good_fraction=g)
    for k in ("R", "filtered", "unfiltered", "conf", "uncertainty"):
        assert np.array_equal(res0[k], res2[k], equal_nan=True), k
    for x, y in zip(a0, a2):
        assert np.array_equal(x, y, equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────────
#  truth-free pre-pass (scene quality and observation masks at keep)
# ─────────────────────────────────────────────────────────────────────────────

def test_packed_rows_count_like_dense_arrays():
    rng = np.random.default_rng(31)
    T, P = 23, 37
    m = rng.random((T, P)) < 0.6
    packed = wc.PackedRows(np.packbits(m, axis=1), P)
    assert packed.shape == (T, P) and len(packed) == T
    for q in range(T):
        assert np.array_equal(packed[q], m[q])
    h = rng.permutation(np.linspace(-1.0, 1.0, T))
    j, *_ = wc.interval_assignment(h)
    c1, s1 = wc.clear_counts_by_interval(packed, None, j)
    c2, s2 = wc.clear_counts_by_interval(m, None, j)
    assert np.array_equal(c1, c2) and np.array_equal(s1, s2)
    sel = rng.random(T) < 0.5
    sub = packed.subset(sel)
    assert np.array_equal(np.array([sub[q] for q in range(len(sub))]), m[sel])


@pytest.mark.parametrize("rule", ["item", "load_ard"])
def test_stream_scene_quality_equals_in_memory(tmp_path, rule):
    path, b03, b08, scl = _synthetic_cube(tmp_path, seed=21, T=12, H=9, W=11)
    T, H, W = scl.shape
    time_index = np.array([1, 4, 5, 9, 11])
    keep = np.array([0, 7, 30, 31, 60, 98])
    q = wc.stream_scene_quality(path, time_index, keep, rule=rule,
                                verbose=False)
    assert q["rule"] == rule and q["px"] == H * W and q["P"] == len(keep)
    for i, t in enumerate(time_index):
        s, g, n = scl[t].ravel(), b03[t].ravel(), b08[t].ravel()
        assert q["load_ard_good"][i] == wc.frame_good_fraction(s)
        assert q["clear_scl"][i] == wc.clear_scl_fraction(s)
        assert q["item_valid"][i] == wc.pixel_valid(s, rule="item").mean()
        assert q["no_data"][i] == np.isnan(s).mean()
        assert q["band_le_load_ard_min_px"][i] == int(((g <= -999) | (n <= -999)).sum())
        assert q["band_below_offset_px"][i] == 0
        nd, v = wc.scene_ndwi(g[keep], n[keep], s[keep], rule=rule)
        obs = v & np.isfinite(nd)
        marea = np.isin(s[keep], wc.CLEAR) & np.isfinite(nd)
        assert np.array_equal(q["obs_keep"][i], obs)
        assert np.array_equal(q["clear_keep"][i], np.isin(s[keep], wc.CLEAR))
        assert q["n_obs_keep"][i] == obs.sum()
        assert q["n_added_keep"][i] == (obs & ~marea).sum()
        assert q["n_removed_keep"][i] == (marea & ~obs).sum()
        assert q["n_both_keep"][i] == (obs & marea).sum()
    if rule == "item":
        assert q["n_removed_keep"].sum() == 0          # ITEM valid set covers C
    sub = wc.subset_scene_quality(q, np.array([True, False, True, False, True]))
    assert len(sub["load_ard_good"]) == 3 and sub["obs_keep"].shape == (3, len(keep))
    assert np.array_equal(sub["obs_keep"][1], q["obs_keep"][2])


def test_stream_scene_quality_rejects_unknown_scl(tmp_path):
    xr = pytest.importorskip("xarray")
    a = np.zeros((2, 3, 3), np.float32)
    scl = np.full((2, 3, 3), 4.0, np.float32)
    scl[1, 1, 1] = 42.0
    xr.Dataset({v: (("t", "y", "x"), x) for v, x in
                (("B03", a + 100), ("B08", a + 50), ("SCL", scl))},
               coords={"t": np.arange(2), "y": [20.0, 10.0, 0.0],
                       "x": [0.0, 10.0, 20.0]}).to_netcdf(tmp_path / "c.nc")
    with pytest.raises(ValueError):
        wc.stream_scene_quality(str(tmp_path / "c.nc"), [0, 1], [0, 4],
                                verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
#  the experiment script: modes, paths, labels, record switch, checks
# ─────────────────────────────────────────────────────────────────────────────

def test_script_outputs_are_separate_and_primary_paths_unchanged():
    v7 = pytest.importorskip("experiments.v7_waterline_contour")
    norm = os.path.normpath
    # flag off: exactly the historical primary paths
    out, csv = v7.output_paths("villaviciosa")
    assert out == "products_marea/nidem"
    assert csv == {"rtk": os.path.join("products_marea/nidem",
                                       "rtk_metrics_with_nidem.csv")}
    # Dutch sites: the products folder of the validation grid (products_<site>_10m)
    from experiments.validation_grid import dutch_products
    pw = dutch_products("wadden")
    out, csv = v7.output_paths("wadden")
    assert out == f"{pw}/nidem_eot20_g1"
    assert csv == {
        "metrics": f"{pw}/comparison_eot20_g1/vaklodingen_metrics_with_nidem.csv",
        "by_band": f"{pw}/comparison_eot20_g1/vaklodingen_by_band_with_nidem.csv"}
    # the sensitivity (MAREA's 97-scene record) has its own directory
    out, csv = v7.output_paths("villaviciosa", sensitivity=True)
    assert norm(out) == norm("products_marea/nidem/sensitivity_marea_record")
    assert norm(csv["rtk"]) == norm(
        "products_marea/nidem/sensitivity_marea_record/rtk_metrics_with_nidem.csv")
    with pytest.raises(ValueError):
        v7.output_paths("ems", sensitivity=True)
    # the variant: separate directories and CSV files
    out, csv = v7.output_paths("villaviciosa", variant=True)
    assert norm(out) == norm("products_marea/nidem/s2_scene_rule")
    assert norm(csv["rtk"]) == norm(
        "products_marea/nidem/s2_scene_rule/rtk_metrics_with_nidem_s2_scene_rule.csv")
    out, csv = v7.output_paths("ems", variant=True)
    assert out == f"{dutch_products('ems')}/nidem_eot20_g1_s2_scene_rule"
    assert csv["metrics"].endswith("comparison_eot20_g1/"
                                   "vaklodingen_metrics_with_nidem_s2_scene_rule.csv")
    assert csv["by_band"].endswith("vaklodingen_by_band_with_nidem_s2_scene_rule.csv")
    # labels per mode
    assert v7.nidem_name("NIDEM", "primary") == "NIDEM"
    assert v7.nidem_name("NIDEM", False) == "NIDEM"
    assert v7.nidem_name("NIDEM", True) == "NIDEM, Sentinel-2 scene rule"
    assert v7.nidem_name("NIDEM", "variant") == "NIDEM, Sentinel-2 scene rule"
    assert v7.nidem_name("NIDEM", "sensitivity") == "NIDEM on MAREA's 97-scene record"
    assert v7.nidem_name(v7.BOUNDED, True) == "NIDEM bounded-conf, Sentinel-2 scene rule"
    assert v7.not_computable_name("primary") == "NIDEM (ITEM rule)"
    # pixel masks per mode
    assert v7.pixel_rule("primary") == v7.pixel_rule("sensitivity") == "item"
    assert v7.pixel_rule("variant") == "load_ard"
    assert v7.run_mode() == "primary"
    assert v7.run_mode(sensitivity=True) == "sensitivity"
    assert v7.run_mode(variant=True) == "variant"
    with pytest.raises(ValueError):
        v7.run_mode(True, True)


def test_frozen_records_the_revision_with_sources():
    v7 = pytest.importorskip("experiments.v7_waterline_contour")
    fz = v7.FROZEN
    rev = fz["revision_2026_09_29_published_rules"]
    assert set(rev) >= {"A_pixel_mask", "B_variant", "C_record", "D_unchanged"}
    pq = fz["pixel_quality_mask_item"]
    assert pq["masked_scl"] == list(wc.ITEM_PQ_MASKED_SCL)
    assert pq["valid_scl"] == [2, 4, 5, 6, 7, 11]
    assert any("p.155" in s for s in pq["sources"])
    assert any("processing step 5" in s for s in pq["sources"])
    assert "465" in fz["record"] and "97-scene" in fz["record"]
    var = fz["variant_s2_scene_rule"]
    assert var["min_gooddata"] == wc.S2_SCENE_MIN_GOODDATA
    assert var["retrieved"] == "2026-09-29"
    assert "Intertidal_elevation_S2" in var["source_notebook"]
    assert "datahandling.py" in var["source_load_ard"]
    assert any("valid_data_mask" in s for s in var["quoted_load_ard"])
    assert "counts as GOOD" in var["rule"]
    assert "decision_a_primary_not_computable" in fz
    assert len(fz["history"]["superseded_2026_09_29"]) == 3


def test_script_refuses_the_superseded_flag_and_bad_combinations():
    v7 = pytest.importorskip("experiments.v7_waterline_contour")
    with pytest.raises(SystemExit):
        v7.main(["villaviciosa", "--sensitivity-all-scenes"])
    with pytest.raises(SystemExit):
        v7.main(["ems", "--sensitivity-marea-record"])
    with pytest.raises(SystemExit):
        v7.main(["villaviciosa", "--sensitivity-marea-record", "--s2-scene-rule"])


def test_villaviciosa_record_switch():
    """Revision C: the Villaviciosa PRIMARY record is ITEM's all-observations
    record (465 scenes of 2023-2025 with an overpass time and a finite EOT20
    level); MAREA's 97-scene record is the labelled sensitivity, and it is a
    subset of the primary with identical levels."""
    v7 = pytest.importorskip("experiments.v7_waterline_contour")
    cfg = v7.SITES["villaviciosa"]
    needed = [cfg["extract"], cfg["overpass"], v7.VVA_NOTEBOOK, "tide_models"]
    if not all(os.path.exists(p) for p in needed):
        pytest.skip("Villaviciosa inputs not available")
    z = np.load(cfg["extract"])
    ext_dates = np.array([str(d) for d in z["dates"]])
    prim = v7.build_record("villaviciosa", cfg, ext_dates)
    sens = v7.build_record("villaviciosa", cfg, ext_dates, marea_record=True)
    assert len(prim["h"]) == v7.VVA_ALL_SCENES == 465
    assert len(sens["h"]) == v7.VVA_SCORED_SCENES == 97
    assert prim["rule"].startswith("PRIMARY") and "all-observations" in prim["rule"]
    assert sens["rule"].startswith(v7.SENSITIVITY_LEAD)
    assert np.all((prim["dates"] >= "2023-01-01") & (prim["dates"] <= "2025-12-31"))
    assert np.all(np.isfinite(prim["h"]))
    pos = {int(t): i for i, t in enumerate(prim["time_index"])}
    assert all(int(t) in pos for t in sens["time_index"])
    idx = np.array([pos[int(t)] for t in sens["time_index"]])
    np.testing.assert_allclose(prim["h"][idx], sens["h"], atol=1e-9)
    with pytest.raises(ValueError):
        v7.build_record("ems", v7.SITES["ems"], ext_dates, marea_record=True)


def test_extraction_check_compares_ndwi_and_counts_mask_changes():
    v7 = pytest.importorskip("experiments.v7_waterline_contour")
    rng = np.random.default_rng(41)
    T, H, W = 3, 4, 5
    scl = rng.choice(np.arange(12), (T, H, W)).astype(np.float32)
    scl[0, 0, 0] = np.nan
    g = rng.integers(1, 900, (T, H, W)).astype(np.float32)
    n = rng.integers(1, 900, (T, H, W)).astype(np.float32)
    g[1, 1, 1] = n[1, 1, 1] = 0.0                       # NaN NDWI
    keep = np.array([0, 3, 6, 7, 12, 19])
    time_index = np.array([10, 11, 12])
    nd, _ = wc.scene_ndwi(g, n, scl)
    Y = nd.reshape(T, -1)[:, keep]
    C = np.isin(scl, wc.CLEAR).reshape(T, -1)[:, keep]
    chk = v7.ExtractionCheck(Y, C, time_index, keep, W)
    for q_, t in enumerate(time_index):
        for r0, r1 in ((0, 2), (2, 4)):
            ndb, vb = wc.scene_ndwi(g[q_, r0:r1], n[q_, r0:r1], scl[q_, r0:r1])
            chk(t, r0, r1, ndb, vb, scl[q_, r0:r1])
    s = chk.summary()
    v = wc.pixel_valid(scl).reshape(T, -1)[:, keep]
    fin = np.isfinite(Y)
    assert s["scene_px_compared (SCL->C and NDWI, every keep px)"] == T * len(keep)
    assert s["ndwi_values_compared_where_both_masks_valid"] == int((v & C).sum())
    assert s["observations_added_vs_marea_C"] == int((v & fin & ~(C & fin)).sum())
    assert s["observations_removed_vs_marea_C"] == 0
    # a changed NDWI or SCL is caught
    bad = Y.copy()
    assert np.isnan(bad[1, 2])                           # the zero-sum pixel
    assert np.isfinite(bad[1, 3])
    bad[1, 3] += 0.5
    chk2 = v7.ExtractionCheck(bad, C, time_index, keep, W)
    with pytest.raises(AssertionError):
        for q_, t in enumerate(time_index):
            ndb, vb = wc.scene_ndwi(g[q_], n[q_], scl[q_])
            chk2(t, 0, H, ndb, vb, scl[q_])
    Cb = C.copy()
    Cb[2, 0] = ~Cb[2, 0]
    chk3 = v7.ExtractionCheck(Y, Cb, time_index, keep, W)
    with pytest.raises(AssertionError):
        for q_, t in enumerate(time_index):
            ndb, vb = wc.scene_ndwi(g[q_], n[q_], scl[q_])
            chk3(t, 0, H, ndb, vb, scl[q_])


def test_not_computable_reason_names_the_scene_and_counts():
    v7 = pytest.importorskip("experiments.v7_waterline_contour")
    share = 27 / 477745
    pre = {v7.COV_KEY: {
               "px": 477745, "expected_valid_share": share,
               "expected_valid_px": 27, "min_valid_share": 0.01},
           "pixel_mask": {"rule": "item"},
           "interval_sizes |S_j|": [1, 29, 59, 50, 46, 48, 60, 86, 83],
           "scenes_of_intervals_below_floor": {"1": [
               {"date": "2024-12-10", "valid_finite_share_on_keep": share}]}}
    r = v7.not_computable_reason(pre, "PRIMARY (ITEM rule)", "x.json")
    assert r.startswith("PRIMARY (ITEM rule): NOT COMPUTABLE")
    assert ("interval 1 holds 1 scene(s) (2024-12-10 valid on 27 of 477,745 "
            "keep px)") in r
    assert "every one of the 9 tide intervals" in r
    assert "pixel mask 'item'" in r


def test_refuse_overwrite_protects_only_current_computed_results(tmp_path):
    import json
    v7 = pytest.importorskip("experiments.v7_waterline_contour")
    j = tmp_path / "nidem_diagnostics.json"
    c = tmp_path / "metrics.csv"
    c.write_text("product,n\nNIDEM,100\n", encoding="utf-8")
    v7._refuse_overwrite(str(j), [str(c)])               # nothing there yet
    j.write_text(json.dumps({"status": "passed",
                             "frozen_constants": {"intervals": "x"}}),
                 encoding="utf-8")
    v7._refuse_overwrite(str(j), [str(c)])               # superseded rules
    j.write_text(json.dumps({"status": "NOT COMPUTABLE",
                             "frozen_constants": v7.FROZEN}), encoding="utf-8")
    v7._refuse_overwrite(str(j), [str(c)])
    j.write_text(json.dumps({"status": "passed",
                             "frozen_constants": v7.FROZEN}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        v7._refuse_overwrite(str(j), [str(c)])
