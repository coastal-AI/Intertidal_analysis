# -*- coding: utf-8 -*-
"""V14, the per-scene water threshold (experiments/variants/otsu.py, the addendum of 2026-10-28). Otsu's threshold
against a brute-force two-class variance search written with plain loops, on bimodal, unimodal (every flat exposed)
and all-water synthetic scenes and on small sets with tied values and values on the bin edges; the point of the bin
returned; the scene rule (clear pixels with a defined NDWI; fewer than 2, no split or outside [-0.2, 0.3]: t_s = 0);
and the two routes of the factor, the competitors' NDWI - t_s > 0 and the explicit wet matrix C & (NDWI > t_s),
equal bit for bit."""
import inspect
import os
import sys
import types
from fractions import Fraction

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.fixedcfg import elev_base as base  # noqa: E402
from experiments.fixedcfg import elev_bracket, elev_freq, elev_logitide  # noqa: E402
from experiments.variants import otsu  # noqa: E402

EDGES = np.linspace(-1.0, 1.0, 257)          # the default binning: multiples of 1/128


# ─────────────────────────────────────────────────────────────────────────────
#  the references
# ─────────────────────────────────────────────────────────────────────────────

def brute_force_otsu(values, bins=256, lo=-1.0, hi=1.0):
    """Otsu by plain loops (the check of otsu_threshold). Every inner edge e_{k+1} is a candidate threshold t and
    splits the values by the rule itself, v > t; each value stands for the centre of its bin, found by scanning the
    edges (e_k < v <= e_{k+1}; below lo the first bin, above hi the last); the two-class criterion is the within-class
    sum of squares, in exact rational arithmetic, minimised (the total is fixed, so the between-class variance is
    maximised); the first minimum wins. NaN when no candidate leaves two non-empty classes."""
    edges = np.linspace(lo, hi, bins + 1)
    vals = [float(v) for v in np.ravel(values) if np.isfinite(v)]
    centre = []                                 # 2k + 1: the bin centre, in units of (hi - lo)/(2 bins) above lo
    for v in vals:
        k = 0
        while k < bins - 1 and v > edges[k + 1]:
            k += 1
        centre.append(2 * k + 1)
    best, best_t = None, np.nan
    for k in range(bins - 1):
        t = edges[k + 1]
        n, s, q = [0, 0], [0, 0], [0, 0]
        for v, c in zip(vals, centre):
            j = 1 if v > t else 0
            n[j] += 1
            s[j] += c
            q[j] += c * c
        if n[0] == 0 or n[1] == 0:
            continue
        within = Fraction(n[0] * q[0] - s[0] * s[0], n[0]) + Fraction(n[1] * q[1] - s[1] * s[1], n[1])
        if best is None or within < best:
            best, best_t = within, float(t)
    return best_t


def bin_by_scan(v, edges=EDGES):
    """The bin of one value by scanning the edges (bins closed on the right, the end bins taking the outside)."""
    k = 0
    while k < len(edges) - 2 and v > edges[k + 1]:
        k += 1
    return k


def same(a, b):
    return a == b or (np.isnan(a) and np.isnan(b))


def scene(kind, n, rng):
    """NDWI of one synthetic scene, float32 as the extraction: 'bimodal' (exposed flats at -0.12, water at 0.38, a
    few mixed pixels between), 'dry' (every flat exposed: one mode at -0.45), 'water' (all water at high tide: one
    mode at 0.55), 'turbid' (one mode at 0.05, inside [-0.2, 0.3])."""
    if kind == "bimodal":
        a, b = n * 60 // 100, n * 37 // 100
        v = np.concatenate([rng.normal(-0.12, 0.05, a), rng.normal(0.38, 0.09, b), rng.uniform(-0.1, 0.35, n - a - b)])
        return rng.permutation(v).astype(np.float32)
    mean, sd = {"dry": (-0.45, 0.06), "water": (0.55, 0.08), "turbid": (0.05, 0.04)}[kind]
    return rng.normal(mean, sd, n).astype(np.float32)


def small_ndwi(rng, m):
    """m NDWI values of small integer reflectances: many ties, many values on the edges, NaN where B03 + B08 = 0."""
    g, n = rng.integers(0, 21, m), rng.integers(0, 21, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(g + n != 0, (g - n) / np.where(g + n != 0, g + n, 1), np.nan).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  otsu_threshold
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["bimodal", "dry", "water", "turbid"])
def test_otsu_is_the_brute_force_maximum_on_synthetic_scenes(kind):
    v = scene(kind, 3000, np.random.default_rng(14))
    assert otsu.otsu_threshold(v) == brute_force_otsu(v)


def test_otsu_is_the_brute_force_maximum_on_small_sets_with_ties_and_edges():
    rng = np.random.default_rng(1979)
    for _ in range(300):
        v = small_ndwi(rng, int(rng.integers(0, 40)))
        assert same(otsu.otsu_threshold(v), brute_force_otsu(v)), v


def test_two_modes_split_in_the_valley_and_one_mode_near_its_mean():
    rng = np.random.default_rng(14)
    t = otsu.otsu_threshold(scene("bimodal", 3000, rng))
    assert 0.0 < t < 0.25                       # between the flats (-0.12) and the water (0.38)
    for kind, mean, sd in (("dry", -0.45, 0.06), ("water", 0.55, 0.08), ("turbid", 0.05, 0.04)):
        assert abs(otsu.otsu_threshold(scene(kind, 3000, rng)) - mean) < sd / 2


def test_the_threshold_is_the_upper_edge_of_the_last_dry_bin():
    """NDWI > t is the split's wet class value by value, and t is the upper edge of the highest dry bin."""
    rng = np.random.default_rng(7)
    on_edges = EDGES[rng.integers(100, 170, 400)]                 # values exactly on the edges
    for v in (scene("bimodal", 2000, rng), np.concatenate([scene("bimodal", 1500, rng), on_edges])):
        t = otsu.otsu_threshold(v)
        k_star = int(np.flatnonzero(EDGES == t)[0]) - 1           # t is an edge: the split after bin k_star
        k = np.array([bin_by_scan(float(x)) for x in v])
        assert np.array_equal(v > t, k > k_star)
        assert k[v <= t].max() == k_star                         # the last dry bin is occupied: t is its upper edge


def test_zero_is_an_edge_and_an_ndwi_of_exactly_zero_is_dry():
    v = np.array([0.0] * 40 + [-0.003] * 10 + [0.4] * 30)       # 0 and -0.003 share the bin (-1/128, 0]
    t = otsu.otsu_threshold(v)
    assert t == 0.0 and t == EDGES[128]
    assert not (v[:40] > t).any() and (v[50:] > t).all()        # as NDWI > 0 has it


def test_a_gap_of_empty_bins_takes_the_lowest_split():
    v = np.array([-0.3] * 30 + [0.4] * 30)                       # -0.3 is in bin 89, (-0.3046875, -0.296875]
    t = otsu.otsu_threshold(v)
    assert t == -1 + 90 / 128 == brute_force_otsu(v)
    gap = EDGES[(EDGES >= t) & (EDGES < 0.4)]                    # every split inside the gap ...
    assert len(gap) > 50 and all(np.array_equal(v > e, v > t) for e in gap)   # ... classifies the pixels alike


def test_fewer_than_two_values_or_one_bin_give_no_threshold():
    for v in ([], [0.2], [0.2, 0.2], [0.2, 0.201], [np.nan, np.nan, 0.1], [np.inf, -np.inf, 0.3], [np.nan] * 5,
              np.full((3, 4), -0.05)):
        assert np.isnan(otsu.otsu_threshold(v)), v
        assert np.isnan(brute_force_otsu(v)), v
    assert otsu.otsu_threshold([-0.1, 0.2]) == -1 + 116 / 128   # two values in different bins: the lower one's edge


def test_values_outside_the_range_count_in_the_end_bins():
    """An NDWI beyond [-1, 1] (a negative reflectance) is a defined NDWI: it counts in the first or last bin."""
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(-0.1, 0.05, 1000), np.full(1500, 1.2), rng.normal(0.4, 0.05, 300),
                        [-1.7, -3.0, 2.5]]).astype(np.float32)
    t = otsu.otsu_threshold(v)
    assert t == otsu.otsu_threshold(np.clip(v, -1, 1)) == brute_force_otsu(v)
    assert t != otsu.otsu_threshold(v[np.abs(v) <= 1])          # left out, they would change the split here
    assert otsu.otsu_threshold([-1.0, -1.0, 1.0]) == EDGES[1]   # lo lies in the first bin, hi in the last


def test_order_and_float32_do_not_change_the_threshold():
    rng = np.random.default_rng(11)
    v32 = scene("bimodal", 2000, rng)
    t = otsu.otsu_threshold(v32)
    assert t == otsu.otsu_threshold(v32.astype(np.float64)) == otsu.otsu_threshold(rng.permutation(v32))
    assert t == otsu.otsu_threshold(v32.reshape(40, 50))


def test_other_binnings_match_the_brute_force():
    rng = np.random.default_rng(5)
    v = scene("bimodal", 800, rng)
    for bins, lo, hi in ((64, -1.0, 1.0), (100, -0.5, 0.8), (2, -1.0, 1.0)):
        assert otsu.otsu_threshold(v, bins, lo, hi) == brute_force_otsu(v, bins, lo, hi)


def test_a_bad_binning_is_an_error():
    for bins, lo, hi in ((1, -1.0, 1.0), (2.5, -1.0, 1.0), (256, 1.0, 1.0), (256, 1.0, -1.0), (256, np.nan, 1.0)):
        with pytest.raises(ValueError):
            otsu.otsu_threshold([-0.5, 0.5], bins, lo, hi)


def test_the_defaults_are_the_addendums():
    sig = inspect.signature(otsu.otsu_threshold).parameters
    assert (sig["bins"].default, sig["lo"].default, sig["hi"].default) == (256, -1.0, 1.0)
    sig = inspect.signature(otsu.scene_thresholds).parameters
    assert (sig["bins"].default, sig["lo"].default, sig["hi"].default) == (256, -1.0, 1.0)
    assert sig["valid"].default == (-0.2, 0.3) and sig["fallback"].default == 0.0
    assert otsu.MIN_PIXELS == 2


# ─────────────────────────────────────────────────────────────────────────────
#  scene_thresholds
# ─────────────────────────────────────────────────────────────────────────────

def extraction(rng, P=2500):
    """Nine synthetic scenes x P extraction pixels (float32 NDWI with NaN, ~85 % clear), one case per scene."""
    kinds = ["bimodal", "water", "dry", "turbid", "bimodal", "bimodal", "bimodal", "bimodal", None]
    Y = np.stack([scene(k, P, rng) if k else np.zeros(P, np.float32) for k in kinds])
    C = rng.random(Y.shape) < 0.85
    Y[rng.random(Y.shape) < 0.01] = np.nan                       # a few undefined NDWI
    C[4] = False
    C[4, 17] = True                                              # one clear pixel
    C[5] = False                                                 # no clear pixel
    Y[6, C[6]] = np.nan                                          # every clear NDWI undefined
    C[7] = False
    C[7, [3, 9]], Y[7, [3, 9]] = True, 0.125                     # two clear pixels, one NDWI: no split
    dry, wet = scene("dry", P, rng), scene("water", P, rng)      # flats and water, but the water is all cloudy
    C[8] = rng.random(P) < 0.5
    Y[8] = np.where(C[8], dry, wet)
    return Y, C


def test_scene_thresholds_is_otsu_of_the_clear_defined_pixels_or_the_fallback():
    Y, C = extraction(np.random.default_rng(20261028))
    t, used = otsu.scene_thresholds(Y, C)
    assert t.dtype == np.float64 and used.dtype == bool and t.shape == used.shape == (9,)
    for s in range(9):
        v = Y[s][C[s] & np.isfinite(Y[s])]
        ref = brute_force_otsu(v) if len(v) >= 2 else np.nan
        inside = -0.2 <= ref <= 0.3
        assert used[s] == (not inside) and t[s] == (ref if inside else 0.0), s
    # bimodal kept; all water, every flat exposed, 1 / 0 clear pixels, no defined NDWI, one value and the scene whose
    # water is cloudy (its clear pixels are the flats alone) fall back; one mode inside [-0.2, 0.3] is kept (the
    # addendum's rule tests the range, not the bimodality)
    assert used.tolist() == [False, True, True, False, True, True, True, True, True]


def test_only_the_clear_pixels_with_a_defined_ndwi_count():
    rng = np.random.default_rng(2)
    Y, C = extraction(rng)
    t, used = otsu.scene_thresholds(Y, C)
    for fill in (np.nan, 0.9, -0.9):
        Y2 = Y.copy()
        Y2[~C] = fill                                            # what the cloudy looks hold does not matter
        t2, used2 = otsu.scene_thresholds(Y2, C)
        assert np.array_equal(t2, t) and np.array_equal(used2, used)
    Y3 = Y.copy()
    Y3[~C] = rng.uniform(-1, 1, int((~C).sum()))
    assert np.array_equal(otsu.scene_thresholds(Y3, C)[0], t)


def test_the_valid_range_is_closed_and_the_fallback_is_the_one_given():
    Y, C = extraction(np.random.default_rng(4))
    t, used = otsu.scene_thresholds(Y, C)
    t0 = t[0]
    assert not used[0] and -0.2 <= t0 <= 0.3
    t_, used_ = otsu.scene_thresholds(Y[:1], C[:1], valid=(t0, t0))       # both ends belong to the range
    assert t_[0] == t0 and not used_[0]
    t1, used1 = otsu.scene_thresholds(Y[:1], C[:1], valid=(np.nextafter(t0, np.inf), 1.0))
    assert used1[0] and t1[0] == 0.0
    t2, used2 = otsu.scene_thresholds(Y, C, fallback=0.05)
    assert np.array_equal(used2, used) and np.all(t2[used] == 0.05) and np.array_equal(t2[~used], t[~used])


def test_a_float32_extraction_gives_the_float64_result():
    Y, C = extraction(np.random.default_rng(8))
    a, b = otsu.scene_thresholds(Y, C), otsu.scene_thresholds(Y.astype(np.float64), C)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = otsu.scene_thresholds(Y, C.astype(np.float64))          # a 0/1 clear mask, as _fit_block takes it
    assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1])


class Scenes:
    """An extraction that hands out one scene at a time and refuses to be read whole (the Wadden's ~150 scenes x
    1.2 million pixels are not copied)."""

    def __init__(self, a):
        self.a, self.shape, self.ndim, self.read = a, a.shape, a.ndim, []

    def __getitem__(self, s):
        self.read.append(s)
        return self.a[s]

    def __array__(self, *args, **kwargs):
        raise AssertionError("the whole extraction was read")


def test_scene_thresholds_reads_one_scene_at_a_time():
    Y, C = extraction(np.random.default_rng(6))
    Ys, Cs = Scenes(Y), Scenes(C)
    t, used = otsu.scene_thresholds(Ys, Cs)
    ref = otsu.scene_thresholds(Y, C)
    assert np.array_equal(t, ref[0]) and np.array_equal(used, ref[1])
    assert Ys.read == Cs.read == list(range(Y.shape[0]))


def test_scene_thresholds_rejects_misshaped_inputs():
    Y = np.zeros((3, 10), np.float32)
    for C in (np.ones((3, 9), bool), np.ones((2, 10), bool), np.ones(10, bool)):
        with pytest.raises(ValueError):
            otsu.scene_thresholds(Y, C)
    with pytest.raises(ValueError):
        otsu.scene_thresholds(Y[0], np.ones(10, bool))
    with pytest.raises(ValueError):
        otsu.scene_thresholds(Y, np.ones((3, 10), bool), bins=1)


# ─────────────────────────────────────────────────────────────────────────────
#  shift_for_threshold and wet_from_threshold
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = np.array([-0.1953125, 0.0, 0.296875, 0.1, -0.15, 1 / 3])    # edges, 0, and floats that are not


def adversarial(rng, t, P=4000, dtype=np.float32):
    """Looks x pixels of NDWI around each look's threshold: random values, the threshold itself and its neighbours
    in the array's precision, signed zeros, the smallest subnormal, infinities and NaN."""
    Y = rng.uniform(-1.2, 1.2, (len(t), P)).astype(dtype)
    for s, ts in enumerate(t):
        x = dtype(ts)
        special = [x, np.nextafter(x, dtype(np.inf)), np.nextafter(x, dtype(-np.inf)), 0.0, -0.0,
                   np.finfo(dtype).smallest_subnormal, -np.finfo(dtype).smallest_subnormal, np.inf, -np.inf, np.nan,
                   np.finfo(dtype).max, ts]
        Y[s, :len(special) * 20] = np.repeat(np.array(special, dtype), 20)
    C = rng.random(Y.shape) < 0.8
    return Y, C


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_shift_then_ndwi_above_0_is_ndwi_above_t(dtype):
    rng = np.random.default_rng(14)
    Y, C = adversarial(rng, THRESHOLDS, dtype=dtype)
    wet = otsu.wet_from_threshold(Y, C, THRESHOLDS)
    assert wet.dtype == bool and wet.shape == Y.shape
    assert np.array_equal(C & (otsu.shift_for_threshold(Y, THRESHOLDS) > 0), wet)
    for s in range(Y.shape[0]):                                  # plain loops on the first 300 pixels
        for p in range(300):
            assert wet[s, p] == (bool(C[s, p]) and float(Y[s, p]) > float(THRESHOLDS[s])), (s, p)
    t, _ = otsu.scene_thresholds(*extraction(rng, P=Y.shape[1]))  # thresholds as V14 makes them
    t = t[:Y.shape[0]]
    assert np.array_equal(C & (otsu.shift_for_threshold(Y, t) > 0), otsu.wet_from_threshold(Y, C, t))


def test_the_shift_keeps_nan_and_finite_values():
    Y, C = adversarial(np.random.default_rng(3), THRESHOLDS)
    Z = otsu.shift_for_threshold(Y, THRESHOLDS)
    assert Z.dtype == np.float64 and Z.shape == Y.shape
    assert np.array_equal(np.isnan(Z), np.isnan(Y)) and np.array_equal(np.isfinite(Z), np.isfinite(Y))
    fin = np.isfinite(Y)
    assert np.array_equal(Z[fin], (Y.astype(np.float64) - THRESHOLDS[:, None])[fin])


def test_a_threshold_of_zero_is_the_common_configuration():
    Y, C = adversarial(np.random.default_rng(9), np.zeros(5))
    with np.errstate(invalid="ignore"):
        common = C & (Y > 0)                                     # W of the common configuration
    for t in (np.zeros(5), 0.0):
        Z = otsu.shift_for_threshold(Y, t)
        assert np.array_equal(Z, Y.astype(np.float64), equal_nan=True)
        assert np.array_equal(np.signbit(Z), np.signbit(Y))      # -0.0 stays -0.0: Y itself, bit for bit
        assert np.array_equal(otsu.wet_from_threshold(Y, C, t), common)


def test_a_clear_look_with_an_undefined_ndwi_is_dry():
    Y = np.array([[np.nan, 0.5, np.nan], [0.2, np.nan, -0.4]], np.float32)
    C = np.array([[True, True, False], [True, True, True]])
    assert otsu.wet_from_threshold(Y, C, [0.1, -0.5]).tolist() == [[False, True, False], [True, False, True]]


def test_bad_thresholds_are_an_error():
    Y, C = np.zeros((3, 4), np.float32), np.ones((3, 4), bool)
    for t in ([0.0, 0.1], [0.0, np.nan, 0.1], [0.0, np.inf, 0.1], np.zeros((3, 1))):
        with pytest.raises(ValueError):
            otsu.shift_for_threshold(Y, t)
        with pytest.raises(ValueError):
            otsu.wet_from_threshold(Y, C, t)
    with pytest.raises(ValueError):
        otsu.shift_for_threshold(Y[0], 0.0)
    with pytest.raises(ValueError):
        otsu.wet_from_threshold(Y, C[:, :3], [0.0, 0.1, 0.2])


def test_the_competitors_read_the_shifted_input_as_ndwi_above_t():
    """LT2D (E4), sampled exceedance (E5c) and the wet/dry bracket (E12), their code unchanged: on NDWI - t_s they
    pass the common configuration's check (W = C & (Y > 0)), their valid looks are the clear looks with a defined
    NDWI and their wet looks are C & (NDWI > t_s); their elevations are those of any input with the same valid and
    wet looks."""
    rng = np.random.default_rng(25)
    T, P = 60, 300
    h = rng.uniform(-2.0, 2.0, T)
    z = rng.uniform(-1.5, 1.5, P)
    flooded = h[:, None] > z[None, :]
    Y = np.where(flooded, rng.normal(0.15, 0.10, (T, P)), rng.normal(-0.05, 0.05, (T, P))).astype(np.float32)
    C = rng.random((T, P)) < 0.85
    Y[rng.random((T, P)) < 0.02] = np.nan
    t = rng.choice(EDGES[(EDGES >= -0.2) & (EDGES <= 0.3)], T)  # one admissible Otsu threshold per look
    wet = otsu.wet_from_threshold(Y, C, t)

    def obs(Yi):
        with np.errstate(invalid="ignore"):
            W = C & (Yi > 0)
        return types.SimpleNamespace(Y=Yi, C=C, W=W, h=h, t=None, keep=np.arange(P), shape=(1, P), pixel_m=10.0,
                                     site="synthetic")

    shifted = obs(otsu.shift_for_threshold(Y, t))
    coded = obs(np.where(np.isfinite(Y), np.where(wet, 0.5, -0.5), np.nan).astype(np.float32))   # the same looks
    base.check_obs(shifted)
    V, Wb = base.valid_and_wet(shifted.Y, shifted.C, shifted.W, slice(0, P))
    assert np.array_equal(V, C & np.isfinite(Y)) and np.array_equal(Wb, wet & np.isfinite(Y))
    assert (wet != (C & (np.nan_to_num(Y, nan=0.0) > 0))).any()  # the thresholds change some looks here
    for run in (elev_logitide.run, elev_freq.run_e5c, elev_bracket.run):
        a, b = run(shifted, {}), run(coded, {})
        assert a["meta"]["status"] == "ok" and np.isfinite(a["z"]).any()
        assert np.array_equal(a["z"], b["z"], equal_nan=True), a["meta"]["method"]
