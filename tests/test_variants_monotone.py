# -*- coding: utf-8 -*-
"""experiments/variants/monotone.py (V11 MAREA-M) on synthetic pixels: hand-worked answers; the pixel with a perfect
step; a plain pool-adjacent-violators reference, in exact fractions, on thousands of random pixels (ties at p = 0.5
exactly, looks at equal levels, censored and rejected pixels); the level-set property the vectorised read-off rests
on; and the limits: an undefined NDWI, all wet, all dry, fewer than 8 looks, tied levels, the explicit wet matrix of
V14, the blocks of pixels and the order of the looks.

Run:  python -m pytest -q tests/test_variants_monotone.py
"""
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.variants.monotone import invert_monotone  # noqa: E402

HALF = Fraction(1, 2)


# ── The reference: the addendum's definition, by plain loops ────────────────

def pava(y, n):
    """Pool-adjacent-violators by plain loops: the non-decreasing least-squares fit to the means y[i]/n[i] with the
    weights n[i], in exact fractions. Returns the fitted value at every position."""
    blocks = []                                      # [sum, weight, positions] of each pooled block, in order
    for yi, ni in zip(y, n):
        blocks.append([yi, ni, 1])
        while len(blocks) > 1 and Fraction(blocks[-2][0], blocks[-2][1]) > Fraction(blocks[-1][0], blocks[-1][1]):
            s, m, k = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += m
            blocks[-1][2] += k
    fitted = []
    for s, m, k in blocks:
        fitted += [Fraction(s, m)] * k
    return fitted


def wet_probability(wet, clear, h):
    """The pixel's distinct levels among its clear looks, ascending, and its isotonic wet probability at each: the
    looks at equal h pooled first (wet looks and looks per level), then pool-adjacent-violators."""
    levels, y, n = [], [], []
    for t in sorted(range(len(h)), key=lambda t: h[t]):
        if not clear[t]:
            continue
        if levels and levels[-1] == h[t]:
            y[-1] += int(wet[t])
            n[-1] += 1
        else:
            levels.append(float(h[t]))
            y.append(int(wet[t]))
            n.append(1)
    return levels, pava(y, n)


def reference(wet, clear, h, min_looks=8):
    """V11 of one pixel by plain loops: (z, censored, p at h_hi). h_hi = the lowest level with p >= 0.5, h_lo = the
    highest level below it, z = (h_lo + h_hi)/2; every level p >= 0.5: z = the lowest level, censored -1; none:
    z = the highest level, censored +1; NaN with fewer than min_looks clear looks or without a wet and a dry one."""
    wet = [bool(w) and bool(c) for w, c in zip(wet, clear)]
    n_clear, n_wet = sum(bool(c) for c in clear), sum(wet)
    if n_clear < min_looks or n_wet == 0 or n_wet == n_clear:
        return np.nan, 0, None
    levels, p = wet_probability(wet, clear, h)
    at_or_above = [i for i in range(len(levels)) if p[i] >= HALF]
    if not at_or_above:
        return levels[-1], 1, None
    i = at_or_above[0]
    if i == 0:
        return levels[0], -1, p[0]
    return (levels[i - 1] + levels[i]) / 2, 0, p[i]


def reference_all(wet, clear, h, min_looks=8):
    """reference() pixel by pixel: (z, censored, p at h_hi) of every column."""
    out = [reference(wet[:, j], clear[:, j], h, min_looks) for j in range(clear.shape[1])]
    return np.array([o[0] for o in out], float), np.array([o[1] for o in out], int), [o[2] for o in out]


def random_window(rng, T, P, decimals):
    """T looks and P pixels. The levels are rounded to `decimals` (few decimals: many looks share a level); the clear
    rate is drawn per pixel; five kinds of pixel, by column: a noisy transition (logistic in h), pure noise
    (P(wet) = 0.5, so ties at p = 0.5 exactly), the transition upside down (wet at low tide: censored pixels), a
    perfect step, and nearly always wet or always dry. NDWI: in (0, 0.8] on the wet looks, in [-0.8, 0] on the dry
    ones with some exactly 0 or undefined (NaN), anything on the looks that are not clear. Returns (h, C, wet, Y)
    with wet the wet clear looks."""
    h = np.round(rng.uniform(-2.5, 2.5, T), decimals)
    clear = rng.random((T, P)) < rng.uniform(0.2, 1.0, P)
    z0, width = rng.uniform(-2.5, 2.5, P), rng.uniform(0.02, 1.5, P)
    pw = 1.0 / (1.0 + np.exp(-(h[:, None] - z0) / width))
    kind = np.arange(P) % 5
    pw[:, kind == 1] = 0.5
    pw[:, kind == 2] = 1.0 - pw[:, kind == 2]
    pw[:, kind == 3] = (h[:, None] > z0)[:, kind == 3]
    pw[:, kind == 4] = rng.choice([0.03, 0.97], int((kind == 4).sum()))
    wet = rng.random((T, P)) < pw
    Y = np.where(wet, rng.uniform(1e-6, 0.8, (T, P)), rng.uniform(-0.8, 0.0, (T, P)))
    Y[~wet & (rng.random((T, P)) < 0.05)] = 0.0       # NDWI exactly 0: dry
    Y[~wet & (rng.random((T, P)) < 0.05)] = np.nan    # undefined NDWI on a clear look: dry
    off = ~clear
    Y[off] = np.where(rng.random(int(off.sum())) < 0.5, rng.uniform(-0.8, 0.8, int(off.sum())), np.nan)
    return h, clear, wet & clear, Y.astype(np.float32)


def same(a, b):
    return all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


H10 = np.arange(10) * 0.1          # ten clear looks at the levels 0.0, 0.1, ..., 0.9


def one(wet, h=H10, clear=None):
    """invert_monotone on one pixel given its wet looks: (z, censored)."""
    wet = np.asarray(wet, bool)[:, None]
    clear = np.ones_like(wet) if clear is None else np.asarray(clear, bool)[:, None]
    z, censored, _, _ = invert_monotone(None, clear, h, wet=wet)
    return z[0], censored[0]


# ── Known answers ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wet, z, censored", [
    ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], (H10[4] + H10[5]) / 2, 0),   # separated: between 0.4 and 0.5
    ([0, 0, 0, 1, 0, 1, 1, 1, 1, 1], (H10[2] + H10[3]) / 2, 0),   # 0.3 wet, 0.4 dry pooled to p = 0.5 exactly: h_hi 0.3
    ([0, 0, 1, 0, 0, 1, 1, 1, 1, 1], (H10[4] + H10[5]) / 2, 0),   # 0.2-0.4 pooled to p = 1/3: h_hi 0.5
    ([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], (H10[0] + H10[1]) / 2, 0),   # alternating: p = 0, 0.5 (x 8), 1: h_hi 0.1
    ([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], (H10[7] + H10[8]) / 2, 0),   # 0.8-0.9 pooled to p = 0.5 at the top: h_hi 0.8
    ([1, 0, 1, 1, 1, 1, 1, 1, 1, 1], H10[0], -1),                 # 0.0-0.1 pooled to 0.5: p >= 0.5 at every level
    ([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], H10[9], +1),                 # 0.7-0.9 pooled to 1/3: p >= 0.5 at no level
    ([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], H10[9], +1),                 # wet low, dry high: one pool, p = 0.2
    ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], H10[0], -1),                 # ... one pool, p = 0.6
])
def test_hand_worked_pixels(wet, z, censored):
    assert one(wet) == (z, censored)
    assert reference(wet, [1] * 10, H10)[:2] == (z, censored)


def test_looks_at_equal_levels_are_pooled_first():
    # at 0.1 a wet look and two dry ones: pooled, p = 1/3 there, so h_hi = 0.2 and z = 0.15; ordering the tie by
    # wetness instead (dry, dry, wet) would give the wet look p = 1, h_hi = 0.1 and z = 0.05
    h = np.array([0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    for wet in ([0, 1, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1, 1, 1]):
        assert one(wet, h) == ((0.1 + 0.2) / 2, 0)
    # a wet and a dry look at 0.1: pooled to p = 0.5 exactly, so h_hi = 0.1, in either order
    h = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    for wet in ([0, 1, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]):
        assert one(wet, h) == ((0.0 + 0.1) / 2, 0)
    # every clear look at one level: p = the wet share there, censored low when >= 0.5, high when below
    h = np.full(8, 0.2)
    assert one([1, 0, 1, 0, 1, 0, 1, 0], h) == (0.2, -1)
    assert one([1, 0, 1, 0, 1, 0, 0, 0], h) == (0.2, +1)
    # two levels only
    h = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    assert one([0, 0, 0, 1, 1, 1, 1, 0], h) == (0.25, 0)        # p = 1/4, 3/4
    assert one([0, 1, 0, 1, 1, 1, 1, 0], h) == (0.0, -1)        # p = 1/2, 3/4


def test_a_perfect_step_is_read_between_the_looks_that_bracket_it():
    rng = np.random.default_rng(3)
    T, P = 60, 3000
    h = rng.uniform(-2.0, 2.0, T)
    z0 = rng.uniform(-1.5, 1.5, P)
    C = rng.random((T, P)) < 0.8
    Y = np.where(h[:, None] > z0, 0.4, -0.3)
    z, censored, n_wet, n_dry = invert_monotone(Y, C, h)
    hh = h[:, None]
    below = np.where(C & (hh <= z0), hh, -np.inf).max(axis=0)   # the pixel's highest dry level ...
    above = np.where(C & (hh > z0), hh, np.inf).min(axis=0)     # ... and its lowest wet one
    ok = np.isfinite(below) & np.isfinite(above)                 # a wet and a dry look (and 8 looks: C.sum >= 8 here)
    assert (C.sum(axis=0) >= 8).all() and ok.sum() > 2900
    assert np.array_equal(z[ok], (below[ok] + above[ok]) / 2) and np.isnan(z[~ok]).all()
    assert (censored == 0).all()
    assert np.array_equal(n_wet, (C & (h[:, None] > z0)).sum(axis=0))
    assert np.array_equal(n_dry, (C & (h[:, None] <= z0)).sum(axis=0))


# ── The plain reference on random pixels ────────────────────────────────────

def test_equals_plain_pool_adjacent_violators_on_random_pixels():
    rng = np.random.default_rng(20261028)
    counts = dict(pixels=0, rejected=0, inside=0, low=0, high=0, half=0, mixed_level=0)
    for T in (1, 2, 7, 8, 9, 13, 30, 45, 60, 90, 125):
        for decimals in (0, 1, 2, 8):
            h, C, wet, Y = random_window(rng, T, 150, decimals)
            z, censored, n_wet, n_dry = invert_monotone(Y, C, h)
            zr, cr, p_hi = reference_all(wet, C, h)
            assert np.array_equal(z, zr, equal_nan=True), (T, decimals)
            assert np.array_equal(censored, cr), (T, decimals)
            assert np.array_equal(n_wet, wet.sum(axis=0)) and np.array_equal(n_dry, (C & ~wet).sum(axis=0))
            # what the pixels exercised
            fin = np.isfinite(z)
            counts["pixels"] += len(z)
            counts["rejected"] += int((~fin).sum())
            counts["inside"] += int((fin & (censored == 0)).sum())
            counts["low"] += int((censored == -1).sum())
            counts["high"] += int((censored == 1).sum())
            counts["half"] += sum(p == HALF for p in p_hi)
            levels = np.unique(h)
            n_lev = np.stack([C[h == v].sum(axis=0) for v in levels])
            w_lev = np.stack([wet[h == v].sum(axis=0) for v in levels])
            counts["mixed_level"] += int((fin & ((n_lev >= 2) & (w_lev > 0) & (w_lev < n_lev)).any(axis=0)).sum())
    assert counts["pixels"] == 11 * 4 * 150
    # ties at p = 0.5 exactly at h_hi, levels holding wet and dry looks, both censorings and the guards all occur
    assert min(counts.values()) >= 150, counts


def test_level_set_property_of_the_isotonic_fit():
    """The read-off rests on it: for any c, the looks with p >= c are the longest suffix of whole levels that
    maximises the sum of (wet - c). Checked in exact fractions on random pooled records."""
    rng = np.random.default_rng(11)
    for _ in range(2000):
        m = int(rng.integers(1, 15))
        n = [int(v) for v in rng.integers(1, 4, m)]                # looks per level
        y = [int(rng.integers(0, k + 1)) for k in n]                # wet looks per level
        p = pava(y, n)
        for c in (Fraction(1, 4), Fraction(1, 3), HALF, Fraction(2, 3), Fraction(3, 4)):
            S = [sum(y[i] - c * n[i] for i in range(k, m)) for k in range(m + 1)]   # k = m: the empty suffix
            k_star = S.index(max(S))                                 # the first maximum: the longest suffix
            assert [i for i in range(m) if p[i] >= c] == list(range(k_star, m)), (y, n, c)


def test_reference_pava_is_scipys_isotonic_regression():
    iso = getattr(pytest.importorskip("scipy.optimize"), "isotonic_regression", None)
    if iso is None:
        pytest.skip("this scipy has no isotonic_regression (it arrived in 1.12)")
    rng = np.random.default_rng(12)
    for _ in range(500):
        m = int(rng.integers(1, 30))
        n = rng.integers(1, 5, m)
        y = rng.integers(0, n + 1)
        ours = np.array([float(v) for v in pava([int(v) for v in y], [int(v) for v in n])])
        assert np.allclose(ours, iso(y / n, weights=n.astype(float)).x, rtol=0, atol=1e-12)


# ── The limits ───────────────────────────────────────────────────────────────

def test_guards_fewer_than_8_looks_all_wet_all_dry():
    h = np.arange(12) * 0.1
    wet, C = np.zeros((12, 6), bool), np.zeros((12, 6), bool)
    C[:7, 0], wet[5:7, 0] = True, True                  # 7 clear looks: no value
    C[:8, 1], wet[5:8, 1] = True, True                  # 8 clear looks: between 0.4 and 0.5
    C[:, 2], wet[:, 2] = True, True                     # all wet
    C[:, 3] = True                                      # all dry
                                                        # 4: no clear look
    C[:, 5], wet[:, 5] = True, True                     # 5: one dry look among 12, at the top: censored low
    wet[11, 5] = False
    z, censored, n_wet, n_dry = invert_monotone(None, C, h, wet=wet)
    assert np.isnan(z[[0, 2, 3, 4]]).all()
    assert z[1] == (h[4] + h[5]) / 2 and z[5] == h[0]
    assert censored.tolist() == [0, 0, 0, 0, 0, -1]
    assert n_wet.tolist() == [2, 3, 12, 0, 0, 11] and n_dry.tolist() == [5, 5, 0, 12, 0, 1]
    z, censored, _, _ = invert_monotone(None, C, h, wet=wet, min_looks=13)
    assert np.isnan(z).all() and (censored == 0).all()
    z, _, _, _ = invert_monotone(None, C, h, wet=wet, min_looks=7)
    assert z[0] == (h[4] + h[5]) / 2


def test_undefined_ndwi_is_dry_and_looks_that_are_not_clear_do_not_count():
    Y = np.repeat(np.where(H10 > 0.45, 0.4, -0.3)[:, None], 4, axis=1).astype(np.float32)
    C = np.ones((10, 4), bool)
    Y[2, 1], Y[3, 1] = np.nan, 0.0          # pixel 1: dry looks with an undefined NDWI and with NDWI = 0: still dry
    Y[5, 2] = np.nan                        # pixel 2: its lowest wet look undefined: now dry, h_hi moves to 0.6
    C[4, 3], Y[4, 3] = False, 0.9           # pixel 3: a wet NDWI on a look that is not clear: ignored, and 0.4 is no
                                            # longer one of its levels, so h_lo = 0.3
    z, censored, n_wet, n_dry = invert_monotone(Y, C, H10)
    assert z.tolist() == [(H10[4] + H10[5]) / 2, (H10[4] + H10[5]) / 2, (H10[5] + H10[6]) / 2, (H10[3] + H10[5]) / 2]
    assert censored.tolist() == [0, 0, 0, 0]
    assert n_wet.tolist() == [5, 5, 4, 5] and n_dry.tolist() == [5, 5, 6, 4]


def test_explicit_wet_matrix():
    rng = np.random.default_rng(14)
    h, C, wet, Y = random_window(rng, 60, 400, 1)
    with np.errstate(invalid="ignore"):
        default_wet = C & (Y > 0)
    assert np.array_equal(default_wet, wet)
    assert same(invert_monotone(Y, C, h), invert_monotone(None, C, h, wet=default_wet))
    # a threshold per scene, as V14 (wet = NDWI > t_s): the reference on that wet; Y is not read
    t_s = rng.uniform(-0.2, 0.3, len(h))
    with np.errstate(invalid="ignore"):
        wet_t = C & (Y > t_s[:, None])
    z, censored, n_wet, n_dry = invert_monotone(None, C, h, wet=wet_t)
    zr, cr, _ = reference_all(wet_t, C, h)
    assert np.array_equal(z, zr, equal_nan=True) and np.array_equal(censored, cr)
    assert np.array_equal(n_wet, wet_t.sum(axis=0)) and np.array_equal(n_dry, (C & ~wet_t).sum(axis=0))
    assert not np.array_equal(z, invert_monotone(Y, C, h)[0], equal_nan=True)
    # a wet look outside the clear ones is ignored
    stray = ~C & (rng.random(C.shape) < 0.5)
    assert same(invert_monotone(None, C, h, wet=wet_t | stray), (z, censored, n_wet, n_dry))


def test_blocks_of_pixels_and_order_of_looks_do_not_matter():
    rng = np.random.default_rng(15)
    h, C, _, Y = random_window(rng, 80, 1000, 1)       # one decimal: many looks share a level
    ref = invert_monotone(Y, C, h)
    for chunk in (1, 7, 333, 4000):
        assert same(invert_monotone(Y, C, h, chunk=chunk), ref), chunk
    for _ in range(5):
        perm = rng.permutation(len(h))                 # the looks reordered, ties included
        assert same(invert_monotone(Y[perm], C[perm], h[perm]), ref)


def test_inputs_and_outputs():
    rng = np.random.default_rng(16)
    h, C, _, Y = random_window(rng, 30, 50, 2)
    z, censored, n_wet, n_dry = invert_monotone(Y, C, h)
    assert Y.dtype == np.float32 and z.dtype == np.float64
    assert all(np.issubdtype(v.dtype, np.integer) for v in (censored, n_wet, n_dry))
    assert all(v.shape == (50,) for v in (z, censored, n_wet, n_dry))
    assert set(np.unique(censored)) <= {-1, 0, 1}
    # z is one of the pixel's levels when censored, strictly between two of them otherwise
    for j in np.flatnonzero(np.isfinite(z)):
        lev = np.unique(h[C[:, j]])
        if censored[j] == -1:
            assert z[j] == lev[0]
        elif censored[j] == 1:
            assert z[j] == lev[-1]
        else:
            i = np.searchsorted(lev, z[j])
            assert 0 < i < len(lev) and lev[i - 1] < z[j] < lev[i] and z[j] == (lev[i - 1] + lev[i]) / 2
    # no pixels; no looks
    out = invert_monotone(np.zeros((5, 0)), np.zeros((5, 0), bool), np.arange(5.0))
    assert all(v.shape == (0,) for v in out)
    z, censored, n_wet, n_dry = invert_monotone(np.zeros((0, 3)), np.zeros((0, 3), bool), np.zeros(0))
    assert np.isnan(z).all() and not censored.any() and not n_wet.any() and not n_dry.any()
    # a level that is not finite, or levels that are not one per look
    with pytest.raises(ValueError):
        invert_monotone(Y, C, np.where(np.arange(30) == 4, np.nan, h))
    with pytest.raises(ValueError):
        invert_monotone(Y, C, h[:-1])
