# -*- coding: utf-8 -*-
"""experiments/variants/interval.py (V13 of experiments/variants_addendum_2026-10-28.md) on synthetic pixels.

* bernoulli_interval: z and sigma are the attribution notebook's invert_bernoulli (copied below, and checked to be
  the builder's code) bit for bit, on every guard, on tied levels, in several blocks and with the wet matrix given
  explicitly (V14); the interval is the brute-force profile-likelihood interval (plain loops, the tabulated
  chi-square quantile); z lies inside it; the guards leave no interval; the limits: a larger level nests a wider
  interval (level 0: the maximum itself, level 1: the whole grid), more looks narrow it, a transition below or above
  every observed level is not bracketed, complete separation spans the unobserved gap; on probit pixels with the
  true elevation on the grid it covers about 95 %;
* error_coverage: a hand case (the ranking with ties and a NaN, ceil(q |F|), own and common pixels, every score by
  hand), notebook 01's scores() (copied below, checked to be the builder's) on ranked subsets bit for bit, the size of
  the subsets free of float noise, the 30-pixel rule;
* block_bootstrap: the blocks, the observed differences are scores()'s, every resample is scores() on the explicit
  concatenation of the drawn blocks (brute force), a known difference is recovered, constant errors give a point,
  identical estimates give exactly 0, block-correlated errors widen the interval over the pixel bootstrap, the seed
  reproduces it, fewer than 30 pixels give no interval.

Run:  python -X utf8 -m pytest -q tests/test_variants_interval.py
"""
import ast
import inspect
import math
import os
import sys
from fractions import Fraction

import numpy as np
import pytest
from scipy.special import log_ndtr, ndtr
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BUILDER = os.path.join(ROOT, "notebooks", "validation", "_build_05_attribution.py")

from experiments.variants import interval as iv  # noqa: E402
from pyintertidal import marea  # noqa: E402

CHI2_1_95 = 3.841458820694124     # the 0.95 quantile of chi-square with 1 d.f. (tables: 3.8415)


# ── the builder's functions, copied (the next test checks they are still its code) ─────────────────────────────

B_MIN_LOOKS = 8           # MAREA-B: at least 8 clear looks, one wet and one dry (the variants file)
B_TIE_TOL = 1e-9          # MAREA-B: z levels whose profile log-likelihood is within this of the maximum are tied


def invert_bernoulli(Y, C, h, lo, hi, min_looks=B_MIN_LOOKS, tie_tol=B_TIE_TOL, mu_points=50, chunk=4000):
    """MAREA-B: per pixel, the probit likelihood of the wet/dry record on the clear looks against the levels h,
    P(wet) = Phi((h - z)/sigma), maximised on invert_series' grid (mu_points levels in [lo, hi], sigma in
    marea.SG_GRID). z = the median of the tied z levels (profile log-likelihood within tie_tol of its maximum);
    sigma = the best sigma at the tied level nearest z. NaN with fewer than min_looks clear looks or without a wet
    and a dry one. Returns (z, sigma, the number of tied levels)."""
    grid, sgs = np.linspace(lo, hi, mu_points), np.asarray(marea.SG_GRID, float)
    u = (np.asarray(h, float)[:, None, None] - grid[None, :, None]) / sgs[None, None, :]
    lw, ld = log_ndtr(u).reshape(len(h), -1), log_ndtr(-u).reshape(len(h), -1)   # (looks, levels x sigmas)
    P = Y.shape[1]
    z, sg, n_tied = np.full(P, np.nan), np.full(P, np.nan), np.zeros(P, int)
    for j in range(0, P, chunk):
        s = slice(j, min(j + chunk, P))
        clear = np.asarray(C[:, s], bool)
        with np.errstate(invalid="ignore"):
            wet = clear & (np.asarray(Y[:, s]) > 0)     # an undefined NDWI is never wet
        dry = clear & ~wet
        ll = (wet.T.astype(float) @ lw + dry.T.astype(float) @ ld).reshape(-1, len(grid), len(sgs))
        prof = ll.max(axis=2)                           # the profile over sigma: (pixels, levels)
        tied = prof >= prof.max(axis=1, keepdims=True) - tie_tol
        zb = np.nanmedian(np.where(tied, grid[None, :], np.nan), axis=1)
        near = np.argmin(np.where(tied, np.abs(grid[None, :] - zb[:, None]), np.inf), axis=1)
        sb = sgs[ll[np.arange(len(near)), near].argmax(axis=1)]
        ok = (clear.sum(axis=0) >= min_looks) & wet.any(axis=0) & dry.any(axis=0)
        z[s], sg[s], n_tied[s] = np.where(ok, zb, np.nan), np.where(ok, sb, np.nan), np.where(ok, tied.sum(1), 0)
    return z, sg, n_tied


def scores(p, t):
    """RMSE, bias, centred RMSE, slope, r, the pixels scored and the coverage of p against t (same pixels)."""
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = np.isfinite(p) & np.isfinite(t)
    out = dict(n_px=int(ok.sum()), coverage=float(ok.mean()) if len(ok) else np.nan,
               rmse=np.nan, bias=np.nan, crmse=np.nan, slope=np.nan, r=np.nan)
    if ok.sum() >= 30:
        d = p[ok] - t[ok]
        pc, tc = p[ok] - np.median(p[ok]), t[ok] - np.median(t[ok])
        out.update(rmse=float(np.sqrt(np.mean(d ** 2))), bias=float(np.median(d)),
                   crmse=float(np.sqrt(np.mean((pc - tc) ** 2))), slope=float(np.polyfit(tc, pc, 1)[0]),
                   r=float(np.corrcoef(tc, pc)[0, 1]))
    return out


def builder_function(name):
    """The definition of `name` in a code cell of the attribution notebook's builder (its cells are string literals
    passed to code())."""
    tree = ast.parse(open(BUILDER, encoding="utf-8").read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "code" and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            try:
                body = ast.parse(node.args[0].value.strip("\n")).body
            except SyntaxError:
                continue
            for fn in body:
                if isinstance(fn, ast.FunctionDef) and fn.name == name:
                    return fn
    raise LookupError(name)


def code_of(fn):
    """ast.dump of a function definition without its docstring."""
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant):
        fn.body = fn.body[1:]
    return ast.dump(fn)


def test_the_copies_are_the_builders_code():
    for f in (invert_bernoulli, scores):
        mine = ast.parse(inspect.getsource(f)).body[0]
        assert code_of(mine) == code_of(builder_function(f.__name__)), \
            f"the builder's {f.__name__} changed: copy it here again and re-check bit for bit"


# ── synthetic looks ─────────────────────────────────────────────────────────────

def synthetic_looks(T=60, P=400, seed=11):
    """Probit wet/dry records with an NDWI magnitude, undefined NDWI, clouds and levels rounded to 0.1 m (several
    looks share a level, but the lowest and the highest look are alone at theirs), Y float32 as the extraction; the
    first pixels for the guards and the limits: 0 every look wet, 1 every look dry, 2 every NDWI undefined (dry),
    3 no clear look, 4 seven clear looks (wet and dry), 5 eight clear looks (wet and dry), 6 complete separation,
    7 wet but at the lowest look (all clear), 8 dry but at the highest look (all clear), 9 NDWI exactly 0 (dry)
    below its transition."""
    rng = np.random.default_rng(seed)
    h = np.round(rng.uniform(-2.0, 2.0, T), 1)
    low, high = np.argmin(h), np.argmax(h)
    h[low], h[high] = h.min() - 0.1, h.max() + 0.1
    z_true, s_true = rng.uniform(-1.6, 1.6, P), rng.uniform(0.03, 0.6, P)
    wet = rng.random((T, P)) < ndtr((h[:, None] - z_true) / s_true)
    Y = np.where(wet, rng.uniform(0.0, 0.6, (T, P)), rng.uniform(-0.5, 0.0, (T, P)))
    Y[rng.random((T, P)) < 0.05] = np.nan                # undefined NDWI: a clear look that is dry
    C = rng.random((T, P)) > 0.2
    order = np.argsort(h, kind="stable")
    Y[:, 0], Y[:, 1], Y[:, 2], C[:, 3] = 0.4, -0.3, np.nan, False
    for j, n_low in ((4, 3), (5, 4)):
        C[:, j] = False
        C[order[:n_low], j] = C[order[-4:], j] = True
        Y[:, j] = np.where(h > 0, 0.4, -0.3)
    Y[:, 6] = np.where(h > 0.35, 0.5, -0.2)
    Y[:, 7], C[:, 7] = np.where(np.arange(T) == low, -0.3, 0.4), True
    Y[:, 8], C[:, 8] = np.where(np.arange(T) == high, 0.4, -0.3), True
    Y[:, 9] = np.where(h > 0.5, 0.4, 0.0)
    return Y.astype(np.float32), C, h


def assert_same(a, b):
    assert np.array_equal(a, b, equal_nan=True), (a, b)


# ── bernoulli_interval: V2 bit for bit ─────────────────────────────────────────

def test_z_and_sigma_are_invert_bernoullis_bit_for_bit():
    Y, C, h = synthetic_looks()
    zr, sr, _ = invert_bernoulli(Y, C, h, h.min(), h.max())
    assert np.isfinite(zr).sum() > 300
    for chunk in (4000, 7):
        z, sg, *_ = iv.bernoulli_interval(Y, C, h, h.min(), h.max(), chunk=chunk)
        assert_same(z, zr)
        assert_same(sg, sr)
    # grid bounds that are not the extremes of the delayed levels, as in the notebook (lo, hi without delay)
    zr, sr, _ = invert_bernoulli(Y, C, h + 0.07, h.min() - 0.1, h.max() + 0.2)
    z, sg, *_ = iv.bernoulli_interval(Y, C, h + 0.07, h.min() - 0.1, h.max() + 0.2)
    assert_same(z, zr)
    assert_same(sg, sr)
    # the Wadden's longest window, three blocks of 4,000 pixels and a remainder
    Y, C, h = synthetic_looks(T=125, P=9000, seed=12)
    zr, sr, _ = invert_bernoulli(Y, C, h, h.min(), h.max())
    z, sg, *_ = iv.bernoulli_interval(Y, C, h, h.min(), h.max())
    assert_same(z, zr)
    assert_same(sg, sr)
    # other guards and grids are passed on
    zr, sr, _ = invert_bernoulli(Y, C, h, h.min(), h.max(), min_looks=20, tie_tol=1e-6, mu_points=60, chunk=999)
    z, sg, *_ = iv.bernoulli_interval(Y, C, h, h.min(), h.max(), min_looks=20, tie_tol=1e-6, mu_points=60, chunk=999)
    assert_same(z, zr)
    assert_same(sg, sr)


def test_an_explicit_wet_matrix():
    Y, C, h = synthetic_looks()
    ref = iv.bernoulli_interval(Y, C, h, h.min(), h.max())
    with np.errstate(invalid="ignore"):
        W, W_any = C & (Y > 0), Y > 0
    for wet in (W, W_any, np.asfortranarray(W)):         # a wet look is a clear look: W_any's cloudy looks drop out
        for a, b in zip(ref, iv.bernoulli_interval(None, C, h, h.min(), h.max(), wet=wet)):   # Y is not read
            assert_same(a, b)
    # V14: a threshold t_s per scene is the default rule on NDWI - t_s
    t_s = np.random.default_rng(1).uniform(-0.2, 0.3, len(h))
    with np.errstate(invalid="ignore"):
        got = iv.bernoulli_interval(None, C, h, h.min(), h.max(), wet=C & (Y > t_s[:, None]))
    for a, b in zip(iv.bernoulli_interval(Y - t_s[:, None], C, h, h.min(), h.max()), got):
        assert_same(a, b)


# ── bernoulli_interval: the interval ───────────────────────────────────────────

def brute_force_interval(wet, clear, h, lo, hi, drop=CHI2_1_95 / 2, min_looks=8):
    """(z_lo, z_hi, bracketed) of one pixel by plain loops: lp(z) = the largest probit log-likelihood over the
    sigmas at each of the 50 grid levels, the lowest and highest level with lp >= max lp - drop."""
    wet = wet & clear
    if clear.sum() < min_looks or not wet.any() or not (clear & ~wet).any():
        return np.nan, np.nan, False
    grid, lp = np.linspace(lo, hi, 50), []
    for zk in grid:
        best = -np.inf
        for sk in marea.SG_GRID:
            ll = 0.0
            for k in range(len(h)):
                if clear[k]:
                    u = (h[k] - zk) / sk
                    ll += float(log_ndtr(u if wet[k] else -u))
            best = max(best, ll)
        lp.append(best)
    inside = [k for k in range(len(grid)) if lp[k] >= max(lp) - drop]
    z_lo, z_hi = grid[inside[0]], grid[inside[-1]]
    return z_lo, z_hi, bool(z_lo > grid[0] and z_hi < grid[-1])


def test_the_cut_is_the_addendums():
    assert round(chi2.ppf(0.95, 1) / 2, 4) == 1.9207
    assert chi2.ppf(0.95, 1) == pytest.approx(CHI2_1_95, rel=1e-12)


def test_the_interval_is_the_brute_force_profile_likelihood_interval():
    Y, C, h = synthetic_looks(T=40, P=36, seed=13)
    z, sg, z_lo, z_hi, br = iv.bernoulli_interval(Y, C, h, h.min(), h.max(), chunk=5)
    assert 0 < br.sum() < np.isfinite(z).sum()           # some bracketed, some not
    for j in range(Y.shape[1]):
        with np.errstate(invalid="ignore"):
            lo_b, hi_b, br_b = brute_force_interval(Y[:, j] > 0, C[:, j], h, h.min(), h.max())
        assert np.isclose(z_lo[j], lo_b, rtol=0, atol=1e-12, equal_nan=True), (j, z_lo[j], lo_b)
        assert np.isclose(z_hi[j], hi_b, rtol=0, atol=1e-12, equal_nan=True), (j, z_hi[j], hi_b)
        assert br[j] == br_b, j


def test_z_is_inside_its_interval_and_no_value_means_no_interval():
    Y, C, h = synthetic_looks()
    z, sg, z_lo, z_hi, br = iv.bernoulli_interval(Y, C, h, h.min(), h.max())
    grid = np.linspace(h.min(), h.max(), 50)
    fin = np.isfinite(z)
    assert np.array_equal(fin, np.isfinite(z_lo)) and np.array_equal(fin, np.isfinite(z_hi))
    assert np.all(z_lo[fin] <= z[fin]) and np.all(z[fin] <= z_hi[fin])
    assert np.all(np.isin(z_lo[fin], grid)) and np.all(np.isin(z_hi[fin], grid))
    assert np.array_equal(br, fin & (np.nan_to_num(z_lo, nan=grid[0]) > grid[0])
                          & (np.nan_to_num(z_hi, nan=grid[-1]) < grid[-1]))
    assert br.dtype == bool


def test_the_guards():
    Y, C, h = synthetic_looks()
    z, sg, z_lo, z_hi, br = iv.bernoulli_interval(Y, C, h, h.min(), h.max())
    for j in (0, 1, 2, 3, 4):        # every look wet, every look dry, every NDWI undefined, no clear look, 7 looks
        assert np.isnan([z[j], sg[j], z_lo[j], z_hi[j]]).all() and not br[j], j
    assert np.isfinite([z[5], sg[5], z_lo[5], z_hi[5]]).all()   # 8 clear looks with a wet and a dry one
    assert np.isfinite(iv.bernoulli_interval(Y, C, h, h.min(), h.max(), min_looks=7)[0][4])
    assert np.isfinite(z[9]) and z[9] > 0.3                      # NDWI = 0 is dry: the transition is at 0.5


def test_the_limits_of_the_level():
    Y, C, h = synthetic_looks()
    grid = np.linspace(h.min(), h.max(), 50)
    z, _, lo95, hi95, br95 = iv.bernoulli_interval(Y, C, h, h.min(), h.max())
    _, _, lo50, hi50, br50 = iv.bernoulli_interval(Y, C, h, h.min(), h.max(), level=0.5)
    _, _, lo99, hi99, br99 = iv.bernoulli_interval(Y, C, h, h.min(), h.max(), level=0.99)
    fin = np.isfinite(z)
    assert np.all(lo99[fin] <= lo95[fin]) and np.all(lo95[fin] <= lo50[fin])     # nested
    assert np.all(hi50[fin] <= hi95[fin]) and np.all(hi95[fin] <= hi99[fin])
    assert np.mean(hi99[fin] - lo99[fin]) > np.mean(hi95[fin] - lo95[fin]) > np.mean(hi50[fin] - lo50[fin])
    assert not np.any(br99 & ~br95) and not np.any(br95 & ~br50)
    # level 0: no drop, the interval is the maximum itself (z, where the maximum is one level)
    _, _, lo0, hi0, _ = iv.bernoulli_interval(Y, C, h, h.min(), h.max(), level=0.0)
    one = fin & (invert_bernoulli(Y, C, h, h.min(), h.max())[2] == 1)
    assert one.sum() > 200 and np.array_equal(lo0[one], z[one]) and np.array_equal(hi0[one], z[one])
    # level 1: an infinite drop, the whole grid, never bracketed
    _, _, lo1, hi1, br1 = iv.bernoulli_interval(Y, C, h, h.min(), h.max(), level=1.0)
    assert np.all(lo1[fin] == grid[0]) and np.all(hi1[fin] == grid[-1]) and not br1.any()


def test_more_looks_narrow_the_interval():
    rng = np.random.default_rng(4)
    width = {}
    for T in (50, 400):
        h = rng.uniform(-2.0, 2.0, T)
        z_true = rng.uniform(-1.0, 1.0, 500)
        wet = rng.random((T, 500)) < ndtr((h[:, None] - z_true) / 0.22)
        _, _, z_lo, z_hi, br = iv.bernoulli_interval(np.where(wet, 0.3, -0.2), np.ones((T, 500), bool), h, -2.0, 2.0)
        assert br.all()
        width[T] = np.median(z_hi - z_lo)
    assert width[400] < 0.5 * width[50], width          # about 1/sqrt(8) as many metres, on a 0.08 m grid


def test_transitions_the_levels_do_not_bound():
    Y, C, h = synthetic_looks()
    grid = np.linspace(h.min(), h.max(), 50)
    z, _, z_lo, z_hi, br = iv.bernoulli_interval(Y, C, h, h.min(), h.max())
    assert np.isfinite(z[7]) and z_lo[7] == grid[0] and not br[7]       # wet at every level but the lowest
    assert np.isfinite(z[8]) and z_hi[8] == grid[-1] and not br[8]      # dry at every level but the highest
    # complete separation inside the range: the interval spans the whole unobserved gap, and is bracketed
    h = np.concatenate([np.linspace(-2.0, -0.3, 30), np.linspace(0.5, 2.0, 30)])
    Y = np.where(h > 0, 0.4, -0.3)[:, None]
    z, sg, z_lo, z_hi, br = iv.bernoulli_interval(Y, np.ones_like(Y, bool), h, h.min(), h.max())
    grid = np.linspace(h.min(), h.max(), 50)
    gap = grid[(grid > -0.3) & (grid < 0.5)]
    assert br[0] and sg[0] == 0.03 and z_lo[0] <= gap.min() and z_hi[0] >= gap.max()


def test_the_interval_covers_the_true_elevation_about_95_percent():
    rng = np.random.default_rng(21)
    T, P = 125, 3000
    h = rng.uniform(-2.0, 2.0, T)
    grid = np.linspace(h.min(), h.max(), 50)
    z_true = grid[rng.integers(8, 42, P)]               # on the grid, away from its ends
    s_true = rng.choice([0.22, 0.32, 0.45], P)          # in marea.SG_GRID
    wet = rng.random((T, P)) < ndtr((h[:, None] - z_true) / s_true)
    _, _, z_lo, z_hi, _ = iv.bernoulli_interval(np.where(wet, 0.3, -0.2), np.ones((T, P), bool), h, h.min(), h.max())
    share = np.mean((z_lo <= z_true) & (z_true <= z_hi))
    assert 0.92 <= share <= 0.97, share


# ── error_coverage ─────────────────────────────────────────────────────────────

def test_error_coverage_by_hand():
    truth = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0])
    rel = np.array([2, 5, 9, 5, 1, 3, np.nan])           # pixel 2 has no truth; pixel 6 no reliability: last
    A = np.array([0.5, 1.0, 9.0, 2.5, np.nan, 4.0, 6.0])
    B = np.array([0.0, 2.0, np.nan, np.nan, 3.0, 4.5, 5.0])
    assert iv.reliability_ranking(truth, rel).tolist() == [1, 3, 5, 0, 4, 6]   # ties by pixel index
    tab = iv.error_coverage(truth, {"A": A, "B": B}, rel, coverages=(0.3, 0.5, 1.0), min_px=1)
    r = math.sqrt
    expected = {   # n_subset, (n_px, rmse, bias, crmse), (the same on the common pixels)
        (0.3, "A"): (2, (2, r(0.125), 0.25, 0.25), (1, 0.0, 0.0, 0.0)),           # {1, 3}; common {1}
        (0.3, "B"): (2, (1, 1.0, 1.0, 0.0), (1, 1.0, 1.0, 0.0)),
        (0.5, "A"): (3, (3, r(0.25 / 3), 0.0, r(1 / 6)), (2, 0.0, 0.0, 0.0)),     # {1, 3, 5}; common {1, 5}
        (0.5, "B"): (3, (2, r(0.625), 0.75, 0.25), (2, r(0.625), 0.75, 0.25)),
        (1.0, "A"): (6, (5, r(0.3), 0.5, r(0.15)), (4, r(0.3125), 0.25, r(0.3125))),   # F; common {0, 1, 5, 6}
        (1.0, "B"): (6, (5, 0.5, 0.0, 0.5), (4, r(0.3125), 0.25, r(0.3125))),
    }
    assert list(zip(tab.coverage, tab.method)) == list(expected)
    for (q, m), (n_subset, own, common) in expected.items():
        row = tab[(tab.coverage == q) & (tab.method == m)].iloc[0]
        assert row.n_subset == n_subset
        for k, v_own, v_common in zip(iv.SCORE_KEYS, own, common):
            assert row[k] == pytest.approx(v_own, abs=1e-12), (q, m, k)
            assert row[f"{k}_common"] == pytest.approx(v_common, abs=1e-12), (q, m, k)
    # the default min_px is scores()'s 30: the pixels are counted, nothing is scored
    tab30 = iv.error_coverage(truth, {"A": A, "B": B}, rel, coverages=(1.0,))
    assert tab30.n_px.tolist() == [5, 5] and tab30.rmse.isna().all() and tab30.crmse_common.isna().all()


def test_error_coverage_is_notebook_scores_on_the_ranked_subsets():
    rng = np.random.default_rng(31)
    P = 2000
    truth = rng.uniform(-1.5, 1.5, P)
    truth[rng.random(P) < 0.1] = np.nan
    rel = rng.integers(0, 15, P).astype(float)           # many ties
    est = {name: np.where(rng.random(P) < keep, truth + rng.normal(bias, sd, P), np.nan)
           for name, keep, bias, sd in (("V0", 1.0, 0.2, 0.3), ("V13a", 0.85, 0.05, 0.2), ("DEA", 0.9, -0.1, 0.25))}
    est["DEA"][rel < 3] = np.nan                         # abstains on the least reliable pixels
    tab = iv.error_coverage(truth, est, rel)
    F = [i for i in range(P) if np.isfinite(truth[i])]
    ranked = sorted(F, key=lambda i: (-rel[i], i))       # the definition, by plain sorting
    for q in (1.0, 0.95, 0.90, 0.85, 0.80):
        n_keep = math.ceil(Fraction(str(q)) * len(F))
        keep = np.zeros(P, bool)
        keep[ranked[:n_keep]] = True
        common = keep & np.all([np.isfinite(p) for p in est.values()], axis=0)
        for m, p in est.items():
            row = tab[(tab.coverage == q) & (tab.method == m)].iloc[0]
            own, com = scores(p[keep], truth[keep]), scores(p[common], truth[common])
            assert row.n_subset == n_keep
            for k in iv.SCORE_KEYS:                       # bit for bit
                assert row[k] == own[k] and row[f"{k}_common"] == com[k], (q, m, k)
    whole = tab[tab.coverage == 1.0].set_index("method")
    for m, p in est.items():                             # q = 1 is the notebook's table
        assert whole.loc[m, "rmse"] == scores(p, truth)["rmse"] and whole.loc[m, "crmse"] == scores(p, truth)["crmse"]


def test_the_subset_is_ceil_of_q_F_without_float_noise():
    for n_F, q, n in ((100, 0.07, 7), (20, 0.95, 19), (20, 0.85, 17), (30, 0.9, 27), (101, 0.8, 81), (7, 0.5, 4),
                      (6, 1.0, 6)):
        truth = np.r_[np.zeros(n_F), np.full(3, np.nan)]   # three pixels without a truth do not count
        tab = iv.error_coverage(truth, {"m": truth}, np.arange(n_F + 3), coverages=(q,), min_px=1)
        assert tab.n_subset.iloc[0] == n == tab.n_px.iloc[0], (n_F, q)


# ── block_bootstrap ────────────────────────────────────────────────────────────

def synthetic_blocks(P=3000, seed=41):
    """P pixels at random positions of a 150 x 100 grid (5 x 4 blocks of 32 pixels, the last ones partial)."""
    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(150 * 100, P, replace=False))
    rows, cols = np.divmod(flat, 100)
    return rng, rows, cols, rng.uniform(-1.5, 1.5, P)


def brute_force_bootstrap(t, pv, pr, rows, cols, block, n_boot, seed):
    """The percentile intervals of the paired differences, each resample built as the concatenation of its drawn
    blocks and scored by scores()."""
    members = {}
    for i in np.flatnonzero(np.isfinite(t) & np.isfinite(pv) & np.isfinite(pr)):
        members.setdefault((int(rows[i]) // block, int(cols[i]) // block), []).append(i)
    blocks = [members[key] for key in sorted(members)]   # numbered in the order (block row, block column)
    rng = np.random.default_rng(seed)
    d_rmse, d_crmse = [], []
    for _ in range(n_boot):
        ii = np.concatenate([blocks[b] for b in rng.integers(0, len(blocks), len(blocks))])
        sv, sr = scores(pv[ii], t[ii]), scores(pr[ii], t[ii])
        d_rmse.append(sv["rmse"] - sr["rmse"])
        d_crmse.append(sv["crmse"] - sr["crmse"])
    return np.quantile(d_rmse, [0.025, 0.975]), np.quantile(d_crmse, [0.025, 0.975])


def two_estimates(rng, t):
    pv = t + rng.normal(0.05, 0.15, t.size)
    pr = t + rng.normal(-0.1, 0.25, t.size)
    pv[rng.random(t.size) < 0.1] = np.nan
    pr[rng.random(t.size) < 0.2] = np.nan
    return pv, pr


def test_the_blocks():
    rows, cols = np.array([0, 31, 32, 0, 100, 5]), np.array([0, 31, 0, 32, 100, 40])
    cl, B = iv.block_index(rows, cols)                   # blocks (0,0) (0,0) (1,0) (0,1) (3,3) (0,1)
    assert B == 4 and cl.tolist() == [0, 0, 2, 1, 3, 1]
    assert iv.block_index(rows, cols, block=1)[1] == 6
    assert iv.block_index(rows + 640, cols + 3200)[0].tolist() == cl.tolist()   # where the pixels are, not the frame


def test_block_bootstrap_observed_is_scores_on_the_pixels_both_have():
    rng, rows, cols, t = synthetic_blocks()
    pv, pr = two_estimates(rng, t)
    t[:50] = np.nan
    out = iv.block_bootstrap(t, pv, pr, rows, cols, n_boot=50)
    both = np.isfinite(t) & np.isfinite(pv) & np.isfinite(pr)
    sv, sr = scores(pv[both], t[both]), scores(pr[both], t[both])
    assert out["n_px"] == both.sum() and out["n_blocks"] == 20
    assert out["rmse_variant"] == sv["rmse"] and out["rmse_reference"] == sr["rmse"]
    assert out["crmse_variant"] == sv["crmse"] and out["crmse_reference"] == sr["crmse"]
    assert out["d_rmse"] == sv["rmse"] - sr["rmse"] and out["d_crmse"] == sv["crmse"] - sr["crmse"]


def test_every_resample_is_scores_on_the_concatenated_blocks():
    rng, rows, cols, t = synthetic_blocks()
    pv, pr = two_estimates(rng, t)
    for block, n_boot, seed in ((32, 200, 20261028), (8, 100, 3)):
        out = iv.block_bootstrap(t, pv, pr, rows, cols, block=block, n_boot=n_boot, seed=seed)
        (r_lo, r_hi), (c_lo, c_hi) = brute_force_bootstrap(t, pv, pr, rows, cols, block, n_boot, seed)
        assert out["d_rmse_lo"] == pytest.approx(r_lo, abs=1e-12) and out["d_rmse_hi"] == pytest.approx(r_hi, abs=1e-12)
        assert out["d_crmse_lo"] == pytest.approx(c_lo, abs=1e-12) and out["d_crmse_hi"] == pytest.approx(c_hi, abs=1e-12)


def test_block_bootstrap_recovers_a_known_difference():
    rows, cols = np.divmod(np.arange(160 * 160), 160)
    rng = np.random.default_rng(51)
    t = rng.uniform(-1.5, 1.5, rows.size)
    pv, pr = t + rng.normal(0, 0.10, t.size), t + rng.normal(0, 0.30, t.size)   # errors of 0.10 and 0.30 m
    out = iv.block_bootstrap(t, pv, pr, rows, cols)
    for k in ("d_rmse", "d_crmse"):
        assert out[k] == pytest.approx(-0.20, abs=0.005)
        assert -0.21 < out[f"{k}_lo"] < -0.20 < out[f"{k}_hi"] < -0.19, (k, out)


def test_constant_and_identical_errors():
    rng, rows, cols, t = synthetic_blocks()
    out = iv.block_bootstrap(t, t + 0.1, t - 0.3, rows, cols, n_boot=200)   # every resample: RMSE 0.1 and 0.3
    for k in ("d_rmse", "d_rmse_lo", "d_rmse_hi"):
        assert out[k] == pytest.approx(-0.2, abs=1e-12), k
    for k in ("d_crmse", "d_crmse_lo", "d_crmse_hi"):                         # no centred error at all
        assert out[k] == pytest.approx(0.0, abs=1e-12), k
    pv, _ = two_estimates(rng, t)
    same = iv.block_bootstrap(t, pv, pv.copy(), rows, cols, n_boot=200)       # paired: exactly 0 in every resample
    assert all(same[k] == 0.0 for k in ("d_rmse", "d_rmse_lo", "d_rmse_hi", "d_crmse", "d_crmse_lo", "d_crmse_hi"))


def test_block_correlated_errors_widen_the_interval_over_the_pixel_bootstrap():
    rows, cols = np.divmod(np.arange(160 * 160), 160)
    rng = np.random.default_rng(8)
    t = rng.uniform(-1.5, 1.5, rows.size)
    cl, B = iv.block_index(rows, cols)
    pv = t + rng.normal(0, 0.2, B)[cl] + rng.normal(0, 0.05, t.size)       # an error shared by each block
    pr = t + rng.normal(0, 0.2, t.size)
    blocks = iv.block_bootstrap(t, pv, pr, rows, cols, block=32, n_boot=400)
    pixels = iv.block_bootstrap(t, pv, pr, rows, cols, block=1, n_boot=400)
    assert blocks["n_blocks"] == 25 and pixels["n_blocks"] == rows.size
    for k in ("d_rmse", "d_crmse"):
        w_blocks, w_pixels = blocks[f"{k}_hi"] - blocks[f"{k}_lo"], pixels[f"{k}_hi"] - pixels[f"{k}_lo"]
        assert w_blocks > 5 * w_pixels, (k, w_blocks, w_pixels)
        assert blocks[f"{k}_lo"] < 0 < blocks[f"{k}_hi"]          # the blocks: no difference shown
        assert 0 < pixels[f"{k}_lo"]                              # the pixels alone would claim one


def test_block_bootstrap_is_reproducible_with_its_seed():
    rng, rows, cols, t = synthetic_blocks()
    pv, pr = two_estimates(rng, t)
    a = iv.block_bootstrap(t, pv, pr, rows, cols, n_boot=200)
    assert a == iv.block_bootstrap(t, pv, pr, rows, cols, n_boot=200)
    c = iv.block_bootstrap(t, pv, pr, rows, cols, n_boot=200, seed=1)
    assert c["d_rmse"] == a["d_rmse"] and c["d_crmse"] == a["d_crmse"]       # the observed values are not drawn
    assert (c["d_rmse_lo"], c["d_rmse_hi"]) != (a["d_rmse_lo"], a["d_rmse_hi"])
    defaults = {k: v.default for k, v in inspect.signature(iv.block_bootstrap).parameters.items()}
    assert (defaults["block"], defaults["n_boot"], defaults["seed"]) == (32, 1000, 20261028)   # the addendum's


def test_block_bootstrap_needs_30_pixels():
    rng, rows, cols, t = synthetic_blocks(P=40)
    pv = t + 0.1
    pv[:11] = np.nan                                      # 29 pixels with both estimates
    out = iv.block_bootstrap(t, pv, t + 0.2, rows, cols)
    assert out["n_px"] == 29
    assert all(np.isnan(out[k]) for k in ("d_rmse", "d_rmse_lo", "d_rmse_hi", "d_crmse", "d_crmse_lo", "d_crmse_hi"))
    assert np.isfinite(iv.block_bootstrap(t, pv, t + 0.2, rows, cols, min_px=20)["d_rmse_lo"])
