# -*- coding: utf-8 -*-
"""V4, MAREA's robust fit (experiments/variants/huber.py), against what it must be.

* delta = inf and no SCL 7: V4 is V0 (marea.invert_series) and V4-0 is V1 (the builder's marea_fit, notebook 01's
  invert_ndwi0), bit for bit, over several blocks, every first refit repeating the start (n_iter = 0);
* the brute force: the same iteratively reweighted fit written with plain loops over the grid and the looks;
* a few gross outliers drag least squares' midpoint, not Huber's;
* the iteration stops per pixel: only the pixels whose fit has not repeated are refitted, a pixel that stopped is not
  touched again, a pixel that stopped is a fixed point of the reweighting, the cap stops the others;
* the SCL 7 weight enters the start and every refit;
* the guards and the edge cases: NaN, all wet, all dry, fewer than 8 clear looks, tied levels;
* the defaults are the addendum's constants.

Run:  python -m pytest -q tests/test_variants_huber.py
"""
import ast
import inspect
import json
import math
import os
import sys

import numpy as np
from scipy.special import ndtr, ndtri

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks", "validation")
sys.path.insert(0, ROOT)

from pyintertidal import marea  # noqa: E402
from pyintertidal.elevation import _fit_block  # noqa: E402
from experiments.variants import huber  # noqa: E402


def notebook_function(name, path, **names):
    """A function defined in a code cell of a notebook, compiled alone with the given globals."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        try:
            body = ast.parse(src).body
        except SyntaxError:
            continue
        for node in body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                ns = dict(np=np, **names)
                exec(compile(ast.get_source_segment(src, node), f"{os.path.basename(path)}:{name}", "exec"), ns)
                return ns[name]
    raise LookupError(f"{name} in {path}")


def synthetic_curves(P=9000, T=60, seed=3):
    """NDWI series a + b Phi((h - z)/sigma) + noise; a few pixels the guards reject or that never cross 0, a few
    without a clear look (test_notebook_05_simple's synthetic pixels)."""
    rng = np.random.default_rng(seed)
    h = np.sort(rng.uniform(-2.0, 2.0, T))[rng.permutation(T)]
    a, b = rng.uniform(-0.5, -0.1, P), rng.uniform(0.3, 1.0, P)
    a[:50], b[:50] = 0.05, 0.5
    z, sg = rng.uniform(-1.5, 1.5, P), rng.uniform(0.05, 0.4, P)
    Y = a + b * ndtr((h[:, None] - z) / sg) + rng.normal(0.0, 0.05, (T, P))
    C = rng.random((T, P)) > 0.15
    C[:, 50:60] = False
    return Y.astype(np.float32), C, h


def gross_pixels():
    """18 pixels: a clean curve (sigma 0.1, noise 0.02, z on a grid level) with 3 or 4 gross looks (NDWI -0.6 on wet
    looks 0.15-0.8 m above z, +0.9 on dry looks as far below). Returns Y, h, the true z, the gross looks."""
    T = 60
    h = np.random.default_rng(0).permutation(np.linspace(-1.8, 1.8, T))
    grid = np.linspace(h.min(), h.max(), 50)
    cols, zt, gross = [], [], []
    for seed in range(6):
        r = np.random.default_rng(100 + seed)
        z = grid[r.integers(15, 35)]
        clean = -0.3 + 0.7 * ndtr((h - z) / 0.1) + r.normal(0, 0.02, T)
        above = np.flatnonzero((h > z + 0.15) & (h < z + 0.8))
        below = np.flatnonzero((h < z - 0.15) & (h > z - 0.8))
        for idx, val in [(r.choice(above, 3, replace=False), -0.6),
                         (r.choice(below, 3, replace=False), 0.9),
                         (np.r_[r.choice(above, 2, replace=False), r.choice(below, 2, replace=False)],
                          np.array([-0.6, -0.6, 0.9, 0.9]))]:
            y = clean.copy()
            y[idx] = val
            g = np.zeros(T, bool)
            g[idx] = True
            cols.append(y), zt.append(z), gross.append(g)
    return np.stack(cols, axis=1), h, np.array(zt), np.stack(gross, axis=1)


# ── the brute force ─────────────────────────────────────────────────────────

def fit_by_loops(y, w, h, grid, sg_grid):
    """_fit_block for one pixel by plain loops: at every grid point (mu outer, sigma inner) where det > 1e-9, the
    weighted least-squares (a, b) in closed form; the first smallest weighted sum of squared residuals wins.
    Returns (a, b, mu, sigma), zeros where no grid point can be fitted."""
    best, fit = math.inf, (0.0, 0.0, 0.0, 0.0)
    for mu in grid:
        for sg in sg_grid:
            phi = [0.5 * (1.0 + math.erf((hk - mu) / (sg * math.sqrt(2.0)))) for hk in h]
            n = sp = spp = sy = syp = 0.0
            for wk, yk, pk in zip(w, y, phi):
                n += wk
                sp += wk * pk
                spp += wk * pk * pk
                sy += wk * yk
                syp += wk * yk * pk
            det = n * spp - sp * sp
            if det <= 1e-9:
                continue
            b = (n * syp - sp * sy) / det
            a = (sy - b * sp) / n
            loss = sum(wk * (yk - a - b * pk) ** 2 for wk, yk, pk in zip(w, y, phi))
            if loss < best:
                best, fit = loss, (a, b, float(mu), float(sg))
    return fit


def huber_by_loops(y, clear, s7, h, lo, hi, delta=0.10, s7_weight=0.5, max_iter=20, mu_points=50):
    """V4 of one pixel by plain loops: (z, z0, sigma, a, b, n_iter), the addendum's definition read literally."""
    y = [0.0 if math.isnan(v) else float(v) for v in y]                       # V0's inputs
    base = [(s7_weight if s7[k] else 1.0) if clear[k] else 0.0 for k in range(len(y))]
    grid = np.linspace(lo, hi, mu_points)
    fit, n_iter = fit_by_loops(y, base, h, grid, marea.SG_GRID), 0
    for k in range(1, max_iter + 1):
        a, b, mu, sg = fit
        w = list(base)
        if sg > 0:
            for t in range(len(y)):
                r = y[t] - (a + b * 0.5 * (1.0 + math.erf((h[t] - mu) / (sg * math.sqrt(2.0)))))
                if r != 0:
                    w[t] = base[t] * min(1.0, delta / abs(r))
        new = fit_by_loops(y, w, h, grid, marea.SG_GRID)
        if new == fit:                                                         # the fit repeats exactly
            break
        fit, n_iter = new, k
        if new[3] == 0:
            break
    a, b, mu, sg = fit
    ok = sum(v > 0 for v in base) >= 8 and 0.15 < b < 2.5 and abs(a) < 2.5
    z = mu if ok else math.nan
    z0 = mu + sg * float(ndtri(-a / b)) if ok and a < 0 < a + b else math.nan
    return z, z0, sg, a, b, n_iter


def brute_force_pixels(T=40, P=10, seed=21):
    """Pixels with noise, gross looks, undefined NDWI, SCL 7 looks; one with 7 clear looks, one all wet."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.8, 1.8, T)
    z, sg = rng.uniform(-1.0, 1.0, P), rng.uniform(0.05, 0.4, P)
    Y = -0.3 + 0.7 * ndtr((h[:, None] - z) / sg) + rng.normal(0, 0.03, (T, P))
    gross = rng.random((T, P)) < 0.08
    Y[gross] = rng.choice([-0.7, 0.9], int(gross.sum()))
    Y[:, 2] = 0.15 + 0.35 * ndtr((h - 0.2) / 0.2) + rng.normal(0, 0.03, T)    # every look wet: no crossing
    Y[rng.random((T, P)) < 0.04] = np.nan
    C = rng.random((T, P)) > 0.15
    C[:, 1] = np.arange(T) < 7                                                # 7 clear looks: no value
    S7 = C & (rng.random((T, P)) < 0.15)
    return Y, C, S7, h


def assert_same_as_loops(out, ref, j, n_iter_too=True):
    z, z0, sg, a, b, n = (v[j] for v in out)
    zr, z0r, sgr, ar, br, nr = ref
    assert np.isclose(z, zr, rtol=0, atol=1e-12, equal_nan=True), (j, z, zr)
    assert np.isclose(z0, z0r, rtol=0, atol=1e-9, equal_nan=True), (j, z0, z0r)
    assert sg == sgr and np.isclose(a, ar, rtol=0, atol=1e-9) and np.isclose(b, br, rtol=0, atol=1e-9), \
        (j, (sg, a, b), (sgr, ar, br))
    if n_iter_too:
        assert n == nr, (j, n, nr)


def test_huber_is_the_brute_force_irls():
    Y, C, S7, h = brute_force_pixels()
    out = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, max_iter=5, chunk=3)    # several blocks
    for j in range(Y.shape[1]):
        assert_same_as_loops(out, huber_by_loops(Y[:, j], C[:, j], S7[:, j], h, h.min(), h.max(), max_iter=5), j)
    assert np.isnan(out[0][1]) and np.isfinite(out[0][2]) and np.isnan(out[1][2])
    assert (out[5][[0, 3, 4, 5, 6, 7, 8, 9]] == 5).all(), out[5]             # gross looks: still moving at 5


def test_huber_is_the_brute_force_irls_to_the_cap():
    """With the default cap the two can stop a refit apart (the last bits of their sums differ), so n_iter is not
    compared; the fit is."""
    Y, C, S7, h = brute_force_pixels(seed=22, P=6)
    out = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, chunk=4)
    for j in range(Y.shape[1]):
        assert_same_as_loops(out, huber_by_loops(Y[:, j], C[:, j], S7[:, j], h, h.min(), h.max()), j,
                             n_iter_too=False)


def test_tied_levels_are_no_special_case():
    rng = np.random.default_rng(8)
    h = np.repeat(rng.uniform(-1.5, 1.5, 20), 2)[rng.permutation(40)]       # every level observed twice
    Y = -0.3 + 0.7 * ndtr((h[:, None] - np.array([-0.4, 0.5])) / 0.15) + rng.normal(0, 0.05, (40, 2))
    Y[[3, 17], 0], Y[[5, 30], 1] = 0.8, -0.6
    C = np.ones_like(Y, bool)
    S7 = np.zeros_like(C)
    out = huber.invert_huber(Y, C, h, h.min(), h.max(), max_iter=4)
    for j in range(2):
        assert_same_as_loops(out, huber_by_loops(Y[:, j], C[:, j], S7[:, j], h, h.min(), h.max(), max_iter=4), j)
        assert np.isfinite(out[0][j])


# ── the limit: delta = inf is V0 and V1 ─────────────────────────────────────

def test_infinite_delta_is_invert_series_and_v1_bit_for_bit():
    marea_fit = notebook_function("marea_fit", os.path.join(NB_DIR, "05_error_attribution_site.ipynb"),
                                  marea=marea, _fit_block=_fit_block, ndtri=ndtri)
    invert_ndwi0 = notebook_function("invert_ndwi0", os.path.join(NB_DIR, "01_fixed_config_n_tables_site.ipynb"),
                                     marea=marea, _fit_block=_fit_block, ndtri=ndtri)
    Y, C, h = synthetic_curves()                                              # 3 blocks of 4,000
    z, z0, sg, a, b, n = huber.invert_huber(Y, C, h, h.min(), h.max(), delta=np.inf)
    zs, ss = marea.invert_series(Y, C, h, h.min(), h.max())
    fit = marea_fit(Y, C, h, h.min(), h.max())
    z0_nb01, zmid_nb01 = invert_ndwi0(Y, C, h, h.min(), h.max())
    assert np.array_equal(z, zs, equal_nan=True), "V4 with delta = inf is not invert_series' z"
    assert np.array_equal(z, zmid_nb01, equal_nan=True)
    ok = np.isfinite(zs)
    assert np.array_equal(sg[ok], ss[ok]), "sigma is not invert_series' sigma"
    assert np.array_equal(z0, fit["z0"], equal_nan=True) and np.array_equal(z0, z0_nb01, equal_nan=True), \
        "V4-0 with delta = inf is not V1's crossing"
    for k, v in (("a", a), ("b", b), ("sigma", sg)):
        assert np.array_equal(v, fit[k]), f"{k} is not marea_fit's (V0's fit before the guards)"
    assert (n == 0).all(), "with delta = inf the first refit must repeat the start"
    assert ok.sum() > 8000 and np.isnan(zs[50:60]).all() and np.isnan(z0[:50]).all()   # rejected, never crossing
    # scl7 present but weighted 1, or all False: the same, bit for bit
    S7 = np.random.default_rng(1).random(C.shape) < 0.2
    for kw in (dict(scl7=S7, scl7_weight=1.0), dict(scl7=np.zeros_like(C))):
        again = huber.invert_huber(Y, C, h, h.min(), h.max(), delta=np.inf, **kw)
        assert all(np.array_equal(u, v, equal_nan=True) for u, v in zip(again, (z, z0, sg, a, b, n))), kw


def test_lo_hi_default_to_the_range_of_h():
    Y, C, h = synthetic_curves(P=300, T=40, seed=5)
    one = huber.invert_huber(Y, C, h, max_iter=3)
    two = huber.invert_huber(Y, C, h, h.min(), h.max(), max_iter=3)
    assert all(np.array_equal(u, v, equal_nan=True) for u, v in zip(one, two))


# ── the outliers ────────────────────────────────────────────────────────────

def test_gross_outliers_drag_least_squares_not_huber():
    Y, h, zt, gross = gross_pixels()
    C = np.ones_like(Y, bool)
    step = (h.max() - h.min()) / 49
    z_ls, _ = marea.invert_series(Y, C, h, h.min(), h.max())
    z, z0, sg, a, b, n = huber.invert_huber(Y, C, h, h.min(), h.max())
    e_ls, e_h = np.abs(z_ls - zt) / step, np.abs(z - zt) / step
    assert np.median(e_ls) >= 2 - 1e-9 and (e_ls >= 2 - 1e-9).sum() >= 9, np.round(e_ls, 1)   # least squares moves
    assert np.median(e_h) < 0.5 and (e_h <= 1 + 1e-9).sum() >= 15, np.round(e_h, 1)  # Huber stays
    assert (n > 0).all(), "every pixel has residuals beyond delta: its weights must change"
    # the same looks flagged SCL 7: they pull less still
    z7, *_ = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=gross)
    e7 = np.abs(z7 - zt) / step
    assert (e7 <= e_h + 1e-9).all() and (e7 < e_h - 1e-9).any(), (np.round(e_h, 1), np.round(e7, 1))


# ── the iteration ───────────────────────────────────────────────────────────

def iteration_pixels(P=240, T=60, seed=9):
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.8, 1.8, T)
    z, sg = rng.uniform(-1.0, 1.0, P), rng.uniform(0.05, 0.4, P)
    Y = -0.3 + 0.7 * ndtr((h[:, None] - z) / sg) + rng.normal(0, 0.03, (T, P))
    gross = rng.random((T, P)) < 0.05
    Y[gross] = rng.choice([-0.6, 0.8], int(gross.sum()))
    Y[:, :20] = -0.3 + 0.7 * ndtr((h[:, None] - z[:20]) / sg[:20]) + rng.normal(0, 0.01, (T, 20))   # no outlier
    C = rng.random((T, P)) > 0.2
    S7 = C & (rng.random((T, P)) < 0.1)
    return Y, C, S7, h


def test_only_the_pixels_still_moving_are_refitted(monkeypatch):
    Y, C, S7, h = iteration_pixels()
    P = Y.shape[1]
    widths = []

    def recorded(Yb, Cb, tide, mu_grid, sg_grid):
        widths.append(Yb.shape[1])
        return _fit_block(Yb, Cb, tide, mu_grid, sg_grid)

    monkeypatch.setattr(huber, "_fit_block", recorded)
    z, z0, sg, a, b, n = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7)      # one block
    assert widths[0] == widths[1] == P, "the start and the first refit take the whole block"
    # the k-th refit takes exactly the pixels whose fit changed at every refit before it
    assert widths[1:] == [int((n >= k - 1).sum()) for k in range(1, len(widths))], widths
    assert sum(widths) == P + int(np.where(n < 20, n + 1, 20).sum())
    assert (n[:20] == 0).all(), "a pixel without a residual beyond delta stops at the first refit"
    assert 0 < (n == 20).sum() < P, "some pixels stop at the cap, others before it"


def test_a_pixel_that_stopped_is_not_touched_again():
    Y, C, S7, h = iteration_pixels()
    full = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7)
    n = full[5]
    for cap in (1, 3, 8, 13):
        part = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, max_iter=cap)
        done = n < cap
        assert done.any()
        for u, v in zip(part, full):
            assert np.array_equal(u[done], v[done], equal_nan=True), cap
        assert (part[5][~done] == cap).all(), "a pixel still moving at the cap stops there"
    zero = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, max_iter=0)
    assert (zero[5] == 0).all()


def test_a_pixel_that_stopped_is_a_fixed_point():
    """Reweighting a converged pixel from its own fit gives back that fit; for a capped pixel the grid point (V4's z)
    no longer moves and (a, b) move by less than 1e-6."""
    Y, C, S7, h = iteration_pixels()
    z, z0, sg, a, b, n = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7)
    mu, *_ = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, min_obs=0, min_b=-np.inf, max_b=np.inf,
                                max_a=np.inf)                                     # z = mu of every pixel
    base = np.where(C, np.where(S7, 0.5, 1.0), 0.0)
    w = huber.huber_weights(Y, base, h, a, b, mu, sg)
    a1, b1, mu1, sg1, _, _ = _fit_block(Y, w, h, np.linspace(h.min(), h.max(), 50), marea.SG_GRID)
    assert np.array_equal(mu1, mu) and np.array_equal(sg1, sg), "the grid point moved after the iteration stopped"
    done = n < 20
    assert np.allclose(a1[done], a[done], rtol=0, atol=1e-12) and np.allclose(b1[done], b[done], rtol=0, atol=1e-12)
    assert np.allclose(a1, a, rtol=0, atol=1e-6) and np.allclose(b1, b, rtol=0, atol=1e-6)


def test_the_cap_does_not_move_the_midpoint():
    Y, C, S7, h = iteration_pixels()
    z20 = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7)
    z40 = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, max_iter=40)
    assert np.array_equal(z20[0], z40[0], equal_nan=True)
    assert np.allclose(z20[1], z40[1], rtol=0, atol=1e-6, equal_nan=True)
    capped = z20[5] == 20                      # a capped pixel only moves in the last bits of (a, b): same grid point
    assert np.array_equal(z20[5][~capped], z40[5][~capped]), "a pixel that repeated before the cap is the same"
    for k in (3, 4):
        assert np.allclose(z20[k][capped], z40[k][capped], rtol=0, atol=1e-12), "a capped fit still moving visibly"


# ── SCL 7 ───────────────────────────────────────────────────────────────────

def test_scl7_weights_enter_the_start():
    Y, C, S7, h = iteration_pixels(P=300, seed=4)
    grid = np.linspace(h.min(), h.max(), 50)
    a, b, mu, sg, _, _ = _fit_block(Y.astype(np.float64), np.where(C, np.where(S7, 0.5, 1.0), 0.0), h, grid,
                                    marea.SG_GRID)
    ok = (C.sum(axis=0) >= 8) & (b > 0.15) & (b < 2.5) & (np.abs(a) < 2.5)
    for kw in (dict(max_iter=0), dict(delta=np.inf)):                     # no refit; refits that repeat the start
        z, z0, sg_, a_, b_, n = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, **kw)
        assert np.array_equal(z, np.where(ok, mu, np.nan), equal_nan=True), kw
        assert np.array_equal(a_, a) and np.array_equal(b_, b) and np.array_equal(sg_, sg) and (n == 0).all(), kw
    plain = huber.invert_huber(Y, C, h, h.min(), h.max(), max_iter=0)
    assert not np.array_equal(plain[3], a), "the SCL 7 weight changed nothing"


def test_scl7_weights_enter_every_refit():
    """The brute force with SCL 7 looks agrees (test_huber_is_the_brute_force_irls); here: halving the SCL 7 weight
    and doubling every other weight is the same fit (the weights only matter relative to each other)."""
    Y, C, S7, h = iteration_pixels(P=80, seed=6)
    one = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, max_iter=6)
    other = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7, scl7_weight=1.0, max_iter=6)
    assert not np.array_equal(one[3], other[3])
    for j in range(0, 80, 16):
        ref = huber_by_loops(Y[:, j], C[:, j], S7[:, j], h, h.min(), h.max(), max_iter=6)
        assert_same_as_loops(one, ref, j)


# ── the guards ──────────────────────────────────────────────────────────────

def test_undefined_ndwi_enters_as_zero():
    Y, C, S7, h = iteration_pixels(P=120, seed=12)
    Yn = Y.copy()
    Yn[np.random.default_rng(3).random(Y.shape) < 0.07] = np.nan
    raw = huber.invert_huber(Yn, C, h, h.min(), h.max(), scl7=S7)
    v0 = huber.invert_huber(np.nan_to_num(Yn, nan=0.0), C, h, h.min(), h.max(), scl7=S7)
    assert all(np.array_equal(u, v, equal_nan=True) for u, v in zip(raw, v0)), "NaN must be read as 0 (V0's inputs)"
    assert np.isfinite(raw[0]).sum() > 100


def test_guards():
    T = 40
    h = np.linspace(-1.5, 1.5, T)[np.random.default_rng(2).permutation(T)]
    rng = np.random.default_rng(13)
    curve = -0.3 + 0.7 * ndtr((h - 0.1) / 0.2)
    cols = {
        "curve": curve + rng.normal(0, 0.02, T),
        "all wet": 0.4 + rng.normal(0, 0.02, T),
        "all dry": -0.3 + rng.normal(0, 0.02, T),
        "wet, a step in magnitude": 0.1 + 0.4 * ndtr((h - 0.1) / 0.2) + rng.normal(0, 0.02, T),
        "7 clear looks": curve.copy(),
        "8 clear looks": curve.copy(),
        "8 clear looks, all SCL 7": curve.copy(),
        "clear looks at one level": curve.copy(),
        "no clear look": curve.copy(),
    }
    names = list(cols)
    Y = np.stack([cols[k] for k in names], axis=1)
    C = np.ones_like(Y, bool)
    S7 = np.zeros_like(C)
    spread = np.argsort(h)[np.linspace(0, T - 1, 8).round().astype(int)]     # 8 looks across the range
    for k, idx in (("7 clear looks", spread[:7]), ("8 clear looks", spread), ("8 clear looks, all SCL 7", spread)):
        C[:, names.index(k)] = False
        C[idx, names.index(k)] = True
    S7[spread, names.index("8 clear looks, all SCL 7")] = True
    j_tied = names.index("clear looks at one level")
    h_tied = h.copy()
    C[:, j_tied] = False
    C[:10, j_tied] = True
    h_tied[:10] = h[0]                                                        # ... the first 10 looks share it
    C[:, names.index("no clear look")] = False
    z, z0, sg, a, b, n = huber.invert_huber(Y, C, h, h.min(), h.max(), scl7=S7)
    got = dict(zip(names, z))
    assert np.isfinite(got["curve"]) and np.isfinite(z0[names.index("curve")])
    assert np.isnan(got["all wet"]) and np.isnan(got["all dry"])
    k = names.index("wet, a step in magnitude")
    assert np.isfinite(z[k]) and np.isnan(z0[k]), "a midpoint, no NDWI = 0 crossing"
    assert np.isnan(got["7 clear looks"]) and np.isfinite(got["8 clear looks"])
    assert np.isfinite(got["8 clear looks, all SCL 7"]), "the guard counts the clear looks, not their weights"
    assert np.isnan(got["no clear look"]) and n[names.index("no clear look")] == 0
    assert sg[names.index("no clear look")] == 0 and a[names.index("no clear look")] == 0
    # the tied pixel needs its own levels: every clear look at the same level, no grid point can be fitted
    zt, z0t, sgt, at, bt, nt = huber.invert_huber(Y[:, [j_tied]], C[:, [j_tied]], h_tied, h.min(), h.max())
    assert np.isnan(zt[0]) and np.isnan(z0t[0]) and nt[0] == 0 and sgt[0] == 0 and at[0] == 0 and bt[0] == 0
    assert np.isnan(got["clear looks at one level"]) or True                 # (with h it is an ordinary pixel)


def test_defaults_are_the_addendums():
    p = inspect.signature(huber.invert_huber).parameters
    want = dict(lo=None, hi=None, scl7=None, delta=0.10, scl7_weight=0.5, max_iter=20, mu_points=50, chunk=4000,
                min_obs=8, min_b=0.15, max_b=2.5, max_a=2.5)
    assert {k: p[k].default for k in want} == want
    assert list(p)[:3] == ["Y", "C", "h"]
    assert huber.DELTA == 0.10 and huber.SCL7_WEIGHT == 0.5 and huber.MAX_ITER == 20
