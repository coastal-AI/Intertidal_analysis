# -*- coding: utf-8 -*-
"""V12 MAREA-P (experiments/variants/depth_model.py), the depth-aware NDWI model of the addendum of 2026-10-28.

* with depth=False (c = 0 everywhere) it is _fit_block and invert_series bit for bit, V0;
* at every grid point its (a, b, c) are the bounded least squares (c >= 0) of NDWI = a + phi (b + c (h - z)), and the
  grid point kept has the least loss: checked against plain loops with scipy's bounded solver;
* on pixels whose NDWI keeps rising with depth above the true elevation it recovers z, where V0's midpoint is high;
* c >= 0 always; the guards of invert_series (N >= 8, 0.15 < b < 2.5, |a| < 2.5) hold.

Run:  python -m pytest -q tests/test_variants_depth_model.py
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import erf, ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.variants import depth_model as v12  # noqa: E402
from pyintertidal import marea  # noqa: E402
from pyintertidal.elevation import _fit_block  # noqa: E402


def curves(P=600, T=60, seed=2, depth_slope=0.0, noise=0.03):
    """NDWI = a + Phi((h - z)/sigma) (b + slope (h - z)) + noise; a few pixels with too few clear looks."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-2.0, 2.0, T)
    z, sg = rng.uniform(-1.2, 1.2, P), rng.uniform(0.05, 0.3, P)
    a, b = rng.uniform(-0.5, -0.2, P), rng.uniform(0.3, 0.7, P)
    slope = np.broadcast_to(np.asarray(depth_slope, float), (P,))
    d = h[:, None] - z
    Y = a + ndtr(d / sg) * (b + slope * d) + rng.normal(0.0, noise, (T, P))
    C = rng.random((T, P)) > 0.15
    C[:, :5] = False
    C[:5, :5] = True                                    # 5 clear looks: the guards reject
    return Y, C, h, z


def test_with_c_zero_it_is_fit_block_and_invert_series_bit_for_bit():
    Y, C, h, _ = curves(P=9000, seed=7, depth_slope=0.3)    # three blocks of 4,000 in invert_depth
    grid = np.linspace(h.min(), h.max(), 50)
    a, b, c, mu, sg, rmse, N = v12.fit_depth_block(Y[:, :500], C[:, :500].astype(float), h, grid, marea.SG_GRID,
                                                   depth=False)
    a0, b0, mu0, sg0, rmse0, N0 = _fit_block(Y[:, :500], C[:, :500].astype(float), h, grid, marea.SG_GRID)
    for u, v in ((a, a0), (b, b0), (mu, mu0), (sg, sg0), (rmse, rmse0), (N, N0)):
        assert np.array_equal(u, v)
    assert (c == 0).all()
    z, s, cc = v12.invert_depth(Y, C, h, h.min(), h.max(), depth=False)
    zs, ss = marea.invert_series(Y, C, h, h.min(), h.max())
    assert np.array_equal(z, zs, equal_nan=True) and np.array_equal(s, ss, equal_nan=True)
    assert np.array_equal(np.isfinite(cc), np.isfinite(z)) and (cc[np.isfinite(cc)] == 0).all()


def brute(Y, C, h, grid):
    """Plain loops: at every grid point the bounded least squares (c >= 0) of y = a + phi b + phi (h - mu) c over the
    clear looks; the grid point of least loss (the first on ties within 1e-12)."""
    best = None
    for mu in grid:
        for sg in marea.SG_GRID:
            phi = 0.5 * (1.0 + erf((h - mu) / (sg * np.sqrt(2.0))))
            m = C
            A = np.column_stack([np.ones(m.sum()), phi[m], phi[m] * (h[m] - mu)])
            if np.linalg.matrix_rank(A[:, :2]) < 2:
                continue
            sol = lsq_linear(A, Y[m], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]), method="bvls",
                             tol=1e-14)
            loss = float(np.sum((A @ sol.x - Y[m]) ** 2))
            if best is None or loss < best[0] - 1e-12:
                best = (loss, mu, sg, *sol.x)
    return best


def test_each_grid_point_is_the_bounded_least_squares_and_the_best_is_kept():
    Y, C, h, _ = curves(P=40, T=45, seed=11, depth_slope=np.r_[np.zeros(20), np.full(20, 0.4)])
    grid = np.linspace(h.min(), h.max(), 50)
    a, b, c, mu, sg, rmse, N = v12.fit_depth_block(Y, C.astype(float), h, grid, marea.SG_GRID)
    checked = 0
    for j in range(5, 40):                              # the first 5 have 5 clear looks: still fitted, not guarded here
        ref = brute(Y[:, j], C[:, j], h, grid)
        loss = rmse[j] ** 2 * N[j]
        assert loss <= ref[0] + 1e-9, (j, loss, ref[0])          # never worse than the brute force optimum
        if abs(loss - ref[0]) < 1e-9 and (mu[j], sg[j]) == (ref[1], ref[2]):
            assert np.allclose([a[j], b[j], c[j]], [ref[3], ref[4], ref[5]], atol=1e-6), j    # b: the step at mu
            checked += 1
    assert checked >= 30, f"only {checked} pixels at the brute force's grid point"
    assert (c >= 0).all()


def test_ndwi_rising_with_depth_moves_v0_up_but_not_v12():
    Y, C, h, z = curves(P=3000, T=80, seed=5, depth_slope=0.35, noise=0.02)
    zp, sp, cp = v12.invert_depth(Y, C, h, h.min(), h.max())
    z0, _ = marea.invert_series(Y, C, h, h.min(), h.max())
    ok = np.isfinite(zp) & np.isfinite(z0)
    e12, e0 = zp[ok] - z[ok], z0[ok] - z[ok]
    step = (h.max() - h.min()) / 49
    assert np.median(e0) > 1.5 * step, f"V0 is not biased high here ({np.median(e0):.3f} m)"
    assert abs(np.median(e12)) < 0.5 * step, f"V12 is biased ({np.median(e12):.3f} m)"
    assert np.sqrt(np.mean(e12 ** 2)) < np.sqrt(np.mean(e0 ** 2))
    assert np.median(cp[ok]) > 0.15, "the depth slope is not found"


def test_without_a_rise_with_depth_it_stays_near_v0():
    Y, C, h, z = curves(P=2000, T=80, seed=6, depth_slope=0.0, noise=0.02)
    zp, _, cp = v12.invert_depth(Y, C, h, h.min(), h.max())
    z0, _ = marea.invert_series(Y, C, h, h.min(), h.max())
    ok = np.isfinite(zp) & np.isfinite(z0)
    step = (h.max() - h.min()) / 49
    assert np.median(np.abs(zp[ok] - z0[ok])) <= step + 1e-9
    assert np.median(cp[ok]) < 0.05


def test_guards_and_c_nonnegative():
    Y, C, h, _ = curves(P=300, seed=8, depth_slope=0.2)
    z, s, c = v12.invert_depth(Y, C, h, h.min(), h.max())
    assert np.isnan(z[:5]).all(), "5 clear looks: rejected"
    fin = np.isfinite(z)
    assert fin[5:].mean() > 0.9 and (c[fin] >= 0).all()
    assert np.array_equal(np.isfinite(s), fin) and np.array_equal(np.isfinite(c), fin)
    flat = np.full_like(Y, -0.2)                         # no transition: b inside the guards nowhere
    zf, _, _ = v12.invert_depth(flat, C, h, h.min(), h.max())
    assert np.isnan(zf).all()


def test_time_at_a_realistic_size():
    Y, C, h, _ = curves(P=8000, T=125, seed=3, depth_slope=0.2)
    t0 = time.perf_counter()
    v12.invert_depth(Y, C, h, h.min(), h.max())
    sec = time.perf_counter() - t0
    print(f"V12: 8,000 pixels x 125 looks in {sec:.1f} s (x20 for the Wadden's 160,000: {20 * sec / 60:.1f} min)")
    assert sec < 120
