# -*- coding: utf-8 -*-
"""V15 MAREA-S (experiments/variants/single_pass.py): the delay fit's own profiled z.

* its maximum log-likelihood is the one the delay fit adds up: on the same arrays, -sum(max_ll)/sum(n) over the pixels
  with at least 6 clear looks is `tide_estimators._band_nll`'s score, so its z is the argmax of the delay fit's
  likelihood, not of an approximation;
* z is the median of the tied levels (brute force by plain loops), sigma the best atom there;
* the guards: at least 6 clear looks with a wet and a dry one;
* on synthetic pixels with a known elevation it recovers it.

Run:  python -m pytest -q tests/test_variants_single_pass.py
"""
import math
import os
import sys

import numpy as np
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.variants import single_pass as v15  # noqa: E402
from pyintertidal import tide_estimators as te  # noqa: E402

SG = (0.06, 0.10, 0.15, 0.22, 0.32, 0.45)           # configs/m2.yaml sigma_perfil_m


def pixels(P=500, T=70, seed=4):
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.8, 1.8, T)
    z, sg = rng.uniform(-1.2, 1.2, P), rng.uniform(0.05, 0.35, P)
    wet = rng.random((T, P)) < ndtr((h[:, None] - z) / sg)
    Y = np.where(wet, rng.uniform(0.05, 0.6, (T, P)), rng.uniform(-0.6, -0.05, (T, P)))
    C = rng.random((T, P)) > 0.2
    Y[rng.random((T, P)) < 0.02] = np.nan               # an undefined NDWI: dry
    C[:, :4] = False
    C[:5, :4] = True                                    # 5 clear looks: below the guard
    grid = np.linspace(h.min() - 0.3, h.max() + 0.3, 60)
    return Y, C, h, z, grid


def test_its_maximum_is_the_delay_fits_likelihood():
    Y, C, h, _, grid = pixels()
    z, s, mx = v15.invert_single(Y, C, h, grid, SG)
    with np.errstate(invalid="ignore"):
        wet = C & (Y > 0)
    nll, npx = te._band_nll(wet, C, h, grid, 0.2, sg_grid=SG)
    n = C.sum(axis=0)
    ok = n >= 6
    assert npx == ok.sum()
    assert np.isclose(-mx[ok].sum() / n[ok].sum(), nll, rtol=1e-12, atol=0), (-mx[ok].sum() / n[ok].sum(), nll)


def brute(wet, clear, h, grid, tie_tol=1e-9):
    """Plain loops: the profiled log-likelihood on the grid, the median of its tied levels."""
    prof = []
    for zk in grid:
        best = -np.inf
        for s0 in SG:
            ll = 0.0
            for k in range(len(h)):
                if clear[k]:
                    p = min(max(0.5 * (1 + math.erf((h[k] - zk) / s0 / math.sqrt(2))), 1e-4), 1 - 1e-4)
                    ll += math.log(p) if wet[k] else math.log1p(-p)
            best = max(best, ll)
        prof.append(best)
    prof = np.array(prof)
    return float(np.median(grid[prof >= prof.max() - tie_tol]))


def test_z_is_the_median_of_the_tied_levels():
    Y, C, h, _, grid = pixels(P=12, T=30, seed=9)
    z, _, _ = v15.invert_single(Y, C, h, grid, SG)
    with np.errstate(invalid="ignore"):
        wet = C & (Y > 0)
    for j in range(4, 12):
        if wet[:, j].any() and (C[:, j] & ~wet[:, j]).any():
            assert np.isclose(z[j], brute(wet[:, j], C[:, j], h, grid), atol=1e-12), j


def test_guards():
    Y, C, h, _, grid = pixels(P=60)
    z, s, mx = v15.invert_single(Y, C, h, grid, SG)
    assert np.isnan(z[:4]).all() and np.isnan(s[:4]).all(), "5 clear looks: rejected"
    assert np.isfinite(mx).all()
    allwet = np.where(C, 0.5, np.nan)
    assert np.isnan(v15.invert_single(allwet, C, h, grid, SG)[0]).all(), "no dry look: rejected"


def test_it_recovers_a_known_elevation():
    Y, C, h, z_true, grid = pixels(P=3000, T=100, seed=2)
    z, _, _ = v15.invert_single(Y, C, h, grid, SG)
    ok = np.isfinite(z)
    step = grid[1] - grid[0]
    assert ok.mean() > 0.95
    assert abs(np.median(z[ok] - z_true[ok])) < step
    assert np.sqrt(np.mean((z[ok] - z_true[ok]) ** 2)) < 0.15
