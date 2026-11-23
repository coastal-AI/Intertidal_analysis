# -*- coding: utf-8 -*-
"""Per-pixel quality rule of the elevation inversion (``marea.QUALITY_MIN_BRACKET``, added 2026-10-06).

An elevation is kept only when at least N clear looks were taken with the (delayed) tide below it and N
with the tide above it, i.e. when it is interpolated from the pixel's own record, not pinned at an end of it.
``marea.invert_series`` applies the rule through ``min_bracket``; ``marea.reconstruct`` through
``quality_min_bracket``. Synthetic pixels only: no data file is read.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # also done by conftest.py; kept so this file runs on its own

from pyintertidal import marea  # noqa: E402


def test_quality_rule_drops_unbracketed_elevations():
    """A pixel near the top of the tide range loses its elevation; a mid-range one keeps it unchanged."""
    rng = np.random.default_rng(0)
    h = rng.uniform(-1, 1, 120)
    z_true = np.array([0.0, 0.97])                  # 0.97: only ~2 scenes with the level above it
    Y = np.where(h[:, None] > z_true[None, :], 0.4, -0.3) + rng.normal(0, 0.05, (120, 2))
    C = np.ones((120, 2))
    z0, _ = marea.invert_series(Y, C, h, -1, 1)
    z3, _ = marea.invert_series(Y, C, h, -1, 1, min_bracket=3)
    assert np.isfinite(z0).all()
    assert np.isfinite(z3[0]) and np.isnan(z3[1])
    assert abs(z3[0] - z0[0]) < 1e-12                  # kept pixels are untouched


def test_quality_rule_counts_only_clear_observations():
    """Cloudy looks do not count towards the N looks on each side."""
    h = np.linspace(-1, 1, 40)
    Y = np.where(h[:, None] > 0.0, 0.4, -0.3)
    C = np.ones((40, 1))
    C[h < -0.2] = 0                                    # every observation below -0.2 m is cloudy
    z, _ = marea.invert_series(Y, C, h, -1, 1, min_bracket=3)
    assert np.isfinite(z[0])                           # still >= 3 clear levels on each side of 0
    C[h < 0.0] = 0                                     # now none below the elevation
    z, _ = marea.invert_series(Y, C, h, -1, 1, min_bracket=3)
    assert np.isnan(z[0])


def test_default_is_the_historical_behaviour():
    """invert_series keeps the historical default (0: no rule); reconstruct applies the rule by default (3)."""
    import inspect
    assert inspect.signature(marea.invert_series).parameters["min_bracket"].default == 0
    assert marea.QUALITY_MIN_BRACKET == 3
    assert inspect.signature(marea.reconstruct).parameters["quality_min_bracket"].default == 3
