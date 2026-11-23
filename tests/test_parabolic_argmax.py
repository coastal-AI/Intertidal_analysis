# -*- coding: utf-8 -*-
"""Sub-grid argmax of the lag scores (m2b, m2c, m2a_hysteresis, p9b)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyintertidal import tide_estimators as te  # noqa: E402


def test_vertex_of_an_exact_parabola():
    grid = np.arange(-45.0, 181.0, 5.0)
    for peak in (-12.0, 0.0, 3.0, 12.0, 58.2, 132.6):
        score = -(grid - peak) ** 2
        assert abs(te._parabolic_argmax(score, grid) - peak) < 1e-9


def test_refinement_moves_towards_the_higher_neighbour():
    grid = np.array([0.0, 10.0, 20.0])
    assert te._parabolic_argmax(np.array([0.0, 1.0, 0.8]), grid) > 10.0
    assert te._parabolic_argmax(np.array([0.8, 1.0, 0.0]), grid) < 10.0


def test_edge_and_degenerate_cases_keep_the_grid_point():
    grid = np.array([0.0, 10.0, 20.0, 30.0])
    assert te._parabolic_argmax(np.array([1.0, 0.5, 0.2, 0.1]), grid) == 0.0
    assert np.isnan(te._parabolic_argmax(np.array([np.nan, 1.0, np.nan, np.nan]), grid))
