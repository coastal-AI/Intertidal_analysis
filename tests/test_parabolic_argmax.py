# -*- coding: utf-8 -*-
"""Sub-grid argmax of the lag scores: ``tide_estimators._parabolic_argmax``.

It refines the best delay of a grid of candidate delays by a parabola through
the three best points. The development estimators use it (m2b_concordance,
m2c_waterline and the deprecated m2a_hysteresis); MAREA's own fit
(m2a_rasch) does not.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # also done by conftest.py; kept so this file runs on its own

from pyintertidal import tide_estimators as te  # noqa: E402


def test_vertex_of_an_exact_parabola():
    """An exact parabola is recovered at its vertex, between grid points too."""
    grid = np.arange(-45.0, 181.0, 5.0)
    for peak in (-12.0, 0.0, 3.0, 12.0, 58.2, 132.6):
        score = -(grid - peak) ** 2
        assert abs(te._parabolic_argmax(score, grid) - peak) < 1e-9


def test_refinement_moves_towards_the_higher_neighbour():
    """The refined delay moves from the best grid point towards the better neighbour."""
    grid = np.array([0.0, 10.0, 20.0])
    assert te._parabolic_argmax(np.array([0.0, 1.0, 0.8]), grid) > 10.0
    assert te._parabolic_argmax(np.array([0.8, 1.0, 0.0]), grid) < 10.0


def test_edge_and_degenerate_cases_keep_the_grid_point():
    """A best point at the grid's end stays there; fewer than 3 finite scores give NaN."""
    grid = np.array([0.0, 10.0, 20.0, 30.0])
    assert te._parabolic_argmax(np.array([1.0, 0.5, 0.2, 0.1]), grid) == 0.0
    assert np.isnan(te._parabolic_argmax(np.array([np.nan, 1.0, np.nan, np.nan]), grid))
