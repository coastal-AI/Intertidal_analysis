# -*- coding: utf-8 -*-
"""Shared set-up of the tests: the repository root on ``sys.path`` and the ``slow`` marker.

The package is used from the repository root (``marea.reconstruct`` reads ``configs/m2.yaml`` next to the package),
so the tests import it from here rather than from an installed copy.

    python -X utf8 -m pytest -q -p no:cacheprovider tests/                  # everything (about 2 minutes)
    python -X utf8 -m pytest -q -p no:cacheprovider -m "not slow" tests/    # without the offline reproduction

What the tests cover: test_publication_guard (nothing private in the repository; run it before every commit),
test_reproduce_villaviciosa (the offline kit reproduces products_marea/, slow), test_sample_cube (the notebook
sample reproduces the product's elevations; skipped without data/villaviciosa_sample.nc), test_quality_rule,
test_overpass_corrections and test_parabolic_argmax (small units, synthetic or tiny inputs).
"""
import os  # paths
import sys  # the import path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # `import pyintertidal` finds the package of this repository


def pytest_configure(config):
    """Register the ``slow`` marker, so ``-m "not slow"`` can deselect it."""
    config.addinivalue_line("markers", "slow: the offline reproduction of the Villaviciosa product (about 1.5 min)")
