# -*- coding: utf-8 -*-
"""Shared set-up of the tests: the repository root on ``sys.path`` and the ``slow`` marker.

The package is used from the repository root (``marea.reconstruct`` reads ``configs/m2.yaml`` next to the package),
so the tests import it from here rather than from an installed copy.

    python -X utf8 -m pytest -q -p no:cacheprovider tests/                  # everything (about 2 minutes)
    python -X utf8 -m pytest -q -p no:cacheprovider -m "not slow" tests/    # without the offline reproduction
"""
import os  # paths
import sys  # the import path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # `import pyintertidal` finds the package of this repository


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: the offline reproduction of the Villaviciosa product (about 1.5 min)")
