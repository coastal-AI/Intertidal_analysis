# -*- coding: utf-8 -*-
"""MAREA's variants of the addendum of 2026-10-28 (experiments/variants_addendum_2026-10-28.md), as plain functions
on the per-pixel inputs of the attribution notebook (notebooks/validation/_build_05_attribution.py): Y (looks x
pixels) NDWI with NaN where undefined, C the clear looks, h the levels read with MAREA's delay.

monotone     V11, the isotonic wet-probability read-off
depth_model  V12, the depth-aware NDWI model
interval     V13, MAREA-B's likelihood interval, the abstention rule, the error-coverage curves and the block bootstrap
otsu         V14, the per-scene water threshold
huber        V4, the robust fit
"""
