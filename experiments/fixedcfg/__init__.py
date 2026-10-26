"""Fixed-configuration validation tables (first author's design, 2026-10-20).

One configuration external to the methods (MAREA's production scene
screening, per-pixel clear mask SCL 4-7, NDWI, wet = NDWI > 0, the boundary
tide h_B as in production), a window of the n most recent screened scenes per
site, and one column per elevation method. The method registry is
:mod:`experiments.fixedcfg.methods`.
"""
