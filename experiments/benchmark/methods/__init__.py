"""Validation v2 — the elevation-method adapters (spec §3.2, §12.1).

One module per elevation method; milestone V2 holds only ``elev_marea_pair``
(the E0 unit: no delay E0a, MAREA E0b, the L01 lag product and the
gauge-lag attribution row E0g), which the lag registry
(``experiments.benchmark.lags.REGISTRY['E0']``) runs in Stage L. Milestone V3
adds the elevation registry and the cards of E1-E8 here.

Nothing is imported at package import time.
"""
