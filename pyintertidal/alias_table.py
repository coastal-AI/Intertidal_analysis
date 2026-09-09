"""
alias_table.py — What a sun-synchronous archive can and cannot see (M2 pre)
===========================================================================

Hard limit, arithmetic not opinion: Sentinel-2 crosses at a fixed solar hour,
so a constituent whose period divides the solar day is observed at ONE phase
forever. No archive length recovers it. Every interior-tide estimator in
phase M2 must take its constituent list from here, and the unobservable ones
stay pinned to the boundary prior (plan v4, fase M2).

The computation lives in :func:`pyintertidal.boundary.alias_periods`; this
module adds the DECISION — which constituents are estimable given the real
overpass-hour spread over the site — and a sealed-friendly report.

Measured for Villaviciosa at adoption (real STAC hours, 11:21 UTC ± 10 min):
M2 (alias 14.8 d), N2 (9.6 d), O1 (14.2 d), Q1 (9.4 d), M4 (7.4 d) and
M6 (4.9 d) are estimable with a 3-10 year archive; K2 (182.6 d) is hard;
K1 and P1 (365.2 d) are confounded with the seasonal cycle; **S2 is frozen**.
"""

from __future__ import annotations

import numpy as np

from .boundary import PERIODS_H, alias_periods


def decide(overpass_hours_utc, max_alias_days=100.0):
    """Estimable constituents for a site, from its REAL overpass hours.

    ``overpass_hours_utc``: array of acquisition hours (float, UTC). The
    hour SPREAD matters: a perfectly fixed hour gives the pure alias table;
    real spread (~±10 min for S2) does not rescue S2 — a 20-minute jitter on
    a 12 h constituent samples ~1 % of its phase circle.

    Returns dict: constituent -> {"alias_days", "estimable", "reason"}.
    """
    hours = np.asarray(overpass_hours_utc, float)
    al = alias_periods(samples_per_day=1.0)
    spread_min = float((np.percentile(hours, 95)
                        - np.percentile(hours, 5)) * 60.0)
    out = {}
    for k, days in al.items():
        if not np.isfinite(days):
            est, why = False, ("frozen: its period divides the solar day; "
                               f"the {spread_min:.0f}-min jitter of the "
                               "real overpass time does not rescue it")
        elif days > 300:
            est, why = False, "~annual alias: confounded with seasonality"
        elif days > max_alias_days:
            est, why = False, f"alias {days:.0f} d: beyond the threshold"
        else:
            est, why = True, f"alias {days:.1f} d, well sampled"
        out[k] = {"alias_days": days, "estimable": est, "reason": why,
                  "period_h": PERIODS_H[k]}
    return out


def estimable(overpass_hours_utc, max_alias_days=100.0):
    """Just the names, sorted by period — what M2 estimators may fit."""
    d = decide(overpass_hours_utc, max_alias_days)
    return sorted((k for k, v in d.items() if v["estimable"]),
                  key=lambda k: PERIODS_H[k])
