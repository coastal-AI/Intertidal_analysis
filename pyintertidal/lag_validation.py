# -*- coding: utf-8 -*-
"""Validation of the interior tidal lag against two level records.

The same three computations serve the external validation (a tide gauge at
the mouth and one inside the estuary) and the on-site validation (pressure
sensors along the ría, converted to level with :func:`pressure_to_level`):

* :func:`level_crossing_lags` — for a given water level and limb (rising or
  falling), the time the tide takes to reach that level at the inner
  station after reaching it at the mouth station, cycle by cycle;
* :func:`same_instant_difference` — the difference in level between the two
  stations at the same instant, i.e. the error made by assuming one
  uniform level over the estuary;
* :func:`gauge_series` — the cleaned, demeaned, regularly sampled record a
  cached IOC station provides (gaps longer than ``max_gap`` stay gaps).

Both records are demeaned before anything else: their zeros are different
datums, and the datum is not what is being validated here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gauges

RHO_SEAWATER = 1025.0
G = 9.80665


def pressure_to_level(p_dbar, p_atm_dbar, rho=RHO_SEAWATER, z_sensor_m=0.0):
    """Water level above the sensor datum from absolute pressure.

    ``p_dbar`` is the sensor's absolute pressure, ``p_atm_dbar`` the
    atmospheric pressure recorded on land at the same instants (both in
    decibar), ``rho`` the water density and ``z_sensor_m`` the surveyed
    elevation of the sensor. 1 dbar = 10 000 Pa.
    """
    p = (np.asarray(p_dbar, float) - np.asarray(p_atm_dbar, float)) * 1e4
    return z_sensor_m + p / (rho * G)


def regularise(times, levels, freq="10min", max_gap="1h", demean=True):
    """A level record on a regular grid; gaps longer than ``max_gap`` stay
    NaN instead of being bridged; optionally demeaned."""
    s = pd.Series(np.asarray(levels, float), index=pd.DatetimeIndex(times))
    s = s[~s.index.duplicated()].sort_index().dropna()
    grid = pd.date_range(s.index.min().floor(freq), s.index.max().ceil(freq), freq=freq)
    r = s.reindex(s.index.union(grid)).interpolate(method="time", limit_area="inside")
    r = r.reindex(grid)
    x = s.index.values.astype("datetime64[s]").astype("int64")
    q = grid.values.astype("datetime64[s]").astype("int64")
    j = np.searchsorted(x, q)
    j0, j1 = np.clip(j - 1, 0, len(x) - 1), np.clip(j, 0, len(x) - 1)
    gap = np.minimum(np.abs(q - x[j0]), np.abs(x[j1] - q))
    r[gap > pd.Timedelta(max_gap).total_seconds()] = np.nan
    if demean:
        r = r - r.mean()
    return r


def gauge_series(code, cache_dir="data_v4/gauges", freq="10min", max_gap="1h"):
    """Cleaned (``gauges.despike``), regular, demeaned record of one cached
    IOC station."""
    df = gauges.load_cached_ioc(code, cache_dir)
    if df is None:
        raise FileNotFoundError(f"no cached IOC record for {code!r} in {cache_dir}")
    df = gauges.despike(df)
    return regularise(df["time"].values, df["level_m"].values, freq, max_gap)


def _crossings(s, level, limb):
    """Times (seconds since epoch) at which ``s`` crosses ``level`` on the
    given limb, linearly interpolated between samples."""
    v = s.values.astype(float)
    t = s.index.values.astype("datetime64[s]").astype("int64").astype(float)
    d = v - level
    if limb == "rising":
        idx = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    elif limb == "falling":
        idx = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0))
    else:
        raise ValueError(limb)
    ok = np.isfinite(d[idx]) & np.isfinite(d[idx + 1])
    idx = idx[ok]
    frac = d[idx] / (d[idx] - d[idx + 1])
    return t[idx] + frac * (t[idx + 1] - t[idx])


def level_crossing_lags(mouth, inner, levels, limbs=("rising", "falling"),
                        max_lag_h=3.0):
    """Lag (minutes) between the mouth and the inner station in reaching
    each ``level`` on each limb, one value per tidal cycle.

    Returns ``(table, per_cycle)``: ``table`` has one row per (level, limb)
    with n, mean, median and standard deviation of the lag; ``per_cycle``
    maps (level, limb) to the array of individual lags. A mouth crossing is
    paired with the nearest inner crossing within ``max_lag_h`` hours; an
    unpaired crossing (a gap in either record) is dropped.
    """
    rows, per_cycle = [], {}
    for level in levels:
        for limb in limbs:
            ta, tb = _crossings(mouth, level, limb), _crossings(inner, level, limb)
            if len(ta) == 0 or len(tb) == 0:
                continue
            j = np.searchsorted(tb, ta)
            j0, j1 = np.clip(j - 1, 0, len(tb) - 1), np.clip(j, 0, len(tb) - 1)
            cand = np.where(np.abs(tb[j1] - ta) < np.abs(tb[j0] - ta), tb[j1], tb[j0])
            lag = (cand - ta) / 60.0
            lag = lag[np.abs(lag) <= max_lag_h * 60.0]
            per_cycle[(float(level), limb)] = lag
            rows.append({"level_m": float(level), "limb": limb, "n": int(len(lag)),
                         "lag_mean_min": float(np.mean(lag)) if len(lag) else np.nan,
                         "lag_median_min": float(np.median(lag)) if len(lag) else np.nan,
                         "lag_std_min": float(np.std(lag)) if len(lag) else np.nan})
    return pd.DataFrame(rows), per_cycle


def same_instant_difference(mouth, inner):
    """Inner minus mouth level at the same instants (both demeaned).

    Returns the aligned difference series and a summary: n, mean, RMS,
    standard deviation and the 5th/95th percentiles, in metres.
    """
    both = pd.concat([mouth.rename("mouth"), inner.rename("inner")], axis=1).dropna()
    d = both["inner"] - both["mouth"]
    return d, {"n": int(len(d)), "mean_m": float(d.mean()),
               "rms_m": float(np.sqrt(np.mean(d.values ** 2))),
               "std_m": float(d.std()), "p05_m": float(d.quantile(0.05)),
               "p95_m": float(d.quantile(0.95))}
