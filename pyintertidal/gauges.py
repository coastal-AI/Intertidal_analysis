"""
gauges.py — Tide gauges: validating the tide model
===================================================

Every elevation this package produces is expressed in the datum of a GLOBAL
tide model, so a systematic error in that model becomes a systematic error in
the DEM. This module compares the model against a real tide gauge, which is
the only way to know how large that error is at your site.

Typical use: download the record of the nearest gauge (in Spain, PORTUS /
REDMAR from Puertos del Estado), load it here, predict the same instants with
a tide model (a :class:`pyintertidal.boundary.PyTMDBoundary`; see
:func:`compare_with_model` for how to pair them in this edition), and look at
the residuals.

How it fits: :mod:`pyintertidal.boundary` uses :func:`load_gauge` (a gauge
record as the tide at the mouth, ``GaugeBoundary``), :func:`despike` and
:func:`load_cached_ioc` (IOC gauges, ``make_boundary``). Records are tidy
DataFrames with ``time`` (UTC) and ``level_m`` (metres).

What the residuals tell you
---------------------------
* a constant offset → a DATUM difference (harmless for morphology, but it
  must be removed before comparing with an independent elevation model);
* a periodic residual → the harmonic model is missing or mis-phasing a
  constituent at this site (common inside estuaries);
* spikes → storm surge, which a harmonic model cannot represent by design.
"""

from __future__ import annotations

import os

import numpy as np


def load_gauge(path, time_column="fecha", level_column="nivel_m", skiprows=0,
               sheet=0):
    """Load a tide-gauge record from CSV or Excel into a DataFrame.

    Column names default to the Spanish PORTUS/REDMAR export
    (``fecha``/``nivel_m``); pass your own for other providers. The result is
    normalised to two columns, ``time`` (datetime) and ``level_m`` (float),
    sorted and free of missing values.

    Parameters
    ----------
    path : str
        A ``.csv``, ``.xlsx`` or ``.xls`` file.
    time_column, level_column : str
        Column names of the instants and of the level (metres).
    skiprows : int
        Lines to skip before the header (e.g. comment lines).
    sheet : int or str
        Excel sheet.

    Raises ``KeyError`` naming the columns found when either column is
    missing. Times are taken as written: give them in UTC.
    """
    import pandas as pd

    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, skiprows=skiprows, sheet_name=sheet)
    else:
        df = pd.read_csv(path, skiprows=skiprows)

    if time_column not in df.columns or level_column not in df.columns:
        raise KeyError(f"expected columns {time_column!r} and {level_column!r}; "
                       f"file has {list(df.columns)}")
    out = pd.DataFrame({
        "time": pd.to_datetime(df[time_column]),
        "level_m": pd.to_numeric(df[level_column], errors="coerce"),
    }).dropna().sort_values("time").reset_index(drop=True)
    return out


def compare_with_model(gauge, tide_service, aoi, verbose=True):
    """Predict the gauge's timestamps with a tide model and pair them up.

    Not usable with this edition alone: it needs the ``TideService`` of the
    full development library (an object with ``_tide_model()`` and
    ``_point(aoi)``). With this edition, pair a gauge with a boundary by
    hand — :func:`load_gauge` returns the times sorted, and
    ``PyTMDBoundary.levels`` returns its levels in time order::

        g = load_gauge("gauge.csv", time_column="time", level_column="level_m")
        model = PyTMDBoundary("EOT20", lat, lon, directory="tide_models")
        cmp = pd.DataFrame({"time": g["time"], "observed_m": g["level_m"],
                            "modelled_m": model.levels(g["time"])})
        gauge_metrics(cmp)

    Parameters
    ----------
    gauge : DataFrame
        Output of :func:`load_gauge`.
    tide_service : object
        A tide service with ``_tide_model()`` and ``_point(aoi)``: the
        ``TideService`` of the full development library, which is not part
        of this edition.
    aoi : AOI
        Study area — its centroid (or the service's explicit location) is the
        prediction point.
    verbose : bool
        Passed to the service's model loader.

    Returns a DataFrame with ``time``, ``observed_m``, ``modelled_m`` and
    ``residual_m`` (observed − modelled).
    """
    import pandas as pd

    times = list(gauge["time"])
    model = tide_service._tide_model(verbose=verbose)
    lat, lon = tide_service._point(aoi)
    modelled = model.get_tide_heights_batch(lat, lon, times)
    df = pd.DataFrame({
        "time": times,
        "observed_m": gauge["level_m"].to_numpy(dtype=float),
        "modelled_m": np.asarray(modelled, dtype=float),
    })
    df["residual_m"] = df["observed_m"] - df["modelled_m"]
    return df.dropna()


def gauge_metrics(comparison, remove_offset=True):
    """Error metrics of a model-vs-gauge comparison.

    With ``remove_offset`` (default) the median residual — the datum
    difference — is removed before computing RMSE/MAE, so the numbers
    describe how well the model reproduces the tide's SHAPE rather than
    which zero it is referenced to. The offset itself is always reported.

    ``comparison`` is a DataFrame with ``observed_m`` and ``modelled_m``
    (metres). Sign: ``datum_offset_m`` is median(modelled − observed), the
    opposite of :func:`compare_with_model`'s ``residual_m``. With fewer than
    3 valid pairs only ``{"n": ...}`` is returned.
    """
    from scipy.stats import pearsonr

    obs = np.asarray(comparison["observed_m"], dtype=float)
    mod = np.asarray(comparison["modelled_m"], dtype=float)
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() < 3:
        return {"n": int(m.sum())}
    diff = mod[m] - obs[m]
    offset = float(np.median(diff))
    d = diff - offset if remove_offset else diff
    return {
        "n": int(m.sum()),
        "datum_offset_m": round(offset, 3),
        "rmse_m": round(float(np.sqrt((d ** 2).mean())), 3),
        "mae_m": round(float(np.abs(d).mean()), 3),
        "max_abs_error_m": round(float(np.abs(d).max()), 3),
        "pearson_r": round(float(pearsonr(obs[m], mod[m])[0]), 4),
        "observed_range_m": round(float(obs[m].max() - obs[m].min()), 2),
    }


def compare_models(gauge, services, aoi, verbose=False):
    """Score several tide models against the same gauge record.

    ``services`` is ``{label: TideService}``. Returns a list of metric dicts
    (one per model, each with a ``model`` key) — ready for a DataFrame and a
    table in the paper.

    Needs the ``TideService`` of the full development library, like
    :func:`compare_with_model`; ``aoi`` and ``verbose`` are passed to it.
    """
    rows = []
    for label, service in services.items():
        comparison = compare_with_model(gauge, service, aoi, verbose=verbose)
        metrics = gauge_metrics(comparison)
        metrics["model"] = label
        rows.append(metrics)
    return rows


def despike(df, window="61min", max_dev_m=0.75, max_abs_m=6.0,
            harmonic_cap_m=2.5):
    """Drop what an IOC record carries that is not sea level.

    Two failure modes, two rules:

    * **spikes** (-10 m sentinels for a few samples, 0.1-0.2 % of the
      Scheldt records): a sample is out when it sits more than
      ``max_dev_m`` from the rolling median of its ``window`` (time-based,
      so irregular sampling is fine) or more than ``max_abs_m`` from the
      record's median;
    * **plateaus** (a stuck sensor: Ferrol1 wrote +4 m for days in 2023),
      invisible to a rolling median: a sample is out when it sits more
      than ``harmonic_cap_m`` from the record's own harmonic fit — a cap
      on surge, which in these seas stays well inside 2.5 m. The fit is
      done twice, the second time without the first pass's outliers.

    ``df`` is a tidy frame (``time``, ``level_m`` in metres), as
    :func:`load_gauge` or :func:`load_cached_ioc` return it. The harmonic
    step runs only when more than 5,000 samples pass the spike rule; shorter
    records get the spike rule alone. ``harmonic_cap_m=None`` skips it.

    Returns the tidy frame (time, level_m) without those rows."""
    import pandas as pd

    from .boundary import PERIODS_H, hours_since_epoch

    s = pd.Series(df["level_m"].to_numpy(float),
                  index=pd.DatetimeIndex(df["time"]))
    s = s[~s.index.duplicated()].sort_index()
    x = s - s.median()
    rm = x.rolling(window, center=True).median()
    ok = ((x - rm).abs() <= max_dev_m) & (x.abs() <= max_abs_m)
    ok = ok.to_numpy()
    if harmonic_cap_m is not None and ok.sum() > 5000:
        t_h = hours_since_epoch(s.index)
        y = x.to_numpy()
        cols = [np.ones_like(t_h)]
        for n in PERIODS_H:
            w = 2 * np.pi / PERIODS_H[n]
            cols += [np.cos(w * t_h), np.sin(w * t_h)]
        A = np.column_stack(cols)
        for _ in range(2):
            fit = ok.copy()
            step = max(1, int(fit.sum() // 200000))     # ~200k rows suffice
            idx = np.flatnonzero(fit)[::step]
            coef = np.linalg.lstsq(A[idx], y[idx], rcond=None)[0]
            resid = y - A @ coef
            ok = ok & (np.abs(resid) <= harmonic_cap_m)
    return pd.DataFrame({"time": s.index[ok], "level_m": s.to_numpy()[ok]})


def load_cached_ioc(code, cache_dir="data_v4/gauges"):
    """One cached IOC sea-level record as a tidy frame (time, level_m).

    :func:`pyintertidal.boundary.fetch_ioc` caches the raw IOC JSON of a
    station as ``ioc_<code>_<t0>_<t1>.json`` in ``cache_dir``; this reads
    EVERY cached file of that station, whatever its period, and drops
    duplicate times. Keeps the majority sensor when a station reports
    several. Returns None when the code has no cached chunks.

    ``code`` is the IOC station code (e.g. ``"vlis"``); ``cache_dir`` is
    relative to the working directory unless absolute.
    """
    import glob
    import json

    import pandas as pd

    raw = []
    for h in sorted(glob.glob(os.path.join(cache_dir, f"ioc_{code}_*.json"))):
        raw += json.load(open(h))
    if not raw:
        return None
    df = pd.DataFrame(raw)
    if "sensor" in df:
        df = df[df["sensor"] == df["sensor"].value_counts().idxmax()]
    return (pd.DataFrame({"time": pd.to_datetime(df["stime"]),
                          "level_m": pd.to_numeric(df["slevel"],
                                                   errors="coerce")})
            .dropna().sort_values("time").drop_duplicates("time"))
