"""
gauges.py — Tide gauges: validating the tide model
===================================================

Every elevation this package produces is expressed in the datum of a GLOBAL
tide model, so a systematic error in that model becomes a systematic error in
the DEM. This module compares the model against a real tide gauge, which is
the only way to know how large that error is at your site.

Typical use: download the record of the nearest gauge (in Spain, PORTUS /
REDMAR from Puertos del Estado), load it here, predict the same instants with
:class:`pyintertidal.tides.TideService`, and look at the residuals.

What the residuals tell you
---------------------------
* a constant offset → a DATUM difference (harmless for morphology, but it
  must be removed before comparing with LiDAR — see
  :func:`pyintertidal.validation.compare_dems`);
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

    Parameters
    ----------
    gauge : DataFrame
        Output of :func:`load_gauge`.
    tide_service : TideService
        Configured model (:mod:`pyintertidal.tides`).
    aoi : AOI
        Study area — its centroid (or the service's explicit location) is the
        prediction point.

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
    """
    rows = []
    for label, service in services.items():
        comparison = compare_with_model(gauge, service, aoi, verbose=verbose)
        metrics = gauge_metrics(comparison)
        metrics["model"] = label
        rows.append(metrics)
    return rows


def load_cached_ioc(code, cache_dir="data_v4/gauges"):
    """One cached IOC sea-level record as a tidy frame (time, level_m).

    The campaign caches raw IOC JSON chunks per station code; this is the
    single loader every comparison uses (three near-copies were folded here
    at delivery time). Keeps the majority sensor when a station reports
    several. Returns None when the code has no cached chunks.
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
