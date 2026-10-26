# -*- coding: utf-8 -*-
"""The window's full level series and its harmonic analysis.

Flood2Topo (E5a), Tseng (E5b) and Coast X-Ray (E7) need the water level over
the whole window, not only at the scene instants: Flood2Topo maps the water
frequency onto the exceedance curve of the full series, Tseng scales between
its lowest and highest level, and Coast X-Ray reads MHWS and MLWS from a
harmonic analysis of it (docs/validation_v2_spec.md s3.2.1). Under the common
conditions the level is the production boundary tide h_B WITHOUT lag, at a
1-min step.

Convention (the runner's choice, recorded in every meta): the series of a
window runs from 00:00 of the window's first scene day to the end of its last
day (end exclusive), unless the runner passes the window's own bounds, e.g.
``end="2026-01-01"`` for windows that end on 2025-12-31.

Use, once per site (nested windows slice the longest one)::

    from experiments.fixedcfg.level_series import LevelSeries
    full = LevelSeries.from_boundary(rec["boundary_obj"], start, end)   # 1 min
    cfg = {"level_series": full.slice(*window_period(t_window))}

``boundary.levels`` is called on chronological chunks: the production
consensus averages a model output sorted by time with gauges read in input
order, so the instants must be sorted (v14.LevelTable's note).
"""
from __future__ import annotations

import numpy as np

#: harmonic analysis: constituents in order of priority; each is kept only
#: if the record resolves it from every one kept before it (Rayleigh)
HARMONIC_ORDER = ("M2", "S2", "N2", "K1", "O1", "K2", "P1", "Q1", "M4", "MS4", "MN4", "M6")
RAYLEIGH = 1.0
CHUNK = 200_000


class LevelSeries:
    """h (float64, metres) at the regular instants t (tz-naive UTC; ``t`` may
    be None when only the values are known, which E7 cannot use)."""

    def __init__(self, h, t=None, source=""):
        import pandas as pd

        self.h = np.asarray(h, float).ravel()
        if t is None:
            self.t = None
        else:
            t = pd.DatetimeIndex(t)
            if t.tz is not None:
                t = t.tz_convert(None)
            if len(t) != len(self.h):
                raise ValueError(f"{len(t)} instants for {len(self.h)} levels")
            if len(t) > 1 and not t.is_monotonic_increasing:
                raise ValueError("the level series must be chronological")
            self.t = t
        self.source = str(source)

    # construction ------------------------------------------------------------
    @classmethod
    def from_boundary(cls, boundary, start, end, step_min=1, chunk=CHUNK, source=None):
        """``boundary.levels`` at every ``step_min`` minutes in [start, end)."""
        import pandas as pd

        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start.tz is not None:
            start = start.tz_convert(None)
        if end.tz is not None:
            end = end.tz_convert(None)
        step = pd.Timedelta(minutes=int(step_min))
        n = int((end - start) // step)
        if n < 2:
            raise ValueError(f"empty series [{start}, {end})")
        t = pd.date_range(start=start, periods=n, freq=step)
        parts = []
        for i in range(0, n, int(chunk)):
            ti = t[i:i + int(chunk)]
            v = np.asarray(boundary.levels(ti), float)
            if v.shape != (len(ti),):
                raise RuntimeError(f"boundary returned {v.shape} levels for {len(ti)} instants")
            parts.append(v)
        name = source or getattr(boundary, "name", type(boundary).__name__)
        return cls(np.concatenate(parts), t, source=f"{name}, {int(step_min)}-min, no lag")

    @classmethod
    def coerce(cls, obj):
        """A LevelSeries from a LevelSeries, a (t, h) pair, a dict with 'h'
        (and 't'), or a bare array of levels; None stays None."""
        if obj is None or isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(obj["h"], obj.get("t"), source=obj.get("source", ""))
        if isinstance(obj, tuple) and len(obj) == 2:
            return cls(obj[1], obj[0])
        return cls(obj)

    # use -----------------------------------------------------------------------
    @property
    def step(self):
        import pandas as pd

        if self.t is None or len(self.t) < 2:
            return None
        return pd.Timedelta(self.t[1] - self.t[0])

    def finite(self):
        return self.h[np.isfinite(self.h)]

    def slice(self, start, end):
        """The part in [start, end)."""
        import pandas as pd

        if self.t is None:
            raise ValueError("a series without times cannot be sliced")
        a = self.t.searchsorted(pd.Timestamp(start), side="left")
        b = self.t.searchsorted(pd.Timestamp(end), side="left")
        return LevelSeries(self.h[a:b], self.t[a:b], source=self.source)

    def summary(self):
        f = self.finite()
        out = {"n": int(len(self.h)), "n_nonfinite": int(len(self.h) - len(f)),
               "min_m": float(f.min()) if len(f) else None,
               "max_m": float(f.max()) if len(f) else None, "source": self.source}
        if self.t is not None and len(self.t):
            out.update(start=str(self.t[0]), end=str(self.t[-1]),
                       step_min=(self.step.total_seconds() / 60.0 if self.step is not None else None))
        return out

    def save(self, path):
        np.savez(path, h=self.h, t_ns=(self.t.asi8 if self.t is not None else np.zeros(0, np.int64)),
                 source=np.array(self.source))

    @classmethod
    def load(cls, path):
        import pandas as pd

        with np.load(path, allow_pickle=False) as z:
            t = pd.DatetimeIndex(z["t_ns"].astype("datetime64[ns]")) if len(z["t_ns"]) else None
            return cls(z["h"], t, source=str(z["source"]))


def window_period(t, end=None):
    """[00:00 of the first scene day, end) for the scene instants ``t``;
    ``end`` defaults to 00:00 after the last scene day."""
    import pandas as pd

    t = pd.DatetimeIndex(t)
    if t.tz is not None:
        t = t.tz_convert(None)
    start = t.min().normalize()
    stop = pd.Timestamp(end) if end is not None else t.max().normalize() + pd.Timedelta(days=1)
    return start, stop


def rayleigh_select(duration_h, order=HARMONIC_ORDER, rayleigh=RAYLEIGH):
    """The constituents of ``order`` resolvable over ``duration_h`` hours: a
    constituent is kept iff |f - f_k| * duration >= rayleigh (cycles) for
    every constituent k already kept."""
    from pyintertidal.boundary import PERIODS_H

    kept = []
    for c in order:
        f = 1.0 / PERIODS_H[c]
        if all(abs(f - 1.0 / PERIODS_H[k]) * duration_h >= rayleigh for k in kept):
            kept.append(c)
    return kept


def harmonic_constants(series, order=HARMONIC_ORDER, rayleigh=RAYLEIGH, chunk=CHUNK):
    """Least-squares harmonic analysis of a level series: the mean level Z0
    and the amplitude of every constituent the record resolves.

    The model is pyintertidal.boundary.fit_harmonics' (a constant plus a
    cosine and a sine per constituent at the astronomical frequencies, phases
    from the package EPOCH, no nodal corrections), solved here by chunked
    normal equations so a multi-year 1-min series is never held as one design
    matrix. Constituents are chosen by the Rayleigh criterion over the record
    length (:func:`rayleigh_select`)."""
    from pyintertidal.boundary import PERIODS_H, hours_since_epoch

    if series is None or series.t is None:
        raise ValueError("the harmonic analysis needs a level series with times")
    ok_all = np.isfinite(series.h)
    if ok_all.sum() < 50:
        raise ValueError("fewer than 50 finite levels")
    dur_h = (series.t[-1] - series.t[0]).total_seconds() / 3600.0
    names = rayleigh_select(dur_h, order, rayleigh)
    w = np.array([2.0 * np.pi / PERIODS_H[c] for c in names])
    K = 1 + 2 * len(names)
    AtA = np.zeros((K, K))
    Aty = np.zeros(K)
    n = len(series.h)
    for i in range(0, n, int(chunk)):
        th = hours_since_epoch(series.t[i:i + int(chunk)])
        y = series.h[i:i + int(chunk)]
        ok = np.isfinite(y)
        th, y = th[ok], y[ok]
        A = np.empty((len(y), K))
        A[:, 0] = 1.0
        ph = np.outer(th, w)
        A[:, 1::2] = np.cos(ph)
        A[:, 2::2] = np.sin(ph)
        AtA += A.T @ A
        Aty += A.T @ y
    coef = np.linalg.solve(AtA, Aty)
    amps = {c: float(np.hypot(coef[1 + 2 * i], coef[2 + 2 * i])) for i, c in enumerate(names)}
    return {"Z0": float(coef[0]), "amplitudes": amps, "constituents": list(names),
            "n": int(ok_all.sum()), "duration_days": dur_h / 24.0,
            "model": "constant + cos/sin per constituent, EPOCH phases, no nodal corrections, "
                     "Rayleigh criterion 1 over the record"}
