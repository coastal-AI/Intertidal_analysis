# -*- coding: utf-8 -*-
"""v13 benchmark — the boundary tide every method receives (§1.3, §2.7).

Question
--------
Which water level does each method see at each record instant, and at each
instant shifted by a lag? Every tide quantity of every method (MAREA's
bank and inversion levels, Granadeiro's heights and lag shifts, the marks,
the gauge truths) must come from ONE provider per site, built exactly as
production builds it, and be read as exact levels, never interpolated or
capped.

Pre-registered rules (FROZEN['levels'], FROZEN['record'])
---------------------------------------------------------
* Provider: ``make_boundary(aoi, "EOT20", n_gauges, period=PERIOD,
  directory="tide_models", cache_dir="data_v4/gauges", exclude=[held-out
  judge])`` at the Dutch sites and Ferrol; ``PyTMDBoundary("EOT20",
  *aoi.centroid, directory="tide_models")`` at Villaviciosa; the Wadden
  provider for the positive control; always after
  ``net.use_system_certificates()``.
* Exact level table LEV (S x T, float64): integer tau in [-90, tau_tab_max]
  with tau_tab_max = max(240, 180 (K_full - 1)); built from sorted provider
  calls on the sorted unique union of t - tau, in chunks of at most 150,000
  instants, mapped back with searchsorted. Every row must be finite and
  LEV[0] must equal the record's h0 bitwise.
* G-levels (batch invariance): every ``marea.BANK_TAUS`` row, every 60th
  row and the last row must equal per-shift ``provider.levels(t - tau)``
  calls bitwise. If any differs, every row is rebuilt by per-shift calls,
  ``batch_invariant = False`` is recorded, and a subset's rows are then
  read by live per-shift calls on t[sel], exactly as ``marea.reconstruct``
  reads them.
* Beyond the table, :meth:`LevelTable.at` makes a live per-shift call on
  t[sel] (identical to ``reconstruct``) and logs it. A lag is never capped.
* 1-min series: ``v8.boundary_series`` over PERIOD +- 2 days (cached with
  v8's tag, written atomically); 10-min series: every 10th sample on
  2023-01-01 00:00 .. 2025-12-31 23:50.
* Marks: ``marks_lp`` = ``g21.tide_marks`` on a centred 61-min running mean
  of the 1-min series (NaN unless all 61 samples are finite), shared by
  every method that uses marks in the controlled rows; ``marks_raw`` =
  ``g21.tide_marks`` on the raw 1-min series, exactly as v8.

Adapters never build a provider: they read the package's LevelTable, which
calls the live provider only beyond the table (or for a batch-variant one).

Validation v2 (docs/validation_v2_spec.md §1.3, §1.5, §1.6, §4.3)
-----------------------------------------------------------------
* The boundary provider is built ONCE per site over the union period
  (:func:`union_provider`: ``make_boundary(..., period=UNION_LEVEL_PERIOD,
  cache_dir=fetch.ioc.UNION_CACHE)``), refused until ``fetch ioc`` covered
  every year 2017-2025 of every member; the exact level table and its
  G-levels check are the v13 ones on the union record; the 1-min series
  takes the period as an argument with v8's tag format.
* Record instants (user decision of 2026-10-14): the catalogue /
  product-name time (the datatake start of ``overpass_times.json``), as in
  every production result; no warning, no flag (:func:`record_instants`).
  A ``sensing_times.json`` (the optional solar-angle diagnostic) is used
  only when its path is given explicitly (CLI ``--sensing-times <file>``).
* The water-level factor (elevation units only, :func:`check_use`): one
  :class:`SourceLevels` per source holds group rows at the record instants
  and a pixel table; LB (:func:`level_lb`, the table's tau = 0 row), LT
  (:func:`level_lt`, eo-tides nodes, nearest node per pixel; the ranked
  ensemble needs a ranking), LM (:func:`level_lm`, CMEMS SSH per 1-km s
  bin, demeaned against h_B over the window, 'credentials missing' ->
  NOT COMPUTABLE), LG (:func:`level_lg`, linear in s through the judges,
  labelled upper bound). A source that cannot serve a window raises
  :class:`NotComputable` / :class:`NotAvailable`; it is never zero and
  never substituted. :data:`LEVEL_RULES` is the v2 'level' section.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd

from experiments.benchmark import config, product
from pyintertidal import granadeiro2021 as g21
from pyintertidal import marea
from pyintertidal import tide_estimators as te

_LEV = config.FROZEN["levels"]
#: first row of the level table (min)
TAU_MIN = int(_LEV["tau_min"])
#: provider instants per batched call
CHUNK = int(_LEV["chunk"])
#: every this-many rows are checked against per-shift calls
CHECK_EVERY = int(_LEV["check_every"])
#: width of the marks' low-pass (1-min samples)
LOWPASS_WIDTH = int(_LEV["marks_lp"]["width_min"])
NS_PER_MIN = 60_000_000_000
LEVELS_FORMAT = "v13-levels-1"
MARKS_FORMAT = "v13-marks-1"
#: the array entries of a g21.tide_marks dict (the rest is bookkeeping)
MARK_ARRAYS = ("t", "h", "is_hw", "sample_index")
del _LEV


def log(msg):
    print(msg, flush=True)


tau_tab_max = config.tau_tab_max


# ─────────────────────────────────────────────────────────────────────────────
#  the provider
# ─────────────────────────────────────────────────────────────────────────────

def make_provider(cfg, period=None, cache_dir=None):
    """The site's boundary provider, built with the production call (§2.1).

    Parameters
    ----------
    cfg : config.SiteConfig
        Its ``boundary`` recipe: ``{"kind": "make_boundary", "aoi",
        "tide_model", "n_gauges", "exclude"}`` or ``{"kind": "pytmd", "aoi",
        "model", "at": "aoi.centroid"}``.
    period, cache_dir : optional
        ``make_boundary``'s period (inclusive dates) and IOC cache; default
        the v13 values ``config.PERIOD`` and ``config.GAUGE_CACHE``. The v2
        provider over the union period is :func:`union_provider`.
    """
    from pyintertidal import net, sites
    from pyintertidal.boundary import PyTMDBoundary, make_boundary

    b = cfg.boundary
    net.use_system_certificates()
    aoi = sites.get(b["aoi"])
    if b["kind"] == "make_boundary":
        return make_boundary(aoi, tide_model=b["tide_model"], n_gauges=b["n_gauges"],
                             period=tuple(period or config.PERIOD),
                             directory=config.TIDE_DIR,
                             cache_dir=cache_dir or config.GAUGE_CACHE,
                             exclude=list(b["exclude"]))
    if b["kind"] == "pytmd":
        if b.get("at", "aoi.centroid") != "aoi.centroid":
            raise ValueError(f"unknown prediction point {b['at']!r}")
        lat, lon = aoi.centroid                 # AOI.centroid is (lat, lon)
        return PyTMDBoundary(b["model"], lat, lon, directory=config.TIDE_DIR)
    raise ValueError(f"unknown boundary kind {b['kind']!r}")


def provider_name(cfg, provider):
    """The provider's name as v8 writes it into the 1-min cache tag
    (``v8.build_boundary``), so the two caches are comparable."""
    b = cfg.boundary
    if b["kind"] == "pytmd":
        lat, lon = provider.lat, provider.lon
        return f"PyTMDBoundary {b['model']} at aoi.centroid ({lat:.5f}, {lon:.5f})"
    return getattr(provider, "name", str(provider))


def describe_provider(provider):
    """``{"name", "models": [...], "gauges": [...]}`` of a provider tree:
    the tide models and the IOC gauge codes actually used."""
    models, gauges = [], []

    def walk(p):
        if hasattr(p, "providers"):              # EnsembleBoundary
            for q in p.providers:
                walk(q)
        elif hasattr(p, "gauges"):               # InterpolatedGauges
            for g in p.gauges:
                walk(g)
        elif hasattr(p, "model"):                # PyTMDBoundary
            models.append({"model": p.model, "lat": float(p.lat), "lon": float(p.lon)})
        else:                                    # GaugeBoundary (name = IOC code)
            gauges.append(str(getattr(p, "name", p)))

    walk(provider)
    return {"name": str(getattr(provider, "name", provider)), "models": models,
            "gauges": gauges}


def boundary_codes(cfg, stations_path=None):
    """The IOC codes ``make_boundary`` takes for the site: the ``n_gauges``
    nearest stations of ``ioc_stations.json`` after removing the held-out
    codes ([] for a pytmd boundary)."""
    b = cfg.boundary
    if b["kind"] != "make_boundary" or not b.get("n_gauges"):
        return []
    from pyintertidal import sites
    from pyintertidal.boundary import nearest_gauges

    lat, lon = sites.get(b["aoi"]).centroid
    excl = {str(c).lower() for c in b["exclude"]}
    near = [g for g in nearest_gauges(lat, lon, int(b["n_gauges"]) + len(excl),
                                      stations_path=stations_path or config.IOC_STATIONS)
            if g[0].lower() not in excl][:int(b["n_gauges"])]
    return [g[0] for g in near]


def ioc_chunks(cfg, provider=None, cache_dir=None, stations_path=None):
    """Every IOC file the provider reads, with sha256 and size.

    ``make_boundary`` takes the ``n_gauges`` nearest stations of
    ``ioc_stations.json`` after removing the held-out codes, and for each one
    ``load_cached_ioc`` reads every ``ioc_<code>_*.json`` chunk of the cache
    (a candidate with too few samples is read, then skipped). The list holds
    the station file and every chunk of every candidate. With ``provider``,
    its gauge members are asserted to be candidates.

    Returns
    -------
    list of dict
        ``{"code", "path" (repository-relative, '/'), "sha256", "size"}``.
    """
    b = cfg.boundary
    if b["kind"] != "make_boundary" or not b.get("n_gauges"):
        return []
    cache_dir = cache_dir or config.GAUGE_CACHE
    stations_path = stations_path or config.IOC_STATIONS
    codes = boundary_codes(cfg, stations_path)
    if provider is not None:
        used = describe_provider(provider)["gauges"]
        if not set(used) <= set(codes):
            raise AssertionError(f"provider gauges {used} are not the candidates {codes}")

    def entry(code, path):
        absolute = os.path.abspath(path)
        try:
            rel = os.path.relpath(absolute, config.ROOT).replace("\\", "/")
        except ValueError:                       # another drive on Windows
            rel = absolute.replace("\\", "/")
        return {"code": code, "path": rel, "sha256": product.sha256_file(absolute),
                "size": int(os.path.getsize(absolute))}

    out = [entry("ioc_stations", stations_path)]
    for code in codes:
        files = sorted(glob.glob(os.path.join(cache_dir, f"ioc_{code}_*.json")))
        if not files:
            raise FileNotFoundError(f"no cached IOC chunk for boundary gauge {code} "
                                    f"in {cache_dir}")
        out += [entry(code, p) for p in files]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the exact level table
# ─────────────────────────────────────────────────────────────────────────────

def _as_index(t):
    idx = pd.DatetimeIndex(t)
    if idx.tz is not None:
        raise ValueError("record instants must be naive UTC")
    idx = idx.as_unit("ns")
    if len(idx) and not (np.diff(idx.asi8) > 0).all():
        raise ValueError("record instants must be strictly increasing")
    return idx


def _as_minutes(tau):
    """An integer number of minutes (int, or a float with an integer value)."""
    if isinstance(tau, (bool, np.bool_)):
        raise TypeError("tau must be a number of minutes")
    f = float(tau)
    if not np.isfinite(f) or f != np.floor(f):
        raise ValueError(f"the level table holds integer minutes; got tau = {tau!r} "
                         "(MAREA and Granadeiro ctl round their lags to the minute)")
    return int(f)


def _levels(provider, idx):
    """``provider.levels`` on a DatetimeIndex, as float64 of the right length."""
    v = np.asarray(provider.levels(idx), dtype=np.float64)
    if v.shape != (len(idx),):
        raise AssertionError(f"provider.levels returned shape {v.shape} for "
                             f"{len(idx)} instants")
    return v


def _bitwise_equal(a, b):
    a = np.ascontiguousarray(a, np.float64)
    b = np.ascontiguousarray(b, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64))


def _assert_finite(H, shifts):
    bad = ~np.isfinite(H).all(axis=1)
    if bad.any():
        raise ValueError(f"non-finite boundary levels at tau = {shifts[bad][:10].tolist()} "
                         f"min ({int(bad.sum())} rows): every table row must be finite")


class LevelTable:
    """Exact boundary levels h_B(t_i - tau) for integer tau at the record
    instants t_i.

    Attributes
    ----------
    shifts : (S,) int64
        Consecutive integer minutes, ``TAU_MIN .. tau_tab_max``.
    H : (S, T) float64, read-only
        ``H[k, i] = h_B(t_i - shifts[k])``.
    t : DatetimeIndex (T,)
        The record instants (naive UTC, ns).
    batch_invariant : bool
        G-levels outcome: table rows may be indexed by any subset.
    live_calls : list of dict
        Every live per-shift provider call this object made (tau, n, why).
    """

    def __init__(self, shifts, H, t, batch_invariant=True, provider=None,
                 checked=(), meta=None):
        shifts = np.asarray(shifts, np.int64)
        if shifts.ndim != 1 or not len(shifts) or np.any(np.diff(shifts) != 1):
            raise ValueError("shifts must be consecutive integer minutes")
        if not shifts[0] <= 0 <= shifts[-1]:
            raise ValueError("the table must hold tau = 0")
        t = _as_index(t)
        H = np.array(H, dtype=np.float64)
        if H.shape != (len(shifts), len(t)):
            raise ValueError(f"H has shape {H.shape}, expected {(len(shifts), len(t))}")
        H.flags.writeable = False
        self.shifts, self.H, self.t = shifts, H, t
        self.batch_invariant = bool(batch_invariant)
        self.checked = np.asarray(checked, np.int64)
        self.meta = dict(meta or {})
        self._provider = provider
        self.live_calls = []
        self.build_seconds = None               # set by build_level_table; not saved

    # basic facts ------------------------------------------------------------
    @property
    def T(self):
        return len(self.t)

    @property
    def tau_lo(self):
        return int(self.shifts[0])

    @property
    def tau_hi(self):
        return int(self.shifts[-1])

    @property
    def h0(self):
        """h_B at the record instants (row tau = 0), a copy."""
        return self.H[-self.tau_lo].copy()

    def __repr__(self):
        return (f"<LevelTable tau {self.tau_lo}..{self.tau_hi} min x {self.T} instants, "
                f"batch_invariant={self.batch_invariant}>")

    def provider(self):
        """The live provider (built on first use when given as a factory)."""
        p = self._provider
        if p is None:
            raise RuntimeError("this LevelTable has no live provider; pass provider= "
                               "(an object with .levels or a zero-argument factory)")
        if not hasattr(p, "levels") and callable(p):
            p = self._provider = p()
        return p

    def _positions(self, sel):
        if sel is None:
            return None
        a = np.asarray(sel)
        if a.dtype == bool:
            if a.shape != (self.T,):
                raise ValueError(f"a boolean sel must have length {self.T}")
            return np.flatnonzero(a)
        if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
            raise TypeError("sel must be record positions (1-D integers) or a mask")
        a = a.astype(np.int64)
        if a.size and (a[0] < 0 or a[-1] >= self.T or np.any(np.diff(a) <= 0)):
            raise ValueError("sel must be sorted unique record positions in [0, T)")
        return a

    # reading ----------------------------------------------------------------
    def at(self, tau_min, sel=None):
        """h_B(t[sel] - tau) in float64.

        A table row (indexed by ``sel``) inside the table; a live per-shift
        provider call on t[sel], exactly as ``marea.reconstruct``, beyond
        the table or, for a batch-variant provider, for a proper subset.

        Parameters
        ----------
        tau_min : int or integer-valued float
            The lag in minutes.
        sel : array of record positions, boolean mask, or None (all)
        """
        tau = _as_minutes(tau_min)
        idx = self._positions(sel)
        full = idx is None or (len(idx) == self.T)
        inside = self.tau_lo <= tau <= self.tau_hi
        if inside and (self.batch_invariant or full):
            row = self.H[tau - self.tau_lo]
            return row.copy() if idx is None else row[idx]
        why = "beyond the table" if not inside else "batch-variant provider, subset"
        tt = self.t if idx is None else self.t[idx]
        h = _levels(self.provider(), tt - pd.Timedelta(minutes=float(tau)))
        if not np.all(np.isfinite(h)):
            raise ValueError(f"the provider gives non-finite levels at tau = {tau} min")
        self.live_calls.append({"tau": tau, "n": int(len(tt)), "why": why})
        log(f"[levels] live per-shift call tau = {tau} min on {len(tt)} instants ({why})")
        return h

    def stack(self, taus, sel=None):
        """``(len(taus), n)`` array of :meth:`at` rows."""
        return np.vstack([self.at(tau, sel) for tau in taus])

    def bank(self, sel=None, taus=None):
        """MAREA's 15-min shift bank on the subset: ``te.ShiftBank(taus,
        [at(tau, sel) for tau in taus])`` with ``taus = marea.BANK_TAUS`` by
        default (``tau120`` passes the taus <= 120)."""
        taus = list(marea.BANK_TAUS if taus is None else taus)
        return te.ShiftBank(taus, [self.at(tau, sel) for tau in taus])

    # persistence ------------------------------------------------------------
    def save(self, path):
        """``levels.npz``: shifts, H, h0, t (int64 ns), batch_invariant,
        checked rows and the build bookkeeping (byte-stable, atomic)."""
        product.save_npz(path, {
            "format": np.array(LEVELS_FORMAT), "shifts": self.shifts, "H": self.H,
            "h0": self.h0, "t_ns": self.t.asi8.astype(np.int64),
            "batch_invariant": np.array(self.batch_invariant),
            "checked": self.checked,
            "meta": np.array(json.dumps(self.meta, sort_keys=True, default=str)),
        }, compress=False)

    @classmethod
    def load(cls, path, provider=None):
        """Read a table written by :meth:`save`; ``provider`` (object or
        zero-argument factory) serves the live calls."""
        z = product.load_npz(path)
        if str(z.get("format")) != LEVELS_FORMAT:
            raise ValueError(f"{path}: not a {LEVELS_FORMAT} file")
        tab = cls(z["shifts"], z["H"], pd.DatetimeIndex(z["t_ns"].view("M8[ns]")),
                  batch_invariant=bool(z["batch_invariant"]), provider=provider,
                  checked=z["checked"], meta=json.loads(str(z["meta"])))
        if not _bitwise_equal(tab.h0, z["h0"]):
            raise ValueError(f"{path}: H[tau = 0] differs from h0")
        return tab


def build_level_table(provider, t, tau_tab_max, chunk=CHUNK, h0=None,
                      check_every=CHECK_EVERY, tau_min=TAU_MIN):
    """The exact level table of a record (§2.7), with the G-levels check.

    Parameters
    ----------
    provider : object with ``levels(times)``
        The site's live provider (:func:`make_provider`).
    t : DatetimeIndex
        Record instants, naive UTC, strictly increasing.
    tau_tab_max : int
        Last row (``config.tau_tab_max(K_full)``).
    chunk : int
        At most this many instants per batched provider call.
    h0 : array, optional
        The record's levels from the record step; ``LEV[0]`` must equal it
        bitwise (a difference is treated as batch variance).
    check_every, tau_min : int
        FROZEN values; tests may change them.

    Returns
    -------
    LevelTable
        With ``batch_invariant`` and, in ``meta``, the checked rows, any
        mismatching rows, the number of unique instants and calls.
    """
    t0 = time.time()
    t = _as_index(t)
    T = len(t)
    if T == 0:
        raise ValueError("empty record")
    tau_tab_max = int(tau_tab_max)
    if tau_tab_max < 0 or int(tau_min) > 0:
        raise ValueError("the table must hold tau = 0")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    shifts = np.arange(int(tau_min), tau_tab_max + 1, dtype=np.int64)
    S = len(shifts)
    inst = t.asi8[None, :] - shifts[:, None] * NS_PER_MIN        # (S, T) int64 ns
    u = np.unique(inst)
    n_unique, n_calls = int(len(u)), int(np.ceil(len(u) / chunk))
    log(f"[levels] table tau {shifts[0]}..{shifts[-1]} min x {T} instants: "
        f"{n_unique:,} unique instants in {n_calls} provider call(s)")
    h_u = np.empty(len(u))
    for a in range(0, len(u), chunk):
        b = min(a + chunk, len(u))
        # sorted unique instants: PyTMDBoundary answers in time order, and an
        # ensemble averages its members elementwise only on aligned input
        h_u[a:b] = _levels(provider, pd.DatetimeIndex(u[a:b].view("M8[ns]")))
    H = h_u[np.searchsorted(u, inst)]
    del inst, u, h_u
    _assert_finite(H, shifts)                   # before any (costly) per-shift call

    # G-levels: the batched rows must be the per-shift calls reconstruct makes
    check = sorted(({_as_minutes(x) for x in marea.BANK_TAUS}
                    | set(shifts[::max(1, int(check_every))].tolist())
                    | {int(shifts[-1])}) & set(shifts.tolist()))
    k0 = -int(shifts[0])
    mismatch = []
    for tau in check:
        v = _levels(provider, t - pd.Timedelta(minutes=float(tau)))
        if not _bitwise_equal(v, H[tau - shifts[0]]):
            mismatch.append(int(tau))
    if h0 is not None and not _bitwise_equal(H[k0], h0):
        mismatch.append("h0")
    batch_invariant = not mismatch
    if not batch_invariant:
        log(f"[levels] G-levels: batched rows differ from per-shift calls at {mismatch}; "
            f"rebuilding all {S} rows by per-shift calls (batch_invariant = False)")
        for k, tau in enumerate(shifts):
            H[k] = _levels(provider, t - pd.Timedelta(minutes=float(tau)))
        if h0 is not None and not _bitwise_equal(H[k0], h0):
            raise RuntimeError("the record's h0 differs from a per-shift call on the "
                               "record instants: the provider is not batch-invariant "
                               "even at tau = 0; rebuild the record with LEV[0]")
        _assert_finite(H, shifts)
    # deterministic bookkeeping only: levels.npz is a keyed package file, so
    # rebuilding the same table must give the same bytes (no timings here)
    meta = {"checked": check, "mismatch": mismatch, "n_unique_instants": n_unique,
            "n_batched_calls": n_calls, "chunk": int(chunk),
            "check_every": int(check_every)}
    seconds = round(time.time() - t0, 1)
    log(f"[levels] G-levels {'PASS' if batch_invariant else 'FAIL (fallback in use)'}: "
        f"{len(check)} rows checked; table built in {seconds} s")
    tab = LevelTable(shifts, H, t, batch_invariant=batch_invariant, provider=provider,
                     checked=check, meta=meta)
    tab.build_seconds = seconds
    return tab


# ─────────────────────────────────────────────────────────────────────────────
#  1-min and 10-min series, marks
# ─────────────────────────────────────────────────────────────────────────────

def minutes_since_epoch(idx):
    """Naive-UTC datetimes -> float minutes since 1970-01-01 (ns-exact), the
    formula of ``v8.minutes_since_epoch``."""
    return np.asarray(pd.DatetimeIndex(idx).values.astype("datetime64[ns]")
                      .astype(np.int64)) / 6e10


def _series_bounds(period=None):
    """``[t0, t1)`` of the 1-min series: ``period`` (inclusive dates; default
    the v13 PERIOD) padded by FROZEN's pad_days, the end day included."""
    period = tuple(period or config.PERIOD)
    pad = int(config.FROZEN["levels"]["series_1min"]["pad_days"])
    t0 = pd.Timestamp(period[0]) - pd.Timedelta(days=pad)
    t1 = pd.Timestamp(period[1]) + pd.Timedelta(days=1 + pad)
    return t0, t1


def series_1min_index(period=None):
    """The 1-min grid of ``v8.boundary_series``: period +- 2 days, the end
    excluded (v13: 2022-12-30 00:00 .. 2026-01-02 23:59)."""
    t0, t1 = _series_bounds(period)
    return pd.date_range(t0, t1, freq="1min", inclusive="left")


def series_1min_tag(name, period=None):
    """v8's cache tag of the 1-min series of a provider called ``name``."""
    t0, t1 = _series_bounds(period)
    return f"{name}|{t0}|{t1}|1min"


def load_series_1min(cache_path, name=None, period=None):
    """``(t_min, h)`` from a 1-min cache without a provider (the package's
    lazy loader); with ``name``, the cache's tag is asserted to be v8's."""
    idx = series_1min_index(period)
    with np.load(cache_path) as z:
        tag, h = str(z["tag"]), np.asarray(z["h"], np.float64)
    if len(h) != len(idx):
        raise ValueError(f"{cache_path}: {len(h)} samples, expected {len(idx)}")
    if name is not None and tag != series_1min_tag(name, period):
        raise ValueError(f"{cache_path}: tag {tag!r} is not "
                         f"{series_1min_tag(name, period)!r}")
    return minutes_since_epoch(idx), h


def _series_1min_compute(provider, name, cache_path, period):
    """``v8.boundary_series`` over any period: v8's 1-min grid, its
    ``SERIES_CHUNK_DAYS`` provider calls in time order, its tag and npz
    members (v8's own function reads its module-level TIME_EXTENT)."""
    from experiments import v8_granadeiro2021 as v8

    idx = series_1min_index(period)
    tag = series_1min_tag(name, period)
    step = int(v8.SERIES_CHUNK_DAYS) * 1440
    h = np.empty(len(idx))
    tt = time.time()
    for a in range(0, len(idx), step):
        b = min(a + step, len(idx))
        v = np.asarray(provider.levels(idx[a:b]), float)
        if v.shape != (b - a,):
            raise AssertionError("boundary.levels returned a wrong length")
        h[a:b] = v
        log(f"    boundary 1-min series {b:,}/{len(idx):,} ({time.time() - tt:.0f} s)")
    np.savez(cache_path, tag=np.array(tag), h=h)
    return minutes_since_epoch(idx), h, False


def boundary_series_1min(provider, name, cache_path, period=None):
    """The 1-min boundary series of ``v8.boundary_series``: ``(t_min, h,
    cached)`` over period +- 2 days, cached in ``cache_path`` with v8's tag.

    ``period`` (inclusive dates) defaults to the v13 PERIOD, served by v8's
    own function. Any other period (the v2 union period, §1.5) is built by
    :func:`_series_1min_compute` with v8's grid, chunking and tag (the
    assertion on v8's TIME_EXTENT applies only to the default). The cache is
    written atomically (v8 writes in place): a stale or foreign cache is
    renamed ``*.stale-<utc>.npz`` and rebuilt.
    """
    from experiments import v8_granadeiro2021 as v8

    v8_period = period is None or tuple(period) == tuple(v8.TIME_EXTENT)
    if period is None and tuple(v8.TIME_EXTENT) != tuple(config.PERIOD):
        raise AssertionError("v8.TIME_EXTENT differs from the benchmark PERIOD")
    if int(v8.SERIES_PAD_DAYS) != int(config.FROZEN["levels"]["series_1min"]["pad_days"]):
        raise AssertionError("v8.SERIES_PAD_DAYS differs from FROZEN['levels']")
    cache_path = os.fspath(cache_path)
    if not cache_path.endswith(".npz"):
        raise ValueError("cache_path must end with .npz")
    tag, n = series_1min_tag(name, period), len(series_1min_index(period))
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as z:
                ok = str(z["tag"]) == tag and len(z["h"]) == n
        except Exception:                       # unreadable: rebuild it
            ok = False
        if ok:
            if v8_period:
                return v8.boundary_series(provider, name, cache_path)
            with np.load(cache_path) as z:
                h = np.asarray(z["h"], np.float64)
            log(f"  boundary series: cached {cache_path}")
            return minutes_since_epoch(series_1min_index(period)), h, True
        stale = f"{cache_path[:-4]}.stale-{product.utc_stamp()}.npz"
        os.replace(cache_path, stale)
        log(f"[levels] 1-min cache {cache_path} has another tag; kept as {stale}")
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    tmp = f"{cache_path[:-4]}.tmp-{os.getpid()}.npz"
    if os.path.exists(tmp):
        os.remove(tmp)
    if v8_period:
        t_min, h, _ = v8.boundary_series(provider, name, tmp)
    else:
        t_min, h, _ = _series_1min_compute(provider, name, tmp, tuple(period))
    os.replace(tmp, cache_path)
    return t_min, h, False


def lowpass(h_1min, width=LOWPASS_WIDTH):
    """Centred running mean of an evenly sampled series.

    ``out[i] = mean(h[i - w//2 .. i + w//2])``, NaN unless all ``width``
    samples are finite (and within the series). The sum runs over the
    window offsets in a fixed order, so the result does not depend on the
    numpy version's reduction strategy.
    """
    h = np.asarray(h_1min, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError("lowpass takes a 1-D series")
    width = int(width)
    if width < 1 or width % 2 == 0:
        raise ValueError("width must be a positive odd number of samples")
    n, half = len(h), width // 2
    out = np.full(n, np.nan)
    if n < width:
        return out
    m = n - width + 1
    acc = h[0:m].copy()
    for k in range(1, width):
        acc += h[k:k + m]
    out[half:half + m] = acc / width
    return out


def marks(t_min, h):
    """Tide-table marks of a dense series: ``g21.tide_marks(t_min, h)``
    (NaN samples dropped)."""
    return g21.tide_marks(t_min, h)


def build_marks(t_min, h, width=LOWPASS_WIDTH):
    """Both mark versions of the 1-min series.

    Returns
    -------
    dict
        ``{"marks_lp": tide_marks on lowpass(h, width) (with "lowpass_width"),
        "marks_raw": tide_marks on h, as v8}``.
    """
    lp = marks(t_min, lowpass(h, width))
    lp["lowpass_width"] = int(width)
    return {"marks_lp": lp, "marks_raw": marks(t_min, h)}


def save_marks(path, **named):
    """Write named mark dicts (e.g. ``marks_lp=..., marks_raw=...``) to one
    npz: the arrays as ``<name>.<field>``, the bookkeeping as JSON."""
    arrays = {"format": np.array(MARKS_FORMAT)}
    for name, mk in named.items():
        if "." in name:
            raise ValueError("mark names may not contain '.'")
        for key in MARK_ARRAYS:
            arrays[f"{name}.{key}"] = np.asarray(mk[key])
        meta = {k: v for k, v in mk.items() if k not in MARK_ARRAYS}
        arrays[f"{name}.meta"] = np.array(json.dumps(meta, sort_keys=True))
    product.save_npz(path, arrays, compress=True)


def load_marks(path):
    """``{name: marks dict}`` from :func:`save_marks` (arrays and bookkeeping
    as ``g21.tide_marks`` returns them)."""
    z = product.load_npz(path)
    if str(z.get("format")) != MARKS_FORMAT:
        raise ValueError(f"{path}: not a {MARKS_FORMAT} file")
    out = {}
    for name in sorted({k.split(".", 1)[0] for k in z if "." in k}):
        mk = {key: z[f"{name}.{key}"] for key in MARK_ARRAYS}
        mk.update(json.loads(str(z[f"{name}.meta"])))
        out[name] = mk
    return out


def series_10min(t_min, h):
    """The 10-min boundary series: every 10th sample of the 1-min series on
    the grid 2023-01-01 00:00 .. 2025-12-31 23:50.

    Returns
    -------
    (DatetimeIndex, ndarray)
    """
    sp = config.FROZEN["levels"]["series_10min"]
    every = int(sp["every"])
    idx = pd.date_range(sp["start"], sp["end"], freq=f"{every}min")
    m = minutes_since_epoch(idx)
    t_min = np.asarray(t_min, np.float64)
    h = np.asarray(h, np.float64)
    if t_min.shape != h.shape or t_min.ndim != 1:
        raise ValueError("t_min and h must be aligned 1-D series")
    pos = np.searchsorted(t_min, m)
    if (pos >= len(t_min)).any() or not np.array_equal(t_min[pos], m):
        raise ValueError("the 10-min grid is not on the 1-min series")
    if np.any(np.diff(pos) != every):
        raise ValueError("the 1-min series is not evenly sampled every minute")
    return idx, h[pos]


def check_boundary_spec(cfg):
    """Raise ValueError when ``cfg.boundary`` is not a recipe of FROZEN['record']
    (the provider must be the production one of its site)."""
    want = config.FROZEN["record"]["h_B"].get(cfg.site)
    got = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.boundary.items()}
    if want is None or config.sha256_json(got) != config.sha256_json(want):
        raise ValueError(f"{cfg.site}: boundary {got} is not FROZEN's {want}")
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  validation v2: the union-period boundary and the water-level sources
#  (docs/validation_v2_spec.md §1.3, §1.5, §1.6, §4.3, Appendix A 'level')
# ═════════════════════════════════════════════════════════════════════════════

#: make_boundary's inclusive period of the v2 provider (settings.UNION_PERIOD
#: is end-exclusive: 2017-01-01 .. 2026-01-01)
UNION_LEVEL_PERIOD = ("2017-01-01", "2025-12-31")
SOURCE_FORMAT = "validation-v2-level-source-1"
SOURCES = ("LB", "LT", "LM", "LG")

STATUS_OK = "ok"
STATUS_CREDENTIALS = "credentials missing"
STATUS_NOT_AVAILABLE = "not available"
STATUS_NOT_APPLICABLE = "not applicable"
STATUS_NOT_COMPUTABLE = "not computable"
STATUSES = (STATUS_OK, STATUS_CREDENTIALS, STATUS_NOT_AVAILABLE, STATUS_NOT_APPLICABLE,
            STATUS_NOT_COMPUTABLE)
#: a source status -> the status of every unit that needs it (§4.4, §9.5)
UNIT_STATUS = {STATUS_OK: "ok", STATUS_CREDENTIALS: "NOT COMPUTABLE",
               STATUS_NOT_COMPUTABLE: "NOT COMPUTABLE", STATUS_NOT_AVAILABLE: "NOT AVAILABLE",
               STATUS_NOT_APPLICABLE: "not_applicable"}
#: LG: a judge value at an instant is the linear interpolation of its centred
#: 10-min series between samples at most this far apart (min)
LG_MAX_GAP_MIN = 10

#: the v2 level rules (JSON-ready; the 'level' section of the v2 FROZEN plan)
LEVEL_RULES = {
    "LB": {"provider": ("make_boundary(aoi, EOT20, n_gauges, period=UNION_LEVEL_PERIOD, "
                        "cache_dir=fetch.ioc.UNION_CACHE, exclude=held-out) or PyTMDBoundary "
                        "at the AOI centroid; built once after fetch ioc wrote every marker"),
           "table": "build_level_table on the union record (v13 rules)"},
    "LT": {"tool": "eo_tides.eo.pixel_tides(GeoBox, resample=False, resolution=1000, "
                   "buffer=1000, extrapolate=True, cutoff=inf, parallel=False)",
           "node_km": 1.0, "pixel_to_node": "nearest (declared)", "primary": "EOT20",
           "ensemble_min_models": 3, "top_n": {"eo-tides": 3, "bt26": 1},
           "ensemble": "equal-weight mean of the members ranked <= top_n per node",
           "ranking": ("eo-tides' default ranking points cover Australia only: the truth-free "
                       "BT26-rank of L12 (top_n 1); without it an ensemble LT is NOT COMPUTABLE"),
           "demean": False},
    "LM": {"datasets": "fetch.cmems.DATASETS (public catalogue 2026-10-14)",
           "selection": "fetch.cmems.choose_dataset per (site, window); none -> NOT AVAILABLE",
           "variable": "zos", "bin_km": 1.0, "unresolved_km": 1.0,
           "channel_cell": "wet cell (finite at every step) whose nearest finite-s pixel "
                           "is within 1 km; its bin is that pixel's floor(s / 1 km)",
           "bin_series": ("mean of the bin's channel cells; a bin without one takes the wet "
                          "cell nearest to its pixels; unresolved if that distance > 1 km"),
           "time": ("4-point cubic Lagrange interpolation on the model's regular steps; "
                    "hourly means need a measured stamp shift"),
           "demean": "against h_B over the window's record instants (declared)",
           "credentials": "fetch.cmems.CREDENTIALS_FILE; missing -> NOT COMPUTABLE"},
    "LG": {"knots": "s = 0: h_B; each included judge location at s_g (pairs averaged)",
           "interpolation": "linear in s; constant beyond the last gauge; a missing judge "
                            "is filled per instant by the same interpolation",
           "gauge_at": f"linear in time on the centred 10-min series, gap <= {LG_MAX_GAP_MIN} min",
           "demean": "each judge against h_B over the window's record instants (declared; "
                     "'native' datum is a sensitivity)",
           "use": "labelled upper bound; never in lag scoring nor a confirmatory family"},
}


class LevelSourceError(RuntimeError):
    """A level source that cannot serve a unit (status and reason kept)."""

    def __init__(self, source, status, reason):
        self.source, self.status, self.reason = source, status, reason
        super().__init__(f"{source}: {status}: {reason}")


class NotComputable(LevelSourceError):
    """The source cannot be computed (credentials missing, ranking missing)."""


class NotAvailable(LevelSourceError):
    """The source does not cover the window (archive depth, data gaps)."""


def _inclusive(period):
    """``(start, end)`` with an exclusive end -> inclusive dates."""
    return (str(period[0])[:10],
            (pd.Timestamp(period[1]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))


def union_provider(cfg, cache_dir=None, stations_path=None):
    """The v2 boundary provider over the union period (§1.5).

    ``make_boundary(..., period=UNION_LEVEL_PERIOD, cache_dir=<union cache>)``
    exactly as production otherwise. It is refused until ``fetch ioc`` has
    covered every year 2017-2025 of every boundary member and written the
    markers, because ``make_boundary`` would otherwise download the union
    in one call and h_B would depend on when the provider was first built.
    Villaviciosa (PyTMDBoundary) needs no gauge.
    """
    from experiments.benchmark.fetch import ioc as fioc

    from experiments.benchmark import settings

    if _inclusive(settings.UNION_PERIOD) != UNION_LEVEL_PERIOD or \
            tuple(fioc.UNION_FETCH_PERIOD) != UNION_LEVEL_PERIOD:
        raise AssertionError("UNION_LEVEL_PERIOD, settings.UNION_PERIOD and the IOC union "
                             "period disagree")
    cache_dir = cache_dir or fioc.UNION_CACHE
    codes = boundary_codes(cfg, stations_path)
    if codes:
        ready, problems = fioc.union_ready(codes, cache_dir)
        if not ready:
            raise RuntimeError(f"{cfg.site}: the union IOC cache is incomplete ({problems}); "
                               "run 'fetch ioc' before building h_B (§1.5)")
    return make_provider(cfg, period=UNION_LEVEL_PERIOD, cache_dir=cache_dir)


def boundary_coverage(provider, t, windows=None):
    """Where h_B is EOT20-only because every gauge member is NaN (§1.5).

    Parameters
    ----------
    provider : boundary provider (an EnsembleBoundary of a model and gauges)
    t : DatetimeIndex
        Record instants.
    windows : mapping, optional
        ``{window id: record positions}``.

    Returns
    -------
    dict
        ``{"n", "gauge_parts", "model_only" (count), "share", "windows":
        {wid: share}}``; ``gauge_parts == 0`` means no gauge member.
    """
    t = _as_index(t)
    parts = list(getattr(provider, "providers", [provider]))
    gauges = [p for p in parts if not hasattr(p, "model")]
    if not gauges:
        return {"n": len(t), "gauge_parts": 0, "model_only": len(t), "share": 1.0,
                "windows": {w: 1.0 for w in (windows or {})}}
    finite = np.zeros(len(t), bool)
    for g in gauges:
        finite |= np.isfinite(np.asarray(g.levels(t), np.float64))
    only = ~finite
    out = {"n": len(t), "gauge_parts": len(gauges), "model_only": int(only.sum()),
           "share": float(only.mean()) if len(t) else float("nan"), "windows": {}}
    for wid, pos in (windows or {}).items():
        pos = np.asarray(pos, np.int64)
        out["windows"][wid] = float(only[pos].mean()) if len(pos) else float("nan")
    return out


def record_instants(site, root=None, instants="catalogue"):
    """The record instants of a site and where they come from (§1.3; user
    decision of 2026-10-14).

    Parameters
    ----------
    site : str
    root : path, optional
        The cubes root (``zarr_cubes.cubes_root``).
    instants : "catalogue" (default) | "labels" | path | mapping
        "catalogue" (alias "labels"): the catalogue / product-name times of
        ``<root>/<site>/overpass_times.json`` (the datatake start), the
        default of every run, without warning or flag. A path: a
        ``sensing_times.json`` given explicitly (the optional diagnostic of
        ``sensing_time.py``; CLI ``--sensing-times``). A mapping:
        ``{date: instant}``.

    Returns
    -------
    dict
        ``dates`` (sorted, valid instant only), ``t`` (DatetimeIndex, naive
        UTC), ``source`` ("catalogue", "sensing_times", "given"), ``path``
        (the file read, or None) and ``dropped`` (dates without an instant).
    """
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs

    def labels():
        lab = zc.read_overpass(site, root)
        if not lab:
            raise FileNotFoundError(f"{zc.overpass_path(site, root)}: no catalogue times")
        return lab, str(zc.overpass_path(site, root)).replace("\\", "/")

    d, source, path = inputs.resolve_instants(instants, labels)
    inputs._check_instants(d)
    dates = sorted(k for k, v in d.items() if not pd.isna(v))
    dropped = sorted(k for k, v in d.items() if pd.isna(v))
    t = pd.DatetimeIndex([d[k] for k in dates]).as_unit("ns")
    if len(t) and not (np.diff(t.asi8) > 0).all():
        raise ValueError(f"{site}: record instants are not strictly increasing")
    return {"dates": dates, "t": t, "source": source, "path": path, "dropped": dropped}


def check_use(source, use):
    """Raise ValueError when a level source is used where §1.6 forbids it.

    ``use``: "elevation" (every source), "lag" (LB only: level-source rows are
    elevation units) or "confirmatory" (not LG, the labelled upper bound).
    """
    if source not in SOURCES:
        raise ValueError(f"unknown level source {source!r}")
    if use == "lag" and source != "LB":
        raise ValueError(f"{source} is an elevation-only level source; lags are scored on LB")
    if use == "confirmatory" and source == "LG":
        raise ValueError("LG uses the in-estuary gauges: a labelled upper bound, never in a "
                         "confirmatory family")
    if use not in ("elevation", "lag", "confirmatory"):
        raise ValueError(f"unknown use {use!r}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  level rows of one source
# ─────────────────────────────────────────────────────────────────────────────

def _fill_knots(R, knots_s):
    """Per instant, NaN knots replaced by the linear interpolation in s of the
    finite knots (constant beyond the end knots)."""
    R = np.array(R, np.float64)
    s = np.asarray(knots_s, np.float64)
    bad = ~np.isfinite(R)
    if not bad.any():
        return R
    for j in np.flatnonzero(bad.any(axis=0)):
        ok = ~bad[:, j]
        if ok.any():
            R[bad[:, j], j] = np.interp(s[bad[:, j]], s[ok], R[ok, j])
    return R


class SourceLevels:
    """Water levels of one source at the record instants (§1.6).

    The level of pixel p at instant i is ``(1 - w_p) R[g0_p, i] + w_p
    R[g1_p, i]`` with R the group rows: 1 km nodes (LT, w = 0), 1-km s bins
    (LM, w = 0), s knots (LG, linear weights) or the single boundary row
    (LB). Rows are stored raw; a demeaned source (LM, LG) subtracts per group
    the mean of ``H - h_B`` over a WINDOW's record positions at read time, so
    one object serves every window and every subset of it.

    Attributes
    ----------
    source, site, status, reason : str
    t : DatetimeIndex (T,)
    H : (G, T) float64 or None (status not ok)
    pix : (n,) int64 sorted frame flat indices; None for a uniform source
    g0, g1 : (n,) int32 (-1: the pixel has no level, e.g. no s)
    w : (n,) float64 in [0, 1]
    ref : (T,) float64 h_B at the instants (demeaning) or None
    knots_s : (G,) float64 (LG: per-instant fill along s) or None
    groups : {name: (G,) array} descriptors (nodes, bins, distances, flags)
    extra : {name: array} (LT ensemble members)
    flags, meta : JSON dicts
    """

    def __init__(self, source, site, t, H=None, pix=None, g0=None, g1=None, w=None, *,
                 status=STATUS_OK, reason="", ref=None, demean=False, knots_s=None,
                 groups=None, extra=None, flags=None, meta=None):
        if source not in SOURCES:
            raise ValueError(f"unknown level source {source!r}")
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        self.source, self.site = source, str(site)
        self.status, self.reason = status, str(reason)
        self.t = _as_index(t) if len(pd.DatetimeIndex(t)) else pd.DatetimeIndex(t).as_unit("ns")
        self.demean = bool(demean)
        self.flags = dict(flags or {})
        self.meta = dict(meta or {})
        self.groups = {k: np.asarray(v) for k, v in (groups or {}).items()}
        self.extra = {k: np.asarray(v) for k, v in (extra or {}).items()}
        self.H = self.pix = self.g0 = self.g1 = self.w = self.ref = self.knots_s = None
        if ref is not None:
            self.ref = np.asarray(ref, np.float64)
            if self.ref.shape != (self.T,):
                raise ValueError("ref must hold h_B at every record instant")
        if knots_s is not None:
            self.knots_s = np.asarray(knots_s, np.float64)
        if status != STATUS_OK:
            return
        H = np.array(H, np.float64)
        if H.ndim != 2 or H.shape[1] != self.T:
            raise ValueError(f"H must be (G, {self.T}); got {H.shape}")
        H.flags.writeable = False
        self.H = H
        if self.demean and (self.ref is None or not np.isfinite(self.ref).all()):
            raise ValueError("a demeaned source needs finite h_B at every instant (ref)")
        if self.knots_s is not None and (self.knots_s.shape != (self.G,)
                                         or np.any(np.diff(self.knots_s) <= 0)):
            raise ValueError("knots_s must be strictly increasing, one per group")
        if pix is None:
            if self.G != 1:
                raise ValueError("only a one-row source may be uniform (pix=None)")
            return
        self.pix = np.asarray(pix, np.int64)
        if self.pix.ndim != 1 or (len(self.pix) and np.any(np.diff(self.pix) <= 0)):
            raise ValueError("pix must be sorted unique frame flat indices")
        self.g0 = np.asarray(g0, np.int32)
        self.g1 = np.asarray(self.g0 if g1 is None else g1, np.int32)
        self.w = np.zeros(len(self.pix)) if w is None else np.asarray(w, np.float64)
        for a in (self.g0, self.g1, self.w):
            if a.shape != self.pix.shape:
                raise ValueError("g0, g1 and w must align with pix")
        if np.any(self.g0 >= self.G) or np.any(self.g1 >= self.G) or \
                np.any((self.g0 < 0) != (self.g1 < 0)):
            raise ValueError("group indices out of range")
        if np.any((self.w < 0) | (self.w > 1)):
            raise ValueError("weights must lie in [0, 1]")

    # facts ------------------------------------------------------------------
    @property
    def T(self):
        return len(self.t)

    @property
    def G(self):
        return 0 if self.H is None else self.H.shape[0]

    @property
    def computable(self):
        return self.status == STATUS_OK

    def __repr__(self):
        return (f"<SourceLevels {self.source} {self.site}: {self.status}"
                f"{' (' + self.reason + ')' if self.reason else ''}, {self.G} groups x "
                f"{self.T} instants>")

    @classmethod
    def unavailable(cls, source, site, t, status, reason, flags=None, meta=None):
        """A status-only source (no rows): every unit that needs it is NOT
        COMPUTABLE / NOT AVAILABLE / not_applicable, never zero."""
        if status == STATUS_OK:
            raise ValueError("an unavailable source cannot have status 'ok'")
        return cls(source, site, t, status=status, reason=reason, flags=flags, meta=meta)

    def require(self):
        """Raise :class:`NotComputable` or :class:`NotAvailable` unless ok."""
        if self.status == STATUS_OK:
            return self
        if self.status in (STATUS_CREDENTIALS, STATUS_NOT_COMPUTABLE):
            raise NotComputable(self.source, self.status, self.reason)
        raise NotAvailable(self.source, self.status, self.reason)

    def _positions(self, sel):
        if sel is None:
            return np.arange(self.T)
        a = np.asarray(sel)
        if a.dtype == bool:
            if a.shape != (self.T,):
                raise ValueError(f"a boolean selection must have length {self.T}")
            return np.flatnonzero(a)
        a = a.astype(np.int64)
        if a.ndim != 1 or (a.size and (a.min() < 0 or a.max() >= self.T
                                       or np.any(np.diff(a) <= 0))):
            raise ValueError("positions must be sorted unique record positions in [0, T)")
        return a

    def _used(self, flat=None):
        """Groups the pixels (all, or ``flat``) read (both ends of a segment
        with a positive weight)."""
        if self.pix is None:
            return np.zeros(1, np.int64)
        i = self._pixel_index(flat)
        g0, g1, w = self.g0[i], self.g1[i], self.w[i]
        return np.unique(np.concatenate([g0[g0 >= 0], g1[(g1 >= 0) & (w > 0)]]))

    # reading ----------------------------------------------------------------
    def offsets(self, window=None):
        """Per group: the mean of ``H - h_B`` over the window's record
        positions (finite pairs); zeros for a source that is not demeaned."""
        self.require()
        if not self.demean:
            return np.zeros(self.G)
        if window is None:
            raise ValueError(f"{self.source} is demeaned against h_B over the window: pass "
                             "the window's record positions")
        pos = self._positions(window)
        D = self.H[:, pos] - self.ref[pos][None, :]
        n = np.isfinite(D).sum(axis=1)
        with np.errstate(invalid="ignore"):
            return np.where(n > 0, np.nansum(D, axis=1) / np.maximum(n, 1), np.nan)

    def rows(self, sel=None, window=None, allow_nan=False, flat=None):
        """Group rows ``(G, n)`` at the record positions ``sel`` (demeaned
        over ``window``; LG knots filled along s).

        Raises
        ------
        NotComputable, NotAvailable
            The source's status; or a NaN in a group the pixels read (an
            instant the source does not cover), unless ``allow_nan``.
        """
        self.require()
        cols = self._positions(sel)
        R = self.H[:, cols] - self.offsets(window)[:, None]
        if self.knots_s is not None:
            R = _fill_knots(R, self.knots_s)
        if not allow_nan:
            used = self._used(flat)
            bad = ~np.isfinite(R[used])
            if bad.any():
                n_inst = int(bad.any(axis=0).sum())
                raise NotAvailable(self.source, STATUS_NOT_AVAILABLE,
                                   f"no level at {n_inst} of {len(cols)} instants "
                                   f"(first {self.t[cols][bad.any(axis=0)][0]})")
        return R

    def _pixel_index(self, flat):
        if self.pix is None:
            raise ValueError("a uniform source has no pixel table")
        if flat is None:
            return np.arange(len(self.pix))
        flat = np.asarray(flat, np.int64)
        i = np.searchsorted(self.pix, flat)
        i = np.minimum(i, len(self.pix) - 1)
        if len(flat) and (len(self.pix) == 0 or not np.array_equal(self.pix[i], flat)):
            raise KeyError("some pixels are not in the source's pixel table")
        return i

    def pixel_levels(self, flat=None, sel=None, window=None, allow_nan=False):
        """Levels ``(n_pixels, n_instants)`` of frame pixels ``flat`` (default
        every pixel of the table). A pixel without a level (g0 = -1) is NaN."""
        if self.pix is None:
            R = self.rows(sel, window, allow_nan)
            n = 1 if flat is None else len(np.asarray(flat))
            return np.broadcast_to(R[0], (n, R.shape[1])).copy()
        i = self._pixel_index(flat)
        R = self.rows(sel, window, allow_nan, flat=flat)
        g0, g1, w = self.g0[i], self.g1[i], self.w[i]
        out = np.full((len(i), R.shape[1]), np.nan)
        m = g0 >= 0
        out[m] = R[g0[m]]
        lin = m & (w > 0)
        if lin.any():
            out[lin] = (1.0 - w[lin])[:, None] * R[g0[lin]] + w[lin][:, None] * R[g1[lin]]
        return out

    def band_series(self, flat, band_of, sel=None, window=None):
        """The per-band polygon series (§1.6: NIDEM and Coast X-Ray under
        LT/LM/LG): per 3-km band, the mean level of the band's pixels that
        have one (declared: equal pixel weights).

        Returns
        -------
        (bands (B,) int64, series (B, n) float64)
        """
        flat = np.asarray(flat, np.int64)
        band_of = np.asarray(band_of)
        if band_of.shape != flat.shape:
            raise ValueError("band_of must align with flat")
        L = self.pixel_levels(flat, sel, window)
        bands = np.unique(band_of[band_of >= 0]).astype(np.int64)
        out = np.full((len(bands), L.shape[1]), np.nan)
        ok = np.isfinite(L).all(axis=1)
        for k, b in enumerate(bands):
            m = (band_of == b) & ok
            if m.any():
                out[k] = L[m].mean(axis=0)
        return bands, out

    def unit_status(self, window=None, flat=None):
        """The status of a unit that heights ``flat`` with this source over
        the window: ``{"source", "status", "reason"}`` with status "ok",
        "NOT COMPUTABLE", "NOT AVAILABLE" or "not_applicable"."""
        out = {"source": self.source, "status": UNIT_STATUS[self.status],
               "reason": self.reason}
        if self.status != STATUS_OK:
            return out
        pos = np.arange(self.T) if window is None else self._positions(window)
        try:
            self.rows(pos, pos, flat=flat)
        except LevelSourceError as e:
            return {"source": self.source, "status": UNIT_STATUS[e.status], "reason": e.reason}
        return out

    # persistence ------------------------------------------------------------
    def save(self, path):
        """Byte-stable npz (``product.save_npz``)."""
        e = np.zeros(0)
        arrays = {"format": np.array(SOURCE_FORMAT), "source": np.array(self.source),
                  "site": np.array(self.site), "status": np.array(self.status),
                  "reason": np.array(self.reason),
                  "t_ns": self.t.asi8.astype(np.int64),
                  "H": e if self.H is None else self.H,
                  "pix": e.astype(np.int64) if self.pix is None else self.pix,
                  "uniform": np.array(self.pix is None),
                  "g0": e.astype(np.int32) if self.g0 is None else self.g0,
                  "g1": e.astype(np.int32) if self.g1 is None else self.g1,
                  "w": e if self.w is None else self.w,
                  "ref": e if self.ref is None else self.ref,
                  "has_ref": np.array(self.ref is not None),
                  "demean": np.array(self.demean),
                  "knots_s": e if self.knots_s is None else self.knots_s,
                  "has_knots": np.array(self.knots_s is not None),
                  "flags": np.array(json.dumps(self.flags, sort_keys=True, default=str)),
                  "meta": np.array(json.dumps(self.meta, sort_keys=True, default=str))}
        for k, v in sorted(self.groups.items()):
            arrays[f"groups.{k}"] = v
        for k, v in sorted(self.extra.items()):
            arrays[f"extra.{k}"] = v
        product.save_npz(path, arrays, compress=True)

    @classmethod
    def load(cls, path):
        z = product.load_npz(path)
        if str(z.get("format")) != SOURCE_FORMAT:
            raise ValueError(f"{path}: not a {SOURCE_FORMAT} file")
        ok = str(z["status"]) == STATUS_OK
        uniform = bool(z["uniform"])
        return cls(str(z["source"]), str(z["site"]),
                   pd.DatetimeIndex(z["t_ns"].view("M8[ns]")),
                   H=z["H"] if ok else None,
                   pix=None if (uniform or not ok) else z["pix"],
                   g0=None if (uniform or not ok) else z["g0"],
                   g1=None if (uniform or not ok) else z["g1"],
                   w=None if (uniform or not ok) else z["w"],
                   status=str(z["status"]), reason=str(z["reason"]),
                   ref=z["ref"] if bool(z["has_ref"]) else None, demean=bool(z["demean"]),
                   knots_s=z["knots_s"] if bool(z["has_knots"]) else None,
                   groups={k[7:]: v for k, v in z.items() if k.startswith("groups.")},
                   extra={k[6:]: v for k, v in z.items() if k.startswith("extra.")},
                   flags=json.loads(str(z["flags"])), meta=json.loads(str(z["meta"])))


def source_path(common_dir, source, tag=""):
    """``<common>/levels_<source>[_<tag>].npz`` (results/validation_v2/common/<site>/)."""
    return os.path.join(common_dir, f"levels_{source}{'_' + tag if tag else ''}.npz")


# ─────────────────────────────────────────────────────────────────────────────
#  LB: the boundary
# ─────────────────────────────────────────────────────────────────────────────

def level_lb(table, site=""):
    """LB as a uniform one-row source: ``h_B(t)`` from the exact level table
    (row tau = 0), the same for every pixel."""
    return SourceLevels("LB", site, table.t, H=table.h0[None, :], pix=None,
                        flags={"instants": table.meta.get("instants_source")},
                        meta={"rule": LEVEL_RULES["LB"]["table"]})


# ─────────────────────────────────────────────────────────────────────────────
#  LT: the per-pixel global tide model
# ─────────────────────────────────────────────────────────────────────────────

def level_lt(grid, t, pix, site="", models=None, directory=None, ranking=None,
             nodes=None, pixel_tides=None, node_m=None, buffer_m=None):
    """LT rows: eo-tides node series and the nearest-node pixel table.

    Parameters
    ----------
    grid : inputs.Grid or mapping
    t : DatetimeIndex
        Record instants (naive UTC).
    pix : (n,) sorted frame flat indices (the package's dom)
    models : list of (model, eo-tides name), optional
        Default ``fetch.tides.lt_models(directory)``.
    ranking : dict, optional
        ``{"kind": "eo-tides" | "bt26", "ranks": {model: rank} or (G, M)
        array over the USED nodes}``; required for an ensemble.
    nodes : dict, optional
        A precomputed ``fetch.tides.node_tides`` result (the fetch cache).
    pixel_tides : callable, optional
        Injected for tests.

    Returns
    -------
    SourceLevels
        Status "ok", or "not computable" for an ensemble without ranking
        (its members are kept in ``extra['members']``: :func:`lt_with_ranking`
        finishes it without recomputing the tides).
    """
    from experiments.benchmark.fetch import tides as ft

    directory = directory or config.TIDE_DIR
    t = _as_index(t)
    if models is None:
        models = ft.lt_models(directory)
    models = [tuple(m) for m in models]
    rule = ft.lt_rule(models)
    use = models if rule == "ensemble" else models[:1]
    if nodes is None:
        nodes = ft.node_tides(grid, t, use, directory=directory,
                              node_m=node_m or ft.NODE_M, buffer_m=buffer_m or ft.BUFFER_M,
                              pixel_tides=pixel_tides)
    if list(nodes["models"]) != [m for m, _ in use]:
        raise ValueError(f"node tides hold {nodes['models']}, the rule needs "
                         f"{[m for m, _ in use]}")
    pix = np.asarray(pix, np.int64)
    node_of = ft.pixel_nodes(grid, pix, nodes["x"], nodes["y"])
    used = np.unique(node_of)
    members = np.asarray(nodes["H"], np.float64)[:, used, :]
    nx = len(nodes["x"])
    groups = {"node": used, "node_x": np.asarray(nodes["x"])[used % nx],
              "node_y": np.asarray(nodes["y"])[used // nx]}
    g0 = np.searchsorted(used, node_of).astype(np.int32)
    flags = {"rule": rule, "models": [m for m, _ in use], "names": [n for _, n in use],
             "pixel_to_node": "nearest (declared)", "instants": None}
    meta = {"node_m": float(node_m or ft.NODE_M), "buffer_m": float(buffer_m or ft.BUFFER_M),
            "installed_candidates": [m for m, _ in models]}
    if not np.isfinite(members).all():
        return SourceLevels.unavailable("LT", site, t, STATUS_NOT_AVAILABLE,
                                        "the tide model gives non-finite levels at some "
                                        "node and instant", flags, meta)
    if rule == "primary":
        return SourceLevels("LT", site, t, H=members[0], pix=pix, g0=g0, groups=groups,
                            flags=flags, meta=meta)
    src = SourceLevels("LT", site, t, status=STATUS_NOT_COMPUTABLE,
                       reason=("LT is the ranked ensemble of "
                               f"{[m for m, _ in use]} and needs the local ranking (eo-tides' "
                               "default ranking points cover Australia only; supply the "
                               "truth-free BT26-rank of L12)"),
                       groups=groups, extra={"members": members, "pix": pix, "g0": g0},
                       flags=flags, meta=meta)
    return src if ranking is None else lt_with_ranking(src, ranking)


def lt_with_ranking(src, ranking):
    """Finish an ensemble LT with a ranking: the equal-weight mean of the
    members ranked <= top_n (``fetch.tides.TOP_N[kind]``) per node."""
    from experiments.benchmark.fetch import tides as ft

    kind = ranking.get("kind")
    if kind not in ft.TOP_N:
        raise ValueError(f"unknown ranking kind {kind!r}; expected one of {sorted(ft.TOP_N)}")
    members = np.asarray(src.extra["members"], np.float64)
    names = list(src.flags["models"])
    ranks = ranking["ranks"]
    if isinstance(ranks, dict):
        missing = [m for m in names if m not in ranks]
        if missing:
            raise ValueError(f"the ranking lacks {missing}")
        ranks = np.array([float(ranks[m]) for m in names])
    H, use = ft.combine(members, ranks, ft.TOP_N[kind])
    flags = {**src.flags, "ranking": kind, "top_n": int(ft.TOP_N[kind])}
    groups = {**src.groups, "members_used": use}
    return SourceLevels("LT", src.site, src.t, H=H, pix=src.extra["pix"],
                        g0=src.extra["g0"], groups=groups, flags=flags,
                        meta={**src.meta, "ranking": {k: v for k, v in ranking.items()
                                                      if k != "ranks"}})


# ─────────────────────────────────────────────────────────────────────────────
#  LM: the regional hydrodynamic model
# ─────────────────────────────────────────────────────────────────────────────

def cubic_at(t_model, Z, t_query):
    """4-point cubic Lagrange interpolation on a regular time axis.

    Parameters
    ----------
    t_model : DatetimeIndex (nt,), regularly spaced (gaps allowed only as
        missing steps: the series is re-indexed onto the regular axis)
    Z : (nt, C) float64
    t_query : DatetimeIndex (nq,)

    Returns
    -------
    (C, nq) float64; NaN where the stencil (i-1, i, i+1, i+2) leaves the
    axis or holds a NaN. An instant on a model step returns that sample.
    """
    tm = pd.DatetimeIndex(t_model).asi8
    Z = np.asarray(Z, np.float64)
    if Z.ndim != 2 or Z.shape[0] != len(tm):
        raise ValueError("Z must be (nt, C) aligned with t_model")
    if len(tm) < 4:
        return np.full((Z.shape[1], len(t_query)), np.nan)
    d = np.diff(tm)
    if np.any(d <= 0):
        raise ValueError("model times must be strictly increasing")
    step = int(np.min(d))
    if np.any(d % step):
        raise ValueError("model times are not on one regular step")
    if np.any(d != step):                          # missing steps -> NaN rows
        full = np.arange(tm[0], tm[-1] + 1, step)
        Zf = np.full((len(full), Z.shape[1]), np.nan)
        Zf[(tm - tm[0]) // step] = Z
        tm, Z = full, Zf
    q = pd.DatetimeIndex(t_query).asi8
    u = (q - tm[0]) / step
    i = np.floor(u).astype(np.int64)
    f = u - i
    out = np.full((Z.shape[1], len(q)), np.nan)
    ok = (i - 1 >= 0) & (i + 2 <= len(tm) - 1)
    exact = (f == 0) & (i >= 0) & (i <= len(tm) - 1)
    if ok.any():
        ii, ff = i[ok], f[ok]
        w = np.stack([-ff * (ff - 1) * (ff - 2) / 6.0, (ff + 1) * (ff - 1) * (ff - 2) / 2.0,
                      -(ff + 1) * ff * (ff - 2) / 2.0, (ff + 1) * ff * (ff - 1) / 6.0])
        acc = np.zeros((Z.shape[1], len(ii)))
        for k in range(4):
            acc += w[k][None, :] * Z[ii - 1 + k].T
        out[:, ok] = acc
    if exact.any():
        out[:, exact] = Z[i[exact]].T
    return out


def level_lm(site, t, pix, pix_s_km, grid, fields, ref, dataset_id, manifest=None,
             stamp_shift_min=None, bin_km=1.0, unresolved_km=1.0):
    """LM rows: regional SSH per 1-km s bin at the record instants (§1.6).

    Parameters
    ----------
    site : str
    t : DatetimeIndex
        Record instants (naive UTC).
    pix, pix_s_km : (n,) frame flat indices (sorted) and their s (km, NaN
        where the pixel has no s)
    grid : inputs.Grid or mapping
    fields : iterable of paths (``fetch.cmems.read_subset``) or of dicts
        ``{t, lon, lat, zos}``; one per (padded) year
    ref : (T,) h_B at the instants (the demeaning reference)
    dataset_id : str
        A ``fetch.cmems.DATASETS`` key.
    manifest : dict, optional
        ``fetch.cmems`` manifest; a status other than "ok" (credentials
        missing, not available) returns a status-only source and reads no
        field.
    stamp_shift_min : float, optional
        Minutes added to the model stamps; required for an hourly-mean
        dataset (measured by ``fetch.cmems.measure_stamp_shift``).

    Returns
    -------
    SourceLevels
        Demeaned against h_B over the window at read time; ``groups``:
        ``bin`` (s bin index), ``s_lo_km``, ``n_cells``, ``d_wet_km``,
        ``unresolved``; flags ``degenerate`` (Villaviciosa).
    """
    from scipy.spatial import cKDTree
    from pyproj import Transformer

    from experiments.benchmark.fetch import cmems
    from experiments.benchmark.fetch import tides as ft

    t = _as_index(t)
    cat = cmems.DATASETS.get(dataset_id)
    flags = {"dataset": dataset_id, "degenerate": site in cmems.DEGENERATE,
             "model": None if cat is None else cat["model"]}
    if manifest is not None and manifest.get("status") != cmems.STATUS_OK:
        st = manifest.get("status")
        status = STATUS_CREDENTIALS if st == cmems.STATUS_CREDENTIALS else STATUS_NOT_AVAILABLE
        return SourceLevels.unavailable("LM", site, t, status,
                                        manifest.get("reason") or str(st), flags)
    if cat is None:
        return SourceLevels.unavailable("LM", site, t, STATUS_NOT_AVAILABLE,
                                        f"unknown dataset {dataset_id}", flags)
    if cat["kind"] == "m" and stamp_shift_min is None:
        return SourceLevels.unavailable(
            "LM", site, t, STATUS_NOT_COMPUTABLE,
            f"{dataset_id} holds hourly MEANS: measure its stamp convention first "
            "(fetch.cmems.measure_stamp_shift)", flags)
    shift = pd.Timedelta(minutes=float(stamp_shift_min or 0.0))
    flags["stamp_shift_min"] = float(stamp_shift_min or 0.0)
    xs, ys, crs = ft.grid_axes(grid)
    W = len(xs)
    pix = np.asarray(pix, np.int64)
    s = np.asarray(pix_s_km, np.float64)
    if s.shape != pix.shape:
        raise ValueError("pix_s_km must align with pix")
    has_s = np.isfinite(s)
    if not has_s.any():
        return SourceLevels.unavailable("LM", site, t, STATUS_NOT_AVAILABLE,
                                        "no pixel has an s value", flags)
    bin_px = np.full(len(pix), -1, np.int64)
    bin_px[has_s] = np.floor(s[has_s] / float(bin_km)).astype(np.int64)
    bins = np.unique(bin_px[has_s])
    ps = np.flatnonzero(has_s)
    pxy = np.c_[xs[pix[ps] % W], ys[pix[ps] // W]]

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    vals = cells = None
    n_fields = 0
    for fld in fields:
        if not isinstance(fld, dict):
            fld = cmems.read_subset(fld)
        lon, lat, Z = np.asarray(fld["lon"]), np.asarray(fld["lat"]), np.asarray(fld["zos"])
        if cells is None:
            LON, LAT = np.meshgrid(lon, lat)
            wet = np.isfinite(Z).all(axis=0)
            cx, cy = tr.transform(LON[wet], LAT[wet])
            cells = {"lon": lon, "lat": lat, "wet": wet, "xy": np.c_[cx, cy]}
            vals = np.full((int(wet.sum()), len(t)), np.nan)
        elif not (np.array_equal(lon, cells["lon"]) and np.array_equal(lat, cells["lat"])):
            raise ValueError("the yearly subsets do not share the model grid")
        v = cubic_at(pd.DatetimeIndex(fld["t"]) + shift, Z[:, cells["wet"]], t)
        fill = ~np.isfinite(vals) & np.isfinite(v)
        vals[fill] = v[fill]
        n_fields += 1
    if cells is None or not len(cells["xy"]):
        return SourceLevels.unavailable("LM", site, t, STATUS_NOT_AVAILABLE,
                                        "no wet model cell in the subset", flags)
    C = len(cells["xy"])
    # channel cells: the nearest finite-s pixel within unresolved_km
    d_c, i_c = cKDTree(pxy).query(cells["xy"])
    match = d_c <= float(unresolved_km) * 1000.0
    cell_bin = np.where(match, bin_px[ps][i_c], -1)
    # each pixel's nearest wet cell
    d_p, i_p = cKDTree(cells["xy"]).query(pxy)
    G = len(bins)
    H = np.full((G, len(t)), np.nan)
    n_cells = np.zeros(G, np.int64)
    d_wet = np.zeros(G)
    for k, b in enumerate(bins):
        m = bin_px[ps] == b
        j = int(np.argmin(np.where(m, d_p, np.inf)))
        d_wet[k] = d_p[j] / 1000.0
        own = np.flatnonzero(cell_bin == b)
        n_cells[k] = len(own)
        H[k] = vals[own].mean(axis=0) if len(own) else vals[i_p[j]]
    unresolved = d_wet > float(unresolved_km)
    g0 = np.full(len(pix), -1, np.int32)
    g0[has_s] = np.searchsorted(bins, bin_px[has_s]).astype(np.int32)
    groups = {"bin": bins, "s_lo_km": bins * float(bin_km), "n_cells": n_cells,
              "d_wet_km": d_wet, "unresolved": unresolved}
    flags.update(n_wet_cells=int(C), n_channel_cells=int(match.sum()),
                 n_unresolved_bins=int(unresolved.sum()), n_fields=n_fields)
    meta = {"bin_km": float(bin_km), "unresolved_km": float(unresolved_km),
            "catalogue": cat, "rule": LEVEL_RULES["LM"]["bin_series"]}
    if manifest is not None:
        meta["files"] = manifest.get("files")
    log(f"[levels] LM {site}: {C} wet cells ({int(match.sum())} channel) -> {G} s bins, "
        f"{int(unresolved.sum())} unresolved; {int(np.isfinite(H).all(axis=0).sum())}/"
        f"{len(t)} instants covered")
    return SourceLevels("LM", site, t, H=H, pix=pix, g0=g0, ref=ref, demean=True,
                        groups=groups, flags=flags, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
#  LG: the gauge network along s (labelled upper bound)
# ─────────────────────────────────────────────────────────────────────────────

def gauge_at(series, t, max_gap_min=LG_MAX_GAP_MIN):
    """A gauge series at the instants: linear interpolation in time between
    the bracketing samples when both exist and are at most ``max_gap_min``
    apart; the sample itself on a sample instant; NaN otherwise."""
    s = series.dropna()
    x = pd.DatetimeIndex(s.index).asi8
    v = s.to_numpy(np.float64)
    q = pd.DatetimeIndex(t).asi8
    out = np.full(len(q), np.nan)
    if not len(x):
        return out
    j = np.searchsorted(x, q, side="right")          # x[j-1] <= q < x[j]
    on = (j >= 1) & (x[np.maximum(j - 1, 0)] == q)
    out[on] = v[j[on] - 1]
    mid = ~on & (j >= 1) & (j < len(x))
    a, b = j[mid] - 1, j[mid]
    gap = x[b] - x[a]
    ok = gap <= float(max_gap_min) * NS_PER_MIN
    f = (q[mid] - x[a]) / gap
    vals = v[a] + f * (v[b] - v[a])
    tmp = np.full(int(mid.sum()), np.nan)
    tmp[ok] = vals[ok]
    out[mid] = tmp
    return out


def lg_knots(site, t, s_g_km, base=None, path=None, max_gap_min=LG_MAX_GAP_MIN,
             conventions=None):
    """The LG knots of a site from the judge CSVs.

    Parameters
    ----------
    site : str
    t : DatetimeIndex
    s_g_km : mapping
        ``{code: s_g}`` of the INCLUDED judges (the judges table); codes of
        ``config.SAME_LOCATION`` pairs become one knot (mean series, mean s).
        A judge whose s_g is not a finite positive distance is dropped
        (logged): it cannot be placed along s.
    conventions : mapping, optional
        The time-stamp conventions (default ``judges.TIMESTAMP_CONVENTIONS``):
        each series is centred with its provider's convention before its
        10-min centring, exactly as the lag truths read it.

    Returns
    -------
    list of dict ``{"code", "s_km", "values" (T,)}``
    """
    from experiments.benchmark import judges

    t = _as_index(t)
    bad = {c: v for c, v in s_g_km.items()
           if v is None or not np.isfinite(float(v)) or float(v) <= 0}
    if bad:
        log(f"[levels] LG {site}: judges without a positive finite s_g dropped: {bad}")
    s_g_km = {c: float(v) for c, v in s_g_km.items() if c not in bad}
    stas = {st["code"]: st for st in judges.stations(site, path)}
    grid = pd.date_range(t[0].floor("10min") - pd.Timedelta(minutes=20),
                         t[-1].ceil("10min") + pd.Timedelta(minutes=20), freq="10min")
    vals = {}
    for code in s_g_km:
        st = stas[code]
        raw = judges.centred(judges.read_series(st, base), st, conventions)
        ser = judges.centred_10min(raw, st["sampling_min"], grid)
        vals[code] = gauge_at(ser, t, max_gap_min)
    knots, done = [], set()
    for code in s_g_km:
        if code in done:
            continue
        pair = next((p for p in config.SAME_LOCATION if code in p), None)
        group = [c for c in (pair or (code,)) if c in s_g_km]
        done.update(group)
        if len(group) > 1:
            st = np.vstack([vals[c] for c in group])
            n = np.isfinite(st).sum(axis=0)
            v = np.where(n > 0, np.nansum(st, axis=0) / np.maximum(n, 1), np.nan)
        else:
            v = vals[code]
        knots.append({"code": "+".join(group), "s_km": float(np.mean([s_g_km[c] for c in group])),
                      "values": v})
    return knots


def level_lg(site, t, ref, knots, pix, pix_s_km, demean=True):
    """LG rows: h_B at s = 0 and the judges at s_g, linear in s, constant
    beyond the last gauge (§1.6). Labelled upper bound.

    Parameters
    ----------
    t : DatetimeIndex
    ref : (T,) h_B at the instants (knot 0)
    knots : list of {"code", "s_km" (> 0), "values" (T,)}
    pix, pix_s_km : frame flat indices (sorted) and their s (km)
    demean : bool
        Each judge against h_B over the window (default, declared); False is
        the 'native datum' sensitivity.

    Returns
    -------
    SourceLevels (status "not applicable" at a site outside LEVEL['LG'].sites)
    """
    from experiments.benchmark import settings

    t = _as_index(t)
    flags = {"upper_bound": True, "confirmatory": False, "lag_scoring": False,
             "label": "uses in-estuary gauges (upper bound)"}
    if site not in settings.LEVEL["LG"].sites:
        return SourceLevels.unavailable("LG", site, t, STATUS_NOT_APPLICABLE,
                                        f"LG is defined at {settings.LEVEL['LG'].sites}", flags)
    dropped = [k.get("code") for k in knots
               if k.get("s_km") is None or not np.isfinite(float(k["s_km"]))
               or float(k["s_km"]) <= 0]
    if dropped:
        flags["dropped_knots"] = dropped
        knots = [k for k in knots if k.get("code") not in dropped]
    if not knots:
        return SourceLevels.unavailable("LG", site, t, STATUS_NOT_AVAILABLE,
                                        "no included judge with a positive finite s_g", flags)
    knots = sorted(knots, key=lambda k: k["s_km"])
    s = np.array([0.0] + [float(k["s_km"]) for k in knots])
    if np.any(np.diff(s) <= 0):
        raise ValueError(f"judge locations must lie at distinct s > 0; got {s[1:].tolist()}")
    ref = np.asarray(ref, np.float64)
    H = np.vstack([ref] + [np.asarray(k["values"], np.float64) for k in knots])
    pix = np.asarray(pix, np.int64)
    ps = np.asarray(pix_s_km, np.float64)
    g0 = np.full(len(pix), -1, np.int32)
    g1 = np.full(len(pix), -1, np.int32)
    w = np.zeros(len(pix))
    fin = np.isfinite(ps)
    sp = np.clip(ps[fin], s[0], s[-1])               # s < 0: h_B; beyond: last gauge
    j = np.clip(np.searchsorted(s, sp, side="right") - 1, 0, len(s) - 1)   # s[j] <= sp
    last = j == len(s) - 1
    j1 = np.where(last, j, j + 1)
    ww = np.where(last, 0.0, (sp - s[j]) / np.where(last, 1.0, s[j1] - s[j]))
    ww = np.clip(ww, 0.0, 1.0)
    g0[fin], g1[fin], w[fin] = j, np.where(ww > 0, j1, j), ww
    groups = {"s_km": s, "code": np.array(["h_B"] + [k["code"] for k in knots]),
              "coverage": np.isfinite(H).mean(axis=1)}
    flags.update(n_knots=len(knots), datum="demeaned vs h_B" if demean else "native")
    return SourceLevels("LG", site, t, H=H, pix=pix, g0=g0, g1=g1, w=w, ref=ref,
                        demean=bool(demean), knots_s=s, groups=groups, flags=flags,
                        meta={"rule": LEVEL_RULES["LG"]})


# ─────────────────────────────────────────────────────────────────────────────
#  the gauge band lags of the attribution row E0g (user decision 2026-10-15)
# ─────────────────────────────────────────────────────────────────────────────

#: the frozen rule of :func:`gauge_band_lags`
GAUGE_LAG_RULE = {
    "lag": ("truth (a), tau_a, of each included judge at the unit's window (declared; "
            "tau_ab and (b) are sensitivities)"),
    "knots": ("s = 0: tau = 0 (MAREA's anchor); each included judge at s_g, "
              "config.SAME_LOCATION pairs averaged (mean s, mean lag), as LG"),
    "monotone": ("isotonic (pool-adjacent-violators, equal weights) over the judges in s "
                 "order, then clipped at 0, the anchor: MAREA's band lags are monotone "
                 "non-decreasing from tau(0) = 0"),
    "interpolation": "linear in s; constant beyond the last gauge; 0 for s <= 0",
    "band": "a band's lag = the curve at the band's representative s (the mean s of its "
            "domain pixels)",
    "use": ("the attribution row E0g only (MAREA's inversion with these lags; labelled "
            "'uses in-estuary gauges'); never in lag scoring nor a confirmatory family"),
}


def _isotonic(y):
    """Non-decreasing least-squares fit of ``y`` (equal weights,
    pool-adjacent-violators)."""
    blocks = []                                      # [sum, count]
    for v in np.asarray(y, np.float64):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    return np.concatenate([np.full(n, s / n) for s, n in blocks]) if blocks else np.zeros(0)


def gauge_band_lags(lags_min, s_g_km, band_s_km, pairs=None):
    """The band lags of the attribution row E0g: the in-estuary judges'
    measured lags, placed along s as LG places their levels
    (:data:`GAUGE_LAG_RULE`).

    Parameters
    ----------
    lags_min : mapping
        ``{code: lag (min)}`` of the INCLUDED judges (truth (a), tau_a, at
        the unit's window).
    s_g_km : mapping
        ``{code: s_g}`` (the judges table). A judge without a positive
        finite s_g, or without a finite lag, is dropped (reported).
    band_s_km : array (B,)
        The representative s of each band (the mean s of its domain
        pixels); NaN gives NaN.
    pairs : sequence of tuples, optional
        Judges at the same location (default ``config.SAME_LOCATION``):
        one knot, mean s and mean lag.

    Returns
    -------
    dict
        ``status`` ("ok" or "not available"), ``reason``, ``tau_band`` (B,)
        float64 (min), ``codes``, ``knots_s_km``, ``raw_tau_min`` and
        ``knots_tau_min`` (after the monotone fit; anchor excluded),
        ``monotone_adjusted`` and ``dropped`` ``{code: reason}``.
    """
    pairs = config.SAME_LOCATION if pairs is None else pairs
    band_s = np.asarray(band_s_km, np.float64)
    dropped, ok = {}, {}
    for code, lag in lags_min.items():
        s = s_g_km.get(code)
        if s is None or not np.isfinite(float(s)) or float(s) <= 0:
            dropped[code] = "no positive finite s_g"
        elif lag is None or not np.isfinite(float(lag)):
            dropped[code] = "no finite lag"
        else:
            ok[code] = (float(s), float(lag))
    knots, done = [], set()
    for code in ok:
        if code in done:
            continue
        pair = next((p for p in pairs if code in p), None)
        group = [c for c in (pair or (code,)) if c in ok]
        done.update(group)
        knots.append(("+".join(group), float(np.mean([ok[c][0] for c in group])),
                      float(np.mean([ok[c][1] for c in group]))))
    out = {"status": "ok", "reason": "", "tau_band": np.full(band_s.shape, np.nan),
           "codes": [], "knots_s_km": [], "raw_tau_min": [], "knots_tau_min": [],
           "monotone_adjusted": False, "dropped": dropped, "rule": GAUGE_LAG_RULE}
    if not knots:
        out.update(status="not available", reason="no included judge with a positive finite "
                                                   "s_g and a finite lag")
        return out
    knots.sort(key=lambda k: k[1])
    s = np.array([k[1] for k in knots])
    if np.any(np.diff(s) <= 0):
        raise ValueError(f"judge locations must lie at distinct s > 0; got {s.tolist()}")
    raw = np.array([k[2] for k in knots])
    fit = np.maximum(_isotonic(raw), 0.0)
    tau = np.full(band_s.shape, np.nan)
    fin = np.isfinite(band_s)
    tau[fin] = np.interp(band_s[fin], np.r_[0.0, s], np.r_[0.0, fit])
    out.update(tau_band=tau, codes=[k[0] for k in knots], knots_s_km=s.tolist(),
               raw_tau_min=raw.tolist(), knots_tau_min=fit.tolist(),
               monotone_adjusted=bool(np.any(np.abs(fit - raw) > 1e-9)))
    return out
