# -*- coding: utf-8 -*-
"""Validation v2 — L14 / L15: global ocean tide models at the gauges (spec §3.3, §3.3.1, §6.3).

Question
--------
What lag does a global ocean tide model give at each in-estuary gauge,
read out exactly as the gauge itself is read out by the lag truths? These
rows are lower bounds (spec §3.3.1: the 1/8 deg models sit 20-33 km away
from the inner gauges) and the "primary global model" comparison of the
confirmatory family L-P1 (spec §6.5).

Pre-registered rules (spec §3.3 rows L14, L15; §3.3.1 "L14/L15/L16"; §6.3
"Point"; CONFIG below, hashed into the unit's config key)
-----------------------------------------------------------------------
* Models: the installed members of :data:`KNOWN_MODELS` (EOT20, GOT4.10c,
  GOT5.5, GOT5.6, FES2014, FES2022, TPXO9, TPXO10, HAMTIDE11, DTU23; the
  eo-tides / pyTMD names in preference order, as ``fetch.tides``). The
  PRIMARY model is FES2022 if installed, else EOT20 [V16]; its name is
  recorded in ``point[code]["model"]``. A primary model that is not
  installed is NOT COMPUTABLE for L14 (never replaced by another model).
* Each model at the gauge coordinates through ``truth.model_provider``
  (``PyTMDBoundary``: pyTMD's nearest ocean cell, extrapolate, no cutoff);
  ``dist_km`` = ``tidecheck.nearest_ocean_km`` (the distance to the nearest
  model cell that predicts a tide).
* Read-out (i), the model in place of the gauge in the truth machinery
  (``truth.e_rows``, ``best_lag``, ``gain_aware``, ``harmonic_fit``,
  ``hw_lw_window``), against h_B = ``inst.levels.stack(truth.TAUS_A)`` at
  the instance's instants and ``inst.series10`` inside the window:

  - ``a``: argmin over tau in [-60, 240] min (1 min) of the median-centred
    RMS of h_B(t_i - tau) - M(t_i) (truth (a));
  - ``ab``: the same with a free gain (truth tau_ab);
  - ``b``: wrap180(phi_M2(model) - phi_M2(h_B)) x 60 / 28.9841042 on the
    window's 10-min grid (``harmonic_fit`` with truth.CONS_B, GOT nodal
    corrections, order 1: the tau of ``truth.m2_lag``; its jackknife SE is
    not needed for a point read-out and is not computed);
  - ``hwlw``: the mean of the HW and LW median lags of the model's 10-min
    series against the boundary's low-passed marks (``truth.hw_lw_window``).

* Read-out (ii), model-internal: the model at the gauge against the SAME
  model at the boundary point, the equal-weight mean of the model at
  ``inst.boundary_members`` (the EOT20 point of the production boundary at
  Villaviciosa): ``internal_a`` (E(tau) with the boundary-point model in
  place of h_B) and ``internal_b`` (M2 phase difference). It is flagged
  "degenerate" when the gauge and a boundary member fall in the same or
  adjacent model cells (:func:`same_or_adjacent`, declared).
* L15: the equal-weight mean of the AVAILABLE models' levels (at the gauge
  and at the boundary point), with the same read-outs; it needs at least
  two models, else L15 is NOT COMPUTABLE. Its entry carries the members'
  'unresolved' and 'degenerate' flags.
* Flag 'unresolved' (never a change of the estimate): the gauge's nearest
  wet model cell is farther than the model's grid spacing (MODEL_RES_DEG x
  111.32 km), or no wet cell lies within the 0.5 deg probe (dist_km inf, as
  GOT4.10c at the Scheldt gauges); 'resolution unknown' when the probe
  failed (:func:`resolution_flag`).
* An unloadable model is recorded as ``available: False`` with its reason;
  nothing is ever substituted and a missing model never reads as zero.
* No gauge series is read (spec §0.5: in-estuary gauges are the lag truth):
  only the judges' coordinates.

Outputs: two point products (``common.lag_result`` kind "point",
lag_kind "linear"; ``b`` and ``internal_b`` are M2 phases wrapped to
+-180 deg, i.e. +-372.6 min, which the scorer treats circularly:
``scoring.point_kinds``).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from experiments.benchmark import config, settings, truth
from experiments.benchmark.lags import common
from experiments.benchmark.product import NotComputable

NS_PER_MIN = 60_000_000_000
NAN = float("nan")

#: (label, eo-tides / pyTMD names in order of preference); the order is the
#: report order. Labels are the spec's (§3.3 L14).
KNOWN_MODELS = (
    ("EOT20", ("EOT20",)),
    ("GOT4.10c", ("GOT4.10_nc", "GOT4.10", "GOT4.10_SAL")),
    ("GOT5.5", ("GOT5.5", "GOT5.5_extrapolated")),
    ("GOT5.6", ("GOT5.6", "GOT5.6_extrapolated")),
    ("FES2014", ("FES2014", "FES2014_extrapolated")),
    ("FES2022", ("FES2022", "FES2022_extrapolated")),
    ("TPXO9", ("TPXO9-atlas-v5-nc", "TPXO9-atlas-v5")),
    ("TPXO10", ("TPXO10-atlas-v2-nc", "TPXO10-atlas-v2")),
    ("HAMTIDE11", ("HAMTIDE11",)),
    ("DTU23", ("DTU23",)),
)
#: the primary global model of L-P1 [V16]: the first installed of these
PRIMARY_ORDER = ("FES2022", "EOT20")
#: native grid spacing (deg) of each model, for the degeneracy flag (declared,
#: from the models' documentation)
MODEL_RES_DEG = {"EOT20": 0.125, "GOT4.10c": 0.5, "GOT5.5": 0.125, "GOT5.6": 0.125,
                 "FES2014": 1.0 / 16.0, "FES2022": 1.0 / 30.0, "TPXO9": 1.0 / 30.0,
                 "TPXO10": 1.0 / 30.0, "HAMTIDE11": 0.125, "DTU23": 0.125}
ENSEMBLE_MIN_MODELS = 2
#: padding (h) of the model's 10-min series around the window for HW/LW
#: (truth.hw_lw_window pads the window by 12 h and bridges 20 min)
HW_PAD_H = 13
#: harmonic constituents of the (b) read-out: None = truth.CONS_B (the
#: truth's own list); tests set a short list for speed
CONSTITUENTS = None
#: the probe of tidecheck.nearest_ocean_km (its defaults)
OCEAN_PROBE = {"search_deg": 0.5, "step_deg": 0.02}

CONFIG = {
    "rule": ("L14: every installed known global tide model at each in-frame judge "
             "(truth.model_provider: pyTMD nearest ocean cell); read-out (i) against h_B: "
             "(a) argmin E(tau), tau_ab with a free gain, (b) the M2 phase lag on the "
             "window's 10-min grid, HW/LW mean lag; read-out (ii) model-internal against the "
             "same model at the boundary point (equal-weight mean over the boundary "
             "members): internal (a) and (b), 'degenerate' when the gauge and a member fall "
             "in the same or adjacent model cells. L15: the equal-weight mean of the "
             "available models' levels, the same read-outs, >= 2 models."),
    "known_models": [[label, list(names)] for label, names in KNOWN_MODELS],
    "primary_order": list(PRIMARY_ORDER),
    "model_res_deg": dict(MODEL_RES_DEG),
    "taus_a_min": [int(truth.TAUS_A[0]), int(truth.TAUS_A[-1]), 1],
    "b": {"constituents": "truth.CONS_B", "corrections": truth.B_CORRECTIONS,
          "order": truth.B_ORDER, "estimand": "wrap180(phi_M2(model) - phi_M2(ref)) x "
                                              "60 / 28.9841042 (the tau of truth.m2_lag)"},
    "hwlw": {"function": "truth.hw_lw_window", "series": "the model at 10 min over the "
             "window +- 13 h", "value": "mean_min (mean of the HW and LW medians)"},
    "boundary_point": ("equal-weight mean of the model at inst.boundary_members (IOC "
                       "members and the model point of a model-only boundary)"),
    "degenerate": ("cell index floor(coordinate / model resolution) of the nearest-ocean "
                   "probe (tidecheck.nearest_ocean_km) of the gauge and of each boundary "
                   "member, grid aligned at 0 deg (declared); same or adjacent (Chebyshev "
                   "distance <= 1) for any member"),
    "ensemble_min_models": ENSEMBLE_MIN_MODELS,
    "ocean_probe": dict(OCEAN_PROBE),
    "hw_pad_h": HW_PAD_H,
    "gauge_series": "never read (coordinates only)",
}

_COST = {"wh_cell": 0.02, "ram_gb": 1.0}           # designs.METHODS["L14"]
_SIGNATURE = {"reads": ["window"], "domain": False}
_REFS_MODELS = [
    {"cite": "Hart-Davis MG et al. (2021) EOT20: a global ocean tide model from multi-mission "
             "satellite altimetry, ESSD 13:3869-3884",
     "doi_or_url": "doi:10.5194/essd-13-3869-2021"},
    {"cite": "Lyard FH et al. (2021) FES2014 global ocean tide atlas, Ocean Sci. 17:615-649",
     "doi_or_url": "doi:10.5194/os-17-615-2021"},
    {"cite": "Egbert GD, Erofeeva SY (2002) Efficient inverse modeling of barotropic ocean "
             "tides (TPXO), J. Atmos. Oceanic Technol. 19:183-204",
     "doi_or_url": "doi:10.1175/1520-0426(2002)019<0183:EIMOBO>2.0.CO;2"},
    {"cite": "Bishop-Taylor R et al. (2025) eo-tides, JOSS",
     "doi_or_url": "doi:10.21105/joss.07786"},
    {"cite": "pyTMD (Sutterley et al.), Python-based tidal prediction software",
     "doi_or_url": "https://github.com/pyTMD/pyTMD"},
]
_DEVIATIONS = [
    {"paper": "v13 baselines: EOT20 and GOT5.6 at the gauge (a), tau_ab, (b) over the record",
     "ours": "every installed known model, per window, plus HW/LW and the model-internal "
             "read-out (ii)", "why": "spec §3.3.1 (two read-outs per model)",
     "rows": "L14, L15", "sensitivity": "none (descriptive models panel)"},
    {"paper": "truth (b) with a quarter (month) jackknife SE",
     "ours": "the point estimate only (the same two harmonic fits)",
     "why": "a model row needs no interval; the jackknife would add 2 x (groups + 1) fits",
     "rows": "L14, L15", "sensitivity": "none"},
    {"paper": "model cells read from the model grid",
     "ours": "same-or-adjacent cells from the nearest-ocean probes and the nominal resolution "
             "(grid aligned at 0 deg)", "why": "no model grid reader in the harness",
     "rows": "L14, L15 (flag 'degenerate' only)", "sensitivity": "none"},
]
_QC = {"judges": "in-frame judges of inst.judges (the scorer applies the per-window "
                 "inclusion)", "taus_edge": "a minimum on the TAUS_A edge is flagged",
       "control": ("not gated by the positive control: no imagery (spec §6.5 item 7); the "
                   "synthetic site has no judges, so the product is empty there")}


def _card(pid, name, variants, extra_dev=()):
    return {"id": pid, "name": name, "family": "models", "kind": "lag", "tier": "1",
            "fidelity": "-",
            "evidence": ("the truth machinery (truth.e_rows, best_lag, gain_aware, harmonic_fit, "
                         "hw_lw_window) applied to the model's levels; unit tests with a planted "
                         "40-min model lag"),
            "references": list(_REFS_MODELS), "output": "point", "lag_kind": "linear",
            "signature": dict(_SIGNATURE), "scan_window": None, "published_window": None,
            "as_published": ("a global ocean tide model read at the point (pyTMD / eo-tides "
                             "defaults: extrapolate to the nearest ocean cell, no cutoff)"),
            "native_cell": None,
            "common_conditions": ("the instance's instants and window; no imagery, no setting "
                                  "enters (signature: window only)"),
            "deviations": list(_DEVIATIONS) + list(extra_dev), "qc": dict(_QC), "n_min": 0,
            "tuning": [], "control_offset_min": 0.0, "cost": dict(_COST), "variants": variants}


CARDS = {
    "L14": _card("L14", "global ocean tide models at the gauges (primary: FES2022 if installed, "
                        "else EOT20)",
                 {"<model>": "every installed known model, in point[code]['models'] (rows "
                             "'L14@<model>')",
                  "internal": "read-out (ii), model-internal (rows 'L14@internal')"}),
    "L15": _card("L15", "equal-weight multi-model ensemble at the gauges",
                 {"internal": "read-out (ii) of the ensemble (rows 'L15@internal')"},
                 extra_dev=[{"paper": "a ranked or weighted ensemble (eo-tides)",
                             "ours": "equal weights over the available models",
                             "why": "spec §3.3 L15", "rows": "L15", "sensitivity": "none"}]),
}

CODE_DEPS = ("experiments/benchmark/truth.py", "experiments/benchmark/judges.py",
             "experiments/benchmark/levels.py", "experiments/benchmark/settings.py",
             "experiments/benchmark/lags/common.py", "pyintertidal/boundary.py",
             "pyintertidal/tidecheck.py")


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  models
# ─────────────────────────────────────────────────────────────────────────────

def installed(installed_models):
    """``[(label, name)]`` of the known models present in ``installed_models``
    (eo-tides names as ``fetch.tides.installed_models`` lists them, or the
    labels themselves), in :data:`KNOWN_MODELS` order; ``name`` is the first
    present eo-tides name (the label when only the label is listed)."""
    have = {str(m) for m in (installed_models or ())}
    out = []
    for label, names in KNOWN_MODELS:
        hit = [n for n in names if n in have]
        if hit:
            out.append((label, hit[0]))
        elif label in have:
            out.append((label, label))
    return out


def primary_label(labels):
    """FES2022 if installed, else EOT20 [V16]; None when neither is."""
    for p in PRIMARY_ORDER:
        if p in labels:
            return p
    return None


def model_cell(label, lat, lon):
    """The model cell index ``(floor(lat / res), floor(lon / res))`` on a grid
    aligned at 0 deg (declared approximation of the model grid)."""
    res = float(MODEL_RES_DEG.get(label, 0.125))
    return int(np.floor(float(lat) / res)), int(np.floor(float(lon) / res))


def same_or_adjacent(a, b):
    """True when two (row, col) cells are equal or neighbours (8-connected)."""
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def ocean_point(label, name, lat, lon, directory=None):
    """The nearest cell where the model predicts a tide
    (``tidecheck.nearest_ocean_km`` with its default probe): ``{"distance_km",
    "lat", "lon"}`` (lat/lon of the closest wet probe)."""
    from pyintertidal import tidecheck

    r = tidecheck.nearest_ocean_km(float(lat), float(lon), model=name,
                                   directory=directory or config.TIDE_DIR, verbose=False,
                                   **OCEAN_PROBE)
    return {"distance_km": float(r["distance_km"]), "lat": r.get("lat"), "lon": r.get("lon")}


#: nearest-ocean probes of this process: a pure function of the model files
#: and the point, shared by the instances (windows) a worker runs
_OCEAN = {}


def _ocean_cached(inst, label, name, lat, lon):
    cache = _cache(inst).setdefault("models.ocean", {})
    key = (label, name, _tide_dir(inst), round(float(lat), 7), round(float(lon), 7))
    if key not in cache:
        if key in _OCEAN and ocean_point is _OCEAN[key][0]:
            cache[key] = _OCEAN[key][1]
        else:
            try:
                cache[key] = dict(ocean_point(label, name, lat, lon, _tide_dir(inst)),
                                  error=None)
                _OCEAN[key] = (ocean_point, cache[key])
            except Exception as e:              # a diagnostic: recorded, never fatal
                cache[key] = {"distance_km": NAN, "lat": None, "lon": None,
                              "error": f"{type(e).__name__}: {e}"}
    return cache[key]


def model_fingerprint(name, directory=None):
    """sha256 of the model's file list (relative path and size; contents are
    GBs and are not hashed, declared), or None when pyTMD cannot list it."""
    try:
        import pyTMD

        m = pyTMD.io.model(directory=directory or config.TIDE_DIR).elevation(name)
        files = m.model_file
        files = list(files.values()) if isinstance(files, dict) else list(np.atleast_1d(files))
        grid = getattr(m, "grid_file", None)
        if grid:
            files.append(grid)
        rows = []
        for f in files:
            p = os.fspath(f)
            rows.append([os.path.basename(p), int(os.path.getsize(p))])
        return config.sha256_json(sorted(rows))
    except Exception:                           # not listable: the key says so
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  instance accessors
# ─────────────────────────────────────────────────────────────────────────────

def _cache(inst):
    c = getattr(inst, "cache", None)
    if c is None:
        c = {}
        try:
            inst.cache = c
        except AttributeError:
            pass
    return c


def _tide_dir(inst):
    return (getattr(inst, "paths", None) or {}).get("tide_dir") or config.TIDE_DIR


def _stage(inst, name):
    st = getattr(inst, "stage", None)
    if st is None:
        import contextlib
        return contextlib.nullcontext()
    return st(name)


def _progress(inst, n=1):
    p = getattr(inst, "progress", None)
    if p is not None:
        p(n)


def unit_field(unit, name, default=None):
    """``unit[name]`` of a unit dict (plan INTERFACES §5) or attribute of a
    unit object; ``default`` when absent or empty."""
    if unit is None:
        return default
    v = unit.get(name) if isinstance(unit, dict) else getattr(unit, name, None)
    return default if v in (None, "", ()) else v


def unit_products(unit, default):
    """The products of a unit, in order (``default`` when not given)."""
    return tuple(unit_field(unit, "products", default))


def in_frame_judges(inst):
    """The judge records of the instance that stand inside the frame."""
    return [j for j in (getattr(inst, "judges", None) or []) if j.get("in_frame", True)]


def file_sha(path):
    """sha256 of a small input file, None when absent."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def judges_sha(inst):
    td = (getattr(inst, "paths", None) or {}).get("truth_dir")
    return file_sha(os.path.join(td, "judges.json")) if td else None


# ─────────────────────────────────────────────────────────────────────────────
#  the read-out context (shared by L14, L15 and L16)
# ─────────────────────────────────────────────────────────────────────────────

class ReadoutContext:
    """What the point read-outs of one instance share: the window, the
    instants, the h_B rows ``H = levels.stack(TAUS_A)``, the window's 10-min
    h_B, the boundary marks, the query time sets of the gauge and the
    boundary point, and the cached harmonic fit of h_B."""

    def __init__(self, inst, constituents=None):
        self.inst = inst
        self.wid = str(inst.window)
        w = settings.window(self.wid)
        self.start, self.end = pd.Timestamp(w.start), pd.Timestamp(w.end)
        self.t = pd.DatetimeIndex(inst.t).as_unit("ns")
        if len(self.t) > 1 and not (np.diff(self.t.asi8) > 0).all():
            raise ValueError("the instance's instants must be strictly increasing")
        self.T = len(self.t)
        cache = _cache(inst)
        hcache = cache.setdefault("models.H_taus_a", {})
        key = (self.T, int(self.t.asi8[0]) if self.T else 0,
               int(self.t.asi8[-1]) if self.T else 0)
        if key not in hcache:
            hcache[key] = np.asarray(inst.levels.stack(truth.TAUS_A), np.float64)
        self.H = hcache[key]
        if self.H.shape != (len(truth.TAUS_A), self.T):
            raise ValueError(f"levels.stack(TAUS_A) has shape {self.H.shape}, expected "
                             f"{(len(truth.TAUS_A), self.T)}")
        idx10, h10 = inst.series10
        idx10 = pd.DatetimeIndex(idx10).as_unit("ns")
        h10 = np.asarray(h10, np.float64)
        m = np.asarray((idx10 >= self.start) & (idx10 < self.end))
        self.idx_w = idx10[m]
        self.h_w = h10[m]
        self.td_w = truth.days_since_1992(self.idx_w) if len(self.idx_w) else np.zeros(0)
        pad = pd.Timedelta(hours=HW_PAD_H)
        self.hw_grid = pd.date_range(self.start - pad, self.end + pad, freq="10min",
                                     inclusive="left").as_unit("ns")
        self.marks_lp = inst.marks_lp
        cons = constituents if constituents is not None else CONSTITUENTS
        self.cons = tuple(truth.CONS_B if cons is None else cons)
        taus = np.asarray(truth.TAUS_A, np.int64)
        self.lag_ns = self.t.asi8[None, :] - taus[:, None] * NS_PER_MIN       # (n_tau, T)
        self.times_gauge = _union(self.t.asi8, self.idx_w.asi8, self.hw_grid.asi8)
        self.times_boundary = _union(self.lag_ns.ravel(), self.idx_w.asi8)
        self._fits = cache.setdefault("models.fits", {})

    # values on a time set -> the pieces the read-outs need ----------------
    def gauge_parts(self, values):
        """Split a series evaluated on :attr:`times_gauge` into (at the
        instants, on the window's 10-min grid, on the HW/LW grid)."""
        v = np.asarray(values, np.float64)
        if v.shape != self.times_gauge.shape:
            raise ValueError(f"expected {self.times_gauge.shape} values, got {v.shape}")
        return (v[_where(self.times_gauge, self.t.asi8)],
                v[_where(self.times_gauge, self.idx_w.asi8)],
                v[_where(self.times_gauge, self.hw_grid.asi8)])

    def boundary_parts(self, values):
        """Split a series evaluated on :attr:`times_boundary` into
        (B(t_i - tau) as (n_tau, T), on the window's 10-min grid)."""
        v = np.asarray(values, np.float64)
        if v.shape != self.times_boundary.shape:
            raise ValueError(f"expected {self.times_boundary.shape} values, got {v.shape}")
        lag = v[_where(self.times_boundary, self.lag_ns.ravel())].reshape(self.lag_ns.shape)
        return lag, v[_where(self.times_boundary, self.idx_w.asi8)]

    # (b) --------------------------------------------------------------------
    def _fit(self, key, td, y):
        k = (self.wid, self.cons) + tuple(key)
        if k not in self._fits:
            self._fits[k] = truth.harmonic_fit(td, y, self.cons)
        return self._fits[k]

    def phase(self, y, ref=None, ref_key=None):
        """The M2 phase lag (min) of the 10-min series ``y`` relative to h_B
        (``ref`` None) or to ``ref`` on their common finite samples:
        wrap180(phi_y - phi_ref) x 60 / 28.9841042, the tau of ``truth.m2_lag``.
        ``ref_key`` caches the reference's fit (shared by the judges)."""
        y = np.asarray(y, np.float64)
        base = self.h_w if ref is None else np.asarray(ref, np.float64)
        out = {"tau": NAN, "n": 0, "amp_ratio": NAN, "flags": []}
        if y.shape != base.shape or not len(y):
            out["flags"].append("no 10-min samples in the window")
            return out
        m = np.isfinite(y) & np.isfinite(base)
        out["n"] = int(m.sum())
        if not m.any():
            out["flags"].append("no common finite 10-min samples")
            return out
        td = self.td_w[m]
        fy = truth.harmonic_fit(td, y[m], self.cons)
        mkey = hashlib.sha256(np.packbits(m).tobytes()).hexdigest()[:16]
        if ref is None:
            fr = self._fit(("hB", mkey), td, base[m])
        elif ref_key is not None:
            fr = self._fit(("ref", repr(ref_key), mkey), td, base[m])
        else:
            fr = truth.harmonic_fit(td, base[m], self.cons)
        if fy is None or fr is None:
            out["flags"].append("too few samples for the harmonic fit")
            return out
        out["tau"] = float(truth.wrap180(fy["m2"][1] - fr["m2"][1]) * truth.MIN_PER_DEG_M2)
        out["amp_ratio"] = float(fy["m2"][0] / fr["m2"][0]) if fr["m2"][0] else NAN
        return out


def _union(*arrays):
    """Sorted unique DatetimeIndex (ns) of int64 nanosecond arrays."""
    a = np.unique(np.concatenate([np.asarray(x, np.int64).ravel() for x in arrays]))
    return pd.DatetimeIndex(a.view("M8[ns]")).as_unit("ns")


def _where(index, ns):
    """Positions of ``ns`` in the sorted unique ``index`` (all must be there)."""
    pos = np.searchsorted(index.asi8, np.asarray(ns, np.int64))
    return pos


def readouts(ctx, m_t, m10_w, m_hw, mb_lag=None, mb10_w=None, ref_key=None):
    """The point read-outs of one series at one gauge (spec §3.3.1 (i), (ii)).

    Parameters
    ----------
    ctx : ReadoutContext
    m_t, m10_w, m_hw : arrays
        The series at the instance's instants, on the window's 10-min grid
        and on the HW/LW grid (:meth:`ReadoutContext.gauge_parts`).
    mb_lag, mb10_w : arrays, optional
        The same model at the boundary point at t_i - tau (n_tau, T) and on
        the window's 10-min grid (:meth:`ReadoutContext.boundary_parts`):
        the model-internal read-out (ii).

    Returns
    -------
    dict
        ``a``, ``ab``, ``b``, ``hwlw``, ``internal_a``, ``internal_b`` (min,
        NaN undefined), the details (intervals, E0, gain, n) and ``flags``.
    """
    out = {k: NAN for k in common.POINT_VALUE_KEYS}
    flags = []
    m_t = np.asarray(m_t, np.float64)
    E = truth.e_rows(ctx.H, m_t)
    a = truth.best_lag(truth.TAUS_A, E)
    k0 = int(np.flatnonzero(truth.TAUS_A == 0)[0])
    out["a"] = a["tau"]
    out.update(a_lo=a["lo"], a_hi=a["hi"], E0_m=float(E[k0]) if np.isfinite(E[k0]) else NAN,
               Emin_m=a["Emin_m"], n_instants=int(np.isfinite(m_t).sum()))
    if a["edge"]:
        flags.append("a at the edge of tau in [-60, 240]")
    Eab, B = truth.gain_aware(ctx.H, m_t)
    ab = truth.best_lag(truth.TAUS_A, Eab)
    kb = int(np.nanargmin(Eab)) if np.isfinite(Eab).any() else None
    out["ab"] = ab["tau"]
    out["gain_ab"] = float(B[kb]) if kb is not None else NAN
    if ab["edge"]:
        flags.append("ab at the edge of tau in [-60, 240]")
    b = ctx.phase(m10_w)
    out["b"] = b["tau"]
    out["amp_ratio"] = b["amp_ratio"]
    out["n_10min"] = b["n"]
    flags += [f"b: {f}" for f in b["flags"]]
    try:
        hl = truth.hw_lw_window(pd.Series(np.asarray(m_hw, np.float64), index=ctx.hw_grid),
                                ctx.marks_lp, ctx.start, ctx.end)
        out["hwlw"] = float(hl.get("mean_min", NAN))
        out["hw_min"] = float(hl.get("hw", {}).get("median_min", NAN))
        out["lw_min"] = float(hl.get("lw", {}).get("median_min", NAN))
    except (AttributeError, KeyError, ValueError) as e:
        flags.append(f"hwlw: {type(e).__name__}: {e}")
    if mb_lag is not None:
        Ei = truth.e_rows(np.asarray(mb_lag, np.float64), m_t)
        ia = truth.best_lag(truth.TAUS_A, Ei)
        out["internal_a"] = ia["tau"]
        if ia["edge"]:
            flags.append("internal_a at the edge of tau in [-60, 240]")
    if mb10_w is not None:
        ib = ctx.phase(m10_w, ref=mb10_w, ref_key=ref_key)
        out["internal_b"] = ib["tau"]
        flags += [f"internal_b: {f}" for f in ib["flags"]]
    out["flags"] = flags
    return out


#: descriptive details a point entry carries beside the six values
_DETAIL_KEYS = ("a_lo", "a_hi", "E0_m", "Emin_m", "gain_ab", "amp_ratio", "hw_min", "lw_min",
                "n_instants", "n_10min")


def _nan_entry(reason, flags=()):
    e = {k: NAN for k in common.POINT_VALUE_KEYS}
    e.update(dist_km=NAN, flags=list(flags), reason=reason)
    return e


# ─────────────────────────────────────────────────────────────────────────────
#  L14 / L15
# ─────────────────────────────────────────────────────────────────────────────

def _levels(prov, times):
    """``prov.levels`` on sorted unique times, checked for length."""
    v = np.asarray(prov.levels(times), np.float64)
    if v.shape != (len(times),):
        raise ValueError(f"the provider returned {v.shape} levels for {len(times)} instants")
    return v


def _members(inst):
    """The boundary point's members ``[{"code", "lat", "lon", "kind"}]``."""
    out = []
    for m in getattr(inst, "boundary_members", None) or []:
        if m.get("lat") is None or m.get("lon") is None:
            continue
        out.append({"code": str(m.get("code")), "lat": float(m["lat"]), "lon": float(m["lon"]),
                    "kind": m.get("kind")})
    return out


def _boundary_model(ctx, label, name, members):
    """The model at the boundary point: the equal-weight mean of the model
    at every member, on ``ctx.times_boundary``. Returns (values or None,
    reason)."""
    if not members:
        return None, "no boundary member"
    acc = None
    for mb in members:
        try:
            v = _levels(truth.model_provider(name, mb["lat"], mb["lon"]), ctx.times_boundary)
        except Exception as e:                  # recorded, never substituted
            return None, f"{mb['code']}: {type(e).__name__}: {e}"
        acc = v if acc is None else acc + v
    return acc / len(members), ""


def _degenerate(inst, label, name, judge, members):
    """``(flag or None, detail)``: the gauge and a boundary member in the same
    or adjacent model cells (nearest-ocean probes), declared."""
    g = _ocean_cached(inst, label, name, judge["lat"], judge["lon"])
    if g.get("lat") is None:
        return None, {"gauge": g}
    cg = model_cell(label, g["lat"], g["lon"])
    det = {"gauge_cell": list(cg), "members": {}}
    hit = []
    for mb in members:
        o = _ocean_cached(inst, label, name, mb["lat"], mb["lon"])
        if o.get("lat") is None:
            det["members"][mb["code"]] = {"error": o.get("error")}
            continue
        cm = model_cell(label, o["lat"], o["lon"])
        det["members"][mb["code"]] = {"cell": list(cm)}
        if same_or_adjacent(cg, cm):
            hit.append(mb["code"])
    if hit:
        return f"degenerate (same or adjacent cells: {', '.join(hit)})", det
    if any("error" in v for v in det["members"].values()):
        return None, det
    return False, det


def run_l14(inst, unit, opts=None):
    """Unit L14: products L14 (the primary model, every model in ``models``)
    and L15 (the equal-weight ensemble); see the module docstring."""
    products = unit_products(unit, ("L14", "L15"))
    judges_in = in_frame_judges(inst)
    base_diag = {"window": str(inst.window), "n_judges_in_frame": len(judges_in)}
    if not judges_in:
        res = {p: common.lag_result(p, "empty", kind="point", point={},
                                    reason="no in-frame judge", diag=dict(base_diag))
               for p in ("L14", "L15")}
        return [res[p] for p in products]
    mods = installed(getattr(inst, "installed_models", ()))
    labels = [lb for lb, _ in mods]
    if not mods:
        raise NotComputable("models", "no known global tide model is installed",
                            {"missing_input": "tide models (fetch tides)",
                             "known": [lb for lb, _ in KNOWN_MODELS],
                             "installed": list(getattr(inst, "installed_models", ()) or ())})
    primary = primary_label(labels)
    members = _members(inst)
    ctx = ReadoutContext(inst)
    with _stage(inst, "boundary"):
        mb_vals, mb_reason = {}, {}
        for label, name in mods:
            v, why = _boundary_model(ctx, label, name, members)
            if v is not None:
                mb_vals[label] = v
            mb_reason[label] = why
    avail_any = {}
    point14, point15 = {}, {}
    ens_members_all = set()
    for rec in judges_in:
        code = str(rec["code"])
        per, gauge_vals = {}, {}
        for label, name in mods:
            with _stage(inst, f"model.{label}"):
                entry = _model_entry(inst, ctx, rec, label, name, members, mb_vals, mb_reason,
                                     gauge_vals)
            per[label] = entry
            avail_any[label] = avail_any.get(label, False) or entry["available"]
            _progress(inst)
        top = per.get(primary) if primary else None
        if top is not None and top["available"]:
            e14 = {k: top[k] for k in common.POINT_VALUE_KEYS + _DETAIL_KEYS if k in top}
            e14.update(dist_km=top["dist_km"], flags=list(top["flags"]))
        else:
            why = ("the primary model is not installed" if primary is None
                   else f"primary model {primary} unavailable: {top.get('reason')}")
            e14 = _nan_entry(why, flags=[why])
        e14["model"] = primary
        e14["models"] = per
        point14[code] = e14
        with _stage(inst, "ensemble"):
            point15[code] = _ensemble_entry(ctx, gauge_vals, mb_vals, per)
        ens_members_all.update(point15[code].get("members", []))
    diag = dict(base_diag, primary=primary, installed=labels,
                names={lb: nm for lb, nm in mods},
                available={lb: bool(avail_any.get(lb)) for lb in labels},
                boundary_members=[m["code"] for m in members],
                boundary_unavailable={lb: r for lb, r in mb_reason.items() if r},
                constituents=list(ctx.cons), n_instants=ctx.T, n_10min=int(len(ctx.idx_w)))
    if primary is None:
        r14 = common.lag_result("L14", "not_computable", kind="point", stage="primary",
                                reason="neither FES2022 nor EOT20 is installed [V16]",
                                diag=dict(diag, missing_input="FES2022 or EOT20", point=point14))
    elif not any(np.isfinite(e["a"]) for e in point14.values()):
        r14 = common.lag_result(
            "L14", "not_computable", kind="point", stage="primary",
            reason=f"the primary model {primary} could not be read at any judge",
            diag=dict(diag, missing_input=f"tide model {primary}", point=point14))
    else:
        r14 = common.lag_result("L14", "ok", kind="point", lag_kind="linear", point=point14,
                                edge_share=_edge_share(point14), diag=diag)
    usable = sorted(lb for lb in labels if avail_any.get(lb))
    if len(usable) < ENSEMBLE_MIN_MODELS:
        r15 = common.lag_result("L15", "not_computable", kind="point", stage="ensemble",
                                reason=(f"the ensemble needs >= {ENSEMBLE_MIN_MODELS} available "
                                        f"models; available: {usable or 'none'}"),
                                diag=dict(diag, missing_input="a second tide model"))
    else:
        r15 = common.lag_result("L15", "ok", kind="point", lag_kind="linear", point=point15,
                                edge_share=_edge_share(point15),
                                diag=dict(diag, ensemble_members=sorted(ens_members_all)))
    res = {"L14": r14, "L15": r15}
    return [res[p] for p in products]


def _edge_share(point):
    vals = [("a at the edge of tau in [-60, 240]" in e.get("flags", []))
            for e in point.values() if np.isfinite(e.get("a", NAN))]
    return float(np.mean(vals)) if vals else NAN


def _model_entry(inst, ctx, rec, label, name, members, mb_vals, mb_reason, gauge_vals):
    """One model's read-outs at one judge (``models[label]`` entry)."""
    entry = {"name": name, "available": False, "reason": None}
    try:
        vals = _levels(truth.model_provider(name, rec["lat"], rec["lon"]), ctx.times_gauge)
    except Exception as e:                      # an unloadable model is recorded
        entry.update(_nan_entry(f"{type(e).__name__}: {e}"))
        entry["available"] = False
        entry["reason"] = f"{type(e).__name__}: {e}"
        return entry
    if not np.isfinite(vals).any():
        entry.update(_nan_entry("the model gives no finite level at the gauge"))
        entry["reason"] = "the model gives no finite level at the gauge"
        return entry
    gauge_vals[label] = vals
    m_t, m10, m_hw = ctx.gauge_parts(vals)
    mb_lag = mb10 = None
    flags = []
    if label in mb_vals:
        mb_lag, mb10 = ctx.boundary_parts(mb_vals[label])
    else:
        flags.append(f"no boundary point: {mb_reason.get(label)}")
    r = readouts(ctx, m_t, m10, m_hw, mb_lag, mb10, ref_key=("boundary", label))
    oc = _ocean_cached(inst, label, name, rec["lat"], rec["lon"])
    if members:
        deg, det = _degenerate(inst, label, name, rec, members)
    else:
        deg, det = None, {"members": {}}
    if deg:
        flags.append("degenerate")
    elif deg is None and members:
        flags.append("degeneracy unknown (no nearest-ocean probe)")
    res_flag = resolution_flag(label, oc.get("distance_km", NAN), oc.get("error"))
    if res_flag:
        flags.append(res_flag)
    entry.update(r)
    entry["flags"] = r["flags"] + flags
    entry.update(available=True, reason=None, dist_km=float(oc.get("distance_km", NAN)),
                 ocean_error=oc.get("error"), cells=det)
    return entry


#: km per degree of latitude (the model grid spacing in km, declared)
KM_PER_DEG = 111.32


def resolution_flag(label, dist_km, error=None):
    """'unresolved (...)' when the gauge's nearest wet model cell is farther
    than the model's grid spacing (or no wet cell is within the probe), else
    None; 'resolution unknown' when the probe failed. A flag only: the
    estimate is never changed (spec §3.3.1: the 1/8 deg models are lower
    bounds in the inner estuaries)."""
    spacing = float(MODEL_RES_DEG.get(label, 0.125)) * KM_PER_DEG
    d = float(dist_km) if dist_km is not None else NAN
    if error:
        return "resolution unknown (no nearest-ocean probe)"
    if np.isnan(d):
        return None
    if not np.isfinite(d):
        return (f"unresolved (no wet {label} cell within the {OCEAN_PROBE['search_deg']} deg "
                "probe)")
    if d > spacing:
        return f"unresolved (nearest wet {label} cell {d:.1f} km > grid spacing {spacing:.1f} km)"
    return None


def _ensemble_entry(ctx, gauge_vals, mb_vals, per):
    """L15 at one judge: the equal-weight mean of the models available there
    (the members' 'unresolved' and 'degenerate' flags are carried)."""
    mem = sorted(gauge_vals)
    if len(mem) < ENSEMBLE_MIN_MODELS:
        e = _nan_entry(f"fewer than {ENSEMBLE_MIN_MODELS} models available at this gauge",
                       flags=["fewer than 2 models"])
        e.update(model=None, models=None, members=mem)
        return e
    m_all = np.mean([gauge_vals[lb] for lb in mem], axis=0)
    m_t, m10, m_hw = ctx.gauge_parts(m_all)
    flags = []
    mb_lag = mb10 = None
    if all(lb in mb_vals for lb in mem):
        mb_lag, mb10 = ctx.boundary_parts(np.mean([mb_vals[lb] for lb in mem], axis=0))
    else:
        flags.append("no boundary point for every member")
    r = readouts(ctx, m_t, m10, m_hw, mb_lag, mb10, ref_key=("boundary", "ens:" + "+".join(mem)))
    deg = [lb for lb in mem if "degenerate" in per[lb].get("flags", [])]
    if deg:
        flags.append("degenerate")
    unres = [lb for lb in mem if any(str(f).startswith("unresolved")
                                     for f in per[lb].get("flags", []))]
    if unres:
        flags.append(f"members unresolved: {', '.join(unres)}")
    unk = [lb for lb in mem if any(str(f).startswith(("degeneracy unknown", "resolution unknown"))
                                   for f in per[lb].get("flags", []))]
    if unk:
        flags.append(f"members with unknown degeneracy or resolution: {', '.join(unk)}")
    e = dict(r)
    e["flags"] = r["flags"] + flags
    dists = [per[lb].get("dist_km", NAN) for lb in mem]
    e.update(dist_km=float(np.nanmax(dists)) if np.isfinite(dists).any() else NAN,
             model="ensemble(" + "+".join(mem) + ")", models=None, members=mem,
             degenerate_members=deg)
    return e


# ─────────────────────────────────────────────────────────────────────────────
#  keys
# ─────────────────────────────────────────────────────────────────────────────

def external_inputs(inst, unit=None):
    """The fingerprints the unit reads besides the package: each installed
    known model's file list (:func:`model_fingerprint`) and the judge table."""
    out = {}
    d = _tide_dir(inst)
    for label, name in installed(getattr(inst, "installed_models", ())):
        out[f"tide_model:{label}"] = model_fingerprint(name, d)
    out["judges.json"] = judges_sha(inst)
    out["boundary_members"] = config.sha256_json(_members(inst))
    return out


def describe():
    """JSON-ready summary of the module (for logs)."""
    return json.loads(json.dumps({"CONFIG": CONFIG, "products": sorted(CARDS)}))
