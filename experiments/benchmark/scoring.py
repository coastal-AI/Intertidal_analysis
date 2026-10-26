# -*- coding: utf-8 -*-
"""Validation v2 — lag scoring: read-outs against the in-estuary truths, level
metrics, truth-side bootstrap intervals, aggregates, family L-P1,
reproducibility, efficiency, the positive-control verdicts and gate
G-lagtruth (spec §6.3-§6.6, §9.7-§9.9, §10.1-§10.2; the scoring spec of the
V2 plan).

Question
--------
How close is each lag method's read-out to each judge's truths, what level
misfit does its lag leave at the gauge, how uncertain is that because of the
truth, and is MAREA's lag closer than the pre-registered competitors'?

Rules (every number in :data:`SCORING_V2`; FROZEN values are read, never
copied by hand)
--------------------------------------------------------------------------
* Cells. A (site, window) is scored only when every year of the window is in
  the package's years (the window-year guard: otherwise every row reads
  "NOT AVAILABLE: store years [...]"), and a judge only where the truth
  document includes it in that window (``stations[*].windows[w].inclusion``;
  otherwise "judge excluded (<rules>)"). Rows always cover the lag plan's
  full set (``plans/lag.csv``): NOT COMPUTABLE (<stage>), not applicable
  (<plan note>), NOT AVAILABLE (<reason>) and "implementation not
  validated" are written, never left blank.
* Read-out (:mod:`readout`, §6.3): the ladder for field and band products on
  the instance's domain; point products read ``point[code][truth]``;
  model-internal values become descriptive rows ``<product>@internal``,
  secondary models ``L14@<model>``, relative-orbit groups ``L01v@<group>``;
  L20 (reference, never ranked) is the scorer's leave-one-out linear
  interpolation in s of the other included locations' truth, anchored at
  tau = 0 at s = 0 and held constant beyond the last location.
* Metrics per (site, window, judge, product, truth) (:class:`LagScorer`):
  e = tau_m - tau_T (circular for phase), |e|, err_int (distance to the
  truth interval), E_g(tau) = the median-centred RMS of
  h_B(t_i - tau) - g_i at the window's S2 instants with a gauge value within
  6 min (``truth.e_rows``; real tau linear between the level rows floor and
  ceil, ``LevelTable.at`` making a live call beyond the table), E0, E(tau_m),
  Emin = E(tau_a), dE, E_all over every 10-min sample (equal to
  ``truth.tau_all_window`` at integer tau) and dE_all, LLE (h_B at tau_m
  against tau_a, no gauge noise), explained = 1 - (Etau/E0)^2, AGF = the
  share of the achievable reduction where E0 - Emin >= 2 cm (else "lag not
  material"), and the interval score IS_0.05 and coverage of the nominal
  95 % intervals (L01 boot30, L08 refits).
* Truth-side month-block bootstrap (B = 1000, :mod:`inference`): one weight
  matrix per (site, judge, window), seeded
  ``[*truth.window_boot_seed(site, code, w), 1]`` and shared by every product
  (paired); per draw tau_a* = argmin of the weighted E over TAUS_A, and
  E0*, Etau*, Emin*, dE*, AGF*, LLE* (against tau_a*) and
  e_a* = tau_m - tau_a*; 2.5/97.5 percentile CIs "truth uncertainty only".
* Locations: the Terneuzen pair (config.SAME_LOCATION) is one location: the
  mean of its gauges' e and metrics (and of their bootstrap draws).
* Aggregates per site, window, product, truth and judge set (headline,
  without_mouth, development_free; the read-out sensitivity step1_only),
  pooled = the mean of the site values (villaviciosa never pooled), profile
  metrics where a site has >= 4 locations (OLS slope of tau_m on tau_T,
  Lin's CCC, shape error RMS(e - mean e), decreases of tau_m along s_g).
* Family L-P1 (:mod:`inference`): confirmatory only at W3 with LOCK.json
  phase A and on rows that passed the printing rule; otherwise the same
  quantities descriptively on each judged site's available window,
  labelled "descriptive, mixed windows, not the pre-registered L-P1", with
  no verdict. The pooled location test is printed as pseudo-replicated.
* Printing rule (§9.7): an imagery lag row is validated when its
  ``control/<product>.json`` verdict is PASS or "control mismatch" (the
  latter flagged) and the gates it needs passed at the site
  (``gates/<site>/<name>.json``); otherwise "implementation not validated"
  (the CSV keeps the numbers with ``validated`` = False; the table prints
  none). Model, physics and reference rows have no imagery to control.
* Reproducibility (§6.4, truth-free): band lags of every band/field product
  across the calendar-year windows (W1 stands for Y2025) matched by their
  3-km band of the centre, and L01v across relative orbits: the median
  |tau_g1 - tau_g2| per band over group pairs and ICC(2,1) over bands; at a
  site without judges (villaviciosa) a method x band table and the
  cross-method spread per band, labelled "lag coherence only".
* Efficiency: the median over instances of wall, CPU and peak RSS from
  run.json (production runs, labelled "contended"), the shares of NOT
  COMPUTABLE units and of zero fallbacks.
* Positive control (:func:`control_verdicts`, §10.2): per product the band
  read-out minus the planted band lag minus the card's expected offset
  (circular for phase); PASS iff the median |error| over bands <= 10 min
  (FROZEN positive_control.tolerances.lag_max_min) in both scenarios
  (:func:`control_overall`), else "control mismatch"; "FAIL (fault)" only by
  a manual flag.
* G-lagtruth (:func:`gate_lagtruth`, §10.1): a planted 75-min lag (gauge =
  h_B(t - 75 min) + 2 cm noise at the record instants and on the 10-min grid)
  is recovered by ``truth.lag_truths_window`` as tau_a and tau_b within
  75 +- 1 min, and the nearest-cell truth lookup equals ``v1.sample_truth``.

Outputs (``<root>/scores/``, through :func:`lagtable.write`): lag_points.csv,
lag_summary.csv, lag_family.csv, lag_repro.csv, lag_cost.csv and
lag_table.md; with ``pre_lock`` every file carries the banner "PRE-LOCK,
descriptive; not the pre-registered analysis".
"""
from __future__ import annotations

import math
import os
import types
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from experiments.benchmark import config, inference, judges, lagtable, product, settings, truth
from experiments.benchmark import readout as rd
from experiments.benchmark.lags import common as lc

V2_ROOT = "results/validation_v2"
LAG_PRODUCT_FORMAT = "v2-lag-product-1"
INSTANCE_FORMAT = "v2-lag-instance-1"
GATE_FORMAT = "v2-gate-1"

_PC = config.FROZEN["positive_control"]
_IMAGERY = ["L01", "L01v", "L02", "L03", "L04w", "L04i", "L07", "L08", "L09", "L10", "L11", "L12"]
#: rows that read no imagery (models and physics; the reference L20 aside):
#: no catalogue-to-sensing offset enters them (their L-P1 rows carry the note)
_NON_IMAGERY = ["L14", "L15", "L16", "L17", "L18"]
#: the lag ESTIMATES without imagery that the per-site offset line compares
#: the imagery rows with: the primary global model, the regional model and
#: the celerity (zero lag is a null baseline, the ensemble carries the
#: secondary models' resolution errors)
_NON_IMAGERY_ESTIMATES = ["L14", "L16", "L18"]
#: pre-lock at 10 m these windows stay blinded unless the user unblinds them
#: (the confirmatory window of L-P1, spec §9.9)
PRELOCK_BLIND_WINDOWS = (str(inference.INFERENCE_V2["lp1"]["window"]),)
PRELOCK_LEDGER = "prelock_ledger.json"
PRELOCK_LEDGER_FORMAT = "v2-prelock-ledger-1"
#: what the catalogue (datatake-start) instants do to an imagery lag (spec
#: header 2026-10-14: no correction, attribute first)
SENSING_OFFSET_NOTE = (
    "Scene instants are the catalogue (datatake-start) times (user decision 2026-10-14). "
    "The tile is sensed 0-10.6 min later (median 6.5 min at the Ems), so every imagery lag "
    "(L01-L12, fitted to h_B at the catalogue instant) is expected to read about that offset "
    "LOW against truths (a) and (b), which compare the gauge with h_B at the same instants; "
    "the model, physics and reference rows (L14-L20) carry no such offset. Nothing is "
    "corrected (attribute first); the per-site line below compares the two groups.")
LP1_OFFSET_NOTE = ("MAREA (imagery) is expected to read the catalogue-to-sensing offset low; "
                   "this competitor reads no imagery")
#: §6.4 / §6.5 items not computed in milestone V2 (printed in lag_table.md)
DEFERRED_ITEMS = (
    "split-half lag reproducibility (disjoint halves of the record): the 'half' rows run in "
    "Stage B (design base) for E0, whose L01 they carry; the other imagery lag methods have "
    "no half rows yet",
    "the lag per season and the solar-zenith > 70 deg sensitivity (spec §6.4, geometry "
    "diagnostics)",
    "the location-year analysis with its cluster bootstrap by location within site (spec "
    "§6.5 item 3; the per-year truths exist)",
)

#: every constant of the lag scoring (lock.py records its sha256)
SCORING_V2 = {
    "readout": {
        "min_px": rd.MIN_PX,
        "ladder": ["median of the finite field over nbhd_g ∩ keep when >= min_px defined",
                   "median over the instance's modal band k_g^inst "
                   "(judges.station_geometry on the instance's domain)",
                   "0, flagged fallback0"],
        "phase": "wrap_phase about the truth-free centre, circ_median (never about a truth)",
        "wrap_centre_min": lc.WRAP_CENTRE_MIN,
        "edge_share_flag": rd.EDGE_SHARE_FLAG,
        "edge_tol_min": lc.EDGE_TOL_MIN,
        "edge": "band methods: share of read-out pixels in edge bands; linear pixel methods: "
                "share on the scan bounds -45 / 180 min",
        "band_interp": "linear in s at the instance's s_g between band medians at the band "
                       "centres, clamped at the end bands",
        "point": "point[code][truth] (the method's estimand matched to the truth of that name)",
        "internal": "descriptive '<product>@internal' rows: internal_a <-> a, internal_b <-> b",
        "secondary_models": "descriptive 'L14@<model>' rows",
        "groups": "descriptive '<product>@<group>' rows (L01v per relative orbit)",
        "l20": "leave-one-out linear interpolation in s of the other included locations' "
               "truth, anchor tau = 0 at s = 0, constant beyond the last location",
    },
    "truths": ["a", "ab", "b", "hwlw"],
    "co_primary": ["a", "ab", "b"],
    "truth_keys": {"a": "a.tau, [a.lo, a.hi]", "ab": "ab.tau, [ab.lo, ab.hi]",
                   "b": "b.tau, [b.lo, b.hi] (jackknife +-1.96 SE)",
                   "hwlw": "hw_lw.mean_min, no interval"},
    "gauge": {"tolerance_min": float(config.FROZEN["truth_lag"]["a"]["tolerance_min"]),
              "centring": "judges.centred with the truth document's conventions",
              "ten_min": "judges.centred_10min on the window's 10-min grid"},
    "E": "sqrt(mean_i (d_i - median d)^2), d_i = h_B(t_i - tau) - g_i (truth.e_rows); real "
         "tau: linear between the level rows floor(tau) and ceil(tau)",
    "E_all": "the same over the window's 10-min samples, h_B from the 1-min series at grid "
             "- tau (linear in minutes)",
    "dE_all": "E_all(tau_m) - min over integer tau in TAUS_A of E_all",
    "LLE": "sqrt(mean_i [h_B(t_i - tau_m) - h_B(t_i - tau_a)]^2) over the instants with a "
           "finite gauge value",
    "material_m": 0.02,
    "agf": "(E0 - Etau) / (E0 - Emin) where E0 - Emin >= material_m, else NaN 'lag not material'",
    "hit_min": 10.0,
    "interval": {"alpha": 0.05,
                 "score": "(hi - lo) + (2/alpha)(lo - tau)1[tau < lo] + (2/alpha)(tau - hi)"
                          "1[tau > hi]",
                 "band_level": "[lo_k, hi_k] of the read-out band",
                 "pixel_level": "medians of lo and hi over the read-out pixels",
                 "phase": "the truth brought to the branch nearest the interval centre"},
    "units": {"tau": "min", "E": "cm in the tables (m internally)"},
    "locations": "config.SAME_LOCATION gauges are one location: the mean of their e, metrics "
                 "and bootstrap draws",
    "judge_sets": ["headline", "without_mouth", "development_free"],
    "sensitivity_sets": ["step1_only"],
    "profile_min_locations": 4,
    "pooled": "the mean of the site values; each site once; sites without judges never pooled",
    "window_guard": "a (site, window) is scored only when every year of the window is in "
                    "pkg.years",
    "repro": {"band_km": float(config.FROZEN["domain"]["bands"]["length_km"]),
              "years": ["Y2023", "Y2024", "Y2025"], "w1_is": "Y2025",
              "match": "bands of different windows paired by the 3-km bin of their centre",
              "statistic": "median |tau_g1 - tau_g2| per band over group pairs; ICC(2,1) "
                           "over bands"},
    "printing": {
        "control_ok": ["PASS", "control mismatch"],
        "imagery": _IMAGERY,
        "required_gates": {p: (["record", "keep", "marea"] if p in ("L01", "L01v")
                               else ["record", "keep"]) for p in _IMAGERY},
        "fidelity_pending": {"L09": "G-hd26 (CoastSat NZ vs TICON-4; outside V2)",
                             "L08": "GM20 on NW Andros (PANGAEA; outside V2)"},
        "rule": "an imagery row prints when its positive control is PASS or 'control "
                "mismatch' and its gates passed at the site; model, physics and reference "
                "rows have no imagery control",
    },
    "control": {"tol_min": float(_PC["tolerances"]["lag_max_min"]),
                "scenarios": ["profile", "zero"],
                "error": "band read-out - planted band lag - expected offset (circular for "
                         "phase)",
                "verdicts": ["PASS", "control mismatch", "FAIL (fault)"]},
    "gate_lagtruth": {"planted_min": 75, "tol_min": 1.0, "noise_m": 0.02, "provider": "RWS",
                      "sampling_min": 10.0, "windows": ["W3", "Y2025", "Y2024", "Y2023"],
                      "seed": "[20261011, SITE_ID, 75]"},
    "banner_pre_lock": "PRE-LOCK, descriptive; not the pre-registered analysis",
    "prelock": {"blind_windows": list(PRELOCK_BLIND_WINDOWS),
                "rule": "pre-lock at 10 m on real products: the confirmatory window stays "
                        "BLINDED (no truth read, no product scored) unless the user unblinds "
                        "it; every pre-lock score is appended to scores/prelock_ledger.json, "
                        "which lock-A copies into the forking-paths disclosure",
                "partial": "a call with sites or windows writes scores/partial_<sites>__"
                           "<windows>/, never the full tables"},
    "staleness": {"control": "the scored product's code_files_key equals the positive "
                             "control's (control/<product>.json)",
                  "gates": "the gate document's code_sha256 equals the current sources "
                           "(gates.gate_code_shas)",
                  "truth": "a judge whose E_g differs from the truth's E_a curve by more "
                           "than 1e-9 m, or whose curve is missing, reads NOT AVAILABLE "
                           "(truth stale)"},
    "interval_at_readout": {
        "L01": "2.5/97.5 percentiles of the 30 bootstrap draws' read-outs: each draw's band "
               "lags through the causality clip and the applied smoothed lag, read out at "
               "the point read-out's positions",
        "L08": "read-out +- 1.96 x IQR(refit read-out - read-out) / 1.349 x sqrt(m/(n - m)), "
               "the 20 refit fields read out at the point read-out's positions (circular)"},
    "point_phase_keys": ["b"],
    "sensing_offset_note": SENSING_OFFSET_NOTE,
    "non_imagery": list(_NON_IMAGERY),
    "offset_line_comparators": ["L14", "L16", "L18"],
    "deferred": list(DEFERRED_ITEMS),
}
del _PC

TRUTHS = tuple(SCORING_V2["truths"])
CO_PRIMARY = tuple(SCORING_V2["co_primary"])
MATERIAL_M = float(SCORING_V2["material_m"])
HIT_MIN = float(SCORING_V2["hit_min"])
IS_ALPHA = float(SCORING_V2["interval"]["alpha"])
TAUS_A = truth.TAUS_A
K0 = int(np.flatnonzero(TAUS_A == 0)[0])
NAN = float("nan")
#: product arrays of a lag product.npz (plan INTERFACES §6)
ARRAY_KEYS = ("field", "band_tau", "band_n_px", "band_edge", "band_lo", "band_hi", "iv_lo",
              "iv_hi", "z", "sigma", "clear_obs", "tau_px")


def scoring_v2_sha():
    """sha256 of :data:`SCORING_V2` (what lock-A records)."""
    return config.sha256_json(SCORING_V2)


def log(msg):
    print(msg, flush=True)


def _f(x):
    """A float (NaN for None or non-numbers)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v


def _cm(x):
    return _f(x) * 100.0


def _nanmean(A, axis=0):
    """``np.nanmean`` without the all-NaN warning (NaN where none is finite)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(A, axis=axis)


def _nanmedian(A, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(A, axis=axis)


# ─────────────────────────────────────────────────────────────────────────────
#  the per-window scorer
# ─────────────────────────────────────────────────────────────────────────────

class LagScorer:
    """Level-domain metrics of one site and window against its judges.

    Parameters
    ----------
    pkg : V2Package (or a stand-in)
        ``t``, ``levels`` (``at``/``stack``: exact h_B(t - tau) at the
        record instants), ``series1`` (t_min, h), ``series10`` (DatetimeIndex,
        h), ``cfg.site``.
    truth_doc : dict
        ``truth/<site>/truth_lag_v2.json``.
    judges_table : list of dict
        Judge-table records (provider, sampling_min, file, location, ...).
    window : str
        The window id.
    B : int
        Bootstrap draws (default INFERENCE_V2 B_descriptive, 1000).
    series_by_code : dict, optional
        Pre-read gauge series (as delivered); default ``judges.read_series``.
    conventions : mapping, optional
        Time-stamp conventions (default the truth document's).

    The weight matrix of a judge (:meth:`boot`) is drawn once and shared by
    every product scored through this object (paired).
    """

    def __init__(self, pkg, truth_doc, judges_table, window, *, B=None, series_by_code=None,
                 conventions=None, site=None):
        self.pkg = pkg
        self.doc = truth_doc or {}
        self.site = (site or getattr(getattr(pkg, "cfg", None), "site", None)
                     or self.doc.get("site"))
        if self.site not in config.SITE_ID:
            raise ValueError(f"unknown site {self.site!r}")
        self.window = settings.window(window).id
        self.B = int(inference.INFERENCE_V2["boot"]["B_descriptive"] if B is None else B)
        self.t_all = truth.record_t(pkg)
        self.pos = truth.window_positions(self.t_all, self.window)
        self.t = self.t_all[self.pos]
        conv = self.doc.get("conventions") if conventions is None else conventions
        self.conventions = conv or None
        self.stations = {st["code"]: st for st in self.doc.get("stations", [])}
        self.table = {r["code"]: r for r in (judges_table or [])}
        # the caller's dict is filled with every series read (one read per
        # site across its windows)
        self._series = series_by_code if series_by_code is not None else {}
        self._H = None
        self._rows = {}
        self._j = {}
        self._lm = {}
        self.warnings = []

    # -- levels ---------------------------------------------------------------
    @property
    def H(self):
        """(len(TAUS_A), T_w) exact rows h_B(t_i - tau) of the window (the
        truth's ``pkg.levels.stack(TAUS_A)[:, positions]``)."""
        if self._H is None:
            self._H = np.asarray(self.pkg.levels.stack(TAUS_A), np.float64)[:, self.pos]
        return self._H

    def row(self, tau):
        """h_B(t_i - tau) at the window instants for an integer tau (a table
        row; a live provider call beyond the table)."""
        tau = int(tau)
        k = tau - int(TAUS_A[0])
        if 0 <= k < len(TAUS_A):
            return self.H[k]
        if tau not in self._rows:
            self._rows[tau] = np.asarray(self.pkg.levels.at(tau), np.float64)[self.pos]
        return self._rows[tau]

    def h_at(self, tau):
        """h_B(t_i - tau) for a real tau: linear between the rows floor(tau)
        and ceil(tau)."""
        tau = float(tau)
        if not np.isfinite(tau):
            return np.full(len(self.pos), np.nan)
        lo = math.floor(tau)
        f = tau - lo
        if f == 0.0:
            return self.row(lo)
        return (1.0 - f) * self.row(lo) + f * self.row(lo + 1)

    # -- judges ---------------------------------------------------------------
    def entry(self, code):
        """The truth document's window entry of a judge (``{}`` if none)."""
        return (self.stations.get(code) or {}).get("windows", {}).get(self.window) or {}

    def judge(self, code):
        """The gauge at the window's instants and on its 10-min grid."""
        j = self._j.get(code)
        if j is not None:
            return j
        rec = dict(self.stations.get(code) or {})
        rec.update(self.table.get(code) or {})
        if "provider" not in rec or "sampling_min" not in rec:
            raise KeyError(f"judge {code}: the judge table record needs provider and "
                           "sampling_min")
        series = self._series.get(code)
        if series is None:
            series = judges.read_series(rec)
            self._series[code] = series
        sc = judges.centred(series, rec, self.conventions)
        g = judges.at_instants(sc, self.t, truth.TOL_MIN)
        m = np.isfinite(g)
        E_curve = truth.e_rows(self.H, g)
        idx10, _ = truth.series10(self.pkg)
        w = settings.window(self.window)
        m10 = (idx10 >= pd.Timestamp(w.start)) & (idx10 < pd.Timestamp(w.end))
        idx_w = idx10[m10]
        g10 = judges.centred_10min(sc, rec["sampling_min"], grid=idx_w).to_numpy(np.float64)
        t1, h1 = truth.series1(self.pkg)
        m10min = levels_minutes(idx_w)
        p10 = np.searchsorted(t1, m10min)
        on = (len(t1) > 0 and (p10 < len(t1)).all()
              and np.array_equal(t1[np.minimum(p10, len(t1) - 1)], m10min))
        if on:
            _, E_all = truth.tau_all_window(idx_w, g10, t1, h1)
        else:
            E_all = np.full(len(TAUS_A), np.nan)
            self.warnings.append(f"{self.site} {self.window} {code}: the 10-min grid is not on "
                                 "the 1-min series; E_all undefined")
        e = self.entry(code)
        j = types.SimpleNamespace(code=code, rec=rec, g=g, m=m, n=int(m.sum()), E_curve=E_curve,
                                  g10=g10, p10=p10 if on else None, t1=t1, h1=h1,
                                  E_all=E_all, boot=None,
                                  tau_a=_f((e.get("a") or {}).get("tau")))
        self._j[code] = j
        return j

    def check_truth_curve(self, code, curves):
        """Max |E_g - truth E_a| over TAUS_A against the truth's curves npz
        (``<code>__<window>__E_a``); None when the curve is absent."""
        if curves is None:
            return None
        key = f"{code}__{self.window}__E_a"
        if key not in curves:
            return None
        ref = np.asarray(curves[key], np.float64)
        j = self.judge(code)
        if ref.shape != j.E_curve.shape:
            return float("inf")
        d = np.abs(ref - j.E_curve)
        d = d[np.isfinite(d)]
        return float(d.max()) if len(d) else 0.0

    # -- E, E_all, LLE ---------------------------------------------------------
    def E(self, code, tau):
        """E_g(tau) (m): the truth's E_a curve at an integer tau of TAUS_A
        (bitwise), else the same statistic at h_B(t - tau)."""
        tau = _f(tau)
        if not np.isfinite(tau):
            return NAN
        j = self.judge(code)
        if tau == round(tau) and TAUS_A[0] <= tau <= TAUS_A[-1]:
            return float(j.E_curve[int(round(tau)) - int(TAUS_A[0])])
        return float(truth.e_rows(self.h_at(tau)[None, :], j.g)[0])

    def E_all(self, code, tau):
        """E_all(tau) (m) over the window's 10-min samples: the truth's
        tau_all curve at an integer tau of TAUS_A, else h_B from the 1-min
        series at grid - tau, linear in minutes."""
        tau = _f(tau)
        j = self.judge(code)
        if not np.isfinite(tau) or j.p10 is None:
            return NAN
        if tau == round(tau) and TAUS_A[0] <= tau <= TAUS_A[-1]:
            return float(j.E_all[int(round(tau)) - int(TAUS_A[0])])
        lo = math.floor(tau)
        f = tau - lo
        q0 = j.p10 - int(lo)
        n1 = len(j.h1)
        hb = np.full(len(j.g10), np.nan)
        if f == 0.0:
            ok = (q0 >= 0) & (q0 < n1)
            hb[ok] = j.h1[q0[ok]]
        else:
            q1 = q0 - 1
            ok = (q1 >= 0) & (q0 < n1)
            hb[ok] = (1.0 - f) * j.h1[q0[ok]] + f * j.h1[q1[ok]]
        mm = np.isfinite(j.g10) & np.isfinite(hb)
        if not mm.any():
            return NAN
        d = hb[mm] - j.g10[mm]
        return float(np.sqrt(np.mean((d - np.median(d)) ** 2)))

    def E_all_min(self, code):
        """min over integer tau in TAUS_A of E_all (m)."""
        E = self.judge(code).E_all
        return float(np.nanmin(E)) if np.isfinite(E).any() else NAN

    def LLE(self, code, tau_m, tau_a):
        """sqrt(mean_i [h_B(t_i - tau_m) - h_B(t_i - tau_a)]^2) (m) over the
        instants with a finite gauge value."""
        j = self.judge(code)
        if not (np.isfinite(_f(tau_m)) and np.isfinite(_f(tau_a))) or not j.m.any():
            return NAN
        d = self.h_at(tau_m)[j.m] - self.h_at(tau_a)[j.m]
        return float(np.sqrt(np.mean(d ** 2)))

    # -- bootstrap -------------------------------------------------------------
    def weights(self, code):
        """(B, T_w) month-block multiplicities of a judge (shared by every
        product: paired)."""
        return self.boot(code).W_full

    def boot(self, code):
        """The judge's bootstrap state: weights, weighted E over TAUS_A,
        tau_a*, E0*, Emin*."""
        j = self.judge(code)
        if j.boot is None:
            seed = [*truth.window_boot_seed(self.site, code, self.window),
                    int(inference.INFERENCE_V2["boot"]["seed_extra"])]
            W = inference.month_block_weights(self.t, j.m, self.B, seed)
            Wm = W[:, j.m]
            if self.B and j.m.any():
                Ew = inference.weighted_mcrms(self.H[:, j.m] - j.g[j.m], Wm)
                fin = np.isfinite(Ew)
                ka = np.where(fin.any(axis=1),
                              np.argmin(np.where(fin, Ew, np.inf), axis=1), K0).astype(np.int64)
                E0 = Ew[:, K0]
                Emin = Ew[np.arange(len(ka)), ka]
            else:
                Ew = np.zeros((self.B, len(TAUS_A)))
                ka = np.zeros(self.B, np.int64)
                E0 = Emin = np.full(self.B, np.nan)
            j.boot = types.SimpleNamespace(seed=seed, W_full=W, W=Wm, Ew=Ew, ka=ka,
                                           tau_a=TAUS_A[ka].astype(np.float64), E0=E0,
                                           Emin=Emin)
        return j.boot

    def boot_draws(self, code, tau_m, lag_kind="linear"):
        """Per-draw e_a*, dE*, LLE*, AGF* (and Etau*) of a lag tau_m."""
        j = self.judge(code)
        bs = self.boot(code)
        if not self.B or not j.m.any() or not np.isfinite(_f(tau_m)):
            empty = np.full(self.B, np.nan)
            return {"e_a": empty, "dE": empty, "LLE": empty, "AGF": empty, "Etau": empty}
        h_tau = self.h_at(tau_m)[j.m]
        Etau = inference.weighted_mcrms((h_tau - j.g[j.m])[None, :], bs.W)[:, 0]
        LLE = np.full(self.B, np.nan)
        Wf = bs.W.astype(np.float64)
        for u in np.unique(bs.ka):
            sel = bs.ka == u
            d2 = (h_tau - self.H[u, j.m]) ** 2
            LLE[sel] = np.sqrt(Wf[sel] @ d2 / Wf[sel].sum(axis=1))
        if lag_kind == "phase":
            e_a = lc.circ_diff(float(tau_m), bs.tau_a)
        else:
            e_a = float(tau_m) - bs.tau_a
        dE = Etau - bs.Emin
        den = bs.E0 - bs.Emin
        with np.errstate(invalid="ignore", divide="ignore"):
            agf = np.where(den >= MATERIAL_M, (bs.E0 - Etau) / den, np.nan)
        return {"e_a": np.asarray(e_a, np.float64), "dE": dE, "LLE": LLE, "AGF": agf,
                "Etau": Etau}

    # -- per-lag metrics ---------------------------------------------------------
    def level_metrics(self, code, tau_m, lag_kind="linear"):
        """The level-domain metrics of a lag tau_m at a judge (cached):
        E0, Etau, Emin, dE, E_all, dE_all, LLE, explained, AGF, material and
        the bootstrap draws and CIs."""
        key = (code, round(_f(tau_m), 12) if np.isfinite(_f(tau_m)) else None, lag_kind)
        if key in self._lm:
            return self._lm[key]
        j = self.judge(code)
        tau_a = j.tau_a
        E0, Etau, Emin = self.E(code, 0), self.E(code, tau_m), self.E(code, tau_a)
        material = bool(np.isfinite(E0) and np.isfinite(Emin) and E0 - Emin >= MATERIAL_M)
        agf = (E0 - Etau) / (E0 - Emin) if material else NAN
        expl = 1.0 - (Etau / E0) ** 2 if np.isfinite(E0) and E0 > 0 else NAN
        Eall_tau, Eall_min = self.E_all(code, tau_m), self.E_all_min(code)
        lle = self.LLE(code, tau_m, tau_a)
        draws = self.boot_draws(code, tau_m, lag_kind)
        ci = {k: inference.percentile_ci(draws[k]) for k in ("e_a", "dE", "LLE", "AGF")}
        out = {"E0": E0, "Etau": Etau, "Emin": Emin, "dE": Etau - Emin, "Eall_tau": Eall_tau,
               "Eall_min": Eall_min, "dE_all": Eall_tau - Eall_min, "LLE": lle,
               "explained": expl, "AGF": agf, "material": material, "n_instants": j.n,
               "draws": draws, "ci": ci}
        self._lm[key] = out
        return out

    def truth_value(self, code, T):
        """``(tau_T, lo, hi)`` of a judge in this window."""
        e = self.entry(code)
        if T in ("a", "ab", "b"):
            d = e.get(T) or {}
            return _f(d.get("tau")), _f(d.get("lo")), _f(d.get("hi"))
        if T == "hwlw":
            return _f((e.get("hw_lw") or {}).get("mean_min")), NAN, NAN
        raise KeyError(f"unknown truth {T!r}")

    def score(self, code, taus, lag_kind="linear", interval=None):
        """Point metrics of one judge for every truth.

        Parameters
        ----------
        taus : dict {truth: tau_m} or float (the same tau_m for every truth)
        lag_kind : "linear", "phase", or dict {truth: kind} (a point
            product whose (b) estimand is an M2 phase: :func:`point_kinds`)
        interval : (lo, hi) or dict {truth: (lo, hi)}, optional
            The product's nominal 95 % interval at the gauge.

        Returns
        -------
        dict {truth: dict}: tau_m, tau_true, I_lo, I_hi, e, abs_e, e_lo, e_hi,
        err_int, the level metrics in cm, AGF and its CI, material, IS,
        covered, n_instants, B, flags, and ``draws`` (shared per tau_m).
        """
        if not isinstance(taus, Mapping):
            taus = {T: taus for T in TRUTHS}
        kinds = lag_kind
        out = {}
        for T in TRUTHS:
            lag_kind = (kinds.get(T, "linear") if isinstance(kinds, Mapping) else kinds)
            tau_m = _f(taus.get(T))
            tv, lo, hi = self.truth_value(code, T)
            r = {"tau_m": tau_m, "tau_true": tv, "I_lo": lo, "I_hi": hi, "flags": []}
            if lag_kind == "phase":
                e = float(lc.circ_diff(tau_m, tv)) if np.isfinite(tau_m) and np.isfinite(tv) \
                    else NAN
            else:
                e = tau_m - tv
            r["e"], r["abs_e"] = e, abs(e)
            if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(tau_m):
                x = tv + e if np.isfinite(e) else tau_m      # tau_m on the truth's branch
                r["err_int"] = 0.0 if lo <= x <= hi else float(min(abs(x - lo), abs(x - hi)))
                r["hitI"] = float(lo <= x <= hi)
            else:
                r["err_int"], r["hitI"] = NAN, NAN
            lm = self.level_metrics(code, tau_m, lag_kind) if np.isfinite(tau_m) else None
            if lm is None:
                r.update({k: NAN for k in ("E0_cm", "Etau_cm", "Emin_cm", "dE_cm", "dE_lo_cm",
                                            "dE_hi_cm", "Eall_tau_cm", "Eall_min_cm",
                                            "dE_all_cm", "LLE_cm", "LLE_lo_cm", "LLE_hi_cm",
                                            "explained", "AGF", "AGF_lo", "AGF_hi", "e_lo",
                                            "e_hi")})
                r.update(material=None, n_instants=self.judge(code).n, B=self.B, draws=None)
            else:
                r.update(E0_cm=_cm(lm["E0"]), Etau_cm=_cm(lm["Etau"]), Emin_cm=_cm(lm["Emin"]),
                         dE_cm=_cm(lm["dE"]), dE_lo_cm=_cm(lm["ci"]["dE"][0]),
                         dE_hi_cm=_cm(lm["ci"]["dE"][1]), Eall_tau_cm=_cm(lm["Eall_tau"]),
                         Eall_min_cm=_cm(lm["Eall_min"]), dE_all_cm=_cm(lm["dE_all"]),
                         LLE_cm=_cm(lm["LLE"]), LLE_lo_cm=_cm(lm["ci"]["LLE"][0]),
                         LLE_hi_cm=_cm(lm["ci"]["LLE"][1]), explained=lm["explained"],
                         AGF=lm["AGF"], AGF_lo=lm["ci"]["AGF"][0], AGF_hi=lm["ci"]["AGF"][1],
                         material=lm["material"], n_instants=lm["n_instants"], B=self.B,
                         draws=lm["draws"])
                if T == "a":
                    r["e_lo"], r["e_hi"] = lm["ci"]["e_a"]
                else:
                    r["e_lo"], r["e_hi"] = NAN, NAN
                if not lm["material"]:
                    r["flags"].append("lag not material")
            iv = interval.get(T) if isinstance(interval, Mapping) else interval
            r["IS"], r["covered"] = interval_score(iv, tv, lag_kind)
            out[T] = r
        return out


def levels_minutes(idx):
    """Minutes since the epoch of a DatetimeIndex (``levels.minutes_since_epoch``)."""
    from experiments.benchmark import levels

    return levels.minutes_since_epoch(idx)


def interval_score(interval, tau_true, lag_kind="linear", alpha=IS_ALPHA):
    """``(IS_alpha, covered)`` of an interval (lo, hi) against a truth.

    IS = (hi - lo) + (2/alpha)(lo - tau)1[tau < lo] + (2/alpha)(tau - hi)1[tau > hi]
    (Gneiting & Raftery 2007). Phase intervals: the truth is moved to the
    branch nearest the interval centre first. NaN when undefined.
    """
    if interval is None:
        return NAN, NAN
    lo, hi = _f(interval[0]), _f(interval[1])
    tv = _f(tau_true)
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(tv)):
        return NAN, NAN
    if lag_kind == "phase":
        c = 0.5 * (lo + hi)
        tv = c + float(lc.circ_diff(tv, c))
    k = 2.0 / float(alpha)
    s = (hi - lo) + k * (lo - tv) * (tv < lo) + k * (tv - hi) * (tv > hi)
    return float(s), float(lo <= tv <= hi)


# ─────────────────────────────────────────────────────────────────────────────
#  products, instances, plans, validation
# ─────────────────────────────────────────────────────────────────────────────

def product_dir(root, site, window, setting, product_id, variant="", family="rand", mode="A",
                N="all", rep=0):
    """``<root>/runs/<site>/<window>/<setting>/<product>[__<variant>]/<family>-<mode>/
    <Nall|N%03d>_r%02d`` (plan INTERFACES §6)."""
    name = f"{product_id}__{variant}" if variant else str(product_id)
    if str(N) == "all":
        n = "Nall"
    elif str(N).startswith("rule:"):
        n = "Nrule" + str(N).split(":", 1)[1]          # instance.leaf_name
    else:
        n = f"N{int(N):03d}"
    return os.path.join(root, "runs", site, window, setting, name, f"{family}-{mode}",
                        f"{n}_r{int(rep):02d}")


def _merge_groups(run):
    out = dict(run)
    for grp in ("identity", "outcome", "keys_"):
        g = run.get(grp)
        if isinstance(g, Mapping):
            for k, v in g.items():
                out.setdefault(k, v)
    return out


def _as_np(v, dtype=np.float64):
    return None if v is None else np.asarray(v, dtype)


def normalise_product(d, path=None):
    """A lag product in one shape, whatever reader produced it
    (``queue_v2.load_lag_product`` or :func:`read_lag_product`): run.json's
    identity and outcome fields at the top, ``field``, ``band`` {tau, n_px,
    edge, lo, hi}, ``interval95`` {level, lo, hi, nominal}, ``point``,
    ``groups`` (tau arrays), ``instance_path`` (absolute), ``timings``."""
    run = _merge_groups(d)
    arrays = {}
    if isinstance(run.get("arrays"), Mapping):
        arrays.update(run["arrays"])
    for k in list(run):
        if k in ARRAY_KEYS and isinstance(run[k], np.ndarray):
            arrays.setdefault(k, run[k])
    band = None
    if arrays.get("band_tau") is not None:
        band = {"tau": _as_np(arrays["band_tau"]),
                "n_px": _as_np(arrays.get("band_n_px"), np.int64),
                "edge": _as_np(arrays.get("band_edge"), bool),
                "lo": _as_np(arrays.get("band_lo")), "hi": _as_np(arrays.get("band_hi"))}
    elif isinstance(run.get("band"), Mapping) and run["band"].get("tau") is not None:
        b = run["band"]
        band = {"tau": _as_np(b["tau"]), "n_px": _as_np(b.get("n_px"), np.int64),
                "edge": _as_np(b.get("edge"), bool), "lo": _as_np(b.get("lo")),
                "hi": _as_np(b.get("hi"))}
    field = arrays.get("field")
    if field is None and isinstance(run.get("field"), (np.ndarray, list)):
        field = run["field"]
    iv = run.get("interval95")
    if isinstance(iv, Mapping):
        iv = dict(iv)
        if iv.get("lo") is None:
            if arrays.get("iv_lo") is not None:
                iv["lo"], iv["hi"] = arrays["iv_lo"], arrays.get("iv_hi")
            elif iv.get("level") == "band" and band is not None and band["lo"] is not None:
                iv["lo"], iv["hi"] = band["lo"], band["hi"]
        iv["lo"], iv["hi"] = _as_np(iv.get("lo")), _as_np(iv.get("hi"))
        if iv["lo"] is None or iv["hi"] is None:
            iv = None
    else:
        iv = None
    groups = run.get("groups")
    if isinstance(groups, Mapping):
        groups = {str(k): dict(g, tau=_as_np(g.get("tau"))) for k, g in groups.items()}
    else:
        groups = None
    inst = run.get("instance")
    ipath, irec = None, None
    if isinstance(inst, Mapping):
        if "keep" in inst:          # queue_v2.load_lag_product: the instance record itself
            irec = inst
        if isinstance(inst.get("path"), str):
            ipath = inst["path"]
    elif isinstance(inst, str):
        ipath = inst
    if ipath is None and path is not None and os.path.exists(os.path.join(path, "run.json")):
        ri = product.read_json(os.path.join(path, "run.json")).get("instance")
        if isinstance(ri, Mapping) and isinstance(ri.get("path"), str):
            ipath = ri["path"]
    if ipath is not None and path is not None and not os.path.isabs(ipath):
        ipath = os.path.normpath(os.path.join(path, ipath))

    def join(o):
        if isinstance(o, Mapping):
            if set(o) == {"__npz__"} and o["__npz__"] in arrays:
                return arrays[o["__npz__"]]
            return {k: join(v) for k, v in o.items()}
        if isinstance(o, list):
            return [join(v) for v in o]
        return o

    run["diag"] = join(run.get("diag") or {})
    return {"dir": path, "product": run.get("product"), "method": run.get("method"),
            "variant": run.get("variant", ""), "site": run.get("site"),
            "window": run.get("window"), "setting": run.get("setting"),
            "status": run.get("status", "ok"), "stage": run.get("stage"),
            "reason": run.get("reason"), "kind": run.get("kind"),
            "lag_kind": run.get("lag_kind") or "linear",
            "edge_share": _f(run.get("edge_share")), "n_defined": run.get("n_defined"),
            "point": run.get("point"), "groups": groups, "interval95": iv,
            "field": _as_np(field), "band": band, "instance_path": ipath,
            "instance_record": irec,
            "timings": run.get("timings") or {}, "unit_total": run.get("unit_total") or {},
            "memory": run.get("memory") or {}, "keys": run.get("keys") or {},
            "monitor": run.get("monitor") or {}, "diag": run.get("diag") or {}}


def read_lag_product(path):
    """A product directory as :func:`normalise_product` (run.json plus
    product.npz); None when it holds no run.json."""
    rj = os.path.join(path, "run.json")
    if not os.path.exists(rj):
        return None
    run = product.read_json(rj)
    npz = os.path.join(path, "product.npz")
    arrays = product.load_npz(npz) if os.path.exists(npz) else {}
    return normalise_product(dict(run, arrays=arrays), path)


def load_product(path):
    """Read a product through ``queue_v2.load_lag_product`` when the queue
    module provides it, else :func:`read_lag_product`; None when absent."""
    if not os.path.exists(os.path.join(path, "run.json")):
        return None
    loader = None
    if os.path.exists(os.path.join(path, "product.npz")):   # the queue's reader needs it
        try:
            from experiments.benchmark import queue_v2
            loader = getattr(queue_v2, "load_lag_product")
        except (ImportError, AttributeError):
            loader = None
    if loader is not None:
        try:
            d = loader(path)
            if d is not None:
                return normalise_product(d, path)
        except Exception as exc:      # fall back to the documented layout
            log(f"[score] queue_v2.load_lag_product({path}) failed ({exc}); reading the files")
    return read_lag_product(path)


def read_instance(path):
    """The instance arrays (``instance.npz``) of a product: keep, band_of,
    s_km, centers, K, edges, sel, t_ns, orbit, platform, status."""
    p = path if str(path).endswith(".npz") else os.path.join(path, "instance.npz")
    return instance_from_arrays(product.load_npz(p), p)


def instance_from_arrays(z, p):
    """The instance dict of :func:`read_instance` from the instance arrays
    (``instance.npz`` members, or ``instance.read_instance``'s record)."""
    keep = np.asarray(z["keep"], np.int64)
    centers = np.asarray(z.get("centers", np.zeros(0)), np.float64)
    out = {"path": p, "keep": keep, "band_of": np.asarray(z["band_of"]).astype(np.int64),
           "s_km": np.asarray(z["s_km"], np.float64), "centers": centers,
           "K": int(z["K"]) if "K" in z else len(centers),
           "edges": np.asarray(z.get("edges", np.zeros(0)), np.float64)}
    for k in ("sel", "t_ns", "orbit", "platform", "status", "h0"):
        if k in z:
            out[k] = z[k]
    return out


def read_plan(root, design="lag"):
    """``plans/<design>.csv`` as a list of dicts (designs.CSV_COLUMNS), or
    None when absent."""
    p = os.path.join(root, "plans", f"{design}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return df.to_dict("records")


def _ledger(root, unit_id):
    p = os.path.join(root, "claims", "status", f"{unit_id}.json")
    if unit_id and os.path.exists(p):
        try:
            return product.read_json(p)
        except (OSError, ValueError):
            return None
    return None


def read_lock(root):
    """``<root>/LOCK.json`` or None."""
    p = os.path.join(root, "LOCK.json")
    return product.read_json(p) if os.path.exists(p) else None


def lock_phase_a(doc):
    """True when a LOCK.json document records phase A (or B, which implies it)."""
    if not isinstance(doc, Mapping):
        return False
    phases = doc.get("phases")
    if isinstance(phases, Mapping):            # lock.write_lock: {"phases": {"A": {...}}}
        return bool(phases.get("A"))
    ph = doc.get("phase")
    if ph in ("A", "B"):
        return True
    if isinstance(phases, (list, tuple)) and ("A" in phases or "B" in phases):
        return True
    return bool(doc.get("phase_A") or doc.get("lock_A"))


def lock_development(doc):
    """The development judges LOCK.json adds (the 'extras' of
    ``judges.development_list``)."""
    if not isinstance(doc, Mapping):
        return ()
    for path in (("development_judges",), ("judges", "development"),
                 ("extras", "development_judges")):
        v = doc
        for k in path:
            v = v.get(k) if isinstance(v, Mapping) else None
        if v:
            return tuple(str(c[0] if isinstance(c, (list, tuple)) else c) for c in v)
    return ()


def _read_json_or_none(p):
    if os.path.exists(p):
        try:
            return product.read_json(p)
        except (OSError, ValueError):
            return None
    return None


def control_verdict_of(doc):
    """The overall verdict of a ``control/<product>.json`` document, or None
    (no control, or the gating world not complete).

    Read in this order: a documented fault set by hand (``"fault": "<what>"``
    -> "FAIL (fault)"); an explicit ``"verdict"``; the control driver's
    ``"gating"`` summary (PASS / control mismatch; "pending" -> None) with a
    "FAIL (fault)" in any gating world winning; ``"overall"``; per-scenario
    ``"scenarios"`` (PASS only when every scenario passed)."""
    if not isinstance(doc, Mapping):
        return None
    if doc.get("fault"):
        return "FAIL (fault)"
    if doc.get("verdict"):
        return str(doc["verdict"])
    g = doc.get("gating")
    if isinstance(g, str) and g:
        worlds = doc.get("worlds") or {}
        if any(str((w or {}).get("verdict", "")).startswith("FAIL")
               for k, w in worlds.items() if str(k).startswith("gating")
               and isinstance(w, Mapping)):
            return "FAIL (fault)"
        return None if g == "pending" else g
    ov = doc.get("overall")
    if isinstance(ov, Mapping) and ov.get("verdict"):
        return str(ov["verdict"])
    sc = doc.get("scenarios")
    if isinstance(sc, Mapping) and sc:
        vs = [str(v.get("verdict") if isinstance(v, Mapping) else v) for v in sc.values()]
        if any(v.startswith("FAIL") for v in vs):
            return "FAIL (fault)"
        return "PASS" if all(v == "PASS" for v in vs) else "control mismatch"
    return None


def _product_code_key(prod):
    """The ``code_files_key`` the scored product's run.json records (None
    for a product written before the key existed)."""
    if not isinstance(prod, Mapping):
        return None
    keys = prod.get("keys")
    if isinstance(keys, Mapping):
        return keys.get("code_files_key")
    return None


def validation(root, site, product_id, prod=None):
    """``(validated, flags, text)`` of a product at a site under the
    printing rule (§9.7; :data:`SCORING_V2` printing).

    An imagery row also needs a CURRENT control and current gates: the
    control document's ``code_files_key`` must equal the scored product's
    (``prod``, its run.json keys; a control made with other adapter code is
    'stale'), and each required gate's ``code_sha256`` must equal the current
    sources (``gates.gate_code_stale``)."""
    pr = SCORING_V2["printing"]
    base = str(product_id).split("@")[0]
    flags = []
    cdoc = _read_json_or_none(os.path.join(root, "control", f"{base}.json"))
    verdict = control_verdict_of(cdoc)
    if base in pr["imagery"]:
        if verdict is None:
            return False, ["no positive control"], ("implementation not validated (no "
                                                    "positive control)")
        if verdict not in pr["control_ok"]:
            return (False, [f"control {verdict}"],
                    f"implementation not validated (control {verdict})")
        if prod is not None:
            pk = _product_code_key(prod)
            ck = (cdoc or {}).get("code_files_key") if isinstance(cdoc, Mapping) else None
            if pk is None or ck is None or pk != ck:
                why = ("the scored product records no code key" if pk is None else
                       "the control records no code key" if ck is None else
                       "the control ran other adapter code")
                return (False, flags + ["control stale"],
                        f"implementation not validated (control stale: {why}; rerun "
                        "control --kind lag)")
        if verdict == "control mismatch":
            flags.append("control mismatch")
        from experiments.benchmark import gates as _gates
        for g in pr["required_gates"].get(base, ()):
            gd = _read_json_or_none(os.path.join(root, "gates", site, f"{g}.json"))
            gv = None if gd is None else gd.get("verdict")
            if gv != "PASS":
                return False, flags + [f"G-{g} {gv or 'NOT RUN'}"], \
                    f"implementation not validated (G-{g} {gv or 'NOT RUN'})"
            stale = _gates.gate_code_stale(gd, g)
            if stale:
                return False, flags + [f"G-{g} stale"], \
                    f"implementation not validated (G-{g} stale: {stale}; rerun gate {g})"
        if base in pr["fidelity_pending"]:
            flags.append(f"fidelity check pending: {pr['fidelity_pending'][base]}")
        return True, flags, None
    if verdict is not None and verdict not in pr["control_ok"]:
        return False, [f"control {verdict}"], f"implementation not validated (control {verdict})"
    if verdict == "control mismatch":
        flags.append("control mismatch")
    return True, flags, None


class _Store:
    """Products, instances and judge geometries of one scoring pass (cached)."""

    def __init__(self, root):
        self.root = root
        self._prod = {}
        self._inst = {}
        self._geom = {}
        self.cache = {}             # per-product derived arrays (band medians)

    def product(self, path):
        if path not in self._prod:
            try:
                self._prod[path] = load_product(path)
            except Exception as exc:
                self._prod[path] = {"status": "unreadable", "reason": f"{type(exc).__name__}: "
                                    f"{exc}", "dir": path}
        return self._prod[path]

    def drop(self, path):
        """Release a product's arrays once it is scored."""
        p = self._prod.get(path)
        if isinstance(p, dict):
            for k in ("field", "band", "interval95", "instance_record"):
                if k in p:
                    p[k] = None
        self.cache.pop((path, "band_meds"), None)
        self.cache.pop((path, "l01_boot_fields"), None)

    def clear_window(self):
        """Release the instances and geometries of a scored (site, window):
        windows never share an instance."""
        self._inst.clear()
        self._geom.clear()
        self.cache.clear()

    def instance(self, prod):
        ip = prod.get("instance_path")
        rec = prod.get("instance_record")
        if ip is None and rec is None:
            raise FileNotFoundError("run.json names no instance")
        key = os.path.normpath(ip) if ip is not None else f"record:{prod.get('dir')}"
        if key not in self._inst:
            if rec is not None:
                if ip is None or str(ip).endswith(".npz"):
                    p = ip
                else:
                    p = os.path.join(ip, "instance.npz")
                self._inst[key] = instance_from_arrays(rec, p or key)
            else:
                self._inst[key] = read_instance(ip)
        return self._inst[key]

    def geom(self, inst, rec, grid):
        key = (inst["path"], rec["code"])
        if key not in self._geom:
            self._geom[key] = rd.judge_geometry(rec, grid, inst["keep"], inst["s_km"],
                                                inst["band_of"], inst["centers"])
        return self._geom[key]


# ─────────────────────────────────────────────────────────────────────────────
#  scoring one site
# ─────────────────────────────────────────────────────────────────────────────

POINTS_COLUMNS = lagtable.POINTS_COLUMNS


def _site_context(root, site, *, cubes_root=None, packages=None, truth_docs=None,
                  judge_tables=None, dev_extra=(), res=10):
    s = types.SimpleNamespace(site=site, pkg=None, pkg_error=None, doc=None, doc_error=None,
                              table=[], table_error=None, grid=None, years=[], curves=None,
                              dev=judges.development_list(dev_extra), series={},
                              truth_from_disk=False, cubes_root=cubes_root)
    if packages is not None and site in packages:
        s.pkg = packages[site]
    else:
        try:
            from experiments.benchmark import package
            s.pkg = package.V2Package(package.v2_site(site, root=root, cubes_root=cubes_root,
                                                      res=int(res or 10)))
        except Exception as exc:
            s.pkg_error = f"{type(exc).__name__}: {exc}"
    if s.pkg is not None:
        s.years = [int(y) for y in getattr(s.pkg, "years", [])]
        try:
            s.grid = getattr(s.pkg, "grid", None)
            if s.grid is None:
                s.grid = judges.grid_of(s.pkg)
        except Exception as exc:
            s.grid = None
            s.table_error = f"grid: {type(exc).__name__}: {exc}"
    tdir = os.path.join(root, "truth", site)
    if truth_docs is not None and site in truth_docs:
        s.doc = truth_docs[site]
    else:
        p = os.path.join(tdir, "truth_lag_v2.json")
        if os.path.exists(p):
            doc = product.read_json(p)
            if doc.get("format") != truth.TRUTH_LAG_V2_FORMAT:
                s.doc_error = f"{p} is not a {truth.TRUTH_LAG_V2_FORMAT} file"
            else:
                s.doc = doc
                s.truth_from_disk = True
        else:
            s.doc_error = f"no lag truth ({p}; run truth {site})"
        cp = os.path.join(tdir, "truth_lag_v2_curves.npz")
        if os.path.exists(cp):
            try:
                s.curves = product.load_npz(cp)
            except Exception:
                s.curves = None
    if judge_tables is not None and site in judge_tables:
        s.table = list(judge_tables[site])
    else:
        p = os.path.join(tdir, "judges.json")
        if os.path.exists(p) and s.grid is not None:
            try:
                s.table = judges.load(p, s.grid)
            except Exception as exc:
                s.table_error = f"{type(exc).__name__}: {exc}"
        elif config.JUDGES.get(site):
            s.table_error = s.table_error or f"no judge table ({p})"
    return s


def _window_guard(s, w):
    """None when the window is scorable at the site, else the row text."""
    if s.pkg is None:
        return f"NOT AVAILABLE (package: {s.pkg_error})"
    miss = [y for y in settings.window(w).years if y not in s.years]
    if miss:
        return f"NOT AVAILABLE: store years {miss}"
    return None


def _judges_at(s, w):
    """Judges of a window: (code, record, included, text) in table order."""
    by_doc = {st["code"]: st for st in (s.doc or {}).get("stations", [])}
    table = {r["code"]: r for r in s.table}
    codes = [r["code"] for r in s.table] + [c for c in by_doc if c not in table]
    out = []
    for code in codes:
        rec = dict(by_doc.get(code) or {})
        rec.update(table.get(code) or {})
        rec.setdefault("location", judges.location_of(code))
        e = (by_doc.get(code) or {}).get("windows", {}).get(w)
        if s.doc is None:
            inc, text = False, f"NOT AVAILABLE ({s.doc_error or 'no lag truth'})"
        elif e is None:
            inc, text = False, "NOT AVAILABLE (no lag truth for this judge and window)"
        else:
            ii = e.get("inclusion") or {}
            inc = bool(ii.get("included"))
            text = None if inc else ("judge excluded (" + ", ".join(ii.get("exclusion") or
                                                                    ["rules"]) + ")")
            if inc and code not in table:
                # the read-out needs the judge-table record (nbhd, position, provider)
                inc = False
                text = ("NOT AVAILABLE (no judge-table record"
                        + (f": {s.table_error}" if s.table_error else "") + ")")
        out.append((code, rec, inc, text))
    return out


def _status_from_plan(root, row, pid, store):
    """``(text or None, product or None, kind)``: the row's status text
    (None = scorable)."""
    if row.get("status") == "not_applicable":
        note = str(row.get("note") or "")
        if note.upper().startswith("NOT AVAILABLE"):
            why = note.split(":", 1)[1].strip() if ":" in note else note
            return f"NOT AVAILABLE ({why})", None, "not_applicable"
        return f"not applicable ({note})", None, "not_applicable"
    d = product_dir(root, row["site"], row["window"], row["setting_id"], pid,
                    row.get("variant") or "", row.get("family") or "rand",
                    row.get("mode") or "A", row.get("N") or "all", row.get("rep") or 0)
    prod = store.product(d)
    if prod is None:
        led = _ledger(root, row.get("unit_id"))
        if led:
            st, why = led.get("status"), led.get("reason")
            if st == "not_computable":
                return f"NOT COMPUTABLE ({why or 'unspecified'})", None, st
            if st == "not_applicable":
                return f"not applicable ({why or row.get('note') or ''})", None, st
            return f"NOT AVAILABLE ({st}{': ' + str(why) if why else ''})", None, str(st)
        return "NOT AVAILABLE (not run)", None, "missing"
    st = prod.get("status")
    if st == "ok":
        return None, prod, "ok"
    stage, why = prod.get("stage"), prod.get("reason")
    if st == "not_computable":
        return f"NOT COMPUTABLE ({stage or why or 'unspecified'})", prod, st
    if st == "empty":
        return f"NOT COMPUTABLE (empty{': ' + str(stage or why) if (stage or why) else ''})", \
            prod, st
    if st == "not_applicable":
        return f"not applicable ({why or row.get('note') or ''})", prod, st
    if st == "unavailable":
        return f"NOT AVAILABLE ({why or 'unavailable'})", prod, st
    return f"NOT AVAILABLE (run status {st}{': ' + str(why) if why else ''})", prod, str(st)


def _base_row(s, w, code, rec, pid, info, row):
    return {"site": s.site, "window": w, "judge": code,
            "location": rec.get("location") or judges.location_of(code),
            "s_g_km": _f(rec.get("s_g_km")), "mouth": bool(rec.get("mouth", False)),
            "development": code in s.dev, "product": pid,
            "method": (row or {}).get("method") or info.get("method", ""),
            "variant": (row or {}).get("variant") or "", "family": info.get("family", ""),
            "fidelity": info.get("fidelity", ""), "lag_kind": info.get("lag_kind", "linear"),
            # judge-table flags carried to the row (scoring spec §3: 'judge sparse')
            "_jflags": ["judge sparse"] if "sparse" in (rec.get("flags") or []) else []}


def _status_rows(base, text, reason=None, validated=None):
    out = []
    for T in TRUTHS:
        r = dict(base, status=text, reason=reason, validated=validated, truth=T,
                 readout_step=None, flags="")
        out.append(r)
    return out


def _point_rows(base, sc, ro_info, validated, vflags, status="ok", extra_flags=()):
    """Rows of one product at one judge from LagScorer.score's output."""
    out, draws = [], []
    for T in TRUTHS:
        m = dict(sc[T])
        d = m.pop("draws", None)
        fl = list(m.pop("flags", [])) + list(vflags) + list(extra_flags) + \
            list(ro_info.get("flags", [])) + list(base.get("_jflags") or [])
        r = dict(base, status=status, reason=None, validated=validated, truth=T,
                 readout_step=ro_info.get("step"), n_readout_px=ro_info.get("n_px"),
                 edge=ro_info.get("edge"), tau_m_band=ro_info.get("tau_band", NAN),
                 tau_m_band_interp=ro_info.get("tau_band_interp", NAN),
                 tau_m_step1=ro_info.get("tau_step1", NAN),
                 flags=";".join(dict.fromkeys(f for f in fl if f)), **m)
        out.append(r)
        draws.append(d if T == "a" or d is None else {k: v for k, v in d.items() if k != "e_a"})
    return out, draws


def _l20_values(s, w, judges_w):
    """{truth: {location: tau}} of the leave-one-out reference (L20)."""
    loc_s, loc_tau = {}, {T: {} for T in TRUTHS}
    for code, rec, inc, _ in judges_w:
        if not inc:
            continue
        loc = rec.get("location") or judges.location_of(code)
        e = ((next((st for st in s.doc.get("stations", []) if st["code"] == code), {}) or {})
             .get("windows", {}).get(w) or {})
        loc_s.setdefault(loc, []).append(_f(rec.get("s_g_km")))
        for T in TRUTHS:
            if T == "hwlw":
                v = _f((e.get("hw_lw") or {}).get("mean_min"))
            else:
                v = _f((e.get(T) or {}).get("tau"))
            loc_tau[T].setdefault(loc, []).append(v)
    out = {T: {} for T in TRUTHS}
    for T in TRUTHS:
        sv = {k: float(np.nanmean(v)) if np.isfinite(v).any() else NAN
              for k, v in loc_s.items()}
        tv = {k: float(np.nanmean(v)) if np.isfinite(v).any() else NAN
              for k, v in loc_tau[T].items()}
        for loc in loc_s:
            nodes = [(sv[k], tv[k]) for k in loc_s if k != loc and np.isfinite(sv[k])
                     and np.isfinite(tv[k])]
            nodes = sorted([(0.0, 0.0)] + nodes)
            xs = np.array([n[0] for n in nodes])
            ys = np.array([n[1] for n in nodes])
            out[T][loc] = float(np.interp(sv[loc], xs, ys)) if np.isfinite(sv[loc]) else NAN
    return out


#: point products whose (b) estimand is an M2 phase lag (wrap180 x 60/28.98...):
#: scored circularly for truth (b) (spec §6.3 "model (b)")
PHASE_POINT_PRODUCTS = ("L14", "L15", "L16")


def point_kinds(pid, lag_kind="linear"):
    """The lag kind per truth of a point product: 'phase' for the (b) key of
    the models (:data:`PHASE_POINT_PRODUCTS`), the product's kind otherwise."""
    base = str(pid).split("@")[0]
    if base not in PHASE_POINT_PRODUCTS:
        return lag_kind
    keys = set(SCORING_V2["point_phase_keys"])
    return {T: ("phase" if T in keys else lag_kind) for T in TRUTHS}


def point_wrap(taus, kinds):
    """Phase-type point values wrapped about the truth-free centre (§6.3)."""
    if not isinstance(kinds, Mapping):
        return taus
    out = dict(taus)
    for T, k in kinds.items():
        v = _f(out.get(T))
        if k == "phase" and np.isfinite(v):
            out[T] = float(lc.wrap_phase(v))
    return out


def l01_boot_fields(prod, inst):
    """(n_boot, n_keep) float32: each bootstrap draw of an L01 product
    (``diag['boot']``, m2a's raw band lags) through MAREA's causality clip
    and applied smoothed lag (``marea_family.causality``, ``applied_lag``) on
    the instance's keep pixels (s from the instance record, float32: the
    rounding to the minute may differ from the product's own field by one
    minute at a few pixels; declared)."""
    from experiments.benchmark.lags import marea_family as mf
    boot = np.asarray((prod.get("diag") or {}).get("boot"), np.float64)
    if boot.ndim != 2:
        return None
    centers = np.asarray(inst["centers"], np.float64)
    s = np.asarray(inst["s_km"], np.float64)
    out = np.empty((boot.shape[0], len(s)), np.float32)
    for b in range(boot.shape[0]):
        tau_used, _ = mf.causality(boot[b])
        out[b] = mf.applied_lag(centers, tau_used, s).astype(np.float32)
    return out


def interval_at_readout(prod, ro, lag_kind, inst=None, store=None):
    """``(lo, hi)``: the product's nominal 95 % interval AT ITS READ-OUT
    (spec §6.4 interval score; :data:`SCORING_V2` interval_at_readout).

    * L01 with its bootstrap (``diag['boot']``): the 2.5/97.5 percentiles of
      the draws' read-outs (:func:`l01_boot_fields`, read at the point
      read-out's positions with the same median);
    * a product with refit fields (``diag['refit_tau']``, L08 GM20): the
      read-out +- 1.96 x IQR(refit read-out - read-out) / 1.349 x
      sqrt(m / (n - m)) (m of n scenes per refit: the subsample spread
      rescaled to the full-sample standard error), circular for phase;
    * otherwise ``readout.interval_readout`` of the stored interval (band:
      [lo_k, hi_k]; pixel: the medians of lo and hi).

    NaN at step 3 (fallback 0) or without an interval."""
    nan = (NAN, NAN)
    if ro is None or ro.get("step") not in (1, 2):
        return nan
    pos = np.asarray(ro.get("positions", ()), np.int64)
    diag = prod.get("diag") or {}
    tau = _f(ro.get("tau"))
    if not np.isfinite(tau):
        return nan
    if diag.get("boot") is not None and inst is not None and len(pos):
        key = (prod.get("dir"), "l01_boot_fields")
        cache = store.cache if store is not None else {}
        if key not in cache:
            cache[key] = l01_boot_fields(prod, inst)
        F = cache[key]
        if F is not None and F.shape[1] == len(np.asarray(inst["keep"])):
            reps = np.array([rd.median_of(F[b, pos], lag_kind) for b in range(F.shape[0])])
            reps = reps[np.isfinite(reps)]
            if len(reps) >= 2:
                if lag_kind == "phase":
                    d = lc.circ_diff(reps, tau)
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    return float(tau + lo), float(tau + hi)
                lo, hi = np.percentile(reps, [2.5, 97.5])
                return float(lo), float(hi)
        return nan
    if diag.get("refit_tau") is not None and len(pos):
        R = np.asarray(diag["refit_tau"])
        m, n = _f(diag.get("refit_subset_size")), _f(diag.get("T"))
        if R.ndim != 2 or not (np.isfinite(m) and np.isfinite(n) and n > m > 0):
            return nan
        reps = np.array([rd.median_of(R[k, pos], lag_kind) for k in range(R.shape[0])])
        d = lc.circ_diff(reps, tau) if lag_kind == "phase" else reps - tau
        d = d[np.isfinite(d)]
        if len(d) < 2:
            return nan
        q25, q75 = np.percentile(d, [25.0, 75.0])
        half = 1.96 * (q75 - q25) / 1.349 * math.sqrt(m / (n - m))
        return float(tau - half), float(tau + half)
    return rd.interval_readout(prod.get("interval95"), ro, lag_kind)


def _score_window(root, s, w, prows, store, *, B, series_by_code, collect):
    """Rows (and bootstrap draws) of every lag product of one site and window."""
    rows, draws = [], []
    guard = _window_guard(s, w)
    judges_w = _judges_at(s, w) if (s.doc is not None or s.table) else []
    lag_items = [(row, pid) for row in prows for pid in str(row.get("products", "")).split()
                 if pid.startswith("L")]

    def add(rr, dd=None):
        rows.extend(rr)
        draws.extend(dd if dd is not None else [None] * len(rr))

    has_judges = bool(config.JUDGES.get(s.site))
    if not judges_w and has_judges:
        # a judged site without truth or judge table: one row per product
        text = guard or f"NOT AVAILABLE ({s.doc_error or s.table_error or 'no judges'})"
        for row, pid in lag_items:
            info = lagtable.product_info(pid)
            base = _base_row(s, w, "", {}, pid, info, row)
            add(_status_rows(base, text))
        return rows, draws
    if guard is not None:
        for row, pid in lag_items:
            info = lagtable.product_info(pid)
            for code, rec, inc, text in judges_w:
                base = _base_row(s, w, code, rec, pid, info, row)
                add(_status_rows(base, guard))
            if not judges_w:
                collect["coherence_status"].append((s.site, w, pid, guard, None))
        return rows, draws
    included = [(c, r) for c, r, inc, _ in judges_w if inc]
    scorer = None
    if included:
        if series_by_code:
            for k, v in series_by_code.items():
                s.series.setdefault(k, v)
        scorer = LagScorer(s.pkg, s.doc, s.table, w, B=B, series_by_code=s.series,
                           site=s.site)
        bad = {}
        for code, _ in included:
            try:
                scorer.judge(code)
            except Exception as exc:          # a missing gauge file: that judge only
                bad[code] = f"NOT AVAILABLE (gauge: {type(exc).__name__}: {exc})"
                collect["warnings"].append(f"{s.site} {w} {code}: {bad[code]}")
                continue
            dev = scorer.check_truth_curve(code, s.curves)
            if dev is not None and dev > 1e-9:
                # the gauge, its conventions or the levels changed after 'truth':
                # every metric of this judge would be against another curve
                bad[code] = (f"NOT AVAILABLE (truth stale: E_g differs from the truth's E_a "
                             f"curve by {dev:.3g} m; rerun truth {s.site})")
                collect["warnings"].append(f"{s.site} {w} {code}: {bad[code]}")
            elif dev is None and s.truth_from_disk:
                bad[code] = ("NOT AVAILABLE (truth stale: no E_a curve of this judge and window "
                             f"in truth_lag_v2_curves.npz; rerun truth {s.site})")
                collect["warnings"].append(f"{s.site} {w} {code}: {bad[code]}")
        if bad:
            judges_w = [(c, r, i and c not in bad, bad.get(c, t)) for c, r, i, t in judges_w]
            included = [(c, r) for c, r, i, _ in judges_w if i]
    l20 = _l20_values(s, w, judges_w) if included else {}
    for row, pid in lag_items:
        info = lagtable.product_info(pid)
        text, prod, pst = _status_from_plan(root, row, pid, store)
        if prod is not None:
            collect["cost"].append((s.site, w, pid, row.get("method", ""), prod, pst))
        validated, vflags, vtext = validation(root, s.site, pid, prod)
        collect["validated"][f"{s.site}|{pid}"] = bool(validated)
        if not validated and vtext:
            collect["validation_text"][f"{s.site}|{w}|{pid}"] = vtext
        lag_kind = (prod or {}).get("lag_kind") or info.get("lag_kind", "linear")
        info = dict(info, lag_kind=lag_kind)
        inst, inst_err = None, None
        if text is None and prod.get("kind") in ("field", "band", "groups"):
            try:
                inst = store.instance(prod)
            except Exception as exc:
                inst_err = f"NOT AVAILABLE (instance: {type(exc).__name__}: {exc})"
        if text is None and inst_err is None and inst is not None:
            try:
                _collect_bands(collect, s, w, pid, prod, inst, lag_kind)
            except Exception as exc:
                collect["warnings"].append(f"{s.site} {w} {pid}: band values not collected "
                                           f"({type(exc).__name__}: {exc})")
        if not included:
            if not judges_w:
                collect["coherence_status"].append((s.site, w, pid, text or inst_err or "ok",
                                                    bool(validated)))
            for code, rec, inc, jtext in judges_w:
                base = _base_row(s, w, code, rec, pid, info, row)
                add(_status_rows(base, jtext))
            store.drop((prod or {}).get("dir"))
            continue
        for code, rec, inc, jtext in judges_w:
            base = _base_row(s, w, code, rec, pid, info, row)
            if not inc:
                add(_status_rows(base, jtext, validated=validated))
                continue
            if text is not None or inst_err is not None:
                add(_status_rows(base, text or inst_err, (prod or {}).get("reason"), validated))
                continue
            try:
                rr, dd = _score_at_judge(scorer, s, pid, prod, inst, store, code, rec, base,
                                         lag_kind, validated, vflags, l20)
            except Exception as exc:          # recorded on the row, never fatal
                msg = f"NOT AVAILABLE (scoring error: {type(exc).__name__}: {exc})"
                collect["warnings"].append(f"{s.site} {w} {pid} {code}: {msg}")
                rr, dd = _status_rows(base, msg, None, validated), None
            add(rr, dd)
        store.drop((prod or {}).get("dir"))
    if scorer is not None:
        collect["warnings"].extend(scorer.warnings)
    return rows, draws


def _score_at_judge(scorer, s, pid, prod, inst, store, code, rec, base, lag_kind, validated,
                    vflags, l20):
    """``(rows, draws)`` of one scorable product at one included judge."""
    rows, draws = [], []

    def add(rr, dd=None):
        rows.extend(rr)
        draws.extend(dd if dd is not None else [None] * len(rr))

    kind = prod.get("kind")
    if pid == "L20":
        loc = rec.get("location") or judges.location_of(code)
        taus = {T: l20[T].get(loc, NAN) for T in TRUTHS}
        sc = scorer.score(code, taus, "linear")
        add(*_point_rows(base, sc, {"step": "point"}, validated, vflags,
                         extra_flags=["reference, never ranked"]))
    elif kind in ("field", "band"):
        fld = prod.get("field")
        band = prod.get("band")
        if fld is None and band is not None:
            fld = lc.broadcast_band(band["tau"], inst["band_of"])
        geom = store.geom(inst, rec, s.grid)
        be = None if band is None else band.get("edge")
        key = (prod.get("dir"), "band_meds")
        if key not in store.cache:
            store.cache[key] = rd.band_medians(fld, inst["band_of"], len(inst["centers"]),
                                               lag_kind)
        ro = rd.readout(fld, inst["keep"], inst["band_of"], inst["s_km"], inst["centers"], rec,
                        s.grid, lag_kind, band_edge=be, geom=geom, band_meds=store.cache[key])
        iv = interval_at_readout(prod, ro, lag_kind, inst, store)
        sc = scorer.score(code, ro["tau"], lag_kind, interval=iv)
        info_ro = {"step": ro["step"], "n_px": ro["n_px"],
                   "edge": bool(np.isfinite(ro["edge_share"]) and
                                ro["edge_share"] > rd.EDGE_SHARE_FLAG),
                   "tau_band": ro["tau_band"], "tau_band_interp": ro["tau_band_interp"],
                   "tau_step1": ro["tau_step1"], "flags": ro["flags"]}
        add(*_point_rows(base, sc, info_ro, validated, vflags))
    elif kind == "point":
        point = prod.get("point") or {}
        pe = point.get(code) if isinstance(point, Mapping) else None
        if pe is None:
            add(_status_rows(base, "NOT AVAILABLE (no value at this gauge)", None, validated))
        else:
            taus = {T: rd.point_readout(point, code, T) for T in TRUTHS}
            kinds = point_kinds(pid, lag_kind)
            taus = point_wrap(taus, kinds)
            sc = scorer.score(code, taus, kinds)
            add(*_point_rows(base, sc, {"step": "point"}, validated, vflags,
                             extra_flags=list(pe.get("flags") or [])))
            add(*_point_extras(scorer, base, pid, pe, code, lag_kind, validated, vflags))
    elif kind == "groups":
        add(_status_rows(dict(base), "ok (descriptive per-group rows)", None, validated))
        geom = store.geom(inst, rec, s.grid)
        gro = rd.group_readouts(prod.get("groups") or {}, inst["keep"], inst["band_of"],
                                inst["s_km"], inst["centers"], rec, s.grid, geom=geom,
                                lag_kind=lag_kind)
        for gid, g in gro.items():
            name = gid if str(gid).startswith(("orbit", "phase")) else f"orbit{gid}"
            gb = dict(base, product=f"{pid}@{name}")
            if g.get("status") != "ok" or not np.isfinite(g.get("tau", NAN)):
                add(_status_rows(gb, f"NOT COMPUTABLE ({g.get('status')}: {g.get('reason')})",
                                 None, validated))
                continue
            sc = scorer.score(code, g["tau"], lag_kind)
            info_ro = {"step": g["step"], "n_px": g["n_px"], "edge": False,
                       "tau_band": g["tau_band"], "tau_band_interp": g["tau_band_interp"],
                       "tau_step1": g["tau_step1"], "flags": g["flags"]}
            add(*_point_rows(gb, sc, info_ro, validated, vflags, extra_flags=["descriptive"]))
    else:
        add(_status_rows(base, f"NOT AVAILABLE (unknown result kind {kind!r})", None,
                         validated))
    return rows, draws


def _point_extras(scorer, base, pid, pe, code, lag_kind, validated, vflags):
    """Descriptive rows of a point product: '<pid>@internal' and, for L14,
    'L14@<model>' per secondary model."""
    rows, draws = [], []
    kinds = point_kinds(pid, lag_kind)
    ia, ib = _f(pe.get("internal_a")), _f(pe.get("internal_b"))
    if np.isfinite(ia) or np.isfinite(ib):
        taus = point_wrap({"a": ia, "ab": NAN, "b": ib, "hwlw": NAN}, kinds)
        sc = scorer.score(code, taus, kinds)
        rr, dd = _point_rows(dict(base, product=f"{pid}@internal"), sc, {"step": "point"},
                             validated, vflags, extra_flags=["descriptive"] +
                             list(pe.get("flags") or []))
        rows += rr
        draws += dd
    models = pe.get("models") or {}
    primary = pe.get("model")
    for name, me in models.items():
        if name == primary:
            continue
        b = dict(base, product=f"{pid}@{name}")
        if not me.get("available", False):
            rr = _status_rows(b, f"NOT AVAILABLE ({me.get('reason') or 'model not installed'})",
                              None, validated)
            rows += rr
            draws += [None] * len(rr)
            continue
        taus = point_wrap({T: _f(me.get(T)) for T in TRUTHS}, kinds)
        sc = scorer.score(code, taus, kinds)
        rr, dd = _point_rows(b, sc, {"step": "point"}, validated, vflags,
                             extra_flags=["descriptive"] + list(me.get("flags") or []))
        rows += rr
        draws += dd
    return rows, draws


def _collect_bands(collect, s, w, pid, prod, inst, lag_kind):
    """Band values of a scored product for the truth-free reproducibility and
    coherence tables."""
    K = len(inst["centers"])
    kind = prod.get("kind")
    if kind == "groups":
        gt = {}
        for gid, g in (prod.get("groups") or {}).items():
            if g.get("status", "ok") == "ok" and g.get("tau") is not None \
                    and len(g["tau"]) == K:
                gt[str(gid)] = np.asarray(g["tau"], np.float64)
        collect["groups"][(s.site, w, pid)] = {"centers": inst["centers"], "tau": gt,
                                               "lag_kind": lag_kind}
        return
    band = prod.get("band")
    if band is not None and band.get("tau") is not None and len(band["tau"]) == K:
        tau = np.asarray(band["tau"], np.float64)
        if lag_kind == "phase":
            tau = lc.wrap_phase(tau)
    elif prod.get("field") is not None:
        tau = rd.band_medians(prod["field"], inst["band_of"], K, lag_kind)
    else:
        return
    collect["bands"][(s.site, w, pid)] = {"centers": inst["centers"], "tau": tau,
                                          "lag_kind": lag_kind}


# ─────────────────────────────────────────────────────────────────────────────
#  aggregates
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_COLUMNS = lagtable.SUMMARY_COLUMNS
_LOC_METRICS = ("e", "abs_e", "dE_cm", "dE_all_cm", "LLE_cm", "AGF", "explained", "hitI",
                "tau_m", "tau_true", "s_g_km", "edge_f")


def _location_frame(df):
    """Location-level means of the ok rows of one (site, window, product,
    truth) group, with the indices of their gauges' rows."""
    df = df.assign(edge_f=df["edge"].map(lambda v: float(bool(v)) if v is not None and
                                         not (isinstance(v, float) and np.isnan(v)) else 0.0))
    out = []
    for loc, g in df.groupby("location", sort=True):
        r = {"location": loc, "idx": list(g.index), "steps": list(g["readout_step"])}
        for k in _LOC_METRICS:
            v = pd.to_numeric(g[k], errors="coerce").to_numpy(np.float64) if k in g else \
                np.array([np.nan])
            r[k] = float(np.nanmean(v)) if np.isfinite(v).any() else NAN
        out.append(r)
    return out


def _draw_mean(arrs):
    """Element-wise nanmean of draw arrays (NaN where none is finite)."""
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None
    return _nanmean(np.vstack(arrs), axis=0)


def _loc_draws(loc, draws, key):
    return _draw_mean([None if draws[i] is None else draws[i].get(key) for i in loc["idx"]])


def _site_draws(locs, draws, B):
    """Per-draw site aggregates (MAE, bias, RMS dE, RMS LLE, mean AGF)."""
    out = {}
    e = [_loc_draws(l, draws, "e_a") for l in locs]
    if B and e and all(x is not None for x in e):
        E = np.vstack(e)
        out["MAE"] = np.mean(np.abs(E), axis=0)
        out["bias"] = np.mean(E, axis=0)
    for key, name, how in (("dE", "rms_dE", "rms"), ("LLE", "rms_LLE", "rms"),
                           ("AGF", "mean_AGF", "mean")):
        xs = [_loc_draws(l, draws, key) for l in locs]
        if B and xs and all(x is not None for x in xs):
            X = np.vstack(xs)
            if how == "rms":
                out[name] = np.sqrt(np.mean(X ** 2, axis=0)) * 100.0
            else:
                out[name] = _nanmean(X, axis=0)
    return out


def _ols_slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return NAN
    xm, ym = x[m], y[m]
    vx = np.var(xm)
    return float(np.mean((xm - xm.mean()) * (ym - ym.mean())) / vx) if vx > 0 else NAN


def profile_metrics(tau_m, tau_true, s_g):
    """Profile metrics over a site's locations (>= 4; spec §6.4): OLS slope
    of tau_m on tau_T (unitless), the apparent celerities d tau_m / d s_g and
    d tau_T / d s_g (OLS, min/km: the spec's "apparent celerity, min/km"),
    Lin's CCC, shape error RMS(e - mean e) and the number of decreases of
    tau_m along s_g."""
    x = np.asarray(tau_true, np.float64)
    y = np.asarray(tau_m, np.float64)
    s = np.asarray(s_g, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    out = {"slope": NAN, "ccc": NAN, "shape_err": NAN, "mono_violations": NAN,
           "celerity_m_min_per_km": _ols_slope(s, y), "celerity_true_min_per_km": _ols_slope(s, x)}
    if m.sum() < 2:
        return out
    xm, ym = x[m], y[m]
    vx = np.var(xm)
    out["slope"] = float(np.mean((xm - xm.mean()) * (ym - ym.mean())) / vx) if vx > 0 else NAN
    out["ccc"] = inference.lin_ccc(xm, ym)
    e = ym - xm
    out["shape_err"] = float(np.sqrt(np.mean((e - e.mean()) ** 2)))
    ms = m & np.isfinite(s)
    order = np.argsort(s[ms], kind="stable")
    ys = y[ms][order]
    out["mono_violations"] = int(np.sum(np.diff(ys) < 0)) if len(ys) > 1 else 0
    return out


def _step_shares(steps):
    st = [str(x) for x in steps if x is not None and str(x) != "nan"]
    if not st:
        return ""
    keys = sorted(set(st))
    return ";".join(f"{k}:{st.count(k) / len(st):.2f}" for k in keys)


def _summ_row(key, locs, draws, B, profile=True):
    n = len(locs)
    a = {k: np.array([l[k] for l in locs], np.float64) for k in _LOC_METRICS}
    r = dict(key, n_locations=n)

    def m(x):
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if len(x) else NAN

    ae = a["abs_e"]
    fin = ae[np.isfinite(ae)]
    r.update(MAE=m(ae), median_abs_e=float(np.median(fin)) if len(fin) else NAN,
             bias=m(a["e"]), max_abs_e=float(np.max(fin)) if len(fin) else NAN,
             hit10=float(np.mean(fin <= HIT_MIN)) if len(fin) else NAN, hitI=m(a["hitI"]),
             rms_dE_cm=float(np.sqrt(np.nanmean(a["dE_cm"] ** 2)))
             if np.isfinite(a["dE_cm"]).any() else NAN,
             rms_dE_all_cm=float(np.sqrt(np.nanmean(a["dE_all_cm"] ** 2)))
             if np.isfinite(a["dE_all_cm"]).any() else NAN,
             rms_LLE_cm=float(np.sqrt(np.nanmean(a["LLE_cm"] ** 2)))
             if np.isfinite(a["LLE_cm"]).any() else NAN,
             mean_AGF=m(a["AGF"]), mean_explained=m(a["explained"]),
             edge_share=m(a["edge_f"]),
             step_shares=_step_shares([s for l in locs for s in l["steps"]]))
    sd = _site_draws(locs, draws, B) if n else {}
    for name, col in (("MAE", "MAE"), ("bias", "bias"), ("rms_dE", "rms_dE"),
                      ("rms_LLE", "rms_LLE"), ("mean_AGF", "mean_AGF")):
        lo, hi = inference.percentile_ci(sd[name]) if name in sd else (NAN, NAN)
        if col in ("rms_dE", "rms_LLE"):
            r[f"{col}_lo_cm"], r[f"{col}_hi_cm"] = lo, hi
        else:
            r[f"{col}_lo"], r[f"{col}_hi"] = lo, hi
    pm = profile_metrics(a["tau_m"], a["tau_true"], a["s_g_km"]) if (
        profile and n >= SCORING_V2["profile_min_locations"]) else \
        {"slope": NAN, "ccc": NAN, "shape_err": NAN, "mono_violations": NAN,
         "celerity_m_min_per_km": NAN, "celerity_true_min_per_km": NAN}
    r.update(pm)
    return r, sd


def _sets_for(s, w, dev):
    """judges.judge_sets of a site and window from the truth document."""
    win = {}
    for st in (s.doc or {}).get("stations", []):
        e = st.get("windows", {}).get(w)
        if e is not None:
            win[st["code"]] = {w: e.get("inclusion") or {}}
    table = [dict(r, location=r.get("location") or judges.location_of(r["code"]))
             for r in (s.table or (s.doc or {}).get("stations", []))]
    for r in table:
        win.setdefault(r["code"], {w: {"included": False}})
    return judges.judge_sets(table, win, w, development=dev)


def summarise(points, draws, sets_by_sw, judge_sets, B, *, label=""):
    """lag_summary rows: per (site, window, product, truth, judge set) plus
    the step1_only sensitivity, the pooled rows per window and the 'mixed'
    pooled rows; returns (DataFrame, site draw store)."""
    df = pd.DataFrame(points)
    rows, store = [], {}
    if df.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS)), store
    df = df.reset_index(drop=True)
    keycols = ["site", "window", "product", "truth"]
    meta_cols = ("method", "variant", "family", "fidelity", "lag_kind", "validated")
    for (site, w, pid, T), g in df.groupby(keycols, sort=False, dropna=False):
        meta = {k: g[k].iloc[0] for k in meta_cols if k in g}
        ok = g[g["status"] == "ok"]
        status = "ok" if len(ok) else str(g["status"].iloc[0])
        sets = sets_by_sw.get((site, w)) or {}
        base_key = dict(site=site, window=w, product=pid, truth=T, status=status, label=label,
                        **meta)
        if not len(ok):
            r = dict(base_key, judge_set="headline", n_locations=0)
            rows.append(r)
            continue
        okf = ok[np.isfinite(pd.to_numeric(ok["tau_m"], errors="coerce"))]
        for js in list(judge_sets) + list(SCORING_V2["sensitivity_sets"]):
            locs_allowed = sets.get("headline" if js == "step1_only" else js)
            sub = okf if locs_allowed is None else okf[okf["location"].isin(locs_allowed)]
            if js == "step1_only":
                sub = sub[sub["readout_step"].map(lambda v: str(v) in ("1", "point"))]
            locs = _location_frame(sub) if len(sub) else []
            r, sd = _summ_row(dict(base_key, judge_set=js), locs, draws, B)
            rows.append(r)
            store[(site, w, pid, T, js)] = {"row": r, "draws": sd}
    out = pd.DataFrame(rows)
    return out, store


def pooled_rows(store, windows_by_site, judged_sites, *, label=""):
    """Pooled = the mean of the site values (each site once), per window, and
    'mixed' (each judged site at its descriptive L-P1 window)."""
    rows = []
    metrics = ("MAE", "median_abs_e", "bias", "max_abs_e", "hit10", "hitI", "rms_dE_cm",
               "rms_dE_all_cm", "rms_LLE_cm", "mean_AGF", "mean_explained", "edge_share")
    dmap = {"MAE": ("MAE_lo", "MAE_hi"), "bias": ("bias_lo", "bias_hi"),
            "rms_dE": ("rms_dE_lo_cm", "rms_dE_hi_cm"),
            "rms_LLE": ("rms_LLE_lo_cm", "rms_LLE_hi_cm"),
            "mean_AGF": ("mean_AGF_lo", "mean_AGF_hi")}
    groups = {}
    for (site, w, pid, T, js), v in store.items():
        if site not in judged_sites or not v["row"].get("n_locations"):
            continue
        groups.setdefault((w, pid, T, js), []).append((site, v))
        if windows_by_site.get(site) == w:
            groups.setdefault(("mixed", pid, T, js), []).append((site, v))
    for (w, pid, T, js), items in sorted(groups.items(), key=lambda x: str(x[0])):
        first = items[0][1]["row"]
        r = {k: first.get(k) for k in ("product", "method", "variant", "family", "fidelity",
                                       "lag_kind", "truth", "judge_set")}
        r.update(site="pooled", window=w, status="ok",
                 validated=all(bool(v["row"].get("validated")) for _, v in items),
                 label=(label + ("; " if label else "") +
                        f"pooled over {', '.join(s for s, _ in items)}" +
                        ("; descriptive, mixed windows" if w == "mixed" else "")),
                 n_locations=int(sum(v["row"]["n_locations"] for _, v in items)))
        for k in metrics:
            vals = np.array([_f(v["row"].get(k)) for _, v in items])
            r[k] = float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else NAN
        for dk, (lo_c, hi_c) in dmap.items():
            arrs = [v["draws"].get(dk) for _, v in items]
            if arrs and all(a is not None for a in arrs):
                r[lo_c], r[hi_c] = inference.percentile_ci(_nanmean(np.vstack(arrs), axis=0))
            else:
                r[lo_c], r[hi_c] = NAN, NAN
        r.update(step_shares="", slope=NAN, ccc=NAN, shape_err=NAN, mono_violations=NAN,
                 celerity_m_min_per_km=NAN, celerity_true_min_per_km=NAN)
        rows.append(r)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  family L-P1
# ─────────────────────────────────────────────────────────────────────────────

def lp1_windows(scored, judged_sites):
    """Each judged site's window for the descriptive L-P1: the first of
    INFERENCE_V2 lp1 descriptive_windows scored there (W3 when present)."""
    order = inference.INFERENCE_V2["lp1"]["descriptive_windows"]
    out = {}
    for site in judged_sites:
        have = [w for w in order if (site, w) in scored]
        if have:
            out[site] = have[0]
    return out


def lp1_family(points, scored, *, lock_doc=None, pre_lock=False, sites=None):
    """lag_family rows: L-P1 per truth (confirmatory or descriptive) and the
    pooled pseudo-replicated tests."""
    lp = inference.INFERENCE_V2["lp1"]
    df = pd.DataFrame(points)
    for c in POINTS_COLUMNS:                 # status-only rows carry no metric columns
        if c not in df.columns:
            df[c] = np.nan
    judged = [s for s in lp["judged_sites"] if sites is None or s in sites]
    cols = list(inference.FAMILY_COLUMNS)
    if df.empty or not judged:
        return pd.DataFrame(columns=cols), {}
    w3_all = all((s, "W3") in scored for s in judged)
    confirm = (not pre_lock) and lock_phase_a(lock_doc) and w3_all
    windows = ({s: "W3" for s in judged} if confirm else lp1_windows(scored, judged))
    if not windows:
        return pd.DataFrame(columns=cols), {"confirmatory": False, "windows": {},
                                            "comparisons": [], "claims": {}}
    ok = df[(df["status"] == "ok") & df["site"].isin(list(windows))]
    ok = ok.loc[np.array([windows.get(s) == w for s, w in zip(ok["site"], ok["window"])],
                         dtype=bool)]
    if confirm:
        ok = ok[ok["validated"].astype(bool)]
    comps = list(lp["comparisons"])
    lock_comps = None
    if isinstance(lock_doc, Mapping):
        lock_comps = (lock_doc.get("lp1_comparisons")
                      or (lock_doc.get("families", {}) or {}).get("L-P1"))
    if confirm and isinstance(lock_comps, (list, tuple)) and lock_comps:
        comps = [str(c) for c in lock_comps]
    elif "L16" in comps:
        l16 = ok[(ok["product"] == "L16") & np.isfinite(pd.to_numeric(ok["abs_e"],
                                                                        errors="coerce"))]
        if not len(l16):
            comps.remove("L16")
    label = lp["confirmatory_label"] if confirm else lp["descriptive_label"]
    unval = sorted(set(ok.loc[~ok["validated"].astype(bool), "product"]) & set(comps + ["L01"]))
    if unval and not confirm:
        label += f" (includes rows not validated: {', '.join(unval)})"
    if pre_lock:
        label = SCORING_V2["banner_pre_lock"] + "; " + label
    frames = []
    for T in lp["truths"]:
        frames.append(inference.family_lp1(ok, T, comps, judged, verdict=confirm, label=label))
    fam = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    claims = inference.ranking_claims(fam) if confirm else {}
    for X, c in claims.items():
        if c:
            fam.loc[fam["comparison"] == X, "label"] += f"; ranking under a, ab, b: {c}"
    pooled = [inference.pooled_sign_test(ok, T, X, label=(SCORING_V2["banner_pre_lock"] + "; "
                                                          if pre_lock else "") +
                                         lp["pooled_label"])
              for T in lp["truths"] for X in comps]
    if pooled:
        fam = pd.concat([fam, pd.DataFrame(pooled, columns=cols)], ignore_index=True)
    if len(fam):
        # MAREA reads the imagery at the catalogue instants, the models and zero
        # lag do not (spec header 2026-10-14: labelled, not corrected)
        m = fam["comparison"].isin(_NON_IMAGERY)
        fam.loc[m, "label"] = fam.loc[m, "label"].astype(str) + "; " + LP1_OFFSET_NOTE
    return fam, {"confirmatory": confirm, "windows": windows, "comparisons": comps,
                 "claims": claims}


# ─────────────────────────────────────────────────────────────────────────────
#  reproducibility, coherence, efficiency
# ─────────────────────────────────────────────────────────────────────────────

REPRO_COLUMNS = lagtable.REPRO_COLUMNS


def _pair_stats(vals, lag_kind):
    v = [x for x in vals if np.isfinite(x)]
    if len(v) < 2:
        return NAN
    d = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            d.append(abs(float(lc.circ_diff(v[i], v[j]))) if lag_kind == "phase"
                     else abs(v[i] - v[j]))
    return float(np.median(d))


def repro_tables(collect, has_judges):
    """lag_repro rows (truth-free): years, orbits and coherence."""
    km = SCORING_V2["repro"]["band_km"]
    rows = []
    year_w = {"Y2023": "Y2023", "Y2024": "Y2024", "Y2025": "Y2025"}
    bands = collect["bands"]
    by_sp = {}
    for (site, w, pid), v in bands.items():
        y = year_w.get(w) or ("Y2025" if w == "W1" else None)
        if y is None:
            continue
        cur = by_sp.setdefault((site, pid), {})
        if y not in cur or w == y:
            cur[y] = v
    for (site, pid), per in sorted(by_sp.items()):
        if len(per) < 2:
            continue
        ys = sorted(per)
        lag_kind = per[ys[0]]["lag_kind"]
        bins = {}
        for y in ys:
            c, t = per[y]["centers"], per[y]["tau"]
            for k in range(len(c)):
                if np.isfinite(c[k]):
                    b = int(math.floor(c[k] / km))
                    bins.setdefault(b, {})[y] = (float(c[k]), float(t[k]))
        Y = []
        for b in sorted(bins):
            vals = [bins[b][y][1] if y in bins[b] else NAN for y in ys]
            cen = float(np.mean([bins[b][y][0] for y in ys if y in bins[b]]))
            Y.append(vals)
            rows.append({"site": site, "window": "+".join(ys), "kind": "years", "product": pid,
                         "band": b, "centre_km": cen, "n_groups": int(np.isfinite(vals).sum()),
                         "groups": ";".join(ys),
                         "values": ";".join(f"{v:.2f}" for v in vals),
                         "median_abs_diff": _pair_stats(vals, lag_kind), "icc21": NAN,
                         "spread_sd": NAN, "spread_range": NAN, "label": "truth-free"})
        rows.append({"site": site, "window": "+".join(ys), "kind": "years", "product": pid,
                     "band": "all", "centre_km": NAN, "n_groups": len(ys), "groups": ";".join(ys),
                     "values": "",
                     "median_abs_diff": float(np.nanmedian([r["median_abs_diff"] for r in rows
                                                           if r["site"] == site and
                                                           r["product"] == pid and
                                                           r["kind"] == "years" and
                                                           r["band"] != "all"] or [NAN])),
                     "icc21": inference.icc21(np.array(Y)) if Y else NAN,
                     "spread_sd": NAN, "spread_range": NAN, "label": "truth-free; ICC(2,1) "
                     "across calendar years over bands"})
    for (site, w, pid), v in sorted(collect["groups"].items()):
        gt = v["tau"]
        if len(gt) < 2:
            continue
        gids = sorted(gt)
        M = np.vstack([gt[g] for g in gids]).T          # bands x groups
        for k in range(M.shape[0]):
            rows.append({"site": site, "window": w, "kind": "orbits", "product": pid, "band": k,
                         "centre_km": float(v["centers"][k]),
                         "n_groups": int(np.isfinite(M[k]).sum()), "groups": ";".join(gids),
                         "values": ";".join(f"{x:.2f}" for x in M[k]),
                         "median_abs_diff": _pair_stats(M[k], v["lag_kind"]), "icc21": NAN,
                         "spread_sd": NAN, "spread_range": NAN, "label": "truth-free"})
        rows.append({"site": site, "window": w, "kind": "orbits", "product": pid, "band": "all",
                     "centre_km": NAN, "n_groups": len(gids), "groups": ";".join(gids),
                     "values": "",
                     "median_abs_diff": float(np.nanmedian([_pair_stats(M[k], v["lag_kind"])
                                                            for k in range(M.shape[0])]))
                     if M.shape[0] else NAN,
                     "icc21": inference.icc21(M), "spread_sd": NAN, "spread_range": NAN,
                     "label": "truth-free; ICC(2,1) across relative orbits over bands"})
    by_sw = {}
    for (site, w, pid), v in bands.items():
        if has_judges.get((site, w)):
            continue
        by_sw.setdefault((site, w), {})[pid] = v
    for (site, w), per in sorted(by_sw.items()):
        pids = sorted(per)
        K = max(len(per[p]["tau"]) for p in pids)
        cen = per[pids[0]]["centers"]
        for k in range(K):
            vals = {p: (float(per[p]["tau"][k]) if k < len(per[p]["tau"]) else NAN)
                    for p in pids}
            fin = np.array([x for x in vals.values() if np.isfinite(x)])
            rows.append({"site": site, "window": w, "kind": "coherence", "product": "all",
                         "band": k, "centre_km": float(cen[k]) if k < len(cen) else NAN,
                         "n_groups": int(len(fin)), "groups": ";".join(pids),
                         "values": ";".join(f"{p}={vals[p]:.2f}" for p in pids),
                         "median_abs_diff": NAN, "icc21": NAN,
                         "spread_sd": float(np.std(fin)) if len(fin) > 1 else NAN,
                         "spread_range": float(np.ptp(fin)) if len(fin) > 1 else NAN,
                         "label": "lag coherence only"})
    return pd.DataFrame(rows, columns=list(REPRO_COLUMNS))


def sensing_offset(root, site, cubes_root=None):
    """The site's median datatake-to-site offset (min) from the OPTIONAL
    ``diag time`` output (``<root>/sensing/<site>/sensing_times.json``, else
    the copy beside the stores), or None when neither exists. Descriptive
    only (spec header 2026-10-14: nothing reads it otherwise)."""
    paths = [os.path.join(root, "sensing", site, "sensing_times.json")]
    try:
        from experiments import zarr_cubes as zc
        paths.append(str(zc.site_dir(site, cubes_root) / "sensing_times.json"))
    except Exception:
        pass
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            doc = product.read_json(p)
        except (OSError, ValueError):
            continue
        v = [_f(e.get("minus_start_min")) for e in (doc.get("dates") or {}).values()
             if isinstance(e, Mapping) and e.get("instant")]
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v):
            return {"median_min": float(np.median(v)), "n": int(len(v)),
                    "range_min": [float(v.min()), float(v.max())],
                    "path": p.replace("\\", "/")}
    return None


def offset_lines(points, root, sites, cubes_root=None):
    """Per scored (site, window): the median signed e (truth a, ok rows of
    the base products) over the imagery rows and over the lag estimates that
    read no imagery (:data:`_NON_IMAGERY_ESTIMATES`: L14, L16, L18), and the
    site's sensing offset when the diagnostic exists
    (:func:`sensing_offset`). Descriptive, for the attribution step."""
    df = pd.DataFrame(points)
    out = []
    if df.empty or "e" not in df.columns:
        return out
    ok = df[(df["status"] == "ok") & (df["truth"] == "a")
            & ~df["product"].astype(str).str.contains("@")]
    sens = {s: sensing_offset(root, s, cubes_root) for s in sites}
    for (site, w), g in ok.groupby(["site", "window"], sort=False):
        e = pd.to_numeric(g["e"], errors="coerce")
        im = e[g["product"].isin(_IMAGERY)].to_numpy(np.float64)
        ni = e[g["product"].isin(_NON_IMAGERY_ESTIMATES)].to_numpy(np.float64)
        im, ni = im[np.isfinite(im)], ni[np.isfinite(ni)]
        out.append({"site": site, "window": w,
                    "imagery_median_e": float(np.median(im)) if len(im) else NAN,
                    "n_imagery": int(len(im)),
                    "non_imagery_median_e": float(np.median(ni)) if len(ni) else NAN,
                    "n_non_imagery": int(len(ni)), "sensing": sens.get(site)})
    return out


COST_COLUMNS = lagtable.COST_COLUMNS


def cost_table(collect, points):
    """lag_cost rows: medians over instances of wall, CPU and peak RSS of
    the method's own stages (``lag.<method>.*``; contended production runs;
    the products of one unit share them), the unit's total wall time, the
    NOT COMPUTABLE share and the share of zero fallbacks among the
    product's read-outs."""
    per = {}
    for site, w, pid, method, prod, st in collect["cost"]:
        t = prod.get("timings") or {}
        stages = {str(k): v for k, v in t.items() if isinstance(v, Mapping)}
        own = [v for k, v in stages.items() if k.startswith("lag.")]
        use = own or list(stages.values())
        ut = prod.get("unit_total") or {}
        if use:
            wall = float(np.nansum([_f(v.get("wall_s")) for v in use]))
            cpu = float(np.nansum([_f(v.get("cpu_self_s")) for v in use]))
            rss = [_f(v.get("peak_rss_mb")) for v in use]
            rss = float(np.nanmax(rss)) if np.isfinite(rss).any() else NAN
            anon = [_f(v.get("peak_anon_mb")) for v in use]
            anon = float(np.nanmax(anon)) if np.isfinite(anon).any() else NAN
        else:
            wall, cpu, rss = _f(ut.get("wall_s")), _f(ut.get("cpu_self_s")), \
                _f(ut.get("peak_rss_mb"))
            anon = _f(ut.get("peak_anon_mb"))
        if not np.isfinite(anon):
            anon = _f((prod.get("memory") or {}).get("peak_anon_mb"))
        e = per.setdefault(pid, {"method": method, "wall": [], "cpu": [], "rss": [],
                                 "anon": [], "unit": [], "n": 0, "nc": 0})
        e["wall"].append(wall)
        e["cpu"].append(cpu)
        e["rss"].append(rss)
        e["anon"].append(anon)
        e["unit"].append(_f(ut.get("wall_s")))
        e["n"] += 1
        e["nc"] += int(st in ("not_computable", "empty"))
    df = pd.DataFrame(points)
    rows = []
    for pid, e in sorted(per.items()):
        f0 = NAN
        if not df.empty:
            sub = df[(df["product"] == pid) & (df["status"] == "ok") & (df["truth"] == "a")]
            st = sub["readout_step"].map(str)
            if len(st):
                f0 = float(np.mean(st == "3"))

        def med(x):
            x = np.asarray(x, float)
            return float(np.nanmedian(x)) if np.isfinite(x).any() else NAN

        rows.append({"product": pid, "method": e["method"], "n_instances": e["n"],
                     "wall_s_median": med(e["wall"]), "cpu_self_s_median": med(e["cpu"]),
                     "peak_rss_mb_median": med(e["rss"]),
                     "peak_anon_mb_median": med(e["anon"]),
                     "unit_wall_s_median": med(e["unit"]),
                     "share_not_computable": e["nc"] / e["n"] if e["n"] else NAN,
                     "share_fallback0": f0,
                     "label": "contended (production runs); method stages lag.<method>.*, "
                              "shared by the products of a unit"})
    return pd.DataFrame(rows, columns=list(COST_COLUMNS))


# ─────────────────────────────────────────────────────────────────────────────
#  score --part lag
# ─────────────────────────────────────────────────────────────────────────────

def score_lag(root=V2_ROOT, sites=None, windows=None, pre_lock=False, B=None,
              judge_sets=("headline", "without_mouth", "development_free"), cubes_root=None, *,
              packages=None, truth_docs=None, judge_tables=None, series_by_code=None,
              lock=None, out_dir=None, show_unvalidated=False, write=True, res=10,
              unblind=(), blind_windows=None, invocation=None):
    """``score --part lag``: every lag row of the lag plan scored against the
    truths, aggregated, tested and written (scoring spec §0-§9).

    Parameters
    ----------
    root : str
        The validation v2 root (plans/, runs/, truth/, control/, gates/).
    sites, windows : sequences, optional
        Restrict the scoring (default every site and window of the plan).
    pre_lock : bool
        Score before LOCK.json phase A: every output carries the banner and
        no confirmatory verdict is issued. Without it, a missing phase-A lock
        is refused.
    B : int, optional
        Truth-side bootstrap draws (default 1000).
    judge_sets : sequence
        The judge sets of the aggregates (judges.judge_sets).
    cubes_root : str, optional
        Passed to the package loader.
    packages, truth_docs, judge_tables, series_by_code : dict, optional
        Inject per-site packages, truth documents, judge tables (records with
        ``nbhd``) and per-code gauge series instead of reading them (tests).
    lock : dict, optional
        A LOCK.json document (default ``<root>/LOCK.json``).
    show_unvalidated : bool
        Print the numbers of rows that are not validated in lag_table.md
        (a development view; the CSVs always keep them).
    res : int
        The grid of the packages read (10: the Zarr packages; 20: the local
        legacy 20 m packages, ``package.v2_site(res=20)``).
    unblind : sequence of window ids
        Pre-lock on real 10 m products (``packages`` not injected): the
        confirmatory windows (:data:`PRELOCK_BLIND_WINDOWS`, W3) the user
        decided to score now; the others stay BLINDED (status rows, nothing
        read). Every such call is appended to ``scores/prelock_ledger.json``.
    blind_windows : sequence, optional
        Override the blinded windows (tests).
    invocation : str, optional
        The command line, recorded in the pre-lock ledger.

    Outputs go to ``out_dir``, by default ``<root>/scores``, or
    ``<root>/scores/partial_<sites>__<windows>`` when ``sites`` or
    ``windows`` restrict the call (the full tables are never replaced by a
    subset).

    Returns
    -------
    dict
        ``{"points", "summary", "family", "repro", "cost", "table_md"}``
        (file paths; ``"ledger"`` for a pre-lock real-data call),
        ``"warnings"`` and ``"data"`` (the DataFrames).
    """
    lock_doc = read_lock(root) if lock is None else lock
    if not pre_lock and not lock_phase_a(lock_doc):
        raise RuntimeError("score --part lag needs LOCK.json phase A (run 'lock --phase A'), "
                           "or --pre-lock for the descriptive pre-lock tables")
    B = int(inference.INFERENCE_V2["boot"]["B_descriptive"] if B is None else B)
    plan = read_plan(root)
    warnings_ = []
    if plan is None:
        raise FileNotFoundError(f"{root}/plans/lag.csv is missing: run 'plan --design lag'")
    if sites:
        plan = [r for r in plan if r["site"] in set(sites)]
    if windows:
        plan = [r for r in plan if r["window"] in set(windows)]
    order = {w: i for i, w in enumerate(("W3", "W2", "W1", "Y2025", "Y2024", "Y2023"))}
    site_list = list(dict.fromkeys(r["site"] for r in plan))
    real = packages is None and int(res or 10) == 10
    unblind = tuple(str(w) for w in (unblind or ()))
    if blind_windows is None:
        blind_windows = (tuple(w for w in PRELOCK_BLIND_WINDOWS if w not in unblind)
                         if (pre_lock and real) else ())
    blind_windows = tuple(blind_windows)
    store = _Store(root)
    collect = {"bands": {}, "groups": {}, "cost": [], "warnings": warnings_,
               "coherence_status": [], "validated": {}, "validation_text": {}}
    points, draws = [], []
    sets_by_sw, scored, has_judges, site_notes = {}, set(), {}, {}
    blinded = []
    dev_extra = lock_development(lock_doc)
    for site in site_list:
        s = _site_context(root, site, cubes_root=cubes_root, packages=packages,
                          truth_docs=truth_docs, judge_tables=judge_tables, dev_extra=dev_extra,
                          res=res)
        site_notes[site] = {"package": s.pkg_error, "truth": s.doc_error, "judges": s.table_error,
                            "years": s.years}
        wins = sorted({r["window"] for r in plan if r["site"] == site},
                      key=lambda w: (order.get(w, 99), w))
        for w in wins:
            prows = [r for r in plan if r["site"] == site and r["window"] == w]
            if w in blind_windows:
                log(f"[score lag] {site} {w}: BLINDED before lock-A ({len(prows)} plan rows)")
                points.extend(_blinded_rows(s, w, prows))
                draws.extend([None] * (len(points) - len(draws)))
                blinded.append((site, w))
                continue
            log(f"[score lag] {site} {w}: {len(prows)} plan rows")
            rr, dd = _score_window(root, s, w, prows, store, B=B, series_by_code=series_by_code,
                                   collect=collect)
            store.clear_window()
            points.extend(rr)
            draws.extend(dd)
            if _window_guard(s, w) is None:
                jw = _judges_at(s, w) if (s.doc is not None or s.table) else []
                inc = any(i for _, _, i, _ in jw)
                has_judges[(site, w)] = inc
                if inc:
                    scored.add((site, w))
                    sets_by_sw[(site, w)] = _sets_for(s, w, s.dev)
        if s.pkg is not None and packages is None and hasattr(s.pkg, "close"):
            s.pkg.close()
    label = SCORING_V2["banner_pre_lock"] if pre_lock else ""
    pts = pd.DataFrame(points)
    for c in POINTS_COLUMNS:
        if c not in pts.columns:
            pts[c] = np.nan
    summ, sstore = summarise(points, draws, sets_by_sw, judge_sets, B, label=label)
    judged = inference.INFERENCE_V2["lp1"]["judged_sites"]
    lpw = lp1_windows(scored, judged)
    pooled = pooled_rows(sstore, lpw, judged, label=label)
    if pooled:
        summ = pd.concat([summ, pd.DataFrame(pooled)], ignore_index=True)
    for c in SUMMARY_COLUMNS:
        if c not in summ.columns:
            summ[c] = np.nan
    fam, fam_info = lp1_family(points, scored, lock_doc=lock_doc, pre_lock=pre_lock,
                               sites=site_list)
    repro = repro_tables(collect, has_judges)
    cost = cost_table(collect, points)
    context = {"pre_lock": pre_lock, "B": B, "lp1": fam_info, "lp1_windows": lpw,
               "site_notes": site_notes, "coherence_status": collect["coherence_status"],
               "warnings": warnings_, "show_unvalidated": show_unvalidated,
               "validated": collect["validated"],
               "validation_text": collect["validation_text"],
               "lock_phase_a": lock_phase_a(lock_doc),
               "blinded": [{"site": a, "window": b} for a, b in blinded],
               "unblinded": list(unblind) if (pre_lock and real) else [],
               "offset_lines": offset_lines(points, root, site_list, cubes_root),
               "sensing_offset_note": SENSING_OFFSET_NOTE,
               "deferred": list(DEFERRED_ITEMS)}
    paths = {}
    if write:
        if out_dir is None:
            out_dir = os.path.join(root, "scores")
            if sites or windows:
                out_dir = os.path.join(out_dir, partial_name(sites, windows))
        paths = lagtable.write(pts[list(POINTS_COLUMNS)], summ[list(SUMMARY_COLUMNS)], fam,
                               repro, plan, out_dir, pre_lock, cost=cost, context=context)
    if pre_lock and real:
        # every pre-lock look at real 10 m lag scores is disclosed, written or not
        paths["ledger"] = append_prelock_ledger(
            root, out_dir=out_dir if write else None, points=pts, sites=site_list,
            windows=windows, blinded=blinded, unblind=unblind, invocation=invocation,
            points_path=paths.get("points"))
    out = dict(paths)
    out["warnings"] = warnings_
    out["data"] = {"points": pts[list(POINTS_COLUMNS)], "summary": summ[list(SUMMARY_COLUMNS)],
                   "family": fam, "repro": repro, "cost": cost, "context": context}
    return out


def partial_name(sites, windows):
    """``partial_<sites>__<windows>`` of a subset score (``all`` for an
    unrestricted part)."""
    def part(v):
        return "+".join(str(x) for x in v) if v else "all"
    return f"partial_{part(sites)}__{part(windows)}"


def _blinded_rows(s, w, prows):
    """One status row per (product, truth) of a blinded window: no truth,
    no product and no package is read for it."""
    text = (f"BLINDED (pre-lock: {w} is the confirmatory window of L-P1 and is scored after "
            f"lock-A; --unblind {w} scores it now, as the user's decision, recorded in "
            "scores/prelock_ledger.json and LOCK.json)")
    out = []
    for row in prows:
        for pid in str(row.get("products", "")).split():
            if not pid.startswith("L"):
                continue
            info = lagtable.product_info(pid)
            base = _base_row(s, w, "", {}, pid, info, row)
            out.extend(_status_rows(base, text))
    return out


def prelock_ledger_path(root):
    return os.path.join(root, "scores", PRELOCK_LEDGER)


def read_prelock_ledger(root):
    """The pre-lock ledger's entries (``[]`` when there is none)."""
    p = prelock_ledger_path(root)
    if not os.path.exists(p):
        return []
    try:
        doc = product.read_json(p)
    except (OSError, ValueError):
        return []
    return list((doc or {}).get("entries") or [])


def append_prelock_ledger(root, *, out_dir, points, sites, windows, blinded, unblind,
                          invocation=None, points_path=None):
    """Append one pre-lock score of real 10 m products to
    ``<root>/scores/prelock_ledger.json`` (spec §9.9 forking paths): when,
    which code, which sites and windows were scored (and which blinded or
    unblinded), the products with scored rows and the sha256 of the points
    written. lock-A copies every entry into LOCK.json's ``forking_paths``."""
    df = pd.DataFrame(points)
    scored = {}
    prods = []
    if not df.empty:
        ok = df[df["status"] == "ok"]
        for (site, w), g in ok.groupby(["site", "window"], sort=False):
            scored.setdefault(str(site), []).append(str(w))
        prods = sorted({str(p) for p in ok["product"]})
    entry = {"utc": product.utc_stamp(), "git": product.git_state(),
             "invocation": invocation,
             "out_dir": None if out_dir is None else str(out_dir).replace("\\", "/"),
             "sites": list(sites), "windows_requested": list(windows) if windows else "all",
             "windows_scored": scored, "blinded": [{"site": a, "window": b} for a, b in blinded],
             "unblinded": list(unblind), "products_scored": prods,
             "points_sha256": (product.sha256_file(points_path)
                               if points_path and os.path.exists(points_path) else None)}
    p = prelock_ledger_path(root)
    entries = read_prelock_ledger(root)
    entries.append(entry)
    product.write_json(p, {"format": PRELOCK_LEDGER_FORMAT, "entries": entries})
    return p.replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
#  positive control (§10.2)
# ─────────────────────────────────────────────────────────────────────────────

def _expected_offset(pid, offsets=None):
    """``(offset_min, source)``: the theory offset of a product in the gating
    world (control.expected_offset, else the card's control_offset_min, else
    0 assumed and flagged)."""
    if offsets is not None:
        if callable(offsets):
            return float(offsets(pid)), "given"
        if pid in offsets:
            return float(offsets[pid]), "given"
    try:
        from experiments.benchmark import control
        return float(control.expected_offset(pid)), "control.expected_offset"
    except Exception:
        pass
    try:
        from experiments.benchmark import lags
        return float(lags.card(pid)["control_offset_min"]), "card"
    except Exception:
        return 0.0, "assumed 0 (no card or control module)"


def control_verdicts(products, planted, tol_min=None, *, band_of=None, domain=None,
                     offsets=None, manual_fault=None):
    """The positive-control verdict of each lag product on the control
    instance (§10.2, scoring spec §8).

    Parameters
    ----------
    products : dict {product_id: LagResult} or list of LagResults
        Results of the adapters on the control instance (lags.common
        lag_result shape; ``band`` with ``tau``, or a ``field`` on keep).
    planted : dict
        ``control.planted_band_lags``: {"bands", "tau_mean_min", "n_px"}.
    tol_min : float
        Default FROZEN positive_control.tolerances.lag_max_min (10 min).
    band_of / domain :
        The control instance's band per keep pixel (``domain.band_of``),
        needed for field products without a band payload.
    offsets : mapping or callable, optional
        Expected offsets (default ``control.expected_offset``).
    manual_fault : dict {product_id: reason}, optional
        Documented implementation faults: verdict "FAIL (fault)".

    Returns
    -------
    dict {product_id: {"verdict", "median_abs_err_min", "n_bands", "per_band",
    "offset_min", "offset_source", "reason"}}
    """
    tol = float(SCORING_V2["control"]["tol_min"] if tol_min is None else tol_min)
    if band_of is None and domain is not None:
        band_of = getattr(domain, "band_of", None)
        if band_of is None and isinstance(domain, Mapping):
            band_of = domain.get("band_of")
    bands = [int(b) for b in planted["bands"]]
    ptau = np.asarray(planted["tau_mean_min"], np.float64)
    pn = list(planted.get("n_px", [None] * len(bands)))
    items = products.items() if isinstance(products, Mapping) else \
        ((r["product"], r) for r in products)
    out = {}
    for pid, res in items:
        off, src = _expected_offset(pid, offsets)
        rec = {"verdict": None, "median_abs_err_min": NAN, "n_bands": 0, "per_band": [],
               "offset_min": off, "offset_source": src, "reason": None}
        if manual_fault and pid in manual_fault:
            rec.update(verdict="FAIL (fault)", reason=str(manual_fault[pid]))
            out[pid] = rec
            continue
        st = res.get("status", "ok")
        if st != "ok":
            rec.update(verdict="control mismatch",
                       reason=f"{st} on the control: {res.get('stage')}: {res.get('reason')}")
            out[pid] = rec
            continue
        lag_kind = res.get("lag_kind") or "linear"
        kind = res.get("kind")
        tau = None
        band = res.get("band")
        if isinstance(band, Mapping) and band.get("tau") is not None:
            tau = np.asarray(band["tau"], np.float64)
            if lag_kind == "phase":
                tau = lc.wrap_phase(tau)
        elif res.get("field") is not None:
            bo = res.get("band_of", band_of)
            if bo is None:
                raise ValueError(f"control_verdicts: {pid} is a field product; pass band_of "
                                 "(or domain) of the control instance")
            K = (int(np.max(bands)) + 1) if bands else 0
            K = max(K, int(np.max(np.asarray(bo))) + 1 if len(np.asarray(bo)) else 0)
            tau = rd.band_medians(res["field"], bo, K, lag_kind)
        elif kind == "groups" and res.get("groups"):
            G = [np.asarray(g["tau"], np.float64) for g in res["groups"].values()
                 if g.get("status", "ok") == "ok"]
            if G:
                tau = _nanmedian(np.vstack(G), axis=0)
        if tau is None:
            rec.update(verdict="not applicable", reason=f"no band read-out for a {kind} product")
            out[pid] = rec
            continue
        errs = []
        for i, b in enumerate(bands):
            r = float(tau[b]) if 0 <= b < len(tau) else NAN
            if lag_kind == "phase":
                err = float(lc.circ_diff(r - off, ptau[i])) if np.isfinite(r) else NAN
            else:
                err = r - ptau[i] - off
            errs.append(err)
            rec["per_band"].append({"band": b, "readout": r, "planted": float(ptau[i]),
                                    "offset": off, "err": err, "n_px": pn[i]})
        e = np.abs(np.asarray(errs, np.float64))
        e = e[np.isfinite(e)]
        rec["n_bands"] = int(len(e))
        if not len(e):
            rec.update(verdict="control mismatch", reason="no band read-out is defined")
        else:
            med = float(np.median(e))
            rec["median_abs_err_min"] = med
            rec["verdict"] = "PASS" if med <= tol else "control mismatch"
        out[pid] = rec
    return out


def control_overall(by_scenario):
    """The per-product verdict over the scenarios (§10.2): PASS only when
    PASS in every scenario given ('profile' and 'zero'); 'FAIL (fault)' when
    any scenario has it; else 'control mismatch'.

    Parameters
    ----------
    by_scenario : dict {scenario: control_verdicts output}
    """
    pids = sorted({p for v in by_scenario.values() for p in v})
    out = {}
    for pid in pids:
        vs = {sc: (v.get(pid) or {}).get("verdict") for sc, v in by_scenario.items()}
        if any(str(x).startswith("FAIL") for x in vs.values()):
            verdict = "FAIL (fault)"
        elif all(x == "PASS" for x in vs.values()) and \
                set(SCORING_V2["control"]["scenarios"]) <= set(vs):
            verdict = "PASS"
        elif all(x == "not applicable" for x in vs.values()):
            verdict = "not applicable"
        else:
            verdict = "control mismatch"
        out[pid] = {"verdict": verdict, "scenarios": vs}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  G-lagtruth (§10.1)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticTide:
    """A smooth boundary tide (M2, S2, O1 at their exact periods) evaluated
    elementwise from nanoseconds; answers in time order like the production
    providers."""

    name = "synthetic tide (G-lagtruth)"

    @staticmethod
    def value(ns):
        x = np.asarray(ns, np.int64) / 3.6e12
        return (1.6 * np.cos(2 * np.pi * x / 12.4206012 + 0.3)
                + 0.45 * np.cos(2 * np.pi * x / 12.0 + 1.1)
                + 0.2 * np.cos(2 * np.pi * x / 25.8193417 - 0.4))

    def levels(self, times):
        ns = np.sort(pd.DatetimeIndex(times).as_unit("ns").asi8)
        return self.value(ns)


def synthetic_package(site, years=(2024,), step_days=2, seed=7, grid=None):
    """A package stand-in on a synthetic boundary tide: record instants every
    ``step_days`` at 10:50 + U(0, 10) min UTC, the exact level table
    (tau -90..240), the 1-min series over the years +- 2 days and the 10-min
    series of the years (``truth.series1``/``series10``)."""
    from experiments.benchmark import levels

    years = tuple(int(y) for y in years)
    rng = np.random.default_rng(seed)
    days = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq=f"{int(step_days)}D")
    off = rng.uniform(0, 600, len(days)).round()
    t = pd.DatetimeIndex(days + pd.Timedelta(hours=10, minutes=50)
                         + pd.to_timedelta(off, unit="s")).as_unit("ns")
    tide = SyntheticTide()
    shifts = np.arange(-90, 241, dtype=np.int64)
    H = np.vstack([tide.value(t.asi8 - int(s) * 60_000_000_000) for s in shifts])
    tab = levels.LevelTable(shifts, H, t, provider=tide)
    g1 = pd.date_range(pd.Timestamp(f"{years[0]}-01-01") - pd.Timedelta(days=2),
                       pd.Timestamp(f"{years[-1] + 1}-01-01") + pd.Timedelta(days=2),
                       freq="1min", inclusive="left").as_unit("ns")
    t_min = levels.minutes_since_epoch(g1)
    h1 = tide.value(g1.asi8)
    in_y = (g1 >= pd.Timestamp(f"{years[0]}-01-01")) & \
        (g1 < pd.Timestamp(f"{years[-1] + 1}-01-01"))
    sel10 = in_y & (g1.minute % 10 == 0)
    pkg = types.SimpleNamespace(
        cfg=types.SimpleNamespace(site=site, truth_dir=None), t=t, T=len(t),
        years=list(years), levels=tab, series1=(t_min, h1),
        series10=(g1[sel10], h1[sel10]), h0=tab.h0, keys={"package_key": "synthetic"},
        tide=tide)
    if grid is not None:
        pkg.grid = grid
    return pkg


def planted_gauge(pkg, wid, lag_min=75, noise_m=0.02, seed=0):
    """The planted gauge of G-lagtruth: h_B(t - lag) + N(0, noise) at the
    window's record instants and on its 10-min grid (one series, both sets
    of stamps)."""
    t_all = truth.record_t(pkg)
    pos = truth.window_positions(t_all, wid)
    rng = np.random.default_rng(seed)
    hv = np.asarray(pkg.levels.at(int(lag_min)), np.float64)[pos]
    idx10, _ = truth.series10(pkg)
    w = settings.window(wid)
    m10 = (idx10 >= pd.Timestamp(w.start)) & (idx10 < pd.Timestamp(w.end))
    idx_w = idx10[m10]
    t1, h1 = truth.series1(pkg)
    p = np.searchsorted(t1, levels_minutes(idx_w)) - int(lag_min)
    ok = (p >= 0) & (p < len(h1))
    h10 = np.full(len(idx_w), np.nan)
    h10[ok] = h1[p[ok]]
    s = pd.concat([pd.Series(hv + rng.normal(0, noise_m, len(hv)), index=t_all[pos]),
                   pd.Series(h10 + rng.normal(0, noise_m, len(h10)), index=idx_w)])
    s = s[np.isfinite(s.to_numpy())].sort_index(kind="stable")
    return s[~s.index.duplicated(keep="first")]


def _nearest_cell_check(seed=8):
    """truth.sample_cells against v1.sample_truth on seeded synthetic sheets
    (the nearest-cell lookup of G-lagtruth): ``(equal, detail)``."""
    cwd = os.getcwd()
    try:
        from experiments import v1_vaklodingen_validate as v1
    except Exception as exc:
        return False, f"v1 not importable: {type(exc).__name__}: {exc}"
    finally:
        os.chdir(cwd)
    rng = np.random.default_rng(seed)

    def sheet(x0, y0, nx, ny, step, nan_share, year):
        x = x0 + step * np.arange(nx)
        y = y0 - step * np.arange(ny)
        z = rng.normal(size=(ny, nx))
        z[rng.random(z.shape) < nan_share] = np.nan
        yr = np.full((ny, nx), year)
        yr[:, nx // 2:] = year + 2
        return x, y, z, yr

    cells = {"A": sheet(0.0, 1000.0, 30, 25, 20.0, 0.3, 2021),
             "B": sheet(300.0, 900.0, 30, 30, 20.0, 0.0, 2019)}
    xq = rng.uniform(-50, 950, 2000)
    yq = rng.uniform(250, 1050, 2000)
    z, yr, *_ = truth.sample_cells(cells, xq, yq)
    z1, yr1 = v1.sample_truth(cells, xq, yq)
    eq = bool(np.array_equal(z, z1, equal_nan=True) and np.array_equal(yr, yr1))
    return eq, f"{int(np.isfinite(z).sum())} of {len(z)} query points matched; equal = {eq}"


def gate_lagtruth(site, root=V2_ROOT, *, cubes_root=None, pkg=None, window=None,
                  planted_min=None, tol_min=None, noise_m=None, constituents=None,
                  write=True, synthetic=None, **kw):
    """Gate G-lagtruth (§10.1): a planted lag is recovered by the truth
    machinery, and the nearest-cell truth lookup equals v1's.

    The planted gauge is h_B(t - 75 min) + 2 cm noise at the site's record
    instants and on the 10-min grid of the window (provider RWS, centred
    10-min stamps); ``truth.lag_truths_window`` must give tau_a and tau_b
    within 75 +- 1 min. The package of the site is used when it exists (its
    level table, 1-min and 10-min series), else a synthetic one
    (:func:`synthetic_package`); ``synthetic=True`` forces it.

    Returns
    -------
    dict
        The verdict document (plan INTERFACES §10): format, gate, site,
        verdict (PASS / FAIL / NOT RUN), reason, package_key, compared,
        first_difference, utc, git; written to ``gates/<site>/lagtruth.json``
        when ``write``.
    """
    g = SCORING_V2["gate_lagtruth"]
    planted = float(g["planted_min"] if planted_min is None else planted_min)
    tol = float(g["tol_min"] if tol_min is None else tol_min)
    noise = float(g["noise_m"] if noise_m is None else noise_m)
    source = "package"
    err = None
    if pkg is None and not synthetic:
        try:
            from experiments.benchmark import package
            pkg = package.V2Package(package.v2_site(site, root=root, cubes_root=cubes_root))
            _ = pkg.levels
            _ = pkg.series1
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            pkg = None
    if pkg is None:
        pkg = synthetic_package(site)
        source = "synthetic" + (f" (no package: {err})" if err else "")
    years = [int(y) for y in getattr(pkg, "years", [])]
    wid = window
    if wid is None:
        for cand in g["windows"]:
            if all(y in years for y in settings.window(cand).years):
                wid = cand
                break
    doc = {"format": GATE_FORMAT, "gate": "lagtruth", "site": site, "verdict": "NOT RUN",
           "reason": None, "package_key": (getattr(pkg, "keys", None) or {}).get("package_key"),
           "compared": {"planted_min": planted, "tol_min": tol, "noise_m": noise,
                        "source": source, "window": wid},
           "first_difference": None, "utc": product.utc_stamp(), "git": None}
    try:
        doc["git"] = product.git_state()
    except Exception:
        doc["git"] = None
    miss = [] if wid is None else [y for y in settings.window(wid).years if y not in years]
    if wid is None:
        doc["reason"] = f"no window of {g['windows']} has all its years in {years}"
    elif miss:
        doc["reason"] = f"window {wid}: store years {miss} missing (window-year guard)"
    else:
        seed = [int(config.SEEDS["truth_boot"]), int(config.SITE_ID[site]), int(planted)]
        series = planted_gauge(pkg, wid, planted, noise, seed)
        rec = {"code": f"planted{int(planted)}", "provider": g["provider"],
               "sampling_min": g["sampling_min"]}
        r, _ = truth.lag_truths_window(pkg, rec, series, wid, constituents=constituents,
                                       boot=False)
        ta, tb = _f(r["a"]["tau"]), _f(r["b"]["tau"])
        ok_a = np.isfinite(ta) and abs(ta - planted) <= tol
        ok_b = np.isfinite(tb) and abs(tb - planted) <= tol
        cell_ok, cell_txt = _nearest_cell_check()
        doc["compared"].update(tau_a=ta, tau_b=tb, tau_ab=_f(r["ab"]["tau"]),
                               tau_all=_f(r["all"].get("tau")),
                               n_instants=r["a"].get("n_instants"),
                               nearest_cell=cell_txt)
        diffs = []
        if not ok_a:
            diffs.append(f"tau_a = {ta} (planted {planted} +- {tol})")
        if not ok_b:
            diffs.append(f"tau_b = {tb} (planted {planted} +- {tol})")
        if not cell_ok:
            diffs.append(f"nearest-cell lookup: {cell_txt}")
        doc["verdict"] = "PASS" if not diffs else "FAIL"
        doc["first_difference"] = diffs[0] if diffs else None
        doc["reason"] = ("planted lag recovered by (a) and (b); nearest-cell lookup equal"
                         if not diffs else "; ".join(diffs))
    if write:
        product.write_json(os.path.join(root, "gates", site, "lagtruth.json"), doc)
    return doc
