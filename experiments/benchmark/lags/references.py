# -*- coding: utf-8 -*-
"""Validation v2 — L20 (gauge-network leave-one-out reference) and the E0g gauge band lags.

Question
--------
How well would the OTHER gauges of the estuary predict a gauge's lag
(L20, a reference that is never ranked, spec §3.3 row L20)? And which band
lags does the attribution row E0g use: MAREA's inversion with the
judges' MEASURED lags instead of the archive estimate (user decision of
2026-10-15)?

Pre-registered rules
--------------------
L20 (spec §3.3 row L20, §3.3.1 "L20-L22", plan scoring spec §1):

* the value at a location is the leave-one-out linear interpolation in s
  of the OTHER included locations' truth T of the same window, anchored at
  tau = 0 at s = 0 and held constant beyond the last location. It needs the
  truths themselves, so the SCORER computes it (package D); this unit
  writes a placeholder point product (status ok, ``point = {}``) whose
  ``diag['computed_by']`` says so, so the plan's row exists and the run
  queue sees it done. It is labelled, never ranked and never used to tune
  anything.

E0g band lags (:func:`gauge_band_taus`; ``levels.GAUGE_LAG_RULE``):

* truth (a), tau_a, of every judge INCLUDED in the unit's window
  (``truth/<site>/truth_lag_v2.json``, ``truth.load_lag_truth_v2``) at its
  s_g (``truth/<site>/judges.json``), placed along s by
  ``levels.gauge_band_lags``: tau(0) = 0 (MAREA's anchor), the Terneuzen
  pair (``config.SAME_LOCATION``) one knot, a monotone (isotonic) fit,
  linear in s, constant beyond the last gauge; a band's lag is the curve at
  the band's representative s, by default the mean s_km of its keep
  pixels. ``truth_key`` 'ab' or 'b' are the declared sensitivities.
* It is an input of the elevation row E0g only (labelled "uses in-estuary
  gauges"); never in lag scoring nor in a confirmatory family.
"""
from __future__ import annotations

import os

import numpy as np

from experiments.benchmark import config, judges, levels, truth
from experiments.benchmark.lags import common, models

NAN = float("nan")
#: where each truth key's lag sits in a truth_lag_v2 window entry
TRUTH_PATHS = {"a": ("a", "tau"), "ab": ("ab", "tau"), "b": ("b", "tau"),
               "all": ("all", "tau"), "hwlw": ("hw_lw", "mean_min")}

CONFIG = {
    "L20": {"rule": ("leave-one-out linear interpolation in s of the other included "
                     "locations' truth (same window), tau(0) = 0, constant beyond the last "
                     "location; computed by the scorer (needs the truths); this unit writes "
                     "a placeholder"),
            "computed_by": "scorer (leave-one-out over the truth nodes)"},
    "gauge_band_taus": {"rule": "levels.GAUGE_LAG_RULE", "default_truth": "a",
                        "band_s": "the mean s_km of each band's keep pixels",
                        "truth_paths": {k: list(v) for k, v in TRUTH_PATHS.items()}},
}

CARDS = {
    "L20": {
        "id": "L20", "name": "REF: gauge-network leave-one-out interpolation in s",
        "family": "reference", "kind": "lag", "tier": "1", "fidelity": "-",
        "evidence": "computed by the scorer from the truth document (package D)",
        "references": [{"cite": "v2 spec §3.3 row L20, §3.3.1 'L20-L22'",
                        "doi_or_url": "docs/validation_v2_spec.md"}],
        "output": "point", "lag_kind": "linear",
        "signature": {"reads": ["window"], "domain": False},
        "scan_window": None, "published_window": None,
        "as_published": "the interpolation a tide-table user would make from the other gauges",
        "native_cell": None,
        "common_conditions": "the window's truths of the other included locations",
        "deviations": [], "qc": {"never_ranked": True, "control": "not applicable (truths)"},
        "n_min": 0, "tuning": [], "control_offset_min": 0.0,
        "cost": {"wh_cell": 0.01, "ram_gb": 0.5}, "variants": {},
    },
}

CODE_DEPS = ("experiments/benchmark/truth.py", "experiments/benchmark/judges.py",
             "experiments/benchmark/levels.py", "experiments/benchmark/lags/common.py",
             "experiments/benchmark/lags/models.py")


# ─────────────────────────────────────────────────────────────────────────────
#  L20
# ─────────────────────────────────────────────────────────────────────────────

def run_l20(inst, unit, opts=None):
    """Unit L20: the placeholder product (the scorer computes the values)."""
    r = common.lag_result("L20", "ok", kind="point", lag_kind="linear", point={},
                          diag={"computed_by": CONFIG["L20"]["computed_by"],
                                "window": str(inst.window), "rule": CONFIG["L20"]["rule"]})
    return [r for _ in models.unit_products(unit, ("L20",))]


def external_inputs(inst, unit=None):
    """The truth document and the judge table (what the scorer reads)."""
    td = (getattr(inst, "paths", None) or {}).get("truth_dir")
    return {"truth_lag_v2.json": models.file_sha(os.path.join(td, "truth_lag_v2.json"))
            if td else None, "judges.json": models.judges_sha(inst)}


# ─────────────────────────────────────────────────────────────────────────────
#  the E0g band lags
# ─────────────────────────────────────────────────────────────────────────────

def band_mean_s(domain):
    """The representative s of each band: the mean s_km of its keep pixels
    (NaN for a band without one)."""
    K = int(domain.K)
    s = np.asarray(domain.s_km, np.float64)
    b = np.asarray(domain.band_of, np.int64)
    out = np.full(K, np.nan)
    for k in range(K):
        m = (b == k) & np.isfinite(s)
        if m.any():
            out[k] = float(np.mean(s[m]))
    return out


def _truth_value(entry, truth_key):
    path = TRUTH_PATHS.get(truth_key)
    if path is None:
        raise ValueError(f"unknown truth key {truth_key!r}; expected one of {sorted(TRUTH_PATHS)}")
    v = entry.get(path[0], {})
    v = v.get(path[1]) if isinstance(v, dict) else None
    return NAN if v is None else float(v)


def gauge_band_taus(inst, window, band_s_km=None, truth_key="a"):
    """The band lags of the attribution row E0g (``levels.gauge_band_lags``).

    Parameters
    ----------
    inst : LagInstance-like
        ``paths['truth_dir']`` and ``domain`` (keep's s_km, band_of, K).
    window : str
        The unit's window: the judges INCLUDED in it and their truth there.
    band_s_km : array (K,), optional
        The representative s of each band (default :func:`band_mean_s`).
    truth_key : str
        'a' (tau_a, the rule) or a declared sensitivity ('ab', 'b').

    Returns
    -------
    dict
        ``levels.gauge_band_lags``'s result (``status``, ``tau_band`` (K,),
        knots, ``dropped``, ...) plus ``window``, ``truth_key``, ``lags_min``,
        ``s_g_km``, ``excluded`` {code: reason}, ``band_s_km`` and
        ``source_shas`` {truth_lag_v2.json, judges.json}.
    """
    td = (getattr(inst, "paths", None) or {}).get("truth_dir")
    if not td:
        raise ValueError("the instance has no truth_dir")
    if truth_key not in TRUTH_PATHS:
        raise ValueError(f"unknown truth key {truth_key!r}; expected one of {sorted(TRUTH_PATHS)}")
    tpath = os.path.join(td, "truth_lag_v2.json")
    jpath = os.path.join(td, "judges.json")
    if band_s_km is None:
        band_s_km = band_mean_s(inst.domain)
    band_s = np.asarray(band_s_km, np.float64)
    shas = {"truth_lag_v2.json": models.file_sha(tpath), "judges.json": models.file_sha(jpath)}
    try:
        doc = truth.load_lag_truth_v2(None, out=td)
    except (OSError, ValueError) as e:            # no lag truth yet: not available, never 0
        return {"status": "not available", "reason": f"no v2 lag truth ({type(e).__name__}: "
                                                      f"{e}): run truth <site>",
                "tau_band": np.full(band_s.shape, np.nan), "codes": [], "knots_s_km": [],
                "raw_tau_min": [], "knots_tau_min": [], "monotone_adjusted": False,
                "dropped": {}, "rule": levels.GAUGE_LAG_RULE, "window": str(window),
                "truth_key": truth_key, "lags_min": {}, "s_g_km": {}, "excluded": {},
                "band_s_km": band_s, "missing_input": tpath.replace("\\", "/"),
                "source_shas": shas}
    s_tab = {}
    if os.path.isfile(jpath):
        for r in judges.read(jpath).get("stations", []):
            s_tab[str(r["code"])] = r.get("s_g_km")
    lags, s_g, excluded = {}, {}, {}
    for st in doc.get("stations", []):
        code = str(st["code"])
        e = (st.get("windows") or {}).get(str(window))
        if e is None:
            excluded[code] = f"no truth entry for {window}"
            continue
        inc = e.get("inclusion") or {}
        if not inc.get("included"):
            excluded[code] = f"excluded in {window}: {inc.get('exclusion')}"
            continue
        lags[code] = _truth_value(e, truth_key)
        s = s_tab.get(code, st.get("s_g_km"))
        s_g[code] = NAN if s is None else float(s)
    res = levels.gauge_band_lags(lags, s_g, band_s)
    res.update(window=str(window), truth_key=truth_key, lags_min=lags, s_g_km=s_g,
               excluded=excluded, band_s_km=band_s, source_shas=shas)
    return res


def describe():
    return {"products": sorted(CARDS), "pairs": [list(p) for p in config.SAME_LOCATION],
            "point_keys": list(common.POINT_VALUE_KEYS)}
