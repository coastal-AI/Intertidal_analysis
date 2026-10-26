# -*- coding: utf-8 -*-
"""Validation v2 — the E0 unit: no delay (E0a), MAREA (E0b), its lag (L01)
and the gauge-lag attribution row (E0g).

Question
--------
What elevation does MAREA's inversion give on an instance at tau = 0 (no
delay), at each pixel's archive-estimated lag (MAREA), and at the lag the
in-estuary gauges measured (the attribution row), with the result bit for
bit what ``marea.reconstruct`` gives on the same inputs?

Rules (docs/validation_v2_spec.md header 2026-10-15, §3.2, §3.3.1, §5.1;
v13 §3.2; FROZEN['marea'])
-------------------------------------------------------------------------
* One unit, several products from ONE fit (§5.1): E0a, E0b and L01 share
  the lag stage of :func:`experiments.benchmark.lags.marea_family.lag_stage`
  (memoised per instance).
* Inversion = ``marea.invert_series(Y, C, h, lo, hi, min_bracket=3)``
  (FROZEN marea min_bracket, = ``marea.QUALITY_MIN_BRACKET``) with
  Y = nan_to_num(Y, 0) (float32, converted to float64 chunk by chunk inside
  invert_series, the same values as production's float64 copy), C = V
  (bool, likewise), lo/hi = min/max of h0 over the record, h = h0 at tau = 0
  and the exact level row ``inst.levels.at(tau)`` otherwise (reconstruct's
  exact shifted level).
* Work per unique tau (E0a: one group, every keep pixel in keep order),
  over that tau's columns in ascending order, in slices of FROZEN marea
  'slice' (200,000) columns, a multiple of invert_series' chunk (4,000):
  every chunk then holds exactly the columns it holds in one call on the
  group, so the result is bitwise production's (test with slice 8, chunk 4).
* Record floor 40 scenes and 'no_intertidal' raise NotComputable for every
  product (v13 D4); 'no_anchor' gives tau = 0 everywhere (E0b = E0a).
* Variants: '' and 'boot30' (opts n_boot: L01's interval95), 'seed<k>'
  (opts seed_offset: the column draw only), 'gaugelag' (E0g: band lags from
  ``lags.references.gauge_band_taus`` at each band's mean s, smoothed with
  ``te.smooth_lag_profile`` and rounded exactly as MAREA's own; NOT
  COMPUTABLE when the gauges give no lag), 'nodelay' (level sources, V3:
  NotComputable).
* ``opts['production_compat']`` (gates only): the clear mask is the pixel
  cloud rule alone, as production's extraction (see marea_family).

No import-time side effects.
"""
from __future__ import annotations

import contextlib
import inspect
import os
import re

import numpy as np

from experiments.benchmark import config, product
from experiments.benchmark.lags import common
from experiments.benchmark.lags import marea_family as mf
from experiments.benchmark.product import NotComputable
from pyintertidal import marea

_M = config.FROZEN["marea"]
#: the per-pixel quality rule (bracket of 3 clear observations each side)
MIN_BRACKET = int(_M["min_bracket"])
#: columns per invert_series call (a multiple of CHUNK)
SLICE = int(_M["slice"])
del _M
#: invert_series' own chunk (its default; passed explicitly so tests may
#: shrink it together with SLICE)
CHUNK = int(inspect.signature(marea.invert_series).parameters["chunk"].default)
#: invert_series' frozen guards (its defaults, recorded in CONFIG)
_INV = {k: inspect.signature(marea.invert_series).parameters[k].default
        for k in ("mu_points", "min_obs", "min_b", "max_b", "max_a")}

if MIN_BRACKET != int(marea.QUALITY_MIN_BRACKET):
    raise AssertionError(f"FROZEN min_bracket {MIN_BRACKET} differs from marea."
                         f"QUALITY_MIN_BRACKET {marea.QUALITY_MIN_BRACKET}")

CODE_DEPS = tuple(mf.CODE_DEPS) + ("experiments/benchmark/lags/marea_family.py",)
REFERENCES_PY = "experiments/benchmark/lags/references.py"

_get = mf._get


def _stage(inst, name):
    fn = getattr(inst, "stage", None)
    return fn(name) if callable(fn) else contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
#  the inversion
# ─────────────────────────────────────────────────────────────────────────────

def invert_columns(inst, cols, h, lo, hi, production_compat=False):
    """``marea.invert_series(min_bracket=3)`` on keep positions ``cols``
    (one tau group, ascending) in slices of SLICE columns.

    Returns z, sigma (float64, NaN where the guards or the bracket reject)
    and the clear count of each column (int64).
    """
    if SLICE <= 0 or CHUNK <= 0 or SLICE % CHUNK:
        raise ValueError(f"slice {SLICE} must be a positive multiple of invert_series' chunk "
                         f"{CHUNK}, or the chunk partition (hence the bits) would change")
    cols = np.asarray(cols, np.int64)
    n = len(cols)
    z = np.full(n, np.nan)
    sg = np.full(n, np.nan)
    nclr = np.zeros(n, np.int64)
    h = np.asarray(h, np.float64)
    for a in range(0, n, SLICE):
        b = min(a + SLICE, n)
        arr = mf.read_columns(inst, cols[a:b], ("Y", "clear"), production_compat)
        z[a:b], sg[a:b] = marea.invert_series(arr["Y"], arr["clear"], h, lo, hi, chunk=CHUNK,
                                              min_bracket=MIN_BRACKET)
        nclr[a:b] = arr["clear"].sum(axis=0)
        del arr
        mf._progress(inst)
    return z, sg, nclr


def invert_by_tau(inst, tau_px, h0, lo, hi, production_compat=False):
    """reconstruct's inversion loop: per unique applied lag (ascending), its
    columns in keep order, h = h0 at 0 and the exact level row otherwise."""
    tau_px = np.asarray(tau_px, np.float64)
    n = len(tau_px)
    z = np.full(n, np.nan)
    sg = np.full(n, np.nan)
    nclr = np.zeros(n, np.int64)
    groups = {}
    for tv in np.unique(tau_px):
        cols = np.where(tau_px == tv)[0]
        if not len(cols):
            continue
        if tv != np.floor(tv):
            raise ValueError(f"applied lags are whole minutes; got {tv}")
        h = h0 if not tv else np.asarray(inst.levels.at(int(tv)), np.float64)
        z[cols], sg[cols], nclr[cols] = invert_columns(inst, cols, h, lo, hi, production_compat)
        groups[str(int(tv))] = int(len(cols))
    return z, sg, nclr, groups


def compute(inst, n_boot=0, boot_threads=1, seed_offset=0, production_compat=False,
            products=("E0a", "E0b", "L01")):
    """Every array of the E0 unit (full precision), for the run and the
    gates: ``z_a``/``sigma_a`` (E0a), ``lag`` (the lag stage), ``z_b``/
    ``sigma_b`` (E0b), ``clear_obs``, ``lo``, ``hi``, ``T``, ``groups``."""
    T = mf._floor(inst)
    d = mf._domain(inst)
    h0 = np.asarray(inst.h0, np.float64)
    if h0.shape != (T,):
        raise ValueError(f"inst.h0 has shape {h0.shape}, not ({T},)")
    lo, hi = float(h0.min()), float(h0.max())
    n = int(len(d.keep))
    out = {"T": T, "lo": lo, "hi": hi, "n_keep": n, "domain_status": d.status}
    if "E0a" in products:
        with _stage(inst, "invert_nodelay"):
            z, sg, nclr = invert_columns(inst, np.arange(n, dtype=np.int64), h0, lo, hi,
                                         production_compat)
        out.update(z_a=z, sigma_a=sg, clear_obs=nclr)
    if "E0b" in products or "L01" in products:
        out["lag"] = mf.lag_stage(inst, n_boot=n_boot, boot_threads=boot_threads,
                                  seed_offset=seed_offset, production_compat=production_compat)
    if "E0b" in products:
        with _stage(inst, "invert_marea"):
            z, sg, nclr, groups = invert_by_tau(inst, out["lag"]["tau_px"], h0, lo, hi,
                                                production_compat)
        out.update(z_b=z, sigma_b=sg, groups=groups)
        out.setdefault("clear_obs", nclr)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  products
# ─────────────────────────────────────────────────────────────────────────────

def _elev(pid, z, sigma, clear_obs, tau_px, diag):
    z32 = np.asarray(z, np.float64).astype(np.float32)
    status = "ok" if np.isfinite(z32).any() else "empty"
    return common.elev_result(pid, status, n_keep=len(z32), z=z32,
                              sigma=np.asarray(sigma, np.float64).astype(np.float32),
                              clear_obs=np.asarray(clear_obs, np.int64).astype(np.int16),
                              tau_px=np.asarray(tau_px, np.float32),
                              reason=None if status == "ok" else "no pixel passes the guards",
                              diag=diag)


def _seed_offset(opts, parts):
    v = _get(opts, "seed_offset", None)
    if v is not None:
        return int(v)
    for p in parts:
        m = re.fullmatch(r"seed(\d+)", p)
        if m:
            return int(m.group(1))
    return 0


def _parse_gauge(res):
    """``(status, reason, tau_band, extras)`` of gauge_band_taus' answer (a
    mapping like ``levels.gauge_band_lags``, or a bare array)."""
    if isinstance(res, dict) or hasattr(res, "get"):
        st = str(res.get("status", "ok"))
        tau = res.get("tau_band")
        extras = {k: v for k, v in res.items() if k not in ("tau_band",)}
        return st, res.get("reason"), tau, extras
    return "ok", None, res, {}


def run_gaugelag(inst, production_compat=False):
    """E0g: MAREA's inversion with the judges' measured band lags."""
    T = mf._floor(inst)
    d = mf._domain(inst)
    K = int(d.K)
    if d.status == "no_anchor" or K < 2:
        raise NotComputable("no_anchor", "one band: the gauge lags cannot be placed along s",
                            {"domain_status": d.status, "K": K})
    band_s = mf.band_mean_s(d.s_km, d.band_of, K)
    from experiments.benchmark.lags import references
    res = references.gauge_band_taus(inst, inst.window, band_s_km=band_s)
    status, reason, tau_band, extras = _parse_gauge(res)
    if status != "ok" or tau_band is None:
        raise NotComputable("gauge_lags", reason or "not available",
                            {"band_s_km": band_s.tolist(), "status": status})
    tau_band = np.asarray(tau_band, np.float64)
    if tau_band.shape != (K,) or not np.isfinite(tau_band).all():
        raise NotComputable("gauge_lags", "the gauges give no finite lag for every band",
                            {"tau_band": tau_band.tolist(), "band_s_km": band_s.tolist()})
    tau_px = mf.applied_lag(d.centers, tau_band, d.s_km)
    h0 = np.asarray(inst.h0, np.float64)
    lo, hi = float(h0.min()), float(h0.max())
    with _stage(inst, "invert_gaugelag"):
        z, sg, nclr, groups = invert_by_tau(inst, tau_px, h0, lo, hi, production_compat)
    diag = {"tau_band": tau_band, "band_s_km": band_s, "centers_km": np.asarray(d.centers),
            "T": T, "lo_m": lo, "hi_m": hi, "groups": groups, "label":
            "uses in-estuary gauges (attribution row; never in lag scoring)",
            "gauge": {k: v for k, v in extras.items()
                      if isinstance(v, (str, int, float, bool, list, dict, type(None)))}}
    return _elev("E0g", z, sg, nclr, tau_px.astype(np.float32), diag)


def run_unit(inst, unit, opts):
    """The E0 unit: one result per product of ``unit['products']``, in order
    (E0a, E0b, L01; or E0g for the variant 'gaugelag')."""
    variant = str(_get(unit, "variant", "") or "")
    parts = [p for p in variant.split("+") if p]
    if "nodelay" in parts:
        raise NotComputable("level_sources", "level sources are V3 (the E0 'nodelay' rows under "
                            "LT, LM and LG run with the levels design)")
    products = tuple(_get(unit, "products", None) or
                     (("E0g",) if "gaugelag" in parts else ("E0a", "E0b", "L01")))
    compat = bool(_get(opts, "production_compat", False))
    if "gaugelag" in parts:
        if products != ("E0g",):
            raise ValueError(f"the 'gaugelag' unit writes E0g only, not {products}")
        return [run_gaugelag(inst, compat)]
    unknown = [p for p in products if p not in ("E0a", "E0b", "L01")]
    if unknown:
        raise ValueError(f"E0 writes E0a, E0b and L01, not {unknown}")
    n_boot = int(_get(opts, "n_boot", 0) or 0)
    seed_offset = _seed_offset(opts, parts)
    boot_threads = int(_get(opts, "boot_threads", 1) or 1)
    res = compute(inst, n_boot, boot_threads, seed_offset, compat, products)
    base = {"T": res["T"], "lo_m": res["lo"], "hi_m": res["hi"], "min_bracket": MIN_BRACKET,
            "domain_status": res["domain_status"], "production_compat": compat}
    out = []
    for p in products:
        if p == "E0a":
            out.append(_elev("E0a", res["z_a"], res["sigma_a"], res["clear_obs"],
                             np.zeros(res["n_keep"], np.float32), dict(base, tau_min=0.0)))
        elif p == "E0b":
            st = res["lag"]
            out.append(_elev("E0b", res["z_b"], res["sigma_b"], res["clear_obs"], st["tau_px"],
                             dict(base, groups=res["groups"],
                                  tau_used=np.asarray(st["tau_used"], float),
                                  flags=list(st["flags"]))))
        else:
            out.append(mf.l01_result(res["lag"]))
    return out


def external_inputs(inst, unit):
    """What E0g reads besides the package: the gauge-lag code and the lag
    truths of the window (its inputs_key changes with them); ``{}`` for
    every other variant."""
    parts = [p for p in str(_get(unit, "variant", "") or "").split("+") if p]
    if "gaugelag" not in parts:
        return {}
    out = {}
    ref = os.path.join(config.ROOT, REFERENCES_PY)
    out[REFERENCES_PY] = config.sha256_source(ref) if os.path.exists(ref) else None
    tdir = _get(getattr(inst, "paths", None), "truth_dir", None)
    for name in ("truth_lag_v2.json", "judges.json"):
        p = os.path.join(tdir, name) if tdir else None
        out[f"truth/{name}"] = (product.sha256_file(p) if p and os.path.exists(p) else None)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  configuration and cards
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "module": "methods.elev_marea_pair",
    "invert_series": {"min_bracket": MIN_BRACKET, "chunk": CHUNK, **{k: _INV[k] for k in _INV}},
    "slice": SLICE,
    "levels": "h = inst.h0 at tau = 0, the exact row inst.levels.at(tau) otherwise",
    "lo_hi": "min and max of h0 over the record",
    "observation": "Y = nan_to_num(Y, 0), C = V (production_compat: C = the pixel cloud rule)",
    "lag": "lags.marea_family.lag_stage (CONFIG of lags.marea_family)",
    "gaugelag": ("lags.references.gauge_band_taus at the band mean s; te.smooth_lag_profile "
                 "3 km; rounded to the minute"),
    "marea_family": mf.CONFIG,
}

_COMMON = mf._COMMON
_REF_INV = [{"cite": ("MAREA (this study): pyintertidal.marea.invert_series, the closed-form "
                      "sigmoid inversion with the per-pixel quality rule (min_bracket 3)"),
             "doi_or_url": "repository: pyintertidal/marea.py, docs/METHOD.md"}]
_QC_INV = {"min_bracket": MIN_BRACKET, "min_obs": _INV["min_obs"],
           "b": [_INV["min_b"], _INV["max_b"]], "abs_a_max": _INV["max_a"],
           "mu_points": _INV["mu_points"], "sg_grid": list(marea.SG_GRID),
           "floor_scenes": mf.FLOOR}
_SIG = mf._signature("E0")
_COST = mf._cost("E0")
_VARIANTS = {"boot30": "30-draw bootstrap of the lag (L01 interval95)",
             "seed<k>": "column draw seed + 1000 k (seed-variance floor)",
             "nodelay": "no delay under LT, LM, LG (levels design, V3)",
             "gaugelag": "E0g: the judges' measured band lags (attribution row)"}

CARDS = {
    "E0a": {
        "id": "E0a", "name": "No delay: MAREA's inversion at tau = 0", "family": "elevation",
        "kind": "elev", "tier": "1", "fidelity": "F0",
        "evidence": ("unit tests: equal to one invert_series(min_bracket=3) call (slice 8, "
                     "chunk 4); G-marea: the production comparison folder's no-delay map"),
        "references": _REF_INV, "output": "elev", "lag_kind": None, "signature": _SIG,
        "scan_window": None, "published_window": None,
        "as_published": {"setting": "C2.raw.ndwi.D1", "window": "W3", "design": "published"},
        "native_cell": {"setting": "C2.raw.ndwi.D1", "window": "W3"},
        "common_conditions": _COMMON, "deviations": [mf._DEV_V], "qc": dict(_QC_INV),
        "n_min": mf.FLOOR, "tuning": [], "control_offset_min": 0.0, "cost": _COST,
        "variants": dict(_VARIANTS),
    },
    "E0b": {
        "id": "E0b", "name": "MAREA: inversion at each pixel's applied (archive) lag",
        "family": "elevation", "kind": "elev", "tier": "1", "fidelity": "F0",
        "evidence": ("G-marea: bitwise reproduction of marea.reconstruct's z on the legacy "
                     "inputs; unit tests per tau group (slice 8, chunk 4)"),
        "references": _REF_INV + mf._REF_MAREA, "output": "elev", "lag_kind": None,
        "signature": _SIG, "scan_window": None, "published_window": None,
        "as_published": {"setting": "C2.raw.ndwi.D1", "window": "W3", "design": "published"},
        "native_cell": {"setting": "C2.raw.ndwi.D1", "window": "W3"},
        "common_conditions": _COMMON,
        "deviations": [mf._DEV_V, mf._DEV_ONEBAND, mf._DEV_BOOT],
        "qc": dict(_QC_INV, lag="L01 (lags.marea_family.lag_stage)"),
        "n_min": mf.FLOOR, "tuning": [], "control_offset_min": 0.0, "cost": _COST,
        "variants": dict(_VARIANTS),
    },
    "L01": mf.CARDS["L01"],
    "E0g": {
        "id": "E0g", "name": "MAREA's inversion with the judges' measured band lags (attribution)",
        "family": "elevation", "kind": "elev", "tier": "1", "fidelity": "F0",
        "evidence": "unit tests: the E0b inversion code with the gauge band lags",
        "references": _REF_INV + [{"cite": ("levels.GAUGE_LAG_RULE / lags.references."
                                            "gauge_band_taus (user decision 2026-10-15)"),
                                   "doi_or_url": "docs/validation_v2_spec.md header"}],
        "output": "elev", "lag_kind": None, "signature": _SIG, "scan_window": None,
        "published_window": None, "as_published": None, "native_cell": None,
        "common_conditions": _COMMON,
        "deviations": [mf._DEV_V,
                       {"paper": "MAREA's band lags from the archive (m2a)",
                        "ours": ("truth (a) of the included judges at the window, isotonic, "
                                 "tau(0) = 0, linear in s, read at each band's mean s"),
                        "why": "attribution: the cost of estimating the lag from the archive",
                        "rows": "levels design, sites with judges",
                        "sensitivity": "labelled 'uses in-estuary gauges'; never ranked"}],
        "qc": dict(_QC_INV, lags="lags.references.gauge_band_taus; not available -> NOT "
                                 "COMPUTABLE"),
        "n_min": mf.FLOOR, "tuning": [], "control_offset_min": 0.0, "cost": _COST,
        "variants": {"gaugelag": "the only variant that writes E0g"},
    },
}
