"""Shared constants and helpers of the lag adapters (validation v2, Stage L).

Every lag method of the catalogue (spec §3, §5 design 'lag') returns its
products through :func:`lag_result` (or :func:`elev_result` for E0's
elevation products), so the queue, the scorer and the table writer read one
canonical shape (plan INTERFACES §5):

* ``kind`` -- ``field`` (a per-pixel lag on ``domain.keep``), ``band`` (one lag
  per 3-km band, broadcast to the field), ``point`` (a lag at each judge),
  ``groups`` (band lags per scene group, e.g. per relative orbit), or
  ``elev`` (E0's elevation products);
* ``lag_kind`` -- ``linear`` (a delay in minutes) or ``phase`` (a circular lag
  of period one M2 cycle, :data:`PERIOD_M2_MIN`), which the scorer wraps
  about the truth-free centre :data:`WRAP_CENTRE_MIN` (spec §6.3) and never
  about a truth.

The common lag window (spec §3.3) is :data:`TAU_SCAN`, -45..180 min in 5-min
steps; a scan method reports the sub-grid vertex of its score through
:func:`scan_vertex` (the parabola through the top three points, exactly
``pyintertidal.tide_estimators._parabolic_argmax``) together with the edge
flag (the best grid point is the first or the last one).

The method cards (``CARDS`` of each adapter module) are checked by
:func:`check_card` against :data:`CARD_KEYS` and the allowed values below.
Nothing here reads data; the helpers are pure numpy.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from pyintertidal import tide_estimators as te

TAU_SCAN = np.arange(-45.0, 180.0 + 1e-9, 5.0)      # common lag window (min), 46 values (spec §3.3)
SCAN_LO, SCAN_HI = -45.0, 180.0
PERIOD_M2_H = 12.4206012
PERIOD_M2_MIN = 60.0 * PERIOD_M2_H                   # 745.236072
OMEGA_M2_PER_MIN = 2.0 * np.pi / PERIOD_M2_MIN       # rad/min
MIN_PER_DEG_M2 = 60.0 / 28.9841042                   # == truth.MIN_PER_DEG_M2
WRAP_CENTRE_MIN = 67.5                               # truth-free wrapping centre (§6.3)
RESULT_KINDS = ("field", "band", "point", "groups", "elev")
LAG_KINDS = ("linear", "phase")
RESULT_STATUSES = ("ok", "empty", "not_computable")
CARD_KEYS = ("id", "name", "family", "kind", "tier", "fidelity", "evidence", "references",
             "output", "lag_kind", "signature", "scan_window", "published_window", "as_published",
             "native_cell", "common_conditions", "deviations", "qc", "n_min", "tuning",
             "control_offset_min", "cost", "variants")
FAMILIES = ("likelihood", "order statistic", "self-consistency", "harmonic", "harmonic/response",
            "model selection", "models", "physics", "reference", "in-house variants", "elevation")

#: values a card may take (plan INTERFACES §2)
CARD_KINDS = ("lag", "elev")
LAG_OUTPUTS = ("field", "band", "point", "groups")
TIERS = ("1", "2", "3")
FIDELITIES = ("F0", "F1", "F2", "F3", "F4", "-")
#: interval95 levels and the two nominal recipes of V2 (L01 boot30, L08 refits)
INTERVAL_LEVELS = ("band", "pixel")
INTERVAL_NOMINALS = ("bootstrap30", "iqr20x1.96/0.674",
                     "iqr20x1.96/0.674xsqrt(m/(n-m))")
#: keys of one judge entry of a point product (plan INTERFACES §5)
POINT_VALUE_KEYS = ("a", "ab", "b", "hwlw", "internal_a", "internal_b")
#: tolerance (min) for "a field value sits on the scan bound"
EDGE_TOL_MIN = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
#  scan vertex and circular helpers
# ─────────────────────────────────────────────────────────────────────────────

def scan_vertex(score, grid=TAU_SCAN) -> tuple[float, bool]:
    """Sub-grid argmax of ``score`` over ``grid`` and the edge flag.

    The vertex is ``te._parabolic_argmax(score, grid)`` (parabola through the
    best point and its two neighbours, clipped to one step); the flag is True
    when the best grid point is the first or the last one, i.e. the optimum
    may lie outside the scanned window. Fewer than three finite scores give
    ``(nan, False)``.
    """
    s = np.asarray(score, dtype=float)
    g = np.asarray(grid, dtype=float)
    if s.shape != g.shape:
        raise ValueError(f"scan_vertex: score {s.shape} and grid {g.shape} differ")
    if np.isfinite(s).sum() < 3:
        return float("nan"), False
    j = int(np.nanargmax(s))
    return float(te._parabolic_argmax(s, g)), bool(j == 0 or j == len(g) - 1)


def wrap_phase(tau, period=PERIOD_M2_MIN, centre=WRAP_CENTRE_MIN) -> np.ndarray:
    """Wrap lags into ``[centre - P/2, centre + P/2)`` (NaN stays NaN)."""
    t = np.asarray(tau, dtype=float)
    return centre + np.mod(t - centre + period / 2.0, period) - period / 2.0


def circ_diff(a, b, period=PERIOD_M2_MIN) -> np.ndarray:
    """Circular difference ``a - b`` in ``[-P/2, P/2)``."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.mod(d + period / 2.0, period) - period / 2.0


def circ_median(x, period=PERIOD_M2_MIN, centre=WRAP_CENTRE_MIN) -> float:
    """Median of the finite values after wrapping about ``centre``; NaN if none."""
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(wrap_phase(v, period, centre)))


def broadcast_band(tau_band, band_of) -> np.ndarray:
    """Band lags broadcast to pixels: float32 ``tau_band[band_of]``.

    NaN where ``band_of < 0`` (pixels without a thalweg distance) or where the
    band lag is NaN. A band index beyond ``len(tau_band)`` is a caller bug and
    raises.
    """
    tb = np.asarray(tau_band, dtype=float).ravel()
    bo = np.asarray(band_of)
    if bo.dtype.kind not in "iu":
        raise ValueError(f"broadcast_band: band_of must be integer, got {bo.dtype}")
    bo = bo.astype(np.int64).ravel()
    if bo.size and bo.max(initial=-1) >= tb.size:
        raise ValueError(f"broadcast_band: band index {int(bo.max())} >= K = {tb.size}")
    out = np.full(bo.size, np.nan, dtype=np.float32)
    ok = bo >= 0
    out[ok] = tb[bo[ok]].astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  canonical results
# ─────────────────────────────────────────────────────────────────────────────

def _bad(product, msg):
    return ValueError(f"lag result {product!r}: {msg}")


def _as_array(product, name, value, target, length=None, length_name="n"):
    """Validate and cast one array; ``target`` in {'float32','float64','int64','int16','bool'}.

    Accepted input kinds: floats and integers for a float target; integers for
    an integer target; booleans (or integers that are all 0/1) for a bool
    target. Anything else (strings, objects, complex, bools for a lag) is
    rejected, as is a wrong length or a non-1-D array.
    """
    try:
        a = np.asarray(value)
    except Exception as exc:  # pragma: no cover - exotic inputs
        raise _bad(product, f"{name} is not an array ({exc})") from None
    if a.ndim != 1:
        raise _bad(product, f"{name} must be 1-D, got shape {a.shape}")
    k = a.dtype.kind
    if target in ("float32", "float64"):
        if k not in "fiu":
            raise _bad(product, f"{name} must be a real numeric array, got dtype {a.dtype}")
    elif target in ("int64", "int16"):
        if k not in "iu":
            raise _bad(product, f"{name} must be an integer array, got dtype {a.dtype}")
        if target == "int16" and a.size and (a.min() < np.iinfo(np.int16).min
                                             or a.max() > np.iinfo(np.int16).max):
            raise _bad(product, f"{name} does not fit int16")
    elif target == "bool":
        if k == "b":
            pass
        elif k in "iu" and np.isin(a, (0, 1)).all():
            pass
        else:
            raise _bad(product, f"{name} must be a boolean array, got dtype {a.dtype}")
    else:  # pragma: no cover - programming error
        raise AssertionError(target)
    if length is not None and a.shape[0] != int(length):
        raise _bad(product, f"{name} has length {a.shape[0]}, expected {length_name} = {int(length)}")
    return np.ascontiguousarray(a.astype(target, copy=False))


def _as_float(product, name, value):
    if value is None:
        return float("nan")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.floating, np.integer)):
        raise _bad(product, f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _check_status(product, status):
    if status not in RESULT_STATUSES:
        raise _bad(product, f"status {status!r} not in {RESULT_STATUSES}")


def _check_text(product, name, value):
    if value is not None and not isinstance(value, str):
        raise _bad(product, f"{name} must be a string or None, got {type(value).__name__}")
    return value


def _point_entry(product, code, entry):
    if not isinstance(entry, dict):
        raise _bad(product, f"point[{code!r}] must be a dict")
    out = {}
    for k in POINT_VALUE_KEYS:
        out[k] = _as_float(product, f"point[{code!r}][{k!r}]", entry.get(k))
    out["dist_km"] = _as_float(product, f"point[{code!r}]['dist_km']", entry.get("dist_km"))
    out["model"] = _check_text(product, f"point[{code!r}]['model']", entry.get("model"))
    models = entry.get("models")
    if models is not None:
        if not isinstance(models, dict):
            raise _bad(product, f"point[{code!r}]['models'] must be a dict or None")
        mm = {}
        for name, m in models.items():
            if not isinstance(m, dict):
                raise _bad(product, f"point[{code!r}]['models'][{name!r}] must be a dict")
            one = {k: _as_float(product, f"models[{name!r}][{k!r}]", m.get(k))
                   for k in POINT_VALUE_KEYS}
            one["dist_km"] = _as_float(product, f"models[{name!r}]['dist_km']", m.get("dist_km"))
            one["available"] = bool(m.get("available", False))
            one["reason"] = _check_text(product, f"models[{name!r}]['reason']", m.get("reason"))
            for extra in set(m) - set(one):
                one[extra] = m[extra]
            mm[str(name)] = one
        models = mm
    out["models"] = models
    flags = entry.get("flags", [])
    if flags is None:
        flags = []
    if isinstance(flags, str) or not all(isinstance(f, str) for f in flags):
        raise _bad(product, f"point[{code!r}]['flags'] must be a list of strings")
    out["flags"] = list(flags)
    for extra in set(entry) - set(out):     # keep adapter extras (descriptive)
        out[extra] = entry[extra]
    return out


def _edge_share_default(kind, field, band):
    if kind == "band" and band is not None:
        tau = band["tau"]
        ok = np.isfinite(tau)
        w = band["n_px"].astype(float) * ok
        if w.sum() > 0:
            return float((w * band["edge"]).sum() / w.sum())
        return float(band["edge"][ok].mean()) if ok.any() else float("nan")
    if field is not None:
        f = field[np.isfinite(field)].astype(float)
        if f.size == 0:
            return float("nan")
        at = (np.abs(f - SCAN_LO) <= EDGE_TOL_MIN) | (np.abs(f - SCAN_HI) <= EDGE_TOL_MIN)
        return float(at.mean())
    return float("nan")


def _n_defined_default(kind, field, band, point, groups):
    if field is not None:
        return int(np.isfinite(field).sum())
    if band is not None:
        return int(np.isfinite(band["tau"]).sum())
    if point is not None:
        return int(sum(1 for v in point.values()
                       if any(np.isfinite(v[k]) for k in ("a", "ab", "b", "hwlw"))))
    if groups is not None:
        return int(sum(1 for g in groups.values() if g["status"] == "ok"))
    return 0


_LAG_FIELDS = ("stage", "reason", "kind", "lag_kind", "field", "band", "point", "groups",
               "interval95", "edge_share", "n_defined", "diag")


def lag_result(product, status="ok", *, n_keep=None, K=None, **fields) -> dict:
    """The canonical LagResult (plan INTERFACES §5) with defaults, validated.

    Unknown keywords are rejected (a typo would otherwise silently drop a
    product's output). ``n_keep`` and ``K`` (when given) check the lengths of
    the field (and of a pixel-level interval) and of the band arrays (and of
    a band-level interval and of every group). Arrays are cast to the
    canonical dtypes when the cast is lossless in kind (floats and integers to
    float, integers to integer, booleans or 0/1 integers to bool); anything
    else raises ``ValueError``. The function is idempotent, so the queue
    re-validates every dict an adapter returns with ``lag_result(**d, n_keep=,
    K=)``.

    Requirements by status: an ``ok`` result needs its ``kind`` and the
    matching payload (``field`` for kind field, ``band`` for kind band --
    band methods also return the broadcast ``field``, which the queue fills
    from ``domain.band_of`` when an adapter leaves it out -- ``point`` or
    ``groups``). ``empty`` and ``not_computable`` results may carry no
    payload; a ``not_computable`` one gets ``reason`` 'unspecified' when none
    is given.
    """
    unknown = set(fields) - set(_LAG_FIELDS)
    if unknown:
        raise _bad(product, f"unknown fields {sorted(unknown)} (put extras in diag)")
    if not isinstance(product, str) or not product:
        raise ValueError(f"lag result: product must be a non-empty string, got {product!r}")
    _check_status(product, status)
    stage = _check_text(product, "stage", fields.get("stage"))
    reason = _check_text(product, "reason", fields.get("reason"))
    if status == "not_computable" and not reason:
        reason = "unspecified"
    kind = fields.get("kind")
    if kind is not None and kind not in RESULT_KINDS[:4]:
        raise _bad(product, f"kind {kind!r} not in {RESULT_KINDS[:4]} (elev results use elev_result)")
    lag_kind = fields.get("lag_kind", "linear")
    if lag_kind is None:
        lag_kind = "linear"
    if lag_kind not in LAG_KINDS:
        raise _bad(product, f"lag_kind {lag_kind!r} not in {LAG_KINDS}")

    field = fields.get("field")
    if field is not None:
        field = _as_array(product, "field", field, "float32", n_keep, "n_keep")

    band = fields.get("band")
    if band is not None:
        if not isinstance(band, dict) or "tau" not in band:
            raise _bad(product, "band must be a dict with at least 'tau'")
        unknown_b = set(band) - {"tau", "n_px", "edge", "lo", "hi"}
        if unknown_b:
            raise _bad(product, f"band has unknown keys {sorted(unknown_b)}")
        tau = _as_array(product, "band.tau", band["tau"], "float64", K, "K")
        kb = tau.shape[0]
        n_px = band.get("n_px")
        n_px = (np.zeros(kb, np.int64) if n_px is None
                else _as_array(product, "band.n_px", n_px, "int64", kb, "K"))
        edge = band.get("edge")
        edge = (np.zeros(kb, bool) if edge is None
                else _as_array(product, "band.edge", edge, "bool", kb, "K"))
        lo = band.get("lo")
        hi = band.get("hi")
        lo = None if lo is None else _as_array(product, "band.lo", lo, "float64", kb, "K")
        hi = None if hi is None else _as_array(product, "band.hi", hi, "float64", kb, "K")
        band = {"tau": tau, "n_px": n_px, "edge": edge, "lo": lo, "hi": hi}

    point = fields.get("point")
    if point is not None:
        if not isinstance(point, dict):
            raise _bad(product, "point must be a dict {code: entry}")
        point = {str(c): _point_entry(product, c, e) for c, e in point.items()}

    groups = fields.get("groups")
    if groups is not None:
        if not isinstance(groups, dict):
            raise _bad(product, "groups must be a dict {group_id: entry}")
        gg = {}
        for gid, g in groups.items():
            if not isinstance(g, dict) or "tau" not in g:
                raise _bad(product, f"groups[{gid!r}] must be a dict with 'tau'")
            gtau = _as_array(product, f"groups[{gid!r}].tau", g["tau"], "float64", K, "K")
            gst = g.get("status", "ok")
            if not isinstance(gst, str):
                raise _bad(product, f"groups[{gid!r}].status must be a string")
            n_sc = g.get("n_scenes", 0)
            if isinstance(n_sc, (bool, np.bool_)) or not isinstance(n_sc, (Integral, np.integer)):
                raise _bad(product, f"groups[{gid!r}].n_scenes must be an integer")
            one = {"tau": gtau, "n_scenes": int(n_sc), "status": gst,
                   "reason": _check_text(product, f"groups[{gid!r}].reason", g.get("reason"))}
            for extra in set(g) - set(one):
                one[extra] = g[extra]
            gg[str(gid)] = one
        groups = gg

    iv = fields.get("interval95")
    if iv is not None:
        if not isinstance(iv, dict):
            raise _bad(product, "interval95 must be a dict or None")
        level = iv.get("level")
        if level not in INTERVAL_LEVELS:
            raise _bad(product, f"interval95.level {level!r} not in {INTERVAL_LEVELS}")
        nominal = iv.get("nominal")
        if not isinstance(nominal, str) or not nominal:
            raise _bad(product, "interval95.nominal must be a non-empty string")
        if level == "band":
            want = K if K is not None else (band["tau"].shape[0] if band is not None else None)
            lname = "K"
        else:
            want = n_keep if n_keep is not None else (field.shape[0] if field is not None else None)
            lname = "n_keep"
        lo_i = _as_array(product, "interval95.lo", iv.get("lo"), "float64", want, lname)
        hi_i = _as_array(product, "interval95.hi", iv.get("hi"), "float64", lo_i.shape[0], lname)
        iv = {"level": level, "lo": lo_i, "hi": hi_i, "nominal": nominal}

    if status == "ok":
        if kind is None:
            raise _bad(product, "an ok result needs its kind")
        need = {"field": field, "band": band, "point": point, "groups": groups}[kind]
        if need is None:
            raise _bad(product, f"an ok result of kind {kind!r} needs its {kind!r} payload")

    diag = fields.get("diag")
    if diag is None:
        diag = {}
    if not isinstance(diag, dict):
        raise _bad(product, "diag must be a dict")
    diag = {str(k): v for k, v in diag.items()}

    edge_share = fields.get("edge_share")
    edge_share = (_edge_share_default(kind, field, band) if edge_share is None
                  else _as_float(product, "edge_share", edge_share))
    n_defined = fields.get("n_defined")
    if n_defined is None:
        n_defined = _n_defined_default(kind, field, band, point, groups)
    elif isinstance(n_defined, (bool, np.bool_)) or not isinstance(n_defined, (Integral, np.integer)):
        raise _bad(product, "n_defined must be an integer")
    return {"product": product, "status": status, "stage": stage, "reason": reason,
            "kind": kind, "lag_kind": lag_kind, "field": field, "band": band,
            "point": point, "groups": groups, "interval95": iv,
            "edge_share": float(edge_share), "n_defined": int(n_defined), "diag": diag}


_ELEV_FIELDS = ("stage", "reason", "kind", "z", "sigma", "clear_obs", "tau_px", "n_defined", "diag")


def elev_result(product, status="ok", *, n_keep=None, **fields) -> dict:
    """The canonical ElevResult of E0 (E0a, E0b or E0g), validated.

    ``z``, ``sigma`` and ``tau_px`` are float32 and ``clear_obs`` int16, all
    of length ``n_keep`` (when given) and of one common length. An ``ok``
    result needs ``z``. Same casting rules and idempotence as
    :func:`lag_result`.
    """
    unknown = set(fields) - set(_ELEV_FIELDS)
    if unknown:
        raise _bad(product, f"unknown fields {sorted(unknown)} (put extras in diag)")
    if not isinstance(product, str) or not product:
        raise ValueError(f"elev result: product must be a non-empty string, got {product!r}")
    _check_status(product, status)
    kind = fields.get("kind", "elev")
    if kind not in (None, "elev"):
        raise _bad(product, f"an elevation result has kind 'elev', got {kind!r}")
    stage = _check_text(product, "stage", fields.get("stage"))
    reason = _check_text(product, "reason", fields.get("reason"))
    if status == "not_computable" and not reason:
        reason = "unspecified"
    arrays = {}
    length = n_keep
    for name, target in (("z", "float32"), ("sigma", "float32"),
                         ("clear_obs", "int16"), ("tau_px", "float32")):
        v = fields.get(name)
        if v is None:
            arrays[name] = None
            continue
        arrays[name] = _as_array(product, name, v, target, length, "n_keep")
        length = arrays[name].shape[0]
    if status == "ok" and arrays["z"] is None:
        raise _bad(product, "an ok elevation result needs z")
    diag = fields.get("diag")
    if diag is None:
        diag = {}
    if not isinstance(diag, dict):
        raise _bad(product, "diag must be a dict")
    n_defined = fields.get("n_defined")
    if n_defined is None:
        n_defined = int(np.isfinite(arrays["z"]).sum()) if arrays["z"] is not None else 0
    return {"product": product, "status": status, "stage": stage, "reason": reason,
            "kind": "elev", **arrays, "n_defined": int(n_defined),
            "diag": {str(k): v for k, v in diag.items()}}


# ─────────────────────────────────────────────────────────────────────────────
#  method cards
# ─────────────────────────────────────────────────────────────────────────────

def check_card(card) -> list[str]:
    """Problems of one method card (empty list = valid).

    Checks the presence of every :data:`CARD_KEYS` entry and the allowed
    values of family (:data:`FAMILIES`), kind ('lag' or 'elev'), tier ('1',
    '2', '3'), fidelity ('F0'..'F4' or '-'), output (a lag card: field, band,
    point or groups; an elevation card: any of :data:`RESULT_KINDS`) and
    lag_kind (:data:`LAG_KINDS`; an elevation card may leave it None).
    """
    if not isinstance(card, dict):
        return [f"card is not a dict ({type(card).__name__})"]
    problems = [f"missing key {k!r}" for k in CARD_KEYS if k not in card]
    fam = card.get("family")
    if "family" in card and fam not in FAMILIES:
        problems.append(f"bad family {fam!r}")
    kind = card.get("kind")
    if "kind" in card and kind not in CARD_KINDS:
        problems.append(f"bad kind {kind!r}")
    if "tier" in card and card["tier"] not in TIERS:
        problems.append(f"bad tier {card['tier']!r} (one of {TIERS}, as strings)")
    if "fidelity" in card and card["fidelity"] not in FIDELITIES:
        problems.append(f"bad fidelity {card['fidelity']!r}")
    if "output" in card:
        allowed = RESULT_KINDS if kind == "elev" else LAG_OUTPUTS
        if card["output"] not in allowed:
            problems.append(f"bad output {card['output']!r} for a {kind!r} card")
    if "lag_kind" in card:
        lk = card["lag_kind"]
        if not (lk in LAG_KINDS or (kind == "elev" and lk is None)):
            problems.append(f"bad lag_kind {lk!r}")
    return problems


def is_nan(x) -> bool:
    """True for a float NaN (helper for JSON-ready records)."""
    return isinstance(x, float) and math.isnan(x)
