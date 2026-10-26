# -*- coding: utf-8 -*-
"""Validation v2, Stage L -- the band-aggregate lag methods L04 XC, L07 LOOP and L10 HD26-R2.

Question
--------
Does the share of a band that is flooded (or the band's median index) follow
the boundary tide h_B(t - tau) best at the lag the in-estuary gauges
measure?  The three methods of this module read the same per-band aggregates
of one instance, built once and shared through ``inst.cache``
(:func:`band_aggregates`):

* **L04 XC** (F4, new; spec §3.3 row L04, §3.3.1, §4.2): the lagged Pearson
  cross-correlation of a band aggregate with h_B(t - tau) over the common lag
  window. XC-wet (product ``L04w``) reads the wet fraction (the binary layer
  W), XC-index (``L04i``) the band-median Y (the standardised, water-high
  index).
* **L07 LOOP** (F4, new; §3.3 row L07, §3.3.1): the hysteresis-loop collapse
  of the band wet fraction. At a wrong lag the rising and the falling limb
  of the (h_B(t - tau), wet fraction) cloud separate into a loop; at the
  right lag they fall on ONE increasing curve, so forcing both limbs onto
  one isotonic curve costs nothing beyond each limb's own isotonic fit.
* **L10 HD26-R2** (F4; §3.3.1): Hart-Davis et al.'s response idea as a
  delay-and-smear model with an isotonic link. It is run by
  :func:`experiments.benchmark.lags.hd26.run_unit` (the L09 unit writes L09
  and L10); its card is :data:`CARD_L10`, listed in ``hd26.CARDS``.

Pre-registered rules (frozen in :data:`CONFIG`; the cards repeat them)
----------------------------------------------------------------------
Band aggregates (one instance; memoised in ``inst.cache['aggregate.band']``):

* band k = the keep pixels with ``domain.band_of == k`` (band -1, a pixel
  without a thalweg distance, is in no band); ``n_px[k]`` its size;
* ``n_valid[t, k]`` = #V and ``n_wet[t, k]`` = #(W & V) over band k at scene t
  (int32); ``cover = n_valid / n_px``; ``wetfrac = n_wet / n_valid`` (NaN
  where n_valid = 0);
* ``median_Y[t, k]`` = the median of Y over the band's valid pixels (float32,
  NaN where none). A band that fits one read (at most ``read_cols`` columns)
  takes ``np.median`` per scene; a bigger band an exact two-pass selection on
  order-preserving uint32 keys of the float32 values (pass 1: a per-scene
  histogram of the top 16 key bits; pass 2: the keys inside the bins that
  hold the middle rank(s)). Both give ``np.median``'s value bit for bit; the
  selection only avoids holding a whole band in memory (declared);
* the band is read through ``inst.obs.read(cols)``, one band at a time, at
  most ``read_cols`` = 100,000 columns and ``read_cells_max`` = 5e7 cells
  per call (the plan's ceiling is 250,000 columns per call).

L04 XC (spec §3.3.1 "XC and LOOP need >= 80 % band coverage per scene
(sensitivities 0.5 and 0.3), >= 30 usable scenes"):

* usable scenes: ``cover >= 0.8`` (card variants cover0.5, cover0.3);
* a band needs >= 30 usable scenes;
* score(tau) = Pearson r(proxy, h_B(t - tau)) over the usable scenes, tau in
  ``common.TAU_SCAN`` (-45..180 min, step 5; spec §3.3); tau_k =
  ``common.scan_vertex(score)``, which flags a best point on the window edge.

L07 LOOP (spec §3.3.1 "... and >= 15 scenes per limb for LOOP"):

* proxy = the wet fraction on usable scenes (cover >= 0.8); >= 30 usable;
* per tau: the limb of a scene is the sign of dh_B/dt at t - tau (central
  difference of the 1-min boundary series over +-1 min,
  ``inst.levels.series_at``); the score is the LOOP GAP, minus (SSE_pooled -
  SSE_rise - SSE_fall), where SSE_pooled is the residual of one increasing
  isotonic fit (pool-adjacent-violators, ``truth.isotonic``) of the proxy on
  h_B(t - tau) through the scenes of both limbs and SSE_rise, SSE_fall the
  residuals of the rising and the falling scenes' own isotonic fits: the
  extra residual of forcing the two limbs onto one curve (>= 0, 0 when they
  collapse); it is undefined unless each limb holds >= 15 usable scenes;
* tau_k = ``scan_vertex(score)``.
* Declared revision (2026-10-20, before lock-A, nothing seen on truth): the
  V2 build scored -SSE_pooled alone, which is exactly L10 HD26-R2's score at
  s = 0 (an isotonic m2c, found by the code review: identical scores and
  lags); the limbs then entered only as a gate. The loop gap is the
  collapse criterion the method's definition names (spec §3.3 'isotonic
  limbs').

L10 HD26-R2 (spec §3.3.1 "y_k(t) = F_k(sum_j w_j h_B(t - j dt)) ... (tau, s)
is chosen by minimum SSE, and s is reported; s = 0 reduces the method to an
isotonic m2c. Needs >= 60 scenes."):

* proxy = the band wet fraction (``proxy='index'``: the band-median Y) on
  scenes with cover >= 0.8 (HD26's band rule, FROZEN['hd26']); >= 60 such
  scenes;
* u(t; tau, s) = sum_j w_j h_B(t - j) over integer minutes j, w a normalised
  Gaussian of centre tau and width s in {0, 15, 30, 60} min truncated at
  +-3 s. s = 0 is the exact level row ``levels.at(tau)``; s > 0 smears the
  1-min boundary series (``levels.series_at``, linear between minutes;
  declared: the interpolation error is below 0.1 mm for the M2 curvature);
* SSE(tau, s) of the increasing isotonic fit F_k of the proxy on u; s* = the
  s whose smallest SSE over the window is smallest (ties: the smaller s);
  tau_k = ``scan_vertex(-SSE(., s*))``; s* is reported per band.

Statuses (every adapter of package B)
-------------------------------------
* ``not_computable``: a rule on the RECORD stops the method before any
  estimation (here: no band holds enough usable scenes; the harness domain
  is ``no_intertidal``, raised as ``product.NotComputable``);
* ``empty``: the estimation ran but identified no band (every score
  undefined); the scorer reports such a product NOT COMPUTABLE (empty), never
  as a zero lag (spec §0.3 item 8: the read-out ladder's step 3 applies
  only to a product with a field that is undefined near one gauge);
* ``ok``: at least one band identified; the band lags are broadcast to the
  band's keep pixels (``common.broadcast_band``).

Conventions: tau > 0 means the local water is later than h_B (spec §3.3).
Nothing here is tuned on truth; competitors and in-house adaptations alike
read the same instance (spec §5.1).
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd

from experiments.benchmark import config, product, truth
from experiments.benchmark.lags import common

# ─────────────────────────────────────────────────────────────────────────────
#  configuration and cards
# ─────────────────────────────────────────────────────────────────────────────

_HD = config.FROZEN["hd26"]

CONFIG = {
    "aggregate": {
        "read_cols": 100_000,
        "read_cells_max": 50_000_000,
        "hist_bits": 16,
        "median": ("np.median per scene when the band fits one read; otherwise an exact "
                   "two-pass selection on order-preserving uint32 keys of the float32 values "
                   "(top 16 bits, then the keys of the middle-rank bins)"),
    },
    "L04": {
        "proxies": {"L04w": "wetfrac", "L04i": "median_Y"},
        "min_cover": 0.8,
        "cover_variants": {"cover0.5": 0.5, "cover0.3": 0.3},
        "min_usable": 30,
        "score": "Pearson r(proxy, h_B(t - tau)) over the usable scenes",
        "scan": [float(common.SCAN_LO), float(common.SCAN_HI), 5.0],
        "vertex": "common.scan_vertex (te._parabolic_argmax + edge flag)",
    },
    "L07": {
        "proxy": "wetfrac",
        "min_cover": 0.8,
        "cover_variants": {"cover0.5": 0.5, "cover0.3": 0.3},
        "min_usable": 30,
        "min_per_limb": 15,
        "rate_dt_min": 1.0,
        "limb": "sign of dh_B/dt at t - tau (central difference of series1 over +-1 min)",
        "score": ("-(SSE_pooled - SSE_rise - SSE_fall): the loop gap of increasing isotonic "
                  "(PAVA) fits, pooled over both limbs against each limb's own"),
        "revision": "2026-10-20: the loop gap replaces -SSE_pooled (= L10 at s = 0)",
        "scan": [float(common.SCAN_LO), float(common.SCAN_HI), 5.0],
    },
    "L10": {
        "proxy": "wetfrac",
        "min_cover": float(_HD["band_valid_share"]),
        "min_scenes": 60,
        "s_grid_min": [0.0, 15.0, 30.0, 60.0],
        "truncate_sd": 3.0,
        "smear_source": "s = 0: levels.at(tau); s > 0: levels.series_at (1-min h_B, linear)",
        "score": "-SSE of the increasing isotonic fit of the proxy on u(t; tau, s)",
        "select": "s* = argmin_s min_tau SSE (ties: smaller s); tau = scan_vertex(-SSE(., s*))",
        "scan": [float(common.SCAN_LO), float(common.SCAN_HI), 5.0],
    },
}
del _HD

#: repository files whose source enters the code key of these units
CODE_DEPS = ("experiments/benchmark/truth.py", "pyintertidal/tide_estimators.py")

_SPEC = "docs/validation_v2_spec.md"
_REF_SPEC_XC = {"cite": "Validation v2 specification §3.3 (row L04) and §3.3.1: in-house "
                        "adaptation (F4), no published implementation", "doi_or_url": _SPEC}
_REF_KNAPP = {"cite": "Knapp CH, Carter GC (1976) The generalized correlation method for "
                      "estimation of time delay. IEEE Trans. Acoust. Speech Signal Process. "
                      "24(4):320-327", "doi_or_url": "doi:10.1109/TASSP.1976.1162830"}
_REF_M2C = {"cite": "MAREA m2c flooded-area rank correlation (the binary-gauging idea that XC "
                    "and LOOP transfer): pyintertidal.tide_estimators.m2c_waterline",
            "doi_or_url": "pyintertidal/tide_estimators.py"}
_REF_PAVA = {"cite": "Ayer M, Brunk HD, Ewing GM, Reid WT, Silverman E (1955) An empirical "
                     "distribution function for sampling with incomplete information. Ann. "
                     "Math. Stat. 26(4):641-647 (pool-adjacent-violators)",
             "doi_or_url": "doi:10.1214/aoms/1177728423"}
_REF_HD26 = {"cite": "Hart-Davis MG, Monahan T, Vos K, Andersen OB (2026) Commun. Earth "
                     "Environ. (satellite-derived tidal constants; the response idea)",
             "doi_or_url": "doi:10.1038/s43247-026-03943-9"}

_SCAN = [float(common.SCAN_LO), float(common.SCAN_HI), 5.0]
_CONTROL_WHY = ("0 min: the gating world (spec §10.2) delays the local level by the planted "
                "lag with an amplitude ratio of 1, so the band aggregate is a monotone "
                "function of h_B(t - tau_k) and its best lag is the band's lag (the band "
                "mean of a planted profile within the band)")


#: the dependency signature and the cost model of package B's units, as
#: ``designs.METHODS`` holds them (designs owns the effective keys; the tests
#: assert the copies are equal, so a card never silently drifts from the plan)
UNIT_SIGNATURE = {"reads": ["cloud", "detect", "index", "level", "prep", "window"],
                  "domain": True}
UNIT_COST = {"L04": (0.06, 1.0), "L07": (0.06, 1.0), "L08": (0.2, 1.5), "L09": (0.2, 1.0),
             "L11": (0.06, 1.0), "L12": (0.06, 1.0)}


def card_signature(method):
    if method not in UNIT_COST:
        raise KeyError(f"{method} is not a unit of package B")
    return {"reads": list(UNIT_SIGNATURE["reads"]), "domain": bool(UNIT_SIGNATURE["domain"])}


def card_cost(method):
    wh, ram = UNIT_COST[method]
    return {"wh_cell": float(wh), "ram_gb": float(ram)}


def _card_xc(pid, proxy_text, layer):
    return {
        "id": pid,
        "name": f"XC-{'wet' if pid == 'L04w' else 'index'}: lagged cross-correlation of the "
                f"band {proxy_text}",
        "family": "order statistic",
        "kind": "lag",
        "tier": "1",
        "fidelity": "F4",
        "evidence": ("F4 adaptation (new code): unit tests on synthetic tides "
                     "(tests/test_benchmark_lags_aggregate.py: planted 0/45/120 min recovered "
                     "within 10 min, edge flag at 200 min, QC thresholds) and the positive "
                     "control of spec §10.2 before printing"),
        "references": [_REF_SPEC_XC, _REF_KNAPP, _REF_M2C],
        "output": "band",
        "lag_kind": "linear",
        "signature": card_signature("L04"),
        "scan_window": list(_SCAN),
        "published_window": None,
        "as_published": ("no published configuration: in-house transfer of lagged "
                         "cross-correlation delay estimation to the band aggregate"),
        "native_cell": None,
        "common_conditions": (f"reads {layer} of the instance's setting on the harness domain "
                              "keep; bands = domain bands (3 km along s); level = exact h_B "
                              "rows at TAU_SCAN; one lag per band broadcast to its pixels"),
        "deviations": [],
        "qc": {"min_cover": 0.8, "min_usable_scenes": 30,
               "score_defined": "r needs a non-constant proxy and level"},
        "n_min": 30,
        "tuning": [],
        "control_offset_min": 0.0,
        "control_offset_why": _CONTROL_WHY,
        "cost": card_cost("L04"),
        "variants": {"cover0.5": "usable scenes at cover >= 0.5 (sensitivity, not scheduled)",
                     "cover0.3": "usable scenes at cover >= 0.3 (sensitivity, not scheduled)"},
    }


CARDS = {
    "L04w": _card_xc("L04w", "wet fraction", "W & V (wet fraction n_wet / n_valid)"),
    "L04i": _card_xc("L04i", "median index", "Y & V (median of the standardised index)"),
    "L07": {
        "id": "L07",
        "name": "LOOP: hysteresis-loop collapse of the band wet fraction (isotonic limbs)",
        "family": "self-consistency",
        "kind": "lag",
        "tier": "1",
        "fidelity": "F4",
        "evidence": ("F4 adaptation (new code): unit tests on synthetic tides "
                     "(tests/test_benchmark_lags_aggregate.py) and the positive control of "
                     "spec §10.2 before printing"),
        "references": [{"cite": "Validation v2 specification §3.3 (row L07) and §3.3.1: "
                                "in-house (F4)", "doi_or_url": _SPEC}, _REF_PAVA, _REF_M2C],
        "output": "band",
        "lag_kind": "linear",
        "signature": card_signature("L07"),
        "scan_window": list(_SCAN),
        "published_window": None,
        "as_published": "no published configuration (in-house)",
        "native_cell": None,
        "common_conditions": ("reads W & V on the harness domain keep (band wet fraction), the "
                              "exact h_B rows at TAU_SCAN and the 1-min h_B for the limb; one "
                              "lag per band broadcast to its pixels"),
        "deviations": [{"paper": "V2 build of 2026-10-19: score = -SSE of one isotonic fit "
                                 "through both limbs (the limbs only gated the score)",
                        "ours": "score = -(SSE_pooled - SSE_rise - SSE_fall), the loop gap",
                        "why": "the earlier score equals L10 HD26-R2 at s = 0 (code review "
                               "2026-10-20; no truth seen): the method is the collapse of the "
                               "two limbs, so the limbs must enter the score",
                        "rows": "L07", "sensitivity": "L10 at s = 0 is the pooled-only score"}],
        "qc": {"min_cover": 0.8, "min_usable_scenes": 30, "min_scenes_per_limb": 15,
               "limb": "sign of dh_B/dt at t - tau, +-1 min central difference",
               "score": "loop gap -(SSE_pooled - SSE_rise - SSE_fall)"},
        "n_min": 30,
        "tuning": [],
        "control_offset_min": 0.0,
        "control_offset_why": _CONTROL_WHY,
        "cost": card_cost("L07"),
        "variants": {"cover0.5": "usable scenes at cover >= 0.5 (sensitivity, not scheduled)",
                     "cover0.3": "usable scenes at cover >= 0.3 (sensitivity, not scheduled)"},
    },
}

#: the L10 card (listed in hd26.CARDS: the L09 unit writes L09 and L10)
CARD_L10 = {
    "id": "L10",
    "name": "HD26-R2: delay-and-smear response of the band wet fraction with an isotonic link",
    "family": "harmonic/response",
    "kind": "lag",
    "tier": "1",
    "fidelity": "F4",
    "evidence": ("F4 adaptation (new code): unit tests (tests/test_benchmark_lags_aggregate.py, "
                 "tests/test_benchmark_lags_hd26.py: planted lags, s = 0 equals the plain "
                 "isotonic scan, the 60-scene rule) and the positive control of spec §10.2"),
    "references": [{"cite": "Validation v2 specification §3.3 (row L10) and §3.3.1",
                    "doi_or_url": _SPEC}, _REF_HD26, _REF_PAVA],
    "output": "band",
    "lag_kind": "linear",
    "signature": card_signature("L09"),
    "scan_window": list(_SCAN),
    "published_window": None,
    "as_published": ("HD26 fit harmonic constants to a satellite level proxy; the response form "
                     "(delay, smear, monotone link) is this study's adaptation (RTide, HD26-R1, "
                     "is Tier 2)"),
    "native_cell": None,
    "common_conditions": ("band wet fraction on scenes with >= 80 % valid band (HD26's band "
                          "rule); exact h_B rows at s = 0, the 1-min h_B smeared for s > 0"),
    "deviations": [{"paper": "harmonic analysis of the proxy (L09)",
                    "ours": "time-domain response y = F(sum_j w_j h_B(t - j)), F isotonic",
                    "why": "a response model needs no 365-day span and no constituent choice",
                    "rows": "L10", "sensitivity": "L09 (HD26-H) on the same unit"}],
    "qc": {"min_cover": 0.8, "min_scenes": 60, "s_grid_min": [0, 15, 30, 60],
           "truncate_sd": 3.0},
    "n_min": 60,
    "tuning": [],
    "control_offset_min": 0.0,
    "control_offset_why": _CONTROL_WHY + "; the gating world has no smear, so s* = 0 is expected",
    "cost": card_cost("L09"),
    "variants": {"index": "proxy = band-median Y (HD26 index proxy; not scheduled)"},
}


# ─────────────────────────────────────────────────────────────────────────────
#  small helpers shared by the run functions of this module
# ─────────────────────────────────────────────────────────────────────────────

def _stage(inst, name):
    """``inst.stage(name)`` (timing.py stopwatch) or a no-op context."""
    fn = getattr(inst, "stage", None)
    return fn(name) if callable(fn) else contextlib.nullcontext()


def _progress(inst, n=1):
    fn = getattr(inst, "progress", None)
    if callable(fn):
        fn(n)


def _cache(inst):
    c = getattr(inst, "cache", None)
    return c if isinstance(c, dict) else None


def _uget(unit, key, default=None):
    if isinstance(unit, dict):
        return unit.get(key, default)
    return getattr(unit, key, default)


def unit_products(unit, default):
    """The products the unit asks for (``unit['products']``), in order; each
    must be one this adapter writes."""
    p = _uget(unit, "products", None)
    if p is None:
        return tuple(default)
    p = tuple(str(x) for x in (p.split() if isinstance(p, str) else p))
    bad = [x for x in p if x not in default]
    if bad:
        raise ValueError(f"products {bad} are not written by this adapter ({default})")
    return p


def variant_parts(unit):
    v = _uget(unit, "variant", "") or ""
    return tuple(x for x in str(v).split("+") if x)


def require_domain(inst):
    """Every adapter of package B consumes the harness domain: none runs on a
    ``no_intertidal`` domain (fewer than 3,000 keep pixels, no geometry;
    MAREA's reconstruct stops there too)."""
    d = inst.domain
    if str(getattr(d, "status", "ok")) == "no_intertidal":
        n = int(len(d.keep))
        raise product.NotComputable(
            "no_intertidal", f"the harness domain is no_intertidal ({n} keep pixels, fewer "
            "than the extraction rule's minimum)", {"n_keep": n})


def t_minutes(t):
    """Naive-UTC instants -> float minutes since 1970-01-01 (ns-exact; the
    formula of ``levels.minutes_since_epoch``)."""
    return np.asarray(pd.DatetimeIndex(t).values.astype("datetime64[ns]")
                      .astype(np.int64)) / 6e10


def _n_keep(inst):
    return int(len(inst.domain.keep))


def chunk_cols(T, cells, cols_range):
    """Columns of one fitting chunk: ``cells / T`` clamped to ``cols_range``
    (per-pixel methods bound their (T, chunk) float64 temporaries this way)."""
    lo, hi = (int(v) for v in cols_range)
    return max(lo, min(hi, int(cells) // max(int(T), 1)))


def _cover_variant(unit, section):
    cv = CONFIG[section]["cover_variants"]
    out = float(CONFIG[section]["min_cover"])
    for part in variant_parts(unit):
        if part in cv:
            out = float(cv[part])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  band aggregates
# ─────────────────────────────────────────────────────────────────────────────

_SIGN = np.uint32(0x80000000)
_LOW31 = np.uint32(0x7FFFFFFF)


def f32_keys(y):
    """Order-preserving uint32 keys of float32 values (radix-sort keys):
    ``a < b`` iff ``key(a) < key(b)`` for non-NaN floats (-0.0 sorts just
    below +0.0). One (T, m) uint32 buffer: negative values flip every bit,
    the others only the sign bit."""
    b = np.ascontiguousarray(y, dtype=np.float32).view(np.uint32)
    k = b >> np.uint32(31)                       # 1 for a negative value
    k *= _LOW31                                  # 0 or 0x7FFFFFFF
    k |= _SIGN                                   # 0x80000000 or 0xFFFFFFFF
    k ^= b
    return k


def keys_to_f32(k):
    """Inverse of :func:`f32_keys`."""
    k = np.atleast_1d(np.asarray(k, dtype=np.uint32))
    b = np.where((k & _SIGN) != 0, k & _LOW31, ~k).astype(np.uint32)
    return b.view(np.float32)


def band_columns(inst):
    """Positions (indices into ``domain.keep``) of each band's pixels."""
    band_of = np.asarray(inst.domain.band_of, np.int64)
    K = int(inst.domain.K)
    order = np.argsort(band_of, kind="stable")
    sb = band_of[order]
    out = []
    for k in range(K):
        a, b = np.searchsorted(sb, k, "left"), np.searchsorted(sb, k, "right")
        out.append(np.sort(order[a:b]).astype(np.int64))
    return out


def read_cols_for(T, read_cols=None):
    """Columns per ``obs.read`` call: at most ``read_cols`` and
    ``read_cells_max / T``."""
    a = CONFIG["aggregate"]
    rc = int(a["read_cols"] if read_cols is None else read_cols)
    return max(1, min(rc, int(a["read_cells_max"]) // max(int(T), 1)))


def _hist_top(counts, top, V, chunk=64):
    """Add the per-scene histogram of ``top`` (only where V) to ``counts``
    (T, n_bins), in scene chunks so the bincount stays small."""
    T, nb = counts.shape
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        r, c = np.nonzero(V[a:b])
        if not len(r):
            continue
        idx = r.astype(np.int64) * nb + top[a:b][r, c].astype(np.int64)
        counts[a:b] += np.bincount(idx, minlength=(b - a) * nb).reshape(b - a, nb).astype(
            counts.dtype)


def _read_band(inst, cols):
    ob = inst.obs.read(np.asarray(cols, np.int64))
    V = np.asarray(ob.V, bool)
    W = np.asarray(ob.W, bool) & V
    Y = np.asarray(ob.Y, np.float32)
    return Y, V, W


def _band_two_pass(inst, blocks, T, n_valid_k, n_wet_k, med_k):
    """Counts and the exact per-scene median of a band read in several
    blocks (pass 1: counts and the top-16-bit key histogram; pass 2: the keys
    of the bins holding the middle ranks)."""
    bits = int(CONFIG["aggregate"]["hist_bits"])
    nb = 1 << bits
    shift = np.uint32(32 - bits)
    counts = np.zeros((T, nb), np.int32)
    for cols in blocks:
        Y, V, W = _read_band(inst, cols)
        n_valid_k += V.sum(axis=1)
        n_wet_k += W.sum(axis=1)
        keys = f32_keys(Y)
        del Y, W
        keys >>= shift                                        # the top bits, in place
        _hist_top(counts, keys, V)
        _progress(inst, len(cols))
        del V, keys
    # key range [lo_key, hi_key) of the bins that hold the middle rank(s);
    # an empty range (1, 0) for a scene without valid pixels
    lo_key = np.ones((T, 1), np.uint64)
    hi_key = np.zeros((T, 1), np.uint64)
    before = np.zeros(T, np.int64)
    r_lo = np.zeros(T, np.int64)
    r_hi = np.zeros(T, np.int64)
    for t in range(T):
        n = int(n_valid_k[t])
        if n == 0:
            continue
        cs = np.cumsum(counts[t], dtype=np.int64)
        if int(cs[-1]) != n:                                  # pragma: no cover
            raise AssertionError("median selection: histogram and counts disagree")
        r_lo[t], r_hi[t] = (n - 1) // 2, n // 2
        b_lo = int(np.searchsorted(cs, r_lo[t], side="right"))
        b_hi = int(np.searchsorted(cs, r_hi[t], side="right"))
        before[t] = int(cs[b_lo]) - int(counts[t, b_lo])
        lo_key[t, 0] = b_lo << (32 - bits)
        hi_key[t, 0] = (b_hi + 1) << (32 - bits)
    del counts
    cand = [[] for _ in range(T)]
    for cols in blocks:
        Y, V, _ = _read_band(inst, cols)
        keys = f32_keys(Y)
        del Y
        sel = V & (keys >= lo_key) & (keys < hi_key)
        for t in np.flatnonzero(sel.any(axis=1)):
            cand[t].append(keys[t][sel[t]])
        _progress(inst, len(cols))
        del V, keys, sel
    for t in range(T):
        if int(n_valid_k[t]) == 0:
            continue
        ks = np.sort(np.concatenate(cand[t]))
        v_lo = keys_to_f32(ks[r_lo[t] - before[t]])[0]
        if r_lo[t] == r_hi[t]:
            med_k[t] = v_lo
        else:
            v_hi = keys_to_f32(ks[r_hi[t] - before[t]])[0]
            med_k[t] = np.median(np.array([v_lo, v_hi], np.float32))


def band_aggregates(inst, read_cols=None, cache=True):
    """The per-band aggregates of an instance (module docstring).

    Returns
    -------
    dict
        ``n_valid``, ``n_wet`` (T, K) int32; ``n_px`` (K,) int64; ``cover``,
        ``wetfrac`` (T, K) float64; ``median_Y`` (T, K) float32; ``path``
        (per band: "single", "two-pass" or "empty"); ``read_cols``; ``K``,
        ``T``.
    """
    c = _cache(inst) if cache else None
    key = "aggregate.band"
    if c is not None and key in c:
        return c[key]
    T = int(inst.obs.T)
    K = int(inst.domain.K)
    cols_k = band_columns(inst)
    rc = read_cols_for(T, read_cols)
    n_valid = np.zeros((T, K), np.int64)
    n_wet = np.zeros((T, K), np.int64)
    med = np.full((T, K), np.nan, np.float32)
    n_px = np.array([len(x) for x in cols_k], np.int64)
    path = []
    for k, cols in enumerate(cols_k):
        if not len(cols):
            path.append("empty")
            continue
        blocks = [cols[i:i + rc] for i in range(0, len(cols), rc)]
        if len(blocks) == 1:
            Y, V, W = _read_band(inst, cols)
            n_valid[:, k] = V.sum(axis=1)
            n_wet[:, k] = W.sum(axis=1)
            for t in range(T):
                v = Y[t][V[t]]
                if v.size:
                    med[t, k] = np.median(v)
            _progress(inst, len(cols))
            del Y, V, W
            path.append("single")
        else:
            nv, nw, mk = n_valid[:, k].copy(), n_wet[:, k].copy(), med[:, k].copy()
            _band_two_pass(inst, blocks, T, nv, nw, mk)
            n_valid[:, k], n_wet[:, k], med[:, k] = nv, nw, mk
            path.append("two-pass")
    with np.errstate(invalid="ignore", divide="ignore"):
        cover = np.where(n_px[None, :] > 0, n_valid / np.maximum(n_px[None, :], 1), np.nan)
        wetfrac = np.where(n_valid > 0, n_wet / np.maximum(n_valid, 1), np.nan)
    out = {"n_valid": n_valid.astype(np.int32), "n_wet": n_wet.astype(np.int32),
           "n_px": n_px, "cover": cover, "wetfrac": wetfrac, "median_Y": med,
           "path": path, "read_cols": int(rc), "K": K, "T": T}
    if c is not None:
        c[key] = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  levels over the common window
# ─────────────────────────────────────────────────────────────────────────────

def _scan_ints():
    return np.asarray(np.round(common.TAU_SCAN), np.int64)


def levels_scan(inst):
    """h_B(t - tau) at the record instants for tau in TAU_SCAN, (46, T)
    float64: the exact level rows (``inst.levels.stack``), memoised in
    ``inst.cache['aggregate.levels_scan']``."""
    c = _cache(inst)
    key = "aggregate.levels_scan"
    if c is not None and key in c:
        return c[key]
    taus = _scan_ints()
    if not np.array_equal(taus.astype(float), common.TAU_SCAN):     # pragma: no cover
        raise AssertionError("TAU_SCAN must hold integer minutes")
    H = np.asarray(inst.levels.stack(taus), np.float64)
    T = int(inst.obs.T)
    if H.shape != (len(taus), T):
        raise ValueError(f"levels.stack gave {H.shape}, expected {(len(taus), T)}")
    if c is not None:
        c[key] = H
    return H


def rate_scan(inst, dt=None):
    """dh_B/dt at t - tau (m/min) for tau in TAU_SCAN, (46, T): the central
    difference of the 1-min boundary series over +-dt minutes
    (``inst.levels.series_at``); memoised in ``inst.cache['aggregate.rate_scan']``."""
    dt = float(CONFIG["L07"]["rate_dt_min"] if dt is None else dt)
    c = _cache(inst)
    key = f"aggregate.rate_scan.{dt:g}"
    if c is not None and key in c:
        return c[key]
    t = t_minutes(inst.t)
    R = np.empty((len(common.TAU_SCAN), len(t)))
    for j, tau in enumerate(common.TAU_SCAN):
        up = np.asarray(inst.levels.series_at(t - tau + dt), np.float64)
        dn = np.asarray(inst.levels.series_at(t - tau - dt), np.float64)
        R[j] = (up - dn) / (2.0 * dt)
    if c is not None:
        c[key] = R
    return R


def gaussian_weights(s, truncate=None):
    """(offsets m, weights w): integer minutes |m| <= truncate * s and the
    normalised Gaussian exp(-m^2 / (2 s^2)); s = 0 gives ([0], [1])."""
    truncate = float(CONFIG["L10"]["truncate_sd"] if truncate is None else truncate)
    s = float(s)
    if s <= 0:
        return np.zeros(1, np.int64), np.ones(1)
    half = int(np.floor(truncate * s))
    m = np.arange(-half, half + 1, dtype=np.int64)
    w = np.exp(-0.5 * (m / s) ** 2)
    return m, w / w.sum()


def smeared_scan(inst, s):
    """u(t_i; tau, s) for tau in TAU_SCAN, (46, T): s = 0 the exact rows
    (:func:`levels_scan`); s > 0 the Gaussian smear of the 1-min series,
    ``sum_m w_m h_B(t_i - tau - m)``. Memoised per s."""
    if float(s) <= 0:
        return levels_scan(inst)
    c = _cache(inst)
    key = f"aggregate.smear.{float(s):g}"
    if c is not None and key in c:
        return c[key]
    m, w = gaussian_weights(s)
    t = t_minutes(inst.t)
    U = np.empty((len(common.TAU_SCAN), len(t)))
    for j, tau in enumerate(common.TAU_SCAN):
        q = t[None, :] - tau - m[:, None].astype(np.float64)        # (M, T)
        h = np.asarray(inst.levels.series_at(q.ravel()), np.float64).reshape(q.shape)
        U[j] = w @ h
    if c is not None:
        c[key] = U
    return U


# ─────────────────────────────────────────────────────────────────────────────
#  the band scores (pure functions of arrays; the run functions call them)
# ─────────────────────────────────────────────────────────────────────────────

def pearson_rows(p, H):
    """Pearson r of ``p`` (n,) with each row of ``H`` (R, n); NaN where a
    variance is zero or a value is not finite."""
    p = np.asarray(p, np.float64)
    H = np.asarray(H, np.float64)
    out = np.full(H.shape[0], np.nan)
    if p.size < 2 or not np.all(np.isfinite(p)):
        return out
    fin = np.all(np.isfinite(H), axis=1)
    if not np.ptp(p) > 0 or not fin.any():           # a constant proxy has no r
        return out
    pc = p - p.mean()
    sp = float(np.sum(pc * pc))
    Hc = H[fin] - H[fin].mean(axis=1, keepdims=True)
    sh = np.sum(Hc * Hc, axis=1)
    num = Hc @ pc
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.ptp(H[fin], axis=1) > 0, num / np.sqrt(sh * sp), np.nan)
    out[fin] = r
    return out


def isotonic_sse(y, x):
    """SSE of the increasing least-squares isotonic fit of ``y`` on ``x``
    (PAVA, ``truth.isotonic``, on the stable order of x)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    o = np.argsort(x, kind="stable")
    ys = y[o]
    fit = truth.isotonic(ys)
    return float(np.sum((ys - fit) ** 2))


def isotonic_scan(y, Hu):
    """SSE of :func:`isotonic_sse` for every row of ``Hu`` (NaN where a row
    holds a non-finite level)."""
    out = np.full(Hu.shape[0], np.nan)
    for j in range(Hu.shape[0]):
        if np.all(np.isfinite(Hu[j])):
            out[j] = isotonic_sse(y, Hu[j])
    return out


def xc_band(proxy, H_u, min_usable=None, grid=common.TAU_SCAN):
    """L04 on one band: ``proxy`` (n,) and ``H_u`` (46, n) on the usable
    scenes. Returns ``{"tau", "edge", "score", "reason"}``."""
    min_usable = int(CONFIG["L04"]["min_usable"] if min_usable is None else min_usable)
    proxy = np.asarray(proxy, np.float64)
    if proxy.size < min_usable:
        return {"tau": np.nan, "edge": False, "score": np.full(len(grid), np.nan),
                "reason": f"usable_lt_{min_usable}"}
    score = pearson_rows(proxy, H_u)
    tau, edge = common.scan_vertex(score, grid)
    return {"tau": tau, "edge": bool(edge), "score": score,
            "reason": "ok" if np.isfinite(tau) else "score_undefined"}


def loop_gap(proxy, h, rate):
    """The loop gap of L07 at one lag: SSE_pooled - SSE_rise - SSE_fall of
    increasing isotonic fits of ``proxy`` on ``h`` over the rising (rate > 0)
    and the falling (rate < 0) scenes (>= 0: the rising and falling fits are
    a relaxation of the pooled one). Scenes with rate == 0 are left out."""
    proxy = np.asarray(proxy, np.float64)
    h = np.asarray(h, np.float64)
    rate = np.asarray(rate, np.float64)
    up, dn = rate > 0, rate < 0
    both = up | dn
    pooled = isotonic_sse(proxy[both], h[both])
    return pooled - isotonic_sse(proxy[up], h[up]) - isotonic_sse(proxy[dn], h[dn])


def loop_band(proxy, H_u, R_u, min_usable=None, min_per_limb=None, grid=common.TAU_SCAN):
    """L07 on one band: ``proxy`` (n,), ``H_u`` and ``R_u`` (46, n) the
    levels and their rates at t - tau on the usable scenes; the score is
    minus the loop gap (:func:`loop_gap`). Returns ``{"tau", "edge",
    "score", "n_rise", "n_fall", "reason"}``."""
    min_usable = int(CONFIG["L07"]["min_usable"] if min_usable is None else min_usable)
    min_limb = int(CONFIG["L07"]["min_per_limb"] if min_per_limb is None else min_per_limb)
    proxy = np.asarray(proxy, np.float64)
    L = len(grid)
    n_rise = np.sum(np.asarray(R_u) > 0, axis=1).astype(np.int64)
    n_fall = np.sum(np.asarray(R_u) < 0, axis=1).astype(np.int64)
    score = np.full(L, np.nan)
    if proxy.size < min_usable:
        return {"tau": np.nan, "edge": False, "score": score, "n_rise": n_rise,
                "n_fall": n_fall, "reason": f"usable_lt_{min_usable}"}
    if not np.ptp(proxy) > 0:                   # a constant proxy fits every lag alike
        return {"tau": np.nan, "edge": False, "score": score, "n_rise": n_rise,
                "n_fall": n_fall, "reason": "proxy_constant"}
    ok = (n_rise >= min_limb) & (n_fall >= min_limb)
    for j in np.flatnonzero(ok):
        if np.all(np.isfinite(H_u[j])) and np.all(np.isfinite(R_u[j])):
            score[j] = -loop_gap(proxy, H_u[j], R_u[j])
    tau, edge = common.scan_vertex(score, grid)
    reason = "ok" if np.isfinite(tau) else (
        f"limb_lt_{min_limb}" if ok.sum() < 3 else "score_undefined")
    return {"tau": tau, "edge": bool(edge), "score": score, "n_rise": n_rise,
            "n_fall": n_fall, "reason": reason}


def l10_band(proxy, U_by_s, s_grid, min_scenes=None, grid=common.TAU_SCAN):
    """L10 on one band: ``U_by_s`` = [(46, n) smeared levels per s] on the
    usable scenes. Returns ``{"tau", "edge", "s", "sse" (n_s, 46), "reason"}``."""
    min_scenes = int(CONFIG["L10"]["min_scenes"] if min_scenes is None else min_scenes)
    proxy = np.asarray(proxy, np.float64)
    sse = np.full((len(s_grid), len(grid)), np.nan)
    if proxy.size < min_scenes:
        return {"tau": np.nan, "edge": False, "s": np.nan, "sse": sse,
                "reason": f"scenes_lt_{min_scenes}"}
    if not np.ptp(proxy) > 0:                   # a constant proxy fits every lag alike
        return {"tau": np.nan, "edge": False, "s": np.nan, "sse": sse,
                "reason": "proxy_constant"}
    for i, U in enumerate(U_by_s):
        sse[i] = isotonic_scan(proxy, U)
    best = np.array([np.nanmin(r) if np.isfinite(r).any() else np.inf for r in sse])
    if not np.isfinite(best).any():
        return {"tau": np.nan, "edge": False, "s": np.nan, "sse": sse,
                "reason": "score_undefined"}
    i_s = int(np.argmin(best))                      # first minimum: the smaller s
    tau, edge = common.scan_vertex(-sse[i_s], grid)
    return {"tau": tau, "edge": bool(edge), "s": float(s_grid[i_s]), "sse": sse,
            "reason": "ok" if np.isfinite(tau) else "score_undefined"}


# ─────────────────────────────────────────────────────────────────────────────
#  results
# ─────────────────────────────────────────────────────────────────────────────

def band_result(pid, inst, tau, edge, n_px, reasons, record_rule, *, lag_kind="linear",
                diag=None):
    """The canonical LagResult of a band method (module docstring, statuses).

    ``record_rule``: the reason prefix (or a tuple of prefixes) of a band
    stopped by a rule on the record (e.g. 'usable_lt_'); when every band
    carries one the product is not_computable, with the first band's reason
    (e.g. 'usable_lt_30') as stage."""
    record_rule = tuple(record_rule) if isinstance(record_rule, (list, tuple)) \
        else (str(record_rule),)
    tau = np.asarray(tau, np.float64)
    edge = np.asarray(edge, bool)
    K = int(inst.domain.K)
    n_keep = _n_keep(inst)
    reasons = [str(r) for r in reasons]
    diag = dict(diag or {})
    diag["band_reason"] = reasons
    diag["n_bands_identified"] = int(np.isfinite(tau).sum())
    if np.isfinite(tau).any():
        return common.lag_result(
            pid, "ok", n_keep=n_keep, K=K, kind="band", lag_kind=lag_kind,
            field=common.broadcast_band(tau, inst.domain.band_of),
            band={"tau": tau, "n_px": np.asarray(n_px, np.int64), "edge": edge},
            diag=diag)
    if reasons and all(r.startswith(record_rule) for r in reasons):
        return common.lag_result(
            pid, "not_computable", n_keep=n_keep, K=K, kind="band", lag_kind=lag_kind,
            stage=reasons[0],
            reason=f"no band passes the record rule ({reasons[0]})", diag=diag)
    return common.lag_result(
        pid, "empty", n_keep=n_keep, K=K, kind="band", lag_kind=lag_kind, stage="qc",
        reason="no band identified (" + ", ".join(sorted(set(reasons))) + ")", diag=diag)


# ─────────────────────────────────────────────────────────────────────────────
#  run functions
# ─────────────────────────────────────────────────────────────────────────────

def run_l04(inst, unit, opts=None):
    """L04 XC: products L04w (wet fraction) and L04i (band-median Y)."""
    require_domain(inst)
    products = unit_products(unit, ("L04w", "L04i"))
    min_cover = _cover_variant(unit, "L04")
    min_usable = int(CONFIG["L04"]["min_usable"])
    with _stage(inst, "aggregate"):
        agg = band_aggregates(inst)
    with _stage(inst, "levels"):
        H = levels_scan(inst)
    K = agg["K"]
    out = []
    for pid in products:
        proxy = agg[CONFIG["L04"]["proxies"][pid]]
        taus, edges, reasons, scores, n_use = [], [], [], [], []
        with _stage(inst, f"scan.{pid}"):
            for k in range(K):
                use = np.isfinite(agg["cover"][:, k]) & (agg["cover"][:, k] >= min_cover)
                use &= np.isfinite(proxy[:, k])
                r = xc_band(proxy[use, k], H[:, use], min_usable)
                taus.append(r["tau"])
                edges.append(r["edge"])
                reasons.append(r["reason"])
                scores.append(r["score"])
                n_use.append(int(use.sum()))
        diag = {"min_cover": min_cover, "n_usable": np.asarray(n_use, np.int64),
                "score": np.asarray(scores), "tau_scan": common.TAU_SCAN.copy(),
                "aggregate_path": list(agg["path"]), "proxy": CONFIG["L04"]["proxies"][pid]}
        out.append(band_result(pid, inst, taus, edges, agg["n_px"], reasons, "usable_lt_",
                               diag=diag))
    return out


def run_l07(inst, unit, opts=None):
    """L07 LOOP: the hysteresis-loop collapse of the band wet fraction."""
    require_domain(inst)
    products = unit_products(unit, ("L07",))
    min_cover = _cover_variant(unit, "L07")
    with _stage(inst, "aggregate"):
        agg = band_aggregates(inst)
    with _stage(inst, "levels"):
        H = levels_scan(inst)
        R = rate_scan(inst)
    K = agg["K"]
    wf = agg["wetfrac"]
    taus, edges, reasons, scores, n_use, n_rise, n_fall = [], [], [], [], [], [], []
    with _stage(inst, "scan"):
        for k in range(K):
            use = np.isfinite(agg["cover"][:, k]) & (agg["cover"][:, k] >= min_cover)
            use &= np.isfinite(wf[:, k])
            r = loop_band(wf[use, k], H[:, use], R[:, use])
            taus.append(r["tau"])
            edges.append(r["edge"])
            reasons.append(r["reason"])
            scores.append(r["score"])
            n_use.append(int(use.sum()))
            n_rise.append(r["n_rise"])
            n_fall.append(r["n_fall"])
    diag = {"min_cover": min_cover, "n_usable": np.asarray(n_use, np.int64),
            "score_neg_loop_gap": np.asarray(scores), "n_rise": np.asarray(n_rise),
            "n_fall": np.asarray(n_fall), "tau_scan": common.TAU_SCAN.copy(),
            "aggregate_path": list(agg["path"])}
    res = band_result("L07", inst, taus, edges, agg["n_px"], reasons, "usable_lt_", diag=diag)
    return [res for _ in products]


def l10_hd26r2(inst, proxy="wet", s_grid=None, product_id="L10"):
    """L10 HD26-R2 on every band (module docstring); called by
    ``hd26.run_unit``. ``proxy``: 'wet' (primary) or 'index'."""
    if proxy not in ("wet", "index"):
        raise ValueError(f"proxy must be 'wet' or 'index', got {proxy!r}")
    cfg = CONFIG["L10"]
    s_grid = [float(s) for s in (cfg["s_grid_min"] if s_grid is None else s_grid)]
    min_cover = float(cfg["min_cover"])
    min_scenes = int(cfg["min_scenes"])
    with _stage(inst, "aggregate"):
        agg = band_aggregates(inst)
    with _stage(inst, "smear"):
        U_all = [smeared_scan(inst, s) for s in s_grid]
    P = agg["wetfrac"] if proxy == "wet" else agg["median_Y"].astype(np.float64)
    K = agg["K"]
    taus, edges, reasons, s_best, sse_min, n_use = [], [], [], [], [], []
    with _stage(inst, "scan.L10"):
        for k in range(K):
            use = np.isfinite(agg["cover"][:, k]) & (agg["cover"][:, k] >= min_cover)
            use &= np.isfinite(P[:, k])
            r = l10_band(P[use, k], [U[:, use] for U in U_all], s_grid, min_scenes)
            taus.append(r["tau"])
            edges.append(r["edge"])
            reasons.append(r["reason"])
            s_best.append(r["s"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sse_min.append(np.nanmin(r["sse"], axis=1) if np.isfinite(r["sse"]).any()
                               else np.full(len(s_grid), np.nan))
            n_use.append(int(use.sum()))
    diag = {"proxy": proxy, "min_cover": min_cover, "s_grid_min": np.asarray(s_grid),
            "s_best_min": np.asarray(s_best, np.float64),
            "sse_min_by_s": np.asarray(sse_min, np.float64),
            "n_usable": np.asarray(n_use, np.int64), "tau_scan": common.TAU_SCAN.copy(),
            "aggregate_path": list(agg["path"])}
    return band_result(product_id, inst, taus, edges, agg["n_px"], reasons,
                       f"scenes_lt_{min_scenes}", diag=diag)
