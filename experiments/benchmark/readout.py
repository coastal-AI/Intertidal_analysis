# -*- coding: utf-8 -*-
"""Validation v2 — lag read-outs: one rule for every method's output (spec §6.3).

Question
--------
Which single lag does a method's output give at a gauge? The scorer, not the
method, reads every applied lag field out with the same rule, so a pixel
method and a band method meet the gauge on equal terms.

Rules (spec §6.3, amending v13 D8; scoring spec §1; the numbers live in
``scoring.SCORING_V2['readout']``)
--------------------------------------------------------------------------
* Field and band products, on the instance's domain ``keep``:
  1. the median of the finite field over nbhd_g ∩ keep (the judge's
     neighbourhood, frame flat indices, intersected with keep by
     ``searchsorted``) when at least 30 pixels are defined
     (FROZEN scoring.readout.min_px);
  2. otherwise the median of the field over the instance's modal band
     k_g^inst = ``judges.station_geometry(lon, lat, grid, keep, s_km,
     band_of, centers)['k_g']`` (the band of most of the 200 keep pixels
     nearest the gauge on THIS instance's domain);
  3. otherwise 0, flagged ``fallback0``.
  Band-broadcast fields always meet step 1; pixel-QC methods (GM20, PPL,
  BT26) may not near harbour walls, and the ladder keeps them from being sent
  straight to 0.
* Also reported: ``tau_band`` (the median over k_g^inst), ``tau_band_interp``
  (linear in s at the gauge's s on the instance, between the band medians at
  the band centres, clamped at the end bands), ``tau_step1`` (NaN when step 1
  fails), the read-out pixel count and the window-edge share of the read-out
  pixels: the share of band-edge bands for band methods (``band_edge``), the
  share of values on the common scan bounds -45 / 180 min for linear pixel
  methods; flag ``edge`` when it exceeds 0.5.
* Phase-type outputs (GM20, HD26, PPL, a model's (b)) are circular: values
  are wrapped to +-P_M2/2 about the truth-free centre 67.5 min
  (``lags.common.wrap_phase``) and their median is ``circ_median``; never
  wrapped about a truth. ``tau_band_interp`` unwraps successive band medians
  before interpolating and wraps the result.
* Point products (models, physics): the method's estimand for the truth of
  the same name, ``point[code][truth]`` (:func:`point_readout`).
* Group products (L01v per relative orbit): each group's band lags are
  broadcast to the field and read out with the same ladder
  (:func:`group_readouts`); descriptive rows only.
* Interval at the gauge (interval score, scoring spec §3): band level =
  [lo_k, hi_k] of the read-out band k_g^inst (steps 1-2); pixel level = the
  medians of lo and hi over the read-out pixels (the step-1 set, else the
  band set); phase intervals take the circular median of their centres and
  the median half-width (:func:`interval_readout`).
"""
from __future__ import annotations

import numpy as np

from experiments.benchmark import config, judges
from experiments.benchmark.lags import common as lc

#: step 1 needs this many defined neighbourhood pixels (FROZEN)
MIN_PX = int(config.FROZEN["scoring"]["readout"]["min_px"])
#: a read-out is flagged 'edge' when more than this share of its pixels sit
#: on the window edge
EDGE_SHARE_FLAG = 0.5
STEPS = {1: "neighbourhood median", 2: "modal band of the instance", 3: "zero (fallback0)"}
LAG_KINDS = lc.LAG_KINDS


def nbhd_positions(nbhd, keep):
    """Positions in ``keep`` (ascending frame flat indices) of the
    neighbourhood pixels that belong to it."""
    keep = np.asarray(keep, np.int64)
    nb = np.asarray(nbhd, np.int64).ravel()
    if not len(keep) or not len(nb):
        return np.empty(0, np.int64)
    pos = np.searchsorted(keep, nb)
    ok = pos < len(keep)
    ok[ok] = keep[pos[ok]] == nb[ok]
    return np.unique(pos[ok]).astype(np.int64)


def _check_kind(lag_kind):
    if lag_kind not in LAG_KINDS:
        raise ValueError(f"lag_kind must be one of {LAG_KINDS}, not {lag_kind!r}")


def median_of(values, lag_kind="linear"):
    """The median of the finite values (circular about 67.5 min for phase)."""
    _check_kind(lag_kind)
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    if lag_kind == "phase":
        return lc.circ_median(v)
    return float(np.median(v))


def band_medians(field, band_of, K, lag_kind="linear"):
    """(K,) median of the finite field per band (circular for phase)."""
    _check_kind(lag_kind)
    f = np.asarray(field, np.float64).ravel()
    bo = np.asarray(band_of).astype(np.int64).ravel()
    out = np.full(int(K), np.nan)
    ok = np.isfinite(f) & (bo >= 0) & (bo < int(K))
    if not ok.any():
        return out
    vals = lc.wrap_phase(f[ok]) if lag_kind == "phase" else f[ok]
    b = bo[ok]
    order = np.argsort(b, kind="stable")
    b, vals = b[order], vals[order]
    ks = np.arange(int(K))
    lo = np.searchsorted(b, ks, side="left")
    hi = np.searchsorted(b, ks, side="right")
    for k in ks[hi > lo]:
        out[k] = float(np.median(vals[lo[k]:hi[k]]))
    return out


def band_interp(medians, centers, s_g, lag_kind="linear"):
    """Linear interpolation in s at ``s_g`` between band medians at the band
    centres, clamped at the end bands (``np.interp``); phase medians are
    unwrapped along s first and the result wrapped about 67.5 min."""
    _check_kind(lag_kind)
    v = np.asarray(medians, np.float64).ravel()
    c = np.asarray(centers, np.float64).ravel()
    if s_g is None or not np.isfinite(s_g) or len(v) != len(c):
        return float("nan")
    m = np.isfinite(v) & np.isfinite(c)
    if not m.any():
        return float("nan")
    order = np.argsort(c[m], kind="stable")
    c, v = c[m][order], v[m][order]
    if lag_kind == "phase":
        v = v[0] + np.concatenate([[0.0], np.cumsum(lc.circ_diff(v[1:], v[:-1]))])
        return float(lc.wrap_phase(np.interp(float(s_g), c, v)))
    return float(np.interp(float(s_g), c, v))


def judge_geometry(judge, grid, keep, s_km, band_of, centers=None):
    """``judges.station_geometry`` of a judge on an instance's domain (its
    modal band k_g^inst and its s on the instance)."""
    return judges.station_geometry(judge["lon"], judge["lat"], grid, np.asarray(keep, np.int64),
                                   np.asarray(s_km, np.float64), np.asarray(band_of),
                                   None if centers is None else np.asarray(centers, np.float64))


def readout(field, keep, band_of, s_km, centers, judge, grid, lag_kind="linear", min_px=MIN_PX,
            *, band_edge=None, geom=None, edge_tol=lc.EDGE_TOL_MIN, band_meds=None):
    """The read-out ladder of spec §6.3 for one field and one judge.

    Parameters
    ----------
    field : (n_keep,) float
        The applied lag (min) on ``keep``; NaN = undefined (band products:
        the broadcast band values).
    keep, band_of, s_km, centers :
        The instance's domain: frame flat indices (ascending), band per keep
        pixel (-1 for NaN s), s (km) and the band centres (K,).
    judge : dict
        A judge-table record: ``nbhd`` (frame flat, ascending), ``lon``,
        ``lat`` and ``s_g_km`` (fallback for the instance's s).
    grid : dict
        ``xs``, ``ys``, ``crs`` of the frame.
    lag_kind : "linear" or "phase"
    min_px : int
        Step 1's minimum of defined neighbourhood pixels.
    band_edge : (K,) bool, optional
        Band methods: the band-edge flags; the edge share is then the share
        of read-out pixels in edge bands.
    geom : dict, optional
        A precomputed :func:`judge_geometry` (cached per instance and judge).
    band_meds : (K,) array, optional
        A precomputed :func:`band_medians` of this field (cached per product).

    Returns
    -------
    dict
        tau, step (1, 2 or 3), n_px (read-out pixels), band (k_g^inst),
        tau_band, tau_band_interp, tau_step1, edge_share, flags, s_g_km (on
        the instance), n_nbhd_keep, positions (keep positions read out).
    """
    _check_kind(lag_kind)
    f = np.asarray(field, np.float64).ravel()
    keep = np.asarray(keep, np.int64).ravel()
    bo = np.asarray(band_of).astype(np.int64).ravel()
    centers = np.asarray(centers, np.float64).ravel()
    if not (len(f) == len(keep) == len(bo)):
        raise ValueError(f"field ({len(f)}), keep ({len(keep)}) and band_of ({len(bo)}) "
                         "must be aligned on the instance's keep")
    K = len(centers)
    if geom is None:
        geom = judge_geometry(judge, grid, keep, s_km, bo, centers)
    k_g = int(geom.get("k_g", -1))
    s_g = geom.get("s_g_km", float("nan"))
    if s_g is None or not np.isfinite(s_g):
        s_g = judge.get("s_g_km", float("nan"))
        s_g = float("nan") if s_g is None else float(s_g)
    flags = []
    nbpos = nbhd_positions(judge["nbhd"], keep)
    def_nb = nbpos[np.isfinite(f[nbpos])]
    if 0 <= k_g < K:
        band_pos = np.flatnonzero((bo == k_g) & np.isfinite(f))
        tau_band = median_of(f[band_pos], lag_kind)
    else:
        band_pos = np.empty(0, np.int64)
        tau_band = float("nan")
        flags.append("no_band")
    meds = band_medians(f, bo, K, lag_kind) if band_meds is None else band_meds
    tau_interp = band_interp(meds, centers, s_g, lag_kind)
    if len(def_nb) >= int(min_px):
        step, pos = 1, def_nb
        tau = median_of(f[pos], lag_kind)
        tau_step1 = tau
    elif len(band_pos):
        step, pos = 2, band_pos
        tau, tau_step1 = tau_band, float("nan")
        flags.append("step2")
    else:
        step, pos = 3, np.empty(0, np.int64)
        tau, tau_step1 = 0.0, float("nan")
        flags.append("fallback0")
    edge_share = float("nan")
    if len(pos):
        if band_edge is not None:
            be = np.asarray(band_edge, bool).ravel()
            b = bo[pos]
            inb = (b >= 0) & (b < len(be))
            if inb.any():
                edge_share = float(np.mean(be[b[inb]]))
        elif lag_kind == "linear":
            v = f[pos]
            at = (np.abs(v - lc.SCAN_LO) <= edge_tol) | (np.abs(v - lc.SCAN_HI) <= edge_tol)
            edge_share = float(np.mean(at))
    if np.isfinite(edge_share) and edge_share > EDGE_SHARE_FLAG:
        flags.append("edge")
    return {"tau": float(tau), "step": int(step), "n_px": int(len(pos)), "band": k_g,
            "tau_band": float(tau_band), "tau_band_interp": float(tau_interp),
            "tau_step1": float(tau_step1), "edge_share": edge_share, "flags": flags,
            "s_g_km": float(s_g), "n_nbhd_keep": int(len(nbpos)), "positions": pos}


def point_readout(point, code, truth_key):
    """The point product's estimand at gauge ``code`` for truth ``truth_key``
    ("a", "ab", "b", "hwlw", or "internal_a"/"internal_b"); NaN when the
    product holds no finite value there."""
    if not point:
        return float("nan")
    e = point.get(code)
    if e is None:
        e = point.get(str(code))
    if not isinstance(e, dict):
        return float("nan")
    v = e.get(truth_key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def group_readouts(groups, keep, band_of, s_km, centers, judge, grid, min_px=MIN_PX, *,
                   geom=None, lag_kind="linear"):
    """Per-group read-outs of a group product (L01v per relative orbit):
    each group's band lags broadcast to the field (``lags.common.
    broadcast_band``) and read out with :func:`readout`.

    Returns
    -------
    dict {group_id: readout dict + status, reason, n_scenes}
    """
    out = {}
    bo = np.asarray(band_of).astype(np.int64)
    if geom is None and groups:
        geom = judge_geometry(judge, grid, keep, s_km, bo, centers)
    for gid, g in (groups or {}).items():
        st = g.get("status", "ok")
        if st != "ok":
            out[str(gid)] = {"tau": float("nan"), "step": None, "status": st,
                             "reason": g.get("reason"), "n_scenes": g.get("n_scenes"),
                             "flags": [], "n_px": 0}
            continue
        fld = lc.broadcast_band(np.asarray(g["tau"], np.float64), bo)
        r = readout(fld, keep, bo, s_km, centers, judge, grid, lag_kind, min_px, geom=geom)
        r.update(status="ok", reason=g.get("reason"), n_scenes=g.get("n_scenes"))
        out[str(gid)] = r
    return out


def interval_readout(interval95, ro, lag_kind="linear"):
    """``(lo, hi)`` of a product's nominal 95 % interval at the gauge.

    Parameters
    ----------
    interval95 : dict or None
        ``{"level": "band" | "pixel", "lo", "hi", "nominal"}`` (band level:
        (K,) arrays; pixel level: (n_keep,) arrays on keep).
    ro : dict
        The :func:`readout` of the product at the gauge.

    Returns
    -------
    (float, float), NaN when undefined (no interval, or step 3).
    """
    nan = (float("nan"), float("nan"))
    if not interval95 or ro is None or ro.get("step") not in (1, 2):
        return nan
    lo = np.asarray(interval95.get("lo"), np.float64).ravel()
    hi = np.asarray(interval95.get("hi"), np.float64).ravel()
    if interval95.get("level") == "band":
        k = int(ro.get("band", -1))
        if not (0 <= k < len(lo)) or len(hi) != len(lo):
            return nan
        return float(lo[k]), float(hi[k])
    pos = np.asarray(ro.get("positions", ()), np.int64)
    if not len(pos) or pos.max() >= len(lo) or len(hi) != len(lo):
        return nan
    a, b = lo[pos], hi[pos]
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return nan
    a, b = a[ok], b[ok]
    if lag_kind == "phase":
        centre = lc.circ_median(a + lc.circ_diff(b, a) / 2.0)
        half = float(np.median(lc.circ_diff(b, a) / 2.0))
        return centre - half, centre + half
    return float(np.median(a)), float(np.median(b))
