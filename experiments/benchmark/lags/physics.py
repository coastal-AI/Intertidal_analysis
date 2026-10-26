# -*- coding: utf-8 -*-
"""Validation v2 — L17 zero lag and L18 shallow-water celerity (spec §3.3, §3.3.1).

Question
--------
How far do the simplest physical baselines get: no lag at all (L17), and
the travel time of a shallow-water wave up the channel, tau(s) =
integral_0^s ds' / sqrt(g hbar(s')), on the EMODnet depth of the permanently
wet channel (L18)?

Pre-registered rules
--------------------
L17 (spec §3.3 row L17, family "physics", a lower bound):

* every truth key (``a``, ``ab``, ``b``, ``hwlw``) is 0 at every in-frame
  judge; the product also carries the field 0 on ``domain.keep`` so the
  scorer's read-out ladder treats it like the imagery rows (kind "point").

L18 (spec §3.3 row L18, §3.3.1 "L18", §6.6 L-physics):

* hbar(s) is the mean depth below MSL of the permanently wet channel per
  1-km s bin: the frame pixels whose full-record water frequency is >= 0.98
  (package geometry ``wf``, the Omega anchor rule) with ``s_frame`` (the
  along-water distance from the mouth, km). Without the geometry the
  instance's sea mask (wf >= 0.90) is used instead (declared, flagged).
* Depth: the EMODnet Digital Bathymetry DTM 2024 (``fetch emodnet``,
  elevation referred to LAT, negative below), resampled to the frame grid
  by nearest neighbour. LAT -> MSL with one site offset per window:
  offset = mean(h_B over the window) - LAT_ref, LAT_ref the lowest h_B of
  the window's 1-min boundary series (declared: an observed minimum that
  includes negative surges, not the 18.6-year astronomical LAT; CONFIG
  ``lat_ref_m`` may give a charted value per site instead). Depth below MSL
  = offset - DTM elevation.
* hbar is floored at 1 m; bins without channel water take the linear
  interpolation in s of the bins that have it (constant beyond the ends).
  g = 9.81 m/s^2.
* Output: a point at each judge's s_g (the same tau for every truth) and
  the field tau(s_km) on ``domain.keep``. No gauge series is read.
* Missing or changed DTM tile: NOT COMPUTABLE ('no_depths'), never zero.
* Variants (in the card, not scheduled): ``site_mean`` (one depth, the mean
  over every channel pixel) and ``hw`` / ``lw`` (speeds sqrt(g (hbar +-
  a_M2)), a_M2 the M2 amplitude of h_B in the window).
* In Dutch waters EMODnet includes the Vaklodingen (the elevation truth);
  this is declared for a lag-only row (spec §3.3.1).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments.benchmark import config, levels, settings, truth
from experiments.benchmark.lags import common, models
from experiments.benchmark.product import NotComputable

NAN = float("nan")
G = 9.81
BIN_KM = 1.0
#: permanent water: full-record water frequency >= 0.98 (package OMEGA_WATER)
WET_MIN = 0.98
DEPTH_FLOOR_M = 1.0
#: charted LAT below MSL per site (m, in h_B's datum) replacing the observed
#: minimum; empty = the declared rule for every site
LAT_REF_M = {}

CONFIG = {
    "L17": {"rule": "tau = 0 at every judge and on keep"},
    "L18": {
        "rule": ("tau(s) = integral_0^s ds' / sqrt(g hbar(s')), hbar(s) the mean depth below "
                 "MSL of the permanently wet channel (geometry wf >= 0.98, s_frame) per 1-km s "
                 "bin from the EMODnet DTM 2024 (nearest neighbour), LAT -> MSL by the site "
                 "offset mean(h_B) - LAT_ref over the window, LAT_ref = min of the window's "
                 "1-min h_B; hbar floored at 1 m, empty bins interpolated in s, constant "
                 "beyond; a point at each judge's s_g and the field on keep"),
        "g": G, "bin_km": BIN_KM, "wet_min": WET_MIN, "depth_floor_m": DEPTH_FLOOR_M,
        "lat_ref": {"rule": "min of the window's 1-min h_B (inst.series1)",
                    "override_m": dict(LAT_REF_M)},
        "dtm": "fetch.emodnet (EMODnet Digital Bathymetry DTM 2024, LAT, elevation negative "
               "below); nearest-neighbour resampling to the frame grid",
        "fallback_channel": "inst.domain.sea (wf >= 0.90) when the geometry is not available "
                            "(declared, flagged)",
        "variants": {"site_mean": "one depth: the mean over every channel pixel",
                     "hw": "sqrt(g (hbar + a_M2))", "lw": "sqrt(g max(hbar - a_M2, floor))"},
    },
}

_COMMON_CARD = {"kind": "lag", "tier": "1", "fidelity": "-", "output": "point",
                "lag_kind": "linear", "scan_window": None, "published_window": None,
                "native_cell": None, "tuning": [], "control_offset_min": 0.0}
CARDS = {
    "L17": dict(_COMMON_CARD, **{
        "id": "L17", "name": "zero lag", "family": "physics",
        "evidence": "trivial; unit test (zeros everywhere, status ok)",
        "references": [{"cite": "v2 spec §3.3 row L17 (lower bound)",
                        "doi_or_url": "docs/validation_v2_spec.md"}],
        "signature": {"reads": ["window"], "domain": False},
        "as_published": "tau = 0 (what every method without a lag assumes)",
        "common_conditions": "the same zero everywhere; the field on keep for the read-out "
                             "ladder",
        "deviations": [], "qc": {"control": "expected 0 in the 'zero' scenario; the planted "
                                            "profile is not recovered by construction"},
        "n_min": 0, "cost": {"wh_cell": 0.0, "ram_gb": 0.5}, "variants": {}}),
    "L18": dict(_COMMON_CARD, **{
        "id": "L18", "name": "shallow-water celerity on the EMODnet channel depth",
        "family": "physics",
        "evidence": ("unit tests: constant depth h gives tau(s) = s / sqrt(g h) exactly; the 1 m "
                     "floor; interpolated empty bins; missing tiles NOT COMPUTABLE"),
        "references": [
            {"cite": "Dean RG, Dalrymple RA (1991) Water Wave Mechanics for Engineers and "
                     "Scientists (shallow-water celerity sqrt(g h))",
             "doi_or_url": "doi:10.1142/1232"},
            {"cite": "EMODnet Bathymetry Consortium: EMODnet Digital Bathymetry (DTM 2024)",
             "doi_or_url": "https://emodnet.ec.europa.eu/geonetwork/srv/api/records/"
                           "cf51df64-56f9-4a99-b1aa-36b8d7b743a1"},
        ],
        "signature": {"reads": ["window"], "domain": True},
        "as_published": "the frictionless shallow-water wave speed on the mean channel depth",
        "common_conditions": "the package geometry (wf, s_frame) and the window's h_B for the "
                             "LAT -> MSL offset",
        "deviations": [
            {"paper": "LAT from the 18.6-year astronomical tide", "ours": "the lowest h_B of "
             "the window (observed, includes surges)", "why": "truth-free, from h_B only",
             "rows": "L18", "sensitivity": "CONFIG lat_ref_m (charted values) if given"},
            {"paper": "a cross-section mean depth", "ours": "the mean over the permanently wet "
             "pixels of a 1-km s bin", "why": "spec §3.3.1", "rows": "L18",
             "sensitivity": "variant site_mean"},
            {"paper": "independent bathymetry", "ours": "EMODnet includes the Vaklodingen in "
             "Dutch waters", "why": "the only consistent DTM; a lag-only row",
             "rows": "L18", "sensitivity": "none (declared)"},
        ],
        "qc": {"depth_floor_m": DEPTH_FLOOR_M, "empty_bins": "interpolated in s",
               "control": "no EMODnet tile exists for the synthetic site (NOT COMPUTABLE there)"},
        "n_min": 0, "cost": {"wh_cell": 0.02, "ram_gb": 1.0},
        "variants": {"site_mean": "one depth over the whole channel",
                     "hw": "HW speed sqrt(g (hbar + a_M2))",
                     "lw": "LW speed sqrt(g max(hbar - a_M2, 1 m))"}}),
}

CODE_DEPS = ("experiments/benchmark/fetch/emodnet.py", "experiments/benchmark/fetch/tides.py",
             "experiments/benchmark/lags/models.py", "experiments/benchmark/lags/common.py",
             "experiments/benchmark/levels.py", "experiments/benchmark/settings.py",
             "experiments/benchmark/truth.py")


def log(msg):
    print(msg, flush=True)


def _keep(inst):
    d = getattr(inst, "domain", None)
    if d is None:
        return np.zeros(0, np.int64), np.zeros(0, np.float64)
    keep = np.asarray(getattr(d, "keep", np.zeros(0)), np.int64)
    s = getattr(d, "s_km", None)
    s = np.full(len(keep), np.nan) if s is None else np.asarray(s, np.float64)
    return keep, s


def _entry(value, flags=(), **extra):
    e = {"a": value, "ab": value, "b": value, "hwlw": value, "internal_a": NAN,
         "internal_b": NAN, "dist_km": NAN, "model": None, "models": None,
         "flags": list(flags)}
    e.update(extra)
    return e


# ─────────────────────────────────────────────────────────────────────────────
#  L17
# ─────────────────────────────────────────────────────────────────────────────

def run_l17(inst, unit, opts=None):
    """Unit L17: zero at every judge and on keep (status ok)."""
    keep, _ = _keep(inst)
    point = {str(r["code"]): _entry(0.0) for r in models.in_frame_judges(inst)}
    r = common.lag_result("L17", "ok", kind="point", lag_kind="linear", point=point,
                          field=np.zeros(len(keep), np.float32), n_keep=len(keep),
                          diag={"window": str(inst.window), "n_judges": len(point)})
    return [r for _ in models.unit_products(unit, ("L17",))]


# ─────────────────────────────────────────────────────────────────────────────
#  L18
# ─────────────────────────────────────────────────────────────────────────────

def tau_of_s(s_km, hbar, bin_km=BIN_KM, g=G):
    """tau(s) (min) = integral_0^s ds' / sqrt(g hbar(s')) with ``hbar`` constant
    on the bins [k bin_km, (k + 1) bin_km), k = 0..len(hbar) - 1, and the last
    bin's depth beyond. NaN for NaN or negative s.

    Parameters
    ----------
    s_km : array
    hbar : (K,) array of positive depths (m)
    """
    h = np.asarray(hbar, np.float64)
    if not len(h) or not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("hbar must be finite and positive in every bin")
    s = np.asarray(s_km, np.float64)
    sec_per_km = 1000.0 / np.sqrt(g * h)                      # s per km, per bin
    cum = np.concatenate([[0.0], np.cumsum(sec_per_km * bin_km)])
    out = np.full(s.shape, np.nan)
    ok = np.isfinite(s) & (s >= 0)
    k = np.minimum(np.floor(s[ok] / bin_km).astype(np.int64), len(h) - 1)
    out[ok] = (cum[k] + (s[ok] - k * bin_km) * sec_per_km[k]) / 60.0
    return out


def bin_depths(s_km, depth_m, n_bins, bin_km=BIN_KM, floor_m=DEPTH_FLOOR_M):
    """hbar per 1-km s bin: the mean depth of the channel pixels in the bin,
    floored at ``floor_m``; bins without a pixel take the linear
    interpolation in s (bin centres) of the bins that have one (constant
    beyond the ends).

    Returns
    -------
    dict
        ``hbar`` (n_bins,), ``raw`` (the unfloored means, NaN where empty),
        ``n_px`` (n_bins,), ``interpolated`` (bool), ``floored`` (bool).
    """
    s = np.asarray(s_km, np.float64)
    d = np.asarray(depth_m, np.float64)
    ok = np.isfinite(s) & np.isfinite(d) & (s >= 0)
    k = np.floor(s[ok] / bin_km).astype(np.int64)
    inside = k < n_bins
    k, dd = k[inside], d[ok][inside]
    n = np.bincount(k, minlength=n_bins)[:n_bins]
    tot = np.bincount(k, weights=dd, minlength=n_bins)[:n_bins]
    raw = np.full(n_bins, np.nan)
    has = n > 0
    raw[has] = tot[has] / n[has]
    if not has.any():
        raise ValueError("no channel pixel with a depth in any bin")
    floored = has & (raw < floor_m)
    h = np.where(has, np.maximum(raw, floor_m), np.nan)
    centres = (np.arange(n_bins) + 0.5) * bin_km
    interp = ~has
    if interp.any():
        h[interp] = np.interp(centres[interp], centres[has], h[has])
    return {"hbar": h, "raw": raw, "n_px": n.astype(np.int64), "interpolated": interp,
            "floored": floored}


def _geometry(inst):
    """``(wet (H, W) bool, s_frame (H, W) float, source)``: the permanently
    wet channel and s from the package geometry, else the instance's sea
    mask (declared)."""
    H, W = (int(v) for v in inst.shape)
    pkg = getattr(inst, "pkg", None)
    g = None
    for src in (getattr(inst, "geometry", None), getattr(pkg, "geometry", None)):
        if src is not None:
            g = src
            break
    s_frame = None
    if g is not None and "s_frame" in g:
        s_frame = np.asarray(g["s_frame"], np.float64).reshape(H, W)
    d = getattr(inst, "domain", None)
    if s_frame is None and getattr(d, "s_frame", None) is not None:
        s_frame = np.asarray(d.s_frame, np.float64).reshape(H, W)
    if s_frame is None:
        raise NotComputable("no_s", "no along-water distance on the frame (geometry s_frame)",
                            {"missing_input": "package geometry s_frame"})
    if g is not None and "wf" in g:
        wf = np.asarray(g["wf"], np.float64).reshape(H, W)
        with np.errstate(invalid="ignore"):
            wet = np.isfinite(wf) & (wf >= WET_MIN)
        return wet, s_frame, "geometry wf >= 0.98"
    sea = getattr(d, "sea", None)
    if sea is None:
        raise NotComputable("no_channel", "neither the geometry wf nor the domain sea mask",
                            {"missing_input": "package geometry wf"})
    return np.asarray(sea, bool).reshape(H, W), s_frame, "domain.sea (wf >= 0.90, declared)"


def _emodnet_dir(inst):
    """The site's EMODnet folder: where ``fetch emodnet`` writes it
    (``<root>/common/<site>/emodnet``), else ``inst.paths['emodnet_dir']`` (or
    its ``<site>`` sub-folder) when that holds the manifest."""
    from experiments.benchmark.fetch import emodnet

    paths = getattr(inst, "paths", None) or {}
    canon = emodnet.site_dir(inst.site, paths.get("root"))
    given = paths.get("emodnet_dir")
    cands = [canon] + ([given, os.path.join(given, str(inst.site))] if given else [])
    for d in cands:
        if os.path.isfile(os.path.join(d, emodnet.MANIFEST_NAME)):
            return str(d).replace("\\", "/")
    return str(given if given and not paths.get("root") else canon).replace("\\", "/")


def lat_offset(inst):
    """``(offset_m, mean_h, lat_ref, rule)``: MSL - LAT in h_B's datum for the
    window (mean of the window's 1-min h_B minus LAT_ref)."""
    w = settings.window(str(inst.window))
    t_min, h = inst.series1
    t_min = np.asarray(t_min, np.float64)
    h = np.asarray(h, np.float64)
    lo, hi = levels.minutes_since_epoch(pd.DatetimeIndex([w.start, w.end]))
    m = (t_min >= lo) & (t_min < hi) & np.isfinite(h)
    if not m.any():
        raise NotComputable("no_level", f"no 1-min h_B in {w.id}",
                            {"missing_input": "the boundary 1-min series over the window"})
    mean_h = float(np.mean(h[m]))
    if inst.site in LAT_REF_M:
        lat_ref, rule = float(LAT_REF_M[inst.site]), "CONFIG lat_ref_m (charted)"
    else:
        lat_ref, rule = float(np.min(h[m])), "min of the window's 1-min h_B"
    return mean_h - lat_ref, mean_h, lat_ref, rule


def run_l18(inst, unit, opts=None):
    """Unit L18 (see the module docstring); ``unit['variant']`` may name a
    card variant (site_mean, hw, lw)."""
    from experiments.benchmark.fetch import emodnet

    variant = str(models.unit_field(unit, "variant", ""))
    products = models.unit_products(unit, ("L18",))
    wet, s_frame, chan_src = _geometry(inst)
    edir = _emodnet_dir(inst)
    try:
        path, man = emodnet.verified_dtm(edir)
    except emodnet.EmodnetError as e:
        raise NotComputable("no_depths", str(e),
                            {"missing_input": f"{edir}/{emodnet.DTM_NAME}"}) from None
    with models._stage(inst, "dtm"):
        elev = emodnet.dtm_on_grid(path, inst.grid)
    if elev.shape != wet.shape:
        raise ValueError(f"the DTM on the grid has shape {elev.shape}, the frame {wet.shape}")
    offset, mean_h, lat_ref, lat_rule = lat_offset(inst)
    depth = offset - elev.astype(np.float64)                  # below MSL, m
    chan = wet & np.isfinite(s_frame) & np.isfinite(depth)
    if not chan.any():
        raise NotComputable("no_depths", "no permanently wet channel pixel has an EMODnet depth",
                            {"missing_input": f"EMODnet depths on the channel ({chan_src})"})
    keep, s_keep = _keep(inst)
    judges_in = models.in_frame_judges(inst)
    s_g = {str(r["code"]): (NAN if r.get("s_g_km") is None else float(r["s_g_km"]))
           for r in judges_in}
    s_all = np.concatenate([s_frame[chan], s_keep[np.isfinite(s_keep)],
                            np.asarray([v for v in s_g.values() if np.isfinite(v)], float)])
    n_bins = int(np.floor(np.nanmax(s_all) / BIN_KM)) + 1
    bd = bin_depths(s_frame[chan], depth[chan], n_bins)
    hbar = bd["hbar"].copy()
    a_m2 = None
    if variant == "site_mean":
        hbar[:] = max(float(np.mean(depth[chan])), DEPTH_FLOOR_M)
    elif variant in ("hw", "lw"):
        a_m2 = _a_m2(inst)
        hbar = np.maximum(hbar + (a_m2 if variant == "hw" else -a_m2), DEPTH_FLOOR_M)
    elif variant:
        raise ValueError(f"unknown L18 variant {variant!r} "
                         f"(card: {sorted(CARDS['L18']['variants'])})")
    field = tau_of_s(s_keep, hbar).astype(np.float32)
    s_chan_max = float(np.max(s_frame[chan]))
    point = {}
    for code, s in s_g.items():
        flags = []
        if not np.isfinite(s):
            flags.append("no s_g")
        elif s > s_chan_max:
            flags.append("s_g beyond the wet channel (depth extrapolated)")
        point[code] = _entry(float(tau_of_s(np.array([s]), hbar)[0]), flags,
                             model="celerity (EMODnet DTM 2024)", s_g_km=s)
    diag = {"window": str(inst.window), "variant": variant, "channel": chan_src,
            "n_channel_px": int(chan.sum()), "bin_km": BIN_KM, "g": G,
            "hbar_m": hbar.tolist(), "raw_mean_depth_m": bd["raw"].tolist(),
            "n_px_bin": bd["n_px"].tolist(), "interpolated_bins": np.flatnonzero(
                bd["interpolated"]).tolist(), "floored_bins": np.flatnonzero(
                bd["floored"]).tolist(), "lat_offset_m": offset, "mean_h_m": mean_h,
            "lat_ref_m": lat_ref, "lat_rule": lat_rule, "a_m2": a_m2,
            "dtm": {"path": path.replace("\\", "/"), "sha256": man["file"]["sha256"],
                    "coverage": man.get("coverage_id")},
            "s_channel_max_km": s_chan_max}
    r = common.lag_result("L18", "ok", kind="point", lag_kind="linear", point=point,
                          field=field, n_keep=len(keep), diag=diag)
    return [r for _ in products]


def _a_m2(inst):
    """The M2 amplitude of h_B over the window's 10-min series (variants hw/lw)."""
    w = settings.window(str(inst.window))
    idx10, h10 = inst.series10
    idx10 = pd.DatetimeIndex(idx10)
    m = np.asarray((idx10 >= pd.Timestamp(w.start)) & (idx10 < pd.Timestamp(w.end)))
    h = np.asarray(h10, np.float64)[m]
    ok = np.isfinite(h)
    cons = models.CONSTITUENTS or truth.CONS_B
    fit = truth.harmonic_fit(truth.days_since_1992(idx10[m][ok]), h[ok], cons)
    if fit is None:
        raise NotComputable("no_level", "too few h_B samples for the M2 amplitude",
                            {"missing_input": "the window's 10-min h_B"})
    return float(fit["m2"][0])


# ─────────────────────────────────────────────────────────────────────────────
#  keys
# ─────────────────────────────────────────────────────────────────────────────

def external_inputs(inst, unit=None):
    """L17: the judge table. L18: also the EMODnet manifest and tile."""
    out = {"judges.json": models.judges_sha(inst)}
    method = str(models.unit_field(unit, "method", ""))
    if method == "L18" or "L18" in models.unit_products(unit, ()):
        from experiments.benchmark.fetch import emodnet

        d = _emodnet_dir(inst)
        out["emodnet_manifest"] = models.file_sha(os.path.join(d, emodnet.MANIFEST_NAME))
        man = emodnet.read_manifest(d) or {}
        out["emodnet_dtm"] = (man.get("file") or {}).get("sha256")
        out["lat_ref_m"] = config.sha256_json(dict(LAT_REF_M))
    return out
