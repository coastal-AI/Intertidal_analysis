# -*- coding: utf-8 -*-
"""Validation v2, Stage L -- L08 GM20: per-pixel 12.42-h sinusoid phase (Geyman & Maloof 2020).

Question
--------
Geyman & Maloof (2020, Earth and Space Science 7:e2019EA000958) fold every
pixel's image time series on the hours since the reference station's high
water and fit one sinusoid of the M2 period: the phase of the crest is the
local tide's delay. How far behind h_B does each intertidal pixel run?

Rules (spec §3.3.1 "L08 GM20"; v13 §3.5; FROZEN['gm20'], repeated in CONFIG)
---------------------------------------------------------------------------
* x_i = hours since the latest HW mark of h_B (``inst.marks_lp``: the HW marks
  of the 61-min low-passed 1-min boundary series) at the scene instant t_i;
  an instant before the first mark is unusable;
* per pixel, over its valid instants (V): the closed-form least squares
  y = c + a cos(2 pi x / P) + b sin(2 pi x / P), P = 12.4206012 h, on Y (the
  setting's standardised, water-high index), vectorised as per-pixel 3 x 3
  normal equations; phase = 60 P atan2(b, a) / (2 pi) minutes (a crest x0
  hours after HW gives +60 x0; the same optimum as the paper's sinusoid fit,
  ``truth.fold_phase`` is the reference implementation of one pixel);
* ``rel`` (primary): minus the same fit of h0 = h_B(t) on the pixel's own
  valid instants (h_B's own fold phase), wrapped with ``truth.wrap_gm20`` to
  [-120, 625.24) min; lag_kind 'phase' (the scorer wraps about 67.5 min);
* QC (pre-registered): n >= 20 valid instants, SD(Y) > 0.2 (ddof 0, Y
  units), 0.05 < wet share (W among V) < 0.95, R^2 >= 0.1; otherwise NaN;
* uncertainty (variant ``boot30`` of the lag design, Stage L at P0 only):
  20 refits on 70 % scene subsets, subset k = the first m = ceil(0.7 T) of
  ``default_rng([SEEDS['gm20'], SITE_ID, window_num, k]).permutation(T)``;
  per pixel the IQR of the refit phases circularly unwrapped about the full
  fit (GM20's published uncertainty); interval95 = tau +- (IQR / 2)(1.96 /
  0.674) sqrt(m / (T - m)), nominal 'iqr20x1.96/0.674xsqrt(m/(n-m))'
  (declared: the normal rescaling of an IQR, and the m-out-of-n subsample
  correction: the refits of m of n scenes spread as sd(theta_m - theta_n)
  = sd(theta_n) sqrt(n/m - 1), so the full-sample standard error is the
  spread x sqrt(m/(n - m)); revision of 2026-10-20 before lock-A, found by
  the code review's coverage check: 74 % nominal-95 % coverage without it,
  90 % with it). The 20 refit fields are stored (``diag['refit_tau']``,
  float32, NaN outside the QC-passed pixels) so the scorer forms the same
  interval AT THE READ-OUT (``scoring.interval_at_readout``);
* the NIR rule of the spec ("for NIR-type indices, P/2 is added") is NOT
  applied: the paper fits an intensity that darkens as the water deepens, so
  its crest marks LOW water and a half period turns it into the high-water
  phase; here Y is oriented water-high for every index (spec §2.4: I3 NIR is
  -rho8 and Y = (X - t_k)/s_k with s_k > 0), so the crest of Y already marks
  high water and adding P/2 would count the sign twice. The paper's full text
  could not be fetched from the build machine on 2026-10-19 (publisher 403);
  the decision rests on the orientation of Y, which holds whatever band the
  paper used, and is re-checked against the paper before lock-A (spec
  §3.3.1). Every run records ``half_period_shift_applied = False``.
* variants 'raw', 'scaled', 'wet' and 'own' are on the card only (not
  scheduled in V2); a unit asking for them is refused.

Memory: the pixels stream through ``inst.obs.iter_blocks(100_000)`` and are
fitted in chunks of about 2.5 million cells (T x columns, 256 to 20,000
columns); no (T, n_keep) array is ever held. With the refits, the
(20, n_keep) float32 refit fields are (80 bytes per keep pixel: about
240 MB at 3 million keep pixels).
"""
from __future__ import annotations

import math
import warnings
import zlib
from fractions import Fraction

import numpy as np

from experiments.benchmark import config, settings, truth
from experiments.benchmark.lags import aggregate, common

_G = config.FROZEN["gm20"]

CONFIG = {
    "period_h": float(_G["period_h"]),
    "reference": str(_G["reference"]),
    "primary": str(_G["primary"]),
    "qc": {"min_n": int(_G["qc"]["min_n"]), "min_sd": float(_G["qc"]["min_sd"]),
           "wet_lo": float(_G["qc"]["wet_lo"]), "wet_hi": float(_G["qc"]["wet_hi"]),
           "min_r2": float(_G["qc"]["min_r2"])},
    "wrap_lo_min": float(_G["wrap_lo_min"]),
    "iqr": {"refits": int(_G["iqr"]["refits"]), "share": float(_G["iqr"]["share"]),
            "seed": int(_G["iqr"]["seed"]),
            "rng": "default_rng([seed, SITE_ID, window_num, k]).permutation(T)[:ceil(share T)]",
            "scale_to_95": "1.96/0.674", "subsample_factor": "sqrt(m / (T - m))",
            "percentiles": "numpy linear, 25 and 75",
            "refit_fields": "diag['refit_tau'] (n_refits, n_keep) float32: the interval at "
                            "the scorer's read-out"},
    "nir_half_period": "not applied (Y is oriented water-high for every index)",
    "solve": "per-pixel 3x3 normal equations; singular when n < 3 or det(A/n) <= 1e-10",
    "block": 100_000,
    "chunk_cells": 2_500_000,
    "chunk_cols": [256, 20_000],
}
del _G

CODE_DEPS = ("experiments/benchmark/truth.py", "experiments/benchmark/lags/aggregate.py")

_SCALE95 = 1.96 / 0.674
_DET_MIN = 1e-10
_CARD_ONLY = ("raw", "scaled", "wet", "own")
#: the nominal of L08's interval (common.INTERVAL_NOMINALS)
NOMINAL = "iqr20x1.96/0.674xsqrt(m/(n-m))"


def subsample_factor(m, n):
    """sqrt(m / (n - m)): the full-sample standard error over the spread of
    refits on m of n scenes (NaN when m >= n)."""
    m, n = float(m), float(n)
    return math.sqrt(m / (n - m)) if n > m > 0 else float("nan")

CARDS = {"L08": {
    "id": "L08",
    "name": "GM20: per-pixel 12.42-h sinusoid phase (Geyman & Maloof 2020), rel",
    "family": "harmonic",
    "kind": "lag",
    "tier": "1",
    "fidelity": "F3",
    "evidence": ("F3 re-implementation from the paper (no authors' code): unit tests of every "
                 "step on synthetic tides (tests/test_benchmark_lags_gm20.py: the closed form "
                 "equals truth.fold_phase on a pixel, rel cancels a common phase, planted "
                 "0/45/120/200 min recovered, each QC rule), the positive control of spec "
                 "§10.2; optional NW Andros check against PANGAEA (doi:10.1594/PANGAEA.911255) "
                 "not run in V2"),
    "references": [
        {"cite": "Geyman EC, Maloof AC (2020) Deriving tidal structure from satellite image "
                 "time series. Earth and Space Science 7:e2019EA000958",
         "doi_or_url": "doi:10.1029/2019EA000958"},
        {"cite": "Geyman & Maloof (2020) data: tidal phase and amplitude, NW Andros Island",
         "doi_or_url": "doi:10.1594/PANGAEA.911255"},
    ],
    "output": "field",
    "lag_kind": "phase",
    "signature": aggregate.card_signature("L08"),
    "scan_window": None,
    "published_window": None,
    "as_published": ("per-pixel least-squares sinusoid of period 12.42 h on the hours since the "
                     "reference station's high water, phase offset = tidal lag; uncertainty = "
                     "IQR over 20 refits on random 70 % subsets; validated within 500 m of "
                     "gauges"),
    "native_cell": None,
    "common_conditions": ("Y of the instance's setting (NDWI at P0) on the harness domain keep "
                          "pixels; reference HW = HW marks of the low-passed h_B (marks_lp); "
                          "closed-form OLS; rel = minus h_B's own fold phase at the pixel's "
                          "instants; pre-registered QC"),
    "deviations": [
        {"paper": "pixel intensity of the imagery (water-dark)",
         "ours": "the setting's standardised index Y, oriented water-high",
         "why": "the index is a setting applied to every method (spec §0.3, item 9)",
         "rows": "L08", "sensitivity": "'wet' variant (binary W; card only, not scheduled)"},
        {"paper": "high waters of a reference tide station",
         "ours": "HW marks of the low-passed boundary series h_B (marks_lp)",
         "why": "h_B is the one boundary every method receives; in-estuary gauges are the truth",
         "rows": "L08", "sensitivity": "none"},
        {"paper": "phase relative to the reference station's high water",
         "ours": "rel: minus h_B's own fold phase at the pixel's valid instants",
         "why": "the lag relative to h_B (the scorer's convention); removes the phase that "
                "fixed solar-time sampling and the mark definition put into the fold",
         "rows": "L08", "sensitivity": "'raw' variant (card only, not scheduled)"},
        {"paper": "+P/2 for intensities that darken with depth (NIR)",
         "ours": "never added: Y is water-high for every index (I3 NIR = -rho8)",
         "why": "the crest of Y already marks high water; adding P/2 would count the sign twice",
         "rows": "L08", "sensitivity": "none (a P/2 error would be obvious in the control)"},
        {"paper": "every pixel, sea included; no spring-neap scaling stated as primary",
         "ours": "harness domain keep pixels; no spring-neap scaling",
         "why": "the read-out ladder needs the common domain; scaling is a variant",
         "rows": "L08", "sensitivity": "'own' and 'scaled' variants (card only)"},
        {"paper": "the IQR of the refits as the uncertainty",
         "ours": ("tau +- (IQR/2)(1.96/0.674) sqrt(m/(n - m)) as a nominal 95 % interval, "
                  "formed by the scorer at the read-out from the 20 stored refit fields"),
         "why": ("the interval score needs a nominal 95 % interval (spec §6.4); refits on m of "
                 "n scenes spread less than the full-sample error by sqrt((n - m)/m) "
                 "(revision 2026-10-20, before lock-A)"),
         "rows": "L08 boot30", "sensitivity": "none"},
    ],
    "qc": {"min_n": 20, "min_sd": 0.2, "wet_lo": 0.05, "wet_hi": 0.95, "min_r2": 0.1},
    "n_min": 20,
    "tuning": [],
    "control_offset_min": 0.0,
    "control_offset_why": ("0 min: in the gating world (spec §10.2) every crest of a pixel's "
                           "index follows the boundary HW by the planted lag, and rel removes "
                           "h_B's own fold phase"),
    "cost": aggregate.card_cost("L08"),
    "variants": {"rel": "primary",
                 "boot30": "the 20 refits on 70 % subsets and the interval (lag design, W3)",
                 "raw": "phase relative to the HW marks, no h_B subtraction (card only)",
                 "scaled": "spring-neap scaled amplitude (card only)",
                 "wet": "fit on the binary W instead of Y (card only)",
                 "own": "every package pixel in the judges' neighbourhoods, sea included "
                        "(card only)"},
}}


# ─────────────────────────────────────────────────────────────────────────────
#  the fold and the vectorised fit
# ─────────────────────────────────────────────────────────────────────────────

def hours_since_hw(t_min, marks):
    """Hours since the latest HW mark at each instant (NaN before the first)."""
    t = np.asarray(t_min, np.float64)
    hw = np.sort(np.asarray(marks["t"], np.float64)[np.asarray(marks["is_hw"], bool)])
    if not len(hw):
        return np.full(t.shape, np.nan)
    i = np.searchsorted(hw, t, side="right") - 1
    x = np.full(t.shape, np.nan)
    ok = i >= 0
    x[ok] = (t[ok] - hw[i[ok]]) / 60.0
    return x


def fold_design(x_hours, period_h=None):
    """``(C, ok)``: the design [1, cos(2 pi x / P), sin(2 pi x / P)] (T, 3),
    zero rows where x is not finite, and the finite-x mask."""
    period_h = float(CONFIG["period_h"] if period_h is None else period_h)
    x = np.asarray(x_hours, np.float64)
    ok = np.isfinite(x)
    w = 2.0 * np.pi * np.where(ok, x, 0.0) / period_h
    C = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w)])
    C[~ok] = 0.0
    return C, ok


def _normal(C, Vf):
    """Per-column normal matrices (m, 3, 3) of the masked design."""
    P = np.column_stack([C[:, 0] * C[:, 0], C[:, 0] * C[:, 1], C[:, 0] * C[:, 2],
                         C[:, 1] * C[:, 1], C[:, 1] * C[:, 2], C[:, 2] * C[:, 2]])
    S = Vf.T @ P
    A = np.empty((S.shape[0], 3, 3))
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = S[:, 0], S[:, 1], S[:, 2]
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = S[:, 1], S[:, 3], S[:, 4]
    A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = S[:, 2], S[:, 4], S[:, 5]
    return A


def _solve(A, b, n):
    """Solve A beta = b per column; NaN where A is singular (n < 3 or
    det(A / n) <= 1e-10)."""
    m = A.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        det = np.linalg.det(A / np.maximum(n, 1.0)[:, None, None])
    ok = (n >= 3) & np.isfinite(det) & (det > _DET_MIN)
    As = np.where(ok[:, None, None], A, np.eye(3)[None, :, :])
    bs = np.where(ok[:, None], b, 0.0)
    beta = np.linalg.solve(As, bs[:, :, None])[:, :, 0]
    beta[~ok] = np.nan
    return beta, ok


def phase_min(beta, period_h=None):
    """Crest phase (min) of a fitted [c, a, b]: 60 P atan2(b, a) / (2 pi)."""
    period_h = float(CONFIG["period_h"] if period_h is None else period_h)
    return 60.0 * period_h * np.arctan2(beta[:, 2], beta[:, 1]) / (2.0 * np.pi)


def _masked(Y, V):
    """(Vf, Yv): the valid mask as float64 and Y with zeros where not valid."""
    V = np.asarray(V, bool)
    return V.astype(np.float64), np.where(V, np.asarray(Y, np.float64), 0.0)


def _core(C, Vf, Yv, h0=None, rows=None):
    """The fit of every column on its valid instants (and, with ``rows``, only
    on the scenes of that subset: the design rows outside it are zeroed, so
    no (T, m) array is rebuilt for a refit). Returns (out, beta, ok, n)."""
    if rows is not None:
        rf = np.asarray(rows, np.float64)
        C = C * rf[:, None]
        n = rf @ Vf
    else:
        n = Vf.sum(axis=0)
    A = _normal(C, Vf)
    beta, ok = _solve(A, Yv.T @ C, n)
    out = {"n": np.rint(n).astype(np.int64), "phase_y": phase_min(beta),
           "amp": np.hypot(beta[:, 1], beta[:, 2]), "solved": ok}
    if h0 is not None:
        bh = Vf.T @ (C * np.asarray(h0, np.float64)[:, None])
        beta_h, _ = _solve(A, bh, n)
        out["phase_h"] = phase_min(beta_h)
    return out, beta, ok, n


def fold_fit(C, Y, V, h0=None, W=None, stats=True, rows=None):
    """GM20's fit of every column of a block.

    Parameters
    ----------
    C : (T, 3) design of :func:`fold_design` (zero rows where x is unusable)
    Y : (T, m) index values (any value where not V)
    V : (T, m) bool, the valid instants (already AND-ed with the usable x)
    h0 : (T,) h_B(t), optional: also fit h0 on each column's instants
    W : (T, m) bool, optional: the wet layer (wet share)
    stats : also R^2 and SD (the QC); False for the refits
    rows : (T,) bool, optional: fit on this scene subset only (the refits)

    Returns
    -------
    dict of (m,) arrays: phase_y, phase_h (with h0), amp, n, solved, and with
    stats r2, sd, wet (with W).
    """
    Vf, Yv = _masked(Y, V)
    out, beta, ok, n = _core(C, Vf, Yv, h0=h0, rows=rows)
    if stats:
        if rows is not None:
            raise ValueError("the QC statistics are those of the full fit (rows=None)")
        with np.errstate(invalid="ignore", divide="ignore"):
            ybar = Yv.sum(axis=0) / np.maximum(n, 1.0)
            fit = C @ np.where(ok[:, None], beta, 0.0).T                 # (T, m)
            sse = np.sum(Vf * (Yv - fit) ** 2, axis=0)
            sst = np.sum(Vf * (Yv - ybar[None, :]) ** 2, axis=0)
            out["r2"] = np.where(ok & (sst > 0), 1.0 - sse / sst, np.nan)
            out["sd"] = np.where(n > 0, np.sqrt(sst / np.maximum(n, 1.0)), np.nan)
            if W is not None:
                out["wet"] = np.where(n > 0, np.sum(np.asarray(W, bool) & np.asarray(V, bool),
                                                    axis=0) / np.maximum(n, 1.0), np.nan)
    return out


def qc_mask(fit):
    """GM20's pre-registered QC on a :func:`fold_fit` result."""
    q = CONFIG["qc"]
    with np.errstate(invalid="ignore"):
        return {"n": fit["n"] >= q["min_n"],
                "sd": fit["sd"] > q["min_sd"],
                "wet": (fit["wet"] > q["wet_lo"]) & (fit["wet"] < q["wet_hi"]),
                "r2": fit["r2"] >= q["min_r2"],
                "solved": fit["solved"]}


def refit_masks(T, site, window, n_refits, share=None, seed_offset=0):
    """The scene subsets of the refits: (n_refits, T) bool and their size."""
    share = float(CONFIG["iqr"]["share"] if share is None else share)
    n_sub = int(math.ceil(Fraction(str(share)) * int(T)))
    site_id = int(config.SITE_ID[site])
    try:
        wnum = int(settings.window_num(window))
    except KeyError:
        wnum = int(zlib.crc32(str(window).encode("utf-8")))
    out = np.zeros((int(n_refits), int(T)), bool)
    for k in range(int(n_refits)):
        seq = [int(CONFIG["iqr"]["seed"]), site_id, wnum, k]
        if int(seed_offset):
            seq.append(int(seed_offset))
        perm = np.random.default_rng(seq).permutation(int(T))
        out[k, perm[:n_sub]] = True
    return out, n_sub


# ─────────────────────────────────────────────────────────────────────────────
#  the unit
# ─────────────────────────────────────────────────────────────────────────────

def run_unit(inst, unit, opts=None):
    """L08 GM20 on the instance's keep pixels (module docstring)."""
    opts = dict(opts or {})
    aggregate.require_domain(inst)
    products = aggregate.unit_products(unit, ("L08",))
    parts = aggregate.variant_parts(unit)
    card_only = [p for p in parts if p in _CARD_ONLY]
    if card_only:
        raise ValueError(f"GM20 variants {card_only} are listed on the card but not "
                         "scheduled in milestone V2")
    n_refits = int(opts.get("refits", 0) or 0)
    if "boot30" in parts and n_refits == 0:
        n_refits = int(CONFIG["iqr"]["refits"])
    seed_offset = int(opts.get("seed_offset", 0) or 0)
    T = int(inst.obs.T)
    n_keep = aggregate._n_keep(inst)
    t_min = aggregate.t_minutes(inst.t)
    x = hours_since_hw(t_min, inst.marks_lp)
    C, x_ok = fold_design(x)
    h0 = np.asarray(inst.h0, np.float64)
    subsets, n_sub = (refit_masks(T, inst.site, inst.window, n_refits, seed_offset=seed_offset)
                      if n_refits else (np.zeros((0, T), bool), 0))
    tau = np.full(n_keep, np.nan, np.float32)
    iqr = np.full(n_keep, np.nan, np.float32) if n_refits else None
    refit = np.full((n_refits, n_keep), np.nan, np.float32) if n_refits else None
    fails = {"n": 0, "sd": 0, "wet": 0, "r2": 0, "solved": 0}
    n_enough = 0
    r2_kept, ph_h_kept = [], []
    chunk = aggregate.chunk_cols(inst.obs.T, CONFIG["chunk_cells"], CONFIG["chunk_cols"])
    with aggregate._stage(inst, "fit"):
        for cols, ob in inst.obs.iter_blocks(int(CONFIG["block"])):
            cols = np.asarray(cols, np.int64)
            for a in range(0, len(cols), chunk):
                b = min(a + chunk, len(cols))
                V = np.asarray(ob.V[:, a:b], bool) & x_ok[:, None]
                Y = ob.Y[:, a:b]
                W = np.asarray(ob.W[:, a:b], bool)
                fit = fold_fit(C, Y, V, h0=h0, W=W)
                q = qc_mask(fit)
                good = np.ones(b - a, bool)
                for name in ("n", "solved", "sd", "wet", "r2"):
                    fails[name] += int(np.sum(good & ~q[name]))
                    good &= q[name]
                n_enough += int(np.sum(q["n"]))
                rel = truth.wrap_gm20(fit["phase_y"] - fit["phase_h"])
                tau[cols[a:b][good]] = rel[good].astype(np.float32)
                r2_kept.append(fit["r2"][good])
                ph_h_kept.append(fit["phase_h"][good])
                if n_refits and good.any():
                    Vf_g, Yv_g = _masked(Y[:, good], V[:, good])
                    d = np.full((n_refits, int(good.sum())), np.nan)
                    gcols = cols[a:b][good]
                    for k in range(n_refits):
                        fk = _core(C, Vf_g, Yv_g, h0=h0, rows=subsets[k])[0]
                        rk = truth.wrap_gm20(fk["phase_y"] - fk["phase_h"])
                        d[k] = common.circ_diff(rk, rel[good])
                        refit[k, gcols] = np.asarray(rk, np.float32)
                    del Vf_g, Yv_g
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        q25, q75 = np.nanpercentile(d, [25.0, 75.0], axis=0)
                    iqr[cols[a:b][good]] = (q75 - q25).astype(np.float32)
                del V, W, fit
            aggregate._progress(inst, len(cols))
    r2_kept = np.concatenate(r2_kept) if r2_kept else np.zeros(0)
    ph_h_kept = np.concatenate(ph_h_kept) if ph_h_kept else np.zeros(0)
    n_def = int(np.isfinite(tau).sum())
    diag = {"n_keep": n_keep, "n_defined": n_def, "n_with_min_n": n_enough,
            "qc_fail_first": fails, "qc": dict(CONFIG["qc"]),
            "median_r2_defined": float(np.median(r2_kept)) if r2_kept.size else float("nan"),
            "median_phase_h0_min": float(np.median(ph_h_kept)) if ph_h_kept.size
            else float("nan"),
            "n_instants_before_first_hw": int(np.sum(~x_ok)),
            "nir_type": bool(getattr(inst, "nir_type", False)),
            "half_period_shift_applied": False,
            "n_refits": n_refits, "refit_subset_size": int(n_sub), "T": T,
            "seed_offset": seed_offset}
    iv = None
    if n_refits:
        fac = subsample_factor(n_sub, T)
        half = iqr.astype(np.float64) / 2.0 * _SCALE95 * fac
        lo = np.where(np.isfinite(tau), tau.astype(np.float64) - half, np.nan)
        hi = np.where(np.isfinite(tau), tau.astype(np.float64) + half, np.nan)
        iv = {"level": "pixel", "lo": lo, "hi": hi, "nominal": NOMINAL}
        fin = np.isfinite(iqr)
        diag["median_iqr_min"] = float(np.median(iqr[fin])) if fin.any() else float("nan")
        diag["subsample_factor"] = fac
        diag["refit_tau"] = refit
    if n_def:
        res = common.lag_result("L08", "ok", n_keep=n_keep, kind="field", lag_kind="phase",
                                field=tau, interval95=iv, edge_share=0.0, diag=diag)
    elif n_enough == 0:
        res = common.lag_result("L08", "not_computable", n_keep=n_keep, kind="field",
                                lag_kind="phase", stage=f"n_lt_{CONFIG['qc']['min_n']}",
                                reason="no keep pixel has 20 valid instants", diag=diag)
    else:
        res = common.lag_result("L08", "empty", n_keep=n_keep, kind="field", lag_kind="phase",
                                stage="qc", reason="every pixel failed the QC", diag=diag)
    return [res for _ in products]
