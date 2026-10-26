# -*- coding: utf-8 -*-
"""E5 - frequency to elevation: Flood2Topo (E5a), Tseng (E5b), sampled (E5c).

With f = wet / valid per pixel over the window's scenes (valid = clear with
a defined NDWI, wet = W of the common configuration) and the window's level
series (docs/validation_v2_spec.md s3.2.1, "E5 frequency-to-elevation"):

* **E5a Flood2Topo** (Fassoni-Andrade et al. 2020, 2021), exactly as
  ``cod_Flood2Topo.m``: the FULL level series of the window (h_B without lag,
  1 min) is sorted in descending order, Freq_k = 100 k / n (k = 1..n), and
  z = interp1(Freq, levels, 100 f) (linear, NaN outside [Freq_1, Freq_n]);
  pixels with f = 0 are NaN.
* **E5b Tseng et al. 2017** linear scaling: z = H_hi - f (H_hi - H_lo), with
  H_lo and H_hi the minimum and maximum of the same level series.
* **E5c sampled exceedance** (declared sensitivity of E5a, in the catalogue):
  z = the (1 - f) quantile (linear interpolation, numpy's default) of the
  levels at the pixel's OWN valid instants; an always-wet pixel gets its
  lowest sampled level and an always-dry pixel its highest.

E5a and E5b read the level series of ``cfg['level_series']``
(:class:`experiments.fixedcfg.level_series.LevelSeries`); without it they are
NOT COMPUTABLE (the series is an input, not something to approximate). E5c
needs only the scene levels.
"""
from __future__ import annotations

import time

import numpy as np

from experiments.fixedcfg import elev_base as base

_COMMON_DEVIATIONS = [
    {"paper": "the paper's own scene selection",
     "here": "MAREA's production transition-zone scene screening; the n most recent screened scenes",
     "why": "common configuration (one scene filter for every method)",
     "sensitivity": "not measured in these tables"},
    {"paper": "the paper's own pixel quality / water mask",
     "here": "valid = SCL {4, 5, 6, 7} with a defined NDWI; wet = NDWI > 0 (W)",
     "why": "common configuration (one cloud mask and one water detection)",
     "sensitivity": "not measured in these tables"},
    {"paper": "(not addressed)",
     "here": "a clear observation with undefined NDWI is dropped (MAREA's inversion reads it as dry; "
             "0.014 % of the clear observations at Villaviciosa)",
     "why": "a frequency needs a wet/dry reading; the undefined index has none",
     "sensitivity": "not measured (too few observations to matter)"},
]

# ─────────────────────────────────────────────────────────────────────────────
#  core arrays
# ─────────────────────────────────────────────────────────────────────────────


def frequency_block(V, Wb):
    """``(f, n_valid)``: wet share of the valid observations (NaN without any)."""
    nv = V.sum(axis=0)
    nw = Wb.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(nv > 0, nw / np.maximum(nv, 1), np.nan)
    return f, nv


def flood2topo(f, levels):
    """cod_Flood2Topo.m: z = interp1(Freq, sort(levels, 'descend'), 100 f) with
    Freq_k = 100 k / n; NaN where f = 0, f is NaN, or 100 f is outside
    [Freq_1, Freq_n] (interp1's default for out-of-range points)."""
    lev = np.sort(np.asarray(levels, float)[np.isfinite(levels)])[::-1]
    n = lev.size
    if n < 2:
        raise base.NotComputable("the level series has fewer than 2 finite values")
    freq = 100.0 * np.arange(1, n + 1, dtype=float) / n
    F = 100.0 * np.asarray(f, float)
    with np.errstate(invalid="ignore"):
        inside = (F >= freq[0]) & (F <= freq[-1]) & (np.asarray(f) > 0)
    z = np.full(F.shape, np.nan)
    z[inside] = np.interp(F[inside], freq, lev)
    return z


def tseng(f, h_lo, h_hi):
    """Tseng et al. 2017 linear scaling z = H_hi - f (H_hi - H_lo)."""
    return h_hi - np.asarray(f, float) * (h_hi - h_lo)


def sampled_exceedance_block(V, Wb, h):
    """E5c: the (1 - f) quantile (linear) of each pixel's valid levels."""
    T, Pb = V.shape
    hs = np.where(V, np.asarray(h, float)[:, None], np.inf)
    hs.sort(axis=0)
    m = V.sum(axis=0)
    k = Wb.sum(axis=0)
    z = np.full(Pb, np.nan)
    ok = m > 0
    if not ok.any():
        return z
    q = 1.0 - k[ok] / m[ok]
    pos = (m[ok] - 1) * q
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, m[ok] - 1)
    frac = pos - lo
    cols = np.flatnonzero(ok)
    a = hs[lo, cols]
    b = hs[hi, cols]
    z[ok] = a + frac * (b - a)
    return z


# ─────────────────────────────────────────────────────────────────────────────
#  E5a Flood2Topo
# ─────────────────────────────────────────────────────────────────────────────

ID_A = "E5a"
LABEL_A = "Flood2Topo (Fassoni-Andrade et al. 2020, 2021)"
DEFAULTS_A = {}
CARD_A = {
    "id": ID_A,
    "label": LABEL_A,
    "reference": ("Fassoni-Andrade, A.C., Paiva, R.C.D., Fleischmann, A.S. (2020). Lake topography and "
                  "active storage from satellite observations of flood frequency. Water Resources Research "
                  "56, e2019WR026362. Fassoni-Andrade, A.C., Paiva, R.C.D., Rudorff, C.M., Barbosa, "
                  "C.C.F., Novo, E.M.L.M. (2021). High-resolution mapping of floodplain topography from "
                  "space: A case study in the Amazon. Remote Sensing of Environment 251, 112065. Code "
                  "cod_Flood2Topo.m, doi:10.17632/jwxxjc5kyf.1"),
    "doi": "10.1029/2019WR026362; 10.1016/j.rse.2020.112065; code 10.17632/jwxxjc5kyf.1",
    "fidelity": ("F3 (re-implemented from the published algorithm as transcribed in "
                 "docs/validation_v2_spec.md s3.2.1; F2 needs the G-flood2topo gate, cod_Flood2Topo.m "
                 "run against this port to 1e-9, not run)"),
    "family": "water frequency -> exceedance level of the full level series",
    "rule": ("f = wet/valid; levels of the full 1-min h_B series of the window sorted descending; "
             "Freq_k = 100 k/n; z = interp1(Freq, levels, 100 f); NaN where f = 0"),
    "needs_full_frame": False,
    "needs": ["level_series"],
    "min_scenes": 1,
    "own_lag": False,
    "expected_offset": ("unbiased only if the scenes sample the level distribution of the full series; "
                        "Sentinel-2's fixed solar time samples the S2 tide at one phase, so the sampled "
                        "and the full distributions differ (Fitton et al. 2021; Sagar et al. 2018) and z "
                        "inherits that bias (E5c isolates it); resolution 1/n_valid in f"),
    "deviations": [
        {"paper": "water frequency from a Landsat / Global Surface Water occurrence archive",
         "here": "f over the window's screened Sentinel-2 scenes (valid = SCL 4-7 with defined NDWI)",
         "why": "common configuration",
         "sensitivity": "not measured in these tables"},
        {"paper": "water level series from gauges or altimetry over the imagery period",
         "here": "h_B without lag at a 1-min step over the window (production boundary)",
         "why": "common conditions: competitors use h_B without lag",
         "sensitivity": "E5c (levels at the scene instants only)"},
        {"paper": "no optical water index of its own (frequency product given)",
         "here": "NDWI > 0 (W)",
         "why": "common configuration",
         "sensitivity": "not measured in these tables"},
    ] + _COMMON_DEVIATIONS,
    "code": "experiments/fixedcfg/elev_freq.py",
    "cost": "one pass over (T, P) plus one sort of the level series",
}


def run_e5a(obs, cfg=None):
    t0 = time.time()
    Y, C, W, h, T, P = base.check_obs(obs)
    opts = base.method_options(cfg, ID_A, DEFAULTS_A)
    meta = base.base_meta(ID_A, T, P, opts)
    try:
        ls = base.require_series(obs, cfg)
        meta["level_series"] = ls.summary()
        f = np.full(P, np.nan)
        nv = np.zeros(P, np.int32)
        for sl in base.column_blocks(P, base.block_px(cfg)):
            V, Wb = base.valid_and_wet(Y, C, W, sl)
            f[sl], nv[sl] = frequency_block(V, Wb)
        z = flood2topo(f, ls.h)
    except base.NotComputable as e:
        return base.not_computable(meta, P, e, t0)
    meta["counts"] = {"f_zero": int(np.sum(f == 0)), "f_one": int(np.sum(f == 1)),
                      "no_valid_obs": int(np.sum(nv == 0))}
    return base.finish(meta, z, t0, extra={"f": f.astype(np.float32), "n_valid": nv})


# ─────────────────────────────────────────────────────────────────────────────
#  E5b Tseng
# ─────────────────────────────────────────────────────────────────────────────

ID_B = "E5b"
LABEL_B = "Tseng et al. 2017 linear scaling"
DEFAULTS_B = {}
CARD_B = {
    "id": ID_B,
    "label": LABEL_B,
    "reference": ("Tseng, K.-H., Kuo, C.-Y., Lin, T.-H., Huang, Z.-C., Lin, Y.-C., Liao, W.-H., Chen, "
                  "C.-F. (2017). Reconstruction of time-varying tidal flat topography using optical "
                  "remote sensing imageries. ISPRS Journal of Photogrammetry and Remote Sensing 131, "
                  "92-103"),
    "doi": "10.1016/j.isprsjprs.2017.07.008",
    "fidelity": "F3 (re-implemented from the paper as transcribed in docs/validation_v2_spec.md s3.2.1)",
    "family": "water frequency -> linear scaling between the lowest and highest level",
    "rule": "z = H_hi - f (H_hi - H_lo), H_lo / H_hi = min / max of the window's 1-min h_B series",
    "role": "catalogue variant of the frequency family (reported descriptively in validation v2)",
    "needs_full_frame": False,
    "needs": ["level_series"],
    "min_scenes": 1,
    "own_lag": False,
    "expected_offset": ("exact only where the level distribution is uniform; for a sinusoidal tide of "
                        "amplitude A the linear map departs from the exceedance curve by up to ~0.2 A "
                        "(z too high in the upper half of the range, too low in the lower half)"),
    "deviations": [
        {"paper": "H_lo and H_hi from DTU10 tidal amplitudes",
         "here": "minimum and maximum of h_B without lag at a 1-min step over the window",
         "why": "common conditions: competitors use h_B without lag",
         "sensitivity": "not measured in these tables"},
        {"paper": "MNDWI with threshold 0 (validation v2 as-published row)",
         "here": "NDWI > 0 (W)",
         "why": "common configuration",
         "sensitivity": "not measured in these tables"},
    ] + _COMMON_DEVIATIONS,
    "code": "experiments/fixedcfg/elev_freq.py",
    "cost": "one pass over (T, P)",
}


def run_e5b(obs, cfg=None):
    t0 = time.time()
    Y, C, W, h, T, P = base.check_obs(obs)
    opts = base.method_options(cfg, ID_B, DEFAULTS_B)
    meta = base.base_meta(ID_B, T, P, opts)
    try:
        ls = base.require_series(obs, cfg)
        fin = ls.finite()
        if fin.size < 2 or not fin.max() > fin.min():
            raise base.NotComputable("the level series has no range")
        h_lo, h_hi = float(fin.min()), float(fin.max())
    except base.NotComputable as e:
        return base.not_computable(meta, P, e, t0)
    meta["level_series"] = ls.summary()
    meta["H_lo_m"], meta["H_hi_m"] = h_lo, h_hi
    f = np.full(P, np.nan)
    nv = np.zeros(P, np.int32)
    for sl in base.column_blocks(P, base.block_px(cfg)):
        V, Wb = base.valid_and_wet(Y, C, W, sl)
        f[sl], nv[sl] = frequency_block(V, Wb)
    z = tseng(f, h_lo, h_hi)
    meta["counts"] = {"no_valid_obs": int(np.sum(nv == 0))}
    return base.finish(meta, z, t0, extra={"f": f.astype(np.float32), "n_valid": nv})


# ─────────────────────────────────────────────────────────────────────────────
#  E5c sampled exceedance
# ─────────────────────────────────────────────────────────────────────────────

ID_C = "E5c"
LABEL_C = "sampled exceedance (frequency at the pixel's own instants)"
DEFAULTS_C = {}
CARD_C = {
    "id": ID_C,
    "label": LABEL_C,
    "reference": ("Declared sensitivity of E5a in docs/validation_v2_spec.md s3.2 (the frequency mapped "
                  "onto the levels the pixel was actually observed at, instead of the full series)"),
    "doi": None,
    "fidelity": "F3 (variant defined in the validation spec; no paper of its own)",
    "family": "water frequency -> quantile of the sampled levels",
    "rule": ("z = the (1 - f) quantile (linear) of the h_B levels at the pixel's own valid instants; "
             "always wet -> lowest sampled level, always dry -> highest"),
    "role": "variant of E5a (isolates the tidal-sampling bias of the full-series mapping)",
    "needs_full_frame": False,
    "needs": [],
    "min_scenes": 1,
    "own_lag": False,
    "expected_offset": ("in a step world the value lies inside the pixel's observed bracket "
                        "[highest dry level, lowest wet level]; always-wet / always-dry pixels are "
                        "clamped to the sampled range"),
    "deviations": [
        {"paper": "E5a maps f onto the full level series",
         "here": "onto the levels of the pixel's own valid scenes",
         "why": "declared sensitivity of E5a (tidal sampling of a sun-synchronous sensor)",
         "sensitivity": "E5a is the reference row"},
    ] + _COMMON_DEVIATIONS,
    "code": "experiments/fixedcfg/elev_freq.py",
    "cost": "one sort of (T, P) per block",
}


def run_e5c(obs, cfg=None):
    t0 = time.time()
    Y, C, W, h, T, P = base.check_obs(obs)
    opts = base.method_options(cfg, ID_C, DEFAULTS_C)
    meta = base.base_meta(ID_C, T, P, opts)
    z = np.full(P, np.nan)
    f = np.full(P, np.nan)
    nv = np.zeros(P, np.int32)
    for sl in base.column_blocks(P, base.block_px(cfg)):
        V, Wb = base.valid_and_wet(Y, C, W, sl)
        f[sl], nv[sl] = frequency_block(V, Wb)
        z[sl] = sampled_exceedance_block(V, Wb, h)
    meta["counts"] = {"f_zero_clamped_high": int(np.sum(f == 0)), "f_one_clamped_low": int(np.sum(f == 1)),
                      "no_valid_obs": int(np.sum(nv == 0))}
    return base.finish(meta, z, t0, extra={"f": f.astype(np.float32), "n_valid": nv})


METHODS = {ID_A: (LABEL_A, run_e5a, CARD_A),
           ID_B: (LABEL_B, run_e5b, CARD_B),
           ID_C: (LABEL_C, run_e5c, CARD_C)}
