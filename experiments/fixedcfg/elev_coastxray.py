# -*- coding: utf-8 -*-
"""E7 - Coast X-Ray (Fitton et al. 2021): stage-interval composites, minimum reducer.

Rule (docs/validation_v2_spec.md s3.2.1, card "E7 Coast X-Ray"):

* tidal stage of every image, exactly as Eq. 1 is printed:
  stage % = (TE + MHWS) / (MHWS - MLWS) x 100, TE = h_B at the image time
  (no lag), MHWS / MLWS = Z0 +/- (M2 + S2 amplitudes) from a harmonic
  analysis of the window's 1-min h_B series (declared);
* 14 stage bins: < -10, [-10, 0), [0, 10), ..., [100, 110), >= 110 %;
* a median composite of NDWI per bin, only for bins holding >= 3 images
  (per pixel over its valid observations in the bin; the exact median of
  :func:`pyintertidal.waterline_contour.interval_statistics`, the NIDEM
  composite statistic);
* water where the composite > 0.2 (the paper's NDWI threshold, its own
  composite rule applied on the common NDWI);
* elevation = per pixel, the MINIMUM over the bins where it is water of the
  mean tide of that bin's images. NaN where no bin is water.

The sensitivity stage = (TE - MLWS) / (MHWS - MLWS) is the 'corrected'
formula; the common-threshold variant classifies the composites at 0 (the
common configuration's NDWI > 0). Both are VARIANTS of the registry.

All of it is per pixel (the composites are per-pixel medians over time), so
no frame is needed.
"""
from __future__ import annotations

import time

import numpy as np

from experiments.fixedcfg import elev_base as base
from experiments.fixedcfg.level_series import harmonic_constants

ID = "E7"
LABEL = "Coast X-Ray (Fitton et al. 2021)"
#: stage bin edges in percent: 13 edges -> 14 bins (< -10 ... >= 110)
STAGE_EDGES = tuple(float(v) for v in range(-10, 111, 10))
DEFAULTS = {"threshold": 0.2, "min_images": 3, "stage_formula": "printed", "block_px": 50_000}
STAGE_FORMULAS = ("printed", "corrected")

CARD = {
    "id": ID,
    "label": LABEL,
    "reference": ("Fitton, J.M., Rennie, A.F., Hansom, J.D., Muir, F.M.E. (2021). Remotely sensed mapping "
                  "of the intertidal zone: A Sentinel-2 and Google Earth Engine methodology. Remote Sensing "
                  "Applications: Society and Environment 22, 100499"),
    "doi": "10.1016/j.rsase.2021.100499",
    "fidelity": ("F3 (re-implemented from the paper as transcribed in docs/validation_v2_spec.md "
                 "s3.2.1; unit tests of Eq. 1, the >= 3-image rule and the minimum reducer)"),
    "family": "tidal-stage interval composites, minimum reducer",
    "rule": ("stage % = (TE + MHWS)/(MHWS - MLWS) x 100 (Eq. 1 as printed); 14 bins of 10 %; median NDWI "
             "composite per bin with >= 3 images; water = composite > 0.2; z = min over water bins of the "
             "bin's mean tide"),
    "needs_full_frame": False,
    "needs": ["level_series"],
    "min_scenes": 3,
    "own_lag": False,
    "expected_offset": ("upward: the minimum reducer returns the mean tide of the lowest bin in which "
                        "the pixel is water, so z exceeds the elevation by about half a bin on average "
                        "(bin = 10 % of MHWS - MLWS) and by up to about one bin (two at most, where the "
                        "levels inside a bin are skewed); the 0.2 threshold moves the crossing further up "
                        "where wet NDWI is below 0.2"),
    "deviations": [
        {"paper": "Sentinel-2 L1C TOA with the QA60 cloud mask, Google Earth Engine",
         "here": "Sentinel-2 L2A (offset-harmonised) with the per-pixel clear mask SCL 4-7",
         "why": "common configuration",
         "sensitivity": "not measured in these tables"},
        {"paper": "image coverage rule (>= 99 % of the AOI, relaxed to keep >= 30 images, never below 90 %)",
         "here": "MAREA's production transition-zone screening; the n most recent screened scenes",
         "why": "common configuration (one scene filter for every method)",
         "sensitivity": "not measured in these tables"},
        {"paper": "five years of imagery",
         "here": "the window of the n most recent screened scenes",
         "why": "the design of the tables (error against n)",
         "sensitivity": "the n curve itself"},
        {"paper": "POLPRED CS20 tide at the water centroid of each 96 km2 hexagon; MHWS / MLWS of the model",
         "here": "h_B without lag (production boundary); MHWS / MLWS = Z0 +/- (M2 + S2) from a harmonic "
                 "analysis (no nodal corrections, Rayleigh-selected constituents) of the window's 1-min h_B",
         "why": "common conditions: competitors use h_B without lag",
         "sensitivity": "not measured in these tables"},
        {"paper": "GEE median reducer (histogram-based)",
         "here": "exact per-pixel median (numpy; NIDEM's interval_statistics)",
         "why": "exact and deterministic",
         "sensitivity": "not measurable"},
        {"paper": "water = NDWI > 0.2 (the paper's threshold)",
         "here": "kept on the median composites; the common configuration's per-observation detector "
                 "(NDWI > 0) is not used by this method",
         "why": "the composite threshold is the method's own rule (validation v2 applies it on Y)",
         "sensitivity": "VARIANT E7_t0 (composite threshold 0)"},
        {"paper": "Eq. 1 printed as (TE + MHWS)/(MHWS - MLWS)",
         "here": "as printed (identical to (TE - MLWS)/(MHWS - MLWS) when Z0 = 0)",
         "why": "transcription as published",
         "sensitivity": "VARIANT E7_corrected"},
        {"paper": "(not addressed)",
         "here": "a clear observation with undefined NDWI is dropped from the composites",
         "why": "the median needs a defined index",
         "sensitivity": "not measured (0.014 % of the clear observations at Villaviciosa)"},
    ],
    "code": "experiments/fixedcfg/elev_coastxray.py",
    "cost": "14 per-pixel medians of (images per bin, P) arrays; one harmonic analysis of the 1-min series",
}


def stage_percent(te, mhws, mlws, formula="printed"):
    """Coast X-Ray Eq. 1: (TE + MHWS)/(MHWS - MLWS) x 100 as printed; the
    'corrected' sensitivity uses (TE - MLWS)."""
    te = np.asarray(te, float)
    rng = float(mhws) - float(mlws)
    if not rng > 0:
        raise base.NotComputable(f"MHWS - MLWS = {rng:.4f} m is not positive")
    if formula == "printed":
        return (te + mhws) / rng * 100.0
    if formula == "corrected":
        return (te - mlws) / rng * 100.0
    raise ValueError(f"stage_formula must be one of {STAGE_FORMULAS}")


def stage_bins(stage):
    """Bin index 0..13 of every stage: 0 = < -10 %, 1 = [-10, 0), ...,
    12 = [100, 110), 13 = >= 110 % (left-closed bins)."""
    return np.digitize(np.asarray(stage, float), STAGE_EDGES, right=False)


def spring_levels(series):
    """(MHWS, MLWS, harmonic analysis) from the window's level series."""
    try:
        hc = harmonic_constants(series)
    except ValueError as e:
        raise base.NotComputable(f"harmonic analysis failed: {e}")
    amps = hc["amplitudes"]
    if "M2" not in amps or "S2" not in amps:
        raise base.NotComputable(f"the level series ({hc['duration_days']:.1f} d) does not resolve M2 and S2")
    a = amps["M2"] + amps["S2"]
    return hc["Z0"] + a, hc["Z0"] - a, hc


def coastxray_block(Y, V, h, bins, eligible, mean_tide, threshold):
    """z of one block: per eligible bin the median composite (valid
    observations only) thresholded, then the minimum mean tide over the
    bins where the pixel is water."""
    from pyintertidal.waterline_contour import interval_statistics

    Pb = Y.shape[1]
    z = np.full(Pb, np.inf)
    n_water = np.zeros(Pb, np.int16)
    for q in eligible:
        rows = np.flatnonzero(bins == q)
        stack = np.where(V[rows], Y[rows], np.nan).astype(np.float32)
        med, _, cnt = interval_statistics(stack)
        with np.errstate(invalid="ignore"):
            water = (cnt > 0) & (med > threshold)
        n_water += water
        z = np.where(water & (mean_tide[q] < z), mean_tide[q], z)
    return np.where(np.isfinite(z), z, np.nan), n_water


def run(obs, cfg=None, variant_id=None, **overrides):
    t0 = time.time()
    Y, C, W, h, T, P = base.check_obs(obs)
    opts = base.method_options(cfg, ID, DEFAULTS)
    opts.update(overrides)
    meta = base.base_meta(variant_id or ID, T, P, opts)
    try:
        ls = base.require_series(obs, cfg, need_times=True)
        mhws, mlws, hc = spring_levels(ls)
        stage = stage_percent(h, mhws, mlws, opts["stage_formula"])
    except base.NotComputable as e:
        return base.not_computable(meta, P, e, t0)
    bins = stage_bins(stage)
    sizes = np.bincount(bins, minlength=len(STAGE_EDGES) + 1)
    eligible = [int(q) for q in np.flatnonzero(sizes >= int(opts["min_images"]))]
    mean_tide = {q: float(np.mean(h[bins == q])) for q in np.flatnonzero(sizes > 0)}
    meta.update(level_series=ls.summary(), MHWS_m=mhws, MLWS_m=mlws, Z0_m=hc["Z0"],
                M2_m=hc["amplitudes"]["M2"], S2_m=hc["amplitudes"]["S2"],
                harmonic_constituents=hc["constituents"], stage_edges=list(STAGE_EDGES),
                images_per_bin=[int(v) for v in sizes], bins_used=eligible,
                mean_tide_per_bin={int(k): v for k, v in mean_tide.items()})
    if not eligible:
        return base.not_computable(meta, P, f"no stage bin holds >= {opts['min_images']} images", t0)
    z = np.full(P, np.nan)
    nw = np.zeros(P, np.int16)
    nv = np.zeros(P, np.int32)
    blk = int(base.as_dict(cfg).get("block_px", opts["block_px"]))
    for sl in base.column_blocks(P, blk):
        V, _ = base.valid_and_wet(Y, C, W, sl)
        Yb = np.asarray(Y[:, sl], np.float32)
        z[sl], nw[sl] = coastxray_block(Yb, V, h, bins, eligible, mean_tide, float(opts["threshold"]))
        nv[sl] = V.sum(0)
    meta["counts"] = {"never_water": int(np.sum(nw == 0)), "water_in_lowest_used_bin": int(
        np.sum(np.isclose(z, mean_tide[eligible[0]]) & np.isfinite(z)))}
    return base.finish(meta, z, t0, extra={"n_water_bins": nw, "n_valid": nv})


def run_t0(obs, cfg=None):
    return run(obs, cfg, variant_id="E7_t0", threshold=0.0)


def run_corrected(obs, cfg=None):
    return run(obs, cfg, variant_id="E7_corrected", stage_formula="corrected")


def _variant_card(vid, label, role):
    c = dict(CARD)
    c.update(id=vid, label=label, role=role)
    return c


METHODS = {ID: (LABEL, run, CARD)}
VARIANTS = {
    "E7_t0": (f"{LABEL}, composite threshold 0", run_t0,
              _variant_card("E7_t0", f"{LABEL}, composite threshold 0",
                            "sensitivity of E7: composites classified at the common NDWI > 0")),
    "E7_corrected": (f"{LABEL}, corrected Eq. 1", run_corrected,
                     _variant_card("E7_corrected", f"{LABEL}, corrected Eq. 1",
                                   "sensitivity of E7: stage = (TE - MLWS)/(MHWS - MLWS)")),
}
