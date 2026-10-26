# -*- coding: utf-8 -*-
"""E12 - in-house wet/dry bracket (legacy UDF, method "pixels").

Port of ``BATHYMETRY_UDF_NDWI`` in :mod:`pyintertidal.legacy.udf` with
``method="pixels"`` (pure bracketing, no optimisation, no diffusion):

* per pixel, ``zmin`` = the HIGHEST level at which the pixel was observed
  dry (land) and ``zmax`` = the LOWEST level at which it was observed wet
  (``_bracket_elevations``);
* the elevation is the midpoint ``(zmin + zmax) / 2`` (``_initial_estimate``),
  NaN when the pixel was never seen dry or never seen wet;
* an inconsistent pixel (``zmin > zmax``: seen dry above a level at which it
  was seen wet) keeps its midpoint, as in the UDF (whose confidence band
  halves it; no confidence band is returned here);
* the levels are float32, as the UDF's ``tide_array``.

The UDF's masks are ``water = clear & finite(NDWI) & (NDWI > threshold)`` and
``land = clear & finite(NDWI) & (NDWI <= threshold)``. Under the common
configuration clear = SCL {4, 5, 6, 7} and threshold = 0, so water = W and
land = V & ~W with V = C & finite(Y). The port is gated bitwise against the
UDF source itself (executed with a stub ``openeo.udf``) in
tests/test_fixedcfg_methods.py.
"""
from __future__ import annotations

import time

import numpy as np

from experiments.fixedcfg import elev_base as base

ID = "E12"
LABEL = "wet/dry bracket (in-house legacy UDF)"
DEFAULTS = {}

CARD = {
    "id": ID,
    "label": LABEL,
    "reference": ("In-house: the legacy openEO UDF BATHYMETRY_UDF_NDWI, method 'pixels' "
                  "(pyintertidal/legacy/udf.py; _bracket_elevations and _initial_estimate)"),
    "doi": None,
    "fidelity": ("F0 (port of in-house code; identity test: equal to the UDF's own functions, "
                 "executed from its source, bitwise on a synthetic cube)"),
    "family": "per-pixel bracket of the highest dry and the lowest wet level",
    "rule": ("z = (zmin + zmax) / 2 with zmin = max h over the pixel's dry observations and zmax = "
             "min h over its wet ones; NaN if either is missing; levels in float32 as the UDF"),
    "needs_full_frame": False,
    "needs": [],
    "min_scenes": 1,
    "own_lag": False,
    "expected_offset": ("none in a step world: the midpoint of the observed bracket, |error| <= half "
                        "the gap between the two bracketing levels; NaN for pixels never seen wet "
                        "(above the sampled range) or never seen dry"),
    "deviations": [
        {"paper": "UDF default NDWI threshold 0.1 (context 'ndwi_threshold')",
         "here": "wet = NDWI > 0 (W of the common configuration)",
         "why": "common configuration: one water detection for every method",
         "sensitivity": "not measured in these tables"},
        {"paper": "UDF default clear classes SCL {4, 5, 6, 12}",
         "here": "clear = SCL {4, 5, 6, 7} (C of the common configuration)",
         "why": "common configuration (MAREA's production clear mask)",
         "sensitivity": "not measured in these tables"},
        {"paper": "one tide height per date from the backend context (EOT20 at the AOI)",
         "here": "h_B without lag at the scene instants (production boundary)",
         "why": "common conditions: competitors use h_B without lag",
         "sensitivity": "not measured in these tables"},
        {"paper": "UDF output bands Confidence / Residual / Observations",
         "here": "only the Elevation band is returned (zmin and zmax in meta['extra'])",
         "why": "the tables score elevation only",
         "sensitivity": "not applicable"},
    ],
    "code": "experiments/fixedcfg/elev_bracket.py",
    "cost": "one pass over (T, P): milliseconds per 10^5 pixels",
}


def bracket_block(V, Wb, h32):
    """``(zmin, zmax, z0, inconsistent, n_obs)`` of one block, exactly as the
    UDF: zmin = max level over the dry observations, zmax = min level over
    the wet ones, z0 = (zmin + zmax) / 2 in float32."""
    land = V & ~Wb
    tide = h32[:, None]
    zmin = np.max(np.where(land, tide, -np.inf), axis=0)
    zmin = np.where(np.isfinite(zmin), zmin, np.nan).astype(np.float32)
    zmax = np.min(np.where(Wb, tide, np.inf), axis=0)
    zmax = np.where(np.isfinite(zmax), zmax, np.nan).astype(np.float32)
    n_obs = np.sum(Wb | land, axis=0).astype(np.float32)
    with np.errstate(invalid="ignore"):
        inconsistent = zmin > zmax
    z0 = ((zmin + zmax) / 2.0).astype(np.float32)
    return zmin, zmax, z0, inconsistent, n_obs


def run(obs, cfg=None):
    t0 = time.time()
    Y, C, W, h, T, P = base.check_obs(obs)
    opts = base.method_options(cfg, ID, DEFAULTS)
    meta = base.base_meta(ID, T, P, opts)
    h32 = np.asarray(h, float).astype(np.float32)
    z = np.full(P, np.nan, np.float32)
    zmin = np.full(P, np.nan, np.float32)
    zmax = np.full(P, np.nan, np.float32)
    n_valid = np.zeros(P, np.int32)
    n_inc = 0
    for sl in base.column_blocks(P, base.block_px(cfg)):
        V, Wb = base.valid_and_wet(Y, C, W, sl)
        a, b, z0, inc, nobs = bracket_block(V, Wb, h32)
        zmin[sl], zmax[sl], z[sl] = a, b, z0
        n_valid[sl] = nobs.astype(np.int32)
        n_inc += int(inc.sum())
    meta["counts"] = {"never_dry": int(np.isnan(zmin).sum()), "never_wet": int(np.isnan(zmax).sum()),
                      "inconsistent_kept": n_inc}
    return base.finish(meta, z, t0, extra={"zmin": zmin, "zmax": zmax, "n_valid": n_valid})


METHODS = {ID: (LABEL, run, CARD)}
