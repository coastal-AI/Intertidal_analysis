# -*- coding: utf-8 -*-
"""v13 benchmark — sites, constants, seeds and the sectioned FROZEN plan.

Question
--------
Which inputs, constants and pre-registered rules define the controlled
benchmark of intertidal lag and elevation methods, site by site and at each
grid resolution (10 m for the reported numbers, 20 m for local tests)?

Pre-registered (docs/v13_benchmark_spec.md, Appendix A; authors' final
decisions of 2026-10-08)
-------------------------------------------------------------------------
* D1: the PRIMARY per-pixel cloud mask is ``marea_clear`` (valid iff SCL in
  {4, 5, 6, 7}, MAREA's production mask); ``blocklist`` (SCL 0, 3, 8, 9, 10
  and no data masked) is the sensitivity. The package domain U_rec is
  computed under blocklist + floor0, a superset of every subset keep under
  marea_clear, so one package serves both rules.
* D2: the primary band preprocessing is ``floor0`` (B03 and B08 floored at
  0 before any index); ``raw`` (production) is the sensitivity.
* D6: Granadeiro at N = all at all five sites, A-mode curves at Escalda and
  Villaviciosa (N in GRAN_N_GRID x 3 reps).
* Every other decision (D3-D5, D7-D23) is the recommended default of §11.
* Validation v2 (docs/validation_v2_spec.md §9.1, §12.3): B = 20,000 for
  every confirmatory family and 1,000 for description, so that the smallest
  bootstrap p, 2 / (B + 1), is below Holm's first threshold of every frozen
  family (:func:`attainability`; P2 holds 128 statistics).

FROZEN holds the analysis plan in sections. Each section has its own sha256
(:func:`frozen_sha`), and every key records only the sections it depends
on, so a later edit invalidates exactly what it changes (§2.11):

* ``record``, ``package``, ``obs``, ``domain``, ``levels``, ``subsets``
  enter the package key: editing them means rebuilding every package;
* a method section (``marea``, ``nidem``, ``granadeiro``, ``gm20``,
  ``hd26``, ``baselines``) and the group's ``threads`` entry enter that
  method's config key only;
* ``judges``, ``truth_elev``, ``truth_lag``, ``scoring``, ``inference``,
  ``curves`` enter the scores only.

FROZEN is deep-frozen (read-only mappings and tuples): code reads it, never
writes it. :func:`frozen_plain` returns an ordinary JSON-ready copy.

This module has no import-time side effects and imports only the standard
library and :mod:`experiments.validation_grid`.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

from experiments.validation_grid import dutch_cube, dutch_products

#: repository root (the CLI chdirs here; every path below is relative to it)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERIOD = ("2023-01-01", "2025-12-31")

SITES = ("escalda", "wadden", "ems", "ferrol", "villaviciosa")
#: the synthetic positive control (§4.2); built from the Wadden 20 m record
CONTROL_SITE = "pc"
ALL_SITES = SITES + (CONTROL_SITE,)
SITE_ID = {"escalda": 1, "wadden": 2, "ems": 3, "ferrol": 4,
           "villaviciosa": 5, "pc": 9}
CONTROL_SCENARIOS = ("zero", "profile")

RES_CHOICES = (10, 20)
#: 20 m exists only for the Dutch cubes (local tests) and the control
RES_SITES = {10: SITES, 20: ("escalda", "wadden", "ems", CONTROL_SITE)}
#: only 10 m numbers are reported; 20 m is for local testing
REPORTED_RES = 10

TIDE_DIR = "tide_models"
GAUGE_CACHE = "data_v4/gauges"
IOC_STATIONS = "data_v4/gauges/ioc_stations.json"
ESTUARY_TRUTH = "data_v4/gauges/estuary_truth"
STATIONS_JSON = f"{ESTUARY_TRUTH}/stations.json"
#: local mirror of server outputs (not tracked); :func:`resolve` falls back
#: to it for production files that exist only on the server
MIRROR_ROOT = "results/server_10m"

#: production boundary tag of the Dutch comparison folders
DUTCH_TAG = {"escalda": "eot20_g2", "wadden": "eot20_g1", "ems": "eot20_g1"}

#: the gauges each production boundary holds out (the notebooks' JUDGE_GAUGES)
HELD_OUT = {"escalda": ("trnz",), "wadden": ("harl",), "ems": ("delf",),
            "ferrol": ("fer2",), "villaviciosa": ()}

#: the production boundary of each site (§2.1); ``aoi`` is the
#: pyintertidal.sites key whose AOI the call uses (pc uses the Wadden one)
BOUNDARY = {
    "escalda": {"kind": "make_boundary", "aoi": "escalda", "tide_model": "EOT20",
                "n_gauges": 2, "exclude": HELD_OUT["escalda"]},
    "wadden": {"kind": "make_boundary", "aoi": "wadden", "tide_model": "EOT20",
               "n_gauges": 1, "exclude": HELD_OUT["wadden"]},
    "ems": {"kind": "make_boundary", "aoi": "ems", "tide_model": "EOT20",
            "n_gauges": 1, "exclude": HELD_OUT["ems"]},
    "ferrol": {"kind": "make_boundary", "aoi": "ferrol", "tide_model": "EOT20",
               "n_gauges": 1, "exclude": HELD_OUT["ferrol"]},
    "villaviciosa": {"kind": "pytmd", "aoi": "villaviciosa", "model": "EOT20",
                     "at": "aoi.centroid"},
}
BOUNDARY["pc"] = dict(BOUNDARY["wadden"])

#: the sea edge(s) of each frame (the notebooks' MOUTH_SIDE)
MOUTH_SIDE = {"escalda": "west", "wadden": "north+west", "ems": "north+west",
              "ferrol": "west", "villaviciosa": "any", "pc": "west"}

ELEVATION_TRUTH = {"escalda": ("vaklodingen",), "wadden": ("vaklodingen",),
                   "ems": ("vaklodingen",), "ferrol": (),
                   "villaviciosa": ("lidar", "rtk"), "pc": ("planted",)}

#: in-estuary gauges (codes of estuary_truth/stations.json); every station is
#: listed, the method-independent inclusion rules decide which are scored
JUDGES = {
    "escalda": ("gatvanborssele", "terneuzen", "terneuzen.westsluis.buiten",
                "borssele"),
    "wadden": ("terschelling.west", "harlingen.waddenzee"),
    "ems": ("emshoern_9340010", "eemshaven.waddenzee", "eemshaven.haven",
            "dukegat_3990020", "knock_3990010", "delfzijl",
            "emden_neue_seeschleuse_3970010", "pogum_3950020",
            "nieuwestatenzijl.dollard", "uithuizerwad"),
    "ferrol": ("fer2", "fer4_3212"),
    "villaviciosa": (),
    "pc": (),
}
#: gauges counted as one location (mean of their errors)
SAME_LOCATION = (("terneuzen", "terneuzen.westsluis.buiten"),)

N_ALL = "all"
N_GRID = (10, 20, 30, 45, 60, 90, 120, 180, 240, 330)          # plus "all" [D5]
GRAN_N_GRID = (20, 45, 90, 180, 330)                           # [D6]
REPS = {"rand_A": 10, "rand_F": 5, "win_A": 5, "floor": 5, "nidem": 10,
        "gran_A": 3, "gran_F": 3}                              # [D6]
GRAN_CURVE_SITES = ("escalda", "villaviciosa")                 # [D6]
SEEDS = dict(subsets=20261008, block_boot=20261009, gm20=20261010,
             truth_boot=20261011, windows=20261012, pc=20261013,
             variogram=20261014,
             m2=20260819, granadeiro=20210119)   # m2, granadeiro: production seeds

#: BLAS/OMP threads per group, fixed and part of the group's config key: a
#: multithreaded dgemv partitions columns by thread and the last bits of
#: invert_series depend on that partition
BLAS_THREADS = {"marea_pair": 4, "nidem": 4, "gm20": 4, "hd26": 4,
                "baselines": 4, "published": 4, "granadeiro": 1}

#: FROZEN sections behind each key (§2.11)
PACKAGE_SECTIONS = ("record", "package", "obs", "domain", "levels", "subsets")
SCORE_SECTIONS = ("truth_elev", "truth_lag", "judges", "scoring", "inference",
                  "curves")
#: the FROZEN section of each adapter group (the as-published loaders have
#: no estimator parameters, hence no section)
METHOD_SECTION = {"marea_pair": "marea", "nidem": "nidem",
                  "granadeiro": "granadeiro", "gm20": "gm20", "hd26": "hd26",
                  "baselines": "baselines", "published": None}


# ─────────────────────────────────────────────────────────────────────────────
#  FROZEN (Appendix A; "rule" is the Appendix A text with the authors'
#  override, the other keys are the values the code reads)
# ─────────────────────────────────────────────────────────────────────────────

def _freeze(obj):
    """Deep read-only copy: mappings -> MappingProxyType, lists -> tuples."""
    if isinstance(obj, Mapping):
        return types.MappingProxyType({str(k): _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"FROZEN holds JSON values only, not {type(obj).__name__}")


def _plain(obj):
    """JSON-ready copy of a frozen value (mappings -> dict, tuples -> list)."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


_NEIGHBOURHOOD = {"n_nearest": 200, "r0_m": 500.0, "pad_m": 300.0,
                  "step_m": 250.0, "min_keep_px": 100, "cap_m": 2000.0}

FROZEN = _freeze({
    "record": {
        "rule": ("cube dates in 2023-01-01..2025-12-31 with an overpass time and "
                 "a finite h_B(t); one provider call on sorted instants"),
        "period": list(PERIOD),
        "dates": ("the production extraction's dates, asserted equal to the cube's "
                  "(v7.cube_grid); the positive control: the cube's dates"),
        "overpass": "overpass_times.json values read as naive UTC (v8.to_utc_naive)",
        "level_call": ("h = provider.levels(t) in ONE call on the sorted candidate "
                       "instants; time_index = candidates with a finite h"),
        "h_B": {s: {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in BOUNDARY[s].items()} for s in ALL_SITES},
        "provider_call": {"directory": TIDE_DIR, "cache_dir": GAUGE_CACHE,
                          "period": list(PERIOD),
                          "certificates": "net.use_system_certificates() first"},
    },
    "package": {
        "rule": ("raw B03/B08 int16 and SCL uint8 at dom = S0 ∪ dilate(U_rec, 3 px) ∪ "
                 "keep_prod, S0 = NB ∪ dilate(keep_full, 300 m) (v2 spec §2.10; D_truth "
                 "joins S0 and the C11/C3 U_rec joins dom with the v2 package); "
                 "U_rec = {clr ≥ 31, wet ≥ 4, dry ≥ 4} under blocklist + floor0; "
                 "frame validity bits; two passes"),
        "stored": {"B03": "int16 raw cube values", "B08": "int16 raw cube values",
                   "SCL": "uint8, fill and NaN -> 255"},
        "dom": "sorted unique flat indices (int64) of S0 | dilate(U_rec, 3 px) | keep_prod",
        "dom_rule": {"urec_dilation_px": 3,
                     "dilation": ("disk: every pixel whose centre lies within the radius "
                                  "(Euclidean) of the centre of a member pixel"),
                     "urec_covers": ("U_rec of the A-mode curve settings P0 and C2 (marea_clear "
                                     "counts never exceed blocklist counts; C2 reads a subset "
                                     "of the record)"),
                     "keep_prod": "the production extraction's keep (legacy gates)"},
        "S0": {"rule": "D_truth ∪ NB ∪ dilate(keep_full(P0, W3), 300 m) ∪ Ω (v2 spec §2.10)",
               "keep_dilation_m": 300.0,
               "omega": ("the D2 domain lies inside dilate(keep_full, 300 m) by its "
                         "definition (v2 spec §2.6): no pixel of its own"),
               "d_truth": "not in this package: frozen at lock-A by the v2 truth module"},
        "U_rec": {"mask": "blocklist", "index": "floor0", "min_clear": 31,
                  "min_wet": 4, "min_dry": 4,
                  "why": ("an exact superset of every A-mode keep under either mask "
                          "rule: marea_clear counts never exceed blocklist counts")},
        "NB": {"rule": ("frame pixels within R_g of each listed station inside the "
                        "frame, R_g by the judges' neighbourhood rule on keep_full"),
               **_NEIGHBOURHOOD},
        "frame_valid": {"mask": "marea_clear", "index": "floor0",
                        "layout": "np.packbits of V over the flat frame, one row per record scene"},
        "counters": {"mask": "marea_clear", "index": "floor0", "dtype": "int16",
                     "names": ["clr_n", "wet_n"]},
        "scene_stats": ("per record scene: SCL counts 0..11 and fill on the frame and on "
                        "keep_prod; g21 frame cloud and no-data shares; load_ard good "
                        "fraction; V share on the frame and on keep_prod; negative-band "
                        "and raw |NDWI| > 1 counts; h0; rising (marks_lp)"),
        "passes": 2,
    },
    "obs": {
        "rule": ("mask rules marea_clear (primary) / blocklist; index modes floor0 "
                 "(primary) / raw; V = mask & finite NDWI; wet = V & NDWI > 0; "
                 "NIR = floored B08/10000"),
        "decision": ("D1 (authors, 2026-10-08): marea_clear primary, blocklist "
                     "sensitivity; D2: floor0 primary, raw sensitivity"),
        "primary_mask": "marea_clear",
        "primary_index": "floor0",
        "mask_rules": {"marea_clear": {"valid_scl": [4, 5, 6, 7]},
                       "blocklist": {"masked_scl": [0, 3, 8, 9, 10],
                                     "no_data": "masked"}},
        "index_modes": {"floor0": "B03, B08 := max(value, 0) for non-fill values, before any index",
                        "raw": "cube values unchanged (production)"},
        "encoding": {"dtype": "int16", "fill": -32768, "scale": None,
                     "boa_min_value": -1000, "scl_codes": list(range(12)),
                     "scl_nodata_u8": 255},
        "ndwi": ("np.where(g + n != 0, (g - n) / (g + n), nan) in float32 on the "
                 "decoded (and floored) bands, as marea.extract and wc.scene_ndwi"),
        "valid": "V = mask_valid(SCL, rule) & isfinite(NDWI)",
        "wet": "V & (NDWI > 0)",
        "nir": "g21.to_reflectance of the floored int16 B08: value / 10000, float64, fill -> NaN",
        "granadeiro_common_mode": "B03, B08 := fill where not V (mask_raw_bands)",
        "item_valid": "ITEM PQ mask (SCL present and not in 0, 1, 3, 8, 9, 10); gates only",
    },
    "domain": {
        "rule": ("extraction rule clr > 30, 0.10 < wf < 0.90, sea wf ≥ 0.90; 3,000 px; "
                 "5,000 finite s; make_bands_fixed(s, 3.0, 500); mouth sides per site"),
        "extract_rule": {"min_clear": 31, "wf_lo": 0.10, "wf_hi": 0.90,
                         "sea_wf_min": 0.90,
                         "counts": "V and wet under the primary mask and index"},
        "min_intertidal": 3000,
        "min_finite_s": 5000,
        "bands": {"length_km": 3.0, "min_px": 500,
                  "function": "tide_estimators.make_bands_fixed"},
        "geometry": ("s_km = geometry.along_distance(sea | inter, geometry.mouth_seeds("
                     "sea, side=mouth_side), pixel_m).ravel()[keep] / 1000 (float64); "
                     "ValueError -> NaN everywhere"),
        "no_anchor": "at most 5,000 finite s: one band, tau = 0 (reconstruct)",
        "no_intertidal": "fewer than 3,000 keep pixels: no domain (reconstruct's sin_intermareal)",
        "F_mode": "the full-record domain (keep_full, sea_full, s_km, bands) for every N",
        "A_mode": ("the rule on the subset's own observations: counts of V and wet over sel "
                   "at dom; a pixel outside dom with clr_n >= 31 and clr_n - wet_n <= 3 is "
                   "sea iff its subset clear count (frame_valid) exceeds 30; geometry and "
                   "bands recomputed; for_subset(arange(T)) == full"),
        "mouth_side": MOUTH_SIDE,
        "strata_outside_keep": ("band_map_frame: raw band floor(s / 3) mapped through the "
                                "merge of make_bands_fixed; no finite s -> stratum 'no s'"),
    },
    "levels": {
        "rule": ("integer τ ∈ [−90, max(240, 180·(K_full − 1))], chunks ≤ 150,000, "
                 "batch-invariance rows, live per-shift beyond; 1-min PERIOD ± 2 d; "
                 "10-min = every 10th; marks_lp = 61-min centred mean; marks_raw as v8"),
        "tau_min": -90,
        "tau_max_floor": 240,
        "tau_per_band": 180,
        "chunk": 150000,
        "check_rows": "every marea.BANK_TAUS row, every 60th row from tau_min, and the last row",
        "check_every": 60,
        "fallback": ("any checked row not bitwise equal to a per-shift call: every row "
                     "rebuilt by per-shift calls, batch_invariant = False, subset rows "
                     "read live"),
        "beyond_table": "live per-shift provider call on t[sel], never capped",
        "series_1min": {"source": "v8.boundary_series", "pad_days": 2,
                        "chunk_days": 92, "step_min": 1},
        "series_10min": {"start": "2023-01-01 00:00", "end": "2025-12-31 23:50",
                         "every": 10},
        "marks_lp": {"width_min": 61, "filter": ("centred running mean; NaN unless all "
                                                 "61 samples are finite"),
                     "marks": "g21.tide_marks"},
        "marks_raw": "g21.tide_marks on the raw 1-min series (v8)",
    },
    "subsets": {
        "rule": ("rand: default_rng([20261008, SITE_ID, r]).permutation(T)[:N] sorted, "
                 "10 reps; win: start default_rng([20261012, SITE_ID, N, r]); rules tz / "
                 "g21 / ard"),
        "site_id": SITE_ID,
        "rand": {"seed": SEEDS["subsets"], "stored_reps": 10,
                 "perm": "np.random.default_rng([seed, SITE_ID[site], r]).permutation(T)",
                 "sel": "np.sort(perm[r][:N]), nested in N within a rep; N = 'all' -> arange(T)"},
        "win": {"seed": SEEDS["windows"],
                "start": "np.random.default_rng([seed, SITE_ID[site], N, r]).integers(0, T - N + 1)",
                "sel": "np.arange(start, start + N)"},
        "rules": {"tz": ("record positions whose date is in the 10 m MAREA transition-zone "
                         "record (result.json record_dates)"),
                  "g21": ("full-frame SCL shares: cloud (SCL 3, 8, 9, 10) < 0.10 and no "
                          "data <= 0.02 (g21.scene_frame_shares, g21.scene_rule)"),
                  "ard": ("load_ard full-frame good fraction >= 0.5 "
                          "(wc.frame_good_fraction, wc.S2_SCENE_MIN_GOODDATA)")},
        "key": "subset_key = sha256(sel.astype('<i8').tobytes())",
    },
    "marea": {
        "rule": ("m2.yaml values, TAU_BOUNDS, monotone, seed 20260819 + 31, smoothing 3 km, "
                 "bracket 3, floor 40 (both products), slices of 200,000; variants tau120; "
                 "bootstrap diagnostic 30 × 600 px"),
        "m2_yaml": {"sigma0_m": 0.20,
                    "sigma_perfil_m": [0.06, 0.1, 0.15, 0.22, 0.32, 0.45],
                    "z_puntos": 60, "max_px_banda_m2a": 1200, "seed": SEEDS["m2"]},
        "tau_bounds": [-45.0, 180.0],
        "monotone": True,
        "column_seed": SEEDS["m2"] + 31,
        "smooth_km": 3.0,
        "min_bracket": 3,
        "floor": 40,
        "floor_sensitivities": [80, 0],
        "slice": 200000,
        "usable_min_clear": 6,
        "identified_min_usable": 50,
        "variants": {"tau120": {"tau_bounds": [-45.0, 120.0], "bank_max_tau": 120.0}},
        "bootstrap": {"n_boot": 30, "max_px_band": 600, "seed": SEEDS["m2"],
                      "role": "diagnostic only"},
    },
    "nidem": {
        "rule": "ITEM rules unchanged; filtered at 0.25; pre-flight 0.01 on keep_full; never relaxed",
        "confidence_max": 0.25,
        "min_valid_share": 0.01,
        "preflight_pixels": "keep_full",
        "tuning_thresholds": [0.15, 0.20, 0.25, 0.30],
    },
    "granadeiro": {
        "rule": ("v8 FROZEN steps; ctl: identity calibration, exact h_B, window −45..180 "
                 "step 5, marks_lp, round(GAM lag); fallback rules (pool empty, < 60 "
                 "unique, GAM failure); pinned = within 0.005 m of (min + max)/2 of "
                 "x_used; seed 20210119; variants eq1_90, calon, paper"),
        "ctl": {"calibration": "identity", "tide": "exact h_B (level-table rows)",
                "lag_window_min": [-45, 180, 5], "marks": "marks_lp",
                "final_lag": "round(gam_lag) to the minute"},
        "fallback": {"pool_empty": "lag 0, final = preliminary DEM",
                     "min_unique_lags": 60, "gam_failure": "lag 0, final = preliminary DEM"},
        "pinned_tol_m": 0.005,
        "seed": SEEDS["granadeiro"],
        "sd_threshold": 0.2,
        "min_scenes": 8,
        "variants": ["ctl", "ctl_unpinned", "eq1_90", "calon", "paper"],
        "tuning": {"sd_threshold": [0.15, 0.20, 0.30],
                   "lag_window_min": [[-90, 90], [-45, 180]]},
    },
    "gm20": {
        "rule": ("12.4206012 h; marks_lp HW; rel primary; QC n ≥ 20, SD > 0.2, "
                 "0.05 < wet < 0.95, R² ≥ 0.1; wrap [−120, 625); 20 × 70 % IQR; own variant"),
        "period_h": 12.4206012,
        "reference": "HW marks of marks_lp",
        "primary": "rel",
        "qc": {"min_n": 20, "min_sd": 0.2, "wet_lo": 0.05, "wet_hi": 0.95, "min_r2": 0.1},
        "wrap_lo_min": -120.0,
        "iqr": {"refits": 20, "share": 0.7, "seed": SEEDS["gm20"]},
        "variants": ["rel", "raw", "scaled", "wet", "own"],
    },
    "hd26": {
        "rule": ("M2, N2, K2, O1, Q1, GOT corrections; detrend + one-pass 3σ; band scene "
                 "≥ 0.8 valid (0.5, 0.9 sensitivities); ≥ 30 points over ≥ 365 d per "
                 "band; rel primary"),
        "constituents": ["m2", "n2", "k2", "o1", "q1"],
        "corrections": "GOT",
        "outlier_sigma": 3.0,
        "band_valid_share": 0.8,
        "band_valid_share_sensitivities": [0.5, 0.9],
        "min_points": 30,
        "min_span_days": 365,
        "primary": "rel",
        "variants": ["rel", "faithful", "local"],
        "vif_flag": 2.0,
    },
    "baselines": {
        "rule": "zero; EOT20; GOT5.6 if loadable",
        "zero": True,
        "models": ["EOT20", "GOT5.6"],
        "role": "lower bounds, never evidence for a method",
    },
    "judges": {
        "rule": ("inclusion: in frame, ≥ 50 % instants within 6 min, > 3 km from boundary "
                 "members, ≤ 10 % of values within 1 cm of the minimum; Terneuzen pair one "
                 "location; neighbourhood R = max(500, d_min + 300) widened by 250 m to "
                 "≥ 100 keep_full px, cap 2,000 m; mouth sensitivity s_g < 3 km"),
        "stations": JUDGES,
        "inclusion": {"in_frame": True, "coverage_min": 0.5, "tolerance_min": 6,
                      "boundary_member_min_km": 3.0, "dry_max_share": 0.10,
                      "dry_tol_m": 0.01},
        "same_location": SAME_LOCATION,
        "neighbourhood": _NEIGHBOURHOOD,
        "sparse_flag": {"radius_m": 2000.0, "min_keep_px": 100},
        "mouth_s_km": 3.0,
    },
    "truth_elev": {
        "rule": ("Vaklodingen 2019-2025 latest per cell; PNOA ground-only (v10.lidar_truth); "
                 "RTK load_rtk() defaults, elev_alicante; range filter [min h0 − 0.25, "
                 "max h0 + 0.25]; D_truth, D_keep, S"),
        "vaklodingen": {"years": [2019, 2025], "rule": "latest finite survey per 20 m RD cell"},
        "lidar": "v10.lidar_truth(rtk.DEFAULT_GRID), PNOA MDT02 ground-only",
        "rtk": "rtk.load_rtk() with default arguments, elev_alicante, not range-filtered",
        "range_pad_m": 0.25,
        "domains": ["D_truth", "D_keep", "S"],
    },
    "truth_lag": {
        "rule": ("(a) nearest within 6 min, τ ∈ [−60, 240] at 1 min, median-centred E, 1-cm "
                 "interval; τ_ab; (b) CONS_B, GOT, order 1, identical instants, quarter-year "
                 "jackknife, ±1.96 SE; 200 month-block bootstraps (seed 20261011)"),
        "a": {"tolerance_min": 6, "tau_range_min": [-60, 240], "step_min": 1,
              "interval_m": 0.01},
        "b": {"constituents": "CONS_B", "corrections": "GOT", "order": 1,
              "jackknife": "12 quarter-years", "z": 1.96},
        "bootstrap": {"n": 200, "block": "month", "seed": SEEDS["truth_boot"]},
    },
    "scoring": {
        "rule": ("MIN_N 30/30/20/30; views and strata of §5.3; cells half-mapped rule; "
                 "read-outs ≥ 30 px, local primary, circular for phase; fallback 0 primary, "
                 "identified-only sensitivity"),
        "min_n": {"vaklodingen": 30, "lidar": 30, "rtk": 20, "planted": 30},
        "views": ["own_D", "own_K", "common_q", "common_raw", "pair", "cells",
                  "survey_ge_2023"],
        "cells_min_mapped_share": 0.5,
        "readout": {"min_px": 30, "primary": "local",
                    "supports": ["local", "band", "band_interp", "s_profile"],
                    "phase": "circular statistics"},
        "fallback": {"primary": 0.0, "sensitivity": "identified_only"},
    },
    "inference": {
        "rule": ("block size per site (v2 spec §9.1, Appendix A analysis_v2): the largest "
                 "practical range of the truth-free variograms of the map differences "
                 "z_A - z_B over every P1/P2/P3 contrast at P0 on D_truth (20,000 px, 32 bins "
                 "to 20 km within the frame, exponential, seed 20261014), rounded UP to the "
                 "lattice {0.25, 0.5, 1, 2, 4, 8, 16} km (no 4 km cap; 16 km when the range "
                 "exceeds it, reported); one size per site so every contrast shares the "
                 "bootstrap weights, seeded from (site, truth, stratum, block size) only; "
                 "≥ 25 blocks; B = 1000 descriptive and 20,000 for every confirmatory family "
                 "(seed 20261009), so that 2/(B + 1) < α/m for every frozen family (v2 spec "
                 "§9.1); Holm α = 0.05; margins 0.02 m and 0.05; rep spread ≥ 90 % at "
                 "N = 330; lag sign tests, Holm over five comparisons per truth, all three "
                 "truths"),
        "variogram": {"max_px": 20000, "n_bins": 32, "max_lag_m": 20000.0,
                      "max_lag_rule": "min(max_lag_m, the frame diagonal / 2)",
                      "model": "exponential", "range_max_m": 16000.0,
                      "lattice_m": [250, 500, 1000, 2000, 4000, 8000, 16000],
                      "rounding": "up (the smallest lattice value >= the practical range)",
                      "contrasts": ("every P1/P2/P3 contrast at P0, on D_truth, map "
                                    "differences only (no truth read)"),
                      "pooling": "the largest range over the site's contrasts (conservative)",
                      "weights": "shared by every contrast of the site",
                      "default_m": 1000, "seed": SEEDS["variogram"]},
        "min_blocks": 25,
        "boot": {"B_descriptive": 1000, "B_primary": 20000, "seed": SEEDS["block_boot"]},
        "holm_alpha": 0.05,
        "families": {
            "rule": ("the confirmatory families of v2 spec §9.2 and §6.5 with their frozen "
                     "number m of view-specific statistics (L-P1, L-P2: comparisons per "
                     "truth); a family is attainable iff its smallest attainable p is below "
                     "Holm's first threshold α/m: 2/(B_primary + 1) for the shared-weight "
                     "bootstrap families, 2/2^n for the exact sign-flip tests over the n "
                     "headline locations (no zero differences)"),
            "bootstrap": {"P1": 16, "P2": 128, "P3": 98},
            "sign_flip": {"L-P1": 5, "L-P2": 1},
            "headline_locations": 13,
        },
        "margins": {"rmse_meanc_m": 0.02, "one_minus_b": 0.05},
        "rep_spread": {"min_share": 0.9, "N": 330},
        "lag_sign_tests": {"comparisons": ["zero", "EOT20", "granadeiro", "gm20", "hd26"],
                           "truths": ["a", "tau_ab", "b"]},
    },
    "curves": {
        "rule": "N grid; reps per family and mode; k/R ≥ 0.5; joint-computable accuracy",
        "N_grid": list(N_GRID) + [N_ALL],
        "gran_N_grid": list(GRAN_N_GRID),
        "reps": REPS,
        "gran_curve_sites": list(GRAN_CURVE_SITES),
        "min_computable_share": 0.5,
        "accuracy": "reps where all compared methods are computable",
    },
    "threads": {
        "rule": ("BLAS threads per group: marea_pair, nidem, gm20, hd26 = 4; granadeiro = 1 "
                 "(part of config_key); replay pools explicit"),
        "blas": BLAS_THREADS,
        "boot_threads_default": 1,
    },
    "positive_control": {
        "rule": ("base wadden 20 m; frame 120 × 640; planted z range [LO + 0.15 R, "
                 "HI − 0.15 R]; scenarios zero / profile (120 min at 12 km); noise and "
                 "reflectance laws of §4.2; SCL window; haze 2 % +400 DN; tolerances of "
                 "D22; seed 20261013"),
        "base": {"site": "wadden", "res_m": 20},
        "frame": {"H": 120, "W": 640, "sea_cols": [0, 19], "channel_rows": [52, 67],
                  "flat_rows": [[12, 51], [68, 107]], "land_rows": [[0, 11], [108, 119]],
                  "chunks": [1, 120, 256]},
        "z": {"flat_range": "[LO + 0.15 R, HI - 0.15 R], linear from the channel edge",
              "land": "HI + 1 m", "sea": "LO - 1 m"},
        "scenarios": {"zero": "tau = 0", "profile": "tau(s) = round(120 * min(s / 12 km, 1)) min"},
        "noise_m": 0.04,
        "reflectance_dn": {"water": {"B03": [600, 150], "B08": [80, 120]},
                           "flat": {"B03": [900, 150], "B08": [1500, 300]},
                           "land": {"B03": [700, 100], "B08": [3000, 400]},
                           "cloud_8_9_10": {"B03": [4000, 800], "B08": [4000, 800]},
                           "shadow_3_factor": 0.3},
        "haze": {"share": 0.02, "dn": 400},
        "tolerances": {"coverage_min": 0.5, "median_abs_dz_max_m": 0.10,
                       "abs_median_bias_max_m": 0.05, "slope": [0.8, 1.2],
                       "lag_max_min": 10.0},
        "must_pass": {"zero_elevation": ["nodelay", "marea", "nidem", "granadeiro"],
                      "profile_elevation": ["marea", "granadeiro"],
                      "profile_lag": ["marea", "granadeiro", "gm20", "hd26"],
                      "zero_lag": "every lag method"},
        "extra_unit": {"scenario": "profile", "N": 120, "rep": 0, "mode": "A"},
        "seed": SEEDS["pc"],
    },
})

FROZEN_SECTIONS = tuple(FROZEN)


def frozen_plain(*sections):
    """Ordinary nested dict/list copy of FROZEN (all sections if none given)."""
    names = sections or FROZEN_SECTIONS
    return {s: _plain(FROZEN[s]) for s in names}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj):
    """sha256 of ``json.dumps(obj, sort_keys=True)`` (default separators,
    ASCII escapes), the one canonical JSON hash of the benchmark."""
    return sha256_bytes(json.dumps(_plain(obj), sort_keys=True).encode("utf-8"))


def frozen_sha(*sections):
    """sha256 of ``json.dumps({s: FROZEN[s] for s in sections}, sort_keys=True)``.

    Parameters
    ----------
    *sections : str
        Section names; at least one.
    """
    if not sections:
        raise ValueError("frozen_sha needs at least one section")
    unknown = [s for s in sections if s not in FROZEN]
    if unknown:
        raise KeyError(f"unknown FROZEN section(s) {unknown}; known: {FROZEN_SECTIONS}")
    return sha256_json({s: FROZEN[s] for s in sections})


def frozen_shas(sections=None):
    """``{section: frozen_sha(section)}`` for the given sections (default all)."""
    return {s: frozen_sha(s) for s in (sections or FROZEN_SECTIONS)}


def frozen_document():
    """The content of ``FROZEN.json``: every section, its sha256 and which
    key each section enters."""
    return {"sections": frozen_plain(), "sha256": frozen_shas(),
            "package_key_sections": list(PACKAGE_SECTIONS),
            "score_sections": list(SCORE_SECTIONS),
            "method_sections": dict(METHOD_SECTION)}


def sha256_source(path):
    """sha256 of a source file with CRLF normalised to LF.

    The laptop checks text out with CRLF (core.autocrlf) and the server with
    LF; without the normalisation the same commit would give two code keys.
    """
    with open(path, "rb") as f:
        return sha256_bytes(f.read().replace(b"\r\n", b"\n"))


# ─────────────────────────────────────────────────────────────────────────────
#  sites and paths
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiteConfig:
    """Everything the benchmark needs to know about one site at one grid.

    Paths are relative to the repository root. ``extract`` is the production
    extraction (keep_prod, the gates); ``transition_result`` the result.json
    holding the 10 m transition-zone ``record_dates`` (the rule-tz subset;
    :func:`resolve` finds the local mirror when the server file is absent);
    ``published`` the as-published folders (may not exist); ``root`` the
    results tree of the site (the control scenarios have their own).
    """

    site: str
    res_m: int
    pixel_m: float
    cube: str
    extract: Optional[str]
    overpass: str
    boundary: Mapping
    mouth_side: str
    elevation_truth: tuple
    judges: tuple
    published: Mapping
    transition_result: Optional[str]
    root: str
    tag: str
    scenario: Optional[str] = None
    extra: Mapping = field(default_factory=dict)

    def __hash__(self):
        # a config is determined by (site, res_m, scenario); the mapping
        # fields are not hashable, so the generated hash would fail
        return hash((self.site, self.res_m, self.scenario))

    @property
    def site_id(self):
        return SITE_ID[self.site]

    @property
    def common_dir(self):
        return f"{self.root}/common/{self.site}"

    @property
    def truth_dir(self):
        return f"{self.root}/truth/{self.site}"

    @property
    def runs_dir(self):
        return f"{self.root}/runs/{self.site}"

    @property
    def gates_dir(self):
        return f"{self.root}/gates/{self.site}"


def check_res(site, res_m):
    """Raise ValueError unless ``site`` exists at ``res_m``."""
    if site not in ALL_SITES:
        raise ValueError(f"unknown site {site!r}; expected one of {ALL_SITES}")
    if res_m not in RES_CHOICES:
        raise ValueError(f"unknown resolution {res_m!r}; expected one of {RES_CHOICES}")
    if site not in RES_SITES[res_m]:
        raise ValueError(f"{site} has no {res_m} m cube (sites at {res_m} m: "
                         f"{RES_SITES[res_m]})")


def out_root(res_m):
    """``results/v13_benchmark_<res>m``: every output of a resolution."""
    if res_m not in RES_CHOICES:
        raise ValueError(f"unknown resolution {res_m!r}; expected one of {RES_CHOICES}")
    return f"results/v13_benchmark_{int(res_m)}m"


def control_dir(scenario):
    """The synthetic positive control of one scenario (built at 20 m)."""
    if scenario not in CONTROL_SCENARIOS:
        raise ValueError(f"unknown control scenario {scenario!r}; expected one of "
                         f"{CONTROL_SCENARIOS}")
    return f"{out_root(20)}/control/{scenario}"


def site_root(site, res_m, scenario=None):
    """The results tree holding ``common/``, ``truth/``, ``runs/`` of a site:
    :func:`out_root` for the real sites, the scenario folder for ``pc``."""
    check_res(site, res_m)
    if site == CONTROL_SITE:
        return control_dir(scenario)
    if scenario is not None:
        raise ValueError("only the positive control has scenarios")
    return out_root(res_m)


def common_dir(site, res_m, scenario=None):
    """``<root>/common/<site>``: the common input package."""
    return f"{site_root(site, res_m, scenario)}/common/{site}"


def truth_dir(site, res_m, scenario=None):
    return f"{site_root(site, res_m, scenario)}/truth/{site}"


def runs_dir(site, res_m, scenario=None):
    return f"{site_root(site, res_m, scenario)}/runs/{site}"


def gates_dir(site, res_m):
    return f"{out_root(res_m)}/gates/{site}"


def logs_dir(res_m):
    return f"{out_root(res_m)}/logs"


def frozen_path(res_m):
    return f"{out_root(res_m)}/FROZEN.json"


def lock_path(res_m):
    return f"{out_root(res_m)}/LOCK.json"


def resolve(path):
    """``path`` if it exists, else its copy under :data:`MIRROR_ROOT` if
    that exists, else ``path`` unchanged (the caller reports it missing)."""
    if path is None or os.path.exists(path):
        return path
    mirror = f"{MIRROR_ROOT}/{path}"
    return mirror if os.path.exists(mirror) else path


def site_config(site, res_m, scenario=None):
    """The :class:`SiteConfig` of ``site`` on the ``res_m`` grid.

    Parameters
    ----------
    site : str
        One of :data:`ALL_SITES`.
    res_m : int
        10 (every site, the reported numbers) or 20 (escalda, wadden, ems and
        the positive control; local tests).
    scenario : str, optional
        Required for the positive control ``pc``: "zero" or "profile".
    """
    check_res(site, res_m)
    root = site_root(site, res_m, scenario)
    common = dict(site=site, res_m=int(res_m), pixel_m=float(res_m),
                  boundary=types.MappingProxyType(dict(BOUNDARY[site])),
                  mouth_side=MOUTH_SIDE[site],
                  elevation_truth=ELEVATION_TRUTH[site], judges=JUDGES[site],
                  root=root, scenario=scenario)
    if site in DUTCH_TAG:
        tag = DUTCH_TAG[site]
        prod = dutch_products(site, res_m)
        prod10 = dutch_products(site, 10)
        return SiteConfig(
            cube=dutch_cube(site, res_m), extract=f"{prod}/marea_extract.npz",
            overpass=f"{prod}/overpass_times.json",
            published=types.MappingProxyType({
                "comparison": f"{prod}/comparison_{tag}", "marea": f"{prod}/marea_{tag}",
                "nidem": f"{prod}/nidem_{tag}",
                "granadeiro": f"{prod}/granadeiro2021_{tag}"}),
            # the rule-tz record is the 10 m one at both grids (same dates)
            transition_result=f"{prod10}/marea_{tag}/result.json",
            tag=tag, **common)
    if site == "ferrol":
        return SiteConfig(
            cube="ndwi_cube_ferrol_2023-2025_10m.nc",
            extract="products_ferrol/marea_extract.npz",
            overpass="products_ferrol/overpass_times.json",
            published=types.MappingProxyType({
                "comparison": "products_ferrol/comparison_eot20_g1",
                "marea": "products_ferrol/marea_eot20_g1"}),
            transition_result="products_ferrol/marea_eot20_g1/result.json",
            tag="eot20_g1", **common)
    if site == "villaviciosa":
        return SiteConfig(
            cube="ndwi_cube_villaviciosa_grande_10y.nc",
            extract="products_marea/marea_extract.npz",
            overpass="products_marea/overpass_times.json",
            published=types.MappingProxyType({
                "external": "products_marea/external_products.npz",
                "marea": "products_marea/marea_eot20_g0_2023-2025",
                "nidem": "products_marea/nidem",
                "granadeiro": "products_marea/granadeiro2021"}),
            transition_result="products_marea/marea_eot20_g0_2023-2025/result.json",
            tag="eot20_g0", **common)
    # the positive control: a synthetic cube written by control.synth
    return SiteConfig(
        cube=f"{root}/cube.nc", extract=None, overpass=f"{root}/overpass_times.json",
        published=types.MappingProxyType({}), transition_result=None,
        tag="planted", extra=types.MappingProxyType({"planted": f"{root}/planted.npz"}),
        **common)


def tau_tab_max(K_full):
    """Last row of the exact level table: max(240, 180 (K_full - 1)) min.

    m2a's monotone increments are each bounded by 180 min, so band k can
    reach 180 k; a lag beyond the table is still read exactly (live call).
    """
    lev = FROZEN["levels"]
    K = int(K_full)
    if K < 1:
        raise ValueError(f"K_full must be >= 1, got {K_full!r}")
    return int(max(lev["tau_max_floor"], lev["tau_per_band"] * (K - 1)))


def attainability(inference=None):
    """Can each frozen confirmatory family reject at all? (v2 spec §9.1, §9.8)

    Holm's first step compares the smallest p of a family of m statistics
    with alpha / m. A two-sided bootstrap p with B draws,
    ``min(1, 2 (1 + #{draws beyond 0}) / (B + 1))``, is at least
    ``2 / (B + 1)``; an exact two-sided sign-flip test over n locations
    without zero differences is at least ``2 / 2**n``. A family whose
    smallest attainable p is not below alpha / m can never reject: revision
    1's B = 5,000 gave 4.0e-4 against 0.05 / 128 = 3.9e-4 for P2.

    Parameters
    ----------
    inference : mapping, optional
        A FROZEN['inference']-like section (default the frozen one); its
        ``boot.B_primary``, ``holm_alpha`` and ``families`` are read.

    Returns
    -------
    dict
        ``{family: {"kind", "m", "B" or "n_locations", "p_min", "holm_first",
        "attainable"}}`` for every bootstrap and sign-flip family.
    """
    inf = FROZEN["inference"] if inference is None else inference
    alpha = float(inf["holm_alpha"])
    B = int(inf["boot"]["B_primary"])
    fam = inf["families"]
    out = {}
    for name, m in fam["bootstrap"].items():
        p_min = 2.0 / (B + 1)
        out[str(name)] = {"kind": "bootstrap", "m": int(m), "B": B, "p_min": p_min,
                          "holm_first": alpha / int(m),
                          "attainable": bool(p_min < alpha / int(m))}
    n = int(fam["headline_locations"])
    for name, m in fam["sign_flip"].items():
        p_min = min(1.0, 2.0 / 2.0 ** n)
        out[str(name)] = {"kind": "sign_flip", "m": int(m), "n_locations": n,
                          "p_min": p_min, "holm_first": alpha / int(m),
                          "attainable": bool(p_min < alpha / int(m))}
    return out
