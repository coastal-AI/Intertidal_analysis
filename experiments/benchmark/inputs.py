# -*- coding: utf-8 -*-
"""Validation v2 — the scene inputs every method reads (spec §1.2, §1.10, §1.11).

Question
--------
What exactly are the scenes of a site: which dates, at which instants, on
which grid, with which radiometric offset, and how are they read so that a
Zarr chunk is decoded once? The harness never reads the stores directly: it
reads them through a :class:`SceneSource` (the Zarr stores of
``experiments.zarr_cubes``, or the legacy netCDF cubes for the reproduction
gates only).

Pre-registered rules (docs/validation_v2_spec.md §1.2-1.4, §1.10, §1.11;
user decisions of 2026-10-13 and 2026-10-14)
--------------------------------------------------------------------------
* Record: the store dates in ``[start, end)`` (end exclusive) that have a
  valid instant, in time order; positions are record positions of the
  source (open it over the union period 2017-01-01..2026-01-01 for global
  positions; a window is a selection, :meth:`SceneSource.window`). An
  instant must fall on its label's UTC date. The finite-h_B condition is
  the package's (``keep_dates``).
* Instants (user decision of 2026-10-14): the DEFAULT scene instant is the
  catalogue / product-name time, i.e. the datatake start of
  ``overpass_times.json`` (``instants="catalogue"``; "labels" is an alias),
  exactly as in every production result; using it raises no warning and
  sets no flag. A ``sensing_times.json`` (the optional solar-angle
  diagnostic of ``sensing_time.py``) is read ONLY when its path is given
  explicitly (``instants=<path>``; CLI ``--sensing-times <file>``); no
  file beside the stores is ever picked up implicitly.
* Raw values: int16 spectral bands exactly as stored (no CF decoding);
  SCL and CLD as uint8 after a range check (``obs.scl_u8``, ``obs.cld_u8``).
* Reading: chunk-aligned blocks of 16 dates x 256 rows (the stores' chunks),
  in chunk order, through an LRU cache of decoded chunk blocks
  (:class:`BlockCache`), so a chunk is decoded once when the cache holds
  the bands of one 16-date block (the default 1 GB does for row blocks; a
  smaller cache only costs time). With ``pair20=True`` the row blocks start
  on the 20 m lattice rows, so the 2 x 2 blocks of the 20 m rule never
  straddle two row blocks.
* ``lattice20``: the phase (py, px) of the native 20 m lattice on the 10 m
  grid, measured from the 2 x 2 replication of B8A, B11, B12 and SCL on up
  to 8 scenes with data (:func:`measure_lattice20`): a phase is exact when
  every pair of rows (columns) it pairs is equal in every band and scene;
  None when no phase is exact (box means at (0, 0), declared). Ambiguity
  (both phases exact, uniform data) is resolved by the grid geometry (20 m
  lattice lines at multiples of 20 m), else None.
* Per-scene BOA offset o_s in {0, 1000} (:func:`decide_offsets`, the frozen
  rule written before lock-A, :data:`OFFSET_RULE`, amended 2026-10-19),
  never by date. Strong decisions, labelled as in V1 and tested in this
  order: (0) B03 and B08 identical to the offset-harmonised legacy netCDF
  cube on the date (the store-year's ``<year>.verify.json``,
  :func:`identity_from_verify`: its comparison covering the store's whole
  frame, bound to the store by its date counts and per-date B12 counts)
  -> 0, on a scene with B12 valid on >= 1 % of the frame, in a store-year
  where an identical date also shows (1), (4) or the dark clear water of
  (3); (1) >= 100 negative values in the delivered
  bands -> 0; (4) the frame share of B12 DN in [1, 500) >= 0.002 (on >= 1 %
  of the frame valid) -> 0; (2) all processing baselines < 04.00 -> 0
  (unclear if the clear water says +1000); (3) at least 1,000 CLEAR-water
  pixels (SCL 6 with CLD <= 10 % and a valid B12): a share of B12 DN < 500
  >= 0.1 -> 0, <= 0.02 -> 1000 when the frame's SCL 8/9/10 share is <= 0.5
  (under more cloud no strong evidence; mixed baselines unclear), in
  between no strong evidence. (5) Per year (scenes whose baselines are not
  all < 04.00): its strong 1000 become weak when it has >= 10 strong 0 and
  they are <= 2 % of its strong decisions; when both values remain, the
  strong decisions of the smaller side are 'unclear' (both sides on a tie,
  and then its weak scenes too); every other scene takes the regime of the
  year's strong decisions, else the store's year ``offset_state``, else
  'unclear'. 'unclear' is an error for the observation
  (:class:`OffsetUnclear`) and fails the inputs check. The legacy netCDF
  cubes are offset-harmonised (value = DN - 1000 >= -1000): o_s = 0 for
  every scene. :func:`offset_rule_sha256` (OFFSET_RULE and the code of the
  decision, ``settings.OFFSET_RULE_IMPL``) is written into the inputs
  report; ``prepare`` refuses a report written with another rule.
* Amendment of 2026-10-19 (declared before lock-A, ``OFFSET_RULE
  ["revision"]``; nothing tuned on truth data). The first inputs run on the
  complete stores decided '+1000' on scenes without any offset: thick cloud
  that Sen2Cor labelled water (SCL 6 inside a frame of SCL 8/9) and
  clear-sky bright water; their years' decided scenes then disagreed, so
  every weakly decided scene of those years became 'unclear' and the check
  failed. New: rule (0); the clear-water filter and the cloud guard of
  (2)-(3) (the guard does not depend on CLD); (4) before (2) and (3) at
  0.002, the downloader's own '+1000' bound (dark pixels outrank bright
  water); the dark-water bound 0.1 and a middle water share that is no
  strong evidence instead of 'unclear'; at least 100 negatives for (1); the
  demotion of an isolated strong '+1000' and the contradiction step of (5);
  the baselines as the union of the scene metadata and the stores'
  ``input_baselines``. The scene statistics keep the SCL-water counts
  (``n_water``, ``n_water_low``), the clear-water counts (``n_water_clear``,
  ``n_water_clear_low``, read in the same pass as B12 and SCL) and the SCL
  cloud counts (``n_scl_cloud``, ``n_scl_valid``); statistics written before
  the amendment have no clear-water counts and fall back to the SCL-water
  counts, and the reason says so. Whether CLD <= 10 % removes the cloud
  labelled water is not verified (no legacy cube has CLD): the report keeps
  both counts for every scene.
* Fingerprint of a store-year: sha256 of the consolidated metadata plus the
  sorted (chunk key, size) list; ``full=True`` adds a content hash.
* Scene metadata sidecar ``cubes/<site>/scene_meta.json``
  (:data:`SCENE_META_FORMAT`): per date the STAC items (one per MGRS tile
  and product) with platform, orbits, processing version, cloud cover,
  mean sun and view angles; built by :func:`scene_meta_from_items` from the
  CDSE STAC features (the fetch step does the network query).
* The inputs report (``results/validation_v2/common/inputs_report.json``,
  :func:`inputs_report`): the §1.11 checks; it writes no package and is
  safe to repeat.

Memory: a row block of 16 dates x 256 rows x the frame width per band, and
the cache (``cache_mb``); nothing holds a whole cube.
"""
from __future__ import annotations

import collections
import collections.abc
import hashlib
import json
import os
import re
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import zarr_cubes as zc
from experiments.benchmark import obs, settings

#: the union period of the stores (end exclusive)
UNION_PERIOD = settings.UNION_PERIOD
#: dates per decoded time block (the stores' time chunk)
T_BLOCK = int(zc.CHUNKS["t"])
#: rows per row block (the stores' row chunk)
ROW_BLOCK = int(zc.CHUNKS["y"])
#: bands of the Zarr stores and of the legacy netCDF cubes
ZARR_BANDS = tuple(zc.BANDS)
LEGACY_BANDS = ("B03", "B08", "SCL")
#: bands decoded to uint8 after a range check
U8_BANDS = ("SCL", "CLD")
#: the bands lattice20 is measured on
LATTICE_BANDS = ("B8A", "B11", "B12", "SCL")
#: scenes sampled for lattice20
LATTICE_SCENES = 8
#: default cache budget (MB)
CACHE_MB = 1024
#: the cache of the inputs scan (every chunk is read once)
SCAN_CACHE_MB = 128
#: file names beside the stores
SCENE_META_NAME = "scene_meta.json"
#: the optional solar-angle diagnostic's file name (read only when its path
#: is given explicitly; user decision of 2026-10-14)
SENSING_NAME = "sensing_times.json"
#: the default scene instants: the catalogue / product-name time (the
#: datatake start), as every production result (user decision 2026-10-14)
CATALOGUE = "catalogue"
CATALOGUE_ALIASES = ("catalogue", "labels")
SCENE_META_FORMAT = "pyintertidal-scene-meta/1"
INPUTS_REPORT_FORMAT = "validation-v2-inputs-report/1"
INPUTS_REPORT = "results/validation_v2/common/inputs_report.json"
#: the key of an instant in sensing_times.json
INSTANT_KEY = "instant"

#: the per-scene offset rule as written before lock-A (milestone V1), kept in
#: the amendment's record
_OFFSET_RULE_V1 = (
    "(1) negative values -> 0; (2) all baselines < 04.00 -> 0 (unclear if the water B12 says "
    "+1000); (3) >= 1,000 SCL-water px with valid B12: share of B12 DN < 500 >= 0.5 -> 0, "
    "<= 0.02 -> 1000, else unclear; (4) frame share of B12 DN in [1, 500) > 0.01 on >= 1 % of "
    "the frame valid -> 0; (5) otherwise the regime of the year's strongly decided scenes if "
    "they agree, else the store's year offset_state, else unclear")
#: the V1 parameters the amendment of 2026-10-19 changed (kept in its record)
_OFFSET_RULE_V1_PARAMETERS = {
    "negative_values": "any (n_neg > 0)",
    "water": "SCL 6 with a valid B12 (no CLD, no cloud guard)",
    "water_low_none": 0.5,
    "water_middle_band": "unclear",
    "frame_low_none": "0.01, strictly above, tested after (2) and (3)",
    "regime": "unanimous; a disagreement made every weak scene of the year unclear",
    "baselines": "scene_meta.json, else the stores' input_baselines",
}
#: the verdicts of ``zarr_cubes.compare_with_netcdf`` under which every
#: compared pixel of every common date is equal. Like 'differs', they grant
#: identity only when the comparison covered the store's whole frame
#: (:func:`_covers_store`: identical grids, or a common window that is the
#: whole store frame): the per-date counts cover the common window only, so
#: on a store that reaches beyond the legacy cube a date could be 'identical'
#: with nothing compared
IDENTITY_VERDICTS = ("identical", "identical on the common window")
#: the bands whose identity with the offset-harmonised legacy cube proves
#: 'no offset' (rule 0; the legacy cubes hold B03, B08 and SCL)
IDENTITY_BANDS = ("B03", "B08")
#: the reason of a rule-0 decision (the start of the string)
IDENTITY_REASON = "identical to the offset-harmonised legacy cube"
#: the store-year sidecar of the comparison with the legacy cube
VERIFY_NAME = "verify.json"

#: the frozen per-scene offset rule (§1.10, §2.3); written before lock-A,
#: amended 2026-10-19 (``revision``). Rules are labelled as in V1 and tested in
#: the order (0), (1), (4), (2), (3), then (5) per year.
OFFSET_RULE = {
    "rule": ("(0) B03 and B08 identical to the offset-harmonised legacy netCDF cube on the "
             "date (the store-year's verify.json, its comparison covering the store's whole "
             "frame, bound to the store by its date counts and per-date B12 counts), on a "
             "scene with B12 valid on >= 1 % of the frame, in a store-year where at least one "
             "identical date also shows (1), (4) or the dark clear water of (3) -> 0; "
             "(1) >= 100 negative values in the delivered bands -> 0; "
             "(4) frame share of B12 DN in [1, 500) >= 0.002 on >= 1 % of the frame valid -> 0 "
             "(tested before (2) and (3)); (2) all baselines < 04.00 -> 0 (unclear if the "
             "clear-water B12 says +1000 under a clear sky, as in (3)); (3) >= 1,000 "
             "CLEAR-water px (SCL 6, CLD <= 10 %, valid B12): share of B12 DN < 500 >= 0.1 -> 0; "
             "<= 0.02 -> 1000 when the frame's SCL 8/9/10 share is <= 0.5 (under more cloud no "
             "strong evidence; mixed baselines -> unclear); in between no strong evidence; "
             "(5) per year, over the scenes whose baselines are not all < 04.00: its strong 1000 "
             "become weak when it has >= 10 strong 0 and they are <= 2 % of its strong "
             "decisions; when both values remain, the strong decisions of the smaller side -> "
             "unclear (both sides on a tie, and then the year's weak scenes too); the other "
             "scenes take the regime of the year's remaining strong decisions, else the store's "
             "year offset_state, else unclear"),
    "low_dn": 500,
    "offset_baseline": "04.00",
    "neg_min_px": 100,
    "water_scl": 6,
    "water_cld_max": 10,
    "water_min_px": 1000,
    "water_low_none": 0.1,
    "water_low_offset": 0.02,
    "cloud_scl": [8, 9, 10],
    "water_offset_max_cloud_share": 0.5,
    "frame_min_valid_share": 0.01,
    "frame_low_none": 0.002,
    "regime_min_strong": 10,
    "regime_outlier_share": 0.02,
    "identity_bands": list(IDENTITY_BANDS),
    "identity_verdicts": list(IDENTITY_VERDICTS),
    "revision": {
        "date": "2026-10-19",
        "kind": ("declared amendment before lock-A: a defect that blocked the inputs check, "
                 "not a change of method; nothing tuned on truth data (the new thresholds are "
                 "the downloader's own '+1000' bound, 0.002, or declared round values: CLD 10 %, "
                 "a cloud majority 0.5, a dark-water share 0.1, 100 negative pixels, 10 strong "
                 "decisions and 2 %; checked, not fitted, on a negative control built from a "
                 "legacy cube)"),
        "previous_rule": _OFFSET_RULE_V1,
        "previous_parameters": _OFFSET_RULE_V1_PARAMETERS,
        "changes": [
            ("(0) new, strongest: a scene whose B03 and B08 equal the offset-harmonised legacy "
             "netCDF cube on its date -> 0. The per-date identity comes from the store-year's "
             "<year>.verify.json (zarr_cubes.compare_with_netcdf): every common date when the "
             "verdict is 'identical' or 'identical on the common window', or when the B03 and "
             "B08 pixel counts show no difference; "
             "with 'differs', a common date absent from the listed differing dates when that "
             "list is complete (not truncated at max_dates_listed), or a listed date whose "
             "differences are in SCL only. A verification grants nothing when its comparison "
             "did not cover the store's whole frame (identical grids, or a common window that "
             "is the whole store frame; 'identical on the common window' counts only then, "
             "since the per-date counts cover the common window only), or when its date "
             "counts or per-date B12 counts are not the store's (another download). Not "
             "applied on a scene with B12 valid on < 1 % of the frame. The legacy "
             "cubes come from the same backend, so the identity shows o_s(Zarr) = o_s(legacy): "
             "a store-year uses it only when at least one of its identical dates also shows "
             "negative or dark values (rules 1, 4 or the dark clear water of 3), impossible "
             "with +1000 in both cubes. An identical date whose own pixels say '+1000' stays 0 "
             "and is listed in the report's warnings"),
            ("(3), (2): the water evidence counts only CLEAR water, SCL 6 with CLD <= "
             "water_cld_max % (CLD present) and a valid B12 (n_water_clear, "
             "n_water_clear_low; the SCL-water counts n_water, n_water_low stay in the "
             "statistics). Statistics written before the amendment have no clear-water "
             "counts: they fall back to the SCL-water counts, and the reason says so"),
            ("(3), (2): bright clear water is '+1000' evidence only under a clear sky: the "
             "frame's SCL 8/9/10 share (n_scl_cloud / n_scl_valid, codes 1-11) <= "
             "water_offset_max_cloud_share; under more cloud it is no strong evidence. This "
             "guard does not depend on CLD"),
            ("(3): a clear-water share of B12 DN < 500 >= 0.1 (was 0.5) -> 0 (with +1000 such a "
             "pixel is a reflectance below -0.05); a share between 0.02 and 0.1 is no strong "
             "evidence (was 'unclear', which failed the check without trying (4) or the "
             "regime)"),
            ("(4) is tested right after (1), before (2) and (3), at a frame share >= 0.002 "
             "(was > 0.01, after (3)): the downloader's own year-level '+1000' bound "
             "(radiometry_summary: p90 < 0.002); a B12 DN in [1, 500) with +1000 would be a "
             "reflectance below -0.05, so dark pixels outrank bright water"),
            ("(1) needs >= 100 negative values (was any): with +1000 in the values a negative "
             "int16 can only be an artefact (e.g. a wrapped saturation code)"),
            ("(5) the regime step no longer spreads one false '+1000' to its year: a year's "
             "strong 1000 become weak when it has >= regime_min_strong strong 0 and they are "
             "<= regime_outlier_share of its strong decisions (listed in the report's "
             "warnings). When both values remain, the strong decisions of the smaller side "
             "are 'unclear' (both sides on a tie, and then the year's weak scenes too): the "
             "contradiction fails the check on the contradicting scenes, also in a year without "
             "weak scenes (before, such a year passed silently), and the weak scenes take the "
             "regime of the strong decisions that remain. Strong 0 are never demoted: "
             "negatives, dark pixels and the self-checked identity are impossible with +1000, "
             "bright water is only the absence of dark water"),
            ("the processing baselines of a date are the union of scene_meta.json and the "
             "stores' input_baselines: a date on both sides of 04.00 is 'mixed' (no rule-(2) "
             "shortcut) and is listed in the report's warnings; '0213' and 'N0213' read as "
             "02.13"),
        ],
        "reason": ("The first inputs run on the complete Zarr stores decided '+1000' on scenes "
                   "that have no offset: bright SCL 'water' that is in fact thick cloud (e.g. "
                   "wadden 2025-12-15), and clear-sky bright water (glint, haze). Their year's "
                   "decided scenes then disagreed, so every weakly decided scene of those years "
                   "became 'unclear', which failed the inputs check (V1 rule: unanimous regime, "
                   "dark pixels tested after bright water, a middle water share 'unclear')"),
        "evidence": ("validation_v2 inputs on the complete Zarr stores (CDSE openEO, 2023-2025, "
                     "9 bands, UTM-extent grid), 2026-10-18/19: villaviciosa offsets "
                     "{'no offset': 235, 'unclear': 219, '+1000 in the values': 12} of T = 466, "
                     "wadden {'no offset': 319, 'unclear': 122, '+1000 in the values': 21} of "
                     "T = 462. Wadden 2025-12-15 (reason 'water B12 at or above 500 DN (share "
                     "0.000)'): 7,240 SCL-water px, none below 500 DN; 84 of 9,507,800 valid "
                     "frame px with B12 DN in [1, 500); no negative value. In the 20 m legacy "
                     "cube (ndwi_cube_wadden_2023-2025_20m.nc) that date is 94.6 % SCL 9 and "
                     "5.3 % SCL 8; its 1,810 SCL-6 px have B08 p5/50/95 = 3449/4554/6588 DN and "
                     "a B03 median of 5114 DN (reflectance about 0.45-0.51): thick cloud "
                     "labelled water, not an offset. The openEO --test format report of "
                     "2026-10-13 found 'no offset' (the backend removes the +1000; processing "
                     "baseline 05.11). The legacy cubes are offset-harmonised: in the 10-year "
                     "Villaviciosa legacy cube (ndwi_cube_villaviciosa_grande_10y.nc, "
                     "2016-2025) negative B08 values (impossible with +1000) occur on 10 to 46 "
                     "dates of every year, and none of its 526 dates with >= 50,000 SCL-water "
                     "px has a water B08 1st percentile >= 900 DN"),
        "control": ("A negative control built from the harmonised Villaviciosa SWIR legacy cube "
                    "(swir_cube_villaviciosa_2023-2025.nc, 465 scenes, B11 standing in for B12, "
                    "every SCL-water pixel taken as clear water: the worst case for CLD): a first "
                    "draft of this amendment (identity and clear water only) gave 21 false strong "
                    "'+1000' and 226 'unclear' (the server's V1 run at Villaviciosa: 12 and 219); "
                    "the amended rule gives 0 wrong and 0 'unclear' (2 isolated '+1000' "
                    "demoted; without the cloud guard 14 '+1000' candidates, each 'unclear' by "
                    "the contradiction step). The same pixels with the +1000 put back give 0 "
                    "wrong and 0 'unclear' (177 strong 1000, the rest by the regime). Rule "
                    "variants were compared on this control only to check them; the values kept "
                    "are the declared ones of 'kind'; no truth data was used"),
        "not_verified": ("CLD <= 10 % on the overcast scenes: no legacy cube has CLD. The rerun "
                         "of inputs reports, per scene, n_water and n_water_clear (and "
                         "n_scl_cloud, n_scl_valid); the cloud guard, rule (4) and the regime "
                         "step do not depend on CLD"),
    },
}
OFFSET_STATES = ("+1000 in the values", "no offset", "unclear")
#: the per-scene statistics the rule reads (scan or store attributes)
OFFSET_STAT_KEYS = ("n_px", "n_valid", "n_low", "n_water", "n_water_low", "n_water_clear",
                    "n_water_clear_low", "n_neg", "n_scl_cloud", "n_scl_valid")
#: the strong 0 that are impossible with +1000 in the values (the rule tags
#: of :func:`_strong_rule`): they let a store-year use rule (0)
HARD_ZERO_TAGS = ("negatives", "frame", "dark water")


def log(msg):
    print(msg, flush=True)


class InputsError(RuntimeError):
    """An input violates the reader contract."""


class OffsetUnclear(InputsError):
    """A scene's BOA offset could not be decided (§1.10): an error."""


class GridMismatch(InputsError):
    """Two grids that must be identical differ."""


# ─────────────────────────────────────────────────────────────────────────────
#  the grid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class Grid:
    """Pixel centres (x ascending, y descending, metres) and the CRS."""

    xs: np.ndarray
    ys: np.ndarray
    crs_wkt: str

    def __post_init__(self):
        xs = np.array(self.xs, np.float64)
        ys = np.array(self.ys, np.float64)
        xs.flags.writeable = False
        ys.flags.writeable = False
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)

    @property
    def H(self):
        return int(len(self.ys))

    @property
    def W(self):
        return int(len(self.xs))

    @property
    def shape(self):
        return self.H, self.W

    @property
    def epsg(self):
        return zc._epsg_from_wkt(self.crs_wkt)

    def same_as(self, other):
        """Identical shape, coordinates (exact) and CRS."""
        return (self.shape == other.shape and np.array_equal(self.xs, other.xs)
                and np.array_equal(self.ys, other.ys) and zc.crs_equal(self.crs_wkt, other.crs_wkt))

    def describe(self):
        dx = float(self.xs[1] - self.xs[0]) if self.W > 1 else None
        dy = float(self.ys[1] - self.ys[0]) if self.H > 1 else None
        return {"H": self.H, "W": self.W, "x0": float(self.xs[0]), "y0": float(self.ys[0]),
                "dx": dx, "dy": dy, "epsg": self.epsg, "sha256": self.sha256()}

    def sha256(self):
        h = hashlib.sha256(self.xs.astype("<f8").tobytes())
        h.update(self.ys.astype("<f8").tobytes())
        h.update(str(self.epsg or self.crs_wkt).encode("utf-8"))
        return h.hexdigest()

    def lonlat_centre(self):
        """(lon, lat) of the frame centre."""
        from pyproj import CRS, Transformer

        tf = Transformer.from_crs(CRS.from_user_input(self.crs_wkt), "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(float(self.xs.mean()), float(self.ys.mean()))
        return float(lon), float(lat)


def grid_of(ds):
    """The :class:`Grid` of an xarray dataset with ``x``, ``y`` and a ``crs``
    (or ``spatial_ref``) variable."""
    return Grid(ds["x"].values, ds["y"].values, zc.grid_info(ds)["crs_wkt"] or "")


def grid_from_geotiff(path):
    """The :class:`Grid` of a north-up GeoTIFF (pixel centres)."""
    import rasterio

    with rasterio.open(path) as r:
        tr, H, W = r.transform, r.height, r.width
        if tr.b != 0 or tr.d != 0:
            raise InputsError(f"{path}: rotated transform")
        xs = tr.c + tr.a * (np.arange(W) + 0.5)
        ys = tr.f + tr.e * (np.arange(H) + 0.5)
        return Grid(xs, ys, r.crs.to_wkt() if r.crs else "")


def assert_same_grid(a, b, what):
    """Raise :class:`GridMismatch` unless ``a`` and ``b`` are identical."""
    if not a.same_as(b):
        raise GridMismatch(f"{what}: grids differ ({a.describe()} vs {b.describe()})")


def expected_lattice20(grid):
    """The 20 m lattice phase implied by the coordinates: Sentinel-2 20 m
    pixel edges lie on multiples of 20 m in the tile's UTM CRS, so the phase
    is 0 when the frame's first 10 m pixel starts a 20 m pixel, else 1."""
    dx = float(grid.xs[1] - grid.xs[0]) if grid.W > 1 else 10.0
    dy = float(grid.ys[0] - grid.ys[1]) if grid.H > 1 else 10.0
    left = float(grid.xs[0]) - dx / 2.0
    top = float(grid.ys[0]) + dy / 2.0
    px = 0 if abs(((left + 1e-6) % 20.0)) < 1e-3 else 1
    py = 0 if abs(((top + 1e-6) % 20.0)) < 1e-3 else 1
    return py, px


# ─────────────────────────────────────────────────────────────────────────────
#  lattice20
# ─────────────────────────────────────────────────────────────────────────────

def lattice_votes(a):
    """``{"y": {p: [equal, pairs]}, "x": {p: [equal, pairs]}}`` of one 2-D
    array: for phase p, the pairs of rows (columns) r, r + 1 with r = p mod 2
    and how many of their elements are equal."""
    a = np.asarray(a)
    out = {"y": {}, "x": {}}
    for ax, n in (("y", a.shape[0]), ("x", a.shape[1])):
        for p in (0, 1):
            r = np.arange(p, n - 1, 2)
            if ax == "y":
                eq = int(np.count_nonzero(a[r, :] == a[r + 1, :]))
                tot = int(r.size * a.shape[1])
            else:
                eq = int(np.count_nonzero(a[:, r] == a[:, r + 1]))
                tot = int(r.size * a.shape[0])
            out[ax][p] = [eq, tot]
    return out


def measure_lattice20(arrays, expected=None):
    """The native 20 m lattice phase from replicated 20 m bands.

    Parameters
    ----------
    arrays : iterable of (band, 2-D array)
        Raw values of the 20 m bands on the 10 m grid (any number of scenes;
        each row block must start on an even row).
    expected : (py, px), optional
        :func:`expected_lattice20`, used only to resolve ambiguity.

    Returns
    -------
    dict
        ``phase`` ((py, px) or None), ``exact`` (bool), ``votes`` per axis
        and phase, ``ambiguous`` axes, ``expected``, ``agrees_with_grid``.
    """
    tot = {"y": {0: [0, 0], 1: [0, 0]}, "x": {0: [0, 0], 1: [0, 0]}}
    bands = set()
    for band, a in arrays:
        bands.add(band)
        v = lattice_votes(a)
        for ax in ("y", "x"):
            for p in (0, 1):
                tot[ax][p][0] += v[ax][p][0]
                tot[ax][p][1] += v[ax][p][1]
    phase, amb = [], []
    for k, ax in enumerate(("y", "x")):
        exact = [p for p in (0, 1) if tot[ax][p][1] > 0 and tot[ax][p][0] == tot[ax][p][1]]
        if len(exact) == 1:
            phase.append(exact[0])
        elif len(exact) == 2:
            amb.append(ax)
            phase.append(expected[k] if expected is not None else None)
        else:
            phase.append(None)
    ph = tuple(phase) if None not in phase else None
    return {"phase": list(ph) if ph else None, "exact": ph is not None,
            "votes": {ax: {str(p): tot[ax][p] for p in (0, 1)} for ax in tot},
            "ambiguous": amb, "bands": sorted(bands),
            "expected": list(expected) if expected is not None else None,
            "agrees_with_grid": (None if expected is None or ph is None
                                 else list(ph) == list(expected))}


# ─────────────────────────────────────────────────────────────────────────────
#  the block cache
# ─────────────────────────────────────────────────────────────────────────────

class BlockCache:
    """LRU cache of decoded chunk blocks, bounded in bytes.

    Keys are ``(source part, band, time block, row chunk)``; values are raw
    arrays of one time block x one row chunk x the frame width. ``loads``
    counts decodes, ``hits`` reuses; a block larger than the budget is
    returned without being kept.
    """

    def __init__(self, max_mb=CACHE_MB):
        self.max_bytes = int(float(max_mb) * 1e6)
        self._d = collections.OrderedDict()
        self.nbytes = 0
        self.loads = 0
        self.hits = 0
        self.evictions = 0

    def get(self, key, load):
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        v = load()
        self.loads += 1
        if v.nbytes <= self.max_bytes:
            while self._d and self.nbytes + v.nbytes > self.max_bytes:
                _, old = self._d.popitem(last=False)
                self.nbytes -= old.nbytes
                self.evictions += 1
            self._d[key] = v
            self.nbytes += v.nbytes
        return v

    def clear(self):
        self._d.clear()
        self.nbytes = 0

    def stats(self):
        return {"loads": self.loads, "hits": self.hits, "evictions": self.evictions,
                "held_mb": round(self.nbytes / 1e6, 1), "max_mb": round(self.max_bytes / 1e6, 1)}


# ─────────────────────────────────────────────────────────────────────────────
#  instants
# ─────────────────────────────────────────────────────────────────────────────

def _naive_utc(v):
    if v is None:
        return pd.NaT
    ts = pd.Timestamp(v)
    if ts is pd.NaT:
        return ts
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def read_sensing_times(src):
    """``{date: Timestamp (naive UTC) or NaT}`` from ``sensing_times.json``.

    Accepts the G-time format ``{"dates": {date: {"instant": iso or null,
    ...}}}``, ``{date: {"instant": ...}}`` and the label format ``{date:
    iso}`` (``overpass_times.json``); ``src`` may be a path or a mapping.
    """
    if isinstance(src, (str, os.PathLike)):
        with open(src, encoding="utf-8") as f:
            src = json.load(f)
    d = src.get("dates", src) if isinstance(src, dict) else src
    out = {}
    for date, v in dict(d).items():
        if isinstance(v, dict):
            v = v.get(INSTANT_KEY)
        out[str(date)[:10]] = _naive_utc(v)
    return out


def resolve_instants(instants, labels):
    """The scene instants and where they come from (user decision of
    2026-10-14).

    Parameters
    ----------
    instants : None, "catalogue", "labels", a path or a mapping
        None, "catalogue" or its alias "labels": the catalogue times
        (``labels()``, the datatake start of the product names: the default,
        no warning, no flag). A path: a ``sensing_times.json`` given
        explicitly (the optional diagnostic). A mapping: ``{date: instant}``.
    labels : callable
        Returns the catalogue ``{date: time}`` (raises when absent).

    Returns
    -------
    (dict, str, str or None)
        ``{date: naive UTC Timestamp or NaT}``, the source ("catalogue",
        "sensing_times" or "given") and the file read (None for a mapping).
    """
    if instants is None or (isinstance(instants, str) and instants in CATALOGUE_ALIASES):
        lab, path = labels()
        return {str(d)[:10]: _naive_utc(t) for d, t in lab.items()}, CATALOGUE, path
    if isinstance(instants, str) and instants in ("sensing", "auto"):
        raise ValueError(f"instants={instants!r} is no longer a mode (user decision "
                         "2026-10-14): the scene instants are the catalogue times by default; "
                         "a sensing_times.json is used only when its PATH is given "
                         "(--sensing-times <file>)")
    if isinstance(instants, (str, os.PathLike)):
        if not os.path.isfile(instants):
            raise FileNotFoundError(f"{instants}: no such sensing_times.json")
        return read_sensing_times(instants), "sensing_times", str(instants).replace("\\", "/")
    return ({str(k)[:10]: _naive_utc(v) for k, v in dict(instants).items()}, "given", None)


def _check_instants(instants):
    """Instants must fall on their label's UTC date."""
    bad = [d for d, t in instants.items()
           if t is not pd.NaT and not pd.isna(t) and t.strftime("%Y-%m-%d") != d]
    if bad:
        raise InputsError(f"instants not on their label date: {bad[:5]}")


def instants_sha(instants):
    """sha256 of a resolved ``{date: instant or NaT}`` mapping (ISO instants,
    None for NaT, dates sorted): the identity of a record's instants."""
    canon = {str(d)[:10]: (None if (v is None or v is pd.NaT or pd.isna(v))
                           else pd.Timestamp(v).isoformat())
             for d, v in sorted(dict(instants).items())}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode("utf-8")).hexdigest()


def catalogue_labels(site, root=None):
    """The catalogue-times reader of a site's Zarr stores (for
    :func:`resolve_instants`): ``<root>/<site>/overpass_times.json``."""
    def labels():
        p = zc.overpass_path(site, root)
        lab = zc.read_overpass(site, root)
        if not lab:
            raise FileNotFoundError(f"{p}: no catalogue times (the downloader writes "
                                    "overpass_times.json beside the stores)")
        return lab, str(p).replace("\\", "/")
    return labels


def file_labels(labels_path):
    """The catalogue-times reader of a legacy cube (its production
    ``overpass_times.json``)."""
    def labels():
        if labels_path is None:
            raise ValueError("the catalogue instants need labels_path (overpass_times.json)")
        return read_sensing_times(labels_path), str(labels_path).replace("\\", "/")
    return labels


def instants_identity(instants, labels, years=None):
    """What a run's instants are, without opening a store: ``{"source",
    "path", "sha256"}`` (:func:`resolve_instants`, :func:`instants_sha`).
    The package compares it with the record's: another source or file is
    refused, other content of the same file rebuilds the record.
    ``years``: hash only the dates of these calendar years (the record's),
    so dates the downloader adds for other years change nothing."""
    d, source, path = resolve_instants(instants, labels)
    if years is not None:
        ys = {str(int(y)) for y in years}
        d = {k: v for k, v in d.items() if str(k)[:4] in ys}
    return {"source": source, "path": path, "sha256": instants_sha(d)}


# ─────────────────────────────────────────────────────────────────────────────
#  the per-scene offset (§1.10, §2.3)
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_version(v):
    """A processing baseline as a number (``"05.11"``, ``"N0511"``, ``"0511"``,
    ``"5.11"`` or 5.11 -> 5.11), or None when it is not one."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    m = re.fullmatch(r"N?(\d{1,2})\.(\d{1,2})", s) or re.fullmatch(r"N?(\d{2})(\d{2})", s)
    if not m:
        return None
    return float(f"{int(m[1])}.{m[2]}")


def _baseline_class(versions):
    """"offset" (every version >= 04.00), "none" (every one < 04.00),
    "mixed", or None (unknown)."""
    vs = [x for x in (_baseline_version(v) for v in (versions or ())) if x is not None]
    if not vs:
        return None
    lim = float(OFFSET_RULE["offset_baseline"])
    if all(v >= lim for v in vs):
        return "offset"
    if all(v < lim for v in vs):
        return "none"
    return "mixed"


def scene_offset_stats(b12_raw, scl_u8, b_neg_raw=(), fill=obs.FILL, cld_u8=None):
    """The per-scene statistics of the offset rule (one scene or a block of
    scenes, (..., H, W)).

    Returns a dict of int arrays over the leading axes: ``n_px``,
    ``n_valid`` (B12 not fill, not 0), ``n_low`` (valid with 1 <= DN <
    500), ``n_water`` (SCL 6 with valid B12), ``n_water_low``, ``n_neg``
    (values < 0, fill excluded, in B12 and every array of ``b_neg_raw``)
    and, when ``cld_u8`` (the CLD of the same scenes, checked uint8 or raw)
    is given, ``n_water_clear`` (SCL 6 with valid B12 and a present CLD <=
    ``OFFSET_RULE["water_cld_max"]``) and ``n_water_clear_low`` (amendment
    of 2026-10-19). Without CLD the clear-water counts are absent and the
    rule falls back to the SCL-water counts (declared in the reason).
    """
    b = np.asarray(b12_raw)
    s = np.asarray(scl_u8)
    low = int(OFFSET_RULE["low_dn"])
    ax = (-2, -1)
    valid = (b != fill) & (b != 0)
    lowv = valid & (b >= 1) & (b < low)
    water = valid & (s == int(OFFSET_RULE["water_scl"]))
    neg = ((b < 0) & (b != fill)).sum(axis=ax)
    for a in b_neg_raw:
        a = np.asarray(a)
        neg = neg + ((a < 0) & (a != fill)).sum(axis=ax)
    out = {"n_px": np.full(b.shape[:-2], b.shape[-2] * b.shape[-1], np.int64),
           "n_valid": valid.sum(axis=ax), "n_low": lowv.sum(axis=ax),
           "n_water": water.sum(axis=ax), "n_water_low": (water & lowv).sum(axis=ax),
           "n_neg": neg}
    if cld_u8 is not None:
        c = np.asarray(cld_u8)
        if c.dtype != np.uint8:
            c = obs.cld_u8(c, fill)
        if c.shape != b.shape:
            raise ValueError(f"CLD shape {c.shape} != B12 shape {b.shape}")
        clear = water & (c != obs.CLD_NODATA_U8) & (c <= int(OFFSET_RULE["water_cld_max"]))
        out["n_water_clear"] = clear.sum(axis=ax)
        out["n_water_clear_low"] = (clear & lowv).sum(axis=ax)
    return out


def scl_cloud_counts(scl):
    """``{"n_scl_cloud", "n_scl_valid"}`` of one scene's SCL histogram (the 13
    counts of ``obs.scl_counts``: codes 0..11, then no data): the pixels of
    the cloud classes ``OFFSET_RULE["cloud_scl"]`` (8, 9, 10) and of the codes
    1..11 (amendment of 2026-10-19: the cloud guard of rule 3)."""
    sc = [int(x) for x in scl]
    return {"n_scl_cloud": sum(sc[c] for c in OFFSET_RULE["cloud_scl"]),
            "n_scl_valid": sum(sc[1:12])}


def _cloud_share(st):
    """The frame's SCL 8/9/10 share of one scene's statistics, or None
    without SCL counts."""
    nc, ns = st.get("n_scl_cloud"), st.get("n_scl_valid")
    if nc is None or not ns:
        return None
    return int(nc) / int(ns)


def _water_counts(st):
    """``(n, n_low, what, note)`` of the water evidence of one scene: the
    CLEAR-water counts (amendment of 2026-10-19), else, for statistics
    written before it (no clear-water counts), the SCL-water counts as the
    V1 rule read them (the note says so); ``(None, None, "", "")`` without
    water counts."""
    cmax = OFFSET_RULE["water_cld_max"]
    if st.get("n_water_clear") is not None and st.get("n_water_clear_low") is not None:
        n = int(st["n_water_clear"])
        return n, int(st["n_water_clear_low"]), "clear water", f"of {n} px with CLD <= {cmax} %"
    if st.get("n_water") is not None and st.get("n_water_low") is not None:
        n = int(st["n_water"])
        return (n, int(st["n_water_low"]), "water",
                f"of {n} SCL-water px; statistics written before the 2026-10-19 amendment: "
                "no clear-water counts, CLD not used")
    return None, None, "", ""


def _strong_rule(st, bl, ident=None):
    """The strong decision of one scene: ``(value, reason, tag)`` with value
    0, 1000, "unclear" or None (no strong evidence), and the tag of the rule
    that decided ("identity", "negatives", "frame", "baseline", "dark water",
    "bright water", "baseline vs water", "mixed baselines", or None).
    Rules in the order (0), (1), (4), (2), (3) of :data:`OFFSET_RULE`.
    ``ident``: the note of the scene's identity with the offset-harmonised
    legacy cube (rule 0, :func:`identity_from_verify`), or None."""
    r = OFFSET_RULE
    low = r["low_dn"]
    nv, npx = st.get("n_valid"), st.get("n_px")
    floor = r["frame_min_valid_share"]
    # (0) the identity with the legacy cube; not on a frame with B12 valid on < 1 %
    if ident and (nv is None or (nv > 0 and (not npx or nv >= floor * npx))):
        return 0, f"{IDENTITY_REASON} ({ident})", "identity"
    # (1) negative values: impossible with +1000 (beyond a few artefacts)
    nneg = st.get("n_neg")
    if nneg is not None and nneg >= r["neg_min_px"]:
        return 0, (f"negative values in the delivered bands ({int(nneg)} px >= "
                   f"{r['neg_min_px']}): no offset"), "negatives"
    # (4) dark frame pixels: with +1000 a DN in [1, 500) is a reflectance below -0.05
    if nv is not None and npx and nv > 0 and nv >= floor * npx and st.get("n_low") is not None:
        fs = st["n_low"] / nv
        if fs >= r["frame_low_none"]:
            return 0, (f"frame B12 below {low} DN (share {fs:.4f} of {int(nv)} valid px >= "
                       f"{r['frame_low_none']})"), "frame"
    nw, nwl, what, note = _water_counts(st)
    wshare = (nwl / nw) if nw else None
    has_water = nw is not None and nw >= r["water_min_px"]
    cs = _cloud_share(st)
    sky = ("frame SCL 8/9/10 share unknown" if cs is None else
           f"frame SCL 8/9/10 share {cs:.3f}")
    bright = has_water and wshare <= r["water_low_offset"]
    cloudy = cs is not None and cs > r["water_offset_max_cloud_share"]
    # (2) every baseline < 04.00
    if bl == "none":
        if bright and not cloudy:
            return "unclear", (f"baseline < 04.00 but the {what} B12 is >= {low} DN (share "
                               f"{wshare:.3f} {note}; {sky})"), "baseline vs water"
        return 0, "processing baseline < 04.00", "baseline"
    # (3) the clear water
    if has_water:
        if wshare >= r["water_low_none"]:
            return 0, f"{what} B12 below {low} DN (share {wshare:.3f} {note})", "dark water"
        if bright:
            if cloudy:
                return None, (f"{what} B12 at or above {low} DN under cloud (share {wshare:.3f} "
                              f"{note}; {sky} > {r['water_offset_max_cloud_share']}): no strong "
                              "evidence"), None
            if bl == "mixed":
                return "unclear", (f"mixed baselines with the {what} B12 >= {low} DN (share "
                                   f"{wshare:.3f} {note}; {sky})"), "mixed baselines"
            return 1000, (f"{what} B12 at or above {low} DN (share {wshare:.3f} {note}; "
                          f"{sky})"), "bright water"
        return None, (f"{what} B12 partly below {low} DN (share {wshare:.3f} {note}, between "
                      f"{r['water_low_offset']} and {r['water_low_none']}): no strong "
                      "evidence"), None
    return None, "no strong evidence", None


def _strong(st, bl, ident=None):
    """The strong decision of one scene, or (None, reason): ``(value,
    reason)`` of :func:`_strong_rule`."""
    val, why, _ = _strong_rule(st, bl, ident)
    return val, why


def _identity_map(identical):
    """``{date: note}`` from a mapping or an iterable of dates (a list, a
    numpy array, a pandas Index; None: {})."""
    if identical is None:
        return {}
    if isinstance(identical, collections.abc.Mapping):
        return {str(k)[:10]: (str(v) if v else VERIFY_NAME) for k, v in identical.items()}
    return {str(k)[:10]: VERIFY_NAME for k in identical}


def decide_offsets(stats, baselines=None, year_state=None, identical=None):
    """The frozen per-scene offset decision (:data:`OFFSET_RULE`).

    Parameters
    ----------
    stats : dict
        ``{date: {"n_px", "n_valid", "n_low", "n_water", "n_water_low",
        "n_water_clear", "n_water_clear_low", "n_neg", "n_scl_cloud",
        "n_scl_valid"}}`` (:func:`scene_offset_stats`,
        :func:`scl_cloud_counts`; keys may be missing, e.g. from the store's
        ``radiometry`` counts only, or the clear-water counts in statistics
        written before the amendment of 2026-10-19).
    baselines : dict, optional
        ``{date: [processing versions]}`` (:meth:`ZarrSource.baselines`).
    year_state : dict, optional
        ``{year: offset_state}`` of the stores' ``radiometry`` attribute.
    identical : dict or iterable, optional
        ``{date: note}`` (or dates) whose B03 and B08 equal the
        offset-harmonised legacy cube (rule 0, :func:`identity_from_verify`).
        A year uses them only when one of its identical dates also shows
        negative or dark values (:data:`HARD_ZERO_TAGS`).

    Returns
    -------
    pandas.DataFrame
        Indexed by date: ``offset`` (0, 1000 or NaN), ``state`` (one of
        :data:`OFFSET_STATES`), ``reason``, ``strong`` (bool) and ``rule``
        (the tag of what decided: a :func:`_strong_rule` tag, "demoted",
        "contradiction", "regime", "year state" or "no evidence").
        ``attrs``: ``identity_years`` ({year: note} for the years with
        identical dates), ``identity_pixels_1000`` (identical dates whose own
        pixels say +1000), ``demoted`` and ``contradicted`` (dates).
    """
    r = OFFSET_RULE
    baselines = baselines or {}
    year_state = {str(k): v for k, v in (year_state or {}).items()}
    identical = _identity_map(identical)
    dates = sorted(stats)
    bls = {d: _baseline_class(baselines.get(d)) for d in dates}
    # the scenes' own pixel evidence: rules (1)-(4) without the identity
    pix = {d: _strong_rule(dict(stats[d]), bls[d]) for d in dates}
    # rule (0), per store-year: the legacy cubes come from the same backend, so the identity
    # is used only where one identical date shows values impossible with +1000 in both cubes
    by_year = collections.defaultdict(list)
    for d in dates:
        if d in identical:
            by_year[d[:4]].append(d)
    use, notes = {}, {}
    for y, ds in sorted(by_year.items()):
        hard = [d for d in ds if pix[d][2] in HARD_ZERO_TAGS]
        use[y] = bool(hard)
        notes[y] = (f"identity used: {len(hard)} of its {len(ds)} identical dates show negative "
                    "or dark B12 values" if hard else
                    f"identity NOT used: none of its {len(ds)} identical dates shows negative or "
                    "dark B12 values (both cubes could carry the +1000)")
    rows, pix1000 = {}, []
    for d in dates:
        if use.get(d[:4]) and d in identical:
            val, why, tag = _strong_rule(dict(stats[d]), bls[d], identical[d])
        else:
            val, why, tag = pix[d]
        if tag == "identity" and pix[d][0] in (1000, "unclear"):
            pv, pw, _ = pix[d]
            why = f"{why}; its pixels alone say {'+1000' if pv == 1000 else 'unclear'}: {pw}"
            if pv == 1000:
                pix1000.append(d)
        rows[d] = {"value": val, "reason": why, "baseline": bls[d], "tag": tag}
    # (5) per year: an isolated strong 1000 beside many strong 0 is demoted to weak; when both
    # values remain, the smaller side contradicts its year ('unclear'; both sides on a tie)
    strong = collections.defaultdict(lambda: {0: [], 1000: []})
    for d, row in rows.items():
        if row["value"] in (0, 1000) and row["baseline"] != "none":
            strong[d[:4]][row["value"]].append(d)
    demoted, contradicted, contested = [], [], set()
    for y, g in sorted(strong.items()):
        n0, n1 = len(g[0]), len(g[1000])
        if not n1 or not n0:
            continue
        if n0 >= r["regime_min_strong"] and n1 <= r["regime_outlier_share"] * (n0 + n1):
            for d in g[1000]:
                rows[d].update(value=None, tag="demoted", reason=(
                    f"{rows[d]['reason']}; demoted: {n1} strong '+1000' beside the year's {n0} "
                    f"strong 0 (<= {r['regime_outlier_share']:.0%} of {n0 + n1})"))
                demoted.append(d)
            continue
        if n0 == n1:
            contested.add(y)
        nid = sum(1 for d in g[0] if rows[d]["tag"] == "identity")
        by = f", {nid} of them by the identity with the legacy cube" if nid else ""
        for side, other, n_other in ((1000, 0, n0), (0, 1000, n1)):
            if len(g[side]) > n_other:
                continue                            # the larger side keeps its decisions
            what = (f"{n_other} strong 0{by}" if other == 0 else f"{n_other} strong 1000")
            for d in g[side]:
                rows[d].update(value="unclear", tag="contradiction", reason=(
                    f"{rows[d]['reason']}; contradicts the year's {what}"))
                contradicted.append(d)
    regime = {}
    for d, row in rows.items():
        if row["value"] in (0, 1000) and row["baseline"] != "none":
            regime.setdefault(d[:4], set()).add(row["value"])
    out = []
    for d, row in rows.items():
        val, why, tag = row["value"], row["reason"], row["tag"]
        strong_ = val in (0, 1000)
        if val is None:
            reg = regime.get(d[:4], set())
            ys = year_state.get(d[:4])
            tail = "" if why == "no strong evidence" else f"; {why}"
            if d[:4] in contested or len(reg) > 1:  # a tie: no regime ((5) leaves one value)
                val, why, tag = ("unclear", f"the year's decided scenes disagree{tail}",
                                 "no evidence")
            elif len(reg) == 1:
                val = next(iter(reg))
                why, tag = f"the regime of the year's decided scenes ({val}){tail}", (
                    "demoted" if tag == "demoted" else "regime")
            elif ys == "+1000 in the values":
                val, why, tag = 1000, f"the store's year offset_state{tail}", "year state"
            elif ys == "no offset":
                val, why, tag = 0, f"the store's year offset_state{tail}", "year state"
            else:
                val, why, tag = "unclear", f"no evidence (year state {ys!r}){tail}", "no evidence"
        state = ("unclear" if val == "unclear" else
                 "+1000 in the values" if val == 1000 else "no offset")
        out.append({"date": d, "offset": np.nan if val == "unclear" else float(val),
                    "state": state, "reason": why, "strong": bool(strong_), "rule": tag})
    df = pd.DataFrame(out, columns=["date", "offset", "state", "reason", "strong", "rule"])
    df = df.set_index("date")
    df.attrs.update(identity_years=notes, identity_pixels_1000=pix1000, demoted=demoted,
                    contradicted=contradicted)
    return df


def radiometry_stats(attrs):
    """``{date: {"n_valid", "n_low"}}`` from a store's ``radiometry``
    attribute (the downloader's per-date counts of B12, else B11), plus the
    year's ``offset_state`` and the band used."""
    rad = zc.store_radiometry(attrs) or {}
    per = rad.get("per_date_valid_low") or {}
    return ({d: {"n_valid": int(v), "n_low": int(lo)} for d, (v, lo) in per.items()},
            rad.get("offset_state"), rad.get("band"))


def read_verify(site, year, root=None):
    """The store-year's ``<year>.verify.json`` (the downloader's comparison
    with the legacy netCDF cube, ``zarr_cubes.compare_with_netcdf``): a
    dict, None when absent, or ``{"verdict": "unreadable verify.json",
    "error": ...}``."""
    p = zc.sidecar_path(site, year, VERIFY_NAME, root)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            v = json.load(f)
    except (OSError, ValueError) as e:
        return {"verdict": "unreadable verify.json", "error": f"{type(e).__name__}: {e}"}
    if not isinstance(v, dict):
        return {"verdict": "unreadable verify.json", "error": "not a JSON object"}
    return v


def _count(x):
    """An integer count from a JSON value, or None."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _bands_clean(bands):
    """Every band of :data:`IDENTITY_BANDS` was compared on some pixels and
    differs on none (values, or no data in one only), over all dates."""
    for b in IDENTITY_BANDS:
        s = bands.get(b)
        if not isinstance(s, collections.abc.Mapping) or not (_count(s.get("equal")) or 0) > 0:
            return False
        if any(_count(s.get(k)) != 0 for k in ("value_differs", "nodata_only_store",
                                                "nodata_only_reference")):
            return False
    return True


def _per_date_counts(per, dates):
    """``{date: (valid, low)}`` of a radiometry ``per_date_valid_low`` mapping
    on ``dates`` (raises on a malformed entry)."""
    ds = set(dates)
    out = {}
    for k, v in dict(per).items():
        d = str(k)[:10]
        if d in ds:
            out[d] = tuple(int(x) for x in v)
    return out


def _covers_store(verify):
    """True when the comparison of a verify.json saw every pixel of the
    store's frame: identical grids, or a common window that is the whole
    store frame (a store inside the legacy cube, e.g. an extract); False
    otherwise or when the file has no grid check. The per-date counts of
    ``zarr_cubes.compare_with_netcdf`` cover the common window only."""
    g = verify.get("grid")
    if not isinstance(g, collections.abc.Mapping) or g.get("crs_equal") is False:
        return False
    if g.get("identical") is True:
        return True
    cw, st = g.get("common_window"), g.get("store")
    if not isinstance(cw, collections.abc.Mapping) or not isinstance(st, collections.abc.Mapping):
        return False
    try:
        h, w = (int(x) for x in st["shape"])
        return ([int(x) for x in cw["store_rows"]] == [0, h]
                and [int(x) for x in cw["store_cols"]] == [0, w])
    except (KeyError, TypeError, ValueError):
        return False


def identity_from_verify(verify, store_dates, year=None, radiometry=None):
    """Rule (0) of :data:`OFFSET_RULE` for one store-year: the store dates
    whose B03 and B08 equal the offset-harmonised legacy netCDF cube on
    every compared pixel, from the year's verify.json.

    A common date is identical when the verdict is one of
    :data:`IDENTITY_VERDICTS`, or the B03 and B08 pixel counts show no
    difference over all dates, or (verdict 'differs') it is absent from
    ``dates_with_differences_listed`` and that list is complete (not
    truncated at ``max_dates_listed``), or it is listed with differences in
    SCL only. Nothing is granted when the comparison did not cover the
    store's whole frame (:func:`_covers_store`; the per-date counts cover
    the common window only), when the verification lacks B03 or B08,
    compared no valid pixel, saw duplicated dates or is malformed, or when
    it is bound to another download: its date counts are not the store's,
    or its per-date B12 counts (``radiometry``, kept by the downloader) are
    not the store's.

    Parameters
    ----------
    verify : dict or None
        :func:`read_verify`.
    store_dates : iterable of str
        Every date of the store (not only the record's).
    year : int, optional
        The store's year (checked against the file's ``year``).
    radiometry : dict, optional
        The store's ``radiometry`` attribute (``zarr_cubes.store_radiometry``):
        its ``per_date_valid_low`` must equal the verification's on the
        window when both are present.

    Returns
    -------
    (dict, str)
        ``{date: note}`` of the identical dates, and a note on the year.
    """
    if not verify:
        return {}, f"no {VERIFY_NAME}"
    name = VERIFY_NAME if year is None else f"{int(year)}.{VERIFY_NAME}"
    if not isinstance(verify, collections.abc.Mapping):
        return {}, f"{name}: not a JSON object, no identity"
    verdict = verify.get("verdict")
    vy = verify.get("year")
    if year is not None and vy is not None and str(vy) != str(int(year)):
        return {}, f"{name} is for the year {vy}: no identity"
    if verdict not in IDENTITY_VERDICTS + ("differs",):
        return {}, f"{name}: verdict {verdict!r}, no identity"
    if not _covers_store(verify):
        return {}, (f"{name}: verdict {verdict!r}, but its comparison did not cover the store's "
                    "whole frame (grids of another extent, or no grid check): the per-date "
                    "counts cover the common window only, no identity")
    if not set(IDENTITY_BANDS) <= set(verify.get("bands_compared") or ()):
        return {}, f"{name}: B03 and B08 were not both compared, no identity"
    bands = verify.get("bands") or {}
    dd = verify.get("dates") or {}
    listed = verify.get("dates_with_differences_listed") or {}
    if not all(isinstance(x, collections.abc.Mapping) for x in (bands, dd, listed)):
        return {}, (f"{name}: malformed (its bands, dates or listed dates are not mappings), no "
                    "identity")
    if not all(isinstance(bands.get(b), collections.abc.Mapping)
               and (_count(bands[b].get("equal")) or 0)
               + (_count(bands[b].get("value_differs")) or 0) > 0 for b in IDENTITY_BANDS):
        return {}, f"{name}: no valid B03 / B08 pixel compared, no identity"
    if dd.get("duplicated_dates"):
        return {}, f"{name}: duplicated dates in the comparison, no identity"
    try:
        w0, w1 = (str(x)[:10] for x in verify["window"])
        p0, p1 = (str(x)[:10] for x in verify["reference_period"])
    except (KeyError, TypeError, ValueError):
        return {}, f"{name}: no window or reference period, no identity"
    zd = sorted({str(d)[:10] for d in store_dates if w0 <= str(d)[:10] < w1})
    only = {str(d)[:10] for d in dd.get("only_store") or ()}
    common = [d for d in zd if p0 <= d < p1 and d not in only]
    if _count(dd.get("store")) != len(zd) or _count(dd.get("common")) != len(common):
        return {}, (f"{name}: its dates (store {dd.get('store')}, common {dd.get('common')}) "
                    f"are not the store's ({len(zd)}, {len(common)}): a verification of another "
                    "download, no identity")
    bound = "its date counts"
    rv = verify.get("radiometry")
    rv = rv.get("per_date_valid_low") if isinstance(rv, collections.abc.Mapping) else None
    rs = (radiometry.get("per_date_valid_low")
          if isinstance(radiometry, collections.abc.Mapping) else None)
    if rv is not None and rs is not None:
        try:
            same = _per_date_counts(rv, zd) == _per_date_counts(rs, zd)
        except (TypeError, ValueError):
            same = False
        if not same:
            return {}, (f"{name}: its per-date B12 counts are not the store's: a verification "
                        "of another download, no identity")
        bound = "its date counts and per-date B12 counts"
    base = f"B03 and B08 bitwise, {name} verdict {verdict!r}"
    if verdict in IDENTITY_VERDICTS:
        return dict.fromkeys(common, base), (f"{name}: verdict {verdict!r}, {len(common)} "
                                             f"common dates (bound to the store by {bound})")
    if _bands_clean(bands):
        note = f"{base}, no B03 or B08 difference on any date"
        return dict.fromkeys(common, note), (f"{name}: verdict 'differs' outside B03 and B08, "
                                             f"{len(common)} common dates (bound to the store "
                                             f"by {bound})")
    n = _count(verify.get("dates_with_differences"))
    complete = n is not None and n == len(listed)
    out = {}
    for d in common:
        e = listed.get(d)
        if e is not None:
            # a listed date's entry counts the differing pixels of each band that differs
            if isinstance(e, collections.abc.Mapping) and all(
                    b not in e or _count(e[b]) == 0 for b in IDENTITY_BANDS):
                out[d] = f"{base}, the date differs in SCL only"
        elif complete:
            out[d] = f"{base}, the date is not among the {n} dates with differences"
    tail = ("the list of dates with differences is complete" if complete else
            f"the list of dates with differences is truncated ({len(listed)} of {n}): the "
            "unlisted dates get no identity")
    return out, (f"{name}: verdict 'differs', {len(out)} of {len(common)} common dates identical "
                 f"in B03 and B08; {tail} (bound to the store by {bound})")


# ─────────────────────────────────────────────────────────────────────────────
#  fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def _sha_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def store_fingerprint(path, full=False):
    """Fingerprint of one store: sha256 of the consolidated metadata and of
    the sorted ``(chunk key, size)`` list of every file below the band
    folders; ``full=True`` adds ``content_sha256`` over (key, file sha256).

    Returns
    -------
    dict
        ``fingerprint``, ``meta_sha256``, ``chunks_sha256``, ``n_files``,
        ``bytes`` (and ``content_sha256``).
    """
    path = Path(path)
    meta = (path / ".zmetadata").read_bytes()
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.parent != path:
            files.append((p.relative_to(path).as_posix(), p.stat().st_size))
    files.sort()
    ch = hashlib.sha256("".join(f"{k}\t{s}\n" for k, s in files).encode("utf-8")).hexdigest()
    ms = hashlib.sha256(meta).hexdigest()
    out = {"fingerprint": hashlib.sha256(f"{ms}\n{ch}".encode()).hexdigest(),
           "meta_sha256": ms, "chunks_sha256": ch, "n_files": len(files),
           "bytes": int(sum(s for _, s in files))}
    if full:
        h = hashlib.sha256()
        for k, _ in files:
            h.update(f"{k}\t{_sha_file(path / k)}\n".encode("utf-8"))
        out["content_sha256"] = h.hexdigest()
    return out


def netcdf_fingerprint(path, full=False):
    """Fingerprint of a legacy netCDF cube: sha256 of its header (dims,
    variables, dtypes, attributes, chunking) and size; ``full=True`` adds
    the sha256 of the file."""
    import xarray as xr

    path = Path(path)
    with xr.open_dataset(path, mask_and_scale=False, decode_times=False) as ds:
        head = {"dims": {k: int(v) for k, v in ds.sizes.items()},
                "vars": {k: {"dtype": str(v.dtype), "dims": list(v.dims),
                             "attrs": {a: str(x) for a, x in v.attrs.items()},
                             "chunks": str(v.encoding.get("chunksizes"))}
                         for k, v in ds.variables.items()},
                "attrs": {a: str(x) for a, x in ds.attrs.items()}}
    hs = hashlib.sha256(json.dumps(head, sort_keys=True).encode("utf-8")).hexdigest()
    size = path.stat().st_size
    out = {"fingerprint": hashlib.sha256(f"{hs}\n{size}".encode()).hexdigest(),
           "header_sha256": hs, "bytes": int(size)}
    if full:
        out["content_sha256"] = _sha_file(path)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the scene source
# ─────────────────────────────────────────────────────────────────────────────

Block = collections.namedtuple("Block", "positions rows data lattice")


class SceneSource:
    """The scenes of one site (the protocol of spec §1.2).

    Members (subclasses fill ``_dates_all``, ``_parts`` and the readers):

    * ``dates``, ``instants``: the record (dates with a valid instant);
    * ``grid``: :class:`Grid`; ``lattice20``: (py, px) or None;
    * ``offset(q)``, ``offsets``: the per-scene BOA offset (0 or 1000);
    * ``position(q)``: record position -> (part, local index);
    * ``read_scene(q, bands, rows=None)``, ``read_block(qs, bands, rows)``;
    * ``row_blocks(bands, block_rows=256, positions=None, pair20=False)``;
    * ``fingerprint(part=None, full=False)``;
    * ``window(start, end)``: record positions of a window.
    """

    kind = "abstract"
    bands = ()

    def __init__(self, site, instants, keep_dates=None, cache_mb=CACHE_MB, lattice="measure",
                 offsets=None):
        self.site = site
        self.cache = BlockCache(cache_mb)
        inst = dict(instants)
        _check_instants(inst)
        self._instants_all = inst
        sel = []
        keep = None if keep_dates is None else {str(d)[:10] for d in keep_dates}
        for i, d in enumerate(self._dates_all):
            t = inst.get(d, pd.NaT)
            if t is pd.NaT or pd.isna(t):
                continue
            if keep is not None and d not in keep:
                continue
            sel.append(i)
        self._sel = np.asarray(sel, np.int64)
        self.dates = [self._dates_all[i] for i in self._sel]
        self.instants = pd.DatetimeIndex([inst[d] for d in self.dates])
        if len(self.instants) > 1 and not (np.diff(self.instants.values) > np.timedelta64(0)).all():
            raise InputsError(f"{site}: record instants are not strictly increasing")
        kept = set(self.dates)
        self.dropped = [d for d in self._dates_all if d not in kept]
        self._lattice = lattice
        self._lattice_report = None
        self._frozen_offsets = None if offsets is None else {str(k)[:10]: v
                                                             for k, v in dict(offsets).items()}
        self._offset_table = None

    # -- record --------------------------------------------------------------
    @property
    def T(self):
        return len(self.dates)

    @property
    def instants_sha256(self):
        """:func:`instants_sha` of the instants of the source's store dates
        (NaT where a date has none): the record's instants, recorded."""
        return instants_sha({d: self._instants_all.get(d, pd.NaT) for d in self._dates_all})

    def __len__(self):
        return self.T

    def window(self, start, end):
        """Record positions of the dates in ``[start, end)``."""
        t0, t1 = zc.parse_date(start), zc.parse_date(end)
        d = pd.DatetimeIndex(pd.to_datetime(self.dates))
        return np.flatnonzero((d >= t0) & (d < t1)).astype(np.int64)

    def position(self, q):
        """Record position ``q`` -> ``(part, local index)`` (part: the year
        of a Zarr store, None for a netCDF cube)."""
        return self._position_of_all(int(self._sel[int(q)]))

    def _check_bands(self, bands):
        bands = [bands] if isinstance(bands, str) else list(bands)
        bad = [b for b in bands if b not in self.bands]
        if bad:
            raise KeyError(f"{self.kind} source of {self.site}: bands {bad} not available "
                           f"(has {self.bands})")
        return bands

    # -- reading -------------------------------------------------------------
    def _chunk(self, part, band, tb, rc):
        """Raw (nt, rows of row chunk rc, W) of one chunk block (cached)."""
        return self.cache.get((part, band, int(tb), int(rc)),
                              lambda: self._load(part, band, int(tb), int(rc)))

    def _rows_of(self, rows):
        if rows is None:
            return 0, self.grid.H
        if isinstance(rows, slice):
            r0, r1, st = rows.indices(self.grid.H)
            if st != 1:
                raise ValueError("row slices must be contiguous")
            return r0, r1
        r0, r1 = (int(v) for v in rows)
        return r0, min(r1, self.grid.H)

    def _assemble(self, part, band, local, r0, r1):
        """Raw values of local indices ``local`` (sorted, one time block or
        more) and rows [r0, r1)."""
        local = np.asarray(local, np.int64)
        out = None
        for tb in np.unique(local // T_BLOCK):
            m = local // T_BLOCK == tb
            li = local[m] - tb * T_BLOCK
            pieces = []
            for rc in range(r0 // ROW_BLOCK, (r1 - 1) // ROW_BLOCK + 1):
                c = self._chunk(part, band, tb, rc)
                a0 = max(r0, rc * ROW_BLOCK) - rc * ROW_BLOCK
                a1 = min(r1, (rc + 1) * ROW_BLOCK) - rc * ROW_BLOCK
                pieces.append(c[li, a0:a1, :])
            blk = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
            if out is None:
                out = np.empty((len(local),) + blk.shape[1:], blk.dtype)
            out[np.flatnonzero(m)] = blk
        return out

    def _finish(self, band, a):
        if band == "SCL":
            return obs.scl_u8(a, self.fill.get(band, obs.FILL))
        if band == "CLD":
            return obs.cld_u8(a, self.fill.get(band, obs.FILL))
        return np.ascontiguousarray(a)

    def read_block(self, qs, bands, rows=None):
        """``{band: (len(qs), rows, W)}`` raw values of record positions
        ``qs`` (int16; SCL and CLD as checked uint8)."""
        bands = self._check_bands(bands)
        qs = np.atleast_1d(np.asarray(qs, np.int64))
        r0, r1 = self._rows_of(rows)
        allpos = [self.position(q) for q in qs]
        out = {}
        for b in bands:
            res = None
            for part in dict.fromkeys(p for p, _ in allpos):
                idx = [k for k, (p, _) in enumerate(allpos) if p == part]
                local = np.asarray([allpos[k][1] for k in idx], np.int64)
                order = np.argsort(local, kind="stable")
                vals = self._assemble(part, b, local[order], r0, r1)
                if res is None:
                    res = np.empty((len(qs),) + vals.shape[1:], vals.dtype)
                res[np.asarray(idx)[order]] = vals
            out[b] = self._finish(b, res)
        return out

    def read_scene(self, q, bands, rows=None):
        """``{band: (rows, W)}`` raw values of record position ``q``."""
        return {b: a[0] for b, a in self.read_block([int(q)], bands, rows).items()}

    def row_blocks(self, bands, block_rows=ROW_BLOCK, positions=None, pair20=False):
        """Chunk-aligned blocks in chunk order: for each time block of 16
        dates (per part), each row block of ``block_rows`` rows.

        Yields
        ------
        Block
            ``positions`` (record positions), ``rows`` (slice), ``data``
            ({band: (n, rows, W)}), ``lattice`` (with ``pair20``: the 20 m
            phase relative to the block's first row; else None). With
            ``pair20`` the row boundaries move to the lattice rows (r = py
            mod 2), so no 20 m block straddles two row blocks and the 20 m
            indices of a block equal the frame's; the cache still decodes
            each chunk once.
        """
        bands = self._check_bands(bands)
        if pair20 and int(block_rows) % 2:
            raise ValueError(f"block_rows {block_rows} is odd: with pair20 a 20 m block would "
                             "straddle two row blocks (use an even number)")
        lat = self.lattice20 if pair20 else None
        H = self.grid.H
        bounds = list(range(0, H, int(block_rows))) + [H]
        if pair20 and lat is not None and lat[0] == 1:
            bounds = [0] + [min(H, b + 1) for b in bounds[1:-1]] + [H]
            bounds = sorted(set(bounds))
        for qs in self.time_blocks(positions):
            for r0, r1 in zip(bounds[:-1], bounds[1:]):
                data = self.read_block(qs, bands, slice(r0, r1))
                local_lat = None if lat is None else ((lat[0] - r0) % 2, lat[1])
                yield Block(np.asarray(qs, np.int64), slice(r0, r1), data, local_lat)

    def time_blocks(self, positions=None):
        """Record positions grouped by stored time block (the 16 dates of one
        chunk of one part), in chunk order: reading one group at once decodes
        each chunk once, whatever the cache size.

        Returns
        -------
        list of int64 arrays
        """
        pos = np.arange(self.T, dtype=np.int64) if positions is None \
            else np.asarray(positions, np.int64)
        if len(pos) > 1 and not (np.diff(pos) > 0).all():
            raise ValueError("positions must be strictly increasing (chunk order)")
        groups = collections.OrderedDict()
        for q in pos:
            part, local = self.position(q)
            groups.setdefault((part, int(local) // T_BLOCK), []).append(int(q))
        return [np.asarray(qs, np.int64) for qs in groups.values()]

    # -- lattice20 -----------------------------------------------------------
    def _sample_positions(self, n=LATTICE_SCENES):
        T = self.T
        if T == 0:
            return []
        return sorted(set(np.linspace(0, T - 1, min(n, T)).round().astype(int).tolist()))

    def measure_lattice20(self, n_scenes=LATTICE_SCENES, max_rows=2 * ROW_BLOCK):
        """Measure the 20 m lattice on up to ``n_scenes`` scenes spread over
        the record (rows of a central, even-aligned band of ``max_rows``)."""
        bands = [b for b in LATTICE_BANDS if b in self.bands]
        if not bands:
            return {"phase": None, "exact": False, "reason": "no 20 m band"}
        H = self.grid.H
        r0 = max(0, (H // 2 - max_rows // 2)) // 2 * 2
        r1 = min(H, r0 + max_rows)
        arrays = []
        for q in self._sample_positions(n_scenes):
            sc = self.read_scene(q, bands, slice(r0, r1))
            arrays += [(b, sc[b]) for b in bands]
        rep = measure_lattice20(arrays, expected_lattice20(self.grid))
        rep.update(rows=[int(r0), int(r1)], n_scenes=len(self._sample_positions(n_scenes)))
        return rep

    @property
    def lattice20(self):
        """The native 20 m lattice phase (py, px), or None (§2.4)."""
        if isinstance(self._lattice, str) and self._lattice == "measure":
            rep = self.measure_lattice20()
            self._lattice_report = rep
            self._lattice = tuple(rep["phase"]) if rep.get("phase") else None
        return self._lattice

    @property
    def lattice_report(self):
        _ = self.lattice20
        return self._lattice_report

    # -- offsets -------------------------------------------------------------
    def offset_table(self):
        """The per-scene offset decisions (:func:`decide_offsets`) of the
        record; frozen ones when given at construction."""
        if self._offset_table is None:
            self._offset_table = self._offsets_decided()
        return self._offset_table

    def _offsets_decided(self):
        if self._frozen_offsets is not None:
            rows = []
            for d in self.dates:
                v = self._frozen_offsets.get(d)
                ok = v is not None and not (isinstance(v, float) and np.isnan(v)) \
                    and int(v) in settings.OFFSETS
                rows.append({"date": d, "offset": float(v) if ok else np.nan,
                             "state": ("unclear" if not ok else "+1000 in the values"
                                       if int(v) == 1000 else "no offset"),
                             "reason": "frozen" if ok else "missing from the frozen offsets",
                             "strong": bool(ok)})
            return pd.DataFrame(rows, columns=["date", "offset", "state", "reason",
                                               "strong"]).set_index("date")
        return self._offsets_provisional()

    def _offsets_provisional(self):
        raise NotImplementedError

    def offset(self, q):
        """The BOA offset (0 or 1000 DN) of record position ``q``;
        :class:`OffsetUnclear` when undecided."""
        d = self.dates[int(q)]
        row = self.offset_table().loc[d]
        if row["state"] == "unclear":
            raise OffsetUnclear(f"{self.site} {d}: BOA offset unclear ({row['reason']})")
        return int(row["offset"])

    @property
    def offsets(self):
        """``(T,)`` int array of the record's offsets (raises if any is unclear)."""
        return np.asarray([self.offset(q) for q in range(self.T)], np.int64)

    # -- misc ----------------------------------------------------------------
    def describe(self):
        return {"site": self.site, "kind": self.kind, "T": self.T,
                "first": self.dates[0] if self.dates else None,
                "last": self.dates[-1] if self.dates else None,
                "dropped_without_instant": len(self.dropped), "bands": list(self.bands),
                "grid": self.grid.describe(), "cache": self.cache.stats()}

    def close(self):
        self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ZarrSource(SceneSource):
    """:class:`SceneSource` over the per-year Zarr stores of a site.

    Parameters
    ----------
    site : str
    start, end : str, optional
        The period (end exclusive); default :data:`UNION_PERIOD`, clipped
        to the years present (``missing="skip"``).
    root : path, optional
        The cubes root (``zarr_cubes.cubes_root``).
    instants : "catalogue" (default), "labels", a path or a mapping
        The catalogue times of ``<root>/<site>/overpass_times.json`` (the
        datatake start of the product names, the default of every run;
        "labels" is an alias); an explicit ``sensing_times.json`` path (the
        optional diagnostic); or ``{date: instant}`` (:func:`resolve_instants`).
    keep_dates : iterable, optional
        Restrict the record (e.g. dates with a finite h_B).
    offsets : mapping, optional
        The frozen ``{date: 0 or 1000}`` (inputs report); without it the
        decision is provisional, from the stores' attributes and the
        scene metadata.
    scene_meta : dict or path, optional
        :func:`read_scene_meta` (default ``cubes/<site>/scene_meta.json`` if
        present).
    lattice : "measure", (py, px) or None
    cache_mb : float
    """

    kind = "zarr"
    bands = ZARR_BANDS

    def __init__(self, site, start=None, end=None, root=None, instants=CATALOGUE,
                 keep_dates=None, offsets=None, scene_meta=None, lattice="measure",
                 cache_mb=CACHE_MB):
        import zarr

        self.root = root
        start = start or UNION_PERIOD[0]
        end = end or UNION_PERIOD[1]
        with warnings.catch_warnings():
            # years that differ in their offset regime are expected: the
            # per-scene offset handles them (kept in radiometry_warning)
            warnings.filterwarnings("ignore", message=".*disagree on the L2A.*")
            ds = zc.open_cube(site, start, end, root=root, missing="skip")
        try:
            self.grid = grid_of(ds)
            self.years = [int(y) for y in ds.attrs.get("years", [])]
            self.missing_years = [int(y) for y in ds.attrs.get("missing_years", [])]
            self.fill = {b: (zc.fill_of(ds[b]) if zc.fill_of(ds[b]) is not None else obs.FILL)
                         for b in zc.band_names(ds)}
            all_dates = zc.dates(ds)
            self.radiometry_warning = ds.attrs.get("radiometry_warning")
        finally:
            ds.close()
        bad = {b: f for b, f in self.fill.items() if f != obs.FILL}
        if bad:
            raise InputsError(f"{site}: unexpected fills {bad}")
        self.bands = tuple(b for b in ZARR_BANDS if b in self.fill)
        self._paths = {y: zc.store_path(site, y, root) for y in self.years}
        self._groups = {y: zarr.open_group(str(p), mode="r", zarr_format=2)
                        for y, p in self._paths.items()}
        self._local = {}
        part_of = []
        for y in self.years:
            with zc.open_store(self._paths[y], bands=["SCL"]) as dy:
                dd = zc.dates(dy)
            self._local[y] = {d: i for i, d in enumerate(dd)}
        for d in all_dates:
            y = int(d[:4])
            if d not in self._local.get(y, {}):
                raise InputsError(f"{site} {d}: not in the {y} store's dates")
            part_of.append((y, self._local[y][d]))
        self._part_of = part_of
        self._dates_all = list(all_dates)
        self._attrs = {y: zc.store_attrs(p) for y, p in self._paths.items()}
        if isinstance(scene_meta, (str, os.PathLike)):
            scene_meta = read_scene_meta(scene_meta)
        elif scene_meta is None:
            p = zc.site_dir(site, root) / SCENE_META_NAME
            scene_meta = read_scene_meta(p) if p.is_file() else None
        self.scene_meta = scene_meta
        instants, self.instant_source, self.instant_path = resolve_instants(
            instants, catalogue_labels(site, root))
        super().__init__(site, instants, keep_dates, cache_mb, lattice, offsets)

    def _position_of_all(self, i):
        return self._part_of[i]

    def _load(self, year, band, tb, rc):
        arr = self._groups[year][band]
        n = arr.shape[0]
        t0, t1 = tb * T_BLOCK, min(n, (tb + 1) * T_BLOCK)
        r0, r1 = rc * ROW_BLOCK, min(self.grid.H, (rc + 1) * ROW_BLOCK)
        return np.asarray(arr[t0:t1, r0:r1, :])

    def fingerprint(self, part=None, full=False):
        """:func:`store_fingerprint` of one year (``part``) or ``{year: ...}``."""
        if part is not None:
            return store_fingerprint(self._paths[int(part)], full)
        return {str(y): store_fingerprint(p, full) for y, p in self._paths.items()}

    def baselines(self):
        """``{date: [processing versions]}``: the union of the scene metadata
        (the STAC items at query time) and the stores' ``input_baselines``
        (the products the job read), so a date whose two lists fall on both
        sides of 04.00 is 'mixed' for the offset rule (amendment of
        2026-10-19; before it the scene metadata alone when present)."""
        out = {}
        for a in self._attrs.values():
            for d, v in ((zc.store_baselines(a) or {}).get("per_date") or {}).items():
                out.setdefault(str(d)[:10], set()).update(str(x) for x in (v or ()))
        if self.scene_meta:
            for d, v in self.scene_meta.get("dates", {}).items():
                out.setdefault(str(d)[:10], set()).update(
                    str(x) for x in v.get("processing_baselines", []))
        return {d: sorted(v) for d, v in out.items()}

    def year_states(self):
        return {str(y): radiometry_stats(a)[1] for y, a in self._attrs.items()}

    def legacy_identity(self):
        """Rule (0) of :data:`OFFSET_RULE` over the source's store years:
        ``({date: note}, {year: note})``, the store dates whose B03 and B08
        equal the offset-harmonised legacy netCDF cube, from each
        ``<year>.verify.json`` beside the stores (:func:`read_verify`,
        :func:`identity_from_verify`, bound to the store by its dates and
        its per-date B12 counts); a malformed file grants nothing."""
        dates, notes = {}, {}
        for y in self.years:
            try:
                ident, note = identity_from_verify(read_verify(self.site, y, self.root),
                                                   self._local[y], y,
                                                   zc.store_radiometry(self._attrs[y]))
            except (AttributeError, TypeError, ValueError, KeyError) as e:
                ident, note = {}, (f"{y}.{VERIFY_NAME}: malformed ({type(e).__name__}: {e}), "
                                   "no identity")
            dates.update(ident)
            notes[str(y)] = note
        return dates, notes

    def _offsets_provisional(self):
        stats = {}
        for y, a in self._attrs.items():
            per, _, _ = radiometry_stats(a)
            H, W = self.grid.shape
            for d, st in per.items():
                stats[d] = dict(st, n_px=H * W)
        for d in self.dates:
            stats.setdefault(d, {})
        ident, _ = self.legacy_identity()
        tab = decide_offsets(stats, self.baselines(), self.year_states(), ident)
        return tab.loc[[d for d in self.dates]]

    def angles(self, lon=None, lat=None):
        """The openEO angle bands at a point (frame centre by default) for
        the record dates (diagnostic only: those angles are quantised;
        sensing times come from the STAC sun angles); None without angles
        stores."""
        if not zc.list_angle_years(self.site, self.root):
            return None
        if lon is None or lat is None:
            lon, lat = self.grid.lonlat_centre()
        start, end = self.dates[0], (pd.Timestamp(self.dates[-1]) + pd.Timedelta(days=1))
        df = zc.angles_at(self.site, lon, lat, start, str(end.date()), root=self.root,
                          missing="skip")
        df.index = df.index.strftime("%Y-%m-%d")
        return df.reindex(self.dates)


class NetCDFSource(SceneSource):
    """:class:`SceneSource` over a legacy netCDF cube (B03, B08, SCL): the
    reproduction gates and the local runs on the 20 m cubes (``--res 20``).
    Values are offset-harmonised (o_s = 0 for every scene,
    ``obs.check_boa``); the lattice is measured on SCL alone (pass
    ``lattice=None`` on a 20 m grid, which has no 20 m sub-lattice).

    Parameters
    ----------
    path : path
    site : str
    instants : "catalogue" (default), "labels", a path or a mapping
        The catalogue times of ``labels_path`` (the production
        overpass_times.json: the datatake start of the product names); an
        explicit ``sensing_times.json`` path; or ``{date: instant}``.
    start, end : str, optional
        Restrict the dates (end exclusive).
    labels_path : path, optional
        The catalogue times (required for "catalogue").
    """

    kind = "netcdf"
    bands = LEGACY_BANDS

    def __init__(self, path, site, instants=CATALOGUE, start=None, end=None, keep_dates=None,
                 labels_path=None, lattice="measure", cache_mb=CACHE_MB):
        import xarray as xr

        self.path = Path(path)
        self._ds = xr.open_dataset(self.path, mask_and_scale=False, decode_times=True)
        ds = self._ds
        self._tdim = zc._time_dim(ds)
        for b in LEGACY_BANDS:
            if b not in ds.data_vars:
                raise InputsError(f"{path}: band {b} missing")
            if ds[b].dims != (self._tdim, "y", "x"):
                raise InputsError(f"{path}: {b} dims {ds[b].dims} are not (t, y, x)")
            if str(ds[b].dtype) != "int16":
                raise InputsError(f"{path}: {b} is {ds[b].dtype}, not int16")
            if any(k in ds[b].attrs or k in ds[b].encoding for k in ("scale_factor", "add_offset")):
                raise InputsError(f"{path}: {b} carries scale/offset attributes")
        self.fill = {b: int(zc.fill_of(ds[b])) for b in LEGACY_BANDS}
        self.grid = grid_of(ds)
        dates = [str(d)[:10] for d in pd.DatetimeIndex(ds[self._tdim].values).strftime("%Y-%m-%d")]
        idx = np.arange(len(dates))
        if start is not None or end is not None:
            d = pd.DatetimeIndex(pd.to_datetime(dates))
            m = np.ones(len(d), bool)
            if start is not None:
                m &= d >= zc.parse_date(start)
            if end is not None:
                m &= d < zc.parse_date(end)
            idx = idx[m]
        self._dates_all = [dates[i] for i in idx]
        self._cube_index = idx

        read_labels = file_labels(labels_path)

        def labels():
            try:
                return read_labels()
            except ValueError:
                self._ds.close()
                raise

        instants, self.instant_source, self.instant_path = resolve_instants(instants, labels)
        super().__init__(site, instants, keep_dates, cache_mb, lattice, offsets=None)
        self.years = sorted({int(d[:4]) for d in self._dates_all})
        self.missing_years = []

    def _position_of_all(self, i):
        return None, int(self._cube_index[i])

    def _load(self, part, band, tb, rc):
        n = self._ds.sizes[self._tdim]
        t0, t1 = tb * T_BLOCK, min(n, (tb + 1) * T_BLOCK)
        r0, r1 = rc * ROW_BLOCK, min(self.grid.H, (rc + 1) * ROW_BLOCK)
        return np.asarray(self._ds[band].isel({self._tdim: slice(t0, t1),
                                               "y": slice(r0, r1)}).values)

    def _offsets_provisional(self):
        return pd.DataFrame({"date": self.dates, "offset": 0.0, "state": "no offset",
                             "reason": "legacy cube: offset-harmonised values (DN - 1000)",
                             "strong": True}).set_index("date")

    def fingerprint(self, part=None, full=False):
        return netcdf_fingerprint(self.path, full)

    def close(self):
        super().close()
        self._ds.close()


# ─────────────────────────────────────────────────────────────────────────────
#  the scene metadata sidecar (§1.10)
# ─────────────────────────────────────────────────────────────────────────────

def scene_meta_path(site, root=None):
    return zc.site_dir(site, root) / SCENE_META_NAME


def _item_record(f):
    p = f.get("properties", {})
    bbox = f.get("bbox")
    cen = None
    if bbox and len(bbox) >= 4:
        cen = [round((bbox[0] + bbox[2]) / 2.0, 6), round((bbox[1] + bbox[3]) / 2.0, 6)]
    tile = str(p.get("grid:code") or p.get("s2:mgrs_tile") or "")
    return {"id": f.get("id"), "tile": tile.replace("MGRS-", ""),
            "datetime": p.get("datetime") or p.get("start_datetime"),
            "platform": p.get("platform"),
            "processing_baseline": p.get("processing:version") or p.get("s2:processing_baseline"),
            "timeliness": p.get("product:timeliness_category") or p.get("product:timeliness"),
            "cloud_cover": p.get("eo:cloud_cover"), "snow_cover": p.get("eo:snow_cover"),
            "relative_orbit": p.get("sat:relative_orbit"),
            "absolute_orbit": p.get("sat:absolute_orbit"),
            "orbit_state": p.get("sat:orbit_state"),
            "sun_azimuth": p.get("view:sun_azimuth"), "sun_elevation": p.get("view:sun_elevation"),
            "view_azimuth": p.get("view:azimuth"), "view_incidence": p.get("view:incidence_angle"),
            "datatake_id": p.get("eopf:datatake_id"), "bbox": bbox, "centre_lonlat": cen}


def scene_meta_from_items(features, site=None, query=None):
    """The scene metadata sidecar from CDSE STAC features (``sentinel-2-l2a``).

    Per date (the UTC date of the item's datetime): every item record
    (:func:`_item_record`), and a summary: platforms, relative orbits,
    processing baselines, tiles, the mean tile cloud cover, and
    ``double_overpass`` (more than one relative orbit on the date).
    """
    dates = {}
    for f in features:
        r = _item_record(f)
        if not r["datetime"]:
            continue
        d = str(_naive_utc(r["datetime"]).date())
        dates.setdefault(d, {"items": []})["items"].append(r)
    for d, v in dates.items():
        it = sorted(v["items"], key=lambda r: (str(r["tile"]), str(r["id"])))
        v["items"] = it
        v["platforms"] = sorted({str(r["platform"]) for r in it if r["platform"]})
        v["relative_orbits"] = sorted({int(r["relative_orbit"]) for r in it
                                       if r["relative_orbit"] is not None})
        v["processing_baselines"] = sorted({str(r["processing_baseline"]) for r in it
                                            if r["processing_baseline"]})
        v["tiles"] = sorted({str(r["tile"]) for r in it if r["tile"]})
        cc = [float(r["cloud_cover"]) for r in it if r["cloud_cover"] is not None]
        v["cloud_cover_mean"] = round(float(np.mean(cc)), 4) if cc else None
        v["double_overpass"] = len(v["relative_orbits"]) > 1
        ids = [r["id"] for r in it]
        v["duplicate_tiles"] = sorted({t for t in v["tiles"]
                                       if sum(1 for r in it if r["tile"] == t) > 1})
        v["n_items"] = len(ids)
    return {"format": SCENE_META_FORMAT, "site": site, "query": query or {},
            "dates": dict(sorted(dates.items()))}


def write_scene_meta(path, meta):
    """Write the sidecar (sorted keys, LF, atomically)."""
    _write_json(path, meta)


def read_scene_meta(path):
    """Read and check a scene metadata sidecar."""
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format") != SCENE_META_FORMAT:
        raise InputsError(f"{path}: not a scene metadata sidecar ({meta.get('format')!r})")
    return meta


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(obj, sort_keys=True, indent=1, default=_jsonable, allow_nan=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data + "\n")
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
#  the inputs report (§1.11)
# ─────────────────────────────────────────────────────────────────────────────

def check_store(path):
    """§1.11 item 1 for one store: complete, FORMAT_ID, dims, dtype and
    ``_FillValue`` per band, no active scale or offset, axes and chunks.

    Returns
    -------
    dict
        ``ok``, ``problems`` (list of str), ``bands`` (dtype, fill, chunks),
        ``offset_state``, ``delivered_attrs`` (the ``delivered_*`` names).
    """
    path = Path(path)
    probs = []
    attrs = zc.store_attrs(path)
    if attrs.get("complete") != 1:
        probs.append("complete != 1")
    if attrs.get("format") != zc.FORMAT_ID:
        probs.append(f"format {attrs.get('format')!r} != {zc.FORMAT_ID!r}")
    meta = (zc._consolidated(path) or {}).get("metadata", {})
    bands = {}
    delivered = {}
    ds = zc.open_store(path)
    try:
        for b in ZARR_BANDS:
            if b not in ds.data_vars:
                probs.append(f"{b} missing")
                continue
            da_ = ds[b]
            za = meta.get(f"{b}/.zattrs", {})
            zr = meta.get(f"{b}/.zarray", {})
            if da_.dims != ("t", "y", "x"):
                probs.append(f"{b} dims {da_.dims}")
            if str(da_.dtype) != "int16":
                probs.append(f"{b} dtype {da_.dtype}")
            f = zc.fill_of(da_)
            if f != obs.FILL:
                probs.append(f"{b} _FillValue {f}")
            for k in ("scale_factor", "add_offset"):
                if k in za or k in da_.attrs or k in da_.encoding:
                    probs.append(f"{b} carries an active {k}")
            delivered[b] = sorted(k for k in za if k.startswith("delivered_"))
            want = list(zc.chunk_shape(da_.shape))
            if zr.get("chunks") and list(zr["chunks"]) != want:
                probs.append(f"{b} chunks {zr['chunks']} != {want}")
            bands[b] = {"dtype": str(da_.dtype), "fill": f, "chunks": zr.get("chunks")}
        x, y = ds["x"].values, ds["y"].values
        if len(x) > 1 and not np.all(np.diff(x) > 0):
            probs.append("x not ascending")
        if len(y) > 1 and not np.all(np.diff(y) < 0):
            probs.append("y not descending")
        if len(x) > 1 and not np.allclose(np.diff(x), 10.0):
            probs.append("x spacing is not 10 m")
        if len(y) > 1 and not np.allclose(np.diff(y), -10.0):
            probs.append("y spacing is not 10 m")
    finally:
        ds.close()
    _, state, band = radiometry_stats(attrs)
    if state is None:
        probs.append("no radiometry attribute")
    return {"ok": not probs, "problems": probs, "bands": bands, "offset_state": state,
            "radiometry_band": band, "delivered_attrs": delivered}


def scan_source(src, positions=None, log_every=50):
    """§1.11 item 2 by streaming B08, B12, SCL and CLD of every scene (row
    blocks of 16 dates): SCL codes (``obs.scl_u8`` raises outside 0..11),
    the SCL histogram, CLD range (``obs.cld_u8``), the offset statistics
    (:func:`scene_offset_stats`; the clear-water counts use the CLD of the
    same blocks, so every chunk is still read once; :func:`scl_cloud_counts`
    from the histogram).

    Returns
    -------
    dict
        ``{date: {"scl": [13 counts], "cld_fill", "cld_max", **offset stats,
        "n_scl_cloud", "n_scl_valid"}}``.
    """
    bands = [b for b in ("B08", "B12", "SCL", "CLD") if b in src.bands]
    out = {}
    t0 = time.time()
    done = 0
    for blk in src.row_blocks(bands, positions=positions):
        if blk.rows.stop == src.grid.H:
            done += len(blk.positions)
            if log_every and done % log_every < len(blk.positions):
                log(f"  [inputs] {src.site}: {done}/{src.T if positions is None else len(positions)}"
                    f" scenes scanned ({time.time() - t0:.0f} s)")
        s8 = blk.data["SCL"]
        st = (scene_offset_stats(blk.data["B12"], s8, [blk.data["B08"]],
                                 src.fill.get("B12", obs.FILL), cld_u8=blk.data.get("CLD"))
              if "B12" in blk.data else None)
        for k, q in enumerate(blk.positions):
            d = src.dates[int(q)]
            e = out.setdefault(d, {"scl": [0] * 13, "cld_fill": 0, "cld_max": None})
            cnt = obs.scl_counts(s8[k])
            e["scl"] = [a + int(b) for a, b in zip(e["scl"], cnt)]
            if "CLD" in blk.data:
                c = blk.data["CLD"][k]
                e["cld_fill"] += int(np.count_nonzero(c == obs.CLD_NODATA_U8))
                pres = c[c != obs.CLD_NODATA_U8]
                if pres.size:
                    m = int(pres.max())
                    e["cld_max"] = m if e["cld_max"] is None else max(e["cld_max"], m)
            if st is not None:
                for key, arr in st.items():            # summed over row blocks
                    e[key] = e.get(key, 0) + int(arr[k])
    for e in out.values():
        e.update(scl_cloud_counts(e["scl"]))
    return out


def _iter_files(top):
    """Files under ``top``, depth first in sorted order, lazily (a chunk
    tree is never listed in full)."""
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def disk_check(paths, read_from=None, read_mb=64):
    """§1.11 item 8: free space per path and the sequential read
    throughput of up to ``read_mb`` MB of the files under ``read_from``
    (page cache not dropped: an upper bound, declared). The files are
    walked lazily and the walk stops after ``read_mb``: pass one store
    (or one band of it), never the whole cubes root."""
    out = {"free_gb": {}}
    for p in paths:
        try:
            out["free_gb"][str(p)] = round(shutil.disk_usage(p).free / 1e9, 1)
        except OSError as exc:
            out["free_gb"][str(p)] = f"error: {exc}"
    if read_from is not None:
        n, k, t0 = 0, 0, time.time()
        for f in _iter_files(read_from):
            with open(f, "rb") as fh:
                n += len(fh.read())
            k += 1
            if n >= read_mb * 1e6:
                break
        dt = max(time.time() - t0, 1e-6)
        out["read"] = {"from": str(read_from), "mb": round(n / 1e6, 1), "files": k,
                       "mb_per_s": round(n / 1e6 / dt, 1), "note": "page cache not dropped"}
    return out


def disk_sample_dir(site, root=None):
    """The directory :func:`disk_check` reads at a site: the B08 array of its
    latest store (a few thousand chunk files), or None without a store."""
    years = zc.list_years(site, root)
    if not years:
        return None
    p = zc.store_path(site, max(years), root) / "B08"
    return str(p) if p.is_dir() else str(zc.store_path(site, max(years), root))


def _instants_doc(src):
    """Where a source's instants come from (for the reports)."""
    out = {"source": src.instant_source, "path": getattr(src, "instant_path", None)}
    p = out["path"]
    if p and os.path.isfile(p):
        out["sha256"] = _sha_file(p)
    if src.instant_source == CATALOGUE:
        out["note"] = ("catalogue / product-name time (datatake start), the default of every "
                       "run (user decision 2026-10-14)")
    return out


def inputs_report(site, root=None, start=None, end=None, legacy=None, rtk_grid=None,
                  instants=CATALOGUE, scan=True, full_fingerprint=False, disk_paths=None,
                  out=None, cache_mb=SCAN_CACHE_MB):
    """The §1.11 checks of one site; returns the report (and writes it to
    ``out`` merged into the existing file's ``sites`` mapping, if given).

    Parameters
    ----------
    site : str
    root : path, optional
        The cubes root.
    start, end : str, optional
        The period (default :data:`UNION_PERIOD`).
    legacy : path, optional
        The legacy netCDF cube (grid equality: the whole grid, or the
        window of a store that is an extract of it, ``extract.py``). G-zarr
        and rule (0) of the offset rule read the stores' ``<year>.verify.json``
        whether or not it is given (``g_zarr``: the verdict, the dates with
        differences, the identity note and ``scenes_identical``).
    rtk_grid : path, optional
        Villaviciosa's ``rtk.DEFAULT_GRID`` GeoTIFF (grid equality).
    instants : see :class:`ZarrSource`; default the catalogue times (no
        warning); a ``sensing_times.json`` only when its path is given.
    scan : bool
        Stream the stores for codes, ranges and the offset statistics.
    cache_mb : float
        The block cache of the scan (each chunk is read once: a small cache
        keeps the memory low).
    """
    t0 = time.time()
    rep = {"format": INPUTS_REPORT_FORMAT, "site": site, "problems": [], "warnings": [],
           "offset_rule": OFFSET_RULE, "source": {"kind": "zarr", "root": str(
               zc.cubes_root(root)).replace("\\", "/")}}
    years = zc.list_years(site, root)
    rep["years"] = years
    rep["stores"] = {str(y): check_store(zc.store_path(site, y, root)) for y in years}
    for y, c in rep["stores"].items():
        rep["problems"] += [f"{y}: {p}" for p in c["problems"]]
    extracts = {}
    for y in years:
        e = zc.store_attrs(zc.store_path(site, y, root)).get("extract")
        if e:
            extracts[str(y)] = json.loads(e) if isinstance(e, str) else e
    if extracts:
        rep["extract"] = extracts
        rep["warnings"].append("these stores are a local EXTRACT (real B03, B08, SCL; synthetic "
                               "B02, B04, B8A, B11, B12, CLD): software checks only, never "
                               "results")
    src = ZarrSource(site, start, end, root=root, instants=instants, cache_mb=cache_mb)
    try:
        rep["instants"] = _instants_doc(src)
        rep["record"] = {"T": src.T, "dropped_without_instant": src.dropped,
                         "instant_source": src.instant_source,
                         "missing_years": src.missing_years}
        rep["grid"] = src.grid.describe()
        if legacy is not None and Path(legacy).is_file():
            import xarray as xr

            with xr.open_dataset(legacy, mask_and_scale=False) as lg:
                lgrid = grid_of(lg)
            wins = {tuple(e["window"]) for e in extracts.values() if e.get("window")}
            if len(wins) == 1:
                r0, r1, c0, c1 = next(iter(wins))
                lgrid = Grid(lgrid.xs[c0:c1], lgrid.ys[r0:r1], lgrid.crs_wkt)
                rep["grid"]["legacy_window"] = [r0, r1, c0, c1]
            same = lgrid.same_as(src.grid)
            rep["grid"]["legacy_equal"] = bool(same)
            if not same:
                rep["problems"].append("the Zarr grid differs from the legacy netCDF grid")
        if rtk_grid is not None and Path(rtk_grid).is_file():
            same = grid_from_geotiff(rtk_grid).same_as(src.grid)
            rep["grid"]["rtk_equal"] = bool(same)
            if not same:
                rep["problems"].append("the Zarr grid differs from rtk.DEFAULT_GRID")
        rep["lattice20"] = src.lattice_report
        if not rep["lattice20"] or not rep["lattice20"].get("exact"):
            rep["warnings"].append("lattice20 is None: 20 m indices use box means at (0, 0)")
        rep["fingerprints"] = src.fingerprint(full=full_fingerprint)
        stats = scan_source(src) if scan else {}
        rep["scenes"] = stats
        if stats:
            scl11 = {}
            for d, e in stats.items():
                scl11[d[:4]] = scl11.get(d[:4], 0) + e["scl"][11]
            rep["scl11_per_year"] = scl11
            bad_cld = [d for d, e in stats.items() if e["cld_max"] is not None and e["cld_max"] > 100]
            if bad_cld:
                rep["problems"].append(f"CLD above 100 on {bad_cld[:5]}")
        off_stats = {}
        for y, a in src._attrs.items():
            per, _, _ = radiometry_stats(a)
            for d, st in per.items():
                off_stats[d] = dict(st, n_px=src.grid.H * src.grid.W)
        for d, e in stats.items():
            off_stats[d] = {k: e[k] for k in OFFSET_STAT_KEYS if k in e}
            if "n_scl_cloud" not in off_stats[d] and e.get("scl") is not None:
                off_stats[d].update(scl_cloud_counts(e["scl"]))
        ident, ident_notes = src.legacy_identity()
        rec_stats = {d: off_stats.get(d, {}) for d in src.dates}
        bls, ystates = src.baselines(), src.year_states()
        tab = decide_offsets(rec_stats, bls, ystates, ident)
        rep["offset_rule_sha256"] = offset_rule_sha256()
        rep["offsets"] = {d: (None if np.isnan(r.offset) else int(r.offset))
                          for d, r in tab.iterrows()}
        dec = {d: {"state": r.state, "reason": r.reason, "strong": bool(r.strong),
                   "rule": r.rule} for d, r in tab.iterrows()}
        by_id = [d for d, r in tab.iterrows() if r.rule == "identity"]
        if by_id:
            # what rules (1)-(5) alone decide on the scenes the identity decided (they have
            # no offset): the false-positive rate of the pixel rules, for the attribution
            alt = decide_offsets(rec_stats, bls, ystates)
            for d in by_id:
                dec[d]["without_identity"] = {"state": alt.loc[d, "state"],
                                              "reason": alt.loc[d, "reason"],
                                              "rule": alt.loc[d, "rule"]}
            rep["identity_overrides"] = dict(sorted(collections.Counter(
                f"{d[:4]} {alt.loc[d, 'state']}" for d in by_id).items()))
        rep["offset_decisions"] = dec
        unclear = [d for d, r in tab.iterrows() if r.state == "unclear"]
        if unclear:
            rep["problems"].append(f"BOA offset unclear on {len(unclear)} scenes: {unclear[:5]}")
        rep["offset_counts"] = tab["state"].value_counts().to_dict()
        rep["offset_rule_counts"] = dict(sorted(collections.Counter(
            str(x) for x in tab["rule"]).items()))
        notes = tab.attrs
        if notes.get("demoted"):
            rep["warnings"].append(
                f"{len(notes['demoted'])} isolated strong '+1000' demoted to weak (<= "
                f"{OFFSET_RULE['regime_outlier_share']:.0%} of their year's strong decisions, "
                f"beside >= {OFFSET_RULE['regime_min_strong']} strong 0): {notes['demoted']}")
        if notes.get("contradicted"):
            rep["warnings"].append(
                f"{len(notes['contradicted'])} strong decisions contradict the larger side of "
                f"their year (unclear): {notes['contradicted']}")
        if notes.get("identity_pixels_1000"):
            rep["warnings"].append(
                f"{len(notes['identity_pixels_1000'])} scenes decided 0 by the identity with the "
                f"legacy cube whose own clear-water test says +1000: "
                f"{notes['identity_pixels_1000']}")
        for y, n in sorted((notes.get("identity_years") or {}).items()):
            if n.startswith("identity NOT used"):
                rep["warnings"].append(f"{y}: {n}")
        mixed = [d for d in src.dates if _baseline_class(bls.get(d)) == "mixed"]
        if mixed:
            rep["warnings"].append(
                f"{len(mixed)} scenes have processing baselines on both sides of 04.00 (the scene "
                f"metadata and the stores' input_baselines together), e.g. {mixed[:5]}: no "
                "rule-(2) shortcut; one o_s per scene holds only under the backend's "
                "harmonisation")
        by_identity = collections.Counter(d[:4] for d in by_id)
        gz = {}
        for y in years:
            v = read_verify(site, y, root)
            if v is None:
                gz[str(y)] = {"verdict": f"no {VERIFY_NAME}"}
            else:
                gz[str(y)] = {"verdict": v.get("verdict"), "dates": v.get("dates_with_differences")}
                if v.get("error"):
                    gz[str(y)]["error"] = v["error"]
                    rep["warnings"].append(f"{y}.{VERIFY_NAME} is unreadable ({v['error']}): no "
                                           "G-zarr verdict and no identity with the legacy cube")
            # rule (0): the record scenes decided by their identity with the legacy cube
            gz[str(y)]["identity"] = ident_notes.get(str(y), "year outside the source's period")
            gz[str(y)]["identity_check"] = (notes.get("identity_years") or {}).get(str(y))
            gz[str(y)]["scenes_identical"] = int(by_identity.get(str(y), 0))
        rep["g_zarr"] = gz
        meta = src.scene_meta
        rep["scene_meta"] = {"present": meta is not None}
        if meta:
            dm = meta.get("dates", {})
            rep["scene_meta"].update(
                missing_dates=[d for d in src.dates if d not in dm],
                double_overpass_dates=sorted(d for d, v in dm.items() if v.get("double_overpass")),
                baselines_per_year=_per_year(dm, "processing_baselines"),
                platforms_per_year=_per_year(dm, "platforms"),
                relative_orbits_per_year=_per_year(dm, "relative_orbits"))
        else:
            rep["warnings"].append("no scene_meta.json: baselines from the stores' attributes")
        if disk_paths:
            rep["disk"] = disk_check(disk_paths, read_from=zc.site_dir(site, root))
        rep["cache"] = src.cache.stats()
    finally:
        src.close()
    rep["ok"] = not rep["problems"]
    rep["seconds"] = round(time.time() - t0, 1)
    if out is not None:
        _merge_report(out, site, rep)
    return rep


class ReportLock:
    """An exclusive lock file ``<path>.lock`` (O_CREAT | O_EXCL) around the
    read-merge-write of the shared inputs report, so two ``inputs``
    processes never drop each other's site. A lock left by a dead process
    of this host is taken over; another holder is waited for up to
    ``timeout`` seconds."""

    def __init__(self, path, timeout=600.0):
        self.path = f"{path}.lock"
        self.timeout = float(timeout)

    def _stale(self):
        import socket
        try:
            with open(self.path, encoding="utf-8") as f:
                info = json.load(f)
            if info.get("host") != socket.gethostname():
                return False
            import psutil
            return not psutil.pid_exists(int(info["pid"]))
        except Exception:                        # unreadable or no psutil: not stale
            return False

    def __enter__(self):
        import socket
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    os.remove(self.path)
                    continue
                if time.time() - t0 > self.timeout:
                    raise InputsError(f"{self.path} is held by another process (delete it if "
                                      "no inputs command is running)")
                time.sleep(0.5)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"host": socket.gethostname(), "pid": os.getpid()}, f)
            return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def _merge_report(out, site, rep):
    """Write one site's report into the shared report file, under
    :class:`ReportLock` (read, merge, write)."""
    with ReportLock(out):
        doc = {"format": INPUTS_REPORT_FORMAT, "sites": {}}
        if Path(out).is_file():
            with open(out, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("format") == INPUTS_REPORT_FORMAT:
                doc = old
        doc["sites"][site] = rep
        _write_json(out, doc)


def legacy_inputs_report(site, cube, labels_path, res_m, instants=CATALOGUE, scan=True,
                         full_fingerprint=False, out=None, cache_mb=SCAN_CACHE_MB):
    """The data checks of a LEGACY netCDF cube (B03, B08, SCL), the source of
    the local runs on the 20 m cubes (``--res 20``; never reported numbers).

    The reader contract (int16, fill -32768, no scale or offset attributes:
    :class:`NetCDFSource` refuses otherwise), the record (catalogue times by
    default), the grid and its pixel size, the netCDF fingerprint, and with
    ``scan`` the SCL histogram of every scene (codes outside 0..11 refused).
    The legacy values are offset-harmonised: every scene's offset is 0.
    The 20 m sub-lattice does not exist on a 20 m grid (``lattice20`` None).

    Returns the report (written to ``out`` merged by site, if given).
    """
    t0 = time.time()
    rep = {"format": INPUTS_REPORT_FORMAT, "site": site, "problems": [], "warnings": [],
           "source": {"kind": "netcdf", "path": str(cube).replace("\\", "/"),
                      "res_m": int(res_m), "bands": list(LEGACY_BANDS)},
           "offset_rule": "legacy cube: offset-harmonised values (DN - 1000 delivered as "
                          "BOA x 10000), o_s = 0 for every scene"}
    lat = None if int(res_m) >= 20 else "measure"
    try:
        src = NetCDFSource(cube, site, instants, labels_path=labels_path, lattice=lat,
                           cache_mb=cache_mb)
    except (InputsError, ValueError, FileNotFoundError) as e:
        rep["problems"].append(f"{type(e).__name__}: {e}")
        rep["ok"] = False
        if out is not None:
            _merge_report(out, site, rep)
        return rep
    try:
        rep["instants"] = _instants_doc(src)
        rep["record"] = {"T": src.T, "dropped_without_instant": src.dropped,
                         "instant_source": src.instant_source, "years": src.years,
                         "first": src.dates[0] if src.dates else None,
                         "last": src.dates[-1] if src.dates else None}
        g = src.grid.describe()
        rep["grid"] = g
        if g["dx"] is not None and abs(abs(g["dx"]) - float(res_m)) > 1e-6:
            rep["problems"].append(f"pixel size {g['dx']} m is not {res_m} m")
        rep["lattice20"] = src.lattice_report if lat else {
            "phase": None, "note": f"{res_m} m grid: no 20 m sub-lattice (indices per pixel)"}
        rep["fingerprints"] = {"netcdf": src.fingerprint(full=full_fingerprint)}
        stats = scan_source(src) if scan else {}
        rep["scenes"] = stats
        if stats:
            scl11 = {}
            for d, e in stats.items():
                scl11[d[:4]] = scl11.get(d[:4], 0) + e["scl"][11]
            rep["scl11_per_year"] = scl11
        tab = src.offset_table()
        rep["offsets"] = {d: int(r.offset) for d, r in tab.iterrows()}
        rep["offset_counts"] = tab["state"].value_counts().to_dict()
        rep["cache"] = src.cache.stats()
        if not src.T:
            rep["problems"].append("no scene has a catalogue time")
    finally:
        src.close()
    rep["warnings"].append(f"legacy {res_m} m netCDF source: local software checks only; the "
                           "reported numbers come from the 10 m Zarr stores")
    rep["ok"] = not rep["problems"]
    rep["seconds"] = round(time.time() - t0, 1)
    if out is not None:
        _merge_report(out, site, rep)
    return rep


def _per_year(dm, key):
    out = {}
    for d, v in dm.items():
        out.setdefault(d[:4], {})
        for x in v.get(key, []):
            out[d[:4]][str(x)] = out[d[:4]].get(str(x), 0) + 1
    return out


def frozen_offsets(report):
    """``{date: 0 or 1000}`` from an inputs report (one site's dict, or a
    file with ``sites``); raises :class:`OffsetUnclear` if any is None."""
    off = report.get("offsets", {})
    bad = [d for d, v in off.items() if v is None]
    if bad:
        raise OffsetUnclear(f"{report.get('site')}: offsets unclear on {bad[:5]}")
    return {d: int(v) for d, v in off.items()}


def offset_rule_sha256():
    """sha256 of the per-scene offset rule: :data:`OFFSET_RULE` (its JSON)
    and the CRLF-normalised source of every function that carries it, in
    the order of ``settings.OFFSET_RULE_IMPL`` (the prep levels hash the
    same list). An inputs report records it; ``prepare`` refuses a report
    written with another (:func:`offset_rule_status`)."""
    h = hashlib.sha256()
    for name in settings.OFFSET_RULE_IMPL:
        h.update(b"\n--- " + name.encode("utf-8") + b"\n")
        h.update(settings.impl_source(name).encode("utf-8"))
    return h.hexdigest()


def offset_rule_status(report):
    """None when one site's inputs report was written with the current
    offset rule (its ``offset_rule_sha256``), else what it was written
    with (a short text)."""
    sha = report.get("offset_rule_sha256")
    cur = offset_rule_sha256()
    if sha == cur:
        return None
    rule = report.get("offset_rule")
    rev = (rule.get("revision") or {}).get("date") if isinstance(rule, dict) else None
    if not sha:
        return ("an offset rule without a recorded sha256 (V1"
                + (f", or a draft of the {rev} amendment" if rev else "") + ")")
    return (f"another offset rule (sha256 {str(sha)[:12]}, revision {rev or 'none'}; the current "
            f"one is {cur[:12]})")


def _site_reports(report):
    """``{site: report}`` from an inputs report: a path, a file's dict (with
    ``sites``) or one site's dict; ``{}`` for None."""
    if report is None:
        return {}
    if isinstance(report, (str, os.PathLike)):
        with open(report, encoding="utf-8") as f:
            report = json.load(f)
    if isinstance(report.get("sites"), dict):
        return dict(report["sites"])
    return {report["site"]: report} if report.get("site") else {}


def offset_summary(report, previous=None, sites=None, max_scenes=40):
    """Lines that summarise the per-scene offset decisions of an inputs
    report (a path, or the dict of the file or of one site), to paste back
    after a run on the server.

    Per site: ok, T, the counts by state and by rule, whether the report was
    written with the current rule, G-zarr and the identity per year, what
    rules (1)-(5) alone decide on the scenes the identity decided, the
    problems and warnings and, against ``previous`` (e.g. the report of the
    V1 rule) when given, the change of state; then one line per scene that
    is unclear, '+1000', demoted or contradicted now, or was '+1000' before
    (at most ``max_scenes``): state, rule, previous state, the water, cloud,
    frame and negative counts, and the reason."""
    cur, old = _site_reports(report), _site_reports(previous)
    sha = offset_rule_sha256()
    out = []
    for s in (sites or sorted(cur)):
        v = cur.get(s)
        if not isinstance(v, dict):
            out.append(f"== {s}: not in the report")
            continue
        o = old.get(s) if isinstance(old.get(s), dict) else {}
        which = "current rule" if v.get("offset_rule_sha256") == sha else "NOT the current rule"
        out.append(f"== {s} | ok {v.get('ok')} | T {(v.get('record') or {}).get('T')} | offsets "
                   f"{v.get('offset_counts')} | {which}")
        out.append(f"   by rule {v.get('offset_rule_counts')}")
        for y, g in sorted((v.get("g_zarr") or {}).items()):
            chk = g.get("identity_check") or "no identical record date"
            out.append(f"   G-zarr {y}: {g.get('verdict')} | {g.get('scenes_identical', 0)} scenes "
                       f"by identity | {chk} | {g.get('identity')}")
        if v.get("identity_overrides"):
            out.append(f"   the identity's scenes by rules (1)-(5) alone: "
                       f"{v['identity_overrides']}")
        for p in v.get("problems") or ():
            out.append(f"   PROBLEM {p}")
        for w in v.get("warnings") or ():
            out.append(f"   warning {w}")
        dec = v.get("offset_decisions") or {}
        odec = o.get("offset_decisions") or {}
        if odec:
            ch = collections.Counter(
                f"{(odec.get(d) or {}).get('state', 'absent')} -> {x['state']}"
                for d, x in dec.items())
            out.append(f"   before -> now: {dict(sorted(ch.items()))}")
        sc = v.get("scenes") or {}
        pick = [d for d in sorted(dec) if dec[d]["state"] != "no offset"
                or dec[d].get("rule") in ("demoted", "contradiction")
                or (odec.get(d) or {}).get("state") == "+1000 in the values"]
        for d in pick[:max_scenes]:
            x, e = dec[d], sc.get(d) or {}
            cs = _cloud_share(e)
            cloud = "-" if cs is None else f"{cs:.3f}"
            before = (odec.get(d) or {}).get("state", "absent") if odec else "-"
            out.append(f"   {d} {x['state']} [{x.get('rule')}] before {before} | water "
                       f"{e.get('n_water')} clear {e.get('n_water_clear')} clear-low "
                       f"{e.get('n_water_clear_low')} | SCL 8/9/10 {cloud} | frame valid "
                       f"{e.get('n_valid')} low {e.get('n_low')} | neg {e.get('n_neg')} | "
                       f"{str(x.get('reason'))[:220]}")
        if len(pick) > max_scenes:
            out.append(f"   ... {len(pick) - max_scenes} more such scenes")
    return out


def print_offset_summary(report=INPUTS_REPORT, previous=None, sites=None, max_scenes=40):
    """Print :func:`offset_summary` (the server one-liner)."""
    for line in offset_summary(report, previous, sites, max_scenes):
        print(line)
