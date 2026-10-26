"""
waterline_contour.py — ITEM/NIDEM waterline-contour intertidal DEM
==================================================================

A from-scratch re-implementation of the Intertidal Extents Model (ITEM v2.0;
Sagar et al., 2017) and of the National Intertidal Digital Elevation Model
(NIDEM; Bishop-Taylor et al., 2019) for Sentinel-2 raw-band cubes
(B03, B08, SCL on a (t, y, x) grid).

The method works in the SPATIAL domain, not per pixel along its own series:

1. scenes are pooled by their boundary tide level into 10 uniform intervals
   of the observed tidal range (OTR), the top two merged, giving 9 intervals
   (Sagar et al. 2017, p.156 and Appendix A p.168);
2. each interval is reduced to one per-pixel MEDIAN NDWI composite over the
   valid observations (Sagar et al. 2017, p.158 s3.2.2; Bishop-Taylor et al.
   2019, p.117);
3. each composite is thresholded at NDWI = 0 (Sagar et al. 2017, p.159) and
   the 9 land/water maps are summed into an ordinal relative-extents map
   R in 0..9 (Sagar et al. 2017, p.159; ITEM v2.0 legend);
4. isolines of R at 0.5 .. 8.5 (marching squares; Bishop-Taylor et al. 2019,
   p.117 s2.2) are tagged with the median tide height of the interval's
   scenes (Bishop-Taylor et al. 2019, p.118; Sagar et al. 2017, p.159 s3.4);
5. a Delaunay TIN (``scipy.interpolate.griddata``, linear) turns the tagged
   vertices into a DEM on the native grid (Bishop-Taylor et al. 2019, p.118);
6. the DEM is kept where R is in 1..8 ("unfiltered"; Bishop-Taylor et al.
   2019, p.118) and additionally where the mean per-interval NDWI standard
   deviation is at most 0.25 ("filtered", the validated product;
   Bishop-Taylor et al. 2019, p.118 s2.3 and p.119 s2.5).

No per-pixel curve, likelihood or lag is fitted, and no parameter is
estimated from any validation data.

Frozen choices
--------------
Published constants (not tuned):

* 10 uniform intervals of OTR = max(h) - min(h) over ALL retained scenes,
  cloudy or not (Sagar et al. 2017, p.156 and Fig. 3); top two merged -> 9.
* composite statistic: per-pixel median of NDWI over the valid observations
  (ITEM pixel-quality mask, see below).
* water iff median NDWI > 0.
* contour levels 0.5, 1.5, ..., 8.5; minimum 2 vertices per line
  (NIDEM_generation.py L218-229, L624).
* gap fill before contouring: binary dilation of the valid pixels by two
  iterations, nearest-valid fill inside it, NaN beyond
  (NIDEM_generation.py L175-187).
* linear TIN interpolation (NIDEM_generation.py L256-263).
* confidence mask: mean NDWI SD <= 0.25 (Bishop-Taylor et al. 2019, p.118;
  NIDEM_generation.py L328).

The 0.25 threshold was defined on NBAR NDWI, which lies in [-1, 1]. Our
inputs are offset-harmonised L2A reflectances, which can be negative, so
|NDWI| > 1 occurs (exactly when B03 and B08 have opposite signs) and inflates
the SD. The primary ``filtered`` product keeps the published statistic on
the raw NDWI, unchanged. A SUPPLEMENTARY ``filtered_bounded`` product
recomputes only the confidence with every NDWI clipped to [-1, 1]
(:func:`bounded_statistics`; rule fixed in review round 1 before any
scoring). The median composites, R and the DEM are identical in both.

Choices the papers leave open, fixed here before any scoring:

* interval boundaries: the lowest interval is closed [e0, e1], the others
  right-closed (e_{k-1}, e_k];
* a median NDWI of exactly 0 counts as LAND (same as MAREA's wet = NDWI > 0);
* R = sum of the 9 land flags (non-monotone pixels are reported, not fixed);
* SD with ddof = 0 for NDWI, ddof = 1 for the tide heights (pandas default,
  as NIDEM's uncertainty code);
* a pixel is valid iff it has >= 1 valid observation in every interval
  (ITEM v2.0 "-6666: no available observations in one or more of the
  percentile interval subsets");
* an empty interval stops the method (:class:`EmptyIntervalError`), and so
  does a record whose intervals are observed so sparsely that fewer than
  :data:`MIN_VALID_SHARE` of the scored pixels would be valid, e.g. when
  the only scene of an extreme interval is overcast
  (:func:`interval_coverage`, :class:`NoClearObservationsInIntervalError`;
  review round 2). Both are stop-and-report guards, not method rules: no
  scene is dropped and no interval redefined;
* Z_j and U_j use every scene of interval j, including cloudy ones;
* contours are KEYED BY LEVEL. NIDEM_generation.py (L244-245 with
  L714-718) pairs contours with elevations by position in a dict that skips
  empty levels, so one empty level shifts every later elevation; that bug is
  deliberately not reproduced;
* interpolation happens in pixel-index (row, col) space, where integer
  indices are pixel centres, so NIDEM's +0.5 px shift (which belongs to its
  affine, georeferenced coordinates) is not applied;
* the NIDEM +25 m SRTM and -25 m bathymetry masks are no-ops at our sites
  and are omitted.

Pixel-quality mask (revision of 2026-09-29, "each method applies its own
published rules"): ITEM's pixel-quality mask mapped to the Sentinel-2 Scene
Classification (SCL). ITEM masks every tile observation for "cloud, cloud
shadow, or saturated in any of the bands" (Sagar et al. 2017, p.155) and
"for cloud, band saturation and contiguity" (ITEM v2.0 product description,
processing step 5). Mapped to SCL, the MASKED classes are
:data:`ITEM_PQ_MASKED_SCL` = {0 no data (contiguity; NaN in our cubes),
1 saturated or defective, 3 cloud shadows, 8 cloud medium probability,
9 cloud high probability, 10 thin cirrus}; every other class (2 dark area /
topographic cast shadow, 4 vegetation, 5 bare soil, 6 water,
7 unclassified, 11 snow or ice) is a valid observation
(:func:`pixel_valid`). NDWI = (B03 - B08) / (B03 + B08), NaN where the sum
is 0 (or a band is missing), exactly as :func:`pyintertidal.marea.extract`.
Before this revision the method used MAREA's clear set SCL {4, 5, 6, 7}
(:data:`CLEAR`), which is now kept only to check the streamed SCL against
the extraction and to count the observations the NIDEM mask adds or removes.

Labelled variant: "NIDEM, Sentinel-2 scene rule"
-------------------------------------------------
NOT the primary product. It replicates the published Sentinel-2 reference
implementation of this workflow, the Digital Earth Africa notebook
"Modelling intertidal elevation using tidal data and Sentinel-2"
(Intertidal_elevation_S2), which loads its data with
``load_ard(..., min_gooddata=0.5)``. ``deafrica_tools.datahandling.load_ard``
(revision of 2026-09-29: now LITERAL):

* scene rule: ``pq_mask = enum_to_bool(SCL, categories_to_mask_s2)`` with
  the default bad classes SCL {1, 3, 8, 9, 10} (:data:`DEAFRICA_S2_BAD_SCL`);
  ``data_perc = (~pq_mask).sum(axis=[1, 2]) / (pq_mask.shape[1] *
  pq_mask.shape[2])``; ``keep = data_perc >= min_gooddata``. The good
  fraction of a scene is therefore the share of ALL H x W frame pixels
  whose SCL is NOT in {1, 3, 8, 9, 10}; no-data (SCL 0, NaN in our cubes)
  counts as GOOD, exactly as in load_ard (:func:`frame_good_fraction`,
  :func:`s2_scene_rule`), before any tide binning;
* pixel mask: ``erase_bad(ds_data, where=pq_mask)`` (SCL {1, 3, 8, 9, 10}
  masked; SCL no-data is not masked), and
  ``valid_data_mask = (ds_data > 1).all(dim="band")`` with
  ``keep_good_only`` (every loaded band must have a raw DN > 1; DN 0 is the
  band nodata). Our cubes hold offset-harmonised values DN - 1000 with DN 0
  as NaN, so the DN rule is value > -999 (:data:`LOAD_ARD_MIN_VALUE`) on the
  two bands we hold, B03 and B08 (the notebook also loads red and blue,
  which are not in our cubes). A band no-data pixel is NaN, so its NDWI is
  NaN (:func:`pixel_valid` with ``rule="load_ard"``).

Everything else is unchanged and runs on the retained scenes: OTR,
intervals, Z_j, U_j, composites, the validity rule, contours, TIN and
filters. The earlier variant statistic (share of the frame with SCL in
{4, 5, 6, 7}, no data NOT good) is reported for information only
(:func:`clear_scl_fraction`).

References
----------
Sagar, S., Roberts, D., Bala, B., Lymburner, L. (2017). Extracting the
    intertidal extent and topography of the Australian coastline from a 28
    year time series of Landsat observations. Remote Sensing of Environment
    195, 153-169. https://doi.org/10.1016/j.rse.2017.04.009
Bishop-Taylor, R., Sagar, S., Lymburner, L., Beaman, R.J. (2019). Between
    the tides: Modelling the elevation of Australia's exposed intertidal
    zone at continental scale. Estuarine, Coastal and Shelf Science 223,
    115-128. https://doi.org/10.1016/j.ecss.2019.03.006
Geoscience Australia, NIDEM reference code ``NIDEM_generation.py``
    (github.com/GeoscienceAustralia/nidem, master; line numbers as of
    2026-09-29). Consulted for constants only; no code is reused.
Geoscience Australia. Intertidal Extents Model (ITEM) v2.0.0 product
    description, DEA Knowledge Hub (processing step 5, "Mask tile
    observations for pixel quality"), retrieved 2026-09-29.
Digital Earth Africa (2026). Modelling intertidal elevation using tidal
    data and Sentinel-2 (Real_world_examples/Intertidal_elevation_S2.ipynb,
    digitalearthafrica/deafrica-sandbox-notebooks, main @ 0ecfbeb,
    2026-09-02) and ``deafrica_tools.datahandling.load_ard`` (same
    repository, Tools/deafrica_tools/datahandling.py @ a309e7c,
    2024-12-02). Retrieved 2026-09-29; used for the VARIANT only.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from skimage.measure import find_contours

#: MAREA's clear-sky SCL classes (vegetation, bare soil, water,
#: unclassified), as pyintertidal.marea.CLEAR. NOT a NIDEM mask since the
#: 2026-09-29 revision: used only to check the streamed SCL against the
#: extraction's C and to count the observations the NIDEM masks add or
#: remove relative to it (and for the superseded variant statistic,
#: :func:`clear_scl_fraction`, information only)
CLEAR = (4, 5, 6, 7)
#: the Sentinel-2 L2A Scene Classification codes
SCL_CODES = tuple(range(12))
#: PRIMARY pixel-quality mask: ITEM's PQ mask (cloud, cloud shadow, band
#: saturation, contiguity; Sagar et al. 2017 p.155, ITEM v2.0 processing
#: step 5) mapped to SCL. MASKED: 0 no data (contiguity; NaN SCL in our
#: cubes), 1 saturated or defective, 3 cloud shadows, 8 cloud medium
#: probability, 9 cloud high probability, 10 thin cirrus. Valid: 2, 4, 5,
#: 6, 7, 11
ITEM_PQ_MASKED_SCL = (0, 1, 3, 8, 9, 10)
#: pixel rules: "item" (PRIMARY) and "load_ard" (VARIANT)
PIXEL_RULES = ("item", "load_ard")
#: uniform partition of the observed tidal range (Sagar et al. 2017, p.156)
N_UNIFORM = 10
#: after merging the 80-90 % and 90-100 % intervals
N_INTERVALS = 9
#: water iff median NDWI > this (Sagar et al. 2017, p.159)
NDWI_THRESHOLD = 0.0
#: filtered product keeps mean NDWI SD <= this (Bishop-Taylor et al. 2019, p.118)
CONFIDENCE_MAX = 0.25
#: SUPPLEMENTARY only (review round 1, 2026-09-29, fixed before scoring): the
#: NBAR NDWI range [-1, 1] on which CONFIDENCE_MAX was defined; the
#: supplementary confidence is recomputed with every NDWI clipped to it
NDWI_BOUND = 1.0
#: PRE-FLIGHT GUARD, not a method constant (review round 2, 2026-09-29; fixed
#: from no truth before any Ems composite or score was computed). A record
#: whose expected valid share of the scored pixels (>= 1 valid observation in
#: every interval, counted before streaming) is below this is declared NOT
#: COMPUTABLE under the ITEM rules and is not scored
#: (:func:`interval_coverage`). It is the analogue of the empty-interval stop
#: (:class:`EmptyIntervalError`): NIDEM has no rule for either case.
MIN_VALID_SHARE = 0.01
#: VARIANT "NIDEM, Sentinel-2 scene rule" ONLY; the primary never uses it.
#: Decided 2026-09-29, truth-blind, before any Ems truth was read. Value of
#: the Digital Earth Africa Sentinel-2 reference notebook
#: (Intertidal_elevation_S2: ``load_ard(..., min_gooddata=0.5, ...)``). A
#: scene is kept iff its full-frame good fraction is >= this (load_ard:
#: ``keep = (data_perc >= min_gooddata)``).
S2_SCENE_MIN_GOODDATA = 0.5
#: VARIANT scene rule AND pixel mask (literal load_ard since the 2026-09-29
#: revision): the SCL classes load_ard masks for Sentinel-2 by default
#: (``categories_to_mask_s2``: saturated or defective, cloud shadows, cloud
#: medium probability, cloud high probability, thin cirrus; codes from the
#: DE Africa s2_l2a_c1 product definition). Everything else is good,
#: including no data (SCL 0 / NaN)
DEAFRICA_S2_BAD_SCL = (1, 3, 8, 9, 10)
#: VARIANT pixel mask: load_ard's ``valid_data_mask = (ds_data > 1)`` on the
#: raw DN of every loaded band (datahandling.py @ a309e7c L544-547). Our
#: cubes hold DN + BOA_ADD_OFFSET with DN 0 as NaN
LOAD_ARD_MIN_RAW_DN_EXCLUSIVE = 1
#: the Sentinel-2 L2A BOA offset (processing baseline >= 04.00) that the
#: cubes' values carry (value = DN - 1000; DN 0 = nodata -> NaN)
BOA_ADD_OFFSET = -1000
#: raw DN > 1 <=> cube value > -999
LOAD_ARD_MIN_VALUE = LOAD_ARD_MIN_RAW_DN_EXCLUSIVE + BOA_ADD_OFFSET
#: isolines of the relative-extents map (NIDEM_generation.py L218-229)
CONTOUR_LEVELS = tuple(float(v) for v in np.arange(0.5, N_INTERVALS, 1.0))
#: minimum vertices per contour line (NIDEM_generation.py L624)
MIN_VERTICES = 2
#: binary-dilation iterations of the valid mask for the gap fill
GAP_FILL_ITERATIONS = 2
#: level perturbation for the find_contours KeyError retry (skimage #4830)
LEVEL_EPS = 1e-12


class EmptyIntervalError(ValueError):
    """An interval of the observed tidal range holds no scene. NIDEM has no
    rule for this case, so the method stops instead of inventing one."""


class NoClearObservationsInIntervalError(EmptyIntervalError):
    """An interval holds scenes, but (almost) no pixel has a clear
    observation in it, so the ITEM validity rule (-6666, "no available
    observations in one or more of the percentile interval subsets") leaves
    (almost) no valid pixel. Typical cause: the only scene of an extreme
    interval is overcast. NIDEM has no rule for this case either, so the
    method stops and reports instead of inventing one. ``coverage`` holds
    the pre-flight numbers (:func:`interval_coverage`) when available."""

    def __init__(self, message, coverage=None):
        super().__init__(message)
        self.coverage = coverage


# ─────────────────────────────────────────────────────────────────────────────
#  0. Pixel masks (primary: ITEM PQ; variant: load_ard) and the VARIANT
#     scene rule (load_ard min_gooddata)
# ─────────────────────────────────────────────────────────────────────────────

def pixel_valid(scl, b03=None, b08=None, rule="item"):
    """Per-pixel observation mask of a scene (or block), BEFORE the NDWI.

    ``rule="item"`` (PRIMARY): ITEM's pixel-quality mask mapped to SCL;
    valid iff the SCL is defined (NaN = SCL 0 no data = contiguity) and not
    in :data:`ITEM_PQ_MASKED_SCL`. The bands are not looked at (a missing
    band gives a NaN NDWI, which is never an observation).

    ``rule="load_ard"`` (VARIANT): load_ard's pixel mask; valid iff the SCL
    is NOT in :data:`DEAFRICA_S2_BAD_SCL` (NaN SCL, i.e. SCL no data, is
    not masked, as in load_ard) AND both bands have a raw DN > 1, i.e. a
    cube value > :data:`LOAD_ARD_MIN_VALUE` (a NaN band fails).
    """
    scl = np.asarray(scl)
    if rule == "item":
        return np.isfinite(scl) & ~np.isin(scl, ITEM_PQ_MASKED_SCL)
    if rule == "load_ard":
        if b03 is None or b08 is None:
            raise ValueError("the load_ard pixel mask needs B03 and B08")
        with np.errstate(invalid="ignore"):
            dn_ok = ((np.asarray(b03) > LOAD_ARD_MIN_VALUE)
                     & (np.asarray(b08) > LOAD_ARD_MIN_VALUE))
        return ~np.isin(scl, DEAFRICA_S2_BAD_SCL) & dn_ok
    raise ValueError(f"unknown pixel rule {rule!r}; expected one of "
                     f"{PIXEL_RULES}")


def load_ard_good(scl):
    """load_ard's per-pixel good flag for the scene rule: ``~pq_mask`` with
    ``pq_mask = enum_to_bool(SCL, categories_to_mask_s2)``, i.e. SCL NOT in
    :data:`DEAFRICA_S2_BAD_SCL`. No data (NaN here, SCL 0 there) is good."""
    return ~np.isin(np.asarray(scl), DEAFRICA_S2_BAD_SCL)


def frame_good_fraction(scl):
    """VARIANT scene-rule statistic, literal load_ard: share of ALL pixels
    of one scene's frame whose SCL is NOT in :data:`DEAFRICA_S2_BAD_SCL`.

    ``scl`` is the SCL of the whole frame (any shape). No data (NaN, the
    SCL nodata 0 of load_ard) counts as GOOD, and so do 2 and 11, exactly
    as load_ard's ``data_perc = (~pq_mask).sum(axis=[1, 2]) /
    (pq_mask.shape[1] * pq_mask.shape[2])``: the denominator is every pixel
    of the frame, and only the pixel-quality band is looked at.
    """
    scl = np.asarray(scl)
    if scl.size == 0:
        raise ValueError("empty frame")
    return float(np.count_nonzero(load_ard_good(scl)) / scl.size)


def clear_scl_fraction(scl):
    """INFORMATION ONLY (the superseded variant statistic, before the
    2026-09-29 revision): share of ALL frame pixels with SCL in
    :data:`CLEAR`; no data (NaN) is NOT counted."""
    scl = np.asarray(scl)
    if scl.size == 0:
        raise ValueError("empty frame")
    return float(np.count_nonzero(np.isin(scl, CLEAR)) / scl.size)


def s2_scene_rule(good_fraction, min_gooddata=S2_SCENE_MIN_GOODDATA):
    """Keep flag per scene under the VARIANT scene rule:
    ``good_fraction >= min_gooddata`` (load_ard keeps
    ``data_perc >= min_gooddata``, so a share of exactly 0.5 is kept).

    Raises ``EmptyIntervalError`` if no scene is retained (then every
    interval is empty).
    """
    g = np.asarray(good_fraction, float).ravel()
    if not np.all(np.isfinite(g)) or np.any((g < 0) | (g > 1)):
        raise ValueError("good fractions must be finite and in [0, 1]")
    if not 0.0 <= float(min_gooddata) <= 1.0:
        raise ValueError("min_gooddata must be in [0, 1]")
    keep = g >= float(min_gooddata)
    if not keep.any():
        raise EmptyIntervalError(
            f"no scene has a full-frame good fraction >= {min_gooddata}; "
            "every interval is empty under the Sentinel-2 scene rule")
    return keep


def good_fractions_from_arrays(good):
    """Full-frame good fraction of each scene of an in-memory (T, H, W)
    good flag: for the VARIANT scene rule pass ``load_ard_good(scl)``
    (then each value equals :func:`frame_good_fraction` of that scene)."""
    good = np.asarray(good, bool)
    if good.ndim != 3:
        raise ValueError("good must be (T, H, W)")
    return good.reshape(good.shape[0], -1).mean(axis=1)


class PackedRows:
    """Read-only (T, P) bool rows stored bit-packed (``np.packbits`` along
    the pixel axis); ``rows[q]`` unpacks one row. Lets
    :func:`clear_counts_by_interval` run on the pre-pass masks without a
    (T, P) bool array in memory."""

    def __init__(self, packed, n):
        self.packed = np.asarray(packed, np.uint8)
        self.n = int(n)
        self.shape = (self.packed.shape[0], self.n)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, q):
        return np.unpackbits(self.packed[q], count=self.n).astype(bool)

    def subset(self, sel):
        return PackedRows(self.packed[np.asarray(sel)], self.n)


def stream_scene_quality(cube_path, time_index, keep, rule="item",
                         verbose=True):
    """Truth-free pre-pass over the given cube scenes, one whole frame of
    SCL, B03 and B08 at a time (about 12 H W bytes of memory).

    Per scene it returns the frame statistics of the scene rules and the
    observation masks at the ``keep`` pixels (flat indices), so that the
    pre-flight guard (:func:`interval_coverage`) can run under the run's
    own pixel mask ``rule`` (:func:`pixel_valid`) before any composite.

    Returns a dict:

    * (T,) float: ``load_ard_good`` (:func:`frame_good_fraction`, the
      VARIANT rule's statistic), ``clear_scl`` (:func:`clear_scl_fraction`,
      the superseded statistic, information only), ``item_valid`` (share of
      the frame valid under the ITEM mask, information only), ``no_data``
      (share of NaN SCL);
    * (T,) int: ``scl_zero_px`` (SCL == 0 pixels, expected 0: no data is
      NaN in our cubes), ``band_le_load_ard_min_px`` (frame pixels with
      B03 or B08 <= :data:`LOAD_ARD_MIN_VALUE`, i.e. raw DN <= 1),
      ``band_below_offset_px`` (B03 or B08 < LOAD_ARD_MIN_VALUE, i.e. raw
      DN <= 0; expected 0 because DN 0 is NaN), and at keep:
      ``n_obs_keep`` (valid under ``rule`` with a finite NDWI),
      ``n_marea_obs_keep`` (SCL in :data:`CLEAR` with a finite NDWI),
      ``n_both_keep``, ``n_added_keep`` (observations of the rule that
      MAREA's C does not have) and ``n_removed_keep`` (MAREA observations
      the rule masks);
    * ``obs_keep`` and ``clear_keep``: :class:`PackedRows` (T, P) of the
      rule's observations (valid and finite NDWI) and of SCL in CLEAR
      (MAREA's C, before the NDWI) at keep;
    * ``rule``, ``px`` (H W), ``P`` (len(keep)) and ``seconds``.

    Raises ``ValueError`` if an SCL value is not one of the codes 0..11.
    """
    import xarray as xr

    if rule not in PIXEL_RULES:
        raise ValueError(f"unknown pixel rule {rule!r}")
    t0 = time.time()
    time_index = np.asarray(time_index, int)
    keep = np.asarray(keep, np.int64)
    T, P = len(time_index), len(keep)
    nb = (P + 7) // 8
    out = {k: np.empty(T, np.float64) for k in
           ("load_ard_good", "clear_scl", "item_valid", "no_data")}
    for k in ("scl_zero_px", "band_le_load_ard_min_px",
              "band_below_offset_px", "n_obs_keep", "n_marea_obs_keep",
              "n_both_keep", "n_added_keep", "n_removed_keep"):
        out[k] = np.zeros(T, np.int64)
    obs_p = np.zeros((T, nb), np.uint8)
    clear_p = np.zeros((T, nb), np.uint8)
    ds = xr.open_dataset(cube_path)
    try:
        t_dim = [k for k in ds["SCL"].dims if k not in ("x", "y")][0]
        for v in ("B03", "B08", "SCL"):
            if ds[v].dims != (t_dim, "y", "x"):
                raise ValueError(f"{v} dims {ds[v].dims} are not (t, y, x)")
        px = int(ds.sizes["y"] * ds.sizes["x"])
        for q, t in enumerate(time_index):
            sl = {t_dim: int(t)}
            scl = ds["SCL"].isel(sl).values.ravel()
            g = ds["B03"].isel(sl).values.ravel().astype(np.float32)
            n = ds["B08"].isel(sl).values.ravel().astype(np.float32)
            fin_scl = np.isfinite(scl)
            if not np.all(np.isin(scl[fin_scl], SCL_CODES)):
                bad = np.unique(scl[fin_scl & ~np.isin(scl, SCL_CODES)])
                raise ValueError(f"scene t={t}: SCL values {bad[:10]} are "
                                 "not SCL codes 0..11")
            out["load_ard_good"][q] = frame_good_fraction(scl)
            out["clear_scl"][q] = clear_scl_fraction(scl)
            out["item_valid"][q] = float(np.count_nonzero(
                pixel_valid(scl, rule="item")) / px)
            out["no_data"][q] = float(np.count_nonzero(~fin_scl) / px)
            out["scl_zero_px"][q] = int(np.count_nonzero(scl == 0))
            with np.errstate(invalid="ignore"):
                out["band_le_load_ard_min_px"][q] = int(np.count_nonzero(
                    (g <= LOAD_ARD_MIN_VALUE) | (n <= LOAD_ARD_MIN_VALUE)))
                out["band_below_offset_px"][q] = int(np.count_nonzero(
                    (g < LOAD_ARD_MIN_VALUE) | (n < LOAD_ARD_MIN_VALUE)))
            s_k, g_k, n_k = scl[keep], g[keep], n[keep]
            ndwi_k, valid_k = scene_ndwi(g_k, n_k, s_k, rule=rule)
            fin = np.isfinite(ndwi_k)
            obs = valid_k & fin
            clear_k = np.isin(s_k, CLEAR)
            marea = clear_k & fin
            out["n_obs_keep"][q] = int(obs.sum())
            out["n_marea_obs_keep"][q] = int(marea.sum())
            out["n_both_keep"][q] = int((obs & marea).sum())
            out["n_added_keep"][q] = int((obs & ~marea).sum())
            out["n_removed_keep"][q] = int((marea & ~obs).sum())
            obs_p[q] = np.packbits(obs)
            clear_p[q] = np.packbits(clear_k)
            if verbose and (q + 1) % 100 == 0:
                print(f"  [scene quality] {q + 1} of {T} scenes "
                      f"({time.time() - t0:.0f} s)", flush=True)
    finally:
        ds.close()
    out["obs_keep"] = PackedRows(obs_p, P)
    out["clear_keep"] = PackedRows(clear_p, P)
    out.update(rule=rule, px=px, P=P, seconds=time.time() - t0)
    return out


def subset_scene_quality(q, sel):
    """The pre-pass dict of :func:`stream_scene_quality` restricted to the
    scenes flagged in ``sel`` (order kept)."""
    sel = np.asarray(sel, bool)
    out = dict(q)
    for k, v in q.items():
        if isinstance(v, PackedRows):
            out[k] = v.subset(sel)
        elif isinstance(v, np.ndarray) and v.shape == sel.shape:
            out[k] = v[sel]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  1. Tidal intervals
# ─────────────────────────────────────────────────────────────────────────────

def interval_assignment(h):
    """Assign each scene to one of the 9 ITEM tidal intervals.

    Implements Sagar et al. (2017), p.156 ("a uniform partition of the OTR
    into 10 intervals"; "merge the 80-90 % and 90-100 %") and Appendix A
    p.168, with the observed tidal range taken over ALL scenes given
    (Fig. 3: tiles are used "regardless of data completeness or quality").
    Z_j is the median and U_j the standard deviation (ddof = 1) of the tide
    heights of all scenes of interval j (Bishop-Taylor et al. 2019, p.118;
    Sagar et al. 2017, p.160).

    Parameters
    ----------
    h : array (T,)
        Boundary tide level of each scene (metres), all finite.

    Returns
    -------
    j : int array (T,)
        Interval of each scene, 1..9. The lowest interval is closed
        [e0, e1]; the others are right-closed (e_{k-1}, e_k]; raw interval
        10 is merged into 9.
    edges : float array (11,)
        The uniform edges e_k = LOT + 0.1 k OTR, k = 0..10. Interval 9 spans
        (e_8, e_10].
    Z : float array (9,)
        Median tide height of each interval's scenes.
    U : float array (9,)
        Standard deviation (ddof = 1) of each interval's tide heights; NaN
        for an interval with a single scene.
    sizes : int array (9,)
        Number of scenes |S_j| per interval.

    Raises
    ------
    EmptyIntervalError
        If any of the 9 intervals is empty.
    """
    h = np.asarray(h, float).ravel()
    if h.size == 0 or not np.all(np.isfinite(h)):
        raise ValueError("interval_assignment needs a non-empty, finite h")
    lot, hot = float(h.min()), float(h.max())
    otr = hot - lot
    if not otr > 0:
        raise ValueError("observed tidal range is zero")
    edges = lot + 0.1 * np.arange(N_UNIFORM + 1) * otr
    # searchsorted(side="left") gives k with e_{k-1} < h <= e_k; h == e_0 -> 0
    raw = np.clip(np.searchsorted(edges, h, side="left"), 1, N_UNIFORM)
    j = np.minimum(raw, N_INTERVALS).astype(int)
    sizes = np.bincount(j, minlength=N_INTERVALS + 1)[1:]
    if np.any(sizes == 0):
        raise EmptyIntervalError(
            f"empty tidal interval(s) {list(np.flatnonzero(sizes == 0) + 1)}; "
            f"|S_j| = {sizes.tolist()}")
    Z = np.array([np.median(h[j == q]) for q in range(1, N_INTERVALS + 1)])
    U = np.array([np.std(h[j == q], ddof=1) if sizes[q - 1] > 1 else np.nan
                  for q in range(1, N_INTERVALS + 1)])
    return j, edges, Z, U, sizes


# ─────────────────────────────────────────────────────────────────────────────
#  1b. Pre-flight: are the intervals observed at all?
# ─────────────────────────────────────────────────────────────────────────────

def clear_counts_by_interval(clear, ndwi, j):
    """Valid observations with a finite NDWI per interval, N_j(p), from
    row-wise (T, P) arrays of the record scenes at P pixels.

    ``clear`` (T, P) bool (the run's pixel mask; any object with ``shape``
    and row indexing, e.g. a memory map or :class:`PackedRows`) and
    ``ndwi`` (T, P) float (or None to count the mask alone, when it
    already includes the finite-NDWI condition): one scene row is read at a
    time.
    ``j`` (T,) is the interval of each scene. The result equals the
    ``count`` of :func:`stream_composites` at those pixels, so it predicts
    the validity map before the cube is streamed.

    Returns ``(counts (9, P) int32, per_scene (T,) int64)``, the latter the
    number of the P pixels at which each scene is clear with a finite NDWI.
    """
    j = np.asarray(j, int)
    T, P = clear.shape
    if len(j) != T or (ndwi is not None and ndwi.shape != (T, P)):
        raise ValueError("clear, ndwi and j do not match")
    counts = np.zeros((N_INTERVALS, P), np.int32)
    per_scene = np.zeros(T, np.int64)
    for q in range(T):
        c = np.asarray(clear[q], bool)
        if ndwi is not None:
            c = c & np.isfinite(np.asarray(ndwi[q]))
        counts[j[q] - 1] += c
        per_scene[q] = int(c.sum())
    return counts, per_scene


def interval_coverage(counts, min_valid_share=MIN_VALID_SHARE,
                      raise_error=True):
    """Pre-flight guard on the per-interval clear counts (P pixels).

    A pixel is valid iff it has >= 1 valid observation in every interval
    (the frozen ITEM rule), so the valid share is known before any
    composite is computed. If it is below ``min_valid_share`` the product
    is NOT COMPUTABLE for this record under the ITEM rules and, with
    ``raise_error``, :class:`NoClearObservationsInIntervalError` is raised
    carrying the numbers. The guard changes no product where it passes.

    Returns a dict: ``px``, ``share_px_with_ge1_clear_obs`` (per interval),
    ``median_N_j`` (per interval), ``intervals_with_zero_clear_obs``,
    ``intervals_below_floor`` (1-based), ``expected_valid_px``,
    ``expected_valid_share``, ``min_valid_share`` and ``passed``.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] != N_INTERVALS:
        raise ValueError("counts must be (9, P)")
    P = counts.shape[1]
    has = counts >= 1
    share = has.mean(axis=1) if P else np.zeros(N_INTERVALS)
    valid = has.all(axis=0)
    n_valid = int(valid.sum())
    valid_share = float(n_valid / P) if P else 0.0
    cov = {
        "px": int(P),
        "share_px_with_ge1_clear_obs": [float(v) for v in share],
        "median_N_j": [float(v) for v in (np.median(counts, axis=1) if P
                                          else np.zeros(N_INTERVALS))],
        "intervals_with_zero_clear_obs":
            [int(q) + 1 for q in np.flatnonzero(~has.any(axis=1))],
        "intervals_below_floor":
            [int(q) + 1 for q in np.flatnonzero(share < min_valid_share)],
        "expected_valid_px": n_valid,
        "expected_valid_share": valid_share,
        "min_valid_share": float(min_valid_share),
        "passed": bool(valid_share >= min_valid_share),
    }
    if raise_error and not cov["passed"]:
        raise NoClearObservationsInIntervalError(
            f"expected valid share {valid_share:.3g} ({n_valid} of {P} px) is "
            f"below the pre-flight floor {min_valid_share}; interval(s) "
            f"{cov['intervals_below_floor']} have a valid observation at "
            f"fewer than that share of the pixels (shares "
            f"{[round(v, 6) for v in cov['share_px_with_ge1_clear_obs']]}). "
            "NIDEM is not computable for this record under the ITEM rules.",
            coverage=cov)
    return cov


# ─────────────────────────────────────────────────────────────────────────────
#  2. Per-interval composites
# ─────────────────────────────────────────────────────────────────────────────

def scene_ndwi(b03, b08, scl, rule="item"):
    """NDWI and pixel mask of one scene (or block).

    NDWI = (B03 - B08) / (B03 + B08) in float32 (McFeeters 1996; Sagar et
    al. 2017, p.157 Eq. 1), NaN where B03 + B08 == 0 (or a band is
    missing), exactly as :func:`pyintertidal.marea.extract` computes it;
    the mask is :func:`pixel_valid` under ``rule`` ("item", the PRIMARY
    ITEM pixel-quality mask, or "load_ard", the VARIANT). Returns
    ``(ndwi, valid)`` with NDWI NOT yet masked.
    """
    g = np.asarray(b03, np.float32)
    n = np.asarray(b08, np.float32)
    valid = pixel_valid(scl, g, n, rule=rule)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(g + n != 0, (g - n) / (g + n), np.nan)
    return ndwi.astype(np.float32, copy=False), valid


def interval_statistics(stack):
    """Median, standard deviation (ddof = 0) and count of NaN-masked NDWI.

    ``stack`` is (n, rows, cols) float32 with NaN for every unusable
    observation (cloudy, or NDWI undefined). It is SORTED IN PLACE (NaN go
    last) to take the median without a copy. The median equals
    ``np.nanmedian(stack, axis=0)`` (mean of the two middle values for an
    even count); the SD is a two-pass float64 ``nanstd`` with ddof = 0.
    Pixels with no observation get NaN median and SD and count 0.

    Returns ``(median float32, sd float32, count int64)``, each (rows, cols).
    """
    stack.sort(axis=0)
    cnt = np.isfinite(stack).sum(axis=0)
    has = cnt > 0
    lo = np.where(has, (cnt - 1) // 2, 0)[None]
    hi = np.where(has, cnt // 2, 0)[None]
    a = np.take_along_axis(stack, lo, axis=0)[0].astype(np.float64)
    b = np.take_along_axis(stack, hi, axis=0)[0].astype(np.float64)
    med = np.where(has, 0.5 * (a + b), np.nan)
    sd = _sorted_nanstd(stack, cnt)
    return med.astype(np.float32), sd.astype(np.float32), cnt


def _sorted_nanstd(stack, cnt):
    """Two-pass float64 SD (ddof = 0) of a stack whose finite values come
    first along axis 0 (``cnt`` of them per pixel); NaN where cnt == 0."""
    n = stack.shape[0]
    denom = np.maximum(cnt, 1)
    s1 = np.zeros(cnt.shape, np.float64)
    for q in range(n):
        s1 += np.where(q < cnt, stack[q], 0.0)
    mean = s1 / denom
    s2 = np.zeros(cnt.shape, np.float64)
    for q in range(n):
        d = np.where(q < cnt, stack[q] - mean, 0.0)
        s2 += d * d
    return np.where(cnt > 0, np.sqrt(s2 / denom), np.nan)


def bounded_statistics(stack, cnt, bound=NDWI_BOUND):
    """SUPPLEMENTARY confidence statistic on NDWI clipped to [-bound, bound].

    The 0.25 confidence threshold was defined on NBAR NDWI, which lies in
    [-1, 1] because NBAR reflectances are non-negative (Bishop-Taylor et al.
    2019, p.118). Offset-harmonised Sentinel-2 L2A reflectances can be
    negative; |NDWI| > 1 exactly when B03 and B08 have opposite signs, and
    it is unbounded when their sum is near 0, so a single such observation
    can dominate the per-interval SD. This function reports those
    observations and the SD they would have inside the NBAR range. It never
    changes the median composites, R or the DEM.

    ``stack`` must be the stack SORTED by :func:`interval_statistics` (NaN
    last) and ``cnt`` its count; it is CLIPPED IN PLACE (clipping is
    monotone, so the order and the NaN positions are kept).

    Returns ``(sd float32, n_above int64, n_below int64, max_abs float)``:
    the ddof = 0 SD of the clipped values, the per-pixel number of
    observations with NDWI > bound and with NDWI < -bound, and the largest
    |NDWI| before clipping (NaN if the stack holds no observation).
    """
    n = stack.shape[0]
    n_above = np.zeros(cnt.shape, np.int64)
    n_below = np.zeros(cnt.shape, np.int64)
    with np.errstate(invalid="ignore"):
        for q in range(n):
            n_above += stack[q] > bound          # NaN compares False
            n_below += stack[q] < -bound
    has = cnt > 0
    if has.any():
        top = np.take_along_axis(stack, np.maximum(cnt - 1, 0)[None], axis=0)[0]
        max_abs = float(max(np.max(np.abs(stack[0][has])),
                            np.max(np.abs(top[has]))))
    else:
        max_abs = float("nan")
    np.clip(stack, -bound, bound, out=stack)
    sd = _sorted_nanstd(stack, cnt)
    return sd.astype(np.float32), n_above, n_below, max_abs


def _composite_rows(read, j, rows, width):
    """Composites of one block of rows. ``read(i)`` returns the masked NDWI
    (rows, width) of record scene i; ``j`` the interval of each scene.
    Returns ``(median, count, sd_mean, extra)`` where ``extra`` holds the
    supplementary bounded-NDWI statistics (:func:`bounded_statistics`)."""
    med = np.empty((N_INTERVALS, rows, width), np.float32)
    cnt = np.empty((N_INTERVALS, rows, width), np.uint16)
    sd_sum = np.zeros((rows, width), np.float64)
    sdb_sum = np.zeros((rows, width), np.float64)
    n_above = np.zeros((rows, width), np.int64)
    n_below = np.zeros((rows, width), np.int64)
    max_abs = float("nan")
    for q in range(1, N_INTERVALS + 1):
        members = np.flatnonzero(j == q)
        stack = np.empty((len(members), rows, width), np.float32)
        for s, i in enumerate(members):
            stack[s] = read(i)
        m_, sd_, c_ = interval_statistics(stack)
        sdb_, a_, b_, mx_ = bounded_statistics(stack, c_)   # clips stack
        del stack
        med[q - 1], cnt[q - 1] = m_, c_
        sd_sum += np.where(c_ > 0, sd_, 0.0)
        sdb_sum += np.where(c_ > 0, sdb_, 0.0)
        n_above += a_
        n_below += b_
        max_abs = _merge_max(max_abs, mx_)
    valid = np.all(cnt >= 1, axis=0)
    sd_mean = np.where(valid, sd_sum / N_INTERVALS, np.nan).astype(np.float32)
    sdb_mean = np.where(valid, sdb_sum / N_INTERVALS, np.nan).astype(np.float32)
    extra = {"sd_mean_bounded": sdb_mean,
             "n_ndwi_above_bound": n_above.astype(np.uint16),
             "n_ndwi_below_minus_bound": n_below.astype(np.uint16),
             "max_abs_ndwi": max_abs}
    return med, cnt, sd_mean, extra


def _merge_max(a, b):
    """NaN-aware maximum of two floats (NaN only if both are NaN)."""
    return float(np.nanmax([a, b])) if np.isfinite([a, b]).any() else float("nan")


def composites_from_arrays(ndwi, clear, j):
    """Per-interval composites from in-memory (T, H, W) arrays.

    ``ndwi`` is the raw NDWI of each record scene, ``clear`` its pixel mask
    (:func:`pixel_valid`), ``j`` the interval of each scene (from
    :func:`interval_assignment`). Returns the same dict as
    :func:`stream_composites`.
    """
    ndwi = np.asarray(ndwi, np.float32)
    clear = np.asarray(clear, bool)
    j = np.asarray(j, int)
    T, H, W = ndwi.shape
    if clear.shape != ndwi.shape or len(j) != T:
        raise ValueError("ndwi, clear and j do not match")
    med, cnt, sd_mean, extra = _composite_rows(
        lambda i: np.where(clear[i], ndwi[i], np.nan).astype(np.float32),
        j, H, W)
    return {"median": med, "count": cnt, "sd_mean": sd_mean, **extra,
            "shape": (H, W)}


def stream_composites(cube_path, time_index, j, row_block=None, check=None,
                      verbose=True, rule="item"):
    """Per-interval composites over the FULL frame, streamed from the cube.

    Implements ITEM's per-interval median NDWI composites (Sagar et al.
    2017, p.158 s3.2.2) with the confidence statistic (p.158; ITEM v2.0
    "Quality assurance"). The cube is read one block of rows at a time and,
    inside a block, one interval at a time, so the working memory is about
    ``max|S_j| x row_block x W x 4`` bytes.

    Parameters
    ----------
    cube_path : str
        netCDF cube with ``B03``, ``B08`` and ``SCL`` on (t, y, x).
    time_index : int array (T_rec,)
        Positions of the record scenes along the cube's time axis.
    j : int array (T_rec,)
        Interval 1..9 of each record scene.
    row_block : int, optional
        Rows per block; default = the cube's y chunk size (or 256).
    check : callable, optional
        ``check(t, r0, r1, ndwi, valid, scl)`` is called with the RAW NDWI,
        the pixel mask and the SCL of cube scene ``t`` for rows ``r0:r1``
        before masking — used to prove the observations equal a previous
        extraction.
    rule : {"item", "load_ard"}
        The pixel mask (:func:`pixel_valid`): "item" for the PRIMARY,
        "load_ard" for the VARIANT.

    Returns
    -------
    dict with ``median`` (9, H, W) float32, ``count`` (9, H, W) uint16
    (valid observations with a defined NDWI), ``sd_mean`` (H, W) float32
    (mean over the 9 intervals of the per-interval NDWI SD, NaN unless every
    interval has an observation), ``shape``, ``row_block`` and
    ``seconds``; plus the SUPPLEMENTARY ``sd_mean_bounded`` (the same mean
    SD with NDWI clipped to [-1, 1]), ``n_ndwi_above_bound`` and
    ``n_ndwi_below_minus_bound`` (H, W) uint16 (valid observations with
    NDWI > 1 and NDWI < -1) and ``max_abs_ndwi`` (float), from
    :func:`bounded_statistics`.
    """
    import xarray as xr

    if rule not in PIXEL_RULES:
        raise ValueError(f"unknown pixel rule {rule!r}")
    t0 = time.time()
    time_index = np.asarray(time_index, int)
    j = np.asarray(j, int)
    if len(time_index) != len(j):
        raise ValueError("time_index and j differ in length")
    ds = xr.open_dataset(cube_path)
    try:
        t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
        for v in ("B03", "B08", "SCL"):
            if ds[v].dims != (t_dim, "y", "x"):
                raise ValueError(f"{v} dims {ds[v].dims} are not (t, y, x)")
        H, W = ds.sizes["y"], ds.sizes["x"]
        if row_block is None:
            ch = ds["B03"].encoding.get("chunksizes")
            row_block = int(ch[1]) if ch else 256
        med = np.empty((N_INTERVALS, H, W), np.float32)
        cnt = np.empty((N_INTERVALS, H, W), np.uint16)
        sd_mean = np.empty((H, W), np.float32)
        extra = {"sd_mean_bounded": np.empty((H, W), np.float32),
                 "n_ndwi_above_bound": np.empty((H, W), np.uint16),
                 "n_ndwi_below_minus_bound": np.empty((H, W), np.uint16),
                 "max_abs_ndwi": float("nan")}
        bands = {v: ds[v] for v in ("B03", "B08", "SCL")}
        for r0 in range(0, H, row_block):
            r1 = min(H, r0 + row_block)

            def read(i, r0=r0, r1=r1):
                t = int(time_index[i])
                sl = {t_dim: t, "y": slice(r0, r1)}
                scl = bands["SCL"].isel(**sl).values
                ndwi, valid = scene_ndwi(bands["B03"].isel(**sl).values,
                                         bands["B08"].isel(**sl).values,
                                         scl, rule=rule)
                if check is not None:
                    check(t, r0, r1, ndwi, valid, scl)
                return np.where(valid, ndwi, np.nan).astype(np.float32)

            m_, c_, s_, x_ = _composite_rows(read, j, r1 - r0, W)
            med[:, r0:r1], cnt[:, r0:r1], sd_mean[r0:r1] = m_, c_, s_
            for k in ("sd_mean_bounded", "n_ndwi_above_bound",
                      "n_ndwi_below_minus_bound"):
                extra[k][r0:r1] = x_[k]
            extra["max_abs_ndwi"] = _merge_max(extra["max_abs_ndwi"],
                                               x_["max_abs_ndwi"])
            if verbose:
                print(f"  [composites] rows {r0}-{r1} of {H} "
                      f"({time.time() - t0:.0f} s)", flush=True)
    finally:
        ds.close()
    return {"median": med, "count": cnt, "sd_mean": sd_mean, **extra,
            "shape": (H, W), "row_block": int(row_block),
            "seconds": time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
#  3. Relative extents
# ─────────────────────────────────────────────────────────────────────────────

def land_flags(median):
    """Land flag per interval: NOT (median NDWI > 0). A median of exactly 0
    counts as land (the frozen tie rule; Sagar et al. 2017, p.159 uses a
    zero threshold without stating the tie)."""
    with np.errstate(invalid="ignore"):
        return ~(np.asarray(median) > NDWI_THRESHOLD)


def relative_extents(median, count):
    """ITEM relative-extents map (Sagar et al. 2017, p.159: "nine layers are
    combined, producing model values ranging from zero to nine").

    R = number of intervals in which the pixel is land; 0 = always water,
    1..8 = exposed from the 0-10 % up to the 70-80 % interval, 9 = land even
    in the 80-100 % composite. A pixel is valid iff every interval has at
    least one valid observation (ITEM v2.0 "-6666" rule).

    Returns ``(R float32 with NaN where invalid, valid bool,
    non_monotone bool)``; ``non_monotone`` flags valid pixels that are water
    in some interval and land in a HIGHER one.
    """
    land = land_flags(median)
    valid = np.all(np.asarray(count) >= 1, axis=0)
    R = land.sum(axis=0).astype(np.float32)
    R[~valid] = np.nan
    non_mono = np.any(land[1:] & ~land[:-1], axis=0) & valid
    return R, valid, non_mono


def gap_fill(R, iterations=GAP_FILL_ITERATIONS):
    """Nearest-valid fill of R within a 2-iteration dilation of the valid
    pixels, NaN beyond — used ONLY for contouring (NIDEM_generation.py
    L175-187)."""
    R = np.asarray(R, np.float32)
    finite = np.isfinite(R)
    if not finite.any():
        raise ValueError("relative-extents map has no valid pixel")
    zone = ndimage.binary_dilation(finite, iterations=iterations)
    idx = ndimage.distance_transform_edt(~finite, return_distances=False,
                                         return_indices=True)
    Rf = R[tuple(idx)]
    Rf[~zone] = np.nan
    return Rf


# ─────────────────────────────────────────────────────────────────────────────
#  4. Waterline contours and the TIN
# ─────────────────────────────────────────────────────────────────────────────

def _contours_at(image, level, mask):
    try:
        return find_contours(image, level, mask=mask)
    except KeyError:          # scikit-image issue 4830 (dea_tools workaround)
        return find_contours(image, level + LEVEL_EPS, mask=mask)


def contours_by_level(R_filled, levels=CONTOUR_LEVELS,
                      min_vertices=MIN_VERTICES):
    """Waterlines as isolines of the gap-filled relative-extents map.

    Marching squares (``skimage.measure.find_contours``; Bishop-Taylor et
    al. 2019, p.117 s2.2) at each level, NaN-safe: an explicit mask of the
    finite pixels, vertices with NaN coordinates dropped, and a retry at
    ``level + 1e-12`` on a KeyError. Lines with fewer than ``min_vertices``
    vertices are dropped (NIDEM_generation.py L624).

    Returns a dict KEYED BY LEVEL, ``{level: [array (n, 2) of (row, col)]}``,
    holding only the levels that produced at least one line. Keying by level
    (not by list position) is what prevents the NIDEM_generation.py
    L244-245 mispairing when a level is empty. Coordinates are pixel
    indices: integer values are pixel centres.
    """
    R_filled = np.asarray(R_filled, float)
    mask = np.isfinite(R_filled)
    out = {}
    for level in levels:
        kept = []
        for line in _contours_at(R_filled, float(level), mask):
            line = line[np.all(np.isfinite(line), axis=1)]
            if len(line) >= min_vertices:
                kept.append(line)
        if kept:
            out[float(level)] = kept
    return out


def level_interval(level):
    """Interval k tagged on contour level k - 0.5 (0.5 -> 1, ..., 8.5 -> 9):
    the waterline between pixels dry in intervals 1..k and pixels wet in
    interval k lies at interval k's median tide height Z_k."""
    return int(round(float(level) + 0.5))


def tin_dem(contours, Z, U, shape):
    """Linear TIN interpolation of the tagged waterline vertices.

    Every vertex of level k - 0.5 carries z = Z_k and u = U_k; all vertices
    are interpolated at every pixel centre with
    ``scipy.interpolate.griddata(method="linear")`` (Qhull Delaunay +
    barycentric weights; Bishop-Taylor et al. 2019, p.118;
    NIDEM_generation.py L256-263), in (row, col) index space WITHOUT a
    half-pixel shift. NaN outside the convex hull of the vertices, and
    everywhere when no triangle exists (< 3 vertices, or all collinear).

    Returns ``(dem, uncertainty, n_vertices)``, the rasters float32 (H, W).
    """
    H, W = shape
    pts, vals = [], []
    for level, lines in contours.items():
        k = level_interval(level)
        for line in lines:
            pts.append(line)
            vals.append(np.tile([Z[k - 1], U[k - 1]], (len(line), 1)))
    if not pts:
        nan = np.full((H, W), np.nan, np.float32)
        return nan, nan.copy(), 0
    pts = np.concatenate(pts)
    vals = np.concatenate(vals)
    if len(pts) < 3:
        nan = np.full((H, W), np.nan, np.float32)
        return nan, nan.copy(), len(pts)
    rr, cc = np.mgrid[0:H, 0:W]
    try:
        out = griddata(pts, vals, (rr.astype(np.float64),
                                   cc.astype(np.float64)), method="linear")
    except QhullError:
        # degenerate vertex set (e.g. all collinear): no triangle, no TIN,
        # the same outcome as fewer than 3 vertices
        nan = np.full((H, W), np.nan, np.float32)
        return nan, nan.copy(), len(pts)
    return (out[..., 0].astype(np.float32), out[..., 1].astype(np.float32),
            len(pts))


# ─────────────────────────────────────────────────────────────────────────────
#  5. The whole product
# ─────────────────────────────────────────────────────────────────────────────

def nidem(composites, Z, U):
    """NIDEM products from the per-interval composites.

    Parameters
    ----------
    composites : dict
        Output of :func:`stream_composites` or
        :func:`composites_from_arrays`.
    Z, U : arrays (9,)
        Interval median tide heights and their SDs (from
        :func:`interval_assignment`).

    Returns
    -------
    dict of full-frame rasters and diagnostics:

    * ``R`` relative extents (float32, NaN where invalid), ``valid``,
      ``non_monotone``, ``conf`` (mean NDWI SD, NaN where invalid);
    * ``R_filled`` (gap-filled R used for contouring);
    * ``dem`` (TIN everywhere inside the vertex hull), ``unfiltered`` (dem
      where R in 1..8, using R before the gap fill), ``filtered``
      (unfiltered where conf <= 0.25), ``uncertainty`` (TIN of U, masked
      like ``unfiltered``);
    * ``contours`` (dict keyed by level), ``contour_stats`` (lines and
      vertices per level), ``empty_levels``, ``n_vertices``, ``seconds``;
    * SUPPLEMENTARY (not the published method): ``conf_bounded`` (mean NDWI
      SD with NDWI clipped to [-1, 1], NaN where invalid) and
      ``filtered_bounded`` (unfiltered where conf_bounded <= 0.25). The
      median composites, R, contours and DEM are the same in both; clipping
      is a contraction, so conf_bounded <= conf and filtered is contained
      in filtered_bounded, which is contained in unfiltered.
    """
    t0 = time.time()
    Z = np.asarray(Z, float)
    U = np.asarray(U, float)
    if Z.shape != (N_INTERVALS,) or U.shape != (N_INTERVALS,):
        raise ValueError("Z and U must have 9 entries")
    R, valid, non_mono = relative_extents(composites["median"],
                                          composites["count"])
    if not valid.any():
        raise NoClearObservationsInIntervalError(
            "no pixel has a valid observation in every interval (ITEM "
            "-6666 everywhere): intervals without any valid observation "
            f"{[int(q) + 1 for q in np.flatnonzero(~np.any(np.asarray(composites['count']) >= 1, axis=(1, 2)))]}")
    conf = np.where(valid, composites["sd_mean"], np.nan).astype(np.float32)
    conf_b = np.where(valid, composites["sd_mean_bounded"],
                      np.nan).astype(np.float32)
    Rf = gap_fill(R)
    contours = contours_by_level(Rf)
    dem, unc, n_vert = tin_dem(contours, Z, U, R.shape)
    with np.errstate(invalid="ignore"):
        intertidal = (R >= 1) & (R <= N_INTERVALS - 1)
        unfiltered = np.where(intertidal, dem, np.nan).astype(np.float32)
        filtered = np.where(intertidal & (conf <= CONFIDENCE_MAX), dem,
                            np.nan).astype(np.float32)
        filtered_b = np.where(intertidal & (conf_b <= CONFIDENCE_MAX), dem,
                              np.nan).astype(np.float32)
    uncertainty = np.where(intertidal, unc, np.nan).astype(np.float32)
    stats = {float(lv): {"lines": len(contours.get(float(lv), [])),
                         "vertices": int(sum(len(x) for x in
                                             contours.get(float(lv), [])))}
             for lv in CONTOUR_LEVELS}
    return {"R": R, "valid": valid, "non_monotone": non_mono, "conf": conf,
            "R_filled": Rf, "dem": dem, "unfiltered": unfiltered,
            "filtered": filtered, "uncertainty": uncertainty,
            "conf_bounded": conf_b, "filtered_bounded": filtered_b,
            "contours": contours, "contour_stats": stats,
            "empty_levels": [float(lv) for lv in CONTOUR_LEVELS
                             if float(lv) not in contours],
            "n_vertices": int(n_vert), "seconds": time.time() - t0}


def nidem_from_arrays(ndwi, clear, h, min_gooddata=None, good_fraction=None):
    """Convenience: the whole method on in-memory (T, H, W) arrays and a
    scene tide series ``h`` (T,). ``clear`` is the run's pixel mask
    (:func:`pixel_valid`). Returns ``(result, assignment)`` with
    ``assignment = (j, edges, Z, U, sizes)``.

    ``min_gooddata`` None (default) is the primary ITEM rule, unchanged. A
    number runs the VARIANT "NIDEM, Sentinel-2 scene rule": scenes whose
    full-frame good fraction ``good_fraction`` (T,) (required; for the
    literal rule :func:`frame_good_fraction` of each scene's SCL, or
    :func:`good_fractions_from_arrays` of :func:`load_ard_good`) is below
    it are dropped FIRST, and the OTR, intervals, Z_j, U_j and composites
    are computed on the retained scenes only; ``j`` then refers to the
    retained scenes (see :func:`s2_scene_rule`).
    """
    if min_gooddata is not None:
        if good_fraction is None:
            raise ValueError("the Sentinel-2 scene rule needs good_fraction "
                             "(the load_ard good share of each scene's frame)")
        good_fraction = np.asarray(good_fraction, float)
        if good_fraction.shape != (np.asarray(h).size,):
            raise ValueError("good_fraction must have one value per scene")
        keep = s2_scene_rule(good_fraction, min_gooddata)
        ndwi = np.asarray(ndwi)[keep]
        clear = np.asarray(clear)[keep]
        h = np.asarray(h, float)[keep]
    j, edges, Z, U, sizes = interval_assignment(h)
    comp = composites_from_arrays(ndwi, clear, j)
    return nidem(comp, Z, U), (j, edges, Z, U, sizes)


def contour_vertices(contours):
    """All contour vertices as ``(rc (N, 2) float, level (N,) float)``."""
    rc, lv = [], []
    for level, lines in contours.items():
        for line in lines:
            rc.append(line)
            lv.append(np.full(len(line), level))
    if not rc:
        return np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(rc), np.concatenate(lv)
