"""V7 — the third method: ITEM/NIDEM waterline-contour interpolation.

Question
--------
How does the established spatial waterline-contour method (ITEM v2.0 /
NIDEM; Sagar et al. 2017, Bishop-Taylor et al. 2019), run on the same
archive as MAREA and Granadeiro but with ITS OWN published scene and pixel
rules, compare with them against the same truth?

Author's principle (binding; revision of 2026-09-29)
----------------------------------------------------
Every method works on the same archive (same cube, Sentinel-2 L2A, same
resolution, 2023-2025, same boundary tide) and is scored on the same pixels
with the same truth and metrics; on top of that, EACH METHOD APPLIES ITS OWN
PUBLISHED RULES (scene selection and pixel masks), replicated faithfully to
its paper or repository. FROZEN['revision_2026_09_29_published_rules']
records the changes and their sources; they were written before any run
under them.

Pre-registered (frozen before any truth is read)
------------------------------------------------
* Method and every constant: :mod:`pyintertidal.waterline_contour`
  (9 intervals of the observed tidal range, median NDWI composites, water
  iff median > 0, R = sum of land flags, contours 0.5..8.5 keyed by level,
  linear TIN in pixel-index space, unfiltered = R in 1..8, filtered =
  unfiltered with mean NDWI SD <= 0.25). No lag, ever. Nothing is tuned.
* Pixel mask (PRIMARY): ITEM's pixel-quality mask mapped to SCL, masked =
  {0 no data (NaN), 1, 3, 8, 9, 10}; valid = {2, 4, 5, 6, 7, 11}
  (Sagar et al. 2017 p.155; ITEM v2.0 processing step 5). NDWI =
  (B03 - B08) / (B03 + B08), NaN where the sum is 0.
* Record per site (ITEM's all-observations rule: every observation of the
  period, "regardless of data completeness or quality", Sagar et al. 2017
  Fig. 3; NIDEM_generation.py interval_uncertainty lists every PQ
  observation):
  - villaviciosa: every 2023-2025 scene of the cube with an overpass time
    and a finite level of h = PyTMDBoundary('EOT20', *aoi.centroid)
    (465 scenes). With ``--sensitivity-marea-record``: MAREA's scored
    97-scene cloud-filtered record (usable dates recovered from the HTML
    output of cell 23 of intertidal_topography_villaviciosa.ipynb, latest
    3-year epoch), a LABELLED SENSITIVITY run, "NIDEM on MAREA's 97-scene
    record".
  - escalda / wadden / ems: every cube scene with an overpass time and a
    finite level of make_boundary(aoi, 'EOT20', n_gauges = 2 / 1 / 1,
    period 2023-2025, exclude = trnz / harl / delf).
* Extraction check: before contouring, at the extraction's keep pixels,
  the streamed SCL is asserted to give exactly the extraction's C
  (SCL in {4, 5, 6, 7}) and the streamed NDWI to equal the extraction's Y
  (raw NDWI, NaN equal to NaN) at EVERY keep pixel, so in particular where
  both masks are valid. The observations the run's pixel mask adds to or
  removes from MAREA's C on keep are counted and reported.
* Pre-flight guard (review round 2, fixed from no truth before any Ems
  composite or score): before streaming, the valid observations per
  interval on keep are counted from a truth-free pre-pass of the record
  scenes under the run's own pixel mask
  (:func:`pyintertidal.waterline_contour.stream_scene_quality`). If fewer
  than 1 % of keep pixels would be valid (>= 1 valid observation in every
  interval), e.g. because the only scene of an extreme interval is
  overcast, the run stops with NoClearObservationsInIntervalError (exit
  code 3), writes nidem_preflight.json and records the product as NOT
  COMPUTABLE (decision (a)): NIDEM has no rule for that case.
* Primary score: the FILTERED DEM sampled on the extraction keep pixels.
  Supplementary: unfiltered, full-frame rows, and (review round 1, rule
  fixed before scoring it) 'NIDEM bounded-conf': the same DEM filtered with
  the confidence recomputed on NDWI clipped to [-1, 1], the NBAR range on
  which 0.25 was defined (L2A reflectances can be negative, so |NDWI| > 1
  occurs). The diagnostics report how much of the primary filter is driven
  by |NDWI| > 1 observations.
  - villaviciosa: RTK development split (rtk.load_rtk only; the sealed
    blocks stay sealed), ev.ols_against_truth unchanged; the published
    rtk_onsite_metrics.csv is first reproduced to 1e-9. Each row names the
    scene record of ITS product (no delay / MAREA: the scored 97 scenes;
    Granadeiro: its own 98-scene rule; NIDEM: the run's record).
  - Dutch sites: the Vaklodingen code of the comparison notebooks' cell 30
    (nearest-cell truth, in-range filter from z_plain +- 0.25 m, 5 x 5
    thinning, median-centred vscore); vaklodingen_metrics.csv and
    vaklodingen_by_band.csv are first reproduced to 1e-9. The old 3-way
    common subset is kept unchanged; a 4-way common subset and the pairwise
    NIDEM-with-MAREA / NIDEM-with-no-delay subsets are added.

Decisions after the Ems pre-flight stop (2026-09-29, made truth-blind by
the orchestrator before any Ems truth was read; recorded in FROZEN)
------------------------------------------------------------------------
(a) PRIMARY stays the ITEM rule at every site. Where its pre-flight stops,
    the primary is reported as NOT COMPUTABLE with the reason: a
    'NIDEM (ITEM rule)' row with n = 0 and a note column in the site's
    metrics CSV, plus nidem_diagnostics.json. The primary is never relaxed.
    ``--record-not-computable`` writes these from the existing
    nidem_preflight.json without touching it; a live pre-flight stop writes
    them too.
(b) LABELLED VARIANT "NIDEM, Sentinel-2 scene rule" (``--s2-scene-rule``),
    run at all four sites on the site's primary record. It replicates the
    Digital Earth Africa Sentinel-2 reference notebook
    (Intertidal_elevation_S2, ``load_ard(..., min_gooddata=0.5)``)
    LITERALLY (revision of 2026-09-29): a scene is dropped when fewer than
    50 % of the pixels of the WHOLE cube frame have an SCL outside
    {1, 3, 8, 9, 10} (no data counted as good, as load_ard), and the pixel
    mask is load_ard's (SCL {1, 3, 8, 9, 10} masked; band no data is NaN;
    raw DN <= 1 in B03 or B08 masked). Everything else is unchanged and
    runs on the retained scenes: OTR, intervals, Z_j, U_j, composites, ITEM
    validity, pre-flight guard, contours, TIN and filters, with the same
    extraction check and the same reproduction asserts. Rows are labelled
    '..., Sentinel-2 scene rule' / 'VARIANT'. If its pre-flight stops, the
    variant is reported NOT COMPUTABLE and not relaxed further.

Superseded (kept in FROZEN['history']): the clear set {4, 5, 6, 7} shared
with MAREA as the NIDEM pixel mask; the variant good fraction on SCL
{4, 5, 6, 7} with no data not good; MAREA's 97-scene record as the
Villaviciosa primary (now the sensitivity) and the variant caveat that
followed from it.

Run (from the repo root)
------------------------
    python -X utf8 -m experiments.v7_waterline_contour villaviciosa
    python -X utf8 -m experiments.v7_waterline_contour villaviciosa --sensitivity-marea-record
    python -X utf8 -m experiments.v7_waterline_contour escalda
    python -X utf8 -m experiments.v7_waterline_contour wadden
    python -X utf8 -m experiments.v7_waterline_contour ems
    python -X utf8 -m experiments.v7_waterline_contour <site> --preflight-only
    python -X utf8 -m experiments.v7_waterline_contour <site> --s2-scene-rule
    python -X utf8 -m experiments.v7_waterline_contour ems --record-not-computable

Outputs (existing result files of other methods are never modified)
--------------------------------------------------------------------
villaviciosa: products_marea/nidem/ (sensitivity: .../sensitivity_marea_record/;
    variant: .../s2_scene_rule/)
    nidem_preflight.json, nidem.npz, nidem_*.tif, item_*.tif,
    nidem_diagnostics.json, rtk_metrics_with_nidem.csv (variant:
    rtk_metrics_with_nidem_s2_scene_rule.csv)
Dutch: products_<site>/nidem_<tag>/ (variant: nidem_<tag>_s2_scene_rule/)
    with nidem_preflight.json, nidem.npz, GeoTIFFs, diagnostics, and
    products_<site>/comparison_<tag>/vaklodingen_metrics_with_nidem.csv,
    vaklodingen_by_band_with_nidem.csv (variant: ..._with_nidem_s2_scene_rule.csv)
"""
import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pyintertidal import waterline_contour as wc  # noqa: E402
from experiments.validation_grid import dutch_cube, dutch_products, thin_mask, thin_px  # noqa: E402

TIME_EXTENT = ("2023-01-01", "2025-12-31")
RTK_CSV = "data/villaviciosa_rtk_gnss.csv"
RTK_GRID = "products_villaviciosa/hsr_eot20_2023-2025.tif"
VVA_NOTEBOOK = "intertidal_topography_villaviciosa.ipynb"

SITES = {
    "villaviciosa": dict(cube="ndwi_cube_villaviciosa_grande_10y.nc",
                         extract="products_marea/marea_extract.npz",
                         overpass="products_marea/overpass_times.json",
                         out="products_marea/nidem"),
    "escalda": dict(cube=dutch_cube("escalda"), n_gauges=2,
                    judge=["trnz"], tag="eot20_g2"),
    "wadden": dict(cube=dutch_cube("wadden"), n_gauges=1,
                   judge=["harl"], tag="eot20_g1"),
    "ems": dict(cube=dutch_cube("ems"), n_gauges=1,
                judge=["delf"], tag="eot20_g1"),
}
for _s in ("escalda", "wadden", "ems"):
    _c, _p = SITES[_s], dutch_products(_s)
    _c["extract"] = f"{_p}/marea_extract.npz"
    _c["overpass"] = f"{_p}/overpass_times.json"
    _c["out"] = f"{_p}/nidem_{_c['tag']}"
    _c["comparison"] = f"{_p}/comparison_{_c['tag']}"

TOL = 1e-9
#: scenes of the scored Villaviciosa no-delay / MAREA record (asserted by
#: build_record on the sensitivity run "NIDEM on MAREA's 97-scene record")
VVA_SCORED_SCENES = 97
#: scenes of the Villaviciosa primary record (ITEM all-observations rule;
#: every 2023-2025 cube scene with an overpass time and a finite level),
#: asserted by build_record
VVA_ALL_SCENES = 465
#: the Villaviciosa Granadeiro product scored in rtk_onsite_metrics.csv
#: (external_products.npz 'Granadeiro' = this file's 'final' on keep)
VVA_GRANADEIRO = "granadeiro_products_2023-2025.npz"

#: decision (b): the labelled variant
VARIANT_TAG = "s2_scene_rule"
VARIANT_NAME = "NIDEM, Sentinel-2 scene rule"
VARIANT_SUFFIX = ", Sentinel-2 scene rule"
VARIANT_LEAD = "VARIANT (Sentinel-2 scene rule)"
#: decision (a): the primary's label where it is not computable
PRIMARY_NOT_COMPUTABLE_NAME = "NIDEM (ITEM rule)"
#: revision of 2026-09-29 (C): the Villaviciosa sensitivity on MAREA's record
SENSITIVITY_TAG = "sensitivity_marea_record"
SENSITIVITY_SUFFIX = f" on MAREA's {VVA_SCORED_SCENES}-scene record"
SENSITIVITY_LEAD = f"SENSITIVITY (NIDEM on MAREA's {VVA_SCORED_SCENES}-scene record)"
#: the run modes
MODES = ("primary", "sensitivity", "variant")


def run_mode(sensitivity=False, variant=False):
    """'primary', 'sensitivity' (Villaviciosa, MAREA's 97-scene record) or
    'variant' (the Sentinel-2 scene rule)."""
    if sensitivity and variant:
        raise ValueError("a run is either the sensitivity or the variant")
    return "sensitivity" if sensitivity else "variant" if variant else "primary"


def pixel_rule(mode):
    """The pixel mask of a run: the variant uses load_ard's, every other
    run ITEM's pixel-quality mask."""
    return "load_ard" if mode == "variant" else "item"


def nidem_name(base, mode):
    """Product label of a NIDEM row: unchanged for the primary, suffixed
    ', Sentinel-2 scene rule' for the variant and " on MAREA's 97-scene
    record" for the Villaviciosa sensitivity. A bool ``mode`` is read as
    the variant flag (True) or the primary (False)."""
    if mode is True:
        mode = "variant"
    elif mode is False:
        mode = "primary"
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    return base + {"primary": "", "variant": VARIANT_SUFFIX,
                   "sensitivity": SENSITIVITY_SUFFIX}[mode]


def output_paths(site, sensitivity=False, variant=False):
    """Output directory and metrics CSV path(s) of a run. With both flags
    off they are exactly the paths of the primary runs; the variant and the
    sensitivity write to their own directories and CSV files."""
    cfg = SITES[site]
    out = cfg["out"]
    if sensitivity:
        if site != "villaviciosa":
            raise ValueError("the sensitivity run is Villaviciosa only")
        out = os.path.join(out, SENSITIVITY_TAG)
    suffix = f"_{VARIANT_TAG}" if variant else ""
    if variant:
        out = (os.path.join(out, VARIANT_TAG) if site == "villaviciosa"
               else out + suffix)
    if site == "villaviciosa":
        csv = {"rtk": os.path.join(out, f"rtk_metrics_with_nidem{suffix}.csv")}
    else:
        d = cfg["comparison"]
        csv = {"metrics": f"{d}/vaklodingen_metrics_with_nidem{suffix}.csv",
               "by_band": f"{d}/vaklodingen_by_band_with_nidem{suffix}.csv"}
    return out, csv


def log(msg):
    print(msg, flush=True)


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_memory_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 20
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  record
# ─────────────────────────────────────────────────────────────────────────────

def villaviciosa_usable_dates(nb_path=VVA_NOTEBOOK):
    """The notebook's cloud-filtered usable dates (transition-zone cloud
    <= 10 %), recovered from the HTML table printed by cell 23 (no file
    stores them). 'ref' and 'keep' rows are usable, 'drop' rows are not."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    html = "".join(nb["cells"][23]["outputs"][0]["data"]["text/html"])
    rows = re.findall(r"<td class='(ref|keep|drop)'>[^<]*</td>"
                      r"<td>(\d{4}-\d\d-\d\d)</td>", html)
    usable = sorted(d for c, d in rows if c in ("ref", "keep"))
    if len(rows) != 1379 or len(usable) != 305:
        raise RuntimeError(f"cell 23 recovery changed: {len(rows)} rows, "
                           f"{len(usable)} usable (expected 1379 / 305)")
    return usable


def build_record(site, cfg, ext_dates, marea_record=False):
    """Scenes, times and boundary levels of the run's record.

    ITEM's all-observations rule at every site (revision of 2026-09-29, C):
    every cube scene of 2023-2025 with an overpass time and a finite
    boundary level. ``marea_record`` (Villaviciosa only) gives instead
    MAREA's scored 97-scene cloud-filtered record, for the labelled
    sensitivity "NIDEM on MAREA's 97-scene record"."""
    import pandas as pd
    import pyintertidal as pit

    if marea_record and site != "villaviciosa":
        raise ValueError("MAREA's cloud-filtered record exists at Villaviciosa only")
    overpass = {k: pd.Timestamp(v).to_pydatetime()
                for k, v in json.load(open(cfg["overpass"])).items()}
    have = np.array([d in overpass for d in ext_dates])
    if site == "villaviciosa":
        from pyintertidal.boundary import PyTMDBoundary
        if marea_record:
            epoch = pit.epochs(villaviciosa_usable_dates(), epoch_years=3)[-1]
            if epoch["label"] != "2023-2025":
                raise RuntimeError(f"latest epoch is {epoch['label']}")
            sel = np.isin(ext_dates, list(epoch["dates"])) & have
            rule = (f"{SENSITIVITY_LEAD}: 2023-2025 epoch of the notebook's "
                    "cloud-filtered usable dates (cell 23), with overpass time "
                    "and finite level (the scored no-delay / MAREA record)")
        else:
            sel = (have & (ext_dates >= TIME_EXTENT[0])
                   & (ext_dates <= TIME_EXTENT[1]))
            rule = ("PRIMARY: all 2023-2025 cube scenes with an overpass time "
                    "and a finite level (ITEM all-observations rule)")
        lat_c, lon_c = pit.sites.get("villaviciosa").centroid
        boundary = PyTMDBoundary("EOT20", lat_c, lon_c, directory="tide_models")
        bname = f"PyTMDBoundary EOT20 at aoi.centroid ({lat_c:.5f}, {lon_c:.5f})"
    else:
        from pyintertidal import net
        from pyintertidal.boundary import make_boundary
        net.use_system_certificates()
        boundary = make_boundary(pit.sites.get(site), tide_model="EOT20",
                                 n_gauges=cfg["n_gauges"], period=TIME_EXTENT,
                                 exclude=cfg["judge"])
        bname = getattr(boundary, "name", str(boundary))
        sel = have
        rule = ("PRIMARY: all cube scenes with an overpass time and a finite "
                "boundary level (ITEM all-observations rule; also the scored "
                "no-delay / MAREA record)")
    idx = np.flatnonzero(sel)
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [overpass[d] for d in ext_dates[idx]])).tz_localize(None)
    if not t_real.is_monotonic_increasing:
        raise RuntimeError("overpass times are not chronological")
    h = np.asarray(boundary.levels(t_real), float)
    ok = np.isfinite(h)
    rec = dict(time_index=idx[ok], dates=ext_dates[idx[ok]], t_real=t_real[ok],
               h=h[ok], boundary=bname, boundary_obj=boundary, rule=rule,
               n_without_time=int((~have).sum()),
               n_nonfinite_level=int((~ok).sum()))
    if site == "villaviciosa":
        want = VVA_SCORED_SCENES if marea_record else VVA_ALL_SCENES
        if len(rec["h"]) != want:
            raise RuntimeError(f"Villaviciosa record has {len(rec['h'])} scenes, "
                               f"expected {want}")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
#  extraction check
# ─────────────────────────────────────────────────────────────────────────────

def _read_exact(f, n):
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            raise EOFError("truncated npy member")
        parts.append(b)
        got += len(b)
    return b"".join(parts)


def npz_rows(npz_path, key, rows, tmpdir, mem_limit=300e6):
    """Rows ``rows`` (sorted, unique) of a 2-D member of an npz, streamed
    from the zip without loading the whole member. Large selections go to
    a memory-mapped file in ``tmpdir``."""
    rows = np.asarray(rows, int)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be sorted and unique")
    with zipfile.ZipFile(npz_path) as zf, zf.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        if fortran or len(shape) != 2:
            raise ValueError(f"{key}: unexpected layout {shape} {fortran}")
        nrow = int(shape[1]) * dtype.itemsize
        if len(rows) * nrow > mem_limit:
            path = os.path.join(tmpdir, f"{key}_rows.npy")
            out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype,
                                            shape=(len(rows), int(shape[1])))
        else:
            out = np.empty((len(rows), int(shape[1])), dtype)
        pos = {int(t): q for q, t in enumerate(rows)}
        for t in range(int(rows.max()) + 1):
            buf = _read_exact(f, nrow)
            q = pos.get(t)
            if q is not None:
                out[q] = np.frombuffer(buf, dtype)
    return out


class ExtractionCheck:
    """Asserts, scene by scene and block by block, that at the keep pixels
    the streamed SCL gives exactly the extraction's C (SCL in {4, 5, 6, 7})
    and the streamed raw NDWI equals the extraction's Y (Y is the raw NDWI
    of every scene at every keep pixel; NaN equal to NaN). It compares
    every keep pixel, so in particular every pixel where both the run's
    pixel mask and MAREA's C are valid, and counts the observations
    (mask valid and finite NDWI) that the run's mask adds to or removes
    from MAREA's (C and finite NDWI)."""

    def __init__(self, Y, C, time_index, keep, W):
        self.Y, self.C = Y, C
        self.row_of_t = {int(t): q for q, t in enumerate(time_index)}
        self.keep, self.W = keep, W
        self.kr, self.kc = keep // W, keep % W
        self.n_c = self.n_y = self.n_clear_nan = 0
        self.n_both_valid_compared = 0
        self.n_obs = self.n_marea_obs = self.n_both = 0
        self.n_added = self.n_removed = 0

    def __call__(self, t, r0, r1, ndwi, valid, scl):
        from pyintertidal import waterline_contour as wc_

        a, b = np.searchsorted(self.keep, [r0 * self.W, r1 * self.W])
        if a == b:
            return
        q = self.row_of_t[int(t)]
        rr, cc = self.kr[a:b] - r0, self.kc[a:b]
        c_x = np.asarray(self.C[q, a:b], bool)
        if not np.array_equal(np.isin(scl[rr, cc], wc_.CLEAR), c_x):
            raise AssertionError(f"SCL in {wc_.CLEAR} differs from the extraction "
                                 f"C at scene t={t}, rows {r0}-{r1}")
        y_s = ndwi[rr, cc]
        y_x = np.asarray(self.Y[q, a:b])
        if not np.array_equal(y_s, y_x, equal_nan=True):
            raise AssertionError(f"NDWI differs from the extraction Y at "
                                 f"scene t={t}, rows {r0}-{r1}")
        v = np.asarray(valid[rr, cc], bool)
        fin = np.isfinite(y_s)
        obs, marea = v & fin, c_x & fin
        self.n_c += b - a
        self.n_y += b - a
        self.n_clear_nan += int((c_x & ~fin).sum())
        self.n_both_valid_compared += int((v & c_x).sum())
        self.n_obs += int(obs.sum())
        self.n_marea_obs += int(marea.sum())
        self.n_both += int((obs & marea).sum())
        self.n_added += int((obs & ~marea).sum())
        self.n_removed += int((marea & ~obs).sum())

    def summary(self):
        return {"scene_px_compared (SCL->C and NDWI, every keep px)": int(self.n_c),
                "ndwi_values_compared_where_both_masks_valid": int(self.n_both_valid_compared),
                "marea_clear_but_nan_ndwi": int(self.n_clear_nan),
                "observations_on_keep (pixel mask & finite NDWI)": int(self.n_obs),
                "marea_observations_on_keep (C & finite NDWI)": int(self.n_marea_obs),
                "observations_in_both": int(self.n_both),
                "observations_added_vs_marea_C": int(self.n_added),
                "observations_removed_vs_marea_C": int(self.n_removed),
                "result": "identical (SCL->C, raw NDWI)"}


# ─────────────────────────────────────────────────────────────────────────────
#  outputs
# ─────────────────────────────────────────────────────────────────────────────

def cube_grid(cube_path):
    import pyproj
    import xarray as xr

    ds = xr.open_dataset(cube_path)
    try:
        t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
        dates = np.array([str(np.datetime64(v, "D")) for v in ds[t_dim].values])
        xs, ys = ds["x"].values.astype(float), ds["y"].values.astype(float)
        wkt = ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref")
    finally:
        ds.close()
    from rasterio.transform import Affine
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    transform = Affine(dx, 0.0, float(xs[0]) - dx / 2, 0.0, dy,
                       float(ys[0]) - dy / 2)
    return dict(dates=dates, xs=xs, ys=ys, H=len(ys), W=len(xs), wkt=wkt,
                crs=pyproj.CRS.from_wkt(wkt), transform=transform)


def write_rasters(out_dir, res, grid):
    from rasterio.crs import CRS
    from pyintertidal.raster import write_geotiff

    crs = CRS.from_wkt(grid["wkt"])
    tr = grid["transform"]
    paths = []
    for name, key in (("nidem_filtered.tif", "filtered"),
                      ("nidem_unfiltered.tif", "unfiltered"),
                      ("nidem_uncertainty.tif", "uncertainty"),
                      ("item_confidence.tif", "conf"),
                      # SUPPLEMENTARY: confidence on NDWI clipped to [-1, 1]
                      ("nidem_filtered_bounded_conf.tif", "filtered_bounded"),
                      ("item_confidence_bounded.tif", "conf_bounded")):
        paths.append(write_geotiff(os.path.join(out_dir, name),
                                   res[key].astype(np.float32), tr, crs))
    R = np.where(np.isfinite(res["R"]), res["R"], 255).astype(np.uint8)
    paths.append(write_geotiff(os.path.join(out_dir, "item_relative_extents.tif"),
                               R, tr, crs, dtype="uint8", nodata=255))
    return paths


def at_keep(raster, keep):
    return np.asarray(raster).ravel()[keep]


# ─────────────────────────────────────────────────────────────────────────────
#  scoring: Villaviciosa RTK
# ─────────────────────────────────────────────────────────────────────────────

RTK_COLS = ("n", "rmse_m", "slope", "intercept", "r", "bias_m")
#: the supplementary product of review round 1 (confidence on NDWI clipped
#: to [-1, 1]; see pyintertidal.waterline_contour.bounded_statistics)
BOUNDED = "NIDEM bounded-conf"
BOUNDED_ROLE = ("supplementary (review R1): filtered with conf on NDWI clipped "
                "to [-1, 1], on keep")


def rtk_records(mode, n_nidem_scenes, keep, shape, granadeiro_raster,
                n_before_scene_rule=None):
    """The scene record behind each product of the RTK table (review round 2:
    labelled per product, not per run). Checks that the scored Granadeiro
    raster is the 'final' map of its own npz, so the label is about it.
    ``mode`` is 'primary', 'sensitivity' or 'variant'; for the variant,
    ``n_before_scene_rule`` is the primary record's size."""
    g = np.load(VVA_GRANADEIRO)
    n_gran = int(np.asarray(g["use"], bool).sum())
    assert n_gran == len(g["h_scene"]), (n_gran, len(g["h_scene"]))
    assert np.array_equal(g["keep"], keep), "Granadeiro keep != extraction keep"
    zg = g["final"].astype(float)
    zg[zg == -9999] = np.nan
    ref = np.full(int(np.prod(shape)), np.nan, np.float32)
    ref[keep] = zg
    assert np.array_equal(ref.reshape(shape), granadeiro_raster, equal_nan=True), \
        "external_products.npz Granadeiro is not granadeiro_products 'final'"
    scored = (f"scored {VVA_SCORED_SCENES}-scene 2023-2025 record (cloud-filtered "
              "usable dates of notebook cell 23; EOT20 at aoi.centroid)")
    gran = (f"own {n_gran}-scene 2023-2025 record (sota_granadeiro 'scene' rule: "
            "scene cloud <= 10 %, nodata <= 2 %, overpass time; Eq. 1 cosine "
            "heights from EOT20 at the bbox centre)")
    allrec = (f"all {VVA_ALL_SCENES} 2023-2025 cube scenes with an overpass time "
              "and a finite level (ITEM all-observations rule; EOT20 at "
              "aoi.centroid)")
    if mode == "sensitivity":
        assert n_nidem_scenes == VVA_SCORED_SCENES
        nid = (f"{SENSITIVITY_LEAD}: {scored}; same scenes and levels as no "
               "delay / MAREA; ITEM pixel-quality mask")
    elif mode == "variant":
        assert n_before_scene_rule == VVA_ALL_SCENES
        assert 0 < n_nidem_scenes <= VVA_ALL_SCENES
        nid = (f"{VARIANT_LEAD}: the {allrec} minus "
               f"{VVA_ALL_SCENES - n_nidem_scenes} scene(s) with a full-frame "
               f"load_ard good fraction < {wc.S2_SCENE_MIN_GOODDATA} -> "
               f"{n_nidem_scenes} scenes, same levels; load_ard pixel mask")
    elif mode == "primary":
        assert n_nidem_scenes == VVA_ALL_SCENES
        nid = f"PRIMARY: {allrec}; ITEM pixel-quality mask"
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return ({"no delay": scored, "MAREA": scored, "Granadeiro": gran, "NIDEM": nid},
            {"no delay": VVA_SCORED_SCENES, "MAREA": VVA_SCORED_SCENES,
             "Granadeiro": n_gran, "NIDEM": int(n_nidem_scenes)})


def mode_leads(mode):
    """(lead, supplementary lead) role labels of a run mode."""
    return {"primary": ("PRIMARY", "supplementary"),
            "sensitivity": (SENSITIVITY_LEAD, f"{SENSITIVITY_LEAD} supplementary"),
            "variant": (VARIANT_LEAD, f"{VARIANT_LEAD} supplementary")}[mode]


def not_computable_name(mode):
    """Product label of a NOT COMPUTABLE row."""
    return {"primary": PRIMARY_NOT_COMPUTABLE_NAME, "variant": VARIANT_NAME,
            "sensitivity": nidem_name("NIDEM", "sensitivity")}[mode]


def score_rtk(path, res, keep, shape, mode, n_nidem_scenes,
              not_computable=None, n_before_scene_rule=None):
    """RTK table. ``res`` None with ``not_computable`` (the reason) writes the
    reproduced published rows plus one NOT COMPUTABLE NIDEM row (n = 0) and
    a note column (decision (a)). ``mode`` is 'primary', 'sensitivity' or
    'variant' and sets the NIDEM labels and records."""
    import pandas as pd
    from pyintertidal import elevation_validation as ev
    from pyintertidal import rtk

    H, W = shape
    dev = rtk.load_rtk(RTK_CSV)
    r_, c_, zt = dev["row"], dev["col"], dev["elev"]
    E = np.load("products_marea/external_products.npz")
    old = {"no delay": E["no_delay"], "MAREA": E["MAREA"],
           "Granadeiro": E["Granadeiro"]}
    for ras in old.values():
        assert ras.shape == (H, W)
    records, n_scenes = rtk_records(mode, n_nidem_scenes, keep, (H, W),
                                    old["Granadeiro"],
                                    n_before_scene_rule=n_before_scene_rule)
    # NIDEM roles are labelled per run mode; each product's own scene record
    # is in the 'record' column (every method applies its own published
    # scene rule, so the records differ by design)
    lead, sup = mode_leads(mode)
    N = lambda b: nidem_name(b, mode)  # noqa: E731

    # 1. reproduce the published table
    ref = pd.read_csv("products_marea/rtk_onsite_metrics.csv").set_index("product")
    worst = 0.0
    for name, ras in old.items():
        f = ev.ols_against_truth(zt, ras[r_, c_])
        assert f["n"] == int(ref.loc[name, "n"]), (name, f["n"])
        for k in RTK_COLS[1:]:
            worst = max(worst, abs(f[k] - float(ref.loc[name, k])))
    assert worst <= TOL, f"rtk_onsite_metrics.csv not reproduced ({worst})"
    log(f"  reproduced rtk_onsite_metrics.csv: max |diff| {worst:.2e}")

    samp = {name: np.asarray(ras[r_, c_], float) for name, ras in old.items()}
    in_keep = np.isin(r_ * W + c_, keep)

    def row(product, subset, role, sel):
        v = np.where(sel, samp[product], np.nan)
        f = ev.ols_against_truth(zt, v)
        base = "NIDEM" if product.startswith("NIDEM") else product
        out = {"product": product, "subset": subset, "role": role,
               "record": records[base], "record_scenes": n_scenes[base]}
        if f is None:
            out.update({k: np.nan for k in RTK_COLS})
            out["n"] = int(np.sum(np.isfinite(v) & np.isfinite(zt)))
        else:
            out.update({k: f[k] for k in RTK_COLS})
        return out

    allpts = np.ones(len(zt), bool)
    rows = [row("no delay", "all", "reproduced (published)", allpts),
            row("MAREA", "all", "reproduced (published)", allpts),
            row("Granadeiro", "all", "reproduced (published)", allpts)]

    if res is None:
        # decision (a) / a stopped variant: reproduced rows + NOT COMPUTABLE
        name = not_computable_name(mode)
        rows.append({"product": name, "subset": "all",
                     "role": f"{lead}: NOT COMPUTABLE",
                     "record": records["NIDEM"],
                     "record_scenes": n_scenes["NIDEM"],
                     **{k: np.nan for k in RTK_COLS}, "n": 0})
        df = pd.DataFrame(rows)
        df["note"] = ["" for _ in range(len(df) - 1)] + [not_computable]
        df.to_csv(path, index=False)
        log(df[["product", "subset", "role", "n"]].to_string(index=False))
        return path, {"dev_points": int(len(zt)),
                      "dev_points_reserved_hidden": int(dev["n_reserved_hidden"]),
                      "reproduction_max_abs_diff": worst,
                      "product_records": records,
                      "not_computable": not_computable}

    def keep_raster(key):
        r = np.full(H * W, np.nan, np.float32)
        r[keep] = at_keep(res[key], keep)
        return r.reshape(H, W)

    samp[N("NIDEM")] = np.asarray(keep_raster("filtered")[r_, c_], float)
    samp[N("NIDEM unfiltered")] = np.asarray(keep_raster("unfiltered")[r_, c_], float)
    samp[N(BOUNDED)] = np.asarray(keep_raster("filtered_bounded")[r_, c_], float)
    samp[N("NIDEM full frame")] = np.asarray(res["filtered"][r_, c_], float)
    samp[N("NIDEM unfiltered full frame")] = np.asarray(res["unfiltered"][r_, c_],
                                                        float)
    fin = {k: np.isfinite(v) for k, v in samp.items()}
    nid = N("NIDEM")
    common4 = fin["no delay"] & fin["MAREA"] & fin["Granadeiro"] & fin[nid]
    pair_m = fin[nid] & fin["MAREA"]
    pair_n = fin[nid] & fin["no delay"]
    bounded_lead = "" if mode == "primary" else f"{lead} "
    rows += [row(nid, "all", f"{lead}: filtered, on keep", allpts),
             row(N("NIDEM unfiltered"), "all", f"{sup}: unfiltered, on keep", allpts),
             row(N(BOUNDED), "all", bounded_lead + BOUNDED_ROLE, allpts),
             row(N("NIDEM full frame"), "all", f"{sup}: filtered, full frame", allpts),
             row(N("NIDEM unfiltered full frame"), "all",
                 f"{sup}: unfiltered, full frame", allpts)]
    for p in ("no delay", "MAREA", "Granadeiro", nid):
        rows.append(row(p, f"common4 (no delay & MAREA & Granadeiro & {nid})",
                        "common subset", common4))
    for p in (nid, "MAREA"):
        rows.append(row(p, f"pair ({nid} & MAREA)", "pairwise subset", pair_m))
    for p in (nid, "no delay"):
        rows.append(row(p, f"pair ({nid} & no delay)", "pairwise subset", pair_n))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log(df[["product", "subset", "n", "rmse_m", "slope", "r", "bias_m"]]
        .to_string(index=False))
    counts = {"dev_points": int(len(zt)),
              "dev_points_reserved_hidden": int(dev["n_reserved_hidden"]),
              "dev_points_in_keep": int(in_keep.sum()),
              "finite_points": {k: int(v.sum()) for k, v in fin.items()},
              "common4_points": int(common4.sum()),
              "pair_nidem_marea_points": int(pair_m.sum()),
              "pair_nidem_nodelay_points": int(pair_n.sum()),
              "reproduction_max_abs_diff": worst,
              "product_records": records}
    return path, counts


# ─────────────────────────────────────────────────────────────────────────────
#  scoring: Dutch Vaklodingen
# ─────────────────────────────────────────────────────────────────────────────

def vscore(p, t):
    """Cell 30 of the comparison notebooks, unchanged."""
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 30:
        return None
    pc, tc = p[m] - np.median(p[m]), t[m] - np.median(t[m])
    return dict(rmse_m=float(np.sqrt(np.mean((pc - tc) ** 2))),
                slope=float(np.polyfit(tc, pc, 1)[0]),
                r=float(np.corrcoef(tc, pc)[0, 1]),
                bias_m=float(np.median(p[m] - t[m])), n=int(m.sum()))


def score_vaklodingen(site, cfg, res, keep, grid, csv_paths, variant=False,
                      not_computable=None):
    """Vaklodingen tables. ``res`` None with ``not_computable`` (the reason)
    writes the reproduced published rows plus NOT COMPUTABLE NIDEM rows
    (n = 0) and a note column (decision (a))."""
    import pandas as pd
    import pyproj
    from experiments import v1_vaklodingen_validate as v1

    d = cfg["comparison"]
    P = np.load(f"{d}/products.npz")
    assert np.array_equal(P["keep"], keep), "products.npz keep != extraction keep"
    H, W = (int(v) for v in P["shape"])
    assert (H, W) == (grid["H"], grid["W"])
    rows_k, cols_k = keep // W, keep % W
    to_rd = pyproj.Transformer.from_crs(grid["crs"], "EPSG:28992", always_xy=True)
    xq, yq = to_rd.transform(grid["xs"][cols_k], grid["ys"][rows_k])
    cells = v1.truth_grid(site)
    truth_raw, _ = v1.sample_truth(cells, xq, yq)
    z_plain = P["z_plain"].astype(float)
    lo_, hi_ = float(np.nanmin(z_plain)), float(np.nanmax(z_plain))
    in_range = (truth_raw >= lo_ - 0.25) & (truth_raw <= hi_ + 0.25)
    truth_v = np.where(in_range, truth_raw, np.nan)
    px_m = abs(float(grid["xs"][1] - grid["xs"][0]))
    thin = thin_mask(rows_k, cols_k, px_m)
    thin_label = f"thinned {thin_px(px_m)}x{thin_px(px_m)}"   # 5x5 at 20 m, 10x10 at 10 m
    vprods = {"no delay": z_plain, "MAREA": P["z_marea"].astype(float)}
    if "z_gran" in P.files:           # the notebooks' superseded Granadeiro (20 m runs only)
        vprods["Granadeiro"] = P["z_gran"].astype(float)
    base = list(vprods)
    ref_label = f"common{len(base)} ({' & '.join(base)}; published)"

    # 1. reproduce the published wide table
    vcommon = np.isfinite(truth_v)
    for v_ in vprods.values():
        vcommon &= np.isfinite(v_)
    wide = []
    for name_, v_ in vprods.items():
        a_ = vscore(v_, truth_v) or {}
        t_ = vscore(np.where(thin, v_, np.nan), truth_v) or {}
        c_ = vscore(np.where(vcommon, v_, np.nan), truth_v) or {}
        wide.append({"product": name_, "n": a_.get("n"), "RMSE (m)": a_.get("rmse_m"),
                     "slope": a_.get("slope"), "r": a_.get("r"), "bias (m)": a_.get("bias_m"),
                     "thinned RMSE (m)": t_.get("rmse_m"), "thinned slope": t_.get("slope"),
                     "common RMSE (m)": c_.get("rmse_m"), "common slope": c_.get("slope"),
                     "common r": c_.get("r")})
    mine = pd.DataFrame(wide).set_index("product")
    ref = pd.read_csv(f"{d}/vaklodingen_metrics.csv").set_index("product")
    a = mine.loc[ref.index, ref.columns].to_numpy(float)
    b = ref.to_numpy(float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    worst = float(np.nanmax(np.abs(a - b)))
    assert worst <= TOL, f"vaklodingen_metrics.csv not reproduced ({worst})"
    log(f"  reproduced vaklodingen_metrics.csv: max |diff| {worst:.2e}; "
        f"old 3-way common subset {int(vcommon.sum()):,} px")

    # by band, reproduced
    band_of = P["band"].astype(int)
    bb = pd.read_csv(f"{d}/vaklodingen_by_band.csv")
    worst_b = 0.0
    band_rows = []
    for name_, v_ in vprods.items():
        for _, rr in bb[bb["product"] == name_].iterrows():
            k = int(rr["band"])
            s_ = vscore(np.where(band_of == k, v_, np.nan), truth_v) or {}
            for kk in ("n", "rmse_m", "slope", "r", "bias_m"):
                x, y = s_.get(kk, np.nan), float(rr[kk])
                if np.isnan(x) != np.isnan(y):
                    raise AssertionError(f"by-band NaN mismatch {name_} {k} {kk}")
                if not np.isnan(x):
                    worst_b = max(worst_b, abs(float(x) - y))
            band_rows.append({"product": name_, "band": k, "s_km": float(rr["s_km"]),
                              "tau_applied_min": float(rr["tau_applied_min"]),
                              **{kk: s_.get(kk, np.nan) for kk in
                                 ("n", "rmse_m", "slope", "r", "bias_m")}})
    assert worst_b <= TOL, f"vaklodingen_by_band.csv not reproduced ({worst_b})"
    log(f"  reproduced vaklodingen_by_band.csv: max |diff| {worst_b:.2e}")

    # 2. NIDEM
    N = lambda b: nidem_name(b, variant)  # noqa: E731
    lead, sup = ((VARIANT_LEAD, f"{VARIANT_LEAD} supplementary") if variant
                 else ("PRIMARY", "supplementary"))
    ones = np.ones(len(keep), bool)
    tv = np.isfinite(truth_v)

    if res is None:
        # decision (a) / a stopped variant: reproduced rows + NOT COMPUTABLE
        name = VARIANT_NAME if variant else PRIMARY_NOT_COMPUTABLE_NAME
        rows = []
        for p in base:
            for subset, sel in (("all", ones), (thin_label, thin), (ref_label, vcommon)):
                s_ = vscore(np.where(sel, vprods[p], np.nan), truth_v) or {}
                rows.append({"product": p, "subset": subset,
                             "role": "reproduced (published)",
                             **{kk: s_.get(kk, np.nan) for kk in
                                ("n", "rmse_m", "slope", "r", "bias_m")},
                             "note": ""})
        rows.append({"product": name, "subset": "all",
                     "role": f"{lead}: NOT COMPUTABLE", "n": 0,
                     "rmse_m": np.nan, "slope": np.nan, "r": np.nan,
                     "bias_m": np.nan, "note": not_computable})
        df = pd.DataFrame(rows)
        path = csv_paths["metrics"]
        df.to_csv(path, index=False)
        log(df[["product", "subset", "role", "n", "rmse_m"]].round(4)
            .to_string(index=False))
        for r_ in band_rows:
            r_["note"] = ""
        bands = bb[bb["product"] == list(vprods)[0]][["band", "s_km",
                                                      "tau_applied_min"]]
        for _, rr in bands.iterrows():
            band_rows.append({"product": name, "band": int(rr["band"]),
                              "s_km": float(rr["s_km"]),
                              "tau_applied_min": float(rr["tau_applied_min"]),
                              "n": 0, "rmse_m": np.nan, "slope": np.nan,
                              "r": np.nan, "bias_m": np.nan,
                              "note": not_computable})
        bpath = csv_paths["by_band"]
        pd.DataFrame(band_rows).to_csv(bpath, index=False)
        return [path, bpath], {
            "truth_px_on_keep": int(np.isfinite(truth_raw).sum()),
            "truth_px_excluded_out_of_range": int((np.isfinite(truth_raw) & ~in_range).sum()),
            "in_range_from_z_plain": [lo_ - 0.25, hi_ + 0.25],
            "common3_px": int(vcommon.sum()),
            "reproduction_max_abs_diff": worst,
            "reproduction_by_band_max_abs_diff": worst_b,
            "not_computable": not_computable}

    nid_name = N("NIDEM")
    nid = {nid_name: at_keep(res["filtered"], keep).astype(float),
           N("NIDEM unfiltered"): at_keep(res["unfiltered"], keep).astype(float),
           N(BOUNDED): at_keep(res["filtered_bounded"], keep).astype(float)}
    allp = {**vprods, **nid}
    fin = {k: np.isfinite(v) for k, v in allp.items()}
    common4 = tv & np.logical_and.reduce([fin[p] for p in base]) & fin[nid_name]
    pair_m = tv & fin[nid_name] & fin["MAREA"]
    pair_n = tv & fin[nid_name] & fin["no delay"]

    def lrow(product, subset, role, sel, values=None, truth=None):
        v = allp[product] if values is None else values
        t = truth_v if truth is None else truth
        s_ = vscore(np.where(sel, v, np.nan), t) or {}
        return {"product": product, "subset": subset, "role": role,
                "n": s_.get("n", int(np.sum(sel & np.isfinite(v) & np.isfinite(t)))),
                "rmse_m": s_.get("rmse_m", np.nan), "slope": s_.get("slope", np.nan),
                "r": s_.get("r", np.nan), "bias_m": s_.get("bias_m", np.nan)}

    rows = []
    for p in base:
        rows.append(lrow(p, "all", "reproduced (published)", ones))
        rows.append(lrow(p, thin_label, "reproduced (published)", thin))
        rows.append(lrow(p, ref_label,
                         "reproduced (published)", vcommon))
    bounded_role = f"{VARIANT_LEAD} {BOUNDED_ROLE}" if variant else BOUNDED_ROLE
    rows.append(lrow(nid_name, "all", f"{lead}: filtered, on keep", ones))
    rows.append(lrow(nid_name, thin_label, f"{lead}: filtered, on keep", thin))
    rows.append(lrow(N("NIDEM unfiltered"), "all", f"{sup}: unfiltered, on keep", ones))
    rows.append(lrow(N("NIDEM unfiltered"), thin_label,
                     f"{sup}: unfiltered, on keep", thin))
    rows.append(lrow(N(BOUNDED), "all", bounded_role, ones))
    rows.append(lrow(N(BOUNDED), thin_label, bounded_role, thin))
    for p in (*base, nid_name):
        rows.append(lrow(p, f"common{len(base) + 1} ({' & '.join(base)} & {nid_name})",
                         "common subset", common4))
    for p in (nid_name, "MAREA"):
        rows.append(lrow(p, f"pair ({nid_name} & MAREA)", "pairwise subset", pair_m))
    for p in (nid_name, "no delay"):
        rows.append(lrow(p, f"pair ({nid_name} & no delay)", "pairwise subset", pair_n))

    # full frame (supplementary): truth sampled at every finite NIDEM pixel
    ff_counts = {}
    for key, label in (("filtered", N("NIDEM full frame")),
                       ("unfiltered", N("NIDEM unfiltered full frame"))):
        flat = np.flatnonzero(np.isfinite(res[key].ravel()))
        fr, fc = flat // W, flat % W
        xf, yf = to_rd.transform(grid["xs"][fc], grid["ys"][fr])
        tf, _ = v1.sample_truth(cells, xf, yf)
        tf = np.where((tf >= lo_ - 0.25) & (tf <= hi_ + 0.25), tf, np.nan)
        vals = res[key].ravel()[flat].astype(float)
        rows.append({**lrow(label, "all (full frame)",
                            f"{sup}: {key}, full frame",
                            np.ones(len(flat), bool), values=vals, truth=tf),
                     "product": label})
        ff_counts[label] = {"pixels": int(len(flat)),
                            "with_truth_in_range": int(np.isfinite(tf).sum())}
    df = pd.DataFrame(rows)
    path = csv_paths["metrics"]
    df.to_csv(path, index=False)
    log(df[["product", "subset", "n", "rmse_m", "slope", "r", "bias_m"]]
        .round(4).to_string(index=False))

    # by band with NIDEM
    bands = bb[bb["product"] == list(vprods)[0]][["band", "s_km", "tau_applied_min"]]
    for name_ in (nid_name, N("NIDEM unfiltered"), N(BOUNDED)):
        for _, rr in bands.iterrows():
            k = int(rr["band"])
            s_ = vscore(np.where(band_of == k, allp[name_], np.nan), truth_v) or {}
            band_rows.append({"product": name_, "band": k, "s_km": float(rr["s_km"]),
                              "tau_applied_min": float(rr["tau_applied_min"]),
                              **{kk: s_.get(kk, np.nan) for kk in
                                 ("n", "rmse_m", "slope", "r", "bias_m")}})
    bpath = csv_paths["by_band"]
    pd.DataFrame(band_rows).to_csv(bpath, index=False)
    counts = {"truth_px_on_keep": int(np.isfinite(truth_raw).sum()),
              "truth_px_excluded_out_of_range": int((np.isfinite(truth_raw) & ~in_range).sum()),
              "in_range_from_z_plain": [lo_ - 0.25, hi_ + 0.25],
              "finite_with_truth": {k: int((v & tv).sum()) for k, v in fin.items()},
              "common3_px": int(vcommon.sum()), "common4_px": int(common4.sum()),
              "pair_nidem_marea_px": int(pair_m.sum()),
              "pair_nidem_nodelay_px": int(pair_n.sum()),
              "full_frame": ff_counts,
              "reproduction_max_abs_diff": worst,
              "reproduction_by_band_max_abs_diff": worst_b}
    return [path, bpath], counts


# ─────────────────────────────────────────────────────────────────────────────
#  diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnostics(res, comp, keep, Z, U, sizes, edges, h):
    q = [0, 5, 25, 50, 75, 95, 100]
    cnt_k = comp["count"].reshape(wc.N_INTERVALS, -1)[:, keep]
    R_k = at_keep(res["R"], keep)
    valid_k = at_keep(res["valid"], keep)
    conf_k = at_keep(res["conf"], keep)
    P = len(keep)
    with np.errstate(invalid="ignore"):
        inter_k = (R_k >= 1) & (R_k <= 8)
        conf_ok_k = inter_k & (conf_k <= wc.CONFIDENCE_MAX)
        inter_all = (res["R"] >= 1) & (res["R"] <= 8)
    frac = lambda m: float(np.sum(m) / P)  # noqa: E731
    lines = {f"{lv:.1f}": v for lv, v in res["contour_stats"].items()}
    return {
        "n_scenes": int(len(h)),
        "LOT_m": float(np.min(h)), "HOT_m": float(np.max(h)),
        "OTR_m": float(np.max(h) - np.min(h)),
        "edges_m (e_0..e_10; interval 9 = (e_8, e_10])": [float(v) for v in edges],
        "interval_sizes |S_j|": [int(v) for v in sizes],
        "Z_j_m (median tide of interval)": [float(v) for v in Z],
        "U_j_m (SD ddof=1 of interval tides)": [None if not np.isfinite(v) else float(v) for v in U],
        "contour_spacing_m (Z_{j+1}-Z_j)": [float(v) for v in np.diff(Z)],
        "N_j_valid_obs_on_keep_percentiles": {
            f"p{p}": [float(v) for v in np.percentile(cnt_k, p, axis=1)] for p in q},
        "min_over_intervals_N_j_on_keep_percentiles": {
            f"p{p}": float(np.percentile(cnt_k.min(axis=0), p)) for p in q},
        "share_keep_with_N_j_ge_3_all_intervals (Fitton rule, info only)":
            frac(np.all(cnt_k >= 3, axis=0)),
        "non_monotone_share_of_valid_keep":
            float(np.sum(at_keep(res["non_monotone"], keep)) / max(1, valid_k.sum())),
        "non_monotone_share_of_valid_full_frame":
            float(res["non_monotone"].sum() / max(1, res["valid"].sum())),
        "coverage_on_keep": {
            "keep_px": int(P),
            "valid": frac(valid_k),
            "R_in_1_8": frac(inter_k),
            "R_in_1_8_and_conf_le_0.25": frac(conf_ok_k),
            "unfiltered_finite": frac(np.isfinite(at_keep(res["unfiltered"], keep))),
            "filtered_finite": frac(np.isfinite(at_keep(res["filtered"], keep))),
            "R_eq_0": frac(valid_k & (R_k == 0)),
            "R_eq_9": frac(valid_k & (R_k == 9)),
            "R_eq_0_of_valid": float(np.sum(R_k == 0) / max(1, valid_k.sum())),
            "R_eq_9_of_valid": float(np.sum(R_k == 9) / max(1, valid_k.sum())),
            "conf_pass_rate_within_R_1_8": float(conf_ok_k.sum() / max(1, inter_k.sum())),
        },
        "R_histogram_on_keep (0..9)": [int(np.sum(R_k == v)) for v in range(10)],
        "full_frame": {
            "px": int(res["R"].size), "valid": int(res["valid"].sum()),
            "R_in_1_8": int(inter_all.sum()),
            "unfiltered_finite": int(np.isfinite(res["unfiltered"]).sum()),
            "filtered_finite": int(np.isfinite(res["filtered"]).sum()),
        },
        "contours_per_level": lines,
        "empty_levels": res["empty_levels"],
        "n_vertices": res["n_vertices"],
        "ndwi_out_of_range": ndwi_out_of_range(res, comp, keep, cnt_k,
                                               inter_k, conf_k),
    }


def ndwi_out_of_range(res, comp, keep, cnt_k, inter_k, conf_k):
    """How much of the 0.25 confidence filter is driven by |NDWI| > 1 (L2A
    negative reflectances), and the SUPPLEMENTARY bounded-NDWI confidence.
    Truth-free."""
    b = wc.NDWI_BOUND
    above_k = at_keep(comp["n_ndwi_above_bound"], keep).astype(np.int64)
    below_k = at_keep(comp["n_ndwi_below_minus_bound"], keep).astype(np.int64)
    n_obs_k = int(cnt_k.astype(np.int64).sum())
    has_out = (above_k + below_k) > 0
    conf_b_k = at_keep(res["conf_bounded"], keep)
    with np.errstate(invalid="ignore"):
        fail_k = inter_k & ~(conf_k <= wc.CONFIDENCE_MAX)
        pass_b_k = inter_k & (conf_b_k <= wc.CONFIDENCE_MAX)
        pass_k = inter_k & (conf_k <= wc.CONFIDENCE_MAX)
    n_inter = max(1, int(inter_k.sum()))
    f_all = np.isfinite(res["filtered"])
    fb_all = np.isfinite(res["filtered_bounded"])
    n_obs_ff = int(comp["count"].astype(np.int64).sum())
    above_ff = int(comp["n_ndwi_above_bound"].astype(np.int64).sum())
    below_ff = int(comp["n_ndwi_below_minus_bound"].astype(np.int64).sum())
    return {
        "note": ("The 0.25 threshold was defined on NBAR NDWI in [-1, 1] "
                 "(Bishop-Taylor et al. 2019, p.118). These cubes hold "
                 "offset-harmonised L2A reflectances, which can be negative; "
                 "|NDWI| > 1 iff B03 and B08 have opposite signs, unbounded as "
                 "B03 + B08 -> 0. PRIMARY 'filtered' keeps the published "
                 "statistic on raw NDWI. SUPPLEMENTARY 'filtered_bounded' "
                 "recomputes only the confidence with NDWI clipped to [-1, 1]; "
                 "medians, R and the DEM are identical. NDWI < -1 observations "
                 "(B03 > 0 > B08, |B08| > B03, typically dark water) still enter "
                 "the raw medians with a land sign in both products (and in "
                 "MAREA's Y)."),
        "bound": b,
        "valid_finite_obs_on_keep": n_obs_k,
        "obs_ndwi_gt_+1_on_keep": int(above_k.sum()),
        "obs_ndwi_lt_-1_on_keep": int(below_k.sum()),
        "share_obs_abs_ndwi_gt_1_on_keep": float((above_k.sum() + below_k.sum())
                                                 / max(1, n_obs_k)),
        "share_keep_px_with_ge1_abs_ndwi_gt_1_obs": float(has_out.mean()),
        "conf_failing_R_1_8_keep_px": int(fail_k.sum()),
        "conf_failing_R_1_8_keep_px_with_abs_ndwi_gt_1_obs": int((fail_k & has_out).sum()),
        "share_of_conf_failing_R_1_8_keep_px_with_abs_ndwi_gt_1_obs":
            float((fail_k & has_out).sum() / max(1, fail_k.sum())),
        "conf_pass_rate_within_R_1_8_on_keep (raw NDWI, PRIMARY)": float(pass_k.sum() / n_inter),
        "conf_pass_rate_within_R_1_8_on_keep (NDWI clipped to [-1,1], SUPPLEMENTARY)":
            float(pass_b_k.sum() / n_inter),
        "filtered_bounded_finite_share_of_keep":
            float(np.isfinite(at_keep(res["filtered_bounded"], keep)).mean()),
        "full_frame": {
            "valid_finite_obs": n_obs_ff,
            "obs_ndwi_gt_+1": above_ff, "obs_ndwi_lt_-1": below_ff,
            "share_obs_abs_ndwi_gt_1": float((above_ff + below_ff) / max(1, n_obs_ff)),
            "max_abs_ndwi": comp["max_abs_ndwi"],
            "filtered_bounded_finite": int(fb_all.sum()),
            "filtered_px_not_in_filtered_bounded (nesting check, expect 0)":
                int((f_all & ~fb_all).sum()),
        },
    }


#: key of the pre-flight coverage in nidem_preflight.json (revision of
#: 2026-09-29: counted from the truth-free cube pre-pass under the run's own
#: pixel mask, no longer from the extraction's C)
COV_KEY = "coverage_on_keep (pre-pass: pixel mask & finite NDWI)"


def pixel_mask_record(mode):
    """The run's pixel mask, as recorded in the pre-flight and diagnostics."""
    rule = pixel_rule(mode)
    if rule == "item":
        return {"rule": "item",
                "masked_scl": list(wc.ITEM_PQ_MASKED_SCL),
                "valid_scl": [c for c in wc.SCL_CODES
                              if c not in wc.ITEM_PQ_MASKED_SCL],
                "nan_scl": "masked (SCL 0 no data; contiguity)",
                "source": FROZEN["pixel_quality_mask_item"]["sources"]}
    return {"rule": "load_ard",
            "masked_scl": list(wc.DEAFRICA_S2_BAD_SCL),
            "nan_scl": "NOT masked (SCL no data is not a load_ard category)",
            "band_rule": (f"B03 and B08 > {wc.LOAD_ARD_MIN_VALUE} (raw DN > "
                          f"{wc.LOAD_ARD_MIN_RAW_DN_EXCLUSIVE}); a NaN band "
                          "gives a NaN NDWI"),
            "source": FROZEN["variant_s2_scene_rule"]["source_load_ard"]}


def scene_quality_summary(q):
    """Truth-free summary of the pre-pass (all scenes it covered)."""
    pc = [0, 5, 25, 50, 75, 95, 100]
    return {
        "rule": q["rule"], "frame_px": int(q["px"]), "keep_px": int(q["P"]),
        "n_scenes": int(len(q["load_ard_good"])),
        "observations_on_keep (pixel mask & finite NDWI)": int(q["n_obs_keep"].sum()),
        "marea_observations_on_keep (C & finite NDWI)": int(q["n_marea_obs_keep"].sum()),
        "observations_in_both": int(q["n_both_keep"].sum()),
        "observations_added_vs_marea_C": int(q["n_added_keep"].sum()),
        "observations_removed_vs_marea_C": int(q["n_removed_keep"].sum()),
        "frame_scl_zero_px_total (expected 0: no data is NaN)": int(q["scl_zero_px"].sum()),
        f"frame_px_with_B03_or_B08_le_{wc.LOAD_ARD_MIN_VALUE} (raw DN <= 1)":
            int(q["band_le_load_ard_min_px"].sum()),
        f"frame_px_with_B03_or_B08_lt_{wc.LOAD_ARD_MIN_VALUE} (raw DN <= 0; expected 0)":
            int(q["band_below_offset_px"].sum()),
        "frame_load_ard_good_fraction_percentiles": {
            f"p{p}": float(np.percentile(q["load_ard_good"], p)) for p in pc},
        "frame_item_valid_share_percentiles (info)": {
            f"p{p}": float(np.percentile(q["item_valid"], p)) for p in pc},
        "frame_no_data_share_percentiles": {
            f"p{p}": float(np.percentile(q["no_data"], p)) for p in pc},
        "seconds": float(q["seconds"]),
    }


def preflight_report(site, rec, record_label, j, edges, Z, U, sizes, cov,
                     per_scene, totals, n_keep, mode="primary", scene_rule=None,
                     quality=None):
    """Truth-free pre-flight record (review round 2): the interval coverage on
    keep predicted, before any composite, from the pre-pass of the record
    scenes under the run's own pixel mask, and the scenes that decide it.
    Written for every run to nidem_preflight.json; when the guard stops a
    run it is the report. For the variant, ``scene_rule`` (the scene drop
    record) is included; ``quality`` is :func:`scene_quality_summary` of
    the record scenes."""
    variant = mode == "variant"
    h = rec["h"]
    share = per_scene / max(1, n_keep)
    bnd = rec["boundary_obj"]
    members = getattr(bnd, "providers", None)

    def scenes(sel):
        sel = np.asarray(sel, int)
        out = [{"date": str(rec["dates"][q]), "t_real": str(rec["t_real"][q]),
                "cube_time_index": int(rec["time_index"][q]),
                "interval": int(j[q]), "h_m": float(h[q]),
                "valid_finite_share_on_keep": float(share[q])} for q in sel]
        if members and len(sel):
            # PyTMDBoundary.levels returns its values in TIME order whatever
            # the input order, so query in time order and map back
            srt = np.argsort(rec["t_real"][sel].values, kind="stable")
            t_sel = rec["t_real"][sel[srt]]
            levs = []
            for p in members:
                lev = np.empty(len(sel))
                lev[srt] = np.asarray(p.levels(t_sel), float)
                levs.append(lev)
                for d, v in zip(out, lev):
                    d.setdefault("boundary_member_levels_m", {})[
                        getattr(p, "name", type(p).__name__)] = float(v)
            # the consensus is the members' mean (EnsembleBoundary.levels)
            assert np.allclose(np.nanmean(levs, axis=0), h[sel], atol=1e-9), \
                "boundary member levels do not reproduce the record levels"
        return out

    order = np.argsort(h, kind="stable")
    flagged = {str(q): scenes(np.flatnonzero(j == q))
               for q in cov["intervals_below_floor"]}
    stopped = not cov["passed"]
    out = {
        "experiment": "v7_waterline_contour", "stage": "preflight",
        "site": site, "record": record_label, "record_rule": rec["rule"],
        "boundary": rec["boundary"], "truth_used": "none", "mode": mode,
        "status": ("STOPPED: NIDEM not computable for this record under the "
                   "ITEM rules (NoClearObservationsInIntervalError); nothing "
                   "was streamed or scored" if stopped else "passed"),
        "guard": FROZEN["preflight_guard"],
        "pixel_mask": pixel_mask_record(mode),
    }
    if variant and stopped:
        out["status"] = (
            f"STOPPED: '{VARIANT_NAME}' not computable for this record (the "
            "ITEM validity rule on the scenes retained by the Sentinel-2 scene "
            "rule; NoClearObservationsInIntervalError); nothing was streamed "
            "or scored")
    if variant:
        out["nidem_variant"] = VARIANT_NAME
        out["s2_scene_rule"] = scene_rule
    out |= {
        "n_scenes": int(len(h)),
        "LOT_m": float(np.min(h)), "HOT_m": float(np.max(h)),
        "OTR_m": float(np.max(h) - np.min(h)),
        "edges_m (e_0..e_10; interval 9 = (e_8, e_10])": [float(v) for v in edges],
        "interval_sizes |S_j|": [int(v) for v in sizes],
        "Z_j_m": [float(v) for v in Z],
        "U_j_m (ddof=1)": [None if not np.isfinite(v) else float(v) for v in U],
        COV_KEY: cov,
        "valid_finite_obs_total_per_interval_on_keep": [int(v) for v in totals],
        "scene_quality_prepass (record scenes)": quality,
        "lowest_5_scenes": scenes(order[:5]),
        "highest_5_scenes": scenes(order[::-1][:5]),
        "scenes_of_intervals_below_floor": flagged,
    }
    if stopped and variant:
        out["decision"] = (
            "Decision (b), 2026-09-29: the variant is NOT relaxed further. It "
            "is reported as NOT COMPUTABLE at this site under the Sentinel-2 "
            "scene rule, with the reason above.")
    elif stopped:
        out["decision"] = (
            "Decision (a), 2026-09-29 (truth-blind): the ITEM rule is never "
            "relaxed; the product is reported as NOT COMPUTABLE for this "
            "record, with the reason above.")
    return out


FROZEN = {
    "reference":"Sagar et al. 2017 RSE 195:153-169; Bishop-Taylor et al. 2019 ECSS 223:115-128",
    "authors_principle": (
        "binding (revision of 2026-09-29): every method works on the same "
        "archive (same cube, Sentinel-2 L2A, same resolution, 2023-2025, same "
        "boundary tide) and is scored on the same pixels with the same truth "
        "and metrics; on top of that, EACH METHOD APPLIES ITS OWN PUBLISHED "
        "RULES (scene selection and pixel masks), replicated faithfully to its "
        "paper or repository."),
    "record": (
        "ITEM all-observations rule at every site (revision of 2026-09-29, C): "
        "every 2023-2025 cube scene with an overpass time and a finite "
        "boundary level; no scene-level cloud filter. Villaviciosa: 465 scenes "
        "(EOT20 at aoi.centroid); MAREA's 97-scene cloud-filtered record is "
        "the labelled SENSITIVITY 'NIDEM on MAREA's 97-scene record'. Dutch "
        "sites: every cube scene (unchanged). Sources: Sagar et al. 2017 "
        "Fig. 3 (all tiles are used 'regardless of data completeness or "
        "quality'); ITEM v2.0 processing step 3 ('All observations within the "
        "nominated cell/polygon and time period is attributed with a tidal "
        "height'); NIDEM_generation.py interval_uncertainty (L767-850: every "
        "dataset of ls5/ls7/ls8_pq_albers in the period, grouped by solar "
        "day, no cloud-cover filter)."),
    "intervals": "10 uniform intervals of OTR = max(h)-min(h) over all record scenes, top two merged -> 9",
    "bin_edges": "lowest closed [e0,e1], others right-closed (e_{k-1},e_k]",
    "Z_j": "median of h over all scenes of interval j (cloudy included)",
    "U_j": "std of h over interval j, ddof=1",
    "composite": ("per-pixel median NDWI over the valid observations of the "
                  "run's pixel mask (PRIMARY: ITEM PQ mask mapped to SCL, see "
                  "pixel_quality_mask_item; VARIANT: load_ard's), NDWI = "
                  "(B03-B08)/(B03+B08), NaN where B03+B08==0"),
    "pixel_quality_mask_item": {
        "status": ("PRIMARY pixel mask since the revision of 2026-09-29 (A); "
                   "replaces MAREA's clear set SCL {4,5,6,7}. Written before "
                   "any run under it."),
        "masked_scl": [0, 1, 3, 8, 9, 10],
        "valid_scl": [2, 4, 5, 6, 7, 11],
        "mapping": {
            "cloud": "SCL 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus",
            "cloud shadow": "SCL 3 cloud shadows",
            "band saturation": "SCL 1 saturated or defective",
            "contiguity (all bands have data)": ("SCL 0 no data = NaN SCL in our "
                                                 "cubes (openEO fill); a NaN band "
                                                 "gives a NaN NDWI, never an observation"),
            "not masked by ITEM": ("SCL 2 dark area pixels / topographic cast "
                                   "shadows (ESA renamed the class at processing "
                                   "baseline 04.00; not cloud shadow), 4 "
                                   "vegetation, 5 bare soil, 6 water, 7 "
                                   "unclassified, 11 snow or ice"),
        },
        "sources": [
            ("Sagar et al. 2017 p.155: 'Each tile was then processed using the "
             "PQ bitmasks to exclude pixels flagged as cloud, cloud shadow, or "
             "saturated in any of the bands.' (also p.161: 'the pixel quality "
             "process masks clouds, cloud shadow or saturated pixels')"),
            ("ITEM v2.0.0 product description, DEA Knowledge Hub, processing "
             "step 5: 'Each tile observation is masked for pixel quality based "
             "on the DEA PQA layer to exclude pixels flagged for cloud, band "
             "saturation and contiguity.' (retrieved 2026-09-29)"),
            ("NIDEM_generation.py (GeoscienceAustralia/nidem, master): applies "
             "no pixel mask of its own (it consumes the ITEM rasters); it opens "
             "the ls*_pq_albers products only to list observation times "
             "(interval_uncertainty, L767-850). The ITEM v2 generation code is "
             "not public (searched 2026-09-29), so the flag list comes from the "
             "two documents above; they agree with the mapping requested."),
        ],
        "ndwi": "unchanged: (B03-B08)/(B03+B08), NaN where the sum is 0",
        "verification": ("mapping verified against the sources above; no "
                         "source shows otherwise. The union of the two lists "
                         "(cloud, cloud shadow, saturation, contiguity) is used."),
    },
    "valid": ">=1 valid obs (pixel mask & finite NDWI) in every interval",
    "water": "median NDWI > 0 (tie = land)",
    "R": "sum of 9 land flags (0..9)",
    "conf": "mean over intervals of NDWI std (ddof=0)",
    "gap_fill": "binary_dilation(valid, 2 iterations), nearest-valid fill, NaN beyond (contouring only)",
    "contours": "skimage find_contours at 0.5..8.5, mask + NaN-vertex drop + 1e-12 KeyError retry, >=2 vertices, keyed by level; level k-0.5 tagged Z_k, U_k",
    "tin": "scipy griddata linear over all vertices in (row, col) pixel-index space, no +0.5 px shift",
    "unfiltered": "DEM where R (before gap fill) in 1..8",
    "filtered": "unfiltered where conf <= 0.25 (PRIMARY)",
    "lag": "none",
    "supplementary_filtered_bounded": (
        "review round 1 (2026-09-29), rule fixed before scoring it: conf "
        "recomputed with every valid NDWI clipped to [-1, 1] (the NBAR range "
        "on which 0.25 was defined); filtered_bounded = unfiltered where this "
        "conf <= 0.25. Medians, R, contours and DEM unchanged. SUPPLEMENTARY "
        "only; the primary stays 'filtered'."),
    "extraction_check": (
        "revision of 2026-09-29 (A): at every keep pixel of every record "
        "scene, SCL in {4,5,6,7} is asserted equal to the extraction's C and "
        "the raw NDWI to the extraction's Y (NaN equal to NaN), so the NDWI "
        "is compared in particular wherever both masks are valid; the "
        "observations the run's mask adds to / removes from MAREA's C on keep "
        "are counted (pre-pass and streaming counts asserted equal)."),
    "preflight_guard": (
        "review round 2 (2026-09-29), fixed from no truth before any Ems "
        "composite or score: before the cube is streamed, N_j(p) = valid "
        "observations with a finite NDWI per interval are counted on keep "
        "(since the revision of 2026-09-29: from a truth-free pre-pass of the "
        "record scenes under the run's own pixel mask, "
        "waterline_contour.stream_scene_quality, with the pre-pass SCL "
        "asserted to reproduce the extraction's C). If the expected "
        "valid share on keep (N_j >= 1 in all 9 intervals) is below "
        "MIN_VALID_SHARE = 0.01, the run stops with "
        "NoClearObservationsInIntervalError, writes nidem_preflight.json and "
        "scores nothing (the analogue of the empty-interval stop; NIDEM has "
        "no rule for either). It drops no scene and redefines no interval, "
        "so it changes no product where it passes; after streaming, the "
        "counts are asserted equal to the composites' count on keep."),
    "decision_a_primary_not_computable": (
        "orchestrator decision (a), 2026-09-29, truth-blind (made before any "
        "Ems truth was read): the PRIMARY stays the ITEM rule at every site "
        "and is never relaxed. Where its pre-flight stops (first seen at Ems "
        "under the superseded clear mask: the only scene of interval 1, "
        "2024-12-10, was clear on 27 of 477,745 keep px), the primary is "
        "reported NOT COMPUTABLE: product 'NIDEM (ITEM rule)', n = 0, with the "
        "reason in a note column of the site's metrics CSV and in "
        "nidem_diagnostics.json. Since the revision of 2026-09-29 the Ems "
        "pre-flight is re-run under the ITEM pixel mask and decides again."),
    "variant_s2_scene_rule": {
        "name": VARIANT_NAME,
        "status": ("LABELLED VARIANT, not the primary. Orchestrator decision "
                   "(b), 2026-09-29, truth-blind; recorded here before the "
                   "variant was run at any site and before any Ems truth was "
                   "read. Made LITERAL load_ard by the revision of 2026-09-29 "
                   "(B), recorded before any run under it."),
        "rule": ("drop a record scene iff its full-frame good fraction < "
                 "S2_SCENE_MIN_GOODDATA; good fraction = share of ALL H x W "
                 "pixels of the cube frame whose SCL is NOT in {1,3,8,9,10} "
                 "(literal load_ard: no data, NaN SCL here and SCL 0 there, "
                 "counts as GOOD; 2 and 11 are good; the bands and the NDWI "
                 "are not looked at); keep iff >= the threshold. Pixel mask of "
                 "the variant = load_ard's: SCL {1,3,8,9,10} masked, SCL no data "
                 "NOT masked, band no data NaN, and raw DN <= 1 in B03 or B08 "
                 "masked (cube value <= -999). Then OTR, intervals, Z_j, U_j, "
                 "per-pixel composites, ITEM validity, pre-flight guard, gap "
                 "fill, contours, TIN, unfiltered/filtered/bounded products "
                 "exactly as the primary, on the retained scenes and on the "
                 "primary record of the site (Villaviciosa: the 465-scene "
                 "record). If its pre-flight stops, it is reported NOT "
                 "COMPUTABLE and not relaxed further."),
        "constant": "pyintertidal.waterline_contour.S2_SCENE_MIN_GOODDATA = 0.5",
        "min_gooddata": 0.5,
        "comparison": "keep iff good fraction >= 0.5 (load_ard: keep = (data_perc >= min_gooddata))",
        "extent": ("the whole loaded frame (load_ard divides the good-pixel count "
                   "by pq_mask.shape[1] * pq_mask.shape[2], every pixel of the "
                   "loaded area); ours = the full cube frame, not keep"),
        "source_notebook": (
            "Digital Earth Africa sandbox notebook 'Modelling intertidal "
            "elevation using tidal data and Sentinel-2', "
            "https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/"
            "Real_world_examples/Intertidal_elevation_S2.html ; source "
            "https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/"
            "blob/main/Real_world_examples/Intertidal_elevation_S2.ipynb "
            "(main @ 0ecfbebd7c48ad8da334272fbd16eef38aa43e11, 2026-09-02)"),
        "source_load_ard": (
            "https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/"
            "blob/main/Tools/deafrica_tools/datahandling.py (load_ard; main @ "
            "a309e7c4b09a98d6e9776a568d3943d546d80c82, 2024-12-02)"),
        "retrieved": "2026-09-29",
        "quoted_notebook_cell_15": ("s2_ds = load_ard(dc=dc, products=['s2_l2a_c1'], "
                                    "output_crs=output_crs, min_gooddata=0.5, "
                                    "group_by='solar_day', dask_chunks={}, **query)"),
        "quoted_notebook_query": "'measurements': ['red', 'green', 'blue', 'nir']",
        "quoted_load_ard": [
            "categories_to_mask_s2=[\"cloud high probability\", \"cloud medium "
            "probability\", \"thin cirrus\", \"cloud shadows\", \"saturated or defective\"]",
            "pq_mask = odc.algo.enum_to_bool(mask=ds[fmask_band], categories=categories_to_mask_s2)",
            "data_perc = (~pq_mask).sum(axis=[1, 2], dtype=\"int32\") / "
            "(pq_mask.shape[1] * pq_mask.shape[2])",
            "keep = (data_perc >= min_gooddata).persist()",
            "ds = ds.sel(time=keep)",
            "# Remove sentinel-2 pixels valued 1 (scene edges, terrain shadow)",
            "valid_data_mask = (ds_data > 1).to_array(dim=\"band\").all(dim=\"band\")",
            "ds_data =  odc.algo.keep_good_only(ds_data, where=valid_data_mask)",
            "ds_data = odc.algo.erase_bad(ds_data, where=mask)   # mask = pq_mask",
            "ds[band] = ds[band] - 1000   # s2_l2a_c1 rescale (after masking)",
        ],
        "class_mapping": (
            "load_ard's default S2 bad classes = SCL {1 saturated or defective, "
            "3 cloud shadows, 8 cloud medium probability, 9 cloud high "
            "probability, 10 thin cirrus} (DE Africa s2_l2a_c1 flags_definition, "
            "SCL nodata 0; enum_to_bool is np.isin). Good there = everything "
            "else, including 0 'no data' (so pixels outside the swath count as "
            "good in the scene rule), 2 dark area pixels, 11 snow or ice. Our "
            "cubes store SCL no data as NaN, which np.isin does not match, so "
            "NaN counts as good exactly like SCL 0 in load_ard."),
        "pixel_mask_check_against_source": (
            "Confirmed against the quoted lines: the per-pixel mask is "
            "erase_bad(pq_mask) (SCL {1,3,8,9,10}) plus band nodata -> NaN "
            "(to_float), AS REQUESTED, and in addition load_ard @ a309e7c "
            "L544-547 keeps only pixels with raw DN > 1 in EVERY loaded band "
            "(the notebook loads red, green, blue, nir). Following the source, "
            "the variant applies that rule to the two bands our cubes hold "
            "(B03 green, B08 nir): raw DN = value + 1000 (BOA_ADD_OFFSET -1000, "
            "DN 0 stored as NaN), so DN > 1 <=> value > -999. Red and blue are "
            "not in the cubes; their DN <= 1 pixels cannot be masked "
            "(documented deviation)."),
    },
    "revision_2026_09_29_published_rules": {
        "recorded": "2026-09-29, before any run under these rules; truth-blind",
        "A_pixel_mask": ("PRIMARY pixel mask = ITEM's PQ mask mapped to SCL "
                         "(pixel_quality_mask_item); replaces MAREA's clear set "
                         "{4,5,6,7}. The extraction check compares NDWI where "
                         "both masks are valid (in fact at every keep px) and "
                         "counts added / removed observations vs MAREA's C."),
        "B_variant": ("VARIANT = literal load_ard: scene good fraction = SCL not "
                      "in {1,3,8,9,10} over all frame pixels, no data counted "
                      "good; pixel mask = load_ard's (variant_s2_scene_rule)."),
        "C_record": ("Villaviciosa PRIMARY = all 465 2023-2025 scenes (ITEM "
                     "all-observations rule); the 97-scene MAREA record becomes "
                     "the labelled sensitivity 'NIDEM on MAREA's 97-scene "
                     "record'. Dutch primaries unchanged (all scenes); Ems "
                     "primary re-evaluated by its truth-free pre-flight."),
        "D_unchanged": ("intervals, thresholds, contouring, TIN, filters, "
                        "bounded supplementary, pre-flight guard (rule and "
                        "floor) unchanged"),
    },
    "history": {
        "superseded_2026_09_29": [
            ("pixel mask = MAREA's clear set SCL {4,5,6,7} (spec step 2, "
             "'so all three methods see the same observations'): superseded by "
             "A (each method applies its own published pixel mask)"),
            ("variant good fraction = share of frame px with SCL in {4,5,6,7}, "
             "no data NOT good, pixel mask {4,5,6,7}: superseded by B (literal "
             "load_ard); the old statistic is still reported per scene "
             "(clear_scl_fraction, information only)"),
            ("Villaviciosa PRIMARY = MAREA's 97-scene record and the "
             "all-scenes 465-scene run as SENSITIVITY "
             "(products_marea/nidem/sensitivity_all_scenes): superseded by C; "
             "the Villaviciosa variant caveat (rule acting on an already "
             "cloud-filtered record) no longer applies because the variant now "
             "acts on the 465-scene primary record"),
        ],
        "backup_of_superseded_outputs": (
            "small files (json/csv) and sources copied to the session "
            "scratchpad nidem_backup_before_literal/ before the re-runs"),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def subset_record(rec, sel, scene_rule_note):
    """The record restricted to the scenes flagged in ``sel`` (time order
    kept). Used only by the variant."""
    sel = np.asarray(sel, bool)
    out = dict(rec)
    out.update(time_index=rec["time_index"][sel], dates=rec["dates"][sel],
               t_real=rec["t_real"][sel], h=rec["h"][sel],
               rule=f"{VARIANT_LEAD}: {rec['rule']}; then {scene_rule_note}",
               n_dropped_scene_rule=int((~sel).sum()))
    return out


def scene_rule_record(rec, q, sel):
    """Truth-free record of the variant's scene drop (all record scenes).
    ``q`` is the pre-pass of the record scenes
    (:func:`pyintertidal.waterline_contour.stream_scene_quality`), ``sel``
    the keep flags of :func:`pyintertidal.waterline_contour.s2_scene_rule`
    on ``q['load_ard_good']``."""
    thr = wc.S2_SCENE_MIN_GOODDATA
    old = q["clear_scl"] >= thr
    pc = [0, 5, 25, 50, 75, 95, 100]

    def scene(i):
        return {"date": str(rec["dates"][i]), "t_real": str(rec["t_real"][i]),
                "cube_time_index": int(rec["time_index"][i]),
                "h_m": float(rec["h"][i]),
                "frame_good_fraction (load_ard)": float(q["load_ard_good"][i]),
                "frame_no_data_share": float(q["no_data"][i]),
                "frame_clear_scl_share (superseded, info)": float(q["clear_scl"][i])}

    h = rec["h"]
    return {
        "name": VARIANT_NAME,
        "rule": FROZEN["variant_s2_scene_rule"]["rule"],
        "min_gooddata": thr,
        "frame_px": int(q["px"]),
        "n_record_scenes_before": int(len(sel)),
        "n_retained": int(sel.sum()), "n_dropped": int((~sel).sum()),
        "record_h_range_before_m": [float(h.min()), float(h.max())],
        "retained_h_range_m": [float(h[sel].min()), float(h[sel].max())],
        "frame_good_fraction_percentiles (all record scenes)": {
            f"p{p}": float(np.percentile(q["load_ard_good"], p)) for p in pc},
        "retained_by_no_data_share_ge_0.5 (swath edge counted good)":
            int((sel & (q["no_data"] >= 0.5)).sum()),
        "dropped_scenes": [scene(i) for i in np.flatnonzero(~sel)],
        "info_superseded_rule (good = SCL in {4,5,6,7}, no data NOT good; NOT used)": {
            "n_retained_if_superseded": int(old.sum()),
            "n_decisions_differ": int((old != sel).sum()),
            "kept_by_literal_dropped_by_superseded": [str(d) for d in rec["dates"][sel & ~old]],
            "dropped_by_literal_kept_by_superseded": [str(d) for d in rec["dates"][~sel & old]],
        },
        "per_scene (all record scenes, time order)": {
            "date": [str(d) for d in rec["dates"]],
            "h_m": [float(v) for v in h],
            "frame_good_fraction": [float(v) for v in q["load_ard_good"]],
            "frame_no_data_share": [float(v) for v in q["no_data"]],
            "retained": [bool(v) for v in sel],
        },
        "seconds": float(q["seconds"]),
    }


def not_computable_reason(pre, lead, ppath):
    """One-line reason from a stopped pre-flight report."""
    cov = pre[COV_KEY]
    P = int(cov["px"])
    parts = []
    for q, scs in pre["scenes_of_intervals_below_floor"].items():
        size = pre["interval_sizes |S_j|"][int(q) - 1]
        desc = ", ".join(
            f"{s['date']} valid on {int(round(s['valid_finite_share_on_keep'] * P)):,}"
            f" of {P:,} keep px" for s in scs[:5])
        more = f", and {len(scs) - 5} more" if len(scs) > 5 else ""
        parts.append(f"interval {q} holds {size} scene(s) ({desc}{more})")
    mask = pre.get("pixel_mask", {}).get("rule", "item")
    return (f"{lead}: NOT COMPUTABLE. Under ITEM's validity rule (a pixel is "
            "valid iff it has >= 1 valid observation in every one of the 9 tide "
            f"intervals; pixel mask '{mask}'), " + "; ".join(parts) +
            f"; expected valid share of keep {cov['expected_valid_share']:.3g} "
            f"({cov['expected_valid_px']:,} of {P:,} px) < pre-flight floor "
            f"{cov['min_valid_share']}. Nothing was streamed or scored; see {ppath}")


REVISION_KEY = "revision_2026_09_29_published_rules"


def _is_current_computed(jpath):
    """True iff ``jpath`` is a computed (not NOT COMPUTABLE) diagnostics file
    written under the current rules (its FROZEN holds REVISION_KEY)."""
    if not os.path.exists(jpath):
        return False
    try:
        d = json.load(open(jpath, encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return (d.get("status") != "NOT COMPUTABLE"
            and REVISION_KEY in (d.get("frozen_constants") or {}))


def _refuse_overwrite(jpath, paths):
    """A not-computable record never overwrites a result computed under the
    current rules. Outputs of the superseded rules (before the 2026-09-29
    revision; backed up) may be replaced."""
    if _is_current_computed(jpath):
        raise RuntimeError(f"{jpath} is a result computed under the current "
                           "rules; refusing to overwrite it with a NOT "
                           "COMPUTABLE record")
    for path in paths:
        if os.path.exists(path) and os.path.exists(jpath):
            txt = open(path, encoding="utf-8").read()
            if "NOT COMPUTABLE" not in txt and _is_current_computed(jpath):
                raise RuntimeError(f"{path} belongs to a computed result; "
                                   "refusing to overwrite it")


def record_not_computable(site, cfg, *, mode, pre, ppath,
                          csv_paths, out_dir, grid, keep, record_label,
                          reason=None, n_before_scene_rule=None):
    """Decision (a) (and a stopped variant / sensitivity): write the metrics
    CSV(s) with the reproduced published rows plus a NOT COMPUTABLE NIDEM
    row (n = 0, note = reason), and nidem_diagnostics.json. The pre-flight
    file is only read."""
    variant = mode == "variant"
    name = not_computable_name(mode)
    lead = {"variant": VARIANT_LEAD, "sensitivity": SENSITIVITY_LEAD,
            "primary": "PRIMARY (ITEM rule)"}[mode]
    if reason is None:
        reason = not_computable_reason(pre, lead, ppath)
    jpath = os.path.join(out_dir, "nidem_diagnostics.json")
    _refuse_overwrite(jpath, list(csv_paths.values()))
    log(f"  recording {name} as NOT COMPUTABLE: {reason}")
    t0 = time.time()
    if site == "villaviciosa":
        path, counts = score_rtk(csv_paths["rtk"], None, keep,
                                 (grid["H"], grid["W"]), mode,
                                 int(pre["n_scenes"]), not_computable=reason,
                                 n_before_scene_rule=n_before_scene_rule)
        written = [path]
        inputs = [VVA_NOTEBOOK, "products_marea/external_products.npz",
                  "products_marea/rtk_onsite_metrics.csv"]
    else:
        written, counts = score_vaklodingen(site, cfg, None, keep, grid,
                                            csv_paths, variant=variant,
                                            not_computable=reason)
        inputs = [f"{cfg['comparison']}/{f}" for f in
                  ("products.npz", "vaklodingen_metrics.csv",
                   "vaklodingen_by_band.csv")]
    out = {
        "experiment": "v7_waterline_contour", "site": site,
        "record": record_label, "record_rule": pre.get("record_rule"),
        "boundary": pre.get("boundary"), "mode": mode,
        "status": "NOT COMPUTABLE",
        "product": name,
        "reason": reason,
        "pixel_mask": pre.get("pixel_mask"),
        "preflight_file": ppath,
        "preflight_status": pre.get("status"),
        "preflight_sha256": sha256(ppath) if os.path.exists(ppath) else None,
        "n_scenes": pre.get("n_scenes"),
        "interval_sizes |S_j|": pre.get("interval_sizes |S_j|"),
        "Z_j_m": pre.get("Z_j_m"),
        COV_KEY: pre.get(COV_KEY),
        "scene_quality_prepass (record scenes)":
            pre.get("scene_quality_prepass (record scenes)"),
        "frozen_constants": FROZEN,
        "scoring": counts,
        "written": list(written) + [jpath],
        "runtime_s": {"scoring_s": time.time() - t0},
        "inputs_sha256": {p_: sha256(p_) for p_ in
                          [cfg["extract"], cfg["overpass"]] + inputs},
    }
    if variant:
        out["nidem_variant"] = VARIANT_NAME
        out["s2_scene_rule"] = pre.get("s2_scene_rule")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    log(f"  wrote {jpath} and {written}")
    return out


def record_label_of(mode):
    return {"primary": "primary: ITEM all-observations record, ITEM pixel-quality mask",
            "sensitivity": f"{SENSITIVITY_LEAD}, ITEM pixel-quality mask",
            "variant": (f"{VARIANT_LEAD}: primary record, scenes with a "
                        f"full-frame load_ard good fraction < "
                        f"{wc.S2_SCENE_MIN_GOODDATA} dropped, load_ard pixel mask")}[mode]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site", choices=sorted(SITES))
    ap.add_argument("--sensitivity-marea-record", action="store_true",
                    help="villaviciosa only: labelled sensitivity 'NIDEM on "
                         "MAREA's 97-scene record'")
    ap.add_argument("--sensitivity-all-scenes", action="store_true",
                    help="SUPERSEDED (2026-09-29): the all-scenes record is the "
                         "Villaviciosa primary; refused")
    ap.add_argument("--s2-scene-rule", action="store_true",
                    help=f"LABELLED VARIANT '{VARIANT_NAME}': literal DE Africa "
                         "load_ard (drop record scenes whose full-frame good "
                         f"fraction is < {wc.S2_SCENE_MIN_GOODDATA}; load_ard "
                         "pixel mask); separate outputs")
    ap.add_argument("--record-not-computable", action="store_true",
                    help="write the NOT COMPUTABLE metrics rows and diagnostics "
                         "from the run's existing STOPPED nidem_preflight.json "
                         "(decision (a)); the pre-flight file is only read")
    ap.add_argument("--row-block", type=int, default=None)
    ap.add_argument("--tmpdir", default=None,
                    help="where large check arrays are memory-mapped")
    ap.add_argument("--preflight-only", action="store_true",
                    help="stop after the truth-free pre-flight (interval "
                         "coverage on keep); writes nidem_preflight.json only")
    args = ap.parse_args(argv)
    site, cfg = args.site, SITES[args.site]
    if args.sensitivity_all_scenes:
        raise SystemExit("--sensitivity-all-scenes is superseded (revision of "
                         "2026-09-29): the all-scenes record is the Villaviciosa "
                         "PRIMARY; the sensitivity is --sensitivity-marea-record")
    sensitivity, variant = args.sensitivity_marea_record, args.s2_scene_rule
    if sensitivity and site != "villaviciosa":
        raise SystemExit("--sensitivity-marea-record applies to villaviciosa only "
                         "(the Dutch MAREA records already use every scene)")
    if sensitivity and variant:
        raise SystemExit("--s2-scene-rule applies to the primary record only")
    if args.record_not_computable and args.preflight_only:
        raise SystemExit("--record-not-computable and --preflight-only exclude "
                         "each other")
    mode = run_mode(sensitivity, variant)
    rule = pixel_rule(mode)
    t_start = time.time()
    timings = {}
    out_dir, csv_paths = output_paths(site, sensitivity, variant)
    os.makedirs(out_dir, exist_ok=True)
    record_label = record_label_of(mode)
    log(f"[v7] {site} ({record_label}) -> {out_dir}")

    # grid and extraction consistency
    grid = cube_grid(cfg["cube"])
    H, W = grid["H"], grid["W"]
    z_ = np.load(cfg["extract"])
    ext_dates = np.array([str(d) for d in z_["dates"]])
    keep = z_["keep"].astype(np.int64)
    ext_shape = tuple(int(v) for v in z_["shape"])
    del z_
    assert ext_shape == (H, W), (ext_shape, (H, W))
    assert np.array_equal(ext_dates, grid["dates"]), "extraction dates != cube dates"
    assert len(set(ext_dates.tolist())) == len(ext_dates), "duplicate dates"
    assert np.all(np.diff(keep) > 0) and keep[0] >= 0 and keep[-1] < H * W
    if site == "villaviciosa":
        mp = np.load("products_marea/marea.npz")
        assert np.array_equal(mp["keep"], keep), "marea.npz keep != extraction keep"
        del mp
        import rasterio
        with rasterio.open(RTK_GRID) as s_:
            assert s_.shape == (H, W) and s_.transform.almost_equals(grid["transform"]), \
                "cube grid differs from the RTK grid"
    log(f"  grid {H} x {W}, {len(ext_dates)} cube scenes, keep {len(keep):,} px; "
        f"pixel mask '{rule}'")
    ppath = os.path.join(out_dir, "nidem_preflight.json")
    nc_kw = dict(mode=mode, ppath=ppath, csv_paths=csv_paths, out_dir=out_dir,
                 grid=grid, keep=keep, record_label=record_label)

    if args.record_not_computable:
        # decision (a): from the existing STOPPED pre-flight, which is only read
        pre = json.load(open(ppath, encoding="utf-8"))
        if pre.get("site") != site or not str(pre.get("status", "")).startswith("STOPPED"):
            raise SystemExit(f"{ppath} is not a STOPPED pre-flight of {site}")
        if pre.get("mode", "variant" if "nidem_variant" in pre else "primary") != mode:
            raise SystemExit(f"{ppath} does not belong to this run mode")
        n_before = (pre.get("s2_scene_rule") or {}).get("n_record_scenes_before")
        return record_not_computable(site, cfg, pre=pre,
                                     n_before_scene_rule=n_before, **nc_kw)

    # record
    t0 = time.time()
    rec = build_record(site, cfg, ext_dates, marea_record=sensitivity)
    timings["record_s"] = time.time() - t0
    n_before_scene_rule = len(rec["h"])
    log(f"  record before any scene rule: {n_before_scene_rule} scenes; {rec['rule']}")

    # truth-free pre-pass of the record scenes: frame statistics of the scene
    # rule(s) and the run's observation masks at keep
    q = wc.stream_scene_quality(cfg["cube"], rec["time_index"], keep, rule=rule)
    timings["scene_quality_prepass_s"] = q["seconds"]
    log(f"  pre-pass ({q['seconds']:.0f} s): observations on keep "
        f"{int(q['n_obs_keep'].sum()):,} vs MAREA C {int(q['n_marea_obs_keep'].sum()):,}"
        f" (+{int(q['n_added_keep'].sum()):,} / -{int(q['n_removed_keep'].sum()):,}); "
        f"frame SCL == 0 px {int(q['scl_zero_px'].sum())}; band <= "
        f"{wc.LOAD_ARD_MIN_VALUE} px {int(q['band_le_load_ard_min_px'].sum())}, < "
        f"{wc.LOAD_ARD_MIN_VALUE} px {int(q['band_below_offset_px'].sum())}")
    if int(q["band_below_offset_px"].sum()) != 0:
        raise AssertionError("cube values below -999 exist: the -1000 BOA offset "
                             "with DN 0 = NaN assumption of the load_ard DN rule "
                             "does not hold")
    scene_rule = None
    if variant:
        # the Sentinel-2 scene rule (literal load_ard): full-frame good
        # fraction of every record scene, drop those below the threshold,
        # BEFORE the OTR and intervals
        sel = wc.s2_scene_rule(q["load_ard_good"])
        scene_rule = scene_rule_record(rec, q, sel)
        rec = subset_record(rec, sel, (
            f"scenes with a full-frame load_ard good fraction < "
            f"{wc.S2_SCENE_MIN_GOODDATA} dropped (Sentinel-2 scene rule, "
            f"load_ard min_gooddata): {int((~sel).sum())} of {len(sel)}"))
        q = wc.subset_scene_quality(q, sel)
        log(f"  Sentinel-2 scene rule: kept {int(sel.sum())} of {len(sel)} record "
            f"scenes (full-frame load_ard good fraction >= "
            f"{wc.S2_SCENE_MIN_GOODDATA}); frame good fraction p50 "
            f"{scene_rule['frame_good_fraction_percentiles (all record scenes)']['p50']:.3f}; "
            f"superseded CLEAR-share rule would keep "
            f"{scene_rule['info_superseded_rule (good = SCL in {4,5,6,7}, no data NOT good; NOT used)']['n_retained_if_superseded']}")
    quality = scene_quality_summary(q)
    h = rec["h"]
    try:
        j, edges, Z, U, sizes = wc.interval_assignment(h)
    except wc.EmptyIntervalError as e:
        # no NIDEM rule for an empty interval: stop and report
        pre = {"experiment": "v7_waterline_contour", "stage": "preflight",
               "site": site, "record": record_label, "record_rule": rec["rule"],
               "boundary": rec["boundary"], "truth_used": "none", "mode": mode,
               "status": f"STOPPED: empty tidal interval ({e})",
               "pixel_mask": pixel_mask_record(mode),
               "n_scenes": int(len(h)),
               "scene_quality_prepass (record scenes)": quality}
        if variant:
            pre["nidem_variant"] = VARIANT_NAME
            pre["s2_scene_rule"] = scene_rule
        with open(ppath, "w", encoding="utf-8") as f:
            json.dump(pre, f, indent=1, default=float)
        record_not_computable(site, cfg, pre=pre, reason=(
            f"{mode_leads(mode)[0]}: NOT COMPUTABLE, empty tidal interval: {e}"),
            n_before_scene_rule=n_before_scene_rule, **nc_kw)
        raise
    log(f"  record: {len(h)} scenes ({rec['dates'][0]} .. {rec['dates'][-1]}), "
        f"boundary {rec['boundary']}, h [{h.min():+.3f}, {h.max():+.3f}] m")
    log(f"  |S_j| = {sizes.tolist()}")
    log(f"  Z_j = {np.round(Z, 3).tolist()}")
    log(f"  U_j = {np.round(U, 3).tolist()}")

    # extraction arrays of the record scenes, for the equality checks
    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="v7_nidem_", dir=args.tmpdir)
    try:
        Cr = npz_rows(cfg["extract"], "C", rec["time_index"], tmpdir)
        # the pre-pass SCL reproduces the extraction's C on keep
        for i in range(len(rec["time_index"])):
            if not np.array_equal(q["clear_keep"].packed[i],
                                  np.packbits(np.asarray(Cr[i], bool))):
                raise AssertionError(f"pre-pass SCL in {wc.CLEAR} differs from "
                                     f"the extraction C at scene "
                                     f"t={int(rec['time_index'][i])}")
        log("  pre-pass SCL reproduces the extraction's C on keep "
            f"({len(rec['time_index'])} scenes)")
        timings["load_check_C_s"] = time.time() - t0

        # pre-flight (review round 2): the validity map on keep is known from
        # the pre-pass before any composite; stop and report if the intervals
        # are observed at (almost) no keep pixel. Truth-free.
        t0 = time.time()
        cnt_pre, per_scene = wc.clear_counts_by_interval(q["obs_keep"], None, j)
        assert np.array_equal(per_scene, q["n_obs_keep"])
        cov = wc.interval_coverage(cnt_pre, raise_error=False)
        pre = preflight_report(site, rec, record_label, j, edges, Z, U, sizes,
                               cov, per_scene, cnt_pre.sum(axis=1, dtype=np.int64),
                               len(keep), mode=mode, scene_rule=scene_rule,
                               quality=quality)
        timings["preflight_s"] = time.time() - t0
        with open(ppath, "w", encoding="utf-8") as f:
            json.dump(pre, f, indent=1, default=float)
        log(f"  pre-flight: share of keep with >= 1 valid obs per interval "
            f"{[round(v, 4) for v in cov['share_px_with_ge1_clear_obs']]}; "
            f"expected valid {cov['expected_valid_share']:.4f} "
            f"({cov['expected_valid_px']:,} px) -> "
            f"{'passed' if cov['passed'] else 'STOP'}; wrote {ppath}")
        if not cov["passed"] or args.preflight_only:
            del Cr, cnt_pre      # release the memmaps before the rmtree
            gc.collect()
        if not cov["passed"]:
            if not args.preflight_only:
                record_not_computable(site, cfg, pre=pre,
                                      n_before_scene_rule=n_before_scene_rule,
                                      **nc_kw)
            raise wc.NoClearObservationsInIntervalError(
                f"{site}: pre-flight stop, expected valid share on keep "
                f"{cov['expected_valid_share']:.3g} < {cov['min_valid_share']} "
                f"(interval(s) {cov['intervals_below_floor']}); nothing streamed "
                f"or scored; see {ppath}", coverage=cov)
        if args.preflight_only:
            log("  --preflight-only: stopping before the cube is streamed")
            return pre
        t0 = time.time()
        Yr = npz_rows(cfg["extract"], "Y", rec["time_index"], tmpdir)
        timings["load_check_Y_s"] = time.time() - t0
        check = ExtractionCheck(Yr, Cr, rec["time_index"], keep, W)

        # composites, streamed
        comp = wc.stream_composites(cfg["cube"], rec["time_index"], j,
                                    row_block=args.row_block, check=check,
                                    rule=rule)
        timings["composites_s"] = comp["seconds"]
        row_block_used = comp["row_block"]
        n_expected = len(rec["time_index"]) * len(keep)
        assert check.n_c == n_expected, (check.n_c, n_expected)
        check_info = check.summary()
        for k_chk, k_pre in (("observations_on_keep (pixel mask & finite NDWI)", "n_obs_keep"),
                             ("marea_observations_on_keep (C & finite NDWI)", "n_marea_obs_keep"),
                             ("observations_in_both", "n_both_keep"),
                             ("observations_added_vs_marea_C", "n_added_keep"),
                             ("observations_removed_vs_marea_C", "n_removed_keep")):
            assert check_info[k_chk] == int(q[k_pre].sum()), (k_chk, check_info[k_chk])
        if rule == "item":
            # ITEM's valid set {2,4,5,6,7,11} contains MAREA's {4,5,6,7}
            assert check_info["observations_removed_vs_marea_C"] == 0
        log(f"  extraction check passed: {check.n_c:,} (scene, px) SCL->C flags and "
            f"raw NDWI values equal to the npz ({check.n_both_valid_compared:,} "
            f"where both masks are valid); observations on keep "
            f"{check.n_obs:,} vs MAREA {check.n_marea_obs:,}: +{check.n_added:,} "
            f"added, -{check.n_removed:,} removed")
        assert np.array_equal(comp["count"].reshape(wc.N_INTERVALS, -1)[:, keep],
                              cnt_pre), "pre-flight counts != composites' count on keep"
        log("  pre-flight counts equal the composites' count on keep")
        del check, Yr, Cr, cnt_pre
        gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # NIDEM
    res = wc.nidem(comp, Z, U)
    timings["nidem_s"] = res["seconds"]
    log(f"  NIDEM: {res['n_vertices']:,} contour vertices, empty levels "
        f"{res['empty_levels']}, {int(np.isfinite(res['filtered']).sum()):,} "
        f"filtered px ({res['seconds']:.0f} s)")
    diag = diagnostics(res, comp, keep, Z, U, sizes, edges, h)

    # products
    rc, lv = wc.contour_vertices(res["contours"])
    cnt_keep = comp["count"].reshape(wc.N_INTERVALS, -1)[:, keep]
    med_keep = comp["median"].reshape(wc.N_INTERVALS, -1)[:, keep]
    above_keep = at_keep(comp["n_ndwi_above_bound"], keep)
    below_keep = at_keep(comp["n_ndwi_below_minus_bound"], keep)
    oor = diag["ndwi_out_of_range"]
    log(f"  |NDWI| > 1: {oor['share_obs_abs_ndwi_gt_1_on_keep']:.2%} of valid obs on "
        f"keep ({oor['obs_ndwi_gt_+1_on_keep']:,} > +1, "
        f"{oor['obs_ndwi_lt_-1_on_keep']:,} < -1), max |NDWI| "
        f"{oor['full_frame']['max_abs_ndwi']:.1f}; in "
        f"{oor['conf_failing_R_1_8_keep_px_with_abs_ndwi_gt_1_obs']:,} of "
        f"{oor['conf_failing_R_1_8_keep_px']:,} conf-failing R 1..8 keep px; conf "
        f"pass rate {oor['conf_pass_rate_within_R_1_8_on_keep (raw NDWI, PRIMARY)']:.1%}"
        f" -> {oor['conf_pass_rate_within_R_1_8_on_keep (NDWI clipped to [-1,1], SUPPLEMENTARY)']:.1%}"
        f" with NDWI clipped to [-1, 1]")
    nest = oor["full_frame"]["filtered_px_not_in_filtered_bounded (nesting check, expect 0)"]
    assert nest == 0, f"{nest} filtered px are not in filtered_bounded"
    del comp
    gc.collect()
    npz_path = os.path.join(out_dir, "nidem.npz")
    extra_npz = {}
    if variant:
        ps = scene_rule["per_scene (all record scenes, time order)"]
        extra_npz = dict(
            scene_rule_min_gooddata=np.float64(wc.S2_SCENE_MIN_GOODDATA),
            scene_rule_record_dates=np.array(ps["date"]),
            scene_rule_record_h=np.array(ps["h_m"]),
            scene_rule_frame_good_fraction=np.array(ps["frame_good_fraction"]),
            scene_rule_frame_no_data_share=np.array(ps["frame_no_data_share"]),
            scene_rule_retained=np.array(ps["retained"], bool))
    np.savez_compressed(
        npz_path, R=res["R"], conf=res["conf"], unfiltered=res["unfiltered"],
        filtered=res["filtered"], uncertainty=res["uncertainty"],
        valid=res["valid"], non_monotone=res["non_monotone"],
        keep=keep, shape=np.array([H, W]),
        R_keep=at_keep(res["R"], keep), conf_keep=at_keep(res["conf"], keep),
        filtered_keep=at_keep(res["filtered"], keep),
        unfiltered_keep=at_keep(res["unfiltered"], keep),
        uncertainty_keep=at_keep(res["uncertainty"], keep),
        count_keep=cnt_keep, median_keep=med_keep,
        conf_bounded=res["conf_bounded"],
        filtered_bounded=res["filtered_bounded"],
        conf_bounded_keep=at_keep(res["conf_bounded"], keep),
        filtered_bounded_keep=at_keep(res["filtered_bounded"], keep),
        n_ndwi_above_1_keep=above_keep, n_ndwi_below_minus_1_keep=below_keep,
        Z=Z, U=U, sizes=sizes, edges=edges, h=h, interval=j,
        time_index=rec["time_index"], dates=rec["dates"],
        t_real=np.array([str(t) for t in rec["t_real"]]),
        contour_rc=rc.astype(np.float32), contour_level=lv.astype(np.float32),
        pixel_rule=np.array(rule),
        scene_obs_keep=q["n_obs_keep"], scene_load_ard_good=q["load_ard_good"],
        scene_item_valid_share=q["item_valid"], scene_no_data_share=q["no_data"],
        **extra_npz)
    tifs = write_rasters(out_dir, res, grid)
    log(f"  wrote {npz_path} and {len(tifs)} GeoTIFFs")

    # scoring
    t0 = time.time()
    if site == "villaviciosa":
        csv_path, score_counts = score_rtk(csv_paths["rtk"], res, keep, (H, W),
                                           mode, len(h),
                                           n_before_scene_rule=n_before_scene_rule)
        written = [csv_path]
    else:
        written, score_counts = score_vaklodingen(site, cfg, res, keep, grid,
                                                  csv_paths, variant=variant)
    timings["scoring_s"] = time.time() - t0
    timings["total_s"] = time.time() - t_start

    inputs = {cfg["extract"]: sha256(cfg["extract"]),
              cfg["overpass"]: sha256(cfg["overpass"])}
    if site == "villaviciosa":
        for p_ in (VVA_NOTEBOOK, "products_marea/external_products.npz",
                   "products_marea/rtk_onsite_metrics.csv"):
            inputs[p_] = sha256(p_)
    else:
        for p_ in (f"{cfg['comparison']}/products.npz",
                   f"{cfg['comparison']}/vaklodingen_metrics.csv",
                   f"{cfg['comparison']}/vaklodingen_by_band.csv"):
            inputs[p_] = sha256(p_)
    st = os.stat(cfg["cube"])
    out = {
        "experiment": "v7_waterline_contour", "site": site, "record": record_label,
        "mode": mode,
        "record_rule": rec["rule"], "boundary": rec["boundary"],
        "scenes_without_overpass_time": rec["n_without_time"],
        "scenes_with_nonfinite_level": rec["n_nonfinite_level"],
        "record_dates_first_last": [str(rec["dates"][0]), str(rec["dates"][-1])],
        "cube": {"path": cfg["cube"], "bytes": st.st_size, "mtime": st.st_mtime,
                 "shape": [H, W], "transform": list(grid["transform"])[:6]},
        "pixel_mask": pixel_mask_record(mode),
        "frozen_constants": FROZEN,
    }
    if variant:
        out["nidem_variant"] = VARIANT_NAME
        out["s2_scene_rule"] = scene_rule
    out |= {
        "extraction_check": check_info,
        "scene_quality_prepass (record scenes)": quality,
        "preflight": {k: pre[k] for k in ("status", "guard", COV_KEY)},
        "row_block": row_block_used,
        **diag,
        "scoring": score_counts,
        "written": [ppath, npz_path] + tifs + list(written),
        "runtime_s": timings,
        "peak_memory_mb (process peak working set)": peak_memory_mb(),
        "inputs_sha256": inputs,
    }
    jpath = os.path.join(out_dir, "nidem_diagnostics.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    log(f"  wrote {jpath}; total {timings['total_s']:.0f} s, "
        f"peak {out['peak_memory_mb (process peak working set)']:.0f} MB")
    return out


if __name__ == "__main__":
    try:
        main()
    except wc.EmptyIntervalError as e:     # includes the pre-flight stop
        print(f"[v7] NOT COMPUTABLE: {e}", file=sys.stderr, flush=True)
        sys.exit(3)
