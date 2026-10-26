"""V8 — Granadeiro et al. (2021), a FAITHFUL reimplementation (comparison method).

Granadeiro, J.P.; Belo, J.; Henriques, M.; Catalao, J.; Catry, T. (2021)
"Using Sentinel-2 Images to Estimate Topography, Tidal-Stage Lags and
Exposure Periods over Large Intertidal Areas", Remote Sensing 13(2):320.

It replaces experiments/sota_granadeiro.py + c0_granadeiro_extract.py +
e0_escalda_prep.py (left untouched) and follows the fidelity audit
(gran_audit_upstream.json, 26 steps) under the author's rules:

1. Faithful to the paper in every step, with ONE allowed exception: the
   imagery is Sentinel-2 L2A (Sen2Cor) from MAREA's own cubes instead of
   ACOLITE, at MAREA's working resolution (10 m Villaviciosa, 20 m Dutch
   sites), bands B03 / B08, period 2023-01-01..2025-12-31 at every site.
2. No computational shortcuts: every masked pixel is fitted, the lag sample
   is min(50,000, pool), every calibration pixel is used.
3. The statistical steps run in the R packages the paper names
   (pyintertidal/r/granadeiro_fit.R): the logistic is nplr's own fit, run
   "lean" (nplr's internals, proven identical to nplr() + getInflexion on
   real pixels at every run); the major axis is lmodel2's MA row (closed
   form proven equal to lmodel2 to 1e-10 on every calibration regression of
   the run); the lag surface is mgcv::gam(lag ~ s(lon, lat)) with defaults.
4. Parallel R worker processes (--workers, default 8), raw float64 files in
   a temporary directory deleted at the end.

The decisions for the steps the paper leaves open are frozen in FROZEN
below (keyed by audit step) and were written before the first run.

Run (from the repo root)
------------------------
    python -X utf8 -m experiments.v8_granadeiro2021 villaviciosa --workers 8
    python -X utf8 -m experiments.v8_granadeiro2021 escalda --workers 8
    python -X utf8 -m experiments.v8_granadeiro2021 wadden --workers 8
    python -X utf8 -m experiments.v8_granadeiro2021 ems --workers 8
Options: --tmpdir DIR (R exchange files; default system temp), --no-wait
(skip the free-RAM guard), --resume (reuse finished R stages of an
interrupted run of the same inputs).

Outputs (existing result files are never modified)
-----------------------------------------------------
villaviciosa: products_marea/granadeiro2021/
Dutch:        products_<site>/granadeiro2021_<tag>/
    granadeiro2021.npz, granadeiro2021_dem.tif, granadeiro2021_lag_min.tif,
    granadeiro2021_intertidal_mask.tif, granadeiro2021_exposure_h.tif,
    granadeiro2021_diagnostics.json, and the scores:
    villaviciosa rtk_metrics_granadeiro2021.csv (RTK development split);
    Dutch vaklodingen_metrics_granadeiro2021.csv and
    vaklodingen_by_band_granadeiro2021.csv.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pyintertidal import granadeiro2021 as g21  # noqa: E402
from experiments.validation_grid import (DUTCH_RES_M, dutch_cube, dutch_products,  # noqa: E402
                                         thin_mask, thin_px)

TIME_EXTENT = ("2023-01-01", "2025-12-31")
#: the 1-min boundary series spans the period +- this many days, so every
#: scene time shifted by +-90 min has a bracketing pair of marks
SERIES_PAD_DAYS = 2
#: days per boundary.levels call (the EOT20 prediction peaks near 0.6 GB for
#: 92 days of 1-min samples)
SERIES_CHUNK_DAYS = 92
RTK_CSV = "data/villaviciosa_rtk_gnss.csv"
RTK_GRID = "products_villaviciosa/hsr_eot20_2023-2025.tif"
SEED = 20210119
PRODUCT = "Granadeiro 2021 (faithful)"
PROOF_TOL = 1e-12
MA_TOL = 1e-10
TOL = 1e-9

SITES = {
    "villaviciosa": dict(cube="ndwi_cube_villaviciosa_grande_10y.nc",
                         extract="products_marea/marea_extract.npz",
                         overpass="products_marea/overpass_times.json",
                         out="products_marea/granadeiro2021", pixel_m=10.0),
    "escalda": dict(cube=dutch_cube("escalda"), n_gauges=2,
                    judge=["trnz"], tag="eot20_g2", pixel_m=float(DUTCH_RES_M)),
    "wadden": dict(cube=dutch_cube("wadden"), n_gauges=1,
                   judge=["harl"], tag="eot20_g1", pixel_m=float(DUTCH_RES_M)),
    "ems": dict(cube=dutch_cube("ems"), n_gauges=1,
                judge=["delf"], tag="eot20_g1", pixel_m=float(DUTCH_RES_M)),
}
for _s in ("escalda", "wadden", "ems"):
    _c, _p = SITES[_s], dutch_products(_s)
    _c["extract"] = f"{_p}/marea_extract.npz"
    _c["overpass"] = f"{_p}/overpass_times.json"
    _c["out"] = f"{_p}/granadeiro2021_{_c['tag']}"
    _c["comparison"] = f"{_p}/comparison_{_c['tag']}"

FROZEN = {
    "reference": g21.PAPER,
    "status": ("decisions for the steps the paper leaves open, fixed by the "
               "author (orchestrator brief of 2026-09-30) before the first run; "
               "keys = step numbers of the fidelity audit "
               "(gran_audit_upstream.json)"),
    "rules": {
        "1_fidelity": ("faithful to the paper in every step; ONE allowed "
                       "exception: Sentinel-2 L2A (Sen2Cor) reflectance from "
                       "MAREA's cubes instead of ACOLITE, at MAREA's working "
                       "resolution; 2023-01-01..2025-12-31 at every site"),
        "2_no_shortcuts": ("no pixel stride, no reduced lag sample, no "
                           "subsampled calibration pixels"),
        "3_software": ("nplr (lean path = nplr internals, proven equal to "
                       "nplr() + getInflexion to 1e-12 with identical failures "
                       "on >= 2,000 real pixels every run), lmodel2 MA row "
                       "(closed form proven to 1e-10 on every calibration "
                       "regression of the run), mgcv::gam defaults"),
        "4_speed": "parallel Rscript workers (default 8), raw float64 LE exchange files",
    },
    "steps": {
        "1_imagery": ("B03 (green 560 nm) and B08 (NIR 833 nm) of the cube; "
                      "reflectance = cube value / 10000. The cubes hold int16 "
                      "values = DN + BOA_ADD_OFFSET (-1000; processing baseline "
                      ">= 04.00), no scale_factor / add_offset attribute, "
                      "_FillValue -32768 = no data (checked at run time). "
                      "Negative reflectance (dark water) is valid data and is "
                      "KEPT (the old extractor turned NIR <= 0 into NaN)."),
        "2_grid": "the cube grid: 10 m Villaviciosa (915 x 915), 20 m Escalda / Wadden / Ems",
        "3_period": "2023-01-01..2025-12-31",
        "4_5_scene_rule": ("a scene is used iff its FULL-FRAME share of SCL "
                           "{3, 8, 9, 10} is < 0.10 (strict) and its no-data "
                           "share (SCL fill / NaN or 0) is <= 0.02 (our reading "
                           "of the paper's whole-tile scenes), it lies in "
                           "2023-2025 and has an overpass time; denominator = "
                           "every frame pixel, every pixel counted (no subsampling)"),
        "6_selection": "every scene passing the rule is used",
        "7_pixel_mask": ("NO per-pixel cloud mask (the scene rule is the "
                         "paper's only cloud filter); no-data pixels are NaN "
                         "and each pixel's logistic drops them as nplr drops NAs"),
        "8_upstream": "no filter, coregistration, saturation index or gridding (none in the paper)",
        "9_reference_scene": ("among the used scenes, the one minimising "
                              "|range_G - median_G| / median_G + |range_NIR - "
                              "median_NIR| / median_NIR, range = p99 - p1 "
                              "(np.percentile linear = R type 7) over the "
                              "finite frame pixels of the UNcalibrated scene; "
                              "first on ties"),
        "10_intercalibration": ("for B08 and B03 separately, every non-reference "
                                "scene is regressed on the reference (y = scene, "
                                "x = reference; lmodel2 row 'MA') over ALL frame "
                                "pixels with a finite reference NIR < 0.05 or > "
                                "0.2 and outside the intertidal mask, pairs "
                                "finite in both scenes; exclusion mask = "
                                "SD(NDWI) > 0.2 from the UNcalibrated bands; "
                                "x_cal = (x - intercept) / slope; the reference "
                                "is left as is; no fallbacks"),
        "11_intertidal_mask": ("population SD (ddof = 0, Eq. 3) of NDWI = (G - "
                               "NIR) / (G + NIR) (Eq. 2) over the used scenes "
                               "from the CALIBRATED bands, per frame pixel, "
                               "over the scenes with a finite NDWI (no data "
                               "and G + NIR == 0 left out); mask = SD > 0.2; "
                               "the histogram valley is reported as a check only"),
        "12_eq1": ("Eq. 1 with the co-author's operation order: hHW - ((hHW - "
                   "hLW) * (cos(pi (T - TLW) / (THW - TLW)) + 1)) / 2, between "
                   "the bracketing marks (previous mark <= t < next mark)"),
        "13_boundary": ("the SAME boundary series the other methods use: "
                        "villaviciosa PyTMDBoundary('EOT20', *aoi.centroid, "
                        "directory='tide_models'); escalda / wadden / ems "
                        "make_boundary(aoi, tide_model='EOT20', n_gauges = 2 / "
                        "1 / 1, period=('2023-01-01','2025-12-31'), exclude = "
                        "['trnz'] / ['harl'] / ['delf']) after "
                        "pyintertidal.net.use_system_certificates()"),
        "14_marks": ("tide-table emulation: the boundary series sampled every "
                     "1 min (chronological) over the period +- 2 days; "
                     "turning points of diff() with zero differences "
                     "forward-filled (plateau -> one turn at its centre); "
                     "alternation HW / LW with >= 3 h between marks enforced "
                     "by deleting, smallest |dh| first, neighbouring pairs "
                     "closer than 3 h, which keeps the global extreme of each "
                     "half-cycle (checked)"),
        "15_rising_ebbing": ("rising iff the scene's bracketing pair goes LW -> "
                             "HW (previous mark is a LW), at lag 0; the split "
                             "stays fixed across lags"),
        "16_standardisation": ("rho = nplr::convertToProp(nir_cal) per pixel "
                               "over the used scenes (min-max, NAs ignored), "
                               "computed once and used by every fit"),
        "17_logistic": ("nplr(x = h, y = rho, useLog = FALSE, npars = 4) with "
                        "the defaults LPweight = 0.25, method 'res', run lean in "
                        "R; failure = nlm iterations == 0, constant fitted "
                        "values, or any error (try-error); elevation = "
                        "getInflexion x = xmid + log10(s)/scal; no other filter"),
        "18_preliminary_dem": "every pixel of the intertidal mask (frame-wide), heights at lag 0",
        "19_pool": ("pixels whose preliminary elevation is within h_c +- "
                    "0.25 m (inclusive), h_c = mean over the used scenes of "
                    "(h_HW + h_LW) / 2 of each scene's Eq. 1 bracket"),
        "20_sample": "n = min(50,000, pool), random without replacement, numpy default_rng(SEED)",
        "21_lags": ("grid -90..+90 min step 5 (37 lags); heights of every "
                    "scene from Eq. 1 at (t - lag) (positive = later local "
                    "tide); separate fits to the rising and to the ebbing "
                    "scenes; lag = first argmin of |h_rising - h_ebbing| over "
                    "the lags where both fits succeed (which.min); no lag "
                    "where no lag has both fits; the share of lags on the grid "
                    "edges is reported (primary grid only, no widening)"),
        "22_gam": ("mgcv::gam(lag ~ s(lon, lat)) with the defaults (bs 'tp', "
                   "k = 30, method 'GCV.Cp'), lon / lat = WGS84 pixel centres "
                   "of the sampled pixels with a lag; set.seed(SEED) before "
                   "the fit (the tprs knot subsample for n > 2000 uses mgcv's "
                   "own fixed seed xt$seed = 1); predicted at the pixels of step 23"),
        "23_final_dem": ("re-estimate ONLY the pixels whose preliminary fit "
                         "converged, each scene's height from Eq. 1 at (t - "
                         "lag_pixel) with the GAM lag in continuous minutes; "
                         "NaN where the final fit fails"),
        "24_exposure": ("Eq. 5 with C = mean duration of the used scenes' "
                        "bracketing cycles (previous mark to the next mark of "
                        "the same type), HW / LW = means of the scenes' "
                        "bracketing marks; no clipping (NaN outside [LW, HW]); "
                        "saved, not scored"),
    },
    "scoring": {
        "villaviciosa": ("RTK development split via pyintertidal.rtk.load_rtk "
                         "only; pyintertidal.elevation_validation."
                         "ols_against_truth; the final DEM sampled on the MAREA "
                         "extraction keep (products_marea/marea_extract.npz); "
                         "the published rtk_onsite_metrics.csv is first "
                         "reproduced to 1e-9"),
        "dutch": ("Vaklodingen code of the comparison notebooks' cell 30 (as "
                  "v7): nearest cell, in-range from z_plain +- 0.25 m, 5 x 5 "
                  "thinning, median-centred vscore; the final DEM sampled on "
                  "products_<site>/marea_extract.npz keep; published "
                  "vaklodingen_metrics.csv / by_band first reproduced to 1e-9"),
    },
    "seed": SEED,
    "proofs": ("every run: (a) lean vs nplr() on >= 2,000 real mask pixels of "
               "the preliminary fit (random, <= 20 % failures when available) "
               "and on >= 2,000 pixels of the final fit, y given to nplr() as "
               "the calibrated NIR with convertToProp in R; (b) lean vs nplr() "
               "on the rising / ebbing fits of real lag-sample pixels at all 37 "
               "lags; statuses identical and |inflection| difference <= 1e-12; "
               "(c) closed-form MA vs lmodel2 MA row on every calibration "
               "regression, |intercept| and |slope| differences <= 1e-10"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  small helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def other_heavy_processes(min_mb=300):
    """python / R processes outside this process tree using > min_mb."""
    import psutil
    me = psutil.Process()
    mine = {me.pid} | {c.pid for c in me.children(recursive=True)}
    out = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            if p.info["pid"] in mine or not any(k in name for k in ("python", "rscript", "rterm", "r.exe")):
                continue
            mb = p.info["memory_info"].rss / 2 ** 20
            if mb >= min_mb:
                out.append((p.info["pid"], name, round(mb),
                            " ".join((p.info["cmdline"] or [])[1:5])[:120]))
        except Exception:
            continue
    return out


def wait_for_ram(label, need_gb, args, poll_s=60):
    """Free-RAM guard before a heavy step: waits until at least ``need_gb``
    is available AND the other processes leave this job at least
    ``--min-free-gb`` (available + this process tree >= min_free_gb)."""
    if args.no_wait:
        return
    import psutil
    me = psutil.Process()
    t0 = time.time()
    waited = False
    while True:
        avail = psutil.virtual_memory().available / 2 ** 30
        own = 0.0
        for p in [me] + me.children(recursive=True):
            try:
                own += p.memory_info().rss / 2 ** 30
            except Exception:
                pass
        if avail >= need_gb and avail + own >= args.min_free_gb:
            if waited:
                log(f"  [ram] {label}: resumed after {time.time() - t0:.0f} s "
                    f"(available {avail:.2f} GB)")
            return
        waited = True
        log(f"  [ram] waiting before {label}: available {avail:.2f} GB, this job "
            f"{own:.2f} GB (need {need_gb:.2f} GB free and {args.min_free_gb:.1f} GB "
            f"for the job); other heavy processes: {other_heavy_processes()}")
        time.sleep(poll_s)


def auto_chunk(n, workers, max_chunk, min_chunk=20):
    """Chunk size giving every worker >= 3 jobs, within [min_chunk, max_chunk]."""
    if n <= 0:
        return max_chunk
    return int(max(min_chunk, min(max_chunk, math.ceil(n / (3 * workers)))))


class RowSubset:
    """Rows ``rows`` of a 2-D array, sliced lazily (no full copy)."""

    def __init__(self, base, rows):
        self.base, self.rows = base, np.asarray(rows, np.int64)
        self.shape = (len(self.rows), base.shape[1])

    def __getitem__(self, sl):
        return self.base[self.rows[sl]]


class LazyRows:
    """Rows ``rows`` of a matrix produced on demand by ``fn(rows_subset)``
    (used to build the standardised NIR of each R chunk from the int16
    values, bit-identical to a precomputed array, without holding it)."""

    def __init__(self, fn, rows, ncol):
        self.fn, self.rows = fn, np.asarray(rows, np.int64)
        self.shape = (len(self.rows), int(ncol))

    def __getitem__(self, sl):
        return self.fn(self.rows[sl])


class StageCache:
    """Finished R stages of an interrupted run (``--resume``), keyed by a
    hash of the stage inputs; removed after a successful run."""

    def __init__(self, directory, enabled):
        self.dir, self.enabled = directory, enabled
        if enabled:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def key(*arrays):
        h = hashlib.sha256()
        for a in arrays:
            a = np.ascontiguousarray(a)
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def get(self, name, key):
        p = os.path.join(self.dir, f"{name}.npz")
        if not (self.enabled and os.path.exists(p)):
            return None
        z = np.load(p)
        if str(z["key"]) != key:
            return None
        log(f"  [resume] {name}: reused {p}")
        return {k: z[k] for k in z.files if k != "key"}

    def put(self, name, key, **arrays):
        if self.enabled:
            np.savez(os.path.join(self.dir, f"{name}.npz"), key=np.array(key), **arrays)

    def clear(self):
        if self.enabled and os.path.isdir(self.dir):
            for f in os.listdir(self.dir):
                os.remove(os.path.join(self.dir, f))
            os.rmdir(self.dir)


# ─────────────────────────────────────────────────────────────────────────────
#  cube, record, boundary
# ─────────────────────────────────────────────────────────────────────────────

def cube_grid(cube_path):
    """Dates, pixel-centre coordinates, CRS and transform of a cube."""
    import pyproj
    import xarray as xr
    from rasterio.transform import Affine

    ds = xr.open_dataset(cube_path)
    try:
        t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
        for v in ("B03", "B08", "SCL"):
            if ds[v].dims != (t_dim, "y", "x"):
                raise ValueError(f"{v} dims {ds[v].dims} are not (t, y, x)")
        dates = np.array([str(np.datetime64(v, "D")) for v in ds[t_dim].values])
        xs, ys = ds["x"].values.astype(float), ds["y"].values.astype(float)
        wkt = ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref")
    finally:
        ds.close()
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    transform = Affine(dx, 0.0, float(xs[0]) - dx / 2, 0.0, dy, float(ys[0]) - dy / 2)
    return dict(dates=dates, xs=xs, ys=ys, H=len(ys), W=len(xs), wkt=wkt,
                crs=pyproj.CRS.from_wkt(wkt), transform=transform, t_dim=t_dim)


def band_encoding(ds, band):
    """The raw int16 encoding of a band: dtype, fill, no scaling (asserted)."""
    da = ds[band]
    if da.dtype != np.int16:
        raise AssertionError(f"{band} is {da.dtype}, expected int16")
    for a in ("scale_factor", "add_offset"):
        if a in da.attrs or a in da.encoding:
            raise AssertionError(f"{band} carries {a}: the /10000 scaling would be wrong")
    fill = da.attrs.get("_FillValue")
    if fill is None:
        raise AssertionError(f"{band} has no _FillValue")
    return int(fill)


def scan_scenes(cube_path, t_idx, log_every=100):
    """[4-5] Full-frame SCL cloud and no-data shares of the given scenes."""
    import xarray as xr

    ds = xr.open_dataset(cube_path, mask_and_scale=False)
    try:
        da = ds["SCL"]
        t_dim = da.dims[0]
        fill = band_encoding(ds, "SCL")
        cloud = np.empty(len(t_idx))
        nodata = np.empty(len(t_idx))
        t0 = time.time()
        for q, t in enumerate(t_idx):
            scl = da.isel({t_dim: int(t)}).values
            ok = (scl != fill)
            codes = np.unique(scl[ok])
            if np.any((codes < 0) | (codes > 11)):
                raise ValueError(f"scene {t}: SCL codes {codes} outside 0..11")
            cloud[q], nodata[q] = g21.scene_frame_shares(scl, fill)
            if (q + 1) % log_every == 0:
                log(f"    SCL scan {q + 1}/{len(t_idx)} ({time.time() - t0:.0f} s)")
    finally:
        ds.close()
    return cloud, nodata


def load_bands(cube_path, t_idx):
    """[1] B03 and B08 of the used scenes as raw int16 (T, H*W) and the fill."""
    import xarray as xr

    ds = xr.open_dataset(cube_path, mask_and_scale=False)
    try:
        t_dim = ds["B03"].dims[0]
        fill_g, fill_n = band_encoding(ds, "B03"), band_encoding(ds, "B08")
        if fill_g != fill_n:
            raise AssertionError("B03 and B08 fills differ")
        H, W = ds.sizes["y"], ds.sizes["x"]
        T = len(t_idx)
        G = np.empty((T, H * W), np.int16)
        N = np.empty((T, H * W), np.int16)
        for q, t in enumerate(t_idx):
            G[q] = ds["B03"].isel({t_dim: int(t)}).values.ravel()
            N[q] = ds["B08"].isel({t_dim: int(t)}).values.ravel()
    finally:
        ds.close()
    # the BOA offset: raw DN >= 1 -> value >= -999; DN 0 is the fill
    for name, a in (("B03", G), ("B08", N)):
        v = a[a != fill_g]
        if v.size and int(v.min()) < -1000:
            raise AssertionError(f"{name} values below -1000: not DN - 1000 harmonised data")
    return G, N, fill_g


def minutes_since_epoch(idx):
    """Naive-UTC datetimes -> float minutes since 1970-01-01 (ns-exact)."""
    return np.asarray(idx.values.astype("datetime64[ns]").astype(np.int64)) / 6e10


def to_utc_naive(v):
    import pandas as pd
    ts = pd.Timestamp(v)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def build_boundary(site, cfg):
    """[13] The boundary series object the other methods use at the site."""
    import pyintertidal as pit
    if site == "villaviciosa":
        from pyintertidal.boundary import PyTMDBoundary
        lat_c, lon_c = pit.sites.get("villaviciosa").centroid
        b = PyTMDBoundary("EOT20", lat_c, lon_c, directory="tide_models")
        return b, f"PyTMDBoundary EOT20 at aoi.centroid ({lat_c:.5f}, {lon_c:.5f})"
    from pyintertidal import net
    from pyintertidal.boundary import make_boundary
    net.use_system_certificates()
    b = make_boundary(pit.sites.get(site), tide_model="EOT20",
                      n_gauges=cfg["n_gauges"], period=TIME_EXTENT,
                      exclude=cfg["judge"])
    return b, getattr(b, "name", str(b))


def boundary_series(boundary, name, cache_path):
    """[14] The boundary sampled every 1 min, chronologically, over the
    period +- SERIES_PAD_DAYS, in SERIES_CHUNK_DAYS calls; cached."""
    import pandas as pd

    t0 = pd.Timestamp(TIME_EXTENT[0]) - pd.Timedelta(days=SERIES_PAD_DAYS)
    t1 = pd.Timestamp(TIME_EXTENT[1]) + pd.Timedelta(days=1 + SERIES_PAD_DAYS)
    idx = pd.date_range(t0, t1, freq="1min", inclusive="left")
    tag = f"{name}|{t0}|{t1}|1min"
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        if str(z["tag"]) == tag and len(z["h"]) == len(idx):
            log(f"  boundary series: cached {cache_path}")
            return minutes_since_epoch(idx), z["h"], True
    step = SERIES_CHUNK_DAYS * 1440
    h = np.empty(len(idx))
    tt = time.time()
    for a in range(0, len(idx), step):
        b = min(a + step, len(idx))
        v = np.asarray(boundary.levels(idx[a:b]), float)
        if v.shape != (b - a,):
            raise AssertionError("boundary.levels returned a wrong length")
        h[a:b] = v
        log(f"    boundary 1-min series {b:,}/{len(idx):,} ({time.time() - tt:.0f} s)")
    np.savez(cache_path, tag=np.array(tag), h=h)
    return minutes_since_epoch(idx), h, False


# ─────────────────────────────────────────────────────────────────────────────
#  proofs
# ─────────────────────────────────────────────────────────────────────────────

def compare_fits(lean_status, lean_infl, ref_status, ref_infl):
    """Equality of the lean path and nplr(): identical status codes and
    |inflection| difference <= PROOF_TOL where both succeeded."""
    ls, rs = np.asarray(lean_status), np.asarray(ref_status)
    li, ri = np.asarray(lean_infl), np.asarray(ref_infl)
    same_status = bool(np.array_equal(ls, rs))
    ok = (ls == 0) & (rs == 0)
    fin_mismatch = int(np.count_nonzero(np.isfinite(li[ok]) != np.isfinite(ri[ok])))
    both = ok & np.isfinite(li) & np.isfinite(ri)
    d = np.abs(li[both] - ri[both])
    out = {"n_fits": int(ls.size), "n_ok": int(np.count_nonzero(ls == 0)),
           "n_failed": int(np.count_nonzero(ls != 0)),
           "failure_codes_lean": {str(int(k)): int(v) for k, v in zip(*np.unique(ls[ls != 0], return_counts=True))},
           "failure_codes_nplr": {str(int(k)): int(v) for k, v in zip(*np.unique(rs[rs != 0], return_counts=True))},
           "status_identical": same_status,
           "finite_mismatch": fin_mismatch,
           "max_abs_diff_inflection": float(d.max()) if d.size else 0.0,
           "n_bitwise_identical": int(np.count_nonzero(li[both] == ri[both])),
           "tolerance": PROOF_TOL}
    out["passed"] = bool(same_status and fin_mismatch == 0
                         and out["max_abs_diff_inflection"] <= PROOF_TOL)
    return out


def proof_selection(ok_mask, n_total, rng, max_fail_share=0.2):
    """Random proof pixels: up to max_fail_share of failures, the rest successes."""
    ok_idx = np.flatnonzero(ok_mask)
    bad_idx = np.flatnonzero(~ok_mask)
    n_bad = min(len(bad_idx), int(n_total * max_fail_share))
    n_ok = min(len(ok_idx), n_total - n_bad)
    sel = np.concatenate([rng.choice(ok_idx, n_ok, replace=False),
                          rng.choice(bad_idx, n_bad, replace=False)])
    return np.sort(sel)


# ─────────────────────────────────────────────────────────────────────────────
#  scoring
# ─────────────────────────────────────────────────────────────────────────────

RTK_COLS = ("n", "rmse_m", "slope", "intercept", "r", "bias_m")


def score_rtk(path, final_frame, keep, shape, n_scenes):
    """Villaviciosa RTK development split (rtk.load_rtk only)."""
    import pandas as pd
    from pyintertidal import elevation_validation as ev
    from pyintertidal import rtk

    H, W = shape
    dev = rtk.load_rtk(RTK_CSV)
    r_, c_, zt = dev["row"], dev["col"], dev["elev"]
    E = np.load("products_marea/external_products.npz")
    old = {"no delay": E["no_delay"], "MAREA": E["MAREA"], "Granadeiro": E["Granadeiro"]}
    ref = pd.read_csv("products_marea/rtk_onsite_metrics.csv").set_index("product")
    worst = 0.0
    for name, ras in old.items():
        assert ras.shape == (H, W)
        f = ev.ols_against_truth(zt, ras[r_, c_])
        assert f["n"] == int(ref.loc[name, "n"]), (name, f["n"])
        for k in RTK_COLS[1:]:
            worst = max(worst, abs(f[k] - float(ref.loc[name, k])))
    assert worst <= TOL, f"rtk_onsite_metrics.csv not reproduced ({worst})"
    log(f"  reproduced rtk_onsite_metrics.csv: max |diff| {worst:.2e}")

    on_keep = np.full(H * W, np.nan)
    on_keep[keep] = np.asarray(final_frame).ravel()[keep]
    samp = {name: np.asarray(ras[r_, c_], float) for name, ras in old.items()}
    samp[PRODUCT] = on_keep.reshape(H, W)[r_, c_]
    samp[PRODUCT + " full frame"] = np.asarray(final_frame, float)[r_, c_]
    records = {
        "no delay": "scored 97-scene 2023-2025 record (notebook cell 52)",
        "MAREA": "scored 97-scene 2023-2025 record (notebook cell 52)",
        "Granadeiro": ("OLD implementation (experiments/sota_granadeiro.py): 98 "
                       "scenes, NIR <= 0 dropped, MAREA keep as candidates, "
                       "stride/least-squares/TPS deviations (audit)"),
        PRODUCT: (f"faithful reimplementation: {n_scenes} scenes of the frame "
                  "cloud rule (< 10 % SCL 3/8/9/10, <= 2 % no data), 2023-2025, "
                  "EOT20 at aoi.centroid, Eq. 1 marks"),
    }
    records[PRODUCT + " full frame"] = records[PRODUCT]
    fin = {k: np.isfinite(v) for k, v in samp.items()}

    def row(product, subset, role, sel):
        v = np.where(sel, samp[product], np.nan)
        f = ev.ols_against_truth(zt, v)
        out = {"product": product, "subset": subset, "role": role,
               "record": records[product]}
        if f is None:
            out.update({k: np.nan for k in RTK_COLS})
            out["n"] = int(np.sum(np.isfinite(v) & np.isfinite(zt)))
        else:
            out.update({k: f[k] for k in RTK_COLS})
        return out

    allp = np.ones(len(zt), bool)
    both_g = fin["Granadeiro"] & fin[PRODUCT]
    both_m = fin["MAREA"] & fin[PRODUCT]
    common4 = fin["no delay"] & fin["MAREA"] & fin["Granadeiro"] & fin[PRODUCT]
    rows = [row("no delay", "all", "reproduced (published)", allp),
            row("MAREA", "all", "reproduced (published)", allp),
            row("Granadeiro", "all", "reproduced (published); OLD implementation", allp),
            row(PRODUCT, "all", "PRIMARY: final DEM on the MAREA extraction keep", allp),
            row(PRODUCT + " full frame", "all",
                "supplementary: final DEM at every RTK point (not restricted to keep)", allp)]
    for p in ("Granadeiro", PRODUCT):
        rows.append(row(p, f"pair (Granadeiro & {PRODUCT})", "pairwise subset", both_g))
    for p in ("MAREA", PRODUCT):
        rows.append(row(p, f"pair (MAREA & {PRODUCT})", "pairwise subset", both_m))
    for p in ("no delay", "MAREA", "Granadeiro", PRODUCT):
        rows.append(row(p, f"common4 (no delay & MAREA & Granadeiro & {PRODUCT})",
                        "common subset", common4))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log(df[["product", "subset", "n", "rmse_m", "slope", "r", "bias_m"]].round(4)
        .to_string(index=False))
    return {"csv": path, "dev_points": int(len(zt)),
            "dev_points_reserved_hidden": int(dev["n_reserved_hidden"]),
            "dev_points_in_keep": int(np.isin(r_ * W + c_, keep).sum()),
            "finite_points": {k: int(v.sum()) for k, v in fin.items()},
            "reproduction_max_abs_diff": worst,
            "rows": [{k: (v if not isinstance(v, float) or np.isfinite(v) else None)
                      for k, v in r.items()} for r in rows]}


def score_vaklodingen(site, cfg, final_keep, keep, grid, out_dir):
    """Dutch Vaklodingen scoring, cell 30 of the comparison notebooks (as v7).
    Written for escalda / wadden / ems; not run in this session."""
    import pandas as pd
    import pyproj
    from experiments import v1_vaklodingen_validate as v1
    from experiments.v7_waterline_contour import vscore

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

    # reproduce the published tables first
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
    log(f"  reproduced vaklodingen_metrics.csv: max |diff| {worst:.2e}")
    band_of = P["band"].astype(int)
    bb = pd.read_csv(f"{d}/vaklodingen_by_band.csv")
    worst_b = 0.0
    for name_, v_ in vprods.items():
        for _, rr in bb[bb["product"] == name_].iterrows():
            s_ = vscore(np.where(band_of == int(rr["band"]), v_, np.nan), truth_v) or {}
            for kk in ("n", "rmse_m", "slope", "r", "bias_m"):
                x, y = s_.get(kk, np.nan), float(rr[kk])
                if np.isnan(x) != np.isnan(y):
                    raise AssertionError(f"by-band NaN mismatch {name_} {kk}")
                if not np.isnan(x):
                    worst_b = max(worst_b, abs(float(x) - y))
    assert worst_b <= TOL, f"vaklodingen_by_band.csv not reproduced ({worst_b})"

    allp = {**vprods, PRODUCT: np.asarray(final_keep, float)}
    tv = np.isfinite(truth_v)
    fin = {k: np.isfinite(v) for k, v in allp.items()}
    ones = np.ones(len(keep), bool)
    common4 = tv & np.logical_and.reduce([fin[p] for p in base]) & fin[PRODUCT]
    pair_m = tv & fin[PRODUCT] & fin["MAREA"]
    pair_g = (tv & fin[PRODUCT] & fin["Granadeiro"]) if "Granadeiro" in fin else np.zeros(len(keep), bool)

    def lrow(product, subset, role, sel):
        s_ = vscore(np.where(sel, allp[product], np.nan), truth_v) or {}
        return {"product": product, "subset": subset, "role": role,
                "n": s_.get("n", int(np.sum(sel & fin[product] & tv))),
                "rmse_m": s_.get("rmse_m", np.nan), "slope": s_.get("slope", np.nan),
                "r": s_.get("r", np.nan), "bias_m": s_.get("bias_m", np.nan)}

    rows = []
    for p in base:
        rows.append(lrow(p, "all", "reproduced (published)", ones))
        rows.append(lrow(p, thin_label, "reproduced (published)", thin))
        rows.append(lrow(p, ref_label,
                         "reproduced (published)", vcommon))
    rows.append(lrow(PRODUCT, "all", "PRIMARY: final DEM on keep", ones))
    rows.append(lrow(PRODUCT, thin_label, "PRIMARY: final DEM on keep", thin))
    for p in (*base, PRODUCT):
        rows.append(lrow(p, f"common{len(base) + 1} ({' & '.join(base)} & {PRODUCT})",
                         "common subset", common4))
    for p in (PRODUCT, "MAREA"):
        rows.append(lrow(p, f"pair ({PRODUCT} & MAREA)", "pairwise subset", pair_m))
    if "Granadeiro" in fin:
        for p in (PRODUCT, "Granadeiro"):
            rows.append(lrow(p, f"pair ({PRODUCT} & Granadeiro)", "pairwise subset", pair_g))
    path = os.path.join(out_dir, "vaklodingen_metrics_granadeiro2021.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    band_rows = []
    bands = bb[bb["product"] == "no delay"][["band", "s_km", "tau_applied_min"]]
    for name_ in (*base, PRODUCT):
        for _, rr in bands.iterrows():
            k = int(rr["band"])
            s_ = vscore(np.where(band_of == k, allp[name_], np.nan), truth_v) or {}
            band_rows.append({"product": name_, "band": k, "s_km": float(rr["s_km"]),
                              "tau_applied_min": float(rr["tau_applied_min"]),
                              **{kk: s_.get(kk, np.nan) for kk in
                                 ("n", "rmse_m", "slope", "r", "bias_m")}})
    bpath = os.path.join(out_dir, "vaklodingen_by_band_granadeiro2021.csv")
    pd.DataFrame(band_rows).to_csv(bpath, index=False)
    log(pd.DataFrame(rows)[["product", "subset", "n", "rmse_m", "slope", "r"]]
        .round(4).to_string(index=False))
    return {"csv": [path, bpath], "reproduction_max_abs_diff": worst,
            "reproduction_by_band_max_abs_diff": worst_b,
            "truth_px_on_keep": int(np.isfinite(truth_raw).sum()),
            "in_range_from_z_plain": [lo_ - 0.25, hi_ + 0.25],
            "common4_px": int(common4.sum()), "pair_marea_px": int(pair_m.sum()),
            "pair_old_granadeiro_px": int(pair_g.sum()),
            "finite_with_truth": {k: int((v & tv).sum()) for k, v in fin.items()}}


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def quantiles(v, q=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"p{k}": float(np.percentile(v, k)) for k in q} if v.size else {}


def status_counts(st):
    u, c = np.unique(np.asarray(st), return_counts=True)
    names = {0: "ok", 1: "nlm_iterations_0", 2: "constant_fit", 3: "error"}
    return {names.get(int(k), str(k)): int(v) for k, v in zip(u, c)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site", choices=sorted(SITES))
    ap.add_argument("--workers", type=int, default=8, help="parallel R workers (default 8)")
    ap.add_argument("--tmpdir", default=None, help="parent directory of the R exchange files")
    ap.add_argument("--rscript", default=None, help="Rscript executable")
    ap.add_argument("--min-free-gb", type=float, default=2.5,
                    help="RAM other processes must leave for this job (default 2.5)")
    ap.add_argument("--no-wait", action="store_true", help="skip the free-RAM guard")
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished R stages of an interrupted run (same inputs)")
    ap.add_argument("--proof-pixels", type=int, default=2500)
    ap.add_argument("--proof-lag-pixels", type=int, default=100)
    ap.add_argument("--chunk-fit", type=int, default=2000)
    ap.add_argument("--chunk-lag", type=int, default=100)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args(argv)
    if args.proof_pixels < 2000:
        raise SystemExit("--proof-pixels must be >= 2000 (author's rule)")
    site, cfg = args.site, SITES[args.site]
    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    timings = {}
    log(f"[v8] Granadeiro et al. 2021 (faithful) at {site} -> {out_dir}; "
        f"{args.workers} R workers")
    wait_for_ram("start", args.min_free_gb, args)

    # ── grid, extraction keep ────────────────────────────────────────────
    t0 = time.time()
    grid = cube_grid(cfg["cube"])
    H, W, P = grid["H"], grid["W"], grid["H"] * grid["W"]
    z_ = np.load(cfg["extract"], allow_pickle=True)
    keep = z_["keep"].astype(np.int64)
    ext_dates = np.array([str(d) for d in z_["dates"]])
    ext_shape = tuple(int(v) for v in z_["shape"])
    del z_
    assert ext_shape == (H, W), (ext_shape, (H, W))
    assert np.array_equal(ext_dates, grid["dates"]), "extraction dates != cube dates"
    assert np.all(np.diff(keep) > 0) and keep[0] >= 0 and keep[-1] < P
    if site == "villaviciosa":
        import rasterio
        with rasterio.open(RTK_GRID) as s_:
            assert s_.shape == (H, W) and s_.transform.almost_equals(grid["transform"]), \
                "cube grid differs from the RTK grid"
    log(f"  grid {H} x {W} ({P:,} px), {len(grid['dates'])} cube scenes, keep {len(keep):,} px")

    # ── [3-6] scenes ─────────────────────────────────────────────────────
    import pandas as pd
    ov = json.load(open(cfg["overpass"], encoding="utf-8"))
    dates = grid["dates"]
    in_period = (dates >= TIME_EXTENT[0]) & (dates <= TIME_EXTENT[1])
    has_time = np.array([d in ov for d in dates])
    cand = np.flatnonzero(in_period & has_time)
    wait_for_ram("SCL scan", 0.5, args)
    cloud_c, nodata_c = scan_scenes(cfg["cube"], cand)
    use_c = g21.scene_rule(cloud_c, nodata_c, np.ones(len(cand), bool), np.ones(len(cand), bool))
    t_idx = cand[use_c]
    T = len(t_idx)
    timings["scene_scan_s"] = time.time() - t0
    log(f"  scene rule: {T} of {len(cand)} 2023-2025 scenes with an overpass time "
        f"(cloud < {g21.SCENE_CLOUD_MAX}, no data <= {g21.SCENE_NODATA_MAX}); "
        f"{int(in_period.sum() - len(cand))} period scenes lack a time")
    if T < 8:
        raise SystemExit(f"only {T} scenes pass the rule")
    u_dates = dates[t_idx]
    t_ts = pd.DatetimeIndex([to_utc_naive(ov[d]) for d in u_dates])
    if not t_ts.is_monotonic_increasing:
        raise AssertionError("overpass times are not chronological")
    t_scene = minutes_since_epoch(t_ts)           # minutes since 1970 (UTC)

    # ── [12-15] boundary, marks, Eq. 1 ───────────────────────────────────
    t0 = time.time()
    boundary, bname = build_boundary(site, cfg)
    wait_for_ram("boundary series", 1.0, args)
    series_t, series_h, cached = boundary_series(
        boundary, bname, os.path.join(out_dir, "boundary_1min_cache.npz"))
    marks = g21.tide_marks(series_t, series_h)
    del series_t, series_h
    h_scene, br = g21.water_height(t_scene, marks)
    rising = br["rising"].copy()
    cyc_h = g21.scene_cycles_hours(t_scene, marks)
    h_c = float(np.mean((br["h_hw"] + br["h_lw"]) / 2.0))
    hw_mean, lw_mean = float(np.mean(br["h_hw"])), float(np.mean(br["h_lw"]))
    cycle_mean = float(np.mean(cyc_h))
    H_lag = g21.lag_heights(t_scene, marks)
    assert np.array_equal(H_lag[list(g21.LAG_GRID_MIN).index(0.0)], h_scene)
    timings["boundary_marks_s"] = time.time() - t0
    marks_info = {k: v for k, v in marks.items() if k not in ("t", "h", "is_hw", "sample_index")}
    log(f"  boundary {bname}: {marks['n_marks']} marks ({marks['n_hw']} HW / "
        f"{marks['n_lw']} LW) from {marks['n_turning_points']} turning points "
        f"({marks['n_pairs_removed']} pairs < 3 h removed; not-global-extreme "
        f"{marks['n_not_global_extreme']})")
    log(f"  Eq. 1 at the scenes: h [{h_scene.min():+.3f}, {h_scene.max():+.3f}] m, "
        f"{int(rising.sum())} rising / {int((~rising).sum())} ebbing; h_c = {h_c:+.3f} m; "
        f"HW {hw_mean:+.3f} / LW {lw_mean:+.3f} m; cycle {cycle_mean:.3f} h")

    # ── [1] bands ────────────────────────────────────────────────────────
    t0 = time.time()
    wait_for_ram("band load", 2 * T * P * 2 / 2 ** 30 + 0.5, args)
    G_raw, N_raw, fill = load_bands(cfg["cube"], t_idx)
    n_nodata_g = int(np.count_nonzero(G_raw == fill))
    n_nodata_n = int(np.count_nonzero(N_raw == fill))
    n_nir_le0 = int(np.count_nonzero((N_raw <= 0) & (N_raw != fill)))
    timings["band_load_s"] = time.time() - t0
    log(f"  bands: 2 x {T} x {P:,} int16 ({2 * G_raw.nbytes / 2 ** 20:.0f} MB), fill {fill}; "
        f"NIR <= 0 (kept) {n_nir_le0:,} obs ({n_nir_le0 / N_raw.size:.1%})")

    # ── [11 for 10] uncalibrated SD mask, [9] reference, [10] calibration ─
    t0 = time.time()
    sd_unc, cnt_unc, nf_unc = g21.frame_ndwi_sd(G_raw, N_raw, fill)
    with np.errstate(invalid="ignore"):
        mask_unc = sd_unc > g21.SD_NDWI_THRESHOLD
    rg, qg = g21.band_ranges(G_raw, fill)
    rn, qn = g21.band_ranges(N_raw, fill)
    ref_pos, ref_score = g21.reference_scene(rg, rn)
    nir_ref = g21.to_reflectance(N_raw[ref_pos], fill)
    cal_idx, n_sea, n_land = g21.calibration_pixels(nir_ref, mask_unc)
    del nir_ref
    coef = {}
    ncal = {}
    for band, raw in (("B03", G_raw), ("B08", N_raw)):
        x = g21.to_reflectance(raw[ref_pos, cal_idx], fill)
        c = np.zeros((T, 2))
        c[:, 1] = 1.0
        nn = np.zeros(T, np.int64)
        for i in range(T):
            if i == ref_pos:
                nn[i] = int(np.isfinite(x).sum())
                continue
            b0, b1, n_ = g21.major_axis(x, g21.to_reflectance(raw[i, cal_idx], fill))
            if not (np.isfinite(b0) and np.isfinite(b1)):
                raise AssertionError(f"{band} scene {u_dates[i]}: lmodel2 MA is NA (r^2 <= eps)")
            c[i] = (b0, b1)
            nn[i] = n_
        coef[band], ncal[band] = c, nn
    timings["calibration_py_s"] = time.time() - t0
    log(f"  uncalibrated SD mask {int(mask_unc.sum()):,} px; reference scene "
        f"{u_dates[ref_pos]} (score {ref_score[ref_pos]:.4f}); calibration pixels "
        f"{len(cal_idx):,} ({n_sea:,} sea, {n_land:,} land); slopes B08 "
        f"[{coef['B08'][:, 1].min():.3f}, {coef['B08'][:, 1].max():.3f}], "
        f"B03 [{coef['B03'][:, 1].min():.3f}, {coef['B03'][:, 1].max():.3f}]")

    # calibrated SD mask
    t0 = time.time()
    sd_cal, cnt_cal, nf_cal = g21.frame_ndwi_sd(G_raw, N_raw, fill, coef["B03"], coef["B08"])
    with np.errstate(invalid="ignore"):
        mask = sd_cal > g21.SD_NDWI_THRESHOLD
    mask_idx = np.flatnonzero(mask)
    n_mask = len(mask_idx)
    hist = g21.sd_histogram_valleys(sd_cal)
    in_keep = np.zeros(P, bool)
    in_keep[keep] = True
    log(f"  intertidal mask (SD(NDWI) > 0.2, calibrated): {n_mask:,} px "
        f"({n_mask / P:.1%} of the frame); keep in mask {int(mask[keep].sum()):,} of "
        f"{len(keep):,}; mask outside keep {int((mask & ~in_keep).sum()):,}; "
        f"histogram peaks {hist['peaks_sd']}, valley nearest 0.2 "
        f"{hist.get('valley_closest_to_threshold')}")

    # [16] the NIR of the mask pixels, pixel-major int16 (a copy); the
    # calibrated NIR and rho = convertToProp(nir_cal) of any rows are built
    # from it on demand (the same float64 operations -> bit-identical)
    N_pm = np.ascontiguousarray(N_raw[:, mask_idx].T)
    timings["mask_s"] = time.time() - t0

    def nir_cal_rows(rows):
        """Calibrated NIR (pixel-major, float64) of mask positions ``rows``."""
        return g21.calibrate(g21.to_reflectance(N_pm[rows].T, fill), coef["B08"]).T

    def rho_rows(rows):
        """[16] rho = nplr::convertToProp(nir_cal) per pixel over the scenes."""
        return g21.convert_to_prop(nir_cal_rows(rows), axis=1)

    rng_proof = np.random.default_rng(SEED + 1)
    cache = StageCache(os.path.join(out_dir, "_resume"), args.resume)
    runner = g21.RRunner(workers=args.workers, tmpdir=args.tmpdir,
                         rscript=args.rscript, log=log)
    proofs = {}
    try:
        versions = g21.r_versions(runner)
        log(f"  R: {versions}")

        # (c) MA proof: every calibration regression against lmodel2
        t0 = time.time()
        wait_for_ram("lmodel2 proof", 1.0, args)
        ma = {}
        for band, raw in (("B03", G_raw), ("B08", N_raw)):
            x = g21.to_reflectance(raw[ref_pos, cal_idx], fill)
            others = [i for i in range(T) if i != ref_pos]
            rows = g21.lmodel2_rows(
                runner, x, lambda k, raw=raw, others=others:
                g21.to_reflectance(raw[others[k], cal_idx], fill), len(others),
                jobs_per_process=2, desc=f"lmodel2_{band}",
                workers=min(args.workers, 3))
            d0 = np.abs(rows[:, 3] - coef[band][others, 0])
            d1 = np.abs(rows[:, 4] - coef[band][others, 1])
            ma[band] = {"n_regressions": len(others),
                        "n_pairs_identical": bool(np.array_equal(rows[:, 0], ncal[band][others])),
                        "max_abs_diff_intercept": float(d0.max()),
                        "max_abs_diff_slope": float(d1.max()),
                        "lmodel2_rows": rows.tolist()}
        ma["passed"] = all(ma[b]["n_pairs_identical"] and ma[b]["max_abs_diff_intercept"] <= MA_TOL
                           and ma[b]["max_abs_diff_slope"] <= MA_TOL for b in ("B03", "B08"))
        ma["tolerance"] = MA_TOL
        proofs["ma_vs_lmodel2"] = {k: (v if k not in ("B03", "B08") else
                                       {kk: vv for kk, vv in v.items() if kk != "lmodel2_rows"})
                                   for k, v in ma.items()}
        timings["lmodel2_proof_s"] = time.time() - t0
        log(f"  MA vs lmodel2 on {ma['B03']['n_regressions'] + ma['B08']['n_regressions']} "
            f"regressions: max |d intercept| {max(ma[b]['max_abs_diff_intercept'] for b in ('B03', 'B08')):.2e}, "
            f"max |d slope| {max(ma[b]['max_abs_diff_slope'] for b in ('B03', 'B08')):.2e} -> "
            f"{'PASSED' if ma['passed'] else 'FAILED'}")
        if not ma["passed"]:
            raise AssertionError("closed-form MA differs from lmodel2")
        lmodel2_rows_saved = {b: np.array(ma[b]["lmodel2_rows"]) for b in ("B03", "B08")}
        del G_raw, N_raw
        gc.collect()

        # ── [17-18] preliminary DEM ──────────────────────────────────────
        t0 = time.time()
        wait_for_ram("preliminary DEM", 0.15 * args.workers + 0.3, args)
        key = StageCache.key(h_scene, N_pm, coef["B08"])
        got = cache.get("prelim", key)
        if got is None:
            fit_p = g21.fit_logistic(runner, LazyRows(rho_rows, np.arange(n_mask), T), x=h_scene,
                                     chunk=auto_chunk(n_mask, args.workers, args.chunk_fit),
                                     desc="prelim")
            cache.put("prelim", key, fit=fit_p)
        else:
            fit_p = got["fit"]
        st_p = fit_p[:, 0].astype(np.int8)
        conv_p = (st_p == 0) & np.isfinite(fit_p[:, 1])
        prelim_frame = np.full(P, np.nan)
        prelim_frame[mask_idx[conv_p]] = fit_p[conv_p, 1]
        timings["prelim_fit_s"] = time.time() - t0
        log(f"  preliminary DEM: {int(conv_p.sum()):,} of {n_mask:,} mask px converged "
            f"({conv_p.mean():.1%}); {status_counts(st_p)}; "
            f"{time.time() - t0:.0f} s")

        # (a) proof on the preliminary fits
        t0 = time.time()
        sel = proof_selection(conv_p, args.proof_pixels, rng_proof)
        ref_p = g21.fit_logistic(runner, nir_cal_rows(sel), x=h_scene, reference=True, prop=True,
                                 chunk=auto_chunk(len(sel), args.workers, 500),
                                 desc="proof_prelim")
        proofs["prelim_lean_vs_nplr"] = compare_fits(fit_p[sel, 0], fit_p[sel, 1],
                                                     ref_p[:, 0], ref_p[:, 1])
        timings["proof_prelim_s"] = time.time() - t0
        log(f"  PROOF lean vs nplr() (preliminary, {len(sel)} px): {proofs['prelim_lean_vs_nplr']}")
        if not proofs["prelim_lean_vs_nplr"]["passed"]:
            raise AssertionError("lean nplr path differs from nplr() (preliminary)")
        fixture_rows = sel[:300]
        fixture = dict(proof_fixture_x=h_scene, proof_fixture_nir_cal=nir_cal_rows(fixture_rows),
                       proof_fixture_rho=rho_rows(fixture_rows),
                       proof_fixture_lean=fit_p[fixture_rows][:, :2],
                       proof_fixture_nplr=ref_p[:300, :2])

        # ── [19-21] lag sample and lags ──────────────────────────────────
        t0 = time.time()
        with np.errstate(invalid="ignore"):
            pool = np.flatnonzero(conv_p & (np.abs(fit_p[:, 1] - h_c) <= g21.POOL_HALF_WIDTH_M))
        n_s = min(g21.N_LAG_SAMPLE, len(pool))
        rng = np.random.default_rng(SEED)
        samp = rng.choice(pool, n_s, replace=False)
        log(f"  lag pool {len(pool):,} px (|prelim - h_c| <= 0.25 m); sample {n_s:,}; "
            f"{T} scenes x {len(g21.LAG_GRID_MIN)} lags x 2 limbs")
        if n_s == 0:
            raise AssertionError("empty lag pool")
        wait_for_ram("lag search", 0.15 * args.workers + 0.3, args)
        key = StageCache.key(H_lag, rising, N_pm[samp], coef["B08"])
        got = cache.get("lags", key)
        if got is None:
            st_l, inf_l = g21.lag_fits(runner, LazyRows(rho_rows, samp, T), H_lag, rising,
                                       chunk=auto_chunk(n_s, args.workers, args.chunk_lag, 5),
                                       desc="lag")
            cache.put("lags", key, st=st_l, inf=inf_l)
        else:
            st_l, inf_l = got["st"], got["inf"]
        lag_s, gap_s, idx_s = g21.choose_lags(st_l, inf_l)
        has_lag = np.isfinite(lag_s)
        edge_lo = int(np.sum(lag_s == g21.LAG_GRID_MIN[0]))
        edge_hi = int(np.sum(lag_s == g21.LAG_GRID_MIN[-1]))
        timings["lag_search_s"] = time.time() - t0
        lag_counts = {f"{v:+.0f}": int(np.sum(lag_s == v)) for v in g21.LAG_GRID_MIN}
        log(f"  lags: {int(has_lag.sum()):,} of {n_s:,} sample px; median "
            f"{np.nanmedian(lag_s):+.0f} min; edges -90: {edge_lo}, +90: {edge_hi} "
            f"({(edge_lo + edge_hi) / max(int(has_lag.sum()), 1):.1%}); {time.time() - t0:.0f} s")

        # (b) proof on the lag fits
        t0 = time.time()
        k_proof = min(args.proof_lag_pixels, n_s)
        sel_l = np.sort(rng_proof.choice(n_s, k_proof, replace=False))
        st_r, inf_r = g21.lag_fits(runner, rho_rows(samp[sel_l]), H_lag, rising, reference=True,
                                   chunk=auto_chunk(k_proof, args.workers, 20, 2),
                                   desc="proof_lag")
        proofs["lag_lean_vs_nplr"] = compare_fits(st_l[sel_l], inf_l[sel_l], st_r, inf_r)
        proofs["lag_lean_vs_nplr"]["n_pixels"] = int(k_proof)
        timings["proof_lag_s"] = time.time() - t0
        log(f"  PROOF lean vs nplr() (lag fits, {k_proof} px x 37 lags x 2): "
            f"{proofs['lag_lean_vs_nplr']}")
        if not proofs["lag_lean_vs_nplr"]["passed"]:
            raise AssertionError("lean nplr path differs from nplr() (lag fits)")

        # ── [22] GAM ─────────────────────────────────────────────────────
        t0 = time.time()
        import pyproj
        to_ll = pyproj.Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)

        def lonlat(flat):
            return to_ll.transform(grid["xs"][flat % W], grid["ys"][flat // W])

        conv_idx = mask_idx[conv_p]                   # step-23 pixels (flat)
        lon_s, lat_s = lonlat(mask_idx[samp[has_lag]])
        lon_p, lat_p = lonlat(conv_idx)
        wait_for_ram("GAM", 1.0, args)
        gam_pred, gam_fit, gam_info = g21.gam_lag_surface(
            runner, lon_s, lat_s, lag_s[has_lag], lon_p, lat_p, SEED)
        gam_frame = np.full(P, np.nan)
        gam_frame[conv_idx] = gam_pred
        timings["gam_s"] = time.time() - t0
        log(f"  GAM lag ~ s(lon, lat): n {gam_info.get('n')}, edf {gam_info.get('edf_smooth'):.2f}, "
            f"GCV {gam_info.get('gcv_score'):.2f}, R2 adj {gam_info.get('r_sq_adj'):.3f}; "
            f"prediction [{gam_pred.min():+.1f}, {gam_pred.max():+.1f}] min on {len(conv_idx):,} px")

        # ── [23] final DEM ───────────────────────────────────────────────
        t0 = time.time()
        conv_rows = np.flatnonzero(conv_p)            # mask positions

        def x_final(a, b):
            tt = t_scene[None, :] - gam_pred[a:b, None]
            return g21.water_height(tt, marks)[0]

        wait_for_ram("final DEM", 0.15 * args.workers + 0.3, args)
        key = StageCache.key(gam_pred, t_scene, marks["t"], marks["h"], N_pm[conv_rows],
                             coef["B08"])
        got = cache.get("final", key)
        if got is None:
            fit_f = g21.fit_logistic(runner, LazyRows(rho_rows, conv_rows, T), x_fn=x_final,
                                     chunk=auto_chunk(len(conv_rows), args.workers, args.chunk_fit),
                                     desc="final")
            cache.put("final", key, fit=fit_f)
        else:
            fit_f = got["fit"]
        st_f = fit_f[:, 0].astype(np.int8)
        ok_f = (st_f == 0) & np.isfinite(fit_f[:, 1])
        final_frame = np.full(P, np.nan)
        final_frame[conv_idx[ok_f]] = fit_f[ok_f, 1]
        timings["final_fit_s"] = time.time() - t0
        log(f"  final DEM: {int(ok_f.sum()):,} of {len(conv_rows):,} re-estimated px "
            f"({ok_f.mean():.2%}); {status_counts(st_f)}; {time.time() - t0:.0f} s")

        # (a) proof on the final fits (per-pixel heights)
        t0 = time.time()
        sel_f = proof_selection(ok_f, args.proof_pixels, rng_proof)
        ref_f = g21.fit_logistic(runner, nir_cal_rows(conv_rows[sel_f]),
                                 x_fn=lambda a, b: g21.water_height(
                                     t_scene[None, :] - gam_pred[sel_f[a:b], None], marks)[0],
                                 reference=True, prop=True,
                                 chunk=auto_chunk(len(sel_f), args.workers, 500),
                                 desc="proof_final")
        proofs["final_lean_vs_nplr"] = compare_fits(fit_f[sel_f, 0], fit_f[sel_f, 1],
                                                    ref_f[:, 0], ref_f[:, 1])
        timings["proof_final_s"] = time.time() - t0
        log(f"  PROOF lean vs nplr() (final, {len(sel_f)} px): {proofs['final_lean_vs_nplr']}")
        if not proofs["final_lean_vs_nplr"]["passed"]:
            raise AssertionError("lean nplr path differs from nplr() (final)")
        runner_stats = dict(runner.stats)
    finally:
        runner.close()
    tmp_removed = not os.path.exists(runner.root)

    # ── [24] exposure ────────────────────────────────────────────────────
    expo = g21.exposure_eq5(final_frame, hw_mean, lw_mean, cycle_mean)

    # ── outputs ──────────────────────────────────────────────────────────
    t0 = time.time()
    final_keep = final_frame[keep]
    npz_path = os.path.join(out_dir, "granadeiro2021.npz")
    np.savez_compressed(
        npz_path,
        site=np.array(site), keep=keep, shape=np.array([H, W]),
        candidate_dates=dates[cand], candidate_cloud_share=cloud_c,
        candidate_nodata_share=nodata_c, candidate_used=use_c,
        scene_dates=u_dates, scene_time_index=t_idx,
        scene_time_utc=np.array([str(t) for t in t_ts]), scene_t_min=t_scene,
        scene_h=h_scene, scene_rising=rising, scene_h_hw=br["h_hw"], scene_h_lw=br["h_lw"],
        scene_t_hw_min=br["t_hw"], scene_t_lw_min=br["t_lw"], scene_cycle_h=cyc_h,
        marks_t_min=marks["t"], marks_h=marks["h"], marks_is_hw=marks["is_hw"],
        lag_grid_min=g21.LAG_GRID_MIN, lag_heights=H_lag,
        reference_pos=np.array(ref_pos), reference_date=np.array(u_dates[ref_pos]),
        range_green=rg, range_nir=rn, reference_score=ref_score,
        cal_B03=coef["B03"], cal_B08=coef["B08"], cal_n_B03=ncal["B03"],
        cal_n_B08=ncal["B08"], cal_pixels=cal_idx,
        lmodel2_B03=lmodel2_rows_saved["B03"], lmodel2_B08=lmodel2_rows_saved["B08"],
        sd_ndwi_uncal=sd_unc.astype(np.float32), sd_ndwi_cal=sd_cal.astype(np.float32),
        mask_uncal=mask_unc, mask=mask, mask_idx=mask_idx,
        prelim=prelim_frame, prelim_status=st_p, prelim_pars=fit_p[:, 2:7].astype(np.float32),
        h_c=np.array(h_c), lag_pool_size=np.array(len(pool)),
        lag_sample_idx=mask_idx[samp], lag_sample_lag=lag_s,
        lag_sample_gap_min=np.nanmin(np.where(np.isfinite(gap_s), gap_s, np.inf), axis=1),
        lag_sample_infl=inf_l.astype(np.float32), lag_sample_status=st_l.astype(np.int8),
        gam_lag=gam_frame, gam_fitted=gam_fit,
        final=final_frame, final_status=st_f, final_keep=final_keep,
        prelim_keep=prelim_frame[keep], gam_lag_keep=gam_frame[keep], mask_keep=mask[keep],
        exposure=expo, exposure_hw=np.array(hw_mean), exposure_lw=np.array(lw_mean),
        exposure_cycle_h=np.array(cycle_mean), **fixture)
    from rasterio.crs import CRS
    from pyintertidal.raster import write_geotiff
    crs = CRS.from_wkt(grid["wkt"])
    tr = grid["transform"]
    tifs = [write_geotiff(os.path.join(out_dir, "granadeiro2021_dem.tif"),
                          final_frame.reshape(H, W), tr, crs),
            write_geotiff(os.path.join(out_dir, "granadeiro2021_lag_min.tif"),
                          gam_frame.reshape(H, W), tr, crs),
            write_geotiff(os.path.join(out_dir, "granadeiro2021_exposure_h.tif"),
                          expo.reshape(H, W), tr, crs),
            write_geotiff(os.path.join(out_dir, "granadeiro2021_intertidal_mask.tif"),
                          mask.reshape(H, W).astype(np.uint8), tr, crs, dtype="uint8",
                          nodata=None)]
    timings["write_s"] = time.time() - t0
    log(f"  wrote {npz_path} and {len(tifs)} GeoTIFFs")

    # ── scoring ──────────────────────────────────────────────────────────
    t0 = time.time()
    score = None
    if not args.no_score:
        if site == "villaviciosa":
            score = score_rtk(os.path.join(out_dir, "rtk_metrics_granadeiro2021.csv"),
                              final_frame.reshape(H, W), keep, (H, W), T)
        else:
            score = score_vaklodingen(site, cfg, final_keep, keep, grid, out_dir)
    timings["scoring_s"] = time.time() - t0
    timings["total_s"] = time.time() - t_start

    # ── diagnostics ──────────────────────────────────────────────────────
    fin_keep = np.isfinite(final_keep)
    scenes = [{"date": str(u_dates[i]), "time_utc": str(t_ts[i]),
               "cloud_share": float(cloud_c[use_c][i]), "nodata_share": float(nodata_c[use_c][i]),
               "h_m": float(h_scene[i]), "rising": bool(rising[i]),
               "h_hw": float(br["h_hw"][i]), "h_lw": float(br["h_lw"][i]),
               "cycle_h": float(cyc_h[i]),
               "cal_B03": [float(v) for v in coef["B03"][i]],
               "cal_B08": [float(v) for v in coef["B08"][i]],
               "cal_n_B03": int(ncal["B03"][i]), "cal_n_B08": int(ncal["B08"][i])}
              for i in range(T)]
    diag = {
        "experiment": "v8_granadeiro2021", "site": site, "product": PRODUCT,
        "paper": g21.PAPER, "frozen": FROZEN,
        "software": {"R": versions, "rscript": runner.rscript, "workers": args.workers},
        "cube": {"path": cfg["cube"], "bytes": os.stat(cfg["cube"]).st_size,
                 "mtime": os.stat(cfg["cube"]).st_mtime, "shape": [H, W],
                 "transform": list(grid["transform"])[:6], "fill": fill,
                 "scaling": "reflectance = int16 value / 10000 (DN - 1000 harmonised); fill = no data"},
        "scenes": {"cube_scenes": int(len(dates)), "period_scenes": int(in_period.sum()),
                   "candidates_with_time": int(len(cand)), "used": T,
                   "rising": int(rising.sum()), "ebbing": int((~rising).sum()),
                   "rejected_cloud": int(np.sum(cloud_c >= g21.SCENE_CLOUD_MAX)),
                   "rejected_nodata": int(np.sum(nodata_c > g21.SCENE_NODATA_MAX)),
                   "cloud_share_used": quantiles(cloud_c[use_c]),
                   "first_last": [str(u_dates[0]), str(u_dates[-1])],
                   "per_scene": scenes},
        "boundary": {"name": bname, "series_cached": bool(cached), "marks": marks_info,
                     "h_scene": quantiles(h_scene), "h_c_m": h_c,
                     "hw_mean_m": hw_mean, "lw_mean_m": lw_mean, "cycle_mean_h": cycle_mean,
                     "cycle_sd_h": float(np.std(cyc_h, ddof=1)),
                     "mean_h_scene_m (old reading of step 19, not used)": float(np.mean(h_scene))},
        "bands": {"nodata_obs_B03": n_nodata_g, "nodata_obs_B08": n_nodata_n,
                  "nir_le_0_obs_kept": n_nir_le0},
        "reference_scene": {"pos": ref_pos, "date": str(u_dates[ref_pos]),
                            "score": float(ref_score[ref_pos]),
                            "range_green": float(rg[ref_pos]), "range_nir": float(rn[ref_pos]),
                            "median_range_green": float(np.median(rg)),
                            "median_range_nir": float(np.median(rn))},
        "calibration": {
            "pixels": int(len(cal_idx)), "sea": n_sea, "land": n_land,
            "exclusion_mask_uncalibrated_px": int(mask_unc.sum()),
            "B03": {"slope": quantiles(np.delete(coef["B03"][:, 1], ref_pos)),
                    "intercept": quantiles(np.delete(coef["B03"][:, 0], ref_pos)),
                    "pairs": quantiles(np.delete(ncal["B03"], ref_pos))},
            "B08": {"slope": quantiles(np.delete(coef["B08"][:, 1], ref_pos)),
                    "intercept": quantiles(np.delete(coef["B08"][:, 0], ref_pos)),
                    "pairs": quantiles(np.delete(ncal["B08"], ref_pos))}},
        "mask": {"sd_threshold": g21.SD_NDWI_THRESHOLD, "px": n_mask,
                 "frame_share": n_mask / P, "uncalibrated_px": int(mask_unc.sum()),
                 "keep_px": int(len(keep)), "keep_in_mask": int(mask[keep].sum()),
                 "keep_in_mask_share": float(mask[keep].mean()),
                 "mask_outside_keep": int((mask & ~in_keep).sum()),
                 "ndwi_nonfinite_obs_uncal": nf_unc, "ndwi_nonfinite_obs_cal": nf_cal,
                 "histogram_check": {k: v for k, v in hist.items() if k not in ("counts", "edges")},
                 "histogram_counts": hist["counts"], "histogram_bin_width": hist["bin_width"]},
        "preliminary": {"fitted_px": n_mask, "converged_px": int(conv_p.sum()),
                        "convergence_rate": float(conv_p.mean()), "status": status_counts(st_p),
                        "elevation": quantiles(fit_p[conv_p, 1]),
                        "keep_converged": int(np.isfinite(prelim_frame[keep]).sum())},
        "lags": {"h_c_m": h_c, "pool_px": int(len(pool)), "sample_px": int(n_s),
                 "with_lag": int(has_lag.sum()), "without_lag": int((~has_lag).sum()),
                 "distribution": quantiles(lag_s), "counts_on_grid": lag_counts,
                 "edge_hits_minus90": edge_lo, "edge_hits_plus90": edge_hi,
                 "edge_share": (edge_lo + edge_hi) / max(int(has_lag.sum()), 1),
                 "fits": int(n_s * len(g21.LAG_GRID_MIN) * 2),
                 "fit_status": status_counts(st_l.ravel())},
        "gam": {**gam_info, "prediction": quantiles(gam_pred),
                "residual_sd_min": float(np.std(lag_s[has_lag] - gam_fit, ddof=0))},
        "final": {"refitted_px": int(len(conv_rows)), "converged_px": int(ok_f.sum()),
                  "convergence_rate": float(ok_f.mean()), "status": status_counts(st_f),
                  "elevation": quantiles(fit_f[ok_f, 1]),
                  "keep_finite": int(fin_keep.sum()), "keep_coverage": float(fin_keep.mean()),
                  "change_vs_prelim_m": quantiles(final_frame[conv_idx] - prelim_frame[conv_idx])},
        "exposure": {"cycle_h": cycle_mean, "hw_m": hw_mean, "lw_m": lw_mean,
                     "finite_px": int(np.isfinite(expo).sum()),
                     "nan_outside_lw_hw_px": int(np.sum(np.isfinite(final_frame) & ~np.isfinite(expo)))},
        "proofs": proofs,
        "scoring": score,
        "runtime_s": timings,
        "memory": {"python_peak_mb": peak_memory_mb(), **{k: runner_stats[k] for k in
                   ("r_worker_peak_mb", "r_workers_concurrent_peak_mb", "python_plus_r_peak_mb")}},
        "r_workers": {"jobs": runner_stats["jobs"], "worker_seconds": runner_stats["worker_seconds"],
                      "temp_peak_mb": runner_stats["temp_peak_bytes"] / 2 ** 20,
                      "temp_dir_removed": tmp_removed},
        "inputs_sha256": {p: sha256(p) for p in (cfg["extract"], cfg["overpass"])},
        "written": [npz_path] + tifs + ([] if score is None else
                                        (score["csv"] if isinstance(score["csv"], list) else [score["csv"]])),
    }
    jpath = os.path.join(out_dir, "granadeiro2021_diagnostics.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    cache.clear()
    log(f"  wrote {jpath}; total {timings['total_s']:.0f} s; python peak "
        f"{diag['memory']['python_peak_mb']:.0f} MB, python + R peak "
        f"{runner_stats['python_plus_r_peak_mb']:.0f} MB")
    return diag


if __name__ == "__main__":
    main()
