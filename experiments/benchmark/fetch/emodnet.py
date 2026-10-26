# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch emodnet``: the EMODnet Bathymetry DTM 2024 per site (L18, cross-tab).

Question
--------
How deep is the permanently wet channel of each site (the shallow-water
celerity row L18, spec §3.3.1), and where is channel water at least 2 m
below LAT (the turbid-channel anchors of the SCL x CLD cross-tab, spec
§2.2)?

Pre-registered rules (CONFIG below; read from the service on 2026-10-19)
------------------------------------------------------------------------
* Source: the public OGC WCS 2.0.1 of EMODnet Bathymetry,
  :data:`WCS_URL`. Its GetCapabilities (read 2026-10-19) lists eight
  coverages; ``emodnet__mean`` ("Mean depth") is the current release and
  its DescribeCoverage links the metadata record
  ``cf51df64-56f9-4a99-b1aa-36b8d7b743a1``, "EMODnet Digital Bathymetry
  (DTM 2024)" (the 2022/2020/2018/2016 releases are separate coverages).
  Grid: EPSG:4326, axis labels ``Lat Long``, 1/16 arc minute (1/960 deg),
  nil value NaN. Vertical datum: LAT (the record's abstract: "water depth
  (refering to the Lowest Astronomical Tide Datum - LAT)"). Values are
  ELEVATIONS, negative below LAT (checked: 53.50 N 4.00 E gives -27.7 m).
* Request per site: ``GetCoverage`` of :data:`COVERAGE_ID` on the frame's
  lon/lat bounding box padded by :data:`PAD_KM` (``fetch.cmems.bbox_lonlat``
  of the frame grid: corners and edge midpoints projected), format
  ``image/tiff``. The frame grid is the v2 record's (``common/<site>/
  record.json``) or, before the record, the Zarr stores'.
* Output: ``results/validation_v2/common/<site>/emodnet/dtm.tif`` and
  ``manifest.json`` (sha256 and size of the file, the request, the raster
  description, UTC), both written atomically (temporary file + rename). A
  cached file whose sha256 matches its manifest and whose request is the
  same is not fetched again.
* Channel anchors (spec §2.2): the frame pixels whose nearest DTM cell is
  at least 2 m below LAT, written as ``common/<site>/channel_anchors.npy``
  (sorted int64 frame flat indices), which ``pilot crosstab`` reads.
* No network, a refused request (HTTP error or an OWS exception report)
  or an unreadable GeoTIFF: the manifest says "not available" with the
  reason and :func:`run` returns a non-zero code. Nothing is substituted.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

import numpy as np

WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
WCS_VERSION = "2.0.1"
COVERAGE_ID = "emodnet__mean"
FORMAT = "image/tiff"
PAD_KM = 2.0
TIMEOUT_S = 300
V2_ROOT = "results/validation_v2"
DTM_NAME = "dtm.tif"
MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "validation-v2-emodnet-1"
CHANNEL_ANCHORS = "channel_anchors.npy"
#: channel anchors: depth >= 2 m below LAT, i.e. DTM elevation <= -2 m (spec §2.2)
CHANNEL_MIN_DEPTH_M = 2.0
STATUS_OK = "ok"
STATUS_NOT_AVAILABLE = "not available"

CONFIG = {
    "service": "EMODnet Bathymetry WCS (OGC WCS 2.0.1)",
    "url": WCS_URL,
    "version": WCS_VERSION,
    "coverage_id": COVERAGE_ID,
    "coverage_title": "Mean depth",
    "product": "EMODnet Digital Bathymetry (DTM 2024)",
    "metadata_record": ("https://emodnet.ec.europa.eu/geonetwork/srv/api/records/"
                        "cf51df64-56f9-4a99-b1aa-36b8d7b743a1"),
    "capabilities_read": ("2026-10-19: GetCapabilities 2.0.1 lists emodnet__mean, "
                          "emodnet__mean_2016/2018/2020/2022, emodnet__mean_atlas_land, "
                          "emodnet__mean_multicolour, emodnet__mean_rainbowcolour; "
                          "DescribeCoverage(emodnet__mean) links the DTM 2024 record"),
    "crs": "EPSG:4326",
    "axis_labels": ["Lat", "Long"],
    "resolution_deg": 1.0 / 960.0,
    "vertical_datum": "LAT",
    "values": "elevation (m), negative below LAT (53.50 N 4.00 E: -27.7 m, checked 2026-10-19)",
    "nodata": "NaN",
    "format": FORMAT,
    "pad_km": PAD_KM,
    "request": "GetCoverage coverageId=emodnet__mean subset=Lat(lo,hi) subset=Long(lo,hi) "
               "format=image/tiff",
    "channel_anchors": {"rule": "frame pixels whose nearest DTM cell is <= -2 m (depth >= 2 m "
                                "below LAT)", "min_depth_m": CHANNEL_MIN_DEPTH_M,
                        "resampling": "nearest"},
}


def log(msg):
    print(msg, flush=True)


class EmodnetError(RuntimeError):
    """The DTM of a site is missing, changed or unreadable."""


# ─────────────────────────────────────────────────────────────────────────────
#  paths, hashes, JSON
# ─────────────────────────────────────────────────────────────────────────────

def default_root():
    """The validation v2 output root: ``$VALIDATION_V2_ROOT``, else
    ``results/validation_v2`` (as the CLI and ``fetch.cmems``)."""
    return os.environ.get("VALIDATION_V2_ROOT") or V2_ROOT


def site_dir(site, root=None):
    """``<root>/common/<site>/emodnet`` (root: the validation v2 output root)."""
    return os.path.join(root or default_root(), "common", str(site),
                        "emodnet").replace("\\", "/")


def anchors_path(site, root=None):
    """``<root>/common/<site>/channel_anchors.npy`` (read by pilot crosstab)."""
    return os.path.join(root or default_root(), "common", str(site),
                        CHANNEL_ANCHORS).replace("\\", "/")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_manifest(out_dir):
    """The site's manifest (dict) or None."""
    p = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def config_sha256():
    from experiments.benchmark import config
    return config.sha256_json(CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
#  the request
# ─────────────────────────────────────────────────────────────────────────────

def bbox_lonlat(grid, pad_km=PAD_KM):
    """``(lon_min, lon_max, lat_min, lat_max)`` of the frame padded by
    ``pad_km`` (``fetch.cmems.bbox_lonlat``)."""
    from experiments.benchmark.fetch import cmems
    return cmems.bbox_lonlat(grid, pad_km=pad_km)


def request_params(bbox):
    """The GetCoverage key-value pairs (a list: ``subset`` appears twice)."""
    lon0, lon1, lat0, lat1 = (float(v) for v in bbox)
    return [("service", "WCS"), ("version", WCS_VERSION), ("request", "GetCoverage"),
            ("coverageId", COVERAGE_ID),
            ("subset", f"Lat({lat0:.6f},{lat1:.6f})"),
            ("subset", f"Long({lon0:.6f},{lon1:.6f})"),
            ("format", FORMAT)]


def request_url(params):
    """The full GET URL of a request (for the manifest and the log)."""
    from urllib.parse import urlencode
    return f"{WCS_URL}?{urlencode(params)}"


def _default_session():
    import requests

    try:
        from pyintertidal import net
        net.use_system_certificates()
    except Exception:                           # optional: the system store if available
        pass
    return requests.Session()


def describe_raster(path):
    """CRS, size, transform, value range and valid count of a GeoTIFF;
    raises :class:`EmodnetError` when it is not a usable DTM."""
    import rasterio

    try:
        with rasterio.open(path) as r:
            a = r.read(1).astype(np.float64)
            nod = r.nodata
            crs = r.crs.to_string() if r.crs else None
            tr = list(r.transform)[:6]
            w, h = r.width, r.height
    except Exception as e:
        raise EmodnetError(f"{path}: not a readable GeoTIFF ({type(e).__name__}: {e})") from None
    ok = np.isfinite(a)
    if nod is not None and np.isfinite(nod):
        ok &= a != nod
    if not ok.any():
        raise EmodnetError(f"{path}: no valid DTM value")
    if crs is None:
        raise EmodnetError(f"{path}: the GeoTIFF has no CRS")
    return {"crs": crs, "width": int(w), "height": int(h), "transform": [float(v) for v in tr],
            "nodata": None if nod is None or not np.isfinite(nod) else float(nod),
            "n_valid": int(ok.sum()), "min": float(a[ok].min()), "max": float(a[ok].max())}


def _refused(resp):
    """The reason a WCS response is not a GeoTIFF, or None."""
    code = int(getattr(resp, "status_code", 0) or 0)
    ctype = str((getattr(resp, "headers", None) or {}).get("Content-Type", "")).lower()
    body = getattr(resp, "content", b"") or b""
    if code != 200:
        return f"HTTP {code}: {body[:300]!r}"
    if "xml" in ctype or body[:5] in (b"<?xml", b"<ows:") or b"ExceptionReport" in body[:400]:
        return f"the service answered with an exception report: {body[:400]!r}"
    if not body:
        return "empty response"
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  fetch one site
# ─────────────────────────────────────────────────────────────────────────────

def fetch_site(site, grid, out_dir, session=None, timeout=TIMEOUT_S):
    """Download (once) the site's DTM tile and write its manifest.

    Returns
    -------
    dict
        The manifest: ``status`` ("ok" or "not available"), ``reason``,
        ``cached`` (True when the file on disk was reused), ``params``,
        ``url``, ``bbox``, ``file`` {name, sha256, size}, ``raster``, ``utc``.
    """
    os.makedirs(out_dir, exist_ok=True)
    bbox = bbox_lonlat(grid)
    params = request_params(bbox)
    man = {"format": MANIFEST_FORMAT, "site": site, "coverage_id": COVERAGE_ID,
           "url": request_url(params), "params": [list(p) for p in params],
           "bbox": [float(v) for v in bbox], "pad_km": PAD_KM, "config_sha256": config_sha256(),
           "vertical_datum": CONFIG["vertical_datum"], "values": CONFIG["values"]}
    dtm = os.path.join(out_dir, DTM_NAME)
    old = read_manifest(out_dir)
    if old and old.get("status") == STATUS_OK and old.get("params") == man["params"] \
            and os.path.isfile(dtm) and sha256_file(dtm) == (old.get("file") or {}).get("sha256"):
        log(f"[fetch emodnet] {site}: cached {dtm} (sha256 matches the manifest)")
        return dict(old, cached=True)
    sess = session or _default_session()
    try:
        resp = sess.get(WCS_URL, params=params, timeout=timeout)
    except Exception as e:                      # no network: recorded, non-zero code
        return _finish(out_dir, dict(man, status=STATUS_NOT_AVAILABLE, utc=_utc(),
                                     reason=f"request failed: {type(e).__name__}: {e}"))
    why = _refused(resp)
    if why:
        return _finish(out_dir, dict(man, status=STATUS_NOT_AVAILABLE, utc=_utc(),
                                     reason=f"request refused: {why}"))
    tmp = f"{dtm}.tmp-{os.getpid()}.tif"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    try:
        rast = describe_raster(tmp)
    except EmodnetError as e:
        os.remove(tmp)
        return _finish(out_dir, dict(man, status=STATUS_NOT_AVAILABLE, utc=_utc(), reason=str(e)))
    os.replace(tmp, dtm)
    man.update(status=STATUS_OK, reason="", utc=_utc(), raster=rast, cached=False,
               file={"name": DTM_NAME, "sha256": sha256_file(dtm),
                     "size": int(os.path.getsize(dtm))})
    log(f"[fetch emodnet] {site}: {dtm} ({man['file']['size'] / 1e6:.1f} MB, "
        f"{rast['width']} x {rast['height']}, {rast['min']:.1f}..{rast['max']:.1f} m)")
    return _finish(out_dir, man)


def _finish(out_dir, man):
    _write_json(os.path.join(out_dir, MANIFEST_NAME), man)
    if man.get("status") != STATUS_OK:
        log(f"[fetch emodnet] {man.get('site')}: {man['status']}: {man.get('reason')}")
    return man


def verified_dtm(out_dir):
    """``(path, manifest)`` of an intact fetched DTM; :class:`EmodnetError`
    when the manifest is missing or not "ok", or the file changed since it
    was fetched (sha256)."""
    man = read_manifest(out_dir)
    if man is None:
        raise EmodnetError(f"no EMODnet manifest in {out_dir}: run fetch emodnet")
    if man.get("status") != STATUS_OK:
        raise EmodnetError(f"the EMODnet fetch of {out_dir} is '{man.get('status')}': "
                           f"{man.get('reason')}")
    p = os.path.join(out_dir, (man.get("file") or {}).get("name") or DTM_NAME)
    if not os.path.isfile(p):
        raise EmodnetError(f"{p} is missing: run fetch emodnet")
    if sha256_file(p) != (man.get("file") or {}).get("sha256"):
        raise EmodnetError(f"{p} changed since it was fetched (sha256): run fetch emodnet")
    return p, man


# ─────────────────────────────────────────────────────────────────────────────
#  the DTM on the frame grid
# ─────────────────────────────────────────────────────────────────────────────

def frame_transform(grid):
    """The Affine of a pixel-centre grid ``{xs, ys}`` (x ascending, y
    descending)."""
    from rasterio.transform import Affine

    xs = np.asarray(grid["xs"], np.float64)
    ys = np.asarray(grid["ys"], np.float64)
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    return Affine(dx, 0.0, float(xs[0]) - dx / 2.0, 0.0, dy, float(ys[0]) - dy / 2.0)


def dtm_on_grid(path, grid):
    """The DTM resampled to the frame grid by nearest neighbour (spec
    §3.3.1 L18): (H, W) float32 elevation (m, LAT, negative below), NaN
    where the DTM has no value."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    from experiments.benchmark.fetch.tides import grid_axes

    xs, ys, crs = grid_axes(grid)
    if hasattr(crs, "to_wkt"):                  # a pyproj CRS (V2Package.grid)
        crs = crs.to_wkt()
    H, W = len(ys), len(xs)
    dst = np.full((H, W), np.nan, np.float32)
    with rasterio.open(path) as r:
        src = r.read(1).astype(np.float32)
        nod = r.nodata
        if nod is not None and np.isfinite(nod):
            src = np.where(src == nod, np.nan, src).astype(np.float32)
        reproject(source=src, destination=dst, src_transform=r.transform, src_crs=r.crs,
                  dst_transform=frame_transform({"xs": xs, "ys": ys}), dst_crs=crs,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def channel_anchors(elev, min_depth_m=CHANNEL_MIN_DEPTH_M):
    """Frame flat indices (sorted int64) where the DTM is at least
    ``min_depth_m`` below LAT."""
    e = np.asarray(elev, np.float64)
    with np.errstate(invalid="ignore"):
        m = np.isfinite(e) & (e <= -float(min_depth_m))
    return np.flatnonzero(m.ravel()).astype(np.int64)


def write_anchors(site, grid, dtm_path, root=None):
    """``common/<site>/channel_anchors.npy`` from the DTM; returns its record."""
    a = channel_anchors(dtm_on_grid(dtm_path, grid))
    p = anchors_path(site, root)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = f"{p}.tmp-{os.getpid()}.npy"
    np.save(tmp, a, allow_pickle=False)
    os.replace(tmp, p)
    return {"path": p, "n_px": int(len(a)), "sha256": sha256_file(p),
            "min_depth_m": CHANNEL_MIN_DEPTH_M}


# ─────────────────────────────────────────────────────────────────────────────
#  the frame grid of a site
# ─────────────────────────────────────────────────────────────────────────────

def frame_grid(site, root=None, cubes_root=None):
    """``{"xs", "ys", "crs"}`` of a site: the v2 record's grid when it exists
    (``common/<site>/record.json``), else the Zarr stores'."""
    rec = os.path.join(root or default_root(), "common", str(site), "record.json")
    if os.path.isfile(rec):
        with open(rec, encoding="utf-8") as f:
            g = json.load(f)["grid"]
        return {"xs": np.asarray(g["xs"], np.float64), "ys": np.asarray(g["ys"], np.float64),
                "crs": g["crs_wkt"], "source": rec.replace("\\", "/")}
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs

    ds = zc.open_cube(site, "2017-01-01", "2026-01-01", root=cubes_root, bands=["SCL"],
                      missing="skip")
    try:
        g = inputs.grid_of(ds)
    finally:
        ds.close()
    return {"xs": np.asarray(g.xs, np.float64), "ys": np.asarray(g.ys, np.float64),
            "crs": g.crs_wkt, "source": "zarr stores"}


# ─────────────────────────────────────────────────────────────────────────────
#  the command
# ─────────────────────────────────────────────────────────────────────────────

def run(sites, root=None, cubes_root=None, session=None, grids=None):
    """``fetch emodnet`` for ``sites``: one DTM tile, its manifest and the
    channel anchors per site.

    Parameters
    ----------
    sites : iterable of str
    root : str, optional
        The validation v2 output root (default ``results/validation_v2``).
    cubes_root : str, optional
        The Zarr cubes root (the frame grid before the record exists).
    session : object with ``get(url, params=, timeout=)``, optional
        Default ``requests.Session()`` (tests inject a fake).
    grids : mapping, optional
        ``{site: {"xs", "ys", "crs"}}`` (default :func:`frame_grid`).

    Returns
    -------
    int
        0 when every site has its DTM, 1 otherwise (no network, a refused
        request, an unreadable tile or no frame grid; the reason is logged
        and written to the site's manifest).
    """
    bad = []
    sess = session
    for site in sites:
        out_dir = site_dir(site, root)
        try:
            grid = (grids or {}).get(site) or frame_grid(site, root, cubes_root)
        except Exception as e:
            log(f"[fetch emodnet] {site}: no frame grid ({type(e).__name__}: {e}); nothing "
                "was requested")
            bad.append(site)
            continue
        if sess is None:
            try:
                sess = _default_session()
            except Exception as e:              # no HTTP client: nothing can be fetched
                log(f"[fetch emodnet] no HTTP session ({type(e).__name__}: {e})")
                return 1
        man = fetch_site(site, grid, out_dir, session=sess)
        if man.get("status") != STATUS_OK:
            bad.append(site)
            continue
        try:
            anc = write_anchors(site, grid, os.path.join(out_dir, DTM_NAME), root)
            if man.get("channel_anchors") != anc:
                man["channel_anchors"] = anc
                _write_json(os.path.join(out_dir, MANIFEST_NAME),
                            {k: v for k, v in man.items() if k != "cached"})
            log(f"[fetch emodnet] {site}: {anc['n_px']:,} channel-anchor pixels "
                f"(DTM <= -{CHANNEL_MIN_DEPTH_M:g} m LAT) -> {anc['path']}")
        except Exception as e:
            log(f"[fetch emodnet] {site}: channel anchors FAILED ({type(e).__name__}: {e})")
            bad.append(site)
    if bad:
        log(f"[fetch emodnet] FAILED at {bad}: rerun 'fetch emodnet' (cached tiles are kept)")
        return 1
    return 0


def _bytes_of_geotiff(array, transform, crs="EPSG:4326", nodata=np.nan):
    """A GeoTIFF in memory (for tests and the local fake service)."""
    from rasterio.io import MemoryFile

    a = np.asarray(array, np.float32)
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                     dtype="float32", crs=crs, transform=transform, nodata=nodata) as dst:
            dst.write(a, 1)
        return bytes(mf.read())


__all__ = ["CONFIG", "run", "fetch_site", "verified_dtm", "dtm_on_grid", "channel_anchors",
           "frame_grid", "site_dir", "anchors_path", "request_params", "read_manifest",
           "EmodnetError"]
