"""
scenes.py — Individual scenes: connect, list dates, download single images
==========================================================================

The analysis path never needs this module: products are computed from the
cached cube (:mod:`pyintertidal.cube`), which is downloaded once. What
single scenes ARE for:

* **looking at your study area** — an RGB image of a specific date, to check
  a feature you see in a product is real and not an artefact;
* **quality control** — the SCL band of a date the filter rejected, to see
  what the algorithm saw;
* **figures and datasets** — RGB panels for a paper, or PNG exports for
  machine-learning work.

Each download is a small OpenEO batch job and is idempotent: an existing
file is never re-downloaded. For many dates prefer :func:`download_scenes`,
which asks the backend for the whole series in ONE job instead of one job
per date.
"""

from __future__ import annotations

import os

from .aoi import as_aoi
from .net import use_system_certificates
from .raster import is_valid_tif


def connect(backend="openeo.dataspace.copernicus.eu", fix_ssl=True,
            interactive=True):
    """Authenticated connection to the Copernicus Data Space.

    Tries the CACHED refresh token before anything interactive. That order
    matters for unattended work: ``authenticate_oidc`` can fall back to the
    device flow, which prints a URL and a code and then blocks until somebody
    types them into a browser — fine in a notebook, fatal in a background job
    or a scheduled run, where the code expires unseen.

    ``interactive=False`` refuses to fall back at all, so a batch job fails
    loudly instead of hanging.

    ``fix_ssl=True`` routes certificate validation through the operating
    system's trust store — see
    :func:`pyintertidal.net.use_system_certificates`.
    """
    if fix_ssl:
        use_system_certificates()
    import openeo

    conn = openeo.connect(backend)
    try:
        conn.authenticate_oidc_refresh_token()
        return conn
    except Exception:
        if not interactive:
            raise RuntimeError(
                "no usable cached token; run connect(interactive=True) once "
                "from a session where you can open a browser")
    conn.authenticate_oidc()          # device / browser flow, then caches
    return conn


def available_dates(connection, aoi, time_extent, max_cloud_cover=100):
    """Sentinel-2 L2A dates available over an AOI (metadata query only).

    Costs nothing in processing credits: it inspects the collection's
    temporal labels rather than loading any pixel. Use it to size a study
    before committing to a download.
    """
    aoi = as_aoi(aoi)
    cube = connection.load_collection(
        "SENTINEL2_L2A", spatial_extent=aoi.bbox,
        temporal_extent=list(time_extent), bands=["B04"],
        max_cloud_cover=max_cloud_cover,
    )
    labels = cube.dimension_labels("t").execute()
    return sorted({str(v)[:10] for v in labels})


def _download_single(connection, aoi, date, bands, out_path, title):
    """One-date, one-job download helper (idempotent)."""
    if os.path.exists(out_path) and is_valid_tif(out_path):
        return out_path
    aoi = as_aoi(aoi)
    cube = connection.load_collection(
        "SENTINEL2_L2A", spatial_extent=aoi.bbox,
        temporal_extent=[date, date], bands=list(bands), max_cloud_cover=100,
    )
    job = cube.save_result(format="GTiff").create_job(title=title)
    job.start_and_wait()
    assets = job.get_results().get_assets()
    if not assets:
        raise RuntimeError(f"no assets returned for {date}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    assets[0].download(out_path)
    return out_path


def download_rgb(connection, aoi, date, out_dir="scenes_rgb"):
    """True-colour scene (B04, B03, B02) for one date → GeoTIFF path."""
    out = os.path.join(out_dir, f"rgb_{date}.tif")
    return _download_single(connection, aoi, date, ["B04", "B03", "B02"],
                            out, f"rgb_{date}")


def download_scl(connection, aoi, date, out_dir="scenes_scl"):
    """Scene Classification band for one date → GeoTIFF path."""
    out = os.path.join(out_dir, f"scl_{date}.tif")
    return _download_single(connection, aoi, date, ["SCL"], out, f"scl_{date}")


def download_indices(connection, aoi, date, bands=("B03", "B08", "B11"),
                     out_dir="scenes_index"):
    """Reflectance bands of one date, to compute indices by hand.

    Defaults cover NDWI (B03/B08) and MNDWI (B03/B11) — useful when you want
    to inspect a single date's index maps rather than a time-series product.
    """
    out = os.path.join(out_dir, f"idx_{date}.tif")
    return _download_single(connection, aoi, date, list(bands), out,
                            f"index_{date}")


def download_scenes(connection, aoi, dates, bands=("B04", "B03", "B02"),
                    out_dir="scenes_rgb", prefix="rgb", keep_only_valid=True,
                    verbose=True):
    """Download MANY dates in a single batch job (much faster than looping).

    One job produces one asset per date; empty assets (AOI outside the
    granule) are discarded when ``keep_only_valid``. Returns the list of
    written paths.
    """
    aoi = as_aoi(aoi)
    dates = sorted(dates)
    if not dates:
        return []
    os.makedirs(out_dir, exist_ok=True)

    cube = connection.load_collection(
        "SENTINEL2_L2A", spatial_extent=aoi.bbox,
        temporal_extent=[dates[0], dates[-1]], bands=list(bands),
        max_cloud_cover=100,
    )
    try:
        cube = cube.filter_labels(dimension="t",
                                  condition=lambda x: x.isin(list(dates)))
    except Exception:
        pass                      # backend without filter_labels: keep all

    job = cube.save_result(format="GTiff").create_job(
        title=f"{prefix}_{aoi.name}_{len(dates)}scenes")
    if verbose:
        print(f"[scenes] one job for {len(dates)} dates…")
    job.start_and_wait()

    written = []
    for asset in job.get_results().get_assets():
        out = os.path.join(out_dir, asset.name)
        asset.download(out)
        if keep_only_valid and not is_valid_tif(out):
            os.remove(out)
            continue
        written.append(out)
    if verbose:
        print(f"[scenes] {len(written)} scenes written to {out_dir}")
    return written
