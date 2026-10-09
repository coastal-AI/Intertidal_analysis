# -*- coding: utf-8 -*-
"""Download the validation cubes again as Zarr, with nine bands, one store per year.

What
----
For the validation v2 (indices NIR/NDWI/MNDWI/AWEI/WRI, SCL and CLD cloud
filters, water detection) the five validation sites are fetched again from
the Copernicus Data Space openEO backend with the bands
``B02 B03 B04 B08 B8A B11 B12 SCL CLD`` of SENTINEL2_L2A, on the SAME 10 m
grids as the existing netCDF cubes, and kept as ONE ZARR STORE PER SITE AND
YEAR, ``cubes/<site>/<year>.zarr`` (format: ``experiments/zarr_cubes.py``;
read any window with ``zarr_cubes.open_cube(site, start, end)``).

The process is the one of the existing cubes (``SentinelCube.ensure`` and,
for Villaviciosa, the notebook download of ``ndwi_cube_villaviciosa_grande_
10y.nc``): ``load_collection(SENTINEL2_L2A, bbox, [start, end), bands,
max_cloud_cover=100)`` -> ``resample_spatial(resolution=10, method="near")``
-> ``save_result``, only with more bands and ``format="Zarr"``. Same bbox,
same resolution, same call, so the same grid: the bbox's UTM envelope
snapped outward to the 10 m lattice of the Sentinel-2 tiles (checked against
the Villaviciosa and Ferrol cubes). Resampling: B02/B03/B04/B08 are native
10 m (the call is the identity for them); B8A/B11/B12, SCL and CLD are 20 m
and reach 10 m by NEAREST NEIGHBOUR, the pipeline's method (its only 20 m
band so far was SCL): no new interpolation, and the 20 m bands can be
aggregated back exactly. The FORMAT REPORT measures it on the delivered
data (share of pixel pairs inside one 20 m cell holding the same value:
100 % = nearest from the native grid). A store whose grid is in another CRS
or off the existing cube's 10 m lattice is refused. Values are kept as
delivered (int16, fill -32768); nothing is rescaled.

One openEO batch job per year (windows [YYYY-01-01, YYYY+1-01-01), end-
exclusive like ``experiments/download_cube.py``), with the same
``JOB_OPTIONS``. Each year: job -> download into ``<year>.part/`` ->
normalise into ``<year>.zarr`` (every block read back and compared with what
the backend delivered) -> compare with the existing netCDF cube of the site
where it covers the year (``<year>.verify.json``) -> overpass times of the
new dates into ``cubes/<site>/overpass_times.json`` (a NEW file; the
pipeline's files are only read). Resumable: a finished year is skipped; a
year that failed is the only one repeated; a job left running by a killed
session is reattached (job id in ``<year>.part/job.json``, or found by its
title on the backend) instead of paid for twice; a job that failed leaves its
backend error log in ``<year>.part/``.

The Zarr output of the backend is flagged experimental and nobody here has
seen it, so the normaliser accepts a directory store, a zipped store, a
store delivered file by file (one asset per file), several assets (split by
band or by date), Zarr v2 or v3 (sharded, big-endian, without dimension
names or group metadata), per-band variables or a ``bands`` dimension, and
netCDF as a fallback; it stops with a clear message otherwise, and also
when the delivered grid is in another CRS or off the 10 m lattice of the
existing cube (``--allow-grid-offset`` overrides). ``--test`` runs ONE tiny
job first (2 km x 2 km inside Villaviciosa, 2025-06-01..15, all bands),
prints what the backend says about its Zarr format and a FORMAT REPORT of
what came back, writes ``<root>/_test/villaviciosa/2025.zarr`` and opens it
through ``zarr_cubes.open_cube``. Run it again to re-normalise the same
download (no new job); delete ``<root>/_test`` to pay for a new one.

Usage (from the repository root; on the server after ``source env.sh``)::

    $PY -X utf8 -m experiments.download_cube_zarr --login          # once, interactive
    $PY -X utf8 -m experiments.download_cube_zarr --test           # the format test
    $PY -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023-2025 --dry-run
    $PY -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023-2025
    $PY -X utf8 -m experiments.download_cube_zarr ems --years 2017-2022
    $PY -X utf8 -m experiments.download_cube_zarr ems --years 2023 --verify-only

Importing this module does nothing (no chdir, no network); ``main`` works
from the repository root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import socket
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import zarr_cubes as zc  # noqa: E402
from experiments.validation_grid import (DUTCH_PERIOD, DUTCH_SITES, dutch_cube,  # noqa: E402
                                         dutch_products)

# ─────────────────────────────────────────────────────────────────────────────
#  what is downloaded
# ─────────────────────────────────────────────────────────────────────────────

BACKEND = "openeo.dataspace.copernicus.eu"
COLLECTION = "SENTINEL2_L2A"
BANDS = zc.BANDS
RESOLUTION_M = 10
#: resample_spatial method of the existing cubes (cube.py, notebook_compat.py)
RESAMPLE_METHOD = "near"
#: as the existing cubes: a property filter that keeps every scene
MAX_CLOUD_COVER = 100
#: the same as experiments.download_cube.JOB_OPTIONS (a test checks it)
JOB_OPTIONS = {"executor-memory": "4G", "executor-memoryOverhead": "4G"}
BACKEND_FORMAT = "Zarr"
DEFAULT_YEARS = (2023, 2024, 2025)

RESAMPLING = (f"resample_spatial(resolution={RESOLUTION_M}, method='{RESAMPLE_METHOD}') on the "
              "whole cube, as the existing netCDF cubes: identity for the native 10 m bands "
              "(B02 B03 B04 B08); nearest neighbour 20 m -> 10 m for B8A B11 B12 SCL CLD")
BAND_NOTES = {
    "SCL": "Sen2Cor scene classification (class codes), 20 m, nearest neighbour to 10 m",
    "CLD": "Sen2Cor cloud probability (%), 20 m, nearest neighbour to 10 m",
}
SPECTRAL_NOTE = ("L2A surface reflectance x 10000 (digital number as delivered by the backend; "
                 "not rescaled)")

#: the sites, the box each existing cube was requested with, that cube, the
#: period it was REQUESTED for (end-exclusive) and the pipeline's overpass
#: file (read only, to keep the times the products used). bbox None = the
#: registry polygon's box, as experiments.download_cube and the notebooks.
#: Villaviciosa's cube came from a wider box than the registry polygon
#: (915 x 915 px; experiments/prototypes/dl_villaviciosa_10y.py): every
#: Villaviciosa product sits on that grid, so that box is used here.
SITE_CONFIG = {
    "villaviciosa": {"bbox": {"west": -5.46, "south": 43.47, "east": -5.35, "north": 43.55},
                     "reference": "ndwi_cube_villaviciosa_grande_10y.nc",
                     "reference_period": ("2016-01-01", "2025-12-31"),
                     "pipeline_overpass": "products_marea/overpass_times.json"},
    "ferrol": {"bbox": None, "reference": "ndwi_cube_ferrol_2023-2025_10m.nc",
               "reference_period": ("2023-01-01", "2025-12-31"),
               "pipeline_overpass": "products_ferrol/overpass_times.json"},
}
SITE_CONFIG.update({s: {"bbox": None, "reference": dutch_cube(s), "reference_period": DUTCH_PERIOD,
                        "pipeline_overpass": f"{dutch_products(s)}/overpass_times.json"}
                    for s in DUTCH_SITES})
SITES = zc.SITES

#: --test: 2 km x 2 km inside the Villaviciosa grid around the outer ria
#: (channel, flats and shore), 2025-06-01..2025-06-15, every band
TEST_SITE = "villaviciosa"
TEST_CENTRE = (-5.395, 43.52)          # lon, lat
TEST_HALF_M = 1000.0
TEST_WINDOW = ("2025-06-01", "2025-06-16")
TEST_ROOT = "_test"

#: dates per year in the existing cubes (Villaviciosa 2016-2025; the other
#: sites have the same within a few), for size estimates of years no cube has
TYPICAL_DATES = {2016: 63, 2017: 126, 2018: 145, 2019: 145, 2020: 144, 2021: 146, 2022: 145,
                 2023: 145, 2024: 147, 2025: 173}
#: estimated bytes on disk per pixel and date for the nine bands, from Blosc/
#: Zstd-5 ratios measured on the Ferrol 10 m cube: 10 m bands ~1.75x, 20 m
#: bands upsampled by nearest ~4.5x, SCL ~100x, CLD ~30x (assumed)
BYTES_PER_PIXEL_DATE = 6.0


class FormatError(RuntimeError):
    """The backend delivered something the normaliser does not understand."""


class JobFailed(RuntimeError):
    """The openEO job ended in error (logs saved in the part folder)."""


class LockedError(RuntimeError):
    """Another download of the same site is running."""


# ─────────────────────────────────────────────────────────────────────────────
#  requests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Request:
    """One store to produce: a site, a year (the store name) and a window."""
    site: str
    year: int
    start: str
    end: str                     # exclusive
    bbox: dict
    root: Path
    bands: tuple = BANDS
    test: bool = False

    @property
    def tag(self):
        return f"{self.site} {self.year}{' TEST' if self.test else ''}"

    @property
    def store(self):
        return zc.store_path(self.site, self.year, self.root)

    @property
    def part(self):
        return zc.site_dir(self.site, self.root) / f"{self.year}.part"

    def sidecar(self, kind):
        return zc.sidecar_path(self.site, self.year, kind, self.root)


def site_bbox(site):
    cfg = SITE_CONFIG[site]
    if cfg["bbox"] is not None:
        return dict(cfg["bbox"])
    import pyintertidal as pit          # the registry; heavy import, only when needed

    return dict(pit.sites.get(site).bbox)


def test_bbox():
    """The --test box: TEST_HALF_M around TEST_CENTRE, in degrees."""
    lon, lat = TEST_CENTRE
    dlat = TEST_HALF_M / 111_320.0
    dlon = TEST_HALF_M / (111_320.0 * math.cos(math.radians(lat)))
    return {"west": round(lon - dlon, 6), "south": round(lat - dlat, 6),
            "east": round(lon + dlon, 6), "north": round(lat + dlat, 6)}


def parse_years(tokens):
    """``["2017-2022", "2025"]`` or ``["2023,2025"]`` -> sorted unique years."""
    years = set()
    for tok in tokens or []:
        for part in str(tok).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"(\d{4})\s*[-:]\s*(\d{4})", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if b < a:
                    raise ValueError(f"empty year range {part!r}")
                years.update(range(a, b + 1))
            elif re.fullmatch(r"\d{4}", part):
                years.add(int(part))
            else:
                raise ValueError(f"not a year or a range: {part!r}")
    bad = [y for y in years if not 2015 <= y <= dt.date.today().year]
    if bad:
        raise ValueError(f"years outside Sentinel-2 L2A: {sorted(bad)}")
    return sorted(years)


def year_requests(sites, years, root):
    return [Request(site=s, year=y, start=zc.year_window(y)[0], end=zc.year_window(y)[1],
                    bbox=site_bbox(s), root=root) for s in sites for y in years]


def test_request(root):
    return Request(site=TEST_SITE, year=int(TEST_WINDOW[0][:4]), start=TEST_WINDOW[0],
                   end=TEST_WINDOW[1], bbox=test_bbox(), root=Path(root) / TEST_ROOT, test=True)


# ─────────────────────────────────────────────────────────────────────────────
#  logging and locking
# ─────────────────────────────────────────────────────────────────────────────

def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def make_log(req):
    """``log(msg)``: a stamped line on stdout (flushed) and in the site's log."""
    path = zc.site_dir(req.site, req.root) / "download.log"

    def log(msg):
        line = f"{utcnow():%Y-%m-%d %H:%M:%S}Z [{req.tag}] {msg}"
        print(line, flush=True)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    return log


def _pid_alive(pid):
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    if os.name == "nt":                 # os.kill(pid, 0) TERMINATES a process on Windows
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SiteLock:
    """One download per site at a time: ``<site>/.download.lock`` (O_EXCL).

    A lock left by a process that is gone (same host) is taken over; a live
    one stops this run before any job is created, so two sessions never pay
    for the same year twice.
    """

    def __init__(self, directory):
        self.path = Path(directory) / ".download.lock"
        self.mine = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        info = {"host": socket.gethostname(), "pid": os.getpid(),
                "started_utc": utcnow().isoformat(timespec="seconds")}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    other = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    other = {}
                stale = (not other or (other.get("host") == info["host"]
                                       and not _pid_alive(other.get("pid", -1))))
                if stale:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise LockedError(
                    f"{self.path} is held by pid {other.get('pid')} on {other.get('host')} since "
                    f"{other.get('started_utc')}: another download of this site is running "
                    f"(delete the file if it is not)")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(info, f)
            self.mine = True
            return self
        raise LockedError(f"could not take {self.path}")

    def __exit__(self, *exc):
        if self.mine:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        return False


def ensure_gitignore(root):
    """``cubes/.gitignore`` with ``*``: the stores never show up in git."""
    root = Path(root)
    try:
        root.resolve().relative_to(ROOT)
    except ValueError:
        return
    root.mkdir(parents=True, exist_ok=True)
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("# Zarr cubes written by experiments/download_cube_zarr.py: data, not code\n*\n",
                      encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  the openEO process
# ─────────────────────────────────────────────────────────────────────────────

def build_process(req, connection=None, backend_format=BACKEND_FORMAT, format_options=None):
    """``load_collection -> resample_spatial(10, near) -> save_result``.

    Without a connection the graph is built offline (dry run, job title);
    with one, the client also checks the bands and the output format
    against the backend.
    """
    from openeo.rest.datacube import DataCube

    kw = dict(spatial_extent=dict(req.bbox), temporal_extent=[req.start, req.end],
              bands=list(req.bands), max_cloud_cover=MAX_CLOUD_COVER)
    if connection is not None:
        cube = connection.load_collection(COLLECTION, **kw)
    else:
        cube = DataCube.load_collection(COLLECTION, connection=None, **kw)
    cube = cube.resample_spatial(resolution=RESOLUTION_M, method=RESAMPLE_METHOD)
    return cube.save_result(format=backend_format, options=format_options or None)


def graph_hash(flat_graph):
    blob = json.dumps(flat_graph, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def job_title(req, ghash):
    if req.test:
        return f"zarr_cube_test_{req.site}_{req.start}_{req.end}_{ghash[:8]}"
    return f"zarr_cube_{req.site}_{req.year}_{ghash[:8]}"


def predict_grid(bbox, res=RESOLUTION_M):
    """The grid openEO makes of a lon/lat box: its UTM envelope (zone of the
    box centre) snapped outward to the ``res`` lattice. Matches the existing
    Villaviciosa and Ferrol 10 m cubes and the Dutch 20 m cubes exactly."""
    from pyproj import Transformer

    lon_c = (bbox["west"] + bbox["east"]) / 2.0
    lat_c = (bbox["south"] + bbox["north"]) / 2.0
    zone = int((lon_c + 180.0) // 6.0) + 1
    epsg = (32600 if lat_c >= 0 else 32700) + zone
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    n = 100
    lon = np.r_[np.linspace(bbox["west"], bbox["east"], n), np.full(n, bbox["east"]),
                np.linspace(bbox["east"], bbox["west"], n), np.full(n, bbox["west"])]
    lat = np.r_[np.full(n, bbox["south"]), np.linspace(bbox["south"], bbox["north"], n),
                np.full(n, bbox["north"]), np.linspace(bbox["north"], bbox["south"], n)]
    X, Y = tf.transform(lon, lat)
    x0 = math.floor(X.min() / res) * res
    x1 = math.ceil(X.max() / res) * res
    y0 = math.floor(Y.min() / res) * res
    y1 = math.ceil(Y.max() / res) * res
    return {"epsg": epsg, "shape": [int(round((y1 - y0) / res)), int(round((x1 - x0) / res))],
            "x_edges": [x0, x1], "y_edges": [y0, y1]}


def reference_summary(site):
    """Grid and dates per year of the site's existing netCDF cube (metadata
    only), or None when it is not on this machine."""
    path = ROOT / SITE_CONFIG[site]["reference"]
    if not path.is_file():
        return None
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(path, mask_and_scale=False) as ds:
        tdim = "t" if "t" in ds.dims else "time"
        t = pd.DatetimeIndex(ds[tdim].values)
        g = zc.grid_info(ds)
    return {"path": str(path), "shape": g["shape"], "epsg": g["epsg"],
            "per_year": {int(k): int(v) for k, v in pd.Series(t.year).value_counts().items()}}


# ─────────────────────────────────────────────────────────────────────────────
#  jobs: ledger, orphans, start, wait, error logs
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE = ("created", "queued", "running")
LEDGER = "job.json"


class Ledger:
    """``<year>.part/job.json``: every job created or adopted for this store.

    The id is written BEFORE the job is started, so a killed session leaves
    a job that the next run finds and reattaches to.
    """

    def __init__(self, part, title=None):
        self.path = Path(part) / LEDGER
        self.data = {"title": title, "jobs": []}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        if title:
            self.data["title"] = title

    def save(self):
        zc.write_json(self.data, self.path)

    def current(self):
        """The last job not known to be useless (None when there is none)."""
        for j in reversed(self.data["jobs"]):
            if j.get("status") not in ("error", "canceled", "gone", "expired"):
                return j
        return None

    def known(self, job_id):
        return any(j["job_id"] == job_id for j in self.data["jobs"])

    def add(self, job_id, status, source):
        self.data["jobs"].append({"job_id": job_id, "status": status, "source": source,
                                  "added_utc": utcnow().isoformat(timespec="seconds")})
        self.save()
        return self.data["jobs"][-1]

    def update(self, entry, **kw):
        entry.update(kw)
        entry["updated_utc"] = utcnow().isoformat(timespec="seconds")
        self.save()


def job_summary(info):
    """The parts of a job description worth keeping (no process graph)."""
    keep = ("id", "title", "status", "created", "updated", "progress", "costs", "usage",
            "plan", "budget")
    return {k: info.get(k) for k in keep if k in info}


def find_orphan(conn, title, ledger, log):
    """A job of the same title still useful on the backend (newest first)."""
    try:
        jobs = list(conn.list_jobs())
    except Exception as exc:              # listing is a convenience, never fatal
        log(f"could not list the backend's jobs ({exc}); no orphan search")
        return None
    cands = [j for j in jobs if j.get("title") == title
             and j.get("status") in ACTIVE + ("finished",) and not ledger.known(j.get("id"))]
    cands.sort(key=lambda j: str(j.get("created", "")), reverse=True)
    return cands[0] if cands else None


def _api_status(exc):
    return getattr(exc, "http_status_code", None)


def start_job(job, ctx, log):
    """Start, retrying while the backend refuses for capacity reasons."""
    from openeo.rest import OpenEoApiPlainError

    t0 = time.time()
    while True:
        try:
            job.start()
            return
        except OpenEoApiPlainError as exc:
            msg = f"{getattr(exc, 'code', '')} {exc}"
            status = _api_status(exc)
            if status == 402 or re.search(r"credit|payment|insufficient funds", msg, re.I):
                raise JobFailed(f"the backend refused to start job {job.job_id}: {msg} "
                                f"(credits exhausted?)") from exc
            transient = status in (429, 502, 503, 504) or re.search(
                r"concurren|too many|rate.?limit|capacity|try again", msg, re.I)
            if not transient or time.time() - t0 > ctx.max_wait_s:
                raise
            log(f"backend refused to start job {job.job_id} ({msg.strip()[:200]}); retrying in "
                f"{ctx.capacity_wait_s // 60} min")
            ctx.sleep(ctx.capacity_wait_s)


def wait_job(job, ctx, log):
    """Poll until the job leaves created/queued/running; return its info."""
    import requests
    from openeo.rest import OpenEoApiPlainError

    t0 = time.time()
    interval = ctx.poll_s
    last, last_print, soft = None, 0.0, 0
    while True:
        try:
            info = job.describe()
            soft = 0
        except (requests.ConnectionError, requests.Timeout) as exc:
            soft += 1
            if soft > 30:
                raise
            log(f"connection problem while polling job {job.job_id} ({exc}); retrying")
            ctx.sleep(30)
            continue
        except OpenEoApiPlainError as exc:
            if _api_status(exc) in (502, 503, 504) and soft < 30:
                soft += 1
                log(f"backend unavailable while polling job {job.job_id} ({exc}); retrying")
                ctx.sleep(30)
                continue
            raise
        status = info.get("status", "N/A")
        now = time.time()
        if status != last or now - last_print > 600:
            prog = info.get("progress")
            log(f"job {job.job_id}: {status}"
                + (f" ({prog:.0f} %)" if isinstance(prog, (int, float)) else "")
                + f" after {dt.timedelta(seconds=int(now - t0))}")
            last, last_print = status, now
        if status not in ACTIVE + ("submitted",):
            return info
        if now - t0 > ctx.max_wait_s:
            raise TimeoutError(
                f"job {job.job_id} still {status} after {ctx.max_wait_s / 3600:.0f} h; it keeps "
                f"running on the backend: run the same command again to reattach")
        ctx.sleep(interval)
        interval = min(ctx.poll_max_s, interval * 1.25)


def save_error_logs(job, part, log):
    """Write the backend's error log of a failed job and show its start."""
    try:
        entries = [dict(e) for e in job.logs(level="error")]
    except Exception as exc:
        log(f"could not fetch the logs of job {job.job_id}: {exc}")
        return None
    path = Path(part) / f"job_{job.job_id}.errors.json"
    zc.write_json(entries, path)
    log(f"job {job.job_id}: {len(entries)} error log entries -> {path}")
    for e in entries[:15]:
        log(f"  backend error: {str(e.get('message', e))[:800]}")
    return path


def ensure_finished_job(ctx, req, title, log):
    """A FINISHED job for ``req``: the ledger's, an orphan of the same title,
    or a new one. Returns ``(job, info)``; raises JobFailed after
    ``ctx.retries`` new jobs failed in this run."""
    conn = ctx.connection()
    led = Ledger(req.part, title)
    created_here = 0
    while True:
        cur = led.current()
        if cur is None:
            orphan = find_orphan(conn, title, led, log)
            if orphan is not None:
                log(f"reattaching to job {orphan['id']} ({orphan.get('status')}, created "
                    f"{orphan.get('created')}) found on the backend with this title")
                cur = led.add(orphan["id"], orphan.get("status"), "orphan")
        if cur is None:
            if created_here > ctx.retries:
                raise JobFailed(f"{created_here} job(s) for {req.tag} failed in this run; see the "
                                f"error logs in {req.part}")
            sr = build_process(req, connection=conn, backend_format=ctx.backend_format,
                               format_options=ctx.format_options)
            flat = sr.flat_graph()
            zc.write_json(flat, Path(req.part) / "process_graph.json")
            job = sr.create_job(title=title, job_options=ctx.job_options)
            created_here += 1
            cur = led.add(job.job_id, "created", "created")
            log(f"created job {job.job_id} '{title}' (job options {ctx.job_options})")
        job = conn.job(cur["job_id"])
        try:
            info = job.describe()
        except Exception as exc:
            if _api_status(exc) in (404, 410):
                log(f"job {cur['job_id']} is gone from the backend; a new one will be made")
                led.update(cur, status="gone")
                continue
            raise
        status = info.get("status")
        if status == "created":
            log(f"starting job {job.job_id}")
            start_job(job, ctx, log)
            status = "queued"
        if status in ACTIVE + ("submitted",):
            info = wait_job(job, ctx, log)
            status = info.get("status")
        led.update(cur, status=status, info=job_summary(info))
        if status == "finished":
            costs, usage = info.get("costs"), info.get("usage")
            log(f"job {job.job_id} finished" + (f"; costs {costs}" if costs is not None else "")
                + (f"; usage {json.dumps(usage)[:400]}" if usage else ""))
            return job, info
        log(f"job {job.job_id} ended '{status}'")
        save_error_logs(job, req.part, log)


# ─────────────────────────────────────────────────────────────────────────────
#  results: assets and their download
# ─────────────────────────────────────────────────────────────────────────────

ASSETS_DONE = "assets.json"


def collect_assets(job, meta):
    """``[(name, href, metadata)]`` of a job's results: the collection's
    ``assets`` or, when the backend lists results as STAC items, theirs."""
    assets = dict(meta.get("assets") or {})
    if not assets:
        for link in meta.get("links") or []:
            if link.get("rel") != "item" or not link.get("href"):
                continue
            item = job.connection.get(link["href"], expected_status=200).json()
            for k, v in (item.get("assets") or {}).items():
                name = k if k not in assets else f"{item.get('id', 'item')}/{k}"
                assets[name] = v
    return [(k, v["href"], v) for k, v in assets.items() if isinstance(v, dict) and v.get("href")]


def results_gone(job):
    """True when the backend no longer has the job's results (404/410)."""
    try:
        job.get_results().get_metadata(force=True)
        return False
    except Exception as exc:
        return _api_status(exc) in (404, 410)


def safe_relpath(name):
    parts = [p for p in re.split(r"[\\/]+", str(name)) if p not in ("", ".", "..")]
    return Path(*parts) if parts else Path("asset")


def asset_relpath(name, href):
    """Where an asset lands under ``<part>/assets``.

    A store delivered file by file (one asset per Zarr file) may name its
    assets by the bare file name ('0.0.0', '.zarray'), which would collide:
    the path inside the store is then taken from the href, from the
    component ending in '.zarr' on. Otherwise the asset name is used.
    """
    from urllib.parse import unquote, urlparse

    comps = [c for c in unquote(urlparse(str(href or "")).path).split("/") if c]
    for i, c in enumerate(comps[:-1]):
        if c.lower().endswith(".zarr"):
            return safe_relpath("/".join(comps[i:]))
    return safe_relpath(name)


def download_results(job, req, log, workers=4):
    """Download every asset into ``<part>/assets`` (resumable per file) and
    write ``assets.json`` when all are complete."""
    from concurrent.futures import ThreadPoolExecutor
    from openeo.rest.job import ResultAsset

    part = Path(req.part)
    results = job.get_results()
    meta = results.get_metadata()
    zc.write_json(meta, part / "job-results.json")
    items = collect_assets(job, meta)
    if not items:
        raise FormatError(f"job {job.job_id} finished with no result assets "
                          f"(results metadata saved in {part / 'job-results.json'})")
    raw = part / "assets"
    raw.mkdir(parents=True, exist_ok=True)
    total = sum(int(m.get("file:size") or 0) for _, _, m in items)
    log(f"{len(items)} asset(s) to download" + (f", {total / 1e9:.2f} GB" if total else ""))

    rels = {}
    for name, href, _ in items:
        rels.setdefault(asset_relpath(name, href), []).append(name)
    clash = {str(k): v for k, v in rels.items() if len(v) > 1}
    if clash:
        raise FormatError(f"job {job.job_id}: several assets would land on the same file "
                          f"({dict(list(clash.items())[:5])}); results metadata in "
                          f"{part / 'job-results.json'}")

    def fetch(item):
        name, href, md = item
        rel = asset_relpath(name, href)
        target = raw / rel
        size = md.get("file:size")
        if target.is_file() and (size is None or target.stat().st_size == int(size)):
            return str(rel), target.stat().st_size, md.get("type"), "kept"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".download")
        err = None
        for attempt in range(1, 5):
            try:
                ResultAsset(job=job, name=name, href=href, metadata=md).download(tmp)
                got = tmp.stat().st_size
                if size is not None and got != int(size):
                    raise IOError(f"{name}: {got} bytes downloaded, {size} announced")
                os.replace(tmp, target)
                return str(rel), got, md.get("type"), "downloaded"
            except Exception as exc:          # signed URLs expire: refresh them
                err = exc
                log(f"download of {name} failed ({exc}); attempt {attempt}/4")
                time.sleep(min(120, 15 * attempt))
                try:
                    fresh = {k: (h, m) for k, h, m in collect_assets(
                        job, results.get_metadata(force=True))}
                    href, md = fresh.get(name, (href, md))
                except Exception:
                    pass
        raise IOError(f"could not download {name}: {err}")

    t0 = time.time()
    done = []
    if len(items) > 8:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(fetch, items), 1):
                done.append(r)
                if k % 200 == 0:
                    log(f"  {k}/{len(items)} assets")
    else:
        done = [fetch(it) for it in items]
    got = sum(d[1] for d in done)
    log(f"downloaded {len(done)} asset(s), {got / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")
    zc.write_json({"job_id": job.job_id, "complete": True,
                   "files": {d[0]: {"size": d[1], "type": d[2]} for d in done}},
                  part / ASSETS_DONE)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
#  what came back: zips, stores, files
# ─────────────────────────────────────────────────────────────────────────────

ZARR_MARKERS = {".zgroup", ".zarray", ".zattrs", ".zmetadata", "zarr.json"}


def extract_zips(part, log=print):
    """Unzip every zip asset into ``<part>/extracted/<name without .zip>``
    (members checked to stay inside it). Returns the extraction folders."""
    part = Path(part)
    done = json.loads((part / ASSETS_DONE).read_text(encoding="utf-8"))
    out = []
    for rel, md in sorted(done["files"].items()):
        src = part / "assets" / rel
        is_zip = rel.lower().endswith(".zip") or "zip" in str(md.get("type") or "").lower()
        if not is_zip:
            continue
        if not zipfile.is_zipfile(src):
            raise FormatError(f"{src} is announced as a zip but is not one")
        dest = part / "extracted" / (rel[:-4] if rel.lower().endswith(".zip") else rel + ".d")
        flag = dest.with_name(dest.name + ".extracted")
        out.append(dest)
        if flag.is_file() and dest.is_dir():
            continue
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        base = dest.resolve()
        n = 0
        with zipfile.ZipFile(src) as zf:
            for m in zf.infolist():
                rel_m = safe_relpath(m.filename)
                target = (dest / rel_m).resolve()
                if base != target and base not in target.parents:
                    raise FormatError(f"{src}: member {m.filename!r} would land outside {dest}")
                if m.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as fi, open(target, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
                n += 1
        flag.write_text(str(n), encoding="utf-8")
        log(f"extracted {n} files from {rel} -> {dest}")
    return out


def _is_array_dir(p):
    p = Path(p)
    if (p / ".zarray").is_file():
        return True
    zj = p / "zarr.json"
    if zj.is_file():
        try:
            return json.loads(zj.read_text(encoding="utf-8")).get("node_type") == "array"
        except ValueError:
            return False
    return False


def discover(dirs):
    """Zarr stores, netCDF and GeoTIFF files under ``dirs``.

    A Zarr root is the top-most folder holding Zarr metadata. Arrays whose
    folder has no group metadata above them (a writer that skips the root
    ``.zgroup``) are gathered into an implicit group of their parent.
    """
    roots, ncs, tifs, other = [], [], [], []
    for base in dirs:
        base = Path(base)
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            names = set(filenames)
            if names & ZARR_MARKERS:
                roots.append(Path(dirpath))
                dirnames[:] = []
                continue
            dirnames.sort()
            for fn in sorted(filenames):
                low = fn.lower()
                if low.endswith((".nc", ".nc4", ".netcdf")):
                    ncs.append(Path(dirpath) / fn)
                elif low.endswith((".tif", ".tiff")):
                    tifs.append(Path(dirpath) / fn)
                elif not low.endswith((".zip", ".extracted", ".json", ".download")):
                    other.append(Path(dirpath) / fn)
    stores, implicit = [], {}
    for r in roots:
        if _is_array_dir(r):
            implicit.setdefault(r.parent, []).append(r)
        else:
            stores.append({"kind": "zarr", "path": r, "arrays": None})
    for parent, arrs in sorted(implicit.items()):
        stores.append({"kind": "zarr-implicit", "path": parent, "arrays": sorted(arrs)})
    return {"stores": stores, "netcdf": ncs, "geotiff": tifs, "other": other}


# ─────────────────────────────────────────────────────────────────────────────
#  reading what came back (zarr-python, not xarray's zarr backend: xarray
#  2025.1 cannot open several Zarr v3 / string layouts written by others)
# ─────────────────────────────────────────────────────────────────────────────

T_NAMES = {"t", "time", "times", "date", "dates", "datetime"}
Y_NAMES = {"y", "lat", "latitude", "northing", "row", "rows"}
X_NAMES = {"x", "lon", "long", "longitude", "easting", "col", "cols", "column", "columns"}
B_NAMES = {"bands", "band", "band_name", "band_names", "spectral", "variable", "variables",
           "channel", "channels", "wavelength"}
CRS_NAMES = {"crs", "spatial_ref", "grid_mapping", "transverse_mercator", "projection"}


def axis_of(name):
    n = str(name).split("/")[-1].lower()
    for axis, names in (("t", T_NAMES), ("y", Y_NAMES), ("x", X_NAMES), ("bands", B_NAMES)):
        if n in names:
            return axis
    return None


def _py(v):
    """numpy scalar -> python (for reports and attrs)."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    try:
        v = np.asarray(v).item()
    except (ValueError, TypeError):
        return str(v)
    if isinstance(v, float) and np.isnan(v):
        return float("nan")
    return v


def _attrs_clean(attrs):
    out = {}
    for k, v in dict(attrs).items():
        if k == "_ARRAY_DIMENSIONS":
            continue
        if isinstance(v, (np.generic, np.ndarray)):
            v = np.asarray(v).tolist()
        out[str(k)] = v
    return out


def _codecs(arr):
    bits = []
    for name in ("filters", "serializer", "compressors"):
        v = getattr(arr, name, None)
        if v:
            bits.append(f"{name}={v}")
    return " ".join(bits) or "?"


@dataclass
class RawData:
    """What one delivered store/file holds, read lazily, values untouched."""
    source: str
    kind: str                    # zarr | zarr-implicit | netcdf
    ds: object                   # xarray.Dataset (dask-backed)
    info: dict                   # variable -> metadata as delivered
    attrs: dict                  # root / global attributes
    zarr_format: object = None
    notes: list = field(default_factory=list)


def _array_dims(arr):
    md = arr.metadata
    names = getattr(md, "dimension_names", None)
    if names and all(n is not None for n in names):
        return tuple(str(n) for n in names), "dimension_names (v3)"
    d = arr.attrs.get("_ARRAY_DIMENSIONS")
    if d:
        return tuple(str(n) for n in d), "_ARRAY_DIMENSIONS"
    return None, None


CONVENTIONAL = {2: ("y", "x"), 3: ("t", "y", "x"), 4: ("t", "bands", "y", "x")}


def _infer_dims(name, shape, axis_len, notes):
    """Dims of an array without dimension names, from the coordinate
    lengths. When several coordinates share a length (a square tile, as many
    dates as rows) the conventional order (t, [bands], y, x) of image
    writers decides, and the notes say so: the netCDF comparison catches a
    transposition at once."""
    conv = CONVENTIONAL.get(len(shape))
    dims, used, assumed = [], set(), False
    for k, n in enumerate(shape):
        cands = {ax: cn for ax, (cn, ln) in axis_len.items() if ln == n and ax not in used}
        if len(cands) == 1:
            ax = next(iter(cands))
        elif len(cands) > 1 and conv and conv[k] in cands:
            ax, assumed = conv[k], True
        else:
            raise FormatError(
                f"array {name!r} {tuple(shape)} has no dimension names and its axis {k} (length "
                f"{n}) matches {len(cands) or 'no'} coordinate(s) {list(cands.values())}; "
                f"coordinates seen: {axis_len}")
        dims.append(cands[ax])
        used.add(ax)
    if assumed:
        notes.append(f"{name}: coordinates share a length; read as the conventional order "
                     f"{tuple(dims)}")
    return tuple(dims)


def read_zarr(store):
    """Open one discovered Zarr store with zarr-python into a RawData."""
    import dask.array as da
    import xarray as xr
    import zarr

    path = Path(store["path"])
    notes = []
    if store["arrays"] is None:
        try:
            node = zarr.open(store=str(path), mode="r")
        except Exception as exc:
            arrays = [p for p in sorted(path.iterdir()) if p.is_dir() and _is_array_dir(p)]
            if not arrays:
                raise FormatError(f"{path}: zarr cannot open it ({type(exc).__name__}: {exc}) "
                                  f"and no array folder sits in it") from exc
            notes.append(f"zarr could not open the root ({type(exc).__name__}: {exc})")
            store = {"kind": "zarr-implicit", "path": path, "arrays": arrays}
    if store["arrays"] is None:
        if isinstance(node, zarr.Array):
            members = {path.name: node}
            gattrs = {}
        else:
            members = {n: m for n, m in node.members(max_depth=None) if isinstance(m, zarr.Array)}
            gattrs = _attrs_clean(node.attrs)
        zfmt = node.metadata.zarr_format
    else:
        members = {p.name: zarr.open_array(store=str(p), mode="r") for p in store["arrays"]}
        gattrs = {}
        zfmt = sorted({m.metadata.zarr_format for m in members.values()})
        notes.append("no group metadata at the store root: arrays read one by one")
    if not members:
        raise FormatError(f"{path}: a Zarr store with no arrays")
    # short names (last path component) unless they collide
    short = {}
    for n in members:
        s = n.split("/")[-1]
        short[n] = s if sum(1 for m in members if m.split("/")[-1] == s) == 1 else n.replace("/", "_")
    info, dims = {}, {}
    for n, arr in members.items():
        d, src = _array_dims(arr)
        if d is None and arr.ndim == 1 and axis_of(short[n]):
            d, src = (short[n],), "own name (1-D)"
        dims[n] = d
        info[short[n]] = {"path": n, "shape": list(arr.shape), "dtype": str(arr.dtype),
                          "chunks": list(arr.chunks),
                          "shards": list(arr.shards) if getattr(arr, "shards", None) else None,
                          "fill_value": _py(arr.fill_value), "dims": list(d) if d else None,
                          "dims_source": src, "codecs": _codecs(arr),
                          "zarr_format": arr.metadata.zarr_format,
                          "attrs": _attrs_clean(arr.attrs)}
    axis_len = {}
    for n, arr in members.items():
        if arr.ndim == 1 and dims[n]:
            ax = axis_of(dims[n][0]) or axis_of(short[n])
            if ax and ax not in axis_len:
                axis_len[ax] = (dims[n][0], arr.shape[0])
    for n, arr in members.items():
        if dims[n] is None and arr.ndim >= 2:
            dims[n] = _infer_dims(short[n], arr.shape, axis_len, notes)
            info[short[n]]["dims"] = list(dims[n])
            info[short[n]]["dims_source"] = "inferred from coordinate lengths"
        elif dims[n] is None and arr.ndim == 0:
            dims[n] = ()
        elif dims[n] is None:
            dims[n] = (short[n],)
    variables = {}
    for n, arr in members.items():
        attrs = _attrs_clean(arr.attrs)
        if arr.ndim <= 1:
            data = np.asarray(arr[...])
            if data.dtype.kind in ("T", "O"):          # numpy StringDType / objects
                data = np.array([str(v) for v in np.ravel(data).tolist()],
                                dtype=str).reshape(data.shape)
        else:
            data = da.from_zarr(arr)
        variables[short[n]] = xr.Variable(dims[n], data, attrs)
    coords = {k: v for k, v in variables.items() if v.ndim == 1 and v.dims == (k,)}
    data_vars = {k: v for k, v in variables.items() if k not in coords}
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=gattrs)
    return RawData(source=str(path), kind=store["kind"], ds=ds, info=info, attrs=gattrs,
                   zarr_format=zfmt, notes=notes)


def read_netcdf(path):
    """A delivered netCDF file (``--backend-format netCDF``, or a backend
    that ignored the Zarr request) as RawData, values untouched."""
    import xarray as xr

    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=True, chunks={})
    info = {}
    for n, v in ds.variables.items():
        info[n] = {"shape": list(v.shape), "dtype": str(v.dtype), "dims": list(v.dims),
                   "dims_source": "netCDF", "chunks": v.encoding.get("chunksizes"),
                   "fill_value": _py(v.attrs.get("_FillValue")),
                   "codecs": f"zlib={v.encoding.get('zlib')} complevel={v.encoding.get('complevel')}",
                   "attrs": _attrs_clean(v.attrs)}
    return RawData(source=str(path), kind="netcdf", ds=ds, info=info, attrs=dict(ds.attrs))


def _decode_time(var):
    import pandas as pd

    v = np.asarray(var.values)
    if np.issubdtype(v.dtype, np.datetime64):
        return v.astype("datetime64[ns]")
    if v.dtype.kind in "iuf":
        units = var.attrs.get("units")
        if not units or "since" not in str(units):
            raise FormatError(f"the time coordinate is numeric without CF units "
                              f"(attrs {var.attrs}); cannot tell what {v[:3]} mean")
        from xarray.coding.times import decode_cf_datetime

        out = decode_cf_datetime(v, str(units), var.attrs.get("calendar", "standard"))
        return pd.to_datetime([str(x) for x in np.ravel(out)]).values.astype("datetime64[ns]") \
            if np.asarray(out).dtype == object else np.asarray(out).astype("datetime64[ns]")
    if v.dtype.kind in "USO":
        t = pd.to_datetime(np.asarray(v).astype(str), utc=True)
        return t.tz_convert(None).values.astype("datetime64[ns]")
    raise FormatError(f"time coordinate of dtype {v.dtype} not understood")


# ─────────────────────────────────────────────────────────────────────────────
#  interpretation: delivered layout -> canonical (t, y, x) bands
# ─────────────────────────────────────────────────────────────────────────────

#: non-spectral SENTINEL2_L2A layers recognised after a prefix ('S2_SCL')
OTHER_LAYERS = ("SCL", "CLD", "SNW", "AOT", "WVP", "CLP", "CLM")


def band_key(name):
    """Comparable band name: 'b2', 'B02', 'SENTINEL2_L2A:B02',
    'SENTINEL2_L2A_B02' -> 'B02'; 'b8a' -> 'B8A'; 'scl', 'S2_SCL' -> 'SCL'."""
    s = re.split(r"[:/]", str(name).strip())[-1].upper()
    tail = re.split(r"[_.\-\s]+", s)[-1]
    for cand in (s, tail):
        m = re.fullmatch(r"B0?(\d{1,2})(A?)", cand)
        if m:
            n = int(m.group(1))
            return f"B{n}A" if m.group(2) else f"B{n:02d}"
    return tail if tail in OTHER_LAYERS else s


FILL_ATTRS = ("_FillValue", "missing_value", "nodata", "no_data", "nodata_value", "NoData",
              "no_data_value")
SCALE_ATTRS = ("scale_factor", "add_offset", "scale", "offset")


def _wkt_from(attrs):
    """A CRS from attributes: WKT text, or an EPSG code made into WKT."""
    for key in ("crs_wkt", "spatial_ref", "proj:wkt2", "wkt", "crs"):
        v = attrs.get(key)
        if isinstance(v, str) and ("PROJCS" in v or "PROJCRS" in v or "GEOGCS" in v
                                   or "GEOGCRS" in v):
            return v, key
    for key in ("epsg", "epsg_code", "proj:epsg", "proj:code", "crs", "EPSG"):
        v = attrs.get(key)
        if v is None:
            continue
        m = re.fullmatch(r"(?:EPSG:)?(\d{4,5})", str(v).strip(), re.I)
        if m:
            code = int(m.group(1))
            try:
                from pyproj import CRS

                return CRS.from_epsg(code).to_wkt("WKT1_GDAL"), key
            except Exception:
                return f"EPSG:{code}", key
    return None, None


def _axes_from_transform(raw, ds, crs_hints=()):
    """Pixel-centre x/y from georeferencing attributes when the store has no
    coordinate arrays: GDAL ``GeoTransform`` (x0 dx 0 y0 0 dy) or an affine
    ``transform`` / ``proj:transform`` (dx 0 x0 0 dy y0) on a grid-mapping
    variable, the root, a band, or the job's STAC metadata."""
    if "y" not in ds.dims or "x" not in ds.dims:
        return None
    H, W = ds.sizes["y"], ds.sizes["x"]
    sources = [(f"variable {v!r}", ds[v].attrs) for v in ds.variables
               if str(v).lower() in CRS_NAMES or ds[v].attrs.get("GeoTransform")]
    sources += [("root attributes", raw.attrs)]
    sources += [(f"attributes of {v!r}", ds[v].attrs) for v in ds.data_vars]
    sources += list(crs_hints)
    for label, attrs in sources:
        gt = attrs.get("GeoTransform")
        if gt is not None:
            vals = [float(v) for v in (gt.split() if isinstance(gt, str) else gt)]
            if len(vals) == 6 and vals[2] == 0 and vals[4] == 0:
                x0, dx, _, y0, _, dy = vals
                return (x0 + dx * (np.arange(W) + 0.5), y0 + dy * (np.arange(H) + 0.5),
                        f"GeoTransform of {label}")
        for key in ("proj:transform", "spatial:transform", "transform"):
            tr = attrs.get(key)
            if isinstance(tr, (list, tuple)) and len(tr) in (6, 9):
                a, b, c, d, e, f = (float(v) for v in tr[:6])
                if b == 0 and d == 0:
                    shape = attrs.get("proj:shape") or attrs.get("spatial:shape")
                    if shape and list(shape)[-2:] != [H, W]:
                        continue
                    return (c + a * (np.arange(W) + 0.5), f + e * (np.arange(H) + 0.5),
                            f"{key} of {label}")
    return None


@dataclass
class Interpretation:
    bands: dict                  # band -> DataArray (t, y, x), delivered values
    t: object
    y: object
    x: object
    crs_wkt: str
    crs_source: str
    fills: dict                  # band -> fill (None: none)
    fill_sources: dict
    delivered: dict              # band -> how it was found
    notes: list
    warnings: list
    source: str
    attrs: dict = field(default_factory=dict)    # root attributes as delivered


def interpret(raw, bands, resolution=RESOLUTION_M, crs_hints=(), ref_grid=None,
              require_all=True):
    """Find the requested bands, the axes, the grid and the CRS in one
    RawData, with the values untouched. Raises FormatError, with what was
    seen, whenever the layout leaves a doubt. ``require_all=False`` (one of
    several assets) accepts a subset of the bands, but not none of them."""
    ds = raw.ds
    notes, warns = list(raw.notes), []
    # axes
    rename, axes = {}, {}
    for d in ds.dims:
        ax = axis_of(d)
        if ax and ax not in axes:
            axes[ax] = d
    for ax in ("t", "y", "x"):
        if ax not in axes:
            raise FormatError(f"{raw.source}: no {ax!r} dimension among {list(ds.dims)}")
    for ax, d in axes.items():
        if d != ax:
            rename[d] = ax
    if rename:
        ds = ds.rename(rename)
        notes.append(f"dimensions renamed {rename}")
    # a coordinate stored under another name than its dimension ('time' on 't')
    for ax in ("t", "y", "x", "bands"):
        if ax in ds.dims and ax not in ds.coords:
            alt = [v for v in ds.variables if v != ax and ds[v].dims == (ax,)
                   and axis_of(v) == ax]
            if len(alt) == 1:
                ds = ds.assign_coords({ax: (ax, np.asarray(ds[alt[0]].values), ds[alt[0]].attrs)})
                ds = ds.drop_vars(alt[0])
                notes.append(f"coordinate {alt[0]!r} used for dimension {ax!r}")
    if ("x" not in ds.coords or "y" not in ds.coords):
        got = _axes_from_transform(raw, ds, crs_hints)
        if got is not None:
            xs, ys, src = got
            ds = ds.assign_coords(x=("x", xs), y=("y", ys))
            notes.append(f"x/y computed from {src} (pixel centres)")
    if "t" not in ds.coords:
        raise FormatError(f"{raw.source}: the time dimension has no coordinate values")
    tvals = _decode_time(ds["t"])
    found, delivered, src_attrs = {}, {}, {}
    wanted = {band_key(b): b for b in bands}
    banddim = axes.get("bands")
    if banddim is not None:
        holders = [v for v in ds.data_vars if "bands" in ds[v].dims]
        if len(holders) != 1:
            raise FormatError(f"{raw.source}: {len(holders)} variables carry the bands dimension "
                              f"({holders}); expected one")
        hv = holders[0]
        if "bands" in ds.coords:
            labels = [str(_py(b)) for b in np.asarray(ds["bands"].values)]
        elif ds.sizes["bands"] == len(bands):
            labels = list(bands)
            warns.append(f"variable {hv!r} has a bands dimension without labels; bands taken in "
                         f"the requested order {list(bands)}")
        else:
            raise FormatError(f"{raw.source}: {ds.sizes['bands']} unlabelled bands for "
                              f"{len(bands)} requested")
        for i, lab in enumerate(labels):
            k = band_key(lab)
            if k in wanted:
                found[wanted[k]] = ds[hv].isel(bands=i, drop=True)
                delivered[wanted[k]] = f"{hv}[bands={lab}]"
                src_attrs[wanted[k]] = (hv, ds[hv].attrs)
        notes.append(f"layout: one variable {hv!r} with a bands dimension, labels {labels}")
    else:
        cands = [v for v in ds.data_vars if {"y", "x"} <= set(ds[v].dims) and "t" in ds[v].dims]
        keys = {}
        for v in cands:
            keys.setdefault(band_key(v), []).append(v)
        for k, b in wanted.items():
            hits = keys.get(k, [])
            if len(hits) > 1:
                raise FormatError(f"{raw.source}: several variables look like band {b}: {hits}")
            if hits:
                found[b] = ds[hits[0]]
                delivered[b] = hits[0]
                src_attrs[b] = (hits[0], ds[hits[0]].attrs)
        if not found and len(cands) == len(bands) and all(
                re.fullmatch(r"(band|b|layer|data)?_?\d+", str(c), re.I) for c in cands):
            for c, b in zip(cands, bands):
                found[b], delivered[b], src_attrs[b] = ds[c], c, (c, ds[c].attrs)
            warns.append(f"generic variable names {cands} mapped to the requested order")
        notes.append(f"layout: one variable per band ({len(cands)} band-like variables: {cands})")
    missing = [b for b in bands if b not in found]
    if missing and (require_all or not found):
        raise FormatError(f"{raw.source}: requested bands {missing} not found; data variables "
                          f"{ {v: list(ds[v].dims) for v in ds.data_vars} }")
    others = [v for v in ds.data_vars if v not in {d.split('[')[0] for d in delivered.values()}]
    if others:
        notes.append(f"other variables (not bands, kept out of the store): {others}")
    # dims per band
    for b in list(found):
        a = found[b]
        extra = [d for d in a.dims if d not in ("t", "y", "x")]
        for d in extra:
            if a.sizes[d] == 1:
                a = a.isel({d: 0}, drop=True)
            else:
                raise FormatError(f"band {b} has an extra dimension {d!r} of length {a.sizes[d]}")
        a = a.transpose("t", "y", "x")
        # positions only from here on: the axes travel in the Interpretation
        found[b] = a.drop_vars(list(a.coords))
    # grid
    if "x" not in ds.coords or "y" not in ds.coords:
        raise FormatError(f"{raw.source}: no x/y coordinate values (coords {list(ds.coords)}); "
                          f"refusing to place the pixels on a grid by guesswork")
    x = np.asarray(ds["x"].values, dtype="float64")
    y = np.asarray(ds["y"].values, dtype="float64")
    order_t = np.argsort(tvals, kind="stable")
    if len(tvals) != len(np.unique(tvals)):
        import pandas as pd

        dup = pd.Index(tvals).duplicated(keep=False)
        raise FormatError(f"{raw.source}: duplicated time labels {sorted(set(map(str, tvals[dup])))[:10]}")
    flips = {}
    if len(y) > 1 and y[1] > y[0]:
        flips["y"] = slice(None, None, -1)
        y = y[::-1]
        notes.append("y was ascending: flipped to descending (north up), values untouched")
    if len(x) > 1 and x[1] < x[0]:
        flips["x"] = slice(None, None, -1)
        x = x[::-1]
        notes.append("x was descending: flipped to ascending, values untouched")
    if not np.all(order_t == np.arange(len(tvals))):
        notes.append("time labels were not sorted: sorted")
    for b in found:
        a = found[b]
        if flips:
            a = a.isel(flips)
        if not np.all(order_t == np.arange(len(tvals))):
            a = a.isel(t=order_t)
        found[b] = a
    tvals = tvals[order_t]
    for name, v, sign in (("x", x, 1.0), ("y", y, -1.0)):
        if len(v) > 1:
            d = np.diff(v)
            if not np.allclose(d, sign * resolution, rtol=0, atol=1e-6 * resolution):
                raise FormatError(f"{raw.source}: {name} spacing {np.unique(np.round(d, 6))[:5]} "
                                  f"is not the {resolution} m grid")
    half = resolution / 2.0
    if len(x) and (abs(((x[0] - half) / resolution) - round((x[0] - half) / resolution)) > 1e-6
                   or abs(((y[0] - half) / resolution) - round((y[0] - half) / resolution)) > 1e-6):
        warns.append(f"pixel centres x0={x[0]} y0={y[0]} are not at half-pixel positions of the "
                     f"{resolution} m lattice (the netCDF cubes' are): corners instead of centres?")
    # CRS
    wkt, csrc = None, None
    gm_names = {str(src_attrs[b][1].get("grid_mapping")) for b in found
                if src_attrs[b][1].get("grid_mapping")}
    for gm in sorted(gm_names) + [v for v in ds.variables if str(v).lower() in CRS_NAMES]:
        if gm in ds.variables:
            w, key = _wkt_from(ds[gm].attrs)
            if w:
                wkt, csrc = w, f"variable {gm!r} attribute {key!r}"
                break
    if wkt is None:
        w, key = _wkt_from(raw.attrs)
        if w:
            wkt, csrc = w, f"global attribute {key!r}"
    if wkt is None:
        for b in found:
            w, key = _wkt_from(src_attrs[b][1])
            if w:
                wkt, csrc = w, f"attribute {key!r} of {src_attrs[b][0]!r}"
                break
    if wkt is None:
        for label, hint in crs_hints:
            w, key = _wkt_from(hint)
            if w:
                wkt, csrc = w, f"{label} ({key})"
                break
    if wkt is None and ref_grid is not None:
        ox = zc._overlap(x, ref_grid["x"])
        oy = zc._overlap(y, ref_grid["y"])
        if ox is not None and oy is not None and ref_grid.get("crs_wkt"):
            wkt = ref_grid["crs_wkt"]
            csrc = (f"reference netCDF cube {ref_grid['path']} (the delivered coordinates lie on "
                    f"its grid; the delivered data carry no CRS)")
            warns.append("CRS not in the delivered data: taken from the reference cube")
    if wkt is None:
        raise FormatError(f"{raw.source}: no CRS found (no grid-mapping variable, no crs/EPSG "
                          f"attribute, nothing in the job metadata)")
    # fill and scale, per band
    fills, fsrc = {}, {}
    for b in found:
        vname, attrs = src_attrs[b]
        dtype = found[b].dtype
        declared, key = None, None
        for k in FILL_ATTRS:
            if attrs.get(k) is not None:
                declared, key = _py(attrs[k]), k
                break
        zfill = raw.info.get(vname, {}).get("fill_value")
        if declared is not None:
            fill, src = declared, f"attribute {key!r}"
        elif zfill is not None and not (isinstance(zfill, float) and np.isnan(zfill)
                                        and dtype.kind != "f"):
            fill, src = zfill, "Zarr fill_value"
        else:
            fill, src = (zc.FILL, "none declared: -32768 assumed") if dtype.kind in "iu" \
                else (float("nan"), "none declared: NaN")
        if dtype.kind in "iu" and fill is not None and not isinstance(fill, (int, np.integer)):
            if isinstance(fill, float) and float(fill).is_integer():
                fill = int(fill)
            else:
                raise FormatError(f"band {b}: integer data with a non-integer fill {fill!r}")
        fills[b], fsrc[b] = fill, src
        if np.dtype(dtype).newbyteorder("=") != np.dtype("int16"):
            warns.append(f"band {b} delivered as {dtype}, not int16: kept as delivered")
        elif fill != zc.FILL:
            warns.append(f"band {b}: delivered fill {fill} ({src}), not -32768: kept as delivered")
        for k in SCALE_ATTRS:
            if k in attrs:
                warns.append(f"band {b}: delivered attribute {k}={attrs[k]!r} NOT applied "
                             f"(kept as delivered_{k})")
    return Interpretation(bands=found, t=tvals, y=y, x=x, crs_wkt=wkt, crs_source=csrc,
                          fills=fills, fill_sources=fsrc, delivered=delivered, notes=notes,
                          warnings=warns, source=raw.source, attrs=dict(raw.attrs))


def combine(parts):
    """One Interpretation from several assets: different bands on one grid
    and dates (merge), or the same bands on one grid with disjoint dates
    (concatenate in time). Anything else stops with a FormatError."""
    if len(parts) == 1:
        return parts[0]
    import xarray as xr

    p0 = parts[0]
    for p in parts[1:]:
        if not (np.array_equal(p.x, p0.x) and np.array_equal(p.y, p0.y)):
            raise FormatError(f"assets on different grids ({p0.source} vs {p.source}); the "
                              f"normaliser does not mosaic tiles")
        if not zc.crs_equal(p.crs_wkt, p0.crs_wkt):
            raise FormatError(f"assets in different CRS ({p0.source} vs {p.source})")
    same_t = all(np.array_equal(p.t, p0.t) for p in parts)
    band_sets = [set(p.bands) for p in parts]
    notes = [f"{len(parts)} assets combined"] + [n for p in parts for n in p.notes]
    warns = [w for p in parts for w in p.warnings]
    if same_t and all(not (a & b) for i, a in enumerate(band_sets) for b in band_sets[i + 1:]):
        merged = {}
        for p in parts:
            merged.update(p.bands)
        return Interpretation(
            bands=merged, t=p0.t, y=p0.y, x=p0.x, crs_wkt=p0.crs_wkt, crs_source=p0.crs_source,
            fills={b: f for p in parts for b, f in p.fills.items()},
            fill_sources={b: f for p in parts for b, f in p.fill_sources.items()},
            delivered={b: f"{Path(p.source).name}:{d}" for p in parts for b, d in p.delivered.items()},
            notes=notes + ["assets hold different bands: merged"], warnings=warns,
            source="; ".join(p.source for p in parts), attrs=p0.attrs)
    if all(s == band_sets[0] for s in band_sets):
        allt = np.concatenate([p.t for p in parts])
        if len(np.unique(allt)) != len(allt):
            raise FormatError("assets share bands and overlap in time: cannot combine")
        order = np.argsort(allt, kind="stable")
        bands = {}
        for b in parts[0].bands:
            cat = xr.concat([p.bands[b] for p in parts], dim="t")
            bands[b] = cat.isel(t=order)
        fills = parts[0].fills
        for p in parts[1:]:
            for b in fills:
                if not (fills[b] == p.fills[b] or (isinstance(fills[b], float) and np.isnan(fills[b])
                                                   and isinstance(p.fills[b], float)
                                                   and np.isnan(p.fills[b]))):
                    raise FormatError(f"band {b}: different fills across assets")
        return Interpretation(
            bands=bands, t=allt[order], y=p0.y, x=p0.x, crs_wkt=p0.crs_wkt,
            crs_source=p0.crs_source, fills=fills, fill_sources=p0.fill_sources,
            delivered=p0.delivered, notes=notes + ["assets hold different dates: concatenated"],
            warnings=warns, source="; ".join(p.source for p in parts), attrs=p0.attrs)
    raise FormatError(f"{len(parts)} assets that neither split the bands nor the dates: "
                      f"{[sorted(s) for s in band_sets]}")


def canonical_dataset(interp, bands):
    """The store's dataset (lazy): bands, coordinates, ``crs``."""
    import xarray as xr

    data_vars = {}
    for b in bands:
        a = interp.bands[b]
        attrs = {"long_name": b, "units": "", "grid_mapping": "crs",
                 "native_resolution_m": zc.NATIVE_RES_M.get(b, RESOLUTION_M),
                 "resampling": ("none (native 10 m grid)" if zc.NATIVE_RES_M.get(b) == 10 else
                                "nearest neighbour 20 m -> 10 m (resample_spatial method 'near')"),
                 "description": BAND_NOTES.get(b, SPECTRAL_NOTE),
                 "delivered_as": interp.delivered.get(b, ""),
                 "delivered_dtype": str(a.dtype)}
        for k in SCALE_ATTRS:
            if k in a.attrs:
                attrs[f"delivered_{k}"] = _py(a.attrs[k])
        data = a.data
        native = np.dtype(a.dtype).newbyteorder("=")
        if np.dtype(a.dtype) != native:
            data = data.astype(native)
        data_vars[b] = (("t", "y", "x"), data, attrs)
    ds = xr.Dataset(data_vars=data_vars,
                    coords={"t": ("t", np.asarray(interp.t, dtype="datetime64[ns]")),
                            "y": ("y", np.asarray(interp.y, dtype="float64")),
                            "x": ("x", np.asarray(interp.x, dtype="float64"))})
    ds["t"].attrs = {"standard_name": "time", "long_name": "acquisition date as labelled by openEO",
                     "axis": "T"}
    ds["y"].attrs = {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y",
                     "long_name": "y coordinate of projection (pixel centre)"}
    ds["x"].attrs = {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X",
                     "long_name": "x coordinate of projection (pixel centre)"}
    x, y = interp.x, interp.y
    dx = float(x[1] - x[0]) if len(x) > 1 else float(RESOLUTION_M)
    dy = float(y[1] - y[0]) if len(y) > 1 else -float(RESOLUTION_M)
    gt = f"{x[0] - dx / 2:.6f} {dx:.6f} 0 {y[0] - dy / 2:.6f} 0 {dy:.6f}"
    epsg = zc._epsg_from_wkt(interp.crs_wkt)
    ds["crs"] = xr.DataArray(np.int32(0), attrs={
        "crs_wkt": interp.crs_wkt, "spatial_ref": interp.crs_wkt, "GeoTransform": gt,
        "epsg_code": f"EPSG:{epsg}" if epsg else "", "long_name": "coordinate reference system"})
    return ds


# ─────────────────────────────────────────────────────────────────────────────
#  writing the store (block by block, every block read back)
# ─────────────────────────────────────────────────────────────────────────────

def write_store(ds, fills, out, attrs, log=print, chunks=None, verify=True):
    """Write the canonical dataset to ``out`` (``<out>.tmp`` then rename).

    Template first (metadata and coordinates), then one region write per
    band and time block of ``CHUNKS['t']`` dates (or of the delivered time
    chunk rounded up to a multiple of it, so a delivered chunk is decoded
    once), each read back and compared bit for bit with the delivered block. ``complete = 1`` and the
    consolidated metadata are written last, then the folder is renamed:
    a store under its final name is always whole.
    """
    import dask.array as da
    import xarray as xr
    import zarr

    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    bands = zc.band_names(ds)
    T, H, W = ds.sizes["t"], ds.sizes["y"], ds.sizes["x"]
    ct, cy, cx = zc.chunk_shape((T, H, W), chunks)
    tmpl = xr.Dataset(
        {b: (("t", "y", "x"), da.empty((T, H, W), dtype=ds[b].dtype, chunks=(ct, cy, cx)),
             ds[b].attrs) for b in bands},
        coords={k: ds[k] for k in ("t", "y", "x")})
    tmpl["crs"] = ds["crs"]
    tmpl.attrs = dict(attrs, complete=0, format=zc.FORMAT_ID)
    enc = zc.encoding(tmpl, bands, fills, chunks)
    tmpl.to_zarr(tmp, mode="w", compute=False, encoding=enc, zarr_format=2, consolidated=False)
    dates = zc.dates(ds)
    src_ct = max([int(ds[b].data.chunksize[0]) for b in bands
                  if getattr(ds[b].data, "chunksize", None)] or [1])
    bt = min(T, ct * max(1, -(-src_ct // ct)))
    if bt > ct:
        log(f"the delivered data come in chunks of {src_ct} dates: blocks of {bt} dates "
            f"(~{bt * H * W * np.dtype(ds[bands[0]].dtype).itemsize / 1e9:.1f} GB per band)")
    nblk = (T + bt - 1) // bt
    t0 = time.time()
    for k, a in enumerate(range(0, T, bt)):
        b_ = min(T, a + bt)
        nbytes = 0
        for b in bands:
            block = np.asarray(ds[b].isel(t=slice(a, b_)).values)
            native = block.dtype.newbyteorder("=")
            if block.dtype != native:
                block = block.astype(native)
            nbytes += block.nbytes
            part = xr.Dataset({b: (("t", "y", "x"), da.from_array(block, chunks=(ct, cy, cx)))})
            part.to_zarr(tmp, region={"t": slice(a, b_), "y": slice(0, H), "x": slice(0, W)},
                         mode="r+", consolidated=False)
            if verify:
                back = zarr.open_array(store=str(tmp / b), mode="r")[a:b_]
                same = (np.array_equal(back, block, equal_nan=True) if block.dtype.kind == "f"
                        else np.array_equal(back, block))
                if not same:
                    raise RuntimeError(f"{b}, dates {dates[a]}..{dates[b_ - 1]}: the block read "
                                       f"back from {tmp} differs from the delivered one")
            del block
        log(f"block {k + 1}/{nblk}: {dates[a]}..{dates[b_ - 1]} ({nbytes / 1e6:.0f} MB) "
            f"written{' and read back identical' if verify else ''} "
            f"[{(time.time() - t0) / 60:.1f} min]")
    g = zarr.open_group(str(tmp), mode="r+")
    g.attrs["complete"] = 1
    zarr.consolidate_metadata(str(tmp))
    chk = zc.open_store(tmp)
    if chk.sizes["t"] != T or zc.dates(chk) != dates or sorted(zc.band_names(chk)) != sorted(bands):
        raise RuntimeError(f"{tmp}: the written store does not reopen as written")
    chk.close()
    old = None
    if out.exists():
        old = out.with_name(f"{out.name}.old-{utcnow():%Y%m%dT%H%M%S}")
        os.replace(out, old)
    os.replace(tmp, out)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the format report
# ─────────────────────────────────────────────────────────────────────────────

def _short(v, n=300):
    s = json.dumps(v, default=str, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"


OUTPUT_FORMAT = "output_format.json"


def describe_output_format(conn, fmt, part, log):
    """What the backend says about the output format (GET /file_formats):
    title, flags, parameters; saved as ``<part>/output_format.json`` for the
    FORMAT REPORT. Never fatal."""
    try:
        out = (conn.list_file_formats() or {}).get("output", {})
    except Exception as exc:
        log(f"could not list the backend's file formats ({type(exc).__name__}: {exc})")
        return None
    key = next((k for k in out if k.lower() == str(fmt).lower()), None)
    if key is None:
        log(f"the backend does not list {fmt!r} among its output formats {sorted(out)}")
        return None
    desc = dict(out[key], name=key)
    zc.write_json(desc, Path(part) / OUTPUT_FORMAT)
    for ln in report_output_format(desc):
        log(ln)
    return desc


def report_output_format(desc):
    if not desc:
        return []
    flags = {k: desc[k] for k in ("experimental", "deprecated") if k in desc}
    L = [f"backend output format {desc.get('name')!r}: title {desc.get('title')!r}, gis_data_types "
         f"{desc.get('gis_data_types')}" + (f", {flags}" if flags else "")]
    if desc.get("description"):
        L.append(f"  description: {_short(desc['description'], 400)}")
    params = desc.get("parameters") or {}
    L.append(f"  save_result options: {len(params)}" + (" (pass with --format-options JSON)"
                                                        if params else ""))
    for name, spec in params.items():
        spec = spec if isinstance(spec, dict) else {"description": spec}
        L.append(f"    {name}: type {spec.get('type')}, default {spec.get('default')!r}; "
                 f"{_short(str(spec.get('description', '')), 240)}")
    return L


def report_job(info, meta, files):
    L = []
    if info:
        L.append(f"job: {info.get('id')} status {info.get('status')} created {info.get('created')} "
                 f"updated {info.get('updated')}")
        if info.get("costs") is not None or info.get("usage"):
            L.append(f"  costs {info.get('costs')} usage {_short(info.get('usage'), 600)}")
    if meta:
        L.append(f"results metadata: type {meta.get('type')} stac_version "
                 f"{meta.get('stac_version')} keys {sorted(meta)}")
        links = meta.get("links") or []
        rels = {}
        for lk in links:
            rels[lk.get("rel")] = rels.get(lk.get("rel"), 0) + 1
        L.append(f"  links by rel: {rels}")
        assets = meta.get("assets") or {}
        L.append(f"  assets listed: {len(assets)}")
        for k, (name, a) in enumerate(sorted(assets.items())):
            if k >= 40:
                L.append(f"  ... {len(assets) - 40} more")
                break
            extra = {kk: a[kk] for kk in a if kk.startswith(("proj:", "file:", "eo:", "raster:"))
                     or kk in ("bands", "roles", "type", "title")}
            host = re.sub(r"^(https?://[^/]+).*$", r"\1", str(a.get("href", "")))
            L.append(f"  asset {name!r}: {_short(extra, 500)} href {host}/...")
    if files:
        tot = sum(v.get("size", 0) for v in files.values())
        L.append(f"downloaded: {len(files)} file(s), {tot / 1e6:.1f} MB")
        for k, (rel, v) in enumerate(sorted(files.items())):
            if k >= 20:
                L.append(f"  ... {len(files) - 20} more")
                break
            L.append(f"  {rel}  {v.get('size', 0) / 1e6:.2f} MB  type {v.get('type')}")
    return L


def report_tree(path, max_lines=30):
    """Directory layout: folders with their file counts and sizes."""
    L, rows = [], []
    path = Path(path)
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        size = sum((Path(dirpath) / f).stat().st_size for f in filenames)
        rel = Path(dirpath).relative_to(path)
        sample = sorted(filenames)[:6]
        rows.append(f"  {str(rel) if str(rel) != '.' else '.'}/  {len(filenames)} files "
                    f"{size / 1e6:.2f} MB  e.g. {sample}")
    L.extend(rows[:max_lines])
    if len(rows) > max_lines:
        L.append(f"  ... {len(rows) - max_lines} more folders")
    return L


def report_raw(raw):
    L = [f"store {raw.source}: kind {raw.kind}, Zarr format {raw.zarr_format}"]
    if raw.attrs:
        L.append(f"  root attributes: {_short(raw.attrs, 800)}")
    for n, i in sorted(raw.info.items()):
        L.append(f"  {n:<14} shape {i['shape']} {i['dtype']} chunks {i.get('chunks')}"
                 + (f" shards {i['shards']}" if i.get("shards") else "")
                 + f" fill {i.get('fill_value')!r} dims {i.get('dims')} ({i.get('dims_source')})")
        L.append(f"  {'':<14} codecs {i.get('codecs')}; attrs {_short(i.get('attrs'), 400)}")
    for n in raw.notes:
        L.append(f"  note: {n}")
    return L


def report_xarray_direct(path):
    import warnings
    import xarray as xr

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_zarr(str(path), consolidated=None, mask_and_scale=False)
        msg = f"xarray.open_zarr opens the delivered store directly: dims {dict(ds.sizes)}, " \
              f"variables {list(ds.data_vars)}"
        ds.close()
        return [msg]
    except Exception as exc:
        return [f"xarray.open_zarr cannot open the delivered store as is "
                f"({type(exc).__name__}: {str(exc)[:200]}); read with zarr-python instead"]


def report_interp(interp, ds):
    import pandas as pd

    t = pd.DatetimeIndex(interp.t)
    L = ["interpretation:"]
    L.append(f"  time: {len(t)} dates {t.min().date() if len(t) else ''} .. "
             f"{t.max().date() if len(t) else ''}; times of day {sorted(set(t.strftime('%H:%M:%S')))}")
    if len(t) <= 40:
        L.append(f"  dates: {[str(d.date()) for d in t]}")
    x, y = interp.x, interp.y
    L.append(f"  grid: {len(y)} rows x {len(x)} cols; x {x[0]:.1f}..{x[-1]:.1f}, y {y[0]:.1f}.."
             f"{y[-1]:.1f} (pixel centres), spacing {x[1] - x[0] if len(x) > 1 else '?'} / "
             f"{y[1] - y[0] if len(y) > 1 else '?'} m")
    L.append(f"  CRS: EPSG {zc._epsg_from_wkt(interp.crs_wkt)} from {interp.crs_source}")
    for b in ds.data_vars:
        if b == "crs":
            continue
        L.append(f"  {b:<4} <- {interp.delivered.get(b)}: dtype {ds[b].attrs.get('delivered_dtype')}, "
                 f"fill {interp.fills.get(b)!r} ({interp.fill_sources.get(b)})")
    for n in interp.notes:
        if n:
            L.append(f"  note: {n}")
    for w in interp.warnings:
        L.append(f"  WARNING: {w}")
    return L


def report_summary(files, found, raws, interp):
    """The answers in a few lines: assets and layout, variables and dims,
    dtypes, fills, scale attributes, chunks, CRS, time, band naming."""
    import pandas as pd

    types = sorted({str(v.get("type")) for v in files.values()})
    zips = [k for k, v in files.items() if k.lower().endswith(".zip") or "zip" in str(v.get("type"))]
    L = ["SUMMARY",
         f"  assets: {len(files)} file(s), types {types}; zipped: {'yes (' + ', '.join(zips[:3]) + ')' if zips else 'no'}",
         f"  layout: {len(found['stores'])} Zarr store(s) "
         f"{sorted({str(r.zarr_format) for r in raws if r.kind != 'netcdf'})} "
         f"[{', '.join(sorted({s['kind'] for s in found['stores']}))}], {len(found['netcdf'])} netCDF; "
         + next((n for n in interp.notes if n.startswith("layout:")), "layout: ?")]
    by_var = {}
    for r in raws:
        for name, i in r.info.items():
            by_var.setdefault(name, i)
    held = []
    for b, d in interp.delivered.items():
        info = by_var.get(d.split("[")[0]) or by_var.get(d.split(":")[-1].split("[")[0], {})
        held.append((b, d, info))
    dims = sorted({tuple(i.get("dims") or ()) for _, _, i in held})
    dtypes = sorted({i.get("dtype") for _, _, i in held if i.get("dtype")})
    chunks = sorted({str(i.get("chunks")) for _, _, i in held})
    codecs = sorted({str(i.get("codecs")) for _, _, i in held})
    L.append(f"  band naming: {', '.join(f'{b}<-{d}' for b, d, _ in held)}")
    L.append(f"  dims as delivered: {dims}; dtypes {dtypes}; chunks {chunks}")
    L.append(f"  codecs: {codecs[:2]}{' ...' if len(codecs) > 2 else ''}")
    fills = sorted({f"{interp.fills[b]!r} ({interp.fill_sources[b]})" for b in interp.fills})
    L.append(f"  no-data: {fills}")
    scale = {b: {k: i.get('attrs', {}).get(k) for k in SCALE_ATTRS if k in i.get('attrs', {})}
             for b, _, i in held}
    scale = {b: v for b, v in scale.items() if v}
    L.append(f"  scale/offset attributes: {scale if scale else 'none'} (never applied)")
    x, y = interp.x, interp.y
    L.append(f"  grid: EPSG {zc._epsg_from_wkt(interp.crs_wkt)} from {interp.crs_source}; "
             f"{len(y)} x {len(x)} px, spacing {x[1] - x[0] if len(x) > 1 else '?'} / "
             f"{y[1] - y[0] if len(y) > 1 else '?'} m, first centre ({x[0]}, {y[0]})")
    tinfo = next((r.info.get(n) for r in raws for n in ("t", "time", "times", "date", "dates")
                  if n in r.info), {}) or {}
    t = pd.DatetimeIndex(interp.t)
    L.append(f"  time: {len(t)} labels, stored as {tinfo.get('dtype')} "
             f"{tinfo.get('attrs', {}).get('units', '')!r}; times of day "
             f"{sorted(set(t.strftime('%H:%M:%S')))[:5]}")
    if interp.warnings:
        L.append(f"  warnings: {len(interp.warnings)} (above)")
    return L


def block_equal_share(a, x, y, valid, cell=20.0):
    """Share of adjacent pixel pairs inside one ``cell`` m cell of the UTM
    lattice that hold the same value: 1.0 when a 20 m band reached 10 m by
    nearest neighbour from its native grid, well below 1 when interpolated."""
    ix = np.floor(np.asarray(x) / cell)
    iy = np.floor(np.asarray(y) / cell)
    sc, sr = ix[1:] == ix[:-1], iy[1:] == iy[:-1]
    eq_c = (a[:, 1:] == a[:, :-1])[:, sc]
    ok_c = (valid[:, 1:] & valid[:, :-1])[:, sc]
    eq_r = (a[1:, :] == a[:-1, :])[sr, :]
    ok_r = (valid[1:, :] & valid[:-1, :])[sr, :]
    n = int(ok_c.sum() + ok_r.sum())
    return float((eq_c[ok_c].sum() + eq_r[ok_r].sum()) / n) if n else float("nan")


def report_resampling(ds, fills, date_index=0):
    """How the 20 m bands reached the 10 m grid, measured on one date."""
    x, y = ds["x"].values, ds["y"].values
    shares = {}
    for b in zc.band_names(ds):
        a = np.asarray(ds[b].isel(t=date_index).values)
        f = fills.get(b)
        valid = ~np.isnan(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
        if f is not None and not (isinstance(f, float) and np.isnan(f)):
            valid &= a != f
        shares[b] = block_equal_share(a, x, y, valid)
    fmt = ", ".join(f"{b} {v * 100:.1f} %" for b, v in shares.items())
    return [f"resampling check on {zc.dates(ds)[date_index]}: share of pixel pairs inside one "
            f"20 m cell of the UTM lattice with equal values: {fmt}",
            "  (100 % for B8A B11 B12 SCL CLD = nearest neighbour from the native 20 m grid; the "
            "10 m bands give the share expected by chance)"]


def report_values(ds, fills, date_index=0):
    """min / max / no-data / zeros of every band on one date."""
    L = [f"values on {zc.dates(ds)[date_index]} (first date):"]
    for b in zc.band_names(ds):
        a = np.asarray(ds[b].isel(t=date_index).values)
        f = fills.get(b)
        nod = (np.isnan(a) if a.dtype.kind == "f" else np.zeros(a.shape, bool))
        if f is not None and not (isinstance(f, float) and np.isnan(f)):
            nod |= a == f
        v = a[~nod]
        pct = (np.percentile(v, [1, 50, 99]).round(1).tolist() if v.size else "-")
        L.append(f"  {b:<4} min {v.min() if v.size else '-'} max {v.max() if v.size else '-'} "
                 f"p1/p50/p99 {pct} no-data {nod.mean() * 100:.1f} % zeros "
                 f"{(a == 0).mean() * 100:.1f} % (-32768: {(a == -32768).sum()} px)")
    spectral = [b for b in zc.band_names(ds) if b not in ("SCL", "CLD")]
    if spectral:
        L.append("  (spectral bands: L2A digital numbers as delivered. Since processing baseline "
                 "04.00 (2022-01-25) the L2A products carry BOA_ADD_OFFSET = -1000: if p1 of "
                 "B11/B12 over water sits near 1000 the offset is still in the values)")
    return L


# ─────────────────────────────────────────────────────────────────────────────
#  verification and overpass times
# ─────────────────────────────────────────────────────────────────────────────

def reference_grid(site):
    path = ROOT / SITE_CONFIG[site]["reference"]
    if not path.is_file():
        return None
    import xarray as xr

    with xr.open_dataset(path, mask_and_scale=False) as r:
        g = zc.grid_info(r)
        return {"path": str(path), "x": np.asarray(r["x"].values), "y": np.asarray(r["y"].values),
                "crs_wkt": g["crs_wkt"]}


def verify_request(req, log):
    """Compare the store with the site's netCDF cube; JSON beside the store."""
    cfg = SITE_CONFIG[req.site]
    ref = ROOT / cfg["reference"]
    base = {"site": req.site, "year": req.year, "window": [req.start, req.end],
            "store": str(req.store), "reference": str(ref),
            "checked_utc": utcnow().isoformat(timespec="seconds")}
    p0, p1 = cfg["reference_period"]
    if not ref.is_file():
        rep = dict(base, verdict="no reference", reason=f"{ref} is not on this machine")
    elif not (req.start < p1 and req.end > p0):
        rep = dict(base, verdict="no reference",
                   reason=f"the reference cube covers [{p0}, {p1}), not this window")
    else:
        rep = zc.compare_with_netcdf(req.store, ref, bands=("B03", "B08", "SCL"),
                                     start=req.start, end=req.end, ref_period=(p0, p1),
                                     log=lambda m: log(f"  {m}"))
        rep = dict(base, **rep)
    others = [y for y in zc.list_years(req.site, req.root) if y != req.year]
    if others and zc.is_complete(req.store):
        a = zc.open_store(req.store)
        b = zc.open_store(zc.store_path(req.site, others[0], req.root))
        rep["same_grid_as_year"] = {str(others[0]): bool(zc._same_grid(a, b))}
        a.close()
        b.close()
    zc.write_json(rep, req.sidecar("verify.json"))
    for line in zc.report_lines(rep) if rep.get("dates") else [
            f"verification: {rep['verdict']} ({rep.get('reason')})"]:
        log(line)
    return rep


def update_overpass(req, dates, log, network=True):
    """Overpass times of the store's dates into ``<site>/overpass_times.json``.

    Seeded from the pipeline's own file of the site (read only), so a date
    keeps the time the existing products used; the rest asked to the STAC
    catalogue, like ``experiments.download_cube.check_overpass``. Returns
    the dates still without a time.
    """
    import pandas as pd

    path = zc.overpass_path(req.site, req.root)
    have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    missing = [d for d in dates if d not in have]
    seeded = 0
    pipe = ROOT / SITE_CONFIG[req.site]["pipeline_overpass"]
    if missing and pipe.is_file():
        old = json.loads(pipe.read_text(encoding="utf-8"))
        for d in missing:
            if d in old:
                have[d] = pd.Timestamp(old[d]).to_pydatetime().isoformat()
                seeded += 1
    missing = [d for d in dates if d not in have]
    asked = 0
    if missing and network:
        import pyintertidal as pit

        pit.net.use_system_certificates()
        fresh = pit.overpass.get_overpass_times(req.bbox, (min(missing), max(missing)),
                                                verbose=False)
        for d in missing:
            if d in fresh:
                have[d] = pd.Timestamp(fresh[d]).to_pydatetime().isoformat()
                asked += 1
    zc.write_json(dict(sorted(have.items())), path)
    still = [d for d in dates if d not in have]
    log(f"overpass times: {len(dates) - len(still)} of {len(dates)} dates in {path} ({seeded} from "
        f"{pipe.name if pipe.is_file() else 'no pipeline file'}, {asked} from the catalogue)"
        + (f"; STILL MISSING {still}" if still else ""))
    return still


# ─────────────────────────────────────────────────────────────────────────────
#  one store, start to end
# ─────────────────────────────────────────────────────────────────────────────

class Context:
    """Run-wide settings and the cached openEO connection."""

    def __init__(self, args=None, connect=None, sleep=time.sleep):
        a = args or argparse.Namespace()
        self.job_options = (json.loads(a.job_options) if getattr(a, "job_options", None)
                            else dict(JOB_OPTIONS))
        self.format_options = (json.loads(a.format_options) if getattr(a, "format_options", None)
                               else None)
        self.backend_format = getattr(a, "backend_format", None) or BACKEND_FORMAT
        self.retries = int(getattr(a, "retries", 1))
        self.max_wait_s = float(getattr(a, "max_wait_hours", 24.0)) * 3600.0
        self.keep_raw = bool(getattr(a, "keep_raw", False))
        self.allow_offset = bool(getattr(a, "allow_grid_offset", False))
        self.network = True
        self.poll_s, self.poll_max_s, self.capacity_wait_s = 10.0, 60.0, 300
        self.sleep = sleep
        self._connect = connect
        self._conn = None

    def connection(self):
        if self._conn is None:
            if self._connect is not None:
                self._conn = self._connect()
            else:
                import pyintertidal as pit

                pit.net.use_system_certificates()
                self._conn = pit.scenes.connect(interactive=False)
        return self._conn


def lattice_check(interp, ref_grid, whole=True):
    """``(error, warning)`` of the delivered grid against the existing cube's.

    error: another CRS, or pixel centres off the cube's 10 m lattice (half a
    pixel would be corners delivered as centres): such a store would not be
    on the grid of the existing cubes. warning: same lattice, another extent
    (expected for --test, ``whole=False``)."""
    if ref_grid is None:
        return None, None
    if ref_grid.get("crs_wkt") and not zc.crs_equal(interp.crs_wkt, ref_grid["crs_wkt"]):
        return (f"the delivered CRS (EPSG {zc._epsg_from_wkt(interp.crs_wkt)}) is not the CRS of "
                f"{ref_grid['path']} (EPSG {zc._epsg_from_wkt(ref_grid['crs_wkt'])})"), None
    rx = np.asarray(ref_grid["x"], dtype="float64")
    ry = np.asarray(ref_grid["y"], dtype="float64")
    if len(rx) < 2 or len(ry) < 2 or not len(interp.x) or not len(interp.y):
        return None, None
    dxr, dyr = float(rx[1] - rx[0]), float(ry[1] - ry[0])
    if not (np.isclose(abs(dxr), RESOLUTION_M) and np.isclose(abs(dyr), RESOLUTION_M)):
        return None, None                  # a reference on another grid says nothing here
    fx = ((float(interp.x[0]) - rx[0]) / dxr) % 1.0
    fy = ((float(interp.y[0]) - ry[0]) / dyr) % 1.0
    fx, fy = min(fx, 1.0 - fx), min(fy, 1.0 - fy)
    if fx > 1e-3 or fy > 1e-3:
        return (f"the delivered pixel centres are offset by ({fx:.3f}, {fy:.3f}) px from the grid "
                f"of {ref_grid['path']} (0.5 = pixel corners delivered as centres?)"), None
    if whole and (len(interp.x) != len(rx) or len(interp.y) != len(ry)
                  or interp.x[0] != rx[0] or interp.y[0] != ry[0]):
        return None, (f"same 10 m lattice as {Path(ref_grid['path']).name} but another extent: "
                      f"{len(interp.y)} x {len(interp.x)} px from ({interp.x[0]}, {interp.y[0]}) "
                      f"against {len(ry)} x {len(rx)} px from ({rx[0]}, {ry[0]})")
    return None, None


def normalise(req, log, job_info=None, results_meta=None, ref_grid=None, report_all=False,
              backend_format=BACKEND_FORMAT, allow_offset=False):
    """Everything between the download and the store: extract, find, read,
    report, interpret. Returns ``(canonical dataset, interpretation, report
    lines)``; the report is also written to ``<year>.format.txt``. A grid
    off the existing cube's lattice (``lattice_check``) stops here unless
    ``allow_offset``."""
    part = Path(req.part)
    files = {}
    if (part / ASSETS_DONE).is_file():
        files = json.loads((part / ASSETS_DONE).read_text(encoding="utf-8"))["files"]
    extracted = extract_zips(part, log) if files else []
    found = discover([part / "assets", part / "extracted"])
    lines = ["=" * 30 + f" FORMAT REPORT {req.tag} " + "=" * 30,
             f"request: {COLLECTION} bands {list(req.bands)} window [{req.start}, {req.end}) bbox "
             f"{req.bbox}"]
    if (part / OUTPUT_FORMAT).is_file():
        lines += report_output_format(json.loads((part / OUTPUT_FORMAT).read_text(encoding="utf-8")))
    lines += report_job(job_info, results_meta, files)
    for d in extracted:
        lines.append(f"zip extracted to {d}:")
        lines += report_tree(d)
    lines.append(f"found: {len(found['stores'])} Zarr store(s), {len(found['netcdf'])} netCDF, "
                 f"{len(found['geotiff'])} GeoTIFF, {len(found['other'])} other file(s)")
    raws, problems = [], []
    for s in found["stores"]:
        try:
            r = read_zarr(s)
            raws.append(r)
            lines += report_raw(r)
            if s["kind"] == "zarr" and report_all:
                lines += ["  " + x for x in report_xarray_direct(s["path"])]
        except FormatError as exc:
            problems.append(str(exc))
            lines.append(f"  store {s['path']}: {exc}")
    if not found["stores"]:
        for p in found["netcdf"]:
            r = read_netcdf(p)
            raws.append(r)
            lines += report_raw(r)
        if found["netcdf"] and str(backend_format).lower() == "zarr":
            lines.append("  WARNING: Zarr was requested but the backend delivered netCDF: the "
                         "store is normalised from it")
    meta = results_meta or {}
    crs_hints = [("job results metadata", meta.get("properties") or {})]
    crs_hints += [(f"job results asset {k!r}", a) for k, a in (meta.get("assets") or {}).items()
                  if isinstance(a, dict)]
    try:
        if not raws:
            hint = ""
            if any("zarr" in str(v.get("type")).lower() for v in files.values()):
                hint = (" The backend announced a Zarr asset that did not download as a store "
                        "(a directory behind one href?): see the parameters of the Zarr format "
                        "in the report of --test and pass one that zips the store with "
                        "--format-options.")
            raise FormatError(f"nothing readable among the results: {found['geotiff'][:3]} "
                              f"{found['other'][:5]} {problems}.{hint}")
        if len(raws) == 1:
            parts = [interpret(raws[0], req.bands, crs_hints=crs_hints, ref_grid=ref_grid)]
        else:                       # several assets: each may hold a subset of the bands
            parts = []
            for r in raws:
                try:
                    parts.append(interpret(r, req.bands, crs_hints=crs_hints, ref_grid=ref_grid,
                                           require_all=False))
                except FormatError as exc:
                    problems.append(str(exc))
                    lines.append(f"  asset skipped: {exc}")
            if not parts:
                raise FormatError(f"none of the {len(raws)} delivered stores/files holds the "
                                  f"requested bands: {problems}")
        interp = combine(parts)
        missing = [b for b in req.bands if b not in interp.bands]
        if missing:
            raise FormatError(f"requested bands {missing} not found in any of the {len(raws)} "
                              f"delivered stores/files (found {sorted(interp.bands)}); "
                              f"skipped: {problems}")
        err, warn = lattice_check(interp, ref_grid, whole=not req.test)
        if warn:
            interp.warnings.append(warn)
        if err and allow_offset:
            interp.warnings.append(err + " (written anyway: --allow-grid-offset)")
        elif err:
            raise FormatError(err + "; nothing written (run again with --allow-grid-offset to "
                              "write the store anyway)")
        ds = canonical_dataset(interp, req.bands)
        lines += report_interp(interp, ds)
        lines += report_values(ds, interp.fills)
        lines += report_resampling(ds, interp.fills)
        lines += report_summary(files, found, raws, interp)
        lines.append("=> normalisable: " + ("YES" if not interp.warnings else
                                             "YES, with the warnings above"))
    except FormatError as exc:
        lines.append(f"=> NOT NORMALISABLE: {exc}")
        lines.append(f"   the backend's files stay in {req.part}; nothing was written to {req.store}")
        lines.append("=" * 76)
        req.sidecar("format.txt").parent.mkdir(parents=True, exist_ok=True)
        Path(req.sidecar("format.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        for ln in lines:
            log(ln)
        raise
    lines.append("=" * 76)
    req.sidecar("format.txt").parent.mkdir(parents=True, exist_ok=True)
    Path(req.sidecar("format.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ds, interp, lines


def provenance(req, job_info, results_meta, interp, ctx, flat_graph):
    import openeo
    import xarray as xr
    import zarr
    import numcodecs

    return {
        "title": f"Sentinel-2 L2A cube, {req.site} {req.year}" + (" (format test)" if req.test else ""),
        "Conventions": "CF-1.9",
        "site": req.site, "year": int(req.year),
        "temporal_extent": json.dumps([req.start, req.end]),
        "temporal_extent_note": "end-exclusive, as openEO",
        "spatial_extent": json.dumps(req.bbox),
        "collection": COLLECTION, "backend": f"https://{BACKEND}",
        "job_id": (job_info or {}).get("id", ""), "job_title": (job_info or {}).get("title", ""),
        "job_created": str((job_info or {}).get("created", "")),
        "job_updated": str((job_info or {}).get("updated", "")),
        "job_costs": json.dumps((job_info or {}).get("costs")),
        "job_usage": json.dumps((job_info or {}).get("usage"))[:4000],
        "job_options": json.dumps(ctx.job_options),
        "backend_format": ctx.backend_format,
        "backend_format_options": json.dumps(ctx.format_options or {}),
        "backend_version": str((results_meta or {}).get("openeo:version", "") or
                               (results_meta or {}).get("properties", {}).get("openeo:version", "")),
        "processing_date": utcnow().isoformat(timespec="seconds") + "Z",
        "bands": json.dumps(list(req.bands)),
        "resampling": RESAMPLING,
        "max_cloud_cover": MAX_CLOUD_COVER,
        "process_graph": json.dumps(flat_graph),
        "delivered": interp.source,
        "delivered_attributes": json.dumps(interp.attrs, default=str)[:4000],
        "delivered_notes": json.dumps(interp.notes),
        "delivered_warnings": json.dumps(interp.warnings),
        "crs_source": interp.crs_source,
        "fill_sources": json.dumps(interp.fill_sources),
        "chunks": json.dumps(zc.CHUNKS), "compressor": json.dumps(zc.COMPRESSOR),
        "grid_epsg": int(zc._epsg_from_wkt(interp.crs_wkt) or 0),
        "software": json.dumps({"python": sys.version.split()[0], "xarray": xr.__version__,
                                "zarr": zarr.__version__, "numcodecs": numcodecs.__version__,
                                "openeo": openeo.__version__}),
        "written_by": "experiments/download_cube_zarr.py",
    }


def run_request(ctx, req, log):
    """Download, normalise, verify and time one store. Returns a summary."""
    t0 = time.time()
    if not req.test and zc.is_complete(req.store):
        log(f"{req.store} is complete: nothing to download")
        ds = zc.open_store(req.store)
        dates = zc.dates(ds)
        ds.close()
        if not req.sidecar("verify.json").is_file():
            verify_request(req, log)
        still = update_overpass(req, dates, log, network=ctx.network)
        return {"status": "skipped", "dates": len(dates), "missing_overpass": still}
    part = Path(req.part)
    part.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(part).free
    log(f"start: window [{req.start}, {req.end}) bbox {req.bbox}; {free / 1e9:.0f} GB free")
    flat = build_process(req, None, ctx.backend_format, ctx.format_options).flat_graph()
    title = job_title(req, graph_hash(flat))
    led_path = part / LEDGER
    if (part / ASSETS_DONE).is_file():
        led = Ledger(part)
        cur = led.current() or {}
        info = cur.get("info", {})
        log(f"results of job {info.get('id', cur.get('job_id'))} already downloaded in {part}")
        meta = json.loads((part / "job-results.json").read_text(encoding="utf-8")) \
            if (part / "job-results.json").is_file() else {}
    else:
        if req.test:
            describe_output_format(ctx.connection(), ctx.backend_format, part, log)
        job, info = ensure_finished_job(ctx, req, title, log)
        info = job_summary(info)
        try:
            meta = download_results(job, req, log)
        except Exception:
            if results_gone(job):
                led = Ledger(part)
                cur = led.current()
                if cur is not None and cur.get("job_id") == job.job_id:
                    led.update(cur, status="expired")
                log(f"the results of job {job.job_id} are no longer on the backend: the next run "
                    f"makes a new job")
            raise
    ref = reference_grid(req.site)
    ds, interp, lines = normalise(req, log, job_info=info, results_meta=meta, ref_grid=ref,
                                  allow_offset=ctx.allow_offset,
                                  report_all=req.test, backend_format=ctx.backend_format)
    if req.test:
        for ln in lines:
            log(ln)
    else:
        for ln in lines:
            if ln.startswith(("=", "found:", "store ", "  WARNING", "interpretation", "  CRS",
                              "  time:", "  grid:")):
                log(ln)
    attrs = provenance(req, info, meta, interp, ctx, flat)
    log(f"writing {req.store} ({ds.sizes['t']} dates, {ds.sizes['y']} x {ds.sizes['x']} px, "
        f"{len(req.bands)} bands)")
    write_store(ds, interp.fills, req.store, attrs, log=log)
    size = sum(f.stat().st_size for f in Path(req.store).rglob("*") if f.is_file())
    log(f"store complete: {req.store} ({size / 1e9:.2f} GB on disk)")
    job_side = {"ledger": json.loads(led_path.read_text(encoding="utf-8"))
                if led_path.is_file() else {}, "job": info, "results_metadata": meta}
    zc.write_json(job_side, req.sidecar("job.json"))
    if req.test:
        cube = zc.open_cube(req.site, req.start, req.end, root=req.root)
        log(f"open_cube({req.site!r}, {req.start!r}, {req.end!r}, root={str(req.root)!r}): "
            f"{dict(cube.sizes)}; dates {zc.dates(cube)}")
        log(f"  bands {zc.band_names(cube)}; dtypes "
            f"{sorted({str(cube[b].dtype) for b in zc.band_names(cube)})}; fill "
            f"{sorted({str(zc.fill_of(cube[b])) for b in zc.band_names(cube)})}; chunks "
            f"{cube['B03'].encoding.get('chunks')}; CRS EPSG {zc.grid_info(cube)['epsg']}")
        cube.close()
    rep = verify_request(req, log)
    dates = zc.dates(ds)
    still = update_overpass(req, dates, log, network=ctx.network)
    if not (ctx.keep_raw or req.test):
        shutil.rmtree(part, ignore_errors=True)
        log(f"raw download removed ({part}); the store keeps the provenance")
    else:
        log(f"raw download kept in {part}")
    log(f"done in {(time.time() - t0) / 60:.0f} min")
    return {"status": "done", "dates": len(dates), "verdict": rep.get("verdict"),
            "missing_overpass": still, "gb": round(size / 1e9, 3), "job": info.get("id")}


# ─────────────────────────────────────────────────────────────────────────────
#  dry run and verify-only
# ─────────────────────────────────────────────────────────────────────────────

def _size(nbytes):
    """Bytes as a short human string (MB below 1 GB)."""
    return f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.1f} MB"


def dry_run(reqs, ctx):
    """Process graphs, outputs, their state and a size estimate; no network."""
    total = {"px_dates": 0.0, "raw": 0.0, "disk": 0.0, "biggest_raw": 0.0, "todo": 0}
    shown = set()
    refs = {}
    for req in reqs:
        sr = build_process(req, None, ctx.backend_format, ctx.format_options)
        flat = sr.flat_graph()
        title = job_title(req, graph_hash(flat))
        print(f"\n== {req.tag}: window [{req.start}, {req.end}) bbox {req.bbox}", flush=True)
        print(f"   job '{title}', format {ctx.backend_format} {ctx.format_options or {}}, "
              f"job options {ctx.job_options}", flush=True)
        if req.site not in shown:
            print("   process graph:\n" + "\n".join("     " + ln for ln in
                                                     json.dumps(flat, indent=2).splitlines()),
                  flush=True)
            shown.add(req.site)
        else:
            print("   process graph: as above, temporal_extent "
                  f"{flat['loadcollection1']['arguments']['temporal_extent']}", flush=True)
        complete = (not req.test) and zc.is_complete(req.store)
        state = ("COMPLETE (skipped)" if complete else
                 "partial download present" if Path(req.part).exists() else "to download")
        led = Ledger(req.part).current() if (Path(req.part) / LEDGER).is_file() else None
        print(f"   outputs: {req.store} [{state}]" + (f"; ledger job {led['job_id']} "
                                                       f"({led.get('status')})" if led else ""),
              flush=True)
        print(f"            {req.sidecar('verify.json').name}, {req.sidecar('format.txt').name}, "
              f"{req.sidecar('job.json').name}, {zc.overpass_path(req.site, req.root)}", flush=True)
        if req.site not in refs:
            refs[req.site] = None if req.test else reference_summary(req.site)
        ref = refs[req.site]
        if ref and not req.test:
            shape, gsrc = ref["shape"], f"existing cube {Path(ref['path']).name}"
        else:
            g = predict_grid(req.bbox)
            shape, gsrc = g["shape"], f"bbox envelope on the 10 m lattice, EPSG:{g['epsg']}"
        days = int((np.datetime64(req.end) - np.datetime64(req.start)).astype(int))
        if ref and req.year in ref["per_year"] and not req.test and days >= 365:
            n = ref["per_year"][req.year]
            dsrc = "dates of that year in the existing cube"
        else:
            n = TYPICAL_DATES.get(req.year, 145) * min(1.0, days / 365.0)
            dsrc = "typical count for the year"
        px = shape[0] * shape[1]
        raw = px * n * 2 * len(req.bands)
        disk = px * n * BYTES_PER_PIXEL_DATE
        if not complete:
            total["px_dates"] += px * n
            total["raw"] += raw
            total["disk"] += disk
            total["biggest_raw"] = max(total["biggest_raw"], raw)
            total["todo"] += 1
        print(f"   grid {shape[0]} x {shape[1]} px ({gsrc}); ~{n:.0f} dates ({dsrc})", flush=True)
        print(f"   size: {px * n / 1e9:.3f} G pixel-dates; {_size(raw)} as int16, ~{_size(disk)} "
              f"on disk (estimate {BYTES_PER_PIXEL_DATE} B per pixel-date)", flush=True)
    root = Path(reqs[0].root) if reqs else zc.cubes_root()
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print(f"\nTOTAL {total['todo']} store(s) to make (of {len(reqs)}): "
          f"{total['px_dates'] / 1e9:.2f} G pixel-dates, {_size(total['raw'])} as int16, "
          f"~{_size(total['disk'])} on disk; while a year is processed its raw download also "
          f"needs up to ~{_size(total['biggest_raw'])} (zip + extracted, removed after it); "
          f"free under {probe}: {_size(free)}", flush=True)
    return 0


def verify_only(reqs):
    bad = 0
    for req in reqs:
        log = make_log(req)
        if not zc.is_complete(req.store):
            log(f"{req.store}: not downloaded (or not complete)")
            bad += 1
            continue
        rep = verify_request(req, log)
        ds = zc.open_store(req.store)
        dates = zc.dates(ds)
        ds.close()
        have = zc.read_overpass(req.site, req.root)
        miss = [d for d in dates if d not in have]
        log(f"overpass times: {len(dates) - len(miss)} of {len(dates)} dates"
            + (f"; missing {miss[:10]}" if miss else ""))
        if rep.get("verdict") == "differs" or miss:
            bad += 1
    return 1 if bad else 0


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sites", nargs="*", help=f"any of {', '.join(SITES)} (default: all)")
    ap.add_argument("--years", nargs="+", default=None,
                    help="years or ranges: 2023-2025 | 2017-2022 2025 | 2023,2025 "
                         "(default 2023-2025)")
    ap.add_argument("--test", action="store_true",
                    help="format test: 2 km x 2 km in Villaviciosa, 2025-06-01..15, all bands, "
                         "into <root>/_test; prints the FORMAT REPORT")
    ap.add_argument("--login", action="store_true",
                    help="only authenticate with the Copernicus Data Space (interactive, once)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the process graphs, the outputs and a size estimate; no network")
    ap.add_argument("--verify-only", action="store_true",
                    help="compare existing stores with the netCDF cubes again; no network")
    ap.add_argument("--root", default=None,
                    help=f"cubes root (default: ${zc.ENV_ROOT} or <repo>/cubes)")
    ap.add_argument("--allow-grid-offset", action="store_true",
                    help="write a store whose grid is off the existing cube's 10 m lattice or in "
                         "another CRS (default: stop with an error)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep <year>.part (the backend's files) after a store is complete")
    ap.add_argument("--retries", type=int, default=1,
                    help="new jobs tried after a failed one, per year and run (default 1)")
    ap.add_argument("--max-wait-hours", type=float, default=24.0,
                    help="give up waiting for one job after this long (it keeps running)")
    ap.add_argument("--job-options", default=None,
                    help=f"JSON job options (default {json.dumps(JOB_OPTIONS)})")
    ap.add_argument("--backend-format", default=BACKEND_FORMAT, choices=("Zarr", "netCDF"),
                    help="what the backend writes (netCDF: fallback; the store is the same)")
    ap.add_argument("--format-options", default=None,
                    help="JSON save_result options for the format (the --test report lists "
                         "the ones the backend declares for Zarr)")
    a = ap.parse_args(argv)
    a.sites = [s.lower() for s in a.sites] or list(SITES)
    unknown = sorted(set(a.sites) - set(SITES))
    if unknown:
        ap.error(f"unknown site(s) {unknown}; choose from {list(SITES)}")
    try:
        a.years = parse_years(a.years) if a.years else list(DEFAULT_YEARS)
    except ValueError as exc:
        ap.error(str(exc))
    for opt in ("job_options", "format_options"):
        v = getattr(a, opt)
        if v:
            try:
                json.loads(v)
            except ValueError as exc:
                ap.error(f"--{opt.replace('_', '-')} is not JSON: {exc}")
    if sum([a.test, a.verify_only, a.login]) > 1:
        ap.error("--test, --verify-only and --login exclude each other")
    return a


def main(argv=None):
    a = parse_args(argv)
    os.chdir(ROOT)
    if a.login:
        import pyintertidal as pit

        pit.scenes.connect(interactive=True)
        print("authenticated; the refresh token is cached for unattended downloads", flush=True)
        return 0
    root = zc.cubes_root(a.root)
    ctx = Context(a)
    reqs = [test_request(root)] if a.test else year_requests(a.sites, a.years, root)
    if a.dry_run:
        return dry_run(reqs, ctx)
    if a.verify_only:
        return verify_only(reqs)
    ensure_gitignore(root)
    results, failed = {}, {}
    by_site = {}
    for r in reqs:
        by_site.setdefault((r.site, str(r.root)), []).append(r)
    try:
        for (site, _), site_reqs in by_site.items():
            log0 = make_log(site_reqs[0])
            try:
                lock = SiteLock(zc.site_dir(site, site_reqs[0].root))
                lock.__enter__()
            except LockedError as exc:
                log0(str(exc))
                for r in site_reqs:
                    failed[r.tag] = str(exc)
                continue
            try:
                for req in site_reqs:
                    log = make_log(req)
                    try:
                        results[req.tag] = run_request(ctx, req, log)
                    except Exception as exc:          # one year never stops the others
                        failed[req.tag] = f"{type(exc).__name__}: {exc}"
                        log(f"FAILED: {type(exc).__name__}: {exc}")
                        for ln in traceback.format_exc().splitlines()[-12:]:
                            log(f"  {ln}")
                        log(f"run the same command again to resume; finished years are skipped, "
                            f"the raw download in {req.part} is reused")
            finally:
                lock.__exit__(None, None, None)
    except KeyboardInterrupt:
        print("\ninterrupted. A job that was running keeps running on the backend; run the same "
              "command again to reattach to it (its id is in <year>.part/job.json).", flush=True)
        return 130
    print("\nSUMMARY", flush=True)
    for tag, r in results.items():
        print(f"  {tag}: {r}", flush=True)
    for tag, e in failed.items():
        print(f"  {tag}: FAILED {e}", flush=True)
    missing = {t: r["missing_overpass"] for t, r in results.items() if r.get("missing_overpass")}
    if missing:
        print(f"  overpass times missing for {missing}: those scenes would be dropped", flush=True)
    return 1 if failed or missing else 0


if __name__ == "__main__":
    sys.exit(main())
