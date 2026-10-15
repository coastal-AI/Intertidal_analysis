# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch cmems``: regional hydrodynamic SSH for LM (§1.6, L16).

Question
--------
What sea surface height (tide plus surge) does the operational regional
model of each site give at the record instants: CMEMS NWS AMM15 (1.5 km)
at Escalda, Wadden and Ems, CMEMS IBI (1/36 deg) at the rías?

Pre-registered rules (levels.LEVEL_RULES['LM'])
-----------------------------------------------
* Datasets: :data:`DATASETS`, read from the PUBLIC Copernicus Marine
  catalogue on 2026-10-14 (``copernicusmarine.describe``, no credentials).
  The archive depths matter:

  - AMM15 (``NWSHELF_ANALYSISFORECAST_PHY_004_013``) starts 2024-08-04 at
    both 15 min and 1 h: LM at the Dutch sites covers W1 and Y2025 only.
    The NWS reanalysis is AMM7 at 7 km (another model); it is listed with
    ``eligible=False`` and is never substituted without the user's decision.
  - IBI analysis-forecast 15-min starts 2024-10-18; the IBI reanalysis at
    the same 1/36 deg (``IBI_MULTIYEAR_PHY_005_002``) covers 1993-2026-06 as
    HOURLY MEANS. It is eligible at the rías (hourly accepted with cubic
    interpolation, §1.6; mean vs instantaneous declared).

* Selection per (site, window): the eligible dataset of the site's model
  that covers the whole window, preferring 15-min instantaneous, then hourly
  instantaneous, then hourly mean (:func:`choose_dataset`). None covering:
  LM is NOT AVAILABLE for that window, never substituted.
* Request: variable ``zos``; the frame's lon/lat bounding box padded by
  :data:`PAD_KM`; ``coordinates_selection_method="outside"`` (the smallest
  model grid covering the box); one NetCDF per calendar year, padded by
  :data:`PAD_TIME_H` hours on both sides so every instant's 4-point cubic
  stencil lies in one file. Files and a manifest (sha256, size, request,
  toolbox version) are cached under ``results/validation_v2/common/<site>/
  cmems/``; a cached file whose sha256 matches the manifest is not
  requested again.
* Credentials: the toolbox credentials file of the project,
  :data:`CREDENTIALS_FILE` (``[credentials] username=..., password=...``).
  Absent, unreadable or without both fields, or rejected by the server: the
  status is ``credentials missing`` and NO request is made (the toolbox
  would otherwise prompt). Every LM unit is then NOT COMPUTABLE, never zero.
* Hourly MEANS: their time stamp convention is measured, not assumed: on an
  overlap month the hourly box mean of the 15-min instantaneous SSH is
  compared with the mean dataset at shifts {-30, 0, +30} min and the best
  shift is recorded (:func:`measure_stamp_shift`). Until it is measured, an
  hourly-mean LM is NOT COMPUTABLE.
* ``dry_run`` checks the dataset ids, the step and the archive depth against
  :data:`DATASETS` (§1.6: before lock-A).
"""
from __future__ import annotations

import configparser
import hashlib
import json
import os

import numpy as np
import pandas as pd

#: the toolbox credentials file inside the project (repository-relative)
CREDENTIALS_FILE = ".config/copernicusmarine/.copernicusmarine-credentials"
VARIABLE = "zos"
PAD_KM = 3.0
PAD_TIME_H = 2
MANIFEST_NAME = "cmems_manifest.json"
MANIFEST_FORMAT = "validation-v2-cmems-1"
CATALOGUE_CHECKED = "2026-10-14"
#: the v2 common tree (spec §13.1); the CLI chdirs to the repository root and
#: ``VALIDATION_V2_ROOT`` moves it (:func:`common_root`)
COMMON_ROOT = "results/validation_v2/common"
#: the Campaign-1 windows LM (levels design) and L16 (lag design) may use
CAMPAIGN1_WINDOWS = ("W3", "W2", "W1", "Y2023", "Y2024", "Y2025")
#: the overlap month on which an hourly-MEAN dataset's stamp convention is
#: measured against the 15-min instantaneous dataset of the same model
#: (inside both archives: IBI 15-min from 2024-10-18, IBI reanalysis to
#: 2026-06-16)
STAMP_OVERLAP = ("2025-01-01", "2025-02-01")
STAMP_SHIFT_NAME = "stamp_shift.json"
STAMP_SHIFT_FORMAT = "validation-v2-cmems-stamp-1"


def common_root():
    """``$VALIDATION_V2_ROOT/common``, else :data:`COMMON_ROOT`."""
    env = os.environ.get("VALIDATION_V2_ROOT")
    return (os.path.join(env, "common") if env else COMMON_ROOT).replace("\\", "/")


def site_cache_dir(site, root=None, dataset_id=None):
    """``<common>/<site>/cmems[/<dataset_id>]``: one folder per dataset (its
    yearly subsets and manifest)."""
    p = os.path.join(root or common_root(), str(site), "cmems")
    if dataset_id:
        p = os.path.join(p, str(dataset_id))
    return p.replace("\\", "/")


STATUS_OK = "ok"
STATUS_CREDENTIALS = "credentials missing"
STATUS_NOT_AVAILABLE = "not available"

#: the site's regional model (§1.1)
SITE_MODEL = {"escalda": "AMM15", "wadden": "AMM15", "ems": "AMM15",
              "ferrol": "IBI", "villaviciosa": "IBI"}
#: the CMEMS region of each regional model (AMM7 is the NWS reanalysis model)
MODEL_FAMILY = {"AMM15": "NWS", "AMM7": "NWS", "IBI": "IBI"}
#: sites where the model does not resolve the water body (flagged degenerate)
DEGENERATE = ("villaviciosa",)

#: the public catalogue on 2026-10-14 (time range in UTC; the end of an
#: analysis-forecast dataset moves forward every day)
DATASETS = {
    "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT15M-i": {
        "product": "NWSHELF_ANALYSISFORECAST_PHY_004_013", "version": "202511",
        "family": "NWS", "model": "AMM15", "resolution": "1.5 km", "step_min": 15, "kind": "i",
        "start": "2024-08-04 00:15:00", "end": "2026-10-20 00:00:00", "eligible": True},
    "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT1H-i": {
        "product": "NWSHELF_ANALYSISFORECAST_PHY_004_013", "version": "202511",
        "family": "NWS", "model": "AMM15", "resolution": "1.5 km", "step_min": 60, "kind": "i",
        "start": "2024-08-04 01:00:00", "end": "2026-10-20 00:00:00", "eligible": True},
    "cmems_mod_nws_phy-ssh_my_7km-2D_PT1H-i": {
        "product": "NWSHELF_MULTIYEAR_PHY_004_009", "version": "202112",
        "family": "NWS", "model": "AMM7", "resolution": "7 km", "step_min": 60, "kind": "i",
        "start": "1993-01-01 01:00:00", "end": "2026-08-01 00:00:00", "eligible": False,
        "note": "AMM7 reanalysis: another model at 7 km; needs the user's decision"},
    "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT15M-i": {
        "product": "IBI_ANALYSISFORECAST_PHY_005_001", "version": "202411",
        "family": "IBI", "model": "IBI", "resolution": "1/36 deg", "step_min": 15, "kind": "i",
        "start": "2024-10-18 00:15:00", "end": "2026-10-23 00:00:00", "eligible": True},
    "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m": {
        "product": "IBI_MULTIYEAR_PHY_005_002", "version": "202511",
        "family": "IBI", "model": "IBI", "resolution": "1/36 deg", "step_min": 60, "kind": "m",
        "start": "1993-01-01 00:00:00", "end": "2026-06-16 23:00:00", "eligible": True},
}
_PREFERENCE = {(15, "i"): 0, (60, "i"): 1, (60, "m"): 2}


def log(msg):
    print(msg, flush=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  credentials
# ─────────────────────────────────────────────────────────────────────────────

def check_credentials(path=CREDENTIALS_FILE, toolbox=None, online=False):
    """Whether the toolbox credentials file can be used.

    Parameters
    ----------
    path : str
        The credentials file (default :data:`CREDENTIALS_FILE`).
    toolbox : module, optional
        ``copernicusmarine`` (default) for the online check.
    online : bool
        Also ask the server (``login(check_credentials_valid=True)``).

    Returns
    -------
    dict
        ``{"status": "ok" | "credentials missing", "reason", "path",
        "verified_online"}``.
    """
    out = {"path": str(path).replace("\\", "/"), "verified_online": False}
    if not os.path.isfile(path):
        return {**out, "status": STATUS_CREDENTIALS,
                "reason": f"the credentials file {path} does not exist"}
    cp = configparser.RawConfigParser()
    try:
        cp.read(path, encoding="utf-8")
        user = cp.get("credentials", "username", fallback="").strip()
        pwd = cp.get("credentials", "password", fallback="").strip()
    except (configparser.Error, UnicodeDecodeError) as e:
        return {**out, "status": STATUS_CREDENTIALS,
                "reason": f"the credentials file is unreadable ({type(e).__name__})"}
    if not user or not pwd:
        return {**out, "status": STATUS_CREDENTIALS,
                "reason": "the credentials file lacks a username or a password "
                          "(section [credentials])"}
    if online:
        if toolbox is None:
            import copernicusmarine as toolbox
        try:
            ok = bool(toolbox.login(credentials_file=path, check_credentials_valid=True))
        except Exception as e:                    # noqa: BLE001 - classified below
            kind = classify_error(e)
            return {**out, "status": kind, "reason": f"login check failed: {type(e).__name__}"}
        if not ok:
            return {**out, "status": STATUS_CREDENTIALS,
                    "reason": "the credentials were rejected by Copernicus Marine"}
        out["verified_online"] = True
    return {**out, "status": STATUS_OK, "reason": ""}


def classify_error(exc):
    """The status of a toolbox exception: credential errors are 'credentials
    missing'; anything else (network, dataset, coverage) 'not available'."""
    name = type(exc).__name__
    if name in ("InvalidUsernameOrPassword", "CredentialsCannotBeNone"):
        return STATUS_CREDENTIALS
    return STATUS_NOT_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
#  datasets
# ─────────────────────────────────────────────────────────────────────────────

def covers(dataset_id, start, end):
    """True if the dataset's archive holds every instant of [start, end)."""
    d = DATASETS[dataset_id]
    return (pd.Timestamp(d["start"]) <= pd.Timestamp(start)
            and pd.Timestamp(end) <= pd.Timestamp(d["end"]))


def choose_dataset(site, start, end, include_ineligible=False):
    """The LM dataset of a (site, window) and why.

    Returns
    -------
    (dataset_id or None, reason)
    """
    model = SITE_MODEL.get(site)
    if model is None:
        return None, f"no regional model configured for {site}"
    family = MODEL_FAMILY[model]
    cands = [(k, d) for k, d in DATASETS.items()
             if d["family"] == family and (d["model"] == model or include_ineligible)
             and (d["eligible"] or include_ineligible)]
    ok = [(k, d) for k, d in cands if covers(k, start, end)]
    if not ok:
        spans = ", ".join(f"{k} from {d['start']}" for k, d in cands)
        return None, (f"no eligible {model} dataset covers {start}..{end} ({spans}); "
                      "LM is NOT AVAILABLE for this window (never substituted)")
    ok.sort(key=lambda kd: _PREFERENCE[(kd[1]["step_min"], kd[1]["kind"])])
    return ok[0][0], "covers the window"


def bbox_lonlat(grid, pad_km=PAD_KM):
    """``(lon_min, lon_max, lat_min, lat_max)`` of the frame padded by
    ``pad_km`` (the four corners and edge midpoints, projected)."""
    from pyproj import Transformer

    from experiments.benchmark.fetch.tides import grid_axes

    xs, ys, crs = grid_axes(grid)
    pad = float(pad_km) * 1000.0
    x0, x1 = float(xs.min()) - pad, float(xs.max()) + pad
    y0, y1 = float(ys.min()) - pad, float(ys.max()) + pad
    px = np.array([x0, x1, x0, x1, (x0 + x1) / 2, (x0 + x1) / 2, x0, x1])
    py = np.array([y0, y0, y1, y1, y0, y1, (y0 + y1) / 2, (y0 + y1) / 2])
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(px, py)
    return (float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))


def year_request(dataset_id, bbox, year, out_dir, credentials_file=CREDENTIALS_FILE):
    """The toolbox ``subset`` keywords of one padded calendar year."""
    a = pd.Timestamp(f"{int(year)}-01-01") - pd.Timedelta(hours=PAD_TIME_H)
    b = pd.Timestamp(f"{int(year) + 1}-01-01") + pd.Timedelta(hours=PAD_TIME_H)
    return dict(dataset_id=dataset_id, variables=[VARIABLE],
                minimum_longitude=bbox[0], maximum_longitude=bbox[1],
                minimum_latitude=bbox[2], maximum_latitude=bbox[3],
                start_datetime=a.strftime("%Y-%m-%dT%H:%M:%S"),
                end_datetime=b.strftime("%Y-%m-%dT%H:%M:%S"),
                coordinates_selection_method="outside",
                output_filename=f"{dataset_id}_{int(year)}.nc",
                output_directory=str(out_dir), credentials_file=str(credentials_file),
                overwrite=True, disable_progress_bar=True)


def fetch_site(site, grid, years, dataset_id, out_dir, toolbox=None,
               credentials_file=CREDENTIALS_FILE, online_check=False):
    """Download (once) the yearly SSH subsets of a site and write the manifest.

    Returns
    -------
    dict
        The manifest: ``status`` ("ok", "credentials missing", "not
        available"), ``reason``, ``dataset``, ``bbox``, ``files`` {year:
        {name, sha256, size}}, ``requests``, ``toolbox_version``.
    """
    os.makedirs(out_dir, exist_ok=True)
    man = {"format": MANIFEST_FORMAT, "site": site, "dataset": dataset_id,
           "catalogue": DATASETS.get(dataset_id), "years": [int(y) for y in years],
           "files": {}, "requests": {}}
    cred = check_credentials(credentials_file, toolbox=toolbox, online=online_check)
    man["credentials"] = {k: cred[k] for k in ("status", "reason", "verified_online")}
    if dataset_id not in DATASETS:
        return _finish(out_dir, {**man, "status": STATUS_NOT_AVAILABLE,
                                 "reason": f"unknown dataset {dataset_id}"})
    if cred["status"] != STATUS_OK:
        log(f"[fetch cmems] {site}: {cred['status']}: {cred['reason']}; no request made")
        return _finish(out_dir, {**man, "status": cred["status"], "reason": cred["reason"]})
    if toolbox is None:
        import copernicusmarine as toolbox
    man["toolbox_version"] = str(getattr(toolbox, "__version__", "unknown"))
    bbox = bbox_lonlat(grid)
    man["bbox"] = list(bbox)
    old = read_manifest(out_dir) or {}
    for y in years:
        req = year_request(dataset_id, bbox, y, out_dir, credentials_file)
        path = os.path.join(out_dir, req["output_filename"])
        prev = (old.get("files") or {}).get(str(int(y)))
        man["requests"][str(int(y))] = {k: v for k, v in req.items()
                                        if k not in ("credentials_file", "output_directory")}
        if prev and os.path.isfile(path) and _sha256(path) == prev["sha256"] \
                and old.get("dataset") == dataset_id:
            man["files"][str(int(y))] = prev
            continue
        try:
            toolbox.subset(**req)
        except Exception as e:                    # noqa: BLE001 - classified below
            status = classify_error(e)
            log(f"[fetch cmems] {site} {y}: {status}: {type(e).__name__}: {e}")
            return _finish(out_dir, {**man, "status": status,
                                     "reason": f"{type(e).__name__}: {str(e)[:300]}"})
        if not os.path.isfile(path):
            return _finish(out_dir, {**man, "status": STATUS_NOT_AVAILABLE,
                                     "reason": f"the toolbox wrote no file {path}"})
        man["files"][str(int(y))] = {"name": os.path.basename(path), "sha256": _sha256(path),
                                     "size": int(os.path.getsize(path))}
        log(f"[fetch cmems] {site} {y}: {os.path.basename(path)} "
            f"({man['files'][str(int(y))]['size'] / 1e6:.1f} MB)")
    return _finish(out_dir, {**man, "status": STATUS_OK, "reason": ""})


def _finish(out_dir, man):
    path = os.path.join(out_dir, MANIFEST_NAME)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)
    return man


def read_manifest(out_dir):
    path = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_files(out_dir, man=None):
    """The manifest's files in year order, each verified against its sha256."""
    man = man or read_manifest(out_dir)
    if not man or man.get("status") != STATUS_OK:
        raise RuntimeError(f"no usable CMEMS manifest in {out_dir}")
    out = []
    for y in sorted(man["files"], key=int):
        e = man["files"][y]
        p = os.path.join(out_dir, e["name"])
        if _sha256(p) != e["sha256"]:
            raise RuntimeError(f"{p} changed since it was fetched (sha256)")
        out.append(p)
    return out


def read_subset(path, variable=VARIABLE):
    """One subset file as plain arrays.

    Returns
    -------
    dict
        ``t`` (DatetimeIndex, naive UTC), ``lon`` (nx,), ``lat`` (ny,),
        ``zos`` (nt, ny, nx) float64 with NaN on land.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        da = ds[variable]
        for d in list(da.dims):
            if d not in ("time", "latitude", "longitude") and da.sizes[d] == 1:
                da = da.isel({d: 0})
        da = da.transpose("time", "latitude", "longitude")
        t = pd.DatetimeIndex(da["time"].values)
        if t.tz is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return {"t": t.as_unit("ns"), "lon": np.asarray(da["longitude"].values, np.float64),
                "lat": np.asarray(da["latitude"].values, np.float64),
                "zos": np.asarray(da.values, np.float64)}


def measure_stamp_shift(mean_t, mean_v, inst_t, inst_v, shifts=(-30, 0, 30)):
    """The time-stamp convention of an hourly-mean series, measured.

    The 15-min instantaneous series is box-averaged over the hour centred
    on each candidate instant ``stamp + shift`` (5 samples, the end samples
    weighted 1/2: the trapezoid mean over 60 min); the shift with the lowest
    RMS difference to the mean series is returned.

    Returns
    -------
    dict
        ``{"shift_min", "rms_m": {shift: rms}, "n"}``.
    """
    s = pd.Series(np.asarray(inst_v, float), index=pd.DatetimeIndex(inst_t))
    mt = pd.DatetimeIndex(mean_t)
    mv = np.asarray(mean_v, float)
    w = np.array([0.5, 1, 1, 1, 0.5]) / 4.0
    rms = {}
    for sh in shifts:
        c = mt + pd.Timedelta(minutes=sh)
        acc = np.zeros(len(c))
        for k, wk in zip(range(-2, 3), w):
            acc += wk * s.reindex(c + pd.Timedelta(minutes=15 * k)).to_numpy(float)
        ok = np.isfinite(acc) & np.isfinite(mv)
        rms[int(sh)] = float(np.sqrt(np.mean((acc[ok] - mv[ok]) ** 2))) if ok.any() \
            else float("nan")
    best = min((k for k in rms if np.isfinite(rms[k])), key=lambda k: rms[k], default=None)
    return {"shift_min": best, "rms_m": rms,
            "n": int(min(len(mt), len(s)))}


# ─────────────────────────────────────────────────────────────────────────────
#  the campaign: datasets per (site, window), fetch, stamp convention
# ─────────────────────────────────────────────────────────────────────────────

def _window(wid):
    from experiments.benchmark import settings
    return settings.window(wid)


def plan_site(site, windows=CAMPAIGN1_WINDOWS):
    """The LM / L16 dataset of each window of a site and what to fetch.

    Returns
    -------
    dict
        ``windows``: ``{wid: {"dataset" (or None: NOT AVAILABLE), "reason"}}``;
        ``fetch``: ``{dataset: [calendar years]}`` (the years of the windows
        it serves, plus the year of :data:`STAMP_OVERLAP` for the 15-min
        dataset an hourly-MEAN dataset is measured against); ``stamp``:
        ``{hourly-mean dataset: 15-min reference dataset or None}``.
    """
    out = {"site": site, "windows": {}, "fetch": {}, "stamp": {}}
    for wid in windows:
        w = _window(wid)
        dsid, why = choose_dataset(site, w.start, w.end)
        out["windows"][wid] = {"dataset": dsid, "reason": why}
        if dsid:
            out["fetch"].setdefault(dsid, set()).update(int(y) for y in w.years)
    for dsid in list(out["fetch"]):
        if DATASETS[dsid]["kind"] != "m":
            continue
        ref, _ = choose_dataset(site, *STAMP_OVERLAP)
        if ref is not None and DATASETS[ref]["kind"] == "m":
            ref = None
        out["stamp"][dsid] = ref
        if ref is not None:
            out["fetch"].setdefault(ref, set()).add(pd.Timestamp(STAMP_OVERLAP[0]).year)
            out["fetch"].setdefault(dsid, set()).add(pd.Timestamp(STAMP_OVERLAP[0]).year)
    out["fetch"] = {k: sorted(v) for k, v in out["fetch"].items()}
    return out


def _paired_cells(a, b, tol_deg=None):
    """Indices ``(ia, ib)`` of the wet cells of field ``a`` and their nearest
    wet cell of ``b`` within half a cell (both 1/36 deg IBI grids)."""
    from scipy.spatial import cKDTree

    def wet_xy(f):
        LON, LAT = np.meshgrid(f["lon"], f["lat"])
        wet = np.isfinite(f["zos"]).all(axis=0)
        return np.c_[LON[wet], LAT[wet]], np.flatnonzero(wet.ravel())

    pa, ia = wet_xy(a)
    pb, ib = wet_xy(b)
    if not len(pa) or not len(pb):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    step = float(np.median(np.abs(np.diff(a["lon"])))) if len(a["lon"]) > 1 else 1 / 36.0
    tol = 0.5 * step if tol_deg is None else float(tol_deg)
    d, j = cKDTree(pb).query(pa)
    ok = d <= tol
    return ia[ok], ib[j[ok]]


def measure_site_stamp_shift(mean_file, inst_file, month=STAMP_OVERLAP):
    """The stamp convention of an hourly-MEAN subset, measured on the
    overlap month against the 15-min instantaneous subset of the same model
    (:func:`measure_stamp_shift` on the means of the paired wet cells).

    Returns
    -------
    dict
        :func:`measure_stamp_shift`'s result plus ``n_cells`` and ``month``.
    """
    a, b = read_subset(mean_file), read_subset(inst_file)
    ia, ib = _paired_cells(a, b)
    if not len(ia):
        raise ValueError("the hourly-mean and 15-min subsets share no wet cell")
    t0, t1 = pd.Timestamp(month[0]), pd.Timestamp(month[1])
    ma = (a["t"] >= t0) & (a["t"] < t1)
    mb = (b["t"] >= t0 - pd.Timedelta(hours=2)) & (b["t"] < t1 + pd.Timedelta(hours=2))
    if not ma.any() or not mb.any():
        raise ValueError(f"the subsets do not both cover {month[0]}..{month[1]}")
    za = a["zos"].reshape(len(a["t"]), -1)[ma][:, ia]
    zb = b["zos"].reshape(len(b["t"]), -1)[mb][:, ib]
    r = measure_stamp_shift(a["t"][ma], za.mean(axis=1), b["t"][mb], zb.mean(axis=1))
    r.update(n_cells=int(len(ia)), month=list(month))
    return r


def read_stamp_shift(out_dir):
    """The measured stamp shift (min) of a dataset folder, or None (not
    measured, or measured on a mean file the manifest no longer lists)."""
    p = os.path.join(out_dir, STAMP_SHIFT_NAME)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        doc = json.load(f)
    v = doc.get("shift_min")
    fm = (doc.get("files") or {}).get("mean") or {}
    man = read_manifest(out_dir)
    if fm and man is not None:
        cur = {e.get("name"): e.get("sha256") for e in (man.get("files") or {}).values()}
        if cur.get(fm.get("name")) != fm.get("sha256"):
            return None
    return None if v is None else float(v)


def fetch_campaign(site, grid, windows=CAMPAIGN1_WINDOWS, root=None, toolbox=None,
                   credentials_file=CREDENTIALS_FILE, online_check=False):
    """Fetch every dataset a site's Campaign-1 windows need (one folder and
    manifest per dataset, :func:`fetch_site`), then measure the stamp
    convention of each hourly-mean dataset on :data:`STAMP_OVERLAP` and
    write ``stamp_shift.json`` beside it.

    Returns
    -------
    dict
        :func:`plan_site` plus ``datasets``: ``{dataset: {"status", "reason",
        "years", "dir"}}``, ``stamp_shift``: ``{dataset: dict}`` and
        ``status``: "ok" (every needed dataset fetched; windows no dataset
        covers are declared NOT AVAILABLE), "credentials missing" or "not
        available" (a fetch failed).
    """
    plan = plan_site(site, windows)
    plan["datasets"], plan["stamp_shift"] = {}, {}
    for dsid, years in plan["fetch"].items():
        d = site_cache_dir(site, root, dsid)
        man = fetch_site(site, grid, years, dsid, d, toolbox=toolbox,
                         credentials_file=credentials_file, online_check=online_check)
        plan["datasets"][dsid] = {"status": man.get("status"), "reason": man.get("reason"),
                                  "years": years, "dir": d}
    for dsid, ref in plan["stamp"].items():
        d = site_cache_dir(site, root, dsid)
        e = {"dataset": dsid, "reference": ref, "format": STAMP_SHIFT_FORMAT}
        st_m = plan["datasets"].get(dsid, {}).get("status")
        st_r = plan["datasets"].get(ref, {}).get("status") if ref else None
        year = str(pd.Timestamp(STAMP_OVERLAP[0]).year)
        if ref is None:
            e.update(shift_min=None, status="no 15-min dataset of the model covers the "
                                            "overlap month: hourly-mean LM NOT COMPUTABLE")
        elif st_m != STATUS_OK or st_r != STATUS_OK:
            e.update(shift_min=None, status=f"not measured (fetch: {st_m} / {st_r})")
        else:
            try:
                fm = read_manifest(d)["files"][year]["name"]
                rd = site_cache_dir(site, root, ref)
                fr = read_manifest(rd)["files"][year]["name"]
                r = measure_site_stamp_shift(os.path.join(d, fm), os.path.join(rd, fr))
                if r["shift_min"] is None:
                    raise ValueError("no overlapping samples to compare (every candidate "
                                     "shift gave a NaN RMS)")
                e.update(r, status="measured", files={
                    "mean": {"name": fm, "sha256": _sha256(os.path.join(d, fm))},
                    "instantaneous": {"name": fr, "sha256": _sha256(os.path.join(rd, fr))}})
                tmp = os.path.join(d, f"{STAMP_SHIFT_NAME}.tmp-{os.getpid()}")
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(e, f, indent=1, sort_keys=True, default=str)
                os.replace(tmp, os.path.join(d, STAMP_SHIFT_NAME))
            except (KeyError, OSError, ValueError) as exc:
                e.update(shift_min=None, status=f"not measured ({type(exc).__name__}: {exc})")
        old = os.path.join(d, STAMP_SHIFT_NAME)
        if e.get("shift_min") is None and os.path.isfile(old):
            os.remove(old)                     # never an outdated measurement
            log(f"[fetch cmems] {site} {dsid}: the previous {STAMP_SHIFT_NAME} was removed")
        plan["stamp_shift"][dsid] = e
        log(f"[fetch cmems] {site} {dsid}: stamp convention {e['status']}"
            + (f", shift {e['shift_min']} min" if e.get("shift_min") is not None else ""))
    stats = [v["status"] for v in plan["datasets"].values()]
    if any(s == STATUS_CREDENTIALS for s in stats):
        plan["status"] = STATUS_CREDENTIALS
    elif any(s != STATUS_OK for s in stats):
        plan["status"] = STATUS_NOT_AVAILABLE
    else:
        plan["status"] = STATUS_OK
    return plan


def dry_run(toolbox=None, dataset_ids=None):
    """Describe every catalogue dataset (public; no credentials) and compare
    its time range and step with :data:`DATASETS`.

    Returns
    -------
    dict
        ``{dataset_id: {"found", "version", "start", "end", "step_min",
        "matches", "problems"}}``.
    """
    if toolbox is None:
        from pyintertidal import net
        net.use_system_certificates()
        import copernicusmarine as toolbox
    out = {}
    for did in (dataset_ids or DATASETS):
        want = DATASETS[did]
        rec = {"found": False, "problems": []}
        try:
            cat = toolbox.describe(dataset_id=did, disable_progress_bar=True)
        except Exception as e:                    # noqa: BLE001
            rec["problems"].append(f"{type(e).__name__}: {str(e)[:200]}")
            out[did] = rec
            continue
        for p in cat.products:
            for d in p.datasets:
                if d.dataset_id != did:
                    continue
                v = d.versions[-1]
                rec.update(found=True, version=v.label)
                for svc in v.parts[0].services:
                    for var in svc.variables:
                        if var.short_name != VARIABLE:
                            continue
                        for c in var.coordinates:
                            if c.coordinate_id == "time" and c.minimum_value is not None:
                                rec["start"] = str(pd.Timestamp(c.minimum_value, unit="ms"))
                                rec["end"] = str(pd.Timestamp(c.maximum_value, unit="ms"))
                                rec["step_min"] = (float(c.step) / 60000.0 if c.step else None)
        if not rec["found"]:
            rec["problems"].append("dataset id not in the catalogue")
        else:
            if rec.get("step_min") not in (None, float(want["step_min"])):
                rec["problems"].append(f"step {rec['step_min']} min, frozen {want['step_min']}")
            if rec.get("start") and pd.Timestamp(rec["start"]) > pd.Timestamp(want["start"]):
                rec["problems"].append(f"archive now starts {rec['start']}, "
                                       f"frozen {want['start']}")
        rec["matches"] = rec["found"] and not rec["problems"]
        out[did] = rec
    return out
