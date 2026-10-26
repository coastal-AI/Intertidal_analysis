# -*- coding: utf-8 -*-
"""Validation v2 — L16: regional hydrodynamic models at the gauges (spec §3.3, §3.3.1, §1.6).

Question
--------
What lag does the operational regional model of each site (CMEMS NWS
AMM15 at 1.5 km at Escalda, Wadden and Ems; CMEMS IBI at 1/36 deg at the
rías) give at each in-estuary gauge, read out as the gauge itself is read
out by the lag truths? It is the "regional model" comparison of family
L-P1 when it is available at lock-A (spec §6.5).

Pre-registered rules (spec §1.6 LM, §3.3 row L16, §3.3.1 "L14/L15/L16";
levels.LEVEL_RULES['LM']; fetch.cmems)
------------------------------------------------------------------------
* Dataset: ``fetch.cmems.choose_dataset(site, window start, window end)``:
  the eligible dataset of the site's model that covers the WHOLE window,
  15-min instantaneous first, then hourly instantaneous, then hourly
  means. None covering: NOT COMPUTABLE ('not available'), never
  substituted (the planner writes these rows as not_applicable already).
* Status checks: the dataset folder's manifest (``fetch.cmems.read_manifest``
  of ``<common>/<site>/cmems/<dataset>``) must read status "ok" with every
  file intact (sha256, ``manifest_files``); otherwise NOT COMPUTABLE with
  stage 'credentials missing' (the manifest says so, or there is no usable
  manifest and ``fetch.cmems.check_credentials`` fails) or 'not available'.
  A fetched, sha-verified subset is used without the account (the
  credentials status is recorded in diag). An hourly-MEAN dataset needs its
  measured stamp convention (``fetch.cmems.read_stamp_shift``): otherwise
  NOT COMPUTABLE ('stamp shift not measured').
* Per judge: the nearest WET model cell (finite SSH at every step of every
  yearly subset), its great-circle distance logged as ``dist_km``; a cell
  more than 1 km away is flagged "unresolved" (expected at the inner Ems and
  in the rías). The SSH at the instants and on the 10-min grids is the
  4-point cubic interpolation of ``levels.cubic_at`` on the model steps
  shifted by the stamp convention (0 for instantaneous datasets).
* Read-outs (i) against h_B and (ii) model-internal exactly as L14
  (``lags.models.readouts``: (a), tau_ab, (b) M2 phase, HW/LW mean; internal
  (a) and (b)); the boundary point is the equal-weight mean of the nearest
  wet cell of each boundary member. Flag "degenerate" at Villaviciosa
  (``fetch.cmems.DEGENERATE``: the model does not resolve the ría) and
  wherever the gauge cell and a member cell are the same or adjacent.
* The SSH includes surge (tide + atmosphere), like a gauge; E(tau) is
  median-centred, so the model datum does not matter.
* No gauge series is read: only the judges' coordinates.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments.benchmark import config, levels, settings
from experiments.benchmark.lags import common, models
from experiments.benchmark.product import NotComputable

NAN = float("nan")
EARTH_RADIUS_KM = 6371.0
#: a gauge whose nearest wet cell is farther than this is "unresolved"
#: (spec §1.6 LM, FROZEN level: unresolved if the nearest wet cell is > 1 km)
UNRESOLVED_KM = 1.0

CONFIG = {
    "rule": ("L16: the CMEMS dataset covering the whole window (fetch.cmems.choose_dataset), "
             "its sha-verified yearly subsets; per in-frame judge the nearest wet model cell "
             "(distance logged, 'unresolved' beyond 1 km); SSH by 4-point cubic interpolation "
             "(levels.cubic_at) on the stamp-shifted model steps; read-outs (i) against h_B "
             "and (ii) against the mean of the boundary members' nearest wet cells, as L14; "
             "'degenerate' at Villaviciosa and where the gauge and a member cell are the same "
             "or adjacent"),
    "datasets": "fetch.cmems.DATASETS (catalogue checked 2026-10-14)",
    "selection": "fetch.cmems.choose_dataset: 15-min instantaneous, then hourly "
                 "instantaneous, then hourly means; none covering: not available",
    "variable": "zos",
    "wet_cell": "finite zos at every step of every yearly subset",
    "unresolved_km": UNRESOLVED_KM,
    "interpolation": "levels.cubic_at (4-point Lagrange), model stamps + measured shift",
    "hourly_means": "NOT COMPUTABLE until fetch.cmems.read_stamp_shift gives the shift",
    "readouts": "lags.models.readouts (L14's read-outs (i) and (ii))",
    "degenerate_sites": "fetch.cmems.DEGENERATE",
    "gauge_series": "never read (coordinates only)",
}

CARDS = {
    "L16": {
        "id": "L16", "name": "regional hydrodynamic models at the gauges (CMEMS NWS AMM15, "
                             "IBI 1/36 deg)",
        "family": "models", "kind": "lag", "tier": "1", "fidelity": "-",
        "evidence": ("the truth machinery applied to the model's SSH (lags.models.readouts); "
                     "unit tests with a planted lag on a synthetic CMEMS grid"),
        "references": [
            {"cite": "Copernicus Marine Service, NWSHELF_ANALYSISFORECAST_PHY_004_013 (AMM15)",
             "doi_or_url": "https://data.marine.copernicus.eu/product/"
                           "NWSHELF_ANALYSISFORECAST_PHY_004_013"},
            {"cite": "Copernicus Marine Service, IBI_ANALYSISFORECAST_PHY_005_001 and "
                     "IBI_MULTIYEAR_PHY_005_002 (IBI 1/36 deg)",
             "doi_or_url": "https://data.marine.copernicus.eu/product/"
                           "IBI_MULTIYEAR_PHY_005_002"},
            {"cite": "Mason DC, Scott TR, Dance SL (2010) ECSS 87(3):487-496 (hydrodynamic "
                     "model levels for waterline heighting)",
             "doi_or_url": "doi:10.1016/j.ecss.2010.01.015"},
        ],
        "output": "point", "lag_kind": "linear",
        "signature": {"reads": ["window"], "domain": False},
        "scan_window": None, "published_window": None,
        "as_published": "the model's SSH at the nearest wet cell, the model's own stamps",
        "native_cell": None,
        "common_conditions": "the instance's instants and window; no imagery",
        "deviations": [
            {"paper": "instantaneous SSH", "ours": "hourly MEANS accepted at the rías with a "
             "measured stamp shift and cubic interpolation", "why": "spec §1.6 (IBI 15-min "
             "archive starts 2024-10-18)", "rows": "L16 (IBI reanalysis windows)",
             "sensitivity": "none (declared)"},
            {"paper": "LM: 1-km s bins of wet channel cells", "ours": "the nearest wet cell to "
             "the gauge (a point read-out)", "why": "spec §3.3.1: model at the gauge "
             "coordinates", "rows": "L16", "sensitivity": "none"},
        ],
        "qc": {"unresolved_km": UNRESOLVED_KM, "degenerate": "Villaviciosa; same or adjacent "
               "gauge/member cells",
               "control": ("not gated by the positive control (no imagery, spec §6.5 item "
                           "7); no regional model exists for the synthetic site")},
        "n_min": 0, "tuning": [], "control_offset_min": 0.0,
        "cost": {"wh_cell": 0.05, "ram_gb": 2.0},
        "variants": {"internal": "read-out (ii), model-internal (rows 'L16@internal')"},
    },
}

CODE_DEPS = ("experiments/benchmark/lags/models.py", "experiments/benchmark/fetch/cmems.py",
             "experiments/benchmark/levels.py", "experiments/benchmark/truth.py",
             "experiments/benchmark/settings.py", "experiments/benchmark/lags/common.py")


def log(msg):
    print(msg, flush=True)


def _cmems():
    from experiments.benchmark.fetch import cmems
    return cmems


def dataset_dir(inst, dsid):
    """The dataset folder ``<common>/<site>/cmems/<dataset>`` of
    ``fetch.cmems.site_cache_dir``. ``inst.paths['cmems_dir']`` is the CMEMS
    root the package uses (``package._cmems_root``: ``<root>/common``) or,
    when it ends in ``cmems``, the site's own cmems folder; without it, the
    common folder under ``inst.paths['root']``."""
    paths = getattr(inst, "paths", None) or {}
    cd = paths.get("cmems_dir")
    if cd:
        cd = str(cd).replace("\\", "/").rstrip("/")
        if os.path.basename(cd) == "cmems":
            return f"{cd}/{dsid}"
        return _cmems().site_cache_dir(inst.site, cd, dsid)
    root = paths.get("root")
    common_root = os.path.join(root, "common") if root else None
    return _cmems().site_cache_dir(inst.site, common_root, dsid)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km), vectorised (judges.haversine_km's formula)."""
    lat1, lon1 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lon1, float))
    lat2, lon2 = np.radians(np.asarray(lat2, float)), np.radians(np.asarray(lon2, float))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class ModelGrid:
    """The wet cells of a set of yearly subsets on one model grid."""

    def __init__(self, fields):
        if not fields:
            raise ValueError("no subset field")
        f0 = fields[0]
        self.lon = np.asarray(f0["lon"], np.float64)
        self.lat = np.asarray(f0["lat"], np.float64)
        wet = None
        for f in fields:
            if not (np.array_equal(np.asarray(f["lon"], float), self.lon)
                    and np.array_equal(np.asarray(f["lat"], float), self.lat)):
                raise ValueError("the yearly subsets do not share the model grid")
            z = np.asarray(f["zos"], np.float64)
            if z.ndim != 3 or z.shape[1:] != (len(self.lat), len(self.lon)):
                raise ValueError(f"zos must be (time, lat, lon); got {z.shape}")
            w = np.isfinite(z).all(axis=0)
            wet = w if wet is None else (wet & w)
        self.wet = wet
        iy, ix = np.nonzero(wet)
        self.iy, self.ix = iy.astype(np.int64), ix.astype(np.int64)

    def nearest(self, lat, lon):
        """``{"iy", "ix", "dist_km", "lat", "lon"}`` of the nearest wet cell."""
        if not len(self.iy):
            raise ValueError("no wet model cell in the subset")
        d = haversine_km(lat, lon, self.lat[self.iy], self.lon[self.ix])
        j = int(np.argmin(d))
        return {"iy": int(self.iy[j]), "ix": int(self.ix[j]), "dist_km": float(d[j]),
                "lat": float(self.lat[self.iy[j]]), "lon": float(self.lon[self.ix[j]])}


def cell_series(fields, cells, times, shift_min=0.0):
    """SSH of ``cells`` [(iy, ix)] at ``times``: ``levels.cubic_at`` on each
    yearly subset (stamps + ``shift_min``), the first finite value kept
    (the subsets overlap by their 2-h padding). Returns (C, n) float64."""
    times = pd.DatetimeIndex(times).as_unit("ns")
    out = np.full((len(cells), len(times)), np.nan)
    if not cells:
        return out
    iy = np.array([c[0] for c in cells], np.int64)
    ix = np.array([c[1] for c in cells], np.int64)
    shift = pd.Timedelta(minutes=float(shift_min or 0.0))
    for f in fields:
        z = np.asarray(f["zos"], np.float64)[:, iy, ix]
        v = levels.cubic_at(pd.DatetimeIndex(f["t"]).as_unit("ns") + shift, z, times)
        fill = ~np.isfinite(out) & np.isfinite(v)
        out[fill] = v[fill]
    return out


def read_fields(paths):
    """The yearly subsets as ``{t, lon, lat, zos}`` (``fetch.cmems.read_subset``)."""
    return [_cmems().read_subset(p) for p in paths]


def _status_checks(inst, dsid, ddir):
    """The manifest, the stamp shift and the files, or NotComputable."""
    cm = _cmems()
    cat = cm.DATASETS[dsid]
    man = cm.read_manifest(ddir)
    cred = cm.check_credentials()
    diag = {"dataset": dsid, "dir": ddir, "credentials": {k: cred.get(k) for k in
                                                          ("status", "reason")}}
    if man is None or man.get("status") != cm.STATUS_OK:
        st = None if man is None else man.get("status")
        why = (man or {}).get("reason") or f"no CMEMS manifest in {ddir}: run fetch cmems"
        if st == cm.STATUS_CREDENTIALS or (cred.get("status") != cm.STATUS_OK
                                           and st in (None, cm.STATUS_CREDENTIALS)):
            raise NotComputable("credentials missing", cred.get("reason") or why,
                                dict(diag, missing_input="Copernicus Marine credentials",
                                     manifest_status=st))
        raise NotComputable("not available", why, dict(diag, missing_input=f"CMEMS {dsid}",
                                                       manifest_status=st))
    shift = 0.0
    if cat["kind"] == "m":
        shift = cm.read_stamp_shift(ddir)
        if shift is None:
            raise NotComputable("stamp shift not measured",
                                f"{dsid} holds hourly MEANS: its stamp convention is not "
                                "measured (fetch cmems writes stamp_shift.json)",
                                dict(diag, missing_input=f"{ddir}/{cm.STAMP_SHIFT_NAME}"))
    try:
        files = cm.manifest_files(ddir, man)
    except (RuntimeError, OSError) as e:
        raise NotComputable("not available", f"{type(e).__name__}: {e}",
                            dict(diag, missing_input=f"CMEMS {dsid} files")) from None
    return man, float(shift), files, diag


def run_unit(inst, unit, opts=None):
    """Unit L16 (product L16): see the module docstring."""
    cm = _cmems()
    products = models.unit_products(unit, ("L16",))
    wid = str(inst.window)
    w = settings.window(wid)
    judges_in = models.in_frame_judges(inst)
    dsid, why = cm.choose_dataset(inst.site, w.start, w.end)
    base = {"window": wid, "dataset": dsid, "selection": why,
            "n_judges_in_frame": len(judges_in)}
    if not judges_in:
        r = common.lag_result("L16", "empty", kind="point", point={},
                              reason="no in-frame judge", diag=base)
        return [r for _ in products]
    if dsid is None:
        raise NotComputable("not available", why,
                            dict(base, missing_input=f"a CMEMS dataset covering {wid}"))
    ddir = dataset_dir(inst, dsid)
    with models._stage(inst, "inputs"):
        man, shift, files, diag = _status_checks(inst, dsid, ddir)
        fields = read_fields(files)
        grid = ModelGrid(fields)
    cat = cm.DATASETS[dsid]
    label = f"{cat['model']} {cat['resolution']} ({dsid})"
    degenerate_site = inst.site in cm.DEGENERATE
    ctx = models.ReadoutContext(inst)
    members = models._members(inst)
    mcells = []
    for mb in members:
        c = grid.nearest(mb["lat"], mb["lon"])
        c["code"] = mb["code"]
        mcells.append(c)
    with models._stage(inst, "boundary"):
        mb_vals = None
        if mcells:
            S = cell_series(fields, [(c["iy"], c["ix"]) for c in mcells], ctx.times_boundary,
                            shift)
            mb_vals = S.mean(axis=0)
    point = {}
    for rec in judges_in:
        code = str(rec["code"])
        c = grid.nearest(rec["lat"], rec["lon"])
        with models._stage(inst, "gauge"):
            vals = cell_series(fields, [(c["iy"], c["ix"])], ctx.times_gauge, shift)[0]
        flags = []
        if not np.isfinite(vals).any():
            e = models._nan_entry("the subset gives no SSH at the gauge's instants",
                                  flags=["no SSH at the instants"])
        else:
            m_t, m10, m_hw = ctx.gauge_parts(vals)
            mb_lag = mb10 = None
            if mb_vals is not None:
                mb_lag, mb10 = ctx.boundary_parts(mb_vals)
            else:
                flags.append("no boundary point (no boundary member)")
            e = models.readouts(ctx, m_t, m10, m_hw, mb_lag, mb10, ref_key=("regional", dsid))
            e["flags"] = e["flags"] + flags
        if c["dist_km"] > UNRESOLVED_KM:
            e["flags"].append("unresolved")
        adj = [m["code"] for m in mcells
               if models.same_or_adjacent((c["iy"], c["ix"]), (m["iy"], m["ix"]))]
        if degenerate_site or adj:
            e["flags"].append("degenerate")
        e.update(dist_km=c["dist_km"], model=label, models=None,
                 cell={"iy": c["iy"], "ix": c["ix"], "lat": c["lat"], "lon": c["lon"]},
                 adjacent_members=adj)
        point[code] = e
        models._progress(inst)
    diag.update(base, model=cat["model"], resolution=cat["resolution"],
                step_min=cat["step_min"], kind=cat["kind"], stamp_shift_min=shift,
                files=[os.path.basename(p) for p in files],
                n_wet_cells=int(len(grid.iy)), degenerate_site=degenerate_site,
                members={m["code"]: {k: m[k] for k in ("iy", "ix", "dist_km", "lat", "lon")}
                         for m in mcells},
                constituents=list(ctx.cons))
    if not any(np.isfinite(e["a"]) for e in point.values()):
        raise NotComputable("not available", "the subset gives no SSH at any gauge's instants",
                            dict(diag, missing_input=f"CMEMS {dsid} coverage of {wid}"))
    r = common.lag_result("L16", "ok", kind="point", lag_kind="linear", point=point,
                          edge_share=models._edge_share(point), diag=diag)
    return [r for _ in products]


def external_inputs(inst, unit=None):
    """The CMEMS manifest (and the stamp-shift file of an hourly-mean
    dataset) of the unit's window, and the judge table."""
    cm = _cmems()
    out = {"judges.json": models.judges_sha(inst),
           "boundary_members": config.sha256_json(models._members(inst))}
    try:
        w = settings.window(str(inst.window))
        dsid, _ = cm.choose_dataset(inst.site, w.start, w.end)
    except Exception:                           # unknown window: no dataset
        dsid = None
    if dsid is not None:
        d = dataset_dir(inst, dsid)
        out["cmems_manifest"] = models.file_sha(os.path.join(d, cm.MANIFEST_NAME))
        if cm.DATASETS[dsid]["kind"] == "m":
            out["cmems_stamp_shift"] = models.file_sha(os.path.join(d, cm.STAMP_SHIFT_NAME))
    return out
