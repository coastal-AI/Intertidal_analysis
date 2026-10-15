# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch vak-meta``: where each Vaklodingen value comes from (§1.8) [V25].

Question
--------
For every sheet and survey year of a site, which source files were merged
into the 20 m grid, what survey method do they name (echo sounding, laser
altimetry, or both), and on which day was each surveyed when the delivery
says so?

Pre-registered rules (§1.8, §7.1)
---------------------------------
* Read from the local sheets only: every Vaklodingen netCDF carries an
  ``isource`` variable (the zero-based index of the source file of each
  cell and stamp) whose ``flag_meanings`` attribute lists the source file
  names. No value is changed and nothing is downloaded.
* Survey method from the file-name tokens (:data:`METHOD_TOKENS`): ``lo``
  (lodingen) = sounding; ``la``, ``laser``, ``lidar`` = LiDAR; ``tt`` =
  combined (status "assumed": the token is read as the merged
  sounding-plus-laser product and must be confirmed in the RWS/Deltares
  documentation); ``KBxxx_yyyy_<date>.asc`` sheet files and anything else
  = unknown.
* Survey date: the stamp itself when it is not 1 January (e.g.
  2005-03-02); else a ``_yy_m_d.asc`` tail whose year matches the stamp
  (status "filename (probable)"); else none, and the epoch convention of
  §1.8 (mid-year) applies.
* Transect spacing and data density are not in the sheets; they stay None
  until filled from the documentation through the ``overrides`` file
  (``{source file: {method, transect_spacing_m, density}}``), which is
  recorded with its sha256.
* The source class enters the "Vaklodingen source class" stratum and the
  direct-measurement sensitivity (LiDAR or dense sounding only).

Output: ``results/validation_v2/common/vak_meta/<site>.json``.
"""
from __future__ import annotations

import datetime as _dt
import re

import numpy as np
import pandas as pd

from experiments.benchmark import config, product

OUT_DIR = "results/validation_v2/common/vak_meta"
META_FORMAT = "validation-v2-vak-meta-1"
#: the survey years read (truth.VAK_YEARS)
YEARS = (2016, 2025)
#: method classes and their codes in the stratum arrays
CLASS_CODES = {"unknown": 0, "sounding": 1, "lidar": 2, "combined": 3}
#: file-name tokens -> (method, status)
METHOD_TOKENS = (
    (re.compile(r"(?i)(laser|lidar)"), "lidar", "documented by the name"),
    (re.compile(r"(?i)\d{2,4}la\d*(?=\.|_|$)"), "lidar", "token 'la'"),
    (re.compile(r"(?i)\d{2,4}lo\d*(?=\.|_|-|$)"), "sounding", "token 'lo' (lodingen)"),
    (re.compile(r"(?i)\d{2,4}tt\d*(?=\.|_|-|$)"), "combined",
     "assumed: token 'tt' read as the merged sounding + laser product"),
)
_TAIL_DATE = re.compile(r"_(\d{2})_(\d{1,2})_(\d{1,2})\.asc$", re.IGNORECASE)


def log(msg):
    print(msg, flush=True)


def decimal_year(d):
    """Decimal year of a date (mid-day)."""
    d = pd.Timestamp(d)
    n = 366 if d.is_leap_year else 365
    return float(d.year + (d.dayofyear - 0.5) / n)


def parse_source_name(name, stamp=None):
    """Method and survey date of one source file name.

    Parameters
    ----------
    name : str
        An ``isource`` flag meaning (a source file name).
    stamp : str or Timestamp, optional
        The sheet's time stamp of the survey that used it.

    Returns
    -------
    dict
        file, method, method_status, survey_date (ISO or None),
        date_status ("stamp", "filename (probable)" or None).
    """
    method, mstatus = "unknown", "no method token"
    for pat, m, status in METHOD_TOKENS:
        if pat.search(name):
            method, mstatus = m, status
            break
    date, dstatus = None, None
    if stamp is not None:
        ts = pd.Timestamp(stamp)
        if (ts.month, ts.day) != (1, 1):
            date, dstatus = ts.date().isoformat(), "stamp"
        else:
            m = _TAIL_DATE.search(name)
            if m:
                yy, mo, dd = (int(v) for v in m.groups())
                try:
                    cand = _dt.date(2000 + yy, mo, dd)
                except ValueError:
                    cand = None
                if cand is not None and cand.year == ts.year:
                    date, dstatus = cand.isoformat(), "filename (probable)"
    return {"file": name, "method": method, "method_status": mstatus,
            "survey_date": date, "date_status": dstatus}


def sheet_meta(path, years=YEARS):
    """The sources of one sheet per kept stamp (only ``isource`` and the
    finite mask of ``z`` are read, one stamp at a time).

    Returns
    -------
    dict
        path, sources (all flag meanings), stamps [{date, year, n_cells,
        sources [{index, file, n_cells, method, survey_date, ...}]}].
    """
    import xarray as xr

    out = {"path": str(path).replace("\\", "/"), "stamps": []}
    with xr.open_dataset(path) as ds:
        meanings = (str(ds["isource"].attrs.get("flag_meanings", "")).split()
                    if "isource" in ds else [])
        out["sources"] = meanings
        t = pd.DatetimeIndex(ds["time"].values)
        for i in np.flatnonzero((t.year >= years[0]) & (t.year <= years[1])):
            z = np.asarray(ds["z"].isel(time=int(i)).values)
            fin = np.isfinite(z)
            rec = {"date": t[i].date().isoformat(), "year": int(t[i].year),
                   "n_cells": int(fin.sum()), "sources": []}
            if "isource" in ds:
                s = np.asarray(ds["isource"].isel(time=int(i)).values)[fin]
                s = s[np.isfinite(s)].astype(np.int64)
                u, c = np.unique(s, return_counts=True)
                for k, n in zip(u, c):
                    name = meanings[k] if 0 <= k < len(meanings) else f"<index {k}>"
                    rec["sources"].append(dict(index=int(k), n_cells=int(n),
                                               **parse_source_name(name, t[i])))
            out["stamps"].append(rec)
    return out


def apply_overrides(meta, overrides):
    """Merge user-entered documentation values (``{file: {...}}``) into
    every source entry of the same file."""
    if not overrides:
        return meta
    for sh in meta["sheets"].values():
        for st in sh["stamps"]:
            for src in st["sources"]:
                extra = overrides.get(src["file"])
                if extra:
                    src.update(extra)
                    src["override"] = True
    return meta


def build(site, years=YEARS, out_dir=OUT_DIR, sheets=None, overrides_path=None):
    """``fetch vak-meta`` for one site.

    Parameters
    ----------
    sheets : mapping, optional
        ``{name: path}`` (default ``truth.vaklodingen_sheets(site)``).
    overrides_path : str, optional
        A JSON of documentation values per source file.

    Returns
    -------
    str
        The path of the written metadata.
    """
    from experiments.benchmark import truth

    sheets = truth.vaklodingen_sheets(site) if sheets is None else sheets
    overrides = None
    ov_entry = None
    if overrides_path:
        overrides = product.read_json(overrides_path)
        ov_entry = {"path": str(overrides_path).replace("\\", "/"),
                    "sha256": product.sha256_file(overrides_path)}
    meta = {"format": META_FORMAT, "site": site, "years": list(years),
            "class_codes": dict(CLASS_CODES), "overrides": ov_entry,
            "sheets": {name: sheet_meta(p, years) for name, p in sheets.items()}}
    apply_overrides(meta, overrides)
    counts = {}
    for sh in meta["sheets"].values():
        for st in sh["stamps"]:
            for src in st["sources"]:
                counts[src["method"]] = counts.get(src["method"], 0) + src["n_cells"]
    meta["cells_by_method"] = counts
    meta["sha256"] = config.sha256_json({k: v for k, v in meta.items() if k != "sha256"})
    path = f"{out_dir}/{site}.json"
    product.write_json(path, meta)
    log(f"[fetch vak-meta] {site}: {len(sheets)} sheets, cells by method {counts} -> {path}")
    return path


def src_dates(meta, sheet_order):
    """Per sheet (in ``sheet_order``), ``{source index: decimal year or
    None}``: the input of ``truth.dated_epochs``."""
    out = []
    for name in sheet_order:
        table = {}
        for st in meta["sheets"].get(name, {}).get("stamps", []):
            for src in st["sources"]:
                d = src.get("survey_date")
                table[int(src["index"])] = decimal_year(d) if d else None
        out.append(table)
    return out


def class_tables(meta, sheet_order):
    """Per sheet, ``{source index: class code}`` (:data:`CLASS_CODES`)."""
    out = []
    for name in sheet_order:
        table = {}
        for st in meta["sheets"].get(name, {}).get("stamps", []):
            for src in st["sources"]:
                table[int(src["index"])] = CLASS_CODES.get(src["method"], 0)
        out.append(table)
    return out


def cell_source_class(SRC, sheet, tables):
    """The source class of every cell and survey year (0 unknown, -1 no
    survey) from ``truth.vaklodingen_cells``' SRC and sheet arrays."""
    SRC = np.asarray(SRC, np.int64)
    sheet = np.asarray(sheet, np.int64)
    out = np.where(SRC >= 0, 0, -1).astype(np.int8)
    for k, table in enumerate(tables):
        rows = (sheet == k)[:, None]
        for idx, code in table.items():
            out[(SRC == int(idx)) & rows] = int(code)
    return out
