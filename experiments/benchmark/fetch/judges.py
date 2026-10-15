# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch judges``: the in-estuary gauge series and their time stamps (§1.7).

Question
--------
Which judge series exist for which calendar years, how is each provider's
value stamped inside its averaging window, and which download commands
would extend the series to 2017-2022, without touching the delivered CSVs?

Pre-registered rules (§1.7)
---------------------------
* The judge CSVs of ``data_v4/gauges/estuary_truth/`` are read as delivered
  (``judges.read_series``); this module never rewrites them.
* Time stamps: each provider's averaging window and stamp position come
  from :data:`judges.TIMESTAMP_CONVENTIONS` (RWS F007 centred 10-min
  means and the PdE centred 5-min bins are documented; WSV and IOC raw
  1-min values are assumed instantaneous until their metadata are checked).
  The written ``judge_conventions.json`` carries, per station, the shift
  that centres its stamps and the +- half-window of the declared timing
  sensitivity. ``judges.read_conventions`` reads it back.
* Coverage: per station and calendar year of the union period, the share
  of the expected samples (sampling_min) present; a year below 50 % is
  "missing" for the extension plan.
* Extension [V5]: the download scripts of ``estuary_truth/scripts`` cover
  2023-2025 only (their periods are hard-coded). The plan lists, per
  station, the missing years, the script and the command, and states
  whether the script can cover them; running is the user's step on the
  server (``execute=True``), and the series are then re-run through the
  scripts' QC (``build_outputs.py``).

Outputs: ``results/validation_v2/common/judge_conventions.json`` and
``results/validation_v2/common/judges_fetch.json``.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

from experiments.benchmark import config, judges, product

OUT_DIR = "results/validation_v2/common"
CONVENTIONS_PATH = f"{OUT_DIR}/judge_conventions.json"
PLAN_PATH = f"{OUT_DIR}/judges_fetch.json"
CONVENTIONS_FORMAT = "validation-v2-judge-conventions-1"
PLAN_FORMAT = "validation-v2-judges-fetch-1"
#: the union period's calendar years (settings.UNION_PERIOD, end exclusive)
UNION_YEARS = tuple(range(2017, 2026))
#: a year with less than this share of its expected samples is missing
MIN_YEAR_SHARE = 0.5
#: the download script of each provider and the period it covers
SCRIPTS = {"RWS": "rws_download.py", "WSV": "wsv_download.py", "IOC": "ioc_stream.py",
           "PdE": "pde_download.py"}
SCRIPT_YEARS = {"RWS": (2023, 2025), "WSV": (2023, 2025), "IOC": (2023, 2025),
                "PdE": (2023, 2025)}


def log(msg):
    print(msg, flush=True)


def script_dir():
    return os.path.join(config.ESTUARY_TRUTH, "scripts").replace("\\", "/")


def script_argument(rec):
    """The station argument a provider's script takes: RWS the provider
    code, WSV the gauge number, IOC and PdE the short code."""
    prov = str(rec.get("provider"))
    code = str(rec.get("provider_code") or rec["code"])
    if prov == "WSV":
        tail = code.rsplit("_", 1)[-1]
        if not tail.isdigit():
            raise ValueError(f"{rec['code']}: no WSV gauge number in {code!r}")
        return tail
    if prov == "PdE":
        return code.split("_", 1)[0]
    return code


def conventions_document(stations, conventions=None):
    """The time-stamp conventions per provider and per station.

    Returns
    -------
    dict
        format, providers (:func:`judges.conventions_doc`), stations {code:
        provider, sampling_min, shift_to_centre_min, half_window_min,
        status}.
    """
    conv = judges.conventions_doc(conventions)
    st = {}
    for rec in stations:
        prov = str(rec.get("provider"))
        if prov not in conv:
            st[rec["code"]] = {"provider": prov, "status": "no convention (refused)"}
            continue
        st[rec["code"]] = {"provider": prov, "sampling_min": rec.get("sampling_min"),
                           "shift_to_centre_min": judges.stamp_shift_min(rec, conv),
                           "half_window_min": judges.half_window_min(rec, conv),
                           "status": conv[prov]["status"]}
    return {"format": CONVENTIONS_FORMAT, "providers": conv, "stations": st}


def coverage_by_year(series, sampling_min, years=UNION_YEARS):
    """Share of the expected samples present per calendar year.

    Returns
    -------
    dict ``{year: share}`` (0 for an absent year).
    """
    step = float(sampling_min)
    out = {}
    yrs = pd.DatetimeIndex(series.index).year
    counts = pd.Series(1, index=yrs).groupby(level=0).sum() if len(series) else pd.Series(
        dtype=np.int64)
    for y in years:
        n_exp = (pd.Timestamp(f"{y + 1}-01-01") - pd.Timestamp(f"{y}-01-01")).total_seconds()
        n_exp = n_exp / 60.0 / step
        out[int(y)] = float(min(1.0, counts.get(y, 0) / n_exp))
    return out


def command(rec, python=None):
    """The download command of one station (argv list)."""
    prov = str(rec.get("provider"))
    if prov not in SCRIPTS:
        raise KeyError(f"no download script for provider {prov!r}")
    return [python or sys.executable, "-X", "utf8", f"{script_dir()}/{SCRIPTS[prov]}",
            script_argument(rec)]


def plan(sites=None, years=UNION_YEARS, stations_path=None, base=None):
    """The fetch plan of the judges: coverage per year, missing years and
    the commands (nothing is downloaded).

    Returns
    -------
    dict
        format, years, stations {code: site, provider, file, coverage
        {year: share}, missing_years, script, script_years, command,
        script_covers_missing}.
    """
    sites = list(sites or config.SITES)
    out = {}
    for site in sites:
        for rec in judges.stations(site, stations_path):
            path = judges.series_path(rec, base)
            if os.path.exists(path):
                cov = coverage_by_year(judges.read_series(rec, base), rec["sampling_min"], years)
            else:
                cov = {int(y): 0.0 for y in years}
            missing = [y for y, s in cov.items() if s < MIN_YEAR_SHARE]
            prov = str(rec.get("provider"))
            sy = SCRIPT_YEARS.get(prov)
            out[rec["code"]] = {
                "site": site, "provider": prov, "file": rec.get("file"),
                "coverage": cov, "missing_years": missing,
                "script": SCRIPTS.get(prov), "script_years": list(sy) if sy else None,
                "command": command(rec) if prov in SCRIPTS else None,
                "script_covers_missing": bool(sy) and all(sy[0] <= y <= sy[1] for y in missing)}
    return {"format": PLAN_FORMAT, "years": [int(y) for y in years], "stations": out}


def run(sites=None, out_dir=OUT_DIR, execute=False, stations_path=None):
    """``fetch judges``: write the conventions and the plan; with
    ``execute`` (the user's step on the server) run the commands whose
    script covers the missing years.

    Returns
    -------
    dict ``{"conventions": path, "plan": path, "ran": [codes]}``
    """
    sites = list(sites or config.SITES)
    stas = [r for s in sites for r in judges.stations(s, stations_path)]
    conv = conventions_document(stas)
    cpath = f"{out_dir}/{os.path.basename(CONVENTIONS_PATH)}"
    ppath = f"{out_dir}/{os.path.basename(PLAN_PATH)}"
    product.write_json(cpath, conv)
    p = plan(sites, stations_path=stations_path)
    ran = []
    if execute:
        for code, e in p["stations"].items():
            if e["missing_years"] and e["script_covers_missing"] and e["command"]:
                log(f"[fetch judges] {code}: {' '.join(e['command'])}")
                subprocess.run(e["command"], check=True)
                ran.append(code)
    p["ran"] = ran
    product.write_json(ppath, p)
    n_miss = sum(bool(e["missing_years"]) for e in p["stations"].values())
    log(f"[fetch judges] {len(p['stations'])} stations, {n_miss} with missing years "
        f"(scripts cover {min(v[0] for v in SCRIPT_YEARS.values())}-"
        f"{max(v[1] for v in SCRIPT_YEARS.values())} only) -> {ppath}")
    return {"conventions": cpath, "plan": ppath, "ran": ran}


def main(argv=None):
    """CLI: ``python -X utf8 -m experiments.benchmark.fetch.judges [--site S]... [--execute]``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--site", action="append", help="site (repeatable; default all)")
    ap.add_argument("--execute", action="store_true", help="run the download commands")
    a = ap.parse_args(argv)
    os.chdir(config.ROOT)
    run(a.site, execute=a.execute)


if __name__ == "__main__":
    main()
