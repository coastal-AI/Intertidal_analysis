# -*- coding: utf-8 -*-
"""Download the Dutch validation cubes on the validation grid (10 m).

The same cube as cell 8 of the comparison notebooks:
``SentinelCube(aoi, 2023-2025, water="ndwi", resolution=DUTCH_RES_M)``,
fetched as ONE openEO batch job PER YEAR on the Copernicus Data Space and
merged. A single 3-year job at 10 m failed on the backend (2026-10-01: the
Wadden lost its Spark executors after 68 min), so each year is a third of
the size, asks for more executor memory, and a failed year is the only thing
to repeat: finished years stay cached as part files.

The yearly windows tile the original period exactly, [2023-01-01, 2025-12-31)
with openEO's end-exclusive temporal extent, so the merged cube holds the same
scenes as one 3-year job. The merge streams scene by scene and keeps the
storage of the downloaded cubes (int16, fill -32768, zlib 6, 1 x 256 x 256
chunks); it checks the grids are identical and the dates strictly increasing.

The script also checks that ``products_<site>_10m/overpass_times.json``
covers every scene of the cube. The notebooks read the overpass times from
that file instead of the STAC catalogue, and would silently drop a scene
that is missing from it, so any missing date is fetched and added.

First time on a machine (prints a URL and a code: open it in any browser and
log in; the refresh token is cached and later runs need no browser):

    python -m experiments.download_cube --login

Then, from the repository root (hours; run it inside tmux or nohup):

    python -m experiments.download_cube escalda wadden ems
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pyintertidal as pit  # noqa: E402
from pyintertidal import SentinelCube  # noqa: E402
from experiments.validation_grid import (DUTCH_PERIOD, DUTCH_RES_M, DUTCH_SITES,  # noqa: E402
                                         dutch_cube, dutch_products)

#: yearly windows, end-exclusive like openEO's temporal extent; together they
#: are exactly the 3-year extent of one job, ("2023-01-01", "2025-12-31")
YEARS = (("2023-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"),
         ("2025-01-01", DUTCH_PERIOD[1]))
#: more room per Spark executor than the backend default
JOB_OPTIONS = {"executor-memory": "4G", "executor-memoryOverhead": "4G"}
BANDS = ("B03", "B08", "SCL")


def part_path(site, year_start):
    return dutch_cube(site).replace(".nc", f".part{year_start[:4]}.nc")


def merge_parts(parts, out):
    """Concatenate the yearly cubes along time into ``out`` (streamed)."""
    import xarray as xr

    dss = [xr.open_dataset(p, chunks={}) for p in parts]
    tdim = "t" if "t" in dss[0].dims else "time"
    for p, d in zip(parts[1:], dss[1:]):
        assert np.array_equal(d["x"].values, dss[0]["x"].values), f"{p}: x grid differs"
        assert np.array_equal(d["y"].values, dss[0]["y"].values), f"{p}: y grid differs"
    ds = xr.concat(dss, dim=tdim, data_vars="minimal", coords="minimal", compat="override")
    t = ds[tdim].values
    assert np.all(np.diff(t) > np.timedelta64(0, "ns")), "dates not strictly increasing across years"
    enc = {}
    for v in BANDS:
        e = dss[0][v].encoding
        enc[v] = {k: e[k] for k in ("dtype", "zlib", "complevel", "shuffle", "chunksizes", "_FillValue")
                  if k in e}
    te = dss[0][tdim].encoding
    enc[tdim] = {k: te[k] for k in ("units", "calendar", "dtype") if k in te}
    tmp = out + ".merge"
    ds.chunk({tdim: 1}).to_netcdf(tmp, encoding=enc)
    for d in dss:
        d.close()
    with xr.open_dataset(tmp) as chk:
        assert chk.sizes[tdim] == len(t), "merged cube lost scenes"
    os.replace(tmp, out)
    for p in parts:
        os.remove(p)
    return len(t)


def check_overpass(site, dates):
    """Make the overpass cache cover every cube date; return the dates still missing."""
    out_dir = dutch_products(site)
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/overpass_times.json"
    have = json.load(open(path)) if os.path.exists(path) else {}
    missing = [d for d in dates if d not in have]
    if missing:
        print(f"[{site}] {len(missing)} cube dates not in {path}: asking the catalogue")
        aoi = pit.sites.get(site)
        fresh = pit.overpass.get_overpass_times(aoi.bbox, DUTCH_PERIOD, verbose=False)
        for d in missing:
            if d in fresh:
                have[d] = pd.Timestamp(fresh[d]).to_pydatetime().isoformat()
        json.dump(dict(sorted(have.items())), open(path, "w"))
        missing = [d for d in dates if d not in have]
    print(f"[{site}] overpass times: {len(dates) - len(missing)} of {len(dates)} cube dates"
          + (f"; STILL MISSING {missing}" if missing else ""))
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sites", nargs="*", help=f"any of {', '.join(DUTCH_SITES)} (default: all)")
    ap.add_argument("--login", action="store_true",
                    help="only authenticate with the Copernicus Data Space (interactive, once)")
    a = ap.parse_args(argv)
    a.sites = a.sites or list(DUTCH_SITES)
    unknown = sorted(set(a.sites) - set(DUTCH_SITES))
    if unknown:
        ap.error(f"unknown site(s) {unknown}; choose from {list(DUTCH_SITES)}")
    if a.login:
        pit.scenes.connect(interactive=True)
        print("authenticated; the refresh token is cached for unattended downloads")
        return 0
    pit.net.use_system_certificates()
    conn = None
    bad = {}
    for site in a.sites:
        aoi = pit.sites.get(site)
        final = SentinelCube(aoi, DUTCH_PERIOD, water="ndwi", cache_path=dutch_cube(site),
                             resolution=DUTCH_RES_M)
        if not final.cached:
            parts = []
            for y0, y1 in YEARS:
                part = SentinelCube(aoi, (y0, y1), water="ndwi", cache_path=part_path(site, y0),
                                    resolution=DUTCH_RES_M)
                if not part.cached:
                    conn = conn or pit.scenes.connect(interactive=False)
                    t0 = time.time()
                    part.ensure(conn, job_options=JOB_OPTIONS)
                    print(f"[{site}] {y0[:4]} downloaded in {(time.time() - t0) / 60:.0f} min", flush=True)
                else:
                    print(f"[{site}] {y0[:4]} already downloaded", flush=True)
                parts.append(part.cache_path)
            n = merge_parts(parts, dutch_cube(site))
            print(f"[{site}] merged {len(parts)} years -> {dutch_cube(site)} ({n} scenes)", flush=True)
        transform, _ = final.grid
        H, W = final.shape
        print(f"[{site}] {dutch_cube(site)}: {len(final.dates)} scenes, {H} x {W} px "
              f"@ {abs(transform.a):g} m", flush=True)
        assert abs(abs(transform.a) - DUTCH_RES_M) < 1e-6, "cube grid is not the validation grid"
        missing = check_overpass(site, list(final.dates))
        if missing:
            bad[site] = missing
    if bad:
        print(f"overpass times missing for {bad}: those scenes would be dropped by the notebooks")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
