# -*- coding: utf-8 -*-
"""Download the Dutch validation cubes on the validation grid (10 m).

The same cube as cell 8 of the comparison notebooks:
``SentinelCube(aoi, 2023-2025, water="ndwi", resolution=DUTCH_RES_M)``.
Each cube is one openEO batch job on the Copernicus Data Space. The file is
downloaded next to the target and renamed only when complete. An existing
cube is never downloaded again.

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

import pandas as pd  # noqa: E402

import pyintertidal as pit  # noqa: E402
from pyintertidal import SentinelCube  # noqa: E402
from experiments.validation_grid import (DUTCH_PERIOD, DUTCH_RES_M, DUTCH_SITES,  # noqa: E402
                                         dutch_cube, dutch_products)


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
        cube = SentinelCube(aoi, DUTCH_PERIOD, water="ndwi", cache_path=dutch_cube(site),
                            resolution=DUTCH_RES_M)
        if not cube.cached:
            conn = conn or pit.scenes.connect(interactive=False)
            t0 = time.time()
            cube.ensure(conn)
            print(f"[{site}] downloaded in {(time.time() - t0) / 60:.0f} min")
        transform, _ = cube.grid
        H, W = cube.shape
        print(f"[{site}] {dutch_cube(site)}: {len(cube.dates)} scenes, {H} x {W} px "
              f"@ {abs(transform.a):g} m")
        assert abs(abs(transform.a) - DUTCH_RES_M) < 1e-6, "cube grid is not the validation grid"
        missing = check_overpass(site, list(cube.dates))
        if missing:
            bad[site] = missing
    if bad:
        print(f"overpass times missing for {bad}: those scenes would be dropped by the notebooks")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
