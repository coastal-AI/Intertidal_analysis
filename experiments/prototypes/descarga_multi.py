"""Cubes for the three sites where method B can be scored in metres.

Method B needs three things at once: large intertidal flats (its calibration
data), an imagery archive over them, and a tide gauge to grade the level it
implies. Among every gauge-pair site used so far only the Scheldt has all
three. These add the strongest remaining candidates on Earth:

  cuxhaven    the German Wadden — the largest flats in Europe, gauge at the
              Elbe mouth in the middle of them
  stmalo      the bay of Mont-Saint-Michel — 12 m of tidal range, kilometres
              of flats; the extreme-range stress test
  sheerness   the outer Thames — Maplin and Foulness sands, gauge at the
              estuary mouth

20 m resolution and tight boxes, same economy as the Scheldt: B consumes
band-level wet fractions and per-pixel NDWI series of the flats, not a 10 m
elevation product.
"""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="backslashreplace")
    except Exception:
        pass

from pyintertidal.net import use_system_certificates
use_system_certificates()

import pyintertidal as pit
from pyintertidal.cube import SentinelCube

SITES = {
    "cuxhaven": {"west": 8.35, "south": 53.83, "east": 8.75, "north": 54.00},
    "stmalo": {"west": -1.95, "south": 48.58, "east": -1.45, "north": 48.68},
    "sheerness": {"west": 0.60, "south": 51.42, "east": 1.00, "north": 51.58},
}
TIME_EXTENT = ("2023-01-01", "2025-12-31")


def main():
    site = os.environ["SITE"]
    aoi = {**SITES[site], "crs": "EPSG:4326"}
    cache = f"ndwi_cube_{site}_2023-2025_20m.nc"
    t0 = time.time()
    cube = SentinelCube(aoi, TIME_EXTENT, water="ndwi",
                        cache_path=cache, resolution=20)
    print(f"[{site}] bandas {cube.bands} · bbox {SITES[site]}", flush=True)
    conn = pit.scenes.connect()
    cube.ensure(conn, verbose=True)
    print(f"[{site}] job {cube.last_job_id} · "
          f"{(time.time()-t0)/60:.1f} min · escenas {len(cube.dates)} · "
          f"rejilla {cube.shape} · "
          f"{os.path.getsize(cache)/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
