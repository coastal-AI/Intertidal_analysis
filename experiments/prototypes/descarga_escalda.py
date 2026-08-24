"""Download the Westerschelde flats: the site where the imagery-lag adapter
can finally be SCORED like everything else.

The adapter derives the tidal lag from the archive alone — flooded-area ranks
against the mouth tide — so it deploys where no gauge exists. Its problem so
far was the opposite: at Villaviciosa there is no gauge, so it could be
built but not scored in metres.

The Scheldt closes the loop. Between Vlissingen and Terneuzen lie some of the
largest intertidal flats in Europe, both gauges are already cached from the
IOC service, and the gauge-measured lag between them is +22.8 min for M2.
The protocol therefore is:

  1. build the adapter FROM THE IMAGERY ONLY, exactly as at Villaviciosa —
     the inner gauge calibrates nothing;
  2. predict the level at Terneuzen as EOT20 shifted by the imagery-derived
     lag;
  3. score against the Terneuzen record, on the same test window as every
     other row of the comparison table (EOT20 solo 0.414, GOT 0.525,
     operador con mareógrafo 0.418).

Two deliberate economies in the download:

  * 20 m resolution, not 10. The lag method consumes wet FRACTIONS per scene
    per along-channel band — a per-pixel elevation product is not the goal —
    and 20 m quarters the size (about 2 GB instead of 8).
  * the AOI spans the full Vlissingen-Terneuzen axis but clips tightly in
    latitude to the flats themselves.
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

CACHE = "ndwi_cube_escalda_2023-2025_20m.nc"
AOI = {"west": 3.55, "south": 51.33, "east": 3.85, "north": 51.46,
       "crs": "EPSG:4326"}
TIME_EXTENT = ("2023-01-01", "2025-12-31")


def main():
    t0 = time.time()
    cube = SentinelCube(AOI, TIME_EXTENT, water="ndwi",
                        cache_path=CACHE, resolution=20)
    print(f"bandas: {cube.bands} · resolucion 20 m", flush=True)
    print(f"AOI Escalda {AOI['west']}-{AOI['east']}E, "
          f"{AOI['south']}-{AOI['north']}N", flush=True)
    conn = pit.scenes.connect()
    cube.ensure(conn, verbose=True)
    print(f"job openEO: {cube.last_job_id}", flush=True)
    print(f"descargado en {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"escenas {len(cube.dates)} · rejilla {cube.shape}", flush=True)
    print(f"tamano {os.path.getsize(CACHE)/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
