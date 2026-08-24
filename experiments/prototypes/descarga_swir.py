"""Re-download Villaviciosa with the red and short-wave infrared bands.

Why the extra bands are worth a download. The relief compression is real and
spatially structured — 85 % of the variance in the per-block slope survives
the sampling noise — but four physical explanations have already failed their
own controls, and what is left is the possibility that the bias is OPTICAL
rather than topographic.

That possibility can only be tested with a second water index. NDWI and MNDWI
see water through different physics — the near infrared for one, the
short-wave infrared for the other — so their noise is largely independent
while the ground beneath them is identical. Fit both, and the DIFFERENCE
between the two elevations is radiometric by construction: topography cancels.
The red band then says whether that difference tracks turbidity, which is the
obvious suspect in an estuary carrying river sediment.

None of this is possible with the cube on disk, which holds only B03, B08 and
SCL.

Deliberate choices:

* 2023-2025, not ten years. That is the epoch every current result is
  validated on, and the ten-year archive was measured to be worse — it
  resolves fewer pixels (31 487 against 33 313) because more of them changed.
* the SAME AOI and the same 10 m grid. A cube on a different grid is how a
  silent index mismatch invalidated this morning's ML numbers, so the script
  refuses to finish quietly if the shape does not match.
* a separate cache file. The existing cube is never touched.
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

SC = os.path.dirname(os.path.abspath(__file__))

CACHE = "swir_cube_villaviciosa_2023-2025.nc"
EXPECTED_SHAPE = (915, 915)
TIME_EXTENT = ("2023-01-01", "2025-12-31")


def main():
    t0 = time.time()
    # NOT sites.get('villaviciosa'): its bbox is smaller than the one the
    # existing cube was actually downloaded with, and using it produced a
    # cube on an 887x673 grid instead of 915x915 — the very mismatch that
    # invalidated this morning's ML numbers. The bbox is therefore read back
    # from the cube on disk, so the two are indexable by the same npz.
    import json as _json
    bb = _json.load(open(os.path.join(SC, "bbox_grande.json")))
    aoi = {"west": bb["west"], "south": bb["south"],
           "east": bb["east"], "north": bb["north"], "crs": "EPSG:4326"}
    cube = SentinelCube(aoi, TIME_EXTENT, water="ndwi",
                        extra_bands=("B04", "B11"),
                        cache_path=CACHE, resolution=10)
    print(f"bandas: {cube.bands}", flush=True)
    print(f"bbox del cubo existente · {TIME_EXTENT[0]} a {TIME_EXTENT[1]}",
          flush=True)

    conn = pit.scenes.connect()
    cube.ensure(conn, verbose=True)
    print(f"job openEO: {cube.last_job_id}", flush=True)

    # ── the grid check that this morning's failure earned ────────────────
    shape = cube.shape
    print(f"\ndescargado en {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"escenas {len(cube.dates)} · rejilla {shape}", flush=True)
    if tuple(shape) != EXPECTED_SHAPE:
        print(f"*** REJILLA DISTINTA: {shape} en vez de {EXPECTED_SHAPE}.")
        print("*** No se puede indexar con el npz existente. Revisar antes")
        print("*** de usar nada de este cubo.")
        sys.exit(2)
    print("rejilla coincide con el cubo actual: indexable con el npz")

    size = os.path.getsize(CACHE) / 1e9
    print(f"tamano {size:.2f} GB")


if __name__ == "__main__":
    main()
