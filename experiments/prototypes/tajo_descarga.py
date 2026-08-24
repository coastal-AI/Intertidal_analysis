"""Download the Sentinel-2 cube for the Tagus validation cell.

Time extent matched to the 2020 survey: on a tidal flat three years of
difference is real morphological change, not error, so validating a
2023-2025 epoch against a 2020 survey would measure the estuary moving
rather than the method working.
"""
import os, sys, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import pyintertidal as pit
from pyintertidal import SentinelCube

AOI = pit.sites.get("tejo")
TIME_EXTENT = ("2019-01-01", "2021-12-31")     # centred on the 2020 survey
CACHE = "ndwi_cube_tejo_2019_2021.nc"

print(f"AOI: {AOI.name}  {AOI.area_km2:.1f} km²")
print(f"  bbox {AOI.bbox}")
print(f"  periodo {TIME_EXTENT[0]} → {TIME_EXTENT[1]}")

conn = pit.scenes.connect()
print("conectado a Copernicus")

cube = SentinelCube(AOI, TIME_EXTENT, water="ndwi",
                    cache_path=CACHE, resolution=10)
t0 = time.time()
cube.ensure(conn)
print(f"\ncubo listo en {(time.time() - t0) / 60:.1f} min")
print(f"  {len(cube.dates)} escenas · {cube.shape[0]}×{cube.shape[1]} px")
print(f"  {os.path.getsize(CACHE) / 1e9:.2f} GB en disco")
print(f"  primera {cube.dates[0]} · ultima {cube.dates[-1]}")
