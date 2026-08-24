"""Santander with every band the three detectors need, in one download."""
import os, sys, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import pyintertidal as pit
from pyintertidal import SentinelCube

AOI = pit.sites.get("santander")
TIME_EXTENT = ("2023-01-01", "2025-12-31")
CACHE = "multi_cube_santander_grande.nc"

# "awei" pulls B03, B08, B11, B12 (+SCL), which between them cover NDWI
# (B03/B08), MNDWI (B03/B11) and AWEI. One download, three detectors, and
# above all the SAME scenes for all of them.
print(f"{AOI.name}: {AOI.area_km2:.0f} km2  bandas {pit.water.bands_for('awei')}")
conn = pit.scenes.connect()
cube = SentinelCube(AOI, TIME_EXTENT, water="awei", cache_path=CACHE,
                    resolution=10)
t0 = time.time()
cube.ensure(conn)
print(f"\ncubo listo en {(time.time()-t0)/60:.1f} min")
print(f"  {len(cube.dates)} escenas · {cube.shape[0]}x{cube.shape[1]} px")
print(f"  {os.path.getsize(CACHE)/1e9:.2f} GB")
