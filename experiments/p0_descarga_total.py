"""Cube download with the ENLARGED Villaviciosa AOI (user decision,
2026-08-19): rectangle lon -5.47..-5.36, lat 43.46..43.56 — the whole ria
with open-sea margin (mouth seeds) and the complete upper tail, versus the
earlier tight polygon. 2023-2025, NDWI + B04/B11 (dual indices), 10 m.
Fire and forget: openEO queues; collect with the same script (it caches).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from pyintertidal.net import use_system_certificates
use_system_certificates()
from pyintertidal.aoi import AOI
from pyintertidal.cube import SentinelCube

aoi = AOI.from_polygon([(-5.47, 43.46), (-5.36, 43.46),
                        (-5.36, 43.56), (-5.47, 43.56)],
                       name="Villaviciosa total")
c = SentinelCube(aoi, ("2023-01-01", "2025-12-31"), water="ndwi",
         cache_path="ndwi_cube_villaviciosa_total_2023-2025.nc",
         resolution=10, extra_bands=("B04", "B11"))
from pyintertidal.scenes import connect
c.ensure(connection=connect(interactive=False))
print("cube ready:", c.cache_path)
