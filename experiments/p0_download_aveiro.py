"""Cube of the Ria de Aveiro (Portugal) — the 'short ria' bathymetry-win
candidate: a shallow, confined lagoon with large interior lags, and EMODnet
LiDAR truth on disk (590_HR_Lidar_Norte).
bbox: the whole lagoon with open-sea margin for the mouth anchor."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from pyintertidal.net import use_system_certificates
use_system_certificates()
from pyintertidal.aoi import AOI
from pyintertidal.cube import SentinelCube
from pyintertidal.scenes import connect

aoi = AOI.from_polygon([(-8.78, 40.55), (-8.58, 40.55),
                        (-8.58, 40.78), (-8.78, 40.78)],
                       name="Aveiro")
c = SentinelCube(aoi, ("2023-01-01", "2025-12-31"), water="ndwi",
                 cache_path="ndwi_cube_aveiro_2023-2025.nc", resolution=10)
c.ensure(connection=connect(interactive=False))
print("cube ready:", c.cache_path)
