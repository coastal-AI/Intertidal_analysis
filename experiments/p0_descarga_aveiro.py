"""Cubo de la Ria de Aveiro (Portugal) — el candidato a victoria de
batimetria en 'ria corta': laguna somera y confinada con retardos interiores
grandes, y verdad EMODnet LiDAR en disco (590_HR_Lidar_Norte).
bbox: la laguna entera con margen de mar abierto para el ancla de la boca."""
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
print("cubo listo:", c.cache_path)
