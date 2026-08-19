"""Descarga del cubo con el AOI AMPLIADO de Villaviciosa (decision del usuario,
2026-08-19): rectangulo lon -5.47..-5.36, lat 43.46..43.56 — toda la ria con
margen de mar abierto (semillas de boca) y la cola alta completa, frente al
poligono cenido anterior. 2023-2025, NDWI + B04/B11 (dobles indices), 10 m.
Lanzar y olvidar: openEO encola; recoger con el mismo script (cachea).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from pyintertidal.net import use_system_certificates
use_system_certificates()
from pyintertidal.aoi import AOI
from pyintertidal.cube import Cube

aoi = AOI.from_polygon([(-5.47, 43.46), (-5.36, 43.46),
                        (-5.36, 43.56), (-5.47, 43.56)],
                       name="Villaviciosa total")
c = Cube(aoi, ("2023-01-01", "2025-12-31"), water="ndwi",
         cache_path="ndwi_cube_villaviciosa_total_2023-2025.nc",
         resolution=10, extra_bands=("B04", "B11"))
c.ensure()
print("cubo listo:", c.cache_path)
