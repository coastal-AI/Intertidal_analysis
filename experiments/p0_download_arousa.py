"""Cube of the Ría de Arousa (Rías Baixas) — the open-ria cotidal site.

The widest of the Rías Baixas: Bijagós-like open geometry with large flats
at Carril-Rianxo and Cambados. Downloaded at 20 m (like the Scheldt cube)
because the full ría spans a 35 x 36 km bbox
(outer islands at the mouth to the tidal limit of the Ulla at Pontecesures). Target of the flagship
cotidal-lines figure and a fresh external site for the interior-tide
estimator.

Run:  python -m experiments.p0_download_arousa     (openEO job, hours)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pyintertidal.net import use_system_certificates
from pyintertidal.aoi import AOI
from pyintertidal.cube import SentinelCube
from pyintertidal.scenes import connect


def main():
    use_system_certificates()
    arousa = AOI.from_bbox(west=-9.06, south=42.44, east=-8.64, north=42.76,
                           name="arousa")
    cube = SentinelCube(arousa, ("2023-01-01", "2025-12-31"), water="ndwi",
                        resolution=20,
                        cache_path="ndwi_cube_arousa_2023-2025_20m.nc")
    cube.ensure(connect())
    print(f"cached -> {cube.cache_path}: {len(cube.dates)} dates, "
          f"shape {cube.shape}")


if __name__ == "__main__":
    main()
