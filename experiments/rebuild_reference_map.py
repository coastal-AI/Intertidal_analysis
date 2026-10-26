# -*- coding: utf-8 -*-
"""Recompute the Villaviciosa reference map (transition / stable water /
stable land) exactly as the topography notebook does, and write it to
products_villaviciosa/reference_map.tif.

The notebook computes the map in memory and never writes it; the file of
that name on disk dated from an August prototype and had no stable-water
class. Same call, same parameters as the notebook's section 3 cell.

Run from the repository root:  python -m experiments.rebuild_reference_map
"""
from __future__ import annotations

import os
import shutil

import numpy as np

CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"
OUT = "products_villaviciosa/reference_map.tif"
# the notebook's parameters (cells 2, 17 and 19)
TIME_EXTENT = ("2016-01-01", "2025-12-31")
NDWI_THRESHOLD = 0.0
CLEAN_SCENE_MAX_BAD = 0.05
STABLE_THRESHOLD = 0.95
TRANSITION_BUFFER_PX = 10


def main():
    import pyintertidal as pit
    from pyintertidal import SentinelCube
    aoi = pit.sites.get("villaviciosa")
    cube = SentinelCube(aoi, TIME_EXTENT, water="ndwi", cache_path=CUBE, resolution=10)
    cube.ensure(None)                                   # cached: no connection needed
    transform, crs = cube.grid
    reference_map, cloud_pct = pit.reference_and_clouds(
        cube, threshold=NDWI_THRESHOLD, bad_classes=pit.water.BAD_CLASSES,
        clean_scene_max_bad=CLEAN_SCENE_MAX_BAD, stable_threshold=STABLE_THRESHOLD,
        transition_buffer_px=TRANSITION_BUFFER_PX)
    px_km2 = abs(transform.a * transform.e) / 1e6
    for code, name in [(0, "transition"), (1, "stable water"), (2, "stable land")]:
        n = int((reference_map == code).sum())
        print(f"  class {code} ({name:13s}): {n:>8,} px  {n * px_km2:6.2f} km²")
    if os.path.exists(OUT):
        shutil.copy(OUT, OUT.replace(".tif", "_prototype_2026-08-12.tif"))
    pit.write_geotiff(OUT, reference_map.astype("uint8"), transform, crs, "uint8", None)
    import json
    json.dump({str(k): float(v) for k, v in dict(cloud_pct).items()},
              open("products_villaviciosa/cloud_pct_per_scene.json", "w"), indent=0)
    print("->", OUT)


if __name__ == "__main__":
    main()
