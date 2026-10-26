# -*- coding: utf-8 -*-
"""Recompute, for one comparison site, the rasters its notebook computes in
memory but never writes: the reference map (transition / stable water /
stable land), the per-scene cloud share, the water frequency and the
intertidal mask. Same calls and parameters as cells 8, 12 and 13 of
tide_boundary_comparison_<site>.ipynb, so the printed class counts must
match the notebook's outputs to the pixel.

Writes products_<site>/reference_map.tif, water_frequency.tif,
intertidal_mask.tif and cloud_pct.json.

Run from the repository root:  python -m experiments.rebuild_site_masks wadden
"""
from __future__ import annotations

import json
import sys

import numpy as np

CUBES = {"escalda": "ndwi_cube_escalda_2023-2025_20m.nc",
         "wadden": "ndwi_cube_wadden_2023-2025_20m.nc",
         "ems": "ndwi_cube_ems_2023-2025_20m.nc",
         "ferrol": "ndwi_cube_ferrol_2023-2025_10m.nc"}
RES = {"escalda": 20, "wadden": 20, "ems": 20, "ferrol": 10}
TIME_EXTENT = ("2023-01-01", "2025-12-31")
# the notebooks' parameters
NDWI_THRESHOLD = 0.0
CLEAN_SCENE_MAX_BAD = 0.05
STABLE_THRESHOLD = 0.95
TRANSITION_BUFFER_PX = 10
CLOUD_THRESHOLD = 0.10
WF_LOW, WF_HIGH = 0.05, 0.95


def main(site):
    import pyintertidal as pit
    from pyintertidal import SentinelCube
    out_dir = f"products_{site}"
    aoi = pit.sites.get(site)
    cube = SentinelCube(aoi, TIME_EXTENT, water="ndwi", cache_path=CUBES[site], resolution=RES[site])
    cube.ensure(None)
    transform, crs = cube.grid
    H, W = cube.shape
    px_km2 = abs(transform.a * transform.e) / 1e6
    print(f"{len(cube.dates)} scenes · grid {H}×{W} @ {abs(transform.a):g} m · {crs}")
    reference_map, cloud_pct = pit.reference_and_clouds(
        cube, threshold=NDWI_THRESHOLD, bad_classes=pit.water.BAD_CLASSES,
        clean_scene_max_bad=CLEAN_SCENE_MAX_BAD, stable_threshold=STABLE_THRESHOLD,
        transition_buffer_px=TRANSITION_BUFFER_PX)
    for code_, name in [(0, "transition"), (1, "stable water"), (2, "stable land")]:
        n = int((reference_map == code_).sum())
        print(f"  class {code_} ({name:13s}): {n:>8,} px  {n * px_km2:6.2f} km²")
    usable_dates = pit.filter_dates(cloud_pct, cloud_threshold=CLOUD_THRESHOLD)
    print(f"{len(usable_dates)} usable dates")
    min_obs = max(8, round(0.05 * len(usable_dates)))
    water_freq = pit.water_frequency(cube, dates=usable_dates, threshold=NDWI_THRESHOLD, min_obs=min_obs)
    transition = reference_map == 0
    otsu_low, otsu_high = pit.multiotsu_window(water_freq, transition)
    polygon_mask = aoi.raster_mask(transform, crs, reference_map.shape)
    intertidal = pit.intertidal_mask(water_freq, reference_map, WF_LOW, WF_HIGH, aoi_mask=polygon_mask)
    print(f"multi-Otsu window [{otsu_low:.2f}, {otsu_high:.2f}]; adopted [{WF_LOW}, {WF_HIGH}] → "
          f"intertidal {int(intertidal.sum()):,} px = {intertidal.sum() * px_km2:.2f} km²")
    pit.write_geotiff(f"{out_dir}/reference_map.tif", reference_map.astype("uint8"), transform, crs, "uint8", None)
    pit.write_geotiff(f"{out_dir}/water_frequency.tif", water_freq.astype("float32"), transform, crs, "float32", np.nan)
    pit.write_geotiff(f"{out_dir}/intertidal_mask.tif", intertidal.astype("uint8"), transform, crs, "uint8", None)
    json.dump({str(k): float(v) for k, v in dict(cloud_pct).items()}, open(f"{out_dir}/cloud_pct.json", "w"), indent=0)
    print(f"-> {out_dir}/reference_map.tif, water_frequency.tif, intertidal_mask.tif, cloud_pct.json")


if __name__ == "__main__":
    main(sys.argv[1])
