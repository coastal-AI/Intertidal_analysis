# -*- coding: utf-8 -*-
"""Download the visible bands (B04, B03, B02) of one Sentinel-2 L2A scene
on the exact grid of the Villaviciosa cube, for the paper's figures.

The cube holds only B03/B08/SCL, so a true-colour view of a scene needs
one small extra request to openEO. Writes
data_v4/villaviciosa_truecolour_<date>.tif (uint16 reflectance x 10000).

Run from the repository root:
    python -m experiments.fetch_true_colour 2024-07-07
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr

CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"
OUT_DIR = "data_v4"


CUBES = {"villaviciosa": "ndwi_cube_villaviciosa_grande_10y.nc",
         "escalda": "ndwi_cube_escalda_2023-2025_20m.nc",
         "wadden": "ndwi_cube_wadden_2023-2025_20m.nc",
         "ems": "ndwi_cube_ems_2023-2025_20m.nc",
         "ferrol": "ndwi_cube_ferrol_2023-2025_10m.nc"}


def grid_bounds(cube_path):
    ds = xr.open_dataset(cube_path)
    x, y = ds["x"].values, ds["y"].values
    wkt = ds["B03"].attrs.get("crs_wkt") or ds["crs"].attrs.get("crs_wkt")
    ds.close()
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    from pyproj import CRS
    epsg = CRS.from_wkt(wkt).to_epsg()
    return {"west": float(x.min()) - dx / 2, "east": float(x.max()) + dx / 2,
            "south": float(y.min()) - dy / 2, "north": float(y.max()) + dy / 2,
            "crs": f"EPSG:{epsg}"}, (len(y), len(x)), dx, epsg


def main(site, date):
    import pyintertidal as pit
    conn = pit.scenes.connect()
    cube_path = CUBES[site]
    extent, shape, res, epsg = grid_bounds(cube_path)
    print("grid", extent, shape, "res", res)
    d1 = str(np.datetime64(date) + np.timedelta64(1, "D"))
    cube = conn.load_collection("SENTINEL2_L2A", spatial_extent=extent, temporal_extent=[date, d1],
                                bands=["B04", "B03", "B02"])
    if res != 10:                                       # the 20-m cubes: average the 10-m bands
        cube = cube.resample_spatial(resolution=res, projection=epsg, method="average")
    cube = cube.reduce_dimension(dimension="t", reducer="mean")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/{site}_truecolour_{date}.tif"
    cube.download(out, format="GTiff")
    import rasterio
    with rasterio.open(out) as s:
        print("->", out, s.count, "bands", s.shape, s.crs, "res", s.res)
        if s.shape != shape:
            print("grid differs from the cube; resampling onto it")
            import pyintertidal as pit_
            ds = xr.open_dataset(cube_path)
            from pyintertidal.cube import grid_from_dataset
            transform, crs = grid_from_dataset(ds); ds.close()
            bands = [pit_.reproject_to_grid(out, transform, crs, shape, band=b + 1) for b in range(3)]
            pit_.write_geotiff(out, np.stack(bands), transform, crs, "float32", np.nan)
            print("-> resampled", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 1:
        main("villaviciosa", a[0])
    else:
        main(a[0], a[1])
