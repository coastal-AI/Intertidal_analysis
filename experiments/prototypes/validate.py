"""Step 3: score every DEM against the IGN 5 m LiDAR on the same grid."""
import os, sys, json
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from pyintertidal.validation import reproject_to_grid, compare_dems

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape

mu, TR, CRS, SHAPE = load("products_villaviciosa/hsr_elevation.tif")
lidar = reproject_to_grid("mdt5_villaviciosa_grande.tif", TR, CRS, SHAPE)
print(f"[lidar] {np.isfinite(lidar).sum():,} px validos, "
      f"[{np.nanmin(lidar):.1f}, {np.nanmax(lidar):.1f}] m")

dems = {
    "diccionario (pixels)": load("bathymetry/bathymetry_pixels.tif")[0],
    "isolineas":            load("bathymetry/bathymetry_isolines.tif")[0],
    "optimizado (legacy)":  load("bathymetry/bathymetry.tif")[0],
    "escalon (DEA)":        load("bathymetry/bathymetry_step.tif")[0],
    "HSR 10 anos":          load("bathymetry_hsr/hsr_mu.tif")[0],
    "HSR epoca 2023-2025":  mu,
}
for k, v in dems.items():
    print(f"  {k:24s} {np.isfinite(v).sum():7,d} px  "
          f"[{np.nanmin(v):6.2f},{np.nanmax(v):6.2f}]")

inter = load("products_villaviciosa/intertidal_mask.tif")[0] > 0
print(f"[mascara intertidal] {inter.sum():,} px")

rows = compare_dems(dems, lidar, mask=inter)
print()
hdr = f"{'metodo':24s} {'n':>7s} {'bias':>7s} {'RMSE':>7s} {'MAE':>7s} {'r':>6s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    if "rmse_m" not in r:
        print(f"{r['method']:24s} {r['n']:7,d}  (pocos px)"); continue
    print(f"{r['method']:24s} {r['n']:7,d} {r['bias_m']:7.2f} "
          f"{r['rmse_m']:7.3f} {r['mae_m']:7.3f} {r['pearson_r']:6.2f}")

# common-pixel comparison: only where EVERY method has a value
common = np.isfinite(lidar) & inter
for v in dems.values():
    common &= np.isfinite(v)
print(f"\npixeles comunes a todos los metodos: {common.sum():,}")
rows_common = compare_dems(dems, lidar, mask=common)
print(hdr); print("-" * len(hdr))
for r in rows_common:
    if "rmse_m" not in r:
        print(f"{r['method']:24s} {r['n']:7,d}  (pocos px)"); continue
    print(f"{r['method']:24s} {r['n']:7,d} {r['bias_m']:7.2f} "
          f"{r['rmse_m']:7.3f} {r['mae_m']:7.3f} {r['pearson_r']:6.2f}")

json.dump({"all": rows, "common": rows_common,
           "n_common": int(common.sum())},
          open(os.path.join(SC, "validation.json"), "w"), indent=1)
np.savez_compressed(os.path.join(SC, "lidar.npz"), lidar=lidar,
                    common=common)
print("\nOK -> validation.json, lidar.npz")
