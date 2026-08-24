"""SCL versus NDWI as water detectors over the Santander flats.

Both run on the SAME scenes and the SAME clear-sky pixels, so any difference
is the detector and nothing else. SCL calls water by classification; NDWI
calls it by reflectance. They fail differently, and the intertidal zone —
wet sediment that is not covered water — is exactly where the disagreement
should concentrate.
"""
import os, sys, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import xarray as xr

import pyintertidal as pit
from pyintertidal.cube import open_cube, grid_from_dataset

CACHE = "ndwi_cube_santander_2023_2025.nc"
OUT = "products_santander"
os.makedirs(OUT, exist_ok=True)

NDWI_THRESHOLD = 0.0
BAD = (3, 8, 9, 10)                 # cloud, shadow, cirrus
WATER_SCL = 6                       # SCL's own water class
MARSH_SCL = 12                      # flooded vegetation
ROW = 64

ds, arrays, t_dim = open_cube(CACHE, ("B03", "B08", "SCL"))
b03, b08, scl = arrays["B03"], arrays["B08"], arrays["SCL"]
T = scl.sizes[t_dim]
H = scl.sizes[scl.dims[1]]
W = scl.sizes[scl.dims[2]]
print(f"{T} escenas · {H}x{W}", flush=True)

n_clear = np.zeros((H, W), np.int32)
n_scl = np.zeros((H, W), np.int32)      # water by SCL
n_ndwi = np.zeros((H, W), np.int32)     # water by NDWI
n_marsh = np.zeros((H, W), np.int32)    # SCL flooded vegetation
cls_disagree = np.zeros((13, H, W), np.int32)   # SCL class where they differ

t0 = time.time()
for y0 in range(0, H, ROW):
    y1 = min(y0 + ROW, H)
    sl = {scl.dims[1]: slice(y0, y1)}
    g = np.asarray(b03.isel(sl).values, np.float32)
    n = np.asarray(b08.isel(sl).values, np.float32)
    s = np.nan_to_num(np.asarray(scl.isel(sl).values), nan=0.0).astype(np.int8)

    den = g + n
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(den != 0, (g - n) / den, np.nan).astype(np.float32)

    clear = (~np.isin(s, BAD)) & np.isfinite(ndwi) & (s > 0)
    w_scl = clear & (s == WATER_SCL)
    w_ndwi = clear & (ndwi > NDWI_THRESHOLD)

    n_clear[y0:y1] = clear.sum(0)
    n_scl[y0:y1] = w_scl.sum(0)
    n_ndwi[y0:y1] = w_ndwi.sum(0)
    n_marsh[y0:y1] = (clear & (s == MARSH_SCL)).sum(0)

    diff = clear & (w_scl != w_ndwi)
    for c in range(13):
        cls_disagree[c, y0:y1] = (diff & (s == c)).sum(0)
    print(f"  filas {y0}-{y1}", end="\r", flush=True)

transform, crs = grid_from_dataset(ds)
ds.close()
print(f"\nrecorrido en {(time.time()-t0)/60:.1f} min", flush=True)

MIN_OBS = 20
ok = n_clear >= MIN_OBS
wf_scl = np.where(ok, n_scl / np.maximum(n_clear, 1), np.nan).astype("float32")
wf_ndwi = np.where(ok, n_ndwi / np.maximum(n_clear, 1), np.nan).astype("float32")
marsh = np.where(ok, n_marsh / np.maximum(n_clear, 1), np.nan).astype("float32")

for nm, a in [("wf_scl", wf_scl), ("wf_ndwi", wf_ndwi), ("marsh_scl", marsh)]:
    pit.write_geotiff(f"{OUT}/{nm}.tif", a, transform, crs, "float32", np.nan)
np.save(f"{OUT}/cls_disagree.npy", cls_disagree)
np.save(f"{OUT}/n_clear.npy", n_clear)

px = abs(transform.a * transform.e) / 1e6
d = wf_ndwi - wf_scl
m = np.isfinite(d)
print(f"\npixeles con >= {MIN_OBS} observaciones limpias: {ok.sum():,} "
      f"({100*ok.mean():.0f} %)")
print(f"diferencia de frecuencia de agua (NDWI - SCL):")
print(f"  mediana {np.nanmedian(d):+.3f}, media {np.nanmean(d):+.3f}")
for lo, hi, lab in [(-2, -0.2, "SCL ve MUCHA mas agua"),
                    (-0.2, -0.05, "SCL ve algo mas"),
                    (-0.05, 0.05, "coinciden"),
                    (0.05, 0.2, "NDWI ve algo mas"),
                    (0.2, 2, "NDWI ve MUCHA mas agua")]:
    n_ = int(np.sum((d > lo) & (d <= hi)))
    print(f"  {lab:26s} {n_:8,d} px  {n_*px:6.2f} km2  "
          f"({100*n_/max(m.sum(),1):5.1f} %)")

# Which SCL classes carry the disagreement?
tot = cls_disagree.sum()
print(f"\nCLASE SCL EN LOS DESACUERDOS (total {tot:,} observaciones):")
for c in np.argsort(cls_disagree.sum(axis=(1, 2)))[::-1][:6]:
    n_ = int(cls_disagree[c].sum())
    if n_:
        print(f"  {c:2d} {pit.water.SCL_CLASSES[c][0]:32s} {n_:10,d}  "
              f"{100*n_/tot:5.1f} %")
