"""SCL vs NDWI vs MNDWI over the whole Bay of Santander.

All three run on the SAME scenes and the SAME clear-sky pixels, so every
difference is the detector. The three fail for different reasons — SCL
classifies categorically at 20 m, MNDWI reads a SWIR band that wet sediment
reflects like water, NDWI uses two native 10 m bands — and the intertidal
zone is where those reasons diverge.
"""
import os, sys, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pyintertidal as pit
from pyintertidal.cube import open_cube, grid_from_dataset

CACHE = "multi_cube_santander_grande.nc"
OUT = "products_santander"
os.makedirs(OUT, exist_ok=True)
BAD = (3, 8, 9, 10)
ROW = 48

ds, arrays, t_dim = open_cube(CACHE, ("B03", "B08", "B11", "SCL"))
b03, b08, b11, scl = (arrays["B03"], arrays["B08"], arrays["B11"],
                      arrays["SCL"])
T = scl.sizes[t_dim]
H = scl.sizes[scl.dims[1]]
W = scl.sizes[scl.dims[2]]
print(f"{T} escenas · {H}x{W}", flush=True)

n_clear = np.zeros((H, W), np.int32)
counts = {k: np.zeros((H, W), np.int32) for k in ("scl", "ndwi", "mndwi")}
cls_only_scl = np.zeros((13, H, W), np.int32)

t0 = time.time()
for y0 in range(0, H, ROW):
    y1 = min(y0 + ROW, H)
    sl = {scl.dims[1]: slice(y0, y1)}
    g = np.asarray(b03.isel(sl).values, np.float32)
    n = np.asarray(b08.isel(sl).values, np.float32)
    s1 = np.asarray(b11.isel(sl).values, np.float32)
    s = np.nan_to_num(np.asarray(scl.isel(sl).values), nan=0.0).astype(np.int8)

    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = g + n
        ndwi = np.where(d1 != 0, (g - n) / d1, np.nan).astype(np.float32)
        d2 = g + s1
        mndwi = np.where(d2 != 0, (g - s1) / d2, np.nan).astype(np.float32)

    clear = ((~np.isin(s, BAD)) & (s > 0)
             & np.isfinite(ndwi) & np.isfinite(mndwi))
    n_clear[y0:y1] = clear.sum(0)
    w = {"scl": clear & (s == 6),
         "ndwi": clear & (ndwi > 0.0),
         "mndwi": clear & (mndwi > 0.0)}
    for k, v in w.items():
        counts[k][y0:y1] = v.sum(0)

    only_scl = clear & w["scl"] & ~w["ndwi"]
    for c in range(13):
        cls_only_scl[c, y0:y1] = (only_scl & (s == c)).sum(0)
    print(f"[det] filas {y1}/{H}  {time.strftime('%H:%M:%S')}", flush=True)

transform, crs = grid_from_dataset(ds)
ds.close()
print(f"recorrido en {(time.time()-t0)/60:.1f} min", flush=True)

MIN_OBS = 20
ok = n_clear >= MIN_OBS
wf = {k: np.where(ok, v / np.maximum(n_clear, 1), np.nan).astype("float32")
      for k, v in counts.items()}
for k, a in wf.items():
    pit.write_geotiff(f"{OUT}/wf_{k}_grande.tif", a, transform, crs,
                      "float32", np.nan)
np.save(f"{OUT}/n_clear_grande.npy", n_clear)

px = abs(transform.a * transform.e) / 1e6
print(f"\npixeles con >= {MIN_OBS} observaciones: {ok.sum():,} "
      f"({ok.sum()*px:.0f} km2)")
print(f"\nfrecuencia de agua media: "
      + "  ".join(f"{k} {np.nanmean(a):.3f}" for k, a in wf.items()))

print("\nDIFERENCIAS FRENTE AL NDWI (referencia de 10 m nativos):")
for k in ("scl", "mndwi"):
    d = wf["ndwi"] - wf[k]
    m = np.isfinite(d)
    print(f"\n  {k.upper()}:  mediana {np.nanmedian(d):+.3f}, "
          f"media {np.nanmean(d):+.3f}")
    for lo, hi, lab in [(-2, -0.2, f"{k} ve MUCHA mas agua"),
                        (-0.2, -0.05, f"{k} ve algo mas"),
                        (-0.05, 0.05, "coinciden"),
                        (0.05, 0.2, "NDWI ve algo mas"),
                        (0.2, 2, "NDWI ve MUCHA mas agua")]:
        n_ = int(np.sum((d > lo) & (d <= hi)))
        print(f"    {lab:28s} {n_:9,d} px {n_*px:7.2f} km2 "
              f"({100*n_/max(m.sum(),1):5.1f} %)")

tot = cls_only_scl.sum()
print(f"\nCLASE SCL DONDE SCL DICE AGUA Y EL NDWI NO ({tot:,} obs):")
for c in np.argsort(cls_only_scl.sum(axis=(1, 2)))[::-1][:5]:
    n_ = int(cls_only_scl[c].sum())
    if n_:
        print(f"  {c:2d} {pit.water.SCL_CLASSES[c][0]:32s} {n_:11,d} "
              f"{100*n_/tot:5.1f} %")
