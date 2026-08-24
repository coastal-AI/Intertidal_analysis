"""Re-run the DEA baseline faithfully, and re-do the field comparison.

Beating a weakened version of somebody else's method proves nothing, so the
baseline is rebuilt to the paper's own settings before any claim is made.
"""
import os, sys, json, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import pyintertidal as pit
from pyintertidal import SentinelCube, terrain

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CACHE = "ndwi_cube_villaviciosa_grande_10y.nc"
aoi = pit.sites.get("villaviciosa")
cube = SentinelCube(aoi, ("2016-01-01", "2025-12-31"), water="ndwi",
                    cache_path=CACHE, resolution=10)
transform, crs = cube.grid

tides = json.load(open(os.path.join(SC, "tides.json")))
ep = pit.epochs(sorted(tides), 3)[-1]
ref = rasterio.open("products_villaviciosa/reference_map.tif").read(1)
wf = rasterio.open("products_villaviciosa/water_frequency.tif").read(1)
poly = aoi.raster_mask(transform, crs, ref.shape)
inter = pit.intertidal_mask(wf, ref, 0.01, 0.99, aoi_mask=poly)
print(f"epoca {ep['label']}, {len(ep['dates'])} fechas, "
      f"mascara {int(inter.sum()):,} px", flush=True)

VARIANTS = {"DEA fiel (su codigo)": dict()}

out = {}
for label, kw in VARIANTS.items():
    t0 = time.time()
    d = pit.fit_step(cube, tides, dates=ep["dates"], min_obs=5,
                     clip_mask=inter, epoch_label=ep["label"], **kw)
    mu = terrain.drop_small_regions(d.mu, min_region_px=25)
    out[label] = mu
    print(f"  {label:36s} {int(np.isfinite(mu).sum()):7,d} px "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    pit.write_geotiff(f"products_villaviciosa/step_{len(out)}.tif", mu,
                      transform, crs, "float32", np.nan)

# ── against the field survey ─────────────────────────────────────────────
fixd = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
fixd = fixd[fixd["Solution status"] == "FIX"]
tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = tf.transform(fixd["Longitude"].values, fixd["Latitude"].values)
col = ((x - transform.c) / transform.a).astype(int)
row = ((y - transform.f) / transform.e).astype(int)
key = row * ref.shape[1] + col
uniq, inv = np.unique(key, return_inverse=True)
gnss = (np.bincount(inv, weights=fixd["Ellipsoidal height"].values)
        / np.bincount(inv))
rr, cc = uniq // ref.shape[1], uniq % ref.shape[1]

with rasterio.open("figuras_pablo/hsr_elevation_v2.tif") as s:
    hsr = s.read(1).astype("float32")
    hsr[hsr == s.nodata] = np.nan
out["HSR (nuestro)"] = hsr
out["diccionario"] = rasterio.open("bathymetry/bathymetry_pixels.tif").read(1)

print(f"\n{'metodo':36s} {'px':>5s} {'RMSE':>8s} {'MAE':>8s} {'r':>7s}")
print("-" * 68)
for k, a in out.items():
    a = np.where(np.isfinite(a) & (a > -900), a, np.nan)
    v = a[rr, cc]
    m = np.isfinite(v)
    if m.sum() < 10:
        print(f"{k:36s} {m.sum():5d}  (pocos)")
        continue
    d_ = gnss[m] - v[m]
    d_ -= np.median(d_)
    print(f"{k:36s} {m.sum():5d} {np.sqrt(np.mean(d_**2)):8.3f} "
          f"{np.mean(np.abs(d_)):8.3f} {np.corrcoef(gnss[m], v[m])[0,1]:7.3f}")

com = np.all([np.isfinite(np.where(a > -900, a, np.nan)[rr, cc])
              for a in out.values()], axis=0)
print(f"\nSOBRE LOS MISMOS {int(com.sum())} PIXELES:")
for k, a in out.items():
    a = np.where(np.isfinite(a) & (a > -900), a, np.nan)
    d_ = gnss[com] - a[rr, cc][com]
    d_ -= np.median(d_)
    print(f"  {k:36s} RMSE {np.sqrt(np.mean(d_**2)):.3f}  "
          f"r {np.corrcoef(gnss[com], a[rr, cc][com])[0,1]:+.3f}")
