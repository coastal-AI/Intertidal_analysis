"""Does borrowing DEA's ideas make HSR better? Judged on the SAME pixels.

Three things taken from their code, each with a reason to expect a gain:
  * weight each tide bin by how many scenes went into it — the principled
    version of the smoothing they apply before reading off a crossing;
  * drop bins backed by fewer than `min_bin_count` scenes;
  * require the pixel's NDWI to correlate with the tide at all.
"""
import os, sys, json, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import pyintertidal as pit
from pyintertidal.elevation import hsr_stage1, epochs
from pyintertidal import terrain

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CACHE = "ndwi_cube_villaviciosa_grande_10y.nc"
aoi = pit.sites.get("villaviciosa")

import xarray as xr
ds = xr.open_dataset(CACHE)
dates = [str(t)[:10] for t in ds.t.values]
transform = None
ds.close()

with rasterio.open("products_villaviciosa/reference_map.tif") as s:
    ref, transform, crs, SH = s.read(1), s.transform, s.crs, s.shape
wf = rasterio.open("products_villaviciosa/water_frequency.tif").read(1)
poly = aoi.raster_mask(transform, crs, SH)
inter = pit.intertidal_mask(wf, ref, 0.01, 0.99, aoi_mask=poly)

tides = json.load(open(os.path.join(SC, "tides.json")))
ep = epochs([d for d in dates if d in tides], 3)[-1]
print(f"epoca {ep['label']}, {len(ep['dates'])} fechas", flush=True)

RUNS = {
    "HSR sin pesos":        dict(weight_bins=False, min_bin_count=1,
                                 min_correlation=None),
    "HSR con pesos":        dict(weight_bins=True,  min_bin_count=1,
                                 min_correlation=None),
    "HSR pesos + corr0.15": dict(weight_bins=True,  min_bin_count=1,
                                 min_correlation=0.15),
}

products = {}
for label, kw in RUNS.items():
    t0 = time.time()
    s1 = hsr_stage1(CACHE, dates, tides, valid_dates=ep["dates"],
                    tide_bins="auto", **kw)
    valid = (s1["valid"] & inter & np.isfinite(s1["sigma_mu"])
             & (s1["sigma_mu"] <= 0.10))
    mu = np.where(valid, s1["mu"], np.nan).astype("float32")
    mu = terrain.drop_small_regions(mu, min_region_px=25)
    products[label] = mu
    print(f"  {label:32s} {int(np.isfinite(mu).sum()):7,d} px "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

# the faithful DEA baseline, already computed
products["DEA fiel"] = rasterio.open(
    "products_villaviciosa/step_1.tif").read(1)

# ── field survey ─────────────────────────────────────────────────────────
fx = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
fx = fx[fx["Solution status"] == "FIX"]
tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = tf.transform(fx["Longitude"].values, fx["Latitude"].values)
col = ((x - transform.c) / transform.a).astype(int)
row = ((y - transform.f) / transform.e).astype(int)
key = row * SH[1] + col
uniq, inv = np.unique(key, return_inverse=True)
gnss = (np.bincount(inv, weights=fx["Ellipsoidal height"].values)
        / np.bincount(inv))
rr, cc = uniq // SH[1], uniq % SH[1]


def clean(a):
    return np.where(np.isfinite(a) & (a > -900), a, np.nan)


print(f"\n{'variante':32s} {'px':>5s} {'RMSE':>8s} {'MAE':>8s} {'r':>7s}")
print("-" * 64)
for k, a in products.items():
    v = clean(a)[rr, cc]
    m = np.isfinite(v)
    d = gnss[m] - v[m]
    d -= np.median(d)
    print(f"{k:32s} {m.sum():5d} {np.sqrt(np.mean(d**2)):8.3f} "
          f"{np.mean(np.abs(d)):8.3f} {np.corrcoef(gnss[m], v[m])[0,1]:7.3f}")

com = np.all([np.isfinite(clean(a)[rr, cc]) for a in products.values()], 0)
print(f"\nSOBRE LOS MISMOS {int(com.sum())} PIXELES (la unica comparacion valida):")
best = None
for k, a in products.items():
    v = clean(a)[rr, cc]
    d = gnss[com] - v[com]
    d -= np.median(d)
    rm = np.sqrt(np.mean(d ** 2))
    r = np.corrcoef(gnss[com], v[com])[0, 1]
    print(f"  {k:32s} RMSE {rm:.3f}   r {r:+.3f}")
    if best is None or rm < best[1]:
        best = (k, rm)
print(f"\nmejor: {best[0]} con RMSE {best[1]:.3f} m")
