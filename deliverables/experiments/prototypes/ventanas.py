"""Two questions against the RTK survey.

1. Does opening the water-frequency window from [0.05, 0.95] to [0.01, 0.99]
   buy anything? Both windows are applied to the SAME fit, so the only thing
   that changes is which pixels are kept — no refitting noise in between.
2. How do the legacy bracketing methods (dictionary, isolines, optimised)
   score against real ground truth?
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

CACHE = "ndwi_cube_villaviciosa_grande_10y.nc"
SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


ref, TR, CRS, SH = load("products_villaviciosa/reference_map.tif")
wf = load("products_villaviciosa/water_frequency.tif")[0]
aoi = pit.sites.get("villaviciosa")
poly = aoi.raster_mask(TR, CRS, SH)

tides = json.load(open(os.path.join(SC, "tides.json")))
import xarray as xr
ds = xr.open_dataset(CACHE)
dates = [str(t)[:10] for t in ds.t.values]
ds.close()
ep = epochs([d for d in dates if d in tides], 3)[-1]
print(f"epoca {ep['label']}: {len(ep['dates'])} fechas", flush=True)

S1 = "products_villaviciosa/stage1_auto.npz"
if os.path.exists(S1):
    s1 = dict(np.load(S1))
    print("stage1 cacheado", flush=True)
else:
    t0 = time.time()
    s1 = hsr_stage1(CACHE, dates, tides, valid_dates=ep["dates"],
                    tide_bins="auto")
    np.savez_compressed(S1, **{k: s1[k] for k in
                               ("a", "b", "mu", "sigma", "rmse", "n_obs",
                                "sigma_mu", "valid")})
    print(f"stage1 en {(time.time()-t0)/60:.1f} min", flush=True)

# ── the field survey ─────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
fix = df[df["Solution status"] == "FIX"]
tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = tf.transform(fix["Longitude"].values, fix["Latitude"].values)
col = ((x - TR.c) / TR.a).astype(int)
row = ((y - TR.f) / TR.e).astype(int)
key = row * SH[1] + col
uniq, inv = np.unique(key, return_inverse=True)
gnss = (np.bincount(inv, weights=fix["Ellipsoidal height"].values)
        / np.bincount(inv))
rr, cc = uniq // SH[1], uniq % SH[1]
print(f"{len(fix)} puntos -> {len(uniq)} pixeles de 10 m\n", flush=True)


def score(arr, label, common=None):
    v = arr[rr, cc]
    m = np.isfinite(v) if common is None else (np.isfinite(v) & common)
    if m.sum() < 10:
        return None
    d = gnss[m] - v[m]
    d = d - np.median(d)
    return (label, int(m.sum()), float(np.sqrt(np.mean(d ** 2))),
            float(np.mean(np.abs(d))),
            float(np.corrcoef(gnss[m], v[m])[0, 1]))


# ── 1 · the two windows, same fit ────────────────────────────────────────
prods = {}
for lo, hi in [(0.05, 0.95), (0.01, 0.99)]:
    mask = pit.intertidal_mask(wf, ref, lo, hi, aoi_mask=poly)
    valid = (s1["valid"] & mask & np.isfinite(s1["sigma_mu"])
             & (s1["sigma_mu"] <= 0.10))
    mu = np.where(valid, s1["mu"], np.nan).astype("float32")
    mu = terrain.drop_small_regions(mu, min_region_px=25)
    prods[f"HSR wf[{lo:.2f},{hi:.2f}]"] = mu
    px_km2 = abs(TR.a * TR.e) / 1e6
    print(f"ventana [{lo:.2f}, {hi:.2f}]: mascara {int(mask.sum()):,} px, "
          f"producto {int(np.isfinite(mu).sum()):,} px = "
          f"{np.isfinite(mu).sum()*px_km2:.2f} km2", flush=True)

# ── 2 · the legacy methods ───────────────────────────────────────────────
for label, path in [("diccionario", "bathymetry/bathymetry_pixels.tif"),
                    ("isolineas", "bathymetry/bathymetry_isolines.tif"),
                    ("optimizado", "bathymetry/bathymetry.tif"),
                    ("escalon (DEA)", "bathymetry/bathymetry_step.tif")]:
    if os.path.exists(path):
        prods[label] = load(path)[0]

print(f"\n{'metodo':22s} {'px':>5s} {'RMSE':>8s} {'MAE':>8s} {'r':>8s}")
print("-" * 56)
rows = [score(a, k) for k, a in prods.items()]
for r in [r for r in rows if r]:
    print(f"{r[0]:22s} {r[1]:5d} {r[2]:8.3f} {r[3]:8.3f} {r[4]:8.3f}")

common = np.all([np.isfinite(a[rr, cc]) for a in prods.values()], axis=0)
print(f"\nSOBRE LOS MISMOS {int(common.sum())} PIXELES:")
for k, a in prods.items():
    r = score(a, k, common)
    if r:
        print(f"  {r[0]:22s} RMSE {r[2]:.3f}  MAE {r[3]:.3f}  r {r[4]:+.3f}")
