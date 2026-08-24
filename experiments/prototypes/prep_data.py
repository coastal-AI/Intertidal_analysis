"""Step 1: tide heights for every cube date + the pixel-window extraction.

Everything cached to .npz so the figure script is fast to iterate on.
"""
import os, sys, json
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import xarray as xr
import rasterio

SCRATCH = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
           r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
           r"\scratchpad")
CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"

# ── tides ────────────────────────────────────────────────────────────────
tide_cache = os.path.join(SCRATCH, "tides.json")
ds = xr.open_dataset(CUBE)
dates = [str(t)[:10] for t in ds.t.values]

if os.path.exists(tide_cache):
    tides = json.load(open(tide_cache))
    print(f"[tides] cached: {len(tides)} dates")
else:
    from pyintertidal import sites
    from pyintertidal.tides import TideService
    aoi = sites.get("villaviciosa")
    ts = TideService(model="GOT4.10", directory="./tide_models")
    tides = ts.heights_for(aoi, dates)
    json.dump(tides, open(tide_cache, "w"))
    print(f"[tides] computed: {len(tides)} dates")

h = np.array(list(tides.values()))
print(f"  range {h.min():.2f} .. {h.max():.2f} m")

# ── products ─────────────────────────────────────────────────────────────
def load(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs

mu, TR, CRS = load("products_villaviciosa/hsr_elevation.tif")
sigma, _, _ = load("products_villaviciosa/hsr_sigma.tif")
unc, _, _ = load("products_villaviciosa/hsr_uncertainty.tif")
inter, _, _ = load("products_villaviciosa/intertidal_mask.tif")
print(f"[hsr] mu valido: {np.isfinite(mu).sum():,} px  "
      f"[{np.nanmin(mu):.2f}, {np.nanmax(mu):.2f}] m")

# ── pick a window with plenty of fitted intertidal pixels ────────────────
ok = np.isfinite(mu)
W = 96
best, best_n = None, -1
for y0 in range(0, 915 - W, 32):
    for x0 in range(0, 915 - W, 32):
        n = ok[y0:y0 + W, x0:x0 + W].sum()
        if n > best_n:
            best_n, best = n, (y0, x0)
y0, x0 = best
print(f"[window] y={y0}..{y0+W} x={x0}..{x0+W} -> {best_n} px ajustados")

# ── read the cube for that window (epoch 2023-2025 + the full record) ────
sub = ds.isel(y=slice(y0, y0 + W), x=slice(x0, x0 + W))
b03 = sub.B03.values.astype("float32")
b08 = sub.B08.values.astype("float32")
scl = sub.SCL.values
ds.close()

ndwi = (b03 - b08) / (b03 + b08)
CLEAR = {4, 5, 6, 7, 11}                       # veg, bare, water, unclass, snow
clear = np.isin(scl, list(CLEAR))
ndwi[~clear] = np.nan
print(f"[window] ndwi {ndwi.shape}, validos {np.isfinite(ndwi).mean()*100:.0f}%")

np.savez_compressed(
    os.path.join(SCRATCH, "window.npz"),
    ndwi=ndwi.astype("float32"), dates=np.array(dates),
    y0=y0, x0=x0, W=W,
    mu=mu[y0:y0 + W, x0:x0 + W], sigma=sigma[y0:y0 + W, x0:x0 + W],
    unc=unc[y0:y0 + W, x0:x0 + W], inter=inter[y0:y0 + W, x0:x0 + W])
print("OK -> window.npz")
