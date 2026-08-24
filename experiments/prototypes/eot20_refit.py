"""HSR and DEA refitted on EOT20, judged against the RTK survey.

Both methods already share a tide in our code, so this is not about making
the comparison fair — it already was. It asks a different question: does a
tide model with an ocean cell 25 km away instead of 36 km make either method
better? EOT20 and GOT4.10 differ by 0.142 m RMS over these scenes, which is
the same order as the 0.135 m RMSE we are trying to beat, so the answer is
not obviously no.

Four fits, one script, identical masking, and every number read off the SAME
field pixels — comparing methods on different pixel sets is how we fooled
ourselves earlier.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer

import pyintertidal as pit
from pyintertidal.elevation import fit_hsr, fit_step, epochs
from pyintertidal import terrain

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CACHE = "ndwi_cube_villaviciosa_grande_10y.nc"

aoi = pit.sites.get("villaviciosa")
cube = pit.SentinelCube(aoi, ["2016-01-01", "2025-12-31"], water="ndwi",
                        cache_path=CACHE)
transform, crs = cube.grid
SH = cube.shape
dates = cube.dates

wf = rasterio.open("products_villaviciosa/water_frequency.tif").read(1)
with rasterio.open("products_villaviciosa/reference_map.tif") as s:
    ref = s.read(1)
poly = aoi.raster_mask(transform, crs, SH)
inter = pit.intertidal_mask(wf, ref, 0.01, 0.99, aoi_mask=poly)

TIDES = {
    "GOT4.10": json.load(open(os.path.join(SC, "tides.json"))),
    "EOT20": json.load(open(os.path.join(SC, "tides_eot20.json"))),
}
ep = epochs([d for d in dates if d in TIDES["GOT4.10"]], 3)[-1]
print(f"epoca {ep['label']}, {len(ep['dates'])} fechas\n", flush=True)

products = {}
for tname, tide in TIDES.items():
    t0 = time.time()
    r = fit_hsr(cube, tide, dates=ep["dates"], clip_mask=inter,
                epoch_label=ep["label"], superresolve=False,
                tide_bins="auto")   # same axis as the 0.158 m baseline
    mu = terrain.drop_small_regions(np.asarray(r.mu, "float32"),
                                    min_region_px=25)
    products[f"HSR / {tname}"] = mu
    print(f"  HSR / {tname:8s} {int(np.isfinite(mu).sum()):7,d} px "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

    t0 = time.time()
    st = fit_step(cube, tide, dates=ep["dates"], clip_mask=inter,
                  epoch_label=ep["label"], verbose=False)
    sv = np.asarray(st.mu, "float32")
    products[f"DEA / {tname}"] = sv
    print(f"  DEA / {tname:8s} {int(np.isfinite(sv).sum()):7,d} px "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

# ── the RTK survey ───────────────────────────────────────────────────────
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
print(f"\n{len(uniq)} pixeles con medida de campo", flush=True)


def clean(a):
    return np.where(np.isfinite(a) & (a > -900), a, np.nan)


com = np.all([np.isfinite(clean(a)[rr, cc]) for a in products.values()], 0)
print(f"\nSOBRE LOS MISMOS {int(com.sum())} PIXELES:")
print(f"{'producto':18s} {'RMSE':>7s} {'MAE':>7s} {'r':>7s} {'pendiente':>10s}")
print("-" * 54)
rows = []
for k, a in products.items():
    v = clean(a)[rr, cc]
    d = gnss[com] - v[com]
    d = d - np.median(d)
    rm = float(np.sqrt(np.mean(d ** 2)))
    r_ = float(np.corrcoef(gnss[com], v[com])[0, 1])
    slope = float(np.polyfit(gnss[com], v[com], 1)[0])
    print(f"{k:18s} {rm:7.3f} {np.mean(np.abs(d)):7.3f} {r_:7.3f} "
          f"{slope:10.3f}")
    rows.append((k, rm, r_, slope))

print("\nefecto de cambiar GOT4.10 -> EOT20:")
for meth in ("HSR", "DEA"):
    g = next(r for r in rows if r[0] == f"{meth} / GOT4.10")
    e = next(r for r in rows if r[0] == f"{meth} / EOT20")
    print(f"  {meth}: RMSE {g[1]:.3f} -> {e[1]:.3f} ({e[1]-g[1]:+.3f} m)   "
          f"pendiente {g[3]:.3f} -> {e[3]:.3f}")

best = min(rows, key=lambda r: r[1])
print(f"\nmejor: {best[0]} con RMSE {best[1]:.3f} m (r {best[2]:+.3f})")
json.dump([{"producto": k, "rmse": rm, "r": r_, "pendiente": sl}
           for k, rm, r_, sl in rows],
          open(os.path.join(SC, "eot20_refit.json"), "w"), indent=1)
