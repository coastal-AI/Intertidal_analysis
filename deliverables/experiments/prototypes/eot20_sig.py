"""Is HSR really level with DEA now, or did we just get a lucky 110 pixels?

The refit says HSR improves by 24 mm when the tide changes and DEA worsens by
15 mm. A change that helps one method and hurts the other is exactly what
noise looks like, so nothing here is claimed until a paired test says the
difference survives resampling. Point RMSEs from 110 pixels are not evidence
on their own.

Second question in the same run: HSR's compression slope moved 0.688 -> 0.853
on a 0.142 m change in tide. If that sensitivity comes from the tide wave
amplifying as it runs up the estuary — which no open-ocean model represents —
the compression must be worse upstream. If the slope is flat along the axis,
that hypothesis dies too.

Products are written to disk this time. Refitting costs 42 minutes and we
have now paid it twice.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import pyintertidal as pit
from pyintertidal.elevation import fit_hsr, fit_step, epochs
from pyintertidal import terrain

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CACHE = "ndwi_cube_villaviciosa_grande_10y.nc"
OUT = "products_villaviciosa"

aoi = pit.sites.get("villaviciosa")
cube = pit.SentinelCube(aoi, ["2016-01-01", "2025-12-31"], water="ndwi",
                        cache_path=CACHE)
transform, crs = cube.grid
SH = cube.shape

wf = rasterio.open(f"{OUT}/water_frequency.tif").read(1)
with rasterio.open(f"{OUT}/reference_map.tif") as s:
    ref = s.read(1)
poly = aoi.raster_mask(transform, crs, SH)
inter = pit.intertidal_mask(wf, ref, 0.01, 0.99, aoi_mask=poly)

TIDES = {"GOT4.10": json.load(open(os.path.join(SC, "tides.json"))),
         "EOT20": json.load(open(os.path.join(SC, "tides_eot20.json")))}
ep = epochs([d for d in cube.dates if d in TIDES["GOT4.10"]], 3)[-1]
print(f"epoca {ep['label']}, {len(ep['dates'])} fechas\n", flush=True)

products = {}
for tname, tide in TIDES.items():
    tag = tname.replace(".", "").lower()
    for meth in ("HSR", "DEA"):
        key = f"{meth} / {tname}"
        path = f"{OUT}/{meth.lower()}_{tag}_{ep['label']}.tif"
        if os.path.exists(path):
            products[key] = rasterio.open(path).read(1)
            print(f"  {key:16s} cacheado", flush=True)
            continue
        t0 = time.time()
        if meth == "HSR":
            r = fit_hsr(cube, tide, dates=ep["dates"], clip_mask=inter,
                        epoch_label=ep["label"], superresolve=False,
                        tide_bins="auto")
            a = terrain.drop_small_regions(np.asarray(r.mu, "float32"),
                                           min_region_px=25)
        else:
            st = fit_step(cube, tide, dates=ep["dates"], clip_mask=inter,
                          epoch_label=ep["label"], verbose=False)
            a = np.asarray(st.mu, "float32")
        pit.write_geotiff(path, a, transform, crs, "float32", np.nan)
        products[key] = a
        print(f"  {key:16s} {int(np.isfinite(a).sum()):7,d} px "
              f"({(time.time()-t0)/60:.1f} min) -> {path}", flush=True)

# ── the RTK survey ───────────────────────────────────────────────────────
fx = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
fx = fx[fx["Solution status"] == "FIX"]
tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = tf.transform(fx["Longitude"].values, fx["Latitude"].values)
col = ((x - transform.c) / transform.a).astype(int)
row = ((y - transform.f) / transform.e).astype(int)
key_ = row * SH[1] + col
uniq, inv = np.unique(key_, return_inverse=True)
gnss = (np.bincount(inv, weights=fx["Ellipsoidal height"].values)
        / np.bincount(inv))
gx = np.bincount(inv, weights=x) / np.bincount(inv)
gy = np.bincount(inv, weights=y) / np.bincount(inv)
rr, cc = uniq // SH[1], uniq % SH[1]


def clean(a):
    return np.where(np.isfinite(a) & (a > -900), a, np.nan)


com = np.all([np.isfinite(clean(a)[rr, cc]) for a in products.values()], 0)
n = int(com.sum())
print(f"\n{n} pixeles comunes\n", flush=True)

res = {}
for k, a in products.items():
    v = clean(a)[rr, cc][com]
    d = gnss[com] - v
    res[k] = d - np.median(d)

print(f"{'producto':18s} {'RMSE':>7s} {'pendiente':>10s}")
print("-" * 38)
for k, d in res.items():
    v = clean(products[k])[rr, cc][com]
    print(f"{k:18s} {np.sqrt(np.mean(d**2)):7.3f} "
          f"{np.polyfit(gnss[com], v, 1)[0]:10.3f}")

# ── paired bootstrap: does any pair actually separate? ───────────────────
rng = np.random.default_rng(20260816)
BOOT = 20000
idx = rng.integers(0, n, size=(BOOT, n))
print(f"\nBOOTSTRAP PAREADO ({BOOT:,} remuestreos de los {n} pixeles)")
print("diferencia de RMSE, negativo = gana el primero")
print(f"{'comparacion':34s} {'dif':>7s} {'IC 95%':>17s} {'p':>7s}")
print("-" * 70)
keys = list(res)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a_, b_ = res[keys[i]] ** 2, res[keys[j]] ** 2
        obs = np.sqrt(a_.mean()) - np.sqrt(b_.mean())
        boot = np.sqrt(a_[idx].mean(1)) - np.sqrt(b_[idx].mean(1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        flag = "SIGNIFICATIVO" if lo * hi > 0 else "no significativo"
        print(f"{keys[i]:>16s} vs {keys[j]:<16s} {obs:+7.3f} "
              f"[{lo:+6.3f},{hi:+6.3f}] {p:7.3f}  {flag}")

# ── does the compression worsen upstream? ────────────────────────────────
# Distance from the AOI's seaward edge along the estuary: the ria runs
# roughly north-south, so northing is a fair proxy for position on the axis.
print("\nCOMPRESION A LO LARGO DE LA RIA (hipotesis de amplificacion)")
dist = (gy[com].max() - gy[com]) / 1000.0
order = np.argsort(dist)
half = n // 2
low, high = order[:half], order[half:]
print(f"  mitad exterior (bocana):  {dist[low].min():.2f}-{dist[low].max():.2f} km")
print(f"  mitad interior (arriba):  {dist[high].min():.2f}-{dist[high].max():.2f} km")
print(f"\n{'producto':18s} {'pend.exterior':>14s} {'pend.interior':>14s} {'cambio':>8s}")
print("-" * 58)
for k in keys:
    v = clean(products[k])[rr, cc][com]
    s_lo = np.polyfit(gnss[com][low], v[low], 1)[0]
    s_hi = np.polyfit(gnss[com][high], v[high], 1)[0]
    print(f"{k:18s} {s_lo:14.3f} {s_hi:14.3f} {s_hi - s_lo:+8.3f}")
print("\nsi la amplificacion estuarina fuese la causa, la pendiente interior")
print("deberia ser CLARAMENTE menor que la exterior en todos los productos")

json.dump({k: res[k].tolist() for k in res},
          open(os.path.join(SC, "residuos.json"), "w"))
print("\nresiduos guardados -> residuos.json")
