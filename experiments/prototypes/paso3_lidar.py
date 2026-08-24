# -*- coding: utf-8 -*-
"""Paso 3 — validación de los métodos de elevación contra el LiDAR 5 m del IGN."""
import sys, os, time, json
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import numpy as np, rasterio
from scipy.ndimage import zoom
from scipy.stats import pearsonr, spearmanr
from intertidal.validation import download_mdt_ign, reproject_to_grid
S=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

# ── 1. LiDAR IGN para el AOI grande ──────────────────────────────────────────
bbox={"west":-5.46,"south":43.47,"east":-5.35,"north":43.55,"crs":"EPSG:4326"}
mdt="mdt5_villaviciosa_grande.tif"
download_mdt_ign(bbox, mdt)
log(f"MDT IGN: {os.path.getsize(mdt)/1e6:.1f} MB")

def grid(p):
    with rasterio.open(p) as s: return s.transform, s.crs, s.shape
def band(p, i=1):
    with rasterio.open(p) as s: return s.read(i).astype(np.float32)

# ── 2. LiDAR al grid de 10 m (AOI grande) y al fino de 2.5 m ─────────────────
tf10, crs10, shp10 = grid("bathymetry/bathymetry.tif")
lid10 = reproject_to_grid(mdt, tf10, crs10, shp10)
tf25, crs25, shp25 = grid("bathymetry_hsr/hsr_dem_fine_v2.tif")
lid25 = reproject_to_grid(mdt, tf25, crs25, shp25)
log(f"LiDAR reproyectado: 10m {shp10}, 2.5m {shp25} | cobertura 10m {100*np.isfinite(lid10).mean():.0f}%")

imask = band("intertidal_mask.tif").astype(bool)

def stats(pred, ref, mask, label):
    m = mask & np.isfinite(pred) & np.isfinite(ref)
    n = int(m.sum())
    if n < 50:
        return {"metodo": label, "n": n}
    d = pred[m] - ref[m]
    bias = float(np.median(d))
    dd = d - bias
    return {
        "metodo": label, "n": n,
        "bias_m": round(bias, 3),
        "rmse_raw_m": round(float(np.sqrt((d**2).mean())), 3),
        "rmse_debias_m": round(float(np.sqrt((dd**2).mean())), 3),
        "mae_debias_m": round(float(np.abs(dd).mean()), 3),
        "pearson_r": round(float(pearsonr(pred[m], ref[m])[0]), 3),
        "spearman": round(float(spearmanr(pred[m], ref[m])[0]), 3),
    }

results=[]

# ── 3. Métodos a 10 m en el AOI grande (enmascarados al intermareal) ─────────
opt = band("bathymetry/bathymetry.tif", 1); opt[opt==0]=np.nan
results.append(stats(opt, lid10, imask, "optimized (bracketing+suavizado)"))
step = band("bathymetry/bathymetry_step.tif", 1); step[step==0]=np.nan
results.append(stats(step, lid10, imask, "step (escalon DEA)"))
mu = band("bathymetry_hsr/hsr_mu.tif")           # ya clip+limpio
results.append(stats(mu, lid10, np.isfinite(mu), "HSR mu (10 m)"))

# ── 4. Sub-píxel: a 2.5 m, HSR fino vs interpolar mu (baseline) ──────────────
hsr25 = band("bathymetry_hsr/hsr_dem_fine_v2.tif")
mu_f = np.where(np.isfinite(mu), mu, np.nan)
base25 = zoom(np.nan_to_num(mu_f, nan=0.0), 4, order=1)
basemask = zoom(np.isfinite(mu_f).astype(float), 4, order=1) > 0.99
base25 = np.where(basemask, base25, np.nan)
m25 = np.isfinite(hsr25)
results.append(stats(hsr25, lid25, m25, "HSR DEM fino (2.5 m)"))
results.append(stats(base25, lid25, m25, "baseline 2.5 m (interp. de mu)"))

# ── 5. Legacy del pentágono (pixels / isolines, su propio grid) ──────────────
for f,label in [("bathymetry/bathymetry_pixels.tif","pixels (pentagono viejo)"),
                ("bathymetry/bathymetry_isolines.tif","isolines (pentagono viejo)")]:
    tfp, crsp, shpp = grid(f)
    lidp = reproject_to_grid(mdt, tfp, crsp, shpp)
    e = band(f, 1); e[e==0]=np.nan
    results.append(stats(e, lidp, np.isfinite(e), label))

print()
print(f"{'método':38s} {'n':>7s} {'bias':>7s} {'RMSE':>6s} {'RMSE*':>6s} {'MAE*':>6s} {'r':>6s} {'rho':>6s}")
for r in results:
    if r.get("n",0) < 50:
        print(f"{r['metodo']:38s} {r['n']:>7d}  (insuficiente)"); continue
    print(f"{r['metodo']:38s} {r['n']:>7d} {r['bias_m']:>7.2f} {r['rmse_raw_m']:>6.2f} "
          f"{r['rmse_debias_m']:>6.2f} {r['mae_debias_m']:>6.2f} {r['pearson_r']:>6.3f} {r['spearman']:>6.3f}")
print("\n(* tras quitar el sesgo de datum LiDAR-vs-MSL; bias = mediana pred-LiDAR)")
json.dump(results, open(os.path.join(S,"paso3_lidar.json"),"w"), indent=2, ensure_ascii=False)
print("PASO3 LIDAR DONE")
