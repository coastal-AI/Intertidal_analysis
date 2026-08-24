"""Points 1-3 on the real cube, and the figure that shows whether they worked.

 1. every date with a tide height, not only the cloud-filtered ones
 2. a wider water-frequency window for the intertidal mask
 3. per-tide-bin medians before fitting
"""
import os, sys, json, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyintertidal as pit
from pyintertidal import viz, terrain
from pyintertidal.elevation import hsr_stage1, epochs
from pyintertidal.validation import reproject_to_grid

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"
OUT = "figuras_pablo"
TIDE_BINS = 40
WF_WINDOW = (0.05, 0.95)          # antes: multi-Otsu daba [0.21, 0.66]
MIN_REGION = 25


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


t_start = time.time()
tides = json.load(open(os.path.join(SC, "tides.json")))
ds = xr.open_dataset(CUBE)
dates = [str(t)[:10] for t in ds.t.values]
ds.close()

wf, TR, CRS, SH = load("products_villaviciosa/water_frequency.tif")
ref = load("products_villaviciosa/reference_map.tif")[0]
aoi = pit.sites.get("villaviciosa")
poly = aoi.raster_mask(TR, CRS, SH)

# ── Punto 2: ventana mas ancha ───────────────────────────────────────────
old_mask = load("products_villaviciosa/intertidal_mask.tif")[0] > 0
new_mask = pit.intertidal_mask(wf, ref, WF_WINDOW[0], WF_WINDOW[1],
                               aoi_mask=poly)
px_km2 = abs(TR.a * TR.e) / 1e6
print(f"[2] mascara intermareal: {old_mask.sum():,} px "
      f"({old_mask.sum()*px_km2:.2f} km2)  ->  {new_mask.sum():,} px "
      f"({new_mask.sum()*px_km2:.2f} km2)   "
      f"x{new_mask.sum()/max(old_mask.sum(),1):.1f}", flush=True)

# ── Punto 1: todas las fechas con marea ──────────────────────────────────
ep_all = epochs([d for d in dates if d in tides], 3)[-1]
print(f"[1] epoca {ep_all['label']}: {len(ep_all['dates'])} fechas "
      f"(antes 97, solo las filtradas por nubes)", flush=True)

# ── Punto 3: ajuste con binning por marea ────────────────────────────────
print(f"[3] ajustando con tide_bins={TIDE_BINS} ...", flush=True)
t0 = time.time()
s1 = hsr_stage1(CUBE, dates, tides, valid_dates=ep_all["dates"],
                tide_bins=TIDE_BINS)
print(f"    etapa 1 en {(time.time()-t0)/60:.1f} min", flush=True)

valid = s1["valid"] & new_mask
bad = ~np.isfinite(s1["sigma_mu"]) | (s1["sigma_mu"] > 0.10)
valid &= ~bad
mu = np.where(valid, s1["mu"], np.nan).astype("float32")
mu = terrain.drop_small_regions(mu, min_region_px=MIN_REGION)

n_new = int(np.isfinite(mu).sum())
old_mu = load("products_villaviciosa/hsr_elevation.tif")[0]
print(f"\nRESULTADO: {int(np.isfinite(old_mu).sum()):,} px -> {n_new:,} px "
      f"({n_new*px_km2:.2f} km2)", flush=True)
print(f"  rango [{np.nanmin(mu):.2f}, {np.nanmax(mu):.2f}] m", flush=True)
sg = np.where(valid, s1["sigma"], np.nan)
print(f"  sigma mediana {np.nanmedian(sg):.3f} m, "
      f"saturada {100*np.nanmean(sg[np.isfinite(sg)] >= 0.649):.1f} %",
      flush=True)

pit.write_geotiff(f"{OUT}/hsr_elevation_v2.tif", mu, TR, CRS, "float32", np.nan)

# ── Figura: antes / despues ──────────────────────────────────────────────
lidar = reproject_to_grid("mdt5_villaviciosa_grande.tif", TR, CRS, SH)
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
viz.plot_map(old_mu, TR, CRS, cmap="viridis", hillshade=True, robust=True,
             title=f"antes · {int(np.isfinite(old_mu).sum()):,} px\n"
                   "97 fechas, ventana [0.21, 0.66], sin binning",
             cbar_label="cota (m)", ax=axes[0])
viz.plot_dem(mu, TR, CRS, context=lidar, vert_exag=8,
             title=f"después · {n_new:,} px\n"
                   f"{len(ep_all['dates'])} fechas, ventana "
                   f"{list(WF_WINDOW)}, binning {TIDE_BINS}", ax=axes[1])
fig.tight_layout()
fig.savefig(f"{OUT}/13_mejoras.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 11))
viz.plot_dem(mu, TR, CRS, context=lidar, vert_exag=8,
             title="Ría de Villaviciosa · elevación intermareal (HSR, 10 m)",
             ax=ax)
fig.tight_layout()
fig.savefig(f"{OUT}/14_villaviciosa_final.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"\nlisto en {(time.time()-t_start)/60:.1f} min -> 13_mejoras.png, "
      f"14_villaviciosa_final.png", flush=True)
