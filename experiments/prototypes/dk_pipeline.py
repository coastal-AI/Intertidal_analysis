"""Second validation: the Danish Wadden. Same pipeline, harder site."""
import os, sys, json, time, glob, zipfile
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import xarray as xr

import pyintertidal as pit
from pyintertidal import SentinelCube, TideService, terrain

OUT = "products_vadehavet"
os.makedirs(OUT, exist_ok=True)
CACHE = "ndwi_cube_vadehavet_2019_2021.nc"
t_all = time.time()

aoi = pit.sites.get("vadehavet")
cube = SentinelCube(aoi, ("2019-01-01", "2021-12-31"), water="ndwi",
                    cache_path=CACHE, resolution=10)
transform, crs = cube.grid
px_km2 = abs(transform.a * transform.e) / 1e6
print(f"[1] {len(cube.dates)} escenas · {cube.shape}", flush=True)

NDWI_THRESHOLD = 0.0
ref, cloud_pct = pit.reference_and_clouds(
    cube, threshold=NDWI_THRESHOLD, bad_classes=pit.water.BAD_CLASSES,
    clean_scene_max_bad=0.05, stable_threshold=0.95, transition_buffer_px=10)
usable = pit.filter_dates(cloud_pct, cloud_threshold=0.10)
print(f"[2] fechas usables: {len(usable)}/{len(cube.dates)}", flush=True)

MIN_OBS = max(8, round(0.05 * len(usable)))
wf = pit.water_frequency(cube, dates=usable, threshold=NDWI_THRESHOLD,
                         min_obs=MIN_OBS)
poly = aoi.raster_mask(transform, crs, ref.shape)
inter = pit.intertidal_mask(wf, ref, 0.05, 0.95, aoi_mask=poly)
print(f"[3] intermareal: {int(inter.sum()):,} px = "
      f"{inter.sum() * px_km2:.2f} km2", flush=True)

tide_point = pit.water_centroid(ref, transform, crs)
tides = TideService(model="GOT4.10", directory="./tide_models",
                    location=tide_point)
tide_heights = tides.heights_for(aoi, cube.dates, verbose=False)
json.dump(tide_heights, open(f"{OUT}/tides.json", "w"))
th = np.array(list(tide_heights.values()))
print(f"[4] marea: {len(tide_heights)} fechas, rango "
      f"{th.min():+.2f}..{th.max():+.2f} m", flush=True)

ep = pit.epochs(sorted(tide_heights), epoch_years=3)[-1]
print(f"[5] epoca {ep['label']} ({len(ep['dates'])} fechas)...", flush=True)
t0 = time.time()
dem = pit.fit_hsr(cube, tide_heights, dates=ep["dates"], scale=4,
                  max_uncertainty=0.10, clip_mask=inter,
                  epoch_label=ep["label"], tide_bins="auto",
                  superresolve=False)
dem.mu = terrain.drop_small_regions(dem.mu, min_region_px=25)
print(f"    HSR en {(time.time()-t0)/60:.1f} min: "
      f"{int(np.isfinite(dem.mu).sum()):,} px", flush=True)

dem_step = pit.fit_step(cube, tide_heights, dates=ep["dates"],
                        threshold=NDWI_THRESHOLD, min_obs=5, clip_mask=inter,
                        epoch_label=ep["label"])
dem_step.mu = terrain.drop_small_regions(dem_step.mu, min_region_px=25)
print(f"    step: {int(np.isfinite(dem_step.mu).sum()):,} px", flush=True)

for nm, a, dt in [("hsr_elevation", dem.mu, "float32"),
                  ("step_elevation", dem_step.mu, "float32"),
                  ("water_frequency", wf, "float32")]:
    pit.write_geotiff(f"{OUT}/{nm}.tif", a, transform, crs, dt, np.nan)
pit.write_geotiff(f"{OUT}/intertidal_mask.tif", inter.astype("uint8"),
                  transform, crs, "uint8", None)

# ── reference ────────────────────────────────────────────────────────────
nc = glob.glob("bathy_candidatos/*Danske*.nc")[0]
ds = xr.open_dataset(nc)
z = ds.elevation.values.astype("float32")
lon, lat = ds.lon.values, ds.lat.values
if lat[0] < lat[-1]:
    z, lat = z[::-1], lat[::-1]
tr = rasterio.transform.from_origin(
    lon[0] - abs(lon[1] - lon[0]) / 2, lat[0] + abs(lat[1] - lat[0]) / 2,
    abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))
pit.write_geotiff(f"{OUT}/bathy_native.tif", z, tr, "EPSG:4326", "float32",
                  np.nan)
bathy = pit.reproject_to_grid(f"{OUT}/bathy_native.tif", transform, crs,
                              ref.shape)
both = inter & np.isfinite(bathy)
print(f"[6] batimetria: {int(both.sum()):,} px en la mascara "
      f"({both.sum() * px_km2:.2f} km2)", flush=True)

print(f"\n{'=' * 60}\nVALIDACION · Wadden danes\n{'=' * 60}")
rows = pit.compare_dems({"HSR": dem.mu, "step (DEA)": dem_step.mu}, bathy,
                        mask=inter)
h = f"{'metodo':14s} {'n':>8s} {'sesgo':>7s} {'RMSE':>7s} {'MAE':>7s} {'r':>7s}"
print(h); print("-" * len(h))
for r in rows:
    if "rmse_m" not in r:
        print(f"{r['method']:14s} {r['n']:8,d} (pocos)"); continue
    print(f"{r['method']:14s} {r['n']:8,d} {r['bias_m']:7.2f} "
          f"{r['rmse_m']:7.3f} {r['mae_m']:7.3f} {r['pearson_r']:7.3f}")

common = both & np.isfinite(dem.mu) & np.isfinite(dem_step.mu)
print(f"\nSOBRE LOS MISMOS {int(common.sum()):,} PIXELES:")
rows2 = pit.compare_dems({"HSR": dem.mu, "step (DEA)": dem_step.mu}, bathy,
                         mask=common)
for r in rows2:
    print(f"  {r['method']:14s} RMSE {r['rmse_m']:.3f}  MAE {r['mae_m']:.3f}"
          f"  r {r['pearson_r']:+.3f}")

print("\ncontrol fisico (rho vs frecuencia de agua):")
for nm, a in [("HSR", dem.mu), ("step", dem_step.mu), ("batimetria", bathy)]:
    c = pit.validation.wf_vs_elevation(wf, a, mask=both)
    print(f"  {nm:12s} rho = {c['spearman']:+.3f}  n = {c['n']:,}")

json.dump({"all": rows, "common": rows2, "n_common": int(common.sum()),
           "epoch": ep["label"], "tide_range": float(th.max() - th.min())},
          open(f"{OUT}/validation.json", "w"), indent=1)
print(f"\ntotal {(time.time()-t_all)/60:.1f} min", flush=True)
