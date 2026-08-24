"""The Tagus validation: full pipeline, then compare against real bathymetry.

The first comparison of this project against a reference that actually
measures the flats — float precision, 100 % surveyed, covering +0.5 to
+3.4 m above LAT.
"""
import os, sys, json, time
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import xarray as xr

import pyintertidal as pit
from pyintertidal import SentinelCube, TideService, terrain

OUT = "products_tejo"
os.makedirs(OUT, exist_ok=True)
CACHE = "ndwi_cube_tejo_2019_2021.nc"
t_all = time.time()

# ── 1 · cube ─────────────────────────────────────────────────────────────
aoi = pit.sites.get("tejo")
cube = SentinelCube(aoi, ("2019-01-01", "2021-12-31"), water="ndwi",
                    cache_path=CACHE, resolution=10)
transform, crs = cube.grid
px_km2 = abs(transform.a * transform.e) / 1e6
print(f"[1] cubo: {len(cube.dates)} escenas · {cube.shape}", flush=True)

# ── 2 · reference map + clouds ───────────────────────────────────────────
NDWI_THRESHOLD = 0.0
ref, cloud_pct = pit.reference_and_clouds(
    cube, threshold=NDWI_THRESHOLD, bad_classes=pit.water.BAD_CLASSES,
    clean_scene_max_bad=0.05, stable_threshold=0.95, transition_buffer_px=10)
usable = pit.filter_dates(cloud_pct, cloud_threshold=0.10)
for code, nm in [(0, "transicion"), (1, "agua estable"), (2, "tierra estable")]:
    n = int((ref == code).sum())
    print(f"    {nm:15s} {n:8,d} px  {n * px_km2:6.2f} km2", flush=True)
print(f"[2] fechas usables: {len(usable)} de {len(cube.dates)}", flush=True)

# ── 3 · water frequency + intertidal ─────────────────────────────────────
MIN_OBS = max(8, round(0.05 * len(usable)))
wf = pit.water_frequency(cube, dates=usable, threshold=NDWI_THRESHOLD,
                         min_obs=MIN_OBS)
poly = aoi.raster_mask(transform, crs, ref.shape)
WF_LOW, WF_HIGH = 0.05, 0.95
inter = pit.intertidal_mask(wf, ref, WF_LOW, WF_HIGH, aoi_mask=poly)
print(f"[3] intermareal: {int(inter.sum()):,} px = "
      f"{inter.sum() * px_km2:.2f} km2", flush=True)

# ── 4 · tides ────────────────────────────────────────────────────────────
tide_point = pit.water_centroid(ref, transform, crs)
tides = TideService(model="GOT4.10", directory="./tide_models",
                    location=tide_point)
tide_heights = tides.heights_for(aoi, cube.dates, verbose=False)
print(f"[4] marea en ({tide_point[0]:.4f}, {tide_point[1]:.4f}): "
      f"{len(tide_heights)} fechas", flush=True)
json.dump(tide_heights, open(f"{OUT}/tides.json", "w"))

# ── 5 · HSR + step, epoch matched to the 2020 survey ─────────────────────
eps = pit.epochs(sorted(tide_heights), epoch_years=3)
ep = eps[-1]
print(f"[5] ajustando epoca {ep['label']} ({len(ep['dates'])} fechas)...",
      flush=True)
t0 = time.time()
dem = pit.fit_hsr(cube, tide_heights, dates=ep["dates"], scale=4,
                  max_uncertainty=0.10, clip_mask=inter,
                  epoch_label=ep["label"], tide_bins=40, superresolve=False)
dem.mu = terrain.drop_small_regions(dem.mu, min_region_px=25)
print(f"    HSR en {(time.time()-t0)/60:.1f} min: "
      f"{int(np.isfinite(dem.mu).sum()):,} px", flush=True)

dem_step = pit.fit_step(cube, tide_heights, dates=ep["dates"],
                        threshold=NDWI_THRESHOLD, min_obs=5, clip_mask=inter,
                        epoch_label=ep["label"])
dem_step.mu = terrain.drop_small_regions(dem_step.mu, min_region_px=25)
print(f"    step: {int(np.isfinite(dem_step.mu).sum()):,} px", flush=True)

pit.write_geotiff(f"{OUT}/hsr_elevation.tif", dem.mu, transform, crs,
                  "float32", np.nan)
pit.write_geotiff(f"{OUT}/step_elevation.tif", dem_step.mu, transform, crs,
                  "float32", np.nan)
pit.write_geotiff(f"{OUT}/water_frequency.tif", wf, transform, crs,
                  "float32", np.nan)
pit.write_geotiff(f"{OUT}/intertidal_mask.tif", inter.astype("uint8"),
                  transform, crs, "uint8", None)

# ── 6 · the bathymetry, onto our grid ────────────────────────────────────
ds = xr.open_dataset("bathy_tagus/1528_IB_Tagus_2020_64.nc")
z = ds.elevation.values.astype("float32")
lon, lat = ds.lon.values, ds.lat.values
# netCDF rows run south->north; a GeoTIFF wants north->south.
if lat[0] < lat[-1]:
    z = z[::-1]
    lat = lat[::-1]
tr = rasterio.transform.from_origin(
    lon[0] - abs(lon[1] - lon[0]) / 2, lat[0] + abs(lat[1] - lat[0]) / 2,
    abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))
pit.write_geotiff(f"{OUT}/bathy_tagus_native.tif", z, tr, "EPSG:4326",
                  "float32", np.nan)
bathy = pit.reproject_to_grid(f"{OUT}/bathy_tagus_native.tif", transform, crs,
                              ref.shape)
print(f"[6] batimetria en nuestra malla: "
      f"{int(np.isfinite(bathy).sum()):,} px", flush=True)

# ── 7 · VALIDATION ───────────────────────────────────────────────────────
both = inter & np.isfinite(bathy)
print(f"\n{'=' * 62}")
print(f"VALIDACION · {int(both.sum()):,} px con mascara + batimetria "
      f"({both.sum() * px_km2:.2f} km2)")
print("=" * 62, flush=True)

v = bathy[both]
print(f"referencia: {np.unique(np.round(v, 2)).size:,} valores distintos, "
      f"{100 * np.mean(v != np.round(v)):.0f} % no enteros")
print(f"            rango {v.min():+.2f} .. {v.max():+.2f} m sobre LAT")
print(f"  (en Villaviciosa: 83 % de la mascara en UN valor, +2.00 m)\n")

rows = pit.compare_dems({"HSR": dem.mu, "step (DEA)": dem_step.mu},
                        bathy, mask=inter)
hdr = f"{'metodo':14s} {'n':>8s} {'sesgo':>8s} {'RMSE':>8s} {'MAE':>8s} {'r':>7s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    if "rmse_m" not in r:
        print(f"{r['method']:14s} {r['n']:8,d}  (pocos px)"); continue
    print(f"{r['method']:14s} {r['n']:8,d} {r['bias_m']:8.2f} "
          f"{r['rmse_m']:8.3f} {r['mae_m']:8.3f} {r['pearson_r']:7.3f}")

print("\ncontrol fisico (rho vs frecuencia de agua, debe ser NEGATIVO):")
for nm, a in [("HSR", dem.mu), ("step", dem_step.mu),
              ("batimetria", bathy)]:
    c = pit.validation.wf_vs_elevation(wf, a, mask=both)
    print(f"  {nm:12s} rho = {c['spearman']:+.3f}  n = {c['n']:,}")

json.dump({"validation": rows, "n_common": int(both.sum()),
           "epoch": ep["label"], "n_dates": len(ep["dates"])},
          open(f"{OUT}/validation.json", "w"), indent=1)
print(f"\ntotal {(time.time()-t_all)/60:.1f} min -> {OUT}/", flush=True)
