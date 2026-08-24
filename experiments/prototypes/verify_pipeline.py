"""End-to-end verification of the notebook's core path on the real cube."""
import sys, os, time
PROJ = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pyintertidal as pit
from pyintertidal import SentinelCube, explain, viz

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

aoi = pit.sites.get("villaviciosa")
cube = SentinelCube(aoi, ("2016-01-01", "2025-12-31"),
                    cache_path="ndwi_cube_villaviciosa_grande_10y.nc")
transform, crs = cube.grid
log(f"cube {len(cube.dates)} dates {cube.shape}")

# calibration cell logic
samples = []
for _, blocks in cube.stream(chunk=8):
    ndwi = pit.water.index_block("ndwi", blocks)
    clear = np.isin(np.nan_to_num(blocks["SCL"], nan=0).astype("int16"),
                    list(pit.water.CLEAR_CLASSES))
    samples.append(ndwi[clear & np.isfinite(ndwi)])
    break
otsu = pit.water.otsu_threshold(np.concatenate(samples))
log(f"otsu threshold = {otsu:+.3f}")

NDWI_THRESHOLD = 0.0
import pickle, os
CACHE_INT = '_verify_intermediates.pkl'
if os.path.exists(CACHE_INT):
    reference_map, cloud_pct, wf_cached = pickle.load(open(CACHE_INT,'rb'))
    log('intermediates loaded from cache')
else:
    wf_cached = None
    reference_map, cloud_pct = pit.reference_and_clouds(
    cube, threshold=NDWI_THRESHOLD, clean_scene_max_bad=0.05,
        stable_threshold=0.95, transition_buffer_px=10)
px_km2 = abs(transform.a * transform.e) / 1e6
log(f"reference map: transition {int((reference_map==0).sum()):,} px "
    f"({(reference_map==0).sum()*px_km2:.2f} km2)")

usable = pit.filter_dates(cloud_pct, cloud_threshold=0.10)
gain = pit.date_recovery_gain(cloud_pct, 0.05, 0.10)
log(f"usable dates: {len(usable)}")

MIN_OBS = max(8, round(0.05 * len(usable)))
if wf_cached is not None:
    wf = wf_cached
else:
    wf = pit.water_frequency(cube, dates=usable, threshold=NDWI_THRESHOLD,
                             min_obs=MIN_OBS)
    pickle.dump((reference_map, cloud_pct, wf), open(CACHE_INT,'wb'))
log(f"water frequency: {int(np.isfinite(wf).sum()):,} px valid")

transition = reference_map == 0
low, high = pit.multiotsu_window(wf, transition)
poly = aoi.raster_mask(transform, crs, reference_map.shape)
inter = pit.intertidal_mask(wf, reference_map, low, high, aoi_mask=poly)
log(f"intertidal window [{low:.2f},{high:.2f}] -> "
    f"{int(inter.sum()):,} px = {inter.sum()*px_km2:.2f} km2")

# tides (real pyTMD)
tide_point = pit.water_centroid(reference_map, transform, crs)
tides = pit.TideService(model="GOT4.10", directory="./tide_models",
                        location=tide_point)
heights = tides.heights_for(aoi, usable, verbose=False)
log(f"tides: {len(heights)} dates | point {tide_point[0]:.3f},{tide_point[1]:.3f}")

times, series = tides.series(aoi, "2024-01-01", "2024-12-31", every_hours=1)
metrics = pit.coverage.coverage_metrics(np.array(list(heights.values())), series)
verdict = pit.coverage.coverage_verdict(metrics)
log(f"coverage {metrics['range_coverage']['coverage_pct']:.0f}% | "
    f"VSR {metrics['resolution']['vsr']:.3f} m | suitable={verdict['suitable']}")

# elevation on the latest epoch
eps = pit.epochs(sorted(heights), epoch_years=3)
latest = eps[-1]
log(f"epochs {[e['label'] for e in eps]} -> fitting {latest['label']} "
    f"({len(latest['dates'])} dates)")
dem = pit.fit_hsr(cube, heights, dates=latest["dates"], scale=4,
                  max_uncertainty=0.10, clip_mask=inter,
                  epoch_label=latest["label"])
dem.summary()

# derived science
hydro = pit.hydroperiod.hydroperiod(dem.mu, series)
zones, labels = pit.hydroperiod.zonation(dem.mu, series)
rows = pit.hydroperiod.zone_areas(zones, labels, transform)
log("zonation: " + " | ".join(f"{r['zone']} {r['area_km2']:.2f}km2" for r in rows))
curve = pit.hypsometry.hypsometric_curve(dem.mu, transform)
log(f"hypsometric integral {pit.hypsometry.hypsometric_integral(curve):.3f}")

# validation
pit.download_mdt_ign({**aoi.bbox, "crs": "EPSG:4326"}, "mdt5_villaviciosa_grande.tif")
lidar = pit.reproject_to_grid("mdt5_villaviciosa_grande.tif", transform, crs,
                              reference_map.shape)
comp = pit.compare_dems({"HSR": dem.mu}, lidar)
log(f"validation: {comp}")

# outputs
os.makedirs("products_villaviciosa", exist_ok=True)
dem.save("products_villaviciosa")
pit.write_geotiff("products_villaviciosa/water_frequency.tif", wf, transform,
                  crs, "float32", np.nan)
item = pit.export.stac_item("products_villaviciosa/hsr_elevation.tif", aoi=aoi,
                            datetime_utc="2025-07-01",
                            properties={"pyintertidal:method": "hsr"})
pit.export.write_stac([item], "products_villaviciosa", verbose=False)

lg = explain.OpLog("verify")
lg.record("acquire", kind="acquire", outputs=["cube"])
lg.record("reference map", kind="reduce", outputs=["reference map"])
lg.record("elevation", kind="apply", outputs=["DEM"])
fig, _ = viz.plot_dem(dem.mu, transform, crs, aoi, title="verify")
rep = pit.report.site_report("products_villaviciosa/report_verify.html", aoi,
                             cube=cube, params={"threshold": NDWI_THRESHOLD},
                             coverage=metrics, verdict=verdict,
                             validation=comp, zones=rows, oplog=lg,
                             figures={"Elevation": [fig]})
log(f"report -> {rep} ({os.path.getsize(rep)/1024:.0f} KB)")
print("PIPELINE VERIFICATION OK")
