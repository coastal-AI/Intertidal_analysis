# -*- coding: utf-8 -*-
"""Build the pyintertidal use-case notebook."""
import nbformat as nbf

OUT = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\intertidal_topography_villaviciosa.ipynb"
C = []
md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
co = lambda t: C.append(nbf.v4.new_code_cell(t))

md("""# Intertidal topography of the Ría de Villaviciosa

**A complete worked study with `pyintertidal`** — from raw Sentinel-2 imagery
to an elevation model, its ecological interpretation and an independent
validation.

This notebook is also the reference example of the library. Every step is an
explicit function call with its parameters written out in variables, and after
each operation a diagram shows what happened to the datacube. Nothing is
hidden behind a "run everything" call: if you can read this notebook, you know
exactly how the result was produced.

**Study site.** The Ría de Villaviciosa (Asturias, N Spain) is a mesotidal
estuary with sandy flats, a meandering main channel and salt marsh in its
upper reaches. It is our reference site: an IGN LiDAR survey covers it, and we
have a field transect at Misiego.

**What we produce**

| Product | What it answers |
|---|---|
| Reference map | where the coast is stable and where it is transitional |
| Water frequency | how often each pixel is wet |
| Intertidal mask | where the intertidal zone is, and how large |
| Elevation + uncertainty | how high each pixel stands, and how sure we are |
| Sub-pixel relief | where there is structure finer than 10 m |
| Hydroperiod & zonation | how long each pixel is submerged → habitat |
| Hypsometry | the morphological signature of the estuary |
| Validation | how it compares with airborne LiDAR |
""")

md("## 0 · Setup\n\nOne import for the library, one connection to Copernicus. "
   "The connection is only used if the datacube is not already cached on disk.")
co("""import numpy as np
import matplotlib.pyplot as plt

import pyintertidal as pit
from pyintertidal import SentinelCube, TideService, explain, viz

%config InlineBackend.figure_formats = ['png']
plt.rcParams['figure.dpi'] = 150

print("pyintertidal", pit.__version__)

# `connect` also fixes SSL certificate handling on Windows before
# authenticating with OpenID Connect (a browser opens the first time).
conn = pit.scenes.connect()""")

md("""## 1 · Study area

The area of interest is a **polygon**, not a bounding box. The bounding box is
used internally to download a (rectangular) datacube, but every product and
every figure is clipped back to the polygon — which is what lets contiguous
tiles of a regional campaign fit together without overlaps.

The site comes from the registry so the same coordinates are used by this
notebook, the validation and the field campaign.""")
co("""aoi = pit.sites.get("villaviciosa")

print(aoi)
print(f"  bounding box used for the download: "
      f"{ {k: round(v, 3) for k, v in aoi.bbox.items()} }")
print(f"  tide prediction point (centroid):   "
      f"{aoi.centroid[0]:.4f}, {aoi.centroid[1]:.4f}")

# How this AOI would be split for a regional campaign (not used further here):
tiles = aoi.tile(cell_km=5)
print(f"  as production tiles of 5 km: {len(tiles)} cells covering "
      f"{sum(t.area_km2 for t in tiles):.1f} km²")""")

md("""## 2 · The datacube

All the imagery this study needs is downloaded **once** and cached as a
netCDF cube. Every later step streams that file in small time blocks, so
memory stays bounded no matter how long the archive is — a decade of imagery
over 80 km² would be ≈ 8 GB if loaded whole, and we never load it whole.

The bands come from the water detector: NDWI needs B03 and B08 (both native
10 m, which is why NDWI has true 10 m resolution), plus SCL for the cloud
mask.""")
co("""# ── Parameters ──────────────────────────────────────────────────────────
TIME_EXTENT   = ("2016-01-01", "2025-12-31")   # ten years of Sentinel-2
WATER         = "ndwi"                          # ndwi | mndwi | awei | scl
RESOLUTION_M  = 10                              # native for B03/B08
CACHE         = "ndwi_cube_villaviciosa_grande_10y.nc"

# ── Acquisition ─────────────────────────────────────────────────────────
cube = SentinelCube(aoi, TIME_EXTENT, water=WATER,
                    cache_path=CACHE, resolution=RESOLUTION_M)
cube.ensure(conn)          # downloads only if the cache does not exist

transform, crs = cube.grid
print(f"{len(cube.dates)} scenes · grid {cube.shape[0]}×{cube.shape[1]} "
      f"@ {abs(transform.a):g} m · {crs}")
print(f"first {cube.dates[0]} · last {cube.dates[-1]}")""")

md("What is actually inside that cube — bands as rows, time as columns, "
   "filled with real (heavily downsampled) imagery so the clouds and tide "
   "states are visible:")
co("""explain.describe_cube(cube, with_data=True, max_columns=6)""")

co("""explain.summarize(cube)""")

md("""We record the rest of the analysis so that, at the end, the notebook can
draw the process graph that produced every number in it.""")
co("""log = explain.OpLog("Villaviciosa 2016–2025")
log.record("acquire datacube", kind="acquire",
           params={"water": WATER, "resolution_m": RESOLUTION_M},
           outputs=["cube"],
           note=f"{len(cube.dates)} scenes, cached once")""")

md("""## 3 · Water detection threshold

NDWI separates water from land, but the optimal cut is site-dependent:
turbid estuarine water sits lower than the textbook 0.0. Rather than guessing,
we let the data decide with Otsu's method over the clearest scenes, and then
look at the number before adopting it.""")
co("""# ── Parameter (auto-calibrated, then pinned explicitly) ─────────────────
otsu = pit.water.otsu_threshold(
    np.concatenate([
        pit.water.index_block(WATER, {b: blocks[b][:3] for b in cube.bands})[
            np.isfinite(pit.water.index_block(WATER,
                        {b: blocks[b][:3] for b in cube.bands}))]
        for _, blocks in [next(iter(cube.stream(chunk=8)))]
    ])
)
print(f"Otsu suggests a threshold of {otsu:+.3f}")

NDWI_THRESHOLD = 0.0        # adopted: Otsu agrees within a few hundredths
print(f"adopted NDWI threshold: {NDWI_THRESHOLD:+.3f}  "
      f"(pixels with NDWI above this count as water)")""")

md("""## 4 · Coastal stability and cloud screening

Two products from two streaming passes:

1. **Reference map** — each pixel voted wet/dry across the CLEAN scenes and
   labelled stable water, stable land or *transition*. The transition class is
   the intertidal candidate zone.
2. **Per-date cloudiness** — measured globally for clear scenes and, for
   cloudy ones, **inside the transition zone only**. That distinction is what
   rescues dates: a scene can be 80 % cloudy overall and perfectly clear over
   the estuary.""")
co("""# ── Parameters ──────────────────────────────────────────────────────────
CLEAN_SCENE_MAX_BAD  = 0.05   # a scene votes in the reference map if <5% cloud
STABLE_THRESHOLD     = 0.95   # 95% of votes agree → stable water/land
TRANSITION_BUFFER_PX = 10     # safety halo around the transition zone
BAD_CLASSES          = pit.water.BAD_CLASSES     # SCL cloud/shadow classes

# ── Reference map + per-date cloudiness ─────────────────────────────────
reference_map, cloud_pct = pit.reference_and_clouds(
    cube,
    threshold=NDWI_THRESHOLD,
    bad_classes=BAD_CLASSES,
    clean_scene_max_bad=CLEAN_SCENE_MAX_BAD,
    stable_threshold=STABLE_THRESHOLD,
    transition_buffer_px=TRANSITION_BUFFER_PX,
)

px_km2 = abs(transform.a * transform.e) / 1e6
for code, name in [(0, "transition"), (1, "stable water"), (2, "stable land")]:
    n = int((reference_map == code).sum())
    print(f"  class {code} ({name:13s}): {n:>8,} px  {n * px_km2:6.2f} km²")

log.record("reference map + cloud stats", kind="reduce",
           params={"stable_threshold": STABLE_THRESHOLD,
                   "buffer_px": TRANSITION_BUFFER_PX},
           inputs=["cube"], outputs=["reference map", "cloud %"])""")

co("""viz.plot_reference_map(reference_map, transform, crs, aoi);""")

md("Now the date filter. We keep the dates that are clear **over the transition "
   "zone**, and measure how many that rescues compared with a whole-scene filter:")
co("""# ── Parameter ───────────────────────────────────────────────────────────
CLOUD_THRESHOLD = 0.10        # ≤10% cloud over the transition zone

usable_dates = pit.filter_dates(cloud_pct, cloud_threshold=CLOUD_THRESHOLD)
gain = pit.date_recovery_gain(cloud_pct,
                              clean_scene_max_bad=CLEAN_SCENE_MAX_BAD,
                              cloud_threshold=CLOUD_THRESHOLD)

log.record("filter dates", kind="filter",
           params={"cloud_threshold": CLOUD_THRESHOLD},
           outputs=[f"{len(usable_dates)} dates"],
           note=f"+{gain['gain_relative_pct']:.0f}% vs a global filter")""")

md("What that filter did to the cube — same dimensions, fewer labels:")
co("""explain.draw_operation(
    "filter dates · clouds ≤ 10% over the transition zone",
    kind="filter",
    before={"bands": cube.bands, "n_dates": len(cube.dates), "shape": cube.shape},
    after={"bands": cube.bands, "n_dates": len(usable_dates), "shape": cube.shape},
    note=f"{len(cube.dates):,} → {len(usable_dates):,} dates",
    params={"cloud_threshold": CLOUD_THRESHOLD})""")

md("""## 5 · Water frequency and the intertidal zone

Water frequency is the fraction of clear observations in which each pixel was
wet. Stable water sits near 1, stable land near 0, and the intertidal zone
spans the middle — ordered by elevation, because higher flats flood less
often.

`min_obs` guards against pixels with too little evidence: a pixel seen twice
can report a frequency of 1.0 and mean nothing.""")
co("""# ── Parameters ──────────────────────────────────────────────────────────
MIN_OBS = max(8, round(0.05 * len(usable_dates)))   # ≥8, or 5% of the archive
print(f"a pixel needs at least {MIN_OBS} clear observations")

# ── Water frequency (streamed over the usable dates) ────────────────────
water_freq = pit.water_frequency(
    cube, dates=usable_dates, threshold=NDWI_THRESHOLD, min_obs=MIN_OBS)

valid = np.isfinite(water_freq)
print(f"water frequency computed for {valid.sum():,} px "
      f"({100 * valid.mean():.1f}% of the grid)")

log.record("water frequency", kind="reduce",
           params={"min_obs": MIN_OBS, "threshold": NDWI_THRESHOLD},
           inputs=[f"{len(usable_dates)} dates"], outputs=["water frequency"])""")

md("The time dimension has collapsed: hundreds of dates became one map.")
co("""explain.draw_operation(
    "water frequency · fraction of clear observations wet",
    kind="reduce",
    before={"bands": cube.bands, "n_dates": len(usable_dates), "shape": cube.shape},
    after={"bands": ["water frequency"], "n_dates": 1, "shape": cube.shape},
    note=f"{len(usable_dates)} dates → 1 map",
    params={"min_obs": MIN_OBS})""")

co("""viz.plot_water_frequency(water_freq, transform, crs, aoi);""")

md("""The intertidal zone is the transition class with a water frequency between
two bounds. The bounds can be pinned by hand, or found from the data: inside
the transition zone the histogram mixes three populations (almost never wet,
truly intertidal, almost always wet) and a 3-class Otsu split finds the two
valleys between them.""")
co("""# ── Parameters ──────────────────────────────────────────────────────────
transition = reference_map == 0
WF_LOW, WF_HIGH = pit.multiotsu_window(water_freq, transition)
print(f"multi-Otsu suggests the intertidal window "
      f"[{WF_LOW:.2f}, {WF_HIGH:.2f}]")

# ── Intertidal mask (also clipped to the study polygon) ─────────────────
polygon_mask   = aoi.raster_mask(transform, crs, reference_map.shape)
intertidal     = pit.intertidal_mask(water_freq, reference_map,
                                     WF_LOW, WF_HIGH, aoi_mask=polygon_mask)

area_km2 = intertidal.sum() * px_km2
print(f"intertidal zone: {int(intertidal.sum()):,} px = {area_km2:.2f} km²")

log.record("intertidal mask", kind="mask",
           params={"wf_low": round(WF_LOW, 3), "wf_high": round(WF_HIGH, 3)},
           outputs=["intertidal mask"], note=f"{area_km2:.2f} km²")""")

co("""viz.plot_intertidal(intertidal, transform, crs, aoi,
                    title=f"Intertidal zone · {aoi.name} · {area_km2:.2f} km²");""")

md("""## 6 · Tides

Elevation methods need the water level at the moment each image was taken, so
we pair every usable date with its real acquisition time (from the Copernicus
catalogue) and predict the tide there with a global harmonic model.

Two refinements worth noting. The prediction point defaults to the polygon
centroid, which in a curved estuary can fall on land where the model returns
nothing; `water_centroid` instead picks the middle of the main channel. And
the continuous series (not the satellite dates) is what later drives the
sampling diagnostics and the hydroperiod.""")
co("""# ── Parameters ──────────────────────────────────────────────────────────
TIDE_MODEL = "GOT4.10"
TIDE_DIR   = "./tide_models"

tide_point = pit.water_centroid(reference_map, transform, crs)
print(f"tide prediction point (mid-channel): "
      f"{tide_point[0]:.4f}, {tide_point[1]:.4f}")

tides = TideService(model=TIDE_MODEL, directory=TIDE_DIR, location=tide_point)

# Tide height at each satellite acquisition …
tide_heights = tides.heights_for(aoi, usable_dates)

# … and the continuous series that represents the real tidal regime.
series_times, series_heights = tides.series(aoi, "2024-01-01", "2024-12-31",
                                            every_hours=1)
print(f"reference series: {len(series_times):,} hourly predictions, "
      f"range [{np.nanmin(series_heights):+.2f}, "
      f"{np.nanmax(series_heights):+.2f}] m")

log.record("tide heights", kind="apply",
           params={"model": TIDE_MODEL,
                   "point": f"{tide_point[0]:.3f},{tide_point[1]:.3f}"},
           outputs=[f"{len(tide_heights)} tide heights"])""")

md("""## 7 · Is this site mappable?

A satellite in a sun-synchronous orbit does not sample the tide evenly, and no
elevation method can reconstruct a tide level that was never imaged. These
diagnostics put a ceiling on what is achievable here — and belong in the
results, not just in the planning.""")
co("""sampled = np.array(list(tide_heights.values()))

metrics = pit.coverage.coverage_metrics(sampled, series_heights)
verdict = pit.coverage.coverage_verdict(metrics)
pit.coverage.print_coverage_report(metrics, verdict,
                                   title=f"Tidal sampling · {aoi.name}")""")

co("""import pandas as pd

sampled_times = [pd.Timestamp(d) for d in tide_heights]
fig, axes = plt.subplots(1, 2, figsize=(17, 5),
                         gridspec_kw={"width_ratios": [3, 1]})
viz.plot_tide_series(series_times, series_heights,
                     title="Modelled tide (2024) ", ax=axes[0])
viz.plot_tide_distribution(series_heights, sampled,
                           title="What the satellite sampled", ax=axes[1])
fig.tight_layout();""")

md("""## 8 · Elevation

Our estimator, **HSR**, uses the physics of mixed pixels. A 10 m pixel on the
waterline is *partially* flooded, so its NDWI is the mixture of its wet and dry
parts, and as the tide rises the wet fraction traces the pixel's internal
hypsometry. Fitting that sigmoid per pixel gives three things at once:

* **μ** — the elevation (the whole curve is used, so no threshold crossing);
* **σ** — the relief *inside* the pixel, a sub-pixel product;
* **σ(μ)** — a Cramér–Rao uncertainty, which doubles as quality control.

Fitted per **epoch**, not per decade: intertidal morphology migrates, and
fitting ten years at once blurs the transition (measured on this site: 87 % of
pixels saturate the fitting grid at 10 years versus 12 % at 3).""")
co("""# ── Parameters ──────────────────────────────────────────────────────────
EPOCH_YEARS     = 3       # morphology is stable enough within ~3 years
SCALE           = 4       # super-resolution factor: 10 m → 2.5 m
MAX_UNCERTAINTY = 0.10    # drop pixels whose σ(μ) exceeds 10 cm
SIGMA_FLOOR     = 0.05    # noise floor of σ, not injected as relief

epoch_list = pit.epochs(sorted(tide_heights), epoch_years=EPOCH_YEARS)
for e in epoch_list:
    print(f"  epoch {e['label']}: {len(e['dates'])} dates")

latest = epoch_list[-1]
print(f"\\nfitting the most recent epoch: {latest['label']}")

# ── HSR fit (local: no cloud job) ───────────────────────────────────────
dem = pit.fit_hsr(
    cube, tide_heights,
    dates=latest["dates"],
    scale=SCALE,
    sigma_floor=SIGMA_FLOOR,
    max_uncertainty=MAX_UNCERTAINTY,
    clip_mask=intertidal,
    epoch_label=latest["label"],
)
dem.summary()

log.record("HSR elevation", kind="apply",
           params={"epoch": latest["label"], "scale": SCALE,
                   "max_uncertainty": MAX_UNCERTAINTY},
           inputs=[f"{len(latest['dates'])} dates", "tide heights"],
           outputs=["elevation", "sub-pixel relief", "uncertainty"])""")

co("""viz.plot_dem(dem.mu, transform, crs, aoi,
             title=f"Intertidal elevation · {aoi.name} · epoch {dem.epoch}");""")

co("""fig, axes = plt.subplots(1, 2, figsize=(19, 8))
viz.plot_uncertainty(dem.sigma_mu, transform, crs, aoi,
                     title="Elevation uncertainty (Cramér–Rao)", ax=axes[0])
viz.plot_subpixel_relief(dem.sigma, transform, crs, aoi,
                         title="Sub-pixel relief σ", ax=axes[1])
fig.tight_layout();""")

md("""### The published baseline, for comparison

The DEA-style **step** method reads the tide at which the rolling median of
NDWI crosses from dry to wet. It is more selective than HSR — it needs a clean
monotonic transition — and provides no uncertainty, but it is the reference
method of the literature and runs from the same cube.""")
co("""dem_step = pit.fit_step(
    cube, tide_heights, dates=latest["dates"],
    threshold=NDWI_THRESHOLD, min_obs=5,
    clip_mask=intertidal, epoch_label=latest["label"],
)
dem_step.summary()

print(f"\\ncoverage: HSR {int(np.isfinite(dem.mu).sum()):,} px vs "
      f"step {int(np.isfinite(dem_step.mu).sum()):,} px")""")

md("""## 9 · Validation against airborne LiDAR

The IGN 5 m LiDAR survey is an independent measurement of the same surface.
One caution before reading the numbers: our elevations are referenced to the
tide-model datum and LiDAR to an orthometric one, so their offset appears as a
**bias** that says nothing about method skill. RMSE and MAE are therefore
computed after removing it, and the bias is reported separately.""")
co("""MDT_PATH = "mdt5_villaviciosa_grande.tif"

pit.download_mdt_ign({**aoi.bbox, "crs": "EPSG:4326"}, MDT_PATH)
lidar = pit.reproject_to_grid(MDT_PATH, transform, crs, reference_map.shape)
print(f"LiDAR reprojected onto the analysis grid: "
      f"{np.isfinite(lidar).mean() * 100:.0f}% coverage")

rows = pit.compare_dems({"HSR": dem.mu, "step (DEA)": dem_step.mu}, lidar)
pd.DataFrame(rows)""")

md("""A second, independent check that needs no elevation model of our own:
water frequency must be NEGATIVELY correlated with elevation, because higher
ground floods less often. This is how the water indices were compared before
any DEM existed.""")
co("""pit.validation.wf_vs_elevation(water_freq, lidar, mask=intertidal)""")

md("""### Field transect — Misiego

The profile we survey on the ground, with both satellite methods and the LiDAR
plotted against it. Bring this figure to the field: the x axis is metres walked
from the start point.""")
co("""TRANSECT_START = (-5.3797, 43.5107)     # (lon, lat)
TRANSECT_END   = (-5.3835, 43.5065)

profiles = {}
for label, raster in [("HSR", dem.mu), ("step (DEA)", dem_step.mu),
                      ("IGN LiDAR", lidar)]:
    distance, values = pit.transect_profile(raster, transform, crs,
                                            TRANSECT_START, TRANSECT_END, n=200)
    profiles[label] = values

viz.plot_transect(distance, profiles,
                  title="Misiego transect · predicted profiles");""")

md("""## 10 · From topography to habitat

Elevation is a means, not an end. What organises intertidal life is the
**hydroperiod**: the fraction of time a spot is submerged. Computing it from
the continuous tide series (not from the satellite dates, which are unevenly
sampled) turns our DEM into an ecological map.""")
co("""hydro = pit.hydroperiod.hydroperiod(dem.mu, series_heights)
zones, zone_labels = pit.hydroperiod.zonation(dem.mu, series_heights)
zone_table = pit.hydroperiod.zone_areas(zones, zone_labels, transform)

print(f"hydroperiod: {np.nanmin(hydro):.0%} – {np.nanmax(hydro):.0%} "
      f"of the time submerged (median {np.nanmedian(hydro):.0%})\\n")
pd.DataFrame(zone_table)""")

co("""fig, axes = plt.subplots(1, 2, figsize=(19, 8))
viz.plot_map(hydro * 100, transform, crs, aoi, cmap="YlGnBu",
             vmin=0, vmax=100, robust=False,
             title="Hydroperiod", cbar_label="Time submerged (%)", ax=axes[0])
viz.plot_map(zones.astype(float), transform, crs, aoi,
             classes={c: (zone_labels[c], col) for c, col in
                      zip(sorted(zone_labels)[1:],
                          ["#08306b", "#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"])},
             title="Ecological zonation", ax=axes[1])
fig.tight_layout();""")

md("""### Morphological signature

The hypsometric curve — how much area lies below each elevation — is the
compact fingerprint of the estuary's shape, and the natural quantity to
compare between epochs or between sites.""")
co("""curve = pit.hypsometry.hypsometric_curve(dem.mu, transform)
summary = pit.hypsometry.describe(curve, aoi.name)

pit.hypsometry.plot_curves({aoi.name: curve},
                           title=f"Hypsometric curve · {aoi.name}");""")

md("""### Change between epochs

With two epochs fitted, the difference is a change map. The rule that keeps it
honest: **never subtract two DEMs without their uncertainties**. Only changes
larger than the propagated uncertainty are reported.""")
co("""if len(epoch_list) >= 2:
    previous = epoch_list[-2]
    print(f"comparing {previous['label']} → {latest['label']}")

    dem_prev = pit.fit_hsr(cube, tide_heights, dates=previous["dates"],
                           scale=SCALE, max_uncertainty=MAX_UNCERTAINTY,
                           clip_mask=intertidal, epoch_label=previous["label"],
                           superresolve=False)

    change = pit.morphodynamics.dem_difference(
        dem_prev.mu, dem.mu, dem_prev.sigma_mu, dem.sigma_mu, confidence=1.96)
    budget = pit.morphodynamics.summarize(change, transform,
                                          f"{previous['label']} → {latest['label']}")

    viz.plot_map(np.where(change["significant"], change["change"], np.nan),
                 transform, crs, aoi, cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                 robust=False, title="Significant elevation change",
                 cbar_label="Δ elevation (m)")
else:
    print("only one epoch available — skip the change analysis")""")

md("""## 11 · Provenance and outputs

The process graph of everything above: which operations ran, in which order,
with which parameters. This is the figure that makes the study auditable by
somebody who did not run it.""")
co("""explain.draw_graph(log)""")

co("""log.table()""")

md("Finally, write the products, package them with STAC metadata that carries "
   "the parameters, and build a single self-contained HTML report.")
co("""OUT_DIR = "products_villaviciosa"

# Rasters (10 m grid, clipped to the polygon) + the super-resolved DEM
dem.save(OUT_DIR)
pit.write_geotiff(f"{OUT_DIR}/water_frequency.tif", water_freq,
                  transform, crs, "float32", np.nan)
pit.write_geotiff(f"{OUT_DIR}/intertidal_mask.tif",
                  intertidal.astype("uint8"), transform, crs, "uint8", None)
pit.write_geotiff(f"{OUT_DIR}/hydroperiod.tif", hydro,
                  transform, crs, "float32", np.nan)

# Provenance travels with the data
item = pit.export.stac_item(
    f"{OUT_DIR}/hsr_elevation.tif", aoi=aoi,
    datetime_utc=f"{latest['label'][-4:]}-07-01",
    properties={"pyintertidal:method": "hsr",
                "pyintertidal:epoch": latest["label"],
                "pyintertidal:ndwi_threshold": NDWI_THRESHOLD,
                "pyintertidal:cloud_threshold": CLOUD_THRESHOLD,
                "pyintertidal:wf_window": [round(WF_LOW, 3), round(WF_HIGH, 3)],
                "pyintertidal:min_obs": MIN_OBS,
                "pyintertidal:tide_model": TIDE_MODEL},
    assets={"uncertainty": f"{OUT_DIR}/hsr_uncertainty.tif",
            "subpixel_relief": f"{OUT_DIR}/hsr_sigma.tif"})
pit.export.write_stac([item], OUT_DIR,
                      description=f"Intertidal products · {aoi.name}")""")

co("""report_path = pit.report.site_report(
    f"{OUT_DIR}/report_villaviciosa.html",
    aoi, cube=cube,
    params={"period": f"{TIME_EXTENT[0]} → {TIME_EXTENT[1]}",
            "water detector": WATER, "NDWI threshold": NDWI_THRESHOLD,
            "cloud threshold": CLOUD_THRESHOLD, "min observations": MIN_OBS,
            "intertidal window": f"[{WF_LOW:.2f}, {WF_HIGH:.2f}]",
            "epoch": latest["label"], "tide model": TIDE_MODEL},
    coverage=metrics, verdict=verdict,
    validation=rows, zones=zone_table, oplog=log,
    notes="Satellite-derived intertidal topography, produced with pyintertidal.")

print("report written to", report_path)""")

md("""## Notes

* **Scaling out.** To map another area, change the polygon in section 1 —
  nothing else. Each tile caches its own cube and can run in a separate
  process; :mod:`pyintertidal.mosaic` merges the results and checks the seams.
* **Method comparison.** The superseded elevation methods (bracketing,
  waterline isolines, and the server-side variants) remain runnable in
  `pyintertidal.legacy`, which is what the paper's comparison section uses.
* **Reproducibility.** Every parameter in this notebook is a named variable,
  and the same values are stored in the STAC item and the HTML report, so a
  result found later can always be traced back to the run that made it.
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"},
               "language_info": {"name": "python"}}
nbf.validate(nb)
_, nb = nbf.validator.normalize(nb)
nbf.write(nb, OUT)
print("written:", OUT, "|", len(C), "cells")
