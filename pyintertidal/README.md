# pyintertidal: the MAREA library

The Python package behind MAREA: Sentinel-2 data cubes from openEO, water
detection, tide levels, the delay of the tide inside an estuary, the
per-pixel elevation inversion and the products derived from it. Every script
in `experiments/` and every test in `tests/` imports it; nothing in it
imports `experiments/`.

The function-by-function reference of the older layers (study area, data,
water, base products, tides, HSR elevation, derived science, outputs) is
[`CATALOG.md`](CATALOG.md). The method itself is derived in
[`../docs/METHOD.md`](../docs/METHOD.md).

## The MAREA path through the package

One call, `marea.reconstruct(cube_path, out_dir)`, chains every step below
except the last; each one is also a function you can call on its own.
Emersion time is not stored: compute it afterwards from the stored z and a
continuous tide series. Two runnable examples (synthetic data; the
Villaviciosa product) are in the [repository README](../README.md#minimal-examples).

| step | function | module |
|---|---|---|
| read the cube in time blocks, in two streaming passes: the water frequency, then the NDWI of every clear look (SCL 4-7) of the intertidal pixels (water frequency between 0.10 and 0.90, over more than 30 clear looks) | `extract` | `marea.py` |
| the mouth (permanent sea touching the frame) and the distance from it along the water | `mouth_seeds`, `along_distance` | `geometry.py` |
| the acquisition time of every scene | `get_overpass_times` (or a cached `overpass_times.json`) | `overpass.py` |
| the tide at the mouth, read tau minutes earlier, for a bank of tau values | `build_bank` (EOT20 via eo-tides, the default), or a `BoundaryProvider` from `make_boundary` (model, gauges or their consensus) passed as `reconstruct(..., boundary=...)`; each study site's choice is in `experiments/benchmark/config.py` (`BOUNDARY`) | `marea.py`, `boundary.py` |
| sections of 3 km along the channel | `make_bands_fixed` | `tide_estimators.py` |
| one delay per section by profiled Bernoulli likelihood, zero at the mouth and never decreasing inland | `m2a_rasch` | `tide_estimators.py` |
| the interval of each section's delay (scene bootstrap) | `bootstrap_lags` | `tide_estimators.py` |
| the delay of each pixel: the section profile smoothed along the channel | `smooth_lag_profile` | `tide_estimators.py` |
| each pixel's elevation z and transition width sigma on the corrected tide; the quality rule applies when `min_bracket=marea.QUALITY_MIN_BRACKET` is passed, as `reconstruct` does (the function's own default, 0, keeps the older behaviour) | `invert_series` (on `elevation._fit_block`) | `marea.py`, `elevation.py` |
| the area-elevation curve with its uncertainty band | `curve_with_uncertainty` | `hypsometry.py` |
| emersion time from a continuous tide series (after `reconstruct`; not stored) | `1 - hydroperiod(z, h)` | `hydroperiod.py` |

What `reconstruct` writes (`marea.npz`, `result.json`, `figure.png`) is
described in [`../docs/products.md`](../docs/products.md). `reconstruct` does
not screen the scenes: it uses every scene with an acquisition time, so pass
only the times of the screened scenes (`stability.reference_and_clouds` and
`filter_dates`), as the production runs did.

To use the delay alone, for example with another elevation method:
`m2a_rasch` gives one delay per section, `smooth_lag_profile` the delay of
each pixel, and `ShiftBank.at(tau)` the mouth tide read that many minutes
earlier (README, first example, steps 4-5).

## Modules

**Study area and data**

| module | what it does |
|---|---|
| `aoi.py` | areas of interest as polygons: masks, tiling, GeoJSON |
| `sites.py` | registry of named study areas (`sites.get("villaviciosa")`) |
| `cube.py` | `SentinelCube`: download a Sentinel-2 L2A cube once with openEO, then stream it in time blocks |
| `scenes.py` | openEO connection, available dates, single-scene downloads |
| `overpass.py` | exact acquisition time of each scene (STAC catalogue) |
| `raster.py` | GeoTIFF read and write |
| `net.py` | network settings shared by every remote call (system certificates on Windows) |

**Water and base products**

| module | what it does |
|---|---|
| `water.py` | water detection: NDWI (default), MNDWI, AWEI and SCL behind one registry |
| `marsh.py` | flooded vegetation (salt marsh) |
| `stability.py` | the reference map (stable water, stable land, transition zone) and the cloud filter measured over the transition zone |
| `frequency.py` | water frequency and the intertidal window |
| `coverage.py` | tidal-sampling diagnostics: which part of the tidal range the satellite saw |
| `alias_table.py` | which tidal constituents a sun-synchronous archive can and cannot resolve |

**Tides and the delay inside the estuary**

| module | what it does |
|---|---|
| `tides.py` | `TideService`: tide height at every scene and continuous series |
| `tidemodels.py` | global tide models through pyTMD (EOT20, GOT, FES) and CMEMS sea level |
| `gauges.py` | tide-gauge records and their comparison with a model |
| `boundary.py` | the water level at the mouth from a model, the nearest gauges or their consensus (`make_boundary`) |
| `geometry.py` | the estuary's shape measured from the archive: mouth, along-water distance, widths |
| `tide_estimators.py` | the delay estimators of the development phase M2, among them `m2a_rasch`, the one MAREA uses |
| `boundary_correction.py` | phase M3: a phase error of the boundary judged from the mouth pixels |
| `operator.py` | phase M4: the distributed tide gauge h(s, t) = alpha(s) h_B(t - tau(s)) as one inspectable object |
| `estuary.py` | transfer of the tide into an estuary, and its check against a gauge |
| `tidecheck.py` | what an estuary does to the tide, and what a satellite can see of it |

**Elevation and derived science**

| module | what it does |
|---|---|
| `marea.py` | MAREA end to end for any cube: `extract`, `invert_series`, `build_bank`, `reconstruct` |
| `elevation.py` | the per-pixel NDWI-versus-tide fit (`_fit_block`), and the earlier HSR and step methods |
| `waterline_contour.py` | a re-implementation of ITEM/NIDEM (Sagar et al. 2017; Bishop-Taylor et al. 2019), the main competitor in the validation |
| `granadeiro2021.py` | a re-implementation of Granadeiro et al. (2021); its statistical fits run in R (`r/granadeiro_fit.R`) |
| `terrain.py` | slope, aspect, hillshade, roughness and contours of any DEM |
| `hydroperiod.py` | share of time under water (and its complement, emersion time); ecological zones |
| `hypsometry.py` | area-elevation curve and hypsometric integral |
| `morphodynamics.py` | change between two epochs, with a significance test |

**Checking and output**

| module | what it does |
|---|---|
| `validation.py` | independent references (IGN LiDAR by WCS), regridding, DEM comparison, transects |
| `elevation_validation.py` | the shared arithmetic of elevation scoring (median-centred RMSE, slope, r, bias) |
| `lag_validation.py` | a fitted delay against two water-level records |
| `rtk.py` | the only reader of the RTK-GNSS survey; it returns the development points and keeps the reserved part locked (tested by `tests/test_r1_guard.py`) |
| `simulator.py` | a calibrated twin of the archive: the matched null and planted mechanisms |
| `seal.py` | sealed predictions: a hash registry written before the evidence is read |
| `viz.py` | maps, profiles and tide plots; every function returns `(fig, ax)` |
| `explain/` | diagrams of the data cube and of each operation in the style of the openEO documentation, and an opt-in log of what the pipeline did |
| `report.py` | a self-contained HTML report per site |
| `export.py` | Cloud-Optimized GeoTIFF and STAC metadata |
| `mosaic.py` | merging tiles of a regional campaign |
| `legacy/` | superseded server-side methods, kept runnable for the comparisons |
| `r/granadeiro_fit.R` | the R side of `granadeiro2021.py` (nplr, lmodel2, mgcv) |

## Design choices

* **No hidden pipeline.** Each function performs one scientific step, takes
  explicit parameters and returns plain arrays; `reconstruct` is only the
  chain of the steps above.
* **One download, then streaming.** Imagery is fetched once per area as a
  cached cube; every product streams it in time blocks, so long archives cost
  time, not memory.
* **Polygons first.** Study areas are polygons; bounding boxes are a
  download detail, and products are clipped to the polygon.
* **One implementation per idea.** Where several versions of an algorithm
  existed, the package keeps one; `legacy/` keeps the superseded ones that the
  comparisons still need.

## Install

`import pyintertidal` needs numpy, scipy, pandas, xarray, rasterio, shapely,
matplotlib and requests. `marea.reconstruct` also needs PyYAML (it reads
`configs/m2.yaml`), pyproj, netCDF4 or h5netcdf (to open the cube) and, for
the EOT20 tide, pyTMD and eo-tides; joblib, if present, parallelises the
bootstrap. Downloads add openeo. Other modules use geopandas, scikit-image,
zarr, plotly (3-D views), contextily (basemaps), psutil, truststore (SSL on
Windows) and copernicusmarine (CMEMS).
The pinned environment is [`../requirements-cluster.txt`](../requirements-cluster.txt).

## Conventions worth knowing

* **Vertical datum.** Elevations are on the vertical scale of the tide
  series they were inverted on (EOT20 heights, about mean sea level); a
  comparison with a survey removes the median offset first and reports it
  separately as the bias.
* **Units.** Heights and elevations in metres; delays in minutes; distances
  along the channel in kilometres; times in UTC.
* **Index and threshold.** NDWI = (B03 - B08) / (B03 + B08), wet above 0, on
  looks whose SCL class is 4-7 (`marea.CLEAR`).
* **Sampling.** A sun-synchronous satellite sees only part of the tide; run
  `coverage` before trusting an elevation model, and report it with the
  results.
* **Language.** Some keys of stored products keep the Spanish names they
  were recorded with (`result.json`: `sitio`, `estado`, `n_escenas`,
  `centros_km`, `tau_usado_min`, ...); [`../docs/products.md`](../docs/products.md)
  translates them.

The history of the methods, including the earlier HSR estimator and the
comparisons that retired some of its numbers, is in
[`../docs/INVESTIGATION_MAP.md`](../docs/INVESTIGATION_MAP.md).
