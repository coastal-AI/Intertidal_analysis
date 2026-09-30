# pyintertidal — function catalogue

One capability, one implementation. Where the legacy package had several
implementations of the same algorithm, the core keeps the one that measured
best and `legacy/` preserves the superseded ones, runnable, for the paper.
The end of this document lists what was deliberately **not** ported, and why.

---

## 1 · Study area

### `aoi.py` — the AOI is a polygon
| Function | What it does |
|---|---|
| `AOI.from_polygon / from_bbox / from_dms` | build the AOI from vertices, a box, or DMS coordinates |
| `AOI.bbox` | download box (an internal detail: cubes are rectangular) |
| `AOI.centroid`, `AOI.area_km2` | default tide point, surface area |
| `AOI.raster_mask(transform, crs, shape)` | polygon mask → clips products and figures |
| `AOI.tile(cell_km, overlap_km)` | split the AOI into contiguous cells for parallel production |
| `AOI.to_geojson()` | export the polygon (inspect the tiling in QGIS) |
| `parse_dms()` | `43°30'24"N 5°25'32"W` → decimal degrees |
| `as_aoi()` | accept AOI / shapely / list / dict in any function |

### `sites.py` — named-site registry
| Function | What it does |
|---|---|
| `get("villaviciosa")` | ready-made AOI (also Santoña, Foz, Urdaibai, …) |
| `register()`, `describe()`, `list_sites()` | add / inspect sites |

---

## 2 · Data

### `cube.py` — the datacube (download once, then stream)
| Function | What it does |
|---|---|
| `SentinelCube(aoi, period, water=...)` | cached cube; bands depend on the water detector |
| `.ensure(conn)` | downloads **only if not cached** (one batch job) |
| `.dates / .grid / .shape` | metadata without loading pixels |
| `.stream(chunk)` | iterator over temporal blocks (bounded RAM) |
| `.thumbnails(n, size)` | real thumbnails for the `explain` diagrams |
| `open_cube`, `auto_chunk`, `grid_from_dataset`, `cube_dates` | netCDF primitives |

### `raster.py` — GeoTIFF (the single definition of IO)
| Function | What it does |
|---|---|
| `write_geotiff`, `read_band`, `grid_of` | products |
| `is_valid_tif`, `read_rgb`, `read_scl`, `scene_coverage` | per-date scenes |
| `percentile_normalize`, `tifs_to_png` | visualisation / ML datasets |

### `scenes.py` — individual scenes (inspection and QA)
| Function | What it does |
|---|---|
| `connect()` | openEO connection (+ Windows SSL fix) |
| `available_dates()` | available dates (metadata only, free) |
| `download_rgb / download_scl / download_indices` | one date |
| `download_scenes()` | **many dates in ONE job** |

---

## 3 · Water detection

### `water.py`
| Function | What it does |
|---|---|
| `SCL_CLASSES` | class table + colours (incl. 12 = flooded marsh) |
| `BAD_CLASSES`, `CLEAR_CLASSES`, `SCL_WATER_CLASSES` | shared constants |
| `bands_for(water)` | which bands to download for each detector |
| `index_block()` | NDWI / MNDWI / AWEI per block |
| `water_land_masks()` | wet/dry/clear masks (clouds ALWAYS via SCL) |
| `scene_quality()` | scene quality (bad fraction + histogram) |
| `otsu_threshold()` | optional threshold calibration (the notebook decides) |

### `marsh.py` — flooded vegetation
| Function | What it does |
|---|---|
| `detect_flooded_vegetation()` | vegetation + high water index = flooded marsh |
| `correct_scl()` | relabel to class 12 |
| `marsh_frequency()` | how often a pixel is marsh (habitat extent) |

---

## 4 · Base products

### `stability.py`
| Function | What it does |
|---|---|
| `reference_and_clouds()` | 0/1/2 reference map + cloud % per date (two streaming passes) |
| `filter_dates()` | valid dates under a threshold |
| `date_recovery_gain()` | **how many dates the transition filter rescues** vs a global filter |
| `transition_geometry()` | vectorise the transition zone (GeoDataFrame) |

### `frequency.py`
| Function | What it does |
|---|---|
| `water_frequency()` | fraction of clear observations that are wet |
| `multiotsu_window()` | [low, high] window by multi-Otsu (optional) |
| `intertidal_mask()` | transition ∩ window ∩ polygon |

---

## 5 · Tides

### `tides.py`
| Function | What it does |
|---|---|
| `TideService(model, directory, location, backend)` | model and point are choices, not constants |
| `.heights_for(aoi, dates)` | tide at each scene's overpass instant |
| `.series(aoi, start, end)` | continuous series (for coverage and hydroperiod) |
| `water_centroid()` | tide point **over water** (better than the centroid) |

### `tidemodels.py`
| Class | What it does |
|---|---|
| `PyTMDTideModel` | GOT4.10 / FES2014 / FES2022; ocean-cell search; vectorised batch |
| `CopernicusTideModel` | CMEMS sea level (includes the meteorological tide) |

### `overpass.py`
| Function | What it does |
|---|---|
| `get_overpass_times()` | real UTC time of every scene (from the product name) |
| `mean_overpass_hour()` | mean overpass hour |

### `gauges.py` — tide-model validation
| Function | What it does |
|---|---|
| `load_gauge()` | load PORTUS/REDMAR records (CSV/Excel) |
| `compare_with_model()` | pair observed vs modelled |
| `gauge_metrics()` | datum offset, RMSE, MAE, correlation |
| `compare_models()` | several models against the same gauge |

### `coverage.py` — is this site mappable at all?
| Function | What it does |
|---|---|
| `tidal_range_coverage()` | % of the tidal range actually sampled |
| `ks_uniformity()` | KS test against uniform sampling |
| `shannon_entropy()` | normalised entropy + empty bins |
| `dispersion_index()` | VMR: regular / random / clustered |
| `gap_statistics()` | where the largest vertical gap sits |
| `vertical_sampling_resolution()` | VSR = effective vertical resolution |
| `representativity()` | % of quantiles sampled |
| `coverage_metrics()`, `coverage_verdict()`, `print_coverage_report()` | the set + verdict + report |

---

## 6 · Elevation

### `elevation.py`
| Function | What it does |
|---|---|
| `epochs()` | split the archive into epochs (anchored at the end) |
| `fit_hsr()` | **HSR** (our method): μ + sub-pixel σ + CR uncertainty + fine DEM |
| `fit_step()` | the step method (DEA), local, no server job |
| `hsr_stage1()`, `hsr_stage2()` | the mathematical core (fit + super-resolution) |
| `ElevationResult` | `.clip_to()`, `.save()`, `.summary()`, `.transform_fine` |

### `terrain.py` — derivatives of any DEM
| Function | What it does |
|---|---|
| `slope`, `aspect`, `hillshade` | NaN-aware terrain derivatives |
| `roughness` | inter-pixel roughness (contrast with HSR's sub-pixel σ) |
| `contours` | isolines as a GeoDataFrame |

---

## 7 · Derived science

### `hydroperiod.py` — the ecological driver
| Function | What it does |
|---|---|
| `hydroperiod()` | fraction of time submerged, per pixel |
| `exposure_time()` | hours/year of air exposure (desiccation stress) |
| `zonation()` | ecological zones (subtidal → supratidal) |
| `zone_areas()` | km² and % per zone |
| `inundation_curve()` | flooded area vs tide height |

### `hypsometry.py` — morphological signature
| Function | What it does |
|---|---|
| `hypsometric_curve()` | cumulative area vs elevation |
| `hypsometric_integral()` | the shape in one number (convex/concave) |
| `describe()`, `compare_curves()`, `plot_curves()` | interpretation and comparison |

### `morphodynamics.py` — change between epochs
| Function | What it does |
|---|---|
| `dem_difference()` | difference with a **significance test** from propagated uncertainty |
| `sediment_budget()` | erosion/accretion volumes and regime |
| `change_by_zone()` | balance per ecological zone |
| `summarize()` | readable report |

---

## 8 · Validation

### `validation.py`
| Function | What it does |
|---|---|
| `download_mdt_ign()` | IGN 5 m LiDAR via WCS |
| `reproject_to_grid()` | any reference onto the analysis grid |
| `compare_dems()` | multi-DEM metrics (datum bias kept separate) |
| `transect_profile()` | GPS-transect profile (**field campaigns**) |
| `wf_vs_elevation()` | WF↔elevation correlation (compare indices without a DEM) |
| `mask_agreement()` | IoU between methods' masks |
| `point_sampling_error()` | point-vs-pixel decomposition of validation error |

---

## 9 · Visualisation and explanation

### `viz.py` — composable, everything returns `(fig, ax)`
| Function | What it does |
|---|---|
| `plot_map()` | **primitive**: polygon clip, zoom, robust scaling, hillshade, classes |
| `plot_dem`, `plot_water_frequency`, `plot_reference_map`, `plot_intertidal` | presets |
| `plot_uncertainty`, `plot_subpixel_relief` | HSR products |
| `plot_dem_3d` | interactive 3-D surface (Plotly) |
| `plot_scene`, `plot_scl`, `plot_scene_grid` | individual scenes |
| `plot_tide_series`, `plot_tide_distribution` | tides + what the satellite sampled |
| `plot_transect` | compared profiles |

### `explain/` — openEO/EOxHub-style diagrams
| Function | What it does |
|---|---|
| `draw_cube()` / `describe_cube()` | bands × time grid, labelled dims, REAL thumbnails |
| `draw_operation(kind=...)` | before→after with openEO semantics: `filter`, `reduce`, `apply`, `aggregate`, `mask` |
| `recording()`, `record()`, `step()`, `OpLog` | **opt-in** operation log |
| `draw_graph()` | process graph (DAG) with parameters in the boxes |
| `summarize()` | xarray-style HTML tables (dims, valid pixels, ranges, memory) |
| `Diagram.save()` | export any diagram to `.svg` |

---

## 10 · Output

| Module | Function | What it does |
|---|---|---|
| `mosaic.py` | `check_tiles()` | verify tiles are mergeable |
| | `seam_offsets()` | measure datum steps between neighbouring tiles |
| | `merge_tiles()` | merge (average overlaps, optional seam levelling) |
| `export.py` | `to_cog()` | Cloud-Optimized GeoTIFF |
| | `stac_item()`, `write_stac()` | STAC catalogue with provenance |
| `report.py` | `site_report()` | **self-contained HTML report** per site |
| | `build_report()` | custom report by sections |

---

## 11 · `legacy/` — superseded, kept runnable (for the paper)

| Function | What it is | Superseded by |
|---|---|---|
| `BathymetryReconstructor.reconstruct(method=...)` | 5 server-side methods: `optimized`, `pixels`, `isolines`, `step`, `siq`. The RMSE figures once quoted here were scored against the IGN LiDAR and are **retired** (see the notice in `legacy/bathymetry.py`) | `elevation.fit_hsr`, `elevation.fit_step` (local) |
| `water_frequency_multi()` | WF with SCL/NDWI/MNDWI/AWEI in **one job** — sound A/B design, but its ranking was scored against the same LiDAR and stays **provisional** | `frequency.water_frequency` |
| `udf.BATHYMETRY_UDF_SCL/_NDWI` | UDF sources kept **verbatim** (bit-exact, verified) | — |

---

## What was deliberately NOT ported, and why

| From the old package | Reason |
|---|---|
| `build_reference_map_local`, `build_reference_map_from_cube`, `build_reference_and_cloud_streaming` (SCL) | **three implementations of one algorithm**; `stability.reference_and_clouds` (streaming, any detector) remains |
| `compute_water_frequency_openeo` / `_from_cube` / `_streaming` / `_mndwi_` | same → `frequency.water_frequency` (+ the multi-index one in legacy) |
| `analyze_scl_cube_openeo`, `SclCubeAnalysis`, `NdwiCubeAnalysis` | "run-everything" orchestrators = the magic we removed; their steps are now standalone functions |
| ~40 `notebook_compat` wrappers (`plot_*`, `tif_to_*`, `download_date_*`) | one-line wrappers over functions that already exist |
| `SCLProcessor.load_stack`, `filter_dates_by_quality`, `correct_with_reference` | per-date file paths, superseded by the cube; `scene_quality` was ported |
| `mapper.evaluate_transition_cloud_coverage` | duplicates pass 2 of `reference_and_clouds` |
| `TideAnalyzer` (class) | flattened into functions in `gauges.py` |
| `raster.convert_tifs_to_png` (DataFrame-coupled) | simplified to `raster.tifs_to_png` |
| old `benchmark.py` | measures its own history; the results live in the profiling notebook |

---

## Ideas on the table (not implemented)

1. **`fusion`** — combine Landsat + Sentinel-2 to double the observation count (what DEA does).
2. **`tides_gridded`** — tide per pixel instead of per centroid (removes the phase error in long rias).
3. **`habitat`** — habitat classification (high/low marsh, mud, sand) from hydroperiod + indices + texture.
4. **`carbon`** — blue-carbon stock per zone from habitat and area.
5. **`sealevel`** — projections: which zones disappear at +0.5 / +1 m.
6. **`quality`** — unified per-pixel quality flags (n. observations, uncertainty, σ saturation).
