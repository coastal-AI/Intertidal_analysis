# pyintertidal — catálogo de funciones (para curar: ✅ entra / ❌ fuera)

Estado: **todo lo de abajo ya está implementado, documentado en inglés y probado**.
Marca lo que sobre y lo quito. 37 módulos, 0 imports del paquete viejo.

**Principio aplicado:** cada capacidad existe UNA sola vez. Donde el paquete
viejo tenía varias implementaciones del mismo algoritmo, el core se queda la
que mejor midió y `legacy/` conserva las superadas (para el paper). Al final
del documento está la lista de lo que **no** se portó y por qué.

---

## 1 · Área de estudio

### `aoi.py` — el AOI es un POLÍGONO
| Función | Qué hace |
|---|---|
| `AOI.from_polygon / from_bbox / from_dms` | crea el AOI desde vértices, caja o coordenadas DMS |
| `AOI.bbox` | caja de descarga (detalle interno: los cubos son rectangulares) |
| `AOI.centroid`, `AOI.area_km2` | punto de marea por defecto, superficie |
| `AOI.raster_mask(transform, crs, shape)` | máscara del polígono → recorta productos y figuras |
| `AOI.tile(cell_km, overlap_km)` | **parte el AOI en celdas contiguas** para producción paralela |
| `AOI.to_geojson()` | exporta el polígono (revisar el teselado en QGIS) |
| `parse_dms()` | `43°30'24"N 5°25'32"W` → decimal |
| `as_aoi()` | acepta AOI / shapely / lista / dict en cualquier función |

### `sites.py` — registro de sitios con nombre
| Función | Qué hace |
|---|---|
| `get("villaviciosa")` | AOI listo (incluye Santoña, Foz, Urdaibai) |
| `register()`, `describe()`, `list_sites()` | añadir/ver sitios |

---

## 2 · Datos

### `cube.py` — el datacubo (descarga 1 vez, luego streaming)
| Función | Qué hace |
|---|---|
| `SentinelCube(aoi, periodo, water=...)` | cubo cacheado; las bandas dependen del detector |
| `.ensure(conn)` | descarga **solo si no está cacheado** (1 job) |
| `.dates / .grid / .shape` | metadatos sin cargar píxeles |
| `.stream(chunk)` | iterador por bloques temporales (RAM acotada) |
| `.thumbnails(n, size)` | miniaturas reales para los diagramas de `explain` |
| `open_cube`, `auto_chunk`, `grid_from_dataset`, `cube_dates` | primitivas netCDF |

### `raster.py` — GeoTIFF (única definición de IO)
| Función | Qué hace |
|---|---|
| `write_geotiff`, `read_band`, `grid_of` | productos |
| `is_valid_tif`, `read_rgb`, `read_scl`, `scene_coverage` | escenas por fecha |
| `percentile_normalize`, `tifs_to_png` | visualización / datasets ML |

### `scenes.py` — escenas sueltas (inspección y QA)
| Función | Qué hace |
|---|---|
| `connect()` | conexión OpenEO (+ fix SSL de Windows) |
| `available_dates()` | fechas disponibles (solo metadatos, gratis) |
| `download_rgb / download_scl / download_indices` | una fecha |
| `download_scenes()` | **muchas fechas en UN job** |

---

## 3 · Detección de agua

### `water.py`
| Función | Qué hace |
|---|---|
| `SCL_CLASSES` | tabla de clases + colores (incl. 12 = marisma) |
| `BAD_CLASSES`, `CLEAR_CLASSES`, `SCL_WATER_CLASSES` | constantes compartidas |
| `bands_for(water)` | qué bandas descargar según detector |
| `index_block()` | NDWI / MNDWI / AWEI por bloque |
| `water_land_masks()` | máscaras mojado/seco/claro (nubes SIEMPRE por SCL) |
| `scene_quality()` | calidad de una escena (fracción mala + histograma) |
| `otsu_threshold()` | calibración opcional del umbral (el notebook decide) |

### `marsh.py` — vegetación inundada
| Función | Qué hace |
|---|---|
| `detect_flooded_vegetation()` | vegetación + índice alto = marisma inundada |
| `correct_scl()` | reetiqueta a clase 12 |
| `marsh_frequency()` | con qué frecuencia un píxel es marisma (extensión de hábitat) |

---

## 4 · Productos base

### `stability.py`
| Función | Qué hace |
|---|---|
| `reference_and_clouds()` | mapa 0/1/2 + % nubes por fecha (2 pasadas streaming) |
| `filter_dates()` | fechas válidas según umbral |
| `date_recovery_gain()` | **cuántas fechas rescata** el filtro de transición vs global |
| `transition_geometry()` | vectoriza la zona de transición (GeoDataFrame) |

### `frequency.py`
| Función | Qué hace |
|---|---|
| `water_frequency()` | fracción de observaciones claras mojadas |
| `multiotsu_window()` | ventana [low, high] por multi-Otsu (opcional) |
| `intertidal_mask()` | transición ∩ ventana ∩ polígono |

---

## 5 · Mareas

### `tides.py`
| Función | Qué hace |
|---|---|
| `TideService(model, directory, location, backend)` | modelo y punto elegibles |
| `.heights_for(aoi, dates)` | marea en la hora de paso de cada escena |
| `.series(aoi, start, end)` | serie continua (para coverage e hydroperiod) |
| `water_centroid()` | punto de marea **sobre agua** (mejor que el centroide) |

### `tidemodels.py`
| Clase | Qué hace |
|---|---|
| `PyTMDTideModel` | GOT4.10 / FES2014 / FES2022; búsqueda de celda oceánica; batch vectorizado |
| `CopernicusTideModel` | nivel del mar CMEMS (incluye marea meteorológica) |

### `overpass.py`
| Función | Qué hace |
|---|---|
| `get_overpass_times()` | hora UTC real de cada escena (del nombre del producto) |
| `mean_overpass_hour()` | hora media de paso |

### `gauges.py` — validación del modelo de marea
| Función | Qué hace |
|---|---|
| `load_gauge()` | carga PORTUS/REDMAR (CSV/Excel) |
| `compare_with_model()` | empareja observado vs modelado |
| `gauge_metrics()` | offset de datum, RMSE, MAE, correlación |
| `compare_models()` | varios modelos contra el mismo mareógrafo |

### `coverage.py` — ¿es mapeable este sitio?
| Función | Qué hace |
|---|---|
| `tidal_range_coverage()` | % del rango mareal muestreado |
| `ks_uniformity()` | test KS contra uniforme |
| `shannon_entropy()` | entropía normalizada + bins vacíos |
| `dispersion_index()` | VMR: regular / aleatorio / agrupado |
| `gap_statistics()` | dónde está el mayor hueco vertical |
| `vertical_sampling_resolution()` | VSR = resolución vertical efectiva |
| `representativity()` | % de cuantiles muestreados |
| `coverage_metrics()`, `coverage_verdict()`, `print_coverage_report()` | conjunto + veredicto + informe |

---

## 6 · Elevación

### `elevation.py`
| Función | Qué hace |
|---|---|
| `epochs()` | parte el archivo en épocas (ancladas al final) |
| `fit_hsr()` | **HSR** (método propio): μ + σ sub-píxel + incertidumbre CR + DEM fino |
| `fit_step()` | método del escalón (DEA) local, sin job |
| `hsr_stage1()`, `hsr_stage2()` | núcleo matemático (ajuste + super-resolución) |
| `ElevationResult` | `.clip_to()`, `.save()`, `.summary()`, `.transform_fine` |

### `terrain.py` — derivados de cualquier DEM
| Función | Qué hace |
|---|---|
| `slope`, `aspect`, `hillshade` | pendiente, orientación, sombreado (NaN-aware) |
| `roughness` | rugosidad entre píxeles (contrasta con σ de HSR) |
| `contours` | isolíneas como GeoDataFrame |

---

## 7 · Ciencia derivada (módulos nuevos)

### `hydroperiod.py` — el driver ecológico
| Función | Qué hace |
|---|---|
| `hydroperiod()` | fracción de tiempo sumergido por píxel |
| `exposure_time()` | horas/año de exposición al aire (estrés por desecación) |
| `zonation()` | zonas ecológicas (subtidal → supratidal) |
| `zone_areas()` | km² y % por zona |
| `inundation_curve()` | área inundada vs altura de marea |

### `hypsometry.py` — firma morfológica
| Función | Qué hace |
|---|---|
| `hypsometric_curve()` | área acumulada vs cota |
| `hypsometric_integral()` | forma en un número (convexo/cóncavo) |
| `describe()`, `compare_curves()`, `plot_curves()` | interpretación y comparación |

### `morphodynamics.py` — cambio entre épocas
| Función | Qué hace |
|---|---|
| `dem_difference()` | resta con **test de significancia** por incertidumbre propagada |
| `sediment_budget()` | volúmenes de erosión/acreción y régimen |
| `change_by_zone()` | balance por zona ecológica |
| `summarize()` | informe legible |

### `shoreline.py` — líneas de costa
| Función | Qué hace |
|---|---|
| `extract_waterline()` | vectoriza la orilla de una escena (con su marea) |
| `waterline_series()` | serie de orillas, opcionalmente a marea fija |
| `shoreline_change()` | posición y tasa de cambio por transectos (estilo DSAS) |

---

## 8 · Validación

### `validation.py`
| Función | Qué hace |
|---|---|
| `download_mdt_ign()` | LiDAR 5 m del IGN por WCS |
| `reproject_to_grid()` | cualquier referencia al grid de análisis |
| `compare_dems()` | métricas multi-DEM (sesgo de datum aparte) |
| `transect_profile()` | perfil por transecto GPS (**campañas de campo**) |
| `wf_vs_elevation()` | correlación WF↔cota (compara índices sin DEM propio) |
| `mask_agreement()` | IoU entre máscaras de métodos distintos |

---

## 9 · Visualización y explicación

### `viz.py` — componible, todo devuelve `(fig, ax)`
| Función | Qué hace |
|---|---|
| `plot_map()` | **primitiva**: clip a polígono, zoom, escala robusta, hillshade, clases |
| `plot_dem`, `plot_water_frequency`, `plot_reference_map`, `plot_intertidal` | presets |
| `plot_uncertainty`, `plot_subpixel_relief` | productos de HSR |
| `plot_dem_3d` | superficie 3D interactiva (Plotly) |
| `plot_scene`, `plot_scl`, `plot_scene_grid` | escenas individuales |
| `plot_tide_series`, `plot_tide_distribution` | mareas + qué muestreó el satélite |
| `plot_transect` | perfiles comparados |

### `explain/` — estética openEO/EOxHub (4 tipos)
| Función | Qué hace |
|---|---|
| `draw_cube()` / `describe_cube()` | **rejilla bandas × tiempo** con dims etiquetadas y miniaturas REALES |
| `draw_operation(kind=...)` | antes→después con semántica openEO: `filter`, `reduce`, `apply`, `aggregate`, `mask` |
| `recording()`, `record()`, `step()`, `OpLog` | registro de operaciones **opt-in** |
| `draw_graph()` | grafo de procesos (DAG) con parámetros en las cajas |
| `summarize()` | tablas HTML tipo xarray (dims, píxeles válidos, rangos, memoria) |
| `Diagram.save()` | exporta cualquier diagrama a `.svg` |

---

## 10 · Salida

| Módulo | Función | Qué hace |
|---|---|---|
| `mosaic.py` | `check_tiles()` | verifica que los tiles son fusionables |
| | `seam_offsets()` | mide escalones de datum entre tiles vecinos |
| | `merge_tiles()` | fusiona (promedia solapes, opción de nivelar costuras) |
| `export.py` | `to_cog()` | Cloud-Optimized GeoTIFF |
| | `stac_item()`, `write_stac()` | catálogo STAC con procedencia |
| `report.py` | `site_report()` | **informe HTML autocontenido** por sitio |
| | `build_report()` | informe a medida por secciones |

---

## 11 · `legacy/` — superado, pero ejecutable (para el paper)

| Función | Qué es | Qué lo sustituye |
|---|---|---|
| `BathymetryReconstructor.reconstruct(method=...)` | 5 métodos server-side: `optimized` (RMSE 1.14 m), `pixels` (3.35), `isolines` (3.39), `step` (0.85), `siq` (sin validar) | `elevation.fit_hsr` (0.76 m) y `elevation.fit_step`, locales |
| `water_frequency_multi()` | WF con SCL/NDWI/MNDWI/AWEI en **un job** (la comparación justa que eligió NDWI) | `frequency.water_frequency` |
| `udf.BATHYMETRY_UDF_SCL/_NDWI` | fuentes UDF **verbatim** (bit-exactas, verificadas) | — |

---

## ❌ Lo que NO se portó (y por qué) — dime si quieres algo de aquí

| Del paquete viejo | Motivo |
|---|---|
| `build_reference_map_local`, `build_reference_map_from_cube`, `build_reference_and_cloud_streaming` (SCL) | **3 implementaciones del mismo algoritmo**. Queda una: `stability.reference_and_clouds` (streaming, cualquier detector) |
| `compute_water_frequency_openeo` / `_from_cube` / `_streaming` / `_mndwi_` | idem → `frequency.water_frequency` (+ la multi-índice en legacy) |
| `analyze_scl_cube_openeo`, `SclCubeAnalysis`, `NdwiCubeAnalysis` | orquestadores "todo en uno" = la magia que quitamos. Sus pasos son ahora funciones sueltas |
| ~40 wrappers de `notebook_compat` (`plot_*`, `tif_to_*`, `download_date_*`) | envoltorios de una línea sobre funciones que ya existen |
| `SCLProcessor.load_stack`, `filter_dates_by_quality`, `correct_with_reference` | ruta de ficheros por fecha, superada por el cubo. `scene_quality` sí se portó |
| `mapper.evaluate_transition_cloud_coverage` | duplica la pasada 2 de `reference_and_clouds` |
| `TideAnalyzer` (clase) | convertido en funciones planas en `gauges.py` |
| `raster.convert_tifs_to_png` (con DataFrame) | simplificado a `raster.tifs_to_png` |
| `benchmark.py` (viejo vs nuevo) | media su propia historia; los resultados ya están en el notebook de perfilado |

---

## Ideas que quedan sobre la mesa (no implementadas)

1. **`fusion`** — combinar Landsat + Sentinel-2 para duplicar observaciones (lo que hace el DEA).
2. **`tides_gridded`** — marea por píxel en vez de por centroide (elimina el error de fase en rías largas).
3. **`habitat`** — clasificación de hábitats (marisma alta/baja, fango, arena) combinando hidroperiodo + índices + textura.
4. **`carbon`** — stock de carbono azul por zona a partir del hábitat y el área.
5. **`sealevel`** — proyecciones: qué zonas desaparecen con +0.5 / +1 m.
6. **`quality`** — banderas de calidad por píxel unificadas (nº de observaciones, incertidumbre, saturación de σ).
