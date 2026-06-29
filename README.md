# Intertidal Analysis Toolkit

Pipeline modular para análisis de zonas intermareales usando Sentinel-2 Scene Classification Layer (SCL) y OpenEO.

## 🎯 Características

- **Arquitectura Modular**: 7 módulos especializados en lugar de código monolítico
- **OpenEO Backend**: Descarga y procesamiento en Copernicus Dataspace
- **Reference Map**: Clasificación automática agua/tierra/transición
- **Filtro Inteligente**: Análisis de nubosidad en zona intermareal
- **Validación de Marea**: Comparación con modelos GOT4.10 y CMEMS

## 📦 Estructura del Paquete

```
intertidal/
├── __init__.py           # Exports públicos
├── geometry.py           # Conversión DMS, polígonos, bbox, grids
├── raster.py            # I/O GeoTIFF, normalización
├── openeo_client.py     # Cliente Copernicus Dataspace
├── scl_processor.py     # Análisis SCL, reference maps
├── mapper.py            # Water frequency, zona intermareal
├── tide_analyzer.py     # Modelos de marea, métricas
└── visualization.py     # Plotting RGB, SCL, mapas
```

## 🚀 Instalación

```bash
# Activar entorno virtual
source .venv/Scripts/activate  # Windows Git Bash
# o
source .venv/bin/activate      # Linux/Mac

# Instalar dependencias
pip install openeo xarray geopandas rioxarray matplotlib contextily rasterio pyproj shapely scikit-image netcdf4 scipy
```

## 📖 Uso Básico

```python
from intertidal import (
    GeometryProcessor,
    RasterProcessor,
    OpenEOClient,
    SCLProcessor,
    Visualizer
)

# 1. Definir AOI
geo = GeometryProcessor()
polygon = geo.make_polygon(aoi_dms_list)
bbox = geo.bbox_from_polygon(polygon)

# 2. Conectar a OpenEO
client = OpenEOClient()
client.connect()

# 3. Descargar datos
client.download_rgb("2024-07-02", bbox, "tifs_rgb")
client.download_scl("2024-07-02", bbox, "tifs_scl")

# 4. Analizar calidad SCL
scl_proc = SCLProcessor(bad_classes=[3, 8, 9, 10, 11])
scl = RasterProcessor.read_scl("tifs_scl/scl_2024-07-02.tif")
stats = scl_proc.compute_stats(scl)
print(f"Píxeles malos: {stats['bad_pct']:.1f}%")

# 5. Construir reference map
stack = scl_proc.load_stack(clean_dates, "tifs_scl")
ref_map, p_water, p_land = scl_proc.build_reference_map_local(stack)

# 6. Visualizar
Visualizer.plot_reference_map(ref_map)
```

## 📓 Notebooks

### `gijon_sentinel2_scl_refactored.ipynb` (RECOMENDADO)
Pipeline completo refactorizado usando el paquete `intertidal`:
- ✅ Código modular y mantenible
- ✅ 13 secciones compactas (vs 70 celdas del original)
- ✅ Imports claros desde `intertidal`
- ✅ ~300 líneas vs ~2300 del original

### `gijon_sentinel2_scl.ipynb` (REFERENCIA)
Notebook original preservado como referencia histórica:
- ⚠️ Usa `utils.py` (ELIMINADO)
- ⚠️ 70 celdas, ~2364 líneas
- ℹ️ Solo para consulta, no ejecutar

## 🗺️ Pipeline Completo

```
1. Definir AOI (DMS → polígono → bbox)
   ↓
2. Conectar a OpenEO Copernicus Dataspace
   ↓
3. Consultar fechas disponibles
   ↓
4. Descargar RGB y SCL (batch)
   ↓
5. Filtro Global: descartar escenas muy nubladas
   ↓
6. Construir Reference Map local
   ↓
7. Filtro Transición: recuperar fechas nubladas fuera del estuario
   ↓
8. Análisis de Marea (GOT4.10 / CMEMS)
   ↓
9. Validación con Mareógrafo
   ↓
10. Visualización y Métricas
```

## 🔧 Módulos Principales

### GeometryProcessor
Operaciones geométricas y de coordenadas.

```python
geo = GeometryProcessor()

# DMS → decimal
lon, lat = geo.dms_to_coords("5°39'27.89\"W 43°33'15.40\"N")

# Crear polígono
polygon = geo.make_polygon(aoi_dms_list)

# Extraer bbox
bbox = geo.bbox_from_polygon(polygon)  # {west, south, east, north}

# Grid regular
grid = geo.make_grid(polygon, cell_size_deg=0.01)
```

### RasterProcessor
I/O de GeoTIFFs y operaciones raster.

```python
# Leer RGB con normalización percentil
rgb = RasterProcessor.read_rgb("rgb_2024-07-02.tif", normalize=True)

# Leer SCL
scl = RasterProcessor.read_scl("scl_2024-07-02.tif")

# Guardar GeoTIFF
RasterProcessor.save_geotiff(
    "output.tif", 
    data, 
    transform, 
    crs="EPSG:4326"
)
```

### OpenEOClient
Interfaz con backend OpenEO.

```python
client = OpenEOClient()
client.connect()

# Fechas disponibles
dates = client.get_available_dates(bbox, time_extent, max_cloud_cover=80)

# Descarga individual
client.download_rgb("2024-07-02", bbox, "tifs_rgb")

# Descarga batch
results = client.download_rgb_batch(date_list, bbox, "tifs_rgb")

# Reference map en backend (UDF)
client.build_reference_map(bbox, time_extent, "ref_map.tif")
```

### SCLProcessor
Análisis de calidad y reference maps.

```python
scl_proc = SCLProcessor(bad_classes=[3, 8, 9, 10, 11])

# Estadísticas de calidad
stats = scl_proc.compute_stats(scl_array)

# Filtrar fechas
result = scl_proc.filter_dates_by_quality(dates, "tifs_scl", max_bad_fraction=0.20)

# Reference map local
stack = scl_proc.load_stack(clean_dates, "tifs_scl")
ref_map, p_water, p_land = scl_proc.build_reference_map_local(stack)

# Estadísticas en zona de transición
transition_mask = (ref_map == 0)
stats = scl_proc.compute_transition_stats(scl_array, transition_mask)
```

### Visualizer
Generación de gráficos.

```python
# RGB + SCL
Visualizer.plot_rgb_with_scl("2024-07-02", "tifs_rgb", "tifs_scl", polygon=aoi)

# Reference map
Visualizer.plot_reference_map(ref_map)

# Grid de RGBs
Visualizer.plot_rgb_grid(clean_dates, "tifs_rgb", ncols=4)

# Serie temporal de marea
Visualizer.plot_tide_timeseries(df_tides, site="Gijón")

# Comparación de modelos
Visualizer.plot_tide_model_comparison(df_comparison)
```

## 🌊 Modelos de Marea

```python
from tidemodel import PyTMDTideModel, CopernicusTideModel

# GOT4.10c (global ocean tide)
tide_model = PyTMDTideModel(
    model_name="GOT4.10c",
    model_dir="tide_models/GOT4.10c"
)

height = tide_model.predict_tide_height(
    lon=-5.66,
    lat=43.54,
    datetime=pd.Timestamp("2024-07-02 11:00:00")
)

# CMEMS (Copernicus Marine)
cmems = CopernicusTideModel()
height = cmems.get_tide_height(lon, lat, datetime)
```

## 📊 Estadísticas SCL

**Clases Scene Classification Layer (Sentinel-2 L2A):**

| Clase | Nombre | Color | Calidad |
|-------|--------|-------|---------|
| 0 | Sin datos | Negro | - |
| 1 | Saturado/Defectuoso | Rojo | ❌ |
| 2 | Sombra oscura | Gris oscuro | ⚠️ |
| 3 | Sombra nube | Marrón | ❌ |
| 4 | Vegetación | Verde | ✅ |
| 5 | No vegetación | Amarillo | ✅ |
| 6 | Agua | Azul | ✅ |
| 7 | Incierto | Gris | ⚠️ |
| 8 | Nube probabilidad media | Gris claro | ❌ |
| 9 | Nube probabilidad alta | Blanco | ❌ |
| 10 | Cirrus | Azul claro | ❌ |
| 11 | Nieve/Hielo | Rosa | ❌ |

**Clases típicamente "malas"**: 3, 8, 9, 10, 11

## 📁 Archivos del Proyecto

```
Intertidal_analysis/
├── intertidal/                      # Paquete modular
│   ├── __init__.py
│   ├── geometry.py                  # ~200 líneas
│   ├── raster.py                    # ~300 líneas
│   ├── openeo_client.py            # ~600 líneas
│   ├── scl_processor.py            # ~450 líneas
│   ├── mapper.py                    # ~350 líneas
│   ├── tide_analyzer.py            # ~300 líneas
│   └── visualization.py            # ~650 líneas
├── gijon_sentinel2_scl_refactored.ipynb  # Notebook refactorizado (USAR)
├── gijon_sentinel2_scl.ipynb       # Notebook original (REFERENCIA)
├── tidemodel.py                     # Wrapper para pyTMD
├── ARCHITECTURE.md                  # Documentación del diseño
├── README.md                        # Este archivo
├── tifs_rgb/                        # RGBs descargados
├── tifs_scl/                        # SCLs descargados
└── tide_models/                     # Modelos de marea
    └── GOT4.10c/
```

## 🔄 Migración desde `utils.py`

| Función antigua (utils.py) | Nueva ubicación |
|----------------------------|-----------------|
| `dms_to_coords()` | `GeometryProcessor.dms_to_coords()` |
| `make_polygon()` | `GeometryProcessor.make_polygon()` |
| `bbox_from_polygon()` | `GeometryProcessor.bbox_from_polygon()` |
| `tif_to_rgb()` | `RasterProcessor.read_rgb()` |
| `tif_to_scl()` | `RasterProcessor.read_scl()` |
| `download_date_rgb()` | `OpenEOClient.download_rgb()` |
| `download_date_scl()` | `OpenEOClient.download_scl()` |
| `compute_scl_stats()` | `SCLProcessor.compute_stats()` |
| `build_reference_map()` | `SCLProcessor.build_reference_map_local()` |
| `compute_transition_cloud_stats()` | `SCLProcessor.compute_transition_stats()` |
| `plot_scl_map()` | `Visualizer.plot_scl_map()` |
| `plot_reference_map()` | `Visualizer.plot_reference_map()` |

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'geopandas'`
```bash
pip install geopandas shapely pyproj
```

### Error: `ModuleNotFoundError: No module named 'intertidal'`
Asegúrate de ejecutar el notebook desde `Intertidal_analysis/` o añadir al PYTHONPATH:
```python
import sys
sys.path.append('c:/Users/Jorge/sketch_fitton/Intertidal_analysis')
```

### Error: OpenEO authentication fails
```python
# Reintentar con OIDC explícito
client.connect(use_oidc=True)
```

## 📚 Referencias

- **OpenEO**: https://openeo.org/
- **Copernicus Dataspace**: https://dataspace.copernicus.eu/
- **Sentinel-2 L2A**: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/product-types/level-2a
- **pyTMD**: https://github.com/tsutterley/pyTMD
- **CMEMS**: https://marine.copernicus.eu/

## 🤝 Contribuciones

Este es un proyecto de investigación. Para modificaciones:

1. Consultar `ARCHITECTURE.md` para entender el diseño
2. Modificar el módulo correspondiente en `intertidal/`
3. Actualizar docstrings y ejemplos
4. Probar con el notebook refactorizado

## 📝 Licencia

Proyecto de investigación académica.

---

**Autor**: Jorge  
**Última actualización**: Junio 2026  
**Python**: 3.13.3  
**OpenEO Backend**: openeo.dataspace.copernicus.eu
