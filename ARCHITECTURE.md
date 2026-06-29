# 🏗️ Arquitectura Modular - Intertidal Analysis

## 📋 Diseño de Estructura

### Estructura de Archivos Propuesta

```
Intertidal_analysis/
├── intertidal/
│   ├── __init__.py                # Exporta clases principales
│   ├── geometry.py                # GeometryProcessor (coordenadas, polígonos, bbox)
│   ├── raster.py                  # RasterProcessor (lectura/escritura GeoTIFF)
│   ├── openeo_client.py           # OpenEOClient (descargas Copernicus)
│   ├── scl_processor.py           # SCLProcessor (análisis calidad SCL)
│   ├── mapper.py                  # IntertidalMapper (reference maps, water frequency)
│   ├── tide_analyzer.py           # TideAnalyzer (análisis mareales)
│   └── visualization.py           # Visualizer (plots y gráficos)
├── gijon_sentinel2_scl_refactored.ipynb  # Notebook refactorizado
├── tidemodel.py                   # Wrapper modelos (se mantiene)
└── requirements.txt               # Dependencias
```

---

## 📦 Módulos y Responsabilidades

### 1. **`geometry.py`** — `GeometryProcessor`

**Responsabilidad**: Operaciones con coordenadas y geometrías espaciales

**Funciones migradas de utils.py**:
- `dms_to_coords()` → método de clase
- `dms_to_decimal()` → método de clase
- `make_polygon()` → método de clase
- `bbox_from_polygon()` → método de clase
- `make_grid()` → método de clase

**API propuesta**:
```python
class GeometryProcessor:
    @staticmethod
    def dms_to_coords(coord_str: str) -> tuple[float, float]:
        """Convierte DMS a (lon, lat) decimal"""
        
    @staticmethod
    def dms_to_decimal(dms_str: str) -> tuple[float, float]:
        """Convierte DMS a (lat, lon) decimal"""
        
    @staticmethod
    def make_polygon(dms_list: list[str]) -> Polygon:
        """Crea polígono desde lista de coordenadas DMS"""
        
    @staticmethod
    def bbox_from_polygon(polygon: Polygon) -> dict:
        """Extrae bounding box para OpenEO"""
        
    @staticmethod
    def make_grid(polygon, cell_size_deg=0.01, crs="EPSG:4326") -> GeoDataFrame:
        """Genera grid regular sobre polígono"""
```

**Uso**:
```python
from intertidal import GeometryProcessor

geo = GeometryProcessor()
polygon = geo.make_polygon(aoi_dms_list)
bbox = geo.bbox_from_polygon(polygon)
```

---

### 2. **`raster.py`** — `RasterProcessor`

**Responsabilidad**: Lectura/escritura de archivos raster GeoTIFF

**Funciones migradas**:
- `is_valid_tif()` → método
- `tif_to_rgb()` → método
- `tif_to_scl()` → método
- `norm_percentile()` → método privado `_norm_percentile()`
- `load_reference_map_tif()` → método

**API propuesta**:
```python
class RasterProcessor:
    @staticmethod
    def is_valid_tif(path: str | Path) -> bool:
        """Verifica si un archivo TIFF es válido"""
        
    @staticmethod
    def read_rgb(tif_path: str, normalize=True) -> np.ndarray | None:
        """Lee RGB como array (H, W, 3)"""
        
    @staticmethod
    def read_scl(tif_path: str) -> np.ndarray | None:
        """Lee SCL como array uint8"""
        
    @staticmethod
    def save_geotiff(path: str, data: np.ndarray, transform, crs):
        """Guarda array como GeoTIFF georeferenciado"""
        
    @staticmethod
    def load_reference_map(ref_map_path: str) -> tuple[np.ndarray, Any, Any]:
        """Carga reference map con metadatos"""
        
    @staticmethod
    def _norm_percentile(band, pmin=2, pmax=98):
        """Normalización por percentiles (privado)"""
```

**Uso**:
```python
from intertidal import RasterProcessor

raster = RasterProcessor()
rgb = raster.read_rgb("data/rgb_2024-07-02.tif")
scl = raster.read_scl("data/scl_2024-07-02.tif")
```

---

### 3. **`openeo_client.py`** — `OpenEOClient`

**Responsabilidad**: Interfaz con backend Copernicus Dataspace (OpenEO)

**Funciones migradas**:
- `download_date_rgb()` → método
- `download_date_scl()` → método
- `download_reference_map_openeo()` → método
- `apply_datacube()` → UDF embebido como método privado

**API propuesta**:
```python
class OpenEOClient:
    def __init__(self, backend_url="openeo.dataspace.copernicus.eu"):
        self.backend_url = backend_url
        self.connection = None
        
    def connect(self, use_oidc=True):
        """Conecta al backend con autenticación OIDC"""
        
    def get_available_dates(self, bbox, time_extent, max_cloud_cover=100) -> list[str]:
        """Lista fechas disponibles para AOI"""
        
    def download_rgb(self, date: str, bbox: dict, output_dir: str) -> str:
        """Descarga RGB para una fecha"""
        
    def download_scl(self, date: str, bbox: dict, output_dir: str) -> str:
        """Descarga SCL para una fecha"""
        
    def download_rgb_batch(self, dates: list[str], bbox: dict, output_dir: str) -> dict:
        """Descarga RGB para múltiples fechas"""
        
    def download_scl_batch(self, dates: list[str], bbox: dict, output_dir: str) -> dict:
        """Descarga SCL para múltiples fechas"""
        
    def build_reference_map(
        self, 
        bbox: dict, 
        time_extent: list[str], 
        output_path: str,
        stable_threshold=0.95,
        coastal_buffer_pixels=10
    ) -> str:
        """Construye reference map vía UDF en OpenEO"""
        
    @staticmethod
    def _get_reference_udf_code() -> str:
        """Retorna código UDF para reference map (privado)"""
```

**Uso**:
```python
from intertidal import OpenEOClient

client = OpenEOClient()
client.connect()

# Descargar datos
client.download_rgb("2024-07-02", bbox, "tifs_rgb")
client.download_scl("2024-07-02", bbox, "tifs_scl")

# Reference map en backend
client.build_reference_map(bbox, ["2024-01-01", "2024-12-31"], "reference.tif")
```

---

### 4. **`scl_processor.py`** — `SCLProcessor`

**Responsabilidad**: Análisis de calidad SCL y filtrado de escenas

**Funciones migradas**:
- `compute_scl_stats()` → método
- `load_scl_stack()` → método
- `build_reference_map()` → método (construcción local)
- `compute_transition_cloud_stats()` → método
- `correct_scl_with_reference()` → método

**API propuesta**:
```python
class SCLProcessor:
    def __init__(self, bad_classes=[3, 8, 9, 10, 11]):
        self.bad_classes = bad_classes
        
    def compute_stats(self, scl_array: np.ndarray) -> dict:
        """Calcula estadísticas de calidad SCL"""
        
    def filter_dates_by_quality(
        self, 
        dates: list[str], 
        scl_dir: str, 
        max_bad_fraction=0.20
    ) -> dict:
        """Filtra fechas por calidad SCL global"""
        
    def load_stack(self, dates: list[str], scl_dir: str) -> np.ndarray:
        """Carga múltiples SCLs como stack 3D"""
        
    def build_reference_map_local(
        self,
        scl_stack: np.ndarray,
        stable_threshold=0.95,
        coastal_buffer_pixels=10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construye reference map desde stack local"""
        
    def compute_transition_stats(
        self,
        scl_array: np.ndarray,
        transition_mask: np.ndarray
    ) -> dict:
        """Calcula estadísticas solo en zona de transición"""
        
    def correct_with_reference(
        self,
        scl_arr: np.ndarray,
        reference_map: np.ndarray
    ) -> np.ndarray:
        """Corrige SCL usando reference map"""
```

**Uso**:
```python
from intertidal import SCLProcessor

processor = SCLProcessor(bad_classes=[3, 8, 9, 10, 11])

# Analizar calidad
stats = processor.compute_stats(scl_array)

# Filtrar fechas
valid_dates = processor.filter_dates_by_quality(
    selected_dates, "tifs_scl", max_bad_fraction=0.20
)

# Construir reference map local
stack = processor.load_stack(clean_dates, "tifs_scl")
ref_map, p_water, p_land = processor.build_reference_map_local(stack)
```

---

### 5. **`mapper.py`** — `IntertidalMapper`

**Responsabilidad**: Mapeo de zona intermareal (water frequency, máscaras)

**Funciones migradas**:
- `get_water_centroid()` → método
- `_transition_geometry_from_reference_map()` → método privado
- `evaluate_transition_cloud_coverage_openeo()` → método
- `quantify_reference_gain()` → método
- `compute_water_frequency_openeo()` → método

**API propuesta**:
```python
class IntertidalMapper:
    def __init__(self, openeo_client: OpenEOClient):
        self.client = openeo_client
        
    def compute_water_frequency(
        self,
        bbox: dict,
        time_extent: list[str],
        output_path: str,
        transition_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Calcula frecuencia de agua vía OpenEO"""
        
    def create_intertidal_mask(
        self,
        water_frequency: np.ndarray,
        min_freq=0.10,
        max_freq=0.95
    ) -> np.ndarray:
        """Genera máscara binaria de zona intermareal"""
        
    def evaluate_transition_cloud_coverage(
        self,
        bbox: dict,
        time_extent: list[str],
        reference_map: np.ndarray,
        transform,
        crs,
        bad_classes: list,
        max_cloud_pct=20
    ) -> dict:
        """Evalúa cobertura de nubes en transición vía OpenEO"""
        
    def quantify_reference_gain(
        self,
        bbox: dict,
        time_extent: list[str],
        reference_map: np.ndarray,
        transform,
        crs,
        bad_classes: list,
        global_threshold=0.20,
        transition_threshold=0.20
    ) -> dict:
        """Cuantifica ganancia del filtro de transición"""
        
    def get_water_centroid(
        self,
        reference_map: np.ndarray,
        transform
    ) -> tuple[float, float]:
        """Encuentra centroide de zona de agua estable"""
        
    @staticmethod
    def _transition_geometry_from_reference_map(
        reference_map: np.ndarray,
        transform,
        crs
    ):
        """Extrae geometría de transición (privado)"""
```

**Uso**:
```python
from intertidal import IntertidalMapper, OpenEOClient

client = OpenEOClient()
client.connect()

mapper = IntertidalMapper(client)

# Water frequency
water_freq = mapper.compute_water_frequency(
    bbox, ["2024-01-01", "2024-12-31"], "water_freq.tif"
)

# Máscara intermareal
intertidal_mask = mapper.create_intertidal_mask(water_freq)
```

---

### 6. **`tide_analyzer.py`** — `TideAnalyzer`

**Responsabilidad**: Análisis de datos mareales y validación

**Funciones migradas**:
- `calculate_tide_model_metrics()` → método
- `print_tide_model_metrics()` → método

**API propuesta**:
```python
class TideAnalyzer:
    def __init__(self, tide_model):
        self.model = tide_model
        
    def load_gauge_data(
        self,
        gauge_file: str,
        time_extent: list[str]
    ) -> pd.DataFrame:
        """Carga datos de mareógrafo"""
        
    def predict_tides(
        self,
        lat: float,
        lon: float,
        times: pd.DatetimeIndex
    ) -> pd.Series:
        """Predice mareas para ubicación y tiempos"""
        
    def calculate_anomalies(
        self,
        observations: pd.Series,
        predictions: pd.Series
    ) -> pd.Series:
        """Calcula anomalías (obs - mean(obs) vs pred - mean(pred))"""
        
    def compare_models(
        self,
        df_comparison: pd.DataFrame,
        cmems_available=False,
        obs_available=False
    ) -> dict:
        """Compara múltiples modelos y calcula métricas"""
        
    @staticmethod
    def print_metrics(metrics: dict):
        """Imprime métricas de comparación"""
```

**Uso**:
```python
from intertidal import TideAnalyzer
from tidemodel import PyTMDTideModel

model = PyTMDTideModel("GOT4.10c", "tide_models/GOT4.10c")
analyzer = TideAnalyzer(model)

# Cargar observaciones
obs = analyzer.load_gauge_data("mareografo_gijon.xlsx", time_extent)

# Predicciones
preds = analyzer.predict_tides(lat, lon, obs.index)

# Comparar
metrics = analyzer.compare_models(df_comparison)
analyzer.print_metrics(metrics)
```

---

### 7. **`visualization.py`** — `Visualizer`

**Responsabilidad**: Generación de gráficos y visualizaciones

**Funciones migradas**:
- `plot_rgb_grid()` → método
- `plot_scl_map()` → método
- `plot_reference_map()` → método
- `plot_water_frequency()` → método
- `plot_tide_timeseries()` → método
- `plot_tide_vertical_distribution()` → método
- `plot_tide_histogram_with_analysis()` → método
- `plot_tide_model_comparison()` → método

**API propuesta**:
```python
class Visualizer:
    @staticmethod
    def plot_rgb_grid(
        dates: list[str],
        rgb_dir: str,
        polygon=None,
        title="RGB Grid",
        cols=4
    ):
        """Grid de imágenes RGB"""
        
    @staticmethod
    def plot_scl_map(
        date: str,
        scl_dir: str,
        scl_colors: dict,
        scl_bad_classes: list,
        scl_stats: dict | None = None,
        scl_max_bad_fraction: float | None = None
    ):
        """Mapa SCL con colores ESA"""
        
    @staticmethod
    def plot_reference_map(reference_map: np.ndarray):
        """Visualiza reference map (agua/tierra/transición)"""
        
    @staticmethod
    def plot_water_frequency(
        water_freq: np.ndarray,
        polygon=None,
        title="Water Frequency"
    ):
        """Mapa de frecuencia de agua"""
        
    @staticmethod
    def plot_tide_timeseries(
        df_tides: pd.DataFrame,
        site: str,
        ref_time_extent: list[str],
        lat_centroid: float,
        lon_centroid: float
    ):
        """Serie temporal de mareas"""
        
    @staticmethod
    def plot_tide_vertical_distribution(
        df_tides: pd.DataFrame,
        site: str,
        ref_time_extent: list[str]
    ):
        """Distribución vertical de mareas"""
        
    @staticmethod
    def plot_tide_histogram(df_tides: pd.DataFrame, site: str):
        """Histograma de alturas de marea"""
        
    @staticmethod
    def plot_model_comparison(
        df_comparison: pd.DataFrame,
        gauge_name: str,
        cmems_available=False
    ):
        """Comparación entre modelos de marea"""
```

**Uso**:
```python
from intertidal import Visualizer

viz = Visualizer()

# Grid RGB
viz.plot_rgb_grid(valid_dates, "tifs_rgb", polygon=aoi_polygon)

# Reference map
viz.plot_reference_map(reference_map)

# Water frequency
viz.plot_water_frequency(water_freq, polygon=aoi_polygon)

# Mareas
viz.plot_tide_timeseries(df_tides, "Gijón", time_extent, lat, lon)
```

---

## 🔄 Archivo `__init__.py`

Exporta todas las clases principales para importación simple:

```python
"""
Intertidal Analysis Toolkit
============================

Biblioteca modular para análisis de zonas intermareales usando Sentinel-2.
"""

from .geometry import GeometryProcessor
from .raster import RasterProcessor
from .openeo_client import OpenEOClient
from .scl_processor import SCLProcessor
from .mapper import IntertidalMapper
from .tide_analyzer import TideAnalyzer
from .visualization import Visualizer

__all__ = [
    "GeometryProcessor",
    "RasterProcessor",
    "OpenEOClient",
    "SCLProcessor",
    "IntertidalMapper",
    "TideAnalyzer",
    "Visualizer",
]

__version__ = "1.0.0"
```

---

## 🚀 Ejemplo de Uso Completo

```python
from intertidal import (
    GeometryProcessor,
    RasterProcessor,
    OpenEOClient,
    SCLProcessor,
    IntertidalMapper,
    TideAnalyzer,
    Visualizer
)

# 1. Geometría
geo = GeometryProcessor()
polygon = geo.make_polygon(aoi_dms_list)
bbox = geo.bbox_from_polygon(polygon)

# 2. Conexión OpenEO
client = OpenEOClient()
client.connect()

# 3. Descargas
client.download_rgb_batch(selected_dates, bbox, "tifs_rgb")
client.download_scl_batch(selected_dates, bbox, "tifs_scl")

# 4. Procesamiento SCL
processor = SCLProcessor(bad_classes=[3, 8, 9, 10, 11])
valid_dates = processor.filter_dates_by_quality(selected_dates, "tifs_scl")

# 5. Reference Map
stack = processor.load_stack(clean_dates, "tifs_scl")
ref_map, p_water, p_land = processor.build_reference_map_local(stack)

# 6. Mapeo Intermareal
mapper = IntertidalMapper(client)
water_freq = mapper.compute_water_frequency(bbox, time_extent, "water_freq.tif")
intertidal_mask = mapper.create_intertidal_mask(water_freq)

# 7. Análisis de Mareas
analyzer = TideAnalyzer(tide_model)
metrics = analyzer.compare_models(df_comparison)

# 8. Visualización
viz = Visualizer()
viz.plot_rgb_grid(valid_dates, "tifs_rgb", polygon)
viz.plot_reference_map(ref_map)
viz.plot_water_frequency(water_freq, polygon)
viz.plot_tide_timeseries(df_tides, "Gijón", time_extent, lat, lon)
```

---

## 📋 Plan de Migración

### Fase 1: Estructura Base ✅
- [x] Diseñar arquitectura modular
- [ ] Crear carpeta `intertidal/`
- [ ] Crear `__init__.py`
- [ ] Definir estructura de clases

### Fase 2: Módulos Básicos
- [ ] Implementar `geometry.py` (GeometryProcessor)
- [ ] Implementar `raster.py` (RasterProcessor)
- [ ] Implementar `openeo_client.py` (OpenEOClient)

### Fase 3: Procesamiento Avanzado
- [ ] Implementar `scl_processor.py` (SCLProcessor)
- [ ] Implementar `mapper.py` (IntertidalMapper)

### Fase 4: Análisis y Visualización
- [ ] Implementar `tide_analyzer.py` (TideAnalyzer)
- [ ] Implementar `visualization.py` (Visualizer)

### Fase 5: Refactorización Notebook
- [ ] Adaptar notebook a nueva arquitectura
- [ ] Eliminar imports de `utils`
- [ ] Usar solo `from intertidal import *`

### Fase 6: Limpieza
- [ ] Eliminar `utils.py`
- [ ] Eliminar `intertidal_toolkit.py` (obsoleto)
- [ ] Actualizar documentación

---

## ✅ Ventajas de Esta Arquitectura

1. **Separación de responsabilidades**: Cada módulo tiene un propósito claro
2. **Reutilizable**: Las clases se pueden usar en otros proyectos
3. **Testeable**: Fácil escribir tests unitarios por módulo
4. **Escalable**: Agregar funcionalidad sin modificar código existente
5. **Mantenible**: Cambios localizados en archivos específicos
6. **Documentable**: Cada módulo tiene su propia documentación
7. **Importación simple**: `from intertidal import *`

---

## 📝 Notas de Diseño

- **Métodos estáticos vs instancia**: GeometryProcessor y Visualizer usan métodos estáticos (stateless), mientras que OpenEOClient, SCLProcessor, etc. mantienen estado
- **Composición**: IntertidalMapper recibe OpenEOClient como dependencia
- **Encapsulación**: Métodos privados (`_`) para helpers internos
- **Type hints**: Todas las funciones tienen anotaciones de tipo
- **Docstrings**: Formato Google/NumPy para documentación
