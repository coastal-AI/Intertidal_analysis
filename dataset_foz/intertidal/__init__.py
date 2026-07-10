"""
Intertidal Analysis Toolkit
============================

Biblioteca modular para análisis de zonas intermareales usando Sentinel-2 y OpenEO.

Módulos:
--------
- geometry: GeometryProcessor - Operaciones con coordenadas DMS, polígonos, bbox
- raster: RasterProcessor - Lectura/escritura GeoTIFF, normalización
- openeo_client: OpenEOClient - Interfaz con Copernicus Dataspace
- scl_processor: SCLProcessor - Análisis de calidad SCL y filtrado
- mapper: IntertidalMapper - Mapeo de zona intermareal (water frequency)
- tide_analyzer: TideAnalyzer - Análisis de datos mareales
- tidemodel: PyTMDTideModel, CopernicusTideModel - Modelos de marea
- tide_metrics: Métricas de calidad para distribuciones de mareas (waterline method)
- visualization: Visualizer - Generación de gráficos y mapas

Ejemplo de uso:
---------------
>>> from intertidal import GeometryProcessor, OpenEOClient, SCLProcessor
>>> 
>>> # Crear polígono de AOI
>>> geo = GeometryProcessor()
>>> polygon = geo.make_polygon(aoi_dms_list)
>>> bbox = geo.bbox_from_polygon(polygon)
>>> 
>>> # Conectar y descargar datos
>>> client = OpenEOClient()
>>> client.connect()
>>> client.download_rgb("2024-07-02", bbox, "tifs_rgb")
>>> 
>>> # Analizar calidad
>>> processor = SCLProcessor()
>>> stats = processor.compute_stats(scl_array)
>>> 
>>> # Evaluar distribución de mareas para modelado batimétrico
>>> from intertidal import calcular_metricas_completas, evaluar_calidad_distribucion
>>> metricas = calcular_metricas_completas(valores_validos, valores_totales)
>>> evaluacion = evaluar_calidad_distribucion(valores_validos, valores_totales)
"""

from .geometry import GeometryProcessor
from .raster import RasterProcessor
from .openeo_client import OpenEOClient
from .scl_processor import SCLProcessor
from .mapper import IntertidalMapper
from .tide_analyzer import TideAnalyzer
from .visualization import Visualizer
from .tide_metrics import (
    calcular_cobertura_rango_mareal,
    calcular_uniformidad_ks,
    calcular_entropia_shannon,
    calcular_indice_dispersion,
    calcular_estadisticos_gaps,
    calcular_vsr_waterline,
    calcular_representatividad,
    calcular_metricas_completas,
    imprimir_metricas_completas,
    evaluar_calidad_distribucion,
)
from .notebook_compat import (
    CoordinateUtils,
    OpenEOManager,
    download_date_rgb,
    download_date_scl,
    tif_to_rgb,
    tif_to_scl,
    compute_scl_stats,
    load_scl_stack,
    build_reference_map,
    compute_transition_cloud_stats,
    plot_scl_map,
    plot_reference_map,
    plot_rgb_grid,
    plot_water_frequency,
    download_reference_map_openeo,
    load_reference_map_tif,
    evaluate_transition_cloud_coverage_openeo,
    quantify_reference_gain,
    compute_water_frequency_openeo,
    get_water_centroid,
)
from .overpass import get_overpass_times, overpass_hour_utc

__all__ = [
    "GeometryProcessor",
    "RasterProcessor",
    "OpenEOClient",
    "SCLProcessor",
    "IntertidalMapper",
    "TideAnalyzer",
    "Visualizer",
    # Métricas de calidad de distribuciones mareales
    "calcular_cobertura_rango_mareal",
    "calcular_uniformidad_ks",
    "calcular_entropia_shannon",
    "calcular_indice_dispersion",
    "calcular_estadisticos_gaps",
    "calcular_vsr_waterline",
    "calcular_representatividad",
    "calcular_metricas_completas",
    "imprimir_metricas_completas",
    "evaluar_calidad_distribucion",
    # Compatibilidad con notebooks
    "CoordinateUtils",
    "OpenEOManager",
    "download_date_rgb",
    "download_date_scl",
    "tif_to_rgb",
    "tif_to_scl",
    "compute_scl_stats",
    "load_scl_stack",
    "build_reference_map",
    "compute_transition_cloud_stats",
    "plot_scl_map",
    "plot_reference_map",
    "plot_rgb_grid",
    "plot_water_frequency",
    "download_reference_map_openeo",
    "load_reference_map_tif",
    "evaluate_transition_cloud_coverage_openeo",
    "quantify_reference_gain",
    "compute_water_frequency_openeo",
    "get_water_centroid",
    "get_overpass_times",
    "overpass_hour_utc",
]

__version__ = "1.0.0"
__author__ = "Intertidal Analysis Team"
