"""
mapper.py — Mapeo de zona intermareal y water frequency
=======================================================

Módulo para análisis de frecuencia de agua, construcción de máscaras
intermareales y evaluación de cobertura de nubes en zona de transición.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio.features import shapes as raster_shapes


class IntertidalMapper:
    """
    Mapeador de zona intermareal usando water frequency.
    
    Funcionalidades:
    - Cálculo de water frequency vía OpenEO
    - Creación de máscaras intermareales
    - Evaluación de cobertura de nubes en transición
    - Cuantificación de ganancia del filtro de transición
    - Extracción de centroide de zona de agua
    
    Attributes
    ----------
    openeo_client : OpenEOClient
        Cliente OpenEO para operaciones en backend
    
    Examples
    --------
    >>> from intertidal import OpenEOClient, IntertidalMapper
    >>> 
    >>> client = OpenEOClient()
    >>> client.connect()
    >>> mapper = IntertidalMapper(client)
    >>> 
    >>> # Water frequency
    >>> water_freq = mapper.compute_water_frequency(
    ...     bbox,
    ...     ["2024-01-01", "2024-12-31"],
    ...     "water_freq.tif"
    ... )
    >>> 
    >>> # Máscara intermareal
    >>> intertidal_mask = mapper.create_intertidal_mask(water_freq)
    """
    
    def __init__(self, openeo_client=None):
        """
        Inicializa el mapeador intermareal.
        
        Parameters
        ----------
        openeo_client : OpenEOClient, optional
            Cliente OpenEO configurado (necesario para operaciones remotas)
        """
        self.client = openeo_client
    
    def compute_water_frequency(
        self,
        bbox: dict,
        time_extent: list[str],
        output_path: str,
        transition_mask: np.ndarray | None = None
    ) -> str:
        """
        Calcula frecuencia de agua vía OpenEO.
        
        Water frequency = suma(SCL==6) / suma(SCL∈{4,5,6})
        Numerador: veces que el píxel fue agua
        Denominador: observaciones claras (agua, vegetación, suelo)
        
        Parameters
        ----------
        bbox : dict
            Bounding box con west/south/east/north
        time_extent : list[str]
            [inicio, fin] del periodo
        output_path : str
            Ruta donde guardar el GeoTIFF resultado
        transition_mask : ndarray, optional
            Máscara de transición para análisis focalizado
            
        Returns
        -------
        str
            'ok', 'skipped' o 'error: ...'
            
        Notes
        -----
        Esta función debe implementar el UDF de water frequency
        o usar métodos directos de OpenEO. Por ahora retorna
        placeholder.
        
        Examples
        --------
        >>> status = mapper.compute_water_frequency(
        ...     bbox,
        ...     ["2024-01-01", "2024-12-31"],
        ...     "water_freq.tif"
        ... )
        """
        # TODO: Implementar UDF de water frequency similar al reference map
        raise NotImplementedError(
            "compute_water_frequency requiere UDF específico. "
            "Usa la función compute_water_frequency_openeo de utils.py "
            "hasta completar la migración."
        )
    
    def create_intertidal_mask(
        self,
        water_frequency: np.ndarray,
        min_freq: float = 0.10,
        max_freq: float = 0.95
    ) -> np.ndarray:
        """
        Genera máscara binaria de zona intermareal.
        
        Umbraliza el raster de water frequency para extraer
        los píxeles que están parcialmente inundados.
        
        Parameters
        ----------
        water_frequency : ndarray
            Array float (H, W) con valores [0, 1]
        min_freq : float, optional
            Frecuencia mínima para considerar intermareal (default: 0.10)
        max_freq : float, optional
            Frecuencia máxima para considerar intermareal (default: 0.95)
            
        Returns
        -------
        ndarray
            Array bool (H, W) con True = zona intermareal
            
        Examples
        --------
        >>> # Leer water frequency
        >>> from intertidal import RasterProcessor
        >>> raster = RasterProcessor()
        >>> water_freq, _, _ = raster.load_reference_map("water_freq.tif")
        >>> 
        >>> # Crear máscara
        >>> mask = mapper.create_intertidal_mask(
        ...     water_freq,
        ...     min_freq=0.10,
        ...     max_freq=0.95
        ... )
        >>> print(f"Píxeles intermareales: {mask.sum()}")
        """
        return (water_frequency >= min_freq) & (water_frequency <= max_freq)
    
    def get_water_centroid(
        self,
        reference_map: np.ndarray,
        transform,
        crs: str = "EPSG:4326"
    ) -> tuple[float, float]:
        """
        Encuentra centroide de zona de agua estable.
        
        Usa transformada de distancia para localizar el centro
        del canal principal (punto más alejado de la costa).
        
        Parameters
        ----------
        reference_map : ndarray
            Array 2D con valores 0=transición, 1=agua, 2=tierra
        transform : Affine
            Transformada afín píxel → coordenadas
        crs : str, optional
            Sistema de referencia (default: "EPSG:4326")
            
        Returns
        -------
        tuple[float, float]
            (lon, lat) del centroide de agua estable
            
        Examples
        --------
        >>> lon, lat = mapper.get_water_centroid(ref_map, transform)
        >>> print(f"Centroide de agua: ({lon:.4f}, {lat:.4f})")
        """
        # Máscara de agua estable
        water_mask = (reference_map == 1)
        
        if not water_mask.any():
            raise ValueError("No hay píxeles de agua estable en el reference map")
        
        # Transformada de distancia: encuentra el punto más alejado de los bordes
        dist = distance_transform_edt(water_mask)
        
        # Localizar píxel con máxima distancia (centro del canal)
        max_idx = np.unravel_index(dist.argmax(), dist.shape)
        row, col = max_idx
        
        # Convertir índices de píxel a coordenadas geográficas
        lon, lat = transform * (col, row)
        
        return (lon, lat)
    
    @staticmethod
    def _transition_geometry_from_reference_map(
        reference_map: np.ndarray,
        reference_transform,
        reference_crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Convierte zona de transición en geometría vectorial.
        
        Extrae los píxeles donde reference_map == 0 y los
        convierte en un polígono o multipolígono.
        
        Parameters
        ----------
        reference_map : ndarray
            Reference map con valores 0/1/2
        reference_transform : Affine
            Transformada afín del raster
        reference_crs : str, optional
            CRS del reference map (default: "EPSG:4326")
            
        Returns
        -------
        GeoDataFrame
            GeoDataFrame con la geometría de la transición en EPSG:4326
            
        Notes
        -----
        Esta función es útil para aggregate_spatial en OpenEO,
        permitiendo calcular estadísticas solo en la zona intermareal.
        """
        transition_mask = (reference_map == 0)
        
        if not np.any(transition_mask):
            return gpd.GeoDataFrame(geometry=[], crs=reference_crs)
        
        # Convertir máscara a geometrías vectoriales
        geoms = []
        for geom, value in raster_shapes(
            transition_mask.astype(np.uint8),
            mask=transition_mask,
            transform=reference_transform,
        ):
            if value == 1:
                geoms.append(shapely_shape(geom))
        
        if not geoms:
            return gpd.GeoDataFrame(geometry=[], crs=reference_crs)
        
        # Unir todos los polígonos en uno solo
        merged = unary_union(geoms)
        gdf = gpd.GeoDataFrame(geometry=[merged], crs=reference_crs)
        
        # Reproyectar a WGS84 si es necesario
        if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
    
    def evaluate_transition_cloud_coverage(
        self,
        bbox: dict,
        time_extent: list[str],
        reference_map: np.ndarray,
        transform,
        crs: str,
        max_cloud_pct: float = 20
    ) -> dict:
        """
        Evalúa cobertura de nubes en transición vía OpenEO.
        
        Para cada fecha del periodo, descarga el SCL y calcula
        el porcentaje de píxeles nubosos dentro de la zona de
        transición del reference map.
        
        Parameters
        ----------
        bbox : dict
            Bounding box
        time_extent : list[str]
            [inicio, fin] del periodo
        reference_map : ndarray
            Reference map con zona de transición
        transform : Affine
            Transformada del reference map
        crs : str
            CRS del reference map
        max_cloud_pct : float, optional
            Umbral máximo de nubes en transición (default: 20%)
            
        Returns
        -------
        dict
            Diccionario con:
            - dates_valid: fechas con nubes < umbral
            - dates_discarded: fechas con nubes > umbral
            - stats: dict {fecha: {bad_pct, ...}}
            
        Notes
        -----
        Esta función requiere descargar el cubo SCL completo o
        implementar un UDF que calcule las estadísticas en backend.
        Por ahora retorna placeholder.
        """
        # TODO: Implementar evaluación vía OpenEO
        raise NotImplementedError(
            "evaluate_transition_cloud_coverage requiere UDF específico. "
            "Usa evaluate_transition_cloud_coverage_openeo de utils.py "
            "hasta completar la migración."
        )
    
    def quantify_reference_gain(
        self,
        bbox: dict,
        time_extent: list[str],
        reference_map: np.ndarray,
        transform,
        crs: str,
        global_threshold: float = 0.20,
        transition_threshold: float = 0.20
    ) -> dict:
        """
        Cuantifica ganancia del filtro de transición vs global.
        
        Compara cuántas fechas se recuperan usando el filtro
        en transición en lugar del filtro global.
        
        Parameters
        ----------
        bbox : dict
            Bounding box
        time_extent : list[str]
            Periodo a evaluar
        reference_map : ndarray
            Reference map
        transform : Affine
            Transformada
        crs : str
            CRS
        global_threshold : float, optional
            Umbral filtro global (default: 0.20)
        transition_threshold : float, optional
            Umbral filtro transición (default: 0.20)
            
        Returns
        -------
        dict
            Diccionario con estadísticas de ganancia
            
        Notes
        -----
        Placeholder - requiere implementación completa.
        """
        # TODO: Implementar cuantificación
        raise NotImplementedError(
            "quantify_reference_gain requiere implementación completa. "
            "Usa quantify_reference_gain de utils.py hasta migración."
        )
