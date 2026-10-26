"""
geometry.py — Procesamiento de geometrías y coordenadas
========================================================

Módulo para operaciones con coordenadas DMS, polígonos y bounding boxes.
"""

import re
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box


class GeometryProcessor:
    """
    Procesador de operaciones geométricas para análisis intermareal.
    
    Funcionalidades:
    - Conversión de coordenadas DMS a decimal
    - Creación de polígonos desde coordenadas DMS
    - Extracción de bounding boxes para OpenEO
    - Generación de grids regulares
    
    Todos los métodos son estáticos (stateless), no requiere instanciación.
    """
    
    @staticmethod
    def dms_to_coords(coord_str: str) -> tuple[float, float]:
        """
        Convierte una cadena DMS a coordenadas decimales (lon, lat).
        
        Formato esperado: 'LATITUD LONGITUD' separados por espacio
        Ejemplo: '43°35'46.50"N 5°43'40.94"W' → (-5.728039, 43.596250)
        
        Parameters
        ----------
        coord_str : str
            Coordenada en formato DMS (grados°minutos'segundos"hemisferio)
            
        Returns
        -------
        tuple[float, float]
            (lon, lat) en orden Shapely/GeoJSON
            
        Raises
        ------
        ValueError
            Si el formato DMS no es válido
        """
        pattern = (
            r'(\d+)°(\d+)\'([\d.]+)"([NS])\s+'
            r'(\d+)°(\d+)\'([\d.]+)"([EW])'
        )
        m = re.match(pattern, coord_str.strip())
        if not m:
            raise ValueError(f"Formato DMS no válido: '{coord_str}'")
        
        lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = m.groups()
        
        # Conversión DMS → decimal
        lat = float(lat_d) + float(lat_m) / 60 + float(lat_s) / 3600
        lon = float(lon_d) + float(lon_m) / 60 + float(lon_s) / 3600
        
        # Aplicar signo según hemisferio
        if lat_dir == "S":
            lat *= -1
        if lon_dir == "W":
            lon *= -1
        
        return (lon, lat)  # Orden Shapely
    
    @staticmethod
    def dms_to_decimal(dms_str: str) -> tuple[float, float]:
        """
        Convierte DMS a coordenadas decimales (lat, lon).
        
        Similar a dms_to_coords pero devuelve orden geográfico (lat, lon).
        
        Parameters
        ----------
        dms_str : str
            Coordenada en formato DMS
            
        Returns
        -------
        tuple[float, float]
            (lat, lon) en orden geográfico convencional
            
        Raises
        ------
        ValueError
            Si el formato DMS no es válido
        """
        pattern = r'(\d+)°(\d+)\'([\d.]+)"([NS])\s+(\d+)°(\d+)\'([\d.]+)"([EW])'
        m = re.match(pattern, dms_str.strip())
        if not m:
            raise ValueError(f"No se puede parsear: {dms_str}")
        
        lat_d, lat_m, lat_s, lat_hem = m.group(1, 2, 3, 4)
        lon_d, lon_m, lon_s, lon_hem = m.group(5, 6, 7, 8)
        
        lat = float(lat_d) + float(lat_m) / 60 + float(lat_s) / 3600
        lon = float(lon_d) + float(lon_m) / 60 + float(lon_s) / 3600
        
        if lat_hem == "S":
            lat = -lat
        if lon_hem == "W":
            lon = -lon
        
        return (lat, lon)
    
    @staticmethod
    def make_polygon(dms_list: list[str]) -> Polygon:
        """
        Crea un polígono Shapely desde lista de vértices DMS.
        
        Parameters
        ----------
        dms_list : list[str]
            Lista de coordenadas en formato DMS
            Ejemplo: ['43°35'24"N 7°17'6"W', '43°35'24"N 7°12'18"W', ...]
            
        Returns
        -------
        Polygon
            Polígono en WGS84 (EPSG:4326) con orden (lon, lat) interno
            
        Examples
        --------
        >>> geo = GeometryProcessor()
        >>> aoi_dms = [
        ...     '43°35'24.00"N 7°17'6.00"W',
        ...     '43°35'24.00"N 7°12'18.00"W',
        ...     '43°32'24.00"N 7°12'18.00"W',
        ... ]
        >>> polygon = geo.make_polygon(aoi_dms)
        """
        coords = [GeometryProcessor.dms_to_decimal(d) for d in dms_list]
        # Shapely usa (x, y) = (lon, lat)
        return Polygon([(lon, lat) for lat, lon in coords])
    
    @staticmethod
    def bbox_from_polygon(polygon: Polygon) -> dict:
        """
        Extrae bounding box en formato OpenEO.
        
        Parameters
        ----------
        polygon : Polygon
            Polígono Shapely del AOI
            
        Returns
        -------
        dict
            Diccionario con claves: west, south, east, north
            (lon_min, lat_min, lon_max, lat_max en WGS84)
            
        Examples
        --------
        >>> bbox = geo.bbox_from_polygon(polygon)
        >>> # bbox = {"west": -7.3, "south": 43.5, "east": -7.2, "north": 43.6}
        """
        minx, miny, maxx, maxy = polygon.bounds
        return {"west": minx, "south": miny, "east": maxx, "north": maxy}
    
    @staticmethod
    def make_grid(
        polygon: Polygon, 
        cell_size_deg: float = 0.01, 
        crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Divide el bbox del polígono en cuadrícula regular.
        
        Útil para explorar el AOI en bloques más pequeños antes de
        descargas masivas.
        
        Parameters
        ----------
        polygon : Polygon
            Polígono que define el AOI
        cell_size_deg : float, optional
            Tamaño de celda en grados (default: 0.01° ≈ 1 km)
        crs : str, optional
            Sistema de referencia (default: "EPSG:4326")
            
        Returns
        -------
        GeoDataFrame
            Celdas que intersectan el polígono
            
        Examples
        --------
        >>> grid = geo.make_grid(polygon, cell_size_deg=0.005)
        >>> print(f"Grid de {len(grid)} celdas")
        """
        minx, miny, maxx, maxy = polygon.bounds
        
        # Crear arrays de posiciones
        cols = np.arange(minx, maxx, cell_size_deg)
        rows = np.arange(miny, maxy, cell_size_deg)
        
        # Generar celdas rectangulares
        cells = [
            box(x, y, x + cell_size_deg, y + cell_size_deg)
            for x in cols for y in rows
        ]
        
        # Filtrar solo celdas que intersectan el polígono
        grid = gpd.GeoDataFrame(geometry=cells, crs=crs)
        return grid[grid.intersects(polygon)].reset_index(drop=True)
