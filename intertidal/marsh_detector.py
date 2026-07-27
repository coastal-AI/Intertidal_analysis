"""
Detección de vegetación inundada (marismas) en zonas intermareales.

Este módulo proporciona funciones para identificar vegetación halófita inundada
(marismas) que el SCL clasifica erróneamente como vegetación seca. Utiliza índices
espectrales (NDWI, MNDWI) para detectar agua bajo la cubierta vegetal.

Problema:
    En zonas de marisma (Spartina, Salicornia), el SCL puede clasificar píxeles
    como "vegetación" (clase 4) incluso cuando están inundados, causando que el
    water frequency subestime el área inundable.

Solución:
    - NDWI: (Green - NIR) / (Green + NIR) usando B03 y B08
    - MNDWI: (Green - SWIR1) / (Green + SWIR1) usando B03 y B11
    - MNDWI es más sensible porque el SWIR es absorbido por agua

Si un píxel tiene SCL=vegetación Y MNDWI>umbral → Es vegetación inundada (clase 12)
"""

import numpy as np
import rasterio
import tempfile
import os
from typing import Optional, Tuple, Dict, List


def compute_ndwi_mndwi_openeo(
    conn,
    date: str,
    bbox: dict,
    polygon=None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calcula NDWI y MNDWI para una fecha usando OpenEO.
    
    NDWI = (Green - NIR) / (Green + NIR)
        Green: B03 (560 nm)
        NIR:   B08 (842 nm)
        
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
        Green: B03 (560 nm)
        SWIR1: B11 (1610 nm)
    
    MNDWI es más sensible a vegetación inundada porque el SWIR
    es absorbido por el agua pero reflejado por vegetación seca.
    
    Parameters
    ----------
    conn : openeo.Connection
        Conexión autenticada a OpenEO
    date : str
        Fecha 'YYYY-MM-DD'
    bbox : dict
        Bounding box {west, south, east, north}
    polygon : shapely.Polygon, optional
        Polígono para recortar la escena
        
    Returns
    -------
    ndwi : ndarray (H, W) or None
        NDWI [-1, 1], valores altos indican agua
    mndwi : ndarray (H, W) or None
        MNDWI [-1, 1], valores altos indican agua o vegetación inundada
    """
    try:
        # Cargar colección Sentinel-2 L2A
        cube = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=bbox,
            temporal_extent=[date, date],
            bands=["B03", "B08", "B11"]  # Green, NIR, SWIR1
        )
        
        # Recortar a polígono si se proporciona
        if polygon is not None:
            from shapely.geometry import mapping
            cube = cube.filter_spatial(mapping(polygon))
        
        # Calcular NDWI = (B03 - B08) / (B03 + B08)
        b03 = cube.band("B03")
        b08 = cube.band("B08")
        b11 = cube.band("B11")
        
        ndwi = (b03 - b08) / (b03 + b08)
        mndwi = (b03 - b11) / (b03 + b11)
        
        # Descargar NDWI
        with tempfile.NamedTemporaryFile(suffix="_ndwi.tif", delete=False) as tmp_ndwi:
            tmp_ndwi_path = tmp_ndwi.name
        
        with tempfile.NamedTemporaryFile(suffix="_mndwi.tif", delete=False) as tmp_mndwi:
            tmp_mndwi_path = tmp_mndwi.name
        
        # Descargar cada índice por separado como GeoTIFF
        ndwi.download(tmp_ndwi_path, format="GTiff")
        mndwi.download(tmp_mndwi_path, format="GTiff")
        
        # Leer con rasterio
        with rasterio.open(tmp_ndwi_path) as src:
            ndwi_array = src.read(1)
        
        with rasterio.open(tmp_mndwi_path) as src:
            mndwi_array = src.read(1)
        
        # Limpiar archivos temporales
        os.unlink(tmp_ndwi_path)
        os.unlink(tmp_mndwi_path)
        
        return ndwi_array, mndwi_array
        
    except Exception as e:
        print(f"⚠️ Error calculando NDWI/MNDWI para {date}: {e}")
        return None, None


def compute_ndwi_mndwi_batch_openeo(
    conn,
    dates: List[str],
    bbox: dict,
    polygon=None,
    output_dir: Optional[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Calcula NDWI y MNDWI para múltiples fechas en un único job OpenEO.
    
    Descarga el cubo temporal completo y extrae cada fecha del netCDF resultante.
    
    Parameters
    ----------
    conn : openeo.Connection
        Conexión autenticada a OpenEO
    dates : list of str
        Lista de fechas 'YYYY-MM-DD'
    bbox : dict
        Bounding box {west, south, east, north}
    polygon : shapely.Polygon, optional
        Polígono para recortar la escena
    output_dir : str, optional
        No usado, mantiene compatibilidad
        
    Returns
    -------
    results : dict
        Diccionario {fecha: (ndwi, mndwi)} con arrays 2D para cada fecha
    """
    if not dates:
        print("⚠️ No hay fechas para procesar")
        return {}
    
    print(f"📦 Procesando {len(dates)} fechas en un único job OpenEO...")
    print(f"   Rango: {dates[0]} → {dates[-1]}")
    print(f"   Esto puede tardar varios minutos...\n")
    
    try:
        from shapely.geometry import mapping
        import xarray as xr
        import pandas as pd
        
        # Crear rango temporal que incluya todas las fechas
        temporal_extent = [dates[0], dates[-1]]
        
        print(f"   ⏳ Creando cubo de datos Sentinel-2...")
        
        # Cargar colección Sentinel-2 L2A
        cube = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=bbox,
            temporal_extent=temporal_extent,
            bands=["B03", "B08", "B11"]
        )
        
        if polygon is not None:
            cube = cube.filter_spatial(mapping(polygon))
        
        # Calcular índices
        b03 = cube.band("B03")
        b08 = cube.band("B08")
        b11 = cube.band("B11")
        
        ndwi = (b03 - b08) / (b03 + b08)
        mndwi = (b03 - b11) / (b03 + b11)
        
        # Descargar como GeoTIFF en lugar de netCDF (más confiable)
        with tempfile.NamedTemporaryFile(suffix="_ndwi.tif", delete=False) as tmp_ndwi:
            tmp_ndwi_path = tmp_ndwi.name
        
        with tempfile.NamedTemporaryFile(suffix="_mndwi.tif", delete=False) as tmp_mndwi:
            tmp_mndwi_path = tmp_mndwi.name
        
        print(f"   ⬇️ Descargando NDWI como GeoTIFF...")
        ndwi.download(tmp_ndwi_path, format="GTiff")
        
        print(f"   ⬇️ Descargando MNDWI como GeoTIFF...")
        mndwi.download(tmp_mndwi_path, format="GTiff")
        
        print(f"   📖 Leyendo archivos GeoTIFF...")
        
        # Leer con rasterio
        with rasterio.open(tmp_ndwi_path) as src:
            ndwi_data = src.read()
            ndwi_profile = src.profile
        
        with rasterio.open(tmp_mndwi_path) as src:
            mndwi_data = src.read()
            mndwi_profile = src.profile
        
        # Determinar si hay dimensión temporal (primera dimensión > 1)
        has_temporal = ndwi_data.shape[0] > 1
        
        # Extraer arrays para cada fecha
        results = {}
        
        if not has_temporal:
            # CASO 1: Solo una fecha (shape: (1, h, w))
            print(f"   ℹ️ Una sola fecha detectada")
            print(f"   ✓ Procesando fecha única: {dates[0]}")
            
            ndwi_2d = ndwi_data[0]  # Extraer primera banda
            mndwi_2d = mndwi_data[0]
            
            results[dates[0]] = (ndwi_2d, mndwi_2d)
            
        else:
            # CASO 2: Múltiples fechas (shape: (n_dates, h, w))
            n_dates = ndwi_data.shape[0]
            print(f"   ✓ {n_dates} fechas detectadas")
            
            print(f"   📊 Extrayendo datos por fecha...")
            for i, date_str in enumerate(dates[:n_dates]):
                ndwi_2d = ndwi_data[i]
                mndwi_2d = mndwi_data[i]
                results[date_str] = (ndwi_2d, mndwi_2d)
                
                if (i + 1) % 10 == 0:
                    print(f"      ✓ Extraídas {i + 1}/{n_dates} fechas...")
        
        # Limpiar archivos temporales
        os.unlink(tmp_ndwi_path)
        os.unlink(tmp_mndwi_path)
        
        print(f"\n{'='*70}")
        print(f"✅ Procesamiento batch completado")
        print(f"{'='*70}")
        print(f"  Fechas procesadas exitosamente: {len(results)}/{len(dates)}")
        print(f"{'='*70}\n")
        
        return results
        
    except Exception as e:
        print(f"\n⚠️ Error en procesamiento batch: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n⚠️ El job batch falló. La celda no continuará con procesamiento individual.")
        return {}


def detect_flooded_vegetation(
    scl: np.ndarray,
    ndwi: np.ndarray,
    mndwi: np.ndarray,
    ndwi_threshold: float = 0.0,
    mndwi_threshold: float = 0.3,
    use_mndwi: bool = True
) -> np.ndarray:
    """
    Detecta píxeles de vegetación inundada (marismas).
    
    Criterio: píxel clasificado como vegetación (SCL = 4 o 5)
              Y con índice de agua alto (NDWI o MNDWI > umbral)
    
    Parameters
    ----------
    scl : ndarray (H, W)
        Array SCL uint8
    ndwi : ndarray (H, W)
        NDWI [-1, 1]
    mndwi : ndarray (H, W)
        MNDWI [-1, 1]
    ndwi_threshold : float, optional
        Umbral para NDWI (default: 0.0)
    mndwi_threshold : float, optional
        Umbral para MNDWI (default: 0.3)
        MNDWI > 0.3 es típico de agua o vegetación muy húmeda
    use_mndwi : bool, optional
        Si True, usa MNDWI (más sensible); si False, usa NDWI
        
    Returns
    -------
    flooded_veg_mask : ndarray (H, W) bool
        True = vegetación inundada detectada
    """
    # Píxeles clasificados como vegetación o suelo
    is_vegetation = np.isin(scl, [4, 5])
    
    # Índice de agua alto
    if use_mndwi:
        has_water = mndwi > mndwi_threshold
    else:
        has_water = ndwi > ndwi_threshold
    
    # Intersección: vegetación Y agua detectada
    flooded_veg = is_vegetation & has_water
    
    return flooded_veg


def correct_scl_for_marshes(
    scl: np.ndarray,
    flooded_veg_mask: np.ndarray,
    marsh_class: int = 12
) -> np.ndarray:
    """
    Corrige el SCL reclasificando vegetación inundada como nueva clase marisma.
    
    Parameters
    ----------
    scl : ndarray (H, W)
        SCL original uint8
    flooded_veg_mask : ndarray (H, W) bool
        Máscara de vegetación inundada
    marsh_class : int, optional
        Clase SCL para vegetación inundada/marisma (default: 12)
        
    Returns
    -------
    scl_corrected : ndarray (H, W) uint8
        SCL corregido donde vegetación inundada → clase 12 (marisma)
    """
    scl_corrected = scl.copy()
    scl_corrected[flooded_veg_mask] = marsh_class  # Nueva clase: vegetación inundada
    return scl_corrected
