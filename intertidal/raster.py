"""
raster.py — Procesamiento de archivos raster GeoTIFF
====================================================

Módulo para lectura/escritura de archivos raster, normalización
y operaciones básicas con GeoTIFFs.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from PIL import Image


class RasterProcessor:
    """
    Procesador de operaciones con archivos raster GeoTIFF.
    
    Funcionalidades:
    - Validación de archivos TIFF
    - Lectura de RGB y SCL
    - Normalización por percentiles
    - Guardado de GeoTIFFs
    - Carga de reference maps
    
    Todos los métodos son estáticos (stateless).
    """
    
    @staticmethod
    def is_valid_tif(path: str | Path) -> bool:
        """
        Verifica si un GeoTIFF contiene datos válidos (>0).
        
        Cuando un batch job de OpenEO falla, el fichero puede existir
        pero estar vacío (todos los valores = 0 o NaN). Esta función
        lo detecta antes de intentar procesarlo.
        
        Parameters
        ----------
        path : str or Path
            Ruta al archivo GeoTIFF
            
        Returns
        -------
        bool
            True si hay al menos un píxel con valor > 0
        """
        try:
            with rasterio.open(path) as src:
                data = src.read()
            return bool(data.max() > 0)
        except Exception:
            return False
    
    @staticmethod
    def read_rgb(tif_path: str, normalize: bool = True) -> np.ndarray | None:
        """
        Lee GeoTIFF RGB y devuelve array (H, W, 3) para visualización.
        
        Cada banda se normaliza individualmente por percentiles (2%-98%)
        para mejorar el contraste visual.
        
        Parameters
        ----------
        tif_path : str
            Ruta al GeoTIFF RGB (3 bandas: B04, B03, B02)
        normalize : bool, optional
            Si True, normaliza por percentiles (default: True)
            
        Returns
        -------
        ndarray or None
            Array float32 (H, W, 3) con valores en [0, 1] si OK
            None si el archivo no existe o falla la lectura
            
        Examples
        --------
        >>> raster = RasterProcessor()
        >>> rgb = raster.read_rgb("tifs_rgb/rgb_2024-07-02.tif")
        >>> plt.imshow(rgb)
        """
        if not os.path.exists(tif_path):
            return None
        
        try:
            with rasterio.open(tif_path) as src:
                data = src.read()  # (3, H, W)
            
            if normalize:
                # Normalizar cada banda por separado
                r = RasterProcessor._norm_percentile(data[0].astype(float))
                g = RasterProcessor._norm_percentile(data[1].astype(float))
                b = RasterProcessor._norm_percentile(data[2].astype(float))
                return np.stack([r, g, b], axis=-1).astype(np.float32)
            else:
                # Sin normalizar: mover eje de bandas al final
                return np.moveaxis(data, 0, -1).astype(np.float32)
                
        except Exception as e:
            print(f"   Error leyendo RGB {tif_path}: {e}")
            return None
    
    @staticmethod
    def read_scl(tif_path: str) -> np.ndarray | None:
        """
        Lee GeoTIFF SCL y devuelve array 2D de clases.
        
        Los valores SCL de ESA van de 0 a 11:
            0  = Sin datos            4  = Vegetación
            1  = Saturado/defectuoso  5  = No vegetación
            2  = Sombra topográfica   6  = AGUA
            3  = Sombra de nube       7  = Sin clasificar
            8  = Nube prob. media     9  = Nube prob. alta
            10 = Cirrus              11  = Nieve/hielo
            
        Parameters
        ----------
        tif_path : str
            Ruta al GeoTIFF SCL (1 banda)
            
        Returns
        -------
        ndarray or None
            Array uint8 (H, W) con clases SCL si OK
            None si el archivo no existe
            
        Examples
        --------
        >>> scl = raster.read_scl("tifs_scl/scl_2024-07-02.tif")
        >>> water_mask = (scl == 6)  # Píxeles de agua
        """
        if not os.path.exists(tif_path):
            return None
        
        try:
            with rasterio.open(tif_path) as src:
                return src.read(1).astype(np.uint8)
        except Exception as e:
            print(f"   Error leyendo SCL {tif_path}: {e}")
            return None
    
    @staticmethod
    def save_geotiff(
        path: str,
        data: np.ndarray,
        transform: Affine,
        crs: str | rasterio.crs.CRS,
        dtype: str = "float32",
        nodata: float | None = None
    ):
        """
        Guarda array como GeoTIFF georeferenciado.
        
        Parameters
        ----------
        path : str
            Ruta del archivo de salida
        data : ndarray
            Array 2D o 3D a guardar
        transform : Affine
            Transformada afín (píxel → coordenadas)
        crs : str or CRS
            Sistema de referencia (ej: "EPSG:4326")
        dtype : str, optional
            Tipo de datos (default: "float32")
        nodata : float, optional
            Valor de nodata (default: None)
            
        Examples
        --------
        >>> raster.save_geotiff(
        ...     "output.tif",
        ...     water_freq,
        ...     transform,
        ...     "EPSG:4326"
        ... )
        """
        # Detectar dimensiones
        if data.ndim == 2:
            height, width = data.shape
            count = 1
            data_to_write = data[np.newaxis, :, :]  # (1, H, W)
        elif data.ndim == 3:
            count, height, width = data.shape
            data_to_write = data
        else:
            raise ValueError(f"Array debe ser 2D o 3D, recibido {data.ndim}D")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Escribir GeoTIFF
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress="lzw"
        ) as dst:
            dst.write(data_to_write)
    
    @staticmethod
    def load_reference_map(ref_map_path: str) -> tuple[np.ndarray, Affine, rasterio.crs.CRS]:
        """
        Carga reference map desde GeoTIFF con metadatos.
        
        Se usa después de construir el reference map para cargarlo
        en memoria y usarlo como máscara en análisis posteriores.
        
        Parameters
        ----------
        ref_map_path : str
            Ruta al GeoTIFF del reference map
            
        Returns
        -------
        reference_map : ndarray
            Array uint8 (H, W) con valores:
            0 = transición, 1 = agua, 2 = tierra
        transform : Affine
            Transformada afín (píxel → coordenadas)
        crs : CRS
            Sistema de referencia de coordenadas
            
        Examples
        --------
        >>> ref_map, transform, crs = raster.load_reference_map("reference.tif")
        >>> transition_mask = (ref_map == 0)
        """
        with rasterio.open(ref_map_path) as src:
            data = src.read(1).astype(np.uint8)
            transform = src.transform
            crs = src.crs
        return data, transform, crs
    
    @staticmethod
    def _norm_percentile(band: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
        """
        Normaliza banda espectral al rango [0, 1] usando percentiles.
        
        Los valores de reflectancia de Sentinel-2 van de 0 a ~10000.
        Usando percentiles p2-p98 se ignora el 2% más oscuro y el 2%
        más brillante, produciendo mejor contraste visual.
        
        Parameters
        ----------
        band : ndarray
            Array 2D con valores de reflectancia (puede haber NaN)
        pmin : float, optional
            Percentil inferior (default: 2%)
        pmax : float, optional
            Percentil superior (default: 98%)
            
        Returns
        -------
        ndarray
            Array float64 en [0, 1], misma forma que band
        """
        # Máscara de píxeles válidos
        valid = np.isfinite(band) & (band > 0)
        
        if valid.sum() == 0:
            return np.zeros_like(band)
        
        # Calcular percentiles solo sobre píxeles válidos
        vmin = np.nanpercentile(band[valid], pmin)
        vmax = np.nanpercentile(band[valid], pmax)
        
        if vmax <= vmin:
            return np.zeros_like(band)
        
        # Estiramiento lineal
        out = (band - vmin) / (vmax - vmin)
        return np.clip(out, 0, 1)
    
    @staticmethod
    def convert_tifs_to_png(
        df,
        rgb_dir,
        scl_dir,
        output_dir,
        percentile_low=2,
        percentile_high=98,
        compress_level=6
    ):
        """
        Convierte imágenes GeoTIFF a PNG normalizado para modelos generativos.
        
        Calcula automáticamente el % de cobertura del tile (píxeles válidos vs NaN)
        y añade la columna 'cobertura_tile_pct' al DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columna 'fecha'
        rgb_dir : Path
            Directorio con archivos rgb_{date}.tif
        scl_dir : Path
            Directorio con archivos scl_{date}.tif
        output_dir : Path
            Directorio base para salida
        percentile_low : float, optional
            Percentil inferior para contrast stretching (default: 2)
        percentile_high : float, optional
            Percentil superior para contrast stretching (default: 98)
        compress_level : int, optional
            Nivel de compresión PNG 0-9 (default: 6)
        
        Returns
        -------
        pd.DataFrame
            DataFrame actualizado con rutas PNG y columna 'cobertura_tile_pct'
        """
        # Paleta de colores SCL oficial ESA
        SCL_COLORS = {
            0:  [0, 0, 0],          # Sin datos (negro)
            1:  [255, 0, 0],        # Saturado/Defectuoso (rojo)
            2:  [47, 47, 47],       # Sombra oscura (gris oscuro)
            3:  [100, 50, 0],       # Sombra de nube (marrón oscuro)
            4:  [0, 160, 0],        # Vegetación (verde)
            5:  [255, 230, 90],     # No vegetación (amarillo)
            6:  [0, 0, 255],        # Agua (azul)
            7:  [128, 128, 128],    # Incierto (gris)
            8:  [192, 192, 192],    # Nube media (gris claro)
            9:  [255, 255, 255],    # Nube alta (blanco)
            10: [100, 200, 255],    # Cirrus (azul claro)
            11: [255, 150, 255],    # Nieve/Hielo (rosa)
        }
        
        # Crear directorios
        png_rgb_dir = output_dir / "rgb_png"
        png_scl_dir = output_dir / "scl_png"
        png_rgb_dir.mkdir(exist_ok=True)
        png_scl_dir.mkdir(exist_ok=True)
        
        print("Convirtiendo imágenes TIF → PNG normalizado...\n")
        
        # Verificar si ya existe columna de cobertura (evitar recálculo)
        cobertura_exists = 'cobertura_tile_pct' in df.columns
        if cobertura_exists:
            print("💡 Columna 'cobertura_tile_pct' ya existe, se preservará\n")
        
        conversion_results = []
        
        for i, row in df.iterrows():
            date = row['fecha']
            
            tif_rgb = rgb_dir / f"rgb_{date}.tif"
            png_rgb = png_rgb_dir / f"rgb_{date}.png"
            tif_scl = scl_dir / f"scl_{date}.tif"
            png_scl = png_scl_dir / f"scl_{date}.png"
            
            try:
                # ----- RGB: Normalización con contrast stretching -----
                # Solo calcular cobertura si no existe ya en el DataFrame
                if cobertura_exists:
                    cobertura_pct = row['cobertura_tile_pct']
                else:
                    cobertura_pct = 0.0
                
                with rasterio.open(tif_rgb) as src:
                    rgb = np.dstack([src.read(i) for i in [1, 2, 3]])
                    
                    valid_pixels = ~np.isnan(rgb).any(axis=2)
                    valid_count = valid_pixels.sum()
                    
                    # Calcular cobertura solo si no existía previamente
                    if not cobertura_exists:
                        total_pixels = valid_pixels.size
                        cobertura_pct = (valid_count / total_pixels) * 100 if total_pixels > 0 else 0.0
                    
                    if valid_count > 0:
                        p_low, p_high = np.percentile(
                            rgb[valid_pixels], 
                            (percentile_low, percentile_high)
                        )
                        rgb_stretched = np.clip(
                            (rgb - p_low) / (p_high - p_low), 0, 1
                        )
                        rgb_uint8 = (rgb_stretched * 255).astype(np.uint8)
                        rgb_uint8[~valid_pixels] = [0, 0, 0]
                        
                        Image.fromarray(rgb_uint8, mode='RGB').save(
                            png_rgb, compress_level=compress_level
                        )
                
                # ----- SCL: Mapeo categórico a colores ESA -----
                with rasterio.open(tif_scl) as src:
                    scl = src.read(1)
                    scl_rgb = np.zeros((*scl.shape, 3), dtype=np.uint8)
                    
                    for class_id, color in SCL_COLORS.items():
                        scl_rgb[scl == class_id] = color
                    
                    scl_rgb[np.isnan(scl)] = SCL_COLORS[0]
                    
                    Image.fromarray(scl_rgb, mode='RGB').save(
                        png_scl, compress_level=compress_level
                    )
                
                conversion_results.append({
                    'date': date,
                    'rgb_png': str(png_rgb.relative_to(output_dir)),
                    'scl_png': str(png_scl.relative_to(output_dir)),
                    'cobertura_tile_pct': round(cobertura_pct, 2)
                })
                
                print(f"  ✓ {date} ({cobertura_pct:.1f}% cobertura)")
                
            except Exception as e:
                print(f"  ✗ {date} - Error: {e}")
        
        print(f"\n✓ Convertidas {len(conversion_results)} fechas a PNG")
        print(f"\nDirectorios PNG:")
        print(f"  RGB: {png_rgb_dir}")
        print(f"  SCL: {png_scl_dir}")
        
        # Actualizar DataFrame
        df_updated = df.copy()
        df_updated['imagen_rgb_tif'] = df_updated['imagen_rgb'].copy()
        df_updated['imagen_scl_tif'] = df_updated['imagen_scl'].copy()
        df_updated['imagen_rgb'] = [f"rgb_png/rgb_{d}.png" for d in df_updated['fecha']]
        df_updated['imagen_scl'] = [f"scl_png/scl_{d}.png" for d in df_updated['fecha']]
        
        # Añadir o actualizar columna de cobertura del tile
        if not cobertura_exists:
            cobertura_dict = {r['date']: r['cobertura_tile_pct'] for r in conversion_results}
            df_updated['cobertura_tile_pct'] = df_updated['fecha'].map(cobertura_dict)
        # Si ya existe, se preserva automáticamente en el copy()
        
        # Estadísticas de cobertura
        print(f"\n📊 Estadísticas de cobertura del tile:")
        print(f"  Media:   {df_updated['cobertura_tile_pct'].mean():>6.2f}%")
        print(f"  Mínima:  {df_updated['cobertura_tile_pct'].min():>6.2f}%")
        print(f"  Máxima:  {df_updated['cobertura_tile_pct'].max():>6.2f}%")
        
        parciales = (df_updated['cobertura_tile_pct'] < 95).sum()
        if parciales > 0:
            print(f"  ⚠️  {parciales} imágenes con cobertura <95% (tiles parciales)")
        
        return df_updated
