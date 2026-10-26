"""scl_processor.py — Procesamiento y análisis de SCL (Scene Classification Layer)
===============================================================================

Módulo para análisis de calidad de escenas Sentinel-2 usando la banda SCL,
filtrado de fechas, construcción de reference maps locales y estadísticas.

La Scene Classification Layer (SCL) de Sentinel-2 L2A es una banda que clasifica
cada píxel en 12 categorías: agua, vegetación, nubes, sombras, nieve, etc.
Este módulo proporciona herramientas para:

- Evaluar la calidad de escenas (% píxeles nublados)
- Filtrar fechas por umbral de calidad
- Construir reference maps (clasificación agua/tierra/transición)
- Analizar nubosidad solo en zona intermareal (filtro inteligente)
- Corregir clasificaciones erróneas usando el reference map

Clases SCL (Sentinel-2 L2A)
---------------------------
0:  Sin datos
1:  Saturado/Defectuoso
2:  Sombra oscura
3:  Sombra de nube (MALO)
4:  Vegetación (BUENO)
5:  No vegetación / suelo desnudo (BUENO)
6:  Agua (BUENO)
7:  Incierto
8:  Nube probabilidad media (MALO)
9:  Nube probabilidad alta (MALO)
10: Cirrus (MALO)
11: Nieve/Hielo (MALO)

Clases típicamente consideradas "malas": [3, 8, 9, 10, 11]

Examples
--------
>>> from intertidal import SCLProcessor, RasterProcessor
>>> 
>>> # Inicializar procesador
>>> scl_proc = SCLProcessor(bad_classes=[3, 8, 9, 10, 11])
>>> 
>>> # Analizar calidad de una escena
>>> scl = RasterProcessor.read_scl("scl_2024-07-02.tif")
>>> stats = scl_proc.compute_stats(scl)
>>> print(f"Píxeles malos: {stats['bad_pct']:.1f}%")
>>> 
>>> # Filtrar fechas por calidad
>>> result = scl_proc.filter_dates_by_quality(
...     dates, "tifs_scl", max_bad_fraction=0.20
... )
>>> print(f"Válidas: {len(result['valid'])}")
>>> 
>>> # Construir reference map
>>> stack = scl_proc.load_stack(clean_dates, "tifs_scl")
>>> ref_map, p_water, p_land = scl_proc.build_reference_map_local(stack)
>>> # ref_map: 0=transición, 1=agua, 2=tierra

Notes
-----
El reference map clasifica cada píxel en tres categorías:
- 0 (TRANSICIÓN): Zona intermareal, unas veces agua y otras tierra
- 1 (AGUA ESTABLE): Siempre sumergido (canal principal, mar abierto)
- 2 (TIERRA ESTABLE): Siempre emergido (costa rocosa, playa, dunas)

La construcción usa dos criterios:
1. Temporal: P(agua) calculado a partir de frecuencia en serie temporal
2. Espacial: Buffer alrededor de la transición (franja costera)
"""

import os
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


class SCLProcessor:
    """
    Procesador de análisis de calidad SCL y construcción de reference maps.
    
    La Scene Classification Layer (SCL) de Sentinel-2 L2A clasifica cada
    píxel en 12 clases (agua, vegetación, nubes, sombras, etc.).
    
    Funcionalidades:
    - Cálculo de estadísticas de calidad por escena
    - Filtrado de fechas por umbral de nubosidad
    - Carga de stacks 3D desde múltiples SCLs
    - Construcción de reference maps locales
    - Análisis de nubosidad en zona de transición
    - Corrección de SCL usando reference map
    
    Attributes
    ----------
    bad_classes : list[int]
        Clases SCL consideradas "malas" para análisis
        Default: [3, 8, 9, 10, 11] (sombras, nubes, cirrus, nieve)
    
    Examples
    --------
    >>> processor = SCLProcessor(bad_classes=[3, 8, 9, 10, 11])
    >>> 
    >>> # Analizar calidad de una escena
    >>> stats = processor.compute_stats(scl_array)
    >>> print(f"Píxeles malos: {stats['bad_pct']:.1f}%")
    >>> 
    >>> # Filtrar fechas por calidad
    >>> valid_dates = processor.filter_dates_by_quality(
    ...     selected_dates, "tifs_scl", max_bad_fraction=0.20
    ... )
    >>> 
    >>> # Construir reference map
    >>> stack = processor.load_stack(clean_dates, "tifs_scl")
    >>> ref_map, p_water, p_land = processor.build_reference_map_local(stack)
    """
    
    def __init__(self, bad_classes: list[int] = None):
        """
        Inicializa el procesador SCL.
        
        Parameters
        ----------
        bad_classes : list[int], optional
            Clases SCL consideradas malas
            Default: [3, 8, 9, 10, 11]
            - 3:  Sombra de nube
            - 8:  Nube probabilidad media
            - 9:  Nube probabilidad alta
            - 10: Cirrus
            - 11: Nieve/hielo
        """
        if bad_classes is None:
            bad_classes = [3, 8, 9, 10, 11]
        self.bad_classes = bad_classes
    
    def compute_stats(self, scl_array: np.ndarray) -> dict:
        """
        Calcula estadísticas de calidad de una escena SCL.
        
        Clasifica píxeles como "malos" si su clase SCL está en bad_classes
        y retorna la fracción de píxeles malos junto con histograma de clases.
        
        Parameters
        ----------
        scl_array : ndarray
            Array 2D uint8 con valores SCL (0-11)
            
        Returns
        -------
        dict
            Diccionario con claves:
            - bad_fraction: fracción de píxeles malos [0, 1]
            - bad_pct: porcentaje de píxeles malos [0, 100]
            - class_counts: dict {clase: n_pixels}
            
        Examples
        --------
        >>> scl = processor.read_scl("scl_2024-07-02.tif")
        >>> stats = processor.compute_stats(scl)
        >>> if stats['bad_fraction'] > 0.20:
        ...     print("Escena muy nublada, descartar")
        """
        total = scl_array.size
        bad_mask = np.isin(scl_array, self.bad_classes)
        bad_frac = bad_mask.sum() / total
        
        # Histograma de clases
        unique, counts = np.unique(scl_array, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        
        return {
            "bad_fraction": float(bad_frac),
            "bad_pct": float(bad_frac * 100),
            "class_counts": class_counts,
        }
    
    def filter_dates_by_quality(
        self,
        dates: list[str],
        scl_dir: str,
        max_bad_fraction: float = 0.20
    ) -> dict:
        """
        Filtra fechas por calidad SCL global.
        
        Para cada fecha calcula la fracción de píxeles malos en toda
        la escena. Las fechas con fracción > max_bad_fraction se descartan.
        
        Parameters
        ----------
        dates : list[str]
            Lista de fechas a evaluar (formato 'YYYY-MM-DD')
        scl_dir : str
            Directorio con archivos scl_{date}.tif
        max_bad_fraction : float, optional
            Umbral máximo de píxeles malos (default: 0.20 = 20%)
            
        Returns
        -------
        dict
            Diccionario con claves:
            - valid: lista de fechas que pasan el umbral
            - discarded: lista de fechas descartadas
            - stats: dict {fecha: stats} con estadísticas de todas
            
        Examples
        --------
        >>> result = processor.filter_dates_by_quality(
        ...     selected_dates,
        ...     "tifs_scl",
        ...     max_bad_fraction=0.20
        ... )
        >>> print(f"Válidas: {len(result['valid'])}")
        >>> print(f"Descartadas: {len(result['discarded'])}")
        """
        from .raster import RasterProcessor
        
        stats_dict = {}
        valid = []
        discarded = []
        
        for date in dates:
            scl_path = os.path.join(scl_dir, f"scl_{date}.tif")
            
            # Leer SCL
            scl_arr = RasterProcessor.read_scl(scl_path)
            if scl_arr is None:
                continue
            
            # Calcular estadísticas
            stats = self.compute_stats(scl_arr)
            stats_dict[date] = stats
            
            # Clasificar
            if stats["bad_fraction"] <= max_bad_fraction:
                valid.append(date)
            else:
                discarded.append(date)
        
        return {
            "valid": valid,
            "discarded": discarded,
            "stats": stats_dict,
        }
    
    def load_stack(
        self,
        dates: list[str],
        scl_dir: str
    ) -> np.ndarray:
        """
        Carga múltiples SCLs y los apila en array 3D.
        
        Asume que todos los SCLs tienen la misma forma espacial
        (garantizado si se descargan del mismo bbox).
        
        Parameters
        ----------
        dates : list[str]
            Lista de fechas a cargar
        scl_dir : str
            Directorio con archivos scl_{date}.tif
            
        Returns
        -------
        ndarray
            Array uint8 (T, H, W) donde T = número de fechas
            
        Examples
        --------
        >>> clean_dates = ["2024-01-04", "2024-03-14", "2024-05-28"]
        >>> stack = processor.load_stack(clean_dates, "tifs_scl")
        >>> print(f"Stack shape: {stack.shape}")  # (3, H, W)
        """
        stack = []
        for date in dates:
            path = os.path.join(scl_dir, f"scl_{date}.tif")
            with rasterio.open(path) as src:
                scl = src.read(1)
                stack.append(scl)
        
        return np.stack(stack)
    
    def build_reference_map_local(
        self,
        scl_stack: np.ndarray,
        stable_threshold: float = 0.98,
        coastal_buffer_pixels: int = 20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye reference map desde stack SCL local.
        
        Clasifica cada píxel en:
        - 0: TRANSICIÓN (zona intermareal)
        - 1: AGUA ESTABLE (siempre sumergido)
        - 2: TIERRA ESTABLE (siempre emergido)
        
        Algoritmo:
        1. Contar votos: agua (SCL=6) vs tierra (SCL=4,5)
        2. Calcular P_water = agua / (agua + tierra)
        3. Clasificar estables: P_water ≥ threshold Y nunca tierra
        4. Aplicar buffer espacial alrededor de transición
        
        Parameters
        ----------
        scl_stack : ndarray
            Array (T, H, W) con T fechas de SCL
        stable_threshold : float, optional
            Umbral P_water para clasificar como estable (default: 0.98)
            Un píxel es agua estable si P_water ≥ 0.98 Y nunca fue tierra
        coastal_buffer_pixels : int, optional
            Ancho del buffer espacial en píxeles (default: 20)
            A 20m/píxel: 20 píxeles = 400m de franja extra
            
        Returns
        -------
        reference_map : ndarray
            Array uint8 (H, W) con valores 0/1/2
        P_water : ndarray
            Array float (H, W) con probabilidad de agua [0, 1]
        P_land : ndarray
            Array float (H, W) con probabilidad de tierra [0, 1]
            
        Examples
        --------
        >>> stack = processor.load_stack(clean_dates, "tifs_scl")
        >>> ref_map, p_water, p_land = processor.build_reference_map_local(
        ...     stack,
        ...     stable_threshold=0.95,
        ...     coastal_buffer_pixels=10
        ... )
        >>> 
        >>> # Analizar zona de transición
        >>> transition_mask = (ref_map == 0)
        >>> print(f"Píxeles intermareales: {transition_mask.sum()}")
        """
        # Paso 1: Contar votos por tipo de superficie
        water_votes = np.sum(scl_stack == 6, axis=0)  # (H, W)
        land_votes = np.sum(np.isin(scl_stack, [4, 5]), axis=0)  # (H, W)
        
        # Total de observaciones válidas
        valid_votes = water_votes + land_votes
        valid_votes[valid_votes == 0] = 1  # Evitar división por cero
        
        # Paso 2: Calcular probabilidad de agua
        P_water = water_votes / valid_votes
        
        # Paso 3: Clasificar en estable/transición
        # Un píxel es estable solo si NUNCA apareció del tipo contrario
        water_stable = (P_water >= stable_threshold) & (land_votes == 0)
        land_stable = (P_water <= (1 - stable_threshold)) & (water_votes == 0)
        
        # Todo lo que no es estable es transición
        temporal_transition = ~water_stable & ~land_stable
        
        # Paso 4: Añadir buffer espacial costero
        # Distancia de cada píxel a las zonas estables
        dist_to_water = distance_transform_edt(~water_stable)
        dist_to_land = distance_transform_edt(~land_stable)
        
        # Píxeles cerca de AMBAS zonas estables → franja costera
        coastal_transition = (
            (dist_to_water <= coastal_buffer_pixels)
            & (dist_to_land <= coastal_buffer_pixels)
        )
        
        # Zona de transición final = temporal + espacial
        transition_zone = temporal_transition | coastal_transition
        
        # Paso 5: Construir mapa final
        reference_map = np.zeros(P_water.shape, dtype=np.uint8)
        reference_map[water_stable] = 1
        reference_map[land_stable] = 2
        reference_map[transition_zone] = 0  # Sobrescribe
        
        return reference_map, P_water, 1 - P_water
    
    def compute_transition_stats(
        self,
        scl_array: np.ndarray,
        transition_mask: np.ndarray
    ) -> dict:
        """
        Calcula estadísticas de nubosidad solo en zona de transición.
        
        En lugar de descartar una escena por nubosidad global, evalúa
        solo la nubosidad dentro de la zona intermareal. Una escena puede
        tener 70% de nubes en total pero estar despejada sobre el estuario.
        
        Parameters
        ----------
        scl_array : ndarray
            Array 2D uint8 con valores SCL de la escena
        transition_mask : ndarray
            Array bool 2D donde True = zona de transición
            (obtenido de: reference_map == 0)
            
        Returns
        -------
        dict
            Diccionario con claves:
            - bad_fraction: fracción de píxeles malos en transición [0, 1]
            - bad_pct: porcentaje de píxeles malos en transición
            - n_transition_pixels: total de píxeles en transición
            - n_bad_transition_pixels: píxeles malos en transición
            
        Examples
        --------
        >>> transition_mask = (ref_map == 0)
        >>> stats = processor.compute_transition_stats(scl_arr, transition_mask)
        >>> print(f"Nubes en estuario: {stats['bad_pct']:.1f}%")
        """
        n_transition = transition_mask.sum()
        
        # Si no hay zona de transición, retornar 100% malos
        if n_transition == 0:
            return {
                "bad_fraction": 1.0,
                "bad_pct": 100.0,
                "n_transition_pixels": 0,
                "n_bad_transition_pixels": 0,
            }
        
        # Píxeles malos en toda la escena
        bad_mask = np.isin(scl_array, self.bad_classes)
        
        # Intersección: malos Y en transición
        bad_transition = bad_mask & transition_mask
        
        # Fracción de malos dentro de transición
        bad_fraction = bad_transition.sum() / n_transition
        
        return {
            "bad_fraction": float(bad_fraction),
            "bad_pct": float(bad_fraction * 100),
            "n_transition_pixels": int(n_transition),
            "n_bad_transition_pixels": int(bad_transition.sum()),
        }
    
    def correct_with_reference(
        self,
        scl_arr: np.ndarray,
        reference_map: np.ndarray
    ) -> np.ndarray:
        """
        Corrige clases SCL inciertas usando reference map.
        
        Si un píxel está clasificado como nube/sombra pero el reference
        map indica que esa zona es siempre agua estable, probablemente
        es un error del clasificador → se reasigna a agua (SCL=6).
        
        Lo mismo para tierra estable: nubes sobre tierra que nunca fue
        agua se reasignan a tierra desnuda (SCL=5).
        
        Parameters
        ----------
        scl_arr : ndarray
            Array 2D con valores SCL originales
        reference_map : ndarray
            Array 2D con valores 0=transición, 1=agua, 2=tierra
            
        Returns
        -------
        ndarray
            Array 2D con clases SCL corregidas
            
        Examples
        --------
        >>> corrected_scl = processor.correct_with_reference(scl, ref_map)
        >>> # Píxeles que eran nubes sobre agua estable ahora son agua
        """
        corrected = scl_arr.copy()
        
        # Píxeles clasificados como inciertos (nubes/sombras)
        uncertain = np.isin(corrected, [3, 8, 9, 10])
        
        # Correcciones según reference map
        water_fix = uncertain & (reference_map == 1)  # Nube sobre agua estable
        land_fix = uncertain & (reference_map == 2)   # Nube sobre tierra estable
        
        # Reasignar
        corrected[water_fix] = 6  # Agua
        corrected[land_fix] = 5   # Tierra desnuda
        
        return corrected
