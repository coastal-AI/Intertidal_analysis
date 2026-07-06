"""
openeo_client.py — Cliente OpenEO para Copernicus Dataspace
===========================================================

Módulo para interfaz con backend de Copernicus Dataspace vía OpenEO.
Gestiona conexión, autenticación y descargas de datos Sentinel-2.
"""

import os
import requests
import openeo
import numpy as np
import xarray
from scipy.ndimage import binary_dilation


class OpenEOClient:
    """
    Cliente para interactuar con Copernicus Dataspace vía OpenEO.
    
    Funcionalidades:
    - Conexión autenticada con OIDC
    - Descarga de RGB y SCL (individual y batch)
    - Construcción de reference maps vía UDF
    - Listado de fechas disponibles
    
    Examples
    --------
    >>> client = OpenEOClient()
    >>> client.connect()
    >>> client.download_rgb("2024-07-02", bbox, "tifs_rgb")
    >>> client.build_reference_map(bbox, ["2024-01-01", "2024-12-31"], "reference.tif")
    """
    
    def __init__(self, backend_url: str = "openeo.dataspace.copernicus.eu"):
        """
        Inicializa el cliente OpenEO.
        
        Parameters
        ----------
        backend_url : str, optional
            URL del backend de OpenEO (default: Copernicus Dataspace)
        """
        self.backend_url = backend_url
        self.connection = None
    
    def connect(self, use_oidc: bool = True) -> openeo.Connection:
        """
        Conecta al backend de OpenEO con autenticación OIDC.
        
        Abre el navegador para login si es la primera vez.
        Las credenciales se guardan localmente para futuras sesiones.
        
        Parameters
        ----------
        use_oidc : bool, optional
            Si True, usa autenticación OIDC (default: True)
            Si False, conexión sin autenticar (limitado)
            
        Returns
        -------
        Connection
            Objeto de conexión de OpenEO
            
        Examples
        --------
        >>> client = OpenEOClient()
        >>> conn = client.connect()
        >>> print(conn.describe_account())
        """
        self.connection = openeo.connect(self.backend_url)
        
        if use_oidc:
            self.connection.authenticate_oidc()
        
        return self.connection
    
    def get_available_dates(
        self,
        bbox: dict,
        time_extent: list[str],
        max_cloud_cover: int = 100
    ) -> list[str]:
        """
        Lista fechas disponibles de Sentinel-2 L2A para el AOI.
        
        Operación "lazy" (no descarga píxeles, solo consulta metadatos).
        
        Parameters
        ----------
        bbox : dict
            Bounding box con claves: west, south, east, north
        time_extent : list[str]
            [fecha_inicio, fecha_fin] en formato 'YYYY-MM-DD'
        max_cloud_cover : int, optional
            Filtro de cobertura de nubes (default: 100 = todas)
            
        Returns
        -------
        list[str]
            Lista de fechas disponibles en formato 'YYYY-MM-DD'
            
        Examples
        --------
        >>> dates = client.get_available_dates(
        ...     bbox,
        ...     ["2024-01-01", "2024-12-31"],
        ...     max_cloud_cover=100
        ... )
        >>> print(f"Fechas disponibles: {len(dates)}")
        """
        # Consulta el catálogo STAC de Copernicus directamente.
        # Esto es instantáneo: no lanza ningún job Spark en el backend.
        STAC_SEARCH = "https://catalogue.dataspace.copernicus.eu/stac/search"

        dates: set[str] = set()
        body: dict = {
            "collections": ["SENTINEL-2"],
            "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
            "datetime": f"{time_extent[0]}T00:00:00Z/{time_extent[1]}T23:59:59Z",
            "filter": (
                f"s2:product_type='S2MSI2A' AND eo:cloud_cover<={max_cloud_cover}"
            ),
            "filter-lang": "cql2-text",
            "limit": 200,
        }

        url: str | None = STAC_SEARCH
        filter_failed = False
        
        while url:
            try:
                resp = requests.post(url, json=body, timeout=30)
                resp.raise_for_status()
            except requests.HTTPError as e:
                # Si falla con filtro CQL2, intentar sin filtro
                if not filter_failed:
                    print(f"Warning: STAC query failed with filter, retrying without filter...")
                    print(f"  Error: {e}")
                    body = {k: v for k, v in body.items() if k not in ["filter", "filter-lang"]}
                    filter_failed = True
                    resp = requests.post(url, json=body, timeout=30)
                    resp.raise_for_status()
                else:
                    raise
            
            data = resp.json()
            features_count = len(data.get("features", []))
            
            if features_count == 0 and not filter_failed:
                print(f"  ⚠ Warning: 0 features found in STAC response")
                print(f"  Query body: {body}")

            for feature in data.get("features", []):
                # Filtrado manual si el filtro CQL2 falló
                props = feature.get("properties", {})
                if filter_failed:
                    # Solo aceptar S2MSI2A y aplicar filtro de nubes manualmente
                    product_type = props.get("s2:product_type", "")
                    cloud_cover = props.get("eo:cloud_cover", 100)
                    if product_type != "S2MSI2A" or cloud_cover > max_cloud_cover:
                        continue
                
                dt = props.get("datetime", "")
                if dt:
                    dates.add(dt[:10])  # YYYY-MM-DD

            # Paginación: buscar enlace "next"
            next_link = next(
                (lk for lk in data.get("links", []) if lk.get("rel") == "next"),
                None,
            )
            if next_link:
                url = next_link["href"]
                body = {}  # El enlace "next" ya lleva todos los parámetros
            else:
                url = None

        print(f"  → Total fechas encontradas: {len(dates)}")
        return sorted(dates)
    
    def download_rgb(
        self,
        date: str,
        bbox: dict,
        output_dir: str
    ) -> str:
        """
        Descarga RGB (B04+B03+B02) para una fecha.
        
        Lanza batch job en OpenEO que descarga las bandas RGB
        del AOI para la fecha especificada. Idempotente: salta
        si el archivo ya existe.
        
        Parameters
        ----------
        date : str
            Fecha en formato 'YYYY-MM-DD'
        bbox : dict
            Bounding box con west/south/east/north
        output_dir : str
            Directorio donde guardar el GeoTIFF
            
        Returns
        -------
        str
            'ok' si éxito, 'skipped' si ya existe, 'error: ...' si falla
            
        Examples
        --------
        >>> status = client.download_rgb("2024-07-02", bbox, "tifs_rgb")
        >>> if status == "ok":
        ...     print("Descarga exitosa")
        """
        if self.connection is None:
            raise RuntimeError("No conectado. Ejecuta connect() primero.")
        
        out_path = os.path.join(output_dir, f"rgb_{date}.tif")
        
        # Idempotencia
        if os.path.exists(out_path):
            print(f"    {date} — ya existe, saltando")
            return "skipped"
        
        try:
            # Crear directorio si no existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Definir proceso graph
            cube = self.connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox,
                temporal_extent=[date, date],
                bands=["B04", "B03", "B02"],  # R, G, B
                max_cloud_cover=100,
            )
            
            # Reducir dimensión temporal (promedio si hay múltiples tomas)
            cube = cube.reduce_dimension(dimension="t", reducer="mean")
            
            # Crear y ejecutar batch job
            job = cube.save_result(format="GTiff").create_job(title=f"rgb_{date}")
            job.start_and_wait()
            
            # Descargar resultado
            assets = job.get_results().get_assets()
            if not assets:
                return "error: sin assets"
            
            assets[0].download(out_path)
            print(f"   {date} → {out_path}")
            return "ok"
            
        except Exception as e:
            print(f"   {date} — error: {e}")
            return f"error: {e}"
    
    def download_scl(
        self,
        date: str,
        bbox: dict,
        output_dir: str
    ) -> str:
        """
        Descarga SCL (Scene Classification Layer) para una fecha.
        
        Similar a download_rgb pero solo descarga la banda SCL
        (clasificación de píxeles en 12 clases).
        
        Parameters
        ----------
        date : str
            Fecha en formato 'YYYY-MM-DD'
        bbox : dict
            Bounding box con west/south/east/north
        output_dir : str
            Directorio donde guardar el GeoTIFF
            
        Returns
        -------
        str
            'ok', 'skipped' o 'error: ...'
            
        Examples
        --------
        >>> status = client.download_scl("2024-07-02", bbox, "tifs_scl")
        """
        if self.connection is None:
            raise RuntimeError("No conectado. Ejecuta connect() primero.")
        
        out_path = os.path.join(output_dir, f"scl_{date}.tif")
        
        if os.path.exists(out_path):
            print(f"    {date} — SCL ya existe, saltando")
            return "skipped"
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            cube = self.connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox,
                temporal_extent=[date, date],
                bands=["SCL"],
                max_cloud_cover=100,
            )
            
            cube = cube.reduce_dimension(dimension="t", reducer="mean")
            
            job = cube.save_result(format="GTiff").create_job(title=f"scl_{date}")
            job.start_and_wait()
            
            assets = job.get_results().get_assets()
            if not assets:
                return "error: sin assets"
            
            assets[0].download(out_path)
            print(f"   {date} → {out_path}")
            return "ok"
            
        except Exception as e:
            print(f"   {date} — error: {e}")
            return f"error: {e}"
    
    def download_rgb_batch(
        self,
        dates: list[str],
        bbox: dict,
        output_dir: str
    ) -> dict[str, str]:
        """
        Descarga RGB para múltiples fechas.
        
        Ejecuta download_rgb() secuencialmente para cada fecha.
        Para no saturar la cuota del backend gratuito.
        
        Parameters
        ----------
        dates : list[str]
            Lista de fechas en formato 'YYYY-MM-DD'
        bbox : dict
            Bounding box
        output_dir : str
            Directorio de salida
            
        Returns
        -------
        dict[str, str]
            Diccionario {fecha: status} con resultado de cada descarga
            
        Examples
        --------
        >>> results = client.download_rgb_batch(
        ...     ["2024-01-01", "2024-07-02"],
        ...     bbox,
        ...     "tifs_rgb"
        ... )
        >>> print(f"OK: {sum(1 for s in results.values() if s == 'ok')}")
        """
        print(f"Descargando {len(dates)} fechas RGB...\n")
        
        results = {}
        for date in dates:
            print(f"→ {date}")
            results[date] = self.download_rgb(date, bbox, output_dir)
        
        print("\n── Resumen RGB ──")
        for date, status in results.items():
            print(f"  {date}: {status}")
        
        return results
    
    def download_scl_batch(
        self,
        dates: list[str],
        bbox: dict,
        output_dir: str
    ) -> dict[str, str]:
        """
        Descarga SCL para múltiples fechas.
        
        Parameters
        ----------
        dates : list[str]
            Lista de fechas
        bbox : dict
            Bounding box
        output_dir : str
            Directorio de salida
            
        Returns
        -------
        dict[str, str]
            Diccionario {fecha: status}
        """
        print(f"Descargando {len(dates)} fechas SCL...\n")
        
        results = {}
        for date in dates:
            print(f"→ {date}")
            results[date] = self.download_scl(date, bbox, output_dir)
        
        print("\n── Resumen SCL ──")
        for date, status in results.items():
            print(f"  {date}: {status}")
        
        return results
    
    def build_reference_map(
        self,
        bbox: dict,
        time_extent: list[str],
        output_path: str,
        bad_classes: list[int] = None,
        bad_fraction_threshold: float = 0.05,
        stable_threshold: float = 0.95,
        transition_buffer_pixels: int = 5,
        force: bool = False
    ) -> str:
        """
        Construye reference map ejecutando UDF en backend OpenEO.
        
        Procesa un año completo de escenas SCL en el servidor y devuelve
        un solo raster clasificado (agua/tierra/transición).
        
        El UDF filtra escenas limpias, cuenta votos agua/tierra por píxel,
        clasifica según estabilidad y aplica buffer espacial.
        
        Parameters
        ----------
        bbox : dict
            Bounding box del AOI
        time_extent : list[str]
            [inicio, fin] del periodo de referencia (ej: año completo)
        output_path : str
            Ruta donde guardar el GeoTIFF resultante
        bad_classes : list[int], optional
            Clases SCL consideradas malas (default: [3,8,9,10,11])
        bad_fraction_threshold : float, optional
            Máximo % de píxeles malos para usar escena (default: 0.05 = 5%)
        stable_threshold : float, optional
            Umbral P_water para clasificar como estable (default: 0.95)
        transition_buffer_pixels : int, optional
            Píxeles de buffer alrededor de transición (default: 5)
        force : bool, optional
            Si True, recalcula aunque el archivo exista (default: False)
            
        Returns
        -------
        str
            'ok', 'skipped' o 'error: ...'
            
        Examples
        --------
        >>> status = client.build_reference_map(
        ...     bbox,
        ...     ["2024-01-01", "2024-12-31"],
        ...     "reference_map.tif",
        ...     stable_threshold=0.95,
        ...     transition_buffer_pixels=10
        ... )
        """
        if self.connection is None:
            raise RuntimeError("No conectado. Ejecuta connect() primero.")
        
        if bad_classes is None:
            bad_classes = [3, 8, 9, 10, 11]
        
        # Idempotencia
        if not force and os.path.exists(output_path):
            print(f"    Reference map ya existe en {output_path}, saltando.")
            return "skipped"
        
        try:
            # Cargar cubo SCL completo del periodo
            cube_scl = self.connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox,
                temporal_extent=time_extent,
                bands=["SCL"],
                max_cloud_cover=100,
            )
            
            # Crear UDF con código y parámetros
            udf_code = self._get_reference_udf_code()
            udf = openeo.UDF(
                code=udf_code,
                runtime="Python",
                context={
                    "stable_threshold": stable_threshold,
                    "bad_fraction_threshold": bad_fraction_threshold,
                    "bad_classes": bad_classes,
                    "transition_buffer_pixels": transition_buffer_pixels,
                },
            )
            
            # Reducir dimensión temporal con el UDF
            ref_map_cube = cube_scl.reduce_dimension(
                dimension="t",
                reducer=udf,
            )
            
            # Crear y ejecutar batch job
            job = (
                ref_map_cube
                .save_result(format="GTiff")
                .create_job(title="reference_map_scl")
            )
            
            print(
                f"  Lanzando batch job reference map "
                f"({time_extent[0]} → {time_extent[1]})..."
            )
            
            job.start_and_wait()
            
            # Descargar resultado
            assets = job.get_results().get_assets()
            if not assets:
                return "error: sin assets en el resultado"
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            assets[0].download(output_path)
            print(f"  Reference map descargado → {output_path}")
            return "ok"
            
        except Exception as e:
            print(f"  Error en batch job reference map: {e}")
            return f"error: {e}"
    
    @staticmethod
    def _get_reference_udf_code() -> str:
        """
        Retorna código Python del UDF para construir reference map.
        
        Este UDF se ejecuta en el backend de OpenEO y procesa
        todo el cubo SCL temporal para generar el reference map.
        
        Returns
        -------
        str
            Código Python completo del UDF
        """
        return '''
import numpy as np
import xarray
from scipy.ndimage import binary_dilation

def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:
    """UDF para construir reference map desde stack SCL."""
    
    # Leer parámetros de configuración
    stable_threshold = context.get("stable_threshold", 0.95)
    bad_fraction_threshold = context.get("bad_fraction_threshold", 0.05)
    bad_classes = context.get("bad_classes", [3, 8, 9, 10, 11])
    transition_buffer_pixels = context.get("transition_buffer_pixels", 5)
    
    # Extraer array: (t, bands, y, x) o (t, y, x)
    arr = cube.values
    
    # Eliminar dimensión de bandas si existe (SCL es monobanda)
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]  # -> (t, y, x)
    
    n_pixels = arr.shape[1] * arr.shape[2]
    
    # Paso 1: Filtrar fechas limpias
    keep = []
    for t in range(arr.shape[0]):
        bad_frac = np.isin(arr[t], bad_classes).sum() / n_pixels
        if bad_frac <= bad_fraction_threshold:
            keep.append(t)
    
    h, w = arr.shape[1:]
    
    if len(keep) == 0:
        # Sin fechas limpias → todo transición (conservador)
        reference_map = np.zeros((h, w), dtype=np.uint8)
    else:
        # Quedarse solo con fechas limpias
        arr = arr[keep]
        
        # Paso 2: Votar por tipo de superficie
        water_votes = np.sum(arr == 6, axis=0)
        land_votes = np.sum(np.isin(arr, [4, 5]), axis=0)
        
        valid_votes = water_votes + land_votes
        valid_votes[valid_votes == 0] = 1  # Evitar división por cero
        
        P_water = water_votes / valid_votes
        P_land = land_votes / valid_votes
        
        # Paso 3: Clasificar píxeles
        water_stable = P_water >= stable_threshold
        land_stable = P_land >= stable_threshold
        transition = ~(water_stable | land_stable)
        
        # Paso 4: Buffer espacial
        if transition_buffer_pixels > 0:
            transition = binary_dilation(
                transition, iterations=transition_buffer_pixels
            )
        
        # Paso 5: Construir mapa final
        reference_map = np.zeros((h, w), dtype=np.uint8)
        reference_map[water_stable] = 1
        reference_map[land_stable] = 2
        reference_map[transition] = 0  # Sobrescribe
    
    # Devolver como DataArray 3D
    return xarray.DataArray(
        reference_map[np.newaxis, :, :],
        dims=["bands", "y", "x"],
        coords={
            "bands": ["reference_map"],
            "y": cube.coords["y"],
            "x": cube.coords["x"],
        },
    )
'''
