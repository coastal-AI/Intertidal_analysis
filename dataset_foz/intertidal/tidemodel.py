import os
import pyTMD
import numpy as np
from pyTMD.datasets import fetch_gsfc_got, fetch_aviso_fes
from datetime import datetime
import copernicusmarine
import pandas as pd


class PyTMDTideModel:
    """
    Interfaz unificada para modelos de marea de PyTMD.
    
    Modelos soportados:
    - GOT4.10: Global Ocean Tide 4.10 (GSFC)
    - FES2022: Finite Element Solution 2022 (AVISO/CNES) - requiere credenciales
    - FES2014: Finite Element Solution 2014 (AVISO/CNES) - requiere credenciales
    
    Para modelos FES, configurar variables de entorno:
        PYTMD_FES_USER=tu_usuario
        PYTMD_FES_PASSWORD=tu_contraseña
    
    Registrarse en: https://www.aviso.altimetry.fr/en/data/data-access.html
    """
    
    SUPPORTED_MODELS = {
        'GOT4.10': {'provider': 'GSFC', 'requires_auth': False},
        'FES2022': {'provider': 'AVISO', 'requires_auth': True},
        'FES2014': {'provider': 'AVISO', 'requires_auth': True},
    }
    
    def __init__(self, model_name="GOT4.10", directory="./tide_models",
                 box_size=0.4, resolution=0.05):
        """
        Args:
            model_name: Name of the tide model to use (ver SUPPORTED_MODELS)
            directory: Directory where tide model data is stored.
            box_size: Half-width of the bounding box in degrees
                      (box spans [lon-box_size, lat-box_size] to
                               [lon+box_size, lat+box_size]).
            resolution: Grid resolution in degrees for sampling.
        """
        if model_name not in self.SUPPORTED_MODELS:
            available = ', '.join(self.SUPPORTED_MODELS.keys())
            raise ValueError(f"Modelo '{model_name}' no soportado. Disponibles: {available}")
        
        self.model_name = model_name
        self.directory = directory
        self.box_size = box_size
        self.resolution = resolution
        self.model_path = os.path.join(directory, model_name)
        self._download_model()

    def _download_model(self):
        """Descarga el modelo de marea si no existe en caché."""
        matches = False
        # check if any directory in the cache starts with the model name
        if os.path.isdir(self.directory):
            subdirs = os.listdir(self.directory)
            # Verificar existencia según modelo específico
            model_dirs = {
                'GOT4.10': 'GOT4.10c',
                'FES2022': 'fes2022b',
                'FES2014': 'fes2014',
            }
            expected_dir = model_dirs.get(self.model_name)
            if expected_dir and expected_dir in subdirs:
                matches = True

        if not matches:
            model_info = self.SUPPORTED_MODELS[self.model_name]
            
            # Verificar credenciales para modelos FES
            if model_info['requires_auth']:
                if not os.environ.get('PYTMD_FES_USER') or not os.environ.get('PYTMD_FES_PASSWORD'):
                    raise EnvironmentError(
                        f"Modelo {self.model_name} requiere credenciales AVISO.\n"
                        "Configurar variables de entorno:\n"
                        "  PYTMD_FES_USER=tu_usuario\n"
                        "  PYTMD_FES_PASSWORD=tu_contraseña\n"
                        "Registrarse en: https://www.aviso.altimetry.fr/en/data/data-access.html"
                    )
            
            print(f"Descargando modelo {self.model_name} ({model_info['provider']})...")
            
            if model_info['provider'] == 'GSFC':
                fetch_gsfc_got(
                    model=self.model_name,
                    directory=self.directory,
                    format="netcdf",
                    compressed=False
                )
            elif model_info['provider'] == 'AVISO':
                fetch_aviso_fes(
                    model=self.model_name,
                    directory=self.directory,
                    user=os.environ.get('PYTMD_FES_USER'),
                    password=os.environ.get('PYTMD_FES_PASSWORD'),
                    compressed=True
                )
        else:
            print(f"Modelo {self.model_name} ya descargado")

    def get_tide_height(self, lat, lon, dt):
        """
        Devuelve la marea en metros en el punto más cercano al dado
        que no sea NaN, dentro del bounding box definido en el constructor.

        Args:
            lat: Latitud del punto de referencia.
            lon: Longitud del punto de referencia.
            dt: datetime para el que calcular la marea.

        Returns:
            float: Altura de la marea en metros.

        Raises:
            ValueError: Si no se definieron lat/lon en el constructor
                        o si no hay puntos válidos no-NaN en la malla.
        """
        if lat is None or lon is None:
            raise ValueError("lat and long cannot be empty")

        min_lon = lon - self.box_size
        min_lat = lat - self.box_size
        max_lon = lon + self.box_size
        max_lat = lat + self.box_size

        lat_lines = np.arange(min_lat, max_lat + self.resolution, self.resolution)
        lon_lines = np.arange(min_lon, max_lon + self.resolution, self.resolution)
        lons_2d, lats_2d = np.meshgrid(lon_lines, lat_lines)
        flat_lats = lats_2d.ravel()
        flat_lons = lons_2d.ravel()

        model_name = (
            f"{self.model_name}_nc"
            if self.model_name == "GOT4.10"
            else self.model_name
        )

        time = np.array([np.datetime64(dt)] * len(flat_lats), dtype="datetime64[ns]")
        tide = pyTMD.compute.tide_elevations(
            flat_lons,
            flat_lats,
            time,
            model=model_name,
            directory=self.directory,
            crs="4326",
            standard="datetime",
        )
        heights = tide.values

        # descartar NaN
        valid = ~np.isnan(heights)
        if not np.any(valid):
            raise ValueError(
                "No valid tide data found in the bounding box. "
                "Try a larger box_size or a different model."
            )

        # distancia euclídea al cuadrado desde (lat, lon) a cada punto válido
        dx = flat_lons[valid] - lon
        dy = flat_lats[valid] - lat
        dist_sq = dx * dx + dy * dy
        best_idx = np.nanargmin(dist_sq)

        # mapear de vuelta al índice original
        original_idx = np.flatnonzero(valid)[best_idx]
        return float(heights[original_idx])

    def get_tide_heights_batch(self, lat: float, lon: float, datetimes: list) -> list[float]:
        """
        Calcula la altura de marea para una lista de datetimes en un único
        punto, usando una sola llamada vectorizada a pyTMD.

        Es ~N veces más rápido que llamar a get_tide_height N veces porque:
          1. Busca el punto válido más cercano UNA SOLA VEZ (con la primera fecha).
          2. Llama a pyTMD.compute.tide_elevations UNA SOLA VEZ con todos los tiempos.

        Parameters
        ----------
        lat, lon : float
            Coordenadas del punto de interés.
        datetimes : list[datetime]
            Lista de datetimes para los que calcular la marea.

        Returns
        -------
        list[float]
            Alturas de marea en metros, en el mismo orden que datetimes.
        """
        model_name = f"{self.model_name}_nc" if self.model_name == "GOT4.10" else self.model_name

        # ── Paso 1: encontrar el punto oceánico válido más cercano ──────────
        # Malla mínima: expandimos desde 0.2° hasta box_size si hace falta.
        best_lat, best_lon = None, None
        for probe_box in (0.2, 0.5, 1.0, self.box_size):
            step = self.resolution
            lat_lines = np.arange(lat - probe_box, lat + probe_box + step, step)
            lon_lines = np.arange(lon - probe_box, lon + probe_box + step, step)
            lons_2d, lats_2d = np.meshgrid(lon_lines, lat_lines)
            flat_lats = lats_2d.ravel()
            flat_lons = lons_2d.ravel()

            probe_time = np.array([np.datetime64(datetimes[0])] * len(flat_lats), dtype="datetime64[ns]")
            probe = pyTMD.compute.tide_elevations(
                flat_lons, flat_lats, probe_time,
                model=model_name, directory=self.directory,
                crs="4326", standard="datetime",
            )
            heights_probe = probe.values
            valid = ~np.isnan(heights_probe)
            if np.any(valid):
                dx = flat_lons[valid] - lon
                dy = flat_lats[valid] - lat
                best = np.flatnonzero(valid)[np.argmin(dx * dx + dy * dy)]
                best_lat = float(flat_lats[best])
                best_lon = float(flat_lons[best])
                break

        if best_lat is None:
            raise ValueError("No valid tide data found in the bounding box.")

        # ── Paso 2: calcular todos los tiempos de golpe en ese único punto ──
        times = np.array([np.datetime64(dt) for dt in datetimes], dtype="datetime64[ns]")
        result = pyTMD.compute.tide_elevations(
            np.full(len(times), best_lon),
            np.full(len(times), best_lat),
            times,
            model=model_name, directory=self.directory,
            crs="4326", standard="datetime",
        )
        return [float(h) for h in result.values]


class CopernicusTideModel:
    def __init__(self):
        copernicusmarine.login()
        self.box_size = 0.1 

    def get_tide_heights_batch(self, lat: float, lon: float, datetimes: list) -> list[float]:
        """
        Calcula la altura de marea para una lista de datetimes en un único
        punto, usando una sola descarga del dataset CMEMS.

        Mucho más rápido que llamar get_tide_height N veces porque:
          1. Descarga los datos UNA SOLA VEZ para todo el rango temporal.
          2. Encuentra el pixel oceánico válido UNA SOLA VEZ.
          3. Interpola todos los tiempos en memoria.

        Parameters
        ----------
        lat, lon : float
            Coordenadas del punto de interés.
        datetimes : list[datetime]
            Lista de datetimes para los que calcular la marea.

        Returns
        -------
        list[float]
            Alturas de marea en metros, en el mismo orden que datetimes.
        """
        if not datetimes:
            return []

        times_pd = pd.to_datetime(datetimes)
        start_dt = times_pd.min() - pd.Timedelta(hours=2)
        end_dt = times_pd.max() + pd.Timedelta(hours=2)

        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

        # Descargar dataset una sola vez para todo el rango temporal
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m",
            minimum_longitude=lon - self.box_size,
            maximum_longitude=lon + self.box_size,
            minimum_latitude=lat - self.box_size,
            maximum_latitude=lat + self.box_size,
            start_datetime=start_str,
            end_datetime=end_str,
            variables=["zos"]
        )

        # Encontrar pixel oceánico válido más cercano
        first_step = ds["zos"].isel(time=0)
        valid_mask = first_step.notnull()
        
        lon_grid, lat_grid = np.meshgrid(ds["longitude"].values, ds["latitude"].values)
        valid_lats = lat_grid[valid_mask.values]
        valid_lons = lon_grid[valid_mask.values]

        if len(valid_lats) == 0:
            raise ValueError(f"No valid ocean pixels found within {self.box_size}° of ({lat}, {lon}).")

        distances = np.sqrt((valid_lats - lat)**2 + (valid_lons - lon)**2)
        closest_idx = np.argmin(distances)
        ocean_lat = valid_lats[closest_idx]
        ocean_lon = valid_lons[closest_idx]

        # Extraer serie temporal completa para ese pixel
        nearest_pixel = ds.sel(latitude=ocean_lat, longitude=ocean_lon)

        # Interpolar todos los tiempos de una vez
        interpolated = nearest_pixel.interp(time=times_pd, method="cubic")
        
        return [float(h) for h in interpolated["zos"].values]

    def get_tide_height(self, lat, lon, dt):
        # 1. Compute the temporal window boundaries dynamically
        target_dt = pd.to_datetime(dt)
        start_dt = target_dt - pd.Timedelta(hours=2)
        end_dt = target_dt + pd.Timedelta(hours=2)

        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

        # 2. Open the dataset with the bounding box
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m",
            minimum_longitude=lon - self.box_size,
            maximum_longitude=lon + self.box_size,
            minimum_latitude=lat - self.box_size,
            maximum_latitude=lat + self.box_size,
            start_datetime=start_str,
            end_datetime=end_str,
            variables=["zos"]
        )

        # 3. Find the nearest VALID ocean pixel spatially
        # We look at the first time step to find where valid ocean data lives
        first_step = ds["zos"].isel(time=0)
        
        # Where is it NOT null? (This ignores land mask NaNs)
        valid_mask = first_step.notnull()
        
        # Convert coordinates to arrays matching the grid shape
        lon_grid, lat_grid = np.meshgrid(ds["longitude"].values, ds["latitude"].values)
        
        # Filter grids to only keep valid ocean coordinates
        valid_lats = lat_grid[valid_mask.values]
        valid_lons = lon_grid[valid_mask.values]

        if len(valid_lats) == 0:
            raise ValueError(f"No valid ocean pixels found within a {self.box_size} degree radius of {lat}, {lon}.")

        # Calculate Euclidean distance from your target point to all valid ocean points
        distances = np.sqrt((valid_lats - lat)**2 + (valid_lons - lon)**2)
        closest_idx = np.argmin(distances)
        
        # Extract the precise coordinates of the nearest open-water pixel
        ocean_lat = valid_lats[closest_idx]
        ocean_lon = valid_lons[closest_idx]

        # 4. Extract the time-series for that specific valid pixel
        # Using exact coordinates bypassing the standard .sel landmask bug
        nearest_pixel = ds.sel(latitude=ocean_lat, longitude=ocean_lon)

        # 5. Interpolate to your exact minute using the 4 valid points
        exact_tide = nearest_pixel.interp(time=target_dt, method="cubic")

        # 6. Extract the value safely
        return float(exact_tide["zos"].values)

# -----------------------------
# 3. Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    #Latitud y longitud de la ubicación deseada
    lat, lon = 43.539,-5.402

    #Hora en UTC para la que se desea obtener la marea
    dt = datetime(
        year=2024,
        month=3,
        day=24,
        hour=10,
        minute=44
    )

    tdmModel = PyTMDTideModel()
    h = tdmModel.get_tide_height(lat, lon, dt)
    print(f"{h:.2f} m")

    tdmFesModel = PyTMDTideModel(model_name='FES2022')
    h = tdmFesModel.get_tide_height(lat, lon, dt)
    print(f"{h:.2f} m")

    copernicusModel = CopernicusTideModel()
    h = copernicusModel.get_tide_height(lat, lon, dt)
    print(f"{h:.2f} m")