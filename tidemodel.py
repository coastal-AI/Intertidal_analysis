import os
import pyTMD
import numpy as np
from pyTMD.datasets import fetch_gsfc_got, fetch_aviso_fes
from datetime import datetime
import copernicusmarine
import pandas as pd


class PyTMDTideModel:
    def __init__(self, model_name="GOT4.10", directory="./tide_models",
                 box_size=0.4, resolution=0.05):
        """
        Args:
            model_name: Name of the tide model to use.
            directory: Directory where tide model data is stored.
            box_size: Half-width of the bounding box in degrees
                      (box spans [lon-box_size, lat-box_size] to
                               [lon+box_size, lat+box_size]).
            resolution: Grid resolution in degrees for sampling.
        """
        self.model_name = model_name
        self.directory = directory
        self.box_size = box_size
        self.resolution = resolution
        self.model_path = os.path.join(directory, model_name)
        self._download_model()

    def _download_model(self):
        matches = False
        # check if any directory in the cache starts with the model name
        if os.path.isdir(self.directory):
            subdirs = os.listdir(self.directory)
            if (self.model_name == 'FES2022' and 'fes2022b' in subdirs) or (self.model_name == 'GOT4.10' and 'GOT4.10c' in subdirs):
                matches = True

        if not matches:
            if self.model_name.startswith('FES') and not os.environ.get('PYTMD_FES_USER'):
                raise EnvironmentError(
                    "Environment variables PYTMD_FES_USER and PYTMD_FES_PASSWORD "
                    "are required for FES models. Create a .env file with your "
                    "AVISO credentials."
                )
            print(f"Descargando modelo {self.model_name}...")
            if self.model_name.startswith('GOT4'):
                fetch_gsfc_got(
                    model=self.model_name,
                    directory=self.directory,
                    format="netcdf",
                    compressed=False
                )
            elif self.model_name.startswith('FES'):
                fetch_aviso_fes(
                    model=self.model_name,
                    directory=self.directory,
                    user=os.environ.get('PYTMD_FES_USER'),
                    password=os.environ.get('PYTMD_FES_PASSWORD'),
                    compressed=True
                )
            else:
                raise ValueError("The model name is not valid") 
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
            chunks="auto",
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

class CopernicusTideModel:
    def __init__(self):
        copernicusmarine.login()
        self.box_size = 0.1 

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