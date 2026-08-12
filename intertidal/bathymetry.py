"""
bathymetry.py — Reconstrucción batimétrica intermareal mediante OpenEO
=====================================================================

Reconstrucción automática de un modelo digital de elevación (DEM)
intermareal a partir de observaciones multitemporales Sentinel-2 y
modelos de marea.

El procesamiento pesado se ejecuta en OpenEO mediante DataCubes y UDFs.
El módulo únicamente orquesta los batch jobs, descarga los resultados
y proporciona utilidades para exportación y visualización.

Pipeline
--------
Sentinel-2 SCL
      │
      ▼
Clasificación agua / tierra
      │
      ▼
Asociación con alturas de marea
      │
      ▼
Reconstrucción DEM
      │
      ▼
Optimización espacial
      │
      ▼
Confidence map
Slope
Aspect
Hillshade
Contours

Examples
--------
>>> reconstructor = BathymetryReconstructor(connection)

>>> result = reconstructor.reconstruct(
...     bbox=bbox,
...     valid_dates=valid_dates,
...     tide_heights=tide_heights,
... )

>>> dem = result.elevation
>>> confidence = result.confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from typing import Optional

import numpy as np
import rasterio
import geopandas as gpd
import openeo
from skimage.measure import find_contours
from shapely.geometry import LineString

from scipy import ndimage


# ---------------------------------------------------------------------
# UDF server-side (OpenEO backend)
# ---------------------------------------------------------------------
#
# NOTA sobre el context: si `tide_heights` crece mucho (varios cientos o
# miles de fechas), algunos backends OpenEO imponen un límite de tamaño
# al payload serializado del UDF context. Si se alcanza ese límite, la
# alternativa es transportar las alturas de marea como una dimensión
# adicional del datacube en vez de vía `context`.
#
# NOTA sobre el runtime: este UDF depende de scipy.ndimage. Verifica que
# el entorno de ejecución Python del backend OpenEO elegido lo incluya;
# no todos los backends lo garantizan por defecto.

# ── LEGACY: UDF de batimetria basado en SCL (agua/tierra por clases SCL). ──
# Se conserva como water_source="scl". El nuevo por defecto es NDWI a 10 m.
_BATHYMETRY_UDF_SCL = '''
import numpy as np
from scipy import ndimage
import xarray as xr
from openeo.udf import XarrayDataCube


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """
    Reconstrucción batimétrica intermareal a partir de un cubo temporal SCL.

    Server-side: clasificación agua/tierra, acotación de elevación mediante
    alturas de marea, optimización basada en energía con preservación de
    taludes y difusión anisótropa tipo Perona-Malik.
    """

    array = cube.get_array()
    array = _drop_band_dimension(array)

    time_dim = _resolve_time_dimension(array)

    spatial_dims = [dim for dim in array.dims if dim != time_dim]
    array = array.transpose(time_dim, *spatial_dims)

    scl = array.values.astype(np.int16)
    dates = _extract_dates(array[time_dim].values)

    tide_heights = context.get("tide_heights") or {}
    valid_dates = context.get("valid_dates")
    water_classes = context.get("water_classes") or [6, 12]
    land_classes = context.get("land_classes") or [4, 5]
    clear_classes = context.get("clear_classes") or [4, 5, 6, 12]
    smooth_lambda = float(context.get("smooth_lambda") or 0.35)
    edge_sigma = float(context.get("edge_sigma") or 0.3)
    diffusion_iterations = int(context.get("diffusion_iterations") or 8)

    # Método de reconstrucción de la elevación:
    #   "optimized" (por defecto): bracketing + optimización + difusión
    #   "pixels":   bracketing puro (punto medio del intervalo por píxel)
    #   "isolines": interpolación desde las líneas de costa (waterlines)
    method = context.get("method") or "optimized"

    tide_array, temporal_mask = _build_temporal_arrays(
        dates=dates,
        tide_heights=tide_heights,
        valid_dates=valid_dates,
    )

    clear_mask = _isin(scl, clear_classes) & temporal_mask[:, None, None]
    water_mask = _isin(scl, water_classes) & clear_mask
    land_mask = _isin(scl, land_classes) & clear_mask

    zmin, zmax, observation_count = _bracket_elevations(
        tide_array=tide_array,
        water_mask=water_mask,
        land_mask=land_mask,
    )

    z0, inconsistent = _initial_estimate(zmin, zmax)

    if method == "pixels":
        # Bracketing puro: la elevación es el punto medio del intervalo, solo
        # en los píxeles acotados por agua y tierra (z0 finito). Sin optimizar.
        elevation = z0.astype(np.float32)

    elif method == "isolines":
        # Interpolación desde las líneas de costa; se restringe a los píxeles
        # acotados (z0 finito) para ser comparable con el método "pixels".
        elevation = _isoline_elevation(
            scl=scl,
            tide_array=tide_array,
            temporal_mask=temporal_mask,
            water_classes=water_classes,
            land_classes=land_classes,
        )
        elevation = np.where(np.isfinite(z0), elevation, np.nan).astype(np.float32)

    else:
        elevation = _optimize_elevation(
            z0=z0,
            valid_mask=~np.isnan(z0),
            smooth_lambda=smooth_lambda,
            edge_sigma=edge_sigma,
        )

        elevation = _anisotropic_diffusion(
            elevation,
            iterations=diffusion_iterations,
            kappa=edge_sigma,
        )

    residual = zmax - zmin

    confidence = _compute_confidence(
        observation_count=observation_count,
        residual=residual,
        inconsistent=inconsistent,
    )

    result = _stack_bands(
        elevation=elevation,
        confidence=confidence,
        residual=residual,
        observation_count=observation_count,
        template=array,
        time_dim=time_dim,
        spatial_dims=spatial_dims,
    )

    return XarrayDataCube(result)


def _drop_band_dimension(array):
    """
    Elimina la dimensión 'bands' cuando el cubo contiene una única banda
    (SCL), preservando el resto de dimensiones intactas.
    """

    if "bands" in array.dims and array.sizes["bands"] == 1:
        array = array.isel(bands=0, drop=True)

    return array


def _resolve_time_dimension(array):
    """
    Identifica el nombre de la dimensión temporal ('t' o 'time').
    """

    for candidate in ("t", "time"):
        if candidate in array.dims:
            return candidate

    raise ValueError("No temporal dimension found in input datacube.")


def _extract_dates(time_values):
    """
    Convierte las coordenadas temporales del cubo en cadenas 'YYYY-MM-DD'.
    """

    dates = []

    for value in time_values:
        as_day = np.datetime64(value, "D")
        dates.append(str(as_day))

    return dates


def _isin(array, classes):
    return np.isin(array, classes)


def _build_temporal_arrays(dates, tide_heights, valid_dates):
    """
    Construye el vector de alturas de marea por fecha de adquisición y la
    máscara temporal de observaciones utilizables (fecha válida y con
    altura de marea conocida).
    """

    n_obs = len(dates)

    tide_array = np.full(n_obs, np.nan, dtype=np.float32)
    temporal_mask = np.zeros(n_obs, dtype=bool)

    valid_dates_set = set(valid_dates) if valid_dates is not None else None

    for index, date in enumerate(dates):

        if valid_dates_set is not None and date not in valid_dates_set:
            continue

        height = tide_heights.get(date)

        if height is None:
            continue

        tide_array[index] = height
        temporal_mask[index] = True

    return tide_array, temporal_mask


def _bracket_elevations(tide_array, water_mask, land_mask):
    """
    Calcula, por píxel:

        zmin: máxima altura de marea observada estando el píxel en tierra
        zmax: mínima altura de marea observada estando el píxel en agua

    junto con el número total de observaciones válidas.
    """

    tide_broadcast = tide_array[:, None, None]

    land_heights = np.where(land_mask, tide_broadcast, -np.inf)
    zmin = np.max(land_heights, axis=0)
    zmin = np.where(np.isfinite(zmin), zmin, np.nan).astype(np.float32)

    water_heights = np.where(water_mask, tide_broadcast, np.inf)
    zmax = np.min(water_heights, axis=0)
    zmax = np.where(np.isfinite(zmax), zmax, np.nan).astype(np.float32)

    observation_count = np.sum(water_mask | land_mask, axis=0).astype(np.float32)

    return zmin, zmax, observation_count


def _initial_estimate(zmin, zmax):
    """
    Estimación inicial z0 = (zmin + zmax) / 2. Marca como inconsistentes
    los píxeles donde zmin > zmax, para penalizar su confianza más tarde.
    """

    with np.errstate(invalid="ignore"):
        inconsistent = zmin > zmax

    z0 = (zmin + zmax) / 2.0

    return z0.astype(np.float32), inconsistent


def _shift(array, dy, dx):
    """
    Desplaza el raster (dy, dx) píxeles reutilizando el valor de borde más
    cercano, mediante scipy.ndimage.shift con interpolación de orden 0.
    """

    return ndimage.shift(array, shift=(-dy, -dx), order=0, mode="nearest")


def _optimize_elevation(z0, valid_mask, smooth_lambda, edge_sigma, iterations=60):
    """
    Optimización basada en energía:

        E = sum((z - z0)^2) + lambda * sum(w_ij * (z_i - z_j)^2)

    con pesos de suavidad w_ij = exp(-delta^2 / sigma^2) que preservan
    taludes pronunciados. Se resuelve mediante iteración tipo Jacobi,
    lo que además permite reconstruir (inpaint) los píxeles sin dato.
    """

    has_valid = np.any(valid_mask)
    fill_value = np.nanmean(z0) if has_valid else 0.0
    z0_filled = np.where(valid_mask, z0, fill_value).astype(np.float32)

    neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    sigma_sq = edge_sigma ** 2 + 1e-6

    weights = []
    for dy, dx in neighbor_offsets:
        neighbor = _shift(z0_filled, dy, dx)
        delta_sq = (z0_filled - neighbor) ** 2
        weights.append(np.exp(-delta_sq / sigma_sq).astype(np.float32))

    data_weight = valid_mask.astype(np.float32)
    z = z0_filled.copy()

    for _ in range(iterations):

        neighbor_sum = np.zeros_like(z)
        weight_sum = np.zeros_like(z)

        for (dy, dx), weight in zip(neighbor_offsets, weights):
            neighbor_z = _shift(z, dy, dx)
            neighbor_sum += weight * neighbor_z
            weight_sum += weight

        numerator = data_weight * z0_filled + smooth_lambda * neighbor_sum
        denominator = data_weight + smooth_lambda * weight_sum
        denominator = np.where(denominator > 1e-6, denominator, 1.0)

        z = numerator / denominator

    return z.astype(np.float32)


def _isoline_elevation(
    scl,
    tide_array,
    temporal_mask,
    water_classes,
    land_classes,
    smooth_lambda=3.0,
    iterations=300,
):
    """
    Reconstrucción por isolíneas (waterlines), server-side.

    Para cada fecha se extrae la línea de costa (píxeles de agua adyacentes a
    tierra y viceversa) y se le asigna la altura de marea de esa fecha. Cada
    orilla es una observación BLANDA de la elevación; el DEM se obtiene
    minimizando por mínimos cuadrados

        sum_i c_i (z_i - obs_i)^2  +  lambda * sum (z_i - z_vecino)^2

    donde c_i es el nº de fechas en que el píxel fue orilla (más peso a los
    más observados). El término de suavidad promedia las orillas en conflicto
    (píxeles vecinos con mareas dispares) y rellena los huecos, dando una
    superficie suave. Se resuelve por iteración de Jacobi (tile-safe).
    """

    n_obs, height, width = scl.shape

    accum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)

    for index in range(n_obs):

        if not temporal_mask[index]:
            continue

        tide = tide_array[index]

        water = np.isin(scl[index], water_classes)
        land = np.isin(scl[index], land_classes)

        shoreline = (water & ndimage.binary_dilation(land)) | (
            land & ndimage.binary_dilation(water)
        )

        accum[shoreline] += tide
        count[shoreline] += 1.0

    has_shoreline = count > 0

    if not has_shoreline.any():
        return np.full((height, width), np.nan, dtype=np.float32)

    # Observación por píxel (media de mareas donde fue orilla) y su peso.
    observation = accum / np.where(count > 0, count, 1.0)
    data_weight = count.astype(np.float64)

    # Inicialización e iteración de Jacobi del sistema de mínimos cuadrados.
    fill = float(observation[has_shoreline].mean())
    z = np.where(has_shoreline, observation, fill).astype(np.float64)

    smooth_lambda = float(smooth_lambda)

    for _ in range(int(iterations)):

        neighbor_sum = (
            _shift(z, -1, 0)
            + _shift(z, 1, 0)
            + _shift(z, 0, -1)
            + _shift(z, 0, 1)
        )

        numerator = data_weight * observation + smooth_lambda * neighbor_sum
        denominator = data_weight + smooth_lambda * 4.0

        z = numerator / denominator

    return z.astype(np.float32)


def _anisotropic_diffusion(elevation, iterations, kappa, step=0.15):
    """
    Difusión anisótropa tipo Perona-Malik. Suaviza terrazas manteniendo
    los taludes pronunciados gracias al coeficiente de conducción
    c = exp(-(gradiente / kappa)^2).
    """

    z = elevation.astype(np.float32).copy()
    kappa = max(float(kappa), 1e-3)

    for _ in range(max(iterations, 0)):

        north = _shift(z, -1, 0) - z
        south = _shift(z, 1, 0) - z
        east = _shift(z, 0, 1) - z
        west = _shift(z, 0, -1) - z

        c_north = np.exp(-(north / kappa) ** 2)
        c_south = np.exp(-(south / kappa) ** 2)
        c_east = np.exp(-(east / kappa) ** 2)
        c_west = np.exp(-(west / kappa) ** 2)

        z = z + step * (
            c_north * north
            + c_south * south
            + c_east * east
            + c_west * west
        )

    return z.astype(np.float32)


def _compute_confidence(
    observation_count,
    residual,
    inconsistent,
    min_observations=4.0,
    residual_scale=1.5,
):
    """
    Mapa de confianza combinando número de observaciones, anchura del
    intervalo (zmax - zmin) y consistencia de la reconstrucción.
    Normalizado en [0, 1].
    """

    observation_term = np.clip(observation_count / min_observations, 0.0, 1.0)

    residual_abs = np.abs(residual)
    residual_term = np.exp(-residual_abs / residual_scale)
    residual_term = np.where(np.isnan(residual), 0.0, residual_term)

    consistency_term = np.where(inconsistent, 0.5, 1.0)

    confidence = observation_term * residual_term * consistency_term
    confidence = np.where(observation_count > 0, confidence, 0.0)

    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _stack_bands(
    elevation,
    confidence,
    residual,
    observation_count,
    template,
    time_dim,
    spatial_dims,
):
    """
    Ensambla el DataArray multibanda de salida (Elevation, Confidence,
    Residual, Observations), conservando las coordenadas espaciales
    originales del cubo de entrada.
    """

    stacked = np.stack([elevation, confidence, residual, observation_count], axis=0)

    coords = {
        name: coordinate
        for name, coordinate in template.coords.items()
        if time_dim not in coordinate.dims
    }
    coords["bands"] = ["Elevation", "Confidence", "Residual", "Observations"]

    result = xr.DataArray(
        stacked,
        dims=["bands"] + list(spatial_dims),
        coords=coords,
    )

    return result
'''


_BATHYMETRY_UDF_NDWI = '''
import numpy as np
from scipy import ndimage
import xarray as xr
from openeo.udf import XarrayDataCube


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """
    Reconstrucción batimétrica intermareal a partir de un cubo temporal SCL.

    Server-side: clasificación agua/tierra, acotación de elevación mediante
    alturas de marea, optimización basada en energía con preservación de
    taludes y difusión anisótropa tipo Perona-Malik.
    """

    array = cube.get_array()
    time_dim = _resolve_time_dimension(array)

    # Bandas por nombre: B03 y B08 son nativas a 10 m -> NDWI a 10 m REAL.
    # El SCL (20 m remuestreado) se usa SOLO como máscara de nubes.
    band_names = [str(b) for b in array.coords["bands"].values]

    def _band(name):
        da = array.isel(bands=band_names.index(name))
        spatial = [dim for dim in da.dims if dim != time_dim]
        return da.transpose(time_dim, *spatial)

    b03 = _band("B03").values.astype(np.float32)
    b08 = _band("B08").values.astype(np.float32)
    template = _band("SCL")
    scl = template.values.astype(np.int16)
    spatial_dims = [dim for dim in template.dims if dim != time_dim]
    dates = _extract_dates(template[time_dim].values)

    tide_heights = context.get("tide_heights") or {}
    valid_dates = context.get("valid_dates")
    clear_classes = context.get("clear_classes") or [4, 5, 6, 12]
    ndwi_threshold = float(context.get("ndwi_threshold", 0.1))
    smooth_lambda = float(context.get("smooth_lambda") or 0.35)
    edge_sigma = float(context.get("edge_sigma") or 0.3)
    diffusion_iterations = int(context.get("diffusion_iterations") or 8)

    # Método de reconstrucción de la elevación:
    #   "optimized" (por defecto): bracketing + optimización + difusión
    #   "pixels":   bracketing puro (punto medio del intervalo por píxel)
    #   "isolines": interpolación desde las líneas de costa (waterlines)
    method = context.get("method") or "optimized"

    tide_array, temporal_mask = _build_temporal_arrays(
        dates=dates,
        tide_heights=tide_heights,
        valid_dates=valid_dates,
    )

    # NDWI = (B03 - B08) / (B03 + B08); agua = clear & NDWI > umbral
    den = b03 + b08
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(den != 0, (b03 - b08) / den, np.nan)

    clear_mask = _isin(scl, clear_classes) & temporal_mask[:, None, None]
    del b03, b08, den, scl

    if method in ("step", "siq"):
        # ── Métodos que usan la SERIE NDWI continua + marea ─────────────────
        #   "step" : mediana móvil NDWI vs marea -> cruce seco->mojado (DEA)
        #   "siq"  : Soft Inundation Quantile (novel) -> inversión de la CDF de
        #            marea usando la frecuencia soft de inundación (NDWI continuo)
        if method == "step":
            elevation, confidence, observation_count = _step_elevation(
                ndwi, tide_array, clear_mask, ndwi_threshold,
            )
        else:
            elevation, confidence, observation_count = _siq_elevation(
                ndwi, tide_array, clear_mask, ndwi_threshold,
            )
        residual = np.zeros_like(elevation, dtype=np.float32)
        del ndwi, clear_mask

    else:
        # ── Métodos de bracketing (agua/tierra binario por NDWI) ────────────
        finite = np.isfinite(ndwi)
        water_mask = clear_mask & finite & (ndwi > ndwi_threshold)
        land_mask = clear_mask & finite & (ndwi <= ndwi_threshold)
        # Liberar los arrays temporales grandes (t,y,x) antes del bracketing.
        del ndwi, finite, clear_mask

        zmin, zmax, observation_count = _bracket_elevations(
            tide_array=tide_array,
            water_mask=water_mask,
            land_mask=land_mask,
        )

        z0, inconsistent = _initial_estimate(zmin, zmax)

        if method == "pixels":
            elevation = z0.astype(np.float32)

        elif method == "isolines":
            elevation = _isoline_elevation(
                water_mask=water_mask,
                land_mask=land_mask,
                tide_array=tide_array,
                temporal_mask=temporal_mask,
            )
            elevation = np.where(np.isfinite(z0), elevation, np.nan).astype(np.float32)

        else:
            elevation = _optimize_elevation(
                z0=z0,
                valid_mask=~np.isnan(z0),
                smooth_lambda=smooth_lambda,
                edge_sigma=edge_sigma,
            )

            elevation = _anisotropic_diffusion(
                elevation,
                iterations=diffusion_iterations,
                kappa=edge_sigma,
            )

        residual = zmax - zmin

        confidence = _compute_confidence(
            observation_count=observation_count,
            residual=residual,
            inconsistent=inconsistent,
        )

    result = _stack_bands(
        elevation=elevation,
        confidence=confidence,
        residual=residual,
        observation_count=observation_count,
        template=template,
        time_dim=time_dim,
        spatial_dims=spatial_dims,
    )

    return XarrayDataCube(result)


def _drop_band_dimension(array):
    """
    Elimina la dimensión 'bands' cuando el cubo contiene una única banda
    (SCL), preservando el resto de dimensiones intactas.
    """

    if "bands" in array.dims and array.sizes["bands"] == 1:
        array = array.isel(bands=0, drop=True)

    return array


def _resolve_time_dimension(array):
    """
    Identifica el nombre de la dimensión temporal ('t' o 'time').
    """

    for candidate in ("t", "time"):
        if candidate in array.dims:
            return candidate

    raise ValueError("No temporal dimension found in input datacube.")


def _extract_dates(time_values):
    """
    Convierte las coordenadas temporales del cubo en cadenas 'YYYY-MM-DD'.
    """

    dates = []

    for value in time_values:
        as_day = np.datetime64(value, "D")
        dates.append(str(as_day))

    return dates


def _isin(array, classes):
    return np.isin(array, classes)


def _build_temporal_arrays(dates, tide_heights, valid_dates):
    """
    Construye el vector de alturas de marea por fecha de adquisición y la
    máscara temporal de observaciones utilizables (fecha válida y con
    altura de marea conocida).
    """

    n_obs = len(dates)

    tide_array = np.full(n_obs, np.nan, dtype=np.float32)
    temporal_mask = np.zeros(n_obs, dtype=bool)

    valid_dates_set = set(valid_dates) if valid_dates is not None else None

    for index, date in enumerate(dates):

        if valid_dates_set is not None and date not in valid_dates_set:
            continue

        height = tide_heights.get(date)

        if height is None:
            continue

        tide_array[index] = height
        temporal_mask[index] = True

    return tide_array, temporal_mask


def _bracket_elevations(tide_array, water_mask, land_mask):
    """
    Calcula, por píxel:

        zmin: máxima altura de marea observada estando el píxel en tierra
        zmax: mínima altura de marea observada estando el píxel en agua

    junto con el número total de observaciones válidas.
    """

    tide_broadcast = tide_array[:, None, None]

    land_heights = np.where(land_mask, tide_broadcast, -np.inf)
    zmin = np.max(land_heights, axis=0)
    zmin = np.where(np.isfinite(zmin), zmin, np.nan).astype(np.float32)

    water_heights = np.where(water_mask, tide_broadcast, np.inf)
    zmax = np.min(water_heights, axis=0)
    zmax = np.where(np.isfinite(zmax), zmax, np.nan).astype(np.float32)

    observation_count = np.sum(water_mask | land_mask, axis=0).astype(np.float32)

    return zmin, zmax, observation_count


def _initial_estimate(zmin, zmax):
    """
    Estimación inicial z0 = (zmin + zmax) / 2. Marca como inconsistentes
    los píxeles donde zmin > zmax, para penalizar su confianza más tarde.
    """

    with np.errstate(invalid="ignore"):
        inconsistent = zmin > zmax

    z0 = (zmin + zmax) / 2.0

    return z0.astype(np.float32), inconsistent


def _shift(array, dy, dx):
    """
    Desplaza el raster (dy, dx) píxeles reutilizando el valor de borde más
    cercano, mediante scipy.ndimage.shift con interpolación de orden 0.
    """

    return ndimage.shift(array, shift=(-dy, -dx), order=0, mode="nearest")


def _optimize_elevation(z0, valid_mask, smooth_lambda, edge_sigma, iterations=60):
    """
    Optimización basada en energía:

        E = sum((z - z0)^2) + lambda * sum(w_ij * (z_i - z_j)^2)

    con pesos de suavidad w_ij = exp(-delta^2 / sigma^2) que preservan
    taludes pronunciados. Se resuelve mediante iteración tipo Jacobi,
    lo que además permite reconstruir (inpaint) los píxeles sin dato.
    """

    has_valid = np.any(valid_mask)
    fill_value = np.nanmean(z0) if has_valid else 0.0
    z0_filled = np.where(valid_mask, z0, fill_value).astype(np.float32)

    neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    sigma_sq = edge_sigma ** 2 + 1e-6

    weights = []
    for dy, dx in neighbor_offsets:
        neighbor = _shift(z0_filled, dy, dx)
        delta_sq = (z0_filled - neighbor) ** 2
        weights.append(np.exp(-delta_sq / sigma_sq).astype(np.float32))

    data_weight = valid_mask.astype(np.float32)
    z = z0_filled.copy()

    for _ in range(iterations):

        neighbor_sum = np.zeros_like(z)
        weight_sum = np.zeros_like(z)

        for (dy, dx), weight in zip(neighbor_offsets, weights):
            neighbor_z = _shift(z, dy, dx)
            neighbor_sum += weight * neighbor_z
            weight_sum += weight

        numerator = data_weight * z0_filled + smooth_lambda * neighbor_sum
        denominator = data_weight + smooth_lambda * weight_sum
        denominator = np.where(denominator > 1e-6, denominator, 1.0)

        z = numerator / denominator

    return z.astype(np.float32)


def _isoline_elevation(
    water_mask,
    land_mask,
    tide_array,
    temporal_mask,
    smooth_lambda=3.0,
    iterations=300,
):
    """
    Reconstrucción por isolíneas (waterlines), server-side.

    Para cada fecha se extrae la línea de costa (píxeles de agua adyacentes a
    tierra y viceversa) y se le asigna la altura de marea de esa fecha. Cada
    orilla es una observación BLANDA de la elevación; el DEM se obtiene
    minimizando por mínimos cuadrados

        sum_i c_i (z_i - obs_i)^2  +  lambda * sum (z_i - z_vecino)^2

    donde c_i es el nº de fechas en que el píxel fue orilla (más peso a los
    más observados). El término de suavidad promedia las orillas en conflicto
    (píxeles vecinos con mareas dispares) y rellena los huecos, dando una
    superficie suave. Se resuelve por iteración de Jacobi (tile-safe).
    """

    n_obs, height, width = water_mask.shape

    accum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)

    for index in range(n_obs):

        if not temporal_mask[index]:
            continue

        tide = tide_array[index]

        water = water_mask[index]
        land = land_mask[index]

        shoreline = (water & ndimage.binary_dilation(land)) | (
            land & ndimage.binary_dilation(water)
        )

        accum[shoreline] += tide
        count[shoreline] += 1.0

    has_shoreline = count > 0

    if not has_shoreline.any():
        return np.full((height, width), np.nan, dtype=np.float32)

    # Observación por píxel (media de mareas donde fue orilla) y su peso.
    observation = accum / np.where(count > 0, count, 1.0)
    data_weight = count.astype(np.float64)

    # Inicialización e iteración de Jacobi del sistema de mínimos cuadrados.
    fill = float(observation[has_shoreline].mean())
    z = np.where(has_shoreline, observation, fill).astype(np.float64)

    smooth_lambda = float(smooth_lambda)

    for _ in range(int(iterations)):

        neighbor_sum = (
            _shift(z, -1, 0)
            + _shift(z, 1, 0)
            + _shift(z, 0, -1)
            + _shift(z, 0, 1)
        )

        numerator = data_weight * observation + smooth_lambda * neighbor_sum
        denominator = data_weight + smooth_lambda * 4.0

        z = numerator / denominator

    return z.astype(np.float32)


def _anisotropic_diffusion(elevation, iterations, kappa, step=0.15):
    """
    Difusión anisótropa tipo Perona-Malik. Suaviza terrazas manteniendo
    los taludes pronunciados gracias al coeficiente de conducción
    c = exp(-(gradiente / kappa)^2).
    """

    z = elevation.astype(np.float32).copy()
    kappa = max(float(kappa), 1e-3)

    for _ in range(max(iterations, 0)):

        north = _shift(z, -1, 0) - z
        south = _shift(z, 1, 0) - z
        east = _shift(z, 0, 1) - z
        west = _shift(z, 0, -1) - z

        c_north = np.exp(-(north / kappa) ** 2)
        c_south = np.exp(-(south / kappa) ** 2)
        c_east = np.exp(-(east / kappa) ** 2)
        c_west = np.exp(-(west / kappa) ** 2)

        z = z + step * (
            c_north * north
            + c_south * south
            + c_east * east
            + c_west * west
        )

    return z.astype(np.float32)


def _compute_confidence(
    observation_count,
    residual,
    inconsistent,
    min_observations=4.0,
    residual_scale=1.5,
):
    """
    Mapa de confianza combinando número de observaciones, anchura del
    intervalo (zmax - zmin) y consistencia de la reconstrucción.
    Normalizado en [0, 1].
    """

    observation_term = np.clip(observation_count / min_observations, 0.0, 1.0)

    residual_abs = np.abs(residual)
    residual_term = np.exp(-residual_abs / residual_scale)
    residual_term = np.where(np.isnan(residual), 0.0, residual_term)

    consistency_term = np.where(inconsistent, 0.5, 1.0)

    confidence = observation_term * residual_term * consistency_term
    confidence = np.where(observation_count > 0, confidence, 0.0)

    return np.clip(confidence, 0.0, 1.0).astype(np.float32)



def _step_elevation(ndwi, tide_array, clear_mask, threshold,
                    n_windows=100, window_frac=0.15, min_obs=5):
    """Método del ESCALÓN (DEA): mediana móvil del NDWI en ventanas de marea;
    la cota es la marea del primer cruce seco->mojado con marea creciente."""
    n_t, h, w = ndwi.shape
    valid_t = np.isfinite(tide_array)
    tides = tide_array[valid_t]
    nan = np.full((h, w), np.nan, np.float32)
    if tides.size < 2:
        return nan, nan.copy(), np.zeros((h, w), np.float32)
    tmin, tmax = float(tides.min()), float(tides.max())
    rng = tmax - tmin
    if rng <= 0:
        return nan, nan.copy(), np.zeros((h, w), np.float32)
    half = 0.5 * window_frac * rng
    centers = np.linspace(tmin, tmax, n_windows)
    nd = np.where(clear_mask, ndwi, np.nan)
    rolling = np.full((n_windows, h, w), np.nan, np.float32)
    for k in range(n_windows):
        tc = centers[k]
        sel = valid_t & (tide_array >= tc - half) & (tide_array <= tc + half)
        if sel.any():
            with np.errstate(invalid="ignore"):
                rolling[k] = np.nanmedian(nd[sel], axis=0)
    obs = np.sum(clear_mask, axis=0).astype(np.float32)
    wet = rolling >= threshold
    dry_low = rolling[0] < threshold
    wet_high = rolling[-1] >= threshold
    inter = dry_low & wet_high & np.isfinite(rolling[0]) & np.isfinite(rolling[-1])
    k1 = np.argmax(wet, axis=0)
    k0 = np.clip(k1 - 1, 0, n_windows - 1)
    r1 = np.take_along_axis(rolling, k1[None], 0)[0]
    r0 = np.take_along_axis(rolling, k0[None], 0)[0]
    c1 = centers[k1]
    c0 = centers[k0]
    d = r1 - r0
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.abs(d) > 1e-6, (threshold - r0) / d, 0.0)
    frac = np.clip(np.nan_to_num(frac), 0, 1)
    z = (c0 + frac * (c1 - c0)).astype(np.float32)
    elevation = np.where(inter & (obs >= min_obs), z, np.nan).astype(np.float32)
    confidence = np.where(
        np.isfinite(elevation), np.clip(obs / (2.0 * min_obs), 0, 1), 0.0
    ).astype(np.float32)
    return elevation, confidence, obs


def _siq_elevation(ndwi, tide_array, clear_mask, threshold,
                   soft_halfwidth=0.1, min_obs=5):
    """Soft Inundation Quantile (SIQ) - método NOVEL. Un píxel a cota z está
    mojado cuando marea > z, luego su frecuencia de inundación F = P(marea > z),
    y por tanto z = cuantil de marea en (1-F). Sin calibración (usa la CDF del
    propio modelo de marea) y con NDWI continuo (frecuencia 'soft')."""
    n_t, h, w = ndwi.shape
    valid_t = np.isfinite(tide_array)
    lo = threshold - soft_halfwidth
    hi = threshold + soft_halfwidth
    with np.errstate(invalid="ignore"):
        w_soft = np.clip((ndwi - lo) / (hi - lo), 0.0, 1.0)
    usable = clear_mask & valid_t[:, None, None]
    w_soft = np.where(usable, w_soft, 0.0)
    clear = usable.astype(np.float32)
    n_wet = np.sum(w_soft, axis=0)
    clear_votes = np.sum(clear, axis=0)
    order = np.argsort(tide_array)[::-1]
    tide_sorted = tide_array[order]
    cum = np.cumsum(clear[order], axis=0)
    reached = cum >= n_wet[None]
    any_reached = reached.any(axis=0)
    k1 = np.argmax(reached, axis=0)
    k0 = np.clip(k1 - 1, 0, n_t - 1)
    cum1 = np.take_along_axis(cum, k1[None], 0)[0]
    cum0 = np.take_along_axis(cum, k0[None], 0)[0]
    t1 = tide_sorted[k1]
    t0 = tide_sorted[k0]
    d = cum1 - cum0
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.abs(d) > 1e-6, (n_wet - cum0) / d, 0.0)
    frac = np.clip(np.nan_to_num(frac), 0, 1)
    z = (t0 + frac * (t1 - t0)).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(clear_votes > 0, n_wet / np.where(clear_votes > 0, clear_votes, 1), np.nan)
    elevation = np.where(
        (clear_votes >= min_obs) & (F > 0.001) & (F < 0.999) & any_reached,
        z, np.nan,
    ).astype(np.float32)
    confidence = np.where(
        np.isfinite(elevation),
        np.clip(4.0 * F * (1.0 - F), 0, 1) * np.clip(clear_votes / (2.0 * min_obs), 0, 1),
        0.0,
    ).astype(np.float32)
    return elevation, confidence, clear_votes.astype(np.float32)

def _stack_bands(
    elevation,
    confidence,
    residual,
    observation_count,
    template,
    time_dim,
    spatial_dims,
):
    """
    Ensambla el DataArray multibanda de salida (Elevation, Confidence,
    Residual, Observations), conservando las coordenadas espaciales
    originales del cubo de entrada.
    """

    stacked = np.stack([elevation, confidence, residual, observation_count], axis=0)

    coords = {
        name: coordinate
        for name, coordinate in template.coords.items()
        if time_dim not in coordinate.dims
    }
    coords["bands"] = ["Elevation", "Confidence", "Residual", "Observations"]

    result = xr.DataArray(
        stacked,
        dims=["bands"] + list(spatial_dims),
        coords=coords,
    )

    return result
'''

# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass(slots=True)
class BathymetryResult:
    """
    Resultado de la reconstrucción batimétrica.
    """

    elevation: np.ndarray
    confidence: np.ndarray

    slope: np.ndarray
    aspect: np.ndarray
    hillshade: np.ndarray

    observation_count: np.ndarray
    residual: np.ndarray

    transform: rasterio.Affine
    crs: object

    # Máscara booleana de la zona batimétrica realmente reconstruida
    # (píxeles con observación agua+tierra que acotan la elevación). Fuera
    # de ella `elevation` es NaN. Consumir esta máscara (o directamente los
    # NaN de `elevation`) evita que cada visualización tenga que recalcularla.
    mask: Optional[np.ndarray] = None

    contours: Optional[gpd.GeoDataFrame] = None

    metadata: dict | None = None

    def clip_to(self, intertidal_mask):
        """Recorta el DEM a la MÁSCARA INTERMAREAL: fuera de ella, la elevación y
        sus derivados pasan a NaN (y la máscara de validez a False). Así la
        batimetría no da valores en píxeles que no son intermareales.

        Alinea rejillas por si hay off-by-one (recorta a la región común).
        Modifica el resultado in situ y lo devuelve (encadenable).
        """
        m = np.asarray(intertidal_mask, dtype=bool)
        shp = self.elevation.shape
        keep = np.zeros(shp, dtype=bool)
        h = min(m.shape[0], shp[0])
        w = min(m.shape[1], shp[1])
        keep[:h, :w] = m[:h, :w]

        for name in ("elevation", "confidence", "slope", "aspect", "hillshade", "residual"):
            arr = getattr(self, name, None)
            if arr is not None and getattr(arr, "shape", None) == shp:
                setattr(self, name, np.where(keep, arr, np.nan).astype(np.float32))
        if self.observation_count is not None and self.observation_count.shape == shp:
            self.observation_count = np.where(keep, self.observation_count, 0).astype(np.float32)
        self.mask = (self.mask & keep) if (self.mask is not None and self.mask.shape == shp) else keep
        return self


class BathymetryReconstructor:
    """
    Reconstrucción batimétrica basada en observaciones multitemporales.

    Toda la computación intensiva se ejecuta en OpenEO.

    El flujo general consiste en:

        Sentinel-2
            ↓
        Water/Land cube
            ↓
        Asociación con mareas
            ↓
        DEM inicial
            ↓
        Optimización
            ↓
        Productos derivados

    Parameters
    ----------
    connection :
        Conexión autenticada OpenEO.

    Examples
    --------

    >>> reconstructor = BathymetryReconstructor(conn)

    >>> result = reconstructor.reconstruct(...)
    """

    def __init__(self, connection):

        self.connection = connection

    def reconstruct(
        self,
        bbox: dict,
        time_extent: Sequence[str],
        tide_heights: dict[str, float],
        valid_dates: Sequence[str] | None = None,
        out_dir: str | Path = "bathymetry",
        force: bool = False,
        min_confidence: float = 0.0,
        ndwi_threshold: float = 0.1,
        water_source: str = "ndwi",
    ) -> BathymetryResult:
        """
        Reconstruct an intertidal DEM from Sentinel-2 SCL observations and
        tide model predictions.

        Parameters
        ----------
        bbox
            Spatial extent.

        time_extent
            Temporal interval.

        tide_heights
            Dictionary mapping acquisition dates ("YYYY-MM-DD") to predicted
            tide height.

        valid_dates
            Dates surviving the cloud filtering stage.

        out_dir
            Output directory.

        force
            Recompute even if previous results already exist.

        min_confidence
            Umbral de confianza por debajo del cual un píxel se considera
            relleno (no reconstruido) y su elevación se pone a NaN.

        Returns
        -------
        BathymetryResult
        """

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # dem_path contiene un GeoTIFF de 4 bandas:
        # 1=Elevation, 2=Confidence, 3=Residual, 4=Observations
        dem_path = out_dir / "bathymetry.tif"
        slope_path = out_dir / "slope.tif"
        hillshade_path = out_dir / "hillshade.tif"
        contours_path = out_dir / "contours.geojson"

        # -------------------------------------------------------------
        # Reuse existing products
        # -------------------------------------------------------------

        if not force and dem_path.exists():
            return self._assemble_result(
                dem_path=dem_path,
                min_confidence=min_confidence,
            )

        # -------------------------------------------------------------
        # Build OpenEO processing graph
        # -------------------------------------------------------------

        cube = self._compute_bathymetry_cube(
            bbox=bbox,
            time_extent=time_extent,
            tide_heights=tide_heights,
            valid_dates=valid_dates,
            ndwi_threshold=ndwi_threshold,
            water_source=water_source,
        )

        # -------------------------------------------------------------
        # Execute Batch Job
        # -------------------------------------------------------------

        self._run_batch_job(
            cube=cube,
            output_path=dem_path,
            title="bathymetry_reconstruction",
        )

        # -------------------------------------------------------------
        # Load DEM (elevation, confidence, residual, observations)
        # -------------------------------------------------------------

        (
            elevation,
            confidence,
            residual,
            observation_count,
            transform,
            crs,
        ) = self._load_multiband_raster(dem_path)

        # -------------------------------------------------------------
        # Enmascarar el relleno: fuera de la zona reconstruida real la
        # elevación pasa a NaN (ver _valid_mask). Así derivadas, contornos
        # y visualizaciones trabajan solo sobre dato real.
        # -------------------------------------------------------------

        mask = self._valid_mask(
            confidence=confidence,
            observation_count=observation_count,
            elevation=elevation,
            min_confidence=min_confidence,
        )

        elevation = np.where(mask, elevation, np.nan).astype(np.float32)

        # -------------------------------------------------------------
        # Terrain derivatives
        # -------------------------------------------------------------

        slope = self._compute_slope(elevation)

        self._save_raster(
            slope,
            slope_path,
            transform,
            crs,
        )

        aspect = self._compute_aspect(elevation)

        hillshade = self._compute_hillshade(
            elevation=elevation,
            slope=slope,
            aspect=aspect,
        )

        self._save_raster(
            hillshade,
            hillshade_path,
            transform,
            crs,
        )

        # -------------------------------------------------------------
        # Contours
        # -------------------------------------------------------------

        contours = self._compute_contours(
            elevation=elevation,
            transform=transform,
            crs=crs,
        )

        contours.to_file(
            contours_path,
            driver="GeoJSON",
        )

        # -------------------------------------------------------------
        # Final result
        # -------------------------------------------------------------

        return BathymetryResult(
            elevation=elevation,
            confidence=confidence,
            slope=slope,
            aspect=aspect,
            hillshade=hillshade,
            observation_count=observation_count,
            residual=residual,
            mask=mask,
            contours=contours,
            transform=transform,
            crs=crs,
        )

    # -----------------------------------------------------------------
    # Métodos alternativos (locales) con la MISMA firma que reconstruct():
    # descargan el SCL vía OpenEO y aplican el algoritmo en Python, para
    # poder comparar enfoques con las mismas visualizaciones.
    # -----------------------------------------------------------------

    def reconstruct_pixels(
        self,
        bbox: dict,
        time_extent: Sequence[str],
        tide_heights: dict[str, float],
        valid_dates: Sequence[str] | None = None,
        out_dir: str | Path = "bathymetry",
        force: bool = False,
        min_confidence: float = 0.0,
        ndwi_threshold: float = 0.1,
        water_source: str = "ndwi",
    ) -> BathymetryResult:
        """
        Método 1 — Bracketing por píxel (diccionario de píxeles), server-side.

        Todo el cómputo ocurre en OpenEO (UDF con ``method="pixels"``): por
        píxel se acota la elevación con ``[max(marea en tierra),
        min(marea en agua)]`` y se toma el punto medio, sin optimización.
        Misma firma que :meth:`reconstruct`.
        """

        return self._reconstruct_udf(
            bbox=bbox,
            time_extent=time_extent,
            tide_heights=tide_heights,
            valid_dates=valid_dates,
            out_dir=out_dir,
            force=force,
            min_confidence=min_confidence,
            method="pixels",
            filename="bathymetry_pixels.tif",
            title="bathymetry_pixels",
            ndwi_threshold=ndwi_threshold,
            water_source=water_source,
        )

    def reconstruct_isolines(
        self,
        bbox: dict,
        time_extent: Sequence[str],
        tide_heights: dict[str, float],
        valid_dates: Sequence[str] | None = None,
        out_dir: str | Path = "bathymetry",
        force: bool = False,
        min_confidence: float = 0.0,
        ndwi_threshold: float = 0.1,
        water_source: str = "ndwi",
    ) -> BathymetryResult:
        """
        Método 2 — Isolíneas (waterlines), server-side.

        Todo el cómputo ocurre en OpenEO (UDF con ``method="isolines"``): la
        línea de costa de cada fecha (z = marea) alimenta una interpolación por
        vecino más cercano del DEM. Misma firma que :meth:`reconstruct`.
        """

        return self._reconstruct_udf(
            bbox=bbox,
            time_extent=time_extent,
            tide_heights=tide_heights,
            valid_dates=valid_dates,
            out_dir=out_dir,
            force=force,
            min_confidence=min_confidence,
            method="isolines",
            filename="bathymetry_isolines.tif",
            title="bathymetry_isolines",
            ndwi_threshold=ndwi_threshold,
            water_source=water_source,
        )

    def reconstruct_step(
        self,
        bbox: dict,
        time_extent: Sequence[str],
        tide_heights: dict[str, float],
        valid_dates: Sequence[str] | None = None,
        out_dir: str | Path = "bathymetry",
        force: bool = False,
        min_confidence: float = 0.0,
        ndwi_threshold: float = 0.1,
        water_source: str = "ndwi",
    ) -> BathymetryResult:
        """Método del ESCALÓN (DEA): mediana móvil del NDWI vs marea; la cota es
        la marea del cruce seco->mojado. Requiere water_source="ndwi"."""
        return self._reconstruct_udf(
            bbox=bbox, time_extent=time_extent, tide_heights=tide_heights,
            valid_dates=valid_dates, out_dir=out_dir, force=force,
            min_confidence=min_confidence, method="step",
            filename="bathymetry_step.tif", title="bathymetry_step",
            ndwi_threshold=ndwi_threshold, water_source=water_source,
        )

    def reconstruct_siq(
        self,
        bbox: dict,
        time_extent: Sequence[str],
        tide_heights: dict[str, float],
        valid_dates: Sequence[str] | None = None,
        out_dir: str | Path = "bathymetry",
        force: bool = False,
        min_confidence: float = 0.0,
        ndwi_threshold: float = 0.1,
        water_source: str = "ndwi",
    ) -> BathymetryResult:
        """Soft Inundation Quantile (SIQ) - método NOVEL: cota = cuantil de marea
        en (1 - frecuencia de inundación soft). Sin calibración, NDWI continuo.
        Requiere water_source="ndwi"."""
        return self._reconstruct_udf(
            bbox=bbox, time_extent=time_extent, tide_heights=tide_heights,
            valid_dates=valid_dates, out_dir=out_dir, force=force,
            min_confidence=min_confidence, method="siq",
            filename="bathymetry_siq.tif", title="bathymetry_siq",
            ndwi_threshold=ndwi_threshold, water_source=water_source,
        )

    def _reconstruct_udf(
        self,
        bbox: dict,
        time_extent: Sequence[str],
        tide_heights: dict[str, float],
        valid_dates: Sequence[str] | None,
        out_dir: str | Path,
        force: bool,
        min_confidence: float,
        method: str,
        filename: str,
        title: str,
        ndwi_threshold: float = 0.1,
        water_source: str = "ndwi",
    ) -> BathymetryResult:
        """
        Ejecuta la reconstrucción server-side con el ``method`` indicado y
        ensambla el resultado. Comparte grafo (``_compute_bathymetry_cube``),
        Batch Job y carga con :meth:`reconstruct`.
        """

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dem_path = out_dir / filename

        if not force and dem_path.exists():
            return self._assemble_result(
                dem_path=dem_path,
                min_confidence=min_confidence,
            )

        cube = self._compute_bathymetry_cube(
            bbox=bbox,
            time_extent=time_extent,
            tide_heights=tide_heights,
            valid_dates=valid_dates,
            method=method,
            ndwi_threshold=ndwi_threshold,
            water_source=water_source,
        )

        self._run_batch_job(
            cube=cube,
            output_path=dem_path,
            title=title,
        )

        return self._assemble_result(
            dem_path=dem_path,
            min_confidence=min_confidence,
        )

    @staticmethod
    def _filter_to_valid_dates(cube, valid_dates):
        """Filtra el cubo a las fechas válidas ANTES del UDF: así el backend no
        carga en memoria las escenas que el UDF va a descartar igualmente (evita
        OOMKilled en rangos largos, p.ej. 10 años -> baja de ~1379 a ~312 fechas).
        Si el backend no soporta filter_labels, se deja el cubo completo (el UDF
        filtra internamente de todos modos)."""
        if not valid_dates:
            return cube
        try:
            return cube.filter_labels(
                dimension="t",
                condition=lambda x: x.isin(list(valid_dates)),
            )
        except Exception:
            return cube

    def _compute_bathymetry_cube(
        self,
        bbox: dict,
        time_extent: Sequence[str],
        tide_heights: dict[str, float],
        valid_dates: Sequence[str] | None = None,
        method: str = "optimized",
        ndwi_threshold: float = 0.1,
        water_source: str = "ndwi",
    ):
        """
        Build the OpenEO processing graph for bathymetry reconstruction.

        Se usa `apply_dimension` (no `reduce_dimension`): el UDF necesita
        ver la dimensión temporal completa y sustituirla por una nueva
        dimensión `bands` de 4 elementos, no colapsarla a un escalar.

        ``method`` selecciona el estimador de elevación en el UDF:
        "optimized" (por defecto), "pixels" o "isolines".

        ``water_source`` elige cómo se detecta agua/tierra:
          - "ndwi" (por defecto): NDWI = (B03-B08)/(B03+B08) a 10 m REAL
            (B03/B08 nativos a 10 m); el SCL solo se usa como máscara de nubes.
          - "scl"  (LEGACY): clasificación agua/tierra por clases SCL (20 m).
        """

        if water_source == "scl":
            # ── LEGACY: agua/tierra por clases SCL (20 m) ────────────────────
            cube = self.connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox,
                temporal_extent=time_extent,
                bands=["SCL"],
                max_cloud_cover=100,
            )
            cube = self._filter_to_valid_dates(cube, valid_dates)
            udf = openeo.UDF(
                code=_BATHYMETRY_UDF_SCL,
                runtime="Python",
                context={
                    "tide_heights": tide_heights,
                    "valid_dates": list(valid_dates)
                    if valid_dates is not None
                    else None,
                    "method": method,
                },
            )
            cube = cube.apply_dimension(
                dimension="t", target_dimension="bands", process=udf,
            )
            return cube

        # ── NDWI (por defecto): agua por índice a 10 m real ──────────────────
        cube = self.connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=bbox,
            temporal_extent=time_extent,
            bands=["B03", "B08", "SCL"],
            max_cloud_cover=100,
        )
        # Rejilla a 10 m: B03/B08 ya lo son (NDWI a 10 m real); SCL sube a 10 m
        # (solo máscara de nubes).
        try:
            cube = cube.resample_spatial(resolution=10, method="near")
        except Exception:
            pass

        cube = self._filter_to_valid_dates(cube, valid_dates)

        udf = openeo.UDF(
            code=_BATHYMETRY_UDF_NDWI,
            runtime="Python",
            context={
                "tide_heights": tide_heights,
                "valid_dates": list(valid_dates)
                if valid_dates is not None
                else None,
                "method": method,
                "ndwi_threshold": float(ndwi_threshold),
            },
        )

        cube = cube.apply_dimension(
            dimension="t",
            target_dimension="bands",
            process=udf,
        )

        return cube

    def _run_batch_job(
        self,
        cube,
        output_path: Path,
        title: str,
    ):
        """
        Execute an OpenEO Batch Job.
        """

        # El UDF carga el stack temporal completo en memoria por executor.
        # Se sube la memoria (sobre todo la del proceso Python del UDF) para
        # evitar OOMKilled en rangos largos (p.ej. 10 años).
        job = (
            cube
            .save_result(format="GTiff")
            .create_job(
                title=title,
                job_options={
                    "executor-memory": "4G",
                    "executor-memoryOverhead": "2G",
                    "python-memory": "6G",
                },
            )
        )

        print(f"Launching '{title}'...")

        job.start_and_wait()

        assets = job.get_results().get_assets()

        if not assets:
            raise RuntimeError(
                f"Batch Job '{title}' returned no assets."
            )

        assets[0].download(str(output_path))

        print(f"Saved to {output_path}")

        return output_path

    def _load_raster(
        self,
        path: str | Path,
    ):
        """
        Read a single-band GeoTIFF.
        """

        path = Path(path)

        with rasterio.open(path) as src:

            raster = src.read(1).astype(np.float32)

            transform = src.transform

            crs = src.crs

        return raster, transform, crs

    def _load_multiband_raster(
        self,
        path: str | Path,
    ):
        """
        Lee el GeoTIFF de 4 bandas producido por el UDF:
        1=Elevation, 2=Confidence, 3=Residual, 4=Observations.
        """

        path = Path(path)

        with rasterio.open(path) as src:

            if src.count < 4:
                raise ValueError(
                    f"Expected a 4-band raster (Elevation, Confidence, "
                    f"Residual, Observations), got {src.count} band(s) "
                    f"in '{path}'."
                )

            elevation = src.read(1).astype(np.float32)
            confidence = src.read(2).astype(np.float32)
            residual = src.read(3).astype(np.float32)
            observation_count = src.read(4).astype(np.float32)

            transform = src.transform
            crs = src.crs

        return elevation, confidence, residual, observation_count, transform, crs

    @staticmethod
    def _valid_mask(
        confidence: np.ndarray,
        observation_count: np.ndarray,
        elevation: np.ndarray | None = None,
        min_confidence: float = 0.0,
    ) -> np.ndarray:
        """
        Máscara de la zona batimétrica realmente reconstruida.

        Un píxel es válido si tuvo observaciones que acotan su elevación por
        ambos lados (agua y tierra). Esto se refleja en confidence > 0: fuera
        del bracket el término de residuo es 0 y la confianza colapsa a 0,
        mientras que la optimización rellena la elevación en toda la escena.
        Sin esta máscara, el DEM denso mezcla ría, tierra y relleno.
        """

        valid = np.isfinite(confidence) & (confidence > min_confidence)

        if observation_count is not None:
            valid &= observation_count > 0

        if elevation is not None:
            valid &= np.isfinite(elevation)

        return valid

    def _save_raster(
        self,
        raster: np.ndarray,
        output_path: str | Path,
        transform,
        crs,
    ):
        """
        Save a float32 raster as GeoTIFF.
        """

        output_path = Path(output_path)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:

            dst.write(
                raster.astype(np.float32),
                1,
            )

    def _assemble_result(
        self,
        dem_path: Path,
        min_confidence: float = 0.0,
    ) -> BathymetryResult:
        """
        Assemble the BathymetryResult object from disk.
        """

        (
            elevation,
            confidence,
            residual,
            observation_count,
            transform,
            crs,
        ) = self._load_multiband_raster(dem_path)

        mask = self._valid_mask(
            confidence=confidence,
            observation_count=observation_count,
            elevation=elevation,
            min_confidence=min_confidence,
        )

        elevation = np.where(mask, elevation, np.nan).astype(np.float32)

        slope = self._compute_slope(elevation)

        aspect = self._compute_aspect(elevation)

        hillshade = self._compute_hillshade(
            elevation,
            slope,
            aspect,
        )

        contours = self._compute_contours(
            elevation,
            transform,
            crs,
        )

        return BathymetryResult(
            elevation=elevation,
            confidence=confidence,
            slope=slope,
            aspect=aspect,
            hillshade=hillshade,
            observation_count=observation_count,
            residual=residual,
            mask=mask,
            contours=contours,
            transform=transform,
            crs=crs,
        )

    def _compute_slope(
        self,
        elevation: np.ndarray,
        pixel_size: float = 10.0,
    ) -> np.ndarray:
        """
        Compute terrain slope (degrees).

        Parameters
        ----------
        elevation
            Elevation raster.

        pixel_size
            Pixel size in metres.

        Returns
        -------
        ndarray
            Slope in degrees.
        """

        dzdx = ndimage.sobel(elevation, axis=1) / (8 * pixel_size)
        dzdy = ndimage.sobel(elevation, axis=0) / (8 * pixel_size)

        slope = np.degrees(
            np.arctan(
                np.sqrt(dzdx**2 + dzdy**2)
            )
        )

        slope[np.isnan(elevation)] = np.nan

        return slope.astype(np.float32)

    def _compute_aspect(
        self,
        elevation: np.ndarray,
        pixel_size: float = 10.0,
    ) -> np.ndarray:
        """
        Compute terrain aspect.

        Returns
        -------
        ndarray
            Aspect (degrees from North).
        """

        dzdx = ndimage.sobel(elevation, axis=1) / (8 * pixel_size)
        dzdy = ndimage.sobel(elevation, axis=0) / (8 * pixel_size)

        aspect = np.degrees(
            np.arctan2(-dzdx, dzdy)
        )

        aspect = (aspect + 360) % 360

        aspect[np.isnan(elevation)] = np.nan

        return aspect.astype(np.float32)

    def _compute_hillshade(
        self,
        elevation: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        azimuth: float = 315,
        altitude: float = 45,
    ) -> np.ndarray:
        """
        Classical GIS hillshade.
        """

        azimuth = np.radians(azimuth)
        altitude = np.radians(altitude)

        slope = np.radians(slope)
        aspect = np.radians(aspect)

        hillshade = (
            np.sin(altitude) * np.cos(slope)
            + np.cos(altitude)
            * np.sin(slope)
            * np.cos(azimuth - aspect)
        )

        hillshade = np.clip(hillshade, 0, 1)

        hillshade *= 255

        hillshade[np.isnan(elevation)] = np.nan

        return hillshade.astype(np.float32)

    def _compute_contours(
        self,
        elevation: np.ndarray,
        transform,
        crs,
        levels: Sequence[float] | None = None,
        interval: float = 0.5,
    ) -> gpd.GeoDataFrame:
        """
        Extrae isolíneas de elevación como GeoDataFrame de LineStrings.

        Parameters
        ----------
        elevation
            Elevation raster.

        transform
            Affine transform del raster.

        crs
            CRS del raster.

        levels
            Cotas explícitas para las isolíneas. Si es None se generan
            automáticamente a partir de `interval`.

        interval
            Separación entre cotas cuando `levels` no se especifica.
        """

        valid = np.isfinite(elevation)
        valid_vals = elevation[valid]

        if valid_vals.size == 0:
            return gpd.GeoDataFrame(
                {"elevation": [], "geometry": []},
                crs=crs,
            )

        if levels is None:
            vmin = float(np.floor(valid_vals.min() / interval) * interval)
            vmax = float(np.ceil(valid_vals.max() / interval) * interval)
            levels = np.arange(vmin, vmax + interval, interval)

        # find_contours no admite NaN. Rellenar con un centinela crearía un
        # salto de valor en el borde dato/relleno y, por tanto, una isolínea
        # espuria que sigue el contorno de la ría. En su lugar se rellenan los
        # píxeles inválidos por vecino más cercano (sin salto de valor) y
        # luego se recortan los vértices que caen fuera de la máscara real.
        if not valid.all():
            _, (iy, ix) = ndimage.distance_transform_edt(
                ~valid,
                return_indices=True,
            )
            filled = np.where(valid, elevation, elevation[iy, ix])
        else:
            filled = elevation

        filled = filled.astype(np.float64)

        valid_f = valid.astype(np.float32)

        records = []

        for level in levels:

            for contour in find_contours(filled, level=float(level)):

                rows = contour[:, 0]
                cols = contour[:, 1]

                # Un vértice se conserva si cae dentro de la zona válida real
                # (>= mitad de su vecindad válida); descarta los tramos que se
                # adentran en la región rellena por vecino más cercano.
                sampled = ndimage.map_coordinates(
                    valid_f,
                    [rows, cols],
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                keep = sampled >= 0.5

                for segment in self._split_by_mask(contour, keep):

                    xs, ys = transform * (segment[:, 1], segment[:, 0])

                    line = LineString(np.column_stack([xs, ys]))

                    if line.length > 0:
                        records.append(
                            {"elevation": float(level), "geometry": line}
                        )

        if not records:
            return gpd.GeoDataFrame(
                {"elevation": [], "geometry": []},
                crs=crs,
            )

        return gpd.GeoDataFrame(records, crs=crs)

    @staticmethod
    def _split_by_mask(
        contour: np.ndarray,
        keep: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Divide una polilínea de contorno en tramos contiguos de vértices
        conservados (keep == True), descartando los que caen en el borde o
        fuera de la zona válida. Devuelve solo tramos con >= 2 vértices.
        """

        segments = []
        current = []

        for point, ok in zip(contour, keep):

            if ok:
                current.append(point)
            elif current:
                if len(current) >= 2:
                    segments.append(np.asarray(current))
                current = []

        if len(current) >= 2:
            segments.append(np.asarray(current))

        return segments

