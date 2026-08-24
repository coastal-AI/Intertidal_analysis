"""
udf.py — Server-side UDF sources (verbatim, for the comparison study)
=====================================================================

These strings are Python programs that run INSIDE the OpenEO backend, one
per water-detection route. They are kept here **unmodified** so the elevation
methods benchmarked in the paper can be re-executed exactly as they were.

Do not refactor them. They are transported as text to a remote interpreter,
so a "harmless" edit here changes what the backend runs, and any drift would
silently invalidate the comparison against the current
:mod:`pyintertidal.elevation` implementations.

``BATHYMETRY_UDF_SCL``
    Water/land from SCL classes (20 m). The original route.
``BATHYMETRY_UDF_NDWI``
    Water/land from NDWI at true 10 m; also implements the ``step`` and
    ``siq`` elevation estimators.

Both expose the same ``method`` context switch: ``"optimized"`` (bracketing +
energy optimisation + anisotropic diffusion), ``"pixels"`` (pure bracketing),
``"isolines"`` (waterline interpolation), plus ``"step"``/``"siq"`` in the
NDWI variant.
"""

BATHYMETRY_UDF_SCL = '''
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

BATHYMETRY_UDF_NDWI = '''
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

