"""Compatibilidad para notebooks legacy sobre la arquitectura modular.

Este módulo concentra los adaptadores necesarios para ejecutar notebooks
históricos sin mantener funciones auxiliares dentro de las celdas.
"""

from __future__ import annotations

from datetime import date
import os
import base64
import zlib
import tempfile
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import rasterio
import openeo

from .geometry import GeometryProcessor
from .raster import RasterProcessor
from .openeo_client import OpenEOClient
from .scl_processor import SCLProcessor
from .visualization import Visualizer


# Alias de nombres legacy
CoordinateUtils = GeometryProcessor


class OpenEOManager(OpenEOClient):
    """Wrapper de compatibilidad para notebooks antiguos."""

    def __init__(self, backend_url: str = "openeo.dataspace.copernicus.eu"):
        super().__init__(backend_url=backend_url)


def download_date_rgb(conn, date, bbox, out_dir, polygon=None):
    client = OpenEOClient()
    client.connection = conn
    return client.download_rgb(date, bbox, out_dir, polygon=polygon)


def download_date_scl(conn, date, bbox, out_dir, polygon=None):
    client = OpenEOClient()
    client.connection = conn
    return client.download_scl(date, bbox, out_dir, polygon=polygon)


def tif_to_rgb(path):
    return RasterProcessor.read_rgb(path)


def tif_to_scl(path):
    return RasterProcessor.read_scl(path)


def compute_scl_stats(scl_array, bad_classes):
    proc = SCLProcessor(bad_classes=bad_classes)
    return proc.compute_stats(scl_array)


def load_scl_stack(dates, scl_dir):
    proc = SCLProcessor()
    return proc.load_stack(dates, scl_dir)


def build_reference_map(scl_stack, stable_threshold=0.98, coastal_buffer_pixels=20):
    proc = SCLProcessor()
    return proc.build_reference_map_local(
        scl_stack,
        stable_threshold=stable_threshold,
        coastal_buffer_pixels=coastal_buffer_pixels,
    )


def compute_transition_cloud_stats(scl_array, transition_mask, bad_classes=None):
    proc = SCLProcessor(bad_classes=bad_classes or [3, 8, 9, 10, 11])
    return proc.compute_transition_stats(scl_array, transition_mask)


def plot_scl_map(
    date,
    scl_dir,
    scl_colors=None,
    scl_bad_classes=None,
    scl_stats=None,
    scl_max_bad_fraction=0.20,
    polygon=None,
):
    return Visualizer.plot_scl_map(
        date=date,
        scl_dir=scl_dir,
        bad_classes=scl_bad_classes,
        scl_stats=scl_stats,
        scl_max_bad_fraction=scl_max_bad_fraction,
        polygon=polygon,
    )


def plot_reference_map(reference_map):
    return Visualizer.plot_reference_map(reference_map)


def plot_rgb_grid(dates, rgb_dir, polygon=None, title=None, cols=4):
    return Visualizer.plot_rgb_grid(dates=dates, rgb_dir=rgb_dir, ncols=cols, polygon=polygon)


def plot_water_frequency(water_freq, transform, crs, title=None, polygon=None,
                         min_water_patch_pixels=20, water_presence_threshold=0.15):
    return Visualizer.plot_water_frequency(
        water_freq=water_freq, transform=transform, crs=crs,
        title=title, polygon=polygon,
        min_water_patch_pixels=min_water_patch_pixels,
        water_presence_threshold=water_presence_threshold,
    )

def plot_intertidal_map(intertidal_mask,
        wf_transform,
        wf_crs,
        intertidal_out_path,
        low_threshold,
        high_threshold,
        area_km2):
    return Visualizer.plot_intertidal_map(intertidal_mask,
            wf_transform,
            wf_crs,
            intertidal_out_path,
            low_threshold,
            high_threshold,
            area_km2
    )
    
def plot_tide_time_series(
        dates_reference,
        heights,
        title: str,):
    return Visualizer.plot_tide_time_series(
        dates_reference=dates_reference,
        heights=heights,
        title=title,
    )
    
def plot_tide_distribution(
        reference_heights,
        selected_heights,
        model_name,
        title: str,):
    return Visualizer.plot_tide_distribution(
        reference_heights=reference_heights,
        selected_heights=selected_heights,
        model_name=model_name,
        title=title,
    )

def plot_bathymetry(
    elevation,
    confidence=None,
    hillshade=None,
    contour_interval=0.25,
    title="Bathymetry",
    figsize=(11, 10),
    alpha_min=0.35,
    alpha_max=1.0,
    min_confidence=0.0,
    **kwargs,
):
    return Visualizer.plot_bathymetry(
        elevation=elevation,
        confidence=confidence,
        hillshade=hillshade,
        contour_interval=contour_interval,
        title=title,
        figsize=figsize,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        min_confidence=min_confidence,
        **kwargs,
    )
    
def plot_bathymetry_uncertainty(
    confidence,
    mask,
    elevation=None,
    figsize=(10, 10),
    **kwargs,
):
    return Visualizer.plot_bathymetry_uncertainty(
        confidence=confidence,
        mask=mask,
        elevation=elevation,
        figsize=figsize,
        **kwargs,
    )

def plot_bathymetry_profile(
    elevation,
    mask,
    start,
    end,
    pixel_size=10.0,
    smooth_sigma=2.0,
    figsize=(12, 5),
    **kwargs,
):
    return Visualizer.plot_bathymetry_profile(
        elevation=elevation,
        mask=mask,
        start=start,
        end=end,
        pixel_size=pixel_size,
        smooth_sigma=smooth_sigma,
        figsize=figsize,
        **kwargs,
    )

def plot_bathymetry_3d(
    elevation,
    mask=None,
    confidence=None,
    min_confidence=0.0,
    pixel_size=10.0,
    vertical_exaggeration=5.0,
    downsample=2,
    fill_region=True,
    clip_to_region=False,
    close_iter=2,
    region_dilate=2,
    despike_size=5,
    smooth_sigma=1.5,
    **kwargs,
):
    return Visualizer.plot_bathymetry_3d(
        elevation=elevation,
        mask=mask,
        confidence=confidence,
        min_confidence=min_confidence,
        pixel_size=pixel_size,
        vertical_exaggeration=vertical_exaggeration,
        downsample=downsample,
        fill_region=fill_region,
        clip_to_region=clip_to_region,
        close_iter=close_iter,
        region_dilate=region_dilate,
        despike_size=despike_size,
        smooth_sigma=smooth_sigma,
        **kwargs,
    )
    
def download_reference_map_openeo(
    conn,
    bbox,
    time_extent,
    out_path,
    bad_classes,
    bad_fraction_threshold,
    stable_threshold,
    transition_buffer_pixels,
    force=False,
):
    client = OpenEOClient()
    client.connection = conn
    return client.build_reference_map(
        bbox=bbox,
        time_extent=time_extent,
        output_path=out_path,
        bad_classes=bad_classes,
        bad_fraction_threshold=bad_fraction_threshold,
        stable_threshold=stable_threshold,
        transition_buffer_pixels=transition_buffer_pixels,
        force=force,
    )


def load_reference_map_tif(path):
    return RasterProcessor.load_reference_map(path)


# ── UDF: porcentaje de nubes en zona de transición, por fecha ─────────────────
# Recibe el stack SCL completo (t, y, x) y la transition_mask codificada en
# base64+zlib. Devuelve un DataArray (t,) con el % de nubes en la transición
# para cada fecha.
_TRANSITION_STATS_UDF = r"""
import base64
import zlib
import numpy as np
import xarray


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:
    # ── Decodificar transition_mask desde base64+zlib ─────────────────────────
    raw = base64.b64decode(context["transition_mask_b64"])
    flat = np.frombuffer(zlib.decompress(raw), dtype=np.uint8).astype(bool)
    h, w = context["shape"]
    transition_mask = flat.reshape(h, w)

    bad_classes = context.get("bad_classes", [3, 8, 9, 10])

    arr = cube.values  # (bands, t, y, x) o (t, y, x)

    # Normalizar a (t, y, x)
    if arr.ndim == 4:
        arr = arr[0]  # quitar dim bands

    n_t = arr.shape[0]
    n_transition = max(int(transition_mask.sum()), 1)

    pcts = np.zeros(n_t, dtype=np.float32)

    for i in range(n_t):
        bad = np.isin(arr[i], bad_classes)
        pcts[i] = float(bad[transition_mask].sum()) / n_transition * 100

    # Devolver DataArray con solo la dimensión temporal
    t_coord = cube.coords["t"] if "t" in cube.coords else cube.coords[list(cube.dims)[0]]
    return xarray.DataArray(
        pcts,
        dims=["t"],
        coords={"t": t_coord},
    )
"""


def _encode_mask(mask: np.ndarray) -> tuple[str, tuple[int, int]]:
    """Comprime la máscara booleana con zlib y la codifica en base64."""
    flat = mask.astype(np.uint8).flatten()
    compressed = zlib.compress(flat.tobytes(), level=9)
    b64 = base64.b64encode(compressed).decode("ascii")
    return b64, mask.shape


def evaluate_transition_cloud_coverage_openeo(
    conn,
    bbox,
    time_extent,
    reference_map,
    bad_classes,
    reference_dates=None,
    global_bad_fraction_threshold=0.05,
    scl_dir="tifs_scl",
):
    """
    Evalúa la cobertura nubosa en la zona de transición para cada fecha
    del rango temporal, descargando el SCL de cada fecha y calculando en local.

    ESTRATEGIA:
    -----------------------------------------------------------------------
    1. Obtener todas las fechas disponibles vía catálogo STAC (sin coste).
    2. Excluir reference_dates si se proporcionan.
    3. Descargar el SCL de cada fecha (idempotente: reutiliza los ya en disco).
    4. Para cada fecha, calcular en local la fracción global de píxeles malos
       y, si supera el umbral, la fracción dentro de la zona de transición.
    5. Devolver el porcentaje relevante por fecha (global si es de referencia,
       de transición si es una fecha evaluada).

    Parámetros:
        conn                            : conexión autenticada a OpenEO
        bbox                            : dict con west/south/east/north
        time_extent                     : [fecha_inicio, fecha_fin]
        reference_map                   : np.ndarray 2D (y, x) con valores 0/1/2
        reference_transform             : (no usado en esta implementación)
        bad_classes                     : clases SCL malas
        reference_dates                 : fechas a excluir (default: None)
        global_bad_fraction_threshold   : umbral de fracción global; por debajo
                                          se devuelve la fracción global (fecha
                                          de referencia), por encima la fracción
                                          de nubes en la zona de transición
        scl_dir                         : directorio donde cachear los SCL

    Devuelve:
        dict {fecha_str: porcentaje_nubes_en_transición}
    """
    if bad_classes is None:
        bad_classes = [3, 8, 9, 10]
    if reference_dates is None:
        reference_dates = []

    transition_mask = reference_map == 0

    client = OpenEOClient()
    client.connection = conn

    import re as _re

    def _to_iso(label) -> str:
        s = str(label)
        m = _re.search(r"\d{4}-\d{2}-\d{2}", s)
        if m:
            return m.group(0)
        # Fallback: parsear formatos no-ISO (p.ej. RFC-2822
        # 'Fri, 03 May 2024 00:00:00 GMT') con pandas.
        try:
            return pd.to_datetime(s).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return s

    # ── 1. Consultar fechas disponibles ──────────────────────────────────────
    # Se usa dimension_labels("t") de openeo (consulta de metadatos, sin coste),
    # que es más fiable que el catálogo STAC (a veces devuelve 400/0 resultados).
    print(f"  Consultando fechas ({time_extent[0]} -> {time_extent[1]})...")
    try:
        cube_explore = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=bbox,
            temporal_extent=time_extent,
            bands=["B04"],
            max_cloud_cover=100,
        )
        raw_labels = cube_explore.dimension_labels("t").execute()
        all_dates = sorted({_to_iso(d) for d in raw_labels})
    except Exception as exc:
        print(f"  dimension_labels falló ({exc}); usando catálogo STAC como fallback")
        # Fallback robusto: usar get_overpass_times (endpoint STAC v1, que sí
        # funciona) en lugar de get_available_dates (endpoint antiguo que
        # devuelve 400/0 resultados).
        try:
            from .overpass import get_overpass_times
            overpass = get_overpass_times(bbox, time_extent)
            all_dates = sorted(overpass.keys())
        except Exception as stac_exc:
            print(f"  STAC también falló ({stac_exc}); usando fechas de archivos locales")
            # Último fallback: leer fechas de los archivos SCL descargados
            # Nota: os y re ya están importados a nivel módulo, no reimportar
            all_dates = []
            if os.path.isdir(scl_dir):
                for filename in os.listdir(scl_dir):
                    match = _re.match(r"scl_(\d{4}-\d{2}-\d{2})\.tif$", filename)
                    if match:
                        all_dates.append(match.group(1))
            all_dates = sorted(set(all_dates))
            if all_dates:
                print(f"  ✓ Encontradas {len(all_dates)} fechas en archivos locales")
            else:
                print(f"  ⚠️ No se pudieron obtener fechas por ningún método")
                return {}

    print(f"  Ejemplo fechas parseadas: {all_dates[:3] if all_dates else '(ninguna)'}")

    ref_set = set(reference_dates)
    eval_dates = [d for d in all_dates if d not in ref_set]
    print(
        f"  Fechas totales: {len(all_dates)} | "
        f"Referencia: {len(ref_set)} | A evaluar: {len(eval_dates)}"
    )

    if not eval_dates:
        print("  No hay fechas a evaluar fuera del periodo de referencia.")
        return {}

    # ── 2. Descargar el cubo SCL completo en UN SOLO batch job (NetCDF) ───────
    # A diferencia del reference map (que reduce la dimensión t en el backend),
    # aquí necesitamos conservar t para tener un valor por fecha. Por eso NO se
    # usa un UDF con reduce_dimension("t") (que exige colapsar t): se descarga
    # el cubo (t, y, x) tal cual y se calculan las estadísticas en local.
    import xarray as _xr

    cube_scl = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[eval_dates[0], eval_dates[-1]],
        bands=["SCL"],
        max_cloud_cover=100,
    )

    tmp_path = tempfile.mktemp(suffix=".nc")
    job = (
        cube_scl
        .save_result(format="netCDF")
        .create_job(title="transition_cloud_stats")
    )
    print("  Lanzando batch job (descarga del cubo SCL completo)...")
    job.start_and_wait()

    assets = job.get_results().get_assets()
    if not assets:
        raise RuntimeError("El job no devolvió assets")
    assets[0].download(tmp_path)
    print(f"  Cubo SCL descargado -> {tmp_path}")

    # ── 3. Leer el cubo y localizar la banda SCL y el eje temporal ───────────
    ds = _xr.open_dataset(tmp_path)
    scl_var = "SCL" if "SCL" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[scl_var]

    # Nombre de la dimensión temporal (t / time)
    t_dim = "t" if "t" in da.dims else ("time" if "time" in da.dims else da.dims[0])

    # Fechas asociadas a cada plano temporal del cubo
    cube_dates = [
        _to_iso(v) for v in np.asarray(da[t_dim].values)
    ]

    def _align_to_mask(arr):
        """Recorta el SCL y la máscara a la región común (evita off-by-one)."""
        h = min(arr.shape[0], transition_mask.shape[0])
        w = min(arr.shape[1], transition_mask.shape[1])
        return arr[:h, :w], transition_mask[:h, :w]

    # ── 4. Calcular estadísticas por fecha en local ──────────────────────────
    # Para cada fecha:
    #   - fracción GLOBAL de píxeles malos en todo el AOI
    #   - fracción de píxeles malos SOLO dentro de la zona de transición
    # Si la global ≤ umbral se devuelve la global (fecha de referencia);
    # en caso contrario se devuelve la de transición (filtro mejorado).
    result = {}
    for i, date in enumerate(cube_dates):
        scl_arr = np.asarray(da.isel({t_dim: i}).values)

        bad_mask = np.isin(scl_arr, bad_classes)
        global_frac = float(bad_mask.sum()) / max(bad_mask.size, 1)

        if global_frac <= global_bad_fraction_threshold:
            pct = global_frac * 100
        else:
            bad_aligned, mask_aligned = _align_to_mask(bad_mask)
            n_trans = max(int(mask_aligned.sum()), 1)
            pct = float(bad_aligned[mask_aligned].sum()) / n_trans * 100

        result[date] = round(pct, 4)

    ds.close()
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    print(f"  Evaluacion completada: {len(result)} fechas")
    return result


def quantify_reference_gain(n_initial_valid, n_recovered, n_total_dates):
    n_final = n_initial_valid + n_recovered
    pct_initial = (100 * n_initial_valid / n_total_dates) if n_total_dates else 0
    pct_final = (100 * n_final / n_total_dates) if n_total_dates else 0
    gain_abs = n_recovered
    gain_rel = (100 * n_recovered / n_initial_valid) if n_initial_valid else 0

    print("\n══ Ganancia del filtro de transición ══")
    print(f"Fechas iniciales (filtro global): {n_initial_valid}")
    print(f"Fechas recuperadas:              {n_recovered}")
    print(f"Fechas finales válidas:          {n_final}")
    print(f"Cobertura inicial:               {pct_initial:.1f}%")
    print(f"Cobertura final:                 {pct_final:.1f}%")
    print(f"Ganancia relativa:               {gain_rel:.1f}%")

    return {
        "n_initial_valid": n_initial_valid,
        "n_recovered": n_recovered,
        "n_final_valid": n_final,
        "n_total_dates": n_total_dates,
        "pct_initial": pct_initial,
        "pct_final": pct_final,
        "gain_relative_pct": gain_rel,
        "gain_absolute": gain_abs,
    }


# UDF enviado al backend: recibe el stack SCL completo (t, y, x) y devuelve
# el water frequency 2D. Para cada píxel:
#   water_freq = nº obs. agua (SCL==6) / nº obs. claras (SCL in {4,5,6})
# Se ejecuta en el backend con reduce_dimension("t") → un ÚNICO job.
#
# Dos filtros anti-ruido:
#   1. valid_dates : si se pasa, solo se usan esas fechas (excluye escenas
#      nubladas descartadas por el filtro de transición).
#   2. min_obs     : píxeles con menos de min_obs observaciones claras se
#      marcan como NaN (evita frecuencias 0/1 espurias por 1-2 observaciones).
_WATER_FREQUENCY_UDF = r"""
import numpy as np
import xarray


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:

    water_class = context.get("water_class", 6)
    # Soportar múltiples clases de "agua" (ej: [6, 12] = agua + vegetación inundada)
    if not isinstance(water_class, list):
        water_class = [water_class]
    
    clear_classes = context.get("clear_classes", [4, 5, 6])
    valid_dates = context.get("valid_dates", None)
    min_obs = int(context.get("min_obs", 0))

    arr = cube.values

    # (t, bands, y, x) -> (t, y, x)
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]

    # ── Filtrar el stack temporal a solo las fechas válidas ───────────────────
    if valid_dates:
        tname = "t" if "t" in cube.dims else cube.dims[0]
        tcoords = np.asarray(cube.coords[tname].values)
        tstr = np.array([str(t)[:10] for t in tcoords])
        keep = np.isin(tstr, list(valid_dates))
        if keep.any():
            arr = arr[keep]

    water_votes = np.sum(np.isin(arr, water_class), axis=0).astype(np.float32)
    clear_votes = np.sum(np.isin(arr, clear_classes), axis=0).astype(np.float32)

    # Evitar división por cero
    safe = np.where(clear_votes == 0, 1.0, clear_votes)
    water_freq = (water_votes / safe).astype(np.float32)

    # Píxeles con muy pocas (o ninguna) observación clara -> NaN (ruido)
    threshold = max(1, min_obs)
    water_freq[clear_votes < threshold] = np.nan

    return xarray.DataArray(
        water_freq[np.newaxis, :, :],
        dims=["bands", "y", "x"],
        coords={
            "bands": ["water_frequency"],
            "y": cube.coords["y"],
            "x": cube.coords["x"],
        },
    )
"""


def compute_water_frequency_openeo(
    conn,
    bbox,
    time_extent,
    valid_dates=None,
    out_path="water_frequency.tif",
    force=False,
    scl_dir="tifs_scl",
    water_class=[6, 12],  # Agua abierta (6) + vegetación inundada/marisma (12)
    clear_classes=(4, 5, 6, 12),  # Incluir marisma como observación válida
    min_obs=8,
):
    """Calcula el water frequency raster ejecutando un UDF en el backend de
    OpenEO (reduce_dimension sobre la dimensión temporal) → un ÚNICO batch job.

    Parameters
    ----------
    valid_dates : list[str] | None
        Si se pasa, solo se usan esas fechas ('YYYY-MM-DD') para el cálculo,
        excluyendo escenas nubladas. Reduce drásticamente el ruido.
    min_obs : int
        Mínimo de observaciones claras por píxel; por debajo se marca NaN.

    Devuelve (water_freq, transform, crs).
    """
    if (not force) and os.path.exists(out_path):
        return RasterProcessor.load_reference_map(out_path)

    # Descargar el stack SCL completo y reducir en el backend en un solo job.
    cube_scl = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=time_extent,
        bands=["SCL"],
        max_cloud_cover=100,
    )

    udf = openeo.UDF(
        code=_WATER_FREQUENCY_UDF,
        runtime="Python",
        context={
            "water_class": water_class,
            "clear_classes": list(clear_classes),
            "valid_dates": list(valid_dates) if valid_dates else None,
            "min_obs": int(min_obs),
        },
    )

    wf_cube = cube_scl.reduce_dimension(dimension="t", reducer=udf)

    job = (
        wf_cube
        .save_result(format="GTiff")
        .create_job(title="water_frequency_scl")
    )

    print(
        f"  Lanzando batch job water frequency "
        f"({time_extent[0]} → {time_extent[1]})..."
    )
    job.start_and_wait()

    assets = job.get_results().get_assets()
    if not assets:
        raise RuntimeError("El job de water frequency no devolvió assets")

    assets[0].download(out_path)
    print(f"  Water frequency descargado -> {out_path}")

    with rasterio.open(out_path) as src:
        water_freq = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    return water_freq, transform, crs


# =====================================================================
# Water frequency por índice espectral (MNDWI) — variante para comparar
# con la clasificación SCL. La detección de agua usa MNDWI > umbral, pero
# el enmascarado de nubes (denominador de observaciones claras) sigue
# usando la máscara SCL: así el ÚNICO cambio respecto a la versión SCL es
# el criterio agua/no-agua (comparación A/B limpia).
# =====================================================================

_WATER_FREQUENCY_MNDWI_UDF = r"""
import numpy as np
import xarray


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:

    threshold = float(context.get("threshold", 0.0))
    clear_classes = context.get("clear_classes", [4, 5, 6, 12])
    valid_dates = context.get("valid_dates", None)
    min_obs = int(context.get("min_obs", 0))

    arr = cube.values.astype("float64")  # (t, bands, y, x)

    bnames = [str(b) for b in cube.coords["bands"].values]
    ib03 = bnames.index("B03")
    ib11 = bnames.index("B11")
    iscl = bnames.index("SCL")

    green = arr[:, ib03, :, :]
    swir = arr[:, ib11, :, :]
    scl = arr[:, iscl, :, :]

    # ── Filtrar a solo las fechas válidas ────────────────────────────────────
    if valid_dates:
        tname = "t" if "t" in cube.dims else cube.dims[0]
        tcoords = np.asarray(cube.coords[tname].values)
        tstr = np.array([str(t)[:10] for t in tcoords])
        keep = np.isin(tstr, list(valid_dates))
        if keep.any():
            green = green[keep]
            swir = swir[keep]
            scl = scl[keep]

    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    den = green + swir
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = np.where(den != 0, (green - swir) / den, np.nan)

    # SCL puede venir remuestreado (20 m -> 10 m); redondear a clase entera.
    scl_int = np.round(scl).astype(np.int16)

    clear = np.isin(scl_int, clear_classes)                    # nubes fuera (igual que SCL)
    water = clear & np.isfinite(mndwi) & (mndwi > threshold)    # agua por MNDWI

    water_votes = np.sum(water, axis=0).astype(np.float32)
    clear_votes = np.sum(clear, axis=0).astype(np.float32)

    safe = np.where(clear_votes == 0, 1.0, clear_votes)
    water_freq = (water_votes / safe).astype(np.float32)

    threshold_obs = max(1, min_obs)
    water_freq[clear_votes < threshold_obs] = np.nan

    return xarray.DataArray(
        water_freq[np.newaxis, :, :],
        dims=["bands", "y", "x"],
        coords={
            "bands": ["water_frequency"],
            "y": cube.coords["y"],
            "x": cube.coords["x"],
        },
    )
"""


def compute_water_frequency_mndwi_openeo(
    conn,
    bbox,
    time_extent,
    valid_dates=None,
    out_path="water_frequency_mndwi.tif",
    force=False,
    threshold=0.0,
    clear_classes=(4, 5, 6, 12),
    min_obs=8,
):
    """Water frequency detectando agua por MNDWI (B03/B11) en lugar de la clase
    SCL, manteniendo la máscara de nubes SCL. Misma firma/uso que
    ``compute_water_frequency_openeo``.

    Parameters
    ----------
    threshold : float
        Umbral de MNDWI para considerar un píxel como agua (default 0.0, Xu 2006).
    clear_classes : tuple[int]
        Clases SCL consideradas observación clara (denominador), idéntico a la
        versión SCL para que la comparación sea justa.
    min_obs : int
        Mínimo de observaciones claras por píxel; por debajo se marca NaN.

    Devuelve (water_freq, transform, crs).
    """
    if (not force) and os.path.exists(out_path):
        return RasterProcessor.load_reference_map(out_path)

    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=time_extent,
        bands=["B03", "B11", "SCL"],
        max_cloud_cover=100,
    )

    # Filtrar a las fechas válidas EN LA CARGA (no dentro del UDF): así el
    # backend solo lee esas escenas en vez de todo el time_extent. Cargar
    # B03 a 10 m para muchas escenas es lo que hace lento el job.
    if valid_dates:
        try:
            cube = cube.filter_labels(
                dimension="t",
                condition=lambda x: x.isin(list(valid_dates)),
            )
        except Exception as exc:
            print(f"  (filter_labels no disponible; se filtrará en el UDF): {exc}")

    udf = openeo.UDF(
        code=_WATER_FREQUENCY_MNDWI_UDF,
        runtime="Python",
        context={
            "threshold": float(threshold),
            "clear_classes": list(clear_classes),
            "valid_dates": list(valid_dates) if valid_dates else None,
            "min_obs": int(min_obs),
        },
    )

    wf_cube = cube.reduce_dimension(dimension="t", reducer=udf)

    job = (
        wf_cube
        .save_result(format="GTiff")
        .create_job(title="water_frequency_mndwi")
    )

    print(
        f"  Lanzando batch job water frequency MNDWI "
        f"({time_extent[0]} → {time_extent[1]})..."
    )
    job.start_and_wait()

    assets = job.get_results().get_assets()
    if not assets:
        raise RuntimeError("El job de water frequency (MNDWI) no devolvió assets")

    assets[0].download(out_path)
    print(f"  Water frequency (MNDWI) descargado -> {out_path}")

    with rasterio.open(out_path) as src:
        water_freq = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    return water_freq, transform, crs


# =====================================================================
# Water frequency MULTI-MÉTODO en un solo job (para comparar de forma justa)
# Baja las bandas una sola vez y calcula, sobre las MISMAS fechas y la MISMA
# máscara de nubes SCL, el water frequency con varios criterios de agua:
#   scl   : clase de agua del SCL (referencia)
#   ndwi  : (B03 - B08)/(B03 + B08) > umbral        (McFeeters 1996)
#   mndwi : (B03 - B11)/(B03 + B11) > umbral        (Xu 2006)
#   awei  : 4(B03-B11) - (0.25·B08 + 2.75·B12) > 0  (Feyisa 2014, AWEI_nsh)
# Devuelve una banda de water frequency por método -> comparación directa.
# =====================================================================

_WATER_FREQUENCY_MULTI_UDF = r"""
import numpy as np
import xarray


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:

    methods = context.get("methods", ["scl", "ndwi", "mndwi", "awei"])
    thresholds = context.get("thresholds", {})
    scl_water = context.get("scl_water_classes", [6, 12])
    clear_classes = context.get("clear_classes", [4, 5, 6, 12])
    valid_dates = context.get("valid_dates", None)
    min_obs = int(context.get("min_obs", 0))

    arr = cube.values.astype("float64")  # (t, bands, y, x)
    bn = [str(b) for b in cube.coords["bands"].values]

    def band(name):
        return arr[:, bn.index(name), :, :]

    green = band("B03"); red = band("B04"); nir = band("B08")
    swir1 = band("B11"); swir2 = band("B12"); scl = band("SCL")

    if valid_dates:
        tname = "t" if "t" in cube.dims else cube.dims[0]
        tstr = np.array([str(t)[:10] for t in np.asarray(cube.coords[tname].values)])
        keep = np.isin(tstr, list(valid_dates))
        if keep.any():
            green, red, nir, swir1, swir2, scl = (
                green[keep], red[keep], nir[keep], swir1[keep], swir2[keep], scl[keep]
            )

    scl_int = np.round(scl).astype(np.int16)
    clear = np.isin(scl_int, clear_classes)

    # Escala de reflectancia (AWEI es lineal -> necesita [0,1]; NDWI/MNDWI son
    # razones y no dependen de la escala).
    finite_g = green[np.isfinite(green)]
    scale = 10000.0 if (finite_g.size and np.nanmax(finite_g) > 1.5) else 1.0
    g, r, n = green / scale, red / scale, nir / scale
    s1, s2 = swir1 / scale, swir2 / scale

    with np.errstate(invalid="ignore", divide="ignore"):
        d_ndwi = g + n
        ndwi = np.where(d_ndwi != 0, (g - n) / d_ndwi, np.nan)
        d_mndwi = g + s1
        mndwi = np.where(d_mndwi != 0, (g - s1) / d_mndwi, np.nan)
        awei = 4.0 * (g - s1) - (0.25 * n + 2.75 * s2)

    clear_votes = np.sum(clear, axis=0).astype("float32")
    safe = np.where(clear_votes == 0, 1.0, clear_votes)
    threshold_obs = max(1, min_obs)

    def water_freq(water_mask):
        wv = np.sum(clear & water_mask, axis=0).astype("float32")
        wf = (wv / safe).astype("float32")
        wf[clear_votes < threshold_obs] = np.nan
        return wf

    out_bands, out_names = [], []
    for m in methods:
        if m == "scl":
            water = np.isin(scl_int, scl_water)
        elif m == "ndwi":
            water = np.isfinite(ndwi) & (ndwi > float(thresholds.get("ndwi", 0.0)))
        elif m == "mndwi":
            water = np.isfinite(mndwi) & (mndwi > float(thresholds.get("mndwi", 0.0)))
        elif m == "awei":
            water = np.isfinite(awei) & (awei > float(thresholds.get("awei", 0.0)))
        else:
            continue
        out_bands.append(water_freq(water))
        out_names.append("wf_" + m)

    stacked = np.stack(out_bands, axis=0)

    return xarray.DataArray(
        stacked,
        dims=["bands", "y", "x"],
        coords={"bands": out_names, "y": cube.coords["y"], "x": cube.coords["x"]},
    )
"""


def compute_water_frequency_multi_openeo(
    conn,
    bbox,
    time_extent,
    valid_dates=None,
    methods=("scl", "ndwi", "mndwi", "awei"),
    thresholds=None,
    clear_classes=(4, 5, 6, 12),
    scl_water_classes=(6, 12),
    min_obs=8,
    max_cloud_cover=40,
    out_path="water_frequency_multi.tif",
    force=False,
):
    """Calcula el water frequency con VARIOS métodos de detección de agua en un
    único batch job (baja las bandas una sola vez). Todos comparten las mismas
    fechas y la misma máscara de nubes SCL -> comparación justa.

    Returns
    -------
    (wf_dict, transform, crs)
        wf_dict = {"scl": arr, "ndwi": arr, "mndwi": arr, "awei": arr}
    """
    methods = list(methods)
    thresholds = dict(thresholds or {"ndwi": 0.0, "mndwi": 0.0, "awei": 0.0})

    if (not force) and os.path.exists(out_path):
        # El GTiff se guarda en el orden de 'methods' (los nombres de banda no
        # siempre se conservan), así que se keyea por posición.
        with rasterio.open(out_path) as src:
            if src.count != len(methods):
                raise ValueError(
                    f"'{out_path}' tiene {src.count} bandas pero methods={methods}. "
                    "Usa force=True o ajusta 'methods' al fichero."
                )
            arrs = {methods[i]: src.read(i + 1).astype(np.float32) for i in range(src.count)}
            return arrs, src.transform, src.crs

    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=time_extent,
        bands=["B03", "B04", "B08", "B11", "B12", "SCL"],
        max_cloud_cover=max_cloud_cover,
    )

    if valid_dates:
        try:
            cube = cube.filter_labels(
                dimension="t", condition=lambda x: x.isin(list(valid_dates))
            )
        except Exception as exc:
            print(f"  (filter_labels no disponible; se filtrará en el UDF): {exc}")

    udf = openeo.UDF(
        code=_WATER_FREQUENCY_MULTI_UDF,
        runtime="Python",
        context={
            "methods": methods,
            "thresholds": thresholds,
            "clear_classes": list(clear_classes),
            "scl_water_classes": list(scl_water_classes),
            "valid_dates": list(valid_dates) if valid_dates else None,
            "min_obs": int(min_obs),
        },
    )

    wf_cube = cube.reduce_dimension(dimension="t", reducer=udf)

    job = (
        wf_cube.save_result(format="GTiff").create_job(title="water_frequency_multi")
    )
    print(f"  Lanzando batch job multi-método ({', '.join(methods)})...")
    job.start_and_wait()

    assets = job.get_results().get_assets()
    if not assets:
        raise RuntimeError("El job multi-método no devolvió assets")
    assets[0].download(out_path)
    print(f"  Descargado -> {out_path}")

    with rasterio.open(out_path) as src:
        arrs = {}
        for i in range(src.count):
            arrs[methods[i]] = src.read(i + 1).astype(np.float32)
        transform, crs = src.transform, src.crs

    return arrs, transform, crs


def get_water_centroid(obj):
    if isinstance(obj, Polygon):
        c = obj.centroid
        return c.x, c.y
    raise TypeError("get_water_centroid espera un shapely Polygon")

def get_tidal_coverage_percentage(date, rgb_dir):
    
    # Calcular cobertura del tile (% de píxeles válidos)
    cobertura_pct = 0.0
    tif_rgb_path = rgb_dir / f"rgb_{date}.tif"
    
    if tif_rgb_path.exists():
        try:
            with rasterio.open(tif_rgb_path) as src:
                rgb = np.dstack([src.read(i) for i in [1, 2, 3]])
                valid_pixels = ~np.isnan(rgb).any(axis=2)
                cobertura_pct = (valid_pixels.sum() / valid_pixels.size) * 100
        except Exception as e:
            print(f"⚠️  Error leyendo {date} para cobertura: {e}")
    return cobertura_pct
