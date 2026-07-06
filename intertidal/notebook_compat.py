"""Compatibilidad para notebooks legacy sobre la arquitectura modular.

Este módulo concentra los adaptadores necesarios para ejecutar notebooks
históricos sin mantener funciones auxiliares dentro de las celdas.
"""

from __future__ import annotations

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


def download_date_rgb(conn, date, bbox, out_dir):
    client = OpenEOClient()
    client.connection = conn
    return client.download_rgb(date, bbox, out_dir)


def download_date_scl(conn, date, bbox, out_dir):
    client = OpenEOClient()
    client.connection = conn
    return client.download_scl(date, bbox, out_dir)


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
    reference_transform,
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
        from .overpass import get_overpass_times

        overpass = get_overpass_times(bbox, time_extent)
        all_dates = sorted(overpass.keys())

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

    water_votes = np.sum(arr == water_class, axis=0).astype(np.float32)
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
    water_class=6,
    clear_classes=(4, 5, 6),
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


def get_water_centroid(obj):
    if isinstance(obj, Polygon):
        c = obj.centroid
        return c.x, c.y
    raise TypeError("get_water_centroid espera un shapely Polygon")
