"""
utils.py
========
Funciones puras de utilidad para el pipeline de análisis Sentinel-2.
Incluye:
- Conversión de coordenadas DMS a decimal   
- Creación de cuadrícula de celdas a partir de un polígono
- Validación de archivos GeoTIFF (presencia de datos válidos)
- Lectura de GeoTIFFs multibanda para visualización RGB
- Descarga de escenas RGB y SCL desde OpenEO
- Normalización de bandas para visualización
- Visualización de cuadrícula RGB con recorte opcional al AOI
- Cálculo de estadísticas SCL y visualización de mapas SCL con leyenda  
"""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import path
import re
import time
from pathlib import Path
from typing import Sequence
import geopandas as gpd
import xarray as xr
from rasterio.mask import geometry_mask
import numpy as np
from pyproj import Transformer
import rasterio
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import asyncio
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import rioxarray
import contextily as ctx
from pyproj import Transformer
import openeo
from math import ceil

# ── Coordenadas ────────────────────────────────────────────────────────────────

def dms_to_coords(coord_str: str) -> tuple[float, float]:
    """
    Convierte una cadena DMS → (lon, lat) decimal.

    Formato esperado:
        '43°35\\'46.50"N 5°43\\'40.94"W'

    Returns:
        (lon, lat) como floats — orden Shapely (x, y).

    Raises:
        ValueError: si el formato no coincide.
    """
    pattern = (
        r'(\d+)°(\d+)\'([\d.]+)"([NS])\s+'
        r'(\d+)°(\d+)\'([\d.]+)"([EW])'
    )
    m = re.match(pattern, coord_str.strip())
    if not m:
        raise ValueError(f"Formato DMS no válido: '{coord_str}'")

    lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = m.groups()

    lat = float(lat_d) + float(lat_m) / 60 + float(lat_s) / 3600
    lon = float(lon_d) + float(lon_m) / 60 + float(lon_s) / 3600

    if lat_dir == "S":
        lat *= -1
    if lon_dir == "W":
        lon *= -1

    return (lon, lat)

def dms_to_decimal(dms_str: str) -> tuple[float, float]:
    """
    Convierte una cadena DMS tipo '43°35\'46.50"N 5°43\'40.94"W'
    a (latitud, longitud) en grados decimales.
    """
    pattern = r'(\d+)°(\d+)\'([\d.]+)"([NS])\s+(\d+)°(\d+)\'([\d.]+)"([EW])'
    m = re.match(pattern, dms_str.strip())
    if not m:
        raise ValueError(f"No se puede parsear: {dms_str}")
    lat_d, lat_m, lat_s, lat_hem = m.group(1,2,3,4)
    lon_d, lon_m, lon_s, lon_hem = m.group(5,6,7,8)
    lat = float(lat_d) + float(lat_m)/60 + float(lat_s)/3600
    lon = float(lon_d) + float(lon_m)/60 + float(lon_s)/3600
    if lat_hem == 'S': lat = -lat
    if lon_hem == 'W': lon = -lon
    return lat, lon


#  ── Raster ─────────────────────────────────────────────────────────────────
def make_grid(polygon, cell_size_deg=0.01, crs="EPSG:4326"):
    """Divide el bbox del AOI en celdas rectangulares."""
    minx, miny, maxx, maxy = polygon.bounds
    cols = np.arange(minx, maxx, cell_size_deg)
    rows = np.arange(miny, maxy, cell_size_deg)
    cells = [
        box(x, y, x + cell_size_deg, y + cell_size_deg)
        for x in cols for y in rows
    ]
    grid = gpd.GeoDataFrame(geometry=cells,crs=crs)
    # Solo celdas que intersectan el AOI
    return grid[grid.intersects(polygon)].reset_index(drop=True)


# ── Validación de TIF ─────────────────────────────────────────────────────────
def is_valid_tif(path: str | Path) -> bool:
    """
    Devuelve True si el TIF contiene al menos un píxel con valor > 0.
    Útil para descartar fechas con datos vacíos o descarga fallida.
    """
    with rasterio.open(path) as src:
        data = src.read()
    return bool(data.max() > 0)


# Leer un GeoTIFF multibanda y devolver un array RGB normalizado para imshow
def tif_to_rgb(tif_path: str) -> np.ndarray | None:
    """
    Lee un GeoTIFF multibanda (B04, B03, B02 = R, G, B) y devuelve
    un array HxWx3 float32 listo para imshow, o None si falla.
    """
    if not os.path.exists(tif_path):
        return None
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()  # (bandas, H, W)
        r = norm_percentile(data[0].astype(float))
        g = norm_percentile(data[1].astype(float))
        b = norm_percentile(data[2].astype(float))
        return np.stack([r, g, b], axis=-1).astype(np.float32)
    except Exception as e:
        print(f"   Error leyendo {tif_path}: {e}")
        return None


def tif_to_scl(tif_path: str) -> np.ndarray | None:
    """
    Lee un GeoTIFF de SCL (banda única) y devuelve array 2D, o None si falla.
    """
    if not os.path.exists(tif_path):
        return None
    try:
        with rasterio.open(tif_path) as src:
            return src.read(1).astype(np.uint8)
    except Exception as e:
        print(f"    Error leyendo SCL {tif_path}: {e}")
        return None

# ── Polígonos y bounding boxes ─────────────────────────────────────────────────
# Crear un polígono a partir de coordenadas DMS
def make_polygon(dms_list: list[str]) -> Polygon:
    """Crea un Polygon de Shapely a partir de una lista de strings DMS."""
    coords = [dms_to_decimal(d) for d in dms_list]
    # Shapely usa (lon, lat)
    return Polygon([(lon, lat) for lat, lon in coords])

# Obtener bbox en formato OpenEO a partir de un polígono
def bbox_from_polygon(polygon: Polygon) -> dict:
    """Devuelve el bounding box de un polígono en formato OpenEO."""
    minx, miny, maxx, maxy = polygon.bounds
    return {"west": minx, "south": miny, "east": maxx, "north": maxy}

# ── OpenEO: descarga de escenas ───────────────────────────────────────────────

# Descargar la imagen RGB (B04, B03, B02) de una fecha concreta como GeoTIFF.
def download_date_rgb(conn, date: str, bbox: dict, out_dir: str) -> str:
    """
    Descarga la imagen RGB (B04, B03, B02) de una fecha concreta como GeoTIFF.
    Devuelve 'ok', 'skipped' o 'error: <msg>'.
    """
    out_path = os.path.join(out_dir, f"rgb_{date}.tif")
    if os.path.exists(out_path):
        print(f"    {date} — ya existe, saltando")
        return "skipped"

    try:
        cube = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=bbox,
            temporal_extent=[date, date],
            bands=["B04", "B03", "B02"],
            max_cloud_cover=100,
        )
        # Reducir la dimensión temporal (solo hay una fecha)
        cube = cube.reduce_dimension(dimension="t", reducer="mean")

        job = cube.save_result(format="GTiff").create_job(title=f"rgb_{date}")
        job.start_and_wait()

        results = job.get_results()
        assets  = results.get_assets()
        if not assets:
            return "error: sin assets"

        # Guardar el primer asset (el GeoTIFF)
        assets[0].download(out_path)
        print(f"   {date} → {out_path}")
        return "ok"

    except Exception as e:
        print(f"   {date} — error: {e}")
        return f"error: {e}"

# Descargar la banda SCL de una fecha concreta como GeoTIFF.
def download_date_scl(conn, date: str, bbox: dict, out_dir: str) -> str:
    """
    Descarga la banda SCL de una fecha concreta como GeoTIFF.
    """
    out_path = os.path.join(out_dir, f"scl_{date}.tif")
    if os.path.exists(out_path):
        print(f"    {date} — SCL ya existe, saltando")
        return "skipped"

    try:
        cube = conn.load_collection(
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
    
# ── Normalización por percentiles para visualización de bandas RGB ─────────────
def norm_percentile(band, pmin=2, pmax=98):
    """Normaliza una banda usando percentiles para mejorar la visualización.
    Ignora valores NaN esenciales (<=0) al calcular los percentiles."""
    valid = np.isfinite(band) & (band > 0)

    if valid.sum() == 0:
        return np.zeros_like(band)

    vmin = np.nanpercentile(band[valid], pmin)
    vmax = np.nanpercentile(band[valid], pmax)

    if vmax <= vmin:
        return np.zeros_like(band)

    out = (band - vmin) / (vmax - vmin)

    return np.clip(out, 0, 1)

# ── Visualización ─────────────────────────────────────────────────────────────
def plot_rgb_grid(
    dates,
    rgb_dir,
    polygon=None,
    cols=4,
    figsize_per_cell=3.5,
    title="Serie temporal Sentinel-2",
):
    """
    Dibuja una cuadrícula de imágenes RGB para una serie temporal de fechas.
    Cada imagen se lee de un GeoTIFF con nombre 'rgb_{date}.tif' en el directorio rgb_dir.
    Opcionalmente, se puede recortar cada imagen al polígono dado (en coordenadas geográficas) usando geometry_mask.
    """
    n = len(dates)

    if n == 0:
        print("No hay imágenes.")
        return

    rows = ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            cols * figsize_per_cell,
            rows * figsize_per_cell,
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for i, date in enumerate(dates):

        ax = axes[i]

        tif_path = os.path.join(
            rgb_dir,
            f"rgb_{date}.tif"
        )

        if not os.path.exists(tif_path):

            ax.text(
                0.5,
                0.5,
                "sin datos",
                ha="center",
                va="center",
            )

            ax.axis("off")
            continue

        with rasterio.open(tif_path) as src:

            rgb = src.read(
                [1, 2, 3]
            ).transpose(
                1, 2, 0
            ).astype(np.float32)

            rgb[rgb < 0] = np.nan

            if polygon is not None:

                poly_proj = (
                    gpd.GeoSeries(
                        [polygon],
                        crs="EPSG:4326",
                    )
                    .to_crs(src.crs)
                    .iloc[0]
                )

                mask = geometry_mask(
                    [poly_proj],
                    transform=src.transform,
                    invert=True,
                    out_shape=(
                        src.height,
                        src.width,
                    ),
                )

                rgb[~mask] = np.nan

            rgb = np.stack(
                [
                    norm_percentile(rgb[..., 0]),
                    norm_percentile(rgb[..., 1]),
                    norm_percentile(rgb[..., 2]),
                ],
                axis=-1,
            )

            # exterior AOI blanco
            rgb[np.isnan(rgb)] = 1.0

            ax.imshow(rgb)

        ax.set_title(
            date,
            fontsize=8,
        )

        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.show()


# ── SCL: clases consideradas "malas" para análisis (nubes, sombras, agua, etc.)
def compute_scl_stats(scl_array: np.ndarray, bad_classes: list) -> dict:
    """
    Calcula estadísticas SCL para una escena:
      - bad_fraction: fracción [0-1] de píxeles en clases malas
      - class_counts: {clase: n_pixels}
    """
    total = scl_array.size
    bad_mask = np.isin(scl_array, bad_classes)
    bad_frac = bad_mask.sum() / total

    unique, counts = np.unique(scl_array, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    return {
        "bad_fraction": float(bad_frac),
        "bad_pct": float(bad_frac * 100),
        "class_counts": class_counts,
    }

def plot_scl_map(
    date: str,
    scl_dir: str,
    scl_colors: dict,
    scl_bad_classes: list,
    scl_stats: dict,
    scl_max_bad_fraction: float,
    polygon=None,
):
    """
    Dibuja el mapa SCL de una fecha con colores ESA y
    recorte opcional al AOI.
    """

    scl_path = os.path.join(
        scl_dir,
        f"scl_{date}.tif"
    )

    if not os.path.exists(scl_path):
        print(f"SCL no disponible para {date}")
        return

    with rasterio.open(scl_path) as src:

        scl_arr = src.read(1).astype(float)

        if polygon is not None:

            poly_proj = (
                gpd.GeoSeries(
                    [polygon],
                    crs="EPSG:4326",
                )
                .to_crs(src.crs)
                .iloc[0]
            )

            mask = geometry_mask(
                [poly_proj],
                transform=src.transform,
                invert=True,
                out_shape=(
                    src.height,
                    src.width,
                ),
            )

            scl_arr[~mask] = np.nan

    # -----------------------------------
    # Colormap ESA
    # -----------------------------------

    color_list = [
        scl_colors.get(c, ("?", "#aaaaaa"))[1]
        for c in range(12)
    ]

    cmap = mcolors.ListedColormap(color_list)

    # exterior AOI blanco
    cmap.set_bad("white")

    norm = mcolors.BoundaryNorm(
        boundaries=list(range(13)),
        ncolors=12,
    )

    classes_present = sorted(
        [
            int(c)
            for c in np.unique(scl_arr[np.isfinite(scl_arr)])
        ]
    )

    fig, (ax_scl, ax_legend) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    ax_scl.imshow(
        scl_arr,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax_scl.set_title(
        f"SCL — {date}",
        fontsize=11,
        fontweight="bold",
    )

    ax_scl.axis("off")

    patches = []

    for c in classes_present:

        label, hex_color = scl_colors.get(
            c,
            (f"Clase {c}", "#aaaaaa"),
        )

        is_bad = c in scl_bad_classes

        patches.append(
            mpatches.Patch(
                color=hex_color,
                label=f"{c} — {label}{' MALO' if is_bad else ''}",
            )
        )

    ax_legend.legend(
        handles=patches,
        loc="center",
        fontsize=8,
        frameon=False,
        title="Clases presentes",
        title_fontsize=9,
    )

    ax_legend.axis("off")

    stats = scl_stats.get(date)

    if stats:

        ax_scl.set_xlabel(
            f"Píxeles malos: {stats['bad_pct']:.1f}%  |  "
            f"{'VÁLIDA' if stats['bad_fraction'] <= scl_max_bad_fraction else 'DESCARTADA'}",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()