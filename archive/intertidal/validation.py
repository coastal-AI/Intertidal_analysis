"""
validation.py — Comparación de métodos de detección de agua y validación
=======================================================================

Utilidades para comparar los distintos criterios de agua del *water frequency*
(SCL, NDWI, MNDWI, AWEI) y validarlos contra un Modelo Digital del Terreno de
referencia (LiDAR del IGN).

Toda la lógica reutilizable vive aquí; los notebooks solo orquestan y pintan.

Funciones
---------
- download_mdt_ign        : descarga un recorte del MDT del IGN (WCS).
- reproject_to_grid       : reproyecta un ráster al grid de destino.
- validate_wf_vs_elevation: correlación WF <-> elevación por método (métrica objetiva).
- intertidal_from_wf      : máscara intermareal por método (transición ∩ WF∈[low,high]).
- pairwise_iou            : IoU (solape) entre las máscaras de los métodos.
"""

from __future__ import annotations

import os
import itertools

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
import requests


# WCS del Modelo Digital del Terreno del IGN (INSPIRE).
IGN_MDT_WCS = "https://servicios.idee.es/wcs-inspire/mdt"


def download_mdt_ign(
    bbox: dict,
    out_path: str,
    coverage: str = "Elevacion4258_5",
    timeout: int = 120,
    force: bool = False,
) -> str:
    """
    Descarga un recorte del MDT del IGN (WCS) para el bbox dado.

    Parameters
    ----------
    bbox
        Dict con ``west/south/east/north`` y ``crs`` (p.ej. EPSG:32629).
    out_path
        Ruta del GeoTIFF de salida.
    coverage
        Cobertura WCS. ``Elevacion4258_5`` = MDT 5 m en ETRS89 geográfico.
    force
        Si False y el fichero existe, no vuelve a descargar.

    Returns
    -------
    str
        ``out_path``.
    """
    if (not force) and os.path.exists(out_path):
        return out_path

    tf = Transformer.from_crs(bbox["crs"], "EPSG:4258", always_xy=True)
    west, south = tf.transform(bbox["west"], bbox["south"])
    east, north = tf.transform(bbox["east"], bbox["north"])

    params = [
        ("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
        ("coverageId", coverage),
        ("subset", f"Lat({south},{north})"),
        ("subset", f"Long({west},{east})"),
        ("format", "image/tiff"),
    ]

    resp = requests.get(IGN_MDT_WCS, params=params, timeout=timeout)
    resp.raise_for_status()

    with open(out_path, "wb") as handle:
        handle.write(resp.content)

    return out_path


def reproject_to_grid(
    src_path: str,
    dst_transform,
    dst_crs,
    dst_shape,
    resampling=Resampling.bilinear,
    nodata_below: float = -1000.0,
) -> np.ndarray:
    """
    Reproyecta la banda 1 de un ráster al grid destino (transform/crs/shape).
    Los valores por debajo de ``nodata_below`` se ponen a NaN.
    """
    out = np.full(dst_shape, np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    return np.where(out < nodata_below, np.nan, out)


def validate_wf_vs_elevation(
    wf_dict: dict,
    elevation: np.ndarray,
    mask: np.ndarray,
    z_range: tuple[float, float] = (0.3, 5.0),
) -> dict:
    """
    Correlación *water frequency* <-> elevación por método.

    En el intermareal el WF debe decrecer con la cota; la correlación (Spearman)
    más negativa indica el método más consistente con la topografía real. Se
    restringe a ``mask`` (p.ej. zona de transición) ∩ flats expuestos con
    elevación en ``z_range`` (donde el LiDAR tiene topografía real, no agua).

    Returns
    -------
    dict
        ``{method: {"spearman": float, "pearson": float, "n": int}}``
    """
    from scipy.stats import spearmanr

    z_lo, z_hi = z_range
    zone = mask & np.isfinite(elevation) & (elevation > z_lo) & (elevation <= z_hi)

    results = {}
    for method, wf in wf_dict.items():
        valid = zone & np.isfinite(wf)
        if valid.sum() < 3:
            results[method] = {"spearman": np.nan, "pearson": np.nan, "n": int(valid.sum())}
            continue
        results[method] = {
            "spearman": float(spearmanr(wf[valid], elevation[valid]).correlation),
            "pearson": float(np.corrcoef(wf[valid], elevation[valid])[0, 1]),
            "n": int(valid.sum()),
        }
    return results


def intertidal_from_wf(
    wf_dict: dict,
    transition: np.ndarray,
    low: float,
    high: float,
) -> dict:
    """
    Máscara intermareal por método, igual que el pipeline (celda 17):
    zona de transición ∩ ``low <= WF <= high``.

    Returns
    -------
    dict
        ``{method: mask_bool}``
    """
    return {
        method: transition & (wf >= low) & (wf <= high) & np.isfinite(wf)
        for method, wf in wf_dict.items()
    }


def pairwise_iou(masks: dict) -> dict:
    """
    IoU (intersección / unión) entre las máscaras de cada par de métodos.

    Returns
    -------
    dict
        ``{(method_a, method_b): iou_float}``
    """
    out = {}
    for a, b in itertools.combinations(masks, 2):
        inter = (masks[a] & masks[b]).sum()
        union = (masks[a] | masks[b]).sum()
        out[(a, b)] = float(inter / max(union, 1))
    return out
