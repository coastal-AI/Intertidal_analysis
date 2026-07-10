"""
overpass.py — Hora exacta de paso de Sentinel-2 sobre un AOI
=============================================================

Consulta el catálogo STAC de Copernicus para obtener el datetime real
de adquisición de cada escena Sentinel-2 L2A. La hora se extrae del
nombre del producto (e.g. S2A_MSIL2A_20240104T110421_...) porque el
campo `datetime` del catálogo suele estar normalizado a 00:00:00 UTC.
"""

import re
import requests
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"


def _create_session_with_retries():
    """Crea una sesión de requests con reintentos automáticos."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # 5 reintentos
        backoff_factor=2,  # Espera 2, 4, 8, 16, 32 segundos entre reintentos
        status_forcelist=[429, 500, 502, 503, 504],  # Códigos HTTP que disparan retry
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_overpass_times(bbox: dict, time_extent: list[str]) -> dict[str, datetime]:
    """
    Devuelve la hora UTC exacta de adquisición de cada escena S-2 L2A.

    Parameters
    ----------
    bbox : dict
        Bounding box con claves west, south, east, north.
    time_extent : list[str]
        [fecha_inicio, fecha_fin] en formato 'YYYY-MM-DD'.

    Returns
    -------
    dict[str, datetime]
        {fecha 'YYYY-MM-DD': datetime con hora UTC real de adquisición}
        Si hay varias pasadas en el mismo día se conserva la primera.

    Examples
    --------
    >>> times = get_overpass_times(bbox, ["2024-01-01", "2024-12-31"])
    >>> print(times["2024-07-02"])
    2024-07-02 11:04:21
    """
    overpass: dict[str, datetime] = {}

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
        "datetime": f"{time_extent[0]}T00:00:00Z/{time_extent[1]}T23:59:59Z",
        "limit": 200,
    }

    session = _create_session_with_retries()
    url: str | None = STAC_SEARCH
    
    try:
        while url:
            resp = session.post(url, json=body, timeout=120)  # Timeout aumentado a 120s
            resp.raise_for_status()
            data = resp.json()

            for feature in data.get("features", []):
                # La hora real está en el nombre del producto: _YYYYMMDDTHHMMSS_
                title = feature["properties"].get("title", "") or feature.get("id", "")
                # Filtrar solo productos L2A (Nivel 2A) en local: su nombre
                # contiene 'MSIL2A' (evita el filtro CQL2 que el backend rechaza).
                if "MSIL2A" not in title:
                    continue
                match = re.search(r"_(\d{8}T\d{6})_", title)
                if not match:
                    continue
                dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
                date_key = dt.strftime("%Y-%m-%d")
                if date_key not in overpass:
                    overpass[date_key] = dt

            next_link = next(
                (lk for lk in data.get("links", []) if lk.get("rel") == "next"), None
            )
            # En paginación POST, el 'next' incluye el cuerpo (con el token de
            # paginación) en next_link["body"]; se reutiliza para la siguiente
            # petición. El href apunta a la misma URL de búsqueda.
            if next_link:
                url = next_link.get("href", STAC_SEARCH)
                body = next_link.get("body", body)
            else:
                url = None
    finally:
        session.close()

    return dict(sorted(overpass.items()))


def overpass_hour_utc(bbox: dict, time_extent: list[str]) -> float:
    """
    Devuelve la hora UTC media de paso del satélite (como float, e.g. 11.07).

    Útil como valor por defecto cuando no tienes el diccionario completo.
    """
    times = get_overpass_times(bbox, time_extent)
    if not times:
        raise ValueError("No se encontraron escenas para el AOI y periodo dados.")
    hours = [dt.hour + dt.minute / 60 + dt.second / 3600 for dt in times.values()]
    return sum(hours) / len(hours)
