"""
overpass.py — Exact Sentinel-2 acquisition times over an AOI
============================================================

Elevation-from-satellite methods need the tide level at the MOMENT each
image was taken, so a date alone is not enough — we need the acquisition
*time*.

This module queries the Copernicus STAC catalogue and extracts the real UTC
acquisition time from the PRODUCT NAME (e.g.
``S2A_MSIL2A_20240104T110421_...``) rather than from the catalogue's
``datetime`` field, because that field is often normalised to 00:00:00Z and
would silently give every scene the same (wrong) tide.

The query is metadata-only: it costs nothing in processing credits and takes
seconds even for a decade of imagery.
"""

from __future__ import annotations

import re
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .net import use_system_certificates

#: Copernicus Data Space STAC search endpoint.
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"


def _session_with_retries():
    """A requests session that retries transient STAC failures.

    Five attempts with exponential backoff (2, 4, 8, 16, 32 s) on the status
    codes that indicate throttling or a temporary server problem. Without
    this, a decade-long query occasionally dies halfway through pagination.

    Certificate validation goes through the operating system's trust store —
    see :func:`pyintertidal.net.use_system_certificates` for why.
    """
    use_system_certificates()
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_overpass_times(bbox, time_extent, verbose=True):
    """Real UTC acquisition time of every Sentinel-2 L2A scene over an AOI.

    Parameters
    ----------
    bbox : dict
        Bounding box with ``west``/``south``/``east``/``north`` (WGS84).
        An :class:`~pyintertidal.aoi.AOI` exposes exactly this via
        ``aoi.bbox``.
    time_extent : (start, end)
        ISO dates ``'YYYY-MM-DD'``.
    verbose : bool
        Print a diagnostic when the catalogue returns nothing.

    Returns
    -------
    dict
        ``{'YYYY-MM-DD': datetime}`` sorted by date. When a day has several
        overpasses (possible near swath overlaps) the FIRST one is kept —
        consistent with how the cube keeps one scene per date.

    Notes
    -----
    Products are filtered to L2A locally (by the ``MSIL2A`` marker in the
    title) instead of with a CQL2 server-side filter, which this backend
    rejects. Pagination follows the STAC ``next`` link, whose POST body
    carries the paging token.
    """
    overpass: dict[str, datetime] = {}

    try:
        bbox_list = [float(bbox["west"]), float(bbox["south"]),
                     float(bbox["east"]), float(bbox["north"])]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid bbox: {exc}") from exc

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox_list,
        "datetime": f"{time_extent[0]}T00:00:00Z/{time_extent[1]}T23:59:59Z",
        "limit": 100,          # small pages: fewer timeouts on long ranges
    }

    session = _session_with_retries()
    url = STAC_SEARCH
    try:
        page, max_pages = 0, 50            # safety stop against paging loops
        while url and page < max_pages:
            try:
                resp = session.post(url, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                page += 1

                features = data.get("features", [])
                if not features and page == 1:
                    if verbose:
                        print(f"[overpass] no scenes found for {bbox_list} "
                              f"in {time_extent}")
                    break

                for feature in features:
                    title = (feature["properties"].get("title", "")
                             or feature.get("id", ""))
                    if "MSIL2A" not in title:          # keep L2A only
                        continue
                    match = re.search(r"_(\d{8}T\d{6})_", title)
                    if not match:
                        continue
                    dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
                    overpass.setdefault(dt.strftime("%Y-%m-%d"), dt)

                nxt = next((lk for lk in data.get("links", [])
                            if lk.get("rel") == "next"), None)
                if nxt:
                    url = nxt.get("href", STAC_SEARCH)
                    body = nxt.get("body", body)
                else:
                    url = None
            except requests.exceptions.RequestException as exc:
                if verbose:
                    print(f"[overpass] STAC request failed (page {page}): {exc}")
                if page == 1:
                    raise                    # nothing retrieved at all
                break                        # keep what we already have
    finally:
        session.close()

    if not overpass and verbose:
        print("[overpass] no acquisition times retrieved — check that the AOI "
              "is within Sentinel-2 coverage and that the period has scenes")
    return dict(sorted(overpass.items()))


def mean_overpass_hour(bbox, time_extent):
    """Mean UTC overpass hour as a decimal float (e.g. 11.07 = 11:04).

    Useful as a fallback when only an approximate acquisition time is needed
    (for instance to sanity-check a tide series) instead of the full
    per-date dictionary.
    """
    times = get_overpass_times(bbox, time_extent)
    if not times:
        raise ValueError("no scenes found for this AOI and period")
    hours = [dt.hour + dt.minute / 60 + dt.second / 3600
             for dt in times.values()]
    return sum(hours) / len(hours)
