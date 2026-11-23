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

import json
import os
import re
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .net import use_system_certificates

#: Copernicus Data Space STAC search endpoint.
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
#: known errors of the overpass-time files, corrected when they are read
CORRECTIONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                           "overpass_corrections.json")


def utm_zone(lon):
    """The UTM zone (1-60) of a longitude in degrees."""
    return int((float(lon) + 180.0) // 6.0) % 60 + 1


def _tile_zone(name):
    """The UTM zone of a Sentinel-2 product from the MGRS tile in its name (``..._T30TWP_...``), or None."""
    match = re.search(r"_T(\d{2})[A-Z]{3}_", name)
    return int(match.group(1)) if match else None


def load_overpass(path, corrections=CORRECTIONS):
    """The overpass times of a downloader's file as ``{date: pandas.Timestamp}``, with the known errors listed for that file in
    ``corrections`` (configs/overpass_corrections.json) put right. Returns ``(times, changed)``, ``changed`` being
    ``{date: (wrong, right)}``; the file itself is never rewritten. An entry whose wrong instant is not the file's raises, so
    a correction cannot silently land on a file that changed."""
    import pandas as pd

    times = {d: pd.Timestamp(v) for d, v in json.load(open(path, encoding="utf-8")).items()}
    fixes = {}
    if corrections and os.path.exists(corrections):
        table = json.load(open(corrections, encoding="utf-8"))
        key = os.path.normpath(path).replace(os.sep, "/")
        fixes = next((v for k, v in table.items() if not k.startswith("_") and key.endswith(k)), {})
    changed = {}
    for d, fix in fixes.items():
        if d not in times:
            continue
        if times[d] != pd.Timestamp(fix["was"]):
            raise ValueError(f"{path} {d}: the correction expects {fix['was']}, the file holds {times[d]}")
        changed[d] = (times[d], pd.Timestamp(fix["time"]))
        times[d] = pd.Timestamp(fix["time"])
    return times, changed


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


def get_overpass_times(bbox, time_extent, verbose=True, retries=4):
    """Real UTC acquisition time of every Sentinel-2 L2A scene over an AOI.

    A catalogue outage used to come back as an EMPTY or PARTIAL dict —
    scenes silently lost their time and the estimators downstream fitted
    fewer scenes, or nothing (2026-09-15/16, three runs at once; Ferrol
    kept 262 of 461 dates). Now any failed page makes the whole search
    retry, ``retries`` times with a growing pause, and after the last
    failure it RAISES instead of returning a short answer.

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
    carries the paging token. A product whose MGRS tile lies more than one
    UTM zone from the box's is ignored: the catalogue once returned, for
    Urdaibai on 2019-02-16, a tile by the antimeridian (01MAV, 22:26 UTC)
    whose footprint wrongly crossed the box, and it came first.
    """
    overpass: dict[str, datetime] = {}

    try:
        bbox_list = [float(bbox["west"]), float(bbox["south"]),
                     float(bbox["east"]), float(bbox["north"])]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid bbox: {exc}") from exc
    zones = {utm_zone(bbox_list[0]), utm_zone(bbox_list[2])}
    near = {(z + d - 1) % 60 + 1 for z in zones for d in (-1, 0, 1)}   # the box's zones and their neighbours

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox_list,
        "datetime": f"{time_extent[0]}T00:00:00Z/{time_extent[1]}T23:59:59Z",
        "limit": 100,          # small pages: fewer timeouts on long ranges
    }

    import time as _time
    for attempt in range(max(1, int(retries))):
        overpass.clear()
        session = _session_with_retries()
        url = STAC_SEARCH
        first_page_failed = False
        try:
            page, max_pages = 0, 50        # safety stop against paging loops
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
                        zone = _tile_zone(title)
                        if zone is not None and zone not in near:   # a tile from elsewhere
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
                    # any lost page means lost scenes: the whole search is
                    # retried (a partial answer once passed as complete —
                    # Ferrol, 262 of 461 dates — and nothing downstream
                    # could tell)
                    first_page_failed = True
                    break
        finally:
            session.close()
        if not first_page_failed:
            break
        if attempt + 1 < retries:
            pause = 30 * (attempt + 1)
            if verbose:
                print(f"[overpass] retrying in {pause} s "
                      f"({attempt + 2}/{retries})")
            _time.sleep(pause)
    else:
        raise RuntimeError("[overpass] the Sentinel-2 catalogue did not "
                           "answer after "
                           f"{retries} attempts; no acquisition times")

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


# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel-1 passes
# ─────────────────────────────────────────────────────────────────────────────

S1_COLLECTION = "sentinel-1-grd"
_S1_IW = re.compile(r"(S1[A-D])_IW_\w+?_(\d{8}T\d{6})_")


def s1_passes(bbox, start, end, retries=4, max_pages=500):
    """Every Sentinel-1 IW pass over a lon/lat box from ``start`` to ``end`` (dates, inclusive).

    Metadata only, as :func:`get_overpass_times`: the instant comes from the
    product name (``S1A_IW_GRDH_1SDV_20170612T061620_...``, the slice's start,
    UTC), the direction and the track from the item's ``sat:orbit_state`` and
    ``sat:relative_orbit``. The slices of one pass (one satellite, relative
    orbit and day) count once, at the earliest slice's start. A lost page
    makes the whole search retry; after ``retries`` attempts it raises,
    never a short answer.

    Returns a DataFrame sorted by instant, columns ``instant`` (naive UTC),
    ``sat``, ``direction`` ('ascending' or 'descending') and ``rel_orbit``.
    """
    import time as _time

    import pandas as pd

    cols = ["instant", "sat", "direction", "rel_orbit"]
    first = {"collections": [S1_COLLECTION],
             "bbox": [float(bbox["west"]), float(bbox["south"]), float(bbox["east"]), float(bbox["north"])],
             "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100}
    for attempt in range(max(1, int(retries))):
        rows, url, body, lost = [], STAC_SEARCH, dict(first), None
        session = _session_with_retries()
        try:
            for _ in range(max_pages):
                resp = session.post(url, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                for f in data.get("features", []):
                    m = _S1_IW.match(f.get("id", ""))
                    if m:
                        p = f.get("properties", {})
                        rows.append({"instant": pd.Timestamp(m.group(2)), "sat": m.group(1),
                                     "direction": p.get("sat:orbit_state"),
                                     "rel_orbit": p.get("sat:relative_orbit")})
                nxt = next((lk for lk in data.get("links", []) if lk.get("rel") == "next"), None)
                if not nxt:
                    break
                url, body = nxt.get("href", STAC_SEARCH), nxt.get("body", body)
            else:
                raise RuntimeError(f"[s1_passes] still paging after {max_pages} pages")
        except requests.exceptions.RequestException as exc:
            lost = exc
        finally:
            session.close()
        if lost is None:
            break
        if attempt + 1 < retries:
            _time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"[s1_passes] the catalogue did not answer after {retries} attempts: {lost}")
    if not rows:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(rows, columns=cols).sort_values("instant", kind="stable")
    d["day"] = d["instant"].dt.normalize()
    return d.drop_duplicates(["sat", "rel_orbit", "day"]).drop(columns="day").reset_index(drop=True)
