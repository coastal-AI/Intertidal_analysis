# -*- coding: utf-8 -*-
"""Validation v2 — fetch the CDSE STAC items of the optional sensing-time diagnostic (spec §1.3, §1.10).

Question
--------
Which Sentinel-2 L2A products cover a site on each date, and what does the
catalogue say about each of them: the MGRS tile and its exact extent, the
platform, relative orbit and datatake, the datatake start, and the tile's
MEAN sun and view angles in float precision? These are the inputs of the
sun-angle inversion of :mod:`experiments.benchmark.sensing_time`.

OPTIONAL DIAGNOSTIC (user decision of 2026-10-14): the scene instants of
every run are the catalogue / product-name times (the datatake start). This
fetch only feeds ``validation_v2 diag time``, which measures how far the
site sensing time is from them; no other command requires it and nothing
reads its output unless ``--sensing-times <file>`` is given.

Pre-registered rules (docs/validation_v2_spec.md §1.3 and §1.10, with the
user decision of 2026-10-13: the solar angles come from the CDSE STAC
catalogue, not from the whole-degree openEO angle bands)
-----------------------------------------------------------------------
* Source: ``POST https://stac.dataspace.copernicus.eu/v1/search``,
  collection ``sentinel-2-l2a``, no authentication. One STAC item is one
  MGRS tile of one datatake. Its ``datetime`` (= ``start_datetime``) is the
  DATATAKE START, identical for every tile of the datatake; its
  ``view:sun_azimuth`` and ``view:sun_elevation`` are the tile's mean sun
  angles (granule metadata).
* Query: the site frame's lon/lat box, one calendar-year chunk at a time
  over ``[start, end)`` (end exclusive), pages of :data:`PAGE_LIMIT`
  items, following the ``next`` link (POST body, merged when the link says
  so, or GET). A chunk is complete only when its last page carries no
  ``next`` link; :data:`MAX_PAGES` pages that still end in a ``next`` link
  fail the chunk (``pyintertidal.overpass`` used to stop there silently).
* Politeness: one request at a time, :data:`PAUSE_S` seconds between
  requests, urllib3 retries with exponential backoff on 429 and 5xx, and up
  to :data:`CHUNK_RETRIES` whole-chunk attempts with a growing pause.
* Cache: ``results/validation_v2/sensing/<site>/stac_items.json``
  (:data:`ITEMS_FORMAT`): the query, per chunk ``{complete, n_items, pages,
  fetched_utc}``, and per item id a reduced record (:func:`item_record`).
  Resumable: a complete chunk is not fetched again unless ``refresh``; the
  file is rewritten atomically after every chunk. A cache made for another
  box or collection is refused.
* Item record: id, tile, platform, relative and absolute orbit, orbit
  state, datatake and datastrip ids, the datatake start, the four mean
  angles and the number of decimals they were delivered with, cloud cover,
  processing version, the footprint (valid-data) geometry and its box, and
  the tile's FULL extent (``proj:bbox`` and ``proj:code`` of the 10 m
  assets: 109.8 km squares) with its centre in UTM and in lon/lat. The
  footprint centre is NOT the tile centre at the swath edges.

Importing this module does nothing (no network, no chdir).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

#: the CDSE STAC search endpoint (as pyintertidal.overpass)
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION = "sentinel-2-l2a"
ITEMS_FORMAT = "validation-v2-stac-items/1"
ITEMS_NAME = "stac_items.json"
#: the G-time output root, relative to the repository root (spec §13.1)
SENSING_ROOT = "results/validation_v2/sensing"
PAGE_LIMIT = 100
MAX_PAGES = 100
#: seconds between two requests
PAUSE_S = 1.0
CHUNK_RETRIES = 3
#: seconds before a whole-chunk retry (times the attempt number)
CHUNK_BACKOFF_S = 30.0
REQUEST_TIMEOUT_S = 120
#: record name -> STAC property of the four mean angles
ANGLE_KEYS = {"sun_azimuth": "view:sun_azimuth", "sun_elevation": "view:sun_elevation",
              "view_azimuth": "view:azimuth", "view_incidence": "view:incidence_angle"}
#: assets whose proj:bbox is the tile's full extent (first one present wins)
PROJ_ASSETS = ("B02_10m", "B03_10m", "B04_10m", "B08_10m", "AOT_10m", "WVP_10m", "TCI_10m")

REPO_ROOT = Path(__file__).resolve().parents[3]


def log(msg):
    print(msg, flush=True)


class FetchError(RuntimeError):
    """A STAC chunk could not be fetched completely."""


# ─────────────────────────────────────────────────────────────────────────────
#  paths and the cache file
# ─────────────────────────────────────────────────────────────────────────────

def sensing_dir(site, out_root=None):
    """``<out_root>/<site>`` (default ``results/validation_v2/sensing``; a
    relative root is taken from the repository root)."""
    p = Path(out_root or SENSING_ROOT)
    p = p if p.is_absolute() else REPO_ROOT / p
    return p / str(site)


def items_path(site, out_root=None):
    return sensing_dir(site, out_root) / ITEMS_NAME


def _write_json(path, obj):
    """Atomic, LF, sorted keys, no NaN."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(obj, sort_keys=True, indent=1, allow_nan=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data + "\n")
    os.replace(tmp, path)


def load_items(path):
    """Read and check a cache file (:data:`ITEMS_FORMAT`)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != ITEMS_FORMAT:
        raise FetchError(f"{path}: not a STAC items cache ({doc.get('format')!r})")
    return doc


def new_cache(site, bbox):
    return {"format": ITEMS_FORMAT, "site": site, "endpoint": STAC_SEARCH,
            "collection": COLLECTION, "bbox": [round(float(v), 6) for v in bbox],
            "chunks": {}, "items": {}}


# ─────────────────────────────────────────────────────────────────────────────
#  item records
# ─────────────────────────────────────────────────────────────────────────────

def decimals(x):
    """Decimals a JSON number was delivered with (the shortest round-trip
    repr of the parsed float), or None. ``149.18`` -> 2."""
    if x is None:
        return None
    s = repr(float(x))
    if "e" in s or "E" in s:
        mant, exp = s.lower().split("e")
        d = len(mant.split(".")[1]) if "." in mant else 0
        return max(0, d - int(exp))
    return len(s.split(".")[1]) if "." in s else 0


def _epsg(code):
    if code is None:
        return None
    s = str(code).upper().replace("EPSG:", "")
    return int(s) if s.isdigit() else None


def tile_extent(feature):
    """``(epsg, [xmin, ymin, xmax, ymax])`` of the tile's full extent from
    the 10 m assets' ``proj:code`` / ``proj:bbox`` (or the item-level
    ``proj:epsg`` / ``proj:bbox``), or ``(None, None)``."""
    assets = feature.get("assets") or {}
    names = [n for n in PROJ_ASSETS if n in assets] + sorted(assets)
    for n in names:
        a = assets.get(n) or {}
        bb = a.get("proj:bbox")
        code = _epsg(a.get("proj:code") or a.get("proj:epsg"))
        if bb and len(bb) == 4 and code:
            return code, [float(v) for v in bb]
    p = feature.get("properties", {})
    bb = p.get("proj:bbox")
    code = _epsg(p.get("proj:code") or p.get("proj:epsg"))
    if bb and len(bb) == 4 and code:
        return code, [float(v) for v in bb]
    return None, None


def _to_lonlat(epsg, x, y):
    from pyproj import Transformer

    tf = Transformer.from_crs(f"EPSG:{int(epsg)}", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(float(x), float(y))
    return [round(float(lon), 8), round(float(lat), 8)]


def item_record(feature):
    """The reduced record of one STAC feature (see the module docstring)."""
    p = feature.get("properties", {})
    tile = str(p.get("grid:code") or p.get("s2:mgrs_tile") or p.get("mgrs:tile") or "")
    tile = tile.replace("MGRS-", "")
    epsg, pbb = tile_extent(feature)
    centre_xy = centre_ll = None
    if pbb is not None:
        centre_xy = [(pbb[0] + pbb[2]) / 2.0, (pbb[1] + pbb[3]) / 2.0]
        centre_ll = _to_lonlat(epsg, *centre_xy)
    rec = {"id": feature.get("id"), "tile": tile,
           "platform": p.get("platform"),
           "relative_orbit": p.get("sat:relative_orbit"),
           "absolute_orbit": p.get("sat:absolute_orbit"),
           "orbit_state": p.get("sat:orbit_state"),
           "datatake_id": p.get("eopf:datatake_id") or p.get("s2:datatake_id"),
           "datastrip_id": p.get("eopf:datastrip_id") or p.get("s2:datastrip_id"),
           "datetime": p.get("datetime") or p.get("start_datetime"),
           "start_datetime": p.get("start_datetime"), "end_datetime": p.get("end_datetime"),
           "cloud_cover": p.get("eo:cloud_cover"),
           "processing_version": p.get("processing:version") or p.get("s2:processing_baseline"),
           "geometry": feature.get("geometry"), "bbox": feature.get("bbox"),
           "proj_epsg": epsg, "proj_bbox": pbb, "tile_centre_xy": centre_xy,
           "tile_centre_lonlat": centre_ll, "angle_decimals": {}}
    for k, key in ANGLE_KEYS.items():
        v = p.get(key)
        rec[k] = None if v is None else float(v)
        rec["angle_decimals"][k] = decimals(v)
    return rec


def item_date(rec):
    """UTC date (``YYYY-MM-DD``) of a record's datatake start."""
    s = str(rec.get("datetime") or "")
    return s[:10] if len(s) >= 10 else None


def items_by_date(doc, start=None, end=None):
    """``{date: [records sorted by datatake, tile, id]}`` of a cache, over
    ``[start, end)`` when given."""
    out = {}
    for rec in doc.get("items", {}).values():
        d = item_date(rec)
        if d is None or (start and d < str(start)[:10]) or (end and d >= str(end)[:10]):
            continue
        out.setdefault(d, []).append(rec)
    for d in out:
        out[d].sort(key=lambda r: (str(r.get("datatake_id")), str(r.get("tile")), str(r.get("id"))))
    return dict(sorted(out.items()))


# ─────────────────────────────────────────────────────────────────────────────
#  the query
# ─────────────────────────────────────────────────────────────────────────────

def year_chunks(start, end):
    """``[(a, b), ...]`` calendar-year pieces of ``[start, end)``."""
    a = dt.date.fromisoformat(str(start)[:10])
    e = dt.date.fromisoformat(str(end)[:10])
    out = []
    while a < e:
        b = min(dt.date(a.year + 1, 1, 1), e)
        out.append((a.isoformat(), b.isoformat()))
        a = b
    return out


def query_body(bbox, a, b, limit=PAGE_LIMIT):
    """The search body of the chunk ``[a, b)``."""
    last = dt.date.fromisoformat(b) - dt.timedelta(days=1)
    return {"collections": [COLLECTION], "bbox": [float(v) for v in bbox],
            "datetime": f"{a}T00:00:00Z/{last.isoformat()}T23:59:59.999Z", "limit": int(limit)}


def make_session():
    """A requests session with urllib3 retries (429, 5xx; backoff 2 s
    doubling) validating TLS against the OS store
    (``pyintertidal.net.use_system_certificates``)."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    from pyintertidal.net import use_system_certificates

    use_system_certificates()
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["POST", "GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch_pages(session, body, url=STAC_SEARCH, max_pages=MAX_PAGES, pause_s=PAUSE_S,
                sleep=time.sleep):
    """Every feature of one search, following ``next`` links.

    Returns
    -------
    (features, pages)

    Raises
    ------
    FetchError
        When ``max_pages`` pages still end in a ``next`` link.
    """
    feats, pages = [], 0
    method, href, cur = "POST", url, dict(body)
    while True:
        if pages:
            sleep(pause_s)
        if method == "POST":
            resp = session.post(href, json=cur, timeout=REQUEST_TIMEOUT_S)
        else:
            resp = session.get(href, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        pages += 1
        feats.extend(data.get("features", []))
        nxt = next((lk for lk in data.get("links", []) if lk.get("rel") == "next"), None)
        if nxt is None:
            return feats, pages
        if pages >= max_pages:
            raise FetchError(f"{pages} pages and still a 'next' link: the chunk is incomplete")
        href = nxt.get("href") or url
        method = str(nxt.get("method") or ("POST" if "body" in nxt else "GET")).upper()
        if method == "POST":
            nb = nxt.get("body")
            cur = ({**cur, **nb} if nxt.get("merge") else dict(nb)) if nb is not None else cur


def _is_l2a(rec):
    return "MSIL2A" in str(rec.get("id") or "")


def fetch_items(site, bbox, start, end, out_root=None, session=None, refresh=False,
                pause_s=PAUSE_S, max_pages=MAX_PAGES, retries=CHUNK_RETRIES,
                backoff_s=CHUNK_BACKOFF_S, sleep=time.sleep, path=None):
    """Fetch (resumably) the STAC items of ``site`` over ``[start, end)``.

    Parameters
    ----------
    site : str
    bbox : (west, south, east, north)
        The frame's lon/lat box.
    start, end : str
        ISO dates, end exclusive; fetched in calendar-year chunks.
    out_root : path, optional
        Default :data:`SENSING_ROOT`.
    session : object, optional
        Anything with ``post(url, json=, timeout=)`` and ``get(url,
        timeout=)`` returning responses with ``raise_for_status()`` and
        ``json()`` (tests pass a recorded fake). Default
        :func:`make_session`.
    refresh : bool
        Fetch complete chunks again.
    path : path, optional
        The cache file (default :func:`items_path`).

    Returns
    -------
    dict
        The cache document (also written to ``path``).
    """
    path = Path(path) if path else items_path(site, out_root)
    bbox = [round(float(v), 6) for v in bbox]
    doc = load_items(path) if path.is_file() else new_cache(site, bbox)
    if doc.get("bbox") != bbox or doc.get("collection") != COLLECTION:
        if not refresh:
            raise FetchError(f"{path}: cached for box {doc.get('bbox')} / {doc.get('collection')}, "
                             f"asked {bbox} / {COLLECTION}; pass refresh=True to start again")
        doc = new_cache(site, bbox)
    own = session is None
    session = session or make_session()
    try:
        for a, b in year_chunks(start, end):
            key = f"{a}/{b}"
            ch = doc["chunks"].get(key)
            if ch and ch.get("complete") and not refresh:
                log(f"[stac] {site} {key}: cached ({ch.get('n_items')} items)")
                continue
            body = query_body(bbox, a, b)
            err = None
            for attempt in range(max(1, int(retries))):
                try:
                    feats, pages = fetch_pages(session, body, max_pages=max_pages,
                                               pause_s=pause_s, sleep=sleep)
                    err = None
                    break
                except FetchError:
                    raise
                except Exception as exc:      # network: retry the whole chunk
                    err = exc
                    log(f"[stac] {site} {key}: attempt {attempt + 1} failed: {exc}")
                    if attempt + 1 < retries:
                        sleep(backoff_s * (attempt + 1))
            if err is not None:
                doc["chunks"][key] = {"complete": False, "error": str(err)[:300]}
                _write_json(path, doc)
                raise FetchError(f"{site} {key}: the catalogue did not answer after "
                                 f"{retries} attempts ({err})") from err
            recs = [item_record(f) for f in feats]
            recs = [r for r in recs if _is_l2a(r) and item_date(r) and a <= item_date(r) < b]
            for r in recs:
                doc["items"][r["id"]] = r
            doc["chunks"][key] = {"complete": True, "n_items": len(recs), "pages": pages,
                                  "body": body,
                                  "fetched_utc": dt.datetime.now(dt.timezone.utc)
                                  .strftime("%Y-%m-%dT%H:%M:%SZ")}
            _write_json(path, doc)
            log(f"[stac] {site} {key}: {len(recs)} items in {pages} page(s)")
            sleep(pause_s)
    finally:
        if own:
            session.close()
    return doc


def chunks_complete(doc, start, end):
    """Whether every year chunk of ``[start, end)`` is complete in ``doc``."""
    return all((doc.get("chunks", {}).get(f"{a}/{b}") or {}).get("complete")
               for a, b in year_chunks(start, end))
