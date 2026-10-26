# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch stac-meta``: the scene metadata sidecar ``cubes/<site>/scene_meta.json`` (spec §1.10).

Question
--------
Which relative orbit (and platform, processing baseline and tile cloud
cover) did each scene of a site's record come from? L01v (MAREA's band lags
per relative orbit, the truth-free orbit check of spec §6.4) groups the
scenes by it; without the sidecar the instance falls back to the 5-day
phase of the date (``instance.scene_orbits``, 'phase5').

Rules
-----
* Items: the CDSE STAC items (collection ``sentinel-2-l2a``, no
  credentials) of the site's frame box, fetched by
  :func:`experiments.benchmark.fetch.sensing.fetch_items` into the SAME
  cache as the optional ``diag time`` (``results/validation_v2/sensing/
  <site>/stac_items.json``: resumable per calendar year, one request at a
  time with pauses and retries; a complete year is never fetched again
  unless ``refresh``). The box is ``sensing_time.site_geometry`` (the first
  Zarr store's frame).
* Period: the calendar years of the site's Zarr stores
  (``zarr_cubes.list_years``), or the years given.
* The sidecar: the reduced item records turned back into STAC-like features
  and summarised per UTC date of the datatake start by
  :func:`experiments.benchmark.inputs.scene_meta_from_items` (the function
  ``inputs`` reads the sidecar with: platforms, relative orbits, processing
  baselines, tiles, mean tile cloud cover, ``double_overpass``), written
  atomically with ``inputs.write_scene_meta``. An existing sidecar with the
  same per-date content is left untouched (its sha256, which enters the
  L01v inputs key, does not move).
* Guard (``guard(site)`` -> reason or None, given by the CLI): a site whose
  package record was built with PROVISIONAL per-scene offsets has the
  sidecar's sha256 among its record inputs (``package.record_inputs``), so a
  new sidecar would rebuild the whole package: such a site is refused.

Importing this module does nothing (no network, no file).
"""
from __future__ import annotations

from pathlib import Path


def log(msg):
    try:
        print(msg, flush=True)
    except (BrokenPipeError, ValueError, OSError):
        pass


def records_to_features(records):
    """STAC-like features (``id``, ``bbox``, ``properties``) of the reduced
    item records of ``fetch.sensing.item_record``, with the property names
    ``inputs._item_record`` reads."""
    feats = []
    for r in records:
        props = {"datetime": r.get("datetime"), "start_datetime": r.get("start_datetime"),
                 "platform": r.get("platform"),
                 "sat:relative_orbit": r.get("relative_orbit"),
                 "sat:absolute_orbit": r.get("absolute_orbit"),
                 "sat:orbit_state": r.get("orbit_state"),
                 "processing:version": r.get("processing_version"),
                 "eo:cloud_cover": r.get("cloud_cover"),
                 "view:sun_azimuth": r.get("sun_azimuth"),
                 "view:sun_elevation": r.get("sun_elevation"),
                 "view:azimuth": r.get("view_azimuth"),
                 "view:incidence_angle": r.get("view_incidence"),
                 "eopf:datatake_id": r.get("datatake_id"),
                 "s2:mgrs_tile": r.get("tile")}
        feats.append({"id": r.get("id"), "bbox": r.get("bbox"), "properties": props})
    return feats


def period(years):
    """``(start, end)`` ISO dates of the calendar years (end exclusive)."""
    ys = sorted(int(y) for y in years)
    return f"{ys[0]}-01-01", f"{ys[-1] + 1}-01-01"


def build_meta(site, doc, years, items_path=None):
    """The sidecar of ``site`` from a STAC items cache document over the
    calendar ``years``."""
    from experiments.benchmark import inputs
    from experiments.benchmark.fetch import sensing as fs
    start, end = period(years)
    recs = [r for rs in fs.items_by_date(doc, start, end).values() for r in rs]
    query = {"source": f"CDSE STAC {fs.COLLECTION} ({fs.STAC_SEARCH})", "bbox": doc.get("bbox"),
             "start": start, "end_exclusive": end, "years": [int(y) for y in sorted(years)],
             "items_cache": None if items_path is None else str(items_path).replace("\\", "/"),
             "builder": "experiments.benchmark.fetch.stac_meta"}
    return inputs.scene_meta_from_items(records_to_features(recs), site=site, query=query)


def run(sites, root=None, refresh=False, years=None, guard=None, session=None, out_root=None):
    """``fetch stac-meta`` for ``sites``; returns the exit code (1 when a site
    could not be written).

    Parameters
    ----------
    sites : sequence of str
    root : str, optional
        The Zarr cubes root (default ZARR_CUBES_ROOT or cubes/).
    refresh : bool
        Fetch complete years of the STAC cache again.
    years : sequence of int, optional
        Default the years of the site's stores.
    guard : callable, optional
        ``guard(site)`` -> a refusal reason or None (the CLI's record check).
    session, out_root :
        Passed to ``fetch.sensing.fetch_items`` (tests: a recorded session;
        the cache root).
    """
    from experiments import zarr_cubes as zc
    from experiments.benchmark import inputs, sensing_time
    from experiments.benchmark.fetch import sensing as fs
    failed = []
    for site in sites:
        ys = sorted(int(y) for y in (years or zc.list_years(site, root)))
        if not ys:
            log(f"[fetch stac-meta] {site}: no Zarr store in the cubes root: nothing to describe")
            failed.append(site)
            continue
        why = guard(site) if guard is not None else None
        if why:
            log(f"[fetch stac-meta] {site}: REFUSED: {why}")
            failed.append(site)
            continue
        start, end = period(ys)
        try:
            geom = sensing_time.site_geometry(site, root)
            ipath = fs.items_path(site, out_root)
            doc = fs.fetch_items(site, geom["bbox"], start, end, out_root=out_root,
                                 session=session, refresh=refresh)
        except Exception as e:                   # network, cache: reported per site
            log(f"[fetch stac-meta] {site}: the STAC items could not be fetched "
                f"({type(e).__name__}: {e})")
            failed.append(site)
            continue
        if not fs.chunks_complete(doc, start, end):
            log(f"[fetch stac-meta] {site}: the STAC cache is incomplete over {start}..{end}: "
                "rerun the command")
            failed.append(site)
            continue
        meta = build_meta(site, doc, ys, ipath)
        path = Path(inputs.scene_meta_path(site, root))
        old = None
        if path.is_file():
            try:
                old = inputs.read_scene_meta(path)
            except Exception:
                old = None
        if old is not None and old.get("dates") == meta["dates"]:
            log(f"[fetch stac-meta] {site}: {path} unchanged ({len(meta['dates'])} dates)")
            continue
        inputs.write_scene_meta(path, meta)
        per_year = {}
        for d, v in meta["dates"].items():
            per_year.setdefault(d[:4], set()).update(v.get("relative_orbits") or [])
        log(f"[fetch stac-meta] {site}: {path} written: {len(meta['dates'])} dates over {ys}; "
            "relative orbits " + ", ".join(f"{y}: {sorted(o)}" for y, o in sorted(per_year.items())))
    return 1 if failed else 0
