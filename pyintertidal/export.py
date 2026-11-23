"""
export.py — Publishable outputs: COGs and STAC metadata
========================================================

Turning results into something other people can use. Two standards, both
achievable without extra dependencies:

**Cloud-Optimized GeoTIFF (COG)** — an ordinary GeoTIFF with internal tiling
and overviews, so a viewer can read a small window (or a zoomed-out preview)
without downloading the whole file. This is what makes a regional mosaic
usable in a web map.

**STAC** — a small JSON document describing what a raster IS: where, when,
which bands, produced how. Written next to the data, it turns a folder of
GeoTIFFs into a catalogue that tools can index and colleagues can cite.

Provenance is deliberately part of the metadata: the parameters that produced
a product travel with it, so a file found six months later can still be
traced back to its run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import rasterio


def to_cog(src_path, out_path=None, overview_levels=(2, 4, 8, 16),
           resampling="average", compress="deflate", blocksize=512,
           verbose=True):
    """Rewrite a GeoTIFF as a Cloud-Optimized GeoTIFF.

    Adds internal tiling and pyramid overviews. The file stays a perfectly
    normal GeoTIFF — every GIS reads it — but remote readers can now fetch
    just the window they need.
    """
    from rasterio.enums import Resampling
    from rasterio.shutil import copy as rio_copy

    out_path = out_path or src_path.replace(".tif", "_cog.tif")
    resampling_enum = getattr(Resampling, resampling)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", tiled=True, blockxsize=blocksize,
                       blockysize=blocksize, compress=compress)
        data = src.read()
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**profile) as tmp:
                tmp.write(data)
                tmp.build_overviews(list(overview_levels), resampling_enum)
                tmp.update_tags(ns="rio_overview", resampling=resampling)
                rio_copy(tmp, out_path, copy_src_overviews=True, **profile)
    if verbose:
        size = os.path.getsize(out_path) / 1e6
        print(f"[export] COG written: {out_path} ({size:.1f} MB, "
              f"overviews {overview_levels})")
    return out_path


def stac_item(raster_path, item_id=None, datetime_utc=None, aoi=None,
              properties=None, assets=None, collection="pyintertidal"):
    """Build a STAC Item (as a dict) describing a product raster.

    Parameters
    ----------
    properties : dict, optional
        Anything worth recording — thresholds, epoch, method, software
        version. This is where provenance lives.
    assets : dict, optional
        Extra files belonging to the same item (uncertainty layer, quicklook),
        as ``{key: path}``.
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        shape = src.shape
        transform = src.transform
        from rasterio.warp import transform_bounds
        west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)

    when = datetime_utc or datetime.now(timezone.utc)
    if isinstance(when, str):
        when_iso = when if when.endswith("Z") else when + "T00:00:00Z"
    else:
        when_iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id or os.path.splitext(os.path.basename(raster_path))[0],
        "collection": collection,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north],
                             [west, north], [west, south]]],
        },
        "bbox": [west, south, east, north],
        "properties": {
            "datetime": when_iso,
            "proj:epsg": crs.to_epsg() if crs else None,
            "proj:shape": [shape[0], shape[1]],
            "proj:transform": list(transform)[:6],
            "gsd": abs(transform.a),
            "processing:software": {"pyintertidal": "0.1.0"},
            **(properties or {}),
        },
        "assets": {
            "data": {"href": os.path.abspath(raster_path),
                     "type": "image/tiff; application=geotiff",
                     "roles": ["data"]},
            **{k: {"href": os.path.abspath(v),
                   "type": "image/tiff; application=geotiff",
                   "roles": ["metadata"]} for k, v in (assets or {}).items()},
        },
        "links": [],
    }
    if aoi is not None:
        item["properties"]["pyintertidal:site"] = aoi.name
        item["properties"]["pyintertidal:area_km2"] = round(aoi.area_km2, 2)
    return item


def write_stac(items, out_dir, collection="pyintertidal",
               description="Intertidal products", verbose=True):
    """Write STAC Items plus a Collection into ``out_dir``.

    Result: a folder that is both the data and its catalogue — enough for a
    colleague to load it, or for a data portal to ingest it.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for item in items:
        path = os.path.join(out_dir, f"{item['id']}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(item, fh, indent=2)
        written.append(path)

    boxes = np.array([it["bbox"] for it in items], dtype=float)
    times = sorted(it["properties"]["datetime"] for it in items)
    coll = {
        "type": "Collection", "stac_version": "1.0.0", "id": collection,
        "description": description, "license": "proprietary",
        "extent": {
            "spatial": {"bbox": [[float(boxes[:, 0].min()),
                                  float(boxes[:, 1].min()),
                                  float(boxes[:, 2].max()),
                                  float(boxes[:, 3].max())]]},
            "temporal": {"interval": [[times[0], times[-1]]]},
        },
        "links": [{"rel": "item", "href": f"./{it['id']}.json"} for it in items],
    }
    coll_path = os.path.join(out_dir, "collection.json")
    with open(coll_path, "w", encoding="utf-8") as fh:
        json.dump(coll, fh, indent=2)
    if verbose:
        print(f"[export] STAC collection with {len(items)} items → {coll_path}")
    return coll_path, written
