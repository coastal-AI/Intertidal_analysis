"""
aoi.py — Areas of interest as POLYGONS (first-class citizens)
=============================================================

Production runs tile the coast into contiguous cells, each one a polygon.
The polygon drives everything user-facing; its bounding box is only an
implementation detail of the satellite download:

* **download** uses ``aoi.bbox`` (satellite cubes are rectangular),
* **analysis** runs on the full rectangle (download time is flat with area,
  so this costs nothing extra),
* **products & figures** are clipped with ``aoi.raster_mask(...)`` so users
  only ever see their polygon, and neighbouring tiles butt together with no
  overlap artefacts.

Examples
--------
>>> aoi = AOI.from_bbox(-5.46, 43.47, -5.35, 43.55, name="Villaviciosa")
>>> aoi = AOI.from_polygon([(-5.42, 43.50), (-5.39, 43.47), ...], name="cell_042")
>>> aoi = AOI.from_dms(['43°30\\'25"N 5°25\\'32"W', ...], name="survey_area")
>>> inside = aoi.raster_mask(transform, crs, shape)   # boolean, True inside
>>> tiles = aoi.tile(cell_km=10)                      # contiguous production cells
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as shp_transform


def parse_dms(text):
    """Parse a DMS coordinate pair into decimal ``(lon, lat)``.

    Accepts the format field surveys and nautical charts use, e.g.
    ``'43°30\\'24.99"N 5°25\\'32.06"W'``. Returns Shapely order (lon, lat) —
    the order every geometry function in this package expects.
    """
    pattern = r'(\d+)°(\d+)\'([\d.]+)"([NSEW])'
    found = re.findall(pattern, str(text))
    if len(found) != 2:
        raise ValueError(f"expected two DMS coordinates, got {text!r}")
    values = {}
    for deg, minute, sec, hemi in found:
        dec = int(deg) + int(minute) / 60 + float(sec) / 3600
        if hemi in ("S", "W"):
            dec = -dec
        values["lat" if hemi in ("N", "S") else "lon"] = dec
    if "lat" not in values or "lon" not in values:
        raise ValueError(f"need one N/S and one E/W coordinate: {text!r}")
    return (values["lon"], values["lat"])


@dataclass(frozen=True)
class AOI:
    """A named study area defined by a WGS84 polygon."""

    polygon: Polygon                 # WGS84 (lon, lat)
    name: str = "site"

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def from_polygon(cls, coords, name="site"):
        """Build from a list of ``(lon, lat)`` vertices or a shapely Polygon."""
        poly = coords if isinstance(coords, Polygon) else Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)    # standard shapely fix for self-intersections
        return cls(polygon=poly, name=name)

    @classmethod
    def from_bbox(cls, west, south, east, north, name="site"):
        """Build from a decimal-degree bounding box (rectangular polygon)."""
        return cls(polygon=box(west, south, east, north), name=name)

    @classmethod
    def from_dms(cls, dms_vertices, name="site"):
        """Build from DMS vertex strings (see :func:`parse_dms`).

        Convenient when the study area was defined on a chart or handed over
        by a field team in degrees-minutes-seconds.
        """
        return cls.from_polygon([parse_dms(v) for v in dms_vertices], name=name)

    # ── derived geometry ────────────────────────────────────────────────────
    @property
    def bbox(self):
        """Bounding box dict (``west/south/east/north``) for the download."""
        w, s, e, n = self.polygon.bounds
        return {"west": w, "south": s, "east": e, "north": n}

    @property
    def centroid(self):
        """``(lat, lon)`` centroid — the default tide-prediction point."""
        c = self.polygon.centroid
        return (c.y, c.x)

    @property
    def area_km2(self):
        """Approximate polygon area in km² (spherical approximation)."""
        lat = np.radians(self.centroid[0])
        deg = 111.32                          # km per degree of latitude
        return float(self.polygon.area * deg * deg * np.cos(lat))

    # ── raster support ──────────────────────────────────────────────────────
    def raster_mask(self, transform, crs, shape):
        """Boolean mask of the polygon on an analysis grid (True = inside).

        The polygon is reprojected from WGS84 into the grid CRS first, so it
        works directly on the UTM grids the pipeline produces.
        """
        from rasterio.features import geometry_mask
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        poly = shp_transform(lambda x, y: tf.transform(x, y), self.polygon)
        return ~geometry_mask([mapping(poly)], out_shape=shape,
                              transform=transform, invert=False)

    # ── production tiling ───────────────────────────────────────────────────
    def tile(self, cell_km=10.0, overlap_km=0.0, name_prefix=None):
        """Split this AOI into contiguous square cells for parallel runs.

        This is how the coast is mapped at scale: each cell becomes its own
        :class:`AOI`, gets its own cached cube, and can be processed in a
        separate notebook or process. Cells that do not intersect the polygon
        are skipped, and each cell is INTERSECTED with the polygon so the
        union of the tiles is exactly the study area — no double counting
        when the products are later merged (:mod:`pyintertidal.mosaic`).

        Parameters
        ----------
        cell_km : float
            Approximate cell side in kilometres.
        overlap_km : float
            Optional halo added to each cell. Useful when an algorithm needs
            neighbourhood context (dilation buffers, smoothing) — set it to
            zero for strictly disjoint products.
        name_prefix : str, optional
            Base for cell names (default: this AOI's name).

        Returns
        -------
        list[AOI] — ordered row by row (north to south, west to east).
        """
        lat0 = self.centroid[0]
        dlat = cell_km / 111.32
        dlon = cell_km / (111.32 * np.cos(np.radians(lat0)))
        pad_lat = overlap_km / 111.32
        pad_lon = overlap_km / (111.32 * np.cos(np.radians(lat0)))

        w, s, e, n = self.polygon.bounds
        prefix = name_prefix or self.name
        cells, index = [], 0
        rows = int(np.ceil((n - s) / dlat))
        cols = int(np.ceil((e - w) / dlon))
        for r in range(rows):
            for c in range(cols):
                cw = w + c * dlon
                cs = n - (r + 1) * dlat
                cell = box(cw - pad_lon, cs - pad_lat,
                           cw + dlon + pad_lon, cs + dlat + pad_lat)
                clipped = cell.intersection(self.polygon)
                if clipped.is_empty or clipped.area <= 0:
                    continue
                cells.append(AOI(polygon=clipped,
                                 name=f"{prefix}_cell{index:03d}"))
                index += 1
        return cells

    def to_geojson(self, path=None):
        """GeoJSON feature of the polygon (written to ``path`` if given).

        Handy to check tiling in QGIS before launching a large campaign.
        """
        import json
        feature = {"type": "Feature", "properties": {"name": self.name,
                                                     "area_km2": self.area_km2},
                   "geometry": mapping(self.polygon)}
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"type": "FeatureCollection", "features": [feature]},
                          fh, indent=2)
        return feature

    def __repr__(self):
        b = self.bbox
        return (f"AOI({self.name!r}, {self.area_km2:.1f} km², "
                f"[{b['west']:.3f}, {b['south']:.3f}, "
                f"{b['east']:.3f}, {b['north']:.3f}])")


def as_aoi(value, name="site"):
    """Coerce user input into an :class:`AOI`.

    Accepts an AOI, a shapely Polygon, a vertex list, or a bbox dict — so
    every function in the package can simply call ``as_aoi(user_input)``.
    """
    if isinstance(value, AOI):
        return value
    if isinstance(value, Polygon):
        return AOI.from_polygon(value, name=name)
    if isinstance(value, dict):
        return AOI.from_bbox(value["west"], value["south"],
                             value["east"], value["north"], name=name)
    return AOI.from_polygon(value, name=name)
