"""A coast-hugging strip: trace the shoreline once, buffer it.

The first attempt drew an inland edge and a seaward edge by hand and came to
158 cells — because the two edges drifted 15 km apart in places and the strip
became two cells deep, most of the second row being open sea or inland farm.

Buffering a single traced line is both less work and less wasteful: the strip
is exactly as deep as the buffer everywhere, and the buffer is a number that
can be justified. 4 km each side covers every Cantabrian ria — Villaviciosa's
mapped intertidal reaches 7 km up its axis, and the ria mouths are what the
line follows.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
from shapely.geometry import LineString, mapping
import pyintertidal as pit

# The shoreline itself, west to east: Fisterra -> Bidasoa. Vertices sit at the
# ria mouths so the buffer reaches up each estuary.
LINE = [
    (-9.28, 42.92), (-9.18, 43.16), (-9.03, 43.24), (-8.87, 43.33),
    (-8.60, 43.32), (-8.42, 43.38), (-8.30, 43.42), (-8.20, 43.47),
    (-8.05, 43.55), (-7.85, 43.70), (-7.70, 43.73), (-7.55, 43.72),
    (-7.42, 43.70), (-7.25, 43.57), (-7.10, 43.55), (-6.95, 43.56),
    (-6.75, 43.57), (-6.55, 43.56), (-6.35, 43.56), (-6.15, 43.57),
    (-5.95, 43.56), (-5.75, 43.57), (-5.60, 43.56), (-5.42, 43.55),
    (-5.25, 43.52), (-5.08, 43.49), (-4.90, 43.47), (-4.72, 43.43),
    (-4.55, 43.41), (-4.38, 43.40), (-4.20, 43.40), (-4.02, 43.42),
    (-3.85, 43.44), (-3.70, 43.43), (-3.55, 43.44), (-3.42, 43.44),
    (-3.25, 43.43), (-3.10, 43.41), (-2.95, 43.42), (-2.80, 43.42),
    (-2.65, 43.42), (-2.50, 43.38), (-2.35, 43.35), (-2.20, 43.34),
    (-2.05, 43.36), (-1.90, 43.38), (-1.78, 43.38),
]

for buf_km in (3.0, 4.0, 5.0):
    lat0 = float(np.mean([p[1] for p in LINE]))
    # degrees are not metres, and longitude degrees shrink with latitude;
    # buffer in a local metric frame rather than in degrees.
    dlat = buf_km / 111.32
    dlon = buf_km / (111.32 * np.cos(np.radians(lat0)))
    scaled = LineString([(x / dlon, y / dlat) for x, y in LINE])
    poly = scaled.buffer(1.0, cap_style=2, join_style=2)
    poly = type(poly)([(x * dlon, y * dlat) for x, y in poly.exterior.coords])

    strip = pit.AOI.from_polygon(list(poly.exterior.coords), name="north_coast")
    cells = strip.tile(cell_km=10)
    keep = [c for c in cells if c.area_km2 >= 5.0]
    print(f"buffer {buf_km:.0f} km -> strip {strip.area_km2:,.0f} km2, "
          f"{len(cells)} cells ({len(keep)} of >= 5 km2), "
          f"{len(keep) * 1.2:,.0f} GB")

    if buf_km == 4.0:
        strip.to_geojson("north_coast_strip.geojson")
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": c.name,
                                               "km2": round(c.area_km2, 1)},
             "geometry": mapping(c.polygon)} for c in keep]},
            open("north_coast_cells.geojson", "w"))
        json.dump([{"name": c.name, "km2": round(c.area_km2, 2),
                    "polygon": [list(p) for p in c.polygon.exterior.coords]}
                   for c in keep], open("north_coast_cells.json", "w"), indent=1)
        print(f"  wrote the {len(keep)}-cell layout for QGIS review")
