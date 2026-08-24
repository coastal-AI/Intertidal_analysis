"""How many 10 km cells does the north coast actually come to?

Before writing an orchestrator, find out what it would be orchestrating. The
strip is drawn by hand: `overpass.py` in this repo is about Sentinel-2
overpass TIMES, not the OSM Overpass API, so there is no coastline to buffer.

The vertices below follow the Cantabrian coast from Fisterra to the Bidasoa,
placed to keep every estuary inside with room to spare — a tight boundary
would clip the intertidal fringe, which is exactly the part being mapped.
"""
import os, sys

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pyintertidal as pit

# (lon, lat) traced west to east along the coast, then back offshore.
# Inland edge first (south), returning along the seaward edge (north).
COAST = [
    (-9.30, 42.85), (-9.28, 43.05), (-9.20, 43.20), (-8.95, 43.28),
    (-8.60, 43.30), (-8.40, 43.33), (-8.20, 43.40), (-7.90, 43.62),
    (-7.60, 43.66), (-7.30, 43.50), (-7.05, 43.52), (-6.80, 43.53),
    (-6.50, 43.53), (-6.20, 43.53), (-5.90, 43.53), (-5.60, 43.52),
    (-5.30, 43.50), (-5.00, 43.42), (-4.70, 43.38), (-4.40, 43.36),
    (-4.10, 43.36), (-3.80, 43.38), (-3.50, 43.36), (-3.20, 43.36),
    (-2.90, 43.35), (-2.60, 43.32), (-2.30, 43.28), (-2.00, 43.30),
    (-1.75, 43.34),
    # back westward along the seaward edge
    (-1.75, 43.44), (-2.00, 43.42), (-2.30, 43.40), (-2.60, 43.44),
    (-2.90, 43.47), (-3.20, 43.48), (-3.50, 43.48), (-3.80, 43.50),
    (-4.10, 43.48), (-4.40, 43.48), (-4.70, 43.50), (-5.00, 43.54),
    (-5.30, 43.62), (-5.60, 43.64), (-5.90, 43.65), (-6.20, 43.65),
    (-6.50, 43.65), (-6.80, 43.65), (-7.05, 43.64), (-7.30, 43.62),
    (-7.60, 43.78), (-7.90, 43.74), (-8.20, 43.52), (-8.40, 43.45),
    (-8.60, 43.42), (-8.95, 43.40), (-9.20, 43.32), (-9.42, 43.17),
    (-9.44, 42.85),
]

strip = pit.AOI.from_polygon(COAST, name="north_coast")
print(f"strip: {strip.area_km2:,.0f} km2, bbox {strip.bbox}")

for km in (10, 15, 20):
    cells = strip.tile(cell_km=km)
    area = sum(c.area_km2 for c in cells)
    print(f"  {km:2d} km cells -> {len(cells):4d} cells, {area:,.0f} km2, "
          f"mean {area/max(len(cells),1):.1f} km2/cell")

cells = strip.tile(cell_km=10)
gb_each = 1.2      # measured: ~1.2 GB per 3-year cube on a 1000x1000 grid
print(f"\nat 10 km and three years: {len(cells)} openEO jobs, "
      f"{len(cells) * gb_each:,.0f} GB transferred in total")
print(f"peak disk with eviction and two concurrent cells: "
      f"{2 * gb_each:.1f} GB")

sizes = np.array([c.area_km2 for c in cells])
print(f"\ncell area: min {sizes.min():.1f}, median {np.median(sizes):.1f}, "
      f"max {sizes.max():.1f} km2")
print(f"cells smaller than 5 km2 (slivers at the polygon edge): "
      f"{int((sizes < 5).sum())}")

strip.to_geojson("north_coast_strip.geojson")
import json
json.dump({"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"name": c.name},
     "geometry": {"type": "Polygon",
                  "coordinates": [list(map(list, c.polygon.exterior.coords))]}}
    for c in cells]}, open("north_coast_cells.geojson", "w"))
print("\nwrote north_coast_strip.geojson and north_coast_cells.geojson "
      "— open them in QGIS before launching anything")
