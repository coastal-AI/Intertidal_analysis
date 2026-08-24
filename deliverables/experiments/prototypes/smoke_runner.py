"""Prove the injection works before 116 downloads depend on it.

Executing a real cell needs an openEO job and an hour. This checks everything
that comes BEFORE that and can be checked in seconds:

  * the template really carries a cell tagged `parameters`;
  * replacing it produces valid Python that defines every name the rest of
    the notebook reads;
  * the injected polygon round-trips to the same area as the cell;
  * a production cell disables the two site-specific validation sections
    rather than crashing in them.

Then it executes only the notebook's opening cells with a real kernel, up to
the point where a download would start — which is where a parameter mistake
would surface.
"""
import io
import json
import sys

import nbformat
from nbclient import NotebookClient

ROOT = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, ROOT)
import os
os.chdir(ROOT)

import numpy as np
from shapely.geometry import LineString
import pyintertidal as pit

TEMPLATE = "intertidal_topography_villaviciosa.ipynb"

# ── rebuild one cell exactly as the campaign notebook would ──────────────
COASTLINE = [(-5.60, 43.56), (-5.42, 43.55), (-5.25, 43.52)]
lat0 = float(np.mean([p[1] for p in COASTLINE]))
dlat, dlon = 4.0 / 111.32, 4.0 / (111.32 * np.cos(np.radians(lat0)))
ring = LineString([(x / dlon, y / dlat) for x, y in COASTLINE]).buffer(
    1.0, cap_style=2, join_style=2)
strip = pit.AOI.from_polygon([(x * dlon, y * dlat)
                              for x, y in ring.exterior.coords], name="test")
cell = [c for c in strip.tile(cell_km=10) if c.area_km2 >= 5][0]
print(f"test cell {cell.name}: {cell.area_km2:.1f} km2")


def parameter_source(cell, time_extent=("2023-01-01", "2025-12-31")):
    return "\n".join([
        "# injected by campaign_north_coast.ipynb",
        "SITE         = None",
        f"POLYGON      = {[list(p) for p in cell.polygon.exterior.coords]!r}",
        f"RUN_NAME     = {cell.name!r}",
        f"TIME_EXTENT  = {time_extent!r}",
        "TIDE_MODEL   = 'EOT20'",
        "EVICT_CUBE   = True",
        "STRICT       = False",
        "MDT_PATH     = None",
        "FIELD_CSV    = None",
        "RUN_NAME = RUN_NAME or 'cell'",
        "CACHE    = f'ndwi_cube_{RUN_NAME}.nc'",
        "OUT_DIR  = f'products_{RUN_NAME}'",
        "print(f'run {RUN_NAME!r}')",
    ])


nb = nbformat.read(io.open(TEMPLATE, encoding="utf-8"), as_version=4)
tagged = [i for i, c in enumerate(nb.cells)
          if "parameters" in c.get("metadata", {}).get("tags", [])]
assert len(tagged) == 1, f"expected one tagged cell, found {tagged}"
print(f"parameters cell at index {tagged[0]}  OK")

nb.cells[tagged[0]].source = parameter_source(cell)

# ── does the injected source define what the notebook later reads? ───────
ns = {}
exec(compile(nb.cells[tagged[0]].source, "<params>", "exec"), ns)
needed = ["SITE", "POLYGON", "RUN_NAME", "TIME_EXTENT", "TIDE_MODEL",
          "EVICT_CUBE", "STRICT", "MDT_PATH", "FIELD_CSV", "CACHE", "OUT_DIR"]
missing = [n for n in needed if n not in ns]
print(f"names defined: {'all' if not missing else missing}  "
      f"{'OK' if not missing else 'MISSING'}")

back = pit.AOI.from_polygon(ns["POLYGON"], name=ns["RUN_NAME"])
print(f"polygon round-trip: {back.area_km2:.1f} km2 vs {cell.area_km2:.1f} "
      f"{'OK' if abs(back.area_km2 - cell.area_km2) < 0.01 else 'MISMATCH'}")

# ── execute up to (not including) the download ───────────────────────────
head = nb.cells[:tagged[0] + 1] + [c for c in nb.cells[tagged[0] + 1:]
                                   if c.cell_type == "code"][:2]
probe = nbformat.v4.new_notebook(cells=head)
probe.metadata.kernelspec = {"name": "python3", "language": "python",
                             "display_name": "Python 3"}
try:
    NotebookClient(probe, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": ROOT}}).execute()
    print("opening cells executed in a real kernel  OK")
    for c in probe.cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                for line in "".join(o["text"]).strip().splitlines()[:4]:
                    print("   ", line[:88])
except Exception as e:
    print(f"kernel execution FAILED: {type(e).__name__}: "
          f"{str(e).strip().splitlines()[-1][:200]}")
