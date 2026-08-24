"""Build campaign_north_coast.ipynb.

Written as a builder rather than hand-authored JSON: an .ipynb is a nested
document where one stray comma is an unopenable file, and the cell sources
here are long enough that hand-editing them in JSON would be asking for it.
"""
import io
import nbformat as nbf

OUT = (r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
       r"\campaign_north_coast.ipynb")

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# Mapping the intertidal zone of the north coast of Spain

**A campaign of 10 km production cells, run from the study notebook itself.**

This notebook does not contain a pipeline. It tiles the coast, then executes
`intertidal_topography_villaviciosa.ipynb` once per cell, replacing only the
cell tagged `parameters` at its top. Everything that runs over the north coast
is therefore a cell you can read in the study notebook — there is no second,
divergent implementation to keep in step.

Each execution is written back to `runs/<cell>/` as a completed notebook.
That is the provenance record: not a log line saying a cell succeeded, but the
actual outputs, figures and printed diagnostics of the run that produced it.

**What this costs.** The Cantabrian coast from Fisterra to the Bidasoa comes
to about 116 cells. Each needs its own openEO job and about 1.2 GB of
three-year cube, so the campaign transfers roughly 140 GB and occupies the
Copernicus queue 116 times. Cubes are deleted once their products are verified,
so peak disk stays near 2.4 GB rather than 140 GB.

**Why two at a time.** This machine has 7.4 GB of RAM and one HSR fit peaked
at 2.9 GB. The limit is memory, not the 12 cores."""))

cells.append(md("## 1 · Parameters"))

cells.append(code(r'''from pathlib import Path

ROOT       = Path.cwd()
TEMPLATE   = ROOT / "intertidal_topography_villaviciosa.ipynb"
RUNS_DIR   = ROOT / "runs"
MANIFEST   = RUNS_DIR / "manifest.json"

CELL_KM        = 10.0                          # production cell size
BUFFER_KM      = 4.0                           # half-width of the coastal strip
MIN_CELL_KM2   = 5.0                           # drop slivers at the strip edge
TIME_EXTENT    = ("2023-01-01", "2025-12-31")  # the epoch Villaviciosa is validated on
MAX_CONCURRENT = 2                             # RAM-bound, not CPU-bound
CELL_TIMEOUT_S = 4 * 3600                      # a cell that hangs is a failed cell

RUNS_DIR.mkdir(exist_ok=True)
print(f"template  {TEMPLATE.name}")
print(f"period    {TIME_EXTENT[0]} to {TIME_EXTENT[1]}")
print(f"runs      {RUNS_DIR}")'''))

cells.append(md(r"""## 2 · The coast, and the cells it becomes

The strip is a single traced shoreline buffered on both sides. Drawing an
inland edge and a seaward edge separately was the first attempt and it gave
158 cells, because the two edges drift apart and the strip silently becomes
two cells deep — most of the second row open sea.

Buffering makes the depth one number that can be argued about. 4 km each side
reaches up every Cantabrian ria; the vertices sit at the ria mouths so the
buffer follows them inland rather than cutting across.

`overpass.py` in this package is about Sentinel-2 overpass **times**, not the
OSM Overpass API, so there is no coastline to fetch — this trace is by hand and
should be checked on the map below before anything is launched."""))

cells.append(code(r'''import json
import numpy as np
from shapely.geometry import LineString, mapping
import pyintertidal as pit

# Shoreline west to east, Fisterra to the Bidasoa. Vertices at the ria mouths.
COASTLINE = [
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

# Buffer in a locally metric frame: a degree of longitude is not a degree of
# latitude, and buffering in raw degrees would make the strip lopsided.
lat0 = float(np.mean([p[1] for p in COASTLINE]))
dlat = BUFFER_KM / 111.32
dlon = BUFFER_KM / (111.32 * np.cos(np.radians(lat0)))
scaled = LineString([(x / dlon, y / dlat) for x, y in COASTLINE])
ring = scaled.buffer(1.0, cap_style=2, join_style=2)
strip = pit.AOI.from_polygon(
    [(x * dlon, y * dlat) for x, y in ring.exterior.coords], name="north_coast")

all_cells = strip.tile(cell_km=CELL_KM)
cells = [c for c in all_cells if c.area_km2 >= MIN_CELL_KM2]

print(f"strip {strip.area_km2:,.0f} km2")
print(f"  {len(all_cells)} cells, {len(cells)} kept at >= {MIN_CELL_KM2} km2")
print(f"  ~{len(cells) * 1.2:,.0f} GB to transfer, {len(cells)} openEO jobs")'''))

cells.append(code(r'''# Look at it before launching 116 downloads. A strip that misses an estuary,
# or wanders inland, costs days to discover any other way.
from pyintertidal import viz

fig, ax = viz.plot_aoi(strip, tiles=cells,
                       title=f"North coast · {len(cells)} production cells "
                             f"of {CELL_KM:.0f} km")'''))

cells.append(md(r"""## 3 · The runner

Parameter injection is exactly what papermill does, and papermill is not
installed here — but `nbclient` and `nbformat`, which papermill is built on,
are. Replacing one tagged cell and executing is about twenty lines.

Each cell runs in its own kernel process, so two running at once are genuinely
parallel and neither can corrupt the other's state."""))

cells.append(code(r'''import time, traceback
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def parameter_source(cell):
    """The parameters cell for one production cell, as Python source."""
    return "\n".join([
        "# injected by campaign_north_coast.ipynb",
        f"SITE         = None",
        f"POLYGON      = {[list(p) for p in cell.polygon.exterior.coords]!r}",
        f"RUN_NAME     = {cell.name!r}",
        f"TIME_EXTENT  = {TIME_EXTENT!r}",
        "TIDE_MODEL   = 'EOT20'",
        "EVICT_CUBE   = True",
        "STRICT       = False",
        "MDT_PATH     = None      # no LiDAR reference for a production cell",
        "FIELD_CSV    = None      # no field survey either",
        "RUN_NAME = RUN_NAME or 'cell'",
        "CACHE    = f'ndwi_cube_{RUN_NAME}.nc'",
        "OUT_DIR  = f'products_{RUN_NAME}'",
        "print(f'run {RUN_NAME!r} · {TIME_EXTENT[0]} → {TIME_EXTENT[1]}')",
    ])


def run_one(cell):
    """Execute the study notebook for one cell. Returns a manifest record."""
    out_dir = RUNS_DIR / cell.name
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    nb = nbformat.read(TEMPLATE.open(encoding="utf-8"), as_version=4)
    for c in nb.cells:
        if "parameters" in c.get("metadata", {}).get("tags", []):
            c.source = parameter_source(cell)
            break
    else:
        raise RuntimeError(f"{TEMPLATE.name} has no cell tagged 'parameters'")

    record = {"cell": cell.name, "km2": round(cell.area_km2, 2),
              "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        client = NotebookClient(nb, timeout=CELL_TIMEOUT_S,
                                kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}},
                                allow_errors=False)
        client.execute()
        record["status"] = "done"
    except CellExecutionError as e:
        # An empty cell — cliff, open sea, no transition zone — is a RESULT,
        # not a failure. Distinguishing the two is what keeps the campaign
        # honest about coverage.
        text = str(e)
        record["status"] = "empty" if "no intertidal" in text.lower() else "failed"
        record["error"] = text.strip().splitlines()[-1][:300]
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        # Write the executed notebook whatever happened: a failed run's
        # traceback in context is worth more than a log line.
        nbformat.write(nb, (out_dir / "run.ipynb").open("w", encoding="utf-8"))
        record["minutes"] = round((time.time() - started) / 60, 1)
    return record'''))

cells.append(md(r"""## 4 · The manifest

The campaign spans days and this machine sleeps. Every cell's outcome is
written to `runs/manifest.json` as it finishes, and a re-run skips anything
already `done` or `empty` — so an interruption costs the cell in flight and
nothing else."""))

cells.append(code(r'''def load_manifest():
    if MANIFEST.exists():
        return {r["cell"]: r for r in json.loads(MANIFEST.read_text())}
    return {}


def save_manifest(records):
    MANIFEST.write_text(json.dumps(sorted(records.values(),
                                          key=lambda r: r["cell"]), indent=1))


manifest = load_manifest()
pending = [c for c in cells
           if manifest.get(c.name, {}).get("status") not in ("done", "empty")]

print(f"{len(cells)} cells · {len(cells) - len(pending)} already finished · "
      f"{len(pending)} to run")
if manifest:
    import pandas as pd
    counts = pd.Series([r["status"] for r in manifest.values()]).value_counts()
    print(counts.to_string())'''))

cells.append(md(r"""## 5 · Run

Two at a time. Each `NotebookClient` starts its own kernel, so this is real
parallelism; the threads only wait on those processes.

Interrupt this cell at any point — the manifest is written after every cell,
and re-running resumes."""))

cells.append(code(r'''from concurrent.futures import ThreadPoolExecutor, as_completed

if not pending:
    print("nothing to do")
else:
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(run_one, c): c for c in pending}
        for fut in as_completed(futures):
            rec = fut.result()
            manifest[rec["cell"]] = rec
            save_manifest(manifest)
            done += 1
            print(f"[{done:3d}/{len(pending)}] {rec['cell']:24s} "
                  f"{rec['status']:7s} {rec['minutes']:6.1f} min"
                  + (f"  {rec.get('error','')}" if rec["status"] == "failed" else ""))'''))

cells.append(md(r"""## 6 · The regional map

`check_tiles` refuses to merge anything whose CRS or resolution disagrees,
rather than silently resampling. `seam_offsets` measures the datum step across
every overlapping pair — each cell predicted its tide at its own centroid, so
neighbouring cells sit on slightly different vertical frames, and a few
centimetres is normal while tens of centimetres means something is wrong.

Levelling is cosmetic: it changes absolute elevations, so it is reported
whenever it is used."""))

cells.append(code(r'''from pathlib import Path

tifs = sorted(str(p) for c in cells
              for p in [ROOT / f"products_{c.name}" / "hsr_elevation.tif"]
              if p.exists())
print(f"{len(tifs)} cells produced an elevation raster")

if tifs:
    info = pit.mosaic.check_tiles(tifs)
    seams = pit.mosaic.seam_offsets(tifs)
    if seams:
        import numpy as np
        off = np.abs([s["offset_m"] for s in seams])
        print(f"\nseams: {len(seams)} overlapping pairs, "
              f"median |offset| {np.median(off):.3f} m, max {off.max():.3f} m")'''))

cells.append(code(r'''if tifs and not info["problems"]:
    out = pit.mosaic.merge_tiles(tifs, "products_north_coast/elevation.tif",
                                 method="average", level_seams=False)
    print("regional mosaic written to", out)
else:
    print("not merging: see the problems reported above")'''))

cells.append(md(r"""## Notes

* **The study notebook is the pipeline.** Nothing here reimplements it. If a
  method changes there, the campaign changes with it — which is the point of
  running the same file rather than a copy.
* **Empty cells are results.** A cliff or an open-sea cell produces no
  intertidal zone, and the manifest records that as `empty` rather than
  `failed`. Reporting how many cells held nothing is part of describing the
  coverage honestly.
* **Cubes are deleted, products are not.** The eviction in the study notebook
  verifies every product opens and holds data before removing the cube. To
  re-analyse a cell later, re-run it: the cube comes back from openEO.
* **Orphaned jobs.** `SentinelCube.ensure` records the openEO job id before it
  blocks, so a killed campaign leaves a traceable job rather than one nobody
  can find."""))

nb["cells"] = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                          "name": "python3"}
nbf.validate(nb)
nbf.write(nb, io.open(OUT, "w", encoding="utf-8"))
print(f"wrote {OUT} ({len(cells)} cells)")
