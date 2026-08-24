# -*- coding: utf-8 -*-
import json
OUT = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\intertidal_mapping_villaviciosa.ipynb"

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def co(t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": t.splitlines(keepends=True)}

C = []

C.append(md("""# Satellite-derived intertidal topography of the Ría de Villaviciosa

This notebook maps the intertidal zone of the Ría de Villaviciosa (Asturias,
N Spain) from a decade of Sentinel-2 imagery using the **`intertidal` SDK**:
extent, water-frequency structure and a tide-calibrated elevation model with
per-pixel uncertainty — the workflow we are scaling to the whole northern
Spanish coast as contiguous polygon tiles.

Everything below is ONE call per stage. Every threshold is auto-calibrated
from the data (the resolved values are reported and stored with the result),
so the same notebook works unchanged on any other tile.
"""))

C.append(md("## 1 · Setup\nConnect to Copernicus (only needed if the Sentinel-2 cube is not cached on disk yet)."))
C.append(co("""%config InlineBackend.figure_formats = ['png']
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

import truststore; truststore.inject_into_ssl()
import openeo

conn = openeo.connect("openeo.dataspace.copernicus.eu")
conn.authenticate_oidc()"""))

C.append(md("""## 2 · Area of interest — a polygon, not a bounding box

Production tiles are contiguous polygons; the bounding box is only used
internally for the satellite download. Every product and figure is clipped
to this polygon, so neighbouring tiles mosaic seamlessly."""))
C.append(co("""from intertidal import AOI

aoi = AOI.from_polygon([
    (-5.455, 43.478),   # SW · Villaviciosa town side
    (-5.425, 43.470),   # S  · upper estuary
    (-5.385, 43.482),   # SE · marsh margin
    (-5.375, 43.520),   # E  · MisIego flats
    (-5.385, 43.545),   # NE · Rodiles beach / mouth
    (-5.425, 43.548),   # N  · open coast
    (-5.455, 43.520),   # W  · left bank
], name="Villaviciosa")
aoi"""))

C.append(md("""## 3 · Run the pipeline

One call does everything: cube acquisition (cached — re-running this cell
never re-downloads), Otsu-calibrated NDWI water detection, reference map,
adaptive cloud filtering, water frequency, the intertidal window
(multi-Otsu), pyTMD tides at the overpass times, and HSR elevation fitted on
the most recent epoch with per-pixel uncertainty."""))
C.append(co("""from intertidal import run_pipeline

result = run_pipeline(
    conn,
    aoi,
    time_extent=("2016-01-01", "2025-12-31"),
    cache_nc="ndwi_cube_villaviciosa_grande_10y.nc",   # existing cube, reused
)"""))

C.append(md("The resolved (auto-calibrated) parameters travel with the result — any run is reproducible from `result.params`."))
C.append(co("""result.params"""))

C.append(md("## 4 · Coastal stability\nWhere the coast is stably wet, stably dry, or transitional (the intertidal candidate zone)."))
C.append(co("""result.plot_reference_map();"""))

C.append(md("## 5 · Water frequency\nFraction of clear observations in which each pixel is wet — the continuous structure of the intertidal zone."))
C.append(co("""result.plot_water_frequency();"""))

C.append(md("## 6 · Intertidal extent"))
C.append(co("""result.plot_intertidal();"""))

C.append(md("""## 7 · Intertidal elevation (HSR)

Our elevation estimator fits, per pixel, the NDWI-vs-tide sigmoid: its centre
is the pixel's elevation, its width the relief *inside* the pixel, and the
fit quality yields a Cramér–Rao uncertainty. Pixels whose uncertainty exceeds
10 cm are dropped automatically — the method flags its own bad fits."""))
C.append(co("""result.plot_elevation();"""))

C.append(md("Sub-pixel relief (a product unique to HSR) and the per-pixel uncertainty map:"))
C.append(co("""from intertidal import viz

viz.plot_map(result.elevation_sigma, result.transform, result.crs, result.aoi,
             cmap="inferno", title="Sub-pixel relief σ",
             cbar_label="Within-pixel relief (m)");
viz.plot_map(result.elevation_uncertainty, result.transform, result.crs, result.aoi,
             cmap="magma", title="Elevation uncertainty (Cramér–Rao)",
             cbar_label="σ of elevation (m)");"""))

C.append(md("""## 8 · Independent validation — IGN 5 m LiDAR

A separate step by design (production tiles do not depend on a national
reference service). The reported bias is dominated by the vertical-datum
offset between the tide-model MSL and the LiDAR orthometric datum; RMSE/MAE
are computed after removing it."""))
C.append(co("""from intertidal import validate_against_lidar

metrics = validate_against_lidar(result)
metrics"""))

C.append(md("""## 9 · Field transect — Misiego

The profile we will survey on the ground, with the satellite prediction to
compare against. Endpoints are the actual GPS coordinates of the walk."""))
C.append(co("""from intertidal.validation import transect_profile, reproject_to_grid
from intertidal.viz import plot_transect

start, end = (-5.3797, 43.5107), (-5.3835, 43.5065)     # (lon, lat)

dist, z_sat = transect_profile(result.elevation, result.transform, result.crs, start, end)
lidar = reproject_to_grid(f"mdt5_{result.aoi.name.lower()}.tif",
                          result.transform, result.crs, result.reference_map.shape)
_, z_lidar = transect_profile(lidar, result.transform, result.crs, start, end)

plot_transect(dist, {"HSR (satellite)": z_sat, "IGN LiDAR": z_lidar},
              title="Misiego transect — predicted profiles");"""))

C.append(md("## 10 · Save products\nEvery raster as GeoTIFF (10 m grid, polygon-clipped), ready for QGIS or the mosaic of tiles."))
C.append(co("""result.save("products_villaviciosa")"""))

C.append(md("""## Notes

* **Method comparisons.** The elevation and water-index methods this SDK
  superseded (SCL detection, bracketing/waterline elevation, multi-index WF)
  remain importable from `intertidal.legacy` — the paper's comparison studies
  run on exactly that code.
* **Epochs.** Elevation is fitted on the latest 3-year epoch because
  intertidal morphology migrates (fitting 10 years at once saturates the
  sigmoid width for 87 % of pixels vs 12 % per epoch). Pass `epoch="all"` to
  `run_pipeline` for a DEA-style per-epoch DEM time series.
* **Scaling out.** To map another tile, change the polygon in section 2 —
  nothing else. Tiles cache their own cube and can run in parallel.
"""))

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
import nbformat
nbf = nbformat.from_dict(nb)
nbformat.validate(nbf)
_, nbf = nbformat.validator.normalize(nbf)
nbformat.write(nbf, OUT)
print("written:", OUT, "|", len(C), "cells")
