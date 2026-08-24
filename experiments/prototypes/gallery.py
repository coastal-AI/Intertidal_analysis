"""Render the repr gallery with the REAL chunk grids, and a PNG to eyeball."""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from pyintertidal.explain import draw_array_box, summarize_array
from pyintertidal.explain.htmlrepr import _CSS, storage_chunks

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

real = storage_chunks("ndwi_cube_villaviciosa_grande_10y.nc")
print("chunks reales del netCDF:", real)

cases = [
    ((1379, 915, 915), real, ("t", "y", "x"),
     "cubo NDWI 10 anos - chunks reales del netCDF"),
    ((305, 915, 915), (8, 256, 256), ("t", "y", "x"),
     "cubo filtrado 305 fechas"),
    ((26, 713, 1145), (4, 256, 256), ("t", "y", "x"),
     "cubo corto y ancho"),
    ((915, 915), (256, 256), ("y", "x"), "producto elevacion 10 m"),
    ((3660, 3660), (256, 256), ("y", "x"), "DEM fino x4 (2.5 m)"),
]

parts = [_CSS, "<body style='background:#1e1f22;padding:24px;"
               "font-family:system-ui'>"
               "<h2 style='color:#e6e6e6'>array boxes</h2>"]
svgs = []
for shape, chunks, names, title in cases:
    box = draw_array_box(shape, chunks=chunks, dim_names=names, size=190,
                         title=title)
    svgs.append(box.svg)
    parts.append(f"<div class='pit-repr' style='display:inline-block;"
                 f"margin:8px'>{box.svg}</div>")

with rasterio.open("products_villaviciosa/hsr_elevation.tif") as src:
    mu = src.read(1).astype("float32")
    mu[mu == src.nodata] = np.nan
    tr, crs = src.transform, src.crs
parts.append(summarize_array(mu, name="elevation (hsr, 2023-2025)",
                             transform=tr, crs=crs, units="m",
                             chunks=(256, 256),
                             long_name="Intertidal elevation").html)

out = os.path.join(SC, "repr_gallery.html")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("<!doctype html><meta charset='utf-8'>" + "".join(parts) + "</body>")
print("OK ->", out, os.path.getsize(out) // 1024, "KB")

# PNG contact sheet so the geometry can be checked visually
try:
    import cairosvg
    for i, svg in enumerate(svgs):
        cairosvg.svg2png(bytestring=svg.encode(), scale=2,
                         write_to=os.path.join(SC, f"box{i}.png"))
    print("PNG OK")
except ImportError:
    print("cairosvg no instalado")
