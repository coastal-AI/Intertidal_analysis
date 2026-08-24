"""Render v3: satellite basemap + native resolution, DEA-style panels."""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyintertidal import viz, terrain

OUT = "figuras_pablo"


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs


mu, TR, CRS = load("bathymetry_hsr_santona/hsr_mu.tif")
mu = terrain.drop_small_regions(mu, min_region_px=25)
ys, xs = np.where(np.isfinite(mu))
h, w = ys.max() - ys.min(), xs.max() - xs.min()
print(f"santona: {int(np.isfinite(mu).sum()):,} px, extension util "
      f"{w} x {h} px = {w*10/1000:.1f} x {h*10/1000:.1f} km")

# Figura al tamano NATIVO del dato: sin ampliar, sin interpolar.
dpi = 100
fig = plt.figure(figsize=((w + 40) / dpi, (h + 40) / dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
viz.plot_map(mu, TR, CRS, cmap="viridis", hillshade=True, shade_smooth=1.0,
             robust=False, vert_exag=10, basemap=True, ax=ax)
ax.set_axis_off()
for cb in fig.axes[1:]:
    cb.remove()
fig.savefig(f"{OUT}/15_santona_basemap.png", dpi=dpi, bbox_inches="tight",
            pad_inches=0)
plt.close(fig)
print("  -> 15_santona_basemap.png")

# Detalle estilo DEA: recorte de ~4 km sobre imagen de satelite
cy, cx = int(ys.mean()), int(xs.mean())
R = 200
sub = mu[cy - R:cy + R, cx - R:cx + R]
sub_tr = rasterio.Affine(TR.a, TR.b, TR.c + (cx - R) * TR.a,
                         TR.d, TR.e, TR.f + (cy - R) * TR.e)
fig = plt.figure(figsize=(9, 9), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
viz.plot_map(sub, sub_tr, CRS, cmap="viridis", hillshade=True,
             shade_smooth=1.0, robust=False, vert_exag=10, basemap=True,
             ax=ax)
ax.set_axis_off()
for cb in fig.axes[1:]:
    cb.remove()
fig.savefig(f"{OUT}/16_santona_detalle.png", dpi=150, bbox_inches="tight",
            pad_inches=0)
plt.close(fig)
print(f"  -> 16_santona_detalle.png ({2*R} x {2*R} px = "
      f"{2*R*10/1000:.1f} km de lado)")
