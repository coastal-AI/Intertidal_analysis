"""Render test: does the new plot_dem reach the quality of the DEA figures?"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyintertidal import viz
from pyintertidal.validation import reproject_to_grid

OUT = "figuras_pablo"


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


# ── Villaviciosa: antes / despues, mismo dato ────────────────────────────
mu, TR, CRS, SH = load("products_villaviciosa/hsr_elevation.tif")
lidar = reproject_to_grid("mdt5_villaviciosa_grande.tif", TR, CRS, SH)
print(f"villaviciosa: {np.isfinite(mu).sum():,} px, "
      f"[{np.nanmin(mu):.2f}, {np.nanmax(mu):.2f}] m")

fig, axes = plt.subplots(1, 2, figsize=(17, 9))
viz.plot_map(mu, TR, CRS, cmap="viridis", hillshade=True, robust=True,
             title="antes — robust=True, sin suavizado ni fondo",
             cbar_label="cota (m)", ax=axes[0])
viz.plot_dem(mu, TR, CRS, context=lidar, vert_exag=8,
             title="despues — rango completo, relieve suavizado, LiDAR de fondo",
             ax=axes[1])
fig.tight_layout()
fig.savefig(f"{OUT}/11_render_villaviciosa.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> 11_render_villaviciosa.png")

# ── Santoña: el sitio que luce ───────────────────────────────────────────
mus, TRs, CRSs, SHs = load("bathymetry_hsr_santona/hsr_mu.tif")
print(f"santona: {np.isfinite(mus).sum():,} px, "
      f"[{np.nanmin(mus):.2f}, {np.nanmax(mus):.2f}] m")
fig, ax = plt.subplots(figsize=(12, 11))
viz.plot_dem(mus, TRs, CRSs, vert_exag=8,
             title="Marismas de Santoña · elevación intermareal HSR", ax=ax)
fig.tight_layout()
fig.savefig(f"{OUT}/12_santona.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("  -> 12_santona.png")
