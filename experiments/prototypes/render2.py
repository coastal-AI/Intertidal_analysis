"""Render v2: despeckled + grey LiDAR context, both sites."""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyintertidal import sites, viz, terrain
from pyintertidal.validation import download_mdt_ign, reproject_to_grid

OUT = "figuras_pablo"
MIN_REGION = 25          # 25 px a 10 m = 0.25 ha


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


# ── Santona ──────────────────────────────────────────────────────────────
mu, TR, CRS, SH = load("bathymetry_hsr_santona/hsr_mu.tif")
before = int(np.isfinite(mu).sum())
mu_clean = terrain.drop_small_regions(mu, min_region_px=MIN_REGION)
after = int(np.isfinite(mu_clean).sum())
print(f"santona: {before:,} -> {after:,} px "
      f"({100*(before-after)/before:.1f} % era moteado suelto)")

aoi = sites.get("santona")
MDT = "mdt5_santona.tif"
try:
    download_mdt_ign({**aoi.bbox, "crs": "EPSG:4326"}, MDT)
    lidar = reproject_to_grid(MDT, TR, CRS, SH)
    print(f"  contexto LiDAR: {np.isfinite(lidar).mean()*100:.0f} % cobertura")
except Exception as e:
    lidar = None
    print("  sin contexto LiDAR:", type(e).__name__, str(e)[:60])

fig, ax = plt.subplots(figsize=(12, 11))
viz.plot_dem(mu_clean, TR, CRS, context=lidar, vert_exag=8,
             title="Marismas de Santoña · elevación intermareal (HSR, 10 m)",
             ax=ax)
fig.tight_layout()
fig.savefig(f"{OUT}/12_santona.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("  -> 12_santona.png")

# ── Villaviciosa antes/despues ───────────────────────────────────────────
mv, TRv, CRSv, SHv = load("products_villaviciosa/hsr_elevation.tif")
lidv = reproject_to_grid("mdt5_villaviciosa_grande.tif", TRv, CRSv, SHv)
mv_clean = terrain.drop_small_regions(mv, min_region_px=MIN_REGION)
print(f"villaviciosa: {int(np.isfinite(mv).sum()):,} -> "
      f"{int(np.isfinite(mv_clean).sum()):,} px")

fig, axes = plt.subplots(1, 2, figsize=(17, 9))
viz.plot_map(mv, TRv, CRSv, cmap="viridis", hillshade=True, robust=True,
             title="antes", cbar_label="cota (m)", ax=axes[0])
viz.plot_dem(mv_clean, TRv, CRSv, context=lidv, vert_exag=8,
             title="después — rango completo, relieve suavizado,\n"
                   "moteado eliminado, LiDAR de fondo", ax=axes[1])
fig.tight_layout()
fig.savefig(f"{OUT}/11_render_villaviciosa.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> 11_render_villaviciosa.png")
