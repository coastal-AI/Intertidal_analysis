"""Where do SCL and NDWI disagree over the Santander flats, and how."""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import ndimage

import pyintertidal as pit
from pyintertidal import viz

OUT = "products_santander"


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs


wf_scl, TR, CRS = load(f"{OUT}/wf_scl.tif")
wf_ndwi = load(f"{OUT}/wf_ndwi.tif")[0]
marsh = load(f"{OUT}/marsh_scl.tif")[0]
d = wf_ndwi - wf_scl

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "figure.facecolor": "white"})

# ── Figure 1: the two maps and their difference ──────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
for a_, img, t in [(ax[0], wf_scl, "Frecuencia de agua · SCL (clase 6)"),
                   (ax[1], wf_ndwi, "Frecuencia de agua · NDWI > 0")]:
    im = a_.imshow(img, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    a_.set_title(t, fontsize=10)
    fig.colorbar(im, ax=a_, fraction=0.046, label="fracción de escenas mojado")
im = ax[2].imshow(d, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                  interpolation="nearest")
ax[2].set_title("NDWI − SCL\nrojo = SCL ve más agua · azul = NDWI ve más",
                fontsize=10)
fig.colorbar(im, ax=ax[2], fraction=0.046, label="diferencia")
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
fig.suptitle("Bahía de Santander · 247 escenas 2023-2025, mismos píxeles "
             "despejados", fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig("figuras_pablo/25_santander_scl_ndwi.png", bbox_inches="tight")
plt.close(fig)
print("-> 25_santander_scl_ndwi.png")

# ── Where does the disagreement concentrate? ─────────────────────────────
strong = np.isfinite(d) & (d < -0.2)          # SCL sees much more water
lbl, n = ndimage.label(strong, ndimage.generate_binary_structure(2, 2))
sizes = np.bincount(lbl.ravel()); sizes[0] = 0
order = np.argsort(sizes)[::-1][:5]
print(f"\n{n} parches donde SCL ve mucha mas agua; los mayores:")
px = abs(TR.a * TR.e) / 1e6
for k, i in enumerate(order):
    if sizes[i] < 50:
        continue
    ys, xs = np.where(lbl == i)
    cy, cx = int(ys.mean()), int(xs.mean())
    lon = TR.c + cx * TR.a
    lat = TR.f + cy * TR.e
    print(f"  {k+1}. {sizes[i]:6,d} px = {sizes[i]*px:5.2f} km2  "
          f"centro ({lat:.4f}, {lon:.4f})  "
          f"marsh_scl medio {np.nanmean(marsh[ys, xs]):.2f}")

# ── Figure 2: zoom on the biggest patch ──────────────────────────────────
i = order[0]
ys, xs = np.where(lbl == i)
pad = 70
sl = (slice(max(0, ys.min() - pad), ys.max() + pad),
      slice(max(0, xs.min() - pad), xs.max() + pad))
sub_tr = rasterio.Affine(TR.a, TR.b, TR.c + sl[1].start * TR.a,
                         TR.d, TR.e, TR.f + sl[0].start * TR.e)

fig, ax = plt.subplots(1, 4, figsize=(19, 5))
viz.plot_map(np.ones_like(wf_scl[sl]) * np.nan, sub_tr, CRS, basemap=True,
             ax=ax[0], cbar_label=None)
ax[0].set_title("imagen de satélite", fontsize=10)
for cb in fig.axes[len(ax):]:
    cb.remove()
for a_, img, t, cm, vm in [
        (ax[1], wf_scl[sl], "SCL", "Blues", (0, 1)),
        (ax[2], wf_ndwi[sl], "NDWI", "Blues", (0, 1)),
        (ax[3], d[sl], "NDWI − SCL", "RdBu_r", (-0.6, 0.6))]:
    im = a_.imshow(img, cmap=cm, vmin=vm[0], vmax=vm[1],
                   interpolation="nearest")
    a_.set_title(t, fontsize=10)
    fig.colorbar(im, ax=a_, fraction=0.046)
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
fig.suptitle(f"El mayor foco de discrepancia · {sizes[i]*px:.2f} km² donde "
             f"SCL clasifica agua y el NDWI no", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("figuras_pablo/26_santander_zoom.png", bbox_inches="tight")
plt.close(fig)
print("-> 26_santander_zoom.png")
