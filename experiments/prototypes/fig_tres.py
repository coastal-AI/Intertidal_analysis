"""The three detectors over the whole bay, and where they part company."""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from pyintertidal import viz

OUT = "products_santander"


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs


scl, TR, CRS = load(f"{OUT}/wf_scl_grande.tif")
ndwi = load(f"{OUT}/wf_ndwi_grande.tif")[0]
mndwi = load(f"{OUT}/wf_mndwi_grande.tif")[0]
px = abs(TR.a * TR.e) / 1e6

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "figure.facecolor": "white"})

# ── the three maps, then the two differences ─────────────────────────────
fig, ax = plt.subplots(1, 5, figsize=(23, 5.6))
for a_, img, t in [(ax[0], scl, "SCL (clase 6)"),
                   (ax[1], ndwi, "NDWI > 0   ·   10 m nativos"),
                   (ax[2], mndwi, "MNDWI > 0   ·   B11 a 20 m")]:
    im = a_.imshow(img, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    a_.set_title(t, fontsize=10)
    fig.colorbar(im, ax=a_, fraction=0.046)
for a_, other, t in [(ax[3], scl, "NDWI − SCL"),
                     (ax[4], mndwi, "NDWI − MNDWI")]:
    im = a_.imshow(ndwi - other, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                   interpolation="nearest")
    a_.set_title(f"{t}\nrojo = el otro ve más agua", fontsize=10)
    fig.colorbar(im, ax=a_, fraction=0.046)
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
fig.suptitle("Bahía de Santander completa · 220 km² · 247 escenas 2023-2025, "
             "mismos píxeles despejados", fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig("figuras_pablo/29_santander_tres.png", bbox_inches="tight")
plt.close(fig)
print("-> 29_santander_tres.png")

# ── biggest focus where the OTHERS see water and NDWI does not ───────────
for name, other in [("scl", scl), ("mndwi", mndwi)]:
    d = ndwi - other
    strong = np.isfinite(d) & (d < -0.2)
    lbl, n = ndimage.label(strong, ndimage.generate_binary_structure(2, 2))
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    i = int(np.argmax(sizes))
    ys, xs = np.where(lbl == i)
    cy, cx = int(ys.mean()), int(xs.mean())
    print(f"{name}: {n} parches, mayor {sizes[i]:,} px = {sizes[i]*px:.2f} km2"
          f"  centro fila {cy} col {cx}")

# zoom on the SCL focus, with the satellite behind
d = ndwi - scl
strong = np.isfinite(d) & (d < -0.2)
lbl, _ = ndimage.label(strong, ndimage.generate_binary_structure(2, 2))
sizes = np.bincount(lbl.ravel()); sizes[0] = 0
ys, xs = np.where(lbl == int(np.argmax(sizes)))
pad = 90
sl = (slice(max(0, ys.min()-pad), ys.max()+pad),
      slice(max(0, xs.min()-pad), xs.max()+pad))
sub_tr = rasterio.Affine(TR.a, TR.b, TR.c + sl[1].start * TR.a,
                         TR.d, TR.e, TR.f + sl[0].start * TR.e)

fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
viz.plot_map(np.full_like(scl[sl], np.nan), sub_tr, CRS, basemap=True, ax=ax[0])
ax[0].set_title("imagen de satélite", fontsize=10)
for cb in fig.axes[len(ax):]:
    cb.remove()
for a_, img, t, cm, vm in [(ax[1], scl[sl], "SCL", "Blues", (0, 1)),
                           (ax[2], ndwi[sl], "NDWI", "Blues", (0, 1)),
                           (ax[3], mndwi[sl], "MNDWI", "Blues", (0, 1))]:
    im = a_.imshow(img, cmap=cm, vmin=vm[0], vmax=vm[1],
                   interpolation="nearest")
    a_.set_title(t, fontsize=10)
    fig.colorbar(im, ax=a_, fraction=0.046)
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
fig.suptitle("El mayor foco de discrepancia, con los tres detectores",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("figuras_pablo/30_santander_zoom3.png", bbox_inches="tight")
print("-> 30_santander_zoom3.png")
