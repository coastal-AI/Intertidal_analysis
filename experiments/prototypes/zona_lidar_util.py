"""Find, inside the intertidal mask, where the LiDAR actually has relief.

Over a flooded flat the survey returns a flat water surface, so the DEM is
locally constant and comparing against it measures nothing. But the flight
covers the whole ria, and parts of it were dry: those keep a real gradient.
Locate them and the transect has something to be validated against.
"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy import ndimage
from pyproj import Transformer

from pyintertidal.validation import reproject_to_grid
from pyintertidal import transect_profile


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


mu, TR, CRS, SH = load("products_villaviciosa/hsr_elevation.tif")
inter = load("products_villaviciosa/intertidal_mask.tif")[0] > 0
lidar = reproject_to_grid("mdt5_villaviciosa_grande.tif", TR, CRS, SH)

both = inter & np.isfinite(lidar) & np.isfinite(mu)
print(f"pixeles con mascara + LiDAR + HSR: {both.sum():,}")
vals, counts = np.unique(np.round(lidar[both], 2), return_counts=True)
print(f"  valores distintos de LiDAR ahi: {vals.size}")
print(f"  el mas repetido: {vals[counts.argmax()]:+.2f} m en el "
      f"{100 * counts.max() / both.sum():.0f} % de los pixeles")

# Local relief of the LiDAR: std in a 7x7 window, computed only where valid.
z = np.where(np.isfinite(lidar), lidar, 0.0)
w = np.isfinite(lidar).astype(float)
k = 7
mean = ndimage.uniform_filter(z, k) / np.maximum(ndimage.uniform_filter(w, k), 1e-6)
msq = ndimage.uniform_filter(z * z, k) / np.maximum(ndimage.uniform_filter(w, k), 1e-6)
relief = np.sqrt(np.maximum(msq - mean ** 2, 0))
relief = np.where(np.isfinite(lidar), relief, np.nan)

r_in = relief[both]
print(f"\nrelieve local del LiDAR dentro de la mascara (std 7x7):")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q}: {np.nanpercentile(r_in, q):.3f} m")

# Candidate zone: mask pixels where the LiDAR is NOT flat.
THR = float(np.nanpercentile(r_in, 90))
useful = both & (relief > THR)
print(f"\nzona util (relieve > p90 = {THR:.2f} m): {useful.sum():,} px "
      f"({100 * useful.sum() / both.sum():.1f} % de la mascara)")

lbl, n = ndimage.label(useful, ndimage.generate_binary_structure(2, 2))
sizes = np.bincount(lbl.ravel()); sizes[0] = 0
print(f"  en {n} parches; el mayor tiene {sizes.max()} px")

# Search a transect inside the biggest patches, maximising LiDAR variation.
tf = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


def lonlat(r, c):
    return tf.transform(TR.c + c * TR.a, TR.f + r * TR.e)


big = np.isin(lbl, np.argsort(sizes)[-6:][sizes[np.argsort(sizes)[-6:]] > 40])
ys, xs = np.where(big)
print(f"  buscando traza en los mayores parches ({len(ys):,} px)...")

rng = np.random.default_rng(7)
best = None
for _ in range(20000):
    i, j = rng.integers(0, len(ys), 2)
    r0, c0, r1, c1 = ys[i], xs[i], ys[j], xs[j]
    L = np.hypot(r1 - r0, c1 - c0) * 10
    if not (150 < L < 700):
        continue
    npts = max(int(L // 10), 10)
    rr = np.linspace(r0, r1, npts).astype(int)
    cc = np.linspace(c0, c1, npts).astype(int)
    lz = lidar[rr, cc]
    hz = mu[rr, cc]
    ok = np.isfinite(lz) & np.isfinite(hz) & inter[rr, cc]
    if ok.mean() < 0.8:
        continue
    spread = np.nanmax(lz[ok]) - np.nanmin(lz[ok])       # LiDAR must VARY
    n_lvl = np.unique(np.round(lz[ok], 2)).size
    sc = spread * min(n_lvl, 12) * ok.mean()
    if best is None or sc > best[0]:
        best = (sc, r0, c0, r1, c1, spread, n_lvl, L, ok.mean())

if best is None:
    print("\nNo hay ninguna traza util: el LiDAR no varia dentro de la mascara.")
else:
    sc, r0, c0, r1, c1, spread, n_lvl, L, frac = best
    A, B = lonlat(r0, c0), lonlat(r1, c1)
    print(f"\nMEJOR TRAZA sobre LiDAR con relieve:")
    print(f"  TRANSECT_START = ({A[0]:.4f}, {A[1]:.4f})")
    print(f"  TRANSECT_END   = ({B[0]:.4f}, {B[1]:.4f})")
    print(f"  {L:.0f} m, {frac*100:.0f} % dentro de la mascara")
    print(f"  el LiDAR varia {spread:.2f} m y toma {n_lvl} valores distintos")
    d, lv = transect_profile(lidar, TR, CRS, A, B, n=200)
    _, hv = transect_profile(mu, TR, CRS, A, B, n=200)
    ok = np.isfinite(lv) & np.isfinite(hv)
    print(f"  HSR {np.isfinite(hv).sum()}/200 pts, LiDAR "
          f"{np.isfinite(lv).sum()}/200")
    if ok.sum() > 30:
        b = np.median(hv[ok] - lv[ok])
        print(f"  correlacion HSR-LiDAR en la traza: "
              f"{np.corrcoef(hv[ok], lv[ok])[0,1]:+.3f}")
        print(f"  RMSE (sesgo {b:+.2f} m retirado): "
              f"{np.sqrt(np.mean((hv[ok]-b-lv[ok])**2)):.3f} m")
    np.savez(os.path.join(os.path.dirname(__file__), "traza.npz"),
             A=A, B=B, relief=relief, useful=useful)
