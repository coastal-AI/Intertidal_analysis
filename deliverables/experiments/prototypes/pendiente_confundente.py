"""Is the upstream slope drop real, or an artefact of splitting the survey?

All four products lose about 0.3 of slope between the seaward and landward
halves of the field survey, which is what a tide wave amplifying up the
estuary would do — and it appears in HSR and DEA alike, so it would live in
the tide rather than in either method. Before that gets written down, three
cheaper explanations have to fail first:

  * regression dilution — a half spanning a narrower range of true elevation
    yields an attenuated slope even with identical data quality;
  * small samples — 55 points per half, so the slopes may simply be noisy;
  * the split itself — northing is a proxy for position along the ria, and
    the survey covers only 720 m of it.

The bootstrap resamples pixels within each half, so it answers the second
directly. The first is judged on the elevation ranges. The third is probed by
regressing residual against distance continuously, with no split at all.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
OUT = "products_villaviciosa"
EP = "2023-2025"
PRODUCTS = {"HSR / GOT4.10": f"{OUT}/hsr_got410_{EP}.tif",
            "DEA / GOT4.10": f"{OUT}/dea_got410_{EP}.tif",
            "HSR / EOT20": f"{OUT}/hsr_eot20_{EP}.tif",
            "DEA / EOT20": f"{OUT}/dea_eot20_{EP}.tif"}

aoi = pit.sites.get("villaviciosa")
with rasterio.open(PRODUCTS["HSR / EOT20"]) as s:
    transform, crs, SH = s.transform, s.crs, s.shape
arrs = {k: rasterio.open(p).read(1) for k, p in PRODUCTS.items()}

fx = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
fx = fx[fx["Solution status"] == "FIX"]
tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = tf.transform(fx["Longitude"].values, fx["Latitude"].values)
col = ((x - transform.c) / transform.a).astype(int)
row = ((y - transform.f) / transform.e).astype(int)
uniq, inv = np.unique(row * SH[1] + col, return_inverse=True)
cnt = np.bincount(inv)
gnss = np.bincount(inv, weights=fx["Ellipsoidal height"].values) / cnt
gx = np.bincount(inv, weights=x) / cnt
gy = np.bincount(inv, weights=y) / cnt
rr, cc = uniq // SH[1], uniq % SH[1]


def clean(a):
    return np.where(np.isfinite(a) & (a > -900), a, np.nan)


com = np.all([np.isfinite(clean(a)[rr, cc]) for a in arrs.values()], 0)
n = int(com.sum())
G = gnss[com]
dist = (gy[com].max() - gy[com]) / 1000.0
order = np.argsort(dist)
low, high = order[:n // 2], order[n // 2:]

print(f"{n} pixeles comunes\n")
print("CONFUNDENTE 1 — rango de cota en cada mitad")
for lab, m in [("exterior", low), ("interior", high)]:
    print(f"  {lab:9s} n={len(m):3d}  cota {G[m].min():+.2f}..{G[m].max():+.2f}"
          f"  rango {np.ptp(G[m]):.3f} m  std {G[m].std():.3f}")
ratio = G[high].std() / G[low].std()
print(f"  ratio de dispersion interior/exterior: {ratio:.3f}")
print("  (si es << 1, la pendiente interior se atenua sola)\n")

print("CONFUNDENTE 2 — la caida de pendiente sobrevive al remuestreo?")
rng = np.random.default_rng(7)
B = 20000
print(f"{'producto':16s} {'caida':>7s} {'IC 95%':>18s}  veredicto")
print("-" * 62)
for k, a in arrs.items():
    v = clean(a)[rr, cc][com]
    obs = (np.polyfit(G[high], v[high], 1)[0]
           - np.polyfit(G[low], v[low], 1)[0])
    il = rng.integers(0, len(low), size=(B, len(low)))
    ih = rng.integers(0, len(high), size=(B, len(high)))
    bl = np.array([np.polyfit(G[low][i], v[low][i], 1)[0] for i in il[:2000]])
    bh = np.array([np.polyfit(G[high][i], v[high][i], 1)[0] for i in ih[:2000]])
    d = bh - bl
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{k:16s} {obs:+7.3f} [{lo:+7.3f},{hi:+7.3f}]  "
          f"{'SIGNIFICATIVO' if lo * hi > 0 else 'no significativo'}")

print("\nCONFUNDENTE 3 — sin partir: residuo contra distancia rio arriba")
print(f"{'producto':16s} {'pend (m/km)':>12s} {'r':>7s}  {'p aprox':>8s}")
print("-" * 50)
for k, a in arrs.items():
    v = clean(a)[rr, cc][com]
    resid = G - v
    resid = resid - np.median(resid)
    sl, _ = np.polyfit(dist, resid, 1)
    r = np.corrcoef(dist, resid)[0, 1]
    t = abs(r) * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(t, n - 2))
    print(f"{k:16s} {sl:12.3f} {r:7.3f}  {p:8.4f}")
print("\nsi la marea real se amplifica rio arriba, el residuo (verdad menos")
print("modelo) debe crecer con la distancia: pendiente POSITIVA y clara")

json.dump({"n": n, "ratio_dispersion": float(ratio),
           "rango_exterior": float(np.ptp(G[low])),
           "rango_interior": float(np.ptp(G[high]))},
          open(os.path.join(SC, "confundente.json"), "w"), indent=1)
