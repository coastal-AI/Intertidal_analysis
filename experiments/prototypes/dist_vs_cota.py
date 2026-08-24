"""Distance up the estuary, or elevation? The split confounds them.

Splitting the survey by northing also splits it by height: the landward half
spans 0.93 m and is missing the lowest 0.38 m entirely. So the slope drop
attributed to the tide wave amplifying upstream is equally consistent with
compression simply being worse at some elevations — and that has nothing to
do with the tide.

Three tests that separate them:

  1. split by ELEVATION instead of distance. If the slope drops here too,
     elevation is sufficient and distance explains nothing extra;
  2. redo the distance split INSIDE the overlapping elevation band, where
     both halves cover the same heights;
  3. fit slope ~ distance + elevation together, and see which coefficient
     survives the presence of the other.

Only if distance survives (2) and (3) does the amplification hypothesis mean
anything.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
OUT = "products_villaviciosa"
EP = "2023-2025"
PRODUCTS = {"HSR / GOT4.10": f"{OUT}/hsr_got410_{EP}.tif",
            "DEA / GOT4.10": f"{OUT}/dea_got410_{EP}.tif",
            "HSR / EOT20": f"{OUT}/hsr_eot20_{EP}.tif",
            "DEA / EOT20": f"{OUT}/dea_eot20_{EP}.tif"}

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
gy = np.bincount(inv, weights=y) / cnt
rr, cc = uniq // SH[1], uniq % SH[1]


def clean(a):
    return np.where(np.isfinite(a) & (a > -900), a, np.nan)


com = np.all([np.isfinite(clean(a)[rr, cc]) for a in arrs.values()], 0)
G = gnss[com]
D = (gy[com].max() - gy[com]) / 1000.0
vals = {k: clean(a)[rr, cc][com] for k, a in arrs.items()}
n = len(G)
print(f"{n} pixeles comunes")
print(f"correlacion distancia-cota: {np.corrcoef(D, G)[0,1]:+.3f}  "
      f"(por eso hay que separarlas)\n")


def halves(by):
    o = np.argsort(by)
    return o[:len(o) // 2], o[len(o) // 2:]


def slope(m, v):
    return float(np.polyfit(G[m], v[m], 1)[0])


print("TEST 1 — partiendo por COTA en vez de por distancia")
lo_e, hi_e = halves(G)
print(f"{'producto':16s} {'cota baja':>10s} {'cota alta':>10s} {'cambio':>8s}")
print("-" * 48)
for k, v in vals.items():
    a_, b_ = slope(lo_e, v), slope(hi_e, v)
    print(f"{k:16s} {a_:10.3f} {b_:10.3f} {b_ - a_:+8.3f}")

print("\nTEST 2 — distancia DENTRO de la banda de cota comun")
band = (G >= max(G[halves(D)[0]].min(), G[halves(D)[1]].min())) & \
       (G <= min(G[halves(D)[0]].max(), G[halves(D)[1]].max()))
Gb, Db = G[band], D[band]
ob = np.argsort(Db)
lo_b, hi_b = ob[:len(ob) // 2], ob[len(ob) // 2:]
print(f"  {int(band.sum())} puntos en la banda comun "
      f"{Gb.min():+.2f}..{Gb.max():+.2f} m")
print(f"  exterior n={len(lo_b)} cota std {Gb[lo_b].std():.3f}   "
      f"interior n={len(hi_b)} cota std {Gb[hi_b].std():.3f}")
rng = np.random.default_rng(11)
print(f"\n{'producto':16s} {'exterior':>9s} {'interior':>9s} {'cambio':>8s}"
      f" {'IC 95%':>18s}")
print("-" * 68)
for k, v in vals.items():
    vb = v[band]
    s_lo = float(np.polyfit(Gb[lo_b], vb[lo_b], 1)[0])
    s_hi = float(np.polyfit(Gb[hi_b], vb[hi_b], 1)[0])
    bl = np.array([np.polyfit(Gb[lo_b][i], vb[lo_b][i], 1)[0]
                   for i in rng.integers(0, len(lo_b), (2000, len(lo_b)))])
    bh = np.array([np.polyfit(Gb[hi_b][i], vb[hi_b][i], 1)[0]
                   for i in rng.integers(0, len(hi_b), (2000, len(hi_b)))])
    d = bh - bl
    lo95, hi95 = np.percentile(d, [2.5, 97.5])
    print(f"{k:16s} {s_lo:9.3f} {s_hi:9.3f} {s_hi - s_lo:+8.3f} "
          f"[{lo95:+7.3f},{hi95:+7.3f}]"
          f"{'  SIGNIF' if lo95 * hi95 > 0 else '  no signif'}")

print("\nTEST 3 — modelo conjunto  v = a + b*G + c*(G*D)")
print("  el termino de interaccion c dice si la pendiente cambia con la")
print("  distancia una vez tenida en cuenta la cota")
print(f"\n{'producto':16s} {'b':>8s} {'c':>9s} {'t(c)':>7s} {'p':>8s}")
print("-" * 52)
from scipy import stats
Gc = G - G.mean()
Dc = D - D.mean()
X = np.column_stack([np.ones(n), Gc, Dc, Gc * Dc])
for k, v in vals.items():
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    resid = v - X @ beta
    s2 = (resid ** 2).sum() / (n - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta[3] / se[3]
    p = 2 * (1 - stats.t.cdf(abs(t), n - X.shape[1]))
    print(f"{k:16s} {beta[1]:8.3f} {beta[3]:9.3f} {t:7.2f} {p:8.4f}")
print("\nc negativo y significativo = la pendiente cae rio arriba de verdad")
