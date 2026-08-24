"""Follow-up: is the error random or systematic, and does the QC catch it?"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy.stats import theilslopes
from pyintertidal.validation import reproject_to_grid

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
R = np.load(os.path.join(SC, "repet.npz"))
muA, muB, both, uncA, uncB = R["muA"], R["muB"], R["both"], R["uncA"], R["uncB"]
dif = np.abs(muA - muB)

print("=== 1. la cola: cuantos pixeles son inestables ===")
for umbral in (0.05, 0.10, 0.20, 0.50):
    print(f"  |A-B| > {umbral*100:3.0f} cm : "
          f"{100*(dif[both] > umbral).mean():5.1f} % de los pixeles")

print("\n=== 2. la incertidumbre del metodo, ¿avisa de esos pixeles? ===")
pred = np.sqrt(uncA ** 2 + uncB ** 2)
q = np.nanpercentile(pred[both], [20, 40, 60, 80, 100])
prev = 0
for i, top in enumerate(q):
    m = both & (pred > prev) & (pred <= top)
    if m.sum() > 20:
        print(f"  quintil {i+1} de incertidumbre predicha "
              f"(<= {top*100:4.1f} cm): dispersion real "
              f"{1.4826*np.median(np.abs(dif[m]-np.median(dif[m])))*100:5.1f} cm"
              f"   mediana |A-B| {np.median(dif[m])*100:5.1f} cm")
    prev = top

# el filtro de produccion descarta unc > 10 cm
qc = both & (np.maximum(uncA, uncB) < 0.10)
print(f"\n  con el filtro de produccion (incertidumbre < 10 cm): "
      f"{qc.sum():,}/{both.sum():,} px")
print(f"  dispersion bruta antes {np.std(muA[both]-muB[both])/2*100:.1f} cm"
      f"  ->  despues {np.std(muA[qc]-muB[qc])/2*100:.1f} cm")

print("\n=== 3. ¿hay compresion sistematica frente al LiDAR? ===")
def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape

hsr, TR, CRS, SHAPE = load("products_villaviciosa/hsr_elevation.tif")
lidar = reproject_to_grid("mdt5_villaviciosa_grande.tif", TR, CRS, SHAPE)
inter = load("products_villaviciosa/intertidal_mask.tif")[0] > 0
m = np.isfinite(hsr) & np.isfinite(lidar) & inter
x, y = lidar[m], hsr[m] - np.median(hsr[m] - lidar[m])
sl, ic, lo, hi = theilslopes(y, x, 0.95)
print(f"  pendiente Theil-Sen HSR vs LiDAR: {sl:.3f} "
      f"(IC 95 % {lo:.3f} - {hi:.3f})")
print(f"  una pendiente < 1 significa que HSR aplana el relieve;"
      f" 1.0 seria perfecto")
rmse = np.sqrt(((y - x) ** 2).mean())
corr = np.sqrt(max(rmse ** 2 - (1 / 12), 0))
print(f"\n  RMSE observado {rmse:.3f} m")
print(f"  quitando la cuantizacion del LiDAR (1 m -> 0.289 m): "
      f"{corr:.3f} m")
print(f"  precision propia de HSR medida por mitades:      0.048 m")
print(f"  -> lo que queda ({corr:.3f} m) no es ruido de HSR, es datum, "
      f"cambio real del estuario o sesgo sistematico")
