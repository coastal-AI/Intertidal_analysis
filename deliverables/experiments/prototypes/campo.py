"""Validate the Villaviciosa DEM against the RTK survey of 2026-08-14.

Two things have to be settled before any number means anything:

* the GNSS gives ELLIPSOIDAL height; ours is referenced to the tide model's
  datum. The difference is dominated by the geoid undulation (~53 m in
  Asturias) plus whatever the antenna height convention was, and it is a
  CONSTANT — so it is removed as a bias, exactly as with every other
  reference. What survives that removal is the real comparison.
* several GNSS points can fall inside one 10 m pixel. Averaging them first
  stops a densely walked patch from dominating the statistics.
"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import pyintertidal as pit

CSV = r"C:\Users\Jorge\Downloads\batimetriavilla.csv"
df = pd.read_csv(CSV)
print(f"{len(df)} puntos leidos")
print(f"  solucion: {df['Solution status'].value_counts().to_dict()}")
print(f"  RMS vertical: mediana {df['Elevation RMS'].median()*100:.1f} cm, "
      f"max {df['Elevation RMS'].max()*100:.1f} cm")
print(f"  altura de antena: {df['Antenna height'].unique()} m")
t = pd.to_datetime(df["Averaging start"].str.replace(" UTC+02:00", "",
                                                     regex=False))
print(f"  ventana: {t.min()} → {t.max()} (local)")

fix = df[df["Solution status"] == "FIX"].copy()
print(f"  {len(fix)} con FIX ({100*len(fix)/len(df):.0f} %)")

lon = fix["Longitude"].values
lat = fix["Latitude"].values
h_ell = fix["Ellipsoidal height"].values
print(f"\nhuella: lon {lon.min():.5f}..{lon.max():.5f}, "
      f"lat {lat.min():.5f}..{lat.max():.5f}")
print(f"  extension {(lon.max()-lon.min())*111320*np.cos(np.radians(43.5)):.0f}"
      f" x {(lat.max()-lat.min())*111320:.0f} m")
print(f"  altura elipsoidal {h_ell.min():.3f} .. {h_ell.max():.3f} m "
      f"(desnivel {h_ell.max()-h_ell.min():.2f} m)")


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


hsr, TR, CRS, SH = load("products_villaviciosa/hsr_elevation.tif")
step = load("bathymetry/bathymetry_step.tif")[0]
wf = load("products_villaviciosa/water_frequency.tif")[0]
inter = load("products_villaviciosa/intertidal_mask.tif")[0] > 0

tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = tf.transform(lon, lat)
col = ((x - TR.c) / TR.a).astype(int)
row = ((y - TR.f) / TR.e).astype(int)
inside = (row >= 0) & (row < SH[0]) & (col >= 0) & (col < SH[1])
print(f"\n{inside.sum()} puntos caen dentro del raster")

# One value per pixel: average the GNSS points sharing a cell.
key = row[inside] * SH[1] + col[inside]
uniq, inv = np.unique(key, return_inverse=True)
gnss = np.bincount(inv, weights=h_ell[inside]) / np.bincount(inv)
n_per = np.bincount(inv)
rr, cc = uniq // SH[1], uniq % SH[1]
print(f"  ocupan {len(uniq)} pixeles de 10 m "
      f"(mediana {np.median(n_per):.0f} puntos por pixel)")
print(f"  {inter[rr, cc].sum()} de esos pixeles estan en la mascara "
      f"intermareal")
print(f"  frecuencia de agua ahi: {np.nanmedian(wf[rr, cc]):.2f}")

print(f"\n{'fuente':16s} {'px':>5s} {'sesgo':>9s} {'RMSE':>8s} {'MAE':>8s} "
      f"{'r':>7s}")
print("-" * 58)
for nm, arr in [("HSR", hsr), ("escalon (DEA)", step)]:
    v = arr[rr, cc]
    m = np.isfinite(v)
    if m.sum() < 10:
        print(f"{nm:16s} {m.sum():5d}  (sin datos suficientes)")
        continue
    d = gnss[m] - v[m]                 # ellipsoidal - our datum
    bias = np.median(d)
    resid = d - bias
    print(f"{nm:16s} {m.sum():5d} {bias:9.3f} "
          f"{np.sqrt(np.mean(resid**2)):8.3f} {np.mean(np.abs(resid)):8.3f} "
          f"{np.corrcoef(gnss[m], v[m])[0,1]:7.3f}")

np.savez("products_villaviciosa/campo.npz", row=rr, col=cc, gnss=gnss,
         n_per=n_per, lon=lon[inside], lat=lat[inside])
print("\n(el sesgo incluye ondulacion del geoide ~53 m + convenio de antena;"
      " lo que importa es RMSE y r)")
