"""Does the Tagus survey actually cover the intertidal flats?"""
import os, sys, zipfile, io
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

from pyintertidal.net import use_system_certificates
use_system_certificates()

import numpy as np
import requests

URL = ("https://downloads.emodnet-bathymetry.eu/high_resolution/"
       "1528_IB_Tagus_2020_64.emo.zip")
OUT = "bathy_tagus"
os.makedirs(OUT, exist_ok=True)
zp = os.path.join(OUT, "tagus.zip")

if not os.path.exists(zp):
    r = requests.get(URL, timeout=600)
    r.raise_for_status()
    open(zp, "wb").write(r.content)
print(f"descargado {os.path.getsize(zp) / 1e6:.1f} MB")

with zipfile.ZipFile(zp) as z:
    names = z.namelist()
    print("contenido:", names[:10])
    z.extractall(OUT)

for root, _, files in os.walk(OUT):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {f:52s} {os.path.getsize(p) / 1e6:8.2f} MB")

# Try to open whatever raster came out
import rasterio
cands = [os.path.join(r, f) for r, _, fs in os.walk(OUT) for f in fs
         if f.lower().endswith((".tif", ".tiff", ".nc", ".asc", ".emo", ".xyz"))]
for c in cands:
    try:
        with rasterio.open(c) as s:
            a = s.read(1).astype(float)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            px_deg = abs(s.transform.a)
            px_m = px_deg * 111320 * np.cos(np.radians(38.7))
            v = a[np.isfinite(a)]
            print(f"\n{os.path.basename(c)}")
            print(f"  {a.shape} px · {s.dtypes[0]} · {s.crs}")
            print(f"  paso {px_deg:.6f}° = {px_m:.1f} m")
            print(f"  cobertura {100 * np.isfinite(a).mean():.0f} %")
            print(f"  profundidad {np.nanmin(v):+.2f} .. {np.nanmax(v):+.2f} m")
            print(f"  valores no enteros: "
                  f"{100 * np.mean(v != np.round(v)):.1f} %  (0 % = cuantizado)")
            # THE question: does it reach above chart datum?
            for lo, hi, label in [(-2, 0, "franja -2..0 m"),
                                  (0, 2, "franja 0..+2 m (INTERMAREAL)"),
                                  (2, 6, "franja +2..+6 m")]:
                n = int(np.sum((v > lo) & (v <= hi)))
                print(f"    {label:32s} {n:9,d} px "
                      f"({100 * n / v.size:5.1f} %)")
    except Exception as exc:
        print(f"  no abre {os.path.basename(c)}: {type(exc).__name__}")
