"""Is EOT20 actually closer to the estuary than GOT4.10, and does it matter?

Two questions, in order. First, geometry: GOT4.10 is on a 0.5 degree grid and
has no ocean cell within 32 km of the Ria de Villaviciosa, so every elevation
we have published rests on a tide imported from open water. EOT20 is 1/8
degree. If it still has nothing nearby, it buys us nothing and we stop here.

Second, magnitude: if the two models disagree by only a few centimetres over
the scenes we actually use, the tide cannot be what separates HSR from DEA
and we should look elsewhere. Only a disagreement of the order of the
0.024 m RMSE gap between the methods would justify refitting everything.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

import pyintertidal as pit
from pyintertidal.tidemodels import PyTMDTideModel

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

aoi = pit.sites.get("villaviciosa")
lat_c, lon_c = aoi.centroid   # AOI.centroid is (lat, lon)
print(f"centro del AOI  lon {lon_c:.4f}  lat {lat_c:.4f}\n", flush=True)

# ── 1. where does each model have water? ─────────────────────────────────
import pyTMD

lons = np.linspace(lon_c - 0.45, lon_c + 0.45, 5)
lats = np.linspace(lat_c - 0.45, lat_c + 0.45, 5)
LO, LA = np.meshgrid(lons, lats)
probe = np.datetime64("2024-01-01T12:00:00")

for name, pytmd_name in [("GOT4.10", "GOT4.10_nc"), ("EOT20", "EOT20")]:
    try:
        h = pyTMD.compute.tide_elevations(
            LO.ravel(), LA.ravel(), np.array([probe] * LO.size, dtype='datetime64[s]'),
            model=pytmd_name, directory="tide_models", crs="4326",
            standard="datetime")
        h = np.ma.filled(np.asarray(h, float).ravel(), np.nan)
        ok = np.isfinite(h)
        if ok.any():
            d = np.hypot((LO.ravel()[ok] - lon_c) * 111 * np.cos(np.radians(lat_c)),
                         (LA.ravel()[ok] - lat_c) * 111)
            print(f"{name:8s} {ok.sum():2d}/{LO.size} puntos con agua, "
                  f"el mas cercano a {d.min():5.1f} km")
        else:
            print(f"{name:8s}  0/{LO.size} puntos con agua")
    except Exception as e:
        print(f"{name:8s} ERROR {type(e).__name__}: {e}")
print(flush=True)

# ── 2. do the two tides differ on the scenes we use? ─────────────────────
got = json.load(open(os.path.join(SC, "tides.json")))
dates = sorted(got)
print(f"comparando sobre las {len(dates)} fechas con marea GOT4.10", flush=True)

times = pd.to_datetime([f"{d} 11:00:00" for d in dates]).to_pydatetime()
eot_model = PyTMDTideModel("EOT20", directory="tide_models", verbose=False)
eot = eot_model.get_tide_heights_batch(lat_c, lon_c, list(times))
eot = np.asarray([np.nan if v is None else v for v in eot], float)
g = np.array([got[d] for d in dates], float)

m = np.isfinite(eot) & np.isfinite(g)
d = eot[m] - g[m]
print(f"\n  n={m.sum()}   EOT20-GOT4.10:")
print(f"    sesgo   {d.mean():+.3f} m")
print(f"    RMS     {np.sqrt((d ** 2).mean()):.3f} m")
print(f"    max abs {np.abs(d).max():.3f} m")
print(f"    correlacion {np.corrcoef(eot[m], g[m])[0, 1]:.4f}")
print(f"    rango GOT {g[m].min():+.2f}..{g[m].max():+.2f}   "
      f"EOT {eot[m].min():+.2f}..{eot[m].max():+.2f}")
print(f"    amplitud (std)  GOT {g[m].std():.3f}   EOT {eot[m].std():.3f}"
      f"   ratio {eot[m].std() / g[m].std():.3f}", flush=True)

out = {d_: float(v) for d_, v in zip(np.array(dates)[m], eot[m])}
json.dump(out, open(os.path.join(SC, "tides_eot20.json"), "w"))
print(f"\nguardadas {len(out)} mareas EOT20 -> tides_eot20.json")
