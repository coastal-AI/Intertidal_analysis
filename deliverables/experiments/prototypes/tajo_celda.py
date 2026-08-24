"""Pick the production cell with the most surveyed flat, and check the tide."""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import xarray as xr

ds = xr.open_dataset("bathy_tagus/1528_IB_Tagus_2020_64.nc")
z = ds.elevation.values.astype(float)
lon, lat = ds.lon.values, ds.lat.values
ok = np.isfinite(z)
LON, LAT = np.meshgrid(lon, lat)
plon, plat = LON[ok], LAT[ok]
cell_km2 = (22.6 ** 2) / 1e6
print(f"{ok.sum():,} celdas medidas = {ok.sum() * cell_km2:.1f} km²")

# Candidate 10 km cells on a regular grid over the footprint.
KM = 10.0
dlat = KM / 111.32
dlon = KM / (111.32 * np.cos(np.radians(38.75)))
w, e, s, n = plon.min(), plon.max(), plat.min(), plat.max()

best = []
for r in range(int(np.ceil((n - s) / dlat)) + 1):
    for c in range(int(np.ceil((e - w) / dlon)) + 1):
        cw, cs = w + c * dlon, s + r * dlat
        inside = ((plon >= cw) & (plon < cw + dlon)
                  & (plat >= cs) & (plat < cs + dlat))
        if inside.sum() > 500:
            best.append((inside.sum(), cw, cs))
best.sort(reverse=True)

print(f"\nceldas de {KM:.0f} km con batimetria (top 5):")
for k, (n_px, cw, cs) in enumerate(best[:5]):
    print(f"  {k+1}. bbox({cw:.4f}, {cs:.4f}, {cw+dlon:.4f}, {cs+dlat:.4f})"
          f"  {n_px:6,d} celdas = {n_px * cell_km2:5.1f} km²")

n_px, cw, cs = best[0]
BBOX = dict(west=round(cw, 4), south=round(cs, 4),
            east=round(cw + dlon, 4), north=round(cs + dlat, 4))
print(f"\nELEGIDA: {BBOX}")
print(f"  {n_px * cell_km2:.1f} km² de llanura medida dentro")

# ── Does the tide model have data here? ──────────────────────────────────
from pyintertidal.tidemodels import PyTMDTideModel

clat = (BBOX["south"] + BBOX["north"]) / 2
clon = (BBOX["west"] + BBOX["east"]) / 2
m = PyTMDTideModel(model_name="GOT4.10", directory="./tide_models",
                   verbose=False)
t = pd.date_range("2020-01-01", periods=400, freq="7h").to_pydatetime()
h = np.array(m.get_tide_heights_batch(clat, clon, list(t)))
print(f"\nmarea GOT4.10 en ({clat:.4f}, {clon:.4f}):")
print(f"  {np.isfinite(h).sum()}/{len(t)} predicciones validas")
print(f"  rango {np.nanmin(h):+.2f} .. {np.nanmax(h):+.2f} m "
      f"(amplitud {np.nanmax(h)-np.nanmin(h):.2f} m)")
print(f"  Villaviciosa, para comparar: 4.73 m de rango")

# How far is the nearest valid model cell?
step = 0.05
for probe in (0.2, 0.5, 1.0):
    la = np.arange(clat - probe, clat + probe + step, step)
    lo = np.arange(clon - probe, clon + probe + step, step)
    LO, LA = np.meshgrid(lo, la)
    t0 = np.array([np.datetime64("2020-06-01T11:00")] * LA.size,
                  dtype="datetime64[ns]")
    hh = np.asarray(m._elevations(LO.ravel(), LA.ravel(), t0))
    v = np.isfinite(hh)
    if v.any():
        d = np.hypot((LA.ravel()[v] - clat) * 111.32,
                     (LO.ravel()[v] - clon) * 111.32 * np.cos(np.radians(clat)))
        print(f"  celda de modelo mas cercana: {d.min():.1f} km "
              f"({v.sum()}/{v.size} validas en ±{probe}°)")
        print(f"  (en Villaviciosa estaba a 32.3 km)")
        break
