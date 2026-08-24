"""Profile along the walked path: GNSS truth against every satellite method.

The survey is a walk, not a straight line, so the natural x axis is distance
travelled in the order the points were taken. That keeps every measurement —
no interpolation onto an artificial line — and it is the view that shows
WHERE each method departs from the ground rather than by how much on average.
"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

CSV = r"C:\Users\Jorge\Downloads\batimetriavilla.csv"
df = pd.read_csv(CSV)
fix = df[df["Solution status"] == "FIX"].copy()
fix["t"] = pd.to_datetime(fix["Averaging start"].str.replace(" UTC+02:00", "",
                                                            regex=False))
fix = fix.sort_values("t").reset_index(drop=True)


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs


hsr, TR, CRS = load("figuras_pablo/hsr_elevation_v2.tif")
METHODS = {
    "HSR": hsr,
    "diccionario": load("bathymetry/bathymetry_pixels.tif")[0],
    "isolíneas": load("bathymetry/bathymetry_isolines.tif")[0],
    "escalón (DEA)": load("bathymetry/bathymetry_step.tif")[0],
}

tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = tf.transform(fix["Longitude"].values, fix["Latitude"].values)
col = ((x - TR.c) / TR.a).astype(int)
row = ((y - TR.f) / TR.e).astype(int)
gnss = fix["Ellipsoidal height"].values

# Distance walked, in the order the points were taken.
step = np.hypot(np.diff(x), np.diff(y))
dist = np.concatenate([[0], np.cumsum(step)])
print(f"{len(fix)} puntos, recorrido {dist[-1]:.0f} m en "
      f"{(fix['t'].iloc[-1]-fix['t'].iloc[0]).total_seconds()/60:.0f} min")
print(f"  paso mediano entre puntos: {np.median(step):.1f} m")

# Datum: one bias per method, from the pixels where it has data.
series = {}
for name, arr in METHODS.items():
    v = arr[row, col]
    m = np.isfinite(v)
    if m.sum() < 10:
        continue
    v = v + np.median(gnss[m] - v[m])
    series[name] = v
    d = gnss[m] - v[m]
    print(f"  {name:16s} {m.sum():3d}/{len(fix)} pts, "
          f"RMSE {np.sqrt(np.mean(d**2)):.3f} m")

COLOURS = {"HSR": "#c53030", "diccionario": "#2b6cb0",
           "isolíneas": "#7b341e", "escalón (DEA)": "#2f855a"}

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 1], height_ratios=[2.4, 1],
                      hspace=0.28, wspace=0.18)

ax = fig.add_subplot(gs[0, 0])
ax.plot(dist, gnss, color="black", lw=2.6, zorder=6, label="GNSS RTK (verdad)")
for name, v in series.items():
    ax.plot(dist, v, lw=1.5, alpha=0.9, color=COLOURS[name],
            label=f"{name}")
ax.set_ylabel("cota (m elipsoidal; sesgo de datum restado)")
ax.set_title("Perfil a lo largo del recorrido de campo · Villaviciosa, "
             "14 ago 2026", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(ncol=5, fontsize=9, loc="upper center")

# Residuals below, same x axis
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
for name, v in series.items():
    ax2.plot(dist, v - gnss, lw=1.2, color=COLOURS[name], alpha=0.9)
ax2.axhline(0, color="black", lw=1.2)
ax2.axhspan(-0.2, 0.2, color="0.6", alpha=0.18)
ax2.set_xlabel("distancia recorrida (m)")
ax2.set_ylabel("error (m)")
ax2.set_ylim(-1.2, 1.2)
ax2.grid(alpha=0.3)
ax2.text(0.995, 0.06, "banda gris = ±20 cm", transform=ax2.transAxes,
         ha="right", fontsize=8, color="0.35")

# The walk on the map
axm = fig.add_subplot(gs[:, 1])
pad = 45
sl = (slice(max(0, row.min()-pad), row.max()+pad),
      slice(max(0, col.min()-pad), col.max()+pad))
im = axm.imshow(hsr[sl], cmap="viridis", interpolation="nearest")
axm.plot(col-sl[1].start, row-sl[0].start, color="black", lw=1.0, alpha=0.85)
axm.scatter(col[0]-sl[1].start, row[0]-sl[0].start, s=60, color="#c53030",
            zorder=5, edgecolors="white", linewidths=1.3)
axm.annotate("inicio", (col[0]-sl[1].start, row[0]-sl[0].start),
             textcoords="offset points", xytext=(8, 8), fontsize=9,
             fontweight="bold", color="#c53030")
axm.set_title("el recorrido sobre el DEM", fontsize=10)
axm.set_xticks([]); axm.set_yticks([]); axm.grid(False)
fig.colorbar(im, ax=axm, fraction=0.046, label="HSR (m)")

fig.savefig("figuras_pablo/27_perfil_gnss.png", dpi=150, bbox_inches="tight")
print("-> 27_perfil_gnss.png")
