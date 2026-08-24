"""One panel per method against the GNSS, on a shared distance axis.

Four separate panels rather than four lines on one: overlaid, the curves hide
each other exactly where they differ most. Shading the gap between each
method and the truth turns the error into an area you can see at a glance,
and stacking the panels keeps the same metre of path aligned across all four.
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

fix = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
fix = fix[fix["Solution status"] == "FIX"].copy()
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
METHODS = [
    ("HSR", hsr, "#c53030"),
    ("diccionario", load("bathymetry/bathymetry_pixels.tif")[0], "#2b6cb0"),
    ("isolíneas", load("bathymetry/bathymetry_isolines.tif")[0], "#7b341e"),
    ("escalón (DEA)", load("bathymetry/bathymetry_step.tif")[0], "#2f855a"),
]

tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = tf.transform(fix["Longitude"].values, fix["Latitude"].values)
col = ((x - TR.c) / TR.a).astype(int)
row = ((y - TR.f) / TR.e).astype(int)
gnss = fix["Ellipsoidal height"].values
dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, sharey=True)
plt.rcParams.update({"font.size": 9})

for ax, (name, arr, colour) in zip(axes, METHODS):
    v = arr[row, col].astype(float)
    m = np.isfinite(v)
    v = v + np.median(gnss[m] - v[m])
    d = v - gnss
    rmse = np.sqrt(np.mean(d[m] ** 2))
    mae = np.mean(np.abs(d[m]))
    r = np.corrcoef(gnss[m], v[m])[0, 1]
    within = 100 * np.mean(np.abs(d[m]) < 0.20)

    # Show the truth ONLY where this method has something to compare with:
    # drawing the GNSS over a gap suggests a comparison that does not exist.
    g_m = np.where(m, gnss, np.nan)
    v_m = np.where(m, v, np.nan)
    ax.plot(dist, g_m, color="black", lw=2.2, zorder=4,
            label="GNSS RTK (solo donde hay dato)")
    ax.plot(dist, v_m, color=colour, lw=1.5, zorder=3, label=name)
    # The gap itself, coloured by which way the method errs.
    ax.fill_between(dist, g_m, v_m, where=m & (v > gnss), color=colour,
                    alpha=0.25, interpolate=True, zorder=2,
                    label="método por encima")
    ax.fill_between(dist, g_m, v_m, where=m & (v < gnss), color="#4a5568",
                    alpha=0.22, interpolate=True, zorder=2,
                    label="método por debajo")
    ax.set_title(f"{name}   ·   {m.sum()}/{len(fix)} puntos   ·   "
                 f"RMSE {rmse:.3f} m   MAE {mae:.3f} m   r {r:.3f}   ·   "
                 f"{within:.0f} % dentro de ±20 cm",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("cota (m)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper right", framealpha=0.9)

axes[-1].set_xlabel("distancia recorrida (m)")
axes[0].set_ylim(51.8, 56.0)
fig.suptitle("Cada método contra la verdad de campo · Villaviciosa, "
             "14 ago 2026 · 361 puntos RTK sobre 2.804 m",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig("figuras_pablo/28_cuatro_perfiles.png", dpi=140,
            bbox_inches="tight")
print("-> 28_cuatro_perfiles.png")
