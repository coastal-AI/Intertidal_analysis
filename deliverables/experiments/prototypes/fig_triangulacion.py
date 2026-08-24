"""Figures 09 and 10: what we CAN measure without a trustworthy reference."""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pyintertidal.validation import reproject_to_grid

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
OUT = "figuras_pablo"
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
AGUA, ACENTO, VERDE = "#2b6cb0", "#c53030", "#2f855a"


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


hsr, TR, CRS, SH = load("products_villaviciosa/hsr_elevation.tif")
wf = load("products_villaviciosa/water_frequency.tif")[0]
step = load("bathymetry/bathymetry_step.tif")[0]
dic = load("bathymetry/bathymetry_pixels.tif")[0]
inter = load("products_villaviciosa/intertidal_mask.tif")[0] > 0
lid = reproject_to_grid("mdt5_villaviciosa_grande.tif", TR, CRS, SH)
m = (inter & np.isfinite(wf) & np.isfinite(lid) & np.isfinite(hsr)
     & np.isfinite(step) & np.isfinite(dic))
print(f"n = {m.sum():,}")

capas = [("HSR epoca 2023-2025", hsr, ACENTO),
         ("escalon (DEA)", step, "#7b341e"),
         ("diccionario", dic, "#6b46c1"),
         ("LiDAR IGN 5 m", lid, VERDE)]

# ── 09: cada cota frente a la frecuencia de agua ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
for ax, (nm, arr, col) in zip(axes, capas):
    x, y = wf[m] * 100, arr[m]
    y = y - np.median(y)
    ax.hexbin(x, y, gridsize=48, cmap="Greys", mincnt=1, linewidths=0)
    bins = np.linspace(0, 100, 21)
    cen = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(y[(x >= a) & (x < b)]) if ((x >= a) & (x < b)).sum() > 20
           else np.nan for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(cen, med, color=col, lw=2.4)
    rho = spearmanr(x, y)[0]
    ax.set_title(f"{nm}\n" + r"$\rho$ = " + f"{rho:+.3f}", fontsize=9.5,
                 color=col if abs(rho) < 0.5 else "black",
                 fontweight="bold" if abs(rho) < 0.5 else "normal")
    ax.set_xlabel("frecuencia de agua (% de fechas mojado)")
    ax.set_ylim(-1.6, 1.6)
axes[0].set_ylabel("cota, centrada en su mediana (m)")
fig.suptitle("Un pixel que se moja mas a menudo TIENE que estar mas bajo.\n"
             "Los tres metodos de satelite lo cumplen; el LiDAR del IGN no "
             "— en el intermareal esta midiendo la lamina de agua, no el "
             "fondo", fontsize=10.5, y=1.13)
fig.savefig(os.path.join(OUT, "09_triangulacion.png"))
plt.close(fig)
print("  -> 09_triangulacion.png")

# ── 10: precision propia (mitades) y acuerdo entre metodos ───────────────
R = np.load(os.path.join(SC, "repet.npz"))
dH = (R["muA"] - R["muB"])[R["both"]]
dS = (R["zA"] - R["zB"])[R["bs"]]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

ax = axes[0]
for dd, nm, col, n in ((dH, "HSR", ACENTO, R["both"].sum()),
                       (dS, "escalon (DEA)", "#7b341e", R["bs"].sum())):
    ax.hist(dd / 2, bins=np.linspace(-0.6, 0.6, 61), histtype="step", lw=2,
            color=col, density=True, label=f"{nm} — {n:,} px")
ax.set_xlabel("diferencia entre las dos mitades / 2 (m)")
ax.set_ylabel("densidad")
ax.legend(fontsize=8)
ax.set_title("Repetibilidad: mismas fechas partidas en dos\n"
             "(no hace falta ninguna referencia)", fontsize=9.5)

ax = axes[1]
etq = ["HSR", "escalon"]
rob = [1.4826 * np.median(np.abs(d - np.median(d))) / 2 * 100
       for d in (dH, dS)]
bru = [np.std(d) / 2 * 100 for d in (dH, dS)]
w = 0.36
ax.bar(np.arange(2) - w / 2, rob, w, color=[ACENTO, "#7b341e"],
       label="pixel tipico (MAD)")
ax.bar(np.arange(2) + w / 2, bru, w, color=[ACENTO, "#7b341e"], alpha=0.42,
       label="todos los pixeles (std)")
for i, (a, b) in enumerate(zip(rob, bru)):
    ax.text(i - w / 2, a + 0.4, f"{a:.0f}", ha="center", fontsize=8.5)
    ax.text(i + w / 2, b + 0.4, f"{b:.0f}", ha="center", fontsize=8.5)
ax.set_xticks(range(2)); ax.set_xticklabels(etq)
ax.set_ylabel("precision (cm)")
ax.legend(fontsize=7.5)
ax.set_title("El pixel tipico repite en ~5 cm,\npero la cola es larga",
             fontsize=9.5)

ax = axes[2]
pares = [("HSR vs escalon", hsr, step), ("HSR vs diccionario", hsr, dic),
         ("HSR vs LiDAR", hsr, lid), ("escalon vs LiDAR", step, lid)]
rr = [abs(spearmanr(a[m], b[m])[0]) for _, a, b in pares]
cols = [AGUA, AGUA, VERDE, VERDE]
ax.barh(range(len(pares)), rr, color=cols, height=0.6)
for i, v in enumerate(rr):
    ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=8.5)
ax.set_yticks(range(len(pares)))
ax.set_yticklabels([p[0] for p in pares], fontsize=8.5)
ax.tick_params(axis="y", pad=6)
ax.invert_yaxis(); ax.set_xlim(0, 1.05)
ax.set_xlabel(r"|$\rho$| de Spearman")
ax.set_title("Los metodos de satelite se dan la razon\nentre ellos, no al "
             "LiDAR", fontsize=9.5)
fig.suptitle("Lo que si podemos medir hoy", fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "10_precision.png"))
plt.close(fig)
print("  -> 10_precision.png")
