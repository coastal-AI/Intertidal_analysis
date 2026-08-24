"""Step 4: the figures for the message to Pablo. Every number is measured."""
import os, sys, json
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.special import erf

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
OUT = "figuras_pablo"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
AGUA, TIERRA, ACENTO = "#2b6cb0", "#b7791f", "#c53030"
TERRAIN = "gist_earth"


def phi(h, mu, sg):
    return 0.5 * (1 + erf((h - mu) / (max(sg, 1e-3) * np.sqrt(2))))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  ->", p)


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a


# ═════════════════════════════════════════════════════════════════════════
print("1. la idea: pixel mixto")
fig = plt.figure(figsize=(10, 4.2))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 2.1], hspace=0.45,
                      wspace=0.35)

MU_D, SG_D = 0.0, 0.35
niveles = [-0.55, 0.0, 0.55]
for j, hz in enumerate(niveles):
    ax = fig.add_subplot(gs[0, j])
    f = phi(hz, MU_D, SG_D)
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#e8d9b5", ec="0.3", lw=1.2))
    ax.add_patch(Rectangle((0, 0), 1, f, fc=AGUA, ec="none", alpha=0.85))
    ax.plot([0, 1], [f, f], color=AGUA, lw=1.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"marea = {hz:+.2f} m\nfraccion mojada = {f*100:.0f} %",
                 fontsize=8.5)
    if j == 0:
        ax.set_ylabel("un pixel\nde 10 x 10 m", fontsize=8.5)

ax = fig.add_subplot(gs[1, :3])
ax.set_axis_off()
ax.text(0.0, 0.75, "El NDWI que mide el satelite es la MEZCLA de las dos "
        "partes:", fontsize=9)
ax.text(0.03, 0.42, r"$NDWI = (1-f)\cdot NDWI_{seco} + f\cdot NDWI_{mojado}$",
        fontsize=11)
ax.text(0.0, 0.08, "y f, la fraccion inundada, la marca la hipsometria "
        "interna del pixel.", fontsize=9)

ax = fig.add_subplot(gs[:, 3])
hh = np.linspace(-1.2, 1.2, 400)
a_d, b_d = -0.25, 0.85
curva = a_d + b_d * np.array([phi(x, MU_D, SG_D) for x in hh])
ax.plot(hh, curva, color=AGUA, lw=2.2)
for hz in niveles:
    ax.plot(hz, a_d + b_d * phi(hz, MU_D, SG_D), "o", color=ACENTO, ms=7,
            zorder=5)
ax.axvline(MU_D, color=ACENTO, ls="--", lw=1.2)
ax.annotate(r"$\mu$ = cota del pixel", (MU_D, a_d + b_d * 0.5),
            xytext=(MU_D + 0.18, a_d + b_d * 0.28), fontsize=8.5,
            color=ACENTO, arrowprops=dict(arrowstyle="->", color=ACENTO, lw=1))
ax.annotate("", (MU_D - SG_D, a_d + b_d * 0.86), (MU_D + SG_D, a_d + b_d * 0.86),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
ax.text(MU_D + 0.05, a_d + b_d * 0.90, r"$2\sigma$ = relieve",
        ha="center", fontsize=8, color="0.3")
ax.text(MU_D + 0.05, a_d + b_d * 0.80, "DENTRO del pixel",
        ha="center", fontsize=8, color="0.3")
ax.annotate("", (1.15, a_d), (1.15, a_d + b_d),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
ax.text(1.08, a_d + b_d / 2, "b (salto seco-mojado)", fontsize=8,
        color="0.3", va="center", ha="right", rotation=90)
ax.set_xlim(-1.25, 1.3)
ax.set_xlabel("altura de marea (m)")
ax.set_ylabel("NDWI del pixel")
ax.set_title(r"$NDWI(t) = a + b\,\Phi\!\left(\frac{h_t-\mu}{\sigma}\right)$",
             fontsize=11)
fig.suptitle("La idea: un pixel de la linea de agua esta PARCIALMENTE "
             "inundado (esquema)", fontsize=11, y=1.02)
save(fig, "01_idea_pixel_mixto.png")

# ═════════════════════════════════════════════════════════════════════════
print("2. curvas reales")
d = np.load(os.path.join(SC, "fit.npz"), allow_pickle=True)
Y = d["epoch_Y"]; C = d["epoch_C"]; t = d["epoch_tide"]
mu, sg, a, b = d["1_mu"], d["1_sigma"], d["1_a"], d["1_b"]
unc, valid, nobs = d["1_sigma_mu"], d["1_valid"], d["1_n_obs"]
H, W = int(d["H"]), int(d["W"])

idx = np.where(valid & (unc < 0.05) & (nobs > 60))[0]
orden = idx[np.argsort(mu[idx])]
picks = [orden[int(f * (len(orden) - 1))] for f in (0.05, 0.35, 0.65, 0.95)]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
hh = np.linspace(t.min() - 0.1, t.max() + 0.1, 300)
for ax, p in zip(axes, picks):
    m = C[:, p]
    ax.scatter(t[m], Y[m, p], s=9, alpha=0.45, color="0.35",
               edgecolors="none", label=f"{m.sum()} obs")
    ax.plot(hh, a[p] + b[p] * np.array([phi(x, mu[p], sg[p]) for x in hh]),
            color=AGUA, lw=2.1, label="sigmoide ajustada")
    ax.axvline(mu[p], color=ACENTO, ls="--", lw=1.3)
    ax.axvspan(mu[p] - sg[p], mu[p] + sg[p], color=ACENTO, alpha=0.10)
    ax.set_title(f"$\\mu$ = {mu[p]:+.2f} m   $\\sigma$ = {sg[p]:.2f} m\n"
                 f"incertidumbre = {unc[p]*100:.1f} cm", fontsize=9)
    ax.set_xlabel("altura de marea (m)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
axes[0].set_ylabel("NDWI")
fig.suptitle("Cuatro pixeles REALES de la ria de Villaviciosa, epoca "
             "2023-2025 (Sentinel-2, 10 m)", fontsize=11, y=1.06)
save(fig, "02_curvas_reales.png")

# ═════════════════════════════════════════════════════════════════════════
print("3. mapas de los metodos")
inter = load("products_villaviciosa/intertidal_mask.tif") > 0
mapas = [
    ("diccionario (pixels)", load("bathymetry/bathymetry_pixels.tif")),
    ("isolineas", load("bathymetry/bathymetry_isolines.tif")),
    ("escalon (DEA)", load("bathymetry/bathymetry_step.tif")),
    ("HSR 10 anos", load("bathymetry_hsr/hsr_mu.tif")),
    ("HSR epoca 2023-2025", load("products_villaviciosa/hsr_elevation.tif")),
]
ys, xs = np.where(inter)
sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
for ax, (lab, arr) in zip(axes, mapas):
    im = ax.imshow(arr[sl], cmap=TERRAIN, vmin=-2.0, vmax=1.2,
                   interpolation="nearest")
    ax.set_title(f"{lab}\n{np.isfinite(arr).sum():,} px", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("cota sobre el datum del modelo de marea (m)")
fig.suptitle("Los cinco metodos sobre la misma zona", fontsize=11, y=1.04)
save(fig, "03_mapas_metodos.png")

# ═════════════════════════════════════════════════════════════════════════
print("4. validacion contra LiDAR")
V = json.load(open(os.path.join(SC, "validation.json")))
orden = ["diccionario (pixels)", "isolineas", "optimizado (legacy)",
         "escalon (DEA)", "HSR 10 anos", "HSR epoca 2023-2025"]
def get(rows, k):
    return {r["method"]: r for r in rows}[k]

SUELO = 1.0 / np.sqrt(12)      # RMSE que introduce por si sola la
                               # cuantizacion a 1 m del LiDAR de referencia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, key, titulo in [
        (ax1, "all", "cada metodo sobre SUS pixeles\n(dentro de la mascara "
                     "intermareal)"),
        (ax2, "common", f"sobre los MISMOS pixeles\n({V['n_common']:,} px "
                        f"comunes a todos)")]:
    rows = [get(V[key], k) for k in orden]
    vals = [r["rmse_m"] for r in rows]
    cols = [ACENTO if "HSR epoca" in r["method"] else "0.62" for r in rows]
    ax.bar(range(len(rows)), vals, color=cols, width=0.62)
    for i, (r, v) in enumerate(zip(rows, vals)):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold" if "HSR epoca" in r["method"] else "normal")
    ax.axhline(SUELO, color="#2f855a", ls="--", lw=1.4)
    ax.text(len(rows) - 0.45, SUELO + 0.012,
            f"suelo impuesto por el LiDAR ({SUELO:.2f} m)",
            fontsize=7.5, color="#2f855a", ha="right")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{k.replace(' ', chr(10))}\nn={r['n']:,}"
                        for k, r in zip(orden, rows)], fontsize=7)
    ax.set_ylabel("RMSE contra LiDAR IGN 5 m (m)")
    ax.set_title(titulo, fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.25)
fig.suptitle("Validacion contra el LiDAR (sesgo de datum restado).\n"
             "ESTE RANKING NO SIRVE: el LiDAR viene cuantizado a 1 m y "
             "ademas no mide el fondo en el intermareal — ver figura 09",
             fontsize=10.5, y=1.10)
save(fig, "04_validacion.png")

# ═════════════════════════════════════════════════════════════════════════
print("5. scatter contra LiDAR")
L = np.load(os.path.join(SC, "lidar.npz"))
lidar, common = L["lidar"], L["common"]
hsr = load("products_villaviciosa/hsr_elevation.tif")
m = np.isfinite(hsr) & np.isfinite(lidar) & inter
x, y = lidar[m], hsr[m]
bias = np.median(y - x)
yb = y - bias

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# 1) el fichero de referencia, tal cual lo sirve el IGN: enteros
with rasterio.open("mdt5_villaviciosa_grande.tif") as s:
    crudo = s.read(1).astype(float)
crudo = crudo[(crudo > -4) & (crudo < 4)]
niv = np.unique(crudo)
axes[0].hist(crudo, bins=np.arange(-3.5, 4.5, 1.0), color="#2f855a",
             rwidth=0.7)
axes[0].set_xlabel("cota del LiDAR IGN, fichero original (m)")
axes[0].set_ylabel("celdas de 5 m")
axes[0].set_title(f"El WCS del IGN lo sirve en int16:\nsolo {niv.size} "
                  f"valores distintos entre -3 y +3 m", fontsize=9.5)
axes[0].set_xticks(np.arange(-3, 4))

# 2) HSR agrupado por cota del LiDAR (ya remuestreado a 10 m, que promedia
#    cuatro celdas y recupera algo de resolucion vertical)
paso = 0.25
cen = np.round(x / paso) * paso
usar = [v for v in np.unique(cen) if (cen == v).sum() >= 30]
datos = [yb[cen == v] for v in usar]
bp = axes[1].boxplot(datos, positions=usar, widths=paso * 0.7,
                     showfliers=False, patch_artist=True, manage_ticks=False)
for b in bp["boxes"]:
    b.set(facecolor=AGUA, alpha=0.55, lw=0.6)
for med in bp["medians"]:
    med.set(color=ACENTO, lw=1.5)
lo = min(min(usar), np.percentile(yb, 1)) - 0.3
hi = max(max(usar), np.percentile(yb, 99)) + 0.3
axes[1].plot([lo, hi], [lo, hi], color="0.35", ls="--", lw=1.3, label="1:1")
axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
axes[1].set_xticks(np.arange(np.floor(lo), np.ceil(hi) + 0.5, 0.5))
axes[1].set_xlabel("cota del LiDAR remuestreado a 10 m (m)")
axes[1].set_ylabel("HSR, sesgo de datum restado (m)")
axes[1].set_title(f"n = {m.sum():,} px   RMSE = "
                  f"{np.sqrt(((yb-x)**2).mean()):.3f} m\n"
                  "la mediana sigue la 1:1, pero la dispersion es del "
                  "orden del escalon", fontsize=9.5)
axes[1].legend(fontsize=8)

# 3) residuo
res = yb - x
axes[2].hist(res, bins=70, range=(-2, 2), color=AGUA, alpha=0.85)
axes[2].axvline(0, color=ACENTO, lw=1.4)
axes[2].axvspan(-0.5, 0.5, color="#2f855a", alpha=0.12)
axes[2].text(-1.95, axes[2].get_ylim()[1] * 0.88,
             "la banda verde es\nmedio escalon del LiDAR", ha="left",
             fontsize=7.5, color="#2f855a")
axes[2].set_xlabel("residuo HSR - LiDAR (m)")
axes[2].set_ylabel("pixeles")
axes[2].set_title(f"mediana {np.median(res):+.3f} m   "
                  f"{100*(np.abs(res) < 0.5).mean():.0f} % dentro de "
                  f"±0.5 m", fontsize=9.5)
fig.suptitle("HSR frente al LiDAR del IGN — y por que este LiDAR no vale "
             "para separar metodos", fontsize=11, y=1.03)
fig.tight_layout()
save(fig, "05_scatter_lidar.png")

# ═════════════════════════════════════════════════════════════════════════
print("6. productos que solo da este metodo")
sig = load("products_villaviciosa/hsr_sigma.tif")
# El raster de incertidumbre se escribe para TODO el cubo, tambien donde no
# hay ajuste valido; sin enmascarar, el mapa es ruido de tierra firme.
unc_map = np.where(np.isfinite(hsr),
                   load("products_villaviciosa/hsr_uncertainty.tif"), np.nan)
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
im = axes[0, 0].imshow(sig[sl], cmap="viridis", vmin=0, vmax=0.65)
axes[0, 0].set_title(r"$\sigma$ — relieve DENTRO del pixel (m)", fontsize=9.5)
fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
v = sig[np.isfinite(sig)]
axes[0, 1].hist(v, bins=40, color="#38a169")
axes[0, 1].axvline(np.median(v), color=ACENTO, lw=1.4,
                   label=f"mediana {np.median(v):.2f} m")
axes[0, 1].set_xlabel(r"$\sigma$ (m)"); axes[0, 1].legend(fontsize=8)
axes[0, 1].set_title(f"saturados en el borde de la rejilla: "
                     f"{100*(v >= 0.649).mean():.1f} %", fontsize=9.5)

u = unc_map[np.isfinite(unc_map)] * 100
tope = np.percentile(u, 98)
im = axes[1, 0].imshow(unc_map[sl] * 100, cmap="magma_r", vmin=0, vmax=tope)
axes[1, 0].set_title("incertidumbre Cramer-Rao (cm)", fontsize=9.5)
fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
axes[1, 1].hist(u[u < tope], bins=45, color="#805ad5")
axes[1, 1].axvline(np.median(u), color=ACENTO, lw=1.4,
                   label=f"mediana {np.median(u):.1f} cm")
axes[1, 1].set_xlabel("incertidumbre (cm)"); axes[1, 1].legend(fontsize=8)
axes[1, 1].set_title("cada pixel trae su propio error teorico", fontsize=9.5)
for ax in (axes[0, 0], axes[1, 0]):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Lo que el ajuste da ademas de la cota", fontsize=11, y=0.98)
fig.tight_layout()
save(fig, "06_sigma_incertidumbre.png")

# ═════════════════════════════════════════════════════════════════════════
print("7. epocas (test controlado sobre la misma ventana)")
sg10, v10, r10 = d["0_sigma"], d["0_valid"], d["0_rmse"]
sg3, v3, r3 = d["1_sigma"], d["1_valid"], d["1_rmse"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
REJILLA = np.array([0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65])
w = 0.38
c10 = [np.isclose(sg10[v10], g).sum() for g in REJILLA]
c3 = [np.isclose(sg3[v3], g).sum() for g in REJILLA]
pos = np.arange(len(REJILLA))
axes[0].bar(pos - w / 2, c10, w, color="0.62",
            label=f"10 anos ({len(sg10[v10]):,} px ajustados)")
axes[0].bar(pos + w / 2, c3, w, color=ACENTO,
            label=f"epoca 2023-2025 ({len(sg3[v3]):,} px)")
axes[0].set_xticks(pos)
axes[0].set_xticklabels([f"{g:g}" for g in REJILLA], fontsize=8)
axes[0].axvspan(len(REJILLA) - 1.5, len(REJILLA) - 0.5, color="0.85",
                zorder=0)
axes[0].text(len(REJILLA) - 1, max(c10 + c3) * 0.92, "borde de\nla rejilla",
             ha="center", fontsize=7.5, color="0.35")
axes[0].set_xlabel(r"$\sigma$ ajustada — valores de la rejilla (m)")
axes[0].set_ylabel("pixeles")
axes[0].legend(fontsize=7)
axes[0].set_title(r"la $\sigma$ baja al acortar la ventana", fontsize=9.5)

for ax, (v10v, v3v, lab, fmt) in zip(axes[1:], [
        ((sg10[v10] >= 0.649).mean() * 100, (sg3[v3] >= 0.649).mean() * 100,
         r"% de pixeles con $\sigma$ saturada", "{:.0f} %"),
        (np.median(r10[v10]), np.median(r3[v3]),
         "RMSE del ajuste (NDWI)", "{:.3f}")]):
    ax.bar([0, 1], [v10v, v3v], color=["0.62", ACENTO], width=0.55)
    for i, val in enumerate([v10v, v3v]):
        ax.text(i, val, fmt.format(val), ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["10 anos", "epoca\n3 anos"],
                                              fontsize=8.5)
    ax.set_title(lab, fontsize=9.5)
    ax.set_ylim(0, max(v10v, v3v) * 1.25)
fig.suptitle("Por que ajustamos por epocas — mismos pixeles, misma rejilla, "
             "solo cambia la ventana temporal", fontsize=10.5, y=1.04)
save(fig, "07_epocas.png")

# ═════════════════════════════════════════════════════════════════════════
print("8. super-resolucion")
fino = load("products_villaviciosa/hsr_dem_fine.tif")
cy, cx = int(np.mean(ys)), int(np.mean(xs))
R = 60
z10 = hsr[cy - R:cy + R, cx - R:cx + R]
z25 = fino[4 * (cy - R):4 * (cy + R), 4 * (cx - R):4 * (cx + R)]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, arr, lab in [(axes[0], z10, "etapa 1 — 10 m (el producto bueno)"),
                     (axes[1], z25, "etapa 2 — 2.5 m (EN PRUEBAS)")]:
    im = ax.imshow(arr, cmap=TERRAIN, vmin=np.nanpercentile(z10, 2),
                   vmax=np.nanpercentile(z10, 98), interpolation="nearest")
    ax.set_title(f"{lab}   ({arr.shape[0]}x{arr.shape[1]} px)", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.colorbar(im, ax=axes, fraction=0.025, label="cota (m)")
fig.suptitle(r"Etapa 2: la $\sigma$ dice cuanto relieve repartir dentro de "
             "cada pixel.\nTodavia deja un patron de tablero: hay que "
             "arreglarlo antes de usarla", fontsize=10.5, y=1.04)
save(fig, "08_superresolucion.png")

print("\nlisto ->", os.path.abspath(OUT))
