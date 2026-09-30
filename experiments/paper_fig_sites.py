# -*- coding: utf-8 -*-
"""Figure "Study sites" of the manuscript (docs/paper/figures/fig_sites.png).

Panel (a): locator map of western Europe (Natural Earth 10 m coastline and
land boundaries, public domain) with the five sites. Panels (b)-(f): one per
site, a false-colour Sentinel-2 composite (NIR, green, green) of a clear
low-water scene from the site's own datacube, with the area of interest,
the intertidal zone of the product, the geodesic distance from the mouth,
the tide gauges (boundary and judge) and, for Villaviciosa, the RTK
transect (development subset only). Everything is read from the cubes and
the products already on disk; nothing is recomputed.

Run from the repository root:  python -m experiments.paper_fig_sites
"""
from __future__ import annotations

import json
import os
import zipfile

import geopandas as gpd
import matplotlib
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from pyproj import CRS, Transformer

import pyintertidal as pit

matplotlib.use("Agg")

OUT = "docs/paper/figures/fig_sites"
NE = "data_v4/naturalearth"
GAUGES = json.load(open("data_v4/gauges/ioc_stations.json", encoding="utf-8"))
GAUGE = {g["code"]: g for g in GAUGES}

SITES = [
    # key, panel title, cube, extract, marea product, gauges (code: role), fixed date
    dict(key="villaviciosa", title="Ría de Villaviciosa (ES)",
         cube="ndwi_cube_villaviciosa_grande_10y.nc", extract=None,
         marea="products_villaviciosa_marea/marea.npz",
         gauges={}, date="2019-02-21", scale_km=2, rtk=True, zoom_to_aoi=True),
    dict(key="ferrol", title="Ría de Ferrol (ES)",
         cube="ndwi_cube_ferrol_2023-2025_10m.nc",
         extract="products_ferrol/marea_extract.npz",
         marea="products_ferrol/marea_eot20_g1/marea.npz",
         gauges={"fer1": "boundary", "fer2": "judge"}, date=None, scale_km=2),
    dict(key="escalda", title="Westerschelde (NL)",
         cube="ndwi_cube_escalda_2023-2025_20m.nc",
         extract="products_escalda/marea_extract.npz",
         marea="products_escalda/marea_eot20_g2/marea.npz",
         gauges={"vlis": "boundary", "brsk": "boundary", "trnz": "judge"},
         date=None, scale_km=5),
    dict(key="ems", title="Ems-Dollard (NL/DE)",
         cube="ndwi_cube_ems_2023-2025_20m.nc",
         extract="products_ems/marea_extract.npz",
         marea="products_ems/marea_eot20_g1/marea.npz",
         gauges={"bork": "boundary", "delf": "judge"}, date=None, scale_km=5),
    dict(key="wadden", title="Wadden Sea, Vlie basin (NL)",
         cube="ndwi_cube_wadden_2023-2025_20m.nc",
         extract="products_wadden/marea_extract.npz",
         marea="products_wadden/marea_eot20_g1/marea.npz",
         gauges={"ters": "boundary", "harl": "judge"}, date=None, scale_km=5),
]

INK = "#14313c"; YEL = "#ffd166"; MAG = "#ff5da2"; WHITE = "white"


def low_water_scene(extract_path):
    """The clearest low-water scene: lowest wet fraction among scenes with
    at least 80 % clear intertidal pixels."""
    d = np.load(extract_path, allow_pickle=True)
    Y, C, dates = d["Y"], d["C"], d["dates"]
    cf = C.mean(axis=1)
    wf = np.array([np.mean(Y[t][C[t]] > 0) if C[t].any() else np.nan
                   for t in range(len(dates))])
    ok = cf >= 0.95
    if not ok.any():
        ok = cf >= 0.85
    t = int(np.nanargmin(np.where(ok, wf, np.nan)))
    return str(dates[t]), float(wf[t]), float(cf[t])


def composite(cube_path, date):
    ds = xr.open_dataset(cube_path)
    tdim = "t" if "t" in ds.dims else "time"
    times = np.array([str(v)[:10] for v in ds[tdim].values])
    i = int(np.where(times == date)[0][0])
    g = ds["B03"].isel({tdim: i}).values.astype(float)
    n = ds["B08"].isel({tdim: i}).values.astype(float)
    x, y = ds["x"].values, ds["y"].values
    wkt = ds["B03"].attrs.get("crs_wkt") or ds["crs"].attrs.get("crs_wkt")
    ds.close()

    def stretch(a):
        lo, hi = np.nanpercentile(a, [1, 99])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.7
    rgb = np.dstack([stretch(n), stretch(g), stretch(g)])
    dx = abs(float(x[1] - x[0])); dy = abs(float(y[1] - y[0]))
    extent = [float(x.min()) - dx / 2, float(x.max()) + dx / 2,
              float(y.min()) - dy / 2, float(y.max()) + dy / 2]
    if y[0] < y[-1]:
        rgb = rgb[::-1]
    return rgb, extent, CRS.from_wkt(wkt), (len(y), len(x))


def product_fields(marea_path, shape):
    d = np.load(marea_path)
    H, W = int(d["shape"][0]), int(d["shape"][1])
    assert (H, W) == shape, (marea_path, (H, W), shape)
    inter = np.zeros(H * W, bool); inter[d["keep"]] = True
    s = np.full(H * W, np.nan, np.float32); s[d["keep"]] = d["s_km"]
    return inter.reshape(H, W), s.reshape(H, W)


def scalebar(ax, extent, km, color=WHITE):
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + 0.06 * (extent[3] - extent[2])
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=color, lw=3,
            solid_capstyle="butt")
    ax.text(x0 + km * 500, y0 + 0.015 * (extent[3] - extent[2]), f"{km} km",
            color=color, ha="center", va="bottom", fontsize=8, fontweight="bold")


def north(ax, extent, color=WHITE):
    x = extent[1] - 0.07 * (extent[1] - extent[0])
    y = extent[3] - 0.16 * (extent[3] - extent[2])
    ax.annotate("N", xy=(x, y + 0.08 * (extent[3] - extent[2])), xytext=(x, y),
                ha="center", va="center", color=color, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def lonlat_ticks(ax, extent, crs, n=3):
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs = np.linspace(extent[0], extent[1], n + 2)[1:-1]
    ys = np.linspace(extent[2], extent[3], n + 2)[1:-1]
    lon, _ = to_ll.transform(xs, np.full_like(xs, extent[2]))
    _, lat = to_ll.transform(np.full_like(ys, extent[0]), ys)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.set_xticklabels([f"{abs(v):.2f}°{'E' if v >= 0 else 'W'}" for v in lon], fontsize=7)
    ax.set_yticklabels([f"{v:.2f}°N" for v in lat], fontsize=7, rotation=90, va="center")
    ax.tick_params(length=2, pad=1)


def draw_site(ax, cfg, letter):
    if cfg["date"] is None:
        date, wf, cf = low_water_scene(cfg["extract"])
    else:
        date, wf, cf = cfg["date"], np.nan, np.nan
    rgb, extent, crs, shape = composite(cfg["cube"], date)
    inter, s_km = product_fields(cfg["marea"], shape)
    to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    ax.imshow(rgb, extent=extent, interpolation="bilinear")
    # intertidal zone of the product
    ax.contour(inter, levels=[0.5], colors=[YEL], linewidths=0.6,
               extent=extent, origin="upper")
    smax = float(np.nanmax(s_km))
    # area of interest
    aoi = pit.sites.get(cfg["key"])
    poly = np.array(aoi.polygon.exterior.coords, float)
    px, py = to_xy.transform(poly[:, 0], poly[:, 1])
    ax.plot(np.r_[px, px[0]], np.r_[py, py[0]], color=WHITE, lw=0.9, ls="--")
    # gauges: markers may sit on the frame edge; gauges outside the frame get
    # an arrow at the edge with their distance
    stroke = [matplotlib.patheffects.withStroke(linewidth=2, foreground=INK)]
    W_ = extent[1] - extent[0]; H_ = extent[3] - extent[2]
    for code, role in cfg["gauges"].items():
        g = GAUGE[code]; gx, gy = to_xy.transform(g["lon"], g["lat"])
        name = (g["Location"].replace(" Noordzee", "").replace(" Fischerbalje", "")
                .replace(" Handelshaven", ""))
        marker = dict(marker="*", ms=13, mfc=YEL) if role == "judge" else \
            dict(marker="o", ms=8, mfc=WHITE)
        inside = extent[0] <= gx <= extent[1] and extent[2] <= gy <= extent[3]
        if inside:
            ax.plot(gx, gy, mec=INK, mew=0.8, ls="", clip_on=False, **marker)
            ha = "right" if gx > extent[0] + 0.8 * W_ else "left"
            va = "top" if gy > extent[2] + 0.85 * H_ else "bottom"
            dx = -6 if ha == "right" else 6; dy = -6 if va == "top" else 6
            ax.annotate(name, (gx, gy), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7, color=WHITE, fontweight="bold", ha=ha, va=va,
                        path_effects=stroke)
        else:
            cx, cy = extent[0] + W_ / 2, extent[2] + H_ / 2
            ex = float(np.clip(gx, extent[0] + 0.06 * W_, extent[1] - 0.06 * W_))
            ey = float(np.clip(gy, extent[2] + 0.06 * H_, extent[3] - 0.06 * H_))
            d_km = np.hypot(gx - ex, gy - ey) / 1000
            ux, uy = (gx - cx), (gy - cy); nrm = np.hypot(ux, uy)
            ax.annotate("", xy=(ex, ey), xytext=(ex - 0.08 * W_ * ux / nrm,
                                                 ey - 0.08 * W_ * uy / nrm),
                        arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=1.2))
            ax.plot(ex, ey, mec=INK, mew=0.8, ls="", clip_on=False, **marker)
            ang = np.degrees(np.arctan2(gy - ey, gx - ex)) % 360
            comp = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"][int(((ang + 22.5) % 360) // 45)]
            ax.annotate(f"{name} ({d_km:.0f} km {comp})", (ex, ey),
                        xytext=(0, -8), textcoords="offset points", fontsize=6.5,
                        color=WHITE, fontweight="bold", ha="center", va="top",
                        path_effects=stroke)
    # RTK transect (development subset only: the reserved blocks stay sealed)
    if cfg.get("rtk"):
        import rasterio
        from pyintertidal import rtk
        dev = rtk.load_rtk()                       # development rows only
        with rasterio.open(rtk.DEFAULT_GRID) as src:
            tr, gcrs = src.transform, src.crs
        ex = tr.c + (dev["col"] + 0.5) * tr.a
        ny = tr.f + (dev["row"] + 0.5) * tr.e
        rx, ry = Transformer.from_crs(CRS.from_wkt(gcrs.to_wkt()), crs,
                                      always_xy=True).transform(ex, ny)
        ax.plot(rx, ry, ".", ms=2.2, color=MAG)
        print(f"  RTK development points drawn: {len(rx)} "
              f"({dev['n_reserved_hidden']} reserved, not drawn)")
    if cfg.get("zoom_to_aoi"):
        mx, my = 0.04 * (px.max() - px.min()), 0.04 * (py.max() - py.min())
        extent = [max(extent[0], px.min() - mx), min(extent[1], px.max() + mx),
                  max(extent[2], py.min() - my), min(extent[3], py.max() + my)]
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    lonlat_ticks(ax, extent, crs)
    scalebar(ax, extent, cfg["scale_km"]); north(ax, extent)
    ax.set_title(f"({letter}) {cfg['title']}\nSentinel-2, {date}",
                 loc="left", fontsize=8.5, pad=3, linespacing=1.1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    print(f"{cfg['key']}: scene {date} (wet fraction {wf:.2f}, clear {cf:.2f}); "
          f"s_max {smax:.1f} km; intertidal px {int(inter.sum()):,}")
    return crs


def draw_locator(ax):
    land = gpd.read_file(f"zip://{NE}/ne_10m_land.zip")
    coast = gpd.read_file(f"zip://{NE}/ne_10m_coastline.zip")
    borders = gpd.read_file(f"zip://{NE}/ne_10m_admin_0_boundary_lines_land.zip")
    land.plot(ax=ax, color="#e9e4d8", edgecolor="none")
    coast.plot(ax=ax, color=INK, linewidth=0.45)
    borders.plot(ax=ax, color="0.55", linewidth=0.4, linestyle=":")
    ax.set_xlim(-10.5, 9.5); ax.set_ylim(41.5, 55.5)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.5)))
    ax.set_facecolor("#dbe9f4")
    for letter, cfg in zip("bcdef", SITES):
        aoi = pit.sites.get(cfg["key"])
        poly = np.array(aoi.polygon.exterior.coords, float)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        ax.plot(cx, cy, "s", ms=6, mfc=YEL, mec=INK, mew=0.8)
        ax.annotate(f"({letter})", (cx, cy), xytext=(6, -3), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=INK)
    for name, (x, y) in {"Bay of Biscay": (-5.5, 45.6), "North Sea": (3.6, 55.0),
                         "Spain": (-3.5, 41.9), "France": (2.0, 47.0),
                         "Netherlands": (5.2, 52.3), "Germany": (8.4, 51.6),
                         "United Kingdom": (-1.8, 52.6)}.items():
        ax.text(x, y, name, fontsize=7, color="0.35", ha="center",
                style="italic" if "Sea" in name or "Bay" in name else "normal")
    ax.set_xticks([-9, -6, -3, 0, 3, 6, 9]); ax.set_yticks([42, 45, 48, 51, 54])
    ax.set_xticklabels([f"{abs(v)}°{'E' if v > 0 else 'W' if v < 0 else ''}" for v in [-9, -6, -3, 0, 3, 6, 9]], fontsize=7)
    ax.set_yticklabels([f"{v}°N" for v in [42, 45, 48, 51, 54]], fontsize=7)
    ax.tick_params(length=2, pad=1)
    ax.set_title("(a) Study sites", loc="left", fontsize=9, pad=3)


def main():
    import matplotlib.patheffects  # noqa: F401  (used through matplotlib.patheffects)
    # two rows by three: at the journal's full text width the figure takes
    # about half a page instead of a whole one (12-page limit)
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 6.2), dpi=220)
    draw_locator(axs[0, 0])
    for ax, letter, cfg in zip([axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]],
                               "bcdef", SITES):
        draw_site(ax, cfg, letter)
    handles = [
        Line2D([], [], color=WHITE, ls="--", lw=0.9, label="area of interest"),
        Line2D([], [], color=YEL, lw=0.8, label="intertidal zone (product)"),
        Line2D([], [], marker="o", ls="", mfc=WHITE, mec=INK, ms=7, label="boundary tide gauge"),
        Line2D([], [], marker="*", ls="", mfc=YEL, mec=INK, ms=11, label="judge tide gauge (held out)"),
        Line2D([], [], marker=".", ls="", color=MAG, ms=6, label="RTK-GNSS transect"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.005), facecolor="0.5")
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.6, w_pad=0.6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=220)
    fig.savefig(OUT + ".pdf")
    print("written", OUT + ".png/.pdf")


if __name__ == "__main__":
    main()
