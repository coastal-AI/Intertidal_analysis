# -*- coding: utf-8 -*-
"""Figure "Census" of the manuscript (docs/paper/figures/fig_census.png).

Map of the north coast of Spain (Fisterra to the Bidasoa) with the 130
terrain-following cells of the MAREA campaign, each filled by the verdict
the campaign reached in that cell: interior tide adjustment applied, or
uniform water level kept. Cells whose status is not ``ok`` (no intertidal
zone found) are hatched grey. Styled after Fig. 1 of Bishop-Taylor et al.
(2019): polygons over a subtle land/sea underlay, legend with counts,
scale bar, graticule, no title.

Inputs (nothing is recomputed):
  north_coast_cells_v2.json            cell polygons (lon/lat), byte-identical
                                       to north_coast_cells.json, the file the
                                       campaign (experiments/campaign_north_v2.py)
                                       reads; produced by experiments/make_cells_v2.py
  deliverables/north_coast_census.csv  one row per cell with the verdict
  data_v4/naturalearth/*.zip           Natural Earth 10 m land, coastline,
                                       land boundaries (public domain)

Run from the repository root:  python -m experiments.paper_fig_census
"""
from __future__ import annotations

import json
import os

import geopandas as gpd
import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from shapely.geometry import Polygon

matplotlib.use("Agg")

OUT = "docs/paper/figures/fig_census.png"
CELLS = "north_coast_cells_v2.json"
CENSUS = "deliverables/north_coast_census.csv"
NE = "data_v4/naturalearth"

# map window (lon/lat, degrees): the whole cell set with a margin
LON0, LON1 = -9.6, -1.1
LAT0, LAT1 = 42.45, 44.15
LAT_MID = 0.5 * (LAT0 + LAT1)

# style shared with paper_fig_sites.draw_locator
INK = "#14313c"; LAND = "#e9e4d8"; SEA = "#dbe9f4"; YEL = "#ffd166"
RED = "#c1121f"; BLUE = "#2a6f97"; GREY = "#b8b8b8"

CLASSES = {
    # key: (label, facecolor rgba, hatch)
    "applied": ("interior lag applied", to_rgba(RED, 0.70), None),
    "uniform": ("uniform level kept", to_rgba(BLUE, 0.50), None),
    "nodata": ("no intertidal zone found", to_rgba(GREY, 0.85), "//////"),
}

# the two study sites of the paper that lie on this coast:
# lon, lat, label offset (points), ha, va -- Ferrol labelled over the open
# sea to its north-west, Villaviciosa over the land to its south-west, so
# neither label sits on a neighbouring cell
SITES = {"Villaviciosa": (-5.39, 43.53, (-4, -5), "right", "top"),
         "Ferrol": (-8.25, 43.48, (-5, 2), "right", "bottom")}


def load_cells():
    raw = json.load(open(CELLS, encoding="utf-8"))
    gdf = gpd.GeoDataFrame(
        {"cell": [c["name"] for c in raw], "km2": [c["km2"] for c in raw]},
        geometry=[Polygon(c["polygon"]).buffer(0) for c in raw], crs="EPSG:4326")
    census = pd.read_csv(CENSUS)
    census["adopted"] = census["interior_tide_adopted"].map(
        lambda v: str(v).strip().lower() == "true")
    gdf = gdf.merge(census, on="cell", how="left", validate="one_to_one")
    if gdf["status"].isna().any():
        missing = gdf.loc[gdf["status"].isna(), "cell"].tolist()
        raise SystemExit(f"cells without a census row: {missing}")
    gdf["klass"] = np.where(
        gdf["status"] != "ok", "nodata",
        np.where(gdf["adopted"], "applied", "uniform"))
    return gdf


def scalebar(ax, lon, lat, km_total=100, km_step=50):
    """Alternating black/white bar with distances, true at its own latitude."""
    deg_per_km = 1.0 / (111.32 * np.cos(np.deg2rad(lat)))
    h = 0.035                                   # bar height, degrees of latitude
    n = int(round(km_total / km_step))
    for i in range(n):
        x = lon + i * km_step * deg_per_km
        ax.add_patch(matplotlib.patches.Rectangle(
            (x, lat), km_step * deg_per_km, h,
            facecolor=INK if i % 2 == 0 else "white", edgecolor=INK,
            linewidth=0.5, zorder=6))
    for i in range(n + 1):
        km = i * km_step
        ax.text(lon + km * deg_per_km, lat + h + 0.02,
                f"{km} km" if i == n else f"{km}", ha="center", va="bottom",
                fontsize=6, color=INK, zorder=6)


def main():
    plt.rcParams.update({"font.size": 7, "hatch.linewidth": 0.35})
    gdf = load_cells()
    counts = gdf["klass"].value_counts().to_dict()
    print("geometry source:", CELLS, f"({len(gdf)} cells)")
    print("classes:", counts)
    for _, r in gdf[gdf.klass == "nodata"].iterrows():
        c = r.geometry.centroid
        print(f"non-ok cell: {r.cell} ({r.status}) at {c.x:.2f}E {c.y:.2f}N")

    # Natural Earth underlay, clipped to the window for speed
    win = (LON0 - 0.5, LAT0 - 0.5, LON1 + 0.5, LAT1 + 0.5)
    land = gpd.read_file(f"zip://{NE}/ne_10m_land.zip").cx[win[0]:win[2], win[1]:win[3]]
    coast = gpd.read_file(f"zip://{NE}/ne_10m_coastline.zip").cx[win[0]:win[2], win[1]:win[3]]
    borders = gpd.read_file(f"zip://{NE}/ne_10m_admin_0_boundary_lines_land.zip") \
        .cx[win[0]:win[2], win[1]:win[3]]

    # figure geometry: the axes box exactly fits the window at the local
    # aspect so no space is wasted; 7.2 in wide (journal text width)
    aspect = 1 / np.cos(np.deg2rad(LAT_MID))
    W_IN = 7.2; ML, MR, MB, MT = 0.40, 0.08, 0.28, 0.06
    ax_w = W_IN - ML - MR
    ax_h = ax_w * (LAT1 - LAT0) * aspect / (LON1 - LON0)
    H_IN = ax_h + MB + MT
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=200)
    ax = fig.add_axes([ML / W_IN, MB / H_IN, ax_w / W_IN, ax_h / H_IN])

    ax.set_facecolor(SEA)
    land.plot(ax=ax, color=LAND, edgecolor="none", zorder=1)
    coast.plot(ax=ax, color=INK, linewidth=0.4, zorder=2)
    borders.plot(ax=ax, color="0.5", linewidth=0.4, linestyle=":", zorder=2)

    # cells: fill by verdict, thin ink edges; hatched when no intertidal zone
    for key, (label, face, hatch) in CLASSES.items():
        sub = gdf[gdf.klass == key]
        if len(sub):
            sub.plot(ax=ax, facecolor=face, edgecolor=INK, linewidth=0.3,
                     hatch=hatch, zorder=4)

    # study sites of the paper on this coast
    stroke = [matplotlib.patheffects.withStroke(linewidth=1.8, foreground="white")]
    for name, (x, y, off, ha, va) in SITES.items():
        ax.plot(x, y, "s", ms=4.5, mfc=YEL, mec=INK, mew=0.7, zorder=7)
        ax.annotate(name, (x, y), xytext=off, textcoords="offset points",
                    fontsize=6.5, fontweight="bold", color=INK, ha=ha,
                    va=va, path_effects=stroke, zorder=7)

    # geographic labels
    for name, (x, y), kw in [
            ("Bay of Biscay", (-5.4, 43.98), dict(style="italic")),
            ("Atlantic Ocean", (-9.0, 43.63), dict(style="italic")),
            ("Spain", (-6.3, 42.75), dict()),
            ("France", (-1.33, 43.32), dict(fontsize=6))]:
        ax.text(x, y, name, color="0.4", ha="center", va="center", zorder=3,
                **{"fontsize": 6.5, **kw})

    scalebar(ax, -9.35, 43.92)

    # legend with the counts, on the land in the lower right
    handles = [Patch(facecolor=face, edgecolor=INK, linewidth=0.4, hatch=hatch,
                     label=f"{label} (n = {counts.get(key, 0)})")
               for key, (label, face, hatch) in CLASSES.items()]
    handles.append(matplotlib.lines.Line2D(
        [], [], marker="s", ls="", mfc=YEL, mec=INK, mew=0.7, ms=4.5,
        label="study site of the paper"))
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.5,
                    title=f"MAREA verdict, {len(gdf)} cells of 10 km",
                    title_fontsize=6.5, frameon=True, framealpha=0.92,
                    edgecolor="0.7", fancybox=False, borderpad=0.5,
                    labelspacing=0.35, handlelength=1.6, handleheight=1.0,
                    handletextpad=0.6, borderaxespad=0.4)
    leg.get_frame().set_linewidth(0.4)
    leg.set_zorder(8)

    # window, aspect, graticule and ticks
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
    ax.set_aspect(aspect)
    xt = np.arange(-9, -1, 1); yt = np.array([42.5, 43.0, 43.5, 44.0])
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([f"{abs(v)}°W" for v in xt], fontsize=7)
    ax.set_yticklabels([f"{v:g}°N" for v in yt], fontsize=7)
    ax.tick_params(length=2, pad=1.5, width=0.5)
    ax.grid(True, color="0.55", linewidth=0.3, linestyle=":", alpha=0.7, zorder=1.5)
    ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"written {OUT}  ({W_IN:.2f} x {H_IN:.2f} in, 200 dpi)")


if __name__ == "__main__":
    main()
