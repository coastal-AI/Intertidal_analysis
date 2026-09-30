# -*- coding: utf-8 -*-
"""Water detection on one site, for the paper (default: the Ría de Ferrol):

(a) the NDWI histogram of the clear-sky pixels of the twelve clearest
    scenes, with the adopted threshold;
(b) the water-frequency map, the notebook's own rendering, framed by the
    study polygon;
(c) the water-frequency histogram of the transition pixels with the
    multi-Otsu cuts and the adopted intertidal window.

Inputs: the site's cube and products_<site>/{water_frequency,reference_map}.tif
(rebuild_site_masks.py, identical to the notebook's in-memory rasters).
Writes docs/paper/figures/fig_water.png.
Run from the repository root:  python -m experiments.paper_fig_water [site]
"""
from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from experiments.paper_fig_pipeline import read  # noqa: E402
from experiments.paper_style import INK, RED, TIDE  # noqa: E402

YEL = "#ffd166"; MUD = "#b08968"
SITES = {
    "villaviciosa": dict(name="Ría de Villaviciosa", cube="ndwi_cube_villaviciosa_grande_10y.nc", res=10,
                         years=("2016-01-01", "2025-12-31"), products="products_villaviciosa"),
    "ferrol": dict(name="Ría de Ferrol", cube="ndwi_cube_ferrol_2023-2025_10m.nc", res=10,
                   years=("2023-01-01", "2025-12-31"), products="products_ferrol"),
    "escalda": dict(name="Westerschelde", cube="ndwi_cube_escalda_2023-2025_20m.nc", res=20,
                    years=("2023-01-01", "2025-12-31"), products="products_escalda"),
}
NDWI_THRESHOLD = 0.0          # the notebooks' adopted threshold
WINDOW = (0.05, 0.95)         # the notebooks' adopted water-frequency window
OUT = "docs/paper/figures/fig_water.png"


def main(site="ferrol"):
    import pyintertidal as pit
    from pyintertidal import SentinelCube, viz, water as W
    from pyintertidal.frequency import multiotsu_window
    S = SITES[site]
    wf, transform, crs = read(f"{S['products']}/water_frequency.tif")
    trans = read(f"{S['products']}/reference_map.tif")[0] == 0     # the notebook's transition class
    aoi = pit.sites.get(site)
    cube = SentinelCube(aoi, S["years"], water="ndwi", cache_path=S["cube"], resolution=S["res"])
    thr, info = W.calibrate_threshold(cube, n_scenes=12, verbose=False)
    y0, y1 = S["years"][0][:4], S["years"][1][:4]

    wide = S["res"] == 10 and site != "villaviciosa"       # Ferrol's box is a wide rectangle
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), dpi=300,
                            gridspec_kw=dict(width_ratios=[1.0, 1.7, 1.0] if wide else [1.25, 1.0, 1.25]))
    ax = axs[0]
    smp = np.asarray(info["sample"])
    ax.hist(smp[smp <= NDWI_THRESHOLD], bins=60, range=(-0.6, NDWI_THRESHOLD), color=MUD, label="read dry (land)")
    ax.hist(smp[smp > NDWI_THRESHOLD], bins=60, range=(NDWI_THRESHOLD, 0.6), color=TIDE, label="read wet (water)")
    ax.axvline(NDWI_THRESHOLD, color=RED, lw=1.2, label=f"threshold {NDWI_THRESHOLD:+.2f}")
    ax.text(NDWI_THRESHOLD - 0.03, 0.93, "land", transform=ax.get_xaxis_transform(), ha="right", fontsize=7, color="#7a5c3a")
    ax.text(NDWI_THRESHOLD + 0.03, 0.93, "water", transform=ax.get_xaxis_transform(), ha="left", fontsize=7, color=TIDE)
    ax.set_xlabel("NDWI, clear-sky pixels of 12 clearest scenes", fontsize=6.5); ax.set_ylabel("pixels", fontsize=7)
    ax.set_title("(a) the water threshold", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="center right"); ax.tick_params(labelsize=6); ax.set_yticks([])
    ax = axs[1]
    # the notebook's own rendering (viz preset: Blues, 0-1, zoom to the AOI)
    viz.plot_water_frequency(wf, transform, crs, aoi, ax=ax, title=None)
    for coll in ax.figure.axes:
        if coll is not ax and coll not in axs:
            coll.tick_params(labelsize=6); coll.set_ylabel(coll.get_ylabel(), fontsize=6)
    # the study polygon is the frame: crop to its extent and drop the box
    pm = aoi.raster_mask(transform, crs, wf.shape)
    pr, pc = np.flatnonzero(pm.any(1)), np.flatnonzero(pm.any(0))
    ax.set_xlim(pc.min() - 3, pc.max() + 3); ax.set_ylim(pr.max() + 3, pr.min() - 3)
    ax.set_title("(b) water frequency", loc="left", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    ax = axs[2]
    lo, hi = multiotsu_window(wf, trans)
    v = wf[trans]; v = v[np.isfinite(v)]
    ax.hist(v[v < WINDOW[0]], bins=5, range=(0, WINDOW[0]), color=MUD)
    ax.hist(v[(v >= WINDOW[0]) & (v <= WINDOW[1])], bins=45, range=WINDOW, color=YEL)
    ax.hist(v[v > WINDOW[1]], bins=5, range=(WINDOW[1], 1), color=TIDE)
    ax.axvline(lo, color=INK, ls="--", lw=0.9); ax.axvline(hi, color=INK, ls="--", lw=0.9, label=f"multi-Otsu cuts {lo:.2f} / {hi:.2f}")
    ax.axvline(WINDOW[0], color=RED, lw=1.0); ax.axvline(WINDOW[1], color=RED, lw=1.0, label=f"adopted window {WINDOW[0]:.2f}–{WINDOW[1]:.2f}")
    for x_, lab, col in ((WINDOW[0] / 2, "supratidal\n(almost never wet)", "#7a5c3a"),
                         (0.5, "intertidal", "#8a6d1a"), ((1 + WINDOW[1]) / 2, "subtidal\n(almost always wet)", TIDE)):
        ax.text(x_, 0.93, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=col)
    ax.set_xlabel("water frequency of transition pixels", fontsize=7); ax.set_yticks([])
    ax.set_title("(c) the intertidal window", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="center"); ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"-> {OUT}  ({S['name']} {y0}–{y1}: Otsu threshold {thr:+.3f} (adopted {NDWI_THRESHOLD:+.2f}), "
          f"multi-Otsu cuts {lo:.2f}/{hi:.2f}, {int(trans.sum()):,} transition px)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ferrol")
