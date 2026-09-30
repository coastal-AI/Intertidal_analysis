# -*- coding: utf-8 -*-
"""Cloud screening over the transition zone, not the frame, for the paper
(default site: the Westerschelde, whose frame is mostly open sea).

Every scene is screened by the cloudy share of the TRANSITION ZONE (the
pixels that are neither always land nor always water), not of the whole
frame, and every observation is then masked pixel by pixel with the
Sentinel-2 scene classification. A scene with clouds over the land or the
open sea but a clear flat is kept; a whole-frame rule would discard it.

(a) the kept scene with the cloudiest frame, in true colour (B04/B03/B02
    fetched on the cube grid by experiments/fetch_true_colour.py);
(b) its Sentinel-2 scene classification (SCL), official colours;
(c) the reference map of the archive: stable water, stable land and the
    transition zone (the notebooks' classes and colours);
(d) the cloud mask of that scene laid over the transition zone: the share
    of the zone under cloud is what the rule looks at;
(e) every scene of the archive: cloudy share of the frame against cloudy
    share of the transition zone, with the 10 % rules and the counts each
    rule keeps.

Writes docs/paper/figures/fig_clouds.png and results/cloud_screening.json.
Run from the repository root:  python -m experiments.paper_fig_clouds [site]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from matplotlib.patches import Patch

matplotlib.use("Agg")

from experiments.paper_fig_pipeline import crop_box  # noqa: E402
from experiments.paper_style import INK, RED, TIDE, clean, outline, scene_bands, true_colour  # noqa: E402

SITES = {
    "villaviciosa": dict(name="Ría de Villaviciosa", cube="ndwi_cube_villaviciosa_grande_10y.nc",
                         products="products_villaviciosa", years="2016–2025"),
    "escalda": dict(name="Westerschelde", cube="ndwi_cube_escalda_2023-2025_20m.nc",
                    products="products_escalda", years="2023–2025"),
    "wadden": dict(name="Wadden Sea (Vlie)", cube="ndwi_cube_wadden_2023-2025_20m.nc",
                   products="products_wadden", years="2023–2025"),
    "ems": dict(name="Ems-Dollard", cube="ndwi_cube_ems_2023-2025_20m.nc",
                products="products_ems", years="2023–2025"),
}
RULE = 0.10                    # the notebooks' CLOUD_THRESHOLD, on the transition zone
CLEAN = 0.05                   # the notebooks' CLEAN_SCENE_MAX_BAD, on the frame
OUT = "docs/paper/figures"
# the notebooks' reference-map colours (pyintertidal.viz.plot_reference_map)
REF_CLASSES = {0: ("transition zone", "#d62728"), 1: ("stable water", "#1f77b4"), 2: ("stable land", "#d2b48c")}


def cloud_shares(cube_path, ref_path):
    """Per scene: cloudy share of the frame and of the transition zone."""
    from pyintertidal.water import BAD_CLASSES
    trans = rasterio.open(ref_path).read(1) == 0
    ds = xr.open_dataset(cube_path)
    tdim = "t" if "t" in ds.dims else "time"
    times = np.array([str(v)[:10] for v in ds[tdim].values])
    y = ds["y"].values
    n = ds.sizes[tdim]
    frame, zone, nodata = np.zeros(n), np.zeros(n), np.zeros(n)
    step = 20
    for i0 in range(0, n, step):
        scl = np.nan_to_num(ds["SCL"].isel({tdim: slice(i0, i0 + step)}).values.astype(float), nan=0).astype(np.int16)
        if y[0] < y[-1]:
            scl = scl[:, ::-1, :]
        bad = np.isin(scl, list(BAD_CLASSES))
        frame[i0:i0 + step] = bad.mean(axis=(1, 2))
        zone[i0:i0 + step] = bad[:, trans].mean(axis=1)
        nodata[i0:i0 + step] = (scl == 0).mean(axis=(1, 2))      # tile edges: no acquisition
    ds.close()
    return times, frame, zone, nodata


def main(site="escalda"):
    from pyintertidal.water import BAD_CLASSES, SCL_CLASSES
    S = SITES[site]
    ref_path = f"{S['products']}/reference_map.tif"
    mask_path = f"{S['products']}/intertidal_mask.tif"
    times, frame, zone, nodata = cloud_shares(S["cube"], ref_path)
    # the notebooks' rule (pyintertidal.stability.reference_and_clouds +
    # filter_dates): a scene with at most CLEAN of its frame cloudy is
    # clean outright; any other scene is judged on its transition zone
    kept_zone = (frame <= CLEAN) | (zone <= RULE)
    kept_frame = frame <= RULE
    recovered = kept_zone & ~kept_frame
    lost = kept_frame & ~kept_zone
    stats = {"site": site, "scenes": int(len(times)), "kept_transition_rule": int(kept_zone.sum()),
             "kept_frame_rule": int(kept_frame.sum()), "recovered": int(recovered.sum()),
             "lost": int(lost.sum()), "rule": RULE, "clean_frame": CLEAN}
    print(stats)
    # the example: the kept scene with the cloudiest frame, among scenes
    # that cover the whole frame (no tile edge inside it)
    cand = np.flatnonzero(kept_zone & (nodata < 0.01))
    j = int(cand[np.argmax(frame[cand])])
    date = times[j]
    rgb_path = f"data_v4/{site}_truecolour_{date}.tif"
    if not os.path.exists(rgb_path):
        raise SystemExit(f"{rgb_path} missing: run  python -m experiments.fetch_true_colour {site} {date}")
    rgb = true_colour(rgb_path)
    _, _, scl = scene_bands(S["cube"], date)
    ref = rasterio.open(ref_path).read(1)
    mask = rasterio.open(mask_path).read(1) > 0
    trans = ref == 0
    rs, cs = crop_box(mask, margin=40)
    bad = np.isin(scl, list(BAD_CLASSES))
    crop = (rs, cs)
    portrait = (rs.stop - rs.start) > (cs.stop - cs.start)

    # layout: four maps and the scatter; the legends of (b) and (c) under the maps
    if portrait:
        fig = plt.figure(figsize=(7.2, 2.9), dpi=300)
        gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.16], width_ratios=[1, 1, 1, 1, 0.3, 1.3], hspace=0.05, wspace=0.08)
        axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
        axg = fig.add_subplot(gs[1, :4]); axe = fig.add_subplot(gs[:, 5])
    else:
        fig = plt.figure(figsize=(7.2, 3.5), dpi=300)
        gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.22], width_ratios=[1, 1, 0.12, 1.15], hspace=0.55, wspace=0.06)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        axg = fig.add_subplot(gs[2, :2]); axe = fig.add_subplot(gs[:2, 3])
    T = dict(loc="left", fontsize=6.5)
    # (a) the scene as the eye sees it
    ax = axes[0]
    ax.imshow(rgb[crop], interpolation="antialiased")
    ax.set_title(f"(a) {date}, true colour\n    frame {frame[j]:.0%} cloudy, zone {zone[j]:.0%}", **T)
    clean(ax)
    # (b) the scene classification
    ax = axes[1]
    present = [c for c in sorted(np.unique(scl[crop]).tolist()) if c in SCL_CLASSES]
    lut = np.zeros((max(SCL_CLASSES) + 1, 3))
    for c, (_, col) in SCL_CLASSES.items():
        lut[c] = matplotlib.colors.to_rgb(col)
    ax.imshow(lut[np.clip(scl[crop], 0, len(lut) - 1)], interpolation="antialiased")
    ax.set_title("(b) scene classification\n    (SCL)", **T)
    clean(ax)
    # (c) the reference map of the archive
    ax = axes[2]
    lut3 = np.array([matplotlib.colors.to_rgb(REF_CLASSES[k][1]) for k in (0, 1, 2)])
    ax.imshow(lut3[np.clip(ref[crop], 0, 2)], interpolation="antialiased")
    ax.set_title(f"(c) reference map of the\n    archive, {S['years']}", **T)
    clean(ax)
    # (d) the scene blurred to grey, the transition zone tinted, the cloudy
    # classes of the classification in their own colours, the shares inside
    from scipy.ndimage import gaussian_filter
    ax = axes[3]
    lum = gaussian_filter(0.30 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2], 2.0)
    base = np.dstack([lum, lum, lum]) * 0.45 + 0.30       # dark enough for white clouds to show
    canvas = base.copy()
    tint = np.array(matplotlib.colors.to_rgb(REF_CLASSES[0][1]))
    canvas[trans] = 0.35 * base[trans] + 0.65 * tint
    for c in (3, 8, 9, 10):                               # shadow, cloud medium, cloud high, cirrus
        canvas[scl == c] = matplotlib.colors.to_rgb(SCL_CLASSES[c][1])
    ax.imshow(canvas[crop], interpolation="antialiased")
    outline(ax, trans[crop], colour=REF_CLASSES[0][1], lw=0.35)
    ax.text(0.03, 0.03, f"cloudy classes: {frame[j]:.0%} of the frame,\n{zone[j]:.0%} of the transition zone",
            transform=ax.transAxes, fontsize=5.3, ha="left", va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_title("(d) cloudy classes of (a) over\n    the transition zone (red)", **T)
    clean(ax)
    # legends of (b), (c) and (d) under the maps
    axg.axis("off")
    handles = [Patch(fc=SCL_CLASSES[c][1], ec="0.4", lw=0.3, label=SCL_CLASSES[c][0].replace(" probability", " prob."))
               for c in present if c != 0]
    handles += [Patch(fc=REF_CLASSES[k][1], ec="0.4", lw=0.3, label=REF_CLASSES[k][0]) for k in (1, 2, 0)]
    handles += []
    axg.legend(handles=handles, ncol=5 if portrait else 4, fontsize=5.2, loc="upper center", frameon=False,
               handlelength=1.2, columnspacing=1.0, borderaxespad=0.0)
    # (e) the archive under the two rules
    ax = axe
    ax.plot(frame[~kept_zone] * 100, zone[~kept_zone] * 100, ".", ms=2.5, color="0.65", alpha=0.7,
            label=f"discarded ({(~kept_zone).sum()})")
    ax.plot(frame[kept_zone] * 100, zone[kept_zone] * 100, ".", ms=3, color=TIDE, alpha=0.8,
            label=f"kept ({kept_zone.sum()})")
    ax.plot(frame[j] * 100, zone[j] * 100, "o", ms=6, mfc="none", mec=RED, mew=1.2, label="scene (a)")
    ax.axhline(RULE * 100, color=RED, lw=0.9)
    ax.axvline(RULE * 100, color=INK, lw=0.9, ls="--")
    ax.text(39.5, 10.8, "rule used: 10 % of the transition zone", color=RED, fontsize=5.2, ha="right", va="bottom")
    ax.text(10.8, 25.5, f"10 % of the frame would\nkeep {kept_frame.sum()}: {recovered.sum()} enter here,\n{lost.sum()} leave",
            color=INK, fontsize=5.2, ha="left", va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.set_xlim(0, 40); ax.set_ylim(0, 40)
    ax.set_xlabel("cloudy share of the frame (%)", fontsize=6.5)
    ax.set_ylabel("cloudy share of the transition zone (%)", fontsize=6.5)
    beyond = int(((frame > 0.4) | (zone > 0.4)).sum())
    ax.set_title(f"(e) {len(times)} scenes, {S['years']}\n    ({beyond} beyond the axes)", **T)
    ax.legend(fontsize=5.2, loc="upper right", framealpha=0.9); ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.11)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig_clouds.png", dpi=300)
    os.makedirs("results", exist_ok=True)
    json.dump({**stats, "example_scene": date, "example_frame": float(frame[j]), "example_zone": float(zone[j])},
              open(f"results/cloud_screening_{site}.json", "w"), indent=1)
    print(f"-> {OUT}/fig_clouds.png  ({site}, example {date}: frame {frame[j]:.0%}, zone {zone[j]:.1%})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "escalda")
