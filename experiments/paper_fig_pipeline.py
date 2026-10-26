# -*- coding: utf-8 -*-
"""Three illustrative figures of the pipeline on the Ría de Villaviciosa,
for the paper, built from the rasters the topography notebook wrote and
one scene of the cube:

* fig_pipeline.png  — the whole pipeline as a row of real thumbnails: a
  scene, its wet/dry reading, the water frequency, the intertidal zone,
  the boundary tide at each visit, the bands along the channel, the
  interior clock, and the elevation map.
* fig_water.png     — the water detection: the NDWI histogram of the
  clearest scenes with its Otsu threshold, the water-frequency map, and
  the water-frequency histogram with the multi-Otsu cuts and the adopted
  intertidal window.
* fig_products.png  — the products at low water: the clearest low-water
  scene, the MAREA elevation map, the IGN LiDAR on the same grid and
  colour scale (datum offset removed), and the hydroperiod computed from
  the MAREA map and the tide series.

Run from the repository root:  python -m experiments.paper_fig_pipeline
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.patches import FancyArrowPatch

matplotlib.use("Agg")

from experiments.paper_fig_sites import composite, low_water_scene  # noqa: E402

INK = "#14313c"; TIDE = "#2a6f97"; RED = "#c1121f"; YEL = "#ffd166"
CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"
EXTRACT = "marea_demo_extract.npz"
MAREA = "products_marea/marea.npz"
WF_TIF = "products_villaviciosa/water_frequency.tif"
MASK_TIF = "products_villaviciosa/intertidal_mask.tif"
LIDAR = "mdt5_villaviciosa_grande.tif"
EPOCH = ("2023-01-01", "2025-12-31")
NDWI_THRESHOLD = 0.0          # the notebook's adopted threshold (turbid water sits below 0.1)
WINDOW = (0.05, 0.95)         # the notebook's adopted water-frequency window
OUT = "docs/paper/figures"


def read(path):
    with rasterio.open(path) as s:
        return s.read(1).astype(float), s.transform, s.crs


def scene_ndwi(date):
    """NDWI and clear-sky mask of one scene of the cube."""
    from pyintertidal.water import CLEAR_CLASSES
    ds = xr.open_dataset(CUBE)
    tdim = "t" if "t" in ds.dims else "time"
    times = np.array([str(v)[:10] for v in ds[tdim].values])
    i = int(np.where(times == date)[0][0])
    g = ds["B03"].isel({tdim: i}).values.astype(float)
    n = ds["B08"].isel({tdim: i}).values.astype(float)
    scl = np.nan_to_num(ds["SCL"].isel({tdim: i}).values.astype(float), nan=0).astype(int)
    y = ds["y"].values
    ds.close()
    ndwi = (g - n) / np.maximum(g + n, 1e-6)
    clear = np.isin(scl, CLEAR_CLASSES)
    if y[0] < y[-1]:
        ndwi, clear = ndwi[::-1], clear[::-1]
    return ndwi, clear


def mid_tide_scene(extract_path):
    """A clear scene with about half the intertidal record wet: the one
    whose wet/dry reading says the most."""
    d = np.load(extract_path, allow_pickle=True)
    Y, C, dates = d["Y"], d["C"], d["dates"]
    cf = C.mean(axis=1)
    wf = np.array([np.mean(Y[t][C[t]] > 0) if C[t].any() else np.nan for t in range(len(dates))])
    ok = cf >= 0.95
    t = int(np.nanargmin(np.where(ok, np.abs(wf - 0.5), np.nan)))
    return str(dates[t]), float(wf[t])


def crop_box(mask, margin=25):
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin, mask.shape[0] - 1)
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin, mask.shape[1] - 1)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def marea_rasters(shape):
    m = np.load(MAREA)
    keep = m["keep"]
    z = np.full(shape[0] * shape[1], np.nan); z[keep] = m["z"]
    b = np.full(shape[0] * shape[1], np.nan); b[keep] = np.where(m["band"] >= 0, m["band"], np.nan)
    return z.reshape(shape), b.reshape(shape), m


def tide_series(aoi_bbox):
    from pyintertidal.boundary import PyTMDBoundary
    lat = 0.5 * (aoi_bbox["south"] + aoi_bbox["north"]); lon = 0.5 * (aoi_bbox["west"] + aoi_bbox["east"])
    eot = PyTMDBoundary("EOT20", lat, lon, directory="tide_models")
    t = pd.date_range(EPOCH[0], EPOCH[1], freq="1h")
    return t, np.asarray(eot.levels(t), float), eot


def scalebar(ax, transform, km=1.0):
    px = km * 1000 / abs(transform.a)
    x1, y1 = ax.get_xlim()[1], ax.get_ylim()[0]
    ax.plot([x1 - px - 8, x1 - 8], [y1 - 12, y1 - 12], "-", color="white", lw=2.6)
    ax.plot([x1 - px - 8, x1 - 8], [y1 - 12, y1 - 12], "-", color=INK, lw=1.2)
    ax.text(x1 - px / 2 - 8, y1 - 18, f"{km:g} km", ha="center", va="bottom", fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc=INK, ec="none", alpha=0.7))


def main():
    import pyintertidal as pit
    from pyintertidal.hydroperiod import hydroperiod
    pit.net.use_system_certificates()
    os.makedirs(OUT, exist_ok=True)

    wf, transform, crs = read(WF_TIF)
    mask = read(MASK_TIF)[0] > 0
    shape = wf.shape
    z, bands, m = marea_rasters(shape)
    rs, cs = crop_box(mask)
    date, wfrac, cfrac = low_water_scene(EXTRACT)
    rgb, extent, ccrs, cshape = composite(CUBE, date)
    assert cshape == shape, (cshape, shape)
    date_mid, wfrac_mid = mid_tide_scene(EXTRACT)
    rgb_mid = composite(CUBE, date_mid)[0]
    ndwi, clear = scene_ndwi(date_mid)
    d = np.load(EXTRACT, allow_pickle=True)
    bbox = d["bbox"]
    bb = {"west": float(bbox[0]), "south": float(bbox[1]), "east": float(bbox[2]), "north": float(bbox[3])}
    t_h, h_h, eot = tide_series(bb)

    # ── fig_water ──────────────────────────────────────────────────────
    from pyintertidal import SentinelCube, water as W
    from pyintertidal.frequency import multiotsu_window
    aoi = pit.sites.get("villaviciosa")
    cube = SentinelCube(aoi, ("2016-01-01", "2025-12-31"), water="ndwi", cache_path=CUBE, resolution=10)
    thr, info = W.calibrate_threshold(cube, n_scenes=12, verbose=False)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), dpi=200, gridspec_kw=dict(width_ratios=[1.25, 0.9, 1.25]))
    ax = axs[0]
    smp = np.asarray(info["sample"])
    ax.hist(smp[smp <= NDWI_THRESHOLD], bins=60, range=(-0.6, NDWI_THRESHOLD), color="#b08968", label="read dry (land)")
    ax.hist(smp[smp > NDWI_THRESHOLD], bins=60, range=(NDWI_THRESHOLD, 0.6), color=TIDE, label="read wet (water)")
    ax.axvline(NDWI_THRESHOLD, color=RED, lw=1.2, label=f"threshold {NDWI_THRESHOLD:+.2f}")
    ax.text(NDWI_THRESHOLD - 0.03, 0.93, "land", transform=ax.get_xaxis_transform(), ha="right", fontsize=7, color="#7a5c3a")
    ax.text(NDWI_THRESHOLD + 0.03, 0.93, "water", transform=ax.get_xaxis_transform(), ha="left", fontsize=7, color=TIDE)
    ax.set_xlabel("NDWI of clear-sky pixels, 12 clearest scenes", fontsize=7); ax.set_ylabel("pixels", fontsize=7)
    ax.set_title("(a) the water threshold", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="center right"); ax.tick_params(labelsize=6); ax.set_yticks([])
    ax = axs[1]
    # the notebook's own rendering (viz preset: Blues, 0-1, zoom to the AOI)
    from pyintertidal import viz
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
    trans = read("products_villaviciosa/reference_map.tif")[0] == 0      # the notebook's transition class (0; 1 water, 2 land)
    lo, hi = multiotsu_window(wf, trans)
    v = wf[trans]
    ax.hist(v[v < WINDOW[0]], bins=5, range=(0, WINDOW[0]), color="#b08968")
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
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_water_villaviciosa.png", dpi=200); plt.close(fig)

    # ── fig_products ───────────────────────────────────────────────────
    # The ría runs NE-SW; every map is rotated so that its axis lies along
    # the panel, cropped to the intertidal zone, and drawn over the
    # low-water scene desaturated and lightened, so the product reads in
    # colour and the surroundings stay recognisable.
    from matplotlib.colors import LinearSegmentedColormap
    from scipy import ndimage
    ELEV = LinearSegmentedColormap.from_list("elev", ["#1d4e89", "#2a6f97", "#3fa7a3", "#8fd17a", "#e8d56b", "#d8c39a"])
    lidar = pit.reproject_to_grid(LIDAR, transform, crs, shape)
    rec = np.isfinite(z)
    lidar_c = np.where(rec, lidar - np.nanmedian(lidar[rec]) + np.nanmedian(z[rec]), np.nan)
    hydro = hydroperiod(z, h_h)
    vmin, vmax = np.nanpercentile(z[rec], [2, 98])
    rr_, cc_ = np.nonzero(mask)
    cov = np.cov(np.vstack([cc_ - cc_.mean(), rr_ - rr_.mean()]))
    w_, v_ = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v_[1, np.argmax(w_)], v_[0, np.argmax(w_)]))   # image axes: y down
    def rot(a, order):
        return ndimage.rotate(a, ang, reshape=True, order=order, cval=np.nan if a.dtype.kind == "f" else 0, prefilter=False)
    lum = 0.55 * rgb[..., 0] + 0.45 * rgb[..., 1]
    back = rot(np.dstack([lum, lum, lum]) * 0.6 + 0.4, 1)
    rgb_r = rot(rgb.astype(float), 1)
    mask_r = rot(mask.astype(np.uint8), 0) > 0
    z_r, lidar_r, hydro_r = (rot(a, 0) for a in (z, lidar_c, hydro))
    rs2, cs2 = crop_box(mask_r, margin=20)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 3.4), dpi=200)
    A = axs.ravel()
    A[0].imshow(np.clip(np.nan_to_num(rgb_r[rs2, cs2], nan=1.0), 0, 1), interpolation="nearest")
    A[0].contour(mask_r[rs2, cs2].astype(float), levels=[0.5], colors=[YEL], linewidths=0.6)
    A[0].set_title(f"(a) low water, {date}", loc="left", fontsize=8)
    for ax, arr, title, cmap, lim, lab in (
            (A[1], z_r, "(b) MAREA elevation", ELEV, (vmin, vmax), "elevation (m, tide-model datum)"),
            (A[2], lidar_r, "(c) IGN LiDAR, datum offset removed", ELEV, (vmin, vmax), "elevation (m)"),
            (A[3], hydro_r, "(d) hydroperiod from (b)", "Blues", (0, 1), "fraction of time submerged")):
        ax.imshow(np.clip(np.nan_to_num(back[rs2, cs2], nan=1.0), 0, 1), interpolation="nearest")
        im = ax.imshow(np.ma.masked_invalid(arr[rs2, cs2]), cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label(lab, fontsize=6)
    for ax in A:
        ax.set_xticks([]); ax.set_yticks([])
    scalebar(A[0], transform)
    # north arrow: the maps are rotated by `ang` degrees
    th = np.radians(-ang)
    x0, y0 = A[0].get_xlim()[0] + 25, A[0].get_ylim()[1] + 25
    A[0].annotate("N", xy=(x0 + 22 * np.sin(th), y0 - 22 * np.cos(th)), xytext=(x0, y0), fontsize=7, color="white",
                  ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color="white", lw=1.0))
    # the paper's products figure is experiments/paper_fig_products.py; this
    # rotated version is kept under another name and never overwrites it
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_products_rotated.png", dpi=200); plt.close(fig)

    # ── fig_pipeline ───────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.7), dpi=200)
    A = axs.ravel()
    T = dict(fontsize=6.5, loc="left")
    A[0].imshow(rgb_mid[rs, cs], interpolation="nearest"); A[0].set_title(f"1  a dated scene, {date_mid}", **T)
    wet = np.where(clear, ndwi > NDWI_THRESHOLD, np.nan)
    A[1].imshow(np.ma.masked_invalid(wet[rs, cs]), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    A[1].set_title(f"2  its wet/dry reading\n    (NDWI > {NDWI_THRESHOLD:g}, clear sky)", **T)
    A[2].imshow(wf[rs, cs], cmap="Blues", vmin=0, vmax=1, interpolation="nearest"); A[2].set_title("3  water frequency\n    over the archive", **T)
    A[3].imshow(np.ma.masked_invalid(np.where(mask, 1.0, np.nan)[rs, cs]), cmap="YlOrBr", vmin=0, vmax=1.6, interpolation="nearest")
    A[3].set_title("4  the intertidal zone", **T)
    # 5 boundary tide at each visit (nominal Sentinel-2 overpass, 11:21 UTC)
    dates = [str(x) for x in d["dates"].tolist()]        # plain str: pandas rejects numpy.str_
    t0 = pd.Timestamp("2024-03-01"); t1 = t0 + pd.Timedelta(days=14)
    sel = (t_h >= t0) & (t_h < t1)
    A[4].plot(t_h[sel], h_h[sel], "-", color=TIDE, lw=0.7)
    vis = pd.DatetimeIndex([pd.Timestamp(x) + pd.Timedelta(hours=11, minutes=21) for x in dates
                            if t0 <= pd.Timestamp(x) < t1])
    if len(vis):
        A[4].plot(vis, np.asarray(eot.levels(vis), float), "o", ms=3, color=RED, label="visits")
        A[4].legend(fontsize=5, loc="lower right", frameon=False)
    A[4].set_title("5  the boundary tide\n    at each visit", **T); A[4].tick_params(labelsize=5)
    A[4].set_xticks([t0, t1 - pd.Timedelta(days=1)]); A[4].set_xticklabels([t0.strftime("%d %b"), (t1 - pd.Timedelta(days=1)).strftime("%d %b %Y")])
    A[4].set_ylabel("level (m)", fontsize=6)
    # 6 bands
    nb = int(np.nanmax(bands)) + 1
    A[5].imshow(np.ma.masked_invalid(bands[rs, cs]), cmap=plt.get_cmap("viridis", nb), vmin=-0.5, vmax=nb - 0.5, interpolation="nearest")
    A[5].set_title("6  bands of distance\n    from the mouth", **T)
    # 7 clock
    centres = np.array([float(np.nanmedian(m["s_km"][m["band"] == k])) for k in range(nb)])
    tau, lo_, hi_ = m["tau_usado_min"], m["tau_lo_min"], m["tau_hi_min"]
    A[6].errorbar(centres, tau, yerr=[np.clip(tau - lo_, 0, None), np.clip(hi_ - tau, 0, None)], fmt="o", ms=3, color=TIDE, capsize=2, lw=0.8)
    A[6].axhline(0, color="0.5", lw=0.6)
    A[6].set_title("7  the interior clock,\n    one lag per band", **T); A[6].tick_params(labelsize=5)
    A[6].set_xlabel("s (km)", fontsize=6); A[6].set_ylabel("lag (min)", fontsize=6)
    # 8 elevation
    A[7].imshow(np.ma.masked_invalid(z[rs, cs]), cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    A[7].set_title("8  elevation, pixel by pixel,\n    on the corrected clock", **T)
    for i, ax in enumerate(A):
        if i not in (4, 6):
            ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout(h_pad=1.8, w_pad=1.4)
    # arrows between consecutive panels of each row
    for i in range(7):
        if i == 3:
            continue
        pa, pb = A[i].get_position(), A[i + 1].get_position()
        fig.patches.append(FancyArrowPatch((pa.x1 + 0.004, (pa.y0 + pa.y1) / 2), (pb.x0 - 0.004, (pb.y0 + pb.y1) / 2),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.8))
    fig.savefig(f"{OUT}/fig_pipeline_flat.png", dpi=200); plt.close(fig)   # the 3-D version (paper_fig_pipeline3d.py) is the paper's
    print(f"-> {OUT}/fig_water.png, fig_products.png, fig_pipeline_flat.png  (scenes {date} low water, {date_mid} mid tide; "
          f"Otsu {thr:+.3f} (clipped: {info.get('clipped')}), multi-Otsu cuts {lo:.2f}/{hi:.2f}, {nb} bands)")


if __name__ == "__main__":
    main()
