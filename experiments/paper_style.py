# -*- coding: utf-8 -*-
"""Shared styling for the paper's map figures.

* a light, slightly blurred grey rendering of a scene for everything that
  is not a product, so the products (elevation, hydroperiod, readings)
  carry the colour;
* a discreet outline for the intertidal zone (thin, dark on grey, white on
  imagery), never a loud colour;
* anti-aliased rendering when a raster is shown smaller than its pixels.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

INK = "#14313c"; TIDE = "#2a6f97"; RED = "#c1121f"; MAG = "#ff4fa3"
OUTLINE_ON_GREY = INK          # thin dark line over the grey background
OUTLINE_ON_IMAGE = "white"     # thin white line over imagery
OUTLINE_LW = 0.45


def scene_bands(cube_path, date):
    """B03, B08 and SCL of one dated scene of the cube, north up."""
    ds = xr.open_dataset(cube_path)
    tdim = "t" if "t" in ds.dims else "time"
    times = np.array([str(v)[:10] for v in ds[tdim].values])
    i = int(np.where(times == date)[0][0])
    g = ds["B03"].isel({tdim: i}).values.astype(float)
    n = ds["B08"].isel({tdim: i}).values.astype(float)
    scl = np.nan_to_num(ds["SCL"].isel({tdim: i}).values.astype(float), nan=0).astype(np.int16)
    flip = bool(ds["y"].values[0] < ds["y"].values[-1])
    ds.close()
    if flip:
        g, n, scl = g[::-1], n[::-1], scl[::-1]
    return g, n, scl


def grey_background(cube_path, date, sigma=1.2, low=0.55, high=0.97):
    """A light, softly blurred greyscale of the visible green band (B03)
    of one scene: the terrain stays recognisable, the products stand out."""
    g, _, _ = scene_bands(cube_path, date)
    g = np.nan_to_num(g, nan=np.nanmedian(g))
    lo, hi = np.nanpercentile(g, [1, 99])
    v = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.6
    v = gaussian_filter(v, sigma)
    v = low + (high - low) * v
    return np.dstack([v, v, v])


def true_colour(path, gain=3.2, gamma=0.85):
    """RGB from a B04/B03/B02 GeoTIFF (reflectance x 10000), stretched."""
    import rasterio
    with rasterio.open(path) as s:
        a = s.read().astype(float) / 10000.0
    a = np.clip(a * gain, 0, 1) ** gamma
    return np.moveaxis(a, 0, -1)


def outline(ax, mask, colour=OUTLINE_ON_GREY, lw=OUTLINE_LW, **kw):
    """The boundary of a boolean mask as a thin contour."""
    return ax.contour(np.asarray(mask, float), levels=[0.5], colors=[colour], linewidths=lw, **kw)


def scale_bar(ax, shape, px_m, km=1.0, colour="white", halo=INK, fontsize=6, at=(0.04, 0.95)):
    """A scale bar of ``km`` kilometres at axes fraction ``at``."""
    import matplotlib.patheffects as pe
    n_px = km * 1000 / px_m
    x0, y0 = shape[1] * at[0], shape[0] * at[1]
    ax.plot([x0, x0 + n_px], [y0, y0], "-", color=halo, lw=3.0)
    ax.plot([x0, x0 + n_px], [y0, y0], "-", color=colour, lw=1.5)
    ax.text(x0 + n_px / 2, y0 - shape[0] * 0.02, f"{km:g} km", color=colour, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", path_effects=[pe.withStroke(linewidth=1.5, foreground=halo)])


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
