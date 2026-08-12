"""
viz.py — Composable maps and profiles
=====================================

One primitive (:func:`plot_map`) plus thin presets. Design rules:

* **The user sees the POLYGON, never the bbox.** Every plot accepts ``aoi=``
  and then masks everything outside the polygon and zooms to it — contiguous
  production tiles render seamlessly side by side.
* **Everything returns ``(fig, ax)``** so plots compose: add transects,
  markers or annotations on top, build multi-panel figures, or restyle.
* **Robust color scaling by default** (2–98 percentiles) so one outlier
  never washes out a map; override with ``vmin``/``vmax``.
* **Sharp by default.** Figures are created at :data:`FIGURE_DPI`, so maps
  are legible when you zoom or export them without configuring anything in
  the notebook. Raise it for print figures, or pass ``dpi=`` per call.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource, Normalize

from .aoi import as_aoi

#: Resolution of the figures this module creates. Matplotlib's default (100)
#: looks blurry for raster maps; 150 is sharp on screen and still light.
FIGURE_DPI = 150


def _apply_aoi(array, transform, crs, aoi):
    """Return a copy with everything outside the AOI polygon set to NaN."""
    if aoi is None:
        return np.asarray(array, dtype=float)
    mask = as_aoi(aoi).raster_mask(transform, crs, array.shape)
    return np.where(mask, np.asarray(array, dtype=float), np.nan)


def _zoom_to_data(ax, array, pad=20):
    """Tighten the axes around the finite data (frames the polygon)."""
    finite = np.isfinite(array)
    if not finite.any():
        return
    ys, xs = np.where(finite)
    ax.set_xlim(max(xs.min() - pad, 0), min(xs.max() + pad, array.shape[1]))
    ax.set_ylim(min(ys.max() + pad, array.shape[0]), max(ys.min() - pad, 0))


def plot_map(
    array,
    transform=None,
    crs=None,
    aoi=None,
    cmap="viridis",
    hillshade=False,
    vert_exag=3.0,
    vmin=None,
    vmax=None,
    robust=True,
    title=None,
    cbar_label=None,
    classes=None,
    figsize=(11, 10),
    ax=None,
):
    """The composable map primitive every preset builds on.

    Parameters
    ----------
    array : 2-D array
        Raster to draw (NaN = transparent).
    aoi : AOI | polygon | bbox, optional
        With ``transform``/``crs`` given, hides everything outside the
        polygon and zooms to it.
    hillshade : bool
        Shade by illumination (DEMs).
    classes : dict {value: (label, color)}, optional
        Draw a categorical map with a legend instead of a continuous one.
    ax : matplotlib Axes, optional
        Draw into an existing axes (multi-panel figures).

    Returns
    -------
    (fig, ax)
    """
    z = _apply_aoi(array, transform, crs, aoi) if aoi is not None \
        else np.asarray(array, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.figure

    if classes is not None:
        import matplotlib.patches as mpatches
        values = sorted(classes)
        cmap_ = mcolors.ListedColormap([classes[v][1] for v in values])
        norm = mcolors.BoundaryNorm(
            [v - 0.5 for v in values] + [values[-1] + 0.5], cmap_.N)
        ax.imshow(z, cmap=cmap_, norm=norm, interpolation="nearest")
        ax.legend(handles=[mpatches.Patch(color=classes[v][1],
                                          label=classes[v][0])
                           for v in values], loc="upper right")
    else:
        finite = np.isfinite(z)
        if vmin is None or vmax is None:
            if robust and finite.any():
                lo, hi = np.nanpercentile(z[finite], [2, 98])
            else:
                lo, hi = np.nanmin(z), np.nanmax(z)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        norm = Normalize(vmin, vmax)
        cm = plt.get_cmap(cmap).copy()
        if hillshade and finite.any():
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(np.ma.masked_array(z, ~finite), cmap=cm, norm=norm,
                           vert_exag=vert_exag, blend_mode="soft")
            rgb[..., 3] = np.where(finite, 1.0, 0.0)
            ax.imshow(rgb, origin="upper")
        else:
            ax.imshow(np.where(finite, z, np.nan), cmap=cm, norm=norm,
                      origin="upper")
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=11)

    _zoom_to_data(ax, z)
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
#  Presets — thin wrappers; anything they do, plot_map can do
# ─────────────────────────────────────────────────────────────────────────────

def plot_dem(elevation, transform=None, crs=None, aoi=None,
             title="Intertidal elevation", **kw):
    """DEM preset: viridis + hillshade, metres above the tide-model MSL."""
    kw.setdefault("hillshade", True)
    kw.setdefault("cbar_label", "Elevation (m above tide-model MSL)")
    return plot_map(elevation, transform, crs, aoi, title=title, **kw)


def plot_water_frequency(wf, transform=None, crs=None, aoi=None,
                         title="Water frequency", **kw):
    """Water-frequency preset: Blues, fixed 0–1 scale."""
    kw.setdefault("cmap", "Blues")
    kw.setdefault("vmin", 0.0); kw.setdefault("vmax", 1.0)
    kw.setdefault("robust", False)
    kw.setdefault("cbar_label", "Fraction of clear observations wet")
    return plot_map(wf, transform, crs, aoi, title=title, **kw)


def plot_reference_map(reference, transform=None, crs=None, aoi=None,
                       title="Coastal stability (reference map)", **kw):
    """Reference-map preset: transition / stable water / stable land."""
    classes = {0: ("Transition (intertidal candidate)", "#d62728"),
               1: ("Stable water", "#1f77b4"),
               2: ("Stable land", "#d2b48c")}
    return plot_map(reference, transform, crs, aoi, classes=classes,
                    title=title, **kw)


def plot_intertidal(mask, transform=None, crs=None, aoi=None,
                    title="Intertidal zone", **kw):
    """Binary intertidal-mask preset."""
    classes = {0: ("Not intertidal", "#eef3f8"), 1: ("Intertidal", "#123c69")}
    return plot_map(np.asarray(mask).astype(np.uint8), transform, crs, aoi,
                    classes=classes, title=title, **kw)


def plot_transect(distances, profiles, title="Transect",
                  ylabel="Elevation (m)", figsize=(12, 5), ax=None):
    """Compare profiles along a transect.

    ``profiles`` is ``{label: values}`` sharing ``distances`` (metres along
    the line, e.g. from :func:`pyintertidal.validation.transect_profile`).
    Ideal for field-survey planning and method comparisons.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.figure
    for label, values in profiles.items():
        ax.plot(distances, values, label=label, lw=1.6)
    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
#  Scene presets (individual dates, from pyintertidal.scenes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scene(rgb, title=None, ax=None, figsize=(9, 9)):
    """Show a true-colour scene (already percentile-stretched).

    Pair it with :func:`plot_scl` to see what the cloud classifier made of
    the same date.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.figure
    ax.imshow(np.asarray(rgb))
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


def plot_scl(scl, title="Scene classification", ax=None, figsize=(10, 9),
             classes=None):
    """Show a Scene Classification band with its official colour legend.

    Only the classes actually present are listed, so the legend stays
    readable. Class 12 is our flooded-vegetation extension
    (:mod:`pyintertidal.marsh`).
    """
    from . import water as W

    table = classes or W.SCL_CLASSES
    present = sorted(int(v) for v in np.unique(np.asarray(scl))
                     if int(v) in table)
    return plot_map(np.asarray(scl), classes={v: table[v] for v in present},
                    title=title, ax=ax, figsize=figsize)


def plot_scene_grid(images, labels=None, ncols=4, figsize=(16, 12),
                    suptitle=None):
    """Thumbnail grid of many scenes — the fastest way to eyeball an archive.

    ``images`` is a list of RGB arrays; ``None`` entries are skipped, which is
    exactly what :func:`pyintertidal.raster.read_rgb` returns for empty files.
    """
    valid = [(i, im) for i, im in enumerate(images) if im is not None]
    if not valid:
        raise ValueError("no valid images to plot")
    nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             dpi=FIGURE_DPI)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for slot, (idx, image) in enumerate(valid):
        axes[slot].imshow(image)
        if labels is not None:
            axes[slot].set_title(str(labels[idx]), fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
#  Elevation presets
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty(sigma_mu, transform=None, crs=None, aoi=None,
                     title="Elevation uncertainty", **kw):
    """Per-pixel elevation uncertainty (metres), magma scale.

    Reading it: bright areas are where the NDWI-vs-tide fit was ambiguous —
    usually few clear observations, a narrow sampled tide range, or a pixel
    whose behaviour changed during the epoch.
    """
    kw.setdefault("cmap", "magma")
    kw.setdefault("cbar_label", "σ of elevation (m)")
    return plot_map(sigma_mu, transform, crs, aoi, title=title, **kw)


def plot_subpixel_relief(sigma, transform=None, crs=None, aoi=None,
                         title="Sub-pixel relief σ", **kw):
    """Within-pixel relief measured by HSR — a roughness map at 10 m.

    High values mark pixels containing structure finer than the sensor
    (channel edges, ripple fields). Values pinned at the top of the fitting
    grid mean the fit could NOT represent the transition and must not be read
    as relief (see :func:`pyintertidal.elevation.hsr_stage2`).
    """
    kw.setdefault("cmap", "inferno")
    kw.setdefault("cbar_label", "Within-pixel relief (m)")
    return plot_map(sigma, transform, crs, aoi, title=title, **kw)


def plot_dem_3d(elevation, pixel_size=10.0, vertical_exaggeration=5.0,
                downsample=2, smooth_sigma=1.0, title="Intertidal DEM"):
    """Interactive 3-D surface of the DEM (Plotly).

    Intertidal relief is centimetric over hundreds of metres, so a true-scale
    surface looks flat: ``vertical_exaggeration`` stretches the Z axis for
    display while the labels keep real metres. Gentle NaN-aware smoothing
    removes the speckle that would otherwise dominate a 3-D view — holes stay
    holes rather than being filled by the filter.

    Returns a Plotly figure (call ``.show()``); requires ``plotly``.
    """
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter

    z0 = np.asarray(elevation, dtype=float)[::downsample, ::downsample]
    z = z0
    if smooth_sigma:
        valid = np.isfinite(z0).astype(float)
        num = gaussian_filter(np.nan_to_num(z0, nan=0.0), smooth_sigma)
        den = gaussian_filter(valid, smooth_sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(den > 1e-6, num / den, np.nan)
        z = np.where(np.isfinite(z0), z, np.nan)

    step = pixel_size * downsample
    x = np.arange(z.shape[1]) * step
    y = np.arange(z.shape[0]) * step
    fig = go.Figure(go.Surface(x=x, y=y, z=z, colorscale="Viridis",
                               colorbar=dict(title="m")))
    span = float(np.nanmax(z) - np.nanmin(z)) if np.isfinite(z).any() else 1.0
    width = max(float(x[-1]) if x.size else 1.0, 1e-6)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)",
            zaxis_title="elevation (m)", aspectmode="manual",
            aspectratio=dict(x=1, y=z.shape[0] / max(z.shape[1], 1),
                             z=max(vertical_exaggeration * span / width, 0.05)),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Tide presets
# ─────────────────────────────────────────────────────────────────────────────

def plot_tide_series(times, heights, sampled_times=None, sampled_heights=None,
                     title="Tide series", figsize=(14, 5), ax=None):
    """Continuous tide curve with the satellite acquisitions marked on it.

    Shows at a glance WHICH part of the tide the archive actually saw — the
    visual counterpart of :mod:`pyintertidal.coverage`.
    """
    import matplotlib.dates as mdates

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.figure
    ax.plot(times, heights, lw=0.6, color="#4a78c4", label="modelled tide")
    if sampled_times is not None and sampled_heights is not None:
        ax.scatter(sampled_times, sampled_heights, s=14, color="#d62728",
                   zorder=3, label="satellite acquisitions")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylabel("Tide height (m)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    return fig, ax


def plot_tide_distribution(reference_heights, sampled_heights=None,
                           title="Tidal sampling", bins=30, figsize=(6, 9),
                           ax=None):
    """Vertical histogram of the tidal frame versus what was sampled.

    Drawn vertically because the y axis IS elevation: gaps in the red bars
    are the tide levels — and therefore the elevations — that no image
    constrains.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.figure
    ref = np.asarray(reference_heights, dtype=float)
    ref = ref[np.isfinite(ref)]
    ax.hist(ref, bins=bins, orientation="horizontal", color="#c6d9f1",
            label="full tidal range")
    if sampled_heights is not None:
        s = np.asarray(sampled_heights, dtype=float)
        s = s[np.isfinite(s)]
        ax.hist(s, bins=bins, orientation="horizontal", color="#d62728",
                alpha=0.75, label="sampled by satellite")
    ax.set_ylabel("Tide height (m)")
    ax.set_xlabel("Frequency")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return fig, ax
