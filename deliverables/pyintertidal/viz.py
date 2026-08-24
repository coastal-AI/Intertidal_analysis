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
from .net import use_system_certificates

#: Resolution of the figures this module creates. Matplotlib's default (100)
#: looks blurry for raster maps; 150 is sharp on screen and still light.
FIGURE_DPI = 150

#: The one colour reserved for "the answer" — a fitted value, a median, the
#: line the reader should look at first.
ACCENT = "#d62728"


def _apply_aoi(array, transform, crs, aoi):
    """Return a copy with everything outside the AOI polygon set to NaN."""
    if aoi is None:
        return np.asarray(array, dtype=float)
    mask = as_aoi(aoi).raster_mask(transform, crs, array.shape)
    return np.where(mask, np.asarray(array, dtype=float), np.nan)


def _zoom_to_data(ax, array, pad=20, extent=None):
    """Tighten the axes around the finite data (frames the polygon).

    Cropping to the data is what keeps the figure at native resolution: a
    10 m product drawn across a page three times wider than its pixel count
    is being upsampled, and the interpolation reads as blur that the product
    does not have.
    """
    finite = np.isfinite(array)
    if not finite.any():
        return
    ys, xs = np.where(finite)
    ny, nx = array.shape[-2], array.shape[-1]
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, nx)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, ny)
    if extent is None:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        return
    # Map pixel indices back to the projected coordinates of `extent`.
    west, east, south, north = extent
    sx, sy = (east - west) / nx, (north - south) / ny
    ax.set_xlim(west + x0 * sx, west + x1 * sx)
    ax.set_ylim(north - y1 * sy, north - y0 * sy)


#: Fraction of the AOI's longest side left as margin around it, so the site
#: is seen in its setting rather than cropped at the polygon edge.
AOI_MARGIN = 0.12


def plot_aoi(aoi, tiles=None, basemap=True, color="#e53e3e", figsize=(10, 8),
             title=None, margin=AOI_MARGIN, source=None, dpi=FIGURE_DPI,
             ax=None):
    """Show where the study area is, over satellite imagery.

    The first thing to look at in any analysis: does the polygon actually
    cover the estuary you meant, and does it stop where you meant it to? It
    draws the **polygon outline**, never the bounding box — the bbox is a
    download detail, and showing it invites you to reason about pixels that
    the analysis will discard.

    Parameters
    ----------
    aoi : AOI | polygon | bbox dict
        The study area (see :func:`pyintertidal.aoi.as_aoi`).
    tiles : sequence of AOI, optional
        Production cells, e.g. from :meth:`pyintertidal.aoi.AOI.tile`. Drawn
        as thin outlines inside the AOI so you can check the split before
        launching one notebook per cell.
    basemap : bool
        Draw Esri World Imagery underneath. Needs ``contextily``; without it
        the outline is still drawn, on a plain background, with a note.
    margin : float
        Padding around the AOI as a fraction of its longest side. The zoom
        level of the basemap follows automatically, so a small AOI gets
        proportionally sharper imagery.
    source
        A ``contextily`` provider; defaults to Esri World Imagery, which has
        the best coastal detail of the free sources.

    Returns
    -------
    (fig, ax)
    """
    import geopandas as gpd

    aoi = as_aoi(aoi)
    WEB = "EPSG:3857"
    gdf = gpd.GeoDataFrame(geometry=[aoi.polygon],
                           crs="EPSG:4326").to_crs(WEB)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=figsize, dpi=dpi)

    west, south, east, north = gdf.total_bounds
    pad = margin * max(east - west, north - south)
    ax.set_xlim(west - pad, east + pad)
    ax.set_ylim(south - pad, north + pad)
    ax.set_aspect("equal")

    # Limits BEFORE the basemap: contextily picks its zoom level from the
    # axes extent, so setting them first is what buys the extra resolution.
    if basemap:
        try:
            import contextily as ctx

            use_system_certificates()      # the tile server is HTTPS too
            ctx.add_basemap(ax, crs=WEB,
                            source=source or ctx.providers.Esri.WorldImagery)
        except Exception as exc:
            # A missing basemap must not cost you the map: the polygon is
            # the point, the imagery is context.
            ax.set_facecolor("#eef1f4")
            reason = ("install contextily for the satellite basemap"
                      if isinstance(exc, ImportError)
                      else f"basemap unavailable ({type(exc).__name__})")
            ax.text(0.5, 0.02, reason, transform=ax.transAxes, ha="center",
                    fontsize=8, color="0.45")

    if tiles:
        cells = gpd.GeoDataFrame(
            geometry=[as_aoi(t).polygon for t in tiles],
            crs="EPSG:4326").to_crs(WEB)
        cells.boundary.plot(ax=ax, color="white", linewidth=0.9, alpha=0.85,
                            label=f"{len(cells)} production cells")
    gdf.boundary.plot(ax=ax, color=color, linewidth=2.2, label="AOI")

    ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
    ax.set_title(title or f"Area of interest — {aoi.name} "
                          f"({aoi.area_km2:.1f} km²)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def _smoothed(z, sigma_px):
    """Gaussian-smoothed copy that ignores NaN, for SHADING only.

    Illumination is a derivative, so it amplifies pixel noise: shading a raw
    per-pixel fit turns speckle into glitter and hides the landforms. The
    standard cartographic answer is to shade a smoothed surface while
    colouring the unsmoothed values, so the relief reads without any number
    on the map being altered.
    """
    if not sigma_px:
        return z
    from scipy.ndimage import gaussian_filter

    finite = np.isfinite(z)
    filled = np.where(finite, z, 0.0)
    num = gaussian_filter(filled, sigma_px)
    den = gaussian_filter(finite.astype(float), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return np.where(finite, out, np.nan)


def _context_relief(elevation, light=0.72, dark=1.0, vert_exag=2.0):
    """The surrounding land as pale grey shaded relief.

    Only the ILLUMINATION of the context matters, never its height: colouring
    it by elevation would make an estuary next to a hill read as a black
    backdrop, and would also compete with the colour bar of the map on top.
    Mapping the shading into a narrow light-grey band keeps the land present
    but visually silent.
    """
    z = np.asarray(elevation, dtype=float)
    finite = np.isfinite(z)
    if not finite.any():
        return np.zeros(z.shape + (4,))
    ls = LightSource(azdeg=315, altdeg=45)
    grey = light + (dark - light) * ls.hillshade(
        np.where(finite, z, 0.0), vert_exag=vert_exag)
    rgba = np.zeros(z.shape + (4,))
    rgba[..., :3] = grey[..., None]
    rgba[..., 3] = np.where(finite, 1.0, 0.0)
    return rgba


def _shade(z, cmap, norm, vert_exag, shade_smooth):
    """RGBA of ``z`` coloured by ``cmap`` and lit from the north-west."""
    finite = np.isfinite(z)
    ls = LightSource(azdeg=315, altdeg=45)
    relief = _smoothed(z, shade_smooth)
    rgb = ls.shade(np.ma.masked_array(relief, ~finite), cmap=cmap, norm=norm,
                   vert_exag=vert_exag, blend_mode="soft")
    # Colour from the RAW values, brightness from the smoothed relief.
    flat = cmap(norm(np.where(finite, z, norm.vmin)))
    rgb[..., :3] = 0.35 * flat[..., :3] + 0.65 * rgb[..., :3]
    rgb[..., 3] = np.where(finite, 1.0, 0.0)
    return rgb


def plot_map(
    array,
    transform=None,
    crs=None,
    aoi=None,
    cmap="viridis",
    hillshade=False,
    vert_exag=3.0,
    shade_smooth=0.0,
    context=None,
    context_range=(0.72, 1.0),
    basemap=False,
    basemap_source=None,
    interpolation="nearest",
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
    shade_smooth : float
        Standard deviation, in pixels, of the smoothing applied **only** to
        the surface used for shading. The colours still come from the raw
        values; see :func:`_smoothed`.
    context : 2-D array, optional
        A wider elevation model (typically the LiDAR) drawn underneath as a
        pale grey hillshade, so the mapped flats sit in their real landscape
        instead of floating on white. This is what makes a narrow intertidal
        fringe legible.
    context_range : (float, float)
        Grey levels the context is drawn between, 0 = black, 1 = white. The
        default keeps it light enough never to compete with the data on top.
    basemap : bool
        Draw satellite imagery underneath instead of (or as well as) a grey
        ``context``. Requires ``transform`` and ``crs``: the axes then carry
        real projected coordinates, which is what lets the tile server be
        asked for the right place and zoom.
    interpolation : str
        Passed to ``imshow``. ``"nearest"`` by default: a satellite-derived
        DEM has one value per 10 m cell and smoothing between cells invents
        detail that is not in the data. Use ``"bilinear"`` only for figures
        printed far below native resolution.
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

    # With a basemap the axes have to carry real projected coordinates, so
    # every layer is placed by `extent` instead of by pixel index.
    extent = None
    if basemap:
        if transform is None or crs is None:
            raise ValueError("basemap=True needs transform and crs")
        ny, nx = z.shape[-2], z.shape[-1]
        west, north = transform.c, transform.f
        extent = (west, west + nx * transform.a,
                  north + ny * transform.e, north)
    kw_img = dict(origin="upper", extent=extent, interpolation=interpolation)

    if context is not None:
        ax.imshow(_context_relief(context, *context_range), zorder=0, **kw_img)

    if classes is not None:
        import matplotlib.patches as mpatches
        values = sorted(classes)
        cmap_ = mcolors.ListedColormap([classes[v][1] for v in values])
        norm = mcolors.BoundaryNorm(
            [v - 0.5 for v in values] + [values[-1] + 0.5], cmap_.N)
        ax.imshow(z, cmap=cmap_, norm=norm, **kw_img)
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
            ax.imshow(_shade(z, cm, norm, vert_exag, shade_smooth),
                      zorder=2, **kw_img)
        else:
            ax.imshow(np.where(finite, z, np.nan), cmap=cm, norm=norm,
                      zorder=2, **kw_img)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=11)

    if basemap:
        _zoom_to_data(ax, z, extent=extent)
        try:
            import contextily as ctx

            use_system_certificates()
            ctx.add_basemap(
                ax, crs=str(crs), zorder=1,
                source=basemap_source or ctx.providers.Esri.WorldImagery)
        except Exception as exc:
            ax.set_facecolor("#dfe4e8")
            ax.text(0.5, 0.01, f"basemap unavailable ({type(exc).__name__})",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color="0.35")
    else:
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
    """DEM preset: viridis, hillshade, metres above the tide-model MSL.

    Two defaults exist to make a satellite-derived DEM readable rather than
    speckled, and both are cartography, not data changes:

    * ``shade_smooth=1.0`` — the relief is lit from a lightly smoothed
      surface while the colours stay raw (see :func:`_smoothed`);
    * ``robust=False`` — the colour bar spans the FULL elevation range.
      Percentile clipping is right for noisy rasters and wrong here: it
      throws away the tails, which on an intertidal DEM is exactly the
      channel floors and the upper flats.

    Pass ``context=`` a wider DEM (e.g. the LiDAR) to place the flats in a
    grey hillshaded landscape.
    """
    kw.setdefault("hillshade", True)
    kw.setdefault("shade_smooth", 1.0)
    kw.setdefault("robust", False)
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
                  ylabel="Elevation (m)", start=None, end=None,
                  basemap_of=None, transform=None, crs=None, aoi=None,
                  figsize=(15, 5), ax=None):
    """Compare profiles along a transect, next to where the line runs.

    ``profiles`` is ``{label: values}`` sharing ``distances`` (metres along
    the line, e.g. from :func:`pyintertidal.validation.transect_profile`).

    A profile you cannot locate is not much use for planning a field survey,
    so passing ``start``/``end`` (lon, lat) together with ``basemap_of`` — any
    raster on the analysis grid, normally the elevation — draws the trace on
    a map beside the profile. That panel is what catches a transect placed
    somewhere it was not meant to go.

    A series that is entirely missing is reported in the legend rather than
    drawn as an invisible line, because "the method has no data here" and
    "the method agrees with the others" must not look the same.
    """
    show_map = (start is not None and end is not None
                and basemap_of is not None and transform is not None)
    if ax is None:
        if show_map:
            fig, axes = plt.subplots(
                1, 2, figsize=figsize, dpi=FIGURE_DPI,
                gridspec_kw={"width_ratios": [2.4, 1]})
            ax, ax_map = axes
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
            ax_map = None
    else:
        fig, ax_map = ax.figure, None

    for label, values in profiles.items():
        v = np.asarray(values, dtype=float)
        n = int(np.isfinite(v).sum())
        if n == 0:
            # Keep it in the legend so the absence is visible, not silent.
            ax.plot([], [], lw=1.6, ls=":", label=f"{label} — no data here")
            continue
        pct = 100.0 * n / v.size
        tag = f"{label}" if pct > 99 else f"{label} ({pct:.0f} % of the line)"
        ax.plot(distances, v, label=tag, lw=1.8)
    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()

    if show_map:
        from pyproj import Transformer

        z = _apply_aoi(basemap_of, transform, crs, aoi) if aoi is not None \
            else np.asarray(basemap_of, dtype=float)
        finite = np.isfinite(z)
        if finite.any():
            lo, hi = np.nanpercentile(z[finite], [2, 98])
            ax_map.imshow(np.where(finite, z, np.nan), cmap="viridis",
                          vmin=lo, vmax=hi, origin="upper",
                          interpolation="nearest")
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = [], []
        for lon, lat in (start, end):
            x, y = tf.transform(lon, lat)
            col = (x - transform.c) / transform.a
            row = (y - transform.f) / transform.e
            xs.append(col); ys.append(row)
        ax_map.plot(xs, ys, color=ACCENT, lw=2.4, zorder=5)
        ax_map.scatter(xs, ys, s=45, color=ACCENT, zorder=6,
                       edgecolors="white", linewidths=1.2)
        ax_map.annotate("start", (xs[0], ys[0]), textcoords="offset points",
                        xytext=(7, 7), color=ACCENT, fontsize=9,
                        fontweight="bold")
        # Frame the LINE, not the whole study area: a 300 m transect inside a
        # 9 km AOI is a dot, and a location map that shows nothing locatable
        # is worse than none.
        half = max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0])) * 0.5 + 5
        cx, cy = np.mean(xs), np.mean(ys)
        pad = max(half * 2.2, 30)
        ax_map.set_xlim(cx - pad, cx + pad)
        ax_map.set_ylim(cy + pad, cy - pad)
        ax_map.set_xticks([]); ax_map.set_yticks([])
        ax_map.set_title(f"Where the line runs · {2 * pad * abs(transform.a) / 1000:.1f} km across",
                         fontsize=10)
    fig.tight_layout()
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


def plot_tide_record(times, heights, sampled_times=None, sampled_heights=None,
                     model_name="", location=None, figsize=(16, 5.2)):
    """The whole tidal record, what was imaged of it, and where it sits.

    Three panels over the FULL study period, because the tidal frame is a
    decadal quantity: the 18.6-year nodal cycle and the annual and semiannual
    terms all move the extremes, and a single year understates the frame the
    elevations are referenced to.

    left
        Every modelled tide height in the period — the envelope the estuary
        actually experiences, with its median and quartiles.
    centre
        Only the heights the satellite caught. Compare the two envelopes:
        the difference is what no image constrains.
    right
        Those same sampled heights as a vertical strip against the true
        extremes, so the unsampled band at each end is a distance you can
        read in metres.

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.dates as mdates

    h = np.asarray(heights, dtype=float)
    finite = np.isfinite(h)
    t = np.asarray(times)[finite]
    h = h[finite]

    fig, axes = plt.subplots(
        1, 3, figsize=figsize, dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [3.1, 3.1, 1.0]})

    site = ""
    if location is not None:
        site = f" at {location[0]:.4f}°N, {location[1]:.4f}°W"
    q25, q50, q75 = np.percentile(h, [25, 50, 75])

    def _frame(ax, tt, hh, title, colour):
        ax.scatter(tt, hh, s=7, alpha=0.45, color=colour, edgecolors="none")
        for value, style, label in ((q50, "--", "median"),
                                    (q25, ":", "P25 / P75"), (q75, ":", None)):
            ax.axhline(value, ls=style, lw=1.1,
                       color=ACCENT if label == "median" else "0.55",
                       label=label)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_ylabel("Tide height (m)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    _frame(axes[0], t, h, f"Modelled tide{site}", "#2b3f8c")

    if sampled_heights is not None:
        s = np.asarray(sampled_heights, dtype=float)
        st = (np.asarray(sampled_times) if sampled_times is not None
              else np.arange(s.size))
        ok = np.isfinite(s)
        s, st = s[ok], st[ok]
        _frame(axes[1], st, s, f"Only the {s.size} imaged dates", "#5a5ad6")

        ax = axes[2]
        rng = np.random.default_rng(0)
        ax.scatter(rng.normal(0, 0.16, s.size), s, s=26, alpha=0.75,
                   color="#7fa6d9", edgecolors="#3d5a80", linewidths=0.5)
        ax.axhline(h.max(), color="#1a6b3c", lw=2)
        ax.text(0.98, h.max(), f" {h.max():+.2f} m", ha="right", va="bottom",
                color="#1a6b3c", fontweight="bold", fontsize=9,
                transform=ax.get_yaxis_transform())
        ax.axhline(h.min(), color="#8b1a1a", lw=2)
        ax.text(0.98, h.min(), f" {h.min():+.2f} m", ha="right", va="top",
                color="#8b1a1a", fontweight="bold", fontsize=9,
                transform=ax.get_yaxis_transform())
        ax.axhline(0, color="0.35", ls="--", lw=1)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_ylabel("Tide height (m)")
        ax.set_title("Sampled vs the\ntrue extremes", fontsize=11,
                     fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        # The bands no image reaches — the ceiling on what can be mapped.
        ax.axhspan(s.max(), h.max(), color="#1a6b3c", alpha=0.07)
        ax.axhspan(h.min(), s.min(), color="#8b1a1a", alpha=0.07)
    else:
        axes[1].set_axis_off()
        axes[2].set_axis_off()

    if model_name:
        fig.suptitle(f"Tidal frame — {model_name}", fontsize=12,
                     fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


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
