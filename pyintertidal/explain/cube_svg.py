"""
cube_svg.py — Draw the datacube the way openEO documentation does
==================================================================

A datacube is not an abstraction you should have to hold in your head. This
module draws it: a GRID of small rasters where **rows are bands** and
**columns are timesteps**, with every dimension labelled — the same visual
language the openEO documentation and the EOxHub notebooks use.

Two levels of fidelity:

* **schematic** — each cell is a flat colour per band. Fast, tiny, and
  enough to see the shape of the cube (how many bands, how many dates).
* **data-driven** — each cell is a real, heavily downsampled thumbnail of
  the actual imagery (see :meth:`pyintertidal.cube.SentinelCube.thumbnails`),
  so you SEE the clouds, the tide state and the missing scenes that the
  numbers only imply.

Both cap the number of columns drawn and insert an ellipsis, because a cube
with 1379 dates must still fit on a screen — and a notebook must not grow by
megabytes just to show a diagram.
"""

from __future__ import annotations

import numpy as np

from .svgbase import Canvas, Diagram, color_for, INK, MUTED


def _hex_to_rgb(colour):
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))


def _thumb_image(tile, size, x, y, canvas, color="#4a78c4"):
    """Draw one small raster as an embedded PNG.

    A thumbnail is encoded as a base64 PNG inside a single ``<image>`` rather
    than one ``<rect>`` per cell. With three bands and six dates the naive
    version emits thousands of rectangles and a half-megabyte of markup;
    this keeps a full data-driven cube diagram at a few tens of KB, which is
    what makes it safe to leave in a notebook.
    """
    import base64
    import io

    from PIL import Image

    tile = np.asarray(tile, dtype=float)
    finite = tile[np.isfinite(tile)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    span = (hi - lo) or 1.0
    norm = np.clip((tile - lo) / span, 0, 1)

    # Tint the band's colour by value; no-data stays white.
    r, g, b = _hex_to_rgb(color)
    alpha = 0.15 + 0.85 * np.nan_to_num(norm, nan=0.0)
    rgb = np.stack([255 - alpha * (255 - r),
                    255 - alpha * (255 - g),
                    255 - alpha * (255 - b)], axis=-1)
    rgb[~np.isfinite(tile)] = 255
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    canvas.raw(f'<image x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" '
               f'height="{size:.1f}" preserveAspectRatio="none" '
               f'image-rendering="pixelated" '
               f'href="data:image/png;base64,{encoded}"/>')


def draw_cube(bands, n_dates, shape, thumbnails=None, dates=None,
              title=None, max_columns=6, cell=52, resolution_m=None):
    """Draw a datacube as a bands × time grid of rasters.

    Parameters
    ----------
    bands : sequence of str
        Band names — one row each, coloured by
        :func:`pyintertidal.explain.svgbase.color_for`.
    n_dates : int
        Total number of timesteps (the label always shows the true count,
        even when only a few columns are drawn).
    shape : (rows, cols)
        Spatial size of the cube, shown on the axis labels.
    thumbnails : dict, optional
        ``{band: list of 2-D arrays}`` — real downsampled imagery, one array
        per drawn column. When omitted the cells are flat colours.
    dates : sequence of str, optional
        Labels for the drawn columns.
    max_columns : int
        How many timesteps to draw before inserting an ellipsis.

    Returns
    -------
    Diagram — renders inline in a notebook.
    """
    bands = list(bands)
    n_cols = min(int(max_columns), int(n_dates)) or 1
    gap = 10
    left, top = 74, 46 if title else 30
    grid_w = n_cols * (cell + gap)
    width = left + grid_w + 96
    height = top + len(bands) * (cell + gap) + 46

    c = Canvas(width, height)
    if title:
        c.text(12, 20, title, size=12, weight="600")

    # Column headers: the temporal dimension. Samples are spread across the
    # whole archive, so the YEAR is the informative part — showing only
    # month-day would make scenes years apart look like consecutive weeks.
    c.text(left, top - 18, "t  (time)", size=9, fill=MUTED)
    multi_year = (dates is not None
                  and len({str(d)[:4] for d in dates if d}) > 1)
    for j in range(n_cols):
        x = left + j * (cell + gap)
        label = ""
        if dates is not None and j < len(dates):
            label = str(dates[j])[:10] if multi_year else str(dates[j])[5:]
        elif n_dates:
            label = f"t{j}"
        c.text(x + cell / 2, top - 6, label, size=6.8 if multi_year else 7.5,
               fill=MUTED, anchor="middle", mono=True)

    # One row per band.
    for i, band in enumerate(bands):
        y = top + i * (cell + gap)
        colour = color_for(band)
        c.text(left - 10, y + cell / 2 + 3, band, size=10, anchor="end",
               weight="600", mono=True)
        for j in range(n_cols):
            x = left + j * (cell + gap)
            tile = None
            if thumbnails and band in thumbnails and j < len(thumbnails[band]):
                tile = thumbnails[band][j]
            if tile is not None:
                _thumb_image(tile, cell, x, y, c, color=colour)
            else:
                c.rect(x, y, cell, cell, fill=colour, opacity=0.55)
            c.rect(x, y, cell, cell, fill="none", stroke=INK, stroke_width=0.7)

    # Ellipsis + total count when the time axis is truncated.
    if n_dates > n_cols:
        x = left + grid_w + 4
        mid = top + (len(bands) * (cell + gap)) / 2
        c.text(x + 10, mid, "…", size=20, fill=MUTED, anchor="middle")
        c.text(x + 10, mid + 18, f"{n_dates:,}", size=8.5, fill=MUTED,
               anchor="middle", mono=True)
        c.text(x + 10, mid + 29, "dates", size=8, fill=MUTED, anchor="middle")

    # Spatial dimensions, annotated on the first cell.
    y_last = top + len(bands) * (cell + gap)
    res = f" @ {resolution_m:.0f} m" if resolution_m else ""
    c.text(left, y_last + 14, f"x × y = {shape[1]} × {shape[0]} px{res}",
           size=9, fill=MUTED, mono=True)
    c.text(left, y_last + 27,
           f"{len(bands)} bands × {n_dates:,} dates × "
           f"{shape[0] * shape[1]:,} pixels", size=9, fill=MUTED)
    return Diagram(c.svg(), title or "datacube")


def show_cube(bands, n_dates, shape, **kwargs):
    """Schematic view of a cube (alias of :func:`draw_cube`)."""
    return draw_cube(bands, n_dates, shape, **kwargs)


def describe_cube(cube, with_data=True, max_columns=6, thumb_px=18):
    """Draw a real :class:`~pyintertidal.cube.SentinelCube`.

    With ``with_data=True`` the diagram shows actual downsampled imagery
    (clouds and all) sampled evenly across the archive; set it to False for a
    schematic view that costs no IO.
    """
    thumbs, dates = None, None
    if with_data:
        try:
            thumbs, dates = cube.thumbnails(n_dates=max_columns, size=thumb_px)
        except Exception:
            thumbs, dates = None, None
    return draw_cube(
        cube.bands, len(cube.dates), cube.shape, thumbnails=thumbs,
        dates=dates, max_columns=max_columns,
        resolution_m=abs(cube.grid[0].a) if cube.cached else None,
        title=(f"{cube.aoi.name} · {cube.time_extent[0]} → "
               f"{cube.time_extent[1]} · water={cube.water}"),
    )
