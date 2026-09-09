"""
ops_svg.py — What each operation does to the cube
==================================================

openEO groups datacube operations into a handful of shapes, and each one has
a canonical picture. This module draws them for OUR pipeline, so a notebook
can show, after every step, exactly what happened to the data:

``filter``
    Same dimensions, fewer labels. Discarded timesteps are drawn greyed and
    crossed out — the cloud filter is the obvious example
    (1379 dates → 305 usable).
``reduce``
    A dimension collapses. The time axis becomes a single map: this is the
    water frequency, the reference map, an elevation fit.
``apply``
    Same shape, new values. Computing NDWI from two reflectance bands, or
    thresholding a map.
``aggregate``
    Coarser grid, same extent — resampling or tiling.
``mask``
    Same grid, pixels removed. Clipping to the intertidal zone or to the AOI
    polygon.

The point is not decoration: seeing that a step reduced 1379 dates to 305, or
collapsed a time axis, is how you catch a mistake before it propagates into
the results.
"""

from __future__ import annotations

from .svgbase import Canvas, Diagram, color_for, INK, MUTED

#: Human-readable summary of what each operation kind does to the dimensions.
KIND_SEMANTICS = {
    "filter": "same dimensions, fewer labels",
    "reduce": "a dimension collapses",
    "apply": "same shape, new values",
    "aggregate": "coarser grid, same extent",
    "mask": "same grid, pixels removed",
    "generic": "",
}


def _mini_cube(c, x, y, bands, n_cols, cell, gap, colour_by_band=True,
               greyed=(), label=None, sublabel=None, coarse=False,
               masked=False):
    """Draw a compact bands × time grid at (x, y); returns its width."""
    for i, band in enumerate(bands):
        by = y + i * (cell + gap)
        colour = color_for(band) if colour_by_band else "#2b8cbe"
        c.text(x - 6, by + cell / 2 + 3, band, size=8, anchor="end",
               fill=MUTED, mono=True)
        for j in range(n_cols):
            bx = x + j * (cell + gap)
            off = j in greyed
            c.rect(bx, by, cell, cell,
                   fill="#c9ced3" if off else colour,
                   opacity=0.35 if off else 0.75)
            if coarse:                                   # aggregate: 2×2 cells
                c.line(bx + cell / 2, by, bx + cell / 2, by + cell,
                       stroke="#ffffff", width=0.8)
                c.line(bx, by + cell / 2, bx + cell, by + cell / 2,
                       stroke="#ffffff", width=0.8)
            if masked:                                   # mask: punched holes
                c.rect(bx + cell * 0.55, by + cell * 0.12, cell * 0.3,
                       cell * 0.3, fill="#ffffff", opacity=0.85)
                c.rect(bx + cell * 0.12, by + cell * 0.55, cell * 0.25,
                       cell * 0.25, fill="#ffffff", opacity=0.85)
            c.rect(bx, by, cell, cell, fill="none", stroke=INK,
                   stroke_width=0.6)
            if off:                                      # filter: crossed out
                c.line(bx + 2, by + 2, bx + cell - 2, by + cell - 2,
                       stroke="#8a9199", width=1.0)
                c.line(bx + cell - 2, by + 2, bx + 2, by + cell - 2,
                       stroke="#8a9199", width=1.0)
    width = n_cols * (cell + gap) - gap
    bottom = y + len(bands) * (cell + gap)
    if label:
        c.text(x, bottom + 2, label, size=9, weight="600")
    if sublabel:
        c.text(x, bottom + 14, sublabel, size=8, fill=MUTED, mono=True)
    return width


def draw_operation(name, kind="generic", before=None, after=None, note=None,
                   params=None, cell=26, gap=4, max_columns=6):
    """Draw a *before → operation → after* panel.

    Parameters
    ----------
    name : str
        What ran, e.g. ``"filter_dates (clouds ≤ 10 % over the transition)"``.
    kind : str
        One of :data:`KIND_SEMANTICS`; decides how the "after" cube is drawn.
    before, after : dict
        ``{"bands": [...], "n_dates": int, "shape": (rows, cols)}`` — the
        cube on each side. For ``reduce`` the "after" naturally has one
        column.
    note : str, optional
        One-line summary of the effect (``"1379 → 305 dates"``).
    params : dict, optional
        Parameters that produced this result; printed under the arrow so the
        diagram is self-documenting.

    Returns
    -------
    Diagram
    """
    before = before or {"bands": ["B03"], "n_dates": 1, "shape": (1, 1)}
    after = after or before

    b_bands = list(before["bands"])
    a_bands = list(after["bands"])
    b_cols = min(max_columns, max(int(before["n_dates"]), 1))
    a_cols = 1 if kind == "reduce" else min(max_columns,
                                            max(int(after["n_dates"]), 1))

    # For a filter, grey out the proportion of columns that were dropped.
    greyed = ()
    if kind == "filter" and before["n_dates"]:
        kept_ratio = after["n_dates"] / before["n_dates"]
        n_off = int(round(b_cols * (1 - kept_ratio)))
        greyed = tuple(range(b_cols - n_off, b_cols))

    rows = max(len(b_bands), len(a_bands))
    left = 62
    arrow_w = 172
    b_w = b_cols * (cell + gap)
    a_w = a_cols * (cell + gap)
    width = left + b_w + arrow_w + a_w + 40
    height = 54 + rows * (cell + gap) + 44
    top = 46

    c = Canvas(width, height)
    c.text(12, 18, name, size=11.5, weight="600")
    semantics = KIND_SEMANTICS.get(kind, "")
    if semantics:
        c.text(12, 32, f"{kind} — {semantics}", size=9, fill=MUTED)

    _mini_cube(c, left, top, b_bands, b_cols, cell, gap, greyed=greyed,
               label="before",
               sublabel=f"{before['n_dates']:,} × {before['shape'][0]}×"
                        f"{before['shape'][1]}")

    ax = left + b_w + 12
    mid = top + rows * (cell + gap) / 2
    c.arrow(ax, mid, ax + arrow_w - 30, mid)
    if note:
        c.text(ax + (arrow_w - 30) / 2, mid - 10, note, size=9.5,
               anchor="middle", weight="600")
    if params:
        text = " · ".join(f"{k}={v}" for k, v in list(params.items())[:3])
        c.text(ax + (arrow_w - 30) / 2, mid + 16, text, size=8, fill=MUTED,
               anchor="middle", mono=True)

    _mini_cube(c, ax + arrow_w, top, a_bands, a_cols, cell, gap,
               coarse=(kind == "aggregate"), masked=(kind == "mask"),
               label="after",
               sublabel=(f"{after['n_dates']:,} × {after['shape'][0]}×"
                         f"{after['shape'][1]}" if kind != "reduce"
                         else f"{after['shape'][0]}×{after['shape'][1]} map"))
    return Diagram(c.svg(), name)


def show_operation(name, before, after, note=None, kind="generic", **kwargs):
    """Alias of :func:`draw_operation` with the arguments in the usual order."""
    return draw_operation(name, kind=kind, before=before, after=after,
                          note=note, **kwargs)
