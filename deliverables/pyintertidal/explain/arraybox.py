"""
arraybox.py — The proportional array box (dask/xarray style)
============================================================

The isometric box that dask and xarray draw for an array: a 3-D wireframe
whose **proportions follow the real shape** and whose grid lines mark the
**chunk boundaries**. One glance tells you what a plain shape tuple does not:

* whether the array is a tall stack of small images or a wide time series,
* how it is cut into chunks, and therefore how it will be processed,
* whether the chunking is sane — one enormous chunk, or thousands of tiny
  ones, are both visible immediately.

Proportions use dask's damped scaling: real Earth-observation arrays have
dimensions that differ by three orders of magnitude (1379 dates × 915 × 915
pixels), and a linear box would be an unreadable sliver. The response curve
compresses extreme ratios while keeping small ones faithful, so the drawing
stays informative without lying about which dimension is bigger.
"""

from __future__ import annotations

import math

from .svgbase import Canvas, Diagram, INK, MUTED

#: Face fill of the box (dask's sand colour).
FACE = "#ECB172"
EDGE = "#000000"


#: Longest:shortest ratio the drawing is allowed to show. Beyond this a box
#: degenerates into a line and stops communicating anything.
MAX_DRAWN_RATIO = 6.0


def _ratio_response(x, damping=0.5):
    """Damp a dimension ratio so the box stays drawable but stays honest.

    Real cubes have dimensions three orders of magnitude apart, and drawing
    them to scale would produce a sliver. The drawn ratio is therefore
    ``true_ratio ** damping``, capped at :data:`MAX_DRAWN_RATIO`:

    ==========  ===========
    true ratio  drawn ratio
    ==========  ===========
    1           1.0
    2           1.4
    10          3.2
    100         6.0 (capped)
    1000        6.0 (capped)
    ==========  ===========

    Monotonic, so a bigger dimension always looks bigger; compressed, so
    nothing collapses. The exact sizes are printed next to the box anyway —
    the drawing is there to convey the SHAPE.
    """
    x = max(float(x), 1.0)
    return min(x ** damping, MAX_DRAWN_RATIO)


def _sizes(shape, size=200):
    """Pixel length of each dimension, proportionally damped."""
    biggest = max(shape) if shape else 1
    return tuple(size / _ratio_response(biggest / max(0.1, d)) for d in shape)


def _boundaries(length, chunk):
    """Fractional positions (0–1) of the chunk boundaries along an axis."""
    if not chunk or chunk >= length:
        return []
    return [i / length for i in range(chunk, int(length), int(chunk))]


def draw_array_box(shape, chunks=None, dim_names=None, size=190,
                   max_lines=24, title=None, background=True):
    """Draw an array as a proportional isometric box with chunk lines.

    Parameters
    ----------
    shape : tuple
        Array shape. 2-D draws a flat rectangle; 3-D and above draw a box
        using the LAST TWO dimensions as the visible face and the first as
        depth (the convention dask uses, and the one that matches how a
        satellite cube is stacked: time in depth, y/x on the face).
    chunks : tuple, optional
        Chunk size per dimension; grid lines are drawn at those boundaries.
    dim_names : tuple, optional
        Names shown next to each axis (e.g. ``("t", "y", "x")``).
    max_lines : int
        Cap on grid lines per axis. Thousands of chunks would turn the box
        solid black; beyond the cap the lines are thinned and the true count
        still appears in the table next to the drawing.

    Returns
    -------
    Diagram
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        shape = (1,) + shape
    chunks = tuple(chunks) if chunks else tuple(shape)
    dim_names = tuple(dim_names) if dim_names else tuple(
        f"dim_{i}" for i in range(len(shape)))

    # Face = last two dims; depth = the product of the leading ones.
    depth_dims = shape[:-2] or (1,)
    ny, nx = shape[-2], shape[-1]
    nz = 1
    for d in depth_dims:
        nz *= d

    wx, wy, wz = _sizes((nx, ny, nz), size=size)
    # The depth axis is drawn at 45°, so it eats width AND height twice over;
    # past this cap the box stops fitting next to the table. Depth is the one
    # axis whose drawn length may therefore understate its true rank — the
    # count printed along it is always the real one.
    wz = min(wz, size * 0.7)

    pad = 34
    width = pad + wx + wz + 58
    height = pad + wy + wz + 46
    if title:
        height += 16
    top = pad + (16 if title else 0)

    c = Canvas(width, height, background="#ffffff" if background else None)
    if title:
        c.text(10, 16, title, size=11, weight="600")

    x0, y0 = pad, top + wz            # front-face upper-left corner

    # ── faces ────────────────────────────────────────────────────────────
    # top: slides back-right by wz; side: the right flank.
    c.raw(f'<polygon points="{x0},{y0} {x0 + wz},{y0 - wz} '
          f'{x0 + wz + wx},{y0 - wz} {x0 + wx},{y0}" fill="{FACE}" '
          f'opacity="0.75" stroke="{EDGE}" stroke-width="1"/>')
    c.raw(f'<polygon points="{x0 + wx},{y0} {x0 + wx + wz},{y0 - wz} '
          f'{x0 + wx + wz},{y0 - wz + wy} {x0 + wx},{y0 + wy}" fill="{FACE}" '
          f'opacity="0.62" stroke="{EDGE}" stroke-width="1"/>')
    c.rect(x0, y0, wx, wy, fill=FACE, opacity=0.95, stroke=EDGE,
           stroke_width=1)

    # ── chunk grid lines ─────────────────────────────────────────────────
    def thin(values):
        """Keep at most ``max_lines`` of the real boundaries, evenly spaced."""
        if len(values) <= max_lines:
            return values
        step = math.ceil(len(values) / max_lines)
        return values[::step]

    for fx in thin(_boundaries(nx, chunks[-1])):          # cuts across x
        x = x0 + fx * wx
        c.line(x, y0, x, y0 + wy, stroke=EDGE, width=0.5)
        c.line(x, y0, x + wz, y0 - wz, stroke=EDGE, width=0.5)
    for fy in thin(_boundaries(ny, chunks[-2])):          # cuts across y
        y = y0 + fy * wy
        c.line(x0, y, x0 + wx, y, stroke=EDGE, width=0.5)
        c.line(x0 + wx, y, x0 + wx + wz, y - wz, stroke=EDGE, width=0.5)
    depth_chunk = chunks[0] if len(chunks) == len(shape) and len(shape) > 2 else nz
    for fz in thin(_boundaries(nz, depth_chunk)):         # cuts across depth
        dx, dy = fz * wz, -fz * wz
        c.line(x0 + dx, y0 + dy, x0 + dx + wx, y0 + dy, stroke=EDGE, width=0.5)
        c.line(x0 + wx + dx, y0 + dy, x0 + wx + dx, y0 + dy + wy,
               stroke=EDGE, width=0.5)

    # ── dimension labels, on their own axis ──────────────────────────────
    c.text(x0 + wx / 2, y0 + wy + 15, f"{nx:,}", size=10, anchor="middle",
           mono=True)
    c.text(x0 + wx / 2, y0 + wy + 26, dim_names[-1], size=8.5, fill=MUTED,
           anchor="middle")
    label_x = x0 + wx + wz + 12
    label_y = y0 - wz + wy / 2
    c.raw(f'<text x="{label_x}" y="{label_y}" font-size="10" '
          f'font-family="ui-monospace, monospace" fill="{INK}" '
          f'transform="rotate(-90 {label_x} {label_y})" '
          f'text-anchor="middle">{ny:,}</text>')
    c.raw(f'<text x="{label_x + 11}" y="{label_y}" font-size="8.5" '
          f'font-family="system-ui, sans-serif" fill="{MUTED}" '
          f'transform="rotate(-90 {label_x + 11} {label_y})" '
          f'text-anchor="middle">{dim_names[-2]}</text>')
    # Depth label rides the VISIBLE depth edge — the top-left one, running
    # up-right from the front face's upper-left corner. Hanging it off the
    # bottom-left corner instead would put it on an edge hidden by the box.
    zx, zy = x0 + wz / 2 - 8, y0 - wz / 2 - 6
    c.raw(f'<text x="{zx}" y="{zy}" font-size="10" '
          f'font-family="ui-monospace, monospace" fill="{INK}" '
          f'transform="rotate(-45 {zx} {zy})" text-anchor="middle">'
          f'{nz:,}</text>')
    if len(shape) > 2:
        c.raw(f'<text x="{zx - 8}" y="{zy - 9}" font-size="8.5" '
              f'font-family="system-ui, sans-serif" fill="{MUTED}" '
              f'transform="rotate(-45 {zx - 8} {zy - 9})" '
              f'text-anchor="middle">{" × ".join(dim_names[:-2])}</text>')
    return Diagram(c.svg(), title or "array")


def chunk_count(shape, chunks):
    """How many chunks the array is cut into."""
    total = 1
    for length, chunk in zip(shape, chunks):
        total *= max(1, math.ceil(length / max(chunk, 1)))
    return total


def human_bytes(n):
    """Bytes as a human-readable string (KiB/MiB/GiB/TiB)."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.2f} TiB"
