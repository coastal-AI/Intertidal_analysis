"""
htmlrepr.py — xarray-style summaries of cubes and products
===========================================================

The repr xarray and dask give you when you type an object's name: a header
with the dimensions, a table of sizes, the **proportional array box** showing
how the data is shaped and chunked, and collapsible sections for coordinates
and attributes.

Why bother reproducing it: the numbers in that panel are the ones people skip
and then misread. How much data is this really? How is it cut up, and will
that fit in memory? How many pixels are actually valid, or is the product
80 % NaN? Printing them next to a picture of the array makes those questions
answer themselves — which is the same reason the streaming design exists at
all (see :class:`pyintertidal.cube.SentinelCube`).

Everything is inline HTML and SVG with an explicit light panel, so it renders
identically in a light or dark notebook theme and survives being exported.
"""

from __future__ import annotations

import numpy as np

from .arraybox import draw_array_box, chunk_count, human_bytes

_CSS = """
<style>
.pit-repr{font-family:system-ui,-apple-system,sans-serif;font-size:12px;
 color:#1c1f23;background:#fff;border:1px solid #d7dce1;border-radius:6px;
 padding:10px 12px;margin:6px 0;max-width:900px}
.pit-repr .hdr{font-size:12.5px;margin-bottom:8px}
.pit-repr .hdr b{font-weight:600}
.pit-repr .dims{color:#1c1f23}
.pit-repr .dims i{color:#6b747d;font-style:normal}
.pit-repr .body{display:flex;align-items:flex-start;gap:18px;flex-wrap:wrap}
.pit-repr table.sizes{border-collapse:collapse;font-size:12px}
.pit-repr table.sizes th{text-align:center;font-weight:600;padding:3px 14px;
 border-bottom:1px solid #d7dce1}
.pit-repr table.sizes td{padding:4px 14px;border-bottom:1px solid #eef1f4}
.pit-repr table.sizes td.k{font-weight:600;text-align:right;color:#39424b}
.pit-repr table.sizes td.v{font-family:ui-monospace,monospace;text-align:center}
.pit-repr table.sizes tr:nth-child(even){background:#f7f9fa}
.pit-repr details{margin-top:8px}
.pit-repr summary{cursor:pointer;color:#39424b;font-weight:600;padding:3px 0}
.pit-repr summary span{color:#6b747d;font-weight:400}
.pit-repr table.coords{border-collapse:collapse;width:100%;font-size:11.5px;
 margin-top:4px}
.pit-repr table.coords td{padding:2px 10px 2px 0;border-bottom:1px solid #f2f4f6}
.pit-repr table.coords td.n{font-weight:600;white-space:nowrap}
.pit-repr table.coords td.d{color:#6b747d;font-family:ui-monospace,monospace;
 white-space:nowrap}
.pit-repr table.coords td.val{font-family:ui-monospace,monospace;color:#39424b}
.pit-repr .note{color:#6b747d;font-size:11.5px;margin-top:8px;line-height:1.45}
</style>
"""

_DB_ICON = ("<svg width='17' height='17' viewBox='0 0 24 24' fill='none' "
            "stroke='#39424b' stroke-width='1.7'><ellipse cx='12' cy='5' "
            "rx='8' ry='3'/><path d='M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5'/>"
            "<path d='M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3'/></svg>")


class _Html:
    """A displayable HTML fragment."""

    def __init__(self, html):
        self.html = html

    def _repr_html_(self):
        return self.html

    def save(self, path):
        """Write the panel to a standalone ``.html`` file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"<!doctype html><meta charset='utf-8'>{self.html}")
        return path

    def __repr__(self):
        return "<pyintertidal summary — display it in a notebook>"


def _sizes_table(rows):
    head = ("<tr><th></th><th>Array</th><th>Chunk</th></tr>"
            if any(len(r) == 3 for r in rows) else "")
    body = []
    for row in rows:
        if len(row) == 3:
            body.append(f"<tr><td class='k'>{row[0]}</td>"
                        f"<td class='v'>{row[1]}</td>"
                        f"<td class='v'>{row[2]}</td></tr>")
        else:
            body.append(f"<tr><td class='k'>{row[0]}</td>"
                        f"<td class='v' colspan='2'>{row[1]}</td></tr>")
    return f"<table class='sizes'>{head}{''.join(body)}</table>"


def _section(title, count, inner, open_=False):
    return (f"<details{' open' if open_ else ''}><summary>{title} "
            f"<span>({count})</span></summary>{inner}</details>")


def _coords_table(rows):
    return "<table class='coords'>" + "".join(
        f"<tr><td class='n'>{n}</td><td class='d'>{d}</td>"
        f"<td class='val'>{v}</td></tr>" for n, d, v in rows) + "</table>"


def _preview(values, limit=3):
    """First and last few values, xarray style: ``a b c ... x y z``."""
    values = list(values)
    if len(values) <= 2 * limit:
        return " ".join(str(v) for v in values)
    head = " ".join(str(v) for v in values[:limit])
    tail = " ".join(str(v) for v in values[-limit:])
    return f"{head} ... {tail}"


# ─────────────────────────────────────────────────────────────────────────────
#  Cube
# ─────────────────────────────────────────────────────────────────────────────

def storage_chunks(path, variable=None):
    """The chunk shape a netCDF file is actually written in, or None.

    Read straight from the file's HDF5 layout, so the grid drawn on the box
    is the real one — the tiles a reader has to pull off disk — rather than a
    decorative guess.
    """
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            for name, var in ds.data_vars.items():
                if variable and name != variable:
                    continue
                sizes = var.encoding.get("chunksizes")
                if sizes and len(sizes) == var.ndim and var.ndim >= 2:
                    return tuple(int(s) for s in sizes)
    except Exception:
        pass
    return None


def summarize_cube(cube, chunk=None, chunks=None):
    """xarray-style panel for a :class:`~pyintertidal.cube.SentinelCube`.

    Shows the full cube next to the box that represents it, and — the point
    of the exercise — contrasts the size of the WHOLE cube against the size
    of the block actually held in memory at any moment.

    The grid on the box is the cube's REAL storage layout, read from the
    netCDF with :func:`storage_chunks`; ``chunks`` overrides it. The time
    blocks a streaming pass reads are a separate thing, and appear in the
    table as ``Streaming``.
    """
    import os

    if not cube.cached:
        return _Html(f"{_CSS}<div class='pit-repr'><div class='hdr'>"
                     f"<b>SentinelCube</b> {cube.aoi.name} — not downloaded "
                     f"yet<br><span class='note'>cache: "
                     f"{cube.cache_path}</span></div></div>")

    from ..cube import auto_chunk

    dates = cube.dates
    ny, nx = cube.shape
    nb = len(cube.bands)
    nt = len(dates)
    transform, crs = cube.grid
    px = abs(transform.a)

    block = auto_chunk(ny * nx, chunk)
    layout = tuple(chunks) if chunks else (
        storage_chunks(cube.cache_path) or (block, ny, nx))
    itemsize = 2                                    # int16 as stored
    total_bytes = nb * nt * ny * nx * itemsize
    chunk_bytes = nb * block * ny * nx * itemsize
    n_chunks = chunk_count((nt, ny, nx), layout) * nb

    box = draw_array_box((nt, ny, nx), chunks=layout,
                         dim_names=("t", "y", "x"), size=180)

    header = (f"<div class='hdr'><b>pyintertidal.SentinelCube</b> "
              f"<span class='dims'>'{cube.aoi.name}' (<i>band:</i> {nb}, "
              f"<i>time:</i> {nt:,}, <i>y:</i> {ny:,}, "
              f"<i>x:</i> {nx:,})</span></div>")

    table = _sizes_table([
        ("Bytes", human_bytes(total_bytes),
         human_bytes(itemsize * int(np.prod(layout)))),
        ("Shape", f"({nb}, {nt:,}, {ny:,}, {nx:,})",
         f"({nb}, {', '.join(f'{s:,}' for s in layout)})"),
        ("Chunks", f"{n_chunks:,} on disk", "×".join(str(s) for s in layout)),
        ("Streaming", f"{human_bytes(chunk_bytes)} in RAM",
         f"{block} dates/pass"),
        ("Data type", "int16 numpy.ndarray", ""),
    ])

    coords = _coords_table([
        ("band", "(band) &lt;U3", " ".join(f"'{b}'" for b in cube.bands)),
        ("t", f"(time) datetime64", _preview(dates)),
        ("y", "(y) float64", f"{transform.f:,.0f} … "
                             f"{transform.f - ny * px:,.0f}"),
        ("x", "(x) float64", f"{transform.c:,.0f} … "
                             f"{transform.c + nx * px:,.0f}"),
    ])
    attrs = _coords_table([
        ("crs", "", str(crs)),
        ("resolution", "", f"{px:g} m"),
        ("transform", "", f"| {transform.a:.2f}, {transform.b:.2f}, "
                          f"{transform.c:.2f} |"),
        ("area_of_interest", "", f"{cube.aoi.name} — "
                                 f"{cube.aoi.area_km2:.1f} km²"),
        ("water_detector", "", cube.water),
        ("cache", "", f"{cube.cache_path} "
                      f"({os.path.getsize(cube.cache_path) / 1e6:,.0f} MB on disk)"),
    ])

    note = (f"The whole cube is <b>{human_bytes(total_bytes)}</b>, far more "
            f"than fits in memory. It is never loaded: every product streams "
            f"it in blocks of <b>{human_bytes(chunk_bytes)}</b>, which is why "
            f"the length of the archive costs time rather than RAM.")

    return _Html(f"{_CSS}<div class='pit-repr'>{header}"
                 f"<div class='body'>{_DB_ICON}{table}{box.svg}</div>"
                 f"{_section('Coordinates', 4, coords, open_=True)}"
                 f"{_section('Attributes', 6, attrs)}"
                 f"<div class='note'>{note}</div></div>")


# ─────────────────────────────────────────────────────────────────────────────
#  Products
# ─────────────────────────────────────────────────────────────────────────────

def summarize_array(array, name="array", transform=None, crs=None, units="",
                    long_name="", chunks=None):
    """xarray-style panel for a product raster.

    The number to read first is **valid**: a product can look complete and
    still be mostly NaN, and every statistic drawn from it would then
    describe a fraction of the study area.

    ``chunks`` is the granularity the raster was actually produced in — for
    a fitted product, the row block of the fitter. Passing it draws the grid
    on the box; leaving it None draws a single-chunk box, which is what an
    array computed in one go really is.
    """
    a = np.asarray(array)
    ny, nx = a.shape[-2], a.shape[-1]
    finite = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
    n_valid = int(finite.sum())
    pct = 100.0 * n_valid / max(a.size, 1)

    chunks = tuple(chunks) if chunks else a.shape
    box = draw_array_box(a.shape, chunks=chunks,
                         dim_names=("y", "x") if a.ndim == 2
                         else tuple("d" * (a.ndim - 2)) + ("y", "x"),
                         size=170)

    dims = ", ".join(f"<i>{n}:</i> {s:,}" for n, s in
                     zip(("y", "x") if a.ndim == 2
                         else [f"dim_{i}" for i in range(a.ndim - 2)] + ["y", "x"],
                         a.shape))
    header = (f"<div class='hdr'><b>pyintertidal.Product</b> "
              f"<span class='dims'>'{name}' ({dims})</span></div>")

    tiled = chunks != a.shape
    n_chunks = chunk_count(a.shape, chunks)
    chunk_bytes = a.itemsize * int(np.prod(chunks))
    rows = [("Bytes", human_bytes(a.nbytes),
             human_bytes(chunk_bytes) if tiled else ""),
            ("Shape", f"({', '.join(f'{s:,}' for s in a.shape)})",
             f"({', '.join(f'{s:,}' for s in chunks)})" if tiled else ""),
            ("Valid", f"{n_valid:,} px ({pct:.1f} %)", ""),
            ("Data type", f"{a.dtype} numpy.ndarray", "")]
    if tiled:
        rows.insert(2, ("Blocks", f"{n_chunks:,} blocks",
                        f"{chunks[-2]} rows"))
    if n_valid:
        vals = a[finite]
        rows.insert(3, ("Range", f"{np.nanmin(vals):.3f} … "
                                 f"{np.nanmax(vals):.3f} {units}".strip(), ""))
    table = _sizes_table(rows)

    attr_rows = []
    if long_name:
        attr_rows.append(("long_name", "", long_name))
    if units:
        attr_rows.append(("units", "", units))
    if transform is not None:
        px = abs(transform.a)
        attr_rows += [
            ("resolution", "", f"{px:g} m"),
            ("extent", "", f"{nx * px / 1000:.2f} × {ny * px / 1000:.2f} km"),
            ("valid_area", "", f"{n_valid * px * px / 1e6:.2f} km²"),
        ]
    if crs is not None:
        attr_rows.append(("crs", "", str(crs)))
    attrs = _coords_table(attr_rows) if attr_rows else ""

    body = (f"<div class='body'>{_DB_ICON}{table}{box.svg}</div>"
            + (_section("Attributes", len(attr_rows), attrs) if attrs else ""))
    return _Html(f"{_CSS}<div class='pit-repr'>{header}{body}</div>")


def summarize(obj, **kwargs):
    """Summarise a cube, an elevation result or a raster — whatever you pass."""
    from ..cube import SentinelCube

    if isinstance(obj, SentinelCube):
        return summarize_cube(obj, **kwargs)
    if hasattr(obj, "mu") and hasattr(obj, "sigma_mu"):        # ElevationResult
        kwargs.setdefault("chunks", (32, obj.mu.shape[-1]))
        return summarize_array(
            obj.mu, name=f"elevation ({obj.method}"
                         f"{', ' + obj.epoch if obj.epoch else ''})",
            transform=obj.transform, crs=obj.crs, units="m",
            long_name="Intertidal elevation above the tide-model datum",
            **kwargs)
    return summarize_array(obj, **kwargs)
