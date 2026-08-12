"""
htmlrepr.py — xarray-style HTML summaries
==========================================

Collapsible tables that describe what an object actually holds: dimensions,
coordinate ranges, dtype, memory footprint, valid-pixel counts. The same
reflex xarray gives you when you type a dataset's name in a notebook, applied
to this package's cubes and products.

Numbers here are the ones people forget to check and then misinterpret: how
many pixels are actually valid, what the value range is, how much of the grid
is empty.
"""

from __future__ import annotations

import numpy as np

_CSS = """
<style>
.pit-box{font-family:system-ui,sans-serif;font-size:12px;color:#222;
 border:1px solid #d7dce1;border-radius:6px;padding:10px 12px;margin:6px 0;
 max-width:760px;background:#fff}
.pit-box h4{margin:0 0 6px 0;font-size:12.5px}
.pit-box table{border-collapse:collapse;width:100%}
.pit-box td{padding:2px 8px 2px 0;vertical-align:top}
.pit-box td.k{color:#6b747d;white-space:nowrap;width:34%}
.pit-box td.v{font-family:ui-monospace,monospace}
.pit-box .sub{color:#6b747d;font-size:11px;margin-top:6px}
.pit-bar{display:inline-block;height:8px;border-radius:3px;background:#2b8cbe;
 vertical-align:middle}
</style>
"""


class _Html:
    """A displayable HTML fragment."""

    def __init__(self, html):
        self.html = html

    def _repr_html_(self):
        return self.html

    def __repr__(self):
        return "<pyintertidal summary — display it in a notebook>"


def _rows(pairs):
    return "".join(f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>'
                   for k, v in pairs)


def summarize_array(array, name="array", transform=None, crs=None, units=""):
    """Summary of a product raster: shape, valid pixels, range, memory.

    The "valid" figure is the one to watch — a product can look complete and
    still be 80 % NaN, and every statistic computed from it would then
    describe a fifth of the study area.
    """
    a = np.asarray(array)
    finite = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
    n_valid = int(finite.sum())
    pct = 100.0 * n_valid / max(a.size, 1)
    pairs = [
        ("shape", f"{a.shape[0]} × {a.shape[1]} ({a.size:,} px)"),
        ("dtype", str(a.dtype)),
        ("valid", f'{n_valid:,} px ({pct:.1f}%) '
                  f'<span class="pit-bar" style="width:{max(pct, 1):.0f}px"></span>'),
    ]
    if n_valid:
        vals = a[finite]
        pairs.append(("range", f"{np.nanmin(vals):.3f} … {np.nanmax(vals):.3f}"
                               f"{' ' + units if units else ''}"))
        pairs.append(("median", f"{np.nanmedian(vals):.3f}"
                                f"{' ' + units if units else ''}"))
    if transform is not None:
        px = abs(transform.a)
        pairs.append(("resolution", f"{px:g} m"))
        pairs.append(("extent", f"{a.shape[1] * px / 1000:.2f} × "
                                f"{a.shape[0] * px / 1000:.2f} km"))
        if n_valid:
            pairs.append(("valid area", f"{n_valid * px * px / 1e6:.2f} km²"))
    if crs is not None:
        pairs.append(("crs", str(crs)))
    pairs.append(("memory", f"{a.nbytes / 1e6:.1f} MB"))
    return _Html(f'{_CSS}<div class="pit-box"><h4>{name}</h4>'
                 f"<table>{_rows(pairs)}</table></div>")


def summarize_cube(cube, sample_dates=3):
    """Summary of a :class:`~pyintertidal.cube.SentinelCube`.

    Reports the archive size, the grid, the on-disk cache and the estimated
    footprint of the full cube in memory — the last number being precisely
    why the package streams instead of loading.
    """
    import os

    cached = cube.cached
    pairs = [("area of interest", f"{cube.aoi.name} · {cube.aoi.area_km2:.1f} km²"),
             ("period", f"{cube.time_extent[0]} → {cube.time_extent[1]}"),
             ("water detector", cube.water),
             ("bands", ", ".join(cube.bands))]
    if cached:
        dates = cube.dates
        shape = cube.shape
        transform, crs = cube.grid
        px = abs(transform.a)
        full_gb = len(dates) * shape[0] * shape[1] * 2 * len(cube.bands) / 1e9
        sample = ", ".join(dates[:: max(1, len(dates) // sample_dates)][:sample_dates])
        pairs += [
            ("dimensions", f"t={len(dates):,} · y={shape[0]} · x={shape[1]} "
                           f"· bands={len(cube.bands)}"),
            ("resolution", f"{px:g} m"),
            ("crs", str(crs)),
            ("first / last date", f"{dates[0]} … {dates[-1]}"),
            ("sample dates", sample),
            ("cache", f"{cube.cache_path} "
                      f"({os.path.getsize(cube.cache_path) / 1e6:.0f} MB)"),
            ("full cube in RAM", f"≈ {full_gb:.1f} GB — streamed in blocks, "
                                 f"never loaded whole"),
        ]
    else:
        pairs.append(("cache", f"{cube.cache_path} (not downloaded yet)"))
    return _Html(f'{_CSS}<div class="pit-box"><h4>SentinelCube</h4>'
                 f"<table>{_rows(pairs)}</table>"
                 f'<div class="sub">Every product streams this cube in '
                 f'time blocks, so memory stays bounded regardless of the '
                 f'archive length.</div></div>')


def summarize(obj, **kwargs):
    """Summarise a cube, an elevation result or a raster — whatever you pass."""
    from ..cube import SentinelCube

    if isinstance(obj, SentinelCube):
        return summarize_cube(obj, **kwargs)
    if hasattr(obj, "mu") and hasattr(obj, "sigma_mu"):        # ElevationResult
        return summarize_array(obj.mu, name=f"elevation ({obj.method})",
                               transform=obj.transform, crs=obj.crs, units="m")
    return summarize_array(obj, **kwargs)
