"""
svgbase.py — Shared drawing primitives for the datacube diagrams
================================================================

Small helpers used by every diagram in :mod:`pyintertidal.explain`. Pure
SVG strings: no plotting library, no external assets, nothing to install —
the diagrams are just markup that any notebook or browser renders.

The visual language follows the openEO documentation: a datacube is drawn as
a GRID of small rasters, rows being bands and columns being timesteps, with
the dimensions labelled around it.
"""

from __future__ import annotations

#: Colour per well-known band or product, so the same quantity always looks
#: the same across every diagram in a notebook.
PALETTE = {
    "B02": "#4a78c4", "B03": "#4aa46b", "B04": "#c44a4a", "B08": "#8a4ac4",
    "B11": "#c4934a", "B12": "#8a6b4a", "SCL": "#9aa4ad",
    "NDWI": "#2b8cbe", "MNDWI": "#2b8cbe", "AWEI": "#2b8cbe",
    "water frequency": "#2b8cbe", "reference map": "#d62728",
    "intertidal": "#123c69", "elevation": "#3b7d4f", "uncertainty": "#a05fb4",
}
DEFAULT_COLOR = "#7f8c8d"
INK = "#333333"
MUTED = "#8a9199"


def color_for(name):
    """Colour of a band or product name (case-insensitive, with fallback)."""
    if name in PALETTE:
        return PALETTE[name]
    lower = str(name).lower()
    for key, value in PALETTE.items():
        if key.lower() in lower:
            return value
    return DEFAULT_COLOR


class Canvas:
    """A tiny SVG accumulator with a fluent API.

    Only what the diagrams need: rectangles, lines, text, arrows and groups.
    Keeping it minimal means the generated markup stays small enough to sit
    inside a notebook without bloating it.
    """

    def __init__(self, width, height, font="system-ui, sans-serif"):
        self.width = width
        self.height = height
        self.font = font
        self.parts = []

    def rect(self, x, y, w, h, fill="none", stroke=None, opacity=1.0,
             stroke_width=0.8, rx=0):
        """A rectangle — the building block of every cube cell."""
        s = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
             f'fill="{fill}" opacity="{opacity:.2f}" rx="{rx}"')
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.parts.append(s + "/>")
        return self

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.0, dash=None):
        """A straight line (``dash`` e.g. "4 2" for dashed)."""
        s = (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
             f'stroke="{stroke}" stroke-width="{width}"')
        if dash:
            s += f' stroke-dasharray="{dash}"'
        self.parts.append(s + "/>")
        return self

    def text(self, x, y, content, size=10, fill=INK, anchor="start",
             weight="normal", mono=False, opacity=1.0):
        """A text label; ``mono=True`` for values and identifiers."""
        family = "ui-monospace, monospace" if mono else self.font
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-family="{family}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" opacity="{opacity:.2f}">'
            f'{escape(str(content))}</text>')
        return self

    def arrow(self, x1, y1, x2, y2, stroke=INK, width=1.6):
        """A line ending in an arrowhead (flow between two things)."""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" '
            f'marker-end="url(#pit-arrow)"/>')
        return self

    def group(self, dx=0, dy=0):
        """Context manager translating everything drawn inside it."""
        return _Group(self, dx, dy)

    def raw(self, markup):
        """Append raw SVG markup, for anything this API does not cover."""
        self.parts.append(markup)
        return self

    def svg(self):
        """The finished SVG document as a string."""
        defs = ('<defs><marker id="pit-arrow" markerWidth="9" markerHeight="7" '
                f'refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, '
                f'0 7" fill="{INK}"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{self.width:.0f}" height="{self.height:.0f}" '
                f'viewBox="0 0 {self.width:.0f} {self.height:.0f}">'
                + defs + "".join(self.parts) + "</svg>")


class _Group:
    def __init__(self, canvas, dx, dy):
        self.canvas = canvas
        self.dx, self.dy = dx, dy
        self.start = 0

    def __enter__(self):
        self.start = len(self.canvas.parts)
        return self.canvas

    def __exit__(self, *exc):
        inner = "".join(self.canvas.parts[self.start:])
        del self.canvas.parts[self.start:]
        self.canvas.parts.append(
            f'<g transform="translate({self.dx:.1f},{self.dy:.1f})">'
            f'{inner}</g>')
        return False


def escape(text):
    """Escape the characters that would break the SVG markup."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


class Diagram:
    """A displayable SVG diagram (renders inline in Jupyter)."""

    def __init__(self, svg, title=""):
        self.svg = svg
        self.title = title

    def _repr_html_(self):
        return self.svg

    def save(self, path):
        """Write the diagram to an ``.svg`` file (for papers and reports)."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.svg)
        return path

    def __repr__(self):
        return f"<pyintertidal diagram{': ' + self.title if self.title else ''} "\
               f"— display it in a notebook or call .save(path)>"
