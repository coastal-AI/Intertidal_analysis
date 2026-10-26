# -*- coding: utf-8 -*-
"""CARTOGRASS 2026 poster, Materials and methods -- the study-site map.

Left: the Atlantic coast of Europe from NW Spain to the German Bight with the
five sites numbered. Right: one small map per site, numbered the same way,
with its intertidal flats and the tide gauges that stand inside the estuary
(they validate, they never feed the method). Poster palette throughout:
water in a pale deep-blue tint, flats in sand, gauges in teal.

Layout: the site maps sit in two rows with the same 9 mm gap between
neighbours. Each map's window is the product's own extent (the Wadden and
Westerschelde windows are the whole raster), so the maps cannot be widened;
instead the lower row is drawn as tall as fits, it fixes the width of the
map block, and the overview takes the rest of the width (its view is
widened east-west to match, never cropped).

Run from the repository root (one process, about 0.4 GB, ~30 s):

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/fig_sites_map.py
        [--width 391] [--height 126]

Inputs (read only)
    Natural Earth 1:10m land, lakes and land boundaries (public domain),
        looked up in poster/tools/cg/cache/ then data_v4/naturalearth/;
        whatever is missing is downloaded once into the git-ignored cache
    products_<site>/water_frequency.tif, intertidal_mask.tif
        the Sentinel-2 water frequency of each site's openEO cube and the
        intertidal mask derived from it (pixels seen wet and dry: water
        frequency 0.05-0.95 inside the transition class). Drawn as three
        classes: water (frequency >= 0.5 outside the flats), intertidal flats,
        land. Components smaller than 0.02 km^2 are drawn as land (speckle;
        invisible at this scale anyway).
    data_v4/gauges/estuary_truth/stations.json
        the in-estuary tide gauges (Rijkswaterstaat, WSV, IOC, Puertos del
        Estado); the outer gauges at the mouths are not drawn
Outputs
    poster/figures/cg/sites_map.{pdf,png}   slot size of cartograss2026.tex
                                            (391 x 126 mm); PNG at 300 dpi
    The flats of Villaviciosa come from its 2016-2025 product, the other four
    from 2023-2025 cubes (Westerschelde, Wadden and Ems at 20 m); the poster's
    data credit covers 2016-2025.
    poster/build/cg/sites_map.json          sites, extents, gauges drawn, sources

Strip layout (2026-10-19): the Ems-Dollard, Westerschelde and Ferrol windows are
widened by a little (``pad_strip``, km on the west, east, south and north
sides) so that no gauge triangle crosses an inset frame; where a widened
window runs past the product raster, that margin is drawn from Natural Earth
(land white, the rest water: tidal flats beyond a raster are not mapped by
Natural Earth). Gauges less than 0.8 km apart share one triangle (Terneuzen
and Terneuzen west lock), and every scale-bar label stays 2 mm inside its frame.
Round 3 (2026-10-19): the strip titles site 4 "Wadden" (``name_strip``; the
caption names it "a Wadden Sea tidal basin"), as wide as its frame, so the gap
after it is the standard 9 mm plus the room the right-aligned "Ems-Dollard"
title needs (17.8 mm, it was 28.9 mm); the freed width goes to the overview.
The Ems window is NOT widened east to fit its title over its own frame: the
Ems river leaves the product raster on its east edge (water in the last
column between about y 5909 and 5923 km, UTM 32N), and Natural Earth would
draw the river past that edge as land.

Round 4 (2026-10-19)
    * The overview's number discs and leaders are placed in figure mm from
      the axes' data transform. GeoPandas' plot() draws the canvas while the
      axes still hold the land layer's autoscaled limits and aspect 'equal',
      which shrinks the axes box; set_xlim/set_ylim do not restore the box
      until the next draw, so the transform read right after them was the
      shrunken box's, and the discs were squeezed toward the centre by
      56.4/66.9 once the overview was widened. ax.apply_aspect() now runs
      before the transform is read, and after the final draw every leader is
      checked to start within 0.2 mm of its site (the script stops if not).
    * Only the 13 headline judges of the frozen validation are drawn
      (experiments/benchmark/judges.HEADLINE, read as a literal, with the
      same-location groups of experiments/benchmark/config.SAME_LOCATION):
      Borssele and Eemshaven haven (coverage) and Ferrol 4 (less than 3 km
      from the boundary gauge Ferrol 1) are excluded by the method-independent
      rules (docs/validation_v2_spec.md), so a triangle on the strip is a
      gauge used as a check, nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, "pyintertidal")):
    sys.exit("run from the repository root (the folder that contains pyintertidal/)")

CACHE = "poster/tools/cg/cache"
NE_DIRS = [CACHE, "data_v4/naturalearth"]
NE_URL = "https://naciscdn.org/naturalearth/10m/{kind}/{name}.zip"
NE_LAYERS = {"land": ("physical", "ne_10m_land"),
             "lakes": ("physical", "ne_10m_lakes"),
             "borders": ("cultural", "ne_10m_admin_0_boundary_lines_land")}
STATIONS = "data_v4/gauges/estuary_truth/stations.json"
JUDGES_PY = "experiments/benchmark/judges.py"       # HEADLINE (read as a literal, not imported)
CONFIG_PY = "experiments/benchmark/config.py"       # SAME_LOCATION (idem)
MARKER_TOL_MM = 0.2                                 # a leader must start this close to its site
OUT_FIG = "poster/figures/cg/sites_map"
OUT_JSON = "poster/build/cg/sites_map.json"
OUT_FIG_STRIP = "poster/figures/cg/sites_map_strip"
OUT_JSON_STRIP = "poster/build/cg/sites_map_strip.json"
STRIP_W, STRIP_H = 391.0, 54.0  # the strip's slot in cartograss2026.tex (\figSitesW x \figSitesH)

# Poster palette (poster/preamble/colors.tex) and two tints of it for the maps
INK, SLATE, DEEP, TEAL, SAND = "#16202A", "#4A5C6B", "#0B3B5A", "#1B7F86", "#C08A3E"
MIST, RULE = "#EEF2F5", "#C9D3DA"


def tint(hex_colour, share):
    """``share`` of the colour over white."""
    c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    return tuple((share * c + (1 - share) * 255) / 255)


WATER = tint(DEEP, 0.15)          # pale deep-blue: sea, rivers, lakes
FLATS = tint(SAND, 0.55)          # sand: intertidal flats
LAND = (1.0, 1.0, 1.0)
MM = 1 / 25.4
MIN_SPECKLE_KM2 = 0.02
MERGE_KM = 0.8                    # gauges closer than this share one triangle

# The overview: Lambert azimuthal equal-area centred on the Bay of Biscay
LAEA = "+proj=laea +lat_0=48.5 +lon_0=-1.0 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
OVERVIEW_LONLAT = (-11.0, 10.0, 41.7, 55.6)      # lon0, lon1, lat0, lat1 kept in view
STRIP_LONLAT = (-11.0, 8.6, 41.7, 55.6)          # the strip's narrower overview (east of the Ems cut)

# Sites in the order of the poster text (text_methods.tex). ``extent`` is the
# window drawn, in the CRS of the site's rasters; ``at`` is the marker of the
# overview (lon, lat) with an optional nudge of the number disc in mm.
SITES = [
    dict(n=1, key="villaviciosa", name="Villaviciosa", products="products_villaviciosa",
         extent=(301020, 310170, 4817400, 4824600), scale_km=2, bar="br",
         at=(-5.41, 43.515), nudge=(0.0, 0.0), nudge_strip=(0.0, 0.0)),
    dict(n=2, key="ferrol", name="Ferrol", products="products_ferrol",
         extent=(554400, 568790, 4809870, 4818920), scale_km=5, bar="br",
         pad_strip=(0.5, 0.0, 0.0, 0.0),
         at=(-8.25, 43.475), nudge=(0.0, 0.0), nudge_strip=(-1.6, 0.0)),
    dict(n=3, key="escalda", name="Westerschelde", products="products_escalda",
         extent=(538200, 559220, 5686660, 5701340), scale_km=5, bar="bl",
         pad_strip=(0.0, 0.0, 1.0, 0.0),
         at=(3.70, 51.39), nudge=(0.0, 0.0), nudge_strip=(-5.0, -1.5)),
    dict(n=4, key="wadden", name="Wadden Sea", name_strip="Wadden", products="products_wadden",
         extent=(642760, 670540, 5891100, 5925360), scale_km=10, bar="bl",
         at=(5.30, 53.29), nudge=(-2.6, 3.6), nudge_strip=(-4.4, 3.4)),
    dict(n=5, key="ems", name="Ems-Dollard", products="products_ems",
         extent=(354000, 387240, 5898080, 5930160), scale_km=10, bar="bl",
         pad_strip=(1.5, 0.0, 1.2, 2.0),
         at=(7.05, 53.36), nudge=(2.4, 3.8), nudge_strip=(4.4, 3.4)),
]
ROWS = [[1, 2, "legend"], [3, 4, 5]]            # chip rows, top to bottom
# Sea names on the overview (lon, lat of the label centre); the strip's overview
# (391 x 54 mm strip: about 55 mm wide, a third of the grid's) has no room for
# a name beside the coasts and the discs, so it carries none
SEAS = {"Atlantic\nOcean": (-8.6, 47.4), "Bay of\nBiscay": (-4.2, 45.3), "North\nSea": (2.6, 54.9)}
STRIP_SEAS = {}                                  # the strip's overview is too narrow for a sea name

# Layout (mm)
OVERVIEW_GAP = 9.0          # overview to the chips
GAP = 9.0                   # between neighbouring site maps, both rows
TITLE_BAND = 8.6            # chip title band (number disc + name)
ROW_GAP = 4.0               # between the two chip rows
EDGE = 0.5                  # keeps every frame line inside the sheet
ROW2_SHARE = 0.58           # share of the map height given to the lower row
TITLE_PT, LABEL_PT = 22, 20
GAUGE_MS = 14.0             # gauge triangle, pt (about 5 mm), white edge


# ─────────────────────────────────────────────────────────────────────────────
#  Fonts: b_common.use_poster_fonts (the TrueType Fira of the figure cache,
#  built from MiKTeX's OTF when missing), the same call as every other figure
#  of the poster, so the output does not depend on which script ran first.
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import use_poster_fonts            # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────────────────
def natural_earth(layer):
    """Path of the zipped Natural Earth layer (land, lakes or borders): found
    in NE_DIRS, else downloaded once into the cache."""
    kind, name = NE_LAYERS[layer]
    for d in NE_DIRS:
        p = os.path.join(d, name + ".zip")
        if os.path.exists(p):
            return p
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + ".zip")
    url = NE_URL.format(kind=kind, name=name)
    print(f"  downloading {url} -> {p}")
    req = urllib.request.Request(url, headers={"User-Agent": "cartograss2026-poster-figure"})
    with urllib.request.urlopen(req, timeout=120) as r, open(p + ".part", "wb") as f:
        f.write(r.read())
    os.replace(p + ".part", p)
    return p


def overview_layers(bbox):
    """The Natural Earth layers of the overview map, clipped to ``bbox`` and
    projected to LAEA; returns them and the files they came from (a layer
    that cannot be read is left out and reported)."""
    import geopandas as gpd
    from shapely.geometry import box
    out, used = {}, {}
    for layer in ("land", "lakes", "borders"):
        try:
            path = natural_earth(layer)
        except Exception as e:                                       # noqa: BLE001
            print(f"  {layer}: unavailable ({e}); drawn without it")
            continue
        g = gpd.read_file(f"zip://{path}", bbox=bbox)
        if layer != "borders":
            g = gpd.clip(g, box(*bbox))
        out[layer] = g.to_crs(LAEA)
        used[layer] = path.replace("\\", "/")
    return out, used


def ne_land_mask(shape, transform, crs):
    """Natural Earth land (True) on a site grid, for a window that runs past the raster."""
    import geopandas as gpd
    from pyproj import Transformer
    from rasterio import features
    h, w = shape
    xs = transform.c + transform.a * np.array([0, w, w, 0])
    ys = transform.f + transform.e * np.array([0, 0, h, h])
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    bbox = (min(lon) - 0.05, min(lat) - 0.05, max(lon) + 0.05, max(lat) + 0.05)
    g = gpd.read_file(f"zip://{natural_earth('land')}", bbox=bbox).to_crs(crs)
    if g.empty:
        return np.zeros(shape, bool)
    return features.rasterize(((geom, 1) for geom in g.geometry), out_shape=shape,
                              transform=transform, fill=0, dtype="uint8").astype(bool)


def site_classes(site, pad=(0.0, 0.0, 0.0, 0.0)):
    """RGB image of the site window: land, water, intertidal flats. ``pad`` widens the
    window by (west, east, south, north) km; what falls outside the raster is drawn
    from Natural Earth (land, else water)."""
    import rasterio
    from rasterio.windows import from_bounds
    from scipy import ndimage
    x0, x1, y0, y1 = site["extent"]
    x0, x1 = x0 - 1000.0 * pad[0], x1 + 1000.0 * pad[1]
    y0, y1 = y0 - 1000.0 * pad[2], y1 + 1000.0 * pad[3]
    with rasterio.open(os.path.join(site["products"], "water_frequency.tif")) as src:
        win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
        wf = src.read(1, window=win, boundless=True, fill_value=np.nan)
        tr = src.window_transform(win)
        crs = src.crs
        res = abs(src.transform.a)
        rb = src.bounds
    with rasterio.open(os.path.join(site["products"], "intertidal_mask.tif")) as src:
        flats = src.read(1, window=win, boundless=True, fill_value=0).astype(bool)
    ext = (tr.c, tr.c + tr.a * wf.shape[1], tr.f + tr.e * wf.shape[0], tr.f)
    water = (np.nan_to_num(wf, nan=0.0) >= 0.5) & ~flats
    # the margin past the raster (a padded window): Natural Earth land, else water
    cx = tr.c + tr.a * (np.arange(wf.shape[1]) + 0.5)
    cy = tr.f + tr.e * (np.arange(wf.shape[0]) + 0.5)
    outside = ((cx[None, :] < rb.left) | (cx[None, :] > rb.right)
               | (cy[:, None] < rb.bottom) | (cy[:, None] > rb.top))
    if outside.any():
        land = ne_land_mask(wf.shape, tr, crs)
        water[outside] = ~land[outside]
        flats[outside] = False
    min_px = MIN_SPECKLE_KM2 * 1e6 / res**2
    for m in (water, flats):
        lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
        if n:
            size = np.bincount(lab.ravel())
            small = size < min_px
            small[0] = False
            m[small[lab]] = False
    rgb = np.empty(wf.shape + (3,), np.float32)
    rgb[:] = LAND
    rgb[water] = WATER
    rgb[flats] = FLATS
    return rgb, ext, crs, {"pixel_m": res, "window_px": [int(wf.shape[1]), int(wf.shape[0])],
                           "nan_px": int(np.isnan(wf).sum()),
                           "pad_km_wesn": list(pad), "px_outside_raster": int(outside.sum())}


def _module_literal(path, name):
    """The value of a module-level ``name = <literal>`` in a Python file, read with
    ast (the benchmark package is not imported: it pulls in the whole pipeline)."""
    import ast
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def headline_locations():
    """{site: set of headline location codes} and {station code: its location}
    (judges.HEADLINE; judges.location_of: the first code of a SAME_LOCATION group)."""
    head = {k: set(v) for k, v in _module_literal(JUDGES_PY, "HEADLINE").items()}
    loc = {c: g[0] for g in _module_literal(CONFIG_PY, "SAME_LOCATION") for c in g}
    return head, loc


def gauges_for(site_key, crs):
    """The site's in-estuary gauges that are headline judges (the others are listed
    with ``headline`` False and are not drawn)."""
    from pyproj import Transformer
    st = json.load(open(STATIONS, encoding="utf-8"))
    key = {"escalda": "escalda", "wadden": "wadden", "ems": "ems", "ferrol": "ferrol"}.get(site_key)
    head, loc = headline_locations()
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    out = []
    for s in st:
        if s.get("site") == key:
            x, y = tr.transform(s["lon"], s["lat"])
            out.append({"code": s["code"], "name": s["name"], "provider": s["provider"],
                        "lon": s["lon"], "lat": s["lat"], "x": x, "y": y,
                        "headline": loc.get(s["code"], s["code"]) in head.get(key, set())})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers (millimetres from the figure's bottom-left corner)
# ─────────────────────────────────────────────────────────────────────────────
def mm_axes(fig, W, H, x, y, w, h):
    """Axes placed in millimetres (x, y from the bottom-left corner) on a W x H mm figure."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def title_of(site, strip):
    """The inset title: the strip may use a shorter one (``name_strip``)."""
    return site.get("name_strip", site["name"]) if strip else site["name"]


def number_disc(fig, T, x, y, n, r=3.5, fontsize=LABEL_PT, zorder=6):
    """The numbered site disc (sand fill, deep-blue outline and number) at (x, y) mm."""
    from matplotlib.patches import Circle
    fig.add_artist(Circle((x, y), r, transform=T, facecolor=SAND, edgecolor=DEEP,
                          linewidth=1.0, zorder=zorder))
    fig.text(x, y - 0.15, str(n), transform=T, ha="center", va="center_baseline",
             fontsize=fontsize, fontweight="semibold", color=DEEP, zorder=zorder + 1)


def frame(ax, colour=SLATE, lw=0.9):
    """A map frame: no ticks, a thin outline in ``colour``."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(lw)


def scale_bar(ax, km, where, fontsize=LABEL_PT, map_w_mm=None, label_w_mm=None, inset_mm=2.0):
    """Scale bar in data units (metres) at a corner of a north-up map. With the map's
    width and the label's width in mm, the label is kept ``inset_mm`` inside the frame
    (centred on the bar when that fits, else pulled in)."""
    import matplotlib.patheffects as pe
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    L = km * 1000.0
    m = 0.05 * w
    xb = x0 + m if where.endswith("l") else x1 - m - L
    yb = y0 + 0.07 * h
    halo = [pe.withStroke(linewidth=5.5, foreground="white")]
    ax.plot([xb, xb + L], [yb, yb], color=INK, lw=3.2, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=5.6, foreground="white"), pe.Normal()], zorder=6)
    xl = xb + L / 2
    if map_w_mm and label_w_mm:
        per_mm = w / map_w_mm                                # metres per mm on this map
        half = 0.5 * label_w_mm * per_mm
        xl = min(max(xl, x0 + inset_mm * per_mm + half), x1 - inset_mm * per_mm - half)
    ax.text(xl, yb + 0.035 * h, f"{km:g} km", ha="center", va="bottom", color=INK,
            fontsize=fontsize, fontweight="semibold", path_effects=halo, zorder=7)


def text_width_mm(fig, T, s, **kw):
    """Width (mm) of a text drawn with these keywords, measured on the figure."""
    t = fig.text(0, 0, s, transform=T, **kw)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 25.4


# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Draw the overview and the five site maps in the chosen layout and write
    the figure and its JSON record (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layout", choices=("grid", "strip"), default="grid",
                    help="grid: overview + two rows of site maps (default, 391 x 126 mm); "
                         "strip: overview + the five site maps in one row, no legend")
    ap.add_argument("--width", type=float, default=None, help="figure width, mm (slot)")
    ap.add_argument("--height", type=float, default=None, help="figure height, mm (slot)")
    a = ap.parse_args()
    strip = a.layout == "strip"
    W = a.width if a.width else (STRIP_W if strip else 391.0)
    H = a.height if a.height else (STRIP_H if strip else 126.0)
    out_fig, out_json = (OUT_FIG_STRIP, OUT_JSON_STRIP) if strip else (OUT_FIG, OUT_JSON)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Polygon
    import geopandas as gpd
    from pyproj import Transformer
    use_poster_fonts()

    # ------------------------------------------------- the site rasters first
    # (their extents set the width of the map block, which sets the overview's)
    by_n = {s["n"]: s for s in SITES}
    data = {}
    for s in SITES:
        rgb, ext, crs, info = site_classes(s, s.get("pad_strip", (0.0, 0.0, 0.0, 0.0))
                                           if strip else (0.0, 0.0, 0.0, 0.0))
        data[s["n"]] = (rgb, ext, crs, info)
    aspect = {n: (d[1][1] - d[1][0]) / (d[1][3] - d[1][2]) for n, d in data.items()}
    gaps_after = {}
    title_shift = {}                 # mm a title starts left of its map (strip: the last one)
    if strip:
        # one row: every map as tall as the strip allows; the gap after a map
        # is GAP, or wider where its title needs the room (a title may reach
        # into the gap after its map, never into the next title)
        maps_h = H - 2 * EDGE - TITLE_BAND
        mfig = plt.figure(figsize=(W * MM, H * MM), dpi=300)
        mT = mtransforms.Affine2D().scale(MM) + mfig.dpi_scale_trans
        need = {s["n"]: 9.0 + text_width_mm(mfig, mT, title_of(s, strip), fontsize=TITLE_PT,
                                            fontweight="semibold") for s in SITES}
        plt.close(mfig)
        order = [s["n"] for s in SITES]
        for i, n in enumerate(order[:-1]):
            gaps_after[n] = max(GAP, need[n] - maps_h * aspect[n] + 3.0)
        # the last map has no gap after it: a longer title reaches LEFT instead,
        # over the gap before its map, which widens by as much
        over = need[order[-1]] - maps_h * aspect[order[-1]]
        if over > 0:
            title_shift[order[-1]] = over + 0.5
            gaps_after[order[-2]] += over + 0.5
        rows = [(maps_h, order)]
        h1 = h2 = maps_h
        chips_w = sum(maps_h * aspect[n] for n in order) + sum(gaps_after.values())
    else:
        maps_h = H - 2 * EDGE - 2 * TITLE_BAND - ROW_GAP
        h2 = ROW2_SHARE * maps_h
        h1 = maps_h - h2
        rows = [(h1, [1, 2, "legend"]), (h2, [3, 4, 5])]
        chips_w = sum(h2 * aspect[n] for n in rows[1][1]) + GAP * (len(rows[1][1]) - 1)
    cx0 = W - EDGE - chips_w
    ow = cx0 - OVERVIEW_GAP - EDGE                                  # overview width, mm

    # ---------------------------------------------------------------- overview
    lonlat = STRIP_LONLAT if strip else OVERVIEW_LONLAT
    lon0, lon1, lat0, lat1 = lonlat
    to_laea = Transformer.from_crs("EPSG:4326", LAEA, always_xy=True)
    lons = np.r_[np.linspace(lon0, lon1, 50), np.full(50, lon1), np.linspace(lon1, lon0, 50), np.full(50, lon0)]
    lats = np.r_[np.full(50, lat0), np.linspace(lat0, lat1, 50), np.full(50, lat1), np.linspace(lat1, lat0, 50)]
    bx, by = to_laea.transform(lons, lats)
    ox0, ox1, oy0, oy1 = bx.min(), bx.max(), by.min(), by.max()
    ow_nat = (H - 2 * EDGE) * (ox1 - ox0) / (oy1 - oy0)
    if ow < ow_nat - 0.5:
        raise SystemExit(f"the overview would be cropped ({ow:.0f} of {ow_nat:.0f} mm): "
                         + ("widen the strip" if strip else "lower ROW2_SHARE"))
    half = 0.5 * (oy1 - oy0) * ow / (H - 2 * EDGE)                  # widened east-west, same centre
    xc = 0.5 * (ox0 + ox1)
    ox0, ox1 = xc - half, xc + half
    layers, ne_used = overview_layers((lon0 - 8, lat0 - 3, lon1 + 8, lat1 + 2))

    fig = plt.figure(figsize=(W * MM, H * MM), dpi=300)
    T = mtransforms.Affine2D().scale(MM) + fig.dpi_scale_trans
    ax = mm_axes(fig, W, H, EDGE, EDGE, ow, H - 2 * EDGE)
    ax.set_facecolor(WATER)
    if "land" in layers:
        layers["land"].plot(ax=ax, color=LAND, edgecolor=SLATE, linewidth=0.45, zorder=1)
    if "lakes" in layers:
        layers["lakes"].plot(ax=ax, color=WATER, edgecolor=SLATE, linewidth=0.3, zorder=2)
    if "borders" in layers:
        layers["borders"].plot(ax=ax, color=RULE, linewidth=0.7, zorder=2)
    # faint graticule: orientation without a north arrow (meridians converge)
    for lon in range(-20, 21, 5):
        gx, gy = to_laea.transform(np.full(100, lon), np.linspace(35, 60, 100))
        ax.plot(gx, gy, color=RULE, lw=0.45, zorder=0.5)
    for lat in range(40, 61, 5):
        gx, gy = to_laea.transform(np.linspace(-25, 25, 250), np.full(250, lat))
        ax.plot(gx, gy, color=RULE, lw=0.45, zorder=0.5)
    ax.set_xlim(ox0, ox1); ax.set_ylim(oy0, oy1)
    # GeoPandas' plot() drew the canvas while the axes held the land layer's own
    # limits and aspect 'equal', which shrank the axes box; the limits above do not
    # restore it until the next draw. apply_aspect() puts the box where the final
    # draw will have it, so the transform read below is the final one.
    ax.apply_aspect()
    frame(ax)
    for txt, (lon, lat) in (STRIP_SEAS if strip else SEAS).items():
        tx, ty = to_laea.transform(lon, lat)
        ax.text(tx, ty, txt, ha="center", va="center", color=SLATE, fontsize=LABEL_PT,
                fontstyle="italic", linespacing=0.95, zorder=3)
    # site markers: numbered discs, a dot on the exact spot when nudged
    data_to_mm = ax.transData + T.inverted()
    markers = []                                   # (n, data x, data y, leader start mm)
    for s in SITES:
        sx, sy = to_laea.transform(*s["at"])
        px, py = data_to_mm.transform((sx, sy))
        dx, dy = s["nudge_strip"] if strip else s["nudge"]
        if dx or dy:
            fig.add_artist(Line2D([px, px + dx], [py, py + dy], transform=T, color=DEEP, lw=0.9, zorder=5))
            ax.plot(sx, sy, "o", ms=3, color=DEEP, zorder=5)
        number_disc(fig, T, px + dx, py + dy, s["n"])
        markers.append((s["n"], sx, sy, px, py, dx, dy))

    # ------------------------------------------------------------------ chips
    record = {"panel_mm": [W, H], "overview": {"crs": LAEA, "lonlat_in_view": lonlat,
              "widened_to_mm": round(ow, 1), "natural_width_mm": round(ow_nat, 1),
              "natural_earth": ne_used},
              "layout_mm": {"gap": GAP, "row_heights": [round(h1, 1), round(h2, 1)],
                            "map_block": [round(cx0, 1), round(chips_w, 1)]}, "sites": []}
    y_top = H - EDGE
    legend_cell = None
    for h, items in rows:
        y_map = y_top - TITLE_BAND - h
        x = cx0
        for j, item in enumerate(items):
            if item == "legend":
                legend_cell = (x, y_map, W - EDGE - x, h + TITLE_BAND)
                break
            s = by_n[item]
            rgb, ext, crs, info = data[item]
            w_map = h * aspect[item]
            # a title may reach into the gap after its map, never into the next title
            # (the strip's last title reaches left instead, by title_shift)
            title = title_of(s, strip)
            w_title = 9.0 + text_width_mm(fig, T, title, fontsize=TITLE_PT, fontweight="semibold")
            gap_after = gaps_after.get(item, GAP)
            shift = title_shift.get(item, 0.0)
            room = w_map + shift + (gap_after - 3.0 if j < len(items) - 1 else 0.0)
            if w_title > room + 0.05:
                print(f"  WARNING: title of {s['name']} is {w_title:.0f} mm, {room:.0f} mm available")
            axc = mm_axes(fig, W, H, x, y_map, w_map, h)
            axc.imshow(rgb, extent=ext, interpolation="antialiased", zorder=1)
            axc.set_xlim(ext[0], ext[1]); axc.set_ylim(ext[2], ext[3])
            frame(axc)
            gs_all = gauges_for(s["key"], crs)
            gs = [g for g in gs_all if g["headline"]]    # the checks: headline judges only
            not_headline = [g["code"] for g in gs_all if not g["headline"]]
            drawn = []
            marks = []                                   # [x, y, codes]: one triangle each
            for g in gs:
                if ext[0] <= g["x"] <= ext[1] and ext[2] <= g["y"] <= ext[3]:
                    for mk in marks:
                        if np.hypot(mk[0] - g["x"], mk[1] - g["y"]) < MERGE_KM * 1000.0:
                            n_ = len(mk[2])
                            mk[0] = (mk[0] * n_ + g["x"]) / (n_ + 1)
                            mk[1] = (mk[1] * n_ + g["y"]) / (n_ + 1)
                            mk[2].append(g["code"])
                            break
                    else:
                        marks.append([g["x"], g["y"], [g["code"]]])
                    drawn.append(g["code"])
            for gx_, gy_, _ in marks:
                axc.plot(gx_, gy_, marker="^", ms=GAUGE_MS, mfc=TEAL, mec="white", mew=1.6,
                         ls="", zorder=5)
            # how close each triangle comes to the frame (mm), for the report
            mm_per_m = w_map / (ext[1] - ext[0])
            edge = [min(gx_ - ext[0], ext[1] - gx_, gy_ - ext[2], ext[3] - gy_) * mm_per_m
                    for gx_, gy_, _ in marks]
            if edge and min(edge) < 3.0:
                print(f"  WARNING: a gauge of {s['name']} is {min(edge):.1f} mm from the frame")
            lab_w = text_width_mm(fig, T, f"{s['scale_km']:g} km", fontsize=LABEL_PT,
                                  fontweight="semibold")
            scale_bar(axc, s["scale_km"], s["bar"], map_w_mm=w_map, label_w_mm=lab_w)
            # title: number disc and name, on the band above the map
            yt = y_map + h + TITLE_BAND / 2 - 0.3
            number_disc(fig, T, x - shift + 3.5, yt, s["n"])
            fig.text(x - shift + 9.0, yt, title, transform=T, ha="left", va="center_baseline",
                     fontsize=TITLE_PT, fontweight="semibold", color=DEEP)
            record["sites"].append({"n": s["n"], "name": s["name"], "title": title,
                                    "rasters": s["products"],
                                    "crs": str(crs), "extent": list(map(float, ext)),
                                    "map_mm": [round(w_map, 1), round(h, 1)], "x_mm": round(x, 1),
                                    **info, "scale_bar_km": s["scale_km"], "gauges_drawn": drawn,
                                    "gauge_marks": [m_[2] for m_ in marks],
                                    "gauge_min_mm_from_frame": round(min(edge), 1) if edge else None,
                                    "gauges_outside_window": [g["code"] for g in gs if g["code"] not in drawn],
                                    "gauges_not_headline": not_headline})
            print(f"  {s['n']} {s['name']}: map {w_map:.1f} x {h:.1f} mm at x {x:.1f}, {len(drawn)} gauges"
                  f" in {len(marks)} triangles"
                  + (f" ({len(gs) - len(drawn)} outside the window)" if len(gs) > len(drawn) else "")
                  + (f"; not headline judges, not drawn: {', '.join(not_headline)}" if not_headline else ""))
            x += w_map + gap_after
        y_top = y_map - ROW_GAP

    # ----------------------------------------------------------------- legend
    # (the strip has none: its key is set in the poster, on the panel's title
    # line, by blk_mm_materials.tex)
    if strip:
        legend_cell = None
    else:
        lx, ly, lw, lh = legend_cell
        items = [("swatch", FLATS, "intertidal flats"),
                 ("swatch", WATER, "water"),
                 ("gauge", TEAL, "tide gauge")]
        need = 10.5 + max(text_width_mm(fig, T, lab, fontsize=LABEL_PT) for _, _, lab in items)
        if need > lw:
            print(f"  WARNING: the legend needs {need:.0f} mm, {lw:.0f} mm available")
        yy = ly + lh - TITLE_BAND / 2 - 0.3
        for kind, colour, label in items:
            if kind == "swatch":
                fig.add_artist(Rectangle((lx + 0.5, yy - 2.6), 7.0, 5.2, transform=T, facecolor=colour,
                                         edgecolor=SLATE, linewidth=0.6, zorder=5))
            else:
                fig.add_artist(Line2D([lx + 4.0], [yy + 0.4], transform=T, marker="^", ms=GAUGE_MS, mfc=TEAL,
                                      mec="white", mew=1.6, ls="", zorder=5))
            fig.text(lx + 10.5, yy, label, transform=T, ha="left", va="center_baseline",
                     fontsize=LABEL_PT, color=INK)
            yy -= 9.4
        record["legend_mm"] = [round(lx, 1), round(lw, 1)]

    if strip:
        record["layout_mm"]["gaps_after"] = {str(k): round(v, 1) for k, v in gaps_after.items()}
    # self-check after the final draw: each leader (or disc, when not nudged) must
    # start on its site, i.e. the transform used above is the one the file gets
    fig.canvas.draw()
    final_mm = ax.transData + T.inverted()
    record["overview"]["markers_mm"] = []
    worst = 0.0
    for n, sx, sy, px, py, dx, dy in markers:
        qx, qy = final_mm.transform((sx, sy))
        err = float(np.hypot(qx - px, qy - py))
        worst = max(worst, err)
        record["overview"]["markers_mm"].append({"n": n, "site_mm": [round(float(qx), 2), round(float(qy), 2)],
                                                 "disc_mm": [round(float(px + dx), 2), round(float(py + dy), 2)],
                                                 "leader_mm": round(float(np.hypot(dx, dy)), 2),
                                                 "error_mm": round(err, 3)})
    print(f"  overview markers: largest offset of a leader from its site {worst:.3f} mm"
          f" (tolerance {MARKER_TOL_MM} mm)")
    if worst > MARKER_TOL_MM:
        raise SystemExit("the overview's discs are not on their sites: nothing written")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig + ".png", dpi=300)
    fig.savefig(out_fig + ".pdf", dpi=300)
    plt.close(fig)
    print(f"  written {out_fig}.png / .pdf  ({W:g} x {H:g} mm)")
    record["gauges_source"] = STATIONS
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(f"  written {out_json}")
    try:
        import psutil
        mi = psutil.Process().memory_info()
        print(f"  peak memory {getattr(mi, 'peak_wset', mi.rss) / 2**20:.0f} MB")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
