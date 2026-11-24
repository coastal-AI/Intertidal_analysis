# -*- coding: utf-8 -*-
"""Write the website's data folder from the MAREA map layers: the files the map viewer reads.

Run it from the site folder with any Python that has NumPy and Pillow:

    python scripts/data_layers.py --layers <folder of the layer files>

The layer folder holds index.json (the sites), heights.json (each site's NMMA constant C, or a note where the heights
stay on the tide model's datum) and one sub-folder per site of SITE_ORDER. The layer index must list exactly the sites
of SITE_ORDER, in that order; a site named in LEFT_OUT is skipped, so leaving a site out of the website's data is one
line (the texts that count or name the sites must then be edited too; a link to the site opens the default site).
Each sub-folder has a manifest.json and, for every map layer, a value image
(<layer>_values.png) on the site's Web Mercator grid; plus the tide lines (tide_lines.geojson, tide_levels.json) and,
at Mundaka, the lines of equal delay (cotidal.geojson). Any coloured images of the layer folder are not used: the
overlays are drawn here, from the values, with the colour scales of src/features/explorer/ramps.json. The names of a
site come from its manifest, except a long name of LONG_NAMES, shortened to fit the site switcher.

Every layer of a site must be drawn on one domain, which its manifest names in a "domain" block: "hand-cleaned" (the
pixels with an elevation inside an intertidal mask cleaned by hand) or "automatic" (inside a mask made by a rule). A
site with an automatic mask is listed as preliminary: the viewer lists it under "Preliminary · automatic mask" in the
site switcher of wide screens, writes "preliminary" on its chip on phones, and gives one line on what that means under
the line of its images. A layer folder without a domain block (layers drawn on a production's own, noisy mask) is
refused.

It writes, under --out (default public/data, next to this script's folder):

    sites/index.json                the site list (schema cg26-sites/1), with a "preliminary" flag per site
    sites/<id>/manifest.json        the site's map: grid, layers, legends and tide levels (schema cg26-explorer-site/2)
    sites/<id>/<layer>.png          overlay images, drawn from the values (<layer>_<k>.png tiles where one image would
                                    be too large)
    sites/<id>/<layer>_values.png   value images, copied byte for byte (code = 256 R + G, value = offset + scale code,
                                    code 0 = no data)
    sites/<id>/tide_lines.geojson   contours at the tidal levels, labelled with the real minus sign; where heights.json
                                    says how the site's C was tied to NMMA, that rule as "C_rule" in the metadata
    sites/<id>/cotidal.geojson      lines of equal delay (where the layer folder has them), without a "status" field
    sites/<id>/thumb.webp           a small picture of the elevation map, shown where a browser cannot draw the map

The overlays are drawn at twice the grid's resolution (2.5 m on the ground) on the grid's own extent. Each overlay
pixel interpolates bilinearly the four grid pixels around it, with the weights of those without a value dropped and the
others scaled to sum to one, so it never shows a value beyond the data's; its opacity interpolates the data mask the
same way, so the edge of the data fades across one grid pixel instead of ending in hard steps. Linear scales colour the
interpolated value. The uncertainty has eight classes: their colours, not their values, are interpolated, so a blend
between two classes never reads as a third. The map viewer draws the overlays with linear resampling; the readout keeps
reading the value images, at the grid's own pixels. Where an overlay would be more than MAX_SIDE pixels wide or tall
(the texture size that phones' graphics chips take), it is cut into tiles, each with its own corners, at rows and
columns where the overlay hardly changes, so that the seams cannot be seen (see seam_steps).

The output is deterministic (sorted keys, LF line ends, PNG written by Pillow at a fixed level): a second run writes
the same bytes. Files under sites/ that this run does not write are removed; nothing else under --out is touched. The
script checks that every input is there, that the value images match their grids and carry no colour profile (so a
browser decodes them exactly), and that the files stay under the size caps of the website.
"""
import argparse  # command line
import hashlib  # file digests in the report
import io  # in-memory WebP encoding
import json  # manifests
import math  # Web Mercator
import os  # paths
import re  # gauge names and colours
import shutil  # byte-for-byte copies
import struct  # PNG chunk headers
import sys  # exit codes

import numpy as np  # value images and overlays
from PIL import Image  # PNG reading and writing, WebP writing

#: The sites of the layer folder, west to east by the longitude of their box centre. Four have a hand-cleaned mask
#: (foz, eo, villaviciosa, mundaka); the other ten an automatic one, so the website lists them as preliminary.
SITE_ORDER = ["corme", "noia", "betanzos", "ferrol", "ortigueira", "foz", "eo", "villaviciosa", "tina_menor",
              "san_vicente", "mogro", "santander", "santona", "mundaka"]
LEFT_OUT = ()  # sites of SITE_ORDER not put on the website, e.g. ("mogro",): one line leaves a site out
#: Long names shorter than the layer folder's, where that one is cut in the site switcher of wide screens (238 px of
#: text; "Ría de San Vicente de la Barquera" needs 259 px). The site list and the site's manifest both take this one.
LONG_NAMES = {"san_vicente": "Ría de San Vicente"}
REGIONS = {  # the region line of each site card
    "corme": "Galicia",
    "noia": "Galicia",
    "betanzos": "Galicia",
    "ferrol": "Galicia",
    "ortigueira": "Galicia",
    "foz": "Galicia",
    "eo": "Galicia–Asturias",
    "villaviciosa": "Asturias",
    "tina_menor": "Cantabria",
    "san_vicente": "Cantabria",
    "mogro": "Cantabria",
    "santander": "Cantabria",
    "santona": "Cantabria",
    "mundaka": "Basque Country",
}
DOMAINS = {"hand-cleaned": False, "automatic": True}  # the domain kinds a layer folder may name: preliminary or not

#: The raster layers the map offers, in the order of its layer list: id, title, one sentence, readout decimals.
RASTERS = [
    ("elevation", "Elevation", "Height of the intertidal ground, from the tide level at which each pixel floods.", 2),
    ("sigma", "Uncertainty", "How sharply each pixel turns from dry to wet: a larger value is a less certain height.", 2),
    ("emersion", "Emersion time", "Share of the time each pixel is out of the water: the darker, the longer.", 0),
    ("delay", "Delay", "How many minutes after the estuary mouth the tide reaches each pixel.", 0),
]
#: The tide-line layer: id, title, one sentence.
CONTOURS = ("tide_lines", "Tide lines", "Where the ground meets each tidal level of the site.")
EQUAL_DELAY_TITLE = "Lines of equal delay"  # the delay layer's lines, where the layer folder has them
#: The delay legend's note where every section has no delay: the layer folder's "No delay here: ..." is said as what
#: MAREA finds, since published tide tables may give a delay where MAREA finds none (e.g. the Bahía de Santander).
NO_DELAY_NOTE = ("No delay here:", "MAREA finds no delay here:")
#: Notes of the delay legend that the layer folder does not carry, by site.
DELAY_NOTES = {"noia": "Counted from the west edge of the map, not from the mouth of the ría."}
#: Units as a legend writes them, where they need more words than the readout's "%".
LEGEND_UNITS = {"emersion": "% of hours out of the water"}

R_EARTH = 6378137.0  # Web Mercator sphere radius (m)
MINUS = "−"  # the real minus sign
CAPS = {  # largest bytes per output file, by kind, and in all
    "overlay": 1_500_000,
    "values": 1_500_000,
    "geojson": 300_000,
    "thumb": 120_000,
    "manifest": 60_000,
    "total": 50_000_000,  # fourteen sites: about 42 MB; a visitor downloads only the site they open
}
THUMB_SIDE = 480  # longest side of the thumbnails (px)
UPSAMPLE = 2  # overlay pixels per grid pixel, along each axis (the interpolation below is written for 2)
MAX_SIDE = 4096  # largest width or height of one overlay image (px)
STRIP = 256  # grid rows interpolated at a time, which bounds the memory the overlays take
PNG_LEVEL = 9  # zlib level of the overlay images
RAMPS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "features", "explorer",
                          "ramps.json")  # the colour scales, shared with the front end's tests


def fail(msg):
    raise SystemExit(f"data_layers: {msg}")


def read_json(path):
    if not os.path.isfile(path):
        fail(f"missing input {os.path.basename(path)}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_text(obj, compact=False):
    """Deterministic JSON text: sorted keys, UTF-8, LF, a final newline."""
    if compact:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n"


def signed(value, decimals=2):
    """+1.19, −0.46, 0.00 (the real minus sign)."""
    text = f"{abs(value):.{decimals}f}"
    if float(text) == 0:
        return text
    return (MINUS if value < 0 else "+") + text


def png_chunks(path):
    """Chunk types of a PNG file and its IHDR (width, height, bit depth, colour type)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        fail(f"{path}: not a PNG file")
    types, ihdr, i = [], None, 8
    while i + 8 <= len(data):
        (n,) = struct.unpack(">I", data[i:i + 4])
        kind = data[i + 4:i + 8].decode("ascii", "replace")
        if kind == "IHDR":
            ihdr = struct.unpack(">IIBB", data[i + 8:i + 18])
        types.append(kind)
        i += 12 + n
    return types, ihdr


def mercator_to_lonlat(x, y):
    lon = math.degrees(x / R_EARTH)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2.0)
    return lon, lat


def corners_of(left, top, right, bottom):
    """[lon, lat] of the top-left, top-right, bottom-right and bottom-left corners of a north-up EPSG:3857 rectangle
    (MapLibre's order), to 7 decimals."""
    pts = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return [[round(v, 7) for v in mercator_to_lonlat(x, y)] for x, y in pts]


def grid_of(manifest, site_id):
    """The site's web grid: size, top-left corner and pixel size in EPSG:3857, corners and bounds in degrees."""
    g = manifest["grid"]
    if g.get("crs") != "EPSG:3857":
        fail(f"{site_id}: grid is not EPSG:3857")
    a, b, c, d, e, f = g["transform"]  # x = c + a col, y = f + e row
    if b != 0 or d != 0 or abs(a + e) > 1e-9 or a <= 0:
        fail(f"{site_id}: grid is not a north-up square grid")
    width, height, res = int(g["width"]), int(g["height"]), float(a)
    if abs(c - g["bounds_3857"][0]) > 0.01 or abs(f - g["bounds_3857"][3]) > 0.01:
        fail(f"{site_id}: transform and bounds_3857 disagree")
    if abs(c + width * res - g["bounds_3857"][2]) > 0.01 or abs(f - height * res - g["bounds_3857"][1]) > 0.01:
        fail(f"{site_id}: grid size and bounds_3857 disagree")
    corners = corners_of(c, f, c + width * res, f - height * res)
    for (lon, lat), (lon2, lat2) in zip(corners, g["corners_4326"]):
        if abs(lon - lon2) > 2e-7 or abs(lat - lat2) > 2e-7:
            fail(f"{site_id}: corners_4326 do not match the grid")
    return {
        "width": width,
        "height": height,
        "x0": c,
        "y0": f,
        "res": res,
        "corners": corners,  # TL, TR, BR, BL (MapLibre order)
        "bounds": [round(v, 6) for v in g["bounds_4326"]],
        "centre": [round(v, 6) for v in g["centre_4326"]],
    }


def value_codes(path, width, height, site_id):
    """The uint16 codes of a value image, after checking it decodes exactly in a browser."""
    types, ihdr = png_chunks(path)
    colour_chunks = {"gAMA", "cHRM", "sRGB", "iCCP", "cICP", "mDCv", "cLLi"} & set(types)
    if colour_chunks:
        fail(f"{site_id}: {os.path.basename(path)} has colour chunks {sorted(colour_chunks)}")
    if ihdr is None or ihdr[:2] != (width, height) or ihdr[2] != 8 or ihdr[3] != 2:
        fail(f"{site_id}: {os.path.basename(path)} is not an 8-bit RGB image of {width} x {height}")
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    if rgb[..., 2].any():
        fail(f"{site_id}: {os.path.basename(path)} has a non-zero blue band")
    return rgb[..., 0].astype(np.uint16) * 256 + rgb[..., 1].astype(np.uint16)


def check_samples(codes):
    """Six [col, row, code] probes: five data pixels spread over the scan order and one no-data pixel."""
    flat = codes.ravel()
    data = np.flatnonzero(flat)
    if data.size == 0:
        fail("a value image without data")
    n = data.size
    picks = [int(data[k]) for k in (0, n // 4, n // 2, (3 * n) // 4, n - 1)]
    empty = np.flatnonzero(flat == 0)
    if empty.size:
        picks.append(int(empty[empty.size // 2]))
    width = codes.shape[1]
    return [[i % width, i // width, int(flat[i])] for i in sorted(set(picks))]


# ---------------------------------------------------------------------------------------------------- colour scales
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
# OKLab (Björn Ottosson, 2020): linear sRGB -> LMS -> cube root -> Lab.
LMS_OF_RGB = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]])
LAB_OF_LMS = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                       [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]])
RGB_OF_LMS = np.linalg.inv(LMS_OF_RGB)
LMS_OF_LAB = np.linalg.inv(LAB_OF_LMS)


def hex_rgb(colour):
    """'#RRGGBB' as sRGB in [0, 1]."""
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64) / 255.0


def srgb_to_oklab(rgb):
    """sRGB in [0, 1] (any leading shape, last axis 3) to OKLab."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ LMS_OF_RGB.T) @ LAB_OF_LMS.T


def oklab_to_srgb(lab):
    """OKLab to sRGB in [0, 1] (colours outside the sRGB gamut clipped)."""
    lin = np.clip(((lab @ LMS_OF_LAB.T) ** 3) @ RGB_OF_LMS.T, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def to_byte(x):
    """[0, 1] to 0..255, rounded half up."""
    return np.floor(np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_ramps(path):
    """The colour scales of ramps.json: {layer: {"stops": [...]} or {"classes": [...]}}, colours checked."""
    ramps = read_json(path)
    if ramps.get("schema") != "cg26-ramps/1":
        fail(f"{os.path.basename(path)}: schema is not cg26-ramps/1")
    for lid, _, _, _ in RASTERS:
        ramp = ramps.get(lid)
        colours = (ramp or {}).get("stops", (ramp or {}).get("classes"))
        if not isinstance(colours, list) or len(colours) < 2 or not all(isinstance(c, str) and HEX.match(c)
                                                                        for c in colours):
            fail(f"{os.path.basename(path)}: no valid colours for {lid}")
    return ramps


def ramp_lut(stops, n=256):
    """``n`` colours (uint8 RGB) evenly over a scale's stops, which are evenly spaced and interpolated in OKLab, as
    the legend's colour bar draws them."""
    labs = np.array([srgb_to_oklab(hex_rgb(c)) for c in stops])
    t = np.linspace(0.0, 1.0, n) * (len(stops) - 1)
    i = np.minimum(np.floor(t).astype(np.intp), len(stops) - 2)
    f = (t - i)[:, None]
    return to_byte(oklab_to_srgb(labs[i] * (1.0 - f) + labs[i + 1] * f))


# ---------------------------------------------------------------------------------------------------- overlays
def upsample2(p):
    """Bilinear interpolation of grid rows at twice their resolution, between pixel centres.

    ``p`` holds the rows to draw with one more row and column of the grid on each side. Overlay pixel (2i + a, 2j + b)
    sits a quarter of a grid pixel from the centre of grid pixel (i, j), towards row i − 1 for a = 0 and row i + 1 for
    a = 1 (columns alike), so on each axis it takes 3/4 of pixel (i, j) and 1/4 of that neighbour.
    """
    rows = np.empty((2 * (p.shape[0] - 2), p.shape[1]))
    rows[0::2] = 0.25 * p[:-2] + 0.75 * p[1:-1]
    rows[1::2] = 0.75 * p[1:-1] + 0.25 * p[2:]
    out = np.empty((rows.shape[0], 2 * (p.shape[1] - 2)))
    out[:, 0::2] = 0.25 * rows[:, :-2] + 0.75 * rows[:, 1:-1]
    out[:, 1::2] = 0.75 * rows[:, 1:-1] + 0.25 * rows[:, 2:]
    return out


def padded(a):
    """``a`` as float32 with its edge rows and columns repeated once: the frame of the grid ends the data with a
    straight cut, so nothing outside the grid is read or faded in."""
    return np.pad(np.asarray(a, dtype=np.float32), 1, mode="edge")


def overlay_rgba(codes, values, scale, ramp, site_id, lid):
    """The RGBA overlay (2 H x 2 W) of a layer from its codes (code 0 = no data) and values.

    Linear scales: the interpolated value (only from grid pixels with a value, weights scaled to sum to one) coloured
    with 256 steps of the ramp. Classes: the OKLab colours of the classes, interpolated the same way. Opacity: the
    interpolated data mask. Pixels without opacity are (0, 0, 0, 0).
    """
    height, width = codes.shape
    valid = codes > 0
    mask_p = padded(valid)
    if scale["type"] == "linear":
        d0, d1 = scale["domain"]
        lut = ramp_lut(ramp["stops"])
        sums_p = [padded(np.where(valid, values, 0.0))]
    else:
        class_values = [c["value"] for c in scale["classes"]]
        labs = np.array([srgb_to_oklab(hex_rgb(c)) for c in ramp["classes"]])
        for c, lab in zip(ramp["classes"], labs):  # a class's own pixels must keep its exact colour
            if not np.array_equal(to_byte(oklab_to_srgb(lab)), np.round(hex_rgb(c) * 255).astype(np.uint8)):
                fail(f"{lid}: colour {c} does not survive OKLab")
        index = np.full(codes.shape, -1, dtype=np.int8)
        for k, v in enumerate(class_values):
            index[valid & (np.abs(values - v) < 1e-6)] = k
        if (index[valid] < 0).any():
            fail(f"{site_id}/{lid}: values outside the classes {class_values}")
        sums_p = [padded(np.where(valid, labs[index, axis], 0.0)) for axis in range(3)]
        del index
    rgba = np.zeros((UPSAMPLE * height, UPSAMPLE * width, 4), dtype=np.uint8)
    for r0 in range(0, height, STRIP):
        r1 = min(height, r0 + STRIP)
        w = upsample2(mask_p[r0:r1 + 2])
        alpha = to_byte(w)
        shown = alpha > 0
        sums = [upsample2(s[r0:r1 + 2]) for s in sums_p]
        mean = [np.divide(s, w, out=np.zeros_like(s), where=w > 0) for s in sums]
        if scale["type"] == "linear":
            t = np.clip((mean[0] - d0) / (d1 - d0), 0.0, 1.0)
            rgb = lut[np.floor(t * 255.0 + 0.5).astype(np.intp)]
        else:
            rgb = to_byte(oklab_to_srgb(np.stack(mean, axis=-1)))
        block = rgba[UPSAMPLE * r0:UPSAMPLE * r1]
        block[..., :3] = np.where(shown[..., None], rgb, 0)
        block[..., 3] = alpha
    return rgba


def parts_of(size):
    """The fewest equal parts of at most MAX_SIDE pixels that a side of ``size`` pixels splits into."""
    n = -(-size // MAX_SIDE)
    while size % n:
        n += 1
    return n


def seam_steps(codes, axis):
    """How much an overlay changes across each boundary between two of its rows (axis 0) or columns (axis 1).

    Entry k − 1 is for the boundary between overlay rows k − 1 and k: grid pixels that differ between grid rows i and
    i + 1 for k = 2 i + 2 (the overlay steps by half their difference there), half as many as differ between rows
    i − 1 and i + 1 for k = 2 i + 1 (a quarter of their difference). The graphics card cannot interpolate across the
    seam between two tiles, so a seam shows that step as it is. Where the grid rows on either side are the same (the
    5 m grid repeats most rows of the 10 m product), the step is nothing and a seam there cannot be seen.
    """
    c = codes if axis == 0 else codes.T
    steps = np.zeros(2 * c.shape[0] - 1)
    steps[1::2] = (c[1:] != c[:-1]).sum(axis=1)
    steps[2:-1:2] = 0.5 * (c[2:] != c[:-2]).sum(axis=1)
    steps[0] = steps[-1] = np.inf  # never at the frame
    return steps


def tile_zoom(width, height, res):
    """The zoom level of the tile grid on which MapLibre places an image of width x height pixels of ``res`` m
    (EPSG:3857): the floor of −log2 of its larger side in world units (the world is 0 to 1). MapLibre rounds an image's
    corners to that grid, so images that share it draw their common edges at one place, with no gap and no overlap."""
    return math.floor(-math.log2(max(width, height) * res / (2.0 * math.pi * R_EARTH)))


def image_zoom(corners):
    """tile_zoom() of an image from its corners, as MapLibre computes it from the [lon, lat] it is given."""
    xs = [(180.0 + lon) / 360.0 for lon, _ in corners]
    ys = [(180.0 - math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))) / 360.0
          for _, lat in corners]
    return math.floor(-math.log2(max(max(xs) - min(xs), max(ys) - min(ys))))


def tiling(steps_y, steps_x, height, width, res):
    """Row and column cuts of a height x width overlay of ``res`` m pixels: none where a side fits in MAX_SIDE; else
    the cuts with the smallest steps (seam_steps, summed over the layers) that keep every tile within MAX_SIDE and
    at one MapLibre zoom (equal parts, which always do, where a side needs more than two)."""
    def options(steps, size):
        n = parts_of(size)
        if n == 1:
            return [((), 0.0)]
        equal = tuple(k * size // n for k in range(1, n))
        if n > 2:
            return [(equal, 0.0)]
        ks = np.arange(size - MAX_SIDE, MAX_SIDE + 1)  # both parts within MAX_SIDE
        best = ks[np.argsort(steps[ks - 1], kind="stable")[:200]]
        return [((int(k),), float(steps[k - 1])) for k in best] + [(equal, float(steps[equal[0] - 1]))]

    def zooms(ys, xs):
        edges_y, edges_x = (0, *ys, height), (0, *xs, width)
        return {tile_zoom(x1 - x0, y1 - y0, res) for y0, y1 in zip(edges_y, edges_y[1:])
                for x0, x1 in zip(edges_x, edges_x[1:])}

    pairs = sorted(((sy + sx, ys, xs) for ys, sy in options(steps_y, height) for xs, sx in options(steps_x, width)),
                   key=lambda p: (p[0], p[1], p[2]))
    for _, ys, xs in pairs:
        if len(zooms(ys, xs)) == 1:
            return list(ys), list(xs)
    fail("no tiling keeps the overlay tiles at one zoom")


def write_overlay(rgba, cuts, grid, out, lid, site_id):
    """Write a layer's overlay (one image, or tiles of at most MAX_SIDE pixels a side, cut at ``cuts`` = (rows,
    columns)) and return its entry."""
    height, width = rgba.shape[:2]
    rows, cols = [0, *cuts[0], height], [0, *cuts[1], width]
    res = grid["res"] / UPSAMPLE  # overlay pixel (EPSG:3857 m)
    # Tile edges from whole pixel counts: neighbours share the very same edge, and the last one is the grid's.
    xs = [grid["x0"] + c * res for c in cols]
    ys = [grid["y0"] - r * res for r in rows]
    os.makedirs(out, exist_ok=True)
    tiles, single = [], len(rows) == 2 and len(cols) == 2
    for ty in range(len(rows) - 1):
        for tx in range(len(cols) - 1):
            name = f"{lid}.png" if single else f"{lid}_{len(tiles)}.png"
            path = os.path.join(out, name)
            tile = np.ascontiguousarray(rgba[rows[ty]:rows[ty + 1], cols[tx]:cols[tx + 1]])
            if max(tile.shape[:2]) > MAX_SIDE:
                fail(f"{site_id}/{lid}: an overlay tile over {MAX_SIDE} px")
            Image.fromarray(tile).save(path, "PNG", compress_level=PNG_LEVEL)
            WRITTEN.append(os.path.abspath(path))
            tiles.append({"file": name, "width": tile.shape[1], "height": tile.shape[0],
                          "corners": corners_of(xs[tx], ys[ty], xs[tx + 1], ys[ty + 1])})
    if len({image_zoom(t["corners"]) for t in tiles}) != 1:
        fail(f"{site_id}/{lid}: overlay tiles at different MapLibre zooms")
    first, last = tiles[0]["corners"], tiles[-1]["corners"]
    if first[0] != grid["corners"][0] or last[2] != grid["corners"][2]:
        fail(f"{site_id}/{lid}: the overlay does not cover the grid")
    return {"width": width, "height": height, "tiles": tiles}


def thumbnail(rgba, out_path):
    """A WebP picture of the elevation overlay, at most THUMB_SIDE pixels on its longest side, alpha kept."""
    im = Image.fromarray(rgba)
    scale = THUMB_SIDE / max(im.size)
    size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
    small = im.convert("RGBa").resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0).convert("RGBA")
    buf = io.BytesIO()
    small.save(buf, "WEBP", quality=82, alpha_quality=90, method=6)
    with open(out_path, "wb") as f:
        f.write(buf.getvalue())
    return size


def gauge_parts(gauge):
    """("Ferrol", "IOC fer2") from "Ferrol (IOC fer2)"."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", gauge or "")
    return (m.group(1), m.group(2)) if m else (gauge or "", "")


def heights_of(entry, site_id):
    """The site's heights, for the site list and the map (the map also gets the gauge name)."""
    if entry.get("datum") == "NMMA":
        if not isinstance(entry.get("C_m"), (int, float)):
            fail(f"{site_id}: NMMA heights without C")
        name, code = gauge_parts(entry.get("gauge"))
        scenes, total = entry.get("scenes"), entry.get("record_scenes")
        images = f"{scenes} images" if scenes == total else f"{scenes} of {total} images"
        source = f"{name} tide gauge ({code}), {images}" if code else f"{name} tide gauge, {images}"
        listed = {"datum": "NMMA", "C": round(float(entry["C_m"]), 4), "sigma": round(float(entry["sigma_m"]), 3),
                  "source": source}
        return listed, dict(listed, gauge=name)
    if entry.get("datum") == "model":
        listed = {"datum": "model", "C": None, "note": entry.get("note") or "Heights above the tide model's mean sea level."}
        return listed, dict(listed)
    fail(f"{site_id}: unknown height datum {entry.get('datum')!r}")


def relabel_lines(src, dst, levels_by_id, with_nmma, c_rule=None):
    """Copy a tide-line GeoJSON, with each label rebuilt as "<datum> <height> m" in the map's heights, and, where
    heights.json says how the site's C was tied to NMMA (its "rule"), that rule as "C_rule" in the metadata."""
    gj = read_json(src)
    if c_rule:
        if not isinstance(gj.get("metadata"), dict):
            fail(f"{os.path.basename(src)}: no metadata block for C_rule")
        gj["metadata"]["C_rule"] = c_rule
    for feat in gj.get("features", []):
        p = feat["properties"]
        if p.get("datum") not in levels_by_id:
            fail(f"{os.path.basename(src)}: unknown datum {p.get('datum')!r}")
        height = p["level_nmma_m"] if with_nmma else p["level_m"]
        p["label"] = f"{p['datum']} {signed(height)} m"
        if feat["geometry"]["type"] not in ("LineString", "MultiLineString"):
            fail(f"{os.path.basename(src)}: a {feat['geometry']['type']} among the lines")
    write_text(dst, json_text(gj, compact=True))
    return len(gj.get("features", []))


def copy_lines(src, dst):
    """Copy a GeoJSON of lines, without the "status" field of the layer folder (in its metadata or at the top)."""
    gj = read_json(src)
    gj.pop("status", None)
    if isinstance(gj.get("metadata"), dict):
        gj["metadata"].pop("status", None)
    for feat in gj.get("features", []):
        if feat["geometry"]["type"] not in ("LineString", "MultiLineString"):
            fail(f"{os.path.basename(src)}: a {feat['geometry']['type']} among the lines")
    write_text(dst, json_text(gj, compact=True))
    return len(gj.get("features", []))


WRITTEN = []  # every file this run writes (absolute paths), for the clean-up and the report


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    WRITTEN.append(os.path.abspath(path))


def copy_bytes(src, dst):
    if not os.path.isfile(src):
        fail(f"missing input {os.path.relpath(src)}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    WRITTEN.append(os.path.abspath(dst))


def scale_of(layer, ramp):
    """The legend of a raster layer: the layer folder's domain and ticks (or class values) with the ramp's colours."""
    cs = layer["colour_scale"]
    if cs["type"] == "linear" and "stops" in ramp:
        return {"type": "linear", "domain": [float(v) for v in cs["domain"]], "ticks": [float(v) for v in cs["ticks"]],
                "stops": list(ramp["stops"])}
    if cs["type"] == "categorical" and "classes" in ramp:
        if len(cs["classes"]) != len(ramp["classes"]):
            fail(f"layer {layer['id']}: {len(cs['classes'])} classes, {len(ramp['classes'])} colours")
        return {"type": "classes", "classes": [{"value": float(c["value"]), "color": colour}
                                               for c, colour in zip(cs["classes"], ramp["classes"])]}
    fail(f"layer {layer['id']}: its {cs['type']!r} scale does not match its ramp")


def site_attribution(production):
    tide = "EOT20 and an IOC tide gauge" if "IOC" in production.get("tide_boundary", "") else "EOT20"
    return f"MAREA maps CC BY 4.0 · modified Copernicus Sentinel-2 data · tide: {tide}"


def build_site(site_id, layers_dir, heights_entry, ramps, out_dir):
    src = os.path.join(layers_dir, site_id)
    man = read_json(os.path.join(src, "manifest.json"))
    if man.get("site", {}).get("id") != site_id:
        fail(f"{site_id}: manifest.json names another site")
    grid = grid_of(man, site_id)
    domain = man.get("domain") or {}
    if domain.get("kind") not in DOMAINS:
        fail(f"{site_id}: the layers name no domain (hand-cleaned or automatic mask); refused")
    preliminary = DOMAINS[domain["kind"]]
    prod = man["site"]["production"]
    by_id = {layer["id"]: layer for layer in man["layers"]}
    listed_heights, map_heights = heights_of(heights_entry, site_id)
    out = os.path.join(out_dir, "sites", site_id)
    levels_doc = read_json(os.path.join(src, "tide_levels.json"))
    levels = [{"id": lv["id"], "name": lv["name"], "level": float(lv["level_m"]),
               "nmma": None if lv.get("level_nmma_m") is None else float(lv["level_nmma_m"]),
               "drawn": bool(lv["drawn"])} for lv in levels_doc["levels"]]
    if [lv["id"] for lv in levels] != ["LAT", "MLWS", "MLWN", "MSL", "MHWN", "MHWS", "HAT"]:
        fail(f"{site_id}: tide levels are not LAT, MLWS, MLWN, MSL, MHWN, MHWS, HAT")
    if any(b["level"] <= a["level"] for a, b in zip(levels, levels[1:])):
        fail(f"{site_id}: tide levels are not increasing")
    with_nmma = listed_heights["datum"] == "NMMA"
    if with_nmma:
        C = listed_heights["C"]
        for lv in levels:
            if lv["nmma"] is None or abs(lv["nmma"] - (lv["level"] + C)) > 6e-4:
                fail(f"{site_id}: tide level {lv['id']} is not level + C in NMMA")

    codes_of = {}  # the codes of every raster layer, read first: the overlay tiles are cut where none of them changes
    for lid, _, _, _ in RASTERS:
        if lid not in by_id:
            fail(f"{site_id}: no {lid} layer")
        values_src = os.path.join(src, by_id[lid]["files"]["values_png"])
        codes_of[lid] = value_codes(values_src, grid["width"], grid["height"], site_id)
        copy_bytes(values_src, os.path.join(out, f"{lid}_values.png"))
    height, width = UPSAMPLE * grid["height"], UPSAMPLE * grid["width"]
    steps_y = sum(seam_steps(c, 0) for c in codes_of.values()) if parts_of(height) > 1 else None
    steps_x = sum(seam_steps(c, 1) for c in codes_of.values()) if parts_of(width) > 1 else None
    cuts = tiling(steps_y, steps_x, height, width, grid["res"] / UPSAMPLE)

    entries, thumb_size = [], None
    for lid, title, about, decimals in RASTERS:
        layer = by_id[lid]
        codes = codes_of.pop(lid)
        v = layer["values"]
        if v.get("nodata_code") != 0:
            fail(f"{site_id}/{lid}: value images must use code 0 for no data")
        scale = scale_of(layer, ramps[lid])
        values = float(v["offset"]) + float(v["scale"]) * codes.astype(np.float64)
        rgba = overlay_rgba(codes, values, scale, ramps[lid], site_id, lid)
        overlay = write_overlay(rgba, cuts, grid, out, lid, site_id)
        if lid == "elevation":
            thumb_size = thumbnail(rgba, os.path.join(out, "thumb.webp"))
            WRITTEN.append(os.path.abspath(os.path.join(out, "thumb.webp")))
        entry = {
            "id": lid,
            "kind": "raster",
            "title": title,
            "about": about,
            "units": layer["units"],
            "decimals": decimals,
            "overlay": overlay,
            "values": {"file": f"{lid}_values.png", "scale": float(v["scale"]), "offset": float(v["offset"]),
                       "check": check_samples(codes)},
            "scale": scale,
        }
        if lid in LEGEND_UNITS:
            entry["legendUnits"] = LEGEND_UNITS[lid]
        if lid == "elevation":
            entry["heights"] = True  # shown in the site's heights (NMMA = value + C where C exists)
        if lid == "delay":
            cot = levels_doc.get("cotidal", {})
            if cot.get("drawn"):
                if not os.path.isfile(os.path.join(src, cot["file"])):
                    fail(f"{site_id}: missing {cot['file']}")
                n = copy_lines(os.path.join(src, cot["file"]), os.path.join(out, "cotidal.geojson"))
                entry["lines"] = {"file": "cotidal.geojson", "title": EQUAL_DELAY_TITLE, "features": n,
                                  "legend": f"{EQUAL_DELAY_TITLE}, every {int(cot['step_min'])} min"}
            elif int(cot.get("max_delay_min", 99)) <= 1 and cot.get("why_not"):
                old, new = NO_DELAY_NOTE
                note = cot["why_not"]
                entry["note"] = new + note[len(old):] if note.startswith(old) else note
            if site_id in DELAY_NOTES:
                entry["note"] = " ".join(filter(None, [entry.get("note"), DELAY_NOTES[site_id]]))
        entries.append(entry)
        del codes, values, rgba

    cid, ctitle, cabout = CONTOURS
    levels_by_id = {lv["id"]: lv for lv in levels}
    lines_src = os.path.join(src, levels_doc.get("lines_file", "tide_lines.geojson"))
    n_lines = relabel_lines(lines_src, os.path.join(out, "tide_lines.geojson"), levels_by_id, with_nmma,
                            heights_entry.get("rule") if with_nmma else None)
    drawn = [lv["id"] for lv in levels if lv["drawn"]]
    entries.append({"id": cid, "kind": "contours", "title": ctitle, "about": cabout, "units": "m",
                    "file": "tide_lines.geojson", "features": n_lines, "levels": levels, "heights": True})
    if not drawn:
        fail(f"{site_id}: no tide line drawn")

    first, last = prod["record_first"], prod["record_last"]
    years = first[:4] if first[:4] == last[:4] else f"{first[:4]}–{last[:4]}"
    long_name = LONG_NAMES.get(site_id, man["site"]["long_name"])  # the name of the switcher, the info line, alt texts
    site_manifest = {
        "schema": "cg26-explorer-site/2",
        "id": site_id,
        "name": man["site"]["name"],
        "longName": long_name,
        "images": int(prod["record_scenes"]),
        "firstImage": first,
        "lastImage": last,
        "years": years,
        "heights": map_heights,
        "grid": grid,
        "layers": entries,
        "attribution": site_attribution(prod),
    }
    write_text(os.path.join(out, "manifest.json"), json_text(site_manifest))
    listed = {
        "id": site_id,
        "name": man["site"]["name"],
        "longName": long_name,
        "region": REGIONS[site_id],
        "firstImage": first,
        "lastImage": last,
        "years": years,
        "images": int(prod["record_scenes"]),
        "centre": [round(v, 5) for v in grid["centre"]],
        "bounds": [round(v, 5) for v in grid["bounds"]],
        "heights": listed_heights,
        "thumb": f"sites/{site_id}/thumb.webp",
        "manifest": f"sites/{site_id}/manifest.json",
        "layers": [e["id"] for e in entries],
        "preliminary": preliminary,
    }
    return listed, {"grid": f"{grid['width']} x {grid['height']}", "thumb": thumb_size, "lines": n_lines,
                    "drawn": drawn, "cuts": cuts, "domain": domain["kind"]}


def clean_stale(out_dir):
    """Remove files under sites/ that this run did not write (and empty folders)."""
    keep = set(WRITTEN)
    removed = 0
    for sub in ("sites",):
        top = os.path.join(out_dir, sub)
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                path = os.path.abspath(os.path.join(root, name))
                if path not in keep:
                    os.remove(path)
                    removed += 1
            if root != top and not os.listdir(root):
                os.rmdir(root)
    return removed


def check_caps(out_dir):
    total = 0
    for path in WRITTEN:
        size = os.path.getsize(path)
        total += size
        name = os.path.basename(path)
        kind = ("values" if name.endswith("_values.png") else "overlay" if name.endswith(".png") else
                "geojson" if name.endswith(".geojson") else "thumb" if name.endswith(".webp") else "manifest")
        if size > CAPS[kind]:
            fail(f"{os.path.relpath(path, out_dir)} is {size} bytes, over the {kind} cap of {CAPS[kind]}")
    if total > CAPS["total"]:
        fail(f"the data folder is {total} bytes, over the cap of {CAPS['total']}")
    return total


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", required=True, help="folder of the layer files (index.json, heights.json, <site>/)")
    ap.add_argument("--heights", help="heights.json (default: the one in --layers)")
    ap.add_argument("--out", default=os.path.join(here, os.pardir, "public", "data"),
                    help="the website's data folder (default: public/data)")
    args = ap.parse_args()
    layers_dir = os.path.abspath(args.layers)
    out_dir = os.path.abspath(args.out)
    ramps = load_ramps(RAMPS_FILE)
    index_in = read_json(os.path.join(layers_dir, "index.json"))
    heights = read_json(args.heights or os.path.join(layers_dir, "heights.json"))
    ids = [s["id"] for s in index_in["sites"]]
    if ids != SITE_ORDER:
        fail(f"the layer index lists {ids}, expected {SITE_ORDER}")
    for name, chosen in (("LEFT_OUT", LEFT_OUT), ("LONG_NAMES", LONG_NAMES)):
        unknown = sorted(set(chosen) - set(SITE_ORDER))
        if unknown:
            fail(f"{name} names sites that are not in SITE_ORDER: {unknown}")
    sites, report = [], {}
    for site_id in [s for s in SITE_ORDER if s not in LEFT_OUT]:
        if site_id not in heights.get("sites", {}):
            fail(f"heights.json has no entry for {site_id}")
        listed, rep = build_site(site_id, layers_dir, heights["sites"][site_id], ramps, out_dir)
        sites.append(listed)
        report[site_id] = rep
    index = {"schema": "cg26-sites/1", "licence": index_in["licence"], "attribution": index_in["attribution"],
             "sites": sites}
    write_text(os.path.join(out_dir, "sites", "index.json"), json_text(index))
    removed = clean_stale(out_dir)
    total = check_caps(out_dir)
    digest = hashlib.sha256()
    for path in sorted(WRITTEN):
        with open(path, "rb") as f:
            digest.update(os.path.relpath(path, out_dir).replace("\\", "/").encode() + b"\0" + f.read())
    for site_id, rep in report.items():
        rows, cols = rep["cuts"]
        cut = f"cut at rows {rows} columns {cols}" if rows or cols else "one image"
        print(f"{site_id:13s} grid {rep['grid']:11s} overlay {cut:30s} thumb {rep['thumb'][0]}x{rep['thumb'][1]}  "
              f"tide lines {rep['lines']} ({', '.join(rep['drawn'])})  domain {rep['domain']}")
    print(f"{len(WRITTEN)} files, {total / 1e6:.2f} MB, {removed} stale removed; digest {digest.hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
