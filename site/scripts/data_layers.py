# -*- coding: utf-8 -*-
"""Copy the MAREA map layers into the website's data folder and write the files the map viewer reads.

Run it from the site folder with any Python that has NumPy and Pillow:

    python scripts/data_layers.py --layers <folder of the layer files>

The layer folder holds index.json (the sites), heights.json (each site's NMMA constant C) and one sub-folder per site
(foz, eo, villaviciosa, santander, santona, mundaka). Each sub-folder has a manifest.json and, for every map layer,
an RGBA overlay image (<layer>.png) and a value image (<layer>_values.png), on the site's Web Mercator grid; plus the
tide lines (tide_lines.geojson, tide_levels.json) and, at Mundaka, the lines of equal delay (cotidal.geojson).

It writes, under --out (default public/data, next to this script's folder):

    sites/index.json                the site list (schema cg26-sites/1)
    sites/<id>/manifest.json        the site's map: grid, layers, legends and tide levels (schema cg26-explorer-site/1)
    sites/<id>/<layer>.png          overlay images, copied byte for byte
    sites/<id>/<layer>_values.png   value images, copied byte for byte (code = 256 R + G, value = offset + scale code,
                                    code 0 = no data)
    sites/<id>/tide_lines.geojson   contours at the tidal levels, labelled with the real minus sign
    sites/<id>/cotidal.geojson      lines of equal delay (where the layer folder has them)
    sites/<id>/thumb.webp           a small picture of the elevation map, shown where a browser cannot draw the map

The output is deterministic (sorted keys, LF line ends): a second run writes the same bytes. Files under sites/ that
this run does not write are removed; nothing else under --out is touched. The script checks that every input is
there, that the images match their grids, that the value images carry no colour profile (so a browser decodes them
exactly) and that the files stay under the size caps of the website.
"""
import argparse  # command line
import hashlib  # file digests in the report
import io  # in-memory WebP encoding
import json  # manifests
import math  # Web Mercator
import os  # paths
import re  # gauge names
import shutil  # byte-for-byte copies
import struct  # PNG chunk headers
import sys  # exit codes

import numpy as np  # value images
from PIL import Image  # PNG reading, WebP writing

SITE_ORDER = ["foz", "eo", "villaviciosa", "santander", "santona", "mundaka"]  # west to east
REGIONS = {  # the region line of each site card
    "foz": "Galicia",
    "eo": "Galicia–Asturias",
    "villaviciosa": "Asturias",
    "santander": "Cantabria",
    "santona": "Cantabria",
    "mundaka": "Basque Country",
}

#: The raster layers the map offers, in the order of its layer list: id, title, one sentence, readout decimals.
RASTERS = [
    ("elevation", "Elevation", "Height of the intertidal ground, from the tide level at which each pixel floods.", 2),
    ("sigma", "Uncertainty", "How sharply each pixel turns from dry to wet: a larger value is a less certain height.", 2),
    ("emersion", "Emersion time", "Share of the time each pixel is out of the water.", 0),
    ("delay", "Delay", "How many minutes after the estuary mouth the tide reaches each pixel.", 0),
]
#: The tide-line layer: id, title, one sentence.
CONTOURS = ("tide_lines", "Tide lines", "Where the ground meets each tidal level of the site.")
EQUAL_DELAY_TITLE = "Lines of equal delay"  # the delay layer's lines, where the layer folder has them

R_EARTH = 6378137.0  # Web Mercator sphere radius (m)
MINUS = "−"  # the real minus sign
CAPS = {  # largest bytes per output file, by kind, and in all
    "overlay": 1_500_000,
    "values": 1_500_000,
    "geojson": 300_000,
    "thumb": 120_000,
    "manifest": 60_000,
    "total": 26_000_000,
}
THUMB_SIDE = 480  # longest side of the thumbnails (px)


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
    corners = [mercator_to_lonlat(c, f), mercator_to_lonlat(c + width * res, f),
               mercator_to_lonlat(c + width * res, f - height * res), mercator_to_lonlat(c, f - height * res)]
    for (lon, lat), (lon2, lat2) in zip(corners, g["corners_4326"]):
        if abs(lon - lon2) > 2e-7 or abs(lat - lat2) > 2e-7:
            fail(f"{site_id}: corners_4326 do not match the grid")
    return {
        "width": width,
        "height": height,
        "x0": c,
        "y0": f,
        "res": res,
        "corners": [[round(lon, 7), round(lat, 7)] for lon, lat in corners],  # TL, TR, BR, BL (MapLibre order)
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


def overlay_ok(path, width, height, site_id):
    types, ihdr = png_chunks(path)
    if ihdr is None or ihdr[:2] != (width, height) or ihdr[3] != 6:
        fail(f"{site_id}: {os.path.basename(path)} is not an RGBA image of {width} x {height}")


def thumbnail(overlay_png, out_path):
    """A WebP picture of the elevation overlay, at most THUMB_SIDE pixels on its longest side, alpha kept."""
    with Image.open(overlay_png) as im:
        rgba = im.convert("RGBA")
    scale = THUMB_SIDE / max(rgba.size)
    size = (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale)))
    small = rgba.convert("RGBa").resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0).convert("RGBA")
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


def relabel_lines(src, dst, levels_by_id, with_nmma):
    """Copy a tide-line GeoJSON, with each label rebuilt as "<datum> <height> m" in the map's heights."""
    gj = read_json(src)
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
    gj = read_json(src)
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


def scale_of(layer):
    """The legend of a raster layer, from its colour scale."""
    cs = layer["colour_scale"]
    if cs["type"] == "linear":
        return {"type": "linear", "domain": [float(v) for v in cs["domain"]], "ticks": [float(v) for v in cs["ticks"]],
                "stops": list(cs["anchors"])}
    if cs["type"] == "categorical":
        return {"type": "classes", "classes": [{"value": float(c["value"]), "color": c["colour"]} for c in cs["classes"]]}
    fail(f"layer {layer['id']}: unsupported colour scale {cs['type']!r}")


def site_attribution(production):
    tide = "EOT20 and an IOC tide gauge" if "IOC" in production.get("tide_boundary", "") else "EOT20"
    return f"MAREA maps CC BY 4.0 · modified Copernicus Sentinel-2 data · tide: {tide}"


def build_site(site_id, layers_dir, heights_entry, out_dir):
    src = os.path.join(layers_dir, site_id)
    man = read_json(os.path.join(src, "manifest.json"))
    if man.get("site", {}).get("id") != site_id:
        fail(f"{site_id}: manifest.json names another site")
    grid = grid_of(man, site_id)
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

    entries = []
    for lid, title, about, decimals in RASTERS:
        if lid not in by_id:
            fail(f"{site_id}: no {lid} layer")
        layer = by_id[lid]
        files = layer["files"]
        overlay_src = os.path.join(src, files["overlay_png"])
        values_src = os.path.join(src, files["values_png"])
        overlay_ok(overlay_src, grid["width"], grid["height"], site_id)
        codes = value_codes(values_src, grid["width"], grid["height"], site_id)
        copy_bytes(overlay_src, os.path.join(out, f"{lid}.png"))
        copy_bytes(values_src, os.path.join(out, f"{lid}_values.png"))
        v = layer["values"]
        if v.get("nodata_code") != 0:
            fail(f"{site_id}/{lid}: value images must use code 0 for no data")
        entry = {
            "id": lid,
            "kind": "raster",
            "title": title,
            "about": about,
            "units": layer["units"],
            "decimals": decimals,
            "overlay": f"{lid}.png",
            "values": {"file": f"{lid}_values.png", "scale": float(v["scale"]), "offset": float(v["offset"]),
                       "check": check_samples(codes)},
            "scale": scale_of(layer),
        }
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
                entry["note"] = cot["why_not"]
        entries.append(entry)
        del codes

    cid, ctitle, cabout = CONTOURS
    levels_by_id = {lv["id"]: lv for lv in levels}
    lines_src = os.path.join(src, levels_doc.get("lines_file", "tide_lines.geojson"))
    n_lines = relabel_lines(lines_src, os.path.join(out, "tide_lines.geojson"), levels_by_id, with_nmma)
    drawn = [lv["id"] for lv in levels if lv["drawn"]]
    entries.append({"id": cid, "kind": "contours", "title": ctitle, "about": cabout, "units": "m",
                    "file": "tide_lines.geojson", "features": n_lines, "levels": levels, "heights": True})
    if not drawn:
        fail(f"{site_id}: no tide line drawn")

    thumb_size = thumbnail(os.path.join(src, by_id["elevation"]["files"]["overlay_png"]), os.path.join(out, "thumb.webp"))
    WRITTEN.append(os.path.abspath(os.path.join(out, "thumb.webp")))

    first, last = prod["record_first"], prod["record_last"]
    years = first[:4] if first[:4] == last[:4] else f"{first[:4]}–{last[:4]}"
    site_manifest = {
        "schema": "cg26-explorer-site/1",
        "id": site_id,
        "name": man["site"]["name"],
        "longName": man["site"]["long_name"],
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
        "longName": man["site"]["long_name"],
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
    }
    return listed, {"grid": f"{grid['width']} x {grid['height']}", "thumb": thumb_size, "lines": n_lines,
                    "drawn": drawn}


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
    index_in = read_json(os.path.join(layers_dir, "index.json"))
    heights = read_json(args.heights or os.path.join(layers_dir, "heights.json"))
    ids = [s["id"] for s in index_in["sites"]]
    if ids != SITE_ORDER:
        fail(f"the layer index lists {ids}, expected {SITE_ORDER}")
    sites, report = [], {}
    for site_id in SITE_ORDER:
        if site_id not in heights.get("sites", {}):
            fail(f"heights.json has no entry for {site_id}")
        listed, rep = build_site(site_id, layers_dir, heights["sites"][site_id], out_dir)
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
        print(f"{site_id:13s} grid {rep['grid']:11s} thumb {rep['thumb'][0]}x{rep['thumb'][1]}  "
              f"tide lines {rep['lines']} ({', '.join(rep['drawn'])})")
    print(f"{len(WRITTEN)} files, {total / 1e6:.2f} MB, {removed} stale removed; digest {digest.hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
