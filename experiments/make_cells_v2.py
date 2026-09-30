"""Terrain-following campaign cells for the north coast (v2).

Replaces the hand-sketched 47-vertex coastline of v1 (whose rectangular
cells cut rias in half and wasted area on open sea — 12 of 38 processed
cells came back ``sin_intermareal``) with cells derived from the real
coastline:

1. OSM ``natural=coastline`` for Fisterra–Bidasoa (Overpass), merged and
   filtered to the mainland component, simplified to 50 m.
2. OSM tidal polygons (``wetland=tidalflat``, tidal/estuarine/lagoon water,
   saltmarsh) — because OSM's coastline stops at estuary mouths, so
   Villaviciosa, Santoña and Urdaibai interiors are water/wetland, not
   coastline.
3. Strip = coastline buffered ±1.5 km (locally metric frame) ∪ the tidal
   pieces connected to it buffered 0.8 km, plus a 0.5 km global margin.
4. Partition by alongshore chainage of a 2.5 km-simplified trunk (10 km
   bins, 150 m label grid, nearest-trunk assignment). Connected blobs
   further than 2 km from the trunk (estuary interiors) are reassigned
   WHOLE to their majority bin, so no ria is split lengthwise.
5. Polygonise each bin (rasterio.features.shapes), drop crumbs < 1 km²
   (they remain covered by neighbouring bboxes — the downloader uses the
   polygon's bbox), simplify to 150 m.
6. Recursively split any cell whose bbox exceeds 150 km² (L-shaped cells
   around headlands produce oversized openEO jobs), dropping slivers
   < 2 km².

Result committed as ``north_coast_cells.json``: 130 cells, bbox median
72 km², max 148 km². v1 preserved as ``north_coast_cells_v1.json``.

Run:  python -m experiments.make_cells_v2          (~2 min + 2 downloads)

The two Overpass responses are cached in ``data_v4/`` (gitignored); delete
them to refetch. Verified checkpoints (all inside one cell each, printed at
the end): Villaviciosa mouth+interior together, Eo, Santoña, Urdaibai,
San Vicente, Ferrol, Bilbao.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

BBOX = (42.70, -9.40, 43.95, -1.60)          # south, west, north, east
LAT0 = 43.4
KX = 111.32 * np.cos(np.radians(LAT0))       # km per degree of longitude
KY = 111.32
KM2 = KX * KY / (111.32 * np.cos(np.radians(LAT0)))  # deg^2 -> km^2 (metric frame is already km)

BUFFER_COAST_KM = 1.5
BUFFER_TIDAL_KM = 0.8
MARGIN_KM = 0.5
TRUNK_SIMPLIFY_KM = 2.5
BIN_KM = 10.0
GRID_RES_KM = 0.15
INTERIOR_DIST_KM = 2.0
CRUMB_KM2 = 1.0
MAX_BBOX_KM2 = 150.0
SLIVER_KM2 = 2.0

COAST_CACHE = "data_v4/osm_coastline_north.json"
TIDAL_CACHE = "data_v4/osm_tidal_polys.json"

OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "pyintertidal-research/1.0",
           "Content-Type": "text/plain; charset=utf-8"}


def _fetch(query, cache, parse):
    if os.path.exists(cache):
        return json.load(open(cache))
    from pyintertidal.net import use_system_certificates
    use_system_certificates()
    import requests
    r = requests.post(OVERPASS, data=query.encode(), headers=HEADERS,
                      timeout=400)
    r.raise_for_status()
    out = parse(r.json())
    json.dump(out, open(cache, "w"))
    return out


def fetch_coastline():
    s, w, n, e = BBOX
    q = (f'[out:json][timeout:180];'
         f'way["natural"="coastline"]({s},{w},{n},{e});out geom;')
    return _fetch(q, COAST_CACHE, lambda d: [
        [(p["lon"], p["lat"]) for p in el["geometry"]]
        for el in d["elements"] if el.get("geometry")])


def fetch_tidal():
    s, w, n, e = BBOX
    b = f"({s},{w},{n},{e})"
    q = ('[out:json][timeout:240];('
         f'way["wetland"="tidalflat"]{b};'
         f'relation["wetland"="tidalflat"]{b};'
         f'way["natural"="water"]["tidal"="yes"]{b};'
         f'relation["natural"="water"]["tidal"="yes"]{b};'
         f'way["water"~"lagoon|estuary|tidal"]{b};'
         f'relation["water"~"lagoon|estuary|tidal"]{b};'
         f'way["natural"="wetland"]["wetland"~"saltmarsh"]{b};'
         f'relation["natural"="wetland"]["wetland"~"saltmarsh"]{b};'
         ');out geom;')

    def parse(d):
        polys = []
        for el in d["elements"]:
            if el["type"] == "way" and el.get("geometry"):
                polys.append([(p["lon"], p["lat"]) for p in el["geometry"]])
            elif el["type"] == "relation":
                for m in el.get("members", []):
                    if m.get("role") == "outer" and m.get("geometry"):
                        polys.append([(p["lon"], p["lat"])
                                      for p in m["geometry"]])
        return polys
    return _fetch(q, TIDAL_CACHE, parse)


def main():
    from shapely.geometry import LineString, Polygon, box, shape as shpshape
    from shapely.ops import linemerge, unary_union
    from shapely import contains_xy
    from scipy.spatial import cKDTree
    from scipy import ndimage
    import rasterio.features
    from affine import Affine

    ways = fetch_coastline()
    # everything below works in a locally metric frame (km): lon/lat scaled
    # by KX/KY so buffers and areas are true kilometres at this latitude
    lines = [LineString([(x * KX, y * KY) for x, y in w])
             for w in ways if len(w) >= 2]
    merged = linemerge(unary_union(lines))
    parts = (list(merged.geoms) if merged.geom_type == "MultiLineString"
             else [merged])
    main_line = max((p for p in parts), key=lambda p: p.length).simplify(0.05)
    print(f"coastline: {main_line.length:.0f} km (mainland component)")

    strip = main_line.buffer(BUFFER_COAST_KM, quad_segs=4)

    rings = fetch_tidal()
    tp = []
    for r in rings:
        if len(r) < 4:
            continue
        try:
            p = Polygon([(x * KX, y * KY) for x, y in r]).buffer(0)
            if p.is_valid and p.area > 0:
                tp.append(p.simplify(0.03))
        except Exception:
            pass
    tidal = unary_union(tp)
    pieces = (list(tidal.geoms) if tidal.geom_type != "Polygon" else [tidal])
    keep = [p for p in pieces if p.intersects(strip)]
    print(f"tidal polygons connected to the coast: {len(keep)} "
          f"({sum(p.area for p in keep):.0f} km2)")
    strip = unary_union(
        [strip] + [p.buffer(BUFFER_TIDAL_KM, quad_segs=4) for p in keep])
    strip = strip.buffer(MARGIN_KM, quad_segs=2)
    print(f"strip: {strip.area:.0f} km2")

    # ── chainage partition ───────────────────────────────────────────────
    trunk = main_line.simplify(TRUNK_SIMPLIFY_KM)
    tr_s = np.arange(0, trunk.length, 0.1)
    tr_xy = np.array([(q.x, q.y) for q in
                      (trunk.interpolate(s) for s in tr_s)])
    tree = cKDTree(tr_xy)

    x0, y0, x1, y1 = strip.bounds
    xs = np.arange(x0, x1 + GRID_RES_KM, GRID_RES_KM)
    ys = np.arange(y0, y1 + GRID_RES_KM, GRID_RES_KM)
    XX, YY = np.meshgrid(xs, ys)
    inside = contains_xy(strip, XX.ravel(), YY.ravel()).reshape(XX.shape)
    dist, idx = tree.query(np.c_[XX[inside], YY[inside]])
    lab = np.full(inside.shape, -1, np.int32)
    lab[inside] = (tr_s[idx] // BIN_KM).astype(int)
    dgrid = np.full(inside.shape, np.inf)
    dgrid[inside] = dist

    # blobs far from the trunk are estuary interiors: reassign each one
    # WHOLE to its majority bin so no ria gets split lengthwise
    comp, ncomp = ndimage.label(inside & (dgrid > INTERIOR_DIST_KM))
    for ci in range(1, ncomp + 1):
        m = comp == ci
        vals = lab[m]
        lab[m] = np.bincount(vals[vals >= 0]).argmax()

    transform = Affine(GRID_RES_KM, 0, x0 - GRID_RES_KM / 2,
                       0, GRID_RES_KM, y0 - GRID_RES_KM / 2)
    polys = []
    for b in sorted(set(lab[lab >= 0].tolist())):
        mask = lab == b
        for geom, v in rasterio.features.shapes(
                mask.astype(np.uint8), mask=mask, transform=transform):
            if v != 1:
                continue
            p = shpshape(geom).buffer(0).simplify(GRID_RES_KM)
            if p.area > CRUMB_KM2:
                polys.append(p)

    # ── bbox cap: split L-shaped cells so openEO jobs stay bounded ──────
    def bbox_km2(p):
        a0, b0, a1, b1 = p.bounds
        return (a1 - a0) * (b1 - b0)

    final = []
    for p0 in polys:
        queue = [p0]
        while queue:
            p = queue.pop()
            if bbox_km2(p) <= MAX_BBOX_KM2:
                if p.area >= SLIVER_KM2:
                    final.append(p)
                continue
            a0, b0, a1, b1 = p.bounds
            if (a1 - a0) >= (b1 - b0):
                mid = 0.5 * (a0 + a1)
                halves = [box(a0, b0, mid, b1), box(mid, b0, a1, b1)]
            else:
                mid = 0.5 * (b0 + b1)
                halves = [box(a0, b0, a1, mid), box(a0, mid, a1, b1)]
            for h in halves:
                g = p.intersection(h).buffer(0)
                for gg in (g.geoms if g.geom_type in
                           ("MultiPolygon", "GeometryCollection") else [g]):
                    if gg.geom_type == "Polygon" and gg.area > 0:
                        queue.append(gg)

    cells = []
    for p in final:
        cells.append({
            "km2": round(p.area, 1),
            "polygon": [[round(x / KX, 5), round(y / KY, 5)]
                        for x, y in p.exterior.coords]})
    cells.sort(key=lambda c: np.mean([q[0] for q in c["polygon"]]))
    out = [{"name": f"nc2_cell{i:03d}", "km2": c["km2"],
            "polygon": c["polygon"]} for i, c in enumerate(cells)]
    json.dump(out, open("north_coast_cells.json", "w"))

    bb = [bbox_km2(p) for p in final]
    print(f"{len(out)} cells; bbox km2 median {np.median(bb):.0f} "
          f"max {max(bb):.0f} total {sum(bb):.0f}")

    # checkpoints from the v1 post-mortem: each named ria must fall inside
    # a single cell (v1's rectangles cut several of them in half)
    from shapely.geometry import Point
    checks = {"Villaviciosa mouth": (-5.385, 43.535),
              "Villaviciosa interior": (-5.395, 43.505),
              "Eo interior": (-7.03, 43.52),
              "Santona E": (-3.44, 43.41),
              "Urdaibai": (-2.67, 43.37),
              "San Vicente": (-4.39, 43.38),
              "Ferrol": (-8.19, 43.47),
              "Bilbao": (-3.05, 43.33)}
    named = [(c["name"], Polygon(c["polygon"])) for c in out]
    for n, (lo, la) in checks.items():
        hits = [nm for nm, pp in named if pp.contains(Point(lo, la))]
        print(f"  {n:22s} -> {hits or '(covered by bbox only)'}")


if __name__ == "__main__":
    main()
