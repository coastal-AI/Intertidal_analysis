# -*- coding: utf-8 -*-
"""CARTOGRASS 2026 poster, context row -- the same flat at low and at high tide.

Two equal panels of the Ria de Villaviciosa (Asturias) seen by Sentinel-2 at
10 m: one scene at low water, one at high water, same frame and same stretch.
Near-infrared reflectance (B08): open water absorbs it and comes out almost
black, exposed wet flats are mid-grey and vegetated land is bright, so the
tide reads from across the room.

Run from the repository root (one process, about 0.35 GB, ~10 s once cached):

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/fig_ctx_tide_pair.py
        [--width 211] [--height 136] [--rescan]

Inputs (read only, never loaded whole)
    ndwi_cube_villaviciosa_total_2023-2025.nc   openEO/CDSE cube, 10 m, EPSG:32630
                                                (B03, B04, B08, B11, SCL; one scene
                                                and one frame-sized window at a time)
    products_marea/overpass_times.json          acquisition time (UTC) of each date
    products_villaviciosa/intertidal_mask.tif   only to rank scenes (flats pixels)
    data_v4/gauges/ioc_gij2_2023-01-01_2025-12-31.json
                                                IOC Gijon gauge, 1-min: checks that the
                                                two scenes really are low and high water
Outputs
    poster/figures/cg/ctx_lowtide.{pdf,png}     slot size of cartograss2026.tex
    poster/figures/cg/ctx_hightide.{pdf,png}    (211 x 136 mm); PNG at 300 dpi
    poster/build/cg/ctx_tide_pair.json          the selection and every number behind it
    poster/tools/cg/cache/villaviciosa_scene_scan_<frame>.csv   per-scene scan (cache)

Scene choice (a rule, not a hand pick)
    1. Frame: 6.5 km wide (E 302900-309400, N from 4819300), height from the slot
       aspect; the flats of the middle ria plus the mouth channel and the sea.
    2. Clear sky: Sentinel-2 scene classification classes 4-7 (vegetation, not
       vegetated, water, unclassified), the same per-pixel mask as the poster's
       method. Candidates are the scenes at least 99.8 % clear over the frame.
    3. Tide state: open-water fraction on the intertidal flats (NDWI > 0 over the
       pixels of the intertidal mask, clear pixels only). Low tide = the
       candidate with the lowest fraction, high tide = the highest.
    4. Check: the Gijon gauge (IOC) at the overpass minute must sit in the lower
       quarter (low) or the upper quarter (high) of that day's range, else the
       script stops. Gauge levels are only checked, never drawn.
    With the 2023-2025 cube this gives 25 Sep 2025 and 2 Oct 2025, both at 11:21
    UTC, one week apart (spring low water, then neap high water).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

import numpy as np

ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, "pyintertidal")):
    sys.exit("run from the repository root (the folder that contains pyintertidal/)")

CUBE = "ndwi_cube_villaviciosa_total_2023-2025.nc"
TIMES = "products_marea/overpass_times.json"
FLATS = "products_villaviciosa/intertidal_mask.tif"
GAUGE = "data_v4/gauges/ioc_gij2_2023-01-01_2025-12-31.json"
CACHE = "poster/tools/cg/cache"
OUT_FIG = "poster/figures/cg"
OUT_JSON = "poster/build/cg/ctx_tide_pair.json"

FRAME_X0, FRAME_Y0, FRAME_W = 302900.0, 4819300.0, 6500.0   # m, EPSG:32630
CLEAR_CLASSES = (4, 5, 6, 7)
MIN_CLEAR = 0.998
NIR_MAX, GAMMA = 0.40, 0.8          # reflectance mapped to white; display gamma
SCALE_KM = 1.0

# Poster palette (poster/preamble/colors.tex)
INK, SLATE, DEEP, TEAL, SAND = "#16202A", "#4A5C6B", "#0B3B5A", "#1B7F86", "#C08A3E"
MIST, RULE = "#EEF2F5", "#C9D3DA"
# NIR ramp: water near-black with a deep-blue bias, wet flats neutral grey,
# vegetated land warm white. Monotone in lightness: it is a single band.
RAMP = [(0.00, "#061520"), (0.12, "#1B2F3D"), (0.42, "#86898A"),
        (0.75, "#D8D7D2"), (1.00, "#FBFAF6")]
MM = 1 / 25.4
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


# ─────────────────────────────────────────────────────────────────────────────
#  Fonts: b_common.use_poster_fonts (the TrueType Fira of the figure cache,
#  built from MiKTeX's OTF when missing), the same call as every other figure
#  of the poster, so the output does not depend on which script ran first.
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import use_poster_fonts            # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  The cube: one window, one scene at a time
# ─────────────────────────────────────────────────────────────────────────────
class Cube:
    """The netCDF cube opened for reading one frame-sized window of one scene
    at a time (``frame`` = (x0, x1, y0, y1) in metres of the cube's CRS)."""

    def __init__(self, path, frame):
        import netCDF4
        self.ds = netCDF4.Dataset(path)
        for v in ("B03", "B08", "SCL"):
            self.ds[v].set_auto_mask(True)
            self.ds[v].set_var_chunk_cache(size=4 * 2**20)   # bounded memory
        x = self.ds["x"][:].data.astype(float)
        y = self.ds["y"][:].data.astype(float)
        self.dates = [(dt.date(1990, 1, 1) + dt.timedelta(days=int(v))).isoformat()
                      for v in self.ds["t"][:].data]
        x0, x1, y0, y1 = frame
        if not (y[0] > y[-1]):
            raise SystemExit("unexpected y order in the cube (expected north-up)")
        # pixel centres inside the frame (10 m grid)
        self.c0 = int(np.searchsorted(x, x0 + 1e-6)); self.c1 = int(np.searchsorted(x, x1 - 1e-6))
        self.r0 = int(np.searchsorted(-y, -(y1 - 1e-6))); self.r1 = int(np.searchsorted(-y, -(y0 + 1e-6)))
        self.x, self.y = x[self.c0:self.c1], y[self.r0:self.r1]
        self.res = abs(float(x[1] - x[0]))
        self.full_x, self.full_y = x, y

    def window(self, band, i):
        """Band ``band`` of scene ``i`` inside the frame, as float32 with NaN for no data."""
        a = self.ds[band][i, self.r0:self.r1, self.c0:self.c1]
        return np.ma.filled(a.astype("f4"), np.nan)

    def close(self):
        """Close the netCDF file."""
        self.ds.close()


def flats_mask(cube):
    """Pixels of the intertidal mask inside the frame (aligned by grid offset;
    the mask is on the 10-year cube's grid, same 10 m lattice). Falls back to
    the whole frame if the mask is missing."""
    shape = (cube.r1 - cube.r0, cube.c1 - cube.c0)
    if not os.path.exists(FLATS):
        print(f"  {FLATS} missing: ranking scenes on the whole frame")
        return np.ones(shape, bool), "whole frame (mask missing)"
    import rasterio
    with rasterio.open(FLATS) as src:
        m = src.read(1).astype(bool); tr = src.transform
    cx0, cy0 = tr.c + tr.a / 2, tr.f + tr.e / 2              # first pixel centre
    co = int(round((cx0 - cube.full_x[0]) / cube.res))
    ro = int(round((cube.full_y[0] - cy0) / cube.res))
    full = np.zeros((len(cube.full_y), len(cube.full_x)), bool)
    r_lo, c_lo = max(ro, 0), max(co, 0)
    r_hi = min(ro + m.shape[0], full.shape[0]); c_hi = min(co + m.shape[1], full.shape[1])
    full[r_lo:r_hi, c_lo:c_hi] = m[r_lo - ro:r_hi - ro, c_lo - co:c_hi - co]
    win = full[cube.r0:cube.r1, cube.c0:cube.c1]
    return win, f"{FLATS} ({int(win.sum())} px in the frame)"


def scan(cube, flats, times, cache_path, rescan=False):
    """Per scene: clear fraction over the frame and open-water fraction on the
    flats (NDWI > 0, clear pixels). Cached as CSV."""
    if os.path.exists(cache_path) and not rescan:
        with open(cache_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) == len(cube.dates):
            print(f"  scan: cached {cache_path}")
            return [{"date": r["date"], "time_utc": r["time_utc"],
                     "clear_frame": float(r["clear_frame"]),
                     "water_flats": float(r["water_flats"]) if r["water_flats"] else np.nan}
                    for r in rows]
    print(f"  scan: {len(cube.dates)} scenes, window {cube.r1 - cube.r0} x {cube.c1 - cube.c0} px")
    rows = []
    for i, d in enumerate(cube.dates):
        scl = cube.window("SCL", i)
        clear = np.isin(np.nan_to_num(scl, nan=0).astype(int), CLEAR_CLASSES)
        cf = float(clear.mean())
        wf = np.nan
        if cf >= 0.99:                                    # NDWI only where it can matter
            g, n = cube.window("B03", i), cube.window("B08", i)
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = (g - n) / (g + n)
            ok = clear & flats & np.isfinite(ndwi)
            if ok.any():
                wf = float((ndwi[ok] > 0).mean())
        rows.append({"date": d, "time_utc": times[d][11:19], "clear_frame": cf, "water_flats": wf})
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "time_utc", "clear_frame", "water_flats"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "clear_frame": f"{r['clear_frame']:.5f}",
                        "water_flats": "" if not np.isfinite(r["water_flats"]) else f"{r['water_flats']:.5f}"})
    print(f"  scan: written {cache_path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Gijon gauge: one day sliced out of the raw IOC bytes (no full json.load)
# ─────────────────────────────────────────────────────────────────────────────
_REC = re.compile(rb'\{"slevel": (null|[-+0-9.eE]+), "stime": "(\d{4}-\d\d-\d\d) (\d\d):(\d\d):(\d\d)", '
                  rb'"sensor": "([^"]*)"\}')


def gauge_day(raw, date):
    """(minute of day, level) of the majority sensor on ``date`` (UTC)."""
    i = raw.find(b'"stime": "' + date.encode())
    if i < 0:
        return None
    nxt = (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat().encode()
    j = raw.find(b'"stime": "' + nxt, i)
    seg = raw[raw.rfind(b"{", 0, i): (j if j > 0 else len(raw))]
    recs = [(int(m.group(3)) * 60 + int(m.group(4)), float(m.group(1)), m.group(6).decode())
            for m in _REC.finditer(seg) if m.group(1) != b"null" and m.group(2).decode() == date]
    if not recs:
        return None
    sensors = [r[2] for r in recs]
    major = max(set(sensors), key=sensors.count)
    t = np.array([r[0] for r in recs if r[2] == major]); v = np.array([r[1] for r in recs if r[2] == major])
    o = np.argsort(t)
    return t[o], v[o], major


def tide_check(date, time_utc):
    """The Gijón gauge level at the overpass minute and its position in that
    day's range (0 = the day's low, 1 = its high), or None without a record."""
    if not os.path.exists(GAUGE):
        print(f"  gauge: {GAUGE} missing, tide state not checked")
        return None
    with open(GAUGE, "rb") as f:
        raw = f.read()
    got = gauge_day(raw, date)
    del raw
    if got is None:
        return None
    t, v, sensor = got
    hh, mm = int(time_utc[:2]), int(time_utc[3:5])
    k = int(np.argmin(np.abs(t - (hh * 60 + mm))))
    if abs(int(t[k]) - (hh * 60 + mm)) > 3:
        return None
    lo, hi = float(v.min()), float(v.max())
    return {"gauge": "IOC gij2 (Gijon)", "sensor": sensor, "level_m": round(float(v[k]), 3),
            "day_min_m": round(lo, 3), "day_max_m": round(hi, 3),
            "position_in_day_range": round((float(v[k]) - lo) / (hi - lo), 3),
            "n_samples": int(len(v))}


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────────────────────────────────────
def nir_image(cube, i):
    """Near-infrared reflectance (B08) of scene ``i`` stretched to [0, 1] with
    the fixed NIR_MAX and GAMMA (no-data pixels take their nearest value);
    returns the image and the number of pixels filled."""
    from scipy import ndimage
    n = cube.window("B08", i) / 10000.0
    bad = ~np.isfinite(n)
    if bad.any():                                    # a handful of no-data pixels
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        n = n[tuple(idx)]
    return np.clip(n / NIR_MAX, 0, 1) ** GAMMA, int(bad.sum())


def mm_transform(fig):
    """A transform that places artists in millimetres from the figure's bottom-left corner."""
    import matplotlib.transforms as mtransforms
    return mtransforms.Affine2D().scale(MM) + fig.dpi_scale_trans


LABEL_PAD, LABEL_INSET = 4.2, 5.0          # mm
TITLE_PT, SUB_PT, SUB_DROP = 34, 21, 14.2  # pt, pt, mm (subtitle below the title top)


def _label_texts(fig, T, title, subtitle, H):
    """Write the panel's title and subtitle in the top-left corner; returns both texts."""
    x = LABEL_INSET + LABEL_PAD
    y = H - LABEL_INSET - LABEL_PAD
    t1 = fig.text(x, y, title, transform=T, ha="left", va="top", color="white",
                  fontsize=TITLE_PT, fontweight="semibold", zorder=5)
    t2 = fig.text(x, y - SUB_DROP, subtitle, transform=T, ha="left", va="top", color=MIST,
                  fontsize=SUB_PT, zorder=5)
    return t1, t2


def label_extent(labels, W, H):
    """(right edge, bottom edge) in mm that fits every (title, subtitle) pair,
    so both panels get the same box and the pair reads as one comparison."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W * MM, H * MM), dpi=300)
    T = mm_transform(fig)
    xr, yb = 0.0, H
    for title, subtitle in labels:
        t1, t2 = _label_texts(fig, T, title, subtitle, H)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for t in (t1, t2):
            (x0, y0), (x1, y1) = T.inverted().transform(t.get_window_extent(r).get_points())
            xr, yb = max(xr, x1), min(yb, y0)
        t1.remove(); t2.remove()
    plt.close(fig)
    return xr + LABEL_PAD, yb - LABEL_PAD + 0.6


def draw_panel(path_stem, img, extent, title, subtitle, W, H, box):
    """Draw one W x H mm panel: the stretched NIR image, the label box (same
    size on both panels), a scale bar and a north arrow; save PNG and PDF."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle, Polygon

    cmap = LinearSegmentedColormap.from_list("nir", RAMP, N=512)
    fig = plt.figure(figsize=(W * MM, H * MM), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, extent=extent, interpolation="nearest")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    T = mm_transform(fig)
    inset = LABEL_INSET

    # -- label: title and date/time in a deep-blue box, top-left (same box on
    #    both panels) -----------------------------------------------------------
    _label_texts(fig, T, title, subtitle, H)
    x_right, y_bottom = box
    fig.add_artist(Rectangle((inset, y_bottom), x_right - inset, H - inset - y_bottom,
                             transform=T, facecolor=DEEP, edgecolor="none", alpha=0.92, zorder=4))

    # -- scale bar and north arrow, bottom-right on land ---------------------
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    km_mm = SCALE_KM * 1000.0 / (extent[1] - extent[0]) * W
    xr = W - inset - 13.0                             # right end of the bar
    xb, yb = xr - km_mm, inset + 2.0
    fig.add_artist(Rectangle((xb, yb), km_mm, 1.7, transform=T, facecolor=INK,
                             edgecolor="white", linewidth=1.0, zorder=5))
    for xe in (xb, xr):                               # end ticks
        fig.add_artist(Rectangle((xe - 0.35, yb), 0.7, 3.4, transform=T, facecolor=INK,
                                 edgecolor="white", linewidth=0.6, zorder=5))
    fig.text((xb + xr) / 2, yb + 4.2, f"{SCALE_KM:g} km", transform=T, ha="center", va="bottom",
             color=INK, fontsize=21, fontweight="semibold", path_effects=halo, zorder=6)
    # north arrow (grid north of UTM 30N; 1.6 deg from true north here)
    xn, yn = W - inset - 4.5, inset + 1.0
    fig.add_artist(Polygon([(xn - 2.6, yn), (xn, yn + 9.0), (xn + 2.6, yn), (xn, yn + 2.4)],
                           closed=True, transform=T, facecolor=INK, edgecolor="white",
                           linewidth=1.0, zorder=5))
    fig.text(xn, yn + 10.0, "N", transform=T, ha="center", va="bottom", color=INK,
             fontsize=21, fontweight="semibold", path_effects=halo, zorder=6)

    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    fig.savefig(path_stem + ".png", dpi=300)
    fig.savefig(path_stem + ".pdf", dpi=300)
    plt.close(fig)
    print(f"  written {path_stem}.png / .pdf  ({W:g} x {H:g} mm)")


def fmt_date(d):
    """'YYYY-MM-DD' as '25 Sep 2025' (abbreviated English month)."""
    y, m, dd = d.split("-")
    return f"{int(dd)} {MONTHS[int(m) - 1]} {y}"


def main():
    """Scan the scenes, choose the low-tide and high-tide pair, check them
    against the Gijón gauge and draw both panels (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=float, default=211.0, help="panel width, mm (slot)")
    ap.add_argument("--height", type=float, default=136.0, help="panel height, mm (slot)")
    ap.add_argument("--rescan", action="store_true", help="ignore the cached scene scan")
    a = ap.parse_args()
    W, H = a.width, a.height

    import matplotlib
    matplotlib.use("Agg")
    use_poster_fonts()

    frame_h = FRAME_W * H / W
    frame = (FRAME_X0, FRAME_X0 + FRAME_W, FRAME_Y0, FRAME_Y0 + frame_h)
    times = json.load(open(TIMES, encoding="utf-8"))
    cube = Cube(CUBE, frame)
    missing = [d for d in cube.dates if d not in times]
    if missing:
        raise SystemExit(f"{len(missing)} cube dates have no overpass time, e.g. {missing[:3]}")
    flats, flats_desc = flats_mask(cube)
    tag = f"{int(FRAME_X0)}_{int(FRAME_Y0)}_{int(FRAME_W)}x{int(round(frame_h))}"
    rows = scan(cube, flats, times, f"{CACHE}/villaviciosa_scene_scan_{tag}.csv", a.rescan)

    cand = [r for r in rows if r["clear_frame"] >= MIN_CLEAR and np.isfinite(r["water_flats"])]
    if len(cand) < 2:
        raise SystemExit(f"only {len(cand)} scenes are >= {MIN_CLEAR:.1%} clear over the frame")
    low = min(cand, key=lambda r: r["water_flats"])
    high = max(cand, key=lambda r: r["water_flats"])
    print(f"  candidates >= {MIN_CLEAR:.1%} clear: " + ", ".join(
        f"{r['date']} ({r['water_flats']:.2f})" for r in sorted(cand, key=lambda r: r["water_flats"])))

    record = {"cube": CUBE, "frame_epsg32630_m": dict(zip(("x0", "x1", "y0", "y1"), frame)),
              "panel_mm": [W, H], "clear_classes": list(CLEAR_CLASSES), "min_clear_frame": MIN_CLEAR,
              "flats_for_ranking": flats_desc, "n_scenes": len(rows), "n_candidates": len(cand),
              "candidates": {r["date"]: round(r["water_flats"], 4) for r in cand},
              "nir_display": {"band": "B08", "white_at_reflectance": NIR_MAX, "gamma": GAMMA},
              "panels": {}}
    panels = (("low", low, "Low tide", lambda p: p <= 0.25),
              ("high", high, "High tide", lambda p: p >= 0.75))
    subs = {key: f"{fmt_date(r['date'])} \u00b7 {r['time_utc'][:5]} UTC" for key, r, _, _ in panels}
    box = label_extent([(word, subs[key]) for key, _, word, _ in panels], W, H)
    for key, r, word, ok in panels:
        chk = tide_check(r["date"], r["time_utc"])
        if chk is not None and not ok(chk["position_in_day_range"]):
            raise SystemExit(f"{r['date']}: gauge at {chk['position_in_day_range']:.2f} of the day's "
                             f"range, not '{word.lower()}'; check the scene choice")
        i = cube.dates.index(r["date"])
        img, n_filled = nir_image(cube, i)
        sub = subs[key]
        stem = f"{OUT_FIG}/" + ("ctx_lowtide" if key == "low" else "ctx_hightide")
        extent = (cube.x[0] - cube.res / 2, cube.x[-1] + cube.res / 2,
                  cube.y[-1] - cube.res / 2, cube.y[0] + cube.res / 2)
        draw_panel(stem, img, extent, word, sub, W, H, box)
        record["panels"][key] = {"file": stem + ".{pdf,png}", "date": r["date"],
                                 "time_utc": r["time_utc"], "label": word, "subtitle": sub,
                                 "clear_fraction_frame": round(r["clear_frame"], 5),
                                 "open_water_fraction_flats": round(r["water_flats"], 4),
                                 "nodata_pixels_filled": n_filled, "gauge_check": chk}
        print(f"  {key}: {r['date']} {r['time_utc']} UTC  clear {r['clear_frame']:.4f}  "
              f"water on flats {r['water_flats']:.3f}  gauge {chk}")
    cube.close()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(f"  written {OUT_JSON}")
    try:
        import psutil
        mi = psutil.Process().memory_info()
        print(f"  peak memory {getattr(mi, 'peak_wset', mi.rss) / 2**20:.0f} MB")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
