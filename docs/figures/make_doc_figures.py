#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures of the public documentation of MAREA (written to docs/figures/doc/).

Usage, from any folder (the repository root is found from this file's path):

    python -X utf8 docs/figures/make_doc_figures.py                 # every figure
    python -X utf8 docs/figures/make_doc_figures.py sites_map ...   # some figures
    python -X utf8 docs/figures/make_doc_figures.py --list          # their names

Inputs, all read-only and in the repository:
    products_marea/marea.npz, result.json, overpass_times.json
        the 10 m MAREA product of the Ria de Villaviciosa, its run summary and
        the acquisition time of every Sentinel-2 scene of the site
    products_villaviciosa/hsr_eot20_2023-2025.tif
        its header only: the canonical 10 m grid of Villaviciosa
    pyintertidal (sites, marea, elevation, tide_estimators)
        the site registry, the product constants and the fitting code itself
    experiments/benchmark/config.py, judges.py
        the role of each validation site, read as literals (ast)
    docs/figures/data/ne_10m_land_europe_atlantic.geojson
        Natural Earth 1:10m land and land borders (public domain), clipped
        and simplified (see build_data)
    docs/figures/data/villaviciosa_water_background.tif
        land/water background of Villaviciosa, from the Sentinel-2 water
        frequency of the site (see build_data)
Synthetic data: the tide of SYNTH (fixed constituents) and the seeded noise
of the flooding-curve example (SEED). Nothing here reads a validation result.

The two files of docs/figures/data/ are rebuilt by

    python -X utf8 docs/figures/make_doc_figures.py --build-data NE_DIR WATER_FREQUENCY_TIF

with NE_DIR holding ne_10m_land.zip and ne_10m_admin_0_boundary_lines_land.zip
(naturalearthdata.com) and WATER_FREQUENCY_TIF the products_villaviciosa/
water_frequency.tif written by the site notebook.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates                                  # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.gridspec import GridSpec                           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "figures" / "doc"
DATA = ROOT / "docs" / "figures" / "data"
COAST = DATA / "ne_10m_land_europe_atlantic.geojson"
BACKGROUND = DATA / "villaviciosa_water_background.tif"
NPZ = ROOT / "products_marea" / "marea.npz"
RESULT = ROOT / "products_marea" / "result.json"
OVERPASS = ROOT / "products_marea" / "overpass_times.json"
GRID_TIF = ROOT / "products_villaviciosa" / "hsr_eot20_2023-2025.tif"
DPI = 200

# Palette of the first author (also the poster's natural palette)
DEEP = "#2F5D7C"        # deep water; first of two compared series
SHALLOW = "#8DB3C7"     # shallow water; water background of the maps
SAND = "#C9A66B"        # sand / mudflat
SAND_DARK = "#8C6D46"
SEAGRASS = "#5B7F3A"
PAPER = "#EDE8DC"       # land / paper
GREY = "#B8B2A3"        # light gray
ACCENT = "#B5523B"      # delay, highlights; second of two compared series
INK = "#2B2B2B"
MUTED = "#6B6B6B"       # secondary text
TEAL = "#1B7F86"        # the poster's teal (poster/tools/cg/b_common.py), middle of the elevation ramp


def tint(color, share):
    """``share`` of ``color`` over white, as an RGB tuple 0-1 (as the poster's figures do)."""
    c = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float)
    return tuple((share * c + (1 - share) * 255) / 255)


def ramp(name, colors, n=256):
    """A sequential color map through ``colors`` (palette colors only, no library default)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors, N=n)


#: The map ramps, all from the palette. Every ramp's light end stays clear of the paper land,
#: the shallow-water background and the gray of the pixels without a value.
RAMP_Z = ("z", [DEEP, TEAL, SAND])                              # elevation: low deep, high sand
RAMP_SIGMA = ("sigma", [tint(SAND, 0.6), SAND_DARK, "#4E3B26"])   # width: sharp light, gradual dark
RAMP_S = ("s", [tint(SEAGRASS, 0.45), SEAGRASS, "#2C3F1D"])      # distance from the mouth
RAMP_TAU = ("tau", [tint(ACCENT, 0.35), ACCENT, "#5E2418"])      # delay: the accent of the delay
RAMP_SSR = ("ssr", [DEEP, SHALLOW, PAPER])                       # misfit: small dark, large light

#: Synthetic mouth tide: constituent -> (amplitude m, speed deg/h, phase deg),
#: h(t) = sum A cos(speed * t - phase), t in hours from T_REF (UTC). The
#: amplitudes are of the order of the Cantabrian coast's. The S2 phase puts
#: the S2 trough at 11:00 UTC, so spring low waters fall near the Sentinel-2
#: pass, as the real record of Villaviciosa suggests (result.json
#: rango_marea_m: the mouth levels at its 97 image times span -2.17 to
#: +1.15 m); the other phases are arbitrary fixed values.
SYNTH = {"M2": (1.30, 28.9841042, 40.0), "S2": (0.45, 30.0, 150.0),
         "N2": (0.27, 28.4397295, 15.0), "K1": (0.07, 15.0410686, 70.0),
         "O1": (0.06, 13.9430356, 320.0)}
T_REF = np.datetime64("2023-01-01T00:00:00")
SEED = 20261026


# ─────────────────────────────────────────────────────────────────────────────
#  Style, data access and small helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_style():
    """A plain journal look: thin axes, no top/right spines, no grid, DejaVu Sans."""
    from cycler import cycler
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9.5, "axes.titlelocation": "left", "axes.titlepad": 7,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "lines.linewidth": 1.2, "mathtext.fontset": "dejavusans",
        "image.interpolation": "nearest", "savefig.facecolor": "white",
        "axes.prop_cycle": cycler(color=[DEEP, ACCENT, SAND_DARK, SEAGRASS]),
    })


def save(fig, name):
    """Write docs/figures/doc/<name>.png at DPI and report its size."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  {path.relative_to(ROOT).as_posix()}  {path.stat().st_size / 1e3:.0f} kB")


def note(fig, text, y=-0.01):
    """Source / settings line under a figure, in the secondary text color."""
    fig.text(0.0, y, text, fontsize=7.5, color=MUTED, ha="left", va="top", wrap=True)


def load_product():
    """(arrays of marea.npz, result.json) of the Villaviciosa product."""
    with np.load(NPZ) as f:
        d = {k: f[k] for k in f.files}
    return d, json.loads(RESULT.read_text(encoding="utf-8"))


def acquisitions(first="2023", last="2025"):
    """Acquisition time (UTC) of every catalogued Sentinel-2 scene of the site in [first, last]."""
    ot = json.loads(OVERPASS.read_text(encoding="utf-8"))
    t = [np.datetime64(v.replace(" ", "T")) for k, v in ot.items() if first <= k[:4] <= last]
    return np.array(sorted(t), dtype="datetime64[s]")


def record_times():
    """Acquisition times of the 97 scenes of the product's record (result.json record_dates)."""
    ot = json.loads(OVERPASS.read_text(encoding="utf-8"))
    _, r = load_product()
    return np.array([np.datetime64(ot[s].replace(" ", "T")) for s in r["record_dates"]],
                    dtype="datetime64[s]")


def synth_tide(t):
    """The synthetic mouth tide (m) at the datetime64 instants ``t``."""
    hours = (np.asarray(t, "datetime64[s]") - T_REF) / np.timedelta64(1, "h")
    return sum(a * np.cos(np.deg2rad(w * hours - g)) for a, w, g in SYNTH.values())


def synth_text():
    """The settings of SYNTH, for the figure notes."""
    amps = ", ".join(f"{k} {a:.2f} m" for k, (a, _, _) in SYNTH.items())
    return f"synthetic tide ({amps}; S2 trough at 11:00 UTC)"


def phi(x):
    """Standard normal CDF."""
    from scipy.special import erf
    return 0.5 * (1.0 + erf(np.asarray(x, float) / np.sqrt(2.0)))


def thousands(n):
    return f"{int(n):,}"


def signed(text):
    """The same minus sign in labels as on the axes (U+2212)."""
    return text.replace("-", "−")


# --- Villaviciosa maps -------------------------------------------------------
def villa_grid():
    """(x0, y0, pixel, H, W) of the canonical 10 m grid (EPSG:32630), from the GeoTIFF header."""
    import rasterio
    with rasterio.open(GRID_TIF) as f:          # header only; the values are not used
        t = f.transform
        return t.c, t.f, t.a, f.height, f.width


def on_grid(values, d):
    """Values given on the product's intertidal pixels, placed on the (H, W) frame (NaN elsewhere)."""
    h, w = (int(v) for v in d["shape"])
    img = np.full(h * w, np.nan)
    img[d["keep"]] = values
    return img.reshape(h, w)


def map_extent_km():
    x0, y0, px, h, w = villa_grid()
    return (x0 / 1e3, (x0 + w * px) / 1e3, (y0 - h * px) / 1e3, y0 / 1e3)


def draw_background(ax, d):
    """Land in paper, water in the shallow-water color, intertidal pixels in light gray
    (they show wherever the product's array is NaN)."""
    from matplotlib.colors import to_rgb
    h, w = (int(v) for v in d["shape"])
    rgb = np.empty((h, w, 3))
    rgb[:] = to_rgb(PAPER)
    if BACKGROUND.exists():
        import rasterio
        with rasterio.open(BACKGROUND) as f:
            rgb[f.read(1) == 1] = to_rgb(SHALLOW)
    inter = np.zeros(h * w, bool)
    inter[d["keep"]] = True
    rgb[inter.reshape(h, w)] = to_rgb(GREY)
    ax.imshow(rgb, extent=map_extent_km(), zorder=0)


#: Map window (km, UTM 30N) around the ría; it leaves out MAP_OUTSIDE scattered up-valley pixels
MAP_CROP = (302.6, 310.17, 4817.7, 4824.84)


def outside_crop(d):
    """(pixels, pixels with an elevation) of the product outside MAP_CROP."""
    x0, y0, px, h, w = villa_grid()
    keep = d["keep"]
    xs, ys = (x0 + (keep % w + 0.5) * px) / 1e3, (y0 - (keep // w + 0.5) * px) / 1e3
    out = (xs < MAP_CROP[0]) | (xs > MAP_CROP[1]) | (ys < MAP_CROP[2]) | (ys > MAP_CROP[3])
    return int(out.sum()), int(np.isfinite(d["z"][out]).sum())


def map_frame(ax, left=True, bottom=True, bar_km=1.0):
    """Axes in UTM 30N km over MAP_CROP, a scale bar and a north mark (grid north)."""
    e = MAP_CROP
    ax.set_xlim(e[0], e[1])
    ax.set_ylim(e[2], e[3])
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.set_xticks(np.arange(304, e[1], 2))
    ax.set_yticks(np.arange(4818, e[3], 2))
    ax.tick_params(labelsize=7.5, labelbottom=bottom, labelleft=left)
    if bottom:
        ax.set_xlabel("UTM 30N easting (km)", fontsize=8)
    if left:
        ax.set_ylabel("northing (km)", fontsize=8)
    x, y = e[1] - 0.45 - bar_km, e[2] + 0.4          # bottom right: open land on this map
    ax.plot([x, x + bar_km], [y, y], color=INK, lw=1.6, solid_capstyle="butt")
    ax.text(x + bar_km / 2, y + 0.15, f"{bar_km:g} km", ha="center", va="bottom", fontsize=7.5)
    xn, yn = e[0] + 0.55, e[3] - 1.25
    ax.annotate("", xy=(xn, yn + 0.75), xytext=(xn, yn),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=8))
    ax.text(xn, yn + 0.85, "N", ha="center", va="bottom", fontsize=8)


def mouth_xy(d):
    """Map position (km) of the pixel nearest the mouth (smallest s_km)."""
    x0, y0, px, h, w = villa_grid()
    i = int(np.nanargmin(d["s_km"]))
    k = int(d["keep"][i])
    return (x0 + (k % w + 0.5) * px) / 1e3, (y0 - (k // w + 0.5) * px) / 1e3


def section_edges(d, r):
    """Edges (km) of the 3-km sections as the product built them (merged sections included)."""
    s, b = d["s_km"], d["band"]
    nb = len(r["centros_km"])
    return [0.0] + [float(np.nanmin(s[b == k])) for k in range(1, nb)] + [float(r["s_max_km"])]


def colorbar(fig, im, ax, label, ticks=None, ticklabels=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=ticks)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5, width=0.5, length=2.5)
    cb.set_label(label, fontsize=8)
    if ticklabels is not None:
        cb.ax.set_yticklabels(ticklabels)
    return cb


# --- Site roles ----------------------------------------------------------------
def literal(path, name):
    """The literal value assigned to ``name`` at module level in ``path`` (not imported)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


#: Validation sites in the order of the poster, with the registry key and a short type
SITES = [("villaviciosa", "Villaviciosa", "ría"), ("ferrol", "Ferrol", "ría"),
         ("escalda", "Westerschelde", "estuary"), ("wadden", "Wadden (Vlie)", "tidal basin"),
         ("ems", "Ems-Dollard", "estuary")]
#: Elevation surveys drawn as an elevation check (keys of config.ELEVATION_TRUTH)
SURVEYS = {"lidar": "PNOA LiDAR", "vaklodingen": "Vaklodingen"}
#: The tide at the mouth (an input), docs/validation_v2_spec.md section 1.1, "Boundary members"
MOUTH_TIDE = {"villaviciosa": "EOT20", "ferrol": "EOT20 + Ferrol 1",
              "escalda": "EOT20 + Vlissingen,\nBreskens", "wadden": "EOT20 + Terschelling",
              "ems": "EOT20 + Borkum"}


def site_roles():
    """{key: (elevation check, number of delay-check gauges)} from experiments/benchmark."""
    truth = literal(ROOT / "experiments" / "benchmark" / "config.py", "ELEVATION_TRUTH")
    judges = literal(ROOT / "experiments" / "benchmark" / "judges.py", "HEADLINE")
    return {k: (", ".join(SURVEYS[s] for s in truth[k] if s in SURVEYS), len(judges[k]))
            for k, _, _ in SITES}


# ─────────────────────────────────────────────────────────────────────────────
#  The figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_sites_map():
    """The five validation sites on the Atlantic coast of Europe, and the role of each."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    from pyintertidal.sites import SITES as REGISTRY

    coast = gpd.read_file(COAST)
    roles = site_roles()
    fig = plt.figure(figsize=(9.4, 4.6))
    ax = fig.add_axes([0.0, 0.12, 0.45, 0.86])
    ax.set_facecolor(SHALLOW)
    coast[coast["layer"] == "land"].plot(ax=ax, color=PAPER, edgecolor=MUTED, linewidth=0.3, zorder=2)
    coast[coast["layer"] == "borders"].plot(ax=ax, color=GREY, linewidth=0.45, zorder=3)
    ax.set_xlim(-11.0, 10.0)
    ax.set_ylim(41.6, 56.0)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.8)))          # equirectangular, true scale at 48.8 N
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels(["10°W", "5°W", "0°", "5°E", "10°E"])
    ax.set_yticks([42, 46, 50, 54])
    ax.set_yticklabels(["42°N", "46°N", "50°N", "54°N"])
    ax.tick_params(labelsize=7.5)
    for name, (x, y) in {"Atlantic\nOcean": (-8.3, 47.6), "Bay of\nBiscay": (-4.3, 45.4),
                         "North Sea": (2.8, 55.1)}.items():
        ax.text(x, y, name, fontsize=8, style="italic", color=MUTED, ha="center", va="center", zorder=4)
    # label offsets (points) keep the two Wadden Sea sites apart; a thin halo keeps the labels
    # legible where they cross a coastline
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.4, foreground=PAPER)]
    offsets = {"villaviciosa": (0, -7, "center", "top"), "ferrol": (3, 6, "left", "bottom"),
               "escalda": (-6, 0, "right", "center"), "wadden": (-6, -1, "right", "center"),
               "ems": (0, 6, "center", "bottom")}
    for n, (key, name, _) in enumerate(SITES, 1):
        lon, lat = Polygon(REGISTRY[key]["polygon"]).centroid.coords[0]
        ax.plot(lon, lat, "o", ms=5.5, mfc=INK, mec="white", mew=0.8, zorder=5)
        dx, dy, ha, va = offsets[key]
        ax.annotate(f"{n}  {name}", (lon, lat), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=8.5, zorder=6, path_effects=halo)

    # the table of roles, booktabs style
    tx = fig.add_axes([0.50, 0.12, 0.50, 0.86])
    tx.axis("off")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    xc = (0.0, 0.34, 0.56, 0.74)                    # column starts
    cols = list(zip(xc, ("Site", "Elevation\ncheck", "Delay\ncheck", "Tide at the\nmouth (input)")))
    y_head, y0, dy = 0.93, 0.79, 0.142
    for x, head in cols:
        tx.text(x, y_head, head, fontsize=8.5, fontweight="bold", va="center")
    tx.plot([0, 1], [0.99, 0.99], color=INK, lw=0.8)
    tx.plot([0, 1], [0.865, 0.865], color=INK, lw=0.5)
    for n, (key, name, kind) in enumerate(SITES, 1):
        y = y0 - (n - 1) * dy
        elev, n_judges = roles[key]
        tx.text(0.0, y + 0.018, f"{n}  {name}", fontsize=8.5, va="center")
        tx.text(0.045, y - 0.032, kind, fontsize=7.5, color=MUTED, va="center")
        tx.text(xc[1], y, elev or "–", fontsize=8.5, va="center")
        tx.text(xc[2], y, f"{n_judges} gauge{'s' if n_judges != 1 else ''}" if n_judges else "–",
                fontsize=8.5, va="center")
        tx.text(xc[3], y, MOUTH_TIDE[key], fontsize=8, va="center", linespacing=1.15)
    tx.plot([0, 1], [y0 - 4 * dy - 0.075] * 2, color=INK, lw=0.8)
    tx.text(0.0, y0 - 4 * dy - 0.11,
            "Elevation check: an independent survey of the flats (PNOA: Spanish national airborne\n"
            "LiDAR; Vaklodingen: Rijkswaterstaat depth surveys). Delay check: tide gauges inside the\n"
            "estuary. Checks are never used to make the maps. Validation in progress: no results shown.",
            fontsize=7.5, color=MUTED, va="top", linespacing=1.3)
    note(fig, "Coastline: Natural Earth 1:10m (public domain). Sites: pyintertidal.sites (polygon centroids). "
              "Roles: experiments/benchmark/config.py (ELEVATION_TRUTH), judges.py (HEADLINE); "
              "docs/validation_v2_spec.md §1.1.", y=0.03)
    save(fig, "sites_map")


def fig_tide_delay_concept():
    """Mouth tide against the delayed tide inside, the Sentinel-2 looks, and the looks the mouth tide gets wrong."""
    tau = np.timedelta64(60, "m")
    z = 0.50
    acq = acquisitions()
    wrong = (synth_tide(acq) > z) != (synth_tide(acq - tau) > z)
    # first 3-day window (from midnight) of 2024 holding a contradicted look and a correct one
    t0 = None
    for i in np.where(wrong & (acq >= np.datetime64("2024-01-01")))[0]:
        start = acq[i].astype("datetime64[D]") - np.timedelta64(1, "D")
        inside = (acq >= start) & (acq < start + np.timedelta64(3, "D"))
        if (inside & ~wrong).any():
            t0 = start.astype("datetime64[s]")
            break
    t = np.arange(t0, t0 + np.timedelta64(3, "D") + np.timedelta64(5, "m"), np.timedelta64(5, "m"))
    hm, hi = synth_tide(t), synth_tide(t - tau)

    fig = plt.figure(figsize=(9.2, 6.4))
    gs = GridSpec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.22, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    td = t.astype("datetime64[ms]").astype(object)
    top = 1.9                                           # gray bands stop here; labels live above
    ax.fill_between(td, -2.6, top, where=(hm > z) != (hi > z), color=GREY, alpha=0.45, lw=0, step="mid")
    ax.plot(td, hm, color=DEEP, lw=1.4)
    ax.plot(td, hi, color=ACCENT, lw=1.4)
    ax.axhline(z, color=INK, lw=0.7, ls=(0, (4, 3)))
    ax.set_ylim(-2.6, 3.0)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlim(td[0], td[-1])
    ax.set_ylabel("water level (m)")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_title("(a) Three days of tide (synthetic) and the Sentinel-2 passes over the site (real times, UTC)")
    # curve labels in the clear strip at the top left (the passes come later in the window)
    ax.text(0.01, 0.97, "mouth  h(t)", transform=ax.transAxes, color=DEEP, va="top", fontsize=8.5)
    ax.text(0.01, 0.89, f"inside  h(t − τ),  τ = {tau.astype(int)} min", transform=ax.transAxes,
            color=ACCENT, va="top", fontsize=8.5)
    ax.text(1.008, z, f"ground z\n= {z:.2f} m", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=8)
    for ta, bad in zip(acq, wrong):
        if not (t[0] <= ta <= t[-1]):
            continue
        x = ta.astype("datetime64[ms]").astype(object)
        ax.plot([x, x], [-2.6, 2.95], color=MUTED, lw=0.6, zorder=1)
        ax.plot(x, synth_tide(ta), "o", ms=5, mfc="white", mec=DEEP, mew=1.1, zorder=4)
        ax.plot(x, synth_tide(ta - tau), "o", ms=5, color=ACCENT, zorder=4)
        lab = f" Sentinel-2 {str(ta)[11:16]}" + ("\n wrong state" if bad else "\n right state")
        ax.text(x, 2.95, lab, ha="left", va="top", fontsize=7.5, color=ACCENT if bad else MUTED)

    # (b), (c): every look of the record, by the level it is given
    rec = record_times()
    h_mouth, h_in = synth_tide(rec), synth_tide(rec - tau)
    wet = h_in > z                                      # the true state: water inside above the ground
    bad = (h_mouth > z) != wet
    jitter = np.random.default_rng(SEED).uniform(-0.09, 0.09, len(rec))
    for col, (x, title) in enumerate([(h_mouth, "(b) Each look given the mouth level h(t)"),
                                      (h_in, "(c) Each look given the level inside h(t − τ)")]):
        a = fig.add_subplot(gs[1, col])
        mis = (x > z) != wet
        a.axvline(z, color=INK, lw=0.7, ls=(0, (4, 3)))
        a.scatter(x[~wet], jitter[~wet], s=16, color=SAND, edgecolor=SAND_DARK, lw=0.4, zorder=3)
        a.scatter(x[wet], 1 + jitter[wet], s=16, color=DEEP, edgecolor="white", lw=0.4, zorder=3)
        a.scatter(x[mis], (wet[mis] + jitter[mis]), s=58, facecolor="none", edgecolor=ACCENT, lw=1.1, zorder=4)
        a.set_yticks([0, 1])
        a.set_yticklabels(["dry", "wet"])
        a.set_ylim(-0.45, 1.45)
        a.set_xlim(-2.3, 1.5)
        a.set_xlabel("water level given to the look (m)")
        a.set_title(title)
        msg = (f"{mis.sum()} of {len(rec)} looks contradict z (circled)" if mis.any()
               else f"0 of {len(rec)} looks contradict z")
        a.text(0.02, 0.5, msg, transform=a.transAxes, fontsize=8, color=ACCENT if mis.any() else INK,
               va="center")
    note(fig, f"The flat is wet when the water inside the estuary stands above it, h(t − τ) > z. Gray bands "
              f"in (a): times when the mouth tide gives the wrong state (one delay wide around each crossing of z); "
              f"open circles: the mouth level at a pass, filled: the level inside. "
              f"Looks in (b, c): the {len(rec)} scenes of the Villaviciosa record (products_marea, real acquisition "
              f"times); levels from a {synth_text()}.", y=0.02)
    save(fig, "tide_delay_concept")


def fig_sentinel_sampling():
    """The tide levels the satellite samples against the full distribution of the tide."""
    _, r = load_product()
    acq, rec = acquisitions(), record_times()
    t_full = np.arange(np.datetime64("2023-01-01T00:00"), np.datetime64("2026-01-01T00:00"),
                       np.timedelta64(10, "m")).astype("datetime64[s]")
    h_full, h_rec = synth_tide(t_full), synth_tide(rec)
    top, low = h_rec.max(), h_rec.min()

    fig = plt.figure(figsize=(9.2, 3.9))
    gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.09, figure=fig)
    ax = fig.add_subplot(gs[0])
    w0, w1 = np.datetime64("2023-06-01T00:00:00"), np.datetime64("2023-07-01T00:00:00")  # 7 kept scenes
    win = (t_full >= w0) & (t_full < w1)
    ax.plot(t_full[win].astype("datetime64[ms]").astype(object), h_full[win], color=SHALLOW, lw=0.8)
    a_in = acq[(acq >= w0) & (acq < w1)]
    r_in = rec[(rec >= w0) & (rec < w1)]
    ax.plot(a_in.astype("datetime64[ms]").astype(object), synth_tide(a_in), "o", ms=4.5,
            mfc="white", mec=DEEP, mew=1.0, label="every Sentinel-2 acquisition")
    ax.plot(r_in.astype("datetime64[ms]").astype(object), synth_tide(r_in), "o", ms=4.5,
            color=DEEP, label="kept in the record (cloud screening)")
    for lev in (top, low):
        ax.axhline(lev, color=INK, lw=0.6, ls=(0, (4, 3)))
    ax.set_ylim(-2.55, 2.95)
    ax.set_xlim(w0.astype(object), w1.astype(object))
    # no tick near the right end, where panel (b)'s first tick label sits
    ax.set_xticks([np.datetime64(f"2023-06-{d:02d}").astype(object) for d in (1, 8, 15, 22)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("2023 (UTC)")
    ax.set_ylabel("tide at the mouth (m)")
    ax.legend(loc="upper left", ncol=2, fontsize=8, borderaxespad=0.2)
    ax.set_title(f"(a) June 2023: every pass falls at about 11:10 UTC")

    b = fig.add_subplot(gs[1], sharey=ax)
    bins = np.arange(-2.5, 2.51, 0.25)
    b.hist(h_full, bins=bins, orientation="horizontal", color=SHALLOW,
           weights=np.full(len(h_full), 100 / len(h_full)))
    b.hist(h_rec, bins=bins, orientation="horizontal", histtype="step", color=DEEP, lw=1.2,
           weights=np.full(len(h_rec), 100 / len(h_rec)))
    for lev in (top, low):
        b.axhline(lev, color=INK, lw=0.6, ls=(0, (4, 3)))
    above, below = (h_full > top).mean() * 100, (h_full < low).mean() * 100
    b.text(0.98, top + 0.08, f"above the highest look:\n{above:.1f} % of the time",
           transform=b.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5)
    b.text(0.98, low - 0.08, f"below the lowest look: {below:.1f} %",
           transform=b.get_yaxis_transform(), ha="right", va="top", fontsize=7.5)
    b.tick_params(labelleft=False)
    b.set_xlabel("share per 25-cm class (%)")
    b.set_title("(b) All times against the looks")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SHALLOW),
                      plt.Line2D([], [], color=DEEP, lw=1.2)],
             labels=["every 10 min, 2023–2025", f"the {len(rec)} kept looks"],
             loc="upper right", fontsize=7.5, borderaxespad=0.1)
    lo_r, hi_r = r["rango_marea_m"]
    note(fig, f"{synth_text()[0].upper()}{synth_text()[1:]}, sampled at the real acquisition times of Villaviciosa "
              f"(overpass_times.json; {len(acq)} acquisitions 2023–2025, {len(rec)} kept). The real record of the "
              f"product, EOT20 at the {len(rec)} image times (result.json rango_marea_m), spans "
              + signed(f"{lo_r:+.2f} to {hi_r:+.2f} m") + ": deep low waters are seen, the highest spring "
              "waters are not.", y=-0.03)
    save(fig, "sentinel_sampling")


def fig_lag_profile_villaviciosa():
    """The delay per 3-km section against distance from the mouth, fitted and applied."""
    from pyintertidal import marea
    d, r = load_product()
    c = np.asarray(r["centros_km"])
    tau, used = np.asarray(r["tau_min"]), np.asarray(r["tau_usado_min"])
    lo, hi = np.asarray(r["tau_lo_min"]), np.asarray(r["tau_hi_min"])
    boot = d["tau_boot_min"]
    edges = section_edges(d, r)
    s, tpx = d["s_km"], d["tau_px_min"]
    fin = np.isfinite(s)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for k in range(len(c)):
        if k % 2 == 0:
            ax.axvspan(edges[k], edges[k + 1], color=PAPER, lw=0, zorder=0)
        n_px = int((d["band"] == k).sum())
        ax.text((edges[k] + edges[k + 1]) / 2, 31.2,
                f"section {k}\n{edges[k]:.0f}–{edges[k + 1]:.1f} km · {thousands(n_px)} px",
                ha="center", va="top", fontsize=7.5, color=MUTED)
    rng = np.random.default_rng(SEED)
    for k in range(len(c)):
        ax.plot(c[k] + rng.uniform(-0.12, 0.12, len(boot)), boot[:, k], "o", ms=2.6,
                color=GREY, zorder=2)
    ax.scatter(s[fin], tpx[fin], s=1.2, color=ACCENT, alpha=0.35, lw=0, zorder=3, rasterized=True)
    ax.errorbar(c, tau, yerr=[tau - lo, hi - tau], fmt="o", ms=6, color=DEEP, ecolor=DEEP,
                elinewidth=1.0, capsize=3, zorder=5)
    ax.plot(c, used, "o", ms=10, mfc="none", mec=DEEP, mew=0.8, zorder=5)
    ax.plot(0, 0, "o", ms=6, mfc="white", mec=INK, mew=1.1, zorder=6, clip_on=False)
    ax.axhline(marea.TAU_DETECT_MIN, color=INK, lw=0.5, zorder=1)
    # direct labels
    ax.text(0.1, -0.7, "τ = 0 at the mouth", fontsize=8, va="top")
    ax.text(c[2] + 0.22, tau[2], f"fitted delay of the section\n(95 % interval of {len(boot)} scene\n"
                                 "bootstrap fits, gray dots)", fontsize=8, va="center", color=DEEP)
    ax.text(6.15, 8.6, "applied to each pixel\n(smoothed, rounded to 1 min)", fontsize=8, color=ACCENT,
            ha="left", va="top")
    ax.set_xlim(-0.3, edges[-1] + 0.35)
    ax.set_ylim(-3.0, 31.5)
    ax.set_xlabel("distance from the mouth along the water, s (km)")
    ax.set_ylabel("delay behind the mouth, τ (min)")
    ax.set_title("Villaviciosa: the delay of the tide inside the ría")
    n_nopath = int((~fin).sum())
    note(fig, "Constraints of the fit: τ = 0 at the mouth; the section delays are cumulative non-negative "
              "increments, so the profile never decreases inland; no detection threshold "
              f"(marea.TAU_DETECT_MIN = {marea.TAU_DETECT_MIN:g} min), so the applied delays (open rings) are "
              f"the fitted ones. {thousands(n_nopath)} pixels with no water path to the mouth keep τ = 0. "
              "Data: products_marea/result.json (centros_km, tau_min, tau_usado_min, tau_lo_min, tau_hi_min) "
              "and marea.npz (tau_boot_min, s_km, tau_px_min).", y=0.0)
    save(fig, "lag_profile_villaviciosa")


def _fit_one(y, clear, h, grid):
    """The product's fit of one pixel: elevation._fit_block over (grid x marea.SG_GRID)."""
    from pyintertidal import marea
    from pyintertidal.elevation import _fit_block
    a, b, mu, sg, _, n = _fit_block(y[:, None], clear[:, None].astype(float), h, grid, marea.SG_GRID)
    return float(a[0]), float(b[0]), float(mu[0]), float(sg[0]), int(n[0])


def fig_flooding_curve():
    """The per-pixel model on synthetic looks: the fit, its grid search and the quality rule."""
    from pyintertidal import marea
    from pyintertidal.elevation import _fit_block
    p = inspect.signature(marea.invert_series).parameters
    min_obs, min_b, max_b, max_a, n_mu = (p[k].default for k in ("min_obs", "min_b", "max_b", "max_a", "mu_points"))
    q = marea.QUALITY_MIN_BRACKET
    rng = np.random.default_rng(SEED)
    h = synth_tide(record_times())              # the level of each look (the mouth tide read the pixel's delay earlier)
    grid = np.linspace(h.min(), h.max(), n_mu)  # candidate elevations: lowest to highest level of the record
    sg_grid = np.asarray(marea.SG_GRID)

    # pixel 1: a flat in mid range; pixel 2: near the top of the sampled range
    true = dict(z=0.30, s=0.15, a=-0.40, b=0.95, noise=0.08)
    clear = rng.random(len(h)) > 0.15
    y = true["a"] + true["b"] * phi((h - true["z"]) / true["s"]) + rng.normal(0, true["noise"], len(h))
    y[~clear] = rng.normal(0.05, 0.3, (~clear).sum())          # cloudy looks: weight 0 in the fit
    a, b, z, sg, n = _fit_one(y, clear, h, grid)
    below, above = int((clear & (h < z)).sum()), int((clear & (h > z)).sum())

    fig = plt.figure(figsize=(9.4, 6.4))
    # the bottom 24 % of the figure is kept for the guards lines and the source note
    gs = GridSpec(2, 2, width_ratios=[1.55, 1], hspace=0.62, wspace=0.28, bottom=0.24, top=0.95, figure=fig)
    ax = fig.add_subplot(gs[:, 0])
    hh = np.linspace(-2.2, 1.35, 400)
    ax.axvspan(z - sg, z + sg, color=GREY, alpha=0.3, lw=0)
    ax.axhline(a, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.axhline(a + b, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    dry = clear & (y <= 0)
    wetl = clear & (y > 0)
    ax.scatter(h[dry], y[dry], s=16, color=SAND, edgecolor=SAND_DARK, lw=0.4, zorder=3,
               label="clear look, NDWI ≤ 0 (dry)")
    ax.scatter(h[wetl], y[wetl], s=16, color=DEEP, edgecolor="white", lw=0.4, zorder=3,
               label="clear look, NDWI > 0 (wet)")
    ax.scatter(h[~clear], y[~clear], s=14, facecolor="none", edgecolor=GREY, lw=0.8,
               label="not clear (weight 0)")
    ax.plot(hh, a + b * phi((hh - z) / sg), color=INK, lw=1.5, zorder=4)
    ax.plot([z, z], [-1.0, a + b / 2], color=INK, lw=0.7)
    ax.plot(z, a + b / 2, "o", ms=5, color=INK, zorder=5)
    ax.set_xlim(hh[0], hh[-1])
    ax.set_ylim(-1.0, 1.4)
    ax.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax.set_xlabel("water level of the look, h(t − τ) (m)")
    ax.set_ylabel("water index NDWI")
    ax.set_title("(a) One pixel: its looks and the fitted flooding curve")
    ax.text(-2.15, 1.37, r"$\mathrm{NDWI} = a + b\,\Phi\!\left(\dfrac{h - z}{\sigma}\right)$",
            fontsize=10.5, va="top")
    ax.text(-2.15, a + b + 0.04, signed(f"a + b = {a + b:.2f}  (flooded)"), fontsize=8, va="bottom", color=MUTED)
    ax.text(1.3, a + 0.04, signed(f"a = {a:.2f}  (dry)"), fontsize=8, ha="right", va="bottom", color=MUTED)
    ax.text(z - 0.04, -0.97, signed(f"z = {z:+.2f} m\n(midpoint)"), fontsize=8, ha="right", va="bottom")
    ax.text(z + sg + 0.04, -0.97, f"gray band: z ± σ\nσ = {sg:.2f} m", fontsize=8, va="bottom", color=MUTED)
    ax.legend(loc="upper right", fontsize=7.5, handletextpad=0.3, borderaxespad=0.2)
    fig.text(0.0, 0.135,
             f"Guards of pixel (a): {n} clear looks (at least {min_obs}); b = {b:.2f} (between {min_b:g} and "
             f"{max_b:g}); |a| = {abs(a):.2f} (below {max_a:g}).\nQuality rule (since 2026-10-06): {below} clear "
             f"looks with the water below z and {above} above (at least {q} each): elevation kept.",
             fontsize=7.8, va="top")

    # (b) the grid search behind the fit: SSR of every (z, sigma) pair, a and b in closed form
    ssr = np.full((len(sg_grid), len(grid)), np.nan)
    for i, s_ in enumerate(sg_grid):
        for j, m_ in enumerate(grid):
            _, _, _, _, rmse, nn = _fit_block(y[:, None], clear[:, None].astype(float), h, [m_], [s_])
            if np.isfinite(rmse[0]):
                ssr[i, j] = rmse[0] ** 2 * nn[0]
    g = fig.add_subplot(gs[0, 1])
    dz = grid[1] - grid[0]
    zedges = np.concatenate([grid - dz / 2, [grid[-1] + dz / 2]])
    im = g.pcolormesh(zedges, np.arange(len(sg_grid) + 1) - 0.5, ssr, cmap=ramp(*RAMP_SSR),
                      norm=LogNorm(np.nanmin(ssr), np.nanmax(ssr)), shading="flat", rasterized=True)
    i0, j0 = np.unravel_index(np.nanargmin(ssr), ssr.shape)
    g.plot(grid[j0], i0, "s", ms=7, mfc="none", mec=ACCENT, mew=1.3)
    g.set_yticks(range(len(sg_grid)))
    g.set_yticklabels([f"{v:g}" for v in sg_grid], fontsize=7)
    g.set_ylabel("σ (m)")
    g.set_xlabel("candidate z (m)")
    g.set_title(f"(b) Grid search: {n_mu} z × {len(sg_grid)} σ")
    cb = colorbar(fig, im, g, "sum of squared residuals")
    cb.ax.tick_params(labelsize=7)

    # (c) the quality rule: a pixel near the top of the sampled range
    y2_true = dict(z=1.05, s=0.06)
    y2 = true["a"] + true["b"] * phi((h - y2_true["z"]) / y2_true["s"]) + rng.normal(0, true["noise"], len(h))
    c2 = np.ones(len(h), bool)
    a2, b2, z2, s2, n2 = _fit_one(y2, c2, h, grid)
    bel2, abv2 = int((c2 & (h < z2)).sum()), int((c2 & (h > z2)).sum())
    e = fig.add_subplot(gs[1, 1])
    d2, w2 = y2 <= 0, y2 > 0
    e.scatter(h[d2], y2[d2], s=12, color=SAND, edgecolor=SAND_DARK, lw=0.4)
    e.scatter(h[w2], y2[w2], s=12, color=DEEP, edgecolor="white", lw=0.4)
    e.plot(hh, a2 + b2 * phi((hh - z2) / s2), color=INK, lw=1.2)
    e.axvline(z2, color=INK, lw=0.7)
    e.set_xlim(hh[0], hh[-1])
    e.set_ylim(-1.0, 1.05)
    e.set_xlabel("water level of the look (m)")
    e.set_ylabel("NDWI")
    e.set_title("(c) Quality rule: a pixel near the top")
    e.text(-2.15, 0.95, f"{abv2} look{'s' if abv2 != 1 else ''} with the water above z\n"
                        f"(fewer than {q}): no elevation", fontsize=7.8, va="top", color=ACCENT)
    note(fig, f"Synthetic looks: levels from the {synth_text()} at the {len(h)} real acquisition times of the "
              f"Villaviciosa record; pixel (a) simulated with z = {true['z']} m, σ = {true['s']} m, "
              + signed(f"a = {true['a']}") + f", b = {true['b']}, NDWI noise {true['noise']} (seed {SEED}), "
              f"15 % of looks not clear; pixel (c) with z = {y2_true['z']} m, σ = {y2_true['s']} m. Fit: "
              "pyintertidal.elevation._fit_block as marea.invert_series calls it: for every (z, σ) on the grid, "
              "a and b by least squares in closed form; the pair with the smallest sum of squared residuals is kept. "
              "The quality rule applies since 2026-10-06 (products_marea was made without it).", y=0.065)
    save(fig, "flooding_curve")


def fig_product_maps_villaviciosa():
    """Every per-pixel map the Villaviciosa product stores: z, sigma, s_km, tau_px_min."""
    d, r = load_product()
    from pyintertidal import marea
    ext = map_extent_km()
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 8.4), gridspec_kw=dict(hspace=0.2, wspace=0.22))
    panels = [
        ("(a) Elevation z (EOT20 vertical scale)", d["z"], dict(cmap=ramp(*RAMP_Z), vmin=-2.2, vmax=1.2), "z (m)"),
        ("(b) Width of the flooding curve σ", d["sigma"], None, "σ (m)"),
        ("(c) Distance from the mouth along the water s", d["s_km"],
         dict(cmap=ramp(*RAMP_S), vmin=0, vmax=float(np.ceil(np.nanmax(d["s_km"])))), "s (km)"),
        ("(d) Delay applied to each pixel τ", d["tau_px_min"].astype(float), None, "τ (min)"),
    ]
    tau = d["tau_px_min"].astype(float)
    tau[~np.isfinite(d["s_km"])] = np.nan                      # no water path: shown as no value
    panels[3] = panels[3][:1] + (tau,) + panels[3][2:]
    for ax, (title, vals, kw, label) in zip(axs.ravel(), panels):
        draw_background(ax, d)
        img = on_grid(vals, d)
        if title.startswith("(b)"):
            sg = np.asarray(marea.SG_GRID)
            bounds = np.concatenate([[0], (sg[1:] + sg[:-1]) / 2, [0.8]])
            cmap = ListedColormap(ramp(*RAMP_SIGMA)(np.linspace(0, 1, len(sg))))
            im = ax.imshow(img, extent=ext, cmap=cmap, norm=BoundaryNorm(bounds, len(sg)), zorder=1)
            colorbar(fig, im, ax, label, ticks=sg, ticklabels=[f"{v:g}" for v in sg])
        elif title.startswith("(d)"):
            vmax = float(np.nanmax(tau))
            bounds = np.arange(-0.5, vmax + 1.5, 1)
            cmap = ListedColormap(ramp(*RAMP_TAU)(np.linspace(0, 1, len(bounds) - 1)))
            im = ax.imshow(img, extent=ext, cmap=cmap, norm=BoundaryNorm(bounds, len(bounds) - 1), zorder=1)
            colorbar(fig, im, ax, label, ticks=np.arange(0, vmax + 1, 4))
        else:
            im = ax.imshow(img, extent=ext, zorder=1, **kw)
            colorbar(fig, im, ax, label)
        if title.startswith("(c)"):
            edges = section_edges(d, r)[1:-1]
            ax.contour(on_grid(d["s_km"], d), levels=edges, colors="white", linewidths=1.3,
                       extent=ext, origin="upper", zorder=2)
        xm, ym = mouth_xy(d)
        ax.plot(xm, ym, "v", ms=6, color=INK, mec="white", mew=0.6, zorder=3)
        ax.text(xm - 0.25, ym - 0.15, "mouth", ha="right", va="top", fontsize=7.5, zorder=3)
        i = list(axs.ravel()).index(ax)
        map_frame(ax, left=i % 2 == 0, bottom=i >= 2)
        ax.set_title(title, fontsize=9)
    n_z = int(np.isfinite(d["z"]).sum())
    n_out, n_out_z = outside_crop(d)
    note(fig, f"Villaviciosa, 10 m pixels, {len(r['record_dates'])} Sentinel-2 scenes of 2023–2025, EOT20 at the "
              f"mouth (products_marea/marea.npz). {thousands(len(d['z']))} intertidal pixels, {thousands(n_z)} "
              "with an elevation; gray: intertidal pixels without a value (rejected by the guards in a, b; "
              "no water path to the mouth in c, d, where the delay is 0). White lines in (c): the edges of the "
              "3-km sections. Background: land (paper) and water (blue) from the Sentinel-2 water frequency of "
              f"the site. The window leaves out {n_out} scattered up-valley pixels ({n_out_z} with an elevation). "
              "Not mapped: emersion time and look counts, which the product does not store (emersion time "
              "needs a continuous tide series at the mouth, which is not in the repository).", y=0.06)
    save(fig, "product_maps_villaviciosa")


def fig_product_distributions():
    """Histogram of z, the hypsometric curve, the distribution of sigma and z along the channel."""
    from pyintertidal import marea
    d, r = load_product()
    z, sg, s = d["z"], d["sigma"], d["s_km"]
    ok = np.isfinite(z)
    px_km2 = 1e-4                                     # one 10 m pixel
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.6), gridspec_kw=dict(hspace=0.5, wspace=0.28))

    a = axs[0, 0]
    # z is one of the 50 candidate levels of the inversion: one bar per level
    lo_, hi_ = r["rango_marea_m"]
    levels = np.linspace(lo_, hi_, inspect.signature(marea.invert_series).parameters["mu_points"].default)
    step = levels[1] - levels[0]
    idx = np.clip(np.round((z[ok] - lo_) / step).astype(int), 0, len(levels) - 1)
    area = np.bincount(idx, minlength=len(levels)) * px_km2
    a.bar(levels, area, width=step * 0.8, color=DEEP)
    med = float(np.median(z[ok]))
    a.axvline(med, color=INK, lw=0.7, ls=(0, (4, 3)))
    a.text(med - 0.05, 0.98, signed(f"median\n{med:+.2f} m"), transform=a.get_xaxis_transform(), fontsize=8,
           ha="right", va="top")
    a.set_xlabel(f"elevation z (m, EOT20 scale): {len(levels)} levels {step * 100:.1f} cm apart")
    a.set_ylabel("area at the level (km²)")
    a.set_title("(a) Elevations of the mapped flats")

    b = axs[0, 1]
    b.fill_betweenx(d["hyp_z"], d["hyp_lo"], d["hyp_hi"], color=SHALLOW, lw=0)
    b.plot(d["hyp_area"], d["hyp_z"], color=DEEP, lw=1.5)
    hi = r["hipsometria"]
    b.text(0.04, 0.95, f"mapped area {hi['area_total_km2']:.2f} km²\nhypsometric integral {hi['integral']:.2f}",
           transform=b.transAxes, fontsize=8, va="top")
    b.text(0.55, 0.25, "pale band: 5–95 % of 30 curves\nwith each z moved by its σ\n(narrow: it hugs the curve)",
           transform=b.transAxes, fontsize=7.5, color=MUTED)
    b.set_xlabel("area below the elevation (km²)")
    b.set_ylabel("elevation z (m)")
    b.set_title("(b) Hypsometric curve")

    c = axs[1, 0]
    grid = np.asarray(marea.SG_GRID)
    share = np.array([np.isclose(sg[ok], v, atol=1e-4).mean() * 100 for v in grid])
    xs = np.arange(len(grid))
    c.bar(xs, share, width=0.6, color=DEEP)
    for x_, v in zip(xs, share):
        c.text(x_, v + 0.4, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    c.set_xticks(xs)
    c.set_xticklabels([f"{v:g}" for v in grid])
    c.set_xlabel("σ, width of the flooding curve (m): the 8 values of the grid")
    c.set_ylabel("share of mapped pixels (%)")
    c.set_title("(c) Width σ")

    e = axs[1, 1]
    edges = np.arange(0, np.nanmax(s) + 0.5, 0.5)
    mid, q1, q2, q3 = [], [], [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = ok & (s >= lo_) & (s < hi_)
        if m.sum() >= 50:
            mid.append((lo_ + hi_) / 2)
            q1.append(np.percentile(z[m], 25)); q2.append(np.median(z[m])); q3.append(np.percentile(z[m], 75))
    e.fill_between(mid, q1, q3, color=SHALLOW, lw=0)
    e.plot(mid, q2, color=DEEP, lw=1.5)
    for x_ in section_edges(d, r)[1:-1]:
        e.axvline(x_, color=GREY, lw=0.8)
    e.text(0.03, 0.97, "line: median; band: 25–75 %;\ngray lines: section edges", transform=e.transAxes,
           fontsize=7.5, color=MUTED, va="top")
    e.set_xlabel("distance from the mouth along the water, s (km)")
    e.set_ylabel("elevation z (m)")
    e.set_title("(d) Elevation along the channel (0.5-km classes)")
    note(fig, f"Villaviciosa MAREA product (products_marea/marea.npz: z, sigma, s_km, hyp_z, hyp_area, hyp_lo, "
              f"hyp_hi; result.json hipsometria). {thousands(ok.sum())} pixels with an elevation, 10 m each.", y=0.02)
    save(fig, "product_distributions")


def fig_affine_identifiability():
    """A tide scaled by alpha gives elevations scaled by alpha and the same emersion time."""
    worlds = [("A", 1.0, 1.0, 0.20, DEEP), ("B", 1.2, 1.2, 0.24, ACCENT)]   # name, gain, z, sigma
    t = np.arange(np.datetime64("2024-03-08T00:00"), np.datetime64("2024-03-10T00:05"),
                  np.timedelta64(5, "m")).astype("datetime64[s]")
    h = synth_tide(t)
    td = t.astype("datetime64[ms]").astype(object)
    t_all = np.arange(np.datetime64("2023-01-01T00:00"), np.datetime64("2026-01-01T00:00"),
                      np.timedelta64(10, "m")).astype("datetime64[s]")
    h_all = synth_tide(t_all)

    fig = plt.figure(figsize=(9.4, 4.7))
    gs = GridSpec(2, 2, height_ratios=[3.2, 0.9], width_ratios=[1.7, 1], hspace=0.12, wspace=0.55, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    strip = fig.add_subplot(gs[1, 0], sharex=ax)
    for i, (name, alpha, z, sig, col) in enumerate(worlds):
        level = alpha * h
        ax.plot(td, level, color=col, lw=1.4)
        ax.axhline(z, color=col, lw=0.8, ls=(0, (4, 3)))
        in_air = level < z                              # emersion
        strip.fill_between(td, 1 - i - 0.38, 1 - i + 0.38, where=in_air, color=SAND, lw=0, step="mid")
        strip.fill_between(td, 1 - i - 0.38, 1 - i + 0.38, where=~in_air, color=SHALLOW, lw=0, step="mid")
        share = (alpha * h_all < z).mean() * 100
        ax.text(1.01, z + (0.04 if i else -0.04), f"z = {z:.1f} m", transform=ax.get_yaxis_transform(),
                color=col, ha="left", va="bottom" if i else "top", fontsize=8)
        strip.text(1.01, 1 - i, f"emersion\ntime {share:.1f} %", transform=strip.get_yaxis_transform(),
                   va="center", fontsize=7.5, color=col, linespacing=1.0)
    for i, (name, alpha, z, sig, col) in enumerate(worlds):
        ax.text(0.01, 0.99 - 0.065 * i, f"world {name}: gain α = {alpha:.1f}, z = {z:.1f} m, σ = {sig:.2f} m",
                transform=ax.transAxes, color=col, fontsize=8, va="top")
    ax.set_ylabel("water level over the pixel (m)")
    ax.set_ylim(-2.4, 3.0)
    ax.tick_params(labelbottom=False)
    ax.set_title("(a) Tide scaled by the gain α: elevations scale, emersion time does not")
    strip.set_yticks([1, 0])
    strip.set_yticklabels(["A", "B"])
    strip.set_ylim(-0.6, 1.6)
    strip.spines["left"].set_visible(False)
    strip.tick_params(axis="y", length=0)
    strip.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    strip.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

    p = fig.add_subplot(gs[:, 1])
    hh = np.linspace(-0.2, 2.0, 300)
    for (name, alpha, z, sig, col), lw, ls in zip(worlds, (3.0, 1.2), ("-", (0, (4, 3)))):
        p.plot(hh, phi((alpha * hh - z) / sig), color=col, lw=lw, ls=ls)
    p.set_xlabel("mouth tide h (m)")
    p.set_ylabel("probability of seeing the pixel wet")
    p.set_title("(b) Identical wet/dry probabilities")
    p.text(0.03, 0.97, r"$\dfrac{(c\alpha)h - cz}{c\sigma} = \dfrac{\alpha h - z}{\sigma}$",
           transform=p.transAxes, fontsize=10, va="top")
    p.text(1.2, 0.45, "A (solid) and\nB (dashed)\ncoincide", fontsize=8, va="center")
    p.set_ylim(-0.03, 1.03)
    note(fig, "Strips: sand, the pixel in the air; blue, under water. Emersion time: the share of three years of "
              "the same tide that the pixel spends in the air. "
              "Both worlds give the same wet/dry record from any set of images, so the gain α cannot be read "
              "from the imagery: it comes with the tide at the mouth, and only the delay τ is fitted "
              f"(docs/METHOD.md, \"What is provably not estimable\"). Mouth tide: the {synth_text()}.", y=0.0)
    save(fig, "affine_identifiability")


def fig_screening_timeline():
    """Scenes per month in 2023-2025: every acquisition against the screened record."""
    _, r = load_product()
    acq = acquisitions()
    months = np.arange(np.datetime64("2023-01"), np.datetime64("2026-01"), np.timedelta64(1, "M"))
    m_all = acq.astype("datetime64[M]")
    m_rec = np.array([np.datetime64(s[:7]) for s in r["record_dates"]])
    n_all = np.array([(m_all == m).sum() for m in months])
    n_rec = np.array([(m_rec == m).sum() for m in months])
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.bar(x, n_all, width=0.72, color=GREY, label="every Sentinel-2 acquisition over the site")
    ax.bar(x, n_rec, width=0.72, color=DEEP, label="kept after cloud screening (the record used)")
    for yr in range(3):
        if yr:
            ax.plot([12 * yr - 0.5] * 2, [0, 18.6], color=INK, lw=0.5)
        sl = slice(12 * yr, 12 * yr + 12)
        ax.text(12 * yr + 5.5, 17.0, f"{2023 + yr}: {n_all[sl].sum()} acquisitions, {n_rec[sl].sum()} kept",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m)[5:7] for m in months], fontsize=7)
    ax.set_xlim(-0.7, len(x) - 0.3)
    ax.set_ylim(0, 23.5)
    ax.set_xlabel("month (2023–2025)")
    ax.set_ylabel("scenes per month")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title("Villaviciosa: the Sentinel-2 archive and the scenes MAREA used")
    note(fig, f"Acquisitions: products_marea/overpass_times.json ({len(acq)} in 2023–2025). Kept: "
              f"products_marea/result.json record_dates ({len(m_rec)} scenes that passed the cloud screening of "
              "the site). A kept scene still contributes only its cloud-free pixels.", y=-0.04)
    save(fig, "screening_timeline")


def fig_sections_villaviciosa():
    """The 3-km sections of distance along the water over the Villaviciosa map, with their delays."""
    d, r = load_product()
    ext = map_extent_km()
    edges = section_edges(d, r)
    nb = len(r["centros_km"])
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    draw_background(ax, d)
    cols = ramp(*RAMP_S)(np.linspace(0.1, 1.0, nb))          # the ramp of the distance map, in steps
    band = d["band"].astype(float)
    band[band < 0] = np.nan
    ax.imshow(on_grid(band, d), extent=ext, cmap=ListedColormap(cols),
              norm=BoundaryNorm(np.arange(-0.5, nb), nb), zorder=1)
    ax.contour(on_grid(d["s_km"], d), levels=edges[1:-1], colors=INK, linewidths=0.6, extent=ext,
               origin="upper", zorder=2)
    # labels on land beside each section (map km, alignment), placed by eye for this map
    spots = [(305.9, 4823.0, "right"), (307.85, 4821.0, "left"), (305.1, 4819.2, "left")]
    for k in range(nb):
        xl, yl, ha = spots[k] if k < len(spots) else (MAP_CROP[0] + 0.3, MAP_CROP[2] + 1.5 + k, "left")
        ax.text(xl, yl, f"section {k}  ({edges[k]:.0f}–{edges[k + 1]:.1f} km)\n"
                        f"τ = {r['tau_usado_min'][k]:.0f} min  [{r['tau_lo_min'][k]:.0f}–{r['tau_hi_min'][k]:.0f}]",
                ha=ha, va="center", fontsize=8, zorder=4)
    xm, ym = mouth_xy(d)
    ax.plot(xm, ym, "v", ms=7, color=INK, mec="white", mew=0.6, zorder=3)
    ax.text(xm - 0.25, ym - 0.15, "mouth, s = 0", ha="right", va="top", fontsize=8, zorder=3)
    map_frame(ax)
    ax.set_title("Villaviciosa: the 3-km sections that share one delay")
    note(fig, "Colors: the section of each intertidal pixel (marea.npz band); black lines: s = 3 and 6 km "
              "(marea.npz s_km, distance along the water from the mouth; sections numbered from 0 as in the "
              "code; the last 0.15 km joins section 2). "
              "τ: the delay applied to the section, with its 95 % bootstrap interval (result.json). Gray: pixels "
              "with no water path to the mouth.", y=0.04)
    save(fig, "sections_villaviciosa")


FIGURES = {
    "sites_map": fig_sites_map,
    "tide_delay_concept": fig_tide_delay_concept,
    "sentinel_sampling": fig_sentinel_sampling,
    "lag_profile_villaviciosa": fig_lag_profile_villaviciosa,
    "flooding_curve": fig_flooding_curve,
    "product_maps_villaviciosa": fig_product_maps_villaviciosa,
    "product_distributions": fig_product_distributions,
    "affine_identifiability": fig_affine_identifiability,
    "screening_timeline": fig_screening_timeline,
    "sections_villaviciosa": fig_sections_villaviciosa,
}


# ─────────────────────────────────────────────────────────────────────────────
#  The two small data files of docs/figures/data/
# ─────────────────────────────────────────────────────────────────────────────
def build_data(ne_dir, water_frequency_tif):
    """Write the Natural Earth subset and the Villaviciosa land/water background.

    * ne_10m_land_europe_atlantic.geojson: Natural Earth 1:10m land and land
      borders clipped to 14 W - 13 E, 40 - 58 N, simplified to 0.01 degree,
      coordinates to 0.001 degree (about 110 kB).
    * villaviciosa_water_background.tif: uint8 on the canonical 10 m grid,
      1 where the Sentinel-2 water frequency of the site is at least 0.5 and
      the pixel is not intertidal in the product, 0 elsewhere.
    """
    import geopandas as gpd
    import rasterio
    import shapely
    from shapely.geometry import box, mapping

    DATA.mkdir(parents=True, exist_ok=True)
    win = box(-14, 40, 13, 58)
    feats = []
    for layer, name in (("land", "ne_10m_land"), ("borders", "ne_10m_admin_0_boundary_lines_land")):
        g = gpd.read_file(f"zip://{Path(ne_dir) / (name + '.zip')}").to_crs(4326)
        geom = shapely.union_all(g.geometry.intersection(win).values)
        geom = shapely.set_precision(geom.simplify(0.01, preserve_topology=True), 0.001)
        feats.append({"type": "Feature", "properties": {"layer": layer}, "geometry": mapping(geom)})
    COAST.write_text(json.dumps({"type": "FeatureCollection", "features": feats,
                                 "source": "Natural Earth 1:10m physical/land and cultural/admin_0_boundary_lines_land "
                                           "(public domain), clipped and simplified by make_doc_figures.build_data"},
                                separators=(",", ":")), encoding="utf-8")
    d, _ = load_product()
    with rasterio.open(water_frequency_tif) as f:
        wf, prof = f.read(1), f.profile
    inter = np.zeros(wf.size, bool)
    inter[d["keep"]] = True
    water = (np.nan_to_num(wf, nan=0.0) >= 0.5) & ~inter.reshape(wf.shape)
    prof.update(dtype="uint8", nodata=None, compress="deflate", predictor=1, count=1)
    with rasterio.open(BACKGROUND, "w", **prof) as f:
        f.write(water.astype(np.uint8), 1)
        f.update_tags(source="1 = water: Sentinel-2 water frequency >= 0.5 (products_villaviciosa/"
                             "water_frequency.tif of the site notebook), outside the intertidal pixels of "
                             "products_marea; 0 = land. Contains modified Copernicus Sentinel data.")
    for p in (COAST, BACKGROUND):
        print(f"  {p.relative_to(ROOT).as_posix()}  {p.stat().st_size / 1e3:.0f} kB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", nargs="*", help=f"figure names (default: all): {', '.join(FIGURES)}")
    ap.add_argument("--list", action="store_true", help="print the figure names and exit")
    ap.add_argument("--build-data", nargs=2, metavar=("NE_DIR", "WATER_FREQUENCY_TIF"),
                    help="rebuild the two files of docs/figures/data/ and exit")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(FIGURES))
        return
    if args.build_data:
        build_data(*args.build_data)
        return
    unknown = [f for f in args.figures if f not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}")
    set_style()
    for name in args.figures or FIGURES:
        print(name)
        FIGURES[name]()


if __name__ == "__main__":
    main()
