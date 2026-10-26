# -*- coding: utf-8 -*-
"""CARTOGRASS 2026 poster, Materials and methods: "How MAREA works".

The whole method told as the journey of ONE intertidal pixel (concept A of
the 2026-10-18 design round, with the judges' grafts), in six cards read in
two rows of three. A legend above the cards names every symbol in words, with
units, so the reader meets h, t, tau and z before card 2 uses them. The two
inputs are joined by a "+" (they are independent: the tide at the mouth is not
derived from the images), the steps by arrows, and a return arrow runs from
card 3 down to card 4:

  1  Seen many times      INPUT 1 · IMAGES: the Sentinel-2 data cube (10-m
                          pixels, 2023-2025, openEO in the Copernicus cloud).
                          One look = one cloud-free view of the pixel; a water
                          index (green vs near-infrared) says wet or dry;
                          cloudy views are not looks
  2  Its tide arrives late INPUT 2 · TIDE AT THE MOUTH (the tab names it, so the
                          delay drawn on this card is not taken for an input:
                          card 3 finds it from the images; the tide's source,
                          global tide model EOT20 + tide gauges at the
                          mouth, is the Input data paragraph's, which carries
                          the same navy INPUT tags, \\cgInputTag). The REAL
                          Ems-Dollard outline cut into 3-km sections along the
                          channel from the mouth (tau = 0 there), the pixel
                          named on it ("pixel", a leader to its square); the
                          pixel's section sees the mouth tide tau later. At
                          the image time t the mouth tide is above the
                          pixel's ground z ("no delay: wet": the mouth tide
                          taken as it is) and its section's water below it:
                          "seen dry"; "image at time t" names the plot's axis.
                          Rule: wet when h(t - tau) > z, then in words: "the
                          mouth tide reaches its section tau later" (the two
                          curves) and "MAREA shifts the tide in time, not
                          height": no gain is fitted (wet/dry images time the
                          tide, they do not measure its height), so where the
                          tide grows taller inland z is a level on the mouth
                          tide's scale, hence "same scale as h"
  3  Its section's delay  one row per delay tried ("minutes, examples"),
                          named by bare numbers. A key names the dots: each is
                          one look of THIS pixel, sand seen dry, teal seen
                          wet, placed at its water level h(t - tau) under that
                          row's delay. One dashed line, "ground z" over its
                          top, runs through every row: under the wrong delays
                          dry and wet looks fall on both sides of it
                          ("mixed"), under the best one they split at it
                          ("clean"). The dots are a subset of its looks chosen
                          to keep the record's mix (simulate(): sub3). The
                          "fit" bars, named over their column, score each
                          delay on the looks of the section's pixels together
                          (simulate(): tried_ll), as MAREA fits one delay per
                          section: "fit: judged on its section's pixels"
                          (production scores a sample of the section's pixels,
                          not all of them); the longest bar is the delay
                          chosen, "best delay: cleanest split". (MAREA also
                          pins tau = 0 at the mouth and lets it only grow
                          inland; since 2026-10-20 the card leaves that out:
                          its line gave way to the key of the dots)
  4  Its elevation        flooding curve of the water index on the corrected
                          tide: midpoint = elevation z, width = uncertainty;
                          a pixel needs wet and dry looks, otherwise it is
                          left empty
  5  Its emerged time     E(z) = share of time its water is below z,
                          h(t - tau) < z, at all times of the corrected tide
                          record, not only at image times (pyintertidal/
                          hydroperiod.py: a continuous tide series, not the
                          satellite dates, which sample the tide unevenly):
                          four days of its water level, and the share of the
                          whole record out of / under the water
  6  All pixels: the maps OUTPUT: elevation with its uncertainty, and emerged
                          time: the base layer for the potential habitat of
                          Zostera noltei (ecologists set the habitat limits,
                          not the method)

What is real and what is a schematic
    REAL       only the Ems-Dollard outline of card 2 and its 3-km sections:
               products_ems/water_frequency.tif (the Sentinel-2 share of
               clear looks in which each pixel was wet; 20 m), classed as
               pyintertidal.marea.extract does (flats 0.10-0.90, permanent
               water >= 0.90); the sections are recomputed the way MAREA
               defines them (pyintertidal.geometry.mouth_seeds and
               along_distance, imported read only: the permanent water that
               touches the north or the west edge is the mouth, the Ems's
               production MOUTH_SIDE "north+west" of
               experiments/benchmark/config.py; s = distance from it through
               water and flats; section k = [3k, 3k + 3) km). No lag,
               elevation or other result of any product is read.
    SCHEMATIC  everything else. The pixel, its looks and its section are
               SIMULATED from the method's own physics (tide at the mouth =
               a few tidal constituents; water at the pixel = that tide tau
               later; water index = dry level + jump x flooding curve +
               noise; Sentinel-2-like overpasses near 11:00 UTC every few
               days of 2023-2025, with clouds). The delay of card 3 is found
               from the simulated section with the method's own criterion
               (the wet/dry looks of its 300 pixels, each pixel with its best
               elevation and width), so the picture behaves as the method
               does: planted 90 min, recovered 90 min. The delays shown are
               examples, not results; card 3 says "examples" and the panel
               title "Schematic". Card 6's layers are made-up surfaces on a
               schematic estuary.

Where the imaginary pixel sits (2026-10-19): section 11 of the real outline
(33-36 km, the inner Dollard), not Emden's section 10. MAREA's delay is the
shift that aligns the whole tide curve; from the gauge records behind b1
(poster/build/cg/b1_tide_arrives_late.json) that shift is about 70 min at
Emden (section 10; its high waters +86 min, the number b1 prints) and about
96 min at Nieuwe Statenzijl (section 12, a sliver on this outline), so a
90-min example belongs between them, where no gauge number is printed on the
sheet and the example cannot be read as a check of one. (The Pogum gauge,
35.8 km from the mouth, also lies in section 11; no Pogum number is printed
on the sheet either.) Round 3 (2026-10-19): with the production mouth
("north+west") only the distances near the mouth change (Eemshaven harbour
6.0 -> 5.0 km); every inner section, section 11 included, keeps its pixels.

The method is drawn as the brief states it: the pixel's water is the mouth
tide shifted by its section's delay. (pyintertidal/marea.py smooths the
section delays along the channel and gives each pixel the value at its own
distance from the mouth; the step picture is that profile's coarse form.)

Drawn at print size: the right half of the Materials-and-methods row,
391 mm wide; the height is the slot's (--height, default 244.6 mm: \\figFlowH of
poster/cartograss2026.tex, the row's 264 mm less the panel title). Type: card
titles 28 pt, key lines and legend symbols 24-26 pt, text and labels 22 pt,
tags and small labels 20 pt (nothing under 20 pt). A built-in check prints any
text that leaves its card or the figure, or overlaps another text.

Run from the repository root (about 20 s, peak about 0.4 GB; the Ems
sections are cached in poster/build/cg/method_flow_ems.npz):

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/fig_method_flow.py
        [--width 391] [--height 244.6] [--no-png]

Writes poster/figures/cg/method_flow.pdf (vector, Fira Sans embedded as
TrueType through b_common.use_poster_fonts) and method_flow.png (300 dpi,
rasterised by MiKTeX's pdftocairo, as the other figures of the poster).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, "pyintertidal")):
    sys.exit("run this script from the repository root")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "poster", "tools", "cg"))

import matplotlib                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap               # noqa: E402
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Polygon, Rectangle)
from matplotlib.path import Path                                     # noqa: E402
from scipy.optimize import curve_fit                                 # noqa: E402
from scipy.stats import norm                                         # noqa: E402

import b_common as bc                                                # noqa: E402

INK, SLATE, DEEP, TEAL, SAND = bc.INK, bc.SLATE, bc.DEEP, bc.TEAL, bc.SAND
MIST, RULE = bc.MIST, bc.RULE
MIKTEX = bc.MIKTEX_BIN
OUT = os.path.join("poster", "figures", "cg", "method_flow")
EMS_WF = os.path.join("products_ems", "water_frequency.tif")
EMS_CACHE = os.path.join("poster", "build", "cg", "method_flow_ems.npz")
EMS_MOUTH_SIDE = "north+west"     # experiments/benchmark/config.py MOUTH_SIDE["ems"]
SECTION_KM = 3.0
PT = 25.4 / 72                                  # mm per point


def tint(colour, share):
    """``share`` of the colour over white, as an RGB tuple 0-1."""
    c = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    return tuple((share * c + (1 - share) * 255) / 255)


WATER = tint(DEEP, 0.15)          # the study-site map's water
FLATS = tint(SAND, 0.55)          # and its intertidal flats
TEAL_PALE = tint(TEAL, 0.16)
SAND_DARK = "#8A5F22"             # PSand darkened for text on white (b2's)

# ═════════════════════════════════════════════════════════════════════════════
#  The simulated world (schematic)
# ═════════════════════════════════════════════════════════════════════════════
M2, S2, N2, K1 = 12.4206012, 12.0, 12.65834751, 23.93447213      # periods, h
TAU_H = 1.5                      # the section's delay behind the mouth, h (planted)
Z0, SIG0, A0, B0 = 0.40, 0.15, -0.30, 0.85   # the pixel: elevation, width, dry level, jump
TRIED_MIN = (0, 45, 90, 135)                 # delays shown in card 3 (examples)
PIX_SECTION = 11                             # the pixel's section on the real outline:
                                             # 33-36 km, inner Dollard (module docstring)
N_LOOKS_ROW = 38                             # looks drawn per row of card 3 (fewer, larger dots)


def tide_mouth(t_h):
    """Tide at the mouth (m above mean sea level) at time t (hours)."""
    w = 2 * np.pi
    return (1.20 * np.cos(w * t_h / M2) + 0.40 * np.cos(w * t_h / S2 - 0.7)
            + 0.24 * np.cos(w * t_h / N2 - 2.0) + 0.06 * np.cos(w * t_h / K1 - 0.3))


def flooding(h, a, b, z, s):
    """The flooding curve a + b * Phi((h - z) / s): water index against level."""
    return a + b * norm.cdf((h - z) / s)


def simulate(seed=20261018):
    """The imaginary pixel and its section, simulated: overpass times, the
    pixel's looks, the section's likelihood for every delay tried, the pixel's
    best ground under each, the flooding-curve fit on the corrected tide and
    the emerged time. Returns everything the cards draw (a dict)."""
    rng = np.random.default_rng(seed)
    # overpasses: two satellites, each every 5 days, about 11:00 UTC, 3 years
    days = np.sort(np.concatenate([np.arange(0, 1096, 5), np.arange(2, 1096, 5)]))
    t = days * 24.0 + 11.0 + rng.normal(0.0, 0.08, days.size)
    n = t.size
    scene_clear = rng.beta(0.55, 0.55, n)            # some scenes clear, some cloudy

    # --- the pixel -----------------------------------------------------------
    h_pix = tide_mouth(t - TAU_H)
    frac = norm.cdf((h_pix - Z0) / SIG0)
    y = A0 + B0 * frac + rng.normal(0.0, 0.055, n) * (1.0 + 1.3 * frac)
    clear = rng.random(n) < scene_clear
    wet = y > 0
    keep = np.flatnonzero(clear)
    if keep.size > 110:                                # a legible number of dots
        keep = np.sort(rng.choice(keep, 110, replace=False))
    look = np.zeros(n, bool)
    look[keep] = True

    # --- the section: its pixels share one delay --------------------------
    NP = 300
    zs = rng.uniform(-1.35, 1.5, NP)
    ss = rng.uniform(0.06, 0.25, NP)
    clear_s = rng.random((NP, n)) < scene_clear[None, :]
    wet_s = rng.random((NP, n)) < norm.cdf((h_pix[None, :] - zs[:, None]) / ss[:, None])
    ZG = np.linspace(-1.95, 1.95, 79)
    SG = np.array([0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45])
    Wm = (clear_s & wet_s).astype(float)
    Dm = (clear_s & ~wet_s).astype(float)

    def profile(delay_h, Wx, Dx):
        """Per pixel, the best log-likelihood of its wet (Wx) and dry (Dx) looks
        over the grid of grounds and widths for this delay, and where it is."""
        h = tide_mouth(t - delay_h)
        P = np.clip(norm.cdf((h[:, None, None] - ZG[None, :, None]) / SG[None, None, :]),
                    1e-6, 1 - 1e-6).reshape(n, -1)
        ll = Wx @ np.log(P) + Dx @ np.log1p(-P)       # (pixels, z x width)
        return ll.max(axis=1), ll.argmax(axis=1)

    delays_min = np.arange(0, 151, 5)
    sec_ll = np.array([profile(d / 60.0, Wm, Dm)[0].sum() for d in delays_min])
    best_min = int(delays_min[np.argmax(sec_ll)])
    tried_ll = np.array([profile(d / 60.0, Wm, Dm)[0].sum() for d in TRIED_MIN])

    # the pixel's own best ground line under each tried delay (its wet/dry looks)
    Wp = (look & wet).astype(float)[None, :]
    Dp = (look & ~wet).astype(float)[None, :]
    strip_z = []
    for d in TRIED_MIN:
        _, arg = profile(d / 60.0, Wp, Dp)
        strip_z.append(ZG[arg[0] // SG.size])

    # --- the flooding curve on the corrected tide (card 4) ------------------
    p, _ = curve_fit(flooding, h_pix[look], y[look], p0=(-0.3, 0.8, 0.3, 0.2),
                     bounds=([-1, 0.1, -2, 0.03], [1, 2, 2, 1.0]))

    # --- emerged time: share of time with the corrected tide below z --------
    tt = np.arange(0.0, 1096 * 24.0, 1 / 6.0)          # 10-min steps, 3 years
    level = tide_mouth(tt - TAU_H)
    emerged = float(np.mean(level < p[2]))
    levels_sorted = np.sort(level[::6])

    # a fixed subset of the pixel's looks for card 3 (fewer, larger dots), chosen to
    # keep the whole record's mix (2026-10-20). At its ground z, of all 110 looks
    # 29/14/1/13 fall on the wrong side under 0/45/90/135 min (45 and 135 alike, as
    # their fit bars); seed 8 draws 12/6/0/6 of 38, with the record's share of wet
    # looks (45 % vs 44 %). Seed 7 drew 13/10/0/2, the least faithful of seeds 0-20:
    # its 135-min row looked split while its bar equals the 45-min one
    sub = np.flatnonzero(look)
    sub = np.sort(np.random.default_rng(8).choice(sub, min(N_LOOKS_ROW, sub.size), replace=False))

    return dict(t=t, h_pix=h_pix, y=y, wet=wet, look=look, clear=clear,
                delays_min=delays_min, sec_ll=sec_ll, best_min=best_min,
                tried_ll=tried_ll, strip_z=np.array(strip_z), fit=p,
                emerged=emerged, levels_sorted=levels_sorted, sub3=sub)


def spring_start():
    """Start of a 4-day window around the largest range of a 2-month stretch."""
    tt = np.arange(24 * 30.0, 24 * 90.0, 0.25)
    lev = tide_mouth(tt - TAU_H)
    day = (tt // 24).astype(int)
    days = np.unique(day)
    rng_ = np.array([np.ptp(lev[day == d]) for d in days])
    d_best = days[np.argmax(rng_)]
    return 24.0 * (d_best - 2) + 3.0


# --- a schematic estuary (km) for card 6: mouth at s = 0, sea to the left ------
def estuary_z(s, n):
    """Elevation (m) of the schematic estuary of card 6 at distance s along it
    and n across it (km), with the channel's centre line and the flats'
    half-width at s."""
    nc = 0.30 * np.sin(2 * np.pi * s / 7.0 + 0.6) * np.clip(s / 2.0, 0, 1)
    w = np.clip(1.32 - 0.050 * s, 0.55, None)
    c = np.clip(0.27 - 0.013 * s, 0.07, None)
    d = np.abs(n - nc)
    q = np.clip((d - c) / (w - c), 0, None)
    z = np.where(d < c, -2.6 + 0.03 * s, -1.80 + 3.80 * q ** 1.25)
    z = np.where(q > 1, 2.6 + 3 * (q - 1), z)
    z = z + 0.14 * np.sin(3.1 * s + 1.7 * n) * np.sin(1.3 * s - 2.2 * n + 0.4) * (q <= 1)
    return np.where(s < 0, -6.0, z), nc, w


# ═════════════════════════════════════════════════════════════════════════════
#  The real outline of card 2: Ems-Dollard in 3-km sections
# ═════════════════════════════════════════════════════════════════════════════
def ems_sections():
    """Section index k (-1 outside), class (1 permanent water, 2 flats), s (km)."""
    st = os.stat(EMS_WF)
    stamp = np.array([st.st_size, int(st.st_mtime)], np.int64)
    if os.path.exists(EMS_CACHE):
        d = np.load(EMS_CACHE)
        # a cache made from another mouth side (the north edge alone, before
        # 2026-10-19 round 3) is stale
        side_ok = "side" in d.files and str(d["side"]) == EMS_MOUTH_SIDE
        if np.array_equal(d["stamp"], stamp) and side_ok:
            return {k: d[k] for k in ("k", "cls", "s_km")}
    import rasterio

    from pyintertidal import geometry

    with rasterio.open(EMS_WF) as src:
        wf = src.read(1)
        res = abs(src.transform.a)
    fin = np.isfinite(wf)
    inter = fin & (wf > 0.10) & (wf < 0.90)            # pyintertidal.marea.extract's classes
    sea = fin & (wf >= 0.90)
    del wf, fin
    seeds = geometry.mouth_seeds(sea, side=EMS_MOUTH_SIDE)
    s_km = (geometry.along_distance(sea | inter, seeds, res) / 1000.0).astype(np.float32)
    del seeds
    est = np.isfinite(s_km)
    k = np.where(est, np.floor(np.nan_to_num(s_km, nan=0.0) / SECTION_KM), -1).astype(np.int16)
    cls = np.zeros(k.shape, np.uint8)
    cls[est & sea] = 1
    cls[est & inter] = 2
    os.makedirs(os.path.dirname(EMS_CACHE), exist_ok=True)
    np.savez_compressed(EMS_CACHE, k=k, cls=cls, s_km=s_km, stamp=stamp, side=EMS_MOUTH_SIDE)
    print(f"  Ems sections computed (mouth: the {EMS_MOUTH_SIDE} edges) and cached -> {EMS_CACHE}")
    return {"k": k, "cls": cls, "s_km": s_km}


# ═════════════════════════════════════════════════════════════════════════════
#  Drawing helpers (everything in mm, origin top-left, y downwards)
# ═════════════════════════════════════════════════════════════════════════════
class Sheet:
    """The W x H mm drawing surface (origin top-left, y downwards) and the
    registry of every text, for the layout checks."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W * bc.MM, H * bc.MM))
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        self.ax = ax
        self.r = self.fig.canvas.get_renderer()
        self.registry = []              # (artist, card index) for the checks

    # text ------------------------------------------------------------------
    def text(self, x, y, s, size=22, color=INK, weight="normal", style="normal",
             ha="left", va="baseline", card=None, **kw):
        """A text at (x, y) mm, registered with the card it belongs to."""
        t = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                         fontstyle=style, ha=ha, va=va, **kw)
        self.registry.append((t, card))
        return t

    def width(self, s, size, weight="normal", style="normal"):
        """Width (mm) of a text in this face, measured on the figure."""
        t = self.ax.text(0, 0, s, fontsize=size, fontweight=weight, fontstyle=style)
        w = t.get_window_extent(renderer=self.r).width / self.fig.dpi * 25.4
        t.remove()
        return w

    def runs_width(self, parts, size):
        """Total width (mm) of a list of runs (text, weight, style)."""
        return sum(self.width(p[0], size, p[1], p[2]) for p in parts)

    def runs(self, x, y, parts, size, color=INK, ha="left", card=None, va="baseline", **kw):
        """Left-to-right runs [(text, weight, style)], aligned as a whole."""
        ws = [self.width(p[0], size, p[1], p[2]) for p in parts]
        total = sum(ws)
        x0 = {"left": x, "center": x - total / 2, "right": x - total}[ha]
        out = []
        for (s, wgt, sty), w in zip(parts, ws):
            out.append(self.text(x0, y, s, size, color, wgt, sty, card=card, va=va, **kw))
            x0 += w
        return out

    # shapes ----------------------------------------------------------------
    def card(self, x, y, w, h, fc="white", ec=RULE, lw=1.4):
        """A rounded card frame at (x, y) mm, w x h mm."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3",
                                         fc=fc, ec=ec, lw=lw, zorder=0.5))

    def arrow(self, x0, y0, x1, y1, color=TEAL, lw=2.8, head=4.2, both=False, z=3):
        """A straight arrow from (x0, y0) to (x1, y1) mm (double-headed with ``both``)."""
        style =f"{'<|' if both else ''}-|>,head_length={head * 0.45},head_width={head * 0.28}"
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                          mutation_scale=10, color=color, lw=lw,
                                          shrinkA=0, shrinkB=0, zorder=z))

    def path_arrow(self, pts, color=TEAL, lw=3.0, head=4.4, z=3):
        """An arrow along the polyline ``pts`` (mm), with the head at its last point."""
        codes =[Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
        style = f"-|>,head_length={head * 0.45},head_width={head * 0.28}"
        self.ax.add_patch(FancyArrowPatch(path=Path(pts, codes), arrowstyle=style,
                                          mutation_scale=10, color=color, lw=lw,
                                          shrinkA=0, shrinkB=0, zorder=z,
                                          joinstyle="miter", capstyle="butt"))

    def inset(self, x, y, w, h, z=2):
        """An axes at (x, y) mm, w x h mm, with no frame, for a picture inside a card."""
        a = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])
        a.axis("off")
        a.set_zorder(z)
        return a

    # checks ----------------------------------------------------------------
    def check(self, cards, tol=0.4):
        """Report texts that leave their card or the figure, or overlap another text."""
        boxes = []
        inv = self.ax.transData.inverted()
        for t, c in self.registry:
            if not t.get_visible() or not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer=self.r)
            (x0, y1), (x1, y0) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            boxes.append((t.get_text(), c, x0, y0, x1, y1))
        problems = 0
        for s, c, x0, y0, x1, y1 in boxes:
            if c is not None:
                cx, cy, cw, ch = cards[c]
                if x0 < cx + 1.5 - tol or x1 > cx + cw - 1.5 + tol or y0 < cy - tol or y1 > cy + ch + tol:
                    print(f"  LEAVES CARD {c + 1}: {s!r} x {x0:.1f}-{x1:.1f} (card {cx:.1f}-{cx + cw:.1f})"
                          f" y {y0:.1f}-{y1:.1f} (card {cy:.1f}-{cy + ch:.1f})")
                    problems += 1
            if x0 < -tol or x1 > self.W + tol or y0 < -tol or y1 > self.H + tol:
                print(f"  LEAVES FIGURE: {s!r}")
                problems += 1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ox = min(a[4], b[4]) - max(a[2], b[2])
                oy = min(a[5], b[5]) - max(a[3], b[3])
                if ox > tol and oy > tol:
                    print(f"  OVERLAP: {a[0]!r} / {b[0]!r} ({ox:.1f} x {oy:.1f} mm)")
                    problems += 1
        sizes = sorted({round(t.get_fontsize(), 1) for t, _ in self.registry if t.get_text().strip()})
        print(f"  text check: {len(boxes)} texts, {problems} problem(s); sizes {sizes} pt")
        return problems


def cloud_glyph(ax, x, y, r, color, z=5):
    """A small cloud of size ``r`` mm centred at (x, y): three discs on a bar."""
    for dx, dy, rr in((-0.55, 0.15, 0.55), (0.0, -0.25, 0.7), (0.55, 0.15, 0.55)):
        ax.add_patch(Circle((x + dx * r, y + dy * r), rr * r, fc=color, ec="none", zorder=z))
    ax.add_patch(Rectangle((x - 1.05 * r, y + 0.1 * r), 2.1 * r, 0.6 * r, fc=color, ec="none", zorder=z))


def satellite(ax, x, y, s=1.0, colour=DEEP, z=9):
    """A small satellite glyph centred at (x, y) mm (y down)."""
    for dx in (-8.0, 2.6):
        ax.add_patch(Rectangle((x + dx * s, y - 1.2 * s), 5.4 * s, 2.4 * s, fc=tint(TEAL, 0.85),
                               ec=colour, lw=1.0, zorder=z))
    ax.add_patch(Rectangle((x - 1.9 * s, y - 1.5 * s), 3.8 * s, 3.0 * s, fc=colour, ec="white",
                           lw=0.8, zorder=z + 0.1))
    ax.add_patch(Polygon([(x - 1.0 * s, y + 1.5 * s), (x + 1.0 * s, y + 1.5 * s), (x, y + 2.9 * s)],
                         closed=True, fc=colour, ec="none", zorder=z + 0.1))


# plain / italic / semibold runs
def R(s):
    """A text run in the regular face, for ``Sheet.runs``."""
    return (s, "normal", "normal")


def I(s):
    """A text run in italic (the symbols h, t, tau, z)."""
    return (s, "normal", "italic")


def B(s):
    """A text run in semibold."""
    return (s, "semibold", "normal")


def BI(s):
    """A text run in semibold italic."""
    return (s, "semibold", "italic")


def bold(parts):
    """The same runs in semibold (key lines: symbols included, as semibold italic)."""
    return [(p[0], "semibold", p[2]) for p in parts]


H_T_TAU = [I("h"), R("("), I("t"), R(" − "), I("τ"), R(")")]
H_T = [I("h"), R("("), I("t"), R(")")]


def skew_quads(bl_x, bot, Lw, Lh, skew, nu, nv):
    """Corner grids (nv+1, nu+1) of an oblique map tile: u along, v into the page."""
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
    return bl_x + u * Lw + v * skew, bot - v * Lh


def tile_outline(bl_x, bot, Lw, Lh, skew):
    """The four corners (mm) of an oblique map tile (bottom-left corner at bl_x, bot)."""
    return [[bl_x, bot], [bl_x + Lw, bot], [bl_x + Lw + skew, bot - Lh], [bl_x + skew, bot - Lh]]


# ═════════════════════════════════════════════════════════════════════════════
#  Layout
# ═════════════════════════════════════════════════════════════════════════════
FS_TITLE, FS_KEY, FS_TXT, FS_LAB, FS_TAG = 28, 24, 22, 22, 20
FS_SYM = 26                          # the symbols of the legend
LEAD_KEY, LEAD_TXT = 9.3, 9.0
MARGIN, GAP, ROW_GAP = 1.25, 8.5, 14.0
COL_W = (97.0, 151.0, 123.5)         # + 2 gaps + 2 margins = 391: the middle column holds
                                     # the two physical relations (cards 2 and 5)
LEGEND_H = 21.0                      # the symbol legend above the cards (two lines)
LEGEND_GAP = 2.0                     # legend to the top of the INPUT tabs
TAB_UP = 3.5                         # a tab stands this far above its card's top border
G0 = 19.0                            # top of a card's graphic area, from the card top
TEXT_FOOT = 6.0                      # last text baseline above a card's bottom border
PAD = 4.2                            # card side padding


class Layout:
    """The legend band and the six card frames (two rows of three columns,
    widths in proportion to COL_W) on a W x H mm sheet."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        scale =(W - 2 * MARGIN - 2 * GAP) / sum(COL_W)
        self.cw = [c * scale for c in COL_W]
        self.legend_y = MARGIN
        top = MARGIN + LEGEND_H + LEGEND_GAP + TAB_UP
        self.ch = (H - MARGIN - top - ROW_GAP) / 2
        self.frames = []
        for row in range(2):
            x = MARGIN
            for col in range(3):
                self.frames.append((x, top + row * (self.ch + ROW_GAP), self.cw[col], self.ch))
                x += self.cw[col] + GAP

    def text_top(self, n_key, n_txt):
        """Baseline of the first key line (from the card top), so that the last line
        sits TEXT_FOOT mm above the card's bottom border (card_words: the first plain
        line follows the last key line by LEAD_TXT - 0.6)."""
        return (self.ch - TEXT_FOOT - n_txt * LEAD_TXT - (n_key - 1) * LEAD_KEY
                + (0.6 if n_txt else 0.0))


def card_head(sh, k, frame, title, emph=None):
    """Card k's frame (``emph`` "key" or "product" for the emphasised cards 3
    and 6), its numbered disc and its title."""
    x, y, w, h = frame
    if emph == "key":
        sh.card(x, y, w, h, fc=MIST, ec=DEEP, lw=2.2)
    elif emph == "product":
        sh.card(x, y, w, h, fc="white", ec=SAND, lw=2.8)
    else:
        sh.card(x, y, w, h)
    sh.ax.add_patch(Circle((x + 9.6, y + 10.6), 6.1, fc=TEAL, ec="none", zorder=4))
    sh.text(x + 9.6, y + 10.7, str(k + 1), 24, "white", "semibold", ha="center",
            va="center_baseline", card=k, zorder=6)
    sh.text(x + 18.6, y + 14.6, title, FS_TITLE, INK, "semibold", card=k)


def card_tab(sh, k, frame, s, fc):
    """INPUT / OUTPUT: a filled tab astride the card's top border, at its right end."""
    x, y, w, h = frame
    tw = sh.width(s, FS_TAG, "semibold") + 5.0
    th = 2 * TAB_UP
    x0 = x + w - 6.0 - tw
    sh.ax.add_patch(FancyBboxPatch((x0, y - th / 2), tw, th, boxstyle="round,pad=0,rounding_size=1.6",
                                   fc=fc, ec="white", lw=1.2, zorder=7))
    sh.text(x0 + tw / 2, y + 0.1, s, FS_TAG, "white", "semibold", ha="center",
            va="center_baseline", card=None, zorder=8)


def card_words(sh, lay, k, frame, keys, lines):
    """Key line(s) in deep blue, then plain lines, centred."""
    x, y, w, h = frame
    cx = x + w / 2
    yy = y + lay.text_top(len(keys), len(lines))
    for key in keys:
        if isinstance(key, str):
            sh.text(cx, yy, key, FS_KEY, DEEP, "semibold", ha="center", card=k)
        else:                                  # key lines are semibold, symbols included
            sh.runs(cx, yy, bold(key), FS_KEY, DEEP, ha="center", card=k)
        yy += LEAD_KEY
    yy += LEAD_TXT - LEAD_KEY - 0.6
    for line in lines:
        if isinstance(line, str):
            sh.text(cx, yy, line, FS_TXT, INK, ha="center", card=k)
        else:
            sh.runs(cx, yy, line, FS_TXT, INK, ha="center", card=k)
        yy += LEAD_TXT


def graphic_bottom(lay, frame, n_key, n_txt, clear=8.4):
    """Lowest y (mm, sheet) the graphic of a card may use."""
    return frame[1] + lay.text_top(n_key, n_txt) - clear


# ═════════════════════════════════════════════════════════════════════════════
#  The legend: every symbol in words, with units, read before the cards
# ═════════════════════════════════════════════════════════════════════════════
def legend(sh, lay):
    """The band above the cards that names h(t), tau and z in words, with units."""
    y0 = lay.legend_y
    sh.ax.add_patch(FancyBboxPatch((MARGIN, y0), lay.W - 2 * MARGIN, LEGEND_H,
                                   boxstyle="round,pad=0,rounding_size=2.5", fc=MIST, ec="none",
                                   zorder=0.5))
    # h is the tide AT THE MOUTH (the pixel's own water is h(t - tau), card 2); t is
    # any instant (card 2: the image's; card 5: every instant of the record); z is
    # a level on the scale of h: wet/dry images time the tide, they do not measure
    # its height inside the estuary (card 2 says so in words). tau is FITTED per
    # 3-km section: true of the fit and of the values MAREA then gives each pixel
    # (pyintertidal/marea.py smooths the section delays along the distance from
    # the mouth, so a pixel's delay is not exactly its section's step)
    entries = [
        ([BI("h"), B("("), BI("t"), B(")")],
         [R("tide at the mouth at time "), I("t")], [R("(m above mean sea level)")]),
        # "per 3 km of channel" (round 4): the sections are 3 km of distance along
        # the channel from the mouth, not 3 x 3 km squares (card 2 draws them)
        ([BI("τ")], [R("delay behind the mouth")], [R("(min; fitted per 3 km of channel)")]),
        ([BI("z")], [R("ground elevation of the pixel")],
         [R("(m, same scale as "), I("h"), R(")")]),
    ]
    b1 = y0 + LEGEND_H / 2 - 2.55                   # baselines of the two lines
    b2 = b1 + 8.8
    bs = (b1 + b2) / 2 + 1.3                        # the symbol, centred on both
    sym_gap = 3.6
    widths = []
    for sym, l1, l2 in entries:
        ws = sh.runs_width(sym, FS_SYM)
        widths.append((ws, ws + sym_gap + max(sh.runs_width(l1, FS_LAB), sh.runs_width(l2, FS_LAB))))
    inner0, inner1 = MARGIN + 8.0, lay.W - MARGIN - 8.0
    gap = (inner1 - inner0 - sum(w for _, w in widths)) / (len(entries) - 1)
    x = inner0
    for (sym, l1, l2), (ws, wt) in zip(entries, widths):
        sh.runs(x, bs, sym, FS_SYM, DEEP)
        sh.runs(x + ws + sym_gap, b1, l1, FS_LAB, INK)
        sh.runs(x + ws + sym_gap, b2, l2, FS_LAB, SLATE)
        x += wt + gap
    if gap < 8.0:
        print(f"  WARNING legend: only {gap:.1f} mm between its entries")


# ═════════════════════════════════════════════════════════════════════════════
#  The six cards
# ═════════════════════════════════════════════════════════════════════════════
# --- 1 · seen many times ------------------------------------------------------
def card1(sh, lay, frame, world):
    """Card 1, "Seen many times" (INPUT 1, images): three looks of the same
    patch of 10-m pixels (dry, cloud, wet) stacked in time, the pixel marked."""
    k = 0
    x, y, w, h = frame
    card_head(sh, k, frame, "Seen many times")
    ax = sh.ax
    gx0, gx1, gy = x + PAD, x + w - PAD, y + G0
    gbot = graphic_bottom(lay, frame, 1, 3)
    card_tab(sh, k, frame, "INPUT 1 · IMAGES", DEEP)
    sh.text(gx0, gy + 5.4, "Sentinel-2 data cube", FS_LAB, INK, "semibold", card=k)
    # three looks of the same 10 x 4 pixels, oldest at the bottom; the ground rises
    # to the right, the pixel (column 4, row 1) sits at elevation Z0
    nu, nv, pi_, pj = 10, 4, 4, 1
    ii, jj = np.meshgrid(np.arange(nu), np.arange(nv))
    rng = np.random.default_rng(5)
    zp = Z0 + 0.30 * (ii - pi_) - 0.10 * (jj - pj) + 0.07 * rng.standard_normal(ii.shape)
    zp[pj, pi_] = Z0
    looks = ("dry", "cloud", "wet")                                 # bottom (old) -> top (new)
    level = {"dry": Z0 - 0.45, "wet": Z0 + 0.45}
    Lw, Lh, skew = 38.0, 6.0, 10.0
    # the line under the stack ("10-m pixels · 2023-2025", 81 mm) runs under the
    # key's "dry": FOOT keeps it a full line below that word
    stack_top = gy + 8.0
    FOOT = 9.8
    dy = min(10.0, (gbot - FOOT - stack_top - Lh) / 2)             # the layers fill the room
    bl_x = gx0 + 8.6
    bot0 = stack_top + (len(looks) - 1) * dy + Lh
    cloud_grey = tint(SLATE, 0.30)
    chip_x = bl_x + Lw + skew + 6.0
    centres = []
    for m, kind in enumerate(looks):
        bot = bot0 - m * dy
        X, Y = skew_quads(bl_x, bot, Lw, Lh, skew, nu, nv)
        if kind == "cloud":
            C = np.zeros((nv, nu))
            cmap = LinearSegmentedColormap.from_list("c", [cloud_grey, cloud_grey])
        else:
            C = (zp < level[kind]).astype(float)                      # 1 = wet
            cmap = LinearSegmentedColormap.from_list("wd", [FLATS, TEAL])
        ax.pcolormesh(X, Y, C, cmap=cmap, vmin=0, vmax=1, shading="flat", zorder=2 + m * 0.01,
                      edgecolors="white", linewidth=0.45)
        ax.add_patch(Polygon(tile_outline(bl_x, bot, Lw, Lh, skew), closed=True, fc="none",
                             ec="white", lw=1.2, zorder=2.5))
        if kind == "cloud":
            cloud_glyph(ax, bl_x + Lw * 0.30 + skew / 2, bot - Lh / 2 + 0.4, 2.3, "white", z=3)
            cloud_glyph(ax, bl_x + Lw * 0.78 + skew / 2, bot - Lh / 2 + 0.2, 2.0, "white", z=3)
        q = [[X[pj, pi_], Y[pj, pi_]], [X[pj, pi_ + 1], Y[pj, pi_ + 1]],
             [X[pj + 1, pi_ + 1], Y[pj + 1, pi_ + 1]], [X[pj + 1, pi_], Y[pj + 1, pi_]]]
        ax.add_patch(Polygon(q, closed=True, fc="none", ec=INK, lw=1.8, zorder=6))
        cx_ = np.mean([p[0] for p in q])
        cy_ = np.mean([p[1] for p in q])
        centres.append((cx_, cy_))
        ax.plot([q[1][0] + 0.6, chip_x - 3.2], [cy_, cy_], color=SLATE, lw=1.0,
                ls=(0, (1.5, 1.5)), zorder=5)
        # each key chip is drawn in its layer's own colours: wet teal, dry the flats'
        # sand tint, cloud the grey layer with a white cloud
        fc = {"wet": TEAL, "dry": FLATS, "cloud": cloud_grey}[kind]
        ax.add_patch(Rectangle((chip_x - 2.7, cy_ - 2.7), 5.4, 5.4, fc=fc, ec="none",
                               zorder=6))
        if kind == "cloud":
            cloud_glyph(ax, chip_x, cy_ + 0.3, 1.5, "white", z=7)
        sh.text(chip_x + 4.2, cy_ + 2.6, kind, FS_LAB, INK if kind != "cloud" else SLATE,
                "semibold" if kind != "cloud" else "normal", card=k)
    xs_, ys_ = zip(*centres)
    ax.plot(xs_, ys_, color=INK, lw=1.0, ls=(0, (1.2, 1.6)), zorder=5)
    sh.arrow(gx0 + 0.8, bot0 - 0.4, gx0 + 0.8, stack_top - 0.6, color=SLATE, lw=1.6, head=3.0)
    sh.text(gx0 + 5.4, (bot0 + stack_top) / 2, "time", FS_LAB, SLATE, ha="center", va="center",
            rotation=90, card=k)
    ys = bot0 + FOOT
    sh.text(gx0, ys, "10-m pixels · 2023–2025", FS_LAB, SLATE, card=k)
    if ys > gbot + 0.05:
        print(f"  WARNING card 1: graphic ends at {ys - y:.1f} mm, room to {gbot - y:.1f} mm")
    # a look is the pixel seen without cloud (MAREA keeps scenes with little cloud
    # and masks the clouds left in them pixel by pixel): the cloudy layer is no look
    card_words(sh, lay, k, frame, ["each look: wet or dry"],
               ["1 look = 1 cloud-free view", "water index from green", "and near-infrared light"])


# --- 2 · its tide arrives late (real outline) ---------------------------------
EMS_COLS = (150, 1580)               # crop of the Ems grid: the mouth to the inner Dollard


def ems_image(g, kp, crop, min_px=1000):
    """RGBA picture of the estuary: sections in two shades, mouth navy, pixel's section teal.

    Pieces of the outline that the crop leaves detached and smaller than ``min_px``
    pixels (0.4 km2 at 20 m) are not drawn: at the card's scale (about 1 mm per
    km) a 0.1-km2 sliver of sea on the top edge read as a stray dash beside the
    "mouth" leader (cartographic generalisation, as the site strip drops
    speckle)."""
    from scipy import ndimage

    k, cls = g["k"][crop].copy(), g["cls"][crop].copy()
    lab, n = ndimage.label(k >= 0, structure=np.ones((3, 3), int))
    if n > 1:
        size = np.bincount(lab.ravel())
        size[0] = min_px                                  # background: never dropped
        drop = (size < min_px)[lab]
        k[drop] = -1
        cls[drop] = 0                                     # neither water nor flats: not drawn
    rgba = np.zeros(k.shape + (4,), np.float32)
    odd = (k % 2) == 1
    sea, fl = cls == 1, cls == 2
    for m, c in ((fl & ~odd, tint(SAND, 0.78)), (fl & odd, tint(SAND, 0.36)),
                 (sea & ~odd, tint(DEEP, 0.34)), (sea & odd, tint(DEEP, 0.12))):
        rgba[m, :3] = c
        rgba[m, 3] = 1.0
    for kk, col, sh_sea, sh_fl in ((0, DEEP, 0.80, 0.92), (kp, TEAL, 0.62, 0.95)):
        m = k == kk
        rgba[m & sea, :3] = tint(col, sh_sea)
        rgba[m & fl, :3] = tint(col, sh_fl)
    return rgba


def card2(sh, lay, frame, world, g):
    """Card 2, "Its tide arrives late" (INPUT 2, tide at the mouth): the real
    Ems-Dollard outline in 3-km sections with the pixel's section, and the
    mouth tide beside the same curve shifted by the delay, at one image time."""
    k = 1
    x, y, w, h = frame
    card_head(sh, k, frame, "Its tide arrives late")
    ax = sh.ax
    gx0, gx1, gy = x + PAD, x + w - PAD, y + G0
    gbot = graphic_bottom(lay, frame, 1, 2, clear=7.6)
    # the tab names the input, so the delay drawn on this card is not read as one
    # (card 3 finds it from the images); its source is the Input data paragraph's
    card_tab(sh, k, frame, "INPUT 2 · TIDE AT THE MOUTH", DEEP)

    # --- the map: real outline, 3-km sections, the pixel's section ----------
    kk, cls = g["k"], g["cls"]
    est = kk >= 0
    r0 = int(np.flatnonzero(est.any(axis=1))[0])
    r1 = int(np.flatnonzero(est.any(axis=1))[-1]) + 1
    c0, c1 = EMS_COLS
    crop = (slice(r0, r1), slice(c0, c1))
    ny, nx = r1 - r0, c1 - c0
    my = gy + 0.5
    mh = gbot - my
    mw = mh * nx / ny
    mx = gx0
    kp = world["pix_section"]
    a = sh.inset(mx, my, mw, mh, z=2)
    a.imshow(ems_image(g, kp, crop), interpolation="antialiased", aspect="auto")
    s = np.ma.masked_invalid(g["s_km"][crop])
    a.contour(s, levels=np.arange(SECTION_KM, 42.0, SECTION_KM), colors="white",
              linewidths=1.1, zorder=4)
    a.set_xlim(-0.5, nx - 0.5)
    a.set_ylim(ny - 0.5, -0.5)

    def to_mm(c, r):
        """Column and row of the map crop to sheet millimetres."""
        return mx + (c + 0.5) / nx * mw, my + (r + 0.5) / ny * mh

    kc = kk[crop]
    # the pixel: a flat pixel of its section, near the section's middle
    sel = np.argwhere((kc == kp) & (cls[crop] == 2))
    cen = sel.mean(axis=0)
    rr, cc = sel[np.argmin(((sel - cen) ** 2).sum(axis=1))]
    a.plot([cc], [rr], marker="s", ms=7.5, mfc="white", mec=INK, mew=1.8, ls="", zorder=8)
    # the square named (2026-10-20 review): every "its" of the cards (its tide, its
    # section, its water level) is this pixel's, and card 3's dots are "this
    # pixel"'s looks. The word stands in the empty corner left of the Dollard (no
    # outline there from 85 to 97 mm down, left of x = 136 mm), level with the
    # square; its leader is drawn in the map's own axes, above the outline (on the
    # sheet's axes the map would hide it)
    sq_x, sq_y = to_mm(cc, rr)
    xr_pix = mx + 0.47 * mw                       # the word's right end (133.0 mm)
    sh.text(xr_pix, sq_y + 2.4, "pixel", FS_LAB, INK, "semibold", ha="right", card=k)
    lead_mm = np.array([[xr_pix + 0.9, sq_y + 0.3], [sq_x - 2.3, sq_y]])
    a.plot((lead_mm[:, 0] - mx) / mw * nx - 0.5, (lead_mm[:, 1] - my) / mh * ny - 0.5,
           color=INK, lw=1.3, solid_capstyle="butt", zorder=7)
    m0 = np.argwhere(kc == 0)
    mouth_end = to_mm(np.percentile(m0[:, 1], 97), np.median(m0[:, 0]))
    mk = np.argwhere(kc == kp)
    sec_end = to_mm(np.percentile(mk[:, 1], 96), np.percentile(mk[:, 0], 30))
    # the outline's name, on the land east of the outer Ems (upper right of the map)
    sh.text(mx + 0.50 * mw, my + 0.30 * mh, "Ems-", FS_TAG, SLATE, card=k)
    sh.text(mx + 0.50 * mw, my + 0.30 * mh + 7.2, "Dollard", FS_TAG, SLATE, card=k)

    # --- the column between the map and the plot: the map's colours named (and
    #     with them the curves'), and the ground line's name at its own height ----
    # (tau = 0 at the mouth: the legend's "delay behind the mouth")
    lx = mx + mw + 3.5
    y_m = gy + 5.0
    sh.text(lx, y_m, "mouth", FS_LAB, DEEP, "semibold", card=k)
    # h(t) on the same line as "mouth" (round 4): the line under it is where the
    # 22-pt "expected wet" needs its room (the plot's top-left corner is empty, so
    # the pair may run over it)
    sh.runs(lx + sh.width("mouth", FS_LAB, "semibold") + 2.4, y_m, H_T, FS_LAB, DEEP, card=k)
    lead_s = 8.0
    y_s = gbot - 2 * lead_s
    sh.text(lx, y_s, "its 3-km", FS_LAB, TEAL, "semibold", card=k)
    sh.text(lx, y_s + lead_s, "section", FS_LAB, TEAL, "semibold", card=k)
    sh.runs(lx, y_s + 2 * lead_s, H_T_TAU, FS_LAB, TEAL, card=k)
    # "ground z" just above the lower group: the ground line is drawn at its height
    y_g = y_s - 5.6 - 0.8 - 1.9                               # its baseline
    sh.runs(lx, y_g, [R("ground "), I("z")], FS_LAB, INK, card=k)
    w_g = sh.runs_width([R("ground "), I("z")], FS_LAB)
    ax.plot([mouth_end[0] + 0.8, lx - 1.2], [mouth_end[1], y_m - 2.6], color=DEEP, lw=1.3, zorder=7)
    ax.plot([sec_end[0] + 0.8, lx - 1.2], [sec_end[1], y_s - 2.6], color=TEAL, lw=1.3, zorder=7)
    lab_w = max(sh.width("its 3-km", FS_LAB, "semibold"), sh.runs_width(H_T_TAU, FS_LAB),
                sh.width("mouth", FS_LAB, "semibold"), w_g)

    # --- the tide at the mouth and at its section --------------------------
    px0, px1 = lx + lab_w + 5.5, gx1
    py0, py1 = gy + 8.5, gbot - 8.0
    t0, t1 = -6.4, 3.4
    A_, lo, hi = 1.0, -1.18, 1.30
    om = 2 * np.pi / M2
    y_z = y_g - 2.0                                  # the ground line, level with its name
    scale = (y_z - py0) / hi                         # mm per m: hi at the top, 0 at y_z
    if y_z + (-lo) * scale > py1 + 0.5:
        print("  WARNING card 2: the tide plot runs into its time label")

    def X(tt):
        """Time (h) to x (mm) on the card's tide plot."""
        return px0 + (tt - t0) / (t1 - t0) * (px1 - px0)

    def Y(v):
        """Level (m) to y (mm) on the card's tide plot."""
        return y_z - v * scale

    tt = np.linspace(t0, t1, 400)
    ax.plot(X(tt), Y(A_ * np.cos(om * tt)), color=DEEP, lw=3.0, zorder=4, solid_capstyle="round")
    ax.plot(X(tt), Y(A_ * np.cos(om * (tt - TAU_H))), color=TEAL, lw=3.0, zorder=4,
            solid_capstyle="round")
    ax.plot([lx + w_g + 1.4, X(t1)], [y_z] * 2, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=3)
    # the image instant: rising tide, the mouth above z, the section still below
    ti = -M2 / 4 + TAU_H / 2
    ym, ys_ = Y(A_ * np.cos(om * ti)), Y(A_ * np.cos(om * (ti - TAU_H)))
    ax.plot([X(ti)] * 2, [py0 - 3.4, py1 - 1.2], color=SLATE, lw=1.3, ls=(0, (2, 2)), zorder=2)
    satellite(ax, X(ti), py0 - 5.6, 0.85)
    # "image at time t" (2026-10-20): the word names the plot's horizontal axis. It
    # stands 2.2 mm higher than before and 3 mm right of its dotted line, so that it
    # is not read on one line with the teal h(t - tau) of the labels' column
    # ("h(t - tau) image at time t", review): 11.9 mm apart, baselines 2.8 mm
    # apart; the curves' troughs stay 4.8 mm above its letters
    sh.runs(X(ti) + 3.0, py1 + 5.2, [R("image at time "), I("t")], FS_LAB, SLATE, ha="center",
            card=k)
    ax.add_patch(Circle((X(ti), ym), 1.7, fc=DEEP, ec="white", lw=1.0, zorder=6))
    ax.add_patch(Rectangle((X(ti) - 2.0, ys_ - 2.0), 4.0, 4.0, fc=SAND, ec="white", lw=1.0,
                           zorder=6))
    # the two states at that instant: judged by the mouth tide (already above z)
    # the pixel should be wet, yet its own water is still below z: seen dry. The
    # card's point, so at the label size (22 pt, round 4), not the tag size.
    # "no delay: wet", not "expected wet" (2026-10-20 review): "expected" by whom?
    # It read as MAREA's own (wrong) prediction; "no delay" says it is the mouth tide
    # taken as it is (tau = 0, card 3's row "0", where such dry looks fall right of z)
    sh.text(X(ti) - 2.4, ym - 2.2, "no delay: wet", FS_LAB, DEEP, "semibold", ha="right", card=k)
    sh.text(X(ti) + 3.0, ys_ + 6.6, "seen dry", FS_LAB, SAND_DARK, "semibold", card=k)
    # the delay between the two high waters
    ya = Y(A_) - 2.6
    for tx in (0.0, TAU_H):
        ax.plot([X(tx)] * 2, [Y(A_) - 0.8, ya - 1.2], color=INK, lw=1.0, zorder=3)
    sh.arrow(X(0.0), ya, X(TAU_H), ya, color=INK, lw=1.3, head=1.7, both=True)
    sh.text(X(TAU_H) + 1.6, ya + 1.6, "τ", FS_LAB + 2, INK, style="italic", card=k)

    # the rule, then in words what the two curves show and the assumption behind
    # it: the pixel's water is the mouth tide only shifted in time. 2026-10-20: the
    # first author did not follow "its water: the mouth tide, tau later" (no verb,
    # "its" unclear) nor "a taller tide inland is not corrected" (who corrects
    # what?). Line 1 names the curves (navy: the mouth tide; teal: that tide at
    # its section, the map's teal label, tau later). Line 2 names the method and
    # sets time against height: MAREA fits no gain (tide_estimators.m2a_rasch,
    # fit_alpha=False; wet/dry looks time the tide, they cannot measure its
    # height), so where the tide grows taller inland z stays a level on the mouth
    # tide's scale (the legend's "same scale as h"). Its subject is MAREA, not the
    # tide: it makes no claim about the real tide, which b1 shows "later and
    # higher". Rejected: "the tide is shifted in time, not in height" (reads as a
    # claim about the estuary), "the timing is corrected, not the height" ("the
    # height" can be read as the ground z labelled above it). Review of the same
    # day: "MAREA shifts it ..." left "it" open (the tide? the pixel, now named on
    # the map?), so the line names the tide; "not height", as "not in height" is
    # 149.5 mm, wider than the card's text room (140.7 mm as set)
    card_words(sh, lay, k, frame, [[R("wet when ")] + H_T_TAU + [R(" > "), I("z")]],
               [[R("the mouth tide reaches its section "), I("τ"), R(" later")],
                "MAREA shifts the tide in time, not height"])


# --- 3 · its section's delay ---------------------------------------------------
def card3(sh, lay, frame, world):
    """Card 3, "Its section's delay": one row per delay tried, the pixel's wet
    and dry looks placed at their water level under that delay, and bars for
    how well each delay fits all the section's pixels; the best one splits the
    looks most cleanly."""
    k = 2
    x, y, w, h = frame
    card_head(sh, k, frame, "Its section’s delay", emph="key")
    ax = sh.ax
    gx0, gx1, gy = x + PAD, x + w - PAD, y + G0
    gbot = graphic_bottom(lay, frame, 1, 1)
    t, wet = world["t"], world["wet"]
    sub = world["sub3"]
    # 2026-10-20: the first author could not tell what the rows and dots stood for.
    # The rows are named by bare numbers (the unit stands in the head line): the
    # 15 mm this frees hold each row's verdict at the right end of its strip
    lab_w = sh.width("135", FS_LAB)
    c2x0 = gx0 + lab_w + 2.5
    c3x0 = gx1 - 19.0
    c2x1 = c3x0 - 3.0
    # "(examples)", not "(simulated)" (round 4): "simulated" next to the delays could
    # be read as delays taken from a model, the opposite of the card's message.
    # "minutes", not "min" (review of 2026-10-20: "(min, examples)" read as
    # "minimum"); at 20 pt (108.5 mm; 119.4 mm at 22 pt, wider than the card's text
    # room) and 1.2 mm higher than before, which gives the rows' label line its room
    sh.runs(gx0, gy + 3.6, [R("delays "), I("τ"), R(" tried (minutes, examples)")], FS_TAG,
            SLATE, card=k)
    # the key of the dots, styled as the pixel-gauge panel's ("● wet: index above 0"):
    # each dot is one look (card 1's word) of THIS pixel, the square named "pixel" on
    # card 2's map, against the section's pixels that the fit judges (the last
    # line). Dry first: the dry looks lie left
    yk = gy + 11.6
    xk = gx0
    for col, word in ((SAND, "dry"), (TEAL, "wet")):
        ax.scatter([xk + 1.5], [yk - 2.0], s=70, color=col, lw=0.6, edgecolors="white", zorder=4)
        sh.text(xk + 3.9, yk, word, FS_LAB, INK, "semibold", card=k)
        xk += 3.9 + sh.width(word, FS_LAB, "semibold") + 3.0
    sh.text(xk - 3.0, yk, ": one look of this pixel", FS_LAB, INK, card=k)
    lo = -1.9

    def X(v):
        """Level (m) to x (mm) in the strips."""
        # a fixed scale (14.8 mm per m): the levels (-1.8 to +1.5 m) keep their
        # spread and every strip keeps 25 mm or more free at its right end for the
        # verdict (5 mm or more from the last dot to the word)
        return c2x0 + 2.2 + (v - lo) * 14.8

    rng = np.random.default_rng(11)
    jit = rng.uniform(-1, 1, t.size)
    ll = world["tried_ll"]
    q = (ll - ll.min()) / (ll.max() - ll.min())
    best = int(np.argmax(ll))
    n_rows = len(TRIED_MIN)
    # the label line right above the rows (review of 2026-10-20): the dashed line's
    # name over its top, "ground z" as on card 2 (a bare "z" in a wider gap between
    # the rows 45 and 90 read as a row label and paired the rows), and the bars' name
    # "fit" over the bars (at their foot, on the axis line, it read as the end of the
    # axis name: "h(t - tau) -> fit"). 20 pt: at 22 pt the line needs 0.7 mm more
    # than the card has. Its text boxes overlap the key line's by 0.27 mm, within the
    # check's 0.4-mm tolerance; the letters stay 1.1 mm or more apart (no descender
    # of the key line stands over these labels)
    yl = yk + 6.7
    xz = X(world["strip_z"][best])         # its ground z: where the best delay splits
    sh.runs(xz + sh.width("z", FS_TAG, style="italic") / 2, yl, [R("ground "), I("z")], FS_TAG,
            INK, ha="right", card=k)       # the "z" right over its line
    # "fit" centred over the bars' column (left-aligned on their axis it would meet
    # the descender of "pixel" in the key line above)
    sh.text((c3x0 + gx1) / 2, yl, "fit", FS_TAG, SLATE, ha="center", card=k)
    # four rows evenly spaced, 6.76 mm tall (the verdicts need 6.7 mm), the first
    # 0.5 mm under the descender of "ground"
    rgap = 1.15
    ry0 = yl + 2.2
    yb = gbot - 0.8                        # the level's name under the rows: its
    ry1 = yb - 7.2                         # parentheses 0.5 mm under the last row
    rh = (ry1 - ry0 - (n_rows - 1) * rgap) / n_rows
    vx = c2x1 - 2.0                        # the verdicts, right-aligned inside the strips
    for m, d in enumerate(TRIED_MIN):
        is_best = m == best
        y0 = ry0 + m * (rh + rgap)
        yc = y0 + rh / 2
        sh.text(gx0 + lab_w, yc + 2.8, f"{d}", FS_LAB, DEEP if is_best else INK,
                "semibold" if is_best else "normal", ha="right", card=k)
        ax.add_patch(FancyBboxPatch((c2x0, y0), c2x1 - c2x0, rh,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    fc="white", ec=DEEP if is_best else RULE,
                                    lw=2.4 if is_best else 1.1, zorder=2))
        # the same looks in every row, each at its water level under that row's delay
        lev = tide_mouth(t - d / 60.0)
        for wflag, col in ((False, SAND), (True, TEAL)):
            sel = sub[wet[sub] == wflag]
            ax.scatter(X(lev[sel]), yc + jit[sel] * (rh / 2 - 2.1), s=38, color=col, lw=0.5,
                       edgecolors="white", zorder=4)
        # each row's verdict in words, true of the looks drawn: of the 38, on the
        # wrong side of the ground z drawn 12/6/0/6 at 0/45/90/135 min, and still
        # 9/4/0/4 at the level with the fewest looks on the wrong side in each row,
        # so "mixed" does not hang on the line drawn. The three wrong delays all
        # named, else the blank ends of their strips look unexplained; 20 pt, so
        # that "clean" clears the frame of the best row. The words have no
        # descender: their baselines centre their ink (ascender to baseline:
        # "mixed" 5.5 mm, semibold "clean" 5.3 mm) in the strip, so each clears its
        # frame by about 0.25 mm above and below (measured on the 300-dpi PNG)
        if is_best:
            # under the best delay its ground z splits dry looks from wet ones: the
            # two sides tinted in this row only (tinted in every row, all four read as
            # split)
            ax.add_patch(Rectangle((c2x0 + 1.2, y0 + 1.2), xz - c2x0 - 1.2, rh - 2.4,
                                   fc=tint(SAND, 0.22), ec="none", zorder=2.5))
            ax.add_patch(Rectangle((xz, y0 + 1.2), c2x1 - 1.2 - xz, rh - 2.4,
                                   fc=tint(TEAL, 0.16), ec="none", zorder=2.5))
            sh.text(vx, yc + 2.6, "clean", FS_TAG, DEEP, "semibold", ha="right", card=k,
                    zorder=6)
        else:
            sh.text(vx, yc + 2.7, "mixed", FS_TAG, SLATE, ha="right", card=k, zorder=6)
        bl = 1.5 + q[m] * (gx1 - c3x0 - 1.5)
        ax.add_patch(Rectangle((c3x0, yc - 2.4), bl, 4.8, fc=DEEP if is_best else tint(DEEP, 0.30),
                               ec="none", zorder=3))
    ax.plot([c3x0, c3x0], [ry0 - 0.5, ry1 + 0.5], color=SLATE, lw=1.2, zorder=3)
    # ONE ground level through every row (2026-10-20), so "mixed" (dry looks right
    # of it, wet looks left of it: card 2's "no delay: wet, seen dry" in the row 0)
    # and "clean" are read against the same mark, named "ground z" over its top. A
    # drawing choice: MAREA refits z under each delay, but this pixel's
    # likelihood-best level under each (strip_z 0.40/0.30/0.30/0.15 m) stays within
    # 0.15 m of it, and of all its 110 looks 29/14/1/13 fall on the wrong side of it
    # against 25/10/1/11 at the level with the fewest looks on the wrong side in each
    # row (29/14/1/12 at each row's strip_z)
    ax.plot([xz] * 2, [yl + 0.8, ry1 + 0.7], color=INK, lw=1.6, ls=(0, (2.5, 1.5)), zorder=5)
    sh.runs(c2x1, yb, [R("its water level, ")] + H_T_TAU + [R(" →")], FS_LAB, SLATE,
            ha="right", card=k)
    # The dots are this pixel's looks; the "fit" bars score each delay on the looks
    # of the section's pixels together (simulate(): tried_ll), the way MAREA fits
    # one delay per section: the last line says so, and names the bars. Round 4:
    # "judged on its section's pixels", not "fit: all pixels of its section":
    # production scores each section on a sample of its pixels (at most 1,200,
    # each with at least 6 clear looks: tide_estimators.m2a_rasch, configs/m2.yaml
    # max_px_banda), not on all of them; "fit" keeps one meaning, the bars' score.
    # "Cleanest", not "clean", in the rule: even the real pixel of "A pixel is a
    # tide gauge" follows the rule in 97 % of its looks. "Best delay", not "right
    # delay" (review of 2026-10-20: beside the bars, "right" read as "on the
    # right"); "best" is also what the bars show, the longest. 2026-10-20: the line
    # "tau = 0 at mouth, only grows inland" (MAREA pins the mouth section and lets
    # the delay only grow inland: m2a_rasch, monotone=True) gave its room to the
    # key of the dots; the legend's "delay behind the mouth" keeps tau = 0 there
    card_words(sh, lay, k, frame, ["best delay: cleanest split"],
               ["fit: judged on its section’s pixels"])


# --- 4 · its elevation -------------------------------------------------------
def card4(sh, lay, frame, world):
    """Card 4, "Its elevation": the pixel's water index against the corrected
    tide with its fitted flooding curve; the midpoint is the elevation z and
    the shaded width its uncertainty."""
    k = 3
    x, y, w, h = frame
    card_head(sh, k, frame, "Its elevation")
    ax = sh.ax
    gx0, gx1, gy = x + PAD, x + w - PAD, y + G0
    gbot = graphic_bottom(lay, frame, 1, 3)
    a_, b_, z_, s_ = world["fit"]
    h_pix, yv, look, wet = world["h_pix"], world["y"], world["look"], world["wet"]
    ax0, ax1 = gx0 + 1.0, gx1 - 1.0
    ay0, ay1 = gy + 7.4, gbot - 8.6
    xlo, xhi, ylo, yhi = -2.7, 2.05, -0.62, 1.12

    def X(v):
        """Water level (m) to x (mm) on the card's plot."""
        return ax0 + (v - xlo) / (xhi - xlo) * (ax1 - ax0)

    def Y(v):
        """Water index to y (mm) on the card's plot."""
        return ay1 - (v - ylo) / (yhi - ylo) * (ay1 - ay0)

    ax.add_patch(Rectangle((X(z_ - s_), ay0), X(z_ + s_) - X(z_ - s_), ay1 - ay0, fc=TEAL_PALE,
                           ec="none", zorder=1))
    ax.plot([ax0, ax0, ax1], [ay0, ay1, ay1], color=SLATE, lw=1.3, zorder=2)
    for wflag, col in ((False, SAND), (True, TEAL)):
        sel = look & (wet == wflag)
        ax.scatter(X(h_pix[sel]), Y(np.clip(yv[sel], ylo + 0.02, yhi - 0.02)), s=20, color=col,
                   edgecolors="white", linewidths=0.5, zorder=3)
    hh = np.linspace(xlo, xhi, 400)
    ax.plot(X(hh), Y(flooding(hh, a_, b_, z_, s_)), color=DEEP, lw=3.0, zorder=4,
            solid_capstyle="butt")
    ym = a_ + b_ / 2
    ax.plot([X(z_)] * 2, [Y(ym), ay1], color=INK, lw=1.5, ls=(0, (4, 3)), zorder=4)
    ax.add_patch(Circle((X(z_), Y(ym)), 2.0, fc=INK, ec="white", lw=1.4, zorder=6))
    sh.text(X(z_) + 1.4, ay1 - 1.6, "z", FS_LAB + 1, INK, style="italic", card=k)
    # the axis named flat above it (a turned label was longer than the axis)
    sh.text(ax0, ay0 - 1.8, "water index", FS_LAB, SLATE, card=k)
    yw = ay0 - 1.4
    ax.plot([X(z_ - s_), X(z_ + s_)], [yw, yw], color=INK, lw=1.5, zorder=6)
    for xb in (X(z_ - s_), X(z_ + s_)):
        ax.plot([xb, xb], [yw - 1.3, yw + 1.3], color=INK, lw=1.5, zorder=6)
    sh.text(X(z_ + s_) + 1.5, ay0 + 0.9, "width", FS_LAB, INK, card=k)
    # the curve named as in "A pixel is a tide gauge" (left half), with a leader to
    # its upper bend; the midpoint just under it
    xr = X(z_ - s_) - 2.4
    yc = Y(ym) - 7.8
    sh.text(xr, yc, "flooding curve", FS_TAG, DEEP, "semibold", ha="right", card=k)
    xq, yq = X(z_ + 1.25 * s_), Y(flooding(z_ + 1.25 * s_, a_, b_, z_, s_))
    ax.plot([xr + 0.8, xq - 0.6], [yc - 1.6, yq + 0.6], color=SLATE, lw=1.2, zorder=5)
    sh.text(X(z_) - 3.4, Y(ym) + 1.2, "midpoint", FS_LAB, INK, ha="right", card=k)
    sh.text(X(xlo) + 1.5, Y(-0.12) + 0.4, "dry", FS_LAB, SAND_DARK, "semibold", card=k)
    sh.text(X(xhi), Y(0.06), "wet", FS_LAB, TEAL, "semibold", ha="right", card=k)
    sh.runs((ax0 + ax1) / 2, ay1 + 7.6, [R("its water level, ")] + H_T_TAU, FS_LAB, SLATE,
            ha="center", card=k)
    # the quality rule (pyintertidal.marea.QUALITY_MIN_BRACKET: at least 3 clear looks
    # with the water below the pixel's level and 3 above it), in plain words
    # ("otherwise", not "else", round 4: "else" reads as code)
    card_words(sh, lay, k, frame, [[R("midpoint = elevation "), I("z")]],
               [[R("width = uncertainty of "), I("z")], "needs wet and dry looks,",
                "otherwise left empty"])


# --- 5 · its emerged time ------------------------------------------------------
def card5(sh, lay, frame, world):
    """Card 5, "Its emerged time": a few days of the corrected tide against the
    pixel's ground z, coloured under and out of the water, and the share of
    time of each over the whole record."""
    k = 4
    x, y, w, h = frame
    card_head(sh, k, frame, "Its emerged time")
    ax = sh.ax
    gx0, gx1, gy = x + PAD, x + w - PAD, y + G0
    gbot = graphic_bottom(lay, frame, 2, 1)
    z_ = world["fit"][2]
    E = world["emerged"]
    t0 = world["spring_t0"]
    t1 = t0 + 24 * 4.0
    # the trace starts lower than the graphic area: its name stands above it, on
    # the baseline of the bar's "share of its time" (two column heads, one line)
    py0, py1 = gy + 9.5, gbot - 9.0
    y_head = py0 - 1.5
    lo, hi = -2.15, 2.15
    tx0, tx1 = gx0 + 6.0, gx0 + 0.50 * (gx1 - gx0)       # its water level, 4 days
    bx0, bx1 = tx1 + 9.0, gx1 - 1.0                      # its share of the whole record

    def Y(v):
        """Level (m) to y (mm) on the card's plot."""
        return py1 - (v - lo) / (hi - lo) * (py1 - py0)

    def Xt(tt):
        """Time (h) to x (mm) on the tide curve."""
        return tx0 + (tt - t0) / (t1 - t0) * (tx1 - tx0)

    def Xb(f):
        """Share of time (0-1) to x (mm) on the bar."""
        return bx0 + f * (bx1 - bx0)

    tt = np.linspace(t0, t1, 2000)
    lev = tide_mouth(tt - TAU_H)
    below = lev < z_
    start = 0
    for i in range(1, tt.size + 1):
        if i == tt.size or below[i] != below[start]:
            j = min(i + 1, tt.size)
            ax.plot(Xt(tt[start:j]), Y(lev[start:j]), color=SAND if below[start] else TEAL, lw=3.0,
                    zorder=4, solid_capstyle="butt")
            start = i
    # its ground: one dashed level from the 4 days across to the bar of the record
    ax.plot([tx0 - 3.0, bx0], [Y(z_)] * 2, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=5)
    sh.text(tx0 - 3.6, Y(z_), "z", FS_LAB + 2, INK, style="italic", ha="right", va="center",
            card=k)
    for dd in range(5):
        ax.plot([Xt(t0 + 24 * dd)] * 2, [py1 + 1.0, py1 + 2.8], color=SLATE, lw=1.2)
    ax.plot([tx0, tx1], [py1 + 1.9] * 2, color=SLATE, lw=1.2)
    # the x axis is TIME (card 4's, beside it, is the water level): the ticks are
    # named as days, and the curve itself is named above it
    sh.text((tx0 + tx1) / 2, py1 + 9.2, "4 days →", FS_LAB, SLATE, ha="center", card=k)
    # the two column heads at the label size of cards 3 and 4 (22 pt, round 4); the
    # trace's head is centred over the trace, which leaves about 8 mm before the
    # bar's right-aligned head on the same line
    sh.runs((tx0 + tx1) / 2, y_head, [R("its water level, ")] + H_T_TAU, FS_LAB, SLATE,
            ha="center", card=k)
    # the whole record at its ground z: out of the water (sand, its water below z)
    # for the share E, under water (teal, above z) for the rest. The labels follow
    # the level axis: "under water" above the bar, "out of the water" below it
    bh = 7.2
    yz = Y(z_)
    ax.add_patch(Rectangle((Xb(0.0), yz - bh / 2), Xb(E) - Xb(0.0), bh, fc=SAND, ec="none",
                           zorder=5))
    ax.add_patch(Rectangle((Xb(E), yz - bh / 2), Xb(1.0) - Xb(E), bh, fc=TEAL, ec="none",
                           zorder=5))
    # the values: ink on the sand segment (white on PSand was 3.0:1, ink is about
    # 5.4:1), white on the teal one (4.7:1); 22 pt, round 4
    for f0, f1, val in ((0.0, E, E), (E, 1.0, 1.0 - E)):
        sh.text((Xb(f0) + Xb(f1)) / 2, yz + 0.3, f"{val:.0%}", FS_LAB,
                INK if f0 == 0.0 else "white", "semibold", ha="center", va="center_baseline",
                card=k, zorder=7)
    # the bar's head, right-aligned like "under water" under it, on the trace's line
    sh.text(bx1, y_head, "share of its time", FS_LAB, SLATE, ha="right", card=k)
    sh.text(bx1, yz - bh / 2 - 1.8, "under water", FS_LAB, TEAL, "semibold", ha="right",
            card=k)
    sh.text(bx0, yz + bh / 2 + 7.6, "out of the water", FS_LAB, SAND_DARK, "semibold", card=k)
    if yz - bh / 2 - 1.8 - 5.6 < y_head + 1.0:
        print("  WARNING card 5: 'under water' runs into 'share of its time'")
    # the relation in symbols AND in words (its water is below its ground), and over
    # what it is counted: the continuous tide record, not the image times
    # (pyintertidal/hydroperiod.py: the satellite samples the tide unevenly)
    card_words(sh, lay, k, frame,
               [[R("emerged time "), I("E"), R("("), I("z"), R(") = share of time")],
                [R("its water is below "), I("z"), R(": ")] + H_T_TAU + [R(" < "), I("z")]],
               ["at all times, not only image times"])


# --- 6 · all pixels: the maps -------------------------------------------------
def card6(sh, lay, frame, world):
    """Card 6, "All pixels: the maps" (output): three tiles of a schematic
    estuary, for elevation, uncertainty and emerged time."""
    k = 5
    x, y, w, h = frame
    card_head(sh, k, frame, "All pixels: the maps", emph="product")
    ax = sh.ax
    gx0, gx1, gy = x + PAD, x + w - PAD, y + G0
    gbot = graphic_bottom(lay, frame, 2, 1)
    s_lo, s_hi, n_half = -0.3, 15.2, 1.45
    ns, nn_ = 168, 36
    S, N = np.meshgrid(np.linspace(s_lo, s_hi, ns + 1), np.linspace(-n_half, n_half, nn_ + 1))
    Sc, Nc = 0.5 * (S[:-1, :-1] + S[1:, 1:]), 0.5 * (N[:-1, :-1] + N[1:, 1:])
    Z, _, _ = estuary_z(Sc, Nc)
    inter = (Z > -1.9) & (Z < 1.9)
    channel = Z <= -1.9                    # permanent water: drawn as water, not mapped
    ls_ = world["levels_sorted"]
    emerged = np.searchsorted(ls_, Z) / ls_.size
    # uncertainty: a smooth made-up field with no structure tied to elevation
    rr = np.random.default_rng(3)
    unc = np.zeros_like(Z)
    for _ in range(6):
        fx, fy, ph = rr.uniform(0.3, 1.1), rr.uniform(0.8, 2.5), rr.uniform(0, 2 * np.pi)
        unc += np.sin(fx * Sc + fy * Nc + ph)
    unc = (unc - unc.min()) / (unc.max() - unc.min())
    # No colour ramp reaches white (white is land only), and the channel takes the
    # water tint of the study-site map. No empty pixels are drawn on this schematic:
    # card 4 states the rule that leaves some empty.
    layers = [
        ("elevation", np.where(inter, Z, np.nan),
         LinearSegmentedColormap.from_list("z", [DEEP, TEAL, tint(SAND, 0.45), SAND]), (-1.9, 1.9),
         ("low", "high")),
        ("uncertainty", np.where(inter, unc, np.nan),
         LinearSegmentedColormap.from_list("u", [tint(SLATE, 0.30), SLATE]), (0, 1),
         ("low", "high")),
        ("emerged time", np.where(inter, emerged, np.nan),
         LinearSegmentedColormap.from_list("e", [tint(SAND, 0.25), SAND]), (0, 1),
         ("0", "100%")),
    ]
    water_cmap = LinearSegmentedColormap.from_list("w", [WATER, WATER])
    Lw, Lh, skew = 41.0, 9.6, 10.0
    bl_x = gx0 + 0.3
    # the tab in dark sand (white on it 5.6:1; on PSand it was 3.0:1); the card's
    # frame stays PSand
    card_tab(sh, k, frame, "OUTPUT", SAND_DARK)
    top0 = gy + 5.0
    gap_l = (gbot - top0 - 3 * Lh) / 2
    tops = [top0 + i * (Lh + gap_l) for i in range(3)]
    lx = bl_x + Lw + skew + 3.5
    for (name, C, cmap, (vmin, vmax), ends), top in zip(layers, tops):
        bot = top + Lh
        Xq, Yq = skew_quads(bl_x, bot, Lw, Lh, skew, ns, nn_)
        ax.add_patch(Polygon(tile_outline(bl_x, bot, Lw, Lh, skew), closed=True, fc="white",
                             ec="none", zorder=2))
        # the tiles are raster maps: drawn as 300-dpi images inside the vector page
        # (as vector quads they made the PDF 0.9 MB and the print file 7 MB); each
        # set has a z-order of its own, so matplotlib rasterizes it in one pass
        ax.pcolormesh(Xq, Yq, np.ma.masked_where(~channel, np.ones_like(Z)), cmap=water_cmap,
                      vmin=0, vmax=1, shading="flat", zorder=2.5, edgecolors="face", linewidth=0.2,
                      rasterized=True)
        ax.pcolormesh(Xq, Yq, np.ma.masked_invalid(C), cmap=cmap, vmin=vmin, vmax=vmax,
                      shading="flat", zorder=3.05, edgecolors="face", linewidth=0.2, rasterized=True)
        ax.add_patch(Polygon(tile_outline(bl_x, bot, Lw, Lh, skew), closed=True, fc="none",
                             ec=SLATE, lw=1.0, zorder=4))
        sh.text(lx, top + 1.2, name, FS_LAB, INK, "semibold", card=k)
        kx0 = lx + sh.width(ends[0], FS_TAG) + 1.5
        kx1 = kx0 + 14.0
        ramp = np.linspace(0, 1, 32)[None, :]
        a = sh.inset(kx0, top + 5.3, kx1 - kx0, 3.2, z=3)
        a.imshow(ramp, cmap=cmap, aspect="auto", interpolation="bilinear")
        sh.text(lx, top + 8.6, ends[0], FS_TAG, SLATE, card=k)
        sh.text(kx1 + 1.5, top + 8.6, ends[1], FS_TAG, SLATE, card=k)
    for cxx in (bl_x, bl_x + Lw + skew):
        for a_, b_ in ((0, 1), (1, 2)):
            ya = tops[a_] + (Lh if cxx == bl_x else 0)
            yb = tops[b_] + (Lh if cxx == bl_x else 0)
            ax.plot([cxx, cxx], [ya + 0.8, yb - 0.8], color=RULE, lw=1.0, ls=(0, (2, 2)), zorder=1)
    # the habitat thresholds are the ecologists' to set, not the method's
    card_words(sh, lay, k, frame,
               ["base layer for the potential", [R("habitat of "), I("Zostera noltei")]],
               ["ecologists set the habitat limits"])


def arrows(sh, lay):
    """The "+" between the two inputs and the arrows that join the cards in
    reading order, including the return from card 3 down to card 4."""
    f = lay.frames
    # top row: INPUT 1 + INPUT 2 (independent inputs: no arrow between them), then on
    c1, c2 = f[0], f[1]
    yy = c1[1] + G0 + 24.0
    xp = (c1[0] + c1[2] + c2[0]) / 2
    for (wd, ht) in ((6.0, 1.8), (1.8, 6.0)):
        sh.ax.add_patch(Rectangle((xp - wd / 2, yy - ht / 2), wd, ht, fc=TEAL, ec="none", zorder=3))
    for row, col in ((0, 1), (1, 0), (1, 1)):
        a, b = f[row * 3 + col], f[row * 3 + col + 1]
        xa = a[0] + a[2]
        yy = a[1] + G0 + 24.0
        sh.arrow(xa + 1.0, yy, b[0] - 1.0, yy, color=TEAL, lw=3.0, head=4.4)
    # the return: from the bottom of card 3, back to the top of card 4. The lane runs
    # 4 mm under the top row, so the last leg (9.2 mm) is longer than the arrowhead
    # (about 7 mm) and shows a stem and the turn down
    c3, c4 = f[2], f[3]
    xs = c3[0] + 0.5 * c3[2]
    xe = c4[0] + 0.5 * c4[2]
    ym = c3[1] + c3[3] + 4.0
    sh.path_arrow([(xs, c3[1] + c3[3] + 0.8), (xs, ym), (xe, ym), (xe, c4[1] - 0.8)], color=TEAL,
                  lw=3.0, head=4.4)


def main():
    """Simulate the pixel, lay out the legend and the six cards, check every
    text and save the figure (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=float, default=391.0)
    ap.add_argument("--height", type=float, default=244.6)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    bc.use_poster_fonts()
    world = simulate()
    world["spring_t0"] = spring_start()
    g = ems_sections()
    world["pix_section"] = PIX_SECTION
    print(f"  simulated pixel: {int(world['look'].sum())} looks; section's best delay "
          f"{world['best_min']} min (planted {TAU_H * 60:.0f}); fit z {world['fit'][2]:+.2f} m, "
          f"width {world['fit'][3]:.2f} m; emerged {world['emerged']:.0%}")
    if world["best_min"] != round(TAU_H * 60):
        print("  WARNING: the section's best delay is not the planted one")

    W, H = args.width, args.height
    lay = Layout(W, H)
    print(f"  cards {', '.join(f'{c:.1f}' for c in lay.cw)} mm wide, {lay.ch:.1f} mm tall")
    sh = Sheet(W, H)
    f = lay.frames
    legend(sh, lay)
    card1(sh, lay, f[0], world)
    card2(sh, lay, f[1], world, g)
    card3(sh, lay, f[2], world)
    card4(sh, lay, f[3], world)
    card5(sh, lay, f[4], world)
    card6(sh, lay, f[5], world)
    arrows(sh, lay)
    n = sh.check(f)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pdf = OUT + ".pdf"
    sh.fig.savefig(pdf, dpi=300)
    plt.close(sh.fig)
    if not args.no_png:
        subprocess.run([os.path.join(MIKTEX, "pdftocairo.exe"), "-r", "300", "-png", "-singlefile",
                        pdf, OUT], check=True, stderr=subprocess.DEVNULL)
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = f"; peak memory {getattr(mi, 'peak_wset', mi.rss) / 2**20:.0f} MB"
    except ImportError:
        peak = ""
    print(f"wrote {pdf}" + ("" if args.no_png else f" and {OUT}.png") + f" ({W:.0f} x {H:.0f} mm)"
          f"{peak}; {n} text problem(s)")


if __name__ == "__main__":
    main()
