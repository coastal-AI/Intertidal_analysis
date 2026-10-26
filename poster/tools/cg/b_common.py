# -*- coding: utf-8 -*-
"""Shared pieces of the CARTOGRASS 2026 tide figures (b1_*, b2_*): palette, the
poster's typeface for matplotlib, a memory-light IOC reader, layout helpers.

Imported by the figure scripts next to it; run those from the repository root.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, "pyintertidal")):
    sys.exit("run from the repository root (the folder that contains pyintertidal/)")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GAUGES = "data_v4/gauges"
TRUTH = "data_v4/gauges/estuary_truth"

# Poster palette (poster/preamble/colors.tex)
INK, SLATE, DEEP, TEAL, SAND = "#16202A", "#4A5C6B", "#0B3B5A", "#1B7F86", "#C08A3E"
MIST, RULE = "#EEF2F5", "#C9D3DA"
MM = 1 / 25.4
M2_HOURS = 12.4206012          # principal lunar semidiurnal period (pyintertidal.boundary.PERIODS_H)

# ─────────────────────────────────────────────────────────────────────────────
#  Fonts: Fira Sans, the poster's face. MiKTeX ships it as CFF OpenType, which
#  matplotlib embeds in a PDF as a CID-keyed CFF font (Type 0C); a TrueType
#  copy (subset to the characters these figures need, cached under the
#  git-ignored cache/, never distributed) embeds as Type 42. Every matplotlib
#  figure of the poster calls this one function, so all of them embed the same
#  way whichever script runs first (the first call builds the cache). Falls
#  back to the OTF, then to DejaVu Sans.
# ─────────────────────────────────────────────────────────────────────────────
FONT_CACHE = "poster/tools/cg/cache/fonts_fira_ttf"
# SemiBoldItalic (2026-10-19): without it matplotlib set every semibold italic run
# (the symbols h, t, tau, z, E inside the bold key relations, "Zostera noltei"
# after "habitat of") in Book Italic, visibly thinner than the words around it.
FONT_WEIGHTS = ("Book", "SemiBold", "BookItalic", "SemiBoldItalic")
MIKTEX_BIN = "C:/Users/Jorge/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
# Latin-1 + Latin Extended-A, Greek (tau, sigma, Delta ...), punctuation, super/subscripts,
# arrows and maths operators; the subsetter keeps whatever of this the face actually has.
_CHARS = "".join(chr(c) for rng in ((0x20, 0x7F), (0xA0, 0x250), (0x370, 0x400), (0x2000, 0x20A0),
                                     (0x2100, 0x2160), (0x2190, 0x2200), (0x2200, 0x2300))
                 for c in range(*rng))


def _kpsewhich(name):
    """Path of a TeX file (here a Fira Sans OTF) found by kpsewhich, or None."""
    exe = shutil.which("kpsewhich") or os.path.join(MIKTEX_BIN, "kpsewhich.exe")
    try:
        p = subprocess.run([exe, name], capture_output=True, text=True, timeout=120).stdout.strip()
        return p if p and os.path.exists(p) else None
    except (OSError, subprocess.SubprocessError):
        return None


def _otf_to_ttf(src, dst):
    """Write a TrueType copy of the OTF ``src`` at ``dst``, subset to the
    characters of ``_CHARS``, with its cubic outlines converted to quadratic
    ones (written atomically: several figure scripts share the cache)."""
    from fontTools import subset
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(src)
    opt = subset.Options()
    opt.layout_features = ["kern", "liga", "lnum", "tnum"]
    opt.name_IDs = ["*"]
    opt.notdef_outline = True
    opt.glyph_names = True
    sub = subset.Subsetter(opt)
    sub.populate(text=_CHARS)
    sub.subset(font)
    order = font.getGlyphOrder()
    gset = font.getGlyphSet()
    glyphs = {}
    for g in order:
        pen = TTGlyphPen(gset)
        gset[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[g] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, glyphs
    del font["CFF "]
    glyf.compile(font)
    for g, gl in glyf.glyphs.items():
        if hasattr(gl, "xMin"):
            font["hmtx"][g] = (font["hmtx"][g][0], gl.xMin)
    maxp = font["maxp"] = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions"):
        setattr(maxp, k, 0)
    maxp.maxComponentElements = max(len(getattr(g, "components", None) or []) for g in glyphs.values())
    maxp.compile(font)
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    font.sfntVersion = "\000\001\000\000"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.{os.getpid()}.tmp"
    font.save(tmp)
    try:
        os.replace(tmp, dst)          # atomic; other figure scripts read this cache too
    except OSError:
        os.remove(tmp)
        if not os.path.exists(dst):
            raise


def _stale(ttf, src):
    """True when the cached TrueType copy lacks a character of _CHARS that the OTF has."""
    if not os.path.exists(ttf):
        return True
    if not src:
        return False
    from fontTools.ttLib import TTFont
    try:
        with TTFont(ttf, lazy=True) as f:
            have = set(f.getBestCmap())
        with TTFont(src, lazy=True) as f:
            want = {ord(c) for c in _CHARS} & set(f.getBestCmap())
    except Exception:                                                   # noqa: BLE001
        return True
    return not want <= have


def use_poster_fonts():
    """Register Fira Sans (FONT_WEIGHTS) with matplotlib and make it the default family."""
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    added = []
    for w in FONT_WEIGHTS:
        ttf = os.path.join(FONT_CACHE, f"FiraSans-{w}.ttf")
        src = _kpsewhich(f"FiraSans-{w}.otf")
        if _stale(ttf, src):
            if src:
                try:
                    _otf_to_ttf(src, ttf)
                except Exception as e:                                  # noqa: BLE001
                    if not os.path.exists(ttf):
                        print(f"  font: TrueType conversion of {w} failed ({e}); using the OTF (CID Type 0C in PDF)")
                        ttf = src
            elif not os.path.exists(ttf):
                ttf = None
        if ttf:
            fm.fontManager.addfont(ttf)
            added.append(os.path.basename(ttf))
    family = "Fira Sans" if added else "DejaVu Sans"
    mpl.rcParams.update({"font.family": family, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    print(f"  font: {family} {added}")
    return family


# ─────────────────────────────────────────────────────────────────────────────
#  IOC reader. pyintertidal.gauges.load_cached_ioc json-loads a whole cached
#  chunk (105 MB of JSON for 2023-2025, ~0.7 GB peak); this slices the
#  requested dates out of the raw bytes first and then applies the same rules
#  (majority sensor, drop missing, sort, de-duplicate).
# ─────────────────────────────────────────────────────────────────────────────
_REC = re.compile(rb'\{"slevel": (null|[-+0-9.eE]+), "stime": "(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", '
                  rb'"sensor": "([^"]*)"\}')


def _first_on_or_after(raw, day, last_day):
    """Byte offset in the raw IOC JSON of the first record dated on or after
    ``day`` (searched day by day up to ``last_day``), or -1."""
    d, end = np.datetime64(day, "D"), np.datetime64(last_day, "D")
    while d <= end:
        i = raw.find(b'"stime": "' + str(d).encode())
        if i >= 0:
            return raw.rfind(b"{", 0, i)
        d += 1
    return -1


def read_ioc(code, t0, t1):
    """IOC records of ``code`` with t0 <= time < t1 as a tidy (time, level_m) frame."""
    parts = []
    for path in sorted(glob.glob(os.path.join(GAUGES, f"ioc_{code}_*.json"))):
        with open(path, "rb") as f:
            raw = f.read()
        a = _first_on_or_after(raw, t0, t1)
        if a < 0:
            continue
        b = _first_on_or_after(raw, t1, np.datetime64(t1, "D") + 400)
        chunk = raw[a:(b if b >= 0 else len(raw))]
        del raw
        rows = _REC.findall(chunk)
        if len(rows) != chunk.count(b'"stime"'):
            raise ValueError(f"{path}: unexpected record layout")
        del chunk
        parts.append(pd.DataFrame({
            "time": pd.to_datetime([r[1].decode() for r in rows]),
            "level_m": [float(r[0]) if r[0] != b"null" else np.nan for r in rows],
            "sensor": [r[2].decode() for r in rows]}))
        del rows
    if not parts:
        raise FileNotFoundError(f"no cached IOC records for {code} in [{t0}, {t1})")
    df = pd.concat(parts, ignore_index=True)
    df = df[df["sensor"] == df["sensor"].value_counts().idxmax()]
    df = df.dropna().sort_values("time").drop_duplicates("time")
    df = df[(df["time"] >= pd.Timestamp(t0)) & (df["time"] < pd.Timestamp(t1))]
    return df[["time", "level_m"]].reset_index(drop=True)


def read_truth_csv(fname, t0, t1):
    """One estuary_truth gauge CSV (time_utc, level_m), rows with t0 <= time < t1, read in chunks."""
    out = []
    for ch in pd.read_csv(os.path.join(TRUTH, fname), usecols=["time_utc", "level_m"], chunksize=250_000):
        ch = ch[(ch["time_utc"] >= str(t0)) & (ch["time_utc"] < str(t1))]
        if len(ch):
            out.append(ch)
    df = pd.concat(out, ignore_index=True)
    df["time"] = pd.to_datetime(df.pop("time_utc"))
    return df.dropna().sort_values("time").drop_duplicates("time")[["time", "level_m"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Layout helpers (everything in millimetres at print size)
# ─────────────────────────────────────────────────────────────────────────────
def axes_mm(fig, x, y, w, h, W, H):
    """Axes placed in millimetres (x, y from the bottom-left corner) on a W x H mm figure."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def style_axes(ax, fs, spines=("left", "bottom")):
    """The poster's axis style: only ``spines`` drawn, slate ticks, tick labels at ``fs`` pt."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
        ax.spines[s].set_color(SLATE)
        ax.spines[s].set_linewidth(1.3)
    ax.tick_params(colors=SLATE, labelcolor=INK, labelsize=fs, width=1.3, length=7, pad=6)
    ax.tick_params(which="minor", colors=SLATE, width=1.1, length=4)


def text_row(fig, x_mm, y_mm, runs, W, H):
    """Left-to-right runs of (text, size, weight, colour, gap_after_mm); returns the end x (mm)."""
    r = fig.canvas.get_renderer()
    x = x_mm
    for txt, size, weight, colour, gap in runs:
        t = fig.text(x / W, y_mm / H, txt, fontsize=size, fontweight=weight, color=colour,
                     ha="left", va="baseline")
        x += t.get_window_extent(renderer=r).width / fig.dpi * 25.4 + gap
    return x


def key_line(fig, x_mm, y_mm, colour, W, H, length=11.0, lw=4.5):
    """Draw a legend line segment at (x_mm, y_mm); returns the x (mm) where it ends."""
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([x_mm / W, (x_mm + length) / W], [y_mm / H] * 2, color=colour, lw=lw,
                          solid_capstyle="butt"))
    return x_mm + length
