# -*- coding: utf-8 -*-
"""CARTOGRASS 2026 poster, figure B1 -- "The tide arrives late".

Measured water level at the mouth of the Ems (Borkum Fischerbalje, IOC) and
inside the estuary (Delfzijl, Rijkswaterstaat; Emden Neue Seeschleuse, WSV)
over one tide (low water, high water, low water; 16 h) on one UTC time axis:
high water reaches the inner gauges later, and the tide is larger there.
Two double arrows mark how much later the high water arrives inside.

Run from the repository root (one process; measured peak working set about 0.4 GB):

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/b1_tide_arrives_late.py
        [--start YYYY-MM-DD] [--days 2] [--year 2025] [--hours 16] [--no-source]

Inputs (read only)
    data_v4/gauges/ioc_bork_*.json                          IOC Borkum, 1-min, gauge zero
    data_v4/gauges/estuary_truth/ems__rws_delfzijl.csv      RWS Delfzijl, 10-min (F007), NAP
    data_v4/gauges/estuary_truth/ems__wsv_emden_neue_seeschleuse_3970010.csv
                                                            WSV Emden, 1-min, NHN, already QC'd
    data_v4/gauges/ioc_stations.json, estuary_truth/stations.json   (names, coordinates)
Outputs
    poster/figures/cg/b1_tide_arrives_late.{pdf,png}  211 x 132 mm (the concept-row slot,
                                                      column B of cartograss2026.tex); PNG at 300 dpi
    poster/build/cg/b1_tide_arrives_late.json         every number behind the figure

How the numbers are made
    * Same treatment for the three gauges: 10-min centred means (the RWS F007
      convention Delfzijl is delivered in: mean of the previous 5 and next 5
      minutes), so smoothing cannot shift one gauge's timing against another.
      Borkum is despiked with pyintertidal.gauges.despike (imported, not copied).
    * Datums differ (IOC gauge zero, NAP, NHN): each record is shown minus its
      own mean over the calendar year of the window.
    * Delay on the figure = the delay of the high water shown: the time of
      each inner high water minus the time of the Borkum one, turning points
      timed by a parabola fitted to the 10-min means within +-60 min. A high
      water is what the eye follows on the curves, so the arrows and their
      numbers measure exactly what they point at. The best-aligning lag (the
      whole-curve shift MAREA estimates per section: the shift of the Borkum
      curve that maximises its correlation with the inner record; 1-min
      steps, parabolic refinement) is written to the JSON for the window
      shown, the two-day window and the whole year, together with the delay
      of every high and low water. In July 2025 it lies between the low-water
      and the high-water delays (Delfzijl about 55 min, Emden about 72 min).
    * Range on the figure: mean |HW - LW| of consecutive turning points in the
      window shown.
    * Window shown: the first Borkum high water of the clean window, from
      7.5 h before it to --hours - 7.5 h after it (16 h: low water, high
      water, low water at all three gauges).
    * Window (unless --start is given): the most recent run of --days days in
      --year that is complete (no gap > 10 min in any record), at spring tide
      (every day's Borkum range >= 93 % of the largest daily range within
      +-7 days), calm (the 24 h 50 min mean of every gauge within 0.10 m of
      its 31-day running median) and free of glitches (both 1-min records
      within 0.08 m of their 11-min running median).
    * Display: 10-min means as computed (about 100 points per curve), no
      smoothing; levels in m with each record's 2025 mean as zero.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import (DEEP, GAUGES, INK, RULE, SAND, SLATE, TEAL, TRUTH, MM,     # noqa: E402
                      axes_mm, key_line, read_ioc, read_truth_csv, style_axes, text_row,
                      use_poster_fonts)

OUT_FIG = "poster/figures/cg/b1_tide_arrives_late"
OUT_JSON = "poster/build/cg/b1_tide_arrives_late.json"
COL = {"Borkum": DEEP, "Delfzijl": TEAL, "Emden": SAND}   # deep water -> shallow water -> flat
# Text in sand on white is set in the darker sand of the poster's texts
# (cg_macros.tex PSandText, the flow chart's SAND_DARK; 5.6:1 on white, PSand
# 3.0:1): the Emden curve and arrow keep PSand, its "+86 min" label does not.
SAND_DARK = "#8A5F22"
COL_TEXT = {"Borkum": DEEP, "Delfzijl": TEAL, "Emden": SAND_DARK}


def station_meta():
    """``{gauge: (lat, lon, name)}`` of Borkum (IOC station list) and of
    Delfzijl and Emden (the in-estuary station list)."""
    with open(os.path.join(GAUGES, "ioc_stations.json"), encoding="utf-8") as f:
        ioc = {r.get("Code"): r for r in json.load(f) if isinstance(r, dict)}
    with open(os.path.join(TRUTH, "stations.json"), encoding="utf-8") as f:
        tru = {r["code"]: r for r in json.load(f)}
    em = tru["emden_neue_seeschleuse_3970010"]
    return {"Borkum": (ioc["bork"]["Lat"], ioc["bork"]["Lon"], ioc["bork"]["Location"]),
            "Delfzijl": (tru["delfzijl"]["lat"], tru["delfzijl"]["lon"], tru["delfzijl"]["name"]),
            "Emden": (em["lat"], em["lon"], em["name"])}


def km_between(a, b):
    """Great-circle distance (km) between two (lat, lon, ...) tuples."""
    (la1, lo1), (la2, lo2) = a[:2], b[:2]
    p1, p2, dl = math.radians(la1), math.radians(la2), math.radians(lo2 - lo1)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


# ─────────────────────────────────────────────────────────────────────────────
#  Series operations (times in minutes since the start of the year)
# ─────────────────────────────────────────────────────────────────────────────
def to_minutes(times, t_ref):
    """Times as float minutes since ``t_ref``."""
    return (pd.DatetimeIndex(times).values - np.datetime64(t_ref)) / np.timedelta64(1, "m")


def f007(t_min, h, grid):
    """10-min centred means on ``grid``: mean of the samples within +-5 min (RWS F007)."""
    out = np.full(len(grid), np.nan)
    lo = np.searchsorted(t_min, grid - 5, side="left")
    hi = np.searchsorted(t_min, grid + 5, side="right")
    cs = np.r_[0.0, np.cumsum(h)]
    n = hi - lo
    ok = n >= 6                                    # at least 6 of the 11 minutes present
    out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / n[ok]
    return out


def max_gap(t_min, w0, w1):
    """Longest gap (min) between samples inside the window [w0, w1], its ends included."""
    i0, i1 = np.searchsorted(t_min, w0), np.searchsorted(t_min, w1)
    return float(np.max(np.diff(np.r_[w0, t_min[i0:i1], w1])))


def turning_points(t, h, kind, half=60.0):
    """Times and levels of high (kind='HW') or low ('LW') waters of a 10-min series."""
    from scipy.signal import find_peaks

    sgn = 1.0 if kind == "HW" else -1.0
    y = np.where(np.isfinite(h), sgn * h, np.nanmin(sgn * h) - 10.0)
    idx, _ = find_peaks(y, distance=int(8 * 6), prominence=0.5)
    out = []
    for i in idx:
        k = np.abs(t - t[i]) <= half
        if k.sum() < 9 or not np.isfinite(h[k]).all():
            continue
        x = t[k] - t[i]
        c = np.polyfit(x, h[k], 2)
        if sgn * c[0] >= 0:
            continue
        xv = -c[1] / (2 * c[0])
        if abs(xv) <= half / 2:
            out.append((t[i] + xv, float(np.polyval(c, xv))))
    return np.array(out).reshape(-1, 2)


def pair(ref, inner, max_lag=240.0):
    """Delay (min) of the first inner turning point after each reference one, with both levels."""
    res = []
    for tr, hr in ref:
        j = np.flatnonzero((inner[:, 0] > tr - 30) & (inner[:, 0] < tr + max_lag))
        if len(j):
            res.append((tr, inner[j[0], 0] - tr, inner[j[0], 1], hr))
    return np.array(res).reshape(-1, 4)


def best_lag(t_ref, h_ref, t_in, h_in, lags=np.arange(-60, 241)):
    """Shift (min) of the reference curve that maximises its correlation with the inner one."""
    ok = np.isfinite(h_in)
    ti, hi = t_in[ok], h_in[ok]
    r = []
    for L in lags:
        hr = np.interp(ti - L, t_ref, h_ref, left=np.nan, right=np.nan)
        m = np.isfinite(hr)
        r.append(np.corrcoef(hr[m], hi[m])[0, 1])
    r = np.array(r)
    i = int(np.nanargmax(r))
    d = 0.0
    if 0 < i < len(r) - 1:
        den = r[i - 1] - 2 * r[i] + r[i + 1]
        d = 0.5 * (r[i - 1] - r[i + 1]) / den if den != 0 else 0.0
    return float(lags[i] + d), float(r[i])


def tidal_range(tp_hw, tp_lw):
    """Mean height difference (m) between consecutive high and low waters."""
    ev = sorted([(t, h, 1) for t, h in tp_hw] + [(t, h, -1) for t, h in tp_lw])
    rng = [abs(a[1] - b[1]) for a, b in zip(ev[:-1], ev[1:]) if a[2] != b[2]]
    return float(np.mean(rng)) if rng else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
#  Window choice
# ─────────────────────────────────────────────────────────────────────────────
def choose_window(series, year, days):
    """Most recent clean spring-tide run of ``days`` days in ``year`` (rules in the docstring)."""
    n_days = int((np.datetime64(f"{year + 1}-01-01") - np.datetime64(f"{year}-01-01")) / np.timedelta64(1, "D"))
    ten = {k: v["g"] for k, v in series.items()}                   # 10-min means on the year grid
    gb = ten["Borkum"]
    rngB = np.full(n_days, np.nan)
    for d in range(n_days):
        seg = gb[d * 144:(d + 1) * 144]
        if np.isfinite(seg).sum() > 120:
            rngB[d] = np.nanmax(seg) - np.nanmin(seg)
    spring = rngB / pd.Series(rngB).rolling(15, center=True, min_periods=8).max().to_numpy()
    calm_dev = {}
    for k, g in ten.items():
        lp = pd.Series(g).rolling(149, center=True, min_periods=120).mean()     # 24 h 50 min
        med = lp.rolling(31 * 144, center=True, min_periods=10 * 144).median()
        calm_dev[k] = (lp - med).abs().to_numpy()
    glitch = {}
    for k in ("Borkum", "Emden"):
        s = pd.Series(series[k]["h"])
        glitch[k] = (s - s.rolling(11, center=True, min_periods=6).median()).abs().to_numpy()
    for d0 in range(n_days - days, -1, -1):
        w0, w1 = d0 * 1440.0, (d0 + days) * 1440.0
        if not all(max_gap(v["t"], w0, w1) <= 10.0 for v in series.values()):
            continue
        sp = spring[d0:d0 + days]
        if not (np.isfinite(sp).all() and sp.min() >= 0.93):
            continue
        g0, g1 = d0 * 144, (d0 + days) * 144
        if not all(np.nanmax(calm_dev[k][g0:g1]) <= 0.10 for k in ten):
            continue
        if any(np.nanmax(glitch[k][np.searchsorted(series[k]["t"], w0):np.searchsorted(series[k]["t"], w1)]) > 0.08
               for k in glitch):
            continue
        day = np.datetime64(f"{year}-01-01") + d0
        return str(day), {"spring_index_min": float(sp.min()),
                          "calm_max_dev_m": {k: float(np.nanmax(calm_dev[k][g0:g1])) for k in ten}}
    raise SystemExit(f"no window in {year} passes the cleanliness rules; pass --start")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Read the three gauges, choose the window, time the high waters, draw
    the figure and write the numbers behind it (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--start", default=None, help="first day of the window (YYYY-MM-DD); default: automatic")
    ap.add_argument("--days", type=int, default=2, help="length of the clean window searched (days)")
    ap.add_argument("--hours", type=float, default=16.0, help="length of the window shown (h)")
    ap.add_argument("--no-source", action="store_true", help="omit the in-figure data-source line")
    ap.add_argument("--out", default=OUT_FIG, help="output path without extension (variants, tests)")
    a = ap.parse_args()
    out_json = OUT_JSON if a.out == OUT_FIG else a.out + ".json"

    year = a.year
    t0, t1 = f"{year}-01-01", f"{year + 1}-01-01"
    print(f"reading {year} ...")
    from pyintertidal.gauges import despike

    recs = {"Borkum": despike(read_ioc("bork", t0, t1)),
            "Delfzijl": read_truth_csv("ems__rws_delfzijl.csv", t0, t1),
            "Emden": read_truth_csv("ems__wsv_emden_neue_seeschleuse_3970010.csv", t0, t1)}
    grid = np.arange(0.0, (np.datetime64(t1) - np.datetime64(t0)) / np.timedelta64(1, "m"), 10.0)
    series = {}
    for k, df in recs.items():
        t = to_minutes(df["time"], t0)
        h = df["level_m"].to_numpy(float)
        if k == "Delfzijl":                       # delivered as F007 10-min means: its own instants
            g = np.full(len(grid), np.nan)
            pos = np.clip(np.searchsorted(grid, t), 0, len(grid) - 1)
            on = np.abs(grid[pos] - t) < 0.01
            g[pos[on]] = h[on]
        else:
            g = f007(t, h, grid)
        series[k] = {"t": t, "h": h, "g": g, "mean": float(np.mean(h)), "n": int(len(h))}
        print(f"  {k}: {len(h)} samples in {year}, mean {np.mean(h):+.3f} m (own datum)")
    del recs

    if a.start:
        start, choice = a.start, {"rule": "given on the command line"}
    else:
        start, choice = choose_window(series, year, a.days)
        choice["rule"] = "most recent window that is complete, spring, calm and glitch-free (module docstring)"
    print(f"  clean window: {start} + {a.days} d  {choice}")
    d0 = (np.datetime64(start) - np.datetime64(t0)) / np.timedelta64(1, "m")
    d1 = d0 + a.days * 1440.0
    # Borkum 10-min means on a 1-min grid (the reference curve for the lag search)
    tb1 = np.arange(-300.0, grid[-1] + 300.0, 1.0)
    hb1 = f007(series["Borkum"]["t"], series["Borkum"]["h"], tb1) - series["Borkum"]["mean"]

    def demeaned(lo, hi):
        """The 10-min grid between lo and hi, and each gauge minus its year mean there."""
        m = (grid >= lo) & (grid <= hi)
        return grid[m], {k: v["g"][m] - v["mean"] for k, v in series.items()}

    # every turning point of the clean window (paired below, all of them to the JSON)
    tgc, zc = demeaned(d0, d1 - 10.0)
    tpc = {k: {kind: turning_points(tgc, zc[k], kind) for kind in ("HW", "LW")} for k in zc}

    # the window shown: from 7.5 h before the first Borkum high water of the clean
    # window to --hours - 7.5 h after it, on the 10-min grid
    pre, post = 7.5 * 60.0, a.hours * 60.0 - 7.5 * 60.0
    room = [t for t, _ in tpc["Borkum"]["HW"] if t - pre >= d0 and t + post <= d1]
    if not room:
        raise SystemExit("no Borkum high water leaves room for the window shown; change --hours or --start")
    t_hwB = float(room[0])
    w0 = float(np.floor((t_hwB - pre) / 10.0) * 10.0)
    w1 = w0 + a.hours * 60.0
    tg, z = demeaned(w0, w1)
    tp = {k: {kind: turning_points(tg, z[k], kind) for kind in ("HW", "LW")} for k in z}

    def utc(m):
        """Minutes since the start of the year as 'YYYY-MM-DDTHH:MM' (UTC)."""
        return str(np.datetime64(t0) + np.timedelta64(int(round(m * 60)), "s"))[:16]

    meta = station_meta()
    hwB = tpc["Borkum"]["HW"][np.argmin(np.abs(tpc["Borkum"]["HW"][:, 0] - t_hwB))]
    res = {"figure": OUT_FIG,
           "window_shown_utc": [utc(w0), utc(w1)],
           "clean_window_utc": [str(np.datetime64(start)), str(np.datetime64(start) + a.days)],
           "choice": choice,
           "treatment": f"10-min centred means (RWS F007) for all three; each minus its own {year} mean",
           "datums": {"Borkum": "IOC gauge zero", "Delfzijl": "NAP", "Emden": "NHN (PNP + W)"},
           "means_m_own_datum": {k: v["mean"] for k, v in series.items()},
           "delay_on_figure": "delay of the high water shown: inner HW time minus Borkum HW time "
                              "(turning points timed by a parabola on the 10-min means, +-60 min)",
           "borkum_hw_shown": {"utc": utc(hwB[0]), "level_m": float(hwB[1])},
           "gauges": {}}
    for k in ("Borkum", "Delfzijl", "Emden"):
        g = {"name": meta[k][2], "lat_lon": meta[k][:2],
             "range_m_window_shown": tidal_range(tp[k]["HW"], tp[k]["LW"]),
             "range_m_clean_window": tidal_range(tpc[k]["HW"], tpc[k]["LW"])}
        if k != "Borkum":
            p_hw = pair(hwB[None, :], tpc[k]["HW"])
            if not len(p_hw):
                raise SystemExit(f"no {k} high water follows the Borkum one at {utc(hwB[0])}")
            g["hw_shown_utc"] = utc(hwB[0] + p_hw[0, 1])
            g["hw_shown_level_m"] = float(p_hw[0, 2])
            g["hw_shown_delay_min"] = float(p_hw[0, 1])
            for tag, (lo, hi, tt, zz) in {"window_shown": (w0, w1, tg, z[k]),
                                          "clean_window": (d0, d1, tgc, zc[k])}.items():
                kb = (tb1 >= lo - 300) & (tb1 < hi + 300)
                L, r = best_lag(tb1[kb], hb1[kb], tt, zz)
                g[f"best_lag_min_{tag}"], g[f"best_lag_r_{tag}"] = L, r
            Ly, ry = best_lag(tb1, hb1, grid, series[k]["g"] - series[k]["mean"])
            g["best_lag_min_whole_year"], g["best_lag_r_whole_year"] = Ly, ry
            for kind in ("HW", "LW"):
                pp = pair(tpc["Borkum"][kind], tpc[k][kind])
                g[f"{kind}_delay_min_each_clean_window"] = [round(float(x), 1) for x in pp[:, 1]]
                g[f"{kind}_delay_min_mean_clean_window"] = float(np.mean(pp[:, 1]))
            g["km_from_borkum_straight_line"] = km_between(meta["Borkum"], meta[k])
        res["gauges"][k] = g
    for k, g in res["gauges"].items():
        print(f"  {k}: " + ", ".join(f"{kk}={vv:.2f}" if isinstance(vv, float) else f"{kk}={vv}"
                                     for kk, vv in g.items() if kk not in ("name", "lat_lon")))

    # ── draw ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import MultipleLocator

    use_poster_fonts()
    FS, FS_KEY = 20, 22                                  # pt at print size (>= 20 everywhere)
    W, H = 211.0, 132.0
    L_, R_, T_ = 25.0, 4.0, 22.0
    B_ = 30.5 if not a.no_source else 22.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = axes_mm(fig, L_, B_, W - L_ - R_, H - B_ - T_, W, H)
    style_axes(ax, FS)

    def as_date(m):
        """Minutes since the start of the year as datetime64 (for the time axis)."""
        return np.datetime64(t0) + (np.asarray(m) * 60).astype("timedelta64[s]")

    td = as_date(tg)
    for k in ("Emden", "Delfzijl", "Borkum"):                      # Borkum drawn last, on top
        ax.plot(td, z[k], color=COL[k], lw=3.2, solid_capstyle="round", solid_joinstyle="round", zorder=3)

    # the high water shown: dotted guides up from each peak, a double arrow per inner gauge
    peaks = {"Borkum": (hwB[0], hwB[1])}
    for k in ("Delfzijl", "Emden"):
        g = res["gauges"][k]
        peaks[k] = (hwB[0] + g["hw_shown_delay_min"], g["hw_shown_level_m"])
    top = max(v[1] for v in peaks.values())
    y_arrow = {"Delfzijl": top + 0.32, "Emden": top + 0.84}
    ylo = min(np.nanmin(v) for v in z.values())
    ax.set_ylim(ylo - 0.15, y_arrow["Emden"] + 0.42)
    guide = dict(color=SLATE, lw=1.3, ls=(0, (1.2, 1.6)), zorder=2, solid_capstyle="round")
    tB, hB = peaks["Borkum"]
    ax.plot([as_date(tB)] * 2, [hB + 0.06, y_arrow["Emden"]], **guide)
    for k in ("Delfzijl", "Emden"):
        tk, hk = peaks[k]
        ax.plot([as_date(tk)] * 2, [hk + 0.06, y_arrow[k]], **guide)
        ax.add_patch(FancyArrowPatch((mdates.date2num(as_date(tB)), y_arrow[k]),
                                     (mdates.date2num(as_date(tk)), y_arrow[k]),
                                     arrowstyle="<|-|>", mutation_scale=16, color=COL[k], lw=2.4,
                                     shrinkA=0, shrinkB=0, zorder=5))
        # both numbers in one column, right of the last guide, so neither sits on a guide
        t_lab = max(peaks["Delfzijl"][0], peaks["Emden"][0])
        ax.annotate(f"+{res['gauges'][k]['hw_shown_delay_min']:.0f} min",
                    xy=(mdates.date2num(as_date(t_lab)), y_arrow[k]), xytext=(2.2 * 72 / 25.4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=FS_KEY,
                    fontweight="semibold", color=COL_TEXT[k], zorder=6)

    ax.set_xlim(td[0], as_date(w1))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(axis="y", color=RULE, lw=1.1, zorder=0)
    ax.set_ylabel(f"Water level (m, {year} mean = 0)", fontsize=FS, color=INK, labelpad=8)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 4, 8, 12, 16, 20]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    day = str(np.datetime64(t0) + np.timedelta64(int(t_hwB // 1440), "D"))
    yy, mo, dd = (int(x) for x in day.split("-"))
    month = ("January February March April May June July August September October November "
             "December").split()[mo - 1]
    ax.set_xlabel(f"{dd} {month} {yy} (UTC)", fontsize=FS, color=INK, labelpad=6)

    # key: row 1 = the gauges (the mouth gauge named), row 2 = their tidal range on the
    # tide shown. "(mouth gauge)", not "(mouth)" (2026-10-19, round 4): MAREA's tide at
    # the mouth (the flow chart's h) is not this gauge but EOT20 at the site averaged
    # with it, so the key must not read as "h". Each entry is as wide as its longer
    # row (a range under a short name used to run under the next entry's line), and
    # the key starts at the plot's left edge less 4 mm, or further left when the
    # three entries need it (never left of 3 mm: the y label's top ends lower).
    y1, y2 = H - 8.0, H - 16.5
    KEY_LEN, KEY_PAD, KEY_GAP = 9.5, 2.5, 7.0
    rend = fig.canvas.get_renderer()

    def tw(s, size, weight):
        """Width (mm) of a text at this size and weight, measured on the figure."""
        t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(renderer=rend).width / fig.dpi * 25.4
        t.remove()
        return w

    entries = []
    for k, word in (("Borkum", "(mouth gauge)"), ("Delfzijl", None), ("Emden", None)):
        runs = [(k, FS_KEY, "semibold", INK, 1.8 if word else 0.0)] + \
               ([(word, FS_KEY, "normal", SLATE, 0.0)] if word else [])
        rng_txt = f"range {res['gauges'][k]['range_m_window_shown']:.1f} m"
        w = max(sum(tw(s, sz, wt) + g for s, sz, wt, _, g in runs), tw(rng_txt, FS, "normal"))
        entries.append((k, runs, rng_txt, w))
    total = sum(KEY_LEN + KEY_PAD + w for *_, w in entries) + KEY_GAP * (len(entries) - 1)
    x = min(L_ - 4.0, W - 1.5 - total)
    if x < 3.0:
        print(f"  WARNING: the key needs {total:.0f} mm, more than the {W - 4.5:.0f} mm available")
        x = 3.0
    for k, runs, rng_txt, w in entries:
        x = key_line(fig, x, y1 + 1.5, COL[k], W, H, length=KEY_LEN)
        x_text = x + KEY_PAD
        text_row(fig, x_text, y1, runs, W, H)
        text_row(fig, x_text, y2, [(rng_txt, FS, "normal", SLATE, 0.0)], W, H)
        x = x_text + w + KEY_GAP
    print(f"  key: {total:.1f} mm wide, from {x - KEY_GAP - total:.1f} to {x - KEY_GAP:.1f} mm "
          f"of the {W:.0f} mm figure")
    if not a.no_source:
        src = "Data: IOC (Borkum), Rijkswaterstaat (Delfzijl), WSV (Emden)"
        t = fig.text((L_ - 4.0) / W, 2.5 / H, src, fontsize=FS, color=SLATE, ha="left", va="baseline")
        fig.canvas.draw()
        right = (L_ - 4.0) + t.get_window_extent().width / fig.dpi * 25.4
        if right > W - 1.0:
            t.set_text("Data: IOC, Rijkswaterstaat, WSV")
            print(f"  data line shortened (the long form ran to {right:.0f} mm)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=300)
    plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    res["figure"] = a.out
    res["figure_size_mm"] = [W, H]
    res["points_per_curve"] = int(len(tg))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {a.out}.pdf/.png and {out_json}")


if __name__ == "__main__":
    main()
