# -*- coding: utf-8 -*-
"""CARTOGRASS 2026 poster, figure B2 -- "Decimetres become hours".

Emerged time as a function of ground elevation, from one year of measured sea
level at Gijon (IOC gij2, Puertos del Estado radar gauge), the gauge relevant
to the Cantabrian rias: a few decimetres of elevation change the time a flat
spends in the air by hours.

Run from the repository root (one process; measured peak working set about 0.5 GB):

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/b2_decimetres_hours.py
        [--year 2024] [--example -0.15 0.15] [--band ZLO ZHI] [--no-source]

Inputs (read only)
    data_v4/gauges/ioc_gij2_*.json     IOC Gijon, 1-min, gauge datum
Outputs
    poster/figures/cg/b2_decimetres_hours.{pdf,png}   250 x 140 mm; PNG at 300 dpi
    poster/build/cg/b2_decimetres_hours.json          every number behind the figure

How the numbers are made
    * The year's 1-min record, despiked with pyintertidal.gauges.despike
      (imported, not copied); 2024 is the default because it is the most
      complete year of the cache (2025 has multi-day gaps).
    * Heights are relative to the record's own mean over the year (the gauge
      datum, demeaned): "0" is the 2024 mean sea level at the gauge, not a
      geodetic datum.
    * Emerged fraction of ground at elevation z: the share of the year's
      samples with the water level below z. Measured levels, so surges,
      set-up and the spring-neap cycle are all included.
    * Hours per tidal cycle = emerged fraction x 12.42 h (the M2 period): the
      mean time in the air per semidiurnal tide.
    * Sensitivity: minutes of emersion per tide gained per 10 cm of elevation
      (centred difference over +-5 cm), summarised between the levels that are
      in the air 10 % and 90 % of the time.
    * The worked example (--example, default the 30 cm centred on mean sea
      level, where the curve is least steep) is drawn on the figure.
    * Habitat thresholds are for the seagrass co-author to decide: nothing is
      marked unless --band ZLO ZHI is given, which shades a generic band
      labelled "intertidal seagrass zone (site-specific)".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import (DEEP, INK, M2_HOURS, MM, RULE, SAND, SLATE, TEAL,           # noqa: E402
                      axes_mm, read_ioc, style_axes, use_poster_fonts)

OUT_FIG = "poster/figures/cg/b2_decimetres_hours"
OUT_JSON = "poster/build/cg/b2_decimetres_hours.json"


def main():
    """Read one year of the Gijón gauge, compute the emerged-time curve, draw
    the figure and write the numbers behind it (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--example", nargs=2, type=float, default=(-0.15, 0.15), metavar=("Z1", "Z2"),
                    help="the elevation step drawn as the worked example (m, relative to the year's mean)")
    ap.add_argument("--band", nargs=2, type=float, default=None, metavar=("ZLO", "ZHI"),
                    help="shade a generic 'intertidal seagrass zone (site-specific)' band (off by default)")
    ap.add_argument("--no-source", action="store_true", help="omit the in-figure data-source line")
    ap.add_argument("--out", default=OUT_FIG, help="output path without extension (variants, tests)")
    a = ap.parse_args()
    out_json = OUT_JSON if a.out == OUT_FIG else a.out + ".json"

    year = a.year
    t0, t1 = f"{year}-01-01", f"{year + 1}-01-01"
    print(f"reading Gijon {year} ...")
    from pyintertidal.gauges import despike

    raw = read_ioc("gij2", t0, t1)
    df = despike(raw)
    n_minutes = int((np.datetime64(t1) - np.datetime64(t0)) / np.timedelta64(1, "m"))
    h = df["level_m"].to_numpy(float)
    gaps = np.diff(df["time"].to_numpy()) / np.timedelta64(1, "m")
    msl = float(np.mean(h))
    zs = np.sort(h - msl)
    n = len(zs)
    print(f"  {len(raw)} samples, {len(raw) - n} removed by despike, coverage {n / n_minutes:.4f}, "
          f"longest gap {gaps.max() / 60:.1f} h, mean {msl:.3f} m (gauge datum)")
    del raw, df, h

    def emerged(z):
        """Share of the year with the water level below z (ground at z in the air)."""
        return np.searchsorted(zs, np.asarray(z, float), side="left") / n

    def z_at(frac):
        """The level (m, relative to the year's mean) exceeded by a share 1 - frac of the year."""
        return float(zs[min(int(round(frac * n)), n - 1)])

    zgrid = np.round(np.arange(np.floor(zs[0] * 100) / 100, np.ceil(zs[-1] * 100) / 100 + 0.005, 0.01), 2)
    E = emerged(zgrid)
    sens = (emerged(zgrid + 0.05) - emerged(zgrid - 0.05)) * M2_HOURS * 60.0      # min per tide per 10 cm
    z10, z90 = z_at(0.10), z_at(0.90)
    mid = (zgrid >= z10) & (zgrid <= z90)
    s_lo, s_hi = float(sens[mid].min()), float(sens[mid].max())
    z1, z2 = sorted(a.example)
    e1, e2 = float(emerged(z1)), float(emerged(z2))
    dh = (e2 - e1) * M2_HOURS
    res = {"figure": OUT_FIG, "gauge": "Gijon (IOC gij2, Puertos del Estado radar)", "year": year,
           "samples": n, "coverage": n / n_minutes, "longest_gap_h": float(gaps.max() / 60),
           "mean_level_gauge_datum_m": msl, "datum": f"gauge datum minus the {year} mean",
           "min_max_rel_m": [float(zs[0]), float(zs[-1])],
           "levels_in_air_10_50_90pct_m": [z10, z_at(0.5), z90],
           "sensitivity_min_per_tide_per_10cm_between_10_90pct": [s_lo, s_hi],
           "example": {"z1_m": z1, "z2_m": z2, "emerged_frac": [e1, e2],
                       "hours_per_tide": [e1 * M2_HOURS, e2 * M2_HOURS], "delta_hours_per_tide": dh,
                       "delta_hours_per_day": (e2 - e1) * 24.0, "delta_hours_per_year": (e2 - e1) * n_minutes / 60},
           "table": [{"z_m": float(z), "emerged_pct": float(100 * e), "hours_per_tide": float(e * M2_HOURS),
                      "min_per_tide_per_10cm": float(s)}
                     for z, e, s in zip(zgrid[::10], E[::10], sens[::10])],
           "band": a.band}
    print(f"  10/50/90 % in-air levels: {z10:+.2f} / {z_at(0.5):+.2f} / {z90:+.2f} m; "
          f"sensitivity {s_lo:.0f}-{s_hi:.0f} min per tide per 10 cm")
    print(f"  example {z1:+.2f} -> {z2:+.2f} m: {100 * e1:.1f} -> {100 * e2:.1f} % in air, "
          f"+{dh:.2f} h per tide, +{(e2 - e1) * 24:.2f} h per day")

    # ── draw ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, MultipleLocator

    use_poster_fonts()
    FS, FS_KEY = 20, 23
    W, H = 250.0, 140.0
    L_, R_, B_, T_ = 27.0, 13.0, 40.0 if not a.no_source else 32.0, 25.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = axes_mm(fig, L_, B_, W - L_ - R_, H - B_ - T_, W, H)
    style_axes(ax, FS, spines=("left", "bottom", "top"))
    x = 100.0 * E
    ylo, yhi = float(zgrid[0]) - 0.05, float(zgrid[-1]) + 0.05
    ax.set_xlim(0, 100)
    ax.set_ylim(ylo, yhi)
    # in the air (left of the curve) / under water (right of it), filled to the axes' edges
    zf, xf = np.r_[ylo, zgrid, yhi], np.r_[0.0, x, 100.0]
    ax.fill_betweenx(zf, 0, xf, color=SAND, alpha=0.16, lw=0, zorder=1)
    ax.fill_betweenx(zf, xf, 100, color=TEAL, alpha=0.10, lw=0, zorder=1)
    if a.band:
        blo, bhi = sorted(a.band)
        ax.axhspan(blo, bhi, color=TEAL, alpha=0.18, lw=0, zorder=1.5)
        ax.text(98.5, bhi - 0.04, "intertidal seagrass zone (site-specific)", ha="right", va="top",
                fontsize=FS, color=DEEP, zorder=6)
    ax.axhline(0.0, color=RULE, lw=1.2, zorder=1.8)
    ax.plot(x, zgrid, color=DEEP, lw=3.2, solid_capstyle="round", zorder=4)

    # the worked example: a step of ground (left) read off as a step of hours (top axis)
    SAND_DARK = "#8A5F22"                          # PSand darkened for text on the pale sand fill
    xs1, xs2 = 100 * e1, 100 * e2
    for zz, xx in ((z1, xs1), (z2, xs2)):
        ax.plot([0, xx], [zz, zz], color=INK, lw=1.4, ls=(0, (3, 2.5)), zorder=5)
        ax.plot([xx, xx], [zz, yhi], color=INK, lw=1.4, ls=(0, (3, 2.5)), zorder=5)
    ax.fill_betweenx([z1, z2], 0, [xs1, xs2], color=SAND, alpha=0.50, lw=0, zorder=2)
    ya = yhi - 0.20
    ax.annotate("", xy=(xs2, ya), xytext=(xs1, ya), zorder=6,
                arrowprops=dict(arrowstyle="<|-|>", color=DEEP, lw=2.0, mutation_scale=16, shrinkA=0, shrinkB=0))
    ax.text(xs2 + 1.5, ya, f"+{dh:.1f} h", ha="left", va="center", fontsize=FS_KEY + 3,
            fontweight="semibold", color=DEEP, zorder=6)
    ax.text(xs2 + 1.5, ya - 0.36, "more in the air per tide", ha="left", va="top", fontsize=FS, color=DEEP,
            zorder=6)
    ax.text(1.2, z2 + 0.07, f"+{100 * (z2 - z1):.0f} cm of ground", ha="left", va="bottom", fontsize=FS_KEY + 3,
            fontweight="semibold", color=SAND_DARK, zorder=6)
    ax.text(22, 1.55, "in the air", ha="center", va="center", fontsize=FS_KEY, color=SAND_DARK, zorder=6)
    ax.text(80, -1.55, "under water", ha="center", va="center", fontsize=FS_KEY, color=TEAL, zorder=6)
    ax.text(99, 0.05, f"{year} mean sea level", ha="right", va="bottom", fontsize=FS, color=SLATE, zorder=6)

    # axes: bottom = % of the year, top = hours per tidal cycle
    ax.xaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel(f"time in the air (share of {year})", fontsize=FS, color=INK, labelpad=6)
    ax.set_ylabel("ground elevation (m)", fontsize=FS, color=INK, labelpad=10)
    top = ax.secondary_xaxis("top", functions=(lambda p: p / 100 * M2_HOURS, lambda hh: hh / M2_HOURS * 100))
    top.xaxis.set_major_locator(FixedLocator([0, 2, 4, 6, 8, 10, 12]))
    top.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} h"))
    top.tick_params(colors=SLATE, labelcolor=INK, labelsize=FS, width=1.3, length=7, pad=4)
    top.spines["top"].set_color(SLATE)
    top.spines["top"].set_linewidth(1.3)
    top.set_xlabel("hours in the air per tidal cycle (12.4 h)", fontsize=FS, color=INK, labelpad=8)

    note = f"mid-intertidal: each 10 cm = {s_lo:.0f}\u2013{s_hi:.0f} min more in the air per tide"
    fig.text((L_ - 4.0) / W, (11.5 if not a.no_source else 3.5) / H, note, fontsize=FS, color=SLATE,
             ha="left", va="baseline")
    if not a.no_source:
        fig.text((L_ - 4.0) / W, 3.5 / H, f"Data: Gij\u00f3n tide gauge (IOC gij2, Puertos del Estado), 1-min, {year}",
                 fontsize=FS, color=SLATE, ha="left", va="baseline")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=300)
    plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    res["figure"] = a.out
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {a.out}.pdf/.png and {out_json}")


if __name__ == "__main__":
    main()
