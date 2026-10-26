# -*- coding: utf-8 -*-
"""Conceptual figure for the paper: why one uniform tide level fails inside
an estuary (in the spirit of Granadeiro et al. 2021, Fig. 4, and of
docs/figures/interior_tide_schematic.png).

(a) Longitudinal profile of a ría at one instant of a rising tide. The tide
    arrives late inland, so the real water surface slopes down from the
    mouth, while every published method assumes the one uniform level the
    ocean model gives at the mouth (dashed red). The difference is the level
    error, largest at the head.
(b) The same profile on the ebb. The interior still lags, so it is now
    HIGHER than the mouth: the sign of the error flips with the limb.
(c) The lag tau(s) grows with distance from the mouth; MAREA estimates one
    lag per fixed 3-km band of geodesic distance.

Geometry follows docs/figures/interior_tide_schematic.png (its generator is
not in the repository; bed and lag shapes were re-derived from the image,
the channel stretched from 10 to 12 km so that four whole 3-km bands fit).
Water levels are a pure lag of a sinusoidal tide,
    h(s, t) = A sin(omega t - omega tau(s)),
with no attenuation, so the two profile panels share the same tau(s) and
the same mouth level, once on the rising and once on the falling limb.
The flood and ebb errors differ in size because a mid-limb instant shifted
by tau is not symmetric about the crest.

Pure matplotlib, no data. Writes docs/paper/figures/fig_schematic.png
(3.5 x 3.5 in, 200 dpi: one column). Run from the repository root:
    python -m experiments.paper_fig_schematic
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

INK = "#14313c"; TIDE = "#2a6f97"; RED = "#c1121f"; SAND = "#d8c39a"; MUD = "#b08968"
OUT = "docs/paper/figures/fig_schematic.png"

L_KM = 12.0              # channel length, mouth to head
BAND_KM = 3.0            # fixed band length of the method
A_M = 1.3                # tidal amplitude at the mouth
PERIOD_MIN = 12.42 * 60  # M2
TAU_HEAD_MIN = 90.0      # lag at the head
PHASE_RISING = 0.35      # omega t at the mouth on the rising limb (mid-limb)
PHASE_EBBING = np.pi - PHASE_RISING   # same mouth level, falling
S_ERR_KM = 8.0           # where the level error is marked
YMIN, YMAX = -2.3, 1.5


def bed(s):
    """Bed elevation (m): deep at the mouth, dry at the head, gentle terraces."""
    x = np.asarray(s, float) / L_KM
    return (-2.0 + 0.6 * x + 2.5 * x ** 2.2
            + 0.13 * np.sin(2 * np.pi * np.asarray(s, float) / 3.1 + 0.6)
            + 0.06 * np.sin(2 * np.pi * np.asarray(s, float) / 1.3))


def tau(s):
    """Lag (min) of the tide at distance s from the mouth, growing inland."""
    return TAU_HEAD_MIN * (np.asarray(s, float) / L_KM) ** 1.4


def level(s, phase):
    """Water level (m) at s when the mouth is at tidal phase `phase`."""
    return A_M * np.sin(phase - 2 * np.pi * tau(s) / PERIOD_MIN)


def profile(ax, phase, caption, err_label, rising, label_lines):
    s = np.linspace(0, L_KM, 800)
    b, z, h0 = bed(s), level(s, phase), float(level(0.0, phase))
    wet = z > b
    ax.fill_between(s, YMIN, b, color=SAND, lw=0)                       # bed
    ax.fill_between(s, b, z, where=wet, color=TIDE, alpha=0.30, lw=0)   # water
    ax.plot(s, b, color=MUD, lw=0.7)
    ax.plot(s[wet], z[wet], color=TIDE, lw=1.2)                         # real surface
    ax.axhline(h0, color=RED, ls=(0, (4, 2)), lw=0.9)                   # uniform level assumed
    # rising / ebbing marker at the mouth
    y0, y1 = (h0 - 0.4, h0 + 0.4) if rising else (h0 + 0.4, h0 - 0.4)
    ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0),
                arrowprops=dict(arrowstyle="-|>", color=TIDE, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0))
    # the level error at one inland point
    ze = float(level(S_ERR_KM, phase))
    ax.annotate("", xy=(S_ERR_KM, ze), xytext=(S_ERR_KM, h0),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(S_ERR_KM - 0.3, h0 + 0.42 * (ze - h0), err_label, color=RED, fontsize=6, ha="right", va="center")
    if label_lines:
        ax.text(1.1, h0 + 0.09, "uniform level assumed", color=RED, fontsize=6, ha="left", va="bottom")
        ax.text(1.1, float(level(1.1, phase)) - 0.09, "real level, arriving late", color=INK, fontsize=6,
                ha="left", va="top")
        ax.text(0.3, float(bed(0.3)) + 0.12, "MOUTH\nτ = 0", color=INK, fontsize=6, ha="left", va="bottom")
        ax.text(L_KM - 0.15, float(bed(L_KM)) + 0.12, "HEAD", color=INK, fontsize=6, ha="right", va="bottom")
    ax.text(0.012, 0.96, caption, transform=ax.transAxes, fontsize=7, color=INK, ha="left", va="top")
    ax.set_xlim(0, L_KM); ax.set_ylim(YMIN, YMAX)
    ax.set_yticks([-2, -1, 0, 1]); ax.set_ylabel("elevation (m)")


def lag_panel(ax):
    s = np.linspace(0, L_KM, 300)
    nb = int(round(L_KM / BAND_KM))
    for k in range(1, nb):
        ax.axvline(k * BAND_KM, color="0.75", lw=0.6, ls=":")
    centres = (np.arange(nb) + 0.5) * BAND_KM
    ax.plot(s, tau(s), color=TIDE, lw=1.2)
    ax.plot(centres, tau(centres), "o", ms=3.4, mfc=TIDE, mec="white", mew=0.5)
    ax.annotate("one lag per band", xy=(centres[1], float(tau(centres[1]))), xytext=(0.5, 50),
                fontsize=6, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5, shrinkB=3))
    ax.text(0.012, 0.96, "(c) the lag grows inland", transform=ax.transAxes, fontsize=7, color=INK,
            ha="left", va="top", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xlim(0, L_KM); ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, L_KM + 0.1, BAND_KM)); ax.set_yticks([0, 30, 60, 90])
    ax.set_xlabel("distance from the mouth s (km)"); ax.set_ylabel("lag τ (min)")


def main():
    plt.rcParams.update({
        "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(3.5, 3.5), dpi=200, sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.85], hspace=0.14))
    fig.subplots_adjust(left=0.125, right=0.985, top=0.985, bottom=0.105)
    profile(a, PHASE_RISING, "(a) rising tide", "level error", rising=True, label_lines=True)
    profile(b, PHASE_EBBING, "(b) ebbing tide: interior still lags, now higher", "error flips sign",
            rising=False, label_lines=False)
    lag_panel(c)
    for ax in (a, b):
        ax.tick_params(labelbottom=False)
    for ax in (a, b, c):
        ax.yaxis.set_label_coords(-0.095, 0.5)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    h0r, h0e = float(level(0.0, PHASE_RISING)), float(level(0.0, PHASE_EBBING))
    print(f"-> {OUT}  mouth level {h0r:+.2f} m both panels; level error at {S_ERR_KM:.1f} km: "
          f"rising {float(level(S_ERR_KM, PHASE_RISING)) - h0r:+.2f} m, "
          f"ebbing {float(level(S_ERR_KM, PHASE_EBBING)) - h0e:+.2f} m; "
          f"head {float(level(L_KM, PHASE_RISING)) - h0r:+.2f} / {float(level(L_KM, PHASE_EBBING)) - h0e:+.2f} m")


if __name__ == "__main__":
    main()
