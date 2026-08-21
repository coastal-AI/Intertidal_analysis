"""MAREA explained in ~85 seconds — a 3blue1brown-style animation.

Five scenes, every curve computed from the project's REAL data (no props):

  1. The problem   — the tide arrives late inside the ria; standard methods
                     assume one level per scene.
  2. The sensor    — a real Villaviciosa pixel: NDWI vs water level is a
                     sigmoid; its inflection IS the elevation, its width the
                     sub-pixel relief.
  3. The clock     — slide the boundary tide by tau and score the whole
                     band's wet/dry record: the profiled Bernoulli
                     likelihood picks the interior clock (real curve, real
                     minimum at the head of the ria).
  4. The theorem   — rescale (alpha, z, sigma) together and the binary data
                     cannot tell: phase is measurable, gain is not.
  5. The operator  — h(s,t) = h_B(t - tau(s)); the measured tau profile and
                     the final DEM.

Renders research/marea_method.mp4 (1280x720, 12 fps) via ffmpeg.
Run:  python research/anim_marea.py       (~6 min: data prep + render)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ── 3b1b-ish look ────────────────────────────────────────────────────────
BG = "#111318"
FG = "#e8e8e8"
BLUE = "#58c4dd"
YELLOW = "#f5d76e"
GREEN = "#83c167"
RED = "#fc6255"
GREY = "#8a8a8a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GREY, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": GREY, "ytick.color": GREY, "font.size": 13,
    "mathtext.fontset": "cm",
})

FPS = 12
print("preparing real data...", flush=True)

# ── real data prep ───────────────────────────────────────────────────────
from pyintertidal.net import use_system_certificates
use_system_certificates()
from eo_tides.model import model_tides
import pandas as pd
from pyintertidal import tide_estimators as te
from pyintertidal.marea import build_bank
import pyintertidal as pit

aoi = pit.sites.get("villaviciosa")
lat_c, lon_c = aoi.centroid

# dense tide for scene 1 (3 days at 10 min)
tt_dense = pd.date_range("2024-03-08", "2024-03-11", freq="10min")
h_dense = model_tides(x=[lon_c], y=[lat_c], time=tt_dense, model="EOT20",
                      directory="tide_models", crs="EPSG:4326",
                      extrapolate=True, cutoff=np.inf,
                      parallel=False).reset_index().sort_values(
    "time")["tide_height"].to_numpy(float)
t_h = (tt_dense - tt_dense[0]).total_seconds().to_numpy() / 3600.0

# the archive: one good real pixel + the head band for the likelihood
d = np.load("data_v4/store/ndwi_intermareal.npz", allow_pickle=True)
dates = np.array([str(s) for s in d["dates"]])
hr = np.load("data_v4/store/mareas_hora_real.npz")
ds_hr = np.array([str(s) for s in hr["dates"]])
have = np.isin(dates, ds_hr)
h0 = hr["tide"][np.searchsorted(ds_hr, dates[have])]
L = np.load("data_v4/store/lamina_base.npz")
good = (L["good"] & (L["b"] > 0.6) & (L["b"] < 1.2)
        & (L["z"] > -0.3) & (L["z"] < 0.5) & (L["sg"] < 0.16))
pix = int(np.where(good)[0][17])
y_pix = np.nan_to_num(d["Y"][have][:, pix], nan=np.nan)
c_pix = d["C"][have][:, pix] > 0
a_p, b_p, z_p, sg_p = (float(L[k][pix]) for k in ("a", "b", "z", "sg"))

# real profiled NLL(tau) of the head band
times = pit.overpass.get_overpass_times(
    aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
t_real = pd.DatetimeIndex(pd.to_datetime(
    [times[s] for s in dates[have] if s in times])).tz_localize(None)
mask_t = np.array([s in times for s in dates[have]])
bank, _ = build_bank(lat_c, lon_c, t_real)
geo = np.load("results/m1_geometry/geometry.npz")
s_keep = geo["s_keep"].astype(float) / 1000.0
head_all = np.where(s_keep > np.nanquantile(s_keep, 5 / 6))[0]
head = np.sort(np.random.default_rng(7).choice(head_all, 900,
                                               replace=False))
Yb = np.nan_to_num(d["Y"][have][:, head], nan=0.0)[mask_t]
Cb = (d["C"][have][:, head] > 0)[mask_t]
wet_b = Cb & (Yb > 0)
z_grid = np.linspace(h0.min() - 0.3, h0.max() + 0.3, 60)
TAUS = np.arange(-20.0, 91.0, 5.0)
NLL = np.array([te._band_nll(wet_b, Cb, bank.at(tv), z_grid, 0.2,
                             sg_grid=(0.06, 0.1, 0.15, 0.22, 0.32))[0]
                for tv in TAUS])
tau_best = float(TAUS[np.argmin(NLL)])
print(f"real NLL(tau) computed: minimum at {tau_best:+.0f} min", flush=True)

# real tau profile + DEM crop
mr = json.load(open("results/m2_real/result.json", encoding="utf-8"))
S_CENT = np.asarray(mr["centros_km"], float)
TAU_PROF = np.asarray(mr["veredicto"]["m2a_tau"]["real"], float)
TAU_USED = np.where(np.asarray(mr["veredicto"]["m2a_tau"]["fuera_del_nulo"]),
                    TAU_PROF, 0.0)
import rasterio
with rasterio.open("products_villaviciosa/hsr_v4_final_2023-2025.tif") as s:
    DEM = s.read(1)
rr, cc = np.where(np.isfinite(DEM))
DEM = DEM[rr.min():rr.max(), cc.min():cc.max()]

# scene 2 sigmoid support
hh = np.linspace(h0.min(), h0.max(), 200)
from scipy.special import erf
PHI = lambda x: 0.5 * (1 + erf(x / np.sqrt(2)))

# scene 3 extras: 14 real head-band pixels, spread in elevation
ok_h = L["good"][head] & np.isfinite(L["z"][head])
cand = head[ok_h]
order = np.argsort(L["z"][cand])
SEL = cand[order[np.linspace(5, len(order) - 6, 14).astype(int)]]
SEL_Z = L["z"][SEL]
Ysel = np.nan_to_num(d["Y"][have][:, SEL], nan=np.nan)[mask_t]
Csel = (d["C"][have][:, SEL] > 0)[mask_t]
WETSEL = Csel & (Ysel > 0)
DRYSEL = Csel & ~(Ysel > 0)

# ── timeline ─────────────────────────────────────────────────────────────
SCENES = [("s1", 13), ("s2", 27), ("s3", 29), ("s4", 15), ("s5", 19)]
TOTAL = sum(n for _, n in SCENES) * FPS
starts = np.cumsum([0] + [n * FPS for _, n in SCENES])

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)


def title(ax, txt, sub=""):
    ax.text(0.02, 1.06, txt, transform=ax.transAxes, fontsize=17,
            color=FG, weight="bold")
    if sub:
        ax.text(0.02, 0.99, sub, transform=ax.transAxes, fontsize=12,
                color=GREY, va="top")


def ease(u):
    return 3 * u ** 2 - 2 * u ** 3


def s1(u):
    ax = fig.add_axes([0.09, 0.12, 0.86, 0.72])
    title(ax, "1 · The problem: inside the ria, the tide arrives late",
          "standard methods assume ONE water level per scene, from an "
          "ocean model at the mouth")
    n = max(2, int(ease(min(u * 1.6, 1)) * len(t_h)))
    ax.plot(t_h[:n], h_dense[:n], color=BLUE, lw=2.5,
            label=r"mouth:  $h_B(t)$  (ocean model)")
    if u > 0.35:
        v = ease(min((u - 0.35) / 0.4, 1))
        lag = 26.0 / 60.0 * v
        ax.plot(t_h[:n] + lag, h_dense[:n], color=YELLOW, lw=2.5,
                alpha=v, label=r"head of the ria:  $h_B(t-\tau)$")
        if v > 0.9:
            i0 = int(0.30 * len(t_h))
            ax.annotate(r"$\tau \approx$ 26 min", color=YELLOW, fontsize=15,
                        xy=(t_h[i0] + lag, h_dense[i0]),
                        xytext=(t_h[i0] + 6, h_dense[i0] + 0.9),
                        arrowprops=dict(color=YELLOW, arrowstyle="->"))
    if u > 0.75:
        ax.axhline(0.4, color=GREEN, ls="--", lw=1.5)
        ax.text(t_h[-1] * 0.995, 0.47, "a pixel at elevation z",
                color=GREEN, ha="right", fontsize=12)
        ax.text(0.5, -0.13,
                "minutes of clock error  =  tens of cm of water level "
                "assigned to every scene", transform=ax.transAxes,
                ha="center", color=RED, fontsize=13)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("water level (m)")
    ax.set_xlim(0, t_h[-1])
    ax.set_ylim(-2.4, 2.6)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.1)


def s2(u):
    ax = fig.add_axes([0.09, 0.12, 0.86, 0.72])
    title(ax, "2 · Every pixel is a threshold sensor",
          "one REAL Villaviciosa pixel: its NDWI in 465 scenes, against the "
          "water level h at each overpass (from the tide model)")
    ok = c_pix & np.isfinite(y_pix)
    n = max(3, int(ease(min(u / 0.28, 1)) * ok.sum()))
    idx = np.where(ok)[0][:n]
    ax.scatter(h0[idx], y_pix[idx], s=16,
               c=[BLUE if w else "#b08046" for w in (y_pix[idx] > 0)],
               alpha=0.65)
    # phase A: the two states of the sensor
    if u > 0.10:
        ax.axhline(a_p, color="#b08046", ls=":", lw=1.5)
        ax.text(h0.min() + 0.06, a_p - 0.10,
                "seen DRY  ->  NDWI = a   (its bare sediment)",
                color="#c89b5f", fontsize=12)
        ax.axhline(a_p + b_p, color=BLUE, ls=":", lw=1.5)
        ax.text(h0.min() + 0.06, a_p + b_p + 0.05,
                "seen WET  ->  NDWI = a + b   (water)",
                color=BLUE, fontsize=12)
    # phase B: slide a candidate step z and watch the residuals
    if 0.32 < u <= 0.66:
        v = ease((u - 0.32) / 0.34)
        z_try = (h0.min() + 0.5) + v * (z_p - (h0.min() + 0.5))
        ax.plot(hh, a_p + b_p * PHI((hh - z_try) / sg_p), color=YELLOW, lw=3)
        for j in idx[::6]:
            pj = a_p + b_p * PHI((h0[j] - z_try) / sg_p)
            ax.plot([h0[j], h0[j]], [y_pix[j], pj], color=RED, lw=0.9,
                    alpha=0.8)
        ax.text(0.03, 0.90, "THE FIT: slide the step z until the residuals "
                            "(red) shrink", transform=ax.transAxes,
                fontsize=13, color=RED)
        ax.text(0.03, 0.83, f"trying   z = {z_try:+.2f} m",
                transform=ax.transAxes, fontsize=14, color=YELLOW)
    # phase C: the fitted sigmoid, annotated
    if u > 0.66:
        ax.plot(hh, a_p + b_p * PHI((hh - z_p) / sg_p), color=YELLOW, lw=3)
        ax.text(0.03, 0.87,
                r"$\mathrm{NDWI}(h) = a + b\,\Phi\!\left(\frac{h - z}"
                r"{\sigma}\right)$",
                transform=ax.transAxes, fontsize=19, color=YELLOW)
        ax.axvline(z_p, color=GREEN, ls="--", lw=2)
        ax.text(z_p + 0.06, -0.66,
                f"the step sits at h = z = {z_p:+.2f} m:\n"
                "the water level at which THIS pixel floods\n"
                "=  its ELEVATION",
                color=GREEN, fontsize=13)
        if u > 0.84:
            ax.annotate("", xy=(z_p - sg_p, 0.55),
                        xytext=(z_p + sg_p, 0.55),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
            ax.text(z_p, 0.62,
                    rf"width $\sigma$ = {100*sg_p:.0f} cm of relief INSIDE "
                    "the 10 m pixel", color=RED, ha="center", fontsize=13)
            ax.text(0.5, -0.135,
                    "a flat pixel flips like a switch (small σ); a rough one "
                    "floods bit by bit (large σ)",
                    transform=ax.transAxes, ha="center", color=GREY,
                    fontsize=12)
    ax.set_xlabel("water level h at overpass (m)")
    ax.set_ylabel("NDWI")
    ax.set_xlim(h0.min(), h0.max())
    ax.set_ylim(-0.8, 0.9)


def s3(u):
    axL = fig.add_axes([0.07, 0.13, 0.42, 0.62])
    axR = fig.add_axes([0.58, 0.13, 0.38, 0.62])
    title(axL, "3 · The clock: ONE lag must explain the whole band",
          "left: 14 real head-band pixels (rows, sorted by elevation); "
          "each dot = one scene placed at the level the candidate clock "
          "claims  ·  blue = seen wet, brown = seen dry")
    # tau sweeps -20 -> tau_best over the first 70 % of the scene
    v = ease(min(u / 0.70, 1))
    tau_now = -20 + v * (tau_best + 20)
    hs = bank.at(tau_now)
    for row in range(len(SEL)):
        w, dr = WETSEL[:, row], DRYSEL[:, row]
        axL.scatter(hs[w], np.full(int(w.sum()), row), s=9, color=BLUE,
                    alpha=0.7)
        axL.scatter(hs[dr], np.full(int(dr.sum()), row), s=9,
                    color="#b08046", alpha=0.55)
        axL.plot(SEL_Z[row], row, marker="|", color=GREEN, ms=15, mew=2.5)
    axL.text(0.03, 1.005, rf"$\tau$ = {tau_now:+.0f} min", color=YELLOW,
             fontsize=16, transform=axL.transAxes, va="bottom")
    axL.text(0.0, -0.185,
             "right clock -> every row splits cleanly at its green z;  "
             "wrong clock -> wet and dry mix",
             transform=axL.transAxes, fontsize=12, color=GREY)
    axL.set_xlabel(r"level under the candidate clock  $h_B(t-\tau)$  (m)")
    axL.set_ylabel("pixel (sorted by elevation)")
    axL.set_xlim(h0.min() - 0.2, h0.max() + 0.2)
    # right: the likelihood curve traced as tau advances
    m = max(2, int(v * len(TAUS)))
    axR.plot(TAUS[:m], NLL[:m], color=BLUE, lw=2.5)
    axR.plot(TAUS[m - 1], NLL[m - 1], "o", color=YELLOW, ms=10)
    if u > 0.72:
        axR.axvline(tau_best, color=GREEN, ls="--", lw=2)
        axR.text(tau_best + 2,
                 np.min(NLL) + 0.55 * (np.max(NLL) - np.min(NLL)),
                 rf"$\hat\tau$ = {tau_best:+.0f} min" + "\n(real data)",
                 color=GREEN, fontsize=13)
    if u > 0.80:
        axR.text(-0.08, -0.20,
                 r"score:  $\mathcal{L}(\tau)=\sum_p \max_{z_p,\sigma_p}"
                 r"\sum_t \log\left[P_{pt}^{y_{pt}}\,(1-P_{pt})"
                 r"^{1-y_{pt}}\right]$",
                 transform=axR.transAxes, fontsize=13, color=FG)
        axR.text(-0.08, -0.275,
                 r"$P_{pt}=\Phi\!\left(\frac{h_B(t-\tau)-z_p}{\sigma_p}"
                 r"\right)$  = P if seen wet, 1$-$P if seen dry;"
                 "  each pixel granted its best (z, σ) — 'profiled'",
                 transform=axR.transAxes, fontsize=11, color=GREY)
    axR.set_xlabel(r"candidate clock $\tau$ (min)")
    axR.set_ylabel("negative log-likelihood")


def s4(u):
    title_ax = fig.add_axes([0, 0, 1, 1]); title_ax.axis("off")
    title_ax.text(0.05, 0.93, "4 · The theorem: what binary data can NEVER "
                              "see", fontsize=17, weight="bold", color=FG)
    title_ax.text(0.05, 0.875,
                  r"$(\alpha, z, \sigma) \rightarrow "
                  r"(c\alpha,\, cz,\, c\sigma)$   leaves every wet/dry "
                  "outcome identical", fontsize=15, color=GREY)
    c = 1.0 + 0.5 * ease(min(u * 1.5, 1))
    for col, (cc_, lab) in enumerate((
            (1.0, r"$\alpha=1,\; z,\; \sigma$"),
            (c, rf"$\alpha={c:.2f},\; {c:.2f}z,\; {c:.2f}\sigma$"))):
        ax = fig.add_axes([0.07 + 0.48 * col, 0.30, 0.40, 0.45])
        i0 = slice(0, len(t_h) // 2)
        ax.plot(t_h[i0], cc_ * h_dense[i0], color=BLUE, lw=2)
        ax.axhline(cc_ * 0.4, color=GREEN, ls="--", lw=2)
        wet = (cc_ * h_dense[i0]) > (cc_ * 0.4)
        ax.scatter(t_h[i0][::6], np.full(wet[::6].shape, -2.6 * cc_),
                   c=np.where(wet[::6], BLUE, "#7a5230"), s=18, marker="s")
        ax.set_title(lab, fontsize=14, color=FG)
        ax.set_ylim(-3.0 * c, 3.0 * c)
        ax.set_xticks([]); ax.set_yticks([])
    title_ax.text(0.5, 0.16, "identical wet/dry sequences  →  the PHASE is "
                             "measurable, the GAIN is not:",
                  ha="center", fontsize=14, color=FG)
    title_ax.text(0.5, 0.10, "amplitude and datum must come from the ocean "
                             "model — declared, bounded by the model "
                             "ensemble", ha="center", fontsize=13,
                  color=YELLOW)


def s5(u):
    axL = fig.add_axes([0.07, 0.14, 0.40, 0.62])
    axR = fig.add_axes([0.54, 0.10, 0.42, 0.72])
    title(axL, "5 · The distributed tide gauge, applied",
          r"the operator:   $h(s,t) \,=\, h_B\!\left(t-\tau(s)\right)$"
          "     (only where the matched null cannot explain it)")
    v = ease(min(u * 1.4, 1))
    m = max(2, int(v * len(S_CENT)))
    axL.plot(S_CENT[:m], TAU_PROF[:m], "o--", color=GREY, lw=1.5,
             label="measured")
    axL.plot(S_CENT[:m], TAU_USED[:m], "o-", color=YELLOW, lw=2.5,
             label="applied (outside the null)")
    axL.axhline(0, color=GREY, lw=0.8)
    if v >= 1:
        axL.text(S_CENT[-1], TAU_USED[-1] - 4, "+26 min at the head",
                 color=YELLOW, ha="right", fontsize=13)
    axL.set_xlabel("distance from the mouth s (km)")
    axL.set_ylabel(r"$\tau(s)$ (min)")
    axL.set_ylim(-12, 32)
    axL.legend(fontsize=10, framealpha=0.1, loc="upper left")
    axR.imshow(DEM, cmap="viridis", alpha=min(1.0, u * 1.8),
               vmin=np.nanpercentile(DEM, 2), vmax=np.nanpercentile(DEM, 98))
    axR.set_xticks([]); axR.set_yticks([])
    axR.set_title("Villaviciosa — MAREA DEM (10 m)", fontsize=13, color=FG)
    if u > 0.75:
        fig.text(0.5, 0.035,
                 "MAREA: the archive measures its own tide — and the DEM "
                 "carries σ per pixel, hydraulics, and its demonstrated "
                 "limits", ha="center", fontsize=13, color=FG)


DRAW = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}


def frame(i):
    fig.clf()
    k = int(np.searchsorted(starts[1:], i, side="right"))
    name, secs = SCENES[k]
    u = (i - starts[k]) / (secs * FPS)
    DRAW[name](u)
    return []


anim = FuncAnimation(fig, frame, frames=TOTAL, blit=False)
out = os.path.join("research", "marea_method.mp4")
anim.save(out, writer=FFMpegWriter(fps=FPS, bitrate=2400))
print("written", out, flush=True)
