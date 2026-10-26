# -*- coding: utf-8 -*-
"""The method in one figure, for the paper.

(a) One intertidal pixel of the Villaviciosa record: its NDWI at every
    clear visit against the water level at the mouth at that instant, the
    fitted transition (``marea.invert_series``: elevation z at the
    midpoint, width sigma), and the wet/dry reading each visit gives.
(b) The bands of one site: the intertidal record coloured by its band of
    geodesic distance from the mouth (fixed 3-km bands, small ones merged).
(c) The interior clock of that site: the lag fitted per band with its
    95 % scene-bootstrap interval, and the smoothed profile every pixel
    received (Gaussian kernel of the band length, anchored at 0).

Reads only files the pipeline notebooks wrote. Writes
docs/paper/figures/fig_method.png.
Run from the repository root:  python -m experiments.paper_fig_method [site]
(site: ferrol, escalda, ems or wadden; default ems).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")

INK = "#14313c"; TIDE = "#2a6f97"; RED = "#c1121f"; MUD = "#b08968"
SITES = {
    "ferrol": dict(name="Ría de Ferrol", marea="products_ferrol/marea_eot20_g1/marea.npz",
                   extract="products_ferrol/marea_extract.npz", res_m=10),
    "escalda": dict(name="Westerschelde", marea="products_escalda/marea_eot20_g2/marea.npz",
                    extract="products_escalda/marea_extract.npz", res_m=20),
    "ems": dict(name="Ems-Dollard", marea="products_ems/marea_eot20_g1/marea.npz",
                extract="products_ems/marea_extract.npz", res_m=20),
    "wadden": dict(name="Wadden Sea (Vlie)", marea="products_wadden/marea_eot20_g1/marea.npz",
                   extract="products_wadden/marea_extract.npz", res_m=20),
}


def one_pixel():
    """NDWI visits of one Villaviciosa pixel against the mouth level, and
    the transition fitted to them."""
    import pyintertidal as pit
    from pyintertidal import marea
    from pyintertidal.boundary import PyTMDBoundary
    pit.net.use_system_certificates()
    d = np.load("marea_demo_extract.npz", allow_pickle=True)
    Y, C, keep, dates, bbox = d["Y"], d["C"].astype(bool), d["keep"], np.array([str(x) for x in d["dates"]]), d["bbox"]
    bb = {"west": float(bbox[0]), "south": float(bbox[1]), "east": float(bbox[2]), "north": float(bbox[3]),
          "crs": "EPSG:4326"}
    cache = "products_marea/overpass_times.json"
    times = json.load(open(cache)) if os.path.exists(cache) else \
        pit.overpass.get_overpass_times(bb, ("2016-01-01", "2025-12-31"), verbose=False)
    has = np.array([x in times for x in dates])
    t = pd.DatetimeIndex(pd.to_datetime([times[x] for x in dates[has]])).tz_localize(None)
    eot = PyTMDBoundary("EOT20", 0.5 * (bb["south"] + bb["north"]), 0.5 * (bb["west"] + bb["east"]),
                        directory="tide_models")
    h = np.asarray(eot.levels(t), float)
    Yh, Ch = Y[has], C[has]
    # a pixel with many clear visits and a transition well inside the range,
    # chosen among a fixed random subset of the record (the full inversion
    # is not needed for one illustrative pixel)
    rng = np.random.default_rng(0)
    cand = np.sort(rng.choice(Yh.shape[1], size=min(2000, Yh.shape[1]), replace=False))
    n_clear = Ch[:, cand].sum(0)
    z_all, s_all = marea.invert_series(np.nan_to_num(Yh[:, cand], nan=0.0).astype(np.float64),
                                       Ch[:, cand].astype(np.float64), h)
    ok = np.isfinite(z_all) & (n_clear >= 150) & (z_all > np.nanpercentile(h, 25)) & (z_all < np.nanpercentile(h, 75))
    pick = np.flatnonzero(ok)
    jj = int(pick[len(pick) // 2])
    j = int(cand[jj])
    m = Ch[:, j] & np.isfinite(h)
    y, hh = Yh[m, j].astype(float), h[m]
    z, s = float(z_all[jj]), float(s_all[jj])
    # the transition's dry and wet levels (a, b) given z and sigma: NDWI ≈ a + b Φ((h − z)/σ)
    phi = norm.cdf((hh - z) / s)
    A = np.column_stack([np.ones_like(phi), phi])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return dict(h=hh, ndwi=y, z=z, sigma=s, a=a, b=b, n=int(m.sum()), pixel=int(keep[j]))


def main(site="ems"):
    from pyintertidal import tide_estimators as te
    cfg = SITES[site]
    m = np.load(cfg["marea"])
    e = np.load(cfg["extract"], allow_pickle=True)
    H, W = (int(v) for v in e["shape"])
    keep, band = m["keep"], m["band"].astype(int)
    s_km = m["s_km"].astype(float)
    tau_used, tau_lo, tau_hi = (np.asarray(m[k], float) for k in ("tau_usado_min", "tau_lo_min", "tau_hi_min"))
    band_km = float(m["band_km"][0])
    nb = len(tau_used)
    centres = np.array([float(np.nanmedian(s_km[band == k])) for k in range(nb)])

    fig = plt.figure(figsize=(7.2, 4.8), dpi=200)      # page budget
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.15])

    # (a) one pixel
    px = one_pixel()
    ax = fig.add_subplot(gs[0, 0])
    wet = px["ndwi"] > (px["a"] + 0.5 * px["b"])
    ax.plot(px["h"][~wet], px["ndwi"][~wet], "o", ms=2.6, color=MUD, alpha=0.8, label="visit read dry")
    ax.plot(px["h"][wet], px["ndwi"][wet], "o", ms=2.6, color=TIDE, alpha=0.8, label="visit read wet")
    hx = np.linspace(px["h"].min() - 0.1, px["h"].max() + 0.1, 300)
    ax.plot(hx, px["a"] + px["b"] * norm.cdf((hx - px["z"]) / px["sigma"]), "-", color=INK, lw=1.4,
            label="fitted transition")
    ax.axvline(px["z"], color=RED, lw=1.0, ls="--")
    ax.text(0.97, 0.04, f"z = {px['z']:+.2f} m\nσ = {px['sigma']:.2f} m", transform=ax.transAxes,
            color=RED, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    ax.set_xlabel("water level at the mouth at the visit (m)", fontsize=8)
    ax.set_ylabel("NDWI of the pixel", fontsize=8)
    ax.set_title(f"(a) one pixel, {px['n']} clear visits", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

    # (b) the bands
    ax = fig.add_subplot(gs[:, 1])
    img = np.full(H * W, np.nan); img[keep] = np.where(band >= 0, band, np.nan)   # no band: no water path
    img = img.reshape(H, W)
    rows = np.flatnonzero(np.isfinite(img).any(1)); cols = np.flatnonzero(np.isfinite(img).any(0))
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    sea = e["sea"].astype(bool)
    bg = np.where(sea, 0.82, 0.95)
    ax.imshow(bg[r0:r1 + 1, c0:c1 + 1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    cm = plt.get_cmap("viridis", nb)
    im = ax.imshow(np.ma.masked_invalid(img[r0:r1 + 1, c0:c1 + 1]), cmap=cm, vmin=-0.5, vmax=nb - 0.5,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=range(nb))
    cb.set_label("band of distance from the mouth", fontsize=7); cb.ax.tick_params(labelsize=6)
    km = 5000 / cfg["res_m"]
    ax.plot([c1 - c0 - km - 10, c1 - c0 - 10], [r1 - r0 - 15, r1 - r0 - 15], "-", color=INK, lw=2)
    ax.text(c1 - c0 - km / 2 - 10, r1 - r0 - 22, "5 km", ha="center", fontsize=7, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(b) {cfg['name']}: {nb} bands of {band_km:.0f} km", loc="left", fontsize=9)

    # (c) the clock
    ax = fig.add_subplot(gs[1, 0])
    sg = np.linspace(0, float(np.nanmax(s_km)), 200)
    curve = te.smooth_lag_profile(centres, tau_used, sg, bandwidth_km=band_km)
    ax.errorbar(centres, tau_used, yerr=[np.clip(tau_used - tau_lo, 0, None), np.clip(tau_hi - tau_used, 0, None)],
                fmt="o", ms=4, color=TIDE, capsize=3, lw=1, label="lag per band, 95 % interval")
    ax.plot(sg, curve, "-", color=RED, lw=1.6, label="lag applied per pixel")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("distance from the mouth s (km)", fontsize=8)
    ax.set_ylabel("lag τ (min)", fontsize=8)
    ax.set_title("(c) the interior clock, measured from the imagery", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

    fig.tight_layout()
    os.makedirs("docs/paper/figures", exist_ok=True)
    fig.savefig("docs/paper/figures/fig_method.png", dpi=200)
    print(f"-> docs/paper/figures/fig_method.png  (pixel {px['pixel']}: z {px['z']:+.2f} m, sigma {px['sigma']:.2f} m; "
          f"{cfg['name']}: {nb} bands, lags {np.round(tau_used, 0).tolist()} min)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ems")
