# -*- coding: utf-8 -*-
"""What the clock does to the likelihood, for the paper's methods section.

One band of the Ems-Dollard record (by default the band that contains the
Delfzijl gauge), exactly as the product fits it: the same wet/dry record,
the same boundary (consensus of EOT20 and the Borkum gauge), the same
profiled Bernoulli likelihood with the same elevation grid and width
atoms (configs/m2.yaml).

(a) One pixel of the band on the ocean clock (tau = 0): its wet and dry
    readings against the water level at the mouth at each visit, and the
    transition P(wet | h) = Phi((h - z)/sigma) that best explains them.
    Wet and dry readings overlap over a wide range of levels.
(b) The same pixel on the band's clock (tau = tau_k): the readings sort
    themselves and the transition sharpens; the pixel's cost (its
    negative log-likelihood) drops.
(c) The band's profiled negative log-likelihood against the lag, relative
    to its minimum, with the fitted lag, its 95 % bootstrap interval and
    the ocean clock at 0.

Reads only files the pipeline notebook wrote. Writes
docs/paper/figures/fig_likelihood.png.
Run from the repository root:  python -m experiments.paper_fig_likelihood [band]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr

matplotlib.use("Agg")

INK = "#14313c"; TIDE = "#2a6f97"; RED = "#c1121f"; MUD = "#b08968"
SITE = "ems"
EXTRACT = "products_ems/marea_extract.npz"
MAREA = "products_ems/marea_eot20_g1/marea.npz"
OVERPASS = "products_ems/overpass_times.json"
PERIOD = ("2023-01-01", "2025-12-31")
JUDGE = ["delf"]
N_GAUGES = 1
TAU_GRID = np.arange(0.0, 181.0, 5.0)


def load_band(k):
    """Wet/dry record of band k (the product's sample size), the visit
    instants and the boundary."""
    import pyintertidal as pit
    from pyintertidal.boundary import make_boundary
    pit.net.use_system_certificates()
    cfg = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
    m = np.load(MAREA)
    band = m["band"].astype(int)
    tau_used = np.asarray(m["tau_usado_min"], float)
    tau_lo, tau_hi = np.asarray(m["tau_lo_min"], float), np.asarray(m["tau_hi_min"], float)
    cols = np.flatnonzero(band == k)
    rng = np.random.default_rng(cfg["seed"] + 31)
    if len(cols) > cfg["max_px_banda"]["m2a"]:
        cols = np.sort(rng.choice(cols, cfg["max_px_banda"]["m2a"], replace=False))
    e = np.load(EXTRACT, allow_pickle=True)
    dates = np.array([str(d) for d in e["dates"]])
    times = json.load(open(OVERPASS))
    have = np.array([d in times for d in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime([times[d] for d in dates[have]])).tz_localize(None)
    Y = np.nan_to_num(e["Y"][have][:, cols], nan=0.0).astype(np.float64)
    C = e["C"][have][:, cols].astype(np.float64)
    wet, clear = (C > 0) & (Y > 0), C > 0
    aoi = pit.sites.get(SITE)
    boundary = make_boundary(aoi, tide_model="EOT20", n_gauges=N_GAUGES, period=PERIOD, exclude=JUDGE)
    print(f"band {k}: {len(cols)} px in the likelihood, {len(t_real)} visits; boundary {boundary.name}; "
          f"lag {tau_used[k]:.1f} min [{tau_lo[k]:.1f}, {tau_hi[k]:.1f}]")
    return dict(wet=wet, clear=clear, ndwi=Y, t=t_real, boundary=boundary, cfg=cfg,
                tau=float(tau_used[k]), lo=float(tau_lo[k]), hi=float(tau_hi[k]), n_px=len(cols))


def levels(boundary, t, tau_min):
    return np.asarray(boundary.levels(t - pd.Timedelta(minutes=float(tau_min))), float)


def pixel_fit(w, c, h, z_grid, sg_grid):
    """Profiled (z, sigma) of one pixel and its total cost (nats)."""
    best = (np.inf, None, None)
    for s0 in sg_grid:
        p = np.clip(ndtr((h[:, None] - z_grid[None, :]) / s0), 1e-4, 1 - 1e-4)
        ll = (w & c).astype(float) @ np.log(p) + (~w & c).astype(float) @ np.log1p(-p)
        j = int(np.argmax(ll))
        if -ll[j] < best[0]:
            best = (-ll[j], float(z_grid[j]), float(s0))
    return best


def main(k=8):
    from pyintertidal import tide_estimators as te
    d = load_band(k)
    cfg, wet, clear, t, bnd = d["cfg"], d["wet"], d["clear"], d["t"], d["boundary"]
    h0 = levels(bnd, t, 0.0)
    ok = np.isfinite(h0)
    wet, clear, t, h0 = wet[ok], clear[ok], t[ok], h0[ok]
    ndwi = d["ndwi"][ok]
    z_grid = np.linspace(h0.min() - 0.3, h0.max() + 0.3, cfg["z_puntos"])
    sg_grid = tuple(cfg["sigma_perfil_m"])
    # (c) the band's profiled NLL against the lag
    nll = np.array([te._band_nll(wet, clear, levels(bnd, t, tv), z_grid, cfg["sigma0_m"], sg_grid)[0]
                    for tv in TAU_GRID])
    # (a), (b) one pixel: many clear visits, transition inside the range
    n_clear = clear.sum(0)
    z0 = np.array([pixel_fit(wet[:, j], clear[:, j], h0, z_grid, sg_grid)[1] for j in range(wet.shape[1])])
    cand = np.flatnonzero((n_clear >= 0.8 * n_clear.max()) & (z0 > np.percentile(h0, 30)) & (z0 < np.percentile(h0, 70)))
    hk = levels(bnd, t, d["tau"])
    gains = []
    for j in cand:
        c0 = pixel_fit(wet[:, j], clear[:, j], h0, z_grid, sg_grid)[0]
        ck = pixel_fit(wet[:, j], clear[:, j], hk, z_grid, sg_grid)[0]
        gains.append(c0 - ck)
    j = int(cand[int(np.argsort(gains)[len(gains) // 2])])      # a typical pixel, not the best case
    print(f"pixel {j}: {int(n_clear[j])} clear visits; median gain over candidates {np.median(gains):.1f} nats")

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), dpi=200)
    for ax, (tau_v, h, title) in zip(axs[:2], [(0.0, h0, "(a) ocean clock, τ = 0"),
                                                (d["tau"], hk, f"(b) the band's clock, τ = {d['tau']:+.0f} min")]):
        cost, z, s = pixel_fit(wet[:, j], clear[:, j], h, z_grid, sg_grid)
        m = clear[:, j]
        w = wet[:, j]
        y = ndwi[:, j]
        # the profiled (z, sigma) of the binary likelihood, drawn on the NDWI
        # scale: a + b Phi((h - z)/sigma) with (a, b) by least squares
        phi = ndtr((h[m] - z) / s)
        a_, b_ = np.linalg.lstsq(np.column_stack([np.ones_like(phi), phi]), y[m], rcond=None)[0]
        ax.plot(h[m][~w[m]], y[m][~w[m]], "o", ms=2.6, color=MUD, alpha=0.75, label="read dry (NDWI ≤ 0)")
        ax.plot(h[m][w[m]], y[m][w[m]], "o", ms=2.6, color=TIDE, alpha=0.75, label="read wet (NDWI > 0)")
        hx = np.linspace(h.min(), h.max(), 300)
        ax.plot(hx, a_ + b_ * ndtr((hx - z) / s), "-", color=INK, lw=1.3, label="fitted transition")
        ax.axhline(0, color="0.5", lw=0.7, ls=":")
        ax.axvline(z, color=RED, lw=0.9, ls="--")
        ax.text(0.03, 0.62, f"z = {z:+.2f} m\nσ = {s:.2f} m\ncost {cost:.1f}", transform=ax.transAxes,
                fontsize=6.5, va="center", bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel("water level at the visit (m)", fontsize=7)
        ax.set_ylabel("NDWI of the pixel", fontsize=7)
        ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
    axs[0].legend(fontsize=5.5, loc="center left", bbox_to_anchor=(0.02, 0.33))
    ax = axs[2]
    rel = (nll - nll.min()) * 1e3
    ax.plot(TAU_GRID, rel, "-", color=INK, lw=1.3)
    ax.axvspan(d["lo"], d["hi"], color=RED, alpha=0.12, lw=0, label="95 % bootstrap interval")
    ax.axvline(d["tau"], color=RED, lw=1.2, label=f"fitted lag, {d['tau']:+.0f} min")
    ax.axvline(0, color="0.45", lw=0.8, label="ocean clock")
    ax.set_title("(c) the band's likelihood", loc="left", fontsize=8)
    ax.set_xlabel("lag τ applied to the boundary (min)", fontsize=7)
    ax.set_ylabel("mean NLL above the minimum (×10⁻³)", fontsize=7)
    ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.08, 1.0)); ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
    fig.tight_layout()
    os.makedirs("docs/paper/figures", exist_ok=True)
    fig.savefig("docs/paper/figures/fig_likelihood.png", dpi=200)
    print(f"-> docs/paper/figures/fig_likelihood.png  (NLL minimum at τ = {TAU_GRID[int(np.argmin(nll))]:.0f} min on the 5-min grid)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
