"""The likelihood of one pixel, drawn: the rule, the bill, and the clock.

Reproduces the 199-visit record of the walkthrough pixel (624, 295) and
draws three panels for docs/one_pixel_marea_example.md:

  A  the pixel's rule: P(wet | level) = Phi((h - z)/sigma) on the ocean
     clock, with every visit placed at its level and three visits named
     (cheap, ambiguous, contradiction);
  B  the cost of a visit as a function of u = (h - z)/sigma, for a wet and
     for a dry reading, with the same three visits;
  C  the bill: the 199 costs sorted, on the ocean clock and on the band
     clock (+18 min) — the dear tail is what the clock removes.

Run:  python -m experiments.onepixel_likelihood_figure
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PIXEL = (624, 295)
CUBE = "ndwi_cube_villaviciosa_grande_10y.nc"
TIDE_POINT = (43.540, -5.381)
TAU_BAND = 18.0
Z0, SG0 = -0.66, 0.65                # the walkthrough's profiled pair at tau=0
P_CLIP = 1e-6
OUT = "docs/figures/onepixel/likelihood_explained.png"
NAMED = {"2016-03-18": "cheap", "2016-10-07": "ambiguous",
         "2016-12-13": "contradiction"}


def main():
    import pandas as pd
    import rasterio
    from scipy.special import ndtr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pyintertidal as pit
    from pyintertidal import water as W
    from pyintertidal.cube import open_cube
    from pyintertidal.boundary import PyTMDBoundary
    pit.net.use_system_certificates()

    R, C = PIXEL
    # ── the pixel's raw record ──────────────────────────────────────────
    ds, arrays, t_dim = open_cube(CUBE, ("B03", "B08", "SCL"))
    try:
        b03 = np.asarray(arrays["B03"][:, R, C].values, float)
        b08 = np.asarray(arrays["B08"][:, R, C].values, float)
        scl = np.nan_to_num(np.asarray(arrays["SCL"][:, R, C].values, float),
                            nan=0).astype(int)
        dates = np.array([str(np.datetime64(v, "D")) for v in ds[t_dim].values])
    finally:
        ds.close()
    # ── the scene screen, exactly as the walkthrough builds it ────────────
    from pyintertidal import SentinelCube
    aoi = pit.sites.get("villaviciosa")
    cube = SentinelCube(aoi, ("2016-01-01", "2025-12-31"), water="ndwi",
                        cache_path=CUBE, resolution=10)
    _, cloud_pct = pit.reference_and_clouds(
        cube, threshold=0.0, bad_classes=W.BAD_CLASSES, clean_scene_max_bad=0.05,
        stable_threshold=0.95, transition_buffer_px=10)
    usable_dates = set(pit.filter_dates(cloud_pct, cloud_threshold=0.10))
    usable = np.array([d in usable_dates for d in dates])
    T = len(dates)
    print(f"usable scenes: {int(usable.sum())} of {T}")

    ndwi = (b03 - b08) / np.where(b03 + b08 != 0, b03 + b08, np.nan)
    clear_px = np.isin(scl, list(W.CLEAR_CLASSES)) & np.isfinite(ndwi)
    kept = usable & clear_px
    print(f"pixel observations after both screens: {int(kept.sum())}")

    # ── the tide at each visit, on two clocks ────────────────────────────
    times = pit.overpass.get_overpass_times(aoi.bbox, ("2016-01-01", "2025-12-31"),
                                            verbose=False)
    have = kept & np.array([d in times for d in dates])
    d_o = dates[have]
    t_real = pd.DatetimeIndex(pd.to_datetime([times[d] for d in d_o])).tz_localize(None)
    eot = PyTMDBoundary("EOT20", TIDE_POINT[0], TIDE_POINT[1], directory="tide_models")
    h0 = np.asarray(eot.levels(t_real), float)
    hb = np.asarray(eot.levels(t_real - pd.Timedelta(minutes=TAU_BAND)), float)
    w = (ndwi[have] > 0.0)
    ok = np.isfinite(h0) & np.isfinite(hb)
    d_o, h0, hb, w = d_o[ok], h0[ok], hb[ok], w[ok]
    print(f"{len(d_o)} observations with a level")

    def terms(h, z, sg):
        P = np.clip(ndtr((h - z) / sg), P_CLIP, 1 - P_CLIP)
        return np.where(w, np.log(P), np.log1p(-P)), P

    # profile on the walk's grids so the numbers match the notebook
    Z = np.linspace(h0.min(), h0.max(), 120)
    S = np.array(list(pit.marea.SG_GRID) + [0.85, 1.10])
    def profile(h):
        best = (-np.inf, np.nan, np.nan)
        for zc in Z:
            for sg in S:
                ll = float(terms(h, zc, sg)[0].sum())
                if ll > best[0]:
                    best = (ll, float(zc), float(sg))
        return best
    ll0, z0, sg0 = profile(h0)
    llb, zb, sgb = profile(hb)
    print(f"tau=0: z={z0:+.2f} sigma={sg0:.2f} ell={ll0:.2f} | "
          f"tau=+{TAU_BAND:.0f}: z={zb:+.2f} sigma={sgb:.2f} ell={llb:.2f}")
    l0, P0 = terms(h0, z0, sg0)
    lb, Pb = terms(hb, zb, sgb)
    u0 = (h0 - z0) / sg0

    # ── the figure ───────────────────────────────────────────────────────
    TIDE = "#2a6f97"; MUD = "#b08968"; RED = "#c1121f"; INK = "#14313c"
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4), dpi=130)

    # A · the rule
    ax = axs[0]
    hg = np.linspace(h0.min() - 0.2, h0.max() + 0.2, 400)
    ax.plot(hg, ndtr((hg - z0) / sg0), color=INK, lw=2.2,
            label=r"$P(\mathrm{wet}\mid h)=\Phi\left(\frac{h-z}{\sigma}\right)$")
    rng = np.random.default_rng(3)
    ax.scatter(h0[w], 1.0 + rng.uniform(0.01, 0.06, w.sum()), s=12, c=TIDE, label="read wet")
    ax.scatter(h0[~w], 0.0 - rng.uniform(0.01, 0.06, (~w).sum()), s=12, c=MUD, label="read dry")
    ax.axvline(z0, color="k", lw=0.8, ls=":")
    ax.text(z0, 0.5, f" z = {z0:+.2f} m", fontsize=9, va="center")
    ax.annotate("", xy=(z0 + sg0, 0.84), xytext=(z0, 0.84),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
    ax.text(z0 + sg0 / 2, 0.87, f"σ = {sg0:.2f} m", ha="center", fontsize=9)
    lines = []
    for n_, (d, name) in enumerate(NAMED.items(), 1):
        k = np.flatnonzero(d_o == d)
        if not len(k):
            continue
        k = k[0]
        ax.plot([h0[k], h0[k]], [0 if not w[k] else 1, P0[k]], color=RED, lw=1.1, ls="--")
        ax.plot(h0[k], P0[k], "o", ms=9, mfc="white", mec=RED, mew=1.8)
        ax.text(h0[k], P0[k], str(n_), color=RED, fontsize=8, ha="center", va="center", fontweight="bold")
        lines.append(f"{n_}  {d}  {name}: read {'wet' if w[k] else 'dry'}, "
                     f"P = {P0[k]:.2f}, ℓ = {l0[k]:+.2f}")
    ax.text(0.98, 0.30, "\n".join(lines), transform=ax.transAxes, ha="right", va="center",
            fontsize=8.5, color=RED, bbox=dict(boxstyle="round", fc="white", ec=RED, alpha=0.9))
    ax.set_xlabel("water level at the visit, ocean clock  h (m)")
    ax.set_ylabel("probability the pixel reads wet")
    ax.set_ylim(-0.12, 1.12)
    ax.set_title(f"A · the pixel's rule, and its {len(d_o)} visits on it", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.2)

    # B · the cost
    ax = axs[1]
    ug = np.linspace(-3.5, 3.5, 500)
    Pg = np.clip(ndtr(ug), P_CLIP, 1 - P_CLIP)
    ax.plot(ug, -np.log(Pg), color=TIDE, lw=2.2, label=r"cost if read wet: $-\ln\Phi(u)$")
    ax.plot(ug, -np.log1p(-Pg), color=MUD, lw=2.2, label=r"cost if read dry: $-\ln(1-\Phi(u))$")
    ax.scatter(u0[w], -l0[w], s=10, c=TIDE, alpha=0.5)
    ax.scatter(u0[~w], -l0[~w], s=10, c=MUD, alpha=0.5)
    lines = []
    for n_, (d, name) in enumerate(NAMED.items(), 1):
        k = np.flatnonzero(d_o == d)
        if not len(k):
            continue
        k = k[0]
        ax.plot(u0[k], -l0[k], "o", ms=9, mfc="white", mec=RED, mew=1.8)
        ax.text(u0[k], -l0[k], str(n_), color=RED, fontsize=8, ha="center", va="center", fontweight="bold")
        lines.append(f"{n_}  {name}: u = {u0[k]:+.2f} → cost {-l0[k]:.2f}")
    ax.text(0.98, 0.62, "\n".join(lines), transform=ax.transAxes, ha="right", va="center",
            fontsize=8.5, color=RED, bbox=dict(boxstyle="round", fc="white", ec=RED, alpha=0.9))
    ax.set_xlabel(r"$u = (h - z)/\sigma$   (water above the pixel, in reliefs)")
    ax.set_ylabel(r"cost of the visit  $-\ell_t$")
    ax.set_ylim(0, 7)
    ax.set_title("B · what a visit costs: sure and right ≈ 0, sure and wrong ≫ 1",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.2)

    # C · the bill under two clocks
    ax = axs[2]
    c0 = np.sort(-l0)[::-1]; cb = np.sort(-lb)[::-1]
    x = np.arange(1, len(c0) + 1)
    ax.bar(x, c0, width=1.0, color="0.75", label=f"ocean clock τ = 0:  Σ = {-l0.sum():.1f}")
    ax.step(x, cb, where="mid", color=RED, lw=1.6,
            label=f"band clock τ = +{TAU_BAND:.0f} min:  Σ = {-lb.sum():.1f}")
    ax.set_xlabel(f"the {len(d_o)} visits, sorted from dearest to cheapest")
    ax.set_ylabel(r"cost of the visit  $-\ell_t$")
    ax.set_xlim(0, 60)
    ax.set_title("C · the bill: the dear tail is what the clock changes", loc="left", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    ax.text(0.98, 0.55, f"{int((c0 > 1).sum())} visits cost > 1 on the ocean clock\n"
            f"{int((cb > 1).sum())} on the band clock",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
