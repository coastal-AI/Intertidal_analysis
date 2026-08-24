"""Profiles across the flats: HSR, the DEA step method and the real survey.

A scatter says how far apart two rasters are on average; a profile says
WHERE and HOW they differ — whether a method rounds off the channel edges,
flattens the high flat, or drifts along the line. That is the figure a
reader trusts.

The transect is chosen by search rather than by eye: the line that crosses
the most flat with all three sources present and the largest drop, so the
comparison is made where there is actually relief to get right.
"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

import pyintertidal as pit
from pyintertidal import viz

SITES = [
    ("Tajo", "products_tejo", "tejo", "batimetría EMODnet 2020 · 22.6 m"),
    ("Wadden danés", "products_vadehavet", "vadehavet",
     "batimetría EMODnet 2020 · 16.5 m"),
]


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


for name, out, key, ref_label in SITES:
    hsr, TR, CRS, SH = load(f"{out}/hsr_elevation.tif")
    step = load(f"{out}/step_elevation.tif")[0]
    inter = load(f"{out}/intertidal_mask.tif")[0] > 0
    bathy = pit.reproject_to_grid(f"{out}/bathy_native.tif"
                                  if os.path.exists(f"{out}/bathy_native.tif")
                                  else f"{out}/bathy_tagus_native.tif",
                                  TR, CRS, SH)

    # Put everything on the reference's datum so the profiles overlay.
    m_all = inter & np.isfinite(bathy)
    for arr in (hsr, step):
        mm = m_all & np.isfinite(arr)
        arr -= np.median(arr[mm] - bathy[mm])

    ok = m_all & np.isfinite(hsr) & np.isfinite(step)
    print(f"\n{name}: {ok.sum():,} px con las tres fuentes")

    ys, xs = np.where(ok)
    tf = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    rng = np.random.default_rng(11)
    idx = rng.choice(len(ys), size=min(6000, len(ys)), replace=False)
    best = None
    for _ in range(30000):
        i, j = rng.choice(idx, 2)
        r0, c0, r1, c1 = ys[i], xs[i], ys[j], xs[j]
        L = np.hypot(r1 - r0, c1 - c0) * 10
        if not (600 < L < 2500):
            continue
        n = int(L // 10)
        rr = np.linspace(r0, r1, n).astype(int)
        cc = np.linspace(c0, c1, n).astype(int)
        frac = ok[rr, cc].mean()
        if frac < 0.9:
            continue
        span = np.nanmax(bathy[rr, cc]) - np.nanmin(bathy[rr, cc])
        score = span * frac * min(L / 1000, 2)
        if best is None or score > best[0]:
            best = (score, r0, c0, r1, c1, span, L, frac)

    _, r0, c0, r1, c1, span, L, frac = best
    A = tf.transform(TR.c + c0 * TR.a, TR.f + r0 * TR.e)
    B = tf.transform(TR.c + c1 * TR.a, TR.f + r1 * TR.e)
    print(f"  traza {L:.0f} m, {frac*100:.0f} % con datos, "
          f"desnivel de la referencia {span:.2f} m")
    print(f"  ({A[0]:.4f}, {A[1]:.4f}) → ({B[0]:.4f}, {B[1]:.4f})")

    profiles = {}
    for label, raster in [("HSR", hsr), ("escalón (DEA)", step),
                          (ref_label, bathy)]:
        d, v = pit.transect_profile(raster, TR, CRS, A, B, n=300)
        profiles[label] = v

    fig, ax = viz.plot_transect(
        d, profiles, title=f"{name} · perfil sobre la llanura",
        ylabel="cota sobre el datum de la referencia (m)",
        start=A, end=B, basemap_of=np.where(ok, bathy, np.nan),
        transform=TR, crs=CRS, figsize=(15, 4.8))
    # colour the reference distinctly: it is the truth, not a third opinion
    for line in ax.get_lines():
        if line.get_label().startswith("batimetría"):
            line.set(color="#1a1a1a", lw=2.6, zorder=5)
    ax.legend(fontsize=9)
    fn = f"figuras_pablo/23_transecto_{key}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close(fig)

    r = {k: v for k, v in profiles.items()}
    ref = r[ref_label]
    good = np.isfinite(ref)
    for k in ("HSR", "escalón (DEA)"):
        s = np.isfinite(r[k]) & good
        print(f"    {k:14s} {s.sum():3d}/300 pts, "
              f"RMSE en la traza {np.sqrt(np.mean((r[k][s]-ref[s])**2)):.3f} m")
    print(f"  -> {fn}")
