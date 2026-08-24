"""Does the ria damp big tides more than small ones? That IS measurable.

The sigmoid fit cannot see a constant amplification: scaling the tide axis is
an affine map and the residual does not move. But amplitude-DEPENDENT damping
is not affine, and it is what a friction-dominated estuary actually does —
large tides feel more bottom friction per cycle than small ones, so the
amplification factor differs between springs and neaps.

The test mirrors the flood/ebb one that gave us the phase lag. Fit the spring
scenes and the neap scenes separately. Under a constant alpha both fits
return the same elevation, because each is internally consistent. If alpha
depends on range, the two disagree — and the disagreement, unlike alpha
itself, is something the imagery can measure.

Sign to expect if friction dominates: springs are damped more, so during
springs the real water level is lower than the model near high water. A pixel
then floods later than predicted and the spring fit places it HIGHER.
The effect should grow upstream, where the accumulated friction is greater.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
N_BANDS = 5
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    SH = tuple(d["shape"])

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        transform = s.transform
    rows = keep // SH[1]
    north = transform.f + (rows + 0.5) * transform.e
    dist = (north.max() - north) / 1000.0

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])

    # The local tidal RANGE around each overpass: model a dense day either
    # side and take peak-to-peak. That is what separates springs from neaps,
    # and it is a property of the date, not of the scene's own height.
    grid_t = pd.DatetimeIndex(np.concatenate(
        [(base + pd.Timedelta(hours=h)).values
         for h in np.arange(-18, 18.5, 0.5)])).unique().sort_values()
    print(f"modelando {len(grid_t):,} instantes...", flush=True)
    df = model_tides(x=[lon_c], y=[lat_c], time=grid_t, model="EOT20",
                     directory="tide_models", crs="EPSG:4326",
                     extrapolate=True, cutoff=np.inf,
                     parallel=False).reset_index()
    lut = pd.Series(df["tide_height"].to_numpy(float),
                    index=pd.DatetimeIndex(df["time"])).groupby(level=0).first()

    H = np.vstack([lut.reindex(base + pd.Timedelta(hours=float(h))
                               ).to_numpy(float)
                   for h in np.arange(-18, 18.5, 0.5)])
    rng_tide = np.nanmax(H, axis=0) - np.nanmin(H, axis=0)
    tide = lut.reindex(base).to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    recent = yrs >= 2023
    med = np.nanmedian(rng_tide[recent])
    spring = recent & (rng_tide >= med)
    neap = recent & (rng_tide < med)
    print(f"\nrango de marea diario: mediana {med:.2f} m")
    print(f"  vivas  {spring.sum():3d} escenas, rango medio "
          f"{np.nanmean(rng_tide[spring]):.2f} m")
    print(f"  muertas {neap.sum():3d} escenas, rango medio "
          f"{np.nanmean(rng_tide[neap]):.2f} m")
    for lab, m in [("vivas", spring), ("muertas", neap)]:
        print(f"  altura muestreada en {lab:8s} "
              f"{np.nanmin(tide[m]):+.2f}..{np.nanmax(tide[m]):+.2f} "
              f"(std {np.nanstd(tide[m]):.2f})")
    print(flush=True)

    # Springs reach lower water than neaps and neaps reach higher, so each
    # subset would get a mu grid shifted the other way and the comparison
    # would measure the grids, not the estuary. Clip both to the window they
    # share and give them one identical grid.
    LO = max(np.nanmin(tide[spring]), np.nanmin(tide[neap]))
    HI = min(np.nanmax(tide[spring]), np.nanmax(tide[neap]))
    GRID = np.linspace(LO, HI, MU_POINTS)
    print(f"ventana comun de marea: {LO:+.2f}..{HI:+.2f} m")
    print(f"  vivas   {int((spring & (tide>=LO) & (tide<=HI)).sum())} escenas dentro")
    print(f"  muertas {int((neap & (tide>=LO) & (tide<=HI)).sum())} escenas "
          f"dentro\n", flush=True)

    # Even inside the shared window the two subsets sit differently: springs
    # cluster at the extremes, neaps in the middle. Match their tide-height
    # histograms bin by bin so the only thing left that differs is the range
    # of the tide that produced each scene.
    BINS = np.linspace(LO, HI, 11)
    who = np.digitize(tide, BINS)
    match_s = np.zeros_like(spring)
    match_n = np.zeros_like(neap)
    _rng = np.random.default_rng(99)
    for k in range(1, len(BINS) + 1):
        si = np.flatnonzero(spring & (who == k) & (tide >= LO) & (tide <= HI))
        ni = np.flatnonzero(neap & (who == k) & (tide >= LO) & (tide <= HI))
        take = min(len(si), len(ni))
        if take == 0:
            continue
        match_s[_rng.choice(si, take, replace=False)] = True
        match_n[_rng.choice(ni, take, replace=False)] = True
    print(f"tras igualar histogramas: {match_s.sum()} vivas, "
          f"{match_n.sum()} muertas")
    print(f"  altura media  vivas {np.nanmean(tide[match_s]):+.3f}  "
          f"muertas {np.nanmean(tide[match_n]):+.3f}")
    print(f"  rango de marea del dia  vivas "
          f"{np.nanmean(rng_tide[match_s]):.2f} m  muertas "
          f"{np.nanmean(rng_tide[match_n]):.2f} m\n", flush=True)
    spring, neap = match_s, match_n

    def fit(mask, cols):
        m = mask & (tide >= LO) & (tide <= HI) & np.isfinite(tide)
        Yv = np.nan_to_num(Y[m][:, cols], nan=0.0).astype(np.float64)
        Cv = C[m][:, cols].astype(np.float64)
        a, b, mu, sg, rmse, N = _fit_block(Yv, Cv, tide[m], GRID, SG_GRID)
        # Only pixels whose transition sits well inside the window: at the
        # edges mu is pinned by the grid rather than measured.
        pad = 0.1 * (HI - LO)
        ok = ((N >= 6) & (b > 0.15) & (mu > LO + pad) & (mu < HI - pad))
        return np.where(ok, mu, np.nan)

    edges = np.quantile(dist, np.linspace(0, 1, N_BANDS + 1))
    print("DIFERENCIA VIVAS - MUERTAS POR BANDA")
    print(f"{'banda (km)':>14s} {'n px':>7s} {'mediana':>9s} {'IQR':>9s}")
    print("-" * 44)
    t0 = time.time()
    res = []
    for i in range(N_BANDS):
        cols = np.flatnonzero((dist >= edges[i]) & (dist < edges[i + 1] + 1e-9))
        if cols.size > 8000:
            cols = np.sort(np.random.default_rng(i).choice(cols, 8000, False))
        ms, mn = fit(spring, cols), fit(neap, cols)
        both = np.isfinite(ms) & np.isfinite(mn)
        if both.sum() < 50:
            print(f"{edges[i]:6.2f}-{edges[i+1]:5.2f} {cols.size:7,d}   "
                  f"pocos px validos")
            continue
        dd = ms[both] - mn[both]
        q1, q3 = np.percentile(dd, [25, 75])
        res.append(((edges[i] + edges[i + 1]) / 2, float(np.median(dd)),
                    int(both.sum())))
        print(f"{edges[i]:6.2f}-{edges[i+1]:5.2f} {int(both.sum()):7,d} "
              f"{np.median(dd):+9.3f} {q3 - q1:9.3f}", flush=True)

    print(f"\n({(time.time()-t0)/60:.1f} min)")
    if len(res) >= 3:
        xs = np.array([r[0] for r in res])
        ys = np.array([r[1] for r in res])
        sl, ic = np.polyfit(xs, ys, 1)
        r = np.corrcoef(xs, ys)[0, 1]
        t_ = abs(r) * np.sqrt((len(xs) - 2) / max(1 - r * r, 1e-12))
        p = 2 * (1 - stats.t.cdf(t_, len(xs) - 2))
        print(f"\ntendencia rio arriba: {sl:+.4f} m por km  "
              f"(r {r:+.3f}, p {p:.3f})")
        print(f"diferencia media global: {ys.mean():+.4f} m")
        print("\nsi la friccion domina, la diferencia debe ser POSITIVA y")
        print("crecer rio arriba; si es plana y ~0, alpha no depende del")
        print("rango y la amortiguacion no explica la compresion")

    json.dump([{"km": float(a_), "dif": float(b_), "n": c_}
               for a_, b_, c_ in res],
              open(os.path.join(SC, "vivas_muertas.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
