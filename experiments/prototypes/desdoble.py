"""The decisive test of the hypothesis, with no labels and no solver.

The claim is that the water surface inside the ria is not flat, and that its
slope changes with the tide. If that is true, one thing must follow, and it
can be checked directly:

    Fit the archive twice — once on the big-range scenes, once on the small-
    range ones — and compare the two elevation maps. Under a FLAT water
    surface the two fits see the same water level everywhere, so wherever
    they disagree, they should disagree by the same amount at the mouth as
    3 km upstream. Under a SLOPING surface whose slope depends on the tide,
    the disagreement must GROW with channel distance.

So the test is: regress (mu_big - mu_small) on s. A slope indistinguishable
from zero refutes the whole method. A slope significantly different from zero
is the effect the method exists to model, measured before modelling it.

This is worth far more than the survey ceiling:

  * it uses the full 3.5 km of the ria, not the 745 m the RTK campaign spans
  * it uses all ~40 000 intertidal pixels, not 192 points
  * no ground truth enters, so it cannot be an artefact of the reference

Two controls, because this measurement has an obvious way to fool itself:

  1. Both subsets are clipped to their COMMON tide window and their tide
     histograms matched, exactly as tidecheck.range_dependence does — without
     that, the two fits cover different parts of the profile and the
     difference is pure artefact. That correction was worth 0.23 m of the
     0.25 m measured this afternoon, so it is not optional.
  2. A PERMUTATION null. A first attempt used a coin flip for the placebo and
     it was not a fair control: the real split leaves 38 scenes per subset
     while a coin flip leaves ~150, so the placebo fits were far less noisy
     and understated the null. It still produced a -0.124 m/km trend at
     t = -28.9 out of nothing, which is the useful lesson — the p-values here
     are worthless, because neighbouring pixels are not independent and both
     subsets share the same pixels.

     A permutation of the daily range was the second attempt, and it is still
     not matched: permuting makes the two groups similar, so their common tide
     window is WIDE, ~150 scenes survive per group and ~30 000 pixels resolve.
     The real split has a narrow window, 38 scenes and 6 888 pixels. Fewer
     scenes means noisier per-pixel fits, so that null is quieter than the
     thing it is supposed to bound, and it flatters the result.

     The MATCHED null fixes it exactly: take the 76 scenes the real split
     actually used, and inside every tide-histogram bin deal them at random
     into two groups of the same size. Scene count, tide window and histograms
     are identical to the real split by construction. The only thing destroyed
     is which scenes were springs and which were neaps.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
from scipy import stats

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 50
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
N_HIST_BINS = 12
N_PERM = 12


def matched_subsets(tide, big, small, rng):
    """Clip to the common tide window, then match the two tide histograms.

    Both corrections are what turned this afternoon's raw -0.23 m artefact
    into a real -0.25 m effect. Without them the two fits simply sample
    different parts of the tidal profile and any trend is meaningless.
    """
    lo = max(tide[big].min(), tide[small].min())
    hi = min(tide[big].max(), tide[small].max())
    win = (tide >= lo) & (tide <= hi)
    big, small = big & win, small & win

    edges = np.linspace(lo, hi, N_HIST_BINS + 1)
    keep_b, keep_s = [], []
    for i in range(N_HIST_BINS):
        b = np.where(big & (tide >= edges[i]) & (tide < edges[i + 1]))[0]
        s = np.where(small & (tide >= edges[i]) & (tide < edges[i + 1]))[0]
        n = min(len(b), len(s))
        if n == 0:
            continue
        keep_b += list(rng.choice(b, n, replace=False))
        keep_s += list(rng.choice(s, n, replace=False))
    mb = np.zeros_like(big); mb[keep_b] = True
    ms = np.zeros_like(small); ms[keep_s] = True
    return mb, ms, (lo, hi)


def fit_mu(Y, C, tide, sel):
    grid = np.linspace(tide[sel].min(), tide[sel].max(), MU_POINTS)
    a, b, mu, sg, rmse, N = _fit_block(Y[sel], C[sel], tide[sel],
                                       grid, SG_GRID)
    return np.where((N >= MIN_OBS) & (b > 0.15), mu, np.nan)


def trend(diff, s_km, label):
    ok = np.isfinite(diff) & np.isfinite(s_km)
    n = int(ok.sum())
    if n < 500:
        print(f"  {label:26s} solo {n} px, insuficiente")
        return None
    res = stats.linregress(s_km[ok], diff[ok])
    print(f"  {label:26s} {res.slope:+7.3f} m/km  "
          f"±{res.stderr:.3f}  t={res.slope/res.stderr:+6.1f}  "
          f"p={res.pvalue:8.2g}  n={n:,}")
    return {"slope": float(res.slope), "stderr": float(res.stderr),
            "p": float(res.pvalue), "n": n,
            "mean_diff": float(np.nanmean(diff[ok]))}


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    ch = np.load(os.path.join(SC, "canal.npz"))
    s_km = ch["s_keep"].astype(float) / 1000.0

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])

    def tides_at(times):
        return model_tides(x=[lon_c], y=[lat_c], time=times, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    cache = os.path.join(SC, "mareas_villa.npz")
    if os.path.exists(cache):
        cc = np.load(cache)
        tide, day_range = cc["tide"], cc["day_range"]
        print("mareas leidas de cache")
    else:
        tide = tides_at(base)
        grid_h = np.vstack([tides_at(base + pd.Timedelta(hours=float(h)))
                            for h in np.arange(-18, 18.5, 3.0)])
        day_range = np.nanmax(grid_h, 0) - np.nanmin(grid_h, 0)
        np.savez_compressed(cache, tide=tide, day_range=day_range,
                            dates=dates)

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide) & np.isfinite(day_range)
    Y, C, tide, day_range = Y[ep], C[ep], tide[ep], day_range[ep]
    Y = np.nan_to_num(Y, nan=0.0).astype(np.float64)
    C = C.astype(np.float64)
    print(f"{Y.shape[0]} escenas · {Y.shape[1]:,} px · rango diario "
          f"{day_range.min():.2f}-{day_range.max():.2f} m")
    print(f"distancia por canal: {np.nanmin(s_km):.2f}-{np.nanmax(s_km):.2f} km\n")

    def matched_null(big, small, seed, label):
        """Re-deal the SAME scenes into two groups, bin by bin.

        Identical scene count, window and tide histogram to the real split.
        Whatever trend this produces is what the machinery manufactures from
        a split that carries no tidal-range information at all.
        """
        r = np.random.default_rng(seed)
        pool = np.where(big | small)[0]
        lo, hi = tide[pool].min(), tide[pool].max()
        edges = np.linspace(lo, hi, N_HIST_BINS + 1)
        edges[-1] += 1e-9
        ma = np.zeros(len(tide), bool)
        mb = np.zeros(len(tide), bool)
        for i in range(N_HIST_BINS):
            inb = pool[(tide[pool] >= edges[i]) & (tide[pool] < edges[i + 1])]
            if len(inb) < 2:
                continue
            perm = r.permutation(inb)
            half = len(inb) // 2
            ma[perm[:half]] = True
            mb[perm[half:2 * half]] = True
        if ma.sum() < 10 or mb.sum() < 10:
            return None
        diff = fit_mu(Y, C, tide, ma) - fit_mu(Y, C, tide, mb)
        res = trend(diff, s_km, label)
        if res:
            res["n_scenes"] = [int(ma.sum()), int(mb.sum())]
        return res

    def one_split(dr, seed, label):
        """Tercile split on `dr`, matched, fitted, regressed on distance."""
        r = np.random.default_rng(seed)
        lo_q, hi_q = np.quantile(dr, [1 / 3, 2 / 3])
        big, small, win = matched_subsets(tide, dr >= hi_q, dr <= lo_q, r)
        if big.sum() < 10 or small.sum() < 10:
            return None
        diff = fit_mu(Y, C, tide, big) - fit_mu(Y, C, tide, small)
        res = trend(diff, s_km, label)
        if res:
            res["n_scenes"] = [int(big.sum()), int(small.sum())]
            res["window"] = [float(win[0]), float(win[1])]
        return res, big, small

    out = {}
    print("REAL — mareas vivas contra muertas")
    real, big, small = one_split(day_range, 7, "vivas - muertas")
    out["real"] = real
    if real:
        print(f"  ventana comun {real['window'][0]:.2f} a "
              f"{real['window'][1]:.2f} m · "
              f"{real['n_scenes']} escenas · {real['n']:,} px resueltos")

    print(f"\nNULO IGUALADO — las MISMAS 76 escenas, repartidas al azar bin a bin")
    null = []
    for k in range(N_PERM):
        res = matched_null(big, small, 900 + k, f"reparto {k+1}")
        if res:
            null.append(res["slope"])
    out["null_matched"] = null

    if real and null:
        r_eff = real["slope"]
        nu = np.array(null)
        beats = int((np.abs(nu) >= abs(r_eff)).sum())
        z = (r_eff - nu.mean()) / nu.std(ddof=1)
        p_perm = (beats + 1) / (len(nu) + 1)
        print(f"\n{'-'*62}")
        print(f"efecto real        {r_eff:+.3f} m/km")
        print(f"nulo igualado      media {nu.mean():+.3f}, "
              f"sd {nu.std(ddof=1):.3f}, |max| {np.abs(nu).max():.3f}")
        print(f"z contra el nulo   {z:+.1f}")
        print(f"repartos que igualan o superan el efecto: "
              f"{beats} de {len(nu)}  (p = {p_perm:.3f})")
        if beats == 0 and z > 3:
            v = "EFECTO ESTABLECIDO: la lamina se inclina y depende de la marea"
        elif beats == 0:
            v = ("SUGERENTE PERO NO ESTABLECIDO: ningun reparto lo iguala, "
                 f"pero con {len(nu)} repartos lo mejor que se puede afirmar "
                 f"es p = {p_perm:.3f}")
        else:
            v = "NO SE DISTINGUE DEL NULO: la hipotesis no se sostiene"
        print(f"\n{v}")
        out["p_perm"] = p_perm
        out["z"] = float(z)

    json.dump(out, open(os.path.join(SC, "desdoble.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
