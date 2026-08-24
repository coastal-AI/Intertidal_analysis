"""Self-consistent tide warping: learn the tide's distortion from disagreement.

The problem this attacks, established today: HSR and the published step method
both compress relief (slope 0.78-0.82 against the RTK survey) by the same
amount, so the error is in what they share — the assumed water level. And the
affine degeneracy says a per-pixel sigmoid fit cannot see a rescaling of the
tide axis, which is why every geometric attempt at correcting it has failed.

The gap in that wall: the degeneracy covers AFFINE maps. It says nothing about
a DISTORTION of the axis. And we have direct evidence the axis is distorted —
fitting the spring scenes and the neap scenes separately gives elevations that
differ by 0.25 m. Under a correct tide they would agree.

So the estimator is: find the monotone warp of the tide axis that makes
subsets of the archive agree with each other.

  * subsets are terciles of the DAILY TIDAL RANGE, because that is the split
    the disagreement was measured on
  * the warp is piecewise linear on a few knots, kept monotone
  * its affine component is projected out at every step, because that part is
    unidentifiable and leaving it free would let the optimiser wander along a
    direction the data cannot judge
  * the objective is the spread of mu across subsets, summed over pixels —
    no ground truth anywhere in it

The RTK survey is never used to fit. It is only read at the end, to see
whether making the archive self-consistent also moved the elevations toward
the ground.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from scipy.optimize import minimize

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
N_KNOTS = 6           # warp resolution; 2 of the dof are affine and removed
N_SUBSETS = 3         # terciles of daily tidal range
SEARCH_PIX = 6000     # pixels used while searching (seconds per evaluation)
MU_POINTS = 40
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 6


def fit_mu(Y, C, tide, mask, grid):
    """Elevation per pixel from one subset of scenes; NaN where unresolved."""
    m = mask & np.isfinite(tide)
    a, b, mu, sg, rmse, N = _fit_block(Y[m], C[m], tide[m], grid, SG_GRID)
    return np.where((N >= MIN_OBS) & (b > 0.15), mu, np.nan)


def make_warp(knots, w):
    """Monotone piecewise-linear warp of the tide axis, affine part removed.

    The affine component — a shift and a stretch — is exactly what the fit
    cannot see, so it is projected out rather than optimised. What remains is
    pure distortion, which is identifiable.
    """
    w = np.asarray(w, float)
    # remove constant and linear trend in knot space
    A = np.column_stack([np.ones_like(knots), knots])
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    w = w - A @ coef
    out = knots + w
    # keep it monotone: a warp that folds the axis is not a tide
    out = np.maximum.accumulate(out)
    return out


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(d["shape"])

    aoi = pit.sites.get("villaviciosa"); lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide = model_tides(x=[lon_c], y=[lat_c], time=base, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    # daily tidal range: the property that defines the subsets
    grid_h = np.vstack([
        model_tides(x=[lon_c], y=[lat_c], time=base + pd.Timedelta(hours=float(h)),
                    model="EOT20", directory="tide_models", crs="EPSG:4326",
                    extrapolate=True, cutoff=np.inf,
                    parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)
        for h in np.arange(-18, 18.5, 2.0)])
    day_range = np.nanmax(grid_h, 0) - np.nanmin(grid_h, 0)

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide) & np.isfinite(day_range)
    Y, C, tide, day_range = Y[ep], C[ep], tide[ep], day_range[ep]
    print(f"{Y.shape[0]} escenas, rango diario "
          f"{day_range.min():.2f}-{day_range.max():.2f} m", flush=True)

    # subsets: terciles of daily range, each spanning the shared tide window
    qs = np.quantile(day_range, np.linspace(0, 1, N_SUBSETS + 1))
    subsets = []
    for i in range(N_SUBSETS):
        m = (day_range >= qs[i]) & (day_range <= qs[i + 1])
        subsets.append(m)
        print(f"  subconjunto {i}: {int(m.sum())} escenas, rango medio "
              f"{day_range[m].mean():.2f} m", flush=True)

    # field pixels are kept out of the search but tracked for the final read
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat]); have = fi >= 0
    fi, target, fflat = fi[have], gnss[have], field_flat[have]

    rng = np.random.default_rng(11)
    pool = np.setdiff1d(np.arange(Y.shape[1]), fi)
    search = np.sort(rng.choice(pool, min(SEARCH_PIX, len(pool)), replace=False))
    Ys = np.nan_to_num(Y[:, search], nan=0.0).astype(np.float64)
    Cs = C[:, search].astype(np.float64)
    print(f"buscando sobre {len(search):,} px (los de campo quedan fuera)",
          flush=True)

    knots = np.linspace(tide.min(), tide.max(), N_KNOTS)

    def disagreement(w, Yv, Cv):
        warped_knots = make_warp(knots, w)
        t = np.interp(tide, knots, warped_knots)
        grid = np.linspace(t.min(), t.max(), MU_POINTS)
        mus = [fit_mu(Yv, Cv, t, m, grid) for m in subsets]
        M = np.vstack(mus)
        ok = np.isfinite(M).all(0)
        if ok.sum() < 200:
            return 10.0
        # spread across subsets, robust to the odd bad pixel
        return float(np.median(np.std(M[:, ok], axis=0)))

    base_cost = disagreement(np.zeros(N_KNOTS), Ys, Cs)
    print(f"\ndesacuerdo entre subconjuntos SIN corregir: "
          f"{base_cost:.4f} m", flush=True)

    t0 = time.time()
    evals = {"n": 0}

    def obj(w):
        evals["n"] += 1
        c = disagreement(w, Ys, Cs)
        if evals["n"] % 10 == 0:
            print(f"   eval {evals['n']:3d}  coste {c:.4f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        return c

    res = minimize(obj, np.zeros(N_KNOTS), method="Powell",
                   options={"maxfev": 120, "xtol": 1e-3, "ftol": 1e-4})
    print(f"\ndesacuerdo tras corregir: {res.fun:.4f} m "
          f"({100*(1-res.fun/base_cost):+.0f} %)", flush=True)

    warped = make_warp(knots, res.x)
    print(f"\n{'marea':>8s} {'corregida':>10s} {'delta':>8s}")
    print("-" * 30)
    for k, wv in zip(knots, warped):
        print(f"{k:8.2f} {wv:10.2f} {wv - k:+8.3f}")

    # ── the survey, read only now ────────────────────────────────────────
    Yf = np.nan_to_num(Y[:, fi], nan=0.0).astype(np.float64)
    Cf = C[:, fi].astype(np.float64)
    allm = np.ones(len(tide), bool)

    def score(t):
        grid = np.linspace(t.min(), t.max(), 60)
        mu = fit_mu(Yf, Cf, t, allm, grid)
        m = np.isfinite(mu) & np.isfinite(target)
        r = target[m] - mu[m]
        r = r - np.median(r)
        return (float(np.sqrt(np.mean(r ** 2))), int(m.sum()),
                float(np.polyfit(target[m], mu[m], 1)[0]))

    t_warp = np.interp(tide, knots, warped)
    print(f"\n{'ajuste':26s} {'RMSE':>7s} {'n':>5s} {'pendiente':>10s}")
    print("-" * 52)
    for lab, t in [("marea EOT20 sin tocar", tide),
                   ("marea deformada", t_warp)]:
        rm, n, sl = score(t)
        print(f"{lab:26s} {rm:7.3f} {n:5d} {sl:10.3f}")

    json.dump({"knots": knots.tolist(), "warped": warped.tolist(),
               "cost_before": base_cost, "cost_after": float(res.fun)},
              open(os.path.join(SC, "warp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
