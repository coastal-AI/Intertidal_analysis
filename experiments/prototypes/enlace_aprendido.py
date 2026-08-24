"""Stop assuming the sub-pixel hypsometry is Gaussian — estimate its shape.

HSR writes the wet fraction as Phi((h-mu)/sigma), and Phi is there because we
assumed the elevations inside a 10 m pixel are normally distributed. That is
the one soft assumption in the method. A real flat with a microchannel is
skewed, not normal.

So learn the shape instead of imposing it. Every pixel keeps its own mu and
sigma; what is shared is the SHAPE of the curve, and there are 40 000 pixels
and millions of observations to estimate one shared function from. No labels
are involved: this is learning from the imagery itself.

  1. fit HSR normally, obtaining a, b, mu, sigma per pixel
  2. for every usable observation compute z = (h - mu)/sigma and the
     normalised response (NDWI - a)/b, which should equal Phi(z) if the
     assumption holds
  3. average the normalised response in bins of z: that IS the empirical
     link function g(z)
  4. compare g against Phi — the gap is the error the assumption costs
  5. refit every pixel using g in place of Phi and score against the field
     survey

Step 4 is worth having whatever step 5 says: it measures how wrong the
Gaussian assumption is, which nobody has reported for this method.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from scipy.special import erf

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
Z_EDGES = np.linspace(-3.0, 3.0, 25)
FIT_PIX = 20000


def phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def fit_with_link(Y, C, tide, mu_grid, sg_grid, link):
    """_fit_block with an arbitrary monotone link in place of Phi.

    Same closed form: with mu and sigma fixed, a and b are least squares, so
    this stays a sweep over a 2-D grid rather than an optimiser.
    """
    Cf = C.astype(np.float64)
    Yc = np.where(Cf > 0, Y, 0.0) * Cf
    N = Cf.sum(0)
    Sy = Yc.sum(0)
    Syy = (np.where(Cf > 0, Y, 0.0) * Yc).sum(0)
    P = Y.shape[1]
    best_loss = np.full(P, np.inf)
    best = np.zeros((4, P))
    for mu in mu_grid:
        for sg in sg_grid:
            g = link((tide - mu) / sg)
            Sp = Cf.T @ g
            Spp = Cf.T @ (g * g)
            Syp = Yc.T @ g
            det = N * Spp - Sp * Sp
            ok = det > 1e-9
            b = np.where(ok, (N * Syp - Sp * Sy) / np.where(ok, det, 1.0), 0.0)
            a = np.where(N > 0, (Sy - b * Sp) / np.where(N > 0, N, 1.0), 0.0)
            loss = (Syy - 2 * a * Sy - 2 * b * Syp + a * a * N
                    + 2 * a * b * Sp + b * b * Spp)
            upd = ok & (loss < best_loss)
            best_loss = np.where(upd, loss, best_loss)
            for i, v in enumerate((a, b, np.full(P, mu), np.full(P, sg))):
                best[i] = np.where(upd, v, best[i])
    return best[0], best[1], best[2], best[3], N


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(d["shape"])

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide = model_tides(x=[lon_c], y=[lat_c], time=times, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)
    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide)
    Y, C, tide = Y[ep], C[ep], tide[ep]

    # field pixels must survive any subsampling: they are the only check
    pos = {int(k): i for i, k in enumerate(keep)}
    fidx = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fidx >= 0
    fidx, target, fflat = fidx[have], gnss[have], field_flat[have]

    rng = np.random.default_rng(5)
    pool = np.setdiff1d(np.arange(Y.shape[1]), fidx)
    sub = np.sort(np.concatenate(
        [fidx, rng.choice(pool, min(FIT_PIX - len(fidx), len(pool)),
                          replace=False)]))
    where_field = np.searchsorted(sub, fidx)
    Ys = np.nan_to_num(Y[:, sub], nan=0.0).astype(np.float64)
    Cs = C[:, sub].astype(np.float64)
    print(f"{Ys.shape[0]} escenas x {Ys.shape[1]:,} px "
          f"({len(fidx)} de campo)", flush=True)

    grid = np.linspace(tide.min(), tide.max(), MU_POINTS)
    a, b, mu, sg, rmse, N = _fit_block(Ys, Cs, tide, grid, SG_GRID)
    good = (N >= 8) & (b > 0.15)
    print(f"{int(good.sum()):,} px con ajuste util", flush=True)

    # ── step 2-3: the empirical link ─────────────────────────────────────
    z = (tide[:, None] - mu[None, good]) / np.maximum(sg[None, good], 1e-3)
    resp = (Ys[:, good] - a[None, good]) / np.maximum(b[None, good], 1e-6)
    m = (Cs[:, good] > 0) & np.isfinite(z) & np.isfinite(resp)

    centres, emp, counts = [], [], []
    for i in range(len(Z_EDGES) - 1):
        sel = m & (z >= Z_EDGES[i]) & (z < Z_EDGES[i + 1])
        k = int(sel.sum())
        if k < 500:
            continue
        centres.append(0.5 * (Z_EDGES[i] + Z_EDGES[i + 1]))
        emp.append(float(np.mean(resp[sel])))
        counts.append(k)
    centres = np.array(centres); emp = np.array(emp)
    print(f"\nfuncion de enlace empirica, {sum(counts):,} observaciones")
    print(f"{'z':>6s} {'empirica':>9s} {'Phi(z)':>8s} {'dif':>7s} {'n':>10s}")
    print("-" * 46)
    for c, e, k in zip(centres, emp, counts):
        print(f"{c:6.2f} {e:9.3f} {phi(c):8.3f} {e - phi(c):+7.3f} {k:10,d}")
    print(f"\ndesviacion RMS respecto a Phi: "
          f"{np.sqrt(np.mean((emp - phi(centres)) ** 2)):.4f}")

    # monotone, clipped, and extended flat beyond the fitted range
    iso = np.maximum.accumulate(emp)
    iso = np.clip(iso, 0.0, 1.0)

    def learned(zz):
        return np.interp(zz, centres, iso, left=iso[0], right=iso[-1])

    # ── step 5: refit with the learned link ──────────────────────────────
    a2, b2, mu2, sg2, N2 = fit_with_link(Ys, Cs, tide, grid, SG_GRID, learned)
    good2 = (N2 >= 8) & (b2 > 0.15)

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        tr = s.transform

    def score(vals, ok):
        v = vals[where_field]
        k = ok[where_field] & np.isfinite(v) & np.isfinite(target)
        r = target[k] - v[k]
        r = r - np.median(r)
        return float(np.sqrt(np.mean(r ** 2))), int(k.sum()), \
            float(np.polyfit(target[k], v[k], 1)[0])

    print(f"\n{'ajuste':28s} {'RMSE':>7s} {'n':>5s} {'pendiente':>10s}")
    print("-" * 54)
    for lab, vals, ok in [("HSR (Phi gaussiana)", mu, good),
                          ("HSR (enlace aprendido)", mu2, good2)]:
        rm, n, sl = score(vals, ok)
        print(f"{lab:28s} {rm:7.3f} {n:5d} {sl:10.3f}")

    json.dump({"z": centres.tolist(), "empirical": emp.tolist(),
               "phi": phi(centres).tolist()},
              open(os.path.join(SC, "enlace.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
