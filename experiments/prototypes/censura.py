"""Censoring, not a wrong water level: pinning down what compresses the relief.

recorte.py found that the compression largely disappears when the surveyed
points within half a metre of the tide window's edge are set aside: the slope
of fitted against surveyed elevation goes from 0.820 to 0.917. The survey
spans -0.63 to +1.38 m in the tide model's frame while the archive's tide
reaches only +1.32, so the points being dropped are the HIGH ones — pixels at
or above the highest water the archive ever saw.

That has a clean physical reading and it is not the one this project has been
working from. A pixel near the top of the tidal range is submerged rarely or
never, so nothing in the archive constrains the upper tail of its transition.
The fit has no evidence above it and settles low. The elevations are censored,
and censored data pulled toward the observed range is exactly what a slope
below one looks like.

Two rival explanations have to be separated before this can be believed, and
they are entangled because high pixels are also thinly observed:

  A  CENSORING — what matters is how close the pixel is to the top of the
     tide window, regardless of how many times it was seen
  B  THIN SAMPLING — what matters is the number of usable observations, and
     elevation is incidental

They are separable because the two are not perfectly correlated: some low
pixels are poorly observed (cloud, shadow) and some high pixels are well
observed. Fitting both terms at once shows which one carries the effect.

Everything is scored against the survey, so the bootstrap is over surveyed
points, and it is a paired bootstrap so the comparison between subsets uses
the same resamples.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy import stats

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
N_BOOT = 4000


def boot_slope(y, v, rng, n=N_BOOT):
    idx = rng.integers(0, len(y), (n, len(y)))
    return np.array([np.polyfit(y[i], v[i], 1)[0] for i in idx])


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    keep, field_flat, gnss, dates = (d["keep"], d["field_flat"], d["gnss"],
                                     d["dates"])
    SH = tuple(int(v) for v in d["shape"])
    tide = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]
    yrs = np.array([int(s[:4]) for s in dates])
    tide = tide[(yrs >= 2023) & np.isfinite(tide)]
    lo, hi = float(tide.min()), float(tide.max())

    B = np.load(os.path.join(SC, "lamina_base.npz"))
    z, good, nobs, sg, b = B["z"], B["good"], B["nobs"], B["sg"], B["b"]

    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]

    zf = np.where(good, z, np.nan)[fi]
    nf, sgf = nobs[fi], sg[fi]
    m = np.isfinite(zf) & np.isfinite(y) & np.isfinite(hsr)
    y, zf, hsr, nf, sgf, fi = y[m], zf[m], hsr[m], nf[m], sgf[m], fi[m]
    off = float(np.median(y - zf))
    y_t = y - off                       # survey in the tide model's frame
    headroom = hi - y_t                 # how much tide sits above the pixel
    print(f"{len(y)} puntos · marea maxima del archivo {hi:+.2f} m")
    print(f"holgura por encima del punto: mediana {np.median(headroom):.2f} m,"
          f" minima {headroom.min():+.2f} m")
    print(f"observaciones utiles por pixel: mediana {np.median(nf):.0f}, "
          f"p10 {np.percentile(nf,10):.0f}\n")
    print(f"correlacion holgura vs n_obs: "
          f"{stats.spearmanr(headroom, nf).statistic:+.3f}  "
          f"(si fuese ~1 no se podrian separar)\n")

    # ── A against B, both terms at once ──────────────────────────────────
    err = y_t - zf
    err = err - np.median(err)
    X = np.column_stack([np.ones_like(headroom), headroom,
                         np.log(np.maximum(nf, 1))])
    coef, *_ = np.linalg.lstsq(X, err, rcond=None)
    resid = err - X @ coef
    dof = len(err) - X.shape[1]
    se = np.sqrt(np.sum(resid ** 2) / dof
                 * np.diag(np.linalg.pinv(X.T @ X)))
    print("error del ajuste explicado por las dos hipotesis a la vez")
    print(f"{'termino':22s} {'coef':>9s} {'ee':>8s} {'t':>7s}")
    print("-" * 50)
    for nm, c, e in zip(("constante", "holgura (censura)",
                         "log n_obs (muestreo)"), coef, se):
        print(f"{nm:22s} {c:+9.4f} {e:8.4f} {c/e:+7.2f}")
    print("\nun coeficiente positivo en holgura significa que el error crece")
    print("cuanto MAS margen de marea hay por encima — es decir, lo contrario")
    print("de la censura. Negativo = los puntos altos salen bajos.\n")

    # ── the subset comparison, with a paired bootstrap ───────────────────
    rng = np.random.default_rng(11)
    print(f"{'subconjunto':30s} {'pend':>7s} {'IC 95%':>17s} {'n':>5s}")
    print("-" * 62)
    out = {}
    sets = [("todos", np.ones(len(y), bool)),
            ("holgura > 0.50 m", headroom > 0.50),
            ("holgura > 0.80 m", headroom > 0.80),
            ("n_obs por encima de la mediana", nf > np.median(nf)),
            ("n_obs alto Y holgura > 0.50", (nf > np.median(nf))
             & (headroom > 0.50))]
    for lab, sel in sets:
        if sel.sum() < 25:
            print(f"{lab:30s}   solo {int(sel.sum())} puntos")
            continue
        bs = boot_slope(y[sel], zf[sel], rng, 1500)
        sl = float(np.polyfit(y[sel], zf[sel], 1)[0])
        ci = np.percentile(bs, [2.5, 97.5])
        print(f"{lab:30s} {sl:7.3f} [{ci[0]:6.3f},{ci[1]:6.3f}] "
              f"{int(sel.sum()):5d}")
        out[lab] = {"slope": sl, "ci": ci.tolist(), "n": int(sel.sum())}

    # is the difference between "all" and "headroom > 0.5" real?
    sel = headroom > 0.50
    idx = rng.integers(0, len(y), (2000, len(y)))
    diffs = []
    for i in idx:
        s = sel[i]
        if s.sum() < 25 or (~s).sum() < 5:
            continue
        diffs.append(np.polyfit(y[i][s], zf[i][s], 1)[0]
                     - np.polyfit(y[i], zf[i], 1)[0])
    diffs = np.array(diffs)
    ci = np.percentile(diffs, [2.5, 97.5])
    print(f"\ndiferencia de pendiente (holgura>0.5 menos todos): "
          f"{diffs.mean():+.3f}  IC 95% [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print("significativa si el intervalo no cruza cero.")

    json.dump({"coef": coef.tolist(), "se": se.tolist(),
               "subsets": out, "diff_ci": ci.tolist(),
               "diff_mean": float(diffs.mean())},
              open(os.path.join(SC, "censura.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
