"""Step 2: fit the sigmoid on the extracted window, for two time windows.

Reuses pyintertidal.elevation._fit_block verbatim, so what the figures show
is what the library computes.
"""
import os, sys, json
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
from scipy.special import erf
from pyintertidal.elevation import _fit_block, epochs

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

d = np.load(os.path.join(SC, "window.npz"), allow_pickle=True)
ndwi, dates = d["ndwi"], list(d["dates"])
tides = json.load(open(os.path.join(SC, "tides.json")))

T, H, W = ndwi.shape
tide = np.array([tides.get(x, np.nan) for x in dates], float)

MU_POINTS = 60
SG_GRID = np.array([0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65])
SG_MAX = SG_GRID[-1]

eps = epochs(dates, 3)
print("epocas:", [(e["label"], len(e["dates"])) for e in eps])
last = eps[-1]["label"]

runs = {"10 anos (2016-2025)": set(dates),
        f"epoca {last}": set(eps[-1]["dates"])}

results = {}
for label, keep in runs.items():
    sel = np.array([(x in keep) and np.isfinite(tide[i])
                    for i, x in enumerate(dates)])
    t = tide[sel]
    Y = ndwi[sel].reshape(sel.sum(), H * W).astype(np.float64)
    C = np.isfinite(Y)
    Y = np.nan_to_num(Y)
    mu_grid = np.linspace(t.min(), t.max(), MU_POINTS)

    a, b, mu, sg, rmse, N = _fit_block(Y, C, t, mu_grid, SG_GRID)
    C2 = C
    for _ in range(1):                                  # robust_iters=1
        sg_safe = np.maximum(sg[None], 1e-3)
        phi = 0.5 * (1 + erf((t[:, None] - mu[None]) / (sg_safe * np.sqrt(2))))
        resid = np.abs(Y - (a[None] + b[None] * phi))
        C2 = C & (resid < 2.5 * np.maximum(rmse[None], 0.02))
        a, b, mu, sg, rmse, N = _fit_block(Y, C2, t, mu_grid, SG_GRID)

    rng = t.max() - t.min()
    valid = ((b > 0.15) & (mu > t.min() + 0.02 * rng)
             & (mu < t.max() - 0.02 * rng) & (N >= 8))

    # Cramer-Rao
    z = (t[None, :] - mu[:, None]) / np.maximum(sg[:, None], 1e-3)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ssum = ((b[:, None] * pdf / np.maximum(sg[:, None], 1e-3)) ** 2).sum(1)
    sigma_mu = np.where(ssum > 1e-9, rmse / np.sqrt(ssum), np.nan)

    sat = (sg[valid] >= SG_MAX).mean() * 100
    print(f"\n{label}: {sel.sum()} fechas, marea [{t.min():.2f},{t.max():.2f}]")
    print(f"  ajustados {valid.sum():,}/{H*W:,} px")
    print(f"  sigma saturada: {sat:.0f} %   mediana sigma {np.median(sg[valid]):.3f} m")
    print(f"  incertidumbre Cramer-Rao mediana: "
          f"{np.nanmedian(sigma_mu[valid])*100:.1f} cm")
    print(f"  rmse ajuste mediano: {np.median(rmse[valid]):.4f} NDWI")

    results[label] = dict(a=a, b=b, mu=mu, sigma=sg, rmse=rmse, n_obs=N,
                          valid=valid, sigma_mu=sigma_mu, tide=t,
                          sat=np.float64(sat))
    if label.startswith("epoca"):
        epoch_pack = dict(epoch_Y=Y.astype("float32"), epoch_C=C2,
                          epoch_tide=t)

np.savez_compressed(
    os.path.join(SC, "fit.npz"),
    labels=np.array(list(results)),
    **{f"{i}_{k}": v for i, r in enumerate(results.values())
       for k, v in r.items()},
    **epoch_pack, H=H, W=W, epoch_label=last)
print("\nOK -> fit.npz")
