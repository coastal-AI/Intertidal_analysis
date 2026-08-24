"""Split-half repeatability: how precise is each method, with no reference DEM.

Split the epoch's dates into two halves that cover the same tide range, fit
each method on each half independently, and look at the spread between the
two answers. That spread IS the method's precision — no LiDAR needed.

    var(A - B) = 2 sigma_half^2      ->  sigma_half = std(A-B)/sqrt(2)
    sigma_full = sigma_half/sqrt(2)  =   std(A-B)/2

Also checks whether HSR's own Cramer-Rao uncertainty is calibrated (does the
predicted error match the observed spread?) and whether the elevation grid
is binding.
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
MU_POINTS = 60
SG_GRID = np.array([0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65])


def robust_sd(x):
    """1.4826 x MAD — a spread that a few wild pixels cannot inflate."""
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def fit_hsr_half(Y, C, t):
    """Stage-1 HSR exactly as the library does it, on a set of dates."""
    mu_grid = np.linspace(t.min(), t.max(), MU_POINTS)
    a, b, mu, sg, rmse, N = _fit_block(Y, C, t, mu_grid, SG_GRID)
    sg_safe = np.maximum(sg[None], 1e-3)
    phi = 0.5 * (1 + erf((t[:, None] - mu[None]) / (sg_safe * np.sqrt(2))))
    C2 = C & (np.abs(Y - (a[None] + b[None] * phi))
              < 2.5 * np.maximum(rmse[None], 0.02))
    a, b, mu, sg, rmse, N = _fit_block(Y, C2, t, mu_grid, SG_GRID)

    rng = t.max() - t.min()
    valid = ((b > 0.15) & (mu > t.min() + 0.02 * rng)
             & (mu < t.max() - 0.02 * rng) & (N >= 8))
    z = (t[None, :] - mu[:, None]) / np.maximum(sg[:, None], 1e-3)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ssum = ((b[:, None] * pdf / np.maximum(sg[:, None], 1e-3)) ** 2).sum(1)
    unc = np.where(ssum > 1e-9, rmse / np.sqrt(ssum), np.nan)
    return np.where(valid, mu, np.nan), unc, valid, sg, mu_grid


def fit_step_half(Y, C, t, threshold=0.0, n_windows=100, window_frac=0.15,
                  min_obs=5):
    """The DEA step method, same parameters as pyintertidal.fit_step."""
    nd = np.where(C, Y, np.nan)
    tmin, tmax = t.min(), t.max()
    half = 0.5 * window_frac * (tmax - tmin)
    centres = np.linspace(tmin, tmax, n_windows)
    P = Y.shape[1]
    rolling = np.full((n_windows, P), np.nan)
    for k, tc in enumerate(centres):
        sel = (t >= tc - half) & (t <= tc + half)
        if sel.any():
            with np.errstate(invalid="ignore"):
                rolling[k] = np.nanmedian(nd[sel], axis=0)
    wet = rolling >= threshold
    crosses = ((rolling[0] < threshold) & (rolling[-1] >= threshold)
               & np.isfinite(rolling[0]) & np.isfinite(rolling[-1]))
    k1 = np.argmax(wet, axis=0)
    k0 = np.clip(k1 - 1, 0, n_windows - 1)
    r1 = np.take_along_axis(rolling, k1[None], 0)[0]
    r0 = np.take_along_axis(rolling, k0[None], 0)[0]
    d = r1 - r0
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.abs(d) > 1e-6, (threshold - r0) / d, 0.0)
    frac = np.clip(np.nan_to_num(frac), 0, 1)
    z = centres[k0] + frac * (centres[k1] - centres[k0])
    good = crosses & (C.sum(0) >= min_obs)
    return np.where(good, z, np.nan)


# ── data ─────────────────────────────────────────────────────────────────
d = np.load(os.path.join(SC, "window.npz"), allow_pickle=True)
ndwi, dates = d["ndwi"], list(d["dates"])
tides = json.load(open(os.path.join(SC, "tides.json")))
T, H, W = ndwi.shape
tide_all = np.array([tides.get(x, np.nan) for x in dates], float)

ep = epochs(dates, 3)[-1]
keep = set(ep["dates"])
sel = np.array([(x in keep) and np.isfinite(tide_all[i])
                for i, x in enumerate(dates)])
t_ep = tide_all[sel]
Y_ep = np.nan_to_num(ndwi[sel].reshape(sel.sum(), H * W).astype(np.float64))
C_ep = np.isfinite(ndwi[sel].reshape(sel.sum(), H * W))
print(f"epoca {ep['label']}: {sel.sum()} fechas, "
      f"marea [{t_ep.min():.2f}, {t_ep.max():.2f}] m\n")

# Halves interleaved BY TIDE HEIGHT, so both see the same tidal range —
# splitting by calendar date would give one half a worse sampling and the
# comparison would measure that instead of the method.
order = np.argsort(t_ep)
A, B = order[0::2], order[1::2]
for nm, idx in (("A", A), ("B", B)):
    print(f"  mitad {nm}: {idx.size} fechas, marea "
          f"[{t_ep[idx].min():.2f}, {t_ep[idx].max():.2f}] m")

# ── HSR ──────────────────────────────────────────────────────────────────
print("\najustando HSR en cada mitad...")
muA, uncA, vA, sgA, mu_grid = fit_hsr_half(Y_ep[A], C_ep[A], t_ep[A])
muB, uncB, vB, sgB, _ = fit_hsr_half(Y_ep[B], C_ep[B], t_ep[B])
both = np.isfinite(muA) & np.isfinite(muB)
dif = muA[both] - muB[both]
sd_half = robust_sd(dif) / np.sqrt(2)
print(f"\nHSR   n={both.sum():,} px en las dos mitades")
print(f"  sigma por mitad  : {sd_half*100:6.1f} cm")
print(f"  sigma con todo   : {sd_half/np.sqrt(2)*100:6.1f} cm  <-- precision")
print(f"  (bruto, sin MAD) : {np.std(dif)/2*100:6.1f} cm")

# is the method's own error model calibrated?
pred = np.sqrt(uncA[both] ** 2 + uncB[both] ** 2)
print(f"  Cramer-Rao predice una diferencia de "
      f"{np.median(pred)*100:.1f} cm; observada "
      f"{robust_sd(dif)*100:.1f} cm  -> "
      f"x{robust_sd(dif)/np.median(pred):.1f}")

paso = mu_grid[1] - mu_grid[0]
print(f"  rejilla de mu: paso {paso*100:.1f} cm -> suelo por rejilla "
      f"{paso/np.sqrt(12)*100:.1f} cm")

# ── step ─────────────────────────────────────────────────────────────────
print("\najustando escalon (DEA) en cada mitad...")
zA = fit_step_half(Y_ep[A], C_ep[A], t_ep[A])
zB = fit_step_half(Y_ep[B], C_ep[B], t_ep[B])
bs = np.isfinite(zA) & np.isfinite(zB)
ds_ = zA[bs] - zB[bs]
sd_half_s = robust_sd(ds_) / np.sqrt(2)
print(f"\nescalon  n={bs.sum():,} px en las dos mitades")
print(f"  sigma por mitad  : {sd_half_s*100:6.1f} cm")
print(f"  sigma con todo   : {sd_half_s/np.sqrt(2)*100:6.1f} cm  <-- precision")
print(f"  (bruto, sin MAD) : {np.std(ds_)/2*100:6.1f} cm")

# ── same pixels, head to head ────────────────────────────────────────────
comun = both & bs
print(f"\nsobre los {comun.sum():,} px que ambos resuelven en las dos mitades:")
for nm, x, y in (("HSR    ", muA, muB), ("escalon", zA, zB)):
    dd = x[comun] - y[comun]
    print(f"  {nm}: precision {robust_sd(dd)/2*100:5.1f} cm   "
          f"(bruto {np.std(dd)/2*100:5.1f} cm)")

np.savez_compressed(os.path.join(SC, "repet.npz"),
                    muA=muA, muB=muB, zA=zA, zB=zB, both=both, bs=bs,
                    comun=comun, uncA=uncA, uncB=uncB, H=H, W=W)
print("\nOK -> repet.npz")
