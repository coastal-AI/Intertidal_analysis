import sys, os, math, re
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
import numpy as np, xarray as xr

# ─────────────────────────────────────────────────────────────────────────────
# MÉTODO ESCALÓN (DEA): mediana móvil NDWI vs marea -> cruce seco->mojado
# ─────────────────────────────────────────────────────────────────────────────
def step_elevation(ndwi, tide_array, clear_mask, threshold,
                   n_windows=100, window_frac=0.15, min_obs=5):
    n_t, h, w = ndwi.shape
    valid_t = np.isfinite(tide_array)
    tides = tide_array[valid_t]
    nan = np.full((h, w), np.nan, np.float32)
    if tides.size < 2:
        return nan, nan.copy(), np.zeros((h, w), np.float32)
    tmin, tmax = float(tides.min()), float(tides.max())
    rng = tmax - tmin
    if rng <= 0:
        return nan, nan.copy(), np.zeros((h, w), np.float32)
    half = 0.5 * window_frac * rng
    centers = np.linspace(tmin, tmax, n_windows)
    nd = np.where(clear_mask, ndwi, np.nan)
    rolling = np.full((n_windows, h, w), np.nan, np.float32)
    for k, tc in enumerate(centers):
        sel = valid_t & (tide_array >= tc - half) & (tide_array <= tc + half)
        if sel.any():
            with np.errstate(invalid="ignore"):
                rolling[k] = np.nanmedian(nd[sel], axis=0)
    obs = np.sum(clear_mask, axis=0).astype(np.float32)
    wet = rolling >= threshold
    dry_low = (rolling[0] < threshold)                 # seco a marea baja
    wet_high = (rolling[-1] >= threshold)              # mojado a marea alta
    inter = dry_low & wet_high & np.isfinite(rolling[0]) & np.isfinite(rolling[-1])
    k1 = np.argmax(wet, axis=0)                         # primera ventana mojada
    k0 = np.clip(k1 - 1, 0, n_windows - 1)
    r1 = np.take_along_axis(rolling, k1[None], 0)[0]
    r0 = np.take_along_axis(rolling, k0[None], 0)[0]
    c1 = centers[k1]; c0 = centers[k0]
    d = r1 - r0
    frac = np.clip(np.where(np.abs(d) > 1e-6, (threshold - r0) / d, 0.0), 0, 1)
    z = (c0 + frac * (c1 - c0)).astype(np.float32)
    elev = np.where(inter & (obs >= min_obs), z, np.nan).astype(np.float32)
    conf = np.where(np.isfinite(elev), np.clip(obs / (2 * min_obs), 0, 1), 0.0).astype(np.float32)
    return elev, conf, obs

# ─────────────────────────────────────────────────────────────────────────────
# MÉTODO NOVEL: Soft Inundation Quantile (SIQ) -> inversión directa de la CDF de mareas
# ─────────────────────────────────────────────────────────────────────────────
def siq_elevation(ndwi, tide_array, clear_mask, threshold,
                  soft_halfwidth=0.1, min_obs=5):
    n_t, h, w = ndwi.shape
    valid_t = np.isfinite(tide_array)
    lo, hi = threshold - soft_halfwidth, threshold + soft_halfwidth
    w_soft = np.clip((ndwi - lo) / (hi - lo), 0.0, 1.0)
    usable = clear_mask & valid_t[:, None, None]
    w_soft = np.where(usable, w_soft, 0.0)
    clear = usable.astype(np.float32)
    n_wet = np.sum(w_soft, axis=0)                      # soft wet count (y,x)
    clear_votes = np.sum(clear, axis=0)                 # (y,x)
    order = np.argsort(tide_array)[::-1]                # marea DESC
    tide_sorted = tide_array[order]
    cum = np.cumsum(clear[order], axis=0)               # obs claras acumuladas desde marea alta
    reached = cum >= n_wet[None]
    any_reached = reached.any(axis=0)
    k1 = np.argmax(reached, axis=0)
    k0 = np.clip(k1 - 1, 0, n_t - 1)
    cum1 = np.take_along_axis(cum, k1[None], 0)[0]
    cum0 = np.take_along_axis(cum, k0[None], 0)[0]
    t1 = tide_sorted[k1]; t0 = tide_sorted[k0]
    d = cum1 - cum0
    frac = np.clip(np.where(np.abs(d) > 1e-6, (n_wet - cum0) / d, 0.0), 0, 1)
    z = (t0 + frac * (t1 - t0)).astype(np.float32)
    F = np.where(clear_votes > 0, n_wet / np.where(clear_votes > 0, clear_votes, 1), np.nan)
    elev = np.where((clear_votes >= min_obs) & (F > 0.001) & (F < 0.999) & any_reached,
                    z, np.nan).astype(np.float32)
    conf = np.where(np.isfinite(elev),
                    np.clip(4 * F * (1 - F), 0, 1) * np.clip(clear_votes / (2 * min_obs), 0, 1),
                    0.0).astype(np.float32)
    return elev, conf, clear_votes.astype(np.float32)

# ── cargar cubo NDWI cacheado + mareas sintéticas ──
NC = r"C:\Users\Jorge\AppData\Local\Temp\ndwi_villaviciosa_1yr.nc"
ds = xr.open_dataset(NC); td = "t" if "t" in ds["SCL"].dims else ds["SCL"].dims[0]
b03 = np.asarray(ds["B03"].transpose(td, "y", "x").values, np.float32)
b08 = np.asarray(ds["B08"].transpose(td, "y", "x").values, np.float32)
scl = np.asarray(ds["SCL"].transpose(td, "y", "x").values, np.int16)
dates = [re.search(r"\d{4}-\d{2}-\d{2}", str(v)).group(0) for v in np.asarray(ds["SCL"][td].values)]
ds.close()
den = b03 + b08
with np.errstate(invalid="ignore", divide="ignore"):
    ndwi = np.where(den != 0, (b03 - b08) / den, np.nan).astype(np.float32)
clear = np.isin(scl, [4, 5, 6, 12])
tide = np.array([2.0*math.sin(i*0.9) + 0.6*math.sin(i*0.27) for i in range(len(dates))], np.float32)
print("cubo:", ndwi.shape, "| mareas [", round(tide.min(),2), ",", round(tide.max(),2), "]")

for name, fn in [("ESCALÓN", step_elevation), ("SIQ (novel)", siq_elevation)]:
    e, c, o = fn(ndwi, tide, clear, 0.0)
    v = np.isfinite(e)
    print(f"\n{name}: válidos {v.sum()} ({100*v.mean():.1f}%) | "
          f"cota [{np.nanmin(e):.2f},{np.nanmax(e):.2f}] m | conf media {np.nanmean(c):.3f}")
    print(f"   dentro del rango de marea: {bool(np.nanmin(e)>=tide.min()-0.5 and np.nanmax(e)<=tide.max()+0.5)}")
print("\nTEST STEP+SIQ DONE")
