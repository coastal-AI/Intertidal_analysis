"""Estimate the estuary's tidal phase lag from the imagery alone.

The idea rests on what is and is not identifiable. Rescaling or shifting the
tide axis is an affine map: the sigmoid fit just remaps mu and sigma and its
residual does not move, so the satellite can never tell us the estuary's
amplification factor. A time lag is different — each scene moves by a
different amount depending on where in the cycle it fell — so it does leave a
signature, and we can find it without any ground truth.

The signature: if the estuary lags the ocean, then on a rising tide the real
water level is BELOW the model and on a falling tide ABOVE it. Fit the rising
scenes and the falling scenes separately and their elevations disagree; the
lag that makes them agree is the estuary's lag.

Whatever disagreement survives the best lag is not a timing problem. Water
ponding on the inner flats keeps a pixel wet after the tide has dropped past
it, and no amount of clock-shifting fixes that — so the residual asymmetry,
and how it grows upstream, separates the two hypotheses that the compression
test left tied.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
LAGS = np.arange(-180, 181, 20)          # minutes; + means the ria lags
PIX_SEARCH = 12000
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss, gy = d["field_flat"], d["gnss"], d["gy"]
    print(f"{Y.shape[0]} fechas x {Y.shape[1]:,} px", flush=True)

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])

    # Every timestamp we will ever need, in one call: the overpass shifted by
    # each candidate lag, plus a pair either side for the time derivative.
    offs = list(LAGS) + [-15, 15, 0]
    stamps = pd.DatetimeIndex(np.concatenate(
        [(base - pd.Timedelta(minutes=int(o))).values for o in offs]))
    uniq_t = stamps.unique().sort_values()
    print(f"modelando {len(uniq_t):,} instantes con EOT20...", flush=True)
    t0 = time.time()
    df = model_tides(x=[lon_c], y=[lat_c], time=uniq_t, model="EOT20",
                     directory="tide_models", crs="EPSG:4326",
                     extrapolate=True, cutoff=np.inf,
                     parallel=False).reset_index()
    lut = pd.Series(df["tide_height"].to_numpy(float),
                    index=pd.DatetimeIndex(df["time"])).groupby(level=0).first()
    print(f"  hecho en {(time.time()-t0)/60:.1f} min", flush=True)

    def tide_at(shift_min):
        return lut.reindex(base - pd.Timedelta(minutes=int(shift_min))
                           ).to_numpy(float)

    # Rising or falling, judged on the ocean model and held FIXED across
    # lags, so the two subsets never change membership as tau varies.
    dhdt = (tide_at(-15) - tide_at(15)) / 0.5
    flood = dhdt > 0
    ebb = ~flood
    h0 = tide_at(0)
    print(f"\nescenas: {flood.sum()} en flujo, {ebb.sum()} en reflujo")
    print(f"  rango de marea en flujo   {h0[flood].min():+.2f}..{h0[flood].max():+.2f}")
    print(f"  rango de marea en reflujo {h0[ebb].min():+.2f}..{h0[ebb].max():+.2f}",
          flush=True)

    rng = np.random.default_rng(5)
    sub = (np.sort(rng.choice(Y.shape[1], PIX_SEARCH, replace=False))
           if Y.shape[1] > PIX_SEARCH else np.arange(Y.shape[1]))
    Ys, Cs = Y[:, sub], C[:, sub]

    def fit(mask, tide):
        Yv = np.nan_to_num(Ys[mask], nan=0.0).astype(np.float64)
        Cv = Cs[mask].astype(np.float64)
        t = tide[mask]
        good = np.isfinite(t)
        mu_grid = np.linspace(np.nanmin(t[good]), np.nanmax(t[good]),
                              MU_POINTS)
        Cv = Cv * good[:, None]
        a, b, mu, sg, rmse, N = _fit_block(Yv[good], Cv[good], t[good],
                                           mu_grid, SG_GRID)
        ok = (N >= MIN_OBS) & (b > 0.15)
        return np.where(ok, mu, np.nan), np.where(ok, rmse, np.nan)

    print(f"\nBUSQUEDA DEL DESFASE  ({len(LAGS)} valores, {len(sub):,} px)")
    print(f"{'lag (min)':>10s} {'mediana mu_reflujo-mu_flujo':>28s} "
          f"{'|mediana|':>10s} {'n px':>7s}")
    print("-" * 60)
    curve = []
    for lag in LAGS:
        t = tide_at(lag)
        mu_f, _ = fit(flood, t)
        mu_e, _ = fit(ebb, t)
        both = np.isfinite(mu_f) & np.isfinite(mu_e)
        diff = float(np.median(mu_e[both] - mu_f[both]))
        curve.append((int(lag), diff, int(both.sum())))
        print(f"{lag:10d} {diff:28.4f} {abs(diff):10.4f} {both.sum():7d}",
              flush=True)

    arr = np.array([(l, v) for l, v, _ in curve])
    best = arr[np.argmin(np.abs(arr[:, 1])), 0]
    # Linear crossing gives a finer estimate than the grid step.
    s = np.argsort(arr[:, 0])
    xs, ys = arr[s, 0], arr[s, 1]
    cross = np.nan
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            cross = xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
            break
    print(f"\nmejor lag de la rejilla: {best:+.0f} min")
    print(f"cruce por cero interpolado: {cross:+.1f} min"
          if np.isfinite(cross) else "sin cruce por cero en el rango probado")
    print("(positivo = la ria va RETRASADA respecto al oceano)")

    json.dump({"curva": curve, "mejor_rejilla": float(best),
               "cruce": None if not np.isfinite(cross) else float(cross),
               "n_flujo": int(flood.sum()), "n_reflujo": int(ebb.sum())},
              open(os.path.join(SC, "desfase.json"), "w"), indent=1)
    print("\nguardado -> desfase.json")


if __name__ == "__main__":
    main()
