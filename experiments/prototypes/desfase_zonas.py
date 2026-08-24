"""Does the tidal lag grow upstream? The global fit could not have seen it.

The earlier search took the median over all 40,000 intertidal pixels and
found no lag. But a single number for the whole ria is exactly the wrong
shape for the question: a tide wave entering an estuary is delayed
progressively, so a global fit returns the average and a gradient cancels
itself out. Villaviciosa's intertidal zone runs several kilometres inland,
far more than the 720 m the field survey covers, so there is room for a
gradient to exist and to have been averaged away.

So: the same flood-versus-ebb lag estimate, run independently in bands along
the estuary axis. If the lag grows inland, the estuary correction is worth
building and we simply measured it at the wrong resolution. If every band
returns zero, the delay really is absent here and only amplification is left
— and that one needs a tide gauge, not cleverness.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
LAGS = np.arange(-120, 121, 20)
N_BANDS = 5
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    SH = tuple(d["shape"])

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        transform = s.transform

    # Northing of every kept pixel: the ria runs broadly north-south, so
    # distance from the seaward end is a fair coordinate along its axis.
    rows = keep // SH[1]
    north = transform.f + (rows + 0.5) * transform.e
    dist = (north.max() - north) / 1000.0
    edges = np.quantile(dist, np.linspace(0, 1, N_BANDS + 1))
    print(f"{len(keep):,} px, la ria abarca {dist.max():.2f} km de eje\n",
          flush=True)

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    offs = list(LAGS) + [-15, 15, 0]
    uniq_t = pd.DatetimeIndex(np.concatenate(
        [(base - pd.Timedelta(minutes=int(o))).values for o in offs])
    ).unique().sort_values()
    df = model_tides(x=[lon_c], y=[lat_c], time=uniq_t, model="EOT20",
                     directory="tide_models", crs="EPSG:4326",
                     extrapolate=True, cutoff=np.inf,
                     parallel=False).reset_index()
    lut = pd.Series(df["tide_height"].to_numpy(float),
                    index=pd.DatetimeIndex(df["time"])).groupby(level=0).first()

    def tide_at(shift):
        return lut.reindex(base - pd.Timedelta(minutes=int(shift))
                           ).to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    recent = yrs >= 2023
    dhdt = (tide_at(-15) - tide_at(15)) / 0.5
    flood = (dhdt > 0) & recent
    ebb = (dhdt <= 0) & recent
    print(f"epoca 2023-2025: {flood.sum()} escenas en flujo, "
          f"{ebb.sum()} en reflujo\n", flush=True)

    def mu_of(mask, tide, cols):
        t = tide[mask]
        good = np.isfinite(t)
        Yv = np.nan_to_num(Y[mask][good][:, cols], nan=0.0).astype(np.float64)
        Cv = C[mask][good][:, cols].astype(np.float64)
        tt = t[good]
        grid = np.linspace(tt.min(), tt.max(), MU_POINTS)
        a, b, mu, sg, rmse, N = _fit_block(Yv, Cv, tt, grid, SG_GRID)
        ok = (N >= MIN_OBS) & (b > 0.15)
        return np.where(ok, mu, np.nan)

    print(f"{'banda (km)':>14s} {'n px':>7s} " +
          " ".join(f"{l:+5d}" for l in LAGS))
    print("-" * (24 + 6 * len(LAGS)))
    out = {}
    t0 = time.time()
    for i in range(N_BANDS):
        lo_e, hi_e = edges[i], edges[i + 1]
        cols = np.flatnonzero((dist >= lo_e) & (dist < hi_e + 1e-9))
        if cols.size > 8000:
            cols = np.sort(np.random.default_rng(i).choice(cols, 8000,
                                                           replace=False))
        vals = []
        for lag in LAGS:
            t = tide_at(lag)
            mf, me = mu_of(flood, t, cols), mu_of(ebb, t, cols)
            both = np.isfinite(mf) & np.isfinite(me)
            vals.append(float(np.median(me[both] - mf[both]))
                        if both.sum() > 50 else np.nan)
        xs, ys = np.array(LAGS, float), np.array(vals)
        cross = np.nan
        for j in range(len(xs) - 1):
            if np.isfinite(ys[j]) and np.isfinite(ys[j + 1]) \
                    and ys[j] * ys[j + 1] < 0:
                cross = xs[j] - ys[j] * (xs[j+1] - xs[j]) / (ys[j+1] - ys[j])
                break
        out[f"{lo_e:.2f}-{hi_e:.2f}"] = {"n": int(cols.size),
                                         "curva": ys.tolist(),
                                         "cruce": None if not np.isfinite(cross)
                                         else float(cross)}
        print(f"{lo_e:6.2f}-{hi_e:5.2f} {cols.size:7,d} " +
              " ".join(f"{v:+5.2f}" if np.isfinite(v) else "   NA"
                       for v in ys), flush=True)

    print(f"\n({(time.time()-t0)/60:.1f} min)\n")
    print("DESFASE ESTIMADO POR BANDA (cruce por cero)")
    print(f"{'banda (km desde la bocana)':>28s} {'lag (min)':>10s}")
    print("-" * 40)
    for k, v in out.items():
        c = v["cruce"]
        print(f"{k:>28s} {('%+.1f' % c) if c is not None else 'sin cruce':>10s}")
    cs = [(float(k.split("-")[0]), v["cruce"]) for k, v in out.items()
          if v["cruce"] is not None]
    if len(cs) >= 3:
        xs_ = np.array([c[0] for c in cs])
        ys_ = np.array([c[1] for c in cs])
        sl, ic = np.polyfit(xs_, ys_, 1)
        r = np.corrcoef(xs_, ys_)[0, 1]
        print(f"\ntendencia: {sl:+.1f} min por km  (r = {r:+.3f})")
        print("si es claramente positiva, la onda SI se retrasa rio arriba")

    json.dump(out, open(os.path.join(SC, "desfase_zonas.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
