"""C5 — the full clock × estimator matrix against the RTK truth (Villaviciosa).

The only site with field truth. Every cell of the matrix is judged on the
same development RTK pixels (the reserved 35 % of 150-m blocks, seed
20260817, stays sealed), median-centred (ellipsoidal RTK vs tide-datum
products), RMSE and OLS slope; then on the common subset so nobody wins on
easier pixels.

Rows (clock × estimator):

  estimator = our NDWI sigmoid (marea.invert_series), 2023-25 epoch
    · ocean clock (tau = 0)                          plain
    · MAREA band clocks, applied product              MAREA
    · MAREA fitted clocks without the threshold       MAREA fitted
    · Granadeiro's per-pixel lag map (crossed)        crossed
  estimator = Granadeiro's NIR logistic, 2023-25 epoch (their <10 % rule)
    · ocean clock                                     Granadeiro (no lags)
    · their lag map                                   Granadeiro (their lags)
    · MAREA band clocks (crossed the other way)       Granadeiro on MAREA clock
  shipped products, for reference
    · HSR 2023-25 (ocean clock; tide-binned sigmoid)  HSR
    · step / DEA-style                                step
    · Granadeiro 10 y (no lags / their lags)

Run:  python -m experiments.c5_validation_matrix
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

EPOCH = (2023, 2025)
BLOCK_M, HOLDOUT_FRACTION, SEED = 150.0, 0.35, 20260817


def main():
    import pandas as pd
    import rasterio

    import pyintertidal as pit
    from pyintertidal import marea
    from pyintertidal.boundary import PyTMDBoundary
    from experiments import sota_granadeiro as sg
    pit.net.use_system_certificates()

    # ── the record and the RTK pixels ────────────────────────────────────
    base = np.load("marea_demo_extract.npz", allow_pickle=True)
    Y, C, keep = base["Y"], base["C"].astype(bool), base["keep"]
    H, W = (int(v) for v in base["shape"])
    dates, bbox = np.array([str(d) for d in base["dates"]]), base["bbox"]
    bb = {"west": float(bbox[0]), "south": float(bbox[1]),
          "east": float(bbox[2]), "north": float(bbox[3]), "crs": "EPSG:4326"}
    campo = np.load("products_villaviciosa/campo.npz", allow_pickle=True)
    rr, cc, gnss = campo["row"], campo["col"], campo["gnss"]
    flat = rr * W + cc
    pos = {int(k): i for i, k in enumerate(keep)}
    ki = np.array([pos.get(int(f), -1) for f in flat])
    inrec = ki >= 0
    print(f"{len(flat)} RTK pixels, {int(inrec.sum())} inside the intertidal record")

    yr = np.array([int(d[:4]) for d in dates])
    times = pit.overpass.get_overpass_times(bb, ("2016-01-01", "2025-12-31"),
                                            verbose=False)
    has_t = np.array([d in times for d in dates])
    ep = (yr >= EPOCH[0]) & (yr <= EPOCH[1]) & has_t
    t_real = pd.DatetimeIndex(pd.to_datetime([times[d] for d in dates[ep]])).tz_localize(None)
    cols = ki[inrec]
    Ye = np.nan_to_num(Y[ep][:, cols], nan=0.0).astype(np.float64)
    Ce = C[ep][:, cols].astype(np.float64)
    print(f"{int(ep.sum())} epoch scenes with an overpass time")

    lat_c, lon_c = 0.5 * (bb["south"] + bb["north"]), 0.5 * (bb["west"] + bb["east"])
    eot = PyTMDBoundary("EOT20", lat_c, lon_c, directory="tide_models")
    h0 = np.asarray(eot.levels(t_real), float)
    lo, hi = float(np.nanmin(h0)), float(np.nanmax(h0))

    def shifted(tau_min):
        return np.asarray(eot.levels(t_real - pd.Timedelta(minutes=float(tau_min))), float)

    def full(vals_on_cols):
        v = np.full(len(flat), np.nan); v[inrec] = vals_on_cols; return v

    prods = {}

    # ── our estimator under four clocks ──────────────────────────────────
    z, _ = marea.invert_series(Ye, Ce, h0, lo, hi)
    prods["sigmoid · ocean clock (plain)"] = full(z)

    mz = np.load("products_villaviciosa_marea/marea.npz")
    assert np.array_equal(mz["keep"], keep)
    band = mz["band"].astype(int)[cols]
    tau_used, tau_fit = np.asarray(mz["tau_usado_min"], float), np.asarray(mz["tau_min"], float)
    prods["sigmoid · MAREA band clocks (product)"] = full(mz["z"].astype(float)[cols])
    for label, taus in (("sigmoid · MAREA clocks, applied (recomputed)", tau_used),
                        ("sigmoid · MAREA fitted clocks, no threshold", tau_fit)):
        zz = np.full(len(cols), np.nan)
        for k in np.unique(band[band >= 0]):
            sel = np.flatnonzero(band == k)
            hk = shifted(taus[k]) if taus[k] else h0
            zz[sel], _ = marea.invert_series(Ye[:, sel], Ce[:, sel], hk, lo, hi)
        prods[label] = full(zz)

    g = np.load("granadeiro_products_2023-2025.npz")
    assert np.array_equal(g["keep"], keep)
    lag = g["lag_all"].astype(float)[cols]
    zz = np.full(len(cols), np.nan)
    lag_q = np.round(lag / 5.0) * 5.0
    for lq in np.unique(lag_q):
        sel = np.flatnonzero(lag_q == lq)
        zz[sel], _ = marea.invert_series(Ye[:, sel], Ce[:, sel], shifted(lq), lo, hi)
    prods["sigmoid · Granadeiro lag map (crossed)"] = full(zz)

    # ── Granadeiro's estimator under three clocks ────────────────────────
    gran = np.load("granadeiro_extract.npz", allow_pickle=True)
    NIR, calib = gran["NIR"], gran["calib"]
    bad, nodata, ref_idx = gran["bad_frac"], gran["nodata_frac"], int(gran["ref_idx"])
    use = (bad <= 0.10) & (nodata <= 0.02) & ep
    NIRu = np.array(NIR[use][:, cols], np.float32)
    calibu = calib[use]
    ref_pos = int(np.flatnonzero(np.flatnonzero(use) == ref_idx)[0]) if use[ref_idx] else 0
    NIRc, _ = sg.intercalibrate(NIRu, calibu, ref_pos)
    NIRc[~C[use][:, cols]] = np.nan
    rho = np.column_stack([sg.standardise(NIRc[:, j]) for j in range(NIRc.shape[1])])
    t_g = pd.DatetimeIndex(pd.to_datetime([times[d] for d in dates[use]])).tz_localize(None)
    t_sat_s = t_g.view("int64") / 1e9
    st = pd.date_range("2022-11-01", "2026-02-01", freq="15min")
    sh = np.asarray(eot.levels(st), float)
    ext_t_dt, ext_h = sg.tide_extremes(st, sh)
    ext_t = ext_t_dt.view("int64") / 1e9
    print(f"Granadeiro rows: {int(use.sum())} scenes under their <10 % rule")

    def logistic_rows(tau_per_px, label):
        zz = np.full(len(cols), np.nan)
        for j in range(len(cols)):
            h_j = sg.hsat_eq1(t_sat_s - float(tau_per_px[j]) * 60.0, ext_t, ext_h)
            f = sg.fit_logistic4(h_j, rho[:, j])
            if f is not None:
                zz[j] = f[3]
        prods[label] = full(zz)

    logistic_rows(np.zeros(len(cols)), "logistic · ocean clock (Granadeiro, no lags)")
    logistic_rows(lag, "logistic · their lag map (Granadeiro)")
    logistic_rows(np.array([tau_used[k] if k >= 0 else 0.0 for k in band]),
                  "logistic · MAREA band clocks (crossed)")

    # ── shipped products ─────────────────────────────────────────────────
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        prods["HSR 2023-25 (shipped, ocean clock)"] = s.read(1)[rr, cc].astype(float)
        tr = s.transform
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        prods["step / DEA-style (shipped)"] = s.read(1)[rr, cc].astype(float)
    g10 = np.load("granadeiro_products.npz")
    p10 = {int(k): i for i, k in enumerate(g10["keep"])}
    gi = np.array([p10.get(int(f), -1) for f in flat])
    for key, label in (("prelim", "logistic 10 y · ocean clock (shipped)"),
                       ("final", "logistic 10 y · their lags (shipped)")):
        v = np.full(len(flat), np.nan); m = gi >= 0; v[m] = g10[key][gi[m]]
        prods[label] = v
    for v in prods.values():
        v[v == -9999] = np.nan

    # ── the sealed split ─────────────────────────────────────────────────
    east = tr.c + (cc + 0.5) * tr.a
    north = tr.f + (rr + 0.5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    blocks = np.unique(block)
    rng = np.random.default_rng(SEED)
    hold = rng.choice(blocks, max(2, int(round(HOLDOUT_FRACTION * len(blocks)))), replace=False)
    dev = ~np.isin(block, hold)
    y = gnss[dev]
    print(f"{int(dev.sum())} development RTK pixels (reserved sealed)\n")

    def score(p, t):
        m = np.isfinite(p) & np.isfinite(t)
        if m.sum() < 20:
            return None
        pc, tc = p[m] - np.median(p[m]), t[m] - np.median(t[m])
        return {"rmse_m": float(np.sqrt(np.mean((pc - tc) ** 2))),
                "slope": float(np.polyfit(tc, pc, 1)[0]), "n": int(m.sum())}

    rows = {k: score(v[dev], y) for k, v in prods.items()}
    print(f"{'DEVELOPMENT SPLIT':48s} {'RMSE':>7s} {'slope':>7s} {'n':>5s}")
    for k, r in rows.items():
        print(f"{k:48s} " + (f"{r['rmse_m']:7.3f} {r['slope']:7.3f} {r['n']:5d}" if r else "      —       —"))

    common = np.isfinite(y)
    for v in prods.values():
        common &= np.isfinite(v[dev])
    print(f"\nCOMMON SUBSET ({int(common.sum())} px)")
    crow = {}
    for k, v in prods.items():
        p = v[dev][common]
        pc, tc = p - np.median(p), y[common] - np.median(y[common])
        crow[k] = {"rmse_m": float(np.sqrt(np.mean((pc - tc) ** 2))),
                   "slope": float(np.polyfit(tc, pc, 1)[0])}
        print(f"{k:48s} {crow[k]['rmse_m']:7.3f} {crow[k]['slope']:7.3f}")

    os.makedirs("results", exist_ok=True)
    json.dump({"epoch": EPOCH, "dev_pixels": int(dev.sum()), "rows": rows,
               "common": {"n": int(common.sum()), "rows": crow}},
              open("results/c5_validation_matrix.json", "w"), indent=1)
    print("\n-> results/c5_validation_matrix.json")


if __name__ == "__main__":
    main()
