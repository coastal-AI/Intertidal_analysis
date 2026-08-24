"""Does the imagery-derived lag actually explain the imagery better?

The lag profile measured this afternoon — near zero at the mouth, +47 min at
1-2 km of channel, +74 min beyond — was estimated from ONE observable, the
rank order of flooded area. If it is real physics and not an artefact of that
observable, it must show up in a second, independent one: the per-pixel
sigmoid fit. A pixel whose water truly runs an hour behind the model should
have its NDWI explained better by h(t - tau) than by h(t). That is the
experiment, and it needs no ground truth of any kind.

Two design decisions, both forced by earlier findings:

* the lag profile is ADAPTIVE to the ria, not banded in fixed kilometres.
  Bands are quantiles of the ria's own channel-distance distribution, so a
  short ria resolves its whole profile just like a long one — fixed edges at
  1 and 2 km would declare a 900 m ria lag-free by construction. The band
  lags are then smoothed by isotonic regression, increasing inland, which is
  the only shape physics permits: a basin cannot lead its own mouth.

* the gate travels with a SIGN-FLIPPED control. The honest worry is that any
  perturbation of the tide axis might improve a flexible fit. It cannot —
  shifting the tide adds no parameters — but the cheap way to prove that is
  to apply the REVERSED profile, -tau(s): if improvements were flexibility,
  minus tau would improve too; if the lag is physics, minus tau must hurt
  roughly as much as plus tau helps.

Scoring: per-pixel weighted fit residual, medianed per band, on the pixels
resolved under every variant, so the comparison is paired. The survey is read
once at the end only to confirm what the geometry already implies: the RTK
points sit in the low-lag bands, so their agreement should not move.
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
TAU_GRID = np.arange(-90, 91, 5)
N_BANDS = 6                       # quantile bands of the ria's own s
MIN_CLEAR = 0.30
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
MIN_B = 0.15
CHUNK = 4000
ROUND_MIN = 10.0                  # pixels sharing a rounded tau share a tide


def wet_fraction(Y, C):
    obs = C > 0
    wet = obs & (Y > 0)
    n = obs.sum(axis=1)
    return (np.where(n > 0, wet.sum(axis=1) / np.maximum(n, 1), np.nan),
            n / Y.shape[1])


def lag_for(frac, h_by_tau):
    rho = np.array([stats.spearmanr(frac, h).statistic for h in h_by_tau])
    j = int(np.nanargmax(rho))
    if 0 < j < len(TAU_GRID) - 1:
        y0, y1, y2 = rho[j - 1], rho[j], rho[j + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        return float(TAU_GRID[j] + np.clip(off, -1, 1) * 5.0)
    return float(TAU_GRID[j])


def tides_at(lon, lat, t):
    from eo_tides.model import model_tides
    return model_tides(x=[lon], y=[lat], time=t, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)


def fit_variant(Y, C, t_real, tau_px, lon, lat):
    """Fit every pixel with its own lagged tide, grouped by rounded lag."""
    tau_r = np.round(tau_px / ROUND_MIN) * ROUND_MIN
    P = Y.shape[1]
    z = np.full(P, np.nan)
    b = np.zeros(P)
    rm = np.full(P, np.nan)
    N = np.zeros(P)
    for tv in np.unique(tau_r):
        sel = np.where(tau_r == tv)[0]
        h = tides_at(lon, lat, t_real - pd.Timedelta(minutes=float(tv)))
        grid = np.linspace(h.min(), h.max(), MU_POINTS)
        for j in range(0, len(sel), CHUNK):
            k = sel[j:j + CHUNK]
            aa, bb, mu, sg, rr, nn = _fit_block(Y[:, k], C[:, k], h,
                                                grid, SG_GRID)
            z[k], b[k], rm[k], N[k] = mu, bb, rr, nn
    ok = (N >= MIN_OBS) & (b > MIN_B) & np.isfinite(z)
    return z, rm, ok


def main():
    use_system_certificates()
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    dates = np.array([str(s) for s in dates])
    ch = np.load(os.path.join(SC, "canal.npz"))
    s_km = ch["s_keep"].astype(float) / 1000.0
    s_fill = np.where(np.isfinite(s_km), s_km, 0.0)

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_all = pd.DatetimeIndex(
        pd.to_datetime([times[s] for s in dates[have]])).tz_localize(None)
    Yh, Ch = Y[have], C[have]

    # ── 1. adaptive lag profile from flooded-area ranks ──────────────────
    frac, cover = wet_fraction(np.nan_to_num(Yh, nan=-9.0), Ch)
    ok_sc = (cover >= MIN_CLEAR) & np.isfinite(frac)
    h_by_tau = np.array([
        tides_at(lon_c, lat_c,
                 t_all[ok_sc] - pd.Timedelta(minutes=float(tv)))
        for tv in TAU_GRID])

    qs = np.nanquantile(s_km, np.linspace(0, 1, N_BANDS + 1))
    qs[0] -= 1e-9
    centers, taus, ns = [], [], []
    print("perfil adaptativo: bandas por CUANTILES de la propia ria")
    print(f"  {'banda (km)':16s} {'px':>7s} {'tau (min)':>10s}")
    print(f"  {'-'*38}")
    for a, b in zip(qs, qs[1:]):
        sel = (s_km > a) & (s_km <= b)
        if sel.sum() < 300:
            continue
        fb, cb = wet_fraction(np.nan_to_num(Yh[:, sel], nan=-9.0),
                              Ch[:, sel])
        # h_by_tau columns are the ok_sc scenes; align fb to those and drop
        # the ones this band never saw
        fb_sc = fb[ok_sc]
        mm = np.isfinite(fb_sc)
        if mm.sum() < 100:
            continue
        tb = lag_for(fb_sc[mm], h_by_tau[:, mm])
        centers.append(float(np.median(s_km[sel])))
        taus.append(tb)
        ns.append(int(sel.sum()))
        print(f"  {f'{a:.2f}-{b:.2f}':16s} {int(sel.sum()):7d} {tb:+10.1f}")

    # isotonic: increasing inland, floored at zero
    from sklearn.isotonic import IsotonicRegression
    # out_of_bounds="clip": the default returns NaN beyond the outermost
    # band centre, which silently dropped every pixel past 1.6 km — the
    # very pixels the lag matters for.
    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=90.0,
                             out_of_bounds="clip")
    w = np.sqrt(ns)
    tau_iso = iso.fit(centers, taus, sample_weight=w)
    tau_px = tau_iso.predict(s_fill)
    print(f"\n  perfil isotonico en los centros: "
          f"{np.round(tau_iso.predict(centers), 1)}")
    print(f"  tau por pixel: mediana {np.median(tau_px):.1f} min, "
          f"max {tau_px.max():.1f} min")

    # ── 2. the experiment: refit with +tau, 0, -tau ──────────────────────
    yrs = np.array([int(s[:4]) for s in dates[have]])
    ep = yrs >= 2023
    Ye = np.nan_to_num(Yh[ep], nan=0.0).astype(np.float64)
    Ce = Ch[ep].astype(np.float64)
    t_ep = t_all[ep]
    print(f"\nrefit 2023-2025: {int(ep.sum())} escenas x {Ye.shape[1]:,} px, "
          f"tres variantes", flush=True)

    variants = {}
    for name, tp in (("sin retardo", np.zeros_like(tau_px)),
                     ("retardo +tau(s)", tau_px),
                     ("control -tau(s)", -tau_px)):
        z, rm, okp = fit_variant(Ye, Ce, t_ep, tp, lon_c, lat_c)
        variants[name] = (z, rm, okp)
        print(f"  {name:18s} px resueltos {int(okp.sum()):,}", flush=True)

    common = np.logical_and.reduce([v[2] for v in variants.values()])
    print(f"\nRESIDUO DEL AJUSTE por banda (mediana, px comunes = "
          f"{int(common.sum()):,})")
    hdr = "".join(f"{k:>18s}" for k in variants)
    print(f"  {'banda (km)':16s}{hdr}")
    print(f"  {'-'*(16+18*len(variants))}")
    rows = []
    for a, b in zip(qs, qs[1:]):
        sel = common & (s_km > a) & (s_km <= b)
        if sel.sum() < 200:
            continue
        vals = {k: float(np.median(v[1][sel])) for k, v in variants.items()}
        cells = "".join(f"{vals[k]:18.4f}" for k in variants)
        print(f"  {f'{a:.2f}-{b:.2f}':16s}{cells}")
        rows.append({"band": [float(a), float(b)], "n": int(sel.sum()),
                     **vals})

    base = "sin retardo"
    plus = "retardo +tau(s)"
    minus = "control -tau(s)"
    inner = [r for r in rows if r["band"][0] >= np.quantile(s_fill, 0.5)]
    d_plus = np.median([r[base] - r[plus] for r in inner]) if inner else np.nan
    d_minus = np.median([r[base] - r[minus] for r in inner]) if inner else np.nan
    print(f"\n  en las bandas interiores: +tau mejora el residuo en "
          f"{d_plus:+.4f}, -tau lo cambia en {d_minus:+.4f}")
    if d_plus > 0 and d_minus < 0.5 * d_plus:
        print("  VEREDICTO: el retardo es fisica — mejora con el signo")
        print("  correcto y no con el contrario.")
    elif d_plus <= 0:
        print("  VEREDICTO: el retardo NO mejora el ajuste por pixel; el")
        print("  area lo ve pero el sigmoide no. Se queda como diagnostico.")
    else:
        print("  VEREDICTO AMBIGUO: ambos signos ayudan, luego era")
        print("  flexibilidad y no fisica. No aplicar.")

    # ── 3. the survey, read once: should be neutral where RTK sits ───────
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    hv = fi >= 0
    fi, yv = fi[hv], gnss[hv]
    print(f"\nGNSS (esta en s = 0.14-0.89 km, retardo ~"
          f"{tau_iso.predict([0.5])[0]:.0f} min): deberia no moverse")
    for name in (base, plus):
        z = variants[name][0]
        v = np.where(variants[name][2], z, np.nan)[fi]
        m = np.isfinite(v) & np.isfinite(yv)
        r = yv[m] - v[m]
        r = r - np.median(r)
        sl = float(np.polyfit(yv[m], v[m], 1)[0])
        print(f"  {name:18s} RMSE {np.sqrt(np.mean(r**2)):.3f} · "
              f"pendiente {sl:.3f} · n {int(m.sum())}")

    json.dump({"centers_km": centers, "tau_band": taus,
               "tau_iso": tau_iso.predict(centers).tolist(),
               "bands": rows,
               "d_plus": float(d_plus), "d_minus": float(d_minus)},
              open(os.path.join(SC, "retardo_adaptativo.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
