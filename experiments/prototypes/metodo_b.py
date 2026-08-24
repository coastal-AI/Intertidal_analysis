"""Method B: a non-parametric tidal transfer, estimated from the archive and
judged out of sample.

The design question this answers: the intertidal fit needs the LOCAL water
level, not the channel depth. So skip the channel entirely — estimate, per
along-estuary band, how the mouth tide must be warped to explain that band's
radiometry best, and let hydraulics enter later only as a plausibility check
on the result.

The warp is two numbers per band, and the choice of which two is Gate 0's
verdict written into the model. A single shift already failed its gate: real
lag exists (triple sign asymmetry) but a shift moves flood and ebb together,
and the ponding evidence — ebb +41 min where flood is 0, the ponding index
predicting field error — says the two limbs behave differently. So each band
gets (tau_up, tau_down): the flood limb's delay and the ebb limb's delay,
separately. Where they come out equal, the wave simply arrives late; where
tau_down >> tau_up, that band ponds. Hysteresis is measured, not imposed.
No gain parameter anywhere: the affine theorem says imagery cannot see one.

Honesty structure, the part that matters:

  * scenes are split 65/35 IN TIME. The pair (tau_up, tau_down) is chosen on
    the TRAIN residual only;
  * the TEST window is opened once, for three variants: the chosen warp, the
    unwarped baseline, and the SIGN-FLIPPED warp. The gate is pre-registered:
    B must beat the baseline out of sample and the flipped control must not.
    A warp that helps with both signs is flexibility, not physics;
  * the pixel parameters (a, b, z, sigma) are refit per variant on TRAIN and
    FROZEN for the test evaluation, so the test measures prediction, not fit.

Sites: Villaviciosa (application, no truth) and the Scheldt (development —
its two gauges grade the inferred profile in Gate 2).
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
from scipy.special import erf

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 50
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
MIN_B = 0.15
CHUNK = 4000
N_BANDS = 6
TRAIN_FRAC = 0.65

SITES = {
    "villaviciosa": dict(
        npz="ndwi_intermareal.npz", epoch_min=2023,
        mouth=None,                       # site centroid, resolved below
        taus_up=(-10, 0, 10, 20, 30),
        taus_dn=(0, 20, 40, 60, 80, 100)),
    "escalda": dict(
        npz="escalda_intermareal.npz", epoch_min=0,
        mouth=(51.443, 3.597),
        taus_up=(-20, -10, -5, 0, 5, 10, 20),
        taus_dn=(-20, -10, -5, 0, 5, 10, 20)),
}


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def fit_and_eval(Ytr, Ctr, htr, Yte, Cte, hte):
    """Fit on train, freeze, score prediction residual per pixel on test."""
    grid = np.linspace(min(htr.min(), hte.min()),
                       max(htr.max(), hte.max()), MU_POINTS)
    P = Ytr.shape[1]
    sse = 0.0
    nob = 0.0
    resolved = 0
    for j in range(0, P, CHUNK):
        s = slice(j, j + CHUNK)
        a, b, z, sg, _, N = _fit_block(Ytr[:, s], Ctr[:, s], htr, grid,
                                       SG_GRID)
        # The closed-form (a, b) is unbounded, and a handful of pathological
        # pixels pass the b > 0.15 gate with amplitudes of tens. On TRAIN
        # residuals that stays hidden; on frozen-parameter TEST prediction it
        # explodes (first run: band RMSE of 18.7 in NDWI units, which is
        # bounded to [-1, 1]). Physical NDWI amplitudes are order one, so
        # gate on that and clip the prediction to the physical range.
        ok = ((N >= MIN_OBS) & (b > MIN_B) & np.isfinite(z)
              & (b < 2.5) & (np.abs(a) < 2.5))
        resolved += int(ok.sum())
        pred = np.clip(
            a[None, :] + b[None, :] * phi(
                (hte[:, None] - z[None, :]) / np.maximum(sg[None, :], 1e-3)),
            -2.0, 2.0)
        w = (Cte[:, s] > 0) & ok[None, :]
        r = np.where(w, Yte[:, s] - pred, 0.0)
        sse += float((r * r).sum())
        nob += float(w.sum())
    return np.sqrt(sse / max(nob, 1)), resolved


def train_score(Ytr, Ctr, htr):
    """The selection criterion: weighted fit residual on train only."""
    grid = np.linspace(htr.min(), htr.max(), MU_POINTS)
    P = Ytr.shape[1]
    sse = 0.0
    nob = 0.0
    for j in range(0, P, CHUNK):
        s = slice(j, j + CHUNK)
        a, b, z, sg, rm, N = _fit_block(Ytr[:, s], Ctr[:, s], htr, grid,
                                        SG_GRID)
        ok = ((N >= MIN_OBS) & (b > MIN_B) & (b < 2.5)
              & (np.abs(a) < 2.5))
        sse += float((rm[ok] ** 2 * N[ok]).sum())
        nob += float(N[ok].sum())
    return np.sqrt(sse / max(nob, 1))


def main():
    site = os.environ.get("SITE", "villaviciosa")
    cfg = SITES[site]
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, cfg["npz"]), allow_pickle=True)
    Y, C, dates = d["Y"], d["C"], np.array([str(s) for s in d["dates"]])
    if site == "villaviciosa":
        ch = np.load(os.path.join(SC, "canal.npz"))
        s_km = ch["s_keep"].astype(float) / 1000.0
        aoi = pit.sites.get("villaviciosa")
        lat_c, lon_c = aoi.centroid
        bbox = aoi.bbox
    else:
        s_km = d["s_km"].astype(float)
        lat_c, lon_c = cfg["mouth"]
        bbox = {"west": 3.55, "south": 51.33, "east": 3.85, "north": 51.46,
                "crs": "EPSG:4326"}

    times = pit.overpass.get_overpass_times(
        bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([(s in times)
                     and (int(s[:4]) >= cfg["epoch_min"]) for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    Y = np.nan_to_num(Y[have], nan=0.0).astype(np.float64)
    C = (C[have] > 0).astype(np.float64)
    print(f"[{site}] {Y.shape[0]} escenas x {Y.shape[1]:,} px", flush=True)

    def tide(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    # limb of each scene, and the tide at every candidate shift, once
    dh = (tide(t_real + pd.Timedelta(minutes=30))
          - tide(t_real - pd.Timedelta(minutes=30)))
    rising = dh > 0
    shifts = sorted(set(cfg["taus_up"]) | set(cfg["taus_dn"])
                    | {-t for t in cfg["taus_up"]}
                    | {-t for t in cfg["taus_dn"]} | {0})
    h_shift = {tv: tide(t_real - pd.Timedelta(minutes=float(tv)))
               for tv in shifts}
    print(f"  {int(rising.sum())} subiendo / {int((~rising).sum())} bajando · "
          f"{len(shifts)} desplazamientos precalculados", flush=True)

    def warped(tu, td):
        return np.where(rising, h_shift[tu], h_shift[td])

    order = np.argsort(t_real.values)
    n_tr = int(TRAIN_FRAC * len(order))
    tr_i = np.zeros(len(order), bool)
    tr_i[order[:n_tr]] = True
    te_i = ~tr_i
    print(f"  train {int(tr_i.sum())} escenas (hasta "
          f"{t_real[order[n_tr-1]].date()}) · test {int(te_i.sum())}",
          flush=True)

    fin = np.isfinite(s_km)
    qs = np.nanquantile(s_km, np.linspace(0, 1, N_BANDS + 1))
    qs[0] -= 1e-9

    out = {"site": site, "bands": []}
    print(f"\n  {'banda':14s} {'px':>7s} {'tau_up':>7s} {'tau_dn':>7s} "
          f"{'TEST base':>10s} {'TEST B':>8s} {'TEST -B':>8s} {'veredicto':>10s}",
          flush=True)
    print("  " + "-" * 78, flush=True)
    for a, b in zip(qs, qs[1:]):
        sel = np.where(fin & (s_km > a) & (s_km <= b))[0]
        if len(sel) < 300:
            continue
        Yb, Cb = Y[:, sel], C[:, sel]

        # selection on TRAIN only
        best = (np.inf, 0, 0)
        for tu in cfg["taus_up"]:
            for td in cfg["taus_dn"]:
                sc = train_score(Yb[tr_i], Cb[tr_i], warped(tu, td)[tr_i])
                if sc < best[0]:
                    best = (sc, tu, td)
        _, tu, td = best

        # the test window, opened once, three variants
        res = {}
        for lab, (u, v) in (("base", (0, 0)), ("B", (tu, td)),
                            ("-B", (-tu, -td))):
            h = warped(u, v)
            res[lab], _ = fit_and_eval(Yb[tr_i], Cb[tr_i], h[tr_i],
                                       Yb[te_i], Cb[te_i], h[te_i])
        gate = (res["B"] < res["base"]) and (res["-B"] >= res["base"])
        verdict = "PASA" if gate else (
            "flex" if res["-B"] < res["base"] else "no")
        print(f"  {f'{a:.2f}-{b:.2f}':14s} {len(sel):7,d} {tu:+7d} {td:+7d} "
              f"{res['base']:10.4f} {res['B']:8.4f} {res['-B']:8.4f} "
              f"{verdict:>10s}", flush=True)
        out["bands"].append(dict(band=[float(a), float(b)], n=len(sel),
                                 tau_up=tu, tau_dn=td, **res,
                                 gate=bool(gate)))

    n_pass = sum(1 for r in out["bands"] if r["gate"])
    print(f"\n  bandas que pasan la puerta: {n_pass} de {len(out['bands'])}")
    inner = [r for r in out["bands"][-2:]]
    if inner:
        gain = np.mean([r["base"] - r["B"] for r in inner])
        print(f"  ganancia media fuera de muestra en las bandas interiores: "
              f"{gain:+.4f} NDWI")
    json.dump(out, open(os.path.join(SC, f"metodo_b_{site}.json"), "w"),
              indent=1)
    print(f"  guardado metodo_b_{site}.json", flush=True)


if __name__ == "__main__":
    main()
