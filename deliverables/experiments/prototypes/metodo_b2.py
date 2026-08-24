"""Method B, faithful to plan v3: amplitude AND phase per band, judged by the
plan's own out-of-sample likelihood.

Plan v3 estimates the transfer non-parametrically — A_k(s) and phi_k(s) from
the archive — and hands every verdict to internal judges, the first of them
out-of-sample likelihood on held-out scenes. This version implements that
minimally per band: a GAIN g on the tidal oscillation plus a lag per limb,

    h_band(t) = g * h_mouth(t - tau_limb)

Whether the gain is estimable from imagery is deliberately NOT decided in
advance here: it is handed to the judge. The run reports, per band, the
held-out prediction residual of four nested variants:

    base        g = 1, tau = 0        (assume inside equals outside)
    solo fase   g = 1, best taus      (the previous run, kept as ablation)
    completo    best g, best taus     (the v3 estimator)
    control     mirrored g, minus taus (the honesty check: a warp that helps
                mirrored cannot be physics)

If "completo" beats "solo fase" out of sample, the amplitude was estimable
and the plan's bet pays; if the two tie, the judge has spoken on gains too —
either way the answer comes from held-out data, which is what v3 demands.

Selection happens on TRAIN only; the pixel parameters are refit per variant
on TRAIN and frozen for the TEST evaluation; the TEST window is opened once.
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
GAINS = (0.85, 0.925, 1.0, 1.075, 1.15)

SITES = {
    "villaviciosa": dict(npz="ndwi_intermareal.npz", epoch_min=2023,
                         mouth=None,
                         taus_up=(0, 15, 30), taus_dn=(0, 20, 40, 60)),
    "escalda": dict(npz="escalda_intermareal.npz", epoch_min=0,
                    mouth=(51.443, 3.597),
                    taus_up=(-15, 0, 15), taus_dn=(-15, 0, 15)),
}


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def fit_and_eval(Ytr, Ctr, htr, Yte, Cte, hte):
    grid = np.linspace(min(htr.min(), hte.min()),
                       max(htr.max(), hte.max()), MU_POINTS)
    P = Ytr.shape[1]
    sse = nob = 0.0
    for j in range(0, P, CHUNK):
        s = slice(j, j + CHUNK)
        a, b, z, sg, _, N = _fit_block(Ytr[:, s], Ctr[:, s], htr, grid,
                                       SG_GRID)
        ok = ((N >= MIN_OBS) & (b > MIN_B) & np.isfinite(z)
              & (b < 2.5) & (np.abs(a) < 2.5))
        pred = np.clip(a[None, :] + b[None, :] * phi(
            (hte[:, None] - z[None, :]) / np.maximum(sg[None, :], 1e-3)),
            -2.0, 2.0)
        w = (Cte[:, s] > 0) & ok[None, :]
        r = np.where(w, Yte[:, s] - pred, 0.0)
        sse += float((r * r).sum())
        nob += float(w.sum())
    return np.sqrt(sse / max(nob, 1))


def train_score(Ytr, Ctr, htr):
    grid = np.linspace(htr.min(), htr.max(), MU_POINTS)
    P = Ytr.shape[1]
    sse = nob = 0.0
    for j in range(0, P, CHUNK):
        s = slice(j, j + CHUNK)
        a, b, z, sg, rm, N = _fit_block(Ytr[:, s], Ctr[:, s], htr, grid,
                                        SG_GRID)
        ok = ((N >= MIN_OBS) & (b > MIN_B) & (b < 2.5) & (np.abs(a) < 2.5))
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
    have = np.array([(s in times) and (int(s[:4]) >= cfg["epoch_min"])
                     for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    # float32 on purpose: the Scheldt arrays in float64 are ~550 MB and the
    # solo-fase run was OOM-killed when two cube downloads landed on top of
    # them. _fit_block casts its own chunks to float64, so nothing is lost.
    Y = np.nan_to_num(Y[have], nan=0.0).astype(np.float32)
    C = (C[have] > 0).astype(np.float32)
    print(f"[{site}] {Y.shape[0]} escenas x {Y.shape[1]:,} px", flush=True)

    def tide(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    dh = (tide(t_real + pd.Timedelta(minutes=30))
          - tide(t_real - pd.Timedelta(minutes=30)))
    rising = dh > 0
    shifts = sorted(set(cfg["taus_up"]) | set(cfg["taus_dn"])
                    | {-t for t in cfg["taus_up"]}
                    | {-t for t in cfg["taus_dn"]} | {0})
    h_shift = {tv: tide(t_real - pd.Timedelta(minutes=float(tv)))
               for tv in shifts}
    print(f"  {int(rising.sum())} subiendo / {int((~rising).sum())} bajando",
          flush=True)

    def warped(g, tu, td):
        return g * np.where(rising, h_shift[tu], h_shift[td])

    order = np.argsort(t_real.values)
    n_tr = int(TRAIN_FRAC * len(order))
    tr_i = np.zeros(len(order), bool)
    tr_i[order[:n_tr]] = True
    te_i = ~tr_i
    print(f"  train {int(tr_i.sum())} / test {int(te_i.sum())}", flush=True)

    fin = np.isfinite(s_km)
    qs = np.nanquantile(s_km, np.linspace(0, 1, N_BANDS + 1))
    qs[0] -= 1e-9

    out = {"site": site, "bands": []}
    print(f"\n  {'banda':14s} {'g':>6s} {'t_up':>5s} {'t_dn':>5s} "
          f"{'base':>8s} {'fase':>8s} {'COMPLETO':>9s} {'control':>8s} "
          f"{'veredicto':>16s}", flush=True)
    print("  " + "-" * 86, flush=True)
    for a, b in zip(qs, qs[1:]):
        sel = np.where(fin & (s_km > a) & (s_km <= b))[0]
        if len(sel) < 300:
            continue
        Yb, Cb = Y[:, sel], C[:, sel]

        # selection on TRAIN: first taus at g=1, then g given the taus, then
        # taus refined at the chosen g — a coordinate sweep, far cheaper than
        # the full grid and adequate for a 3-parameter search
        best_t = (np.inf, 0, 0)
        for tu in cfg["taus_up"]:
            for td in cfg["taus_dn"]:
                sc = train_score(Yb[tr_i], Cb[tr_i],
                                 warped(1.0, tu, td)[tr_i])
                if sc < best_t[0]:
                    best_t = (sc, tu, td)
        _, tu, td = best_t
        best_g = (np.inf, 1.0)
        for g in GAINS:
            sc = train_score(Yb[tr_i], Cb[tr_i], warped(g, tu, td)[tr_i])
            if sc < best_g[0]:
                best_g = (sc, g)
        _, g = best_g
        best_t2 = (np.inf, tu, td)
        for tu2 in cfg["taus_up"]:
            for td2 in cfg["taus_dn"]:
                sc = train_score(Yb[tr_i], Cb[tr_i],
                                 warped(g, tu2, td2)[tr_i])
                if sc < best_t2[0]:
                    best_t2 = (sc, tu2, td2)
        _, tu, td = best_t2

        res = {}
        variants = {"base": (1.0, 0, 0), "fase": (1.0, tu, td),
                    "completo": (g, tu, td),
                    "control": (2.0 - g, -tu, -td)}
        for lab, (gv, u, v) in variants.items():
            h = warped(gv, u, v)
            res[lab] = fit_and_eval(Yb[tr_i], Cb[tr_i], h[tr_i],
                                    Yb[te_i], Cb[te_i], h[te_i])
        gate_full = (res["completo"] < res["base"]
                     and res["control"] >= res["base"])
        amp_adds = res["completo"] < res["fase"] - 1e-6
        verdict = ("PASA+amplitud" if gate_full and amp_adds else
                   "PASA fase" if gate_full else
                   "flex" if res["control"] < res["base"] else "no")
        print(f"  {f'{a:.2f}-{b:.2f}':14s} {g:6.3f} {tu:+5d} {td:+5d} "
              f"{res['base']:8.4f} {res['fase']:8.4f} {res['completo']:9.4f} "
              f"{res['control']:8.4f} {verdict:>16s}", flush=True)
        out["bands"].append(dict(band=[float(a), float(b)], n=len(sel),
                                 g=g, tau_up=tu, tau_dn=td, **res,
                                 gate=bool(gate_full),
                                 amp_adds=bool(amp_adds)))

    n_pass = sum(1 for r in out["bands"] if r["gate"])
    n_amp = sum(1 for r in out["bands"] if r["gate"] and r["amp_adds"])
    print(f"\n  pasan la puerta: {n_pass} de {len(out['bands'])} · "
          f"la amplitud aporta en {n_amp}")
    json.dump(out, open(os.path.join(SC, f"metodo_b2_{site}.json"), "w"),
              indent=1)
    print(f"  guardado metodo_b2_{site}.json", flush=True)


if __name__ == "__main__":
    main()
