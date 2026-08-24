"""The tidal flat as its own tide gauge: solve elevation and water level together.

Every intertidal method we know of, ours and DEA's alike, takes the tide as a
known input from a global model and solves for elevation. Today's work says
that input is the dominant error: the estuary is not the open ocean, and no
global model represents what happens inside it.

So stop treating it as known. In

    NDWI(p,t) = a(p) + b(p) * Phi( (h(t) - mu(p)) / sigma(p) )

let h(t) be unknown too. With 40,000 pixels and 465 scenes there is far more
data than unknowns, and the logic inverts cleanly: which pixels are wet in a
scene tells you how high the water stood, with forty thousand pixels voting.
The flat is a giant staff gauge and every image is a reading.

Solved by alternation — fit the pixels with h fixed (the ordinary fit), then
fit h with the pixels fixed (a one-dimensional problem per scene, done for
all scenes at once as two matrix products).

The affine degeneracy proved earlier is what makes this well posed rather
than what breaks it: the system is determined up to ONE global (scale,
datum) pair, which a single anchor fixes. Here the anchor is the harmonic
model's overall mean and spread, so recovered levels stay in its datum and
everything that survives is the part the model cannot express — surge, river
discharge, estuarine lag and amplification.

What would confirm it works, and none of it involves the RTK survey, which
is held back as an independent check:
  * the recovered levels should track the harmonic model closely but not
    exactly, and the departures should be physically ordered, not noise;
  * they should reproduce the spring/neap bias measured independently
    this afternoon (-0.25 m) without having been told about it;
  * elevations built on them should agree better with the field survey.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from scipy.special import erf

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
N_CAND = 240          # candidate water levels per scene
N_ITER = 8
PRIOR_SIGMA = 0.20   # metres we are willing to let the model be wrong by
MIN_PIX_PER_SCENE = 400
INVERT_PIX = 20000


def phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def solve_levels(A, Cc, SSy, a, b, mu, sg, cand,
                 h_prior=None, lam=0.0):
    """Water level of every scene at once, by exhaustive search.

    For each candidate level the predicted NDWI of every pixel is known, so
    the residual of scene t is a quadratic expanded into two matrix products
    over pixels — no loop over scenes, no optimiser.
    """
    z = (cand[:, None] - mu[None, :]) / np.maximum(sg[None, :], 1e-3)
    pred = (a[None, :] + b[None, :] * phi(z)).astype(np.float32)   # (K, P)
    loss = (SSy[:, None]
            - 2.0 * (A @ pred.T)
            + (Cc @ (pred * pred).T))
    if h_prior is not None and lam > 0:
        # The harmonic model is incomplete, not wrong. Left free, h(t)
        # absorbs whatever varies scene to scene — haze, glint, turbidity —
        # because that lowers the residual more cheaply than water does.
        # A Gaussian prior on the departure keeps h answering the question
        # it was asked.
        loss = loss + lam * (cand[None, :] - h_prior[:, None]) ** 2
    k = np.argmin(loss, axis=1)
    # Parabolic refinement between the neighbouring candidates.
    k1 = np.clip(k, 1, len(cand) - 2)
    rows = np.arange(loss.shape[0])
    y0, y1, y2 = (loss[rows, k1 - 1], loss[rows, k1], loss[rows, k1 + 1])
    den = (y0 - 2 * y1 + y2)
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    step = cand[1] - cand[0]
    return cand[k1] + np.clip(shift, -1, 1) * step, loss[rows, k]


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(d["shape"])

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide_model = model_tides(x=[lon_c], y=[lat_c], time=base, model="EOT20",
                             directory="tide_models", crs="EPSG:4326",
                             extrapolate=True, cutoff=np.inf,
                             parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide_model)
    Y, C = Y[ep], C[ep]
    h_model = tide_model[ep]
    stamps = base[ep]
    T = Y.shape[0]
    print(f"{T} escenas x {Y.shape[1]:,} px", flush=True)

    rng = np.random.default_rng(17)
    sub = (np.sort(rng.choice(Y.shape[1], INVERT_PIX, replace=False))
           if Y.shape[1] > INVERT_PIX else np.arange(Y.shape[1]))
    Ys = np.nan_to_num(Y[:, sub], nan=0.0).astype(np.float32)
    Cs = C[:, sub].astype(np.float32)
    A = (Ys * Cs)
    SSy = (Ys * A).sum(axis=1)
    per_scene = Cs.sum(axis=1)
    usable = per_scene >= MIN_PIX_PER_SCENE
    print(f"{usable.sum()}/{T} escenas con >= {MIN_PIX_PER_SCENE} px claros\n",
          flush=True)

    cand = np.linspace(np.nanmin(h_model) - 0.3, np.nanmax(h_model) + 0.3,
                       N_CAND)
    h = h_model.copy()
    t0 = time.time()
    print(f"{'iter':>4s} {'residuo':>10s} {'RMS vs modelo':>14s} "
          f"{'max dif':>8s}")
    print("-" * 42)
    for it in range(N_ITER):
        grid = np.linspace(np.nanmin(h), np.nanmax(h), MU_POINTS)
        a, b, mu, sg, rmse, N = _fit_block(
            Ys.astype(np.float64), Cs.astype(np.float64), h, grid, SG_GRID)
        good = (N >= 8) & (b > 0.15)
        if not good.any():
            print("sin pixeles utiles"); return
        # Scene-wide NDWI noise — haze, glint, atmospheric correction — moves
        # every pixel of an image together, and to first order that is
        # indistinguishable from the water having risen. It is why a free
        # h(t) drifts: absorbing the haze is cheaper than describing water.
        # Pixels far above or below the waterline in a given scene are
        # saturated and CANNOT respond to level, so whatever they shift by is
        # atmosphere. Measure it there and take it out.
        zz = (h[:, None] - mu[None, good]) / np.maximum(sg[None, good], 1e-3)
        blind = (np.abs(zz) > 2.5) & (Cs[:, good] > 0)
        expect = (a[None, good] + b[None, good] * phi(zz))
        num = np.where(blind, Ys[:, good] - expect, 0.0).sum(axis=1)
        den = blind.sum(axis=1)
        off = np.where(den >= 200, num / np.maximum(den, 1), 0.0)
        print(f"     offset de escena: mediana {np.median(np.abs(off)):.4f}, "
              f"p90 {np.percentile(np.abs(off), 90):.4f} NDWI  "
              f"({int((den >= 200).sum())} escenas calibradas)", flush=True)
        Ys_c = Ys - off[:, None].astype(np.float32)
        A_c = Ys_c * Cs
        SSy_c = (Ys_c * A_c).sum(axis=1)

        # lam converts the prior width into the units of the pixel
        # residual: how many NDWI-squared a metre of departure is worth.
        noise = float(np.nanmedian(rmse[good])) or 0.05
        lam = float(Cs[:, good].sum(axis=1).mean()) * noise ** 2             / PRIOR_SIGMA ** 2
        h_new, loss = solve_levels(A_c[:, good], Cs[:, good], SSy_c,
                                   a[good], b[good], mu[good], sg[good], cand,
                                   h_prior=h_model, lam=lam)
        # Scenes with too little clear ground keep the model's value: their
        # level is not observable and a free parameter would just absorb noise.
        h_new = np.where(usable, h_new, h_model)
        # No re-standardising: the prior already anchors datum and scale, and
        # rescaling a noisy recovery to the model's spread just re-inflates
        # the noise it was meant to suppress.
        diff = h_new - h_model
        print(f"{it:4d} {loss.sum():10.1f} {np.sqrt(np.mean(diff**2)):14.4f} "
              f"{np.abs(diff).max():8.3f}", flush=True)
        if np.max(np.abs(h_new - h)) < 0.002:
            h = h_new
            break
        h = h_new
    print(f"\nconvergido en {(time.time()-t0)/60:.1f} min\n", flush=True)

    # ── is the recovered series physically ordered, or noise? ────────────
    diff = h - h_model
    grid_t = pd.DatetimeIndex(np.concatenate(
        [(stamps + pd.Timedelta(hours=float(x))).values
         for x in np.arange(-18, 18.5, 0.5)])).unique().sort_values()
    df = model_tides(x=[lon_c], y=[lat_c], time=grid_t, model="EOT20",
                     directory="tide_models", crs="EPSG:4326",
                     extrapolate=True, cutoff=np.inf,
                     parallel=False).reset_index()
    lut = pd.Series(df["tide_height"].to_numpy(float),
                    index=pd.DatetimeIndex(df["time"])).groupby(level=0).first()
    Hm = np.vstack([lut.reindex(stamps + pd.Timedelta(hours=float(x))
                                ).to_numpy(float)
                    for x in np.arange(-18, 18.5, 0.5)])
    rango = np.nanmax(Hm, axis=0) - np.nanmin(Hm, axis=0)

    u = usable
    print("EL NIVEL RECUPERADO CONTRA EL MODELO ARMONICO")
    print(f"  RMS de la diferencia      {np.sqrt(np.mean(diff[u]**2)):.3f} m")
    print(f"  correlacion con el modelo {np.corrcoef(h[u], h_model[u])[0,1]:.4f}")
    med = np.median(rango[u])
    viva, muerta = u & (rango >= med), u & (rango < med)
    print(f"\n  desviacion en VIVAS   {diff[viva].mean():+.3f} m "
          f"({viva.sum()} escenas, rango medio {rango[viva].mean():.2f} m)")
    print(f"  desviacion en MUERTAS {diff[muerta].mean():+.3f} m "
          f"({muerta.sum()} escenas, rango medio {rango[muerta].mean():.2f} m)")
    print(f"  separacion vivas-muertas {diff[viva].mean()-diff[muerta].mean():+.3f} m")
    print("  (medido esta tarde por otra via: -0.25 m en la cota, que")
    print("   corresponde a nivel de agua MAS ALTO en vivas)")
    r_ = np.corrcoef(rango[u], diff[u])[0, 1]
    print(f"\n  correlacion desviacion vs rango de marea: {r_:+.3f}")

    # ── the independent check: elevations against the RTK survey ─────────
    print("\nCOTA CONTRA EL LEVANTAMIENTO GNSS (control independiente)")
    pos = {int(k): i for i, k in enumerate(keep)}
    idx = np.array([pos.get(int(f), -1) for f in field_flat])
    have = idx >= 0
    idx, G = idx[have], gnss[have]

    out = {}
    for lab, hh in [("marea del modelo", h_model), ("marea invertida", h)]:
        grid = np.linspace(np.nanmin(hh), np.nanmax(hh), MU_POINTS)
        a, b, mu, sg, rmse, N = _fit_block(
            np.nan_to_num(Y, nan=0.0).astype(np.float64),
            C.astype(np.float64), hh, grid, SG_GRID)
        v = np.where((N >= 8) & (b > 0.15), mu, np.nan)
        out[lab] = v
    ok = np.isfinite(out["marea del modelo"][idx]) & \
        np.isfinite(out["marea invertida"][idx])
    idx2, G2 = idx[ok], G[ok]
    n = len(idx2)
    print(f"{n} pixeles de campo comunes\n")
    print(f"{'ajuste':20s} {'RMSE':>7s} {'r':>7s} {'pendiente':>10s}")
    print("-" * 48)
    res = {}
    for lab in out:
        v = out[lab][idx2]
        dd = G2 - v
        dd = dd - np.median(dd)
        res[lab] = dd
        print(f"{lab:20s} {np.sqrt(np.mean(dd**2)):7.3f} "
              f"{np.corrcoef(G2, v)[0,1]:7.3f} "
              f"{np.polyfit(G2, v, 1)[0]:10.3f}")

    rb = np.random.default_rng(4).integers(0, n, size=(20000, n))
    a_ = res["marea del modelo"] ** 2
    b_ = res["marea invertida"] ** 2
    obs = np.sqrt(b_.mean()) - np.sqrt(a_.mean())
    boot = np.sqrt(b_[rb].mean(1)) - np.sqrt(a_[rb].mean(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\ncambio al invertir la marea: {obs:+.4f} m  "
          f"IC 95% [{lo:+.4f},{hi:+.4f}]")
    print("  " + ("SIGNIFICATIVO" if lo * hi > 0 else "no significativo"))

    json.dump({"h": h.tolist(), "h_model": h_model.tolist(),
               "fechas": [str(s)[:10] for s in stamps],
               "usable": u.tolist(),
               "delta_rmse": float(obs), "ic": [float(lo), float(hi)]},
              open(os.path.join(SC, "mareografo.json"), "w"))
    print("\nserie recuperada -> mareografo.json")


if __name__ == "__main__":
    main()
