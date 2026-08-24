"""Can noise alone manufacture a compression FIELD? The master null.

The finding this has to survive: the slope of fitted against surveyed
elevation is not one number but a field. Across nine blocks of 150 m with
eight or more points it runs from -0.019 to 0.979, and a heterogeneity test
says Q = 54.5 on 8 degrees of freedom, p < 0.0001, I2 = 85 %. Read naively
that means 85 % of the spread between blocks is real structure rather than
sampling noise, and a real structure demands a physical explanation.

The catch is that I2 is only as trustworthy as the standard errors underneath
it, and those come from an ordinary least-squares formula that assumes the
residuals of z_hat are independent and identically distributed. They are
neither. Elevation errors are spatially correlated, heteroscedastic in the
tidal range, and driven by scene-level systematics shared across pixels. Every
one of those makes the OLS standard error too small, and every I2 built on a
standard error that is too small is too big.

An earlier version of this project's reasoning proposed regression dilution as
the culprit. That diagnosis does not apply here: attenuation biases a slope
when the noise sits in the PREDICTOR, and the predictor is the RTK survey at
1.3 cm while the noise sits in the response. What remains is the more mundane
possibility that the per-block slopes are simply far noisier than their
nominal standard errors admit.

So this simulates the null directly rather than arguing about it. Each field
pixel is planted at ITS OWN surveyed elevation, given ITS OWN fitted a, b and
sigma, ITS OWN residual noise and ITS OWN real cloud mask, and the water level
series is the real one. Physical compression is exactly zero by construction:
the planted truth IS the survey. Anything the pipeline then reports —
compression below one, a spread of slopes between blocks, a large I2 — was
manufactured by the estimator and the sampling, not by the ground.

Two numbers come out, and they answer different questions:

    mean slope    is the estimator biased at all, with no physics to blame?
    I2            can noise alone fabricate the observed 85 %?
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy import stats
from scipy.special import erf

from pyintertidal.elevation import _fit_block

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
BLOCK_M = 150.0
MIN_PER_BLOCK = 8
N_REP = 200


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def block_slopes(y, v, blk, min_n=MIN_PER_BLOCK):
    """Per-block slope of estimate on truth, with its OLS standard error."""
    out = []
    for b in np.unique(blk):
        s = blk == b
        yy, vv = y[s], v[s]
        m = np.isfinite(yy) & np.isfinite(vv)
        if m.sum() < min_n:
            continue
        yy, vv = yy[m], vv[m]
        sxx = np.sum((yy - yy.mean()) ** 2)
        if sxx <= 0:
            continue
        sl, ic = np.polyfit(yy, vv, 1)
        res = vv - (sl * yy + ic)
        n = len(yy)
        se = np.sqrt(np.sum(res ** 2) / max(n - 2, 1) / sxx)
        out.append((int(b), n, float(sl), float(max(se, 1e-6))))
    return out


def heterogeneity(rows):
    """Cochran Q and I2: how much of the spread survives the sampling noise."""
    if len(rows) < 3:
        return np.nan, np.nan, np.nan
    sl = np.array([r[2] for r in rows])
    se = np.array([r[3] for r in rows])
    w = 1.0 / se ** 2
    mu = float(np.sum(w * sl) / np.sum(w))
    Q = float(np.sum(w * (sl - mu) ** 2))
    df = len(sl) - 1
    return mu, Q, float(max(0.0, 100.0 * (Q - df) / Q)) if Q > 0 else 0.0


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])
    tide_all = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]
    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide_all)
    tide = tide_all[ep]
    Yr = np.nan_to_num(Y[ep], nan=0.0).astype(np.float64)
    Cr = C[ep].astype(np.float64)

    B = np.load(os.path.join(SC, "lamina_base.npz"))
    a, b, z, sg, nobs, good = (B["a"], B["b"], B["z"], B["sg"], B["nobs"],
                               B["good"])

    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]
        tr = s.transform

    zf = np.where(good, z, np.nan)[fi]
    m = np.isfinite(zf) & np.isfinite(y) & np.isfinite(hsr)
    fi, y, zf, hsr, rr, cc = fi[m], y[m], zf[m], hsr[m], rr[m], cc[m]

    east = tr.c + (cc + .5) * tr.a
    north = tr.f + (rr + .5) * tr.e
    blk = (((east - east.min()) // BLOCK_M).astype(int) * 1000
           + ((north - north.min()) // BLOCK_M).astype(int))

    # ── the observed field ───────────────────────────────────────────────
    obs_rows = block_slopes(y, zf, blk)
    mu_o, Q_o, I2_o = heterogeneity(obs_rows)
    print(f"OBSERVADO — {len(obs_rows)} bloques con {MIN_PER_BLOCK}+ puntos")
    print(f"  pendiente media ponderada {mu_o:.3f} · Q {Q_o:.1f} · "
          f"I2 {I2_o:.0f} %")
    print(f"  rango de pendientes {min(r[2] for r in obs_rows):+.3f} a "
          f"{max(r[2] for r in obs_rows):+.3f}\n")

    # ── the null: plant each pixel at its own surveyed elevation ─────────
    off = float(np.median(y - zf))          # survey -> tide-model datum
    z_true = y - off
    af, bf, sgf, Cf = a[fi], b[fi], sg[fi], Cr[:, fi]
    pred_real = af[None, :] + bf[None, :] * phi(
        (tide[:, None] - z[fi][None, :]) / np.maximum(sgf[None, :], 1e-3))
    w = Cf > 0
    noise = np.sqrt(np.where(w, (Yr[:, fi] - pred_real) ** 2, 0.0).sum(0)
                    / np.maximum(w.sum(0), 1))
    scene_off = np.nanmedian(np.where(w, Yr[:, fi] - pred_real, np.nan), 1)
    sd_scene = float(np.nanstd(scene_off))
    print(f"NULO — {len(fi)} px plantados en su propia cota RTK, "
          f"compresion fisica CERO")
    print(f"  ruido por pixel mediana {np.median(noise):.4f} · "
          f"sistematico de escena sd {sd_scene:.4f} NDWI")
    print(f"  se conservan la marea real, la mascara de nubes real y los "
          f"(a, b, sigma) reales\n")

    grid = np.linspace(tide.min(), tide.max(), MU_POINTS)
    clean = af[None, :] + bf[None, :] * phi(
        (tide[:, None] - z_true[None, :]) / np.maximum(sgf[None, :], 1e-3))

    mus, I2s, spans, lows = [], [], [], []
    for k in range(N_REP):
        rng = np.random.default_rng(1000 + k)
        Ysim = (clean
                + rng.normal(0.0, 1.0, clean.shape) * noise[None, :]
                + rng.normal(0.0, sd_scene, (clean.shape[0], 1)))
        aa, bb, mu_hat, ss, _, N = _fit_block(Ysim, Cf, tide, grid, SG_GRID)
        ok = (N >= MIN_OBS) & (bb > 0.15)
        v = np.where(ok, mu_hat, np.nan)
        rows = block_slopes(y, v, blk)
        if len(rows) < 3:
            continue
        mu_n, Q_n, I2_n = heterogeneity(rows)
        mus.append(mu_n)
        I2s.append(I2_n)
        spans.append(max(r[2] for r in rows) - min(r[2] for r in rows))
        lows.append(min(r[2] for r in rows))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{N_REP} replicas", flush=True)

    mus, I2s, spans = np.array(mus), np.array(I2s), np.array(spans)
    print(f"\n{len(mus)} replicas utiles")
    print(f"\n{'':28s} {'observado':>10s} {'nulo medio':>11s} "
          f"{'nulo p95':>9s} {'nulo max':>9s}")
    print("-" * 72)
    print(f"{'pendiente media':28s} {mu_o:10.3f} {mus.mean():11.3f} "
          f"{np.percentile(mus,5):9.3f} {mus.min():9.3f}")
    print(f"{'I2 (% de varianza real)':28s} {I2_o:10.0f} {I2s.mean():11.0f} "
          f"{np.percentile(I2s,95):9.0f} {I2s.max():9.0f}")
    print(f"{'recorrido de pendientes':28s} "
          f"{max(r[2] for r in obs_rows)-min(r[2] for r in obs_rows):10.3f} "
          f"{spans.mean():11.3f} {np.percentile(spans,95):9.3f} "
          f"{spans.max():9.3f}")

    p_i2 = float((I2s >= I2_o).mean())
    p_mu = float((mus <= mu_o).mean())
    print(f"\np(I2 del nulo alcanza el observado)        = {p_i2:.3f}")
    print(f"p(pendiente del nulo baja tanto como la real) = {p_mu:.3f}")

    print()
    if p_i2 > 0.05:
        print("PUERTA NO SUPERADA: el ruido solo ya fabrica esta estructura.")
        print("El campo c(x) es en buena parte estadistico, y la busqueda de")
        print("un mecanismo fisico se reordena hacia el modelo de error.")
    elif p_mu > 0.05:
        print("MATIZ IMPORTANTE: la estructura es real, pero el NIVEL medio de")
        print("compresion tambien lo produce el estimador sin fisica ninguna.")
        print("Hay que explicar la estructura, no la media.")
    else:
        print("PUERTA SUPERADA: ni la estructura ni el nivel de compresion se")
        print("pueden fabricar con ruido. Hay campo fisico c(x) que explicar.")

    json.dump({"observed": {"mu": mu_o, "Q": Q_o, "I2": I2_o,
                            "rows": obs_rows},
               "null": {"mu_mean": float(mus.mean()),
                        "mu_p05": float(np.percentile(mus, 5)),
                        "I2_mean": float(I2s.mean()),
                        "I2_p95": float(np.percentile(I2s, 95)),
                        "I2_max": float(I2s.max()),
                        "n_rep": int(len(mus))},
               "p_I2": p_i2, "p_mu": p_mu},
              open(os.path.join(SC, "nulo_bloques.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
