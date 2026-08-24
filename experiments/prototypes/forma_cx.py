"""What does the compression depend on? The functional form of c(x).

Established just now: the compression is physical. Planting pixels at their
own surveyed elevations with zero compression, the real tide, the real cloud
masks and the real per-pixel noise returns a mean slope of 1.019 across 200
replicates, and never once drops to the 0.561 observed. So something on the
ground, or in the optics, bends the estimate — and it is not sampling noise.

The structure is real too, though weaker than it first looked: the naive I2 of
86 % has to be read against a null that already manufactures 40 % on average
and 85 % at its worst, because ordinary least-squares standard errors
understate the true scatter of a per-block slope. The observed value clears
every one of 200 replicates, but only just.

Nine blocks cannot support a multivariate regression, so this works at pixel
level with the specification the problem actually implies. If the estimate
compresses by a factor c that varies in space,

    z_hat = a + c(x) * z_true      =>      z_hat - z_true = a + (c(x)-1)*z_true

so a spatially varying compression is an INTERACTION between the true
elevation and whatever c depends on. The main effect of z_true is the global
compression, and it is tautological — it restates the slope, exactly the trap
this project fell into once already with the tidal-headroom regression. The
INTERACTION terms are not tautological, and they are the answer.

Every covariate is available with no ground truth: channel distance, distance
to the sea, the fitted sigma and amplitude, the observation count and the wet
fraction. Whatever comes out is the shape any physical mechanism will have to
reproduce, and it gets sealed before any mechanism is allowed to see it.

The null is the same simulation as before — zero compression, real noise —
pushed through this identical regression, because with 192 correlated points
an ordinary p-value is not worth reading.
"""
import os
import sys
import json
import hashlib

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy.special import erf

from pyintertidal.elevation import _fit_block

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
N_REP = 200


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def design(z_true, cov):
    """[1, z, covariates, z x covariates] — the interactions are the target."""
    zc = z_true - z_true.mean()
    cols = [np.ones_like(zc), zc]
    names = ["constante", "cota (compresion global)"]
    for nm, v in cov:
        vs = (v - np.nanmean(v)) / max(np.nanstd(v), 1e-9)
        cols.append(vs)
        names.append(f"{nm}")
        cols.append(zc * vs)
        names.append(f"cota x {nm}")
    return np.column_stack(cols), names


def fit(X, r):
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    return coef


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])
    ch = np.load(os.path.join(SC, "canal.npz"))
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
    zf = np.where(good, z, np.nan)[fi]

    with rasterio.open("products_villaviciosa/water_frequency.tif") as s:
        wf = np.nan_to_num(s.read(1), nan=0.0)[rr, cc]

    # Terrain slope and roughness, from the product itself. These carry the
    # point-versus-pixel hypothesis: mu is the MEDIAN elevation of a 10 m
    # pixel while the RTK measures one point inside it, so on sloping or
    # broken ground the two disagree by construction, and the disagreement
    # grows with the local gradient. It is the most plausible mechanism left
    # once channel distance has been ruled out, and it costs nothing to test
    # because the DEM is already on disk.
    from scipy import ndimage
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        dem = s.read(1)
        px = abs(s.transform.a)
    dem_f = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    gy, gx = np.gradient(dem_f, px)
    slope = np.hypot(gx, gy)[rr, cc]
    rough = ndimage.generic_filter(dem_f, np.std, size=3)[rr, cc]

    s_km = ch["geo"][rr, cc] / 1000.0
    d_sea = ch["eucl"][rr, cc] / 1000.0
    wetf = np.where(Cr[:, fi] > 0, Yr[:, fi] > 0, 0).sum(0) / np.maximum(
        (Cr[:, fi] > 0).sum(0), 1)

    m = (np.isfinite(zf) & np.isfinite(y) & np.isfinite(s_km)
         & np.isfinite(d_sea) & np.isfinite(slope) & np.isfinite(rough))
    fi, y, zf, s_km, d_sea, wf, wetf, slope, rough = (
        fi[m], y[m], zf[m], s_km[m], d_sea[m], wf[m], wetf[m],
        slope[m], rough[m])
    off = float(np.median(y - zf))
    z_true = y - off

    cov = [("pendiente del terreno", slope),
           ("rugosidad local", rough),
           ("distancia por canal", s_km),
           ("distancia al mar", d_sea),
           ("sigma ajustada", sg[fi]),
           ("amplitud b", b[fi]),
           ("log n_obs", np.log(np.maximum(nobs[fi], 1))),
           ("frecuencia de agua", wf),
           ("fraccion mojada", wetf)]

    X, names = design(z_true, cov)
    resid = zf - z_true
    coef = fit(X, resid)
    print(f"{len(y)} puntos · {X.shape[1]} terminos")
    print(f"compresion global implicada: {1 + coef[1]:.3f}\n")

    # ── null: same regression on simulated data with zero compression ────
    af, bf, sgf, Cf = a[fi], b[fi], sg[fi], Cr[:, fi]
    pred_real = af[None, :] + bf[None, :] * phi(
        (tide[:, None] - z[fi][None, :]) / np.maximum(sgf[None, :], 1e-3))
    w = Cf > 0
    noise = np.sqrt(np.where(w, (Yr[:, fi] - pred_real) ** 2, 0.0).sum(0)
                    / np.maximum(w.sum(0), 1))
    sd_scene = float(np.nanstd(np.nanmedian(
        np.where(w, Yr[:, fi] - pred_real, np.nan), 1)))
    clean = af[None, :] + bf[None, :] * phi(
        (tide[:, None] - z_true[None, :]) / np.maximum(sgf[None, :], 1e-3))
    grid = np.linspace(tide.min(), tide.max(), MU_POINTS)

    null = []
    for k in range(N_REP):
        rng = np.random.default_rng(3000 + k)
        Ysim = (clean + rng.normal(0.0, 1.0, clean.shape) * noise[None, :]
                + rng.normal(0.0, sd_scene, (clean.shape[0], 1)))
        aa, bb, mu_hat, ss, _, N = _fit_block(Ysim, Cf, tide, grid, SG_GRID)
        ok = (N >= MIN_OBS) & (bb > 0.15)
        v = np.where(ok, mu_hat, np.nan)
        g = np.isfinite(v)
        if g.sum() < 60:
            continue
        null.append(fit(X[g], (v - z_true)[g]))
        if (k + 1) % 50 == 0:
            print(f"  nulo {k+1}/{N_REP}", flush=True)
    NU = np.array(null)

    print(f"\n{len(NU)} replicas nulas\n")
    print(f"{'termino':30s} {'coef':>8s} {'nulo sd':>9s} {'z':>7s} {'p':>7s}")
    print("-" * 66)
    rows = []
    for i, nm in enumerate(names):
        c0 = coef[i]
        nsd = NU[:, i].std(ddof=1)
        nmean = NU[:, i].mean()
        zsc = (c0 - nmean) / max(nsd, 1e-9)
        p = float((np.abs(NU[:, i] - nmean) >= abs(c0 - nmean)).mean())
        flag = "  <--" if p < 0.05 and "cota x" in nm else ""
        print(f"{nm:30s} {c0:+8.4f} {nsd:9.4f} {zsc:+7.2f} {p:7.3f}{flag}")
        rows.append({"term": nm, "coef": float(c0), "null_sd": float(nsd),
                     "z": float(zsc), "p": float(p)})

    print("\nSolo los terminos 'cota x ...' dicen algo nuevo: son la variacion")
    print("espacial de la compresion. El termino 'cota' solo es la compresion")
    print("global, que ya conociamos, y la constante es el datum.")

    inter = [r for r in rows if r["term"].startswith("cota x")
             and r["p"] < 0.05]
    print()
    if inter:
        print("FORMA DE c(x) — la compresion depende de:")
        for r in sorted(inter, key=lambda r: -abs(r["z"])):
            direc = "mas compresion" if r["coef"] < 0 else "menos compresion"
            print(f"  {r['term'][7:]:26s} z={r['z']:+.2f}  "
                  f"(al subir la covariable, {direc})")
        print("\nEsto es lo que cualquier mecanismo fisico tendra que")
        print("reproducir. Se sella antes de que ningun mecanismo lo vea.")
    else:
        print("NINGUNA covariable explica la variacion espacial. El campo")
        print("c(x) es real pero no lo capturan las covariables disponibles;")
        print("hace falta una que no tenemos (sustrato, vegetacion, pendiente).")

    payload = {"n": int(len(y)), "global_compression": float(1 + coef[1]),
               "terms": rows, "n_null": int(len(NU))}
    blob = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()[:16]
    print(f"\nsello del resultado: sha256 {payload['sha256']}")
    json.dump(payload, open(os.path.join(SC, "forma_cx.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
