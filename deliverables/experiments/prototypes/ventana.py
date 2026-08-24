"""If censoring is the cause, widening the tide window must reduce it.

The censoring account says the relief compresses because pixels near the top
of the archive's tide range have nothing above them to constrain their
transition: the fit sees only the dry tail and settles low. The coefficient
measured on the survey is -0.189 m of underestimate per metre of missing
headroom, at t = -5.3, and it survives controlling for how often the pixel
was observed.

That account makes a prediction which costs one extra fit to test, and which
comes from a completely different direction than the survey subsetting that
produced it:

    raise the highest tide in the archive, and the compression must fall

The three-year epoch reaches +1.32 m. The ten-year archive is in the same
file and reaches higher, simply because more overpasses give more chances of
catching a high spring tide. So refit on ten years and compare.

The prediction is quantitative, not just directional. Every surveyed point
gains the same extra headroom, so from the fitted coefficient the mean bias
should shrink by 0.189 * (new max - old max) metres.

This is not a free win: ten years is a WORSE archive in other respects, as
established earlier — noisier, and it resolves fewer pixels (31 487 against
33 313) because more of them changed over the decade. If the compression
still falls despite that handicap, the censoring account is hard to avoid.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio

from pyintertidal.elevation import _fit_block

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
PIX_CHUNK = 4000


def fit(Y, C, tide):
    grid = np.linspace(tide.min(), tide.max(), MU_POINTS)
    P = Y.shape[1]
    z = np.full(P, np.nan); ok = np.zeros(P, bool)
    for j in range(0, P, PIX_CHUNK):
        s = slice(j, j + PIX_CHUNK)
        a, b, mu, sg, _, N = _fit_block(Y[:, s], C[:, s], tide, grid, SG_GRID)
        z[s] = mu
        ok[s] = (N >= MIN_OBS) & (b > 0.15)
    return np.where(ok, z, np.nan)


def score(y, v, label, hi):
    m = np.isfinite(y) & np.isfinite(v)
    r = y[m] - v[m]
    off = np.median(r)
    sl = float(np.polyfit(y[m], v[m], 1)[0])
    rms = float(np.sqrt(np.mean((r - off) ** 2)))
    head = hi - (y[m] - off)
    print(f"{label:26s} {sl:7.3f} {rms:7.3f} {int(m.sum()):5d} "
          f"{np.median(head):9.2f}")
    return {"slope": sl, "rmse": rms, "n": int(m.sum()),
            "median_headroom": float(np.median(head))}


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])
    tide_all = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]

    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y = fi[have], gnss[have]

    Yf = np.nan_to_num(Y[:, fi], nan=0.0).astype(np.float64)
    Cf = C[:, fi].astype(np.float64)
    yrs = np.array([int(s[:4]) for s in dates])
    fin = np.isfinite(tide_all)

    print(f"{'epoca':26s} {'pend':>7s} {'RMSE':>7s} {'n':>5s} "
          f"{'holgura':>9s}")
    print("-" * 60)
    out = {}
    epochs = [("2023-2025 (3 anos)", (yrs >= 2023) & fin),
              ("2016-2025 (10 anos)", fin)]
    tops = {}
    for label, sel in epochs:
        t = tide_all[sel]
        tops[label] = float(t.max())
        v = fit(Yf[sel], Cf[sel], t)
        out[label] = score(y, v, label, t.max())
        out[label]["n_scenes"] = int(sel.sum())
        out[label]["tide_max"] = float(t.max())
        out[label]["tide_min"] = float(t.min())

    a, b = [e[0] for e in epochs]
    gain = tops[b] - tops[a]
    print(f"\nmarea maxima: {tops[a]:+.2f} m con 3 anos, "
          f"{tops[b]:+.2f} m con 10  ->  {gain:+.2f} m mas de techo")
    print(f"escenas: {out[a]['n_scenes']} contra {out[b]['n_scenes']}")
    pred = 0.189 * gain
    print(f"\nsesgo que el modelo de censura predice recuperar: "
          f"{pred:+.3f} m")
    print(f"cambio de pendiente observado: "
          f"{out[b]['slope'] - out[a]['slope']:+.3f}")
    if gain <= 0.02:
        print("\nLos diez anos apenas amplian el techo, asi que esta prueba")
        print("no discrimina: el resultado no dice nada en ningun sentido.")
    elif out[b]["slope"] > out[a]["slope"]:
        print("\nLA PREDICCION SE CUMPLE: mas techo, menos compresion, y eso")
        print("pese a que el archivo de diez anos es peor en todo lo demas.")
    else:
        print("\nLA PREDICCION FALLA: mas techo no reduce la compresion.")

    json.dump(out, open(os.path.join(SC, "ventana.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
