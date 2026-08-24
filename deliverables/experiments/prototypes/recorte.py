"""Is the relief compression physics at all, or the search grid's edge?

This project has taken the compression — fitted relief only ~0.78 of the
surveyed relief, in HSR and in the published step method alike — as evidence
that the assumed water level is wrong. That reading has just lost its main
support: with the water-surface tilt estimated in a form that cannot absorb
noise, it comes out at 0.003 m/km, seven millimetres across the whole ria,
which predicts a compression of 0.999 rather than the 0.78 observed.

So the cause is elsewhere, and there is a much duller candidate sitting in
plain sight. The elevation of a pixel is found by searching a grid that runs
from the lowest tide in the archive to the highest. A pixel whose true
elevation lies outside that window cannot be assigned it — the best the
search can do is the nearest edge. Every such pixel is pulled inward, and
pulling the extremes inward is exactly what a slope below one looks like.

Nobody in this literature reports it, and it would apply to every published
method equally, which is precisely the observation that started this: HSR and
the step method compress by the same amount.

Three things decide it, all cheap:

  1. how many fitted elevations sit at or beside the edge of the grid
  2. whether the compression survives when only pixels comfortably INSIDE
     the window are scored — if it vanishes, the case is closed
  3. whether the surveyed elevations themselves fall outside the tide window,
     because if they do the effect is not a possibility but a certainty
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")


def slope_rmse(truth, est):
    m = np.isfinite(truth) & np.isfinite(est)
    if m.sum() < 12:
        return np.nan, np.nan, int(m.sum())
    r = truth[m] - est[m]
    r = r - np.median(r)
    return (float(np.polyfit(truth[m], est[m], 1)[0]),
            float(np.sqrt(np.mean(r ** 2))), int(m.sum()))


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    keep, field_flat, gnss = d["keep"], d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])
    tide = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]
    dates = d["dates"]
    yrs = np.array([int(s[:4]) for s in dates])
    tide = tide[(yrs >= 2023) & np.isfinite(tide)]
    lo, hi = float(tide.min()), float(tide.max())
    print(f"ventana de busqueda = rango de marea del archivo: "
          f"{lo:+.2f} a {hi:+.2f} m  (amplitud {hi-lo:.2f} m)\n")

    B = np.load(os.path.join(SC, "lamina_base.npz"))
    z, good = B["z"], B["good"]
    zg = z[good]
    step = (hi - lo) / 59.0     # MU_POINTS = 60
    at_edge = (zg <= lo + step) | (zg >= hi - step)
    print(f"cotas ajustadas: {len(zg):,} px")
    print(f"  pegadas al borde de la rejilla: {int(at_edge.sum()):,} "
          f"({100*at_edge.mean():.1f} %)")
    print(f"  en el borde BAJO {int((zg <= lo+step).sum()):,} · "
          f"en el ALTO {int((zg >= hi-step).sum()):,}")

    # ── the survey, and where it sits relative to the window ─────────────
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]

    m0 = np.isfinite(hsr) & np.isfinite(y)
    off = float(np.median(y[m0] - hsr[m0]))
    y_tide = y - off          # survey brought into the tide model's frame
    print(f"\ndatum entre GNSS y modelo de marea: {off:+.3f} m")
    print(f"GNSS en el marco de la marea: {np.nanmin(y_tide):+.2f} a "
          f"{np.nanmax(y_tide):+.2f} m")
    out_lo = float(np.mean(y_tide < lo))
    out_hi = float(np.mean(y_tide > hi))
    print(f"  por DEBAJO de la ventana: {100*out_lo:.0f} %")
    print(f"  por ENCIMA de la ventana: {100*out_hi:.0f} %")
    if out_lo + out_hi > 0.02:
        print("  -> hay puntos que la busqueda no puede alcanzar por"
              " construccion")

    # ── does the compression survive inside the window? ──────────────────
    zf = np.where(good, z, np.nan)[fi]
    print(f"\n{'subconjunto':34s} {'pend':>7s} {'RMSE':>7s} {'n':>5s}")
    print("-" * 58)
    margins = [(0.0, "todos"), (0.15, "a mas de 0.15 m del borde"),
               (0.30, "a mas de 0.30 m del borde"),
               (0.50, "a mas de 0.50 m del borde")]
    res = {}
    for mg, lab in margins:
        inside = (y_tide > lo + mg) & (y_tide < hi - mg)
        sl, rm, n = slope_rmse(np.where(inside, y, np.nan), hsr)
        sl2, rm2, n2 = slope_rmse(np.where(inside, y, np.nan), zf)
        print(f"{lab:34s} {sl:7.3f} {rm:7.3f} {n:5d}   (producto)")
        print(f"{'':34s} {sl2:7.3f} {rm2:7.3f} {n2:5d}   (este ajuste)")
        res[lab] = {"slope_product": sl, "rmse_product": rm, "n": n,
                    "slope_here": sl2, "rmse_here": rm2}

    print("\nSi la pendiente sube hacia 1 al alejarse del borde, la compresion")
    print("es el recorte de la rejilla y no un error del nivel de agua.")
    json.dump({"window": [lo, hi], "at_edge_frac": float(at_edge.mean()),
               "datum": off, "outside_lo": out_lo, "outside_hi": out_hi,
               "subsets": res},
              open(os.path.join(SC, "recorte.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
