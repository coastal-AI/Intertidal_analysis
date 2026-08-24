"""Gate 1: is the fitted gamma physics, or the estimator looking at itself?

A gamma that is merely non-zero proves nothing — a noisy estimator produces
non-zero parameters for free. Two things have to hold before the fitted water
surface can be believed:

  1. gamma must depend on the tide. A gamma identical on every date is only a
     static bias field, and cannot produce the relief compression this whole
     line of work is trying to explain.

  2. neither the size of gamma nor its correlation with the tide may be
     reproducible from data that contain no sloping surface at all.

Point 2 is what this script exists for. lamina.py run with SYNTH rebuilds the
NDWI observations from its own fit with a PERFECTLY FLAT water surface, adds
noise matched to the real residual, and turns the same solver loose on them.
Whatever gamma comes back is manufactured by the estimator. If the real gamma
is not clearly bigger, the method is measuring its own reflection.

That check is not optional here. Twice today a result that looked
overwhelming — p = 3.8e-37 in one case — collapsed the moment it was compared
against a properly matched null instead of against zero.
"""
import os
import sys
import glob
import json

import numpy as np
from scipy import stats

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")


def describe(gamma, h0, label):
    ok = np.isfinite(gamma) & np.isfinite(h0)
    r = stats.pearsonr(h0[ok], gamma[ok])
    g1, g0 = np.polyfit(h0[ok], gamma[ok], 1)
    return {"label": label, "sd": float(np.std(gamma[ok])),
            "median": float(np.median(gamma[ok])),
            "r": float(r.statistic), "p": float(r.pvalue),
            "g1": float(g1), "g0": float(g0)}


def load(name):
    p = os.path.join(SC, name)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def report(real_file, null_glob, title):
    real = load(real_file)
    if real is None:
        print(f"\n{title}: falta {real_file}")
        return None
    rows = [describe(real["gamma"], real["h0"], "REAL")]
    for f in sorted(glob.glob(os.path.join(SC, null_glob))):
        s = np.load(f, allow_pickle=True)
        rows.append(describe(s["gamma"], s["h0"],
                             "nulo " + os.path.basename(f).split("sint")[-1][0]))

    print(f"\n{title}")
    print("=" * 68)
    print(f"{'':22s} {'sd gamma':>9s} {'mediana':>9s} {'r(g,h0)':>9s} "
          f"{'g1':>8s}")
    print("-" * 68)
    for r in rows:
        print(f"{r['label']:22s} {r['sd']:9.3f} {r['median']:+9.3f} "
              f"{r['r']:+9.3f} {r['g1']:+8.3f}")
    if "theta" in real:
        t = np.asarray(real["theta"])
        print(f"\n  gamma(t) = {t[0]:+.4f} {t[1]:+.4f}*h0 "
              f"{t[2]:+.4f}*(rango - medio)   m/km")
    return rows


def main():
    print("PUERTA 1 — la inclinacion ajustada, contra un mundo sin inclinacion")
    print("\nsd en m/km · g1 en (m/km)/m: cuanto cambia la inclinacion por cada")
    print("metro de marea. g1 distinto de cero es lo que comprime el relieve.")
    print("Los 'nulos' son NDWI regenerado con lamina PERFECTAMENTE PLANA y el")
    print("mismo ruido, pasado por el mismo solver: lo que salga ahi es lo que")
    print("el estimador fabrica solo.")

    rows_ps = report("lamina.npz", "lamina_sint*.npz",
                     "A — gamma libre por escena (465 parametros)")
    rows_td = report("lamina_tidal.npz", "lamina_tidal_sint*.npz",
                     "B — gamma como funcion de la marea (3 parametros)")

    rows = rows_td or rows_ps
    if rows is None:
        return
    null = [r for r in rows if r["label"] != "REAL"]
    if not null:
        print("\nsin nulos sinteticos todavia")
        return

    R = rows[0]
    sd_n = np.array([r["sd"] for r in null])
    r_n = np.array([abs(r["r"]) for r in null])
    g1_n = np.array([abs(r["g1"]) for r in null])
    print("\n" + "=" * 68)
    print(f"{'magnitud de gamma':24s} real {R['sd']:.3f} · "
          f"nulo {sd_n.mean():.3f} (max {sd_n.max():.3f}) · "
          f"razon {R['sd']/max(sd_n.mean(),1e-9):.2f}x")
    print(f"{'|correlacion con h0|':24s} real {abs(R['r']):.3f} · "
          f"nulo {r_n.mean():.3f} (max {r_n.max():.3f})")
    print(f"{'|g1|':24s} real {abs(R['g1']):.3f} · "
          f"nulo {g1_n.mean():.3f} (max {g1_n.max():.3f})")

    # NOT compared on r(gamma, h0): in the three-parameter model gamma is a
    # linear function of h0 by construction, so that correlation is +-1 for
    # every run, real or synthetic, and comparing it measures nothing. An
    # earlier version passed the gate on a seventh-decimal tie between two
    # values that were both 0.997. The magnitude of g1 is the real test.
    passes = (abs(R["g1"]) > 3 * g1_n.max()) and (R["sd"] > 2 * sd_n.max())
    print()
    if passes:
        print("PUERTA 1 SUPERADA: la lamina que ve el solver no se puede")
        print("fabricar a partir de datos sin lamina.")
    else:
        why = []
        if R["sd"] <= 2 * sd_n.max():
            why.append("su tamano no supera al del nulo")
        if abs(R["r"]) <= r_n.max():
            why.append("su correlacion con la marea tampoco")
        print("PUERTA 1 NO SUPERADA: " + " y ".join(why) + ".")
        print("El conjunto reservado del GNSS NO se abre.")

    json.dump({"per_scene": rows_ps, "tidal": rows_td,
               "passes": bool(passes)},
              open(os.path.join(SC, "puerta1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
