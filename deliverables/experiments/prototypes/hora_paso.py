"""The tide has been read at the wrong time. How much has that cost?

Every result in this project pairs a scene with the tide height at 11:00 UTC,
because the cube stores dates and not times. Sentinel-2 does not cross at
11:00 every day: the two satellites have different local times, the swath
position within the orbit shifts the crossing, and the nominal local solar
time itself drifts slowly. A twenty-minute error near mid-tide, where the
water moves fastest, is of the order of twenty centimetres — comparable to
everything this project has been trying to explain.

pyintertidal.overpass.get_overpass_times exists for exactly this and queries
the real acquisition times from the STAC catalogue. This measures what
assuming 11:00 actually costs, in metres of water level, using the real tide
at both the assumed and the true instants.

It also settles a question the boundary-agnostic redesign raises: whether the
archive can rank global tide models at all. It can only do so if the models
disagree AT THE INSTANTS WE OBSERVE. If they differ by a centimetre there, no
amount of statistical machinery over 40 000 pixels will separate them, and a
ranking would be reading noise.

Finally, the alias table. Sentinel-2 is sun-synchronous, so every acquisition
happens at essentially the same solar time. A constituent whose period divides
the solar day exactly is then frozen at one phase and cannot be seen at all,
no matter how long the archive. That is a hard limit on what any interior tide
model can hope to estimate from imagery, and it is arithmetic, not opinion.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

# period in hours, and whether it is a major constituent
CONSTITUENTS = {
    "M2": 12.4206012, "S2": 12.0000000, "N2": 12.6583475, "K2": 11.9672348,
    "K1": 23.9344697, "O1": 25.8193417, "P1": 24.0658902, "Q1": 26.8683567,
    "M4": 6.2103006, "M6": 4.1402004,
}


def alias_table():
    """Which constituents a sun-synchronous sensor can see at all."""
    print("TABLA DE ALIAS — muestreo a hora solar fija (1 muestra/dia)")
    print(f"{'constituyente':14s} {'periodo (h)':>12s} {'alias':>14s} "
          f"{'estimable':>11s}")
    print("-" * 56)
    rows = {}
    for name, per in CONSTITUENTS.items():
        f = 24.0 / per                       # cycles per solar day
        f_alias = abs(f - round(f))          # folded into [0, 0.5]
        if f_alias < 1e-6:
            per_alias, verdict = np.inf, "NO — congelado"
        else:
            per_alias = 1.0 / f_alias        # days
            if per_alias > 300:
                verdict = "dudoso (~anual)"
            elif per_alias > 100:
                verdict = "dificil"
            else:
                verdict = "si"
        lab = "infinito" if not np.isfinite(per_alias) else f"{per_alias:.1f} d"
        print(f"{name:14s} {per:12.4f} {lab:>14s} {verdict:>11s}")
        rows[name] = {"period_h": per, "alias_days": float(per_alias),
                      "verdict": verdict}
    print("\nS2 es el constituyente SOLAR: su periodo divide el dia solar")
    print("exacto, asi que a hora fija siempre se observa en la misma fase y")
    print("es invisible por construccion. Ningun archivo, por largo que sea,")
    print("lo recupera. K1 y P1 caen en el alias anual y se confunden con la")
    print("senal estacional.")
    return rows


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    dates = np.array([str(s) for s in d["dates"]])
    yrs = np.array([int(s[:4]) for s in dates])
    dates = dates[yrs >= 2023]

    rows = alias_table()

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    print(f"\n\nHORA REAL DE PASO — consultando el catalogo STAC…", flush=True)
    try:
        times = pit.overpass.get_overpass_times(
            aoi.bbox, ("2023-01-01", "2025-12-31"), verbose=False)
    except Exception as e:
        print(f"  fallo la consulta: {type(e).__name__}: {str(e)[:120]}")
        return
    print(f"  {len(times)} fechas con hora real")

    have = [dt for dt in dates if dt in times]
    real = pd.to_datetime([times[dt] for dt in have]).tz_localize(None)
    assumed = pd.to_datetime([f"{dt} 11:00:00" for dt in have])
    dmin = (real - assumed).total_seconds() / 60.0
    print(f"  coinciden con nuestras escenas: {len(have)}")
    print(f"  desviacion respecto a las 11:00: mediana {np.median(dmin):+.1f} "
          f"min, p5 {np.percentile(dmin,5):+.1f}, p95 "
          f"{np.percentile(dmin,95):+.1f}, recorrido "
          f"{dmin.max()-dmin.min():.0f} min")

    def tides_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=pd.DatetimeIndex(t),
                           model="EOT20", directory="tide_models",
                           crs="EPSG:4326", extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    h_real, h_assumed = tides_at(real), tides_at(assumed)
    err = h_assumed - h_real
    print(f"\nCOSTE EN NIVEL DE AGUA de suponer las 11:00:")
    print(f"  sesgo    {np.mean(err):+.3f} m")
    print(f"  sd       {np.std(err):.3f} m")
    print(f"  |p95|    {np.percentile(np.abs(err),95):.3f} m")
    print(f"  |max|    {np.abs(err).max():.3f} m")
    print(f"\n  para comparar: el error del metodo contra el GNSS ronda")
    print(f"  0.20 m, y el muestreo sub-pixel del RTK vale 0.214 m.")

    # ── can the archive rank global tide models? ─────────────────────────
    print(f"\n\nDISCREPANCIA ENTRE MODELOS EN LOS INSTANTES OBSERVADOS")
    hs = {}
    for m in ("EOT20", "GOT4.10"):
        try:
            hs[m] = model_tides(
                x=[lon_c], y=[lat_c], time=pd.DatetimeIndex(real), model=m,
                directory="tide_models", crs="EPSG:4326", extrapolate=True,
                cutoff=np.inf, parallel=False).reset_index().sort_values(
                "time")["tide_height"].to_numpy(float)
        except Exception as e:
            print(f"  {m}: no disponible ({type(e).__name__})")
    ks = list(hs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            dd = hs[ks[i]] - hs[ks[j]]
            print(f"  {ks[i]} - {ks[j]}: sesgo {np.mean(dd):+.3f} m, "
                  f"sd {np.std(dd):.3f} m, |max| {np.abs(dd).max():.3f} m")
    print("\nSi los modelos discrepan MENOS que el error de hora que acabamos")
    print("de medir, ordenarlos desde el archivo es leer ruido, y la seccion")
    print("de 'evaluar modelos globales' del rediseno no se sostiene todavia.")

    json.dump({"alias": rows,
               "n_matched": len(have),
               "minutes": {"median": float(np.median(dmin)),
                           "p5": float(np.percentile(dmin, 5)),
                           "p95": float(np.percentile(dmin, 95))},
               "tide_error": {"bias": float(np.mean(err)),
                              "sd": float(np.std(err)),
                              "p95_abs": float(np.percentile(np.abs(err), 95)),
                              "max_abs": float(np.abs(err).max())}},
              open(os.path.join(SC, "hora_paso.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
