"""Build the eo-tides ensemble tide at Villaviciosa, DEA's own machinery.

The paper's ensemble ranks tide models at each location and blends the top
three. Of its nine members we can reach three without credentials — EOT20,
GOT4.10 and GOT5.6_extrapolated — and since the default `ensemble_top_n` is
three, an ensemble over exactly those three has the same structure as theirs,
just a smaller pool to choose from.

One caveat travels with every number below. The rankings that make the
ensemble "locally optimised" are 32,936 points, all in Australian waters; the
nearest is about 15,000 km from here, so the inverse-distance weighting
returns something close to the Australian average ranking rather than a local
one. That is what eo-tides does outside its calibration area, and it is worth
measuring rather than assuming.

An unweighted mean of the same three models is computed alongside, as the
thing to compare against: if the two agree, the borrowed rankings carry no
information here, which is itself the answer.

Everything lives under a __main__ guard: eo-tides parallelises with spawned
processes, and on Windows each one re-imports this module top to bottom —
without the guard the script forks itself instead of doing the work.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates

import pyintertidal as pit
from eo_tides.model import model_tides

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
RANK = os.path.join(SC, "rankings_local.fgb")
MEMBERS = ["EOT20", "GOT4.10_nc", "GOT5.6_extrapolated"]


def main():
    use_system_certificates()

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid

    got = json.load(open(os.path.join(SC, "tides.json")))
    dates = sorted(got)
    times = pd.to_datetime([f"{d} 11:00:00" for d in dates])
    print(f"{len(dates)} fechas, centro lon {lon_c:.4f} lat {lat_c:.4f}\n",
          flush=True)

    # ONE point, many times. Passing a point per timestamp makes eo-tides
    # evaluate the full cross product — 1379 x 1379 — for a tide that is
    # constant across an estuary this small.
    common = dict(x=[lon_c], y=[lat_c], time=times,
                  directory="tide_models", crs="EPSG:4326",
                  extrapolate=True, cutoff=np.inf, parallel=False)

    def series(**kw):
        df = model_tides(**common, **kw).reset_index()
        return df.sort_values("time")["tide_height"].to_numpy(float)

    per_model = {}
    for m in MEMBERS:
        v = series(model=m)
        per_model[m] = v
        print(f"  {m:22s} std {np.nanstd(v):.3f}  "
              f"rango {np.nanmin(v):+.2f}..{np.nanmax(v):+.2f}", flush=True)

    ens = series(model="ensemble", ensemble_models=MEMBERS,
                 ranking_points=RANK)
    print(f"\n  {'ENSEMBLE eo-tides':22s} std {np.nanstd(ens):.3f}  "
          f"rango {np.nanmin(ens):+.2f}..{np.nanmax(ens):+.2f}", flush=True)

    mean3 = np.nanmean(np.vstack([per_model[m] for m in MEMBERS]), axis=0)
    print(f"  {'media simple de 3':22s} std {np.nanstd(mean3):.3f}  "
          f"rango {np.nanmin(mean3):+.2f}..{np.nanmax(mean3):+.2f}")

    d = ens - mean3
    print(f"\nensemble vs media simple: RMS {np.sqrt(np.nanmean(d**2)):.4f} m"
          f"  max {np.nanmax(np.abs(d)):.4f} m")
    print("(si es ~0, los rankings australianos no aportan informacion aqui)")

    print("\ndiferencias del ensemble con cada miembro:")
    for m, v in per_model.items():
        dd = ens - v
        print(f"  vs {m:22s} RMS {np.sqrt(np.nanmean(dd**2)):.3f} m")

    for name, arr in [("ensemble", ens), ("mean3", mean3)]:
        ok = np.isfinite(arr)
        json.dump({d_: float(v) for d_, v in zip(np.array(dates)[ok], arr[ok])},
                  open(os.path.join(SC, f"tides_{name}.json"), "w"))
        print(f"guardadas {int(ok.sum())} mareas -> tides_{name}.json",
              flush=True)


if __name__ == "__main__":
    main()
