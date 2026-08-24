"""Does the ported code reproduce what the scratchpad measured?

A finding that only exists in a throwaway script is not a finding. These are
the two numbers from today that went into `tidecheck`, re-derived through the
package's own API rather than the scripts that first produced them:

  * phase lag at Villaviciosa: -8.8 min (i.e. none)
  * spring-minus-neap elevation: -0.25 m

If either differs materially, the port is wrong, not the measurement.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
import pyintertidal as pit
from pyintertidal import tidecheck

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    # Reuse the array today's scripts built, so this checks the FUNCTIONS
    # rather than re-testing the extraction at the same time.
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, dates = d["Y"], d["C"], d["dates"]

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])

    def tide_series(shift_min=0):
        t = base - pd.Timedelta(minutes=int(shift_min))
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    lags = list(range(-120, 121, 20))
    cache = {s: tide_series(s) for s in set(lags) | {-15, 15, 0}}

    yrs = np.array([int(s[:4]) for s in dates])
    recent = yrs >= 2023
    dhdt = (cache[-15] - cache[15]) / 0.5
    flood = (dhdt > 0) & recent
    ebb = (dhdt <= 0) & recent

    print("=" * 62)
    print("PHASE LAG  (scratchpad said -8.8 min)")
    print("=" * 62)
    r = tidecheck.phase_lag(Y, C, lambda m: cache[m], flood, ebb, lags=lags,
                            verbose=False)
    print(f"  crossing: {r['crossing']:+.1f} min"
          if r["crossing"] is not None else "  no crossing")

    print()
    print("=" * 62)
    print("RANGE DEPENDENCE  (scratchpad said -0.25 m)")
    print("=" * 62)
    grid_h = np.vstack([
        model_tides(x=[lon_c], y=[lat_c],
                    time=base + pd.Timedelta(hours=float(h)), model="EOT20",
                    directory="tide_models", crs="EPSG:4326",
                    extrapolate=True, cutoff=np.inf,
                    parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)
        for h in np.arange(-18, 18.5, 1.0)])
    day_range = np.nanmax(grid_h, axis=0) - np.nanmin(grid_h, axis=0)

    tide0 = cache[0].copy()
    tide0[~recent] = np.nan
    dr = day_range.copy()
    dr[~recent] = np.nan
    out = tidecheck.range_dependence(Y, C, tide0, dr)

    json.dump({"phase_lag_min": r["crossing"],
               "range_dependence_m": out["difference_m"]},
              open(os.path.join(SC, "verifica_tidecheck.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
