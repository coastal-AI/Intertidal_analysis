"""Settle the epoch claim by measuring it.

`elevation.py` says 87 % of pixels saturate the sigma grid over ten years
against 12 % over three; notebook cell 39 says 12 % against 6 %, and adds a
fit residual dropping 0.187 -> 0.171 NDWI. Both cannot be right, and the
package should not carry a number nobody can reproduce.

Saturating the grid means the fitted sigma landed on the LARGEST value the
search offers: the pixel's transition is wider than anything the grid can
express, so the width — and with it the elevation — is not actually measured,
only bounded. That is the honest reason to fit per epoch, and it deserves an
honest number.

Same cube, same mask, same grids; only the date range differs.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, dates = d["Y"], d["C"], d["dates"]

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide = model_tides(x=[lon_c], y=[lat_c], time=times, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    runs = {"10 anos (2016-2025)": np.isfinite(tide),
            "3 anos (2023-2025)": np.isfinite(tide) & (yrs >= 2023)}

    print(f"{'periodo':22s} {'fechas':>7s} {'px validos':>11s} "
          f"{'satura sigma':>13s} {'residuo':>9s}")
    print("-" * 68)
    out = {}
    for label, sel in runs.items():
        t = tide[sel]
        grid = np.linspace(t.min(), t.max(), MU_POINTS)
        a, b, mu, sg, rmse, N = _fit_block(
            np.nan_to_num(Y[sel], nan=0.0).astype(np.float64),
            C[sel].astype(np.float64), t, grid, SG_GRID)
        ok = (N >= 8) & (b > 0.15)
        # Saturated = sigma pinned at the top of the search grid, i.e. the
        # transition is wider than the grid can describe.
        sat = ok & (sg >= max(SG_GRID) - 1e-9)
        frac = 100.0 * sat.sum() / max(ok.sum(), 1)
        res = float(np.median(rmse[ok]))
        out[label] = {"n_dates": int(sel.sum()), "n_valid": int(ok.sum()),
                      "pct_saturated": round(frac, 1),
                      "median_residual_ndwi": round(res, 4)}
        print(f"{label:22s} {int(sel.sum()):7d} {int(ok.sum()):11,d} "
              f"{frac:12.1f} % {res:9.4f}", flush=True)

    json.dump(out, open(os.path.join(SC, "epocas.json"), "w"), indent=1)
    print("\nlo que el modulo y el notebook deben decir, ambos, es lo de "
          "arriba")


if __name__ == "__main__":
    main()
