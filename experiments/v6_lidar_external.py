# -*- coding: utf-8 -*-
"""External validation of the elevation maps — the paper table and figure,
assembled from what the pipeline notebooks wrote.

The computation lives in the notebooks: section 10b of
``tide_boundary_comparison_<site>`` (IGN 5 m LiDAR for Ferrol, the
Rijkswaterstaat Vaklodingen for the three Dutch sites) and section 9b of
``intertidal_topography_villaviciosa`` (IGN LiDAR). This script only reads
their CSVs so that the paper carries the notebooks' numbers and nothing
else:

* ``lidar_metrics.csv`` / ``lidar_external_metrics.csv`` — three products
  against the LiDAR, on every pixel each resolves and on the common pixels;
* ``vaklodingen_metrics.csv`` — the same against the Vaklodingen, plus the
  5 x 5-thinned scores;
* ``vaklodingen_by_band.csv`` — RMSE and slope by band of distance from the
  mouth, the figure.

Writes results/v6_elevation_external.json, results/v6_elevation_external.md
and docs/paper/figures/fig_vaklodingen_bands.png.
Run from the repository root:  python -m experiments.v6_lidar_external
"""
from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

SITES = {
    "villaviciosa": dict(name="Ría de Villaviciosa", reference="IGN LiDAR",
                         metrics="products_marea/lidar_external_metrics.csv"),
    "ferrol": dict(name="Ría de Ferrol", reference="IGN LiDAR",
                   metrics="products_ferrol/comparison_eot20_g1/lidar_metrics.csv"),
    "escalda": dict(name="Westerschelde", reference="Vaklodingen",
                    metrics="products_escalda/comparison_eot20_g2/vaklodingen_metrics.csv",
                    by_band="products_escalda/comparison_eot20_g2/vaklodingen_by_band.csv"),
    "ems": dict(name="Ems-Dollard", reference="Vaklodingen",
                metrics="products_ems/comparison_eot20_g1/vaklodingen_metrics.csv",
                by_band="products_ems/comparison_eot20_g1/vaklodingen_by_band.csv"),
    "wadden": dict(name="Wadden Sea (Vlie)", reference="Vaklodingen",
                   metrics="products_wadden/comparison_eot20_g1/vaklodingen_metrics.csv",
                   by_band="products_wadden/comparison_eot20_g1/vaklodingen_by_band.csv"),
}
PRODUCTS = ("MAREA", "Granadeiro")        # the method comparison; the CSVs also hold "no delay"
COLOURS = {"no delay": "0.45", "MAREA": "#c1121f", "Granadeiro": "#2a6f97"}


def _col(df, *names):
    """First of several candidate column names present in ``df``."""
    for n in names:
        if n in df.columns:
            return n
    return None


def read_metrics(path):
    df = pd.read_csv(path)
    rows = {}
    c_n = _col(df, "n")
    c_rmse = _col(df, "RMSE (m)", "rmse_m")
    c_slope = _col(df, "slope")
    c_r = _col(df, "r")
    c_bias = _col(df, "bias (m)", "bias_m")
    c_crmse = _col(df, "common RMSE (m)", "common RMSE")
    c_cslope = _col(df, "common slope")
    c_cr = _col(df, "common r")
    c_trmse = _col(df, "thinned RMSE (m)", "thinned RMSE")
    c_tslope = _col(df, "thinned slope")
    for _, r in df.iterrows():
        rows[str(r["product"])] = {
            "n": None if c_n is None else int(r[c_n]) if pd.notna(r[c_n]) else None,
            "rmse_m": None if c_rmse is None else float(r[c_rmse]),
            "slope": None if c_slope is None else float(r[c_slope]),
            "r": None if c_r is None else float(r[c_r]),
            "bias_m": None if c_bias is None else float(r[c_bias]),
            "common_rmse_m": None if c_crmse is None else float(r[c_crmse]),
            "common_slope": None if c_cslope is None else float(r[c_cslope]),
            "common_r": None if c_cr is None else float(r[c_cr]),
            "thinned_rmse_m": None if c_trmse is None else float(r[c_trmse]),
            "thinned_slope": None if c_tslope is None else float(r[c_tslope]),
        }
    return rows


def main():
    out = {}
    for key, cfg in SITES.items():
        if not os.path.exists(cfg["metrics"]):
            print(f"{cfg['name']}: {cfg['metrics']} not found — skipped")
            continue
        out[key] = {"site": cfg["name"], "reference": cfg["reference"], "rows": read_metrics(cfg["metrics"])}
        if cfg.get("by_band") and os.path.exists(cfg["by_band"]):
            out[key]["by_band"] = pd.read_csv(cfg["by_band"]).to_dict(orient="records")
        rr = out[key]["rows"]
        print(f"{cfg['name']} ({cfg['reference']}): " + "; ".join(
            f"{p} RMSE {rr[p]['rmse_m']:.3f} m, slope {rr[p]['slope']:.2f}" for p in PRODUCTS if p in rr and rr[p]["rmse_m"] is not None))

    # figure: RMSE by band against the Vaklodingen, one panel per Dutch site
    dutch = [k for k in ("escalda", "ems", "wadden") if k in out and "by_band" in out[k]]
    if dutch:
        fig, axs = plt.subplots(1, len(dutch), figsize=(7.2, 1.65), dpi=200, sharey=False)
        axs = [axs] if len(dutch) == 1 else list(axs)
        for ax, key in zip(axs, dutch):
            bb = pd.DataFrame(out[key]["by_band"])
            for p in PRODUCTS:
                t = bb[bb["product"] == p]
                if t.empty:
                    continue
                ax.plot(t["s_km"], t["rmse_m"], "o-", ms=3, lw=1.2, color=COLOURS[p], label=p)
            ax.set_title(SITES[key]["name"], fontsize=9, loc="left")
            ax.set_xlabel("distance from the mouth (km)", fontsize=8)
            ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        axs[0].set_ylabel("RMSE vs Vaklodingen (m), median-centred", fontsize=8)
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        os.makedirs("docs/paper/figures", exist_ok=True)
        fig.savefig("docs/paper/figures/fig_vaklodingen_bands.png", dpi=200)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/v6_elevation_external.json", "w"), indent=1)
    lines = ["| site | reference | product | n | RMSE (m) | slope | r | bias (m) | common RMSE (m) | common slope |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in out.items():
        for p in PRODUCTS:
            if p not in r["rows"]:
                continue
            m = r["rows"][p]
            f = lambda v, fmt: "" if v is None else format(v, fmt)
            lines.append(f"| {r['site']} | {r['reference']} | {p} | {f(m['n'], ',d')} | {f(m['rmse_m'], '.3f')} | "
                         f"{f(m['slope'], '.2f')} | {f(m['r'], '.2f')} | {f(m['bias_m'], '+.2f')} | "
                         f"{f(m['common_rmse_m'], '.3f')} | {f(m['common_slope'], '.2f')} |")
    md = "\n".join(lines)
    open("results/v6_elevation_external.md", "w", encoding="utf-8").write(md + "\n")
    print("\n" + md)
    print("\n-> results/v6_elevation_external.json, results/v6_elevation_external.md"
          + (", docs/paper/figures/fig_vaklodingen_bands.png" if dutch else ""))


if __name__ == "__main__":
    main()
