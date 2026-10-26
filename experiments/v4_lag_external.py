# -*- coding: utf-8 -*-
"""External validation of the interior lag: gauge pairs — the paper figure
and table, assembled from what the pipeline notebooks wrote.

The computation lives in section 10 of ``tide_boundary_comparison_<site>``
(builder ``scratchpad/build_escalda_nb.py``); this script only reads its
CSVs so that the paper carries the notebook's numbers and nothing else:

* ``lag_pair_by_level.csv`` — lag between the mouth gauge and the inner
  gauge in reaching the same level, per limb and level;
* ``lag_pair_metrics.csv`` — the gauge-pair lag (all / rising / falling),
  MAREA's lag at the inner gauge (band, 95 % interval, smoothed profile),
  and the same-instant level difference;
* ``judge_metrics.csv`` — the level error at the inner gauge without and
  with the lag applied (rows ``no delay`` and ``MAREA``).

The loops of the figure (inner against mouth level at the same instant)
are redrawn from the cached gauge records, as in the notebook.

Writes results/v4_lag_external.json, results/v4_lag_external.md and
docs/paper/figures/fig_lag_external.png.
Run from the repository root:  python -m experiments.v4_lag_external
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyintertidal import lag_validation as lv

matplotlib.use("Agg")

SITES = {
    "ferrol": dict(name="Ría de Ferrol", mouth="fer1", inner="fer2", s_judge=10.1,
                   folder="products_ferrol/comparison_eot20_g1"),
    "escalda": dict(name="Westerschelde", mouth="vlis", inner="trnz", s_judge=21.9,
                    folder="products_escalda/comparison_eot20_g2"),
    "ems": dict(name="Ems-Dollard", mouth="bork", inner="delf", s_judge=24.4,
                folder="products_ems/comparison_eot20_g1"),
    "wadden": dict(name="Wadden Sea (Vlie)", mouth="ters", inner="harl", s_judge=18.3,
                   folder="products_wadden/comparison_eot20_g1"),
}


def _metric(df, key):
    """Value of one row of lag_pair_metrics.csv (matched by the start of
    its quantity label), as float when it parses."""
    hit = df[df["quantity"].str.startswith(key)]
    if hit.empty:
        return float("nan")
    v = hit["value"].iloc[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def read_site(cfg):
    f = cfg["folder"]
    pair = pd.read_csv(f"{f}/lag_pair_metrics.csv")
    judge = pd.read_csv(f"{f}/judge_metrics.csv")
    by_level = pd.read_csv(f"{f}/lag_pair_by_level.csv")
    band_row = pair[pair["quantity"].str.startswith("MAREA lag, band 0 to band")]
    band_k = int(band_row["quantity"].iloc[0].split("band ")[-1].split(" ")[0]) if not band_row.empty else -1
    out = {
        "site": cfg["name"], "mouth_gauge": cfg["mouth"], "inner_gauge": cfg["inner"],
        "s_judge_km": cfg["s_judge"],
        "gauge_lag_mean_min": _metric(pair, "gauge-to-gauge lag, all"),
        "gauge_lag_rising_min": _metric(pair, "gauge-to-gauge lag, rising"),
        "gauge_lag_falling_min": _metric(pair, "gauge-to-gauge lag, falling"),
        "marea_band_of_judge": band_k,
        "marea_lag_band_min": _metric(pair, "MAREA lag, band 0 to band"),
        "marea_lag_band_ci95_min": _metric(pair, "MAREA lag, 95 % bootstrap"),
        "marea_lag_at_judge_min": _metric(pair, "MAREA lag at the judge's distance"),
        "same_instant_rms_m": _metric(pair, "same-instant level difference inner − mouth, RMS"),
        "same_instant_mean_m": _metric(pair, "same-instant level difference inner − mouth, mean"),
        "n_samples": _metric(pair, "10-min samples"),
        "gauge_lag_table": by_level.to_dict(orient="records"),
        "judge_rmse_m": {r["method"]: float(r["RMSE at the judge (m)"]) for _, r in judge.iterrows()},
        "judge_lag_min": {r["method"]: float(r["lag at the judge (min)"]) for _, r in judge.iterrows()},
        "judge_n": {r["method"]: int(r["n"]) for _, r in judge.iterrows()},
    }
    return out


def main():
    out = {}
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.95), dpi=200)   # one row: a quarter page in print
    for ax, (key, cfg) in zip(axs.ravel(), SITES.items()):
        if not all(os.path.exists(f"{cfg['folder']}/{f}") for f in
                   ("judge_metrics.csv", "lag_pair_metrics.csv", "lag_pair_by_level.csv")):
            print(f"{cfg['name']}: notebook outputs not found in {cfg['folder']} — skipped")
            ax.set_axis_off()
            continue
        r = read_site(cfg)
        out[key] = r
        by = pd.DataFrame(r["gauge_lag_table"])
        ci = r["marea_lag_band_ci95_min"]
        lo_hi = re.findall(r"[-+]?\d+", str(ci))
        lo_, hi_ = (float(lo_hi[0]), float(lo_hi[1])) if len(lo_hi) >= 2 else (np.nan, np.nan)
        ax.axhspan(lo_, hi_, color="#c1121f", alpha=0.12, lw=0,
                   label="MAREA band, 95 % interval" if key == "ferrol" else None)
        ax.axhline(r["marea_lag_at_judge_min"], color="#c1121f", lw=1.4,
                   label="MAREA, lag applied at the gauge" if key == "ferrol" else None)
        ax.axhline(0, color="0.45", lw=0.8, label="no delay" if key == "ferrol" else None)
        for limb, col, mk in (("rising", "#1a76b3", "^"), ("falling", "#e0a000", "v")):
            t = by[by["limb"] == limb]
            ax.errorbar(t["level_m"], t["lag_mean_min"], yerr=t["lag_std_min"] / np.sqrt(t["n"].clip(lower=1)),
                        fmt=mk + "-", ms=4, lw=1, color=col, capsize=2,
                        label=f"gauge to gauge, {limb} limb" if key == "ferrol" else None)
        ax.set_title(f"({'abcd'[list(SITES).index(key)]}) {cfg['name']}", fontsize=8, loc="left")
        ax.set_xlabel("level at the mouth gauge (m)", fontsize=7)
        if key == "ferrol":
            ax.set_ylabel("lag to the inner gauge (min)", fontsize=7)
        ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
        e0, e1 = r["judge_rmse_m"].get("no delay", np.nan), r["judge_rmse_m"].get("MAREA", np.nan)
        print(f"{cfg['name']}: gauge lag {r['gauge_lag_mean_min']:+.0f} min (rising {r['gauge_lag_rising_min']:+.0f}, "
              f"falling {r['gauge_lag_falling_min']:+.0f}); MAREA band {r['marea_band_of_judge']} "
              f"{r['marea_lag_band_min']:+.0f} {r['marea_lag_band_ci95_min']}, at the judge {r['marea_lag_at_judge_min']:+.0f}; "
              f"same-instant RMS {r['same_instant_rms_m']:.3f} m; level error no delay {e0:.3f} / MAREA {e1:.3f} m")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs("docs/paper/figures", exist_ok=True)
    fig.savefig("docs/paper/figures/fig_lag_external.png", dpi=200, bbox_inches="tight")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/v4_lag_external.json", "w"), indent=1)

    lines = ["| site | gauge lag (min): all / rising / falling | MAREA at the inner gauge (min): applied (band [95 % CI]) "
             "| same-instant RMS (m) | level error at the inner gauge (m): no delay / MAREA |",
             "|---|---|---|---|---|"]
    for key, r in out.items():
        lines.append(f"| {r['site']} | {r['gauge_lag_mean_min']:+.0f} / {r['gauge_lag_rising_min']:+.0f} / "
                     f"{r['gauge_lag_falling_min']:+.0f} | {r['marea_lag_at_judge_min']:+.0f} "
                     f"({r['marea_lag_band_min']:+.0f} {r['marea_lag_band_ci95_min']}) | {r['same_instant_rms_m']:.2f} | "
                     f"{r['judge_rmse_m'].get('no delay', float('nan')):.3f} / {r['judge_rmse_m'].get('MAREA', float('nan')):.3f} |")
    md = "\n".join(lines)
    open("results/v4_lag_external.md", "w", encoding="utf-8").write(md + "\n")
    print("\n" + md)
    print("\n-> results/v4_lag_external.json, results/v4_lag_external.md, docs/paper/figures/fig_lag_external.png")


if __name__ == "__main__":
    main()
