"""C4 — Granadeiro's Eq. (1) driven by a tide GAUGE, and by a consensus of
gauges, instead of the ocean model.

Their method interpolates the water height of each scene between the
bracketing low and high waters of ONE reference record (Bubaque). Here
that record is, in turn: the nearest gauge (Gijon, IOC gij2) and the mean
of 2, 3 and 4 demeaned Cantabrian gauges (Gijon, Santander, Bilbao,
Ferrol), resampled to 15 min. Everything else in the method is untouched;
the epoch is 2023-2025 (the gauge records' span), so the fair reference is
the epoch-matched EOT20 run. Judged afterwards by c1 on the RTK dev blocks.

Run:  python -m experiments.c4_granadeiro_gauges
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SETS = {"gij2": ["gij2"], "cons2": ["gij2", "san2"],
        "cons3": ["gij2", "san2", "bil3"],
        "cons4": ["gij2", "san2", "bil3", "fer2"]}


def gauge_series(codes, t0="2023-01-01", t1="2025-12-31 23:45"):
    import pandas as pd
    from pyintertidal.gauges import load_cached_ioc

    idx = pd.date_range(t0, t1, freq="15min")
    stack = []
    for c in codes:
        df = load_cached_ioc(c)
        if df is None or len(df) < 1000:
            raise SystemExit(f"gauge {c}: no cached record")
        s = (df.set_index("time")["level_m"].sort_index()
             .loc[t0:t1].resample("15min").mean().interpolate(limit=8))
        s = s - s.median()
        stack.append(s.reindex(idx).to_numpy(float))
    return idx, np.nanmean(np.vstack(stack), axis=0)


def main():
    from experiments.sota_granadeiro import main as run
    for tag, codes in SETS.items():
        st, sh = gauge_series(codes)
        cov = float(np.isfinite(sh).mean())
        print(f"\n=== boundary {tag} = {codes}: {cov:.0%} of the epoch covered")
        # 1000 lag pixels per variant (their sample scaled to this flat and
        # to four runs); the epoch-matched EOT20 reference used 2500
        run("villaviciosa", n_lag_px=1000, years=(2023, 2025),
            boundary=(st, sh), tag=tag)


if __name__ == "__main__":
    main()
