# -*- coding: utf-8 -*-
"""Builder smoke run of the fixed-configuration competitors on real windows.

NOT the table runner (that is experiments/v19_fixed_config_tables.py): a
cross-check that every competitor of :mod:`experiments.fixedcfg.methods` runs
on a site's real common inputs and scores sensibly, with "no delay" (E0a,
v14.invert_subset) as the in-house reference row.

Per site:
* the units, truth and inverted pixels of v14.load_site (lags 'fixed',
  --max-px 100000, seed 20261019: the scored set of v14 / v15 / v16);
* the screened list S: Villaviciosa = the notebook's 305 usable dates
  (v7.villaviciosa_usable_dates) with an extraction row, an overpass time and
  a finite h_B; Dutch sites = result.json record_dates. Gates: S's
  2023-2025 part equals record_dates, its h_B equals v14's level table, and
  its range equals rango_marea_m;
* window(n) = S[-n:] for each requested n (n > |S| is 'not reachable');
* the common inputs of a window: raw NDWI Y (NaN kept) and C = SCL 4-7 of
  the extraction at the inverted pixels, W = C & (Y > 0), h = h_B without lag
  at the overpass instants (one boundary call over S's sorted instants);
* the level series (h_B, 1 min, no lag) from 00:00 of the longest window's
  first day to 2026-01-01, cached in --out, sliced per window from 00:00 of
  its first scene day.

Outputs (new files under --out, which must not be a production folder):
<site>_competitors.csv (one row per n x method), <site>_competitors.json
(meta and gates), <site>_level_series.npz (cache).

Run from the repository root, e.g.:
    python -X utf8 -m experiments.fixedcfg.smoke_competitors villaviciosa --out /tmp/fixedcfg_smoke \\
        --product results/server_10m/products_marea/marea_eot20_g0_2023-2025      # laptop mirror
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N_GRID = (125, 100, 75, 60, 50, 40)
WINDOW_END = "2026-01-01"           # windows end 2025-12-31 (end exclusive)
CSV_COLUMNS = ("site", "n", "method", "label", "status", "reason", "rmse", "bias", "slope", "r",
               "coverage", "n_units", "window_first_scene", "window_last_scene", "scenes_sha1",
               "median_valid_obs", "seconds")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha1_dates(dates):
    return hashlib.sha1(",".join(map(str, dates)).encode()).hexdigest()[:16]


def screened_list(site, extras, cfg, v7):
    """S: chronological screened dates with their extraction rows."""
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        all_ext = np.asarray(ex["dates"]).astype(str)
    overpass = json.load(open(v7.SITES[site]["overpass"], encoding="utf-8"))
    if site == "villaviciosa":
        usable = set(v7.villaviciosa_usable_dates())
        S = [d for d in all_ext if d in usable and d in overpass]
    else:
        S = [d for d in extras["rec_dates"] if d in overpass]
    pos = {d: i for i, d in enumerate(all_ext)}
    rows = np.array([pos[d] for d in S], np.int64)
    if np.any(np.diff(rows) <= 0):
        raise RuntimeError("S is not chronological / unique")
    return np.array(S), rows, overpass


def no_delay_gate(site, cfg, px, z0_mb0, score, published_csv):
    """E0a on the full production record against the published no-delay
    product: per pixel against comparison products.npz z_plain (Dutch sites)
    and, where the scored units are the published set, against the published
    'no delay' row (Villaviciosa). Reported, not a stop."""
    g = {}
    pz = f"{cfg['comparison']}/products.npz" if cfg.get("comparison") else None
    if pz and os.path.exists(pz):
        with np.load(pz, allow_pickle=False) as P:
            zp = P["z_plain"][px].astype(float)
        fa, fb = np.isfinite(z0_mb0), np.isfinite(zp)
        both = fa & fb
        dz = float(np.max(np.abs(z0_mb0[both] - zp[both]))) if both.any() else None
        g["vs_z_plain"] = {"path": pz, "validity_flips": int((fa != fb).sum()), "max_abs_dz_m": dz,
                           "ok": bool((fa == fb).all() and dz is not None and dz <= 1e-6)}
    if os.path.exists(published_csv):
        import pandas as pd

        d = pd.read_csv(published_csv)
        a = d[(d["product"] == "no delay") & (d["subset"] == "all")]
        if len(a) and int(a.iloc[0]["n"]) == int(score["n"]):
            pub = {k: float(a.iloc[0][c]) for k, c in (("rmse", "rmse_m"), ("bias", "bias_m"),
                                                         ("slope", "slope"), ("r", "r"))}
            dd = max(abs(score[k] - pub[k]) for k in pub)
            g["vs_published_row"] = {"path": published_csv, "published": pub, "max_abs_diff": dd,
                                     "ok": dd <= 1e-6}
        elif len(a):
            g["vs_published_row"] = {"path": published_csv, "checked": False,
                                     "reason": f"published n {int(a.iloc[0]['n'])} != scored units "
                                               f"{int(score['n'])} (subsample)"}
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site", choices=("villaviciosa", "escalda", "wadden", "ems"))
    ap.add_argument("--n", type=int, nargs="+", default=list(N_GRID))
    ap.add_argument("--out", required=True)
    ap.add_argument("--product", default=None, help="override of the production product folder")
    ap.add_argument("--max-px", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--methods", default=None, help="comma-separated ids (default: every METHODS entry)")
    ap.add_argument("--variants", action="store_true", help="also run the VARIANTS")
    args = ap.parse_args(argv)

    import pandas as pd
    from experiments import v14_scene_count_curves as v14
    from experiments import v7_waterline_contour as v7
    from experiments.fixedcfg import methods as M
    from experiments.fixedcfg.level_series import LevelSeries
    from types import SimpleNamespace

    v14.guard_out(args.out)
    os.makedirs(args.out, exist_ok=True)
    site = args.site
    cfg = v14.site_config(site)
    if args.product:
        cfg["product"] = args.product
    t0 = time.time()
    sd, meta, extras = v14.load_site(site, cfg, "fixed", args.max_px, args.seed, ["screened"])
    px = extras["px"]
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        H, Wd = (int(v) for v in ex["shape"])
    log(f"[{site}] load_site: {len(px):,} inverted px, {len(sd.unit_col):,} units [{time.time() - t0:.0f} s]")

    # the screened list, its instants and levels; gates
    S, rows_S, overpass = screened_list(site, extras, cfg, v7)
    t_S = pd.DatetimeIndex(pd.to_datetime([pd.Timestamp(overpass[d]).to_pydatetime() for d in S]))
    if t_S.tz is not None:
        t_S = t_S.tz_localize(None)
    if not t_S.is_monotonic_increasing:
        raise RuntimeError("overpass instants are not chronological")
    boundary = extras["rec"]["boundary_obj"]
    h_S = np.asarray(boundary.levels(t_S), float)
    if not np.all(np.isfinite(h_S)):
        raise RuntimeError(f"{int((~np.isfinite(h_S)).sum())} screened scene(s) without a finite h_B")
    rec_dates = list(extras["rec_dates"])
    part = [i for i, d in enumerate(S) if "2023-01-01" <= d <= "2025-12-31"]
    gates = {"S_2023_2025_equals_record_dates": [str(S[i]) for i in part] == rec_dates}
    pos_all = {d: i for i, d in enumerate(extras["dates_all"])}
    both = [i for i, d in enumerate(S) if d in pos_all]
    dl = float(np.max(np.abs(h_S[both] - sd.levels[0][[pos_all[S[i]] for i in both]])))
    gates["h_vs_v14_level_table_max_abs_m"] = dl
    lohi = [float(h_S[part].min()), float(h_S[part].max())]
    dr = float(np.max(np.abs(np.array(lohi) - np.asarray(extras["res"]["rango_marea_m"], float))))
    gates["record_range_vs_rango_marea_m_max_abs"] = dr
    ok = gates["S_2023_2025_equals_record_dates"] and dl <= 1e-9 and dr <= 1e-9
    log(f"[{site}] S = {len(S)} screened scenes ({S[0]} .. {S[-1]}); gates {gates} -> {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(2)

    # raw Y and C of every S row at the inverted pixels
    t1 = time.time()
    Yr = v14.stream_columns(cfg["extract"], "Y", rows_S, [px])[0]
    Cr = v14.stream_columns(cfg["extract"], "C", rows_S, [px])[0].astype(bool)
    log(f"[{site}] streamed {len(rows_S)} rows x {len(px):,} px [{time.time() - t1:.0f} s]")

    # the level series over the longest reachable window
    reach = [n for n in args.n if n <= len(S)]
    ls_path = os.path.join(args.out, f"{site}_level_series.npz")
    full = None
    if reach:
        start = t_S[-max(reach)].normalize()
        if os.path.exists(ls_path):
            full = LevelSeries.load(ls_path)
            if full.t[0] > start or full.t[-1] < pd.Timestamp(WINDOW_END) - pd.Timedelta(minutes=1):
                full = None
        if full is None:
            t1 = time.time()
            full = LevelSeries.from_boundary(boundary, start, WINDOW_END)
            full.save(ls_path)
            log(f"[{site}] level series {full.summary()} [{time.time() - t1:.0f} s]")

    ids = (args.methods.split(",") if args.methods else list(M.METHODS)) + \
        (list(M.VARIANTS) if args.variants else [])
    rows_out, metas = [], {}
    for n in args.n:
        if n > len(S):
            for mid in ["E0a"] + ids:
                rows_out.append(dict(site=site, n=n, method=mid, status="not reachable",
                                     reason=f"{len(S)} screened scenes < {n}"))
            continue
        w = slice(len(S) - n, len(S))
        Y, C = Yr[w], Cr[w]
        with np.errstate(invalid="ignore"):
            W = C & (Y > 0)
        obs = SimpleNamespace(Y=Y, C=C, W=W, h=h_S[w], t=t_S[w], keep=extras["keep"][px], shape=(H, Wd),
                              pixel_m=10.0, site=site)
        ls = full.slice(t_S[w][0].normalize(), WINDOW_END)
        common = dict(site=site, n=n, window_first_scene=str(S[w][0]), window_last_scene=str(S[w][-1]),
                      scenes_sha1=sha1_dates(S[w]))
        nvalid = (C & np.isfinite(Y)).sum(0)[sd.unit_col]
        # E0a no delay, the in-house reference (production inversion at tau = 0)
        t1 = time.time()
        z0, _, (lo, hi), _, _ = v14.invert_subset(np.nan_to_num(Y, nan=0.0).astype(np.float32), C,
                                                  {0: h_S[w]}, np.arange(n), v14.lag_groups(np.zeros(len(px))),
                                                  [0, 3])
        for mb in (0, 3):
            s = v14.score(sd.kind, sd.truth, z0[mb][sd.unit_col])
            rows_out.append(dict(common, method=f"E0a_mb{mb}", label=f"no delay (mb {mb})", status="ok",
                                 rmse=s["rmse"], bias=s["bias"], slope=s["slope"], r=s["r"],
                                 coverage=s["coverage"], n_units=s["n"], median_valid_obs=float(np.median(nvalid)),
                                 seconds=round(time.time() - t1, 2)))
            if mb == 0 and [str(d) for d in S[w]] == rec_dates:
                gates["no_delay_full_record"] = no_delay_gate(
                    site, cfg, px, z0[0], s, f"results/method_comparison_same_scenes/{site}.csv")
                log(f"[{site}] no-delay gate on the production record: {gates['no_delay_full_record']}")
        for mid in ids:
            label, run, card = M.get(mid)
            out = run(obs, {"level_series": ls})
            m = out["meta"]
            s = v14.score(sd.kind, sd.truth, out["z"].astype(float)[sd.unit_col])
            rows_out.append(dict(common, method=mid, label=label, status=m["status"], reason=m["reason"],
                                 rmse=s["rmse"], bias=s["bias"], slope=s["slope"], r=s["r"],
                                 coverage=s["coverage"], n_units=s["n"],
                                 median_valid_obs=float(np.median(nvalid)), seconds=m["seconds"]))
            metas[f"{n}:{mid}"] = {k: v for k, v in m.items() if k != "extra"}
        log(f"[{site}] n={n}: " + "; ".join(f"{r['method']} {r.get('rmse', float('nan')):.3f}"
                                           for r in rows_out if r.get("n") == n and r.get("status") == "ok"))

    csv_path = os.path.join(args.out, f"{site}_competitors.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        wr.writeheader()
        for r in rows_out:
            wr.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    v14.write_json(os.path.join(args.out, f"{site}_competitors.json"),
                   {"site": site, "gates": gates, "S": [str(d) for d in S], "n_grid": args.n,
                    "fixed_set": meta.get("fixed_set"), "level_series": full.summary() if full else None,
                    "meta": metas, "seconds": round(time.time() - t0, 1)})
    log(f"[{site}] wrote {csv_path} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
