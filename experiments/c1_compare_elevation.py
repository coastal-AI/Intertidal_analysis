"""C1 — the honest four-way elevation comparison on the DEVELOPMENT split.

Contenders, all on the same Villaviciosa grid and judged against the same
RTK pixels (median-centred axes, since the GNSS heights are ellipsoidal and
every product lives on the EOT20 MSL datum — a constant offset apart):

* Granadeiro et al. 2021 — the state of the art, implemented to the letter
  (experiments/sota_granadeiro.py), both WITHOUT (prelim) and WITH (final)
  its own tide-stage-lag correction;
* the step method (DEA-style), our long-standing baseline;
* HSR — our sub-pixel mixing estimator, no interior tide;
* MAREA — binary inversion with the band-clock operator (when its product
  exists on disk).

THE SEAL. The RTK blocks were split once (150-m UTM blocks, seed 20260817,
35 % held out — deliverables/experiments/prototypes/holdout.py) and the
held-out half is opened ONCE, at the end, on a frozen table. This script
therefore reproduces that split bit-for-bit (same block formula, same
seed) and evaluates ONLY the training half. It prints nothing about the
reserved points beyond their count.

Metrics: RMSE and OLS slope, both at once — improving one by ruining the
other does not count. Also per-method n (coverage of the RTK pixels).

Run:  python -m experiments.c1_compare_elevation
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import rasterio

BLOCK_M = 150.0
HOLDOUT_FRACTION = 0.35
SEED = 20260817


def load_products(rr, cc, H, W):
    """Each contender's elevation at the RTK pixels."""
    out = {}
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        out["HSR (no interior tide)"] = s.read(1)[rr, cc]
        transform = s.transform
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        out["step (DEA-style)"] = s.read(1)[rr, cc]

    flat = rr * W + cc
    for fname, tag in (("granadeiro_products.npz", "10y"),
                       ("granadeiro_products_2023-2025.npz", "2023-25")):
        if not os.path.exists(fname):
            continue
        g = np.load(fname)
        pos = {int(k): i for i, k in enumerate(g["keep"])}
        gi = np.array([pos.get(int(f), -1) for f in flat])
        for label, key in ((f"Granadeiro {tag} (no lags)", "prelim"),
                           (f"Granadeiro {tag} (their lags)", "final")):
            v = np.full(len(flat), np.nan)
            m = gi >= 0
            v[m] = g[key][gi[m]]
            out[label] = v

    marea_npz = "products_villaviciosa_marea/marea.npz"
    if os.path.exists(marea_npz):
        mz = np.load(marea_npz)
        posm = {int(k): i for i, k in enumerate(mz["keep"])}
        mi = np.array([posm.get(int(f), -1) for f in flat])
        v = np.full(len(flat), np.nan)
        m = mi >= 0
        v[m] = mz["z"][mi[m]]
        out["MAREA (band clocks)"] = v
    else:
        print("NOTE: no MAREA product on disk yet "
              "(products_villaviciosa_marea) — row omitted")
    return out, transform


def main():
    campo = np.load("products_villaviciosa/campo.npz", allow_pickle=True)
    rr, cc, gnss = campo["row"], campo["col"], campo["gnss"]
    H = W = 915

    prods, tr = load_products(rr, cc, H, W)

    # ── the split, reproduced exactly (holdout.py's formula and seed) ────
    east = tr.c + (cc + 0.5) * tr.a
    north = tr.f + (rr + 0.5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    blocks = np.unique(block)
    rng = np.random.default_rng(SEED)
    n_hold = max(2, int(round(HOLDOUT_FRACTION * len(blocks))))
    hold_blocks = rng.choice(blocks, n_hold, replace=False)
    is_hold = np.isin(block, hold_blocks)
    print(f"{len(gnss)} RTK pixels in {len(blocks)} blocks of "
          f"{BLOCK_M:.0f} m -> {int((~is_hold).sum())} development / "
          f"{int(is_hold.sum())} RESERVED (untouched)")

    y = gnss[~is_hold]
    rows = []
    for name, v in prods.items():
        p = v[~is_hold]
        m = np.isfinite(p) & np.isfinite(y)
        if m.sum() < 20:
            rows.append({"method": name, "n": int(m.sum())})
            continue
        yc = y[m] - np.median(y[m])          # ellipsoidal vs MSL: constant
        pc = p[m] - np.median(p[m])
        rmse = float(np.sqrt(np.mean((pc - yc) ** 2)))
        slope = float(np.polyfit(yc, pc, 1)[0])
        rows.append({"method": name, "rmse_m": round(rmse, 3),
                     "slope": round(slope, 3), "n": int(m.sum())})

    print(f"\n{'DEVELOPMENT SPLIT':28s} {'RMSE':>7s} {'slope':>7s} "
          f"{'n':>5s}")
    print("-" * 52)
    for r in rows:
        if "rmse_m" in r:
            print(f"{r['method']:28s} {r['rmse_m']:7.3f} "
                  f"{r['slope']:7.3f} {r['n']:5d}")
        else:
            print(f"{r['method']:28s} {'—':>7s} {'—':>7s} {r['n']:5d}")

    # ── same pixels for everyone: the only apples-to-apples table ────────
    named = {k: v[~is_hold] for k, v in prods.items()}
    common = np.isfinite(y)
    for v in named.values():
        common &= np.isfinite(v)
    print(f"\nCOMMON SUBSET (finite in every method): {int(common.sum())} px")
    print(f"{'':28s} {'RMSE':>7s} {'slope':>7s}")
    print("-" * 46)
    yc = y[common] - np.median(y[common])
    common_rows = []
    for name, v in named.items():
        pc = v[common] - np.median(v[common])
        rmse = float(np.sqrt(np.mean((pc - yc) ** 2)))
        slope = float(np.polyfit(yc, pc, 1)[0])
        common_rows.append({"method": name, "rmse_m": round(rmse, 3),
                            "slope": round(slope, 3)})
        print(f"{name:28s} {rmse:7.3f} {slope:7.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/c1_dev_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"split": {"blocks": int(len(blocks)),
                             "reserved_blocks": int(n_hold),
                             "seed": SEED, "block_m": BLOCK_M},
                   "rows": rows,
                   "common_subset": {"n": int(common.sum()),
                                     "rows": common_rows}}, f, indent=1)
    print("\n-> results/c1_dev_comparison.json "
          "(the reserved half stays sealed)")


if __name__ == "__main__":
    main()
