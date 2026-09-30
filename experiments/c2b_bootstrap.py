"""C2b — are the boundary differences real? Paired block bootstrap.

Every variant was inverted on the same pixels, so the honest question is
paired: for each 150-m RTK block, resample blocks with replacement, recompute
each variant's RMSE and slope on the resample (median-centred), and report
the distribution of the DIFFERENCE against the EOT20 reference. A gain
whose 95 % interval straddles zero is noise, however tidy the point value.

Run:  python -m experiments.c2b_bootstrap
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def score(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    pc, tc = p[m] - np.median(p[m]), t[m] - np.median(t[m])
    return (float(np.sqrt(np.mean((pc - tc) ** 2))),
            float(np.polyfit(tc, pc, 1)[0]) if m.sum() > 3 else np.nan)


def main(n_boot=2000, seed=7):
    d = np.load("results/c2_preds.npz", allow_pickle=True)
    gnss, block, names, preds = d["gnss"], d["block"], list(d["names"]), d["preds"]
    ref = names.index("EOT20 (model)")
    blocks = np.unique(block)
    rng = np.random.default_rng(seed)
    idx_by_block = {b: np.flatnonzero(block == b) for b in blocks}
    rows = {}
    print(f"{len(gnss)} pixels in {len(blocks)} blocks; {n_boot} resamples")
    print(f"{'variant':40s} {'dRMSE (mm)':>22s} {'dslope':>22s}")
    for j, name in enumerate(names):
        drm, dsl = [], []
        for _ in range(n_boot):
            pick = rng.choice(blocks, len(blocks), replace=True)
            ii = np.concatenate([idx_by_block[b] for b in pick])
            r0, s0 = score(preds[ref][ii], gnss[ii])
            r1, s1 = score(preds[j][ii], gnss[ii])
            drm.append(r1 - r0); dsl.append(s1 - s0)
        drm, dsl = np.array(drm) * 1000, np.array(dsl)
        lo, hi = np.percentile(drm, [2.5, 97.5]); slo, shi = np.percentile(dsl, [2.5, 97.5])
        rows[name] = {"drmse_mm": [float(np.median(drm)), float(lo), float(hi)],
                      "dslope": [float(np.median(dsl)), float(slo), float(shi)]}
        flag = "real" if (hi < 0 or lo > 0) else "noise"
        print(f"{name:40s} {np.median(drm):+6.1f} [{lo:+6.1f},{hi:+6.1f}]  "
              f"{np.median(dsl):+.3f} [{slo:+.3f},{shi:+.3f}]  {flag if j != ref else 'ref'}")
    json.dump(rows, open("results/c2b_bootstrap.json", "w"), indent=1)
    print("-> results/c2b_bootstrap.json")


if __name__ == "__main__":
    main()
