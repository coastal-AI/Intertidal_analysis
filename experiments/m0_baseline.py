"""M0 baseline: re-measure the per-block structure the whole plan targets.

Gate M0 of plan v4 (first half). The plan exists to explain a spatially
structured compression of recovered relief, so before any mechanism is
touched the pipeline must demonstrate it still measures that structure —
otherwise every later comparison floats on air.

What this reproduces, and what it deliberately does not:

* reproduced: the DEV-side structure — pooled slope of product elevation on
  surveyed elevation over the development blocks, with per-block slopes and
  bootstrap intervals. Historical reference at adoption: pooled dev slope
  0.698 [0.608, 0.782] on the product-matched subset.
* NOT reproduced: the reserved-side figure (0.890). Reading the reserved
  survey to check it would violate hard rule R1; that number stays a cited
  historical fact until experiment V3 (run once, by the human).

Partition note (R7): slopes here use the CANONICAL partition of
``pyintertidal.rtk`` (all 361 FIX points; dev 270). The historical 0.698 was
measured on the 192 product-matched points, a subset. The gate therefore
checks (a) the canonical dev pooled slope falls inside the historical CI,
and (b) the measured-at-adoption value stored in ``configs/m0.yaml`` is
reproduced to ±tolerance — self-consistency for every future run.

Outputs ``results/m0_baseline/result.json`` citing the SHA256 of every input
(R7). Run:  python -m experiments.m0_baseline
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import rasterio

from pyintertidal import seal
from pyintertidal.rtk import load_rtk

PRODUCT = "products_villaviciosa/hsr_eot20_2023-2025.tif"
N_BOOT = 4000
SEED_BOOT = 11
MIN_PER_BLOCK = 8
OUT = os.path.join("results", "m0_baseline")


def boot_slope(y, v, rng, n=N_BOOT):
    # bootstrap over points (pairs resampled together): CI of the slope
    idx = rng.integers(0, len(y), (n, len(y)))
    sl = np.array([np.polyfit(y[i], v[i], 1)[0] for i in idx])
    return float(np.polyfit(y, v, 1)[0]), np.percentile(sl, [2.5, 97.5])


def main():
    dev = load_rtk()                          # R1: dev only, guard enforced
    with rasterio.open(PRODUCT) as s:
        zhat = s.read(1)[dev["row"], dev["col"]]
    m = np.isfinite(zhat) & np.isfinite(dev["elev"])
    y, v, blk = dev["elev"][m], zhat[m], dev["block"][m]
    print(f"dev: {len(y)} points matched to the product · "
          f"{dev['n_reserved_hidden']} reserved NEVER read")

    rng = np.random.default_rng(SEED_BOOT)
    pooled, ci = boot_slope(y, v, rng)
    print(f"pooled dev slope: {pooled:.3f}  "
          f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

    rows = []
    for b in np.unique(blk):
        s_ = blk == b
        if s_.sum() < MIN_PER_BLOCK:
            continue
        sl, cib = boot_slope(y[s_], v[s_], rng, 1500)
        rows.append({"block": int(b), "n": int(s_.sum()),
                     "slope": sl, "ci": cib.tolist()})
        print(f"  block {b:5d}  n={s_.sum():3d}  slope {sl:+.3f} "
              f"[{cib[0]:+.3f}, {cib[1]:+.3f}]")
    spread = (max(r["slope"] for r in rows) - min(r["slope"] for r in rows)
              if len(rows) >= 2 else 0.0)
    print(f"blocks with {MIN_PER_BLOCK}+ points: {len(rows)} · "
          f"slope spread {spread:.3f}")

    os.makedirs(OUT, exist_ok=True)
    result = {
        "pooled_dev_slope": pooled, "pooled_ci": ci.tolist(),
        "blocks": rows, "spread": spread,
        "n_dev": int(len(y)), "n_reserved_hidden": dev["n_reserved_hidden"],
        "inputs": {
            "product": PRODUCT,
            "product_sha256": seal._sha256(PRODUCT),
            "store_seals": [e for e in seal._load_registry()
                            if e["file"].startswith("data_v4/store/")],
        },
        "seeds": {"partition": 20260817, "bootstrap": SEED_BOOT},
    }
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"written {OUT}/result.json")


if __name__ == "__main__":
    main()
