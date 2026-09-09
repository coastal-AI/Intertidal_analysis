"""V3 — open the reserved RTK. ONLY THE HUMAN RUNS THIS. ONCE.

This is the only module in the repository authorised to pass
``allow_reserved=True`` (rule R1; the dynamic guard in rtk.py verifies this
file's name and tests/test_r1_guard.py scans that nobody else does it).
Running it consumes the reserved set forever: after looking, no number
drawn from it can ever again count as validation.

STANDING RECORD (rtk.py, 2026-08-18): the current reserved partition was
opened several times that day BEFORE the v4 spec was adopted, so it is
BURNED for publication-grade claims. The V3 verdicts of this campaign move
to the blocks of a future campaign. This script remains the correct tool
for that day.

Usage (human, deliberate):

    V3_CONFIRMO=SI python -m experiments.v3_open_reserved <product.tif>

Without the environment variable, it prints the record and exits without
touching anything.
"""
import json
import os
import sys
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np


def main():
    if os.environ.get("V3_CONFIRMO") != "SI":
        print(__doc__)
        print("V3_CONFIRMO is not 'SI': nothing is opened. (Correct by "
              "default.)")
        return

    import rasterio
    from pyintertidal import rtk

    product = sys.argv[1] if len(sys.argv) > 1 else rtk.DEFAULT_GRID
    _dev, res = rtk.load_rtk(allow_reserved=True)
    with rasterio.open(product) as s:
        arr = s.read(1)
    def score(sub):
        z = arr[sub["row"], sub["col"]]
        ok = np.isfinite(z) & (z > -100)
        if ok.sum() < 5:
            return {"n": int(ok.sum())}
        e = z[ok] - sub["elev"][ok]
        e = e - np.median(e)          # datum: RTK is ellipsoidal, the
                                      # product sits on the tide datum
        return {"n": int(ok.sum()),
                "pendiente": float(np.polyfit(sub["elev"][ok], z[ok], 1)[0]),
                "rmse_centrado": float(np.sqrt(np.mean(e ** 2)))}

    both = {k: np.concatenate([_dev[k], res[k]])
            for k in ("row", "col", "elev")}
    rec = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"),
           "producto": product,
           "reservado": score(res), "dev": score(_dev),
           "todos_361": score(both),
           "nota": "V3 opening recorded; the reserved set is consumed"}
    log = os.path.join("results", "v3_aperturas.jsonl")
    os.makedirs("results", exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
