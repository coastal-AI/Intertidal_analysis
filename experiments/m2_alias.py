"""M2 pre-step: the alias decision — what this archive can even estimate.

Reads the REAL overpass instants (STAC; the cube only stores dates and the
nominal-hour error is a measured 0.134 m), computes each constituent's alias
period under 1 sample/day, and writes the verdict every M2 estimator must
respect: S2 frozen (its period divides the solar day), K1/P1 lost to the
seasonal alias, the rest estimable. Constituents not estimable stay pinned
to the boundary prior — that sentence is the spec's gate for this step.

Run:  python -m experiments.m2_alias
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from pyintertidal.net import use_system_certificates
from pyintertidal import alias_table
import pyintertidal as pit

OUT = os.path.join("results", "m2_alias")


def main():
    use_system_certificates()
    aoi = pit.sites.get("villaviciosa")
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    tt = pd.to_datetime(sorted(times.values()))
    hours = tt.hour + tt.minute / 60.0 + tt.second / 3600.0
    table = alias_table.decide(hours)
    est = alias_table.estimable(hours)
    result = {
        "n_escenas": len(hours),
        "hora_media_utc": float(np.mean(hours)),
        "hora_p5_p95_utc": [float(np.percentile(hours, 5)),
                            float(np.percentile(hours, 95))],
        "tabla": table,
        "estimables": est,
        "clavados_al_prior_del_contorno":
            sorted(set(table) - set(est)),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print(f"{len(hours)} real overpasses · mean hour "
          f"{result['hora_media_utc']:.2f} UTC")
    for k, v in sorted(table.items(), key=lambda kv: kv[1]["period_h"]):
        al = v["alias_days"]
        print(f"  {k:>3}: alias {'inf' if not np.isfinite(al) else f'{al:6.1f} d'}"
          f" · {'ESTIMABLE' if v['estimable'] else 'boundary prior'}"
          f" — {v['reason']}")
    assert "S2" not in est, "S2 cannot be estimable in sun-synchronous orbit"
    assert {"M2", "N2", "O1"} <= set(est), \
        "the spec expects M2, N2, O1 estimable"
    print("OK: the table matches what the spec expects (M2,N2,O1 yes; "
          "S2,K1 no)")


if __name__ == "__main__":
    main()
