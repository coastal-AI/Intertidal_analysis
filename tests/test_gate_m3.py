"""Gate M3: a planted boundary error lands in gamma-hat, not in T.

Reads results/m3_gate_sim/result.json (python -m experiments.m3_gate_sim)
and asserts against configs/m3.yaml: the mouth judge recovered the planted
(gamma, dt) within tolerance AND the interior operator stayed flat once the
boundary was corrected — the error did not leak into fictitious estuarine
physics.

Run:  python -m tests.test_gate_m3
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

CFG = yaml.safe_load(open("configs/m3.yaml", encoding="utf-8"))


def test_m3_gate():
    p = "results/m3_gate_sim/result.json"
    assert os.path.exists(p), "run first: python -m experiments.m3_gate_sim"
    r = json.load(open(p, encoding="utf-8"))
    assert r.get("version_puerta") == 3, "stale result: re-run"
    dt_pl = CFG["plantado"]["dt_m2_min"]
    assert r["plantado"]["dt_m2_min"] == dt_pl
    # v3: the boundary error's phase is recovered; gamma is not corrected
    # (affine theorem) and its planted presence must not fake phase
    assert abs(r["recuperado"]["dt_min"] - dt_pl) \
        <= CFG["puerta"]["tol_dt_min"]
    assert r["recuperado"]["gamma"] == 0.0, "v3 does not correct gain"
    assert r["recuperado"]["adopted"], \
        "the OOS judge failed to adopt a correction that was real"
    assert r["residuales"]["rms_tau_min"] \
        <= CFG["puerta"]["max_tau_residual_rms_min"], "leak into the operator"
    assert r["T_sin_corregir_fuga"]["max_min"] \
        >= CFG["puerta"]["min_fuga_sin_corregir_min"], \
        "the leak does not appear uncorrected: the gate proves nothing"
    assert r["puerta"]["PASA"]
    assert os.path.exists("results/m3_gate_sim/figure.png")


if __name__ == "__main__":
    test_m3_gate()
    print("OK  gate M3 (v3): the boundary error's PHASE lands in dt-hat,"
          " not in T; gain stays with the prior and does not fake phase")
    print("GATE M3: GREEN")
