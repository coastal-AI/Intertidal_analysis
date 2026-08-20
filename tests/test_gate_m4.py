"""Gate M4: the operator bites, does no harm at the mouth, and contracts.

Reads results/m4_gate_sim/result.json (python -m experiments.m4_gate_sim)
and asserts against configs/m4.yaml: at least the pre-registered fraction of
the planted phase damage recovered in the interior bands, no degradation at
the anchor, and the fixed-point refit returns identity within tolerance.

Run:  python -m tests.test_gate_m4
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml

CFG = yaml.safe_load(open("configs/m4.yaml", encoding="utf-8"))


def test_m4_gate():
    p = "results/m4_gate_sim/result.json"
    assert os.path.exists(p), "run first: python -m experiments.m4_gate_sim"
    r = json.load(open(p, encoding="utf-8"))
    assert r["fraccion_dano_recuperado"] \
        >= CFG["puerta"]["min_fraccion_dano_recuperado"], \
        f"only {100*r['fraccion_dano_recuperado']:.0f} % of damage recovered"
    mouth_delta = (r["rmse_z"]["operador"][0] - r["rmse_z"]["uniforme"][0])
    assert mouth_delta <= CFG["puerta"]["max_degradacion_boca_m"], \
        f"the mouth degrades by {mouth_delta:.3f} m"
    assert r["contraccion"]["max_tau_residual_min"] \
        <= CFG["puerta"]["max_tau_residual_min"], "not a fixed point"
    assert r["puerta"]["PASA"]
    assert os.path.exists("results/m4_gate_sim/figure.png")


if __name__ == "__main__":
    test_m4_gate()
    print("OK  gate M4: it bites, does no harm at the mouth, and contracts")
    print("GATE M4: GREEN")
