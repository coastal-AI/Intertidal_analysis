"""Gates of the bathymetry innovations — regression tests of the RECORDED
verdicts (B5, B6, B7). The B phases closed with two honest negatives and one
conservative product (PHASE_LOG 2026-08-19); these tests pin that recorded
state and its internal consistency, so a silent re-run cannot rewrite
history without failing here.

Run:  python -m tests.test_gates_b
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np


def _load(path):
    assert os.path.exists(path), f"missing {path}"
    return json.load(open(path, encoding="utf-8"))


def test_b5_negative_result_recorded():
    r = _load("results/b5_gate_sim/veredicto_v3.json")
    assert r["version"] == 3
    # the final verdict is NEGATIVE and must stay so barring a documented
    # re-run (e.g. a 10-year archive): the per-limb clock split is not
    # adoptable with 465 scenes under the matched-null threshold
    assert r["puerta"]["PASA"] is False
    assert r["puerta"]["control_ok"] is True, \
        "the null control must be clean for the negative to count"
    assert r["umbral_nulo_delta_oos"] > 0


def test_b6_operating_point_recorded():
    r = _load("results/b6_hydraulic_dem/result.json")
    # the flag that lights up is trustworthy (real judge vs matched null)
    assert r["juez_real"]["firma_observada"] > r["juez_real"]["nulo_p95"]
    # and the bounded recall is declared (self-hiding censoring)
    assert r["juez_simulacion"]["exhaustividad"] < 0.7
    assert os.path.exists("results/b6_hydraulic_dem/capas.npz")


def test_b7_conservative_intervals_recorded():
    r = _load("results/b7_incertidumbre/result.json")
    # the interval covers AT LEAST the nominal (conservative, never
    # optimistic)
    assert r["cobertura_dev"] >= 0.68
    # and is not absurd (covering everything would inform nothing)
    assert r["cobertura_dev"] <= 0.95
    tab = np.asarray(r["tabla_sigma_z"], float)
    assert np.isfinite(tab).sum() >= 8, "empty sigma_z table"
    assert os.path.exists("results/b7_incertidumbre/sigma_z.npz")


if __name__ == "__main__":
    test_b5_negative_result_recorded()
    print("OK  B5: negative result recorded with a clean control")
    test_b6_operating_point_recorded()
    print("OK  B6: flag validated on real data; bounded recall declared")
    test_b7_conservative_intervals_recorded()
    print("OK  B7: conservative intervals, never optimistic")
    print("B PHASES: recorded state is consistent")
