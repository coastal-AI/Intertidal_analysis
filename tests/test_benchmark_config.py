"""Tests of experiments.benchmark.config: the confirmatory families can reject.

Validation v2 spec §9.1 and §9.8 (defect 5 of §12.3, milestone V0): a family
of m view-specific statistics controlled at alpha = 0.05 can reject only if
its smallest attainable p is below Holm's first threshold alpha / m. For a
two-sided bootstrap p with B draws that minimum is 2 / (B + 1); for an exact
two-sided sign-flip test over n locations without zero differences it is
2 / 2^n. Revision 1's B = 5,000 gave 4.0e-4 against 0.05 / 128 = 3.9e-4 for
P2, so P2 could never reject; B = 20,000 gives 1.0e-4.

Run:  python -X utf8 -m pytest tests/test_benchmark_config.py -q
"""
import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config  # noqa: E402


def test_b_primary_is_20000_and_descriptive_1000():
    boot = config.FROZEN["inference"]["boot"]
    assert boot["B_primary"] == 20000
    assert boot["B_descriptive"] == 1000


def test_frozen_family_sizes_are_the_spec_products():
    """§9.2: P1 = 4 sites x 2 quantities x 2 views; P2 = 8 competitors x 4
    sites x 2 x 2; P3 = 7 competitors x (3 + 4) x 2 views; L-P1 five
    comparisons per truth, L-P2 one; 13 headline locations (§1.7)."""
    fam = config.FROZEN["inference"]["families"]
    assert dict(fam["bootstrap"]) == {"P1": 4 * 2 * 2, "P2": 8 * 4 * 2 * 2,
                                      "P3": 7 * (3 + 4) * 2}
    assert dict(fam["sign_flip"]) == {"L-P1": 5, "L-P2": 1}
    assert fam["headline_locations"] == 13


def test_every_frozen_family_is_attainable():
    att = config.attainability()
    assert set(att) == {"P1", "P2", "P3", "L-P1", "L-P2"}
    alpha = config.FROZEN["inference"]["holm_alpha"]
    for name, r in att.items():
        assert r["holm_first"] == pytest.approx(alpha / r["m"])
        assert r["attainable"], (name, r)
        assert r["p_min"] < r["holm_first"], name
    B = config.FROZEN["inference"]["boot"]["B_primary"]
    for name in ("P1", "P2", "P3"):
        assert att[name]["kind"] == "bootstrap"
        assert att[name]["p_min"] == 2.0 / (B + 1)
    for name in ("L-P1", "L-P2"):
        assert att[name]["kind"] == "sign_flip"
        assert att[name]["p_min"] == 2.0 / 2.0 ** 13
    # the largest family is the binding one: 2/20001 < 0.05/128
    assert 2.0 / (20000 + 1) < 0.05 / 128


def test_revision_1_b_made_p2_unattainable():
    inf = config.frozen_plain("inference")["inference"]
    inf = copy.deepcopy(inf)
    inf["boot"]["B_primary"] = 5000
    att = config.attainability(inf)
    assert not att["P2"]["attainable"]
    assert att["P2"]["p_min"] == pytest.approx(4.0e-4, rel=1e-3)
    assert att["P1"]["attainable"] and att["P3"]["attainable"]
    # the smallest B that makes the largest family attainable
    m = max(inf["families"]["bootstrap"].values())
    b_min = next(b for b in range(5000, 30000) if 2.0 / (b + 1) < 0.05 / m)
    assert b_min <= config.FROZEN["inference"]["boot"]["B_primary"]


def test_sign_flip_with_too_few_locations_is_unattainable():
    inf = copy.deepcopy(config.frozen_plain("inference")["inference"])
    inf["families"]["headline_locations"] = 7          # 2/128 = 0.0156 > 0.05/5
    att = config.attainability(inf)
    assert not att["L-P1"]["attainable"] and att["L-P2"]["attainable"]
