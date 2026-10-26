# -*- coding: utf-8 -*-
"""Validation v2: the pre-registration lock (``experiments/benchmark/lock.py``;
spec §9.8, §9.9). Code only: nobody but the user locks real data."""
import json
import os

import pytest

from experiments.benchmark import config, judges, product
from experiments.benchmark import lock as L

def _clean():
    return ""


def _fake_world(monkeypatch, root, cards=True, gates_ok=True):
    """Everything lock-A reads, present (stand-ins for the other packages)."""
    product.write_json(os.path.join(root, "attain.json"), {"families": {"L-P1": {"m": 5}}})
    product.write_json(os.path.join(root, "truth", "truth_error.json"), {"sha256": "x"})
    product.write_json(os.path.join(root, "common", "crosstab.json"), {"p": 20})
    product.write_json(os.path.join(root, "plans", "index.json"),
                       {"format": "validation-v2-plan/1", "plans": {"lag": {"sha256": "abc"}}})
    monkeypatch.setattr(L, "elevation_cards", lambda: {p: ({"id": p} if cards else None)
                                                      for p in ("E0a", "E0b", "E1")})
    monkeypatch.setattr(L, "lag_cards", lambda: ({"L02": "s1"}, {"L02": "c1"}, {}))
    verdict = "PASS" if gates_ok else "FAIL"
    monkeypatch.setattr(L, "_gates", lambda root, sites: (object(), {
        s: {"record": {"verdict": verdict, "reason": "r"}, "keep": {"verdict": "PASS"}}
        for s in sites}))
    monkeypatch.setattr(L, "_module_dict_sha", lambda module, name: f"sha-{module}")


def test_lock_refuses_a_dirty_tree(tmp_path):
    with pytest.raises(L.LockRefused, match="dirty"):
        L.write_lock("A", str(tmp_path), git_status=lambda: " M experiments/benchmark/x.py\n")
    assert not os.path.exists(L.lock_path(str(tmp_path)))


def test_lock_refuses_a_missing_attain_and_missing_gates(tmp_path, monkeypatch):
    root = str(tmp_path)
    _fake_world(monkeypatch, root)
    os.remove(os.path.join(root, "attain.json"))
    with pytest.raises(L.LockRefused) as ei:
        L.write_lock("A", root, git_status=_clean, sites=["ems"])
    assert any("attain.json is missing" in r for r in ei.value.reasons)
    monkeypatch.setattr(L, "_gates", lambda root, sites: (object(), {"ems": {"record": None}}))
    with pytest.raises(L.LockRefused) as ei:
        L.write_lock("A", root, git_status=_clean, sites=["ems"], allow_missing=["attain"])
    assert ei.value.reasons == ["gate record has no verdict at ems"]
    doc = L.write_lock("A", root, git_status=_clean, sites=["ems"],
                       allow_missing=["attain", "ems:record"])
    allowed = {a["item"] for a in doc["phases"]["A"]["allowed_missing"]}
    assert allowed == {"attain", "ems:record"}


def test_lock_a_refuses_while_an_elevation_card_is_missing(tmp_path, monkeypatch):
    root = str(tmp_path)
    _fake_world(monkeypatch, root, cards=False)
    with pytest.raises(L.LockRefused) as ei:
        L.write_lock("A", root, git_status=_clean, sites=["ems"],
                     allow_missing=["E0a", "card:E0a", "E1"])
    msg = " ".join(ei.value.reasons)
    assert "Tier-1 elevation method cards missing" in msg and "E1" in msg and "E0a" in msg


def test_lock_a_refuses_a_failed_gate(tmp_path, monkeypatch):
    root = str(tmp_path)
    _fake_world(monkeypatch, root, gates_ok=False)
    with pytest.raises(L.LockRefused, match="FAILED"):
        L.write_lock("A", root, git_status=_clean, sites=["ems"])


def test_the_real_catalogue_lists_the_tier1_elevation_products():
    prods = L.elevation_tier1_products()
    for p in ("E0a", "E0b", "E1", "E2", "E4", "E5a", "E6a", "E6b", "E7", "E8"):
        assert p in prods
    assert "L01" not in prods and "L06" not in prods
    cards = L.elevation_cards()
    assert set(cards) == set(prods)


def test_lock_writes_phase_a_then_b_and_amendments(tmp_path, monkeypatch):
    root = str(tmp_path)
    _fake_world(monkeypatch, root)
    doc = L.write_lock("A", root, extra_dev_judges=("delfzijl",), git_status=_clean,
                       sites=["escalda", "ems"])
    a = doc["phases"]["A"]
    for k in ("git", "frozen_shas", "settings_shas", "lag_cards", "lag_configs",
              "elevation_cards", "plans_index", "development_judges", "forking_paths",
              "packages", "truth_error", "crosstab", "offset_rule_sha256", "scoring_sha256",
              "inference_sha256", "gates", "attain", "utc"):
        assert k in a, k
    assert a["frozen_shas"] == config.frozen_shas()
    assert a["development_judges"] == judges.development_list(("delfzijl",))
    assert "harlingen.waddenzee" in a["development_judges"]
    assert len(a["forking_paths"]) == len(L.FORKING_PATHS)
    assert all("authors complete" in f["status"] for f in a["forking_paths"])
    assert a["crosstab"]["p"] == 20 and a["plans_index"]["plans"]["lag"]["sha256"] == "abc"
    assert set(a["packages"]) == {"escalda", "ems"} and a["packages"]["ems"]["dom"] is None
    assert a["attain"]["doc"]["families"]["L-P1"]["m"] == 5
    # no plans/lag.csv: L16 is not available at lock-A, the family has 4 members
    assert a["lp1"]["comparisons"] == ["L17", "L14", "L08", "L09"]
    assert doc["lp1_comparisons"] == a["lp1"]["comparisons"]
    assert L.has_phase(root, "A") and not L.has_phase(root, "B")
    with pytest.raises(L.LockRefused, match="never overwritten"):
        L.write_lock("A", root, git_status=_clean, sites=["ems"])
    product.write_json(os.path.join(root, "control", "L02.json"), {"gating": "PASS"})
    doc = L.write_lock("B", root, git_status=_clean)
    b = doc["phases"]["B"]
    assert b["control"]["L02.json"]["gating"] == "PASS" and "environment" in b
    assert doc["phases"]["A"]["utc"] == a["utc"]              # phase A untouched
    doc = L.amend(root, "methods.L08", "card typo", ["u1", "u2"])
    assert doc["amendments"][-1]["affected_units"] == ["u1", "u2"]
    on_disk = json.load(open(L.lock_path(root), encoding="utf-8"))
    assert set(on_disk["phases"]) == {"A", "B"} and len(on_disk["amendments"]) == 1


def test_lp1_comparisons_freeze_l16_only_when_available(tmp_path, monkeypatch):
    """Spec §6.5: L16 is in L-P1 only when its archive covers W3 at a judged
    site (a planned row in plans/lag.csv) and the CMEMS credentials exist;
    the scorer reads the frozen list from LOCK.json."""
    from experiments.benchmark import scoring
    from experiments.benchmark.fetch import cmems
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "plans"))
    rows = [("escalda", "W3", "L16", "not_applicable"), ("ferrol", "W3", "L16", "planned"),
            ("ferrol", "W3", "L08", "planned"), ("villaviciosa", "W3", "L16", "planned")]
    with open(os.path.join(root, "plans", "lag.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("site,window,method,status\n" + "".join(f"{','.join(r)}\n" for r in rows))
    monkeypatch.setattr(cmems, "check_credentials", lambda *a, **k: {"status": "ok"})
    out = L.lp1_comparisons(root)
    assert out["comparisons"] == ["L17", "L14", "L16", "L08", "L09"]
    assert out["l16"]["available"] and out["l16"]["planned_sites"] == ["ferrol"]
    monkeypatch.setattr(cmems, "check_credentials", lambda *a, **k: {
        "status": "credentials missing", "reason": "no file"})
    out = L.lp1_comparisons(root)
    assert out["comparisons"] == ["L17", "L14", "L08", "L09"] and not out["l16"]["available"]
    assert "credentials missing" in out["l16"]["reason"]
    # the scorer's confirmatory family takes the frozen list from LOCK.json,
    # even where L16 rows have finite errors
    sites = ["escalda", "wadden", "ems", "ferrol"]
    points = [{"site": s, "window": "W3", "judge": f"{s}{k}", "location": f"{s}{k}",
               "product": p, "truth": T, "status": "ok", "validated": True,
               "e": 3.0 + k + i, "abs_e": 3.0 + k + i}
              for s in sites for k in range(2) for i, p in
              enumerate(("L01", "L17", "L14", "L16", "L08", "L09")) for T in ("a", "ab", "b")]
    scored = {(s, "W3") for s in sites}
    lock_doc = {"phases": {"A": {"utc": "x"}}, "lp1_comparisons": out["comparisons"]}
    _, info = scoring.lp1_family(points, scored, lock_doc=lock_doc)
    assert info["confirmatory"] and info["comparisons"] == ["L17", "L14", "L08", "L09"]
    _, info = scoring.lp1_family(points, scored, lock_doc={"phases": {"A": {"utc": "x"}}})
    assert "L16" in info["comparisons"]                 # without the frozen list


def test_lock_b_needs_lock_a(tmp_path):
    with pytest.raises(L.LockRefused, match="needs lock-A"):
        L.write_lock("B", str(tmp_path), git_status=_clean)
    with pytest.raises(ValueError):
        L.write_lock("C", str(tmp_path), git_status=_clean)
    with pytest.raises(L.LockRefused, match="no LOCK.json"):
        L.amend(str(tmp_path), "x", "y")


def test_git_porcelain_runs_on_the_repository():
    out = L.git_porcelain(paths=("experiments/benchmark/config.py",))
    assert isinstance(out, str)


def test_lock_a_discloses_the_pre_lock_scores_and_needs_an_amendment_after_an_unblind(
        tmp_path, monkeypatch):
    """Every pre-lock score of real 10 m lag products (scores/
    prelock_ledger.json) enters LOCK.json's forking paths; one that unblinded
    the confirmatory window makes lock-A refuse until the user records the
    §9.9 amendment (``unblind_amendment``)."""
    root = str(tmp_path)
    _fake_world(monkeypatch, root)
    entries = [{"utc": "20261020T200000Z", "invocation": "score --pre-lock",
                "windows_scored": {"ems": ["Y2023"]}, "products_scored": ["L01", "L17"],
                "unblinded": [], "blinded": [{"site": "wadden", "window": "W3"}]},
               {"utc": "20261021T080000Z", "invocation": "score --pre-lock --unblind W3",
                "windows_scored": {"wadden": ["W3"]}, "products_scored": ["L01"],
                "unblinded": ["W3"], "blinded": []}]
    product.write_json(os.path.join(root, L.PRELOCK_LEDGER),
                       {"format": "v2-prelock-ledger-1", "entries": entries})
    with pytest.raises(L.LockRefused) as ei:
        L.write_lock("A", root, git_status=_clean, sites=["ems"])
    assert any("--amend-unblind" in r and "['W3']" in r for r in ei.value.reasons)
    doc = L.write_lock("A", root, git_status=_clean, sites=["ems"],
                       unblind_amendment="the user scored W3 at Wadden on 2026-10-21")
    fp = doc["phases"]["A"]["forking_paths"]
    assert len(fp) == len(L.FORKING_PATHS) + 2
    assert fp[-1]["ledger_entry"]["unblinded"] == ["W3"] and "UNBLINDED" in fp[-1]["seen"]
    assert "the L-P1 comparison list" in fp[-2]["could_influence"]
    am = doc["amendments"][-1]
    assert am["reason"].startswith("the user scored W3") and am["affected"][0]["unblinded"] == ["W3"]


def test_lock_a_without_pre_lock_scores_has_the_static_list_only(tmp_path, monkeypatch):
    root = str(tmp_path)
    _fake_world(monkeypatch, root)
    assert L.prelock_entries(root) == [] and L.prelock_forking_paths(root) == []
    doc = L.write_lock("A", root, git_status=_clean, sites=["ems"])
    assert len(doc["phases"]["A"]["forking_paths"]) == len(L.FORKING_PATHS)
    assert doc["amendments"] == []
