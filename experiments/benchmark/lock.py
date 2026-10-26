# -*- coding: utf-8 -*-
"""Validation v2 — the pre-registration lock LOCK.json (spec §9.8, §9.9).

Question
--------
What exactly is frozen before the first lag score (lock-A) and before the
elevation stage (lock-B), and what must exist before either may be written?

Pre-registered rules (spec §9.8, §9.9, §13.1)
---------------------------------------------
* ``lock`` refuses while ``git status --porcelain -- experiments/benchmark
  pyintertidal tests`` is not empty (a dirty tree could not be reproduced).
* lock-A needs ``<root>/attain.json`` (``pilot attain``, §9.8) and the gate
  verdicts of every site (``gates.read_gate`` for every gate of
  ``gates.GATES``); ``allow_missing`` names the ones the user accepts as
  missing ('attain', '<gate>' or '<site>:<gate>', a truth-error or scoring
  item by its name), and they are recorded as such.
* lock-A freezes: every FROZEN section sha (``config.frozen_shas``) and the
  settings registries' sections; the sha of every lag method card and of
  every lag adapter CONFIG; **every Tier-1 elevation method card** (V3: lock-A
  REFUSES while any is missing and lists them); ``plans/index.json`` (the
  design shas, FROZEN['design']); the development-judge list
  (``judges.development_list`` with the authors' extra codes); the L-P1
  comparison list (:func:`lp1_comparisons`: L16 only when it is planned at
  W3 at a judged site and the CMEMS credentials exist; also written at the
  top level as ``lp1_comparisons``, which the scorer reads); the
  forking-paths disclosure of §9.9 (the static list, each entry marked 'the
  authors complete before lock-A', plus one entry per pre-lock score of real
  10 m lag products that ``score --pre-lock`` recorded in
  ``scores/prelock_ledger.json``; lock-A REFUSES when an entry unblinded the
  confirmatory window unless ``unblind_amendment`` gives the reason, written
  to ``amendments``); per site the dom version and its freeze state,
  the D_truth id and the S0 sha; the truth-error model sha; C3's cross-tab
  cut-off p; the per-scene offset rule sha (``inputs.OFFSET_RULE``); the sha
  of the scoring and inference constants (``scoring.SCORING_V2``,
  ``inference.INFERENCE_V2``); the gate verdicts and attain.json.
* lock-B (§9.9) appends implementation-level items only: the code hashes of
  the elevation adapters, the gate and positive-control verdicts and the
  environment; it needs lock-A and cannot change anything lock-A froze.
* LOCK.json keeps an ``amendments`` list (:func:`amend`: date, section,
  reason, affected units); a written phase is never overwritten.

Nobody runs ``lock`` on real data except the user (plan, milestone V2).
"""
from __future__ import annotations

import glob
import importlib
import os
import subprocess

from experiments.benchmark import config

LOCK_FORMAT = "v2-lock-1"
LOCK_NAME = "LOCK.json"
GIT_PATHS = ("experiments/benchmark", "pyintertidal", "tests")

#: the forking-paths disclosure of §9.9: the truth-based results the authors
#: had seen, and the design choices each could have influenced
_CHOICES = ["P0", "the accuracy metric", "the views", "the margins", "TAU_BOUNDS", "C2",
            "the judges"]
FORKING_PATHS = tuple(
    {"item": item, "seen": seen, "could_influence": list(_CHOICES),
     "status": "listed by the spec (§9.9); the authors complete it before lock-A"}
    for item, seen in (
        ("results/v1_vaklodingen_validation.*", "Vaklodingen elevation validation v1"),
        ("results/v4_lag_external.*", "lags against external gauges (v4)"),
        ("results/v5_rtk_onsite.json", "RTK on-site comparison (v5)"),
        ("results/v6_elevation_external.*", "external elevation comparison (v6)"),
        ("results/v6_lidar_external.*", "LiDAR comparison (v6)"),
        ("results/c1_dev_comparison.json", "development comparison c1"),
        ("results/c2_boundary_variants.json", "boundary variants c2"),
        ("results/c5_validation_matrix.*", "validation matrix c5"),
        ("results/server_10m/", "the v7-v10 server runs"),
        ("results/method_comparison_provisional_20m_2026-10-01/",
         "provisional 20 m method comparison"),
        ("results/scene_rule_study_2026-09-30/", "scene-rule study"),
        ("results/lag_estimator_agreement_2026-10-06/", "lag-estimator agreement"),
        ("results/topographic_lag_scan_2026-10-07/", "topographic lag scan"),
        ("results/v13_benchmark_20m/", "v13 benchmark at 20 m"),
        ("p7_tabla_mareografos/", "the gauge table"),
        ("the Harlingen comparison", "the comparison that widened TAU_BOUNDS"),
    ))


class LockRefused(RuntimeError):
    """lock refused; ``reasons`` lists every missing or failing item."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("lock refused: " + "; ".join(self.reasons))


def lock_path(root):
    return os.path.join(root, LOCK_NAME)


def read_lock(root):
    from experiments.benchmark import product
    p = lock_path(root)
    return product.read_json(p) if os.path.exists(p) else None


def has_phase(root, phase):
    doc = read_lock(root)
    return bool(doc and (doc.get("phases") or {}).get(phase))


def git_porcelain(paths=GIT_PATHS, cwd=None):
    """``git status --porcelain -- <paths>`` output ('' = clean). A failing
    git raises RuntimeError (a lock needs a known tree)."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=cwd or config.ROOT,
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"git status failed: {e}") from None
    if r.returncode != 0:
        raise RuntimeError(f"git status failed: {r.stderr.strip()}")
    return r.stdout


def _sha(obj):
    from experiments.benchmark import queue_v2
    return queue_v2.sha_json(obj)


def _file_sha(path):
    from experiments.benchmark import product
    return product.sha256_file(path) if os.path.exists(path) else None


# ─────────────────────────────────────────────────────────────────────────────
#  what is collected
# ─────────────────────────────────────────────────────────────────────────────

def elevation_tier1_products():
    """The Tier-1 elevation products of the catalogue (designs.METHODS)."""
    from experiments.benchmark import designs
    out = []
    for m in designs.METHODS.values():
        if m.tier != "1" or not m.elevation:
            continue
        for p in m.products:
            if p.startswith("L"):                 # E0's L01, E3's L06: lag products
                continue
            out.append(p)
    return out


def elevation_cards():
    """``{product: card or None}`` of the Tier-1 elevation products: the
    methods package's ``card(pid)`` or ``CARDS`` when it has them (V3), else
    the lag registry (E0's products)."""
    from experiments.benchmark import lags
    try:
        methods = importlib.import_module("experiments.benchmark.methods")
    except ImportError:
        methods = None
    out = {}
    for pid in elevation_tier1_products():
        c = None
        if methods is not None:
            fn = getattr(methods, "card", None)
            if callable(fn):
                try:
                    c = fn(pid)
                except Exception:
                    c = None
            if c is None and isinstance(getattr(methods, "CARDS", None), dict):
                c = methods.CARDS.get(pid)
        if c is None and pid in lags.PRODUCT_METHOD:
            try:
                c = lags.card(pid)
            except Exception:
                c = None
        out[pid] = c
    return out


def lag_cards():
    """``(card shas, config shas, problems)`` of the lag registry."""
    from experiments.benchmark import lags
    cards = {pid: _sha(c) for pid, c in sorted(lags.all_cards().items())}
    configs = {}
    for m in sorted(lags.REGISTRY):
        try:
            configs[m] = _sha(lags.module_attrs(m)["CONFIG"])
        except Exception:
            configs[m] = None
    problems = {k: v for k, v in lags.check_all_cards().items() if v}
    return cards, configs, problems


def _packages(root, sites):
    from experiments.benchmark import package, product
    out = {}
    for site in sites:
        try:
            s2 = package.v2_site(site, root=root)
        except Exception as e:
            out[site] = {"error": f"{type(e).__name__}: {e}"}
            continue
        cur = package.current_dom(s2)
        sup = os.path.join(s2.dir, "support.npz")
        s0 = None
        if os.path.exists(sup):
            try:
                import numpy as np
                z = product.load_npz(sup)
                s0 = config.sha256_bytes(np.asarray(z["s0"], "<i8").tobytes())
            except Exception:
                s0 = None
        out[site] = {"dom": cur, "d_truth": package._d_truth_id(s2), "s0_sha256": s0}
    return out


def _truth_error(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "truth", "**", "truth_error*.json"),
                              recursive=True)):
        out[os.path.relpath(p, root).replace("\\", "/")] = _file_sha(p)
    return out


def _offset_rule_sha():
    from experiments.benchmark import inputs
    fn = getattr(inputs, "offset_rule_sha256", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    return _sha(inputs.OFFSET_RULE)


def _module_dict_sha(module, name):
    try:
        m = importlib.import_module(f"experiments.benchmark.{module}")
    except ImportError:
        return None
    d = getattr(m, name, None)
    return None if d is None else _sha(d)


def _gates(root, sites):
    try:
        from experiments.benchmark import gates
    except ImportError:
        return None, {}
    out = {}
    for site in sites:
        out[site] = {}
        for name in gates.GATES:
            try:
                out[site][name] = gates.read_gate(root, site, name)
            except Exception:
                out[site][name] = None
    return gates, out


def lp1_comparisons(root):
    """The L-P1 comparison list frozen at lock-A (spec §6.5): the nominal
    five of ``inference.INFERENCE_V2['lp1']`` (zero lag, the primary global
    model, the regional model L16, GM20, HD26-H), without L16 when the
    regional model is not available at lock-A: no judged site has a
    planned L16 row at the L-P1 window in ``plans/lag.csv`` (its archive
    does not cover the window, ``designs.applicable``), or the Copernicus
    Marine credentials are missing (``fetch.cmems.check_credentials``).
    Truth-free. The scorer reads the list from LOCK.json
    (``lp1_comparisons``) instead of deciding L16 by data availability.

    Returns
    -------
    dict
        ``{"comparisons": [...], "l16": {"available": bool, "reason": str,
        "planned_sites": [...]}}``.
    """
    import csv

    from experiments.benchmark import inference
    lp = inference.INFERENCE_V2["lp1"]
    comps = [str(c) for c in lp["comparisons"]]
    window = lp["window"]
    plan = os.path.join(root, "plans", "lag.csv")
    planned = []
    if os.path.exists(plan):
        with open(plan, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("method") == "L16" and r.get("window") == window
                        and r.get("status") == "planned"
                        and r.get("site") in lp["judged_sites"]):
                    planned.append(r["site"])
    planned = sorted(set(planned))
    if not planned:
        why = (f"no judged site has a planned L16 row at {window} in plans/lag.csv"
               if os.path.exists(plan) else "plans/lag.csv is missing (plan --design lag)")
        avail = False
    else:
        try:
            from experiments.benchmark.fetch import cmems
            cred = cmems.check_credentials()
        except Exception as e:                   # the toolbox module is optional
            cred = {"status": "credentials missing", "reason": f"{type(e).__name__}: {e}"}
        avail = cred.get("status") == "ok"
        why = (f"planned at {window} at {planned}; credentials {cred.get('status')}"
               + ("" if avail else f" ({cred.get('reason')})"))
    if not avail and "L16" in comps:
        comps.remove("L16")
    return {"comparisons": comps, "window": window,
            "l16": {"available": avail, "reason": why, "planned_sites": planned}}


#: the pre-lock scores the code itself records (``score --pre-lock``)
PRELOCK_LEDGER = os.path.join("scores", "prelock_ledger.json")
#: what a pre-lock look at real lag scores could have influenced (§9.9)
PRELOCK_CHOICES = ["the lag methods (cards, CONFIG)", "the L-P1 comparison list",
                   "the development judges", "the margins"]


def prelock_entries(root):
    """The entries of ``<root>/scores/prelock_ledger.json`` (``[]`` when
    there is none): every pre-lock score of real 10 m lag products."""
    from experiments.benchmark import product
    p = os.path.join(root, PRELOCK_LEDGER)
    if not os.path.exists(p):
        return []
    try:
        doc = product.read_json(p)
    except (OSError, ValueError):
        return []
    return list((doc or {}).get("entries") or [])


def prelock_forking_paths(root):
    """One forking-paths entry per pre-lock score (spec §9.9): what was
    seen and what it could have influenced."""
    out = []
    for e in prelock_entries(root):
        seen = (f"pre-lock lag scores ({e.get('invocation') or 'score --pre-lock'}) of "
                f"{e.get('windows_scored') or {}}; products {e.get('products_scored') or []}"
                + (f"; UNBLINDED {e.get('unblinded')}" if e.get("unblinded") else ""))
        out.append({"item": f"{PRELOCK_LEDGER.replace(os.sep, '/')} @ {e.get('utc')}",
                    "seen": seen, "could_influence": list(PRELOCK_CHOICES),
                    "status": "recorded by the code (score --pre-lock)", "ledger_entry": e})
    return out


def _allowed(item, allow):
    allow = set(allow or ())
    if item in allow:
        return True
    if ":" in item and item.split(":", 1)[1] in allow:
        return True
    return False


def collect(root, sites=None, extra_dev_judges=()):
    """Everything lock-A records (no refusal logic)."""
    from experiments.benchmark import judges, package, product, settings
    sites = list(sites or config.SITES)
    cards, configs, problems = lag_cards()
    elev = elevation_cards()
    idx_path = os.path.join(root, "plans", "index.json")
    attain_path = os.path.join(root, "attain.json")
    _, gate_docs = _gates(root, sites)
    ct_path = os.path.join(root, "common", package.CROSSTAB_JSON)
    return {
        "git": product.git_state(),
        "frozen_shas": config.frozen_shas(),
        "settings_shas": {k: _sha(v) for k, v in settings.frozen_sections().items()},
        "lag_cards": cards, "lag_configs": configs, "lag_card_problems": problems,
        "elevation_cards": {k: (None if v is None else _sha(v)) for k, v in elev.items()},
        "plans_index": product.read_json(idx_path) if os.path.exists(idx_path) else None,
        "development_judges": judges.development_list(extra_dev_judges),
        "lp1": lp1_comparisons(root),
        "forking_paths": [dict(f) for f in FORKING_PATHS] + prelock_forking_paths(root),
        "packages": _packages(root, sites),
        "truth_error": _truth_error(root),
        "crosstab": {"p": package.read_crosstab_p(root), "sha256": _file_sha(ct_path)},
        "offset_rule_sha256": _offset_rule_sha(),
        "scoring_sha256": _module_dict_sha("scoring", "SCORING_V2"),
        "inference_sha256": _module_dict_sha("inference", "INFERENCE_V2"),
        "gates": gate_docs,
        "attain": {"sha256": _file_sha(attain_path),
                   "doc": product.read_json(attain_path) if os.path.exists(attain_path)
                   else None},
        "sites": sites,
    }


def missing_items(doc, phase="A"):
    """``[(item id, message)]`` that block a lock (before ``allow_missing``)."""
    out = []
    if doc["attain"]["sha256"] is None:
        out.append(("attain", "attain.json is missing (run 'pilot attain', §9.8)"))
    if doc["gates"] in (None, {}):
        out.append(("gates", "gates.py is not built: no gate verdict can be read"))
    else:
        for site, gs in doc["gates"].items():
            for name, g in gs.items():
                if g is None:
                    out.append((f"{site}:{name}", f"gate {name} has no verdict at {site}"))
                elif g.get("verdict") == "FAIL":
                    out.append((f"{site}:{name}", f"gate {name} FAILED at {site}: "
                                                   f"{g.get('reason')}"))
    if doc["scoring_sha256"] is None:
        out.append(("scoring", "scoring.SCORING_V2 is not available"))
    if doc["inference_sha256"] is None:
        out.append(("inference", "inference.INFERENCE_V2 is not available"))
    if not doc["truth_error"]:
        out.append(("truth_error", "no truth-error model document (truth_error*.json) under "
                                   "truth/"))
    if doc["crosstab"]["p"] is None:
        out.append(("crosstab", "C3's cut-off p is not frozen (pilot crosstab)"))
    return out


def write_lock(phase, root, extra_dev_judges=(), allow_missing=(), sites=None,
               git_status=None, unblind_amendment=None):
    """Write phase ``phase`` ('A' or 'B') of ``<root>/LOCK.json``.

    Parameters
    ----------
    phase : "A" or "B"
    root : str
    extra_dev_judges : sequence of str
        The development judges the authors add to Harlingen (§1.7).
    allow_missing : sequence of str
        Items the user accepts as missing (recorded): 'attain', 'scoring',
        'inference', 'truth_error', 'crosstab', '<gate>' or '<site>:<gate>'.
        Missing Tier-1 elevation cards are never allowed (lock-A refuses).
    sites : sequence, optional
        Default every site.
    git_status : callable, optional
        Returns the porcelain text (tests); default :func:`git_porcelain`.
    unblind_amendment : str, optional
        lock-A after a pre-lock score of the confirmatory window (``score
        --unblind W3``, recorded in the pre-lock ledger): the reason, written
        as a §9.9 amendment. Without it such a lock-A is refused.

    Returns
    -------
    dict
        The LOCK.json document written.

    Raises
    ------
    LockRefused
    """
    from experiments.benchmark import product
    if phase not in ("A", "B"):
        raise ValueError(f"phase must be 'A' or 'B', not {phase!r}")
    porcelain = (git_status or git_porcelain)()
    if porcelain.strip():
        lines = [ln for ln in porcelain.splitlines() if ln.strip()]
        raise LockRefused([f"the tree is dirty in {', '.join(GIT_PATHS)} "
                           f"({len(lines)} path(s), e.g. {lines[0].strip()})"])
    old = read_lock(root) or {"format": LOCK_FORMAT, "phases": {}, "amendments": []}
    if (old.get("phases") or {}).get(phase):
        raise LockRefused([f"LOCK.json already holds phase {phase}: a written phase is never "
                           "overwritten (use lock.amend for a logged change)"])
    if phase == "B" and not (old.get("phases") or {}).get("A"):
        raise LockRefused(["lock-B needs lock-A"])
    reasons = []
    if phase == "A":
        doc = collect(root, sites, extra_dev_judges)
        miss_cards = [p for p, c in doc["elevation_cards"].items() if c is None]
        if miss_cards:
            reasons.append("Tier-1 elevation method cards missing (milestone V3): "
                           + ", ".join(miss_cards))
        allowed = []
        for item, msg in missing_items(doc, phase):
            if _allowed(item, allow_missing):
                allowed.append({"item": item, "message": msg})
            else:
                reasons.append(msg)
        if doc["lag_card_problems"]:
            bad = {k: v for k, v in doc["lag_card_problems"].items()
                   if not _allowed(f"card:{k}", allow_missing)}
            if bad:
                reasons.append("lag cards with problems: " + "; ".join(
                    f"{k}: {', '.join(v)}" for k, v in sorted(bad.items())))
        unblinded = [e for e in prelock_entries(root) if e.get("unblinded")]
        if unblinded and not unblind_amendment:
            wins = sorted({w for e in unblinded for w in e.get("unblinded") or []})
            reasons.append(f"the confirmatory window(s) {wins} were scored before lock-A "
                           f"({len(unblinded)} pre-lock ledger entr"
                           f"{'y' if len(unblinded) == 1 else 'ies'}, {PRELOCK_LEDGER}): pass "
                           "--amend-unblind '<reason>' to record the §9.9 amendment")
        if reasons:
            raise LockRefused(reasons)
        doc["allowed_missing"] = allowed
        if unblinded:
            doc["unblind_amendment"] = {
                "utc": product.utc_stamp(), "section": "§9.9 (lock-A after a pre-lock look "
                                                       "at the confirmatory window)",
                "reason": str(unblind_amendment),
                "affected": [{"utc": e.get("utc"), "unblinded": e.get("unblinded"),
                              "windows_scored": e.get("windows_scored"),
                              "products_scored": e.get("products_scored")}
                             for e in unblinded]}
    else:
        doc = _phase_b(root)
    doc["phase"] = phase
    doc["utc"] = product.utc_stamp()
    out = dict(old)
    out.setdefault("format", LOCK_FORMAT)
    out.setdefault("amendments", [])
    out.setdefault("phases", {})
    out["phases"][phase] = doc
    if phase == "A":
        # the scorer reads the frozen development judges and the frozen L-P1
        # comparison list at the top level too
        out["development_judges"] = list(doc["development_judges"])
        out["lp1_comparisons"] = list(doc["lp1"]["comparisons"])
        if doc.get("unblind_amendment"):
            out["amendments"].append(dict(doc["unblind_amendment"]))
    product.write_json(lock_path(root), out)
    return out


def _phase_b(root):
    from experiments.benchmark import product
    codes = {}
    for p in sorted(glob.glob(os.path.join(config.ROOT, "experiments", "benchmark", "methods",
                                           "*.py"))):
        codes[os.path.relpath(p, config.ROOT).replace("\\", "/")] = config.sha256_source(p)
    ctl = {}
    for p in sorted(glob.glob(os.path.join(root, "control", "*.json"))):
        try:
            ctl[os.path.basename(p)] = product.read_json(p)
        except (OSError, ValueError):
            ctl[os.path.basename(p)] = None
    _, gate_docs = _gates(root, list(config.SITES))
    return {"git": product.git_state(), "elevation_code": codes, "control": ctl,
            "gates": gate_docs, "environment": product.runtime_provenance()}


def amend(root, section, reason, affected_units=()):
    """Append an amendment (date, section, reason, affected units) to
    LOCK.json; every affected method reruns (spec §9.9, v13 §2.11)."""
    from experiments.benchmark import product
    doc = read_lock(root)
    if doc is None:
        raise LockRefused(["no LOCK.json to amend"])
    doc.setdefault("amendments", []).append({"utc": product.utc_stamp(), "section": section,
                                             "reason": reason,
                                             "affected_units": list(affected_units)})
    product.write_json(lock_path(root), doc)
    return doc
