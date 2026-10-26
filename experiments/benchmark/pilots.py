# -*- coding: utf-8 -*-
"""Validation v2 — ``pilot reps``: the truth-free computability of the curve points (spec §5.10, §9.8).

Question
--------
Before any curve unit runs, how many scenes and clear observations does
every (site, N, rep) point of a curve design give, which harness domain does
its A-mode subset produce, and which lag methods are expected to be
computable on it, so the k/R >= 0.5 rule of the curves (Appendix A
``curves``) can be anticipated without reading any truth?

Pre-registered rules (spec §2.8, §4.2, §5.10; Appendix A ``curves``)
-------------------------------------------------------------------
* Points: the rows of ``plans/<design>.csv`` (default ``curves-k1``; the
  design is built in memory when its plan is not written), one per (site,
  window, setting, family, mode, N, rep); the scenes are the queue's
  (:func:`instance.instance_selection`).
* n_clear: the valid-observation counts of the point's setting at the
  keep_full(P0, W3) pixels (``pkg.count``), on at most 200,000 of them drawn
  without replacement with ``default_rng([20261008, SITE_ID])`` (declared
  subsample); median and 10th percentile.
* The A-mode domain of the subset (``domain.get_v2``): status, n_keep, K.
* Expected computability (truth-free, from each card's ``n_min`` when the
  card exists, else these rules): E0 / L01 / L01v the MAREA floor of 40
  scenes and at least 3,000 keep pixels (domain status 'ok'); m2b (L02) a
  median of at least 10 clear observations; m2c (L03) 60 scenes; XC (L04)
  30 usable scenes (approximated by the scene count, declared); LOOP (L07)
  30 scenes and 15 per tidal limb; HD26-H (L09) 30 scenes over at least 365
  days; HD26-R2 (L10) 60 scenes; GM20, PPL and BT26 (L08, L11, L12) a
  median of at least 20 valid observations per pixel; the model, physics and
  reference rows read no image (always computable here).
* k/R: per (site, window, N, product) the share of the replicates expected
  computable; flagged when below 0.5 (the curve point would be dropped).

Output: ``<root>/common/pilot_reps.json`` and a printed table.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from experiments.benchmark import config, settings

PILOT_FORMAT = "v2-pilot-reps-1"
MAX_PX = 200_000
#: the built-in expected-computability rules (spec §3.3.1, §4.2, §5.8)
RULES = {
    "E0": {"min_scenes": 40, "min_keep": 3000},
    "L01v": {"min_scenes": 40, "min_keep": 3000},
    "L02": {"min_clear_median": 10},
    "L03": {"min_scenes": 60},
    "L04": {"min_scenes": 30},
    "L07": {"min_scenes": 30, "min_per_limb": 15},
    "L08": {"min_clear_median": 20},
    "L09": {"min_scenes": 30, "min_span_days": 365},
    "L10": {"min_scenes": 60},
    "L11": {"min_clear_median": 20},
    "L12": {"min_clear_median": 20},
}
NO_IMAGE = ("L14", "L15", "L16", "L17", "L18", "L20")
KR_MIN = float(config.FROZEN["curves"]["min_computable_share"])


def log(msg):
    print(msg, flush=True)


def _rows(root, design, sites):
    from experiments.benchmark import designs, queue_v2
    p = queue_v2.plan_path(root, design)
    if os.path.exists(p):
        rows = queue_v2.read_plan(root, design)
    else:
        rows = designs.plan_rows(design, sites=sites)
    return [r for r in rows if r.get("status") == "planned" and r.get("site") in sites]


def _card_n_min(product):
    from experiments.benchmark import lags
    try:
        c = lags.card(product)
    except Exception:
        return None
    v = c.get("n_min") if isinstance(c, dict) else None
    return int(v) if isinstance(v, (int, float)) and v else None


def expected(product, method, n_scenes, n_keep, status, clear_median, n_rise, n_fall,
             span_days):
    """``(computable, reason)`` of one product on one point (truth-free)."""
    if product in NO_IMAGE or method in NO_IMAGE:
        return True, "reads no image"
    rule = dict(RULES.get(product, RULES.get(method, {})))
    nm = _card_n_min(product)
    if nm is not None:
        rule["min_scenes"] = nm
    why = []
    if "min_scenes" in rule and n_scenes < rule["min_scenes"]:
        why.append(f"{n_scenes} scenes < {rule['min_scenes']}")
    if "min_keep" in rule and (status != "ok" or n_keep < rule["min_keep"]):
        why.append(f"domain {status}, {n_keep} keep px < {rule['min_keep']}")
    if "min_clear_median" in rule and not (clear_median >= rule["min_clear_median"]):
        why.append(f"median n_clear {clear_median} < {rule['min_clear_median']}")
    if "min_per_limb" in rule and min(n_rise, n_fall) < rule["min_per_limb"]:
        why.append(f"limbs {n_rise}/{n_fall} < {rule['min_per_limb']}")
    if "min_span_days" in rule and span_days < rule["min_span_days"]:
        why.append(f"span {span_days:.0f} d < {rule['min_span_days']}")
    if status != "ok" and method not in NO_IMAGE and "min_keep" not in rule:
        why.append(f"domain {status}")
    return (not why), "; ".join(why) or "ok"


def pilot_reps(site2s, design="curves-k1", root=None, max_px=MAX_PX, write=True):
    """The truth-free table of every curve point of ``design`` at the sites.

    Parameters
    ----------
    site2s : sequence of package.V2Site
    design : str
    root : str, optional
        Output root (default the first site's).
    max_px : int
        Subsample of keep_full(P0, W3) for n_clear.

    Returns
    -------
    dict
        The document written to ``<root>/common/pilot_reps.json``.
    """
    from experiments.benchmark import domain as dm
    from experiments.benchmark import instance as inst_mod
    from experiments.benchmark import package, product
    site2s = list(site2s)
    root = root or (site2s[0].root if site2s else "results/validation_v2")
    sites = [s.site for s in site2s]
    rows = _rows(root, design, sites)
    doc = {"format": PILOT_FORMAT, "design": design, "max_px": int(max_px),
           "rules": RULES, "no_image": list(NO_IMAGE), "kR_min": KR_MIN,
           "subsample": f"<= {max_px} keep_full(P0, W3) pixels, default_rng([20261008, "
                        "SITE_ID]) without replacement (declared)",
           "truth_free": True, "sites": {}}
    for s2 in site2s:
        site = s2.site
        srows = [inst_mod.norm_row(r) for r in rows if r["site"] == site]
        pkg = package.V2Package(s2)
        full = pkg.full
        pos_all = full.pos if full.pos is not None else np.empty(0, np.int64)
        rng = np.random.default_rng([int(config.SEEDS["subsets"]), int(config.SITE_ID[site])])
        if len(pos_all) > max_px:
            sub = np.sort(rng.choice(pos_all, int(max_px), replace=False))
        else:
            sub = np.asarray(pos_all, np.int64)
        groups = OrderedDict()
        for r in srows:
            groups.setdefault(inst_mod.instance_key(r), []).append(r)
        points = []
        for key, rs in groups.items():
            r0 = rs[0]
            pt = {"window": r0["window"], "setting": r0["setting"], "family": r0["family"],
                  "mode": r0["mode"], "N": r0["N"], "rep": r0["rep"]}
            try:
                sel = inst_mod.instance_selection(pkg, r0)
            except inst_mod.InstanceUnavailable as e:
                pt.update(status="unavailable", reason=e.reason)
                points.append(pt)
                continue
            except ValueError as e:                  # e.g. N larger than the record
                pt.update(status="invalid", reason=str(e))
                points.append(pt)
                continue
            T_w = len(pkg.window(r0["window"]))
            if isinstance(r0["N"], int):
                R = inst_mod.curve_R(r0["design"], r0["window"], r0["N"], r0["mode"])
                pt["subset_kind"] = settings.subset_kind(T_w, r0["N"], R)
                pt["R"] = R
            else:
                pt["subset_kind"] = "all"
                pt["R"] = 1
            clr = (pkg.count(r0["setting"], sel, pos=sub)[0] if len(sub)
                   else np.zeros(0, np.int32))
            med = float(np.median(clr)) if len(clr) else 0.0
            p10 = float(np.percentile(clr, 10)) if len(clr) else 0.0
            d = dm.get_v2(pkg, r0["setting"], r0["window"], sel)
            dates = pkg.t[sel]
            span = float((dates.max() - dates.min()).days) if len(sel) else 0.0
            rise = np.asarray(pkg.rising[sel], bool)
            comp = {}
            for r in rs:
                for p in r["products"]:
                    ok, why = expected(p, r["method"], len(sel), d.n_keep, d.status, med,
                                       int(rise.sum()), int((~rise).sum()), span)
                    comp[p] = {"computable": bool(ok), "reason": why}
            pt.update(status="ok", n_scenes=int(len(sel)), n_clear_median=med,
                      n_clear_p10=p10, n_px_counted=int(len(clr)), span_days=span,
                      n_rising=int(rise.sum()), domain_status=str(d.status),
                      n_keep=int(d.n_keep), K=int(d.K), computable=comp)
            points.append(pt)
        kr = {}
        for pt in points:
            if pt.get("status") != "ok":
                continue
            k = f"{pt['window']}|{pt['N']}"
            for p, c in pt["computable"].items():
                e = kr.setdefault(k, {}).setdefault(p, {"k": 0, "R": 0})
                e["R"] += 1
                e["k"] += int(c["computable"])
        for k, per in kr.items():
            for p, e in per.items():
                e["share"] = e["k"] / e["R"] if e["R"] else 0.0
                e["flag"] = bool(e["share"] < KR_MIN)
        doc["sites"][site] = {"n_keep_full": int(len(full.keep)), "n_px_counted": int(len(sub)),
                              "points": points, "kR": kr}
        pkg.close()
    if write:
        product.write_json(os.path.join(root, "common", "pilot_reps.json"), doc)
    print_table(doc)
    return doc


def print_table(doc):
    log(f"[pilot reps] design {doc['design']} (truth-free; n_clear on <= {doc['max_px']:,} "
        "keep_full px)")
    for site, e in doc["sites"].items():
        log(f"[pilot reps] {site}: keep_full {e['n_keep_full']:,} px")
        log(f"  {'win':<6}{'N':>5}{'rep':>4} {'kind':<9}{'n':>5}{'clr50':>7}{'clr10':>7} "
            f"{'domain':<14}{'n_keep':>8}  not computable")
        for pt in e["points"]:
            if pt.get("status") != "ok":
                log(f"  {pt['window']:<6}{str(pt['N']):>5}{pt['rep']:>4} "
                    f"{pt['status'].upper()} {pt.get('reason')}")
                continue
            bad = [p for p, c in pt["computable"].items() if not c["computable"]]
            log(f"  {pt['window']:<6}{str(pt['N']):>5}{pt['rep']:>4} {pt['subset_kind']:<9}"
                f"{pt['n_scenes']:>5}{pt['n_clear_median']:>7.0f}{pt['n_clear_p10']:>7.0f} "
                f"{pt['domain_status']:<14}{pt['n_keep']:>8}  {', '.join(bad) or '-'}")
        flags = [(k, p) for k, per in e["kR"].items() for p, v in per.items() if v["flag"]]
        if flags:
            log(f"  k/R < {KR_MIN}: " + ", ".join(f"{p}@{k}" for k, p in flags))
