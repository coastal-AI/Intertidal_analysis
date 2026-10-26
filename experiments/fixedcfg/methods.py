# -*- coding: utf-8 -*-
"""Registry of the elevation methods of the fixed-configuration tables.

``METHODS = {id: (label, run, card)}``, in table-column order, holds the
competitors that run on the common per-pixel inputs; ``VARIANTS`` holds their
declared sensitivities (same interface, not columns of the default tables).
MAREA (E0b) and no delay (E0a) are run by the runner with v14's code, and
NIDEM (E1) through its own hook, so they are not here.

Interface (see :mod:`experiments.fixedcfg.elev_base`)::

    out = run(obs, cfg)        # out = dict(z=float32 (P,), sigma=None, meta=dict)
    out["meta"]["status"]      # 'ok' or 'not computable' (+ 'reason')

``cfg`` shared keys: ``level_series`` (needed by E5a, E5b and E7: h_B without
lag at 1 min over the window, a :class:`~experiments.fixedcfg.level_series.
LevelSeries`) and ``block_px``; per-method options under the method id.

card = dict(reference, doi, fidelity, deviations, needs_full_frame) plus
id, label, family, rule, needs (inputs beyond obs), min_scenes, own_lag,
expected_offset, code and cost. Each deviation is
dict(paper, here, why, sensitivity).

Other builders can add a method without editing this file. Any module of
this package that, at its top level, either assigns ``METHODS`` (and
optionally ``VARIANTS``) with the same interface, or defines
``register(METHODS)`` (a function that adds its entries to the dict it is
given), is merged after the built-in ones, sorted by module name. Modules are
screened by parsing their source, so a module that does neither (e.g. a
side-interpreter worker) is never imported. A module that fails to import or
to register is skipped and its error is recorded in ``REGISTRY_ERRORS`` (the
runner should print it). :func:`catalogue_order` sorts ids by catalogue
number (E2, E4, E5a, ..., E12) for the tables.

DEA Intertidal 2.5.3 (dea_intertidal.py, its own ``register``) is added last
by :func:`register_dea` under the catalogue id E2, importing the adapter only
there, and only when its side interpreter (venv-dea or $MAREA_DEA_PYTHON)
exists; otherwise it is skipped cleanly and ``REGISTRY_SKIPPED`` says why.
"""
from __future__ import annotations

import ast
import importlib
import os
import pkgutil
import re

from experiments.fixedcfg import elev_base as base

#: the built-in competitors, in column order (catalogue ids,
#: docs/validation_v2_spec.md s3.2)
BUILTIN = ("elev_logitide", "elev_freq", "elev_coastxray", "elev_bracket")
#: DEA Intertidal 2.5.3 (dea_intertidal.py): registered by :func:`register_dea`
#: under its catalogue id, only when its side interpreter exists
DEA_MODULE = "dea_intertidal"
DEA_ID = "E2"
#: modules of the package that are never discovered as method modules (DEA
#: is registered by register_dea; dea_worker runs in the side interpreter)
_NOT_METHODS = {"methods", "elev_base", "level_series", "smoke_competitors", DEA_MODULE, "dea_worker"}

METHODS = {}
VARIANTS = {}
REGISTRY_ERRORS = {}
#: methods not registered in this environment, with why (not errors)
REGISTRY_SKIPPED = {}


def _add(modname, entries, target, strict):
    for mid, entry in entries.items():
        if mid in METHODS or mid in VARIANTS:
            raise ValueError(f"method id {mid!r} of {modname} is already registered")
        label, run, card = entry
        if not callable(run):
            raise TypeError(f"{modname}:{mid}: run is not callable")
        if strict:
            base.check_card(card)          # built-ins: also the deviation format
        else:
            miss = [k for k in base.CARD_KEYS if k not in card]
            if miss:
                raise ValueError(f"{modname}:{mid}: card lacks {miss}")
    target.update(entries)


def _merge(modname, strict=False):
    mod = importlib.import_module(f"experiments.fixedcfg.{modname}")
    if hasattr(mod, "METHODS") or hasattr(mod, "VARIANTS"):
        _add(modname, dict(getattr(mod, "METHODS", {})), METHODS, strict)
        _add(modname, dict(getattr(mod, "VARIANTS", {})), VARIANTS, strict)
    elif callable(getattr(mod, "register", None)):
        got = {}
        mod.register(got)
        _add(modname, got, METHODS, strict)


def _defines_methods(path):
    """True iff the module source assigns METHODS or defines register() at its
    top level (parsed, not imported)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "register":
            return True
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == "METHODS" for t in targets):
            return True
    return False


def _discover():
    here = os.path.dirname(os.path.abspath(__file__))
    found = sorted(m.name for m in pkgutil.iter_modules([here]) if not m.ispkg)
    return [m for m in found if m not in BUILTIN and m not in _NOT_METHODS
            and _defines_methods(os.path.join(here, m + ".py"))]


def register_dea(methods=None, skipped=None, method_id=DEA_ID):
    """Register DEA Intertidal (experiments/fixedcfg/dea_intertidal.py,
    imported here only) through its own ``register`` under ``method_id``
    (the catalogue's E2) when its side interpreter exists; otherwise skip it
    cleanly: nothing is registered and ``skipped`` (default
    REGISTRY_SKIPPED) says why. The DEA card names its sources under
    'references'; the registry's 'reference' is set to them. Returns the
    side interpreter's path, or None when skipped."""
    methods = METHODS if methods is None else methods
    skipped = REGISTRY_SKIPPED if skipped is None else skipped
    from experiments.fixedcfg import dea_intertidal as dea

    try:
        side = dea.find_side_python()
    except dea.DEAEnvironmentError as e:
        skipped[method_id] = f"{dea.LABEL}: no side interpreter ({e})"
        return None
    got = dea.register({}, method_id=method_id)
    label, run, card = got[method_id]
    card = dict(card)
    card.setdefault("reference", "; ".join(r["cite"] for r in card.get("references", [])) or card.get("name", label))
    card.setdefault("side_interpreter", str(side))
    miss = [k for k in base.CARD_KEYS if k not in card]
    if miss:
        raise ValueError(f"{DEA_MODULE}:{method_id}: card lacks {miss}")
    if method_id in methods or (methods is METHODS and method_id in VARIANTS):
        raise ValueError(f"method id {method_id!r} of {DEA_MODULE} is already registered")
    if not callable(run):
        raise TypeError(f"{DEA_MODULE}:{method_id}: run is not callable")
    methods[method_id] = (label, run, card)
    skipped.pop(method_id, None)
    return side


for _m in BUILTIN:
    _merge(_m, strict=True)          # a broken built-in is a hard error
for _m in _discover():
    try:
        _merge(_m)
    except Exception as _e:   # another builder's module: record, do not break the registry
        REGISTRY_ERRORS[_m] = f"{type(_e).__name__}: {_e}"
try:
    register_dea()
except Exception as _e:       # the DEA adapter itself is broken: record, do not break the registry
    REGISTRY_ERRORS[DEA_MODULE] = f"{type(_e).__name__}: {_e}"


def get(method_id):
    """(label, run, card) of a method or a variant."""
    if method_id in METHODS:
        return METHODS[method_id]
    if method_id in VARIANTS:
        return VARIANTS[method_id]
    raise KeyError(f"unknown method {method_id!r}; methods {list(METHODS)}, variants {list(VARIANTS)}")


def needs_level_series(method_id):
    return "level_series" in get(method_id)[2].get("needs", [])


def catalogue_order(ids):
    """``ids`` sorted by catalogue number and suffix (E2 < E4 < E5a < E5b <
    E7 < E12; variants after their method)."""
    def key(mid):
        m = re.match(r"E(\d+)([a-z]*)(.*)", str(mid))
        return (int(m.group(1)), m.group(2), m.group(3)) if m else (10 ** 6, str(mid), "")
    return sorted(ids, key=key)


def catalogue():
    """One row per registered method and variant (for the report)."""
    rows = []
    for kind, reg in (("method", METHODS), ("variant", VARIANTS)):
        for mid, (label, _, card) in reg.items():
            rows.append({"id": mid, "kind": kind, "label": label, "fidelity": card["fidelity"],
                         "doi": card.get("doi"), "needs": ",".join(card.get("needs", [])),
                         "needs_full_frame": card["needs_full_frame"],
                         "n_deviations": len(card["deviations"])})
    return rows
