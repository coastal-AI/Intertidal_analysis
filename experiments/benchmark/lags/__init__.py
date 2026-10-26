"""Lag-method registry of the validation v2 lag benchmark (spec Stage L, §5 design 'lag').

``REGISTRY`` maps a method id (the ``method`` column of a plan row,
``designs.Unit.method``) to the ``"module:function"`` that runs one unit of it;
``PRODUCT_METHOD`` maps every product id a unit writes (``designs.Unit.products``)
to the method whose module carries its card. E0 writes (E0a, E0b, L01) -- or
(E0g,) for its variant 'gaugelag' -- from one fit (spec §5.1, E0's L01 is the
MAREA band lag with its bootstrap).

Nothing is imported at package import time: :func:`load` imports an adapter
lazily, so a module that is not built yet only fails its own units (the queue
records status 'error', reason 'module not built') and never the import of
this package or of the other methods.

Each adapter module exposes ``CONFIG`` (JSON-ready dict), ``CARDS``
({product_id: card}, see :func:`common.check_card`), ``CODE_DEPS`` (repository
paths whose source enters the unit's code key), optionally
``external_inputs(inst, unit) -> {name: sha256 or None}``, and the run
functions named here: ``run_x(inst, unit, opts) -> list[dict]``, one
canonical result per product of the unit, in order (plan INTERFACES §5).
"""
from __future__ import annotations

import importlib

REGISTRY = {"E0": "experiments.benchmark.methods.elev_marea_pair:run_unit",
            "L01v": "experiments.benchmark.lags.marea_family:run_l01v",
            "L02": "experiments.benchmark.lags.marea_family:run_l02",
            "L03": "experiments.benchmark.lags.marea_family:run_l03",
            "L04": "experiments.benchmark.lags.aggregate:run_l04",
            "L07": "experiments.benchmark.lags.aggregate:run_l07",
            "L08": "experiments.benchmark.lags.gm20:run_unit",
            "L09": "experiments.benchmark.lags.hd26:run_unit",
            "L11": "experiments.benchmark.lags.ppl:run_unit",
            "L12": "experiments.benchmark.lags.bt26:run_unit",
            "L14": "experiments.benchmark.lags.models:run_l14",
            "L16": "experiments.benchmark.lags.regional:run_unit",
            "L17": "experiments.benchmark.lags.physics:run_l17",
            "L18": "experiments.benchmark.lags.physics:run_l18",
            "L20": "experiments.benchmark.lags.references:run_l20"}

PRODUCT_METHOD = {"E0a": "E0", "E0b": "E0", "L01": "E0", "E0g": "E0",
                  "L01v": "L01v", "L02": "L02", "L03": "L03",
                  "L04w": "L04", "L04i": "L04", "L07": "L07", "L08": "L08",
                  "L09": "L09", "L10": "L09", "L11": "L11", "L12": "L12",
                  "L14": "L14", "L15": "L14", "L16": "L16", "L17": "L17",
                  "L18": "L18", "L20": "L20"}


class ModuleNotBuilt(ImportError):
    """The adapter module of a method does not exist (yet)."""


def _split(method):
    try:
        target = REGISTRY[method]
    except KeyError:
        raise KeyError(f"lag method {method!r} is not in the registry "
                       f"({', '.join(sorted(REGISTRY))})") from None
    module, _, func = target.partition(":")
    return module, func


def module_of(method) -> str:
    """Dotted module path of ``method``'s adapter (no import)."""
    return _split(method)[0]


def import_module(method):
    """Import the adapter module of ``method``; :class:`ModuleNotBuilt` if absent.

    Only a missing adapter module itself is reported as 'module not built'; an
    import error raised from inside an existing module (a missing dependency
    of it) propagates unchanged, so a real bug is never disguised.
    """
    module = module_of(method)
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module == missing or module.startswith(missing + ".")):
            raise ModuleNotBuilt(f"module not built: {module} (lag method {method})") from exc
        raise


def load(method):
    """The run function of ``method`` (lazy import).

    Raises :class:`ModuleNotBuilt` (an ``ImportError``) with the message
    'module not built: <module> (lag method <id>)' when the adapter module does
    not exist, and ``ImportError`` when the module lacks the function.
    """
    module, func = _split(method)
    mod = import_module(method)
    fn = getattr(mod, func, None)
    if fn is None or not callable(fn):
        raise ImportError(f"module {module} has no run function {func!r} (lag method {method})")
    return fn


def method_of(product_id) -> str:
    """The method whose adapter writes ``product_id``."""
    try:
        return PRODUCT_METHOD[product_id]
    except KeyError:
        raise KeyError(f"product {product_id!r} is not in PRODUCT_METHOD") from None


def card(product_id) -> dict:
    """The method card of ``product_id`` (``CARDS[product_id]`` of its module)."""
    mod = import_module(method_of(product_id))
    cards = getattr(mod, "CARDS", None) or {}
    if product_id not in cards:
        raise KeyError(f"module {mod.__name__} has no card for product {product_id!r}")
    return cards[product_id]


def all_cards() -> dict:
    """``{product_id: card}`` over every product whose adapter module is built.

    Products of modules that are not built yet (or lack the card) are left
    out; :func:`check_all_cards` lists them.
    """
    out = {}
    for pid in PRODUCT_METHOD:
        try:
            out[pid] = card(pid)
        except (ModuleNotBuilt, KeyError):
            continue
    return out


def check_all_cards() -> dict[str, list[str]]:
    """``{product_id: problems}`` for every product of the registry.

    A product whose module is not built reads ['module not built: ...']; one
    whose module has no card reads ['no card ...']; otherwise the problems of
    :func:`common.check_card` plus an id mismatch (empty list = valid).
    """
    from . import common

    out = {}
    for pid in PRODUCT_METHOD:
        try:
            c = card(pid)
        except ModuleNotBuilt as exc:
            out[pid] = [str(exc)]
            continue
        except KeyError as exc:
            out[pid] = [f"no card: {exc.args[0] if exc.args else exc}"]
            continue
        problems = common.check_card(c)
        if isinstance(c, dict) and c.get("id") not in (None, pid):
            problems.append(f"card id {c.get('id')!r} differs from product {pid!r}")
        out[pid] = problems
    return out


def module_attrs(method) -> dict:
    """``CONFIG``, ``CARDS``, ``CODE_DEPS`` and ``external_inputs`` of an adapter.

    Missing attributes read as ``{}``, ``{}``, ``()`` and ``None``.
    """
    mod = import_module(method)
    return {"module": mod.__name__,
            "file": getattr(mod, "__file__", None),
            "CONFIG": getattr(mod, "CONFIG", {}) or {},
            "CARDS": getattr(mod, "CARDS", {}) or {},
            "CODE_DEPS": tuple(getattr(mod, "CODE_DEPS", ()) or ()),
            "external_inputs": getattr(mod, "external_inputs", None)}
