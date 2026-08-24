"""
oplog.py — Record what the pipeline did (opt-in provenance)
============================================================

A record of the operations a notebook ran, with their parameters and the
shape of the data before and after each one. It powers the process graph
(:mod:`pyintertidal.explain.dag`) and the run report, and it is what you
paste into a paper's methods section when a reviewer asks "which threshold
did you actually use?".

**Opt-in by design.** Nothing is recorded unless you open a log, because
silent global state is exactly the kind of magic this package avoids:

>>> from pyintertidal import explain
>>> with explain.recording() as log:
...     reference, clouds = reference_and_clouds(cube, threshold=0.02)
...     wf = water_frequency(cube, dates=usable, threshold=0.02)
>>> log.table()          # what ran, with parameters
>>> explain.show_graph(log)

Core functions call :func:`record` when they finish; with no active log that
call costs a dictionary lookup and returns. You can also record steps by hand
— useful for work the library does not know about.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Operation:
    """One recorded step."""

    name: str
    kind: str = "generic"                 # filter / reduce / apply / …
    params: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    before: dict | None = None
    after: dict | None = None
    note: str = ""
    seconds: float = 0.0


class OpLog:
    """An ordered list of :class:`Operation` records."""

    def __init__(self, name="pipeline"):
        self.name = name
        self.operations: list[Operation] = []

    def add(self, operation):
        """Append an already-built :class:`Operation`."""
        self.operations.append(operation)
        return operation

    def record(self, name, kind="generic", params=None, inputs=None,
               outputs=None, before=None, after=None, note="", seconds=0.0):
        """Append a step by hand."""
        return self.add(Operation(name=name, kind=kind, params=dict(params or {}),
                                  inputs=list(inputs or []),
                                  outputs=list(outputs or []),
                                  before=before, after=after, note=note,
                                  seconds=seconds))

    # ── reporting ───────────────────────────────────────────────────────────
    def table(self, as_dataframe=True):
        """The log as a table (pandas DataFrame when available)."""
        rows = [{
            "step": i + 1, "operation": op.name, "kind": op.kind,
            "inputs": ", ".join(op.inputs), "outputs": ", ".join(op.outputs),
            "note": op.note,
            "params": " · ".join(f"{k}={v}" for k, v in op.params.items()),
            "seconds": round(op.seconds, 1),
        } for i, op in enumerate(self.operations)]
        if as_dataframe:
            try:
                import pandas as pd
                return pd.DataFrame(rows)
            except ImportError:
                pass
        return rows

    def parameters(self):
        """Every parameter used, keyed by operation — the provenance record."""
        return {op.name: op.params for op in self.operations if op.params}

    def __len__(self):
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)

    def __repr__(self):
        return f"OpLog({self.name!r}, {len(self.operations)} operations)"


#: The log currently recording, if any (``None`` = nothing is recorded).
_ACTIVE: OpLog | None = None


def active():
    """The active log, or ``None``."""
    return _ACTIVE


@contextmanager
def recording(name="pipeline"):
    """Record every library operation performed inside the ``with`` block.

    Nested blocks are not supported on purpose: one log per analysis keeps
    the resulting graph honest.
    """
    global _ACTIVE
    previous = _ACTIVE
    log = OpLog(name)
    _ACTIVE = log
    try:
        yield log
    finally:
        _ACTIVE = previous


def record(name, kind="generic", params=None, inputs=None, outputs=None,
           before=None, after=None, note="", seconds=0.0):
    """Record a step if a log is active; otherwise do nothing.

    This is the hook the library's compute functions call. It is deliberately
    cheap and side-effect-free when recording is off.
    """
    if _ACTIVE is None:
        return None
    return _ACTIVE.record(name, kind=kind, params=params, inputs=inputs,
                          outputs=outputs, before=before, after=after,
                          note=note, seconds=seconds)


@contextmanager
def step(name, kind="generic", params=None, inputs=None, outputs=None,
         note=""):
    """Time a block and record it (no-op when nothing is recording).

    >>> with explain.step("water frequency", kind="reduce",
    ...                   params={"min_obs": 15}):
    ...     wf = water_frequency(cube, dates=usable)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        record(name, kind=kind, params=params, inputs=inputs, outputs=outputs,
               note=note, seconds=time.perf_counter() - start)
