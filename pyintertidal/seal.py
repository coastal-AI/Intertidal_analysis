"""
seal.py — Sealed predictions: commit to a claim before the data can answer it
==============================================================================

Implements hard rule **R6** of plan v4.

A sealed file is a prediction written down BEFORE the evidence that will judge
it is looked at: a compression field before the reserved survey is opened, a
tidal atlas before a waterline campaign, water levels before a SWOT pass. The
seal is what later makes the contrast believable — without it, nothing
distinguishes a prediction from a postdiction.

Mechanics, deliberately boring:

* ``sealed/registry.jsonl`` is APPEND-ONLY. One JSON line per seal:
  file path (relative to the repo), SHA256 of its bytes, UTC timestamp, and a
  free-text note saying what the file claims and what will judge it.
* :func:`create_seal` refuses to seal a path twice — a "resealed" prediction
  is exactly the fraud the mechanism exists to prevent.
* :func:`verify_seal` recomputes the hash and compares. Any mismatch is a
  loud failure, not a warning.
* :func:`verify_registry` re-verifies EVERY entry, so tampering with either a
  sealed file or the registry itself is caught by the test suite
  (``tests/test_seals.py`` runs it).

This module has no dependencies beyond the standard library on purpose: the
integrity layer must not break when the science stack does.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

#: repo root = parent of the package directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(_ROOT, "sealed", "registry.jsonl")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    return os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/")


def _load_registry():
    if not os.path.exists(REGISTRY):
        return []
    out = []
    with open(REGISTRY, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def create_seal(path, note):
    """Seal a file: record its hash and the claim it commits to.

    Refuses if the file is already sealed (append-only means append-only) or
    if the note is empty — a seal without a stated claim is worthless.
    """
    if not note or not str(note).strip():
        raise ValueError("un sello sin nota no compromete a nada: di que "
                         "afirma el fichero y que lo juzgara")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rel = _rel(path)
    for e in _load_registry():
        if e["file"] == rel:
            raise RuntimeError(f"{rel} ya esta sellado ({e['date']}); "
                               f"re-sellar es exactamente el fraude que R6 "
                               f"impide")
    entry = {"file": rel, "sha256": _sha256(path),
             "date": _dt.datetime.now(_dt.timezone.utc)
             .strftime("%Y-%m-%dT%H:%M:%SZ"),
             "note": str(note)}
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def verify_seal(path):
    """True if the file matches its seal. Raises if it was never sealed."""
    rel = _rel(path)
    entries = [e for e in _load_registry() if e["file"] == rel]
    if not entries:
        raise KeyError(f"{rel} no esta en el registro de sellos")
    return _sha256(path) == entries[0]["sha256"]


def verify_registry():
    """Re-verify every seal. Returns (n_ok, list_of_broken)."""
    broken = []
    entries = _load_registry()
    for e in entries:
        p = os.path.join(_ROOT, e["file"])
        if not os.path.exists(p):
            broken.append({**e, "reason": "fichero ausente"})
        elif _sha256(p) != e["sha256"]:
            broken.append({**e, "reason": "hash distinto"})
    return len(entries) - len(broken), broken


def is_sealed(path):
    rel = _rel(path)
    return any(e["file"] == rel for e in _load_registry())
