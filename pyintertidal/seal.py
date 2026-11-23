"""
seal.py — SHA-256 of an input file (provenance)
===============================================

:func:`pyintertidal.marea.reconstruct` writes the SHA-256 of the cube it read
into ``result.json`` (``inputs_sha``), so every product names the exact bytes
it was made from. This module holds that one helper and needs nothing beyond
the standard library.

The helper's leading underscore is historical: it is the module's only
function, ``marea.reconstruct`` calls it, and ``scripts/marea_tools.sha256``
computes the same digest.
"""

from __future__ import annotations

import hashlib


def _sha256(path):
    """SHA-256 of a file's bytes (hex digest), read in 1 MiB chunks.

    ``path`` is any file path; returns the 64-character lower-case hex string
    (the same as ``sha256sum`` or ``certutil -hashfile <file> SHA256``).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
