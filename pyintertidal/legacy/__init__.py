"""
pyintertidal.legacy — Superseded methods, kept runnable for the paper
=====================================================================

Everything in this subpackage was replaced by something better in the core
package, but is still importable because the paper's comparison studies run
on exactly this code. Nothing here is recommended for new work; each entry
says what replaced it and why.

What lives here
---------------
:mod:`~pyintertidal.legacy.udf`
    The server-side UDF sources, verbatim. Never edit them: they are shipped
    as text to a remote interpreter, so any change silently alters what the
    benchmark ran.
:mod:`~pyintertidal.legacy.bathymetry`
    ``BathymetryReconstructor`` with the four superseded elevation
    estimators (``optimized``, ``pixels``, ``isolines``, ``siq``) and the
    server-side ``step``.
    **Replaced by** :func:`pyintertidal.elevation.fit_hsr` (RMSE 0.76 m vs
    1.14–3.4 m) and :func:`pyintertidal.elevation.fit_step`, both local and
    cloud-free.
:mod:`~pyintertidal.legacy.products`
    ``water_frequency_multi`` — the four-index water-frequency comparison in
    a single backend job, the fair A/B that chose NDWI.
    **Replaced for daily use by** :func:`pyintertidal.frequency.water_frequency`.

What is deliberately NOT here
-----------------------------
The old package also carried several *reimplementations of the same
algorithm* (a local reference map, a streaming reference map and a
server-side one; three water-frequency variants; thin plotting wrappers).
Those are not ported: duplicating an algorithm is how implementations drift
apart. The surviving implementation of each is in the core package, and the
genuinely different EXECUTION MODEL (compute in the backend) is what this
subpackage preserves.

Example
-------
>>> from pyintertidal.legacy import BathymetryReconstructor
>>> rec = BathymetryReconstructor(connection, water_source="ndwi")
>>> result = rec.reconstruct(aoi, ("2016-01-01", "2025-12-31"), tides,
...                          valid_dates=dates, method="step")
"""

from .bathymetry import BathymetryReconstructor, BathymetryResult, METHODS
from .products import water_frequency_multi
from . import udf

__all__ = ["BathymetryReconstructor", "BathymetryResult", "METHODS",
           "water_frequency_multi", "udf"]
