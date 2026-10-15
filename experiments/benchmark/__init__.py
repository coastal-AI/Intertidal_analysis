# -*- coding: utf-8 -*-
"""v13 — controlled benchmark of intertidal lag and elevation methods.

Every method receives the same scenes, the same per-pixel cloud mask, the
same band preprocessing, the same boundary tide (as exact levels), the same
date subsets and the same lag search window; only the estimator changes.
The design is ``docs/v13_benchmark_spec.md`` (revision 2 with the authors'
final decisions of 2026-10-08).

Modules (imported by name; this file imports nothing, so the CLI can set the
BLAS/OMP thread counts before numpy is loaded):

``config``   sites, constants, seeds, the sectioned FROZEN plan, paths
``settings`` validation v2: the factor registries, Setting, ids, effective keys
``inputs``   validation v2: SceneSource (Zarr, NetCDF), offsets, inputs report
``obs``      THE observation: SCL mask rules, band preprocessing, NDWI, V;
             v2: observe(raw, setting) -> X, Y, V, W
``levels``   boundary provider, exact level table, 1-/10-min series, marks
``product``  Unit, Product, statuses, atomic IO and every key
``package``  the common input package (two streaming passes, manifest);
             v2: year shards keyed by dom_version, pixel-major tiles, uint8
             counters, OCM step, settings pass, MNAR table
``domain``   full-record and A-mode subset domains; v2: the domain of an
             instance (setting, window, subset)
``designs``  validation v2: design tables, plan CSVs, the 10-day cap and cuts,
             the rows of the error attribution (noise, lag, elevation)
``judges``   in-estuary gauges: geometry, neighbourhoods, inclusion
``truth``    elevation and lag truths, scoring domains
``runner``   unit queue, claims, resume
``fetch``    validation v2: the fetch commands (IOC, tides, CMEMS, judges, ...)
``extract``  validation v2: a small Zarr extract of a legacy cube (local checks)
``sensing_time``  validation v2: the OPTIONAL sensing-time diagnostic (solar
             angles); no run needs it: the scene instants are the catalogue
             times (user decision of 2026-10-14)

Run v13 through ``python -X utf8 -m experiments.v13_benchmark`` and validation
v2 through ``python -X utf8 -m experiments.validation_v2``.
"""

__all__ = ["config", "settings", "inputs", "obs", "levels", "product", "package",
           "domain", "designs", "judges", "truth", "runner", "extract", "sensing_time"]
