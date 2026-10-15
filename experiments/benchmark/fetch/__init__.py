# -*- coding: utf-8 -*-
"""Validation v2 — the fetch commands (docs/validation_v2_spec.md §1.3, §1.5, §1.6, §12.1).

Each module downloads or computes one external input ONCE, caches it under a
fixed path with a manifest (sha256, sizes, request), and reports a status
that the planner turns into NOT COMPUTABLE / NOT AVAILABLE rows. Nothing is
ever substituted silently.

Water levels (§1.5, §1.6):

``ioc``    the boundary members' IOC chunks over 2017-2025, in a cache of
           their own (the production 2023-2025 cache is never touched)
``tides``  the per-pixel global tide model LT (eo-tides ``pixel_tides`` on a
           1 km node grid)
``cmems``  the regional hydrodynamic SSH LM (CMEMS AMM15 / IBI through the
           copernicusmarine toolbox), with the 'credentials missing' path

Truths and metadata: ``judges`` (the judge series and their time-stamp
conventions), ``vak_meta`` (Vaklodingen survey sources and dates),
``pnoa_meta`` (PNOA flight metadata).

``sensing`` (the CDSE STAC items of the OPTIONAL solar-angle sensing-time
diagnostic, ``validation_v2 diag time``) is not an input of any run: the
scene instants are the catalogue times (user decision of 2026-10-14).

Modules are imported by name; this file imports nothing (no import-time
side effects).
"""
