# -*- coding: utf-8 -*-
"""Validation v3, the Cantabrian estuaries: PNOA LiDAR as the elevation truth and windows centred on the LiDAR flight,
as the design of 2026-10-29 fixes them (experiments/validation_v3_cantabrian_2026-10-29.md, the reference for every
constant here).

windows  the n screened scenes nearest in time to the truth date (the symmetric window, its half-width, the scenes
         before and after the date, whether the cube covers it) and the class-balance mask of the scored pixels, with
         its sensitivity
"""
