# experiments/benchmark/vendor/: third-party code, copied verbatim

| file | what it is |
|---|---|
| `spa.py` | the NREL Solar Position Algorithm (Reda & Andreas 2004, *Solar Energy* 76:577-589, doi:10.1016/j.solener.2003.12.003) as implemented in pvlib python 0.13.0 (`pvlib/spa.py`, taken unchanged from the wheel `pvlib-0.13.0-py3-none-any.whl` on PyPI) |
| `LICENSE-pvlib.txt` | its license: BSD-3-Clause, copyright Sandia National Laboratories (2013), the PVLIB python Development Team (2014) and the pvlib python contributors (2023) |
| `__init__.py` | the source, license and sha256 of `spa.py` (`SPA_SHA256`); importing the package does nothing |

`spa.py` must stay byte-identical to the original: its sha256 is pinned in
`__init__.py` and checked by `tests/test_benchmark_sensing_time.py`. Every
adaptation lives in `experiments/benchmark/sensing_time.py`.

**What it is used for.** `sensing_time.py` inverts the sun angles of each
scene into the time at which Sentinel-2 imaged the site, which comes
minutes after the catalog time of the scene (the start of the datatake).
Since 2026-10-14 this is an optional diagnostic
(`python -X utf8 -m experiments.validation_v2 diag time`), not a step of
the validation runs: every run uses the catalog time.

**Why vendored.** pvlib is not in the project's environments, and the solar
ephemeris is one self-contained numpy module. Copying it keeps that step
free of a new dependency and pins the exact code used.
