# -*- coding: utf-8 -*-
"""Third-party code vendored verbatim for validation v2.

``spa.py``
    The NREL Solar Position Algorithm (Reda & Andreas 2004, Solar Energy
    76:577-589, doi:10.1016/j.solener.2003.12.003; corrigendum Solar Energy
    81:838, 2007) as implemented in pvlib python 0.13.0 (``pvlib/spa.py``,
    taken unchanged from the wheel ``pvlib-0.13.0-py3-none-any.whl`` on
    PyPI). Licence: BSD-3-Clause, copyright Sandia National Laboratories
    (2013), the PVLIB python Development Team (2014) and the pvlib python
    contributors (2023); the full text is ``LICENSE-pvlib.txt`` beside it.
    The file must stay byte-identical to the original
    (:data:`SPA_SHA256`, checked by tests/test_benchmark_sensing_time.py);
    every adaptation lives in ``experiments/benchmark/sensing_time.py``.
    It needs only numpy (numba is used only if ``PVLIB_USE_NUMBA`` is set
    and numba is installed; neither is the case in this project).

Why vendored and not installed: pvlib is not in the local or the server
venv, the server venv is built with uv from requirements-cluster.txt, and
the solar ephemeris is one self-contained numpy module. Vendoring keeps the
G-time step free of a new dependency for the user to install, and pins the
exact code the sensing times were computed with.

Importing this package does nothing.
"""

#: sha256 of ``spa.py`` exactly as in the pvlib 0.13.0 wheel (LF line ends)
SPA_SHA256 = "ad7762ccca5fd9611ef9e2a9b1e37f95fe4c5b31628616b2ad3e84ea1e5f9202"
#: where it came from
SPA_SOURCE = "pvlib 0.13.0 (PyPI wheel pvlib-0.13.0-py3-none-any.whl), pvlib/spa.py"
SPA_LICENSE = "BSD-3-Clause (LICENSE-pvlib.txt)"
