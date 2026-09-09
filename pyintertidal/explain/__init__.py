"""
pyintertidal.explain — See what the pipeline is doing
======================================================

Earth-observation analysis is easy to run and hard to picture. This
subpackage makes each step visible, in the visual language of the openEO
documentation and EOxHub notebooks, so a reader of your notebook can follow
the data rather than trust it.

Four kinds of output
--------------------
**The datacube** (:func:`draw_cube`, :func:`describe_cube`) — a bands × time
grid of small rasters with every dimension labelled, optionally filled with
REAL downsampled imagery so you see the clouds and the tide states.

**Operations** (:func:`draw_operation`) — before → after panels with openEO
semantics: ``filter`` (fewer labels), ``reduce`` (a dimension collapses),
``apply`` (same shape, new values), ``aggregate`` (coarser grid), ``mask``
(pixels removed).

**The process graph** (:func:`recording`, :func:`draw_graph`) — record an
analysis and draw it as a provenance DAG with the parameters in the boxes.

**Summaries** (:func:`summarize`) — the xarray/dask panel: header with the
dimensions, sizes table, the **proportional array box** whose shape and grid
lines follow the real dimensions and chunking, and collapsible coordinates and
attributes. :func:`draw_array_box` renders that box on its own.

Example
-------
>>> from pyintertidal import explain
>>> explain.describe_cube(cube)                     # what did we download?
>>> explain.summarize(cube)                         # the numbers behind it
>>> with explain.recording("Villaviciosa") as log:
...     ...                                         # run the analysis
>>> explain.draw_graph(log)                         # how it was produced
"""

from .svgbase import Diagram, Canvas, color_for, PALETTE
from .cube_svg import draw_cube, show_cube, describe_cube
from .ops_svg import draw_operation, show_operation, KIND_SEMANTICS
from .oplog import OpLog, Operation, recording, record, step, active
from .dag import draw_graph, show_graph
from .arraybox import draw_array_box, human_bytes, chunk_count
from .htmlrepr import summarize, summarize_array, summarize_cube
from .reports import dates_available, cloud_screening, intertidal_breakdown

__all__ = [
    "Diagram", "Canvas", "color_for", "PALETTE",
    "draw_cube", "show_cube", "describe_cube",
    "draw_operation", "show_operation", "KIND_SEMANTICS",
    "OpLog", "Operation", "recording", "record", "step", "active",
    "draw_graph", "show_graph",
    "summarize", "summarize_array", "summarize_cube",
    "draw_array_box", "human_bytes", "chunk_count",
    "dates_available", "cloud_screening", "intertidal_breakdown",
]
