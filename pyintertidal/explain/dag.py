"""
dag.py — The process graph of an analysis
==========================================

Draws a recorded :class:`~pyintertidal.explain.oplog.OpLog` as a layered
directed graph, in the style of the openEO web editor: operations are boxes
(coloured by kind, annotated with their parameters) and the products flowing
between them are the edges.

This is the provenance figure of a study. One picture answers "where did this
DEM come from, and with which thresholds?" — which is what makes an analysis
auditable by somebody who did not run it.
"""

from __future__ import annotations

from .svgbase import Canvas, Diagram, INK, MUTED

#: Box colour per operation kind.
KIND_COLORS = {
    "acquire": "#4a78c4", "filter": "#c4934a", "reduce": "#4aa46b",
    "apply": "#8a4ac4", "aggregate": "#2b8cbe", "mask": "#c44a4a",
    "validate": "#5f6b76", "generic": "#7f8c8d",
}


def draw_graph(log, box_w=232, box_h=48, gap_y=30, max_params=3,
               title=None):
    """Render an operation log as a vertical process graph.

    Parameters
    ----------
    log : OpLog
        A log filled by :func:`pyintertidal.explain.recording`.
    max_params : int
        Parameters shown per box (the rest stay in ``log.parameters()``).

    Returns
    -------
    Diagram
    """
    ops = list(log)
    if not ops:
        c = Canvas(360, 60)
        c.text(12, 30, "no operations recorded", size=11, fill=MUTED)
        return Diagram(c.svg(), "process graph")

    left = 26
    top = 52 if title or getattr(log, "name", None) else 24
    width = left + box_w + 210
    height = top + len(ops) * (box_h + gap_y) + 18

    c = Canvas(width, height)
    heading = title or f"process graph · {getattr(log, 'name', 'pipeline')}"
    c.text(12, 22, heading, size=12, weight="600")
    c.text(12, 36, f"{len(ops)} operations, in execution order", size=9,
           fill=MUTED)

    for i, op in enumerate(ops):
        y = top + i * (box_h + gap_y)
        colour = KIND_COLORS.get(op.kind, KIND_COLORS["generic"])

        c.rect(left, y, box_w, box_h, fill=colour, opacity=0.14, rx=6)
        c.rect(left, y, box_w, box_h, fill="none", stroke=colour,
               stroke_width=1.4, rx=6)
        c.rect(left, y, 5, box_h, fill=colour, opacity=0.95, rx=2)

        c.text(left + 14, y + 19, op.name, size=10.5, weight="600")
        params = " · ".join(f"{k}={v}" for k, v in
                            list(op.params.items())[:max_params])
        if params:
            c.text(left + 14, y + 33, params, size=8, fill=MUTED, mono=True)
        if op.note:
            c.text(left + 14, y + 44, op.note, size=8, fill=MUTED)

        c.text(left + box_w - 8, y + 19, op.kind, size=8, fill=colour,
               anchor="end", weight="600")
        if op.seconds:
            c.text(left + box_w - 8, y + 33, f"{op.seconds:.1f}s", size=8,
                   fill=MUTED, anchor="end", mono=True)

        # Edge to the next operation, labelled with what flows along it.
        if i < len(ops) - 1:
            x_mid = left + box_w / 2
            c.arrow(x_mid, y + box_h, x_mid, y + box_h + gap_y - 4)
            flowing = ", ".join(op.outputs) if op.outputs else ""
            if flowing:
                c.text(x_mid + 10, y + box_h + gap_y / 2 + 3, flowing,
                       size=8, fill=MUTED, mono=True)

        # Side annotation: the products this step produced.
        if op.outputs:
            c.text(left + box_w + 16, y + 19, " → " + ", ".join(op.outputs),
                   size=8.5, fill=INK, mono=True)
    return Diagram(c.svg(), heading)


def show_graph(log, **kwargs):
    """Alias of :func:`draw_graph`."""
    return draw_graph(log, **kwargs)
