# -*- coding: utf-8 -*-
"""Shared plumbing of the per-pixel competitors (E4, E5a, E5b, E5c, E7, E12).

The common inputs (shared interface of the fixed-configuration tables)
--------------------------------------------------------------------
Every competitor is ``run(obs, cfg) -> dict(z, sigma, meta)``:

* ``obs`` is a namespace with ``Y`` float (T, P) NDWI = (B03 - B08) / (B03 +
  B08), NaN where undefined; ``C`` bool (T, P) clear = SCL in {4, 5, 6, 7};
  ``W`` bool (T, P) wet = C & (Y > 0); ``h`` float (T,) the boundary tide h_B
  at the scene instants, WITHOUT lag; ``t`` the scene instants (T,);
  ``keep`` int (P,) flat frame indices; ``shape`` (H, W); ``pixel_m``;
  ``site``. Y must keep its NaN: a clear observation whose NDWI is undefined
  is dropped by these methods (it is neither wet nor dry), whereas MAREA's
  inversion reads it as 0 (dry). That difference is declared in every card.
* ``cfg`` is a dict (or None). Shared keys: ``level_series`` (a
  :class:`experiments.fixedcfg.level_series.LevelSeries`: h_B without lag at
  a 1-min step over the window; needed by E5a, E5b and E7) and ``block_px``
  (columns per block). Method options go under the method id, e.g.
  ``cfg = {"E7": {"threshold": 0.0}}``; an unknown option is an error.
* The result: ``z`` float32 (P,) metres in the h_B datum, NaN where the
  method gives no value; ``sigma`` None (no method here publishes a
  per-pixel uncertainty); ``meta`` a dict with ``status`` ('ok' or 'not
  computable'), ``reason``, counts, timings, the options used and, under
  ``extra``, small per-pixel diagnostics (P,).

Every method is per pixel, so the columns are processed in blocks and P can
be the scored units' pixels or every keep pixel: the values do not depend on
the blocking.
"""
from __future__ import annotations

import time

import numpy as np

#: the fields of the common inputs every competitor reads or checks
OBS_FIELDS = ("Y", "C", "W", "h", "t", "keep", "shape", "pixel_m", "site")
STATUS_OK = "ok"
STATUS_NOT_COMPUTABLE = "not computable"
#: default columns per block (float64 temporaries of T x BLOCK_PX)
BLOCK_PX = 50_000
#: keys every method card must carry (the shared interface) ...
CARD_KEYS = ("reference", "doi", "fidelity", "deviations", "needs_full_frame")
#: ... and the keys of every declared deviation (the deviations table of
#: docs/validation_v2_spec.md s11.4: what the paper does, what we do, why,
#: and the row that measures it)
DEVIATION_KEYS = ("paper", "here", "why", "sensitivity")


class NotComputable(Exception):
    """The method's own rules give no product for this input."""


def as_dict(cfg):
    """``cfg`` as a plain dict (None, a dict, or any object with attributes)."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    return dict(vars(cfg))


def method_options(cfg, method_id, defaults):
    """The method's options: ``defaults`` updated with ``cfg[method_id]``.
    An option the method does not have is an error (a typo would otherwise
    silently run the default)."""
    opts = dict(defaults)
    per = as_dict(cfg).get(method_id)
    if per is None:
        return opts
    per = as_dict(per)
    unknown = sorted(set(per) - set(defaults))
    if unknown:
        raise ValueError(f"{method_id}: unknown option(s) {unknown}; known: {sorted(defaults)}")
    opts.update(per)
    return opts


def block_px(cfg, default=BLOCK_PX):
    v = int(as_dict(cfg).get("block_px", default))
    if v < 1:
        raise ValueError("block_px must be >= 1")
    return v


def column_blocks(P, block):
    """Slices of at most ``block`` columns covering 0..P-1."""
    for b0 in range(0, int(P), int(block)):
        yield slice(b0, min(b0 + int(block), int(P)))


def check_obs(obs, check_wet=True, block=BLOCK_PX):
    """Validate the common inputs and return ``(Y, C, W, h, T, P)``.

    Raises ValueError when a field is missing or malformed, when ``h`` is not
    finite, or (``check_wet``) when W is not C & (Y > 0), the common
    configuration's water detection."""
    missing = [k for k in ("Y", "C", "W", "h") if not hasattr(obs, k)]
    if missing:
        raise ValueError(f"obs lacks {missing}")
    Y, C, W = obs.Y, obs.C, obs.W
    if getattr(Y, "ndim", 0) != 2:
        raise ValueError("obs.Y must be a (T, P) array")
    T, P = (int(v) for v in Y.shape)
    if not np.issubdtype(np.asarray(Y[:0]).dtype, np.floating):
        raise ValueError("obs.Y must be floating point with NaN where NDWI is undefined")
    for name, a in (("C", C), ("W", W)):
        if tuple(a.shape) != (T, P) or np.asarray(a[:0]).dtype != bool:
            raise ValueError(f"obs.{name} must be a bool (T, P) = ({T}, {P}) array")
    h = np.asarray(obs.h, float)
    if h.shape != (T,):
        raise ValueError(f"obs.h must have shape ({T},), got {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError("obs.h must be finite (h_B at every scene of the window)")
    t = getattr(obs, "t", None)
    if t is not None and len(t) != T:
        raise ValueError(f"obs.t has {len(t)} instants for {T} scenes")
    keep = getattr(obs, "keep", None)
    if keep is not None and len(keep) != P:
        raise ValueError(f"obs.keep has {len(keep)} pixels for {P} columns")
    if check_wet:
        for sl in column_blocks(P, block):
            Yb = np.asarray(Y[:, sl])
            with np.errstate(invalid="ignore"):
                expect = np.asarray(C[:, sl]) & (Yb > 0)
            if not np.array_equal(expect, np.asarray(W[:, sl])):
                raise ValueError("obs.W is not C & (Y > 0): the common configuration's "
                                 "water detection does not hold")
    return Y, C, W, h, T, P


def valid_and_wet(Y, C, W, sl):
    """``(V, Wb)`` of the columns ``sl``: V = clear with a defined NDWI (the
    observations these methods use), Wb = wet (a subset of V)."""
    Yb = np.asarray(Y[:, sl])
    V = np.asarray(C[:, sl], bool) & np.isfinite(Yb)
    Wb = np.asarray(W[:, sl], bool) & V
    return V, Wb


def level_series(obs, cfg):
    """The window's 1-min level series: ``cfg['level_series']`` or
    ``obs.level_series`` (coerced to a LevelSeries), or None."""
    from experiments.fixedcfg.level_series import LevelSeries

    obj = as_dict(cfg).get("level_series")
    if obj is None:
        obj = getattr(obs, "level_series", None)
    return LevelSeries.coerce(obj)


def require_series(obs, cfg, need_times=False):
    """The level series, checked against the scenes; raises NotComputable when
    it is missing (it is an input of the method, not something to guess)."""
    ls = level_series(obs, cfg)
    if ls is None:
        raise NotComputable("needs cfg['level_series']: h_B without lag at a 1-min step over the "
                            "window (experiments.fixedcfg.level_series.LevelSeries.from_boundary)")
    if need_times and ls.t is None:
        raise NotComputable("the level series carries no times (needed for the harmonic analysis)")
    if ls.t is not None and getattr(obs, "t", None) is not None and len(obs.t):
        import pandas as pd

        t = pd.DatetimeIndex(obs.t)
        if t.tz is not None:
            t = t.tz_convert(None)
        step = ls.step if ls.step is not None else pd.Timedelta(0)
        if t.min() < ls.t[0] - step or t.max() > ls.t[-1] + step:
            raise ValueError(f"the level series ({ls.t[0]} .. {ls.t[-1]}) does not span the "
                             f"window's scenes ({t.min()} .. {t.max()})")
    return ls


def base_meta(method_id, T, P, opts):
    return {"method": method_id, "status": STATUS_OK, "reason": "", "n_scenes": int(T),
            "n_px": int(P), "params": {k: _plain(v) for k, v in opts.items()}}


def finish(meta, z, t0, extra=None, sigma=None):
    """The result dict: z as float32, the finite count and the wall time."""
    z = np.asarray(z, np.float32)
    meta["n_finite"] = int(np.isfinite(z).sum())
    meta["seconds"] = round(time.time() - t0, 3)
    if extra:
        meta["extra"] = extra
    return {"z": z, "sigma": sigma, "meta": meta}


def not_computable(meta, P, reason, t0):
    meta["status"] = STATUS_NOT_COMPUTABLE
    meta["reason"] = str(reason)
    return finish(meta, np.full(int(P), np.nan, np.float32), t0)


def _plain(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, tuple):
        return list(v)
    return v


def check_card(card):
    """Raise ValueError when a method card lacks a required key or a
    deviation lacks one of DEVIATION_KEYS."""
    miss = [k for k in CARD_KEYS if k not in card]
    if miss:
        raise ValueError(f"card lacks {miss}")
    for i, d in enumerate(card["deviations"]):
        bad = [k for k in DEVIATION_KEYS if not str(d.get(k, "")).strip()]
        if bad:
            raise ValueError(f"deviation {i} lacks {bad}")
    return True
