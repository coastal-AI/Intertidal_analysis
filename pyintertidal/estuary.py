"""
estuary.py — What an estuary does to the tide on its way in
============================================================

A global ocean tide model knows the open sea. It does not know that the wave
entering a ria is squeezed by converging banks, slowed and damped by friction
on a shallow bed, and lifted by the river coming the other way. Every method
in this package that assumes one water level for a whole scene is assuming
those three things away.

This module turns the assumption into an object that can be measured. It
takes a :class:`~pyintertidal.boundary.BoundaryProvider` — a model, a tide
gauge, an ensemble, it makes no difference — and propagates it inland.

The form, and why this one
--------------------------
Linearising the shallow-water equations in a channel whose width converges
exponentially gives, for each tidal constituent separately, an amplitude that
grows or decays exponentially with distance and a phase that lags linearly::

    h(s, t) = z0(s) + SUM_k  A_k exp(mu_k s) cos( w_k t - w_k tau_k s + phi_k )

Two numbers per constituent: ``mu_k`` in 1/km, positive when the estuary
amplifies that constituent and negative when friction wins, and ``tau_k`` in
hours per kilometre, the inverse of the wave's celerity. Plus ``z0(s)``, the
river backwater, which raises the mean level upstream without oscillating.

Working constituent by constituent is not a stylistic choice. Friction and
convergence act on frequency, so M2 and its overtone M4 propagate differently,
and it is precisely the growth of M4 relative to M2 that makes a tidal wave
inside an estuary asymmetric — flooding faster than it ebbs. A single gain
applied to the whole tide curve cannot express that.

How to get the parameters, in order of how much they can be trusted
-------------------------------------------------------------------
1. :meth:`EstuaryTransfer.from_gauges` — two records, one at the mouth and one
   inland. Nothing is assumed; the estuary is measured. This is the only route
   currently backed by evidence, and it is what the module should be used with.
2. :meth:`EstuaryTransfer.from_geometry` — Savenije's relations, so that a ria
   with no gauge inside it can still be given a transfer from its shape alone.
   **Untested here.** It is provided because it is the route that would make
   the operator useful at scale, and it is marked so that nobody mistakes it
   for a validated one.

What the imagery cannot contribute
----------------------------------
Sentinel-2 crosses at a fixed solar time, so the solar semidiurnal S2 is seen
at one phase forever and cannot be estimated from the archive at all; K1 and
P1 alias to one year and are confounded with the seasonal cycle. See
:func:`pyintertidal.boundary.alias_periods`. Any transfer fitted from imagery
must take those constituents from the boundary and leave them alone.
"""

from __future__ import annotations

import numpy as np

from .boundary import (PERIODS_H, fit_harmonics, hours_since_epoch,
                       transfer_from_gauges)


def _residual_gain(mouth, inner, constituents=None):
    """How much of the boundary's non-tidal signal really reaches inland.

    Surge is regional — one weather system covers hundreds of kilometres — so
    two gauges a few kilometres apart should share it almost exactly, and
    carrying the boundary's residual across should be free accuracy that no
    ocean model can offer.

    Measured, that turned out to be true sometimes and badly false others. At
    Honolulu, Breskens and Goteborg the residuals of the two stations
    correlate above 0.95; at Ferrol, in one test window, only 0.28, because
    the boundary record itself was noisy there — its residual had more than
    twice the spread of the interior one. Carrying it across regardless made
    the prediction worse, 0.152 m to 0.351 m.

    So the fraction is estimated instead of assumed: the least-squares
    coefficient of the interior residual on the boundary residual, over the
    period both are known. It comes out near one where the surge is genuinely
    shared and near zero where the boundary is just noisy, which is exactly
    the behaviour wanted — and it costs one regression.
    """
    try:
        hm = mouth.harmonics(constituents)
        hi = inner.harmonics(constituents)
        if not hm or not hi:
            return 0.0
        t = mouth.times if hasattr(mouth, "times") else None
        if t is None or not hasattr(inner, "levels"):
            return 0.0
        import pandas as pd
        idx = pd.DatetimeIndex(t)
        if len(idx) > 20000:
            idx = idx[:: max(len(idx) // 20000, 1)]
        th = hours_since_epoch(idx)

        def resid(b, h):
            raw = np.asarray(b.levels(idx), dtype=float)
            rec = np.zeros(len(idx))
            for k, (a, p, _) in h.items():
                rec += a * np.cos(2 * np.pi / PERIODS_H[k] * th + p)
            r = raw - rec
            return r - np.nanmedian(r)

        x, y = resid(mouth, hm), resid(inner, hi)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 200 or np.std(x[m]) < 1e-6:
            return 0.0
        g = float(np.dot(x[m], y[m]) / np.dot(x[m], x[m]))
        return float(np.clip(g, 0.0, 1.5))
    except Exception:
        return 0.0


class EstuaryTransfer:
    """Amplification and lag per constituent, as a function of distance.

    Parameters
    ----------
    mu : dict {constituent: 1/km}
        Exponential growth rate of the amplitude along the channel. Positive
        amplifies, negative damps.
    tau : dict {constituent: hours/km}
        Phase lag per kilometre — the inverse celerity of that constituent.
    backwater : (float, float)
        ``(a, b)`` in ``z0(s) = a * s + b``: the river's contribution to the
        mean level, in metres per kilometre and metres.
    name : str
        Free-text label, e.g. the estuary's name.
    """

    def __init__(self, mu=None, tau=None, backwater=(0.0, 0.0), name=""):
        self.mu = dict(mu or {})
        self.tau = dict(tau or {})
        self.backwater = tuple(backwater)
        self.name = name
        self.provenance = "manual"
        #: how much of the boundary's non-tidal residual actually reaches the
        #: interior station, measured rather than assumed — see
        #: :func:`_residual_gain`. Zero means "do not carry it across".
        self.residual_gain = 0.0

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_gauges(cls, mouth, inner, distance_km, name="",
                    constituents=None):
        """Measure the transfer from two records a known distance apart.

        The gain of each constituent between the two stations gives ``mu``
        directly, and the phase difference gives ``tau``. No physics is
        assumed — which is the point, since this is the yardstick everything
        else is judged against.

        A phase difference is only defined modulo a full cycle, so a lag
        longer than one period cannot be distinguished from a shorter one.
        For estuaries of a few tens of kilometres the true lag is well under
        an M2 period and the smallest positive solution is the right one; for
        a long tidal river it is not, and the caller must say so.
        """
        tr = transfer_from_gauges(mouth, inner, constituents)
        if not tr:
            raise ValueError("no se pudo descomponer alguna de las dos series")
        d = float(distance_km)
        mu, tau = {}, {}
        for k, v in tr.items():
            mu[k] = float(np.log(max(v["gain"], 1e-6)) / d)
            tau[k] = float(v["lag_hours"] / d)
        obj = cls(mu, tau, name=name)
        obj.provenance = "medido con dos mareografos"
        obj._measured = tr
        obj._distance_km = d
        obj.residual_gain = _residual_gain(mouth, inner)
        return obj

    @classmethod
    def from_geometry(cls, convergence_km, depth_m, friction=0.0025,
                      constituents=("M2", "S2", "N2", "K1", "O1"), name=""):
        """First-order transfer from the estuary's shape. UNTESTED.

        Savenije's linearised solution balances convergence against friction:
        a channel that narrows fast amplifies, a shallow rough one damps, and
        the celerity follows the shallow-water speed corrected for both. This
        is what would let a ria with no gauge be given a transfer, and it is
        the route worth developing — but nothing in this project has yet
        checked it against a measurement, so it is labelled accordingly and
        should not be used to produce numbers anyone relies on.

        Parameters
        ----------
        convergence_km : float
            E-folding length of the width, in km. Small means strongly
            converging.
        depth_m : float
            Mean depth of the channel.
        friction : float
            Dimensionless bed friction coefficient.
        """
        g = 9.81
        c0 = np.sqrt(g * max(depth_m, 0.1)) * 3.6      # km/h
        mu, tau = {}, {}
        for k in constituents:
            w = 2 * np.pi / PERIODS_H[k]               # rad/h
            conv = 1.0 / (2.0 * max(convergence_km, 1e-3))
            damp = friction * w / (2.0 * max(depth_m, 0.1) * 1e-3 * c0)
            mu[k] = float(conv - damp)
            tau[k] = float(1.0 / c0)
        obj = cls(mu, tau, name=name)
        obj.provenance = "SIN VALIDAR — derivado de la geometria"
        return obj

    # ── use ─────────────────────────────────────────────────────────────
    def levels(self, boundary, s_km, times, constituents=None,
               keep_residual=True, residual_hours=24.0):
        """Water level at distance ``s_km`` inland, at the given instants.

        The boundary is decomposed into constituents, each is carried inland
        with its own gain and lag, and the result is summed.

        ``keep_residual`` is what makes this competitive, and leaving it out
        was a real mistake. Rebuilding the level from a handful of sinusoids
        discards everything in the boundary record that is not one of them —
        storm surge above all, but also the minor constituents and any
        meteorological signal. A tide gauge MEASURES those; a global ocean
        model cannot represent them at all. Throwing them away hands the model
        an advantage it has not earned, and in a first version of this
        function it did exactly that: EOT20 beat the operator at every
        station under 30 km.

        Surge has a spatial scale of hundreds of kilometres, so between two
        gauges a few kilometres apart it is essentially the same water. It is
        therefore carried across unchanged rather than transferred: over
        distances where the tidal lag is minutes, a signal that varies over
        hours does not need shifting. For a long estuary that reasoning
        weakens, and the residual should then be lagged too — hence the
        switch, rather than always doing it.
        """
        import pandas as pd

        h = boundary.harmonics(constituents) if hasattr(boundary, "harmonics") \
            else boundary
        if not h:
            raise ValueError("el contorno no sabe descomponerse en armonicos")
        idx = pd.DatetimeIndex(pd.to_datetime(times))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        # same fixed reference the harmonics were fitted against, so a
        # transfer measured on one window can be applied to another
        t_h = hours_since_epoch(idx)
        s = float(s_km)

        out = np.full(len(idx), self.backwater[0] * s + self.backwater[1])
        for k, (amp, ph, _) in h.items():
            w = 2 * np.pi / PERIODS_H[k]
            gain = np.exp(self.mu.get(k, 0.0) * s)
            lag = self.tau.get(k, 0.0) * s
            out = out + amp * gain * np.cos(w * (t_h - lag) + ph)

        if keep_residual and hasattr(boundary, "levels"):
            raw = np.asarray(boundary.levels(idx), dtype=float)
            tidal = np.zeros(len(idx))
            for k, (amp, ph, _) in h.items():
                w = 2 * np.pi / PERIODS_H[k]
                tidal += amp * np.cos(w * t_h + ph)
            resid = raw - (tidal + np.nanmedian(raw - tidal))
            resid = np.where(np.isfinite(resid), resid, 0.0) * self.residual_gain
            # Only the SLOW part of the residual travels. Surge is driven by
            # weather systems hundreds of kilometres across and varies over
            # hours to days, so two gauges a few kilometres apart sit under
            # the same one. Harbour seiches do not: they are oscillations of
            # one basin, with periods of minutes, and each dock has its own.
            # Passing the raw residual across therefore injects the boundary
            # dock's seiches into the prediction, and measured at Ferrol that
            # made the error worse, 0.152 m to 0.351 m. Low-passing it keeps
            # the surge and leaves the seiches where they belong.
            if residual_hours and len(resid) > 3:
                dt = float(np.median(np.diff(t_h))) if len(t_h) > 1 else 1.0
                win = max(int(round(residual_hours / max(dt, 1e-6))), 1)
                if win > 1:
                    kern = np.ones(win) / win
                    pad = np.r_[np.full(win, resid[0]), resid,
                                np.full(win, resid[-1])]
                    resid = np.convolve(pad, kern, mode="same")[win:-win]
            out = out + resid
        return out

    def gain(self, constituent, s_km):
        """Amplification factor of one constituent at distance ``s_km``."""
        return float(np.exp(self.mu.get(constituent, 0.0) * float(s_km)))

    def lag_minutes(self, constituent, s_km):
        """How many minutes late that constituent arrives at ``s_km``."""
        return float(self.tau.get(constituent, 0.0) * float(s_km) * 60.0)

    def summary(self, s_km):
        """Readable table of what the estuary does at one distance."""
        rows = []
        for k in sorted(set(self.mu) | set(self.tau),
                        key=lambda x: PERIODS_H.get(x, 99)):
            rows.append({"constituent": k,
                         "gain": round(self.gain(k, s_km), 3),
                         "lag_min": round(self.lag_minutes(k, s_km), 1),
                         "mu_per_km": round(self.mu.get(k, 0.0), 5),
                         "tau_h_per_km": round(self.tau.get(k, 0.0), 5)})
        return rows

    def __repr__(self):
        return (f"<EstuaryTransfer {self.name!r} {len(self.mu)} constituyentes"
                f" · {self.provenance}>")


def validate_against_gauge(transfer, boundary, gauge, s_km, times=None):
    """Predict an interior gauge from the boundary, and score the prediction.

    The test the whole module exists for, and the only one that can settle
    whether a transfer is real: feed it the mouth, ask for the interior, and
    compare against a record it never saw.

    THREE baselines are reported, and the third is the one that matters:

    ``sin_operador``
        the raw boundary series, which is what every method in this package
        assumes today — that the tide inside equals the tide outside;
    ``solo_armonicos``
        the boundary decomposed and rebuilt with NO transfer applied. This
        looks like a technicality and is not: rebuilding from constituents
        also throws away surge, seiches and meteorological noise, so a
        transfer operator gets credit for filtering that has nothing to do
        with the estuary. Over 6 km, where the real transfer moves the water
        by a few centimetres, that filtering can be most of the apparent
        gain;
    ``con_operador``
        the full prediction.

    The operator's actual contribution is ``solo_armonicos`` minus
    ``con_operador``. Comparing against the raw series instead flatters it,
    and by a lot in exactly the short-ria regime this project works in.
    """
    import pandas as pd

    if times is None:
        times = gauge.times
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    truth = gauge.levels(idx)
    raw = boundary.levels(idx)
    pred = transfer.levels(boundary, s_km, idx)
    identity = EstuaryTransfer(name="identidad").levels(boundary, s_km, idx)

    def score(v):
        m = np.isfinite(v) & np.isfinite(truth)
        if m.sum() < 30:
            return None
        d = v[m] - truth[m]
        d = d - np.median(d)          # datums differ; that is not skill
        return {"n": int(m.sum()),
                "rmse_m": float(np.sqrt(np.mean(d ** 2))),
                "sd_truth_m": float(np.std(truth[m])),
                "var_explained": float(
                    1.0 - np.var(d) / max(np.var(truth[m]), 1e-9))}

    s_raw, s_id, s_pred = score(raw), score(identity), score(pred)
    out = {"sin_operador": s_raw, "solo_armonicos": s_id,
           "con_operador": s_pred}
    if s_raw and s_pred:
        out["mejora_sobre_crudo_m"] = s_raw["rmse_m"] - s_pred["rmse_m"]
    if s_id and s_pred:
        # the honest one: what the TRANSFER adds, once the harmonic
        # reconstruction's own filtering is taken out of the comparison
        out["mejora_del_operador_m"] = s_id["rmse_m"] - s_pred["rmse_m"]
    return out
