"""
simulator.py — The tribunal: a calibrated twin of the estimator (plan v4)
=========================================================================

Implements the pluggable simulator of plan v4 §3 and hard rules **R3** (every
verdict against a matched null) and **R7** (explicit seeds).

Every claim this project makes is judged against data manufactured HERE: real
tide series, real per-pixel parameter distributions, real noise, pixels
planted at KNOWN elevations. Whatever the estimator reports on such data with
all mechanisms OFF is what it fabricates on its own — that is the null. A
mechanism is only believed when planting it reproduces something observed
that the null cannot.

Two calibration lessons are baked in, both paid for on 2026-08-18:

* pixels are resampled WHOLE. Drawing a, b, sigma and noise independently
  manufactures combinations that do not exist and the estimator wastes its
  behaviour on a fictional population.
* noise has a SCENE-WIDE component (haze, sun angle, turbidity move a whole
  scene together). A per-pixel-only null is quieter than the archive and
  flatters every result — measured: it fabricated only I2=25 % where the
  matched null fabricates 40 %.

Mechanisms (plan v4: "cada mecanismo un módulo con parámetros; nulo = todos
apagados") subclass :class:`Mechanism` and may hook three points:

* ``level(h, s_km, rising)``  -> the water level each pixel experiences
  (tidal transfer f(h), hysteresis via limb-dependent lags);
* ``truth(z, sg)``            -> the planted ground (copulas sigma-z);
* ``optics(clean, wet_frac)`` -> the radiometry (optical offset delta,
  connectivity censoring).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def _phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
#  Mechanisms
# ─────────────────────────────────────────────────────────────────────────────

class Mechanism:
    """Base: identity everywhere. The null is simulate(..., mechanisms=())."""

    name = "identity"

    def level(self, h, s_km, rising):
        """(T,) tide -> (T, P) or (T,) level actually felt by the pixels."""
        return h

    def truth(self, z, sg):
        return z, sg

    def optics(self, clean, wet_frac):
        return clean


class Transfer(Mechanism):
    """Tidal transfer f(h, s): amplification/distortion along the channel."""

    name = "transfer"

    def __init__(self, fn):
        self.fn = fn                    # fn(h (T,), s_km (P,)) -> (T, P)

    def level(self, h, s_km, rising):
        return self.fn(h, s_km)


class Hysteresis(Mechanism):
    """Limb-dependent lag: flood arrives late, ebb leaves later (ponding).

    Needs the tide sampled at shifted instants — pass ``h_shift`` mapping
    minutes -> (T,) series, plus per-pixel lags in minutes (rounded to the
    available shifts).
    """

    name = "hysteresis"

    def __init__(self, h_shift, tau_up_px, tau_dn_px):
        self.h_shift = {int(k): np.asarray(v, float)
                        for k, v in h_shift.items()}
        self.tau_up = np.asarray(tau_up_px)
        self.tau_dn = np.asarray(tau_dn_px)

    def level(self, h, s_km, rising):
        T = len(h)
        P = len(self.tau_up)
        out = np.empty((T, P))
        for tv in np.unique(np.concatenate([self.tau_up, self.tau_dn])):
            hv = self.h_shift[int(tv)]
            up_cols = self.tau_up == tv
            dn_cols = self.tau_dn == tv
            if up_cols.any():
                out[np.ix_(rising, up_cols)] = hv[rising, None]
            if dn_cols.any():
                out[np.ix_(~rising, dn_cols)] = hv[~rising, None]
        return out


class OpticalOffset(Mechanism):
    """Additive radiometric bias delta(x) — e.g. the NDWI/MNDWI disagreement."""

    name = "optical_offset"

    def __init__(self, delta_px):
        self.delta = np.asarray(delta_px, float)

    def optics(self, clean, wet_frac):
        return clean + self.delta[None, :]


class SigmaCopula(Mechanism):
    """Plant a dependence sigma(z) instead of independent draws."""

    name = "sigma_copula"

    def __init__(self, fn):
        self.fn = fn                    # fn(z (P,)) -> sigma (P,)

    def truth(self, z, sg):
        return z, self.fn(z)


class ConnectivityCensoring(Mechanism):
    """A pixel only floods when the water can REACH it (minimax threshold).

    ``threshold_px`` >= true elevation: the level at which the pixel first
    floods, from depression filling (see lamina.flood_threshold prototype).
    """

    name = "connectivity"

    def __init__(self, threshold_px):
        self.thr = np.asarray(threshold_px, float)

    def level(self, h, s_km, rising):
        # censoring acts on the EFFECTIVE level: below the sill nothing
        # arrives — the pixel stays DRY however close h is to its own z.
        # (Bug caught by the B6 gate on 2026-08-19: the original expression
        # carried a `- 1e3 * 0` that made the else-branch a no-op, so the
        # planted censoring barely acted and recall was judged against a
        # world with no censoring in it.)
        h2 = h if h.ndim == 2 else np.broadcast_to(h[:, None],
                                                   (len(h), len(self.thr)))
        return np.where(h2 >= self.thr[None, :], h2, h2 - 10.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration and simulation
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(store_npz, base_npz, tide, epoch_years=None, sd_level=0.0):
    """Template from the archive: whole pixels + scene systematic (R7).

    ``store_npz``: the sealed per-pixel series (Y, C, dates).
    ``base_npz``: the sealed gamma=0 fit (a, b, z, sg, good).
    Returns everything :func:`simulate` needs.

    ``sd_level`` (metres): the WATER-LEVEL uncertainty of the archive — the
    tide model is wrong by a random amount each scene (measured on
    2026-08-18: 0.134 m from real-vs-nominal overpass hour alone; ~0.22 m
    for EOT20 against gauges inside rias). The archive's fitted sigma has
    that error folded in: the fit absorbs scene-level noise by WIDENING the
    transition. A null that replants the inflated sigma while expressing the
    same energy as white radiometric noise refits systematically NARROWER
    sigmas (measured: median 0.150 against 0.220). So calibration
    DECONVOLVES it — sg_topo = sqrt(max(sg^2 - sd_level^2, floor^2)) — and
    :func:`simulate` re-injects it as a per-scene level jitter. This is the
    sigma_topo / sigma_level decomposition of phase B2, living where it
    belongs: in the null.
    """
    d = np.load(store_npz, allow_pickle=True)
    Y, C, dates = d["Y"], d["C"], np.array([str(s) for s in d["dates"]])
    if epoch_years is not None:
        keep = np.array([int(s[:4]) >= epoch_years for s in dates])
        Y, C = Y[keep], C[keep]
    Y = np.nan_to_num(Y, nan=0.0).astype(np.float32)
    Cb = (C > 0)

    B = np.load(base_npz)
    a, b, z, sg, good = B["a"], B["b"], B["z"], B["sg"], B["good"]

    pred = a[None, :] + b[None, :] * _phi(
        (np.asarray(tide, float)[:, None] - z[None, :])
        / np.maximum(sg[None, :], 1e-3))
    resid = np.where(Cb, Y - pred, np.nan)
    noise = np.sqrt(np.nanmean(resid ** 2, axis=0))
    sd_scene = float(np.nanstd(np.nanmedian(resid, axis=1)))

    g = good & np.isfinite(noise) & (noise > 0)
    sg_topo = np.sqrt(np.maximum(sg ** 2 - float(sd_level) ** 2, 0.03 ** 2))
    return {"a": a[g], "b": b[g], "sg": sg[g], "sg_topo": sg_topo[g],
            "sd_level": float(sd_level), "noise": noise[g],
            "z": z[g], "obs_frac": float(Cb.mean()),
            "sd_scene": sd_scene, "n_template": int(g.sum()),
            # column indices into the store, so callers can hand simulate()
            # the REAL cloud mask of the very pixels they sample. An iid
            # mask is measurably too kind: real cloud gaps cluster in time,
            # fits are worse conditioned, and fitted sigma inflates — with
            # iid masks the null recovered sigma medians 30 % low.
            "idx": np.where(g)[0]}


def simulate(template, z_true, tide, seed, mechanisms=(), rising=None,
             s_km=None, C_real=None, template_rows=None):
    """Manufacture an archive over pixels planted at ``z_true``.

    All mechanisms OFF (the default) is the matched null (R3). Returns
    ``(Y, C)`` shaped (T, P).

    ``template_rows``: use these template rows one-to-one instead of a random
    whole-pixel resample — for nulls that replay REAL pixels (their own a, b,
    sigma, noise and cloud mask) with only the truth planted.
    """
    rng = np.random.default_rng(seed)
    tide = np.asarray(tide, float)
    P = len(z_true)
    if template_rows is not None:
        pick = np.asarray(template_rows)
        assert len(pick) == P
    else:
        pick = rng.integers(0, template["n_template"], P)
    a = template["a"][pick]
    b = template["b"][pick]
    sg = template.get("sg_topo", template["sg"])[pick]
    nz = template["noise"][pick]

    z, sg = np.asarray(z_true, float), np.asarray(sg, float)
    for m in mechanisms:
        z, sg = m.truth(z, sg)

    # the null carries the archive's measured level uncertainty: a random
    # scene-wide error of the tide itself (see calibrate). Without it, sigma
    # marginals cannot be reproduced.
    sd_lv = float(template.get("sd_level", 0.0))
    h = tide + (rng.normal(0.0, sd_lv, len(tide)) if sd_lv > 0 else 0.0)
    for m in mechanisms:
        h = m.level(h, s_km, rising)
    if h.ndim == 1:
        h = np.broadcast_to(h[:, None], (len(tide), P))

    wet = _phi((h - z[None, :]) / np.maximum(sg[None, :], 1e-3))
    clean = a[None, :] + b[None, :] * wet
    for m in mechanisms:
        clean = m.optics(clean, wet)

    Y = (clean + rng.normal(0.0, 1.0, clean.shape) * nz[None, :]
         + rng.normal(0.0, template["sd_scene"], (len(tide), 1)))
    if C_real is not None:
        C = np.asarray(C_real)[:, :P] > 0
    else:
        C = rng.random(Y.shape) < template["obs_frac"]
    return Y.astype(np.float32), C
