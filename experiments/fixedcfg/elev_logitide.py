# -*- coding: utf-8 -*-
"""E4 - LogiTide2DEM (Chen & Wang 2025): per-pixel binary logistic on the tide.

Rule (docs/validation_v2_spec.md s3.2.1, card "E4 LogiTide2DEM"), per pixel:

* the binary class of every valid observation: land (dry) = valid & not wet,
  water = W; valid = clear (SCL 4-7) with a defined NDWI;
* logistic regression ``logit P(land) = a + b h`` by IRLS (unpenalised
  maximum likelihood); the elevation is the 0.5 crossing ``Z = -a / b``;
* the modified success rate ``MSR = 0.5 SRW + 0.5 SRL`` (share of the water
  and of the land observations the fitted model classifies correctly; an
  observation is predicted land iff P(land) > 0.5); pixels with MSR < 0.6, or
  with Z outside the pixel's observed tide range, are dropped;
* at least 27 images per period: a window with fewer than 27 scenes is NOT
  COMPUTABLE.

Complete separation (every land observation at or below every water
observation) is common on clean pixels and has no maximum-likelihood
estimate. The frozen primary rule: Z = the midpoint of the separating gap
[max h(land), min h(water)] (equal to the E12 bracket midpoint on those
pixels). A pixel separated the other way round (land above water) gets the
midpoint of its own gap and is counted. Sensitivities (VARIANTS of the
registry): a Firth-penalised fit (``separation='firth'``) and the
unpenalised fit with a 25-iteration cap (``separation='cap25'``, R
``glm.fit``'s iteration and default ``maxit``).

IRLS is R's ``glm.fit`` for the binomial family with the logit link (start
mu = (y + 0.5) / 2, working response and weights, R's link clamps), solved
per pixel in closed form for the two coefficients, with h centred on the
pixel's mean valid level. The primary adds step-halving when the deviance
rises (a numerical safeguard that does not change the maximum) and a
tighter tolerance (1e-10) so that Z agrees with the exact maximum to 1e-6
(tested against scikit-learn's unpenalised fit).
"""
from __future__ import annotations

import time

import numpy as np

from experiments.fixedcfg import elev_base as base

ID = "E4"
LABEL = "LogiTide2DEM (Chen & Wang 2025)"
SEPARATION_RULES = ("gap_midpoint", "firth", "cap25")
DEFAULTS = {"separation": "gap_midpoint", "msr_min": 0.6, "min_scenes": 27, "min_obs_px": 0,
            "maxit": 100, "tol": 1e-10, "block_px": 20_000}
#: R's logit link clamps (src/library/stats/src/family.c)
THRESH, MTHRESH = 30.0, -30.0
DBL_EPS = float(np.finfo(float).eps)
INVEPS = 1.0 / DBL_EPS
#: R glm.control defaults, used by the 'cap25' sensitivity
GLM_MAXIT, GLM_EPSILON = 25, 1e-8
#: logistf-style Firth iteration controls (step cap, half-steps)
FIRTH_MAXSTEP, FIRTH_MAXHS = 5.0, 5

CARD = {
    "id": ID,
    "label": LABEL,
    "reference": ("Chen, Y.-C., Wang, S.-F. (2025). LogiTide2DEM: A method for reconstructing intertidal "
                  "topography in complex tidal flats using logistic regression with multi-temporal "
                  "Sentinel-2 and Landsat imagery. International Journal of Applied Earth Observation and "
                  "Geoinformation 139, 104561"),
    "doi": "10.1016/j.jag.2025.104561",
    "fidelity": ("F3 (re-implemented from the paper as transcribed in docs/validation_v2_spec.md "
                 "s3.2.1; IRLS agrees with scikit-learn's unpenalised fit to 1e-6 on non-separable "
                 "pixels; planted-Z tests; the Changhua data were not available)"),
    "family": "per-pixel binary logistic regression on tide height",
    "rule": ("logit P(land) = a + b h by IRLS on the valid observations; Z = -a/b; complete separation -> "
             "midpoint of the gap [max h(land), min h(water)]; drop MSR < 0.6 or Z outside the pixel's "
             "observed range; window >= 27 scenes"),
    "needs_full_frame": False,
    "needs": [],
    "min_scenes": 27,
    "own_lag": False,
    "expected_offset": ("none in a logistic world (the 0.5 crossing); on separated pixels the bracket "
                        "midpoint, |error| <= half the gap between the bracketing levels"),
    "deviations": [
        {"paper": "MNDWI (20 m) classified by a per-image Otsu threshold",
         "here": "NDWI > 0 (W of the common configuration), 10 m",
         "why": "common configuration: one index and one water detection for every method",
         "sensitivity": "not measured in these tables (validation v2 native cell C1.floor0.mndwi.D2)"},
        {"paper": "scenes with cloud <= 70 %",
         "here": "MAREA's production transition-zone screening; per-pixel clear mask SCL 4-7",
         "why": "common configuration",
         "sensitivity": "not measured in these tables"},
        {"paper": "levels of a coastal tide gauge at the image times",
         "here": "h_B without lag at the scene instants (production boundary)",
         "why": "common conditions: competitors use h_B without lag",
         "sensitivity": "not measured in these tables"},
        {"paper": "periods of a 22-year Landsat + Sentinel-2 archive, at least 27 images per period",
         "here": "the window of the n most recent screened scenes; >= 27 applies to the window's scene "
                 "count, so a cloud-masked pixel can enter with fewer valid observations "
                 "(option min_obs_px for a per-pixel reading; 0 = off)",
         "why": "the per-pixel cloud mask of the common configuration; the paper states the rule per "
                "period",
         "sensitivity": "option min_obs_px=27 (not run by default)"},
        {"paper": "complete separation not addressed",
         "here": "frozen primary: midpoint of the separating gap; reversed separation (land above water) "
                 "handled the same way and counted",
         "why": "no maximum-likelihood estimate exists under separation",
         "sensitivity": "VARIANTS E4_firth (Firth-penalised) and E4_cap25 (25-iteration cap)"},
        {"paper": "observed tide range (of the period)",
         "here": "the pixel's own range of valid levels",
         "why": "the per-pixel cloud mask: the levels the pixel's fit saw",
         "sensitivity": "not measured in these tables"},
        {"paper": "(not addressed)",
         "here": "a clear observation with undefined NDWI is dropped (MAREA's inversion reads it as dry)",
         "why": "the binary class needs a defined index",
         "sensitivity": "not measured (0.014 % of the clear observations at Villaviciosa)"},
    ],
    "code": "experiments/fixedcfg/elev_logitide.py",
    "cost": "a few IRLS iterations of (T, P) arrays; ~seconds per 10^5 pixels",
}


# ─────────────────────────────────────────────────────────────────────────────
#  R's logit link and the binomial deviance
# ─────────────────────────────────────────────────────────────────────────────

def linkinv(eta):
    """R ``binomial()$linkinv`` (logit): exp(eta)/(1 + exp(eta)) with eta's
    exponential clamped at DBL_EPSILON below -30 and 1/DBL_EPSILON above 30."""
    e = np.asarray(eta, float)
    tmp = np.where(e < MTHRESH, DBL_EPS, np.where(e > THRESH, INVEPS, np.exp(np.clip(e, MTHRESH, THRESH))))
    return tmp / (1.0 + tmp)


def mu_eta(eta):
    """R ``binomial()$mu.eta`` (logit)."""
    e = np.asarray(eta, float)
    ex = np.exp(np.clip(e, MTHRESH, THRESH))
    op = 1.0 + ex
    return np.where((e > THRESH) | (e < MTHRESH), DBL_EPS, ex / (op * op))


def deviance(y, mu, v):
    """Binomial deviance per column over the valid observations (y in {0, 1})."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(y > 0.5, -2.0 * np.log(mu), -2.0 * np.log1p(-mu))
    return np.where(v, r, 0.0).sum(axis=0)


def _wls(x, zw, w2):
    """Weighted least squares of zw on [1, x] per column (closed form)."""
    S0 = w2.sum(axis=0)
    S1 = (w2 * x).sum(axis=0)
    S2 = (w2 * x * x).sum(axis=0)
    T0 = (w2 * zw).sum(axis=0)
    T1 = (w2 * x * zw).sum(axis=0)
    det = S0 * S2 - S1 * S1
    with np.errstate(invalid="ignore", divide="ignore"):
        b0 = (S2 * T0 - S1 * T1) / det
        b1 = (S0 * T1 - S1 * T0) / det
    bad = ~(det > 0)
    b0[bad] = np.nan
    b1[bad] = np.nan
    return b0, b1


def _irls_step(x, y, v, eta, mu):
    """One glm.fit iteration from the current (eta, mu): the WLS solution."""
    me = mu_eta(eta)
    var = mu * (1.0 - mu)
    good = v & (me != 0) & (var > 0)
    zw = eta + (y - mu) / np.where(good, me, 1.0)
    w2 = np.where(good, me * me / np.where(good, var, 1.0), 0.0)
    return _wls(x, zw, w2)


def irls_logit(x, y, v, maxit=100, tol=1e-10, halving=True):
    """Unpenalised logistic regression of y on [1, x] per column, R glm.fit
    style (start mu = (y + 0.5)/2), with optional step-halving when the
    deviance rises. ``x`` (T, P) float (centred), ``y`` (T, P) {0, 1}, ``v``
    (T, P) bool. Returns (b0, b1, converged, iterations)."""
    x = np.where(v, x, 0.0)
    y = np.where(v, y, 0.0).astype(float)
    P = x.shape[1]
    mu0 = np.where(v, (y + 0.5) / 2.0, 0.5)
    eta = np.log(mu0 / (1.0 - mu0))
    mu = linkinv(eta)
    devold = deviance(y, mu, v)
    b0 = np.full(P, np.nan)
    b1 = np.full(P, np.nan)
    conv = np.zeros(P, bool)
    iters = np.zeros(P, np.int32)
    active = np.ones(P, bool)
    have_beta = False
    for it in range(1, int(maxit) + 1):
        cols = np.flatnonzero(active)
        if cols.size == 0:
            break
        xc, yc, vc = x[:, cols], y[:, cols], v[:, cols]
        n0, n1 = _irls_step(xc, yc, vc, eta[:, cols], mu[:, cols])
        ne = n0[None, :] + n1[None, :] * xc
        nm = linkinv(ne)
        dev = deviance(yc, nm, vc)
        if halving and have_beta:
            o0, o1 = b0[cols], b1[cols]
            for _ in range(30):
                worse = ~(dev <= devold[cols] * (1.0 + 1e-12) + 1e-12)
                if not worse.any():
                    break
                n0 = np.where(worse, 0.5 * (n0 + o0), n0)
                n1 = np.where(worse, 0.5 * (n1 + o1), n1)
                ne = n0[None, :] + n1[None, :] * xc
                nm = linkinv(ne)
                dev = deviance(yc, nm, vc)
        bad = ~np.isfinite(n0) | ~np.isfinite(n1)
        b0[cols], b1[cols] = n0, n1
        eta[:, cols], mu[:, cols] = ne, nm
        iters[cols] = it
        with np.errstate(invalid="ignore"):
            done = np.abs(dev - devold[cols]) / (np.abs(dev) + 0.1) < tol
        conv[cols] = done & ~bad
        devold[cols] = dev
        active[cols[done | bad]] = False
        have_beta = True
    return b0, b1, conv, iters


def firth_logit(x, y, v, maxit=100, tol=1e-8):
    """Firth-penalised logistic regression of y on [1, x] per column
    (Jeffreys-prior penalty 0.5 log det I; modified score with the hat
    values), logistf-style: start at (logit(mean y), 0), Newton steps capped
    at FIRTH_MAXSTEP, up to FIRTH_MAXHS half-steps while the penalised
    log-likelihood falls. Returns (b0, b1, converged, iterations)."""
    x = np.where(v, x, 0.0)
    y = np.where(v, y, 0.0).astype(float)
    vf = v.astype(float)
    P = x.shape[1]
    n = vf.sum(axis=0)
    ybar = np.clip(y.sum(axis=0) / np.maximum(n, 1), 1e-6, 1 - 1e-6)
    b0 = np.log(ybar / (1.0 - ybar))
    b1 = np.zeros(P)
    conv = np.zeros(P, bool)
    iters = np.zeros(P, np.int32)
    active = np.ones(P, bool)
    for it in range(1, int(maxit) + 1):
        cols = np.flatnonzero(active)
        if cols.size == 0:
            break
        xs, ys, vs = x[:, cols], y[:, cols], vf[:, cols]
        c0, c1 = b0[cols], b1[cols]
        sub = _FirthCols(xs, ys, vs, v[:, cols])
        mu, S0, S1, S2, det, hat, ll = sub.pieces(c0, c1)
        r = vs * (ys - mu + hat * (0.5 - mu))
        U0, U1 = r.sum(0), (r * xs).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d0 = (S2 * U0 - S1 * U1) / det
            d1 = (S0 * U1 - S1 * U0) / det
        mx = np.maximum(np.abs(d0), np.abs(d1))
        scale = np.where(mx > FIRTH_MAXSTEP, FIRTH_MAXSTEP / np.where(mx > 0, mx, 1.0), 1.0)
        d0, d1 = d0 * scale, d1 * scale
        n0, n1 = c0 + d0, c1 + d1
        ll_new = sub.pieces(n0, n1)[6]
        for _ in range(FIRTH_MAXHS):
            worse = ~(ll_new >= ll - 1e-12)
            if not worse.any():
                break
            d0 = np.where(worse, 0.5 * d0, d0)
            d1 = np.where(worse, 0.5 * d1, d1)
            n0, n1 = c0 + d0, c1 + d1
            ll_new = sub.pieces(n0, n1)[6]
        bad = ~np.isfinite(n0) | ~np.isfinite(n1)
        b0[cols], b1[cols] = n0, n1
        iters[cols] = it
        done = np.maximum(np.abs(d0), np.abs(d1)) < tol
        conv[cols] = done & ~bad
        active[cols[done | bad]] = False
    return b0, b1, conv, iters


class _FirthCols:
    """The Firth pieces (mu, information, hat values, penalised log-lik) of
    a fixed set of columns."""

    def __init__(self, x, y, vf, v):
        self.x, self.y, self.vf, self.v = x, y, vf, v

    def pieces(self, c0, c1):
        x, y, vf, v = self.x, self.y, self.vf, self.v
        eta = c0[None, :] + c1[None, :] * x
        mu = linkinv(eta)
        w = mu * (1.0 - mu) * vf
        S0, S1, S2 = w.sum(0), (w * x).sum(0), (w * x * x).sum(0)
        det = S0 * S2 - S1 * S1
        with np.errstate(divide="ignore", invalid="ignore"):
            hat = w * (S2[None] - 2.0 * S1[None] * x + S0[None] * x * x) / det[None]
            ll = (np.where(v, y * np.log(mu) + (1.0 - y) * np.log1p(-mu), 0.0).sum(0)
                  + 0.5 * np.log(det))
        return mu, S0, S1, S2, det, hat, ll


# ─────────────────────────────────────────────────────────────────────────────
#  the method on one block
# ─────────────────────────────────────────────────────────────────────────────

def logitide_block(V, Wb, h, opts):
    """Z, MSR, the case of every column and the counts, for one block."""
    sep_rule = opts["separation"]
    T, Pb = V.shape
    hh = np.asarray(h, float)[:, None]
    L = V & ~Wb
    n = V.sum(0)
    nL = L.sum(0)
    nW = Wb.sum(0)
    hmaxL = np.max(np.where(L, hh, -np.inf), axis=0)
    hminL = np.min(np.where(L, hh, np.inf), axis=0)
    hmaxW = np.max(np.where(Wb, hh, -np.inf), axis=0)
    hminW = np.min(np.where(Wb, hh, np.inf), axis=0)
    hmin_v = np.min(np.where(V, hh, np.inf), axis=0)
    hmax_v = np.max(np.where(V, hh, -np.inf), axis=0)
    both = (nL > 0) & (nW > 0) & (n >= int(opts["min_obs_px"]))
    sep = both & (hmaxL <= hminW)
    rev = both & ~sep & (hmaxW <= hminL)
    fitc = both & ~sep & ~rev
    Z = np.full(Pb, np.nan)
    s = np.zeros(Pb)
    nonconv = 0
    if sep_rule == "gap_midpoint":
        Z[sep] = 0.5 * (hmaxL[sep] + hminW[sep])
        s[sep] = -1.0
        Z[rev] = 0.5 * (hmaxW[rev] + hminL[rev])
        s[rev] = 1.0
        cols = np.flatnonzero(fitc)
    else:
        cols = np.flatnonzero(both)
    if cols.size:
        vv = V[:, cols]
        cmean = np.where(vv, hh, 0.0).sum(0) / np.maximum(n[cols], 1)
        x = hh - cmean[None, :]
        y = L[:, cols].astype(float)
        if sep_rule == "gap_midpoint":
            b0, b1, cv, _ = irls_logit(x, y, vv, maxit=opts["maxit"], tol=opts["tol"], halving=True)
            ok = cv & np.isfinite(b1) & (b1 != 0)
            nonconv = int((~cv).sum())
        elif sep_rule == "cap25":
            b0, b1, cv, _ = irls_logit(x, y, vv, maxit=GLM_MAXIT, tol=GLM_EPSILON, halving=False)
            ok = np.isfinite(b1) & (b1 != 0)
            nonconv = int((~cv).sum())
        elif sep_rule == "firth":
            b0, b1, cv, _ = firth_logit(x, y, vv, maxit=opts["maxit"])
            ok = cv & np.isfinite(b1) & (b1 != 0)
            nonconv = int((~cv).sum())
        else:
            raise ValueError(f"separation must be one of {SEPARATION_RULES}")
        with np.errstate(invalid="ignore", divide="ignore"):
            zc = cmean - b0 / b1
        Z[cols] = np.where(ok, zc, np.nan)
        s[cols] = np.where(ok, np.sign(b1), 0.0)
    # modified success rate of the fitted classifier
    with np.errstate(invalid="ignore"):
        pred_land = (s[None, :] * (hh - Z[None, :])) > 0
    srl = np.where(nL > 0, (L & pred_land).sum(0) / np.maximum(nL, 1), np.nan)
    srw = np.where(nW > 0, (Wb & ~pred_land).sum(0) / np.maximum(nW, 1), np.nan)
    msr = 0.5 * srl + 0.5 * srw
    keep, drop = apply_filters(Z, msr, hmin_v, hmax_v, opts["msr_min"])
    case = np.zeros(Pb, np.int8)          # 0 one class / too few obs
    case[sep] = 1                         # separated, land below water
    case[rev] = 2                         # separated, land above water
    case[fitc] = 3                        # not separable: fitted
    counts = {"one_class_or_few": int((~both).sum()), "separated": int(sep.sum()),
              "separated_reversed": int(rev.sum()), "fitted": int(fitc.sum()),
              "slope_positive": int(((s > 0) & np.isfinite(Z)).sum()), "nonconverged": nonconv,
              "dropped_msr": int(drop["msr"].sum()), "dropped_range": int(drop["range"].sum()),
              "kept": int(keep.sum())}
    return np.where(keep, Z, np.nan), msr, case, n, counts


def apply_filters(Z, msr, hmin, hmax, msr_min):
    """LogiTide2DEM's two filters: MSR >= msr_min and Z inside the observed
    range [hmin, hmax]. Returns (keep, {'msr': dropped by MSR, 'range':
    dropped by the range among those MSR kept})."""
    fin = np.isfinite(Z)
    with np.errstate(invalid="ignore"):
        msr_ok = np.isfinite(msr) & (msr >= msr_min)
        in_range = (Z >= hmin) & (Z <= hmax)
    drop_msr = fin & ~msr_ok
    drop_range = fin & msr_ok & ~in_range
    return fin & msr_ok & in_range, {"msr": drop_msr, "range": drop_range}


def run(obs, cfg=None, variant_id=None, **overrides):
    t0 = time.time()
    Y, C, W, h, T, P = base.check_obs(obs)
    opts = base.method_options(cfg, ID, DEFAULTS)
    opts.update(overrides)
    if opts["separation"] not in SEPARATION_RULES:
        raise ValueError(f"separation must be one of {SEPARATION_RULES}")
    meta = base.base_meta(variant_id or ID, T, P, opts)
    if T < int(opts["min_scenes"]):
        return base.not_computable(meta, P, f"LogiTide2DEM needs at least {opts['min_scenes']} images per "
                                            f"period; the window holds {T}", t0)
    z = np.full(P, np.nan)
    msr = np.full(P, np.nan)
    case = np.zeros(P, np.int8)
    nv = np.zeros(P, np.int32)
    counts = {}
    blk = int(base.as_dict(cfg).get("block_px", opts["block_px"]))
    for sl in base.column_blocks(P, blk):
        V, Wb = base.valid_and_wet(Y, C, W, sl)
        zb, mb, cb, nb, cnt = logitide_block(V, Wb, h, opts)
        z[sl], msr[sl], case[sl], nv[sl] = zb, mb, cb, nb
        for k, v in cnt.items():
            counts[k] = counts.get(k, 0) + v
    meta["counts"] = counts
    return base.finish(meta, z, t0, extra={"msr": msr.astype(np.float32), "case": case, "n_valid": nv})


def run_firth(obs, cfg=None):
    return run(obs, cfg, variant_id="E4_firth", separation="firth")


def run_cap25(obs, cfg=None):
    return run(obs, cfg, variant_id="E4_cap25", separation="cap25")


def _variant_card(rule, text):
    c = dict(CARD)
    c["id"] = f"{ID}_{rule}"
    c["label"] = f"{LABEL}, {text}"
    c["role"] = f"sensitivity of E4 (separation rule '{rule}')"
    return c


METHODS = {ID: (LABEL, run, CARD)}
VARIANTS = {
    "E4_firth": (f"{LABEL}, Firth-penalised", run_firth, _variant_card("firth", "Firth-penalised")),
    "E4_cap25": (f"{LABEL}, 25-iteration cap", run_cap25, _variant_card("cap25", "25-iteration cap")),
}
