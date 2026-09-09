"""
operator.py — Fase M4: el mareógrafo distribuido (el operador T)
================================================================

The deliverable of the whole M-chain, and deliberately NOT a black box: it is
the explicit composition of three measured pieces, each with its own gate
behind it, each inspectable on the returned object:

    h(s, t) = alpha(s) * h_B'(t - tau(s, limb))

* ``h_B'`` — the boundary: any provider (ocean model, ensemble, climatology),
  passed through the M3 mouth-judged correction (gain/lag on the estimable
  constituents only; datum and frozen constituents stay with the prior — the
  affine theorem says imagery cannot audit them, so T never pretends to);
* ``alpha(s)``, ``tau(s)`` — the interior transfer measured by the M2
  estimators from wet/dry imagery alone, piecewise-linear between band
  centres, anchored at the mouth (alpha=1, tau=0 by construction);
* ``limb`` — optionally two lag profiles (flood/ebb), because the measured
  asymmetry (ponding: ebb late where flood is not) is a hysteresis no single
  clock can express.

Anchor and honesty rules baked into the object:

* it NEVER calibrates against a tide gauge (the user's hard constraint;
  gauges are allowed only to VALIDATE at sites that have them);
* every parameter carries its provenance in ``meta`` (which experiment,
  which config, which data hashes), so a map made with this operator can be
  traced to the run that measured each number.
"""

from __future__ import annotations

import numpy as np


class InteriorTide:
    """The distributed tide gauge: water level anywhere along the estuary.

    Parameters
    ----------
    bank : tide_estimators.ShiftBank
        The (possibly M3-corrected) boundary series pre-evaluated on a lag
        grid over the scene instants — the operator works on the archive's
        own time axis, which is where elevations are inverted.
    centers_km, alpha, tau_min : arrays (n_bands,)
        The M2a transfer profiles at band centres (mouth first).
    tau_ebb_min : array or None
        Optional distinct ebb-limb lag profile (hysteresis). ``None`` means
        one clock for both limbs.
    meta : dict
        Provenance: config paths, result-file paths, data hashes.
    """

    def __init__(self, bank, centers_km, alpha, tau_min, tau_ebb_min=None,
                 meta=None):
        self.bank = bank
        self.centers = np.asarray(centers_km, float)
        self.alpha = np.asarray(alpha, float)
        self.tau = np.asarray(tau_min, float)
        self.tau_ebb = (None if tau_ebb_min is None
                        else np.asarray(tau_ebb_min, float))
        self.meta = dict(meta or {})

    # profiles are piecewise-linear in s, clamped at the ends: beyond the
    # last band centre there is no measurement, so the last measured value
    # holds rather than an extrapolation inventing physics
    def alpha_at(self, s_km):
        return np.interp(np.asarray(s_km, float), self.centers, self.alpha)

    def tau_at(self, s_km, limb="flood"):
        prof = (self.tau if (limb == "flood" or self.tau_ebb is None)
                else self.tau_ebb)
        return np.interp(np.asarray(s_km, float), self.centers, prof)

    def level(self, s_km, rising=None):
        """Water level series felt at ``s_km`` — shape (T,) per scalar s,
        (T, P) per array. ``rising`` (T,) selects the limb profile when the
        operator carries hysteresis."""
        s = np.atleast_1d(np.asarray(s_km, float))
        a = self.alpha_at(s)
        t_f = self.tau_at(s, "flood")
        t_e = self.tau_at(s, "ebb")
        T = len(self.bank.at(0.0))
        out = np.empty((T, len(s)))
        for j in range(len(s)):
            hf = self.bank.at(t_f[j])
            if self.tau_ebb is None or rising is None:
                h = hf
            else:
                h = np.where(rising, hf, self.bank.at(t_e[j]))
            out[:, j] = a[j] * h
        return out[:, 0] if np.isscalar(s_km) else out

    def describe(self):
        """The whole operator in one printable table — the anti-black-box."""
        lines = ["InteriorTide (mareógrafo distribuido)",
                 f"  contorno: {self.meta.get('boundary', 'desconocido')}",
                 f"  correccion M3: {self.meta.get('correction', 'ninguna')}",
                 "  s (km)   alpha    tau_flood (min)" +
                 ("   tau_ebb (min)" if self.tau_ebb is not None else "")]
        for i, c in enumerate(self.centers):
            row = f"  {c:6.2f}  {self.alpha[i]:6.3f}   {self.tau[i]:8.1f}"
            if self.tau_ebb is not None:
                row += f"        {self.tau_ebb[i]:8.1f}"
            lines.append(row)
        for k, v in self.meta.get("hashes", {}).items():
            lines.append(f"  sha256 {k}: {str(v)[:16]}")
        return "\n".join(lines)
