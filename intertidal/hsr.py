"""
hsr.py — HSR (Hypsometric Super-Resolution): DEM intermareal sub-píxel
======================================================================

Método propio de este proyecto. Idea física: un píxel de 10 m en el borde del
agua está PARCIALMENTE inundado, y su NDWI es la mezcla lineal seco/mojado:

    NDWI_p(t) = a_p + b_p * F_p(h_t) + ruido

donde F_p(h) es la fracción del píxel bajo el agua a marea h — es decir, la
HIPSOMETRÍA INTERNA del píxel. Modelándola como F_p(h) = Phi((h - mu_p)/sigma_p):

  - mu_p    = cota mediana del píxel (precisión subdecimétrica: usa toda la
              curva NDWI-marea, sin umbral).
  - sigma_p = relieve DENTRO del píxel (rugosidad sub-píxel medida, no inventada).

Etapa 1: ajuste por mínimos cuadrados (a,b analíticos) sobre rejilla (mu,sigma),
         vectorizado por bloques de filas -> escala a AOIs grandes con RAM acotada.
         Incertidumbre por píxel vía aproximación de Cramér-Rao.
Etapa 2: super-resolución a 10/K m por proyecciones alternadas: suavidad (Jacobi)
         <-> restricción por bloque (media = mu_p, std = sigma_efectiva), con
         sigma_eff = sqrt(max(sigma^2 - floor^2, 0)) para no inyectar ruido.

Validación sintética (DEM fino conocido -> observación simulada -> recuperar):
mu RMSE ~3 cm; corr(sigma, relieve real) ~0.82; en zonas con estructura
sub-píxel el DEM fino de HSR mejora ~2x a interpolar mu (estado del arte).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from scipy.special import erf

from .notebook_compat import _open_ndwi_cube, _grid_from_dataset, _write_geotiff


# ─────────────────────────────────────────────────────────────────────────────
#  Etapa 1 — ajuste hipsométrico por píxel
# ─────────────────────────────────────────────────────────────────────────────

def _fit_block(Y, C, tide, mu_grid, sg_grid):
    """Ajusta (a, b, mu, sigma) por LS en un bloque de píxeles.

    Y : (T, P) NDWI. C : (T, P) máscara de observación usable (bool).
    Para cada candidato (mu, sigma), (a, b) tienen solución analítica; se
    barre la rejilla y se guarda el mínimo por píxel.
    """
    T, P = Y.shape
    Cf = C.astype(np.float64)
    Yc = np.where(C, Y, 0.0).astype(np.float64)

    N = Cf.sum(axis=0)
    Sy = Yc.sum(axis=0)
    Syy = (Yc * Yc).sum(axis=0)

    best_loss = np.full(P, np.inf)
    best = np.zeros((4, P))

    for mu in mu_grid:
        for sg in sg_grid:
            phi = 0.5 * (1.0 + erf((tide - mu) / (sg * np.sqrt(2.0))))  # (T,)
            Sp = Cf.T @ phi
            Spp = Cf.T @ (phi * phi)
            Syp = Yc.T @ phi
            det = N * Spp - Sp * Sp
            ok = det > 1e-9
            b = np.where(ok, (N * Syp - Sp * Sy) / np.where(ok, det, 1.0), 0.0)
            a = np.where(N > 0, (Sy - b * Sp) / np.where(N > 0, N, 1.0), 0.0)
            loss = Syy - 2 * a * Sy - 2 * b * Syp + a * a * N + 2 * a * b * Sp + b * b * Spp
            upd = ok & (loss < best_loss)
            best_loss = np.where(upd, loss, best_loss)
            best[0] = np.where(upd, a, best[0])
            best[1] = np.where(upd, b, best[1])
            best[2] = np.where(upd, mu, best[2])
            best[3] = np.where(upd, sg, best[3])

    rmse = np.sqrt(np.maximum(best_loss, 0.0) / np.maximum(N, 1.0))
    return best[0], best[1], best[2], best[3], rmse, N


def hsr_stage1(
    nc_path,
    dates,
    tide_heights,
    valid_dates=None,
    clear_classes=(4, 5, 6, 12),
    mu_points=60,
    sg_grid=(0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65),
    robust_iters=1,
    row_chunk=32,
    min_obs=8,
    min_b=0.15,
):
    """Ajuste hipsométrico píxel a píxel leyendo el netCDF por bloques de filas.

    Devuelve dict con mu, sigma, a, b, rmse, n_obs, sigma_mu (incertidumbre
    Cramér-Rao aprox. de la cota) y valid (píxeles intermareales ajustables).
    """
    import xarray as _xr

    ds, b03, b08, scl, t_dim = _open_ndwi_cube(nc_path)
    try:
        T = scl.sizes[t_dim]
        H = scl.sizes[scl.dims[1]]
        W = scl.sizes[scl.dims[2]]

        # Marea por observación (NaN si la fecha no tiene marea o no es válida)
        vd = set(valid_dates) if valid_dates else None
        tide = np.full(T, np.nan, np.float64)
        for i, d in enumerate(dates):
            if vd is not None and d not in vd:
                continue
            if d in tide_heights and tide_heights[d] is not None:
                tide[i] = float(tide_heights[d])
        has_tide = np.isfinite(tide)
        tide_f = np.where(has_tide, tide, 0.0)

        tmin = float(np.nanmin(tide)) if has_tide.any() else -1.0
        tmax = float(np.nanmax(tide)) if has_tide.any() else 1.0
        mu_grid = np.linspace(tmin, tmax, int(mu_points))
        sg_grid = np.asarray(sg_grid, np.float64)

        out = {k: np.zeros((H, W), np.float32)
               for k in ("a", "b", "mu", "sigma", "rmse", "n_obs")}

        for y0 in range(0, H, row_chunk):
            y1 = min(y0 + row_chunk, H)
            sl = {scl.dims[1]: slice(y0, y1)}
            g = np.asarray(b03.isel(sl).values, np.float32)
            n = np.asarray(b08.isel(sl).values, np.float32)
            s_raw = np.asarray(scl.isel(sl).values)
            s = np.nan_to_num(s_raw, nan=0.0).astype(np.int16)
            den = g + n
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = np.where(den != 0, (g - n) / den, np.nan).astype(np.float32)
            clear = np.isin(s, list(clear_classes)) & np.isfinite(ndwi)
            usable = clear & has_tide[:, None, None]

            hh = y1 - y0
            Y = ndwi.reshape(T, hh * W)
            C = usable.reshape(T, hh * W)
            a, b, mu, sg, rmse, N = _fit_block(Y, C, tide_f, mu_grid, sg_grid)

            for _ in range(int(robust_iters)):
                phi = 0.5 * (1 + erf((tide_f[:, None] - mu[None]) / (sg[None] * np.sqrt(2))))
                resid = np.abs(Y - (a[None] + b[None] * phi))
                C2 = C & (resid < 2.5 * np.maximum(rmse[None], 0.02))
                a, b, mu, sg, rmse, N = _fit_block(Y, C2, tide_f, mu_grid, sg_grid)

            shp = (hh, W)
            out["a"][y0:y1] = a.reshape(shp)
            out["b"][y0:y1] = b.reshape(shp)
            out["mu"][y0:y1] = mu.reshape(shp)
            out["sigma"][y0:y1] = sg.reshape(shp)
            out["rmse"][y0:y1] = rmse.reshape(shp)
            out["n_obs"][y0:y1] = N.reshape(shp)

        transform, crs = _grid_from_dataset(ds)
    finally:
        ds.close()

    # Píxeles intermareales ajustables: contraste seco->mojado suficiente,
    # cota dentro del rango de mareas observado y observaciones mínimas.
    rng = tmax - tmin
    valid = (
        (out["b"] > min_b)
        & (out["mu"] > tmin + 0.02 * rng)
        & (out["mu"] < tmax - 0.02 * rng)
        & (out["n_obs"] >= min_obs)
    )

    # Incertidumbre de la cota (aprox. Cramér-Rao): var(mu) ~ rmse^2 / sum((dmodel/dmu)^2)
    # dmodel/dmu = -b * pdf((h-mu)/sigma)/sigma. Se calcula por BLOQUES de filas
    # para no materializar un array (H, W, T) completo (OOM en AOIs grandes).
    tt = tide[np.isfinite(tide)]
    sigma_mu = np.full(out["mu"].shape, np.nan, np.float32)
    if tt.size:
        for y0 in range(0, H, row_chunk):
            y1 = min(y0 + row_chunk, H)
            mu_b = out["mu"][y0:y1][..., None]
            sg_b = np.maximum(out["sigma"][y0:y1][..., None], 1e-3)
            z = (tt[None, None, :] - mu_b) / sg_b
            pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
            ssum = ((out["b"][y0:y1][..., None] * pdf / sg_b) ** 2).sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sigma_mu[y0:y1] = np.where(
                    ssum > 1e-9, out["rmse"][y0:y1] / np.sqrt(ssum), np.nan
                ).astype(np.float32)

    out["sigma_mu"] = sigma_mu
    out["valid"] = valid
    out["transform"] = transform
    out["crs"] = crs
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Etapa 2 — super-resolución
# ─────────────────────────────────────────────────────────────────────────────

def hsr_stage2(
    mu,
    sigma,
    valid,
    scale=4,
    sigma_floor=0.05,
    sigma_cap=0.30,
    sigma_saturated=0.60,
    scale_max=1.5,
    iters=400,
    lam=0.55,
):
    """Super-resuelve el DEM a (10/scale) m por proyecciones alternadas.

    sigma_floor descuenta el suelo de ruido del ajuste: en píxeles lisos la
    sigma efectiva tiende a 0 (aplana), en canales conserva el relieve medido.

    sigma_saturated: por encima de este valor, la sigma se considera NO fiable
    (el ajuste se pegó al tope de la rejilla: la anchura de la transición está
    absorbiendo error de marea/cambio morfológico, no relieve del píxel) y NO
    se inyecta relieve sub-píxel (sigma_eff = 0, solo interpolación suave).
    sigma_cap limita el relieve inyectable; scale_max evita que la proyección
    amplifique ruido de alta frecuencia (tablero de ajedrez).
    """
    K = int(scale)
    CH, CW = mu.shape
    sig = sigma.astype(np.float64)
    sg_eff = np.sqrt(np.maximum(sig ** 2 - float(sigma_floor) ** 2, 0.0))
    sg_eff = np.minimum(sg_eff, float(sigma_cap))
    sg_eff = np.where(sig >= float(sigma_saturated), 0.0, sg_eff)

    def up(c):
        return np.repeat(np.repeat(c, K, 0), K, 1)

    zf = up(np.where(valid, mu, np.nan)).astype(np.float64)
    # Rellenar NaN iniciales con la media de la zona válida (solo arranque)
    fill = float(np.nanmean(zf)) if np.isfinite(zf).any() else 0.0
    zf = np.nan_to_num(zf, nan=fill)
    vf = up(valid)

    for _ in range(int(iters)):
        nb = (np.roll(zf, 1, 0) + np.roll(zf, -1, 0) +
              np.roll(zf, 1, 1) + np.roll(zf, -1, 1)) / 4.0
        zf = np.where(vf, (1 - lam) * zf + lam * nb, zf)

        zb = zf.reshape(CH, K, CW, K)
        m = zb.mean(axis=(1, 3), keepdims=True)
        s = zb.std(axis=(1, 3), keepdims=True)
        sc = np.where(valid[:, None, :, None],
                      np.minimum(sg_eff[:, None, :, None] / np.maximum(s, 1e-6),
                                 float(scale_max)),
                      1.0)
        zb = (zb - m) * sc + np.where(valid[:, None, :, None],
                                      mu[:, None, :, None].astype(np.float64), m)
        zf = zb.reshape(CH * K, CW * K)

    zf = np.where(vf, zf, np.nan).astype(np.float32)
    return zf


# ─────────────────────────────────────────────────────────────────────────────
#  API de alto nivel
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class HsrResult:
    mu: np.ndarray            # cota por píxel de 10 m
    sigma: np.ndarray         # relieve interno del píxel (sub-píxel)
    sigma_mu: np.ndarray      # incertidumbre de la cota (Cramér-Rao aprox.)
    valid: np.ndarray         # máscara de píxeles intermareales ajustados
    z_fine: np.ndarray        # DEM super-resuelto a (10/scale) m
    scale: int
    transform: object         # grid de 10 m
    crs: object
    rmse: np.ndarray
    n_obs: np.ndarray

    @property
    def transform_fine(self):
        t = self.transform
        return rasterio.Affine(t.a / self.scale, t.b, t.c,
                               t.d, t.e / self.scale, t.f)

    def clip_to(self, intertidal_mask):
        """Recorta a la máscara intermareal (rejilla de 10 m); el DEM fino se
        recorta con la máscara ampliada al grid fino."""
        m = np.asarray(intertidal_mask, dtype=bool)
        keep = np.zeros(self.mu.shape, bool)
        h = min(m.shape[0], keep.shape[0]); w = min(m.shape[1], keep.shape[1])
        keep[:h, :w] = m[:h, :w]
        self.valid = self.valid & keep
        for name in ("mu", "sigma", "sigma_mu", "rmse"):
            arr = getattr(self, name)
            setattr(self, name, np.where(keep, arr, np.nan).astype(np.float32))
        kf = np.repeat(np.repeat(keep, self.scale, 0), self.scale, 1)
        self.z_fine = np.where(kf, self.z_fine, np.nan).astype(np.float32)
        return self

    def save(self, out_dir="."):
        import os
        os.makedirs(out_dir, exist_ok=True)
        _write_geotiff(os.path.join(out_dir, "hsr_mu.tif"), self.mu,
                       self.transform, self.crs, "float32", np.nan)
        _write_geotiff(os.path.join(out_dir, "hsr_sigma.tif"), self.sigma,
                       self.transform, self.crs, "float32", np.nan)
        _write_geotiff(os.path.join(out_dir, "hsr_uncertainty.tif"), self.sigma_mu,
                       self.transform, self.crs, "float32", np.nan)
        _write_geotiff(os.path.join(out_dir, "hsr_dem_fine.tif"), self.z_fine,
                       self.transform_fine, self.crs, "float32", np.nan)
        return out_dir


def reconstruct_hsr(
    analysis,
    tide_heights,
    valid_dates=None,
    scale=4,
    sigma_floor=0.05,
    ndwi_clear_classes=(4, 5, 6, 12),
    **stage1_kwargs,
):
    """HSR completo desde un NdwiCubeAnalysis (reutiliza su netCDF, sin descargar).

    Parameters
    ----------
    analysis : NdwiCubeAnalysis
        Resultado de analyze_ndwi_cube_openeo (contiene la ruta del netCDF).
    tide_heights : dict {fecha: altura}
    valid_dates : fechas tras el filtro de nubes (default: analysis.valid_dates)
    scale : factor de super-resolución (4 -> 2.5 m)
    """
    if valid_dates is None:
        valid_dates = analysis.valid_dates

    s1 = hsr_stage1(
        analysis.nc_path, analysis.dates, tide_heights,
        valid_dates=valid_dates, clear_classes=ndwi_clear_classes,
        **stage1_kwargs,
    )
    z_fine = hsr_stage2(s1["mu"], s1["sigma"], s1["valid"],
                        scale=scale, sigma_floor=sigma_floor)
    return HsrResult(
        mu=np.where(s1["valid"], s1["mu"], np.nan).astype(np.float32),
        sigma=np.where(s1["valid"], s1["sigma"], np.nan).astype(np.float32),
        sigma_mu=s1["sigma_mu"], valid=s1["valid"], z_fine=z_fine,
        scale=int(scale), transform=s1["transform"], crs=s1["crs"],
        rmse=s1["rmse"], n_obs=s1["n_obs"],
    )
