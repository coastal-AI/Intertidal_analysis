# -*- coding: utf-8 -*-
"""
HSR (Hypsometric Super-Resolution) — prototipo + validación sintética.

Etapa 1: por píxel grueso (10 m), ajusta NDWI(t) = a + b*Phi((h_t - mu)/sigma)
         por mínimos cuadrados sobre una rejilla (mu, sigma) con solución
         analítica de (a,b) — vectorizado sobre todos los píxeles.
Etapa 2: super-resuelve a 2.5 m por proyecciones alternadas:
         suavidad (Jacobi) <-> restricción por bloque (media=mu_p, std=sigma_p).

Validación: DEM fino sintético -> observaciones sintéticas -> ¿recupera HSR
el DEM fino mejor que interpolar mu (lo que haría el estado del arte)?
"""
import numpy as np
from scipy.special import erf
rng = np.random.default_rng(42)

# ═════════════════ 1. DEM fino "verdad" (2.5 m) ═════════════════
K = 4                      # factor de escala (10 m -> 2.5 m)
CH, CW = 60, 60            # píxeles gruesos
FH, FW = CH*K, CW*K        # 240x240 finos
yy, xx = np.mgrid[0:FH, 0:FW].astype(float)

# rampa mareal + canal principal serpenteante + canales secundarios + barras
z = 1.6*(xx/FW) - 0.8
chan = 0.9*np.exp(-((yy - (120+35*np.sin(xx/25)))**2)/(2*9**2))       # canal ppal (ancho ~22 m)
chan2 = 0.45*np.exp(-((xx - (60+20*np.sin(yy/18)))**2)/(2*5**2))      # canal 2 (ancho ~12 m: SUBPIXEL)
bars = 0.18*np.sin(yy/7.0)*np.sin(xx/9.0)                              # barras/ripples subpixel
z_true = (z - chan - chan2 + bars).astype(np.float64)

# ═════════════════ 2. Observación sintética estilo Sentinel ═════════════════
T = 300
tide = (1.9*np.sin(np.arange(T)*0.9) + 0.7*np.sin(np.arange(T)*0.263) +
        rng.normal(0, 0.05, T))                    # mareas + error del modelo de marea 5 cm
A_DRY, B_WET = -0.15, 0.55                          # NDWI seco / (seco->mojado)
NOISE = 0.06                                        # ruido radiométrico NDWI
P_CLOUD = 0.08                                      # 8% de obs con nube residual

zb = z_true.reshape(CH, K, CW, K)
wetfrac = (zb[None] < tide[:, None, None, None, None]).mean(axis=(2, 4))  # (T,CH,CW)
Y = A_DRY + B_WET*wetfrac + rng.normal(0, NOISE, (T, CH, CW))
cloud = rng.random((T, CH, CW)) < P_CLOUD
Y[cloud] = rng.uniform(-0.4, 0.4, int(cloud.sum()))                        # nubes = basura
CLEAR = ~cloud                                                             # máscara de "claro" conocida a medias:
# simulamos que el SCL detecta el 70% de las nubes; el 30% restante es ruido no marcado
detected = cloud & (rng.random(cloud.shape) < 0.7)
CLEAR_OBS = ~detected

mu_true = zb.mean(axis=(1, 3))
sd_true = zb.std(axis=(1, 3))
print(f"DEM fino {z_true.shape} rango [{z_true.min():.2f},{z_true.max():.2f}] m | "
      f"relieve interno medio {sd_true.mean()*100:.1f} cm | mareas [{tide.min():.2f},{tide.max():.2f}]")

# ═════════════════ 3. ETAPA 1 — ajuste (mu, sigma) por MLE-LS vectorizado ═══
def stage1_fit(Y, tide, clear, mu_grid, sg_grid, refine=True):
    T2, H, W = Y.shape
    P = H*W
    Yf = Y.reshape(T2, P)
    Cf = clear.reshape(T2, P).astype(np.float64)
    best_loss = np.full(P, np.inf)
    best = np.zeros((4, P))                       # a, b, mu, sigma
    N = Cf.sum(axis=0)
    Sy = (Yf*Cf).sum(axis=0)
    Syy = (Yf*Yf*Cf).sum(axis=0)
    for mu in mu_grid:
        for sg in sg_grid:
            phi = 0.5*(1.0 + erf((tide - mu)/(sg*np.sqrt(2.0))))   # (T,)
            Sp = Cf.T @ phi                                        # Σ phi     (P,)
            Spp = Cf.T @ (phi*phi)
            Syp = (Yf*Cf).T @ phi
            det = N*Spp - Sp*Sp
            ok = det > 1e-9
            b = np.where(ok, (N*Syp - Sp*Sy)/np.where(ok, det, 1), 0.0)
            a = np.where(ok, (Sy - b*Sp)/np.where(N > 0, N, 1), 0.0)
            loss = Syy - 2*a*Sy - 2*b*Syp + a*a*N + 2*a*b*Sp + b*b*Spp
            upd = ok & (loss < best_loss)
            best_loss = np.where(upd, loss, best_loss)
            for k, v in enumerate((a, b, np.full(P, mu), np.full(P, sg))):
                best[k] = np.where(upd, v, best[k])
    a, b, mu, sg = best
    rmse = np.sqrt(np.maximum(best_loss, 0)/np.maximum(N, 1))
    return (a.reshape(H, W), b.reshape(H, W), mu.reshape(H, W),
            sg.reshape(H, W), rmse.reshape(H, W))

mu_grid = np.linspace(tide.min(), tide.max(), 60)
sg_grid = np.array([0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65])
import time; t0 = time.time()
a1, b1, mu1, sg1, r1 = stage1_fit(Y, tide, CLEAR_OBS, mu_grid, sg_grid)
# IRLS robusto: 1 pasada quitando outliers (>2.5 sigma del residuo)
phi_all = 0.5*(1+erf((tide[:, None, None]-mu1[None])/(sg1[None]*np.sqrt(2))))
resid = Y - (a1[None] + b1[None]*phi_all)
good = CLEAR_OBS & (np.abs(resid) < 2.5*np.maximum(r1[None], 0.02))
a1, b1, mu1, sg1, r1 = stage1_fit(Y, tide, good, mu_grid, sg_grid)
t_fit = time.time()-t0

valid = (b1 > 0.15) & (mu1 > tide.min()+0.1) & (mu1 < tide.max()-0.1)
err_mu = mu1[valid]-mu_true[valid]
print(f"\nETAPA 1 ({t_fit:.1f}s, {valid.sum()} px intermareales validos):")
print(f"  cota mu:    RMSE {np.sqrt((err_mu**2).mean())*100:.1f} cm | bias {err_mu.mean()*100:+.1f} cm")
cc = np.corrcoef(sg1[valid], sd_true[valid])[0, 1]
print(f"  relieve sg: corr(sg_est, sg_true) = {cc:.3f}  <- relieve SUBPIXEL medido, no inventado")

# ═════════════════ 4. ETAPA 2 — super-resolución 2.5 m ═════════════════
def upsample_nn(c):
    return np.repeat(np.repeat(c, K, 0), K, 1)

def stage2_superres(mu, sg, valid, iters=400, lam=0.55):
    zf = upsample_nn(mu).astype(np.float64)
    vf = upsample_nn(valid)
    for _ in range(iters):
        # suavidad (Jacobi con vecinos)
        nb = (np.roll(zf, 1, 0)+np.roll(zf, -1, 0)+np.roll(zf, 1, 1)+np.roll(zf, -1, 1))/4.0
        zf = np.where(vf, (1-lam)*zf + lam*nb, zf)
        # proyeccion por bloque: media=mu_p, std=sg_p
        zb2 = zf.reshape(CH, K, CW, K)
        m = zb2.mean(axis=(1, 3), keepdims=True)
        s = zb2.std(axis=(1, 3), keepdims=True)
        scale = np.where(s > 1e-6, np.minimum(sg[:, None, :, None]/np.maximum(s, 1e-6), 3.0), 1.0)
        zb2 = (zb2 - m)*np.where(valid[:, None, :, None], scale, 1.0) + \
              np.where(valid[:, None, :, None], mu[:, None, :, None], m)
        zf = zb2.reshape(FH, FW)
    return zf

t0 = time.time()
z_hsr = stage2_superres(mu1, sg1, valid)
t_sr = time.time()-t0

# baseline: lo que haria el estado del arte = interpolar mu (bilinear) sin sigma
from scipy.ndimage import zoom
z_base = zoom(mu1, K, order=1)

vf = upsample_nn(valid)
e_hsr = (z_hsr-z_true)[vf]
e_base = (z_base-z_true)[vf]
print(f"\nETAPA 2 ({t_sr:.1f}s) — RMSE del DEM a 2.5 m vs verdad:")
print(f"  baseline (interpolar mu, estado del arte): {np.sqrt((e_base**2).mean())*100:.1f} cm")
print(f"  HSR (mu + sigma + proyecciones):           {np.sqrt((e_hsr**2).mean())*100:.1f} cm")
gain = 100*(1-np.sqrt((e_hsr**2).mean())/np.sqrt((e_base**2).mean()))
print(f"  mejora: {gain:.1f}%")

# detalle subpixel: solo en pixeles con relieve interno alto (canales finos)
hi = upsample_nn(sd_true > 0.15) & vf
if hi.any():
    print(f"  en zonas de detalle subpixel (canal de 12 m, barras):")
    print(f"    baseline {np.sqrt(((z_base-z_true)[hi]**2).mean())*100:.1f} cm | "
          f"HSR {np.sqrt(((z_hsr-z_true)[hi]**2).mean())*100:.1f} cm")

np.savez(r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad\hsr_proto.npz",
         z_true=z_true, z_hsr=z_hsr, z_base=z_base, mu=mu1, sg=sg1, valid=valid,
         mu_true=mu_true, sd_true=sd_true)
print("\nHSR PROTOTYPE DONE")
