"""
visualization.py — Visualización de datos intermareales y marea
==============================================================

Módulo para generación de gráficos y mapas de análisis intermareal.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import dates as mdates
import rasterio
from rasterio.mask import geometry_mask
import geopandas as gpd
import contextily as ctx
import pandas as pd


class Visualizer:
    """
    Herramientas de visualización para análisis intermareal.
    
    Todas las funciones son estáticas y pueden usarse sin instanciar
    la clase. Incluye visualizaciones de RGB, SCL, reference maps,
    water frequency, series de marea, etc.
    
    Examples
    --------
    >>> from intertidal import Visualizer
    >>> 
    >>> # Visualizar RGB con SCL
    >>> Visualizer.plot_rgb_with_scl(
    ...     "2024-07-02",
    ...     "tifs_rgb",
    ...     "tifs_scl",
    ...     polygon=aoi_polygon
    ... )
    >>> 
    >>> # Visualizar reference map
    >>> Visualizer.plot_reference_map(ref_map)
    """
    
    # Paleta de colores SCL oficial ESA + clase personalizada
    SCL_COLORS = {
        0:  ("Sin datos",           "#000000"),
        1:  ("Saturado/Defectuoso", "#ff0000"),
        2:  ("Sombra oscura",       "#2f2f2f"),
        3:  ("Sombra nube",         "#643200"),
        4:  ("Vegetación",          "#00a000"),
        5:  ("No vegetación",       "#ffe65a"),
        6:  ("Agua",                "#0000ff"),
        7:  ("Incierto",            "#808080"),
        8:  ("Nube media",          "#c0c0c0"),
        9:  ("Nube alta",           "#ffffff"),
        10: ("Cirrus",              "#64c8ff"),
        11: ("Nieve/Hielo",         "#ff96ff"),
        12: ("Vegetación inundada", "#009999"),  # Verde-azulado (marismas)
    }
    
    @staticmethod
    def plot_rgb_with_scl(
        date: str,
        rgb_dir: str,
        scl_dir: str,
        polygon=None,
        scl_stats: dict = None,
        scl_max_bad_fraction: float = 0.20,
        figsize: tuple = (16, 6)
    ):
        """
        Visualiza RGB y SCL lado a lado.
        
        Parameters
        ----------
        date : str
            Fecha 'YYYY-MM-DD'
        rgb_dir : str
            Directorio con archivos rgb_{date}.tif
        scl_dir : str
            Directorio con archivos scl_{date}.tif
        polygon : shapely.Polygon, optional
            Polígono AOI para recortar visualización
        scl_stats : dict, optional
            Estadísticas SCL de la fecha
        scl_max_bad_fraction : float, optional
            Umbral de píxeles malos (default: 0.20)
        figsize : tuple, optional
            Tamaño de figura (default: (16, 6))
            
        Examples
        --------
        >>> Visualizer.plot_rgb_with_scl(
        ...     "2024-07-02",
        ...     "tifs_rgb",
        ...     "tifs_scl",
        ...     polygon=aoi_polygon,
        ...     scl_stats=stats_dict
        ... )
        """
        from .raster import RasterProcessor
        
        rgb_path = os.path.join(rgb_dir, f"rgb_{date}.tif")
        scl_path = os.path.join(scl_dir, f"scl_{date}.tif")
        
        if not os.path.exists(rgb_path) or not os.path.exists(scl_path):
            print(f"Datos no disponibles para {date}")
            return
        
        # Leer RGB y SCL
        rgb_arr = RasterProcessor.read_rgb(rgb_path)
        scl_arr = RasterProcessor.read_scl(scl_path)
        
        # Aplicar máscara de AOI si se proporciona
        if polygon is not None:
            with rasterio.open(scl_path) as src:
                poly_proj = (
                    gpd.GeoSeries([polygon], crs="EPSG:4326")
                    .to_crs(src.crs)
                    .iloc[0]
                )
                mask = geometry_mask(
                    [poly_proj],
                    transform=src.transform,
                    invert=True,
                    out_shape=(src.height, src.width),
                )
                rgb_arr[~mask] = 0
                scl_arr[~mask] = np.nan
        
        # Crear figura
        fig, (ax_rgb, ax_scl, ax_legend) = plt.subplots(
            1, 3, figsize=figsize, gridspec_kw={"width_ratios": [3, 3, 1]}
        )
        
        # Panel RGB
        ax_rgb.imshow(rgb_arr)
        ax_rgb.set_title(f"RGB — {date}", fontsize=12, fontweight="bold")
        ax_rgb.axis("off")
        
        # Panel SCL
        scl_float = scl_arr.astype(float)
        color_list = [Visualizer.SCL_COLORS.get(c, ("?", "#aaaaaa"))[1] for c in range(13)]
        cmap = mcolors.ListedColormap(color_list)
        cmap.set_bad("white")
        norm = mcolors.BoundaryNorm(boundaries=list(range(14)), ncolors=13)
        
        ax_scl.imshow(scl_float, cmap=cmap, norm=norm, interpolation="nearest")
        ax_scl.set_title(f"SCL — {date}", fontsize=12, fontweight="bold")
        ax_scl.axis("off")
        
        # Añadir estadísticas si están disponibles
        if scl_stats and date in scl_stats:
            stats = scl_stats[date]
            status = "VÁLIDA" if stats['bad_fraction'] <= scl_max_bad_fraction else "DESCARTADA"
            ax_scl.set_xlabel(
                f"Píxeles malos: {stats['bad_pct']:.1f}%  |  {status}",
                fontsize=10,
            )
        
        # Leyenda de clases SCL presentes
        classes_present = sorted([int(c) for c in np.unique(scl_arr[np.isfinite(scl_arr)])])
        patches = []
        for c in classes_present:
            label, hex_color = Visualizer.SCL_COLORS.get(c, (f"Clase {c}", "#aaaaaa"))
            patches.append(mpatches.Patch(color=hex_color, label=f"{c} — {label}"))
        
        ax_legend.legend(
            handles=patches,
            loc="center",
            fontsize=8,
            frameon=False,
            title="Clases SCL",
            title_fontsize=9,
        )
        ax_legend.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_scl_map(
        date: str,
        scl_dir: str,
        bad_classes: list = None,
        scl_stats: dict = None,
        scl_max_bad_fraction: float = 0.20,
        polygon=None,
        figsize: tuple = (12, 5)
    ):
        """
        Visualiza solo el mapa SCL con leyenda.
        
        Parameters
        ----------
        date : str
            Fecha 'YYYY-MM-DD'
        scl_dir : str
            Directorio con archivos scl_{date}.tif
        bad_classes : list, optional
            Clases consideradas malas
        scl_stats : dict, optional
            Estadísticas de calidad
        scl_max_bad_fraction : float, optional
            Umbral de píxeles malos (default: 0.20)
        polygon : shapely.Polygon, optional
            Polígono AOI para recortar
        figsize : tuple, optional
            Tamaño de figura (default: (12, 5))
        """
        from .raster import RasterProcessor
        
        if bad_classes is None:
            bad_classes = [3, 8, 9, 10, 11]
        
        scl_path = os.path.join(scl_dir, f"scl_{date}.tif")
        if not os.path.exists(scl_path):
            print(f"SCL no disponible para {date}")
            return
        
        scl_arr = RasterProcessor.read_scl(scl_path)
        
        # Aplicar máscara AOI si existe
        if polygon is not None:
            with rasterio.open(scl_path) as src:
                poly_proj = (
                    gpd.GeoSeries([polygon], crs="EPSG:4326")
                    .to_crs(src.crs)
                    .iloc[0]
                )
                mask = geometry_mask(
                    [poly_proj],
                    transform=src.transform,
                    invert=True,
                    out_shape=(src.height, src.width),
                )
                scl_arr[~mask] = np.nan
        
        # Configurar colores
        color_list = [Visualizer.SCL_COLORS.get(c, ("?", "#aaaaaa"))[1] for c in range(13)]
        cmap = mcolors.ListedColormap(color_list)
        cmap.set_bad("white")
        norm = mcolors.BoundaryNorm(boundaries=list(range(14)), ncolors=13)
        
        # Crear figura
        fig, (ax_scl, ax_legend) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
        )
        
        ax_scl.imshow(scl_arr.astype(float), cmap=cmap, norm=norm, interpolation="nearest")
        ax_scl.set_title(f"SCL — {date}", fontsize=11, fontweight="bold")
        ax_scl.axis("off")
        
        # Leyenda
        classes_present = sorted([int(c) for c in np.unique(scl_arr[np.isfinite(scl_arr)])])
        patches = []
        for c in classes_present:
            label, hex_color = Visualizer.SCL_COLORS.get(c, (f"Clase {c}", "#aaaaaa"))
            is_bad = c in bad_classes
            patches.append(
                mpatches.Patch(
                    color=hex_color,
                    label=f"{c} — {label}{' MALO' if is_bad else ''}",
                )
            )
        
        ax_legend.legend(
            handles=patches,
            loc="center",
            fontsize=8,
            frameon=False,
            title="Clases presentes",
            title_fontsize=9,
        )
        ax_legend.axis("off")
        
        # Estadísticas
        if scl_stats and date in scl_stats:
            stats = scl_stats[date]
            status = "VÁLIDA" if stats['bad_fraction'] <= scl_max_bad_fraction else "DESCARTADA"
            ax_scl.set_xlabel(
                f"Píxeles malos: {stats['bad_pct']:.1f}%  |  {status}",
                fontsize=10,
            )
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_reference_map(
        reference_map: np.ndarray,
        figsize: tuple = (8, 8)
    ):
        """
        Visualiza reference map con tres colores.
        
        Parameters
        ----------
        reference_map : ndarray
            Array uint8 (H, W) con valores:
            - 0: Transición (rojo)
            - 1: Agua estable (azul)
            - 2: Tierra estable (arena)
        figsize : tuple, optional
            Tamaño de figura (default: (8, 8))
            
        Examples
        --------
        >>> Visualizer.plot_reference_map(ref_map)
        """
        cmap = mcolors.ListedColormap(["#ff0000", "#0066ff", "#d2b48c"])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        
        plt.figure(figsize=figsize)
        plt.imshow(reference_map, cmap=cmap, norm=norm, interpolation="nearest")
        
        # Leyenda
        legend = [
            mpatches.Patch(color="#ff0000", label="Transición (intermareal)"),
            mpatches.Patch(color="#0066ff", label="Agua estable"),
            mpatches.Patch(color="#d2b48c", label="Tierra estable"),
        ]
        
        plt.legend(handles=legend, loc="upper right")
        plt.title("Mapa de Referencia — Estabilidad Costera", fontsize=12, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_water_frequency(
        water_freq: np.ndarray,
        transform=None,
        crs: str = "EPSG:4326",
        title: str = None,
        polygon=None,
        intertidal_threshold: float = 0.5,
        despeckle_min_pixels: int = 6,
        min_water_patch_pixels: int = 20,
        water_presence_threshold: float = 0.15,
        zoom_to_aoi: bool = True,
        figsize: tuple = (10, 8),
    ):
        """
        Visualiza mapa de water frequency con el mismo estilo que utils
        (plot_scl_map / plot_rgb_grid): recorte al polígono del AOI, fondo
        blanco fuera de la ría, zoom al AOI y limpieza de píxeles aislados.

        Rampa Blues: blanco (#ffffff) = frecuencia 0 (tierra) →
        azul oscuro (#08306b) = frecuencia 1 (agua permanente).

        Parameters
        ----------
        water_freq : ndarray
            Array float (H, W) con valores [0, 1] (o NaN)
        transform : Affine, optional
            Transformada afín del raster (para el extent y la máscara del AOI)
        crs : str, optional
            Sistema de referencia del raster (para reproyectar el polígono)
        title : str, optional
            Título del gráfico
        polygon : shapely Polygon, optional
            Polígono del AOI en EPSG:4326. Si se pasa, todo lo que quede fuera
            se pinta en blanco y se hace zoom a su extensión (como en utils).
        intertidal_threshold : float, optional
            Frecuencia del contorno amarillo (default: 0.5)
        despeckle_min_pixels : int, optional
            Elimina islas de datos aisladas menores que este nº de píxeles
            (huecos de datos por nubes residuales). 0 = desactivado.
        min_water_patch_pixels : int, optional
            Elimina parches de AGUA aislados menores que este nº de píxeles
            (falsos positivos: sombras, píxeles sueltos dentro del estuario).
            Equivalente a remove_isolated_water de Fitton. 0 = desactivado.
        water_presence_threshold : float, optional
            Frecuencia por encima de la cual un píxel se considera "agua" al
            evaluar el tamaño de los parches (default: 0.15).
        zoom_to_aoi : bool, optional
            Ajustar los límites del eje al polígono del AOI (default: True)
        figsize : tuple, optional
            Tamaño de figura (default: (10, 8))
        """
        wf = np.array(water_freq, dtype=np.float32, copy=True)

        # ── Limpieza de ruido: quitar islas de datos aisladas ─────────────────
        if despeckle_min_pixels and despeckle_min_pixels > 0:
            try:
                from scipy import ndimage

                finite = np.isfinite(wf)
                labels, n = ndimage.label(finite)
                if n > 0:
                    sizes = ndimage.sum(
                        np.ones_like(labels, dtype=np.int32),
                        labels,
                        index=np.arange(1, n + 1),
                    )
                    small_ids = np.where(sizes < despeckle_min_pixels)[0] + 1
                    if small_ids.size:
                        wf[np.isin(labels, small_ids)] = np.nan
            except Exception:
                pass

        # ── Quitar parches de agua aislados (falsos positivos) ────────────────
        # Etiqueta las regiones donde hay "agua" (freq > umbral) y descarta las
        # que tengan menos de min_water_patch_pixels píxeles, poniéndolas a 0
        # (tierra). El canal principal, al ser grande y conexo, se conserva.
        if min_water_patch_pixels and min_water_patch_pixels > 0:
            try:
                from scipy import ndimage

                water_bin = np.nan_to_num(wf, nan=0.0) > water_presence_threshold
                wlabels, wn = ndimage.label(water_bin)
                if wn > 0:
                    wsizes = ndimage.sum(
                        np.ones_like(wlabels, dtype=np.int32),
                        wlabels,
                        index=np.arange(1, wn + 1),
                    )
                    small_water = np.where(wsizes < min_water_patch_pixels)[0] + 1
                    if small_water.size:
                        wf[np.isin(wlabels, small_water)] = 0.0
            except Exception:
                pass

        # ── Máscara del AOI (recorte al polígono, como en utils) ──────────────
        if polygon is not None and transform is not None and crs is not None:
            try:
                poly_proj = (
                    gpd.GeoSeries([polygon], crs="EPSG:4326")
                    .to_crs(crs)
                    .iloc[0]
                )
                aoi_mask = geometry_mask(
                    [poly_proj],
                    transform=transform,
                    invert=True,
                    out_shape=wf.shape,
                )
                wf[~aoi_mask] = np.nan
            except Exception:
                poly_proj = None
        else:
            poly_proj = None

        # Rampa Blues con NaN/tierra en blanco
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("white")

        # Extent georreferenciado si hay transform, si no en píxeles
        extent = None
        if transform is not None:
            extent = [
                transform[2],
                transform[2] + transform[0] * wf.shape[1],
                transform[5] + transform[4] * wf.shape[0],
                transform[5],
            ]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")

        im = ax.imshow(
            wf,
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )

        # Contorno amarillo en el umbral intertidal
        try:
            ax.contour(
                wf,
                levels=[intertidal_threshold],
                colors="yellow",
                linewidths=1.0,
                extent=extent,
                origin="upper",
            )
        except Exception:
            pass

        # Zoom al AOI para que la ría llene el encuadre (como plot_scl_map)
        if zoom_to_aoi and poly_proj is not None and extent is not None:
            minx, miny, maxx, maxy = poly_proj.bounds
            pad_x = (maxx - minx) * 0.05
            pad_y = (maxy - miny) * 0.05
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

        # Barra de color con etiquetas de porcentaje
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Frecuencia de agua", rotation=270, labelpad=20)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])

        ax.set_title(
            title or "Water Frequency Map",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_axis_off()

        plt.tight_layout()
        plt.show()
    
    def plot_intertidal_map(
        intertidal_mask,
        wf_transform,
        wf_crs,
        intertidal_out_path,
        low_threshold,
        high_threshold,
        area_km2,
        title
    ):
        height, width = intertidal_mask.shape
        with rasterio.open(
            intertidal_out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=rasterio.uint8,
            crs=wf_crs,
            transform=wf_transform,
        ) as dst:
            dst.write(intertidal_mask.astype(rasterio.uint8), 1)
        # Visualization
        
        # ── Visualización final ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(intertidal_mask, cmap="Blues")
        ax.set_title(
            f"¨{title}\n"
            f"Umbral: WF ∈ [{low_threshold:.2f}, {high_threshold:.2f}]  |  "
            f"{area_km2:.2f} km²"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rgb_grid(
        dates: list[str],
        rgb_dir: str,
        ncols: int = 4,
        figsize: tuple = (16, 12),
        polygon=None
    ):
        """
        Visualiza grid de imágenes RGB.
        
        Parameters
        ----------
        dates : list[str]
            Lista de fechas a visualizar
        rgb_dir : str
            Directorio con archivos rgb_{date}.tif
        ncols : int, optional
            Número de columnas en el grid (default: 4)
        figsize : tuple, optional
            Tamaño de figura (default: (16, 12))
            
        Examples
        --------
        >>> Visualizer.plot_rgb_grid(
        ...     clean_dates[:12],
        ...     "tifs_rgb",
        ...     ncols=4
        ... )
        """
        from .raster import RasterProcessor
        
        nrows = (len(dates) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, date in enumerate(dates):
            rgb_path = os.path.join(rgb_dir, f"rgb_{date}.tif")
            
            if os.path.exists(rgb_path):
                rgb = RasterProcessor.read_rgb(rgb_path)
                if polygon is not None:
                    with rasterio.open(rgb_path) as src:
                        poly_proj = (
                            gpd.GeoSeries([polygon], crs="EPSG:4326")
                            .to_crs(src.crs)
                            .iloc[0]
                        )
                        mask = geometry_mask(
                            [poly_proj],
                            transform=src.transform,
                            invert=True,
                            out_shape=(src.height, src.width),
                        )
                        rgb[~mask] = 1
                axes[idx].imshow(rgb)
                axes[idx].set_title(date, fontsize=9)
            else:
                axes[idx].text(
                    0.5, 0.5, "No disponible",
                    ha="center", va="center",
                    fontsize=10, color="red"
                )
            
            axes[idx].axis("off")
        
        # Ocultar ejes vacíos
        for idx in range(len(dates), len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_bathymetry(
        elevation: np.ndarray,
        confidence: np.ndarray | None = None,
        hillshade: np.ndarray | None = None,
        contour_interval: float = 0.20,
        title: str = "Bathymetry",
        figsize: tuple = (11, 10),
        alpha_min: float = 0.35,
        alpha_max: float = 1.0,
        min_confidence: float = 0.0,
    ):
        """
        Publication-quality bathymetry visualization.
        La función representa el DEM tal y como ha sido reconstruido,
        sin modificar los datos. La confianza únicamente modula la
        transparencia de los píxeles.

        min_confidence
            Umbral por debajo del cual un píxel se considera "sin dato"
            (relleno) y se excluye del hillshade, del alpha y de los
            contornos, aunque el DEM tenga un valor numérico finito ahí.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource

        dem = elevation.astype(float)
        finite = np.isfinite(dem)

        # --------------------------------------------------------------
        # Máscara de validez real: no basta con isfinite(dem), porque el
        # raster puede tener un valor de relleno (p.ej. ~0) fuera de las
        # zonas reconstruidas. Usamos la confianza para distinguir dato
        # real de relleno.
        # --------------------------------------------------------------
        if confidence is not None:
            conf = confidence.astype(float)
            valid = finite & np.isfinite(conf) & (conf > min_confidence)
        else:
            valid = finite

        # --------------------------------------------------------------
        # Colormap
        # --------------------------------------------------------------
        cmap = plt.cm.gist_earth.copy()
        cmap.set_bad("white")

        # --------------------------------------------------------------
        # Hillshade (enmascarado a la zona con dato real)
        # --------------------------------------------------------------
        if hillshade is None:
            ls = LightSource(azdeg=315, altdeg=45)
            dem_masked = np.ma.masked_array(dem, mask=~valid)
            shaded = ls.shade(
                dem_masked,
                cmap=cmap,
                vert_exag=1.3,
                blend_mode="overlay",
            )
        else:
            if hillshade.ndim == 2:
                shaded = hillshade / np.nanmax(hillshade)
            else:
                shaded = hillshade

        hillshade_alpha = np.where(valid, 0.40, 0.0)

        # --------------------------------------------------------------
        # Transparencia del DEM a partir de la confianza
        # --------------------------------------------------------------
        if confidence is None:
            alpha = np.where(valid, 0.85, 0.0)
        else:
            conf_vals = conf[valid]
            if conf_vals.size > 0:
                cmin, cmax = np.nanpercentile(conf_vals, [1, 99])
                if cmax <= cmin:
                    cmin, cmax = np.nanmin(conf_vals), np.nanmax(conf_vals)
            else:
                cmin, cmax = 0.0, 1.0

            rng = (cmax - cmin) if (cmax - cmin) > 1e-12 else 1.0
            conf_norm = np.zeros_like(dem, dtype=float)
            conf_norm[valid] = np.clip((conf[valid] - cmin) / rng, 0, 1)

            alpha = np.zeros_like(dem, dtype=float)
            alpha[valid] = alpha_min + (alpha_max - alpha_min) * conf_norm[valid]

        # --------------------------------------------------------------
        # Plot
        # --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        # cmap="gray" solo aplica si 'shaded' es 2D (hillshade normalizado);
        # si es RGBA (salida de LightSource.shade) matplotlib lo ignora.
        ax.imshow(shaded, origin="upper", alpha=hillshade_alpha, cmap="gray")
        im = ax.imshow(dem, cmap=cmap, origin="upper", alpha=alpha)

        # --------------------------------------------------------------
        # Contornos (solo en la zona válida)
        # --------------------------------------------------------------
        if np.count_nonzero(valid) > 50:
            dem_contour = np.where(valid, dem, np.nan)
            vmin = np.nanmin(dem_contour)
            vmax = np.nanmax(dem_contour)
            levels = np.arange(
                np.floor(vmin / contour_interval) * contour_interval,
                np.ceil(vmax / contour_interval) * contour_interval,
                contour_interval,
            )
            ax.contour(dem_contour, levels=levels, colors="0.25", linewidths=0.30, alpha=0.40)

        # --------------------------------------------------------------
        # Colorbar
        # --------------------------------------------------------------
        cbar = plt.colorbar(im, ax=ax, shrink=0.86, pad=0.03)
        cbar.set_label("Elevation (m)", fontsize=12)

        # --------------------------------------------------------------
        # Layout
        # --------------------------------------------------------------
        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_bathymetry_uncertainty(
        confidence: np.ndarray,
        mask: np.ndarray,
        elevation: np.ndarray | None = None,
        figsize: tuple = (10, 10),
    ):
        """
        Visualiza la confianza de la reconstrucción, restringida a la
        máscara intermareal real (calculada externamente, p.ej. por
        umbral de water frequency).

        Parameters
        ----------
        confidence
            Raster de confianza, denso (mismo grid que la máscara).
        mask
            Máscara booleana de la zona intermareal real (True = dentro).
            Debe venir ya calculada (p.ej. por umbral WF ∈ [0.01, 0.99]).
        elevation
            Opcional, solo para excluir además píxeles sin dato de
            elevación dentro de la máscara.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import distance_transform_edt

        conf = confidence.astype(float).copy()
        valid = mask.astype(bool).copy()

        if elevation is not None:
            valid &= np.isfinite(elevation)

        # ------------------------------------------------------------
        # Relleno de pequeños huecos dentro de la máscara real mediante
        # vecino más cercano (sin triangulación -> sin facetas).
        # ------------------------------------------------------------
        conf_filled = conf.copy()
        holes = valid & ~np.isfinite(conf)
        if holes.any():
            # índices del píxel válido más cercano para cada hueco
            _, (iy, ix) = distance_transform_edt(
                ~(valid & np.isfinite(conf)),
                return_indices=True,
            )
            conf_filled[holes] = conf[iy[holes], ix[holes]]

        conf_display = np.where(valid, conf_filled, np.nan)

        # ------------------------------------------------------------
        # Escalado automático (solo sobre la zona válida real)
        # ------------------------------------------------------------
        valid_vals = conf_display[valid]
        if valid_vals.size > 0:
            vmin = np.nanpercentile(valid_vals, 5)
            vmax = np.nanpercentile(valid_vals, 95)
        else:
            vmin, vmax = 0.0, 1.0

        # ------------------------------------------------------------
        # Figura
        # ------------------------------------------------------------
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad("white")

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            conf_display,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        levels = np.linspace(vmin, vmax, 8)
        ax.contour(conf_display, levels=levels, colors="k", linewidths=0.3, alpha=0.35)

        cb = plt.colorbar(im, ax=ax, shrink=0.82)
        cb.set_label("Confidence", fontsize=11)

        ax.set_title("Bathymetry confidence", fontsize=18, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_bathymetry_profile(
        elevation: np.ndarray,
        mask: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        pixel_size: float = 10.0,
        smooth_sigma: float = 2.0,
        figsize: tuple = (12, 5),
    ):
        """
        Representa un transecto batimétrico entre dos puntos.

        Parameters
        ----------
        elevation
            DEM batimétrico (puede tener valores de relleno fuera de la
            zona intermareal real).
        mask
            Máscara booleana de la zona intermareal real (True = dentro).
        start
            (row, col) inicial.
        end
            (row, col) final.
        pixel_size
            Tamaño del píxel (m).
        smooth_sigma
            Suavizado gaussiano aplicado únicamente para visualización.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import map_coordinates, gaussian_filter1d, distance_transform_edt

        dem_raw = elevation.astype(float).copy()
        valid = mask.astype(bool).copy()

        # ------------------------------------------------------------
        # DEM "limpio" para mostrar (NaN fuera de la máscara real) y
        # con huecos pequeños DENTRO de la máscara rellenos por vecino
        # más cercano (sin triangulación -> sin facetas)
        # ------------------------------------------------------------
        dem_display = np.where(valid, dem_raw, np.nan)
        inner_holes = valid & ~np.isfinite(dem_raw)
        if inner_holes.any():
            _, (iy, ix) = distance_transform_edt(
                ~(valid & np.isfinite(dem_raw)), return_indices=True
            )
            dem_display[inner_holes] = dem_raw[iy[inner_holes], ix[inner_holes]]

        # ------------------------------------------------------------
        # DEM SOLO para interpolar el perfil (sin NaN en ningún sitio):
        # fuera de la máscara también se rellena por vecino más cercano,
        # solo como truco numérico para que map_coordinates(order=1) no
        # "envenene" a NaN puntos válidos cercanos al borde.
        # ------------------------------------------------------------
        if not np.all(valid):
            _, (iy_all, ix_all) = distance_transform_edt(~valid, return_indices=True)
            dem_for_interp = np.where(valid, dem_display, dem_display[iy_all, ix_all])
        else:
            dem_for_interp = dem_display

        # ------------------------------------------------------------
        # Transecto
        # ------------------------------------------------------------
        n = int(np.hypot(end[0] - start[0], end[1] - start[1]))
        rows = np.linspace(start[0], end[0], n)
        cols = np.linspace(start[1], end[1], n)

        profile = map_coordinates(dem_for_interp, [rows, cols], order=1, mode="nearest")

        mask_profile = (
            map_coordinates(valid.astype(float), [rows, cols], order=1, mode="constant", cval=0)
            >= 0.5
        )

        distance = np.linspace(0, n * pixel_size, n)

        profile_smooth = gaussian_filter1d(profile, sigma=smooth_sigma)
        profile_smooth = np.where(mask_profile, profile_smooth, np.nan)

        # Puntos donde el transecto entra/sale de la zona válida (para
        # marcar visualmente el límite de la franja intermareal)
        boundary_idx = np.where(np.diff(mask_profile.astype(int)) != 0)[0]
        boundary_distances = distance[boundary_idx]

        # ------------------------------------------------------------
        # Figura 1: perfil
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(distance, profile_smooth, color="#2b6cb0", linewidth=2.5)

        finite_profile = profile_smooth[np.isfinite(profile_smooth)]
        baseline = np.nanmin(finite_profile) if finite_profile.size > 0 else 0.0
        ax.fill_between(distance, profile_smooth, baseline, color="#2b6cb0", alpha=0.18)

        for bd in boundary_distances:
            ax.axvline(bd, color="darkorange", linestyle=":", linewidth=1.2, alpha=0.7)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, distance[-1])
        ax.set_xlabel("Distance (m)", fontsize=11)
        ax.set_ylabel("Elevation (m)", fontsize=11)
        ax.set_title("Bathymetric transect", fontsize=17, fontweight="bold")

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------
        # Figura 2: transecto sobre el DEM (solo zona real, resto blanco;
        # se recupera el interpolation="bicubic" de la versión original)
        # ------------------------------------------------------------
        cmap = plt.cm.gist_earth.copy()
        cmap.set_bad("white")

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(dem_display, cmap=cmap, interpolation="bicubic")

        ax.plot([start[1], end[1]], [start[0], end[0]], color="red", linewidth=2)
        ax.scatter(
            [start[1], end[1]], [start[0], end[0]],
            c=["lime", "red"], s=60, edgecolors="black", zorder=3,
        )

        ax.set_title("Transect location", fontsize=15, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()    
    @staticmethod
    def _nan_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
        """
        Suavizado gaussiano que ignora los NaN (normalized convolution):
        no propaga los huecos ni contamina el borde con ceros.
        """
        import numpy as np
        from scipy import ndimage

        finite = np.isfinite(arr)
        weight = finite.astype(np.float64)
        filled = np.where(finite, arr, 0.0).astype(np.float64)

        smooth_val = ndimage.gaussian_filter(filled, sigma, mode="nearest")
        smooth_wgt = ndimage.gaussian_filter(weight, sigma, mode="nearest")

        out = np.where(smooth_wgt > 1e-6, smooth_val / np.where(smooth_wgt > 1e-6, smooth_wgt, 1.0), np.nan)
        return out

    @staticmethod
    def plot_bathymetry_3d(
        elevation: np.ndarray,
        mask: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        min_confidence: float = 0.0,
        pixel_size: float = 10.0,
        vertical_exaggeration: float = 5.0,
        downsample: int = 2,
        fill_region: bool = True,
        clip_to_region: bool = False,
        close_iter: int = 2,
        region_dilate: int = 2,
        despike_size: int = 5,
        smooth_sigma: float = 1.5,
    ):
        """
        Interactive 3D bathymetry using Plotly.

        La zona batimétrica realmente reconstruida (observación agua+tierra)
        suele ser dispersa/moteada, por lo que dibujarla tal cual produce una
        superficie deshilachada con cortinas verticales. Por defecto
        (``fill_region=True``) se cierra la máscara en una región continua y se
        interpola la elevación sobre ella, obteniendo una superficie limpia
        sin inventar batimetría lejos del dato real.

        Parameters
        ----------
        elevation
            DEM batimétrico (NaN fuera de la zona reconstruida).
        mask
            Máscara booleana de la zona intermareal (p.ej. ``intertidal_mask``
            o ``result.mask``). Acota la región representada; no se extrapola
            fuera de ella.
        confidence
            Raster de confianza opcional. Los píxeles con
            ``confidence <= min_confidence`` no se usan como dato.
        min_confidence
            Umbral de confianza por debajo del cual un píxel se descarta.
        pixel_size
            Pixel size (m).
        vertical_exaggeration
            Exageración vertical. Se aplica al ASPECTO visual, no a los datos:
            el eje Z sigue mostrando metros reales.
        downsample
            Spatial decimation for faster rendering.
        fill_region
            Si True, rellena/suaviza para una superficie continua sin paredes.
            Si False, dibuja solo el dato real (moteado, con cortinas).
        clip_to_region
            Solo con ``fill_region``. Si True recorta la superficie a la zona
            reconstruida (honesto: sin relieve extrapolado, pero reaparecen
            paredes en el borde). Si False (por defecto) muestra una superficie
            continua en el entorno de la ría.
        close_iter
            Iteraciones de cierre morfológico para consolidar la región
            (mayor = puentea huecos más grandes, pero extrapola más).
        region_dilate
            Dilatación (px) de la región a representar. Da cuerpo a la banda y
            elimina las paredes/cortinas que Plotly dibuja en los bordes NaN.
        despike_size
            Tamaño del filtro de mediana para quitar picos/terrazas del método
            waterline (0 lo desactiva).
        smooth_sigma
            Sigma del suavizado gaussiano posterior (0 lo desactiva).
        """

        import numpy as np
        import plotly.graph_objects as go

        from scipy import ndimage
        from scipy.interpolate import griddata

        dem = elevation.astype(float).copy()

        # ------------------------------------------------------------
        # Puntos con dato real (elevación reconstruida)
        # ------------------------------------------------------------
        data_valid = np.isfinite(dem)
        if confidence is not None:
            conf = confidence.astype(float)
            data_valid &= np.isfinite(conf) & (conf > min_confidence)

        if not data_valid.any():
            raise ValueError(
                "No hay píxeles con dato para representar. Revisa 'confidence'/"
                "'elevation' (¿coinciden las dimensiones?)."
            )

        # ------------------------------------------------------------
        # Región de referencia (la zona intermareal) para acotar/recortar
        # ------------------------------------------------------------
        if mask is not None:
            region = ndimage.binary_fill_holes(mask.astype(bool))
        else:
            region = ndimage.binary_closing(
                data_valid, iterations=max(int(close_iter), 1)
            )
            region = ndimage.binary_fill_holes(region)

        if fill_region:
            # Superficie continua. Clave: Plotly dibuja una pared vertical en
            # CADA borde NaN, y una banda intermareal larga y sinuosa tiene un
            # perímetro enorme -> cortinas por todas partes. Por eso se
            # interpola una superficie SIN NaN (lineal + vecino más cercano),
            # se despica (mediana) y se suaviza. El recorte al bounding box de
            # la región (con margen) evita mostrar toda la escena.
            yy, xx = np.indices(dem.shape)
            pts = np.column_stack([yy[data_valid], xx[data_valid]])
            vals = dem[data_valid]

            grid = griddata(pts, vals, (yy, xx), method="linear")
            gap = ~np.isfinite(grid)
            if gap.any():
                nearest = griddata(pts, vals, (yy, xx), method="nearest")
                grid[gap] = nearest[gap]

            if despike_size and despike_size > 1:
                grid = ndimage.median_filter(grid, size=int(despike_size))
            if smooth_sigma and smooth_sigma > 0:
                grid = ndimage.gaussian_filter(grid, smooth_sigma)

            if clip_to_region:
                # Modo honesto: solo la zona reconstruida (reaparecen paredes
                # en el borde, pero no se muestra relieve extrapolado).
                keep = region
                if region_dilate and region_dilate > 0:
                    keep = ndimage.binary_dilation(keep, iterations=int(region_dilate))
                dem_show = np.where(keep, grid, np.nan)
            else:
                dem_show = grid
        else:
            # Solo el dato real (moteado, con paredes) — sin interpolar.
            keep = data_valid if mask is None else (data_valid & mask.astype(bool))
            dem_show = np.where(keep, dem, np.nan)

        if not np.isfinite(dem_show).any():
            raise ValueError("La región a representar quedó vacía tras el recorte.")

        # ------------------------------------------------------------
        # Recortar al bounding box de la región (con un margen), no a toda la
        # escena. Así la superficie continua no arrastra la imagen completa.
        # ------------------------------------------------------------
        rows = np.any(region, axis=1)
        cols = np.any(region, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = int(max(region_dilate, 0)) + 4
        H, W = dem_show.shape
        r0 = max(r0 - pad, 0); r1 = min(r1 + pad, H - 1)
        c0 = max(c0 - pad, 0); c1 = min(c1 + pad, W - 1)
        dem_show = dem_show[r0:r1 + 1, c0:c1 + 1]

        # ------------------------------------------------------------
        # Reducir resolución
        # ------------------------------------------------------------
        dem_ds = dem_show[::downsample, ::downsample]

        ny, nx = dem_ds.shape

        x = np.arange(nx) * pixel_size * downsample
        y = np.arange(ny) * pixel_size * downsample

        X, Y = np.meshgrid(x, y)

        # Elevación REAL en el eje Z (la exageración va en el aspecto visual).
        Z = dem_ds

        # ------------------------------------------------------------
        # Rango de color robusto sobre la zona real (percentiles 2-98)
        # ------------------------------------------------------------
        finite_vals = dem_ds[np.isfinite(dem_ds)]
        if finite_vals.size > 0:
            cmin, cmax = np.nanpercentile(finite_vals, [2, 98])
            if cmax <= cmin:
                cmin, cmax = float(finite_vals.min()), float(finite_vals.max())
        else:
            cmin, cmax = 0.0, 1.0

        # ------------------------------------------------------------
        # Exageración vertical vía aspecto (el eje Z conserva metros reales).
        # z_aspect controla cuánto se estira visualmente el relieve.
        # ------------------------------------------------------------
        z_aspect = float(np.clip(0.05 * vertical_exaggeration, 0.1, 0.6))

        fig = go.Figure()

        fig.add_trace(

            go.Surface(

                x=X,
                y=Y,
                z=Z,

                surfacecolor=dem_ds,

                colorscale="Earth",

                cmin=cmin,
                cmax=cmax,

                colorbar=dict(
                    title="Elevation (m)"
                ),

                lighting=dict(

                    ambient=0.55,
                    diffuse=0.85,
                    fresnel=0.10,
                    roughness=0.50,
                    specular=0.15,

                ),

                lightposition=dict(

                    x=1000,
                    y=-1000,
                    z=1500,

                ),

                hidesurface=False,

                connectgaps=False,

            )

        )

        fig.update_layout(

            title="Interactive 3D Bathymetry",

            width=1000,
            height=800,

            scene=dict(

                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Elevation (m)",

                # Las filas de imagen crecen hacia abajo; invertir el eje Y
                # para que la orientación coincida con los mapas 2D.
                yaxis=dict(autorange="reversed"),

                zaxis=dict(nticks=6),

                aspectmode="manual",

                aspectratio=dict(

                    x=1,
                    y=ny / nx if nx else 1,
                    z=z_aspect,

                ),

                camera=dict(

                    eye=dict(

                        x=1.8,
                        y=-1.8,
                        z=0.9,

                    )

                ),

            ),

            margin=dict(
                l=0,
                r=0,
                b=0,
                t=40,
            ),

        )

        fig.show()
            
    @staticmethod
    def plot_tide_distribution(
        reference_heights,
        selected_heights,
        model_name: str = "GOT4.10",
        title: str = None,
    ):
        """
        Visualiza la distribución vertical de las mareas para un único modelo.

        Parameters
        ----------
        reference_heights : array-like
            Serie completa de alturas de marea utilizada como referencia.
        selected_heights : array-like
            Alturas de marea correspondientes únicamente a las fechas
            seleccionadas.
        model_name : str, optional
            Nombre del modelo de marea.
        title : str, optional
            Título del gráfico.
        """

        reference_heights = np.asarray(reference_heights)
        selected_heights = np.asarray(selected_heights)

        # Normalización respecto a la media de la distribución completa
        mean_reference = reference_heights.mean()

        reference_norm = reference_heights - mean_reference
        selected_norm = selected_heights - mean_reference

        # Límites del gráfico
        ref_min = reference_norm.min()
        ref_max = reference_norm.max()

        margin = 0.05 * (ref_max - ref_min)

        fig, ax = plt.subplots(figsize=(5, 8))

        # Scatter vertical con ligero jitter horizontal
        x = np.random.normal(0, 0.02, len(selected_norm))

        ax.scatter(
            x,
            selected_norm,
            color="steelblue",
            alpha=0.6,
            s=25,
            edgecolors="black",
            linewidth=0.5,
        )

        # Líneas de referencia
        ax.axhline(
            ref_max,
            color="darkgreen",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )

        ax.axhline(
            ref_min,
            color="darkred",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
        )

        # Etiquetas
        ax.text(
            0.22,
            ref_max,
            f"{ref_max:+.2f} m",
            color="darkgreen",
            fontsize=10,
            fontweight="bold",
            va="center",
        )

        ax.text(
            0.22,
            ref_min,
            f"{ref_min:+.2f} m",
            color="darkred",
            fontsize=10,
            fontweight="bold",
            va="center",
        )

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(ref_min - margin, ref_max + margin)

        ax.set_xticks([])
        ax.set_ylabel("Normalized tide height (m)", fontsize=11)

        ax.set_title(
            title
            or f"{model_name}\nSelected dates (n={len(selected_norm)})",
            fontsize=12,
            fontweight="bold",
        )

        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_tide_time_series(
        dates_reference,
        heights,
        title: str = "Reference time series — GOT4.10",
    ):
        # Convertir fechas a datetime si no lo son
        dates_reference = pd.to_datetime(dates_reference)

        # Centrar la serie respecto a su media
        heights = np.asarray(heights)
        heights = heights - heights.mean()

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.scatter(
            dates_reference,
            heights,
            color="blue",
            alpha=0.6,
            s=20,
            edgecolors="black",
            linewidth=0.5,
        )

        # Percentiles
        p25, p50, p75 = np.percentile(heights, [25, 50, 75])

        ax.axhline(
            p50,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Mediana",
        )

        ax.axhline(
            p25,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label="P25 / P75",
        )

        ax.axhline(
            p75,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )

        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Altura de marea (m)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.grid(True, alpha=0.3)
        ax.legend()

        # Formato del eje X: locator automático según el rango temporal
        # (meses si es ~1 año, años si son varios) -> legible en cualquier span.
        locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_tide_model_comparison(
        df_comparison,
        site: str = "Sitio",
        figsize: tuple = (14, 6)
    ):
        """
        Compara modelos de marea vs observaciones.
        
        Parameters
        ----------
        df_comparison : DataFrame
            DataFrame con columnas:
            - datetime
            - observed_height_m
            - got410_height_m
            - cmems_height_m (opcional)
        site : str, optional
            Nombre del sitio (default: "Sitio")
        figsize : tuple, optional
            Tamaño de figura (default: (14, 6))
            
        Examples
        --------
        >>> Visualizer.plot_tide_model_comparison(
        ...     df_comparison,
        ...     site="Gijón"
        ... )
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Observado
        if "observed_height_m" in df_comparison.columns:
            ax.plot(
                df_comparison["datetime"],
                df_comparison["observed_height_m"],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=4,
                label="Mareógrafo (observado)",
                color="black",
            )
        
        # GOT4.10
        if "got410_height_m" in df_comparison.columns:
            ax.plot(
                df_comparison["datetime"],
                df_comparison["got410_height_m"],
                marker="s",
                linestyle="--",
                linewidth=1.5,
                markersize=3,
                label="GOT4.10c (modelo)",
                color="steelblue",
            )
        
        # CMEMS
        if "cmems_height_m" in df_comparison.columns:
            ax.plot(
                df_comparison["datetime"],
                df_comparison["cmems_height_m"],
                marker="^",
                linestyle=":",
                linewidth=1.5,
                markersize=3,
                label="CMEMS (modelo)",
                color="orange",
            )
        
        ax.set_title(f"Comparación de Modelos de Marea — {site}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fecha", fontsize=10)
        ax.set_ylabel("Altura de marea (m)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        
        plt.tight_layout()
        plt.show()
