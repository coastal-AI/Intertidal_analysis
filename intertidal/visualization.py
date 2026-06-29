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
import rasterio
from rasterio.mask import geometry_mask
import geopandas as gpd
import contextily as ctx


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
    
    # Paleta de colores SCL oficial ESA
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
        color_list = [Visualizer.SCL_COLORS.get(c, ("?", "#aaaaaa"))[1] for c in range(12)]
        cmap = mcolors.ListedColormap(color_list)
        cmap.set_bad("white")
        norm = mcolors.BoundaryNorm(boundaries=list(range(13)), ncolors=12)
        
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
        color_list = [Visualizer.SCL_COLORS.get(c, ("?", "#aaaaaa"))[1] for c in range(12)]
        cmap = mcolors.ListedColormap(color_list)
        cmap.set_bad("white")
        norm = mcolors.BoundaryNorm(boundaries=list(range(13)), ncolors=12)
        
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
        transform,
        crs: str = "EPSG:4326",
        figsize: tuple = (10, 8),
        add_basemap: bool = True
    ):
        """
        Visualiza mapa de water frequency.
        
        Parameters
        ----------
        water_freq : ndarray
            Array float (H, W) con valores [0, 1]
        transform : Affine
            Transformada afín del raster
        crs : str, optional
            Sistema de referencia (default: "EPSG:4326")
        figsize : tuple, optional
            Tamaño de figura (default: (10, 8))
        add_basemap : bool, optional
            Añadir mapa base de OpenStreetMap (default: True)
            
        Examples
        --------
        >>> Visualizer.plot_water_frequency(
        ...     water_freq,
        ...     transform,
        ...     add_basemap=True
        ... )
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Visualizar water frequency
        im = ax.imshow(
            water_freq,
            cmap="RdYlBu",
            vmin=0,
            vmax=1,
            extent=[
                transform[2],
                transform[2] + transform[0] * water_freq.shape[1],
                transform[5] + transform[4] * water_freq.shape[0],
                transform[5],
            ],
            interpolation="nearest",
        )
        
        # Barra de color
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Water Frequency [0-1]", rotation=270, labelpad=20)
        
        # Basemap
        if add_basemap:
            try:
                ctx.add_basemap(
                    ax,
                    crs=crs,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    alpha=0.5,
                )
            except Exception as e:
                print(f"No se pudo añadir basemap: {e}")
        
        ax.set_title("Water Frequency Map", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_rgb_grid(
        dates: list[str],
        rgb_dir: str,
        ncols: int = 4,
        figsize: tuple = (16, 12)
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
    def plot_tide_timeseries(
        df_tides,
        site: str = "Sitio",
        figsize: tuple = (14, 5)
    ):
        """
        Visualiza serie temporal de marea.
        
        Parameters
        ----------
        df_tides : DataFrame
            DataFrame con columnas 'date' y 'tide_height_m'
        site : str, optional
            Nombre del sitio (default: "Sitio")
        figsize : tuple, optional
            Tamaño de figura (default: (14, 5))
            
        Examples
        --------
        >>> Visualizer.plot_tide_timeseries(df_tides, site="Gijón")
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot principal
        ax.plot(
            df_tides["date"],
            df_tides["tide_height_m"],
            marker="o",
            linestyle="-",
            markersize=4,
            linewidth=1,
            color="steelblue",
        )
        
        # Línea de referencia media
        mean_tide = df_tides["tide_height_m"].mean()
        ax.axhline(
            mean_tide,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Media: {mean_tide:.2f} m",
        )
        
        ax.set_title(f"Serie Temporal de Marea — {site}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fecha", fontsize=10)
        ax.set_ylabel("Altura de marea (m)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
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
