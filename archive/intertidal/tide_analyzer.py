"""
tide_analyzer.py — Análisis y validación de modelos de marea
============================================================

Módulo para procesamiento de datos de marea, comparación de modelos
predictivos (GOT4.10, CMEMS) con observaciones de mareógrafos.
"""

import numpy as np
import pandas as pd


class TideAnalyzer:
    """
    Analizador de datos y modelos de marea.
    
    Funcionalidades:
    - Carga de datos de mareógrafos (PORTUS, REDMAR)
    - Predicción de mareas usando modelos globales
    - Cálculo de anomalías (residuales)
    - Comparación de modelos vs observaciones
    - Métricas de error (MAE, RMSE)
    
    Attributes
    ----------
    tide_model : str
        Nombre del modelo de marea (e.g., 'GOT4.10c')
    tide_dir : str
        Directorio con archivos del modelo
    
    Examples
    --------
    >>> from intertidal import TideAnalyzer
    >>> 
    >>> analyzer = TideAnalyzer()
    >>> 
    >>> # Cargar datos de mareógrafo
    >>> gauge_data = analyzer.load_gauge_data(
    ...     "mareografo_gijon.xlsx",
    ...     date_col="fecha",
    ...     height_col="nivel_m"
    ... )
    >>> 
    >>> # Calcular métricas de error
    >>> metrics = analyzer.calculate_metrics(df_comparison)
    >>> analyzer.print_metrics(metrics)
    """
    
    def __init__(self, tide_model: str = "GOT4.10c", tide_dir: str = None):
        """
        Inicializa el analizador de mareas.
        
        Parameters
        ----------
        tide_model : str, optional
            Nombre del modelo de marea (default: "GOT4.10c")
        tide_dir : str, optional
            Directorio con archivos del modelo
        """
        self.tide_model = tide_model
        self.tide_dir = tide_dir
    
    def load_gauge_data(
        self,
        filepath: str,
        date_col: str = "fecha",
        height_col: str = "nivel_m",
        skiprows: int = 0
    ) -> pd.DataFrame:
        """
        Carga datos de mareógrafo desde Excel o CSV.
        
        Parameters
        ----------
        filepath : str
            Ruta al archivo de datos
        date_col : str, optional
            Nombre de la columna de fechas (default: "fecha")
        height_col : str, optional
            Nombre de la columna de altura (default: "nivel_m")
        skiprows : int, optional
            Filas a saltar al leer (default: 0)
            
        Returns
        -------
        DataFrame
            DataFrame con columnas normalizadas: datetime, height_m
            
        Examples
        --------
        >>> gauge = analyzer.load_gauge_data(
        ...     "mareografo_gijon.xlsx",
        ...     date_col="FECHA",
        ...     height_col="ALTURA_M"
        ... )
        """
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, skiprows=skiprows)
        else:
            df = pd.read_csv(filepath, skiprows=skiprows)
        
        # Normalizar nombres de columnas
        df = df.rename(columns={date_col: "datetime", height_col: "height_m"})
        
        # Convertir a datetime si no lo es
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        return df
    
    def predict_tides(
        self,
        dates: list[str],
        lon: float,
        lat: float
    ) -> pd.DataFrame:
        """
        Predice alturas de marea usando pyTMD.
        
        Parameters
        ----------
        dates : list[str]
            Lista de fechas 'YYYY-MM-DD' o timestamps
        lon : float
            Longitud del punto
        lat : float
            Latitud del punto
            
        Returns
        -------
        DataFrame
            DataFrame con columnas: datetime, predicted_height_m
            
        Notes
        -----
        Esta función requiere pyTMD instalado y configurado.
        Por ahora retorna placeholder.
        
        Examples
        --------
        >>> predictions = analyzer.predict_tides(
        ...     ["2024-01-01", "2024-01-02"],
        ...     lon=-5.66,
        ...     lat=43.54
        ... )
        """
        # TODO: Implementar predicción con pyTMD
        raise NotImplementedError(
            "predict_tides requiere pyTMD y tidemodel.py. "
            "Usa las funciones de tidemodel.py directamente hasta completar migración."
        )
    
    def calculate_anomalies(
        self,
        observed: pd.DataFrame,
        predicted: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula anomalías (residuales) entre observado y predicho.
        
        Parameters
        ----------
        observed : DataFrame
            Datos observados con columnas: datetime, height_m
        predicted : DataFrame
            Datos predichos con columnas: datetime, predicted_height_m
            
        Returns
        -------
        DataFrame
            DataFrame con columnas: datetime, observed, predicted, anomaly
            
        Examples
        --------
        >>> anomalies = analyzer.calculate_anomalies(gauge, predictions)
        >>> print(f"Anomalía media: {anomalies['anomaly'].mean():.2f} m")
        """
        # Merge por datetime
        merged = pd.merge(
            observed,
            predicted,
            on="datetime",
            how="inner",
            suffixes=("_obs", "_pred")
        )
        
        # Calcular residual
        merged["anomaly"] = merged["height_m"] - merged["predicted_height_m"]
        
        return merged
    
    @staticmethod
    def calculate_metrics(
        df_comparison: pd.DataFrame,
        cmems_available: bool = False,
        obs_available: bool = False
    ) -> dict:
        """
        Calcula métricas de error para modelos de marea.
        
        Compara predicciones de modelos (GOT4.10, CMEMS) con
        observaciones de mareógrafo y calcula MAE, RMSE.
        
        Parameters
        ----------
        df_comparison : DataFrame
            DataFrame con columnas:
            - got410_height_m: predicción GOT4.10
            - observed_height_m: observación
            - cmems_height_m: predicción CMEMS (opcional)
        cmems_available : bool, optional
            Si hay columna CMEMS (default: False)
        obs_available : bool, optional
            Si hay observaciones reales (default: False)
            
        Returns
        -------
        dict
            Diccionario con estructura:
            {
                'got410': {
                    'mae': float,
                    'rmse': float,
                    'min_error': float,
                    'max_error': float,
                    'n_samples': int
                },
                'cmems': {...} o None,
                'has_observations': bool
            }
            
        Examples
        --------
        >>> metrics = TideAnalyzer.calculate_metrics(
        ...     df_comparison,
        ...     cmems_available=True,
        ...     obs_available=True
        ... )
        >>> print(f"GOT4.10 MAE: {metrics['got410']['mae']:.3f} m")
        """
        metrics = {
            'got410': None,
            'cmems': None,
            'has_observations': obs_available
        }
        
        if not obs_available or not df_comparison['observed_height_m'].notna().any():
            return metrics
        
        # Filtrar solo filas con observaciones válidas
        df_valid = df_comparison.dropna(subset=['observed_height_m'])
        
        # GOT4.10 vs Observado
        got_errors = np.abs(df_valid['got410_height_m'] - df_valid['observed_height_m'])
        got_mae = got_errors.mean()
        got_rmse = np.sqrt(
            ((df_valid['got410_height_m'] - df_valid['observed_height_m'])**2).mean()
        )
        
        metrics['got410'] = {
            'mae': got_mae,
            'rmse': got_rmse,
            'min_error': got_errors.min(),
            'max_error': got_errors.max(),
            'n_samples': len(df_valid)
        }
        
        # CMEMS vs Observado (si disponible)
        if cmems_available:
            df_valid_cmems = df_valid.dropna(subset=['cmems_height_m'])
            
            if len(df_valid_cmems) > 0:
                cmems_errors = np.abs(
                    df_valid_cmems['cmems_height_m'] - df_valid_cmems['observed_height_m']
                )
                cmems_mae = cmems_errors.mean()
                cmems_rmse = np.sqrt(
                    ((df_valid_cmems['cmems_height_m'] - df_valid_cmems['observed_height_m'])**2).mean()
                )
                
                metrics['cmems'] = {
                    'mae': cmems_mae,
                    'rmse': cmems_rmse,
                    'min_error': cmems_errors.min(),
                    'max_error': cmems_errors.max(),
                    'n_samples': len(df_valid_cmems)
                }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: dict):
        """
        Imprime métricas de error de forma formateada.
        
        Parameters
        ----------
        metrics : dict
            Diccionario retornado por calculate_metrics()
            
        Examples
        --------
        >>> metrics = TideAnalyzer.calculate_metrics(df_comparison, obs_available=True)
        >>> TideAnalyzer.print_metrics(metrics)
        """
        if not metrics['has_observations']:
            print("⚠️  No se pueden calcular métricas sin datos del mareógrafo")
            return
        
        print("=" * 70)
        print("📊 MÉTRICAS DE ERROR — Modelos de Marea vs Mareógrafo")
        print("=" * 70)
        
        if metrics['got410'] is not None:
            print("\n🌊 GOT4.10c (Modelo Global Ocean Tide)")
            print("-" * 70)
            print(f"   MAE  (Error Absoluto Medio):     {metrics['got410']['mae']:.4f} m")
            print(f"   RMSE (Raíz del Error Cuadrático): {metrics['got410']['rmse']:.4f} m")
            print(f"   Error mínimo:                     {metrics['got410']['min_error']:.4f} m")
            print(f"   Error máximo:                     {metrics['got410']['max_error']:.4f} m")
            print(f"   Muestras comparadas:              {metrics['got410']['n_samples']}")
        
        if metrics['cmems'] is not None:
            print("\n🌊 CMEMS (Copernicus Marine Service)")
            print("-" * 70)
            print(f"   MAE  (Error Absoluto Medio):     {metrics['cmems']['mae']:.4f} m")
            print(f"   RMSE (Raíz del Error Cuadrático): {metrics['cmems']['rmse']:.4f} m")
            print(f"   Error mínimo:                     {metrics['cmems']['min_error']:.4f} m")
            print(f"   Error máximo:                     {metrics['cmems']['max_error']:.4f} m")
            print(f"   Muestras comparadas:              {metrics['cmems']['n_samples']}")
        
        print("=" * 70)
    
    def compare_models(
        self,
        df_comparison: pd.DataFrame,
        models: list[str] = None
    ) -> pd.DataFrame:
        """
        Compara múltiples modelos de marea.
        
        Parameters
        ----------
        df_comparison : DataFrame
            DataFrame con predicciones de varios modelos
        models : list[str], optional
            Nombres de columnas a comparar
            (default: ['got410_height_m', 'cmems_height_m'])
            
        Returns
        -------
        DataFrame
            DataFrame con estadísticas comparativas
            
        Notes
        -----
        Placeholder - requiere implementación completa.
        """
        # TODO: Implementar comparación detallada
        raise NotImplementedError(
            "compare_models requiere implementación completa."
        )
