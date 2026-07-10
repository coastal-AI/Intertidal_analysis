"""
Métricas de calidad para distribuciones de mareas.

Este módulo proporciona funciones para evaluar la idoneidad de distribuciones
de alturas de marea para modelado batimétrico intermareal usando el método waterline.

El método waterline reconstruye topografía intermareal a partir de múltiples líneas
de costa observadas en diferentes niveles de marea. La calidad del modelo batimétrico
resultante depende críticamente de la distribución de alturas de marea en las
observaciones disponibles.

Métricas implementadas:
- Tidal Range Coverage (TRC): Cobertura del rango mareal
- Vertical Sampling Resolution (VSR): Resolución vertical del muestreo
- Uniformidad de la distribución (KS test, entropía)
- Representativeness Index: Comparación con distribución ideal
- Gap Analysis: Análisis de huecos en la cobertura

Referencias:
-----------
- Bell et al. (2016): "Shallow water bathymetry derived from an analysis of 
  X-band marine radar images of waves" - RMSE y métricas de precisión
- Heygster et al. (2010): "Topographic mapping of the German tidal flats analyzing
  SAR images with the waterline method" - Análisis de cobertura mareal
- Mason et al. (1995): "Construction of an inter-tidal digital elevation model by
  the 'water-line' method" - Metodología waterline clásica
- Kolmogorov-Smirnov test: Massey (1951)
- Shannon entropy: Shannon (1948)
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings


def calcular_cobertura_rango_mareal(
    valores_validos: np.ndarray,
    valores_totales: np.ndarray
) -> Dict[str, float]:
    """
    Calcula la cobertura del rango mareal de un subconjunto respecto al total.
    
    Esta métrica evalúa qué porción del rango completo de mareas está representada
    en las fechas seleccionadas. Un valor cercano a 1.0 indica que las muestras
    cubren todo el espectro de alturas de marea posibles.
    
    Parámetros
    ----------
    valores_validos : array-like
        Alturas de marea en las fechas seleccionadas (m)
    valores_totales : array-like
        Alturas de marea de la distribución completa de referencia (m)
    
    Retorna
    -------
    dict
        rango_valido : float
            Amplitud del subconjunto (m)
        rango_total : float
            Amplitud de la distribución completa (m)
        cobertura : float
            Fracción del rango total cubierto (0-1)
        cobertura_pct : float
            Porcentaje de cobertura
        min_valido, max_valido : float
            Valores extremos del subconjunto
        min_total, max_total : float
            Valores extremos de la distribución completa
    """
    valores_validos = np.asarray(valores_validos)
    valores_totales = np.asarray(valores_totales)
    
    min_valido, max_valido = valores_validos.min(), valores_validos.max()
    min_total, max_total = valores_totales.min(), valores_totales.max()
    
    rango_valido = max_valido - min_valido
    rango_total = max_total - min_total
    
    cobertura = rango_valido / rango_total if rango_total > 0 else 0.0
    
    return {
        'rango_valido': float(rango_valido),
        'rango_total': float(rango_total),
        'cobertura': float(cobertura),
        'cobertura_pct': float(cobertura * 100),
        'min_valido': float(min_valido),
        'max_valido': float(max_valido),
        'min_total': float(min_total),
        'max_total': float(max_total),
    }


def calcular_uniformidad_ks(
    valores: np.ndarray,
    rango_referencia: Optional[Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Test de Kolmogorov-Smirnov para evaluar uniformidad de la distribución.
    
    Compara la distribución empírica de las muestras con una distribución
    uniforme ideal en el rango dado. Un p-value alto indica que la distribución
    es estadísticamente uniforme.
    
    Referencias
    -----------
    Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit.
    Journal of the American statistical Association, 46(253), 68-78.
    
    Parámetros
    ----------
    valores : array-like
        Alturas de marea a evaluar
    rango_referencia : tuple (min, max), optional
        Rango de la distribución uniforme de referencia.
        Si None, usa (min(valores), max(valores))
    
    Retorna
    -------
    dict
        ks_statistic : float
            Estadístico KS (0-1, menor es más uniforme)
        ks_pvalue : float
            P-value del test (>0.05 sugiere uniformidad)
        uniforme : bool
            True si p-value > 0.05 (no se puede rechazar uniformidad)
        interpretacion : str
            Interpretación cualitativa del resultado
    """
    valores = np.asarray(valores)
    
    if rango_referencia is None:
        rango_min, rango_max = valores.min(), valores.max()
    else:
        rango_min, rango_max = rango_referencia
    
    # Normalizar valores al rango [0, 1] para comparar con uniforme(0,1)
    valores_norm = (valores - rango_min) / (rango_max - rango_min)
    
    # Test KS contra distribución uniforme [0, 1]
    ks_stat, ks_pval = stats.kstest(valores_norm, 'uniform')
    
    # Interpretación
    if ks_pval > 0.1:
        interp = "Altamente uniforme"
    elif ks_pval > 0.05:
        interp = "Razonablemente uniforme"
    elif ks_pval > 0.01:
        interp = "Moderadamente no uniforme"
    else:
        interp = "Altamente no uniforme"
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'uniforme': bool(ks_pval > 0.05),
        'interpretacion': interp
    }


def calcular_entropia_shannon(
    valores: np.ndarray,
    n_bins: int = 20,
    rango_referencia: Optional[Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Calcula la entropía de Shannon normalizada de la distribución.
    
    La entropía de Shannon mide el contenido de información o uniformidad
    de una distribución. Se normaliza dividiendo por la entropía máxima
    posible (log(n_bins)), resultando en un valor entre 0 y 1.
    
    Valores cercanos a 1.0 indican distribución uniforme (máxima entropía).
    Valores cercanos a 0.0 indican concentración en pocos valores.
    
    Referencias
    -----------
    Shannon, C. E. (1948). A mathematical theory of communication.
    Bell system technical journal, 27(3), 379-423.
    
    Parámetros
    ----------
    valores : array-like
        Alturas de marea a evaluar
    n_bins : int, default=20
        Número de bins para discretizar la distribución
    rango_referencia : tuple (min, max), optional
        Rango para los bins. Si None, usa (min(valores), max(valores))
    
    Retorna
    -------
    dict
        entropia : float
            Entropía de Shannon (bits)
        entropia_max : float
            Entropía máxima posible = log2(n_bins)
        entropia_norm : float
            Entropía normalizada (0-1, mayor es más uniforme)
        n_bins_vacios : int
            Número de bins sin muestras
        n_bins_ocupados : int
            Número de bins con al menos una muestra
    """
    valores = np.asarray(valores)
    
    if rango_referencia is None:
        rango_min, rango_max = valores.min(), valores.max()
    else:
        rango_min, rango_max = rango_referencia
    
    # Crear histograma
    counts, _ = np.histogram(valores, bins=n_bins, range=(rango_min, rango_max))
    
    # Calcular probabilidades (ignorando bins vacíos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Eliminar bins vacíos
    
    # Entropía de Shannon en bits
    entropia = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
    entropia_max = np.log2(n_bins)
    entropia_norm = entropia / entropia_max if entropia_max > 0 else 0.0
    
    n_bins_ocupados = np.sum(counts > 0)
    n_bins_vacios = n_bins - n_bins_ocupados
    
    return {
        'entropia': float(entropia),
        'entropia_max': float(entropia_max),
        'entropia_norm': float(entropia_norm),
        'n_bins': int(n_bins),
        'n_bins_ocupados': int(n_bins_ocupados),
        'n_bins_vacios': int(n_bins_vacios),
    }


def calcular_indice_dispersion(valores: np.ndarray) -> Dict[str, float]:
    """
    Calcula el índice de dispersión espacial (VMR - Variance-to-Mean Ratio).
    
    El VMR compara la varianza con la media de las distancias entre vecinos.
    Interpretación:
    - VMR ≈ 1: distribución aleatoria (Poisson)
    - VMR < 1: distribución regular/uniforme (valores más espaciados)
    - VMR > 1: distribución agrupada/clustered
    
    Para modelado batimétrico, valores VMR < 1 son deseables (regularidad).
    
    Referencias
    -----------
    Ludwig, J. A., & Reynolds, J. F. (1988). Statistical ecology: a primer
    in methods and computing (Vol. 1). John Wiley & Sons.
    
    Parámetros
    ----------
    valores : array-like
        Alturas de marea a evaluar
    
    Retorna
    -------
    dict
        vmr : float
            Variance-to-Mean Ratio
        patron : str
            "Regular" (VMR < 0.8), "Aleatorio" (0.8-1.2), "Agrupado" (>1.2)
        media_gaps : float
            Media de distancias entre vecinos
        varianza_gaps : float
            Varianza de distancias entre vecinos
    """
    valores = np.asarray(valores)
    valores_ordenados = np.sort(valores)
    gaps = np.diff(valores_ordenados)
    
    if len(gaps) == 0:
        return {
            'vmr': np.nan,
            'patron': 'Indefinido',
            'media_gaps': np.nan,
            'varianza_gaps': np.nan
        }
    
    media_gaps = np.mean(gaps)
    varianza_gaps = np.var(gaps, ddof=1)
    
    vmr = varianza_gaps / media_gaps if media_gaps > 0 else np.inf
    
    # Clasificación del patrón
    if vmr < 0.8:
        patron = "Regular (uniforme)"
    elif vmr <= 1.2:
        patron = "Aleatorio (Poisson)"
    else:
        patron = "Agrupado (clustered)"
    
    return {
        'vmr': float(vmr),
        'patron': patron,
        'media_gaps': float(media_gaps),
        'varianza_gaps': float(varianza_gaps),
    }


def calcular_estadisticos_gaps(valores: np.ndarray) -> Dict[str, float]:
    """
    Calcula estadísticos detallados de las distancias entre vecinos (gaps).
    
    Analiza la distribución de distancias entre muestras consecutivas cuando
    se ordenan por altura de marea. Gaps pequeños y uniformes son deseables
    para modelado batimétrico de alta resolución.
    
    Parámetros
    ----------
    valores : array-like
        Alturas de marea a evaluar
    
    Retorna
    -------
    dict
        n_muestras : int
            Número de muestras
        n_gaps : int
            Número de huecos (n_muestras - 1)
        media : float
            Distancia media entre vecinos (m)
        mediana : float
            Mediana de distancias (m)
        std : float
            Desviación estándar (m)
        cv : float
            Coeficiente de variación (std/media)
        min : float
            Distancia mínima (m)
        max : float
            Distancia máxima (m)
        p25, p75 : float
            Percentiles 25 y 75 (m)
        iqr : float
            Rango intercuartílico (p75 - p25)
        skewness : float
            Asimetría de la distribución de gaps
        kurtosis : float
            Curtosis de la distribución de gaps
    """
    valores = np.asarray(valores)
    valores_ordenados = np.sort(valores)
    gaps = np.diff(valores_ordenados)
    
    if len(gaps) == 0:
        return {k: np.nan for k in ['n_muestras', 'n_gaps', 'media', 'mediana',
                                     'std', 'cv', 'min', 'max', 'p25', 'p75',
                                     'iqr', 'skewness', 'kurtosis']}
    
    media = np.mean(gaps)
    std = np.std(gaps, ddof=1)
    p25 = np.percentile(gaps, 25)
    p75 = np.percentile(gaps, 75)
    
    return {
        'n_muestras': int(len(valores)),
        'n_gaps': int(len(gaps)),
        'media': float(media),
        'mediana': float(np.median(gaps)),
        'std': float(std),
        'cv': float(std / media if media > 0 else np.inf),
        'min': float(np.min(gaps)),
        'max': float(np.max(gaps)),
        'p25': float(p25),
        'p75': float(p75),
        'iqr': float(p75 - p25),
        'skewness': float(stats.skew(gaps)),
        'kurtosis': float(stats.kurtosis(gaps)),
    }


def calcular_vsr_waterline(valores: np.ndarray) -> Dict[str, float]:
    """
    Calcula la Resolución Vertical de Muestreo (VSR - Vertical Sampling Resolution).
    
    Métrica específica para el método waterline que cuantifica la resolución vertical
    efectiva del muestreo batimétrico. En el método waterline, cada observación
    corresponde a una línea de costa a un nivel de marea específico. La VSR mide
    la densidad de estos niveles en el espacio vertical.
    
    VSR = distancia media entre niveles mareales consecutivos (ordenados)
    
    Interpretación:
    - VSR baja (~0.1 m o menos): Alta resolución, ideal para topografía detallada
    - VSR media (0.1-0.3 m): Resolución moderada, adecuada para análisis generales
    - VSR alta (>0.3 m): Baja resolución, puede perder detalles topográficos
    
    La VSR depende directamente del número de observaciones y del rango mareal.
    Para un mismo rango mareal, más observaciones resultan en una VSR menor (mejor).
    
    Referencias
    -----------
    Basado en análisis de resolución vertical en:
    - Mason et al. (1995): Construction of an inter-tidal digital elevation model
    - Heygster et al. (2010): Topographic mapping of German tidal flats
    
    Parámetros
    ----------
    valores : array-like
        Alturas de marea en las observaciones (m)
    
    Retorna
    -------
    dict
        vsr : float
            Resolución vertical de muestreo (distancia media entre niveles, m)
        vsr_mediana : float
            Mediana de las distancias entre niveles (m)
        vsr_min : float
            Mínima distancia entre niveles consecutivos (m)
        vsr_max : float
            Máxima distancia entre niveles consecutivos (m)
        max_gap : float
            Mayor hueco en la cobertura vertical (m)
        max_gap_ubicacion : tuple (float, float)
            Límites del mayor hueco [nivel_inferior, nivel_superior] (m)
        n_niveles : int
            Número de niveles mareales únicos
        calidad_vsr : str
            Evaluación cualitativa: "Alta", "Media", "Baja"
    """
    valores = np.asarray(valores)
    valores_ordenados = np.sort(np.unique(valores))  # Eliminar duplicados
    
    if len(valores_ordenados) < 2:
        return {
            'vsr': np.nan,
            'vsr_mediana': np.nan,
            'vsr_min': np.nan,
            'vsr_max': np.nan,
            'max_gap': np.nan,
            'max_gap_ubicacion': (np.nan, np.nan),
            'n_niveles': len(valores_ordenados),
            'calidad_vsr': 'Indefinido'
        }
    
    gaps = np.diff(valores_ordenados)
    vsr = np.mean(gaps)
    vsr_mediana = np.median(gaps)
    max_gap = np.max(gaps)
    max_gap_idx = np.argmax(gaps)
    max_gap_ubicacion = (
        float(valores_ordenados[max_gap_idx]),
        float(valores_ordenados[max_gap_idx + 1])
    )
    
    # Evaluación cualitativa de la VSR
    if vsr <= 0.1:
        calidad = "Alta (excelente para topografía detallada)"
    elif vsr <= 0.2:
        calidad = "Buena (adecuada para la mayoría de análisis)"
    elif vsr <= 0.3:
        calidad = "Moderada (puede perder algunos detalles)"
    else:
        calidad = "Baja (resolución insuficiente para detalles finos)"
    
    return {
        'vsr': float(vsr),
        'vsr_mediana': float(vsr_mediana),
        'vsr_min': float(np.min(gaps)),
        'vsr_max': float(np.max(gaps)),
        'max_gap': float(max_gap),
        'max_gap_ubicacion': max_gap_ubicacion,
        'n_niveles': int(len(valores_ordenados)),
        'calidad_vsr': calidad,
    }


def calcular_representatividad(
    valores_validos: np.ndarray,
    valores_totales: np.ndarray,
    n_quantiles: int = 10
) -> Dict[str, float]:
    """
    Calcula el índice de representatividad de la distribución de muestreo.
    
    Evalúa qué tan bien la distribución empírica de las muestras representa
    la distribución teórica completa. Se divide el rango en quantiles y se
    verifica qué porcentaje de quantiles tienen al menos una muestra.
    
    Esta métrica es complementaria a la cobertura del rango, ya que evalúa
    la uniformidad de la cobertura en lugar de solo los extremos.
    
    Parámetros
    ----------
    valores_validos : array-like
        Alturas de marea en las fechas seleccionadas
    valores_totales : array-like
        Distribución completa de referencia
    n_quantiles : int, default=10
        Número de quantiles para dividir el rango (deciles por defecto)
    
    Retorna
    -------
    dict
        representatividad : float
            Fracción de quantiles con al menos una muestra (0-1)
        representatividad_pct : float
            Porcentaje de quantiles cubiertos
        quantiles_cubiertos : int
            Número de quantiles con muestras
        quantiles_totales : int
            Número total de quantiles
        quantiles_vacios : list of int
            Índices de los quantiles sin muestras
        distribucion_muestras : array
            Número de muestras en cada quantil
    """
    valores_validos = np.asarray(valores_validos)
    valores_totales = np.asarray(valores_totales)
    
    # Definir los límites de los quantiles basados en la distribución total
    quantile_edges = np.linspace(valores_totales.min(), valores_totales.max(), n_quantiles + 1)
    
    # Contar muestras en cada quantil
    counts, _ = np.histogram(valores_validos, bins=quantile_edges)
    
    quantiles_cubiertos = np.sum(counts > 0)
    quantiles_vacios = list(np.where(counts == 0)[0])
    representatividad = quantiles_cubiertos / n_quantiles
    
    return {
        'representatividad': float(representatividad),
        'representatividad_pct': float(representatividad * 100),
        'quantiles_cubiertos': int(quantiles_cubiertos),
        'quantiles_totales': int(n_quantiles),
        'quantiles_vacios': quantiles_vacios,
        'distribucion_muestras': counts.tolist(),
    }


def calcular_metricas_completas(
    valores_validos: np.ndarray,
    valores_totales: np.ndarray,
    n_bins: int = 20,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Calcula todas las métricas de calidad de distribución de mareas.
    
    Función de conveniencia que ejecuta todas las métricas disponibles
    y retorna un diccionario completo con los resultados.
    
    Parámetros
    ----------
    valores_validos : array-like
        Alturas de marea en las fechas seleccionadas
    valores_totales : array-like
        Alturas de marea de la distribución completa de referencia
    n_bins : int, default=20
        Número de bins para el cálculo de entropía
    verbose : bool, default=False
        Si True, imprime un resumen de las métricas
    
    Retorna
    -------
    dict
        Diccionario con claves:
        - 'cobertura': resultados de calcular_cobertura_rango_mareal()
        - 'uniformidad_ks': resultados de calcular_uniformidad_ks()
        - 'entropia': resultados de calcular_entropia_shannon()
        - 'dispersion': resultados de calcular_indice_dispersion()
        - 'gaps': resultados de calcular_estadisticos_gaps()
        - 'vsr': resultados de calcular_vsr_waterline()
        - 'representatividad': resultados de calcular_representatividad()
    """
    valores_validos = np.asarray(valores_validos)
    valores_totales = np.asarray(valores_totales)
    
    # Rango de referencia para uniformidad y entropía
    rango_ref = (valores_totales.min(), valores_totales.max())
    
    metricas = {
        'cobertura': calcular_cobertura_rango_mareal(valores_validos, valores_totales),
        'uniformidad_ks': calcular_uniformidad_ks(valores_validos, rango_ref),
        'entropia': calcular_entropia_shannon(valores_validos, n_bins, rango_ref),
        'dispersion': calcular_indice_dispersion(valores_validos),
        'gaps': calcular_estadisticos_gaps(valores_validos),
        'vsr': calcular_vsr_waterline(valores_validos),
        'representatividad': calcular_representatividad(valores_validos, valores_totales, n_quantiles=10),
    }
    
    if verbose:
        imprimir_metricas_completas(metricas)
    
    return metricas


def imprimir_metricas_completas(metricas: Dict[str, Dict], titulo: str = ""):
    """
    Imprime un resumen formateado de todas las métricas.
    
    Parámetros
    ----------
    metricas : dict
        Diccionario retornado por calcular_metricas_completas()
    titulo : str, optional
        Título opcional para el reporte
    """
    if titulo:
        print(f"\n{'='*70}")
        print(f"{titulo}")
        print(f"{'='*70}")
    
    # Cobertura
    cob = metricas['cobertura']
    print(f"\n📏 COBERTURA DEL RANGO MAREAL:")
    print(f"  • Rango válido:     {cob['rango_valido']:.2f} m  [{cob['min_valido']:.2f} → {cob['max_valido']:.2f}]")
    print(f"  • Rango total:      {cob['rango_total']:.2f} m  [{cob['min_total']:.2f} → {cob['max_total']:.2f}]")
    print(f"  • Cobertura:        {cob['cobertura']:.3f} ({cob['cobertura_pct']:.1f}%)")
    
    # Uniformidad KS
    ks = metricas['uniformidad_ks']
    print(f"\n📐 TEST DE UNIFORMIDAD (Kolmogorov-Smirnov):")
    print(f"  • Estadístico KS:   {ks['ks_statistic']:.4f}")
    print(f"  • P-value:          {ks['ks_pvalue']:.4f}")
    print(f"  • ¿Uniforme?        {'Sí' if ks['uniforme'] else 'No'} ({ks['interpretacion']})")
    
    # Entropía
    ent = metricas['entropia']
    print(f"\n🔢 ENTROPÍA DE SHANNON:")
    print(f"  • Entropía:         {ent['entropia']:.3f} bits")
    print(f"  • Entropía máx:     {ent['entropia_max']:.3f} bits")
    print(f"  • Normalizada:      {ent['entropia_norm']:.3f} (0-1, mayor = más uniforme)")
    print(f"  • Bins ocupados:    {ent['n_bins_ocupados']}/{ent['n_bins']} ({ent['n_bins_vacios']} vacíos)")
    
    # Dispersión
    disp = metricas['dispersion']
    print(f"\n📊 ÍNDICE DE DISPERSIÓN (VMR):")
    print(f"  • VMR:              {disp['vmr']:.3f}")
    print(f"  • Patrón:           {disp['patron']}")
    print(f"  • Media gaps:       {disp['media_gaps']:.4f} m")
    print(f"  • Varianza gaps:    {disp['varianza_gaps']:.4f} m²")
    
    # Estadísticos de gaps
    gaps = metricas['gaps']
    print(f"\n📏 ESTADÍSTICOS DE GAPS (distancias entre vecinos):")
    print(f"  • N° muestras:      {gaps['n_muestras']}")
    print(f"  • N° gaps:          {gaps['n_gaps']}")
    print(f"  • Media:            {gaps['media']:.4f} m")
    print(f"  • Mediana:          {gaps['mediana']:.4f} m")
    print(f"  • Desv. estándar:   {gaps['std']:.4f} m")
    print(f"  • Coef. variación:  {gaps['cv']:.3f}")
    print(f"  • Min / Max:        {gaps['min']:.4f} m / {gaps['max']:.4f} m")
    print(f"  • IQR (p25-p75):    {gaps['iqr']:.4f} m")
    print(f"  • Asimetría:        {gaps['skewness']:.3f}")
    print(f"  • Curtosis:         {gaps['kurtosis']:.3f}")
    
    # VSR (Vertical Sampling Resolution) - Métrica específica waterline
    vsr = metricas['vsr']
    print(f"\n🌊 VSR - RESOLUCIÓN VERTICAL DE MUESTREO (método waterline):")
    print(f"  • VSR (media):      {vsr['vsr']:.4f} m")
    print(f"  • VSR (mediana):    {vsr['vsr_mediana']:.4f} m")
    print(f"  • Niveles únicos:   {vsr['n_niveles']}")
    print(f"  • Gap mínimo:       {vsr['vsr_min']:.4f} m")
    print(f"  • Gap máximo:       {vsr['vsr_max']:.4f} m")
    if not np.isnan(vsr['max_gap_ubicacion'][0]):
        print(f"  • Mayor hueco:      {vsr['max_gap']:.4f} m  [{vsr['max_gap_ubicacion'][0]:.2f} → {vsr['max_gap_ubicacion'][1]:.2f}]")
    print(f"  • Calidad:          {vsr['calidad_vsr']}")
    
    # Representatividad
    rep = metricas['representatividad']
    print(f"\n📈 ÍNDICE DE REPRESENTATIVIDAD:")
    print(f"  • Representatividad: {rep['representatividad']:.3f} ({rep['representatividad_pct']:.1f}%)")
    print(f"  • Quantiles cubiertos: {rep['quantiles_cubiertos']}/{rep['quantiles_totales']}")
    if rep['quantiles_vacios']:
        print(f"  • Quantiles vacíos:  {rep['quantiles_vacios']}")
    else:
        print(f"  • Quantiles vacíos:  Ninguno (cobertura completa)")



def evaluar_calidad_distribucion(
    valores_validos: np.ndarray,
    valores_totales: np.ndarray,
    umbral_cobertura: float = 0.85,
    umbral_entropia: float = 0.70,
    umbral_vmr: float = 1.2,
    umbral_vsr: float = 0.25,
    umbral_representatividad: float = 0.80
) -> Dict[str, any]:
    """
    Evalúa la calidad de una distribución de mareas con criterios cuantitativos.
    
    Retorna una evaluación booleana de si la distribución es adecuada para
    modelado batimétrico usando el método waterline basándose en umbrales
    de las métricas clave.
    
    Parámetros
    ----------
    valores_validos : array-like
        Alturas de marea en las fechas seleccionadas
    valores_totales : array-like
        Distribución completa de referencia
    umbral_cobertura : float, default=0.85
        Cobertura mínima aceptable (0-1)
    umbral_entropia : float, default=0.70
        Entropía normalizada mínima aceptable (0-1)
    umbral_vmr : float, default=1.2
        VMR máximo aceptable (>1 indica clustering)
    umbral_vsr : float, default=0.25
        VSR máxima aceptable en metros (menor es mejor)
    umbral_representatividad : float, default=0.80
        Representatividad mínima aceptable (0-1)
    
    Retorna
    -------
    dict
        metricas : dict
            Todas las métricas calculadas
        criterios : dict
            Evaluación booleana de cada criterio
        n_criterios_cumplidos : int
            Número de criterios que se cumplen
        calidad_global : str
            "Excelente", "Buena", "Aceptable", "Pobre"
        apta : bool
            True si cumple los criterios mínimos esenciales
        recomendaciones : list of str
            Sugerencias para mejorar la distribución
    """
    metricas = calcular_metricas_completas(valores_validos, valores_totales)
    
    criterios = {
        'cobertura_suficiente': metricas['cobertura']['cobertura'] >= umbral_cobertura,
        'uniforme_ks': metricas['uniformidad_ks']['uniforme'],
        'entropia_alta': metricas['entropia']['entropia_norm'] >= umbral_entropia,
        'no_agrupado': metricas['dispersion']['vmr'] <= umbral_vmr,
        'vsr_adecuado': metricas['vsr']['vsr'] <= umbral_vsr,
        'representativo': metricas['representatividad']['representatividad'] >= umbral_representatividad,
    }
    
    n_criterios_cumplidos = sum(criterios.values())
    n_criterios_totales = len(criterios)
    
    # Clasificación de calidad
    ratio = n_criterios_cumplidos / n_criterios_totales
    if ratio >= 0.85:
        calidad = "Excelente"
    elif ratio >= 0.65:
        calidad = "Buena"
    elif ratio >= 0.50:
        calidad = "Aceptable"
    else:
        calidad = "Pobre"
    
    # Criterios esenciales para método waterline
    apta = (criterios['cobertura_suficiente'] and 
            criterios['vsr_adecuado'] and 
            criterios['representativo'])
    
    # Generar recomendaciones
    recomendaciones = []
    if not criterios['cobertura_suficiente']:
        cob = metricas['cobertura']['cobertura_pct']
        recomendaciones.append(
            f"Cobertura insuficiente ({cob:.1f}%). Añadir imágenes en mareas extremas."
        )
    if not criterios['vsr_adecuado']:
        vsr = metricas['vsr']['vsr']
        recomendaciones.append(
            f"VSR demasiado alta ({vsr:.3f} m). Aumentar el número de observaciones."
        )
    if not criterios['representativo']:
        rep = metricas['representatividad']['representatividad_pct']
        vacios = metricas['representatividad']['quantiles_vacios']
        recomendaciones.append(
            f"Baja representatividad ({rep:.1f}%). Añadir imágenes en quantiles {vacios}."
        )
    if not criterios['uniforme_ks']:
        recomendaciones.append(
            "Distribución no uniforme. Seleccionar imágenes más distribuidas en el rango mareal."
        )
    if not criterios['no_agrupado']:
        vmr = metricas['dispersion']['vmr']
        recomendaciones.append(
            f"Muestras agrupadas (VMR={vmr:.2f}). Distribuir observaciones más uniformemente."
        )
    
    return {
        'metricas': metricas,
        'criterios': criterios,
        'n_criterios_cumplidos': n_criterios_cumplidos,
        'n_criterios_totales': n_criterios_totales,
        'calidad_global': calidad,
        'apta': apta,
        'recomendaciones': recomendaciones,
    }
