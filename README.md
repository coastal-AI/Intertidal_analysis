# Intertidal Analysis

Herramienta para el análisis de zonas intermareales a partir de imágenes Sentinel-2 y la **Scene Classification Layer (SCL)** utilizando OpenEO y modelos de marea.

El proyecto implementa un flujo de trabajo para:

* Descargar imágenes Sentinel-2 desde Copernicus Dataspace.
* Filtrar escenas según su calidad.
* Construir mapas de referencia agua/tierra.
* Analizar la variabilidad de la zona intermareal.
* Comparar los resultados con modelos de marea.

## Estructura

```
Intertidal_analysis/
├── intertidal/                     # Código principal
├── stand-by/                       # Código y notebooks antiguos
├── gijon_sentinel2_scl_refactored.ipynb
├── tidemodel.py
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone https://github.com/coastal-AI/Intertidal_analysis.git
cd Intertidal_analysis

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Uso

El punto de entrada recomendado es el notebook:

```
gijon_sentinel2_scl_refactored.ipynb
```

El código está organizado en varios módulos dentro del paquete `intertidal`, que agrupan la funcionalidad relacionada con:

* geometría
* descarga de datos mediante OpenEO
* procesamiento de la capa SCL
* generación de mapas
* visualización
* análisis de marea

## Estado

Proyecto en desarrollo.
