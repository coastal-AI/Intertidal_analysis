# Dataset Zona Intermareal - Ría de Foz

**Imágenes Sentinel-2 RGB sincronizadas con predicciones de altura de marea**

---

## 📋 Descripción General

Este dataset contiene pares de **imágenes satelitales y alturas de marea** para el estuario de la Ría de Foz (Galicia, España). Está diseñado para entrenar **modelos generativos para rellenar huecos en distribuciones de rango mareal**.

### Características

- **Imágenes satelitales**: Sentinel-2 L2A RGB (10m resolución)
- **Clasificación de escena**: Máscaras SCL de ESA con 12 clases
- **Altura de marea**: Modelo GOT4.10 en momento exacto de adquisición
- **Filtrado de calidad**: Cobertura de nubes <10% en zona de transición
- **Cobertura temporal**: 5 años (2021–2026)
- **Formato normalizado**: PNG uint8 (0-255) listo para ML

---

## 📁 Estructura del Dataset

```
dataset_foz/
├── README.md                    # Este archivo
├── dataset_metadata.yml         # Metadatos estructurados
├── dataset_foz.csv              # Archivo índice principal
├── requirements.txt             # Dependencias Python
│
├── rgb_png/                     # Imágenes RGB normalizadas (PNG)
│   ├── rgb_2021-07-16.png
│   └── ...
│
├── scl_png/                     # Máscaras de clasificación (PNG)
│   ├── scl_2021-07-16.png
│   └── ...
│
├── rgb/                         # GeoTIFF originales (reflectancia 0-10000)
├── scl/                         # GeoTIFF SCL originales
├── reference_map.tif            # Máscara de zona de transición
├── tide_models/                 # Archivos del modelo GOT4.10
│
└── intertidal/                  # Paquete Python (código fuente)
```

---

## 📊 Especificación del CSV

### Esquema de `dataset_foz.csv`

| Columna            | Tipo     | Descripción                                       |
| ------------------ | -------- | ------------------------------------------------- |
| `fecha`            | date     | Fecha de adquisición (YYYY-MM-DD)                 |
| `hora_utc`         | time     | Hora de adquisición (HH:MM:SS UTC)                |
| `datetime_utc`     | datetime | Timestamp ISO 8601                                |
| `imagen_rgb`       | path     | Ruta relativa al PNG RGB normalizado              |
| `imagen_scl`       | path     | Ruta relativa al PNG SCL                          |
| `marea_m`          | float    | Altura de marea en metros (GOT4.10)               |
| `nubes_pct`        | float    | % de nubes en zona de transición                  |
| `cobertura_tile_pct` | float  | % del AOI con datos válidos (detección de tiles parciales) |
| `lat`              | float    | Latitud de referencia para cálculo de marea       |
| `lon`              | float    | Longitud de referencia para cálculo de marea      |
| `imagen_rgb_tif`   | path     | Ruta al GeoTIFF original (backup)                 |
| `imagen_scl_tif`   | path     | Ruta al GeoTIFF SCL original (backup)             |

### Propiedades de las Imágenes

**Imágenes RGB** (`rgb_png/*.png`)
- **Formato**: PNG, RGB, uint8 (0-255)
- **Bandas**: Rojo (B04), Verde (B03), Azul (B02)
- **Normalización**: Estiramiento de percentiles (p2-p98)
- **Resolución**: ~10m/píxel (UTM Zone 29N, EPSG:32629)
- **Dimensiones**: ~659×786 píxeles
- **Fondo**: Píxeles negros (0,0,0) fuera del polígono del AOI

**Imágenes SCL** (`scl_png/*.png`)
- **Formato**: PNG, RGB (coloreado desde datos categóricos)
- **Clases**: 12 clases SCL de ESA Sentinel-2
- **Resolución**: ~20m/píxel nativo, remuestreado a 10m
- **Paleta**: Colores oficiales ESA

### Leyenda de Colores SCL

| Clase | Etiqueta              | Color       | Hex       |
| ----- | --------------------- | ----------- | --------- |
| 0     | Sin datos             | Negro       | `#000000` |
| 1     | Saturado/Defectuoso   | Rojo        | `#FF0000` |
| 2     | Área oscura           | Gris oscuro | `#2F2F2F` |
| 3     | Sombra de nube        | Marrón      | `#643200` |
| 4     | Vegetación            | Verde       | `#00A000` |
| 5     | No vegetación         | Amarillo    | `#FFE65A` |
| 6     | Agua                  | Azul        | `#0000FF` |
| 7     | No clasificado        | Gris        | `#808080` |
| 8     | Nube (prob. media)    | Gris claro  | `#C0C0C0` |
| 9     | Nube (prob. alta)     | Blanco      | `#FFFFFF` |
| 10    | Cirrus fino           | Azul claro  | `#64C8FF` |
| 11    | Nieve/Hielo           | Rosa        | `#FF96FF` |

---

## 🚀 Uso Rápido

### Cargar Dataset en Python

```python
import pandas as pd
from PIL import Image

# Cargar índice del dataset
df = pd.read_csv('dataset_foz.csv')
print(f"Total de muestras: {len(df)}")

# Cargar primera muestra
row = df.iloc[0]
rgb = Image.open(row['imagen_rgb'])
scl = Image.open(row['imagen_scl'])
marea = row['marea_m']

print(f"Fecha: {row['fecha']}, Marea: {marea:.2f}m")
```

### PyTorch DataLoader (opcional)

```python
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class IntertidalDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or T.ToTensor()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = Image.open(row['imagen_rgb'])
        marea = row['marea_m']
        
        if self.transform:
            rgb = self.transform(rgb)
        
        return rgb, marea

# Uso
dataset = IntertidalDataset('dataset_foz.csv')
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### Filtrar por Cobertura de Tile

```python
# Filtrar solo imágenes con cobertura completa (≥95%)
df_full = df[df['cobertura_tile_pct'] >= 95.0]
print(f"Imágenes con cobertura completa: {len(df_full)} / {len(df)}")

# O mantener todas las imágenes con >50% de datos válidos
df_partial = df[df['cobertura_tile_pct'] >= 50.0]
print(f"Imágenes con ≥50% cobertura: {len(df_partial)} / {len(df)}")
```

---

## 🌍 Área de Estudio

**Ría de Foz** (Galicia, España)
- **Coordenadas**: 43.5333°N, 7.2417°W
- **Tipo**: Estuario mesomareal (rango de marea 2-4m)
- **Polígono**: Pentágono cubriendo ~52 km²

**Características de marea**:
- Rango medio: ~2.5 m
- Régimen: Semidiurno (2 altas/bajas por día)

---

## 🛠️ Pipeline de Procesamiento

1. **Adquisición**: Sentinel-2 L2A desde Copernicus Dataspace (OpenEO API)
2. **Filtrado de calidad**: Cobertura de nubes <10% en zona de transición
3. **Recorte espacial**: Polígono exacto del AOI (no bounding box)
4. **Cálculo de marea**: Modelo GOT4.10 con radio de búsqueda 2°
5. **Normalización**: 
   - RGB: Estiramiento de percentiles (p2-p98) → uint8
   - SCL: Colorización categórica con paleta ESA

Ver [`dataset_metadata.yml`](dataset_metadata.yml) para detalles completos.

---

## 📜 Fuentes de Datos

- **Imágenes Sentinel-2**: © European Space Agency (ESA) / Copernicus
- **Modelo de marea GOT4.10**: Richard D. Ray, NASA GSFC
- **Librería pyTMD**: Tyler Sutterley

---

## � Notas Técnicas: Tiles de Sentinel-2

### Sistema de Cuadrículas del Satélite

Sentinel-2 captura imágenes organizadas en **tiles** (cuadrículas) de aproximadamente **100×100 km**. Cada tile tiene un identificador único (ej: `29TNH`, `29TPH`).

**¿Por qué algunas imágenes están parcialmente visibles?**

El área de estudio (Ría de Foz) se encuentra en el **límite entre varios tiles de Sentinel-2**. Dependiendo de la órbita del satélite en cada fecha:

- **Cobertura completa**: El AOI cae completamente dentro de un tile → imagen completa
- **Cobertura parcial**: El AOI está dividido entre dos tiles, pero solo uno fue capturado en esa órbita → imagen parcial

**Impacto en el dataset**:
- ✅ Las imágenes parciales **contienen información válida** de la zona intermareal
- ✅ La altura de marea es correcta para toda el área (calculada en el centroide)
- ⚠️ Algunos bordes del polígono pueden aparecer sin datos (píxeles negros)
- 📊 La columna **`cobertura_tile_pct`** indica el % del AOI con datos válidos (útil para filtrar imágenes parciales si es necesario)

Esta es una característica normal de los datos satelitales y **no afecta el entrenamiento de modelos generativos**, ya que:
1. Los modelos aprenden de regiones válidas (no-NaN)
2. La distribución de mareas se mantiene consistente
3. El filtrado de calidad (<10% nubes) garantiza suficiente información útil

---

## �🔧 Reproducir el Dataset

Para regenerar este dataset:

1. Ver notebook `dataset_generator.ipynb` en el directorio padre
2. Modificar parámetros `site_name`, `aoi_dms`, `time_extent`
3. Ejecutar celdas secuencialmente

---

**Última actualización**: 2026-07-09  
**Versión del dataset**: 1.0




