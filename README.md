# ¿Póster o sinopsis? Desbalance entre modalidades en películas de ayer y de hoy

Proyecto final del curso **Deep Representation Learning para Ciencia de Datos** (Escuela Colombiana de Ingeniería).

**Integrantes:** Michelle Ruiz, Diego Avella

## Resumen

Un modelo que combina imagen y texto puede terminar apoyándose casi solo en una de las dos fuentes
y desaprovechar la otra (*desbalance entre modalidades* o *greedy modality*). Este proyecto lo estudia
con películas: el **póster** (imagen) y la **sinopsis** (texto) de cada película, para predecir y buscar
por **género**.

**Pregunta central:** ¿el modelo depende más del póster o de la sinopsis, y ese desbalance cambia con
películas más recientes?

El proyecto tiene dos partes que comparten datos y representaciones:

1. **Comparación temporal:** se entrenan clasificadores de género con MM-IMDB (películas hasta 2016) y se
   evalúan con películas de 2020 a 2026 descargadas de TMDB. Se separa el efecto de la **época** del efecto
   de la **fuente de datos**.
2. **Buscador semántico:** el usuario escribe una descripción en español o inglés ("una casa embrujada en
   la nieve") y obtiene el póster y el título de las películas más parecidas. Se compara buscar por póster,
   por sinopsis o por ambos, antes y después de adaptar la representación con entrenamiento.

Las dos partes siguen el mismo flujo: **representación inicial** (encoders preentrenados y congelados) →
**entrenamiento** de capas pequeñas → **representación final** → **efecto en la tarea**.

## Datos y modelos

| | Qué es | Uso |
| --- | --- | --- |
| [MM-IMDB](https://www.kaggle.com/datasets/johnarevalo/mmimdb) | 25 959 películas con póster, sinopsis y 23 géneros (multietiqueta) | Dataset principal |
| [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets) | Año de estreno, votos y géneros actuales | Año de cada película y etiquetas actualizadas |
| [TMDB API](https://developer.themoviedb.org/) | Pósters y sinopsis | Películas 2020-2026 y grupo de control |
| CLIP ViT-B/32 (OpenAI) | Encoder de imagen y texto | Representación del póster |
| DistilBERT | Encoder de texto | Representación de la sinopsis (clasificación) |
| [CLIP multilingüe](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) | Texto en 50+ idiomas al espacio de CLIP | Búsquedas en español e inglés contra los pósters |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | Encoder de texto para búsqueda | Representación de la sinopsis (buscador) |

Todos los encoders se usan **congelados**: solo se entrenan capas pequeñas encima, así todo corre en Colab.

> Los datos de IMDb son de uso personal y no comercial. Este producto usa la API de TMDB pero no está
> respaldado ni certificado por TMDB.

## Notebooks

| # | Notebook | Qué hace |
| --- | --- | --- |
| 01 | `01_descarga_y_preparacion` | Descarga MM-IMDB, decodifica sinopsis y pósters → `mmimdb_meta.parquet` |
| 02 | `02_exploracion` | Desbalance de clases, co-ocurrencia de géneros, largo de las sinopsis |
| 03 | `03_extraccion_embeddings` | Embeddings congelados: CLIP (imagen y texto) y DistilBERT |
| 04 | `04_baselines` | Solo póster vs. solo sinopsis vs. fusión, 5 semillas |
| 05 | `05_anios_y_peliculas_recientes` | Año y etiquetas actuales de IMDb, películas 2020-2026 y grupo de control |
| 06 | `06_diagnostico_modal` | ¿De qué modalidad depende la fusión? Ablación, fusión tardía, información aportada, calibración, por época |
| 07 | `07_tmdb_embeddings` | Descarga pósters y sinopsis de TMDB y los procesa igual que MM-IMDB |
| 08 | `08_comparacion_temporal` | Efecto de la fuente y de la época sobre cada modalidad |
| 09 | `09_buscador_semantico` | Buscador con CLIP multilingüe: por póster, por sinopsis o ambos |
| 10 | `10_buscador_evaluacion` | Sinopsis con E5 y evaluación del buscador (precision@10, mAP) |
| 11 | `11_buscador_adaptado` | Adaptador entrenado por modalidad: representación y búsqueda antes vs. después |

`multimodal_exploration.ipynb` es un notebook didáctico que construye un mini-CLIP desde cero.

## Resultados principales (hasta ahora)

**Clasificación de géneros (notebook 04, F1 macro en test, 5 semillas):**
solo póster 47.7 · solo sinopsis 59.1 · **fusión 63.6**. La fusión supera a la sinopsis sola por +4.5 ± 0.2
puntos, así que con encoders congelados la imagen sí aporta.

**Buscador (notebooks 10 y 11, 88 búsquedas de género en español e inglés, solo películas de test):**

| Forma de buscar | mAP sin entrenar | mAP con adaptador |
| --- | --- | --- |
| Solo póster | 0.209 | 0.374 |
| Solo sinopsis | 0.226 | 0.452 |
| **Póster + sinopsis** | 0.272 | **0.513** |

- Combinar póster y sinopsis supera a cada uno por separado.
- Cuando la sinopsis está mal representada (CLIP), la combinación casi la ignora (mejor peso del póster:
  0.9). Con una buena representación (E5), el mejor peso queda cerca de 0.5: **el desbalance depende de
  la calidad de la representación de cada modalidad**.
- El adaptador mejora la búsqueda en los 22 géneros evaluados y separa mejor los géneros en el espacio de
  representaciones.

Los resultados de la comparación temporal (notebooks 06-08) se agregarán cuando estén completos.

## Estructura del repositorio

| Carpeta | Contenido |
| --- | --- |
| `notebooks/` | Notebooks del 01 al 11 (se abren en Colab con el botón de cada uno) |
| `src/` | Código compartido: carga de datos, encoders, modelos y métricas |
| `docs/` | Propuesta inicial, brief y material de estudio |
| `results/` | Tablas y figuras de resultados |
| `configs/`, `scripts/` | Parámetros y scripts ejecutables |

## Cómo reproducir

Los datos y embeddings no se suben a GitHub por su tamaño. Los notebooks esperan esta estructura en Google Drive:

```
Mi unidad/proyecto_multimodal/
├── data/          # parquets (meta, años, TMDB), miniaturas de pósters
├── embeddings/    # vectores .npy (CLIP, DistilBERT, E5, adaptados)
├── modelos/       # pesos de los adaptadores del buscador
└── results/       # métricas en CSV
```

1. Abrir cada notebook en Colab (botón *Open in Colab*) y correrlo **en orden** desde el 01.
2. Usar GPU (T4) para el 03, 07 y 10. El resto funciona en CPU.
3. El notebook 07 necesita una llave gratuita de la API de TMDB.
