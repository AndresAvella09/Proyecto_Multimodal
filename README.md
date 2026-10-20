# Diagnóstico estadístico y corrección del desbalance modal en clasificación multimodal

Proyecto final del curso **Deep Representation Learning para Ciencia de Datos**.

**Integrantes:** Michelle Ruiz, Diego Avella

## Resumen

En modelos que combinan imagen y texto, una modalidad suele dominar el entrenamiento
(*greedy modality*) y la otra queda subaprovechada. Este proyecto mide ese desbalance en el
espacio de representaciones y propone una fusión ponderada por la incertidumbre de cada
modalidad.

- **Dataset:** MM-IMDB (pósters + sinopsis, 23 géneros, multietiqueta)
- **Encoders:** CLIP ViT-B/32 (imagen y texto), congelados
- **Métrica principal:** F1 macro, media ± IC 95% sobre 5 semillas

## Estructura

| Carpeta | Contenido |
| --- | --- |
| `configs/` | Parámetros de los experimentos |
| `data/` | Datos y embeddings (no se suben; ver abajo) |
| `docs/` | Propuesta, brief y material de estudio |
| `notebooks/` | Notebooks de exploración y análisis |
| `results/` | Tablas y figuras de resultados |
| `scripts/` | Scripts ejecutables (extracción, entrenamiento) |
| `src/` | Código reutilizable (modelos, métricas) |

## Datos

Los datos y embeddings no se guardan en GitHub por su tamaño. Viven en Google Drive y se
copian a `data/` localmente o se montan desde Colab.
