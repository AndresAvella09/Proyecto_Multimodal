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

**Respuesta corta:** la fusión depende sobre todo de la sinopsis (índice de dependencia de la imagen 0.20), y con
películas de 2020-2026 la sinopsis pierde ~4 puntos de AP mientras el póster se mantiene. Ese cambio no es un
desbalance de aprendizaje que se pueda corregir repesando o modulando el entrenamiento: el texto de las películas
nuevas **se equivoca con seguridad**, sin anunciar su incertidumbre. En la búsqueda semántica, en cambio, las
películas nuevas se encuentran igual de bien.

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
| 12 | `12_buscador_recientes` | Agrega al buscador las películas 2020-2026 (sin reentrenar) y mide el efecto de la fuente y de la época en la búsqueda |
| 13 | `13_app_buscador` | Exporta los datos de la app (`scripts/exportar_app.py`), la lanza desde Colab y la publica en Hugging Face |

Los notebooks 01-08 son la base común. Desde ahí hay dos líneas:

- **Buscador (09-13):** búsqueda semántica por póster y sinopsis, y su app.
- **Clasificación y desbalance (C1-C3):** métodos para corregir el desbalance entre modalidades en la
  clasificación de géneros, sobre los modelos del 06 y los grupos del 08.

| # | Notebook | Qué hace |
| --- | --- | --- |
| C1 | `C1_fusion_incertidumbre` | Fusión ponderada por incertidumbre (compuerta por entropía) contra la fusión ingenua, en los cuatro grupos del 08 |
| C2 | `C2_dial_degradacion` | Provoca el desbalance: degrada una modalidad en test (borrado, ruido y mezcla con otra película) y entrena con pocos datos; compuerta ajustada con degradación sintética |
| C3 | `C3_ogm_ge_tabla_final` | OGM-GE (modulación del gradiente por modalidad), tabla final con todos los métodos en los cuatro grupos y análisis de fallos |

`multimodal_exploration.ipynb` es un notebook didáctico que construye un mini-CLIP desde cero.

## Resultados principales

### 1. Clasificación de géneros (notebooks 04 y 06)

F1 macro en test, 5 semillas: solo póster 47.7 · solo sinopsis 59.1 · **fusión 63.6**. La fusión supera a la
sinopsis sola por +4.5 ± 0.2 puntos: con encoders congelados la imagen sí aporta.

### 2. Diagnóstico del desbalance (notebook 06, etiquetas de IMDb hoy, test, 5 semillas)

- La fusión (AP macro 63.8) supera al texto solo (59.1) y a la imagen sola (44.3): con encoders congelados no
  aparece la *greedy modality* clásica.
- Pero la fusión depende sobre todo del texto: al apagar la imagen cae 6.3 puntos de AP y al apagar el texto
  cae 25.1 (índice de dependencia de la imagen 0.20). La información nueva que aporta la imagen dado el texto es
  0.29 nats por película (IC 95% [0.27, 0.31]).
- Las tres ramas están igual de levemente sobreconfiadas (temperatura ≈ 1.08), y la fusión tardía con pesos fijos
  no le gana a la concatenación.

### 3. Comparación temporal (notebooks 07 y 08, mismos modelos sin reentrenar)

- **Fuente** (mismas películas con póster y sinopsis de TMDB en vez de MM-IMDB): sin efecto detectable en el AP.
- **Época** (películas 2020-2026 contra el control, ambos de TMDB): el texto pierde 4.1 puntos de AP
  (IC [−6.9, −1.1]) y la imagen no cambia (+0.1). La fusión pasa a depender más de la imagen (índice 0.22 → 0.27)
  y la información que aporta la imagen dado el texto sube de 0.32 a 0.46 nats. **El desbalance entre modalidades
  no es fijo: cambia con la época de las películas.**
- La caída del texto se mantiene dentro de los mismos tercios de popularidad, así que no es un efecto de que las
  recientes sean más votadas.

### 4. ¿Se puede corregir? (notebooks C1-C3)

| Método | Notebook | Idea | AP − ingenua en test | AP − ingenua en recientes |
| --- | --- | --- | --- | --- |
| `tardia` / `compuerta` | C1 | Pesos por género / más peso a la rama segura en cada película | +0.09 / +0.09 | −0.04 / −0.06 |
| **`tardia3` / `compuerta3`** | C1 | Lo mismo, con la fusión ingenua como tercera fuente | **+0.39 \*** / **+0.38 \*** | +0.30 / +0.30 |
| `tardia_aum` / `compuerta_aum` | C2 | Compuerta ajustada con copias degradadas de dev | −0.17 / −0.36 \* | −0.36 / −0.42 |
| `ogm` / `ogm_ge` | C3 | Frenar el gradiente de la modalidad dominante (OGM-GE) | −0.03 / +0.05 | −0.08 / −0.13 |

Puntos de AP macro contra la fusión ingenua, bootstrap por películas (\* = IC 95% sin el 0). Ningún método
mejora con significancia en las recientes, ni gana más en las recientes que en el control.

- **C1 — La entropía no avisa.** Con las recientes, la entropía del texto casi no cambia (0.186 → 0.192), pero su
  *exceso* de sorpresa (log-loss − entropía) pasa de 0.001 a 0.028: **el texto se equivoca con seguridad**. La
  compuerta aprende β < 0 (modera a la rama segura) y su peso para la imagen no sube en las recientes. Un oráculo
  ajustado con las propias recientes no mejora: el límite es la información que recibe la compuerta.
- **C2 — Provocar el desbalance.** Degradando una modalidad en test:
  - *Borrado* (encoger el embedding): el texto mantiene su AP pero triplica su entropía; queda subconfiado.
  - *Ruido*: pierde AP y también queda subconfiado.
  - *Mezcla con otra película* (exploratoria): el AP del texto cae de 0.59 a 0.21 con la **entropía igual** y el
    exceso positivo, el mismo patrón de las recientes, que equivalen a una mezcla de ρ ≈ 0.2.

  En todo el dial, el término de entropía no agrega nada a la fusión tardía. Lo que sí ayuda cuando el texto se
  daña fuerte es mover el peso fijo hacia la imagen (+3.9 puntos con mezcla ρ = 0.8), pero con el daño leve de las
  recientes ese cambio cuesta más de lo que gana. Entrenando con solo el 2% del train la fusión sigue ganándole a
  la mejor rama (+3.6 puntos): **no aparece greedy modality ni con pocos datos**.
- **C3 — Modular el entrenamiento.** OGM-GE casi no actúa: según su propia medida las modalidades ya aportan
  parecido (razón ≈ 0.95), y el índice de dependencia no cambia (0.20 → 0.19). La fusión depende más del texto
  porque el texto trae más información, no porque la imagen haya aprendido menos.
- **Análisis de fallos.** Por género, la fusión hereda el cambio del texto (r = 0.95). Lo que más pierde el texto
  con la época: War (−25 puntos), Musical (−23), Family, Sci-Fi y Documentary (−17). En las películas donde falla
  con seguridad, el texto cae a géneros "por defecto" (Drama, Romance, Comedy) en películas de terror, thriller o
  animación, y la imagen acierta más.

**Conclusión:** el desplazamiento de época no es un desbalance de aprendizaje sino un **error seguro del texto**.
Ni repesar las modalidades al combinar (con la confianza de cada rama) ni modular los gradientes al entrenar lo
detecta, porque la rama que falla no anuncia su incertidumbre.

### 5. Buscador semántico (notebooks 10-12, 88 búsquedas de género en español e inglés)

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
- **Películas 2020-2026 (notebook 12, sin reentrenar):** a diferencia de la clasificación, la búsqueda **no
  envejece**. Sin adaptador las recientes se buscan mejor que el control (+0.048 AP, IC sin el 0); con el
  adaptador del 11 la diferencia desaparece (+0.009, IC [−0.026, +0.042]). La fuente no tiene efecto y el mejor
  peso del póster es 0.4 en todos los grupos. Una explicación posible: E5 es más nuevo que DistilBERT y está hecho
  para buscar.
- **Catálogo mezclado de la app:** con la estandarización global, los estrenos salían 1.7 veces más de lo que les
  corresponde en el top 50, y el doble entre los errores. El sesgo viene de la época, no de la fuente. Estandarizar
  los puntajes **dentro de cada fuente** lo baja a 1.16 y cuesta 0.002 de mAP; es la corrección que usa la app.

## La app

Una app web en Streamlit (`app/`) junta las dos partes del proyecto. Usa el código de `src/` (el mismo de los
notebooks) y no entrena nada: todo lo pesado se calcula antes, al exportar.

| Página | Qué muestra |
| --- | --- |
| Inicio | Portada |
| Buscador | Búsqueda por póster, sinopsis o ambos, sin entrenar o con el adaptador, con los estrenos 2020-2026 (z-score por fuente). Incluye la arquitectura, los resultados del buscador y la explicación técnica |
| Clasificador | Para una película del catálogo, los géneros que predicen el póster solo, la sinopsis sola y la fusión (modelos del 06, logits precalculados: no carga CLIP ni DistilBERT) |
| Desbalance | En esa misma película: apagar cada rama, pesos de la compuerta del C1 y el dial del C2 (borrado, ruido o mezcla con otra película) |
| Ayer y hoy | Resultados del 08, 12 y C1-C3 leídos de `results/`: fuente contra época, tabla final de métodos y fallos por género |
| Espacio 3D | PCA y UMAP en 3D (Plotly) de los géneros antes y después del adaptador |

**Datos.** `scripts/exportar_app.py` (sección 1 del notebook 13, en Colab con el Drive montado) llena
`app_buscador/` en Drive: `catalogo.parquet` (con los pósters), vectores del buscador, `clasificador.npz` (logits
calibrados por película, umbrales, pesos de las cabezas y de las compuertas), `clasif_img.npy` y `clasif_txt.npy`,
`espacio3d.parquet`, `config.json` y una copia de los CSV de `results/` y `resultados/`. Todo sin pickle. La app busca
esa carpeta en `APP_DATOS`, luego en `app/datos/` (ignorada por git) y, si se define `APP_DATASET`, la descarga de ese
repo de datasets de Hugging Face.

**Correrla en la PC.** Copiar `app_buscador/` de Drive a `app/datos/` y, desde la raíz del repo:

```bash
pip install -r app/requirements.txt
```

```bash
streamlit run app/Inicio.py
```

**Publicarla en Hugging Face** (sección 4 del notebook 13). `scripts/publicar_hf.py` sube los datos (~400 MB) a un
repo de datasets privado y el código (`app/`, `src/`, los CSV de `results/`, `app/Dockerfile` y
`app/requirements.txt`) a un Space de Docker, y le pone al Space la variable `APP_DATASET`. Pasos:

1. Crear un token de escritura en Hugging Face y guardarlo en los Secretos de Colab como `HF_TOKEN`. Las llaves de
   TMDB y de Hugging Face **solo** van en los Secretos de Colab o del Space, nunca en el código: el repo es público.
2. Correr la sección 4 del notebook 13 con `USUARIO_HF` y `PUBLICAR = True`.
3. En el Space, *Settings → Variables and secrets*, agregar el secreto `HF_TOKEN` con un token de **lectura** (el
   dataset es privado). La primera construcción tarda unos 5-10 minutos.

Las páginas que necesitan algo que falta (por ejemplo los CSV del C2 y el C3 si no están en `results/`) muestran un
aviso en vez de fallar.

## Estructura del repositorio

| Carpeta | Contenido |
| --- | --- |
| `notebooks/` | Notebooks 01-13 y C1-C3 (se abren en Colab con el botón de cada uno) |
| `src/` | Código compartido: carga de datos, encoders, modelos, métricas, fusión y buscador |
| `app/` | App web en Streamlit (`Inicio.py`, `comun.py`, `pages/`), `Dockerfile` y `requirements.txt` del Space |
| `scripts/` | `exportar_app.py` (datos de la app) y `publicar_hf.py` (Hugging Face) |
| `docs/` | Propuesta inicial, brief y material de estudio |
| `results/` | Tablas de resultados de clasificación (`exp1`-`exp6`) y del buscador (`results/buscador/`) |
| `configs/` | Parámetros |

## Cómo reproducir

Los datos y embeddings no se suben a GitHub por su tamaño. Los notebooks esperan esta estructura en Google Drive:

```
Mi unidad/proyecto_multimodal/
├── data/          # parquets (meta, años, TMDB), miniaturas de pósters
├── embeddings/    # vectores .npy (CLIP, DistilBERT, E5, adaptados)
├── modelos/       # pesos de los adaptadores del buscador
├── results/       # clasificación y comparación temporal (06-08, C1-C3): CSV y artefactos de los modelos
├── resultados/    # buscador (10-12): CSV
└── app_buscador/  # datos de la app (scripts/exportar_app.py, desde el notebook 13)
```

Todo tiene que estar en **una sola carpeta** `proyecto_multimodal`: si la corren dos personas, una la comparte
y la otra agrega un acceso directo a ella en su `Mi unidad`. El notebook 12, por ejemplo, usa a la vez archivos
del 07 (TMDB) y del 11 (adaptadores).

1. Abrir cada notebook en Colab (botón *Open in Colab*) y correrlo **en orden** desde el 01.
2. Usar GPU (T4) para el 03, 07, 10, 12 y C3. El resto funciona en CPU (el C2 tarda unos 10 minutos).
3. El notebook 07 necesita una llave gratuita de la API de TMDB.
