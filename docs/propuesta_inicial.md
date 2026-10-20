# Propuesta de proyecto — Diagnóstico y corrección del desbalance modal en modelos multimodales

> **Para qué sirve este documento:** explicarte el proyecto desde cero, sin asumir que ya conocés la literatura multimodal ni la jerga estadística. La sección 1-3 son la idea; la sección 7 es el glosario con TODOS los términos técnicos. Si tenés poco tiempo ahora, leé 1, 2 y 7.

---

## 1. Resumen en 5 líneas

- **Problema:** en modelos que combinan imagen + texto, una modalidad termina dominando el entrenamiento y la otra se aprende a medias. El modelo multimodal rinde igual (o peor) que el mejor modelo de una sola modalidad.
- **Idea:** medir estadísticamente dónde y cuándo se pierde esa información, y usar esa medición para construir una fusión que la recupere.
- **Reparto:** yo llevo la formalización matemática, las métricas de diagnóstico y el análisis. Vos llevás el pipeline de datos, los entrenamientos y la adaptación de baselines.
- **Tiempo:** ~8 h/semana cada uno, ~8 semanas.
- **Seguro:** el plan está partido en 4 entregables escalonados (A→D); si se acaba el tiempo, cada nivel cerrado se defiende por sí solo.

---

## 2. El problema, contado simple

### El fenómeno: "greedy modality"

Entrenamos un modelo que recibe **imagen + texto** y predice algo (un género de película, si un meme es odioso, etc.). La intuición dice que dos fuentes de información deberían dar mejor que una. En la práctica, muchas veces no:

> **Ejemplo concreto (Hateful Memes):** un meme tiene una foto inofensiva y un texto inofensivo, pero juntos son odiosos. Un modelo que solo mira la foto (o solo el texto) no puede resolverlo. Sin embargo, en la competencia de Facebook de este dataset, los modelos multimodales de los ganadores apenas igualaron a los mejores modelos de una sola modalidad. Toda la información extra estaba ahí, y el modelo no la usó.

**¿Por qué pasa?** Durante el entrenamiento, una de las dos ramas (la más "fácil" o con más señal) baja el error más rápido. El optimizador descubre que ajustando esa rama ya mejora mucho, y deja de invertir en la otra. Es como estudiar con dos libros donde uno es mucho más fácil: dejás de abrir el difícil porque con el fácil ya pasás el examen. El resultado: la rama débil queda subdesarrollada y la fusión de las dos no aporta casi nada sobre la fuerte sola.

A esto se le llama **greedy modality** (modalidad glotona/avariciosa) y está documentado formalmente en:

- Peng et al., CVPR 2022 — caracterización del fenómeno.
- Peng et al., CVPR 2022 (OGM-GE) — un intento de corrección (ver glosario).

### El costo del problema

Si la segunda modalidad no aporta, entonces **recolectarla, limpiarla y procesarla fue un gasto inútil**. Para la industria esto es relevante: se pagan pipelines multimodales que rinden como unimodales.

---

## 3. La idea del proyecto (la apuesta)

### Pregunta de investigación

> ¿Podemos **cuantificar estadísticamente** en qué momento del entrenamiento y en qué parte del espacio de representaciones una modalidad suprime a la otra, y usar esa cuantificación para diseñar una fusión que supere a las correcciones heurísticas del estado del arte (OGM-GE)?

### Hipótesis

> Si cada modalidad estima **su propia incertidumbre de forma calibrada**, una fusión que pondera cada modalidad inversamente a su incertidumbre recupera parte de la información de la modalidad débil que la fusión ingenua desperdicia. El efecto debería ser mayor cuando hay pocos datos.

### Las dos contribuciones (y su orden de importancia)

Esto es clave para entender el plan:

1. **Primaria — el diagnóstico:** un conjunto de mediciones estadísticas (con intervalos de confianza) que localizan la pérdida de información. Nadie más reporta esto. **Aunque el método no funcione, esto solo ya es resultado.**
2. **Secundaria — el método:** una fusión ponderada por incertidumbre calibrada, que compite contra OGM-GE.

La lógica que une las dos: el diagnóstico muestra el **mecanismo** del problema (la modalidad fuerte está *sobreconfiada*, entonces la fusión ingenua le da demasiado peso y el de la débil da igual). El método ataca exactamente ese mecanismo. No son dos proyectos engrapados: el segundo existe porque el primero señala dónde intervenir.

---

## 4. Cómo funciona el pipeline técnico

La arquitectura ya está decidida. De la entrada a la salida:

1. **Encoder de imagen:** CLIP (ViT-B/32), preentrenado, **congelado**. Convierte cada imagen en un vector de 512 números.
2. **Encoder de texto:** DistilBERT, preentrenado, **congelado**. Convierte cada texto en un vector de 768 números (lo proyectamos a 512).
3. **Una cabeza por modalidad:** una red chica (MLP de 2 capas) sobre cada vector. Cada cabeza produce: (a) la predicción de clasificación, y (b) una medida de su incertidumbre.
4. **Fusión:** se combinan las salidas de las dos cabezas, ponderando cada una según su incertidumbre (la versión simple pondera logits; la elegante usa teoría de evidencia — solo si hay tiempo).
5. **Pérdida:** entropía cruzada + un regularizador de calibración.

### ¿Por qué encoders congelados? (importante para el plan)

Congelados = **no se entrenan**, se usan tal cual vienen preentrenados. Solo se entrenan las cabezas (capas chicas). Esto es lo que hace viable el proyecto:

- Entrenar una cabeza sobre vectores precalculados toma **minutos**, no días.
- No necesitamos GPU potente: alcanza la T4 gratuita de Google Colab.
- Truco clave: **precalculamos todos los vectores UNA vez** al inicio y los guardamos a disco. Después, todos los experimentos corren sobre esos vectores, nunca sobre imágenes/textos crudos.

### Datos

| Dataset | Qué es | Tamaño | Tarea | Rol |
|---|---|---|---|---|
| **MM-IMDB** | Póster de película + sinopsis | ~26K | Géneros (multietiqueta, F1 macro) | **Principal** |
| Hateful Memes | Meme + texto | ~10K | Odioso / no odioso (binaria) | Solo si sobra tiempo |
| Food-101 | Foto + texto de web | 101 clases | Clasificación | Fuera del alcance |

---

## 5. Contra quién comparamos

| # | Método | Qué es | Estado |
|---|---|---|---|
| 1 | Unimodal imagen | Solo la rama de imagen | Referencia |
| 2 | Unimodal texto | Solo la rama de texto | Referencia |
| 3 | Fusión ingenua | Concatenar los dos vectores + MLP | Debe mostrar el problema: rendir ≈ mejor unimodal |
| 4 | **OGM-GE** | Estado del arte 2022: modula gradientes durante el entrenamiento | Baseline fuerte (adaptar código público) |
| 5 | PMR | Otra corrección con prototipos | Solo si sobra tiempo |
| 6 | **Nuestro método** | Fusión ponderada por incertidumbre | Lo que proponemos |

**Regla de rigor del proyecto:** todo número que entre al paper se reporta como media ± intervalo de confianza sobre al menos 3 semillas (5 si el tiempo da). Sin eso, un resultado no existe: podría ser suerte.

---

## 6. El plan: 4 entregables escalonados

Cada nivel cierra con una tabla o figura + un párrafo de hallazgo. Si se acaba el tiempo, entregamos hasta el último nivel completo y el argumento se sostiene solo.

| Tier | Qué entrega | Qué afirmamos | Cuándo |
|---|---|---|---|
| **A** | Tabla: unimodales vs fusión ingenua en MM-IMDB | "Caracterizamos el greedy modality en el régimen de encoders congelados, que la literatura no aisló" | Semana 3 |
| **B** | Panel de diagnóstico: ECE por modalidad, brecha modal con IC bootstrap, cota de información, curvas de entrenamiento | "Cuantificamos cuándo y dónde se suprime la modalidad débil, y cuánta información queda sin usar" | Semana 4 |
| **C** | Tabla comparativa: ingenua vs OGM-GE vs nuestro método | "Ponderar por incertidumbre calibrada recupera parte del gap sin modular gradientes" | Semana 6 |
| **D** | Dial de desbalance (F1 vs nivel de degradación) + curva de pocos datos + semillas completas + análisis de fallos | "Nuestro método degrada suave; la ingenua colapsa. Ventaja en pocos datos" | Semana 7 |

**El dial de desbalance** (nuestro experimento estrella): degradamos artificialmente una modalidad (ruido sobre sus vectores) en niveles crecientes y graficamos cómo responde cada método. La fusión ingenua debería colapsar al quedar ciega; la nuestra debería degradar suavemente porque "sabe" cuánto confiar en cada rama. Es la figura que cuenta toda la historia en una imagen.

### Punto de decisión en la semana 3

El fenómeno greedy podría ser más débil con encoders congelados (la literatura lo estudió entrenando los encoders). Si en MM-IMDB la fusión ingenua no muestra el problema, **no cambiamos de dataset**: lo inducimos con el dial de degradación sobre el mismo pipeline. Cambiar de dataset costaría una semana que no tenemos.

### Qué pasa si el método no le gana a OGM-GE

Nada fatal: el diagnóstico (Tier B) sigue siendo la contribución primaria y el paper se enmarca como "entendemos y diagnosticamos el fenómeno; proponemos una corrección simple basada en calibración". Perder la tabla comparativa no hunde el proyecto; por eso los tiers están en este orden.

---

## 7. Quién hace qué

### Vos — ejecución e infraestructura (el "cómo")

- Semana 1-2: pipeline de datos de MM-IMDB + precalcular los vectores de imagen y texto, guardarlos a disco.
- Entrenamientos de cabezas: unimodales, fusión ingenua (Tiers A-B).
- Adaptar el código público de OGM-GE a nuestro setup (Tier C).
- Ejecutar los experimentos del dial y las semillas (Tier D).
- Repo, reproducibilidad (seeds, configs YAML, un script por experimento).

### Yo — diseño y análisis (el "qué" y el "por qué")

- Formalización del problema (notación, hipótesis formal, 2 páginas).
- El **paquete de métricas de diagnóstico** (ECE, brecha modal, cota de información): es código puro sobre los vectores guardados, sin GPU — lo escribo yo.
- **Specs de experimento:** antes de que corras algo, te paso media página con la configuración, qué esperar y cómo detectar que algo salió mal.
- Análisis de resultados, tablas, figuras del paper.
- Secciones teóricas y de método del paper.

### Las reglas para que esto no colapse

1. **Interfaz fija (semana 1):** acordamos el formato exacto de los vectores guardados (`.npy`) + labels + configs YAML. Vos producís eso, yo consumo eso.
2. **Nada se entrena sin spec previo.** No para controlarte: para que ningún experimento se corre mal configurado y nos demos cuenta 3 semanas después.
3. **Ningún número entra al paper sin pasar por análisis mío.** Es la garantía de rigor estadístico, que es nuestro diferenciador contra proyectos de posgrado.
4. **Sincronización corta y frecuente:** ~2 h semanales mías dedicadas a revisar tus corridas y prepararte el siguiente spec. Nunca estás bloqueada esperando definiciones.

### Por qué este reparto te conviene

- Todo lo que se entrena corre en minutos sobre vectores precalculados: mucha iteración, poca espera.
- Tu parte (pipeline, entrenamiento, OGM-GE) es exactamente la experiencia de ingeniería DL que un paper conjunto te deja ganada.
- Los specs te llegan con criterios de éxito claros: no tenés que adivinar qué significa "bien".

---

## 8. Cronograma (8 semanas, ~8 h/semana c/u)

| Semana | Vos | Yo | Hito |
|---|---|---|---|
| 1 | Datos MM-IMDB + precalcular vectores | Formalización + esqueleto de métricas + formato de interfaz | Interfaz acordada |
| 2 | Unimodales + fusión ingenua (3 semillas) | Métricas v1 (ECE, brecha) + esqueleto del paper | — |
| 3 | Adaptar OGM-GE | Panel de diagnóstico sobre los baselines | **Tier A + decisión fenómeno** |
| 4 | Cerrar panel; preparar runs del método | Cerrar figura central + spec del método | **Tier B** |
| 5 | Método v0 (ponderación por entropía) + comparación | Análisis de tabla preliminar | Decisión: ¿método v0 funciona? |
| 6 | Método v1 (si hay tiempo) + dial | Análisis | **Tier C** |
| 7 | Dial completo + curva pocos datos + semillas extra | Análisis de fallos + freeze de código | **Tier D** |
| 8 | Detalles de reproducibilidad | Ensamblar paper + presentación | Entrega |

El paper se escribe **en paralelo desde la semana 2** (la introducción y el trabajo relacionado no dependen de los resultados). La semana 8 es ensamblar, no escribir.

---

## 9. Glosario — todos los términos técnicos

**Embedding (vector de características).** Representar un dato (imagen, texto) como una lista de números. CLIP convierte una imagen en 512 números; DistilBERT convierte un texto en 768. Dos datos semánticamente parecidos quedan con vectores parecidos. Todo nuestro entrenamiento ocurre sobre estos vectores, no sobre datos crudos.

**Encoder congelado.** Modelo preentrenado que usamos tal cual, sin entrenar sus pesos. "Congelado" = sus parámetros no cambian durante el entrenamiento. Solo entrenamos las capas nuevas que le ponemos encima.

**Cabeza (head) / MLP.** Red neuronal chica (en nuestro caso, 2 capas) que se conecta al final de un encoder congelado y aprende la tarea específica. Es lo único que entrenamos.

**Fusión.** El paso donde se combinan las salidas de las dos ramas (imagen y texto) para producir una sola predicción. "Fusión ingenua" = concatenar los dos vectores y meterlos en un MLP, sin ningún mecanismo de balanceo.

**Greedy modality.** El fenómeno del problema: durante el entrenamiento, la modalidad más fácil domina el aprendizaje y la otra queda subdesarrollada. Ver sección 2.

**CLIP (ViT-B/32).** Modelo de OpenAI preentrenado con imágenes + textos (aprendió a relacionarlos). ViT-B/32 es la versión mediana basada en Vision Transformer. Nos sirve porque ya "entiende" imágenes genéricas sin entrenamiento adicional.

**DistilBERT.** Versión reducida y rápida de BERT (el modelo de lenguaje clásico). Preentrenada para entender texto. Salida de 768 dimensiones.

**MM-IMDB.** Dataset público de pósters de películas + sinopsis, con etiquetas de géneros. "Multietiqueta" = una película puede ser Drama Y Romance a la vez (no es elegir una sola categoría).

**F1 macro.** Métrica de clasificación que combina precisión y cobertura. "Macro" = promediar la métrica por clase sin ponderar por frecuencia, así los géneros raros valen lo mismo que los comunes. Es la métrica estándar en MM-IMDB.

**OGM-GE.** (*On-the-fly Gradient Modulation — Generalization Enhancement*, CVPR 2022.) El estado del arte que vamos a superar. Idea: durante el entrenamiento, mira los gradientes que cada modalidad le manda al modelo; si una domina, amplifica el gradiente de la débil y frena el de la fuerte. Es una corrección heurística de ingeniería: funciona, pero sin una explicación estadística de cuánto ni por qué. Esa brecha es nuestro espacio.

**PMR.** (*Prototypical Modal Rebalance.*) Otra corrección del mismo problema usando prototipos por clase. Solo entra si sobra tiempo (es menos prioritaria que OGM-GE y que nuestro método).

**ECE (Expected Calibration Error).** Mide cuánto miente la *confianza* de un modelo. Ejemplo: juntá todas las predicciones donde el modelo dijo "estoy 90% seguro". Si el 90% de esas eran correctas, está calibrado. Si solo el 70% lo era, está sobreconfiado (ECE ≈ 0.20). Nos importa porque: la modalidad fuerte tiende a estar **sobreconfiada** y la débil **subconfiada**; si la fusión no corrige eso, hereda el defecto. Nuestra fusión pondera cada rama según qué tan confiable es su confianza.

**Calibración.** Que la confianza declarada coincida con la tasa real de acierto. Se visualiza con *reliability diagrams* (curva confianza vs. acierto real).

**Incertidumbre / evidential deep learning (Dirichlet).** Maneras de que una red diga "no sé" con números. La versión elegante (Dirichlet): la red produce "evidencia" para cada clase y de ahí sale una distribución de probabilidad *más* una medida de incertidumbre. Es el plan B del proyecto (inestable a veces). El plan A es más simple: usar la **entropía del softmax** (cuán repartida está la probabilidad entre clases) como medida de incertidumbre. La hipótesis es la misma con cualquiera de las dos.

**Información mutua condicional — I(X_débil; Y | X_fuerte).** El corazón teórico. Se lee: "cuánta información extra sobre la etiqueta Y aporta la modalidad débil, si ya tengo la fuerte". Si es > 0, la modalidad débil SIRVE para algo y descartarla es perder información medible. Es el techo teórico: ningún modelo puede extraer más que eso.

**InfoNCE / cota.** La información mutua real es imposible de calcular exactamente en alta dimensión. InfoNCE es una fórmula (la misma forma de la pérdida contrastiva de CLIP) que da una **cota inferior**: un número que es *como mucho* el valor real. Reportamos cotas, no valores puntuales, y eso es metodológicamente honesto.

**IC bootstrap (intervalo de confianza).** Forma de calcular un margen de error sin suponer distribuciones: re-muestrear los resultados con reemplazo muchas veces, recalcular la métrica en cada re-muestreo, y ver el rango donde cae el 95% de ellos. Reportamos "F1 = 58.3 ± 1.2" en vez de un número pelado.

**Semilla (seed).** El número que fija el azar (inicialización de pesos, orden de los datos). Correr con 3-5 semillas distintas y promediar demuestra que el resultado no fue suerte.

**Dial de desbalance.** Nuestro experimento estrella: degradar una modalidad artificialmente (ruido creciente sobre sus vectores) para dibujar la curva "rendimiento vs. nivel de degradación". Muestra de un vistazo qué método aguanta mejor la pérdida de una rama. Ver sección 6.

**Hateful Memes.** Dataset de Facebook: memes etiquetados como odiosos/no odiosos. Está diseñado para que la imagen sola o el texto solo sean ambiguos: la ofensa vive en la combinación. Es el caso extremo del problema; solo entra si sobra tiempo.

---

## 10. Decisiones ya tomadas (no negociables) vs. abiertas

**Ya tomadas** (justificadas por tiempo, hardware y el ángulo estadístico):
- Encoders congelados; solo se entrenan cabezas.
- MM-IMDB como dataset principal; Food-101 fuera; Hateful Memes solo como bonus.
- Mínimo 3 semillas + IC para todo lo que entre al paper.
- El diagnóstico como contribución primaria, el método como secundaria.

**Abiertas — tu opinión vale y se decide en conjunto:**
- Detalles del formato de la interfaz (sem. 1).
- Framework: PyTorch (mi propuesta, por el código público de OGM-GE) — discutible si tenés más fluidez en otra cosa.
- Organización del repo y del registro de experimentos.
- Cómo repartimos la escritura de las secciones de implementación del paper.

---

*Documento vivo: cualquier corrección o desacuerdo, se discute y se edita. La meta es que las dos firmemos algo que nos enorguezca.*
