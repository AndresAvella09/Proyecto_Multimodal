# BRIEF DE PROYECTO — Marco estadístico para diagnosticar y corregir el desbalance modal en aprendizaje multimodal

## 1. CONTEXTO Y RESTRICCIONES

Soy estudiante de pregrado en Ingeniería Estadística, cursando una materia demaestría en Deep Learning (Deep Representation Learning para Ciencia de Datos).El proyecto es en equipo de 2 personas y compite contra proyectos de posgradode toda la facultad, así que el estándar de rigor debe ser de paper deconferencia, no de tarea de curso.

RESTRICCIONES DE HARDWARE (crítico):

- No tenemos GPUs potentes. Contamos con Google Colab (T4 gratuita o Pro),o en su defecto una GPU de laptop de 6-8 GB.
- TODO el diseño ya está optimizado para esto: encoders congelados y soloentrenamos cabezas pequeñas. No proponer entrenar modelos grandes.

RESTRICCIÓN DE PERFIL:

- Mi fortaleza es estadística (inferencia, calibración, teoría de lainformación). Mi debilidad relativa es ingeniería de DL a gran escala.
- El diferenciador del proyecto es el RIGOR ESTADÍSTICO aplicado a un problemaque la literatura trata con heurísticas de ingeniería. Cualquier sugerenciaque diluya ese ángulo estadístico va en contra del proyecto.

## 2. PROBLEMA Y PREGUNTA DE INVESTIGACIÓN

El fenómeno del "greedy modality": en modelos multimodales (imagen+texto),la modalidad dominante suprime el aprendizaje de la débil durante elentrenamiento. Resultado: el modelo multimodal rinde igual o peor que elmejor modelo unimodal, desperdiciando el costo de recolectar y procesar lasegunda modalidad.

Esto está documentado en:

- Peng et al. (CVPR 2022), "Characterizing and Overcoming the Greedy Natureof Learning in Multimodal Models"
- Peng et al. (CVPR 2022), "Balanced Multimodal Learning via On-the-flyGradient Modulation" (OGM-GE)
- El reto Hateful Memes de Facebook: los ganadores apenas empataron a losmodelos unimodales.

PREGUNTA DE INVESTIGACIÓN:¿Se puede cuantificar estadísticamente el punto del entrenamiento y del espacio de representaciones donde una modalidad suprime a la otra, y usaresa cuantificación para diseñar una fusión que supere a los métodosheurísticos del estado del arte (OGM-GE, PMR)?

HIPÓTESIS\:Si cada modalidad estima su propia incertidumbre de forma calibrada, unafusión ponderada por esa incertidumbre (en vez de gradientes modulados adhoc) recupera parte de la información de la modalidad débil que la fusióningenua desperdicia, y el efecto es mayor en regímenes de pocos datos.

## 3. MARCO TEÓRICO — LAS TRES LECTURAS ESTADÍSTICAS

Este es el ALMA del proyecto. Ningún paper del área combina las tres.

LECTURA A — Descomposición del error de Bayes\:El clasificador óptimo h\*(x\_img, x\_txt) supera al unimodal fuerte si y solo si la información mutua condicional I(X\_débil; Y | X\_fuerte) > 0. Vamos aESTIMAR esa cantidad (con cotas InfoNCE o MINE, no estimadores exactos) yusarla como referencia teórica: "el gap entre lo logrado y la cota es elcosto del desbalance".

LECTURA B — La brecha modal como fenómeno geométrico-estadístico\:Liang et al. (NeurIPS 2022), "Understanding the Modality Gap in Multi-modalContrastive Representation Learning": las representaciones de dosmodalidades viven en conos disjuntos del espacio de embeddings. Nosotroscuantificamos esa brecha con estadística formal: distancia media entreembeddings con IC bootstrap, razón de varianza inter-clase vs intra-clasepor modalidad, espectro de autovalores de la covarianza, y su evolucióndurante el entrenamiento.

LECTURA C — Calibración por modalidad (la punta de lanza)\:La modalidad fuerte es sobreconfiada, la débil subconfiada, la fusióningenua hereda el defecto. Solución: cada rama produce una distribución deincertidumbre (evidential deep learning con parámetros Dirichlet — Sensoyet al., NeurIPS 2018), y la fusión pondera cada modalidad inversamente asu incertidumbre estimada. Esto es calibrable y verificable con curvas deconfiabilidad (reliability diagrams) y ECE por modalidad.

## 4. ARQUITECTURA TÉCNICA (ya decidida, no cambiar sin justificación fuerte)

Pipeline:

1. ENCODER IMAGEN: CLIP ViT-B/32, CONGELADO, salida 512-d.
2. ENCODER TEXTO: DistilBERT, CONGELADO, salida 768-d (proyectar a 512-d).
3. CABEZA POR MODALIDAD: MLP de 2 capas que produce\:a) logits de clasificaciónb) parámetros de concentración Dirichlet (evidential DL)
4. FUSIÓN: combinación ponderada de las dos distribuciones, pesos derivadosde la incertidumbre (inversamente proporcionales a la confianza total).Versión simple: promedio ponderado de logits. Versión elegante: fusiónde Dempster-Shafer (extensión solo si hay tiempo).
5. PÉRDIDA: entropía cruzada + regularizador de calibración + opcionalmenteun término contrastivo prototípico estilo PMR para acelerar la rama lenta.

CONGELAR ENCODERS NO ES NEGOCIABLE: es lo que hace el proyecto viable ennuestro hardware. Todo lo que se entrena son las cabezas y la fusión.

## 5. DATASETS (los tres benchmarks estándar del área)

1. MM-IMDB (\~26K muestras): póster (imagen) + sinopsis (texto),clasificación multietiqueta de géneros. Métrica: F1 macro.ES EL DATASET PRINCIPAL. Empezar aquí.
2. Hateful Memes (\~10K): meme (imagen) + texto, clasificación binaria.Es donde el desbalance es más severo y documentado.
3. UPMC Food-101: SUBMUESTREAR a \~30K. Foto + página web, 101 clases.

PREPROCESAMIENTO CRÍTICO: precomputar todos los embeddings con los encoderscongelados UNA VEZ y guardarlos a disco (numpy o similar). Después, todo elentrenamiento de cabezas trabaja sobre embeddings, no sobre datos crudos.Esto reduce el cómputo por experimento de días a horas en una T4.

## 6. BASELINES Y MÉTODOS A COMPARAR (la tabla que hay que llenar)

1. Unimodal imagen (referencia inferior/superior)
2. Unimodal texto (referencia)
3. Fusión ingenua (concatenación de embeddings + MLP) — debe mostrar elfenómeno greedy modality: rendir ≈ igual que el mejor unimodal
4. OGM-GE (CVPR 2022) — reimplementar o adaptar código público
5. PMR (Prototypical Modal Rebalance, CVPR) — reimplementar
6. NUESTRO MÉTODO — fusión ponderada por incertidumbre calibrada

TODO resultado reportado como media ± IC 95% sobre mínimo 5 semillas.Sin eso, el resultado no existe para efectos del paper.

## 7. MÉTRICAS

PRIMARIAS: F1 macro (MM-IMDB), accuracy (Food-101), F1 (Hateful Memes).DIAGNÓSTICAS (nuestras, nadie más las reporta):

- ECE (Expected Calibration Error) por modalidad y para la fusión
- Distancia de brecha modal con IC bootstrap
- Estimación de I(X\_débil; Y | X\_fuerte) con cotas InfoNCE
- Curvas de convergencia por modalidad durante entrenamiento

## 8. EXPERIMENTOS CLAVE (en orden de prioridad)

EXP 1 (semanas 1-2) — Demostrar el fenómeno\:Entrenar unimodales y fusión ingenua en MM-IMDB. Verificar quefusión ≈ mejor unimodal. Si no aparece el fenómeno en MM-IMDB,probar Hateful Memes ANTES de seguir.

EXP 2 (semanas 3-4) — El diagnóstico\:Implementar el paquete de medidas (MI condicional, estadísticas debrecha, ECE) y correrlo sobre los baselines. Entregable: panel quemuestra cuándo en el entrenamiento y dónde en el espacio ocurre lasupresión.

EXP 3 (semanas 5-6) — El método\:Entrenar la fusión ponderada por incertidumbre. Comparar contraOGM-GE y PMR. Entregable: tabla comparativa preliminar.

EXP 4 (semana 7) — El dial de desbalance (el experimento que ganacompetencias)\:Degradar artificialmente una modalidad (ruido gaussiano sobre losembeddings de imagen O reducir la fracción de datos de entrenamientode texto). Graficar F1 vs. nivel de degradación para fusión ingenua,OGM-GE y nuestro método. Nuestra curva debe degradar suavemente;la ingenua debe colapsar al dominante.Incluir TAMBIÉN la curva: F1 vs. tamaño del conjunto de entrenamiento(ventaja esperada en el régimen de pocos datos).

EXP 5 (semana 8) — Rigor\:Todo repetido con 5 semillas. Análisis de fallos: ¿en qué clasesfalla nuestra fusión y por qué?

## 9. CONVENCIONES DE CÓDIGO (no negociables para reproducibilidad)

- Python, PyTorch (o TensorFlow si es más natural para el agente, peroser consistentes).
- Seeds fijadas en TODOS los experimentos (numpy, torch, random).
- Configuraciones de cada experimento en archivos YAML, no hardcodeadas.
- Un script por experimento, ejecutable con un solo comando.
- Estructura sugerida:/src (código), /configs (YAMLs), /scripts (una por experimento),/results (salidas: métricas en CSV, figuras en PNG),/notebooks (solo análisis exploratorio y visualización, NO entrenamiento),README.md con instrucciones exactas de reproducción.

## 10. CRONOGRAMA (9-10 semanas)

- S1-2: Pipeline + EXP 1 (fenómeno)
- S3-4: Diagnóstico (EXP 2)
- S5-6: Método + comparación (EXP 3)
- S7: Ablaciones + dial de desbalance (EXP 4)
- S8: Semillas, CIs, análisis de fallos (EXP 5)
- S9-10: Paper (estructura NeurIPS: abstract, intro, related work,method, experiments, conclusion), dashboard en Gradio, presentación.

DIVISIÓN DEL TRABAJO (equipo de 2):

- Yo (perfil estadístico): diagnóstico, MI, calibración, análisis,marco teórico del paper.
- Compañero (perfil implementación): pipeline, entrenamientos,OGM-GE y PMR, dashboard, repo.

## 11. RIESGOS YA IDENTIFICADOS Y MITIGACIONES

1. "No superamos a OGM-GE en F1" → El diagnóstico es contribución porsí solo. Enmarcar como "entendemos + corregimos", no "batimos SOTA".
2. "MI condicional difícil de estimar" → Usar cotas InfoNCE (que esliteralmente la pérdida contrastiva), reportar cotas no estimacionespuntuales.
3. "Evidential DL inestable" → Empezar con proxy simple (entropía delsoftmax como incertidumbre, ponderación inversa). Dirichlet solo siconverge.
4. "Falta cómputo" → Encoders congelados, MM-IMDB + Hateful Memescomo mínimo viable, Food-101 como trabajo futuro.

## 12. QUÉ ESPERO DEL AGENTE

Tu rol es ayudarme a implementar, depurar y mejorar este proyecto bajoestas restricciones. Específicamente:

- Cuando escribas código: seeds fijadas, reproducible, comentado enespañol o inglés pero consistente, siguiendo la estructura de /src.
- Cuando propongas alternativas: justificar en términos estadísticos(estimador, varianza, garantía), no solo de conveniencia ingenieril.
- Cuando detectes un error metodológico mío (fuga de datos, comparacióninjusta, métrica mal calculada): señalarlo inmediatamente. Un errormetodológico no detectado nos cuesta la competición.
- NO proponer entrenar modelos grandes, cambiar los datasets principales,ni abandonar el ángulo estadístico. Esas decisiones ya están tomadasy justificadas.
- Si algo del brief es ambiguo, preguntar antes de asumir.