"""Buscador semántico: lo que hacía la app del notebook 13 (póster, sinopsis o ambos; sin entrenar vs.
adaptador; estrenos 2020-2026 con la corrección de escala por fuente del 12)."""
import html
import os

import altair as alt
import pandas as pd
import streamlit as st

import comun
from comun import ROJO

comun.configurar_pagina('Buscador · películas')
if st.session_state.pop('ir_a_clasificador', False):     # botón "¿Qué géneros le ve el modelo?" del diálogo
    st.switch_page('pages/2_Clasificador.py')

cat, CFG = comun.cargar_catalogo(), comun.cargar_config()
ESTRENO = comun.estreno()
comun.cargar_vectores()
comun.cargar_codificadores(CFG['modelo_poster'], CFG['modelo_sinopsis'])


@st.dialog('Sinopsis', width='large')
def ver_sinopsis(i):
    r = cat.iloc[i]
    izq, der = st.columns([1, 3])
    with izq:
        st.image(comun.imagen(i), width=190)
    with der:
        st.markdown(f'### {html.escape(str(r["titulo"]))}')
        st.markdown(f'<div class="tarjeta-meta">{comun.anio_de(r)} · '
                    f'{html.escape((r["generos"] or "").replace(",", " · "))}</div>', unsafe_allow_html=True)
        st.write('')
        st.write(comun.sinopsis_de(i))
        if comun.existe('clasificador.npz') and st.button('¿Qué géneros le ve el modelo? →', key='a_clasif'):
            # El diálogo es un fragmento: su botón solo lo reejecuta a él. Se marca y se reejecuta toda la página,
            # que hace el cambio de página arriba.
            st.session_state['pelicula'] = i
            st.session_state['ir_a_clasificador'] = True
            st.rerun()
    if ESTRENO[i]:
        st.caption('Póster y sinopsis de TMDB. Este producto usa la API de TMDB pero no está respaldado ni certificado por TMDB.')
    else:
        st.caption('Las sinopsis de MM-IMDB vienen en minúsculas y con los números reemplazados por "N".')


comun.encabezado('Buscador de', 'películas')

tab_buscar, tab_arq, tab_res, tab_tec = st.tabs(['Buscador', 'Arquitectura', 'Resultados', 'Explicación técnica'])

# ============================================================================= BUSCADOR
with tab_buscar:
    EJEMPLOS = ['una casa embrujada en la nieve', 'a robot in space', 'princesas',
                'un detective investiga un asesinato', 'dibujos animados de colores', 'piratas en el mar']
    if 'consulta' not in st.session_state:
        st.session_state.consulta = ''

    def usar_ejemplo(texto):
        st.session_state.consulta = texto

    st.text_input('¿Qué película buscas? Escribe en español o en inglés', key='consulta',
                  placeholder='Ej.: una casa embrujada en la nieve')
    st.caption('O prueba con:')
    for fila in range(0, len(EJEMPLOS), 3):
        for c, ej in zip(st.columns(3), EJEMPLOS[fila:fila + 3]):
            c.button(ej, on_click=usar_ejemplo, args=(ej,), width='stretch', key=f'ej_{ej}')

    with st.expander('Opciones de búsqueda', expanded=False):
        o1, o2, o3 = st.columns([1.1, 1.2, 1.1])
        with o1:
            modo = st.radio('Buscar por', ['Ambos', 'Póster', 'Sinopsis'], horizontal=True)
            st.markdown('<div class="opcion-nota"><b>Póster:</b> lo que se ve. <b>Sinopsis:</b> lo que se cuenta. '
                        '<b>Ambos:</b> combina los dos; fue la forma que mejor buscó.</div>', unsafe_allow_html=True)
        with o2:
            alfa = st.slider('Peso del póster (modo "Ambos")', 0.0, 1.0, float(CFG['alfa']), 0.1,
                             disabled=(modo != 'Ambos'))
            st.markdown(f'<div class="opcion-nota">0 = solo sinopsis · 1 = solo póster. Viene en '
                        f'<b>{CFG["alfa"]:.1f}</b> porque fue el peso que <b>mostró los mejores resultados</b> '
                        f'en la evaluación (mAP 0.517 en películas de prueba, notebook 11).</div>',
                        unsafe_allow_html=True)
        with o3:
            version = st.radio('Representación', ['Entrenada', 'Sin entrenar'], horizontal=True)
            st.markdown('<div class="opcion-nota"><b>Entrenada:</b> vectores ajustados con nuestro adaptador. '
                        '<b>Sin entrenar:</b> los modelos tal cual. Cámbiala para ver el antes y el después.</div>',
                        unsafe_allow_html=True)
        o4, o5, o6 = st.columns([2, 1, 1])
        with o4:
            desde, hasta = st.slider('Años de estreno', CFG['anio_min'], CFG['anio_max'],
                                     (CFG['anio_min'], CFG['anio_max']))
        with o5:
            k = st.select_slider('Número de resultados', [6, 12, 18, 24], value=12)
        with o6:
            con_estrenos = st.toggle('Incluir estrenos 2020-2026', value=True, disabled=not ESTRENO.any())
            st.markdown('<div class="opcion-nota">Películas nuevas de TMDB que el modelo <b>nunca vio</b> al '
                        'entrenarse.</div>', unsafe_allow_html=True)

    consulta = st.session_state.consulta.strip()
    if not consulta:
        st.markdown(comun.ficha('', 'Escribe una descripción o elige un ejemplo para empezar.'), unsafe_allow_html=True)
    else:
        with st.spinner('Buscando...'):
            top, por_poster = comun.buscar(consulta, modo, version, alfa, desde, hasta, k, con_estrenos)
        if not top:
            st.warning(f'No hay películas entre {desde} y {hasta}.')
        else:
            resumen = f'**{len(top)} películas** para *"{html.escape(consulta)}"*'
            if modo == 'Ambos':
                n_p = sum(por_poster.values())
                resumen += (f' · encontradas más por el póster: **{n_p}** · '
                            f'más por la sinopsis: **{len(top) - n_p}**')
            st.markdown(resumen)
            st.caption('Haz clic en "Ver sinopsis" debajo de cualquier película para leer de qué trata.'
                       + (' Las marcadas como ESTRENO vienen de TMDB.' if ESTRENO[top].any() else ''))
            COLUMNAS = 6
            for fila in range(0, len(top), COLUMNAS):
                for c, i in zip(st.columns(COLUMNAS), top[fila:fila + COLUMNAS]):
                    r = cat.iloc[i]
                    generos = (r['generos'] or '').replace(',', ' · ')
                    with c:
                        st.image(comun.imagen(i), width='stretch')
                        tarjeta = (f'<div class="tarjeta-titulo">{html.escape(str(r["titulo"]))}</div>'
                                   f'<div class="tarjeta-meta">{comun.anio_de(r)} · {html.escape(generos)}</div>')
                        if modo == 'Ambos':
                            tarjeta += ('<span class="etiqueta">por el póster</span>' if por_poster[i]
                                        else '<span class="etiqueta sinopsis">por la sinopsis</span>')
                        if ESTRENO[i]:
                            tarjeta += ' <span class="etiqueta estreno">estreno</span>'
                        st.markdown(tarjeta, unsafe_allow_html=True)
                        if st.button('Ver sinopsis', key=f'sin_{i}', width='stretch'):
                            ver_sinopsis(i)

# ============================================================================= ARQUITECTURA
with tab_arq:
    st.markdown('## Cómo funciona el buscador')
    st.write('La búsqueda se hace por **dos caminos en paralelo**, uno por cada modalidad. En cada camino, '
             'la frase del usuario y las películas se convierten en vectores dentro del **mismo espacio**, y se '
             'buscan los más cercanos. Al final, los dos puntajes se combinan.')

    def caja(nombre, detalle, roja=False):
        return (f'<div class="caja{" roja" if roja else ""}"><div class="nombre">{nombre}</div>'
                f'<div class="det">{detalle}</div></div>')

    FL = '<div class="flecha">➜</div>'
    st.markdown(
        '<div class="carril">Camino 1 · el póster (lo que se ve)</div><div class="flujo">'
        + caja('Tu búsqueda', 'texto en español o inglés')
        + FL + caja('CLIP multilingüe', 'DistilBERT multilingüe (6 capas) → promedio → capa lineal 768→512')
        + FL + caja('Similitud coseno', 'contra el vector de cada póster')
        + '</div><div class="flujo">'
        + caja('Póster', 'imagen 224×224 dividida en parches de 32×32')
        + FL + caja('CLIP ViT-B/32', 'Vision Transformer de 12 capas → vector de 512 (congelado)')
        + FL + caja('Adaptador', 'capa lineal residual entrenada con los géneros', roja=True)
        + '</div>'
        '<div class="carril">Camino 2 · la sinopsis (lo que se cuenta)</div><div class="flujo">'
        + caja('Tu búsqueda', 'con el prefijo "query:"')
        + FL + caja('multilingual-e5-base', 'XLM-RoBERTa (12 capas) → promedio → vector de 768')
        + FL + caja('Similitud coseno', 'contra el vector de cada sinopsis')
        + '</div><div class="flujo">'
        + caja('Sinopsis', 'texto completo, hasta 512 tokens, prefijo "passage:"')
        + FL + caja('multilingual-e5-base', 'el mismo modelo, congelado → vector de 768')
        + FL + caja('Adaptador', 'capa lineal residual entrenada con los géneros', roja=True)
        + '</div>'
        '<div class="carril">Combinación</div><div class="flujo">'
        + caja('Estandarizar', 'cada puntaje se pasa a z-score dentro de su fuente (MM-IMDB o TMDB)')
        + FL + caja('Mezclar', f'α · póster + (1 − α) · sinopsis, con α = {CFG["alfa"]:.1f}', roja=True)
        + FL + caja('Ranking', 'las películas con mayor puntaje, filtradas por año')
        + '</div>', unsafe_allow_html=True)

    st.markdown('### Los modelos, uno por uno')
    fichas = [
        ('CLIP ViT-B/32 (OpenAI)',
         'Entrenado con 400 millones de pares imagen-texto para que una foto y su descripción queden cerca. '
         'Su parte de imagen es un <b>Vision Transformer</b>: corta el póster en 49 parches de 32×32 píxeles, '
         'los trata como "palabras" y los procesa con 12 capas de atención. Resultado: 512 números por póster.'),
        ('CLIP multilingüe (sentence-transformers)',
         'El CLIP original solo entiende inglés. Esta versión es un <b>DistilBERT multilingüe</b> entrenado por '
         '<b>destilación</b>: aprendió a imitar al codificador de texto de CLIP en más de 50 idiomas. Por eso '
         '"una película de terror" y "a horror movie" caen casi en el mismo punto (similitud 0.99).'),
        ('multilingual-e5-base (Microsoft)',
         'Un <b>XLM-RoBERTa</b> de 12 capas entrenado para <b>buscar documentos largos a partir de frases cortas</b> '
         'en ~100 idiomas. Lo usamos para la sinopsis porque el texto de CLIP solo sirve para frases cortas: con CLIP, '
         'todas las sinopsis "se parecían igual" a cualquier búsqueda.'),
        ('Adaptador residual (nuestro)',
         'Una capa lineal que <b>corrige un poco</b> el vector de cada película: v′ = normalizar(v + W·v). '
         'W empieza en cero (al inicio no cambia nada) y se entrena para que las películas queden cerca de las '
         'frases de sus géneros. Hay uno para el póster (512×512) y otro para la sinopsis (768×768). '
         'Las búsquedas no se modifican, por eso cualquier frase libre sigue funcionando.'),
        ('Clasificadores de género (notebooks 04 y 06)',
         'En la otra parte del proyecto, sobre los mismos vectores congelados (CLIP para el póster, DistilBERT para '
         'la sinopsis) se entrenan cabezas pequeñas que predicen los géneros: solo póster, solo sinopsis y la '
         'fusión de ambos. Sirven para medir de qué modalidad depende el modelo (páginas Clasificador y Desbalance).'),
    ]
    a1, a2 = st.columns(2)
    for j, (titulo, texto) in enumerate(fichas):
        (a1 if j % 2 == 0 else a2).markdown(comun.ficha(titulo, texto), unsafe_allow_html=True)
    st.caption('Todos los modelos preentrenados se usan congelados: solo se entrenan los adaptadores y las cabezas, '
               'por eso todo el proyecto corre en Google Colab.')

# ============================================================================= RESULTADOS
with tab_res:
    barras = comun.barras
    st.markdown('## Lo que encontramos')
    comun.kpis([('0.27 → 0.51', 'mAP del buscador antes y después de entrenar'),
                ('7.7 / 10', 'aciertos en los 10 primeros resultados (antes: 5.6)'),
                ('22 / 22', 'géneros en los que mejora la búsqueda'),
                ('63.6', 'F1 macro al clasificar géneros con póster + sinopsis')])

    st.markdown('### 1. El entrenamiento mejora la búsqueda')
    antes_despues = pd.DataFrame({
        'modo': ['Póster', 'Póster', 'Sinopsis', 'Sinopsis', 'Ambos', 'Ambos'],
        'etapa': ['Sin entrenar', 'Entrenada'] * 3,
        'mAP': [0.209, 0.374, 0.226, 0.452, 0.272, 0.513]})
    r1, r2 = st.columns([1.3, 1])
    r1.altair_chart(barras(antes_despues, 'modo', 'mAP', 'etapa', ['Sin entrenar', 'Entrenada'],
                           'mAP en películas de prueba (7 090 que el modelo nunca vio)'), width='stretch', theme=None)
    r2.markdown(comun.ficha('Qué significa',
                            'El <b>mAP</b> mide qué tan arriba quedan las películas correctas (1 = perfecto, azar ≈ 0.11). '
                            'Después de entrenar el adaptador, buscar con póster y sinopsis juntos pasa de '
                            '<b>0.27 a 0.51</b>. La mejora es real: su intervalo de confianza del 95 % va de +0.20 a '
                            '+0.28, lejos de cero, y es igual con 3 semillas distintas.'), unsafe_allow_html=True)

    st.markdown('### 2. Combinar póster y sinopsis es mejor que cualquiera por separado')
    modos = pd.DataFrame({
        'forma de buscar': ['Póster (CLIP)', 'Sinopsis (CLIP)', 'Sinopsis (E5)', 'Ambos (CLIP)', 'Ambos (E5)'],
        'mAP': [0.204, 0.121, 0.222, 0.188, 0.263], 'tipo': ['sin entrenar'] * 5})
    r1, r2 = st.columns([1.3, 1])
    r1.altair_chart(barras(modos, 'mAP', 'forma de buscar', 'tipo', ['sin entrenar'],
                           'mAP sin entrenar, según cómo se representa cada modalidad', horizontal=True, alto=230)
                    .encode(color=alt.value(ROJO)), width='stretch', theme=None)
    r2.markdown(comun.ficha('El desbalance depende de la representación',
                            'Cuando la sinopsis estaba mal representada (texto de CLIP), el mejor peso para el póster '
                            'era <b>0.9</b>: la combinación casi ignoraba la sinopsis. Con una buena representación (E5) '
                            'el mejor peso bajó a <b>0.5</b>, y después de entrenar a <b>0.4</b>: las dos modalidades '
                            'aportan. <b>El desbalance no es culpa de la modalidad, sino de cómo la representamos.</b>'),
                unsafe_allow_html=True)

    st.markdown('### 3. Dónde mejora más')
    genero = pd.DataFrame({
        'género': ['Animation', 'Documentary', 'Action', 'Western', 'Adventure', 'Horror', 'Family', 'Music',
                   'Drama', 'Sport', 'Comedy', 'Crime', 'Mystery', 'Biography', 'Romance', 'History', 'Sci-Fi',
                   'War', 'Thriller', 'Musical', 'Fantasy', 'Film-Noir'],
        'mejora en AP': [0.460, 0.435, 0.417, 0.320, 0.319, 0.315, 0.259, 0.256, 0.228, 0.224, 0.217, 0.216,
                         0.209, 0.207, 0.185, 0.185, 0.179, 0.160, 0.158, 0.120, 0.117, 0.112],
        'tipo': ['mejora'] * 22})
    r1, r2 = st.columns([1.3, 1])
    r1.altair_chart(barras(genero, 'mejora en AP', 'género', 'tipo', ['mejora'],
                           'Mejora por género al entrenar (póster + sinopsis)', horizontal=True, alto=470)
                    .encode(color=alt.value(ROJO)), width='stretch', theme=None)
    with r2:
        st.markdown(comun.ficha('Cada modalidad para lo suyo',
                                'Sin entrenar, el <b>póster</b> gana en los géneros que se <i>ven</i>: animación, '
                                'western, acción, terror, familia. La <b>sinopsis</b> gana en los que se <i>cuentan</i>: '
                                'documental (0.49 contra 0.12 del póster), música, biografía, guerra e historia.'),
                    unsafe_allow_html=True)
        st.markdown(comun.ficha('Idioma', 'El buscador funciona en los dos idiomas: mAP 0.283 en inglés y 0.244 en '
                                'español. Escribir una sola palabra ("terror") funciona casi igual que una frase '
                                'completa.'), unsafe_allow_html=True)
        st.markdown(comun.ficha('Clasificación de géneros', 'F1 macro en prueba, 5 semillas: solo póster <b>47.7</b>, '
                                'solo sinopsis <b>59.1</b>, fusión <b>63.6</b>. La imagen sí aporta (+4.5 puntos sobre '
                                'la sinopsis sola).'), unsafe_allow_html=True)

    st.markdown('### 4. El espacio de representaciones antes y después')
    if comun.existe('umap.png'):
        st.image(os.path.join(comun.carpeta_datos(), 'umap.png'), width='stretch')
    st.write('Cada punto es una película de prueba, coloreada por su género. **Antes** de entrenar los géneros '
             'están mezclados; **después** se agrupan. Con números: de los 10 vecinos más cercanos de cada película, '
             'la proporción que comparte algún género sube de 0.71 a 0.76 en el póster y de 0.76 a 0.83 en la '
             'sinopsis (el azar da 0.54).')
    st.page_link('pages/5_Espacio_3D.py', label='Verlo en 3D →')

    st.markdown('### 5. Películas de ayer y de hoy')
    TEMP = CFG.get('temporal', {})
    if not TEMP:
        st.markdown(comun.ficha('Sin datos', 'Faltan las cifras de los notebooks 08 y 12.'), unsafe_allow_html=True)
    else:
        st.write('Todo se entrenó con películas de MM-IMDB (hasta 2016) y se probó, **sin reentrenar**, con 3 000 '
                 'películas de 2020 a 2026 descargadas de TMDB. Para separar el efecto de la **época** del de la '
                 '**fuente** de los datos, se compara contra un grupo de control: películas viejas cuyo póster y '
                 'sinopsis también se descargaron de TMDB.')
        lista = []
        if 'clasificacion' in TEMP:
            c_ = TEMP['clasificacion']
            lista += [(f"{100 * c_['ap_texto']['diferencia']:+.1f}",
                       'puntos de AP pierde la sinopsis al clasificar géneros de películas nuevas'),
                      (f"{100 * c_['ap_imagen']['diferencia']:+.1f}", 'puntos de AP cambia el póster (prácticamente nada)')]
        if 'dependencia' in TEMP and {'control_tmdb', 'recientes_tmdb'} <= set(TEMP['dependencia']):
            d_ = TEMP['dependencia']
            lista.append((f"{d_['control_tmdb']:.2f} → {d_['recientes_tmdb']:.2f}",
                          'cuánto depende la clasificación del póster (control → recientes)'))
        if 'busqueda' in TEMP:
            m_ = TEMP['busqueda']['map_despues']
            lista.append((f"{m_['control_tmdb']['ambos']:.2f} → {m_['recientes_tmdb']['ambos']:.2f}",
                          'mAP del buscador con póster + sinopsis (control → recientes)'))
        comun.kpis(lista)
        filas_t = []
        if 'clasificacion' in TEMP:
            for m_, nombre in [('ap_imagen', 'Póster'), ('ap_texto', 'Sinopsis'), ('ap_fusion', 'Ambos')]:
                filas_t.append({'tarea': 'Clasificar géneros', 'modalidad': nombre,
                                'cambio (puntos)': round(100 * TEMP['clasificacion'][m_]['diferencia'], 1)})
        if 'busqueda' in TEMP:
            for m_, nombre in [('poster', 'Póster'), ('sinopsis', 'Sinopsis'), ('ambos', 'Ambos')]:
                filas_t.append({'tarea': 'Buscar por género', 'modalidad': nombre,
                                'cambio (puntos)': round(100 * TEMP['busqueda']['epoca'][m_]['diferencia'], 1)})
        r1, r2 = st.columns([1.3, 1])
        if filas_t:
            r1.altair_chart(barras(pd.DataFrame(filas_t), 'tarea', 'cambio (puntos)', 'modalidad',
                                   ['Póster', 'Sinopsis', 'Ambos'], 'Cambio con la época: recientes − control (AP)'),
                            width='stretch', theme=None)
        r2.markdown(comun.ficha('Qué significa',
                                'Con películas nuevas la <b>sinopsis</b> pierde capacidad de predecir el género y el '
                                '<b>póster</b> se mantiene. El modelo pasa a <b>depender más de la imagen</b>: el '
                                'desbalance entre modalidades no es fijo, cambia con la época de las películas. '
                                'En la búsqueda, en cambio, las películas nuevas se encuentran igual de bien.'),
                    unsafe_allow_html=True)
        st.page_link('pages/4_Ayer_y_hoy.py', label='Todos los resultados de ayer y hoy →')
        st.caption('Estrenos: datos e imágenes de TMDB. Este producto usa la API de TMDB pero no está respaldado ni '
                   'certificado por TMDB.')

# ============================================================================= EXPLICACIÓN TÉCNICA
with tab_tec:
    st.markdown('## Qué hicimos, explicado')
    st.write('Los conceptos del curso detrás del proyecto, de lo más general a lo más técnico.')

    with st.expander('1 · Representaciones: convertir películas en números', expanded=True):
        st.markdown(
            'Una **representación** (o *embedding*) es un vector de números que resume un dato. Un buen modelo pone '
            'cosas parecidas en vectores cercanos: dos pósters de terror quedan cerca entre sí y lejos de uno de '
            'dibujos animados. Todo el proyecto ocurre en esos espacios: **buscar es encontrar los vectores más '
            'cercanos** a la búsqueda.\n\n'
            'Usamos modelos **preentrenados y congelados**: ya aprendieron a representar imágenes y textos con '
            'millones de ejemplos, y nosotros solo entrenamos capas pequeñas encima (*transfer learning*). '
            'Por eso todo corre en Google Colab.')

    with st.expander('2 · Las redes neuronales: Transformers y atención'):
        st.markdown(
            'Todos los modelos que usamos son **Transformers**. Su pieza clave es la **atención**: cada palabra '
            '(o cada parche de imagen) mira a todas las demás y decide cuáles le importan para entender su significado.')
        st.latex(r'\text{Atención}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V')
        st.markdown(
            '- **Q, K, V** (consultas, claves, valores) son proyecciones de cada elemento. $QK^\\top$ mide qué tanto '
            'se relaciona cada par, y el softmax lo convierte en pesos que suman 1.\n'
            '- **Vision Transformer (ViT):** corta la imagen en parches de 32×32 y los trata como palabras.\n'
            '- **BERT / DistilBERT / XLM-RoBERTa:** Transformers de texto. DistilBERT es una versión reducida '
            '(6 capas en vez de 12) que conserva casi todo el rendimiento.\n'
            '- **Destilación:** un modelo "alumno" aprende a imitar las salidas de un "maestro". Así se hizo el '
            'CLIP multilingüe: aprendió a producir los mismos vectores que CLIP, pero en más de 50 idiomas.')

    with st.expander('3 · Aprendizaje contrastivo: cómo CLIP une imágenes y textos'):
        st.markdown(
            'CLIP se entrenó con pares (imagen, texto). Para cada lote, la pérdida **InfoNCE** premia que cada imagen '
            'quede cerca de *su* texto y lejos de los textos de las demás:')
        st.latex(r'\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(x_i, t_i)/\tau)}'
                 r'{\sum_{j=1}^{N}\exp(\text{sim}(x_i, t_j)/\tau)}')
        st.markdown(
            'Por eso un texto y una imagen se pueden comparar directamente: viven en el mismo espacio. '
            'Un efecto curioso que encontramos: CLIP también aprendió a **leer texto en las imágenes**, así que a '
            'veces encuentra un póster porque la palabra buscada está escrita en él (*Murder!* para "asesinato").')

    with st.expander('4 · Búsqueda semántica, brecha modal y z-score'):
        st.markdown('La cercanía entre la búsqueda $q$ y una película $v$ se mide con la **similitud coseno**:')
        st.latex(r'\text{sim}(q, v) = \frac{q \cdot v}{\lVert q \rVert\,\lVert v \rVert}')
        st.markdown(
            'Problema: texto contra texto da similitudes más altas que texto contra imagen. Es la **brecha modal**: '
            'en CLIP, imágenes y textos ocupan zonas distintas del espacio. Si se sumaran tal cual, la sinopsis '
            'dominaría siempre. Por eso cada puntaje se **estandariza** antes de combinarlos:')
        st.latex(r'\text{puntaje} = \alpha\, z_{\text{póster}} + (1-\alpha)\, z_{\text{sinopsis}}, '
                 r'\qquad z = \frac{s - \bar{s}}{\sigma_s}')
        st.markdown(
            'Con los estrenos de TMDB en el catálogo aparece un problema parecido: sus vectores tienen otro nivel de '
            'similitud y salían 1.7 veces más de lo que les corresponde. Por eso el z-score se calcula **dentro de '
            'cada fuente** (MM-IMDB y TMDB por separado), lo que baja el sesgo a 1.16 y casi no cambia el mAP.')

    with st.expander('5 · El adaptador: cómo entrenamos la representación'):
        st.markdown('Una capa lineal **residual** que corrige el vector de cada película:')
        st.latex(r"v' = \frac{v + Wv}{\lVert v + Wv \rVert}, \qquad W_{\text{inicial}} = 0")
        st.markdown(
            'Para cada película y cada género $g$ se calcula un puntaje $\\ell_g = s \\cdot \\text{sim}(v\', q_g) + b_g$, '
            'donde $q_g$ es una frase del género ("cine de terror", "a horror story"). Se entrena con **entropía '
            'cruzada binaria multietiqueta** (una película puede tener varios géneros) más un castigo que evita que '
            '$W$ se aleje demasiado:')
        st.latex(r'\mathcal{L} = -\frac{1}{G}\sum_{g}\big[y_g \log \sigma(\ell_g) + (1-y_g)\log(1-\sigma(\ell_g))\big]'
                 r' + \lambda \lVert W \rVert^2')
        st.markdown(
            '- **Optimizador Adam**, lotes de 256, **recorte de gradiente** para evitar saltos.\n'
            '- **Early stopping:** se detiene cuando la pérdida en validación deja de bajar.\n'
            '- **3 semillas** para comprobar que el resultado no depende del azar.\n'
            '- **Sin trampa:** se entrena con películas de *train*, se ajusta con *dev* y se evalúa solo con *test* '
            '(películas que nunca vio), y con frases distintas a las de evaluación.')

    with st.expander('6 · Métricas: cómo medimos'):
        st.markdown(
            '- **precision@10:** de los 10 primeros resultados, ¿qué fracción es del género buscado? Es lo que ve el usuario.\n'
            '- **AP (average precision):** promedia la precisión en cada posición donde aparece una película correcta. '
            'Premia ponerlas arriba. **mAP** es su promedio sobre todas las búsquedas. 1 = perfecto; el azar da '
            'la proporción de películas del género.')
        st.latex(r'\text{AP} = \frac{1}{R}\sum_{k=1}^{n} P(k)\cdot \text{rel}(k)')
        st.markdown(
            '- **F1 macro** (clasificación): media armónica de precisión y cobertura, calculada para cada género y '
            'promediada sin ponderar, así los géneros raros pesan igual que los comunes.\n'
            '- **Intervalo de confianza bootstrap:** se remuestrean los géneros con reemplazo 2 000 veces y se mira '
            'dónde cae el 95 % de los resultados. Si el intervalo de la mejora no incluye el cero, la mejora es real.\n'
            '- **Pureza de vecinos:** de los 10 vecinos más cercanos de cada película, ¿cuántos comparten algún '
            'género? Mide si el espacio quedó mejor organizado.\n'
            '- **UMAP:** proyecta los vectores de 512 o 768 dimensiones a 2D para poder verlos, intentando conservar '
            'quién está cerca de quién.')

    with st.expander('7 · La pregunta del proyecto: el desbalance entre modalidades'):
        st.markdown(
            'En un modelo que combina imagen y texto, una de las dos puede **dominar** y la otra quedar casi sin usar '
            '(*greedy modality*). Lo medimos de dos formas:\n\n'
            '1. **Clasificación:** comparando solo póster, solo sinopsis y la fusión (la fusión gana: la imagen aporta).\n'
            '2. **Búsqueda:** buscando el mejor peso α entre póster y sinopsis. Con la sinopsis mal representada, el '
            'mejor α era 0.9 (la sinopsis casi se ignoraba). Con buenas representaciones, 0.4-0.5.\n\n'
            '**Conclusión:** el desbalance depende de la calidad con la que se representa cada modalidad. '
            'Con **películas recientes** (2020-2026) la sinopsis pierde más que el póster al clasificar, y lo hace '
            '**con seguridad**: se equivoca sin anunciar su incertidumbre (páginas Desbalance y Ayer y hoy).')

comun.creditos()
