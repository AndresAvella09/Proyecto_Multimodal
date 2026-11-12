"""Clasificador de géneros: para una película del catálogo, qué géneros predice el póster solo, la sinopsis
sola y la fusión de ambos (modelos del notebook 06). Usa logits precalculados: no carga CLIP ni DistilBERT."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import comun
from comun import COLOR_RAMA, NOMBRE_RAMA

comun.configurar_pagina('Clasificador · películas')
C = comun.requiere_clasificador()
comun.encabezado('¿Qué género', 've el modelo?')

st.write('Tres clasificadores entrenados con MM-IMDB (notebook 06) miran la misma película: uno **solo con el '
         'póster**, otro **solo con la sinopsis** y la **fusión** de los dos. Cada uno da una probabilidad para cada '
         'género; un género se predice si pasa el umbral elegido en validación.')

i = comun.elegir_pelicula()
st.write('')
comun.ficha_pelicula(i)

G = C['generos']
y = comun.etiquetas_de(i, G)
RAMAS = ['imagen', 'texto', 'fusion']
P = {r: comun.sigmoide(C['Z'][r][i]) for r in RAMAS}
U = {r: C['umbral'].get(r, np.full(len(G), 0.5)) for r in RAMAS}
pred = {r: P[r] >= U[r] for r in RAMAS}

st.markdown('### Lo que predice cada rama')
cols = st.columns(3)
for col, r in zip(cols, RAMAS):
    aciertos, sobran = int((pred[r] & (y > 0)).sum()), int((pred[r] & (y == 0)).sum())
    chips = ''.join(f'<span class="etiqueta{" sinopsis" if y[k] else " gris"}">{"✓ " if y[k] else ""}{html.escape(G[k])}</span> '
                    for k in np.argsort(-P[r]) if pred[r][k]) or '<span class="tarjeta-meta">ningún género pasa el umbral</span>'
    resumen = (f'acierta <b>{aciertos} de {int(y.sum())}</b> · {sobran} de más' if y.sum()
               else f'{int(pred[r].sum())} géneros (esta película no tiene etiquetas entre los {len(G)} evaluados)')
    col.markdown(f'<div class="ficha" style="border-left-color:{COLOR_RAMA[r]}"><h4>{NOMBRE_RAMA[r]}</h4>'
                 f'<p>{resumen}</p><div>{chips}</div></div>', unsafe_allow_html=True)

# Géneros a mostrar: los reales y los más probables para alguna rama
orden = np.argsort(-np.max([P[r] for r in RAMAS], axis=0))
mostrar = [k for k in orden if y[k]] + [k for k in orden if not y[k]][:max(3, 8 - int(y.sum()))]
mostrar = sorted(set(mostrar), key=lambda k: -P['fusion'][k])[::-1]
etiqueta = [f'★ {G[k]}' if y[k] else G[k] for k in mostrar]

fig = go.Figure()
for r in RAMAS:
    fig.add_bar(y=etiqueta, x=P[r][mostrar], name=NOMBRE_RAMA[r], orientation='h', marker_color=COLOR_RAMA[r],
                hovertemplate='%{y}: %{x:.2f}<extra>' + NOMBRE_RAMA[r] + '</extra>')
fig.update_layout(barmode='group', bargap=0.25, xaxis=dict(range=[0, 1], title='probabilidad'))
c1, c2 = st.columns([1.6, 1])
c1.plotly_chart(comun.estilo_plotly(fig, alto=110 + 42 * len(mostrar), titulo='Probabilidad por género (★ = género real)'),
                width='stretch', config={'displayModeBar': False})

# ¿A cuál rama se parece la fusión?
d_img = float(np.abs(P['fusion'] - P['imagen']).mean())
d_txt = float(np.abs(P['fusion'] - P['texto']).mean())
manda = 'la sinopsis' if d_txt < d_img else 'el póster'
with c2:
    st.markdown(comun.ficha('¿A quién le hace caso la fusión?',
                            f'Diferencia media entre las probabilidades de la fusión y las de cada rama: '
                            f'<b>{d_img:.3f}</b> con el póster y <b>{d_txt:.3f}</b> con la sinopsis. En esta película '
                            f'la fusión se parece más a <b>{manda}</b>. En promedio, la fusión depende sobre todo del '
                            f'texto (índice de dependencia de la imagen 0.20).'), unsafe_allow_html=True)
    if y.sum():
        L = {r: float(-(y * np.log(P[r] + 1e-9) + (1 - y) * np.log(1 - P[r] + 1e-9)).sum()) for r in RAMAS}
        mejor = min(L, key=L.get)
        st.markdown(comun.ficha('Pérdida en esta película',
                                ' · '.join(f'{NOMBRE_RAMA[r]} <b>{L[r]:.2f}</b>' for r in RAMAS)
                                + f'<br>Log-loss sumada sobre los {len(G)} géneros (menos es mejor). '
                                  f'La que mejor la clasifica: <b>{NOMBRE_RAMA[mejor].lower()}</b>.'),
                    unsafe_allow_html=True)
    st.page_link('pages/3_Desbalance.py', label='Apagar ramas y dañar modalidades en esta película →')

with st.expander('Detalles: probabilidades y umbrales de todos los géneros'):
    tabla = pd.DataFrame({'género': G, 'real': ['✓' if v else '' for v in y],
                          **{f'{NOMBRE_RAMA[r]}': P[r].round(3) for r in RAMAS},
                          **{f'umbral {NOMBRE_RAMA[r].lower()}': U[r].round(2) for r in RAMAS}})
    st.dataframe(tabla.sort_values('Fusión', ascending=False), hide_index=True, width='stretch')
    st.caption(f'Probabilidades calibradas: cada logit dividido por la temperatura que se ajustó en validación '
               f'(semilla {C["cfg"]["semilla"]} de las 5 del 06). Solo se muestran los {len(G)} géneros evaluados '
               f'en la comparación temporal.')

# ============================================================================= EXPLICACIÓN
st.markdown('## Cómo funciona')

with st.expander('1 · Los codificadores: póster y sinopsis convertidos en vectores', expanded=True):
    st.markdown(
        'Cada película entra al clasificador como dos vectores que **no se entrenan** (codificadores congelados):\n\n'
        '- **Póster → CLIP ViT-B/32** (512 números). Un Vision Transformer que corta la imagen en parches de 32×32 '
        'y que aprendió, con millones de pares imagen-texto, a ubicar imágenes parecidas cerca.\n'
        '- **Sinopsis → DistilBERT** (768 números). Un Transformer de texto de 6 capas, versión destilada de BERT.\n\n'
        'Antes de entrar a la cabeza, cada vector se **estandariza** con la media y la desviación de entrenamiento.')

with st.expander('2 · La cabeza: tres modelos con la misma arquitectura'):
    st.markdown('Lo único que se entrena es una cabeza pequeña. En la fusión, cada modalidad tiene su propia proyección:')
    st.latex(r'h = \big[\,\text{proy}_{\text{img}}(x_{\text{img}})\;;\;\text{proy}_{\text{txt}}(x_{\text{txt}})\,\big], '
             r'\qquad z = W\,\text{Dropout}(\text{ReLU}(h)) + b')
    st.markdown(
        '- $z$ tiene un **logit por género**, y $\\sigma(z_g)$ es la probabilidad de que la película sea de ese género. '
        'Es **multietiqueta**: una película puede ser Drama y Romance a la vez.\n'
        '- **Solo póster** y **solo sinopsis** usan la misma cabeza con una sola entrada. Comparar las tres dice '
        'cuánto aporta cada modalidad.\n'
        '- Se entrena con **entropía cruzada binaria** sobre todos los géneros, AdamW y *early stopping* según el AP '
        'en validación, con **5 semillas** para que el resultado no dependa del azar (aquí se muestra una).\n'
        '- A esta fusión la llamamos **ingenua**: concatena y deja que el entrenamiento decida, sin ningún mecanismo '
        'para equilibrar las modalidades.')

with st.expander('3 · Calibración y umbrales'):
    st.markdown(
        'Una red puede decir 0.95 cuando en realidad acierta el 70 % de las veces. Para que las probabilidades '
        'signifiquen lo que dicen, cada rama se **calibra por temperatura**: se busca en validación un único número '
        '$T$ y se usa $\\sigma(z/T)$.')
    st.latex(r'T^{*} = \arg\min_T \; \text{log-loss}\big(\sigma(z_{\text{dev}}/T),\, y_{\text{dev}}\big)')
    st.markdown(
        'Calibrar no cambia el orden de las películas (el AP queda igual), pero sí la **entropía**, que el C1 usa '
        'para medir qué tan segura está cada rama. Después, para cada género se elige en validación el **umbral** '
        'que maximiza el F1: un género raro suele necesitar un umbral más bajo que 0.5.')

with st.expander('4 · Métricas de esta página'):
    st.markdown(
        '- **Log-loss** (pérdida logarítmica): castiga mucho equivocarse con seguridad. Es la misma pérdida del '
        'entrenamiento.')
    st.latex(r'\text{log-loss} = -\sum_g \big[y_g\log p_g + (1-y_g)\log(1-p_g)\big]')
    st.markdown(
        '- **AP por género** y su promedio **mAP**: miden si las películas del género quedan arriba en el ranking, '
        'sin depender del umbral. Es la métrica principal del proyecto.\n'
        '- **Índice de dependencia de la imagen**: cuánto cae la fusión sin póster dividido entre lo que cae sin '
        'póster más lo que cae sin sinopsis. En test vale **0.20**: la fusión se apoya sobre todo en el texto, '
        'pero la imagen sí aporta (sin ella el AP baja). En la página Desbalance se calcula película por película.')

with st.expander('5 · Qué encontramos'):
    st.markdown(
        '- La **fusión le gana a cada rama sola**, incluso entrenando con solo el 2 % de los datos (C2): no hay una '
        'modalidad "golosa" que deje sin aprender a la otra.\n'
        '- Con **películas de 2020-2026** la sinopsis pierde mucho AP y el póster casi nada. Los géneros que más '
        'caen con el texto son Guerra, Musical, Familia, Ciencia ficción y Documental.\n'
        '- El texto falla **con seguridad**: ante sinopsis nuevas cae en los géneros por defecto (Drama, Romance, '
        'Comedia) sin volverse más incierto. Eso explica por qué los métodos de balanceo no lo arreglan '
        '(página Desbalance y Ayer y hoy).')

comun.creditos()
