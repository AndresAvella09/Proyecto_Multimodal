"""Desbalance entre modalidades, película por película: apagar cada rama (06), pesos de la compuerta por
incertidumbre (C1) y el dial de degradación (C2: borrado, ruido y mezcla con otra película)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import comun
from comun import AMBAR, BLANCO, COLOR_RAMA, GRIS, NOMBRE_RAMA, ROJO
from src.fusion import entropia_binaria

comun.configurar_pagina('Desbalance · películas')
C = comun.requiere_clasificador()
comun.encabezado('¿Póster o', 'sinopsis?')
st.write('¿De cuál de las dos modalidades depende la fusión? Aquí se mide en una sola película: apagando cada '
         'rama, mirando cuánto peso le da la compuerta a cada una y dañando una modalidad a propósito.')

i = comun.elegir_pelicula()
st.write('')
comun.ficha_pelicula(i, ancho=140)

G = C['generos']
y = comun.etiquetas_de(i, G)
hay_etiquetas = bool(y.sum())
Z = C['Z']
sig = comun.sigmoide


def logloss(p):
    return float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())


def generos_a_mostrar(*probs, n=8):
    orden = np.argsort(-np.max(probs, axis=0))
    k = [g for g in orden if y[g]] + [g for g in orden if not y[g]]
    return sorted(k[:max(n, int(y.sum()))], key=lambda g: probs[0][g])


def nombres(ks):
    return [f'★ {G[k]}' if y[k] else G[k] for k in ks]


def explicacion():
    st.markdown('## Cómo funciona')

    with st.expander('1 · Qué es el desbalance entre modalidades', expanded=True):
        st.markdown(
            'Un modelo que combina póster y sinopsis puede terminar **apoyándose casi solo en una** y desaprovechar '
            'la otra. En la literatura se habla de dos causas:\n\n'
            '- **De aprendizaje** (*greedy modality*): la modalidad más fácil baja la pérdida primero y la otra '
            'deja de recibir gradiente útil. Se arregla cambiando el entrenamiento (por ejemplo, OGM-GE).\n'
            '- **De datos**: una modalidad simplemente informa peor en ciertos casos. Ahí lo que importa es que el '
            'modelo **se dé cuenta** y le dé más peso a la otra.\n\n'
            'El proyecto pregunta cuál de las dos pasa con películas recientes (2020-2026), que el modelo nunca vio.')

    with st.expander('2 · Apagar ramas e índice de dependencia (notebook 06)'):
        st.markdown(
            'Para "apagar" una modalidad se reemplaza su vector por el **promedio de entrenamiento** y se vuelve a '
            'pasar la película por la fusión. Con la caída de AP sin cada rama se arma el índice:')
        st.latex(r'\text{dependencia de la imagen} = '
                 r'\frac{\Delta_{\text{sin imagen}}}{\Delta_{\text{sin imagen}} + \Delta_{\text{sin texto}}}')
        st.markdown(
            'Vale 0 si la fusión ignora el póster y 1 si ignora la sinopsis. En test MM-IMDB da **0.20**. Que no sea 0 '
            'es importante: la imagen sí aporta, aunque pese menos.')

    with st.expander('3 · La compuerta por incertidumbre (C1)'):
        st.markdown(
            'En vez de concatenar, se combinan los **logits calibrados** $\\tilde z_m$ de cada rama con un peso que '
            'depende del género $c$ y de la **entropía** $H_m$ de la rama en esa película:')
        st.latex(r'w_{m,c} = \text{softmax}_m(\alpha_{m,c} - \beta\, H_{m,c}), \qquad '
                 r'z_c = b_c + \lambda \sum_m w_{m,c}\,\tilde z_{m,c}')
        st.markdown(
            '- La entropía binaria $H = -p\\log p - (1-p)\\log(1-p)$ vale 0 si la rama está segura y 0.69 si dice 0.5.\n'
            '- Con **β > 0**, la rama insegura cede peso a la otra: justo lo que se querría si el texto falla.\n'
            '- Con **β = 0** el peso solo depende del género: es la **fusión tardía**, el control que permite saber '
            'si la entropía aporta algo.\n'
            '- Las versiones **3** agregan como tercera fuente la fusión ingenua del 06.\n\n'
            'Para saber si una rama "avisa" de su error se usa el **exceso**: log-loss − entropía. Si es positivo, la '
            'rama se equivoca más de lo que anuncia. Con las películas recientes la entropía del texto casi no '
            'cambia (0.186 → 0.192), pero el exceso sube: **falla sin avisar**. La compuerta aprendió β = −0.42 '
            '(modera a la rama segura en vez de castigar a la insegura).')

    with st.expander('4 · El dial de degradación (C2)'):
        st.markdown(
            'Para entender cómo falla el texto con películas nuevas, se daña una modalidad a propósito con una '
            'intensidad ρ entre 0 y 1, sobre el vector estandarizado:')
        st.latex(r'\text{borrado: } (1-\rho)\,x \qquad \text{ruido: } \sqrt{1-\rho}\,x + \sqrt{\rho}\,\varepsilon '
                 r'\qquad \text{mezcla: } \sqrt{1-\rho}\,x + \sqrt{\rho}\,x_{\text{otra}}')
        st.markdown(
            '- **Borrado y ruido** dejan a la rama **insegura**: el daño se anuncia.\n'
            '- **Mezcla** con otra película deja a la rama **segura y equivocada**: la entropía no sube y el exceso '
            'sí. Es la firma de las películas recientes, que se parecen a una mezcla con ρ ≈ 0.2.\n'
            '- Las compuertas **_aum** se ajustaron con copias degradadas de validación (borrado y ruido) para que '
            'aprendieran a reaccionar. Aprendieron β = +0.69, pero ni así el término de entropía mejora a la '
            'tardía equivalente: con daño fuerte ayuda pasar peso fijo al póster, y eso cuesta en las recientes.\n'
            '- **Pocos datos:** incluso entrenando con el 2 % de train, la fusión supera a la mejor rama sola por '
            '3.6 puntos de AP. No aparece una modalidad golosa.')

    with st.expander('5 · Modular el gradiente: OGM-GE (C3)'):
        st.markdown(
            'OGM-GE ataca el desbalance **de aprendizaje**. En cada lote mide cuánto aporta cada rama ($s_m$: '
            'probabilidad media en los géneros correctos usando solo su parte del logit) y frena el gradiente de '
            'la que domina:')
        st.latex(r'r = \frac{s_{\text{img}}}{s_{\text{txt}}}, \qquad k_{\text{dominante}} = 1 - \tanh\big(\alpha\,(r-1)\big)')
        st.markdown(
            'La variante **GE** suma ruido gaussiano al gradiente para no perder generalización. Resultado: la razón '
            'entre ramas quedó en r ≈ 0.95, casi balanceada desde el principio, así que OGM casi no actúa (el índice '
            'pasa de 0.200 a 0.193) y no cambia el AP.')

    with st.expander('6 · Conclusión'):
        st.markdown(
            '- Ningún método mejora en las películas recientes. Lo mejor en test MM-IMDB fueron **tardía3** y '
            '**compuerta3** (+0.39 puntos de AP, significativo), pero en las recientes la mejora no es significativa.\n'
            '- El cambio de época **no es un desbalance de aprendizaje**: el texto se equivoca **con seguridad** '
            '(cae en Drama, Romance y Comedia por defecto). Por eso ni reponderar por entropía ni modular el gradiente '
            'lo corrigen: el modelo no tiene ninguna señal de que debería desconfiar de la sinopsis.\n'
            '- Aun así, en el 42 % de las películas recientes el póster solo tiene menor log-loss que la sinopsis sola.')


# ============================================================================= 1. apagar ramas
st.markdown('## 1. Apagar cada rama')
st.write('Se reemplaza el embedding de una modalidad por el promedio de entrenamiento (como si no llegara nada) y '
         'se vuelve a pasar la película por la **fusión**. Si la predicción casi no cambia, esa rama no pesaba.')
P = {'completa': sig(Z['fusion'][i]), 'sin_imagen': sig(Z['fusion_sin_imagen'][i]),
     'sin_texto': sig(Z['fusion_sin_texto'][i])}
d_img = float(np.abs(P['completa'] - P['sin_imagen']).mean())
d_txt = float(np.abs(P['completa'] - P['sin_texto']).mean())
lista = [(f'{d_img:.3f}', 'cambio medio de probabilidad al apagar el póster'),
         (f'{d_txt:.3f}', 'cambio medio al apagar la sinopsis'),
         (f'{d_img / (d_img + d_txt + 1e-12):.2f}', 'dependencia del póster en esta película (0 = nada, 1 = todo; '
                                                    'promedio en test 0.20)')]
if hay_etiquetas:
    lista.append((f'{logloss(P["sin_imagen"]) - logloss(P["completa"]):+.3f} / '
                  f'{logloss(P["sin_texto"]) - logloss(P["completa"]):+.3f}',
                  'cuánto sube la pérdida sin póster / sin sinopsis'))
comun.kpis(lista)

ks = generos_a_mostrar(P['completa'], P['sin_imagen'], P['sin_texto'])
fig = go.Figure()
for clave, nombre, color in [('completa', 'Fusión completa', AMBAR), ('sin_imagen', 'Sin póster (solo sinopsis)', BLANCO),
                             ('sin_texto', 'Sin sinopsis (solo póster)', ROJO)]:
    fig.add_bar(y=nombres(ks), x=P[clave][ks], name=nombre, orientation='h', marker_color=color)
fig.update_layout(barmode='group', bargap=0.25, xaxis=dict(range=[0, 1], title='probabilidad'))
st.plotly_chart(comun.estilo_plotly(fig, alto=110 + 40 * len(ks), titulo='Probabilidad de la fusión (★ = género real)'),
                width='stretch', config={'displayModeBar': False})

# ============================================================================= 2. compuerta
st.markdown('## 2. La compuerta por incertidumbre (C1)')
NOMBRES_MET = {'compuerta': 'compuerta (póster + sinopsis)', 'compuerta3': 'compuerta3 (+ fusión ingenua)',
               'tardia': 'tardía (sin incertidumbre)', 'tardia3': 'tardía3 (sin incertidumbre)',
               'compuerta_aum': 'compuerta_aum (C2, con degradación)', 'tardia_aum': 'tardía_aum (C2)'}
disponibles = [m for m in NOMBRES_MET if m in C['compuertas']]
st.write('En vez de un peso fijo, la compuerta decide **en cada película y género** cuánto pesa cada rama, según '
         'qué tan segura está (su entropía): $w = \\mathrm{softmax}(\\alpha_c - \\beta H)$.')
met = st.radio('Método', disponibles, format_func=NOMBRES_MET.get, horizontal=True, key='desb_metodo')
entradas = C['cfg']['compuertas'][met]['entradas']
Zi = {'imagen': Z['imagen'][i:i + 1], 'texto': Z['texto'][i:i + 1], 'fusion_ingenua': Z['fusion'][i:i + 1]}
zc, w = comun.aplicar_compuerta(C, met, Zi)
modelo = C['compuertas'][met]
beta, lam = float(modelo.beta.detach()), float(modelo.log_lambda.detach().exp())
base = np.exp(modelo.alfa.detach().numpy())
base = base / base.sum(1, keepdims=True)
H = {n: entropia_binaria(Zi[n][0]) for n in entradas}

pc = sig(zc[0])
ks = generos_a_mostrar(pc, sig(Z['fusion'][i]))
COL_FUENTE = {'imagen': ROJO, 'texto': BLANCO, 'fusion_ingenua': AMBAR}
NOM_FUENTE = {'imagen': 'Póster', 'texto': 'Sinopsis', 'fusion_ingenua': 'Fusión ingenua'}
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                    subplot_titles=('Peso de cada rama en esta película (◆ = peso base del género)',
                                    'Entropía de cada rama (0 = segura, 0.69 = no sabe)'))
for m, n in enumerate(entradas):
    fig.add_bar(y=nombres(ks), x=w[0, ks, m], name=NOM_FUENTE[n], orientation='h', marker_color=COL_FUENTE[n],
                row=1, col=1, legendgroup=n)
    fig.add_scatter(y=nombres(ks), x=H[n][ks], name=NOM_FUENTE[n], mode='markers',
                    marker=dict(color=COL_FUENTE[n], size=11, line=dict(color='#000', width=1)),
                    row=1, col=2, legendgroup=n, showlegend=False)
fig.add_scatter(y=nombres(ks), x=base[ks, 0], mode='markers', marker=dict(symbol='diamond', size=9, color=GRIS),
                name='peso base del póster', row=1, col=1)
fig.update_layout(barmode='stack')
fig.update_xaxes(range=[0, 1], row=1, col=1)
fig.update_xaxes(range=[0, 0.72], row=1, col=2)
st.plotly_chart(comun.estilo_plotly(fig, alto=170 + 40 * len(ks), leyenda='abajo'), width='stretch',
                config={'displayModeBar': False})
c1, c2 = st.columns(2)
c1.markdown(comun.ficha('Qué aprendió',
                        f'β = <b>{beta:+.3f}</b>, λ = <b>{lam:.2f}</b>. '
                        + ('Con β = 0 es la fusión tardía: el peso solo depende del género.' if abs(beta) < 1e-9 else
                           'β negativo: la compuerta <b>modera</b> a la rama segura (al sumar logits, una rama segura '
                           'ya pesa más por sí sola).' if beta < 0 else
                           'β positivo: a la rama insegura le quita todavía más peso.')),
            unsafe_allow_html=True)
c2.markdown(comun.ficha('Lo que encontramos',
                        'Con las películas recientes la sinopsis <b>no anuncia</b> su error: su entropía casi no sube '
                        '(0.186 → 0.192), así que la compuerta no tiene cómo darle más peso al póster. El texto se '
                        'equivoca <b>con seguridad</b>.'), unsafe_allow_html=True)

# ============================================================================= 3. dial
st.markdown('## 3. El dial: dañar una modalidad (C2)')
if not C['X'] or not C['cabezas']:
    st.info('Falta `clasif_img.npy` / `clasif_txt.npy` en los datos: vuelve a correr la exportación.')
    explicacion()
    comun.creditos()
    st.stop()

TIPOS = {'Borrado': 'borrado', 'Ruido': 'ruido', 'Mezcla con otra película': 'mezcla'}
EXPLICA = {
    'borrado': 'x′ = (1 − ρ)·x: el embedding se encoge hacia el promedio. La rama pierde información y debería '
               'volverse <b>insegura</b> (sube su entropía): una degradación <b>anunciada</b>.',
    'ruido': 'x′ = √(1 − ρ)·x + √ρ·ε, con ε al azar: misma escala, pero basura. La rama puede seguir segura.',
    'mezcla': 'x′ = √(1 − ρ)·x + √ρ·x<sub>otra</sub>: la entrada sigue pareciendo una película real, pero de otra. '
              'La rama queda <b>segura y equivocada</b>, el mismo patrón del texto con las películas recientes '
              '(equivalen a ρ ≈ 0.2).',
}
o1, o2, o3 = st.columns([1, 1.4, 1.6])
mod = o1.radio('Modalidad que se daña', ['Sinopsis', 'Póster'], key='dial_mod')
m_x = 'txt' if mod == 'Sinopsis' else 'img'
rama = 'texto' if m_x == 'txt' else 'imagen'
tipo = TIPOS[o2.radio('Tipo de daño', list(TIPOS), key='dial_tipo')]
rho = o3.slider('Intensidad ρ', 0.0, 1.0, 0.2 if tipo == 'mezcla' else 0.8, 0.05, key=f'dial_rho_{tipo}')
o3.markdown(f'<div class="opcion-nota">{EXPLICA[tipo]}</div>', unsafe_allow_html=True)

cat = comun.cargar_catalogo()
x0 = {m: np.asarray(C['X'][m][i], dtype=np.float32) for m in ('img', 'txt')}
if tipo == 'mezcla':
    clave = f'mezcla_{i}'
    if clave not in st.session_state:
        st.session_state[clave] = int(np.random.default_rng(i).choice(np.delete(np.arange(len(cat)), i)))
    j = st.session_state[clave]
    a, b = st.columns([0.6, 5])
    a.image(comun.imagen(j), width=60)
    b.markdown(f'Se mezcla con **{comun.nombre_pelicula(j)}** · '
               f'<span class="tarjeta-meta">{(cat.iloc[j]["generos"] or "").replace(",", " · ")}</span>',
               unsafe_allow_html=True)
    if b.button('Cambiar de película', key='mezcla_otra'):
        st.session_state[clave] = int(np.random.default_rng().choice(np.delete(np.arange(len(cat)), i)))
        st.rerun()
    eps = np.asarray(C['X'][m_x][j], dtype=np.float32)
else:
    eps = np.random.default_rng((i, 1 if m_x == 'txt' else 0)).standard_normal(x0[m_x].shape[0]).astype(np.float32)

RHOS = np.round(np.unique(np.concatenate([np.linspace(0, 1, 21), [rho]])), 4)
X = {m: np.repeat(x0[m][None], len(RHOS), 0) for m in x0}
X[m_x] = np.stack([comun.degradar(x0[m_x], tipo, r, eps) for r in RHOS])
Zv = comun.logits_vivos(C, X)
met_dial = 'compuerta_aum' if 'compuerta_aum' in C['compuertas'] else 'compuerta'
zg, wg = comun.aplicar_compuerta(C, met_dial, Zv)
k_rho = int(np.where(RHOS == np.round(rho, 4))[0][0])

H_rama = entropia_binaria(Zv[rama]).mean(1)
curvas = {'entropía (lo que la rama anuncia)': H_rama}
if hay_etiquetas:
    L_rama = np.array([logloss(sig(z)) for z in Zv[rama]])
    curvas['log-loss (cuánto se equivoca)'] = L_rama
    curvas['exceso = log-loss − entropía'] = L_rama - H_rama

lista = [(f'{H_rama[0]:.3f} → {H_rama[k_rho]:.3f}', f'entropía media de la {mod.lower()} (sin daño → con daño)')]
if hay_etiquetas:
    lista.append((f'{L_rama[0]:.3f} → {L_rama[k_rho]:.3f}', f'log-loss media de la {mod.lower()}'))
lista.append((f'{wg[0, :, 0].mean():.2f} → {wg[k_rho, :, 0].mean():.2f}',
              f'peso medio del póster en la {met_dial}'))
comun.kpis(lista)

g1, g2 = st.columns(2)
fig = go.Figure()
for (nombre, v), color in zip(curvas.items(), [GRIS, ROJO, AMBAR]):
    fig.add_scatter(x=RHOS, y=v, mode='lines+markers', name=nombre, line=dict(color=color), marker=dict(size=4))
fig.add_vline(x=rho, line=dict(color='#444', dash='dot'))
fig.update_xaxes(title='ρ')
g1.plotly_chart(comun.estilo_plotly(fig, alto=400, titulo=f'{tipo.capitalize()} de la {mod.lower()}: ¿avisa la rama?',
                                    leyenda='abajo'),
                width='stretch', config={'displayModeBar': False})

fig = go.Figure()
fig.add_scatter(x=RHOS, y=wg[:, :, 0].mean(1), mode='lines+markers', name=f'peso del póster ({met_dial})',
                line=dict(color=ROJO), marker=dict(size=4))
if hay_etiquetas:
    for nombre, zz, color in [('log-loss fusión ingenua', Zv['fusion_ingenua'], AMBAR), (f'log-loss {met_dial}', zg, BLANCO)]:
        fig.add_scatter(x=RHOS, y=[logloss(sig(z)) for z in zz], mode='lines', name=nombre,
                        line=dict(color=color, dash='dash'), yaxis='y2')
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='log-loss'))
fig.add_vline(x=rho, line=dict(color='#444', dash='dot'))
fig.update_xaxes(title='ρ')
fig.update_layout(yaxis=dict(title='peso del póster', range=[0, 1]))
g2.plotly_chart(comun.estilo_plotly(fig, alto=400, titulo='¿La compuerta le pasa el peso al póster?', leyenda='abajo'),
                width='stretch', config={'displayModeBar': False})

ks = generos_a_mostrar(sig(Zv[rama][0]), sig(Zv[rama][k_rho]), n=6)
fig = go.Figure()
for nombre, z, color in [(f'{mod} sin daño', Zv[rama][0], GRIS), (f'{mod} con daño (ρ = {rho:.2f})', Zv[rama][k_rho], COLOR_RAMA[rama]),
                         (f'Fusión con daño', Zv['fusion_ingenua'][k_rho], AMBAR)]:
    fig.add_bar(y=nombres(ks), x=sig(z)[ks], name=nombre, orientation='h', marker_color=color)
fig.update_layout(barmode='group', bargap=0.25, xaxis=dict(range=[0, 1], title='probabilidad'))
st.plotly_chart(comun.estilo_plotly(fig, alto=110 + 40 * len(ks), titulo='Probabilidades antes y después del daño (★ = género real)'),
                width='stretch', config={'displayModeBar': False})
st.caption(f'Mismos modelos del 06 (semilla {C["cfg"]["semilla"]}) sobre los embeddings estandarizados de esta película; '
           'el ruido es fijo por película para que la curva sea suave. En el C2 el dial se aplicó a todo el test a la vez: '
           'ver la página Ayer y hoy.')
explicacion()
comun.creditos()
