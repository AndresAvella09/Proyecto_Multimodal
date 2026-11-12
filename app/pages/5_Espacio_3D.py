"""Espacio de representaciones en 3D (Plotly): géneros antes y después del adaptador del notebook 11.

Usa `espacio3d.parquet` (PCA y UMAP calculados al exportar). Si no está, calcula un PCA aquí mismo con una
muestra del catálogo.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import comun

comun.configurar_pagina('Espacio 3D · películas')
comun.encabezado('El espacio de', 'las películas')

GEN_MAPA = ['Horror', 'Animation', 'Documentary', 'Western', 'Romance', 'Sci-Fi']   # mismos del UMAP del 11
COLORES = {'Horror': '#dc2626', 'Animation': '#f59e0b', 'Documentary': '#16a34a', 'Western': '#b45309',
           'Romance': '#db2777', 'Sci-Fi': '#3b82f6'}
ESPACIOS = ['poster_orig', 'poster_adapt', 'sinopsis_orig', 'sinopsis_adapt']


@st.cache_resource(show_spinner='Preparando el mapa...')
def cargar_mapa():
    if comun.existe('espacio3d.parquet'):
        return pd.read_parquet(os.path.join(comun.carpeta_datos(), 'espacio3d.parquet'))
    # Sin archivo: misma selección que la exportación, con PCA calculado aquí
    cat, V = comun.cargar_catalogo(), comun.cargar_vectores()
    gen = cat['generos'].fillna('').str.split(',').apply(lambda g: [x for x in g if x in GEN_MAPA])
    uno = gen.apply(len) == 1
    rng = np.random.default_rng(42)
    sel = np.concatenate([rng.permutation(np.where(uno & (cat['particion'] == 'test'))[0])[:2000],
                          rng.permutation(np.where(uno & (cat['fuente'] == 'TMDB'))[0])[:600]])
    df = pd.DataFrame({'idx': sel, 'genero': [gen.iloc[i][0] for i in sel], 'fuente': cat['fuente'].values[sel]})
    llave = {'poster_orig': ('poster', 'Sin entrenar'), 'poster_adapt': ('poster', 'Entrenada'),
             'sinopsis_orig': ('sinopsis', 'Sin entrenar'), 'sinopsis_adapt': ('sinopsis', 'Entrenada')}
    for esp, k in llave.items():
        X = V[k][sel] - V[k][sel].mean(0)
        xyz = X @ np.linalg.svd(X, full_matrices=False)[2][:3].T
        for j, eje in enumerate('xyz'):
            df[f'{esp}_pca_{eje}'] = xyz[:, j]
    return df


@st.cache_data(show_spinner=False)
def pureza(espacio, con_estrenos, k=10):
    """De los k vecinos más cercanos (coseno, en el espacio completo), fracción con el mismo género."""
    mapa = cargar_mapa()
    m = mapa if con_estrenos else mapa[mapa['fuente'] != 'TMDB']
    mod, etapa = espacio.split('_')
    X = comun.cargar_vectores()[(mod, 'Entrenada' if etapa == 'adapt' else 'Sin entrenar')][m['idx'].values]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    vecinos = np.argsort(-S, axis=1)[:, :k]
    g = m['genero'].values
    return float((g[vecinos] == g[:, None]).mean())


mapa = cargar_mapa()
metodos = [mt for mt in ['umap', 'pca'] if f'poster_orig_{mt}_x' in mapa]
hay_estrenos = (mapa['fuente'] == 'TMDB').any()

st.write('Cada punto es una película (de prueba de MM-IMDB y, si quieres, estrenos 2020-2026), coloreada por su '
         'género. Se proyectan los vectores de 512 (póster) o 768 (sinopsis) dimensiones a 3D. **Antes** del '
         'adaptador los géneros están mezclados; **después** se agrupan. Gira el mapa con el mouse.')
o1, o2, o3 = st.columns([1, 1, 1.2])
mod = o1.radio('Modalidad', ['poster', 'sinopsis'], format_func={'poster': 'Póster', 'sinopsis': 'Sinopsis'}.get,
               horizontal=True)
metodo = o2.radio('Proyección', metodos, format_func=str.upper, horizontal=True)
con_estrenos = o3.toggle('Incluir estrenos 2020-2026', value=False, disabled=not hay_estrenos)
o3.markdown('<div class="opcion-nota"><b>PCA:</b> las 3 direcciones de mayor varianza (lineal). '
            '<b>UMAP:</b> conserva quién está cerca de quién (no lineal).</div>', unsafe_allow_html=True)

m = mapa if con_estrenos else mapa[mapa['fuente'] != 'TMDB']
cat = comun.cargar_catalogo()
titulos = cat['titulo'].values[m['idx'].values]
anios = cat['anio'].fillna(0).astype(int).values[m['idx'].values]

cols = st.columns(2)
for col, etapa, nombre in [(cols[0], 'orig', 'Antes (sin entrenar)'), (cols[1], 'adapt', 'Después (con adaptador)')]:
    esp = f'{mod}_{etapa}'
    fig = go.Figure()
    for g in GEN_MAPA:
        for fuente, simbolo in [('MM-IMDB', 'circle'), ('TMDB', 'diamond')]:
            k = ((m['genero'] == g) & (m['fuente'] == fuente)).values
            if not k.any():
                continue
            fig.add_scatter3d(x=m[f'{esp}_{metodo}_x'].values[k], y=m[f'{esp}_{metodo}_y'].values[k],
                              z=m[f'{esp}_{metodo}_z'].values[k], mode='markers',
                              name=g if fuente == 'MM-IMDB' else f'{g} (estreno)', legendgroup=g,
                              marker=dict(size=2.6 if fuente == 'MM-IMDB' else 4, color=COLORES[g], symbol=simbolo,
                                          opacity=0.8, line=dict(width=0)),
                              text=[f'{t} ({a})' for t, a in zip(titulos[k], anios[k])],
                              hovertemplate='%{text}<extra>' + g + '</extra>')
    eje = dict(showticklabels=False, title='', backgroundcolor='#000', gridcolor='#1f1f1f', zerolinecolor='#333')
    fig.update_layout(scene=dict(xaxis=eje, yaxis=eje, zaxis=eje, bgcolor='#000'),
                      legend=dict(itemsizing='constant'))
    comun.estilo_plotly(fig, alto=580, titulo=f'{nombre} · pureza de vecinos {pureza(esp, con_estrenos):.2f}',
                        leyenda='abajo')
    fig.update_layout(legend=dict(y=0.02, yanchor='top'), margin=dict(b=70))
    col.plotly_chart(fig, width='stretch', config={'displaylogo': False})

azar = float((m['genero'].value_counts(normalize=True) ** 2).sum())
st.markdown(comun.ficha('Cómo leerlo',
                        f'La <b>pureza de vecinos</b> es la fracción de los 10 vecinos más cercanos de cada película '
                        f'(en el espacio completo, no en 3D) que tiene su mismo género; con estas {len(m):,} películas '
                        f'el azar daría <b>{azar:.2f}</b>. El adaptador la sube: acerca las películas del mismo género '
                        f'sin tocar la búsqueda, por eso cualquier frase libre sigue funcionando.'),
            unsafe_allow_html=True)
if 'umap' not in metodos:
    st.caption('Solo PCA: para el UMAP 3D, instala umap-learn en Colab y vuelve a correr la exportación.')
comun.creditos()
