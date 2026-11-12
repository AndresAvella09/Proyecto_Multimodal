"""Portada de la app (diseño de Michelle). Correr desde la raíz del repo:

    streamlit run app/Inicio.py

Póster: CLIP ViT-B/32 multilingüe · Sinopsis: multilingual-e5-base · Adaptadores entrenados (notebook 11) ·
Clasificadores de género del 06 y compuertas del C1/C2. Los datos se buscan como explica `comun.py`.
"""
import streamlit as st

import comun

comun.configurar_pagina('Buscador de películas')

try:
    cfg = comun.cargar_config()
    TEXTO_CATALOGO = f"{cfg['n_peliculas']:,} películas".replace(',', ' ')
    if cfg.get('n_recientes'):
        TEXTO_CATALOGO += f" · incluye estrenos hasta {cfg['anio_max']}"
except Exception:
    TEXTO_CATALOGO = '23 000+ películas'

comun.cinta()
st.markdown(
    '<div class="portada">'
    f'<div class="camaras">{comun.camara_svg(170)}{comun.camara_svg(120, espejo=True)}</div>'
    '<div class="marquesina">Buscador de <span>películas</span></div>'
    '<div class="lema">Describe una escena, un personaje o una idea. Te mostramos la película.</div>'
    f'<div class="detalle">Búsqueda multimodal por póster y sinopsis · en español o en inglés · {TEXTO_CATALOGO}</div>'
    '</div>', unsafe_allow_html=True)
_, centro, _ = st.columns([1, 1.5, 1])
if centro.button('Bienvenido · Entrar', width='stretch', key='entrar'):
    st.switch_page('pages/1_Buscador.py')

st.write('')
_, menu, _ = st.columns([0.5, 4, 0.5])
with menu:
    st.markdown('<div class="tarjeta-meta" style="text-align:center">O ve directo a:</div>', unsafe_allow_html=True)
    for col, (ruta, nombre) in zip(st.columns(len(comun.PAGINAS) - 1), comun.PAGINAS[1:]):
        col.page_link(ruta, label=nombre)
comun.cinta()
comun.creditos()
