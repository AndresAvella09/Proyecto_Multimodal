"""Piezas compartidas por todas las páginas de la app: estilo, navegación, datos y modelos.

Los datos (lo que exporta `scripts/exportar_app.py`) se buscan, en este orden:
1. en la carpeta de la variable de entorno `APP_DATOS`;
2. en `app/datos/` (carpeta local, ignorada por git);
3. si está definida `APP_DATASET` (p. ej. `usuario/peliculas-multimodal-app`), se descargan una vez de ese
   repo de datasets de Hugging Face a `app/datos/` (o a `APP_DATOS`, si se definió y no existe todavía).

La lógica (búsqueda, cabezas de clasificación, compuertas) viene de `src/`, la misma de los notebooks.
"""
import html
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from src.buscador import normalizar  # noqa: E402

ROJO, BLANCO, GRIS, AMBAR = '#e50914', '#f5f5f5', '#8a8a8a', '#f59e0b'
COLOR_RAMA = {'imagen': ROJO, 'texto': BLANCO, 'fusion': AMBAR}
NOMBRE_RAMA = {'imagen': 'Póster', 'texto': 'Sinopsis', 'fusion': 'Fusión'}
PAGINAS = [('Inicio.py', 'Inicio'), ('pages/1_Buscador.py', 'Buscador'), ('pages/2_Clasificador.py', 'Clasificador'),
           ('pages/3_Desbalance.py', 'Desbalance'), ('pages/4_Ayer_y_hoy.py', 'Ayer y hoy'),
           ('pages/5_Espacio_3D.py', 'Espacio 3D')]

# ============================================================================= estilo (diseño de Michelle)
CINTA_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='34' height='40'>"
             "<rect width='34' height='40' fill='%23000'/>"
             "<rect x='9' y='4' width='16' height='9' rx='2.5' fill='%23f5f5f5'/>"
             "<rect x='9' y='27' width='16' height='9' rx='2.5' fill='%23f5f5f5'/></svg>")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, button {font-family: 'Inter', sans-serif;}
.stApp {background: #000;}
.block-container {padding-top: 1.2rem; max-width: 1200px;}
h1, h2, h3, .marquesina {font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.04em; font-weight: 400 !important;}
h2 {font-size: 2.2rem !important;} h3 {font-size: 1.6rem !important;}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarNav"] {display: none;}

/* cinta de película */
.cinta {height: 40px; background-image: url("CINTA_SVG"); background-repeat: repeat-x;
        border-top: 1px solid #222; border-bottom: 1px solid #222; margin: 0.4rem 0 1.2rem 0;}
.cinta.delgada {height: 40px; opacity: 0.85; margin: 0.2rem 0 0.8rem 0;}

/* portada */
.portada {text-align: center; padding: 1.5rem 0 0.5rem 0;}
.portada .marquesina {font-size: 5.2rem; line-height: 0.95; color: #f5f5f5; margin: 0.6rem 0 0.2rem 0;}
.portada .marquesina span {color: #e50914;}
.portada .lema {font-size: 1.15rem; color: #cfcfcf; margin-bottom: 0.4rem;}
.portada .detalle {font-size: 0.9rem; color: #8a8a8a;}
.camaras {display: flex; justify-content: center; align-items: flex-end; gap: 3.5rem; margin-top: 0.5rem;}
.creditos {position: fixed; right: 14px; bottom: 10px; font-size: 10.5px; color: #8a8a8a; text-align: right;
           line-height: 1.35; z-index: 1000; letter-spacing: 0.02em;}
.creditos b {color: #cfcfcf; font-weight: 600;}

/* botones */
.stButton > button {border-radius: 2px; border: 1px solid #e50914; background: #0a0a0a; color: #f5f5f5;
                    font-weight: 500; transition: all 0.15s;}
.stButton > button:hover {background: #e50914; color: #fff; border-color: #e50914;}
.st-key-entrar button {font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.08em;
                    padding: 0.55rem 2.5rem; background: #e50914 !important; color: #fff !important; min-height: 3.2rem;}
.st-key-entrar button p {font-family: 'Bebas Neue', sans-serif !important; font-size: 1.8rem !important;
                         white-space: nowrap; overflow: visible; text-overflow: clip;}
.st-key-entrar button:hover {background: #b20710 !important;}

/* menú de páginas */
[data-testid="stPageLink"] a {border-bottom: 2px solid transparent; border-radius: 0; padding: 0.15rem 0.2rem;
                              justify-content: center;}
[data-testid="stPageLink"] a p, [data-testid="stPageLink"] a span {font-family: 'Bebas Neue', sans-serif !important;
                              font-size: 1.25rem !important; letter-spacing: 0.06em; color: #bdbdbd;}
[data-testid="stPageLink"] a:hover {background: transparent; border-bottom-color: #e50914;}
[data-testid="stPageLink"] a[aria-current="page"], [data-testid="stPageLink-NavLink"][aria-current="page"]
    {border-bottom-color: #e50914;}
[data-testid="stPageLink"] a[aria-current="page"] p, [data-testid="stPageLink"] a[aria-current="page"] span {color: #fff;}

/* pestañas */
.stTabs [data-baseweb="tab-list"] {gap: 0.6rem; border-bottom: 1px solid #222;}
.stTabs [data-baseweb="tab"] {font-family: 'Bebas Neue', sans-serif; font-size: 1.35rem; letter-spacing: 0.06em;
                              color: #bdbdbd; padding: 0.3rem 0.9rem;}
.stTabs [aria-selected="true"] {color: #fff !important;}

/* panel de opciones */
.opcion-nota {font-size: 0.78rem; color: #9a9a9a; line-height: 1.3; margin-top: -0.4rem;}
.opcion-nota b {color: #e50914; font-weight: 600;}

/* tarjetas de resultados */
.tarjeta-titulo {font-weight: 600; font-size: 0.9rem; line-height: 1.2; margin-top: 0.35rem; color: #f5f5f5;}
.tarjeta-meta {font-size: 0.76rem; color: #9a9a9a; line-height: 1.25;}
.etiqueta {display: inline-block; font-size: 0.68rem; padding: 1px 8px; border-radius: 2px; margin: 4px 0 2px 0;
           border: 1px solid #e50914; color: #e50914; text-transform: uppercase; letter-spacing: 0.04em;}
.etiqueta.sinopsis {border-color: #f5f5f5; color: #f5f5f5;}
.etiqueta.estreno {border-color: #f59e0b; color: #f59e0b;}
.etiqueta.gris {border-color: #555; color: #bdbdbd;}
[data-testid="stImage"] img {border: 3px solid #111; outline: 1px solid #2a2a2a;}

/* sin pantalla completa en las imágenes */
[data-testid="stElementToolbar"], [data-testid="StyledFullScreenButton"],
button[title="View fullscreen"], button[aria-label="View fullscreen"],
[data-testid="stImage"] button, [data-testid="stImageContainer"] button {display: none !important;}

/* bloques de explicación */
.ficha {border: 1px solid #262626; border-left: 3px solid #e50914; padding: 0.8rem 1rem; margin-bottom: 0.8rem;
        background: #070707;}
.ficha h4 {font-family: 'Bebas Neue', sans-serif; font-size: 1.35rem; letter-spacing: 0.05em; margin: 0 0 0.3rem 0;
           color: #fff; font-weight: 400;}
.ficha p, .ficha li {font-size: 0.92rem; color: #d6d6d6; margin-bottom: 0.3rem;}
.kpi {border: 1px solid #262626; padding: 0.8rem 1rem; text-align: center; background: #070707;}
.kpi .valor {font-family: 'Bebas Neue', sans-serif; font-size: 2.3rem; color: #e50914; line-height: 1.1;
             white-space: nowrap;}
.kpi .texto {font-size: 0.82rem; color: #bdbdbd;}

/* diagrama de arquitectura */
.flujo {display: flex; align-items: stretch; gap: 0.5rem; margin: 0.4rem 0 0.9rem 0; flex-wrap: wrap;}
.caja {flex: 1; min-width: 150px; border: 1px solid #333; background: #0b0b0b; padding: 0.6rem 0.7rem;}
.caja.roja {border-color: #e50914;}
.caja .nombre {font-family: 'Bebas Neue', sans-serif; font-size: 1.2rem; letter-spacing: 0.05em; color: #fff;}
.caja .det {font-size: 0.78rem; color: #a8a8a8; line-height: 1.3;}
.flecha {display: flex; align-items: center; color: #e50914; font-size: 1.4rem;}
.carril {font-size: 0.75rem; color: #e50914; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.6rem;}
</style>
""".replace('CINTA_SVG', CINTA_SVG)


def configurar_pagina(titulo='Buscador de películas'):
    """Primera llamada de cada página: título de la pestaña, ancho y estilo."""
    st.set_page_config(page_title=titulo, page_icon='🎬', layout='wide', initial_sidebar_state='collapsed')
    st.markdown(CSS, unsafe_allow_html=True)


def cinta(delgada=False):
    st.markdown(f'<div class="cinta{" delgada" if delgada else ""}"></div>', unsafe_allow_html=True)


def camara_svg(ancho=150, espejo=False):
    """Cámara de cine antigua con dos carretes (dibujo propio, genérico)."""
    giro = ' transform="scale(-1,1) translate(-200,0)"' if espejo else ''
    radios = ''.join(
        f'<line x1="{cx}" y1="44" x2="{cx + 20 * np.cos(a):.1f}" y2="{44 + 20 * np.sin(a):.1f}" '
        f'stroke="#f5f5f5" stroke-width="3"/>'
        for cx in (62, 128) for a in np.linspace(0, 2 * np.pi, 6)[:-1])
    return (f'<svg width="{ancho}" viewBox="0 0 200 175" xmlns="http://www.w3.org/2000/svg"><g{giro}>'
            '<circle cx="62" cy="44" r="32" fill="#000" stroke="#f5f5f5" stroke-width="4"/>'
            '<circle cx="128" cy="44" r="32" fill="#000" stroke="#f5f5f5" stroke-width="4"/>'
            f'{radios}'
            '<circle cx="62" cy="44" r="7" fill="#e50914"/><circle cx="128" cy="44" r="7" fill="#e50914"/>'
            '<rect x="38" y="80" width="112" height="56" rx="5" fill="#e50914" stroke="#f5f5f5" stroke-width="4"/>'
            '<rect x="52" y="92" width="34" height="18" rx="2" fill="#000"/>'
            '<polygon points="150,96 188,82 188,134 150,120" fill="#000" stroke="#f5f5f5" stroke-width="4"/>'
            '<line x1="94" y1="136" x2="64" y2="172" stroke="#f5f5f5" stroke-width="4"/>'
            '<line x1="94" y1="136" x2="124" y2="172" stroke="#f5f5f5" stroke-width="4"/>'
            '<line x1="94" y1="136" x2="94" y2="172" stroke="#f5f5f5" stroke-width="4"/>'
            '</g></svg>')


def creditos():
    st.markdown('<div class="creditos"><b>Michelle Ruiz · Diego Avella</b><br>Ingeniería Estadística<br>'
                'Estrenos: datos de TMDB (no respaldado ni certificado por TMDB)</div>', unsafe_allow_html=True)


def encabezado(titulo, resaltado=''):
    """Título de la página, menú de páginas y cinta."""
    st.markdown(f'<div class="marquesina" style="font-size:2.6rem; line-height:1; margin-top:0.3rem;">'
                f'🎬 {titulo} <span style="color:#e50914">{resaltado}</span></div>', unsafe_allow_html=True)
    for col, (ruta, nombre) in zip(st.columns(len(PAGINAS)), PAGINAS):
        col.page_link(ruta, label=nombre)
    cinta(delgada=True)


def ficha(titulo, texto):
    return f'<div class="ficha">{f"<h4>{titulo}</h4>" if titulo else ""}<p>{texto}</p></div>'


def kpis(valores):
    """valores: lista de (valor, texto) -> fila de recuadros con cifras."""
    for col, (valor, texto) in zip(st.columns(max(len(valores), 1)), valores):
        col.markdown(f'<div class="kpi"><div class="valor">{valor}</div><div class="texto">{texto}</div></div>',
                     unsafe_allow_html=True)
    st.write('')


def barras(df, x, y, color, orden_color, titulo, horizontal=False, alto=280):
    """Gráfica de barras con la estética de la app (Altair)."""
    import altair as alt
    esc = alt.Scale(domain=orden_color, range=[GRIS, ROJO, BLANCO][:len(orden_color)])
    if horizontal:
        enc = dict(y=alt.Y(f'{y}:N', sort='-x', title=None), x=alt.X(f'{x}:Q', title=None))
    else:
        enc = dict(x=alt.X(f'{x}:N', title=None, sort=None, axis=alt.Axis(labelAngle=0)),
                   y=alt.Y(f'{y}:Q', title=None), xOffset=alt.XOffset(f'{color}:N', sort=orden_color))
    ch = (alt.Chart(df, title=titulo).mark_bar()
          .encode(color=alt.Color(f'{color}:N', scale=esc, sort=orden_color,
                                  legend=alt.Legend(orient='top', title=None)),
                  tooltip=list(df.columns), **enc)
          .properties(height=alto))
    return (ch.configure(background='#000')
              .configure_view(stroke=None)
              .configure_axis(labelColor='#cfcfcf', titleColor='#cfcfcf', gridColor='#1f1f1f', domainColor='#444')
              .configure_legend(labelColor='#cfcfcf')
              .configure_title(color='#f5f5f5', font='Inter', fontSize=14, anchor='start'))


def estilo_plotly(fig, alto=380, titulo=None, leyenda='arriba'):
    """Fondo negro, letra Inter y rejilla tenue, como las gráficas de Altair. El título va arriba del todo y
    la leyenda justo encima de la gráfica (o debajo, con leyenda='abajo', si hay títulos de subgráficas)."""
    arriba = leyenda == 'arriba'
    fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', height=alto,
                      font=dict(family='Inter, sans-serif', color='#cfcfcf', size=12),
                      title=dict(text=titulo, font=dict(color='#f5f5f5', size=14), x=0, xref='paper',
                                 y=1, yref='container', yanchor='top', pad=dict(t=8)) if titulo else None,
                      margin=dict(l=10, r=10, t=(34 if titulo else 0) + (36 if arriba else 24), b=10 if arriba else 90),
                      legend=dict(orientation='h', x=0, title=None, yanchor='bottom' if arriba else 'top',
                                  y=1.01 if arriba else -0.2))
    fig.update_xaxes(gridcolor='#1f1f1f', zerolinecolor='#444')
    fig.update_yaxes(gridcolor='#1f1f1f', zerolinecolor='#444')
    return fig


# ============================================================================= datos
def _tiene_datos(d):
    return bool(d) and os.path.exists(os.path.join(d, 'config.json'))


@st.cache_resource(show_spinner='Descargando los datos de la app (solo la primera vez)...')
def carpeta_datos():
    """Carpeta con lo que exporta scripts/exportar_app.py (ver el docstring del módulo)."""
    propia = os.environ.get('APP_DATOS')
    local = os.path.join(RAIZ, 'app', 'datos')
    for d in (propia, local):
        if _tiene_datos(d):
            return d
    repo = os.environ.get('APP_DATASET')
    if repo:
        from huggingface_hub import snapshot_download
        destino = propia or local
        snapshot_download(repo_id=repo, repo_type='dataset', local_dir=destino,
                          token=os.environ.get('HF_TOKEN') or None)
        if _tiene_datos(destino):
            return destino
    st.error('No encontré los datos de la app. Define `APP_DATOS` (carpeta que generó '
             '`scripts/exportar_app.py`) o `APP_DATASET` (repo de datasets de Hugging Face).')
    st.stop()


def existe(nombre):
    return os.path.exists(os.path.join(carpeta_datos(), nombre))


@st.cache_resource(show_spinner=False)
def cargar_config():
    with open(os.path.join(carpeta_datos(), 'config.json'), encoding='utf-8') as fh:
        return json.load(fh)


@st.cache_resource(show_spinner='Cargando películas...')
def cargar_catalogo():
    cat = pd.read_parquet(os.path.join(carpeta_datos(), 'catalogo.parquet'))
    if 'particion' not in cat:                      # exportación vieja del notebook 13
        cat['particion'] = np.where(cat.get('fuente', 'MM-IMDB') == 'TMDB', 'reciente', '?')
    if 'fuente' not in cat:
        cat['fuente'] = 'MM-IMDB'
    return cat


def estreno():
    return (cargar_catalogo()['fuente'] == 'TMDB').values


@st.cache_resource(show_spinner='Cargando vectores...')
def cargar_vectores():
    d = carpeta_datos()
    cargar = lambda n: np.load(os.path.join(d, n)).astype(np.float32)
    V = {('poster', 'Entrenada'): cargar('poster_adapt.npy'), ('sinopsis', 'Entrenada'): cargar('sinopsis_adapt.npy'),
         ('poster', 'Sin entrenar'): cargar('poster_orig.npy'), ('sinopsis', 'Sin entrenar'): cargar('sinopsis_orig.npy')}
    assert all(len(v) == len(cargar_catalogo()) for v in V.values())
    return V


class _CodificadorFalso:
    """Solo para pruebas sin internet (APP_CODIFICADOR_FALSO=1): vectores al azar fijos por frase."""

    def __init__(self, d):
        self.d = d

    def encode(self, textos, **_):
        import zlib
        out = np.stack([np.random.default_rng(zlib.crc32(t.encode())).standard_normal(self.d) for t in textos])
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


@st.cache_resource(show_spinner='Cargando los modelos de lenguaje (la primera vez tarda un poco)...')
def cargar_codificadores(nombre_poster, nombre_sinopsis):
    if os.environ.get('APP_CODIFICADOR_FALSO'):
        V = cargar_vectores()
        return (_CodificadorFalso(V[('poster', 'Entrenada')].shape[1]),
                _CodificadorFalso(V[('sinopsis', 'Entrenada')].shape[1]))
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nombre_poster), SentenceTransformer(nombre_sinopsis)


@st.cache_data(max_entries=200, show_spinner=False)
def codificar(consulta):
    cfg = cargar_config()
    clip, e5 = cargar_codificadores(cfg['modelo_poster'], cfg['modelo_sinopsis'])
    qp = clip.encode([consulta], convert_to_numpy=True)[0]
    qs = e5.encode([f'query: {consulta}'], convert_to_numpy=True, normalize_embeddings=True)[0]
    return qp / np.linalg.norm(qp), qs


def buscar(consulta, modo, version, alfa, desde, hasta, k, con_estrenos=True):
    """Como la app del 13: puntajes estandarizados (por fuente si así lo eligió el 12) y combinados."""
    cat, V, cfg = cargar_catalogo(), cargar_vectores(), cargar_config()
    ESTRENO = estreno()
    anios = cat['anio'].fillna(0).astype(int).values
    qp, qs = codificar(consulta)
    fuente, correccion = cat['fuente'].values, cfg.get('correccion', 'global')
    zp = normalizar((V[('poster', version)] @ qp)[None], fuente, correccion)[0]
    zs = normalizar((V[('sinopsis', version)] @ qs)[None], fuente, correccion)[0]
    puntaje = {'Póster': zp, 'Sinopsis': zs}.get(modo, alfa * zp + (1 - alfa) * zs)
    puntaje = np.where((anios >= desde) & (anios <= hasta) & (con_estrenos | ~ESTRENO), puntaje, -np.inf)
    top = [int(i) for i in np.argsort(-puntaje)[:k] if np.isfinite(puntaje[i])]
    por_poster = {i: bool(alfa * zp[i] > (1 - alfa) * zs[i]) for i in top}
    return top, por_poster


# ============================================================================= películas
def presentar_sinopsis(texto):
    """MM-IMDB trae las sinopsis en minúsculas: mayúscula al inicio de cada oración y en "i"."""
    texto = (texto or '').strip()
    if not texto:
        return 'Esta película no tiene sinopsis en el dataset.'
    texto = re.sub(r'\bi\b', 'I', texto)
    return re.sub(r'(^|[.!?]\s+)([a-záéíóúñ])', lambda m: m.group(1) + m.group(2).upper(), texto)


def imagen(i):
    from PIL import Image
    return Image.open(io.BytesIO(cargar_catalogo().iloc[i]['jpg']))


def anio_de(r):
    return int(r['anio']) if pd.notna(r['anio']) else '?'


def sinopsis_de(i):
    r = cargar_catalogo().iloc[i]
    if r['fuente'] == 'TMDB':
        return r.get('sinopsis', '') or 'Esta película no tiene sinopsis en TMDB.'
    return presentar_sinopsis(r.get('sinopsis', ''))


NOTA_PARTICION = {
    'train': 'El modelo **vio esta película al entrenar**: que acierte aquí no prueba nada.',
    'dev': 'Película de **validación**: se usó para calibrar y elegir umbrales, no para entrenar.',
    'test': 'Película de **prueba**: el modelo nunca la vio.',
    'reciente': 'Estreno 2020-2026 de TMDB: el modelo **nunca vio** películas de esta época.',
}


def nombre_pelicula(i):
    r = cargar_catalogo().iloc[i]
    return f'{r["titulo"]} ({anio_de(r)})'


def elegir_pelicula(clave='pelicula'):
    """Selector compartido por Clasificador y Desbalance (la elección se recuerda entre páginas)."""
    cat = cargar_catalogo()
    rng = np.random.default_rng()
    grupos = {'Prueba (MM-IMDB)': cat['particion'].values == 'test',
              'Estrenos 2020-2026': cat['fuente'].values == 'TMDB',
              'Todas': np.ones(len(cat), bool)}
    grupos = {k: v for k, v in grupos.items() if v.any()}
    if clave not in st.session_state:
        st.session_state[clave] = int(rng.choice(np.where(next(iter(grupos.values())))[0]))

    c1, c2, c3 = st.columns([1.2, 2.2, 0.8])
    grupo = c1.radio('De dónde', list(grupos), horizontal=False, key=f'{clave}_grupo')
    texto = c2.text_input('Buscar por título', key=f'{clave}_texto', placeholder='Ej.: alien, love, night...')
    mascara = grupos[grupo]
    if texto.strip():
        mascara = mascara & cat['titulo'].str.contains(re.escape(texto.strip()), case=False, na=False).values
    opciones = np.where(mascara)[0][:300].tolist()
    if c3.button('🎲 Al azar', width='stretch', key=f'{clave}_azar') and mascara.any():
        st.session_state[clave] = int(rng.choice(np.where(mascara)[0]))
    actual = st.session_state[clave]
    if actual not in opciones:
        opciones = [actual] + opciones
    elegida = c2.selectbox('Película', opciones, index=opciones.index(actual), format_func=nombre_pelicula,
                           key=f'{clave}_lista_{actual}', label_visibility='collapsed')
    st.session_state[clave] = int(elegida)
    return int(elegida)


def ficha_pelicula(i, ancho=170, con_sinopsis=True):
    """Póster, título, géneros reales y en qué partición está."""
    cat = cargar_catalogo()
    r = cat.iloc[i]
    izq, der = st.columns([1, 3.2])
    izq.image(imagen(i), width=ancho)
    with der:
        st.markdown(f'### {html.escape(str(r["titulo"]))}')
        etiquetas = ''.join(f'<span class="etiqueta gris">{html.escape(g)}</span> '
                            for g in (r['generos'] or '').split(',') if g)
        st.markdown(f'<div class="tarjeta-meta">{anio_de(r)} · {r["fuente"]}</div>'
                    f'<div>{etiquetas or "<span class=tarjeta-meta>sin géneros</span>"}</div>', unsafe_allow_html=True)
        nota = NOTA_PARTICION.get(r['particion'])
        if nota:
            st.caption(nota)
        if con_sinopsis:
            with st.expander('Sinopsis'):
                st.write(sinopsis_de(i))
    return r


# ============================================================================= clasificador y compuertas
@st.cache_resource(show_spinner='Cargando el clasificador...')
def cargar_clasificador():
    """Logits precalculados, umbrales, cabezas del 06 y compuertas del C1/C2 (reconstruidas con src/)."""
    import torch
    from src.fusion import CompuertaIncertidumbre
    from src.modelos import Clasificador

    d, cfg = carpeta_datos(), cargar_config().get('clasificador')
    if not cfg or not existe('clasificador.npz'):
        return None
    z = np.load(os.path.join(d, 'clasificador.npz'), allow_pickle=False)
    out = {'cfg': cfg, 'generos': [str(g) for g in z['generos']],
           'Z': {k[7:]: z[k].astype(np.float32) for k in z.files if k.startswith('logits_')},
           'umbral': {k[7:]: z[k].astype(np.float32) for k in z.files if k.startswith('umbral_')}}

    n_salidas = len(cfg['generos_entrenamiento'])
    cabezas = {}
    for nombre, mods in cfg['modalidades'].items():
        estado = {k.split('__', 2)[2]: torch.from_numpy(z[k].copy()) for k in z.files
                  if k.startswith(f'cabeza__{nombre}__')}
        if estado:
            m = Clasificador([cfg['dims'][x] for x in mods], cfg['dim_proy'], cfg['dropout'], n_salidas)
            m.load_state_dict(estado)
            cabezas[nombre] = m.eval()
    out['cabezas'] = cabezas

    compuertas = {}
    for met, info in cfg['compuertas'].items():
        estado = {k.split('__', 2)[2]: torch.from_numpy(z[k].copy()) for k in z.files
                  if k.startswith(f'compuerta__{met}__')}
        if estado:
            m = CompuertaIncertidumbre(len(info['entradas']), len(out['generos']), info['incertidumbre'])
            m.load_state_dict(estado)
            compuertas[met] = m.eval()
    out['compuertas'] = compuertas
    out['X'] = {m: np.load(os.path.join(d, f'clasif_{m}.npy'), mmap_mode='r')
                for m in ('img', 'txt') if existe(f'clasif_{m}.npy')}
    return out


def requiere_clasificador():
    C = cargar_clasificador()
    if C is None:
        st.warning('Esta página necesita `clasificador.npz`: vuelve a correr `scripts/exportar_app.py` con los '
                   'artefactos del 06 y del C1 (`exp2_hoy_artefactos.pt`, `exp4_compuertas.pt`).')
        st.stop()
    return C


def etiquetas_de(i, generos):
    gs = set((cargar_catalogo().iloc[i]['generos'] or '').split(','))
    return np.array([g in gs for g in generos], dtype=np.float32)


def sigmoide(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def degradar(x, tipo, rho, eps):
    """Las degradaciones del C2 sobre un embedding estandarizado: borrado (encoger hacia la media de
    train), ruido y mezcla con otra película (eps = ruido N(0, I) o el embedding de la otra)."""
    if tipo == 'borrado':
        return ((1 - rho) * x).astype(np.float32)
    return (np.sqrt(1 - rho) * x + np.sqrt(rho) * eps).astype(np.float32)


def logits_vivos(C, X):
    """X: {'img': (n, 512), 'txt': (n, 768)} estandarizados -> logits calibrados (n, G) de cada cabeza.
    Mismo cálculo que el exportador: salida de la cabeza en los géneros del C1 dividida por su temperatura."""
    import torch
    from src.modelos import predecir_logits
    cfg, cols = C['cfg'], np.array(C['cfg']['columnas'])
    out = {}
    for nombre, modelo in C['cabezas'].items():
        xs = [torch.from_numpy(np.ascontiguousarray(X[m], dtype=np.float32)) for m in cfg['modalidades'][nombre]]
        out[nombre] = predecir_logits(modelo, xs)[:, cols] / cfg['temperaturas'][nombre]
    return out


def aplicar_compuerta(C, met, Z):
    """Z: {'imagen', 'texto', 'fusion_ingenua'} -> (logits (n, G), pesos (n, G, M)) con src.fusion.combinar."""
    from src.fusion import combinar
    entradas = C['cfg']['compuertas'][met]['entradas']
    return combinar(C['compuertas'][met], np.stack([Z[n] for n in entradas], axis=-1))
