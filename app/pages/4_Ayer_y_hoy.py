"""Películas de ayer y de hoy: resultados de los notebooks 08, 12 y C1-C3 leídos de results/*.csv.

Busca cada archivo en results/ y results/buscador/ del repo y, si no está, en results/ y resultados/ de la
carpeta de datos (la exportación copia ahí los CSV de Drive). Lo que falte se omite con un aviso.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import comun
from comun import AMBAR, BLANCO, GRIS, RAIZ, ROJO

comun.configurar_pagina('Ayer y hoy · películas')
comun.encabezado('Películas de', 'ayer y de hoy')


def ruta(nombre):
    d = comun.carpeta_datos()
    for carpeta in [f'{RAIZ}/results', f'{RAIZ}/results/buscador', f'{d}/results', f'{d}/resultados']:
        if os.path.exists(f'{carpeta}/{nombre}'):
            return f'{carpeta}/{nombre}'
    return None


@st.cache_data(show_spinner=False)
def leer(nombre, **kw):
    r = ruta(nombre)
    if r is None:
        return None
    if nombre.endswith('.json'):
        with open(r, encoding='utf-8') as fh:
            return json.load(fh)
    return pd.read_csv(r, **kw)


def falta(*nombres):
    st.info('Para esta sección faltan ' + ', '.join(f'`{n}`' for n in nombres) +
            ': cópialos de `results/` en Drive a `results/` del repo (o vuelve a correr la exportación).')


GRUPO = {'test_mmimdb': 'Test MM-IMDB', 'control_mmimdb': 'Control (MM-IMDB)', 'control_tmdb': 'Control (TMDB)',
         'recientes_tmdb': 'Recientes 2020-26', 'dif_en_dif': 'Recientes − control'}
RAMA = {'imagen': ('Póster', ROJO), 'texto': ('Sinopsis', BLANCO), 'fusion': ('Fusión', AMBAR),
        'fusion_ingenua': ('Fusión', AMBAR), 'ogm_ge': ('OGM-GE', GRIS)}
CONFIG = {'displayModeBar': False}


def puntos_con_ic(filas, titulo, alto=330, eje='puntos de AP'):
    """filas: lista de (grupo_x, serie, color, valor, inf, sup), en puntos -> puntos con barras de error."""
    fig = go.Figure()
    for serie in dict.fromkeys(f[1] for f in filas):
        fs = [f for f in filas if f[1] == serie]
        fig.add_scatter(x=[f[0] for f in fs], y=[f[3] for f in fs], name=serie, mode='markers',
                        marker=dict(size=11, color=fs[0][2], line=dict(color='#000', width=1)),
                        error_y=dict(type='data', symmetric=False, array=[f[5] - f[3] for f in fs],
                                     arrayminus=[f[3] - f[4] for f in fs], color=fs[0][2], thickness=1.5, width=6))
    fig.add_hline(y=0, line=dict(color='#666', width=1))
    fig.update_layout(scattermode='group', scattergap=0.5)
    fig.update_yaxes(title=eje)
    return comun.estilo_plotly(fig, alto=alto, titulo=titulo)


st.write('Todos los modelos se entrenaron con películas de MM-IMDB (hasta 2016) y se probaron, **sin reentrenar**, '
         'con películas de 2020 a 2026 de TMDB. El grupo de **control** son películas viejas descargadas también de '
         'TMDB: comparar control con MM-IMDB mide el efecto de la **fuente**; recientes con control, el de la **época**.')

tab_clasif, tab_busq, tab_metodos, tab_fallos = st.tabs(
    ['Clasificación (08)', 'Búsqueda (12)', '¿Se puede corregir? (C1-C3)', 'Fallos por género'])

# ============================================================================= 08
with tab_clasif:
    desc, sem = leer('exp3_descomposicion.csv'), leer('exp3_por_semilla.csv')
    if desc is None or sem is None:
        falta('exp3_descomposicion.csv', 'exp3_por_semilla.csv')
    else:
        d = desc.set_index(['efecto', 'medida'])
        dep = sem.groupby('grupo')[['indice_dependencia_imagen', 'cmi_img_dado_txt']].mean()
        comun.kpis([(f"{100 * d.loc[('epoca', 'ap_texto'), 'diferencia']:+.1f}", 'puntos de AP pierde la sinopsis con la época'),
                    (f"{100 * d.loc[('epoca', 'ap_imagen'), 'diferencia']:+.1f}", 'puntos de AP cambia el póster'),
                    (f"{dep.loc['control_tmdb', 'indice_dependencia_imagen']:.2f} → "
                     f"{dep.loc['recientes_tmdb', 'indice_dependencia_imagen']:.2f}",
                     'dependencia de la fusión en el póster (control → recientes)'),
                    (f"{dep.loc['control_tmdb', 'cmi_img_dado_txt']:.2f} → {dep.loc['recientes_tmdb', 'cmi_img_dado_txt']:.2f}",
                     'nats de información que aporta el póster dado el texto')])
        c1, c2 = st.columns([1.1, 1])
        filas = [(('Fuente' if e == 'fuente' else 'Época'), RAMA[m][0], RAMA[m][1],
                  100 * d.loc[(e, f'ap_{m}'), 'diferencia'], 100 * d.loc[(e, f'ap_{m}'), 'ic_inf'],
                  100 * d.loc[(e, f'ap_{m}'), 'ic_sup'])
                 for e in ['fuente', 'epoca'] for m in ['imagen', 'texto', 'fusion']]
        c1.plotly_chart(puntos_con_ic(filas, 'Cambio de AP macro, IC 95% (bootstrap)'), width='stretch', config=CONFIG)
        ap = sem.groupby('grupo')[['ap_imagen', 'ap_texto', 'ap_fusion']].mean().reindex(list(GRUPO)[:4])
        fig = go.Figure()
        for m in ['imagen', 'texto', 'fusion']:
            fig.add_bar(x=[GRUPO[g] for g in ap.index], y=100 * ap[f'ap_{m}'], name=RAMA[m][0], marker_color=RAMA[m][1])
        fig.update_layout(barmode='group')
        fig.update_yaxes(title='AP macro (puntos)')
        c2.plotly_chart(comun.estilo_plotly(fig, alto=330, titulo='AP por grupo (media de 5 semillas)'),
                        width='stretch', config=CONFIG)
        st.markdown(comun.ficha('Qué significa',
                                'Cambiar de fuente (mismas películas, póster y sinopsis de TMDB) no mueve el AP. Cambiar '
                                'de época sí: la <b>sinopsis</b> pierde ~4 puntos y el <b>póster</b> se queda igual. La '
                                'fusión pasa a apoyarse más en la imagen. <b>El desbalance entre modalidades cambia con la '
                                'época de las películas.</b>'), unsafe_allow_html=True)

# ============================================================================= 12
with tab_busq:
    res = leer('recientes_resumen.json')
    if res is None:
        falta('recientes_resumen.json')
    else:
        m_ = res['map_despues']
        comun.kpis([(f"{m_['control_tmdb']['ambos']:.3f} → {m_['recientes_tmdb']['ambos']:.3f}",
                     'mAP de póster + sinopsis con adaptador (control → recientes)'),
                    (f"{100 * res['epoca']['ambos']['diferencia']:+.1f}", 'puntos de AP por la época (IC con el 0)'),
                    (f"{res['mejor_alfa_despues']['recientes_tmdb']:.1f}", 'mejor peso del póster en todos los grupos')])
        c1, c2 = st.columns([1.1, 1])
        filas = [(('Fuente' if e == 'fuente' else 'Época'), RAMA[k][0], RAMA[k][1],
                  100 * res[e][m]['diferencia'], 100 * res[e][m]['ic'][0], 100 * res[e][m]['ic'][1])
                 for e in ['fuente', 'epoca'] for m, k in [('poster', 'imagen'), ('sinopsis', 'texto'), ('ambos', 'fusion')]]
        c1.plotly_chart(puntos_con_ic(filas, 'Cambio de AP en la búsqueda, con adaptador (IC 95%)'),
                        width='stretch', config=CONFIG)
        sesgo = res.get('sesgo', {})
        corr = [('global', 'z-score global'), ('por_fuente', 'z-score por fuente (la app)'),
                ('por_fuente_media', 'solo restar la media')]
        fig = go.Figure()
        for modo, (nombre, color) in [('poster', RAMA['imagen']), ('sinopsis', RAMA['texto']), ('ambos', RAMA['fusion'])]:
            fig.add_bar(x=[c[1] for c in corr], y=[sesgo.get(f'app|{c[0]}|{modo}', {}).get('cociente', np.nan) for c in corr],
                        name=nombre, marker_color=color)
        fig.add_hline(y=1, line=dict(color='#666', dash='dot'))
        fig.update_layout(barmode='group')
        fig.update_yaxes(title='veces lo que les corresponde')
        c2.plotly_chart(comun.estilo_plotly(fig, alto=330, titulo='¿Cuánto salen los estrenos en el top 50?'),
                        width='stretch', config=CONFIG)
        st.markdown(comun.ficha('Qué significa',
                                'A diferencia de la clasificación, la búsqueda <b>no envejece</b>: con el adaptador las '
                                'recientes se encuentran igual de bien que el control. Lo que sí apareció al mezclar las '
                                'dos fuentes en un catálogo es un sesgo de escala: con el z-score global los estrenos '
                                'salían 1.7 veces más de lo que les toca. Estandarizar <b>dentro de cada fuente</b> lo '
                                'baja a ~1.16 y cuesta 0.002 de mAP.'), unsafe_allow_html=True)

# ============================================================================= C1-C3
with tab_metodos:
    ORIGEN = {'tardia': 'C1', 'compuerta': 'C1', 'tardia3': 'C1', 'compuerta3': 'C1', 'tardia_aum': 'C2',
              'compuerta_aum': 'C2', 'ogm': 'C3', 'ogm_ge': 'C3'}
    IDEA = {'tardia': 'pesos fijos por género', 'compuerta': 'más peso a la rama segura en cada película',
            'tardia3': 'tardía + fusión ingenua como 3.ª fuente', 'compuerta3': 'compuerta + fusión ingenua',
            'tardia_aum': 'tardía ajustada con copias degradadas', 'compuerta_aum': 'compuerta con copias degradadas',
            'ogm': 'frenar el gradiente de la modalidad dominante', 'ogm_ge': 'OGM + ruido en el gradiente'}
    boot = leer('exp6_bootstrap.csv')
    fuente_tabla = 'exp6_bootstrap.csv (C3, todos los métodos)'
    if boot is None:
        boot = leer('exp4_bootstrap.csv')
        if boot is not None:
            boot = boot[boot['comparacion'].str.endswith('− ingenua')].copy()
            boot['metodo'] = boot['comparacion'].str.split(' ').str[0]
            boot.loc[boot['grupo'] == 'dif_en_dif', 'grupo'] = 'dif_en_dif'
            fuente_tabla = 'exp4_bootstrap.csv (solo métodos del C1: falta exp6_bootstrap.csv)'
    st.markdown('### Tabla final: todos los métodos contra la fusión ingenua')
    if boot is None:
        falta('exp6_bootstrap.csv', 'exp4_bootstrap.csv')
    else:
        b = boot[boot['medida'] == 'ap_macro'].copy()
        b['sin_cero'] = (b['ic_inf'] > 0) | (b['ic_sup'] < 0)
        b['texto'] = b.apply(lambda r: f"{100 * r.diferencia:+.2f} [{100 * r.ic_inf:+.2f}, {100 * r.ic_sup:+.2f}]"
                                       f"{' *' if r.sin_cero else ''}", axis=1)
        metodos = [m for m in ORIGEN if m in set(b['metodo'])]
        grupos = [g for g in GRUPO if g in set(b['grupo'])]
        tabla = b.pivot(index='metodo', columns='grupo', values='texto').reindex(index=metodos, columns=grupos)
        tabla.columns = [GRUPO[g] for g in tabla.columns]
        tabla.insert(0, 'idea', [IDEA[m] for m in tabla.index])
        tabla.insert(0, 'notebook', [ORIGEN[m] for m in tabla.index])
        st.dataframe(tabla, width='stretch')
        st.caption(f'Puntos de AP macro, método − fusión ingenua, bootstrap 95% por películas (* = IC sin el 0). '
                   f'Fuente: {fuente_tabla}.')
        COL_MET = {'C1': ROJO, 'C2': BLANCO, 'C3': AMBAR}
        filas = [(GRUPO[g], m, COL_MET[ORIGEN[m]], 100 * r.diferencia, 100 * r.ic_inf, 100 * r.ic_sup)
                 for m in metodos for g in ['test_mmimdb', 'recientes_tmdb']
                 for _, r in b[(b.metodo == m) & (b.grupo == g)].iterrows()]
        fig = puntos_con_ic(filas, 'Ganancia sobre la fusión ingenua (rojo C1, blanco C2, ámbar C3)', alto=380)
        for t in fig.data:
            t.marker.symbol = {'C1': 'circle', 'C2': 'square', 'C3': 'diamond'}[ORIGEN[t.name]]
        st.plotly_chart(fig, width='stretch', config=CONFIG)
        st.markdown(comun.ficha('Qué significa',
                                'Solo los métodos que suman la fusión ingenua como tercera fuente mejoran con '
                                'significancia en test. <b>Ningún método mejora con significancia en las recientes</b>, ni '
                                'gana más en las recientes que en el control.'), unsafe_allow_html=True)

    st.markdown('### C1 · La entropía no avisa')
    sorp = leer('exp4_sorpresa.csv')
    if sorp is None:
        falta('exp4_sorpresa.csv')
    else:
        c1, c2 = st.columns([1.2, 1])
        s = sorp[sorp['conjunto'] != 'dev_mmimdb']
        fig = go.Figure()
        for modelo in ['imagen', 'texto', 'fusion_ingenua']:
            d = s[s['modelo'] == modelo].set_index('conjunto').reindex(list(GRUPO)[:4])
            fig.add_bar(x=[GRUPO[g] for g in d.index], y=d['exceso'], name=RAMA[modelo][0], marker_color=RAMA[modelo][1])
        fig.update_layout(barmode='group')
        fig.update_yaxes(title='nats por celda')
        c1.plotly_chart(comun.estilo_plotly(fig, alto=320, titulo='Exceso de sorpresa = log-loss − entropía'),
                        width='stretch', config=CONFIG)
        t = s[s['modelo'] == 'texto'].set_index('conjunto')
        c2.markdown(comun.ficha('Error seguro',
                                f'Con las recientes, la entropía del texto casi no cambia '
                                f'({t.loc["control_mmimdb", "entropia_media"]:.3f} → '
                                f'{t.loc["recientes_tmdb", "entropia_media"]:.3f}), pero su exceso pasa de '
                                f'{t.loc["control_mmimdb", "exceso"]:.3f} a {t.loc["recientes_tmdb", "exceso"]:.3f}. '
                                'El texto <b>se equivoca sin anunciarlo</b>, así que una compuerta que mira la entropía '
                                'no tiene cómo enterarse.'), unsafe_allow_html=True)

    st.markdown('### C2 · El dial de degradación')
    dial = leer('exp5_dial.csv')
    if dial is None:
        falta('exp5_dial.csv')
        st.caption('Puedes probar el dial con una sola película en la página Desbalance.')
    else:
        a1, a2 = st.columns(2)
        mod = a1.radio('Modalidad dañada', ['texto', 'imagen'], format_func=lambda m: RAMA[m][0], horizontal=True, key='c2_mod')
        tipos = [t for t in ['borrado', 'ruido', 'mezcla'] if t in set(dial['tipo'])]
        tipo = a2.radio('Tipo', tipos, horizontal=True, key='c2_tipo')
        d = dial[(dial['modalidad'] == mod) & (dial['tipo'] == tipo)]
        c1, c2 = st.columns(2)
        fig = go.Figure()
        COL_D = {mod: GRIS, 'ingenua': AMBAR, 'tardia': '#a3a3a3', 'compuerta': ROJO, 'tardia_aum': '#525252',
                 'compuerta_aum': BLANCO}
        for met, color in COL_D.items():
            g = d[d['metodo'] == met].groupby('rho')['ap_macro'].mean()
            if len(g):
                fig.add_scatter(x=g.index, y=100 * g.values, mode='lines+markers', name=f'solo {RAMA[met][0].lower()}'
                                if met == mod else met, line=dict(color=color, dash='dash' if met == mod else 'solid'))
        fig.update_xaxes(title='ρ')
        fig.update_yaxes(title='AP macro (puntos)')
        c1.plotly_chart(comun.estilo_plotly(fig, alto=340, titulo=f'AP en test al dañar la {RAMA[mod][0].lower()} ({tipo})'),
                        width='stretch', config=CONFIG)
        fig = go.Figure()
        r = d[d['metodo'] == mod].groupby('rho')[['entropia', 'exceso']].mean()
        fig.add_scatter(x=r.index, y=r['entropia'], name='entropía (lo que anuncia)', line=dict(color=GRIS))
        fig.add_scatter(x=r.index, y=r['exceso'], name='exceso (lo que no anuncia)', line=dict(color=ROJO))
        fig.update_xaxes(title='ρ')
        fig.update_yaxes(title='nats por celda')
        c2.plotly_chart(comun.estilo_plotly(fig, alto=340, titulo='¿La rama dañada avisa?'), width='stretch', config=CONFIG)
        st.markdown(comun.ficha('Qué significa',
                                'El <b>borrado</b> se anuncia (la entropía sube), el <b>ruido</b> y la <b>mezcla</b> no. La '
                                'mezcla con otra película reproduce el patrón de las recientes: el AP cae con la entropía '
                                'igual. En todo el dial el término de entropía no agrega nada a la fusión tardía.'),
                    unsafe_allow_html=True)

    st.markdown('### C2 · Pocos datos: ¿aparece una modalidad golosa?')
    pocos = leer('exp5_pocos_datos.csv')
    if pocos is None:
        falta('exp5_pocos_datos.csv')
    else:
        pm = pocos.groupby(['fraccion', 'n_train']).mean(numeric_only=True).reset_index().sort_values('fraccion')
        etiqueta_x = [f'{100 * f:g} %' for f in pm['fraccion']]
        hover = [f'{n:,} películas' for n in pm['n_train']]
        c1, c2 = st.columns(2)
        fig = go.Figure()
        for col, nombre, color, guion in [('ap_imagen', 'Solo póster', ROJO, 'dash'), ('ap_texto', 'Solo sinopsis', BLANCO, 'dash'),
                                          ('ap_ingenua', 'Fusión ingenua', AMBAR, 'solid'),
                                          ('ap_tardia', 'Fusión tardía', GRIS, 'dot')]:
            if col in pm:
                fig.add_scatter(x=etiqueta_x, y=100 * pm[col], name=nombre, mode='lines+markers', text=hover,
                                line=dict(color=color, dash=guion),
                                hovertemplate='%{x} de train (%{text}): %{y:.1f}<extra>' + nombre + '</extra>')
        fig.update_xaxes(title='fracción de train', type='category')
        fig.update_yaxes(title='AP macro (puntos)')
        c1.plotly_chart(comun.estilo_plotly(fig, alto=340, titulo='AP en test según cuántos datos de entrenamiento'),
                        width='stretch', config=CONFIG)
        fig = go.Figure()
        fig.add_scatter(x=etiqueta_x, y=100 * pm['ganancia_fusion'], name='fusión − mejor rama sola', mode='lines+markers',
                        line=dict(color=AMBAR), text=hover,
                        hovertemplate='%{x} (%{text}): %{y:+.1f} pts<extra></extra>')
        fig.add_scatter(x=etiqueta_x, y=pm['indice_dependencia_imagen'], name='dependencia del póster (0-1)',
                        mode='lines+markers', line=dict(color=ROJO), yaxis='y2', text=hover,
                        hovertemplate='%{x} (%{text}): %{y:.2f}<extra></extra>')
        fig.update_xaxes(title='fracción de train', type='category')
        fig.update_layout(yaxis=dict(title='ganancia de la fusión (puntos)', rangemode='tozero'),
                          yaxis2=dict(overlaying='y', side='right', showgrid=False, title='dependencia', range=[0, 1]))
        c2.plotly_chart(comun.estilo_plotly(fig, alto=340, titulo='¿La fusión sigue usando las dos?', leyenda='abajo'),
                        width='stretch', config=CONFIG)
        chico = pm.iloc[0]
        st.markdown(comun.ficha('Qué significa',
                                f'Con solo el <b>{100 * chico["fraccion"]:g} %</b> de train ({int(chico["n_train"]):,} '
                                f'películas) la fusión supera a la mejor rama sola por <b>{100 * chico["ganancia_fusion"]:.1f}'
                                f'</b> puntos, y el póster sigue aportando (dependencia {chico["indice_dependencia_imagen"]:.2f}). '
                                'Si hubiera una modalidad golosa, con pocos datos la fusión se pegaría a una sola rama y '
                                'la ganancia caería a cero. No pasa: el problema de las recientes no es de aprendizaje.'),
                    unsafe_allow_html=True)

    st.markdown('### C3 · Modular el entrenamiento (OGM-GE)')
    eq = leer('exp6_equilibrio.csv')
    if eq is None:
        falta('exp6_equilibrio.csv')
    else:
        e = eq.groupby('modelo')[['ap', 'indice_dependencia_imagen']].mean()
        comun.kpis([(f"{e.loc['fusion_ingenua', 'indice_dependencia_imagen']:.2f} → "
                     f"{e.loc['ogm_ge', 'indice_dependencia_imagen']:.2f}",
                     'dependencia del póster: fusión ingenua → OGM-GE'),
                    (f"{100 * e.loc['fusion_ingenua', 'ap']:.1f} → {100 * e.loc['ogm_ge', 'ap']:.1f}", 'AP macro en test')])
        st.markdown(comun.ficha('Qué significa',
                                'OGM-GE casi no actúa: según su propia medida las dos modalidades ya aportan parecido. La '
                                'fusión depende más del texto porque el texto trae más información, no porque la imagen '
                                'haya aprendido menos.'), unsafe_allow_html=True)

# ============================================================================= fallos
with tab_fallos:
    pg = leer('exp6_por_genero.csv', index_col=0)
    origen = 'exp6_por_genero.csv (C3)'
    if pg is None:
        pg = leer('exp3_por_genero.csv', index_col=0)
        origen = 'exp3_por_genero.csv (08)'
    if pg is None:
        falta('exp6_por_genero.csv', 'exp3_por_genero.csv')
    else:
        pg = pg.sort_values('cambio_texto')
        ramas = [m for m in ['imagen', 'texto', 'fusion_ingenua', 'ogm_ge'] if f'cambio_{m}' in pg]
        r = np.corrcoef(pg['cambio_texto'], pg['cambio_fusion_ingenua'])[0, 1]
        fig = go.Figure()
        for m in ramas:
            fig.add_bar(x=pg.index, y=100 * pg[f'cambio_{m}'], name=RAMA[m][0], marker_color=RAMA[m][1])
        fig.update_layout(barmode='group')
        fig.update_yaxes(title='AP recientes − control (puntos)')
        st.plotly_chart(comun.estilo_plotly(fig, alto=400, titulo='Efecto de la época por género'),
                        width='stretch', config=CONFIG)
        peores = ', '.join(f'{g} ({100 * v:+.0f})' for g, v in pg['cambio_texto'].head(5).items())
        st.markdown(comun.ficha('La fusión hereda el fallo del texto',
                                f'Correlación entre géneros del cambio del texto y el de la fusión: <b>r = {r:.2f}</b>. '
                                f'Lo que más pierde el texto: {peores}.'), unsafe_allow_html=True)
        st.caption(f'Fuente: {origen}.')
    fallos = leer('exp6_fallos.csv')
    st.markdown('### Las recientes donde el texto se equivoca con más seguridad')
    if fallos is None:
        falta('exp6_fallos.csv')
    else:
        st.dataframe(fallos, hide_index=True, width='stretch')
        st.caption('Entre las recientes con la entropía del texto por debajo de la mediana, las de mayor log-loss del '
                   'texto. El texto cae en géneros "por defecto" (Drama, Romance, Comedy) y la imagen acierta más.')

comun.creditos()
