"""Exporta todo lo que necesita la app (carpeta `app_buscador/` en Drive).

Reemplaza la sección de exportación del notebook 13. Pensado para correr en Colab con el Drive montado:

    !python scripts/exportar_app.py                      # usa /content/drive/MyDrive/proyecto_multimodal
    !python scripts/exportar_app.py --base /otra/carpeta --sin-umap

Genera en `<base>/app_buscador/`:

- Buscador (lo mismo que exportaba el 13): `catalogo.parquet` (MM-IMDB + estrenos 2020-2026, ahora con la
  columna `particion`), `poster_{orig,adapt}.npy`, `sinopsis_{orig,adapt}.npy`, `config.json`, `umap.png`.
- Clasificador y desbalance: `clasificador.npz` con, para cada película del catálogo, los logits calibrados
  (divididos por la temperatura del 06) de imagen, texto y fusión ingenua, y de la fusión con cada rama
  apagada; los umbrales; los pesos de las tres cabezas y de las compuertas del C1 (y del C2 si están).
  `clasif_img.npy` y `clasif_txt.npy`: los embeddings estandarizados (CLIP y DistilBERT) para el dial del C2.
- Espacio 3D: `espacio3d.parquet` con PCA (y UMAP si está instalado) en 3D de una muestra de películas,
  antes y después del adaptador.
- `results/` y `resultados/`: copia de los CSV de clasificación y del buscador para la página de resultados.

Todo se guarda sin pickle (npz, npy, parquet, json), así la app no necesita `torch.load` de archivos bajados.
"""
import argparse
import base64
import json
import os
import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src.buscador import MODELO_POSTER, MODELO_SINOPSIS, l2  # noqa: E402
from src.datos import aplicar_estandarizacion  # noqa: E402
from src.fusion import cargar_compuertas  # noqa: E402
from src.modelos import cargar_artefactos, predecir_logits  # noqa: E402

SEMILLA = 0                 # semilla de los modelos del 06 y de las compuertas que usa la app
ALFA_APP = 0.4              # mejor peso del póster en el notebook 11
GEN_MAPA = ['Horror', 'Animation', 'Documentary', 'Western', 'Romance', 'Sci-Fi']   # mismos del UMAP del 11
N_MAPA_MM, N_MAPA_REC = 2000, 600
URL_NB11 = 'https://raw.githubusercontent.com/AndresAvella09/Proyecto_Multimodal/main/notebooks/11_buscador_adaptado.ipynb'


def tamano(ruta):
    return f'{os.path.getsize(ruta) / 1e6:7.1f} MB'


# ============================================================================= 1. buscador
def exportar_buscador(DATA, EMB, SALIDA):
    """Lo mismo que la sección 1 del notebook 13. Devuelve el catálogo y de dónde sale cada fila."""
    pelis = pd.read_parquet(f'{DATA}/mmimdb_con_anio.parquet')
    mini = pd.read_parquet(f'{DATA}/posters_mini.parquet')
    assert (pelis['fila_hdf5'].values == np.arange(len(pelis))).all() and len(mini) == len(pelis)

    filas = np.where(pelis['es_pelicula'].fillna(False).values.astype(bool))[0]
    cat = pd.DataFrame({
        'fila_hdf5': filas,
        'tconst': pelis['tconst'].values[filas],
        'titulo': pelis['titulo'].fillna('(sin título)').values[filas],
        'anio': pelis['anio'].values[filas],
        'generos': pelis['generos_imdb_hoy'].fillna('').values[filas],
        'sinopsis': pelis['sinopsis'].fillna('').values[filas],
        'jpg': mini['jpg'].values[filas],
        'fuente': 'MM-IMDB',
        'particion': pelis['particion'].values[filas],     # train / dev / test: para no presumir con train
    })

    vectores = {
        'poster_orig.npy': l2(np.load(f'{EMB}/clip_img.npy')),
        'sinopsis_orig.npy': l2(np.load(f'{EMB}/e5_txt.npy')),
        'poster_adapt.npy': l2(np.load(f'{EMB}/poster_adapt.npy')),
        'sinopsis_adapt.npy': l2(np.load(f'{EMB}/e5_adapt.npy')),
    }
    for nombre, Vx in vectores.items():
        assert len(Vx) == len(pelis), nombre
        vectores[nombre] = Vx[filas]

    # Películas de 2020 a 2026 (notebooks 07 y 12). Mismos codificadores y mismos adaptadores del 11.
    RECIENTES = {'meta': f'{DATA}/tmdb_meta.parquet', 'mini': f'{DATA}/tmdb_posters_mini.parquet',
                 'poster_orig.npy': f'{EMB}/tmdb_clip_img.npy', 'sinopsis_orig.npy': f'{EMB}/tmdb_e5_txt.npy',
                 'poster_adapt.npy': f'{EMB}/tmdb_poster_adapt.npy', 'sinopsis_adapt.npy': f'{EMB}/tmdb_e5_adapt.npy'}
    con_recientes = all(os.path.exists(r) for r in RECIENTES.values())
    rec = np.array([], dtype=int)
    if con_recientes:
        tm = pd.read_parquet(RECIENTES['meta'])
        mini_tm = pd.read_parquet(RECIENTES['mini'])
        assert (mini_tm['tconst'].values == tm['tconst'].values).all()
        rec = np.where((tm['grupo'] == 'reciente') & ~tm['tconst'].isin(cat['tconst']))[0]   # el control ya está
        cols_hoy = [c for c in tm.columns if c.startswith('hoy_')]
        generos_tm = [','.join(c[4:] for c in cols_hoy if fila[c] == 1) for _, fila in tm[cols_hoy].iterrows()]
        cat = pd.concat([cat, pd.DataFrame({
            'fila_hdf5': -1,
            'tconst': tm['tconst'].values[rec],
            'titulo': tm['titulo'].fillna('(sin título)').values[rec],
            'anio': tm['anio'].astype(float).values[rec],
            'generos': np.array(generos_tm)[rec],
            'sinopsis': tm['overview'].fillna('').values[rec],     # texto original de TMDB, para mostrar
            'jpg': mini_tm['jpg'].values[rec],
            'fuente': 'TMDB',
            'particion': 'reciente',
        })], ignore_index=True)
        for nombre in vectores:
            Vt = l2(np.load(RECIENTES[nombre]))
            assert len(Vt) == len(tm), nombre
            vectores[nombre] = np.concatenate([vectores[nombre], Vt[rec]])
        print(f'Películas 2020-2026 agregadas: {len(rec):,}')
    else:
        print('Sin películas 2020-2026 (faltan archivos de los notebooks 07 y 12):',
              [r for r in RECIENTES.values() if not os.path.exists(r)])

    cat.to_parquet(f'{SALIDA}/catalogo.parquet', index=False)
    for nombre, Vx in vectores.items():
        assert len(Vx) == len(cat), nombre
        np.save(f'{SALIDA}/{nombre}', Vx.astype(np.float16))   # float16: pesa la mitad y no cambia el orden
    return cat, filas, rec, con_recientes, vectores


def cifras_temporales(BASE):
    """Cifras de la comparación temporal (notebooks 08 y 12), igual que en el 13."""
    temporal = {}
    RES_CLASIF, RES_BUSQ = f'{BASE}/results', f'{BASE}/resultados'
    if os.path.exists(f'{RES_CLASIF}/exp3_descomposicion.csv'):
        e = pd.read_csv(f'{RES_CLASIF}/exp3_descomposicion.csv').query("efecto == 'epoca'").set_index('medida')
        temporal['clasificacion'] = {m: {'diferencia': float(e.loc[m, 'diferencia']),
                                         'ic': [float(e.loc[m, 'ic_inf']), float(e.loc[m, 'ic_sup'])]}
                                     for m in ['ap_imagen', 'ap_texto', 'ap_fusion']}
    if os.path.exists(f'{RES_CLASIF}/exp3_por_semilla.csv'):
        ps = pd.read_csv(f'{RES_CLASIF}/exp3_por_semilla.csv')
        temporal['dependencia'] = ps.groupby('grupo')['indice_dependencia_imagen'].mean().round(3).to_dict()
    if os.path.exists(f'{RES_BUSQ}/recientes_resumen.json'):
        with open(f'{RES_BUSQ}/recientes_resumen.json', encoding='utf-8') as fh:
            temporal['busqueda'] = json.load(fh)
    print('Cifras de la comparación temporal:', list(temporal) or 'ninguna (faltan los notebooks 08 y 12)')
    return temporal


def copiar_figura_umap(SALIDA):
    """Figura UMAP (antes vs. después) del notebook 11 publicado en GitHub, como en el 13."""
    try:
        nb11 = json.loads(urllib.request.urlopen(URL_NB11, timeout=30).read())
        celda = next(c for c in nb11['cells'] if c['cell_type'] == 'code' and 'umap.UMAP' in ''.join(c['source']))
        png = next(o['data']['image/png'] for o in celda['outputs'] if 'image/png' in o.get('data', {}))
        with open(f'{SALIDA}/umap.png', 'wb') as fh:
            fh.write(base64.b64decode(png))
        print('Figura UMAP copiada del notebook 11')
    except Exception as e:  # sin red o sin la figura: la página la omite
        print('No se pudo copiar la figura UMAP (la app la omitirá):', e)


# ============================================================================= 2. clasificador
def exportar_clasificador(DATA, EMB, RES, SALIDA, cat, filas, rec, con_recientes):
    """Logits calibrados por película, umbrales y pesos de las cabezas (06) y de las compuertas (C1, C2)."""
    ruta_art, ruta_c1 = f'{RES}/exp2_hoy_artefactos.pt', f'{RES}/exp4_compuertas.pt'
    if not (os.path.exists(ruta_art) and os.path.exists(ruta_c1)):
        print('Sin clasificador: faltan', [r for r in (ruta_art, ruta_c1) if not os.path.exists(r)])
        return None
    art = cargar_artefactos(ruta_art)
    c1 = cargar_compuertas(ruta_c1)
    G_ENT, G_EVAL, GENEROS = art['generos_entrenamiento'], art['generos_evaluacion'], c1['generos']
    COLS = np.array([G_ENT.index(g) for g in GENEROS])
    s = SEMILLA

    # Embeddings del clasificador (CLIP imagen y DistilBERT) en el mismo orden del catálogo
    X = {'img': np.load(f'{EMB}/clip_img.npy')[filas], 'txt': np.load(f'{EMB}/distilbert_txt.npy')[filas]}
    if con_recientes and len(rec):
        X['img'] = np.concatenate([X['img'], np.load(f'{EMB}/tmdb_clip_img.npy')[rec]])
        X['txt'] = np.concatenate([X['txt'], np.load(f'{EMB}/tmdb_distilbert_txt.npy')[rec]])
    assert all(len(v) == len(cat) for v in X.values())
    Xz = aplicar_estandarizacion(X, art['stats_estandarizacion'])   # estadísticas de train de MM-IMDB

    def logits(nombre, apagar=()):
        # apagar: modalidades reemplazadas por la media de train (vector cero tras estandarizar), como en el 06
        xs = [torch.zeros(len(cat), Xz[m].shape[1]) if m in apagar else torch.from_numpy(Xz[m])
              for m in art['modalidades'][nombre]]
        return predecir_logits(art['modelos'][(nombre, s)], xs)[:, COLS] / art['temperaturas'][f'{nombre}_s{s}']

    out = {
        'generos': np.array(GENEROS),
        'logits_imagen': logits('imagen'),
        'logits_texto': logits('texto'),
        'logits_fusion': logits('fusion_ingenua'),
        'logits_fusion_sin_imagen': logits('fusion_ingenua', ['img']),
        'logits_fusion_sin_texto': logits('fusion_ingenua', ['txt']),
    }
    for k in list(out):
        if k.startswith('logits_'):
            out[k] = out[k].astype(np.float16)

    # Umbrales como probabilidad calibrada. Los del 06 se eligieron sobre sigmoide(logit crudo) en G_EVAL:
    # p_cal = sigmoide(logit(u) / T). Los del C1 ya están sobre logits calibrados y en GENEROS.
    i_eval = np.array([G_EVAL.index(g) for g in GENEROS])
    for nombre, corto in [('imagen', 'imagen'), ('texto', 'texto')]:
        u = np.clip(np.asarray(art['umbrales'][f'{nombre}_s{s}'], dtype=np.float64)[i_eval], 1e-4, 1 - 1e-4)
        T = art['temperaturas'][f'{nombre}_s{s}']
        out[f'umbral_{corto}'] = 1 / (1 + np.exp(-np.log(u / (1 - u)) / T))
    for met, u in c1['umbrales'].items():
        nombre, sem = met.rsplit('_s', 1)
        if int(sem) == s:
            out[f'umbral_{"fusion" if nombre == "ingenua" else nombre}'] = np.asarray(u, dtype=np.float64)

    # Pesos de las cabezas (para el dial del C2, sin pickle): cabeza__<nombre>__<parámetro>
    for nombre in ['imagen', 'texto', 'fusion_ingenua']:
        for k, v in art['modelos'][(nombre, s)].state_dict().items():
            out[f'cabeza__{nombre}__{k}'] = v.cpu().numpy()

    # Compuertas del C1 y, si existen, las del C2 (ajustadas con degradación sintética)
    compuertas = {met: {'entradas': c1['entradas'][met], 'incertidumbre': bool(c1['incertidumbre'][met])}
                  for met in c1['entradas']}
    fuentes = [(c1, list(c1['entradas']))]
    ruta_c2 = f'{RES}/exp5_compuertas_aum.pt'
    if os.path.exists(ruta_c2):
        c2 = cargar_compuertas(ruta_c2)
        assert c2['generos'] == GENEROS
        fuentes.append((c2, list(c2['entradas'])))
        compuertas.update({met: {'entradas': c2['entradas'][met], 'incertidumbre': bool(c2['incertidumbre'][met])}
                           for met in c2['entradas']})
    for c, metodos in fuentes:
        for met in metodos:
            for k, v in c['compuertas'][(met, s)].state_dict().items():
                out[f'compuerta__{met}__{k}'] = v.cpu().numpy()

    np.savez_compressed(f'{SALIDA}/clasificador.npz', **out)
    np.save(f'{SALIDA}/clasif_img.npy', Xz['img'].astype(np.float16))
    np.save(f'{SALIDA}/clasif_txt.npy', Xz['txt'].astype(np.float16))

    print(f'Clasificador: {len(GENEROS)} géneros, semilla {s}, compuertas {sorted(compuertas)}')
    return {
        'semilla': s,
        'generos': GENEROS,
        'generos_entrenamiento': G_ENT,
        'columnas': COLS.tolist(),
        'modalidades': art['modalidades'],
        'dims': {m: int(Xz[m].shape[1]) for m in Xz},
        'dim_proy': art['config']['dim_proy'],
        'dropout': art['config']['dropout'],
        'temperaturas': {n: float(art['temperaturas'][f'{n}_s{s}']) for n in ['imagen', 'texto', 'fusion_ingenua']},
        'compuertas': compuertas,
    }


# ============================================================================= 3. espacio 3D
def exportar_espacio3d(cat, vectores, SALIDA, con_umap=True):
    """PCA (y UMAP) 3D de una muestra con un solo género de GEN_MAPA: test de MM-IMDB y estrenos."""
    rng = np.random.default_rng(42)
    gen = cat['generos'].fillna('').str.split(',')
    cuenta = gen.apply(lambda g: [x for x in g if x in GEN_MAPA])
    uno = cuenta.apply(len) == 1
    mm = np.where(uno & (cat['particion'] == 'test'))[0]
    rc = np.where(uno & (cat['fuente'] == 'TMDB'))[0]
    sel = np.concatenate([rng.permutation(mm)[:N_MAPA_MM], rng.permutation(rc)[:N_MAPA_REC]])
    if len(sel) < 20:
        print('Espacio 3D: muy pocas películas para el mapa, se omite')
        return None
    df = pd.DataFrame({'idx': sel, 'genero': [cuenta.iloc[i][0] for i in sel], 'fuente': cat['fuente'].values[sel]})

    umap = None
    if con_umap:
        try:
            import umap  # noqa: F401
        except ImportError:
            print('umap-learn no está instalado: solo PCA (en Colab: pip install umap-learn)')
            umap = None
    for espacio in ['poster_orig', 'poster_adapt', 'sinopsis_orig', 'sinopsis_adapt']:
        V = vectores[f'{espacio}.npy'][sel].astype(np.float32)
        Vc = V - V.mean(0)
        _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
        xyz = Vc @ Vt[:3].T
        for j, eje in enumerate('xyz'):
            df[f'{espacio}_pca_{eje}'] = xyz[:, j].astype(np.float32)
        if umap is not None:
            xyz = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine',
                            random_state=42).fit_transform(V)
            for j, eje in enumerate('xyz'):
                df[f'{espacio}_umap_{eje}'] = xyz[:, j].astype(np.float32)
    df.to_parquet(f'{SALIDA}/espacio3d.parquet', index=False)
    metodos = ['pca'] + (['umap'] if umap is not None else [])
    print(f'Espacio 3D: {len(df):,} películas ({(df.fuente == "TMDB").sum()} estrenos), métodos {metodos}')
    return {'generos': GEN_MAPA, 'metodos': metodos, 'n': int(len(df))}


# ============================================================================= 4. resultados
def copiar_resultados(BASE, SALIDA):
    """Copia los CSV/JSON de results/ (clasificación) y resultados/ (buscador) para la página de resultados."""
    copiados = 0
    for origen, destino, prefijos in [('results', 'results', ('exp',)), ('resultados', 'resultados', ('',))]:
        if not os.path.isdir(f'{BASE}/{origen}'):
            continue
        os.makedirs(f'{SALIDA}/{destino}', exist_ok=True)
        for f in sorted(os.listdir(f'{BASE}/{origen}')):
            if f.endswith(('.csv', '.json')) and f.startswith(prefijos):
                shutil.copy2(f'{BASE}/{origen}/{f}', f'{SALIDA}/{destino}/{f}')
                copiados += 1
    print(f'Resultados copiados: {copiados} archivos')


# ============================================================================= principal
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--base', default=os.environ.get('PM_BASE', '/content/drive/MyDrive/proyecto_multimodal'),
                   help='carpeta proyecto_multimodal (con data/, embeddings/, results/, resultados/)')
    p.add_argument('--salida', default=None, help='carpeta de salida (por defecto <base>/app_buscador)')
    p.add_argument('--sin-umap', action='store_true', help='solo PCA en el espacio 3D (más rápido)')
    p.add_argument('--sin-figura', action='store_true', help='no descargar la figura UMAP del notebook 11')
    a = p.parse_args(argv)

    BASE = a.base
    DATA, EMB, RES = f'{BASE}/data', f'{BASE}/embeddings', f'{BASE}/results'
    if not os.path.exists(f'{EMB}/clip_img.npy') and os.path.exists(f'{EMB}/embeddings/clip_img.npy'):
        EMB = f'{EMB}/embeddings'
    SALIDA = a.salida or f'{BASE}/app_buscador'
    os.makedirs(SALIDA, exist_ok=True)
    print('Vectores en:', EMB, '| Salida:', SALIDA)

    cat, filas, rec, con_recientes, vectores = exportar_buscador(DATA, EMB, SALIDA)
    temporal = cifras_temporales(BASE)
    clasif = exportar_clasificador(DATA, EMB, RES, SALIDA, cat, filas, rec, con_recientes)
    espacio = exportar_espacio3d(cat, vectores, SALIDA, con_umap=not a.sin_umap)
    copiar_resultados(BASE, SALIDA)
    if not a.sin_figura:
        copiar_figura_umap(SALIDA)

    anios = cat['anio'].dropna().astype(int)
    config = {'modelo_poster': MODELO_POSTER,
              'modelo_sinopsis': MODELO_SINOPSIS,
              'alfa': ALFA_APP,
              'anio_min': int(anios.min()), 'anio_max': int(anios.max()),
              'n_peliculas': int(len(cat)), 'n_recientes': int((cat['fuente'] == 'TMDB').sum()),
              'temporal': temporal,
              # cómo estandarizar los puntajes cuando el catálogo mezcla MM-IMDB y TMDB (elegido en el 12)
              'correccion': temporal.get('busqueda', {}).get('correccion_app', 'global') if con_recientes else 'global',
              'clasificador': clasif,
              'espacio3d': espacio}
    with open(f'{SALIDA}/config.json', 'w', encoding='utf-8') as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    print('Corrección de escala entre fuentes:', config['correccion'])

    print(f'\n{len(cat):,} películas | años {config["anio_min"]}-{config["anio_max"]}')
    for f in sorted(os.listdir(SALIDA)):
        if os.path.isfile(f'{SALIDA}/{f}'):
            print(f'  {f:<24} {tamano(f"{SALIDA}/{f}")}')
    return config


if __name__ == '__main__':
    main()
