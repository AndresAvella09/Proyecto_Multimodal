"""Carga de embeddings precalculados y etiquetas de MM-IMDB."""
import numpy as np
import pandas as pd

ARCHIVOS = {'img': 'clip_img.npy', 'txt': 'distilbert_txt.npy', 'clip_txt': 'clip_txt.npy'}
PARTICIONES = ['train', 'dev', 'test']


def cargar(data_dir, emb_dir, modalidades=('img', 'txt'), etiquetas='hoy'):
    """Devuelve X (modalidad -> (N, d)), Y (N, 23), idx (partición -> filas), géneros y meta.

    etiquetas='originales': géneros de MM-IMDB (hasta 10 por película), desde mmimdb_meta.parquet.
    etiquetas='hoy': géneros actuales de IMDb (hasta 3), desde mmimdb_con_anio.parquet
        (notebook 05). Las películas sin etiquetas de hoy quedan fuera de idx.
    meta: DataFrame alineado con las filas (año, votos, tipo de título, etc.).
    """
    if etiquetas == 'originales':
        meta = pd.read_parquet(f'{data_dir}/mmimdb_meta.parquet')
        generos = list(meta.columns[4:27])
        Y = meta[generos].values.astype(np.float32)
        validas = np.ones(len(meta), dtype=bool)
    elif etiquetas == 'hoy':
        meta = pd.read_parquet(f'{data_dir}/mmimdb_con_anio.parquet')
        generos = list(meta.columns[4:27])
        Y = meta[[f'hoy_{g}' for g in generos]].values.astype(np.float32)
        validas = meta['etiquetas_hoy_ok'].values.astype(bool)
    else:
        raise ValueError(f"etiquetas debe ser 'originales' u 'hoy', no {etiquetas!r}")

    assert (meta['fila_hdf5'].values == np.arange(len(meta))).all(), "el parquet debe estar en orden del HDF5"
    X = {m: np.load(f'{emb_dir}/{ARCHIVOS[m]}') for m in modalidades}
    assert all(len(v) == len(meta) for v in X.values())
    idx = {s: np.where((meta['particion'] == s).values & validas)[0] for s in PARTICIONES}
    return X, Y, idx, generos, meta


def ajustar_estandarizacion(X, idx_train):
    """Media y desviación por dimensión, calculadas solo en train (sin fuga)."""
    return {m: (v[idx_train].mean(0), v[idx_train].std(0) + 1e-6) for m, v in X.items()}


def aplicar_estandarizacion(X, stats):
    """z-score con estadísticos ya ajustados; sirve también para datos nuevos (p. ej. TMDB)."""
    return {m: ((v - stats[m][0]) / stats[m][1]).astype(np.float32) for m, v in X.items()}


def estandarizar(X, idx_train):
    """Atajo: ajusta en train y aplica a todo. Devuelve (Xz, stats)."""
    stats = ajustar_estandarizacion(X, idx_train)
    return aplicar_estandarizacion(X, stats), stats
