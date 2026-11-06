"""Carga de embeddings precalculados y etiquetas de MM-IMDB."""
import numpy as np
import pandas as pd

ARCHIVOS = {'img': 'clip_img.npy', 'txt': 'distilbert_txt.npy', 'clip_txt': 'clip_txt.npy'}
PARTICIONES = ['train', 'dev', 'test']


def cargar(data_dir, emb_dir, modalidades=('img', 'txt')):
    """Devuelve X (dict modalidad -> (N, d)), Y (N, 23), idx (dict partición -> filas) y géneros."""
    df = pd.read_parquet(f'{data_dir}/mmimdb_meta.parquet')
    assert (df['fila_hdf5'].values == np.arange(len(df))).all(), "el parquet debe estar en orden del HDF5"
    generos = list(df.columns[4:])
    Y = df[generos].values.astype(np.float32)
    X = {m: np.load(f'{emb_dir}/{ARCHIVOS[m]}') for m in modalidades}
    assert all(len(v) == len(df) for v in X.values())
    idx = {s: np.where(df['particion'] == s)[0] for s in PARTICIONES}
    return X, Y, idx, generos


def estandarizar(X, idx_train):
    """z-score por dimensión con media y desviación calculadas solo en train (sin fuga)."""
    out = {}
    for m, v in X.items():
        media, desv = v[idx_train].mean(0), v[idx_train].std(0) + 1e-6
        out[m] = ((v - media) / desv).astype(np.float32)
    return out
