"""Piezas del buscador semántico (notebooks 09-13): búsquedas de evaluación, adaptador y métricas.

Son las mismas definiciones de los notebooks 10 y 11, reunidas aquí para aplicar el buscador a
películas nuevas (TMDB) exactamente igual que a MM-IMDB.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELO_POSTER = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'   # consulta -> espacio de CLIP
MODELO_SINOPSIS = 'intfloat/multilingual-e5-base'                       # consulta y sinopsis (E5)

# Búsquedas de evaluación del notebook 10: (frase es, palabra es, frase en, palabra en)
CONSULTAS = {
    'Drama': ('una película dramática', 'drama', 'a drama movie', 'drama'),
    'Comedy': ('una película de comedia', 'comedia', 'a comedy movie', 'comedy'),
    'Romance': ('una película romántica', 'romance', 'a romantic movie', 'romance'),
    'Thriller': ('una película de suspenso', 'suspenso', 'a thriller movie', 'thriller'),
    'Crime': ('una película de crimen', 'crimen', 'a crime movie', 'crime'),
    'Action': ('una película de acción', 'acción', 'an action movie', 'action'),
    'Adventure': ('una película de aventuras', 'aventura', 'an adventure movie', 'adventure'),
    'Horror': ('una película de terror', 'terror', 'a horror movie', 'horror'),
    'Documentary': ('un documental', 'documental', 'a documentary', 'documentary'),
    'Mystery': ('una película de misterio', 'misterio', 'a mystery movie', 'mystery'),
    'Sci-Fi': ('una película de ciencia ficción', 'ciencia ficción', 'a science fiction movie', 'sci-fi'),
    'Fantasy': ('una película de fantasía', 'fantasía', 'a fantasy movie', 'fantasy'),
    'Family': ('una película familiar', 'familia', 'a family movie', 'family'),
    'Biography': ('una película biográfica', 'biografía', 'a biographical movie', 'biography'),
    'War': ('una película de guerra', 'guerra', 'a war movie', 'war'),
    'History': ('una película histórica', 'historia', 'a historical movie', 'history'),
    'Music': ('una película sobre música', 'música', 'a movie about music', 'music'),
    'Animation': ('una película animada', 'animación', 'an animated movie', 'animation'),
    'Musical': ('un musical', 'musical', 'a musical', 'musical'),
    'Western': ('una película del oeste', 'vaqueros', 'a western movie', 'western'),
    'Sport': ('una película de deportes', 'deportes', 'a sports movie', 'sports'),
    'Short': ('un cortometraje', 'corto', 'a short film', 'short'),
    'Film-Noir': ('una película de cine negro', 'cine negro', 'a film noir', 'noir'),
}
TIPOS = [('es', 'frase'), ('es', 'palabra'), ('en', 'frase'), ('en', 'palabra')]


def l2(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)


def zscore(S):
    """Estandariza cada fila (una búsqueda) sobre el catálogo: pone póster y sinopsis en la misma escala."""
    return (S - S.mean(axis=-1, keepdims=True)) / (S.std(axis=-1, keepdims=True) + 1e-8)


def puntajes(qp, qs, Vp, Vs, modo, alfa=0.5):
    """qp, qs: consultas (Q, d) en el espacio del póster y de la sinopsis; Vp, Vs: catálogo (n, d)."""
    sp, ss = qp @ Vp.T, qs @ Vs.T
    if modo == 'poster':
        return sp
    if modo == 'sinopsis':
        return ss
    if modo == 'ambos':
        return alfa * zscore(sp) + (1 - alfa) * zscore(ss)
    raise ValueError(modo)


CORRECCIONES = ['global', 'por_fuente', 'por_fuente_media']


def normalizar(S, fuente, correccion='global'):
    """Pone en escala comparable los puntajes de un catálogo que mezcla fuentes (MM-IMDB y TMDB).

    S: (Q, n) similitudes crudas; fuente: (n,) etiqueta de la fuente de cada película.
    - 'global': z-score sobre todo el catálogo (lo que hacía la app).
    - 'por_fuente': z-score dentro de cada fuente; las dos quedan con la misma media y escala.
    - 'por_fuente_media': solo se resta la media de cada fuente; la escala es la del catálogo entero.
    """
    if correccion == 'global':
        return zscore(S)
    out = np.empty_like(S, dtype=np.float32)
    for f in np.unique(fuente):
        m = fuente == f
        out[:, m] = zscore(S[:, m]) if correccion == 'por_fuente' else S[:, m] - S[:, m].mean(1, keepdims=True)
    if correccion == 'por_fuente_media':
        out /= out.std(1, keepdims=True) + 1e-8
    return out


def puntajes_mixtos(qp, qs, Vp, Vs, fuente, modo, alfa=0.5, correccion='global'):
    """Como `puntajes`, pero normalizando cada modalidad según `correccion` (ver `normalizar`)."""
    zp, zs = normalizar(qp @ Vp.T, fuente, correccion), normalizar(qs @ Vs.T, fuente, correccion)
    return {'poster': zp, 'sinopsis': zs, 'ambos': alfa * zp + (1 - alfa) * zs}[modo]


class Adaptador(nn.Module):
    """El adaptador residual del notebook 11: v' = normalizar(v + W v). Misma estructura para poder
    cargar sus pesos (modelos/adaptador_{poster,sinopsis}_s{semilla}.pt)."""

    def __init__(self, d, n_generos):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.W.weight)
        self.log_escala = nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))
        self.sesgo = nn.Parameter(torch.zeros(n_generos))

    def forward(self, v):
        return F.normalize(v + self.W(v), dim=-1)


def cargar_adaptador(ruta, d, n_generos):
    modelo = Adaptador(d, n_generos)
    modelo.load_state_dict(torch.load(ruta, map_location='cpu'))
    return modelo.eval()


@torch.no_grad()
def adaptar(modelo, V, lote=8192):
    """Aplica un adaptador a vectores ya normalizados (n, d) -> (n, d) normalizados."""
    return np.concatenate([modelo(torch.as_tensor(V[a:a + lote], dtype=torch.float32)).numpy()
                           for a in range(0, len(V), lote)])


def ap_por_consulta(Y, S):
    """AP de cada búsqueda. Y: (n, Q) relevancia binaria de cada película para cada búsqueda;
    S: (Q, n) puntajes. Vectorizado (igual a sklearn salvo empates). Sin relevantes -> NaN."""
    orden = np.argsort(-S.T, axis=0, kind='stable')
    y = np.take_along_axis(Y, orden, axis=0)
    precision = np.cumsum(y, axis=0) / np.arange(1, len(Y) + 1)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        return (precision * y).sum(0) / y.sum(0)


def precision_en_k(Y, S, k=10):
    top = np.argsort(-S, axis=1)[:, :k]                      # (Q, k)
    return np.take_along_axis(Y.T, top, axis=1).mean(1)
