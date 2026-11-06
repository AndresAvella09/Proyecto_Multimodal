"""Métricas multietiqueta, intervalos de confianza y calibración."""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


# --- Clasificación -------------------------------------------------------------

def f1_por_clase(Y, P):
    """Y, P: (n, C) binarios -> (C,) F1 de cada clase (0 si la clase no aparece ni se predice)."""
    tp = (Y * P).sum(0)
    fp = ((1 - Y) * P).sum(0)
    fn = (Y * (1 - P)).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def f1_micro(Y, P):
    tp = (Y * P).sum()
    return 2 * tp / (2 * tp + ((1 - Y) * P).sum() + (Y * (1 - P)).sum())


def elegir_umbrales(Y, prob, grilla):
    """Para cada clase, el umbral de la grilla que maximiza su F1 en estos datos (usar dev)."""
    P = (prob[:, :, None] >= grilla[None, None, :]).astype(np.float32)   # (n, C, T)
    Yb = Y[:, :, None]
    tp = (Yb * P).sum(0)
    fp = ((1 - Yb) * P).sum(0)
    fn = (Yb * (1 - P)).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)                         # (C, T)
    return grilla[f1.argmax(1)]


def metricas(Y, prob, umbrales):
    P = (prob >= umbrales).astype(np.float32)
    return {'f1_macro': f1_por_clase(Y, P).mean(), 'f1_micro': f1_micro(Y, P),
            'ap_macro': average_precision_score(Y, prob, average='macro')}


def ap_por_clase(Y, score):
    return np.array([average_precision_score(Y[:, c], score[:, c]) for c in range(Y.shape[1])])


# --- Inferencia ------------------------------------------------------------------

def ic95(x):
    """Media y semiancho del IC 95% t de Student (para pocas semillas)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def bootstrap_media(v, n_boot=1000, semilla=0):
    """IC 95% percentil de la media de v sobre su primer eje (v: (n,) o (n, k))."""
    rng = np.random.default_rng(semilla)
    medias = np.stack([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(n_boot)])
    return v.mean(0), np.percentile(medias, 2.5, axis=0), np.percentile(medias, 97.5, axis=0)


# --- Pérdida y calibración ----------------------------------------------------------

def bce_por_etiqueta(Y, logits):
    """Entropía cruzada binaria en nats, (n, C), estable a partir de logits."""
    return np.logaddexp(0, logits) - Y * logits


def ece(Y, prob, n_bins=15):
    """Expected Calibration Error de predicciones binarias (todas las celdas juntas)."""
    y, p = Y.ravel(), prob.ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    n_b = np.bincount(bins, minlength=n_bins)
    conf = np.bincount(bins, weights=p, minlength=n_bins)
    acc = np.bincount(bins, weights=y, minlength=n_bins)
    ok = n_b > 0
    return np.sum(np.abs(conf[ok] - acc[ok])) / len(p)


def ece_macro(Y, prob, n_bins=15):
    """ECE calculado por clase y promediado: no deja que las clases frecuentes dominen."""
    return np.mean([ece(Y[:, c], prob[:, c], n_bins) for c in range(Y.shape[1])])


def curva_confiabilidad(Y, prob, n_bins=15):
    """Por bin: confianza media, frecuencia observada y número de predicciones."""
    y, p = Y.ravel(), prob.ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    n_b = np.bincount(bins, minlength=n_bins)
    conf = np.bincount(bins, weights=p, minlength=n_bins) / np.maximum(n_b, 1)
    acc = np.bincount(bins, weights=y, minlength=n_bins) / np.maximum(n_b, 1)
    return conf, acc, n_b


def ajustar_temperatura(Y, logits):
    """T que minimiza la BCE de sigmoid(logits / T) (ajustar en dev).

    T > 1: el modelo está sobreconfiado (hay que achicar los logits).
    T < 1: está subconfiado.
    """
    perdida = lambda log_t: bce_por_etiqueta(Y, logits / np.exp(log_t)).mean()
    return float(np.exp(minimize_scalar(perdida, bounds=(-3, 3), method='bounded').x))
