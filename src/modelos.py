"""Cabezas de clasificación sobre embeddings congelados y su entrenamiento."""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


class Clasificador(nn.Module):
    """Una proyección lineal por modalidad -> concatenar -> ReLU -> Dropout -> logits.

    Con una sola modalidad es un MLP de 2 capas; con dos, la fusión ingenua (concatenación).
    """

    def __init__(self, dims, dim_proy, dropout, n_clases):
        super().__init__()
        self.proys = nn.ModuleList([nn.Linear(d, dim_proy) for d in dims])
        self.salida = nn.Sequential(nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(dim_proy * len(dims), n_clases))

    def forward(self, xs):
        return self.salida(torch.cat([p(x) for p, x in zip(self.proys, xs)], dim=1))


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


@torch.no_grad()
def predecir_logits(modelo, xs):
    modelo.eval()
    return modelo(xs).cpu().numpy()


def entrenar(xs_tr, y_tr, xs_dev, y_dev, cfg, semilla):
    """Entrena con early stopping sobre el AP macro de dev.

    xs_tr, xs_dev: listas de tensores (una por modalidad) ya en el device.
    y_tr: tensor (n, C) en el device. y_dev: array numpy (n_dev, C).
    Devuelve el modelo con los pesos de la mejor época y el historial.
    """
    fijar_semilla(semilla)
    device = xs_tr[0].device
    modelo = Clasificador([x.shape[1] for x in xs_tr], cfg['dim_proy'], cfg['dropout'],
                          y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    gen = torch.Generator(device='cpu').manual_seed(semilla)   # orden de los lotes reproducible

    n = len(y_tr)
    mejor_ap, mejor_estado, sin_mejora = -1, None, 0
    hist = {'perdida_train': [], 'ap_dev': []}
    for _ in range(cfg['max_epocas']):
        modelo.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for a in range(0, n, cfg['batch']):
            b = perm[a:a + cfg['batch']]
            perdida = F.binary_cross_entropy_with_logits(modelo([x[b] for x in xs_tr]), y_tr[b])
            opt.zero_grad()
            perdida.backward()
            opt.step()
            total += perdida.item() * len(b)
        ap = average_precision_score(y_dev, predecir_logits(modelo, xs_dev), average='macro')
        hist['perdida_train'].append(total / n)
        hist['ap_dev'].append(ap)
        if ap > mejor_ap:
            mejor_ap, sin_mejora = ap, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= cfg['paciencia']:
                break
    modelo.load_state_dict(mejor_estado)
    hist['mejor_epoca'] = int(np.argmax(hist['ap_dev'])) + 1
    return modelo, hist
