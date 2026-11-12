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


def cargar_artefactos(ruta, device='cpu'):
    """Lee los artefactos guardados por el notebook 06 y reconstruye los modelos.

    Devuelve el diccionario guardado, con 'modelos' convertido en {(nombre, semilla): Clasificador}.
    weights_only=False porque el archivo trae arrays de numpy (estandarización, umbrales); usar
    solo con archivos propios.
    """
    art = torch.load(ruta, map_location=device, weights_only=False)
    dims = {m: len(mu) for m, (mu, _) in art['stats_estandarizacion'].items()}
    modelos = {}
    for clave, estado in art['modelos'].items():
        nombre, semilla = clave.rsplit('_s', 1)
        modelo = Clasificador([dims[m] for m in art['modalidades'][nombre]], art['config']['dim_proy'],
                              art['config']['dropout'], len(art['generos_entrenamiento']))
        modelo.load_state_dict(estado)
        modelos[(nombre, int(semilla))] = modelo.to(device).eval()
    art['modelos'] = modelos
    return art


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


@torch.no_grad()
def aporte_por_modalidad(modelo, xs, y):
    """Puntaje de cada modalidad dentro de la fusión ingenua (para OGM-GE).

    La salida es lineal sobre la concatenación de ReLU(proyección_m), así que el logit se separa en una
    parte por modalidad: z_m = W_m h_m + b / 2. El puntaje es la probabilidad media que z_m le da a los
    géneros correctos (el análogo multietiqueta de la suma de softmax de la clase correcta del artículo).
    """
    lineal, d = modelo.salida[-1], modelo.proys[0].out_features
    s = []
    for k, (p, x) in enumerate(zip(modelo.proys, xs)):
        z = F.relu(p(x)) @ lineal.weight[:, k * d:(k + 1) * d].T + lineal.bias / 2
        s.append(((torch.sigmoid(z) * y).sum() / y.sum().clamp(min=1)).item())
    return s


def entrenar_ogm(xs_tr, y_tr, xs_dev, y_dev, cfg, semilla, alfa, ge=True):
    """Fusión ingenua entrenada con OGM-GE (Peng et al., CVPR 2022), dos modalidades.

    En cada lote: r = s_0 / s_1 (aporte_por_modalidad). A la modalidad que domina (r > 1) se le
    multiplica el gradiente de su proyección por k = 1 - tanh(alfa * (r - 1)); la otra queda con k = 1.
    Con ge=True se suma ruido gaussiano con la desviación del gradiente a las proyecciones
    (Generalization Enhancement). Mismo early stopping que `entrenar`. El historial guarda el r medio
    y los coeficientes de cada época.
    """
    assert len(xs_tr) == 2
    fijar_semilla(semilla)
    device = xs_tr[0].device
    modelo = Clasificador([x.shape[1] for x in xs_tr], cfg['dim_proy'], cfg['dropout'], y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    gen = torch.Generator(device='cpu').manual_seed(semilla)
    gen_ruido = torch.Generator(device=device).manual_seed(semilla)

    n = len(y_tr)
    mejor_ap, mejor_estado, sin_mejora = -1, None, 0
    hist = {'perdida_train': [], 'ap_dev': [], 'razon': [], 'k0': [], 'k1': []}
    for _ in range(cfg['max_epocas']):
        modelo.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total, rs, ks = 0.0, [], []
        for a in range(0, n, cfg['batch']):
            b = perm[a:a + cfg['batch']]
            xb, yb = [x[b] for x in xs_tr], y_tr[b]
            perdida = F.binary_cross_entropy_with_logits(modelo(xb), yb)
            opt.zero_grad()
            perdida.backward()
            s0, s1 = aporte_por_modalidad(modelo, xb, yb)
            r = s0 / max(s1, 1e-8)
            k = [1 - np.tanh(alfa * max(r - 1, 0)), 1 - np.tanh(alfa * max(1 / max(r, 1e-8) - 1, 0))]
            for km, proy in zip(k, modelo.proys):
                for prm in proy.parameters():
                    g = prm.grad
                    ruido = (torch.randn(g.shape, generator=gen_ruido, device=device) * (g.std() + 1e-8)) if ge else 0
                    prm.grad = g * km + ruido
            opt.step()
            total += perdida.item() * len(b)
            rs.append(r)
            ks.append(k)
        ap = average_precision_score(y_dev, predecir_logits(modelo, xs_dev), average='macro')
        hist['perdida_train'].append(total / n)
        hist['ap_dev'].append(ap)
        hist['razon'].append(float(np.mean(rs)))
        hist['k0'].append(float(np.mean([k[0] for k in ks])))
        hist['k1'].append(float(np.mean([k[1] for k in ks])))
        if ap > mejor_ap:
            mejor_ap, sin_mejora = ap, 0
            mejor_estado = {k_: v.detach().clone() for k_, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= cfg['paciencia']:
                break
    modelo.load_state_dict(mejor_estado)
    hist['mejor_epoca'] = int(np.argmax(hist['ap_dev'])) + 1
    return modelo, hist
