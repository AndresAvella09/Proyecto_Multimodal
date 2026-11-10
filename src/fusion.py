"""Fusión tardía ponderada por incertidumbre (compuerta por entropía).

Combina logits ya calibrados (divididos por su temperatura) de M fuentes (imagen, texto y, si se
quiere, la fusión ingenua). Para cada película i y género c:

    g_icm = alfa_cm - beta * H_icm                 H = entropía binaria de la salida de la fuente m
    w_ic  = softmax_m(g_ic)                        peso de cada fuente
    z_ic  = sesgo_c + lambda * sum_m w_icm * z_icm

- alfa_c: preferencia fija de cada género por cada fuente (la fusión tardía por género).
- beta: cuánto se le quita peso a una fuente cuando está insegura en esa película. beta = 0 deja
  exactamente la fusión tardía por género, así que el modelo sin incertidumbre es un caso particular
  (control anidado) y el aporte de la incertidumbre se lee en beta.
- lambda, sesgo: reescalan la combinación (promediar logits los encoge) y ajustan la prevalencia.

Con dos fuentes, w_img = sigmoide((alfa_img - alfa_txt) + beta * (H_txt - H_img)).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropia_binaria(z):
    """Entropía (nats) de sigmoide(z), estable: H = softplus(z) - z * sigmoide(z). Acepta numpy o torch."""
    if isinstance(z, torch.Tensor):
        return F.softplus(z) - z * torch.sigmoid(z)
    return np.logaddexp(0, z) - z / (1 + np.exp(-z))


class CompuertaIncertidumbre(nn.Module):
    def __init__(self, n_fuentes, n_clases, usar_incertidumbre=True):
        super().__init__()
        self.alfa = nn.Parameter(torch.zeros(n_clases, n_fuentes))
        self.sesgo = nn.Parameter(torch.zeros(n_clases))
        self.log_lambda = nn.Parameter(torch.zeros(()))
        if usar_incertidumbre:
            self.beta = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer('beta', torch.zeros(()))

    def forward(self, Z):
        """Z: (n, C, M) logits calibrados -> logits combinados (n, C) y pesos (n, C, M)."""
        w = torch.softmax(self.alfa - self.beta * entropia_binaria(Z), dim=-1)
        return self.sesgo + self.log_lambda.exp() * (w * Z).sum(-1), w


def ajustar_compuerta(Z, Y, usar_incertidumbre=True, l2=1e-4, max_iter=300):
    """Ajusta la compuerta minimizando la entropía cruzada binaria (usar en dev).

    Z: (n, C, M) logits calibrados. Y: (n, C) binario. l2 penaliza alfa (lo hace identificable,
    porque el softmax no cambia si se suma lo mismo a todas las fuentes).
    """
    Zt = torch.as_tensor(Z, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    modelo = CompuertaIncertidumbre(Z.shape[2], Z.shape[1], usar_incertidumbre)
    opt = torch.optim.LBFGS(modelo.parameters(), lr=1, max_iter=max_iter, line_search_fn='strong_wolfe')

    def cierre():
        opt.zero_grad()
        z, _ = modelo(Zt)
        perdida = F.binary_cross_entropy_with_logits(z, Yt) + l2 * modelo.alfa.pow(2).sum()
        perdida.backward()
        return perdida

    opt.step(cierre)
    return modelo.eval()


@torch.no_grad()
def combinar(modelo, Z):
    """Aplica una compuerta ajustada. Z: (n, C, M) numpy -> (logits (n, C), pesos (n, C, M)) numpy."""
    z, w = modelo(torch.as_tensor(Z, dtype=torch.float32))
    return z.numpy(), w.numpy()


def cargar_compuertas(ruta):
    """Lee lo guardado por el notebook C1: devuelve el diccionario con 'compuertas' como
    {(metodo, semilla): CompuertaIncertidumbre}."""
    art = torch.load(ruta, map_location='cpu', weights_only=False)
    compuertas = {}
    for clave, estado in art['compuertas'].items():
        metodo, semilla = clave.rsplit('_s', 1)
        modelo = CompuertaIncertidumbre(len(art['entradas'][metodo]), len(art['generos']),
                                        art['incertidumbre'][metodo])
        modelo.load_state_dict(estado)
        compuertas[(metodo, int(semilla))] = modelo.eval()
    art['compuertas'] = compuertas
    return art
