"""Preprocesamiento y encoders congelados (CLIP ViT-B/32 y DistilBERT).

Es el mismo procedimiento del notebook 03, reunido aquí para procesar datos nuevos (TMDB)
exactamente igual que MM-IMDB.
"""
import re

import numpy as np
import torch

CLIP_ID = 'openai/clip-vit-base-patch32'
BERT_ID = 'distilbert-base-uncased'
FORMA_MMIMDB = (256, 160)                                     # alto, ancho de los pósters del HDF5
MEDIA_CLIP = np.array([0.48145466, 0.4578275, 0.40821073])   # media RGB de entrenamiento de CLIP
COLOR_RELLENO = np.rint(MEDIA_CLIP * 255).astype(np.uint8)


# --- Pósters ----------------------------------------------------------------------

def poster_a_rgb(img):
    """(3, 256, 160) en formato VGG (BGR, media restada) -> (256, 160, 3) RGB uint8."""
    medias_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)[:, None, None]
    rgb = (img.astype(np.float32) + medias_bgr)[::-1]
    return np.clip(np.rint(np.transpose(rgb, (1, 2, 0))), 0, 255).astype(np.uint8)


def redimensionar_como_mmimdb(imagen_pil):
    """Imagen PIL cualquiera -> (256, 160, 3) RGB uint8, la forma de los pósters de MM-IMDB."""
    from PIL import Image
    alto, ancho = FORMA_MMIMDB
    return np.asarray(imagen_pil.convert('RGB').resize((ancho, alto), Image.BICUBIC), dtype=np.uint8)


def rellenar_cuadrado(rgb):
    """(H, W, 3) -> (L, L, 3) con L = max(H, W), imagen centrada y relleno con la media de CLIP."""
    h, w, _ = rgb.shape
    lado = max(h, w)
    out = np.empty((lado, lado, 3), dtype=np.uint8)
    out[:] = COLOR_RELLENO
    y0, x0 = (lado - h) // 2, (lado - w) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return out


# --- Sinopsis ---------------------------------------------------------------------

def normalizar_sinopsis(texto):
    """Lleva un texto al estilo de las sinopsis de MM-IMDB (tras la limpieza del notebook 01).

    MM-IMDB está en minúsculas, con los números reemplazados por 'N' y los guiones entre palabras
    separados por espacios ('ex - partner'). Las comillas tipográficas se pasan a rectas.
    """
    t = (texto or '').replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    t = t.lower()
    t = re.sub(r'\d+(?:[.,]\d+)*', 'N', t)
    t = re.sub(r'(?<=\w)-(?=\w)', ' - ', t)
    return re.sub(r'\s+', ' ', t).strip()


# --- Encoders ---------------------------------------------------------------------

class Encoders:
    """CLIP (imagen y texto, salida proyectada de 512) y DistilBERT (mean pooling, 768), congelados."""

    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPModel, CLIPTokenizer
        self.device = device
        self.clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.proc_img = CLIPImageProcessor.from_pretrained(CLIP_ID)
        self.tok_clip = CLIPTokenizer.from_pretrained(CLIP_ID)
        self.tok_bert = AutoTokenizer.from_pretrained(BERT_ID)
        self.bert = AutoModel.from_pretrained(BERT_ID).to(device).eval()

    @torch.no_grad()
    def imagen(self, imgs_rgb, lote=128):
        """Lista de (L, L, 3) uint8 ya rellenadas -> (N, 512), sin normalizar."""
        out = []
        for s in range(0, len(imgs_rgb), lote):
            pix = self.proc_img(images=imgs_rgb[s:s + lote], return_tensors='pt')['pixel_values'].to(self.device)
            pooled = self.clip.vision_model(pixel_values=pix).pooler_output
            out.append(self.clip.visual_projection(pooled).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def fragmentar_clip(self, textos):
        """Fragmentos de 75 tokens + inicio y fin. Devuelve ids (F, 77), máscara (F, 77) y dueño (F,)."""
        utiles = self.tok_clip.model_max_length - 2
        bos, eos, pad = self.tok_clip.bos_token_id, self.tok_clip.eos_token_id, self.tok_clip.pad_token_id
        ids, masc, duenio = [], [], []
        for i, t in enumerate(self.tok_clip(list(textos), add_special_tokens=False)['input_ids']):
            t = t or [eos]   # sinopsis vacía: un fragmento mínimo
            for s in range(0, len(t), utiles):
                frag = [bos] + t[s:s + utiles] + [eos]
                relleno = utiles + 2 - len(frag)
                ids.append(frag + [pad] * relleno)
                masc.append([1] * len(frag) + [0] * relleno)
                duenio.append(i)
        return np.array(ids), np.array(masc), np.array(duenio)

    @torch.no_grad()
    def texto_clip(self, textos, lote=512):
        """Cada fragmento se codifica y normaliza; se promedian los de la misma sinopsis -> (N, 512)."""
        ids, masc, duenio = self.fragmentar_clip(textos)
        vecs = np.zeros((len(ids), 512), dtype=np.float32)
        for s in range(0, len(ids), lote):
            i = torch.as_tensor(ids[s:s + lote], device=self.device)
            m = torch.as_tensor(masc[s:s + lote], device=self.device)
            pooled = self.clip.text_model(input_ids=i, attention_mask=m).pooler_output
            vecs[s:s + lote] = self.clip.text_projection(pooled).float().cpu().numpy()
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        suma = np.zeros((len(textos), 512), dtype=np.float64)
        np.add.at(suma, duenio, vecs)
        return (suma / np.bincount(duenio, minlength=len(textos))[:, None]).astype(np.float32)

    @torch.no_grad()
    def texto_bert(self, textos, lote=64):
        """Mean pooling de la última capa sobre tokens reales, truncado a 512 -> (N, 768)."""
        textos = list(textos)
        orden = np.argsort([len(t) for t in textos])   # lotes de largo parecido = menos padding
        out = np.zeros((len(textos), 768), dtype=np.float32)
        for s in range(0, len(textos), lote):
            idx = orden[s:s + lote]
            enc = self.tok_bert([textos[i] for i in idx], padding=True, truncation=True,
                                max_length=512, return_tensors='pt').to(self.device)
            h = self.bert(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
        return out
