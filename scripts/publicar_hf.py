"""Publica la app en Hugging Face: los datos en un repo de datasets y el código en un Space (Docker).

Pensado para Colab, después de `scripts/exportar_app.py`. El token se lee de la variable de entorno HF_TOKEN
(en Colab: Secretos 🔑 -> HF_TOKEN con permiso de escritura; nunca en el código):

    import os; from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    !python scripts/publicar_hf.py --usuario mi-usuario

Qué hace:
1. Datos: sube `<base>/app_buscador/` a `<usuario>/<dataset>` (repo de datasets, privado salvo `--publico`).
2. Space: arma una carpeta con app/, src/, los CSV de results/, .streamlit/, el Dockerfile y el
   requirements.txt de app/ (en la raíz) y un README con la ficha del Space, y la sube a `<usuario>/<space>`.
   Deja la variable APP_DATASET del Space apuntando al dataset.

Si el dataset es privado, el Space necesita un token de **lectura** como secreto `HF_TOKEN`
(Settings del Space -> Variables and secrets -> New secret). El script no lo pone por ti para no copiar tu
token de escritura.
"""
import argparse
import os
import shutil
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FICHA = """---
title: Buscador de películas
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Póster o sinopsis, desbalance entre modalidades
---

App del proyecto [Proyecto_Multimodal](https://github.com/AndresAvella09/Proyecto_Multimodal)
(Deep Representation Learning, Michelle Ruiz y Diego Avella): buscador semántico de películas por póster y
sinopsis, clasificador de géneros y desbalance entre modalidades en películas de ayer (MM-IMDB) y de hoy (TMDB).

Los datos se descargan al arrancar del repo de datasets `{dataset}`.
Este producto usa la API de TMDB pero no está respaldado ni certificado por TMDB.
"""


def armar_space(destino, dataset_id):
    ignorar = shutil.ignore_patterns('__pycache__', '*.pyc', 'datos', '.ipynb_checkpoints')
    shutil.copytree(f'{RAIZ}/app', f'{destino}/app', ignore=ignorar)
    shutil.copytree(f'{RAIZ}/src', f'{destino}/src', ignore=ignorar)
    shutil.copytree(f'{RAIZ}/.streamlit', f'{destino}/.streamlit')
    for carpeta, _, archivos in os.walk(f'{RAIZ}/results'):
        for f in archivos:
            if f.endswith(('.csv', '.json')):
                rel = os.path.relpath(f'{carpeta}/{f}', RAIZ)
                os.makedirs(os.path.dirname(f'{destino}/{rel}'), exist_ok=True)
                shutil.copy2(f'{carpeta}/{f}', f'{destino}/{rel}')
    shutil.move(f'{destino}/app/Dockerfile', f'{destino}/Dockerfile')
    shutil.move(f'{destino}/app/requirements.txt', f'{destino}/requirements.txt')
    with open(f'{destino}/README.md', 'w', encoding='utf-8') as fh:
        fh.write(FICHA.format(dataset=dataset_id))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--usuario', required=True, help='usuario u organización de Hugging Face')
    p.add_argument('--dataset', default='peliculas-multimodal-app', help='nombre del repo de datasets')
    p.add_argument('--space', default='buscador-peliculas', help='nombre del Space')
    p.add_argument('--datos', default=os.path.join(os.environ.get('PM_BASE', '/content/drive/MyDrive/proyecto_multimodal'),
                                                   'app_buscador'), help='carpeta que generó exportar_app.py')
    p.add_argument('--publico', action='store_true', help='dataset público (por defecto privado)')
    p.add_argument('--solo-datos', action='store_true')
    p.add_argument('--solo-space', action='store_true')
    a = p.parse_args(argv)

    from huggingface_hub import HfApi
    token = os.environ.get('HF_TOKEN')
    if not token:
        sys.exit('Falta la variable de entorno HF_TOKEN (en Colab: Secretos -> HF_TOKEN).')
    api = HfApi(token=token)
    dataset_id, space_id = f'{a.usuario}/{a.dataset}', f'{a.usuario}/{a.space}'

    if not a.solo_space:
        if not os.path.exists(f'{a.datos}/config.json'):
            sys.exit(f'No encuentro {a.datos}/config.json: corre antes scripts/exportar_app.py')
        api.create_repo(dataset_id, repo_type='dataset', private=not a.publico, exist_ok=True)
        api.upload_folder(folder_path=a.datos, repo_id=dataset_id, repo_type='dataset',
                          allow_patterns=['*.parquet', '*.npy', '*.npz', 'config.json', 'umap.png',
                                          'results/*', 'resultados/*'],
                          commit_message='Datos de la app (scripts/exportar_app.py)')
        print(f'Datos en https://huggingface.co/datasets/{dataset_id} ({"público" if a.publico else "privado"})')

    if not a.solo_datos:
        api.create_repo(space_id, repo_type='space', space_sdk='docker', exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            armar_space(tmp, dataset_id)
            api.upload_folder(folder_path=tmp, repo_id=space_id, repo_type='space',
                              commit_message='Código de la app')
        api.add_space_variable(space_id, 'APP_DATASET', dataset_id)
        print(f'Space en https://huggingface.co/spaces/{space_id} (se construye en ~5-10 min; ver la pestaña Logs)')
        if not a.publico:
            print('El dataset es privado: agrega al Space un secreto HF_TOKEN con un token de LECTURA '
                  '(Settings -> Variables and secrets).')


if __name__ == '__main__':
    main()
