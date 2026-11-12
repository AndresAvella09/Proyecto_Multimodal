# Imagen del Space de Hugging Face (scripts/publicar_hf.py la copia a la raíz del Space).
# El Space solo lleva código; los datos se bajan al arrancar del repo de datasets que diga APP_DATASET.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# APP_DATOS: carpeta donde se descargan los datos (escribible). HF_HOME: caché de los modelos de lenguaje.
ENV HOME=/home/user HF_HOME=/tmp/hf APP_DATOS=/tmp/datos_app

EXPOSE 7860
CMD ["streamlit", "run", "app/Inicio.py", "--server.port=7860", "--server.address=0.0.0.0"]
