# media-tools

Interfaz web local para extraer capturas de un vídeo y transcribir audio a texto/subtítulos.

La aplicación se ejecuta en `http://127.0.0.1:8000` y reutiliza la misma librería de procesamiento:
`ffmpeg` para las capturas y `faster-whisper` para la transcripción.

## Requisitos

- Python 3.12 (gestionado con pyenv en este proyecto)
- ffmpeg en `PATH`
- Opcional para GPU: `nvidia-cublas-cu12` y `nvidia-cudnn-cu12` (extra `gpu`)

## Instalación

```bash
source .venv/bin/activate
pip install -e ".[dev,gpu]"
```

Si no hay GPU o no se quieren instalar las librerías CUDA:

```bash
pip install -e ".[dev]"
```

## Arranque

```bash
python -m media_tools.web
```

Abre `http://127.0.0.1:8000` en el navegador.

Variables de entorno opcionales:

| Variable | Descripción | Por defecto |
| --- | --- | --- |
| `MEDIA_TOOLS_HOST` | Interfaz de red donde escucha el servidor | `127.0.0.1` |
| `MEDIA_TOOLS_PORT` | Puerto del servidor | `8000` |
| `MEDIA_TOOLS_DATA_DIR` | Carpeta para subidas y resultados | `./data` |

## Uso

### Capturas de vídeo

1. Abre la pestaña **Capturas de vídeo**.
2. Arrastra un vídeo o selecciónalo desde el explorador.
3. Ajusta el intervalo entre capturas, la calidad JPEG y el número inicial.
4. Pulsa **Extraer capturas**.

La interfaz muestra una galería con las capturas, el índice con el tiempo aproximado de cada
imagen y botones para descargar el ZIP completo o el CSV.

### Transcripción de audio

1. Abre la pestaña **Transcripción de audio**.
2. Arrastra un audio (o vídeo) o selecciónalo desde el explorador.
3. Configura modelo, dispositivo, tipo de cálculo, idioma y beam size.
4. Opcionalmente escribe un contexto con términos, nombres o siglas que el modelo deba reconocer.
5. Pulsa **Transcribir**.

La interfaz muestra el progreso, el texto completo, la lista de segmentos con tiempos y botones
para descargar el TXT y el SRT.

### GPU en WSL

El extra `gpu` instala cuBLAS y cuDNN para Linux. Al procesar, `media_tools.cuda` añade esos
directorios a `LD_LIBRARY_PATH` y precarga las librerías para que ctranslate2 encuentre CUDA.
Usa el dispositivo `auto` para intentar CUDA y caer a CPU si no está disponible.

## Cómo funciona

- Cada archivo subido y cada resultado se guarda en `data/jobs/<id>/`.
- Los trabajos se ejecutan en segundo plano y la interfaz consulta su estado cada segundo.
- Al terminar, los archivos se sirven mediante `/api/jobs/<id>/download/<artefacto>`.
- El botón **Descartar** elimina el trabajo y sus archivos del disco.

## API

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/` | Página principal |
| `POST` | `/api/frames` | Sube un vídeo y crea un trabajo de capturas |
| `POST` | `/api/transcribe` | Sube un audio y crea un trabajo de transcripción |
| `GET` | `/api/jobs/{id}` | Estado, progreso y resultado del trabajo |
| `DELETE` | `/api/jobs/{id}` | Elimina el trabajo y sus archivos |
| `GET` | `/api/jobs/{id}/download/{artefacto}` | Descarga `zip`, `csv`, `txt` o `srt` |
| `GET` | `/api/jobs/{id}/frames/{archivo}` | Sirve una captura para previsualizarla |

## Estructura

```
.
├── src/media_tools/
│   ├── web/
│   │   ├── __main__.py      # python -m media_tools.web
│   │   ├── app.py           # FastAPI, rutas y ejecución de trabajos
│   │   ├── jobs.py          # gestor de trabajos en segundo plano
│   │   ├── storage.py       # carpetas, subidas, ZIP y rutas seguras
│   │   ├── templates/       # plantillas Jinja2
│   │   └── static/          # CSS y JavaScript
│   ├── cuda.py              # configuración de librerías CUDA
│   ├── ffmpeg.py            # localización y ejecución de ffmpeg
│   ├── frames.py            # extracción de capturas e índice CSV
│   ├── timestamps.py        # formato de tiempos SRT/HH:MM:SS
│   └── transcribe.py        # transcripción y escritura TXT/SRT
├── tests/
├── pyproject.toml
└── README.md
```

## Desarrollo

```bash
pytest
ruff check .
ruff format --check .
```
