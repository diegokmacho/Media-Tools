from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from media_tools.frames import (
    DEFAULT_INTERVAL,
    DEFAULT_QUALITY,
    extract_frames,
    list_frames,
)
from media_tools.timestamps import seconds_to_hhmmss
from media_tools.transcribe import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_CONTEXT,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    SegmentLike,
    transcribe_audio,
)
from media_tools.web import storage
from media_tools.web.jobs import Job, JobManager

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEVICES = ["auto", "cpu", "cuda"]
COMPUTE_TYPES = ["auto", "int8", "float16", "float32", "int8_float16"]

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs = JobManager()
    try:
        yield
    finally:
        app.state.jobs.shutdown()


app = FastAPI(
    title="media-tools",
    description="Extrae capturas de vídeo y transcribe audio desde el navegador.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def get_manager(request: Request) -> JobManager:
    return request.app.state.jobs


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "models": MODELS,
            "devices": DEVICES,
            "compute_types": COMPUTE_TYPES,
            "defaults": {
                "intervalo": DEFAULT_INTERVAL,
                "calidad": DEFAULT_QUALITY,
                "modelo": DEFAULT_MODEL,
                "dispositivo": DEFAULT_DEVICE,
                "tipo_calculo": DEFAULT_COMPUTE_TYPE,
                "idioma": DEFAULT_LANGUAGE,
                "beam_size": DEFAULT_BEAM_SIZE,
            },
        },
    )


@app.post("/api/frames", status_code=status.HTTP_202_ACCEPTED)
async def start_frames(
    request: Request,
    file: Annotated[UploadFile, File()],
    intervalo: Annotated[float, Form()] = DEFAULT_INTERVAL,
    calidad: Annotated[int, Form()] = DEFAULT_QUALITY,
    inicio: Annotated[int, Form()] = 0,
) -> dict[str, str]:
    if intervalo <= 0:
        raise HTTPException(status_code=400, detail="El intervalo debe ser mayor que cero.")
    if not 1 <= calidad <= 31:
        raise HTTPException(status_code=400, detail="La calidad JPEG debe estar entre 1 y 31.")
    if inicio < 0:
        raise HTTPException(status_code=400, detail="El número inicial no puede ser negativo.")

    manager = get_manager(request)
    job = manager.create("frames")
    try:
        video = await _store_upload(file, job, fallback="video")
    except Exception:
        manager.delete(job.id)
        raise

    runner = lambda current: _run_frames(  # noqa: E731
        manager,
        current,
        video=video,
        intervalo=intervalo,
        calidad=calidad,
        inicio=inicio,
    )
    manager.submit(job, runner)
    return {"job_id": job.id}


@app.post("/api/transcribe", status_code=status.HTTP_202_ACCEPTED)
async def start_transcription(
    request: Request,
    file: Annotated[UploadFile, File()],
    modelo: Annotated[str, Form()] = DEFAULT_MODEL,
    dispositivo: Annotated[str, Form()] = DEFAULT_DEVICE,
    tipo_calculo: Annotated[str, Form()] = DEFAULT_COMPUTE_TYPE,
    idioma: Annotated[str, Form()] = DEFAULT_LANGUAGE,
    contexto: Annotated[str, Form()] = "",
    beam_size: Annotated[int, Form()] = DEFAULT_BEAM_SIZE,
    sin_vad: Annotated[bool, Form()] = False,
) -> dict[str, str]:
    if modelo not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo desconocido: {modelo}")
    if dispositivo not in DEVICES:
        raise HTTPException(status_code=400, detail=f"Dispositivo desconocido: {dispositivo}")
    if tipo_calculo not in COMPUTE_TYPES:
        raise HTTPException(status_code=400, detail=f"Tipo de cálculo desconocido: {tipo_calculo}")
    if beam_size < 1:
        raise HTTPException(status_code=400, detail="beam_size debe ser mayor o igual que 1.")

    manager = get_manager(request)
    job = manager.create("transcribe")
    try:
        audio = await _store_upload(file, job, fallback="audio")
    except Exception:
        manager.delete(job.id)
        raise

    language = None if not idioma or idioma.lower() == "auto" else idioma
    prompt = contexto.strip() or DEFAULT_CONTEXT

    runner = lambda current: _run_transcription(  # noqa: E731
        manager,
        current,
        audio=audio,
        modelo=modelo,
        dispositivo=dispositivo,
        tipo_calculo=tipo_calculo,
        idioma=language,
        contexto=prompt,
        beam_size=beam_size,
        sin_vad=sin_vad,
    )
    manager.submit(job, runner)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> dict[str, Any]:
    snapshot = get_manager(request).snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return snapshot


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, request: Request) -> dict[str, bool]:
    if not get_manager(request).delete(job_id):
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return {"deleted": True}


@app.get("/api/jobs/{job_id}/download/{artifact}")
def download_artifact(job_id: str, artifact: str, request: Request):
    snapshot = get_manager(request).snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")

    artifacts = (snapshot.get("result") or {}).get("artifacts") or {}
    relative = artifacts.get(artifact)
    if not relative:
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.")

    try:
        path = storage.resolve_job_path(job_id, relative)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.")
    return FileResponse(path, filename=path.name)


@app.get("/api/jobs/{job_id}/frames/{filename}")
def frame_image(job_id: str, filename: str, request: Request):
    snapshot = get_manager(request).snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")

    available = (snapshot.get("result") or {}).get("frames") or []
    name = storage.safe_filename(filename)
    if name not in available:
        raise HTTPException(status_code=404, detail="Captura no encontrada.")

    try:
        path = storage.resolve_job_path(job_id, f"capturas/{name}")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Captura no encontrada.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Captura no encontrada.")
    return FileResponse(path)


async def _store_upload(upload: UploadFile, job: Job, *, fallback: str) -> Path:
    name = storage.safe_filename(upload.filename, fallback=fallback)
    destination = storage.job_dir(job.id) / storage.UPLOADS_DIR_NAME / name
    await storage.save_upload(upload, destination)
    if destination.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
    return destination


def _run_frames(
    manager: JobManager,
    job: Job,
    *,
    video: Path,
    intervalo: float,
    calidad: int,
    inicio: int,
) -> dict[str, Any]:
    manager.update(job, message="Extrayendo capturas con ffmpeg...")
    output_dir = storage.job_dir(job.id) / "capturas"
    result = extract_frames(
        video,
        output_dir=output_dir,
        interval=intervalo,
        quality=calidad,
        start_number=inicio,
        force=True,
    )
    manager.update(job, message="Generando descargas...")

    frames = list_frames(output_dir)
    rows = storage.read_csv_rows(result.index_path)
    archive = storage.zip_directory(output_dir, storage.job_dir(job.id) / "capturas.zip")

    return {
        "frame_count": result.frame_count,
        "frames": [frame.name for frame in frames],
        "index": rows,
        "artifacts": {
            "zip": _relative(job, archive),
            "csv": _relative(job, result.index_path),
        },
    }


def _run_transcription(
    manager: JobManager,
    job: Job,
    *,
    audio: Path,
    modelo: str,
    dispositivo: str,
    tipo_calculo: str,
    idioma: str | None,
    contexto: str | None,
    beam_size: int,
    sin_vad: bool,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []

    def handle_segment(index: int, segment: SegmentLike) -> None:
        segments.append(
            {
                "index": index,
                "start": segment.start,
                "end": segment.end,
                "time": seconds_to_hhmmss(segment.start),
                "text": segment.text.strip(),
            }
        )

    def handle_progress(fraction: float) -> None:
        manager.update(
            job,
            progress=fraction,
            message=f"Transcribiendo... {fraction:.0%}",
        )

    manager.update(job, message="Cargando modelo y transcribiendo...")
    result = transcribe_audio(
        audio,
        model_name=modelo,
        device=dispositivo,
        compute_type=tipo_calculo,
        language=idioma,
        initial_prompt=contexto,
        beam_size=beam_size,
        vad_filter=not sin_vad,
        on_segment=handle_segment,
        on_progress=handle_progress,
    )
    manager.update(job, message="Guardando resultados...")

    return {
        "segment_count": result.segment_count,
        "device": result.device,
        "compute_type": result.compute_type,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "text": result.txt_path.read_text(encoding="utf-8"),
        "segments": segments,
        "artifacts": {
            "txt": _relative(job, result.txt_path),
            "srt": _relative(job, result.srt_path),
        },
    }


def _relative(job: Job, path: Path) -> str:
    return path.resolve().relative_to(storage.job_dir(job.id).resolve()).as_posix()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("MEDIA_TOOLS_HOST", "127.0.0.1")
    port = int(os.environ.get("MEDIA_TOOLS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
