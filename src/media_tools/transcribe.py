from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from media_tools.cuda import configure_cuda_libraries
from media_tools.timestamps import seconds_to_srt

DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "es"
DEFAULT_BEAM_SIZE = 5
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "auto"
CPU_COMPUTE_TYPE = "int8"
CUDA_COMPUTE_TYPE = "float16"

DEFAULT_CONTEXT = (
    "Capacitación técnica de ingeniería de datos en español latinoamericano. "
    "El instructor es peruano. Conservar términos técnicos en inglés y siglas. "
    "Temas: Databricks, Apache Spark, PySpark, Scala, Python, SQL, Java, "
    "RDD, Spark Core, DataFrame, Dataset, API, ETL, Delta Lake, "
    "cluster, driver, executor, notebook, pipeline y Data Lake. "
    "No traducir nombres de tecnologías, funciones, clases ni comandos. "
    "Usar puntuación y mayúsculas apropiadas."
)


class SegmentLike(Protocol):
    start: float
    end: float
    text: str


SegmentCallback = Callable[[int, SegmentLike], None]
ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class TranscriptionResult:
    audio: Path
    txt_path: Path
    srt_path: Path
    segment_count: int
    device: str
    compute_type: str
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None


def resolve_output_paths(
    audio: Path,
    txt_path: Path | None = None,
    srt_path: Path | None = None,
) -> tuple[Path, Path]:
    audio = audio.expanduser()
    base = audio.with_suffix("")
    txt = txt_path.expanduser() if txt_path is not None else Path(f"{base}_transcripcion.txt")
    srt = srt_path.expanduser() if srt_path is not None else Path(f"{base}_transcripcion.srt")
    return txt, srt


def resolve_device(device: str) -> str:
    """Resuelve `auto` usando CUDA si ctranslate2 detecta una GPU disponible."""
    if device != "auto":
        return device

    try:
        import ctranslate2
    except ImportError:
        return "cpu"

    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


def resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    return CUDA_COMPUTE_TYPE if device == "cuda" else CPU_COMPUTE_TYPE


def write_transcription(
    segments: Iterable[SegmentLike],
    txt_path: Path,
    srt_path: Path,
    *,
    on_segment: SegmentCallback | None = None,
) -> int:
    """Escribe TXT y SRT a partir de los segmentos y devuelve cuántos guardó."""
    txt_path = txt_path.expanduser()
    srt_path = srt_path.expanduser()
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with (
        txt_path.open("w", encoding="utf-8") as txt,
        srt_path.open("w", encoding="utf-8") as srt,
    ):
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            if count:
                txt.write(" ")
            count += 1
            txt.write(text)
            srt.write(
                f"{count}\n"
                f"{seconds_to_srt(segment.start)} --> {seconds_to_srt(segment.end)}\n"
                f"{text}\n\n"
            )

            if on_segment is not None:
                on_segment(count, segment)

        if count:
            txt.write("\n")

    return count


def transcribe_audio(
    audio: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str | None = DEFAULT_LANGUAGE,
    initial_prompt: str | None = DEFAULT_CONTEXT,
    beam_size: int = DEFAULT_BEAM_SIZE,
    vad_filter: bool = True,
    txt_path: Path | None = None,
    srt_path: Path | None = None,
    cuda_libs: Path | None = None,
    on_segment: SegmentCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    audio = audio.expanduser()
    if not audio.is_file():
        raise FileNotFoundError(f"No se encontró el audio: {audio}")
    if beam_size < 1:
        raise ValueError("beam_size debe ser mayor o igual que 1.")

    configure_cuda_libraries(cuda_libs)
    resolved_device = resolve_device(device)
    resolved_compute_type = resolve_compute_type(resolved_device, compute_type)

    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )
    except Exception:
        if device != "auto" or resolved_device != "cuda":
            raise
        warnings.warn(
            "No se pudo inicializar CUDA; se usará CPU con int8.",
            RuntimeWarning,
            stacklevel=2,
        )
        resolved_device = "cpu"
        resolved_compute_type = CPU_COMPUTE_TYPE
        model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )
    segments, info = model.transcribe(
        str(audio),
        language=language,
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    duration = getattr(info, "duration", None)

    def handle_segment(index: int, segment: SegmentLike) -> None:
        if on_segment is not None:
            on_segment(index, segment)
        if on_progress is not None and duration:
            on_progress(min(segment.end / duration, 1.0))

    txt, srt = resolve_output_paths(audio, txt_path, srt_path)
    segment_count = write_transcription(segments, txt, srt, on_segment=handle_segment)

    return TranscriptionResult(
        audio=audio,
        txt_path=txt,
        srt_path=srt,
        segment_count=segment_count,
        device=resolved_device,
        compute_type=resolved_compute_type,
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration=duration,
    )
