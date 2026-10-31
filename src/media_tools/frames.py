from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from media_tools.ffmpeg import find_ffmpeg, run_ffmpeg
from media_tools.timestamps import seconds_to_hhmmss

DEFAULT_INTERVAL = 10.0
DEFAULT_QUALITY = 2
OUTPUT_SUFFIX = "_capturas"
INDEX_FILENAME = "indice_capturas.csv"
FRAME_PATTERN = "captura_*.jpg"


@dataclass(frozen=True)
class FrameExtractionResult:
    video: Path
    output_dir: Path
    index_path: Path
    frame_count: int


def build_output_dir(video: Path, output_dir: Path | None = None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser()
    return video.parent / f"{video.stem}{OUTPUT_SUFFIX}"


def build_ffmpeg_command(
    ffmpeg: str,
    video: Path,
    output_dir: Path,
    *,
    interval: float,
    quality: int,
    start_number: int = 0,
) -> list[str]:
    pattern = output_dir / "captura_%05d.jpg"
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps=1/{interval:g}",
        "-q:v",
        str(quality),
        "-start_number",
        str(start_number),
        "-y",
        str(pattern),
    ]


def ensure_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"La ruta de salida no es una carpeta: {output_dir}")
        if not force and any(output_dir.iterdir()):
            raise FileExistsError(
                f"La carpeta de salida ya contiene archivos: {output_dir}. "
                "Usa --forzar para sobrescribir."
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)


def clear_previous_output(output_dir: Path) -> None:
    for previous in output_dir.glob(FRAME_PATTERN):
        previous.unlink()
    index_path = output_dir / INDEX_FILENAME
    if index_path.is_file():
        index_path.unlink()


def list_frames(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob(FRAME_PATTERN))


def write_csv_index(images: Iterable[Path], index_path: Path, *, interval: float) -> int:
    count = 0
    with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["archivo", "tiempo_aproximado_segundos", "tiempo_hhmmss"])
        for position, image in enumerate(images):
            seconds = position * interval
            writer.writerow([image.name, _csv_seconds(seconds), seconds_to_hhmmss(seconds)])
            count += 1
    return count


def extract_frames(
    video: Path,
    *,
    output_dir: Path | None = None,
    interval: float = DEFAULT_INTERVAL,
    quality: int = DEFAULT_QUALITY,
    start_number: int = 0,
    force: bool = False,
    ffmpeg_path: str | None = None,
) -> FrameExtractionResult:
    video = video.expanduser()
    if not video.is_file():
        raise FileNotFoundError(f"No se encontró el vídeo: {video}")
    if interval <= 0:
        raise ValueError("El intervalo debe ser mayor que cero.")
    if not 1 <= quality <= 31:
        raise ValueError("La calidad JPEG debe estar entre 1 y 31.")
    if start_number < 0:
        raise ValueError("El número inicial no puede ser negativo.")

    destination = build_output_dir(video, output_dir)
    ensure_output_dir(destination, force=force)
    if force:
        clear_previous_output(destination)

    ffmpeg = find_ffmpeg(ffmpeg_path)
    command = build_ffmpeg_command(
        ffmpeg,
        video,
        destination,
        interval=interval,
        quality=quality,
        start_number=start_number,
    )
    run_ffmpeg(command)

    images = list_frames(destination)
    index_path = destination / INDEX_FILENAME
    frame_count = write_csv_index(images, index_path, interval=interval)

    return FrameExtractionResult(
        video=video,
        output_dir=destination,
        index_path=index_path,
        frame_count=frame_count,
    )


def _csv_seconds(seconds: float) -> int | float:
    if float(seconds).is_integer():
        return int(seconds)
    return round(seconds, 3)
