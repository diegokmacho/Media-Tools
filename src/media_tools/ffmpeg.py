from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    """No se encontró un ejecutable de ffmpeg."""


class FFmpegError(RuntimeError):
    """ffmpeg terminó con un código de error."""


def find_ffmpeg(explicit: str | None = None) -> str:
    """Localiza ffmpeg en PATH o en la ruta indicada explícitamente."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise FFmpegNotFoundError(f"No existe el ejecutable indicado: {candidate}")

    binary = shutil.which("ffmpeg")
    if binary is None:
        raise FFmpegNotFoundError(
            "No se encontró ffmpeg en PATH. Instálalo (por ejemplo, "
            "`sudo apt install ffmpeg`) o indícalo con --ffmpeg."
        )
    return binary


def run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Ejecuta ffmpeg y lanza FFmpegError si el proceso falla."""
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(f"No se pudo ejecutar ffmpeg: {command[0]}") from exc
    except OSError as exc:
        raise FFmpegError(f"No se pudo ejecutar ffmpeg: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-10:])
        message = f"ffmpeg terminó con código {result.returncode}."
        if tail:
            message = f"{message}\n{tail}"
        raise FFmpegError(message)

    return result
