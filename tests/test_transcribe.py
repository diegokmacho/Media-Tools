from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from media_tools.transcribe import (
    resolve_compute_type,
    resolve_output_paths,
    write_transcription,
)


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


def test_write_transcription_creates_txt_and_srt(tmp_path: Path) -> None:
    segments = [
        FakeSegment(0.0, 1.5, " Hola "),
        FakeSegment(1.5, 2.0, "mundo "),
    ]
    txt_path = tmp_path / "salida.txt"
    srt_path = tmp_path / "salida.srt"

    count = write_transcription(segments, txt_path, srt_path)

    assert count == 2
    assert txt_path.read_text(encoding="utf-8") == "Hola mundo\n"
    assert srt_path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,500\nHola\n\n2\n00:00:01,500 --> 00:00:02,000\nmundo\n\n"
    )


def test_write_transcription_skips_empty_segments(tmp_path: Path) -> None:
    segments = [
        FakeSegment(0.0, 1.0, "   "),
        FakeSegment(1.0, 2.0, "contenido"),
        FakeSegment(2.0, 3.0, ""),
    ]
    txt_path = tmp_path / "salida.txt"
    srt_path = tmp_path / "salida.srt"

    count = write_transcription(segments, txt_path, srt_path)

    assert count == 1
    assert txt_path.read_text(encoding="utf-8") == "contenido\n"
    assert srt_path.read_text(encoding="utf-8").startswith("1\n00:00:01,000")


def test_write_transcription_creates_parent_directories(tmp_path: Path) -> None:
    txt_path = tmp_path / "textos" / "salida.txt"
    srt_path = tmp_path / "subtitulos" / "salida.srt"

    write_transcription([FakeSegment(0.0, 1.0, "hola")], txt_path, srt_path)

    assert txt_path.is_file()
    assert srt_path.is_file()


def test_write_transcription_calls_progress_callback(tmp_path: Path) -> None:
    seen: list[tuple[int, str]] = []
    segments = [FakeSegment(0.0, 1.0, "uno"), FakeSegment(1.0, 2.0, "dos")]

    write_transcription(
        segments,
        tmp_path / "salida.txt",
        tmp_path / "salida.srt",
        on_segment=lambda index, segment: seen.append((index, segment.text)),
    )

    assert seen == [(1, "uno"), (2, "dos")]


def test_resolve_output_paths_defaults(tmp_path: Path) -> None:
    audio = tmp_path / "charla.wav"

    txt_path, srt_path = resolve_output_paths(audio)

    assert txt_path == tmp_path / "charla_transcripcion.txt"
    assert srt_path == tmp_path / "charla_transcripcion.srt"


def test_resolve_output_paths_custom(tmp_path: Path) -> None:
    audio = tmp_path / "charla.wav"

    txt_path, srt_path = resolve_output_paths(audio, tmp_path / "otro.txt", tmp_path / "otro.srt")

    assert txt_path == tmp_path / "otro.txt"
    assert srt_path == tmp_path / "otro.srt"


def test_resolve_compute_type() -> None:
    assert resolve_compute_type("cpu", "auto") == "int8"
    assert resolve_compute_type("cuda", "auto") == "float16"
    assert resolve_compute_type("cuda", "float32") == "float32"
