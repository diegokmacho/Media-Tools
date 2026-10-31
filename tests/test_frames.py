from __future__ import annotations

import csv
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from media_tools import frames


def test_build_ffmpeg_command(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    output_dir = tmp_path / "capturas"

    command = frames.build_ffmpeg_command(
        "/usr/bin/ffmpeg",
        video,
        output_dir,
        interval=10,
        quality=2,
        start_number=0,
    )

    assert command[0] == "/usr/bin/ffmpeg"
    assert command[command.index("-vf") + 1] == "fps=1/10"
    assert command[command.index("-q:v") + 1] == "2"
    assert command[-1] == str(output_dir / "captura_%05d.jpg")


def test_extract_frames_writes_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video")

    def fake_run(command: list[str]) -> CompletedProcess[str]:
        output_dir = Path(command[-1]).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "captura_00000.jpg").write_bytes(b"frame-0")
        (output_dir / "captura_00001.jpg").write_bytes(b"frame-1")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(frames, "find_ffmpeg", lambda explicit=None: "/usr/bin/ffmpeg")
    monkeypatch.setattr(frames, "run_ffmpeg", fake_run)

    result = frames.extract_frames(video, interval=10)

    assert result.frame_count == 2
    assert result.index_path.is_file()

    with result.index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["archivo"] for row in rows] == ["captura_00000.jpg", "captura_00001.jpg"]
    assert [row["tiempo_aproximado_segundos"] for row in rows] == ["0", "10"]
    assert [row["tiempo_hhmmss"] for row in rows] == ["00:00:00", "00:00:10"]


def test_extract_frames_missing_video(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No se encontró el vídeo"):
        frames.extract_frames(tmp_path / "no-existe.mp4")


def test_extract_frames_requires_force_for_non_empty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video")
    output_dir = tmp_path / "capturas"
    output_dir.mkdir()
    (output_dir / "captura_00000.jpg").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="--forzar"):
        frames.extract_frames(video, output_dir=output_dir)


def test_extract_frames_force_removes_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video")
    output_dir = tmp_path / "capturas"
    output_dir.mkdir()
    (output_dir / "captura_00000.jpg").write_bytes(b"old")
    (output_dir / "captura_00001.jpg").write_bytes(b"old")
    (output_dir / "indice_capturas.csv").write_text("viejo", encoding="utf-8")

    def fake_run(command: list[str]) -> CompletedProcess[str]:
        destination = Path(command[-1]).parent
        (destination / "captura_00000.jpg").write_bytes(b"new")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(frames, "find_ffmpeg", lambda explicit=None: "/usr/bin/ffmpeg")
    monkeypatch.setattr(frames, "run_ffmpeg", fake_run)

    result = frames.extract_frames(video, output_dir=output_dir, force=True)

    assert result.frame_count == 1
    assert not (output_dir / "captura_00001.jpg").exists()
    assert "viejo" not in result.index_path.read_text(encoding="utf-8-sig")


def test_write_csv_index_with_fractional_interval(tmp_path: Path) -> None:
    images = [tmp_path / "captura_00000.jpg", tmp_path / "captura_00001.jpg"]
    index_path = tmp_path / "indice_capturas.csv"

    frames.write_csv_index(images, index_path, interval=2.5)

    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["tiempo_aproximado_segundos"] for row in rows] == ["0", "2.5"]
    assert [row["tiempo_hhmmss"] for row in rows] == ["00:00:00", "00:00:02"]
