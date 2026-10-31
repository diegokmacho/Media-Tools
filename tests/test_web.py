from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_tools.frames import FrameExtractionResult
from media_tools.transcribe import DEFAULT_CONTEXT, TranscriptionResult
from media_tools.web import app as web_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_TOOLS_DATA_DIR", str(tmp_path / "data"))
    with TestClient(web_app.app) as test_client:
        yield test_client


def wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["state"] in {"done", "error"}:
            return data
        time.sleep(0.02)
    raise AssertionError(f"El trabajo {job_id} no terminó a tiempo.")


def test_index_renders_forms(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Extraer capturas" in response.text
    assert "Transcribir audio" in response.text
    assert "/static/app.js" in response.text


def test_frames_job_flow(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_extract(video: Path, *, output_dir: Path, **_kwargs: object) -> FrameExtractionResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "captura_00000.jpg").write_bytes(b"frame-0")
        (output / "captura_00001.jpg").write_bytes(b"frame-1")
        index_path = output / "indice_capturas.csv"
        index_path.write_text(
            "archivo,tiempo_aproximado_segundos,tiempo_hhmmss\n"
            "captura_00000.jpg,0,00:00:00\n"
            "captura_00001.jpg,10,00:00:10\n",
            encoding="utf-8-sig",
        )
        return FrameExtractionResult(
            video=Path(video),
            output_dir=output,
            index_path=index_path,
            frame_count=2,
        )

    monkeypatch.setattr(web_app, "extract_frames", fake_extract)

    response = client.post(
        "/api/frames",
        files={"file": ("video.mp4", b"fake-video", "video/mp4")},
        data={"intervalo": "10", "calidad": "2", "inicio": "0"},
    )

    assert response.status_code == 202
    job = wait_for_job(client, response.json()["job_id"])
    assert job["state"] == "done", job
    result = job["result"]

    assert result["frame_count"] == 2
    assert result["frames"] == ["captura_00000.jpg", "captura_00001.jpg"]
    assert [row["tiempo_hhmmss"] for row in result["index"]] == ["00:00:00", "00:00:10"]

    job_id = job["id"]
    archive = client.get(f"/api/jobs/{job_id}/download/zip")
    assert archive.status_code == 200
    assert archive.content.startswith(b"PK")

    csv_response = client.get(f"/api/jobs/{job_id}/download/csv")
    assert csv_response.status_code == 200
    assert "tiempo_hhmmss" in csv_response.text

    frame = client.get(f"/api/jobs/{job_id}/frames/captura_00000.jpg")
    assert frame.status_code == 200
    assert frame.content == b"frame-0"

    missing = client.get(f"/api/jobs/{job_id}/frames/otra.jpg")
    assert missing.status_code == 404


def test_transcription_job_flow(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass
    class FakeSegment:
        start: float
        end: float
        text: str

    captured: dict[str, object] = {}

    def fake_transcribe(audio: Path, **kwargs: object) -> TranscriptionResult:
        captured.update(kwargs)
        on_segment = kwargs.get("on_segment")
        on_progress = kwargs.get("on_progress")
        segments = [FakeSegment(0.0, 1.0, "Hola"), FakeSegment(1.0, 2.0, "mundo")]

        base = Path(audio).with_suffix("")
        txt_path = base.parent / "transcripcion.txt"
        srt_path = base.parent / "transcripcion.srt"
        txt_path.write_text("Hola mundo\n", encoding="utf-8")
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")

        for index, segment in enumerate(segments, start=1):
            if callable(on_segment):
                on_segment(index, segment)
            if callable(on_progress):
                on_progress(segment.end / 2)

        return TranscriptionResult(
            audio=Path(audio),
            txt_path=txt_path,
            srt_path=srt_path,
            segment_count=2,
            device="cpu",
            compute_type="int8",
            language="es",
            language_probability=0.98,
            duration=2.0,
        )

    monkeypatch.setattr(web_app, "transcribe_audio", fake_transcribe)

    response = client.post(
        "/api/transcribe",
        files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
        data={
            "modelo": "medium",
            "dispositivo": "cpu",
            "tipo_calculo": "int8",
            "idioma": "es",
            "beam_size": "5",
        },
    )

    assert response.status_code == 202
    job = wait_for_job(client, response.json()["job_id"])
    assert job["state"] == "done", job
    result = job["result"]

    assert result["segment_count"] == 2
    assert result["language"] == "es"
    assert result["text"] == "Hola mundo\n"
    assert [segment["text"] for segment in result["segments"]] == ["Hola", "mundo"]
    assert captured["initial_prompt"] == DEFAULT_CONTEXT
    assert captured["vad_filter"] is True

    job_id = job["id"]
    assert client.get(f"/api/jobs/{job_id}/download/txt").text == "Hola mundo\n"
    assert "00:00:00,000" in client.get(f"/api/jobs/{job_id}/download/srt").text


def test_unknown_job_and_artifact_return_404(client: TestClient) -> None:
    assert client.get("/api/jobs/no-existe").status_code == 404
    assert client.get("/api/jobs/no-existe/download/txt").status_code == 404


def test_job_error_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_extract(*_args: object, **_kwargs: object) -> FrameExtractionResult:
        raise RuntimeError("ffmpeg de prueba falló")

    monkeypatch.setattr(web_app, "extract_frames", failing_extract)

    response = client.post(
        "/api/frames",
        files={"file": ("video.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 202
    job = wait_for_job(client, response.json()["job_id"])
    assert job["state"] == "error"
    assert "ffmpeg de prueba falló" in job["error"]


def test_invalid_frames_parameters_return_400(client: TestClient) -> None:
    response = client.post(
        "/api/frames",
        files={"file": ("video.mp4", b"fake-video", "video/mp4")},
        data={"intervalo": "0"},
    )

    assert response.status_code == 400
    assert "intervalo" in response.json()["detail"]


def test_delete_job_removes_it(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_extract(video: Path, *, output_dir: Path, **_kwargs: object) -> FrameExtractionResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "captura_00000.jpg").write_bytes(b"frame-0")
        index_path = output / "indice_capturas.csv"
        index_path.write_text(
            "archivo,tiempo_aproximado_segundos,tiempo_hhmmss\ncaptura_00000.jpg,0,00:00:00\n",
            encoding="utf-8-sig",
        )
        return FrameExtractionResult(
            video=Path(video),
            output_dir=output,
            index_path=index_path,
            frame_count=1,
        )

    monkeypatch.setattr(web_app, "extract_frames", fake_extract)

    response = client.post(
        "/api/frames",
        files={"file": ("video.mp4", b"fake-video", "video/mp4")},
    )
    job_id = response.json()["job_id"]
    wait_for_job(client, job_id)

    job_directory = tmp_path / "data" / "jobs" / job_id
    assert job_directory.is_dir()

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not job_directory.exists()
