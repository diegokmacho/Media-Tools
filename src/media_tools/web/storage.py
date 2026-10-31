from __future__ import annotations

import csv
import os
import shutil
import zipfile
from pathlib import Path

from fastapi import UploadFile

DEFAULT_DATA_DIR_NAME = "data"
JOBS_DIR_NAME = "jobs"
UPLOADS_DIR_NAME = "uploads"
CHUNK_SIZE = 1024 * 1024


def data_root() -> Path:
    configured = os.environ.get("MEDIA_TOOLS_DATA_DIR")
    root = Path(configured).expanduser() if configured else Path.cwd() / DEFAULT_DATA_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def job_dir(job_id: str) -> Path:
    return data_root() / JOBS_DIR_NAME / job_id


def create_job_dir(job_id: str) -> Path:
    directory = job_dir(job_id)
    (directory / UPLOADS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    return directory


def remove_job_dir(job_id: str) -> None:
    directory = job_dir(job_id)
    if directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)


def safe_filename(filename: str | None, fallback: str = "archivo") -> str:
    raw = (filename or "").replace("\\", "/")
    name = Path(raw).name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        return fallback
    return name


def resolve_job_path(job_id: str, relative: str) -> Path:
    base = job_dir(job_id).resolve()
    candidate = (base / relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError("Ruta fuera del directorio del trabajo.")
    return candidate


async def save_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while chunk := await upload.read(CHUNK_SIZE):
            handle.write(chunk)
    await upload.close()
    return destination


def zip_directory(directory: Path, archive: Path) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(directory).as_posix())
    return archive


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
