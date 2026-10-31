from __future__ import annotations

import os
from pathlib import Path

import pytest

from media_tools import cuda


def test_configure_cuda_libraries_prepends_discovered_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvidia_library = tmp_path / "nvidia-cublas" / "lib"
    nvidia_library.mkdir(parents=True)
    existing_library = tmp_path / "existing"
    existing_library.mkdir()

    monkeypatch.setattr(cuda, "_WSL_DRIVER_DIRECTORY", tmp_path / "sin-wsl")
    monkeypatch.setattr(cuda, "discover_nvidia_libraries", lambda: [nvidia_library])
    monkeypatch.setenv(cuda._LIBRARY_PATH_VARIABLE, str(existing_library))

    directories = cuda.configure_cuda_libraries()

    assert directories == [nvidia_library.resolve()]
    entries = os.environ[cuda._LIBRARY_PATH_VARIABLE].split(os.pathsep)
    assert entries[0] == str(nvidia_library.resolve())
    assert str(existing_library) in entries


def test_configure_cuda_libraries_accepts_extra_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra_directory = tmp_path / "cuda-libs"
    extra_directory.mkdir()

    monkeypatch.setattr(cuda, "_WSL_DRIVER_DIRECTORY", tmp_path / "sin-wsl")
    monkeypatch.setattr(cuda, "discover_nvidia_libraries", lambda: [])
    monkeypatch.setenv(cuda._LIBRARY_PATH_VARIABLE, "")

    directories = cuda.configure_cuda_libraries(extra_directory)

    assert directories == [extra_directory.resolve()]
    entries = os.environ[cuda._LIBRARY_PATH_VARIABLE].split(os.pathsep)
    assert entries[0] == str(extra_directory.resolve())


def test_configure_cuda_libraries_rejects_missing_extra_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cuda, "_WSL_DRIVER_DIRECTORY", tmp_path / "sin-wsl")
    monkeypatch.setattr(cuda, "discover_nvidia_libraries", lambda: [])

    with pytest.raises(FileNotFoundError, match="No existe el directorio"):
        cuda.configure_cuda_libraries(tmp_path / "no-existe")


def test_configure_cuda_libraries_without_libraries_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cuda, "_WSL_DRIVER_DIRECTORY", tmp_path / "sin-wsl")
    monkeypatch.setattr(cuda, "discover_nvidia_libraries", lambda: [])
    monkeypatch.setenv(cuda._LIBRARY_PATH_VARIABLE, "")

    assert cuda.configure_cuda_libraries() == []
    assert os.environ[cuda._LIBRARY_PATH_VARIABLE] == ""
