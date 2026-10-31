from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path

_LIBRARY_PATH_VARIABLE = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
_WSL_DRIVER_DIRECTORY = Path("/usr/lib/wsl/lib")
_SHARED_LIBRARY_HANDLES: list[object] = []
_DLL_DIRECTORY_HANDLES: list[object] = []


def discover_nvidia_libraries() -> list[Path]:
    """Devuelve los directorios `lib` de los wheels `nvidia-*-cu12` instalados."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []

    libraries: list[Path] = []
    for location in spec.submodule_search_locations:
        libraries.extend(sorted(Path(location).glob("*/lib")))
    return [library for library in libraries if library.is_dir()]


def configure_cuda_libraries(extra_directory: Path | str | None = None) -> list[Path]:
    """Prepara cuBLAS/cuDNN para que ctranslate2 pueda usar CUDA.

    Añade los directorios encontrados a LD_LIBRARY_PATH (Linux) o PATH (Windows)
    y precarga las librerías compartidas para que el cargador dinámico las
    encuentre aunque el entorno se modifique después de iniciar Python.
    """
    directories = discover_nvidia_libraries()

    if os.name != "nt" and _WSL_DRIVER_DIRECTORY.is_dir():
        directories.insert(0, _WSL_DRIVER_DIRECTORY)

    if extra_directory is not None:
        custom = Path(extra_directory).expanduser().resolve()
        if not custom.is_dir():
            raise FileNotFoundError(f"No existe el directorio de librerías CUDA: {custom}")
        directories.append(custom)

    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)

    if not unique:
        return []

    entries = [
        entry for entry in os.environ.get(_LIBRARY_PATH_VARIABLE, "").split(os.pathsep) if entry
    ]
    for directory in reversed(unique):
        value = str(directory)
        if value in entries:
            entries.remove(value)
        entries.insert(0, value)
    os.environ[_LIBRARY_PATH_VARIABLE] = os.pathsep.join(entries)

    if os.name == "nt":
        _register_windows_dll_directories(unique)
    else:
        _preload_shared_libraries(unique)

    return unique


def _register_windows_dll_directories(directories: list[Path]) -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    for directory in directories:
        try:
            _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(directory)))
        except OSError:
            continue


def _preload_candidates(directories: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for directory in directories:
        if directory == _WSL_DRIVER_DIRECTORY:
            candidates.extend(sorted(directory.glob("libcuda.so*")))
        else:
            candidates.extend(sorted(directory.glob("*.so*")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _preload_shared_libraries(directories: list[Path]) -> None:
    pending = _preload_candidates(directories)
    while pending:
        remaining: list[Path] = []
        loaded_any = False
        for library in pending:
            try:
                handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                remaining.append(library)
            else:
                _SHARED_LIBRARY_HANDLES.append(handle)
                loaded_any = True
        if not loaded_any:
            break
        pending = remaining
