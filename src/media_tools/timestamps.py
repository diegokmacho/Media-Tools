from __future__ import annotations

_MS_PER_SECOND = 1_000
_MS_PER_MINUTE = 60 * _MS_PER_SECOND
_MS_PER_HOUR = 60 * _MS_PER_MINUTE


def _split_milliseconds(seconds: float) -> tuple[int, int, int, int]:
    total_milliseconds = round(seconds * _MS_PER_SECOND)
    if total_milliseconds < 0:
        raise ValueError("El tiempo no puede ser negativo.")

    hours, remainder = divmod(total_milliseconds, _MS_PER_HOUR)
    minutes, remainder = divmod(remainder, _MS_PER_MINUTE)
    secs, milliseconds = divmod(remainder, _MS_PER_SECOND)
    return hours, minutes, secs, milliseconds


def seconds_to_srt(seconds: float) -> str:
    """Devuelve un timestamp con formato SRT: HH:MM:SS,mmm."""
    hours, minutes, secs, milliseconds = _split_milliseconds(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def seconds_to_hhmmss(seconds: float) -> str:
    """Devuelve un timestamp legible: HH:MM:SS."""
    hours, minutes, secs, _ = _split_milliseconds(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
