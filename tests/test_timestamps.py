from __future__ import annotations

import pytest

from media_tools.timestamps import seconds_to_hhmmss, seconds_to_srt


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (59.999, "00:00:59,999"),
        (60.0, "00:01:00,000"),
        (3599.999, "00:59:59,999"),
        (3600.001, "01:00:00,001"),
    ],
)
def test_seconds_to_srt(seconds: float, expected: str) -> None:
    assert seconds_to_srt(seconds) == expected


def test_seconds_to_srt_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negativo"):
        seconds_to_srt(-0.1)


def test_seconds_to_hhmmss() -> None:
    assert seconds_to_hhmmss(0) == "00:00:00"
    assert seconds_to_hhmmss(3661.9) == "01:01:01"
