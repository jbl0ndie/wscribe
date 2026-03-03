"""
compare.py — inter-transcription agreement scoring for wscribe JSON output.
"""
from __future__ import annotations

import re
from datetime import datetime


def parse_timestamp(s: float | str) -> float:
    """Convert a wscribe timestamp string "HH:MM:SS.mmm" to seconds.

    Floats are returned unchanged (backends store timestamps as floats
    before serialisation).
    """
    if isinstance(s, (int, float)):
        return float(s)
    t = datetime.strptime(s, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for text comparison."""
    return re.sub(r"[^\w\s]", "", text).strip().lower()
