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


def align_by_time_window(
    reference_words: list[dict],
    other_words: list[list[dict]],
    tolerance: float,
) -> list[list[dict | None]]:
    """For each reference word, find the closest-start word in each other
    transcription within ±tolerance seconds.

    Returns a list of length len(reference_words). Each element is a list
    of length len(other_words): the matched WordData dict, or None.
    """
    result: list[list[dict | None]] = []
    for ref_word in reference_words:
        ref_start = parse_timestamp(ref_word["start"])
        matches: list[dict | None] = []
        for words in other_words:
            best: dict | None = None
            best_diff = float("inf")
            for w in words:
                diff = abs(parse_timestamp(w["start"]) - ref_start)
                if diff <= tolerance and diff < best_diff:
                    best = w
                    best_diff = diff
            matches.append(best)
        result.append(matches)
    return result
