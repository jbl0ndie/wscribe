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


def score_word_group(
    reference_word: dict,
    other_matches: list[dict | None],
) -> tuple[float, str]:
    """Compute inter-transcription agreement score and majority-vote text.

    Args:
        reference_word: The WordData from the reference (first) transcription.
        other_matches: One entry per other transcription; None if no match found.

    Returns:
        (score, text) where score is in [0.0, 1.0] and text is the majority text.
    """
    # Special case: word absent from all other transcriptions (likely spurious)
    if all(m is None for m in other_matches):
        return 0.0, reference_word["text"].strip()

    n_transcriptions = 1 + len(other_matches)  # reference + others
    ref_norm = normalize_text(reference_word["text"])

    # Count votes per normalised text form
    votes: dict[str, int] = {}
    votes[ref_norm] = 1
    for match in other_matches:
        if match is not None:
            norm = normalize_text(match["text"])
            votes[norm] = votes.get(norm, 0) + 1

    majority_norm, majority_count = max(votes.items(), key=lambda kv: kv[1])
    score = round(majority_count / n_transcriptions, 4)

    # Recover original-cased text for the majority form
    # Prefer the reference word's text if it matches the majority
    if normalize_text(reference_word["text"]) == majority_norm:
        text = reference_word["text"].strip()
    else:
        # Find the first other match that produced the majority text
        text = reference_word["text"].strip()  # fallback
        for match in other_matches:
            if match is not None and normalize_text(match["text"]) == majority_norm:
                text = match["text"].strip()
                break

    return score, text
