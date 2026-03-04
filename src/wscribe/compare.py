"""
compare.py — inter-transcription agreement scoring for wscribe JSON output.
"""
from __future__ import annotations

import re
from datetime import datetime

import structlog

LOGGER = structlog.get_logger()


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


def compare_transcriptions(
    transcriptions: list[list[dict]],
    tolerance: float = 0.2,
) -> list[dict]:
    """Compare N transcriptions and return a scored result in wscribe JSON format.

    Args:
        transcriptions: List of parsed wscribe JSON documents (each a list of
            TranscribedData dicts). The first entry is used as the reference;
            its segment boundaries and timestamps are preserved in the output.
        tolerance: Time-window tolerance in seconds for word alignment.

    Returns:
        list[TranscribedData] with scores replaced by inter-transcription
        agreement scores. Ready to pass to WriteJSON.
    """
    reference = transcriptions[0]
    others = transcriptions[1:]

    # Warn if any transcription lacks word-level data
    for i, t in enumerate(transcriptions):
        if any(not seg.get("words") for seg in t):
            LOGGER.warning(
                "transcription missing word-level data; falling back to segment alignment",
                transcription_index=i,
            )

    # Flatten each transcription to a word list for alignment
    def flatten_words(segs: list[dict]) -> list[dict]:
        out = []
        for seg in segs:
            for w in seg.get("words") or []:
                out.append({
                    **w,
                    "start": parse_timestamp(w["start"]),
                    "end": parse_timestamp(w["end"]),
                })
        return out

    ref_words = flatten_words(reference)
    other_word_lists = [flatten_words(t) for t in others]

    # Align and score every reference word
    alignments = align_by_time_window(ref_words, other_word_lists, tolerance)
    scored_words: list[tuple[float, str]] = [
        score_word_group(ref_words[i], alignments[i])
        for i in range(len(ref_words))
    ]

    # Map scored words back onto reference segments
    word_cursor = 0
    result: list[dict] = []

    for seg in reference:
        seg_words = seg.get("words") or []
        out_words: list[dict] = []

        for w in seg_words:
            score, text = scored_words[word_cursor]
            word_cursor += 1
            out_words.append({
                "text": text,
                "start": parse_timestamp(w["start"]),
                "end": parse_timestamp(w["end"]),
                "score": score,
            })

        seg_score = (
            round(sum(w["score"] for w in out_words) / len(out_words), 4)
            if out_words
            else 0.0
        )

        result.append({
            "text": " ".join(w["text"] for w in out_words).strip() or seg["text"].strip(),
            "start": parse_timestamp(seg["start"]),
            "end": parse_timestamp(seg["end"]),
            "score": seg_score,
            "words": out_words,
        })

    return result
