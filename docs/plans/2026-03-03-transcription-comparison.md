# Transcription Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `wscribe compare` — a CLI command that aligns multiple wscribe JSON transcriptions of the same audio by timestamp and replaces each word/segment `score` with an inter-transcription agreement score, surfacing likely errors for human review in wscribe-editor.

**Architecture:** Pure post-processing module (`src/wscribe/compare.py`) with no backend dependency, a thin Click command (`src/wscribe/cli/compare.py`), and registration in the existing `cli/main.py`. Alignment uses a ±time-tolerance window on word start timestamps; scoring is majority-vote proportion.

**Tech Stack:** Python 3.11+, Click (already used), standard library only (`json`, `datetime`, `re`). No new dependencies.

**Design doc:** `docs/plans/2026-03-03-transcription-comparison-design.md`

---

### Task 1: `parse_timestamp` and `normalize_text` helpers

The comparison logic needs to convert `"HH:MM:SS.mmm"` strings to float seconds, and to
normalize word text for comparison (lowercase, strip punctuation). These are self-contained
and easy to test first.

**Files:**
- Create: `src/wscribe/compare.py`
- Test: `tests/test_compare.py`

**Step 1: Write the failing tests**

Create `tests/test_compare.py`:

```python
import pytest
from wscribe.compare import parse_timestamp, normalize_text


class TestParseTimestamp:
    def test_zero(self):
        assert parse_timestamp("00:00:00.000") == 0.0

    def test_milliseconds(self):
        assert parse_timestamp("00:00:07.260") == pytest.approx(7.260, abs=1e-6)

    def test_minutes(self):
        assert parse_timestamp("00:01:00.000") == pytest.approx(60.0, abs=1e-6)

    def test_hours(self):
        assert parse_timestamp("01:00:00.000") == pytest.approx(3600.0, abs=1e-6)

    def test_full(self):
        assert parse_timestamp("01:23:45.678") == pytest.approx(
            3600 + 23 * 60 + 45 + 0.678, abs=1e-6
        )

    def test_float_passthrough(self):
        assert parse_timestamp(7.26) == pytest.approx(7.26, abs=1e-6)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello") == "hello"

    def test_strips_punctuation(self):
        assert normalize_text("hello,") == "hello"
        assert normalize_text("it's") == "its"

    def test_strips_surrounding_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_empty(self):
        assert normalize_text("") == ""
```

**Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_compare.py -v
```

Expected: `ModuleNotFoundError: No module named 'wscribe.compare'`

**Step 3: Create `src/wscribe/compare.py` with helpers**

```python
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
```

**Step 4: Run tests to confirm they pass**

```
uv run pytest tests/test_compare.py::TestParseTimestamp tests/test_compare.py::TestNormalizeText -v
```

Expected: all 10 tests PASS

**Step 5: Commit**

```bash
git add src/wscribe/compare.py tests/test_compare.py
git commit -m "feat: add compare.py with parse_timestamp and normalize_text helpers"
```

---

### Task 2: `align_by_time_window`

Align a flat list of reference words against per-transcription word lists using a ±tolerance
window. Returns, for each reference word, a list of matched words (one per transcription,
`None` if no match).

**Files:**
- Modify: `src/wscribe/compare.py`
- Test: `tests/test_compare.py`

The alignment function signature:

```python
def align_by_time_window(
    reference_words: list[dict],       # list of WordData with float start/end
    other_words: list[list[dict]],     # one list per other transcription
    tolerance: float,
) -> list[list[dict | None]]:
    ...
```

Returns: for each reference word, a list of length `len(other_words)` with the matched word
dict or `None`.

**Step 1: Write failing tests**

Add to `tests/test_compare.py`:

```python
from wscribe.compare import align_by_time_window


def _word(start: float, text: str) -> dict:
    return {"start": start, "end": start + 0.5, "text": text, "score": 1.0}


class TestAlignByTimeWindow:
    def test_exact_match(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.0, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert len(result) == 1
        assert result[0][0]["text"] == "hello"

    def test_within_tolerance(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.15, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is not None

    def test_outside_tolerance(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.21, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is None

    def test_exactly_at_tolerance_boundary(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.2, "hello")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is not None

    def test_no_match_in_other(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(5.0, "world")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is None

    def test_multiple_transcriptions(self):
        ref = [_word(1.0, "hello")]
        other = [[_word(1.0, "hello")], [_word(9.0, "world")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0] is not None
        assert result[0][1] is None

    def test_multiple_reference_words(self):
        ref = [_word(1.0, "hello"), _word(2.0, "world")]
        other = [[_word(1.0, "hello"), _word(2.0, "world")]]
        result = align_by_time_window(ref, other, tolerance=0.2)
        assert result[0][0]["text"] == "hello"
        assert result[1][0]["text"] == "world"
```

**Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_compare.py::TestAlignByTimeWindow -v
```

Expected: `ImportError` or `AttributeError`

**Step 3: Implement `align_by_time_window`**

Add to `src/wscribe/compare.py`:

```python
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
```

**Step 4: Run tests to confirm they pass**

```
uv run pytest tests/test_compare.py::TestAlignByTimeWindow -v
```

Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add src/wscribe/compare.py tests/test_compare.py
git commit -m "feat: add align_by_time_window to compare.py"
```

---

### Task 3: `score_word_group`

Given a reference word and its matched words from other transcriptions, compute the
agreement score and majority-vote output text.

**Files:**
- Modify: `src/wscribe/compare.py`
- Test: `tests/test_compare.py`

**Step 1: Write failing tests**

Add to `tests/test_compare.py`:

```python
from wscribe.compare import score_word_group


class TestScoreWordGroup:
    def test_all_agree(self):
        ref = _word(1.0, "hello")
        others = [_word(1.0, "hello"), _word(1.0, "hello")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(1.0)
        assert text == "hello"

    def test_all_disagree(self):
        ref = _word(1.0, "hello")
        others = [_word(1.0, "world"), _word(1.0, "foo")]
        score, text = score_word_group(ref, others)
        # ref + 0 matching others = 1 of 3
        assert score == pytest.approx(1 / 3, abs=0.01)

    def test_majority_wins(self):
        ref = _word(1.0, "hello")
        others = [_word(1.0, "hello"), _word(1.0, "world")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(2 / 3, abs=0.01)
        assert text == "hello"

    def test_none_match_scores_zero(self):
        ref = _word(1.0, "hello")
        others = [None, None]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(0.0)
        assert text == "hello"  # falls back to reference

    def test_punctuation_ignored_in_comparison(self):
        ref = _word(1.0, "hello,")
        others = [_word(1.0, "hello")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(1.0)

    def test_case_ignored_in_comparison(self):
        ref = _word(1.0, "Hello")
        others = [_word(1.0, "hello")]
        score, text = score_word_group(ref, others)
        assert score == pytest.approx(1.0)
```

**Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_compare.py::TestScoreWordGroup -v
```

Expected: `ImportError`

**Step 3: Implement `score_word_group`**

Add to `src/wscribe/compare.py`:

```python
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
```

**Step 4: Run tests to confirm they pass**

```
uv run pytest tests/test_compare.py::TestScoreWordGroup -v
```

Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add src/wscribe/compare.py tests/test_compare.py
git commit -m "feat: add score_word_group to compare.py"
```

---

### Task 4: `compare_transcriptions` — top-level function

Wire the helpers together: load segments → flatten words → align → score → rebuild segments.

**Files:**
- Modify: `src/wscribe/compare.py`
- Test: `tests/test_compare.py`

**Step 1: Write failing tests**

Add to `tests/test_compare.py`:

```python
from wscribe.compare import compare_transcriptions


def _seg(start: float, end: float, text: str, words: list[dict]) -> dict:
    return {"start": start, "end": end, "text": text, "score": 1.0, "words": words}


class TestCompareTranscriptions:
    def _two_identical(self):
        w = [_word(1.0, "hello"), _word(2.0, "world")]
        seg = [_seg(0.0, 3.0, "hello world", w)]
        return [seg, seg]

    def test_identical_inputs_score_one(self):
        result = compare_transcriptions(self._two_identical(), tolerance=0.2)
        for seg in result:
            assert seg["score"] == pytest.approx(1.0)
            for w in seg["words"]:
                assert w["score"] == pytest.approx(1.0)

    def test_output_schema(self):
        result = compare_transcriptions(self._two_identical(), tolerance=0.2)
        assert isinstance(result, list)
        seg = result[0]
        assert set(seg.keys()) == {"text", "start", "end", "score", "words"}
        word = seg["words"][0]
        assert set(word.keys()) == {"text", "start", "end", "score"}

    def test_disjoint_words_score_zero(self):
        w_a = [_word(1.0, "hello")]
        w_b = [_word(9.0, "world")]
        seg_a = [_seg(0.0, 2.0, "hello", w_a)]
        seg_b = [_seg(8.0, 10.0, "world", w_b)]
        result = compare_transcriptions([seg_a, seg_b], tolerance=0.2)
        for seg in result:
            for w in seg["words"]:
                assert w["score"] == pytest.approx(0.0)

    def test_segment_score_is_mean_of_word_scores(self):
        w = [_word(1.0, "hello"), _word(2.0, "world")]
        # second transcription agrees on first word, differs on second
        w2 = [_word(1.0, "hello"), _word(2.0, "other")]
        seg = [_seg(0.0, 3.0, "hello world", w)]
        seg2 = [_seg(0.0, 3.0, "hello other", w2)]
        result = compare_transcriptions([seg, seg2], tolerance=0.2)
        word_scores = [w["score"] for w in result[0]["words"]]
        expected_seg_score = round(sum(word_scores) / len(word_scores), 4)
        assert result[0]["score"] == pytest.approx(expected_seg_score, abs=0.01)
```

**Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_compare.py::TestCompareTranscriptions -v
```

Expected: `ImportError`

**Step 3: Implement `compare_transcriptions`**

Add to `src/wscribe/compare.py`:

```python
import structlog

LOGGER = structlog.get_logger()


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
                out.append({**w, "start": parse_timestamp(w["start"]), "end": parse_timestamp(w["end"])})
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
                "start": w["start"],
                "end": w["end"],
                "score": score,
            })

        seg_score = (
            round(sum(w["score"] for w in out_words) / len(out_words), 4)
            if out_words
            else 0.0
        )

        result.append({
            "text": " ".join(w["text"] for w in out_words).strip() or seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
            "score": seg_score,
            "words": out_words,
        })

    return result
```

**Step 4: Run tests to confirm they pass**

```
uv run pytest tests/test_compare.py::TestCompareTranscriptions -v
```

Expected: all 4 tests PASS

**Step 5: Run all compare tests**

```
uv run pytest tests/test_compare.py -v
```

Expected: all tests PASS

**Step 6: Commit**

```bash
git add src/wscribe/compare.py tests/test_compare.py
git commit -m "feat: add compare_transcriptions orchestration function"
```

---

### Task 5: CLI command `wscribe compare`

Add the Click command and register it.

**Files:**
- Create: `src/wscribe/cli/compare.py`
- Modify: `src/wscribe/cli/main.py`
- Test: `tests/test_compare.py` (CLI smoke test)

**Step 1: Write failing CLI smoke test**

Add to `tests/test_compare.py`:

```python
import json
import os
import tempfile
from click.testing import CliRunner
from wscribe.cli.main import cli


SAMPLE_JSON = os.path.join(
    os.environ.get("PROJECT_ROOT", ""),
    "examples", "output", "sample.json",
)


class TestCompareCLI:
    def test_requires_two_files(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", SAMPLE_JSON, "--output", "/tmp/x.json"])
        assert result.exit_code != 0

    def test_smoke(self):
        if not os.path.exists(SAMPLE_JSON):
            pytest.skip("sample.json not available")
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = runner.invoke(
                cli,
                ["compare", SAMPLE_JSON, SAMPLE_JSON, "--output", out_path],
            )
            assert result.exit_code == 0, result.output
            data = json.loads(open(out_path).read())
            assert isinstance(data, list)
            assert len(data) > 0
            seg = data[0]
            assert {"text", "start", "end", "score", "words"} <= set(seg.keys())
            # timestamp strings
            assert isinstance(seg["start"], str)
            assert ":" in seg["start"]
        finally:
            os.unlink(out_path)
```

**Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_compare.py::TestCompareCLI -v
```

Expected: `ModuleNotFoundError` or `CommandError`

**Step 3: Create `src/wscribe/cli/compare.py`**

```python
import json
from pathlib import Path
from typing import Any

import click
import structlog

from ..compare import compare_transcriptions
from ..writers import WriteJSON

LOGGER = structlog.get_logger(ui="cli")


@click.command()
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(exists=False, resolve_path=True),
    help="Path to write the comparison JSON",
)
@click.option(
    "--time-tolerance",
    default=0.2,
    show_default=True,
    type=float,
    help="Seconds within which two words are considered to refer to the same audio moment",
)
def compare(files: tuple[str, ...], output: str, time_tolerance: float) -> None:
    """Compare two or more wscribe JSON transcriptions of the same audio.

    Produces a single JSON file where each word score reflects inter-transcription
    agreement. Low-scoring words are likely transcription errors.
    """
    if len(files) < 2:
        raise click.UsageError("compare requires at least 2 input files")

    transcriptions: list[Any] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise click.ClickException(f"{path}: expected a JSON array")
            transcriptions.append(data)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"{path}: invalid JSON — {e}") from e

    LOGGER.info("comparing transcriptions", n=len(transcriptions), tolerance=time_tolerance)
    result = compare_transcriptions(transcriptions, tolerance=time_tolerance)

    writer = WriteJSON(result=result, destination=Path(output))
    writer.write()
    LOGGER.info("written", output=output)
```

**Step 4: Register the command in `src/wscribe/cli/main.py`**

Add the import after the existing imports block (around line 14):

```python
from .compare import compare as compare_command
```

Add the command registration after the `cli` group definition (after `def cli(): ...`):

```python
cli.add_command(compare_command, name="compare")
```

**Step 5: Run tests to confirm they pass**

```
uv run pytest tests/test_compare.py::TestCompareCLI -v
```

Expected: all 2 tests PASS

**Step 6: Run all compare tests**

```
uv run pytest tests/test_compare.py -v
```

Expected: all tests PASS

**Step 7: Manual smoke check**

```bash
uv run wscribe compare \
  examples/output/sample.json \
  examples/output/sample.json \
  --output /tmp/compare_test.json
python3 -c "import json; d=json.load(open('/tmp/compare_test.json')); print(d[0]['score'], d[0]['words'][0])"
```

Expected: score `1.0` (identical inputs), first word printed.

**Step 8: Commit**

```bash
git add src/wscribe/cli/compare.py src/wscribe/cli/main.py tests/test_compare.py
git commit -m "feat: add wscribe compare CLI command"
```

---

### Task 6: Full test pass and tidy

**Step 1: Run the full test suite**

```
uv run pytest -v
```

Expected: all tests PASS (existing `TestFastWhisper` plus all new compare tests)

**Step 2: Type-check**

```
uv run mypy src/wscribe/compare.py src/wscribe/cli/compare.py
```

Fix any errors before continuing.

**Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: mypy type errors in compare module"
```

(Skip this step if there were no errors.)

---

### Task 7: Update `research.md`

Add a §14 (or append to §13) documenting the new `compare` module as a worked example of a
post-processing extension.

**Files:**
- Modify: `src/wscribe/research.md` — wait, this is at the workspace root:
- Modify: `research.md`

**Step 1: Add section to `research.md`**

Find the end of the file and append:

```markdown
## 14. Transcription Comparison (`wscribe compare`)

`compare.py` is a pure post-processing module — no backend dependency — that aligns two or
more wscribe JSON files by timestamp and replaces every word/segment `score` with an
**inter-transcription agreement score**.

### Key functions

| Function | Purpose |
|---|---|
| `parse_timestamp(s)` | `"HH:MM:SS.mmm"` → float seconds; floats pass through unchanged |
| `normalize_text(text)` | lowercase + strip punctuation for comparison |
| `align_by_time_window(ref, others, tolerance)` | match words across transcriptions by start time |
| `score_word_group(ref_word, matches)` | majority-vote score + output text selection |
| `compare_transcriptions(transcriptions, tolerance)` | orchestrate the above; return `list[TranscribedData]` |

### Agreement score

For a word present in N transcriptions, the score is `k / N` where k is the number of
transcriptions that agree on the normalised text. A score of `0.0` means the word appears in
only one transcription (highest priority for human review). Scores map directly onto
wscribe-editor's existing `wordColor` thresholds with no editor changes.

### CLI

```
wscribe compare file1.json file2.json [file3.json ...] --output out.json [--time-tolerance 0.2]
```
```

**Step 2: Commit**

```bash
git add research.md
git commit -m "docs: document compare module in research.md"
```
