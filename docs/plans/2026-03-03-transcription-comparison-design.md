# Design: Transcription Comparison (`wscribe compare`)

**Date:** 2026-03-03  
**Status:** Approved

---

## Overview

Add a `wscribe compare` command that accepts two or more wscribe JSON output files (all
transcriptions of the same audio), compares them word-by-word using time-window alignment,
and produces a single output JSON file in the standard wscribe format where each word and
segment `score` reflects **inter-transcription agreement** rather than model-reported
probability.

The output feeds directly into wscribe-editor without any editor changes — low-agreement
regions surface as orange/red highlights via the existing `wordColor` thresholds, directing
humans to the parts most likely to be wrong.

---

## CLI

```
wscribe compare file1.json file2.json [file3.json ...]  --output comparison.json  [--time-tolerance 0.2]
```

| Option | Default | Description |
|---|---|---|
| `FILES` (positional, ≥2) | — | wscribe JSON files to compare |
| `--output / -o` | required | Path to write the comparison JSON |
| `--time-tolerance` | `0.2` | Seconds within which two words are considered to refer to the same audio moment |

`compare` is registered as a peer of `transcribe` in `cli/main.py`.

### Optional convenience flag on `transcribe`

A future follow-on (not part of this implementation) could add:

```
wscribe transcribe audio.mp3 --runs 3 --output-dir ./runs/
```

…producing `runs/audio_1.json`, `runs/audio_2.json`, `runs/audio_3.json` for immediate use
with `compare`. This is deferred — `compare` is useful without it.

---

## Algorithm

### 1. Load and parse

Load each file into `list[TranscribedData]`. All timestamp strings are parsed to floats
(seconds) using a helper in `compare.py`:

```python
from datetime import datetime

def parse_timestamp(s: str) -> float:
    t = datetime.strptime(s, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000
```

`%f` handles 3-digit milliseconds by padding to 6-digit microseconds internally, preserving
full millisecond precision. On output, floats are serialised back to strings using the
existing `format_timestamp()` from `writers.py`.

### 2. Flatten words

Each transcription is flattened to a list of `(start_s, end_s, text, source_index)` tuples,
one per word. Segment membership is noted so segments can be rebuilt later.

### 3. Align by time window

For each word in the **reference transcription** (first file), collect all candidate words
from every other transcription whose `start` falls within `±time_tolerance` seconds.

```
reference word start: T
candidates: words in other transcriptions where |start − T| ≤ time_tolerance
```

This gives a **word group** for each reference word: one entry per transcription (or `None`
if a transcription had nothing in that window).

### 4. Score word groups

For each word group:

- **Agreement score** = proportion of transcriptions that produced the same normalised text
  (lowercase, stripped punctuation).
  - All N agree → `1.0`
  - k of N agree (majority) → `k / N`
  - No majority, or word absent from all others → `0.0`

- **Output text** = majority-vote text. Tie broken by the reference transcription's text.

### 5. Rebuild segments

Words are re-grouped into segments using the reference transcription's segment boundaries.
Each segment's `score` is the mean of its word scores (same as wscribe-editor's existing
aggregation logic). Timestamps are taken from the reference transcription.

### 6. Write output

Result is written via the existing `WriteJSON` writer, so timestamp strings, field names,
and structure are identical to normal wscribe output.

---

## File Layout

```
src/wscribe/
    compare.py          # pure comparison logic; no CLI dependency
    cli/
        compare.py      # Click command; imports from compare.py
        main.py         # registers `compare` command (existing file, small edit)
```

`compare.py` works entirely in terms of existing `TranscribedData` and `WordData` TypedDicts.
No new dataclasses are needed.

---

## Error Handling and Edge Cases

| Case | Behaviour |
|---|---|
| Fewer than 2 input files | Fail fast with clear Click error |
| File is not valid wscribe JSON | Fail fast with filename and parse error |
| Word present in only one transcription | Score → `0.0`; text taken from that transcription |
| Transcription without word-level data (`words` absent or empty) | Fall back to segment-level alignment; emit a warning; word scores unavailable |
| Timestamps stored as floats (not yet serialised) | `parse_timestamp` handles `str`; floats passed through unchanged |
| Unequal numbers of segments/words across files | Normal — handled naturally by time-window alignment |

---

## Testing

New tests alongside the existing suite in `tests/test_wscribe.py`:

| Test | Assertion |
|---|---|
| Identical inputs (N copies) | All word scores == `1.0` |
| Fully disjoint words (no time overlap) | All word scores == `0.0` |
| 2-of-3 agreement | Word scores == `0.67` (rounded to 2 dp) |
| Majority text selection | Output text matches the 2-of-3 majority, not the outlier |
| Tolerance boundary | Word at exactly `±tolerance` is included; word at `tolerance + ε` is excluded |
| CLI smoke test | `wscribe compare a.json b.json --output out.json` exits 0 and produces structurally valid JSON |
| Round-trip schema | Output has `text`, `start`, `end`, `score`, `words` at every level; timestamps are `"HH:MM:SS.mmm"` strings |
| Missing `words` field | Emits a warning; segment-level comparison still produces output |

No new test dependencies required.

---

## Design Decisions and Alternatives Considered

**Why a separate `compare` command rather than a flag on `transcribe`?**  
Separation of concerns: `compare` is pure post-processing with no backend dependency. It can
be used with output produced by any combination of backends, models, or settings, including
files generated days apart.

**Why time-window alignment rather than sequence/edit-distance alignment?**  
Timestamps are the most reliable cross-transcription anchor — two words about the same audio
moment have similar start times regardless of what the model heard. Edit-distance alignment
on the full text is brittle when transcriptions diverge significantly in word count.

**Why reuse the existing `score` field rather than adding a new field?**  
wscribe-editor already visualises `score` with colour-coded confidence bands. Reusing it
means zero editor changes and immediate compatibility.
